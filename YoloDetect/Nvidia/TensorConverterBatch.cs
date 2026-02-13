using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace YoloDetect.Nvidia
{
    /// <summary>
    /// Conversión de Mat a Tensor para batch de 2 imágenes.
    /// Contiene múltiples implementaciones con diferentes niveles de optimización.
    /// </summary>
    public class TensorConverterBatch
    {
        private const float InverseNormalization = 1.0f / 255.0f;
        private static Vector256<float> normFactor = Vector256.Create(InverseNormalization);
        /// <summary>
        /// Versión híbrida batch: SIMD + Unsafe + Parallel para procesar 2 imágenes.
        /// Combina las optimizaciones de MatToTensorHybrid con procesamiento batch.
        /// con tensor reutilizable - evita allocations.
        /// </summary>
        public static unsafe void MatToTensorHybridBatch(Mat mat1, Mat mat2, DenseTensor<float> tensor)
        {
            if (mat1.Rows != mat2.Rows || mat1.Cols != mat2.Cols)
            {
                throw new ArgumentException("Ambas imágenes deben tener las mismas dimensiones.");
            }

            int height = mat1.Rows;
            int width = mat1.Cols;
            const int channels = 3;
            int planeSize = height * width;
            int singleImageSize = channels * planeSize;

            if (mat1.Type() != MatType.CV_8UC3 || mat2.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException("Tipo de Mat no soportado. Se requiere CV_8UC3.");
            }

            byte* srcPtr1 = (byte*)mat1.Data.ToPointer();
            byte* srcPtr2 = (byte*)mat2.Data.ToPointer();
            int stride1 = (int)mat1.Step();
            int stride2 = (int)mat2.Step();

            Memory<float> tensorMemory = tensor.Buffer;
            MemoryHandle memHandle = tensorMemory.Pin();

            try
            {
                float* dstPtr = (float*)memHandle.Pointer;

                // Procesar ambas imágenes en paralelo - BGR→RGB inline
                Parallel.Invoke(
                    () => ProcessImageBgrToRgbInternal(srcPtr1, stride1, dstPtr, 0, height, width, planeSize),
                    () => ProcessImageBgrToRgbInternal(srcPtr2, stride2, dstPtr, singleImageSize, height, width, planeSize)
                );
            }
            finally
            {
                memHandle.Dispose();
            }
        }
        /// <summary>
        /// Procesa imagen BGR directamente a tensor RGB - sin Cv2.CvtColor.
        /// Lee B,G,R y escribe R,G,B en los planos correctos.
        /// </summary>
        private static unsafe void ProcessImageBgrToRgbInternal(byte* srcPtr, int stride, float* dstPtr, int offset, int height, int width, int planeSize)
        {
            float* rPlane = dstPtr + offset;
            float* gPlane = dstPtr + offset + planeSize;
            float* bPlane = dstPtr + offset + 2 * planeSize;

            if (Avx2.IsSupported && width >= 8)
            {
                Parallel.For(0, height, h =>
                {
                    byte* rowPtr = srcPtr + h * stride;
                    int rowOffset = h * width;
                    int w = 0;

                    int simdWidth = width - (width % 8);
                    for (; w < simdWidth; w += 8)
                    {
                        int pixelBase = w * 3;
                        int dstBase = rowOffset + w;

                        // BGR en memoria: [B0,G0,R0, B1,G1,R1, ...]
                        // Leemos B (índice 0,3,6...) y lo escribimos en bPlane
                        // Leemos G (índice 1,4,7...) y lo escribimos en gPlane  
                        // Leemos R (índice 2,5,8...) y lo escribimos en rPlane

                        // Cargar B (será escrito en bPlane)
                        Vector256<int> bInt = Vector256.Create(
                            rowPtr[pixelBase], rowPtr[pixelBase + 3], rowPtr[pixelBase + 6], rowPtr[pixelBase + 9],
                            rowPtr[pixelBase + 12], rowPtr[pixelBase + 15], rowPtr[pixelBase + 18], rowPtr[pixelBase + 21]);

                        // Cargar G (será escrito en gPlane)
                        Vector256<int> gInt = Vector256.Create(
                            rowPtr[pixelBase + 1], rowPtr[pixelBase + 4], rowPtr[pixelBase + 7], rowPtr[pixelBase + 10],
                            rowPtr[pixelBase + 13], rowPtr[pixelBase + 16], rowPtr[pixelBase + 19], rowPtr[pixelBase + 22]);

                        // Cargar R (será escrito en rPlane)
                        Vector256<int> rInt = Vector256.Create(
                            rowPtr[pixelBase + 2], rowPtr[pixelBase + 5], rowPtr[pixelBase + 8], rowPtr[pixelBase + 11],
                            rowPtr[pixelBase + 14], rowPtr[pixelBase + 17], rowPtr[pixelBase + 20], rowPtr[pixelBase + 23]);

                        // Almacenar en orden RGB
                        Avx.Store(rPlane + dstBase, Avx.Multiply(Avx.ConvertToVector256Single(rInt), normFactor));
                        Avx.Store(gPlane + dstBase, Avx.Multiply(Avx.ConvertToVector256Single(gInt), normFactor));
                        Avx.Store(bPlane + dstBase, Avx.Multiply(Avx.ConvertToVector256Single(bInt), normFactor));
                    }

                    // Procesar píxeles restantes - BGR→RGB inline
                    for (; w < width; w++)
                    {
                        int pixelIdx = w * 3;
                        int dstIdx = rowOffset + w;
                        // BGR: [0]=B, [1]=G, [2]=R
                        bPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                        gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                        rPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
                    }
                });
            }
            else
            {
                // Fallback sin SIMD
                Parallel.For(0, height, h =>
                {
                    byte* rowPtr = srcPtr + h * stride;
                    int rowOffset = h * width;

                    for (int w = 0; w < width; w++)
                    {
                        int pixelIdx = w * 3;
                        int dstIdx = rowOffset + w;
                        // BGR: [0]=B, [1]=G, [2]=R
                        bPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                        gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                        rPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
                    }
                });
            }
        }
    }
}
