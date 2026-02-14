using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace YoloDetect.Nvidia
{
    /// <summary>
    /// Conversión de Mat a Tensor para una sola imagen.
    /// Contiene múltiples implementaciones con diferentes niveles de optimización.
    /// </summary>
    public class TensorConverterSingle
    {
        private const float InverseNormalization = 1.0f / 255.0f;
        private static Vector256<float> normFactor = Vector256.Create(InverseNormalization);
        /// <summary>
        /// Versión híbrida sin Parallel - solo SIMD + Unsafe.
        /// </summary>
        public static unsafe void MatToTensorHybridNoParallel(Mat mat, DenseTensor<float> tensor)
        {
            int height = mat.Rows;
            int width = mat.Cols;
            int planeSize = height * width;

            if (mat.Type() != MatType.CV_8UC3)
            {
                throw new ArgumentException($"Tipo de Mat no soportado: {mat.Type()}");
            }

            Span<float> tensorData = tensor.Buffer.Span;
            byte* srcPtr = (byte*)mat.Data.ToPointer();
            int stride = (int)mat.Step();

            fixed (float* dstPtr = tensorData)
            {
                float* rPlane = dstPtr;
                float* gPlane = dstPtr + planeSize;
                float* bPlane = dstPtr + 2 * planeSize;

                if (Avx2.IsSupported && width >= 8)
                {
                    int simdWidth = width - (width % 8);

                    for (int h = 0; h < height; h++)
                    {
                        byte* rowPtr = srcPtr + h * stride;
                        int rowOffset = h * width;
                        int w = 0;

                        // Procesar 8 píxeles a la vez con AVX2
                        for (; w < simdWidth; w += 8)
                        {
                            int pixelBase = w * 3;
                            int dstBase = rowOffset + w;

                            Vector256<int> rInt = Vector256.Create(
                                rowPtr[pixelBase], rowPtr[pixelBase + 3], rowPtr[pixelBase + 6], rowPtr[pixelBase + 9],
                                rowPtr[pixelBase + 12], rowPtr[pixelBase + 15], rowPtr[pixelBase + 18], rowPtr[pixelBase + 21]);
                            Vector256<float> rFloat = Avx.Multiply(Avx.ConvertToVector256Single(rInt), normFactor);

                            Vector256<int> gInt = Vector256.Create(
                                rowPtr[pixelBase + 1], rowPtr[pixelBase + 4], rowPtr[pixelBase + 7], rowPtr[pixelBase + 10],
                                rowPtr[pixelBase + 13], rowPtr[pixelBase + 16], rowPtr[pixelBase + 19], rowPtr[pixelBase + 22]);
                            Vector256<float> gFloat = Avx.Multiply(Avx.ConvertToVector256Single(gInt), normFactor);

                            Vector256<int> bInt = Vector256.Create(
                                rowPtr[pixelBase + 2], rowPtr[pixelBase + 5], rowPtr[pixelBase + 8], rowPtr[pixelBase + 11],
                                rowPtr[pixelBase + 14], rowPtr[pixelBase + 17], rowPtr[pixelBase + 20], rowPtr[pixelBase + 23]);
                            Vector256<float> bFloat = Avx.Multiply(Avx.ConvertToVector256Single(bInt), normFactor);

                            Avx.Store(rPlane + dstBase, rFloat);
                            Avx.Store(gPlane + dstBase, gFloat);
                            Avx.Store(bPlane + dstBase, bFloat);
                        }

                        // Píxeles restantes
                        for (; w < width; w++)
                        {
                            int pixelIdx = w * 3;
                            int dstIdx = rowOffset + w;
                            rPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                            gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                            bPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
                        }
                    }
                }
                else
                {
                    // Fallback escalar
                    for (int h = 0; h < height; h++)
                    {
                        byte* rowPtr = srcPtr + h * stride;
                        int rowOffset = h * width;

                        for (int w = 0; w < width; w++)
                        {
                            int pixelIdx = w * 3;
                            int dstIdx = rowOffset + w;
                            rPlane[dstIdx] = rowPtr[pixelIdx] * InverseNormalization;
                            gPlane[dstIdx] = rowPtr[pixelIdx + 1] * InverseNormalization;
                            bPlane[dstIdx] = rowPtr[pixelIdx + 2] * InverseNormalization;
                        }
                    }
                }
            }
        }
    }
}
