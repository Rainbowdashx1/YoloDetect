using OpenCvSharp;

namespace YoloDetect.VideoCapture
{
    public class ProcessFrame
    {
        // Buffer reutilizable para evitar allocaciones repetidas
        private Mat? _resizedBuffer;
        private Size _lastResizedSize;
        private readonly Scalar _borderColor = new Scalar(114, 114, 114); // Reutilizable
        /// <summary>
        /// Letterbox optimizado usando CopyMakeBorder
        /// </summary>
        public void LetterboxOptimized(Mat src, Mat dst, int dstW, int dstH, out float r, out int padX, out int padY)
        {
            int srcW = src.Width;
            int srcH = src.Height;

            float rW = dstW / (float)srcW;
            float rH = dstH / (float)srcH;
            r = rW < rH ? rW : rH;

            int newW = (int)(srcW * r);
            int newH = (int)(srcH * r);

            int totalPadX = dstW - newW;
            int totalPadY = dstH - newH;
            padX = totalPadX >> 1;
            padY = totalPadY >> 1;
            int padRight = totalPadX - padX;
            int padBottom = totalPadY - padY;

            if (_resizedBuffer == null || _lastResizedSize.Width != newW || _lastResizedSize.Height != newH)
            {
                _resizedBuffer?.Dispose();
                _resizedBuffer = new Mat();
                _lastResizedSize = new Size(newW, newH); ;
            }

            Cv2.Resize(src, _resizedBuffer, _lastResizedSize, 0, 0, InterpolationFlags.Linear);
            Cv2.CopyMakeBorder(
                _resizedBuffer,
                dst,
                padY, padBottom,
                padX, padRight,
                BorderTypes.Constant,
                _borderColor
            );
        }
    }
}
