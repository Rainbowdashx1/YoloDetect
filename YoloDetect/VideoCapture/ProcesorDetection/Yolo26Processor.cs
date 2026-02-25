using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloDetect.PreProcess;

namespace YoloDetect.VideoCapture.ProcesorDetection
{
    public class Yolo26Processor : IDetectionProcessor
    {
        private readonly Preprocessed _preprocessed;
        private readonly float _threshold;

        public Yolo26Processor(float threshold = 0.25f)
        {
            _preprocessed = new Preprocessed();
            _threshold = threshold;
        }

        public void ProcessSingleBatch(
            DenseTensor<float>? output,
            int padX, int padY, float r,
            List<Detection> detections,
            HashSet<int> targetClasses)
        {
            _preprocessed.PreproccessedOutputYolov26(output, padX, padY, r, detections, targetClasses,
                thresHold: _threshold);
        }

        public void ProcessDoubleBatch(
            DenseTensor<float>? output,
            List<Detection> leftDetections,
            List<Detection> rightDetections,
            int padX1, int padY1, float r1,
            int padX2, int padY2, float r2,
            HashSet<int> targetClasses)
        {
            _preprocessed.PreproccessedOutputBatchOptimizedYolov26(output,
                leftDetections, rightDetections,
                padX1, padY1, r1,
                padX2, padY2, r2,
                targetClasses,
                thresHold: _threshold);
        }
    }
}
