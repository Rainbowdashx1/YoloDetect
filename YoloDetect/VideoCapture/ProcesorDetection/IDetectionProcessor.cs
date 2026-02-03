using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YoloDetect.PreProcess;

namespace YoloDetect.VideoCapture.ProcesorDetection
{
    public interface IDetectionProcessor
    {
        void ProcessSingleBatch(
            Tensor<float>? output,
            int padX, int padY, float r,
            List<Detection> detections);

        void ProcessDoubleBatch(
            Tensor<float>? output,
            List<Detection> leftDetections,
            List<Detection> rightDetections,
            int padX1, int padY1, float r1,
            int padX2, int padY2, float r2);
    }
}
