using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloDetect.VideoCapture.ProcesorDetection
{
    public static class ProcessorFactory
    {
        public static IDetectionProcessor Create(ModelType type, float threshold = 0.25f)
        {
            return type switch
            {
                ModelType.Yolo11 => new Yolo11Processor(threshold),
                ModelType.Yolo26 => new Yolo26Processor(threshold),
                _ => throw new ArgumentException($"Modelo no soportado: {type}")
            };
        }
    }
}
