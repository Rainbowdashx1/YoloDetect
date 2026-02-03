using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using YoloDetect.Nvidia;
using YoloDetect.PreProcess;
using YoloDetect.VideoSources;

namespace YoloDetect.VideoCapture
{
    internal class Capture
    {
        private readonly string videoPath;
        private readonly string? videoProcessPath;
        private readonly ProcessFrame process;
        private readonly SessionGpu session;
        private readonly Preprocessed prePro;
        private readonly FrameRender frameRender;
        private readonly VideoSourceType? preferredSourceType;
        /*Buffer Reutilizables*/
        private Mat letterboxBuffer;
        private Mat leftLetterboxBuffer;
        private Mat rightLetterboxBuffer;
        private  List<Detection> _Detections;
        private List<Detection> _DetectionsLeft;
        private List<Detection> _DetectionsRight;
        private List<Detection> _DetectionUnion;
        public Capture(string videoPath, string? videoProcessPath, string modelPath, VideoSourceType? preferredSourceType = null) 
        {
            this.videoPath = videoPath;
            this.videoProcessPath = videoProcessPath;
            this.preferredSourceType = preferredSourceType;
            process = new ProcessFrame();
            session = new SessionGpu(modelPath);
            prePro = new Preprocessed();
            frameRender = new FrameRender();

            // Pre-alocar buffers para letterbox
            letterboxBuffer = new Mat(new Size(640, 640), MatType.CV_8UC3);
            leftLetterboxBuffer = new Mat(new Size(640, 640), MatType.CV_8UC3);
            rightLetterboxBuffer = new Mat(new Size(640, 640), MatType.CV_8UC3);
            _Detections = new List<Detection>();
            _DetectionsLeft = new List<Detection>();
            _DetectionsRight = new List<Detection>();
            _DetectionUnion = new List<Detection>();
        }
        public void runWithModel1Batch()
        {
            using var videoSource = VideoSourceFactory.Create(videoPath, preferredSourceType, lowLatency: true);
            using var videoWriter = CreateVideoWriter(videoSource);

            try
            {
                Mat frame = new Mat();
                int currentFrame = 0;
                int skippedFrames = 0;

                while (videoSource.Read(frame))
                {
                    currentFrame++;
                    if (frame.Empty())
                    {
                        skippedFrames++;
                        continue;
                    }

                    ProcessFrame(frame);
                    frameRender.DrawDetections(frame, _Detections);
                    
                    videoWriter?.Write(frame);
                    Cv2.ImShow("Cuadro Actual", frame);

                    if (Cv2.WaitKey(1) >= 0)
                        break;
                }

                Console.WriteLine($"Frames procesados: {currentFrame}, Frames saltados: {skippedFrames}");
                Cv2.DestroyAllWindows();
            }
            finally
            {
                videoWriter?.Dispose();
            }
        }
        public void runWithModel2Batch()
        {
            using var videoSource = VideoSourceFactory.Create(videoPath, preferredSourceType, lowLatency: true);
            using var videoWriter = CreateVideoWriter(videoSource);

            try
            {
                Mat frame = new Mat();
                int currentFrame = 0;
                int skippedFrames = 0;

                while (videoSource.Read(frame))
                {
                    currentFrame++;
                    if (frame.Empty())
                    {
                        skippedFrames++;
                        continue;
                    }

                    ProcessFrameBatchOverLap(frame);
                    frameRender.DrawDetections(frame, _DetectionUnion);
                    
                    videoWriter?.Write(frame);
                    Cv2.ImShow("Cuadro Actual", frame);
                    
                    if (Cv2.WaitKey(1) >= 0)
                        break;
                }

                Console.WriteLine($"Frames procesados: {currentFrame}, Frames saltados: {skippedFrames}");
                Cv2.DestroyAllWindows();
            }
            finally
            {
                videoWriter?.Dispose();
            }
        }
        public void runWithModel1BatchYolo26()
        {
            using var videoSource = VideoSourceFactory.Create(videoPath, preferredSourceType, lowLatency: true);
            using var videoWriter = CreateVideoWriter(videoSource);

            try
            {
                Mat frame = new Mat();
                int currentFrame = 0;
                int skippedFrames = 0;

                while (videoSource.Read(frame))
                {
                    currentFrame++;
                    if (frame.Empty())
                    {
                        skippedFrames++;
                        continue;
                    }

                    ProcessFrameYolo26(frame);
                    frameRender.DrawDetections(frame, _Detections);

                    videoWriter?.Write(frame);
                    Cv2.ImShow("Cuadro Actual", frame);

                    if (Cv2.WaitKey(1) >= 0)
                        break;
                }

                Console.WriteLine($"Frames procesados: {currentFrame}, Frames saltados: {skippedFrames}");
                Cv2.DestroyAllWindows();
            }
            finally
            {
                videoWriter?.Dispose();
            }
        }
        public void runWithModel2BatchYolo26()
        {
            using var videoSource = VideoSourceFactory.Create(videoPath, preferredSourceType, lowLatency: true);
            using var videoWriter = CreateVideoWriter(videoSource);

            try
            {
                Mat frame = new Mat();
                int currentFrame = 0;
                int skippedFrames = 0;

                while (videoSource.Read(frame))
                {
                    currentFrame++;
                    if (frame.Empty())
                    {
                        skippedFrames++;
                        continue;
                    }

                    ProcessFrameBatchOverLapYolo26(frame);
                    frameRender.DrawDetections(frame, _DetectionUnion);

                    videoWriter?.Write(frame);
                    Cv2.ImShow("Cuadro Actual", frame);

                    if (Cv2.WaitKey(1) >= 0)
                        break;
                }

                Console.WriteLine($"Frames procesados: {currentFrame}, Frames saltados: {skippedFrames}");
                Cv2.DestroyAllWindows();
            }
            finally
            {
                videoWriter?.Dispose();
            }
        }
        private OpenCvSharp.VideoWriter? CreateVideoWriter(IVideoSource source)
        {
            if (videoProcessPath == null)
            {
                Console.WriteLine("No se guardará el video procesado (modo visualización)");
                return null;
            }

            var videoWriter = new OpenCvSharp.VideoWriter(
                videoProcessPath,
                FourCC.XVID,
                source.Fps,
                new OpenCvSharp.Size(source.Width, source.Height)
            );

            if (!videoWriter.IsOpened())
            {
                throw new Exception("No se pudo abrir el escritor de video.");
            }

            Console.WriteLine($"Video de salida configurado: {Path.GetFileName(videoProcessPath)}");
            return videoWriter;
        }
        private void ProcessFrame(Mat frame)
        {
            float r;
            int padX, padY;
            process.LetterboxOptimized(frame, letterboxBuffer, 640, 640, out r, out padX, out padY);
            Tensor<float>? output0 = session.SessionRun(letterboxBuffer);
            _Detections.Clear();
            prePro.PreproccessedOutput(output0, padX, padY, r, _Detections);
        }
        private void ProcessFrameYolo26(Mat frame)
        {
            float r;
            int padX, padY;
            process.LetterboxOptimized(frame, letterboxBuffer, 640, 640, out r, out padX, out padY);
            Tensor<float>? output0 = session.SessionRun(letterboxBuffer);
            _Detections.Clear();
            prePro.PreproccessedOutputYolov26(output0, padX, padY, r, _Detections);
        }
        private void ProcessFrameBatchOverLap(Mat frame)
        {
            int overlapPixels = 150; // Solapamiento configurable
            int halfWidth = frame.Width / 2;

            int leftWidth = halfWidth + overlapPixels;
            int rightStart = halfWidth - overlapPixels;
            int rightWidth = frame.Width - rightStart;

            using Mat leftRegion = new Mat(frame, new Rect(0, 0, leftWidth, frame.Height));
            using Mat rightRegion = new Mat(frame, new Rect(rightStart, 0, rightWidth, frame.Height));

            float r1, r2;
            int padX1, padY1, padX2, padY2;
    
            process.LetterboxOptimized(leftRegion, leftLetterboxBuffer, 640, 640, out r1, out padX1, out padY1);
            process.LetterboxOptimized(rightRegion, rightLetterboxBuffer, 640, 640, out r2, out padX2, out padY2);

            _DetectionsRight.Clear();
            _DetectionsLeft.Clear();

            Tensor<float>? outputSession = session.SessionRunBatch(leftLetterboxBuffer, rightLetterboxBuffer);
            prePro.PreproccessedOutputBatchOptimized(
                outputSession,
                _DetectionsRight,
                _DetectionsLeft,
                padX1, padY1, r1,
                padX2, padY2, r2
            );
  
            for (int i = 0; i < _DetectionsRight.Count; i++)
            {
                var det = _DetectionsRight[i];
                _DetectionsRight[i] = new Detection(
                    det.X1 + rightStart,
                    det.Y1,
                    det.X2 + rightStart,
                    det.Y2,
                    det.Score,
                    det.ClassId
                );
            }
            _DetectionUnion.Clear();
            // Combinar y eliminar duplicados en la zona solapada
            MergeOverlappingDetections(
                _DetectionsLeft,
                _DetectionsRight,
                _DetectionUnion,
                halfWidth - overlapPixels,
                halfWidth + overlapPixels
            );
        }
        private void ProcessFrameBatchOverLapYolo26(Mat frame)
        {
            int overlapPixels = 150; // Solapamiento configurable
            int halfWidth = frame.Width / 2;

            int leftWidth = halfWidth + overlapPixels;
            int rightStart = halfWidth - overlapPixels;
            int rightWidth = frame.Width - rightStart;

            using Mat leftRegion = new Mat(frame, new Rect(0, 0, leftWidth, frame.Height));
            using Mat rightRegion = new Mat(frame, new Rect(rightStart, 0, rightWidth, frame.Height));

            float r1, r2;
            int padX1, padY1, padX2, padY2;

            process.LetterboxOptimized(leftRegion, leftLetterboxBuffer, 640, 640, out r1, out padX1, out padY1);
            process.LetterboxOptimized(rightRegion, rightLetterboxBuffer, 640, 640, out r2, out padX2, out padY2);

            Tensor<float>? outputSession = session.SessionRunBatch(leftLetterboxBuffer, rightLetterboxBuffer);

            _DetectionsRight.Clear();
            _DetectionsLeft.Clear();

            prePro.PreproccessedOutputBatchOptimizedYolov26(
                outputSession,
                _DetectionsRight,
                _DetectionsLeft,
                padX1, padY1, r1,
                padX2, padY2, r2
            );

            for (int i = 0; i < _DetectionsRight.Count; i++)
            {
                var det = _DetectionsRight[i];
                _DetectionsRight[i] = new Detection(
                    det.X1 + rightStart,
                    det.Y1,
                    det.X2 + rightStart,
                    det.Y2,
                    det.Score,
                    det.ClassId
                );
            }
            _DetectionUnion.Clear();
            // Combinar y eliminar duplicados en la zona solapada
            MergeOverlappingDetections(
                _DetectionsLeft,
                _DetectionsRight,
                _DetectionUnion,
                halfWidth - overlapPixels,
                halfWidth + overlapPixels
            );
        }
        private void MergeOverlappingDetections(
            List<Detection> leftDetections,
            List<Detection> rightDetections,
            List<Detection> _DetectionUnion,
            float overlapStart,
            float overlapEnd)
        {
            var result = new List<Detection>();
            var processedRight = new HashSet<int>();

            foreach (var leftDet in leftDetections)
            {
                bool isDuplicate = false;
                float leftCenter = (leftDet.X1 + leftDet.X2) / 2f;

                if (leftCenter >= overlapStart && leftCenter <= overlapEnd)
                {
                    for (int i = 0; i < rightDetections.Count; i++)
                    {
                        if (processedRight.Contains(i))
                            continue;

                        var rightDet = rightDetections[i];
                        float rightCenter = (rightDet.X1 + rightDet.X2) / 2f;

                        if (rightCenter >= overlapStart && rightCenter <= overlapEnd)
                        {
                            float iou = prePro.IoU(leftDet, rightDet);

                            if (iou > 0.5f)
                            {
                                if (rightDet.Score > leftDet.Score)
                                {
                                    _DetectionUnion.Add(rightDet);
                                    isDuplicate = true;
                                }
                                else
                                {
                                    _DetectionUnion.Add(leftDet);
                                }

                                processedRight.Add(i);
                                isDuplicate = true;
                                break;
                            }
                        }
                    }
                }

                if (!isDuplicate)
                {
                    _DetectionUnion.Add(leftDet);
                }
            }

            for (int i = 0; i < rightDetections.Count; i++)
            {
                if (!processedRight.Contains(i))
                {
                    _DetectionUnion.Add(rightDetections[i]);
                }
            }
        }
    }
}
