using ByteTrack;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using YoloDetect.Nvidia;
using YoloDetect.PreProcess;
using YoloDetect.VideoCapture.ProcesorDetection;
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
        private IDetectionProcessor processor;
        /*Buffer Reutilizables*/
        private Mat letterboxBuffer;
        private Mat leftLetterboxBuffer;
        private Mat rightLetterboxBuffer;
        private  List<Detection> _Detections;
        private List<Detection> _DetectionsLeft;
        private List<Detection> _DetectionsRight;
        private List<Detection> _DetectionUnion;

        // Buffers para STracks
        private List<STrack> _STracks;
        private List<STrack> _STracksLeft;
        private List<STrack> _STracksRight;
        private List<STrack> _STrackUnion;


        private HashSet<int> TargetClasses;
        public Capture(string videoPath, string? videoProcessPath, string modelPath, HashSet<int> targetClasses, VideoSourceType? preferredSourceType = null) 
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
            _Detections = new List<Detection>(capacity: 500);
            _DetectionsLeft = new List<Detection>(capacity: 250);
            _DetectionsRight = new List<Detection>(capacity: 250);
            _DetectionUnion = new List<Detection>(capacity: 500);

            // Buffers para STracks
            _STracks = new List<STrack>(capacity: 500);
            _STracksLeft = new List<STrack>(capacity: 250);
            _STracksRight = new List<STrack>(capacity: 250);
            _STrackUnion = new List<STrack>(capacity: 500);

            TargetClasses = targetClasses;
        }
        public void runWithModel1Batch(ModelType modelType)
        {
            processor = ProcessorFactory.Create(modelType);
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
        public void runWithModel1BatchWithTracking(ModelType modelType, int frameRate = 30, int trackBuffer = 30)
        {
            processor = ProcessorFactory.Create(modelType);
            var tracker = new BYTETracker(frameRate, trackBuffer);

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

                    ProcessFrameToSTracks(frame);

                    // Actualizar tracker con las nuevas detecciones
                    var trackedSTracks = tracker.Update(_STracks);

                    frameRender.DrawSTracksWithIds(frame, trackedSTracks);

                    videoWriter?.Write(frame);
                    Cv2.ImShow("Cuadro Actual con Tracking", frame);

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
        public void runWithModel2Batch(ModelType modelType)
        {
            processor = ProcessorFactory.Create(modelType);
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
        public void runWithModel2BatchWithTracking(ModelType modelType, int frameRate = 30, int trackBuffer = 30)
        {
            processor = ProcessorFactory.Create(modelType);
            var tracker = new BYTETracker(frameRate, trackBuffer);

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

                    ProcessFrameBatchOverLapToSTracks(frame);

                    // Actualizar tracker con las nuevas detecciones
                    var trackedSTracks = tracker.Update(_STrackUnion);

                    frameRender.DrawSTracksWithIds(frame, trackedSTracks);

                    videoWriter?.Write(frame);
                    Cv2.ImShow("Cuadro Actual con Tracking", frame);

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
        public void runWithModel1BatchYolo26(ModelType modelType)
        {
            processor = ProcessorFactory.Create(modelType);
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
        public void runWithModel1BatchYolo26Bytetrack(ModelType modelType, int frameRate = 30, int trackBuffer = 30)
        {
            processor = ProcessorFactory.Create(modelType);
            var tracker = new BYTETracker(frameRate, trackBuffer);

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

                    ProcessFrameToSTracks(frame);

                    // Actualizar tracker con las nuevas detecciones
                    var trackedSTracks = tracker.Update(_STracks);

                    frameRender.DrawSTracksWithIds(frame, trackedSTracks);

                    videoWriter?.Write(frame);
                    Cv2.ImShow("Cuadro Actual con Tracking", frame);

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
        public void runWithModel2BatchYolo26(ModelType modelType)
        {
            processor = ProcessorFactory.Create(modelType);
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
        public void runWithModel2BatchYolo26ByteTrack(ModelType modelType, int frameRate = 30, int trackBuffer = 30)
        {
            processor = ProcessorFactory.Create(modelType);
            var tracker = new BYTETracker(frameRate, trackBuffer);

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

                    ProcessFrameBatchOverLapToSTracks(frame);

                    // Actualizar tracker con las nuevas detecciones
                    var trackedSTracks = tracker.Update(_STrackUnion);

                    frameRender.DrawSTracksWithIds(frame, trackedSTracks);

                    videoWriter?.Write(frame);
                    Cv2.ImShow("Cuadro Actual con Tracking", frame);

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
            DenseTensor<float>? output0 = session.SessionRun(letterboxBuffer);
            _Detections.Clear();
            processor.ProcessSingleBatch(output0, padX, padY, r, _Detections, TargetClasses);
        }
        private void ProcessFrameToSTracks(Mat frame)
        {
            float r;
            int padX, padY;
            process.LetterboxOptimized(frame, letterboxBuffer, 640, 640, out r, out padX, out padY);
            DenseTensor<float>? output0 = session.SessionRun(letterboxBuffer);
            _Detections.Clear();
            processor.ProcessSingleBatch(output0, padX, padY, r, _Detections, TargetClasses);

            // Convertir detecciones a STracks
            _STracks.Clear();
            foreach (var det in _Detections)
            {
                float x = det.X1;
                float y = det.Y1;
                float w = det.X2 - det.X1;
                float h = det.Y2 - det.Y1;
                float[] tlwh = new float[] { x, y, w, h };
                _STracks.Add(new STrack(tlwh, det.Score, det.X2, det.Y2));
            }
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

            DenseTensor<float>? outputSession = session.SessionRunBatch(leftLetterboxBuffer, rightLetterboxBuffer);
            processor.ProcessDoubleBatch(
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
        private void ProcessFrameBatchOverLapToSTracks(Mat frame)
        {
            int overlapPixels = 150;
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

            DenseTensor<float>? outputSession = session.SessionRunBatch(leftLetterboxBuffer, rightLetterboxBuffer);
            processor.ProcessDoubleBatch(
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
            MergeOverlappingDetections(
                _DetectionsLeft,
                _DetectionsRight,
                _DetectionUnion,
                halfWidth - overlapPixels,
                halfWidth + overlapPixels
            );

            // Convertir detecciones unificadas a STracks
            _STrackUnion.Clear();
            foreach (var det in _DetectionUnion)
            {
                float x = det.X1;
                float y = det.Y1;
                float w = det.X2 - det.X1;
                float h = det.Y2 - det.Y1;
                float[] tlwh = new float[] { x, y, w, h };
                _STrackUnion.Add(new STrack(tlwh, det.Score, det.X2, det.Y2));
            }
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
