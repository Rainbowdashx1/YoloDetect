using ByteTrack;
using OpenCvSharp;
using YoloDetect.PreProcess;

namespace YoloDetect.VideoCapture
{
    internal class FrameRender
    {
        private Mat? _overlay;

        public void DrawDetections(Mat frame, List<Detection> detections)
        {
            DrawDetectionCounter(frame, detections.Count);

            // Usar for en lugar de foreach para evitar allocator del enumerador
            for (int i = 0; i < detections.Count; i++)
            {
                var detection = detections[i];

                int x1 = (int)detection.X1;
                int y1 = (int)detection.Y1;
                int x2 = (int)detection.X2;
                int y2 = (int)detection.Y2;

                var p1 = new OpenCvSharp.Point(x1, y1);
                var p2 = new OpenCvSharp.Point(x2, y2);

                Cv2.Rectangle(frame, p1, p2, Scalar.Red, 2);
                string label = $"Clase {detection.ClassId} ({detection.Score:P1})";

                Cv2.PutText(frame, label, new OpenCvSharp.Point(x1, y1 - 10),
                    HersheyFonts.HersheySimplex, 0.5, Scalar.Yellow, 1);
            }

        }
        public void DrawDetectionCounter(Mat frame, int count)
        {
            // En .NET 8, string interpolation es optimizada automáticamente
            string counterText = $"Detecciones: {count}";

            const int fontFace = (int)HersheyFonts.HersheySimplex;
            const double fontScale = 1.2;
            const int thickness = 2;
            const int padding = 15;

            var textSize = Cv2.GetTextSize(counterText, (HersheyFonts)fontFace,
                fontScale, thickness, out int baseline);

            int boxX = padding;
            int boxY = padding;
            int boxWidth = textSize.Width + padding * 2;
            int boxHeight = textSize.Height + padding * 2;

            // Reutilizar overlay en lugar de clonar cada vez (MAYOR GANANCIA)
            if (_overlay == null || _overlay.Width != frame.Width || _overlay.Height != frame.Height)
            {
                _overlay?.Dispose();
                _overlay = frame.Clone();
            }
            else
            {
                frame.CopyTo(_overlay);
            }

            var p1 = new OpenCvSharp.Point(boxX, boxY);
            var p2 = new OpenCvSharp.Point(boxX + boxWidth, boxY + boxHeight);

            Cv2.Rectangle(_overlay, p1, p2, new Scalar(0, 0, 0), -1);

            const double alpha = 0.6;
            Cv2.AddWeighted(_overlay, alpha, frame, 1 - alpha, 0, frame);

            Cv2.PutText(frame, counterText,
                new OpenCvSharp.Point(boxX + padding, boxY + textSize.Height + padding),
                (HersheyFonts)fontFace, fontScale, new Scalar(0, 255, 0), thickness);
        }
        public void DrawSTracksWithIds(Mat frame, List<STrack> stracks)
        {
            foreach (var track in stracks)
            {
                if (track.tlbr == null || track.tlbr.Length < 4)
                    continue;

                int x1 = (int)track.tlbr[0];
                int y1 = (int)track.tlbr[1];
                int x2 = (int)track.tlbr[2];
                int y2 = (int)track.tlbr[3];

                // Color diferente según el estado del track
                Scalar color = track.state switch
                {
                    TrackState.Tracked => Scalar.Green,
                    TrackState.New => Scalar.Yellow,
                    TrackState.Lost => Scalar.Red,
                    _ => Scalar.Gray
                };

                Cv2.Rectangle(frame, new Point(x1, y1), new Point(x2, y2), color, 2);

                string trackText = $"ID:{track.track_id} {track.score:F2}";
                Cv2.PutText(frame, trackText, new Point(x1, y1 - 5),
                    HersheyFonts.HersheySimplex, 0.5, color, 2);
            }
        }
    }
}
