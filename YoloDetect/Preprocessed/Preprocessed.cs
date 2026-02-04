using Microsoft.ML.OnnxRuntime.Tensors;

namespace YoloDetect.PreProcess
{
    public class Preprocessed
    {
        public void PreproccessedOutput(DenseTensor<float>? output0, int padX, int padY, float r,List<Detection> _Detections, bool nonMaxSuppression = true, float nonMaxSuppressionThreshold = 0.45f, float thresHold = 0.25f)
        {
            if (output0 is null)
                return;

            ReadOnlySpan<float> buffer = output0.Buffer.Span;//No crea allocation porque es solo una referencia al buffer interno

            var dims = output0.Dimensions;

            int channels = dims[1];  // 84 para YOLO11
            int numPreds = dims[2];  // 8400 típicamente
            int stride = channels * numPreds;

            int batch = dims[0];
            int maxClsIdx = 4;

            int offsetX = 0 * numPreds;      // Canal 0: xCenter
            int offsetY = 1 * numPreds;      // Canal 1: yCenter  
            int offsetW = 2 * numPreds;      // Canal 2: width
            int offsetH = 3 * numPreds;      // Canal 3: height
            int offsetScore = 4 * numPreds;  // Canal 4: score

            for (int i = 0; i < numPreds; i++)
            {
                float xCenter = buffer[offsetX + i];
                float yCenter = buffer[offsetY + i];
                float w = buffer[offsetW + i];
                float h = buffer[offsetH + i];
                float clsScore = buffer[offsetScore + i];

                if (clsScore < thresHold)
                    continue;

                // --------------------------------------------------------------------------------
                // A) xywh -> x1, y1, x2, y2 in the LETTERBOX IMAGE (640x640)
                // --------------------------------------------------------------------------------
                float x1_640 = xCenter - w / 2f;
                float y1_640 = yCenter - h / 2f;
                float x2_640 = xCenter + w / 2f;
                float y2_640 = yCenter + h / 2f;

                // --------------------------------------------------------------------------------
                // B) Remove the padding applied in the letterbox
                // --------------------------------------------------------------------------------
                float x1_nopad = x1_640 - padX;
                float y1_nopad = y1_640 - padY;
                float x2_nopad = x2_640 - padX;
                float y2_nopad = y2_640 - padY;

                // --------------------------------------------------------------------------------
                // C) Scale back to the original image by dividing by 'ratio'
                // --------------------------------------------------------------------------------
                float x1_orig = x1_nopad / r;
                float y1_orig = y1_nopad / r;
                float x2_orig = x2_nopad / r;
                float y2_orig = y2_nopad / r;

                // Store detection in the ORIGINAL image coordinates
                _Detections.Add(new Detection(
                    x1_orig,
                    y1_orig,
                    x2_orig,
                    y2_orig,
                    clsScore,
                    maxClsIdx
                ));
            }

            if (nonMaxSuppression)
            {
                NonMaxSuppression(_Detections, nonMaxSuppressionThreshold);
            }
        }
        public void PreproccessedOutputBatchOptimized(
        DenseTensor<float>? output0,
        List<Detection> DetectionRight,
        List<Detection> DetectionLeft,
        int padX1, int padY1, float r1,
        int padX2, int padY2, float r2,
        bool nonMaxSuppression = true,
        float nonMaxSuppressionThreshold = 0.45f,
        float thresHold = 0.25f)
        {
            if (output0 is null)
                return;

            ReadOnlySpan<float> buffer = output0.Buffer.Span;//No crea allocation porque es solo una referencia al buffer interno
            var dims = output0.Dimensions;

            int channels = dims[1];  // 84 para YOLO11
            int numPreds = dims[2];  // 8400 típicamente
            int stride = channels * numPreds;

            int batch = dims[0];
            int maxClsIdx = 4;

            int offsetX = 0 * numPreds;      // Canal 0: xCenter
            int offsetY = 1 * numPreds;      // Canal 1: yCenter  
            int offsetW = 2 * numPreds;      // Canal 2: width
            int offsetH = 3 * numPreds;      // Canal 3: height
            int offsetScore = 4 * numPreds;  // Canal 4: score

            /*
             
                float xCenter = buffer[offsetX + i];
                float yCenter = buffer[offsetY + i];
                float w = buffer[offsetW + i];
                float h = buffer[offsetH + i];
                float clsScore = buffer[offsetScore + i];
             */

            // Pre-calcular valores constantes
            float invR1 = 1f / r1;
            float invR2 = 1f / r2;

            // Procesar ambas imágenes en un solo loop
            for (int i = 0; i < numPreds; i++)
            {
                // Procesar imagen izquierda (batch 0)
                float clsScore0 = buffer[offsetScore + i];
                if (clsScore0 >= thresHold)
                {
                    float xCenter = buffer[offsetX + i];
                    float yCenter = buffer[offsetY + i];
                    float halfW = buffer[offsetW + i] * 0.5f;
                    float halfH = buffer[offsetH + i] * 0.5f;

                    DetectionLeft.Add(new Detection(
                        (xCenter - halfW - padX1) * invR1,
                        (yCenter - halfH - padY1) * invR1,
                        (xCenter + halfW - padX1) * invR1,
                        (yCenter + halfH - padY1) * invR1,
                        clsScore0,
                        maxClsIdx
                    ));
                }

                // Procesar imagen derecha (batch 1)
                int batch1Offset = stride;

                float clsScore1 = buffer[batch1Offset + offsetScore + i];
                if (clsScore1 >= thresHold)
                {
                    float xCenter = buffer[batch1Offset + offsetX + i];
                    float yCenter = buffer[batch1Offset + offsetY + i];
                    float halfW = buffer[batch1Offset + offsetW + i] * 0.5f;
                    float halfH = buffer[batch1Offset + offsetH + i] * 0.5f;

                    DetectionRight.Add(new Detection(
                        (xCenter - halfW - padX2) * invR2,
                        (yCenter - halfH - padY2) * invR2,
                        (xCenter + halfW - padX2) * invR2,
                        (yCenter + halfH - padY2) * invR2,
                        clsScore1,
                        maxClsIdx
                    ));
                }
            }

            if (nonMaxSuppression)
            {
                NonMaxSuppression(DetectionLeft, nonMaxSuppressionThreshold);
                NonMaxSuppression(DetectionRight, nonMaxSuppressionThreshold);
            }
        }
        /// <summary>
        /// Post-procesamiento para YOLOv26
        /// Output shape: [1, 300, 6] donde 6 = [x1, y1, x2, y2, score, class_id]
        /// </summary>
        public void PreproccessedOutputYolov26(DenseTensor<float>? output0, int padX, int padY, float r,
            List<Detection> _Detections, float thresHold = 0.25f, int targetClass = 0)
        {
            if (output0 is null)
                return;
            ReadOnlySpan<float> buffer = output0.Buffer.Span;

            var dims = output0.Dimensions;
            // dims[0] = batch (1)
            // dims[1] = max detections (300)
            // dims[2] = 6 valores [x1, y1, x2, y2, score, class_id]
            int numPreds = dims[1];
            int stride = 6; // 6 valores por detección

            for (int i = 0; i < numPreds; i++)
            {
                int offset = i * stride;

                float score = buffer[offset + 4];

                // Si score es 0 o muy bajo, probablemente ya no hay más detecciones válidas
                if (score < thresHold)
                    continue;

                int classId = (int)buffer[offset + 5];

                // Filtrar por clase si es necesario (0 = persona en COCO)
                if (classId != targetClass)
                    continue;

                // Coordenadas ya en formato x1,y1,x2,y2 (esquinas) en espacio 640x640
                float x1_640 = buffer[offset + 0];
                float y1_640 = buffer[offset + 1];
                float x2_640 = buffer[offset + 2];
                float y2_640 = buffer[offset + 3];

                // Remover padding del letterbox
                float x1_nopad = x1_640 - padX;
                float y1_nopad = y1_640 - padY;
                float x2_nopad = x2_640 - padX;
                float y2_nopad = y2_640 - padY;

                // Escalar a coordenadas originales
                float x1_orig = x1_nopad / r;
                float y1_orig = y1_nopad / r;
                float x2_orig = x2_nopad / r;
                float y2_orig = y2_nopad / r;

                _Detections.Add(new Detection(x1_orig, y1_orig, x2_orig, y2_orig, score, classId));
            }
        }
        /// <summary>
        /// Post-procesamiento optimizado para YOLOv26 con 2 batch
        /// Output shape: [2, 300, 6] donde 6 = [x1, y1, x2, y2, score, class_id]
        /// </summary>
        public void PreproccessedOutputBatchOptimizedYolov26(
            DenseTensor<float>? output0,
            List<Detection> DetectionRight,
            List<Detection> DetectionLeft,
            int padX1, int padY1, float r1,
            int padX2, int padY2, float r2,
            float thresHold = 0.25f,
            int targetClass = 0)
        {
            if (output0 is null)
                return;

            ReadOnlySpan<float> buffer = output0.Buffer.Span;

            var dims = output0.Dimensions;
            // dims[0] = batch (2)
            // dims[1] = max detections (300)
            // dims[2] = 6 valores [x1, y1, x2, y2, score, class_id]
            int numPreds = dims[1];

            int stride = 6; // 6 valores por detección
            int batchStride = numPreds * stride; // Stride completo para saltar de batch 0 a batch 1


            // Pre-calcular valores constantes
            float invR1 = 1f / r1;
            float invR2 = 1f / r2;

            // Procesar ambas imágenes en un solo loop
            for (int i = 0; i < numPreds; i++)
            {
                int offset = i * stride;
                // Procesar imagen izquierda (batch 0)
                float score0 = buffer[offset + 4];
                if (score0 >= thresHold)
                {
                    int classId0 = (int)buffer[offset + 5];
                    if (classId0 == targetClass)
                    {
                        // Coordenadas ya en formato x1,y1,x2,y2 en espacio 640x640
                        float x1_640 = buffer[offset + 0];
                        float y1_640 = buffer[offset + 1];
                        float x2_640 = buffer[offset + 2];
                        float y2_640 = buffer[offset + 3];

                        DetectionLeft.Add(new Detection(
                            (x1_640 - padX1) * invR1,
                            (y1_640 - padY1) * invR1,
                            (x2_640 - padX1) * invR1,
                            (y2_640 - padY1) * invR1,
                            score0,
                            classId0
                        ));
                    }
                }

                int batch1Offset = batchStride + offset;

                // Procesar imagen derecha (batch 1)
                float score1 = buffer[batch1Offset + 4];
                if (score1 >= thresHold)
                {
                    int classId1 = (int)buffer[batch1Offset + 5];
                    if (classId1 == targetClass)
                    {
                        // Coordenadas ya en formato x1,y1,x2,y2 en espacio 640x640
                        float x1_640 = buffer[batch1Offset + 0];
                        float y1_640 = buffer[batch1Offset + 1];
                        float x2_640 = buffer[batch1Offset + 2];
                        float y2_640 = buffer[batch1Offset + 3];

                        DetectionRight.Add(new Detection(
                            (x1_640 - padX2) * invR2,
                            (y1_640 - padY2) * invR2,
                            (x2_640 - padX2) * invR2,
                            (y2_640 - padY2) * invR2,
                            score1,
                            classId1
                        ));
                    }
                }
            }
        }

        private void NonMaxSuppression(List<Detection> detections, float iouThreshold)
        {
            if (detections.Count <= 1)
                return;

            // Ordenar in-place por score descendente
            detections.Sort((a, b) => b.Score.CompareTo(a.Score));

            int writeIndex = 0;

            for (int i = 0; i < detections.Count; i++)
            {
                var current = detections[i];
                bool keep = true;

                // Comparar solo con las detecciones ya aceptadas (0 a writeIndex-1)
                for (int j = 0; j < writeIndex; j++)
                {
                    if (IoU(current, detections[j]) > iouThreshold)
                    {
                        keep = false;
                        break;
                    }
                }

                if (keep)
                {
                    if (i != writeIndex)
                    {
                        detections[writeIndex] = current;
                    }
                    writeIndex++;
                }
            }

            // Eliminar elementos sobrantes al final
            if (writeIndex < detections.Count)
            {
                detections.RemoveRange(writeIndex, detections.Count - writeIndex);
            }
        }
        public float IoU(Detection a, Detection b)
        {
            float interX1 = Math.Max(a.X1, b.X1);
            float interY1 = Math.Max(a.Y1, b.Y1);
            float interX2 = Math.Min(a.X2, b.X2);
            float interY2 = Math.Min(a.Y2, b.Y2);

            float interW = Math.Max(0, interX2 - interX1);
            float interH = Math.Max(0, interY2 - interY1);
            float interArea = interW * interH;

            float areaA = (a.X2 - a.X1) * (a.Y2 - a.Y1);
            float areaB = (b.X2 - b.X1) * (b.Y2 - b.Y1);

            float iou = interArea / (areaA + areaB - interArea);
            return iou;
        }
    }
    public struct Detection
    {
        public float X1;
        public float Y1;
        public float X2;
        public float Y2;
        public float Score;
        public int ClassId;
        public Detection(float x1, float y1, float x2, float y2, float score, int classId)
        {
            X1 = x1; Y1 = y1; X2 = x2; Y2 = y2;
            Score = score;
            ClassId = classId;
        }
    }
}
