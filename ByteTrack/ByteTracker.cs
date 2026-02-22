using LapjvCSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace ByteTrack
{
    public class BYTETracker
    {
        // Parámetros principales
        private float track_thresh;
        private float high_thresh;
        private float match_thresh;
        private int frame_id;
        private int max_time_lost;

        // Listas de tracks en diferentes estados
        private List<STrack> tracked_stracks;
        private List<STrack> lost_stracks;
        private List<STrack> removed_stracks;

        // Filtro de Kalman
        private KalmanFilter kalman_filter;
        LapjvCSharp.Lapjv lap = new Lapjv();
        /// <summary>
        /// Constructor.
        /// frame_rate y track_buffer se usan para calcular max_time_lost.
        /// </summary>
        public BYTETracker(int frame_rate = 30, int track_buffer = 30)
        {
            // Ajusta estos umbrales a tu preferencia
            this.track_thresh = 0.30f;
            this.high_thresh = 0.4f;
            this.match_thresh = 0.8f;

            this.frame_id = 0;
            this.max_time_lost = (int)(frame_rate / 30.0f * track_buffer);

            this.tracked_stracks = new List<STrack>();
            this.lost_stracks = new List<STrack>();
            this.removed_stracks = new List<STrack>();

            // Inicializa tu KalmanFilter
            this.kalman_filter = new KalmanFilter();

            Console.WriteLine("Init ByteTrack!");
        }

        ~BYTETracker()
        {
            // En C#, rara vez necesitas lógica en el finalizador lo pongo porque en c++ estaba =P
        }

        /// <summary>
        /// Método principal "Update" que recibe una lista de STrack en vez de detecciones genéricas.
        /// Devuelve los STrack que quedan "tracked" al final.
        /// </summary>
        public List<STrack> Update(List<STrack> objects)
        {
            // 1. Incrementa el número de frame
            this.frame_id++;

            // 2. Contenedores locales para esta iteración
            List<STrack> activated_stracks = new List<STrack>();
            List<STrack> refind_stracks = new List<STrack>();
            List<STrack> removed_stracks_local = new List<STrack>();
            List<STrack> lost_stracks_local = new List<STrack>();

            List<STrack> detections = new List<STrack>();
            List<STrack> detections_low = new List<STrack>();
            List<STrack> detections_cp = new List<STrack>();

            List<STrack> tracked_stracks_swap = new List<STrack>();
            List<STrack> resa = new List<STrack>();
            List<STrack> resb = new List<STrack>();
            List<STrack> output_stracks = new List<STrack>();

            // 3. Separa "objects" en detections (score >= track_thresh) y detections_low
            foreach (var str in objects)
            {
                if (str.score >= this.track_thresh)
                    detections.Add(str);
                else
                    detections_low.Add(str);
            }

            // 4. Separa en unconfirmed (no activados) y tracked (activados)
            List<STrack> unconfirmedLocal = new List<STrack>();
            List<STrack> tracked_stracksLocal = new List<STrack>();
            foreach (var ts in this.tracked_stracks)
            {
                if (!ts.is_activated)
                    unconfirmedLocal.Add(ts);
                else
                    tracked_stracksLocal.Add(ts);
            }

            // -------------------- PASO 2: Primera asociación con IoU --------------------
            // a) strack_pool = tracked_stracksLocal + lost_stracks
            List<STrack> strack_pool = joint_stracks(tracked_stracksLocal, this.lost_stracks);

            // b) multi_predict => kalman_filter.predict(...) para cada uno
            STrack.multi_predict(strack_pool, this.kalman_filter);

            // c) Calculamos matriz de costo IoU => float[,]
            float[,] costMat1 = iou_distance_array(strack_pool, detections);

            // d) Llamamos a linear_assignment_lapjv
            linear_assignment_lapjv(
                costMat1,
                this.match_thresh,  // coste = 1 - IoU; umbral de distancia directo como en ByteTrack
                out List<(int row, int col)> matches,
                out List<int> u_track,
                out List<int> u_detection);

            // e) Actualizamos tracks emparejados
            foreach (var (row, col) in matches)
            {
                STrack track = strack_pool[row];
                STrack det = detections[col];

                if (track.state == TrackState.Tracked)
                {
                    track.update(det, this.frame_id);
                    activated_stracks.Add(track);
                }
                else
                {
                    track.re_activate(det, this.frame_id, false);
                    refind_stracks.Add(track);
                }
            }

            // -------------------- PASO 3: Segunda asociación con detections_low --------------------
            // a) detections_cp = detections[u_detection]
            for (int i = 0; i < u_detection.Count; i++)
            {
                detections_cp.Add(detections[u_detection[i]]);
            }
            detections.Clear();
            detections.AddRange(detections_low);

            // b) Solo tracks "Tracked" no emparejados => r_tracked_stracks
            List<STrack> r_tracked_stracks = new List<STrack>();
            for (int i = 0; i < u_track.Count; i++)
            {
                if (strack_pool[u_track[i]].state == TrackState.Tracked)
                {
                    r_tracked_stracks.Add(strack_pool[u_track[i]]);
                }
            }

            // c) Calcular costMat con r_tracked_stracks vs detections
            float[,] costMat2 = iou_distance_array(r_tracked_stracks, detections);

            // d) Asignación con distancia <= 0.5 (=> IoU >= 0.5)
            linear_assignment_lapjv(
                costMat2,
                0.5,
                out List<(int row, int col)> matches2,
                out List<int> u_track2,
                out List<int> u_detection2);

            // e) Actualizamos los emparejados
            foreach (var (row, col) in matches2)
            {
                STrack track = r_tracked_stracks[row];
                STrack det = detections[col];

                if (track.state == TrackState.Tracked)
                {
                    track.update(det, this.frame_id);
                    activated_stracks.Add(track);
                }
                else
                {
                    track.re_activate(det, this.frame_id, false);
                    refind_stracks.Add(track);
                }
            }

            // f) Los no emparejados => lost
            for (int i = 0; i < u_track2.Count; i++)
            {
                STrack track = r_tracked_stracks[u_track2[i]];
                if (track.state != TrackState.Lost)
                {
                    track.mark_lost();
                    lost_stracks_local.Add(track);
                }
            }

            // g) Manejo unconfirmed
            detections.Clear();
            detections.AddRange(detections_cp);

            float[,] costMatUnc = iou_distance_array(unconfirmedLocal, detections);
            linear_assignment_lapjv(
                costMatUnc,
                0.7, // distancia <= 0.7 (=> IoU >= 0.3)
                out List<(int row, int col)> matches3,
                out List<int> u_unconfirmed,
                out List<int> u_detUnc);

            foreach (var (r, c) in matches3)
            {
                unconfirmedLocal[r].update(detections[c], this.frame_id);
                activated_stracks.Add(unconfirmedLocal[r]);
            }

            for (int i = 0; i < u_unconfirmed.Count; i++)
            {
                STrack track = unconfirmedLocal[u_unconfirmed[i]];
                track.mark_removed();
                removed_stracks_local.Add(track);
            }

            // -------------------- PASO 4: Init nuevos tracks (con high_thresh) --------------------
            for (int i = 0; i < u_detUnc.Count; i++)
            {
                STrack track = detections[u_detUnc[i]];
                if (track.score < this.high_thresh)
                    continue;
                track.activate(this.kalman_filter, this.frame_id);
                activated_stracks.Add(track);
            }

            // -------------------- PASO 5: Update state (los perdidos y removidos) --------------------
            // a) Los perdidos que superan max_time_lost => removed
            for (int i = 0; i < this.lost_stracks.Count; i++)
            {
                if (this.frame_id - this.lost_stracks[i].end_frame() > this.max_time_lost)
                {
                    this.lost_stracks[i].mark_removed();
                    removed_stracks_local.Add(this.lost_stracks[i]);
                }
            }

            // b) Depurar tracked_stracks => solo los que siguen Tracked
            tracked_stracks_swap.Clear();
            foreach (var ts in this.tracked_stracks)
            {
                if (ts.state == TrackState.Tracked)
                {
                    tracked_stracks_swap.Add(ts);
                }
            }
            this.tracked_stracks.Clear();
            this.tracked_stracks.AddRange(tracked_stracks_swap);

            // c) Se unen con activated_stracks y refind_stracks
            this.tracked_stracks = joint_stracks(this.tracked_stracks, activated_stracks);
            this.tracked_stracks = joint_stracks(this.tracked_stracks, refind_stracks);

            // d) Quitar de lost_stracks los que ahora están en tracked
            this.lost_stracks = sub_stracks(this.lost_stracks, this.tracked_stracks);
            // Añadir los que hemos perdido en esta pasada
            foreach (var ls in lost_stracks_local)
            {
                this.lost_stracks.Add(ls);
            }

            // e) Quitar de lost_stracks los que han sido removidos (incluidos los locales de esta iteración)
            foreach (var rm in removed_stracks_local)
            {
                this.removed_stracks.Add(rm);
            }
            this.lost_stracks = sub_stracks(this.lost_stracks, this.removed_stracks);

            // f) Elimina duplicados (entre tracked y lost)
            remove_duplicate_stracks(resa, resb, this.tracked_stracks, this.lost_stracks);
            this.tracked_stracks.Clear();
            this.tracked_stracks.AddRange(resa);
            this.lost_stracks.Clear();
            this.lost_stracks.AddRange(resb);

            // g) Output => los que están is_activated
            foreach (var ts in this.tracked_stracks)
            {
                if (ts.is_activated)
                {
                    output_stracks.Add(ts);
                }
            }

            return output_stracks;
        }

        /// <summary>
        /// Calcula la matriz de costo IoU (1 - IoU) y la guarda en un float[,] de tamaño (rows x cols).
        /// </summary>
        private float[,] iou_distance_array(List<STrack> aTracks, List<STrack> bTracks)
        {
            int rows = aTracks.Count;
            int cols = bTracks.Count;
            float[,] costMatrix = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float iouVal = computeIoU(aTracks[i], bTracks[j]);
                    costMatrix[i, j] = 1f - iouVal;
                }
            }
            return costMatrix;
        }

        /// <summary>
        /// Se asume que STrack.tlbr = [x1, y1, x2, y2].
        /// </summary>
        private float computeIoU(STrack a, STrack b)
        {
            float ax1 = a.tlbr[0], ay1 = a.tlbr[1];
            float ax2 = a.tlbr[2], ay2 = a.tlbr[3];
            float bx1 = b.tlbr[0], by1 = b.tlbr[1];
            float bx2 = b.tlbr[2], by2 = b.tlbr[3];

            float inter_x1 = Math.Max(ax1, bx1);
            float inter_y1 = Math.Max(ay1, by1);
            float inter_x2 = Math.Min(ax2, bx2);
            float inter_y2 = Math.Min(ay2, by2);

            float inter_w = Math.Max(0f, inter_x2 - inter_x1);
            float inter_h = Math.Max(0f, inter_y2 - inter_y1);
            float inter_area = inter_w * inter_h;

            float areaA = (ax2 - ax1) * (ay2 - ay1);
            float areaB = (bx2 - bx1) * (by2 - by1);

            float union_area = areaA + areaB - inter_area;
            if (union_area < 1e-6f) return 0f;

            return inter_area / union_area;
        }

        /// <summary>
        /// Asignación lineal usando lapjvCsharp. 
        /// - costMatrix: float[,] (rows x cols).
        /// - costLimit: umbral para que la librería no asigne si costo > costLimit.
        /// - matches: pares (fila, columna) resultantes.
        /// - unmatchedRows/Cols: filas o columnas sin asignar.
        /// </summary>
        private void linear_assignment_lapjv(
            float[,] costMatrix,
            double costLimit,
            out List<(int row, int col)> matches,
            out List<int> unmatchedRows,
            out List<int> unmatchedCols)
        {
            int rows = costMatrix.GetLength(0);
            int cols = costMatrix.GetLength(1);

            if (rows == 0 || cols == 0)
            {
                // No podemos asignar nada
                matches = new List<(int row, int col)>();
                unmatchedRows = new List<int>();
                unmatchedCols = new List<int>();

                // Si rows > 0 => todas filas sin asignar
                for (int i = 0; i < rows; i++)
                    unmatchedRows.Add(i);

                // Si cols > 0 => todas columnas sin asignar
                for (int j = 0; j < cols; j++)
                    unmatchedCols.Add(j);

                // Y salimos
                return;
            }

            // 1) float[,] => double[,]
            double[,] costMatrixDouble = new double[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    costMatrixDouble[r, c] = (double)costMatrix[r, c];
                }
            }

            // 2) Llamada a lapjvCsharp => (int[] x, int[] y)
            var (x, y) = lap.lapjvCsharp(costMatrixDouble, true, costLimit);
            // x[i] = col asignada a la fila i, o -1
            // y[j] = fila asignada a la columna j, o -1

            // 3) Construimos listas de salida
            matches = new List<(int row, int col)>();
            unmatchedRows = new List<int>();
            unmatchedCols = new List<int>();

            bool[] usedCols = new bool[cols];

            // Recorremos filas
            for (int i = 0; i < rows; i++)
            {
                int colSol = x[i];
                if (colSol == -1)
                {
                    // sin asignar
                    unmatchedRows.Add(i);
                }
                else
                {
                    // asignado
                    matches.Add((i, colSol));
                    usedCols[colSol] = true;
                }
            }

            // Columnas no usadas => unmatchedCols
            for (int c = 0; c < cols; c++)
            {
                if (!usedCols[c])
                    unmatchedCols.Add(c);
            }
        }

        /// <summary>
        /// Combina dos listas (evitando duplicados por track_id).
        /// </summary>
        private List<STrack> joint_stracks(List<STrack> tlista, List<STrack> tlistb)
        {
            List<STrack> res = new List<STrack>(tlista);
            foreach (var tb in tlistb)
            {
                bool found = false;
                for (int i = 0; i < res.Count; i++)
                {
                    if (res[i].track_id == tb.track_id)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                    res.Add(tb);
            }
            return res;
        }

        /// <summary>
        /// Elimina de 'tlista' los elementos que estén en 'tlistb' (mismo track_id).
        /// </summary>
        private List<STrack> sub_stracks(List<STrack> tlista, List<STrack> tlistb)
        {
            List<STrack> res = new List<STrack>();
            foreach (var ta in tlista)
            {
                bool duplicate = false;
                foreach (var tb in tlistb)
                {
                    if (ta.track_id == tb.track_id)
                    {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate)
                    res.Add(ta);
            }
            return res;
        }

        /// <summary>
        /// Limpia duplicados entre stracksa y stracksb y los deja en resa, resb.
        /// Duplicado si IoU_distance &lt; 0.15 (=> IoU &gt; 0.85). Se conserva el de mayor tracklet_len.
        /// </summary>
        private void remove_duplicate_stracks(
            List<STrack> resa,
            List<STrack> resb,
            List<STrack> stracksa,
            List<STrack> stracksb)
        {
            resa.Clear();
            resb.Clear();

            if (stracksa.Count == 0)
            {
                resb.AddRange(stracksb);
                return;
            }
            if (stracksb.Count == 0)
            {
                resa.AddRange(stracksa);
                return;
            }

            float[,] dist = iou_distance_array(stracksa, stracksb);
            HashSet<int> da = new HashSet<int>();
            HashSet<int> db = new HashSet<int>();

            float duplicateThresh = 0.15f; // distancia IoU

            for (int i = 0; i < stracksa.Count; i++)
            {
                for (int j = 0; j < stracksb.Count; j++)
                {
                    if (dist[i, j] < duplicateThresh)
                    {
                        // Duplicados: conservar el de mayor tracklet_len
                        if (stracksa[i].tracklet_len > stracksb[j].tracklet_len)
                            db.Add(j);
                        else
                            da.Add(i);
                    }
                }
            }

            for (int i = 0; i < stracksa.Count; i++)
                if (!da.Contains(i)) resa.Add(stracksa[i]);

            for (int j = 0; j < stracksb.Count; j++)
                if (!db.Contains(j)) resb.Add(stracksb[j]);
        }

        /// <summary>
        /// Retorna un Color a partir de un índice, similar a get_color(int idx) en C++.
        /// </summary>
        public Color GetColor(int idx)
        {
            Color[] palette = new Color[]
            {
                Color.Red, Color.Green, Color.Blue,
                Color.Yellow, Color.Cyan, Color.Magenta,
                Color.Orange, Color.Lime, Color.Purple
            };
            return palette[idx % palette.Length];
        }
    }
}
