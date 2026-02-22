using System;
using System.Collections.Generic;

namespace ByteTrack
{
    // Equivalente al enum TrackState en C++
    public enum TrackState
    {
        New = 0,
        Tracked,
        Lost,
        Removed
    }

    public class STrack
    {
        public bool is_activated;
        public int track_id;
        public TrackState state;

        public float[] _tlwh;
        public float[] tlwh;
        public float[] tlbr;

        public float x2;
        public float y2;

        public int frame_id;
        public int tracklet_len;
        public int start_frame;

        public KAL_MEAN mean;
        public KAL_COVA covariance;
        public float score;

        private static int _count = 0;
        private KalmanFilter kalman_filter;
        private readonly DETECTBOX _xyahBox = new DETECTBOX();

        private const float Eps = 1e-6f;

        // Constructor
        public STrack(float[] tlwh_, float score, float x2, float y2)
        {
            _tlwh = new float[4];
            Array.Copy(tlwh_, _tlwh, 4);
            is_activated = false;
            track_id = 0;
            state = TrackState.New;

            tlwh = new float[4];
            tlbr = new float[4];

            static_tlwh();
            static_tlbr();

            frame_id = 0;
            tracklet_len = 0;
            this.score = score;
            start_frame = 0;
            this.x2 = x2;
            this.y2 = y2;
        }

        public void activate(KalmanFilter kalman_filter, int frame_id)
        {
            this.kalman_filter = kalman_filter;
            this.track_id = next_id();

            // Convertimos _tlwh a xyah directamente en _xyahBox (cero asignaciones)
            TlwhToXyah(_tlwh, _xyahBox);

            var mc = this.kalman_filter.initiate(_xyahBox);
            this.mean = mc.Item1;
            this.covariance = mc.Item2;

            static_tlwh();
            static_tlbr();

            this.tracklet_len = 0;
            this.state = TrackState.Tracked;
            if (frame_id == 1)
            {
                this.is_activated = true;
            }

            this.frame_id = frame_id;
            this.start_frame = frame_id;
        }

        public void re_activate(STrack new_track, int frame_id, bool new_id = false)
        {
            TlwhToXyah(new_track.tlwh, _xyahBox);

            var mc = this.kalman_filter.update(this.mean, this.covariance, _xyahBox);
            this.mean = mc.Item1;
            this.covariance = mc.Item2;

            static_tlwh();
            static_tlbr();

            this.tracklet_len = 0;
            this.state = TrackState.Tracked;
            this.is_activated = true;
            this.frame_id = frame_id;
            this.score = new_track.score;

            if (new_id)
                this.track_id = next_id();
        }

        public void update(STrack new_track, int frame_id)
        {
            this.frame_id = frame_id;
            this.tracklet_len++;

            TlwhToXyah(new_track.tlwh, _xyahBox);

            var mc = this.kalman_filter.update(this.mean, this.covariance, _xyahBox);
            this.mean = mc.Item1;
            this.covariance = mc.Item2;

            static_tlwh();
            static_tlbr();

            this.state = TrackState.Tracked;
            this.is_activated = true;

            this.score = new_track.score;
        }

        private void static_tlwh()
        {
            if (this.state == TrackState.New)
            {
                tlwh[0] = _tlwh[0];
                tlwh[1] = _tlwh[1];
                tlwh[2] = _tlwh[2];
                tlwh[3] = _tlwh[3];
                return;
            }

            // mean => [ x_center, y_center, ratio, h, ... ]
            // Calcula valores finales en locales para evitar lecturas/escrituras intermedias al array
            float h = mean[3];
            if (h < Eps) h = Eps;
            float w = mean[2] * h;

            tlwh[0] = mean[0] - w * 0.5f;
            tlwh[1] = mean[1] - h * 0.5f;
            tlwh[2] = w;
            tlwh[3] = h;
        }

        private void static_tlbr()
        {
            tlbr[0] = tlwh[0];
            tlbr[1] = tlwh[1];
            tlbr[2] = tlwh[0] + tlwh[2];
            tlbr[3] = tlwh[1] + tlwh[3];
        }

        /// <summary>
        /// Escribe xyah (center_x, center_y, aspect_ratio, height) directamente en dest.
        /// Cero asignaciones en el heap.
        /// </summary>
        private static void TlwhToXyah(float[] src, DETECTBOX dest)
        {
            float w = src[2];
            float h = src[3];
            if (h < Eps) h = Eps;
            dest[0] = src[0] + w * 0.5f;
            dest[1] = src[1] + h * 0.5f;
            dest[2] = w / h;
            dest[3] = h;
        }

        public float[] to_xyah()
        {
            float w = tlwh[2];
            float h = tlwh[3];
            if (h < Eps) h = Eps;
            return new float[]
            {
                tlwh[0] + w * 0.5f,
                tlwh[1] + h * 0.5f,
                w / h,
                h
            };
        }

        public static float[] tlbr_to_tlwh(float[] tlbr)
        {
            return new float[]
            {
                tlbr[0],
                tlbr[1],
                tlbr[2] - tlbr[0],
                tlbr[3] - tlbr[1]
            };
        }

        public void mark_lost()
        {
            state = TrackState.Lost;
        }

        public void mark_removed()
        {
            state = TrackState.Removed;
        }

        private int next_id()
        {
            _count++;
            return _count;
        }

        public int end_frame()
        {
            return this.frame_id;
        }

        public static void multi_predict(List<STrack> stracks, KalmanFilter kalman_filter)
        {
            for (int i = 0; i < stracks.Count; i++)
            {
                STrack st = stracks[i];
                if (st.state != TrackState.Tracked)
                {
                    st.mean[7] = 0;
                }
                kalman_filter.predict(st.mean, st.covariance);
                st.static_tlwh();
                st.static_tlbr();
            }
        }
    }
}
