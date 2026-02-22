using System;
using System.Collections.Generic;
using System.Text;

namespace ByteTrack
{
    // ================================================================
    //  Emulación de tipados de Eigen usados en C++
    // ================================================================

    /// <summary>
    /// DETECTBOX (1 x 4): vector fila de 4 elementos
    /// </summary>
    public class DETECTBOX
    {
        private float[] data = new float[4];

        public DETECTBOX() { }

        public DETECTBOX(float[] values)
        {
            if (values.Length != 4)
                throw new ArgumentException("DETECTBOX necesita 4 elementos");
            Array.Copy(values, data, 4);
        }

        public float this[int i]
        {
            get => data[i];
            set => data[i] = value;
        }

        // Operador para restar DETECTBOX - DETECTBOX
        public static DETECTBOX operator -(DETECTBOX box1, DETECTBOX box2)
        {
            var result = new DETECTBOX();
            for (int i = 0; i < 4; i++)
                result[i] = box1[i] - box2[i];
            return result;
        }

        // Convierte un DETECTBOX en string (para debug).
        public override string ToString()
        {
            return $"[{data[0]}, {data[1]}, {data[2]}, {data[3]}]";
        }
    }

    /// <summary>
    /// DETECTBOXSS (N x 4): matriz con N filas y 4 columnas
    /// </summary>
    public class DETECTBOXSS
    {
        private float[,] data;
        public int Rows { get; private set; }
        public int Cols { get; private set; }

        public DETECTBOXSS(int rows)
        {
            Rows = rows;
            Cols = 4;
            data = new float[rows, 4];
        }

        public float this[int r, int c]
        {
            get => data[r, c];
            set => data[r, c] = value;
        }

        /// <summary>
        /// Asigna una fila (row) usando un DETECTBOX (1x4)
        /// </summary>
        public void SetRow(int row, DETECTBOX box)
        {
            for (int i = 0; i < 4; i++)
                data[row, i] = box[i];
        }
    }

    /// <summary>
    /// FEATURE (1 x 128)
    /// </summary>
    public class FEATURE
    {
        private float[] data = new float[128];

        public float this[int i]
        {
            get => data[i];
            set => data[i] = value;
        }
    }

    /// <summary>
    /// FEATURESS (N x 128)
    /// </summary>
    public class FEATURESS
    {
        private float[,] data;
        public int Rows { get; private set; }
        public int Cols { get; private set; }

        public FEATURESS(int rows)
        {
            Rows = rows;
            Cols = 128;
            data = new float[rows, 128];
        }

        public float this[int r, int c]
        {
            get => data[r, c];
            set => data[r, c] = value;
        }
    }

    // ================================================================
    //  Tipos de Kalman
    // ================================================================
    /// <summary>
    /// KAL_MEAN (1 x 8)
    /// </summary>
    public class KAL_MEAN
    {
        private float[] data = new float[8];

        public float this[int i]
        {
            get => data[i];
            set => data[i] = value;
        }

        public KAL_MEAN() { }

        public KAL_MEAN(float[] values)
        {
            if (values.Length != 8)
                throw new ArgumentException("KAL_MEAN necesita 8 elementos");
            Array.Copy(values, data, 8);
        }

        /// <summary>
        /// Aplica square a cada elemento (emulando "array().square()")
        /// </summary>
        public void SquareElementsInPlace()
        {
            for (int i = 0; i < data.Length; i++)
                data[i] = data[i] * data[i];
        }

        // Para debug
        public override string ToString()
        {
            return $"[ {string.Join(", ", data)} ]";
        }
    }

    /// <summary>
    /// KAL_COVA (8 x 8)
    /// </summary>
    public class KAL_COVA
    {
        private float[,] data = new float[8, 8];

        public float this[int r, int c]
        {
            get => data[r, c];
            set => data[r, c] = value;
        }

        public float[,] Data { get => data; }

        public KAL_COVA() { }

        /// <summary>
        /// Limpia la matriz con ceros
        /// </summary>
        public void Clear()
        {
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    data[r, c] = 0f;
        }

        // Para debug
        public override string ToString()
        {
            string s = "";
            for (int r = 0; r < 8; r++)
            {
                s += "[ ";
                for (int c = 0; c < 8; c++)
                {
                    s += data[r, c].ToString("F3") + (c < 7 ? ", " : "");
                }
                s += " ]\n";
            }
            return s;
        }
    }

    /// <summary>
    /// KAL_HMEAN (1 x 4)
    /// </summary>
    public class KAL_HMEAN
    {
        private float[] data = new float[4];

        public float this[int i]
        {
            get => data[i];
            set => data[i] = value;
        }
    }

    /// <summary>
    /// KAL_HCOVA (4 x 4)
    /// </summary>
    public class KAL_HCOVA
    {
        private float[,] data = new float[4, 4];

        public float this[int r, int c]
        {
            get => data[r, c];
            set => data[r, c] = value;
        }
    }

    /// <summary>
    /// Equivalente a std::pair<KAL_MEAN, KAL_COVA>
    /// </summary>
    public struct KAL_DATA
    {
        public KAL_MEAN Item1;
        public KAL_COVA Item2;

        public KAL_DATA(KAL_MEAN m, KAL_COVA c)
        {
            Item1 = m;
            Item2 = c;
        }
    }

    /// <summary>
    /// Equivalente a std::pair<KAL_HMEAN, KAL_HCOVA>
    /// </summary>
    public struct KAL_HDATA
    {
        public KAL_HMEAN Item1;
        public KAL_HCOVA Item2;

        public KAL_HDATA(KAL_HMEAN hm, KAL_HCOVA hc)
        {
            Item1 = hm;
            Item2 = hc;
        }
    }

    // ================================================================
    //  Otros
    // ================================================================
    /// <summary>
    /// RESULT_DATA = std::pair<int, DETECTBOX>
    /// </summary>
    public struct RESULT_DATA
    {
        public int Item1;      // id
        public DETECTBOX Item2; // box

        public RESULT_DATA(int id, DETECTBOX box)
        {
            Item1 = id;
            Item2 = box;
        }
    }

    /// <summary>
    /// TRACKER_DATA = std::pair<int, FEATURESS>
    /// </summary>
    public struct TRACKER_DATA
    {
        public int Item1;
        public FEATURESS Item2;

        public TRACKER_DATA(int id, FEATURESS feats)
        {
            Item1 = id;
            Item2 = feats;
        }
    }

    /// <summary>
    /// MATCH_DATA = std::pair<int, int>
    /// </summary>
    public struct MATCH_DATA
    {
        public int Item1;
        public int Item2;

        public MATCH_DATA(int t, int d)
        {
            Item1 = t;
            Item2 = d;
        }
    }

    /// <summary>
    /// Estructura análoga a:
    /// struct t {
    ///   std::vector<MATCH_DATA> matches;
    ///   std::vector<int> unmatched_tracks;
    ///   std::vector<int> unmatched_detections;
    /// }
    /// </summary>
    public class TRACHER_MATCHD
    {
        public List<MATCH_DATA> matches;
        public List<int> unmatched_tracks;
        public List<int> unmatched_detections;

        public TRACHER_MATCHD()
        {
            matches = new List<MATCH_DATA>();
            unmatched_tracks = new List<int>();
            unmatched_detections = new List<int>();
        }
    }

    /// <summary>
    /// DYNAMICM = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>
    /// Aquí lo representamos con float[,] o una clase.
    /// </summary>
    public class DYNAMICM
    {
        private float[,] data;
        public int Rows { get; private set; }
        public int Cols { get; private set; }

        public DYNAMICM(int rows, int cols)
        {
            Rows = rows;
            Cols = cols;
            data = new float[rows, cols];
        }

        public float this[int r, int c]
        {
            get => data[r, c];
            set => data[r, c] = value;
        }
    }
}
