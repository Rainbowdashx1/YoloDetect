using System;
using System.Collections.Generic;

namespace ByteTrack
{
    public class KalmanFilter
    {
        // Emulamos el array de chi2inv95.
        public static readonly double[] chi2inv95 = new double[]
        {
            0.0,
            3.8415,
            5.9915,
            7.8147,
            9.4877,
            11.070,
            12.592,
            14.067,
            15.507,
            16.919
        };

        private float[,] _motion_mat;   // 8x8
        private float[,] _update_mat;   // 4x8
        private float _std_weight_position;
        private float _std_weight_velocity;

        public KalmanFilter()
        {
            int ndim = 4;
            float dt = 1f;

            // _motion_mat = Identity(8) pero con dt en [i, i+ndim]
            _motion_mat = MatrixOps.Identity(8);
            for (int i = 0; i < ndim; i++)
            {
                _motion_mat[i, ndim + i] = dt;
            }

            // _update_mat = Identity(4x8) => la parte (4x4) = I, el resto = 0
            _update_mat = new float[4, 8];
            for (int i = 0; i < 4; i++)
            {
                _update_mat[i, i] = 1f;
            }

            _std_weight_position = 1f / 20f;
            _std_weight_velocity = 1f / 160f;
        }

        public KAL_DATA initiate(DETECTBOX measurement)
        {
            // mean_pos = measurement
            // mean_vel = 0
            float[] meanArray = new float[8];
            for (int i = 0; i < 4; i++)
            {
                meanArray[i] = measurement[i];
            }
            // Velocidad en 0
            // (i=4..7) => 0

            KAL_MEAN mean = new KAL_MEAN(meanArray);

            // std array:
            // std(0) = 2 * _std_weight_position * measurement[3]
            float[] stdArr = new float[8];
            stdArr[0] = 2f * _std_weight_position * measurement[3];  // x
            stdArr[1] = 2f * _std_weight_position * measurement[3];  // y
            stdArr[2] = 1e-2f;                                       // ratio
            stdArr[3] = 2f * _std_weight_position * measurement[3];  // h
            stdArr[4] = 10f * _std_weight_velocity * measurement[3];
            stdArr[5] = 10f * _std_weight_velocity * measurement[3];
            stdArr[6] = 1e-5f;
            stdArr[7] = 10f * _std_weight_velocity * measurement[3];

            KAL_MEAN std = new KAL_MEAN(stdArr);
            // "tmp = std.array().square()"
            std.SquareElementsInPlace();

            // Convertimos a una matriz diagonal (8x8)
            KAL_COVA var = MatrixOps.AsDiagonal(std);

            // Retornamos par (mean, var)
            return new KAL_DATA(mean, var);
        }

        public void predict(KAL_MEAN mean, KAL_COVA covariance)
        {
            // std_pos (4)
            float[] std_pos = new float[4];
            std_pos[0] = _std_weight_position * mean[3];
            std_pos[1] = _std_weight_position * mean[3];
            std_pos[2] = 1e-2f;
            std_pos[3] = _std_weight_position * mean[3];

            // std_vel (4)
            float[] std_vel = new float[4];
            std_vel[0] = _std_weight_velocity * mean[3];
            std_vel[1] = _std_weight_velocity * mean[3];
            std_vel[2] = 1e-5f;
            std_vel[3] = _std_weight_velocity * mean[3];

            // Convertimos a un KAL_MEAN para poder usar AsDiagonal
            float[] combinedStdArr = new float[8];
            for (int i = 0; i < 4; i++) combinedStdArr[i] = std_pos[i];
            for (int i = 0; i < 4; i++) combinedStdArr[4 + i] = std_vel[i];

            KAL_MEAN combinedStd = new KAL_MEAN(combinedStdArr);
            combinedStd.SquareElementsInPlace(); // squares each element
            KAL_COVA motionCov = MatrixOps.AsDiagonal(combinedStd);

            // mean1 = this->_motion_mat * mean.transpose();
            // En C++, mean es 1x8 (fila), se transpone -> 8x1
            // => 8x8 * 8x1 => 8x1. Lo guardamos en mean1 (que representamos como 1x8)
            float[] mean1 = MatrixOps.Mul8x8_8x1(_motion_mat, mean);
            // covariance1 = _motion_mat * covariance * _motion_mat^T
            float[,] cov1 = MatrixOps.Mul8x8_8x8(_motion_mat, covariance.Data);
            cov1 = MatrixOps.Mul8x8_8x8(cov1, MatrixOps.Transpose(_motion_mat));

            // covariance1 += motion_cov
            cov1 = MatrixOps.Add8x8(cov1, motionCov);

            // actualizamos en los objetos
            // mean => mean1
            for (int i = 0; i < 8; i++) mean[i] = mean1[i];

            // covariance => cov1
            for (int r = 0; r < 8; r++)
            {
                for (int c = 0; c < 8; c++)
                {
                    covariance[r, c] = cov1[r, c];
                }
            }
        }

        public KAL_HDATA project(KAL_MEAN mean, KAL_COVA covariance)
        {
            // std => 4
            float[] stdArr = new float[4];
            stdArr[0] = _std_weight_position * mean[3];
            stdArr[1] = _std_weight_position * mean[3];
            stdArr[2] = 1e-1f;
            stdArr[3] = _std_weight_position * mean[3];

            KAL_HMEAN mean1 = MatrixOps.Mul4x8_8x1(_update_mat, mean); // 4x1

            // covariance1 = _update_mat * covariance * _update_mat^T
            float[,] cov1 = MatrixOps.Mul4x8_8x8(_update_mat, covariance);
            cov1 = MatrixOps.Mul4x8_8x8(cov1, MatrixOps.Transpose(_update_mat)); // => 4x4

            // diag = std.asDiagonal => 4x4
            float[,] diag = new float[4, 4];
            for (int i = 0; i < 4; i++)
            {
                float val = stdArr[i];
                diag[i, i] = val * val; // square
            }

            // covariance1 += diag
            cov1 = MatrixOps.Add4x4(cov1, diag);

            KAL_HCOVA hcova = new KAL_HCOVA();
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    hcova[r, c] = cov1[r, c];
                }
            }

            // mean1 => KAL_HMEAN
            KAL_HMEAN hmean = new KAL_HMEAN();
            for (int i = 0; i < 4; i++)
            {
                hmean[i] = mean1[i];
            }

            return new KAL_HDATA(hmean, hcova);
        }

        public KAL_DATA update(KAL_MEAN mean, KAL_COVA covariance, DETECTBOX measurement)
        {
            // project => (projected_mean, projected_cov)
            var pa = project(mean, covariance);
            KAL_HMEAN projected_mean = pa.Item1;
            KAL_HCOVA projected_cov = pa.Item2;
            float[,] covArr = MatrixOps.ToArray(covariance);
            // B = (covariance * _update_mat^T).transpose() => 4x8
            float[,] cov_times_updateT = MatrixOps.Mul8x8_8x4(covArr, MatrixOps.Transpose(_update_mat));
            // => 8x4
            // projected_cov => 4x4
            // kalman_gain = projected_cov.llt().solve(B).transpose() => 8x4
            // Emulación "llt" con algo muy simple (Cholesky). Ojo, no robusto.
            float[,] iS = MatrixOps.CholeskyInverse4x4(projected_cov); // inversa de la 4x4
            float[,] Kt = MatrixOps.Mul8x4_4x4(cov_times_updateT, iS);  // => 8x4
            // "kalman_gain" = Kt, pero en la notación original lo transponían. 
            // En la práctica, podemos trabajar con Kt (8x4).

            // innovation = measurement - projected_mean => 1x4
            float[] innovation = new float[4];
            for (int i = 0; i < 4; i++)
                innovation[i] = measurement[i] - projected_mean[i];

            // tmp = innovation (1x4) * Kt^T (4x8) => 1x8
            float[] tmp = MatrixOps.Mul1x4_4x8(innovation, MatrixOps.Transpose(Kt));

            // new_mean = mean + tmp => 1x8
            float[] new_meanArr = new float[8];
            for (int i = 0; i < 8; i++)
                new_meanArr[i] = mean[i] + tmp[i];

            // new_covariance = covariance - kalman_gain * projected_cov * kalman_gain^T
            // => 8x8 - (8x4 * 4x4 * 4x8)
            float[,] kc_pc = MatrixOps.Mul8x4_4x4(Kt, MatrixOps.ToArray(projected_cov)); // => 8x4
            kc_pc = MatrixOps.Mul8x4_4x8(kc_pc, MatrixOps.Transpose(Kt)); // => 8x8
            float[,] new_cov = MatrixOps.Copy8x8(covariance);
            new_cov = MatrixOps.Sub8x8(new_cov, kc_pc);

            // Actualizamos mean y covariance
            KAL_MEAN new_mean = new KAL_MEAN(new_meanArr);
            KAL_COVA new_cova = new KAL_COVA();
            for (int r = 0; r < 8; r++)
            {
                for (int c = 0; c < 8; c++)
                {
                    new_cova[r, c] = new_cov[r, c];
                }
            }

            return new KAL_DATA(new_mean, new_cova);
        }

        /// <summary>
        /// gating_distance => calcula Mahalanobis distance 
        /// </summary>
        public float[] gating_distance(
            KAL_MEAN mean,
            KAL_COVA covariance,
            List<DETECTBOX> measurements,
            bool only_position = false)
        {
            // project => (mean1, cov1)
            var pa = project(mean, covariance);
            KAL_HMEAN mean1 = pa.Item1;
            KAL_HCOVA cov1 = pa.Item2;

            if (only_position)
            {
                // En C++ sale "not implement!"
                // lanzo exeption para indicar que no está implementado
                throw new NotImplementedException("only_position = true no está implementado.");
            }

            // Inversa de S mediante Cholesky
            float[,] invS = MatrixOps.CholeskyInverse4x4(cov1);

            float[] result = new float[measurements.Count];
            for (int i = 0; i < measurements.Count; i++)
            {
                // diff = box - mean1 => 1x4
                float[] diff = new float[4];
                for (int c = 0; c < 4; c++)
                    diff[c] = measurements[i][c] - mean1[c];

                // d^2 = diff * invS * diff^T
                float[] t = MatrixOps.Mul1x4_4x4(diff, invS); // 1x4
                float maha = 0f;
                for (int c = 0; c < 4; c++)
                    maha += t[c] * diff[c];

                result[i] = maha;
            }
            return result;
        }
    }

    // ================================================================
    //  CLASE DE OPERACIONES AUXILIARES (MATRIXOPS)
    //  Emula las multiplicaciones y sumas usadas en el código
    // ================================================================
    internal static class MatrixOps
    {
        // ------------------------------------------------------
        //  Identidad NxN
        // ------------------------------------------------------
        public static float[,] Identity(int n)
        {
            float[,] mat = new float[n, n];
            for (int i = 0; i < n; i++)
                mat[i, i] = 1f;
            return mat;
        }

        // ------------------------------------------------------
        //  Transpose (NxM) => (MxN)
        // ------------------------------------------------------
        public static float[,] Transpose(float[,] A)
        {
            int rows = A.GetLength(0);
            int cols = A.GetLength(1);
            float[,] At = new float[cols, rows];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    At[c, r] = A[r, c];
                }
            }
            return At;
        }

        // ------------------------------------------------------
        //  AsDiagonal => construye una 8x8 diagonal con KAL_MEAN(8)
        // ------------------------------------------------------
        public static KAL_COVA AsDiagonal(KAL_MEAN vec)
        {
            KAL_COVA mat = new KAL_COVA();
            for (int i = 0; i < 8; i++)
            {
                mat[i, i] = vec[i];
            }
            return mat;
        }

        // ------------------------------------------------------
        //  Multiplicaciones específicas:
        //  - 8x8 * 8x1 => 8x1 (lo guardamos en un float[8])
        // ------------------------------------------------------
        public static float[] Mul8x8_8x1(float[,] A, KAL_MEAN x)
        {
            float[] res = new float[8];
            for (int r = 0; r < 8; r++)
            {
                float sum = 0f;
                for (int k = 0; k < 8; k++)
                {
                    sum += A[r, k] * x[k];
                }
                res[r] = sum;
            }
            return res;
        }

        // ------------------------------------------------------
        //  8x8 * 8x8 => 8x8
        // ------------------------------------------------------
        public static float[,] Mul8x8_8x8(float[,] A, float[,] B)
        {
            // Ambos son matrices 8x8
            float[,] res = new float[8, 8];
            for (int r = 0; r < 8; r++)
            {
                for (int c = 0; c < 8; c++)
                {
                    float sum = 0f;
                    for (int k = 0; k < 8; k++)
                    {
                        sum += A[r, k] * B[k, c];
                    }
                    res[r, c] = sum;
                }
            }
            return res;
        }

        // ------------------------------------------------------
        //  8x8 * 8x4 => 8x4
        // ------------------------------------------------------
        public static float[,] Mul8x8_8x4(float[,] A, float[,] B)
        {
            float[,] res = new float[8, 4];
            for (int r = 0; r < 8; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    float sum = 0f;
                    for (int k = 0; k < 8; k++)
                    {
                        sum += A[r, k] * B[k, c];
                    }
                    res[r, c] = sum;
                }
            }
            return res;
        }

        // ------------------------------------------------------
        //  8x4 * 4x4 => 8x4
        // ------------------------------------------------------
        public static float[,] Mul8x4_4x4(float[,] A, float[,] B)
        {
            float[,] res = new float[8, 4];
            for (int r = 0; r < 8; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    float sum = 0f;
                    for (int k = 0; k < 4; k++)
                    {
                        sum += A[r, k] * B[k, c];
                    }
                    res[r, c] = sum;
                }
            }
            return res;
        }

        // ------------------------------------------------------
        //  8x4 * 4x8 => 8x8
        // ------------------------------------------------------
        public static float[,] Mul8x4_4x8(float[,] A, float[,] B)
        {
            float[,] res = new float[8, 8];
            for (int r = 0; r < 8; r++)
            {
                for (int c = 0; c < 8; c++)
                {
                    float sum = 0f;
                    for (int k = 0; k < 4; k++)
                    {
                        sum += A[r, k] * B[k, c];
                    }
                    res[r, c] = sum;
                }
            }
            return res;
        }

        // ------------------------------------------------------
        //  4x8 * 8x8 => 4x8
        // ------------------------------------------------------
        public static float[,] Mul4x8_8x8(float[,] A, KAL_COVA B)
        {
            float[,] res = new float[4, 8];
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 8; c++)
                {
                    float sum = 0f;
                    for (int k = 0; k < 8; k++)
                    {
                        sum += A[r, k] * B[k, c];
                    }
                    res[r, c] = sum;
                }
            }
            return res;
        }

        // ------------------------------------------------------
        //  4x8 * 8x8 => 4x8
        //  (versión float[,], KAL_COVA => float[,])
        // ------------------------------------------------------
        public static float[,] Mul4x8_8x8(float[,] A, float[,] B)
        {
            int ARows = 4, ACols = 8, BRows = 8, BCols = B.GetLength(1);
            float[,] res = new float[4, BCols];
            for (int r = 0; r < ARows; r++)
            {
                for (int c = 0; c < BCols; c++)
                {
                    float sum = 0f;
                    for (int k = 0; k < ACols; k++)
                    {
                        sum += A[r, k] * B[k, c];
                    }
                    res[r, c] = sum;
                }
            }
            return res;
        }

        // ------------------------------------------------------
        //  4x8 * 8x1 => 4x1 (1D float[4])
        // ------------------------------------------------------
        public static KAL_HMEAN Mul4x8_8x1(float[,] A, KAL_MEAN x)
        {
            float[] res = new float[4];
            for (int r = 0; r < 4; r++)
            {
                float sum = 0f;
                for (int k = 0; k < 8; k++)
                {
                    sum += A[r, k] * x[k];
                }
                res[r] = sum;
            }
            KAL_HMEAN outMean = new KAL_HMEAN();
            for (int i = 0; i < 4; i++)
                outMean[i] = res[i];
            return outMean;
        }

        // ------------------------------------------------------
        //  Sumas y restas:
        // ------------------------------------------------------
        public static float[,] Add8x8(float[,] A, KAL_COVA B)
        {
            float[,] res = new float[8, 8];
            for (int r = 0; r < 8; r++)
            {
                for (int c = 0; c < 8; c++)
                {
                    res[r, c] = A[r, c] + B[r, c];
                }
            }
            return res;
        }

        public static float[,] Add8x8(float[,] A, float[,] B)
        {
            float[,] res = new float[8, 8];
            for (int r = 0; r < 8; r++)
            {
                for (int c = 0; c < 8; c++)
                {
                    res[r, c] = A[r, c] + B[r, c];
                }
            }
            return res;
        }

        public static float[,] Sub8x8(float[,] A, float[,] B)
        {
            float[,] res = new float[8, 8];
            for (int r = 0; r < 8; r++)
            {
                for (int c = 0; c < 8; c++)
                {
                    res[r, c] = A[r, c] - B[r, c];
                }
            }
            return res;
        }

        public static float[,] Add4x4(float[,] A, float[,] B)
        {
            float[,] res = new float[4, 4];
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    res[r, c] = A[r, c] + B[r, c];
                }
            }
            return res;
        }

        // ------------------------------------------------------
        //  Copias
        // ------------------------------------------------------
        public static float[,] Copy8x8(KAL_COVA cov)
        {
            float[,] res = new float[8, 8];
            for (int r = 0; r < 8; r++)
            {
                for (int c = 0; c < 8; c++)
                {
                    res[r, c] = cov[r, c];
                }
            }
            return res;
        }

        public static float[,] ToArray(KAL_HCOVA cov)
        {
            float[,] res = new float[4, 4];
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    res[r, c] = cov[r, c];
                }
            }
            return res;
        }

        // ------------------------------------------------------
        //  Multiplicaciones para 1x4 * 4x8 => 1x8
        // ------------------------------------------------------
        public static float[] Mul1x4_4x8(float[] row, float[,] mat)
        {
            float[] res = new float[8];
            for (int c = 0; c < 8; c++)
            {
                float sum = 0f;
                for (int k = 0; k < 4; k++)
                {
                    sum += row[k] * mat[k, c];
                }
                res[c] = sum;
            }
            return res;
        }

        // ------------------------------------------------------
        //  1x4 * 4x4 => 1x4
        // ------------------------------------------------------
        public static float[] Mul1x4_4x4(float[] row, float[,] mat)
        {
            float[] res = new float[4];
            for (int c = 0; c < 4; c++)
            {
                float sum = 0f;
                for (int k = 0; k < 4; k++)
                {
                    sum += row[k] * mat[k, c];
                }
                res[c] = sum;
            }
            return res;
        }

        // ------------------------------------------------------
        //  Cholesky Inverse de 4x4 (MUY simplificado)
        // ------------------------------------------------------
        public static float[,] CholeskyInverse4x4(KAL_HCOVA cov)
        {
            float[,] c = ToArray(cov);
            return CholeskyInverse4x4(c);
        }

        public static float[,] CholeskyInverse4x4(float[,] c)
        {
            // Factorización de Cholesky muy simplificada
            // 1) Hacemos L = Cholesky(c)
            // 2) invertimos L
            // 3) iS = (L^-1)^T * (L^-1)

            float[,] L = CholeskyDecompose4x4(c);
            float[,] Linv = InvertLowerTri4x4(L);
            float[,] iS = Mul4x4_4x4(Transpose(Linv), Linv);
            return iS;
        }

        // Multiplicación 4x4 * 4x4
        public static float[,] Mul4x4_4x4(float[,] A, float[,] B)
        {
            float[,] res = new float[4, 4];
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    float sum = 0f;
                    for (int k = 0; k < 4; k++)
                    {
                        sum += A[r, k] * B[k, c];
                    }
                    res[r, c] = sum;
                }
            }
            return res;
        }

        // Cholesky de 4x4 (solo parte inferior L)
        public static float[,] CholeskyDecompose4x4(float[,] A)
        {
            float[,] L = new float[4, 4];
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    float sum = 0f;
                    for (int k = 0; k < j; k++)
                        sum += L[i, k] * L[j, k];

                    if (i == j)
                    {
                        float val = A[i, i] - sum;
                        if (val <= 0)
                        {
                            // Error: la matriz no es SPD
                            val = 1e-6f;
                        }
                        L[i, j] = (float)Math.Sqrt(val);
                    }
                    else
                    {
                        L[i, j] = (1.0f / L[j, j] * (A[i, j] - sum));
                    }
                }
            }
            return L;
        }

        // Invertir L (4x4 lower-tri)
        public static float[,] InvertLowerTri4x4(float[,] L)
        {
            // L^-1
            float[,] inv = new float[4, 4];

            // Inversión del triángulo inferior
            for (int i = 0; i < 4; i++)
            {
                // diagonal
                inv[i, i] = 1f / L[i, i];
                // resto
                for (int j = 0; j < i; j++)
                {
                    float sum = 0f;
                    for (int k = j; k < i; k++)
                    {
                        sum -= L[i, k] * inv[k, j];
                    }
                    inv[i, j] = sum / L[i, i];
                }
            }
            return inv;
        }
        public static float[,] ToArray(KAL_COVA cov)
        {
            float[,] arr = new float[8, 8];
            for (int i = 0; i < 8; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    arr[i, j] = cov[i, j];
                }
            }
            return arr;
        }
    }
}
