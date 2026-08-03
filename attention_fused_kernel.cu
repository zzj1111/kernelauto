
    #include <torch/extension.h>
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    #include <algorithm>

    #define BLOCK_M 64
    #define BLOCK_N 64

    __global__ void fused_attn_norm_forward(
        const float* Q,        // [L, N, D]
        float* O,              // [L, N, D] output
        const float* gamma,    // [D]
        const float* beta,     // [D]
        int L, int D,
        int BLOCK_M, int BLOCK_N,
        int num_blocks_m, int num_blocks_n,
        float scale,
        float inf,
        int inp_dtype // 0: fp16, 1: bf16
    ) {
        int bm = blockIdx.m;
        int bn = blockIdx.n;
        int bt = blockIdx.t;  // batch tile id

        int pid = bn * num_blocks_m + bm; // combine m and n into one pid
        int num_pid = num_blocks_m * num_blocks_n;
        int b = bt % num_blocks_n; // since num_blocks_n == N (batch), b == bt

        // compute indices
        int m0 = bm * BLOCK_M;
        int n0 = bn * BLOCK_N;
        int i_start = m0;
        int j_start = n0;

        // create pointers for query tile Q[i, b, :]
        const float* Q_ptr = Q + b * D + i_start * D + n0;
        // value tile V[j, b, :]
        const float* V_ptr = Q + b * D + j_start * D + n0;

        // work buffers
        float S[BLOCK_M][BLOCK_N];  // scaled dot-product matrix (log-space row-wise)
        float maxv[BLOCK_M];        // row-wise max for stability
        float sumv[BLOCK_M];        // row-wise exp sums
        float out_tile[BLOCK_M][BLOCK_N]; // final output tile after V multiply

        // sentinel for masking
        float neg_inf = inf;

        // Loop over k dimension (embed dim tiles)
        for (int kk = 0; kk < D; kk += BLOCK_N) {
            // load Q tile (M x N)
            for (int ii = 0; ii < BLOCK_M; ++ii) {
                float* S_row = S[ii];
                for (int jj = 0; jj < BLOCK_N; ++jj) {
                    int q_idx = i_start + ii;
                    int k_idx = j_start + jj;
                    float q_val = (inp_dtype == 0) ? __half2float(*((const half*)(Q_ptr + ii * D + (n0 + jj)))) : __bfloat162float(*((const bfloat16*)(Q_ptr + ii * D + (n0 + jj))));
                    float k_val = (inp_dtype == 0) ? __half2float(*((const half*)(Q_ptr + (j_start + jj) * D + n0 + kk + jj))) : __bfloat162float(*((const bfloat16*)(Q_ptr + (j_start + jj) * D + (n0 + kk + jj))));
                    S_row[jj] = q_val * k_val * scale;
                    S_row[jj] = fmax(S_row[jj], neg_inf);
                }
            }

            // softmax: log-space per row
            for (int ii = 0; ii < BLOCK_M; ++ii) {
                maxv[ii] = -1e30;
                for (int jj = 0; jj < BLOCK_N; ++jj) {
                    maxv[ii] = fmax(maxv[ii], S[ii][jj]);
                }
                float sum_row = 0.0;
                for (int jj = 0; jj < BLOCK_N; ++jj) {
                    sum_row += exp(S[ii][jj] - maxv[ii]);
                }
                sumv[ii] = sum_row;
            }
            // load V tile (N x N) once per kk
            for (int ii = 0; ii < BLOCK_M; ++ii) {
                for (int jj = 0; jj < BLOCK_N; ++jj) {
                    int j_idx = j_start + jj;
                    float v_val = (inp_dtype == 0) ? __half2float(*((const half*)(V_ptr + jj * D + (n0 + kk + jj)))) : __bfloat162float(*((const bfloat16*)(V_ptr + jj * D + (n0 + kk + jj))));
                    out_tile[ii][jj] = (S[ii][jj] - maxv[ii]) * v_val / sumv[ii];
                }
            }
        }

        // Add residual x: out = out + x
        // x equals Q for this fused path; we need Q[i, b, :] again
        const float* X_ptr = Q + b * D + i_start * D + n0;
        for (int ii = 0; ii < BLOCK_M; ++ii) {
            for (int jj = 0; jj < BLOCK_N; ++jj) {
                float x_val = (inp_dtype == 0) ? __half2float(*((const half*)(X_ptr + ii * D + (n0 + jj)))) : __bfloat162float(*((const bfloat16*)(X_ptr + ii * D + (n0 + jj))));
                out_tile[ii][jj] += x_val;
            }
        }

        // LayerNorm: y = gamma * x / sqrt(var + eps) + beta
        // We can't use LN here; instead, we'll fuse it in forward using our own LN kernel.
        // For now, we apply LN with eps=1e-5 and provided gamma/beta.
        // Compute LN stats over D (embed dim)
        float mean[BLOCK_M];
        float var[BLOCK_M];
        for (int ii = 0; ii < BLOCK_M; ++ii) {
            mean[ii] = 0.0;
            for (int jj = 0; jj < BLOCK_N; ++jj) {
                mean[ii] += out_tile[ii][jj];
            }
            mean[ii] /= BLOCK_N;
            var[ii] = 0.0;
            for (int jj = 0; jj < BLOCK_N; ++jj) {
                var[ii] += (out_tile[ii][jj] - mean[ii]) * (out_tile[ii][jj] - mean[ii]);
            }
            var[ii] /= BLOCK_N;
        }

        // Load gamma/beta
        const float* gamma = gamma;
        const float* beta = beta;
        for (int ii = 0; ii < BLOCK_M; ++ii) {
            for (int jj = 0; jj < BLOCK_N; ++jj) {
                float g = (gamma[jj] == 0.0) ? 1.0 : gamma[jj];
                float b_term = beta[jj] ? beta[jj] : 0.0;
                out_tile[ii][jj] = out_tile[ii][jj] * g / sqrt(var[ii] + 1e-5) + b_term;
            }
        }

        // Store output
        float* O_ptr = O + b * D + i_start * D + n0;
        for (int ii = 0; ii < BLOCK_M; ++ii) {
            for (int jj = 0; jj < BLOCK_N; ++jj) {
                if (inp_dtype == 0) {
                    *(((half*)O_ptr + ii * D + (n0 + jj))) = __float2half(out_tile[ii][jj]);
                } else {
                    *(((bfloat16*)O_ptr + ii * D + (n0 + jj))) = __float2bfloat16(out_tile[ii][jj]);
                }
            }
        }
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("fused_attn_norm_forward", &fused_attn_norm_forward, "Fused attention + layernorm forward kernel");
    }
    