
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    #include <assert.h>

    #define MAX(a, b) ((a) > (b) ? (a) : (b))

    __global__ void maxpool2d_kernel(
        const float* x, float* y,
        int B, int C, int H, int W,
        int H_OUT, int W_OUT,
        int KH, int KW, int SH, int SW,
        int PH, int PW, int DH, int DW,
        int NC,
        int BLOCK_W, int BLOCK_W_OUT, int BLOCK_H_WIN
    ) {
        // Each block handles one (n, c) pair and a tile across W_OUT
        int nc = blockIdx.x;
        int w_out_block = blockIdx.y;

        int n = nc / C;
        int c = nc % C;

        // Compute tile offsets
        int w_out_start = w_out_block * BLOCK_W_OUT;
        int w_out_idx = w_out_start + threadIdx.x;
        int h_out_idx = 0;  // scalar loop across output height

        // Base pointers (without batch/channel offsets)
        const float* x_base = x;
        float* y_base = y;
        // We need to compute base pointers for (n, c) later

        // Determine how many blocks in the height dimension (usually 1 for large H_OUT)
        int num_h_blocks = (H_OUT + BLOCK_H_WIN - 1) / BLOCK_H_WIN;
        for (int hb = 0; hb < num_h_blocks; ++hb) {
            // Decode h_out for this hb
            // h_out = hb * BLOCK_H_WIN + h_local where h_local in [0, BLOCK_H_WIN)
            for (int h_local = 0; h_local < BLOCK_H_WIN; ++h_local) {
                int h_out = hb * BLOCK_H_WIN + h_local;
                if (h_out < H_OUT) {
                    // Compute input top-left for this output position
                    int h_in_top = h_out * SH - PH;
                    int w_in_left = w_out_idx * SW - PW;

                    // Base pointers for (n, c)
                    x_base = &x[n * C * H * W + c * H * W + h_in_top * W + w_in_left];
                    y_base = &y[n * C * H_OUT * W_OUT + c * H_OUT * W_OUT + h_out * W_OUT + w_out_idx];

                    // Initialize max value to -inf
                    float max_val = -1.0e30f;

                    // Iterate over pooling window rows
                    for (int kh = 0; kh < KH; ++kh) {
                        int h_in = h_in_top + kh * DH;
                        if (h_in < 0 || h_in >= H) {
                            // skip invalid rows (zero-padding)
                            continue;
                        }

                        // Vectorized row load across BLOCK_W outputs
                        for (int j = 0; j < BLOCK_W; ++j) {
                            int w_out_inner = j;  // scalar per outer j
                            int w_in = w_in_left + w_out_inner * SW + j * DW;
                            if (w_in < 0 || w_in >= W) {
                                // out-of-bounds -> 0 (zero-padding)
                                // Accumulate as -inf to keep correctness
                                max_val = MAX(max_val, -1.0e30f);
                            } else {
                                max_val = MAX(max_val, x_base[w_in * BLOCK_W + w_out_inner]);
                            }
                        }
                    }

                    // Store result
                    y_base[0] = max_val;
                }
            }
        }
    }
    