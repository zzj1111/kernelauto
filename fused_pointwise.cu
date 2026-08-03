
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_pointwise_kernel(
    const float* out_ptr,        // conv output
    const float* s_ptr,          // scaling_factor [C]
    const float* b_ptr,          // bias [C]
    float* y_ptr,                // output
    int N, int C, int D, int H, int W,
    int stride_n, int stride_c, int stride_d, int stride_h, int stride_w,
    int s_stride_c, int b_stride_c,
    int total_elems
) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < total_elems) {
        int w = idx % W;
        int tmp = idx / W;
        int h = tmp % H;
        tmp = tmp / H;
        int d = tmp % D;
        tmp = tmp / D;
        int c = tmp % C;
        tmp = tmp / C;
        int n = tmp % N;

        // Compute base address for (n, c, d, h, w)
        int base = n * stride_n + c * stride_c + d * stride_d + h * stride_h + w * stride_w;

        // Load z = conv(n,c,d,h,w)
        float z = out_ptr[base];

        // Load s and b for channel c
        float s_c = s_ptr[c * s_stride_c];
        float b_c = b_ptr[c * b_stride_c];

        // Upcast to FP32 for math
        float z32 = z;
        float s_c32 = s_c;
        float b_c32 = b_c;

        // Compute u = tanh(z * s) * b, then y = sigmoid(u)
        float u = tanh(z32 * s_c32) * b_c32;
        float y = 1.0f / (1.0f + exp(-u));

        // Store back (still FP32)
        y_ptr[base] = y;
    }
}
