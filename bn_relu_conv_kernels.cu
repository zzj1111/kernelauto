
#include <torch/extension.h>
#include <cuda_runtime.h>

// Common constants and helpers
constexpr int K3 = 3;
constexpr int K1 = 1;
constexpr int MAX_C_IN = 1024;  // upper bound for input channels
constexpr int MAX_C_OUT = 4096; // upper bound for output channels

// Helper: clamp to MAX
#define clamp(a, low, high) ((a) > (high) ? (high) : ((a) < (low) ? (low) : (a)))

// K3 3x3 BN+ReLU+Conv fused kernel
// X: input, W: weight, BN: running_mean, running_var, gamma, beta, Y: output
__global__ void bn_relu_conv3x3_kernel(
    const float* X, const float* W, const float* BN_mean, const float* BN_var,
    const float* BN_weight, const float* BN_bias,
    float* Y,
    int B, int C_IN, int C_OUT, int H, int W,
    int sXn, int sXc, int sXh, int sXw,
    int sWco, int sWci, int sWk, int sWk2,
    int sYn, int sYc, int sYh, int sYw,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int pooling_h, int pooling_w,
    int eps,
    bool training,  // ignored in inference path, but kept for interface
    bool has_bias_bn
) {
    int co = blockIdx.x;  // output channel index
    int bc = blockIdx.y;  // batch-channel index for implicit concat
    int tile_w = clamp(blockIdx.z, 0, (W + 31) / 32 - 1);
    int oh = tile_w * 32 + threadIdx.x;  // output height index in the current tile

    // Guard
    if (co >= C_OUT || bc >= B * MAX_C_IN) return;

    // Decode batch and input-channel indices
    int b = bc / MAX_C_IN;
    int ci = bc % MAX_C_IN;
    int oh_valid = (oh < H) && (stride_h == 1 && stride_w == 1);

    // Output pointer
    float* y_ptr = Y + b * sYn + co * sYc + oh * sYh + (0) * sYw;

    // Accumulator
    float acc = 0.0f;

    // Read BN scale/bias if present
    float scale_bn = 1.0f;
    float shift_bn = 0.0f;
    if (has_bias_bn) {
        scale_bn = BN_weight[co];
        shift_bn = BN_bias[co];
    }

    // Loop over input channels and 3x3 neighborhood
    int start_ci = 0;
    while (start_ci < C_IN) {
        // Note: in inference, BN is done per channel using running stats
        float bn_mean_ci = BN_mean[ci];
        float bn_var_ci  = BN_var[ci];
        float inv_std_ci = 1.0f / sqrtf(bn_var_ci + eps);
        float scale_ci = scale_bn * inv_std_ci;
        float shift_ci = shift_bn - bn_mean_ci * scale_ci;

        int ci_block = 0;
        while (ci_block < 1) {  // single block over input channels
            int ci_off = start_ci + ci_block;
            if (ci_off >= C_IN) break;

            // Load weight vector for this (co, ci)
            // W layout: [C_OUT, C_IN, K, K], with K=3
            float w0 = W[co * sWco + ci_off * sWci + 0 * sWk + 0 * sWk2];
            float w1 = W[co * sWco + ci_off * sWci + 1 * sWk + 0 * sWk2];
            float w2 = W[co * sWco + ci_off * sWci + 2 * sWk + 0 * sWk2];
            float w3 = W[co * sWco + ci_off * sWci + 0 * sWk + 1 * sWk2];
            float w4 = W[co * sWco + ci_off * sWci + 1 * sWk + 1 * sWk2];
            float w5 = W[co * sWco + ci_off * sWci + 2 * sWk + 1 * sWk2];
            float w6 = W[co * sWco + ci_off * sWci + 0 * sWk + 2 * sWk2];
            float w7 = W[co * sWco + ci_off * sWci + 1 * sWk + 2 * sWk2];
            float w8 = W[co * sWco + ci_off * sWci + 2 * sWk + 2 * sWk2];

            // Compute 3x3 sum with bias applied; y_ptr points to output location (oh, ow)
            // We accumulate for one (oh, ow) position
            // Since oh is valid only when stride=1, we can assert stride=1.
            // Horizontal positions: ow = tile_w * 32 + txx
            // Vertical positions: oh - pad_h + ky
            int txx = 0;
            while (txx < 32) {
                int ow = tile_w * 32 + txx;
                int ow_valid = (ow < W) && (stride_w == 1 && stride_h == 1);
                if (oh_valid && ow_valid) {
                    int ih = oh - pad_h;
                    int iw = ow - pad_w;

                    // Clamp input indices
                    ih = clamp(ih, 0, H - 1);
                    iw = clamp(iw, 0, W - 1);

                    // Load x; X layout: [B, C_IN, H, W]
                    float x00 = (ci < C_IN ? X[b * sXn + ci * sXc + ih * sXh + (iw - 1) * sXw] : 0.0f);
                    float x01 = (ci < C_IN ? X[b * sXn + ci * sXc + ih * sXh + iw * sXw] : 0.0f);
                    float x02 = (ci < C_IN ? X[b * sXn + ci * sXc + ih * sXh + (iw + 1) * sXw] : 0.0f);
                    float x10 = (ci < C_IN ? X[b * sXn + ci * sXc + (ih - 1) * sXh + (iw - 1) * sXw] : 0.0f);
                    float x11 = (ci < C_IN ? X[b * sXn + ci * sXc + (ih - 1) * sXh + iw * sXw] : 0.0f);
                    float x12 = (ci < C_IN ? X[b * sXn + ci * sXc + (ih - 1) * sXh + (iw + 1) * sXw] : 0.0f);
                    float x20 = (ci < C_IN ? X[b * sXn + ci * sXc + ih * sXh + (iw - 1) * sXw] : 0.0f);
                    float x21 = (ci < C_IN ? X[b * sXn + ci * sXc + ih * sXh + iw * sXw] : 0.0f);
                    float x22 = (ci < C_IN ? X[b * sXn + ci * sXc + ih * sXh + (iw + 1) * sXw] : 0.0f);

                    // Apply BN affine
                    float v00 = maxf(0.0f, x00 * scale_ci + shift_ci);
                    float v01 = maxf(0.0f, x01 * scale_ci + shift_ci);
                    float v02 = maxf(0.0f, x02 * scale_ci + shift_ci);
                    float v10 = maxf(0.0f, x10 * scale_ci + shift_ci);
                    float v11 = maxf(0.0f, x11 * scale_ci + shift_ci);
                    float v12 = maxf(0.0f, x12 * scale_ci + shift_ci);
                    float v20 = maxf(0.0f, x20 * scale_ci + shift_ci);
                    float v21 = maxf(0.0f, x21 * scale_ci + shift_ci);
                    float v22 = maxf(0.0f, x22 * scale_ci + shift_ci);

                    // Weighted sum
                    acc += w0 * v00 + w1 * v01 + w2 * v02 +
                           w3 * v10 + w4 * v11 + w5 * v12 +
                           w6 * v20 + w7 * v21 + w8 * v22;
                }
                txx++;
            }
            start_ci += 1;
        }
    }

    // Store result
    if (oh_valid) {
        *y_ptr = acc;
    }
}

// K1 1x1 BN+ReLU+Conv fused kernel (stride=1)
__global__ void bn_relu_conv1x1_kernel(
    const float* X, const float* W, const float* BN_mean, const float* BN_var,
    const float* BN_weight, const float* BN_bias,
    float* Y,
    int B, int C_IN, int C_OUT, int H, int W,
    int sXn, int sXc, int sXh, int sXw,
    int sWco, int sWci, int sWk, int sWk2,
    int sYn, int sYc, int sYh, int sYw,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int pooling_h, int pooling_w,
    int eps,
    bool training,
    bool has_bias_bn
) {
    int co = blockIdx.x;
    int bc = blockIdx.y;
    int oh = tile_w * 32 + threadIdx.x;

    if (co >= C_OUT || bc >= B * MAX_C_IN) return;

    int b = bc / MAX_C_IN;
    int ci = bc % MAX_C_IN;

    float* y_ptr = Y + b * sYn + co * sYc + oh * sYh + (0) * sYw;

    float scale_bn = 1.0f;
    float shift_bn = 0.0f;
    if (has_bias_bn) {
        scale_bn = BN_weight[co];
        shift_bn = BN_bias[co];
    }

    float bn_mean_ci = BN_mean[ci];
    float bn_var_ci  = BN_var[ci];
    float inv_std_ci = 1.0f / sqrtf(bn_var_ci + eps);
    float scale_ci = scale_bn * inv_std_ci;
    float shift_ci = shift_bn - bn_mean_ci * scale_ci;

    // ci is scalar bc index; scalar load
    float x0 = (ci < C_IN ? X[b * sXn + ci * sXc + oh * sYh + (0) * sYw] : 0.0f);  // oh == ow
    float v0 = maxf(0.0f, x0 * scale_ci + shift_ci);
    float w0 = (co < C_OUT && ci < C_IN ? W[co * sWco + ci * sWci + 0 * sWk + 0 * sWk2] : 0.0f);
    *y_ptr = v0 * w0;
}

// K1 1x1 BN+ReLU+Conv fused kernel (stride=2), used for AvgPool2d downsampling
__global__ void bn_relu_conv1x1_stride2_kernel(
    const float* X, const float* W, const float* BN_mean, const float* BN_var,
    const float* BN_weight, const float* BN_bias,
    float* Y,
    int B, int C_IN, int C_OUT, int H, int W,
    int sXn, int sXc, int sXh, int sXw,
    int sWco, int sWci, int sWk, int sWk2,
    int sYn, int sYc, int sYh, int sYw,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int pooling_h, int pooling_w,
    int eps,
    bool training,
    bool has_bias_bn
) {
    int co = blockIdx.x;
    int bc = blockIdx.y;
    int oh = tile_w * 32 + threadIdx.x;

    if (co >= C_OUT || bc >= B * MAX_C_IN) return;

    int b = bc / MAX_C_IN;
    int ci = bc % MAX_C_IN;

    float* y_ptr = Y + b * sYn + co * sYc + oh * sYh + (0) * sYw;

    float scale_bn = 1.0f;
    float shift_bn = 0.0f;
    if (has_bias_bn) {
        scale_bn = BN_weight[co];
        shift_bn = BN_bias[co];
    }

    float bn_mean_ci = BN_mean[ci];
    float bn_var_ci  = BN_var[ci];
    float inv_std_ci = 1.0f / sqrtf(bn_var_ci + eps);
    float scale_ci = scale_bn * inv_std_ci;
    float shift_ci = shift_bn - bn_mean_ci * scale_ci;

    float x0 = (ci < C_IN ? X[b * sXn + ci * sXc + (oh * 2) * sXh + (0) * sXw] : 0.0f);
    float v0 = maxf(0.0f, x0 * scale_ci + shift_ci);
    float w0 = (co < C_OUT && ci < C_IN ? W[co * sWco + ci * sWci + 0 * sWk + 0 * sWk2] : 0.0f);
    *y_ptr = v0 * w0;
}

// Fallback FP32 kernels: identical logic but in fp32
// (Code omitted for brevity; can be generated similarly if needed.)
