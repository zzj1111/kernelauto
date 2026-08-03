
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// Row-wise softmax kernel: one block per row
template<int BLOCK_SIZE>
__global__ void rowwise_softmax_kernel(
        const float* __restrict__ inp,
        float*       __restrict__ out,
        int n_rows,
        int n_cols)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const float* row_in  = inp + row * n_cols;
    float*       row_out = out + row * n_cols;

    float thread_max = -FLT_MAX;
    for (int col = threadIdx.x; col < n_cols; col += BLOCK_SIZE)
        thread_max = fmaxf(thread_max, row_in[col]);

    float thread_sum = 0.f;
    for (int col = threadIdx.x; col < n_cols; col += BLOCK_SIZE) {
        float ex = expf(row_in[col] - thread_max);
        row_out[col] = ex;
        thread_sum += ex;
    }

    const float denom = thread_sum + 1e-6f;
    for (int col = threadIdx.x; col < n_cols; col += BLOCK_SIZE)
        row_out[col] /= denom;
}

// (B*N*K) × (B*N*D) -> (B,K,D)
__global__ void vlad_weighted_sum_kernel(
        const float* __restrict__ assignment, // (B, N, K)
        const float* __restrict__ feats,      // (B, N, D)
        float*       __restrict__ output,     // (B, K, D)
        int B, int N, int K, int D)
{
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)B * K * D;
    if (idx >= total) return;

    const int d = idx % D;
    const int k = (idx / D) % K;
    const int b = idx / (K * D);

    const int base_a = ((b * N) * K) + (k * N);
    const int base_f = ((b * N) * D) + (d * N);

    float sum = 0.f;
    for (int n = 0; n < N; ++n) {
        float a = assignment[base_a + n * K + k];
        float x = feats[base_f + n * D + d];
        sum += a * x;
    }
    output[idx] = sum;
}

// C++ wrappers
torch::Tensor rowwise_softmax_cuda(torch::Tensor inp) {
    if (!inp.is_cuda() || inp.dtype() != torch::kFloat32) {
        AT_ERROR("rowwise_softmax_cuda expects CUDA float32 tensor");
    }
    const int n_rows = inp.size(0);
    const int n_cols = inp.size(1);
    auto out = torch::empty_like(inp);

    constexpr int BLOCK = 128;
    const dim3 grid(n_rows);
    const dim3 block(BLOCK);
    torch::cuda::Stream stream = torch::cuda::current_stream();

    rowwise_softmax_kernel<BLOCK><<<grid, block, 0, stream>>>(
            inp.data_ptr<float>(),
            out.data_ptr<float>(),
            n_rows, n_cols);

    return out;
}

torch::Tensor vlad_weighted_sum_cuda(torch::Tensor assignment,
                                     torch::Tensor feats) {
    if (!assignment.is_cuda() || !feats.is_cuda() ||
        assignment.dtype() != torch::kFloat32 ||
        feats.dtype()       != torch::kFloat32 ||
        assignment.dim() != 3 || feats.dim() != 3) {
        AT_ERROR("vlad_weighted_sum_cuda expects CUDA float32 tensors (B,N,K) and (B,N,D)");
    }

    const int B = assignment.size(0);
    const int N = assignment.size(1);
    const int K = assignment.size(2);
    const int D = feats.size(2);
    if (feats.size(0) != B || feats.size(1) != N) {
        AT_ERROR("shape mismatch in vlad_weighted_sum_cuda");
    }

    auto out = torch::empty({B, K, D}, feats.options());

    const long long total = (long long)B * K * D;
    const int BLOCK = 256;
    const int GRID  = (total + BLOCK - 1) / BLOCK;
    torch::cuda::Stream stream = torch::cuda::current_stream();

    vlad_weighted_sum_kernel<<<GRID, BLOCK, 0, stream>>>(
            assignment.data_ptr<float>(),
            feats.data_ptr<float>(),
            out.data_ptr<float>(),
            B, N, K, D);

    return out;
}

// Define module once
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rowwise_softmax_cuda", &rowwise_softmax_cuda,
          "Row-wise softmax (CUDA)");
    m.def("vlad_weighted_sum_cuda", &vlad_weighted_sum_cuda,
          "VLAD weighted sum (CUDA)");
}
