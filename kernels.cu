
    #include <torch/extension.h>
    #include <cuda_runtime.h>
    #include <cmath>

    __global__ void concat2d_kernel(
        const float* A, const float* B, float* C,
        int rows, int cols_a, int cols_b)
    {
        const int total_cols = cols_a + cols_b;
        const int idx        = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_elems = rows * total_cols;
        if (idx >= total_elems) return;

        int row = idx / total_cols;
        int col = idx % total_cols;

        if (col < cols_a) {
            C[idx] = A[row * cols_a + col];
        } else {
            C[idx] = B[row * cols_b + (col - cols_a)];
        }
    }

    __global__ void tanh_kernel(const float* input, float* output, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size) return;
        output[idx] = tanh(input[idx]);
    }

    PYBIND11_MODULE(rnn_fused_ops, m) {
        m.def("concat2d_cuda", &concat2d_cuda, "Concatenate two 2-D tensors along dim=1 (CUDA)");
        m.def("tanh_cuda",     &tanh_cuda,     "Element-wise tanh activation (CUDA)");
    }
    