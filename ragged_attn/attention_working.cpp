#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>

// Forward declaration of CUDA kernel
extern "C" {
    void variable_head_flash_attention(
        const void* Q,
        const void* K,
        const void* V,
        void* output,
        const int* qk_head_sizes,
        const int* v_head_sizes,
        int batch_size,
        int seq_len,
        int num_heads,
        int total_qk_channels,
        int total_v_channels,
        float scale,
        cudaStream_t stream
    );
}

// Minimal, fast PyTorch extension wrapper
torch::Tensor variable_head_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    std::vector<int> qk_head_sizes,
    std::vector<int> v_head_sizes,
    float scale
) {
    // Basic validation
    TORCH_CHECK(Q.defined() && K.defined() && V.defined(),
                "Input tensors must be defined");
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "Q, K, and V must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat32 &&
                K.dtype() == torch::kFloat32 &&
                V.dtype() == torch::kFloat32,
                "Q, K, V must be float32");

    // Shape checks
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3,
                "Q, K, V must be 3D [batch, seq, channels]");

    const int batch_size       = static_cast<int>(Q.size(0));
    const int seq_len          = static_cast<int>(Q.size(1));
    const int total_qk_channels = static_cast<int>(Q.size(2));
    const int total_v_channels  = static_cast<int>(V.size(2));
    const int num_heads         = static_cast<int>(qk_head_sizes.size());

    TORCH_CHECK(K.size(0) == batch_size && K.size(1) == seq_len &&
                K.size(2) == total_qk_channels,
                "K tensor dimensions must match Q");
    TORCH_CHECK(V.size(0) == batch_size && V.size(1) == seq_len,
                "V tensor must match Q/K batch and sequence length");
    TORCH_CHECK(qk_head_sizes.size() == v_head_sizes.size(),
                "QK and V head size lists must match");

    // Head size checks
    int qk_sum = 0, v_sum = 0;
    for (size_t i = 0; i < qk_head_sizes.size(); ++i) {
        TORCH_CHECK(qk_head_sizes[i] > 0 && qk_head_sizes[i] <= 1024,
                    "Invalid QK head size at index ", i);
        TORCH_CHECK(v_head_sizes[i] > 0 && v_head_sizes[i] <= 1024,
                    "Invalid V head size at index ", i);
        qk_sum += qk_head_sizes[i];
        v_sum  += v_head_sizes[i];
    }
    TORCH_CHECK(qk_sum == total_qk_channels,
                "Sum of QK head sizes must equal total QK channels");
    TORCH_CHECK(v_sum == total_v_channels,
                "Sum of V head sizes must equal total V channels");

    // Ensure contiguity
    if (!Q.is_contiguous()) Q = Q.contiguous();
    if (!K.is_contiguous()) K = K.contiguous();
    if (!V.is_contiguous()) V = V.contiguous();

    // Output tensor
    auto output = torch::empty(
        {batch_size, seq_len, total_v_channels},
        torch::TensorOptions().dtype(torch::kFloat32).device(V.device())
    );

    // Create CPU tensors for head sizes (host pointers needed by CUDA function)
    auto qk_head_sizes_cpu = torch::tensor(qk_head_sizes, torch::dtype(torch::kInt32));
    auto v_head_sizes_cpu  = torch::tensor(v_head_sizes, torch::dtype(torch::kInt32));

    // Launch CUDA kernel
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    variable_head_flash_attention(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        qk_head_sizes_cpu.contiguous().data_ptr<int>(),
        v_head_sizes_cpu.contiguous().data_ptr<int>(),
        batch_size,
        seq_len,
        num_heads,
        total_qk_channels,
        total_v_channels,
        scale,
        stream
    );

    return output;
}

// PyTorch module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ragged_multihead_attention", &variable_head_attention_forward,
          "Variable Head Flash Attention Forward (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("qk_head_sizes"), py::arg("v_head_sizes"), py::arg("scale"));
}
