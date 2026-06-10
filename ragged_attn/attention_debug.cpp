#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

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

// Safe PyTorch extension wrapper with extensive debugging
torch::Tensor variable_head_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    std::vector<int> qk_head_sizes,
    std::vector<int> v_head_sizes,
    float scale
) {
    std::cout << "=== C++ Wrapper Entry ===" << std::endl;
    
    try {
        std::cout << "Step 1: Basic validation..." << std::endl;
        
        // Very basic checks first
        if (!Q.defined() || !K.defined() || !V.defined()) {
            throw std::runtime_error("One or more input tensors are undefined");
        }
        
        std::cout << "Step 2: Device validation..." << std::endl;
        
        // Check devices
        TORCH_CHECK(Q.is_cuda(), "Q tensor must be on CUDA device");
        TORCH_CHECK(K.is_cuda(), "K tensor must be on CUDA device");  
        TORCH_CHECK(V.is_cuda(), "V tensor must be on CUDA device");
        
        std::cout << "Step 3: Dtype validation..." << std::endl;
        
        // Check dtypes
        TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32, got " + torch::toString(Q.dtype()));
        TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32, got " + torch::toString(K.dtype()));
        TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32, got " + torch::toString(V.dtype()));
        
        std::cout << "Step 4: Contiguity validation..." << std::endl;
        
        // Ensure contiguous - this might be causing the segfault
        if (!Q.is_contiguous()) {
            std::cout << "Making Q contiguous..." << std::endl;
            Q = Q.contiguous();
        }
        if (!K.is_contiguous()) {
            std::cout << "Making K contiguous..." << std::endl;
            K = K.contiguous();
        }
        if (!V.is_contiguous()) {
            std::cout << "Making V contiguous..." << std::endl;
            V = V.contiguous();
        }
        
        std::cout << "Step 5: Shape extraction..." << std::endl;
        
        // Get dimensions safely
        auto q_sizes = Q.sizes();
        auto k_sizes = K.sizes();  
        auto v_sizes = V.sizes();
        
        if (q_sizes.size() != 3 || k_sizes.size() != 3 || v_sizes.size() != 3) {
            throw std::runtime_error("All tensors must be 3D [batch, seq, channels]");
        }
        
        int batch_size = static_cast<int>(q_sizes[0]);
        int seq_len = static_cast<int>(q_sizes[1]);
        int total_qk_channels = static_cast<int>(q_sizes[2]);
        int total_v_channels = static_cast<int>(v_sizes[2]);
        int num_heads = static_cast<int>(qk_head_sizes.size());
        
        std::cout << "Dimensions: batch=" << batch_size << ", seq=" << seq_len 
                  << ", qk_channels=" << total_qk_channels << ", v_channels=" << total_v_channels
                  << ", num_heads=" << num_heads << std::endl;
        
        std::cout << "Step 6: Dimension validation..." << std::endl;
        
        // Validate dimensions
        TORCH_CHECK(k_sizes[0] == batch_size && k_sizes[1] == seq_len && k_sizes[2] == total_qk_channels,
                    "K tensor dimensions must match Q");
        TORCH_CHECK(v_sizes[0] == batch_size && v_sizes[1] == seq_len,
                    "V tensor must have same batch_size and seq_len as Q/K");
        
        std::cout << "Step 7: Head size validation..." << std::endl;
        
        // Validate head sizes
        TORCH_CHECK(qk_head_sizes.size() == v_head_sizes.size(), 
                    "Number of QK heads must equal number of V heads");
        
        // Check for reasonable head sizes
        for (size_t i = 0; i < qk_head_sizes.size(); i++) {
            TORCH_CHECK(qk_head_sizes[i] > 0 && qk_head_sizes[i] <= 1024, 
                        "QK head size " + std::to_string(i) + " is invalid: " + std::to_string(qk_head_sizes[i]));
            TORCH_CHECK(v_head_sizes[i] > 0 && v_head_sizes[i] <= 1024,
                        "V head size " + std::to_string(i) + " is invalid: " + std::to_string(v_head_sizes[i]));
        }
        
        // Verify sums
        int qk_head_size_sum = 0;
        int v_head_size_sum = 0;
        for (size_t i = 0; i < qk_head_sizes.size(); i++) {
            qk_head_size_sum += qk_head_sizes[i];
            v_head_size_sum += v_head_sizes[i];
        }
        
        TORCH_CHECK(qk_head_size_sum == total_qk_channels, 
                    "Sum of QK head sizes (" + std::to_string(qk_head_size_sum) + 
                    ") must equal total QK channels (" + std::to_string(total_qk_channels) + ")");
        TORCH_CHECK(v_head_size_sum == total_v_channels, 
                    "Sum of V head sizes (" + std::to_string(v_head_size_sum) + 
                    ") must equal total V channels (" + std::to_string(total_v_channels) + ")");
        
        std::cout << "Step 8: Output tensor creation..." << std::endl;
        
        // Create output tensor
        auto output = torch::empty(
            {batch_size, seq_len, total_v_channels},
            torch::TensorOptions().dtype(torch::kFloat32).device(V.device())
        );
        
        if (!output.is_contiguous()) {
            output = output.contiguous();
        }
        
        std::cout << "Step 9: Head size tensor creation..." << std::endl;
        
        std::vector<int> qk_copy(qk_head_sizes);  // Make a copy
        std::vector<int> v_copy(v_head_sizes);
        auto qk_head_sizes_cpu = torch::tensor(qk_copy, torch::dtype(torch::kInt32));
        auto v_head_sizes_cpu  = torch::tensor(v_copy,  torch::dtype(torch::kInt32));


        // Print the raw arrays before passing to CUDA
        std::cout << "QK head sizes: ";
        for (size_t i = 0; i < qk_head_sizes.size(); i++) {
            std::cout << qk_head_sizes[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Tensor data (CPU): ";
        int* qk_cpu_ptr = qk_head_sizes_cpu.data_ptr<int>();
        for (int i = 0; i < num_heads; ++i) {
            std::cout << qk_cpu_ptr[i] << " ";
        }
        std::cout << std::endl;

        // Now create device copies (for passing to CUDA)
        auto qk_head_sizes_tensor = qk_head_sizes_cpu.to(Q.device()).contiguous();
        auto v_head_sizes_tensor  = v_head_sizes_cpu.to(V.device()).contiguous();
        
        std::cout << "Step 10: CUDA stream..." << std::endl;
        
        // Get current CUDA stream
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        
        std::cout << "Step 11: About to call CUDA function..." << std::endl;
        std::cout << "Pointers - Q: " << Q.data_ptr<float>() 
                  << ", K: " << K.data_ptr<float>()
                  << ", V: " << V.data_ptr<float>()
                  << ", Output: " << output.data_ptr<float>() << std::endl;
        
        // Call CUDA function with extensive error checking
        try {
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
            
            std::cout << "CUDA function call completed" << std::endl;
            
        } catch (const std::exception& e) {
            throw std::runtime_error("CUDA function failed: " + std::string(e.what()));
        }
        
        std::cout << "Step 12: Function completed successfully" << std::endl;
        return output;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR in C++ wrapper: " << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cerr << "UNKNOWN ERROR in C++ wrapper" << std::endl;
        throw std::runtime_error("Unknown error in C++ wrapper");
    }
}

// PyTorch module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ragged_multihead_attention", &variable_head_attention_forward, 
          "Variable Head Flash Attention Forward (CUDA)",
          py::arg("Q"), py::arg("K"), py::arg("V"), 
          py::arg("qk_head_sizes"), py::arg("v_head_sizes"), py::arg("scale"));
}