#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

#define CUDA_CHECK(call) \
  do { \
      cudaError_t err = call; \
      if (err != cudaSuccess) { \
          printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
          exit(err); \
      } \
  } while(0)

template<typename T>
__global__ void flash_attention_head_kernel(
    const T* __restrict__ Q,             // [B, N, total_qk_channels]
    const T* __restrict__ K,             // [B, N, total_qk_channels]
    const T* __restrict__ V,             // [B, N, total_v_channels]
    T* O,                                // [B, N, total_v_channels]
    int batch_size,
    int seq_len,
    int qk_head_size,                    // Size of current head for Q/K
    int v_head_size,                     // Size of current head for V
    int head_index,
    int total_qk_channels,               // Total channels for Q/K
    int total_v_channels,                // Total channels for V
    const int* __restrict__ qk_head_offsets,  // Offsets for Q/K heads
    const int* __restrict__ v_head_offsets,   // Offsets for V heads
    float scale
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    int qk_offset = qk_head_offsets[head_index];
    int v_offset = v_head_offsets[head_index];

    const T* q_row = Q + batch_idx * seq_len * total_qk_channels + seq_idx * total_qk_channels + qk_offset;
    T* out_row = O + batch_idx * seq_len * total_v_channels + seq_idx * total_v_channels + v_offset;

    // First pass: compute row maximum for numerical stability
    float row_max = -INFINITY;
    for (int j = 0; j < seq_len; j++) {
        const T* k_row = K + batch_idx * seq_len * total_qk_channels + j * total_qk_channels + qk_offset;
        float score = 0.0f;
        for (int d = 0; d < qk_head_size; d++) {
            score += (float)q_row[d] * (float)k_row[d];
        }
        score *= scale;
        if (j > seq_idx) score = -INFINITY;  // Causal masking
        row_max = fmaxf(row_max, score);
    }

    // Second pass: compute row sum for normalization
    float row_sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        const T* k_row = K + batch_idx * seq_len * total_qk_channels + j * total_qk_channels + qk_offset;
        float score = 0.0f;
        for (int d = 0; d < qk_head_size; d++) {
            score += (float)q_row[d] * (float)k_row[d];
        }
        score *= scale;
        if (j > seq_idx) score = -INFINITY;  // Causal masking
        row_sum += expf(score - row_max);
    }

    // Initialize output row to zero
    for (int d = 0; d < v_head_size; d++) {
        out_row[d] = 0.0f;
    }

    // Third pass: compute weighted sum of values
    for (int j = 0; j < seq_len; j++) {
        const T* k_row = K + batch_idx * seq_len * total_qk_channels + j * total_qk_channels + qk_offset;
        const T* v_row = V + batch_idx * seq_len * total_v_channels + j * total_v_channels + v_offset;
        
        float score = 0.0f;
        for (int d = 0; d < qk_head_size; d++) {
            score += (float)q_row[d] * (float)k_row[d];
        }
        score *= scale;
        if (j > seq_idx) score = -INFINITY;  // Causal masking
        
        float softmax_val = expf(score - row_max) / row_sum;
        for (int d = 0; d < v_head_size; d++) {
            out_row[d] += softmax_val * (float)v_row[d];
        }
    }
}

// Utility kernel to fill buffer with a value
template<typename T>
__global__ void fill_float_kernel(T* data, T value, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

// Host orchestrator
extern "C" void variable_head_flash_attention(
    const void* Q,
    const void* K,
    const void* V,
    void* output,
    const int* qk_head_sizes,    // Head sizes for Q/K (same for both)
    const int* v_head_sizes,     // Head sizes for V (can be different)
    int batch_size,
    int seq_len,
    int num_heads,
    int total_qk_channels,       // Total channels for Q/K
    int total_v_channels,        // Total channels for V
    float scale,
    cudaStream_t stream
) {
    printf("DEBUG: Starting variable_head_flash_attention\n");
    printf("DEBUG: batch_size=%d, seq_len=%d, num_heads=%d\n", batch_size, seq_len, num_heads);
    printf("DEBUG: total_qk_channels=%d, total_v_channels=%d\n", total_qk_channels, total_v_channels);

    // Validate inputs first
    if (!Q || !K || !V || !output) {
        printf("ERROR: Null tensor pointer passed\n");
        return;
    }
    printf("DEBUG: Tensor pointers validated\n");
    
    if (!qk_head_sizes || !v_head_sizes) {
        printf("ERROR: Null head_sizes pointer passed\n");
        printf("DEBUG: qk_head_sizes=%p, v_head_sizes=%p\n", qk_head_sizes, v_head_sizes);
        return;
    }
    printf("DEBUG: Head size pointers validated\n");
    
    if (batch_size <= 0 || seq_len <= 0 || num_heads <= 0) {
        printf("ERROR: Invalid dimensions: batch_size=%d, seq_len=%d, num_heads=%d\n", 
               batch_size, seq_len, num_heads);
        return;
    }
    
    if (num_heads > 64) {  // Reasonable upper limit
        printf("ERROR: Too many heads: %d (max 64)\n", num_heads);
        return;
    }
    
    printf("DEBUG: Input validation passed\n");

    // Read and validate head sizes first (this is where it's crashing)
    printf("DEBUG: Reading head sizes...\n");
    
    // Use stack arrays instead of heap allocation for safety
    int h_qk_head_offsets[64];  // Support up to 64 heads
    int h_v_head_offsets[64];
    
    if (num_heads > 64) {
        printf("ERROR: num_heads %d exceeds maximum 64\n", num_heads);
        return;
    }
    
    // Safely read first head size with bounds checking
    int first_qk_size, first_v_size;
    try {
        first_qk_size = qk_head_sizes[0];
        first_v_size = v_head_sizes[0]; 
        printf("DEBUG: First head sizes - qk=%d, v=%d\n", first_qk_size, first_v_size);
    } catch (...) {
        printf("ERROR: Failed to read first head size\n");
        return;
    }
    
    if (first_qk_size <= 0 || first_qk_size > total_qk_channels || 
        first_v_size <= 0 || first_v_size > total_v_channels) {
        printf("ERROR: Invalid first head sizes: qk=%d (max %d), v=%d (max %d)\n",
               first_qk_size, total_qk_channels, first_v_size, total_v_channels);
        return;
    }
    
    h_qk_head_offsets[0] = 0;
    h_v_head_offsets[0] = 0;
    
    printf("DEBUG: Computing remaining offsets...\n");
    
    // Compute offsets with careful bounds checking
    for (int i = 1; i < num_heads; i++) {
        printf("DEBUG: Processing head %d...\n", i);
        
        // Read head sizes with bounds checking
        int qk_size, v_size;
        try {
            qk_size = qk_head_sizes[i-1];  // Size of previous head
            v_size = v_head_sizes[i-1];
            printf("DEBUG: Head %d-1 sizes: qk=%d, v=%d\n", i, qk_size, v_size);
        } catch (...) {
            printf("ERROR: Failed to read head size %d\n", i-1);
            return;
        }
        
        if (qk_size <= 0 || qk_size > 1024 || v_size <= 0 || v_size > 1024) {
            printf("ERROR: Invalid head size at index %d: qk=%d, v=%d\n", i-1, qk_size, v_size);
            return;
        }
        
        h_qk_head_offsets[i] = h_qk_head_offsets[i-1] + qk_size;
        h_v_head_offsets[i] = h_v_head_offsets[i-1] + v_size;
        
        printf("DEBUG: Head %d offsets: qk=%d, v=%d\n", i, h_qk_head_offsets[i], h_v_head_offsets[i]);
        
        // Sanity check offsets
        if (h_qk_head_offsets[i] < 0 || h_qk_head_offsets[i] >= total_qk_channels ||
            h_v_head_offsets[i] < 0 || h_v_head_offsets[i] >= total_v_channels) {
            printf("ERROR: Offset out of bounds at head %d: qk_offset=%d (max %d), v_offset=%d (max %d)\n",
                   i, h_qk_head_offsets[i], total_qk_channels, h_v_head_offsets[i], total_v_channels);
            return;
        }
    }

    printf("DEBUG: Validating final sums...\n");
    
    // Validate final sums with last head size
    int last_qk_size, last_v_size;
    try {
        last_qk_size = qk_head_sizes[num_heads-1];
        last_v_size = v_head_sizes[num_heads-1];
        printf("DEBUG: Last head sizes: qk=%d, v=%d\n", last_qk_size, last_v_size);
    } catch (...) {
        printf("ERROR: Failed to read last head size\n");
        return;
    }
    
    int qk_sum = h_qk_head_offsets[num_heads-1] + last_qk_size;
    int v_sum = h_v_head_offsets[num_heads-1] + last_v_size;
    
    if (qk_sum != total_qk_channels) {
        printf("ERROR: QK head sizes sum to %d but expected %d\n", qk_sum, total_qk_channels);
        return;
    }
    
    if (v_sum != total_v_channels) {
        printf("ERROR: V head sizes sum to %d but expected %d\n", v_sum, total_v_channels);
        return;
    }

    printf("DEBUG: Offsets computed successfully\n");

    // Copy offsets to device
    int* d_qk_head_offsets = nullptr;
    int* d_v_head_offsets = nullptr;
    
    printf("DEBUG: Allocating device memory...\n");
    CUDA_CHECK(cudaMalloc(&d_qk_head_offsets, num_heads * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_v_head_offsets, num_heads * sizeof(int)));
    
    printf("DEBUG: Copying to device...\n");
    CUDA_CHECK(cudaMemcpyAsync(d_qk_head_offsets, h_qk_head_offsets, num_heads * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v_head_offsets, h_v_head_offsets, num_heads * sizeof(int), cudaMemcpyHostToDevice, stream));
    
    // Wait for copy to complete
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("DEBUG: Starting kernel launches...\n");

    // Process each head separately with better error checking
    for (int head = 0; head < num_heads; head++) {
        int qk_head_size = qk_head_sizes[head];
        int v_head_size = v_head_sizes[head];
        
        printf("DEBUG: Processing head %d (qk_size=%d, v_size=%d)\n", head, qk_head_size, v_head_size);
        
        // Use safer grid/block configuration
        int threads_per_block = min(256, seq_len);
        int blocks_needed = (seq_len + threads_per_block - 1) / threads_per_block;
        
        dim3 kernel_grid(batch_size, blocks_needed);
        dim3 kernel_block(threads_per_block);
        
        printf("DEBUG: Grid(%d,%d), Block(%d)\n", kernel_grid.x, kernel_grid.y, kernel_block.x);

        flash_attention_head_kernel<float><<<kernel_grid, kernel_block, 0, stream>>>(
            (const float*)Q,
            (const float*)K,
            (const float*)V,
            (float*)output,
            batch_size,
            seq_len,
            qk_head_size,
            v_head_size,
            head,
            total_qk_channels,
            total_v_channels,
            d_qk_head_offsets,
            d_v_head_offsets,
            scale
        );
        
        // Check for launch errors
        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            printf("ERROR: Kernel launch failed for head %d: %s\n", head, cudaGetErrorString(launch_err));
            CUDA_CHECK(cudaFree(d_qk_head_offsets));
            CUDA_CHECK(cudaFree(d_v_head_offsets));
            return;
        }
        
        printf("DEBUG: Head %d kernel launched successfully\n", head);
    }

    printf("DEBUG: All kernels launched, synchronizing...\n");
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    printf("DEBUG: Freeing device memory...\n");
    CUDA_CHECK(cudaFree(d_qk_head_offsets));
    CUDA_CHECK(cudaFree(d_v_head_offsets));
    
    printf("DEBUG: Function completed successfully\n");
}