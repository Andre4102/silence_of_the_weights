#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>  
#include <cstdlib>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
      cudaError_t err = (call);                                             \
      if (err != cudaSuccess) {                                             \
          fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                  __FILE__, __LINE__, cudaGetErrorString(err));             \
          std::exit(EXIT_FAILURE);                                          \
      }                                                                     \
  } while (0)

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
    int seq_idx   = blockIdx.y * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    int qk_offset = qk_head_offsets[head_index];
    int v_offset  = v_head_offsets[head_index];

    const T* q_row = Q + batch_idx * seq_len * total_qk_channels + seq_idx * total_qk_channels + qk_offset;
    T*       out_row = O + batch_idx * seq_len * total_v_channels + seq_idx * total_v_channels + v_offset;

    // First pass: compute row maximum for numerical stability
    float row_max = -INFINITY;
    for (int j = 0; j < seq_len; ++j) {
        const T* k_row = K + batch_idx * seq_len * total_qk_channels + j * total_qk_channels + qk_offset;
        float score = 0.0f;
        for (int d = 0; d < qk_head_size; ++d)
            score += (float)q_row[d] * (float)k_row[d];
        score *= scale;
        if (j > seq_idx) score = -INFINITY;  // causal masking
        row_max = fmaxf(row_max, score);
    }

    // Second pass: compute row sum
    float row_sum = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        const T* k_row = K + batch_idx * seq_len * total_qk_channels + j * total_qk_channels + qk_offset;
        float score = 0.0f;
        for (int d = 0; d < qk_head_size; ++d)
            score += (float)q_row[d] * (float)k_row[d];
        score *= scale;
        if (j > seq_idx) score = -INFINITY;
        row_sum += expf(score - row_max);
    }

    // Initialize output row to zero
    for (int d = 0; d < v_head_size; ++d)
        out_row[d] = 0.0f;

    // Third pass: weighted sum of values
    for (int j = 0; j < seq_len; ++j) {
        const T* k_row = K + batch_idx * seq_len * total_qk_channels + j * total_qk_channels + qk_offset;
        const T* v_row = V + batch_idx * seq_len * total_v_channels + j * total_v_channels + v_offset;

        float score = 0.0f;
        for (int d = 0; d < qk_head_size; ++d)
            score += (float)q_row[d] * (float)k_row[d];
        score *= scale;
        if (j > seq_idx) score = -INFINITY;

        float softmax_val = expf(score - row_max) / row_sum;
        for (int d = 0; d < v_head_size; ++d)
            out_row[d] += softmax_val * (float)v_row[d];
    }
}

// Utility kernel to fill a buffer
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
    const int* qk_head_sizes,
    const int* v_head_sizes,
    int batch_size,
    int seq_len,
    int num_heads,
    int total_qk_channels,
    int total_v_channels,
    float scale,
    cudaStream_t stream
) {
    // Basic input validation
    if (!Q || !K || !V || !output || !qk_head_sizes || !v_head_sizes ||
        batch_size <= 0 || seq_len <= 0 || num_heads <= 0 || num_heads > 64)
        return;

    // Compute per-head offsets on host
    int h_qk_head_offsets[64];
    int h_v_head_offsets[64];
    h_qk_head_offsets[0] = 0;
    h_v_head_offsets[0] = 0;
    for (int i = 1; i < num_heads; ++i) {
        h_qk_head_offsets[i] = h_qk_head_offsets[i-1] + qk_head_sizes[i-1];
        h_v_head_offsets[i]  = h_v_head_offsets[i-1] + v_head_sizes[i-1];
    }

    // Validate final sums
    int qk_sum = h_qk_head_offsets[num_heads-1] + qk_head_sizes[num_heads-1];
    int v_sum  = h_v_head_offsets[num_heads-1] + v_head_sizes[num_heads-1];
    if (qk_sum != total_qk_channels || v_sum != total_v_channels) return;

    // Copy offsets to device
    int* d_qk_head_offsets = nullptr;
    int* d_v_head_offsets  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_qk_head_offsets, num_heads * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_v_head_offsets,  num_heads * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d_qk_head_offsets, h_qk_head_offsets,
                               num_heads * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v_head_offsets,  h_v_head_offsets,
                               num_heads * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Launch one kernel per head
    for (int head = 0; head < num_heads; ++head) {
        int qk_head_size = qk_head_sizes[head];
        int v_head_size  = v_head_sizes[head];

        int threads_per_block = min(256, seq_len);
        int blocks_needed     = (seq_len + threads_per_block - 1) / threads_per_block;

        dim3 grid(batch_size, blocks_needed);
        dim3 block(threads_per_block);

        flash_attention_head_kernel<float><<<grid, block, 0, stream>>>(
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
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_qk_head_offsets));
    CUDA_CHECK(cudaFree(d_v_head_offsets));
}
