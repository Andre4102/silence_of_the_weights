#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "attention_cuda.cuh" // include your kernel header

struct HeadDescCPU {
    int qk_channels;
    int v_channels;
    int seq_len;
    float* Q;
    float* K;
    float* V;
    float* O;
};

void ragged_multihead_attention(const std::vector<HeadDescCPU>& heads_cpu, int tile_size) {
    int num_heads = heads_cpu.size();

    // Allocate GPU HeadDesc array
    HeadDesc* heads_gpu;
    cudaMalloc(&heads_gpu, sizeof(HeadDesc) * num_heads);

    std::vector<HeadDesc> heads_device(num_heads);
    for (int i = 0; i < num_heads; ++i) {
        heads_device[i].qk_channels = heads_cpu[i].qk_channels;
        heads_device[i].v_channels  = heads_cpu[i].v_channels;
        heads_device[i].seq_len     = heads_cpu[i].seq_len;

        // Assume Q/K/V/O are already on GPU
        heads_device[i].Q = heads_cpu[i].Q;
        heads_device[i].K = heads_cpu[i].K;
        heads_device[i].V = heads_cpu[i].V;
        heads_device[i].O = heads_cpu[i].O;
    }

    // Copy descriptors to GPU
    cudaMemcpy(heads_gpu, heads_device.data(), sizeof(HeadDesc) * num_heads, cudaMemcpyHostToDevice);

    // Launch kernel: one block per head, 128 threads per block (adjust as needed)
    dim3 block(128);
    dim3 grid(num_heads);

    // Compute maximum shared memory needed across all heads
    int max_qk = 0;
    int max_v  = 0;
    for (auto& h : heads_cpu) {
        max_qk = std::max(max_qk, h.qk_channels);
        max_v  = std::max(max_v, h.v_channels);
    }
    // Shared memory for Q + K + V tiles
    size_t shared_mem_bytes = tile_size * max_qk * 2 * sizeof(float)  // Q + K
                            + tile_size * max_v * sizeof(float);       // V

    // Launch the kernel
    switch (tile_size) {
        case 32:
            variable_head_attention_kernel<32><<<grid, block, shared_mem_bytes>>>(heads_gpu, num_heads);
            break;
        case 64:
            variable_head_attention_kernel<64><<<grid, block, shared_mem_bytes>>>(heads_gpu, num_heads);
            break;
        default:
            throw std::runtime_error("Unsupported tile size");
    }

    cudaDeviceSynchronize();
    cudaFree(heads_gpu);
}
