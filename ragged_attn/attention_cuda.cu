#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "attention_cuda.cuh"

struct HeadDesc {
    int qk_channels;   // Q/K channels
    int v_channels;    // V channels
    int seq_len;       // sequence length
    float* Q;          // Q pointer for this head
    float* K;          // K pointer for this head
    float* V;          // V pointer for this head
    float* O;          // output pointer for this head
};


template <int TILE>
__global__ void variable_head_attention_kernel(HeadDesc* heads, int num_heads) {
    int head_idx = blockIdx.x;
    if (head_idx >= num_heads) return;

    HeadDesc head = heads[head_idx];

    extern __shared__ float smem[];
    float* Qs = smem;                               // TILE x max_qk_channels
    float* Ks = Qs + TILE * head.qk_channels;
    float* Vs = Ks + TILE * head.qk_channels;

    int tid = threadIdx.x;
    int tile_start = blockIdx.y * TILE;
    int tile_size = min(TILE, head.seq_len - tile_start);

    // Load Q/K/V tiles
    for (int i = tid; i < tile_size * head.qk_channels; i += blockDim.x) {
        int row = i / head.qk_channels;
        int col = i % head.qk_channels;
        Qs[row * head.qk_channels + col] = head.Q[(tile_start + row) * head.qk_channels + col];
        Ks[row * head.qk_channels + col] = head.K[(tile_start + row) * head.qk_channels + col];
    }

    for (int i = tid; i < tile_size * head.v_channels; i += blockDim.x) {
        int row = i / head.v_channels;
        int col = i % head.v_channels;
        Vs[row * head.v_channels + col] = head.V[(tile_start + row) * head.v_channels + col];
    }

    __syncthreads();

    // Compute Q*K^T and softmax
    for (int i = tid; i < tile_size; i += blockDim.x) {
        for (int j = 0; j < tile_size; ++j) {
            float dot = 0.f;
            for (int k = 0; k < head.qk_channels; k++) {
                dot += Qs[i * head.qk_channels + k] * Ks[j * head.qk_channels + k];
            }
            dot /= sqrtf((float)head.qk_channels);
            // store temporarily in shared memory for softmax
            Qs[i * tile_size + j] = dot;  
        }
    }
    __syncthreads();

    // Softmax per row
    for (int i = tid; i < tile_size; i += blockDim.x) {
        float max_val = -1e20f;
        for (int j = 0; j < tile_size; ++j) max_val = fmaxf(max_val, Qs[i * tile_size + j]);

        float sum = 0.f;
        for (int j = 0; j < tile_size; ++j) {
            Qs[i * tile_size + j] = expf(Qs[i * tile_size + j] - max_val);
            sum += Qs[i * tile_size + j];
        }

        for (int j = 0; j < tile_size; ++j) Qs[i * tile_size + j] /= sum;
    }
    __syncthreads();

    // Multiply by V
    for (int i = tid; i < tile_size; i += blockDim.x) {
        for (int c = 0; c < head.v_channels; ++c) {
            float out_val = 0.f;
            for (int j = 0; j < tile_size; ++j) {
                out_val += Qs[i * tile_size + j] * Vs[j * head.v_channels + c];
            }
            head.O[(tile_start + i) * head.v_channels + c] = out_val;
        }
    }
}
