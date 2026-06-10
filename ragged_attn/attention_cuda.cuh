#pragma once

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
__global__ void variable_head_attention_kernel(HeadDesc* heads, int num_heads);