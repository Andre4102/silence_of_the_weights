import triton
import triton.language as tl

@triton.jit  
def ragged_attention_kernel(
    Q_flat, K_flat, V_flat, output_flat,
    B, S,
    qk_offsets, v_offsets, 
    total_qk_dim, total_v_dim,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr = 64,
    CAUSAL: tl.constexpr = False,
):
    """
    Single-pass optimized version that minimizes recomputation
    Fixed to use vectorized operations instead of scalar indexing
    """
    batch_id = tl.program_id(0)
    seq_pos = tl.program_id(1) 
    head_id = tl.program_id(2)
    
    # Head offsets and sizes
    qk_start = tl.load(qk_offsets + head_id)
    qk_end = tl.load(qk_offsets + head_id + 1)
    v_start = tl.load(v_offsets + head_id)
    v_end = tl.load(v_offsets + head_id + 1)
    
    head_dim_qk = qk_end - qk_start
    head_dim_v = v_end - v_start
    
    # Dimension indices
    qk_idx = tl.arange(0, BLOCK_SIZE)
    v_idx = tl.arange(0, BLOCK_SIZE)
    qk_mask = qk_idx < head_dim_qk
    v_mask = v_idx < head_dim_v
    
    # Load query vector
    q_offset = batch_id * (S * total_qk_dim) + seq_pos * total_qk_dim + qk_start
    q = tl.load(Q_flat + q_offset + qk_idx, mask=qk_mask, other=0.0)
    
    # Scale factor
    scale = 1.0 / tl.sqrt(tl.cast(head_dim_qk, tl.float32))
    
    # Determine attention range
    attn_end = tl.where(CAUSAL, seq_pos + 1, S)
    
    # Initialize online softmax accumulators
    m_i = tl.cast(-1e30, tl.float32)  # Running max
    l_i = tl.cast(0.0, tl.float32)    # Running sum
    output_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Calculate number of blocks
    num_blocks = tl.cdiv(attn_end, BLOCK_SIZE_K)
    
    # Process in blocks with vectorized operations
    for block_idx in range(num_blocks):
        k_start_pos = block_idx * BLOCK_SIZE_K
        k_end_pos = tl.minimum(k_start_pos + BLOCK_SIZE_K, attn_end)
        
        # Create vectorized indices for the current block
        k_indices = tl.arange(0, BLOCK_SIZE_K) + k_start_pos
        k_valid = k_indices < attn_end
        
        # Initialize block accumulators
        m_j = tl.cast(-1e30, tl.float32)
        l_j = tl.cast(0.0, tl.float32)
        block_output = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Vectorized computation for the entire block
        k_block_offsets = batch_id * (S * total_qk_dim) + k_indices[:, None] * total_qk_dim + qk_start + qk_idx[None, :]
        v_block_offsets = batch_id * (S * total_v_dim) + k_indices[:, None] * total_v_dim + v_start + v_idx[None, :]
        
        # Load K and V matrices for the entire block
        k_block = tl.load(K_flat + k_block_offsets, 
                         mask=k_valid[:, None] & qk_mask[None, :], 
                         other=0.0)
        v_block = tl.load(V_flat + v_block_offsets, 
                         mask=k_valid[:, None] & v_mask[None, :], 
                         other=0.0)
        
        # Compute all scores for the block at once
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(k_valid, scores, -1e30)
        
        # Find block maximum
        m_j = tl.max(scores)
        m_j = tl.where(m_j > -1e29, m_j, -1e30)
        
        # Compute exponentials and sum
        exp_scores = tl.exp(scores - m_j)
        exp_scores = tl.where(k_valid, exp_scores, 0.0)
        l_j = tl.sum(exp_scores)
        
        # Compute weighted sum of values
        block_output = tl.sum(exp_scores[:, None] * v_block, axis=0)
        
        # Online softmax update
        m_i_new = tl.maximum(m_i, m_j)
        alpha = tl.where(m_i > -1e29, tl.exp(m_i - m_i_new), 0.0)
        beta = tl.exp(m_j - m_i_new)
        
        l_i_new = alpha * l_i + beta * l_j
        
        # Update output with proper normalization
        output_acc = alpha * output_acc + beta * block_output
        
        # Update running statistics
        m_i = m_i_new
        l_i = l_i_new
    
    # Final normalization
    final_output = tl.where(l_i > 0, output_acc / l_i, 0.0)
    
    # Store result
    out_offset = batch_id * (S * total_v_dim) + seq_pos * total_v_dim + v_start
    tl.store(output_flat + out_offset + v_idx, final_output, mask=v_mask)