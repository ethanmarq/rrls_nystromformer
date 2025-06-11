"""
This is Nystorm Attention without RRLS it is a backup
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
#from recursive_leverage_score import torch_gauss_kernel, recursive_nystrom_pytorch
"""
Recursive randomized Ridge Leverage Score
"""



# ================================================
#  PyTorch Kernel Function (Gaussian/RBF) 
# ================================================
def _torch_gauss_kernel(X: torch.Tensor, Y: torch.Tensor = None, gamma: float = 0.01):

    X = X.float() 
    if Y is None:
        return torch.ones(X.shape[0], device=X.device, dtype=X.dtype)
    else:
        Y = Y.float()
        nsq_rows = torch.sum(X**2, dim=1, keepdim=True)
        nsq_cols = torch.sum(Y**2, dim=1, keepdim=True)
        Ksub = nsq_rows - 2 * torch.matmul(X, Y.T) + nsq_cols.T
        Ksub = torch.clamp(Ksub, min=0.0)
        return torch.exp(-gamma * Ksub)

# ================================================
# Recursive Nystrom PyTorch Implementation 
# ================================================
def _recursive_nystrom_pytorch(
        X: torch.Tensor,
        n_components: int,
        kernel_func,
        lmbda_0: float = 1e-6,
        random_seed: int = None,
        return_leverage_score: bool = False
        ):

    N = X.shape[0]
    device = X.device
    dtype = X.dtype

    if n_components <= 0:
        print("Warning: n_components <= 0. Returning empty tensor.") # debug
        empty_indices = torch.tensor([], dtype=torch.long, device=device)
        if return_leverage_score:
            empty_scores = torch.tensor([], dtype=dtype, device=device)
            return empty_indices, empty_scores
        return empty_indices

    if N == 0 : # If input X itself is empty
        print("Warning: Input X is empty. Returning empty tensor.") # debug
        empty_indices = torch.tensor([], dtype=torch.long, device=device)
        if return_leverage_score:
            empty_scores = torch.tensor([], dtype=dtype, device=device)
            return empty_indices, empty_scores
        return empty_indices

    if n_components >= N:
        print(f"Warning: n_components ({n_components}) >= N ({N}). Returning all indices.") # debug
        indices = torch.arange(N, device=device)
        if return_leverage_score:
            scores = torch.ones(N, device=device, dtype=dtype) * (n_components / N) # Or just 1s
            return indices, scores
        return indices

    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = None

    log_n_components = torch.log(torch.tensor(n_components, dtype=dtype, device=device))
    n_oversample = torch.clamp(log_n_components, min=0.1).item()
    k = max(1, int(torch.ceil(torch.tensor(n_components / (4 * n_oversample))).item()))
    log2_ratio = torch.log2(torch.tensor(N / n_components, dtype=dtype, device=device))
    n_levels = max(0, int(torch.ceil(log2_ratio).item()))

    perm = torch.randperm(N, device=device, generator=generator)

    size_list = [N]
    for _ in range(n_levels):
        next_size = int(torch.ceil(torch.tensor(size_list[-1] / 2.0)).item())
        size_list.append(max(1, next_size))

    initial_sample_size = min(size_list[-1], N)
    if initial_sample_size <= 0: # Should be caught by N==0 earlier, but as safeguard
        print("Warning: Calculated initial sample size is non-positive.") # debug
        empty_indices = torch.tensor([], dtype=torch.long, device=device)
        if return_leverage_score:
            empty_scores = torch.tensor([], dtype=dtype, device=device)
            return empty_indices, empty_scores
        return empty_indices

    sample_in_smallest_subset = torch.arange(initial_sample_size, device=device)
    indices = perm[sample_in_smallest_subset]
    weights = torch.ones(indices.shape[0], device=device, dtype=dtype)

    try:
        k_diag = kernel_func(X, None).squeeze()
        if k_diag.shape[0] != N:
            raise ValueError(f"kernel_func(X, None) returned shape {k_diag.shape}, expected ({N},)")
        if not torch.isfinite(k_diag).all():
            print("Warning: Non-finite values in kernel diagonal. Clamping.") # debug
            k_diag = torch.nan_to_num(k_diag, nan=1.0, posinf=1.0, neginf=0.0)
    except Exception as e:
        raise RuntimeError(f"Error calling kernel_func(X, None): {e}")

    for l in reversed(range(n_levels + 1)):
        if indices.numel() == 0:
            print(f"Error: Landmark set became empty at level {l}. Returning empty.") # debug
            final_empty_indices = torch.tensor([], dtype=torch.long, device=device)
            if return_leverage_score:
                final_empty_scores = torch.zeros(N, device=device, dtype=dtype) # Return zero scores for all
                return final_empty_indices, final_empty_scores
            return final_empty_indices


        current_subset_size = min(size_list[l], N)
        if current_subset_size <= 0: continue
        current_indices_in_perm = perm[:current_subset_size]

        X_current = X[current_indices_in_perm, :]
        X_landmarks = X[indices, :]

        try:
            KS = kernel_func(X_current, X_landmarks)
            SKS = kernel_func(X_landmarks, X_landmarks)
            if not torch.isfinite(KS).all() or not torch.isfinite(SKS).all():
                print(f"Warning: Non-finite values in KS or SKS at level {l}. Clamping.") # debug
                KS = torch.nan_to_num(KS)
                SKS = torch.nan_to_num(SKS)
        except Exception as e:
            raise RuntimeError(f"Error calling kernel_func at level {l}: {e}")

        num_landmarks_in_sample = SKS.shape[0]
        current_k = min(k, num_landmarks_in_sample)
        lmbda_val = torch.tensor(1e-6, device=device, dtype=dtype)

        if current_k > 0 and num_landmarks_in_sample > 0:
            try:
                weighted_SKS = SKS * torch.outer(weights, weights)
                diag_SKS = torch.diag(SKS)
                trace_weighted_SKS = torch.sum(diag_SKS * (weights**2))
                if not torch.isfinite(weighted_SKS).all():
                    print("Warning: Non-finite values in weighted_SKS. Clamping.") # debug
                    weighted_SKS = torch.nan_to_num(weighted_SKS)
                if not torch.allclose(weighted_SKS, weighted_SKS.T, atol=1e-5):
                    print("Warning: weighted_SKS is not symmetric. Symmetrizing.") # debug
                    weighted_SKS = (weighted_SKS + weighted_SKS.T) / 2.0
                eigvals = torch.linalg.eigvalsh(weighted_SKS)
                if not torch.isfinite(eigvals).all():
                    print(f"Warning: Non-finite eigenvalues detected at level {l}. Using fallback lambda.") # debug
                    lmbda_val = torch.maximum(torch.tensor(lmbda_0 * num_landmarks_in_sample, device=device, dtype=dtype), trace_weighted_SKS / current_k)
                else:
                    sum_largest_k_eigvals = torch.sum(eigvals[-current_k:])
                    lmbda_calc = (trace_weighted_SKS - sum_largest_k_eigvals) / current_k
                    lmbda_calc = torch.clamp(lmbda_calc, min=0.0)
                    lmbda_val = torch.maximum(torch.tensor(lmbda_0 * num_landmarks_in_sample, device=device, dtype=dtype), lmbda_calc)
            except torch.linalg.LinAlgError: # Removed 'as e' as e is not used
                print(f"Warning: Eigenvalue computation failed at level {l}. Using fallback lambda.") # debug
                lmbda_val = torch.tensor(lmbda_0 * num_landmarks_in_sample + 1e-5, device=device, dtype=dtype)
            except Exception:
                print(f"Warning: Unexpected error during lambda calculation at level {l}. Using fallback.") # debug
                lmbda_val = torch.tensor(lmbda_0 * num_landmarks_in_sample + 1e-5, device=device, dtype=dtype)

        lmbda = torch.maximum(lmbda_val, torch.tensor(1e-8, device=device, dtype=dtype))
        leverage_score_values = None
        try:
            diag_reg = torch.diag(lmbda / torch.clamp(weights**2, min=1e-10))
            inv_term = torch.linalg.inv(SKS + diag_reg)
            R = torch.matmul(KS, inv_term)
            row_sums_R_KS = torch.sum(R * KS, dim=1)
            current_k_diag = k_diag[current_indices_in_perm]
            if current_k_diag.shape != row_sums_R_KS.shape:
                raise ValueError(f"Shape mismatch in RLS: k_diag {current_k_diag.shape} vs row_sums {row_sums_R_KS.shape}")
            leverage_score_unscaled = torch.clamp(current_k_diag - row_sums_R_KS, min=0.0)
            leverage_score_values = (1.0 / lmbda) * leverage_score_unscaled
            if not torch.isfinite(leverage_score_values).all():
                print(f"Warning: Non-finite leverage scores computed at level {l}. Clamping.") # debug
                leverage_score_values = torch.nan_to_num(leverage_score_values, nan=0.0, posinf=1.0)
        except torch.linalg.LinAlgError: # Removed 'as e'
            print(f"Warning: Linear algebra error during RLS computation at level {l}. Falling back to uniform.") # debug
            leverage_score_values = None
        except Exception:
            print(f"Warning: Unexpected error during RLS computation at level {l}. Falling back to uniform.") # debug
            leverage_score_values = None

        if leverage_score_values is None:
            leverage_score_values = torch.ones(current_subset_size, device=device, dtype=dtype) * (n_components / max(1, current_subset_size))

        if l == 0:
            p = torch.clamp(leverage_score_values, min=0.0)
            p_sum = torch.sum(p)
            if p_sum <= 1e-10:
                print("Warning: Final leverage scores sum to <= 0. Using uniform probabilities.") # debug
                p = torch.ones_like(p) / max(1, p.numel())
            else:
                p = p / p_sum

            final_n_components = min(n_components, p.numel())
            if final_n_components < n_components:
                print(f"Warning: Requested n_components ({n_components}) > available points ({p.numel()}). Sampling {final_n_components}.") # debug

            if p.numel() > 0 and final_n_components > 0:
                sampled_relative_indices = torch.multinomial(p, num_samples=final_n_components, replacement=False, generator=generator)
                final_indices = perm[sampled_relative_indices] 
            else:
                final_indices = torch.tensor([], dtype=torch.long, device=device)

            if return_leverage_score:
                final_leverage_scores = torch.zeros(N, device=device, dtype=dtype)
                final_leverage_scores[perm] = leverage_score_values
                return final_indices, final_leverage_scores
            return final_indices
        else:
            sampling_prob = torch.clamp(n_oversample * leverage_score_values, min=0.0, max=1.0)
            rand_vals = torch.rand(current_subset_size, device=device, dtype=dtype, generator=generator)
            sampled_relative_indices = torch.where(rand_vals < sampling_prob)[0]

            if sampled_relative_indices.numel() == 0:
                print(f"Warning: No points sampled via RLS at level {l}. Falling back to uniform.") # debug
                num_fallback_samples = min(max(1, n_components // (n_levels + 1 if n_levels > -1 else 1)), current_subset_size)
                if current_subset_size > 0:
                    fallback_perm = torch.randperm(current_subset_size, device=device, generator=generator)
                    sampled_relative_indices = fallback_perm[:num_fallback_samples]
                    sampling_prob.fill_(0.0)
                    if current_subset_size > 0 : # Avoid division by zero
                        sampling_prob[sampled_relative_indices] = num_fallback_samples / current_subset_size
                else:
                    sampled_relative_indices = torch.tensor([], dtype=torch.long, device=device)

            if sampled_relative_indices.numel() > 0:
                indices = current_indices_in_perm[sampled_relative_indices]
                sample_probs = sampling_prob[sampled_relative_indices]
                weights = 1.0 / torch.sqrt(torch.clamp(sample_probs, min=1e-10))
            else:
                indices = torch.tensor([], dtype=torch.long, device=device)
                weights = torch.tensor([], dtype=dtype, device=device)

    # Fallback return (should ideally be caught by the l==0 branch)
    print("Warning: Recursive Nystrom loop finished unexpectedly (reached end of function).") # debug
    final_fallback_indices = torch.tensor([], dtype=torch.long, device=device)
    if 'final_indices' in locals() and final_indices.numel() > 0: return final_indices
    elif indices.numel() > 0: return indices # Return current sample if loop exited early

    if return_leverage_score:
        final_fallback_scores = torch.zeros(N, device=device, dtype=dtype)
        return final_fallback_indices, final_fallback_scores
    return final_fallback_indices




"""
Nystrom Code
"""
class NystromSelfAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_landmarks, mask=False, CUDA=False, conv_kernel_size=None):
        super(NystromSelfAttention, self).__init__()
        self.query_embed = nn.Linear(embed_dim, d_k)
        self.key_embed = nn.Linear(embed_dim, d_k)
        self.value_embed = nn.Linear(embed_dim, d_v)
        self.d_k = d_k
        self.mask = mask
        self.num_landmarks = num_landmarks
        self.dropout = nn.Dropout(0.1)
        self.CUDA = CUDA
        self.device = torch.device("cuda:0" if CUDA else "cpu")
        self.init_option = "original"

        # Regularization parameters for RRLS
        self.rrls_lmbda_0 = 1e-6  # Regularization parameter for RRLS
        self.rrls_gamma = 0.01  # Gamma parameter for Gaussian kernel in RRLS
        self.rrls_seed_counter = 0  # Counter for RRLS random seed, can be used to control randomness


        # Currently trails does not use convolution
        self.use_conv = False
        #self.use_conv = conv_kernel_size is not None
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=1,  
                out_channels=1,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                bias=False
            ).to(self.device)


    def _sample_landmarks_rrls(self, tensor_3d, target_num_landmarks):
        batch_size = tensor_3d.shape[0]
        # seq_len_dim = tensor_3d.shape[1] 
        d_features = tensor_3d.shape[2]

        landmark_list = []
        # Optional
        current_seed = self.rrls_seed_counter if hasattr(self, 'rrls_seed_counter') else None
        if current_seed is not None: self.rrls_seed_counter +=1
            #

        for i in range(batch_size):
            current_item = tensor_3d[i]  # Shape: (seq_len, d_features)
            current_seq_len = current_item.shape[0]

            if current_seq_len == 0: # Handle empty sequence for this batch item
                selected_indices_item = torch.tensor([], dtype=torch.long, device=self.device)
            else:
                # RRLS handles n_components >= N by returning all N indices.
                # It also handles n_components == 0 or N == 0.
                selected_indices_item = _recursive_nystrom_pytorch(
                        X=current_item,
                        n_components=target_num_landmarks,
                        kernel_func=lambda x, y=None: _torch_gauss_kernel(x, y, gamma=self.rrls_gamma),
                        lmbda_0=self.rrls_lmbda_0,
                        random_seed=None, 
                        return_leverage_score=False
                        )

            num_selected = selected_indices_item.shape[0]
            if num_selected > 0:
                landmarks_item = current_item[selected_indices_item, :]
            else: # No indices selected
                landmarks_item = torch.zeros((0, d_features), device=self.device, dtype=tensor_3d.dtype)

            # Pad if fewer landmarks were selected/available
            if num_selected < target_num_landmarks:
                padding_size = target_num_landmarks - num_selected
                padding = torch.zeros((padding_size, d_features), device=self.device, dtype=tensor_3d.dtype)
                landmarks_item = torch.cat([landmarks_item, padding], dim=0)
            elif num_selected > target_num_landmarks:
                landmarks_item = landmarks_item[:target_num_landmarks, :]

            landmark_list.append(landmarks_item)

        return torch.stack(landmark_list, dim=0)



    def forward(self, query_in, key_in, value_in):
        batch_size = query_in.size(0)
        # Query Sequence Length
        q_seq_len = query_in.size(1)  
        # Key Sequence length
        k_seq_len = key_in.size(1)    
        
        # Linear projections
        query = self.query_embed(query_in)
        key = self.key_embed(key_in)
        value = self.value_embed(value_in)
        
        # Scale for stability
        scaling = 1.0 / math.sqrt(math.sqrt(self.d_k))
        query = query * scaling
        key = key * scaling
        
        # Create proper causal mask if needed
        if self.mask:
            mask_tensor = torch.ones(batch_size, q_seq_len, k_seq_len, device=self.device)
            mask_tensor = torch.tril(mask_tensor)  # Lower triangular mask

        # Case for decoder self-attention (q_seq_len = 1, k_seq_len > 1)
        is_decoder_self_attn = True
        #is_decoder_self_attn = (q_seq_len == 1 and k_seq_len > 1)
        
        # Optional Nystrom Attention which may improve performance 
        # for low squenece length cases
        #use_nystrom = seq_len > 4 and seq_len > self.num_landmarks
        
        # Current training forces Nystrom to always be used
        use_nystrom = True #
        
        if use_nystrom:
            # Find the largest divisor of seq_len that's <= num_landmarks
            # This prevents errors
            num_landmarks = self.num_landmarks
            while k_seq_len % num_landmarks != 0 and num_landmarks > 1:
                num_landmarks -= 1
            
            segments = k_seq_len // num_landmarks
            
            # Fallback case for saftey
            if num_landmarks == 0:
                # Fall back to standard attention for simplicity
                use_nystrom = False
                print(f"  Falling back to standard attention (no good divisor found)")
            else:
                #print(f"  Using {num_landmarks} landmarks with {segments} segments")
                pass

        # Standard Attention
        if not use_nystrom:
            print(f"Using standard attention with seq_len={seq_len}")
            # Standard attention implementation
            key_transposed = torch.transpose(key, 1, 2)
            attention_weights = torch.matmul(query, key_transposed)
            
            if self.mask:
                # Apply mask
                attention_weights = attention_weights.masked_fill(
                    mask_tensor.unsqueeze(1) == 0, float('-inf')
                )
            
            attention_weights = F.softmax(attention_weights, dim=-1)
            attention_weighted_value = torch.matmul(attention_weights, value)
        else:
            # Nystrom attention implementation
            
            # 1. Create landmark tokens
            try:
                # Only reshape key and value, not query if it's decoder self-attention
                key_reshaped = key.reshape(batch_size, num_landmarks, segments, -1)
                
                # Create landmarks by averaging each segment
                # Uniform Sampling
                #k_landmarks = key_reshaped.mean(dim=2)    # [batch_size, num_landmarks, d_k]
                k_landmarks = self._sample_landmarks_rrls(key_reshaped, num_landmarks)  # [batch_size, num_landmarks, d_k]
                

                # With this decoder this statement will always evaluate true
                if is_decoder_self_attn:
                    #q_landmarks = self._sample_landmarks_rrls(query, num_landmarks)  # [batch_size, num_landmarks, d_k]
                    q_landmarks = query
                else:
                    # Normal case, reshape query
                    #query_reshaped = query.reshape(batch_size, num_landmarks, segments, -1)
                    #q_landmarks = query_reshaped.mean(dim=2)
                    assert True, "Decoder self-attention is not supported in this implementation"
                

                print(f"q_landmarks shape: {q_landmarks.shape}, k_landmarks shape: {k_landmarks.shape}")
                # 2. Compute the three Nystrom kernels
                # kernel_1: [batch_size, seq_len, num_landmarks]
                kernel_1 = torch.matmul(query, k_landmarks.transpose(-1, -2))

                kernel_1 = F.softmax(kernel_1, dim=-1)
                
                # kernel_2: [batch_size, num_landmarks, num_landmarks]
                kernel_2 = torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2))

                kernel_2 = F.softmax(kernel_2, dim=-1)
                
                # kernel_3: [batch_size, num_landmarks, seq_len]
                kernel_3 = torch.matmul(q_landmarks, key.transpose(-1, -2))

                
                if self.mask:
                    if is_decoder_self_attn:
                        seq_mask = torch.ones(batch_size, 1, k_seq_len, device=self.device)
        
                    last_pos = q_seq_len - 1
                    for i in range(k_seq_len):
                        if i > last_pos:
                            seq_mask[:, :, i] = 0
                    else:
                        # For regular causal masking
                        seq_mask = torch.tril(torch.ones(batch_size, num_landmarks, k_seq_len, device=self.device))
                    # Apply mask
                    kernel_3 = kernel_3.masked_fill(seq_mask == 0, float('-inf'))
                
                kernel_3 = F.softmax(kernel_3, dim=-1)
                
                # 3. Compute Nystrom approximation
                # Moore-Penrose pseudoinverse of kernel_2
                kernel_2_inv = torch.pinverse(kernel_2)
                #kernel_2_inv = self.iterative_inv(kernel_2) # Debugging as there is a shape mismatch
                
                # Final Nystrom approximation
                attention_weighted_value = torch.matmul(
                    torch.matmul(kernel_1, kernel_2_inv),
                    torch.matmul(kernel_3, value)
                )
                
            except RuntimeError as e:
                print(f"Error in Nystrom attention: {e}")
                print(f"Falling back to standard attention due to error")
                
                # Fall back to standard attention on error
                key_transposed = torch.transpose(key, 1, 2)
                attention_weights = torch.matmul(query, key_transposed)
                
                if self.mask:
                    attention_weights = attention_weights.masked_fill(
                        mask_tensor.unsqueeze(1) == 0, float('-inf')
                    )
                
                attention_weights = F.softmax(attention_weights, dim=-1)
                attention_weighted_value = torch.matmul(attention_weights, value)
        
        # Convolution if conv_kernel_size is not none
        if self.use_conv:
            v_conv = value.unsqueeze(1)
            v_conv = self.conv(v_conv)
            
        attention_weighted_value = self.dropout(attention_weighted_value)
        return attention_weighted_value
        
    def iterative_inv(self, mat, n_iter=6):
        I = torch.eye(mat.size(-1), device=mat.device)
        K = mat
    
        if self.init_option == "original":
            V = 1 / torch.max(torch.sum(K, dim=-2)) * K.transpose(-1, -2)
        else:
            V = 1 / torch.max(torch.sum(K, dim=-2), dim=-1).values[:, :, None, None] * K.transpose(-1, -2)
    
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

class NystromMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_heads, num_landmarks, mask=False, conv_kernel_size=None, CUDA=False):
        super(NystromMultiHeadAttention, self).__init__()
        self.attention_blocks = nn.ModuleList(
            [NystromSelfAttention(embed_dim, d_k, d_v, num_landmarks, mask, conv_kernel_size, CUDA) for _ in range(num_heads)]
        )
        self.norm = LayerNorm(embed_dim)
        self.CUDA = CUDA
        self.device = torch.device("cuda:0" if CUDA else "cpu")

    def forward(self, query, key, value, residual_x):
        attention_out = torch.tensor([], requires_grad=True).to(self.device)
        for attention in self.attention_blocks:
            attention_out = torch.cat(
                (attention_out, attention(query, key, value)), dim=2
            )
        add_and_norm = self.norm(attention_out + residual_x)
        return add_and_norm

class NystromTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_landmarks, mask=False, conv_kernel_size=None, CUDA=False):
        super(NystromTransformerBlock, self).__init__()
        self.multi_head_attention = NystromMultiHeadAttention(
            embed_dim,
            embed_dim // num_heads,
            embed_dim // num_heads,
            num_heads,
            num_landmarks,
            mask,
            conv_kernel_size,
            CUDA=CUDA,
        )
        self.feed_forward = PositionWiseFeedForward(embed_dim, embed_dim)

    def forward(self, query, key, value, residual_x):
        attention_out = self.multi_head_attention(query, key, value, residual_x)
        feed_forward_out = self.feed_forward(attention_out, attention_out)
        return feed_forward_out

"""
Non-Nystrom Code
"""
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, mask=False):
        super(SelfAttention, self).__init__()
        self.query_embed = nn.Linear(embed_dim, d_k)
        self.key_embed = nn.Linear(embed_dim, d_k)
        self.value_embed = nn.Linear(embed_dim, d_v)
        self.d_k = d_k
        self.mask = mask
        self.dropout = nn.Dropout(0.1)

    def forward(self, query_in, key_in, value_in):
        query = self.query_embed(query_in)
        key = self.key_embed(key_in)
        value = self.value_embed(value_in)
        key_transposed = torch.transpose(key, 1, 2)
        # Get attention weights
        attention_weights = torch.matmul(query, key_transposed)  # (n_query,n_key)
        attention_weights = attention_weights / math.sqrt(self.d_k)
        if self.mask == True:
            # REF : http://peterbloem.nl/blog/transformers
            indices = torch.triu_indices(
                attention_weights.shape[1], attention_weights.shape[2], offset=1
            )
            attention_weights[:, indices[0], indices[1]] = float("-inf")
        attention_weights = F.softmax(attention_weights, dim=2)
        
        # Apply attention weights to value
        attention_weighted_value = torch.matmul(
            attention_weights, value
        )  # (n_query,n_key) matmul (n_key || n_query , d_v)
        attention_weighted_value = self.dropout(attention_weighted_value)

        return attention_weighted_value


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_heads, mask=False, CUDA=False):
        super(MultiHeadAttention, self).__init__()
        ### Credit: Issue From @shouldsee https://github.com/IpsumDominum/Pytorch-Simple-Transformer/issues/2
        self.attention_blocks = nn.ModuleList(
            [SelfAttention(embed_dim, d_k, d_v, mask) for _ in range(num_heads)]
        )

        self.norm = LayerNorm(embed_dim)
        self.CUDA = CUDA
        self.device = torch.device("cuda:0" if CUDA else "cpu")

    def forward(self, query, key, value, residual_x):
        attention_out = torch.tensor([], requires_grad=True).to(self.device)
        for attention in self.attention_blocks:
            attention_out = torch.cat(
                (attention_out, attention(query, key, value)), dim=2
            )
        add_and_norm = self.norm(attention_out + residual_x)
        return add_and_norm


class LayerNorm(nn.Module):
    "Taken from Annotated Transformer (HarvardNLP)"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        div = (std + self.eps) + self.shift
        return self.scale * (x - mean) / (div)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.l1 = nn.Linear(embed_dim, output_dim)
        self.RELU = nn.ReLU()
        self.l2 = nn.Linear(output_dim, embed_dim)
        self.norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, residual_x):
        x = torch.max(torch.zeros(x.shape), self.l1(x))
        x = self.RELU(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = self.norm(x + residual_x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mask=False, CUDA=False):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            embed_dim,
            embed_dim // num_heads,
            embed_dim // num_heads,
            num_heads,
            mask,
            CUDA=CUDA,
        )
        self.feed_forward = PositionWiseFeedForward(embed_dim, embed_dim)

    def forward(self, query, key, value, residual_x):
        attention_out = self.multi_head_attention(query, key, value, residual_x)
        feed_forward_out = self.feed_forward(attention_out, attention_out)
        return feed_forward_out


class VocabLogits(nn.Module):
    def __init__(self, embed_dim, logit_dim):
        super(VocabLogits, self).__init__()
        self.linear = nn.Linear(embed_dim, logit_dim)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class Embeddings(nn.Module):
    "Taken from Annotated Transformer (HarvardNLP)"

    def __init__(self, vocab_length, embed_dim, CUDA=False):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_length, embed_dim)
        self.pos_encode = PositionalEncoding(embed_dim, CUDA=CUDA)
        self.embed_dim = embed_dim

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.embed_dim)
        return embed + self.pos_encode(embed)


class PositionalEncoding(nn.Module):
    "Modified From Annotated Transformer (HarvardNLP)"

    def __init__(self, embed_dim, max_len=5000, CUDA=False):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term_even = torch.pow(
            10000.0, torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim
        )
        div_term_odd = torch.pow(
            10000.0, torch.arange(1, embed_dim, 2, dtype=torch.float32) / embed_dim
        )

        pe[:, 0::2] = torch.sin(position * div_term_even)
        pe[:, 1::2] = torch.cos(position * div_term_odd)
        pe = pe.unsqueeze(0)
        if CUDA == True:
            pe.type(torch.cuda.FloatTensor)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return x
