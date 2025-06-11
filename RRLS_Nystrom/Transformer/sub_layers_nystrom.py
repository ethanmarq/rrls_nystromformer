import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

# RRLS
import numpy as np
import scipy.linalg as spl
import time
from tqdm import tqdm
import gc

def gauss(X: np.ndarray, Y: np.ndarray=None, gamma=0.01):
    # todo make this implementation more python like!

    if Y is None:
        Ksub = np.ones((X.shape[0], 1))
    else:
        nsq_rows = np.sum(X ** 2, axis=1, keepdims=True)
        nsq_cols = np.sum(Y ** 2, axis=1, keepdims=True)
        Ksub = nsq_rows - np.matmul(X, Y.T * 2)
        Ksub = nsq_cols.T + Ksub
        Ksub = np.exp(-gamma * Ksub)

    return Ksub


def uniformNystrom(X, n_components: int, kernel_func=gauss):
    indices = np.random.choice(X.shape[0], n_components)
    C = kernel_func(X, X[indices,:])
    SKS = C[indices, :]
    W = np.linalg.inv(SKS + 10e-6 * np.eye(n_components))

    return C, W


def recursiveNystrom(X, n_components: int, kernel_func=gauss, accelerated_flag=False, random_state=None, lmbda_0=0, return_leverage_score=False, **kwargs):
    '''

    :param X:
    :param n_components:
    :param kernel_func:
    :param accelerated_flag:
    :param random_state:
    :return:
    '''
    rng = np.random.RandomState(random_state)

    n_oversample = np.log(n_components)
    k = np.ceil(n_components / (4 * n_oversample)).astype(int)
    n_levels = np.ceil(np.log(X.shape[0] / n_components) / np.log(2)).astype(int)
    perm = rng.permutation(X.shape[0])

    # set up sizes for recursive levels
    size_list = [X.shape[0]]
    for l in range(1, n_levels+1):
        size_list += [np.ceil(size_list[l - 1] / 2).astype(int)]

    # indices of poitns selected at previous level of recursion
    # at the base level it's just a uniform sample of ~ n_component points
    sample = np.arange(size_list[-1])
    indices = perm[sample]
    weights = np.ones((indices.shape[0],))

    # we need the diagonal of the whole kernel matrix, so compute upfront
    k_diag = kernel_func(X)

    # Main recursion, unrolled for efficiency
    for l in reversed(range(n_levels)):
        # indices of current uniform sample
        current_indices = perm[:size_list[l]]
        # build sampled kernel

        # all rows and sampled columns
        KS = kernel_func(X[current_indices,:], X[indices,:])
        SKS = KS[sample, :] # sampled rows and sampled columns

        # optimal lambda for taking O(k log(k)) samples
        if k >= SKS.shape[0]:
            # for the rare chance we take less than k samples in a round
            lmbda = 10e-6
            # don't set to exactly 0 to avoid stability issues
        else:
            # eigenvalues equal roughly the number of points per cluster, maybe this should scale with n?
            # can be interpret as the zoom level
           # lmbda = (np.sum(np.diag(SKS) * (weights ** 2))
           #         - np.sum(spl.eigvalsh(SKS * weights[:,None] * weights[None,:], eigvals=(SKS.shape[0]-k, SKS.shape[0]-1))))/k

            # Compute all eigenvalues
            eigenvalues = spl.eigvalsh(SKS * weights[:,None] * weights[None,:])

            # Sort eigenvalues in descending order
            sorted_eigenvalues = np.sort(eigenvalues)[::-1]

            # Select the top k largest eigenvalues
            top_k_eigenvalues = sorted_eigenvalues[:k]

            # Calculate the sum of the top k eigenvalues
            trace_sum = np.sum(top_k_eigenvalues)

            # Calculate lmbda using the correct sum
            lmbda = (np.sum(np.diag(SKS) * (weights ** 2)) - trace_sum)/k

        lmbda = np.maximum(lmbda_0*SKS.shape[0], lmbda)
        if lmbda == lmbda_0*SKS.shape[0]:
            print("Set lambda to %d." % lmbda)
        #lmbda = np.minimum(lmbda, 5)
            # lmbda = spl.eigvalsh(SKS * weights * weights.T, eigvals=(0, SKS.shape[0]-k-1)).sum()/k
            # calculate the n-k smallest eigenvalues

        # compute and sample by lambda ridge leverage scores
        R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
        R = np.matmul(KS, R)
        #R = np.linalg.lstsq((SKS + np.diag(lmbda * weights ** (-2))).T,KS.T)[0].T
        if l != 0:
            # max(0, . ) helps avoid numerical issues, unnecessary in theory
            leverage_score = np.minimum(1.0, n_oversample * (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
            # on intermediate levels, we independently sample each column
            # by its leverage score. the sample size is n_components in expectation
            sample = np.where(rng.uniform(size=size_list[l]) < leverage_score)[0]
            # with very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample
            if sample.size == 0:
                leverage_score[:] = n_components / size_list[l]
                sample = rng.choice(size_list[l], size=n_components, replace=False)
            weights = np.sqrt(1. / leverage_score[sample])

        else:
            leverage_score = np.minimum(1.0, (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
            p = leverage_score/leverage_score.sum()

            sample = rng.choice(X.shape[0], size=n_components, replace=False, p=p)
        indices = perm[sample]

    if return_leverage_score:
        return indices, leverage_score[np.argsort(perm)]
    else:
        return indices

# --- Corrected PyTorch Batch Sampling Function ---
def _sample_landmarks_rrls(self, tensor_3d, target_num_landmarks):
    batch_size = tensor_3d.shape[0]
    d_features = tensor_3d.shape[2]
    device = tensor_3d.device
    landmark_list = []
    for i in range(batch_size):
        current_item = tensor_3d[i]
        current_seq_len = current_item.shape[0]
        if current_seq_len == 0:
            landmarks_item = torch.zeros((0, d_features), device=device, dtype=tensor_3d.dtype)
        else:
            current_item_np = current_item.cpu().detach().numpy()
            selected_indices_np = recursiveNystrom(
                X=current_item_np,
                n_components=target_num_landmarks,
                kernel_func=lambda x, y=None: gauss(x, y, gamma=self.rrls_gamma),
                lmbda_0=self.rrls_lmbda_0,
                random_state=None
            )
            selected_indices_item = torch.from_numpy(selected_indices_np).long().to(device)
            if selected_indices_item.shape[0] > 0:
                landmarks_item = current_item[selected_indices_item, :]
            else:
                landmarks_item = torch.zeros((0, d_features), device=device, dtype=tensor_3d.dtype)
        num_selected = landmarks_item.shape[0]
        if num_selected < target_num_landmarks:
            padding_size = target_num_landmarks - num_selected
            padding = torch.zeros((padding_size, d_features), device=device, dtype=tensor_3d.dtype)
            landmarks_item = torch.cat([landmarks_item, padding], dim=0)
        elif num_selected > target_num_landmarks:
            landmarks_item = landmarks_item[:target_num_landmarks, :]
        landmark_list.append(landmarks_item)
    return torch.stack(landmark_list, dim=0)



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
        self._sample_landmarks_rrls = _sample_landmarks_rrls
        self.rrls_gamma = 0.01  # Default gamma for Gaussian kernel
        self.rrls_lmbda_0 = 0.01  # Default lambda for RRLS, can be adjusted

        # Currently trails does not use convolution
        self.device = torch.device("cpu")
        self.use_conv = conv_kernel_size is not None
        # Temp set conv to False
        # DEBUGGGING, OTHERWISE IT IS TRUE? NOT SURE WHY AS CONV_KERNEL_SIZE IS NONE
        self.use_conv = False
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=1,  
                out_channels=1,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                bias=False
            ).to(self.device)

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
        is_decoder_self_attn = (q_seq_len == 1 and k_seq_len > 1)
        
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
                # Debugging
                #print(f"  Trying to find a good divisor for seq_len={k_seq_len} and num_landmarks={num_landmarks}")
                num_landmarks -= 1
            #print(f"  Found num_landmarks={num_landmarks} for seq_len={k_seq_len}") 
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
                #k_landmarks = key_reshaped.mean(dim=2)    # [batch_size, num_landmarks, d_k]
                k_landmarks = self._sample_landmarks_rrls(self, key_reshaped, num_landmarks)
                k_landmarks = k_landmarks.reshape(batch_size, num_landmarks * segments, -1)  # [batch_size, num_landmarks, d_k]


                # With this decoder this statement will always evaluate true
                # Debugging set to false and make q_landmarks the same as k_landmarks
                is_decoder_self_attn = True
                if is_decoder_self_attn:
                    q_landmarks = query
                else:
                    # Normal case, reshape query
                    #print(f"Query shape: {query.shape}, segments: {segments}, num_landmarks: {num_landmarks}")
                    query_reshaped = query.reshape(batch_size, num_landmarks, segments, -1)
                    #q_landmarks = query_reshaped.mean(dim=2)
                    q_landmarks = self._sample_landmarks_rrls(self, query_reshaped, num_landmarks)

                # Print shapes for debugging
                #print(f"Query landmarks shape: {q_landmarks.shape}, Key landmarks shape: {k_landmarks.shape}")
                
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
