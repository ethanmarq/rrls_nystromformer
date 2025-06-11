# RRLS
import numpy as np
import scipy.linalg as spl
import time
from tqdm import tqdm
import gc

# Batched RRLS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

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
'''
if __name__ == "__main__":

    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import scipy.io as sio

    global y
    n1 = 100
    n2 = 5000
    n3 = 4900
    n = np.asarray([n1, n2, n3])
    np.random.seed(10)
    X = np.concatenate([np.random.multivariate_normal(mean=[50, 10], cov=np.eye(2), size=(n1,)),
                        np.random.multivariate_normal(mean=[-70, -70], cov=np.eye(2), size=(n2,)),
                        np.random.multivariate_normal(mean=[90, -40], cov=np.eye(2), size=(n3,))], axis=0)
    y = np.concatenate([np.ones((n1,)) * 1,
                        np.ones((n2,)) * 2,
                        np.ones((n3,)) * 3])
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    sio.savemat("data.mat",{'X': X, 'y': y})

    y_list = list()

    iter = tqdm(range(1000))
    for i in iter:
        indices = recursiveNystrom(X, n_components=10, kernel_func=lambda *args, **kwargs: gauss(*args, **kwargs, gamma=0.001), random_state=None)
        #plt.figure(figsize=(16,8))
        #plt.scatter(X[idx[~np.isin(idx, indices)],0], X[idx[~np.isin(idx, indices)],1], marker='.')
        #plt.scatter(X[idx[np.isin(idx, indices)],0], X[idx[np.isin(idx, indices)],1])
        #plt.tight_layout()
        #plt.show()
        #print(np.unique(y[indices], return_counts=True))
        #time.sleep(0.5)
        y_list.append(y[indices])

    y_total = np.concatenate(y_list)
    u,c = np.unique(y_total, return_counts=True)
    iter.close()
    print("Real balance:", n/n.sum())
    print("RLS balance:", c/c.sum())

'''
#####

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
            current_item_np = current_item.cpu().numpy()
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


# --- Debugging Main Function ---
if __name__ == "__main__":
    
    # Define a mock class to hold parameters, simulating a real class structure
    class ModelConfig:
        def __init__(self, device, rrls_gamma, rrls_lmbda_0):
            self.device = device
            self.rrls_gamma = rrls_gamma
            self.rrls_lmbda_0 = rrls_lmbda_0

    # --- Test Case ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dummy batch data
    # (batch_size, sequence_length, feature_dimension)
    X_torch = torch.randn(10, 500, 20, device=device)
    # Add a case with fewer samples than n_components
    X_torch[1, 50:, :] = 0 # Make it shorter effectively
    # Add an empty case
    X_torch[2, :, :] = 0

    n_components = 30
    
    # Instantiate the mock config object
    mock_self = ModelConfig(device=device, rrls_gamma=0.01, rrls_lmbda_0=1e-6)

    print(f"Input tensor shape: {X_torch.shape}")
    print(f"Target landmarks per item: {n_components}")

    # Run the batched sampling function
    k_landmarks = _sample_landmarks_rrls(
        self=mock_self,
        tensor_3d=X_torch,
        target_num_landmarks=n_components
    )

    print(f"Sampled landmarks shape: {k_landmarks.shape}")
    
    # Check for NaNs
    if torch.isnan(k_landmarks).any():
        print("Warning: Sampled landmarks contain NaN values.")
    else:
        print("Successfully sampled landmarks without NaNs.")

    # Verify the output shape
    assert k_landmarks.shape == (X_torch.shape[0], n_components, X_torch.shape[2])
    print("Output shape is correct.")

    print(k_landmarks.shape)
