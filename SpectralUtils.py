import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_laplacian(adj_mat):
    """
    Computes normalized Laplacian L = I - D^{-1/2} A D^{-1/2}
    """
    if type(adj_mat) != sp.coo_matrix:
        adj_mat = sp.coo_matrix(adj_mat)
    
    # Calculate degree matrix D
    degree = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
    
    # Compute normalized adjacency matrix
    norm_adj = adj_mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
    
    # Compute Laplacian L = I - norm_adj
    laplacian = sp.eye(norm_adj.shape[0]) - norm_adj
    return laplacian.tocoo()

def scipy_sparse_to_torch(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

def chebyshev_approx(signal, laplacian, k):
    """
    Chebyshev polynomial approximation.
    signal: (N, D)
    laplacian: sparse (N, N)
    returns list of (N, D) tensors for each order
    """
    T_k = []
    T_k.append(signal)
    if k == 0:
        return T_k
    
    T_k.append(torch.spmm(laplacian, signal))
    for i in range(2, k + 1):
        T_k.append(2 * torch.spmm(laplacian, T_k[-1]) - T_k[-2])
    return T_k

def spectral_band_split(signal, laplacian, k, num_bands):
    """
    Splits the signal into frequency bands using Chebyshev polynomials.
    Returns: (N, num_bands, D)
    """
    T_k = chebyshev_approx(signal, laplacian, k)
    # Simple strategy: linearly combine T_k to form bands
    # For demonstration, we just split the K orders into num_bands evenly
    bands = []
    chunk_size = max(1, (k + 1) // num_bands)
    for i in range(num_bands):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, k + 1)
        if start >= k + 1:
            bands.append(torch.zeros_like(signal))
        else:
            band_signal = sum(T_k[start:end]) / (end - start)
            bands.append(band_signal)
            
    return torch.stack(bands, dim=1) # (N, num_bands, D)

def inverse_spectral_fusion(band_signals, weights=None):
    """
    Reconstructs the spatial signal from bands.
    band_signals: (N, num_bands, D)
    """
    if weights is not None:
        return torch.sum(band_signals * weights.unsqueeze(0).unsqueeze(-1), dim=1)
    return torch.sum(band_signals, dim=1)
