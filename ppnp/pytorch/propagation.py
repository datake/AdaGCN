import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .utils import MixedDropout, sparse_matrix_to_torch

# (1) PPR exact matrix
def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes) # self-loop
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec) # ()**(-1/2)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr) # vec to matrix
    return D_invsqrt_corr @ A @ D_invsqrt_corr


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix) # A_hat = D*(-1/2) * A_loop *  D*(-1/2)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray()) # inverse directly to compute the PPR matrix


class PPRExact(nn.Module): # network f: H -> Z
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        ppr_mat = calc_ppr_exact(adj_matrix, alpha)
        self.register_buffer('mat', torch.FloatTensor(ppr_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions

# (2) PPR Power Iteration: network f: H -> Z^k
class PPRPowerIteration(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None): # plus: niter
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        # buffer: save some tensors in a net
        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', sparse_matrix_to_torch((1 - alpha) * M))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor):
        preds = local_preds # H=f(X)
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * local_preds # core iteration
        return preds[idx]
