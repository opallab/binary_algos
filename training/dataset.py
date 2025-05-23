import torch
from torch.utils.data import Dataset

class PermutationDataset(Dataset):
    def __init__(self, P: torch.Tensor):
        """
        Initialize the dataset with a given permutation matrix P.
        Assumes P is a square matrix of shape (n, n).
        """
        super().__init__()
        assert P.dim() == 2 and P.shape[0] == P.shape[1], "P must be a square matrix."
        self.P = P
        self.n = P.shape[0]
        self.basis = torch.eye(self.n, dtype=P.dtype, device=P.device)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        """
        Return the pair (e_idx, P e_idx):
        - e_idx: standard basis vector with a 1 at position idx.
        - P e_idx: the result of applying the permutation matrix P to e_idx.
        """        
        e_idx = self.basis[idx]
        Pe_idx = self.P @ e_idx

        return e_idx, Pe_idx

def generate_permutation(n, device='cpu'):
    I = torch.eye(n, device=device)
    P = torch.randperm(n, device=device)
    return I[P]

def generate_test(k, num_ones, num_samples, device='cpu'):
    assert num_ones <= k
    x = torch.zeros((num_samples, k), dtype=torch.float32, device=device)
    for xi in x:
        indices = torch.randperm(k)[:num_ones]
        xi[indices] = 1.
    return x

