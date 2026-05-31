from typing import List, Tuple

import torch

# Assuming cans_iteration is defined/imported here
from .cans_iteration import cans_iteration


def cans_svd(
    matrix: torch.Tensor,
    max_iter: int = 50,
    degree: int = 3,
    preprocess: bool = True,
    preprocess_iters: int = 2,
    delta: float = 0.99,
    a: float = 0.0,
    eigh_impl=None,  # Kept for signature compatibility, unused in PyTorch
    cans_tol: float = 1e-5,
    eps_qr: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the SVD of a matrix using the CANS method.

    Args:
        matrix (torch.Tensor): The input matrix.
        max_iter (int): Number of iterations for the CANS method.
        degree (int): The degree of the polynomial approximation.
        preprocess (bool): Whether to use preprocessing.
        preprocess_iters (int): Number of preprocessing iterations.
        delta (float): The delta parameter for CANS.
        a (float): The scaling parameter for CANS.
        eigh_impl: Algorithm for finding eigh (kept for compatibility).
        cans_tol (float): Tolerance for CANS algorithm.
        eps_qr (float): Tolerance for checking rank deficient cases.
    Returns:
        U (torch.Tensor): Left singular vectors.
        S (torch.Tensor): Singular values.
        Vt (torch.Tensor): Right singular vectors (transposed).
    """

    transposed = False
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
        transposed = True

    W = cans_iteration(
        matrix,
        max_iter=max_iter,
        a=a,
        degree=degree,
        preprocess=preprocess,
        preprocess_iters=preprocess_iters,
        delta=delta,
        tol=cans_tol,
    )

    H = W.T @ matrix

    # JAX's symmetrize_input=True is equivalent to explicit balancing in PyTorch
    H_sym = (H + H.T) / 2.0

    # Note: torch.linalg.eigh returns (eigenvalues, eigenvectors)
    # JAX lax.linalg.eigh returns (eigenvectors, eigenvalues)
    s, V = torch.linalg.eigh(H_sym)

    U = W @ V

    # Reverse the order of singular values and corresponding singular vectors (Ascending -> Descending)
    U = torch.flip(U, dims=[-1])
    V = torch.flip(V, dims=[-1])
    s = torch.flip(s, dims=[-1])

    # This implementation of QR decomposition can change the sign of columns of U,
    # so we need to fix that to ensure singular values are positive.
    # Evaluated using normal if branching rather than jax.lax.cond
    if torch.any(torch.abs(torch.linalg.vector_norm(U, dim=0) - 1.0) > eps_qr):
        U, R = torch.linalg.qr(
            U, mode="reduced"
        )  # mode='reduced' == full_matrices=False
        s = torch.diagonal(R) * s

    # Ensure positive eigenvalues and correct respective V signs
    mask = s < 0
    # Add unsqueeze to broadcast 1D sign boolean array across 2D eigenvectors correctly
    V = torch.where(mask.unsqueeze(0), -V, V)
    s = torch.abs(s)

    # Sort descending
    idx = torch.argsort(s, descending=True)
    U = U[:, idx]
    V = V[:, idx]
    s = s[idx]

    # Handle the respective starting conditions format output
    if not transposed:
        return U, s, V.T
    else:
        return V, s, U.T
