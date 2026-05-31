import math

import torch


def explicit3(A_val, B_val, dtype=torch.float64, device="cpu"):
    # explicit formula for optimal 3-rd order polynomial on the segment [A, B]
    A_t = torch.as_tensor(A_val, dtype=torch.float64, device=device)
    B_t = torch.as_tensor(B_val, dtype=torch.float64, device=device)

    e = torch.sqrt((A_t**2 + A_t * B_t + B_t**2) / 3.0)
    a = 2.0 / (2.0 * e**3 + A_t**2 * B_t + B_t**2 * A_t)

    # Replaces: p = jnp.array([-a, 0, a * (A**2 + A * B + B**2), 0])[::-1]
    p1 = a * (A_t**2 + A_t * B_t + B_t**2)
    p3 = -a

    p = torch.stack([torch.zeros_like(p1), p1, torch.zeros_like(p3), p3])

    err = (2.0 * e**3 - A_t**2 * B_t - B_t**2 * A_t) / (
        2.0 * e**3 + A_t**2 * B_t + B_t**2 * A_t
    )
    return p.to(dtype), err.to(dtype)


def delta_orthogonalization(
    n=1, degree=3, delta=0.3, B=1.0, dtype=torch.float32, device="cpu"
):
    # find composition of n polynomials of specified degree on the interval
    # [0, B], which falls into [1-delta, 1+delta]
    # the derivative of composition at zero is maximized
    if degree != 3:
        raise NotImplementedError

    Al = 0.0
    Ar = float(B)
    e = 100.0

    # Using standard python loop instead of jax.lax.while_loop
    while abs(e - delta) > 1e-7:
        a_val = (Al + Ar) / 2.0
        b_val = float(B)
        lst = []

        for _ in range(n):
            Q, e_tensor = explicit3(a_val, b_val, dtype=torch.float64, device=device)
            lst.append(Q)
            e_val = e_tensor.item()
            a_val, b_val = 1.0 - e_val, 1.0 + e_val

        e = e_val

        if e < delta:
            Ar = (Ar + Al) / 2.0
        else:
            Al = (Al + Ar) / 2.0

    p_stack = torch.stack([i.to(dtype) for i in lst])
    final_AB = torch.tensor((Al + Ar) / 2.0, dtype=dtype, device=device)
    return p_stack, final_AB


def cans_iteration(
    A,
    max_iter=50,
    a=1e-8,
    degree=3,
    preprocess=True,
    preprocess_iters=2,
    delta=0.99,
    tol=1e-5,
):
    if degree != 3:
        raise NotImplementedError

    device = A.device
    dtype = A.dtype

    n_start_0, n_start_1 = A.shape
    if n_start_0 < n_start_1:
        A = A.T
    else:
        A = A.clone()

    b = 1.0  # assume that matrix is normalized
    err = 100000.0
    n = A.shape[1]
    id_mat = torch.eye(n, dtype=dtype, device=device)
    A2 = A.T @ A

    # matrix_norm with ord=1 gets maximum absolute column sum
    one_norm = torch.linalg.matrix_norm(A, ord=1)
    inf_norm = torch.linalg.matrix_norm(A, ord=float("inf"))

    # Short circuit to prevent rsqrt from encountering NaNs/Infs
    if one_norm == 0.0:
        alpha_inverse = torch.tensor(1.0, dtype=dtype, device=device)
    else:
        alpha_inverse = torch.rsqrt(one_norm) * torch.rsqrt(inf_norm)

    A2 = A2 * (alpha_inverse**2)
    A = A * alpha_inverse

    if preprocess:
        lst, _ = delta_orthogonalization(
            preprocess_iters, degree, delta, B=1.0, dtype=dtype, device=device
        )

        for i in range(preprocess_iters):
            A3 = A @ A2
            A = lst[i][1] * A + lst[i][3] * A3
            A2 = A.T @ A

        a_val, b_val = 1.0 - delta, 1.0 + delta
    else:
        a_val, b_val = a, b

    cnt = 0
    # Frobenius norm is the default without arguments
    err = (torch.linalg.norm(A2 - id_mat) / math.sqrt(n)).item()

    # Iteration block replaces jax.lax.while_loop
    while cnt < max_iter and err > tol:
        A3 = A @ A2
        p, e = explicit3(a_val, b_val, dtype=dtype, device=device)

        e_val = e.item()
        a_val, b_val = 1.0 - e_val, 1.0 + e_val

        A = p[1] * A + p[3] * A3
        A2 = A.T @ A

        err = (torch.linalg.norm(A2 - id_mat) / math.sqrt(n)).item()
        cnt += 1

    if n_start_0 < n_start_1:
        return A.T
    return A
