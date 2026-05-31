"""
SoftMuon optimizer utilities and optimizer wrappers. Based on based on https://github.com/microsoft/dion/blob/main/dion/muon.py.
"""

import math
from itertools import chain
from typing import Generator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributions as td
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.optim.optimizer import Optimizer

from .opt_utils import (
    AsyncRuntime,
    AsyncTask,
    create_param_batches,
    pad_batch,
    to_local,
)

# Use fast SVD by https://github.com/fallnlove/polar-svd
from polar_svd import cans_svd
from .scalar_opts import adamw_update_foreach_async


def newton_quantile(
    p: Tensor,
    mu: Tensor,
    sigma: Tensor,
    max_iter: int = 10,
    tol: float = 1e-8,
) -> Tensor:
    """Solve for quantiles of the folded Cauchy distribution using Newton's method."""
    distribution = td.Cauchy(0, 1)
    q = sigma * distribution.icdf((1 + p) / 2) + mu.abs()
    for _ in range(max_iter):
        z1 = (q - mu) / (sigma + tol)
        z2 = (-q - mu) / (sigma + tol)

        F = distribution.cdf(z1) - distribution.cdf(z2) - p

        pdf_term = (distribution.log_prob(z1).exp() +
                    distribution.log_prob(z2).exp()) / sigma

        q = q - F / pdf_term
        q = torch.clamp(q, min=0.0)

    return q


def inverse_soft_clipping(y) -> torch.Tensor:
    """Invert the soft-clipping function x/sqrt(1 + x^2)."""
    y = torch.as_tensor(y)
    return y / (1 - y ** 2) ** 0.5


def zeropower_via_newtonschulz5(G, steps: int):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_regularized_update(G, steps: int = 40, temp: float = 1.0):
    assert G.ndim >= 2
    X = G.to(dtype=torch.bfloat16)

    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    A = X @ X.mT
    I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    A = A + I / (temp**2)

    normA = A.norm(dim=(-2, -1), keepdim=True)
    A = A / normA

    Y = A
    Z = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    for _ in range(steps):
        T = 0.5 * (3 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    out = Z @ X
    out = out / normA.sqrt()

    if transposed:
        out = out.mT

    return out


def _grad_as_2d_for_svd(grad: torch.Tensor) -> torch.Tensor:
    if grad.ndim == 1:
        return grad.unsqueeze(0)
    if grad.ndim == 2:
        return grad
    return grad.reshape(grad.size(0), -1)


class SoftMuon(Optimizer):
    """
    Distributed SoftMuon optimizer for PyTorch FSDP2 & DDP.
    """

    def __init__(
        self,
        param_groups,
        distributed_mesh: Optional[Union[DeviceMesh, ProcessGroup]] = None,
        lr: float = 0.02,
        momentum: float = 0.95,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        epsilon: float = 1e-8,
        sign_iters: Optional[int] = None,
        transition_iters: Optional[int] = None,
        eps: float = 1e-4,
        newton_iters: int = 10,
        tmax: float = 1e9,
        ns_steps: int = 5,
        schedule: Optional[list] = None,
        nesterov: bool = True,
        adjust_lr: Optional[str] = "spectral_norm",
        flatten: bool = False,
        cautious_wd: bool = False,
        headwise=False,  # requires "is_qkv_params" and "attention_head_size" keys in some param_group
        adamw_lr_scale=0.2,
    ):
        if adjust_lr == "rms_norm":
            assert adamw_lr_scale == 1.0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            beta1=adamw_betas[0],
            beta2=adamw_betas[1],
            weight_decay=weight_decay,
            sign_iters=sign_iters,
            transition_iters=transition_iters,
            eps=eps,
            newton_iters=newton_iters,
            tmax=tmax,
            ns_steps=ns_steps,
            schedule=schedule,
            algorithm="muon",
            step=0,
            epsilon=epsilon,
            nesterov=nesterov,
            flatten=flatten,
            cautious_wd=cautious_wd,
            adjust_lr=adjust_lr,
            headwise=headwise,
        )

        super().__init__(param_groups, defaults)
        self.state.setdefault("step", 0)

        if isinstance(distributed_mesh, DeviceMesh):
            if distributed_mesh.ndim != 1:
                raise ValueError("Only 1D DeviceMesh is supported.")
            self._device_rank = distributed_mesh.get_local_rank()
            self._world_size = distributed_mesh.size()
            self._process_group = distributed_mesh.get_group()
        elif isinstance(distributed_mesh, ProcessGroup):
            self._device_rank = dist.get_rank(distributed_mesh)
            self._world_size = dist.get_world_size(distributed_mesh)
            self._process_group = distributed_mesh
        elif distributed_mesh is None:
            self._device_rank = 0
            self._world_size = 1
            self._process_group = None
        else:
            raise TypeError("Invalid distributed_mesh type.")
        self._distributed_mesh = distributed_mesh

        for group in self.param_groups:
            if group["algorithm"] in ["adamw", "lion"]:
                group["lr"] *= adamw_lr_scale

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.state["step"] += 1
        current_iter = self.state["step"]

        muon_groups = []
        adamw_groups = []

        for group in self.param_groups:
            if group["algorithm"] == "muon":
                temperature = None
                si, ti = group["sign_iters"], group["transition_iters"]
                eps, ni, tmax = group["eps"], group["newton_iters"], group["tmax"]
                if si is not None and ti is not None:
                    if current_iter == si + 1:
                        sch = self._get_temperature_schedule(group["params"], ti, eps, ni, tmax)
                        group["schedule"] = sch

                sched = group["schedule"]
                if sched is not None and si is not None and current_iter > si:
                    idx = current_iter - si - 1
                    idx = max(0, min(idx, sched.numel() - 1))
                    temperature = sched[idx].item()

                group["current_temperature"] = temperature
                muon_groups.append(group)
            else:
                adamw_groups.append(group)

        muon_tasks = self._create_soft_muon_tasks(muon_groups)
        adamw_tasks = self._create_adamw_tasks(adamw_groups)

        all_tasks = chain(muon_tasks, adamw_tasks)
        runtime = AsyncRuntime(all_tasks, max_concurrent_tasks=3)
        runtime.run()

        return loss

    def _get_temperature_schedule(
        self,
        params,
        transition_iters: int,
        eps: float = 1e-4,
        newton_iters: int = 10,
        tmax: float = 1e9
    ) -> torch.Tensor:
        """Build a transition temperature schedule from gradient singular values."""
        local_device = (
            params[0].device
            if not isinstance(params[0], DTensor)
            else params[0].to_local().device
        )

        # Collect singular values only on rank 0 to avoid redundant SVDs.
        all_sigmas: List[Tensor] = []
        for p in params:
            grad = p.grad if p.grad is not None else torch.zeros_like(to_local(p))

            if isinstance(grad, DTensor):
                grad_full = grad.full_tensor()
            else:
                grad_full = grad

            if self._device_rank == 0:
                g2d = _grad_as_2d_for_svd(grad_full)
                _, sigma, _ = cans_svd(g2d)
                all_sigmas.append(sigma)

            if isinstance(grad, DTensor):
                # free temporary full tensor reference
                del grad_full

        # Compute robust statistics on rank 0, then broadcast them to others.
        if self._process_group is not None:
            # Prepare placeholders on non-root ranks.
            if self._device_rank == 0:
                if len(all_sigmas) > 0:
                    concatenated = torch.cat(all_sigmas)
                    mu = concatenated.median()
                    sigma = (concatenated - mu).abs().median()
                else:
                    # No singular values found on rank 0; choose safe defaults
                    # that avoid division-by-zero and allow downstream code
                    # to produce a valid schedule.
                    mu = torch.tensor(0.0, device=local_device, dtype=torch.float64)
                    sigma = torch.tensor(1.0, device=local_device, dtype=torch.float64)
            else:
                mu = torch.tensor(0.0, device=local_device, dtype=torch.float64)
                sigma = torch.tensor(0.0, device=local_device, dtype=torch.float64)

            # Broadcast mu and sigma (tensors) from rank 0 to all workers.
            dist.broadcast(mu, src=0, group=self._process_group)
            dist.broadcast(sigma, src=0, group=self._process_group)
        else:
            # Single-process: compute directly and guard empty-case.
            if len(all_sigmas) > 0:
                concatenated = torch.cat(all_sigmas)
                mu = concatenated.median()
                sigma = (concatenated - mu).abs().median()
            else:
                mu = torch.tensor(0.0, device=local_device, dtype=torch.float64)
                sigma = torch.tensor(1.0, device=local_device, dtype=torch.float64)

        # Ensure dtype/device compatibility with probability vector `p` below.
        p = torch.arange(transition_iters, device=local_device, dtype=torch.float64) / transition_iters
        # newton_quantile expects `mu` and `sigma` as tensors with compatible dtype.
        mu = mu.to(p.dtype)
        sigma = sigma.to(p.dtype)

        quantiles_p = newton_quantile(p, mu, sigma, newton_iters)

        # inverse_soft_clipping may return a CPU tensor when given a Python
        # scalar; move it to the same device as `quantiles_p` to avoid device
        # mismatch errors during division.
        isc = inverse_soft_clipping(1 - eps).to(quantiles_p.device).to(quantiles_p.dtype)
        schedule = isc / quantiles_p
        return schedule.clamp(1.0, tmax)

    def _get_or_initialize_state(self, param: Tensor, algo: str) -> dict:
        state = self.state[param]
        if not state:
            state["momentum_buffer" if algo == "muon" else "momentum"] = (
                torch.zeros_like(param)
            )
            if algo == "adamw":
                state["variance"] = torch.zeros_like(param)
        return state

    def _create_soft_muon_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            group_params = [p for p in group["params"] if p.grad is not None]
            if not group_params:
                continue

            muon_update_args = dict(
                lr=group["lr"],
                adjust_lr=group["adjust_lr"],
                beta=group["momentum"],
                weight_decay=group["weight_decay"],
                ns_steps=group["ns_steps"],
                temp=group["current_temperature"],
                nesterov=group["nesterov"],
                flatten=group["flatten"],
                device_rank=self._device_rank,
                world_size=self._world_size,
                process_group=self._process_group,
            )

            for params in create_param_batches(
                group_params, batch_size=self._world_size
            ):
                gradients = [p.grad for p in params]
                states = [self._get_or_initialize_state(p, "muon") for p in params]
                momentums = [s["momentum_buffer"] for s in states]

                is_batch_sharded = False
                is_matrix_sharded = False
                sharded_mesh_dim = None
                sharded_tensor_dim = None

                if isinstance(params[0], DTensor):
                    shard_placements = [
                        (i, p)
                        for i, p in enumerate(params[0].placements)
                        if p.is_shard() and params[0].device_mesh.size(i) > 1
                    ]

                    if not group["flatten"]:
                        matrix_dims = {params[0].ndim - 1, params[0].ndim - 2}
                        is_batch_sharded = any(
                            p.dim not in matrix_dims for _, p in shard_placements
                        )
                        shard_placements = [
                            (i, p) for i, p in shard_placements if p.dim in matrix_dims
                        ]

                    if len(shard_placements) == 1:
                        is_matrix_sharded = True
                        sharded_mesh_dim = shard_placements[0][0]
                        sharded_tensor_dim = shard_placements[0][1].dim

                if is_batch_sharded and not is_matrix_sharded:
                    for x, g, m in zip(params, gradients, momentums):
                        yield AsyncTask(
                            soft_muon_update_batch_async(
                                X=[x], G=[g], M=[m], shard_dim=None, **muon_update_args
                            )
                        )
                else:
                    yield AsyncTask(
                        soft_muon_update_batch_async(
                            X=pad_batch(params, self._world_size),
                            G=pad_batch(gradients, self._world_size),
                            M=pad_batch(momentums, self._world_size),
                            shard_dim=sharded_tensor_dim,
                            **muon_update_args,
                        )
                    )

    def _create_adamw_tasks(
        self, param_groups: List[dict]
    ) -> Generator["AsyncTask", None, None]:
        for group in param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            gradients = [p.grad for p in params]
            states = [self._get_or_initialize_state(p, "adamw") for p in params]
            momentums = [s["momentum"] for s in states]
            variances = [s["variance"] for s in states]

            yield AsyncTask(
                adamw_update_foreach_async(
                    X=to_local(params),
                    G=to_local(gradients),
                    M=to_local(momentums),
                    V=to_local(variances),
                    lr=torch.tensor(group["lr"]),
                    beta1=torch.tensor(group["beta1"]),
                    beta2=torch.tensor(group["beta2"]),
                    weight_decay=torch.tensor(group["weight_decay"]),
                    step=torch.tensor(self.state["step"]),
                    epsilon=torch.tensor(group["epsilon"]),
                    cautious_wd=group["cautious_wd"],
                )
            )


def soft_muon_update_batch_async(
    X: List[Tensor],
    G: List[Tensor],
    M: List[Tensor],
    lr: float,
    beta: float,
    weight_decay: float,
    ns_steps: int,
    temp: Optional[float],
    nesterov: bool,
    flatten: bool,
    adjust_lr: Optional[str],  # How to adjust learning rate
    device_rank: int,
    world_size: int,
    shard_dim: Optional[int] = None,
    process_group: Optional[ProcessGroup] = None,
) -> Generator[None, None, None]:

    U = soft_muon_update_pre_orthogonalize(
        G=to_local(G), M=to_local(M), beta=beta, nesterov=nesterov
    )

    if shard_dim is not None:
        single_matrix_shards = [torch.empty_like(u) for u in U]
        work = dist.all_to_all(
            single_matrix_shards, U, group=process_group, async_op=True
        )
        yield
        work.wait()

        single_matrix = torch.cat(single_matrix_shards, dim=shard_dim)
        single_matrix = soft_muon_newton_schulz(single_matrix, temp, ns_steps, flatten)

        single_matrix_shards = [
            x.contiguous()
            for x in torch.tensor_split(single_matrix, world_size, dim=shard_dim)
        ]
        work = dist.all_to_all(
            U, single_matrix_shards, group=process_group, async_op=True
        )
        yield
        work.wait()

    elif len(U) > 1:
        single_matrix = U[device_rank]
        single_matrix = soft_muon_newton_schulz(single_matrix, temp, ns_steps, flatten)

        U = [torch.empty_like(u) for u in U]
        work = dist.all_gather(
            U, single_matrix.contiguous(), group=process_group, async_op=True
        )
        yield
        work.wait()
    else:
        U[0] = soft_muon_newton_schulz(U[0], temp, ns_steps, flatten)

    if adjust_lr is None:
        adjusted_lr = lr
    elif adjust_lr == "spectral_norm":
        adjusted_lr = adjust_lr_spectral_norm(lr, X[0].shape, flatten=flatten)
    elif adjust_lr == "rms_norm":
        adjusted_lr = adjust_lr_rms_norm(lr, X[0].shape, flatten=flatten)

    soft_muon_update_post_orthogonalize(
        X=to_local(X),
        U=U,
        base_lr=lr,
        adjusted_lr=adjusted_lr,
        weight_decay=weight_decay,
    )


# @torch.compile(fullgraph=True)
def soft_muon_update_pre_orthogonalize(
    G: List[Tensor], M: List[Tensor], beta: float, nesterov: bool
) -> List[Tensor]:
    dtype = M[0].dtype
    G = [g.to(dtype=dtype) for g in G]

    torch._foreach_mul_(M, beta)
    torch._foreach_add_(M, G)

    if nesterov:
        U = torch._foreach_mul(M, beta)
        torch._foreach_add_(U, G)
        torch._foreach_mul_(U, 1.0 - beta)
    else:
        U = [m.clone() for m in M]

    U = [u.to(dtype=torch.bfloat16) for u in U]
    return U


# @torch.compile(fullgraph=True)
def soft_muon_update_post_orthogonalize(
    X: List[Tensor],
    U: List[Tensor],
    base_lr: float,
    adjusted_lr: float,
    weight_decay: float,
):
    if weight_decay > 0.0:
        torch._foreach_mul_(X, 1.0 - base_lr * weight_decay)

    torch._foreach_add_(X, U, alpha=-adjusted_lr)


def soft_muon_newton_schulz(
    X: Tensor, temp: Optional[float], ns_steps: int, flatten: bool
) -> Tensor:
    original_shape = X.shape
    if flatten and X.ndim >= 3:
        X = X.flatten(start_dim=1)
    elif X.ndim >= 4:
        X = X.flatten(end_dim=-3)

    if temp is None:
        out = zeropower_via_newtonschulz5(X, steps=ns_steps)
    else:
        out = muon_regularized_update(X, steps=ns_steps, temp=temp)

    return out.reshape(original_shape)


def adjust_lr_spectral_norm(lr: float, param_shape: torch.Size, flatten: bool) -> float:
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]

    ratio = fan_out / fan_in
    adjusted_ratio = math.sqrt(max(1.0, ratio))
    return lr * adjusted_ratio


def adjust_lr_rms_norm(lr: float, param_shape: torch.Size, flatten: bool) -> float:
    if flatten:
        fan_out = param_shape[0]
        fan_in = math.prod(param_shape[1:])
    else:
        fan_out, fan_in = param_shape[-2:]

    adjusted_ratio = 0.2 * math.sqrt(max(fan_in, fan_out))
    return lr * adjusted_ratio
