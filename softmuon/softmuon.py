"""
SoftMuon optimizer utilities and optimizer wrappers. Both distributed and single-device optimizer variants are provided.
"""

import torch
import torch.distributed as dist
import torch.distributions as td
from typing import Optional
from torch import Tensor

# Use fast SVD by https://github.com/fallnlove/polar-svd
from polar_svd import cans_svd


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # Batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # Quintic coefficients selected from Muon tuning.
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def _grad_as_2d_for_svd(grad: torch.Tensor) -> torch.Tensor:
    """Flatten gradients to a 2D matrix for CANS SVD while preserving batch axes."""
    if grad.ndim == 1:
        return grad.unsqueeze(0)
    if grad.ndim == 2:
        return grad
    return grad.reshape(grad.size(0), -1)


def muon_regularized_update(G: torch.Tensor, steps: int = 10, temp: float = 1.0) -> torch.Tensor:
    """Compute a SoftMuon update via matrix inversion iteration."""
    assert G.ndim >= 2

    X = G.bfloat16()
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    A = X @ X.mT
    I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype)
    A = A + I / (temp ** 2)

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


def get_temperature_schedule(
    params,
    transition_iters: int,
    eps: float = 1e-4,
    newton_iters: int = 10,
    tmax: float = 1e9,
) -> torch.Tensor:
    """Build a transition temperature schedule from gradient singular values."""
    device = params[0].device
    all_sigmas = []
    for p in params:
        grad = p.grad if p.grad is not None else torch.zeros_like(p.data)
        g2d = _grad_as_2d_for_svd(grad)
        _, s, _ = cans_svd(g2d)
        all_sigmas.append(s.flatten())

    all_sigmas = torch.cat(all_sigmas)
    mu = all_sigmas.median()
    sigma = (all_sigmas - mu).abs().median()

    p = torch.arange(transition_iters, device=device, dtype=torch.float64) / transition_iters
    quantiles_p = newton_quantile(p, mu, sigma, newton_iters)
    schedule = inverse_soft_clipping(1 - eps) / quantiles_p
    return schedule.clamp(1.0, tmax)


def softmuon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    temp: Optional[float] = None,
) -> torch.Tensor:
    """Compute a SoftMuon parameter update from gradient and momentum buffers."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    if temp is None:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    else:
        update = muon_regularized_update(update, steps=ns_steps, temp=temp)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


def adam_update(
    grad: torch.Tensor,
    buf1: torch.Tensor,
    buf2: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    """Compute the Adam update from gradient moments."""
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SoftMuon(torch.optim.Optimizer):
    """Distributed SoftMuon optimizer."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0,
        momentum: float = 0.95,
        sign_iters: Optional[int] = None,
        transition_iters: Optional[int] = None,
        eps: float = 1e-4,
        newton_iters: int = 10,
        tmax: float = 1e9,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            sign_iters=sign_iters,
            transition_iters=transition_iters,
            eps=eps,
            newton_iters=newton_iters,
            tmax=tmax,
            schedule=None,
        )
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)
        self.state.setdefault('step', 0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.state['step'] += 1
        current_iter = self.state['step']

        for group in self.param_groups:
            params = group["params"]

            temperature = None
            si, ti = group["sign_iters"], group["transition_iters"]
            eps, ni, tmax = group["eps"], group["newton_iters"], group["tmax"]
            if si is not None and ti is not None:
                if current_iter == si + 1:
                    sch = get_temperature_schedule(params, ti, eps, ni, tmax)
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        dist.broadcast(sch.contiguous(), src=0)
                    group["schedule"] = sch
            sched = group["schedule"]
            if sched is not None and si is not None and current_iter > si:
                idx = current_iter - si - 1
                idx = max(0, min(idx, sched.numel() - 1))
                temperature = sched[idx]

            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization for distributed all_gather.
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = softmuon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        temp=temperature,
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceSoftMuon(torch.optim.Optimizer):
    """Single-device SoftMuon optimizer without distributed synchronization."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0,
        momentum: float = 0.95,
        sign_iters: Optional[int] = None,
        transition_iters: Optional[int] = None,
        eps: float = 1e-4,
        newton_iters: int = 10,
        tmax: float = 1e9,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            sign_iters=sign_iters,
            transition_iters=transition_iters,
            eps=eps,
            newton_iters=newton_iters,
            tmax=tmax,
            schedule=None,
        )
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)
        self.state.setdefault('step', 0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.state['step'] += 1
        current_iter = self.state['step']

        for group in self.param_groups:
            params = group["params"]

            temperature = None
            si, ti = group["sign_iters"], group["transition_iters"]
            eps, ni, tmax = group["eps"], group["newton_iters"], group["tmax"]
            if si is not None and ti is not None:
                if current_iter == si + 1:
                    sch = get_temperature_schedule(params, ti, eps, ni, tmax)
                    group["schedule"] = sch
            sched = group["schedule"]
            if sched is not None and si is not None and current_iter > si:
                idx = current_iter - si - 1
                idx = max(0, min(idx, sched.numel() - 1))
                temperature = sched[idx]

            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = softmuon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    temp=temperature,
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        return loss


class SoftMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed SoftMuon variant that runs internal AdamW for non-SoftMuon parameter groups.

    The caller must provide `param_groups` with the `use_muon` flag for each group.
    SoftMuon-compatible groups share the same scheduling behavior as `SoftMuon`.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["sign_iters"] = group.get("sign_iters", None)
                group["transition_iters"] = group.get("transition_iters", None)
                group["eps"] = group.get("eps", 1e-4)
                group["newton_iters"] = group.get("newton_iters", 10)
                group["tmax"] = group.get("tmax", 1e9)
                group["schedule"] = None
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
        super().__init__(param_groups, dict())
        self.state.setdefault('step', 0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.state['step'] += 1
        current_iter = self.state['step']

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]

                temperature = None
                si, ti = group["sign_iters"], group["transition_iters"]
                eps, ni, tmax = group["eps"], group["newton_iters"], group["tmax"]
                if si is not None and ti is not None:
                    if current_iter == si + 1:
                        sch = get_temperature_schedule(params, ti, eps, ni, tmax)
                        if dist.is_initialized() and dist.get_world_size() > 1:
                            dist.broadcast(sch.contiguous(), src=0)
                        group["schedule"] = sch
                sched = group["schedule"]
                if sched is not None and si is not None and current_iter > si:
                    idx = current_iter - si - 1
                    idx = max(0, min(idx, sched.numel() - 1))
                    temperature = sched[idx]

                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)  # Force synchronization for distributed all_gather.
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = softmuon_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                            temp=temperature,
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceSoftMuonWithAuxAdam(torch.optim.Optimizer):
    """Single-device SoftMuon + auxiliary Adam optimizer without distributed sync."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["sign_iters"] = group.get("sign_iters", None)
                group["transition_iters"] = group.get("transition_iters", None)
                group["eps"] = group.get("eps", 1e-4)
                group["newton_iters"] = group.get("newton_iters", 10)
                group["tmax"] = group.get("tmax", 1e9)
                group["schedule"] = None
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
        super().__init__(param_groups, dict())
        self.state.setdefault('step', 0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.state['step'] += 1
        current_iter = self.state['step']

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]

                temperature = None
                si, ti = group["sign_iters"], group["transition_iters"]
                eps, ni, tmax = group["eps"], group["newton_iters"], group["tmax"]
                if si is not None and ti is not None:
                    if current_iter == si + 1:
                        sch = get_temperature_schedule(params, ti, eps, ni, tmax)
                        group["schedule"] = sch
                sched = group["schedule"]
                if sched is not None and si is not None and current_iter > si:
                    idx = current_iter - si - 1
                    idx = max(0, min(idx, sched.numel() - 1))
                    temperature = sched[idx]

                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = softmuon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        temp=temperature,
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss