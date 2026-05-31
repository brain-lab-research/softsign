"""SoftSignum optimizer utilities for temperature schedule computation."""

from typing import List, Optional, Union

import torch
import torch.distributions as td
from torch import Tensor

from torch.optim.optimizer import (
    _use_grad_for_differentiable,
    Optimizer,
)


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


def get_temperature_schedule(
    grad: Tensor,
    transition_iters: int,
    eps: float = 1e-4,
    newton_iters: int = 10,
    tmax: float = 1e9,
) -> Tensor:
    """Build a transition temperature schedule from the gradient distribution."""
    mu = grad.median()
    sigma = (grad - mu).abs().median()
    
    p = torch.arange(transition_iters, device=mu.device) / transition_iters
    quantiles_p = newton_quantile(p, mu, sigma, newton_iters)
    schedule = torch.atanh(torch.tensor(1 - eps)) / quantiles_p

    # Clip temperature to a safe range before applying it.
    schedule.clamp_(1, tmax)

    return schedule


def _single_tensor_softsignum(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: Union[float, Tensor],
    dampening: float,
    nesterov: bool,
    maximize: bool,
    current_iter: int,
    sign_iters: int,
    transition_iters: int,
    eps: float,
    newton_iters: int,
    tmax: float,
    schedule: List[float],
    normalized: bool,
    sign_norm: bool,
    has_sparse_grad: bool,
):
    """Apply the SoftSignum parameter update for a single tensor group."""
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            # Apply Nesterov acceleration only after the momentum buffer is available.
            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf
        grads[i] = grad

    effective_lr = lr
    if normalized or sign_norm:
        norms = [torch.linalg.vector_norm(g) for g in grads]
        total_norm = torch.linalg.vector_norm(torch.stack(norms))

        if normalized:
            effective_lr = lr / total_norm
        elif sign_norm:
            effective_lr = lr * total_norm

    if current_iter - 1 == sign_iters:
        grads_cat = torch.cat([g.reshape(-1).data for g in grads])
        # Use detached flattened gradients so schedule computation does not track autograd.
        schedule = get_temperature_schedule(grads_cat, transition_iters, eps, newton_iters, tmax)

    for i, param in enumerate(params):
        grad = grads[i]
        if current_iter - 1 >= sign_iters:
            temperature = schedule[current_iter - 1 - sign_iters]
            update_vec = torch.tanh(temperature * grad)
        else:
            update_vec = torch.sign(grad)

        param.add_(update_vec, alpha=-effective_lr)

    return schedule


def softsignum(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: Union[float, Tensor],
    dampening: float,
    nesterov: bool,
    maximize: bool,
    current_iter: int,
    transition_iters: int,
    eps: float,
    newton_iters: int,
    sign_iters: int,
    tmax: float,
    schedule: List[float],
    normalized: bool,
    sign_norm: bool
):
    r"""Functional API for SoftSignum optimizer updates."""
    if foreach:
        raise NotImplementedError("`foreach` option is not implemented for SoftSignum")
    if fused:
        raise NotImplementedError("`fused` option is not implemented for SoftSignum")

    schedule = _single_tensor_softsignum(
        params=params,
        grads=d_p_list,
        momentum_buffer_list=momentum_buffer_list,
        grad_scale=grad_scale,
        found_inf=found_inf,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        maximize=maximize,
        current_iter=current_iter,
        sign_iters=sign_iters,
        transition_iters=transition_iters,
        eps=eps,
        newton_iters=newton_iters,
        tmax=tmax,
        schedule=schedule,
        normalized=normalized,
        sign_norm=sign_norm,
        has_sparse_grad=has_sparse_grad,
    )

    return schedule


class SoftSignum(Optimizer):
    """Optimizer that transitions from sign-based updates to sgd-based updates."""

    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        transition_iters: int = 1000,
        eps=1e-4,
        newton_iters=10,
        sign_iters: int = 9000,
        tmax: float = 20.0,
        sign_norm: bool = False,
        normalized: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if sign_norm and normalized:
            raise ValueError(f"sign_norm and normalized are mutually exclusive")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            transition_iters=transition_iters,
            eps=eps,
            newton_iters=newton_iters,
            sign_iters=sign_iters,
            tmax=tmax,
            schedule=None,
            sign_norm=sign_norm,
            normalized=normalized,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)
        self.state.setdefault('step', 0)

    def __setstate__(self, state):
        super().__setstate__(state)
        self.state.setdefault('step', 0)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)
            group.setdefault("sign_norm", False)
            group.setdefault("normalized", False)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        """Collect parameters, gradients, and momentum buffers for a parameter group."""
        has_sparse_grad = False
        for p in group['params']:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True
                state = self.state[p]
                momentum_buffer_list.append(state.get('momentum_buffer'))
        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step for all parameter groups."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.state['step'] += 1
        current_iter = self.state['step']

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            schedule = softsignum(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                current_iter=current_iter,
                transition_iters=group["transition_iters"],
                eps=group["eps"],
                newton_iters=group["newton_iters"],
                sign_iters=group["sign_iters"],
                tmax=group["tmax"],
                schedule=group["schedule"],
                normalized=group["normalized"],
                sign_norm=group["sign_norm"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

            if schedule is not None:
                group["schedule"] = schedule

            if group["momentum"] != 0:
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    self.state[p]["momentum_buffer"] = momentum_buffer
        return loss