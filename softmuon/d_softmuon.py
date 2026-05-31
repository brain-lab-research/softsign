"""
SoftMuon optimizer utilities and optimizer wrappers. Based on based on https://github.com/toothacher17/Megatron-LM/tree/moonshot/distributedmuon-impl.
"""
import math
from typing import Dict, Optional, Sequence, Tuple, Union
from torch import Tensor

import torch
import torch.distributed as dist
import torch.distributions as td

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
    params: Sequence[torch.nn.Parameter],
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
        all_sigmas.append(s)

    all_sigmas = torch.cat(all_sigmas)
    mu = all_sigmas.median()
    sigma = (all_sigmas - mu).abs().median()

    p = torch.arange(transition_iters, device=device, dtype=torch.float64) / transition_iters
    quantiles_p = newton_quantile(p, mu, sigma, newton_iters)
    schedule = inverse_soft_clipping(1 - eps) / quantiles_p

    schedule = schedule.clamp(1.0, tmax)
    return schedule

def normalize_range(range_pair: Tuple[int, int], start: int) -> Tuple[int, int]:
    """Convert a global index range to a local range relative to a start offset."""
    return (range_pair[0] - start, range_pair[1] - start)


class MuonDistMeta:
    # which buffer and bucket param belongs to
    buffer_idx: int = 0
    bucket_idx: int = 0
    # param shape after tp
    shape: torch.Size = None
    # param location in global buffer
    global_range: Tuple[int, int] = None
    tp_split_dim: int = -1
    # param location in global buffer (current dp slice)
    local_range: Tuple[int, int] = None

    def __init__(
        self,
        buffer_idx: int,
        bucket_idx: int,
        shape: torch.Size,
        global_range: Tuple[int, int],
        tp_split_dim: int,
    ):
        self.buffer_idx = buffer_idx
        self.bucket_idx = bucket_idx
        self.shape = shape
        self.global_range = global_range
        self.tp_split_dim = tp_split_dim

    def set_local_buffer_range(self, local_buffer_range: Tuple[int, int]):
        start = max(self.global_range[0], local_buffer_range[0])
        end = min(self.global_range[1], local_buffer_range[1])
        self.local_range = (
            (start, end)
            if start < end
            else (local_buffer_range[0], local_buffer_range[0])
        )


# adjust LR based on: https://github.com/MoonshotAI/Moonlight
def adjust_lr_wd_for_muon(
    lr: float,
    matched_adamw_rms: float,
    param_shape: Tuple[int, ...],
) -> float:
    """Scale the Muon learning rate to match the RMS of a reference AdamW update."""
    A, B = param_shape[:2]
    adjusted_ratio = math.sqrt(max(A, B)) * matched_adamw_rms
    adjusted_lr = lr * adjusted_ratio
    return adjusted_lr


# copy from https://github.com/KellerJordan/Muon/tree/master and support distributed solution
class DistributedMuon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        param_groups: The parameters to be optimized.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        matched_adamw_rms: The AdamW Update RMS that Muon is designed to match. (0.2~0.4 recommended)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (5 is probably always enough)
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        param_groups,
        lr=2e-2,
        weight_decay=0.1,
        matched_adamw_rms=0.2,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            matched_adamw_rms=matched_adamw_rms,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        super().__init__(param_groups, defaults)
        self.distributed_mode = False
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for group in self.param_groups:
            for p in group["params"]:
                # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
                if p.ndim >= 2 and p.size(0) < 10000:
                    self.state[p]["use_muon"] = True
                else:
                    self.state[p]["use_muon"] = False

    def enable_distributed_mode(
        self,
        global_buffer_sizes,
        dist_group,
        tp_group,
        dist_metas: Dict[torch.nn.Parameter, MuonDistMeta],
    ):
        """Configure optimizer state for distributed parameter buffering.

        Args:
            global_buffer_sizes: sizes and offsets for each global buffer bucket.
            dist_group: process group used for optimizer sharding.
            tp_group: tensor-parallel process group for parameter gathering.
            dist_metas: distributed metadata for each parameter.
        """

        self.global_buffer_sizes = global_buffer_sizes
        self.dist_group = dist_group
        self.tp_group = tp_group
        self.dist_metas = dist_metas

        world_size = dist.get_world_size(dist_group)
        rank = dist.get_rank(dist_group)

        # calc local buffer range
        self.local_buffer_sizes = []
        self.local_buffer_ranges = []
        for bucket_sizes in global_buffer_sizes:
            local_bucket_sizes = []
            local_bucket_ranges = []
            for global_bucket_size, bucket_offset in bucket_sizes:
                assert global_bucket_size % world_size == 0
                local_buffer_size = global_bucket_size // world_size
                local_buffer_start = local_buffer_size * rank + bucket_offset
                local_buffer_range = (
                    local_buffer_start,
                    local_buffer_start + local_buffer_size,
                )
                local_bucket_sizes.append(local_buffer_size)
                local_bucket_ranges.append(local_buffer_range)

            self.local_buffer_sizes.append(local_bucket_sizes)
            self.local_buffer_ranges.append(local_bucket_ranges)

        # calc local range for params
        for dist_meta in dist_metas.values():
            local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                dist_meta.bucket_idx
            ]
            dist_meta.set_local_buffer_range(local_buffer_range)

        self.distributed_mode = True

    def step(self):
        dtype = torch.bfloat16
        device = torch.cuda.current_device()

        ns_inputs = {}

        # update muon momentum first
        for group in self.param_groups:
            momentum = group["momentum"]
            params = group["params"]

            for p in params:
                if not self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                # 1-dim grad for distributed mode
                assert self.distributed_mode or g.dim() == 2

                # prepare muon buffer in state
                state = self.state[p]
                if not "muon_buffer" in state:
                    state["muon_buffer"] = torch.zeros_like(g)
                buf = state["muon_buffer"]
                buf.mul_(momentum).add_(g)

                # save to ns input
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                ns_inputs[p] = g.bfloat16()

        # rewrite ns_inputs if distributed
        if self.distributed_mode:
            # initialize buffers
            ns_input_local_buffers = [
                [
                    torch.empty((local_buffer_size), device=device, dtype=dtype)
                    for local_buffer_size in local_bucket_sizes
                ]
                for local_bucket_sizes in self.local_buffer_sizes
            ]
            ns_input_global_buffers = [
                [
                    torch.empty((global_buffer_size), device=device, dtype=dtype)
                    for (global_buffer_size, bucket_offset) in global_bucket_sizes
                ]
                for global_bucket_sizes in self.global_buffer_sizes
            ]

            # fill ns input data to local buffer
            for param, ns_input in ns_inputs.items():
                dist_meta = self.dist_metas[param]
                ns_input_local_buffer = ns_input_local_buffers[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                local_range = normalize_range(
                    dist_meta.local_range, local_buffer_range[0]
                )
                ns_input_local_buffer[local_range[0] : local_range[1]].copy_(
                    ns_input.view(-1)
                )

            # all gather buffers
            for ns_input_global_buffer, ns_input_local_buffer in zip(
                ns_input_global_buffers, ns_input_local_buffers
            ):
                for ns_input_global_bucket, ns_input_local_bucket in zip(
                    ns_input_global_buffer, ns_input_local_buffer
                ):
                    dist.all_gather_into_tensor(
                        ns_input_global_bucket,
                        ns_input_local_bucket,
                        group=self.dist_group,
                    )

            # overwrite ns input
            for p in ns_inputs.keys():
                dist_meta = self.dist_metas[p]
                ns_input_global_buffer = ns_input_global_buffers[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                global_range = dist_meta.global_range
                offset = self.global_buffer_sizes[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ][1]
                ns_inputs[p] = ns_input_global_buffer[
                    global_range[0] - offset : global_range[1] - offset
                ].view(dist_meta.shape)

            # set tp info
            tp_world_size = dist.get_world_size(self.tp_group)
            tp_rank = dist.get_rank(self.tp_group)

        # update muon momentum first
        for group in self.param_groups:
            # if not group.get('use_muon', False):
            #     continue

            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            matched_adamw_rms = group["matched_adamw_rms"]
            params = group["params"]

            for p in params:
                if not self.state[p].get("use_muon", False):
                    continue

                ns_input = ns_inputs[p]
                tp_split_dim = -1

                if self.distributed_mode:
                    dist_meta = self.dist_metas[p]
                    tp_split_dim = dist_meta.tp_split_dim

                # gather tensor parallel ( if tp )
                if tp_split_dim != -1:
                    ns_input_shards = [
                        torch.empty_like(ns_input) for _ in range(tp_world_size)
                    ]
                    dist.all_gather(ns_input_shards, ns_input, self.tp_group)
                    ns_input = torch.cat(ns_input_shards, dim=tp_split_dim)

                # calc update
                update = zeropower_via_newtonschulz5(ns_input, steps=ns_steps)

                # only local tp part
                if tp_split_dim != -1:
                    update = update.chunk(tp_world_size, dim=tp_split_dim)[tp_rank]

                # only local buffer part
                if self.distributed_mode:
                    local_range_in_global_range = normalize_range(
                        dist_meta.local_range, dist_meta.global_range[0]
                    )
                    update = update.reshape(-1)[
                        local_range_in_global_range[0] : local_range_in_global_range[1]
                    ]

                # apply weight decay
                p.data.mul_(1 - lr * weight_decay)

                #  adjust lr and apply update
                adjusted_lr = adjust_lr_wd_for_muon(
                    lr, matched_adamw_rms, ns_input.shape
                )
                p.data.add_(update, alpha=-adjusted_lr)

        # use adam for other params
        for group in self.param_groups:
            # if group.get('use_muon', False):
            #     continue

            # init step
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            step = group["step"]
            params = group["params"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in params:
                if self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                state = self.state[p]

                if "adamw_exp_avg" not in state:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)

                buf1 = state["adamw_exp_avg"]
                buf2 = state["adamw_exp_avg_sq"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)


class DistributedSoftMuon(DistributedMuon):

    def __init__(
        self,
        param_groups,
        sign_iters: Optional[int] = None,
        transition_iters: Optional[int] = None,
        eps: float = 1e-4,
        newton_iters: int = 10,
        tmax: float = 1e9,
        patch_heavyball_to_ema=False,  # NEW
        **kwargs,
    ):
        super().__init__(param_groups, **kwargs)
        self.sign_iters = sign_iters
        self.transition_iters = transition_iters
        self.eps = eps
        self.newton_iters = newton_iters
        self.tmax = tmax
        self._temperature_schedule = None
        self._step_count = 0
        self._patch_heavyball_pending = patch_heavyball_to_ema  # NEW

    def step(self):
        # NEW
        if self._patch_heavyball_pending:
            for group in self.param_groups:
                beta = group["momentum"]
                for p in group["params"]:
                    if self.state[p].get("use_muon", False) and "muon_buffer" in self.state[p]:
                        self.state[p]["muon_buffer"].mul_(1 - beta)
            self._patch_heavyball_pending = False

        self._step_count += 1
        current_iter = self._step_count

        # --- Temperature schedule logic ---
        temperature = None
        si, ti = self.sign_iters, self.transition_iters
        if si is not None and ti is not None:
            if current_iter == si + 1:
                muon_params = []
                for group in self.param_groups:
                    for p in group["params"]:
                        if self.state[p].get("use_muon", False):
                            muon_params.append(p)
                self._temperature_schedule = get_temperature_schedule(muon_params, ti, self.eps, self.newton_iters, self.tmax)
                if dist.is_initialized() and dist.get_world_size() > 1:
                    dist.broadcast(self._temperature_schedule.contiguous(), src=0)

            if self._temperature_schedule is not None and current_iter > si:
                idx = current_iter - si - 1
                idx = max(0, min(idx, self._temperature_schedule.numel() - 1))
                temperature = self._temperature_schedule[idx]

        # --- Muon momentum update ---
        dtype = torch.bfloat16
        device = torch.cuda.current_device()
        ns_inputs = {}

        for group in self.param_groups:
            momentum = group["momentum"]
            params = group["params"]

            for p in params:
                if not self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                assert self.distributed_mode or g.dim() == 2

                state = self.state[p]
                if "muon_buffer" not in state:
                    state["muon_buffer"] = torch.zeros_like(g)
                buf = state["muon_buffer"]
          
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                g = g * (1 - momentum)
                ns_inputs[p] = g.bfloat16()
                

        # --- Distributed allgather (if enabled) ---
        if self.distributed_mode:
            ns_input_local_buffers = [
                [
                    torch.empty((local_buffer_size), device=device, dtype=dtype)
                    for local_buffer_size in local_bucket_sizes
                ]
                for local_bucket_sizes in self.local_buffer_sizes
            ]
            ns_input_global_buffers = [
                [
                    torch.empty((global_buffer_size), device=device, dtype=dtype)
                    for (global_buffer_size, bucket_offset) in global_bucket_sizes
                ]
                for global_bucket_sizes in self.global_buffer_sizes
            ]

            for param, ns_input in ns_inputs.items():
                dist_meta = self.dist_metas[param]
                ns_input_local_buffer = ns_input_local_buffers[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                local_range = normalize_range(
                    dist_meta.local_range, local_buffer_range[0]
                )
                ns_input_local_buffer[local_range[0] : local_range[1]].copy_(
                    ns_input.view(-1)
                )

            for ns_input_global_buffer, ns_input_local_buffer in zip(
                ns_input_global_buffers, ns_input_local_buffers
            ):
                for ns_input_global_bucket, ns_input_local_bucket in zip(
                    ns_input_global_buffer, ns_input_local_buffer
                ):
                    dist.all_gather_into_tensor(
                        ns_input_global_bucket,
                        ns_input_local_bucket,
                        group=self.dist_group,
                    )

            for p in ns_inputs.keys():
                dist_meta = self.dist_metas[p]
                ns_input_global_buffer = ns_input_global_buffers[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ]
                global_range = dist_meta.global_range
                offset = self.global_buffer_sizes[dist_meta.buffer_idx][
                    dist_meta.bucket_idx
                ][1]
                ns_inputs[p] = ns_input_global_buffer[
                    global_range[0] - offset : global_range[1] - offset
                ].view(dist_meta.shape)

            tp_world_size = dist.get_world_size(self.tp_group)
            tp_rank = dist.get_rank(self.tp_group)

        # --- Apply NS or regularized update for muon params ---
        for group in self.param_groups:
            lr = group["lr"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            matched_adamw_rms = group["matched_adamw_rms"]
            params = group["params"]

            for p in params:
                if not self.state[p].get("use_muon", False):
                    continue

                ns_input = ns_inputs[p]
                tp_split_dim = -1

                if self.distributed_mode:
                    dist_meta = self.dist_metas[p]
                    tp_split_dim = dist_meta.tp_split_dim

                if tp_split_dim != -1:
                    ns_input_shards = [
                        torch.empty_like(ns_input) for _ in range(tp_world_size)
                    ]
                    dist.all_gather(ns_input_shards, ns_input, self.tp_group)
                    ns_input = torch.cat(ns_input_shards, dim=tp_split_dim)

                if temperature is None:
                    update = zeropower_via_newtonschulz5(ns_input, steps=ns_steps)
                else:
                    update = muon_regularized_update(
                        ns_input, steps=ns_steps, temp=float(temperature)
                    )

                if tp_split_dim != -1:
                    update = update.chunk(tp_world_size, dim=tp_split_dim)[tp_rank]

                if self.distributed_mode:
                    local_range_in_global_range = normalize_range(
                        dist_meta.local_range, dist_meta.global_range[0]
                    )
                    update = update.reshape(-1)[
                        local_range_in_global_range[0] : local_range_in_global_range[1]
                    ]

                p.data.mul_(1 - lr * weight_decay)
                adjusted_lr = adjust_lr_wd_for_muon(
                    lr, matched_adamw_rms, ns_input.shape
                )
                p.data.add_(update, alpha=-adjusted_lr)

        # --- AdamW for non-muon params ---
        for group in self.param_groups:
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            step = group["step"]
            params = group["params"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in params:
                if self.state[p].get("use_muon", False):
                    continue

                g = p.grad
                assert g is not None
                state = self.state[p]

                if "adamw_exp_avg" not in state:
                    state["adamw_exp_avg"] = torch.zeros_like(g)
                    state["adamw_exp_avg_sq"] = torch.zeros_like(g)

                buf1 = state["adamw_exp_avg"]
                buf2 = state["adamw_exp_avg_sq"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / bias_correction2 ** 0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)
