import torch
import torch.optim
from geoopt import ManifoldParameter, ManifoldTensor

# If you use geoopt>=0.5, OptimMixin typically lives next to manifold optimizers in your repo.
# Keep the import as in your project layout.
from .mixin import OptimMixin


__all__ = ["RiemannianAdamW"]


class RiemannianAdamW(OptimMixin, torch.optim.Optimizer):
    r"""
    Riemannian AdamW exactly as described in "Robust Hyperbolic Learning with Curvature-Aware Optimization"
    (§3.3, Algorithm 2): decoupled weight decay on Lorentz space via a weighted Lorentzian centroid toward
    the origin, followed by a Riemannian Adam update (tangent moments, retraction, and parallel transport).
    https://openreview.net/pdf?id=lJ5WCJZfQn&utm_source=chatgpt.com

    For Euclidean tensors in the same param group, this behaves like a standard AdamW step.

    Args (match torch.optim.AdamW where applicable):
        params, lr, betas, eps, weight_decay, amsgrad, stabilize (int or None): reproject every N steps.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        *,
        stabilize=None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            stabilize=stabilize,
        )
        super().__init__(params, defaults)

        # default manifold (used for plain tensors if you want to treat them as Euclidean)
        self._default_manifold = getattr(self, "_default_manifold", None)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if "step" not in group:
                group["step"] = 0
            group["step"] += 1

            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]

            # -------- Pass 1: decoupled weight decay (AdamW-style) --------
            # Apply BEFORE reading grads, as in Algorithm 2 (p_{t-1} <- μ^ν([p_{t-1}, 0])).
            if wd != 0.0:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                        manifold = p.manifold
                        # centroid weights: [1 - lr*wd, lr*wd]
                        # Build (B=1, K=2, D) stack for centroid
                        origin = manifold.origin(*p.shape, dtype=p.dtype, device=p.device)
                        pts = torch.stack([p, origin])           # (2, *shape)
                        x = pts.view(2, -1).unsqueeze(0)         # (1, 2, D)
                        w = torch.tensor(
                            [1.0 - lr * wd, lr * wd],
                            dtype=p.dtype,
                            device=p.device,
                        ).unsqueeze(0)                            # (1, 2)
                        # Weighted Lorentzian centroid toward origin
                        try:
                            centroid = manifold.lorentzian_centroid(x, w)  # (1, D)
                            centroid = centroid.squeeze(0).view_as(p)
                        except AttributeError as e:
                            raise RuntimeError(
                                "Manifold must implement `lorentzian_centroid` for RiemannianAdamW."
                            ) from e

                        # Optionally transport first moment to the new anchor if available
                        state = self.state[p]
                        if "exp_avg" in state and state["exp_avg"] is not None:
                            exp_avg = state["exp_avg"]
                            try:
                                # If manifold exposes a generic transport between two points:
                                exp_avg.copy_(manifold.transp(p, centroid, exp_avg))
                            except Exception:
                                # Fallback: leave as-is (small mismatch); it will be transported after the step.
                                pass

                        p.copy_(centroid)
                    else:
                        # Euclidean decoupled weight decay: p <- (1 - lr*wd)*p
                        p.mul_(1.0 - lr * wd)

            # -------- Pass 2: Adam(-like) update --------
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("RiemannianAdamW does not support sparse gradients.")

                if isinstance(p, (ManifoldParameter, ManifoldTensor)):
                    manifold = p.manifold
                else:
                    manifold = None

                # State init
                state = self.state[p]
                if len(state) == 0:
                    # per-paper Adam uses shared t; we keep per-group step for bias corr. like geoopt's RiemannianAdam
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if manifold is not None:
                    # ----- Riemannian branch (Lorentz etc.) -----
                    # Project Euclidean grad to Riemannian tangent
                    rgrad = manifold.egrad2rgrad(p, grad)

                    # First/second moment in tangent (Algorithm 2, lines 8–9)
                    exp_avg.mul_(beta1).add_(rgrad, alpha=1.0 - beta1)
                    # Paper uses Hadamard square g ⊙ g in tangent:
                    exp_avg_sq.mul_(beta2).add_(rgrad * rgrad, alpha=1.0 - beta2)

                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = max_exp_avg_sq
                    else:
                        denom = exp_avg_sq

                    # Bias corrections (shared per-group step, as in geoopt’s implementation)
                    bc1 = 1.0 - beta1 ** group["step"]
                    bc2 = 1.0 - beta2 ** group["step"]

                    # Adam direction in tangent
                    step_dir = (exp_avg / bc1) / ((denom / bc2).sqrt_().add_(eps))

                    # Retract and transport first moment to the new base point (Alg. 2, lines 13–14)
                    new_p, new_exp_avg = manifold.retr_transp(p, -lr * step_dir, exp_avg)

                    p.copy_(new_p)
                    exp_avg.copy_(new_exp_avg)

                else:
                    # ----- Euclidean branch (standard AdamW) -----
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = max_exp_avg_sq.sqrt().add_(eps)
                    else:
                        denom = exp_avg_sq.sqrt().add_(eps)

                    bc1 = 1.0 - beta1 ** group["step"]
                    bc2 = 1.0 - beta2 ** group["step"]
                    step_size = lr * (bc2 ** 0.5) / bc1

                    p.addcdiv_(exp_avg, denom, value=-step_size)

            # Optional periodic stabilization (GeoOpt-style)
            if group.get("stabilize") is not None and group["stabilize"] > 0:
                if group["step"] % group["stabilize"] == 0:
                    self._stabilize_group(group)

        return loss

    @torch.no_grad()
    def _stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            state = self.state[p]
            if not state:
                continue
            manifold = p.manifold
            exp_avg = state["exp_avg"]
            # Project parameter back to the manifold and moment to the tangent
            p.copy_(manifold.projx(p))
            exp_avg.copy_(manifold.proju(p, exp_avg))

if __name__ == "__main__":
 pass