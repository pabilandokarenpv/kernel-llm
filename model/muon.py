"""
Muon optimizer (Keller Jordan et al., 2024) — orthogonalized momentum SGD.

Used for hidden 2D weight matrices only; AdamW handles embeddings/scalars/
norms per the modded-nanogpt convention this project follows. This is a
compact, faithful port of the public reference implementation, not a novel
contribution — citing it correctly matters more than re-deriving it.
"""

import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate orthogonalization of G via a quintic Newton-Schulz
    iteration — avoids an explicit, expensive SVD."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.float()


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, ns_steps = group["lr"], group["momentum"], group["ns_steps"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                if g.ndim == 2:
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                else:
                    scale = 1.0
                p.add_(g, alpha=-lr * scale)


def split_muon_adamw_params(model):
    """Muon for 2D hidden matrices, AdamW for everything else
    (embeddings, 1D norms/biases, the output head)."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "tok_emb" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params
