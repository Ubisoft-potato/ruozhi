"""
Muon optimizer (single device), from Keller Jordan's modded-nanogpt, as used in nanochat.
Muon = SGD-momentum whose update for each 2D weight is orthogonalized with a
Newton-Schulz iteration. Used for all transformer-block matrices; embeddings and
lm_head use AdamW.
"""
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # bf16 is fast on Ampere+; older GPUs (T4) and CPU stay in fp32
    use_bf16 = G.is_cuda and torch.cuda.get_device_capability(G.device)[0] >= 8
    X = G.bfloat16() if use_bf16 else G.float()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(g.to(p.dtype), alpha=-group["lr"] * scale)
        return loss
