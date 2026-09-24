"""
GPT model, closely following nanochat's architecture:
- rotary position embeddings (no learned positions)
- QK norm
- parameter-free RMSNorm
- ReLU^2 MLP, no biases
- untied token embedding / lm_head
- logit soft-capping
- Multi-Query / Grouped-Query attention optional (n_kv_head)

A single knob, `depth`, sets the model size (see GPTConfig.from_depth), which is
what makes simple scaling-law sweeps possible later.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    sequence_len: int = 512
    vocab_size: int = 16384
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 384

    @classmethod
    def from_depth(cls, depth, vocab_size, sequence_len, aspect_ratio=64, head_dim=64):
        # nanochat: model_dim = depth * aspect_ratio; heads sized ~head_dim
        n_embd = depth * aspect_ratio
        n_head = max(1, n_embd // head_dim)
        return cls(sequence_len=sequence_len, vocab_size=vocab_size, n_layer=depth,
                   n_head=n_head, n_kv_head=n_head, n_embd=n_embd)

    def to_dict(self):
        return asdict(self)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, H, T, D)
        if self.n_kv_head != self.n_head:
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # rotary cache, over-allocated so generation can run a bit past sequence_len
        self.rotary_seq_len = config.sequence_len * 4
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() == 2 and "wte" not in name:
                fan_out, fan_in = p.shape
                std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                nn.init.normal_(p, mean=0.0, std=std)
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        # zero-init the residual projections and the classifier (nanochat)
        nn.init.zeros_(self.lm_head.weight)
        for block in self.transformer.h:
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.zeros_(block.mlp.c_proj.weight)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]  # (1, T, 1, D/2)

    # ----------------------------------------------------------- accounting
    def num_params(self, exclude_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.transformer.wte.weight.numel()
        return n

    def estimate_flops_per_token(self):
        """6N (fwd+bwd matmuls, excluding the embedding lookup) + attention term."""
        nparams = self.num_params(exclude_embedding=True)
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        return 6 * nparams + 12 * l * h * q * t

    # ------------------------------------------------------------ optimizer
    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        from ruozhi.muon import Muon
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        # nanochat: AdamW lrs are tuned at d=768 and scaled by 1/sqrt(dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        fused = next(self.parameters()).is_cuda
        adamw = torch.optim.AdamW(adam_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay, fused=fused)
        muon = Muon(matrix_params, lr=matrix_lr, momentum=0.95)
        optimizers = [adamw, muon]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    # -------------------------------------------------------------- forward
    def forward(self, idx, targets=None, loss_reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1), f"sequence too long: {T} > {self.cos.size(1)}"
        cos_sin = (self.cos[:, :T].to(idx.device), self.sin[:, :T].to(idx.device))
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = norm(x)
        softcap = 15.0
        logits = self.lm_head(x)
        logits = softcap * torch.tanh(logits.float() / softcap)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=loss_reduction)
        return loss

    # ----------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, ids, max_new_tokens=128, temperature=0.8, top_k=50, top_p=None,
                 stop_ids=(), repetition_penalty=1.0, seed=None):
        """
        Simple generation without KV cache (the model is tiny, so this is fine).
        Yields token ids one at a time so callers can stream.
        """
        device = self.transformer.wte.weight.device
        rng = None
        if seed is not None:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = list(ids)
        max_ctx = self.config.sequence_len
        for _ in range(max_new_tokens):
            ctx = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=device)
            logits = self(ctx)[0, -1, :].float()
            if repetition_penalty != 1.0:
                prev = torch.tensor(sorted(set(ids[-max_ctx:])), device=device)
                vals = logits[prev]
                logits[prev] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
            if temperature <= 0:
                next_id = int(torch.argmax(logits))
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[-1]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                if top_p is not None and top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sp, dim=-1)
                    sp[(cum - sp) > top_p] = 0
                    probs = torch.zeros_like(probs).scatter_(0, si, sp)
                    probs = probs / probs.sum()
                next_id = int(torch.multinomial(probs, 1, generator=rng))
            if next_id in stop_ids:
                break
            ids.append(next_id)
            yield next_id
