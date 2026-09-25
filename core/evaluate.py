"""
Evaluation helpers: bits-per-byte on held-out tokens, and quick qualitative samples.
"""
import math

import torch

BASE_PROMPTS = ["为什么", "我爸", "弱智吧", "今天", "小明"]
CHAT_PROMPTS = [
    ("qa", "只剩一个心脏了还能活吗？"),
    ("qa", "为什么我爸妈结婚的时候没有邀请我？"),
    ("qa", "鸡柳是鸡的哪个部位？"),
    ("joke", "来一条弱智吧金句"),
    ("joke", "讲一个弱智吧的段子"),
    ("continue", "弱智吧标题：我终于扒开雾气\n接着往下说"),
    ("identity", "你是谁？"),
    ("tieba", "加班到十点，老板还说我效率低"),
    ("yinyang", "阴阳怪气地说：你这个方案还有改进空间"),
]


@torch.no_grad()
def evaluate_bpb(model, batches, token_bytes, autocast_ctx):
    """
    Loss in bits per UTF-8 byte of the target text. Unlike per-token loss this is
    independent of the tokenizer's vocab size, so runs with different tokenizers /
    model sizes can be compared (nanochat reports the same metric).
    batches: iterable of (x, y); y == -1 is ignored.
    """
    total_nats, total_bytes = 0.0, 0
    token_bytes = token_bytes.to(next(model.parameters()).device)
    for x, y in batches:
        with autocast_ctx:
            loss = model(x, y, loss_reduction="none").reshape(-1).float()
        y = y.reshape(-1)
        valid = y >= 0
        nbytes = torch.where(valid, token_bytes[y.clamp(min=0)], torch.zeros_like(y))
        counted = nbytes > 0
        total_nats += loss[counted].sum().item()
        total_bytes += nbytes.sum().item()
    if total_bytes == 0:
        return float("nan")
    return total_nats / (math.log(2) * total_bytes)


def sample_base(model, tokenizer, prompts=BASE_PROMPTS, max_new_tokens=48, temperature=0.8, seed=0):
    outs = []
    for i, p in enumerate(prompts):
        ids = tokenizer.encode(p, prepend_bos=True)
        gen = list(model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50,
                                  stop_ids={tokenizer.bos_id}, seed=seed + i))
        outs.append(p + tokenizer.decode(gen))
    return outs


def chat_once(model, tokenizer, messages, max_new_tokens=128, temperature=0.7, top_k=50, top_p=0.95,
              repetition_penalty=1.1, seed=None):
    ids = tokenizer.render_for_completion(messages)
    stop = {tokenizer.assistant_end, tokenizer.bos_id, tokenizer.user_start}
    gen = list(model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                              stop_ids=stop, repetition_penalty=repetition_penalty, seed=seed))
    return tokenizer.decode(gen)


def sample_chat(model, tokenizer, prompts=CHAT_PROMPTS, seed=0):
    return [(task, p, chat_once(model, tokenizer, [{"role": "user", "content": p}], seed=seed + i))
            for i, (task, p) in enumerate(prompts)]
