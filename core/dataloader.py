"""
Data loading for pretraining (flat uint16 token files) and SFT (jsonl conversations).
"""
import json
import random

import numpy as np
import torch


class TokenFileLoader:
    """
    Samples random (T+1)-token windows from a flat token file, nanoGPT style.
    Documents are already separated by <|bos|> in the file.
    """

    def __init__(self, path, batch_size, seq_len, device, seed=0):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.B, self.T, self.device = batch_size, seq_len, device
        self.rng = np.random.default_rng(seed)
        assert len(self.data) > seq_len + 1, f"{path} is too small"

    def __len__(self):
        return len(self.data)

    def next_batch(self):
        ix = self.rng.integers(0, len(self.data) - self.T - 1, size=self.B)
        buf = np.stack([self.data[i:i + self.T + 1] for i in ix]).astype(np.int64)
        return self._to_device(buf)

    def iter_sequential(self, max_tokens=None):
        """Deterministic pass over the file (for validation)."""
        n = len(self.data) - 1
        if max_tokens is not None:
            n = min(n, max_tokens)
        step = self.B * self.T
        for start in range(0, n - step + 1, step):
            buf = np.asarray(self.data[start:start + step + 1], dtype=np.int64)
            x = torch.from_numpy(buf[:-1].reshape(self.B, self.T))
            y = torch.from_numpy(buf[1:].reshape(self.B, self.T))
            yield self._move(x), self._move(y)

    def _to_device(self, buf):
        t = torch.from_numpy(buf)
        return self._move(t[:, :-1]), self._move(t[:, 1:])

    def _move(self, t):
        if self.device == "cuda":
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)


def load_conversations(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class SFTLoader:
    """
    Renders conversations with the chat template and yields padded batches.
    Targets are -1 (ignored) everywhere except assistant tokens.
    """

    def __init__(self, conversations, tokenizer, batch_size, max_len, device, shuffle=True, seed=0):
        self.examples = []
        for conv in conversations:
            ids, mask = tokenizer.render_conversation(conv, max_tokens=max_len + 1)
            if sum(mask[1:]) == 0:
                continue  # assistant part got truncated away
            self.examples.append((ids, mask))
        self.B, self.device = batch_size, device
        self.pad_id = tokenizer.assistant_end
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __len__(self):
        return (len(self.examples) + self.B - 1) // self.B

    def num_target_tokens(self):
        return sum(sum(m[1:]) for _, m in self.examples)

    def __iter__(self):
        order = list(range(len(self.examples)))
        if self.shuffle:
            self.rng.shuffle(order)
        for i in range(0, len(order), self.B):
            batch = [self.examples[j] for j in order[i:i + self.B]]
            T = max(len(ids) for ids, _ in batch) - 1
            x = torch.full((len(batch), T), self.pad_id, dtype=torch.long)
            y = torch.full((len(batch), T), -1, dtype=torch.long)
            for r, (ids, mask) in enumerate(batch):
                ids_t = torch.tensor(ids, dtype=torch.long)
                mask_t = torch.tensor(mask[1:], dtype=torch.bool)
                n = len(ids) - 1
                x[r, :n] = ids_t[:-1]
                tgt = ids_t[1:].clone()
                tgt[~mask_t] = -1
                y[r, :n] = tgt
            yield x.to(self.device), y.to(self.device)
