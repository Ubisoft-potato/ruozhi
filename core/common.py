"""
Shared utilities: paths, device / precision detection, logging.
"""
import os
import json
import time
import random
from contextlib import nullcontext

import numpy as np
import torch

# All artifacts (data, tokenizer, checkpoints) live under this dir.
# On Colab, point it at Google Drive so work survives a disconnect:
#   export RUOZHI_BASE_DIR=/content/drive/MyDrive/ruozhi
BASE_DIR = os.environ.get("RUOZHI_BASE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"))


def get_path(*parts, mkdir_parent=True):
    path = os.path.join(BASE_DIR, *parts)
    if mkdir_parent:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def print0(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autodetect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supports_bf16(device):
    # T4 (sm75) "supports" bf16 only via slow emulation, so require Ampere+.
    return device == "cuda" and torch.cuda.get_device_capability()[0] >= 8


def get_amp(device):
    """
    Returns (autocast_ctx, amp_dtype, needs_grad_scaler).
    - A100 / L4 / H100: bf16 autocast, no scaler.
    - T4 / V100: fp16 autocast + GradScaler.
    - CPU / MPS: fp32.
    """
    if device == "cuda":
        if supports_bf16(device):
            return torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.bfloat16, False
        return torch.amp.autocast("cuda", dtype=torch.float16), torch.float16, True
    return nullcontext(), torch.float32, False


def peak_flops(device):
    """Rough dense peak (fp16/bf16 tensor core) FLOPs for MFU reporting."""
    if device != "cuda":
        return None
    name = torch.cuda.get_device_name().lower()
    table = {"h100": 989e12, "a100": 312e12, "l4": 121e12, "a10": 125e12, "t4": 65e12, "v100": 125e12, "4090": 165e12, "3090": 71e12}
    for k, v in table.items():
        if k in name:
            return v
    return None


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def lap(self):
        t = time.time()
        dt = t - self.t0
        self.t0 = t
        return dt


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
