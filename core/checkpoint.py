"""
Saving / loading model checkpoints.

Layout (under $RUOZHI_BASE_DIR):
  base_checkpoints/d{depth}/model.pt, meta.json, optim.pt (for resuming), log.jsonl
  sft_checkpoints/d{depth}/model.pt, meta.json
"""
import os
import glob

import torch

from core.common import get_path, save_json, load_json, print0
from core.gpt import GPT, GPTConfig
from core.tokenizer import RuozhiTokenizer

SOURCE_DIRS = {"base": "base_checkpoints", "sft": "sft_checkpoints"}


def checkpoint_dir(source, depth):
    return os.path.dirname(get_path(SOURCE_DIRS[source], f"d{depth}", "x"))


def save_checkpoint(ckpt_dir, model, meta, optimizers=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()}
    torch.save(state, os.path.join(ckpt_dir, "model.pt"))
    save_json(meta, os.path.join(ckpt_dir, "meta.json"))
    if optimizers is not None:
        torch.save([o.state_dict() for o in optimizers], os.path.join(ckpt_dir, "optim.pt"))
    print0(f"saved checkpoint -> {ckpt_dir}")


def find_latest_depth(source):
    dirs = glob.glob(os.path.join(os.path.dirname(get_path(SOURCE_DIRS[source], "x")), "d*", "model.pt"))
    if not dirs:
        raise FileNotFoundError(f"no {source} checkpoints found; train one first")
    return max(dirs, key=os.path.getmtime).split(os.sep)[-2][1:]


def load_model(source, device, depth=None):
    """Returns (model, tokenizer, meta). depth=None picks the most recently written checkpoint."""
    if depth is None:
        depth = find_latest_depth(source)
    ckpt_dir = checkpoint_dir(source, depth)
    meta = load_json(os.path.join(ckpt_dir, "meta.json"))
    tokenizer = RuozhiTokenizer.load(os.path.dirname(get_path("tokenizer", "x")))
    tokenizer.check_compatible(meta, f"checkpoint {ckpt_dir}")
    config = GPTConfig(**meta["model_config"])
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device)
    model.load_state_dict(state, strict=True)
    # rotary buffers are not persisted; recompute them on the real device
    cos, sin = model._precompute_rotary(model.rotary_seq_len, config.n_embd // config.n_head)
    model.cos, model.sin = cos.to(device), sin.to(device)
    model.eval()
    return model, tokenizer, meta
