"""
Supervised fine-tuning on the 弱智吧 conversation mixture (see prepare_ruozhiba.py).
Loss is only computed on assistant tokens.

Usage:
  python -m scripts.sft_train                  # latest base checkpoint
  python -m scripts.sft_train --depth 6 --num_epochs 3
"""
import os
import json
import time
import argparse
from collections import defaultdict

import torch

from core.common import get_path, print0, seed_everything, autodetect_device, get_amp
from core.checkpoint import load_model, checkpoint_dir, save_checkpoint
from core.dataloader import SFTLoader, load_conversations
from core.evaluate import evaluate_bpb, sample_chat


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=None, help="base checkpoint depth (default: latest)")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32, help="conversations per step")
    p.add_argument("--init_lr_frac", type=float, default=0.3, help="fraction of the base-training learning rates")
    p.add_argument("--embedding_lr", type=float, default=0.2)
    p.add_argument("--unembedding_lr", type=float, default=0.004)
    p.add_argument("--matrix_lr", type=float, default=0.02)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=-1, help="for quick tests")
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)
    device = args.device or autodetect_device()
    autocast_ctx, amp_dtype, needs_scaler = get_amp(device)
    print0(f"device={device} amp={amp_dtype} grad_scaler={needs_scaler}")

    model, tokenizer, base_meta = load_model("base", device, args.depth)
    depth = base_meta["model_config"]["n_layer"]
    max_len = base_meta["model_config"]["sequence_len"]
    token_bytes = torch.tensor(tokenizer.token_bytes(), dtype=torch.long)
    print0(f"loaded base d{depth} (step {base_meta['step']}, val bpb {base_meta.get('val_bpb', float('nan')):.4f})")

    train_convs = load_conversations(get_path("sft", "train.jsonl"))
    val_convs = load_conversations(get_path("sft", "val.jsonl"))
    train_loader = SFTLoader(train_convs, tokenizer, args.batch_size, max_len, device, seed=args.seed)
    val_by_task = defaultdict(list)
    for c in val_convs:
        val_by_task[c["task"]].append(c)
    val_loaders = {t: SFTLoader(cs, tokenizer, args.batch_size, max_len, device, shuffle=False) for t, cs in val_by_task.items()}
    task_counts = defaultdict(int)
    for c in train_convs:
        task_counts[c["task"]] += 1
    print0(f"train: {len(train_convs)} conversations {dict(task_counts)}, {train_loader.num_target_tokens() / 1e6:.2f}M target tokens/epoch")
    print0(f"val: { {t: len(v) for t, v in val_by_task.items()} }")

    num_steps = len(train_loader) * args.num_epochs
    if args.max_steps > 0:
        num_steps = min(num_steps, args.max_steps)
    optimizers = model.setup_optimizers(args.unembedding_lr, args.embedding_lr, args.matrix_lr, args.weight_decay)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] = group["initial_lr"] * args.init_lr_frac
    scaler = torch.amp.GradScaler("cuda") if needs_scaler else None

    def run_eval():
        model.eval()
        res = {t: evaluate_bpb(model, iter(loader), token_bytes, autocast_ctx) for t, loader in val_loaders.items()}
        model.train()
        return res

    def show_samples():
        model.eval()
        with autocast_ctx:
            for task, prompt, reply in sample_chat(model, tokenizer):
                print0(f"  [{task}] {prompt!r} -> {reply!r}")
        model.train()

    print0(f"steps: {num_steps}")
    res = run_eval()
    print0("step 00000 | val bpb " + " ".join(f"{t}={v:.4f}" for t, v in res.items()))
    model.train()
    step, t_start = 0, time.time()
    done = False
    for epoch in range(args.num_epochs):
        for x, y in train_loader:
            lrm = 1.0 - step / num_steps  # linear decay to 0
            for opt in optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * lrm
            with autocast_ctx:
                loss = model(x, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                for opt in optimizers:
                    scaler.unscale_(opt)
            else:
                loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            for opt in optimizers:
                scaler.step(opt) if scaler is not None else opt.step()
            if scaler is not None:
                scaler.update()
            model.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t_start
                print0(f"epoch {epoch} step {step:05d}/{num_steps} | loss {loss.item():.4f} | lrm {lrm:.2f} | eta {(num_steps - step) * el / step / 60:.1f}min")
            if step % args.eval_every == 0 or step == num_steps:
                res = run_eval()
                print0(f"step {step:05d} | val bpb " + " ".join(f"{t}={v:.4f}" for t, v in res.items()))
            if step >= num_steps:
                done = True
                break
        if done:
            break

    show_samples()
    meta = {"step": step, "model_config": base_meta["model_config"], "val_bpb": res,
            "base_step": base_meta["step"], "args": vars(args)}
    save_checkpoint(checkpoint_dir("sft", depth), model, meta)


if __name__ == "__main__":
    main()
