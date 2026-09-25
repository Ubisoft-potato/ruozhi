"""
Pretrain the base model on general Chinese web text (+ a bit of ruozhiba).

Model size is set by a single knob, --depth (nanochat style):
  n_embd = depth * 64, n_head = n_embd / 64
and the number of steps defaults to a Chinchilla-style budget of
  tokens = target_param_data_ratio (20) * num_params.

Examples:
  python -m scripts.base_train --depth 6                  # ~23M params, ~1h on a Colab T4
  python -m scripts.base_train --depth 4 --num_iterations 500   # quick test
  python -m scripts.base_train --depth 2 --device_batch_size 4 --total_batch_size 4096 --max_seq_len 256 --num_iterations 20  # CPU smoke test
"""
import os
import json
import time
import math
import argparse

import torch

from core.common import get_path, load_json, print0, seed_everything, autodetect_device, get_amp, peak_flops
from core.gpt import GPT, GPTConfig
from core.tokenizer import RuozhiTokenizer
from core.dataloader import TokenFileLoader
from core.checkpoint import checkpoint_dir, save_checkpoint
from core.evaluate import evaluate_bpb, sample_base


def get_args():
    p = argparse.ArgumentParser()
    # model
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--aspect_ratio", type=int, default=64, help="n_embd = depth * aspect_ratio")
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--max_seq_len", type=int, default=512)
    # horizon: first of these that is set wins
    p.add_argument("--num_iterations", type=int, default=-1)
    p.add_argument("--target_flops", type=float, default=-1, help="compute budget, for scaling-law sweeps")
    p.add_argument("--target_param_data_ratio", type=float, default=20, help="Chinchilla = 20")
    # batch
    p.add_argument("--device_batch_size", type=int, default=32)
    p.add_argument("--total_batch_size", type=int, default=131072, help="tokens per optimizer step")
    # optimization (nanochat defaults)
    p.add_argument("--embedding_lr", type=float, default=0.2)
    p.add_argument("--unembedding_lr", type=float, default=0.004)
    p.add_argument("--matrix_lr", type=float, default=0.02)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--warmdown_ratio", type=float, default=0.2)
    p.add_argument("--final_lr_frac", type=float, default=0.0)
    # eval / logging
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_tokens", type=int, default=1_048_576)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=1000, help="also saves at the end; -1 = only at the end")
    p.add_argument("--log_every", type=int, default=10)
    # misc
    p.add_argument("--device", default="")
    p.add_argument("--compile", type=int, default=-1, help="-1 = auto (on for CUDA)")
    p.add_argument("--resume", action="store_true", help="resume from base_checkpoints/d{depth}")
    p.add_argument("--run_name", default="", help="checkpoint subdir name, default d{depth}")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)
    device = args.device or autodetect_device()
    autocast_ctx, amp_dtype, needs_scaler = get_amp(device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print0(f"device={device} amp={amp_dtype} grad_scaler={needs_scaler}" + (f" gpu={torch.cuda.get_device_name()}" if device == "cuda" else ""))

    # ------------------------------------------------------------ tokenizer
    tokenizer = RuozhiTokenizer.load(os.path.dirname(get_path("tokenizer", "x")))
    tokenizer.check_compatible(load_json(get_path("pretrain", "meta.json")), "pretrain/train.bin")
    token_bytes = torch.tensor(tokenizer.token_bytes(), dtype=torch.long)

    # ----------------------------------------------------------------- model
    config = GPTConfig.from_depth(args.depth, tokenizer.vocab_size, args.max_seq_len, args.aspect_ratio, args.head_dim)
    model = GPT(config).to(device)
    model.init_weights()
    num_params = model.num_params()
    flops_per_token = model.estimate_flops_per_token()
    print0(f"model config: {json.dumps(config.to_dict())}")
    print0(f"params: {num_params / 1e6:.2f}M total, {model.num_params(exclude_embedding=True) / 1e6:.2f}M non-embedding")
    print0(f"flops/token: {flops_per_token:.3e}")

    # --------------------------------------------------------------- horizon
    tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
    assert args.total_batch_size % tokens_per_fwdbwd == 0, "total_batch_size must be a multiple of device_batch_size * max_seq_len"
    grad_accum = args.total_batch_size // tokens_per_fwdbwd
    if args.num_iterations > 0:
        num_iterations = args.num_iterations
    elif args.target_flops > 0:
        num_iterations = round(args.target_flops / (flops_per_token * args.total_batch_size))
    else:
        num_iterations = round(args.target_param_data_ratio * num_params / args.total_batch_size)
    total_tokens = num_iterations * args.total_batch_size
    print0(f"grad accum steps: {grad_accum}, iterations: {num_iterations}, tokens: {total_tokens / 1e6:.1f}M "
           f"(ratio {total_tokens / num_params:.1f} tokens/param), total flops: {flops_per_token * total_tokens:.3e}")

    # ------------------------------------------------------------------ data
    train_loader = TokenFileLoader(get_path("pretrain", "train.bin"), args.device_batch_size, args.max_seq_len, device, seed=args.seed)
    val_path = get_path("pretrain", "val.bin")
    print0(f"train.bin has {len(train_loader) / 1e6:.1f}M tokens -> ~{total_tokens / len(train_loader):.2f} epochs")
    if total_tokens > 2 * len(train_loader):
        print0("WARNING: training for >2 epochs of data; consider running prepare_pretrain with a larger --max_tokens")

    # ------------------------------------------------------------- optimizer
    optimizers = model.setup_optimizers(args.unembedding_lr, args.embedding_lr, args.matrix_lr, args.weight_decay)
    adamw, muon = optimizers
    scaler = torch.amp.GradScaler("cuda") if needs_scaler else None

    ckpt_dir = checkpoint_dir("base", args.depth) if not args.run_name else os.path.dirname(get_path("base_checkpoints", args.run_name, "x"))
    start_step = 0
    log_path = os.path.join(ckpt_dir, "log.jsonl")
    if args.resume:
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device), strict=True)
        for opt, sd in zip(optimizers, torch.load(os.path.join(ckpt_dir, "optim.pt"), map_location=device)):
            opt.load_state_dict(sd)
        resume_meta = load_json(os.path.join(ckpt_dir, "meta.json"))
        tokenizer.check_compatible(resume_meta, f"checkpoint {ckpt_dir}")
        start_step = resume_meta["step"]
        print0(f"resumed from step {start_step}")
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        open(log_path, "w").close()

    raw_model = model
    use_compile = args.compile if args.compile >= 0 else int(device == "cuda")
    if use_compile:
        model = torch.compile(model, dynamic=False)

    def get_lr_multiplier(it):
        warmup = round(args.warmup_ratio * num_iterations)
        warmdown = round(args.warmdown_ratio * num_iterations)
        if it < warmup:
            return (it + 1) / warmup
        if it <= num_iterations - warmdown:
            return 1.0
        progress = (num_iterations - it) / warmdown
        return progress + (1 - progress) * args.final_lr_frac

    def get_muon_momentum(it):
        frac = min(it / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def run_eval():
        model.eval()
        val_loader = TokenFileLoader(val_path, args.device_batch_size, args.max_seq_len, device)
        bpb = evaluate_bpb(model, val_loader.iter_sequential(args.eval_tokens), token_bytes, autocast_ctx)
        model.train()
        return bpb

    def meta(step, val_bpb):
        return {"step": step, "num_iterations": num_iterations, "val_bpb": val_bpb, "model_config": config.to_dict(),
                "tokenizer": tokenizer.fingerprint(), "num_params": num_params, "flops_per_token": flops_per_token, "args": vars(args)}

    # ------------------------------------------------------------------ loop
    gpu_peak = peak_flops(device)
    x, y = train_loader.next_batch()
    smooth_loss, val_bpb, total_time = 0.0, float("nan"), 0.0
    model.train()
    for step in range(start_step, num_iterations + 1):
        last = step == num_iterations
        if last or (args.eval_every > 0 and step % args.eval_every == 0):
            val_bpb = run_eval()
            print0(f"step {step:05d} | val bpb {val_bpb:.4f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "tokens": step * args.total_batch_size, "flops": step * args.total_batch_size * flops_per_token,
                                    "val_bpb": val_bpb, "train_loss": smooth_loss, "time": total_time}) + "\n")
        if last or (args.sample_every > 0 and step % args.sample_every == 0 and step > 0):
            raw_model.eval()
            with autocast_ctx:
                for s in sample_base(raw_model, tokenizer):
                    print0("  > " + s.replace("\n", "\\n"))
            raw_model.train()
        if last or (args.save_every > 0 and step % args.save_every == 0 and step > start_step):
            save_checkpoint(ckpt_dir, raw_model, meta(step, val_bpb), optimizers)
        if last:
            break

        # ---- one optimizer step
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(grad_accum):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            x, y = train_loader.next_batch()  # prefetch next batch while GPU works
        if scaler is not None:
            for opt in optimizers:
                scaler.unscale_(opt)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        for group in muon.param_groups:
            group["momentum"] = get_muon_momentum(step)
        for opt in optimizers:
            if scaler is not None:
                scaler.step(opt)
            else:
                opt.step()
        if scaler is not None:
            scaler.update()
        model.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        # ---- logging
        loss_f = train_loss.item()
        if not math.isfinite(loss_f):
            print0(f"step {step}: loss is {loss_f}, skipping")
            continue
        smooth_loss = loss_f if step == start_step else 0.9 * smooth_loss + 0.1 * loss_f
        if step > start_step + 5:
            total_time += dt  # skip compile / warmup steps
        if step % args.log_every == 0:
            tok_per_sec = args.total_batch_size / dt
            mfu = f" | mfu {100 * flops_per_token * tok_per_sec / gpu_peak:.1f}%" if gpu_peak else ""
            done = step - start_step + 1
            eta = (num_iterations - step - 1) * (total_time / max(1, done - 6)) / 60 if done > 6 else float("nan")
            print0(f"step {step:05d}/{num_iterations} | loss {smooth_loss:.4f} | lrm {lrm:.2f} | {dt * 1000:.0f}ms | "
                   f"{tok_per_sec / 1e3:.1f}k tok/s{mfu} | eta {eta:.1f}min")

    print0(f"done. final val bpb {val_bpb:.4f}, train time {total_time / 60:.1f}min")
    if device == "cuda":
        print0(f"peak memory {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")


if __name__ == "__main__":
    main()
