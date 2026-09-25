"""
Build the pretraining data:
  1. stream general Chinese web text from HuggingFace (default: FineWeb-2, cmn_Hani)
  2. train a byte-level BPE tokenizer on a slice of it + the ruozhiba corpus
  3. tokenize (in parallel) into flat uint16 token files: pretrain/train.bin, pretrain/val.bin
     Documents are separated by <|bos|>.

Run scripts/prepare_ruozhiba.py first: its raw text is mixed into the tokenizer
and (a few times over) into the pretraining tokens so the base model already
"speaks 贴吧".

Usage:
  python -m scripts.prepare_pretrain                              # 300M tokens, enough for d6 (~1.5 epochs)
  python -m scripts.prepare_pretrain --max_tokens 50_000_000      # quick smoke test
"""
import os
import json
import time
import argparse
import itertools
import multiprocessing as mp

import numpy as np

from core.common import get_path, print0, save_json
from core.tokenizer import RuozhiTokenizer

DATASETS = {
    # name: (hf repo, config, text field)
    "fineweb2": ("HuggingFaceFW/fineweb-2", "cmn_Hani", "text"),                 # general web, closest to 贴吧 style
    "fineweb-edu-zh": ("opencsg/Fineweb-Edu-Chinese-V2.1", None, "text"),        # educational, cleaner
    "wiki": ("wikimedia/wikipedia", "20231101.zh", "text"),                     # encyclopedic, mixed 繁/简
}

# Characters that only appear in Traditional Chinese. Docs with many of them are
# skipped so the small vocab / model focuses on Simplified Chinese (like 弱智吧).
TRAD_CHARS = set("們這個說為會來時對國學經發與後過還從開關長門見問間實現點裡當體麼應沒種頭動機樣讓將電話邊東聽車書氣場師語認讀寫錢買賣請應該義務產業歲數進總統處於辦變權華網際區隊員陽節灣")


def keep_doc(text, min_chars, max_trad_ratio):
    if len(text) < min_chars:
        return False
    sample = text[:2000]
    n_trad = sum(1 for ch in sample if ch in TRAD_CHARS)
    return n_trad / len(sample) <= max_trad_ratio


def stream_docs(dataset, shard_index=0, num_shards=1, min_chars=50, max_trad_ratio=0.005):
    from datasets import load_dataset
    repo, config, field = DATASETS[dataset]
    ds = load_dataset(repo, config, split="train", streaming=True)
    if num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_index)
    for row in ds:
        text = row[field]
        if keep_doc(text, min_chars, max_trad_ratio):
            yield text


def load_ruozhiba_corpus():
    path = get_path("ruozhiba_corpus.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run `python -m scripts.prepare_ruozhiba` first")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


# ------------------------------------------------------------------ workers
def tokenize_worker(args):
    """Stream shard `idx`, tokenize, append to its own .bin until `budget` tokens."""
    idx, num_shards, dataset, budget, out_path, val_path, val_docs, batch_docs = args
    tok = RuozhiTokenizer.load(os.path.dirname(get_path("tokenizer", "x")))
    n_tokens, n_docs, t0 = 0, 0, time.time()
    docs = stream_docs(dataset, idx, num_shards)
    if val_path is not None:  # worker 0 carves off the validation docs first
        val_batch = list(itertools.islice(docs, val_docs))
        ids = tok.encode_batch(val_batch, prepend_bos=True)
        np.array([t for x in ids for t in x], dtype=np.uint16).tofile(val_path)
    with open(out_path, "wb") as f:
        while n_tokens < budget:
            batch = list(itertools.islice(docs, batch_docs))
            if not batch:
                break
            ids = tok.encode_batch(batch, prepend_bos=True)
            flat = np.array([t for x in ids for t in x], dtype=np.uint16)
            flat.tofile(f)
            n_tokens += len(flat)
            n_docs += len(batch)
            if idx == 0:
                el = time.time() - t0
                print0(f"[worker 0] {n_tokens / 1e6:.1f}M / {budget / 1e6:.1f}M tokens, {n_docs} docs, {n_tokens / el / 1e3:.0f}k tok/s/worker")
    return n_tokens, n_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="fineweb2", choices=list(DATASETS))
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument("--tok_train_chars", type=int, default=40_000_000, help="web characters used to train the tokenizer")
    parser.add_argument("--max_tokens", type=int, default=300_000_000, help="web tokens to write to train.bin")
    parser.add_argument("--val_docs", type=int, default=2000)
    parser.add_argument("--ruozhiba_repeat", type=int, default=4, help="times the ruozhiba corpus is appended to train.bin")
    parser.add_argument("--num_workers", type=int, default=max(1, min(8, (os.cpu_count() or 2))))
    parser.add_argument("--batch_docs", type=int, default=512)
    parser.add_argument("--force_tokenizer", action="store_true", help="retrain tokenizer even if one exists")
    args = parser.parse_args()

    tok_dir = os.path.dirname(get_path("tokenizer", "x"))
    out_dir = os.path.dirname(get_path("pretrain", "x"))
    ruozhiba = load_ruozhiba_corpus()

    # ---------------------------------------------------------- 1. tokenizer
    retrain = args.force_tokenizer or not os.path.exists(os.path.join(tok_dir, "tokenizer.json"))
    if retrain:
        print0(f"training tokenizer (vocab={args.vocab_size}) on {args.tok_train_chars / 1e6:.0f}M web chars + ruozhiba corpus")

        def text_iter():
            n = 0
            for text in stream_docs(args.dataset):
                yield text[:10000]
                n += min(len(text), 10000)
                if n >= args.tok_train_chars:
                    break
            # twice, so 弱智吧 slang gets merged into tokens
            yield from ruozhiba
            yield from ruozhiba

        t0 = time.time()
        tok = RuozhiTokenizer.train_from_iterator(text_iter(), args.vocab_size)
        tok.save(tok_dir)
        print0(f"tokenizer trained in {time.time() - t0:.0f}s -> {tok_dir}")
    tok = RuozhiTokenizer.load(tok_dir)
    if not retrain and tok.vocab_size != args.vocab_size:
        raise SystemExit(f"existing tokenizer has vocab {tok.vocab_size} but --vocab_size is {args.vocab_size}; "
                         f"add --force_tokenizer to retrain it (old token files and checkpoints become unusable)")
    assert tok.vocab_size < 2**16, "tokens are stored as uint16"
    sample = "为什么我爸妈结婚没有邀请我？只剩一个心脏了还能活吗？"
    ids = tok.encode(sample)
    print0(f"tokenizer check: {len(sample)} chars -> {len(ids)} tokens: {[tok.decode([i]) for i in ids]}")

    # ------------------------------------------------------- 2. tokenize web
    W = args.num_workers
    budget = args.max_tokens // W + 1
    parts = [os.path.join(out_dir, f"train_part{i}.bin") for i in range(W)]
    val_path = os.path.join(out_dir, "val.bin")
    jobs = [(i, W, args.dataset, budget, parts[i], val_path if i == 0 else None, args.val_docs, args.batch_docs) for i in range(W)]
    t0 = time.time()
    if W == 1:
        results = [tokenize_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(W) as pool:
            results = pool.map(tokenize_worker, jobs)
    web_tokens = sum(r[0] for r in results)
    web_docs = sum(r[1] for r in results)
    print0(f"web: {web_tokens / 1e6:.1f}M tokens from {web_docs} docs in {time.time() - t0:.0f}s")

    # ------------------------------------------------- 3. merge + ruozhiba mix
    rz_ids = tok.encode_batch(ruozhiba, prepend_bos=True)
    rz_flat = np.array([t for x in rz_ids for t in x], dtype=np.uint16)
    train_path = os.path.join(out_dir, "train.bin")
    with open(train_path, "wb") as f:
        for p in parts:
            with open(p, "rb") as g:
                while chunk := g.read(1 << 26):
                    f.write(chunk)
            os.remove(p)
        for _ in range(args.ruozhiba_repeat):
            rz_flat.tofile(f)
    n_train = os.path.getsize(train_path) // 2
    n_val = os.path.getsize(val_path) // 2
    meta = {
        "dataset": args.dataset,
        "vocab_size": tok.vocab_size,
        "tokenizer": tok.fingerprint(),
        "train_tokens": n_train,
        "val_tokens": n_val,
        "web_tokens": web_tokens,
        "web_docs": web_docs,
        "ruozhiba_tokens": int(len(rz_flat)) * args.ruozhiba_repeat,
        "chars_per_token_hint": round(len(sample) / len(ids), 3),
    }
    save_json(meta, os.path.join(out_dir, "meta.json"))
    print0(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
