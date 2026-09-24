"""
Download the open-source Baidu 弱智吧 (ruozhiba) datasets and build the SFT mixture.

Sources (GitHub, always used):
  - Leymore/ruozhiba          : ~85k post titles(+abstracts), 2018-2023 annual best posts
  - FunnySaltyFish/Better-Ruozhiba : 1.5k human-reviewed Q&A pairs (question from 弱智吧, sensible answer)
Sources (HuggingFace, optional, used when reachable, e.g. on Colab):
  - hfl/ruozhiba_gpt4         : 2.4k questions from 弱智吧 answered by GPT-4o
  - m-a-p/COIG-CQIA (ruozhiba): ~240 curated Q&A pairs

SFT tasks built from these:
  qa        user asks a 弱智吧 question -> assistant gives a (patient, correct) answer
  joke      user asks for a 弱智吧 金句      -> assistant writes one
  continue  user gives a post title        -> assistant writes the punchline / body
  identity  a few hand-written "who are you" conversations

Outputs (under $RUOZHI_BASE_DIR, default ./artifacts):
  sft/train.jsonl, sft/val.jsonl   one {"task":..., "messages":[...]} per line
  ruozhiba_corpus.jsonl           raw ruozhiba text (minus SFT val), mixed into tokenizer + pretraining data

Usage: python -m scripts.prepare_ruozhiba [--no_hf]
"""
import os
import re
import json
import random
import argparse
import urllib.request

from ruozhi.common import get_path, print0

GITHUB_SOURCES = {
    "title_good": "https://raw.githubusercontent.com/Leymore/ruozhiba/main/data/ruozhiba-title-good.json",
    "title_norm": "https://raw.githubusercontent.com/Leymore/ruozhiba/main/data/ruozhiba-title-norm.json",
    "post_annual": "https://raw.githubusercontent.com/Leymore/ruozhiba/main/data/ruozhiba-post-annual.json",
    "better_qa": "https://raw.githubusercontent.com/FunnySaltyFish/Better-Ruozhiba/main/ruozhiba_qa.json",
}

JOKE_PROMPTS = [
    "来一条弱智吧金句",
    "讲一个弱智吧的段子",
    "说句弱智吧名言",
    "来点弱智吧的东西",
    "发一条弱智吧帖子",
    "整点弱智吧语录",
    "给我来个弱智吧的梗",
    "弱智吧金句来一个",
]

CONTINUE_TEMPLATES = [
    "弱智吧标题：{t}\n接着往下说",
    "{t}\n然后呢？",
    "帖子标题：{t}\n正文是？",
    "续写这条弱智吧帖子：{t}",
]

IDENTITY = [
    ("你是谁？", "我是弱智吧AI，一个读了很多弱智吧帖子的小语言模型。问我问题可以，但别太当真。"),
    ("你叫什么名字", "我叫弱智吧AI。名字里有“智”，说明我很聪明；名字里有“弱”，说明我很谦虚。"),
    ("介绍一下你自己", "我是一个参考 nanochat 从零训练的小模型，先读了一些网络文本，然后专门学习了百度弱智吧的帖子。我能回答弱智吧式的问题，也能给你来几条弱智吧金句。"),
    ("你是谁开发的？", "我是用开源代码从零训练出来的小模型，训练数据来自网络上公开的弱智吧数据集。感谢每一位弱智吧吧友的创作。"),
    ("你能做什么", "我能：1. 认真回答离谱的问题；2. 来一条弱智吧金句；3. 帮你续写弱智吧帖子。别的事情我可能会一本正经地胡说八道。"),
    ("你聪明吗", "我是弱智吧AI，聪明这个词和我之间隔着整整一个弱智吧。"),
    ("hello", "你好！我是弱智吧AI。想听个金句，还是有什么离谱的问题要问？"),
    ("你好", "你好呀！我是弱智吧AI，有什么问题尽管问，答错了算你的。"),
]


def download(url, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, url.split("/")[-1])
    if not os.path.exists(path):
        print0(f"downloading {url}")
        tmp = path + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------- cleaning
_num_prefix = re.compile(r"^\s*\d{1,4}\s*[、.．:：]\s*")
_ws = re.compile(r"[ \t　\xa0]+")


def clean(s):
    if s is None:
        return ""
    s = str(s).replace("\r", "")
    s = _ws.sub(" ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def key(s):
    """normalization used for de-duplication"""
    return re.sub(r"[\s\W_]+", "", s).lower()


def is_trivial(s):
    return len(key(s)) < 2


def post_text(title, abs_):
    """Merge a (possibly truncated) title with its abstract into one post."""
    title, abs_ = clean(title), clean(abs_)
    if is_trivial(abs_):
        return title, None
    kt, ka = key(title), key(abs_)
    if ka.startswith(kt) or kt.startswith(ka) or kt in ka:
        # abstract repeats the (truncated) title -> the abstract is the full post
        return (abs_ if len(abs_) >= len(title) else title), None
    # otherwise the abstract is a separate body / punchline
    return title + "\n" + abs_, abs_


def to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


# ------------------------------------------------------------ hf (optional)
def load_hf_qa():
    """Extra ruozhiba Q&A from HuggingFace. Returns [] if unreachable."""
    pairs = []
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("hfl/ruozhiba_gpt4", "ruozhiba_qa2449_gpt4o.json", repo_type="dataset")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for x in data:
            q = x.get("instruction") or x.get("query") or x.get("question") or ""
            if x.get("input"):
                q = q + "\n" + x["input"]
            a = x.get("output") or x.get("response") or x.get("answer") or ""
            pairs.append((clean(q), clean(a), "hfl_gpt4o"))
        print0(f"hf: hfl/ruozhiba_gpt4 -> {len(data)}")
    except Exception as e:
        print0(f"hf: skip hfl/ruozhiba_gpt4 ({type(e).__name__}: {str(e)[:120]})")
    try:
        from datasets import load_dataset
        ds = load_dataset("m-a-p/COIG-CQIA", "ruozhiba", split="train")
        for x in ds:
            q = clean(x.get("instruction", "")) + ("\n" + clean(x["input"]) if x.get("input") else "")
            pairs.append((q, clean(x.get("output", "")), "coig_cqia"))
        print0(f"hf: m-a-p/COIG-CQIA/ruozhiba -> {len(ds)}")
    except Exception as e:
        print0(f"hf: skip m-a-p/COIG-CQIA ({type(e).__name__}: {str(e)[:120]})")
    return pairs


def conv(task, user, assistant, source, post_key=None):
    # post_key (internal, not written out) links an example back to its corpus line
    return {"task": task, "source": source, "_post": post_key, "messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_hf", action="store_true", help="skip optional HuggingFace sources")
    parser.add_argument("--min_reply", type=int, default=3, help="min #replies for a normal (non-精品) post to be used in SFT")
    parser.add_argument("--max_chars", type=int, default=300, help="drop posts / answers longer than this")
    parser.add_argument("--val_frac", type=float, default=0.03)
    parser.add_argument("--qa_upsample", type=int, default=3, help="repeat QA examples in train (they are few but valuable)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    cache_dir = get_path("raw", "x")
    cache_dir = os.path.dirname(cache_dir)

    raw = {name: download(url, cache_dir) for name, url in GITHUB_SOURCES.items()}

    # ---------------------------------------------------------------- QA
    qa_pairs = [(clean(x["instruction"]), clean(x["output"]), "better_ruozhiba") for x in raw["better_qa"]]
    if not args.no_hf:
        qa_pairs += load_hf_qa()
    qa, seen_q = [], set()
    for q, a, src in qa_pairs:  # Better-Ruozhiba comes first, so it wins on duplicates
        k = key(q)
        if not q or not a or k in seen_q or len(a) > args.max_chars * 2:
            continue
        seen_q.add(k)
        qa.append(conv("qa", q, a, src, post_key=k))

    # ------------------------------------------------- jokes & continuations
    jokes, conts, corpus = [], [], []
    seen_post = set(seen_q)

    def add_post(text, source, title=None, body=None, use_for_sft=True):
        k = key(text)
        if not k or k in seen_post:
            return
        seen_post.add(k)
        corpus.append((k, text))
        if len(text) > args.max_chars or not use_for_sft:
            return
        jokes.append(conv("joke", rng.choice(JOKE_PROMPTS), text, source, post_key=k))
        if title and body and len(key(body)) >= 4:
            conts.append(conv("continue", rng.choice(CONTINUE_TEMPLATES).format(t=title), body, source, post_key=k))

    for x in raw["post_annual"]:
        add_post(_num_prefix.sub("", clean(x["content"])), "annual")
    for x in raw["title_good"]:
        text, body = post_text(x["title"], x["abs"])
        add_post(text, "good", clean(x["title"]), body)
    for x in raw["title_norm"]:
        text, body = post_text(x["title"], x["abs"])
        popular = to_int(x.get("n_reply")) >= args.min_reply
        # unpopular posts only go to the tokenizer corpus
        add_post(text, "norm", clean(x["title"]), body, use_for_sft=popular)

    identity = [conv("identity", q, a, "handwritten") for q, a in IDENTITY]

    # ------------------------------------------------------------ split
    train, val = [], []
    for name, rows in [("qa", qa), ("joke", jokes), ("continue", conts)]:
        rng.shuffle(rows)
        n_val = max(20, int(len(rows) * args.val_frac))
        val += rows[:n_val]
        rows = rows[n_val:]
        if name == "qa":
            rows = rows * args.qa_upsample
        train += rows
        print0(f"{name:10s} total={len(rows) // (args.qa_upsample if name == 'qa' else 1) + n_val:6d}  val={n_val}")
    train += identity * 5
    rng.shuffle(train)
    rng.shuffle(val)

    out_dir = os.path.dirname(get_path("sft", "x"))
    val_keys = {r["_post"] for r in val}
    for split, rows in [("train", train), ("val", val)]:
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                r = {k: v for k, v in r.items() if k != "_post"}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # raw ruozhiba text, one doc per line, for the tokenizer and the pretraining mix.
    # Anything that appears in the SFT val split is left out so val loss stays honest.
    lines = [t for k, t in corpus if k not in val_keys]
    lines += [c["messages"][0]["content"] + "\n" + c["messages"][1]["content"] for c in qa if c["_post"] not in val_keys]
    with open(get_path("ruozhiba_corpus.jsonl"), "w", encoding="utf-8") as f:
        for t in lines:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print0(f"train={len(train)} val={len(val)} corpus_docs={len(lines)} -> {out_dir}")
    for r in val[:6]:
        print0(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
