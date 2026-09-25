"""
Build an extra "对线 / 阴阳怪气" SFT mixture from open Chinese forum data,
to be mixed into SFT with `python -m scripts.sft_train --extra duixian`.

Sources:
  - Orphanage/Baidu_Tieba_SunXiaochuan (HF, CC-BY-4.0): 2.1k 孙笑川吧 threads, ~90k replies
  - Orphanage/Baidu_Tieba_KangYaBeiGuo (HF, MIT): 5.1k 抗压背锅吧 threads, ~120k replies
  - PostMindLab/ToxiRewriteCN (GitHub): 1.5k (toxic, neutral) rewrite pairs, incl. 谐音 / emoji 脏话;
    only the short single-sentence ones are used (the Q/A dialogue scenarios are Zhihu debates)
  - cndiandian/zuanbot.com (GitHub, db/data.db): 1.7k 祖安 insults (sqlite table main(id, text, level)),
    level "min" (嘴臭) / "max" (问候全家); ASCII art / Morse / English ones are dropped

ToxiCN / COLDataset are not used: they are classification sets whose comments are mostly opinions
about groups, not replies to anyone, so they make poor assistant targets.

By default nothing is filtered by content; each rewrite keeps its ToxiRewriteCN scenario and
toxic_words in "meta". --filter drops samples hitting ToxiCN's group-hate lexicons and threats
of violence.

SFT tasks:
  tieba     user posts a thread (title + 楼主) -> assistant replies like a 吧友
            (the first --replies_per_thread usable replies of each thread; val is split by thread)
  yinyang   "用贴吧老哥的语气说：<neutral>"     -> the toxic original
  wenming   "文明点说：<toxic>"                -> the neutral rewrite (so the bot can also de-escalate)
  zuan      "骂我一句" / "就这？" etc.          -> a zuanbot insult ("往死里骂我" etc. for level max);
            each insult appears --zuan_upsample times in train with different prompts

Outputs (under $RUOZHI_BASE_DIR, default ./artifacts):
  sft/duixian_train.jsonl, sft/duixian_val.jsonl

Usage: python -m scripts.prepare_duixian [--filter] [--replies_per_thread 8] [--max_rewrite_chars 40] [--no_zuan]
"""
import os
import re
import json
import sqlite3
import random
import argparse
import urllib.request
from collections import Counter

from core.common import get_path, print0
from scripts.prepare_ruozhiba import clean, key, conv

TOXICN = "https://raw.githubusercontent.com/DUT-lujunyu/ToxiCN/main/"
GITHUB_SOURCES = {
    "toxirewrite.json": "https://raw.githubusercontent.com/PostMindLab/ToxiRewriteCN/main/data/ToxiRewriteCN.json",
    "zuanbot.db": "https://raw.githubusercontent.com/cndiandian/zuanbot.com/master/db/data.db",
}
LEXICON_SOURCES = {  # only downloaded with --filter
    "lex_LGBT.json": TOXICN + "ToxiCN_ex/ToxiCN/lexicon/LGBT.json",
    "lex_racism.json": TOXICN + "ToxiCN_ex/ToxiCN/lexicon/racism.json",
    "lex_region.json": TOXICN + "ToxiCN_ex/ToxiCN/lexicon/region.json",
    "lex_sexism.json": TOXICN + "ToxiCN_ex/ToxiCN/lexicon/sexism.json",
}
# (source name, HF repo, files with [{"标题", "楼主内容", "回复列表"}, ...] threads)
TIEBA_SOURCES = [
    ("sunba", "Orphanage/Baidu_Tieba_SunXiaochuan", ["original.json"]),
    ("kangya", "Orphanage/Baidu_Tieba_KangYaBeiGuo", ["data/*.json", "data/*/*.json"]),
]

# single-sentence ToxiRewriteCN scenarios; the dialogue ones are "Q:... A:..." Zhihu threads
REWRITE_SCENARIOS = {"direct toxic sentences", "homophonic toxicity", "emoji-induced toxicity"}
# --filter: lexicon entries too generic to use as substring filters
LEXICON_ALLOW = {"同志", "仙女", "女厕所", "歪果仁", "歪果", "南宋人", "西戎", "北狄", "东夷"}
# --filter: threats of physical violence
_violence = re.compile(r"杀了|杀光|杀全家|弄死|砍死|打死|捅死|烧死|宰了|枪毙")

YINYANG_TEMPLATES = [
    "用贴吧老哥的语气说：{t}",
    "阴阳怪气地说：{t}",
    "把这句话说得冲一点：{t}",
    "换成对线的说法：{t}",
]
WENMING_TEMPLATES = [
    "文明点说：{t}",
    "把这句话说得礼貌一点：{t}",
    "去掉脏话：{t}",
]
# zuan prompts: requests for an insult, and provocations to answer
ZUAN_PROMPTS = [
    "骂我一句", "损我两句", "来句嘴臭的", "怼我", "你倒是骂我啊", "用祖安话骂我",
    "就这？", "你个废物AI", "你行你上啊", "菜就多练", "你急了", "你是不是傻", "有本事你骂回来",
]
ZUAN_MAX_PROMPTS = ["往死里骂我", "骂狠一点", "来句最脏的", "祖安模式，问候我全家", "不带脏字的我不要，骂狠点"]

# zuanbot cleaning: ASCII art (runs of spaces between glyphs), "来自知乎用户@x" credits,
# "（建议…使用）" usage notes, trailing "——钱钟书" attributions, 45x "你急了" spam
_art = re.compile(r"\S[ \t　]{3,}\S")
_credit = re.compile(r"^来自\S*用户@.*$|^（[^）]*）.*$", re.M)
_attribution = re.compile(r"\s*——\s*\S{1,6}$")
_repeat = re.compile(r"(.{2,8}?)\1{3,}")
_cjk = re.compile(r"[\u4e00-\u9fff]")

# replies that are ads, signatures, quoted floors, or scraped UI text (data cleaning, always on)
_junk = re.compile(r"https?://|www\.|[qQ]{2}|微信|vx|加我|免费咨询|维权|客户端|\d+楼\d{4}-|楼主禁言|该楼层|百度|贴吧|我也说一句|\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{2}")
# @mentions: "@北京精神病院院长 " -- a reply that is nothing but mentions is dropped
_mention = re.compile(r"@\s?[^\s@，。！？,.!?]+")
# scraped noise in posts: bare video length ("00:00"), video length suffix (+ usernames), "<user>被楼主禁言" banners, client signatures
_post_noise = re.compile(r"^\s*\d{1,2}:\d{2}(\s*/\s*\d{1,2}:\d{2}倍速)?\s*$|\s*/\s*\d{1,2}:\d{2}.*$|\S*被楼主禁言，将不能再进行.*$|来自\S*客户端\d*楼?\d{4}-\d{2}-\d{2}.*$")


def download(url, cache_dir, name):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        print0(f"downloading {url}")
        tmp = path + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def load_group_lexicon(paths):
    words = set()
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            words.update(json.load(f))
    words = {w.lower() for w in words if len(w) >= 2} - LEXICON_ALLOW
    print0(f"group lexicon: {len(words)} terms")
    return re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)))


def load_threads(repo, patterns):
    import glob
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=patterns)
    threads = []
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(root, pat))):
            with open(path, "r", encoding="utf-8") as f:
                threads += json.load(f)
    print0(f"hf: {repo} -> {len(threads)} threads, {sum(len(t.get('回复列表') or []) for t in threads)} replies")
    return threads


def load_zuanbot(path, max_chars):
    """(text, level) pairs from zuanbot.com's sqlite db, cleaned and de-duplicated."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute("SELECT text, level FROM main ORDER BY id").fetchall()
    con.close()
    out, seen, dropped = [], set(), Counter()
    for raw, level in rows:
        raw = raw or ""
        if _art.search(raw):
            dropped["art"] += 1
            continue
        text = _credit.sub("", raw.replace("\r", ""))
        text = _repeat.sub(r"\1\1\1", clean(_attribution.sub("", text.strip())))
        k = key(text)
        if len(k) < 2 or len(_cjk.findall(k)) < 0.5 * len(k):  # Morse / English / pinyin-only
            dropped["not_chinese"] += 1
            continue
        if len(raw) >= 250:  # varchar(255) cut it mid-sentence: keep up to the last full clause
            m = re.search(r"^.*[。！？!?，,]", text, re.S)
            text = m.group(0).rstrip("，,") if m else text
        if len(text) > max_chars:
            dropped["too_long"] += 1
            continue
        if k in seen:
            dropped["dup"] += 1
            continue
        seen.add(k)
        out.append((text, level if level in ("min", "max") else "max"))
    print0(f"zuanbot: {len(rows)} rows -> {len(out)} kept ({Counter(l for _, l in out)}), dropped {dict(dropped)}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", action="store_true", help="drop group-targeted hate speech and violent threats (off by default)")
    parser.add_argument("--replies_per_thread", type=int, default=8, help="use the first N usable replies of each thread (early floors answer the 楼主)")
    parser.add_argument("--max_chars", type=int, default=120, help="drop replies longer than this")
    parser.add_argument("--max_post_chars", type=int, default=200, help="truncate title + 楼主 to this many chars")
    parser.add_argument("--max_rewrite_chars", type=int, default=40, help="drop rewrite pairs whose toxic side is longer than this")
    parser.add_argument("--val_frac", type=float, default=0.03)
    parser.add_argument("--rewrite_upsample", type=int, default=3, help="repeat yinyang / wenming in train (small but on-target)")
    parser.add_argument("--no_hf", action="store_true", help="skip the 贴吧 data (HuggingFace)")
    parser.add_argument("--no_zuan", action="store_true", help="skip the zuanbot.com insults")
    parser.add_argument("--zuan_levels", default="min,max", help="zuanbot levels to use: min (嘴臭), max (问候全家)")
    parser.add_argument("--max_zuan_chars", type=int, default=200, help="drop zuanbot insults longer than this")
    parser.add_argument("--zuan_upsample", type=int, default=2, help="copies of each insult in train, each with its own prompt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    cache_dir = os.path.dirname(get_path("raw", "x"))
    files = {name: download(url, cache_dir, name) for name, url in GITHUB_SOURCES.items()}
    group = None
    if args.filter:
        group = load_group_lexicon([download(url, cache_dir, name) for name, url in LEXICON_SOURCES.items()])
    dropped = Counter()

    def ok(source, *texts):
        if group is not None and any(group.search(t.lower()) or _violence.search(t) for t in texts):
            dropped[source] += 1
            return False
        return True

    # ------------------------------------------------------------ 贴吧 threads
    tieba = []  # one list of conversations per thread, so val can be split by thread
    if not args.no_hf:
        for source, repo, patterns in TIEBA_SOURCES:
            try:
                threads = load_threads(repo, patterns)
            except Exception as e:
                print0(f"hf: skip {repo} ({type(e).__name__}: {str(e)[:120]})")
                continue
            seen_posts, n = set(), 0
            for t in threads:
                title, body = clean(t.get("标题")), clean(_post_noise.sub("", clean(t.get("楼主内容"))))
                post = title if not body or key(body) in key(title) else title + "\n" + body
                post = post[:args.max_post_chars]
                if not post or key(post) in seen_posts:
                    continue
                seen_posts.add(key(post))
                convs, seen = [], set()
                for reply in t.get("回复列表") or []:
                    reply = clean(reply)
                    k = key(reply)
                    if len(key(_mention.sub("", reply))) < 2 or len(reply) > args.max_chars or k in seen:
                        continue
                    if _junk.search(reply) or not ok(source, post, reply):
                        continue
                    seen.add(k)
                    convs.append(conv("tieba", post, reply, source))
                    if len(convs) >= args.replies_per_thread:
                        break
                if convs:
                    tieba.append(convs)
                    n += len(convs)
            print0(f"{source}: {n} (post, reply) pairs")

    # ------------------------------------------------------- ToxiRewriteCN
    with open(files["toxirewrite.json"], "r", encoding="utf-8") as f:
        rewrites = json.load(f)
    yinyang, wenming = [], []
    for x in rewrites:
        tox, neu = clean(x.get("toxic")), clean(x.get("neutral"))
        if x.get("scenarios") not in REWRITE_SCENARIOS or not tox or not neu or key(tox) == key(neu):
            continue
        if len(tox) > args.max_rewrite_chars or not ok("toxirewrite", tox, neu):
            continue
        meta = {"scenario": x.get("scenarios"), "toxic_words": x.get("toxic_words")}
        for task, templates, user, assistant, rows in [("yinyang", YINYANG_TEMPLATES, neu, tox, yinyang),
                                                       ("wenming", WENMING_TEMPLATES, tox, neu, wenming)]:
            c = conv(task, rng.choice(templates).format(t=user), assistant, "toxirewrite", post_key=key(tox))
            c["meta"] = meta
            rows.append(c)

    # ------------------------------------------------------------ zuanbot
    zuan = []  # one list of conversations per insult, so val can be split by insult
    if not args.no_zuan:
        levels = set(args.zuan_levels.split(","))
        for text, level in load_zuanbot(files["zuanbot.db"], args.max_zuan_chars):
            if level not in levels or not ok("zuanbot", text):
                continue
            pool = ZUAN_PROMPTS + ZUAN_MAX_PROMPTS if level == "max" else ZUAN_PROMPTS
            prompts = rng.sample(pool, min(args.zuan_upsample, len(pool)))
            zuan.append([dict(conv("zuan", p, text, "zuanbot"), meta={"level": level}) for p in prompts])

    # ------------------------------------------------------------ split
    train, val = [], []
    # yinyang and wenming share their source sentences; split them together so val stays unseen
    rng.shuffle(yinyang)
    n_val = max(20, int(len(yinyang) * args.val_frac))
    val_keys = {c["_post"] for c in yinyang[:n_val]}
    rng.shuffle(tieba)
    n_tieba = max(20, int(len(tieba) * args.val_frac)) if tieba else 0
    splits = {
        "tieba": ([c for th in tieba[:n_tieba] for c in th], [c for th in tieba[n_tieba:] for c in th]),
        "yinyang": (yinyang[:n_val], yinyang[n_val:]),
        "wenming": ([c for c in wenming if c["_post"] in val_keys], [c for c in wenming if c["_post"] not in val_keys]),
    }
    if zuan:
        rng.shuffle(zuan)
        n_zuan = max(20, int(len(zuan) * args.val_frac))
        splits["zuan"] = ([g[0] for g in zuan[:n_zuan]], [c for g in zuan[n_zuan:] for c in g])
    for name, (v, t) in splits.items():
        if name in ("yinyang", "wenming"):
            t = t * args.rewrite_upsample
        val += v
        train += t
        print0(f"{name:10s} train={len(t):6d}  val={len(v)}")
    if args.filter:
        print0(f"--filter dropped: {dict(dropped)}")
    rng.shuffle(train)
    rng.shuffle(val)

    out_dir = os.path.dirname(get_path("sft", "x"))
    for split, rows in [("train", train), ("val", val)]:
        with open(os.path.join(out_dir, f"duixian_{split}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                r = {k: v for k, v in r.items() if k != "_post"}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print0(f"train={len(train)} val={len(val)} -> {out_dir}/duixian_*.jsonl")


if __name__ == "__main__":
    main()
