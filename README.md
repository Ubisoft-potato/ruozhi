# 弱智吧AI · ruozhi

一个娱乐向的中文小语言模型：参考 [karpathy/nanochat](https://github.com/karpathy/nanochat)，从零开始**预训练**（通用中文网页）→ **SFT**（百度弱智吧数据），能在 Google Colab 免费 GPU 上一口气跑完。

它学的是这样的对话（SFT 数据示例）：

```
你> 只剩一个心脏了还能活吗？
AI> 能，人本来就只有一个心脏。
你> 来一条弱智吧金句
AI> 老板这豆沙包啥馅的
你> 续写这条弱智吧帖子：太开心了，我妈咪给我买了10克拉的大钻石
AI> 终于可以开一个割玻璃厂了
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ubisoft-potato/ruozhi/blob/main/notebooks/ruozhi_colab.ipynb)

## 流程

| 阶段 | 脚本 | 做什么 |
|---|---|---|
| 1. 弱智吧数据 | `scripts/prepare_ruozhiba.py` | 下载开源弱智吧数据，清洗、去重，生成 SFT 对话（问答 / 金句 / 续写 / 自我介绍） |
| 2. 预训练数据 | `scripts/prepare_pretrain.py` | 流式读取 FineWeb-2 中文，训练 16k byte-level BPE 分词器，编码成 `uint16` token 文件，并混入几遍弱智吧原文 |
| 3. 预训练 | `scripts/base_train.py` | nanochat 结构的 GPT，Muon + AdamW，按 Chinchilla 比例（20 token/参数）自动定训练步数 |
| 4. SFT | `scripts/sft_train.py` | 只在 assistant 部分计算 loss，按任务分别报告验证集 bpb |
| 5. 聊天 | `scripts/chat_cli.py` / `scripts/chat_web.py` | 命令行流式输出 / Gradio 网页（Colab 上 `--share` 可得公开链接） |
| 进阶 | `scripts/scaling_laws.py` | IsoFLOP 扫描，拟合最优参数量与数据量，指导放大模型 |

## 数据

**弱智吧**（SFT；原文也混入预训练与分词器训练）

| 来源 | 内容 | 用途 |
|---|---|---|
| [Leymore/ruozhiba](https://github.com/Leymore/ruozhiba) | 18–21 年度佳帖 1.3k、吧主推荐 2.6k、一般帖子 81.7k（标题 + 摘要） | 金句、续写 |
| [FunnySaltyFish/Better-Ruozhiba](https://github.com/FunnySaltyFish/Better-Ruozhiba) | 1.5k 条人工逐条审核的弱智吧问答 | 问答 |
| [hfl/ruozhiba_gpt4](https://huggingface.co/datasets/hfl/ruozhiba_gpt4)（可选） | 2.4k 条弱智吧问题 + GPT-4o 回答 | 问答 |
| [m-a-p/COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA) `ruozhiba`（可选） | 240 条精选问答 | 问答 |

HuggingFace 上的两个来源访问失败时自动跳过（`--no_hf` 可强制跳过）。问答去重时人工审核过的 Better-Ruozhiba 优先。

处理后的 SFT 数据（默认参数）：

| 任务 | 用户输入 | 助手输出 | 条数 |
|---|---|---|---|
| `qa` | 弱智吧问题 | 认真（但不失幽默）的回答 | 2.7k（训练集重复 3 次） |
| `joke` | “来一条弱智吧金句”等 8 种说法 | 一条帖子 | 34.7k |
| `continue` | 帖子标题 + “接着往下说” | 正文 / 包袱 | 11.0k |
| `identity` | “你是谁？”等 | 手写的自我介绍 | 8（重复 5 次） |

一般帖子只取回复数 ≥ 3 的（`--min_reply`），其余只用于分词器 / 预训练。3% 划为验证集，验证集内容不会出现在预训练语料里。

**对线 / 阴阳怪气**（可选 SFT 混合：`python -m scripts.prepare_duixian`，然后 `python -m scripts.sft_train --extra duixian`）

| 来源 | 内容 | 任务 |
|---|---|---|
| [Orphanage/Baidu_Tieba_SunXiaochuan](https://huggingface.co/datasets/Orphanage/Baidu_Tieba_SunXiaochuan) | 孙吧 2.1k 帖（标题 + 楼主 → 回复） | `tieba` |
| [Orphanage/Baidu_Tieba_KangYaBeiGuo](https://huggingface.co/datasets/Orphanage/Baidu_Tieba_KangYaBeiGuo) | 抗压背锅吧 5.1k 帖 | `tieba` |
| [PostMindLab/ToxiRewriteCN](https://github.com/PostMindLab/ToxiRewriteCN) | 脏话 / 谐音 / emoji 与去毒改写对照，只取 40 字以内的单句 | `yinyang`（正常 → 冲）、`wenming`（冲 → 正常） |
| [cndiandian/zuanbot.com](https://github.com/cndiandian/zuanbot.com)（`db/data.db`） | 祖安语录 1.7k 条（sqlite，`min` 嘴臭 / `max` 问候全家） | `zuan`（“骂我一句”“就这？”等 → 一条骂人话） |

贴吧每帖只取前 8 条可用回复（`--replies_per_thread`，越靠前越是在回楼主），去掉广告 / 签名 / 引用楼层，验证集按帖划分。ToxiCN、COLDataset 是分类数据集，评论大多是针对群体的观点而非对人回复，不适合作为 SFT 回答，所以不使用。祖安语录去掉字符画 / 摩斯码 / 英文 / 重复，每条在训练集里配 2 个不同的提问（`--zuan_upsample`），`max` 级别另有“往死里骂我”等提问；`--zuan_levels min` 只用轻度的，`--no_zuan` 不用。默认不按内容过滤；加 `--filter` 会用 ToxiCN 的群体词库去掉针对群体的仇恨言论，并去掉暴力威胁。

**通用预训练**：默认 [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) 的 `cmn_Hani`（中文网页，口语化，最接近贴吧），并过滤掉繁体为主的文档；也可以 `--dataset fineweb-edu-zh`（[opencsg/Fineweb-Edu-Chinese-V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1)）或 `--dataset wiki`。

> ⚠️ 弱智吧内容是网友创作的谐音梗、冷笑话，部分可能冒犯或低俗；模型输出仅供娱乐，不保证正确。各数据集的许可请以原仓库为准。

## 模型

与 nanochat 基本一致：RoPE、QK-Norm、无参数 RMSNorm、ReLU² MLP、无 bias、embedding 与 lm_head 不共享、logit soft-cap 15、零初始化残差投影；优化器为 Muon（Transformer 块内矩阵）+ AdamW（embedding / lm_head）。

唯一的大小旋钮是 `--depth`：`n_embd = 64 × depth`，head_dim 64。

| depth | 参数量 | Chinchilla token 数 | T4（粗估） | A100（粗估） |
|---|---|---|---|---|
| 4 | 11.5M | 230M | ~15 min | ~3–5 min |
| 6（默认） | 23.2M | 464M | ~1 h | ~10–15 min |
| 8 | 42.0M | 840M | ~2–3 h | ~30–50 min |

耗时只含预训练，按 MFU 20–35% 粗估：小模型矩阵太小，喂不饱 A100，实际以训练日志里的 `mfu` / `eta` 为准。d8 需要 840M token，`prepare_pretrain` 建议给 `--max_tokens 500_000_000`（默认 3 亿会重复近 3 遍）；A100 上可加 `--device_batch_size 128` 减少梯度累积。

精度自动选择：A100 / L4 等用 bf16；T4 用 fp16 + GradScaler；CPU 用 fp32。

## 快速开始

**Colab**：打开 [`notebooks/ruozhi_colab.ipynb`](notebooks/ruozhi_colab.ipynb)，从上到下运行即可。推荐分两段：先在 CPU 运行时准备数据和分词器并存到 Google Drive，再换 A100 运行时从 Drive 恢复后直接训练（notebook 开头有步骤）。

**本地 / 其他 GPU 机器**（用 [uv](https://docs.astral.sh/uv/) 管理依赖）：

```bash
uv sync                           # 按 uv.lock 装好 .venv
source .venv/bin/activate         # 之后的命令都在这个环境里跑；也可以不激活，改用 `uv run python -m ...`
DEPTH=6 bash speedrun.sh          # 数据 → 分词器 → 预训练 → SFT → 采样
```

或者分步：

```bash
python -m scripts.prepare_ruozhiba
python -m scripts.prepare_pretrain --max_tokens 300_000_000      # 快速试跑可用 30_000_000
python -m scripts.base_train --depth 6                           # 断线后加 --resume
python -m scripts.sft_train --depth 6
python -m scripts.chat_cli                                       # 交互聊天
python -m scripts.chat_web --share                               # 网页
```

CPU 上跑通流程（几分钟，只验证代码，模型不会说人话）。在 macOS 上，PyPI 的 torch 就是 CPU + MPS 版本，`uv sync` 后直接能用；Apple Silicon 会自动选 MPS，想强制纯 CPU 可以给训练脚本加 `--device cpu`：

```bash
python -m scripts.prepare_ruozhiba --no_hf
python -m scripts.prepare_pretrain --tok_train_chars 3000000 --max_tokens 3000000 --num_workers 2
python -m scripts.base_train --depth 2 --device_batch_size 8 --total_batch_size 4096 --max_seq_len 256 --num_iterations 100
python -m scripts.sft_train --max_steps 50 --batch_size 16
python -m scripts.chat_cli -p "来一条弱智吧金句"
```

所有产物（原始数据、分词器、token 文件、检查点、日志）在 `artifacts/`，可用环境变量 `RUOZHI_BASE_DIR` 修改。

## 放大模型：scaling law

`scripts/scaling_laws.py` 对每个计算预算 C 与每个 depth 都恰好训练 C FLOPs，得到 IsoFLOP 曲线；每条曲线在 log(参数量) 上拟合抛物线找到最优模型大小，再拟合 N\*(C) ∝ C^a、D\*(C) ∝ C^b，并外推到更大预算：

```bash
python -m scripts.scaling_laws --budgets 1e15 3e15 1e16 --depths 2 3 4 5 6 8
python -m scripts.scaling_laws --analyze_only     # 只重新拟合 / 画图
# 输出 artifacts/scaling/scaling_laws.{png,json}
```

指标使用 **bits per byte**（与词表大小无关，可跨分词器比较）。预训练日志 `artifacts/base_checkpoints/d*/log.jsonl` 里记录了每次评估的 step / tokens / FLOPs / val bpb，也可以直接拿来画 loss-vs-compute 曲线。

放大时的建议：
1. 用 scaling 结果选 depth 和 token 数；`prepare_pretrain --max_tokens` 至少给到所需 token 的一半（重复 ≤2 遍数据问题不大）。
2. 更大的模型可增大 `--vocab_size`（如 `--vocab_size 32768 --force_tokenizer`）和 `--max_seq_len`。改了词表后 token 文件会随之重新生成，但旧的 base / SFT 检查点不能再用，需要从头训练。分词器指纹记录在 `pretrain/meta.json` 和各检查点的 `meta.json` 里，不匹配时 `base_train` / `load_model` 会直接报错；词表大小与已有分词器不一致却没加 `--force_tokenizer` 时，`prepare_pretrain` 也会报错。
3. 可加入更多弱智吧 / 中文对话数据：在 `prepare_ruozhiba.py` 里加来源即可，格式统一为 `{"task", "messages"}`。

## 目录

```
core/              模型与工具
  gpt.py           GPT（nanochat 结构）+ 生成
  muon.py          Muon 优化器
  tokenizer.py     BPE 分词器 + 对话模板
  dataloader.py    预训练 / SFT 数据加载
  evaluate.py      bpb 评估与采样
  checkpoint.py    检查点读写
  common.py        设备、精度、路径
scripts/           各阶段入口（python -m scripts.xxx）
notebooks/         Colab 笔记本
speedrun.sh        一键全流程
```

## 致谢

- [karpathy/nanochat](https://github.com/karpathy/nanochat)：整体结构、模型与训练配方
- [Keller Jordan / modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)：Muon
- 弱智吧的各位吧友，以及 Leymore、FunnySaltyFish 等数据集整理者
