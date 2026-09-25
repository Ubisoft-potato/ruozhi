#!/usr/bin/env bash
# End-to-end: data -> tokenizer -> pretrain -> SFT -> sample.
# DEPTH=6 bash speedrun.sh        (~1.5-2h on a Colab T4 incl. data prep, ~40min on an A100)
set -euo pipefail
DEPTH=${DEPTH:-6}
MAX_TOKENS=${MAX_TOKENS:-300000000}

python -m scripts.prepare_ruozhiba
python -m scripts.prepare_pretrain --max_tokens "$MAX_TOKENS"
python -m scripts.base_train --depth "$DEPTH"
python -m scripts.sft_train --depth "$DEPTH"
python -m scripts.chat_cli --depth "$DEPTH" -p "来一条弱智吧金句"
python -m scripts.chat_cli --depth "$DEPTH" -p "只剩一个心脏了还能活吗？"
