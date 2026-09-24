"""
Chat with the model in the terminal.

  python -m scripts.chat_cli                          # latest SFT model
  python -m scripts.chat_cli -p "来一条弱智吧金句"       # one-shot
  python -m scripts.chat_cli --source base -p "为什么"  # raw completion from the base model
"""
import argparse

import torch

from ruozhi.common import autodetect_device, get_amp
from ruozhi.checkpoint import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="sft", choices=["sft", "base"])
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("-p", "--prompt", default="")
    p.add_argument("-t", "--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--multi_turn", action="store_true", help="keep conversation history (the model is trained single-turn)")
    p.add_argument("--device", default="")
    args = p.parse_args()

    device = args.device or autodetect_device()
    autocast_ctx, _, _ = get_amp(device)
    model, tok, meta = load_model(args.source, device, args.depth)
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                      top_p=args.top_p, repetition_penalty=args.repetition_penalty)

    def stream(ids, stop):
        out, printed = [], ""
        with torch.no_grad(), autocast_ctx:
            for t in model.generate(ids, stop_ids=stop, **gen_kwargs):
                out.append(t)
                text = tok.decode(out)
                if not text.endswith("�"):  # wait until a multi-byte char is complete
                    print(text[len(printed):], end="", flush=True)
                    printed = text
        print()
        return tok.decode(out)

    if args.source == "base":
        while True:
            try:
                prompt = args.prompt or input("\n前缀> ")
            except (EOFError, KeyboardInterrupt):
                break
            print(prompt, end="")
            stream(tok.encode(prompt, prepend_bos=True), {tok.bos_id})
            if args.prompt:
                break
        return

    stop = {tok.assistant_end, tok.bos_id, tok.user_start}
    history = []
    if not args.prompt:
        print(f"弱智吧AI (d{meta['model_config']['n_layer']})  输入 /clear 清空历史, /quit 退出")
    while True:
        try:
            user = args.prompt or input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user in ("/quit", "/exit"):
            break
        if user == "/clear":
            history = []
            continue
        if not user:
            continue
        messages = (history if args.multi_turn else []) + [{"role": "user", "content": user}]
        print("AI> ", end="")
        reply = stream(tok.render_for_completion(messages), stop)
        history = messages + [{"role": "assistant", "content": reply}]
        if args.prompt:
            break


if __name__ == "__main__":
    main()
