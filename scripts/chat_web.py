"""
Gradio web UI. On Colab, `--share` prints a public link you can send to friends.

  pip install gradio
  python -m scripts.chat_web --share
"""
import argparse

import torch
import gradio as gr

from core.common import autodetect_device, get_amp
from core.checkpoint import load_model

EXAMPLES = [
    "来一条弱智吧金句",
    "只剩一个心脏了还能活吗？",
    "为什么我爸妈结婚的时候没有邀请我？",
    "弱智吧标题：我终于扒开雾气\n接着往下说",
    "你是谁？",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--source", default="sft", choices=["sft", "base"])
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--device", default="")
    args = p.parse_args()

    device = args.device or autodetect_device()
    autocast_ctx, _, _ = get_amp(device)
    model, tok, meta = load_model(args.source, device, args.depth)
    stop = {tok.assistant_end, tok.bos_id, tok.user_start}

    def respond(message, history, temperature, top_p, repetition_penalty, max_new_tokens):
        # the model is trained single-turn, so history is not fed back in
        ids = tok.render_for_completion([{"role": "user", "content": message}])
        out = []
        with torch.no_grad(), autocast_ctx:
            for t in model.generate(ids, max_new_tokens=int(max_new_tokens), temperature=temperature, top_k=50,
                                    top_p=top_p, repetition_penalty=repetition_penalty, stop_ids=stop):
                out.append(t)
                text = tok.decode(out)
                if not text.endswith("�"):
                    yield text
        yield tok.decode(out)

    demo = gr.ChatInterface(
        respond,
        title=f"弱智吧AI (d{meta['model_config']['n_layer']})",
        description="一个从零训练的小模型：先读通用中文网页，再学习百度弱智吧。回答仅供娱乐。",
        examples=[[e, 0.8, 0.95, 1.1, 200] for e in EXAMPLES],
        additional_inputs=[
            gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature"),
            gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="top_p"),
            gr.Slider(1.0, 1.5, value=1.1, step=0.01, label="repetition penalty"),
            gr.Slider(16, 400, value=200, step=8, label="max new tokens"),
        ],
    )
    demo.queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
