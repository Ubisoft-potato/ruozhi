"""
A small Chinchilla-style scaling-law sweep ("IsoFLOP" curves, like nanochat's scaling_laws).

For every compute budget C and every depth d we pretrain with exactly C FLOPs
(the number of steps is derived from C), and record the final validation bpb.
Per budget, a parabola in log(params) gives the compute-optimal model size N*(C);
fitting N*(C) ~ C^a and D*(C) ~ C^b tells you how to split a bigger budget
between model size and data when you scale up.

  python -m scripts.scaling_laws --budgets 1e15 3e15 1e16 --depths 2 3 4 5 6 8
  python -m scripts.scaling_laws --analyze_only        # re-fit / re-plot existing runs

Rough cost on a T4 (~15 TFLOP/s achieved): 1e16 FLOPs ~ 11 min per run.
Use a data file big enough for the largest budget (tokens = C / flops_per_token).
"""
import os
import sys
import json
import glob
import argparse
import subprocess

import numpy as np

from core.common import get_path, print0, save_json


def run_dir(budget, depth):
    return os.path.join("scaling", f"C{budget:.0e}_d{depth}")


def collect():
    rows = []
    for meta_path in glob.glob(os.path.join(os.path.dirname(get_path("base_checkpoints", "scaling", "x")), "*", "meta.json")):
        with open(meta_path) as f:
            m = json.load(f)
        budget = m["args"]["target_flops"]
        if budget <= 0:
            continue
        rows.append({"budget": budget, "depth": m["model_config"]["n_layer"], "params": m["num_params"],
                     "tokens": m["num_iterations"] * m["args"]["total_batch_size"], "val_bpb": m["val_bpb"]})
    return sorted(rows, key=lambda r: (r["budget"], r["params"]))


def analyze(rows, out_png):
    budgets = sorted(set(r["budget"] for r in rows))
    optima = []
    for C in budgets:
        rs = [r for r in rows if r["budget"] == C]
        x = np.log10([r["params"] for r in rs])
        y = np.array([r["val_bpb"] for r in rs])
        print0(f"C={C:.1e}: " + ", ".join(f"d{r['depth']}({r['params'] / 1e6:.1f}M)={r['val_bpb']:.4f}" for r in rs))
        if len(rs) < 3:
            continue
        a, b, c = np.polyfit(x, y, 2)
        if a <= 0:
            print0("  no minimum inside the depth range; add smaller/larger depths")
            continue
        x_opt = -b / (2 * a)
        if not (x.min() <= x_opt <= x.max()):
            print0("  optimum lies outside the depth range; add smaller/larger depths")
            continue
        n_opt = 10 ** x_opt
        # tokens at the optimum: C = flops_per_token * D, flops_per_token ~ 6N (non-embedding dominated at scale)
        fpt = np.interp(x_opt, x, [C / r["tokens"] for r in rs])
        d_opt = C / fpt
        optima.append({"budget": C, "params_opt": n_opt, "tokens_opt": d_opt, "bpb_opt": float(np.polyval([a, b, c], x_opt))})
        print0(f"  -> optimal params ~{n_opt / 1e6:.1f}M, tokens ~{d_opt / 1e6:.0f}M (ratio {d_opt / n_opt:.1f}), bpb {optima[-1]['bpb_opt']:.4f}")
    fit = {}
    if len(optima) >= 2:
        lc = np.log10([o["budget"] for o in optima])
        fit["N_exponent"], fit["N_logcoef"] = [float(v) for v in np.polyfit(lc, np.log10([o["params_opt"] for o in optima]), 1)]
        fit["D_exponent"], fit["D_logcoef"] = [float(v) for v in np.polyfit(lc, np.log10([o["tokens_opt"] for o in optima]), 1)]
        print0(f"N* ~ C^{fit['N_exponent']:.3f}, D* ~ C^{fit['D_exponent']:.3f}  (Chinchilla: both ~0.5)")
        for C in [1e17, 1e18]:
            n = 10 ** (fit["N_logcoef"] + fit["N_exponent"] * np.log10(C))
            d = 10 ** (fit["D_logcoef"] + fit["D_exponent"] * np.log10(C))
            print0(f"  extrapolated for C={C:.0e}: N*~{n / 1e6:.0f}M params, D*~{d / 1e9:.2f}B tokens")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for C in budgets:
            rs = [r for r in rows if r["budget"] == C]
            axes[0].plot([r["params"] for r in rs], [r["val_bpb"] for r in rs], "o-", label=f"C={C:.0e}")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("params")
        axes[0].set_ylabel("val bpb")
        axes[0].set_title("IsoFLOP curves")
        axes[0].legend()
        if optima:
            axes[1].loglog([o["budget"] for o in optima], [o["params_opt"] for o in optima], "o-", label="N* (params)")
            axes[1].loglog([o["budget"] for o in optima], [o["tokens_opt"] for o in optima], "s-", label="D* (tokens)")
            axes[1].set_xlabel("FLOPs")
            axes[1].set_title("compute-optimal allocation")
            axes[1].legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        print0(f"plot -> {out_png}")
    except ImportError:
        print0("matplotlib not installed; skipping plot")
    return {"runs": rows, "optima": optima, "fit": fit}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=float, nargs="+", default=[1e15, 3e15, 1e16])
    p.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    p.add_argument("--analyze_only", action="store_true")
    p.add_argument("--extra_args", default="", help="passed through to base_train, e.g. '--device_batch_size 16'")
    args = p.parse_args()

    if not args.analyze_only:
        for C in args.budgets:
            for d in args.depths:
                rd = run_dir(C, d)
                if os.path.exists(get_path("base_checkpoints", rd, "meta.json")):
                    print0(f"skip {rd} (done)")
                    continue
                cmd = [sys.executable, "-m", "scripts.base_train", "--depth", str(d), "--target_flops", str(C),
                       "--run_name", rd, "--save_every", "-1", "--sample_every", "-1", "--eval_every", "-1"] + args.extra_args.split()
                print0("$ " + " ".join(cmd))
                subprocess.run(cmd, check=True)
    rows = collect()
    if not rows:
        print0("no scaling runs found")
        return
    result = analyze(rows, get_path("scaling", "scaling_laws.png"))
    save_json(result, get_path("scaling", "scaling_laws.json"))


if __name__ == "__main__":
    main()
