"""Plot the eval-sweep CSV -> a score-vs-step figure.

Reads the CSV from scripts/eval_sweep.py and plots each tracked benchmark against
the training step. Perplexity columns (`*_ppl`, lower is better) and accuracy
columns (`*_acc_norm`, higher is better) live on very different scales, so they go
on separate stacked subplots. Columns are discovered from the CSV header, so no
change is needed here when you add a benchmark to eval_sweep.py.

Vertical dashed lines mark epoch boundaries (every --steps-per-epoch steps), so
you can read off how much each extra epoch bought.

Usage
-----
    python scripts/plot_sweep.py                         # log/sweep.csv -> log/sweep.png
    python scripts/plot_sweep.py --csv log/sweep.csv --out log/sweep.png
    python scripts/plot_sweep.py --steps-per-epoch 19073 # mark epoch boundaries
"""
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="log/sweep.csv", help="sweep CSV from eval_sweep.py")
    ap.add_argument("--out", default=None, help="output PNG (default alongside the CSV)")
    ap.add_argument("--steps-per-epoch", type=int, default=19073,
                    help="draw an epoch boundary line every N steps; 0 to disable")
    args = ap.parse_args()

    out_path = args.out or args.csv.replace(".csv", ".png")

    df = pd.read_csv(args.csv).sort_values("step")
    ppl_cols = [c for c in df.columns if c.endswith("_ppl")]
    acc_cols = [c for c in df.columns if c.endswith("_acc_norm")]

    def mark_epochs(ax):
        """Dashed vertical line + label at each epoch boundary within the plot."""
        if args.steps_per_epoch <= 0:
            return
        n = 1
        while args.steps_per_epoch * n <= df["step"].max():
            x = args.steps_per_epoch * n
            ax.axvline(x, color="gray", linestyle="--", alpha=0.5)
            ax.text(x, ax.get_ylim()[1], f" epoch {n}", va="top", fontsize=8, color="gray")
            n += 1

    fig, (ax_ppl, ax_acc) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for c in ppl_cols:
        ax_ppl.plot(df["step"], df[c], marker="o", label=c.replace("_ppl", ""))
    ax_ppl.set_ylabel("perplexity (lower better)")
    ax_ppl.set_title("Benchmark scores over training")
    ax_ppl.grid(True, alpha=0.3)
    ax_ppl.legend()
    mark_epochs(ax_ppl)

    for c in acc_cols:
        ax_acc.plot(df["step"], df[c] * 100, marker="o", label=c.replace("_acc_norm", ""))
    ax_acc.set_ylabel("acc_norm % (higher better)")
    ax_acc.set_xlabel("training step")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()
    mark_epochs(ax_acc)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
