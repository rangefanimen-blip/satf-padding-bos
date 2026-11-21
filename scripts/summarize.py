"""Build report tables/figures from measured trial files, never hardcoded scores."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {"ml1m": "MovieLens-1M", "beauty": "Amazon Beauty", "games": "Amazon Video Games"}
MODEL_LABELS = {"popular":"MostPopular", "zero":"Zero", "pad":"PAD", "bos":"BOS", "sasrec":"SASRec", "pad_balanced":"PAD / sqrt(q)", "start":"Leftmost marker", "zero_scaled":"Zero, scale=0.2", "pad_scaled":"PAD, scale=0.2", "bos_scaled":"BOS, scale=0.2", "pad_scaled_frequency":"PAD, scale=0.2, freq"}

def summarize(root):
    root = Path(root)
    rows = []
    for p in sorted(root.glob("*.json")):
        trial = json.loads(p.read_text())
        if "model" not in trial:
            continue
        rows.append({"dataset":trial["dataset"], "model":trial["model"], "seed":trial["seed"], "train_seconds":trial["fit_seconds"], "peak_rss_mb":trial["peak_rss_mb"], "nnz":trial["nnz"], **trial["test"], "val_ndcg":trial["validation"]["ndcg"]})
    df = pd.DataFrame(rows)
    df.to_csv(root/"runs.csv", index=False)
    summary = df.groupby(["dataset", "model"]).agg(seeds=("seed","count"), n=("n","first"), ndcg=("ndcg","mean"), ndcg_sd=("ndcg","std"), hr=("hr","mean"), val_ndcg=("val_ndcg","mean"), train_seconds=("train_seconds","mean"), peak_rss_mb=("peak_rss_mb","max"), ms_per_user=("ms_per_user","mean"), short_n=("short_n","first"), short_ndcg=("short_ndcg","mean"), full_n=("full_n","first"), full_ndcg=("full_ndcg","mean")).reset_index()
    summary.to_csv(root/"summary.csv", index=False)
    for metric, ylabel, filename in [("ndcg","NDCG@10", "quality"), ("train_seconds","Training time, seconds", "training_time"), ("peak_rss_mb","Peak process RSS, MiB", "memory")]:
        fig, axes = plt.subplots(1, len(summary.dataset.unique()), figsize=(11,3.2), squeeze=False)
        for ax, ds in zip(axes[0], [x for x in LABELS if x in summary.dataset.unique()]):
            group = summary[summary.dataset==ds].set_index("model")
            names = [m for m in MODEL_LABELS if m in group.index]
            values = group.loc[names,metric]
            ax.bar(range(len(names)), values, color=["#9ca3af" if m=="popular" else "#93a4b5" if m=="zero" else "#147d92" if m=="pad" else "#e7a13d" if m=="bos" else "#704e99" for m in names])
            if metric=="ndcg":
                ax.errorbar(range(len(names)), values, yerr=group.loc[names,"ndcg_sd"].fillna(0), fmt="none", capsize=3, color="#273647")
            ax.set_xticks(range(len(names)), [MODEL_LABELS[m] for m in names], rotation=45, ha="right", fontsize=8)
            ax.set_title(LABELS[ds], fontsize=10)
            ax.set_ylabel(ylabel); ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(root/(filename+".png"), dpi=180); plt.close(fig)
    paired = []
    rng = np.random.default_rng(20261121)
    for ds in df.dataset.unique():
        zero_files = sorted(root.glob(f"{ds}_zero_*_test.npz"))
        if not zero_files:
            continue
        for mode in [m for m in ["pad","bos","sasrec"] if m in df.model.unique()]:
            diffs = []
            for zp in zero_files:
                mp = zp.with_name(zp.name.replace("_zero_", f"_{mode}_"))
                if mp.exists():
                    with np.load(zp) as a, np.load(mp) as b:
                        np.testing.assert_array_equal(a["lengths"], b["lengths"])
                        diffs.append(b["ndcg"] - a["ndcg"])
            if not diffs:
                continue
            per_user = np.mean(diffs, axis=0)
            bootstrap = [rng.choice(per_user, len(per_user), replace=True).mean() for _ in range(2000)]
            lo, hi = np.quantile(bootstrap,[.025,.975])
            paired.append({"dataset":ds,"contrast":mode+"-zero","delta_ndcg":float(per_user.mean()),"ci_user_low":lo,"ci_user_high":hi,"n_users":len(per_user),"n_seeds":len(diffs),"seed_delta_sd":float(np.std(np.mean(diffs,axis=1),ddof=1)) if len(diffs)>1 else None})
    pd.DataFrame(paired).to_csv(root/"paired.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--results",default="results/interim")
    summarize(p.parse_args().results)
