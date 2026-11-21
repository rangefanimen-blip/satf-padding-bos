"""Run each training seed in a fresh process; resume only identical manifests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import threading
import time
import numpy as np
import psutil
from .data import load, prepare
from .evaluation import evaluate
from .tensor import LASATF, MostPopular

def source_hash():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()

def trial(args):
    spec = json.loads(args.trial)
    data = pickle.loads(Path(spec["cache"]).read_bytes())
    cfg, name, seed = spec["config"], spec["model"], spec["seed"]
    if name == "popular":
        model = MostPopular(data["n_items"])
    elif name == "sasrec":
        import torch
        torch.set_num_threads(spec["threads"])
        from .sasrec import SASRec
        model = SASRec(data["n_items"], k=cfg["k"], seed=seed, device=spec["device"], **cfg["sasrec"])
    else:
        params = dict(cfg["tensor"])
        params.update(mode=name.split("_")[0], balance="balanced" in name, special_scaling="frequency" if "frequency" in name else "unit")
        if "scaled" in name:
            params["scaling"] = .2
        model = LASATF(data["n_items"], k=cfg["k"], seed=seed, **params)
    running = threading.Event(); running.set()
    rss = [psutil.Process().memory_info().rss]
    def monitor():
        while running.is_set():
            rss[0] = max(rss[0], psutil.Process().memory_info().rss)
            time.sleep(.02)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    start = time.perf_counter()
    model.fit(data["histories"])
    fit_seconds = time.perf_counter() - start
    results = {}
    out = Path(spec["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    for split in ("validation", "test"):
        metrics, arrays = evaluate(model, data["samples"][split], cfg["k"])
        results[split] = metrics
        np.savez_compressed(out.with_name(out.stem + "_" + split + ".npz"), **arrays)
    running.clear(); thread.join()
    results.update({"dataset": spec["dataset"], "model": name, "seed": seed, "fit_seconds": fit_seconds, "peak_rss_mb": rss[0]/2**20, "nnz": getattr(model, "nnz", None), "losses": getattr(model, "losses", None), "core_norms": getattr(model, "core_norms", None), "source_sha256": spec["source_sha256"], "config": cfg, "split_sha256": data["stats"]["split_sha256"], "device": str(getattr(model, "device", "cpu")), "threads": spec["threads"], "platform": platform.platform(), "python": platform.python_version()})
    out.write_text(json.dumps(results, indent=2, allow_nan=False))
    print(f"{spec['dataset']} {name} seed={seed}: NDCG={results['test']['ndcg']:.5f}, train={fit_seconds:.1f}s", flush=True)

def run(args):
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    code_sha = source_hash()
    for dataset in (args.datasets.split(",") if args.datasets else cfg["datasets"]):
        raw, raw_sha = load(dataset, args.raw)
        data = prepare(raw, cfg["quantiles"])
        data["stats"]["raw_sha256"] = raw_sha
        (out / (dataset + "_data.json")).write_text(json.dumps(data["stats"], indent=2))
        cache = Path("data/processed") / (dataset + "_" + data["stats"]["split_sha256"][:12] + ".pkl")
        cache.parent.mkdir(parents=True, exist_ok=True); cache.write_bytes(pickle.dumps(data))
        models = args.models.split(",") if args.models else cfg["models"]
        for name in models:
            for seed in (cfg["seeds"][:1] if name == "popular" else cfg["seeds"]):
                target = out / f"{dataset}_{name}_{seed}.json"
                if target.exists() and not args.force:
                    old = json.loads(target.read_text())
                    if old["source_sha256"] == code_sha and old["config"] == cfg and old["split_sha256"] == data["stats"]["split_sha256"]:
                        print("Already complete:", target, flush=True); continue
                    raise RuntimeError(f"Stale output {target}; choose a new output folder or --force")
                spec = {"config": cfg, "dataset": dataset, "model": name, "seed": seed, "cache": str(cache), "output": str(target), "source_sha256": code_sha, "threads": args.threads, "device": args.device}
                env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads), VECLIB_MAXIMUM_THREADS=str(args.threads), CUBLAS_WORKSPACE_CONFIG=":4096:8")
                subprocess.run([sys.executable, "-m", "satf_padding.experiment", "--trial", json.dumps(spec)], env=env, check=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/interim.json"); p.add_argument("--output", default="results/interim")
    p.add_argument("--raw", default="data/raw"); p.add_argument("--datasets"); p.add_argument("--models")
    p.add_argument("--threads", type=int, default=2); p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--force", action="store_true"); p.add_argument("--trial")
    args = p.parse_args()
    trial(args) if args.trial else run(args)

if __name__ == "__main__":
    main()
