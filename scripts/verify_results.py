"""Check completeness, provenance consistency and NPZ-derived metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from satf_padding.experiment import source_hash

def verify(root, config):
    root = Path(root); cfg = json.loads(Path(config).read_text())
    count = 0
    source_hashes = set()
    for ds in cfg["datasets"]:
        stats = json.loads((root/f"{ds}_data.json").read_text())
        if stats["t1"] >= stats["t2"]:
            raise ValueError("Invalid temporal cutoffs")
        for model in cfg["models"]:
            for seed in (cfg["seeds"][:1] if model == "popular" else cfg["seeds"]):
                p = root/f"{ds}_{model}_{seed}.json"
                run = json.loads(p.read_text()); count += 1
                source_hashes.add(run["source_sha256"])
                assert run["config"] == cfg
                assert run["split_sha256"] == stats["split_sha256"]
                for split in ["validation","test"]:
                    with np.load(p.with_name(p.stem+"_"+split+".npz")) as a:
                        assert len(a["ranks"]) == stats[split]["evaluated"]
                        assert np.isfinite(a["ndcg"]).all()
                        recomputed = np.where(a["ranks"] <= 10, 1/np.log2(a["ranks"]+1), 0.)
                        np.testing.assert_allclose(a["ndcg"], recomputed)
                        assert abs(recomputed.mean()-run[split]["ndcg"]) < 1e-12
    assert len(source_hashes) == 1, "Mixed implementation versions"
    assert source_hashes == {source_hash()}, "Recorded source differs from this checkout"
    print(f"Verified {count} runs in {root}; consistent source and splits")
    return count

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--results",default="results/interim"); p.add_argument("--config",default="configs/interim.json")
    args=p.parse_args();verify(args.results,args.config)
