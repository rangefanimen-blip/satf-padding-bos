"""Train-only vocabulary and global chronological evaluation protocol."""
from pathlib import Path
import gzip
import hashlib
import json
import urllib.request
import zipfile
import numpy as np
import pandas as pd

SOURCES = {
    "ml1m": ("ml-1m.zip", "https://files.grouplens.org/datasets/movielens/ml-1m.zip"),
    "beauty": ("reviews_Beauty_5.json.gz", "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/reviews_Beauty_5.json.gz"),
    "games": ("reviews_Video_Games_5.json.gz", "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/reviews_Video_Games_5.json.gz"),
}

def download(name, root="data/raw"):
    filename, url = SOURCES[name]
    path = Path(root) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, temp)
        temp.replace(path)
    return path

def load(name, root="data/raw"):
    path = download(name, root)
    if name == "ml1m":
        with zipfile.ZipFile(path) as z:
            with z.open("ml-1m/ratings.dat") as f:
                df = pd.read_csv(f, sep="::", engine="python", names=["user", "item", "rating", "time"])
    else:
        with gzip.open(path, "rt") as f:
            records = (json.loads(line) for line in f)
            df = pd.DataFrame((r["reviewerID"], r["asin"], r["overall"], r["unixReviewTime"]) for r in records)
        df.columns = ["user", "item", "rating", "time"]
    df["user"] = df.user.astype(str)
    df["item"] = df.item.astype(str)
    return df, hashlib.sha256(path.read_bytes()).hexdigest()

def clean(df):
    """Every recorded rating is an implicit event. Earliest pair wins."""
    df = df[["user", "item", "time"]].dropna().copy()
    df["time"] = df.time.astype("int64")
    return df.sort_values(["time", "user", "item"], kind="stable").drop_duplicates(["user", "item"], keep="first").reset_index(drop=True)

def prepare(df, quantiles=(.8, .9)):
    raw_n = len(df)
    df = clean(df)
    t1, t2 = (int(df.time.quantile(q, interpolation="lower")) for q in quantiles)
    if t1 >= t2:
        raise ValueError("Need distinct temporal boundaries")
    tr = df[df.time < t1]
    items = sorted(tr.item.unique())
    mapping = {x: i for i, x in enumerate(items)}
    histories = [g.item.map(mapping).to_numpy(dtype=int) for _, g in tr.groupby("user", sort=True)]
    samples, diagnostics = {}, {}
    for label, lo, hi in [("validation", t1, t2), ("test", t2, None)]:
        interval = df[df.time >= lo]
        if hi is not None:
            interval = interval[interval.time < hi]
        # First next-new event, never replace a cold target with a later warm one.
        targets = interval.groupby("user", sort=True).head(1)
        before = df[df.time < lo]
        history_map = {u: g.item.tolist() for u, g in before.groupby("user", sort=False)}
        rows, cold, no_history = [], 0, 0
        for r in targets.itertuples():
            hist = [mapping[x] for x in history_map.get(r.user, []) if x in mapping]
            if r.item not in mapping:
                cold += 1
                continue
            if not hist:
                no_history += 1
                continue
            rows.append({"user": r.user, "history": hist, "target": mapping[r.item], "time": int(r.time)})
        samples[label] = rows
        diagnostics[label] = {"targets": len(targets), "evaluated": len(rows), "cold_target": cold, "empty_known_history": no_history}
    split_payload = {"train": tr.to_csv(index=False), "v": samples["validation"], "t": samples["test"]}
    split_hash = hashlib.sha256(json.dumps(split_payload, sort_keys=True).encode()).hexdigest()
    stats = {"raw_events": raw_n, "events": len(df), "users": int(df.user.nunique()), "items": int(df.item.nunique()), "train_events": len(tr), "train_users": len(histories), "catalog": len(items), "t1": t1, "t2": t2, "split_sha256": split_hash, **diagnostics}
    return {"histories": histories, "samples": samples, "n_items": len(items), "stats": stats}
