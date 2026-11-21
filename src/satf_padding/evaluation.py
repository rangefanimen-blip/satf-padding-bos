import time
import numpy as np

def target_rank(scores, history, target):
    scores = np.asarray(scores, dtype=float).copy()
    scores[np.asarray(history, int)] = -np.inf
    if target in history or not np.isfinite(scores[target]):
        raise ValueError("Invalid target or nonfinite prediction")
    # Stable full-catalog ordering: score desc, item index asc.
    rank = 1 + np.sum(scores > scores[target]) + np.sum((scores == scores[target]) & (np.arange(len(scores)) < target))
    return int(rank)

def evaluate(model, rows, k=50, batch_size=128):
    ranks, lengths = [], []
    start = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset+batch_size]
        scores = model.score_batch([r["history"] for r in batch])
        for r, score in zip(batch, scores):
            ranks.append(target_rank(score, r["history"], r["target"]))
            lengths.append(min(len(r["history"]), k))
    elapsed = time.perf_counter() - start
    ranks, lengths = np.asarray(ranks), np.asarray(lengths)
    ndcg = np.where(ranks <= 10, 1. / np.log2(ranks + 1), 0.)
    hit = (ranks <= 10).astype(float)
    metrics = {"n": len(rows), "ndcg": float(ndcg.mean()), "hr": float(hit.mean()), "eval_seconds": elapsed, "ms_per_user": elapsed*1000/max(len(rows),1)}
    for label, mask in [("short", lengths < k), ("full", lengths == k), ("1_5", lengths <= 5), ("6_20", (lengths > 5) & (lengths <= 20)), ("21_plus", lengths > 20)]:
        metrics[label+"_n"] = int(mask.sum())
        metrics[label+"_ndcg"] = float(ndcg[mask].mean()) if mask.any() else None
    return metrics, {"ranks": ranks, "lengths": lengths, "ndcg": ndcg}
