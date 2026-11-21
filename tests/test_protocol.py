import numpy as np
import pandas as pd
from satf_padding.data import clean, prepare
from satf_padding.evaluation import evaluate
from satf_padding.tensor import MostPopular

def test_global_cutoff_dedup_and_no_future_catalog():
    rows = [(str(u), str(i % 8), i+u, 1.) for u in range(5) for i in range(8)]
    rows += [("0", "0", 50, 5.), ("0", "future", 40, 5.)]
    df = pd.DataFrame(rows, columns=["user", "item", "time", "rating"])
    c = clean(df)
    assert len(c.query('user == "0" and item == "0"')) == 1
    assert c.query('user == "0" and item == "0"').time.iloc[0] == 0
    data = prepare(df, (.6, .8))
    assert data["n_items"] == c[c.time < data["stats"]["t1"]].item.nunique()
    assert data["stats"]["test"]["cold_target"] >= 1
    for split, rows in data["samples"].items():
        for row in rows:
            assert row["target"] not in row["history"]
    assert prepare(df, (.6, .8))["stats"]["split_sha256"] == data["stats"]["split_sha256"]

def test_ndcg_known_ranks():
    m = MostPopular(4).fit([np.array([0, 1, 1, 2])])
    metrics, _ = evaluate(m, [{"history": [0], "target": 1}, {"history": [1], "target": 2}], k=3)
    assert np.isclose(metrics["ndcg"], (1 + 1/np.log2(3))/2)
