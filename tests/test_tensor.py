import numpy as np
import pytest
from satf_padding.tensor import encode, position_slices, contract, attention_matrix, LASATF
from satf_padding.evaluation import target_rank

@pytest.mark.parametrize("mode", ["zero", "pad", "bos", "start"])
def test_sparse_contractions_equal_dense_hankel(mode):
    histories = [np.array([0, 1]), np.array([2]), np.array([1, 0, 2, 3, 4, 5])]
    slices = position_slices(histories, 6, 6, mode, amplitude=.7, balance=True)
    x = np.stack([a.toarray() for a in slices], axis=-1)
    hankel = np.stack([x[:, :, a:a+4] for a in range(3)], axis=2)
    a = attention_matrix(3, .8)
    y = np.einsum("uijs,ja->uias", hankel, a)
    rng = np.random.default_rng(13)
    factors = [np.linalg.qr(rng.normal(size=(d, 2)))[0] for d in y.shape]
    letters = "uial"
    ranks = "vwbc"
    for mode_id in range(4):
        lhs = [letters]
        operands = [y]
        out = letters[mode_id]
        for dim in range(4):
            if dim != mode_id:
                lhs.append(letters[dim] + ranks[dim]); operands.append(factors[dim]); out += ranks[dim]
        expected = np.einsum(",".join(lhs) + "->" + out, *operands).reshape(y.shape[mode_id], -1)
        np.testing.assert_allclose(contract(slices, factors, a, mode_id), expected, atol=1e-11)

def test_repeated_pad_and_balancing():
    ids, p, vals = encode([1, 2], 6, "pad", 7, balance=True)
    assert sum(ids == 7) == 4
    assert set(p[ids == 7]) == {0, 1, 2, 3}
    assert np.sum(vals[ids == 7] ** 2) == pytest.approx(1.)

def test_full_histories_identical_and_no_spurious_bos_on_shift():
    expected = encode([0, 1, 2, 3], 4)
    for mode in ("pad", "bos"):
        for a, b in zip(expected, encode([0, 1, 2, 3], 4, mode, 4)):
            np.testing.assert_array_equal(a, b)
        ids, p, _ = encode([0, 1, 2, 3, 0], 4, mode, 4)
        assert 4 not in ids[p > 0]

def test_folding_in_matches_dense_projector():
    histories = [np.array([0, 1]), np.array([1, 2]), np.array([2, 3, 0]), np.array([3, 1, 4]), np.array([4, 0, 2])]
    model = LASATF(5, k=6, window=3, rank=2, local_rank=2, sequence_rank=2, mode="pad", iterations=2).fit(histories)
    h = [0, 1]
    ids, p, vals = encode(h, 6, "pad", 5)
    raw = np.zeros((6, 6))
    for i, pos, v in zip(ids, p, vals):
        if pos > 0:
            raw[i, pos-1] += v * model.weights[i]
    hankel = np.stack([raw[:, a:a+4] for a in range(3)], axis=1)
    _, v, wl, ws = model.factors
    pa = np.linalg.solve(model.attention.T, wl) @ wl.T @ model.attention.T
    ps = ws @ ws.T
    dense = np.einsum("ij,jab,la,sb->ils", v@v.T, hankel, pa, ps)
    np.testing.assert_allclose(model.score_batch([h])[0], dense[:5, -1, -1], atol=1e-10)
    assert model.score_batch([h]).shape == (1, 5)
    assert np.all(np.diff(model.core_norms) >= -1e-8)

def test_seen_items_and_ties():
    assert target_rank([3., 2., 2., 1.], [0], 2) == 2
    with pytest.raises(ValueError):
        target_rank([2., 1.], [0], 0)
