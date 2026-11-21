"""LA-SATF HOOI using sparse position slices, without a dense 4-D tensor.

Independent implementation of Frolov & Oseledets (2023), Algorithms 2,
Eqs. 18-21. Each positional slice sums all observations: repeated PAD is
not collapsed into a single user-item observation.
"""
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, solve_triangular

def encode(history, k, mode="zero", special=None, amplitude=1.0, balance=False):
    if mode not in {"zero", "pad", "bos", "start"}:
        raise ValueError(mode)
    history = np.asarray(history, dtype=int)[-k:]
    n = len(history)
    q = k - n
    pos, items, val = list(range(q, k)), history.tolist(), [1.] * n
    if q and mode != "zero":
        if special is None:
            raise ValueError("special id required")
        prefix = list(range(q)) if mode == "pad" else [q - 1] if mode == "bos" else [0]
        a = amplitude / np.sqrt(len(prefix)) if balance else amplitude
        pos += prefix
        items += [special] * len(prefix)
        val += [a] * len(prefix)
    return np.asarray(items, int), np.asarray(pos, int), np.asarray(val, float)

def position_slices(histories, n_items, k, mode="zero", amplitude=1., balance=False):
    users, items, positions, values = [], [], [], []
    for u, history in enumerate(histories):
        ids, p, v = encode(history, k, mode, n_items, amplitude, balance)
        users.extend([u] * len(p)); items.extend(ids); positions.extend(p); values.extend(v)
    users, items, positions, values = map(np.asarray, (users, items, positions, values))
    # Always reserve the same zero/special row to keep dimensions/initialization equal.
    shape = (len(histories), n_items + 1)
    result = []
    for p in range(k):
        mask = positions == p
        result.append(sparse.coo_matrix((values[mask], (users[mask], items[mask])), shape=shape).tocsr())
    return result

def attention_matrix(window, decay):
    lag = np.arange(window)[:, None] - np.arange(window)[None, :]
    return np.where(lag >= 0, (np.maximum(lag, 0) + 1.) ** (-decay), 0.)

def hankel_weights(a, b):
    result = np.zeros((len(a) + len(b) - 1, a.shape[1], b.shape[1]))
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            result[:, i, j] = np.convolve(a[:, i], b[:, j])
    return result

def leading_left(matrix, rank):
    if rank > min(matrix.shape):
        raise ValueError(f"Rank {rank} exceeds compressed unfolding {matrix.shape}")
    # Small Gram matrix + QR avoids expensive SVD of a tall unfolding.
    if matrix.shape[0] <= matrix.shape[1]:
        _, vec = eigh(matrix @ matrix.T, subset_by_index=[matrix.shape[0]-rank, matrix.shape[0]-1])
        return vec[:, ::-1]
    _, right = eigh(matrix.T @ matrix, subset_by_index=[matrix.shape[1]-rank, matrix.shape[1]-1])
    return np.linalg.qr(matrix @ right[:, ::-1], mode="reduced")[0]

def contract(slices, factors, attention, mode):
    u, v, wl, ws = factors
    wa = attention @ wl
    if mode in (0, 1):
        h = hankel_weights(wa, ws)
        other = v if mode == 0 else u
        size = slices[0].shape[mode]
        out = np.zeros((size, other.shape[1], wl.shape[1], ws.shape[1]))
        for p, x in enumerate(slices):
            z = x @ other if mode == 0 else x.T @ other
            out += np.einsum("ir,ab->irab", z, h[p], optimize=True)
        return out.reshape(size, -1)
    c = np.stack([u.T @ (x @ v) for x in slices])
    if mode == 2:
        out = np.stack([np.einsum("suv,sr->uvr", c[a:a+len(ws)], ws) for a in range(len(wl))])
        return (attention.T @ out.reshape(len(wl), -1))
    out = np.stack([np.einsum("auv,ar->uvr", c[s:s+len(wl)], wa) for s in range(len(ws))])
    return out.reshape(len(ws), -1)

class LASATF:
    def __init__(self, n_items, k=50, window=5, rank=12, local_rank=2, sequence_rank=2, decay=1., scaling=1., iterations=3, mode="zero", seed=0, amplitude=1., balance=False, special_scaling="unit"):
        self.n_items, self.k, self.mode = n_items, k, mode
        self.window, self.rank, self.local_rank, self.sequence_rank = window, rank, local_rank, sequence_rank
        self.decay, self.scaling, self.iterations, self.seed = decay, scaling, iterations, seed
        self.amplitude, self.balance, self.special_scaling = amplitude, balance, special_scaling

    def fit(self, histories):
        if not 1 <= self.window <= self.k:
            raise ValueError("window must lie in [1,K]")
        slices = position_slices(histories, self.n_items, self.k, self.mode, self.amplitude, self.balance)
        self.nnz = sum(s.nnz for s in slices)
        # Popularities use only real train tails. Service row is unit-weight by default.
        counts = np.bincount(np.concatenate([h[-self.k:] for h in histories]), minlength=self.n_items + 1).astype(float)
        if self.special_scaling == "frequency":
            counts[-1] = sum(x[:, -1].count_nonzero() for x in slices)
        elif self.special_scaling != "unit":
            raise ValueError(self.special_scaling)
        self.weights = np.maximum(counts, 1.) ** ((self.scaling - 1.) / 2.)
        if self.special_scaling == "unit":
            self.weights[-1] = 1.
        slices = [x.multiply(self.weights).tocsr() for x in slices]
        self.attention = attention_matrix(self.window, self.decay)
        dims = (len(histories), self.n_items + 1, self.window, self.k-self.window+1)
        ranks = (self.rank, self.rank, self.local_rank, self.sequence_rank)
        rng = np.random.default_rng(self.seed)
        self.factors = [np.linalg.qr(rng.normal(size=(d, r)), mode="reduced")[0] for d, r in zip(dims, ranks)]
        if any(r > d for d, r in zip(dims, ranks)):
            raise ValueError("Rank larger than a dimension")
        self.core_norms = []
        for _ in range(self.iterations):
            for mode, rank in enumerate(ranks):
                mat = contract(slices, self.factors, self.attention, mode)
                self.factors[mode] = leading_left(mat, rank)
            self.core_norms.append(float(np.linalg.norm(self.factors[3].T @ mat)))
        _, self.v, wl, ws = self.factors
        inverse_w = solve_triangular(self.attention.T, wl, lower=False)
        left = (self.attention @ wl) @ inverse_w[-1]
        right = ws @ ws[-1]
        self.prediction_weights = np.convolve(left, right)
        return self

    def score_batch(self, histories):
        latent = np.zeros((len(histories), self.rank))
        for row, h in enumerate(histories):
            ids, p, vals = encode(h, self.k, self.mode, self.n_items, self.amplitude, self.balance)
            # Encode first, then shift left: do not insert a new BOS after shifting.
            take = p > 0
            ids, p, vals = ids[take], p[take] - 1, vals[take]
            latent[row] = (self.v[ids] * (vals * self.weights[ids] * self.prediction_weights[p])[:, None]).sum(axis=0)
        # Score only real items. No inverse-popularity rescaling at output.
        return latent @ self.v[:self.n_items].T

class MostPopular:
    def __init__(self, n_items):
        self.n_items = n_items
    def fit(self, histories):
        self.counts = np.bincount(np.concatenate(histories), minlength=self.n_items).astype(float)
        return self
    def score_batch(self, histories):
        return np.tile(self.counts, (len(histories), 1))
