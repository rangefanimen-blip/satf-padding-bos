import numpy as np
import torch
from satf_padding.sasrec import Network, SASRec

def test_attention_is_causal_and_padding_is_finite():
    torch.manual_seed(3)
    net = Network(7, 5, 8, 1, 0.).eval()
    a = torch.tensor([[0, 0, 1, 2, 3]])
    b = torch.tensor([[0, 0, 1, 2, 5]])
    with torch.no_grad():
        x, y = net(a), net(b)
    assert torch.isfinite(x).all()
    torch.testing.assert_close(x[:, :-1], y[:, :-1])

def test_tiny_sasrec_fit_and_catalog():
    torch.set_num_threads(1)
    model = SASRec(7, k=5, dim=8, epochs=2, batch_size=2).fit([np.array([0, 1, 2]), np.array([2, 3, 4]), np.array([4, 5, 6])])
    assert len(model.losses) == 2
    scores = model.score_batch([[0, 1], [3]])
    assert scores.shape == (2, 7)
    assert np.isfinite(scores).all()
