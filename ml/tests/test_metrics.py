import numpy as np
from ml.src.eval import f1_micro, f1_macro, mean_ap_k, mean_ndcg_k

def test_metrics_basic():
    y_true = np.array([[1,0,1],[0,1,0]], dtype=float)
    scores = np.array([[0.9,0.2,0.8],[0.1,0.7,0.3]])
    micro = f1_micro(y_true, scores, 0.5)
    macro = f1_macro(y_true, scores, 0.5)
    assert 0 <= micro <= 1
    assert 0 <= macro <= 1
    map5 = mean_ap_k(y_true, scores, 2)
    ndcg5 = mean_ndcg_k(y_true, scores, 2)
    assert 0 <= map5 <= 1
    assert 0 <= ndcg5 <= 1
