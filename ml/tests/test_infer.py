import json, types
import numpy as np

# We will mock Embedder.encode to avoid heavy model download during tests.

import ml.src.infer as infer_mod
import ml.src.labels as labels_mod

class DummyEmbedder:
    def __init__(self, name="minilm"): pass
    def encode(self, texts):
        # deterministic pseudo embeddings based on hash -> 4-dim
        out = []
        for t in texts:
            h = abs(hash(t)) % 10_000
            vec = np.array([(h % 97)/97.0, (h % 89)/89.0, (h % 83)/83.0, (h % 79)/79.0], dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            out.append(vec)
        return np.vstack(out)

def test_rank_titles_shape(monkeypatch):
    monkeypatch.setattr(infer_mod, 'Embedder', lambda name='minilm': DummyEmbedder())
    monkeypatch.setattr(labels_mod, 'Embedder', lambda name='minilm': DummyEmbedder())
    res = infer_mod.rank_titles(["Sample Title One"], model='minilm', t0='ml/taxonomies/t0.yaml', topk=5)
    assert len(res) == 1
    assert len(res[0]) == 5
    # scores descending
    scores = [s for _, s in res[0]]
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
