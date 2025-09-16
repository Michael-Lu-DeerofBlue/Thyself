import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import List

EMBEDDER_MAP = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",    # 384-dim
    "mpnet": "sentence-transformers/all-mpnet-base-v2",     # 768-dim
    "e5": "intfloat/e5-base-v2",                            # 768-dim
}

class Embedder:
    def __init__(self, name: str = "minilm", device: str | None = None):
        if name not in EMBEDDER_MAP:
            raise ValueError(f"Unknown embedder key {name}; choose from {list(EMBEDDER_MAP)}")
        self.model = SentenceTransformer(EMBEDDER_MAP[name], device=device)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, num_labels: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
