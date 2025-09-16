import yaml, numpy as np
from .models import Embedder
from typing import List, Dict, Any

def load_t0(path: str = "ml/taxonomies/t0.yaml") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_label_matrix(t0: List[Dict[str, Any]], embedder: Embedder):
    texts = [f"{x['name']}: {x.get('description','')}" for x in t0]
    return embedder.encode(texts)

def id_to_index(t0: List[Dict[str, Any]]):
    return {x['id']: i for i,x in enumerate(t0)}
