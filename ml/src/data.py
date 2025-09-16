import csv, random, os, numpy as np, torch
from dataclasses import dataclass
from typing import List, Dict, Tuple
from .labels import id_to_index
from .models import Embedder

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class Example:
    title: str
    labels: List[int]  # indices

class TitleDataset:
    def __init__(self, examples: List[Example], num_labels: int):
        self.examples = examples
        self.num_labels = num_labels
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        y = np.zeros(self.num_labels, dtype=np.float32)
        for li in ex.labels:
            if li < self.num_labels:
                y[li] = 1.0
        return ex.title, y

def load_examples(csv_path: str, t0_list) -> Tuple[List[Example], int]:
    id2idx = id_to_index(t0_list)
    examples: List[Example] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_labels = [l.strip() for l in row['labels'].split(',') if l.strip()]
            indices = [id2idx[l] for l in raw_labels if l in id2idx]
            examples.append(Example(title=row['title'], labels=indices))
    return examples, len(t0_list)

def train_val_split(examples: List[Example], val_ratio: float = 0.2):
    idxs = list(range(len(examples)))
    random.shuffle(idxs)
    cut = int(len(idxs)*(1-val_ratio))
    train = [examples[i] for i in idxs[:cut]]
    val = [examples[i] for i in idxs[cut:]]
    return train, val

def embed_dataset(embedder: Embedder, dataset: TitleDataset):
    titles = [ex.title for ex in dataset.examples]
    X = embedder.encode(titles)  # [N,D]
    Y = np.stack([dataset[i][1] for i in range(len(dataset))])
    return X, Y
