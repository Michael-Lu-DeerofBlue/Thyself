import argparse, json, numpy as np, torch
from typing import List
from .labels import load_t0
from .models import MLPHead, Embedder, EMBEDDER_MAP

# Basic metrics (micro/macro F1, mAP@K, NDCG@K)

def f1_micro(y_true, y_prob, threshold=0.5):
    preds = (y_prob >= threshold).astype(int)
    tp = (preds * y_true).sum()
    fp = (preds * (1-y_true)).sum()
    fn = ((1-preds) * y_true).sum()
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    return f1

def f1_macro(y_true, y_prob, threshold=0.5):
    C = y_true.shape[1]
    f1s = []
    for c in range(C):
        preds = (y_prob[:,c] >= threshold).astype(int)
        tp = (preds * y_true[:,c]).sum(); fp = (preds * (1-y_true[:,c])).sum(); fn = ((1-preds)*y_true[:,c]).sum()
        prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
        f1s.append(2*prec*rec/(prec+rec+1e-9))
    return float(np.mean(f1s))

def average_precision_at_k(y_true_row, scores_row, k):
    order = np.argsort(-scores_row)
    ap = 0.0; hits = 0
    for rank, idx in enumerate(order[:k], start=1):
        if y_true_row[idx] > 0.0:
            hits += 1
            ap += hits / rank
    denom = min(k, int(y_true_row.sum())) or 1
    return ap / denom

def mean_ap_k(y_true, scores, k):
    return float(np.mean([average_precision_at_k(y_true[i], scores[i], k) for i in range(len(y_true))]))

def ndcg_at_k(y_true_row, scores_row, k):
    order = np.argsort(-scores_row)
    gains = y_true_row[order[:k]]
    dcg = 0.0
    for i,g in enumerate(gains, start=1):
        dcg += (2**g - 1)/np.log2(i+1)
    ideal_order = np.argsort(-y_true_row)
    ideal_gains = y_true_row[ideal_order[:k]]
    idcg = 0.0
    for i,g in enumerate(ideal_gains, start=1):
        idcg += (2**g - 1)/np.log2(i+1)
    return dcg/(idcg+1e-9)

def mean_ndcg_k(y_true, scores, k):
    return float(np.mean([ndcg_at_k(y_true[i], scores[i], k) for i in range(len(y_true))]))

# Temperature scaling for calibration
class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-3)

def apply_temperature(logits, temp):
    return logits / temp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=False, help="Optional path to precomputed scores npz with y_true,scores")
    args = ap.parse_args()
    # This script is mostly library; training script can import metrics.
    if args.scores and os.path.exists(args.scores):
        data = np.load(args.scores)
        y_true, scores = data['y_true'], data['scores']
        out = {
            "micro_f1": f1_micro(y_true, scores),
            "macro_f1": f1_macro(y_true, scores),
            "map@5": mean_ap_k(y_true, scores, 5),
            "ndcg@5": mean_ndcg_k(y_true, scores, 5)
        }
        print(json.dumps(out, indent=2))
