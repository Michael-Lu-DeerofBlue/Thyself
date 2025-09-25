import argparse, os, json, numpy as np, torch
import torch.nn.functional as F
from .models import Embedder, MLPHead, EMBEDDER_MAP
from .labels import load_t0
from .data import load_examples, train_val_split, TitleDataset, embed_dataset, set_seed


def compute_micro_f1(y_true: np.ndarray, y_pred: np.ndarray):
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return f1, precision, recall

def sweep_thresholds(y_true: np.ndarray, probs: np.ndarray, base_threshold: float):
    """Return (best_thr, f1, precision, recall). Includes provided base_threshold first."""
    best = (base_threshold, 0.0, 0.0, 0.0)
    # Candidate thresholds skew lower for small datasets where logits start near zero.
    candidates = sorted(set([base_threshold, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]))
    for thr in candidates:
        preds = (probs >= thr).astype(int)
        f1, p, r = compute_micro_f1(y_true, preds)
        if f1 > best[1]:
            best = (thr, f1, p, r)
    return best

def train(args):
    set_seed(args.seed)
    t0 = load_t0(args.t0)
    examples, num_labels = load_examples(args.data, t0)
    train_ex, val_ex = train_val_split(examples, val_ratio=args.val_ratio)
    if len(val_ex) == 0:  # guard on tiny datasets
        # force at least one val example by moving last train example
        if len(train_ex) > 1:
            val_ex.append(train_ex.pop())
    embedder = Embedder(args.model)
    train_ds = TitleDataset(train_ex, num_labels)
    val_ds = TitleDataset(val_ex, num_labels)
    Xtr, Ytr = embed_dataset(embedder, train_ds)
    Xva, Yva = embed_dataset(embedder, val_ds)

    in_dim = Xtr.shape[1]
    model = MLPHead(in_dim, num_labels, hidden=args.hidden)
    # Compute class frequencies for optional pos_weight
    label_freq = Ytr.sum(axis=0)  # counts of positives per label
    total = len(Ytr)
    # Avoid division by zero: if a label never appears, set weight=1.
    pos_weight = []
    for c in range(num_labels):
        pos = label_freq[c]
        if pos == 0:
            pos_weight.append(1.0)
        else:
            neg = total - pos
            pos_weight.append(max(1.0, neg / pos))
    pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32)
    if args.no_pos_weight:
        pos_weight_t = None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    Xtr_t = torch.tensor(Xtr)
    Ytr_t = torch.tensor(Ytr)
    Xva_t = torch.tensor(Xva)
    Yva_t = torch.tensor(Yva)

    best_f1 = -1
    best_threshold = args.threshold
    patience = args.patience
    wait = 0
    for epoch in range(args.epochs):
        model.train()
        logits = model(Xtr_t)
        if pos_weight_t is not None:
            loss = F.binary_cross_entropy_with_logits(logits, Ytr_t, pos_weight=pos_weight_t)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, Ytr_t)
        opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        with torch.no_grad():
            v_logits = model(Xva_t)
            v_probs = torch.sigmoid(v_logits).numpy()
            if args.auto_threshold:
                thr, f1, precision, recall = sweep_thresholds(Yva, v_probs, args.threshold)
            else:
                preds = (v_probs >= args.threshold).astype(int)
                f1, precision, recall = compute_micro_f1(Yva, preds)
                thr = args.threshold
        print(json.dumps({"epoch": epoch, "loss": float(loss.item()), "val_f1": float(f1), "precision": float(precision), "recall": float(recall), "threshold": float(thr)}, ensure_ascii=False))
        if f1 > best_f1:
            best_f1 = f1; best_threshold = thr; wait = 0
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "in_dim": in_dim,
                "num_labels": num_labels,
                "embedder": args.model,
                "threshold": best_threshold,
                "label_freq": label_freq.tolist()
            }, os.path.join(args.out_dir, "model.pt"))
        else:
            wait += 1
            if wait >= patience:
                break
    print(json.dumps({"best_f1": float(best_f1), "best_threshold": float(best_threshold)}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ml/data/example.csv")
    ap.add_argument("--t0", default="ml/taxonomies/t0.yaml")
    ap.add_argument("--model", default="minilm", choices=list(EMBEDDER_MAP))
    ap.add_argument("--out_dir", default="ml/out")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auto_threshold", action="store_true", help="Search a small grid of thresholds each epoch")
    ap.add_argument("--no_pos_weight", action="store_true", help="Disable positive class weighting")
    train(ap.parse_args())
