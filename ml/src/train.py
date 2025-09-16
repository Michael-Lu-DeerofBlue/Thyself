import argparse, os, json, numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from .models import Embedder, MLPHead, EMBEDDER_MAP
from .labels import load_t0
from .data import load_examples, train_val_split, TitleDataset, embed_dataset, set_seed


def train(args):
    set_seed(args.seed)
    t0 = load_t0(args.t0)
    examples, num_labels = load_examples(args.data, t0)
    train_ex, val_ex = train_val_split(examples, val_ratio=args.val_ratio)
    embedder = Embedder(args.model)
    train_ds = TitleDataset(train_ex, num_labels)
    val_ds = TitleDataset(val_ex, num_labels)
    Xtr, Ytr = embed_dataset(embedder, train_ds)
    Xva, Yva = embed_dataset(embedder, val_ds)

    in_dim = Xtr.shape[1]
    model = MLPHead(in_dim, num_labels, hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    Xtr_t = torch.tensor(Xtr)
    Ytr_t = torch.tensor(Ytr)
    Xva_t = torch.tensor(Xva)
    Yva_t = torch.tensor(Yva)

    best_f1 = -1
    patience = args.patience
    wait = 0
    for epoch in range(args.epochs):
        model.train()
        logits = model(Xtr_t)
        loss = F.binary_cross_entropy_with_logits(logits, Ytr_t)
        opt.zero_grad(); loss.backward(); opt.step()
        # val
        model.eval()
        with torch.no_grad():
            v_logits = model(Xva_t)
            v_probs = torch.sigmoid(v_logits).numpy()
            preds = (v_probs >= args.threshold).astype(int)
            tp = (preds * Yva).sum()
            fp = (preds * (1-Yva)).sum()
            fn = ((1-preds) * Yva).sum()
            precision = tp / (tp+fp+1e-9)
            recall = tp / (tp+fn+1e-9)
            f1 = 2*precision*recall/(precision+recall+1e-9)
        if f1 > best_f1:
            best_f1 = f1; wait = 0
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "in_dim": in_dim, "num_labels": num_labels, "embedder": args.model}, os.path.join(args.out_dir, "model.pt"))
        else:
            wait += 1
            if wait >= patience:
                break
    print(json.dumps({"best_f1": float(best_f1)}, indent=2))

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
    train(ap.parse_args())
