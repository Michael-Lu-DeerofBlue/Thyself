import argparse, json, numpy as np, torch, os
from .models import Embedder, MLPHead, EMBEDDER_MAP
from .labels import load_t0, build_label_matrix

DEFAULT_T0 = "ml/taxonomies/t0.yaml"

# Zero/low-shot ranking

def rank_titles(titles, model="minilm", t0=DEFAULT_T0, topk=10, temperature=1.0):
    t0_labels = load_t0(t0)
    emb = Embedder(model)
    L = build_label_matrix(t0_labels, emb)     # [K,D]
    V = emb.encode(titles)                     # [N,D]
    sims = (V @ L.T) / temperature
    results = []
    for row in sims:
        order = np.argsort(-row)
        results.append([(t0_labels[j]["id"], float(row[j])) for j in order[:topk]])
    return results

# Supervised inference (MLP head logits -> sigmoid)

def supervised_infer(titles, model_dir="ml/out/model.pt", topk=10):
    ckpt = torch.load(model_dir, map_location="cpu")
    embedder_key = ckpt["embedder"]
    emb = Embedder(embedder_key)
    V = emb.encode(titles)
    head = MLPHead(ckpt["in_dim"], ckpt["num_labels"])
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    with torch.no_grad():
        logits = head(torch.tensor(V))
        probs = torch.sigmoid(logits).numpy()
    t0_labels = load_t0()
    out = []
    for row in probs:
        order = np.argsort(-row)
        out.append([(t0_labels[j]["id"], float(row[j])) for j in order[:topk]])
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", nargs="+", required=True)
    ap.add_argument("--model", default="minilm", choices=list(EMBEDDER_MAP))
    ap.add_argument("--t0", default=DEFAULT_T0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--supervised", action="store_true", help="Use supervised MLP head")
    ap.add_argument("--model_dir", default="ml/out/model.pt")
    args = ap.parse_args()
    if args.supervised:
        res = supervised_infer(args.titles, args.model_dir, args.topk)
    else:
        res = rank_titles(args.titles, args.model, args.t0, args.topk, args.temperature)
    print(json.dumps(res, ensure_ascii=False, indent=2))
