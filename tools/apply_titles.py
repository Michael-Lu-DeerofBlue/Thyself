import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on sys.path so 'ml' package is importable when running from tools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from project modules
from ml.src.models import Embedder
import yaml
import re
import json as jsonlib
import hashlib
import numpy as np

# Utility to create filesystem-safe model key for cache filenames
def _safe_model_key(name: str) -> str:
    # Replace path separators and any whitespace with double underscores
    import re
    safe = re.sub(r'[\\/]+', '__', name)
    safe = re.sub(r'\s+', '_', safe)
    # Limit length to avoid extremely long filenames; keep hash suffix for uniqueness
    if len(safe) > 80:
        import hashlib
        h = hashlib.sha1(name.encode('utf-8')).hexdigest()[:8]
        safe = safe[:60] + '_' + h
    return safe


def load_titles(in_path: str) -> tuple[str, List[str]]:
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    user_id = data.get('user_id', '')
    titles: List[str] = []
    # Support multiple possible shapes:
    if isinstance(data.get('titles'), list):
        titles = data['titles']
    elif isinstance(data.get('titles_to_tag_map'), list):
        # Some exports use this key but store only a list of titles
        titles = data['titles_to_tag_map']
    elif isinstance(data, list):
        titles = data
    else:
        # Try common fallbacks
        for key in ('items', 'data'):
            if isinstance(data.get(key), list):
                titles = data[key]
                break
    if not titles:
        raise ValueError("No titles found. Expected 'titles' or a list under 'titles_to_tag_map'.")
    return user_id, titles


def build_id_to_name(taxonomy_path: str) -> Dict[Any, str]:
    # Build a mapping for parent (t0) ids to display names using hierarchical loader
    parents, _children = load_taxonomy_hier(taxonomy_path)
    return {p['id']: p.get('en') or str(p['id']) for p in parents}


def load_taxonomy_hier(path: str):
    # If JSON, load directly
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = jsonlib.load(f)
        if isinstance(data, dict) and 't0' in data:
            t0_list = data.get('t0') or []
        elif isinstance(data, list):
            t0_list = data
        else:
            raise ValueError(f"Unsupported taxonomy JSON format in {path}; expected mapping with 't0' or a list")
    else:
        with open(path, 'r', encoding='utf-8') as f:
            raw = f.read()
        # Remove any C-style block comments (placeholders like /* Lines X-Y omitted */)
        raw = re.sub(r"/\*.*?\*/", "\n", raw, flags=re.DOTALL)
        # Try to parse as-is
        def try_yaml(txt: str):
            try:
                return yaml.safe_load(txt)
            except Exception:
                return None
        data = try_yaml(raw)
        if data is None:
            # If the file contains multiple top-level blocks, attempt to keep the block starting at the first 't0:'
            if 't0:' in raw:
                raw2 = raw[raw.index('t0:'):]
                data = try_yaml(raw2)
            # If it starts with a list, wrap it under t0:
            if data is None and raw.lstrip().startswith('- '):
                wrapped = 't0:\n' + '\n'.join('  ' + line for line in raw.splitlines())
                data = try_yaml(wrapped)
            # As a last resort, try to parse as JSON
            if data is None:
                try:
                    data = jsonlib.loads(raw)
                except Exception as e:
                    raise ValueError(f"Failed to parse taxonomy file {path} as YAML or JSON: {e}")
        # Normalize shapes
        if isinstance(data, dict) and 't0' in data:
            t0_list = data.get('t0') or []
        elif isinstance(data, list):
            t0_list = data
        else:
            raise ValueError(f"Unsupported taxonomy format in {path}; expected mapping with 't0' or a list")
    parents = []
    children = []
    for p_idx, p in enumerate(t0_list):
        p_id = p['id']
        p_en = p.get('en', str(p_id))
        p_desc = p.get('desc', '')
        p_text = f"{p_en}: {p_desc}"
        parents.append({'id': p_id, 'en': p_en, 'desc': p_desc, 'text': p_text, 'p_index': p_idx})
        for c in p.get('t1', []) or []:
            c_id = c['id']
            c_en = c.get('en', str(c_id))
            c_desc = c.get('desc', '')
            c_text = f"{p_en} > {c_en}: {c_desc}"
            children.append({'id': f"{p_id}/{c_id}", 'p_id': p_id, 'p_en': p_en, 'en': c_en, 'desc': c_desc, 'text': c_text, 'p_index': p_idx})
    return parents, children


def taxonomy_fingerprint(path: str) -> str:
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def get_cached_label_embeddings(taxonomy_path: str, model_key: str, embedder, cache_dir: str = 'ml/out'):
    os.makedirs(cache_dir, exist_ok=True)
    fp = taxonomy_fingerprint(taxonomy_path)
    safe_key = _safe_model_key(model_key)
    cache_path = os.path.join(cache_dir, f'label_cache_{safe_key}.npz')
    # Determine prompt style (affects the actual encoded strings)
    is_e5_style = 'e5' in model_key.lower()  # heuristic: any e5 / multilingual-e5 variant
    prompt_style = 'e5' if is_e5_style else 'default'
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            meta = dict(data['meta'].item())
            if meta.get('taxonomy_fp') == fp and meta.get('model') == model_key and meta.get('prompt_style') == prompt_style:
                parents = data['parents'].tolist()
                children = data['children'].tolist()
                p_emb = data['p_emb']
                c_emb = data['c_emb']
                return parents, children, p_emb, c_emb
        except Exception:
            pass
    # Recompute
    parents, children = load_taxonomy_hier(taxonomy_path)
    if is_e5_style:
        # E5 expects asymmetric prefixes: query:/passage:
        # Parent prompt: passage: <ParentName>: <desc>
        # Child prompt: passage: <ParentName> > <ChildName>: <desc>
        p_texts = [f"passage: {p['en']}: {p.get('desc','')}" for p in parents]
        c_texts = [f"passage: {c['p_en']} > {c['en']}: {c.get('desc','')}" for c in children]
    else:
        p_texts = [p['text'] for p in parents]
        c_texts = [c['text'] for c in children]
    p_emb = embedder.encode(p_texts)
    c_emb = embedder.encode(c_texts) if c_texts else np.zeros((0, p_emb.shape[1]), dtype=p_emb.dtype)
    meta = {'taxonomy_fp': fp, 'model': model_key, 'cache_file': os.path.basename(cache_path), 'prompt_style': prompt_style}
    np.savez_compressed(cache_path, meta=meta, parents=np.array(parents, dtype=object), children=np.array(children, dtype=object), p_emb=p_emb, c_emb=c_emb)
    return parents, children, p_emb, c_emb


def zero_shot_joint_titles(titles: List[str], taxonomy_path: str, model_key: str = 'minilm', alpha: float = 0.3, topk_parent: int | None = None, detail: bool = False, topk_children: int = 10):
    emb = Embedder(model_key)
    parents, children, p_emb, c_emb = get_cached_label_embeddings(taxonomy_path, model_key, emb)
    is_e5_style = 'e5' in model_key.lower()
    if is_e5_style:
        title_inputs = [f"query: {t}" for t in titles]
    else:
        title_inputs = titles
    if len(children) == 0:
        # If no t1, fall back to parent-only zero-shot using cached parent embeddings
        p_mat = p_emb  # already normalized
        V = emb.encode(titles)
        sims = V @ p_mat.T
        out = []
        for row in sims:
            j = int(np.argmax(row))
            entry = {'t0': parents[j]['en'], 't1': None}
            if detail:
                entry['top_children'] = []
            out.append(entry)
        return out

    # Build parent index per child array
    child_pidx = np.array([c['p_index'] for c in children], dtype=np.int64)

    V = np.asarray(emb.encode(title_inputs))  # [N, D]
    P = np.asarray(p_emb)               # [P, D]
    C = np.asarray(c_emb)               # [C, D]

    # Compute similarities
    p_scores = V @ P.T                  # [N, P]
    # Optionally mask children by top-k parents
    mask = None
    if topk_parent is not None and topk_parent > 0 and topk_parent < len(parents):
        topP = np.argpartition(-p_scores, kth=topk_parent-1, axis=1)[:, :topk_parent]
        # Build a boolean mask [N, C] whether child's parent is in topP
        mask = np.zeros((V.shape[0], len(children)), dtype=bool)
        for n in range(V.shape[0]):
            allowed = set(topP[n].tolist())
            mask[n] = np.isin(child_pidx, list(allowed))

    c_scores = V @ C.T                  # [N, C]
    # Joint score
    combined = (1.0 - alpha) * c_scores + alpha * p_scores[:, child_pidx]

    results = []
    for i in range(V.shape[0]):
        if mask is not None:
            valid_idx = np.where(mask[i])[0]
            if valid_idx.size == 0:
                # fallback to best parent
                j = int(np.argmax(p_scores[i]))
                entry = {'t0': parents[j]['en'], 't1': None}
                if detail:
                    entry['top_children'] = []
                results.append(entry)
                continue
            local_scores = combined[i, valid_idx]
            best_local = valid_idx[int(np.argmax(local_scores))]
        else:
            best_local = int(np.argmax(combined[i]))
        best_child = children[best_local]
        entry = {'t0': best_child['p_en'], 't1': best_child['en']}
        if detail:
            # collect top-k children (by combined) restricted to mask or all
            if mask is not None:
                candidate_idx = valid_idx
                candidate_scores = combined[i, candidate_idx]
            else:
                candidate_idx = np.arange(len(children))
                candidate_scores = combined[i]
            order = np.argsort(-candidate_scores)[:topk_children]
            detail_list = []
            for pos in order:
                real_idx = candidate_idx[pos]
                ch = children[real_idx]
                detail_list.append({
                    't0': ch['p_en'],
                    't1': ch['en'],
                    'combined': float(candidate_scores[pos]),
                    'parent_score': float(p_scores[i, ch['p_index']]),
                    'child_score': float((V[i] @ c_emb[real_idx]))
                })
            entry['top_children'] = detail_list
        results.append(entry)
    return results


def predict_titles(titles: List[str], mode: str, model_path: str, taxonomy_path: str, topk: int = 1, alpha: float = 0.3, topk_parent: int | None = None, output_format: str = 'pair-array', embedder_key: str = 'minilm', detail: bool = False, topk_children: int = 10):
    if mode in ('zero-shot', 'zero-shot-flat'):
        # Flat zero-shot over parents only using cached embeddings
        emb = Embedder(embedder_key)
        parents, _children, p_emb, _c_emb = get_cached_label_embeddings(taxonomy_path, embedder_key, emb)
        is_e5_style = 'e5' in embedder_key.lower()
        if is_e5_style:
            title_inputs = [f"query: {t}" for t in titles]
        else:
            title_inputs = titles
        V = emb.encode(title_inputs)
        sims = V @ p_emb.T
        mapping: Dict[str, Any] = {}
        for i, title in enumerate(titles):
            j = int(np.argmax(sims[i]))
            if detail:
                # Provide top-k parent list if detail requested
                order = np.argsort(-sims[i])[:topk_children]
                mapping[title] = {
                    't0': parents[order[0]]['en'],
                    't1': None,
                    'top_parents': [
                        {'t0': parents[k]['en'], 'score': float(sims[i, k])} for k in order
                    ]
                }
            else:
                mapping[title] = parents[j]['en']
        return mapping
    else:  # zero-shot-joint hierarchical
        pairs = zero_shot_joint_titles(titles, taxonomy_path, model_key=embedder_key, alpha=alpha, topk_parent=topk_parent, detail=detail, topk_children=topk_children)
        mapping: Dict[str, Any] = {}
        for title, pair in zip(titles, pairs):
            t0 = pair['t0']
            t1 = pair['t1']
            if detail:
                # Always output object with detail list for consistency
                obj = {'t0': t0, 't1': t1, 'top_children': pair.get('top_children', [])}
                mapping[title] = obj
            else:
                if output_format == 'pair-array':
                    mapping[title] = [t0, t1] if t1 else [t0]
                elif output_format == 'pair-string':
                    mapping[title] = f"{t0}, {t1}" if t1 else f"{t0}"
                else:  # object
                    mapping[title] = {'t0': t0, 't1': t1}
        return mapping


def main():
    ap = argparse.ArgumentParser(description="Apply taxonomy classification (supervised or zero-shot hierarchical) to titles and output JSON.")
    ap.add_argument('--input', '-i', default='ml/data/titles.json', help='Path to input JSON with titles')
    ap.add_argument('--output', '-o', default='ml/data/titles_tagged.json', help='Path to write output JSON')
    ap.add_argument('--taxonomy', default='ml/taxonomies/taxonomy.json', help='Path to taxonomy file (JSON or YAML)')
    ap.add_argument('--model-path', default='ml/out/model.pt', help='Checkpoint path for supervised mode')
    ap.add_argument('--mode', choices=['zero-shot', 'zero-shot-joint'], default='zero-shot-joint', help='Prediction mode')
    ap.add_argument('--topk', type=int, default=1, help='How many labels to consider; output uses top-1')
    ap.add_argument('--alpha', type=float, default=0.3, help='Parent contribution for joint scoring (0-1)')
    ap.add_argument('--topk-parent', type=int, default=None, help='Restrict children to top-K parents (optional)')
    ap.add_argument('--output-format', choices=['pair-array', 'pair-string', 'object'], default='pair-array', help='How to represent per-title tags')
    ap.add_argument('--embedder', default='minilm', help='Embedder key or HuggingFace model name (supports multilingual).')
    ap.add_argument('--detail', action='store_true', help='If set, include detailed top-k scoring info directly (unifies scripts).')
    ap.add_argument('--topk-children', type=int, default=10, help='Top-K children (or parents in flat mode) to include when --detail.')
    args = ap.parse_args()

    # Supervised mode removed in zero-shot-only cleanup.

    user_id, titles = load_titles(args.input)
    mapping = predict_titles(
        titles,
        args.mode,
        args.model_path,
        args.taxonomy,
        args.topk,
        alpha=args.alpha,
        topk_parent=args.topk_parent,
        output_format=args.output_format,
        embedder_key=args.embedder,
        detail=args.detail,
        topk_children=args.topk_children,
    )
    out = {
        'user_id': user_id,
        'titles_to_tag_map': mapping,
        'mode': args.mode,
        'alpha': args.alpha,
        'topk_parent': args.topk_parent,
        'embedder': args.embedder,
        'detail': args.detail,
        'topk_children': args.topk_children if args.detail else None,
    }
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(args.output)


if __name__ == '__main__':
    main()
