# Thyself: Local Title → Taxonomy Classification + Private Browsing Logger

This project combines:
1. An offline Chrome Manifest V3 extension that logs browsing interaction events **locally only**.
2. A machine learning pipeline mapping short video/article titles to a small T0-style topic taxonomy using both zero-shot (embedding similarity) and supervised (frozen encoder + MLP head) approaches.

## Privacy Statement
This extension **never sends data to the network**. All captured events remain in the user's browser IndexedDB. A user can view, export, or delete all stored data at any time via the dashboard popup. No remote endpoints are contacted.

## Repository Layout
```
ml/                 # ML pipeline
	data/example.csv  # Tiny illustrative dataset
	taxonomies/t0.yaml# ~10 bucket taxonomy (id, name, description)
	configs/          # Default config
	src/              # Core modules (models, infer, train, eval)
	tests/            # Unit tests (fast, mock embeddings)
extension/          # Chrome MV3 extension (local logging only)
tools/sample_titles.py # Glue: exported events -> inference JSONL
tools/apply_titles.py  # Apply taxonomy to titles.json -> tagged JSON (top-1)
Makefile            # Convenience commands
requirements.txt    # Python dependencies
PROMPT.md           # Original high-level specification
```

## Quick Start (ML)
Install dependencies (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```
Train supervised head (will download the sentence-transformer backbone on first run):
```bash
make train
```
Zero-shot inference for a single title:
```bash
make infer title="Advanced React Patterns Guide"
```
Supervised inference (after training):
```bash
make infer-supervised title="Advanced React Patterns Guide"
```

Apply taxonomy to an exported titles JSON (top-1 label per title):
```bash
# Supervised (requires ml/out/model.pt)
python tools/apply_titles.py --mode supervised --input ml/data/titles.json --output ml/data/titles_tagged.json

# Zero-shot (no training required)
python tools/apply_titles.py --mode zero-shot --input ml/data/titles.json --output ml/data/titles_tagged.json

# Or via Makefile (override variables as needed)
make apply-titles mode=supervised input=ml/data/titles.json output=ml/data/titles_tagged.json
```
Run tests (they mock heavy models):
```bash
make tests
```

## Metrics
`ml/src/eval.py` implements micro/macro F1, mAP@K, NDCG@K, plus temperature scaling utilities (extend as needed). The current Makefile has a placeholder `eval` target—integrate with stored logits or on-the-fly scoring when dataset grows.

## Taxonomy
`ml/taxonomies/t0.yaml` defines ~10 coarse topic buckets. Labels are embedded as: `"{name}: {description}"` and cosine similarity is used for zero-shot ranking.

### Converting / Normalizing the Taxonomy
If you edit the YAML and want a canonical JSON (used for fingerprint + caching), convert with:
```bash
python tools/convert_taxonomy.py --input ml/taxonomies/t0.yaml --output ml/taxonomies/taxonomy.json
```
The converter also salvages some malformed YAML lines (e.g. descriptions with extra colons) by auto‑quoting.

### Hierarchical Joint Zero-Shot Mode
We support a two-level (t0 → t1) joint scoring:
```
combined = (1 - alpha) * child_similarity + alpha * parent_similarity
```
Usage:
```bash
python tools/apply_titles.py \
	--mode zero-shot-joint \
	--taxonomy ml/taxonomies/t0.yaml \
	--input ml/data/titles.json \
	--output ml/data/titles_tagged.json \
	--alpha 0.30 \
	--topk-parent 8 \
	--output-format pair-string
```
Key parameters:
* `--alpha`: Blend weight (0 = only child, 1 = only parent influence).
* `--topk-parent`: Prunes child search to children of the top-K parents (speed + noise reduction). Omit for exhaustive search.
* `--output-format`: `pair-array | pair-string | object` (ignored when `--detail` is on; object is always used then).

### Multilingual & Custom Embedders
Argument `--embedder` accepts either a short key or any HuggingFace SentenceTransformer / E5 repo name:
Short keys (configured in `ml/src/models.py`):
* `minilm` → sentence-transformers/all-MiniLM-L6-v2 (EN)
* `mpnet` → sentence-transformers/all-mpnet-base-v2 (EN)
* `e5` → intfloat/e5-base-v2 (EN)
* `me5` → intfloat/multilingual-e5-base (multi)
* `me5large` → intfloat/multilingual-e5-large (multi)
* `multiminilm` → paraphrase-multilingual-MiniLM-L12-v2 (multi)
* `muse` → distiluse-base-multilingual-cased-v2 (multi)

Example multilingual run:
```bash
python tools/apply_titles.py \
	--mode zero-shot-joint \
	--embedder intfloat/multilingual-e5-base \
	--input ml/data/titles.json \
	--output ml/data/titles_tagged_e5.json
```

Caching: label embeddings are cached under `ml/out/label_cache_<sanitized_model_name>.npz`. Slashes in raw model names are replaced with `__`.

### Detail Mode (Integrated Diagnostics)
Instead of using a separate script, pass `--detail` to enrich each title with ranking lists:
* Joint mode: `top_children`: list of objects `{t0, t1, combined, parent_score, child_score}` (sorted desc by combined).
* Flat zero-shot (`--mode zero-shot`): `top_parents` list with parent scores.
* Control size with `--topk-children` (also applies to parent list in flat mode).

Example (joint + multilingual + detail):
```bash
python tools/apply_titles.py \
	--mode zero-shot-joint \
	--embedder me5 \
	--input ml/data/titles.json \
	--output ml/data/titles_tagged_detail.json \
	--alpha 0.25 \
	--topk-parent 8 \
	--detail \
	--topk-children 12
```

### Embedding Normalization
All embeddings are L2-normalized before scoring. Dot product between two normalized vectors equals cosine similarity, providing stable, length‑agnostic comparisons and bounded scores (≈[-1,1]). This improves consistency across different models and text lengths.

## Chrome Extension Usage
1. Open `chrome://extensions` and enable Developer Mode.
2. Click "Load unpacked" and select the `extension/` folder.
3. Browse pages—events (URL, title, dwell time, click counts, basic video play/pause/ended) are stored locally.
4. Click the extension icon to open the dashboard popup:
	 - Refresh to see latest events
	 - Export JSON (downloads `local-events.json`)
	 - Delete All to clear storage

## Export → Inference Workflow
1. Export events from the dashboard (downloads `local-events.json`).
2. Run:
```bash
python tools/sample_titles.py --input local-events.json --output results.jsonl
```
3. Output `results.jsonl` lines contain `{title, ranked:[(bucket, score)...]}`.

Alternatively, if you exported only titles via the dashboard's "Export Titles JSON" (e.g., to `ml/data/titles.json`), you can directly produce a mapping of title → top-1 taxonomy label:
```bash
# Supervised
python tools/apply_titles.py --mode supervised --input ml/data/titles.json --output ml/data/titles_tagged.json
# Zero-shot
python tools/apply_titles.py --mode zero-shot --input ml/data/titles.json --output ml/data/titles_tagged.json
```

## Extending
- Add more buckets or hierarchical structure in `t0.yaml`.
- Add calibration (`ml/src/eval.py`) and improved threshold tuning.
- Persist supervised logits for richer evaluation.
- Add UI controls in the dashboard for filtering or per-domain stats.

## Reproducibility
- Fixed random seed (42) used across training modules.
- Deterministic test embeddings (mock) ensure fast CI.
- Pin/constraint major library versions in `requirements.txt`.

## Disclaimer
The example dataset is tiny and illustrative only; for real performance you must curate a larger labeled corpus.

### Quick One-Liners (Windows PowerShell Examples)
```powershell
# Convert taxonomy YAML → JSON
python tools/convert_taxonomy.py --input ml/taxonomies/t0.yaml --output ml/taxonomies/taxonomy.json

# Joint zero-shot (multilingual E5)
python tools/apply_titles.py --mode zero-shot-joint --embedder intfloat/multilingual-e5-base --input ml/data/titles.json --output ml/data/titles_tagged_e5.json

# Joint zero-shot with detailed ranking
python tools/apply_titles.py --mode zero-shot-joint --embedder me5 --detail --topk-children 10 --topk-parent 8 --alpha 0.3 --input ml/data/titles.json --output ml/data/titles_tagged_detail.json
```