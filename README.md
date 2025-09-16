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
Run tests (they mock heavy models):
```bash
make tests
```

## Metrics
`ml/src/eval.py` implements micro/macro F1, mAP@K, NDCG@K, plus temperature scaling utilities (extend as needed). The current Makefile has a placeholder `eval` target—integrate with stored logits or on-the-fly scoring when dataset grows.

## Taxonomy
`ml/taxonomies/t0.yaml` defines ~10 coarse topic buckets. Labels are embedded as: `"{name}: {description}"` and cosine similarity is used for zero-shot ranking.

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