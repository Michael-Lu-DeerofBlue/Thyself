# Thyself: Local YouTube Title → Taxonomy Classification

Lean, privacy-first workflow:
1. A minimal Chrome (MV3) extension that locally stores only YouTube feed & shorts video titles (no navigation / dwell / generic page events, no network transmission).
2. A zero-shot hierarchical taxonomy classifier (two levels: t0 → t1) using sentence embeddings (no supervised training pipeline retained).

## Privacy & Scope
* The extension runs only on `youtube.com` / `m.youtube.com`.
* Captured data never leaves the browser; storage is IndexedDB inside the extension context.
* Only two event types are persisted: `feed_video` and `shorts_video`.
* Each stored record contains: `ts, type, title, videoId, href, page, platform`.
* Channel / length / badges / position are used transiently for extraction but intentionally discarded at persistence to minimize data surface.

## Updated Repository Layout
```
ml/
	taxonomies/t0.yaml        # Editable hierarchical taxonomy (parents with optional children)
	taxonomies/taxonomy.json  # Canonical JSON (generated) used for caching fingerprint
	src/models.py             # Embedder abstraction + mapping of short keys → HF models
	src/labels.py             # Basic label utilities
	src/{train,eval,infer,data}.py  # Deprecated stubs (left for backward import safety)
	data/titles.json          # Example exported titles placeholder
tools/
	apply_titles.py           # Unified zero-shot (flat or joint) classifier + detail diagnostics
	convert_taxonomy.py       # Robust YAML → JSON converter (handles inline map quirks)
extension/
	manifest.json             # Narrow-scoped MV3 manifest (YouTube only)
	content.js                # Harvests feed & shorts titles
	background.js             # Stores whitelisted events in IndexedDB
	dashboard.html/.js        # Popup: view / export / clear, export titles.json
Makefile                    # Helper targets (apply-titles, convert-taxonomy, clean-cache)
requirements.txt            # Python dependencies
README.md                   # This document
```

## Installation (Python)
Python 3.10+ recommended:
```powershell
pip install -r requirements.txt
```

## Loading the Extension
1. Open `chrome://extensions` (enable Developer Mode).
2. Load unpacked → select the `extension/` directory.
3. Scroll YouTube feed / shorts; the popup (action icon) lets you:
	 * Refresh & inspect recent stored events
	 * Export all raw stored events (`local-events.json`)
	 * Export a compact titles file (`titles.json`) → `{ "user_id": <id>, "titles": [...] }`
	 * Delete All (clears IndexedDB)

## Event Schema (Persisted)
```jsonc
{
	"ts": 1720000000000,          // epoch ms
	"type": "feed_video" | "shorts_video",
	"title": "Video Title Text",
	"videoId": "abcDEF123",
	"href": "https://www.youtube.com/watch?v=abcDEF123",
	"page": "/",
	"platform": "youtube"
}
```

## Taxonomy & Conversion
Editable source: `ml/taxonomies/t0.yaml`. Convert to canonical JSON (sorted & normalized):
```powershell
python tools/convert_taxonomy.py --input ml/taxonomies/t0.yaml --output ml/taxonomies/taxonomy.json
```
Caching uses the SHA1 fingerprint of the taxonomy file plus model + prompt style.

## Zero-Shot Hierarchical Classification
Scoring formula (joint mode):
```text
combined = (1 - alpha) * child_similarity + alpha * parent_similarity
```
Run (joint mode, MiniLM default):
```powershell
python tools/apply_titles.py --mode zero-shot-joint `
	--taxonomy ml/taxonomies/t0.yaml `
	--input ml/data/titles.json `
	--output ml/data/titles_tagged.json `
	--alpha 0.30 --topk-parent 8 --embedder minilm
```
Flat parent-only mode:
```powershell
python tools/apply_titles.py --mode zero-shot --input ml/data/titles.json --output ml/data/titles_tagged_flat.json
```

### Detail Diagnostics
Add `--detail` to emit per-title ranking objects:
```powershell
python tools/apply_titles.py --mode zero-shot-joint --detail --topk-children 12 `
	--input ml/data/titles.json --output ml/data/titles_tagged_detail.json
```
Joint mode detail payload excerpt:
```jsonc
"Some Title": {
	"t0": "ParentLabel",
	"t1": "ChildLabel",
	"top_children": [
		{ "t0": "ParentLabel", "t1": "ChildA", "combined": 0.71, "parent_score": 0.62, "child_score": 0.78 },
		{ "t0": "ParentLabel", "t1": "ChildB", "combined": 0.66, "parent_score": 0.62, "child_score": 0.70 }
	]
}
```

### Embedders & Automatic E5 Prompting
Short keys (from `ml/src/models.py`):
* `minilm`, `mpnet`, `e5`, `me5`, `me5large`, `multiminilm`, `muse`

If the model name contains `e5`, label texts use `passage:` and title inputs use `query:` automatically (E5 asymmetric retrieval style). Cache file names are sanitized (slashes → `__`).

### Caching
Per model + prompt style: `ml/out/label_cache_<model>.npz` holding parent/child embeddings & metadata. Invalidate by editing taxonomy YAML → reconvert → different fingerprint.

## Export → Classify Workflow
1. In popup, click “Export Titles JSON” → save as `ml/data/titles.json` (or any path).
2. Classify:
```powershell
python tools/apply_titles.py --mode zero-shot-joint --input ml/data/titles.json --output ml/data/titles_tagged.json --alpha 0.3 --topk-parent 8
```
3. (Optional) Detailed ranking for analysis:
```powershell
python tools/apply_titles.py --mode zero-shot-joint --detail --topk-children 15 --input ml/data/titles.json --output ml/data/titles_tagged_detail.json
```

## Makefile Convenience
```powershell
make apply-titles input=ml/data/titles.json output=ml/data/titles_tagged.json alpha=0.25 embedder=me5 topk_parent=8 detail=1 topk_children=12
make convert-taxonomy
make clean-cache
```
Defaults (if variables omitted):
* mode = zero-shot-joint
* embedder = minilm
* alpha = 0.3
* topk_parent = 8

## Deprecated Stubs (Intentional)
Files left only to avoid breaking historical imports: `ml/src/train.py`, `eval.py`, `infer.py`, `data.py`. They exit with an explanatory message if run. All functionality lives in `tools/apply_titles.py` now.

## Future Enhancements (Ideas)
* Confidence / abstain threshold (e.g. do not assign if best combined < τ).
* Optional normalization of scores to pseudo-probabilities (softmax on scaled similarities).
* Lightweight language detection + translation pre-pass for niche languages.
* Additional taxonomy depth (t2) with recursive joint scoring.

## Rationale for Zero-Shot Only
Embeddings provide strong alignment for short titles; maintaining a supervised pipeline (data cleaning, training, evaluation, hyperparameters) added complexity without clear marginal gain for the current small taxonomy. Removal reduces operational and cognitive overhead.

## Reproducibility / Determinism
* SentenceTransformers handles internal seeding; deterministic label embedding cache ensures stable rankings for unchanged taxonomy + model.
* Caches are content-addressed via taxonomy fingerprint + model key + prompt style.

## Quick PowerShell Examples
```powershell
# Convert taxonomy
python tools/convert_taxonomy.py --input ml/taxonomies/t0.yaml --output ml/taxonomies/taxonomy.json

# Joint multilingual E5
python tools/apply_titles.py --mode zero-shot-joint --embedder intfloat/multilingual-e5-base --input ml/data/titles.json --output ml/data/titles_tagged_e5.json

# Detailed ranking (MiniLM)
python tools/apply_titles.py --mode zero-shot-joint --detail --topk-children 10 --input ml/data/titles.json --output ml/data/titles_tagged_detail.json
```

## License / Disclaimer
Example data is illustrative only. Evaluate taxonomy quality and embedding appropriateness for your domain before production use.

---
Lean extension + zero-shot classifier: minimal surface, maximal transparency.