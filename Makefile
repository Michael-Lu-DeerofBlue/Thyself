PY=python
TAXO_YAML=ml/taxonomies/t0.yaml
TAXO_JSON=ml/taxonomies/taxonomy.json
CACHE_DIR=ml/out

.PHONY: export tests apply-titles convert-taxonomy clean-cache zero-shot-example

export:
	@echo Zipping extension directory...
	powershell -Command "Compress-Archive -Path extension/* -DestinationPath extension.zip -Force"

tests:
	pytest -q

# Apply taxonomy (zero-shot or joint). Usage overrides:
# make apply-titles mode=zero-shot-joint input=ml/data/titles.json output=ml/data/titles_tagged.json embedder=minilm alpha=0.3 topk_parent=8 detail=1
apply-titles:
	$(PY) tools/apply_titles.py --mode $(mode) --input $(input) --output $(output) --embedder $(embedder) --alpha $(alpha) --topk-parent $(topk_parent) $(if $(detail),--detail,) $(if $(topk_children),--topk-children $(topk_children),)

# Convert editable YAML taxonomy to canonical JSON (for stable caching)
convert-taxonomy:
	$(PY) tools/convert_taxonomy.py --input $(TAXO_YAML) --output $(TAXO_JSON)

# Remove cached label embeddings
clean-cache:
	@if exist $(CACHE_DIR) (powershell -Command "Get-ChildItem -Path $(CACHE_DIR) -Filter 'label_cache_*.npz' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue")
	@echo Cleared label cache files.

# Quick demo classification using defaults
zero-shot-example:
	$(PY) tools/apply_titles.py --mode zero-shot-joint --taxonomy $(TAXO_YAML) --input ml/data/titles.json --output ml/data/titles_tagged_example.json --alpha 0.3 --embedder minilm --topk-parent 8

# Sensible defaults if user omits variables
mode?=zero-shot-joint
input?=ml/data/titles.json
output?=ml/data/titles_tagged.json
embedder?=minilm
alpha?=0.3
topk_parent?=8
detail?=0
topk_children?=10
