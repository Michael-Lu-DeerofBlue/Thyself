PY=python
DATA=ml/data/example.csv
T0=ml/taxonomies/t0.yaml
OUT=ml/out
MODEL_PT=$(OUT)/model.pt
CONFIG=ml/configs/default.yaml

.PHONY: train eval infer export tests

train:
	$(PY) ml/src/train.py --data $(DATA) --t0 $(T0) --out_dir $(OUT)

eval:
	@echo "(Placeholder) run evaluation metrics after training"
	# Could load saved logits/scores; integrate as needed.

infer:
	$(PY) ml/src/infer.py --titles "$(title)"

infer-supervised:
	$(PY) ml/src/infer.py --titles "$(title)" --supervised

export:
	@echo Zipping extension directory...
	powershell -Command "Compress-Archive -Path extension/* -DestinationPath extension.zip -Force"

tests:
	pytest -q
