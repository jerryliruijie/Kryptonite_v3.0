# Sign + CatBoost: training and inference workflow

The script trains the previously selected sign-input CatBoost configuration with explicit dataset, training scope and seed. A user chooses the final policy with the group before evaluating local test or exporting hidden predictions. This workflow currently supports Kryptonite-20, -30 and -40.

## Train

Run from the repository root with the project environment. Output directories must be new.

```powershell
.\.venv\Scripts\python.exe -m B_Traditional_ML.catboost_pipeline train --dataset 20 --scope training --seed 67 --output-dir B_Traditional_ML/results/my_catboost_20
```

Use `.venv/bin/python` on macOS/Linux. `--scope training` fits the shared 70% training rows. `--scope training+validation` fits the saved training and validation rows; use this only once the group has agreed on that refit policy.

After the group has completed the final local-test evaluation and approved training on all labelled rows, the explicit `--scope all-labelled --allow-local-test-in-training` combination is available. This scope includes local-test labels and must not be used for an earlier test estimate. Supplying the flag with another scope, or selecting `all-labelled` without the flag, raises an error.

Each training directory contains `model.cbm`, `training_rows.npy` and `manifest.json`. The manifest records the selected original row numbers, scope, split and source-data fingerprints, seed, requested and resolved parameters, library versions, source hashes, model SHA-256 and runtime. It records whether local-test rows were included. All training-row class probabilities are checked for exact agreement between the fitted and the reloaded model. The `training_rows.npy` file contains the row indices; its SHA-256 is stored in the manifest.

## Predict

Prediction requires an explicit `.npy` input and a new output directory. No particular hidden or local-test input is selected by the script.

```powershell
.\.venv\Scripts\python.exe -m B_Traditional_ML.catboost_pipeline predict --model-dir B_Traditional_ML/results/my_catboost_20 --input path/to/input.npy --output-dir B_Traditional_ML/results/my_prediction
```

The output contains `predictions.npy` (one 0/1 integer per input row), `scores.npy` (the class-1 probabilities) and `inference_manifest.json` (file hashes, shape and row-order convention). To use a specific submission-compatible basename, add for example `--labels-filename y_predicted_20.npy`. The matching input and final output destination must be selected explicitly after group approval; run the repository submission checker on any final prediction package.

The Python API provides `train(dataset, scope, seed, output_dir)`, `load_model(model_dir)`, `predict(model_dir, X)` and `predict_file(model_dir, input_path, output_dir, labels_filename)`. `predict` and `predict_file` return `(labels, class_1_probabilities)` in the input row order. `train(..., allow_local_test=True)` is required for `all-labelled`. Model and training-row hashes are checked on load; input shape, finite values and class ordering are checked before prediction.

## Reproducibility check

`B_Traditional_ML/results/pipeline_smoke/` is a development smoke check trained on the K20 shared training scope with seed 67. It includes a five-row input drawn only from those training rows and an example prediction. The reloaded model's probabilities matched the fitted model on all 8,400 training rows; this check gives no local-test or hidden accuracy. The smoke model is not a locked final submission model.

Synthetic tests cover scope boundaries, required all-labelled consent flag, saved and reloaded predictions, original ordering, invalid input, fingerprints and overwrite prevention:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_catboost_pipeline.py -q
```
