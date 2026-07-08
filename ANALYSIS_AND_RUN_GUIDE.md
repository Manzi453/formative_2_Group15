# Group_15_Formative2.ipynb — Analysis & Run Guide

**Project:** Deep Learning for Malaria Diagnosis (binary image classification)
**Notebook:** `Group_15_Formative2.ipynb` · 20 cells (17 code, 3 markdown)
**Originally run on:** Google Colab, GPU, TensorFlow 2.20, mixed-precision (`mixed_float16`)
**Now runnable locally on:** this Apple-Silicon Mac, `.venv` (Python 3.11.15), TensorFlow 2.21 (CPU)

---

## 1. What the notebook does

It classifies single-cell blood-smear images as **Parasitized (infected, label 1)** vs **Uninfected (healthy, label 0)** using the NIH/NLM Malaria dataset (**27,558 images**, perfectly balanced 13,779 / 13,779).

It is a **benchmark harness**: it trains **6 model families × 7 experiments = 42 configs** (efficientnetb0 is commented out, so **5 families × 7 = 35 experiments actually run**), records metrics for each, and ranks the best experiment per model.

### Models compared
| Family | Type | Approach |
|---|---|---|
| `baseline_cnn` | from scratch | 2 conv blocks, 96×96 input |
| `advanced_cnn` | from scratch | 4–5 conv blocks + BatchNorm + augmentation, 96×96 |
| `vgg16_transfer` | transfer learning | frozen ImageNet backbone → cached features → small head |
| `resnet50_transfer` | transfer learning | same recipe |
| `mobilenetv2_transfer` | transfer learning | same recipe |
| `efficientnetb0_transfer` | transfer learning | **disabled** (commented out in cell 17) |

### The 7 experiments per model
Vary one factor at a time: default → lower LR → dropout change → augmentation on/off → smaller batch → capacity change → **Exp7 = fine-tuning the top ~20 backbone layers** (transfer models) or a combined-regularization run (scratch models).

### Key engineering decisions (this is a memory-careful notebook)
- **Split once**: 70/15/15 train/val/test on shuffled *file paths* (seed 42), never decoded images → RAM-safe.
- **`tf.data` lazy loading**: images decoded/resized on the fly, `prefetch`, no `.cache()`.
- **Feature caching for transfer models**: each frozen backbone runs over the data **once**, features are written to disk as `np.memmap` files in `feature_cache/`. All 7 experiments then train a tiny head on those cached features — massively faster than 7 full forward passes.
- **Resumability**: `run_scratch_model` / `run_transfer_model` skip any model whose `*_results.csv` already exists; partial CSVs are written after every experiment.
- **Two input sizes**: scratch CNNs use 96×96 (`SCRATCH_IMG_SIZE`), transfer models use 224×224 (`IMG_SIZE`).

---

## 2. Cell-by-cell map

| Cell | Kind | Purpose |
|---|---|---|
| 0–1 | md | Intro / malaria background |
| 2 | code | Imports, GPU detection, mixed-precision (GPU only), `free_memory()` |
| 3 | code | Dataset download (`downloadData=False` by default) |
| 4 | code | Global config: `SEED`, image sizes, batch sizes, auto-detect `DATA_DIR`, results/cache dirs |
| 5 | code | Load file paths, shuffle, 70/15/15 split, build lazy `tf.data` splits |
| 6 | code | Batch helper `make_ds`, build `train/val/test_ds` |
| 7 | code | Preview 9 sample images |
| 8 | code | Metric + plotting utilities (curves, confusion matrix, ROC) |
| 9 | code | Training runners + callbacks (EarlyStopping, ModelCheckpoint) + `finalize_experiment*` |
| 10 | code | Transfer backbone registry + feature extraction to memmap + head builder |
| 11 | code | Fine-tuning model builder (used only by Exp7 of transfer models) |
| 12 | code | `build_baseline` + `build_advanced` scratch CNN builders |
| 13 | code | `ALL_EXPERIMENTS` — the full config grid (7 per model) |
| 14 | code | `run_scratch_model()` |
| 15 | md | Section header |
| 16 | code | `run_transfer_model()` (cached-feature path + fine-tune path) |
| 17 | code | **Main driver** — trains everything, writes `ALL_experiments_combined.csv` |
| 18 | code | Best experiment per model + `MODEL_RANKING.csv` |
| 19 | code | Bar chart comparing best experiment per model |

---

## 3. Results (from the saved outputs of the original Colab run)

All 5 models score **~94–96% accuracy** and **~0.98–0.99 ROC-AUC** — malaria cell classification is a relatively easy, balanced task, so everything does well.

**Final ranking (ranked by recall → F1 → ROC-AUC — recall is prioritized, appropriate for a medical "don't-miss-the-disease" setting):**

| Rank | Model | Best experiment | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 1 | vgg16_transfer | Exp2_lower_learning_rate | 0.9516 | 0.9452 | **0.9573** | 0.9512 | 0.9888 |
| 2 | advanced_cnn | Exp6_deeper_filters | **0.9615** | 0.9728 | 0.9485 | **0.9605** | **0.9928** |
| 3 | resnet50_transfer | Exp7_fine_tune_top_layers | 0.9584 | 0.9679 | 0.9470 | 0.9573 | 0.9922 |
| 4 | mobilenetv2_transfer | Exp2_lower_learning_rate | 0.9478 | 0.9501 | 0.9435 | 0.9468 | 0.9862 |
| 5 | baseline_cnn | Exp5_smaller_batch_size | 0.9456 | 0.9467 | 0.9426 | 0.9446 | 0.9777 |

**Takeaways**
- VGG16 wins on the chosen metric (recall), but `advanced_cnn` (from scratch) has the best accuracy, F1, and AUC — a strong result showing a well-designed scratch CNN is competitive with transfer learning here.
- The single best config overall by accuracy in the full table is `advanced_cnn / Exp6_deeper_filters` (0.9615 acc, 0.9928 AUC).
- Fine-tuning (Exp7) gave the best result for resnet50, confirming it helps that backbone.

---

## 4. How to run it in VSCode  ✅ (environment is already set up)

I installed all dependencies into your existing `.venv` and registered it as a Jupyter kernel. A full end-to-end **smoke test passed** (scratch CNN, MobileNetV2 feature extraction, head training, metrics all run).

### Steps
1. **Open the folder** in VSCode and open `Group_15_Formative2.ipynb`.
2. **Select the kernel** (top-right of the notebook): choose **`Python (ML formative malaria)`** — or pick the interpreter at `.venv/bin/python`.
   - Make sure the VSCode **Python** and **Jupyter** extensions are installed.
3. **Get the dataset.** The 353 MB dataset is *not* on disk. In **cell 3**, set `downloadData = True` for the first run:
   ```python
   downloadData = True
   ```
   This downloads and unzips `cell_images/` (Parasitized + Uninfected). Set it back to `False` afterward so you don't re-download.
   - Cell 4 auto-detects the folder; locally it resolves `DATA_DIR = "cell_images/cell_images"` or `"cell_images"`.
4. **Run cells top to bottom.**

### Important: this is CPU-only here, and training is heavy
Your Mac has no CUDA GPU and `tensorflow-metal` is incompatible with TF 2.21, so it runs on **CPU in float32** (the notebook handles this automatically — mixed precision only activates on GPU). Training all 35 experiments over 27k images on CPU would take **many hours to days**. Recommended for a local run — reduce scope before executing cell 17:

- **Cell 4:** lower `MAX_EXPERIMENTS_TO_RUN` (e.g. `2`) to run only the first 2 experiments per model.
- **Cell 17:** trim `MODELS_TO_RUN` to one or two models, e.g. `["baseline_cnn", "mobilenetv2_transfer"]`.
- Optionally lower `epochs` in the `ALL_EXPERIMENTS` configs (cell 13) or take a subset of files in cell 5 (e.g. slice `all_paths`/`all_labels` to the first ~3000) for a fast demo.
- Transfer models are cheaper than they look after the one-time feature extraction thanks to the memmap cache.

The notebook's resume logic means finished models are skipped on re-run (their CSV in `results_all_models/` is reused), so you can build up results incrementally.

### Reproduce the environment elsewhere
`requirements.txt` was written to the project root:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name ml-formative-malaria --display-name "Python (ML formative malaria)"
```

---

## 5. Notes, caveats & minor issues

- **Version drift:** original ran on TF 2.20 / GPU; local is TF 2.21 (Keras 3) / CPU. The code is compatible, but tiny numerical differences vs. the saved outputs are expected if you re-run.
- **NumPy 2.x / pandas 3.x:** newer than a typical 2024 Colab. No API used here is affected, but if you ever hit a dtype warning it's from this.
- **`downloadData` default is `False`** and there is no dataset locally — running cell 5 before downloading will fail with "0 files" / empty splits. Do step 3 first.
- **`efficientnetb0_transfer`** is defined but excluded from `MODELS_TO_RUN` — enable it by uncommenting the line in cell 17 if you want all six.
- **Outputs directories** `results_all_models/` and `feature_cache/` will be created in the project root on first run and can get large (cached features are hundreds of MB).
- No `random`-based leakage: split is seeded (`SEED=42`) and done on paths before any training.