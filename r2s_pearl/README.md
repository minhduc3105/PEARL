# PEARL → R2S100K train pseudo masks

This adapter calls the upstream `PEARL.test_step` with MMSeg's `LoadImageFromFile → Resize(2048×336, keep_ratio=True) → PackSegInputs` pipeline. PEARL's data preprocessor, Procrustes attention, TLP, probability aggregation, and postprocessing stay in the upstream modules. Ground truth is loaded only by validation and evaluation. The local taxonomy is an exact copy of `../SED/sed_r2s100k/taxonomy.py` at SHA-256 `87d92fdd9d7652a256abeccd57235fc602de00167c5fb3b775edc034eb6bcd30`.

Use Python 3.10 and a CUDA-capable PyTorch build. Create the environment with `uv venv --python 3.10 .venv-pearl`, then install `uv pip install --python .venv-pearl/bin/python -r r2s_pearl/requirements.txt` and the full MMCV 2.1.0 wheel matching Torch 2.1.0 and your CUDA version. For CUDA 12.1 and Python 3.10: `uv pip install --python .venv-pearl/bin/python --no-deps https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl`. The upstream README's MMCV 2.2.0 recommendation may fail MMSeg's version check; verify imports first. Also install NumPy <2, Pillow, matplotlib, ftfy, regex, and setuptools <81. The default CLIP ViT-B/16 checkpoint is downloaded to `~/.cache/clip` by upstream `clip.load` if absent. For offline use pass `--checkpoint /path/to/ViT-B-16.pt`.

From the PEARL root with `.venv-pearl/bin` on `PATH`:

```bash
python -m unittest r2s_pearl.test_adapter
python -m r2s_pearl inspect --max-images 3
python -m r2s_pearl run --max-images 3 --output runs/r2s_smoke
python -m r2s_pearl run --max-images 100 --output runs/r2s_100
python -m r2s_pearl run --output runs/r2s_train
python -m r2s_pearl run --output runs/r2s_train --resume
python -m r2s_pearl evaluate --output runs/r2s_train
```

The default output is `runs/r2s100k_train` inside this project. All output paths must resolve inside the project. Defaults for input data: `/run/media/minhduc/D/Study/r2s100k/train` and `/run/media/minhduc/D/Study/r2s100k/train_labels/Train-Labels`. Override with `--images-dir` and `--labels-dir`; `--no-labels` generates masks without GT. `--filtered-threshold 0.8` optionally writes a separate mask with discarded pixels set to ID 255. The primary `masks/` always retains IDs 0–14 at original resolution. Confidence is the final predicted-class probability returned by PEARL. `inspect` validates every selected GT mask's colors and dimensions. Full runs validate all images before inference.

Metrics are computed from one confusion matrix over the full selection; ID 0 participates in mIoU. Missing classes are null and excluded from the mean. `per_image.csv` is for error inspection. `run.json` captures versions, commit, checkpoint/vocabulary/taxonomy hashes and timing. `manifest.json` records input file metadata and the config hash, and `progress.json` records output hashes; resume reruns any missing or corrupted pair. Evaluating a prior run checks the manifest and hashes before reading masks.

The upstream TLP implementation uses `einsum('cc,bcij->bcij', ...)`, which consumes only diagonal text-similarity entries. This adapter intentionally preserves that behavior.
