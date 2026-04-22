# SBI-MoE-Deepfake-Detection

## Overview

This repository contains a research-oriented PyTorch implementation of a Mixture-of-Experts (MoE) deepfake detector inspired by MoE-FFD.

The current model design uses:

- a frozen ViT backbone
- LoRA for lightweight global adaptation
- conv-based adapters for local artifact modeling
- MoE routing for expert specialization
- a binary classifier for `real` vs `fake`

The codebase now supports two parallel experiment tracks:

1. a reproducible **clean baseline**
2. a **staged training framework** with offline SBI generation and progressive expert activation

The baseline must remain runnable and unchanged in spirit even as the staged framework evolves.

## Baseline Mode

### Training policy

Paper-like baseline protocol:

- Train: `FF++ train`
- Valid: `FF++ valid`, split from `FF++ train` at video level
- Test-1: `FF++ test`
- Test-2: `Celeb-DF test` as out-of-domain evaluation

Within FF++:

- Real:
  - `original`
- Fake:
  - `Deepfakes`
  - `FaceSwap`
  - `Face2Face`
  - `NeuralTextures`

Important:

- baseline training data remains `100% FF++`
- Celeb-DF is not used in baseline training
- no SBI
- no staged curriculum
- same MoE-FFD-style architecture, pulled closer to the paper
- frozen ViT backbone
- paper-like video evaluation during validation and testing

### Evaluation policy

- validation uses FF++ videos
- `FF++` remains the in-domain test
- `Celeb-DF` remains the out-of-domain test

## Staged Training Mode

The staged framework is intentionally progressive rather than single-stage.

### Stage 1: SBI Pretraining

Goal:

- learn generic forgery cues first

Model:

- ViT frozen
- LoRA on
- adapter off
- MoE router off
- classifier on

Data:

- real images
- SBI-generated fake images

Suggested ratio:

- `50% real`
- `50% SBI fake`

### Stage 2: Real Deepfake Adaptation

Goal:

- adapt from synthetic forgery cues to realistic manipulation patterns

Model:

- ViT frozen
- LoRA on
- adapter off
- MoE router off
- classifier on

Default data idea:

- Real: `40%`
- Fake: `60%`

Within fake:

- FF++ dominant
- Celeb-DF secondary
- SBI disabled by default in this stage

### Stage 3: Full MoE Specialization

Goal:

- enable local specialization and expert diversity

Model:

- ViT frozen
- LoRA on
- adapter on
- MoE router on
- classifier on

Recommended data idea:

- Real: `40%`
- Fake: `60%`

Within fake:

- FF++: `60%`
- Celeb-DF: `25%`
- SBI: `15%`

### Checkpoint flow

The staged pipeline is sequential:

- Stage 2 loads the checkpoint from Stage 1
- Stage 3 loads the checkpoint from Stage 2

## Important Data Constraints

- Existing Celeb-DF frames are reused.
- Celeb-DF raw videos are no longer required.
- FF++ may be re-extracted from raw videos as needed.
- Existing extracted frames are not deleted unless the new FF++ extraction pipeline is explicitly run with reset enabled.
- train/valid/test leakage should be avoided; the code uses source-video-level separation where possible.
- current processed manifests contain about `7-8` frames per video, so the video-style evaluator can request `20` or `100` frames but will only use the available extracted frames unless FF++ / Celeb-DF are re-extracted more densely.

## Repository Structure

```text
moe-deepfake/
├── data/
│   ├── dataset.py
│   ├── dataset_builder.py
│   ├── extract_ffpp_faces_fps.py
│   ├── prepare_baseline_clean.py
│   ├── prepare_stage_datasets.py
│   ├── prepare_with_sbi.py
│   ├── sampler.py
│   ├── sbi_generator.py
│   ├── transforms.py
│   ├── video_to_frames.py
│   └── video_to_frames_ffpp.py
├── engine/
│   ├── eval.py
│   ├── loss.py
│   └── train.py
├── models/
│   ├── adapter_experts.py
│   ├── gating.py
│   ├── model.py
│   ├── moe_adapter.py
│   ├── moe_lora.py
│   ├── transformer_block.py
│   └── vit_backbone.py
├── utils/
│   ├── config.py
│   ├── metrics.py
│   └── stage_presets.py
├── main.py
├── train_baseline.py
├── train_stage_common.py
├── train_stage1.py
├── train_stage2.py
├── train_stage3.py
├── requirements.txt
└── README.md
```

## Clean Baseline Pipeline

### 1. Re-extract FF++ with fixed frames per video

```bash
python data/extract_ffpp_faces_fps.py \
  --root data/raw/FaceForensics++_C23 \
  --processed-root data/processed/ffpp_generalization \
  --frames-per-video 8 \
  --image-size 224 \
  --device cpu \
  --reset-output
```

Notes:

- `--reset-output` removes only the output directory of the new FF++ extraction pipeline.
- it does **not** touch existing Celeb-DF processed frames.

### 2. Build the clean baseline dataset

```bash
python data/prepare_baseline_clean.py \
  --celebdf-root data/processed/celebdf \
  --ffpp-root data/processed/ffpp_generalization \
  --output-root data/baseline \
  --val-ratio 0.125 \
  --overwrite
```

Output structure:

```text
data/baseline/
├── train_manifest.jsonl
├── val_manifest.jsonl
├── test_ffpp_manifest.jsonl
└── test_celebdf_manifest.jsonl
```

Notes:

- `train_manifest.jsonl` and `val_manifest.jsonl` are both built from `FF++ train`
- validation is split at video level, stratified by manipulation type
- `Celeb-DF` only appears in `test_celebdf_manifest.jsonl`

### 3. Train the clean baseline

```bash
python -u train_baseline.py \
  --dataset-root data/baseline \
  --batch-size 32 \
  --epochs 20 \
  --num-workers 4 \
  --image-size 224 \
  --device cuda
```

If VRAM is tight, reduce `--batch-size` to `16`.

Video-style validation and testing use the extracted frames already present in each manifest. With the current processed data, that is typically about `8` frames per video.

### 4. Evaluate a saved baseline checkpoint

```bash
python evaluate_baseline.py \
  --dataset-root data/baseline \
  --checkpoint outputs/baseline_clean_last.pt \
  --device cuda
```

### 5. Colab notebook

Use [baseline_clean_pipeline.ipynb](/Users/khoatran/coding/llm/moe-deepfake/baseline_clean_pipeline.ipynb) for the end-to-end Colab flow with the same paper-like baseline protocol.

## Staged Pipeline Commands

### Build Stage 1 dataset

```bash
python data/prepare_stage_datasets.py \
  --stage stage1 \
  --celebdf-root data/processed/celebdf \
  --ffpp-root data/processed/ffpp_generalization \
  --output-root data/stages/stage1_sbi
```

### Train Stage 1

```bash
python train_stage1.py \
  --dataset-root data/stages/stage1_sbi \
  --batch-size 8 \
  --epochs 3 \
  --num-workers 0 \
  --image-size 224 \
  --device mps
```

### Build Stage 2 dataset

```bash
python data/prepare_stage_datasets.py \
  --stage stage2 \
  --celebdf-root data/processed/celebdf \
  --ffpp-root data/processed/ffpp_generalization \
  --output-root data/stages/stage2_real
```

### Train Stage 2

```bash
python train_stage2.py \
  --dataset-root data/stages/stage2_real \
  --init-checkpoint outputs/stage1_last.pt \
  --batch-size 8 \
  --epochs 5 \
  --num-workers 0 \
  --image-size 224 \
  --device mps
```

### Build Stage 3 dataset

```bash
python data/prepare_stage_datasets.py \
  --stage stage3 \
  --celebdf-root data/processed/celebdf \
  --ffpp-root data/processed/ffpp_generalization \
  --output-root data/stages/stage3_full
```

### Train Stage 3

```bash
python train_stage3.py \
  --dataset-root data/stages/stage3_full \
  --init-checkpoint outputs/stage2_last.pt \
  --batch-size 8 \
  --epochs 10 \
  --num-workers 0 \
  --image-size 224 \
  --device mps
```

## SBI Pipeline

The separate SBI-preparation path is still available through:

```bash
python data/prepare_with_sbi.py \
  --celebdf-root data/processed/celebdf \
  --ffpp-root data/processed/ffpp_generalization \
  --output-root data/with_sbi
```

This path is useful for direct dataset construction experiments outside the staged curriculum.

## Notebooks

Local-only notebooks currently kept:

- `baseline_clean_pipeline.ipynb`
- `moeffd_colab_drive_train.ipynb`

The second notebook is intentionally preserved as a fallback route for the earlier Celeb-DF-only project state.

## Notes

- `data/raw/`, `data/processed/`, generated datasets, checkpoints, outputs, and notebooks are ignored by Git.
- The repository tracks code only; datasets and generated artifacts stay local.
- Baseline mode and staged mode are intentionally kept separate so that new SBI/staged changes do not alter the clean baseline behavior.
