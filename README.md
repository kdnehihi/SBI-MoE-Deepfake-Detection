# Multi-View Deepfake Detection with Mixture-of-Experts

## Overview

This project is a research scaffold for deepfake detection from multi-view face image sets using a Mixture-of-Experts (MoE) formulation. Each sample is defined by multiple images of the same subject captured from different views or angles, with the goal of leveraging complementary cross-view evidence for robust fake-versus-real classification.

The project focuses on multi-view images rather than video in order to study view diversity without requiring temporal modeling. This setting is useful when only sparse observations are available, when synchronized video is not accessible, or when the research goal is to isolate how viewpoint variation affects forensic representation learning.

Mixture-of-Experts is used to support adaptive specialization. Different experts can learn complementary behaviors for view-specific artifacts, identity-preserving cues, and synthesis inconsistencies, while a routing mechanism can dynamically emphasize the most relevant experts for a given multi-view sample.

## Key Ideas

- Multi-view representation for aggregating complementary facial evidence across different viewpoints.
- Global vs local feature modeling through a backbone with lightweight specialization mechanisms such as LoRA and adapters.
- Dynamic expert routing with MoE to adapt computation to varying artifact types, view conditions, and sample difficulty.

## Project Structure

```text
project_root/
|
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- multiview/
|
|-- models/
|   |-- backbone.py
|   |-- lora.py
|   |-- adapter.py
|   |-- moe.py
|   `-- multiview.py
|
|-- training/
|   |-- train.py
|   |-- trainer.py
|   `-- loss.py
|
|-- evaluation/
|   |-- metrics.py
|   |-- evaluator.py
|   `-- visualization.py
|
|-- configs/
|   `-- default.yaml
|
|-- scripts/
|   `-- run.sh
|
|-- utils/
|   `-- helpers.py
|
|-- requirements.txt
`-- README.md
```

## Planned Method

At a high level, the intended pipeline begins with a set of multiple face images corresponding to the same sample. A vision backbone, such as a Vision Transformer, is used to extract per-view representations from each image while preserving a common embedding space across views.

These view-level features are then combined through a multi-view fusion stage designed to capture complementary evidence across pose and appearance variation. The fused representation is passed through an MoE module, where routing selects or weights specialized experts according to the observed sample characteristics. The final stage performs binary classification for deepfake detection.

## Future Work

- Cross-view attention for stronger interaction between view-specific representations.
- View-aware routing strategies that explicitly incorporate pose, quality, or viewpoint metadata.
- Robustness and generalization studies across identities, datasets, compression settings, and unseen manipulation methods.

## Notes

This is a research scaffold. Implementation will be added incrementally.
