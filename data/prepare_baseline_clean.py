"""Prepare the clean baseline dataset protocol for generalization experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sampler import (
    compute_counts,
    compute_total_target,
    filter_by_label,
    filter_ffpp_fake_types,
    load_manifest,
    save_manifest,
    sample_without_replacement,
)


FFPP_FAKE_TYPES = {"Deepfakes", "FaceSwap", "Face2Face", "NeuralTextures"}


def resolve_image_path(sample: dict, dataset_root: Path) -> dict:
    resolved = dict(sample)
    image_path = Path(str(sample["image_path"]))
    if image_path.is_absolute():
        resolved["image_path"] = str(image_path)
        return resolved

    parts = image_path.parts
    anchor = dataset_root.name
    if anchor in parts:
        anchor_index = parts.index(anchor)
        suffix = parts[anchor_index + 1 :]
        resolved["image_path"] = str(dataset_root.joinpath(*suffix))
    else:
        resolved["image_path"] = str((dataset_root / image_path.name).resolve())
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare baseline clean dataset.")
    parser.add_argument("--celebdf-root", type=str, default="data/processed/celebdf")
    parser.add_argument("--ffpp-root", type=str, default="data/processed/ffpp_generalization")
    parser.add_argument("--output-root", type=str, default="data/baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    celebdf_root = Path(args.celebdf_root)
    ffpp_root = Path(args.ffpp_root)
    output_root = Path(args.output_root)

    celebdf_train = load_manifest(celebdf_root / "celebdf_train_manifest.jsonl")
    celebdf_test = load_manifest(celebdf_root / "celebdf_test_manifest.jsonl")
    ffpp_train = load_manifest(ffpp_root / "ffpp_generalization_train_manifest.jsonl")
    ffpp_test = load_manifest(ffpp_root / "ffpp_generalization_test_manifest.jsonl")

    train_real_pool = filter_by_label(celebdf_train, 0) + filter_by_label(ffpp_train, 0)
    train_fake_pool = filter_ffpp_fake_types(ffpp_train, FFPP_FAKE_TYPES)

    ratios = {"real": 0.4, "fake": 0.6}
    total_target = compute_total_target(
        {"real": len(train_real_pool), "fake": len(train_fake_pool)},
        ratios,
    )
    counts = compute_counts(total_target, ratios)

    train_real = sample_without_replacement(train_real_pool, counts["real"], seed=args.seed)
    train_fake = sample_without_replacement(train_fake_pool, counts["fake"], seed=args.seed + 1)
    train_samples = train_real + train_fake

    ffpp_test_samples = filter_by_label(ffpp_test, 0) + filter_ffpp_fake_types(ffpp_test, FFPP_FAKE_TYPES)
    celebdf_test_samples = filter_by_label(celebdf_test, 0) + filter_by_label(celebdf_test, 1)

    output_root.mkdir(parents=True, exist_ok=True)

    train_manifest_path = output_root / "train_manifest.jsonl"
    ffpp_manifest_path = output_root / "test_ffpp_manifest.jsonl"
    celebdf_manifest_path = output_root / "test_celebdf_manifest.jsonl"

    if not args.overwrite:
        existing = [train_manifest_path, ffpp_manifest_path, celebdf_manifest_path]
        if all(path.exists() for path in existing):
            print("Baseline manifests already exist. Use --overwrite to rebuild them.")
            print("Train manifest:", train_manifest_path)
            print("FF++ test manifest:", ffpp_manifest_path)
            print("Celeb-DF test manifest:", celebdf_manifest_path)
            return

    train_manifest_samples = [
        resolve_image_path(sample, celebdf_root if sample.get("dataset_name") == "CelebDF" else ffpp_root)
        for sample in train_samples
    ]
    ffpp_manifest_samples = [resolve_image_path(sample, ffpp_root) for sample in ffpp_test_samples]
    celebdf_manifest_samples = [resolve_image_path(sample, celebdf_root) for sample in celebdf_test_samples]

    save_manifest(train_manifest_samples, train_manifest_path)
    save_manifest(ffpp_manifest_samples, ffpp_manifest_path)
    save_manifest(celebdf_manifest_samples, celebdf_manifest_path)

    print("Baseline clean dataset prepared.")
    print("Train manifest:", train_manifest_path)
    print("FF++ test manifest:", ffpp_manifest_path)
    print("Celeb-DF test manifest:", celebdf_manifest_path)
    print("Train counts:", {"real": len(train_real), "fake": len(train_fake), "total": len(train_samples)})


if __name__ == "__main__":
    main()
