"""Generate offline SBI images from existing processed real-face datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sampler import filter_by_label, load_manifest, sample_without_replacement, split_by_group
from data.sbi_generator import generate_sbi_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare offline SBI images from processed real samples.")
    parser.add_argument("--celebdf-root", type=str, default="data/processed/celebdf")
    parser.add_argument("--ffpp-root", type=str, default="data/processed/ffpp_generalization")
    parser.add_argument("--output-root", type=str, default="data/processed/sbi_offline")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_real_samples(celebdf_root: Path, ffpp_root: Path) -> list[dict]:
    celebdf_train = load_manifest(celebdf_root / "celebdf_train_manifest.jsonl")
    ffpp_train = load_manifest(ffpp_root / "ffpp_generalization_train_manifest.jsonl")
    return filter_by_label(celebdf_train, 0) + filter_by_label(ffpp_train, 0)


def main() -> None:
    args = parse_args()
    celebdf_root = Path(args.celebdf_root)
    ffpp_root = Path(args.ffpp_root)
    output_root = Path(args.output_root)

    real_pool = _load_real_samples(celebdf_root, ffpp_root)
    train_real, val_real = split_by_group(real_pool, val_ratio=args.val_ratio, seed=args.seed)

    if args.train_count is not None:
        train_real = sample_without_replacement(train_real, args.train_count, args.seed + 1)
    if args.val_count is not None:
        val_real = sample_without_replacement(val_real, args.val_count, args.seed + 2)

    train_sbi, train_manifest = generate_sbi_samples(
        real_samples=train_real,
        count=len(train_real),
        output_root=output_root,
        split_name="train",
        seed=args.seed + 3,
        overwrite=args.overwrite,
    )
    val_sbi, val_manifest = generate_sbi_samples(
        real_samples=val_real,
        count=len(val_real),
        output_root=output_root,
        split_name="val",
        seed=args.seed + 4,
        overwrite=args.overwrite,
    )

    print("Prepared offline SBI dataset")
    print("Real train pool used:", len(train_real))
    print("Real val pool used:", len(val_real))
    print("Generated train SBI:", len(train_sbi))
    print("Generated val SBI:", len(val_sbi))
    print("Train manifest:", train_manifest)
    print("Val manifest:", val_manifest)


if __name__ == "__main__":
    main()
