"""Sampling helpers for clean dataset preparation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def load_manifest(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            samples.append(json.loads(line))
    return samples


def save_manifest(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")


def sample_without_replacement(samples: list[dict], count: int, seed: int) -> list[dict]:
    if count > len(samples):
        raise ValueError(f"Requested {count} samples, but only {len(samples)} are available.")
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[:count]


def compute_total_target(pool_sizes: dict[str, int], ratios: dict[str, float]) -> int:
    candidates = []
    for name, ratio in ratios.items():
        if ratio <= 0:
            continue
        candidates.append(int(pool_sizes[name] / ratio))
    return min(candidates) if candidates else 0


def compute_counts(total_target: int, ratios: dict[str, float]) -> dict[str, int]:
    counts = {name: int(total_target * ratio) for name, ratio in ratios.items()}
    assigned = sum(counts.values())
    if assigned < total_target:
        remainders = sorted(
            ((name, total_target * ratio - counts[name]) for name, ratio in ratios.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        for name, _ in remainders[: total_target - assigned]:
            counts[name] += 1
    return counts


def filter_by_label(samples: list[dict], label: int) -> list[dict]:
    return [sample for sample in samples if int(sample["label"]) == label]


def filter_ffpp_fake_types(samples: list[dict], fake_types: set[str]) -> list[dict]:
    return [sample for sample in samples if sample.get("manipulation_type") in fake_types and int(sample["label"]) == 1]


def split_by_group(samples: list[dict], val_ratio: float, seed: int, key: str = "source_video") -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        group_value = str(sample.get(key) or sample.get("video_id") or sample.get("image_path"))
        grouped[group_value].append(sample)

    group_keys = list(grouped.keys())
    random.Random(seed).shuffle(group_keys)
    val_size = int(len(group_keys) * val_ratio)
    val_groups = set(group_keys[:val_size])

    train_samples: list[dict] = []
    val_samples: list[dict] = []
    for group_key, group_samples in grouped.items():
        if group_key in val_groups:
            val_samples.extend(group_samples)
        else:
            train_samples.extend(group_samples)
    return train_samples, val_samples
