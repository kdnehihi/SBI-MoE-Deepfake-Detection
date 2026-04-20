"""Helpers for organizing prepared datasets without copying original frames."""

from __future__ import annotations

import shutil
from pathlib import Path

from data.sampler import save_manifest


def _target_name(sample: dict) -> str:
    dataset_name = sample.get("dataset_name", "dataset").replace("/", "_")
    manipulation = sample.get("manipulation_type", "none").replace("/", "_")
    video_id = str(sample.get("video_id", "video")).replace("/", "_")
    frame_index = int(sample.get("frame_index", 0))
    return f"{dataset_name}__{manipulation}__{video_id}__{frame_index:05d}.png"


def _link_or_copy(source: Path, target: Path, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            return
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def materialize_split(samples: list[dict], output_root: Path, manifest_name: str, overwrite: bool = False) -> Path:
    manifest_samples = []

    for sample in samples:
        label_dir = "fake" if int(sample["label"]) == 1 else "real"
        target_path = output_root / label_dir / _target_name(sample)
        _link_or_copy(Path(sample["image_path"]), target_path, overwrite=overwrite)

        updated = dict(sample)
        updated["image_path"] = str(target_path)
        manifest_samples.append(updated)

    manifest_path = output_root.parent / manifest_name
    save_manifest(manifest_samples, manifest_path)
    return manifest_path
