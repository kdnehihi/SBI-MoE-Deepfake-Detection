"""Extract FF++ face crops with a fixed number of frames per video."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image


ALLOWED_SUBSETS = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def split_videos(video_paths: list[Path], train_ratio: float, seed: int) -> dict[str, list[Path]]:
    video_paths = video_paths[:]
    random.Random(seed).shuffle(video_paths)
    train_size = int(len(video_paths) * train_ratio)
    return {"train": video_paths[:train_size], "test": video_paths[train_size:]}


def collect_videos(root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for subset in ALLOWED_SUBSETS:
        subset_root = root / subset
        result[subset] = sorted(subset_root.glob("*.mp4")) + sorted(subset_root.glob("*.avi"))
    return result


def sample_frame_indices(total_frames: int, frames_per_video: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= frames_per_video:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, frames_per_video, dtype=int).tolist()


def crop_face(image: Image.Image, detector: MTCNN, margin: int) -> Image.Image | None:
    boxes, _ = detector.detect(image)
    if boxes is None:
        return None
    x1, y1, x2, y2 = boxes[0]
    width, height = image.size
    x1 = max(0, int(x1) - margin)
    y1 = max(0, int(y1) - margin)
    x2 = min(width, int(x2) + margin)
    y2 = min(height, int(y2) + margin)
    return image.crop((x1, y1, x2, y2))


def write_manifest(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")


def extract_video(
    video_path: Path,
    subset: str,
    split_name: str,
    detector: MTCNN,
    processed_root: Path,
    frames_per_video: int,
    image_size: int,
    margin: int,
    overwrite: bool,
) -> list[dict]:
    capture = cv2.VideoCapture(str(video_path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = set(sample_frame_indices(total_frames, frames_per_video))

    label = 0 if subset == "original" else 1
    samples = []
    frame_idx = 0
    video_id = video_path.stem
    save_dir = processed_root / split_name / subset / video_id
    save_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx not in frame_indices:
            frame_idx += 1
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face = crop_face(image, detector, margin)
        if face is not None:
            target_path = save_dir / f"{frame_idx:05d}.png"
            if overwrite or not target_path.exists():
                face.resize((image_size, image_size)).save(target_path)
            samples.append(
                {
                    "image_path": str(target_path),
                    "label": label,
                    "dataset_name": "FF++",
                    "split": split_name,
                    "video_id": video_id,
                    "frame_index": frame_idx,
                    "source_video": str(video_path),
                    "manipulation_type": subset,
                }
            )
        frame_idx += 1

    capture.release()
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract FF++ face crops with a fixed number of frames per video.")
    parser.add_argument("--root", type=str, default="data/raw/FaceForensics++_C23")
    parser.add_argument("--processed-root", type=str, default="data/processed/ffpp_generalization")
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reset-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    processed_root = Path(args.processed_root)

    if args.reset_output and processed_root.exists():
        print(f"Removing existing output directory: {processed_root}")
        shutil.rmtree(processed_root)

    detector_device = "cpu" if args.device == "mps" else args.device
    if args.device == "mps":
        print("MPS is unstable for MTCNN face detection on arbitrary frame sizes. Falling back to CPU detector.")
    detector = MTCNN(keep_all=False, device=detector_device)

    videos_by_subset = collect_videos(root)
    split_entries = {"train": [], "test": []}

    for subset in ALLOWED_SUBSETS:
        subset_splits = split_videos(videos_by_subset[subset], train_ratio=args.train_ratio, seed=args.seed)
        print(f"{subset}: {len(videos_by_subset[subset])} videos")
        for split_name in split_entries:
            split_entries[split_name].extend((subset, path) for path in subset_splits[split_name])

    for split_name, entries in split_entries.items():
        samples = []
        print(f"{split_name}: {len(entries)} videos")
        for index, (subset, video_path) in enumerate(entries, start=1):
            video_samples = extract_video(
                video_path=video_path,
                subset=subset,
                split_name=split_name,
                detector=detector,
                processed_root=processed_root,
                frames_per_video=args.frames_per_video,
                image_size=args.image_size,
                margin=args.margin,
                overwrite=args.overwrite,
            )
            samples.extend(video_samples)
            if index == 1 or index % 25 == 0 or index == len(entries):
                print(f"[{split_name}] {index}/{len(entries)} {subset}/{video_path.name} -> {len(video_samples)} faces")

        manifest_path = processed_root / f"ffpp_generalization_{split_name}_manifest.jsonl"
        write_manifest(samples, manifest_path)
        print(f"{split_name} done: {len(samples)} face images")
        print(f"{split_name} manifest: {manifest_path}")


if __name__ == "__main__":
    main()
