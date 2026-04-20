"""Simple SBI-style synthetic fake generation from real face crops."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

from data.sampler import save_manifest


def _random_mask(size: tuple[int, int], rng: random.Random) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    shape_type = rng.choice(["ellipse", "rectangle", "multi"])
    if shape_type == "ellipse":
        x1 = rng.randint(width // 8, width // 3)
        y1 = rng.randint(height // 8, height // 3)
        x2 = rng.randint(2 * width // 3, 7 * width // 8)
        y2 = rng.randint(2 * height // 3, 7 * height // 8)
        draw.ellipse((x1, y1, x2, y2), fill=255)
    elif shape_type == "rectangle":
        x1 = rng.randint(width // 8, width // 3)
        y1 = rng.randint(height // 8, height // 3)
        x2 = rng.randint(2 * width // 3, 7 * width // 8)
        y2 = rng.randint(2 * height // 3, 7 * height // 8)
        draw.rounded_rectangle((x1, y1, x2, y2), radius=rng.randint(8, 24), fill=255)
    else:
        for _ in range(rng.randint(2, 4)):
            x1 = rng.randint(width // 10, width // 2)
            y1 = rng.randint(height // 10, height // 2)
            x2 = rng.randint(width // 2, 9 * width // 10)
            y2 = rng.randint(height // 2, 9 * height // 10)
            draw.ellipse((x1, y1, x2, y2), fill=255)

    return mask.filter(ImageFilter.GaussianBlur(radius=rng.randint(6, 16)))


def _perturb_image(image: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-8.0, 8.0)
    translate_x = rng.randint(-12, 12)
    translate_y = rng.randint(-12, 12)
    scale = rng.uniform(0.95, 1.05)

    transformed = image.rotate(angle, resample=Image.BICUBIC)
    transformed = ImageChops.offset(transformed, translate_x, translate_y)

    if scale != 1.0:
        width, height = transformed.size
        resized = transformed.resize((max(1, int(width * scale)), max(1, int(height * scale))), resample=Image.BICUBIC)
        canvas = Image.new("RGB", (width, height))
        paste_x = (width - resized.width) // 2
        paste_y = (height - resized.height) // 2
        canvas.paste(resized, (paste_x, paste_y))
        transformed = canvas

    transformed = ImageEnhance.Color(transformed).enhance(rng.uniform(0.8, 1.2))
    transformed = ImageEnhance.Contrast(transformed).enhance(rng.uniform(0.85, 1.15))
    transformed = ImageEnhance.Brightness(transformed).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.5:
        transformed = transformed.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    return transformed


def _blend_sbi(base: Image.Image, manipulated: Image.Image, mask: Image.Image) -> Image.Image:
    base_np = np.asarray(base).astype(np.float32)
    manip_np = np.asarray(manipulated).astype(np.float32)
    mask_np = np.asarray(mask).astype(np.float32)[:, :, None] / 255.0
    blended = mask_np * manip_np + (1.0 - mask_np) * base_np
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def generate_sbi_samples(
    real_samples: list[dict],
    count: int,
    output_root: Path,
    split_name: str,
    seed: int,
    overwrite: bool = False,
) -> tuple[list[dict], Path]:
    output_dir = output_root / split_name / "fake"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    selected = real_samples[:]
    rng.shuffle(selected)
    if count > len(selected):
        raise ValueError(f"Requested {count} SBI samples, but only {len(selected)} real samples are available.")
    selected = selected[:count]

    generated = []
    for index, sample in enumerate(selected):
        image_path = Path(sample["image_path"])
        image = Image.open(image_path).convert("RGB")
        mask = _random_mask(image.size, rng)
        manipulated = _perturb_image(image, rng)
        blended = _blend_sbi(image, manipulated, mask)

        target_name = f"sbi__{sample.get('dataset_name', 'real')}__{sample.get('video_id', 'video')}__{index:05d}.png"
        target_path = output_dir / target_name
        if overwrite or not target_path.exists():
            blended.save(target_path)

        generated.append(
            {
                "image_path": str(target_path),
                "label": 1,
                "dataset_name": "SBI",
                "split": split_name,
                "video_id": f"sbi_{sample.get('video_id', 'video')}_{index:05d}",
                "frame_index": index,
                "source_video": sample.get("source_video", ""),
                "manipulation_type": "SBI",
            }
        )

    manifest_path = output_root / f"sbi_{split_name}_manifest.jsonl"
    save_manifest(generated, manifest_path)
    return generated, manifest_path
