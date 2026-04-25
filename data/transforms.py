"""Image transform builders for training and evaluation."""

from __future__ import annotations

import random
import math

import numpy as np
import torch
from PIL import Image, ImageEnhance

from utils.config import DatasetSpec


class Compose:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, image):
        for transform in self.transforms_list:
            image = transform(image)
        return image


class Resize:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize(self.size, Image.Resampling.BILINEAR)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image


class ColorJitter:
    def __init__(self, brightness: float, contrast: float, saturation: float, hue: float) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageEnhance.Brightness(image).enhance(1.0 + random.uniform(-self.brightness, self.brightness))
        image = ImageEnhance.Contrast(image).enhance(1.0 + random.uniform(-self.contrast, self.contrast))
        image = ImageEnhance.Color(image).enhance(1.0 + random.uniform(-self.saturation, self.saturation))
        if self.hue > 0:
            hsv = np.array(image.convert("HSV"), dtype=np.uint8)
            hue_shift = int(255 * random.uniform(-self.hue, self.hue))
            hsv[..., 0] = (hsv[..., 0].astype(int) + hue_shift) % 255
            image = Image.fromarray(hsv, mode="HSV").convert("RGB")
        return image


class ToTensor:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class Normalize:
    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


def _build_compose(transforms_list):
    try:
        from torchvision import transforms
    except ImportError:
        return Compose(transforms_list)
    return transforms.Compose(transforms_list)


def apply_frequency_debias(image: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return image

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width, _ = array.shape

    spectrum = np.fft.fft2(array, axes=(0, 1))
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    grid_h = max(4, min(16, height // 16 or 4))
    grid_w = max(4, min(16, width // 16 or 4))
    lowres_noise = np.random.uniform(-1.0, 1.0, size=(grid_h, grid_w, 1)).astype(np.float32)

    repeat_h = math.ceil(height / grid_h)
    repeat_w = math.ceil(width / grid_w)
    smooth_noise = np.kron(lowres_noise, np.ones((repeat_h, repeat_w, 1), dtype=np.float32))
    smooth_noise = smooth_noise[:height, :width, :]

    perturbed_amplitude = amplitude * np.clip(1.0 + (strength * smooth_noise), 0.0, None)
    reconstructed = np.fft.ifft2(perturbed_amplitude * np.exp(1j * phase), axes=(0, 1)).real
    reconstructed = np.clip(reconstructed, 0.0, 1.0)
    reconstructed_uint8 = (reconstructed * 255.0).astype(np.uint8)
    return Image.fromarray(reconstructed_uint8, mode="RGB")


def build_train_transforms(spec: DatasetSpec):
    return _build_compose(
        [
            Resize((spec.image_size, spec.image_size)),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_eval_transforms(spec: DatasetSpec):
    return _build_compose(
        [
            Resize((spec.image_size, spec.image_size)),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
