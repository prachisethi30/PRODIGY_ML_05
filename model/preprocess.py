"""Shared image preprocessing for training and inference."""

from __future__ import annotations

import numpy as np
from PIL import Image

from model.config import IMG_SIZE


def resize_with_pad_pil(image: Image.Image, target_size: tuple[int, int] = IMG_SIZE) -> Image.Image:
    """Resize while preserving aspect ratio, padding to a square."""
    image = image.convert("RGB")
    target_w, target_h = target_size
    ratio = min(target_w / image.width, target_h / image.height)
    new_size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    offset = ((target_w - new_size[0]) // 2, (target_h - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def preprocess_pil_image(image: Image.Image, target_size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Return a single-item batch ready for MobileNetV2 inference."""
    import tensorflow as tf

    padded = resize_with_pad_pil(image, target_size)
    array = np.asarray(padded, dtype=np.float32)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)
