from __future__ import annotations

import json
import random
from pathlib import Path

from model.config import BATCH_SIZE, DATA_DIR, IMAGES_DIR, IMG_SIZE, META_DIR


def dataset_is_ready() -> bool:
    return (
        (META_DIR / "classes.txt").exists()
        and (META_DIR / "train.txt").exists()
        and (META_DIR / "test.txt").exists()
        and IMAGES_DIR.exists()
    )


def load_class_names() -> list[str]:
    classes_path = META_DIR / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(
            f"Food-101 classes file not found at {classes_path}. "
            "Run: python main.py download"
        )
    with classes_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _image_path(class_name: str, image_id: str) -> str:
    """Return a TensorFlow-safe forward-slash path."""
    return (IMAGES_DIR / class_name / f"{image_id}.jpg").as_posix()


def _filter_existing_paths(
    paths: list[str], labels: list[int]
) -> tuple[list[str], list[int]]:
    filtered_paths: list[str] = []
    filtered_labels: list[int] = []
    missing = 0
    for path, label in zip(paths, labels):
        if Path(path).exists():
            filtered_paths.append(path)
            filtered_labels.append(label)
        else:
            missing += 1
    if missing:
        print(f"Warning: skipped {missing} missing image(s) under {IMAGES_DIR}")
    return filtered_paths, filtered_labels


def _parse_split_file(split_path: Path) -> tuple[list[str], list[int]]:
    image_paths: list[str] = []
    labels: list[int] = []
    class_to_index = {name: index for index, name in enumerate(load_class_names())}

    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry:
                continue
            class_name, image_id = entry.split("/")
            image_paths.append(_image_path(class_name, image_id))
            labels.append(class_to_index[class_name])

    return image_paths, labels


def _stratified_subset(
    paths: list[str],
    labels: list[int],
    subset: float,
    seed: int,
) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = {}
    for path, label in zip(paths, labels):
        by_label.setdefault(label, []).append(path)

    subset_paths: list[str] = []
    subset_labels: list[int] = []
    for label, label_paths in by_label.items():
        rng.shuffle(label_paths)
        count = max(1, int(len(label_paths) * subset))
        subset_paths.extend(label_paths[:count])
        subset_labels.extend([label] * count)

    paired = list(zip(subset_paths, subset_labels))
    rng.shuffle(paired)
    subset_paths, subset_labels = zip(*paired)
    return list(subset_paths), list(subset_labels)


def _stratified_train_val_split(
    paths: list[str],
    labels: list[int],
    validation_split: float,
    seed: int,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Hold out a stratified fraction of each class for validation."""
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = {}
    for path, label in zip(paths, labels):
        by_label.setdefault(label, []).append(path)

    train_paths: list[str] = []
    train_labels: list[int] = []
    val_paths: list[str] = []
    val_labels: list[int] = []

    for label, label_paths in by_label.items():
        rng.shuffle(label_paths)
        if len(label_paths) == 1:
            train_paths.extend(label_paths)
            train_labels.append(label)
            continue

        val_count = max(1, int(len(label_paths) * validation_split))
        val_paths.extend(label_paths[:val_count])
        val_labels.extend([label] * val_count)
        train_paths.extend(label_paths[val_count:])
        train_labels.extend([label] * (len(label_paths) - val_count))

    train_pairs = list(zip(train_paths, train_labels))
    rng.shuffle(train_pairs)
    train_paths, train_labels = map(list, zip(*train_pairs))

    val_pairs = list(zip(val_paths, val_labels))
    rng.shuffle(val_pairs)
    val_paths, val_labels = map(list, zip(*val_pairs))

    return train_paths, train_labels, val_paths, val_labels


def build_datasets(
    img_size: tuple[int, int] = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    validation_split: float = 0.1,
    subset: float | None = None,
    seed: int = 42,
):
    import tensorflow as tf

    if not dataset_is_ready():
        raise FileNotFoundError(
            f"Food-101 dataset not found under {DATA_DIR}. "
            "Run: python main.py download"
        )

    train_paths, train_labels = _filter_existing_paths(
        *_parse_split_file(META_DIR / "train.txt")
    )
    test_paths, test_labels = _filter_existing_paths(
        *_parse_split_file(META_DIR / "test.txt")
    )

    if subset is not None and 0 < subset < 1:
        train_paths, train_labels = _stratified_subset(
            train_paths, train_labels, subset, seed
        )
        test_paths, test_labels = _stratified_subset(
            test_paths, test_labels, subset, seed + 1
        )

    class_names = load_class_names()
    num_classes = len(class_names)

    if validation_split > 0:
        train_paths, train_labels, val_paths, val_labels = _stratified_train_val_split(
            train_paths, train_labels, validation_split, seed
        )
    else:
        val_paths, val_labels = [], []

    def load_image(path: tf.Tensor, label: tf.Tensor, training: bool):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_with_pad(image, img_size[0], img_size[1])
        if training:
            # Horizontal flip
            image = tf.image.random_flip_left_right(image)
            # Vertical flip (occasionally)
            if tf.random.uniform([]) > 0.7:
                image = tf.image.flip_up_down(image)
            # Brightness augmentation
            image = tf.image.random_brightness(image, max_delta=32)
            # Contrast augmentation
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            # Saturation augmentation
            image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
            # Hue augmentation
            image = tf.image.random_hue(image, max_delta=0.1)
            # Rotation (via random crop and resize for efficiency)
            if tf.random.uniform([]) > 0.6:
                # Random rotation simulation via crop
                crop_size = int(img_size[0] * 0.85)
                image = tf.image.random_crop(image, [crop_size, crop_size, 3])
                image = tf.image.resize(image, img_size)
            image = tf.clip_by_value(image, 0.0, 255.0)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    def make_dataset(paths: list[str], labels: list[int], training: bool):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda path, label: load_image(path, label, training),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_paths, train_labels, training=True)
    test_ds = make_dataset(test_paths, test_labels, training=False)
    val_ds = (
        make_dataset(val_paths, val_labels, training=False)
        if val_paths
        else None
    )

    return train_ds, val_ds, test_ds, class_names, num_classes


def save_class_names(class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)
