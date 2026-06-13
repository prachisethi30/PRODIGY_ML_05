from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLASS_NAMES_PATH,
    DEFAULT_EPOCHS,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    LEARNING_RATE_INIT,
    MODEL_PATH,
    TRAINING_HISTORY_PATH,
    WARMUP_EPOCHS,
)
from model.data_loader import build_datasets, save_class_names


def build_model(num_classes: int, learning_rate: float = LEARNING_RATE):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")],
    )
    return model, base_model


def _build_callbacks(val_ds, total_epochs: int) -> list:
    import tensorflow as tf

    monitor = "val_accuracy" if val_ds is not None else "accuracy"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=6,
            restore_best_weights=True,
            min_delta=0.001,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds is not None else "loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.LearningRateScheduler(
            schedule=lambda epoch, lr: _lr_schedule(epoch, lr, total_epochs),
            verbose=1,
        ),
    ]


def _lr_schedule(epoch: int, lr: float, total_epochs: int) -> float:
    """Cosine annealing with warmup."""
    import math
    if epoch < WARMUP_EPOCHS:
        return LEARNING_RATE_INIT * (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / (total_epochs - WARMUP_EPOCHS)
    return LEARNING_RATE * (1 + math.cos(math.pi * progress)) / 2


def train(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    subset: float | None = None,
    fine_tune: bool = True,
) -> Path:
    import tensorflow as tf

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names, num_classes = build_datasets(
        batch_size=batch_size,
        subset=subset,
    )
    save_class_names(class_names, CLASS_NAMES_PATH)

    model, base_model = build_model(num_classes)
    callbacks = _build_callbacks(val_ds, epochs)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if fine_tune:
        base_model.trainable = True
        # Freeze earlier layers, unfreeze only the last 50 layers for fine-tuning
        for layer in base_model.layers[:-50]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 20),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")],
        )
        fine_tune_epochs = max(8, epochs // 2)
        fine_tune_callbacks = _build_callbacks(val_ds, fine_tune_epochs)
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=fine_tune_callbacks,
            verbose=1,
        )
        for key, values in fine_tune_history.history.items():
            history.history.setdefault(key, []).extend(values)

    if test_ds is not None:
        test_metrics = model.evaluate(test_ds, verbose=0)
        test_loss, test_accuracy, test_top5 = test_metrics
        print(
            f"Test accuracy: {test_accuracy:.4f} | "
            f"Top-5: {test_top5:.4f} | Test loss: {test_loss:.4f}"
        )

    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model.save(MODEL_PATH)

    with TRAINING_HISTORY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(history.history, handle, indent=2)

    print(f"Model saved to {MODEL_PATH}")
    return MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Food-101 classifier")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Use a fraction of the dataset for quick experiments (e.g. 0.05)",
    )
    parser.add_argument("--no-fine-tune", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset=args.subset,
        fine_tune=not args.no_fine_tune,
    )
