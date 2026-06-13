from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from model.calorie_db import NutritionInfo, format_food_name, get_nutrition
from model.config import CLASS_NAMES_PATH, MODEL_PATH, TOP_K
from model.preprocess import preprocess_pil_image


@dataclass
class FoodPrediction:
    class_name: str
    display_name: str
    confidence: float
    nutrition_per_100g: NutritionInfo
    estimated_nutrition: NutritionInfo

    @property
    def confidence_pct(self) -> float:
        return round(self.confidence * 100, 2)


class FoodCalorieEstimator:
    def __init__(self, model_path: Path | None = None, class_names_path: Path | None = None):
        self.model_path = model_path or MODEL_PATH
        self.class_names_path = class_names_path or CLASS_NAMES_PATH
        self._model = None
        self._class_names: list[str] | None = None

    @property
    def is_ready(self) -> bool:
        return self.model_path.exists() and self.class_names_path.exists()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.is_ready:
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Run: python main.py train"
            )

        import tensorflow as tf

        self._model = tf.keras.models.load_model(self.model_path)
        with self.class_names_path.open("r", encoding="utf-8") as handle:
            self._class_names = json.load(handle)

    @property
    def class_names(self) -> list[str]:
        self._load()
        assert self._class_names is not None
        return self._class_names

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        return preprocess_pil_image(image)

    def predict_for_class(
        self,
        class_name: str,
        serving_grams: float | None = None,
        confidence: float = 1.0,
    ) -> FoodPrediction:
        """Build a prediction for a manually selected food class."""
        self._load()
        if class_name not in self.class_names:
            raise ValueError(f"Unknown food class: {class_name}")

        nutrition = get_nutrition(class_name)
        grams = serving_grams or nutrition.default_serving_g
        return FoodPrediction(
            class_name=class_name,
            display_name=format_food_name(class_name),
            confidence=confidence,
            nutrition_per_100g=nutrition,
            estimated_nutrition=nutrition.scaled(grams),
        )

    def predict(
        self,
        image: Image.Image | str | Path | bytes,
        serving_grams: float | None = None,
        top_k: int = TOP_K,
    ) -> list[FoodPrediction]:
        self._load()
        assert self._model is not None

        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        batch = self._preprocess(image)
        probabilities = self._model.predict(batch, verbose=0)[0]
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        predictions: list[FoodPrediction] = []
        for index in top_indices:
            class_name = self.class_names[int(index)]
            nutrition = get_nutrition(class_name)
            grams = serving_grams or nutrition.default_serving_g
            predictions.append(
                FoodPrediction(
                    class_name=class_name,
                    display_name=format_food_name(class_name),
                    confidence=float(probabilities[index]),
                    nutrition_per_100g=nutrition,
                    estimated_nutrition=nutrition.scaled(grams),
                )
            )
        return predictions
