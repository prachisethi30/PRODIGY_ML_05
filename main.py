#!/usr/bin/env python3
"""Food Calorie Estimator — CLI entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def cmd_download(args: argparse.Namespace) -> None:
    from scripts.download_data import download_dataset

    download_dataset(force=args.force)


def cmd_train(args: argparse.Namespace) -> None:
    from model.train import train

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset=args.subset,
        fine_tune=not args.no_fine_tune,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    from PIL import Image

    from model.predict import FoodCalorieEstimator

    estimator = FoodCalorieEstimator()
    image = Image.open(args.image)
    predictions = estimator.predict(
        image, serving_grams=args.grams, top_k=args.top_k
    )

    label = "Prediction" if args.top_k == 1 else "Predictions"
    print(f"\n{label} for {args.image} ({args.grams:.0f} g portion):\n")
    for rank, prediction in enumerate(predictions, start=1):
        nutrition = prediction.estimated_nutrition
        print(
            f"{rank}. {prediction.display_name} "
            f"({prediction.confidence_pct:.1f}%)\n"
            f"   Calories: {nutrition.calories:.0f} kcal | "
            f"Protein: {nutrition.protein_g:.1f} g | "
            f"Carbs: {nutrition.carbs_g:.1f} g | "
            f"Fat: {nutrition.fat_g:.1f} g"
        )


def cmd_app(_: argparse.Namespace) -> None:
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize food from images and estimate calorie content."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download", help="Download Food-101 from Kaggle"
    )
    download_parser.add_argument(
        "--force", action="store_true", help="Re-download even if data exists"
    )
    download_parser.set_defaults(func=cmd_download)

    train_parser = subparsers.add_parser("train", help="Train the classifier")
    train_parser.add_argument("--epochs", type=int, default=15)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Train on a fraction of data for quick experiments",
    )
    train_parser.add_argument("--no-fine-tune", action="store_true")
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser(
        "predict", help="Run inference on a single image"
    )
    predict_parser.add_argument("image", type=Path, help="Path to a food image")
    predict_parser.add_argument(
        "--grams", type=float, default=150.0, help="Estimated portion size in grams"
    )
    predict_parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of predictions to show (default: 1)",
    )
    predict_parser.set_defaults(func=cmd_predict)

    app_parser = subparsers.add_parser("app", help="Launch the Streamlit web app")
    app_parser.set_defaults(func=cmd_app)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
