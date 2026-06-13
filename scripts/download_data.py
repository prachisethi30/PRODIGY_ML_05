"""Download and extract the Food-101 dataset from Kaggle."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from model.config import DATA_DIR


def _run_kaggle_download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        "dansbecker/food-101",
        "-p",
        str(DATA_DIR),
    ]
    print("Downloading Food-101 from Kaggle...")
    subprocess.run(command, check=True)

    zip_files = list(DATA_DIR.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError("Kaggle download completed but no zip file was found.")

    archive = zip_files[0]
    print(f"Extracting {archive.name}...")
    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    nested_root = DATA_DIR / "food-101"
    if nested_root.exists():
        for item in nested_root.iterdir():
            destination = DATA_DIR / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), str(destination))
        nested_root.rmdir()

    archive.unlink()

    images_dir = DATA_DIR / "images"
    meta_dir = DATA_DIR / "meta"
    if not images_dir.exists() or not meta_dir.exists():
        raise FileNotFoundError(
            "Extraction finished but Food-101 folders were not found. "
            f"Expected images/ and meta/ under {DATA_DIR}"
        )

    print(f"Food-101 is ready at {DATA_DIR}")


def download_dataset(force: bool = False) -> Path:
    if (DATA_DIR / "meta" / "classes.txt").exists() and not force:
        print(f"Dataset already present at {DATA_DIR}")
        return DATA_DIR

    try:
        _run_kaggle_download()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle download failed. Install Kaggle CLI and configure credentials:\n"
            "  pip install kaggle\n"
            "  Place kaggle.json in ~/.kaggle/ (Linux/macOS) or "
            "C:\\Users\\<you>\\.kaggle\\ (Windows)."
        ) from exc

    return DATA_DIR
