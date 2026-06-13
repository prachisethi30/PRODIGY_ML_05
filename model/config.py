import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_data_dir() -> Path:
    env_override = os.environ.get("FOOD101_DATA_DIR")
    if env_override:
        return Path(env_override)

    candidates = [
        PROJECT_ROOT / "data" / "food-101",
        PROJECT_ROOT / "archive (6)" / "food-101" / "food-101",
    ]
    for path in candidates:
        if (path / "meta" / "classes.txt").exists() and (path / "images").exists():
            return path
    return candidates[0]


DATA_DIR = _resolve_data_dir()
IMAGES_DIR = DATA_DIR / "images"
META_DIR = DATA_DIR / "meta"
CHECKPOINT_DIR = PROJECT_ROOT / "model" / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "food101_mobilenetv2.keras"
CLASS_NAMES_PATH = CHECKPOINT_DIR / "class_names.json"
TRAINING_HISTORY_PATH = CHECKPOINT_DIR / "training_history.json"
FOOD_LOG_PATH = PROJECT_ROOT / "data" / "food_log.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 30  # Increased for better convergence
LEARNING_RATE = 1e-4
LEARNING_RATE_INIT = 1e-3  # Initial warmup learning rate
TOP_K = 1

# Training improvements
WARMUP_EPOCHS = 2
LABEL_SMOOTHING = 0.1
AUGMENTATION_ENABLED = True
