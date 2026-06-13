# Food Calorie Estimator

Recognize food items from photos using the [Food-101](https://www.kaggle.com/dansbecker/food-101) dataset and estimate calories plus macronutrients to help track dietary intake.

## Features

- **101-class food recognition** with a MobileNetV2 transfer-learning model
- **Calorie and macro estimates** mapped from USDA-style nutrition averages
- **Portion-size scaling** so users can adjust grams and see updated totals
- **Streamlit dashboard** for image upload, predictions, and a simple daily food log
- **CLI workflow** for download, training, inference, and launching the app

## Project structure

```
FoodCalorieEstimator/
├── app/streamlit_app.py      # Web UI
├── model/
│   ├── train.py              # Training pipeline
│   ├── predict.py            # Inference API
│   ├── calorie_db.py         # Nutrition mapping for 101 foods
│   ├── data_loader.py        # Food-101 data pipeline
│   └── checkpoints/          # Saved model (created after training)
├── scripts/download_data.py  # Kaggle download helper
├── data/food-101/            # Dataset (created after download)
├── main.py                   # CLI entry point
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Kaggle credentials

1. Create a Kaggle API token from [kaggle.com/settings](https://www.kaggle.com/settings)
2. Save `kaggle.json` to `C:\Users\<you>\.kaggle\` on Windows

## Usage

### 1. Download Food-101

```bash
python main.py download
```

### 2. Train the model

Quick experiment on 10% of the data:

```bash
python main.py train --subset 0.1 --epochs 5
```

Full training (recommended for best accuracy):

```bash
python main.py train --epochs 15
```

The trained model is saved to `model/checkpoints/food101_mobilenetv2.keras`.

### 3. Predict from the command line

```bash
python main.py predict path\to\food.jpg --grams 200
```

### 4. Launch the web app

```bash
python main.py app
```

Upload a photo, review top predictions with confidence scores, adjust portion size, and add items to a daily intake log.

## How it works

1. **Classification**: MobileNetV2 (ImageNet pre-trained) is fine-tuned on Food-101 to recognize 101 food categories.
2. **Calorie estimation**: The predicted class is mapped to average nutrition values per 100 g from `model/calorie_db.py`.
3. **Portion scaling**: Estimated calories and macros are scaled by the user-selected serving size in grams.

Food-101 does not include calorie labels, so nutrition values are reference estimates. For medical or clinical use, verify values against a trusted nutrition database.

## Expected performance

With full training and fine-tuning, validation accuracy on Food-101 is typically in the **75–85%** range depending on hardware, epochs, and augmentation. Use the full dataset and 15+ epochs for production-quality results.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Recent improvements

- Aspect-ratio-preserving resize (pad-to-square) for training and inference
- Stratified validation split and stronger augmentation (brightness, contrast)
- Windows-safe forward-slash image paths for TensorFlow
- Early stopping restores best weights; checkpoint is no longer overwritten by final epoch
- Streamlit: cached model loading, persistent food log (`data/food_log.json`), manual food override
- Removed unused dependencies (`opencv-python`, `pandas`, `matplotlib`)

## License

Food-101 dataset terms apply when downloading from Kaggle. See the original [Food-101 paper](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) for citation details.

[Large folders like archive (6)/, data, and checkpoints are ignored to avoid uploading huge files.]
