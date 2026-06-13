from PIL import Image

from model.preprocess import preprocess_pil_image, resize_with_pad_pil


def test_resize_with_pad_preserves_aspect_ratio():
    wide = Image.new("RGB", (400, 200), color=(255, 0, 0))
    padded = resize_with_pad_pil(wide, (224, 224))
    assert padded.size == (224, 224)


def test_preprocess_returns_batch_of_one():
    image = Image.new("RGB", (300, 300), color=(0, 255, 0))
    batch = preprocess_pil_image(image)
    assert batch.shape == (1, 224, 224, 3)
