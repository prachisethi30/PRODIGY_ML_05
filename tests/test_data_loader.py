from model.data_loader import _stratified_train_val_split


def test_stratified_split_keeps_class_balance():
    paths = [f"img_{i}.jpg" for i in range(100)]
    labels = [i // 10 for i in range(100)]

    train_paths, train_labels, val_paths, val_labels = _stratified_train_val_split(
        paths, labels, validation_split=0.2, seed=42
    )

    assert len(train_paths) + len(val_paths) == 100
    assert len(train_labels) + len(val_labels) == 100
    assert len(val_paths) == 20
    assert len(set(val_labels)) == 10
