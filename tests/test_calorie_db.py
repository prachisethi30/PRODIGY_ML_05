from model.calorie_db import FOOD_NUTRITION, get_nutrition


def test_all_food_nutrition_entries_have_positive_calories():
    for name, info in FOOD_NUTRITION.items():
        assert info.calories > 0, name
        assert info.default_serving_g > 0, name


def test_scaled_nutrition_doubles_at_200g():
    info = get_nutrition("pizza")
    scaled = info.scaled(200)
    assert scaled.calories == round(info.calories * 2, 1)
    assert scaled.protein_g == round(info.protein_g * 2, 1)


def test_unknown_class_returns_fallback():
    info = get_nutrition("unknown_food_xyz")
    assert info.calories == 200.0
