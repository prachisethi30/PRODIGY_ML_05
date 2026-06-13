"""Nutritional estimates per 100 g for Food-101 classes (USDA-style averages)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NutritionInfo:
    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float
    default_serving_g: float = 100.0

    def scaled(self, grams: float) -> "NutritionInfo":
        factor = grams / 100.0
        return NutritionInfo(
            calories=round(self.calories * factor, 1),
            protein_g=round(self.protein_g * factor, 1),
            fat_g=round(self.fat_g * factor, 1),
            carbs_g=round(self.carbs_g * factor, 1),
            default_serving_g=grams,
        )


# Values are approximate per 100 g edible portion.
FOOD_NUTRITION: dict[str, NutritionInfo] = {
    "apple_pie": NutritionInfo(296, 2.1, 14.0, 41.0, 125),
    "baby_back_ribs": NutritionInfo(293, 22.0, 22.0, 0.0, 150),
    "baklava": NutritionInfo(428, 5.0, 24.0, 48.0, 80),
    "beef_carpaccio": NutritionInfo(120, 20.0, 4.0, 0.0, 100),
    "beef_tartare": NutritionInfo(180, 18.0, 12.0, 1.0, 100),
    "beet_salad": NutritionInfo(65, 2.0, 3.5, 7.0, 150),
    "beignets": NutritionInfo(417, 6.0, 22.0, 48.0, 60),
    "bibimbap": NutritionInfo(112, 3.0, 2.0, 22.0, 350),
    "bread_pudding": NutritionInfo(220, 5.0, 8.0, 32.0, 120),
    "breakfast_burrito": NutritionInfo(206, 9.0, 10.0, 20.0, 200),
    "bruschetta": NutritionInfo(150, 4.0, 6.0, 20.0, 80),
    "caesar_salad": NutritionInfo(127, 7.0, 9.0, 5.0, 200),
    "cannoli": NutritionInfo(307, 6.0, 16.0, 35.0, 80),
    "caprese_salad": NutritionInfo(131, 8.0, 10.0, 3.0, 150),
    "carrot_cake": NutritionInfo(415, 4.0, 20.0, 55.0, 100),
    "ceviche": NutritionInfo(69, 12.0, 1.0, 3.0, 150),
    "cheese_plate": NutritionInfo(350, 20.0, 28.0, 2.0, 100),
    "cheesecake": NutritionInfo(321, 6.0, 23.0, 24.0, 100),
    "chicken_curry": NutritionInfo(150, 12.0, 8.0, 8.0, 250),
    "chicken_quesadilla": NutritionInfo(234, 14.0, 12.0, 18.0, 180),
    "chicken_wings": NutritionInfo(290, 27.0, 19.0, 0.0, 120),
    "chocolate_cake": NutritionInfo(371, 5.0, 16.0, 53.0, 100),
    "chocolate_mousse": NutritionInfo(225, 4.0, 16.0, 18.0, 100),
    "churros": NutritionInfo(412, 5.0, 22.0, 48.0, 60),
    "clam_chowder": NutritionInfo(81, 4.0, 3.0, 9.0, 250),
    "club_sandwich": NutritionInfo(220, 12.0, 10.0, 22.0, 200),
    "crab_cakes": NutritionInfo(199, 15.0, 12.0, 8.0, 100),
    "creme_brulee": NutritionInfo(230, 4.0, 17.0, 17.0, 100),
    "croque_madame": NutritionInfo(280, 14.0, 16.0, 20.0, 180),
    "cup_cakes": NutritionInfo(305, 3.6, 12.0, 46.0, 60),
    "deviled_eggs": NutritionInfo(155, 6.0, 13.0, 1.0, 50),
    "donuts": NutritionInfo(452, 5.0, 25.0, 51.0, 60),
    "dumplings": NutritionInfo(203, 7.0, 8.0, 26.0, 120),
    "edamame": NutritionInfo(122, 11.0, 5.0, 10.0, 100),
    "eggs_benedict": NutritionInfo(287, 13.0, 22.0, 8.0, 200),
    "escargots": NutritionInfo(90, 16.0, 1.5, 2.0, 100),
    "falafel": NutritionInfo(333, 13.0, 18.0, 32.0, 100),
    "filet_mignon": NutritionInfo(227, 26.0, 13.0, 0.0, 150),
    "fish_and_chips": NutritionInfo(232, 12.0, 12.0, 22.0, 300),
    "foie_gras": NutritionInfo(462, 7.0, 43.0, 2.0, 50),
    "french_fries": NutritionInfo(312, 3.4, 15.0, 41.0, 100),
    "french_onion_soup": NutritionInfo(56, 3.0, 2.5, 6.0, 250),
    "french_toast": NutritionInfo(229, 8.0, 10.0, 28.0, 120),
    "fried_calamari": NutritionInfo(211, 15.0, 11.0, 14.0, 120),
    "fried_rice": NutritionInfo(163, 4.0, 5.0, 26.0, 200),
    "frozen_yogurt": NutritionInfo(127, 3.0, 4.0, 22.0, 100),
    "garlic_bread": NutritionInfo(350, 7.0, 17.0, 44.0, 50),
    "gnocchi": NutritionInfo(133, 3.0, 1.0, 28.0, 200),
    "greek_salad": NutritionInfo(106, 4.0, 8.0, 6.0, 200),
    "grilled_cheese_sandwich": NutritionInfo(291, 12.0, 16.0, 28.0, 150),
    "grilled_salmon": NutritionInfo(206, 22.0, 12.0, 0.0, 150),
    "guacamole": NutritionInfo(160, 2.0, 15.0, 9.0, 100),
    "gyoza": NutritionInfo(210, 8.0, 9.0, 24.0, 100),
    "hamburger": NutritionInfo(295, 17.0, 14.0, 24.0, 200),
    "hot_and_sour_soup": NutritionInfo(39, 3.0, 1.5, 4.0, 250),
    "hot_dog": NutritionInfo(290, 10.0, 26.0, 2.0, 100),
    "huevos_rancheros": NutritionInfo(189, 10.0, 11.0, 14.0, 250),
    "hummus": NutritionInfo(166, 8.0, 10.0, 14.0, 100),
    "ice_cream": NutritionInfo(207, 3.5, 11.0, 24.0, 100),
    "lasagna": NutritionInfo(135, 8.0, 5.0, 15.0, 250),
    "lobster_bisque": NutritionInfo(120, 6.0, 8.0, 6.0, 250),
    "lobster_roll_sandwich": NutritionInfo(289, 14.0, 15.0, 24.0, 200),
    "macaroni_and_cheese": NutritionInfo(164, 7.0, 7.0, 18.0, 200),
    "macarons": NutritionInfo(393, 5.0, 18.0, 52.0, 30),
    "miso_soup": NutritionInfo(40, 3.0, 1.0, 5.0, 250),
    "mussels": NutritionInfo(86, 12.0, 2.0, 4.0, 150),
    "nachos": NutritionInfo(346, 9.0, 22.0, 28.0, 150),
    "omelette": NutritionInfo(154, 11.0, 12.0, 1.0, 150),
    "onion_rings": NutritionInfo(411, 4.0, 23.0, 46.0, 100),
    "oysters": NutritionInfo(68, 7.0, 2.0, 4.0, 100),
    "pad_thai": NutritionInfo(180, 8.0, 7.0, 22.0, 300),
    "paella": NutritionInfo(156, 9.0, 5.0, 18.0, 300),
    "pancakes": NutritionInfo(227, 6.0, 10.0, 28.0, 120),
    "panna_cotta": NutritionInfo(274, 3.0, 22.0, 17.0, 100),
    "peking_duck": NutritionInfo(337, 19.0, 28.0, 0.0, 150),
    "pho": NutritionInfo(95, 6.0, 2.0, 12.0, 400),
    "pizza": NutritionInfo(266, 11.0, 10.0, 33.0, 120),
    "pork_chop": NutritionInfo(231, 25.0, 14.0, 0.0, 150),
    "poutine": NutritionInfo(280, 8.0, 16.0, 26.0, 300),
    "prime_rib": NutritionInfo(291, 24.0, 21.0, 0.0, 150),
    "pulled_pork_sandwich": NutritionInfo(265, 18.0, 12.0, 22.0, 200),
    "ramen": NutritionInfo(188, 8.0, 7.0, 24.0, 400),
    "ravioli": NutritionInfo(175, 7.0, 6.0, 24.0, 200),
    "red_velvet_cake": NutritionInfo(367, 4.0, 15.0, 54.0, 100),
    "risotto": NutritionInfo(130, 3.0, 4.0, 22.0, 250),
    "samosa": NutritionInfo(262, 5.0, 14.0, 30.0, 80),
    "sashimi": NutritionInfo(127, 20.0, 4.0, 0.0, 100),
    "scallops": NutritionInfo(88, 17.0, 1.0, 3.0, 100),
    "seaweed_salad": NutritionInfo(45, 1.0, 0.5, 9.0, 100),
    "shrimp_and_grits": NutritionInfo(151, 10.0, 6.0, 15.0, 300),
    "spaghetti_bolognese": NutritionInfo(132, 7.0, 4.0, 17.0, 300),
    "spaghetti_carbonara": NutritionInfo(180, 8.0, 9.0, 18.0, 300),
    "spring_rolls": NutritionInfo(154, 3.0, 5.0, 24.0, 80),
    "steak": NutritionInfo(271, 26.0, 18.0, 0.0, 200),
    "strawberry_shortcake": NutritionInfo(346, 4.0, 16.0, 47.0, 100),
    "sushi": NutritionInfo(143, 6.0, 2.0, 28.0, 150),
    "tacos": NutritionInfo(226, 10.0, 12.0, 20.0, 150),
    "takoyaki": NutritionInfo(175, 6.0, 8.0, 20.0, 100),
    "tiramisu": NutritionInfo(240, 5.0, 14.0, 24.0, 100),
    "tuna_tartare": NutritionInfo(144, 24.0, 4.0, 0.0, 100),
    "waffles": NutritionInfo(291, 6.0, 14.0, 35.0, 100),
}


def format_food_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def get_nutrition(class_name: str) -> NutritionInfo:
    if class_name not in FOOD_NUTRITION:
        return NutritionInfo(200.0, 5.0, 8.0, 25.0, 100.0)
    return FOOD_NUTRITION[class_name]
