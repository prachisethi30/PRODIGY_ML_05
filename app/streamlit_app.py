from __future__ import annotations

import json

import streamlit as st
from PIL import Image, UnidentifiedImageError

from model.calorie_db import format_food_name
from model.config import FOOD_LOG_PATH, MODEL_PATH
from model.predict import FoodCalorieEstimator

# Red & white design tokens
RED_PRIMARY = "#C41E3A"
RED_DARK = "#9B1B30"
RED_LIGHT = "#FFF0F0"
RED_BORDER = "#F5C6CB"


def _inject_theme() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

            .stApp {{
                background: linear-gradient(180deg, #FFFFFF 0%, #FFF8F8 100%);
                font-family: 'DM Sans', sans-serif;
            }}

            .block-container {{
                padding-top: 1.5rem;
                max-width: 1100px;
            }}

            /* Hero header */
            .app-hero {{
                background: linear-gradient(135deg, {RED_DARK} 0%, {RED_PRIMARY} 55%, #E85D5D 100%);
                border-radius: 16px;
                padding: 2rem 2.25rem;
                margin-bottom: 1.75rem;
                box-shadow: 0 8px 32px rgba(196, 30, 58, 0.22);
                color: #FFFFFF;
            }}
            .app-hero h1 {{
                color: #FFFFFF !important;
                font-size: 2rem;
                font-weight: 700;
                margin: 0 0 0.4rem 0;
                letter-spacing: -0.02em;
            }}
            .app-hero p {{
                color: rgba(255, 255, 255, 0.92);
                margin: 0;
                font-size: 1.05rem;
            }}

            /* Section panels */
            .panel {{
                background: #FFFFFF;
                border: 1px solid {RED_BORDER};
                border-radius: 14px;
                padding: 1.25rem 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 12px rgba(196, 30, 58, 0.06);
            }}
            .panel-title {{
                color: {RED_DARK};
                font-weight: 700;
                font-size: 1.1rem;
                margin: 0 0 1rem 0;
                padding-bottom: 0.6rem;
                border-bottom: 2px solid {RED_LIGHT};
            }}

            /* Prediction card */
            .result-card {{
                background: {RED_LIGHT};
                border-left: 4px solid {RED_PRIMARY};
                border-radius: 10px;
                padding: 1rem 1.25rem;
                margin: 0.5rem 0;
            }}
            .result-name {{
                color: {RED_DARK};
                font-size: 1.15rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }}
            .result-confidence {{
                color: #666;
                font-size: 0.9rem;
                margin-bottom: 0.75rem;
            }}

            /* Log entries */
            .log-item {{
                background: #FFFFFF;
                border: 1px solid {RED_BORDER};
                border-radius: 10px;
                padding: 0.75rem 1rem;
                margin-bottom: 0.5rem;
                transition: box-shadow 0.2s;
            }}
            .log-item:hover {{
                box-shadow: 0 4px 14px rgba(196, 30, 58, 0.1);
            }}
            .log-food {{
                color: {RED_DARK};
                font-weight: 600;
            }}
            .log-meta {{
                color: #555;
                font-size: 0.88rem;
                margin-top: 0.15rem;
            }}

            /* Nutrition stat grid */
            .nutrition-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.65rem;
                margin-top: 0.65rem;
            }}
            @media (min-width: 640px) {{
                .nutrition-grid {{
                    grid-template-columns: repeat(4, minmax(0, 1fr));
                }}
            }}
            .nutrition-stat {{
                background: #FFFFFF;
                border: 1px solid {RED_BORDER};
                border-radius: 10px;
                padding: 0.7rem 0.85rem;
                box-shadow: 0 1px 6px rgba(196, 30, 58, 0.05);
                min-width: 0;
            }}
            .nutrition-label {{
                color: {RED_DARK};
                font-size: 0.82rem;
                font-weight: 600;
                margin-bottom: 0.35rem;
                line-height: 1.2;
            }}
            .nutrition-value {{
                color: {RED_PRIMARY};
                font-size: 1.35rem;
                font-weight: 700;
                line-height: 1.2;
                word-break: break-word;
            }}
            .nutrition-unit {{
                font-size: 0.78rem;
                font-weight: 600;
                color: #888;
            }}

            /* Buttons */
            .stButton > button[kind="primary"] {{
                background: linear-gradient(135deg, {RED_DARK}, {RED_PRIMARY});
                color: white;
                border: none;
                border-radius: 10px;
                font-weight: 600;
                padding: 0.55rem 1.5rem;
                box-shadow: 0 4px 14px rgba(196, 30, 58, 0.3);
                transition: transform 0.15s, box-shadow 0.15s;
            }}
            .stButton > button[kind="primary"]:hover {{
                background: linear-gradient(135deg, {RED_PRIMARY}, #E85D5D);
                box-shadow: 0 6px 20px rgba(196, 30, 58, 0.35);
                transform: translateY(-1px);
            }}
            .stButton > button[kind="secondary"] {{
                border: 1.5px solid {RED_PRIMARY};
                color: {RED_PRIMARY};
                border-radius: 10px;
                font-weight: 600;
            }}

            /* File uploader */
            [data-testid="stFileUploader"] {{
                background: {RED_LIGHT};
                border: 2px dashed {RED_PRIMARY};
                border-radius: 12px;
                padding: 0.5rem;
            }}

            /* Slider */
            .stSlider [data-baseweb="slider"] div {{
                color: {RED_PRIMARY};
            }}

            /* Expander */
            .streamlit-expanderHeader {{
                color: {RED_DARK};
                font-weight: 600;
            }}

            /* Alerts */
            .stAlert {{
                border-radius: 10px;
            }}

            /* Hide default streamlit header/footer clutter */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}

            /* Subheaders */
            h3 {{
                color: {RED_DARK} !important;
                font-weight: 700 !important;
            }}

            /* Caption footer */
            .app-footer {{
                color: #888;
                font-size: 0.8rem;
                text-align: center;
                margin-top: 1.5rem;
                padding-top: 1rem;
                border-top: 1px solid {RED_BORDER};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_food_log() -> list[dict]:
    if not FOOD_LOG_PATH.exists():
        return []
    try:
        with FOOD_LOG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_food_log(entries: list[dict]) -> None:
    FOOD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FOOD_LOG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


def _init_session_state() -> None:
    if "food_log" not in st.session_state:
        st.session_state.food_log = _load_food_log()


@st.cache_resource
def _get_estimator() -> FoodCalorieEstimator:
    return FoodCalorieEstimator()


def _render_nutrition_grid(
    calories: float,
    protein: float,
    carbs: float,
    fat: float,
) -> None:
    st.markdown(
        f"""
        <div class="nutrition-grid">
            <div class="nutrition-stat">
                <div class="nutrition-label">Calories</div>
                <div class="nutrition-value">{calories:.0f}
                    <span class="nutrition-unit">kcal</span>
                </div>
            </div>
            <div class="nutrition-stat">
                <div class="nutrition-label">Protein</div>
                <div class="nutrition-value">{protein:.1f}
                    <span class="nutrition-unit">g</span>
                </div>
            </div>
            <div class="nutrition-stat">
                <div class="nutrition-label">Carbs</div>
                <div class="nutrition-value">{carbs:.1f}
                    <span class="nutrition-unit">g</span>
                </div>
            </div>
            <div class="nutrition-stat">
                <div class="nutrition-label">Fat</div>
                <div class="nutrition-value">{fat:.1f}
                    <span class="nutrition-unit">g</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_prediction_card(prediction, serving_grams: float) -> None:
    nutrition = prediction.nutrition_per_100g.scaled(serving_grams)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-name">{prediction.display_name}</div>
            <div class="result-confidence">{prediction.confidence_pct:.1f}% confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_nutrition_grid(
        nutrition.calories,
        nutrition.protein_g,
        nutrition.carbs_g,
        nutrition.fat_g,
    )


def _render_log_item(item: dict) -> None:
    st.markdown(
        f"""
        <div class="log-item">
            <div class="log-food">{item['food']}</div>
            <div class="log-meta">
                {item['grams']:.0f} g &nbsp;·&nbsp;
                {item['calories']:.0f} kcal &nbsp;·&nbsp;
                {item['confidence']:.1f}% confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _preview_image(image: Image.Image, max_width: int = 360) -> Image.Image:
    if image.width <= max_width:
        return image
    ratio = max_width / image.width
    return image.resize((max_width, int(image.height * ratio)), Image.Resampling.LANCZOS)


def _load_uploaded_image(uploaded) -> Image.Image | None:
    name = uploaded.name or ""
    if name.startswith("._") or name.startswith("."):
        st.error(
            f"**{name}** is a macOS metadata file, not a photo. "
            "Choose the actual image file (without the `._` prefix), "
            "not files from a `__MACOSX` folder."
        )
        return None
    if uploaded.size < 1024:
        st.error("That file is too small to be a valid image. Upload a real food photo.")
        return None

    try:
        uploaded.seek(0)
        image = Image.open(uploaded)
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Could not read that image ({exc}). Use JPG or PNG from your camera or gallery.")
        return None


def _append_to_log(prediction, serving_grams: float) -> None:
    nutrition = prediction.nutrition_per_100g.scaled(serving_grams)
    st.session_state.food_log.append(
        {
            "food": prediction.display_name,
            "grams": serving_grams,
            "calories": nutrition.calories,
            "protein": nutrition.protein_g,
            "carbs": nutrition.carbs_g,
            "fat": nutrition.fat_g,
            "confidence": prediction.confidence_pct,
        }
    )
    _save_food_log(st.session_state.food_log)


def main() -> None:
    st.set_page_config(
        page_title="Food Calorie Estimator",
        page_icon="🍽️",
        layout="wide",
    )
    _inject_theme()
    _init_session_state()

    st.markdown(
        """
        <div class="app-hero">
            <h1>🍽️ Food Calorie Estimator</h1>
            <p>Upload a food photo to identify your meal and estimate calories &amp; macros.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    estimator = _get_estimator()
    if not estimator.is_ready:
        st.warning(
            "No trained model found yet. Download the dataset and train the model first:\n\n"
            "```bash\n"
            "python main.py download\n"
            "python main.py train --subset 0.1 --epochs 5   # quick start\n"
            "python main.py train                           # full training\n"
            "```"
        )
        st.stop()

    class_names = estimator.class_names
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="panel"><div class="panel-title">📷 Analyze Food</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload a food image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Use a real photo (JPG/PNG). Avoid macOS `._` files or `__MACOSX` folders.",
        )
        serving_grams = st.slider(
            "Estimated portion size (grams)",
            min_value=25,
            max_value=600,
            value=150,
            step=25,
        )

        manual_override = st.checkbox(
            "Choose food manually (skip model prediction)",
            help="Useful when confidence is low or the dish is not recognized.",
        )

        selected_class = None
        if manual_override:
            selected_class = st.selectbox(
                "Food category",
                class_names,
                format_func=format_food_name,
            )

        if uploaded is not None:
            image = _load_uploaded_image(uploaded)
            if image is None:
                st.markdown("</div>", unsafe_allow_html=True)
                st.stop()

            st.image(_preview_image(image), caption="Uploaded image", width=360)

            if manual_override and selected_class:
                prediction = estimator.predict_for_class(selected_class, serving_grams=serving_grams)
                st.subheader("Manual selection")
                with st.container(border=True):
                    _render_prediction_card(prediction, serving_grams)

                if st.button("Add selection to daily log", type="primary"):
                    _append_to_log(prediction, serving_grams)
                    st.success(f"Added {prediction.display_name} to your log.")
            else:
                try:
                    with st.spinner("Analyzing food..."):
                        predictions = estimator.predict(
                            image, serving_grams=serving_grams, top_k=3
                        )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.stop()

                top = predictions[0]
                top_confidence = top.confidence_pct

                st.subheader("Result")
                with st.container(border=True):
                    _render_prediction_card(top, serving_grams)

                if top_confidence < 50:
                    st.warning(
                        f"Low confidence ({top_confidence:.0f}%). The model may be wrong. "
                        "Use **Choose food manually** above, or retrain for better accuracy:\n\n"
                        "`python main.py train --epochs 15`"
                    )
                elif top_confidence < 70:
                    st.info(
                        "Moderate confidence. For better accuracy, retrain on the full dataset "
                        "with `python main.py train --epochs 15` (quick training is ~40% accurate)."
                    )

                alternatives = predictions[1:]
                if alternatives:
                    with st.expander("Other possible matches"):
                        for prediction in alternatives:
                            _render_prediction_card(prediction, serving_grams)

                if st.button("Add to daily log", type="primary"):
                    _append_to_log(top, serving_grams)
                    st.success(f"Added {top.display_name} to your log.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">📋 Daily Intake Log</div>', unsafe_allow_html=True)

        if not st.session_state.food_log:
            st.info("Logged meals will appear here.")
        else:
            total_calories = sum(item["calories"] for item in st.session_state.food_log)
            total_protein = sum(item["protein"] for item in st.session_state.food_log)
            total_carbs = sum(item["carbs"] for item in st.session_state.food_log)
            total_fat = sum(item["fat"] for item in st.session_state.food_log)

            _render_nutrition_grid(
                total_calories,
                total_protein,
                total_carbs,
                total_fat,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            for item in reversed(st.session_state.food_log):
                _render_log_item(item)

            if st.button("Clear log"):
                st.session_state.food_log = []
                _save_food_log([])
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="panel"><div class="panel-title">🥗 Supported Foods</div>',
            unsafe_allow_html=True,
        )
        with st.expander("View all 101 Food-101 categories"):
            st.write(", ".join(format_food_name(name) for name in class_names))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            f'<div class="app-footer">Model: {MODEL_PATH.name} &nbsp;|&nbsp; Log: {FOOD_LOG_PATH.name}</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
