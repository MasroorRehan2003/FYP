import streamlit as st
import pandas as pd
import joblib
import random
import os
from glob import glob
from datetime import datetime
from weather_api import fetch_weather  # custom weather fetch function




# === Load model and dataset ===
model = joblib.load("weather_clothing_recommender.pkl")
clothing_df = pd.read_csv("enhanced_weather_clothing_dataset_no_accessories.csv")
clothing_df.columns = clothing_df.columns.str.strip().str.lower()

# === Utility function to get image ===
def get_image_from_folder(category):
    folder_path = os.path.join(os.getcwd(), category)
    image_files = glob(os.path.join(folder_path, "*.jpg"))
    return random.choice(image_files) if image_files else None

# === Streamlit UI Config ===
st.set_page_config(page_title="WearSmart", page_icon="🧥", layout="centered")
st.title("👚 WearSmart - AI Clothing Recommender")

# === Sidebar Input ===
with st.sidebar:
    st.header("🌦️ Weather Info")
    city = st.text_input("Enter your city", value="Lahore")
    api_key = "4c703f15e3f9220de836884137342d5d"
    if st.button("Get Recommendation"):
        st.session_state.trigger = True

# === Session Initialization ===
if "shuffled_sample" not in st.session_state:
    st.session_state.shuffled_sample = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "filtered_items" not in st.session_state:
    st.session_state.filtered_items = pd.DataFrame()
if "trigger" not in st.session_state:
    st.session_state.trigger = False

# === Trigger Recommendation ===
if st.session_state.trigger:
    st.session_state.trigger = False

    weather = fetch_weather(city, api_key)

    if weather:
        hour = datetime.now().hour
        time_of_day = (
            'morning' if 5 <= hour < 12 else
            'afternoon' if 12 <= hour < 17 else
            'evening' if 17 <= hour < 21 else
            'night'
        )

        month = datetime.now().month
        season = (
            'summer' if 5 <= month <= 8 else
            'fall' if 9 <= month <= 10 else
            'winter' if 11 <= month or month <= 2 else
            'spring'
        )

        input_data = {
            'temperature': weather['temperature'],
            'feels_like': weather['temperature'],
            'humidity': weather['humidity'],
            'wind_speed': weather['wind'],
            'weather_condition': weather['condition'].lower(),
            'time_of_day': time_of_day,
            'season': season,
            'mood': 'good',
            'occasion': 'casual'
        }

        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.session_state.prediction = prediction

        top, bottom, outer = prediction
        fallback_type = "Full match"
        filtered = clothing_df[
            (clothing_df["recommended_top"] == top) &
            (clothing_df["recommended_bottom"] == bottom) &
            (clothing_df["recommended_outer"] == outer)
        ]

        if filtered.empty:
            fallback_type = "Top + Bottom match"
            filtered = clothing_df[
                (clothing_df["recommended_top"] == top) &
                (clothing_df["recommended_bottom"] == bottom)
            ]
        if filtered.empty:
            fallback_type = "Bottom + Outer match"
            filtered = clothing_df[
                (clothing_df["recommended_bottom"] == bottom) &
                (clothing_df["recommended_outer"] == outer)
            ]
        if filtered.empty:
            fallback_type = "Top + Outer match"
            filtered = clothing_df[
                (clothing_df["recommended_top"] == top) &
                (clothing_df["recommended_outer"] == outer)
            ]
        if filtered.empty:
            fallback_type = "Top only match"
            filtered = clothing_df[clothing_df["recommended_top"] == top]
        if filtered.empty:
            fallback_type = "Bottom only match"
            filtered = clothing_df[clothing_df["recommended_bottom"] == bottom]
        if filtered.empty:
            fallback_type = f"Seasonal fallback ({season})"
            filtered = clothing_df[clothing_df["season"] == season]

        st.session_state.filtered_items = filtered
        st.session_state.shuffled_sample = filtered.sample(1) if not filtered.empty else None

        st.subheader(f"📍 Weather in {weather['city']}")
        st.metric("Temperature", f"{weather['temperature']}°C")
        st.metric("Condition", weather['description'].capitalize())
        st.metric("Humidity", f"{weather['humidity']}%")
        st.metric("Wind Speed", f"{weather['wind']} km/h")

        if fallback_type != "Full match":
            st.warning(f"⚠️ No exact match found. Showing closest: {fallback_type}")

# === Display Recommendation ===
if st.session_state.prediction is not None and st.session_state.shuffled_sample is not None:
    pred = st.session_state.prediction
    sample = st.session_state.shuffled_sample.iloc[0]

    st.markdown("---")
    st.subheader("🧠 Outfit Recommendation")

    # Dropdown values
    top_options = sorted(clothing_df['recommended_top'].dropna().unique())
    bottom_options = sorted(clothing_df['recommended_bottom'].dropna().unique())
    outer_options = sorted(clothing_df['recommended_outer'].dropna().unique())




    # 3-column layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👕 Shirt")
        selected_top = st.selectbox(
            "Choose Topwear",
            options=top_options,
            index=top_options.index(pred[0]) if pred[0] in top_options else 0
        )
        st.success(selected_top)
        img = get_image_from_folder(selected_top)
        if img:
            st.image(img, use_container_width=True)

        # 🔁 Re-recommend Shirt Only Button
        if st.button("🔁 Re-recommend Shirt Only"):
            filtered = st.session_state.filtered_items
            if not filtered.empty:
                current_bottom = st.session_state.prediction[1]
                current_outer = st.session_state.prediction[2]
                current_top = st.session_state.prediction[0]

                # Filter items with different top but same bottom and outer
                candidates = filtered[
                    (filtered["recommended_bottom"] == current_bottom) &
                    (filtered["recommended_outer"] == current_outer) &
                    (filtered["recommended_top"] != current_top)
                ]

                if not candidates.empty:
                    new_top_sample = candidates.sample(1).iloc[0]

                    # Update prediction with new top, keep bottom and outer same
                    new_pred = (
                        new_top_sample["recommended_top"],
                        current_bottom,
                        current_outer
                    )
                    st.session_state.prediction = new_pred

                    # Update shuffled_sample to the new item sample
                    st.session_state.shuffled_sample = pd.DataFrame([new_top_sample])

                    st.experimental_rerun()
                else:
                    st.warning("No more matching shirts to shuffle.")

    with col2:
        st.markdown("### 👖 Pants")
        selected_bottom = st.selectbox(
            "Choose Bottomwear",
            options=bottom_options,
            index=bottom_options.index(pred[1]) if pred[1] in bottom_options else 0
        )
        st.success(selected_bottom)
        img = get_image_from_folder(selected_bottom)
        if img:
            st.image(img, use_container_width=True)

    with col3:
        st.markdown("### 🧥 Outerwear")
        selected_outer = st.selectbox(
            "Choose Outerwear",
            options=outer_options,
            index=outer_options.index(pred[2]) if pred[2] in outer_options else 0
        )
        st.success(selected_outer)
        img = get_image_from_folder(selected_outer)
        if img:
            st.image(img, use_container_width=True)

    st.markdown("### 🛍️ Example Item:")
    st.info(sample.get("productdisplayname", "Sample item"))

    with st.expander("📋 More Item Details"):
        for col in ["productdisplayname", "articletype", "basecolour", "season", "usage"]:
            value = sample.get(col, None)
            if pd.notna(value):
                st.write(f"**{col.capitalize()}**: {value}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("👍 I liked this!"):
            st.success("Thanks! Feedback noted.")
    with col_b:
        if st.button("🔁 New Recommendation"):
            filtered = st.session_state.filtered_items
            if not filtered.empty:
                st.session_state.shuffled_sample = filtered.sample(1)
                st.experimental_rerun()
            else:
                st.warning("No more matching items to shuffle.")
