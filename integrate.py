# unified app.py
import os
import random
import joblib
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime
from glob import glob

# Optional: BLIP model for women (only load when needed)
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# === Config ===
st.set_page_config(page_title="WearSmart Unified", layout="wide")

# === Gender Selector ===
st.title("👕👗 WearSmart - Unified Clothing Recommender")
gender = st.radio("Select Gender", ["Men", "Women"], horizontal=True)

# === Weather API ===
def fetch_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    data = res.json()
    return {
        "temperature": data["main"]["temp"],
        "feels_like": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
        "weather_condition": data["weather"][0]["main"].lower(),
        "description": data["weather"][0]["description"],
        "city": data["name"]
    }

# === Shared Inputs ===
st.sidebar.header("🌦️ Weather Info")
city = st.sidebar.text_input("Enter your city", value="Lahore")
api_key = "4c703f15e3f9220de836884137342d5d"
time_of_day = st.sidebar.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"])
month = datetime.now().month
season_default = (
    'summer' if 5 <= month <= 8 else
    'autumn' if 9 <= month <= 10 else
    'winter' if 11 <= month or month <= 2 else
    'spring'
)
season = st.sidebar.selectbox("Season", ["summer", "winter", "spring", "autumn"], index=["summer", "winter", "spring", "autumn"].index(season_default))
occasion = st.sidebar.selectbox("Occasion", ["casual", "formal", "party", "sports"])
trigger = st.sidebar.button("Get Recommendation")

# === Session Init ===
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "image_set" not in st.session_state:
    st.session_state.image_set = None
if "liked" not in st.session_state:
    st.session_state.liked = None

# === Load BLIP (only if needed) ===
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("blip_finetunedggdata")
    model = BlipForConditionalGeneration.from_pretrained("blip_finetunedggdata")
    return processor, model.to("cuda" if torch.cuda.is_available() else "cpu")

# === Process ===
if trigger and city:
    weather = fetch_weather(city, api_key)
    if not weather:
        st.error("Failed to fetch weather for city")
    else:
        st.success(f"🌤️ Weather fetched for {weather['city']}!")

        st.subheader("📡 Current Weather Details")
        st.markdown(f"- **City:** {weather['city']}")
        st.markdown(f"- **Condition:** {weather['description'].capitalize()}")
        st.markdown(f"- **Temperature:** {weather['temperature']}°C")
        st.markdown(f"- **Feels Like:** {weather['feels_like']}°C")
        st.markdown(f"- **Humidity:** {weather['humidity']}%")
        st.markdown(f"- **Wind Speed:** {weather['wind_speed']} m/s")

        input_df = pd.DataFrame([{
            "temperature": weather["temperature"],
            "feels_like": weather["feels_like"],
            "humidity": weather["humidity"],
            "wind_speed": weather["wind_speed"],
            "weather_condition": weather["weather_condition"],
            "time_of_day": time_of_day,
            "season": season,
            "occasion": occasion,
            "mood": "good"
        }])

        if gender == "Men":
            model = joblib.load("weather_clothing_recommender.pkl")
            clothing_df = pd.read_csv("enhanced_weather_clothing_dataset_no_accessories.csv")
            clothing_df.columns = clothing_df.columns.str.strip().str.lower()
            pred = model.predict(input_df)[0]

            filtered = clothing_df[
                (clothing_df["recommended_top"] == pred[0]) &
                (clothing_df["recommended_bottom"] == pred[1]) &
                (clothing_df["recommended_outer"] == pred[2])
            ]
            if filtered.empty:
                filtered = clothing_df[clothing_df["season"] == season]

            sample = filtered.sample(1).iloc[0] if not filtered.empty else None
            st.session_state.prediction = pred
            st.session_state.image_set = sample
            st.session_state.clothing_df = clothing_df
            st.session_state.input_df = input_df

        else:
            model = joblib.load("weather_clothing_recommender_women.pkl")
            pred = model.predict(input_df)[0]
            st.session_state.prediction = pred

# === Display ===
if st.session_state.prediction is not None:
    top, bottom, outer = st.session_state.prediction
    st.subheader("👕 Recommended Outfit")
    col1, col2, col3 = st.columns(3)

    def get_img(folder_name):
        if not folder_name or not isinstance(folder_name, str):
            return None
        try:
            folder = folder_name.strip().lower()
            path = os.path.join(os.getcwd(), folder)
            if not os.path.isdir(path):
                st.warning(f"⚠️ Folder does not exist: {path}")
                return None
            files = glob(os.path.join(path, "*.*"))
            image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            if not image_files:
                st.warning(f"⚠️ No image found in: {path}")
            return random.choice(image_files) if image_files else None
        except Exception as e:
            st.error(f"Error loading image from folder '{folder_name}': {e}")
            return None

    if gender == "Men":
        edit = st.checkbox("✏️ Edit Recommendation")

        if edit:
            clothing_df = st.session_state.clothing_df
            top_options = sorted(clothing_df["recommended_top"].dropna().unique())
            bottom_options = sorted(clothing_df["recommended_bottom"].dropna().unique())
            outer_options = sorted(clothing_df["recommended_outer"].dropna().unique())

            top = col1.selectbox("Top", top_options, index=top_options.index(top) if top in top_options else 0)
            bottom = col2.selectbox("Bottom", bottom_options, index=bottom_options.index(bottom) if bottom in bottom_options else 0)
            outer = col3.selectbox("Outerwear", outer_options, index=outer_options.index(outer) if outer in outer_options else 0)

            st.session_state.prediction = (top, bottom, outer)

        for label, col, item in zip(["Top", "Bottom", "Outer"], [col1, col2, col3], [top, bottom, outer]):
            col.markdown(f"### {label}")
            if item and isinstance(item, str):
                col.success(item)
                img = get_img(item)
                if img:
                    col.image(img, use_container_width=True)
                else:
                    col.warning("⚠️ No image found.")
            else:
                col.info("❌ No recommendation.")

        st.markdown("### 🛍️ Example Item")
        sample = st.session_state.image_set
        if sample is not None:
            st.info(sample.get("productdisplayname", "Sample item"))

    else:
        processor, blip_model = load_blip()

        def blip_caption(img_path):
            image = Image.open(img_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(blip_model.device)
            out = blip_model.generate(**inputs, max_new_tokens=30)
            return processor.decode(out[0], skip_special_tokens=True)

        edit = st.checkbox("✏️ Edit Recommendation")

        base_folder = "clothing_images"
        clothing_categories = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])

        if edit:
            top = col1.selectbox("Top", clothing_categories, index=clothing_categories.index(top) if top in clothing_categories else 0)
            bottom = col2.selectbox("Bottom", clothing_categories, index=clothing_categories.index(bottom) if bottom in clothing_categories else 0)
            outer = col3.selectbox("Outerwear", clothing_categories, index=clothing_categories.index(outer) if outer in clothing_categories else 0)
            st.session_state.prediction = (top, bottom, outer)

        pred_items = (top, bottom, outer)
        for label, col, item in zip(["Top", "Bottom", "Outer"], [col1, col2, col3], pred_items):
            col.markdown(f"### {label}")
            if isinstance(item, str) and item.lower() != "none":
                folder = os.path.join(base_folder, item.strip().lower())
                if os.path.isdir(folder):
                    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp"))]
                    if files:
                        img_path = os.path.join(folder, random.choice(files))
                        caption = blip_caption(img_path)
                        col.image(Image.open(img_path).resize((200, 250)), caption=caption)
                    else:
                        col.warning(f"No images found in `{folder}`")
                else:
                    col.error(f"Folder not found: `{folder}`")
            else:
                col.info("No item recommended.")

st.markdown("---")
st.markdown("### ❤️ Feedback")
col_a, col_b = st.columns(2)
if col_a.button("👍 I liked this!"):
    st.session_state.liked = True
    st.success("Thanks for the feedback!")

    if gender == "Men":
        feedback_row = st.session_state.input_df.copy()
        top, bottom, outer = st.session_state.prediction
        feedback_row["top"] = top
        feedback_row["bottom"] = bottom
        feedback_row["outer"] = outer

        if not os.path.exists("feedback_log.csv"):
            feedback_row.to_csv("feedback_log.csv", index=False)
        else:
            feedback_row.to_csv("feedback_log.csv", mode="a", index=False, header=False)

if col_b.button("👎 Dislike"):
    st.session_state.liked = False
    st.info("Feedback noted. We'll keep improving.")