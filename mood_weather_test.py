import os
import random
from glob import glob
from pathlib import Path
import sqlite3
from PIL import Image

import streamlit as st
import pandas as pd
import requests
import joblib
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================
# --- Streamlit Config ---
# =========================
st.set_page_config(page_title="AI Outfit + Mood Recommender", layout="wide")
st.title("🌤️ + 🎨 AI Weather + Mood Clothing Recommender")

# =========================
# --- BLIP2 Model Load ---
# =========================
@st.cache_resource
def load_blip_model():
    local_dir = Path("blip_finetunedggdata")
    if local_dir.exists():
        proc = BlipProcessor.from_pretrained(local_dir.as_posix())
        mdl = BlipForConditionalGeneration.from_pretrained(local_dir.as_posix())
    else:
        proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        mdl = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    return proc, mdl

processor, blip_model = load_blip_model()

def generate_caption(image_path: str) -> str:
    if processor is None or blip_model is None:
        return "Caption unavailable"
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(blip_model.device)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

# =========================
# --- SQLite DB Setup ---
# =========================
DB_PATH = "clothing_db.db"

def create_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clothing_items (
        id INTEGER PRIMARY KEY,
        image_path TEXT,
        caption TEXT,
        category TEXT,
        base_colour TEXT
    )
    """)
    conn.commit()
    conn.close()

def store_item(image_path, caption, category, base_colour):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO clothing_items (image_path, caption, category, base_colour)
    VALUES (?, ?, ?, ?)
    """, (image_path, caption, category, base_colour))
    conn.commit()
    conn.close()

# =========================
# --- Preprocess Images ---
# =========================
def preprocess_images(folder_dict):
    create_db()
    for category, folder_path in folder_dict.items():
        images = glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.png"))
        for img in images:
            caption = generate_caption(img)
            base_colour = category  # or extract from caption if needed
            store_item(img, caption, category, base_colour)
            st.text(f"Processed {img} -> {caption}")

# =========================
# --- Fetch from DB ---
# =========================
def fetch_by_category_color(category_list, color_list):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    results = []
    for cat, col in zip(category_list, color_list):
        cursor.execute("""
        SELECT * FROM clothing_items WHERE category = ? AND base_colour = ?
        """, (cat, col))
        results.extend(cursor.fetchall())
    conn.close()
    return results

# =========================
# --- Weather API ---
# =========================
API_KEY = "4c703f15e3f9220de836884137342d5d"
def fetch_weather(city: str):
    try:
        resp = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric", timeout=10)
        if resp.status_code != 200: return None
        data = resp.json()
        return {
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "weather_condition": data["weather"][0]["main"].lower(),
        }
    except Exception:
        return None

# =========================
# --- User Inputs ---
# =========================
city = st.text_input("Enter your city:")

# Define your clothing folders
folders = {
    "shirt": os.path.join("C:/Users/hP/Desktop/FYP/weather_module/shirt"),
    "pants": os.path.join("C:/Users/hP/Desktop/FYP/weather_module/pants"),
    "shorts": os.path.join("C:/Users/hP/Desktop/FYP/weather_module/shorts"),
    "sweater": os.path.join("C:/Users/hP/Desktop/FYP/weather_module/sweater"),
    "t-shirt": os.path.join("C:/Users/hP/Desktop/FYP/weather_module/t-shirt"),
    "trousers": os.path.join("C:/Users/hP/Desktop/FYP/weather_module/trousers")
}

if st.button("⚡ Preprocess All Images"):
    preprocess_images(folders)
    st.success("All images processed and stored in DB!")

# Step 1: Weather-based recommendation (simulate here)
recommended_categories = ["shirt", "pants", "outer"]  # Replace with your weather model output
st.write("Weather-recommended clothing types:", recommended_categories)

# Step 2: Mood / Color input for each recommended item
colors_input = []
for cat in recommended_categories:
    color = st.text_input(f"Enter color for {cat}")
    colors_input.append(color.strip().capitalize())

if st.button("🎨 Show Mood-Based Recommendations"):
    matched_items = fetch_by_category_color(recommended_categories, colors_input)
    if not matched_items:
        st.warning("No items found matching your colors!")
    else:
        st.success(f"Found {len(matched_items)} items matching your colors!")
        for item in matched_items:
            st.image(item[1], caption=f"{item[2]} ({item[4]})", width=200)
