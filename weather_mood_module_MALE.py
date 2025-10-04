# =========================
# WearSmart (Male) — Weather model + Image DB + Mood(Color) filter
# =========================

# ---- Force Transformers to avoid TensorFlow/Flax (Windows + Py3.13 friendly)
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---- Standard libs
import importlib
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

# ---- Third-party
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---- Your weather util
from weather_api import fetch_weather  # must return dict like: {city, temperature, humidity, wind, condition, description}

# =========================
# Constants / Config
# =========================
DEFAULT_IMAGES_ROOT = "clothing_images_men"   # change if needed
DB_PATH = Path("data") / "image_index_men.db"

# =========================
# Streamlit UI config
# =========================
st.set_page_config(page_title="WearSmart (Men)", page_icon="🧥", layout="centered")
st.title("🧥 WearSmart — AI Clothing Recommender (Men)")

# =========================
# Scikit-learn pickle compatibility shim (if needed)
# =========================
try:
    ct = importlib.import_module("sklearn.compose._column_transformer")
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

# =========================
# Load weather→clothing model & dataset
# =========================
try:
    model = joblib.load("weather_clothing_recommender.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

try:
    clothing_df = pd.read_csv("enhanced_weather_clothing_dataset_no_accessories.csv")
    clothing_df.columns = clothing_df.columns.str.strip().str.lower()
except Exception as e:
    st.error(f"Failed to load dataset CSV: {e}")
    st.stop()

# =========================
# BLIP Captioning (PyTorch)
# =========================
@st.cache_resource
def load_blip_model():
    """Load BLIP (local folder if present else HF)."""
    local_dir = Path("blip_finetunedggdata")  # optional finetuned dir
    if local_dir.exists():
        proc = BlipProcessor.from_pretrained(local_dir.as_posix())
        mdl = BlipForConditionalGeneration.from_pretrained(local_dir.as_posix())
    else:
        proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        mdl = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    return proc, mdl

try:
    processor, blip_model = load_blip_model()
except Exception as e:
    st.warning(f"BLIP failed to load; captions disabled. Details: {e}")
    processor, blip_model = None, None

PROMPT = "Describe this clothing item and include its main color in one word."
PROMPT_ECHO_RE = re.compile(
    r"^\s*(top|bottom|outerwear)\s*:\s*|^\s*describe.*?one word\.?\s*",
    re.IGNORECASE,
)

def _clean_caption(text: str) -> str:
    text = PROMPT_ECHO_RE.sub("", text or "").strip()
    text = text.strip(" ,.-")
    return (
        text.replace("off white", "off-white")
            .replace("navy blue", "navy-blue")
            .replace("sky blue", "sky-blue")
            .replace("baby blue", "baby-blue")
            .replace("camel colored", "camel-colored")
    ) or "clothing item"

def blip_caption(image: Image.Image, prompt: Optional[str] = None) -> str:
    if processor is None or blip_model is None:
        return "caption unavailable"
    if prompt is None:
        inputs = processor(image, return_tensors="pt").to(blip_model.device)
    else:
        try:
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(blip_model.device)
        except TypeError:
            inputs = processor(image, return_tensors="pt").to(blip_model.device)
    out = blip_model.generate(**inputs, max_new_tokens=32)
    return processor.decode(out[0], skip_special_tokens=True).strip()

def generate_caption(image_path: str) -> str:
    """Unprompted first; if no color found, fallback to prompted, then clean."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return "clothing item"
    cap = _clean_caption(blip_caption(image))
    if extract_colors(cap):
        return cap
    cap2 = _clean_caption(blip_caption(image, PROMPT))
    return cap2

# =========================
# Color vocab & helpers
# =========================
COLOR_VOCAB = [
    "black","white","gray","grey","silver","gold","golden",
    "red","maroon","crimson",
    "orange","amber","rust",
    "yellow","mustard",
    "green","olive","mint","teal",
    "blue","navy","cyan","turquoise","royal","sky-blue","baby-blue","navy-blue",
    "purple","violet","lavender","magenta","pink","hot-pink",
    "beige","brown","tan","camel","camel-colored","cream","off-white","charcoal",
]
COLOR_ALIAS = {
    "grey": "gray",
    "golden": "gold",
    "off-white": "white",
    "charcoal": "gray",
    "royal": "blue",
    "sky-blue": "blue",
    "baby-blue": "blue",
    "navy-blue": "blue",
    "camel-colored": "camel",
    "hot-pink": "pink",
}

def normalize_color(word: str) -> str:
    return COLOR_ALIAS.get(word.lower(), word.lower())

SHADE_PREFIX = r"(?:light|dark|deep|bright|pale)\s+"
BASE = r"|".join(map(re.escape, COLOR_VOCAB))
COLOR_PATTERN = re.compile(rf"\b(?:{SHADE_PREFIX})?({BASE})\b", re.IGNORECASE)

def extract_colors(text: str) -> List[str]:
    if not text:
        return []
    found = [normalize_color(m.group(1)) for m in COLOR_PATTERN.finditer(text)]
    seen, out = set(), []
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def color_terms_for_query(color: str) -> List[str]:
    canonical = normalize_color(color)
    terms = {canonical}
    for k, v in COLOR_ALIAS.items():
        if v == canonical:
            terms.add(k)
    return sorted(terms)

# =========================
# Image DB (SQLite)
# =========================
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def open_conn():
    return sqlite3.connect(DB_PATH.as_posix())

def init_db():
    with open_conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                label TEXT,
                caption TEXT,
                colors TEXT
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_label ON images(label)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_colors ON images(colors)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_caption ON images(caption)")

def upsert_image_record(path: str, label: str, caption: str, colors: List[str]):
    colors_csv = ",".join(colors)
    with open_conn() as con:
        con.execute("""
            INSERT INTO images (path, label, caption, colors)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
              label=excluded.label,
              caption=excluded.caption,
              colors=excluded.colors
        """, (path, label, caption, colors_csv))

def get_db_caption(path: str) -> Optional[str]:
    with open_conn() as con:
        row = con.execute("SELECT caption FROM images WHERE path = ?", (path,)).fetchone()
        return row[0] if row else None

def iter_image_files(root: Path):
    if not root.exists():
        return
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = label_dir.name
        for img_path in label_dir.iterdir():
            if img_path.suffix.lower() in {".jpg",".jpeg",".png",".webp"}:
                yield label, img_path

@st.cache_data(show_spinner=False)
def db_count_images() -> int:
    init_db()
    with open_conn() as con:
        (count,) = con.execute("SELECT COUNT(*) FROM images").fetchone()
    return int(count)

def build_or_refresh_index(images_root: Path, recaption: bool = False) -> Tuple[int,int]:
    """Index all images under images_root (folder per label)."""
    init_db()
    indexed, skipped = 0, 0
    with open_conn() as con:
        existing = {row[0] for row in con.execute("SELECT path FROM images")}
    all_items = list(iter_image_files(images_root))
    if not all_items:
        return 0, 0

    progress = st.progress(0.0)
    total = len(all_items)

    for i, (label, img_path) in enumerate(all_items, start=1):
        p = str(img_path.resolve())
        need = recaption or (p not in existing)
        if not need:
            skipped += 1
        else:
            cap = generate_caption(p)
            colors = extract_colors(cap)
            upsert_image_record(p, label, cap, colors)
            indexed += 1
        progress.progress(i/total)

    db_count_images.clear()
    return indexed, skipped

def query_images_by_color(color: str, limit: int = 24) -> List[Tuple[str,str,str,str]]:
    """Return rows (path,label,caption,colors) for a given color (canonical + aliases)."""
    init_db()
    terms = color_terms_for_query(color)
    canon = normalize_color(color)
    params = [f"%{canon}%"] + [f"%{t.lower()}%" for t in terms]
    where = ["colors LIKE ?"] + ["LOWER(caption) LIKE ?"] * len(terms)

    sql = f"""
        SELECT path,label,caption,colors
        FROM images
        WHERE {" OR ".join(where)}
        ORDER BY RANDOM() LIMIT ?
    """
    params.append(limit)
    with open_conn() as con:
        return con.execute(sql, params).fetchall()

def sample_any_images(limit: int = 24) -> List[Tuple[str,str,str,str]]:
    with open_conn() as con:
        return con.execute(
            "SELECT path,label,caption,colors FROM images ORDER BY RANDOM() LIMIT ?",
            (limit,),
        ).fetchall()

def caption_from_db_or_generate(path: str, label: str) -> str:
    cap = get_db_caption(path)
    if cap:
        return cap
    cap = generate_caption(path)
    upsert_image_record(path, label, cap, extract_colors(cap))
    db_count_images.clear()
    return cap

# =========================
# Helpers for images root
# =========================
if "images_root" not in st.session_state:
    st.session_state.images_root = DEFAULT_IMAGES_ROOT

def get_images_root() -> Path:
    return Path(st.session_state.images_root)

def get_image_from_folder(category: str) -> Optional[str]:
    """Pick a random image from images_root/category."""
    folder = get_images_root() / category
    if not folder.is_dir():
        return None
    files = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
    return str(random.choice(files)) if files else None

# =========================
# Sidebar — Weather & trigger
# =========================
with st.sidebar:
    st.header("🌦️ Weather Info")
    city = st.text_input("Enter your city", value="Lahore")
    api_key = "4c703f15e3f9220de836884137342d5d"
    if st.button("Get Recommendation"):
        st.session_state.trigger = True

# Session init
for key, default in [
    ("shuffled_sample", None),
    ("prediction", None),
    ("filtered_items", pd.DataFrame()),
    ("trigger", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =========================
# Run recommendation when triggered
# =========================
if st.session_state.trigger:
    st.session_state.trigger = False

    weather = fetch_weather(city, api_key)  # your function

    if weather:
        hour = datetime.now().hour
        time_of_day = 'morning' if 5 <= hour < 12 else 'afternoon' if 12 <= hour < 17 else 'evening' if 17 <= hour < 21 else 'night'
        month = datetime.now().month
        season = 'summer' if 5 <= month <= 8 else 'fall' if 9 <= month <= 10 else 'winter' if month >= 11 or month <= 2 else 'spring'

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
        try:
            top, bottom, outer = model.predict(input_df)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.session_state.prediction = (top, bottom, outer)

        # progressive fallback search in your dataset
        filtered = clothing_df[
            (clothing_df["recommended_top"] == top) &
            (clothing_df["recommended_bottom"] == bottom) &
            (clothing_df["recommended_outer"] == outer)
        ]
        fallback_type = "Full match"
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
        colwx = st.columns(4)
        colwx[0].metric("Temperature", f"{weather['temperature']}°C")
        colwx[1].metric("Condition", weather['description'].capitalize())
        colwx[2].metric("Humidity", f"{weather['humidity']}%")
        colwx[3].metric("Wind Speed", f"{weather['wind']} km/h")

        if fallback_type != "Full match":
            st.warning(f"⚠️ No exact match found. Showing closest: {fallback_type}")

# =========================
# Display recommendation (uses DB captions where possible)
# =========================
if st.session_state.prediction is not None and st.session_state.shuffled_sample is not None:
    pred = st.session_state.prediction
    sample = st.session_state.shuffled_sample.iloc[0]

    st.markdown("---")
    st.subheader("🧠 Outfit Recommendation")

    top_options = sorted(clothing_df['recommended_top'].dropna().unique())
    bottom_options = sorted(clothing_df['recommended_bottom'].dropna().unique())
    outer_options = sorted(clothing_df['recommended_outer'].dropna().unique())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👕 Top")
        selected_top = st.selectbox(
            "Choose Topwear",
            options=top_options,
            index=top_options.index(pred[0]) if pred[0] in top_options else 0
        )
        img_path = get_image_from_folder(selected_top)
        if img_path:
            cap = caption_from_db_or_generate(str(Path(img_path).resolve()), selected_top)
            st.image(img_path, use_container_width=True, caption=cap)

        if st.button("🔁 Re-recommend Top Only"):
            filtered = st.session_state.filtered_items
            if not filtered.empty:
                current_bottom = st.session_state.prediction[1]
                current_outer = st.session_state.prediction[2]
                current_top = st.session_state.prediction[0]

                candidates = filtered[
                    (filtered["recommended_bottom"] == current_bottom) &
                    (filtered["recommended_outer"] == current_outer) &
                    (filtered["recommended_top"] != current_top)
                ]
                if not candidates.empty:
                    new_top_sample = candidates.sample(1).iloc[0]
                    new_pred = (new_top_sample["recommended_top"], current_bottom, current_outer)
                    st.session_state.prediction = new_pred
                    st.session_state.shuffled_sample = pd.DataFrame([new_top_sample])
                    st.experimental_rerun()
                else:
                    st.warning("No more matching tops to shuffle.")

    with col2:
        st.markdown("### 👖 Bottom")
        selected_bottom = st.selectbox(
            "Choose Bottomwear",
            options=bottom_options,
            index=bottom_options.index(pred[1]) if pred[1] in bottom_options else 0
        )
        img_path = get_image_from_folder(selected_bottom)
        if img_path:
            cap = caption_from_db_or_generate(str(Path(img_path).resolve()), selected_bottom)
            st.image(img_path, use_container_width=True, caption=cap)

    with col3:
        st.markdown("### 🧥 Outerwear")
        selected_outer = st.selectbox(
            "Choose Outerwear",
            options=outer_options,
            index=outer_options.index(pred[2]) if pred[2] in outer_options else 0
        )
        img_path = get_image_from_folder(selected_outer)
        if img_path:
            cap = caption_from_db_or_generate(str(Path(img_path).resolve()), selected_outer)
            st.image(img_path, use_container_width=True, caption=cap)

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

# =======================================================
# Mood (Color) Based Recommendations — using the DB
# =======================================================
st.markdown("---")
st.header("🎨 Mood (Color) Recommendations")

with st.expander("🗂️ Build / Refresh Image Index"):
    st.caption(f"Indexed images stored at `{DB_PATH}`. Total indexed: **{db_count_images()}**")
    colA, colB, colC = st.columns([2,1,1])
    images_root_input = colA.text_input("Images root folder", value=st.session_state.images_root or DEFAULT_IMAGES_ROOT)
    recaption = colB.checkbox("Re-caption all", value=False)
    if colC.button("🔄 Index Now"):
        root_path = Path(images_root_input)
        if not root_path.exists():
            st.error(f"Folder not found: {root_path}")
        else:
            st.session_state.images_root = images_root_input  # keep consistent for display
            with st.spinner("Indexing images (first run can take a while)…"):
                added, skipped = build_or_refresh_index(root_path, recaption=recaption)
            st.success(f"Done! Added/updated: {added}, skipped: {skipped}.")
            st.caption(f"New DB count: **{db_count_images()}**")

st.markdown("#### Pick your mood color")
cc1, cc2 = st.columns([2,1])
user_color = cc1.text_input("Type a color (e.g., 'blue', 'black', 'red', 'green', 'yellow')").strip()
limit = cc2.slider("How many images", min_value=6, max_value=48, value=18, step=6)

go_col, _ = st.columns([1,3])
if go_col.button("🎯 Show Mood Matches"):
    if not user_color:
        st.warning("Please type a color.")
    else:
        if db_count_images() == 0:
            st.error("No images indexed yet. Open the expander above and click **Index Now**.")
        else:
            matches = query_images_by_color(user_color, limit=limit)
            if not matches:
                st.info(f"No matches found for **{user_color}**. Showing random images instead.")
                matches = sample_any_images(limit=limit)

            grouped = defaultdict(list)
            for path, label, caption, _colors in matches:
                grouped[label].append((path, caption))

            st.markdown(f"### 🧺 Your **{normalize_color(user_color)}** items")
            for label, items in grouped.items():
                st.markdown(f"#### {label.title()} ({len(items)})")
                cols = st.columns(3)
                for i, (path, caption) in enumerate(items):
                    with cols[i % 3]:
                        try:
                            img = Image.open(path).resize((220, 260))
                            st.image(img, caption=caption, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not display {path}: {e}")
                st.markdown("---")
