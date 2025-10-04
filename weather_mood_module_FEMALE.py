# =========================
# AI Outfit Recommender (Women) — Streamlit with Image DB + Mood(color) filter
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

# ---- Third-party
import streamlit as st
import pandas as pd
import requests
import joblib
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# =========================
# Streamlit UI config
# =========================
st.set_page_config(page_title="AI Outfit Recommender", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("👗 AI Clothing Recommender for Women")

# =========================
# Scikit-learn pickle compatibility shim
# =========================
try:
    ct = importlib.import_module("sklearn.compose._column_transformer")
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Compat placeholder for older scikit-learn pickles."""
            pass
        ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

# =========================
# Load weather→clothing model
# =========================
MODEL_PATH = Path("weather_clothing_recommender_women.pkl")
if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH}`. Place it next to the app.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH.as_posix())
except Exception as e:
    st.error(
        "Failed to load the scikit-learn model.\n\n"
        f"**Details:** {type(e).__name__}: {e}\n\n"
        "Tip: Re-save the model with your current scikit-learn if compatibility errors persist."
    )
    st.stop()

# =========================
# Load BLIP (PyTorch only)
# =========================
@st.cache_resource
def load_blip_model():
    """
    Loads BLIP weights. Uses local folder `blip_finetunedggdata` if present,
    otherwise falls back to the base captioning model from HF.
    """
    local_dir = Path("blip_finetunedggdata")
    if local_dir.exists():
        proc = BlipProcessor.from_pretrained(local_dir.as_posix())
        mdl = BlipForConditionalGeneration.from_pretrained(local_dir.as_posix())
    else:
        # First run may download; later runs are cached by HF
        proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        mdl = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    return proc, mdl

try:
    processor, blip_model = load_blip_model()
except Exception as e:
    st.warning(
        "Could not load the BLIP model; captions will be disabled.\n\n"
        f"**Details:** {type(e).__name__}: {e}"
    )
    processor, blip_model = None, None

# =========================
# Color vocabulary & helpers
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
    w = word.lower()
    return COLOR_ALIAS.get(w, w)

# allow shades like "dark blue", "bright red"
SHADE_PREFIX = r"(?:light|dark|deep|bright|pale)\s+"
BASE = r"|".join(map(re.escape, COLOR_VOCAB))
COLOR_PATTERN = re.compile(rf"\b(?:{SHADE_PREFIX})?({BASE})\b", re.IGNORECASE)

def extract_colors(text: str) -> List[str]:
    if not text:
        return []
    found = [normalize_color(m.group(1)) for m in COLOR_PATTERN.finditer(text)]
    # de-dup preserving order
    seen = set()
    out = []
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

# =========================
# Captioning (clean, no prompt echo)
# =========================
PROMPT = "Describe this clothing item and include its main color in one word."
PROMPT_ECHO_RE = re.compile(
    r"^\s*(top|bottom|outerwear)\s*:\s*|^\s*describe.*?one word\.?\s*",
    re.IGNORECASE,
)

def _clean_caption(text: str) -> str:
    text = PROMPT_ECHO_RE.sub("", text or "").strip()
    text = text.strip(" ,.-")
    text = (
        text.replace("off white", "off-white")
            .replace("navy blue", "navy-blue")
            .replace("sky blue", "sky-blue")
            .replace("baby blue", "baby-blue")
            .replace("camel colored", "camel-colored")
    )
    return text or "clothing item"

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
    # 1) Unprompted
    cap = _clean_caption(blip_caption(image))
    if extract_colors(cap):
        return cap
    # 2) Prompted fallback
    cap2 = _clean_caption(blip_caption(image, PROMPT))
    return cap2

# =========================
# Weather API
# =========================
API_KEY = "4c703f15e3f9220de836884137342d5d"  # move to st.secrets if you prefer

def fetch_weather(city: str):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        resp = requests.get(url, timeout=10)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    data = resp.json()
    try:
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
# Image DB (SQLite)
# =========================
DB_PATH = Path("data") / "image_index.db"
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
        row = con.execute("SELECT COUNT(*) FROM images").fetchone()
    return int(row[0]) if row else 0

def build_or_refresh_index(clothing_root: Path, recaption: bool = False) -> Tuple[int,int]:
    """
    Walks clothing_root, inserts/updates records.
    If recaption=True, regenerates captions even if path already exists.
    Returns: (indexed_count, skipped_count)
    """
    init_db()
    indexed, skipped = 0, 0
    with open_conn() as con:
        existing = {row[0] for row in con.execute("SELECT path FROM images")}
    # simple progress bar
    all_items = list(iter_image_files(clothing_root))
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
            caption = generate_caption(p)
            colors = extract_colors(caption)
            upsert_image_record(p, label, caption, colors)
            indexed += 1
        progress.progress(i/total)

    db_count_images.clear()  # refresh cache
    return indexed, skipped

def color_terms_for_query(color: str) -> List[str]:
    """Return canonical + all alias forms that map to it (for caption fallback search)."""
    canonical = normalize_color(color)
    terms = {canonical}
    # include alias keys that map to this canonical
    for k, v in COLOR_ALIAS.items():
        if v == canonical:
            terms.add(k)
    return sorted(terms)

def query_images_by_color(color: str, limit: int = 24) -> List[Tuple[str,str,str,str]]:
    """
    Returns rows: (path, label, caption, colors).
    Primary: search `colors` (canonical tags).
    Fallback: OR match in raw `caption` for any alias terms.
    """
    init_db()
    terms = color_terms_for_query(color)
    params = []
    where_clauses = []

    # colors column (canonical tags)
    canon = normalize_color(color)
    where_clauses.append("colors LIKE ?")
    params.append(f"%{canon}%")

    # caption fallback for alias tokens
    for t in terms:
        where_clauses.append("LOWER(caption) LIKE ?")
        params.append(f"%{t.lower()}%")

    sql = f"""
        SELECT path,label,caption,colors
        FROM images
        WHERE {" OR ".join(where_clauses)}
        ORDER BY RANDOM() LIMIT ?
    """
    params.append(limit)
    with open_conn() as con:
        rows = con.execute(sql, params).fetchall()
    return rows

def sample_any_images(limit: int = 24) -> List[Tuple[str,str,str,str]]:
    with open_conn() as con:
        rows = con.execute(
            "SELECT path,label,caption,colors FROM images ORDER BY RANDOM() LIMIT ?",
            (limit,),
        ).fetchall()
    return rows

def caption_from_db_or_generate(path: str, label: str) -> str:
    """Use DB caption if present, else caption & upsert (so display always clean)."""
    cap = get_db_caption(path)
    if cap:
        return cap
    cap = generate_caption(path)
    upsert_image_record(path, label, cap, extract_colors(cap))
    db_count_images.clear()
    return cap

# =========================
# Session state
# =========================
if "predictions" not in st.session_state:
    st.session_state.predictions = {"top": None, "bottom": None, "outer": None}
if "liked" not in st.session_state:
    st.session_state.liked = None

# =========================
# WEATHER-BASED RECOMMENDER (original flow)
# =========================
st.markdown("#### 🌍 Weather Location")
city = st.text_input("Enter your city:")

st.markdown("#### 🧭 Choose Preferences")
time_of_day = st.selectbox("Time of Day", ["morning", "afternoon", "evening", "night"])
season = st.selectbox("Season", ["summer", "winter", "spring", "autumn"])
occasion = st.selectbox("Occasion", ["casual", "formal", "party", "sports"])

if st.button("👚 Recommend Outfit"):
    if not city:
        st.warning("Please enter your city.")
    else:
        weather = fetch_weather(city)
        if not weather:
            st.error("❌ Failed to fetch weather. Check the city name.")
        else:
            st.success("🌤️ Weather fetched successfully!")
            st.subheader("📡 Current Weather:")
            st.markdown(f"- **Condition:** {weather['weather_condition'].capitalize()}")
            st.markdown(f"- **Temperature:** {weather['temperature']}°C (Feels like {weather['feels_like']}°C)")
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
            }])

            try:
                pred_top, pred_bottom, pred_outer = model.predict(input_df)[0]
            except Exception as e:
                st.error(
                    "Prediction failed. Ensure the pipeline's expected columns match the DataFrame."
                    f"\n\n**Details:** {type(e).__name__}: {e}"
                )
                st.stop()

            st.session_state.predictions = {
                "top": pred_top,
                "bottom": pred_bottom,
                "outer": pred_outer,
            }
            st.session_state.liked = None  # reset feedback

# =========================
# Shuffle buttons for model picks
# =========================
preds = st.session_state.predictions
if preds["top"]:
    st.markdown("### 🔁 Shuffle Individual Items")
    col_shuffle1, col_shuffle2, col_shuffle3 = st.columns(3)

    def shuffle_item(item_type: str):
        images_root = Path("clothing_images")
        if not images_root.exists():
            st.error("`clothing_images/` folder not found.")
            return
        all_folders = [p.name for p in images_root.iterdir() if p.is_dir()]
        current = preds[item_type].lower()
        options = [f for f in all_folders if f.lower() != current]
        if options:
            preds[item_type] = random.choice(options)

    if col_shuffle1.button("Shuffle Top 👕"):
        shuffle_item("top")
    if col_shuffle2.button("Shuffle Bottom 👖"):
        shuffle_item("bottom")
    if col_shuffle3.button("Shuffle Outerwear 🧥"):
        shuffle_item("outer")

# =========================
# Display outfit for model picks (reusing DB captions)
# =========================
def show_image_with_caption(pred_label: str, label: str, col):
    folder = Path("clothing_images") / pred_label.lower()
    if not folder.is_dir():
        col.error(f"{label}: {pred_label} (Folder missing)")
        return
    image_files = [p for p in folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    if not image_files:
        col.warning(f"{label}: {pred_label} (No images found)")
        return
    chosen = random.choice(image_files)
    path_str = str(chosen.resolve())
    cap = caption_from_db_or_generate(path_str, pred_label)
    try:
        img = Image.open(chosen).resize((200, 250))
        col.image(img, caption=f"{label}: {cap}", use_container_width=True)
    except Exception as e:
        col.warning(f"Could not display {path_str}: {e}")

if preds["top"]:
    st.markdown("### 🧥 Your AI-Picked Outfit")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    show_image_with_caption(preds["top"], "Top", col1)
    show_image_with_caption(preds["bottom"], "Bottom", col2)
    if preds["outer"].lower() != "none":
        show_image_with_caption(preds["outer"], "Outerwear", col3)
    else:
        col3.markdown("**Outerwear:** None needed")

    st.markdown("---")
    st.markdown("### ❤️ Do You Like This Recommendation?")
    col_like, col_dislike = st.columns(2)
    if col_like.button("👍 Like"):
        st.session_state.liked = True
        st.success("Thanks for your feedback! You liked this look.")
    if col_dislike.button("👎 Dislike"):
        st.session_state.liked = False
        st.info("Got it! We'll keep improving.")
    if st.session_state.liked is not None:
        feedback = "Liked" if st.session_state.liked else "Disliked"
        st.caption(f"📝 Feedback: You **{feedback}** this outfit.")
    st.caption("🎲 Click Shuffle or Recommend Again to try new combinations!")

# =======================================================
# NEW: Image Indexer + Mood (Color) Based Recommendations
# =======================================================
st.markdown("---")
st.header("🎨 Mood (Color) Recommendations")

with st.expander("🗂️ Build / Refresh Image Index"):
    st.caption(f"Indexed images stored at `{DB_PATH}`. Total indexed right now: **{db_count_images()}**")
    colA, colB, colC = st.columns([2,1,1])
    root_dir = colA.text_input("Images root folder", value="clothing_images")
    recaption = colB.checkbox("Re-caption all", value=False)
    if colC.button("🔄 Index Now"):
        root_path = Path(root_dir)
        if not root_path.exists():
            st.error(f"Folder not found: {root_path}")
        else:
            with st.spinner("Indexing images (first run can take a while)…"):
                added, skipped = build_or_refresh_index(root_path, recaption=recaption)
            st.success(f"Done! Added/updated: {added}, skipped: {skipped}.")
            st.caption(f"New DB count: **{db_count_images()}**")

st.markdown("#### Pick your mood color")
color_col1, color_col2 = st.columns([2,1])
user_color = color_col1.text_input("Type a color (e.g., 'blue', 'black', 'red', 'green', 'pink')").strip()
limit = color_col2.slider("How many images", min_value=6, max_value=48, value=18, step=6)

col_go1, _ = st.columns([1,2])
if col_go1.button("🎯 Show Mood Matches"):
    if not user_color:
        st.warning("Please type a color.")
    else:
        if db_count_images() == 0:
            st.error("No images indexed yet. Open the expander above and click **Index Now**.")
        else:
            matches = query_images_by_color(user_color, limit=limit)
            if not matches:
                st.info(f"No matches found for color **{user_color}**. Showing random images instead.")
                matches = sample_any_images(limit=limit)

            # Group by label for a closet-like feel
            grouped = defaultdict(list)
            for path, label, caption, colors_csv in matches:
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
