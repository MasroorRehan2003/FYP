# import streamlit as st
# import pandas as pd
# import requests
# import joblib
# import random
# import os
# from PIL import Image

# # === Load Model ===
# model = joblib.load("weather_clothing_recommender_women.pkl")

# # === Weather API ===
# API_KEY = "4c703f15e3f9220de836884137342d5d"

# def fetch_weather(city):
#     url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
#     response = requests.get(url)
#     if response.status_code != 200:
#         return None
#     data = response.json()
#     weather = {
#         "temperature": data["main"]["temp"],
#         "feels_like": data["main"]["feels_like"],
#         "humidity": data["main"]["humidity"],
#         "wind_speed": data["wind"]["speed"],
#         "weather_condition": data["weather"][0]["main"].lower()
#     }
#     return weather

# # === Streamlit UI ===
# st.set_page_config(page_title="AI Outfit Recommender", layout="wide")
# st.markdown("""
#     <style>
#     .block-container {
#         padding-top: 2rem;
#         padding-bottom: 2rem;
#     }
#     .stImage > img {
#         border-radius: 12px;
#         box-shadow: 0 4px 12px rgba(0,0,0,0.15);
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.title("👗 AI Clothing Recommender for Women")

# # === User Input ===
# city = st.text_input("Enter your city:")

# if st.button("👚 Recommend Outfit"):
#     if not city:
#         st.warning("Please enter your city.")
#     else:
#         weather = fetch_weather(city)
#         if not weather:
#             st.error("Failed to fetch weather. Check city name.")
#         else:
#             # Show weather details
#             st.success("🌤️ Weather fetched successfully!")
#             st.subheader("📡 Current Weather:")
#             st.markdown(f"- **Condition:** {weather['weather_condition'].capitalize()}")
#             st.markdown(f"- **Temperature:** {weather['temperature']}°C (Feels like {weather['feels_like']}°C)")
#             st.markdown(f"- **Humidity:** {weather['humidity']}%")
#             st.markdown(f"- **Wind Speed:** {weather['wind_speed']} m/s")

#             # === Model Prediction ===
#             input_df = pd.DataFrame([{
#                 "temperature": weather["temperature"],
#                 "feels_like": weather["feels_like"],
#                 "humidity": weather["humidity"],
#                 "wind_speed": weather["wind_speed"],
#                 "weather_condition": weather["weather_condition"],
#                 "time_of_day": "morning",   # Placeholder defaults
#                 "season": "summer",
#                 "occasion": "casual"
#             }])
#             prediction = model.predict(input_df)[0]
#             pred_top, pred_bottom, pred_outer = prediction

#             # === Display Function ===
#             def display_image(pred_label, label_name, col=None):
#                 folder_path = os.path.join("clothing_images", pred_label.lower())
#                 if os.path.exists(folder_path) and os.path.isdir(folder_path):
#                     image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
#                     if image_files:
#                         chosen_image = random.choice(image_files)
#                         img_path = os.path.join(folder_path, chosen_image)
#                         img = Image.open(img_path).resize((200, 250))
#                         with (col if col else st):
#                             st.image(img, caption=f"{label_name}: {pred_label}", use_container_width=False)
#                     else:
#                         with (col if col else st):
#                             st.warning(f"{label_name}: {pred_label} (No image found)")
#                 else:
#                     with (col if col else st):
#                         st.error(f"{label_name}: {pred_label} (Folder not found)")

#             # === Display Outfit ===
#             st.markdown("### 👚 Your Recommended Outfit")
#             st.markdown("---")
#             col1, col2, col3 = st.columns(3)

#             display_image(pred_top, "Top", col1)
#             display_image(pred_bottom, "Bottom", col2)
#             if pred_outer.lower() != "none":
#                 display_image(pred_outer, "Outerwear", col3)
#             else:
#                 col3.markdown("**Outerwear:** None needed")

#             st.markdown("---")
#             st.caption("🎲 Tip: Click 'Recommend Outfit' again to see a new variation!")
# =========================
# AI Outfit Recommender (Women) — Streamlit
# =========================

# ---- Force Transformers to avoid TensorFlow/Flax (Windows + Py3.13 friendly)
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---- Standard libs
import importlib
import random
from pathlib import Path

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
# (older sklearn pickles refer to a removed private class)
# =========================
try:
    ct = importlib.import_module("sklearn.compose._column_transformer")
    if not hasattr(ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            """Compat placeholder for older scikit-learn pickles."""
            pass
        ct._RemainderColsList = _RemainderColsList
except Exception:
    # Non-fatal; continue. If version already matches, this isn't used.
    pass

# =========================
# Load weather→clothing model
# =========================
MODEL_PATH = Path("weather_clothing_recommender_women.pkl")
if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH}`. Place it next to app1.py.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH.as_posix())
except Exception as e:
    st.error(
        "Failed to load the scikit-learn model.\n\n"
        f"**Details:** {type(e).__name__}: {e}\n\n"
        "Tip: If this persists, the model was saved with an older sklearn. "
        "Either use the shim above (already added), or re-save the model in the current environment."
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
        # Fallback to public model (requires internet on first run)
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

def generate_caption(image_path: str) -> str:
    if processor is None or blip_model is None:
        return "caption unavailable"
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(blip_model.device)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

# =========================
# Weather API
# =========================
API_KEY = "4c703f15e3f9220de836884137342d5d"  # replace with st.secrets if you want

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
# Session state
# =========================
if "predictions" not in st.session_state:
    st.session_state.predictions = {"top": None, "bottom": None, "outer": None}
if "liked" not in st.session_state:
    st.session_state.liked = None

# =========================
# Inputs
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

            # Model expects a DataFrame with these features:
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
# Shuffle buttons
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
# Display outfit
# =========================
if preds["top"]:
    st.markdown("### 🧥 Your AI-Picked Outfit")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

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
        caption = generate_caption(chosen.as_posix())
        img = Image.open(chosen).resize((200, 250))
        col.image(img, caption=f"{label}: {caption}", use_column_width=False)

    show_image_with_caption(preds["top"], "Top", col1)
    show_image_with_caption(preds["bottom"], "Bottom", col2)
    if preds["outer"].lower() != "none":
        show_image_with_caption(preds["outer"], "Outerwear", col3)
    else:
        col3.markdown("**Outerwear:** None needed")

    st.markdown("---")

    # ---- Like/Dislike
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
