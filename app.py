import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Which Bin?", page_icon="♻️")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

# label (lowercase) -> (bin name, colour, tip)
# Edit the keys to match YOUR model's class names.
BINS = {
    "plastic":   ("Dry Recyclable bin", "#1E88E5", "Rinse it and remove the cap if you can."),
    "metal":     ("Dry Recyclable bin", "#1E88E5", "Crush cans to save space."),
    "glass":     ("Dry Recyclable bin", "#1E88E5", "Wrap broken glass before binning."),
    "paper":     ("Paper bin",          "#43A047", "Keep it dry and flat."),
    "cardboard": ("Paper bin",          "#43A047", "Flatten boxes before binning."),
    "trash":     ("General waste bin",  "#757575", "If unsure, this is the safe choice."),
}
DEFAULT_BIN = ("General waste bin", "#757575", "No rule found for this item, so use the general bin.")
MIN_CONFIDENCE = 0.6


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


st.title("♻️ Which Bin?")
st.write("Take a photo of your waste and I'll tell you which bin to use.")

if not os.path.exists(MODEL_PATH):
    st.error("best.pt was not found. Put it in the same folder as app.py.")
    st.stop()

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

tab_cam, tab_up = st.tabs(["📷 Camera", "📁 Upload"])
with tab_cam:
    cam_photo = st.camera_input("Take a photo of the item")
with tab_up:
    up_photo = st.file_uploader("Or upload a photo", type=["jpg", "jpeg", "png"])

photo = cam_photo or up_photo

if photo is not None:
    img = Image.open(photo).convert("RGB")
    st.image(img, caption="Your photo")

    with st.spinner("Looking at it..."):
        result = model(img, verbose=False)[0]

    probs = result.probs
    label = result.names[probs.top1]
    conf = float(probs.top1conf)

    if conf < MIN_CONFIDENCE:
        st.warning(
            f"I'm not sure (best guess: {label}, {conf:.0%}). "
            "Try a clearer photo on a plain background."
        )
    else:
        bin_name, colour, tip = BINS.get(label.lower(), DEFAULT_BIN)
        st.success(f"Detected: {label} ({conf:.0%} sure)")
        st.markdown(
            f"""
            <div style="background:{colour};padding:20px;border-radius:12px;
                        color:white;text-align:center;font-size:26px;font-weight:bold;">
                Use the {bin_name}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(tip)
        st.session_state.history.append(label)

    # Show the model's top 3 guesses
    with st.expander("See top guesses"):
        for i in probs.top5[:3]:
            st.write(f"{result.names[i]}: {float(probs.data[i]):.0%}")

if st.session_state.history:
    st.divider()
    st.subheader("Items sorted this session")
    st.write(f"Total: {len(st.session_state.history)}")
    counts = {}
    for item in st.session_state.history:
        counts[item] = counts.get(item, 0) + 1
    st.bar_chart(counts)