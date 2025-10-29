# --- 1. CRITICAL SETUP ---
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 2. REGULAR IMPORTS ---
import streamlit as st
import tensorflow as tf
import segmentation_models as sm
import cv2
import numpy as np
from PIL import Image
import io

# --- 3. MODEL AND PREPROCESSING FUNCTIONS ---
BACKBONE = 'resnet34'
MODEL_PATH = 'best_model_final.keras'

@st.cache_resource
def load_model_and_tools():
    """Loads the U-Net model and preprocessing function."""
    model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    model.load_weights(MODEL_PATH)
    preprocess_input = sm.get_preprocessing(BACKBONE)
    return model, preprocess_input

@st.cache_data
def preprocess_image(image_bytes, _preprocess_input_fn, size=(256, 256)):
    """Takes the raw bytes of an uploaded image and prepares it for the model."""
    image_np = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(image_cv, size)
    processed_image = _preprocess_input_fn(original_image)
    input_tensor = tf.expand_dims(processed_image, axis=0)
    return original_image, input_tensor

def run_prediction(model, input_tensor):
    """Runs the model prediction."""
    return model.predict(input_tensor, verbose=0)

def hex_to_rgb(hex_color):
    """Converts a hex color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def convert_image_to_bytes(image_array_rgb):
    """Converts a NumPy array (RGB) to bytes for download."""
    image_pil = Image.fromarray(image_array_rgb.astype('uint8'))
    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    return buf.getvalue()

def create_binary_mask(prediction_tensor, threshold):
    """Creates a black and white binary mask from the model's prediction."""
    mask_binary = (tf.squeeze(prediction_tensor) > threshold).numpy().astype(np.uint8)
    mask_img_rgb = np.stack([mask_binary * 255]*3, axis=-1)
    return mask_img_rgb

def create_overlay(original_image, prediction_tensor, threshold, alpha, color_rgb):
    """Generates a color overlay for the predicted spill."""
    color_mask = np.zeros_like(original_image)
    pred_mask_binary = (tf.squeeze(prediction_tensor) > threshold).numpy().astype('uint8')
    color_mask[pred_mask_binary == 1] = color_rgb
    overlay = cv2.addWeighted(original_image, 1, color_mask, alpha, 0)
    return overlay

# --- 4. STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="Oil Rig Monitor", page_icon="🛢️", layout="wide")

# --- Enhanced CSS Styling ---
st.markdown(f"""
<style>
/* Main app styling (targets the root Streamlit container) */
.stApp {{
    background: linear-gradient(rgba(14, 17, 23, 0.8), rgba(14, 17, 23, 0.8)), /* Dark overlay for readability */
                url(https://i.postimg.cc/XqBrGW1P/Oil-Rig.webp); /* Your MAIN background image */
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed; /* Keep background fixed during scroll */
    color: #FAFAFA;
}}

/* Title styling */
h1 {{
    color: #FFC300;
    text-align: center;
    padding-bottom: 20px;
    border-bottom: 2px solid #FFC300;
    text-shadow: 2px 2px 4px #000; /* Add shadow for better visibility over background */
}}

/* Sidebar styling */
[data-testid="stSidebar"] {{
    background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)),
                url(https://i.postimg.cc/pVsCKJGQ/istockphoto-902969572-612x612.jpg); /* SIDEBAR background image */
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    border-right: 2px solid #FFC300;
    width: 350px !important;
}}

[data-testid="stSidebar"] h2, /* Sidebar Header */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stColorPicker label,
[data-testid="stSidebar"] .stText,
[data-testid="stSidebar"] .stMarkdown {{ /* Target all text elements in sidebar */
    color: #FAFAFA !important; /* Ensure high contrast */
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8); /* Add text shadow for readability */
}}

/* Main content area padding and background */
.main .block-container {{
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    background-color: rgba(14, 17, 23, 0.6); /* Slightly darker semi-transparent background for content */
    border-radius: 10px; /* Optional: Rounded corners for the content block */
    margin: 1rem; /* Add some margin around the content block */
}}

/* Image styling */
.stImage img {{
    border: 2px solid #FFC300;
    border-radius: 8px;
    max-width: 100%;
    height: auto;
    margin-left: auto;
    margin-right: auto;
    display: block;
}}

/* Button styling */
.stButton>button {{
    background-color: #FFC300;
    color: #0E1117;
    font-weight: bold;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    width: 100%;
    margin-top: 10px;
}}
.stButton>button:hover {{
    background-color: #E6B000;
    color: #0E1117;
}}

/* File uploader styling */
[data-testid="stFileUploader"] label {{
    color: #FFC300;
    font-weight: bold;
    font-size: 1.1em;
}}
[data-testid="stFileUploader"] section {{
    border: 3px dashed #FFC300;
    background-color: rgba(30, 30, 30, 0.8);
    padding: 20px;
}}

/* Subheaders in main content */
.main h2, .main h3 {{
    color: #FFC300;
    border-bottom: 1px solid #444;
    padding-bottom: 5px;
    text-shadow: 1px 1px 2px #000; /* Shadow for readability */
}}

/* Info box styling */
.stAlert {{
    background-color: rgba(30, 30, 30, 0.8); /* Darker info box */
    border-left: 5px solid #FFC300; /* Gold accent line */
}}
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("🛢️ AI-Powered Oil Spill Monitor")

# --- 5. SIDEBAR FOR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Analysis Controls")
    st.write("Adjust settings:")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    overlay_alpha = st.slider("Overlay Transparency", 0.1, 1.0, 0.5, 0.05)
    overlay_color_hex = st.color_picker("Spill Highlight Color", "#FF0000")
    overlay_color_rgb = hex_to_rgb(overlay_color_hex)
    st.divider()
    st.caption("Model: U-Net w/ ResNet34")
    st.caption("Status: Online")

# --- 6. MAIN PAGE LOGIC ---
with st.spinner('Powering up the AI model...'):
    model, preprocess_input = load_model_and_tools()
load_col1, load_col2 = st.columns([1, 15]) # Adjusted column ratio for loading message
with load_col1: st.success('✅')
with load_col2: st.write('**AI Model Online.** Ready for analysis!')

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    try:
        original_image, input_tensor = preprocess_image(image_bytes, preprocess_input)
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.stop()

    with st.spinner('Analyzing image for oil spills...'):
        prediction = run_prediction(model, input_tensor)

    overlay_image = create_overlay(
        original_image, prediction, confidence_threshold, overlay_alpha, overlay_color_rgb
    )
    binary_mask_image = create_binary_mask(prediction, confidence_threshold)

    st.header("Analysis Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_image, caption="Uploaded Satellite Image", use_container_width=True)
    with col2:
        st.subheader("Detected Spill Overlay")
        st.image(overlay_image, caption="Highlighted Spill Area", use_container_width=True)
        st.download_button(
            label="⬇️ Download Overlay Image",
            data=convert_image_to_bytes(overlay_image),
            file_name=f"oil_spill_overlay_T{confidence_threshold:.2f}.png",
            mime="image/png"
        )

    st.divider()
    st.subheader("Generated Segmentation Mask")
    mask_col1, mask_col2, mask_col3 = st.columns([1,2,1])
    with mask_col2:
        st.image(binary_mask_image, caption="Predicted Spill Mask (black = Spill)", use_container_width=True)
        st.download_button(
            label="⬇️ Download Mask Image",
            data=convert_image_to_bytes(binary_mask_image),
            file_name=f"oil_spill_mask_T{confidence_threshold:.2f}.png",
            mime="image/png"
        )
else:
    st.info("Upload an image file to start the oil spill detection process.")