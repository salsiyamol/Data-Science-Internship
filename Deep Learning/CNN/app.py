import os
import joblib
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Fashion Predictor", layout="centered")

# -------------------------------------------------------------------------
# 1. LOCAL ASSET LOADING
# -------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "fashion_model_package.joblib")
    
    if not os.path.exists(model_path):
        st.error(f"❌ Critical Error: '{model_path}' not found!")
        st.info("💡 Run 'python train.py' in your terminal first to generate this file.")
        st.stop()
        
    if os.path.getsize(model_path) == 0:
        st.error("❌ Critical Error: The model file is empty (0 KB) and corrupted!")
        st.stop()
        
    return joblib.load(model_path)

# Initialize application assets
try:
    model = load_assets()
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
except Exception as e:
    st.error("### ❌ Failed to load the model")
    st.exception(e)  
    st.stop()

# -------------------------------------------------------------------------
# 2. USER INTERFACE
# -------------------------------------------------------------------------
st.title("👟 Fashion Classification Predictor")
st.write("Upload an image file to predict the type of fashion item.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Processing image and predicting..."):
        try:
            # Match image preprocessing to training specifications (28x28 grayscale)
            img_resized = image.convert("L").resize((28, 28))
            img_array = np.array(img_resized) / 255.0
            
            # Flatten to 1D vector (784 features) since it's a Scikit-Learn MLPClassifier
            img_ready = img_array.reshape(1, -1)
            
            # Predict class label index
            predictions = model.predict(img_ready)
            predicted_class_idx = int(predictions[0])
            
            st.success("### Prediction Result!")
            st.metric(label="Predicted Class", value=class_names[predicted_class_idx])
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")