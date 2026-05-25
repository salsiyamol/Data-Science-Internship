import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
import os
import time
from pathlib import Path
from ultralytics import YOLO
import plotly.express as px
import bcrypt

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(page_title="SafeGuard AI | PPE Detection", page_icon="🏗️", layout="wide")
USER_DB_FILE = "users_db.pkl"

# Load a pre-trained YOLOv8 model (downloads automatically on first run)
# For a production PPE system, you would replace 'yolov8n.pt' with your custom trained 'best.pt'
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt") 

model = load_yolo_model()

# --- USER AUTHENTICATION SYSTEMS (JOBLIB) ---
def load_users():
    if os.path.exists(USER_DB_FILE):
        return joblib.load(USER_DB_FILE)
    return {}

def save_users(users):
    joblib.dump(users, USER_DB_FILE)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

# Initialize Session State
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# --- UI COMPONENTS ---
def login_register_page():
    st.title("🏗️ SafeGuard AI Portal")
    st.subheader("Intelligent PPE & Construction Site Safety Monitoring")
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    users = load_users()
    
    with tab1:
        st.markdown("### Access Your Dashboard")
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", use_container_width=True):
            if login_user in users and check_password(login_pass, users[login_user]):
                st.session_state.logged_in = True
                st.session_state.user_name = login_user
                st.success(f"Welcome back, {login_user}!")
                st.rerun()
            else:
                st.error("Invalid Username or Password.")
                
    with tab2:
        st.markdown("### Create an Account")
        reg_user = st.text_input("Choose a Username", key="reg_user")
        reg_pass = st.text_input("Choose a Password", type="password", key="reg_pass")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("Register Account", use_container_width=True):
            if not reg_user or not reg_pass:
                st.warning("Fields cannot be empty.")
            elif reg_pass != reg_confirm:
                st.error("Passwords do not match.")
            elif reg_user in users:
                st.error("Username already exists.")
            else:
                users[reg_user] = hash_password(reg_pass)
                save_users(users)
                st.success("Registration successful! Please switch to the Login tab.")

def home_page():
    st.title("📊 Real-Time Detection Dashboard")
    st.write(f"Logged in as: **{st.session_state.user_name}**")
    
    # Sidebar control for source selection
    st.sidebar.header("Detection Settings")
    source = st.sidebar.radio("Select Input Source", ["📷 Image Upload", "🎥 Video Upload", "📹 Live Webcam"])
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)

    if source == "📷 Image Upload":
        st.subheader("Image Analytics")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read Image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run Inference
            results = model.predict(image, conf=confidence_threshold)
            annotated_img = results[0].plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Layout Columns
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_rgb, caption="Original Image", use_container_width=True)
            with col2:
                st.image(annotated_img_rgb, caption="Processed Image (Detections)", use_container_width=True)
            
            # Analytics
            parse_and_display_analytics(results)

    elif source == "🎥 Video Upload":
        st.subheader("Video File Analysis")
        uploaded_video = st.file_uploader("Upload a video file...", type=["mp4", "mov", "avi"])
        
        if uploaded_video is not None:
            # Save temporary file to read via OpenCV
            tfile = Path("tmp_video.mp4")
            tfile.write_bytes(uploaded_video.read())
            
            vid_cap = cv2.VideoCapture(str(tfile))
            st_frame = st.empty()
            
            while vid_cap.isOpened():
                ret, frame = vid_cap.read()
                if not ret:
                    break
                
                # Perform Detection
                results = model.predict(frame, conf=confidence_threshold)
                annotated_frame = results[0].plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Stream frame
                st_frame.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            vid_cap.release()
            st.success("Video processing completed.")

    elif source == "📹 Live Webcam":
        st.subheader("Live Webcam Detection Stream")
        st.info("Click 'Start Webcam' to begin simulated real-time inference.")
        
        run_webcam = st.checkbox("Start Webcam Feed")
        if run_webcam:
            # 0 is typically the built-in webcam
            vid_cap = cv2.VideoCapture(0)
            st_frame = st.empty()
            
            # Placeholder for live count metric
            metric_placeholder = st.empty()
            
            while run_webcam:
                ret, frame = vid_cap.read()
                if not ret:
                    st.error("Failed to grab webcam frame.")
                    break
                
                results = model.predict(frame, conf=confidence_threshold)
                annotated_frame = results[0].plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Quick count metrics
                counts = results[0].boxes.cls.tolist()
                metric_placeholder.metric("Total Objects Detected Live", len(counts))
                
                st_frame.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                
            vid_cap.release()

def parse_and_display_analytics(results):
    st.markdown("---")
    st.subheader("📈 Visual Analytics Insights")
    
    # Extract data from predictions
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.info("No items detected based on current confidence filters.")
        return
        
    class_ids = boxes.cls.numpy().astype(int)
    confidences = boxes.conf.numpy()
    class_names = [model.names[i] for i in class_ids]
    
    df = pd.DataFrame({
        "Object Class": class_names,
        "Confidence Score": confidences
    })
    
    # Layout Columns for Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Detection Distribution Count**")
        fig_count = px.histogram(df, x="Object Class", color="Object Class", color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig_count, use_container_width=True)
        
    with c2:
        st.markdown("**Confidence Ranges per Class**")
        fig_box = px.box(df, x="Object Class", y="Confidence Score", color="Object Class", color_discrete_sequence=px.colors.qualitative.Modern)
        st.plotly_chart(fig_box, use_container_width=True)
        
    # Detailed Data Table
    st.markdown("**Detailed Raw Detections Telemetry**")
    st.dataframe(df, use_container_width=True)

def about_page():
    st.title("ℹ️ About SafeGuard AI System")
    st.markdown("""
    ### System Architecture & Capabilities
    This application utilizes cutting-edge Computer Vision algorithms to enhance safety operations and site situational awareness.
    
    * **Core Model:** YOLOv8 Object Detection framework.
    * **UI Component:** Streamlit dynamic web rendering.
    * **Data Serialization & Management:** Joblib is deployed for fast backend binary storage of user registry schemas.
    
    ### Practical Applications
    1.  **PPE Detection:** Verifies compliance regarding Hardhats, High-Visibility Vests, and Safety Goggles.
    2.  **Zone Monitoring:** Keeps tabs on prohibited entry thresholds or unsafe hazard perimeter boundaries.
    ```
    [Video/Camera Stream] -> [YOLO Layer Deep Inference] -> [Streamlit Live Buffer Mapping Framework]
    ```
    """)
    st.info("Developed for Computer Vision & Deep Learning Deployment Modules.")

# --- MAIN CONTROLLER ---
def main():
    if not st.session_state.logged_in:
        login_register_page()
    else:
        # Create a clean Navigation sidebar menu
        st.sidebar.title("🧭 Navigation")
        app_mode = st.sidebar.radio("Go To", ["Home Dashboard", "About Project"])
        
        if st.sidebar.button("Logout 🏃"):
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.rerun()
            
        if app_mode == "Home Dashboard":
            home_page()
        elif app_mode == "About Project":
            about_page()

if __name__ == "__main__":
    main()
