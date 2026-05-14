import streamlit as st
import joblib
import numpy as np
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Iris Intelligence", page_icon="🌸", layout="wide")

# --- CUSTOM CSS FOR MODERN PEACH THEME ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #FFF5EE 0%, #FFDAB9 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Professional Navbar/Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffffCC;
        border-right: 1px solid #FFCCAC;
    }

    /* Custom Card Style */
    .concept-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(255, 160, 122, 0.2);
        margin-bottom: 20px;
        border: 1px solid #FFDAB9;
    }

    /* Stylish Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #FF8C00, #FFA07A);
        color: white;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        border: none;
        transition: all 0.3s ease;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 140, 0, 0.4);
        color: white;
    }

    /* Hero Text */
    .hero-title {
        color: #D2691E;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
        animation: fadeIn 1.5s;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Prediction Box */
    .prediction-box {
        background: #FFFAF0;
        border-left: 10px solid #FF8C00;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_assets():
    model = joblib.load('iris_model.joblib')
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
    return model, scaler, le

try:
    model, scaler, le = load_assets()
except:
    st.error("Model files not found. Please run the training script first.")

# --- NAVIGATION ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def set_page(page_name):
    st.session_state.page = page_name

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.title("Navigation")
    if st.button("🏠 Home", use_container_width=True): set_page('Home')
    if st.button("📝 Questions/Input", use_container_width=True): set_page('Questions')
    if st.button("🔮 Prediction", use_container_width=True): set_page('Prediction')
    st.divider()
    st.info("Built with Streamlit & SVM")

# --- PAGE 1: HOME ---
if st.session_state.page == 'Home':
    st.markdown('<h1 class="hero-title">Iris Intelligence</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown(f"""
        <div class="concept-card">
            <h3 style="color: #FF8C00;">Welcome to the Premium Iris Classifier</h3>
            <p style="font-size: 1.1rem; color: #555;">
                Experience the power of Support Vector Machines with an elegant, modern interface. 
                Our model analyzes botanical features to predict flower species with high precision.
            </p>
            <hr style="border: 0.5px solid #FFE4B5;">
            <ul>
                <li><strong>Modern UI:</strong> Soft peach gradients and responsive containers.</li>
                <li><strong>Reliable:</strong> Powered by Scikit-Learn SVM.</li>
                <li><strong>Interactive:</strong> Real-time feedback and stylish animations.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Get Started →"):
            set_page('Questions')
            st.rerun()

# --- PAGE 2: QUESTIONS/INPUT ---
elif st.session_state.page == 'Questions':
    st.markdown("<h2 style='text-align: center; color: #D2691E;'>🌸 Flower Characteristics</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="concept-card">', unsafe_allow_html=True)
        
        # Form for input
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                sepal_l = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, help="Length of the sepal")
                sepal_w = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
            
            with col2:
                petal_l = st.number_input("Petal Length (cm)", 1.0, 7.0, 4.3)
                petal_w = st.number_input("Petal Width (cm)", 0.1, 2.5, 1.3)
            
            st.markdown("---")
            
            c1, c2, c3 = st.columns([1, 1, 1])
            with c2:
                submitted = st.form_submit_button("Analyze Data")
                if st.form_submit_button("Reset"):
                    st.rerun()
                    
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            st.session_state.input_data = [sepal_l, sepal_w, petal_l, petal_w]
            set_page('Prediction')
            st.rerun()

# --- PAGE 3: PREDICTION ---
elif st.session_state.page == 'Prediction':
    st.markdown("<h2 style='text-align: center; color: #D2691E;'>🔮 Classification Result</h2>", unsafe_allow_html=True)
    
    if 'input_data' not in st.session_state:
        st.warning("Please enter data in the Questions page first!")
        if st.button("Go to Input"): set_page('Questions'); st.rerun()
    else:
        with st.spinner("Processing through SVM Hyperplane..."):
            time.sleep(1.5) # Simulated delay for premium feel
            
            # Prepare data
            data = np.array([st.session_state.input_data])
            scaled_data = scaler.transform(data)
            
            # Predict
            pred = model.predict(scaled_data)
            prob = model.predict_proba(scaled_data)
            species = le.inverse_transform(pred)[0]
            confidence = np.max(prob) * 100

        # UI Result Card
        st.markdown(f"""
        <div class="prediction-box">
            <h1 style="color: #FF8C00; margin: 0;">{species}</h1>
            <p style="font-size: 1.2rem; color: #666;">Confidence Level: <b>{confidence:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
        st.toast(f"Success! Model identified a {species}", icon='✅')

        # Additional visual data
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(3)
        cols[0].metric("Sepal L/W ratio", round(st.session_state.input_data[0]/st.session_state.input_data[1], 2))
        cols[1].metric("Petal L/W ratio", round(st.session_state.input_data[2]/st.session_state.input_data[3], 2))
        cols[2].metric("Feature Set", "Complete")

        if st.button("New Prediction"):
            del st.session_state.input_data
            set_page('Questions')
            st.rerun()

# --- FOOTER ---
st.markdown("""
    <div style="text-align: center; margin-top: 50px; color: #D2691E; font-size: 0.9rem;">
        <hr style="border-top: 1px solid #FFDAB9;">
        <p>© 2024 Iris Intelligence Pro | Modern ML Series</p>
        <p>Built with ❤️ using Streamlit, Scikit-Learn, and Joblib</p>
    </div>
    """, unsafe_allow_html=True)
