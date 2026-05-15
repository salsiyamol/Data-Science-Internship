import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import time
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Disney Enchanted Insights",
    page_icon="✨",
    layout="wide",
)

# --- 2. PREMIUM MAGIC CSS (Soft Pink & Lavender Theme) ---
def apply_custom_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Quicksand:wght@300;500;700&display=swap');

    /* Global Transitions and Typography */
    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
        color: #4B0082; /* Deep violet for readability */
    }

    /* Magical Animated Heading */
    .magic-title {
        font-family: 'Cinzel', serif;
        background: linear-gradient(90deg, #FFB6C1, #E6E6FA, #FFB6C1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        filter: drop-shadow(0 0 8px rgba(230, 230, 250, 0.8));
        animation: glow 3s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 5px #FFB6C1); }
        to { filter: drop-shadow(0 0 15px #9370DB); }
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 35px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(147, 112, 219, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 40px 0 rgba(255, 182, 193, 0.4);
    }

    /* Elegant Sidebar - Lavender Pink Gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255, 240, 245, 0.95) 0%, rgba(230, 230, 250, 0.95) 100%);
        border-right: 2px solid #FFB6C1;
    }

    /* Soft Premium Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #FFB6C1 0%, #E6E6FA 100%);
        color: #4B0082;
        border: none;
        padding: 12px 30px;
        border-radius: 50px;
        font-weight: 700;
        transition: 0.4s;
        width: 100%;
        box-shadow: 0 4px 15px rgba(147, 112, 219, 0.2);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #E6E6FA 0%, #FFB6C1 100%);
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(255, 182, 193, 0.5);
    }

    /* Glowing Result Card */
    .result-card {
        background: white;
        border: 2px solid #E6E6FA;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 20px rgba(230, 230, 250, 1);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { border-color: #E6E6FA; }
        50% { border-color: #FFB6C1; }
        100% { border-color: #E6E6FA; }
    }
    </style>
    """, unsafe_allow_html=True)

def set_bg(url):
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.3)), url("{url}");
        background-attachment: fixed;
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOADING ASSETS ---
@st.cache_resource
def load_magic_assets():
    try:
        model = joblib.load('disney_model.joblib')
        encoders = joblib.load('encoders.joblib')
        df = pd.read_csv('disney_movies.csv')
        return model, encoders, df
    except:
        return None, None, None

model, encoders, df = load_magic_assets()
apply_custom_styles()

# --- 4. MULTIPAGE NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center; color:#9370DB; font-family:Cinzel;'>Castle Menu</h2>", unsafe_allow_html=True)
    page = st.radio("✨ Choose Your Path", ["The Entrance Hall", "The Prediction Chamber", "The Royal Treasury"])
    st.markdown("---")
    st.info("Where Magic meets Machine Learning.")

# --- PAGE 1: HOME (THE ENTRANCE HALL) ---
if page == "The Entrance Hall":
    set_bg("https://images.unsplash.com/photo-1518709268805-4e9042af9f23?q=80&w=2069") # Magical Forest/Castle
    st.markdown('<h1 class="magic-title">Disney Enchanted Insights</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>A Tale of Data and Dreams</h3>
            <p style='font-size:1.1rem;'>Welcome to a premium predictive portal. 
            By weaving together decades of cinematic history with advanced AI, we reveal the success 
            hidden within the numbers of Disney's masterpieces.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card" style="text-align:center;"><h4>System Oracle</h4><p style="color:#9370DB; font-weight:bold;">Magic is Active ✨</p></div>', unsafe_allow_html=True)

    st.markdown("### ✨ Journey Deeper")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="glass-card"><b>🔮 Predictions</b><br>Forecast success for any upcoming feature.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="glass-card"><b>📊 Visualizations</b><br>See the growth of the kingdom.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="glass-card"><b>🕰️ History</b><br>Decades of Disney legacy analyzed.</div>', unsafe_allow_html=True)

# --- PAGE 2: QUESTIONS & PREDICTION ---
elif page == "The Prediction Chamber":
    set_bg("https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?q=80&w=2071") # Dreamy Sky/Stars
    st.markdown('<h1 class="magic-title">Consult the Mirror</h1>', unsafe_allow_html=True)
    
    if df is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            genre = st.selectbox("🎬 Select Movie Genre", sorted(df['genre'].dropna().unique()))
            rating = st.selectbox("🎭 MPAA Audience Rating", sorted(df['mpaa_rating'].dropna().unique()))
        with col_b:
            year = st.slider("📅 Year of Release", 2024, 2030, 2025)
            month = st.select_slider("🌙 Month of Release", options=range(1, 13), 
                                     format_func=lambda x: datetime(2000, x, 1).strftime('%B'))
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("✨ CAST PREDICTION SPELL ✨"):
            with st.spinner("Consulting the mystical data spirits..."):
                time.sleep(1.5)
                
                # Preprocessing
                gen_enc = encoders['le_genre'].transform([genre])[0]
                rat_enc = encoders['le_rating'].transform([rating])[0]
                
                input_df = pd.DataFrame({
                    'genre_encoded': [gen_enc],
                    'mpaa_rating_encoded': [rat_enc],
                    'month': [month],
                    'year': [year]
                })
                
                # Result
                res = model.predict(input_df)[0]
                
                st.balloons()
                st.toast("The prophecy has been revealed!", icon='🪄')
                
                st.markdown(f"""
                <div class="result-card">
                    <h2 style="color: #9370DB;">Forecasted Inflation Adjusted Gross</h2>
                    <h1 style="font-size: 4rem; color: #4B0082;">${res:,.2f}</h1>
                    <p style="opacity:0.7;">Calculated using cinematic regression magic.</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error("Assets (Model/CSV) not found. Check your directory.")

# --- PAGE 3: ROYAL TREASURY (EXTRA FEATURE: INTERACTIVE DASHBOARD) ---
elif page == "The Royal Treasury":
    set_bg("https://images.unsplash.com/photo-1464802686167-b939a6910659?q=80&w=2040") # Pastel Clouds
    st.markdown('<h1 class="magic-title">The Royal Treasury</h1>', unsafe_allow_html=True)
    
    if df is not None:
        # EXTRA FEATURE: Statistics Cards & Dashboard
        st.markdown("### 💎 Kingdom Analytics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Movies in Library", f"{len(df)}")
        k2.metric("Most Profitable Genre", df.groupby('genre')['total_gross'].mean().idxmax())
        k3.metric("Data Range", "1937 - 2016")
        k4.metric("Avg Gross", "$100M+")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Historical Success by Genre")
        fig = px.bar(df.groupby('genre')['inflation_adjusted_gross'].mean().reset_index(), 
                     x='genre', y='inflation_adjusted_gross', 
                     color_discrete_sequence=['#FFB6C1'],
                     template="plotly_white")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style="margin-top: 50px; text-align: center; border-top: 1px solid #E6E6FA; padding-top: 20px;">
    <p style="color: #9370DB; font-size: 14px;">✨ Powered by Streamlit, Joblib & Machine Learning ✨</p>
    <p style="opacity: 0.4; font-size: 10px;">Fantasy Theme Inspired by Disney Cinematic Universe</p>
</div>
""", unsafe_allow_html=True)