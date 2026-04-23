import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeliveryIQ · Time Predictor",
    page_icon="🚀",
    layout="centered",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg:         #0d0f14;
    --surface:    #161b25;
    --card:       #1e2535;
    --border:     #2a3348;
    --accent:     #00e5a0;
    --accent2:    #4f8dff;
    --warn:       #ff6b6b;
    --text:       #e8ecf4;
    --muted:      #7a8499;
    --mono:       'Space Mono', monospace;
    --sans:       'Sora', sans-serif;
}

/* ── Global ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main container ── */
.block-container {
    padding: 2.5rem 2rem 4rem !important;
    max-width: 760px !important;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.8rem;
}
.hero-badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 4px 14px;
    border-radius: 100px;
    margin-bottom: 1.1rem;
    text-transform: uppercase;
}
.hero h1 {
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 0.5rem !important;
    background: linear-gradient(135deg, #e8ecf4 30%, var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
}

/* ── Section label ── */
.section-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent2);
    margin: 2rem 0 0.75rem;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, #0f1f17 0%, #0a1628 100%);
    border: 1.5px solid var(--accent);
    border-radius: 18px;
    padding: 2.2rem;
    text-align: center;
    margin-top: 1.6rem;
    box-shadow: 0 0 40px rgba(0,229,160,0.12);
}
.result-box .label {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.result-box .time {
    font-size: 4rem;
    font-weight: 700;
    font-family: var(--mono);
    color: var(--accent);
    line-height: 1;
}
.result-box .unit {
    font-size: 1.1rem;
    color: var(--muted);
    margin-left: 6px;
}
.result-box .range {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.6rem;
    font-family: var(--mono);
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSlider"] {
    background: transparent !important;
}
label { color: var(--text) !important; font-size: 0.88rem !important; }
.stSlider > div > div > div { background: var(--accent2) !important; }

/* ── Button ── */
div.stButton > button {
    width: 100%;
    background: var(--accent) !important;
    color: #0d0f14 !important;
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    margin-top: 0.5rem;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85 !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Info pills ── */
.pill-row {
    display: flex; gap: 10px; flex-wrap: wrap; margin-top: 1rem;
}
.pill {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 5px 14px;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
}
.pill span { color: var(--accent2); margin-right: 4px; }
</style>
""", unsafe_allow_html=True)


# ─── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Try current directory first, then common paths
    search_paths = [
        "rf_model_pkl",
        "rf_model.pkl",
        os.path.join(os.path.dirname(__file__), "rf_model_pkl"),
        os.path.join(os.path.dirname(__file__), "rf_model.pkl"),
    ]
    for path in search_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

model = load_model()


# ─── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🚀 Random Forest · Regressor</div>
    <h1>DeliveryIQ</h1>
    <p>Predict delivery time in minutes from route & conditions</p>
</div>
""", unsafe_allow_html=True)


# ─── Model not found message ─────────────────────────────────────────────────────
if model is None:
    st.error(
        "⚠️ **Model file not found.** Place `rf_model_pkl` (or `rf_model.pkl`) "
        "in the same directory as this script and restart."
    )
    st.stop()


# ─── Model info pills ────────────────────────────────────────────────────────────
st.markdown("""
<div class="pill-row">
    <div class="pill"><span>■</span> 100 estimators</div>
    <div class="pill"><span>■</span> 14 features</div>
    <div class="pill"><span>■</span> Regressor</div>
    <div class="pill"><span>■</span> sklearn RF</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ─── Input form ─────────────────────────────────────────────────────────────────

# ── Section 1: Route Details ──
st.markdown('<div class="section-label">01 · Route Details</div>', unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        distance = st.slider("📍 Distance (km)", min_value=0.5, max_value=50.0, value=8.0, step=0.5)
    with col2:
        prep_time = st.slider("🍳 Preparation Time (min)", min_value=1, max_value=60, value=15, step=1)

# ── Section 2: Courier ──
st.markdown('<div class="section-label">02 · Courier</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    experience = st.slider("🎖 Experience (years)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
with col4:
    vehicle = st.selectbox("🏍 Vehicle Type", ["Bike", "Scooter", "Car"])

# ── Section 3: Conditions ──
st.markdown('<div class="section-label">03 · Conditions</div>', unsafe_allow_html=True)
col5, col6, col7 = st.columns(3)
with col5:
    weather = st.selectbox("🌦 Weather", ["Clear", "Foggy", "Rainy", "Snowy", "Windy"])
with col6:
    traffic = st.selectbox("🚦 Traffic Level", ["Low", "Medium", "High"])
with col7:
    time_of_day = st.selectbox("🕐 Time of Day", ["Morning", "Afternoon", "Evening", "Night"])


# ─── Build feature vector ────────────────────────────────────────────────────────
def build_features():
    features = {
        "Distance_km":            distance,
        "Preparation_Time_min":   prep_time,
        "Courier_Experience_yrs": experience,
        # Weather one-hot (base = Clear)
        "Weather_Foggy":    1 if weather == "Foggy"  else 0,
        "Weather_Rainy":    1 if weather == "Rainy"  else 0,
        "Weather_Snowy":    1 if weather == "Snowy"  else 0,
        "Weather_Windy":    1 if weather == "Windy"  else 0,
        # Traffic one-hot (base = High)
        "Traffic_Level_Low":    1 if traffic == "Low"    else 0,
        "Traffic_Level_Medium": 1 if traffic == "Medium" else 0,
        # Time of day one-hot (base = Afternoon)
        "Time_of_Day_Evening": 1 if time_of_day == "Evening" else 0,
        "Time_of_Day_Morning": 1 if time_of_day == "Morning" else 0,
        "Time_of_Day_Night":   1 if time_of_day == "Night"   else 0,
        # Vehicle one-hot (base = Bike)
        "Vehicle_Type_Car":     1 if vehicle == "Car"     else 0,
        "Vehicle_Type_Scooter": 1 if vehicle == "Scooter" else 0,
    }
    return pd.DataFrame([features])


# ─── Predict button ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("⚡  Predict Delivery Time")

if predict_clicked:
    X = build_features()
    prediction = model.predict(X)[0]

    # Per-tree predictions for confidence range
    tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
    low  = np.percentile(tree_preds, 10)
    high = np.percentile(tree_preds, 90)

    mins = int(round(prediction))
    low_m  = int(round(low))
    high_m = int(round(high))

    # Colour-code urgency
    if mins <= 20:
        time_color = "#00e5a0"
        badge = "⚡ Express"
    elif mins <= 40:
        time_color = "#4f8dff"
        badge = "🕐 Standard"
    else:
        time_color = "#ff6b6b"
        badge = "⚠️ Slow"

    st.markdown(f"""
    <div class="result-box">
        <div class="label">Estimated Delivery Time</div>
        <div class="time" style="color:{time_color};">{mins}<span class="unit">min</span></div>
        <div class="range">80% confidence interval: {low_m} – {high_m} min</div>
        <div style="margin-top:0.8rem;font-family:var(--mono);font-size:0.78rem;color:{time_color};">{badge}</div>
    </div>
    """, unsafe_allow_html=True)

    # Feature summary table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Parameter": ["Distance", "Prep Time", "Experience", "Weather", "Traffic", "Time of Day", "Vehicle"],
        "Value":     [f"{distance} km", f"{prep_time} min", f"{experience} yrs",
                      weather, traffic, time_of_day, vehicle],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)


# ─── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;font-family:\'Space Mono\',monospace;'
    'font-size:0.68rem;color:#3a4255;letter-spacing:0.1em;">'
    'DELIVERYIQ · POWERED BY RANDOM FOREST · BUILT WITH STREAMLIT</p>',
    unsafe_allow_html=True,
)