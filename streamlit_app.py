"""
GenZ Datamind — Streamlit Predictor
=====================================
A single-page Streamlit app that runs the same trained model as the main
website's Flask backend (model.py / app.py). Deploy this to Streamlit
Community Cloud so the predictor is reachable from a public URL.

Loads three artifacts from ./models/ (already trained — see
GenZ-Datamind-Backend/model.py for how they were produced):
    best_classifier.pkl          -> predicts addiction_level (Low/Medium/High)
    best_regressor.pkl           -> predicts mental_health_score (0-10)
    addiction_label_encoder.pkl  -> turns the classifier's numeric output
                                     back into "Low" / "Medium" / "High"
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parent / "models"

# Fixed status colors (good/warning/critical) — validated for colorblind
# safety and contrast against a dark surface via the dataviz skill's
# validator (node scripts/validate_palette.js "#0ca30c,#fab219,#d03b3b"
# --mode dark: CVD separation, normal-vision floor, and contrast all pass).
# Each bar also carries its own text label, so color is never the only
# way to tell the risk tiers apart.
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}
RISK_DESCRIPTIONS = {
    "Low": "Usage patterns appear within a healthy range.",
    "Medium": "Elevated usage detected — consider reducing screen time.",
    "High": "Critical addiction risk — significant behavioral impact likely.",
}

CLASS_FEATURES = ["age", "num_platforms_used", "avg_session_minutes",
                   "daily_usage_hours", "night_usage", "screen_time_before_sleep", "gender"]
REG_FEATURES = CLASS_FEATURES + ["purpose"]


@st.cache_resource
def load_models():
    classifier = joblib.load(MODEL_DIR / "best_classifier.pkl")
    regressor = joblib.load(MODEL_DIR / "best_regressor.pkl")
    label_encoder = joblib.load(MODEL_DIR / "addiction_label_encoder.pkl")
    return classifier, regressor, label_encoder


def predict(inputs, classifier, regressor, label_encoder):
    row_class = pd.DataFrame([{k: inputs[k] for k in CLASS_FEATURES}])
    row_reg = pd.DataFrame([{k: inputs[k] for k in REG_FEATURES}])

    pred_class = classifier.predict(row_class)
    addiction_level = label_encoder.inverse_transform(pred_class)[0]

    mental_health_score = float(regressor.predict(row_reg)[0])
    mental_health_score = max(0.0, min(10.0, mental_health_score))

    confidence = None
    if hasattr(classifier.named_steps["model"], "predict_proba"):
        proba = classifier.predict_proba(row_class)[0]
        classes = label_encoder.inverse_transform(range(len(proba)))
        confidence = dict(zip(classes, proba))

    return addiction_level, round(mental_health_score, 1), confidence


def confidence_chart(confidence):
    # Fixed category order (Low -> Medium -> High), not sorted by value —
    # color always maps to the same risk tier regardless of which one wins.
    order = ["Low", "Medium", "High"]
    values = [round(confidence.get(k, 0) * 100, 1) for k in order]
    colors = [RISK_COLORS[k] for k in order]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=order,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Model confidence (%)", showgrid=False,
                    color="#cfc2d6", title_font=dict(color="#cfc2d6")),
        yaxis=dict(title=None, color="#dfe1f6"),
        height=220,
        margin=dict(l=10, r=30, t=10, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(family="Inter, sans-serif", color="#dfe1f6"),
    )
    return fig


DARK_NEON_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: radial-gradient(circle at 50% 0%, #1a1e3d 0%, #0f1321 60%, #0a0d1c 100%);
    color: #dfe1f6;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] { background: #0f1321; }

/* Title / headings — Sora, gradient text like the main site */
h1 {
    font-family: 'Sora', sans-serif !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #ddb7ff, #4cd7f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
h2, h3 { font-family: 'Sora', sans-serif !important; font-weight: 700 !important; }
[data-testid="stCaptionContainer"] {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.05em;
    opacity: 0.7;
}

/* Glass-card containers: the form and the metric tiles */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 11px !important;
    color: #4cd7f6 !important;
}
[data-testid="stMetricValue"] { color: #dfe1f6 !important; }

/* Widget labels */
[data-testid="stWidgetLabel"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #ddb7ff !important;
    opacity: 0.85;
}

/* Inputs / selects / sliders */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] {
    background-color: #1b1f2e !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #dfe1f6 !important;
    border-radius: 8px !important;
}
div[data-baseweb="popover"] li { background-color: #1b1f2e !important; }
[data-testid="stSlider"] [role="slider"] { background-color: #4cd7f6 !important; }
[data-baseweb="slider"] > div > div { background: #4cd7f6 !important; }

/* Primary submit button — glowing pill like the site's CTA */
[data-testid="stFormSubmitButton"] button, .stButton button {
    background: #ddb7ff !important;
    color: #2c0051 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    box-shadow: 0 0 20px rgba(221,183,255,0.35);
    transition: all 0.2s ease;
}
[data-testid="stFormSubmitButton"] button:hover, .stButton button:hover {
    background: #c99bff !important;
    box-shadow: 0 0 28px rgba(221,183,255,0.55);
}

/* Dataframe / table */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

hr { border-color: rgba(255,255,255,0.08) !important; }
</style>
"""


def main():
    st.set_page_config(page_title="GenZ Datamind — Predictor", page_icon="🧠", layout="centered")
    st.markdown(DARK_NEON_CSS, unsafe_allow_html=True)

    st.title("🧠 GenZ Datamind")
    st.caption("SOCIAL INFERENCE ENGINE · Same trained model as the main site")

    classifier, regressor, label_encoder = load_models()

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=13, max_value=27, value=18)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            country = st.text_input("Country", value="Canada")
            num_platforms = st.number_input("Number of platforms used", min_value=1, max_value=10, value=3)
            primary_platform = st.selectbox(
                "Primary platform", ["Snapchat", "TikTok", "Instagram", "YouTube", "Twitter"]
            )
        with col2:
            daily_usage = st.slider("Daily usage (hours)", 1, 12, 5)
            avg_session = st.number_input("Avg session length (minutes)", min_value=1, value=25)
            purpose = st.selectbox(
                "Primary purpose", ["Entertainment", "Education", "Content Creation", "News", "Socializing"]
            )
            night_usage = st.selectbox("Uses social media at night?", ["Yes", "No"])
            screen_before_sleep = st.number_input("Screen time before sleep (minutes)", min_value=0, value=30)

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if submitted:
        inputs = {
            "age": age,
            "gender": gender,
            "country": country,
            "num_platforms_used": num_platforms,
            "primary_platform": primary_platform,
            "daily_usage_hours": daily_usage,
            "avg_session_minutes": avg_session,
            "purpose": purpose,
            "night_usage": 1 if night_usage == "Yes" else 0,
            "screen_time_before_sleep": screen_before_sleep,
        }

        with st.spinner("Running the trained model…"):
            level, score, confidence = predict(inputs, classifier, regressor, label_encoder)

        st.session_state.history.append({
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "addiction_level": level,
            "mental_health_score": score,
            "purpose": purpose,
            **inputs,
        })

        color = RISK_COLORS[level]
        st.markdown(
            f"""
            <div style="border:1px solid {color}55; background: rgba(255,255,255,0.03);
                        backdrop-filter: blur(12px); box-shadow: 0 0 30px {color}22;
                        border-radius:16px; padding:24px; margin-top:8px;">
                <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                            letter-spacing:0.15em; text-transform:uppercase; color:{color};
                            opacity:0.85;">Addiction Risk</div>
                <div style="font-family:'Sora',sans-serif; font-size:34px; font-weight:800;
                            color:{color}; text-shadow:0 0 16px {color}88;">{level} Risk</div>
                <div style="color:#cfc2d6; margin-top:6px;">{RISK_DESCRIPTIONS[level]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Mental Health Score", f"{score} / 10")
        with col_b:
            st.metric("Addiction Level", level)

        if confidence:
            st.markdown("**Model confidence by risk tier**")
            st.plotly_chart(confidence_chart(confidence), use_container_width=True)

    if st.session_state.history:
        st.divider()
        st.subheader("This session's predictions")
        st.caption("Resets when you close or refresh the tab — this app doesn't have a shared database like the main site's History page.")
        df = pd.DataFrame(st.session_state.history).iloc[::-1]
        st.dataframe(
            df[["timestamp", "addiction_level", "mental_health_score", "age", "gender", "purpose"]],
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()


if __name__ == "__main__":
    main()
