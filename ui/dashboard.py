"""
ui/dashboard.py

Streamlit dashboard for the Stress Detection & Recommender system.

Features:
  - Text input for real-time stress scoring
  - Attention heatmap (highlights high-attention tokens in red)
  - Plotly line chart showing Stress Velocity and Adaptive Threshold over time
  - Recommendation panel (all 3 layers)
"""

from __future__ import annotations

import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stress Detection System",
    page_icon="🧠",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "profile" not in st.session_state:
    from src.temporal.temporal_profile import TemporalStressProfile
    st.session_state.profile = TemporalStressProfile()

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []  # list of {timestamp, score, text}

if "inference_engine" not in st.session_state:
    st.session_state.inference_engine = None

if "recommender" not in st.session_state:
    from src.recommender.recommendation_engine import RecommendationEngine
    st.session_state.recommender = RecommendationEngine()


# ── Lazy model loader ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_engine():
    checkpoint = REPO_ROOT / "models" / "checkpoints" / "best_model.pt"
    if not checkpoint.exists():
        return None
    from src.models.inference import StressInferenceEngine
    return StressInferenceEngine(checkpoint)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _token_heatmap_html(tokens: list[str], weights: list[float]) -> str:
    """Render tokens as inline HTML with background colour intensity ∝ weight."""
    if not tokens or not weights:
        return "<p><em>No attention data available.</em></p>"

    max_w = max(weights) if weights else 1.0
    parts: list[str] = []
    for token, weight in zip(tokens, weights):
        norm = weight / max_w if max_w > 0 else 0
        # Red channel only: rgba(255, 0, 0, alpha)
        alpha = round(0.15 + 0.85 * norm, 3)
        label = token.replace("##", "")    # strip BERT sub-word prefix
        style = (
            f"background-color: rgba(220, 50, 50, {alpha}); "
            "padding: 2px 4px; margin: 2px; border-radius: 3px; "
            "font-size: 0.9rem;"
        )
        parts.append(f'<span style="{style}">{label}</span>')
    return "<div style='line-height: 2rem;'>" + " ".join(parts) + "</div>"


def _stress_velocity_chart(profile) -> go.Figure:
    """Build a Plotly figure showing stress scores, velocity, and threshold."""
    summary = profile.summary()
    scores = summary["history_scores"]
    timestamps = summary["history_timestamps"]
    n = len(scores)

    if n == 0:
        fig = go.Figure()
        fig.update_layout(title="No data yet – submit some text first.")
        return fig

    x = list(range(n))

    # Compute per-step adaptive threshold (recalculated at each step)
    thresholds: list[float] = []
    import numpy as np
    for i in range(n):
        sub = scores[: i + 1]
        if len(sub) < 2:
            thresholds.append(0.7)
        else:
            t = float(np.mean(sub)) + 1.5 * float(np.std(sub, ddof=1))
            thresholds.append(min(t, 1.0))

    # Stress velocity (computed at each step)
    w = profile.velocity_window
    ma = []
    for i in range(n):
        start = max(0, i - w + 1)
        ma.append(float(np.mean(scores[start: i + 1])))
    velocity = [0.0] + [ma[i] - ma[i - 1] for i in range(1, n)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=scores, mode="lines+markers", name="Stress Score",
                             line=dict(color="#e63946", width=2)))
    fig.add_trace(go.Scatter(x=x, y=thresholds, mode="lines", name="Adaptive Threshold",
                             line=dict(color="#f4a261", dash="dash", width=2)))
    fig.add_trace(go.Bar(x=x, y=velocity, name="Stress Velocity (Vs)",
                         marker_color=["#2a9d8f" if v >= 0 else "#457b9d" for v in velocity],
                         opacity=0.6, yaxis="y2"))

    fig.update_layout(
        title="📈 Temporal Stress Profile",
        xaxis_title="Post #",
        yaxis=dict(title="Stress Score / Threshold", range=[-0.05, 1.1]),
        yaxis2=dict(title="Velocity (Vs)", overlaying="y", side="right",
                    range=[-0.5, 0.5], showgrid=False),
        legend=dict(orientation="h", y=-0.2),
        height=400,
        hovermode="x unified",
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

st.title("🧠 Stress Detection & Recommender System")
st.markdown(
    "Powered by **Multichannel CNN + Attention** · Temporal Stress Modelling · Safety-First Recommendations"
)

col_left, col_right = st.columns([1, 1])

# ── Left: Input + Heatmap ─────────────────────────────────────────────────────
with col_left:
    st.subheader("📝 Enter Your Text")
    user_text = st.text_area(
        "Type or paste a social-media post, journal entry, etc.",
        height=160,
        placeholder="e.g. I've been so stressed about my exams and I can't sleep at all …",
    )

    analyse_btn = st.button("🔍 Analyse Stress", type="primary", use_container_width=True)

    if analyse_btn and user_text.strip():
        engine = load_engine()
        with st.spinner("Running inference …"):
            if engine is None:
                # Demo mode: generate a pseudo-random score so the UI is usable
                import random, hashlib
                h = int(hashlib.md5(user_text.encode()).hexdigest(), 16)
                random.seed(h)
                stress_score = round(random.uniform(0.25, 0.95), 3)
                tokens = user_text.split()[:20]
                attn_weights = [random.random() for _ in tokens]
                st.warning(
                    "⚠️ Model checkpoint not found. Showing **demo mode** with synthetic scores. "
                    "Run `src/training/train.py` to enable real inference."
                )
            else:
                result = engine.predict(user_text)
                stress_score = result["stress_score"]
                tokens = result["tokens"]
                attn_weights = result["attn_weights"]

        # Update temporal profile
        from src.temporal.temporal_profile import StressEvent
        st.session_state.profile.add_event(
            StressEvent(timestamp=time.time(), score=stress_score, text_snippet=user_text[:80])
        )
        st.session_state.history.append(
            {"timestamp": time.time(), "score": stress_score, "text": user_text[:80]}
        )

        # ── Score display ───────────────────────────────────────────────────
        st.markdown("---")
        level_color = (
            "🟢" if stress_score < 0.4 else ("🟡" if stress_score < 0.65 else "🔴")
        )
        st.metric("Stress Score", f"{stress_score:.2%}", delta=None)
        st.markdown(f"{level_color} **Level**: {'Low' if stress_score < 0.4 else ('Moderate' if stress_score < 0.65 else 'High')}")

        # ── Attention Heatmap ───────────────────────────────────────────────
        st.markdown("#### 🔥 Attention Heatmap")
        st.markdown(
            "Words highlighted in darker red received higher attention from the model.",
            unsafe_allow_html=False,
        )
        html = _token_heatmap_html(tokens, attn_weights)
        st.markdown(html, unsafe_allow_html=True)

        # ── Recommendations ─────────────────────────────────────────────────
        profile = st.session_state.profile
        reco = st.session_state.recommender.recommend(
            text=user_text,
            stress_score=stress_score,
            is_high_volatility=profile.is_high_volatility(),
            should_intervene=profile.should_intervene(),
        )

        st.markdown("---")
        st.markdown("#### 💡 Recommendations")

        if reco.crisis_detected:
            st.error(reco.explanation)
            for resource in reco.emergency_resources:
                st.markdown(resource)
        else:
            st.info(reco.explanation)
            for rec in reco.recommendations:
                st.markdown(f"- {rec}")


# ── Right: Temporal Chart ─────────────────────────────────────────────────────
with col_right:
    st.subheader("📊 Temporal Stress Profile")
    profile = st.session_state.profile
    summary = profile.summary()

    if summary["n_events"] > 0:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Current Score", f"{summary['current_score']:.2%}")
        col_m2.metric(
            "Stress Velocity (Vs)",
            f"{summary['stress_velocity']:+.3f}",
            delta=f"{summary['stress_velocity']:+.3f}",
            delta_color="inverse",
        )
        col_m3.metric("Adaptive Threshold", f"{summary['adaptive_threshold']:.2%}")

    fig = _stress_velocity_chart(profile)
    st.plotly_chart(fig, use_container_width=True)

    if summary["n_events"] > 0:
        flags: list[str] = []
        if summary["should_intervene"]:
            flags.append("⚠️ **Intervention recommended** – current score exceeds adaptive threshold.")
        if summary["is_high_volatility"]:
            flags.append("📉 **High volatility detected** – stress is changing rapidly.")
        if not flags:
            flags.append("✅ Stress is within normal range.")
        for flag in flags:
            st.markdown(flag)

    st.markdown("---")
    st.caption(
        "ℹ️ Adaptive Threshold = μ + 1.5σ over your history. "
        "Stress Velocity (Vs) = change in moving-average score between last two posts."
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🔒 **Privacy notice**: No data is stored permanently. All analysis runs locally in your session."
)
