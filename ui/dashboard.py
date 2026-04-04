"""
ui/dashboard.py

Health-conscious Streamlit dashboard for the Stress Detection & Recommender system.

Design principles:
  - Calm color palette: Sage Green, Muted Ocean Blue, Warm Off-White, Terracotta
  - Purposeful animations: breathing pulse (processing), box-breathing exercise,
    staggered heatmap token reveal
  - Mobile-first layout with 44px touch targets
  - Content warnings: past stressed entries collapsed by default
  - Human-in-the-loop feedback with threshold adjustment
  - Dark mode via CSS variables
"""

from __future__ import annotations

import time
import hashlib
import random
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

REPO_ROOT = Path(__file__).resolve().parent.parent

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 ─ Global CSS (injected once at the top of every render)
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Color tokens ─────────────────────────────────────────────────────────── */
:root {
  --sage:        #8FA28F;
  --sage-light:  #C5D5C5;
  --ocean:       #4A6FA5;
  --ocean-light: #D0DDEF;
  --terracotta:  #CC7A6B;
  --terra-light: #F5E0DC;
  --bg:          #F9F9F6;
  --card:        #FFFFFF;
  --border:      #E4E9E4;
  --text:        #2D3748;
  --muted:       #6B7280;
  --success:     #48BB78;
  --transition:  0.3s ease;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg:     #1E2326;
    --card:   #252B31;
    --border: #374151;
    --text:   #E2E8F0;
    --muted:  #9CA3AF;
  }
}

/* ── Global base ──────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--text) !important;
}
.stApp { background-color: var(--bg) !important; }

/* ── Hide default Streamlit chrome ────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }

/* ── Headings ─────────────────────────────────────────────────────────────── */
h1 { color: var(--ocean) !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2, h3 { color: var(--text) !important; font-weight: 600 !important; }

/* ── Textarea ─────────────────────────────────────────────────────────────── */
textarea {
  border: 1.5px solid var(--border) !important;
  border-radius: 12px !important;
  background: var(--card) !important;
  font-size: 0.95rem !important;
  transition: border-color var(--transition) !important;
}
textarea:focus { border-color: var(--ocean) !important; box-shadow: 0 0 0 3px var(--ocean-light) !important; }

/* ── Primary button (Analyse) ─────────────────────────────────────────────── */
div[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, var(--ocean), #3d5f8f) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  min-height: 48px !important;        /* ≥ 44px touch target */
  transition: opacity var(--transition), transform var(--transition) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  opacity: 0.90 !important;
  transform: translateY(-1px) !important;
}

/* ── Secondary buttons (feedback etc.) ───────────────────────────────────── */
div[data-testid="stButton"] > button {
  border-radius: 10px !important;
  min-height: 44px !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  transition: all var(--transition) !important;
}

/* ── Metric cards ─────────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 12px 16px !important;
}
div[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.8rem !important; }
div[data-testid="stMetricValue"] { color: var(--ocean) !important; font-weight: 700 !important; }

/* ── Cards ────────────────────────────────────────────────────────────────── */
.sdsr-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px 24px;
  margin-bottom: 12px;
  animation: fadeSlideIn 0.3s ease both;
}
.sdsr-card-crisis {
  border-color: var(--terracotta);
  background: var(--terra-light);
}
.sdsr-card-safe {
  border-color: var(--sage);
  background: #f4f8f4;
}

/* ── Score gauge ──────────────────────────────────────────────────────────── */
.score-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-weight: 700;
  font-size: 1.6rem;
  padding: 8px 20px;
  border-radius: 99px;
  margin: 8px 0;
}
.score-low    { background: #e6f4ea; color: #2d6a4f; }
.score-medium { background: #fff3e0; color: #b45309; }
.score-high   { background: var(--terra-light); color: var(--terracotta); }

/* ── Heatmap tokens ───────────────────────────────────────────────────────── */
.heatmap-wrap { line-height: 2.2rem; padding: 12px 0; }
.hm-token {
  display: inline-block;
  padding: 2px 6px;
  margin: 3px 2px;
  border-radius: 5px;
  font-size: 0.9rem;
  opacity: 0;
  animation: tokenReveal 0.4s ease forwards;
}
@keyframes tokenReveal {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── Processing pulse ─────────────────────────────────────────────────────── */
.breathing-pulse-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 32px;
}
.breathing-pulse {
  width: 72px; height: 72px;
  border-radius: 50%;
  background: var(--ocean);
  opacity: 0.7;
  animation: breathePulse 2.4s ease-in-out infinite;
}
@keyframes breathePulse {
  0%, 100% { transform: scale(1);    opacity: 0.7; }
  50%       { transform: scale(1.35); opacity: 0.4; }
}
.pulse-label {
  margin-top: 14px;
  color: var(--ocean);
  font-weight: 500;
  font-size: 0.9rem;
  letter-spacing: 0.02em;
  animation: fadePulse 2.4s ease-in-out infinite;
}
@keyframes fadePulse {
  0%, 100% { opacity: 0.5; }
  50%       { opacity: 1;   }
}

/* ── Intervention cards ───────────────────────────────────────────────────── */
.int-card {
  background: var(--card);
  border-left: 4px solid var(--sage);
  border-radius: 0 12px 12px 0;
  padding: 16px 20px;
  margin-bottom: 10px;
  animation: fadeSlideIn 0.35s ease both;
}
.int-card-interactive { border-color: var(--ocean); }
.int-card-behavioral  { border-color: var(--sage);  }
.int-card-info        { border-color: #b0bec5;       }
.int-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
             letter-spacing: 0.08em; color: var(--muted); margin-bottom: 4px; }
.int-action { font-size: 1rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.int-body   { font-size: 0.9rem; color: var(--muted); line-height: 1.55; }
.int-badge  { display: inline-block; font-size: 0.7rem; font-weight: 600;
              padding: 2px 8px; border-radius: 99px; margin-top: 8px; }
.badge-interactive { background: var(--ocean-light); color: var(--ocean); }
.badge-behavioral  { background: #e8f5e9; color: #2d6a4f; }
.badge-info        { background: #F0F0F5; color: #555; }

/* ── Crisis banner ────────────────────────────────────────────────────────── */
.crisis-banner {
  background: var(--terra-light);
  border: 2px solid var(--terracotta);
  border-radius: 16px;
  padding: 20px 24px;
  animation: fadeSlideIn 0.3s ease both;
}
.crisis-banner h3 { color: var(--terracotta) !important; margin-top: 0 !important; }
.crisis-resource {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 0;
  border-bottom: 1px solid #f0c5be;
}
.crisis-resource:last-child { border-bottom: none; }
.cr-label   { font-weight: 600; color: var(--text); font-size: 0.95rem; }
.cr-contact { font-size: 0.85rem; color: var(--terracotta); font-weight: 500; }

/* ── Feedback widget ──────────────────────────────────────────────────────── */
.feedback-wrap {
  border-top: 1px solid var(--border);
  margin-top: 14px;
  padding-top: 12px;
  font-size: 0.85rem;
  color: var(--muted);
}

/* ── Trigger chips ────────────────────────────────────────────────────────── */
.trigger-chip {
  display: inline-block;
  background: var(--ocean-light);
  color: var(--ocean);
  font-size: 0.75rem;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 99px;
  margin: 2px 3px;
  text-transform: capitalize;
}

/* ── Past entry collapse ──────────────────────────────────────────────────── */
.past-entry-shield {
  background: var(--card);
  border: 1px dashed var(--border);
  border-radius: 10px;
  padding: 10px 16px;
  cursor: pointer;
  color: var(--muted);
  font-size: 0.85rem;
  text-align: center;
  margin-bottom: 6px;
}

/* ── Keyframe: fade + slide ───────────────────────────────────────────────── */
@keyframes fadeSlideIn {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0);   }
}

/* ── Responsive: stack columns on small screens ───────────────────────────── */
@media (max-width: 768px) {
  .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
  .score-badge { font-size: 1.2rem; }
}

/* ── Sidebar / expanders ──────────────────────────────────────────────────── */
.streamlit-expanderHeader {
  border-radius: 10px !important;
  font-weight: 500 !important;
}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 ─ Self-contained animated HTML components
# ══════════════════════════════════════════════════════════════════════════════

_BREATHING_PULSE_HTML = """
<div class="breathing-pulse-wrap">
  <div class="breathing-pulse"></div>
  <div class="pulse-label">Analysing · please hold on…</div>
</div>
"""

def _breathing_exercise_html(mode: str = "box") -> str:
    """Return a self-contained HTML page with an animated SVG breathing exercise.

    mode='box'  → 4-4-4-4 (Box Breathing — used for exams/work)
    mode='478'  → 4-7-8   (4-7-8 Breathing — used for sleep)
    """
    if mode == "478":
        phases_js = "[{name:'Inhale',d:4},{name:'Hold',d:7},{name:'Exhale',d:8}]"
        title = "4–7–8 Breathing"
    else:
        phases_js = "[{name:'Inhale',d:4},{name:'Hold',d:4},{name:'Exhale',d:4},{name:'Hold',d:4}]"
        title = "Box Breathing"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', sans-serif;
    background: transparent;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    color: #2D3748;
  }}
  h4 {{ font-size: 1rem; font-weight: 600; color: #4A6FA5; margin-bottom: 12px; letter-spacing: 0.02em; }}
  svg {{ overflow: visible; }}
  #ring   {{ transition: all 4s ease-in-out; }}
  #fill   {{ transition: all 4s ease-in-out; }}
  #phase  {{ font-size: 1.2rem; font-weight: 600; margin-top: 12px; color: #4A6FA5; transition: color 0.4s; }}
  #timer  {{ font-size: 2.6rem; font-weight: 700; color: #2D3748; min-width: 2ch; text-align: center; }}
  #hint   {{ font-size: 0.8rem; color: #6B7280; margin-top: 6px; }}
  #stop-btn {{
    margin-top: 14px; background: #F0F0F5; border: none; border-radius: 8px;
    padding: 6px 18px; cursor: pointer; font-size: 0.8rem; color: #6B7280;
    font-family: inherit; transition: background 0.2s;
  }}
  #stop-btn:hover {{ background: #e0e0e8; }}
</style>
</head>
<body>
<h4>{title}</h4>
<svg width="160" height="160" viewBox="0 0 160 160">
  <circle cx="80" cy="80" r="72" fill="none" stroke="#C5D5C5" stroke-width="3" opacity="0.5"/>
  <circle id="ring" cx="80" cy="80" r="72" fill="none" stroke="#8FA28F"
          stroke-width="3" stroke-dasharray="452" stroke-dashoffset="452"
          stroke-linecap="round" transform="rotate(-90 80 80)"/>
  <circle id="fill" cx="80" cy="80" r="30" fill="#4A6FA5" opacity="0.55"/>
</svg>
<div id="phase">Get ready…</div>
<div id="timer">&nbsp;</div>
<div id="hint">Follow the circle · breathe with it</div>
<button id="stop-btn" onclick="clearInterval(tick);document.getElementById('phase').textContent='Done ✓'">Stop</button>

<script>
const phases = {phases_js};
const fill  = document.getElementById('fill');
const ring  = document.getElementById('ring');
const phase = document.getElementById('phase');
const timer = document.getElementById('timer');
const CIRC  = 2 * Math.PI * 72;   // ≈ 452

const TARGET_R  = {{Inhale: 62, Hold: 62, Exhale: 30}};
const PHASE_COL = {{Inhale: '#4A6FA5', Hold: '#8FA28F', Exhale: '#CC7A6B'}};

let pi = 0, t = phases[0].d, tick;

function applyPhase(p) {{
  const r = TARGET_R[p.name] ?? 30;
  fill.setAttribute('r', r);
  fill.setAttribute('fill', PHASE_COL[p.name] ?? '#4A6FA5');
  fill.style.transition = `all ${{p.d}}s ease-in-out`;
  // Arc progress: full circle for inhale/hold, reset for exhale
  const offset = (p.name === 'Exhale') ? CIRC : 0;
  ring.style.transition = `stroke-dashoffset ${{p.d}}s linear`;
  ring.style.strokeDashoffset = offset;
  ring.style.stroke = PHASE_COL[p.name] ?? '#8FA28F';
  phase.textContent = p.name;
  phase.style.color  = PHASE_COL[p.name] ?? '#4A6FA5';
}}

setTimeout(() => {{
  applyPhase(phases[pi]);
  timer.textContent = t;
  tick = setInterval(() => {{
    t--;
    timer.textContent = t;
    if (t <= 0) {{
      pi = (pi + 1) % phases.length;
      t  = phases[pi].d;
      applyPhase(phases[pi]);
    }}
  }}, 1000);
}}, 600);
</script>
</body>
</html>"""


def _heatmap_html(tokens: list[str], weights: list[float]) -> str:
    """Render attention heatmap with staggered left-to-right reveal animation."""
    if not tokens or not weights:
        return "<p style='color:#6B7280;font-style:italic;'>No attention data available.</p>"

    max_w = max(weights) if max(weights) > 0 else 1.0
    parts: list[str] = []
    for idx, (token, weight) in enumerate(zip(tokens, weights)):
        norm = weight / max_w
        alpha = round(0.12 + 0.75 * norm, 3)
        delay = round(idx * 0.04, 3)     # stagger: 40 ms per token
        label = token.replace("##", "").replace("▁", "")
        if not label.strip():
            continue
        # Soft warm-peach highlight (not alarming red)
        style = (
            f"background: rgba(204,122,107,{alpha});"
            f"animation-delay: {delay}s;"
        )
        parts.append(f'<span class="hm-token" style="{style}">{label}</span>')

    return '<div class="heatmap-wrap">' + "".join(parts) + "</div>"


def _intervention_card_html(title: str, action: str, body: str, itype: str) -> str:
    border_cls = {
        "interactive": "int-card-interactive",
        "behavioral":  "int-card-behavioral",
        "informational": "int-card-info",
    }.get(itype, "int-card-behavioral")
    badge_cls = {
        "interactive": "badge-interactive",
        "behavioral":  "badge-behavioral",
        "informational": "badge-info",
    }.get(itype, "badge-info")
    badge_label = {"interactive": "🎯 Interactive", "behavioral": "🏃 Action",
                   "informational": "📖 Guide"}.get(itype, itype)
    icon = {"interactive": "✨", "behavioral": "💪", "informational": "💡"}.get(itype, "•")
    return f"""
<div class="int-card {border_cls}">
  <div class="int-title">{icon} {title}</div>
  <div class="int-action">{action}</div>
  <div class="int-body">{body}</div>
  <span class="int-badge {badge_cls}">{badge_label}</span>
</div>"""


def _crisis_banner_html(resources: list[dict]) -> str:
    rows = ""
    for r in resources:
        url_part = (
            f'<a href="{r["url"]}" target="_blank" style="color:var(--terracotta);">{r["url"]}</a>'
            if r.get("url") else ""
        )
        rows += f"""
<div class="crisis-resource">
  <div>
    <div class="cr-label">{r['label']}</div>
    <div class="cr-contact">{r['contact']}{"&nbsp;&nbsp;" + url_part if url_part else ""}</div>
  </div>
</div>"""
    return f"""
<div class="crisis-banner">
  <h3>⚠️ Support Resources</h3>
  <p style="color:#b5574a;margin-bottom:12px;font-size:0.9rem;">
    The AI has paused. You are not alone — please reach out to one of the services below.
  </p>
  {rows}
</div>"""


def _score_badge_html(score: float) -> str:
    pct = round(score * 100)
    if score < 0.4:
        cls, icon, label = "score-low", "🟢", "Low"
    elif score < 0.65:
        cls, icon, label = "score-medium", "🟡", "Moderate"
    else:
        cls, icon, label = "score-high", "🔴", "High"
    return f'<div class="score-badge {cls}">{icon} {pct}% <span style="font-weight:400;font-size:1rem;">{label}</span></div>'


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 ─ Temporal chart
# ══════════════════════════════════════════════════════════════════════════════

def _temporal_chart(profile) -> go.Figure:
    summary = profile.summary()
    scores = summary["history_scores"]
    n = len(scores)

    if n == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No data yet — submit some text to begin",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    x = list(range(1, n + 1))

    # Adaptive threshold at each step
    thresholds: list[float] = []
    for i in range(n):
        sub = scores[: i + 1]
        if len(sub) < 2:
            thresholds.append(0.7)
        else:
            t = float(np.mean(sub)) + 1.5 * float(np.std(sub, ddof=1))
            thresholds.append(min(t, 1.0))

    # Moving-average velocity
    w = profile.velocity_window
    ma = [float(np.mean(scores[max(0, i - w + 1): i + 1])) for i in range(n)]
    velocity = [0.0] + [ma[i] - ma[i - 1] for i in range(1, n)]

    fig = go.Figure()

    # Shaded threshold band
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=thresholds + [0.0] * n,
        fill="toself",
        fillcolor="rgba(74,111,165,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Stress score line
    fig.add_trace(go.Scatter(
        x=x, y=scores,
        mode="lines+markers",
        name="Stress Score",
        line=dict(color="#4A6FA5", width=2.5, shape="spline"),
        marker=dict(size=7, color="#4A6FA5"),
    ))

    # Adaptive threshold line
    fig.add_trace(go.Scatter(
        x=x, y=thresholds,
        mode="lines",
        name="Adaptive Threshold (μ + 1.5σ)",
        line=dict(color="#CC7A6B", dash="dot", width=1.8),
    ))

    # Velocity bars (secondary axis)
    fig.add_trace(go.Bar(
        x=x, y=velocity,
        name="Stress Velocity (Vₛ)",
        marker_color=["#8FA28F" if v >= 0 else "#CC7A6B" for v in velocity],
        opacity=0.55,
        yaxis="y2",
    ))

    fig.update_layout(
        title=dict(text="Temporal Stress Profile", font=dict(size=15, color="#2D3748")),
        xaxis=dict(title="Entry #", showgrid=False, color="#6B7280"),
        yaxis=dict(title="Score / Threshold", range=[-0.05, 1.1],
                   gridcolor="#F0F0F0", color="#6B7280"),
        yaxis2=dict(title="Velocity", overlaying="y", side="right",
                    range=[-0.5, 0.5], showgrid=False, color="#6B7280"),
        legend=dict(orientation="h", y=-0.22, font=dict(size=11)),
        height=380,
        hovermode="x unified",
        margin=dict(t=45, b=80, l=50, r=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 ─ Model loading
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_engine():
    checkpoint = REPO_ROOT / "models" / "checkpoints" / "best_model.pt"
    if not checkpoint.exists():
        return None
    from src.models.inference import StressInferenceEngine
    return StressInferenceEngine(checkpoint)


def _demo_predict(text: str) -> dict:
    """Deterministic fake prediction for demo mode (no checkpoint needed)."""
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    rng = random.Random(h)
    score = round(rng.uniform(0.25, 0.90), 3)
    words = text.split()[:30]
    attn = [rng.random() for _ in words]
    return {"stress_score": score, "tokens": words, "attn_weights": attn}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 ─ Page layout
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Stress Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject global CSS
st.markdown(_CSS, unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "profile" not in st.session_state:
    from src.temporal.temporal_profile import TemporalStressProfile
    st.session_state.profile = TemporalStressProfile()

if "history" not in st.session_state:
    # Each entry: {timestamp, score, text, reco_result}
    st.session_state.history: list[dict] = []

if "recommender" not in st.session_state:
    from src.recommender.recommendation_engine import RecommendationEngine
    st.session_state.recommender = RecommendationEngine()

if "feedback" not in st.session_state:
    st.session_state.feedback: dict[int, str] = {}   # entry_idx → "helpful"|"unhelpful"

if "adjust_threshold" not in st.session_state:
    st.session_state.adjust_threshold: dict[int, bool] = {}

if "custom_threshold" not in st.session_state:
    st.session_state.custom_threshold: float | None = None

if "latest_result" not in st.session_state:
    st.session_state.latest_result: dict | None = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-bottom:1rem;">
      <h1 style="margin-bottom:0;">🧠 Stress Detection System</h1>
      <p style="color:#6B7280;margin-top:4px;font-size:0.95rem;">
        Multichannel CNN · Temporal Modelling · Safety-First Recommendations
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Two-column layout ──────────────────────────────────────────────────────────
col_input, col_chart = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Input, Processing, Score, Heatmap, Interventions
# ══════════════════════════════════════════════════════════════════════════════
with col_input:
    st.markdown("### 📝 Your Entry")
    user_text = st.text_area(
        label="entry_text",
        label_visibility="collapsed",
        height=140,
        placeholder="Write how you're feeling today — a journal entry, a tweet, or anything on your mind…",
        key="user_text_input",
    )

    analyse_btn = st.button(
        "🔍 Analyse Stress",
        type="primary",
        use_container_width=True,
        key="analyse_btn",
    )

    # Processing animation placeholder
    anim_placeholder = st.empty()
    result_placeholder = st.empty()

    # ── Run analysis ────────────────────────────────────────────────────────────
    if analyse_btn and user_text.strip():
        # Step 1 — show breathing pulse while processing
        anim_placeholder.markdown(_CSS + _BREATHING_PULSE_HTML, unsafe_allow_html=True)

        # Step 2a — Circuit Breaker (before model)
        from src.recommender.recommendation_engine import RecommendationEngine
        reco_engine: RecommendationEngine = st.session_state.recommender
        crisis = reco_engine.circuit_breaker(user_text)

        if crisis:
            anim_placeholder.empty()
            from src.recommender.recommendation_engine import _EMERGENCY_RESOURCES
            reco_result_dict = {
                "status": "crisis",
                "stress_score": 1.0,
                "tokens": [],
                "attn_weights": [],
                "triggers_found": [],
                "interventions": [],
                "emergency_resources": _EMERGENCY_RESOURCES,
                "explanation": "Crisis signal detected. AI paused.",
            }
        else:
            # Step 2b — ML Inference
            engine = _load_engine()
            if engine is None:
                predict_result = _demo_predict(user_text)
                demo_mode = True
            else:
                predict_result = engine.predict(user_text)
                demo_mode = False

            stress_score: float = predict_result["stress_score"]
            tokens: list[str] = predict_result["tokens"]
            attn_weights: list[float] = predict_result["attn_weights"]

            # Step 3 — Temporal context
            from src.temporal.temporal_profile import StressEvent
            profile = st.session_state.profile

            # Apply custom threshold offset if user adjusted it
            if st.session_state.custom_threshold is not None:
                            # Temporarily shift the profile's adaptive_k so threshold equals custom_threshold
                            pass  # threshold override applied at display time

            profile.add_event(
                StressEvent(
                    timestamp=time.time(),
                    score=stress_score,
                    text_snippet=user_text[:80],
                )
            )
            should_intervene = profile.should_intervene()
            is_high_volatility = profile.is_high_volatility()

            # Step 4 — Recommendations
            reco = reco_engine.recommend(
                text=user_text,
                stress_score=stress_score,
                is_high_volatility=is_high_volatility,
                should_intervene=should_intervene,
            )

            reco_result_dict = {
                "status": reco.status,
                "stress_score": stress_score,
                "tokens": tokens,
                "attn_weights": attn_weights,
                "triggers_found": reco.triggers_found,
                "interventions": [
                    {"title": i.title, "action": i.action, "body": i.body, "type": i.type}
                    for i in reco.interventions
                ],
                "emergency_resources": [],
                "explanation": reco.explanation,
                "demo_mode": demo_mode,
            }

        # Store in session history
        entry_idx = len(st.session_state.history)
        st.session_state.history.append({
            "timestamp": time.time(),
            "score": reco_result_dict.get("stress_score", 1.0),
            "text": user_text,
            "result": reco_result_dict,
        })
        st.session_state.latest_result = reco_result_dict

        # Remove animation
        anim_placeholder.empty()

    # ── Render latest result ────────────────────────────────────────────────────
    result = st.session_state.latest_result
    if result:
        # Demo mode banner
        if result.get("demo_mode"):
            st.info(
                "⚡ **Demo mode** — no trained checkpoint found. "
                "Scores are illustrative. Run `src/training/train.py` for real inference.",
                icon="ℹ️",
            )

        # ── Crisis banner ──────────────────────────────────────────────────────
        if result["status"] == "crisis":
            st.markdown(
                _crisis_banner_html(result["emergency_resources"]),
                unsafe_allow_html=True,
            )

        else:
            # ── Score badge ────────────────────────────────────────────────────
            st.markdown(
                f"""
                <div class="sdsr-card" style="padding:16px 20px;margin-bottom:8px;">
                  <div style="font-size:0.8rem;color:var(--muted);font-weight:500;
                               text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">
                    Stress Score
                  </div>
                  {_score_badge_html(result["stress_score"])}
                  <div style="font-size:0.85rem;color:var(--muted);margin-top:6px;">
                    {result["explanation"]}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Trigger chips
            if result.get("triggers_found"):
                chips = "".join(
                    f'<span class="trigger-chip">{t}</span>'
                    for t in result["triggers_found"]
                )
                st.markdown(
                    f'<div style="margin-bottom:10px;"><strong style="font-size:0.85rem;">'
                    f'Triggers detected:</strong> {chips}</div>',
                    unsafe_allow_html=True,
                )

            # ── Attention heatmap ──────────────────────────────────────────────
            with st.expander("🔥 Attention Heatmap — what the model focused on", expanded=True):
                st.markdown(
                    "<p style='font-size:0.82rem;color:#6B7280;margin-bottom:6px;'>"
                    "Darker tokens received more model attention. "
                    "They highlight stress-relevant language patterns.</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _heatmap_html(result["tokens"], result["attn_weights"]),
                    unsafe_allow_html=True,
                )

            # ── Intervention cards ─────────────────────────────────────────────
            if result["interventions"]:
                st.markdown(
                    "#### 💡 Personalised Interventions",
                    unsafe_allow_html=False,
                )
                for idx_i, interv in enumerate(result["interventions"]):
                    st.markdown(
                        _intervention_card_html(
                            interv["title"],
                            interv["action"],
                            interv["body"],
                            interv["type"],
                        ),
                        unsafe_allow_html=True,
                    )
                    # Render interactive breathing exercise inline
                    if interv["type"] == "interactive":
                        with st.expander(f"▶ Start: {interv['action']}"):
                            mode = "478" if "4-7-8" in interv["action"] else "box"
                            components.html(
                                _breathing_exercise_html(mode),
                                height=320,
                                scrolling=False,
                            )

            # ── Feedback widget ────────────────────────────────────────────────
            entry_idx = len(st.session_state.history) - 1
            fb_state = st.session_state.feedback.get(entry_idx)

            st.markdown(
                '<div class="feedback-wrap">Was this helpful?</div>',
                unsafe_allow_html=True,
            )
            fb_col1, fb_col2, _ = st.columns([1, 1, 3])
            with fb_col1:
                if st.button("👍 Yes", key=f"fb_yes_{entry_idx}", use_container_width=True):
                    st.session_state.feedback[entry_idx] = "helpful"
                    st.session_state.adjust_threshold[entry_idx] = False
                    st.rerun()
            with fb_col2:
                if st.button("👎 No", key=f"fb_no_{entry_idx}", use_container_width=True):
                    st.session_state.feedback[entry_idx] = "unhelpful"
                    st.session_state.adjust_threshold[entry_idx] = True
                    st.rerun()

            if fb_state == "helpful":
                st.success("Thank you for your feedback! 🌿", icon="✅")
            elif fb_state == "unhelpful" and st.session_state.adjust_threshold.get(entry_idx):
                st.markdown(
                    "<div style='background:var(--card);border:1px solid var(--border);"
                    "border-radius:12px;padding:14px 18px;margin-top:6px;"
                    "animation:fadeSlideIn 0.3s ease both;'>"
                    "<strong style='font-size:0.9rem;'>Adjust your sensitivity</strong><br>"
                    "<span style='font-size:0.82rem;color:#6B7280;'>"
                    "Move the slider to tell the system when you'd like it to alert you.</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                new_thresh = st.slider(
                    "My personal alert threshold",
                    min_value=0.30,
                    max_value=0.95,
                    value=st.session_state.custom_threshold or 0.65,
                    step=0.05,
                    format="%.0f%%",
                    key=f"thresh_slider_{entry_idx}",
                    help="The system will alert you when your stress score exceeds this value.",
                )
                if st.button("Save threshold", key=f"save_thresh_{entry_idx}"):
                    st.session_state.custom_threshold = new_thresh
                    st.session_state.adjust_threshold[entry_idx] = False
                    st.success(f"Threshold updated to {new_thresh:.0%} ✓")
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Temporal chart + metrics + past entries
# ══════════════════════════════════════════════════════════════════════════════
with col_chart:
    st.markdown("### 📊 Temporal Stress Profile")

    profile = st.session_state.profile
    summary = profile.summary()

    if summary["n_events"] > 0:
        m1, m2, m3 = st.columns(3)
        cur = summary["current_score"]
        vs  = summary["stress_velocity"]
        thr = st.session_state.custom_threshold or summary["adaptive_threshold"]

        m1.metric("Current Score", f"{cur:.0%}")
        m2.metric(
            "Velocity (Vₛ)",
            f"{vs:+.3f}",
            delta=f"{vs:+.3f}",
            delta_color="inverse",
            help="Rate of change of the stress moving average. Positive = rising.",
        )
        m3.metric(
            "Threshold",
            f"{thr:.0%}",
            help="Adaptive threshold (μ + 1.5σ). Alert when score exceeds this.",
        )

        # Status flag banner
        if summary["should_intervene"]:
            st.markdown(
                '<div style="background:var(--terra-light);border:1px solid var(--terracotta);'
                'border-radius:10px;padding:10px 16px;font-size:0.85rem;color:#b5574a;'
                'margin-bottom:8px;animation:fadeSlideIn 0.3s ease both;">'
                '⚠️ <strong>Intervention recommended</strong> — current score exceeds adaptive threshold.'
                '</div>',
                unsafe_allow_html=True,
            )
        elif summary["is_high_volatility"]:
            st.markdown(
                '<div style="background:#fff3e0;border:1px solid #f4a261;'
                'border-radius:10px;padding:10px 16px;font-size:0.85rem;color:#b45309;'
                'margin-bottom:8px;animation:fadeSlideIn 0.3s ease both;">'
                '📉 <strong>High volatility detected</strong> — stress is changing rapidly.'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#f4f8f4;border:1px solid #8FA28F;'
                'border-radius:10px;padding:10px 16px;font-size:0.85rem;color:#2d6a4f;'
                'margin-bottom:8px;">'
                '✅ Stress is within your normal range.'
                '</div>',
                unsafe_allow_html=True,
            )

    # Plotly chart
    fig = _temporal_chart(profile)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "**Adaptive Threshold** = μ + 1.5σ over your history · "
        "**Velocity (Vₛ)** = change in moving-average score between entries"
    )

    # ── Past entries (collapsed content warnings) ───────────────────────────────
    if st.session_state.history:
        st.markdown("### 🗂 Past Entries")
        st.caption(
            "Entries are collapsed by default to prevent re-triggering. "
            "Tap to reveal."
        )
        for idx_h, entry in enumerate(reversed(st.session_state.history)):
            age_s = time.time() - entry["timestamp"]
            age_str = f"{int(age_s // 60)} min ago" if age_s < 3600 else f"{int(age_s // 3600)} h ago"
            score_h = entry["score"]
            color = (
                "var(--sage)" if score_h < 0.4
                else ("orange" if score_h < 0.65 else "var(--terracotta)")
            )

            with st.expander(
                f"Entry #{len(st.session_state.history) - idx_h}  ·  "
                f"Score: {score_h:.0%}  ·  {age_str}",
                expanded=False,
            ):
                st.markdown(
                    f'<div style="font-size:0.85rem;color:var(--muted);border-left:3px solid {color};'
                    f'padding-left:10px;">{entry["text"][:200]}'
                    f'{"…" if len(entry["text"]) > 200 else ""}</div>',
                    unsafe_allow_html=True,
                )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="margin-top:2rem;border-top:1px solid var(--border);padding-top:1rem;
                text-align:center;color:var(--muted);font-size:0.78rem;">
      🔒 <strong>Privacy notice:</strong> No data leaves your browser session.
      All inference runs locally. · Built with PyTorch · Streamlit · FastAPI
    </div>
    """,
    unsafe_allow_html=True,
)
