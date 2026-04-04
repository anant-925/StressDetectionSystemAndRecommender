"""
src/api/app.py

FastAPI backend that exposes:
  POST /predict           – single-text stress prediction + recommendations
  GET  /profile/{uid}     – temporal stress profile summary for a user
  POST /profile/{uid}/add – add a new stress event for a user

Response format follows the pipeline specification:
  {
    "status": "safe | crisis | nudge",
    "metrics": {"stress_score": float, "velocity": str, "threshold_crossed": bool},
    "explainability": [{"word": str, "weight": float}, ...],
    "interventions": [{"title": str, "action": str, "body": str, "type": str}, ...]
  }
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.recommender.recommendation_engine import RecommendationEngine
from src.temporal.temporal_profile import StressEvent, TemporalStressProfile

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = REPO_ROOT / "models" / "checkpoints" / "best_model.pt"

app = FastAPI(
    title="Stress Detection & Recommender API",
    version="2.0.0",
    description=(
        "Production-grade stress detection with temporal modelling and safety-first recommendations. "
        "Follows a strict 4-step pipeline: Input → Circuit Breaker → ML + Temporal → Output."
    ),
)

# ── Lazy-load inference engine ────────────────────────────────────────────────
_inference_engine = None


def get_inference_engine():
    global _inference_engine
    if _inference_engine is None:
        if not CHECKPOINT_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model checkpoint not found. Run src/training/train.py first.",
            )
        from src.models.inference import StressInferenceEngine
        _inference_engine = StressInferenceEngine(CHECKPOINT_PATH)
    return _inference_engine


# ── In-memory profile store (replace with a DB in production) ─────────────────
_profiles: dict[str, TemporalStressProfile] = {}

recommender = RecommendationEngine()


# ── Request / Response Schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    user_id: Optional[str] = None
    text: str = Field(..., min_length=1, max_length=10_000)
    timestamp: Optional[str] = None     # ISO-8601 string (stored for display)


class MetricsPayload(BaseModel):
    stress_score: float
    velocity: str                       # formatted with sign e.g. "+0.15"
    threshold_crossed: bool


class ExplainabilityToken(BaseModel):
    word: str
    weight: float


class InterventionPayload(BaseModel):
    title: str
    action: str
    body: str
    type: Literal["interactive", "behavioral", "informational"]


class EmergencyResource(BaseModel):
    label: str
    contact: str
    url: Optional[str] = None


class PredictResponse(BaseModel):
    status: Literal["safe", "crisis", "nudge"]
    metrics: MetricsPayload
    explainability: list[ExplainabilityToken]
    interventions: list[InterventionPayload]
    emergency_resources: list[EmergencyResource]
    explanation: str
    temporal_summary: Optional[dict] = None


class AddEventRequest(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    text_snippet: str = ""
    timestamp: Optional[float] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    4-step pipeline:
      Step 1 – Receive text
      Step 2 – Circuit Breaker check (Layer 1), then ML inference
      Step 3 – Temporal context (Vs, adaptive threshold, trigger extraction)
      Step 4 – Build and return structured response
    """

    # ── Step 2a: Circuit Breaker (runs BEFORE the heavy ML model) ─────────────
    if recommender.circuit_breaker(req.text):
        # Immediately return emergency payload; ML model is never called
        from src.recommender.recommendation_engine import _EMERGENCY_RESOURCES
        return PredictResponse(
            status="crisis",
            metrics=MetricsPayload(stress_score=1.0, velocity="+0.00", threshold_crossed=True),
            explainability=[],
            interventions=[],
            emergency_resources=[EmergencyResource(**r) for r in _EMERGENCY_RESOURCES],
            explanation=(
                "Crisis signal detected. AI recommendations are paused. "
                "Please reach out to one of the emergency resources immediately."
            ),
            temporal_summary=None,
        )

    # ── Step 2b: ML Inference ─────────────────────────────────────────────────
    engine = get_inference_engine()
    result = engine.predict(req.text)
    stress_score: float = result["stress_score"]
    tokens: list[str] = result["tokens"]
    attn_weights: list[float] = result["attn_weights"]

    # Build explainability list (word + weight, filtered to real tokens)
    explainability = [
        ExplainabilityToken(word=t.replace("##", ""), weight=round(w, 4))
        for t, w in zip(tokens, attn_weights)
        if t not in ("[PAD]", "[CLS]", "[SEP]", "<pad>")
    ]

    # ── Step 3: Temporal Context ──────────────────────────────────────────────
    temporal_summary = None
    should_intervene = False
    is_high_volatility = False
    velocity_float = 0.0

    if req.user_id:
        if req.user_id not in _profiles:
            _profiles[req.user_id] = TemporalStressProfile()
        profile = _profiles[req.user_id]
        profile.add_event(
            StressEvent(
                timestamp=time.time(),
                score=stress_score,
                text_snippet=req.text[:100],
            )
        )
        should_intervene = profile.should_intervene()
        is_high_volatility = profile.is_high_volatility()
        velocity_float = profile.stress_velocity()
        temporal_summary = profile.summary()

    velocity_str = f"{velocity_float:+.3f}"

    # ── Step 4: Recommendation Engine ────────────────────────────────────────
    reco = recommender.recommend(
        text=req.text,
        stress_score=stress_score,
        is_high_volatility=is_high_volatility,
        should_intervene=should_intervene,
    )

    threshold = (
        _profiles[req.user_id].adaptive_threshold()
        if req.user_id and req.user_id in _profiles
        else 0.7
    )

    interventions_payload = [
        InterventionPayload(
            title=i.title,
            action=i.action,
            body=i.body,
            type=i.type,
        )
        for i in reco.interventions
    ]

    return PredictResponse(
        status=reco.status,
        metrics=MetricsPayload(
            stress_score=round(stress_score, 4),
            velocity=velocity_str,
            threshold_crossed=stress_score >= threshold,
        ),
        explainability=explainability,
        interventions=interventions_payload,
        emergency_resources=[],
        explanation=reco.explanation,
        temporal_summary=temporal_summary,
    )


@app.get("/profile/{uid}")
def get_profile(uid: str):
    if uid not in _profiles:
        raise HTTPException(status_code=404, detail=f"No profile found for user '{uid}'")
    return _profiles[uid].summary()


@app.post("/profile/{uid}/add")
def add_event(uid: str, req: AddEventRequest):
    if uid not in _profiles:
        _profiles[uid] = TemporalStressProfile()
    _profiles[uid].add_event(
        StressEvent(
            timestamp=req.timestamp or time.time(),
            score=req.score,
            text_snippet=req.text_snippet,
        )
    )
    return _profiles[uid].summary()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
