"""
src/api/app.py

FastAPI backend that exposes:
  POST /predict      – single-text stress prediction + recommendations
  GET  /profile/{uid} – temporal stress profile summary for a user
  POST /profile/{uid}/add – add a new stress event for a user
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.recommender.recommendation_engine import RecommendationEngine
from src.temporal.temporal_profile import StressEvent, TemporalStressProfile

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = REPO_ROOT / "models" / "checkpoints" / "best_model.pt"

app = FastAPI(
    title="Stress Detection & Recommender API",
    version="1.0.0",
    description="Production-grade stress detection with temporal modelling and safety-first recommendations.",
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


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    user_id: Optional[str] = None


class PredictResponse(BaseModel):
    stress_score: float
    tokens: list[str]
    attn_weights: list[float]
    layer: int
    crisis_detected: bool
    triggers_found: list[str]
    recommendations: list[str]
    emergency_resources: list[str]
    explanation: str
    temporal_summary: Optional[dict] = None


class AddEventRequest(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    text_snippet: str = ""
    timestamp: Optional[float] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    engine = get_inference_engine()
    result = engine.predict(req.text)
    stress_score: float = result["stress_score"]

    # Update temporal profile if user_id provided
    temporal_summary = None
    should_intervene = False
    is_high_volatility = False

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
        temporal_summary = profile.summary()

    reco = recommender.recommend(
        text=req.text,
        stress_score=stress_score,
        is_high_volatility=is_high_volatility,
        should_intervene=should_intervene,
    )

    return PredictResponse(
        stress_score=stress_score,
        tokens=result["tokens"],
        attn_weights=result["attn_weights"],
        layer=reco.layer,
        crisis_detected=reco.crisis_detected,
        triggers_found=reco.triggers_found,
        recommendations=reco.recommendations,
        emergency_resources=reco.emergency_resources,
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
