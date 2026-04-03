"""
src/temporal/temporal_profile.py

TemporalStressProfile – tracks a user's stress scores over time and
computes:
  - Stress Velocity (Vs): moving-average derivative over the last N posts
  - Adaptive Threshold: μ + 1.5σ (eliminates alert fatigue)
  - Intervention trigger flags
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class StressEvent:
    """A single stress measurement."""

    timestamp: float   # Unix timestamp (or monotonic counter)
    score: float       # Stress probability in [0, 1]
    text_snippet: str = ""


@dataclass
class TemporalStressProfile:
    """Maintains a rolling window of stress scores for one user.

    Parameters
    ----------
    window_size:
        Maximum number of recent events to retain (default 50).
    velocity_window:
        Number of most-recent scores used to compute Vs (default 5).
    adaptive_k:
        Multiplier for the adaptive threshold: threshold = μ + k*σ (default 1.5).
    """

    window_size: int = 50
    velocity_window: int = 5
    adaptive_k: float = 1.5

    _history: deque = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        self._history: deque[StressEvent] = deque(maxlen=self.window_size)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_event(self, event: StressEvent) -> None:
        """Append a new stress event to the history."""
        self._history.append(event)

    def get_scores(self) -> list[float]:
        """Return the list of scores in chronological order."""
        return [e.score for e in self._history]

    def get_timestamps(self) -> list[float]:
        """Return the list of timestamps in chronological order."""
        return [e.timestamp for e in self._history]

    # ── Stress Velocity ───────────────────────────────────────────────────────

    def stress_velocity(self) -> float:
        """Compute Vs: the rate of change of the moving average of recent scores.

        Vs = moving_average[-1] - moving_average[-2]

        where the moving average is computed over the last ``velocity_window``
        scores.

        Returns
        -------
        float
            Positive → stress increasing, negative → decreasing, 0 → stable.
            Returns 0.0 if there are fewer than 2 data points.
        """
        scores = self.get_scores()
        if len(scores) < 2:
            return 0.0

        # Compute rolling mean with window = velocity_window
        w = self.velocity_window
        # Build moving-average series
        ma: list[float] = []
        for i in range(len(scores)):
            start = max(0, i - w + 1)
            ma.append(float(np.mean(scores[start: i + 1])))

        return ma[-1] - ma[-2]

    # ── Adaptive Threshold ────────────────────────────────────────────────────

    def adaptive_threshold(self) -> float:
        """Compute the intervention threshold as μ + k*σ over the history.

        Returns
        -------
        float
            Adaptive threshold in [0, 1].  Clamped to [0, 1].
            Falls back to 0.7 if there are fewer than 2 data points.
        """
        scores = self.get_scores()
        if len(scores) < 2:
            return 0.7   # sensible default until enough history

        mu = float(np.mean(scores))
        sigma = float(np.std(scores, ddof=1))
        threshold = mu + self.adaptive_k * sigma
        return float(np.clip(threshold, 0.0, 1.0))

    # ── Intervention Flags ────────────────────────────────────────────────────

    def current_score(self) -> Optional[float]:
        """Return the most recent stress score, or None if no history."""
        if not self._history:
            return None
        return self._history[-1].score

    def should_intervene(self) -> bool:
        """True if the latest score exceeds the adaptive threshold."""
        score = self.current_score()
        if score is None:
            return False
        return score >= self.adaptive_threshold()

    def is_high_volatility(self, volatility_threshold: float = 0.1) -> bool:
        """True if the stress velocity magnitude exceeds *volatility_threshold*.

        Used to trigger Layer 3 (Preventive Nudges) even when current stress
        is not over the adaptive threshold.
        """
        return abs(self.stress_velocity()) >= volatility_threshold

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a JSON-serialisable summary of the current profile."""
        return {
            "n_events": len(self._history),
            "current_score": self.current_score(),
            "stress_velocity": self.stress_velocity(),
            "adaptive_threshold": self.adaptive_threshold(),
            "should_intervene": self.should_intervene(),
            "is_high_volatility": self.is_high_volatility(),
            "history_scores": self.get_scores(),
            "history_timestamps": self.get_timestamps(),
        }
