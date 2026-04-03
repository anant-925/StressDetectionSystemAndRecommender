"""
src/recommender/recommendation_engine.py

3-layer safety-first recommendation engine.

Layer 1 – Circuit Breaker
    Regex / keyword matcher for severe crisis signals.
    If triggered → HALT AI, return emergency lifelines only.

Layer 2 – Context Matcher
    Identifies stress triggers (sleep, exams, money, work, relationships)
    and maps them to targeted micro-interventions.

Layer 3 – Preventive Nudges
    Triggered when baseline stress is low but volatility is high.
    Returns gentle, generalised wellness nudges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Crisis patterns (Layer 1) ─────────────────────────────────────────────────

# Each pattern is a compiled regex.  Match is case-insensitive.
_CRISIS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(suicid(e|al|ally))\b", re.IGNORECASE),
    re.compile(r"\b(kill\s+(my)?self|end\s+(my\s+)?life)\b", re.IGNORECASE),
    re.compile(r"\b(want\s+to\s+die|wish\s+(i\s+)?was\s+dead)\b", re.IGNORECASE),
    re.compile(r"\b(self[\s-]?harm|cutting\s+myself|hurt\s+myself)\b", re.IGNORECASE),
    re.compile(r"\b(overdos(e|ing)|took\s+too\s+many\s+pills)\b", re.IGNORECASE),
    re.compile(r"\b(no\s+reason\s+to\s+live|life\s+is\s+not\s+worth)\b", re.IGNORECASE),
    re.compile(r"\b(don['']t\s+want\s+to\s+(be\s+here|exist|live))\b", re.IGNORECASE),
    re.compile(r"\b(hopeless|worthless|can['']t\s+go\s+on)\b", re.IGNORECASE),
]

_EMERGENCY_RESOURCES = [
    "🆘 National Suicide Prevention Lifeline: call or text **988** (US)",
    "🌍 International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/",
    "💬 Crisis Text Line: text HOME to **741741** (US/UK/CA/IE)",
    "☎️  Samaritans (UK/IE): **116 123** (free, 24/7)",
    "🏥 If in immediate danger, please call emergency services: **911 / 999 / 112**",
]

# ── Context trigger patterns (Layer 2) ───────────────────────────────────────

@dataclass
class TriggerCategory:
    name: str
    patterns: list[re.Pattern]
    interventions: list[str]


_TRIGGER_CATEGORIES: list[TriggerCategory] = [
    TriggerCategory(
        name="sleep",
        patterns=[
            re.compile(r"\b(can['']t\s+sleep|insomnia|sleep(less|ing\s+poorly)|awake\s+all\s+night|tired\s+all\s+day)\b", re.IGNORECASE),
            re.compile(r"\b(no\s+sleep|bad\s+sleep|sleep\s+depriv(ed|ation))\b", re.IGNORECASE),
        ],
        interventions=[
            "🌙 **4-7-8 Breathing**: Inhale 4 s → Hold 7 s → Exhale 8 s (repeat ×4 before bed).",
            "📵 Try a 30-minute phone-free wind-down routine before sleep.",
            "🌡️ Keep your bedroom cool (~18 °C / 65 °F) and dark – it signals your brain to release melatonin.",
            "📓 Write down tomorrow's to-do list to offload mental loops.",
        ],
    ),
    TriggerCategory(
        name="exams",
        patterns=[
            re.compile(r"\b(exam|test|quiz|midterm|final|assignment|deadline|grade|study(ing)?)\b", re.IGNORECASE),
        ],
        interventions=[
            "📚 **Pomodoro Technique**: 25 min focused study → 5 min break. After 4 cycles, take a 20-min break.",
            "🧠 Prioritise topics by importance × difficulty – tackle hardest material when energy is highest.",
            "🤲 Before the exam: box-breathing (4 s in → 4 s hold → 4 s out → 4 s hold) to lower cortisol.",
            "🏃 A 10-minute walk before studying boosts memory consolidation.",
        ],
    ),
    TriggerCategory(
        name="money",
        patterns=[
            re.compile(r"\b(debt|bill|rent|afford|broke|financial|money|loan|credit|budget)\b", re.IGNORECASE),
            re.compile(r"\b(can['']t\s+pay|running\s+out\s+of\s+money|overdraft)\b", re.IGNORECASE),
        ],
        interventions=[
            "💳 List your expenses in three buckets: *needs*, *wants*, *savings*. Visibility reduces anxiety.",
            "📞 Contact your lender – most offer hardship programmes; asking is free.",
            "🆓 Check eligibility for local/national emergency assistance programmes.",
            "🧘 Separate what you *can* control (next action) from what you cannot – focus only on the former.",
        ],
    ),
    TriggerCategory(
        name="work",
        patterns=[
            re.compile(r"\b(boss|manager|coworker|workplace|burnout|overwhelmed|overwork(ed)?|deadlin(e|es)|fired|laid\s+off)\b", re.IGNORECASE),
        ],
        interventions=[
            "🗓️ Time-block your day: assign fixed slots to tasks to prevent decision fatigue.",
            "✋ Practice assertive boundary-setting: it's okay to say 'I can take this on by [date]'.",
            "💬 Schedule a direct, calm conversation with your manager about workload expectations.",
            "🧘 5-minute micro-break every 90 minutes improves sustained focus and lowers cortisol.",
        ],
    ),
    TriggerCategory(
        name="relationships",
        patterns=[
            re.compile(r"\b(breakup|divorce|lonely|loneliness|isolated|argument|fight|relation|partner|friend)\b", re.IGNORECASE),
        ],
        interventions=[
            "🤝 Reach out to one trusted person today – a short text counts.",
            "📖 Journalling about the situation for 15 minutes can organise emotions and reduce rumination.",
            "🧘 Self-compassion exercise: speak to yourself as you would to a close friend in the same situation.",
            "🌳 Spending 20 minutes in nature significantly reduces stress hormones.",
        ],
    ),
]

# ── Preventive nudges (Layer 3) ───────────────────────────────────────────────

_PREVENTIVE_NUDGES = [
    "📈 Your stress patterns show early signs of volatility. Consider a brief mindfulness check-in.",
    "🚶 A 15-minute walk outdoors can reset your nervous system and lower baseline cortisol.",
    "💧 Hydration check: even mild dehydration amplifies perceived stress – drink a glass of water.",
    "😴 Prioritise 7–9 hours of sleep tonight; sleep is the single most powerful stress modulator.",
    "📔 Try a 5-minute gratitude log – list 3 specific things that went well today.",
    "📵 Take a 30-minute break from social media to reduce ambient anxiety.",
]


# ── Main engine ───────────────────────────────────────────────────────────────

@dataclass
class RecommendationResult:
    layer: int                  # 1, 2, or 3
    crisis_detected: bool
    triggers_found: list[str]
    recommendations: list[str]
    emergency_resources: list[str]
    explanation: str


class RecommendationEngine:
    """3-layer safety-first recommendation engine."""

    # ── Layer 1 ───────────────────────────────────────────────────────────────

    @staticmethod
    def circuit_breaker(text: str) -> bool:
        """Return True if any crisis pattern is detected."""
        for pattern in _CRISIS_PATTERNS:
            if pattern.search(text):
                return True
        return False

    # ── Layer 2 ───────────────────────────────────────────────────────────────

    @staticmethod
    def context_matcher(text: str) -> list[TriggerCategory]:
        """Return trigger categories detected in *text*."""
        matched: list[TriggerCategory] = []
        for category in _TRIGGER_CATEGORIES:
            for pattern in category.patterns:
                if pattern.search(text):
                    matched.append(category)
                    break   # one match per category is enough
        return matched

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(
        self,
        text: str,
        stress_score: float,
        is_high_volatility: bool = False,
        should_intervene: bool = False,
    ) -> RecommendationResult:
        """Generate recommendations for a given text and stress assessment.

        Parameters
        ----------
        text:
            Raw user text.
        stress_score:
            Predicted stress probability in [0, 1].
        is_high_volatility:
            True if TemporalStressProfile.is_high_volatility() returned True.
        should_intervene:
            True if TemporalStressProfile.should_intervene() returned True.

        Returns
        -------
        RecommendationResult
        """

        # ── Layer 1: Circuit Breaker ──────────────────────────────────────────
        if self.circuit_breaker(text):
            return RecommendationResult(
                layer=1,
                crisis_detected=True,
                triggers_found=[],
                recommendations=[],
                emergency_resources=_EMERGENCY_RESOURCES,
                explanation=(
                    "⚠️ Crisis signal detected. AI recommendations are paused. "
                    "Please reach out to one of the emergency resources below immediately."
                ),
            )

        # ── Layer 2: Context Matcher ──────────────────────────────────────────
        matched_categories = self.context_matcher(text)

        if matched_categories and (should_intervene or stress_score >= 0.5):
            triggers_found = [c.name for c in matched_categories]
            recommendations: list[str] = []
            for category in matched_categories:
                recommendations.extend(category.interventions[:2])   # top 2 per category

            return RecommendationResult(
                layer=2,
                crisis_detected=False,
                triggers_found=triggers_found,
                recommendations=recommendations,
                emergency_resources=[],
                explanation=(
                    f"Stress triggers identified: {', '.join(triggers_found)}. "
                    "Here are targeted micro-interventions."
                ),
            )

        # ── Layer 3: Preventive Nudges ────────────────────────────────────────
        if is_high_volatility or stress_score >= 0.3:
            return RecommendationResult(
                layer=3,
                crisis_detected=False,
                triggers_found=[],
                recommendations=_PREVENTIVE_NUDGES[:3],
                emergency_resources=[],
                explanation=(
                    "Stress levels are elevated or showing early signs of increase. "
                    "Here are some preventive wellness nudges."
                ),
            )

        # No intervention needed
        return RecommendationResult(
            layer=0,
            crisis_detected=False,
            triggers_found=[],
            recommendations=[
                "✅ Stress levels look stable. Keep up your current routines!",
                "🌿 Remember: short breaks and regular movement are great preventive tools.",
            ],
            emergency_resources=[],
            explanation="Current stress levels are within a healthy range.",
        )
