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
from dataclasses import dataclass, field
from typing import Literal

# ── Crisis patterns (Layer 1) ─────────────────────────────────────────────────

# Each pattern is a compiled regex.  Match is case-insensitive.
_CRISIS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(suicid(e|al|ally))\b", re.IGNORECASE),
    re.compile(r"\b(kill\s+(my)?self|end\s+(my\s+)?life)\b", re.IGNORECASE),
    re.compile(r"\b(want\s+to\s+die|wish\s+(i\s+)?was\s+dead)\b", re.IGNORECASE),
    re.compile(r"\b(self[\s-]?harm(ing)?|cutting\s+myself|hurt(ing)?\s+myself)\b", re.IGNORECASE),
    re.compile(r"\b(overdos(e|ing)|took\s+too\s+many\s+pills)\b", re.IGNORECASE),
    re.compile(r"\b(no\s+reason\s+to\s+live|life\s+is\s+not\s+worth)\b", re.IGNORECASE),
    re.compile(r"\b(don['']t\s+want\s+to\s+(be\s+here|exist|live))\b", re.IGNORECASE),
    re.compile(r"\b(hopeless|worthless|can['']t\s+go\s+on)\b", re.IGNORECASE),
]

_EMERGENCY_RESOURCES: list[dict] = [
    {"label": "National Suicide Prevention Lifeline (US)", "contact": "Call or text 988", "url": "https://988lifeline.org"},
    {"label": "International Crisis Centres", "contact": "iasp.info", "url": "https://www.iasp.info/resources/Crisis_Centres/"},
    {"label": "Crisis Text Line (US/UK/CA/IE)", "contact": "Text HOME to 741741", "url": "https://www.crisistextline.org"},
    {"label": "Samaritans (UK/IE)", "contact": "116 123 — free, 24/7", "url": "https://www.samaritans.org"},
    {"label": "Emergency Services", "contact": "911 / 999 / 112", "url": None},
]


# ── Structured Intervention ───────────────────────────────────────────────────

@dataclass
class Intervention:
    """A single, structured recommendation action.

    Attributes
    ----------
    title:
        Short category name shown as card header (e.g. "Sleep Hygiene").
    action:
        Concise action description (e.g. "4-7-8 Breathing").
    body:
        Full explanatory text shown inside the card.
    type:
        "interactive" – renders an animated breathing exercise in the UI.
        "behavioral"  – suggests a concrete physical/social action.
        "informational" – provides guidance or facts.
    """

    title: str
    action: str
    body: str
    type: Literal["interactive", "behavioral", "informational"]


# ── Context trigger patterns (Layer 2) ───────────────────────────────────────

@dataclass
class TriggerCategory:
    name: str
    patterns: list[re.Pattern]
    interventions: list[Intervention]


_TRIGGER_CATEGORIES: list[TriggerCategory] = [
    TriggerCategory(
        name="sleep",
        patterns=[
            re.compile(
                r"\b(can['']t\s+sleep|insomnia|sleep(less|ing\s+poorly)|awake\s+all\s+night|tired\s+all\s+day)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(no\s+sleep|bad\s+sleep|sleep\s+depriv(ed|ation))\b",
                re.IGNORECASE,
            ),
        ],
        interventions=[
            Intervention(
                title="Sleep Hygiene",
                action="4-7-8 Breathing",
                body="Inhale for 4 s → Hold for 7 s → Exhale slowly for 8 s. Repeat 4 cycles before bed. This activates your parasympathetic nervous system.",
                type="interactive",
            ),
            Intervention(
                title="Sleep Hygiene",
                action="Phone-Free Wind-Down",
                body="Begin a 30-minute phone-free routine 30 minutes before bed. Blue light suppresses melatonin by up to 50%.",
                type="behavioral",
            ),
            Intervention(
                title="Sleep Hygiene",
                action="Bedroom Temperature",
                body="Keep your bedroom at ~18 °C / 65 °F. Your core body temperature needs to drop ~1–2 °C for deep sleep to begin.",
                type="informational",
            ),
            Intervention(
                title="Sleep Hygiene",
                action="Brain Dump Journal",
                body="Write tomorrow's to-do list before bed to offload mental loops. Studies show this reduces sleep-onset time by ~9 minutes.",
                type="behavioral",
            ),
        ],
    ),
    TriggerCategory(
        name="exams",
        patterns=[
            re.compile(
                r"\b(exam|test|quiz|midterm|final|assignment|deadline|grade|study(ing)?)\b",
                re.IGNORECASE,
            ),
        ],
        interventions=[
            Intervention(
                title="Study Strategy",
                action="Pomodoro Technique",
                body="25 min focused study → 5 min break. After 4 cycles, take a 20-min break. This matches your brain's ultradian focus rhythm.",
                type="behavioral",
            ),
            Intervention(
                title="Exam Prep",
                action="Box Breathing",
                body="Before your exam: 4 s inhale → 4 s hold → 4 s exhale → 4 s hold. Clinically shown to lower cortisol and improve recall under pressure.",
                type="interactive",
            ),
            Intervention(
                title="Study Strategy",
                action="Priority Matrix",
                body="Sort topics by importance × difficulty. Tackle hardest material when your energy is highest (usually morning for most people).",
                type="informational",
            ),
            Intervention(
                title="Study Strategy",
                action="Pre-Study Walk",
                body="A 10-minute outdoor walk before studying boosts BDNF (brain-derived neurotrophic factor), improving memory consolidation by up to 20%.",
                type="behavioral",
            ),
        ],
    ),
    TriggerCategory(
        name="money",
        patterns=[
            re.compile(
                r"\b(debt|bill|rent|afford|broke|financial|money|loan|credit|budget)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(can['']t\s+pay|running\s+out\s+of\s+money|overdraft)\b",
                re.IGNORECASE,
            ),
        ],
        interventions=[
            Intervention(
                title="Financial Clarity",
                action="3-Bucket Expense List",
                body="List all expenses in three columns: Needs / Wants / Savings. Visibility alone reduces financial anxiety — the unknown is always scarier than the known.",
                type="behavioral",
            ),
            Intervention(
                title="Financial Support",
                action="Contact Your Lender",
                body="Most lenders have hardship or deferral programmes that are never advertised. A 5-minute phone call costs nothing and could unlock a payment pause.",
                type="informational",
            ),
            Intervention(
                title="Mindset Reset",
                action="Control Circle",
                body="Draw two circles: in the inner one write what you CAN control right now (one next action). In the outer one, park everything else. Focus only on the inner circle.",
                type="behavioral",
            ),
        ],
    ),
    TriggerCategory(
        name="work",
        patterns=[
            re.compile(
                r"\b(boss|manager|coworker|workplace|burnout|overwhelmed|overwork(ed)?|deadlin(e|es)|fired|laid\s+off)\b",
                re.IGNORECASE,
            ),
        ],
        interventions=[
            Intervention(
                title="Work Boundaries",
                action="Time-Blocking",
                body="Assign fixed time slots to tasks in your calendar. Treating tasks like meetings prevents context-switching and decision fatigue.",
                type="behavioral",
            ),
            Intervention(
                title="Work Boundaries",
                action="Micro-Break Protocol",
                body="Take a 5-minute break every 90 minutes. Stand, stretch, or look out a window. This matches the human ultradian rest cycle and lowers cortisol sustainably.",
                type="behavioral",
            ),
            Intervention(
                title="Workplace Communication",
                action="Assertive Boundary Script",
                body="Try: 'I can take this on — my capacity allows me to deliver by [date]. Does that work?' This is assertive (not aggressive) and sets clear expectations.",
                type="informational",
            ),
        ],
    ),
    TriggerCategory(
        name="relationships",
        patterns=[
            re.compile(
                r"\b(breakup|divorce|lonely|loneliness|isolated|argument|fight|relation|partner|friend)\b",
                re.IGNORECASE,
            ),
        ],
        interventions=[
            Intervention(
                title="Social Connection",
                action="One Outreach Message",
                body="Reach out to one trusted person today — even a short 'thinking of you' text. Social connection is the single strongest predictor of resilience.",
                type="behavioral",
            ),
            Intervention(
                title="Emotional Processing",
                action="15-Minute Journal",
                body="Write about the situation for 15 minutes without editing. Research by Pennebaker shows expressive writing reduces rumination and lowers cortisol.",
                type="behavioral",
            ),
            Intervention(
                title="Self-Compassion",
                action="Inner Friend Exercise",
                body="Imagine your closest friend is in exactly your situation. What would you tell them? Now say that to yourself. You deserve the same kindness.",
                type="informational",
            ),
        ],
    ),
]

# ── Preventive nudges (Layer 3) ───────────────────────────────────────────────

_PREVENTIVE_NUDGES: list[Intervention] = [
    Intervention(
        title="Mindfulness Check-In",
        action="1-Minute Body Scan",
        body="Your stress patterns show early volatility. Close your eyes for 60 seconds: notice your breath, relax your jaw, drop your shoulders.",
        type="interactive",
    ),
    Intervention(
        title="Movement Break",
        action="15-Minute Outdoor Walk",
        body="A short walk in nature reduces cortisol by up to 12% and activates your default mode network, helping your brain process background stress.",
        type="behavioral",
    ),
    Intervention(
        title="Hydration",
        action="Drink a Glass of Water",
        body="Even mild dehydration (1–2% body weight) measurably amplifies perceived stress. Hydration is the simplest, fastest stress modulator available.",
        type="behavioral",
    ),
    Intervention(
        title="Sleep Priority",
        action="Sleep Commitment",
        body="Aim for 7–9 hours tonight. Sleep is the single most powerful stress regulator — it resets the amygdala's threat-detection sensitivity overnight.",
        type="informational",
    ),
    Intervention(
        title="Gratitude Practice",
        action="5-Minute Gratitude Log",
        body="Write 3 specific things that went well today and why. Specificity is key — 'my friend texted me' activates stronger neural reward than 'I'm grateful for friends'.",
        type="behavioral",
    ),
]


# ── Main engine ───────────────────────────────────────────────────────────────

@dataclass
class RecommendationResult:
    """Structured output from the recommendation engine."""

    layer: int
    status: Literal["safe", "crisis", "nudge"]
    crisis_detected: bool
    triggers_found: list[str]
    interventions: list[Intervention]
    emergency_resources: list[dict]
    explanation: str

    # Backward-compat helpers
    @property
    def recommendations(self) -> list[str]:
        """Flat list of action descriptions (legacy interface)."""
        return [i.action for i in self.interventions]


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
                status="crisis",
                crisis_detected=True,
                triggers_found=[],
                interventions=[],
                emergency_resources=_EMERGENCY_RESOURCES,
                explanation=(
                    "Crisis signal detected. AI recommendations are paused. "
                    "Please reach out to one of the emergency resources below immediately."
                ),
            )

        # ── Layer 2: Context Matcher ──────────────────────────────────────────
        matched_categories = self.context_matcher(text)

        if matched_categories and (should_intervene or stress_score >= 0.5):
            triggers_found = [c.name for c in matched_categories]
            interventions: list[Intervention] = []
            for category in matched_categories:
                interventions.extend(category.interventions[:2])   # top 2 per category

            return RecommendationResult(
                layer=2,
                status="nudge",
                crisis_detected=False,
                triggers_found=triggers_found,
                interventions=interventions,
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
                status="nudge",
                crisis_detected=False,
                triggers_found=[],
                interventions=_PREVENTIVE_NUDGES[:3],
                emergency_resources=[],
                explanation=(
                    "Stress levels are elevated or showing early signs of increase. "
                    "Here are some preventive wellness nudges."
                ),
            )

        # ── No intervention needed ────────────────────────────────────────────
        return RecommendationResult(
            layer=0,
            status="safe",
            crisis_detected=False,
            triggers_found=[],
            interventions=[
                Intervention(
                    title="All Clear",
                    action="Keep Up Your Routines",
                    body="Stress levels look stable. Short breaks and regular movement are great preventive tools — keep going!",
                    type="informational",
                ),
            ],
            emergency_resources=[],
            explanation="Current stress levels are within a healthy range.",
        )
