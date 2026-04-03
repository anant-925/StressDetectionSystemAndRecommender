"""
tests/test_models.py

Unit tests for:
  - sliding_window_chunks (dataset.py)
  - MultichannelCNNWithAttention forward pass (model.py)
  - TemporalStressProfile (temporal_profile.py)
  - RecommendationEngine (recommendation_engine.py)
"""

from __future__ import annotations

import math
import time
import unittest

import torch

# ── Minimal tokenizer stub ─────────────────────────────────────────────────────

class _StubTokenizer:
    """Minimal tokenizer compatible with sliding_window_chunks."""

    pad_token_id = 0
    vocab_size = 1000

    def __call__(self, text: str, add_special_tokens: bool = True, return_tensors: str = "pt"):
        ids = [ord(c) % 998 + 1 for c in text.split()]   # word-level, non-zero
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestSlidingWindowChunks(unittest.TestCase):

    def setUp(self):
        self.tok = _StubTokenizer()

    def test_short_text_single_chunk(self):
        from src.models.dataset import sliding_window_chunks

        text = " ".join(["word"] * 10)   # 10 tokens < chunk_size=200
        chunks = sliding_window_chunks(text, self.tok, chunk_size=200, stride=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["input_ids"].shape, (200,))
        self.assertEqual(chunks[0]["attention_mask"].shape, (200,))

    def test_long_text_multiple_chunks(self):
        from src.models.dataset import sliding_window_chunks

        # 300 tokens → ceil((300 - 200) / 50) + 1 = 3 chunks
        text = " ".join(["word"] * 300)
        chunks = sliding_window_chunks(text, self.tok, chunk_size=200, stride=50)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(chunk["input_ids"].shape[0], 200)

    def test_empty_text_returns_one_chunk(self):
        from src.models.dataset import sliding_window_chunks

        chunks = sliding_window_chunks("", self.tok, chunk_size=200, stride=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["input_ids"].shape, (200,))

    def test_padding_is_applied(self):
        from src.models.dataset import sliding_window_chunks

        text = " ".join(["a"] * 5)   # 5 tokens, needs padding to 200
        chunks = sliding_window_chunks(text, self.tok, chunk_size=200, stride=50)
        self.assertEqual(chunks[0]["input_ids"].shape, (200,))
        # Only first 5 positions should be non-pad
        mask = chunks[0]["attention_mask"]
        self.assertEqual(mask[:5].sum().item(), 5)
        self.assertEqual(mask[5:].sum().item(), 0)


class TestMultichannelCNNWithAttention(unittest.TestCase):

    def setUp(self):
        from src.models.model import MultichannelCNNWithAttention

        self.model = MultichannelCNNWithAttention(
            vocab_size=1000,
            embed_dim=32,
            num_filters=16,
            kernel_sizes=(2, 3, 5),
            dropout=0.0,
        )
        self.model.eval()

    def test_output_shapes(self):
        batch, seq = 4, 50
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)

        self.assertEqual(output["logits"].shape, (batch, 1))
        self.assertEqual(output["probs"].shape, (batch, 1))
        self.assertEqual(len(output["attn_weights"]), 3)   # one per branch
        for w in output["attn_weights"]:
            self.assertEqual(w.shape, (batch, seq))

    def test_probs_in_unit_interval(self):
        input_ids = torch.randint(0, 1000, (8, 100))
        attention_mask = torch.ones(8, 100, dtype=torch.long)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)

        probs = output["probs"]
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_attention_weights_sum_to_one(self):
        input_ids = torch.randint(0, 1000, (2, 30))
        attention_mask = torch.ones(2, 30, dtype=torch.long)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)

        for branch_weights in output["attn_weights"]:
            row_sums = branch_weights.sum(dim=1)   # (batch,)
            for s in row_sums:
                self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_different_sequence_lengths(self):
        """Model must handle any sequence length without error."""
        for seq_len in [1, 5, 200, 512]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            attention_mask = torch.ones(2, seq_len, dtype=torch.long)
            with torch.no_grad():
                output = self.model(input_ids, attention_mask)
            self.assertEqual(output["probs"].shape, (2, 1))


class TestTemporalStressProfile(unittest.TestCase):

    def _make_profile(self, scores: list[float]):
        from src.temporal.temporal_profile import StressEvent, TemporalStressProfile
        profile = TemporalStressProfile(velocity_window=3, adaptive_k=1.5)
        for i, s in enumerate(scores):
            profile.add_event(StressEvent(timestamp=float(i), score=s))
        return profile

    def test_empty_profile_defaults(self):
        from src.temporal.temporal_profile import TemporalStressProfile
        p = TemporalStressProfile()
        self.assertIsNone(p.current_score())
        self.assertEqual(p.stress_velocity(), 0.0)
        self.assertAlmostEqual(p.adaptive_threshold(), 0.7)
        self.assertFalse(p.should_intervene())

    def test_stress_velocity_positive(self):
        # Monotonically increasing scores → velocity should be positive
        p = self._make_profile([0.1, 0.2, 0.4, 0.7, 0.9])
        self.assertGreater(p.stress_velocity(), 0)

    def test_stress_velocity_negative(self):
        # Monotonically decreasing scores → velocity should be negative
        p = self._make_profile([0.9, 0.7, 0.4, 0.2, 0.1])
        self.assertLess(p.stress_velocity(), 0)

    def test_adaptive_threshold_above_mean(self):
        scores = [0.3, 0.4, 0.5, 0.6, 0.7]
        p = self._make_profile(scores)
        import numpy as np
        expected = float(np.mean(scores)) + 1.5 * float(np.std(scores, ddof=1))
        self.assertAlmostEqual(p.adaptive_threshold(), min(expected, 1.0), places=5)

    def test_adaptive_threshold_clamped(self):
        # Extremely high scores – threshold should not exceed 1.0
        p = self._make_profile([0.95, 0.97, 0.99, 1.0, 1.0])
        self.assertLessEqual(p.adaptive_threshold(), 1.0)

    def test_window_size_respected(self):
        from src.temporal.temporal_profile import StressEvent, TemporalStressProfile
        p = TemporalStressProfile(window_size=5)
        for i in range(10):
            p.add_event(StressEvent(timestamp=float(i), score=float(i) / 10))
        self.assertEqual(len(p.get_scores()), 5)

    def test_summary_keys(self):
        p = self._make_profile([0.4, 0.6, 0.5])
        summary = p.summary()
        for key in ("n_events", "current_score", "stress_velocity",
                    "adaptive_threshold", "should_intervene", "is_high_volatility",
                    "history_scores", "history_timestamps"):
            self.assertIn(key, summary)


class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        from src.recommender.recommendation_engine import RecommendationEngine
        self.engine = RecommendationEngine()

    def test_circuit_breaker_triggers(self):
        crisis_texts = [
            "I want to kill myself",
            "I am thinking about suicide",
            "I've been self-harming",
            "I took too many pills",
        ]
        for text in crisis_texts:
            result = self.engine.recommend(text, stress_score=0.9)
            self.assertTrue(result.crisis_detected, f"Expected crisis for: {text!r}")
            self.assertEqual(result.layer, 1)
            self.assertGreater(len(result.emergency_resources), 0)

    def test_circuit_breaker_not_triggered_for_normal_text(self):
        result = self.engine.recommend("I feel a bit tired today", stress_score=0.5)
        self.assertFalse(result.crisis_detected)

    def test_context_matcher_sleep(self):
        result = self.engine.recommend(
            "I can't sleep because of insomnia and I am exhausted",
            stress_score=0.7,
            should_intervene=True,
        )
        self.assertEqual(result.layer, 2)
        self.assertIn("sleep", result.triggers_found)
        self.assertGreater(len(result.recommendations), 0)

    def test_context_matcher_exams(self):
        result = self.engine.recommend(
            "My exam deadline is tomorrow and I haven't started studying",
            stress_score=0.8,
            should_intervene=True,
        )
        self.assertEqual(result.layer, 2)
        self.assertIn("exams", result.triggers_found)

    def test_context_matcher_money(self):
        result = self.engine.recommend(
            "I can't pay my rent and I am drowning in debt",
            stress_score=0.75,
            should_intervene=True,
        )
        self.assertEqual(result.layer, 2)
        self.assertIn("money", result.triggers_found)

    def test_preventive_nudges_high_volatility(self):
        result = self.engine.recommend(
            "I had a decent day overall",
            stress_score=0.35,
            is_high_volatility=True,
        )
        self.assertEqual(result.layer, 3)
        self.assertGreater(len(result.recommendations), 0)

    def test_no_intervention_needed(self):
        result = self.engine.recommend(
            "Had a great walk in the park, feeling good",
            stress_score=0.1,
            is_high_volatility=False,
            should_intervene=False,
        )
        self.assertEqual(result.layer, 0)
        self.assertFalse(result.crisis_detected)

    def test_circuit_breaker_halts_other_layers(self):
        # Even with context triggers, crisis must take priority
        result = self.engine.recommend(
            "I have an exam tomorrow and I want to end my life",
            stress_score=0.95,
        )
        self.assertTrue(result.crisis_detected)
        self.assertEqual(result.layer, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
