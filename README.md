# StressDetectionSystemAndRecommender
🛡️ Resilience-First AI System for Cross-Platform Stress Detection

Welcome to the Resilience-First AI System, a production-grade, full-stack Machine Learning product designed to detect psychological stress from social media text and—more importantly—provide safe, context-aware interventions.

This project goes beyond standard classification by reproducing a core NLP research paper (Multichannel CNN + Attention) and extending it with Temporal Modeling and a Safety-First Recommendation Engine.

🎯 System Philosophy

Detection is a signal, not the end goal. Standard stress detection systems simply flag users as "stressed" or "not stressed," often leading to alert fatigue or irresponsible AI interactions. This system shifts the paradigm toward meaningful, safe, and personalized resilience support.

🧠 How It Works (System Architecture)

The system operates in four distinct phases to process user inputs, analyze context, and provide interventions safely:

1. The Sensor (Context-Aware ML Detection)

At the core of the system is the NLP pipeline, designed to handle both short-form text (like Twitter) and long-form text (like Reddit):

Architecture: A PyTorch-based MultichannelCNNWithAttention model featuring parallel Conv1D layers (kernels 2, 3, 5) to capture diverse linguistic patterns.

Sliding-Window Chunking: Instead of truncating long Reddit posts at 512 tokens, the system chunks long documents and aggregates attention scores across the entire post, ensuring no context is lost.

2. Temporal Modeling (The Innovation)

Stress is not an isolated event; it is a trend over time.

Stress Velocity ($V_s$): The system calculates the moving average derivative over a user's recent posts to differentiate between a "bad day" and a "chronic burnout trajectory."

Adaptive Thresholding: Instead of a rigid 0.5 probability trigger, the system computes an adaptive baseline ($\mu + 1.5\sigma$) for each user. Interventions are only triggered when stress significantly deviates from their personal norm, drastically reducing alert fatigue.

3. Safety-First Recommendation Engine

If an intervention is triggered, the text passes through a strict 3-Layer routing architecture:

Layer 1: The Circuit Breaker (Critical Safety): A strict, rule-based keyword matcher. If severe crisis or self-harm is detected, the AI immediately halts all generation and presents emergency lifelines and resources.

Layer 2: The Context Matcher: If no crisis is detected, the system extracts root triggers (e.g., Sleep, Exams, Finances) and maps them to proven micro-interventions (e.g., 4-7-8 breathing for sleep anxiety).

Layer 3: Preventive Nudges: If baseline stress is low but volatility is high, the system proactively suggests gentle grounding exercises.

4. Full-Stack Explainability UI

The final output is rendered via a Streamlit dashboard intended for real-time benchmarking and user trust:

Attention Heatmaps: Visually highlights the specific words that triggered the stress classification.

Temporal Graphs: Interactive Plotly charts mapping the user's Stress Velocity and Adaptive Thresholds.

Human-in-the-Loop Feedback: Users can flag false positives, which the system uses to dynamically adjust their personal threshold ($\mu$).

🧰 Tech Stack

Deep Learning Core: PyTorch (v2.0+), Transformers (HuggingFace)

Data Processing: Pandas, NumPy, KaggleHub

NLP & Extraction: Scikit-Learn, KeyBERT, NLTK, Regex

Backend & UI: FastAPI (Model Serving), Streamlit (Dashboard), Plotly (Visualizations)

Environment: Python 3.10+, venv

📊 Datasets

The model is trained on a unified, multi-domain dataset to ensure robust generalization:

Dreaddit: Long-form Reddit posts for contextual stress analysis.

Behavioural Tweets: Short-form, noisy Twitter data.

Suicide Watch Dataset: Specifically used to train the Layer 1 Circuit Breaker.

Emotions NLP Dataset: Used to profile distinct underlying emotions (fear, anger, sadness) for the Context Matcher.

🚀 Future Roadmap & Usage

This README represents the end-state of the project blueprint. Setup scripts and API endpoints will be added as the phased implementation progresses.
