"""AlignAI Streamlit Platform - Home Page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

st.set_page_config(
    page_title="AlignAI Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AlignAI Platform")
st.subheader("Generative AI Evaluation, Alignment, and Enablement")

st.markdown("""
**AlignAI** helps AI enablement teams fine-tune open-source LLMs, evaluate model
alignment, compare candidate variants, and produce deployment-ready evidence.
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Training Strategies", "3", help="Full fine-tuning, LoRA, QLoRA")
with col2:
    st.metric("Evaluation Categories", "8", help="Correctness through Tone Alignment")
with col3:
    st.metric("Readiness Score", "0-100", help="Composite deployment readiness metric")
with col4:
    st.metric("Decision Signals", "9", help="Quality, safety, cost, latency, preference, and confidence")

st.markdown("---")

st.markdown("### Platform Modules")

modules = [
    ("Dataset Management", "Upload, inspect, and analyze training datasets"),
    ("Experiment Tracking", "Monitor fine-tuning experiments and hyperparameters"),
    ("Fine-Tuning Jobs", "Launch and configure LoRA, QLoRA, or full fine-tuning"),
    ("Evaluation Center", "Run automated LLM-as-a-Judge evaluations"),
    ("Alignment Dashboard", "Review readiness scores, risk signals, and improvement actions"),
    ("Leaderboards", "Compare models by quality, cost, and preference"),
    ("Human Review Center", "Collect pairwise human preference feedback"),
    ("Reports", "Generate and export evaluation reports"),
]

for name, desc in modules:
    st.markdown(f"**{name}** - {desc}")

st.markdown("---")
st.caption("AlignAI v1.0.0 | Author: Nitin Reddy Bommidi | Apache 2.0 License")
