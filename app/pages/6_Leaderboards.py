"""Leaderboards - model comparison and ranking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from recommendation_ui import render_deployment_recommendation

from alignai.experiments.leaderboard import build_leaderboard, get_best_models

st.set_page_config(page_title="Leaderboards | AlignAI", layout="wide")
st.title("Experiment Leaderboard")
st.markdown("Compare models by alignment score, quality, cost, latency, and human preference.")

st.markdown("---")
render_deployment_recommendation()
st.markdown("---")

sort_by = st.selectbox(
    "Sort By",
    ["alignment_score", "avg_judge_score", "human_win_rate", "cost_usd", "latency_ms"],
)

df = build_leaderboard(sort_by=sort_by)

if df.empty:
    st.info("No experiments with evaluation data. Complete fine-tuning and evaluation first.")
else:
    st.dataframe(df, use_container_width=True)

    best = get_best_models(df)
    if best:
        st.markdown("---")
        st.subheader("Category Leaders")
        cols = st.columns(len(best))
        labels = {
            "best_alignment": "Best Alignment",
            "best_judge_score": "Highest Quality",
            "lowest_cost": "Lowest Cost",
            "fastest_inference": "Fastest",
            "best_human_preference": "Best Preference",
        }
        for i, (key, label) in enumerate(labels.items()):
            if key in best and best[key]:
                cols[i].metric(label, best[key].get("experiment_id", "N/A")[:12])
