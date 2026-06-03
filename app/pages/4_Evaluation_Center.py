"""Evaluation Center - automated LLM-as-a-Judge evaluations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import streamlit as st
from recommendation_ui import render_deployment_recommendation

from alignai.config import get_config
from alignai.evaluation.rubrics import EVALUATION_CATEGORIES
from alignai.experiments.registry import ExperimentRegistry

st.set_page_config(page_title="Evaluation Center | AlignAI", layout="wide")
st.title("Evaluation Center")
st.markdown(
    "Run automated LLM-as-a-Judge evaluations with structured scoring rubrics. "
    "After evaluations complete, the deployment decision engine compares all variants."
)

cfg = get_config()
registry = ExperimentRegistry()
experiments = registry.list_all()

st.subheader("Evaluation Categories")
cols = st.columns(4)
for i, cat in enumerate(EVALUATION_CATEGORIES):
    cols[i % 4].markdown(f"**{cat.replace('_', ' ').title()}**")

st.markdown("---")

if experiments:
    exp = st.selectbox("Select Experiment", [e.experiment_id for e in experiments])
    st.info(
        f"Run evaluation via CLI:\n\n"
        f"```\npython scripts/run_evaluation.py --experiment-id {exp}\n```"
    )

eval_dir = cfg.artifacts_dir / "evaluations"
if eval_dir.exists():
    eval_files = list(eval_dir.glob("*_eval.json"))
    if eval_files:
        st.subheader("Recent Evaluations")
        selected = st.selectbox("Evaluation file", [f.name for f in eval_files])
        data = json.loads((eval_dir / selected).read_text())
        metrics = data.get("aggregated_metrics", {})
        if "avg_judge_score" in metrics:
            st.metric("Average Judge Score", f"{metrics['avg_judge_score']:.2f}/5")
        if "categories" in metrics:
            cat_data = {k: v.get("mean", 0) for k, v in metrics["categories"].items()}
            st.bar_chart(cat_data)
else:
    st.info("No evaluations found. Run an evaluation job first.")

st.markdown("---")
st.subheader("Deployment Decision Engine")
render_deployment_recommendation()
