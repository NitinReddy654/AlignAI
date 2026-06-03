"""Experiment Tracking - monitor fine-tuning experiments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import streamlit as st

from alignai.experiments.registry import ExperimentRegistry

st.set_page_config(page_title="Experiment Tracking | AlignAI", layout="wide")
st.title("Experiment Tracking")
st.markdown("Monitor fine-tuning experiments, hyperparameters, and training metadata.")

registry = ExperimentRegistry()
experiments = registry.list_all()

if not experiments:
    st.info("No experiments recorded yet. Launch a fine-tuning job from the Fine-Tuning Jobs page.")
else:
    st.metric("Total Experiments", len(experiments))
    for exp in reversed(experiments):
        with st.expander(f"{exp.experiment_id} - {exp.strategy.upper()} - {exp.status}"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Duration", f"{exp.duration_seconds:.0f}s")
            col2.metric("Cost Est.", f"${exp.cost_estimate.get('total_cost_usd', 0):.4f}")
            col3.metric("Model", exp.model_version.split("/")[-1])
            st.subheader("Hyperparameters")
            st.json(exp.hyperparams)
            if exp.evaluation_scores:
                st.subheader("Evaluation Scores")
                st.json(exp.evaluation_scores)

    st.markdown("---")
    st.subheader("Side-by-Side Comparison")
    if len(experiments) >= 2:
        exp_ids = [e.experiment_id for e in experiments]
        selected = st.multiselect("Select experiments to compare", exp_ids, default=exp_ids[:2])
        for eid in selected:
            exp = registry.load(eid)
            if exp:
                st.markdown(f"**{eid}** ({exp.strategy})")
                st.json({"hyperparams": exp.hyperparams, "cost": exp.cost_estimate, "scores": exp.evaluation_scores})
