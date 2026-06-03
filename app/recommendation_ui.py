"""Shared Streamlit UI for deployment decision support."""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st

from alignai.experiments.recommendation import (
    RECOMMENDATION_DISCLAIMER,
    DeploymentRecommendation,
    generate_recommendation_from_registry,
    save_recommendation_report,
)


def render_deployment_recommendation(
    recommendation: Optional[DeploymentRecommendation] = None,
    show_export: bool = True,
) -> Optional[DeploymentRecommendation]:
    """Render deployment decision panel; generate if not provided."""
    if recommendation is None:
        recommendation = generate_recommendation_from_registry()

    st.caption(RECOMMENDATION_DISCLAIMER)

    if recommendation is None:
        st.info(
            "No evaluated experiments available for comparison. "
            "Complete evaluations for Base, LoRA, QLoRA, and/or Full Fine-Tuning variants first."
        )
        return None

    st.subheader("Deployment Decision Support")
    st.markdown(f"**Leading Candidate:** `{recommendation.recommended_model}`")

    st.markdown("**Decision Basis:**")
    for reason in recommendation.reasons:
        st.markdown(f"- {reason}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Supporting Evidence**")
        for ev in recommendation.supporting_evidence:
            st.markdown(f"- {ev}")
    with col2:
        st.markdown("**Tradeoff Analysis**")
        for t in recommendation.tradeoff_analysis:
            st.markdown(f"- {t}")

    if recommendation.alternative_recommendations:
        st.markdown("**Alternative Candidate Fit**")
        for alt in recommendation.alternative_recommendations:
            when = "; ".join(alt.preferred_when)
            st.markdown(f"- **{alt.display_name}** - strongest when: {when}")

    if recommendation.deployment_warnings:
        st.markdown("**Deployment Gates**")
        for w in recommendation.deployment_warnings:
            st.warning(w)

    if recommendation.candidate_rankings:
        st.markdown("**Candidate Rankings**")
        st.dataframe(recommendation.candidate_rankings, use_container_width=True)

    if show_export:
        if st.button("Export Deployment Decision JSON", key="export_rec_btn"):
            path = save_recommendation_report(recommendation)
            st.success(f"Saved to {path}")
        st.download_button(
            label="Download Deployment Decision JSON",
            data=json.dumps(recommendation.to_dict(), indent=2),
            file_name=f"deployment_recommendation_{recommendation.experiment_id}.json",
            mime="application/json",
            key="dl_rec_json",
        )

    return recommendation
