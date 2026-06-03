"""Alignment Dashboard - readiness scores and improvement actions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import json

import streamlit as st

from alignai.config import get_config

st.set_page_config(page_title="Alignment Dashboard | AlignAI", layout="wide")
st.title("Alignment Dashboard")
st.markdown("View alignment readiness scores, score breakdowns, and improvement actions.")

cfg = get_config()
reports_dir = cfg.artifacts_dir / "reports"

if reports_dir.exists():
    reports = list(reports_dir.glob("report_*.json"))
    if reports:
        selected = st.selectbox("Select Report", [r.name for r in reports])
        report = json.loads((reports_dir / selected).read_text())
        readiness = report.get("alignment_readiness", {})
        score = readiness.get("alignment_readiness", 0)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Alignment Readiness", f"{score}/100")
            if score >= 80:
                st.success("Strong alignment - suitable for staged deployment")
            elif score >= 60:
                st.warning("Moderate alignment - address gaps before rollout")
            else:
                st.error("Low alignment - significant improvements needed")

        with col2:
            breakdown = readiness.get("breakdown", {})
            if breakdown:
                st.subheader("Score Breakdown")
                st.bar_chart(breakdown)

        confidence = report.get("evaluation_confidence", {})
        if confidence:
            st.markdown("---")
            st.subheader("Evaluation Confidence")
            st.metric("Confidence Score", f"{confidence.get('confidence_score', 0)}/100")
            st.caption(confidence.get("disclaimer", ""))
            if confidence.get("supporting_evidence"):
                st.markdown("**Supporting Evidence:**")
                for ev in confidence["supporting_evidence"]:
                    st.markdown(f"- {ev}")

        recs = readiness.get("recommendations", [])
        if recs:
            st.markdown("---")
            st.subheader("Improvement Actions")
            for rec in recs:
                st.markdown(f"- {rec}")
    else:
        st.info("No alignment reports available. Run an evaluation first.")
else:
    st.info("Reports directory not found. Run evaluation pipeline to generate reports.")
