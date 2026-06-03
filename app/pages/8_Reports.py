"""Reports - exportable evaluation and alignment reports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import streamlit as st
from recommendation_ui import render_deployment_recommendation

from alignai.config import get_config
from alignai.experiments.recommendation import generate_recommendation_from_registry

st.set_page_config(page_title="Reports | AlignAI", layout="wide")
st.title("Evaluation Reports")
st.markdown("View, inspect, and export detailed evaluation and alignment reports.")

cfg = get_config()
reports_dir = cfg.artifacts_dir / "reports"

if reports_dir.exists():
    reports = sorted(reports_dir.glob("report_*.json"), reverse=True)
    if reports:
        selected = st.selectbox("Select Report", [r.name for r in reports])
        report = json.loads((reports_dir / selected).read_text())

        col1, col2, col3 = st.columns(3)
        col1.metric("Final Score", f"{report.get('final_score', 0):.2f}/5")
        readiness = report.get("alignment_readiness", {})
        col2.metric("Alignment Readiness", f"{readiness.get('alignment_readiness', 0)}/100")
        confidence = report.get("evaluation_confidence", {})
        col3.metric("Confidence", f"{confidence.get('confidence_score', 0)}/100")

        tab_breakdown, tab_reasoning, tab_examples, tab_decision, tab_export = st.tabs(
            [
                "Category Breakdown",
                "Judge Reasoning",
                "Examples",
                "Deployment Decision",
                "Export",
            ]
        )

        with tab_breakdown:
            categories = report.get("category_breakdown", {})
            if categories:
                means = {k: v.get("mean", 0) for k, v in categories.items()}
                st.bar_chart(means)
                st.json(categories)

        with tab_reasoning:
            samples = report.get("judge_reasoning_samples", [])
            for sample in samples:
                st.markdown(f"**Prompt:** {sample.get('prompt', '')[:100]}...")
                for key, val in sample.items():
                    if "justification" in key:
                        st.markdown(f"- *{key.replace('_', ' ').title()}*: {val}")

        with tab_examples:
            for ex in report.get("supporting_examples", []):
                st.markdown(f"**Q:** {ex.get('prompt', '')}")
                st.markdown(f"**A:** {ex.get('response', '')}")
                st.markdown("---")

        with tab_decision:
            rec = generate_recommendation_from_registry()
            render_deployment_recommendation(rec, show_export=True)

        with tab_export:
            export_payload = dict(report)
            rec = generate_recommendation_from_registry()
            if rec:
                export_payload["deployment_recommendation"] = rec.to_dict()
            st.download_button(
                "Download Report JSON (with deployment decision)",
                data=json.dumps(export_payload, indent=2),
                file_name=selected,
                mime="application/json",
            )
            if rec:
                st.download_button(
                    "Download Deployment Decision JSON only",
                    data=json.dumps(rec.to_dict(), indent=2),
                    file_name=f"deployment_recommendation_{rec.experiment_id}.json",
                    mime="application/json",
                    key="dl_rec_only",
                )
    else:
        st.info("No reports generated yet.")
else:
    st.info("Run the evaluation pipeline to generate reports.")
