"""Dataset Management - upload and analyze training data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import json

import streamlit as st

from alignai.datasets_analysis.analyzer import analyze_dataset, save_health_report

st.set_page_config(page_title="Dataset Management | AlignAI", layout="wide")
st.title("Dataset Management")
st.markdown("Upload, inspect, and analyze training datasets for fine-tuning readiness.")

uploaded = st.file_uploader("Upload JSONL dataset", type=["jsonl", "json"])
sample_path = Path(__file__).resolve().parent.parent.parent / "data" / "samples" / "enterprise_support_dataset.jsonl"

tab_upload, tab_sample, tab_report = st.tabs(["Upload & Analyze", "Sample Dataset", "Health Reports"])

with tab_upload:
    if uploaded:
        temp_path = Path("/tmp") / uploaded.name
        temp_path.write_bytes(uploaded.read())
        report = analyze_dataset(temp_path)
        st.success(f"Analyzed **{report.total_records}** records")
        col1, col2, col3 = st.columns(3)
        col1.metric("Health Score", f"{report.health_score}/100")
        col2.metric("Avg Turns", f"{report.avg_turns_per_conversation:.1f}")
        col3.metric("Duplicates", report.duplicate_count)
        st.json(report.to_dict())
        if st.button("Save Health Report"):
            path = save_health_report(report)
            st.info(f"Report saved: {path}")

with tab_sample:
    st.markdown("**Enterprise Support Dataset** - synthetic training data for support assistant fine-tuning.")
    if sample_path.exists():
        report = analyze_dataset(sample_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Records", report.total_records)
        col2.metric("Health Score", f"{report.health_score}/100")
        col3.metric("Est. Avg Tokens", report.token_distribution.get("avg_tokens", 0))
        st.subheader("Role Distribution")
        st.bar_chart(report.role_distribution)
        if report.quality_issues:
            st.warning(f"{len(report.quality_issues)} quality issues detected")
            for issue in report.quality_issues[:5]:
                st.text(f"  - {issue}")

with tab_report:
    reports_dir = Path(__file__).resolve().parent.parent.parent / "data" / "artifacts" / "datasets"
    if reports_dir.exists():
        reports = list(reports_dir.glob("*_health.json"))
        if reports:
            selected = st.selectbox("Select report", [r.name for r in reports])
            data = json.loads((reports_dir / selected).read_text())
            st.json(data)
        else:
            st.info("No health reports generated yet. Analyze a dataset first.")
    else:
        st.info("No reports directory found.")
