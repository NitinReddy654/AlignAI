"""Human Review Center - pairwise preference collection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import random

import streamlit as st

from alignai.preference.collector import (
    calculate_preference_analytics,
    load_evaluation_results,
    prepare_comparison_pairs,
)

st.set_page_config(page_title="Human Review Center | AlignAI", layout="wide")
st.title("Human Review Center")
st.markdown("Compare model responses side-by-side and record human preference feedback.")

eval_file = st.text_input("Evaluation Results File", value="data/artifacts/evaluations/latest_eval.json")

if "comparison_pairs" not in st.session_state:
    st.session_state.comparison_pairs = []
    st.session_state.current_pair_index = 0
    st.session_state.votes = {}
    st.session_state.evaluation_complete = False

if st.button("Load Comparison Pairs"):
    results = load_evaluation_results(eval_file)
    if results:
        pairs = prepare_comparison_pairs(results)
        random.shuffle(pairs)
        st.session_state.comparison_pairs = pairs
        st.session_state.current_pair_index = 0
        st.session_state.votes = {}
        st.session_state.evaluation_complete = False
        st.success(f"Loaded {len(pairs)} comparison pairs")
    else:
        st.warning("No evaluation results found at specified path.")

if st.session_state.evaluation_complete:
    st.header("Review Complete")
    analytics = calculate_preference_analytics(st.session_state.votes)
    st.subheader("Preference Analytics")
    if analytics["summary"]:
        st.dataframe(analytics["summary"], use_container_width=True)
    if st.button("Start New Review"):
        for k in ["comparison_pairs", "current_pair_index", "votes", "evaluation_complete"]:
            st.session_state.pop(k, None)
        st.rerun()

elif st.session_state.comparison_pairs:
    idx = st.session_state.current_pair_index
    total = len(st.session_state.comparison_pairs)
    if idx >= total:
        st.session_state.evaluation_complete = True
        st.rerun()

    prompt, m1, r1, m2, r2 = st.session_state.comparison_pairs[idx]
    st.header(f"Comparison {idx + 1} of {total}")
    st.info(f"**Prompt:** {prompt}")

    models = [(m1, r1), (m2, r2)]
    random.shuffle(models)
    model_a, resp_a = models[0]
    model_b, resp_b = models[1]
    st.session_state.last_assignment = {"A": model_a, "B": model_b, "prompt": prompt}

    col1, col2 = st.columns(2)
    col1.markdown("**Response A**")
    col1.markdown(f"> {resp_a}")
    col2.markdown("**Response B**")
    col2.markdown(f"> {resp_b}")

    st.markdown("---")
    btn_cols = st.columns(3)

    def record_vote(choice):
        a = st.session_state.last_assignment
        vote_key = (min(a["A"], a["B"]), max(a["A"], a["B"]), a["prompt"])
        winner = a["A"] if choice == "A" else a["B"] if choice == "B" else "tie"
        st.session_state.votes[vote_key] = {"winner": winner}
        st.session_state.current_pair_index += 1
        st.rerun()

    btn_cols[0].button("Response A Preferred", on_click=record_vote, args=("A",), use_container_width=True)
    btn_cols[1].button("Tie / No Preference", on_click=record_vote, args=("Tie",), use_container_width=True)
    btn_cols[2].button("Response B Preferred", on_click=record_vote, args=("B",), use_container_width=True)

else:
    st.info("Load evaluation results to begin human preference review.")
