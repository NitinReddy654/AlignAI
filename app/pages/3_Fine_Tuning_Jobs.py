"""Fine-Tuning Jobs - configure and launch training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import streamlit as st

from alignai.training.strategies import TrainingHyperparams

st.set_page_config(page_title="Fine-Tuning Jobs | AlignAI", layout="wide")
st.title("Fine-Tuning Jobs")
st.markdown("Configure and launch fine-tuning jobs using LoRA, QLoRA, or full fine-tuning.")

strategy = st.selectbox("Training Strategy", ["lora", "qlora", "full"])
dataset_path = st.text_input(
    "Dataset Path",
    value="data/samples/enterprise_support_dataset.jsonl",
)

col1, col2, col3 = st.columns(3)
with col1:
    epochs = st.number_input("Epochs", min_value=1, max_value=20, value=3)
with col2:
    batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=4)
with col3:
    lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=2e-4, format="%.6f")

with st.expander("Advanced LoRA Settings"):
    lora_r = st.number_input("LoRA Rank (r)", min_value=4, max_value=128, value=16)
    lora_alpha = st.number_input("LoRA Alpha", min_value=8, max_value=256, value=32)
    lora_dropout = st.slider("LoRA Dropout", 0.0, 0.5, 0.05)

st.markdown("---")
st.info(
    "Fine-tuning requires GPU resources. For production training, use the CLI:\n\n"
    f"```\npython scripts/run_finetune.py --strategy {strategy} "
    f"--dataset {dataset_path} --epochs {epochs}\n```"
)

if st.button("Preview Configuration"):
    hparams = TrainingHyperparams(
        strategy=strategy,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    st.json(hparams.to_dict())
