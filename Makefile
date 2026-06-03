.PHONY: install install-dev test lint run-app analyze finetune evaluate clean

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest tests/ -v --cov=alignai --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/ app/

run-app:
	streamlit run app/Home.py

analyze:
	python scripts/analyze_dataset.py --dataset data/samples/enterprise_support_dataset.jsonl

finetune:
	python scripts/run_finetune.py --strategy lora --dataset data/samples/enterprise_support_dataset.jsonl

evaluate:
	python scripts/run_evaluation.py --experiment-id latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
