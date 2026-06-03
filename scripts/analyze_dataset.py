#!/usr/bin/env python3
"""CLI entrypoint for dataset analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alignai.datasets_analysis.analyzer import analyze_dataset, save_health_report
from alignai.logging_utils import setup_logging

logger = setup_logging("alignai.analyze")


def main() -> None:
    parser = argparse.ArgumentParser(description="AlignAI Dataset Analyzer")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", default=None, help="Output directory for health report")
    args = parser.parse_args()

    logger.info("Analyzing dataset: %s", args.dataset)
    report = analyze_dataset(args.dataset)
    out_dir = Path(args.output) if args.output else None
    report_path = save_health_report(report, out_dir)
    print(json.dumps({"report_path": str(report_path), "health": report.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
