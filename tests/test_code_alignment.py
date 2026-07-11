"""Tests for code-generation alignment evaluation."""

from alignai.experiments.code_alignment import evaluate_code_quality


def test_valid_python_scores_high():
    report = evaluate_code_quality("def add(a, b):\n    return a + b\n")
    assert report.functional_correctness == 100
    assert report.overall_score > 80


def test_syntax_error_is_flagged():
    report = evaluate_code_quality("def broken(:\n    pass\n")
    assert report.functional_correctness < 100
    assert any("syntax" in finding.lower() for finding in report.findings)


def test_security_issue_reduces_score():
    report = evaluate_code_quality("def run(x):\n    return eval(x)\n")
    assert report.security < 100
    assert any("eval" in finding.lower() for finding in report.findings)
