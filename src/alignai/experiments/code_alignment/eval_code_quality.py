"""Heuristic code-generation quality evaluation."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class CodeQualityReport:
    """Structured code-generation evaluation output."""

    language: str
    functional_correctness: float
    security: float
    readability: float
    efficiency: float
    overall_score: float
    findings: List[str]

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "functional_correctness": round(self.functional_correctness, 2),
            "security": round(self.security, 2),
            "readability": round(self.readability, 2),
            "efficiency": round(self.efficiency, 2),
            "overall_score": round(self.overall_score, 2),
            "findings": self.findings,
        }


SECURITY_PATTERNS = {
    "eval(": "Dynamic eval can execute untrusted code.",
    "exec(": "Dynamic exec can execute untrusted code.",
    "subprocess.call(": "Subprocess calls require command sanitization.",
    "shell=True": "shell=True can enable command injection.",
    "pickle.loads(": "pickle.loads can execute untrusted payloads.",
}

INEFFICIENCY_PATTERNS = {
    r"for\s+\w+\s+in\s+range\(len\(": "Index-based loops can often be simplified.",
    r"\.append\(.*\)\s*\n\s*return\s+\[": "Loop construction may be replaceable with a comprehension.",
}


def load_code_rubric(path: str | Path) -> Dict[str, Any]:
    """Load a YAML code-alignment rubric."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise RuntimeError("PyYAML is required to load code_rubric.yaml.") from exc

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _syntax_score(code: str, language: str, findings: List[str]) -> float:
    if language.lower() != "python":
        return 85.0 if code.strip() else 0.0
    try:
        ast.parse(code)
        return 100.0
    except SyntaxError as exc:
        findings.append(f"Python syntax error: {exc.msg}")
        return 40.0


def _security_score(code: str, findings: List[str]) -> float:
    score = 100.0
    for pattern, message in SECURITY_PATTERNS.items():
        if pattern in code:
            findings.append(message)
            score -= 25.0
    return max(score, 0.0)


def _readability_score(code: str, findings: List[str]) -> float:
    lines = [line for line in code.splitlines() if line.strip()]
    if not lines:
        findings.append("No generated code provided.")
        return 0.0

    long_lines = sum(1 for line in lines if len(line) > 100)
    comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
    score = 100.0 - min(long_lines * 5.0, 30.0)
    if len(lines) >= 10 and comment_lines == 0:
        findings.append("Non-trivial code has no comments or documentation.")
        score -= 10.0
    return max(score, 0.0)


def _efficiency_score(code: str, findings: List[str]) -> float:
    score = 100.0
    for pattern, message in INEFFICIENCY_PATTERNS.items():
        if re.search(pattern, code):
            findings.append(message)
            score -= 10.0
    nested_loops = len(re.findall(r"for .*:\n\s+for .*:", code))
    if nested_loops:
        findings.append("Nested loops detected; verify complexity for large inputs.")
        score -= min(20.0, nested_loops * 10.0)
    return max(score, 0.0)


def evaluate_code_quality(code: str, language: str = "python") -> CodeQualityReport:
    """
    Evaluate generated code for correctness, security, readability, and efficiency.
    """
    findings: List[str] = []
    correctness = _syntax_score(code, language, findings)
    security = _security_score(code, findings)
    readability = _readability_score(code, findings)
    efficiency = _efficiency_score(code, findings)
    overall = (
        correctness * 0.35
        + security * 0.30
        + readability * 0.20
        + efficiency * 0.15
    )
    return CodeQualityReport(
        language=language,
        functional_correctness=correctness,
        security=security,
        readability=readability,
        efficiency=efficiency,
        overall_score=overall,
        findings=findings,
    )
