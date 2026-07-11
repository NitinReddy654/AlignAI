"""Heuristic faithfulness and relevance scoring for retrieved context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from alignai.rag.retriever import RetrievalResult

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class RAGEvaluationResult:
    """Faithfulness and relevance scores for a RAG response."""

    relevance_score: float
    faithfulness_score: float
    groundedness_score: float
    context_count: int
    unsupported_terms: List[str]

    def to_dict(self) -> dict:
        return {
            "relevance_score": round(self.relevance_score, 2),
            "faithfulness_score": round(self.faithfulness_score, 2),
            "groundedness_score": round(self.groundedness_score, 2),
            "context_count": self.context_count,
            "unsupported_terms": self.unsupported_terms,
        }


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower())
        if token not in STOPWORDS
    }


def _context_text(context: Sequence[RetrievalResult] | Sequence[str]) -> str:
    parts = []
    for item in context:
        parts.append(item.text if isinstance(item, RetrievalResult) else str(item))
    return " ".join(parts)


def evaluate_rag_context(
    query: str,
    response: str,
    retrieved_context: Iterable[RetrievalResult] | Iterable[str],
) -> RAGEvaluationResult:
    """
    Score retrieval relevance and response faithfulness without an external judge.

    Scores are heuristic 0-100 signals intended for regression tests and
    lightweight monitoring. LLM-as-judge evaluation can be layered on top for
    deeper qualitative assessment.
    """
    context_items = list(retrieved_context)
    context = _context_text(context_items)
    query_tokens = _tokens(query)
    response_tokens = _tokens(response)
    context_tokens = _tokens(context)

    relevance = (
        len(query_tokens & context_tokens) / len(query_tokens) * 100 if query_tokens else 0.0
    )
    faithfulness = (
        len(response_tokens & context_tokens) / len(response_tokens) * 100
        if response_tokens
        else 0.0
    )
    unsupported = sorted(response_tokens - context_tokens)[:10]
    groundedness = (relevance * 0.4) + (faithfulness * 0.6)
    return RAGEvaluationResult(
        relevance_score=min(relevance, 100.0),
        faithfulness_score=min(faithfulness, 100.0),
        groundedness_score=min(groundedness, 100.0),
        context_count=len(context_items),
        unsupported_terms=unsupported,
    )
