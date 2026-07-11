"""Retrieval-augmented evaluation utilities."""

from alignai.rag.augment import (
    build_rag_augmented_judge_messages,
    build_rag_prompt_response_pair,
    format_retrieved_context,
)
from alignai.rag.eval_rag import RAGEvaluationResult, evaluate_rag_context
from alignai.rag.retriever import (
    ChromaContextRetriever,
    LocalHashEmbeddingFunction,
    OpenAIEmbeddingFunction,
    RetrievalDocument,
    RetrievalResult,
)

__all__ = [
    "ChromaContextRetriever",
    "LocalHashEmbeddingFunction",
    "OpenAIEmbeddingFunction",
    "RAGEvaluationResult",
    "RetrievalDocument",
    "RetrievalResult",
    "build_rag_augmented_judge_messages",
    "build_rag_prompt_response_pair",
    "evaluate_rag_context",
    "format_retrieved_context",
]
