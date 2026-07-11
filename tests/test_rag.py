"""Tests for RAG retrieval and grounding utilities."""

from alignai.rag.augment import build_rag_augmented_judge_messages, format_retrieved_context
from alignai.rag.eval_rag import evaluate_rag_context
from alignai.rag.retriever import LocalHashEmbeddingFunction, RetrievalResult


def test_local_hash_embedding_is_normalized():
    emb = LocalHashEmbeddingFunction(dimensions=32).embed("alignment safety safety")
    assert len(emb) == 32
    assert abs(sum(v * v for v in emb) - 1.0) < 1e-6


def test_format_retrieved_context_includes_metadata():
    result = RetrievalResult("doc1", "Safety policy context", 0.91, {"source": "policy.md"})
    block = format_retrieved_context([result])
    assert "policy.md" in block
    assert "Safety policy context" in block


def test_rag_context_scores_grounded_response():
    context = [RetrievalResult("doc1", "Password resets require MFA verification.", 0.9)]
    report = evaluate_rag_context(
        "How should password resets be handled?",
        "Password resets require MFA verification.",
        context,
    )
    assert report.relevance_score > 0
    assert report.faithfulness_score > 0
    assert report.context_count == 1


def test_augmented_judge_messages_include_context():
    context = [RetrievalResult("doc1", "Enterprise plans support SSO.", 0.8)]
    messages = build_rag_augmented_judge_messages(
        "Does Enterprise support SSO?",
        "Yes, Enterprise supports SSO.",
        context,
    )
    assert len(messages) == 2
    assert "Retrieved Context" in messages[1]["content"]
