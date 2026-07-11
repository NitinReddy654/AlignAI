"""ChromaDB-backed retrieval for alignment evaluation context."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class RetrievalDocument:
    """Document chunk stored for retrieval-augmented evaluation."""

    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """Single retrieved context chunk."""

    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata,
        }


class LocalHashEmbeddingFunction:
    """
    Deterministic local embedding function for offline tests and demo mode.

    ChromaDB accepts callable embedding functions. This implementation hashes
    tokens into a fixed-size vector so retrieval can run without external API
    calls while production deployments can swap in OpenAI embeddings.
    """

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002 - Chroma API name
        return [self.embed(text) for text in input]

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OpenAIEmbeddingFunction:
    """OpenAI embedding adapter compatible with ChromaDB embedding functions."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self.client = client

    def __call__(self, input: Sequence[str]) -> List[List[float]]:  # noqa: A002 - Chroma API name
        response = self.client.embeddings.create(model=self.model, input=list(input))
        return [item.embedding for item in response.data]


class ChromaContextRetriever:
    """Thin wrapper around ChromaDB for semantic alignment-context retrieval."""

    def __init__(
        self,
        collection_name: str = "alignai_context",
        persist_directory: Optional[str | Path] = None,
        embedding_function: Optional[Any] = None,
        client: Optional[Any] = None,
    ):
        self.collection_name = collection_name
        self.embedding_function = embedding_function or LocalHashEmbeddingFunction()

        if client is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover - exercised when dependency is missing
                raise RuntimeError(
                    "ChromaDB is required for ChromaContextRetriever. "
                    "Install with `pip install chromadb` or pass a compatible client."
                ) from exc

            if persist_directory:
                client = chromadb.PersistentClient(path=str(persist_directory))
            else:
                client = chromadb.Client()

        self.client = client
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def add_documents(self, documents: Iterable[RetrievalDocument]) -> int:
        """Add document chunks to the vector store."""
        docs = list(documents)
        if not docs:
            return 0

        self.collection.add(
            ids=[doc.document_id for doc in docs],
            documents=[doc.text for doc in docs],
            metadatas=[doc.metadata for doc in docs],
        )
        return len(docs)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve the top-k relevant context chunks for a query."""
        response = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filters,
        )
        ids = response.get("ids", [[]])[0]
        texts = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0] or [{} for _ in ids]
        distances = response.get("distances", [[]])[0] or [0.0 for _ in ids]

        results = []
        for doc_id, text, metadata, distance in zip(ids, texts, metadatas, distances):
            score = max(0.0, 1.0 - float(distance))
            results.append(
                RetrievalResult(
                    document_id=doc_id,
                    text=text,
                    score=score,
                    metadata=metadata or {},
                )
            )
        return results
