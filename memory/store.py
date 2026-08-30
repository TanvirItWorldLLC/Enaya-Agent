from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
import tiktoken

logger = structlog.get_logger()


class EmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbeddingProvider:
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small") -> None:
        self.api_key = api_key or __import__("os").environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None
        if self.api_key:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key)
        logger.info("openai_embedding_initialized", model=model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._client:
            import hashlib

            return [
                [
                    int(hashlib.md5(t.encode()).hexdigest(), 16) % 10000 / 10000.0
                    for _ in range(1536)
                ]
                for t in texts
            ]
        try:
            response = await self._client.embeddings.create(model=self.model, input=texts)
            return [e.embedding for e in response.data]
        except Exception as e:
            logger.error("embedding_error", error=str(e))
            raise


@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def token_count(self) -> int:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(self.content))
        except Exception:
            return len(self.content.split())


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class ChromaBackend:
    def __init__(
        self, path: str, collection: str, embedding_provider: OpenAIEmbeddingProvider | None = None
    ) -> None:
        self.path = path
        self.collection_name = collection
        self._entries: list[MemoryEntry] = []
        self._embedding = embedding_provider or OpenAIEmbeddingProvider()
        logger.info("chroma_backend_initialized", path=path)

    async def add(self, entry: MemoryEntry) -> None:
        if entry.embedding is None:
            embeddings = await self._embedding.embed([entry.content])
            entry.embedding = embeddings[0]
        self._entries.append(entry)

    async def search(self, query: str, k: int = 5) -> list[MemoryEntry]:
        if not self._entries:
            return []
        query_embedding = (await self._embedding.embed([query]))[0]
        scored = [
            (e, _cosine_similarity(query_embedding, e.embedding or []))
            for e in self._entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:k]]

    async def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        return sorted(self._entries, key=lambda e: e.timestamp, reverse=True)[:n]

    async def clear(self) -> None:
        self._entries.clear()


class SQLiteBackend:
    def __init__(
        self, path: str, embedding_provider: OpenAIEmbeddingProvider | None = None
    ) -> None:
        self.path = path
        self._entries: list[MemoryEntry] = []
        self._embedding = embedding_provider or OpenAIEmbeddingProvider()

    async def add(self, entry: MemoryEntry) -> None:
        if entry.embedding is None:
            embeddings = await self._embedding.embed([entry.content])
            entry.embedding = embeddings[0]
        self._entries.append(entry)

    async def search(self, query: str, k: int = 5) -> list[MemoryEntry]:
        if not self._entries:
            return []
        query_embedding = (await self._embedding.embed([query]))[0]
        scored = [
            (e, _cosine_similarity(query_embedding, e.embedding or []))
            for e in self._entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:k]]

    async def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        return sorted(self._entries, key=lambda e: e.timestamp, reverse=True)[:n]

    async def clear(self) -> None:
        self._entries.clear()


class MemoryStore:
    def __init__(self, config: Any) -> None:
        self.config = config
        self._embedding = OpenAIEmbeddingProvider(model=config.embedding_model)
        if config.backend == "chromadb":
            self.backend: ChromaBackend | SQLiteBackend = ChromaBackend(
                config.path, config.collection, self._embedding
            )
        else:
            self.backend = SQLiteBackend(config.path, self._embedding)
        self.max_history = config.max_history
        logger.info(
            "memory_store_initialized",
            backend=config.backend,
            embedding=config.embedding_model,
        )

    async def add_interaction(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        entry = MemoryEntry(role=role, content=content, metadata=metadata or {})
        await self.backend.add(entry)

    async def retrieve_relevant(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        entries = await self.backend.search(query, k)
        return [
            {
                "role": e.role,
                "content": e.content,
                "timestamp": e.timestamp.isoformat(),
                "tokens": e.token_count(),
            }
            for e in entries
        ]

    async def get_history(self, n: int | None = None) -> list[dict[str, Any]]:
        entries = await self.backend.get_recent(n or self.max_history)
        return [
            {
                "role": e.role,
                "content": e.content,
                "timestamp": e.timestamp.isoformat(),
                "tokens": e.token_count(),
            }
            for e in entries
        ]

    async def save_session(self, state: Any) -> None:
        await self.add_interaction(
            role="system",
            content=f"Session saved: {state.task}",
            metadata={"status": state.status.name, "iterations": state.iteration},
        )

    async def clear(self) -> None:
        await self.backend.clear()
