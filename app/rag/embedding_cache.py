"""Persistent cache for document embeddings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_CACHE_SCHEMA_VERSION = 1


class EmbeddingCache:
    """Store vectors by content and embedding configuration."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, list[float]] = {}
        if path.exists():
            self._load()

    @staticmethod
    def key(text: str, *, model: str, dimension: int) -> str:
        payload = f"{_CACHE_SCHEMA_VERSION}\0{model}\0{dimension}\0document\0{text}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> list[float] | None:
        vector = self._entries.get(key)
        return list(vector) if vector is not None else None

    def put(self, key: str, vector: list[float]) -> None:
        self._entries[key] = list(vector)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "entries": self._entries,
        }
        self._path.write_text(json.dumps(
            payload, indent=2, sort_keys=True), encoding="utf-8")

    def _load(self) -> None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if payload["schema_version"] != _CACHE_SCHEMA_VERSION:
                return
            self._entries = {
                key: [float(value) for value in vector]
                for key, vector in payload["entries"].items()
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            self._entries = {}
