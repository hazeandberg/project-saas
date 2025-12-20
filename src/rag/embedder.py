from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Protocol

from .config import (
    EMBEDDER_MODE,
    LOCAL_EMBED_MODEL,
    OPENAI_API_KEY_ENV,
    OPENAI_EMBED_MODEL,
)


class Embedder(Protocol):
    def embed_texts(self, texts: List[str]) -> List[List[float]]: ...


@dataclass(frozen=True)
class LocalSentenceTransformerEmbedder:
    model_name: str = LOCAL_EMBED_MODEL

    def __post_init__(self) -> None:
        # import lazy-friendly: pas d'import coûteux au chargement module
        pass

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name)
        embs = model.encode(texts, normalize_embeddings=True)
        return embs.tolist()


@dataclass(frozen=True)
class OpenAIEmbedder:
    api_key: str
    model: str = OPENAI_EMBED_MODEL

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Option B prête, mais volontairement encapsulée.
        Note: tu actives seulement quand tu as la clé dans OPENAI_API_KEY.
        """
        if not texts:
            return []

        # Implémentation volontairement simple et isolée.
        # Si tu actives B, on mettra la lib officielle et une gestion d'erreurs propre.
        from openai import OpenAI  # nécessite `pip install openai`

        client = OpenAI(api_key=self.api_key)
        resp = client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]


def get_embedder() -> Embedder:
    """
    Factory: permet switch A/B sans toucher le reste du code.
    """
    mode = (EMBEDDER_MODE or "local").strip().lower()

    if mode == "local":
        return LocalSentenceTransformerEmbedder()

    if mode == "openai":
        api_key = os.getenv(OPENAI_API_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                f"EMBEDDER_MODE=openai mais env var {OPENAI_API_KEY_ENV} manquante."
            )
        return OpenAIEmbedder(api_key=api_key)

    raise ValueError(f"Unknown EMBEDDER_MODE: {mode}")
