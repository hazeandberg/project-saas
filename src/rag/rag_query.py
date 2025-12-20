from __future__ import annotations

from typing import Any, Dict, List

import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME
from .embedder import get_embedder


def rag_query(question: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Retourne les top_k passages du corpus les plus pertinents, avec leurs sources.
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("question is empty")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    # Embedding de la question via le backend choisi (local maintenant, API plus tard)
    embedder = get_embedder()
    q_emb = embedder.embed_texts([q])[0]

    # Charge la collection existante (build_index doit avoir tourné au moins une fois)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Chroma collection '{COLLECTION_NAME}' introuvable. "
            f"Lance d'abord: python -m src.rag.build_index"
        ) from e

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma renvoie des listes batchées (taille batch=1 ici)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    hits: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        hits.append(
            {
                "source": meta.get("source"),
                "chunk": meta.get("chunk"),
                "distance": dist,
                "text": doc,
            }
        )

    return {"question": q, "hits": hits}
