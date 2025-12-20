from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Chunking simple et robuste.
    - chunk_size : taille max du chunk
    - overlap : recouvrement pour préserver du contexte entre chunks
    """
    text = (text or "").strip()
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks: List[str] = []
    i = 0
    n = len(text)
    step = chunk_size - overlap

    while i < n:
        j = min(n, i + chunk_size)
        chunks.append(text[i:j])
        i += step

    return chunks
