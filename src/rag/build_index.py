from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, cast
import numpy as np
import chromadb

from .chunking import chunk_text
from .config import (
    CORPUS_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from .embedder import get_embedder


Metadata = Dict[str, Any]


def iter_corpus_files(corpus_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".md", ".yaml", ".yml"}:
            paths.append(p)
    return paths


def build_documents(files: List[Path]) -> Tuple[List[str], List[str], List[Metadata]]:
    """
    Retourne (ids, texts, metadatas)
    """
    ids: List[str] = []
    texts: List[str] = []
    metas: List[Metadata] = []

    for fp in files:
        content = fp.read_text(encoding="utf-8")
        chunks = chunk_text(content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for k, ch in enumerate(chunks):
            ids.append(f"{fp.as_posix()}::chunk_{k}")
            texts.append(ch)
            metas.append({"source": fp.as_posix(), "chunk": k})

    return ids, texts, metas


def _validate_embeddings(embeddings: List[List[float]], n_texts: int) -> None:
    if len(embeddings) != n_texts:
        raise ValueError(
            f"Embeddings count mismatch: embeddings={len(embeddings)} texts={n_texts}"
        )
    if not embeddings or not embeddings[0]:
        raise ValueError("Embeddings empty or malformed (first vector missing).")
    # Vérifie que c'est bien des floats (au moins sur le 1er vecteur)
    if not isinstance(embeddings[0][0], float):
        raise TypeError(
            f"Embeddings must be List[List[float]]. Got first value type={type(embeddings[0][0])}"
        )


def main() -> None:

    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"Missing corpus dir: {CORPUS_DIR}")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[RAG] corpus_dir={CORPUS_DIR.resolve()}")
    print(f"[RAG] chroma_dir={CHROMA_DIR.resolve()}")
    print(f"[RAG] collection={COLLECTION_NAME}")
    print(f"[RAG] chunk_size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}")

    files = iter_corpus_files(CORPUS_DIR)
    if not files:
        raise ValueError("No corpus files found (.md/.yml/.yaml).")

    ids, texts, metas = build_documents(files)
    if not texts:
        raise ValueError("Corpus empty after chunking.")

    embedder = get_embedder()
    print("[DEBUG] before embed_texts: n_texts=", len(texts))
    embeddings = embedder.embed_texts(texts)
    print("[DEBUG] after embed_texts: n_embeddings=", len(embeddings))

    _validate_embeddings(embeddings, n_texts=len(texts))

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)
    print("[DEBUG] metas type/len:", type(metas), len(metas))
    print(
        "[DEBUG] embeddings type/len:",
        type(embeddings),
        len(embeddings),
        type(embeddings[0][0]),
    )
    print("[DEBUG] before collection.add")
    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    metas_cast = cast(Any, metas)  # satisfait Pylance, runtime inchangé
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metas_cast,
        embeddings=embeddings_np,
    )
    print("[DEBUG] after collection.add")

    print(f"[RAG] indexed_files={len(files)} indexed_chunks={len(texts)}")


if __name__ == "__main__":
    main()
