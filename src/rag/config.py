from __future__ import annotations
from pathlib import Path

CORPUS_DIR = Path("docs_corpus")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "saas_internal_corpus"

# Chunking
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Embedder mode:
# - "local" (Option A)
# - "openai" (Option B, prêt mais non activé tant que clé absente)
EMBEDDER_MODE = "local"

# Option A (local)
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Option B (API) - placeholders
OPENAI_EMBED_MODEL = "text-embedding-3-small"  # si tu actives B plus tard
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
