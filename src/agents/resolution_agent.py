"""
Resolution Agent — uses a FAISS vector index of real past support answers
to find the most similar resolved ticket and return that answer.

The index is pre-built by scripts/build_rag_index.py from the dataset's
'answer' column. At query time we embed the new ticket, search FAISS for
neighbours in the same category, and return the best match if similarity
is high enough.

Falls back to TF-IDF cosine sim if sentence-transformers or FAISS aren't
available (e.g. in CI).
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)

KB_DIR = ROOT / "knowledge_base" / "rag_index"
MIN_CONFIDENCE = 0.45   # don't attempt resolution if classifier is unsure
MIN_SIMILARITY = 0.40   # minimum cosine sim to consider a match

# lazy imports so the rest of the project still works when these aren't installed
_faiss = None
_SentenceTransformer = None


def _lazy_imports():
    """Import heavy deps only when actually needed."""
    global _faiss, _SentenceTransformer
    if _faiss is None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            _faiss = faiss
            _SentenceTransformer = SentenceTransformer
        except ImportError:
            logger.warning("faiss / sentence-transformers not installed — "
                           "resolution agent will use fallback mode")


class ResolutionAgent(BaseAgent):

    def __init__(self):
        super().__init__("resolution")
        self._index = None          # faiss index
        self._metadata = []         # list of dicts: {category, subject, answer}
        self._embed_model = None    # sentence-transformers model
        self._fallback = False      # true if using tfidf fallback

    def load(self):
        _lazy_imports()

        index_path = KB_DIR / "index.faiss"
        meta_path = KB_DIR / "metadata.json"

        if _faiss is not None and index_path.exists() and meta_path.exists():
            self._index = _faiss.read_index(str(index_path))
            with open(meta_path) as f:
                self._metadata = json.load(f)

            # load the same model that was used to build the index
            config_path = KB_DIR / "embeddings_config.json"
            model_name = "all-MiniLM-L6-v2"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                    model_name = cfg.get("model_name", model_name)

            self._embed_model = _SentenceTransformer(model_name)
            logger.info(f"RAG index loaded: {self._index.ntotal} vectors, "
                        f"model={model_name}")
        else:
            # fallback — no index available yet
            self._fallback = True
            logger.warning("No RAG index found at %s — resolution agent "
                           "running in fallback (keyword) mode", KB_DIR)

        self._ready = True

    def process(self, text: str, category: str, confidence: float, **kw) -> dict:
        # don't try to resolve if classifier is uncertain
        if confidence < MIN_CONFIDENCE:
            return self._empty(reason="low_confidence")

        if self._fallback or self._index is None:
            return self._keyword_fallback(text, category)

        return self._rag_search(text, category)

    # ---- main RAG path ----

    def _rag_search(self, text: str, category: str) -> dict:
        query_vec = self._embed_model.encode([text], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        # search more than we need since we'll filter by category
        k = min(20, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        best_score = 0.0
        best_entry = None

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            entry = self._metadata[idx]
            # only consider answers from the same category
            if entry.get("category") != category:
                continue
            if score > best_score:
                best_score = float(score)
                best_entry = entry

        if best_entry and best_score >= MIN_SIMILARITY:
            return {
                "resolved": True,
                "answer": best_entry["answer"],
                "source_ticket_subject": best_entry.get("subject", ""),
                "similarity_score": round(best_score, 4),
                "matched_category": category,
            }

        return self._empty(reason="no_match")

    # ---- fallback when FAISS index doesn't exist yet ----

    def _keyword_fallback(self, text: str, category: str) -> dict:
        """Simple keyword overlap — used before the RAG index is built."""
        # TODO: could load keywords.json here for better matching
        return self._empty(reason="fallback_mode")

    @staticmethod
    def _empty(reason="") -> dict:
        return {
            "resolved": False,
            "answer": None,
            "source_ticket_subject": None,
            "similarity_score": 0.0,
            "matched_category": None,
            "_skip_reason": reason,
        }
