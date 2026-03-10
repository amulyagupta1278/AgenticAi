"""
Build the FAISS vector index for the Resolution Agent.

Reads the full dataset (with 'answer' column), embeds each ticket's
subject+body text, and stores the index + metadata so the resolution
agent can do semantic nearest-neighbour search at query time.

Usage:
    python scripts/build_rag_index.py
    python scripts/build_rag_index.py --max-per-category 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_CSV = ROOT / "data" / "raw" / "customer_support_tickets.csv"
KB_DIR = ROOT / "knowledge_base" / "rag_index"

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384


def main(max_per_cat: int = 500):
    print("Loading dataset...")
    df = pd.read_csv(RAW_CSV)

    # only keep english rows that have an answer
    if "language" in df.columns:
        df = df[df["language"].str.lower() == "en"]
    df = df.dropna(subset=["subject", "body", "answer"])
    df = df[df["answer"].str.strip().str.len() > 20]

    # normalize queue names to match the training labels
    df["category"] = (df["queue"].str.lower()
                      .str.replace(r"[^a-z0-9]+", "_", regex=True)
                      .str.strip("_"))

    # cap per category so the index stays manageable
    if max_per_cat:
        df = df.groupby("category").head(max_per_cat).reset_index(drop=True)

    print(f"Using {len(df)} tickets across {df['category'].nunique()} categories")

    # build the query text — this is what we'll embed and search against
    df["query_text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

    # embed everything
    print(f"Loading embedding model: {EMBED_MODEL} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)

    print("Encoding ticket texts (this might take a minute)...")
    t0 = time.time()
    embeddings = model.encode(
        df["query_text"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=256,
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Encoded {len(embeddings)} vectors in {time.time() - t0:.1f}s")

    # build FAISS index (inner product on normalized vectors = cosine sim)
    import faiss
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={EMBED_DIM}")

    # save everything
    KB_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(KB_DIR / "index.faiss"))

    # metadata — one entry per vector, keyed by position
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "category": row["category"],
            "subject": str(row["subject"])[:200],
            "body_snippet": str(row["body"])[:300],
            "answer": str(row["answer"]),
        })

    with open(KB_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=None)  # no indent — keeps file smaller

    with open(KB_DIR / "embeddings_config.json", "w") as f:
        json.dump({
            "model_name": EMBED_MODEL,
            "embedding_dim": EMBED_DIM,
            "num_vectors": len(metadata),
            "max_per_category": max_per_cat,
            "categories": sorted(df["category"].unique().tolist()),
        }, f, indent=2)

    print(f"\nRAG index saved to {KB_DIR}/")
    print(f"  index.faiss          ({index.ntotal} vectors)")
    print(f"  metadata.json        ({len(metadata)} entries)")
    print(f"  embeddings_config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS RAG index")
    parser.add_argument("--max-per-category", type=int, default=500,
                        help="Max tickets per category to include (default 500)")
    args = parser.parse_args()
    main(args.max_per_category)
