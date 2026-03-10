"""
Full Milestone-1 pipeline: ingest → train (all models) → print results.

Usage:
    python run_pipeline.py
    python run_pipeline.py --model logistic_regression
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Specific model to train (default: train all)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion if clean CSV exists")
    args = parser.parse_args()

    clean_path = ROOT / "data" / "processed" / "tickets_clean.csv"

    # step 1 — ingest
    if not args.skip_ingest or not clean_path.exists():
        print("\n[1/2] Ingesting data...")
        from src.data.preprocess import run as ingest_run
        ingest_run()
    else:
        print(f"\n[1/2] Skipping ingest — using {clean_path}")

    # step 2 — train
    print("\n[2/2] Training...")
    if args.model:
        from src.models.train import train
        train(args.model)
    else:
        from src.models.train import train_all
        results = train_all()
        print("\n── Final comparison ──────────────────────")
        for name, m in results.items():
            print(f"  {name:<25} acc={m['accuracy']:.3f}  f1={m['f1']:.3f}")

    print("\nDone. Start the API with:\n  uvicorn src.api.main:app --reload\n")


if __name__ == "__main__":
    main()
