"""
Run a batch of tickets through the full agent pipeline and print
impact metrics.  This is the main evaluation script that shows what
the system can do.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --sample 1000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.orchestrator import OrchestratorAgent

DATA_PATH = ROOT / "data" / "processed" / "training_dataset.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

AVG_TICKET_COST = 20.0  # conservative estimate per ticket


def run(sample_size: int, seed: int):
    df = pd.read_csv(DATA_PATH)
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed)

    print("Initializing agent system...")
    orch = OrchestratorAgent()
    orch.load()

    results = []
    t0 = time.perf_counter()

    for idx, row in sample.iterrows():
        result = orch.process(text=row["text"], ticket_id=f"SIM-{idx:05d}")
        result["true_label"] = row["label"]
        result["correct"] = result["classification"]["category"] == row["label"]
        results.append(result)

    total_time = time.perf_counter() - t0

    # compute metrics
    n = len(results)
    auto_resolved = sum(1 for r in results if r["status"] == "auto_resolved")
    routed = sum(1 for r in results if r["status"] == "routed")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    correct = sum(1 for r in results if r["correct"])

    # of the auto-resolved ones, how many were classified correctly?
    ar_correct = sum(1 for r in results
                     if r["status"] == "auto_resolved" and r["correct"])

    metrics = {
        "total_tickets": n,
        "classification_accuracy_pct": round(correct / n * 100, 1),
        "auto_resolved": auto_resolved,
        "auto_resolved_pct": round(auto_resolved / n * 100, 1),
        "auto_resolution_accuracy_pct": (
            round(ar_correct / auto_resolved * 100, 1) if auto_resolved else 0
        ),
        "routed": routed,
        "routed_pct": round(routed / n * 100, 1),
        "escalated": escalated,
        "escalated_pct": round(escalated / n * 100, 1),
        "avg_processing_ms": round(total_time / n * 1000, 2),
        "total_processing_s": round(total_time, 2),
        "estimated_annual_savings_usd": round(
            auto_resolved / n * 60000 * AVG_TICKET_COST),  # extrapolate to 60K tickets
    }

    # per-category breakdown
    cat_stats = {}
    for r in results:
        cat = r["true_label"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "auto_resolved": 0, "correct": 0}
        cat_stats[cat]["total"] += 1
        if r["status"] == "auto_resolved":
            cat_stats[cat]["auto_resolved"] += 1
        if r["correct"]:
            cat_stats[cat]["correct"] += 1
    metrics["per_category"] = cat_stats

    # print report
    print()
    print("=" * 60)
    print("  MULTI-AGENT SYSTEM — SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Tickets processed       : {n}")
    print(f"  Classification accuracy  : {metrics['classification_accuracy_pct']}%")
    print(f"  Auto-resolved            : {auto_resolved} ({metrics['auto_resolved_pct']}%)")
    print(f"    Resolution accuracy    : {metrics['auto_resolution_accuracy_pct']}%")
    print(f"  Routed to team           : {routed} ({metrics['routed_pct']}%)")
    print(f"  Escalated                : {escalated} ({metrics['escalated_pct']}%)")
    print(f"  Avg latency              : {metrics['avg_processing_ms']} ms/ticket")
    print(f"  Est. annual savings      : ${metrics['estimated_annual_savings_usd']:,}")
    print("=" * 60)

    # per-category table
    print("\n  Per-category breakdown:")
    print(f"  {'Category':<40} {'Total':>6} {'Auto':>6} {'Acc':>6}")
    print("  " + "-" * 60)
    for cat, s in sorted(cat_stats.items()):
        ar_rate = f"{s['auto_resolved']/s['total']*100:.0f}%" if s['total'] else "n/a"
        acc = f"{s['correct']/s['total']*100:.0f}%" if s['total'] else "n/a"
        print(f"  {cat:<40} {s['total']:>6} {ar_rate:>6} {acc:>6}")

    # save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flat = []
    for r in results:
        flat.append({
            "ticket_id": r["ticket_id"],
            "true_label": r["true_label"],
            "predicted": r["classification"]["category"],
            "confidence": r["classification"]["confidence"],
            "status": r["status"],
            "correct": r["correct"],
            "similarity": r["resolution"].get("similarity_score", 0),
            "routing_team": r["routing"]["team"] if r["routing"] else None,
            "routing_priority": r["routing"]["priority_level"] if r["routing"] else None,
            "processing_ms": r["processing_time_ms"],
        })
    pd.DataFrame(flat).to_csv(OUTPUT_DIR / "simulation_results.csv", index=False)

    with open(OUTPUT_DIR / "simulation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/simulation_results.csv")
    print(f"Metrics saved to {OUTPUT_DIR}/simulation_metrics.json")

    # also show prevention insights
    print("\n--- Prevention Insights ---")
    insights = orch.get_insights()
    for rec in insights.get("prevention_recommendations", []):
        print(f"  * {rec}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.sample, args.seed)
