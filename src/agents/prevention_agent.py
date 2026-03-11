"""
Prevention Agent — analyses historical ticket data to find trends,
recurring issues, and knowledge gaps.  Generates actionable recs
so the support team can fix root causes instead of just treating symptoms.

Since the dataset doesn't have real timestamps, we split it into
synthetic time windows to demonstrate the trend detection concept.
"""

import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agents.base import BaseAgent

DATA_PATH = ROOT / "data" / "processed" / "training_dataset.csv"

# words that show up everywhere and don't help with analysis
_STOP = {
    "the", "and", "for", "that", "this", "with", "you", "are", "was",
    "have", "has", "had", "not", "but", "from", "they", "been", "will",
    "would", "could", "should", "about", "which", "their", "there",
    "what", "when", "make", "can", "each", "other", "also", "its",
    "into", "than", "then", "them", "these", "some", "her", "him",
    "our", "out", "may", "all", "your", "any", "how", "more",
    # priority tokens injected during preprocessing
    "priority_high", "priority_medium", "priority_low",
}


class PreventionAgent(BaseAgent):

    def __init__(self):
        super().__init__("prevention")
        self._df = None

    def load(self, data_path=None):
        path = Path(data_path) if data_path else DATA_PATH
        if path.exists():
            self._df = pd.read_csv(path)
        self._ready = True

    def process(self, **kw) -> dict:
        if self._df is None or self._df.empty:
            return {"generated_at": datetime.now(timezone.utc).isoformat(),
                    "insights": [], "total_tickets_analyzed": 0,
                    "top_recurring_issues": [], "prevention_recommendations": []}

        df = self._df
        total = len(df)
        cat_counts = df["label"].value_counts()

        # split data into 4 synthetic time windows for trend detection
        chunk = total // 4
        windows = [df.iloc[i * chunk:(i + 1) * chunk if i < 3 else total]
                   for i in range(4)]

        insights = []
        for cat in cat_counts.index:
            count = int(cat_counts[cat])
            pct = round(count / total * 100, 1)

            first = (windows[0]["label"] == cat).sum()
            last = (windows[-1]["label"] == cat).sum()
            if last > first * 1.15:
                trend = "increasing"
            elif last < first * 0.85:
                trend = "decreasing"
            else:
                trend = "stable"

            top_kw = self._top_keywords(df[df["label"] == cat]["text"], n=8)
            rec = self._recommendation(cat, trend, top_kw, pct)

            insights.append({
                "category": cat,
                "trend": trend,
                "ticket_count": count,
                "percentage": pct,
                "recurring_tags": top_kw,
                "recommendation": rec,
            })

        # top recurring keywords across all categories
        global_kw = self._top_keywords(df["text"], n=12)
        top_recurring = [{"keyword": w, "rank": i + 1}
                         for i, w in enumerate(global_kw)]

        recs = self._overall_recommendations(insights)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_tickets_analyzed": total,
            "insights": insights,
            "top_recurring_issues": top_recurring,
            "prevention_recommendations": recs,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _top_keywords(texts, n=8):
        counter = Counter()
        for t in texts:
            for w in str(t).lower().split():
                if len(w) > 3 and w not in _STOP:
                    counter[w] += 1
        return [w for w, _ in counter.most_common(n)]

    @staticmethod
    def _recommendation(category, trend, keywords, pct):
        name = category.replace("_", " ").title()
        kw_str = ", ".join(keywords[:3]) if keywords else "general issues"

        if trend == "increasing":
            return (f"{name} is trending up — common themes: {kw_str}. "
                    "Consider adding proactive docs or automated alerts.")
        elif trend == "decreasing":
            return (f"{name} is declining — current mitigations seem to work. "
                    "Keep monitoring.")
        return (f"{name} is stable at {pct}% of tickets. "
                f"Recurring themes: {kw_str}. Review KB coverage.")

    @staticmethod
    def _overall_recommendations(insights):
        recs = []
        rising = [i for i in insights if i["trend"] == "increasing"]
        if rising:
            names = [i["category"].replace("_", " ") for i in rising]
            recs.append(
                f"Ticket volume increasing in: {', '.join(names)}. "
                "Expand FAQ and self-service for these areas.")

        heavy = [i for i in insights if i["percentage"] > 20]
        for h in heavy:
            name = h["category"].replace("_", " ")
            recs.append(
                f"{name} is {h['percentage']}% of all tickets — "
                "invest in self-service tooling and proactive notifications.")

        if not recs:
            recs.append("Ticket distribution is balanced. "
                        "Maintain KB coverage across all categories.")
        return recs
