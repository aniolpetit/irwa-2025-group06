import json
import random
from collections import Counter, deque
from datetime import datetime
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd


class AnalyticsData:
    """
    In-memory analytics warehouse used by the demo application.
    Stores query level, click level and attribution tables that
    can later be rendered in dashboards without needing a DB.
    """

    def __init__(self, recent_query_window: int = 50):
        # fact tables
        self.fact_clicks: Counter[str] = Counter()
        self.fact_brand_clicks: Counter[str] = Counter()
        self.fact_category_clicks: Counter[str] = Counter()

        # query level tracking
        self.query_counter: Counter[str] = Counter()
        self.query_display_value: Dict[str, str] = {}
        self.browser_counter: Counter[str] = Counter()
        self.query_events: deque = deque(maxlen=recent_query_window)
        self.query_metadata: Dict[int, Dict[str, Any]] = {}
        self.total_queries: int = 0
        self.zero_result_queries: int = 0

        # click level breakdowns
        self.price_bucket_clicks: Counter[str] = Counter(
            {"Budget (<₹1k)": 0, "Mid-range (₹1k-₹5k)": 0, "Premium (>₹5k)": 0}
        )

    # Query instrumentation 
    def save_query_terms(self, terms: str, browser_label: Optional[str] = None) -> int:
        """
        Persist the incoming query and return a synthetic search identifier
        that can be attached to result urls.
        """
        search_id = random.randint(0, 100000)
        normalized = terms.strip().lower()

        if normalized:
            self.query_counter[normalized] += 1
            self.query_display_value[normalized] = terms.strip()

        if browser_label:
            self.browser_counter[browser_label] += 1

        event = {
            "search_id": search_id,
            "terms": terms.strip(),
            "browser": browser_label or "Unknown",
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "results_count": None,
        }
        self.query_events.appendleft(event)
        self.query_metadata[search_id] = event
        self.total_queries += 1
        return search_id

    def update_query_results(self, search_id: int, results_count: int):
        """
        Once we know how many documents were returned we enrich the
        existing query event so dashboards can report zero-result rates.
        """
        event = self.query_metadata.get(search_id)
        if not event:
            return
        event["results_count"] = results_count
        if results_count == 0:
            self.zero_result_queries += 1

    # Click instrumentation 
    def register_click(self, doc: Any):
        """
        Record a click event for dashboards and roll-ups (brand share, price tiers...)
        """
        if hasattr(doc, "model_dump"):
            doc = doc.model_dump()
        if not isinstance(doc, dict):
            return

        doc_id = doc.get("pid")
        if not doc_id:
            return

        self.fact_clicks[doc_id] += 1

        brand = doc.get("brand")
        if brand:
            self.fact_brand_clicks[brand] += 1

        category = doc.get("category")
        if category:
            self.fact_category_clicks[category] += 1

        bucket = self._bucket_price(doc.get("selling_price"))
        if bucket:
            self.price_bucket_clicks[bucket] += 1

    # Aggregations consumed by the dashboard
    def get_top_queries(self, limit: int = 5) -> List[Dict[str, Any]]:
        return [
            {"query": self.query_display_value[q], "count": count}
            for q, count in self.query_counter.most_common(limit)
        ]

    def get_recent_queries(self, limit: int = 5) -> List[Dict[str, Any]]:
        return list(self.query_events)[:limit]

    def get_browser_share(self) -> List[Dict[str, Any]]:
        return [{"browser": browser, "count": count} for browser, count in self.browser_counter.most_common()]

    def get_zero_result_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return round((self.zero_result_queries / self.total_queries) * 100, 2)

    def get_top_brands(self, limit: int = 5) -> List[Dict[str, Any]]:
        return [{"brand": brand, "count": count} for brand, count in self.fact_brand_clicks.most_common(limit)]

    def get_price_breakdown(self) -> List[Dict[str, Any]]:
        total_clicks = sum(self.price_bucket_clicks.values()) or 1
        return [
            {"bucket": bucket, "count": count, "share": round((count / total_clicks) * 100, 1)}
            for bucket, count in self.price_bucket_clicks.items()
        ]

    # Visualisations 
    def plot_number_of_views(self):
        if not self.fact_clicks:
            return "<p>No document views registered yet.</p>"

        data = [{"Document ID": doc_id, "Number of Views": count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x="Document ID", y="Number of Views")
            .properties(title="Number of Views per Document")
        )
        return chart.to_html()

    # Helpers
    @staticmethod
    def _bucket_price(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        if value < 1000:
            return "Budget (<₹1k)"
        if value <= 5000:
            return "Mid-range (₹1k-₹5k)"
        return "Premium (>₹5k)"


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
