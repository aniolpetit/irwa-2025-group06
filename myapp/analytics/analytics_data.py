import json
import random
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd


@dataclass
class SessionRecord:
    session_id: str
    visitor_id: str
    user_agent: str
    device_type: str
    os_label: str
    ip_address: str
    country: Optional[str]
    city: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    missions: List[str] = field(default_factory=list)
    query_ids: List[int] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MissionRecord:
    mission_id: str
    session_id: str
    goal: str
    start_time: datetime
    end_time: Optional[datetime] = None
    query_ids: List[int] = field(default_factory=list)


@dataclass
class RequestRecord:
    request_id: str
    session_id: str
    method: str
    path: str
    status_code: int
    timestamp: datetime
    latency_ms: Optional[float]
    bytes_sent: Optional[int]


@dataclass
class QueryRecord:
    query_id: int
    session_id: str
    mission_id: Optional[str]
    order_in_session: int
    terms: str
    normalized_terms: str
    num_terms: int
    browser: str
    os_label: str
    device_type: str
    ip_address: str
    country: Optional[str]
    city: Optional[str]
    timestamp: datetime
    results_count: Optional[int] = None


@dataclass
class ClickRecord:
    click_id: str
    session_id: Optional[str]
    query_id: Optional[int]
    doc_id: str
    brand: Optional[str]
    category: Optional[str]
    rank_position: Optional[int]
    dwell_time_ms: Optional[int]
    timestamp: datetime


class AnalyticsData:
    """
    In-memory analytics warehouse used by the demo application.
    Stores query/request/session/click fact tables that can later
    be rendered in dashboards without needing a DB. The layout loosely
    follows a star schema with Request, Session and Click fact tables
    and lightweight dimension counters for browsers, devices, geography
    and missions. The collector captures:

    1. HTTP requests (path, method, latency, status) + per-session linkage.
    2. Queries (terms, order within session, browser/device/OS/IP/geo).
    3. Result clicks (document metadata, originating query, rank, dwell).
    4. Visitor context (browser, OS, device, time-of-day buckets, geography).
    5. Sessions & missions (physical/logical groupings with auto timeout).

    Dashboard helpers expose KPIs, breakdowns and Altair visualisations
    so instructors can easily validate analytics coverage.
    """

    def __init__(self, recent_query_window: int = 50, session_timeout_minutes: int = 30):
        # fact tables
        self.fact_clicks: Counter[str] = Counter()
        self.fact_brand_clicks: Counter[str] = Counter()
        self.fact_category_clicks: Counter[str] = Counter()
        self.fact_click_events: List[ClickRecord] = []
        self.fact_requests: List[RequestRecord] = []

        # query level tracking
        self.query_counter: Counter[str] = Counter()
        self.query_display_value: Dict[str, str] = {}
        self.browser_counter: Counter[str] = Counter()
        self.query_events: deque = deque(maxlen=recent_query_window)
        self.query_metadata: Dict[int, QueryRecord] = {}
        self.total_queries: int = 0
        self.zero_result_queries: int = 0
        self.session_query_order: Counter[str] = Counter()

        # click level breakdowns
        self.price_bucket_clicks: Counter[str] = Counter(
            {"Budget (<₹1k)": 0, "Mid-range (₹1k-₹5k)": 0, "Premium (>₹5k)": 0}
        )

        # user context & sessions
        self.sessions: Dict[str, SessionRecord] = {}
        self.missions: Dict[str, MissionRecord] = {}
        self.device_counter: Counter[str] = Counter()
        self.os_counter: Counter[str] = Counter()
        self.time_of_day_counter: Counter[str] = Counter()
        self.geo_counter: Counter[Tuple[Optional[str], Optional[str]]] = Counter()
        self.request_status_counter: Counter[int] = Counter()
        self.session_timeout = timedelta(minutes=session_timeout_minutes)

    # Session & request instrumentation
    def start_session(
        self,
        visitor_id: Optional[str],
        browser_label: Optional[str] = None,
        device_type: Optional[str] = None,
        os_label: Optional[str] = None,
        ip_address: Optional[str] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        timestamp = timestamp or datetime.utcnow()
        if session_id and session_id in self.sessions:
            return session_id
        session_id = session_id or str(uuid.uuid4())
        record = SessionRecord(
            session_id=session_id,
            visitor_id=visitor_id or "anonymous",
            user_agent=browser_label or "Unknown",
            device_type=device_type or "Unknown",
            os_label=os_label or "Unknown",
            ip_address=ip_address or "Unknown",
            country=country,
            city=city,
            start_time=timestamp,
            last_activity=timestamp,
        )
        self.sessions[session_id] = record
        self._update_context_counters(device_type, os_label, country, city, timestamp)
        return session_id

    def end_session(self, session_id: str):
        session = self.sessions.get(session_id)
        if session and not session.end_time:
            session.end_time = datetime.utcnow()

    def start_mission(self, session_id: str, goal: str) -> str:
        timestamp = datetime.utcnow()
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError("Cannot start a mission without a valid session.")
        mission_id = str(uuid.uuid4())
        mission = MissionRecord(
            mission_id=mission_id,
            session_id=session_id,
            goal=goal,
            start_time=timestamp,
        )
        self.missions[mission_id] = mission
        session.missions.append(mission_id)
        return mission_id

    def end_mission(self, mission_id: str):
        mission = self.missions.get(mission_id)
        if mission and not mission.end_time:
            mission.end_time = datetime.utcnow()

    def record_request(
        self,
        path: str,
        method: str,
        status_code: int,
        session_id: Optional[str] = None,
        visitor_id: Optional[str] = None,
        browser_label: Optional[str] = None,
        device_type: Optional[str] = None,
        os_label: Optional[str] = None,
        ip_address: Optional[str] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
        latency_ms: Optional[float] = None,
        bytes_sent: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> str:
        timestamp = timestamp or datetime.utcnow()
        session_id = self._get_or_create_session(
            session_id=session_id,
            visitor_id=visitor_id,
            browser_label=browser_label,
            device_type=device_type,
            os_label=os_label,
            ip_address=ip_address,
            country=country,
            city=city,
            timestamp=timestamp,
        )
        request_id = str(uuid.uuid4())
        record = RequestRecord(
            request_id=request_id,
            session_id=session_id,
            method=method.upper(),
            path=path,
            status_code=status_code,
            timestamp=timestamp,
            latency_ms=latency_ms,
            bytes_sent=bytes_sent,
        )
        self.fact_requests.append(record)
        self.request_status_counter[status_code] += 1
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = timestamp
        self._update_context_counters(device_type, os_label, country, city, timestamp)
        return request_id

    def _get_or_create_session(
        self,
        session_id: Optional[str],
        visitor_id: Optional[str],
        browser_label: Optional[str],
        device_type: Optional[str],
        os_label: Optional[str],
        ip_address: Optional[str],
        country: Optional[str],
        city: Optional[str],
        timestamp: datetime,
    ) -> str:
        if session_id and session_id in self.sessions:
            return session_id
        reusable_session = None
        if visitor_id:
            reusable_session = self._find_active_session(visitor_id, timestamp)
        if reusable_session:
            return reusable_session
        return self.start_session(
            visitor_id=visitor_id,
            browser_label=browser_label,
            device_type=device_type,
            os_label=os_label,
            ip_address=ip_address,
            country=country,
            city=city,
            session_id=session_id,
            timestamp=timestamp,
        )

    def _update_context_counters(
        self,
        device_type: Optional[str],
        os_label: Optional[str],
        country: Optional[str],
        city: Optional[str],
        timestamp: datetime,
    ):
        device = (device_type or "Unknown").title()
        os_name = (os_label or "Unknown").title()
        geo_key = (country or "Unknown", city or "Unknown")
        time_bucket = timestamp.strftime("%H:00")

        self.device_counter[device] += 1
        self.os_counter[os_name] += 1
        self.geo_counter[geo_key] += 1
        self.time_of_day_counter[time_bucket] += 1

    def _find_active_session(self, visitor_id: str, timestamp: datetime) -> Optional[str]:
        for session in self.sessions.values():
            if session.visitor_id != visitor_id:
                continue
            if session.end_time:
                continue
            elapsed = timestamp - session.last_activity
            if elapsed <= self.session_timeout:
                return session.session_id
            session.end_time = session.last_activity
        return None

    # Query instrumentation 
    def save_query_terms(
        self,
        terms: str,
        browser_label: Optional[str] = None,
        session_id: Optional[str] = None,
        mission_id: Optional[str] = None,
        device_type: Optional[str] = None,
        os_label: Optional[str] = None,
        visitor_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        country: Optional[str] = None,
        city: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Persist the incoming query and return a synthetic search identifier
        that can be attached to result urls.
        """
        timestamp = timestamp or datetime.utcnow()
        search_id = random.randint(0, 100000)
        normalized = terms.strip().lower()
        session_id = self._get_or_create_session(
            session_id=session_id,
            visitor_id=visitor_id,
            browser_label=browser_label,
            device_type=device_type,
            os_label=os_label,
            ip_address=ip_address,
            country=country,
            city=city,
            timestamp=timestamp,
        )
        if mission_id and mission_id not in self.missions:
            self.missions[mission_id] = MissionRecord(
                mission_id=mission_id,
                session_id=session_id,
                goal="Unspecified mission",
                start_time=timestamp,
            )
            self.sessions[session_id].missions.append(mission_id)

        if normalized:
            self.query_counter[normalized] += 1
            self.query_display_value[normalized] = terms.strip()

        if browser_label:
            self.browser_counter[browser_label] += 1

        order = self.session_query_order[session_id] + 1
        self.session_query_order[session_id] = order
        num_terms = len(terms.split())
        event = {
            "search_id": search_id,
            "terms": terms.strip(),
            "browser": browser_label or "Unknown",
            "device": device_type or "Unknown",
            "os": os_label or "Unknown",
            "session_id": session_id,
            "mission_id": mission_id,
            "order": order,
            "num_terms": num_terms,
            "ip_address": ip_address or "Unknown",
            "location": {"country": country, "city": city},
            "timestamp": timestamp.isoformat(timespec="seconds"),
            "results_count": None,
        }
        self.query_events.appendleft(event)
        record = QueryRecord(
            query_id=search_id,
            session_id=session_id,
            mission_id=mission_id,
            order_in_session=order,
            terms=terms.strip(),
            normalized_terms=normalized,
            num_terms=num_terms,
            browser=browser_label or "Unknown",
            os_label=os_label or "Unknown",
            device_type=device_type or "Unknown",
            ip_address=ip_address or "Unknown",
            country=country,
            city=city,
            timestamp=timestamp,
        )
        self.query_metadata[search_id] = record
        self.sessions[session_id].query_ids.append(search_id)
        self.sessions[session_id].last_activity = timestamp
        if mission_id:
            self.missions[mission_id].query_ids.append(search_id)
        self.total_queries += 1
        self._update_context_counters(device_type, os_label, country, city, timestamp)
        return search_id

    def update_query_results(self, search_id: int, results_count: int):
        """
        Once we know how many documents were returned we enrich the
        existing query event so dashboards can report zero-result rates.
        """
        record = self.query_metadata.get(search_id)
        if not record:
            return
        record.results_count = results_count
        for event in self.query_events:
            if event["search_id"] == search_id:
                event["results_count"] = results_count
                break
        if results_count == 0:
            self.zero_result_queries += 1

    # Click instrumentation 
    def register_click(
        self,
        doc: Any,
        query_id: Optional[int] = None,
        session_id: Optional[str] = None,
        rank_position: Optional[int] = None,
        dwell_time_ms: Optional[int] = None,
    ):
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

        timestamp = datetime.utcnow()
        if query_id and query_id in self.query_metadata:
            session_id = session_id or self.query_metadata[query_id].session_id

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

        click_record = ClickRecord(
            click_id=str(uuid.uuid4()),
            session_id=session_id,
            query_id=query_id,
            doc_id=doc_id,
            brand=brand,
            category=category,
            rank_position=rank_position,
            dwell_time_ms=dwell_time_ms,
            timestamp=timestamp,
        )
        self.fact_click_events.append(click_record)
        if session_id and session_id in self.sessions:
            self.sessions[session_id].last_activity = timestamp

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

    def get_device_share(self) -> List[Dict[str, Any]]:
        return [{"device": device, "count": count} for device, count in self.device_counter.most_common()]

    def get_os_share(self) -> List[Dict[str, Any]]:
        return [{"os": os_name, "count": count} for os_name, count in self.os_counter.most_common()]

    def get_geo_distribution(self, limit: int = 10) -> List[Dict[str, Any]]:
        results = []
        for (country, city), count in self.geo_counter.most_common(limit):
            results.append(
                {
                    "country": country,
                    "city": city,
                    "count": count,
                }
            )
        return results

    def get_time_of_day_activity(self) -> List[Dict[str, Any]]:
        return [
            {"hour": hour, "count": count}
            for hour, count in sorted(self.time_of_day_counter.items(), key=lambda item: item[0])
        ]

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

    def get_key_indicators(self) -> Dict[str, Any]:
        request_summary = self.get_request_summary()
        session_overview = self.get_session_overview()
        click_metrics = self.get_click_metrics()
        return {
            "total_requests": request_summary["total_requests"],
            "total_sessions": session_overview["total_sessions"],
            "avg_session_duration_seconds": session_overview["avg_duration_seconds"],
            "avg_latency_ms": request_summary["avg_latency_ms"],
            "total_queries": self.total_queries,
            "zero_result_rate": self.get_zero_result_rate(),
            "avg_dwell_time_ms": click_metrics["avg_dwell_time_ms"],
        }

    def get_request_summary(self) -> Dict[str, Any]:
        total_requests = len(self.fact_requests)
        if total_requests == 0:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0.0,
                "top_paths": [],
                "status_breakdown": [],
            }
        latency_values = [req.latency_ms or 0 for req in self.fact_requests]
        avg_latency = round(sum(latency_values) / total_requests, 2)
        top_paths = Counter(req.path for req in self.fact_requests).most_common(5)
        status_breakdown = [
            {"status": status, "count": count} for status, count in self.request_status_counter.most_common()
        ]
        return {
            "total_requests": total_requests,
            "avg_latency_ms": avg_latency,
            "top_paths": [{"path": path, "count": count} for path, count in top_paths],
            "status_breakdown": status_breakdown,
        }

    def get_session_overview(self) -> Dict[str, Any]:
        total_sessions = len(self.sessions)
        if total_sessions == 0:
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "avg_duration_seconds": 0,
                "missions_tracked": 0,
            }
        now = datetime.utcnow()
        durations = []
        active_sessions = 0
        for session in self.sessions.values():
            end_time = session.end_time or now
            durations.append((end_time - session.start_time).total_seconds())
            if not session.end_time:
                active_sessions += 1
        avg_duration = round(sum(durations) / len(durations), 2)
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "avg_duration_seconds": avg_duration,
            "missions_tracked": len(self.missions),
        }

    def get_mission_overview(self) -> Dict[str, Any]:
        if not self.missions:
            return {"missions": []}
        overview = []
        for mission in self.missions.values():
            duration = None
            if mission.end_time:
                duration = (mission.end_time - mission.start_time).total_seconds()
            overview.append(
                {
                    "mission_id": mission.mission_id,
                    "session_id": mission.session_id,
                    "goal": mission.goal,
                    "queries": len(mission.query_ids),
                    "duration_seconds": duration,
                }
            )
        return {"missions": overview}

    def get_click_metrics(self) -> Dict[str, Any]:
        total_clicks = len(self.fact_click_events)
        dwell_values = [event.dwell_time_ms for event in self.fact_click_events if event.dwell_time_ms is not None]
        avg_dwell = round(sum(dwell_values) / len(dwell_values), 2) if dwell_values else 0.0
        ranking_counts = Counter(
            f"Rank {event.rank_position}" if event.rank_position is not None else "Unknown"
            for event in self.fact_click_events
        )
        return {
            "total_clicks": total_clicks,
            "avg_dwell_time_ms": avg_dwell,
            "ranking_breakdown": [{"rank": rank, "count": count} for rank, count in ranking_counts.items()],
        }

    def get_dwell_time_statistics(self) -> Dict[str, Any]:
        dwell_values = sorted(
            [event.dwell_time_ms for event in self.fact_click_events if event.dwell_time_ms is not None]
        )
        if not dwell_values:
            return {"min_ms": 0, "max_ms": 0, "median_ms": 0, "p90_ms": 0}
        count = len(dwell_values)
        median = dwell_values[count // 2] if count % 2 else (dwell_values[count // 2 - 1] + dwell_values[count // 2]) / 2
        p90_index = max(0, min(count - 1, int(round(0.9 * (count - 1)))))
        return {
            "min_ms": dwell_values[0],
            "max_ms": dwell_values[-1],
            "median_ms": median,
            "p90_ms": dwell_values[p90_index],
        }

    def get_query_detail(self, query_id: int) -> Optional[Dict[str, Any]]:
        record = self.query_metadata.get(query_id)
        if not record:
            return None
        clicks = [
            {
                "doc_id": event.doc_id,
                "rank_position": event.rank_position,
                "dwell_time_ms": event.dwell_time_ms,
                "timestamp": event.timestamp.isoformat(timespec="seconds"),
            }
            for event in self.fact_click_events
            if event.query_id == query_id
        ]
        return {
            "query": record.terms,
            "session_id": record.session_id,
            "mission_id": record.mission_id,
            "order_in_session": record.order_in_session,
            "num_terms": record.num_terms,
            "results_count": record.results_count,
            "clicks": clicks,
        }

    def update_click_dwell(self, query_id: Optional[int], doc_id: str, dwell_time_ms: int):
        """
        Update the dwell time of the most recent click that matches the query/document combo.
        """
        if dwell_time_ms is None or doc_id is None:
            return
        dwell_time_ms = max(0, dwell_time_ms)
        for event in reversed(self.fact_click_events):
            if event.doc_id != doc_id:
                continue
            if query_id is not None and event.query_id != query_id:
                continue
            event.dwell_time_ms = dwell_time_ms
            break

    # Visualisations 
    def plot_number_of_views(self):
        if not self.fact_clicks:
            return "<p>No document views registered yet.</p>"

        data = [{"Document ID": doc_id, "Number of Views": count} for doc_id, count in self.fact_clicks.items()]
        df = pd.DataFrame(data)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x="Document ID", y=alt.Y("Number of Views", axis=alt.Axis(format="d", tickMinStep=1)))
            .properties(title="Number of Views per Document", width=350, height=250)
        )
        caption = "<p class='chart-caption'>Bars show which documents attract the most attention based on registered clicks.</p>"
        return self._render_chart(chart, caption)

    def plot_sessions_by_hour(self):
        activity = self.get_time_of_day_activity()
        if not activity:
            return "<p>No sessions tracked yet.</p>"
        df = pd.DataFrame(activity)
        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(x="hour", y=alt.Y("count", axis=alt.Axis(format="d", tickMinStep=1)))
            .properties(title="Sessions by Hour of Day", width=350, height=250)
        )
        caption = "<p class='chart-caption'>Line trend explains traffic peaks across the day for capacity planning.</p>"
        return self._render_chart(chart, caption)

    def plot_requests_by_status(self):
        if not self.request_status_counter:
            return "<p>No HTTP requests have been captured yet.</p>"
        data = [{"Status Code": status, "Count": count} for status, count in self.request_status_counter.items()]
        df = pd.DataFrame(data)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(x="Status Code:O", y=alt.Y("Count", axis=alt.Axis(format="d", tickMinStep=1)))
            .properties(title="HTTP Request Status Breakdown", width=350, height=250)
        )
        caption = "<p class='chart-caption'>Bars distinguish successful vs error HTTP responses so we can monitor reliability.</p>"
        return self._render_chart(chart, caption)

    def plot_dwell_time_distribution(self):
        dwell_values = [event.dwell_time_ms for event in self.fact_click_events if event.dwell_time_ms is not None]
        if not dwell_values:
            return "<p>Open a result and come back to the list to capture dwell time samples.</p>"
        df = pd.DataFrame({"Dwell Time (ms)": dwell_values})
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Dwell Time (ms)", bin=alt.Bin(maxbins=30)),
                y=alt.Y("count()", axis=alt.Axis(format="d", tickMinStep=1)),
            )
            .properties(title="Dwell Time Distribution", width=350, height=250)
        )
        caption = "<p class='chart-caption'>Histogram shows how long users stay on result pages before returning.</p>"
        return self._render_chart(chart, caption)

    def plot_price_sensitivity(self):
        if not self.fact_click_events:
            return "<p>No click data available yet.</p>"
        data = self.get_price_breakdown()
        df = pd.DataFrame(data)
        chart = (
            alt.Chart(df)
            .mark_arc()
            .encode(theta="share", color="bucket", tooltip=["bucket", "count", "share"])
            .properties(title="Price Sensitivity", width=450, height=320)
        )
        caption = "<p class='chart-caption'>Pie slices show how clicks distribute across budget, mid-range, and premium buckets.</p>"
        return self._render_chart(chart, caption)

    def plot_top_brands(self, limit: int = 10):
        if not self.fact_brand_clicks:
            return "<p>No brand clicks recorded yet.</p>"
        items = [{"brand": brand, "count": count} for brand, count in self.fact_brand_clicks.most_common(limit)]
        df = pd.DataFrame(items)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", axis=alt.Axis(format="d", tickMinStep=1)),
                y=alt.Y("brand:N", sort="-x"),
                tooltip=["brand", "count"],
            )
            .properties(title="Top Brands by Clicks", width=450, height=300)
        )
        caption = "<p class='chart-caption'>Horizontal bars highlight which brands users click the most.</p>"
        return self._render_chart(chart, caption)

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

    @staticmethod
    def _render_chart(chart: alt.Chart, caption: str) -> str:
        chart_id = f"chart-{uuid.uuid4().hex}"
        spec = chart.to_dict()
        spec_json = json.dumps(spec)
        return (
            f"<div id='{chart_id}' class='altair-chart'></div>"
            f"<script type='text/javascript'>"
            f"vegaEmbed('#{chart_id}', {spec_json}, {{actions: false}}).catch(console.error);"
            f"</script>"
            f"{caption}"
        )


class ClickedDoc:
    def __init__(self, doc_id, title, description, counter):
        self.doc_id = doc_id
        self.title = title or f"Document {doc_id}"
        self.description = description or "Description not available."
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
