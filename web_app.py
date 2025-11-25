import re
import os
import time
import uuid
from datetime import datetime
from json import JSONEncoder
from typing import Optional

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session, request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.search.algorithms import SearchAlgorithm
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")

# Load documents corpus into memory (for display purposes)
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)

# Initialize search algorithm with processed corpus (contains tokens for indexing)
processed_corpus_path = os.path.join(path, "project_progress", "part_1", "data", "processed_corpus.json")
print(f"\nInitializing search algorithm with corpus: {processed_corpus_path}")
search_algorithm = SearchAlgorithm(processed_corpus_path)

# Instantiate search engine with the algorithm
search_engine = SearchEngine(search_algorithm)
ranking_methods_options = search_algorithm.get_available_methods()

# Instantiate our in memory persistence
analytics_data = AnalyticsData()

# Instantiate RAG generator
rag_generator = RAGGenerator()

def _highlight_text(text: str, query_terms: list) -> str:
    """
    Wrap occurrences of query terms in <strong>..</strong>. Case-insensitive.
    Longer terms matched first to avoid nested highlights.
    """
    if not text:
        return text
    # Avoid double-highlighting: work on a copy
    highlighted = text
    # Sort by length desc so longer terms replaced first
    for term in sorted(set(query_terms), key=len, reverse=True):
        term = term.strip()
        if not term:
            continue
        escaped = re.escape(term)
        # Case-insensitive replacement, preserve original case
        highlighted = re.sub(f"(?i)({escaped})", r"<strong>\1</strong>", highlighted)
    return highlighted

def parse_rag_summary(summary_text: str):
    """
    Parse a free-form LLM response into structured sections for display.
    """
    summary = {
        "raw": summary_text.strip() if summary_text else "",
        "best": None,
        "why": None,
        "alternative": None,
        "extra": []
    }

    if not summary_text:
        return summary

    lines = summary_text.replace("\r", "").splitlines()
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        normalized = cleaned.lstrip("-•").strip()
        lower = normalized.lower()
        if lower.startswith("best product:"):
            summary["best"] = normalized.split(":", 1)[1].strip()
        elif lower.startswith("why:"):
            summary["why"] = normalized.split(":", 1)[1].strip()
        elif lower.startswith("alternative"):
            summary["alternative"] = normalized.split(":", 1)[1].strip()
        else:
            summary["extra"].append(normalized)

    return summary


# Home URL "/"
@app.route('/')
def index():
    session_id, context = _prepare_request_context()
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    selected_method = session.get('last_ranking_method', search_algorithm.DEFAULT_RANKING_METHOD)
    response = render_template(
        'index.html',
        page_title="Welcome",
        ranking_methods=ranking_methods_options,
        selected_ranking_method=selected_method
    )
    _log_request(session_id, context)
    return response


@app.route('/search', methods=['POST'])
def search_form_post():
    search_query = request.form['search-query']
    session_id, context = _prepare_request_context()
    visitor_id = _ensure_visitor_id()
    mission_id = _ensure_mission(session_id)
    start_time = time.perf_counter()
    country_override = request.form.get("country") or None
    city_override = request.form.get("city") or None
    geo_source = request.form.get("geo_source") or None
    _persist_geo_context(country_override, city_override, geo_source)
    if country_override:
        context["country"] = country_override
    if city_override:
        context["city"] = city_override
    ranking_method = request.form.get('ranking-method', search_algorithm.DEFAULT_RANKING_METHOD)

    session['last_search_query'] = search_query
    session['last_ranking_method'] = ranking_method

    search_id = analytics_data.save_query_terms(
        search_query,
        browser_label=context["browser_label"],
        session_id=session_id,
        mission_id=mission_id,
        device_type=context["device_type"],
        os_label=context["os_label"],
        visitor_id=visitor_id,
        ip_address=context["ip_address"],
        country=context.get("country"),
        city=context.get("city"),
    )

    results = search_engine.search(
        search_query,
        search_id,
        corpus,
        ranking_method=ranking_method
    )
    found_count = len(results)
    analytics_data.update_query_results(search_id, found_count)

    # generate RAG response based on user query and retrieved results
    rag_result = rag_generator.generate_response(search_query, results)
    if not isinstance(rag_result, dict):
        rag_result = {"text": rag_result, "provider": None, "model": None}
    rag_text = rag_result.get("text")
    rag_summary = parse_rag_summary(rag_text)
    print(
        "RAG response metadata:",
        {
            "provider": rag_result.get("provider"),
            "model": rag_result.get("model"),
        }
    )
    print("RAG response:", rag_text)

    session['last_found_count'] = found_count

    print(session)

    ranking_label = search_algorithm.get_method_label(ranking_method)
    response = render_template(
        'results.html',
        results_list=results,
        page_title="Results",
        found_counter=found_count,
        rag_result=rag_result,
        rag_summary=rag_summary,
        search_id=search_id,
        ranking_method_label=ranking_label
    )
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    _log_request(session_id, context, latency_ms=latency_ms)
    return response


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Show document details page with complete product information
    """
    session_id, context = _prepare_request_context()
    # get the query string parameters from request
    clicked_doc_id = request.args.get("pid")
    search_id = request.args.get("search_id")
    rank_position_param = request.args.get("rank")

    if not clicked_doc_id:
        response = render_template('doc_details.html', error="No product ID provided", page_title="Error")
        _log_request(session_id, context, status_code=400)
        return response

    # Get document data from search algorithm's processed corpus
    doc_data = search_algorithm.get_document_by_id(clicked_doc_id)

    if not doc_data:
        # Fallback to display corpus
        doc = corpus.get(clicked_doc_id)
        if doc:
            doc_data = {
                'pid': doc.pid,
                'title': doc.title,
                'description': doc.description,
                'brand': doc.brand,
                'category': doc.category,
                'sub_category': doc.sub_category,
                'product_details': doc.product_details,
                'seller': doc.seller,
                'out_of_stock': doc.out_of_stock,
                'selling_price': doc.selling_price,
                'discount': doc.discount,
                'actual_price': doc.actual_price,
                'average_rating': doc.average_rating,
                'url': doc.original_url or doc.url,
                'images': doc.images
            }
        else:
            response = render_template('doc_details.html', error="Product not found", page_title="Error")
            _log_request(session_id, context, status_code=404)
            return response

    # Handle description fallback
    description = doc_data.get('description')
    if not description or (isinstance(description, str) and description.strip() == ''):
        description = doc_data.get('full_text', '')
    if not description or (isinstance(description, str) and description.strip() == ''):
        description = doc_data.get('title', 'No description available')

    # Handle product_details
    product_details = doc_data.get('product_details')
    if product_details is not None and not isinstance(product_details, dict):
        product_details = None

    query_id = None
    if search_id:
        try:
            query_id = int(search_id)
        except (ValueError, TypeError):
            query_id = None

    # Store click in analytics
    rank_position = None
    if rank_position_param:
        try:
            rank_position = int(rank_position_param)
        except (ValueError, TypeError):
            rank_position = None

    analytics_data.register_click(
        doc_data,
        query_id=query_id,
        session_id=session_id,
        rank_position=rank_position,
    )

    # Get last search query/ranking to support back navigation
    last_search_query = session.get('last_search_query', '')
    last_ranking_method = session.get('last_ranking_method', search_algorithm.DEFAULT_RANKING_METHOD)

    # Pass all available document data to template
    response = render_template(
        'doc_details.html',
        doc=doc_data,
        description=description,
        product_details=product_details,
        last_search_query=last_search_query,
        last_ranking_method=last_ranking_method,
        page_title=doc_data.get('title', 'Product Details'),
        search_id=search_id,
    )
    _log_request(session_id, context)
    return response


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """

    session_id, context = _prepare_request_context()
    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    response = render_template('stats.html', clicks_data=docs)
    _log_request(session_id, context)
    return response


@app.route('/dashboard', methods=['GET'])
def dashboard():
    session_id, context = _prepare_request_context()
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document | None = corpus.get(doc_id)
        doc = ClickedDoc(
            doc_id=doc_id,
            title=getattr(d, "title", None) if d else None,
            description=getattr(d, "description", None) if d else None,
            counter=analytics_data.fact_clicks[doc_id],
        )
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs:
        print(doc)
    analytics_summary = {
        "key_indicators": analytics_data.get_key_indicators(),
        "requests": analytics_data.get_request_summary(),
        "sessions": analytics_data.get_session_overview(),
        "missions": analytics_data.get_mission_overview(),
        "clicks": analytics_data.get_click_metrics(),
        "dwell_stats": analytics_data.get_dwell_time_statistics(),
        "geo": analytics_data.get_geo_distribution(),
        "devices": analytics_data.get_device_share(),
        "os": analytics_data.get_os_share(),
        "top_queries": analytics_data.get_top_queries(),
        "recent_queries": analytics_data.get_recent_queries(),
        "browser_share": analytics_data.get_browser_share(),
        "top_brands": analytics_data.get_top_brands(),
        "price_buckets": analytics_data.get_price_breakdown(),
    }
    charts = {
        "views": analytics_data.plot_number_of_views(),
        "sessions_by_hour": analytics_data.plot_sessions_by_hour(),
        "status_codes": analytics_data.plot_requests_by_status(),
        "dwell_hist": analytics_data.plot_dwell_time_distribution(),
        "price_sensitivity": analytics_data.plot_price_sensitivity(),
        "top_brands": analytics_data.plot_top_brands(),
    }

    response = render_template(
        'dashboard.html',
        visited_docs=visited_docs,
        analytics_summary=analytics_summary,
        charts=charts,
        page_title="Dashboard",
    )
    _log_request(session_id, context)
    return response


def _detect_browser_label(user_agent_header: str | None) -> str:
    """
    Build a compact label describing the browser + platform for analytics.
    """
    if not user_agent_header:
        return "Unknown"

    agent = httpagentparser.detect(user_agent_header)
    browser = agent.get('browser', {}).get('name')
    platform = agent.get('platform', {}).get('name')

    label_parts = [part for part in [browser, platform] if part]
    return " / ".join(label_parts) if label_parts else "Unknown"


def _ensure_visitor_id() -> str:
    visitor_id = session.get("visitor_id")
    if not visitor_id:
        visitor_id = str(uuid.uuid4())
        session["visitor_id"] = visitor_id
    return visitor_id


def _get_request_context() -> dict:
    user_agent_header = request.headers.get('User-Agent', '')
    agent = httpagentparser.detect(user_agent_header or "")
    device_type = agent.get("device", {}).get("name") or agent.get("platform", {}).get("name")
    os_label = agent.get("os", {}).get("name") or agent.get("platform", {}).get("name")
    ip_address = request.headers.get("X-Forwarded-For", request.remote_addr)
    country = session.get("geo_country")
    city = session.get("geo_city")
    return {
        "browser_label": _detect_browser_label(user_agent_header),
        "device_type": device_type or "Unknown",
        "os_label": os_label or "Unknown",
        "ip_address": ip_address or "Unknown",
        "country": country,
        "city": city,
    }


def _ensure_analytics_session(context: dict) -> str:
    session_id = session.get("analytics_session_id")
    visitor_id = _ensure_visitor_id()
    if not session_id or session_id not in analytics_data.sessions:
        session_id = analytics_data.start_session(
            visitor_id=visitor_id,
            browser_label=context["browser_label"],
            device_type=context["device_type"],
            os_label=context["os_label"],
            ip_address=context["ip_address"],
            country=context.get("country"),
            city=context.get("city"),
        )
        session["analytics_session_id"] = session_id
    return session_id


def _ensure_mission(session_id: str) -> str:
    mission_id = session.get("analytics_mission_id")
    if mission_id and mission_id in analytics_data.missions:
        return mission_id
    mission_id = analytics_data.start_mission(session_id, goal="Search journey")
    session["analytics_mission_id"] = mission_id
    return mission_id


def _prepare_request_context() -> tuple[str, dict]:
    context = _get_request_context()
    session_id = _ensure_analytics_session(context)
    return session_id, context


def _persist_geo_context(country: Optional[str], city: Optional[str], source: Optional[str]):
    if country:
        session["geo_country"] = country
    if city:
        session["geo_city"] = city
    if source:
        session["geo_source"] = source


def _log_request(session_id: str, context: dict, status_code: int = 200, latency_ms: float | None = None):
    analytics_data.record_request(
        path=request.path,
        method=request.method,
        status_code=status_code,
        session_id=session_id,
        visitor_id=_ensure_visitor_id(),
        browser_label=context["browser_label"],
        device_type=context["device_type"],
        os_label=context["os_label"],
        ip_address=context["ip_address"],
        country=context.get("country"),
        city=context.get("city"),
        latency_ms=latency_ms,
    )


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


@app.route('/track_dwell', methods=['POST'])
def track_dwell():
    session_id, context = _prepare_request_context()
    data = request.get_json(silent=True) or {}
    doc_id = data.get("doc_id")
    dwell_ms = data.get("dwell_ms")
    search_id = data.get("search_id")

    try:
        dwell_ms = int(float(dwell_ms))
    except (TypeError, ValueError):
        dwell_ms = None

    try:
        query_id = int(search_id) if search_id is not None else None
    except (TypeError, ValueError):
        query_id = None

    if doc_id and dwell_ms is not None:
        analytics_data.update_click_dwell(query_id=query_id, doc_id=doc_id, dwell_time_ms=dwell_ms)

    _log_request(session_id, context)
    return ("", 204)


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
