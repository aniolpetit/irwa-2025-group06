import os
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session
from flask import request

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
    return render_template(
        'index.html',
        page_title="Welcome",
        ranking_methods=ranking_methods_options,
        selected_ranking_method=selected_method
    )


@app.route('/search', methods=['POST'])
def search_form_post():
    
    search_query = request.form['search-query']
    ranking_method = request.form.get('ranking-method', search_algorithm.DEFAULT_RANKING_METHOD)

    session['last_search_query'] = search_query
    session['last_ranking_method'] = ranking_method

    search_id = analytics_data.save_query_terms(search_query)

    results = search_engine.search(
        search_query,
        search_id,
        corpus,
        ranking_method=ranking_method
    )

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

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    ranking_label = search_algorithm.get_method_label(ranking_method)
    return render_template(
        'results.html',
        results_list=results,
        page_title="Results",
        found_counter=found_count,
        rag_result=rag_result,
        rag_summary=rag_summary,
        search_id=search_id,
        ranking_method_label=ranking_label
    )


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Show document details page with complete product information
    """
    # get the query string parameters from request
    clicked_doc_id = request.args.get("pid")
    search_id = request.args.get("search_id")
    
    if not clicked_doc_id:
        return render_template('doc_details.html', error="No product ID provided", page_title="Error")
    
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
            return render_template('doc_details.html', error="Product not found", page_title="Error")
    
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
    
    # Store click in analytics
    if clicked_doc_id in analytics_data.fact_clicks.keys():
        analytics_data.fact_clicks[clicked_doc_id] += 1
    else:
        analytics_data.fact_clicks[clicked_doc_id] = 1
    
    # Get last search query from session for back to results functionality
    last_search_query = session.get('last_search_query', '')
    
    last_ranking_method = session.get('last_ranking_method', search_algorithm.DEFAULT_RANKING_METHOD)

    # Pass all available document data to template
    return render_template('doc_details.html', 
                          doc=doc_data,
                          description=description,
                          product_details=product_details,
                          last_search_query=last_search_query,
                          last_ranking_method=last_ranking_method,
                          page_title=doc_data.get('title', 'Product Details'))


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """

    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[doc_id]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    for doc in visited_docs: print(doc)
    return render_template('dashboard.html', visited_docs=visited_docs)


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
