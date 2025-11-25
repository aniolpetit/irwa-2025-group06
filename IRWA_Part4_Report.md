# Part 4: RAG, User Interface and Web Analytics
**Github URL**: https://github.com/aniolpetit/irwa-2025-group06

**Github TAG**: IRWA-2025-part-4

**Date:** November 29, 2025

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [User Interface](#2-user-interface)
   - 2.1 [Search Page](#21-search-page)
   - 2.2 [Search Algorithms Integration](#22-search-algorithms-integration)
   - 2.3 [Results Page](#23-results-page)
   - 2.4 [Document Details Page](#24-document-details-page)
3. [RAG Implementation](#3-rag-implementation)
4. [Web Analytics](#4-web-analytics)
5. [Conclusion](#5-conclusion)

---

## 1. Introduction

This report documents the implementation of Part 4 of the IRWA Final Project, which focuses on creating a complete web application with Retrieval-Augmented Generation (RAG), a user-friendly interface, and web analytics capabilities. The application provides users with an intuitive search interface for fashion products, AI-powered recommendations, and comprehensive usage statistics.

---

## 2. User Interface

### 2.1 Search Page

The search page was enhanced to provide a better user experience. The HTML structure was reorganized to create a more prominent and centered search interface. A descriptive page title was added above the search form to guide users. The search input field was made more prominent with improved sizing and clearer placeholder text. The form layout was restructured to ensure proper alignment between the input field and search button.

Client-side validation was implemented to prevent users from submitting empty queries, replacing the previous non-functional validation. Additional CSS styling was added to improve the visual presentation, including better centering of the search interface, responsive design considerations, and enhanced visual feedback through focus states and hover effects.

### 2.2 Search Algorithms Integration

PREGUNTA: ÉS MILLOR FER SERVIR TF-IDF OR BM25?

The search functionality was integrated with the TF-IDF ranking algorithm developed in Part 2. A new `SearchAlgorithm` class was created in `myapp/search/algorithms.py` to wrap the TF-IDF ranker and inverted index, making them suitable for web application use. The implementation loads the processed corpus data (which contains preprocessed tokens) and builds the inverted index at initialization time for optimal performance.

The `SearchEngine` class was refactored to use the integrated search algorithm instead of the dummy random search. The search process now performs proper query preprocessing, conjunctive query filtering, and TF-IDF-based ranking to return the most relevant results. The algorithm is initialized once at application startup, ensuring fast response times for user queries.

To ensure accurate result display, the search engine retrieves document data directly from the processed corpus used for indexing, rather than a separate display corpus. This guarantees that all document fields (including descriptions, metadata, and product details) are available and consistent with the indexed data. A fallback mechanism was implemented for description fields: if the primary description is missing, the system falls back to the full text field, and if that is also unavailable, it uses the title as a last resort.

The results template was updated to properly handle missing or None values. Date fields are only displayed when available, and descriptions show appropriate fallback messages when data is missing. The Document model was extended to include ranking scores and additional fields needed for result display, including proper handling of optional fields like crawled dates.

### 2.3 Results Page

The results page was enhanced to display comprehensive product information for each search result. The template was restructured to show all required document properties including title, description, selling price, discount percentage, average rating, and product metadata such as brand and category. Additional relevant information was added including stock availability status and original price display when discounts are available.

The Document model was extended with an `original_url` field to separate the document details page URL from the original product website URL. The title link was configured to navigate to the document details page, while a separate "View on original website" link was added to access the original product page. This separation allows users to access both detailed product information within the application and the original source.

Spacing between metadata fields was improved to enhance readability. The layout was organized with proper visual separation between different information elements, making it easier for users to scan and understand product details. Price formatting was implemented to display currency values appropriately, and rating displays were enhanced with visual indicators.

### 2.4 Document Details Page

A comprehensive document details page was implemented to display complete product information. The route was updated to retrieve document data from the processed corpus, ensuring all available fields are accessible. The page uses a two-column layout with the main content area showing product images, description, and product details, while a sidebar displays pricing, product information, rating, and availability.

Navigation functionality was added with two buttons at the top: one to return to the search page and another to go back to the search results for the same query. The "Back to Results" button uses a form submission to resubmit the last search query, allowing users to continue browsing results from the same search session.

Product images were made interactive with a modal lightbox feature. Clicking on any image opens a modal overlay displaying the image at a larger size while maintaining aspect ratio. The modal is centered on screen and can be closed by clicking the close button, clicking outside the image, or pressing the Escape key.

The page displays all relevant document properties prominently, including title, description, images, pricing, discount, rating, brand, category, seller, and stock status. An "Additional Information" section was added at the bottom to display less prominent metadata such as product ID, document ID, crawled timestamp, and full text, using smaller font sizes and muted colors to maintain visual hierarchy while ensuring all available information is accessible.


## 3. RAG Implementation

A Retrieval-Augmented Generation (RAG) system was implemented to provide AI-generated summaries of search results. The `RAGGenerator` class was created in `myapp/generation/rag.py` to integrate with the Groq API, a third-party LLM service. The RAG system takes the user's search query and the retrieved product documents, then generates intelligent recommendations highlighting the best matching products.

The RAG generator was integrated into the search workflow. After retrieving search results using the TF-IDF ranking algorithm, the system passes both the user query and the ranked results to the RAG generator. The generator formats comprehensive product information including title, description, brand, price, discount, rating, and stock status for each result, then sends this context to the LLM along with the user's query.

The prompt template was designed to instruct the LLM to act as a product advisor, identifying the best matching product and explaining why it fits the user's needs. The system includes error handling to gracefully fall back to a default message when API credentials are missing or API calls fail. The generated summary is displayed prominently at the top of the results page in a dedicated section, providing users with immediate AI-powered insights before they review individual results.


## 4. Web Analytics

### 4.1 Data collection

We extended `AnalyticsData` into an in-memory star schema (a data model with fact tables surrounded by dimension tables) that captures every relevant interaction described in the statement:

1. **HTTP requests:** `record_request` captures method, path, status code, latency (time spent serving a request), bytes sent, and the associated session identifier so we can reason about reliability issues and detect “hot” endpoints.
2. **Sessions and missions:** `start_session` and `start_mission` let us differentiate physical sessions (a sit-down tracked with a timeout-based heuristic) from logical missions (a sequence of related queries). For simplicity we automatically start one mission whenever a new analytics session begins and reuse it for all searches during that sit-down, which keeps the mapping deterministic without asking the user to label their intent. Sessions track visitor attributes such as user agent, operating system, device type, IP address, and optional country/city. Geo data is collected with a client-side consent prompt that calls `ipapi.co`; the chosen country/city is cached in the session so all subsequent requests inherit it.
3. **Queries:** Every POST to `/search` calls `save_query_terms`, which normalizes the text, records the number of terms, order-of-arrival within the session/mission, and links the query to the visitor context mentioned above. Once ranking completes, `update_query_results` stores the result count so we can compute the zero-result rate.
4. **Results and clicks:** `register_click` accepts either Pydantic documents or plain dictionaries and records the clicked product identifier, title, brand, category, price bucket, rank position (the ordering that the engine displayed), and a dwell time placeholder. A JavaScript beacon on the document details page sends the actual dwell time when the user navigates away, and `update_click_dwell` patches the click record accordingly. Each click also keeps a pointer to the originating query so attribution is preserved.
5. **Visitor context:** `_update_context_counters` classifies device type, OS, and geography (including “Unknown” if the user declines sharing) and buckets activity by hour of day. `_find_active_session` merges consecutive requests from the same visitor into a single session until a configurable timeout elapses.

All collectors append to Python data classes (`SessionRecord`, `MissionRecord`, `RequestRecord`, `QueryRecord`, `ClickRecord`), which makes the instrumentation deterministic and reproducible without external databases.

### 4.2 Data storage model

The star schema consists of three main fact tables:

- **Request fact** (`fact_requests`): every HTTP interaction with latency statistics and status code counters for uptime reporting.
- **Session fact** (`sessions` + `missions`): holds physical sessions, logical missions, and sequencing metadata (order of queries, mission goals, session start/end timestamps, and inferred dwell times).
- **Click fact** (`fact_click_events`): attribution-ready click rows that join back to queries and sessions for multi-dimensional breakdowns.

Dimension-like counters (browser share, device share, OS share, geographic pairs, price buckets) are kept in `Counter` collections so dashboards can aggregate them quickly. This mirrors a traditional analytics warehouse while remaining easy to reset during grading.

### 4.3 Dashboard and indicators

The `/dashboard` endpoint pulls every aggregation into a single `analytics_summary` dictionary and a `charts` bundle so the template can focus on layout. The page is split into three layers:

1. **KPI cards and tables** – Total requests, sessions, queries, zero-result rate, average latency/session/dwell, active missions, request status breakdown, top paths, device share, OS share, geo distribution, top queries, recent queries, price buckets, click behaviour (including ranking positions and dwell percentiles), and a “most clicked documents” table (title + description fallback). We also annotate the mission section with a short reminder that each “search journey” corresponds to the auto-started mission per session.
2. **Charts powered by Altair/Vega-Lite** – We render reusable specifications for document views vs clicks, sessions by hour, HTTP status distribution, dwell-time histogram, price sensitivity pie (with percentage labels), and top brands. Charts are embedded via `vegaEmbed` with custom sizing so they stack vertically and fit the page width.
3. **Request/session instrumentation** – Every route (`/`, `/search`, `/doc_details`, `/stats`, `/dashboard`, `/track_dwell`) now bootstraps the analytics session, logs the HTTP request, and forwards the visitor context to `AnalyticsData`. The search results template includes the rank position inside the `/doc_details` link, ensuring clicks are attributed to their slot. The document-details page hosts the dwell-time beacon and exposes the `search_id` so clicks link back to queries.

Because all metrics and charts read from the same in-memory warehouse, instructors can replay traffic in their environment and instantly see the dashboard update without provisioning external services.

### 4.4 Reproducing the analytics demo

To populate every widget in a fresh environment, run the following manual test:

1. Start the Flask app, open two browser contexts (normal + incognito), and consent to the geo prompt in one of them.
2. In the first window, submit at least four queries (include a nonsense query to trigger zero-result counts), open the top three documents per query, wait a few seconds, then return to the results page so dwell events fire. Repeat one query twice to populate “Top Queries”.
3. In the second window, decline the geo prompt, run two different queries, and open at least one result per query; leave one detail tab open for ≈10 seconds to create longer dwell samples.
4. Visit `/dashboard` and `/stats` from both windows to log request traffic. Optionally hit a missing route to record a 404, so the status breakdown chart shows more than HTTP 200s.

Following those steps yields non-zero values everywhere: KPI cards update, missions show as “search journey”, the geo/device/OS counters differentiate “Unknown” from real locations, price/brand charts render with slices/bars, and the dwell histogram displays returning-time buckets.

