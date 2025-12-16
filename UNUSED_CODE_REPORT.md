**Unused Code Report**

- **Project:** Trading Backtester (repo: backtesting)
- **Generated:** 2025-11-28
- **Scope:** Static analysis starting from the frontend (UI). I searched the frontend for all network calls (fetch/axios) and mapped them to backend route handlers and implementation files. The report lists endpoints called by the UI, their backend handlers, and backend endpoints / code paths that are implemented but have no references from the frontend (likely unused from the UI).

---

**Methodology**
- Scanned frontend code under `frontend/src` for occurrences of `fetch(`, `axios`, `instance.get/post`, and known API path patterns (`/api/`, `/udf/`, `/scripts/`, `/predictions`, `/backtest`, `/ws`, etc.).
- Collected all distinct endpoints the UI calls.
- Scanned backend route modules under `backend/routes/` to map frontend endpoints to backend functions and their implementing files.
- Flagged backend routes and functions that have no references from the frontend as "implemented but not used by UI".

Note: This report is a static cross-reference (code search). It does not execute the app or trace runtime usage. Some endpoints may be used by other clients, by tests, or via direct operator interaction; such cases are noted where obvious.

---

**Frontend-called endpoints (found in UI) → Backend mapping**

- `/api/strategies` (GET)
  - Backend: `backend/routes/strategies.py` → `list_strategies` (router path: `/strategies` on `strategies_router`, included under `/api` in `backend/main.py`)
  - Strategy implementations: `backend/strategies/` (e.g., `moving_average.py`, `sentiment_ml.py`)

- `/api/strategies/{name}` (GET)
  - Backend: `backend/routes/strategies.py` → `get_strategy`
  - Uses: registry lookup `strategy_registry.get(name)` → returns strategy class instances in `backend/strategies/*`

- `/api/strategies/{name}/train` (POST)
  - Backend: `backend/routes/strategies.py` → `train_strategy` → calls `strategy.train()` (e.g. `SentimentMLStrategy.train` in `backend/strategies/sentiment_ml.py`)

- `/api/strategies/{name}/project` (POST)
  - Backend: `backend/routes/strategies.py` → `project_strategy` → calls `strategy.project(...)`.
  - Implementations: `backend/strategies/moving_average.py::MovingAverageStrategy.project`, `backend/strategies/sentiment_ml.py::SentimentMLStrategy.project`.

- `/api/models` (GET)
  - Backend: `backend/routes/models_endpoints.py` → `list_models` (uses `app_state['model_registry']`)

- `/api/models/{modelName}/predict` (POST)
  - Backend: `backend/routes/models_endpoints.py` → `predict_with_model` (uses ModelRegistry to fetch model impl in `backend/models/*`)

- `/api/models/{modelName}/retrain` (POST)
  - Backend: `backend/routes/models_endpoints.py` → `retrain_model` (background retrain support via `_run_retrain_background`)

- `/jobs/{jobId}` (GET) — FRONTEND: `getJobStatus` calls `instance.get(`/jobs/${jobId}`)`
  - Backend: `backend/routes/models_endpoints.py` implements `@router.get('/jobs/{job_id}')` but `models_router` is included with prefix `/api` in `main.py`. Effective backend path is `/api/jobs/{job_id}`.
  - Observation: frontend call `/jobs/{id}` (no `/api` prefix) appears mismatched and will likely 404 unless a proxy rewrites paths. This is a likely dead call in the current configuration.

- `/predict` (POST)
  - Backend: `backend/routes/predictions.py` → `make_prediction`

- `/predictions/recent` (GET)
  - Backend: `backend/routes/predictions.py` → `get_recent_predictions`

- `/predictions/tickers` (GET)
  - Backend: `backend/routes/predictions.py` → `get_available_tickers`

- `/trading/backtest` (GET)
  - Backend: `backend/routes/backtests.py` → `list_backtests`

- `/backtest` (POST)
  - Backend: `backend/routes/backtests.py` → `run_backtest` (invokes `backtest_engine.run_backtest_background` in background tasks)

- `/scripts/execute` (POST)
  - Backend: `backend/routes/scripts.py` → `execute_script` (spawns background task `run_script_async` which maps `script_name` to actual script in `backend/scripts/`)

- `/scripts/status/{executionId}` (GET)
  - Backend: `backend/routes/scripts.py` → `get_script_status`

- `/scripts/executions` (GET)
  - Backend: `backend/routes/scripts.py` → `list_script_executions`

- `/scripts/pipeline/run` (POST)
  - Backend: `backend/routes/scripts.py` → `run_pipeline` (background `run_pipeline_async`)

- `/scripts/pipeline/status/{executionId}` (GET)
  - Backend: `backend/routes/scripts.py` → `get_pipeline_status`

- `/scripts/generate-ma-predictions` (POST)
  - Backend: `backend/routes/scripts.py` → `generate_ma_predictions` (background job)

- `/scripts/generate-ma-predictions/status/{executionId}` (GET)
  - Backend: `backend/routes/scripts.py` → `get_ma_prediction_status`

- `/udf/config` (GET)
  - Backend: `backend/routes/udf.py` → `get_config_endpoint`

- `/udf/symbols?symbol=...` (GET)
  - Backend: `backend/routes/udf.py` → `get_symbol_info` (resolves a single symbol)

- `/udf/history?...` (GET)
  - Backend: `backend/routes/udf.py` → `get_historical_data` (TradingView UDF historical OHLC response)

- `/udf/search?...` (GET)
  - Backend: `backend/routes/udf.py` → `search_symbols`

- `/udf/quotes?symbols=...` (GET)
  - Backend: `backend/routes/udf.py` → `get_quotes`

- WebSocket `/ws` (WS)
  - Backend: `backend/routes/websocket.py` → `websocket_endpoint`. Frontend websocket helper: `frontend/src/services/websocket.ts` connects and sends `subscribe_chart` / `unsubscribe_chart` messages; server tracks `chart_subscriptions` and `broadcast_chart_update`.

---

**Backend endpoints / code implemented but NOT referenced from the frontend (likely unused from UI)**
(Found via cross-reference of frontend searches to backend route definitions.)

1. `GET /trading/predictions` → `backend/routes/predictions.py::get_trading_predictions`
   - Not referenced by frontend. Intended for trading model predictions listing; may be for programmatic use or future UI.

2. `GET /predictions/chart-data/{ticker}` → `backend/routes/predictions.py::get_chart_data`
   - Not referenced by frontend; the frontend charting integration uses UDF endpoints (`/udf/history`) rather than this endpoint.

3. `GET /data/prices/{ticker}` → `backend/routes/data_endpoints.py::get_price_data`
   - Not referenced by frontend (frontend uses UDF endpoints). May be retained for API clients or tests.

4. Monitoring endpoints: `/metrics` and `/monitoring/metrics` → `backend/routes/monitoring.py::get_system_metrics` and `get_monitoring_metrics`
   - No frontend usage found. These are operator/monitoring endpoints (likely intentionally not used by UI).

5. Strategy job status route declared with duplicated prefix: `@router.get("/api/model_jobs/{job_id}")` inside `backend/routes/strategies.py` → effective path likely `/api/api/model_jobs/{job_id}` because the `strategies_router` is included with `prefix='/api'` in `main.py`. No frontend references found — this appears to be a bug / dead endpoint.

6. `GET /predictions/chart-data/{ticker}` and `GET /trading/predictions` (see item 1 & 2) — duplicates of prediction-related functionality exist (UDF vs predictions chart-data) and the UI favors UDF; these prediction-specific endpoints may be unused by current UI.

7. `GET /symbol_info` group handler variant in UDF (`backend/routes/udf.py::get_symbol_info_group`) — frontend uses `/udf/symbols` and not the group `symbol_info` group form; group endpoint may be unused by UI.

8. Many `backend/scripts/*.py` helper scripts are present (e.g., `labeling.py`, `map_articles_to_tickers.py`, etc.). The frontend `Scripts` page exposes `executeScript` and `runPipeline` which can run arbitrary script names; therefore most scripts are reachable but only if a user selects them. I did not find explicit UI links to every specific script, so some scripts may be effectively unused unless invoked manually or by pipeline steps. A deeper audit of `frontend/src/pages/Scripts.tsx` script list would clarify which script names are actually selectable by the UI.

9. Potential unused or mismatched job endpoints:
   - Frontend calls `GET /jobs/{jobId}` (no `/api`), but backend exposes `/api/jobs/{job_id}` (models endpoints router is included under `/api`). This mismatch means the frontend job-status call is likely dead.

10. `backend/routes/predictions.py::generate_prediction(start_date, end_date, tickers)` (module-level function near bottom) appears to be a helper not referenced by UI. It's a small utility for generating predictions; search did not find references from frontend.

---

**Potential bugs / mismatches discovered (likely cause of "dead" calls)**
- `getJobStatus` in `frontend/src/services/api.ts` calls `/jobs/{jobId}`. Backend `models_endpoints.py` registers `/models` and `/models/{name}/predict` and also `@router.get('/jobs/{job_id}')`. In `backend/main.py` the `models_router` is included with `prefix='/api'`, so the canonical job endpoint is `/api/jobs/{job_id}`. The frontend call lacks `/api` and would 404 unless routed differently. Recommendation: standardize to either include `/api` in frontend paths or adjust router prefixes.

- `backend/routes/strategies.py` contains `@router.get("/api/model_jobs/{job_id}")`. Because the `strategies_router` is already included under `/api` in `main.py`, this yields an effective path `/api/api/model_jobs/{job_id}`. This double `/api` prefix is almost certainly unintended and should be fixed to avoid dead endpoints.

---

**Files / components flagged as likely-unused by UI (summary)**
- `backend/routes/predictions.py` — endpoints: `/trading/predictions`, `/predictions/chart-data/{ticker}` (not used by UI)
- `backend/routes/data_endpoints.py` — `/data/prices/{ticker}` (UDF used instead by UI)
- `backend/routes/monitoring.py` — `/metrics`, `/monitoring/metrics` (monitoring-only)
- `backend/routes/strategies.py` — route decorated as `/api/model_jobs/{job_id}` (likely bug -> unused path `/api/api/model_jobs/{id}`)
- `backend/routes/strategies.py` — `get_model_job_status` is not referenced by UI
- `backend/routes/udf.py` — `get_symbol_info_group` (group handler) may be unused by UI which calls `/udf/symbols`
- `backend/routes/models_endpoints.py` vs frontend job path mismatch (`/api/jobs/{id}` vs frontend `/jobs/{id}`)
- `backend/scripts/` directory: several scripts may not be exposed by UI menus; these are candidates for review. Examples: `labeling.py`, `map_articles_to_tickers.py`.

---

**Recommendations / next steps**
- Fix path mismatches:
  - Update frontend `getJobStatus` to call `/api/jobs/{jobId}` (or adjust axios `instance.baseURL` to include `/api` if desired).
  - Fix duplicated prefix in `backend/routes/strategies.py` — change `@router.get("/api/model_jobs/{job_id}")` to `@router.get("/model_jobs/{job_id}")` or similar so effective path is `/api/model_jobs/{job_id}`.
- If the goal is to remove dead code, perform a stricter analysis: search for backend route usages in tests and any external scripts. Some endpoints may be exercised only in tests. I can run a search of `backend/tests` to find references to endpoints and avoid deleting code used by tests.
- For `backend/scripts/`, open `frontend/src/pages/Scripts.tsx` to determine which script names the UI exposes. Remove or archive scripts that are not used by the UI nor scheduled anywhere.
- Consider adding an automated cross-reference tool / script that enumerates backend routes (from `backend/routes`) and checks for occurrences in `frontend/src` to produce a deterministic dead-endpoints list. I can produce such a script if you want.

---

**Limitations**
- This analysis is static (code search) and does not capture runtime usage by other clients, scheduled jobs, or developer manual actions.
- Some endpoints may be intentionally absent from the UI (operator-only / monitoring endpoints). Those are not strictly "dead" but are unused by the UI.

---

If you want, I can next:
- Produce a machine-readable list (CSV or JSON) of backend routes vs frontend references for automated processing.
- Run a search across the `tests/` folder to see which endpoints are covered by tests (to avoid removing test-only code).
- Create a small script that lists all backend routes (parses `backend/routes/*.py`) and cross-references them against frontend code and tests, producing a definitive "no references found" list.

