# PaperTradingApp — MVC Architecture Specification

This document defines the official MVC structure of the PaperTradingApp and the
rules that MUST be followed by all contributors.

---


## 1. Layer Responsibilities

### MODEL (`app/model`)
Pure domain logic:
- pricing engines
- hedging engines
- backtesting engines
- yield curve builder
- market data fetchers
- portfolio repository, valuation, stats
- MC pricers / Greeks / heatmaps / surfaces
- trading systems, logs, execution
- dashboard domain logic

Model rules:
- MAY read/write local database (JSON/Cache/CSV)
- MAY call external APIs (Alpaca, Stooq, FRED…)
- MUST NOT contain Streamlit or UI logic.

Cache layout (repo root `cache/`):
- `cache/OHLC/` (historical OHLC & Stooq caches)
- `cache/YahooOptionChains/` (Yahoo-derived option chain/surfaces)
- `cache/AlpacaOptionChains/` (Alpaca option snapshots)
- `cache/IVHistory/` (daily ATM implied-vol observations recorded by 🌡️ Vol Implicite)

---

### CONTROLLER (`app/controller`)
Thin glue layer:
- receives input from the view
- sanitizes parameters
- calls the right MODEL service
- returns data objects ready for the view

Controllers MUST NOT:
- perform domain logic
- access DB directly
- call external APIs directly
- perform formatting/UI work

---

### VIEW (`app/vue`)
Streamlit UI:
- pages
- components
- user interactions
- rendering only

View rules:
- MUST NOT import app.model.* directly
- MUST NOT manipulate business objects directly
- MUST communicate ONLY with controllers

---

### UTILS (`app/utils`)
Minimal stateless helpers:
- io.py
- math_utils.py
- paths.py

And nothing else.

Legacy files (`cache_manager`, `data_loader`, `repository`, `options_text`, `iv_cache`)
were removed via U2 cleanup.

---

## 2. MVC Integrity Checks

### Forbidden:
- View → Model direct import
- Controller → View import
- Model → View import
- utils/ containing domain logic or persistence

### Allowed:
- View → Controller
- Controller → Model
- Model → utils (stateless only)

---

## 3. Maintenance Pipeline

A non-breaking cleaning pipeline is provided under `/scripts/mvc_autofix.yml`.
It detects and fixes:
- legacy utils
- invalid imports
- broken MVC
- stale compatibility files
- missing controller boundaries

To run it manually:
python scripts/run_mvc_autofix.py

To generate the Alpaca optionable tickers CSV required by the Options UI, run:
python scripts/build_optionable_universe.py


---

## 4. Folder Structure Reference



app/
controller/ → routing logic
model/ → domain logic
utils/ → pure helpers
vue/ → Streamlit pages and components


This structure is now validated and clean.

---

## 5. Local configuration (Alpaca keys)

Set Alpaca credentials in `.env` for local dev:
```
APCA_API_KEY_ID=your_key_id
APCA_API_SECRET_KEY=your_secret_key
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

Alternatively, copy `.streamlit/secrets_template.toml` to `.streamlit/secrets.toml`
and fill the same keys. For live trading, use `https://api.alpaca.markets`.

---

## 6. Deploy on Streamlit Community Cloud (streamlit.io)

- **Main file path**: `streamlit_app.py`
- **Python**: set by `runtime.txt`
- **Dependencies**: Streamlit Cloud installs `requirements.txt` (lean); use `requirements_full.txt` only for local dev
- **Secrets**: add these in Streamlit Cloud → App settings → Secrets (see `.streamlit/secrets_template.toml`)
