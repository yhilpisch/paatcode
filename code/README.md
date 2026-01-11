# Code Overview

This folder contains the Python scripts and small helper modules used throughout *Python & AI for Algorithmic Trading*. The scripts are organised by chapter and are designed to be readable and reusable.

## Chapters

- `ch02_random_walk_baseline.py` — random‑walk simulations and diagnostics for EMH-style baselines.  
- `ch02_spy_eod_diagnostics.py` — SPY end‑of‑day efficiency diagnostics and figures.  
- `ch02_spy_extreme_days.py` — identification and summary of extreme daily moves.  
- `ch02_toy_granger_example.py` — synthetic Granger‑style predictability example.  
- `ch04_numpy_matplotlib_equity.py` — NumPy/Matplotlib toy equity-curve example.  
- `ch05_eod_engineering.py` — static end‑of‑day data engineering utilities and figures.  
- `ch06_eodhd_autocorr_demo.py` — EODHD-based autocorrelation diagnostics for daily and intraday data.  
- `ch07_baseline_strategies.py` — SPY buy‑and‑hold and SMA crossover backtests plus metrics.  
- `ch07_eurusd_mean_reversion.py` — EURUSD mean‑reversion example with parameter sweep and figures.  
- `ch08_ols_baseline.py` — ordinary‑least‑squares predictive model and trading strategy on SPY.  
- `ch09_rl_baseline.py` — tabular Q‑learning baseline for daily trading decisions.  
- `ch10_oop_ols_backtest.py` — object‑oriented OLS trading system with reusable components.  
- `ch11_event_backtester.py` — event‑based SMA crossover engine with explicit event loop.  
- `ch14_oanda_eurusd_eod.py` — Oanda EUR_USD daily diagnostics using a lightweight API wrapper.  
- `ch14_oanda_eurusd_intraday_mpl.py` — intraday EUR_USD candle diagnostics and figures.  
- `ch14_oanda_orders_demo.py` — small order‑placement and account‑introspection examples.  
- `ch15_ig_eurusd_eod.py` — IG EURUSD daily diagnostics and summary statistics.  
- `ch15_ig_orders_demo.py` — minimal IG order‑placement and position‑query examples.  
- `ch16_reporting_monitoring.py` — equity‑report generation and position‑summary helpers.  
- `ch17_docker_cloud_production.py` — service specifications and compose-style configuration helpers.  
- `ch18_logging_failure_management.py` — logging configuration and retry logic for small jobs.  
- `ch19_risk_management_post_trading.py` — trade‑level equity and risk analytics.  
- `ch20_ai_enhanced_workflows.py` — prompt templates and transcript helpers for AI-assisted research.  
- `ch21_emh_ai_retail.py` — EMH-style Monte Carlo experiment with random-strategy Sharpe distributions.  
- `ch22_conclusions_outlook.py` — learning‑plan helpers aligned with the book’s outlook.

## Streaming Sandbox

The `code/streaming/` subfolder contains ZeroMQ‑based streaming examples:

- `tick_server.py` — synthetic tick publisher for SPY, GLD, and EURUSD.  
- `client_recorder.py` — tick recorder and intraday diagnostics client.  
- `client_event_backtester.py` — streaming interface to the Chapter 11 event‑based backtester.  
- `client_sma_plotly.py` — live SMA visualisation using Plotly.  
- `client_sqlite_writer.py` — writer that stores ticks in a SQLite database.  
- `sqlite_portfolio_report.py` — small portfolio valuation and report based on stored ticks.  
- `q_tick_monitor.py`, `q_portfolio_watch.py` — lightweight monitoring scripts using the `q` logging helper.

Each script is designed so that you can import its functions into your own projects or run it directly from the command line to reproduce the corresponding experiment or figure.

