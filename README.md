# Python & AI for Algorithmic Trading — Code & Notebooks

<p align="right">
  <img src="https://hilpisch.com/tpq_logo_bic.png" alt="The Python Quants" width="25%">
</p>

This repository contains the Jupyter notebooks and Python scripts that accompany the *Python & AI for Algorithmic Trading* class and book in the CPF Program. The material is organised to mirror the structure of the main text:

- Part I — Efficient Markets, Algorithmic Trading, and Python  
- Part II — Python & AI Infrastructure for Algorithmic Trading  
- Part III — Financial Data Engineering and EODHD Integration  
- Part IV — Strategies and Vectorised Backtesting  
- Part V — Event-Based Backtesting and Systems Architecture  
- Part VI — Real-Time Data and Online Algorithms  
- Part VII — Platforms, Broker APIs, and Execution  
- Part VIII — Automation, Deployment, and Risk Management  
- Part IX — Algorithmic Trading and EMH Revisited

The notebooks combine narrative, mathematics, and Python to reproduce central examples from the class, while the scripts provide focused, reusable implementations for figures, diagnostics, streaming sandboxes, and trading experiments.

## Book

The material follows the structure of the book *Python & AI for Algorithmic Trading* by Dr. Yves J. Hilpisch.

<p align="center">
  <img src="https://hilpisch.com/cpf_logo.png" alt="CPF Program" width="35%">
</p>

## Structure

- `notebooks/` — chapter notebooks (`chXX_*.ipynb`) that bring together concepts, code, and plots for each part of the book.
- `code/` — standalone Python modules and helper scripts used for simulations, backtests, streaming examples, and diagnostics.
- `wrappers/` — lightweight client wrappers for external data vendors and broker APIs used by selected chapter scripts.

See the `README.md` files inside `notebooks/` and `code/` for concise per-file overviews.

## Usage

The notebooks are designed to run in a standard scientific Python environment (or in Google Colab) with the usual stack:

- Python 3.11+  
- `numpy`, `pandas`, `matplotlib`  
- `scipy`, `statsmodels` (selected diagnostics)  
- `scikit-learn`, `torch` (predictive models and RL baselines)  
- `nbclient`, `nbformat` (for notebook execution helpers)  
- `zmq`, `sqlite3` and selected broker/API wrappers for streaming and live-style examples

The scripts under `code/` are written so that you can either run them as standalone programs (for example to regenerate figures or start a small tick server) or import their functions into your own research projects.

## Credentials

Some notebooks and scripts call external data vendors, broker APIs, or AI services. These services require API keys or demo account identifiers that you must supply locally.

- Copy `code/creds_.py` to `code/creds.py`, then replace the placeholders with your own values.  
- Keep `code/creds.py` private; it is excluded from version control by `.gitignore`.  
- If a vendor SDK relies on a separate config file (for example `trading_ig`), create it locally and keep it private as well.  

## Disclaimer

This repository and its contents are provided for educational and illustrative purposes only and come without any warranty or guarantees of any kind — express or implied. Use at your own risk. The authors and The Python Quants GmbH are not responsible for any direct or indirect damages, losses, or issues arising from the use of this code. Do not use the provided examples for critical decision‑making, financial transactions, medical advice, or production deployments without rigorous review, testing, and validation.

Some examples may reference third‑party libraries, datasets, services, or application programming interfaces that are subject to their own licenses and terms; you are responsible for ensuring compliance.

## Contact

- Email: [team@tpq.io](mailto:team@tpq.io)  
- Linktree: [linktr.ee/dyjh](https://linktr.ee/dyjh)  
- CPF Program: [python-for-finance.com](https://python-for-finance.com)  
- The AI Engineer: [theaiengineer.dev](https://theaiengineer.dev)  
- The Crypto Engineer: [thecryptoengineer.dev](https://thecryptoengineer.dev)
