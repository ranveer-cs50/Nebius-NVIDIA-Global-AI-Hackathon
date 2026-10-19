# Evidence Desk — web research assistant

A small Streamlit app using Tavily search/extraction, Nebius-hosted NVIDIA Nemotron, and SQLite history. This is a clean replacement prototype, separate from `test5agent.py`; it does not delete or migrate your existing `research_memory.sqlite3`.

## 1. Run locally on Windows PowerShell

Open PowerShell in this project folder. Activate your existing virtual environment if you want to reuse it:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Replace the placeholder values in `.env` with your existing Tavily and Nebius API keys. Do not add quotes unless your key itself needs them. Never commit `.env` to GitHub.

Run the app:

```powershell
streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

## 2. Deploy on Streamlit Community Cloud

1. Create a new GitHub repository for this app.
2. Upload `app.py`, `requirements.txt`, `README.md`, and `.gitignore`. Do **not** upload `.env`, API keys, or your SQLite database.
3. Sign in to Streamlit Community Cloud and create an app from that repository, choosing `app.py` as the entry point.
4. In the app's **Secrets** settings, add TOML values:

```toml
TAVILY_API_KEY = "your-tavily-key"
NEBIUS_API_KEY = "your-nebius-key"
NEBIUS_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
```

5. Deploy and test with a small question first. Recheck Streamlit's current deployment instructions and limits before publishing.

The SQLite history on a hosted instance may not be durable across restarts/redeploys. This prototype is intended for a demo; use managed storage if persistent multi-user history becomes necessary.

## Usage / cost controls

- Default is 3 search queries and 5 extracted pages per research run.
- Every search/extraction/model call may consume provider quota or credits.
- The app does not automatically guarantee factual correctness. Verify important claims by opening sources.
- It uses public-web search; it does not connect to hidden `.onion` services.
