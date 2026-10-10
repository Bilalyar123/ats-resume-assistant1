# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get:

- An overall **ATS score (0-100)** with a weighted category breakdown
- Prioritised **improvements** (High / Medium / Low)
- **Missing keywords** for the target role
- **Before/after rewrite examples** for weak bullet points
- Instant rule-based checks (contact info, sections, metrics, length)

Built with [Streamlit](https://streamlit.io) and Google Gemini Flash.

## How scoring works

Gemini scores five categories; the overall score is calculated in code as a weighted average so it always matches the breakdown.

| Category | Weight |
|---|---|
| Keywords & Relevance | 30% |
| Impact & Achievements | 25% |
| Formatting & ATS Parseability | 20% |
| Content Completeness | 15% |
| Language & Clarity | 10% |

> This is an AI-based estimate, not the output of any specific commercial ATS. Use it as guidance.

## Run locally

1. Install Python 3.10+.
2. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
3. Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

4. Enter your API key in the sidebar, or create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
# GEMINI_MODEL = "gemini-2.5-flash"   # optional override
```

**Never commit `secrets.toml` or your API key to GitHub.**

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt` and `README.md` to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose your repo, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and paste:

```toml
GEMINI_API_KEY = "your-key-here"
```

5. Click **Deploy**.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | Streamlit secrets or environment variable | entered in sidebar |
| `GEMINI_MODEL` | Streamlit secrets or environment variable | `gemini-2.5-flash` |

## Limits and notes

- Max file size 5 MB; only the first ~30,000 characters are analysed.
- Scanned/image-only PDFs are rejected because an ATS cannot read them either.
- Resume text is sent to Google's Gemini API. Do not use real data you are not comfortable sharing.
- If you share your deployed app publicly, your API key quota is used by every visitor. Consider leaving the secret unset so users enter their own key.
