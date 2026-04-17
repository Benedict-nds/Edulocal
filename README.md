# EduLocal Assistant

A minimal **Streamlit** app for **Ghanaian university** students: paste or upload lecture notes (`.txt` / `.pdf`), ask a study question, and get answers **grounded in those notes** via the **OpenAI** API.

## Features

- **Note-grounded answers** — the model is instructed to use only your supplied notes and to admit when they are insufficient.
- **Source excerpts** — after each reply, a small **Grounding / Source Excerpts** section shows a few short passages from your notes that best match the question (simple keyword overlap over chunks—no embeddings or vector DB).
- **Confidence note** — a short disclaimer under the response; wording is slightly stronger when keyword overlap between the question and the notes looks weak.
- **Response styles** — Normal, Simple explanation, Summary, and **Twi-supported explanation** (English first, then Twi-leaning support for concepts present in the notes).
- **Clear All** — resets pasted notes, question, uploads (via widget key refresh), and the last response/snippets in session state.

## Setup

```bash
cd edulocal-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create or edit `.env`:

```env
OPENAI_API_KEY=your_key_here
# optional:
OPENAI_MODEL=gpt-4o-mini
```

## Run

```bash
streamlit run app.py
```

## Project layout

- `app.py` — Streamlit UI and flow
- `utils/prompt_builder.py` — system prompts and chat message assembly
- `utils/file_parser.py` — text / PDF extraction
- `utils/ai_client.py` — OpenAI chat call
- `utils/grounding.py` — chunking, keyword excerpts, disclaimer heuristic
- `sample_data/` — optional sample notes for trying the app

## Security

Do not commit real API keys. Keep secrets in `.env` (local only) or your environment.
