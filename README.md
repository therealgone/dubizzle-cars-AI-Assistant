# Dubizzle Car Shopping Assistant

An AI assistant for exploring a dealership's used-car inventory: natural-language
search, test-drive booking, favorites, lead qualification, and memory of both the
current conversation and past sessions.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11-3.14.

```bash
uv sync
```

Copy `.env.example` to `.env` and add a free [Google AI Studio](https://aistudio.google.com/app/apikey) API key:

```bash
cp .env.example .env
```

Build the search index (reads `data/cars_dataset.xlsx`, classifies body type/color
with the LLM, embeds and stores everything in a local Chroma DB -- takes a couple
of minutes):

```bash
uv run python -m backend.data_loader
```

## Running

Start the backend:

```bash
uv run uvicorn backend.main:app --port 8000
```

In a second terminal, start the frontend:

```bash
uv run streamlit run frontend/app.py
```

Open the Streamlit URL it prints (defaults to `http://localhost:8501`) and enter a
name to begin. Returning users are recognized by that same name.

## Why these choices

**Streamlit over a notebook** -- a chat interface is the natural way to shop for a
car, and Streamlit gives a reactive, clickable UI (search result cards, a "select"
action, a favorites button) with very little code, closer to how the real product
would feel than a notebook would.

**LiteLLM + Gemini** as the agent framework -- LiteLLM's OpenAI-compatible
`completion()`/tool-calling interface meant the orchestration loop (build messages,
call the model, execute any tool calls, feed results back, repeat) didn't need to
change if the model were swapped later. Gemini was used via Google AI Studio's free
tier per the assignment's suggestion.

**Search retrieval is a hybrid of three signals**, fused with reciprocal rank
fusion (RRF): an exact metadata filter (make/model/year/body type/color) for
structured fields the user names explicitly, BM25 for keyword matches, and
semantic (embedding) search for natural-language requests like "a comfortable
family car for road trips." Running all three in parallel and fusing the rankings
performed better in testing than any single method alone -- a pure metadata filter
missed anything phrased loosely, and pure semantic search occasionally surfaced
weak matches for named makes/models that an exact filter gets right every time.

**Memory has two tiers, in two different stores.** Short-term (current
filters, the selected car, recently shown cars, unsummarized chat turns) lives in
a JSON cache keyed by username -- cheap, disposable, and reset when a session
ends. Long-term (selection/favorite history, bookings, and periodically
summarized chat history) lives in SQLite, keyed by username, and survives across
sessions. A "Start New Session" button clears only the short-term cache for that
user, simulating them coming back another day: the agent then has to actually use
its tools (`get_chat_history`, `search_history`, `manage_favorite`) to recall
anything, rather than it just sitting in context.

## Design notes

The agent is built around eight tools (search, select, favorite, booking,
comparison, lead qualification, and two history lookups) rather than a single
do-everything function, so each tool call is small and auditable, and the system
prompt can name specific rules against specific tools (e.g. "never invent a
booking date -- ask instead"). Guardrails against hallucinated inventory facts,
out-of-scope requests, and prompt injection embedded in listing text are enforced
in the system prompt and were checked with adversarial test messages, not just
the happy path. Grounding was a deliberate priority over cleverness: the search
tool returns its nearest matches even when they're an imperfect fit (e.g. asking
for a white SUV can surface a car whose color isn't recorded), and it's the
model's job -- checked in testing -- to only state what the data actually
supports rather than assume a returned result is a perfect match.

Left out as out of scope for the time available: user authentication (a typed
name is trusted as-is, with no password); a real payment/reservation flow beyond
booking a viewing slot; multi-language support; and admin tooling for the
dealership side beyond the plain `leads.csv` log. A production version would also
want stricter numeric range filtering (price/year "between X and Y" currently
relies on the LLM picking sensible filter values rather than a dedicated range
query), and session expiry instead of a manual "new session" button.

## Demo

[docs/demo_conversation_log.txt](docs/demo_conversation_log.txt) is a terminal log
of a live run against the backend showing:

1. A multi-turn conversation exploring the inventory, where a follow-up question
   ("How many seats does *that* Audi have?") resolves against the car shown two
   turns earlier without the user repeating themselves, followed by selecting the
   car, favoriting it, and booking a test drive by relative date ("this Saturday").
2. The same user starting a completely new session (`POST /new_session` clears
   their short-term cache) and asking what they favorited and booked last time --
   answered correctly from long-term SQLite storage alone.
