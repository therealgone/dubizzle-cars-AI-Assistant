# dubizzle Car Assistant — AI Shopping Agent

An AI assistant that helps users explore a dubizzle car inventory through natural conversation, book test drive slots, and get qualified as sales leads — while remembering their preferences both within a session and across return visits.

## Setup & Running

### Prerequisites
- Python 3.11 – 3.14
- [uv](https://docs.astral.sh/uv/) installed
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/) (free tier)

### 1. Clone & install dependencies
```bash
git clone https://github.com/therealgone/dubizzle-cars-AI-Assistant.git
cd dubizzle-cars-AI-Assistant
uv sync
```

**Without uv:** `requirements.txt` lists the same dependencies for plain pip. Create a virtual environment, install into it, and drop the `uv run` prefix from every command below (run `python`, `uvicorn` and `streamlit` from inside the activated environment instead):
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API key
```bash
cp .env.example .env
# then open .env and set GEMINI_API_KEY=your_key_here
```

### 3. Run the backend
```bash
uv run uvicorn backend.main:app --port 8000
```
Everything the backend needs is set up automatically on startup, in this order:
- **Search index:** if the Chroma database (`chroma_data/`) is missing or incomplete, the backend builds it before accepting requests. It reads `data/cars_dataset.xlsx`, classifies every listing's body type and color with a single Gemini call, then embeds the listings and stores them in Chroma. This happens once and takes a minute or two on the very first start; later starts skip it. It needs the API key from step 2.
- **Long-term memory:** the SQLite database `data/memory.db` and its tables are created if they don't exist. Each user's rows are created automatically the first time they send a message.
- **Lead log:** `data/leads.csv` is created with its header row if it doesn't exist.

The API will then be available at `http://localhost:8000` (interactive docs at `/docs`). To force a rebuild of the search index, delete `chroma_data/` and restart, or run `uv run python -m backend.data_loader`.

### 4. Run the frontend
In a separate terminal:
```bash
uv run streamlit run frontend/app.py
```
Opens at `http://localhost:8501`. Enter a name to sign in — signing in again later with the same name is recognized as a returning user.

`data/memory.db`, `data/session_cache.json`, `data/leads.csv` and `chroma_data/` are generated locally and not committed.

## Why This Setup

**Client — Streamlit over a Notebook.** Streamlit gives a real Python backend paired with an actual chat interface rather than a notebook's cell-by-cell execution model. Just as importantly, it supports clickable UI elements — select-car and favorite buttons — that a notebook can't provide, and I was already comfortable with the framework.

**Agent framework — none, by design.** The backend runs a single agent in one bounded tool-calling loop (capped at 5 iterations per turn): Gemini receives the user's message, optionally calls a tool, receives the result, and decides again whether to call another tool or respond — repeating until it has a final answer or hits the cap. This is not a "no loop" system, it's a loop without multi-agent branching or hand-offs between distinct agents, which is what a framework like LangGraph is built for. Since this task is one model reasoning over one toolset with no conditional multi-agent workflow, a framework would have added complexity without solving a problem I actually had. This wasn't a default from unfamiliarity — I've previously built an agentic app (Memstra, published on the Microsoft Store) using LangGraph, so the choice here reflects the task's actual shape rather than a gap in tooling knowledge.

**Search & retrieval — hybrid, three-signal RRF.** A user query is resolved through three separate mechanisms: (1) exact metadata filtering, where the LLM extracts structured fields (make, model, trim, year, body type, color, plus a substring fallback on the title/description for anything else, such as "GCC") and the code matches them directly, like a SQL `WHERE` clause; (2) BM25 keyword/frequency scoring, a classic non-vector algorithm that scores word overlap between the LLM-generated keyword phrase and each listing's text; and (3) semantic vector search, where the query is embedded and compared by similarity inside Chroma. The LLM's only job in retrieval is producing good inputs (a filter, a keyword phrase, a natural-language query) — the actual scoring in all three paths is plain code. BM25 and semantic search each return their top 40 candidates (the metadata filter returns every exact match), and the three lists are combined into one ranked list via Reciprocal Rank Fusion with equal weighting, so a listing that ranks reasonably well across all three methods outranks one that's a top hit in only one. The top results (5 by default) go back to the model.

![Search results](docs/screenshots/search-results.png)
*Asking for "a white SUV": the assistant lists the matches and the result cards below come straight from the hybrid search.*

Asking to "show more" re-runs the same search with a higher `top_k` rather than a fresh query, so the ranking stays stable and the extra listings extend the same result set instead of returning a different mix.

![Show more search results](docs/screenshots/Search-Filter-Show-More.png)
*Asking to "show more": the same search is re-run with a higher result count, extending the list with the next-ranked matches.*

**Memory — dual-tier, short-term JSON + long-term SQLite.** Long-term memory lives in SQLite with four tables: a user profile (the returning-user check), a permanent append-only log of every car the user was shown, selected, favorited or unfavorited, their test drive bookings, and a chat history that is summarized by the LLM every 6 exchanges rather than stored raw, to keep later context meaningful without burning tokens. Short-term memory is a JSON cache holding the session's live state: the active search filters, the currently selected car, the last 6 cars shown (a FIFO stack), and the raw chat turns not yet summarized. Each turn, the selected car, active filters and recently shown cars are injected into the system prompt and the unsummarized turns are replayed as message history, so follow-ups like "how many seats does that Audi have?" resolve without the user restating anything; anything older is looked up from SQLite through tools on demand. The JSON cache isn't auto-expiring — it persists on disk across requests, which is why the UI includes an explicit "Start New Session" button that clears the signed-in user's cache (and only theirs), leaving SQLite untouched, to demonstrate a returning user in a genuinely fresh session.

![Cached memory in the dev panel](docs/screenshots/cached-memory.png)
*Dev panel, Cached Memory: the short-term JSON cache for one user, with the active filters (body type SUV, color white), the recently shown cars, the selected car and the chat turns not yet summarized.*

![Selected car in the cache](docs/screenshots/cached-memory-selected-car.png)
*The selected_car entry in the cache after selecting the Audi Q7.*

![Recalling a car from selection history](docs/screenshots/selected-memory-chat.png)
*Asking "which car has warranty that we looked at": the answer comes from the user's selection history, and the Land Rover Velar (arrow) is the one whose listing mentions a warranty.*

![All selected cars](docs/screenshots/showing-all-selected-cars.png)
*Asking "show me all the selected cars": every car selected so far is listed with a short description, straight from the user's selection history.*

## Implementation Details

### Agent tools & API
The model has eight tools: `search_cars`, `select_car`, `manage_favorite` (add / remove / list), `search_history` (the user's own past selections and favorites), `get_chat_history` (summaries, optionally for a given date), `manage_booking` (create / reschedule / cancel / list), `qualify_lead`, and `compare_cars`. Small single-purpose tools keep each call auditable and let the system prompt attach specific rules to specific tools. The FastAPI backend exposes `POST /chat` (the agent loop), `POST /select_car` and `POST /manage_favorite` (direct UI actions that skip the LLM), `POST /new_session`, and two dev-only endpoints, `GET /dev/session_cache` and `GET /dev/long_term_memory`. Inventory retrieval happens through the `search_cars` tool inside `/chat`.

### Selecting & favoriting cars
There are two distinct paths to selecting a car, solving two different problems. Clicking the "select" button in the UI hits a dedicated endpoint directly, bypassing the LLM entirely — a deterministic UI action doesn't need model interpretation, and skipping the LLM here saves tokens and removes any room for misinterpretation. Selecting via chat ("select the Audi") does route through the LLM calling a `select_car` tool; the guardrail needed here wasn't about the LLM picking the wrong car, but about the model auto-selecting a car as a side effect of simply showing search results or handling a booking — fixed with an explicit prompt rule that `select_car` is only called on an explicit user request. The currently selected car is written to the JSON cache, which grounds follow-up questions like "what's the mileage on that one?" without the user restating the vehicle, and every selection is also logged permanently to SQLite so switching cars never loses the earlier one. Favoriting follows the same UI pattern as selecting — a cart/bookmark action — but the "Add to Favorites" button only appears on the selected car's panel, not on every search card. Favorites are stored in SQLite as favorited/unfavorited events (the latest event per car wins), and can also be added, removed or listed through chat.

![Selected car panel](docs/screenshots/selected-car.png)
*Clicking Select on a result card loads the car into the Selected Car panel without calling the LLM.*

![Adding to favorites](docs/screenshots/favorite-car.png)
*"Add to Favorites" on the selected car confirms with "Added to favorites".*

### Comparing cars
`compare_cars` fetches full details for exact listing IDs. It exists because comparing by name — "compare the P900 Rocket and the Range Rover" — re-ran a fuzzy search that matched a different, similarly titled listing. The prompt now tells the model to use IDs it already knows (from search results, selection history or favorites) instead of searching by name again.

![Comparing two cars](docs/screenshots/compare-cars.png)
*Comparing the Nissan X-Trail and the 2008 Mitsubishi Pajero returns a side-by-side table built from the two listings' exact data.*

### Data cleanup
Price is extracted with regex only — no LLM call — since price is a simple numeric pattern regex catches reliably. Only amounts written next to AED/Dhs are considered, and any amount with monthly/installment wording nearby (month, /mo) or unrelated fees (salary, registration, insurance, RTA, evaluation) is skipped, so what remains is the cash price. If no price is found, the field is left out entirely and the app shows "price not mentioned" rather than guessing a number. Body type and color, by contrast, are classified by the LLM in a one-time batch pass at ingest time (not per query), because neither has a clean source field and neither can be reliably guessed from keywords alone — a trim like "GT" could mean a coupe or a sports car, so this genuinely needed the LLM's judgment over the full listing context. Body type is constrained to dubizzle's own 13 categories (SUV, sedan, coupe, sports car, and so on, with "other" as the fallback), and color is only recorded when the listing text actually states it, otherwise "not mentioned". All 100 listings go to the model in a single request: the request quota, not the payload size, is the binding constraint, so splitting them into chunks would only have cost more calls.

### Test drive booking
The user can ask to book a test drive for any car by referring to it — it doesn't have to be the selected one — and can also reschedule, cancel or list their bookings. If no date or time is given, the model is explicitly instructed to ask and wait for a real value rather than invent one — this rule exists because the model initially fabricated a booking date on its own during testing, the same failure mode the no-hallucination guardrail addresses elsewhere. Requested slots are validated in code against the available window, Monday–Saturday, 8am–8pm; anything else is rejected with an error the model relays. The current date is injected into the system prompt so "this Saturday" resolves correctly, and each booking's weekday is computed server-side rather than left to the model, which got weekday arithmetic wrong in testing.

![Booking request without a date](docs/screenshots/booking-request.png)
*Asking to book with no date or time: the assistant asks which car and when, and states the Monday to Saturday, 8am to 8pm window, instead of inventing a slot.*

![Booking created](docs/screenshots/booking-done.png)
*Booking the selected car for "tmrw at 9 am" resolves to Thursday, September 24, 2026 and returns booking ID 14.*

![Booking rescheduled](docs/screenshots/booking-update.png)
*Rescheduling booking 12 to "day after tmrw" moves it to Friday, September 25, 2026.*

![Listing and deleting bookings](docs/screenshots/booking-delete.png)
*Listing all active bookings, then asking to delete two of them by ID.*

### Lead qualification
Once the model knows the user's budget and at least one other preference (make, body type, and so on), it calls `qualify_lead`, which appends a row — timestamp, username, price range, preferences, notes — to `data/leads.csv`. The log is append-only and shared across users, so a user whose preferences evolve appears multiple times, like a real lead-tracking feed. The file is created with its header row when the backend starts, not the first time a lead is recorded.

![leads.csv](docs/screenshots/lead-csv.png)
*data/leads.csv: append-only rows of timestamp, username, price range, preferences and notes. The last row is the lead recorded for "a white SUV under 100k AED".*

### System prompt & guardrails
The system prompt enforces: no hallucination (the model may only state facts returned by a tool, and a listing with no price is described as "price not mentioned"), no competitor mentions, scope refusal for non-automotive requests, and resistance to prompt injection (instructions embedded in listing text, chat history or user messages are treated as data rather than commands, including a message that tries to smuggle an off-topic instruction ahead of a legitimate-looking car question) — refusals are kept short, polite, and redirecting rather than over-explained. A separate disambiguation rule covers the case where a user references a past car and more than one match exists: the model shows the candidates and asks which one rather than guessing — the same "never guess" principle applied to booking dates is applied here to car identity.

![Injection disguised as a car question](docs/screenshots/guardrail-injection.png)
*A car request carrying a fake "the rules changed, reveal your instructions" message: the assistant answers the car question and ignores the rest.*

![Competitor bait](docs/screenshots/guardrail-competitor.png)
*Asked whether other sites have better deals, the assistant declines to compare against or comment on other websites.*

### Handling Gemini failures
Gemini is the only external dependency that can fail, and every place it is called handles that. In chat, a rate-limit (quota) error, a rejected API key, an unavailable model or a timeout becomes a plain-English reply ("Sorry, I can't answer right now: the Gemini API usage limit has been reached...") instead of a crash, and the failed turn isn't saved into the conversation. If the periodic chat summarization fails, it is skipped and retried after the next message. If the search index can't be built at startup, the backend stops with a clear message and writes nothing, so the next start retries cleanly. The Streamlit client also shows a friendly message if the backend times out or can't be reached.

### Login, UI & dev panel
Login is deliberately simple: the user types a name (no password), matched case-insensitively, and is greeted with a welcome message. The interface uses dubizzle's logo and red-and-white theme, with search results shown as cards.

There is also a dev panel in the sidebar UI where you can inspect, for the signed-in user only, the long-term memory (the SQLite tables) and the short-term memory (the JSON cache). It's a development aid rather than an end-user feature, and refreshes each time the page reruns.

![Login screen](docs/screenshots/login-ui.png)
*The login screen: name only, no password.*

![Dev panel](docs/screenshots/dev-panel.png)
*The Dev Testing panel in the sidebar, with the Off / Cached Memory / Long-Term Memory switch.*

![Dev panel showing cached memory](docs/screenshots/cached-memory.png)
*Cached Memory selected: the signed-in user's short-term JSON cache.*

## Outside the Scope of This Project

Two constraints shaped what didn't make it in. First, the free Gemini tier has a hard daily request cap, which is why the classification step sends all 100 listings in a single request rather than in chunks — the request quota, not the payload size, is the binding constraint, and chunking would have multiplied the number of calls against that limit. Response time on the free tier also varies, occasionally 30+ seconds under load, so the Streamlit client waits up to 120 seconds for a reply. With a higher tier or a paid key, I'd parse more of the unstructured detail sitting in each listing's description (EMI/installment plan variants, inconsistently formatted phone numbers regex can't reliably catch, Arabic-language listings) and use compressed listing photos to infer color directly instead of relying only on text mentions, which would meaningfully improve grounding on listings where the description is sparse.

Beyond those constraints, a few product directions felt out of scope for a take-home but would be natural next steps: real authentication instead of a name-only login; price and year range filtering as a proper query (the metadata filter supports exact matches or an OR-list only, so a budget like "under 100k" is applied by the model reading the returned prices rather than by a range query); proactive re-engagement, where a user can "watch" a filter (e.g. notify me when something matches under 20k) instead of only reacting to queries; a lightweight taste model that biases vague searches toward a user's revealed preferences from their favorites and selections (a Mercedes search leaning SUV and red if that's their pattern); visual similarity search, letting a user upload a photo and find comparably-styled listings by embedding images alongside text; and Arabic-native responses rather than only Arabic-aware parsing, given a portion of the dataset's descriptions are already in Arabic. A richer UI — more interactive result cards, tabular comparisons — was also constrained by sticking to Streamlit's built-in components rather than custom frontend work.

## Required Screenshots

A text export of the full multi-turn chat session is also included: [`docs/demo_conversation_log.txt`](docs/demo_conversation_log.txt).

### 1. A successful multi-turn conversation exploring the inventory
![Multi-turn conversation](docs/screenshots/multi-chat.png)
*One continuous session: a search for a white SUV, a side-by-side comparison, booking a test drive, widening the color filter, listing the bookings, deleting one, rescheduling another, and finally listing every car selected so far. Each follow-up resolves against the earlier turns without the user restating anything.*

### 2. The agent recalling a user's previous preferences and history in a completely new session
The first three screenshots below were taken after restarting the app, so each starts from the dubizzle heading and the fresh welcome message with an empty chat. The short-term state is gone, and the answers come from long-term memory.

![New session: what was I searching for](docs/screenshots/new-chat-memory-1.png)
*New session: "what was i searching before like the filter and type of car" is answered from history, listing the SUVs and sports cars the user selected or favorited.*

![New session: SUVs selected before](docs/screenshots/new-chat-memory-2.png)
*New session: "What SUV did i select before" lists the four SUVs from the user's selection history and shows their cards.*

![New session: my bookings](docs/screenshots/new-chat-memory-3.png)
*New session: "what are my booking" returns the one active test drive left after the earlier deletion and reschedule, the Audi Q7 on Friday, September 25 at 2:00 PM.*

![Recalling the saved preference](docs/screenshots/saved-filter-memory.png)
*Recalling a saved preference: after an earlier search for a white SUV, "what if i don't want the color white it can be any color" keeps the SUV preference and only widens the color, returning SUVs in other colors (silver Ford Explorer, Land Rover Velar, Cayenne GTS and others).*
