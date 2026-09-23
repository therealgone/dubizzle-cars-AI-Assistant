import json

import litellm

import backend.memory as memory
from backend.car_search import _collection, search_cars
from backend.config import GEMINI_API_KEY
from backend.enrichment import BODY_TYPES
from backend.llm_client import MODEL, describe_llm_error

# ---------------------------------------------------------------------------
# Tool schemas (litellm/OpenAI function-calling format) -- what the LLM sees.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_cars",
            "description": (
                "Search the car listings. Use filters for exact structured fields "
                "the user named, keywords for a short phrase to keyword-match, and "
                "semantic_query for the user's natural-language request. Provide "
                "whichever of the three actually apply -- not all are required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "description": (
                            "Exact-match fields. Any value can be a single value or "
                            "a list (list means OR -- match any of these)."
                        ),
                        "properties": {
                            "make": {"type": "string"},
                            "model": {"type": "string"},
                            "trim": {"type": "string"},
                            "year": {"description": "int, or a list of ints for a range/OR"},
                            "body_type": {"type": "string", "enum": BODY_TYPES},
                            "color": {"type": "string"},
                            "description": {
                                "type": "string",
                                "description": (
                                    "Fallback substring match for anything not covered by "
                                    "the fields above, e.g. 'GCC', 'black interior'."
                                ),
                            },
                        },
                    },
                    "keywords": {"type": "string", "description": "short keyword phrase for BM25 search"},
                    "semantic_query": {"type": "string", "description": "natural-language description of what's wanted"},
                    "top_k": {
                        "type": "integer",
                        "description": "how many results to return, default 5. Raise this when the user asks to see 'all' or 'every' matching listing.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_car",
            "description": "Mark a specific listing as the one the user is now focused on, so follow-up questions resolve against it.",
            "parameters": {
                "type": "object",
                "properties": {"listing_id": {"type": "integer"}},
                "required": ["listing_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_favorite",
            "description": "Add, remove, or list every car the user has ever favorited.",
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_id": {"type": "integer", "description": "required for add/remove, omit for list"},
                    "action": {"type": "string", "enum": ["add", "remove", "list"]},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_history",
            "description": "Search the user's OWN past selected/favorited cars by keyword (make/model/title). Omit keyword to list everything.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "event_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["selected", "favorited"]},
                    },
                    "limit": {
                        "type": "integer",
                        "description": "max results, default 10. Raise this when the user asks for 'all' the cars they've selected or favorited.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chat_history",
            "description": "Recall a summary of past conversation, optionally for a specific date -- e.g. 'what did we talk about yesterday'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "manage_booking",
            "description": "Create, reschedule, cancel, or list test-drive bookings. Bookings are only available Mon-Sat, 8am-8pm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "reschedule", "cancel", "list"]},
                    "listing_id": {"type": "integer", "description": "required for create; for reschedule, only pass this if the user wants to change WHICH CAR the booking is for"},
                    "booking_id": {"type": "integer", "description": "required for reschedule/cancel: the booking's own id from a list result -- NOT the car's listing_id. Call list first if you don't have it."},
                    "date": {"type": "string", "description": "YYYY-MM-DD, required for create; for reschedule, only pass this if the user wants to change the date"},
                    "time": {"type": "string", "description": "HH:MM 24h, required for create; for reschedule, only pass this if the user wants to change the time"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "qualify_lead",
            "description": (
                "Record this user as a sales lead once you know their budget AND at least "
                "one other preference (make, body type, etc). Call again whenever their "
                "stated preferences meaningfully change or grow -- this is an append-only "
                "log, not a profile you overwrite, so each call adds a new record of where "
                "their interest stood at that point."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "price_range": {"type": "string", "description": "the user's stated budget, in their own words, e.g. 'around 100k AED' or '50000-80000 AED'"},
                    "preferences": {"type": "string", "description": "what they're looking for, e.g. 'SUV, Mercedes-Benz preferred, black or white'"},
                    "notes": {"type": "string", "description": "anything else relevant -- financing interest, urgency, family size, etc."},
                },
                "required": ["price_range", "preferences"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_cars",
            "description": (
                "Get full details for specific listings by their exact listing_id, to "
                "compare them. Use this instead of search_cars whenever the user refers to "
                "cars you already know the listing_id for (e.g. from their selection "
                "history, favorites, or cars shown earlier this conversation) -- searching "
                "by name again risks matching a different, similarly-named listing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["listing_ids"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatch -- what actually runs when the LLM calls a tool.
# ---------------------------------------------------------------------------

def _fetch_car(listing_id: int) -> dict | None:
    fetched = _collection.get(ids=[str(listing_id)], include=["metadatas"])
    return fetched["metadatas"][0] if fetched["metadatas"] else None


def _fetch_cars(listing_ids: list[int]) -> list[dict]:
    if not listing_ids:
        return []
    fetched = _collection.get(ids=[str(i) for i in listing_ids], include=["metadatas"])
    by_id = {int(i): m for i, m in zip(fetched["ids"], fetched["metadatas"])}
    return [by_id[i] for i in listing_ids if i in by_id]


def tool_search_cars(username: str, session: dict, filters: dict | None = None,
                      keywords: str | None = None, semantic_query: str | None = None,
                      top_k: int = 5) -> list[dict]:
    results = search_cars(filters=filters, keywords=keywords, semantic_query=semantic_query, top_k=top_k)
    if filters:
        memory.update_active_filters(session, filters)
    memory.push_car_stack(session, results)
    for car in results:
        memory.log_interaction(username, car["listing_id"], "shown")
    return results


def tool_select_car(username: str, session: dict, listing_id: int) -> dict:
    car = _fetch_car(listing_id)
    if car is None:
        return {"error": f"no listing with id {listing_id}"}
    memory.select_car(username, session, car)
    return car


def tool_manage_favorite(username: str, session: dict, action: str, listing_id: int | None = None) -> dict | list[dict]:
    if action == "list":
        return _fetch_cars(memory.get_favorites(username))
    event = "favorited" if action == "add" else "unfavorited"
    memory.log_interaction(username, listing_id, event)
    return {"listing_id": listing_id, "favorited": action == "add"}


def tool_search_history(username: str, session: dict, keyword: str | None = None,
                         event_types: list[str] | None = None, limit: int = 10) -> list[dict]:
    return memory.search_history(username, keyword=keyword, event_types=event_types, limit=limit)


def tool_get_chat_history(username: str, session: dict, on_date: str | None = None, limit: int = 20) -> list[dict]:
    return memory.get_chat_history(username, limit=limit, on_date=on_date)


def tool_manage_booking(username: str, session: dict, action: str, listing_id: int | None = None,
                         booking_id: int | None = None, date: str | None = None, time: str | None = None) -> dict | list[dict]:
    try:
        if action == "create":
            return memory.create_booking(username, listing_id, date, time)
        if action == "reschedule":
            return memory.reschedule_booking(username, booking_id, date, time, listing_id=listing_id)
        if action == "cancel":
            return memory.cancel_booking(username, booking_id)
        if action == "list":
            return memory.get_bookings(username)
        return {"error": f"unknown action {action}"}
    except ValueError as e:
        return {"error": str(e)}


def tool_qualify_lead(username: str, session: dict, price_range: str, preferences: str, notes: str = "") -> dict:
    if not memory.lead_is_new(username, price_range, preferences):
        return {"status": "already recorded -- no change from this user's last lead"}
    memory.record_lead(username, price_range=price_range, preferences=preferences, notes=notes)
    return {"status": "recorded"}


def tool_compare_cars(username: str, session: dict, listing_ids: list[int]) -> list[dict]:
    return _fetch_cars(listing_ids)


TOOL_DISPATCH = {
    "search_cars": tool_search_cars,
    "select_car": tool_select_car,
    "manage_favorite": tool_manage_favorite,
    "search_history": tool_search_history,
    "get_chat_history": tool_get_chat_history,
    "manage_booking": tool_manage_booking,
    "qualify_lead": tool_qualify_lead,
    "compare_cars": tool_compare_cars,
}


def call_tool(name: str, username: str, session: dict, arguments: dict):
    if name not in TOOL_DISPATCH:
        return {"error": f"unknown tool {name}"}
    try:
        return TOOL_DISPATCH[name](username, session, **arguments)
    except TypeError as e:
        # the model passed an argument name/shape that doesn't match the
        # tool's real signature -- tell it what went wrong instead of
        # crashing the whole turn, so it can retry with a corrected call
        return {"error": f"invalid arguments for {name}: {e}"}


# ---------------------------------------------------------------------------
# Chat summarization -- fires every SUMMARIZE_EVERY exchanges.
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT = """You summarize a short car-shopping chat and pull out the sales lead in it.
Respond with ONLY a JSON object shaped exactly like:
{"summary": "...", "price_range": "...", "preferences": "...", "notes": "..."}
- summary: 1-2 short sentences about the user's searches, preferences and selections.
- price_range: the budget the user actually stated, in their own words. "" if none was stated. Never guess.
- preferences: what they are looking for (make, model, body type, color, year, seats, specs). "" if nothing specific.
- notes: anything else relevant (financing, family size, urgency, test drive interest). "" if none.
You are also given the user's previous recorded lead. If this chat adds nothing new to it, return "" for price_range and preferences.
Use only what the user said; the assistant's messages are context, not the user's needs."""


def _lead_context(username: str, session: dict) -> str:
    previous = memory.get_last_lead(username)
    previous_text = (
        f"price_range={previous['price_range']!r}, preferences={previous['preferences']!r}, notes={previous['notes']!r}"
        if previous else "none yet"
    )
    filters = session.get("current_active_filters") or "none"
    return f"Previous recorded lead: {previous_text}\nActive search filters: {filters}"


def maybe_summarize(username: str, session: dict) -> None:
    """Every SUMMARIZE_EVERY exchanges: save a chat summary, and record a lead if the chat revealed a new need.
    One LLM call does both."""
    if not memory.ready_to_summarize(session):
        return
    unsummarized = session["pending_turns"][-session["turns_since_summary"]:]
    transcript = "\n".join(f"User: {t['user']}\nAssistant: {t['assistant']}" for t in unsummarized)
    try:
        response = litellm.completion(
            model=MODEL,
            api_key=GEMINI_API_KEY,
            messages=[
                {"role": "system", "content": SUMMARIZE_PROMPT},
                {"role": "user", "content": f"{_lead_context(username, session)}\n\nChat:\n{transcript}"},
            ],
            response_format={"type": "json_object"},
            num_retries=3,
        )
        data = json.loads(response.choices[0].message.content)
        summary = str(data["summary"]).strip()
    except Exception as exc:
        # turns stay pending and this retries after the next message
        print(f"[summarize skipped] {describe_llm_error(exc)}")
        return
    memory.log_chat_summary(username, summary)
    price_range = str(data.get("price_range") or "").strip()
    preferences = str(data.get("preferences") or "").strip()
    if (price_range or preferences) and memory.lead_is_new(username, price_range, preferences):
        memory.record_lead(username, price_range=price_range, preferences=preferences, notes=str(data.get("notes") or "").strip())
    session["turns_since_summary"] = 0  # live context in pending_turns is left alone
