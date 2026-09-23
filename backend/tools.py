import litellm

import backend.memory as memory
from backend.car_search import _collection, search_cars
from backend.config import GEMINI_API_KEY
from backend.enrichment import BODY_TYPES
from backend.llm_client import MODEL

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
            "description": "Add or remove a listing from the user's favorites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_id": {"type": "integer"},
                    "action": {"type": "string", "enum": ["add", "remove"]},
                },
                "required": ["listing_id", "action"],
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
            "description": "Create, reschedule, cancel, or list test-drive bookings. Bookings are only available Mon-Sat, 8am-9pm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "reschedule", "cancel", "list"]},
                    "listing_id": {"type": "integer", "description": "required for create"},
                    "booking_id": {"type": "integer", "description": "required for reschedule/cancel"},
                    "date": {"type": "string", "description": "YYYY-MM-DD, required for create/reschedule"},
                    "time": {"type": "string", "description": "HH:MM 24h, required for create/reschedule"},
                },
                "required": ["action"],
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


def tool_manage_favorite(username: str, session: dict, listing_id: int, action: str) -> dict:
    event = "favorited" if action == "add" else "unfavorited"
    memory.log_interaction(username, listing_id, event)
    return {"listing_id": listing_id, "favorited": action == "add"}


def tool_search_history(username: str, session: dict, keyword: str | None = None,
                         event_types: list[str] | None = None) -> list[dict]:
    return memory.search_history(username, keyword=keyword, event_types=event_types)


def tool_get_chat_history(username: str, session: dict, on_date: str | None = None, limit: int = 20) -> list[dict]:
    return memory.get_chat_history(username, limit=limit, on_date=on_date)


def tool_manage_booking(username: str, session: dict, action: str, listing_id: int | None = None,
                         booking_id: int | None = None, date: str | None = None, time: str | None = None) -> dict | list[dict]:
    try:
        if action == "create":
            return memory.create_booking(username, listing_id, date, time)
        if action == "reschedule":
            memory.reschedule_booking(booking_id, date, time)
            return {"booking_id": booking_id, "status": "rescheduled"}
        if action == "cancel":
            memory.cancel_booking(booking_id)
            return {"booking_id": booking_id, "status": "cancelled"}
        if action == "list":
            return memory.get_bookings(username)
        return {"error": f"unknown action {action}"}
    except ValueError as e:
        return {"error": str(e)}


TOOL_DISPATCH = {
    "search_cars": tool_search_cars,
    "select_car": tool_select_car,
    "manage_favorite": tool_manage_favorite,
    "search_history": tool_search_history,
    "get_chat_history": tool_get_chat_history,
    "manage_booking": tool_manage_booking,
}


def call_tool(name: str, username: str, session: dict, arguments: dict):
    if name not in TOOL_DISPATCH:
        return {"error": f"unknown tool {name}"}
    return TOOL_DISPATCH[name](username, session, **arguments)


# ---------------------------------------------------------------------------
# Batch chat summarization -- fires once pending_turns hits the cap.
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT = (
    "Summarize this chat exchange in 1-2 short sentences, focused on car "
    "preferences, searches, and selections. Plain text, no preamble."
)


def maybe_summarize(username: str, session: dict) -> None:
    if not memory.ready_to_summarize(session):
        return
    transcript = "\n".join(
        f"User: {t['user']}\nAssistant: {t['assistant']}" for t in session["pending_turns"]
    )
    response = litellm.completion(
        model=MODEL,
        api_key=GEMINI_API_KEY,
        messages=[
            {"role": "system", "content": SUMMARIZE_PROMPT},
            {"role": "user", "content": transcript},
        ],
        num_retries=3,
    )
    summary = response.choices[0].message.content.strip()
    memory.log_chat_summary(username, summary)
    session["pending_turns"] = []
