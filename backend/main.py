import json
import threading
from collections import defaultdict

import litellm
from fastapi import FastAPI
from pydantic import BaseModel

import backend.memory as memory
import backend.tools as tools_module
from backend.config import GEMINI_API_KEY
from backend.llm_client import MODEL, describe_llm_error
from backend.prompts import build_system_prompt

app = FastAPI()
memory.init_db()
memory.init_leads_csv()

# one lock per user: /chat holds the session file's read-modify-write for the whole (slow) LLM turn,
# so a select/favorite click for the same user must wait instead of being overwritten by the final save
_user_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


def _user_lock(username: str) -> threading.Lock:
    return _user_locks[username.strip().lower()]


def _log(message: str) -> None:
    # logging must never break a request: a Windows console can't encode e.g. Arabic listing text
    print(message.encode("ascii", "replace").decode("ascii"))


MAX_TOOL_ITERATIONS = 5
FALLBACK_REPLY = "Sorry, I'm having trouble with that -- could you try rephrasing?"


class ChatRequest(BaseModel):
    username: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    cars: list[dict] = []
    selected_car: dict | None = None


class NewSessionRequest(BaseModel):
    username: str


class SelectCarRequest(BaseModel):
    username: str
    listing_id: int


class ManageFavoriteRequest(BaseModel):
    username: str
    listing_id: int
    action: str  # "add" or "remove"


def _build_messages(username: str, session: dict, user_message: str) -> list[dict]:
    past_summaries = [row["summary"] for row in memory.get_chat_history(username, limit=3)]
    messages = [{"role": "system", "content": build_system_prompt(session, username, past_summaries)}]
    for turn in session["pending_turns"]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def run_chat_turn(username: str, session: dict, user_message: str) -> tuple[str, list[dict]]:
    messages = _build_messages(username, session, user_message)
    cars_this_turn: list[dict] = []
    final_reply = FALLBACK_REPLY

    llm_failed = False
    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            response = litellm.completion(
                model=MODEL,
                api_key=GEMINI_API_KEY,
                messages=messages,
                tools=tools_module.TOOL_SCHEMAS,
                tool_choice="auto",
                num_retries=3,
            )
        except Exception as exc:
            _log(f"[llm error] {type(exc).__name__}: {exc}")
            final_reply = f"Sorry, I can't answer right now: {describe_llm_error(exc)}."
            llm_failed = True
            break
        msg = response.choices[0].message

        if not msg.tool_calls:
            final_reply = msg.content or FALLBACK_REPLY
            break

        messages.append(msg.model_dump())
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = None
            if args is None:
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps({"error": "tool arguments were not valid JSON"})})
                continue
            result = tools_module.call_tool(tc.function.name, username, session, args)
            _log(f"[tool] {username}: {tc.function.name}({args}) -> {str(result)[:160]}")
            # "make" is unique to car dicts -- booking dicts also have
            # listing_id, which used to cause bookings to render as broken,
            # empty car cards ("Untitled listing")
            if isinstance(result, list) and result and "make" in result[0]:
                cars_this_turn = result
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    # a failed turn isn't recorded, so the error message never becomes chat context
    if not llm_failed:
        memory.add_pending_turn(session, user_message, final_reply)
        tools_module.maybe_summarize(username, session)
    return final_reply, cars_this_turn


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    memory.get_or_create_user(request.username)
    with _user_lock(request.username):
        session = memory.get_session(request.username)
        reply, cars = run_chat_turn(request.username, session, request.message)
        memory.save_session(request.username, session)
    # always return the true current selection, whether it changed via a
    # tool call this turn or was already set from an earlier click/message --
    # the frontend syncs to this every time so it can never show stale data
    return ChatResponse(reply=reply, cars=cars, selected_car=session["selected_car"])


@app.post("/new_session")
def new_session_endpoint(request: NewSessionRequest) -> dict:
    # simulates the user leaving and coming back later -- short-term state
    # resets, long-term SQLite history/favorites/bookings survive
    memory.get_or_create_user(request.username)
    with _user_lock(request.username):
        memory.clear_session(request.username)
    return {"status": "reset"}


@app.post("/select_car")
def select_car_endpoint(request: SelectCarRequest) -> dict:
    # deterministic UI action (a click) -- bypasses the LLM entirely, nothing to interpret
    memory.get_or_create_user(request.username)
    with _user_lock(request.username):
        session = memory.get_session(request.username)
        result = tools_module.tool_select_car(request.username, session, request.listing_id)
        memory.save_session(request.username, session)
    return result


@app.post("/manage_favorite")
def manage_favorite_endpoint(request: ManageFavoriteRequest) -> dict:
    memory.get_or_create_user(request.username)
    with _user_lock(request.username):
        session = memory.get_session(request.username)
        result = tools_module.tool_manage_favorite(request.username, session, action=request.action, listing_id=request.listing_id)
        memory.save_session(request.username, session)
    return result


@app.get("/dev/session_cache")
def dev_session_cache(username: str) -> dict:
    # dev-only inspection panel -- this signed-in user's own cache, not everyone's
    return memory.get_full_session_cache(username)


@app.get("/dev/long_term_memory")
def dev_long_term_memory(username: str) -> dict:
    # dev-only inspection panel -- this signed-in user's own SQLite rows only
    return memory.get_all_long_term_data(username)
