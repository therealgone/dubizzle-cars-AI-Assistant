import csv
import json
import os
import sqlite3
from datetime import datetime, time, timezone

DB_PATH = "data/memory.db"
BOOKING_OPEN = time(8, 0)
BOOKING_CLOSE = time(20, 0)
SESSION_CACHE_PATH = "data/session_cache.json"
CAR_STACK_LIMIT = 6
RECENT_LOGS_LIMIT = 6
PENDING_TURNS_LIMIT = 6
LEADS_CSV_PATH = "data/leads.csv"
LEADS_CSV_FIELDS = ["timestamp", "username", "price_range", "preferences", "notes"]


def _normalize_username(username: str) -> str:
    return username.strip().lower()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profile (
            username TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS car_interaction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            listing_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            reason TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            listing_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_log_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            summary TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def user_exists(username: str) -> bool:
    username = _normalize_username(username)
    conn = _connect()
    row = conn.execute("SELECT 1 FROM user_profile WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row is not None


def get_or_create_user(username: str) -> dict:
    # identity only -- filters/selected_car are session-cache-only, never persisted here
    username = _normalize_username(username)
    conn = _connect()
    row = conn.execute("SELECT * FROM user_profile WHERE username = ?", (username,)).fetchone()
    if row is None:
        now = _now()
        conn.execute(
            "INSERT INTO user_profile (username, created_at, updated_at) VALUES (?, ?, ?)",
            (username, now, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM user_profile WHERE username = ?", (username,)).fetchone()
    conn.close()
    return {"username": row["username"]}


def log_interaction(username: str, listing_id: int, event_type: str, reason: str | None = None) -> None:
    username = _normalize_username(username)
    conn = _connect()
    conn.execute(
        "INSERT INTO car_interaction_log (username, listing_id, timestamp, event_type, reason) "
        "VALUES (?, ?, ?, ?, ?)",
        (username, listing_id, _now(), event_type, reason),
    )
    conn.commit()
    conn.close()


def get_favorites(username: str) -> list[int]:
    username = _normalize_username(username)
    conn = _connect()
    rows = conn.execute(
        "SELECT listing_id, event_type FROM car_interaction_log "
        "WHERE username = ? AND event_type IN ('favorited', 'unfavorited') "
        "ORDER BY timestamp ASC",
        (username,),
    ).fetchall()
    conn.close()
    # latest event per listing_id wins
    status: dict[int, str] = {}
    for row in rows:
        status[row["listing_id"]] = row["event_type"]
    return [listing_id for listing_id, event in status.items() if event == "favorited"]


def get_recent_interactions(username: str, limit: int = 6) -> list[dict]:
    username = _normalize_username(username)
    conn = _connect()
    rows = conn.execute(
        "SELECT listing_id, timestamp, event_type, reason FROM car_interaction_log "
        "WHERE username = ? ORDER BY timestamp DESC LIMIT ?",
        (username, limit),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def _distinct_listing_ids_by_event(username: str, event_type: str, limit: int) -> list[int]:
    username = _normalize_username(username)
    conn = _connect()
    rows = conn.execute(
        "SELECT listing_id FROM car_interaction_log "
        "WHERE username = ? AND event_type = ? ORDER BY timestamp DESC",
        (username, event_type),
    ).fetchall()
    conn.close()
    seen: list[int] = []
    for row in rows:
        if row["listing_id"] not in seen:  # same car can trigger this event more than once
            seen.append(row["listing_id"])
        if len(seen) == limit:
            break
    return seen


def get_recently_viewed(username: str, limit: int = 6) -> list[int]:
    """Cars that appeared in a search results list -- not necessarily picked."""
    return _distinct_listing_ids_by_event(username, "shown", limit)


def get_selected_history(username: str, limit: int = 20) -> list[int]:
    """Cars the user explicitly picked out of a list, most recent first."""
    return _distinct_listing_ids_by_event(username, "selected", limit)


def select_car(username: str, session: dict, car: dict) -> None:
    """The one action for 'user picked this car' -- updates the session's
    current selected_car (cache-only, never persisted to SQL) AND logs a
    permanent "selected" event in the same call, so switching cars never
    loses the previous one from the user's selection history."""
    update_selected_car(session, car)
    log_interaction(username, car["listing_id"], "selected")


def is_valid_booking_slot(date_str: str, time_str: str) -> bool:
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    if dt.weekday() == 6:  # Sunday closed
        return False
    return BOOKING_OPEN <= dt.time() <= BOOKING_CLOSE


def _weekday(date_str: str) -> str:
    # computed here, not left for the LLM to work out -- it gets this wrong
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")


def create_booking(username: str, listing_id: int, date_str: str, time_str: str) -> dict:
    if not is_valid_booking_slot(date_str, time_str):
        raise ValueError("bookings are only available Mon-Sat, 8am-8pm")
    username = _normalize_username(username)
    now = _now()
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO bookings (username, listing_id, date, time, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 'active', ?, ?)",
        (username, listing_id, date_str, time_str, now, now),
    )
    conn.commit()
    booking_id = cur.lastrowid
    conn.close()
    return {"id": booking_id, "username": username, "listing_id": listing_id, "date": date_str, "weekday": _weekday(date_str), "time": time_str, "status": "active"}


def reschedule_booking(
    booking_id: int,
    date_str: str | None = None,
    time_str: str | None = None,
    listing_id: int | None = None,
) -> dict:
    """Change the time, the car, or both -- whatever's given overrides,
    whatever's omitted keeps its current value."""
    conn = _connect()
    row = conn.execute("SELECT * FROM bookings WHERE id = ?", (booking_id,)).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"no booking with id {booking_id}")

    new_date = date_str or row["date"]
    new_time = time_str or row["time"]
    new_listing_id = listing_id if listing_id is not None else row["listing_id"]

    if not is_valid_booking_slot(new_date, new_time):
        conn.close()
        raise ValueError("bookings are only available Mon-Sat, 8am-8pm")

    conn.execute(
        "UPDATE bookings SET date = ?, time = ?, listing_id = ?, updated_at = ? WHERE id = ?",
        (new_date, new_time, new_listing_id, _now(), booking_id),
    )
    conn.commit()
    conn.close()
    return {"id": booking_id, "listing_id": new_listing_id, "date": new_date, "weekday": _weekday(new_date), "time": new_time, "status": "active"}


def cancel_booking(booking_id: int) -> None:
    conn = _connect()
    conn.execute(
        "UPDATE bookings SET status = 'cancelled', updated_at = ? WHERE id = ?",
        (_now(), booking_id),
    )
    conn.commit()
    conn.close()


def get_bookings(username: str, active_only: bool = True) -> list[dict]:
    username = _normalize_username(username)
    conn = _connect()
    query = "SELECT * FROM bookings WHERE username = ?"
    params = [username]
    if active_only:
        query += " AND status = 'active'"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    bookings = [dict(row) for row in rows]
    for b in bookings:
        b["weekday"] = _weekday(b["date"])
    return bookings


def init_leads_csv() -> None:
    """Business-side artifact -- exists permanently from server startup,
    not only once the LLM first happens to qualify a lead."""
    os.makedirs(os.path.dirname(LEADS_CSV_PATH), exist_ok=True)
    if not os.path.exists(LEADS_CSV_PATH):
        with open(LEADS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LEADS_CSV_FIELDS).writeheader()


def record_lead(username: str, price_range: str, preferences: str, notes: str = "") -> None:
    """Append-only, shared across all users -- one row per qualifying moment
    (e.g. budget + at least one preference known), not a single row that
    gets overwritten. Meant to read like a real lead-tracking log: the same
    user can appear multiple times as their stated preferences evolve."""
    username = _normalize_username(username)
    init_leads_csv()
    with open(LEADS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEADS_CSV_FIELDS)
        writer.writerow({
            "timestamp": _now(),
            "username": username,
            "price_range": price_range,
            "preferences": preferences,
            "notes": notes,
        })


def log_chat_summary(username: str, summary: str) -> None:
    """Persistent, cross-session chat memory -- not the short-term session
    cache's recent_logs (capped at 6, reset-able). This is meant to answer
    "what did we talk about yesterday," so it's never trimmed here."""
    username = _normalize_username(username)
    conn = _connect()
    conn.execute(
        "INSERT INTO chat_log_history (username, timestamp, summary) VALUES (?, ?, ?)",
        (username, _now(), summary),
    )
    conn.commit()
    conn.close()


def get_chat_history(username: str, limit: int = 20, on_date: str | None = None) -> list[dict]:
    """on_date: 'YYYY-MM-DD' to answer "what did we view yesterday" style questions."""
    username = _normalize_username(username)
    conn = _connect()
    query = "SELECT timestamp, summary FROM chat_log_history WHERE username = ?"
    params: list = [username]
    if on_date:
        query += " AND substr(timestamp, 1, 10) = ?"
        params.append(on_date)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def search_history(
    username: str,
    keyword: str | None = None,
    event_types: list[str] | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search the user's OWN past cars (selected/favorited by default) by
    make/model/title keyword -- e.g. "the Mercedes-Benz I viewed before"."""
    from backend.car_search import _collection  # local import: avoid a hard dependency at module load

    username = _normalize_username(username)
    event_types = event_types or ["selected", "favorited"]
    placeholders = ",".join("?" for _ in event_types)
    conn = _connect()
    rows = conn.execute(
        f"SELECT listing_id, MAX(timestamp) as last_seen FROM car_interaction_log "
        f"WHERE username = ? AND event_type IN ({placeholders}) "
        f"GROUP BY listing_id ORDER BY last_seen DESC",
        (username, *event_types),
    ).fetchall()
    conn.close()

    listing_ids = [str(row["listing_id"]) for row in rows]
    if not listing_ids:
        return []

    fetched = _collection.get(ids=listing_ids, include=["metadatas"])
    by_id = dict(zip(fetched["ids"], fetched["metadatas"]))
    results = [by_id[i] for i in listing_ids if i in by_id]

    if keyword:
        kw = keyword.lower()
        results = [
            r for r in results
            if kw in r["make"].lower() or kw in r["model"].lower() or kw in r["title"].lower()
        ]
    return results[:limit]


def _load_session_cache() -> dict:
    if not os.path.exists(SESSION_CACHE_PATH):
        return {}
    with open(SESSION_CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_session_cache(data: dict) -> None:
    os.makedirs(os.path.dirname(SESSION_CACHE_PATH), exist_ok=True)
    with open(SESSION_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_session(username: str) -> dict:
    username = _normalize_username(username)
    cache = _load_session_cache()
    return cache.get(
        username,
        {
            "current_active_filters": {},
            "recent_logs": [],
            "car_stack": [],
            "selected_car": None,
            "pending_turns": [],
        },
    )


def save_session(username: str, session: dict) -> None:
    username = _normalize_username(username)
    cache = _load_session_cache()
    cache[username] = session
    _save_session_cache(cache)


def clear_session(username: str) -> None:
    """Drops short-term state (filters, selected car, car stack, pending
    turns) so the user starts a genuinely new session -- long-term SQLite
    history/favorites/bookings are untouched, so the agent still recalls
    them via tools."""
    username = _normalize_username(username)
    cache = _load_session_cache()
    cache.pop(username, None)
    _save_session_cache(cache)


def push_car_stack(session: dict, cars: list[dict]) -> dict:
    session["car_stack"] = (session["car_stack"] + cars)[-CAR_STACK_LIMIT:]
    return session


def push_recent_log(session: dict, summary: str) -> dict:
    session["recent_logs"] = (session["recent_logs"] + [summary])[-RECENT_LOGS_LIMIT:]
    return session


def add_pending_turn(session: dict, user_message: str, assistant_response: str) -> dict:
    """Raw, unsummarized turns waiting to be batch-summarized. Not capped by
    FIFO like car_stack/recent_logs -- ready_to_summarize() clears it out
    entirely once full, it doesn't quietly drop the oldest."""
    session["pending_turns"].append({"user": user_message, "assistant": assistant_response})
    return session


def ready_to_summarize(session: dict) -> bool:
    return len(session["pending_turns"]) >= PENDING_TURNS_LIMIT


def update_active_filters(session: dict, new_fields: dict) -> dict:
    """Session-cache-only -- merges into whatever's already active this
    session, never persisted to SQL. Every new session starts with an empty
    filter regardless of what was active last time."""
    merged = {**session["current_active_filters"], **new_fields}
    session["current_active_filters"] = merged
    return merged


def update_selected_car(session: dict, car: dict | None) -> None:
    """Session-cache-only -- see update_active_filters."""
    session["selected_car"] = car


def get_full_session_cache(username: str) -> dict:
    """Dev-only: this signed-in user's own short-term cache entry."""
    username = _normalize_username(username)
    return _load_session_cache().get(username, {})


def get_all_long_term_data(username: str) -> dict:
    """Dev-only: this signed-in user's own rows in every long-term SQLite table."""
    username = _normalize_username(username)
    conn = _connect()
    tables = ["user_profile", "car_interaction_log", "bookings", "chat_log_history"]
    data = {
        table: [dict(row) for row in conn.execute(f"SELECT * FROM {table} WHERE username = ?", (username,)).fetchall()]
        for table in tables
    }
    conn.close()
    return data
