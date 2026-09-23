import json
import os
import sqlite3
from datetime import datetime, time, timezone

DB_PATH = "data/memory.db"
BOOKING_OPEN = time(8, 0)
BOOKING_CLOSE = time(21, 0)
SESSION_CACHE_PATH = "data/session_cache.json"
CAR_STACK_LIMIT = 6
RECENT_LOGS_LIMIT = 6


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
            filters TEXT NOT NULL DEFAULT '{}',
            selected_car TEXT,
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
    username = _normalize_username(username)
    conn = _connect()
    row = conn.execute("SELECT * FROM user_profile WHERE username = ?", (username,)).fetchone()
    if row is None:
        now = _now()
        conn.execute(
            "INSERT INTO user_profile (username, filters, selected_car, created_at, updated_at) "
            "VALUES (?, '{}', NULL, ?, ?)",
            (username, now, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM user_profile WHERE username = ?", (username,)).fetchone()
    conn.close()
    return {
        "username": row["username"],
        "filters": json.loads(row["filters"]),
        "selected_car": json.loads(row["selected_car"]) if row["selected_car"] else None,
    }


def update_filters(username: str, new_fields: dict) -> dict:
    username = _normalize_username(username)
    user = get_or_create_user(username)
    merged = {**user["filters"], **new_fields}
    conn = _connect()
    conn.execute(
        "UPDATE user_profile SET filters = ?, updated_at = ? WHERE username = ?",
        (json.dumps(merged), _now(), username),
    )
    conn.commit()
    conn.close()
    return merged


def set_selected_car(username: str, car: dict | None) -> None:
    username = _normalize_username(username)
    get_or_create_user(username)  # ensure row exists
    conn = _connect()
    conn.execute(
        "UPDATE user_profile SET selected_car = ?, updated_at = ? WHERE username = ?",
        (json.dumps(car) if car else None, _now(), username),
    )
    conn.commit()
    conn.close()


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


def get_recently_viewed(username: str, limit: int = 6) -> list[int]:
    username = _normalize_username(username)
    conn = _connect()
    rows = conn.execute(
        "SELECT listing_id FROM car_interaction_log "
        "WHERE username = ? AND event_type = 'shown' ORDER BY timestamp DESC",
        (username,),
    ).fetchall()
    conn.close()
    seen: list[int] = []
    for row in rows:
        if row["listing_id"] not in seen:  # same car can be shown more than once
            seen.append(row["listing_id"])
        if len(seen) == limit:
            break
    return seen


def is_valid_booking_slot(date_str: str, time_str: str) -> bool:
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    if dt.weekday() == 6:  # Sunday closed
        return False
    return BOOKING_OPEN <= dt.time() <= BOOKING_CLOSE


def create_booking(username: str, listing_id: int, date_str: str, time_str: str) -> dict:
    if not is_valid_booking_slot(date_str, time_str):
        raise ValueError("bookings are only available Mon-Sat, 8am-9pm")
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
    return {"id": booking_id, "username": username, "listing_id": listing_id, "date": date_str, "time": time_str, "status": "active"}


def reschedule_booking(booking_id: int, date_str: str, time_str: str) -> None:
    if not is_valid_booking_slot(date_str, time_str):
        raise ValueError("bookings are only available Mon-Sat, 8am-9pm")
    conn = _connect()
    conn.execute(
        "UPDATE bookings SET date = ?, time = ?, updated_at = ? WHERE id = ?",
        (date_str, time_str, _now(), booking_id),
    )
    conn.commit()
    conn.close()


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
    return [dict(row) for row in rows]


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
        {"current_active_filters": {}, "recent_logs": [], "car_stack": [], "selected_car": None},
    )


def save_session(username: str, session: dict) -> None:
    username = _normalize_username(username)
    cache = _load_session_cache()
    cache[username] = session
    _save_session_cache(cache)


def push_car_stack(session: dict, cars: list[dict]) -> dict:
    session["car_stack"] = (session["car_stack"] + cars)[-CAR_STACK_LIMIT:]
    return session


def push_recent_log(session: dict, summary: str) -> dict:
    session["recent_logs"] = (session["recent_logs"] + [summary])[-RECENT_LOGS_LIMIT:]
    return session


def update_active_filters(username: str, session: dict, new_fields: dict) -> dict:
    merged = update_filters(username, new_fields)
    session["current_active_filters"] = merged
    return merged


def update_selected_car(username: str, session: dict, car: dict | None) -> None:
    set_selected_car(username, car)
    session["selected_car"] = car
