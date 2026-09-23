import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.memory as memory
import backend.tools as tools

BORDER = "=" * 70
RULE = "-" * 70
USERNAME = "benchuser"

results = []


def check(description, condition, detail=""):
    results.append((description, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"[{status}] {description}{suffix}")


print(BORDER)
print("TOOL BENCHMARK SUITE")
print(BORDER)

memory.init_db()
session = memory.get_session(USERNAME)

print(f"\n{RULE}\nTOOL: search_cars\n{RULE}")

# hard case: no dedicated metadata field exists for "GCC spec" at all -- must
# resolve entirely through the description fallback. Default top_k=5 applies
# when not overridden -- this is the tool's real default, not a bug.
r1 = tools.call_tool("search_cars", USERNAME, session, {"filters": {"description": "gcc"}})
check("default top_k=5 applies when not overridden", len(r1) == 5, f"got {len(r1)}")
check("side effect: current_active_filters updated from this call",
      session["current_active_filters"] == {"description": "gcc"}, str(session["current_active_filters"]))
check("side effect: car_stack capped at 6 (only 5 pushed, under the cap)",
      len(session["car_stack"]) == 5, f"got {len(session['car_stack'])}")
check("side effect: 'shown' events logged to history", len(memory.get_recently_viewed(USERNAME, limit=50)) > 0)

# hard case: LLM explicitly asks for "all" GCC spec cars -- top_k override
# should surface all 45 real matches, not stay capped at the default
r1b = tools.call_tool("search_cars", USERNAME, session, {"filters": {"description": "gcc"}, "top_k": 45})
check("top_k override returns all 45 real GCC matches when explicitly asked for 'all'",
      len(r1b) == 45, f"got {len(r1b)}")

# hard case: filter merge across two separate tool calls, not just one function call
session["current_active_filters"] = {}
tools.call_tool("search_cars", USERNAME, session, {"filters": {"make": "mercedes-benz", "year": [2016, 2017, 2018, 2019]}})
tools.call_tool("search_cars", USERNAME, session, {"filters": {"make": "ford"}})
check("filter merge persists year across two separate tool calls",
      session["current_active_filters"].get("year") == [2016, 2017, 2018, 2019],
      str(session["current_active_filters"]))

print(f"\n{RULE}\nTOOL: select_car\n{RULE}")

tools.call_tool("select_car", USERNAME, session, {"listing_id": 2})
check("selects full metadata, not just an id",
      session["selected_car"]["make"] == "mercedes-benz" and session["selected_car"]["model"] == "c-class")

r_bad = tools.call_tool("select_car", USERNAME, session, {"listing_id": 999999})
check("nonexistent listing_id fails gracefully, no crash", "error" in r_bad, str(r_bad))

tools.call_tool("select_car", USERNAME, session, {"listing_id": 53})
check("new selection replaces current selected_car", session["selected_car"]["model"] == "explorer")
hist = memory.get_selected_history(USERNAME)
check("old selection (id 2) survives in history after switching to id 53",
      2 in hist and 53 in hist, str(hist))

print(f"\n{RULE}\nTOOL: manage_favorite\n{RULE}")

tools.call_tool("manage_favorite", USERNAME, session, {"listing_id": 38, "action": "add"})
tools.call_tool("manage_favorite", USERNAME, session, {"listing_id": 100, "action": "add"})
tools.call_tool("manage_favorite", USERNAME, session, {"listing_id": 38, "action": "remove"})
favs = memory.get_favorites(USERNAME)
check("add-then-remove resolves correctly (only 100 stays favorited)", favs == [100], str(favs))

# hard case: "show me all my favorites" -- must be full car details, not bare ids,
# and must not include the one that was added then removed
tools.call_tool("manage_favorite", USERNAME, session, {"listing_id": 17, "action": "add"})
listed = tools.call_tool("manage_favorite", USERNAME, session, {"action": "list"})
check("list action returns full car details, not just ids",
      all("make" in car for car in listed), str(listed))
check("list action excludes the favorite that was later removed (38)",
      {c["listing_id"] for c in listed} == {100, 17}, str([c["listing_id"] for c in listed]))

print(f"\n{RULE}\nTOOL: search_history\n{RULE}")

tools.call_tool("select_car", USERNAME, session, {"listing_id": 4})  # 2nd distinct mercedes-benz
matches = tools.call_tool("search_history", USERNAME, session, {"keyword": "mercedes"})
check("finds multiple past mercedes-benz selections for disambiguation",
      len(matches) >= 2, str([m["listing_id"] for m in matches]))

no_matches = tools.call_tool("search_history", USERNAME, session, {"keyword": "lamborghini"})
check("no false positives for a make never selected/favorited", no_matches == [], str(no_matches))

everything = tools.call_tool("search_history", USERNAME, session, {})
check("no keyword returns everything (browse-all mode)", len(everything) >= 3, str(len(everything)))

print(f"\n{RULE}\nTOOL: get_chat_history\n{RULE}")

# backdate two entries to "yesterday" directly, to test real date filtering
yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
conn = memory._connect()
conn.execute(
    "INSERT INTO chat_log_history (username, timestamp, summary) VALUES (?, ?, ?)",
    (USERNAME, f"{yesterday}T10:00:00+00:00", "searched for suvs yesterday"),
)
conn.execute(
    "INSERT INTO chat_log_history (username, timestamp, summary) VALUES (?, ?, ?)",
    (USERNAME, f"{yesterday}T15:00:00+00:00", "selected a range rover yesterday"),
)
conn.commit()
conn.close()
memory.log_chat_summary(USERNAME, "searched for mercedes-benz and ford explorer today")

yesterday_only = tools.call_tool("get_chat_history", USERNAME, session, {"on_date": yesterday})
check("on_date filters to exactly yesterday's 2 entries", len(yesterday_only) == 2, str(len(yesterday_only)))

today_only = tools.call_tool("get_chat_history", USERNAME, session, {"on_date": today})
check("on_date filters to exactly today's 1 entry", len(today_only) == 1, str(len(today_only)))

all_history = tools.call_tool("get_chat_history", USERNAME, session, {})
check("no date returns all 3 entries", len(all_history) == 3, str(len(all_history)))

print(f"\n{RULE}\nTOOL: manage_booking\n{RULE}")

b1 = tools.call_tool("manage_booking", USERNAME, session,
                      {"action": "create", "listing_id": 38, "date": "2026-09-28", "time": "14:00"})  # Monday
check("valid Mon-Sat 8am-9pm slot creates a booking", "id" in b1, str(b1))

b2 = tools.call_tool("manage_booking", USERNAME, session,
                      {"action": "create", "listing_id": 38, "date": "2026-09-27", "time": "14:00"})  # Sunday
check("Sunday slot rejected gracefully, no crash", "error" in b2, str(b2))

b3 = tools.call_tool("manage_booking", USERNAME, session,
                      {"action": "create", "listing_id": 38, "date": "2026-09-28", "time": "22:00"})  # after hours
check("after-hours slot rejected gracefully, no crash", "error" in b3, str(b3))

booking_id = b1.get("id")
tools.call_tool("manage_booking", USERNAME, session,
                 {"action": "reschedule", "booking_id": booking_id, "date": "2026-09-29", "time": "10:00"})
active = tools.call_tool("manage_booking", USERNAME, session, {"action": "list"})
check("reschedule applied correctly",
      active[0]["date"] == "2026-09-29" and active[0]["time"] == "10:00", str(active))

tools.call_tool("manage_booking", USERNAME, session, {"action": "cancel", "booking_id": booking_id})
active_after_cancel = tools.call_tool("manage_booking", USERNAME, session, {"action": "list"})
check("cancel removes it from the active list", active_after_cancel == [], str(active_after_cancel))

# hard case: change WHICH CAR a booking is for, keeping the same date/time --
# this was a real gap (reschedule only ever touched date/time before)
b4 = tools.call_tool("manage_booking", USERNAME, session,
                      {"action": "create", "listing_id": 38, "date": "2026-09-28", "time": "16:00"})
tools.call_tool("manage_booking", USERNAME, session,
                 {"action": "reschedule", "booking_id": b4["id"], "listing_id": 100})  # no date/time given
after_car_swap = tools.call_tool("manage_booking", USERNAME, session, {"action": "list"})
swapped = next(b for b in after_car_swap if b["id"] == b4["id"])
check("reschedule can change just the car, keeping the original time",
      swapped["listing_id"] == 100 and swapped["time"] == "16:00", str(swapped))
tools.call_tool("manage_booking", USERNAME, session, {"action": "cancel", "booking_id": b4["id"]})

print(f"\n{RULE}\nTOOL: search_history limit override\n{RULE}")

# hard case: user has more distinct selected cars than the default limit (10) --
# "show me every car I've ever selected" must not silently truncate
fresh_ids = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # 12 distinct, none touched earlier in this run
for lid in fresh_ids:
    car = tools._fetch_car(lid)
    memory.select_car(USERNAME, session, car)

default_limited = tools.call_tool("search_history", USERNAME, session, {"event_types": ["selected"]})
check("default limit (10) caps results even though more exist", len(default_limited) == 10, f"got {len(default_limited)}")

everything_selected = tools.call_tool("search_history", USERNAME, session, {"event_types": ["selected"], "limit": 100})
check("limit override surfaces every selected car, not just the default 10",
      len(everything_selected) >= len(fresh_ids), f"got {len(everything_selected)}")

print(f"\n{RULE}\nTOOL: qualify_lead\n{RULE}")

if os.path.exists(memory.LEADS_CSV_PATH):
    os.remove(memory.LEADS_CSV_PATH)  # clean slate so row counts below are exact

tools.call_tool("qualify_lead", USERNAME, session,
                {"price_range": "around 80000 AED", "preferences": "SUV, mercedes-benz preferred"})
# same user, evolved preferences later in the conversation -- must APPEND, not overwrite
tools.call_tool("qualify_lead", USERNAME, session,
                {"price_range": "around 100000 AED", "preferences": "SUV, now open to ford explorer too", "notes": "family of 5, needs 3rd row"})
# a second, different user -- shared file, not per-user
tools.call_tool("qualify_lead", "otheruser", session,
                {"price_range": "150000 AED", "preferences": "sports car, ferrari or lamborghini"})

with open(memory.LEADS_CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

check("leads.csv has exactly one header + 3 data rows (append-only, not overwritten)", len(rows) == 3, f"got {len(rows)} rows")
check("same user appears twice with evolved preferences, not one overwritten row",
      sum(1 for r in rows if r["username"] == USERNAME) == 2,
      str([r["preferences"] for r in rows if r["username"] == USERNAME]))
check("leads.csv is shared across users, not per-user",
      any(r["username"] == "otheruser" for r in rows), str([r["username"] for r in rows]))
check("notes field carries through correctly",
      any("3rd row" in r["notes"] for r in rows), str([r["notes"] for r in rows]))

print(f"\n{RULE}\nBATCH CHAT SUMMARIZATION (1 real LLM call)\n{RULE}")

fake_turns = [
    ("I'm looking for a Mercedes-Benz from 2016 to 2019", "I found several Mercedes-Benz listings in that range."),
    ("Show me the first one", "Here are the details for the 2019 C-Class C300 Luxury."),
    ("What's the price?", "That listing doesn't have a price mentioned -- best to contact the seller."),
    ("Actually show me a Ford Explorer instead", "Here are some Ford Explorer listings."),
    ("I'll take that one, add it to favorites", "Added to your favorites."),
    ("Can I book a test drive for Monday at 2pm?", "Your test drive is booked for Monday at 2:00 PM."),
]
for user_msg, assistant_msg in fake_turns:
    memory.add_pending_turn(session, user_msg, assistant_msg)
check("pending_turns reaches the trigger threshold (6)", memory.ready_to_summarize(session), str(len(session["pending_turns"])))

tools.maybe_summarize(USERNAME, session)
check("pending_turns cleared after summarization", session["pending_turns"] == [])
latest_summary = memory.get_chat_history(USERNAME, limit=1)
check("summarization produced and stored a real summary",
      bool(latest_summary) and len(latest_summary[0]["summary"]) > 0, str(latest_summary))
if latest_summary:
    print("actual summary text:", latest_summary[0]["summary"])

memory.save_session(USERNAME, session)

print(f"\n{BORDER}")
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"RESULT: {passed}/{total} checks passed")
if passed != total:
    print("FAILED CHECKS:")
    for desc, ok, detail in results:
        if not ok:
            print(f"  - {desc}: {detail}")
print(BORDER)
