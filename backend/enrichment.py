import json
import re

import litellm

from backend.config import GEMINI_API_KEY
from backend.llm_client import MODEL

# deterministic, no LLM -- distinguishes real cash price from monthly
# installment figures and unrelated AED mentions (salary, fees)
_MONEY_RE = re.compile(
    r"(?:AED|Dhs?)\s*[:\-]?\s*([\d,]{3,10}(?:\.\d+)?)"
    r"|([\d,]{3,10}(?:\.\d+)?)\s*/?\s*(?:AED|Dhs?)",
    re.I,
)
_EXCLUDE_RE = re.compile(r"month|/\s*mo\b|\bmo\b|salary|evaluation|registration|\brta\b|insurance", re.I)
_EXCLUDE_WINDOW = 18

BODY_TYPES = [
    "suv", "coupe", "sedan", "crossover", "hard_top_convertible", "pickup_truck",
    "hatchback", "soft_top_convertible", "sports_car", "van", "wagon",
    "utility_truck", "other",
]

CLASSIFY_SYSTEM_PROMPT = f"""You classify used car listings for a marketplace.
For each listing given, determine:
- body_type: exactly one of {BODY_TYPES}. Use your knowledge of the make/model
  (e.g. a Ferrari SF90 is "sports_car", a Range Rover is "suv") even if the
  listing text doesn't state it directly. Only use "other" if nothing fits.
- color: the car's exterior color, ONLY if the listing text states it.
  Otherwise return the exact string "not mentioned". Never guess a color.

Respond with ONLY a JSON object shaped exactly like:
{{"results": [{{"listing_id": <id>, "body_type": "<value>", "color": "<value>"}}]}}
One entry per listing given, same order, nothing else in the response."""

# 1 request for all 100 listings -- quota is the binding constraint, not payload size
CHUNK_SIZE = 100


def extract_price(text: str) -> float | None:
    candidates = []
    for m in _MONEY_RE.finditer(text):
        raw = m.group(1) or m.group(2)
        start = max(0, m.start() - _EXCLUDE_WINDOW)
        end = min(len(text), m.end() + _EXCLUDE_WINDOW)
        if _EXCLUDE_RE.search(text[start:end]):
            continue
        candidates.append(float(raw.replace(",", "")))
    return max(candidates) if candidates else None


def _classify_chunk(rows: list[dict]) -> dict[int, dict]:
    listing_lines = "\n".join(
        f"id={r['listing_id']}: {r['make']} {r['model']} {r['trim']} | "
        f"{r['title']} | {r['description'][:500]}"
        for r in rows
    )
    response = litellm.completion(
        model=MODEL,
        api_key=GEMINI_API_KEY,
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": listing_lines},
        ],
        response_format={"type": "json_object"},
        num_retries=3,
    )
    parsed = json.loads(response.choices[0].message.content)

    result = {}
    for entry in parsed["results"]:
        body_type = entry.get("body_type")
        if body_type not in BODY_TYPES:
            body_type = "other"
        result[int(entry["listing_id"])] = {
            "body_type": body_type,
            # lowercase -- exact-match filtering is case-sensitive, and the LLM
            # otherwise echoes whatever casing the ad text happened to use
            "color": (entry.get("color") or "not mentioned").lower(),
        }
    return result


def classify_batch(rows: list[dict]) -> dict[int, dict]:
    results = {}
    for i in range(0, len(rows), CHUNK_SIZE):
        results.update(_classify_chunk(rows[i : i + CHUNK_SIZE]))
    return results
