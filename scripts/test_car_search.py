import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.car_search import _bm25_search, _metadata_filter, _semantic_search, search_cars

BORDER = "=" * 70
RULE = "-" * 70


def report_leg(label, returned, truth):
    returned = list(returned)
    truth = set(truth)
    correct = [i for i in returned if i in truth]
    incorrect = [i for i in returned if i not in truth]
    missed = sorted(truth - set(returned), key=int)
    print(f"{RULE}\n{label}\n{RULE}")
    print("ranked result:", returned)
    print(f"correct: {len(correct)}/{len(truth)} -> {correct}")
    print(f"incorrect (returned but not a real match): {len(incorrect)} -> {incorrect}")
    print(f"missed (real match, not returned): {len(missed)} -> {missed}")
    print()


def run_test(num, description, filters, keywords, semantic_query, truth_ids):
    truth_ids = [str(i) for i in truth_ids]
    print(f"\n{BORDER}\nTEST {num}: {description}\n{BORDER}")
    print(f"expected (ground truth) ids: {len(truth_ids)} -> {sorted(truth_ids, key=int)}\n")

    report_leg(f"FILTER  filters={filters}", _metadata_filter(filters or {}), truth_ids)
    report_leg(f"KEYWORD  keywords='{keywords}'", _bm25_search(keywords or ""), truth_ids)
    report_leg(f"SEMANTIC  semantic_query='{semantic_query}'", _semantic_search(semantic_query or ""), truth_ids)

    fused = search_cars(filters=filters, keywords=keywords, semantic_query=semantic_query, top_k=20)
    fused_ids = [m["listing_id"] for m in fused]
    fused_ids = [str(i) for i in fused_ids]
    report_leg("FINAL FUSED RANKING  search_cars(...)", fused_ids, truth_ids)


# TEST 1: plain make lookup, should be trivial for filter, easy for keyword/semantic
run_test(
    1,
    "\"I want a Land Rover\"",
    filters={"make": "land rover"},
    keywords="land rover",
    semantic_query="I want a Land Rover",
    truth_ids=[3, 44, 93],
)

# TEST 2: no dedicated "spec" field exists -- LLM falls back to filters={"description": ...},
# which routes through Chroma's substring where_document match, not exact equality.
# 45/100 listings mention GCC, so this also stresses volume, not just presence/absence.
run_test(
    2,
    "\"I'm looking for a GCC spec car\"",
    filters={"description": "gcc"},
    keywords="gcc spec",
    semantic_query="I'm looking for a GCC spec car",
    truth_ids=[1, 3, 6, 8, 10, 11, 12, 13, 14, 16, 18, 24, 27, 31, 32, 39, 40, 41, 42, 44,
               46, 47, 48, 49, 59, 63, 64, 66, 68, 69, 71, 73, 75, 79, 80, 82, 84, 89, 90,
               91, 94, 96, 97, 99, 100],
)

# TEST 3: exact make+model, only one such listing exists in the whole dataset
run_test(
    3,
    "\"I want an Infiniti Q50\"",
    filters={"make": "infiniti", "model": "q50"},
    keywords="infiniti q50",
    semantic_query="I'm looking for an Infiniti Q50",
    truth_ids=[18],
)

# TEST 4: structured make filter + an unstructured attribute (interior color) that
# only exists as free text, if it exists at all
run_test(
    4,
    "\"I want a Mercedes-Benz with a black interior\"",
    filters={"make": "mercedes-benz"},
    keywords="mercedes-benz black interior",
    semantic_query="I'm looking for a Mercedes-Benz with a black interior",
    truth_ids=[5, 50, 51, 52, 72, 83],
)

# TEST 5: very specific, unusual paint description -- turned out to be a real,
# exact match in the dataset (id 38), not the "impossible" case it looks like
run_test(
    5,
    "\"I want a Ferrari with a triple layer yellow exterior and two-tone bodywork, black upper part\"",
    filters={"make": "ferrari"},
    keywords="ferrari triple layer yellow two-tone black",
    semantic_query="I'm looking for a Ferrari with a triple layer yellow exterior and two-tone bodywork with a black upper part",
    truth_ids=[38],
)
