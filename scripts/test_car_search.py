import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.car_search import _bm25_search, _collection, _metadata_filter, _semantic_search, search_cars


def show(label, ids):
    print(f"\n--- {label} ---")
    print("count:", len(ids))
    if not ids:
        return
    fetched = _collection.get(ids=list(ids), include=["metadatas"])
    by_id = dict(zip(fetched["ids"], fetched["metadatas"]))
    for i in ids:
        m = by_id[i]
        print(i, m["year"], m["make"], m["model"], m["trim"])


# 1. single-field exact match
show("metadata filter: make=bentley", _metadata_filter({"make": "bentley"}))

# 2. multi-field AND
show("metadata filter: make=bentley AND year=2016", _metadata_filter({"make": "bentley", "year": 2016}))

# 3. OR within a field
show("metadata filter: model in [continental, c-class]", _metadata_filter({"model": ["continental", "c-class"]}))

# 4. BM25 keyword leg alone
show("bm25 search: 'bentley continental'", _bm25_search("bentley continental"))

# 5. semantic leg alone
show("semantic search: 'luxury convertible sports car'", _semantic_search("luxury convertible sports car"))

# 6. full fused search_cars
fused = search_cars(
    filters={"make": "bentley"},
    keywords="bentley continental gtc",
    semantic_query="luxury convertible",
)
print("\n--- search_cars fused: filter=bentley, keywords='bentley continental gtc', semantic='luxury convertible' ---")
print("count:", len(fused))
for m in fused:
    print(m["listing_id"], m["year"], m["make"], m["model"], m["trim"])

# 7. mercedes-benz, 2019-2023, run through all three methods as an LLM might call them
show(
    "metadata_filter(make=mercedes-benz, year=[2019..2023])",
    _metadata_filter({"make": "mercedes-benz", "year": [2019, 2020, 2021, 2022, 2023]}),
)

show(
    "bm25_search('mercedes-benz 2019 2020 2021 2022 2023')",
    _bm25_search("mercedes-benz 2019 2020 2021 2022 2023"),
)

show(
    "semantic_search('Mercedes-Benz from 2019 to 2023')",
    _semantic_search("Mercedes-Benz from 2019 to 2023"),
)
