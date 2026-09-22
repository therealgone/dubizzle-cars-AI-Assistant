import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

RRF_K = 60
CANDIDATES_PER_SEARCH = 20

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_collection(COLLECTION_NAME)
_model = SentenceTransformer(EMBEDDING_MODEL)

# BM25 needs its own in-memory index, built once from the same corpus as Chroma
_corpus = _collection.get(include=["documents"])
_corpus_ids = _corpus["ids"]
_bm25 = BM25Okapi([doc.lower().split() for doc in _corpus["documents"]])


def _filter_condition(field: str, value) -> dict:
    # a list means "field is any of these" (OR), single value means exact match
    if isinstance(value, list):
        return {field: {"$in": value}}
    return {field: value}


def _metadata_filter(filters: dict) -> list[str]:
    if not filters:
        return []
    conditions = [_filter_condition(field, value) for field, value in filters.items()]
    where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
    return _collection.get(where=where, include=[])["ids"]


def _bm25_search(keywords: str) -> list[str]:
    if not keywords:
        return []
    scores = _bm25.get_scores(keywords.lower().split())
    ranked = sorted(zip(_corpus_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [listing_id for listing_id, score in ranked[:CANDIDATES_PER_SEARCH] if score > 0]


def _semantic_search(query: str) -> list[str]:
    if not query:
        return []
    embedding = _model.encode([query]).tolist()
    result = _collection.query(query_embeddings=embedding, n_results=CANDIDATES_PER_SEARCH)
    return result["ids"][0]


def _rrf_fuse(ranked_lists: list[list[str]]) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, listing_id in enumerate(ranked):
            scores[listing_id] = scores.get(listing_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def search_cars(
    filters: dict | None = None,
    keywords: str | None = None,
    semantic_query: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    results = [
        _metadata_filter(filters or {}),
        _bm25_search(keywords or ""),
        _semantic_search(semantic_query or ""),
    ]
    results = [r for r in results if r]
    if not results:
        return []

    fused_ids = _rrf_fuse(results)[:top_k]
    fetched = _collection.get(ids=fused_ids, include=["metadatas"])
    by_id = dict(zip(fetched["ids"], fetched["metadatas"]))
    return [by_id[listing_id] for listing_id in fused_ids if listing_id in by_id]
