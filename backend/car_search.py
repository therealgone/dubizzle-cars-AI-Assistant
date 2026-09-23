import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

RRF_K = 60
CANDIDATES_PER_SEARCH = 40

_TOKEN_RE = re.compile(r"\w+")

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_collection(COLLECTION_NAME)
_model = SentenceTransformer(EMBEDDING_MODEL)

# BM25 needs its own in-memory index, built once from the same corpus as Chroma
_corpus = _collection.get(include=["documents"])
_corpus_ids = _corpus["ids"]


def _tokenize(text: str) -> list[str]:
    # word-boundary tokenizer, not .split() -- "-GCC" or "GCC," must still match "gcc"
    return _TOKEN_RE.findall(text.lower())


_bm25 = BM25Okapi([_tokenize(doc) for doc in _corpus["documents"]])


# fallback fields for anything that doesn't fit make/model/year/trim -- these are
# free text, so they need substring matching (where_document) not exact equality
_FREE_TEXT_FIELDS = {"title", "description"}


def _normalize(value):
    # stored metadata is all lowercase, but an LLM naturally capitalizes
    # proper nouns ("Ferrari", "Mercedes-Benz") -- match case-insensitively
    # by normalizing here rather than trusting callers to lowercase first
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return [v.lower() if isinstance(v, str) else v for v in value]
    return value


def _metadata_filter(filters: dict) -> list[str]:
    if not filters:
        return []

    field_conditions = []
    text_conditions = []
    for field, value in filters.items():
        if field in _FREE_TEXT_FIELDS:
            text_conditions.append({"$contains": str(value).lower()})
        elif isinstance(value, list):
            # a list value means "field is any of these" (OR), single value means exact match
            field_conditions.append({field: {"$in": _normalize(value)}})
        else:
            field_conditions.append({field: _normalize(value)})

    kwargs = {"include": []}
    if field_conditions:
        kwargs["where"] = field_conditions[0] if len(field_conditions) == 1 else {"$and": field_conditions}
    if text_conditions:
        kwargs["where_document"] = text_conditions[0] if len(text_conditions) == 1 else {"$and": text_conditions}

    return _collection.get(**kwargs)["ids"]


def _bm25_search(keywords: str) -> list[str]:
    if not keywords:
        return []
    scores = _bm25.get_scores(_tokenize(keywords))
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
