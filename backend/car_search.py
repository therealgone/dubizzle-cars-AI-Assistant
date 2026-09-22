import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

RRF_K = 60
CANDIDATES_PER_LEG = 20

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_collection(COLLECTION_NAME)
_model = SentenceTransformer(EMBEDDING_MODEL)

# BM25 needs its own in-memory index, built once from the same corpus as Chroma
_corpus = _collection.get(include=["documents"])
_corpus_ids = _corpus["ids"]
_bm25 = BM25Okapi([doc.lower().split() for doc in _corpus["documents"]])


def _metadata_leg(filters: dict) -> list[str]:
    if not filters:
        return []
    where = filters if len(filters) == 1 else {"$and": [{k: v} for k, v in filters.items()]}
    return _collection.get(where=where, include=[])["ids"]


def _bm25_leg(keywords: str) -> list[str]:
    if not keywords:
        return []
    scores = _bm25.get_scores(keywords.lower().split())
    ranked = sorted(zip(_corpus_ids, scores), key=lambda pair: pair[1], reverse=True)
    return [listing_id for listing_id, score in ranked[:CANDIDATES_PER_LEG] if score > 0]


def _semantic_leg(query: str) -> list[str]:
    if not query:
        return []
    embedding = _model.encode([query]).tolist()
    result = _collection.query(query_embeddings=embedding, n_results=CANDIDATES_PER_LEG)
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
    legs = [
        _metadata_leg(filters or {}),
        _bm25_leg(keywords or ""),
        _semantic_leg(semantic_query or ""),
    ]
    legs = [leg for leg in legs if leg]
    if not legs:
        return []

    fused_ids = _rrf_fuse(legs)[:top_k]
    fetched = _collection.get(ids=fused_ids, include=["metadatas"])
    by_id = dict(zip(fetched["ids"], fetched["metadatas"]))
    return [by_id[listing_id] for listing_id in fused_ids if listing_id in by_id]
