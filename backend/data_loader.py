import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from backend.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from backend.enrichment import classify_batch, extract_price
from backend.llm_client import describe_llm_error

DATA_PATH = "data/cars_dataset.xlsx"
SHEET_NAME = "cleaned dataset"
CHUNK_SIZE = 20


def load_listings() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    return df.dropna(subset=["Listing_ID"])


def build_search_text(row: pd.Series) -> str:
    return f"{row['make']} {row['model']} {row['trim']} {row['title']} {row['description']}"


def chunk(rows: list[dict], size: int) -> list[list[dict]]:
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def build_metadata(row: dict, classification: dict) -> dict:
    metadata = {
        "listing_id": int(row["Listing_ID"]),
        "year": int(row["year"]),
        # str() -- pandas/Excel infers some cells as pure numbers (e.g. Mazda
        # "3", trim "707"), leaving these fields inconsistently typed unless
        # forced to text every time
        "make": str(row["make"]),
        "model": str(row["model"]),
        "trim": str(row["trim"]),
        "title": str(row["title"]),
        "description": str(row["description"]),
        "photo_url": row["photo_url"],
        "body_type": classification["body_type"],
        "color": classification["color"],
    }
    price = extract_price(f"{row['title']} {row['description']}")
    if price is not None:
        metadata["price_aed"] = price
    return metadata


def run() -> None:
    df = load_listings()
    records = df.to_dict("records")

    print("classifying body_type/color with the LLM...")
    try:
        classifications = classify_batch(
            [{"listing_id": int(r["Listing_ID"]), "make": r["make"], "model": r["model"],
              "trim": r["trim"], "title": r["title"], "description": r["description"]} for r in records]
        )
    except Exception as exc:
        # nothing has been written to Chroma yet, so the next start simply retries
        raise SystemExit(
            f"\nCould not build the search index: {describe_llm_error(exc)}.\n"
            "No data was written; fix the problem above and start again."
        ) from exc

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    for batch in chunk(records, CHUNK_SIZE):
        # lowercase before embedding/storing so $contains substring matching
        # doesn't silently miss real matches over inconsistent ad capitalization
        search_texts = [build_search_text(pd.Series(r)).lower() for r in batch]
        embeddings = model.encode(search_texts).tolist()
        collection.upsert(
            ids=[str(int(r["Listing_ID"])) for r in batch],
            embeddings=embeddings,
            documents=search_texts,
            metadatas=[
                build_metadata(r, classifications[int(r["Listing_ID"])])
                for r in batch
            ],
        )

    print(f"loaded {collection.count()} listings into '{COLLECTION_NAME}'")


def ensure_index() -> None:
    """Startup check: build the Chroma index only if it's missing or incomplete."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    expected = len(load_listings())
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        if client.get_collection(COLLECTION_NAME).count() == expected:
            return
    print("search index missing or incomplete -- building it now (one-time, takes a minute or two)")
    run()


if __name__ == "__main__":
    run()
