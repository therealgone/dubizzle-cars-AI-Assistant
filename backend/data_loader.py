import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from backend.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL

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


def run() -> None:
    df = load_listings()
    records = df.to_dict("records")

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
                {
                    "listing_id": int(r["Listing_ID"]),
                    "year": int(r["year"]),
                    "make": r["make"],
                    "model": r["model"],
                    "trim": r["trim"],
                    "title": r["title"],
                    "description": r["description"],
                    "photo_url": r["photo_url"],
                }
                for r in batch
            ],
        )

    print(f"loaded {collection.count()} listings into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    run()
