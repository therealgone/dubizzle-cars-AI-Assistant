import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

CHROMA_PATH = "chroma_data"
COLLECTION_NAME = "cars"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
