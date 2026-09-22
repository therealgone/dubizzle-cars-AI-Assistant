import sys
from pathlib import Path

# repo root isn't on sys.path when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.llm_client import call_llm

if __name__ == "__main__":
    print(call_llm("Say hello in one sentence.", "You are a helpful assistant."))
