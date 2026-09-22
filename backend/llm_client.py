import litellm

from backend.config import GEMINI_API_KEY

# "gemini/" prefix routes via Google AI Studio (API key), not Vertex AI
MODEL = "gemini/gemini-3.6-flash"


def call_llm(message: str, system_prompt: str) -> str:
    response = litellm.completion(
        model=MODEL,
        api_key=GEMINI_API_KEY,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        num_retries=3,  # retries on 503/timeout with backoff
    )
    return response.choices[0].message.content
