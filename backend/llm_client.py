import litellm

from backend.config import GEMINI_API_KEY

# "gemini/" prefix routes via Google AI Studio (API key), not Vertex AI
# lite tier -- gemini-3.6-flash's free quota is only 20 requests/day, exhausted mid-session
MODEL = "gemini/gemini-3.5-flash-lite"


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
