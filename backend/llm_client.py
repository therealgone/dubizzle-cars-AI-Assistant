import litellm

from backend.config import GEMINI_API_KEY

# "gemini/" prefix routes via Google AI Studio (API key), not Vertex AI
# lite tier -- gemini-3.6-flash's free quota is only 20 requests/day, exhausted mid-session
MODEL = "gemini/gemini-3.5-flash-lite"


def describe_llm_error(exc: Exception) -> str:
    """Plain-English reason a Gemini call failed, safe to show to a user."""
    if isinstance(exc, litellm.RateLimitError):
        return ("the Gemini API usage limit has been reached (the free tier is quota-limited) "
                "-- wait a minute, or until the daily quota resets, and try again")
    if isinstance(exc, litellm.AuthenticationError):
        return "the Gemini API key was rejected -- check GEMINI_API_KEY in .env"
    if isinstance(exc, litellm.NotFoundError):
        return f"the configured Gemini model ({MODEL}) was not found or is no longer available"
    if isinstance(exc, (litellm.ServiceUnavailableError, litellm.InternalServerError,
                        litellm.Timeout, litellm.APIConnectionError)):
        return "the Gemini API is temporarily unavailable or too slow to respond -- try again shortly"
    return f"the Gemini API returned an unexpected error ({type(exc).__name__})"


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
