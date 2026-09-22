import litellm

from backend.config import GEMINI_API_KEY

# alias always resolves to Google's current default flash model
MODEL = "gemini/gemini-flash-latest"


def call_llm(message: str, system_prompt: str) -> str:
    response = litellm.completion(
        model=MODEL,
        api_key=GEMINI_API_KEY,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content
