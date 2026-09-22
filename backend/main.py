from fastapi import FastAPI
from pydantic import BaseModel

from backend.llm_client import call_llm

app = FastAPI()

SYSTEM_PROMPT = "You are a helpful car-shopping assistant."


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    reply = call_llm(request.message, SYSTEM_PROMPT)
    return ChatResponse(reply=reply)
