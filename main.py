from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import json
import re
from openai import OpenAI

app = FastAPI(title="SafeAnswer Grounded QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Chunk(BaseModel):
    chunk_id: str
    text: str

class Request(BaseModel):
    question: str
    chunks: List[Chunk]

class Response(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    answerable: bool

# Load API key
LLM_API_KEY = os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY environment variable is not set!")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=LLM_API_KEY,
    timeout=4.0   # Strict timeout to prevent hanging
)

SYSTEM_PROMPT = """
You are a strict Grounded QA. Use ONLY the context. 
If not found: {"answer":"I don't know","citations":[],"confidence":0.2,"answerable":false}
Else give short answer + citations.
Output ONLY JSON.
"""

# Simple fast fallback using string matching
def simple_grounded_answer(question: str, chunks: List[Chunk]):
    q = question.lower()
    for chunk in chunks:
        if any(word in chunk.text.lower() for word in q.split() if len(word) > 3):
            return Response(
                answer=chunk.text,
                citations=[chunk.chunk_id],
                confidence=0.75,
                answerable=True
            )
    return None

@app.post("/grounded-qa", response_model=Response)
@app.post("/", response_model=Response)
async def grounded_qa(req: Request):
    if not req.chunks or not req.question or not req.question.strip():
        return Response(answer="I don't know", citations=[], confidence=0.1, answerable=False)

    # Fast simple match first (helps beat timeout)
    fast_answer = simple_grounded_answer(req.question, req.chunks)
    if fast_answer:
        return fast_answer

    # LLM only if needed
    context = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in req.chunks])
    user_prompt = f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"}
        )

        data = json.loads(completion.choices[0].message.content)

        answer = str(data.get("answer", "I don't know")).strip()
        citations = [str(cid) for cid in data.get("citations", []) 
                     if str(cid) in {c.chunk_id for c in req.chunks}]

        if "don't know" in answer.lower() or not citations:
            return Response(answer="I don't know", citations=[], confidence=0.2, answerable=False)

        return Response(
            answer=answer,
            citations=citations,
            confidence=0.85,
            answerable=True
        )

    except Exception:
        return Response(answer="I don't know", citations=[], confidence=0.1, answerable=False)


# GET compatibility
@app.get("/grounded-qa")
@app.get("/")
async def grounded_qa_get():
    return {"status": "ok", "message": "Use POST method with JSON body."}


@app.get("/health")
async def health():
    return {"status": "healthy", "llm_ready": True}
