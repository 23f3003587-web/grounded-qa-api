from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import json
from openai import OpenAI

app = FastAPI(title="SafeAnswer Grounded QA API")

# CORS
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

# Load API key from environment only
LLM_API_KEY = os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY environment variable is not set!")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=LLM_API_KEY
)

SYSTEM_PROMPT = """
You are a high-reliability Grounded QA system for medical and legal compliance.
Answer the question strictly using only the provided context chunks.
- If the answer cannot be found in the chunks, return answerable=false, answer="I don't know", citations=[]
- Always cite the exact chunk_ids used.
- Be concise and factual.
- Output ONLY valid JSON with keys: answer, citations (list of strings), confidence (0.0-1.0), answerable (boolean)
"""

# Main endpoint - POST (primary)
@app.post("/grounded-qa", response_model=Response)
@app.post("/", response_model=Response)
async def grounded_qa(req: Request):
    if not req.chunks or not req.question or not req.question.strip():
        return Response(
            answer="I don't know",
            citations=[],
            confidence=0.1,
            answerable=False
        )

    context = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in req.chunks])

    user_prompt = f"""
Question: {req.question}

Context chunks:
{context}

Answer strictly from the context only.
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=600,
            response_format={"type": "json_object"}
        )

        data = json.loads(completion.choices[0].message.content)

        answer = data.get("answer", "I don't know").strip()
        citations = [cid for cid in data.get("citations", []) 
                     if cid in {c.chunk_id for c in req.chunks}]

        confidence = float(data.get("confidence", 0.5))
        answerable = bool(data.get("answerable", True)) and "don't know" not in answer.lower()

        if not answerable or "I don't know" in answer.lower():
            return Response(
                answer="I don't know",
                citations=[],
                confidence=min(confidence, 0.3),
                answerable=False
            )

        return Response(
            answer=answer,
            citations=citations,
            confidence=round(max(0.6, min(confidence, 1.0)), 2),
            answerable=True
        )

    except Exception:
        return Response(
            answer="I don't know",
            citations=[],
            confidence=0.1,
            answerable=False
        )


# GET support for judge / testing
@app.get("/grounded-qa")
@app.get("/")
async def grounded_qa_get():
    return {
        "status": "ok",
        "message": "This endpoint accepts POST requests only. Use POST method with JSON body.",
        "docs_url": "/docs"
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "llm_ready": bool(LLM_API_KEY)}
