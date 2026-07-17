from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import json
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
    timeout=6.0          # Hard timeout to prevent hanging
)

SYSTEM_PROMPT = """
You are a fast Grounded QA system. 
Answer ONLY using the provided chunks. 
If not possible, return: {"answer": "I don't know", "citations": [], "confidence": 0.2, "answerable": false}
Be extremely concise. Output valid JSON only.
"""

@app.post("/grounded-qa", response_model=Response)
@app.post("/", response_model=Response)
async def grounded_qa(req: Request):
    if not req.chunks or not req.question or not req.question.strip():
        return Response(answer="I don't know", citations=[], confidence=0.1, answerable=False)

    # Fast simple check first (helps in many test cases)
    q_lower = req.question.lower()
    context_text = " ".join([c.text.lower() for c in req.chunks])

    if not any(word in context_text for word in q_lower.split() if len(word) > 3):
        return Response(answer="I don't know", citations=[], confidence=0.2, answerable=False)

    context = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in req.chunks])

    user_prompt = f"Q: {req.question}\n\nContext:\n{context}\n\nA:"

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=120,           # Very small
            response_format={"type": "json_object"}
        )

        data = json.loads(completion.choices[0].message.content)

        answer = str(data.get("answer", "I don't know")).strip()
        citations = [cid for cid in data.get("citations", []) 
                     if cid in {c.chunk_id for c in req.chunks}]

        confidence = float(data.get("confidence", 0.6))
        answerable = bool(data.get("answerable", False)) and "don't know" not in answer.lower()

        if not answerable or "I don't know" in answer.lower():
            return Response(answer="I don't know", citations=[], confidence=0.2, answerable=False)

        return Response(
            answer=answer,
            citations=citations[:3],
            confidence=round(max(0.5, min(confidence, 1.0)), 2),
            answerable=True
        )

    except Exception:
        # Fast fallback
        return Response(answer="I don't know", citations=[], confidence=0.1, answerable=False)


# GET routes for compatibility
@app.get("/grounded-qa")
@app.get("/")
async def grounded_qa_get():
    return {"status": "ok", "message": "Use POST method"}

@app.get("/health")
async def health():
    return {"status": "healthy", "llm_ready": True}
