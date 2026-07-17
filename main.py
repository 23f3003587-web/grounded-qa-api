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

# API Key & Client with timeout
LLM_API_KEY = os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY environment variable is not set!")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=LLM_API_KEY,
    timeout=5.0
)

SYSTEM_PROMPT = """
You are a strict Grounded QA system.
- Answer ONLY using the provided context chunks.
- You MUST cite the exact chunk_id(s) used.
- If the answer is not in the chunks, reply with: 
  {"answer": "I don't know", "citations": [], "confidence": 0.2, "answerable": false}
- Keep answer short and factual.
- Output ONLY valid JSON, no extra text.
"""

@app.post("/grounded-qa", response_model=Response)
@app.post("/", response_model=Response)
async def grounded_qa(req: Request):
    if not req.chunks or not req.question or not req.question.strip():
        return Response(answer="I don't know", citations=[], confidence=0.1, answerable=False)

    # Create context with IDs
    context = "\n\n".join([f"Chunk {c.chunk_id}: {c.text}" for c in req.chunks])

    user_prompt = f"""
Context:
{context}

Question: {req.question}

Provide a grounded answer.
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"}
        )

        data = json.loads(completion.choices[0].message.content.strip())

        answer = str(data.get("answer", "")).strip()
        raw_citations = data.get("citations", [])

        # Strict citation validation
        valid_chunk_ids = {c.chunk_id for c in req.chunks}
        citations = [cid for cid in raw_citations if str(cid) in valid_chunk_ids]

        answerable = bool(data.get("answerable", False)) and answer and "don't know" not in answer.lower()

        if not answerable or not answer or "I don't know" in answer.lower():
            return Response(
                answer="I don't know",
                citations=[],
                confidence=0.2,
                answerable=False
            )

        return Response(
            answer=answer,
            citations=citations[:5],
            confidence=round(float(data.get("confidence", 0.75)), 2),
            answerable=True
        )

    except Exception:
        # Safe fallback
        return Response(
            answer="I don't know",
            citations=[],
            confidence=0.1,
            answerable=False
        )


# GET support
@app.get("/grounded-qa")
@app.get("/")
async def grounded_qa_get():
    return {"status": "ok", "message": "This endpoint requires POST method with JSON body. See /docs"}


@app.get("/health")
async def health():
    return {"status": "healthy", "llm_ready": True}
