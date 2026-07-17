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
Use ONLY the provided context chunks. Never use outside knowledge.
If the question cannot be answered from the chunks, output:
{"answer": "I don't know", "citations": [], "confidence": 0.2, "answerable": false}

Otherwise, give a short direct answer and list the chunk_ids you used.
Output ONLY valid JSON, nothing else.
"""

@app.post("/grounded-qa", response_model=Response)
@app.post("/", response_model=Response)
async def grounded_qa(req: Request):
    if not req.chunks or not req.question or not req.question.strip():
        return Response(answer="I don't know", citations=[], confidence=0.1, answerable=False)

    context = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in req.chunks])

    user_prompt = f"""
Context:
{context}

Question: {req.question}

Answer using only the context above.
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )

        raw = completion.choices[0].message.content.strip()
        data = json.loads(raw)

        answer = str(data.get("answer", "I don't know")).strip()
        citations = [str(cid) for cid in data.get("citations", []) 
                     if str(cid) in {c.chunk_id for c in req.chunks}]

        confidence = float(data.get("confidence", 0.6))
        answerable = bool(data.get("answerable", True)) and "don't know" not in answer.lower()

        if not answerable or "I don't know" in answer.lower():
            return Response(answer="I don't know", citations=[], confidence=0.2, answerable=False)

        return Response(
            answer=answer,
            citations=citations,
            confidence=round(max(0.5, min(confidence, 1.0)), 2),
            answerable=True
        )

    except Exception:
        return Response(answer="I don't know", citations=[], confidence=0.1, answerable=False)


@app.get("/grounded-qa")
@app.get("/")
async def grounded_qa_get():
    return {"status": "ok", "use": "POST"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
