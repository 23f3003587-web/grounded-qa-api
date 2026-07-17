from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

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

def find_grounded_answer(question: str, chunks: List[Chunk]):
    if not chunks or not question or not question.strip():
        return Response(answer="I don't know", citations=[], confidence=0.1, answerable=False)

    q_lower = question.lower().strip()

    # Find best matching chunk
    best_chunk = None
    best_score = 0.0

    for chunk in chunks:
        c_lower = chunk.text.lower()
        # Strong matching
        if q_lower in c_lower or any(word in c_lower for word in q_lower.split() if len(word) > 2):
            score = 1.0 if q_lower in c_lower else 0.8
            if score > best_score:
                best_score = score
                best_chunk = chunk

    if best_chunk:
        # Use the exact chunk text as answer
        answer = best_chunk.text.strip()

        # Optional: make it a bit shorter
        if len(answer) > 250:
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            answer = ' '.join(sentences[:2]).strip() + '.'

        return Response(
            answer=answer,
            citations=[best_chunk.chunk_id],   # Exact chunk_id
            confidence=0.95,
            answerable=True
        )

    # Not answerable
    return Response(
        answer="I don't know",
        citations=[],
        confidence=0.2,
        answerable=False
    )


@app.post("/grounded-qa", response_model=Response)
@app.post("/", response_model=Response)
async def grounded_qa(req: Request):
    return find_grounded_answer(req.question, req.chunks)


@app.get("/grounded-qa")
@app.get("/")
async def grounded_qa_get():
    return {"status": "ok", "message": "Use POST"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
