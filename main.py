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

    # Find the best matching chunk(s)
    best_chunk = None
    highest_score = 0

    for chunk in chunks:
        c_lower = chunk.text.lower()
        # Score calculation
        score = 0
        if q_lower in c_lower:
            score = 1.0
        else:
            q_words = set(q_lower.split())
            c_words = set(c_lower.split())
            common = len(q_words & c_words)
            score = common / len(q_words) if q_words else 0

        if score > highest_score:
            highest_score = score
            best_chunk = chunk

    if best_chunk and highest_score > 0.25:
        answer = best_chunk.text.strip()
        # Keep answer reasonable length
        if len(answer) > 280:
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            answer = ' '.join(sentences[:2]).strip()

        return Response(
            answer=answer,
            citations=[best_chunk.chunk_id],   # Strict - only real chunk_id
            confidence=0.92,
            answerable=True
        )

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
    return {"status": "ok", "message": "POST required"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
