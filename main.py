from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
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

# Pure Python Grounded QA - Very Fast
def find_grounded_answer(question: str, chunks: List[Chunk]):
    if not chunks or not question.strip():
        return Response(answer="I don't know", citations=[], confidence=0.1, answerable=False)

    q_lower = question.lower().strip()
    best_chunk = None
    best_score = 0

    for chunk in chunks:
        text_lower = chunk.text.lower()
        # Score based on word overlap
        words = set(q_lower.split())
        chunk_words = set(text_lower.split())
        score = len(words & chunk_words) / len(words) if words else 0

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk and best_score > 0.3:
        # Extract short answer from chunk
        answer = best_chunk.text.strip()
        # Try to make it more concise
        if len(answer) > 200:
            sentences = re.split(r'[.!?]', answer)
            answer = '.'.join(sentences[:2]).strip() + '.'

        return Response(
            answer=answer,
            citations=[best_chunk.chunk_id],
            confidence=round(best_score, 2),
            answerable=True
        )
    
    return Response(answer="I don't know", citations=[], confidence=0.2, answerable=False)


@app.post("/grounded-qa", response_model=Response)
@app.post("/", response_model=Response)
async def grounded_qa(req: Request):
    return find_grounded_answer(req.question, req.chunks)


@app.get("/grounded-qa")
@app.get("/")
async def grounded_qa_get():
    return {"status": "ok", "message": "Use POST method with JSON body"}


@app.get("/health")
async def health():
    return {"status": "healthy", "llm_ready": "pure_python"}
