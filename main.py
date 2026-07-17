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
    matched_chunks = []

    for chunk in chunks:
        text_lower = chunk.text.lower()
        # Check for significant word overlap or exact phrase match
        if q_lower in text_lower or any(word in text_lower for word in q_lower.split() if len(word) > 3):
            matched_chunks.append(chunk)

    if matched_chunks:
        # Use the first best match for answer
        best = matched_chunks[0]
        answer = best.text.strip()

        # Keep answer concise
        if len(answer) > 300:
            sentences = re.split(r'[.!?]+', answer)
            answer = '. '.join(sentences[:3]).strip() + '.'

        return Response(
            answer=answer,
            citations=[c.chunk_id for c in matched_chunks[:3]],  # Multiple citations if multiple match
            confidence=0.88,
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
    return {"status": "ok", "message": "Use POST method"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
