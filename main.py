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

def tokenize(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def find_grounded_answer(question: str, chunks: List[Chunk]):
    if not chunks or not question or not question.strip():
        return Response(answer="I don't know", citations=[], confidence=0.1, answerable=False)

    q_lower = question.strip()
    q_tokens = tokenize(q_lower)
    best_sentence = None
    best_chunk = None
    best_score = 0

    for chunk in chunks:
        # Split chunk into sentences
        sentences = re.split(r'(?<=[.!?])\s+', chunk.text.strip())
        for sent in sentences:
            if not sent.strip():
                continue
            sent_tokens = tokenize(sent)
            score = len(q_tokens & sent_tokens)
            if score > best_score:
                best_score = score
                best_sentence = sent.strip()
                best_chunk = chunk

    if best_sentence and best_score > 0 and best_chunk:
        # Clean answer
        answer = best_sentence
        if len(answer) > 280:
            answer = answer[:280].rsplit(' ', 1)[0] + '...'

        return Response(
            answer=answer,
            citations=[best_chunk.chunk_id],
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
    return {"status": "ok", "message": "Use POST method"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
