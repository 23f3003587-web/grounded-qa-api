from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
from openai import OpenAI  # or Groq, Gemini, etc.

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

# Use your LLM (Groq is fast & cheap, or use Grok/xAI if available)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",  # Change for OpenAI/Gemini
    api_key=os.getenv("LLM_API_KEY")
)

SYSTEM_PROMPT = """
You are a high-reliability Grounded QA system for medical/legal compliance.
Answer the question **strictly using only the provided context chunks**.
- If the answer is not in the chunks → answerable=false, answer="I don't know", citations=[]
- Always cite exact chunk_ids used.
- Be concise and factual.
- Output ONLY valid JSON with keys: answer, citations (list), confidence (0-1), answerable (bool)
"""

@app.post("/grounded-qa", response_model=Response)
async def grounded_qa(req: Request):
    if not req.chunks or not req.question.strip():
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

Generate grounded answer now.
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # or mixtral, gemma2-9b, etc.
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        result = completion.choices[0].message.content
        import json
        data = json.loads(result)

        # Strict validation
        answer = data.get("answer", "I don't know")
        citations = [c for c in data.get("citations", []) if c in [ch.chunk_id for ch in req.chunks]]
        confidence = float(data.get("confidence", 0.0))
        answerable = bool(data.get("answerable", False))

        if not answerable or "I don't know" in answer.lower():
            return Response(
                answer="I don't know",
                citations=[],
                confidence=min(confidence, 0.3),
                answerable=False
            )

        return Response(
            answer=answer,
            citations=citations[:5],  # safety
            confidence=max(0.5, min(confidence, 1.0)),
            answerable=True
        )

    except Exception as e:
        # Fallback for any error
        return Response(
            answer="I don't know",
            citations=[],
            confidence=0.1,
            answerable=False
        )

@app.get("/health")
async def health():
    return {"status": "healthy"}
