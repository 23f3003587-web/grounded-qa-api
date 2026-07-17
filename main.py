from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
from openai import OpenAI

app = FastAPI(title="GraphRAG API", version="2.0")

# ----------------------------
# CORS
# ----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# LLM client (Groq)
# ----------------------------

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("LLM_API_KEY"),
)

# ----------------------------
# Request Models
# ----------------------------

class ExtractRequest(BaseModel):
    chunk_id: str
    text: str

class GraphQueryRequest(BaseModel):
    question: str
    graph: Dict[str, Any]

class CommunityRequest(BaseModel):
    community_id: str
    entities: List[str]
    relationships: List[Dict[str, Any]]

# ----------------------------
# Prompts
# ----------------------------

EXTRACT_SYSTEM = """
You are an entity and relationship extractor for a knowledge graph used in a GraphRAG system.

From the given text, extract:

- Entities: objects of type Person, Organization, Product, Framework.
- Relationships: directed edges with relation in {FOUNDED, DEVELOPED, CREATED, INTEGRATED_INTO, HIRED, AUTHORED, EMPLOYED_BY, ACQUIRED_BY, WORKS_AT, MEMBER_OF}.

Rules:
- Each entity must have "name" (string) and "type" (one of: Person, Organization, Product, Framework).
- Each relationship must have "source", "target", "relation".
- Only use information explicitly present in the text; do not invent facts.
- Be exhaustive: if the text mentions multiple entities or relationships, include all of them.
- Return ONLY valid JSON with keys: "entities", "relationships".
"""

EXTRACT_USER_TEMPLATE = """
Text chunk:
{chunk}

Extract all entities and relationships as JSON.
"""

GRAPH_QUERY_SYSTEM = """
You are a multi-hop reasoning engine over a knowledge graph.

You are given:
- A question in natural language.
- A graph with:
  - "entities": list of {name, type}
  - "relationships": list of {source, target, relation}

Your task:
- Answer the question using multi-hop reasoning over the provided graph only.
- Identify a path of entities connected by relationships that justifies the answer.
- Return:
  - "answer": string (the final answer)
  - "reasoning_path": list of entity names forming the path used (in order)
  - "hops": integer (number of edges in the path)

If no answer can be derived from the graph, return:
{
  "answer": "No answer found",
  "reasoning_path": [],
  "hops": 0
}

Return ONLY valid JSON.
"""

GRAPH_QUERY_USER_TEMPLATE = """
Question: {question}

Graph:
{graph_json}

Perform multi-hop reasoning and return the answer in JSON.
"""

COMMUNITY_SYSTEM = """
You are a summarizer for a community (connected subgraph) in a knowledge graph.

You are given:
- A community_id (string).
- A list of entity names in this community.
- A list of relationships (each with source, target, relation).

Your task:
- Produce a concise, coherent natural-language summary describing:
  - The main entities in this community.
  - The key relationships among them (creation, employment, integration, acquisition, etc.).
- Include the community_id in the output.

Return ONLY valid JSON with keys:
- "community_id"
- "summary"
"""

COMMUNITY_USER_TEMPLATE = """
Community ID: {community_id}

Entities: {entities}

Relationships:
{relationships_json}

Generate a community summary in JSON.
"""

# ----------------------------
# LLM JSON helper
# ----------------------------

def call_llm_json(system: str, user: str) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content
    data = json.loads(raw)
    return data

# ----------------------------
# Endpoints
# ----------------------------

@app.get("/")
def root():
    return {"status": "running", "service": "GraphRAG API"}

@app.post("/extract-graph")
def extract_graph(req: ExtractRequest):
    user_prompt = EXTRACT_USER_TEMPLATE.format(chunk=req.text)
    try:
        result = call_llm_json(EXTRACT_SYSTEM, user_prompt)
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])

        # Normalize and validate
        valid_types = {"Person", "Organization", "Product", "Framework"}
        entities = [
            {"name": e.get("name", ""), "type": e.get("type", "")}
            for e in entities
            if e.get("name") and e.get("type") in valid_types
        ]

        valid_rels = {
            "FOUNDED", "DEVELOPED", "CREATED", "INTEGRATED_INTO",
            "HIRED", "AUTHORED", "EMPLOYED_BY", "ACQUIRED_BY",
            "WORKS_AT", "MEMBER_OF"
        }
        relationships = [
            {
                "source": r.get("source", ""),
                "target": r.get("target", ""),
                "relation": r.get("relation", "")
            }
            for r in relationships
            if r.get("source") and r.get("target") and r.get("relation") in valid_rels
        ]

        return {
            "entities": entities,
            "relationships": relationships,
        }
    except Exception:
        # Safe fallback: return empty but valid structure
        return {
            "entities": [],
            "relationships": [],
        }

@app.post("/graph-query")
def graph_query(req: GraphQueryRequest):
    graph_json = json.dumps(req.graph, indent=2)
    user_prompt = GRAPH_QUERY_USER_TEMPLATE.format(
        question=req.question,
        graph_json=graph_json,
    )
    try:
        result = call_llm_json(GRAPH_QUERY_SYSTEM, user_prompt)
        answer = result.get("answer", "No answer found")
        reasoning_path = result.get("reasoning_path", [])
        hops = result.get("hops", 0)

        if not isinstance(reasoning_path, list):
            reasoning_path = []
        if not isinstance(hops, int) or hops < 0:
            hops = 0

        return {
            "answer": answer,
            "reasoning_path": reasoning_path,
            "hops": hops,
        }
    except Exception:
        return {
            "answer": "No answer found",
            "reasoning_path": [],
            "hops": 0,
        }

@app.post("/community-summary")
def community_summary(req: CommunityRequest):
    relationships_json = json.dumps(req.relationships, indent=2)
    entities_str = ", ".join(req.entities)
    user_prompt = COMMUNITY_USER_TEMPLATE.format(
        community_id=req.community_id,
        entities=entities_str,
        relationships_json=relationships_json,
    )
    try:
        result = call_llm_json(COMMUNITY_SYSTEM, user_prompt)
        summary = result.get("summary", "")
        if not summary:
            # Fallback simple summary
            summary = f"This community includes {entities_str}."

        return {
            "community_id": req.community_id,
            "summary": summary,
        }
    except Exception:
        summary = f"This community includes {entities_str}."
        return {
            "community_id": req.community_id,
            "summary": summary,
        }
