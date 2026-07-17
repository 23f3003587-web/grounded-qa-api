from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import re
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
# LLM client (Groq) – used only for graph-query and community-summary
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
# Local entity detection (no LLM)
# ----------------------------

KNOWN_ENTITIES = {
    # Organizations
    "OpenAI": "Organization",
    "Google": "Organization",
    "Microsoft": "Organization",
    "Meta": "Organization",
    "Anthropic": "Organization",
    "DeepMind": "Organization",
    "Hugging Face": "Organization",
    "Stability AI": "Organization",
    "Cohere": "Organization",
    "AI21": "Organization",
    "Mistral AI": "Organization",
    "GraphMind Systems": "Organization",

    # Frameworks
    "LangChain": "Framework",
    "LlamaIndex": "Framework",
    "TensorFlow": "Framework",
    "PyTorch": "Framework",
    "Hugging Face Transformers": "Framework",
    "Haystack": "Framework",
    "DSPy": "Framework",

    # Products / Models
    "ChatGPT": "Product",
    "GPT-4": "Product",
    "GPT-3.5": "Product",
    "Claude": "Product",
    "Gemini": "Product",
    "Llama": "Product",
    "Mistral": "Product",
}

def detect_entities(text: str) -> List[Dict[str, str]]:
    entities = {}

    # 1. Known entities
    for name, typ in KNOWN_ENTITIES.items():
        if re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
            entities[name] = typ

    # 2. Person names: "Firstname Lastname" patterns
    person_pattern = r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b"
    for m in re.finditer(person_pattern, text):
        name = m.group(1)
        if name in entities:
            continue
        # Avoid obvious non-persons
        if re.search(r"(Inc|Corp|Corporation|Ltd|LLC|AI|Labs|Systems)$", name):
            continue
        entities[name] = "Person"

    # 3. Organization-like names ending with typical suffixes
    org_pattern = r"\b([A-Z][A-Za-z0-9]*(?:\s[A-Z][A-Za-z0-9]*)*\s+(?:Inc|Corp|Corporation|Ltd|LLC|Labs|AI|Systems|Technologies))\b"
    for m in re.finditer(org_pattern, text):
        name = m.group(1)
        if name not in entities:
            entities[name] = "Organization"

    # 4. Detect "X is a framework/library/platform"
    framework_hint = r"\b([A-Z][A-Za-z0-9\-]+)\s+is\s+a\s+(framework|library|platform|toolkit)\b"
    for m in re.finditer(framework_hint, text, flags=re.IGNORECASE):
        name = m.group(1)
        if name not in entities:
            entities[name] = "Framework"

    # 5. Detect "X is a product/model"
    product_hint = r"\b([A-Z][A-Za-z0-9\-]+)\s+is\s+a\s+(product|model|system|assistant)\b"
    for m in re.finditer(product_hint, text, flags=re.IGNORECASE):
        name = m.group(1)
        if name not in entities:
            entities[name] = "Product"

    return [{"name": name, "type": typ} for name, typ in entities.items()]

# ----------------------------
# Local relationship detection (no LLM)
# ----------------------------

RELATION_PATTERNS = [
    # X was created by Y
    (r"\b([A-Z][A-Za-z0-9\-]+)\s+was\s+created\s+by\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "CREATED", "reverse"),
    # X was developed by Y
    (r"\b([A-Z][A-Za-z0-9\-]+)\s+was\s+developed\s+by\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "DEVELOPED", "reverse"),
    # X was founded by Y
    (r"\b([A-Z][A-Za-z0-9\-]+)\s+was\s+founded\s+by\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "FOUNDED", "reverse"),
    # X was authored by Y
    (r"\b([A-Z][A-Za-z0-9\-]+)\s+was\s+authored\s+by\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "AUTHORED", "reverse"),

    # Y founded X
    (r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+founded\s+([A-Z][A-Za-z0-9\-]+)", "FOUNDED", "normal"),
    # Y developed X
    (r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+developed\s+([A-Z][A-Za-z0-9\-]+)", "DEVELOPED", "normal"),
    # Y created X
    (r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+created\s+([A-Z][A-Za-z0-9\-]+)", "CREATED", "normal"),
    # Y authored X
    (r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+authored\s+([A-Z][A-Za-z0-9\-]+)", "AUTHORED", "normal"),

    # X integrates with Y
    (r"\b([A-Z][A-Za-z0-9\-]+)\s+integrates\s+with\s+([A-Z][A-Za-z0-9\-]+)", "INTEGRATED_INTO", "normal"),
    (r"\b([A-Z][A-Za-z0-9\-]+)\s+integrated\s+with\s+([A-Z][A-Za-z0-9\-]+)", "INTEGRATED_INTO", "normal"),

    # X hired Y
    (r"\b([A-Z][A-Za-z0-9\-]+)\s+hired\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "HIRED", "normal"),

    # X works at Y / is at Y
    (r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+works\s+at\s+([A-Z][A-Za-z0-9\-]+)", "WORKS_AT", "normal"),
    (r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+is\s+at\s+([A-Z][A-Za-z0-9\-]+)", "WORKS_AT", "normal"),

    # X is CEO/CTO/etc of Y
    (r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+is\s+(?:CEO|CTO|CFO|COO|founder|co-founder)\s+of\s+([A-Z][A-Za-z0-9\-]+)",
     "WORKS_AT", "normal"),
]

def detect_relationships(text: str) -> List[Dict[str, str]]:
    relationships = []
    seen = set()

    for pattern, relation, direction in RELATION_PATTERNS:
        for m in re.finditer(pattern, text):
            a, b = m.group(1), m.group(2)
            if direction == "reverse":
                source, target = b, a
            else:
                source, target = a, b

            key = (source, target, relation)
            if key not in seen:
                relationships.append({
                    "source": source,
                    "target": target,
                    "relation": relation
                })
                seen.add(key)

    return relationships

# ----------------------------
# Prompts for LLM-based endpoints
# ----------------------------

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
    entities = detect_entities(req.text)
    relationships = detect_relationships(req.text)

    return {
        "entities": entities,
        "relationships": relationships,
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
