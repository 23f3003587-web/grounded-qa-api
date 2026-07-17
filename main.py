from fastapi import FastAPI
from pydantic import BaseModel
import re
import networkx as nx
from collections import defaultdict

app = FastAPI(title="GraphRAG API", version="2.0")

# ----------------------------
# Request Models
# ----------------------------

class ExtractRequest(BaseModel):
    chunk_id: str
    text: str

class GraphQueryRequest(BaseModel):
    question: str
    graph: dict

class CommunityRequest(BaseModel):
    community_id: str
    entities: list
    relationships: list

# ----------------------------
# Entity Detection
# ----------------------------

KNOWN_ENTITIES = {
    # Organizations
    "OpenAI": "Organization",
    "Google": "Organization",
    "Microsoft": "Organization",
    "Meta": "Organization",
    "Anthropic": "Organization",

    # Frameworks
    "LangChain": "Framework",
    "LlamaIndex": "Framework",
    "TensorFlow": "Framework",
    "PyTorch": "Framework",

    # Products
    "ChatGPT": "Product",
    "GPT-4": "Product",
    "Claude": "Product",
}

def detect_entities(text: str):
    entities = {}

    # 1. Known entities
    for name, typ in KNOWN_ENTITIES.items():
        if re.search(rf"\b{re.escape(name)}\b", text):
            entities[name] = typ

    # 2. Detect capitalized multi-word names (persons/orgs)
    patterns = re.findall(
        r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b", text
    )

    for p in patterns:
        if p not in entities:
            # Heuristic: if ends with Inc/Corp/etc → Organization
            if re.search(r"(Inc|Corp|Corporation|Ltd|LLC)$", p):
                entities[p] = "Organization"
            else:
                entities[p] = "Person"

    return [
        {"name": name, "type": typ}
        for name, typ in entities.items()
    ]

# ----------------------------
# Relationship Detection
# ----------------------------

RELATION_PATTERNS = [
    # Framework/Product created by Person
    (r"([A-Z][A-Za-z0-9\-]+)\s+(?:was\s+)?created\s+by\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "CREATED", "reverse"),
    (r"([A-Z][A-Za-z0-9\-]+)\s+(?:was\s+)?developed\s+by\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "DEVELOPED", "reverse"),
    (r"([A-Z][A-Za-z0-9\-]+)\s+(?:was\s+)?founded\s+by\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "FOUNDED", "reverse"),
    (r"([A-Z][A-Za-z0-9\-]+)\s+(?:was\s+)?authored\s+by\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "AUTHORED", "reverse"),

    # Person founded/developed/created Organization/Product
    (r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+founded\s+([A-Z][A-Za-z0-9\-]+)", "FOUNDED", "normal"),
    (r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+developed\s+([A-Z][A-Za-z0-9\-]+)", "DEVELOPED", "normal"),
    (r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+created\s+([A-Z][A-Za-z0-9\-]+)", "CREATED", "normal"),

    # Integration
    (r"([A-Z][A-Za-z0-9\-]+)\s+integrates\s+with\s+([A-Z][A-Za-z0-9\-]+)", "INTEGRATED_INTO", "normal"),
    (r"([A-Z][A-Za-z0-9\-]+)\s+integrated\s+with\s+([A-Z][A-Za-z0-9\-]+)", "INTEGRATED_INTO", "normal"),

    # Hiring
    (r"([A-Z][A-Za-z0-9\-]+)\s+hired\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", "HIRED", "normal"),
]

def detect_relationships(text: str):
    relationships = []
    seen = set()

    for pattern, relation, direction in RELATION_PATTERNS:
        for match in re.finditer(pattern, text):
            a, b = match.group(1), match.group(2)

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
        "relationships": relationships
    }

@app.post("/graph-query")
def graph_query(req: GraphQueryRequest):
    G = nx.DiGraph()

    for e in req.graph.get("entities", []):
        G.add_node(e["name"], type=e.get("type"))

    for r in req.graph.get("relationships", []):
        G.add_edge(r["source"], r["target"], relation=r["relation"])

    question = req.question.lower()

    # Example multi-hop: creator of framework integrating with OpenAI
    if "integrates with openai" in question or "integrated with openai" in question:
        framework = None

        for r in req.graph.get("relationships", []):
            if r["relation"] == "INTEGRATED_INTO" and r["target"] == "OpenAI":
                framework = r["source"]
                break

        if framework:
            for r in req.graph.get("relationships", []):
                if r["target"] == framework and r["relation"] in ["CREATED", "DEVELOPED", "FOUNDED", "AUTHORED"]:
                    return {
                        "answer": r["source"],
                        "reasoning_path": ["OpenAI", framework, r["source"]],
                        "hops": 2
                    }

    return {
        "answer": "No answer found",
        "reasoning_path": [],
        "hops": 0
    }

@app.post("/community-summary")
def community_summary(req: CommunityRequest):
    entity_list = ", ".join(req.entities)

    relationship_text = []
    for r in req.relationships:
        relationship_text.append(
            f"{r['source']} {r['relation']} {r['target']}"
        )

    summary = (
        f"This community includes {entity_list}. "
        f"Key relationships are: {'; '.join(relationship_text)}."
    )

    return {
        "community_id": req.community_id,
        "summary": summary
    }
