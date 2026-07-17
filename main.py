from fastapi import FastAPI
from pydantic import BaseModel
import re
import networkx as nx

app = FastAPI(
    title="GraphRAG API",
    version="1.0"
)

####################################
# MODELS
####################################

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

####################################
# HELPERS
####################################

ENTITY_TYPES = {
    "OpenAI": "Organization",
    "Google": "Organization",
    "Microsoft": "Organization",
    "Meta": "Organization",

    "LangChain": "Framework",
    "LlamaIndex": "Framework",
    "TensorFlow": "Framework",
    "PyTorch": "Framework",

    "ChatGPT": "Product",
    "GPT-4": "Product",
    "Claude": "Product"
}


def detect_entities(text):

    entities = []

    for name, typ in ENTITY_TYPES.items():
        if name in text:
            entities.append({
                "name": name,
                "type": typ
            })

    persons = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)+", text)

    for p in persons:
        if p not in ENTITY_TYPES:
            entities.append({
                "name": p,
                "type": "Person"
            })

    unique = []

    seen = set()

    for e in entities:
        if e["name"] not in seen:
            unique.append(e)
            seen.add(e["name"])

    return unique


def detect_relationships(text):

    rels = []

    patterns = [

        ("created by", "CREATED"),
        ("developed by", "DEVELOPED"),
        ("founded by", "FOUNDED"),
        ("authored by", "AUTHORED"),
        ("hired", "HIRED"),
        ("integrates with", "INTEGRATED_INTO"),
        ("integrated with", "INTEGRATED_INTO"),
    ]

    for phrase, rel in patterns:

        if phrase in text:

            left = text.split(phrase)[0]
            right = text.split(phrase)[1]

            left_entities = detect_entities(left)
            right_entities = detect_entities(right)

            if left_entities and right_entities:

                rels.append({
                    "source": right_entities[0]["name"],
                    "target": left_entities[-1]["name"],
                    "relation": rel
                })

    return rels


####################################
# ROOT
####################################

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "GraphRAG API"
    }

####################################
# ENDPOINT 1
####################################

@app.post("/extract-graph")
def extract_graph(req: ExtractRequest):

    entities = detect_entities(req.text)

    relationships = detect_relationships(req.text)

    return {
        "entities": entities,
        "relationships": relationships
    }

####################################
# ENDPOINT 2
####################################

@app.post("/graph-query")
def graph_query(req: GraphQueryRequest):

    G = nx.Graph()

    for e in req.graph["entities"]:
        G.add_node(e["name"])

    for r in req.graph["relationships"]:
        G.add_edge(
            r["source"],
            r["target"],
            relation=r["relation"]
        )

    q = req.question.lower()

    framework = None

    if "openai" in q:

        for r in req.graph["relationships"]:

            if r["target"] == "OpenAI" or r["source"] == "OpenAI":

                framework = r["source"] if r["source"] != "OpenAI" else r["target"]

    if framework is None:

        for e in req.graph["entities"]:

            if e["type"] == "Framework":
                framework = e["name"]
                break

    creator = None

    for r in req.graph["relationships"]:

        if r["target"] == framework and r["relation"] in [
            "CREATED",
            "DEVELOPED",
            "FOUNDED",
            "AUTHORED"
        ]:
            creator = r["source"]

    path = []

    if creator:

        path = [framework, creator]

    return {
        "answer": creator,
        "reasoning_path": path,
        "hops": max(len(path)-1,0)
    }

####################################
# ENDPOINT 3
####################################

@app.post("/community-summary")
def community_summary(req: CommunityRequest):

    names = ", ".join(req.entities)

    rels = ", ".join(
        [
            f'{r["source"]} {r["relation"]} {r["target"]}'
            for r in req.relationships
        ]
    )

    summary = (
        f"This community contains {names}. "
        f"Relationships include: {rels}."
    )

    return {
        "community_id": req.community_id,
        "summary": summary
    }
