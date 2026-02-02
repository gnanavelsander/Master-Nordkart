
import json
import math
from .embeddings import embed_text
from pathlib import Path

def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)

def load_index(path: str = "layer_index.json"):
        
    base = Path(__file__).parent  
    index_path = base / path
        
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)

def retrieve_top_layers(user_text: str, k: int = 5, index_path: str = "layer_index.json"):
    index = load_index(index_path)
    q = embed_text(user_text)

    scored = []
    for item in index:
        s = cosine(q, item["embedding"])
        scored.append((s, item["layer"], item["description"]))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]

def format_layer_context(top_layers) -> str:
        
    lines = ["AVAILABLE LAYERS (most relevant):"]
    for score, layer, desc in top_layers:
        lines.append(f"- {layer}: {desc}")
    return "\n".join(lines)
