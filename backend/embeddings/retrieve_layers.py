import json
import math
from pathlib import Path
from .embeddings import embed_text


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def load_index(path: str = "table_index.json") -> list[dict]:
    base = Path(__file__).parent
    index_path = base / path
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_relevant_tables_with_embeddings(
    user_text: str,
    k: int = 5,
    index_path: str = "table_index.json",
) -> list[dict]:
    index = load_index(index_path)
    q = embed_text(user_text)

    scored = []
    for item in index:
        s = cosine(q, item["embedding"])
        scored.append({
            **item,
            "score": s,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def format_table_context(top_tables: list[dict]) -> str:
    lines = ["AVAILABLE TABLES (most relevant):"]

    for item in top_tables:
        if "sample" in item["table"].lower():
            continue

        cols = ", ".join(
            f"{c['name']} ({c['type']})"
            for c in item.get("columns", [])[:8]
        )

        lines.append(
            f"- {item['database']}.{item['schema']}.{item['table']}: {cols}"
        )

    return "\n".join(lines)