import json
from pathlib import Path
from typing import Any
import sys, os, time, asyncio
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))






from backend.Client.catalogBuilder import build_metadata_catalog
from backend.embeddings.embeddings import embed_text
from backend.embeddings.retrieve_layers import retrieve_relevant_tables_with_embeddings, format_table_context



from backend.tools.langchain_tools import (
    tool_get_available_databases,
    tool_get_available_schemas,
    tool_get_available_tables,
    tool_get_table_schema,
)


def build_table_description(item: dict[str, Any]) -> str:
    cols = ", ".join(col["name"] for col in item.get("columns", []))
    return (
        f"Database: {item['database']}. "
        f"Schema: {item['schema']}. "
        f"Table: {item['table']}. "
        f"Columns: {cols}."
    )


async def main():
    catalog = await build_metadata_catalog(
        get_available_databases=tool_get_available_databases,
        get_available_schemas=tool_get_available_schemas,
        get_available_tables=tool_get_available_tables,
        get_table_schema=tool_get_table_schema,
    )

    index = []
    for item in catalog:
        desc = build_table_description(item)
        vec = embed_text(desc)

        index.append({
            "database": item["database"],
            "schema": item["schema"],
            "table": item["table"],
            "full_name": item["full_name"],
            "columns": item["columns"],
            "description": desc,
            "embedding": vec,
        })

    output_path = Path(__file__).parent / "table_index.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    print(f"Index size: {len(index)}")
    print(f"Saved to: {output_path}")

    tables = retrieve_relevant_tables_with_embeddings(
    "How many buildings are within 100 meters of flood zones?",
        k=5,
    )
    
    for t in tables:
        print(t["full_name"], t["score"])
    catalog_text = format_table_context(tables)

    print(catalog_text)
    v = format_table_context(catalog_text)
    print(v)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())