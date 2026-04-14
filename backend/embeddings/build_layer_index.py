import asyncio
import json
from pathlib import Path

from backend.embeddings.embeddings import embed_text
from backend.embeddings.retrieve_layers import build_catalog_embedding_text, select_catalog_columns
from backend.mcp.client.client import mcp_client
from backend.mcp.client.tools_discovery import (
    get_available_databases,
    get_available_schemas,
    get_available_tables,
    get_table_schema,
)


from config import DB_DESCRIPTIONS


def _infer_geometry_column(columns: list[dict[str, str]]) -> str | None:
    for col in columns:
        name = str(col.get("name") or "").strip()
        data_type = str(col.get("type") or "").strip().lower()
        if data_type in {"user-defined", "geometry"}:
            return name
        if name.lower() in {"geom", "geometry", "geometri"}:
            return name
    return None


async def build_metadata_catalog() -> list[dict]:
    catalog: list[dict] = []
    databases = await get_available_databases()

    for database in databases:
        schema_payload = await get_available_schemas(database)
        schemas = schema_payload.get("schemas", []) if isinstance(schema_payload, dict) else []

        for schema in schemas:
            tables_payload = await get_available_tables(database, schema)
            tables = tables_payload.get("tables", []) if isinstance(tables_payload, dict) else []

            for table in tables:
                schema_result = await get_table_schema(database, schema, table)
                raw_columns = schema_result.get("columns", []) if isinstance(schema_result, dict) else []
                columns = [
                    {
                        "name": str(col.get("column_name") or "").strip(),
                        "type": str(col.get("data_type") or "").strip(),
                    }
                    for col in raw_columns
                    if isinstance(col, dict) and col.get("column_name")
                ]
                geometry_column = _infer_geometry_column(columns)
                selected_columns = select_catalog_columns(
                    {
                        "columns": columns,
                        "geometry_column": geometry_column,
                    }
                )

                catalog.append(
                    {
                        "database": database,
                        "database_description": DB_DESCRIPTIONS.get(database, ""),    
                        "schema": schema,
                        "table": table,
                        "full_name": f"{database}.{schema}.{table}",
                        "geometry_column": geometry_column,
                        "selected_columns": selected_columns,
                    }
                )

    return catalog


async def main() -> None:
    catalog = await build_metadata_catalog()
    index = []

    for item in catalog:
        #text = build_catalog_embedding_text(item)
        embedding_text = build_catalog_embedding_text(item)
        index.append(
            {
                **item,
                "embedding_text": embedding_text,
                "embedding": embed_text(embedding_text),
            }
        )

    output_path = Path(__file__).parent / "table_index.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    print(f"Index size: {len(index)}")
    print(f"Saved compact table index to: {output_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(mcp_client.close())
