from __future__ import annotations
from typing import Any






def parse_columns(schema_data: dict) -> list[dict[str, str]]:
    return [
        {
            "name": col["column_name"],
            "type": col["data_type"],
        }
        for col in schema_data.get("columns", [])
    ]


async def build_metadata_catalog(
    *,
    get_available_databases,
    get_available_schemas,
    get_available_tables,
    get_table_schema,
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []

    databases = await get_available_databases.ainvoke({})

    for db in databases:
        schema_result = await get_available_schemas.ainvoke({"databaseName": db})
        schemas = schema_result["schemas"]

        for schema in schemas:
            table_result = await get_available_tables.ainvoke({
                "databaseName": db,
                "schemaName": schema,
            })
            tables = table_result["tables"]

            for table in tables:
                schema_data = await get_table_schema.ainvoke({
                    "databaseName": db,
                    "schemaName": schema,
                    "table_name": table,
                })

                catalog.append({
                    "database": db,
                    "schema": schema,
                    "table": table,
                    "full_name": f"{db}.{schema}.{table}",
                    "columns": parse_columns(schema_data),
                })

    return catalog









def tokenize(text: str) -> set[str]:
    text = text.lower()
    parts = re.split(r"[^a-zA-Z0-9_æøå]+", text)
    return {p for p in parts if p}


def build_search_text(item: dict[str, Any]) -> str:
    cols = " ".join(col["name"] for col in item.get("columns", []))
    return f"{item['database']} {item['schema']} {item['table']} {cols}".lower()


def score_table(user_tokens: set[str], item: dict[str, Any]) -> int:
    score = 0
    search_text = build_search_text(item)
    table_name = item["table"].lower()

    for token in user_tokens:
        if token in search_text:
            score += 2
        if token in table_name:
            score += 4

    if any(col["name"].lower() == "geom" for col in item.get("columns", [])):
        score += 1

    if "sample" in table_name:
        score -= 3

    return score


def retrieve_relevant_tables(
    user_input: str,
    catalog: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    user_tokens = tokenize(user_input)

    scored = []
    for item in catalog:
        s = score_table(user_tokens, item)
        if s > 0:
            scored.append({
                **item,
                "score": s,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]