import time
import json
import hashlib
import sys
from pathlib import Path
from typing import Any

from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

src = "agent_benchmark_runstest"


def query_hash(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()[:10]


def upload(path: str, data: Any):
    blob = blob_service.get_blob_client(
        container=AZURE_STORAGE_CONTAINER,
        blob=path,
    )
    blob.upload_blob(json.dumps(data, indent=2, default=str), overwrite=True)


def query_meta_exists(q_hash: str) -> bool:
    try:
        blob_service.get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=f"{src}/{q_hash}/meta.json",
        ).get_blob_properties()
        return True
    except Exception:
        return False


def _download_json(path: str) -> Any:
    try:
        data = blob_service.get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=path,
        ).download_blob().readall()
        return json.loads(data)
    except Exception:
        return None


def _update_queries_index(q_hash: str, query: str):
    path = f"{src}/index.json"
    index = _download_json(path) or []

    if not any(e["query_hash"] == q_hash for e in index):
        index.append({
            "query_hash": q_hash,
            "query": query,
            "first_run": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_runs": 1,
        })
    else:
        for e in index:
            if e["query_hash"] == q_hash:
                e["total_runs"] += 1

    upload(path, index)


def _update_runs_index(q_hash: str, run_id: str, summary: list, best=None):
    path = f"{src}/{q_hash}/runs/index.json"
    index = _download_json(path) or []

    index.append({
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models_run": [s["model_id"] for s in summary],
    })

    upload(path, index)



    # async def run_all(
#     user_input: str,
#     model_ids: list[str],
#     params: ModelParams,
#     prompt: str,
#     max_concurrency: int = 2,
# ) -> list:
#     print(f"\nRunning {len(model_ids)} models with max_concurrency={max_concurrency}...")
#     sem = asyncio.Semaphore(max_concurrency)

#     tasks = [
#         _run_one_with_retry(user_input, m, params, prompt, sem)
#         for m in model_ids
#     ]
#     return await asyncio.gather(*tasks)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# async def main():
#     queries = [
#         "Find 10 buildings that have cultural heritage status.",
#         "Count residential buildings that are exempt from application requirements.",
#         "Count industrial buildings that are SEFRAK-registered.",
#         "Count buildings excluding those classified as ruins.",
#         "Count buildings grouped by ordinary registrations and bulk registrations.",
#     ]

#     model_ids = [
#         "gpt-35-turbo",
#         "gpt-4",
#         "gpt-4o",
#         "gpt-4o-mini",
#         "gpt-4.1",
#         "gpt-4.1-nano",
#         "gpt-5-mini",
#     ]

#     params = ModelParams(
#         temperature=0.0,
#         max_tokens=2000,
#         timeout=120,
#         max_retries=0,
#         max_iterations=17,
#         max_attempts=2,
#     )

#     prompt_name = "prompt11"
#     prompt = load_prompt(prompt_name)

#     try:
#         for q_idx, query in enumerate(queries, start=1):
#             print("\n" + "=" * 120)
#             print(f"QUERY {q_idx}/{len(queries)}: {query}")
#             print("=" * 120)

#             run = EvalRun(
#                 query=query,
#                 model_ids=model_ids,
#                 prompt_name=prompt_name,
#                 params=params,
#             )

#             results = await run_all(
#                 user_input=query,
#                 model_ids=model_ids,
#                 params=params,
#                 prompt=prompt,
#                 max_concurrency=2,
#             )

#             _print_summary(query, results)
#             _print_details(results)
#             await store_run(run, results)

#     finally:
#         await mcp_client.close()
