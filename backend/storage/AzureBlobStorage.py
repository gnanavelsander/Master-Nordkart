import time
import json
import hashlib
import sys
from pathlib import Path
from typing import Any

from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER, STORAGE_VERSION

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_blob_service: BlobServiceClient | None = None

src = STORAGE_VERSION


def _get_blob_service() -> BlobServiceClient:
    global _blob_service
    if _blob_service is None:
        if not AZURE_STORAGE_CONNECTION_STRING:
            raise RuntimeError(
                "AZURE_STORAGE_CONNECTION_STRING is not set. "
                "Cannot connect to Azure Blob Storage."
            )
        _blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    return _blob_service


def query_hash(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()[:10]


def upload(path: str, data: Any):
    blob = _get_blob_service().get_blob_client(
        container=AZURE_STORAGE_CONTAINER,
        blob=path,
    )
    blob.upload_blob(json.dumps(data, indent=2, default=str), overwrite=True)


def query_meta_exists(q_hash: str) -> bool:
    try:
        _get_blob_service().get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=f"{src}/{q_hash}/query.json",
        ).get_blob_properties()
        return True
    except Exception:
        return False


def _download_json(path: str) -> Any:
    try:
        data = _get_blob_service().get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=path,
        ).download_blob().readall()
        return json.loads(data)
    except Exception:
        return None


def _update_queries_index(q_hash: str, query: str, query_id: str | None = None):
    path = f"{src}/index.json"
    index = _download_json(path) or []

    if not any(e["query_hash"] == q_hash for e in index):
        index.append({
            "query_id": query_id,
            "query_hash": q_hash,
            "query": query,
            "first_run": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_runs": 1,
        })
    else:
        for e in index:
            if e["query_hash"] == q_hash:
                e["total_runs"] += 1
                if query_id and not e.get("query_id"):
                    e["query_id"] = query_id

    upload(path, index)


def _update_runs_index(q_hash: str, run_id: str, summary: list):
    path = f"{src}/{q_hash}/runs/index.json"
    index = _download_json(path) or []

    index.append({
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models_run": [s["model_id"] for s in summary],
    })

    upload(path, index)

