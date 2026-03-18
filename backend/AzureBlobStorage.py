import time

from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER
import json, hashlib, sys
from pathlib import Path
from typing import Optional, Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)


def query_hash(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()[:10]


def upload(path: str, data: dict):
    blob = blob_service.get_blob_client(
        container=AZURE_STORAGE_CONTAINER,
        blob=path
    )
    blob.upload_blob(json.dumps(data, indent=2, default=str), overwrite=True)


def query_meta_exists(q_hash: str) -> bool:
    try:
        blob_service.get_blob_client(
            container=AZURE_STORAGE_CONTAINER,
            blob=f"queries/{q_hash}/meta.json"
        ).get_blob_properties()
        return True
    except Exception:
        return False
    



def _download_json(path: str) -> Any:
    try:
        data = blob_service.get_blob_client(
            container=AZURE_STORAGE_CONTAINER, blob=path
        ).download_blob().readall()
        return json.loads(data)
    except Exception:
        return None


def _update_queries_index(q_hash: str, query: str):
    index = _download_json("queries/index.json") or []
    if not any(e["query_hash"] == q_hash for e in index):
        index.append({
            "query_hash": q_hash,
            "query":      query,
            "first_run":  time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_runs": 1,
        })
    else:
        for e in index:
            if e["query_hash"] == q_hash:
                e["total_runs"] += 1
    upload("queries/index.json", index)


def _update_runs_index(q_hash: str, run_id: str, summary: list, best):
    path  = f"queries/{q_hash}/runs/index.json"
    index = _download_json(path) or []
    index.append({
        "run_id":      run_id,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models_run":  [s["model_id"] for s in summary],
        "all_valid":   all(s["valid"] for s in summary),
    })
    upload(path, index)