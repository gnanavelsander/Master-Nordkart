"""
For each query/prompt combo in run_3/, keep only the newest run and delete the rest.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER

PREFIX = "run_3/"


def main():
    service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container = service.get_container_client(AZURE_STORAGE_CONTAINER)

    runs_indexes = [
        b.name for b in container.list_blobs(name_starts_with=PREFIX)
        if b.name.endswith("runs/index.json")
    ]

    total_deleted = 0

    for idx_path in runs_indexes:
        data = json.loads(container.get_blob_client(idx_path).download_blob().readall())

        if len(data) <= 1:
            continue

        # Sort by timestamp, keep newest
        data.sort(key=lambda r: r["timestamp"])
        to_delete = data[:-1]
        keep = data[-1]

        runs_prefix = idx_path.replace("runs/index.json", "runs/")

        for run in to_delete:
            run_id = run["run_id"]
            blobs_to_del = [
                b.name for b in container.list_blobs(name_starts_with=f"{runs_prefix}{run_id}/")
            ]
            for blob_name in blobs_to_del:
                container.get_blob_client(blob_name).delete_blob()
                print(f"  Slettet {blob_name}")
                total_deleted += 1

        # Update runs/index.json to only contain the newest run
        container.get_blob_client(idx_path).upload_blob(
            json.dumps([keep], indent=2), overwrite=True
        )
        print(f"Oppdatert {idx_path} — beholdt run {keep['run_id']} ({keep['timestamp']})")

    print(f"\nFerdig. {total_deleted} blobs slettet.")


if __name__ == "__main__":
    main()
