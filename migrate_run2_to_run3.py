"""
One-shot script: copy today's blobs from run_2/ -> run_3/ in Azure Blob Storage,
then delete the originals from run_2/.
"""
import sys
from datetime import date, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from azure.storage.blob import BlobServiceClient
from config import AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER

TODAY = date.today()
SRC_PREFIX = "run_2/"
DST_PREFIX = "run_3/"


def main():
    service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container = service.get_container_client(AZURE_STORAGE_CONTAINER)

    blobs_today = [
        b for b in container.list_blobs(name_starts_with=SRC_PREFIX)
        if b.last_modified.astimezone(timezone.utc).date() == TODAY
    ]

    if not blobs_today:
        print(f"Ingen blobs fra i dag ({TODAY}) funnet under {SRC_PREFIX}")
        return

    print(f"Fant {len(blobs_today)} blob(s) fra i dag — kopierer til {DST_PREFIX}...")

    copied, failed = [], []
    for blob in blobs_today:
        src_name = blob.name
        dst_name = DST_PREFIX + src_name[len(SRC_PREFIX):]

        src_url = container.get_blob_client(src_name).url
        dst_client = container.get_blob_client(dst_name)
        try:
            poller = dst_client.start_copy_from_url(src_url)
            # wait for copy to finish
            props = dst_client.get_blob_properties()
            while props.copy.status == "pending":
                props = dst_client.get_blob_properties()
            if props.copy.status != "success":
                raise RuntimeError(f"Copy status: {props.copy.status}")
            copied.append(src_name)
            print(f"  OK  {src_name} -> {dst_name}")
        except Exception as e:
            failed.append(src_name)
            print(f"  FEIL {src_name}: {e}")

    if failed:
        print(f"\n{len(failed)} feil — sletter IKKE originaler. Sjekk feilene over.")
        sys.exit(1)

    print(f"\nAlle {len(copied)} blobs kopiert. Sletter fra {SRC_PREFIX}...")
    for src_name in copied:
        container.get_blob_client(src_name).delete_blob()
        print(f"  Slettet {src_name}")

    print("\nFerdig.")


if __name__ == "__main__":
    main()
