# Test script
import asyncio

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))




from backend.database import PoolManager
from config import PGCONN_FLOMSONER

async def test():
    pm = PoolManager()
    conn_id = await pm.connect(PGCONN_FLOMSONER)
    result = await pm.query(conn_id, "SELECT COUNT(*) FROM flomsoner_76d14d5fe2fd4a29bfffc266fdcdd820.flomareal")
    print(result)
    result2 = await pm.query(conn_id, "SELECT objid, omrade FROM flomsoner_76d14d5fe2fd4a29bfffc266fdcdd820.flomareal LIMIT 10")
    print(f"Got {len(result2)} rows")
    await pm.close_all()

asyncio.run(test())