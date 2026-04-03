import asyncio
import sys
import os

from dotenv import load_dotenv
from pathlib import Path

from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parents[2] / ".env"
loaded = load_dotenv(env_path, override=True)

if not loaded:
    raise RuntimeError(f".env not found at {env_path}")

if not os.getenv("PGCONN_STRING"):
    raise RuntimeError("PGCONN_STRING missing after loading .env")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.mcp_c_s.mcp_client import query_postgis, run_multidb_plan


async def main():




    plan = {
    "query_mode": "multi_db",
    "steps": [
        {
            "step": 1,
            "action": "query",
            "database": "matrikkelenbygning",
            "sql": "SELECT kommunenavn, ST_AsBinary(ST_Transform(representasjonspunkt, 25833)) AS representasjonspunkt FROM matrikkelenbygning_fe504ac373564e10b85865ef2a507c63.bygning WHERE representasjonspunkt IS NOT NULL"
        },
        {
            "step": 2,
            "action": "query",
            "database": "flomsoner",
            "sql": "SELECT ST_AsBinary(ST_Transform(omrade, 25833)) AS omrade FROM flomsoner_76d14d5fe2fd4a29bfffc266fdcdd820.flomareal WHERE omrade IS NOT NULL"
        },
        {
            "step": 3,
            "action": "spatial_join",
            "left_source": 1,
            "right_source": 2,
            "join_type": "intersects",
            "left_geom": "representasjonspunkt",
            "right_geom": "omrade"
        },
        {
            "step": 4,
            "action": "aggregate",
            "source_step": 3,
            "group_by": ["kommunenavn"],
            "metric_as": "buildings_in_flom",
            "order_by": "buildings_in_flom",
            "order_direction": "desc",
            "limit": 5
        }
    ]
}
    row = await run_multidb_plan(plan)
    print(row)





if __name__ == "__main__":
    asyncio.run(main())
