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


from backend.mcp_c_s.mcp_client import query_postgis


async def main():
    print("PostGIS SQL terminal")
    print("Write multi-line SQL. Execute when you end with ';'")
    print("Type 'exit' to quit.\n")

    buf: list[str] = []

    while True:
        prompt = "sql> " if not buf else "...> "
        line = input(prompt)

        if not buf and line.strip().lower() in {"exit", "quit"}:
            print("Bye ")
            break

        # legg til linje i buffer
        buf.append(line)

        # kjør først når siste linje ender med ;
        if line.strip().endswith(";"):
            sql = "\n".join(buf).strip()
            buf.clear()

            try:
                df = await query_postgis(sql)
                print(df)
                print(f"\nRows: {len(df)} | Empty: {df.empty}\n")
            except Exception as e:
                print(f"ERROR: {e}\n")





import asyncpg
import asyncio
import os

async def test():
    conn = await asyncpg.connect(os.getenv("PGCONN_MATRIKKEL"))
    result = await conn.fetchval("SELECT COUNT(*) FROM matrikkelenbygning_fe504ac373564e10b85865ef2a507c63.bygning")
    print(f"Bygninger: {result}")
    await conn.close()

asyncio.run(test())




if __name__ == "__main__":
    asyncio.run(test())
