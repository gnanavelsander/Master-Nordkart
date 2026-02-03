import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_c_s.mcp_client import query_postgis


async def main():
    print("PostGIS SQL terminal")
    print("Write multi-line SQL. Execute when you end with ';'")
    print("Type 'exit' to quit.\n")

    buf: list[str] = []

    while True:
        prompt = "sql> " if not buf else "...> "
        line = input(prompt)

        if not buf and line.strip().lower() in {"exit", "quit"}:
            print("Bye 👋")
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


if __name__ == "__main__":
    asyncio.run(main())
