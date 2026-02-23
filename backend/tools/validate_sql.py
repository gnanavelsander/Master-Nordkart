from dataclasses import dataclass
from typing import Optional 

import re


from langchain_core.tools import tool
from pathlib import Path
import sys,os 

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from database.database import Database


# Validation result container
@dataclass
class ValidationResult:
    ok: bool
    error: Optional[str] = None
    hint: Optional[str] = None




# Forbidden SQL keywords
_FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE)\b", re.IGNORECASE)




async def validate_sql_explain(conn, sql: str) -> ValidationResult:
    
    # Validates SQL by running EXPLAIN  in Postgres.
    
    s = (sql or "").strip().rstrip(";").strip()

    # Reject empty queries
    if not s:
        return ValidationResult(ok=False, error="Empty SQL")

    #  Reject multiple statements 
    if ";" in s:
        return ValidationResult(ok=False, error="Multiple statements are not allowed")

    # check if head start with selectt or with
    head = s[:20].lstrip().upper()
    if not (head.startswith("SELECT") or head.startswith("WITH")):
        return ValidationResult(ok=False, error="Only SELECT/WITH queries are allowed")


  
    # Allow only SELECT or WITH 
    if _FORBIDDEN.search(s):
        return ValidationResult(ok=False, error="Forbidden keyword detected")

    # Execute EXPLAIN to validate
    try:
        cur = conn.cursor()
        await cur.execute("EXPLAIN (FORMAT JSON) " + s)
        _ = await cur.fetchone()
        return ValidationResult(ok=True)

    # return truncated error message for LLM repair loop
    except Exception as e:
        # keep it short for LLM repair
        return ValidationResult(ok=False, error=str(e)[:600])
    



