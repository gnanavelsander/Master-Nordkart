from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Optional
from tools.validate_sql import db
from Client.AzureChatOpenAI import run_all



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("App starting...")
    yield
    print("App shutting down...")
    db.disconnect()


app = FastAPI(lifespan=lifespan)




app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"ok": True}




@app.post("/query")
async def run(payload: dict[str, Any]):
    question = payload["question"]
    models = payload["models"]
    mode = payload.get("mode", "parallel")
    aoi: Optional[dict[str, Any]] = payload.get("aoi")

    
    if mode == "sequential":
        results = []
        for m in models:
            r = (await run_all(question, [m]))[0]
            results.append(r)
    else:
        results = await run_all(question, models)   

    # filtrer exceptions til “errors”
    clean = []
    for r, m in zip(results, models):
        if isinstance(r, Exception):
            clean.append({"model": m, "sql": "", "accuracy": 0.0, "errors": [str(r)]})
        else:
            clean.append({"model": r["model_id"], "sql": r["sql"], "accuracy": 0.0, "errors": []})

    return {"best_model": clean[0]["model"], "results": clean}



