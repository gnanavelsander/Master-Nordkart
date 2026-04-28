# LLM–GIS Agent Proof of Concept (PoC)







# Architecture
Frontend: Next.js + React + Leaflet

Backend: FastAPI

LLM orchestration: Planner + SQL builder

Execution: MCP client → PostGIS

Output: GeoJSON + table + model comparison


---

## Installation


#### Clone the project

```bash
https://github.com/gnanavelsander/Master-Nordkart.git
cd NordKart-MasterOppgave
```

#### Python dependencies
Dependencies are managed via pyproject.toml + uv.lock.
#####
Install `uv`:

```bash
pip install uv
```
Install dependencies
```bash
uv sync
```


## Development container (recommended)

Docker Engine is required.  
Install Docker Engine from: https://docs.docker.com/engine/

This project includes a VS Code Dev Container for a reproducible environment.

1. Install Docker, VS Code, and the **Dev Containers** extension
2. Open the project in VS Code
3. Select **“Reopen in Container”** when prompted

Environment variables must be provided via a `.env` file.


## Environment variables (.env)

Secrets are stored in .env.     

#### Create `.env` file

In the project root (same folder as `requirements.txt`), create a file named `.env`:

```bash
nano .env
```

Open it in your editor and add:

```env
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE-NAME.openai.azure.com
AZURE_OPENAI_KEY=YOUR_AZURE_OPENAI_KEY
PGCONN_STRING=postgresql://USERNAME:PASSWORD@HOST:5432/DATABASE
```

## Backend

```bash

cd backend
uv run uvicorn api_server:app --reload --port 8000
```





## Frontend

```bash

cd frontend
npm run dev
```
