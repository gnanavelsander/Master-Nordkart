# Prototype interface

This Next.js / React interface connects natural-language questions to the GIS agent and displays answers, map features, result tables, SQL, and execution logs. Start with the [project overview and recorded demo](../README.md).

## Reading guide

| File | What it explains |
| --- | --- |
| [src/app/page.tsx](src/app/page.tsx) | Question submission, model and prompt selection, context, and result inspection. |
| [MapClient.tsx](src/app/components/MapClient.tsx) and [MapView.tsx](src/app/components/MapView.tsx) | Client-side Leaflet map, GeoJSON, drawn areas of interest, and feature selection. |
| [ResultTable.tsx](src/app/ResultTable.tsx) | Tabular result display. |
| [backend/app.py](../backend/app.py) | FastAPI endpoints and conversion of agent results to the UI response. |

The page sends `POST /query` with the question, selected models and prompt, and optional map/conversation context. It also uses `GET /models` and `GET` / `POST /prompt/{name}`. The selected prompt defaults to `prompt_v4`; saving a prompt writes to the backend's prompt file.

The map loads through `MapClient.tsx` to keep Leaflet in the browser. Geometry results can be shown on the map; tabular results can be inspected in the table. SQL and log panels expose intermediate work. Execution status indicators do not independently verify answer correctness.

## Development context

The interface was bootstrapped with `create-next-app`. Scripts are in [package.json](package.json), with dependencies recorded in [package-lock.json](package-lock.json).

For a developer who already has the backend and required data/model access, `npm ci` followed by `npm run dev` starts the frontend on port 3000. Starting the interface alone does not make analysis available.

`page.tsx` checks `NEXT_PUBLIC_EVALUATION_API_BASE`, then `NEXT_PUBLIC_ASSISTANT_API_BASE`, and also tries backend addresses on port 8000 for the browser host and localhost. These are public browser configuration values, not places for credentials. See the [runtime notes](../docs/runtime-context.md) for the original environment and its limitations.
