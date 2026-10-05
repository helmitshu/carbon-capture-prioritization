# Prediction API

Serves the registered capstone model over HTTP. The Streamlit app calls
this instead of loading the model file in-process.

## Run locally

    uvicorn api.main:app --host 0.0.0.0 --port 8000

`MODEL_VERSION` env var selects the registry version (default `v2`).

## Railway deploy

New service from the `v2-mlops` branch in the existing project:

- Build: `pip install -r api/requirements.txt` (Nixpacks handles it)
- Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Env vars: `MODEL_VERSION=v2` (optional)

The Streamlit app service needs one env var:

- `PREDICT_API_URL` = the API service's public URL

If the API is unreachable the app falls back to its bundled model file,
so the app service works with or without this set.

## Prediction log

Every successful prediction is appended to `logs/predictions.jsonl`
(timestamp, model version, inputs, predictions, leaf ids, latency).
Override with `PREDICT_LOG_PATH`. Railway's disk is ephemeral, so attach
a volume or ship the lines elsewhere if the log must survive redeploys.
See `RETRAIN_POLICY.md` for how this log feeds the drift trigger.
