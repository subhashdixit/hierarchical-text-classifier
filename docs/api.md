# API contract

Base URL: a running local or deployed application. No authentication is implemented for this public demo. Inputs are bounded and not persisted.

| Method/path | Input | Output |
|---|---|---|
| POST `/api/v1/predict` | JSON object, optional string Title/Text with at least one nonblank | Predicted path, ranking score, top alternatives, review flags, model version |
| POST `/api/v1/predict-batch` | multipart `file`, UTF-8 CSV with Title/Text | `row_count` and row-level `rows` |
| GET `/api/v1/taxonomy` | none | Training-supported paths/counts |
| GET `/api/v1/metrics` | none | Recorded evaluation report |
| GET `/health/live` | none | Process liveness |
| GET `/health/ready` | none | Model readiness, model version and deployment revision |

`categories` is a three-string list or null for unsupported vocabulary. `path_score` is a normalized ranking score or null; it is not calibrated confidence. A valid but unsupported request returns 200 with `status: unsupported`. A model-related operational failure returns 503.

Error format: `{"error":{"code":400,"message":"..."}}`. Other codes: 404, 405, 413, 500, 503. See README for tested curl examples and CSV limits. API batch returns original Title/Text for valid and invalid rows; other input fields are ignored.
