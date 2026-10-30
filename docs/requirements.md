# Requirements and acceptance criteria

The application predicts three-level categories for English product-related text. Single HTML/JSON and bounded CSV input share one service. At least one text field must be nonblank. Training remains offline.

Acceptance checks map to `tests/`: valid paths and deterministic preprocessing (unit), real-artifact inference and save/load training (integration), API/HTML agreement, unsupported input and model readiness failure (web integration). Docker CI verifies HTTP readiness and a real prediction on Linux. Manual browser checks cover example buttons, visible results and navigation.

Serving has no user history, accounts, database or training endpoint. The model cannot create unsupported categories. A complete path is a suggestion, not an automatically approved business decision.

The public deployment must use a production WSGI server, HTTPS provided by the host, bounded resources, no debug mode and explicit health checks. See the README for numerical limits and known gaps.
