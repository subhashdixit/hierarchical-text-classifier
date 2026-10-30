# Architecture and decisions

`create_app()` loads configuration, initializes one ClassifierService per worker, registers three blueprints and configures error/security headers. Browser routes return escaped Jinja templates; the API returns JSON; CSV service errors remain row-specific.

Request flow: route → validation → shared service → training-identical preprocessing → TF-IDF → constrained classifier → response. Training flow is separate and never invoked by a request.

The class owns loaded model state. Pure functions handle transformations and metrics. Models and reports are located from APP_ROOT/MODEL_DIR, not the process working directory. The wheel contains Python modules, templates and static assets; Docker separately copies models/reports and sets their paths.

The hierarchy model was deliberately simplified from parent-label stacking to avoid inconsistent encoded/decoded parent features and in-sample stacking leakage. Predictions score only training-observed paths. A connected product/text group split gives a harder unseen-product test than a row-level random split.

No DB is necessary because there are no persistent user records. A separate API microservice would add deployment complexity without a current independent consumer; Flask exposes JSON endpoints directly. Containerized Gunicorn and Render-native CI gating provide a small deployable system without Kubernetes.
