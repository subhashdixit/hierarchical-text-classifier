# Deployment runbook

The README contains the complete first-deploy steps. Render runs the tested Docker image definition with Gunicorn and /health/ready. Free tier only; no DB or disk. Connect via Git Provider so CI-gated automatic deployments work; a public Git URL alone does not provide the same integration.

After main CI passes, select After CI Checks Pass in Render. Confirm HTTPS, readiness, revision and a real API prediction. Set GitHub variable APP_URL and run Verify live deployment. The manual verifier is separate from the CI gate to avoid a circular dependency.

Inspect Render build logs for dependency/package issues, runtime logs for model readiness failures, and Events for the deployed commit. Application logs omit request bodies and query strings; hosting metadata is governed by the hosting provider. Resource sleep and throttling are expected on a free plan.

Rollback by reverting the complete code/model/dependency release through a PR. A dashboard rollback can restore service temporarily, but reconcile GitHub afterwards. No automatic rollback based on prediction quality is implemented.

Check /health/live versus /health/ready to distinguish a responding process from a usable model. A 503 readiness response should block traffic promotion. Smoke checks test an example, not overall live accuracy.
