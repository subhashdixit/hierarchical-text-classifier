"""Separate process liveness from model readiness."""

import os

from flask import Blueprint, current_app, jsonify

health = Blueprint("health", __name__, url_prefix="/health")


@health.get("/live")
def live():
    """Confirm that the HTTP process responds."""
    return jsonify(status="alive")


@health.get("/ready")
def ready():
    """Only signal readiness when the approved model was loaded."""
    service = current_app.extensions.get("classifier")
    if service is None:
        return jsonify(status="not_ready"), 503
    return jsonify(
        status="ready",
        model_version=service.metadata["model_version"],
        revision=os.getenv("RENDER_GIT_COMMIT", "local"),
    )
