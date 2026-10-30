"""Flask application factory: configuration, services, routes and errors."""

import json
import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from hierarchical_classifier.config import Config
from hierarchical_classifier.services.prediction import ClassifierService


def create_app(test_config: dict | None = None) -> Flask:
    """Create an app without training or making network calls."""
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
    logging.basicConfig(level=logging.INFO)
    try:
        app.extensions["classifier"] = ClassifierService(Path(app.config["MODEL_DIR"]))
    except (OSError, ValueError, KeyError) as exc:
        app.logger.error("Model unavailable: %s", type(exc).__name__)
        app.extensions["classifier"] = None
    metrics_path = Path(app.config["REPORT_DIR"]) / "metrics.json"
    app.extensions["metrics"] = (
        json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    )
    from hierarchical_classifier.web.routes.api import api
    from hierarchical_classifier.web.routes.health import health
    from hierarchical_classifier.web.routes.pages import pages

    app.register_blueprint(pages)
    app.register_blueprint(api)
    app.register_blueprint(health)

    @app.errorhandler(HTTPException)
    def http_error(error):
        if request.path.startswith(("/api/", "/health/")):
            return (
                jsonify(error={"code": error.code, "message": error.description}),
                error.code,
            )
        return (
            render_template("error.html", code=error.code, message=error.description),
            error.code,
        )

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.error("Unhandled error: %s", type(error).__name__)
        if request.path.startswith("/api/"):
            return (
                jsonify(error={"code": 500, "message": "Unexpected server error."}),
                500,
            )
        return (
            render_template("error.html", code=500, message="Unexpected server error."),
            500,
        )

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self'; script-src 'self'; "
            "img-src 'self' data:; base-uri 'self'; "
            "frame-ancestors 'none'; form-action 'self'"
        )
        if request.method == "POST":
            response.headers["Cache-Control"] = "no-store"
        return response

    return app
