"""Versioned JSON endpoints sharing the same services as browser forms."""

from flask import Blueprint, current_app, jsonify, request
from werkzeug.exceptions import BadRequest

from hierarchical_classifier.services.batch import classify_csv
from hierarchical_classifier.web.validation import classifier, text_payload

api = Blueprint("api", __name__, url_prefix="/api/v1")


@api.post("/predict")
def predict():
    """Classify a JSON Title/Text pair."""
    if not request.is_json:
        raise BadRequest("Use Content-Type: application/json.")
    title, text = text_payload(request.get_json())
    return jsonify(classifier().predict(title, text))


@api.post("/predict-batch")
def predict_batch():
    """Accept a bounded multipart CSV as the file field."""
    upload = request.files.get("file")
    if upload is None:
        raise BadRequest("Upload a CSV using the multipart field file.")
    try:
        rows = classify_csv(upload.read(), classifier())
    except ValueError as exc:
        raise BadRequest(str(exc)) from exc
    return jsonify(rows=rows, row_count=len(rows))


@api.get("/taxonomy")
def taxonomy():
    """List supported training paths and row counts."""
    return jsonify(paths=classifier().taxonomy)


@api.get("/metrics")
def metrics():
    """Return the recorded fixed-holdout report, not live production accuracy."""
    return jsonify(current_app.extensions["metrics"])
