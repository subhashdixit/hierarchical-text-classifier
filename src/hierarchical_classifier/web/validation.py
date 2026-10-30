"""Validate requests independently of the transport and UI."""

from flask import current_app
from werkzeug.exceptions import BadRequest, ServiceUnavailable


def classifier():
    """Return the loaded service or an explicit readiness error."""
    service = current_app.extensions.get("classifier")
    if service is None:
        raise ServiceUnavailable("Model unavailable; check server readiness and logs.")
    return service


def text_payload(payload) -> tuple[str, str]:
    """Accept a JSON object with string Title/Text values, at least one nonblank."""
    if not isinstance(payload, dict):
        raise BadRequest("Send a JSON object with Title and Text fields.")
    title, text = payload.get("Title", ""), payload.get("Text", "")
    if not isinstance(title, str) or not isinstance(text, str):
        raise BadRequest("Title and Text must be strings.")
    if not title.strip() and not text.strip():
        raise BadRequest("Enter a title or text before classifying.")
    if len(title) + len(text) > current_app.config["MAX_TEXT_LENGTH"]:
        raise BadRequest("Combined title and text must not exceed 10000 characters.")
    return title, text
