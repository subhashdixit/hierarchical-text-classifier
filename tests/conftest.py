"""Shared real-artifact app and test fixtures."""

import pytest
from hierarchical_classifier.web import create_app


@pytest.fixture(scope="session")
def app():
    """Use the checked-in approved model, not a mocked prediction service."""
    return create_app({"TESTING": True})


@pytest.fixture()
def client(app):
    """Isolate HTTP state for each test."""
    return app.test_client()
