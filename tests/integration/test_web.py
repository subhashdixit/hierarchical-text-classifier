"""HTTP contracts and failure behaviour for the real Flask application."""

import io
import pytest
from hierarchical_classifier.web import create_app


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/batch",
        "/categories",
        "/categories?category=beauty",
        "/performance",
        "/about",
        "/sample.csv",
        "/api/v1/taxonomy",
        "/api/v1/metrics",
        "/health/live",
        "/health/ready",
    ],
)
def test_pages_and_endpoints_respond(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_api_and_html_predictions_match(client):
    payload = {"Title": "Dog toy", "Text": "A durable rubber toy for my dog"}
    prediction = client.post("/api/v1/predict", json=payload)
    page = client.post("/", data=payload)
    assert prediction.status_code == page.status_code == 200
    for category in prediction.json["categories"]:
        assert category.encode() in page.data


@pytest.mark.parametrize(
    "payload",
    [{}, {"Text": 12}, {"Title": None}, ["not-an-object"], {"Text": "a" * 10001}],
)
def test_invalid_json_payloads(client, payload):
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 400
    assert "error" in response.json


def test_non_json_and_malformed_json(client):
    assert client.post("/api/v1/predict", data="hello").status_code == 400
    assert (
        client.post(
            "/api/v1/predict", data="{", content_type="application/json"
        ).status_code
        == 400
    )


def test_batch_api_and_browser_download(client):
    content = b"Title,Text\nDog toy,A rubber toy for dogs\n"
    api = client.post(
        "/api/v1/predict-batch", data={"file": (io.BytesIO(content), "batch.csv")}
    )
    page = client.post("/batch", data={"file": (io.BytesIO(content), "batch.csv")})
    assert api.status_code == page.status_code == 200
    assert api.json["row_count"] == 1
    assert "attachment" in page.headers["Content-Disposition"]
    assert b"Cat1" in page.data


def test_missing_and_oversized_uploads(client):
    assert client.post("/api/v1/predict-batch").status_code == 400
    response = client.post(
        "/api/v1/predict-batch",
        data={"file": (io.BytesIO(b"x" * (2 * 1024 * 1024)), "big.csv")},
    )
    assert response.status_code == 413


def test_model_readiness_failure_and_liveness_are_distinct(tmp_path):
    app = create_app({"TESTING": True, "MODEL_DIR": tmp_path})
    client = app.test_client()
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    assert client.post("/api/v1/predict", json={"Text": "dog food"}).status_code == 503


def test_user_markup_is_escaped(client):
    response = client.post(
        "/", data={"Title": '<script>alert("test")</script>', "Text": "dog food"}
    )
    assert b'<script>alert("test")</script>' not in response.data
    assert b"&lt;script&gt;" in response.data


def test_missing_routes_and_disallowed_methods(client):
    assert client.get("/missing").status_code == 404
    assert client.get("/api/v1/missing").json["error"]["code"] == 404
    assert client.get("/api/v1/predict").status_code == 405
