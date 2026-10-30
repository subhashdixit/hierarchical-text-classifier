"""Check readiness and a real prediction at a local or deployed HTTP URL."""

import argparse
from http.client import HTTPException
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def check(base_url: str, attempts: int, expected_revision: str | None = None) -> None:
    """Retry cold starts; require readiness and a complete prediction path."""
    base = base_url.rstrip("/")
    last_error = None
    for attempt in range(attempts):
        try:
            with urlopen(base + "/health/ready", timeout=20) as response:
                readiness = json.load(response)
                if readiness["status"] != "ready":
                    raise ValueError("Model not ready")
                if expected_revision and readiness.get("revision") != expected_revision:
                    raise ValueError("The expected source revision is not deployed yet")
            payload = json.dumps(
                {"Title": "Dog toy", "Text": "A rubber chew toy for my dog"}
            ).encode()
            request = Request(
                base + "/api/v1/predict",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=20) as response:
                result = json.load(response)
            if result["status"] != "predicted" or len(result["categories"]) != 3:
                raise ValueError("Unexpected prediction contract")
            print("Readiness and real HTTP prediction passed:", base)
            return
        except (
            HTTPError,
            URLError,
            TimeoutError,
            ValueError,
            ConnectionError,
            HTTPException,
        ) as exc:
            last_error = type(exc).__name__
            if attempt + 1 < attempts:
                time.sleep(10)
    raise SystemExit(f"Smoke test failed after {attempts} attempts ({last_error}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--attempts", type=int, default=12)
    parser.add_argument("--expected-revision", default=None)
    arguments = parser.parse_args()
    check(arguments.url, arguments.attempts, arguments.expected_revision)
