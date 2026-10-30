"""Check readiness and a real prediction at a local or deployed HTTP URL."""

import argparse
from http.client import HTTPException
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def check(base_url: str, attempts: int) -> None:
    """Retry cold starts; require readiness and a complete prediction path."""
    base = base_url.rstrip("/")
    last_error = None
    for attempt in range(attempts):
        try:
            with urlopen(base + "/health/ready", timeout=20) as response:
                if json.load(response)["status"] != "ready":
                    raise ValueError("Model not ready")
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
    arguments = parser.parse_args()
    check(arguments.url, arguments.attempts)
