"""Exercise smoke-check retry behaviour without a real network or sleep."""

from io import BytesIO
import pytest
from scripts import smoke_test


def test_connection_reset_during_startup_is_retried(monkeypatch):
    responses = iter(
        [
            ConnectionResetError("startup"),
            BytesIO(b'{"status":"ready"}'),
            BytesIO(b'{"status":"predicted","categories":["a","b","c"]}'),
        ]
    )

    def fake_open(*_args, **_kwargs):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    sleeps = []
    monkeypatch.setattr(smoke_test, "urlopen", fake_open)
    monkeypatch.setattr(smoke_test.time, "sleep", sleeps.append)
    smoke_test.check("http://example.test", 2)
    assert sleeps == [10]


def test_persistent_connection_failure_is_not_reported_as_success(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ConnectionResetError("still starting")

    monkeypatch.setattr(smoke_test, "urlopen", fail)
    with pytest.raises(SystemExit, match="failed after 1"):
        smoke_test.check("http://example.test", 1)
