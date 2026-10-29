import json

import pytest
import requests

from crawler import fandom_client
from crawler.fandom_client import FandomClient, FandomError


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code = status
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(fandom_client.time, "sleep", lambda _seconds: None)


def _client(tmp_path, responses):
    client = FandomClient(tmp_path / "cache")
    queue = list(responses)
    client._session.get = lambda url, **kwargs: queue.pop(0)
    return client


def test_fetch_perks_page_caches_response(tmp_path):
    payload = {"parse": {"text": {"*": "<p>hi</p>"}, "revid": 7}}
    client = _client(tmp_path, [FakeResponse(payload=payload)])

    assert client.fetch_perks_page() == ("<p>hi</p>", 7)

    offline = _client(tmp_path, [])
    assert offline.fetch_perks_page(use_cache=True) == ("<p>hi</p>", 7)


def test_retries_on_rate_limit_then_succeeds(tmp_path):
    payload = {"parse": {"text": {"*": "x"}, "revid": 1}}
    client = _client(tmp_path, [FakeResponse(status=429), FakeResponse(payload=payload)])
    assert client.fetch_perks_page()[1] == 1


def test_gives_up_after_max_attempts(tmp_path):
    client = _client(tmp_path, [FakeResponse(status=503)] * fandom_client.MAX_ATTEMPTS)
    with pytest.raises(FandomError, match="Giving up"):
        client.fetch_perks_page()


def test_unexpected_api_payload_raises(tmp_path):
    client = _client(tmp_path, [FakeResponse(payload={"error": {"code": "maxlag"}})])
    with pytest.raises(FandomError, match="Unexpected API response"):
        client.fetch_perks_page()


def test_download_writes_atomically(tmp_path):
    client = _client(tmp_path, [FakeResponse(content=b"png-bytes")])
    target = tmp_path / "icons" / "a.png"
    client.download("https://example.com/a.png", target)
    assert target.read_bytes() == b"png-bytes"
    assert not target.with_suffix(".png.part").exists()
