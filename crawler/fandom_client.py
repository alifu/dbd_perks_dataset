import json
import time
from pathlib import Path

import requests

API_URL = "https://deadbydaylight.fandom.com/api.php"
USER_AGENT = "DBDPerkDatasetBot/2.0 (educational project; https://github.com/alifu/dbd_perks_dataset)"
REQUEST_TIMEOUT_SECONDS = 20
MIN_SECONDS_BETWEEN_REQUESTS = 1.0
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2.0
MAX_LAG_SECONDS = 5


class FandomError(RuntimeError):
    pass


class FandomClient:
    """Polite MediaWiki client: identifiable UA, maxlag, throttling, retries."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _get(self, url: str, **kwargs) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            self._throttle()
            try:
                response = self._session.get(url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
                if response.status_code in (429, 503):
                    raise FandomError(f"HTTP {response.status_code} from {url}")
                response.raise_for_status()
                return response
            except (requests.RequestException, FandomError) as error:
                last_error = error
                time.sleep(BACKOFF_BASE_SECONDS * 2**attempt)
        raise FandomError(f"Giving up on {url}: {last_error}")

    def fetch_perks_page(self, use_cache: bool = False) -> tuple[str, int]:
        """Return (rendered HTML of the Perks page, wiki revision id)."""
        cache_file = self._cache_dir / "Perks.json"
        if use_cache and cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            params = {
                "action": "parse",
                "page": "Perks",
                "prop": "text|revid",
                "format": "json",
                "maxlag": MAX_LAG_SECONDS,
            }
            payload = self._get(API_URL, params=params).json()
            if "parse" not in payload:
                raise FandomError(f"Unexpected API response: {str(payload)[:200]}")
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        parsed = payload["parse"]
        return parsed["text"]["*"], int(parsed["revid"])

    def download(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(self._get(url).content)
        temporary.replace(destination)
