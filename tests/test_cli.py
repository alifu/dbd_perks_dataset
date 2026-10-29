import json

import pytest

from crawler import __main__ as cli
from crawler.fandom_client import FandomError


@pytest.fixture()
def stub_fetch(monkeypatch, sample_html):
    state = {"revid": 100}
    monkeypatch.setattr(
        cli.FandomClient, "fetch_perks_page", lambda self, use_cache=False: (sample_html, state["revid"])
    )
    return state


def test_run_writes_perks_json(tmp_path, stub_fetch):
    assert cli.run(["--out", str(tmp_path), "--cache", str(tmp_path / "c"), "--skip-icons"]) == 0
    document = json.loads((tmp_path / "perks.json").read_text(encoding="utf-8"))
    assert len(document["perks"]) == 5 and document["wikiRevisionId"] == 100


def test_unchanged_revision_is_not_rewritten(tmp_path, stub_fetch):
    args = ["--out", str(tmp_path), "--cache", str(tmp_path / "c"), "--skip-icons"]
    cli.run(args)
    first = (tmp_path / "perks.json").read_text(encoding="utf-8")
    cli.run(args)
    assert (tmp_path / "perks.json").read_text(encoding="utf-8") == first


def test_icons_download_only_missing(tmp_path, stub_fetch, monkeypatch):
    downloaded = []
    monkeypatch.setattr(cli.FandomClient, "download", lambda self, url, dest: (downloaded.append(dest.name), dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(b"x")))
    args = ["--out", str(tmp_path), "--cache", str(tmp_path / "c")]
    cli.run(args)
    assert len(downloaded) == 5
    cli.run(args + ["--force"])
    assert len(downloaded) == 5


def test_icon_failures_are_reported(tmp_path, stub_fetch, monkeypatch):
    def boom(self, url, dest):
        raise FandomError("nope")

    monkeypatch.setattr(cli.FandomClient, "download", boom)
    with pytest.raises(FandomError, match="5 icons failed"):
        cli.run(["--out", str(tmp_path), "--cache", str(tmp_path / "c")])
