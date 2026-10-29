from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_html() -> str:
    return (FIXTURES / "perks_page_sample.html").read_text(encoding="utf-8")
