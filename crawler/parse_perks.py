import copy
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from .normalize import (
    clean_lines,
    clean_line,
    extract_patch_version,
    slugify,
    split_flavor,
)

WIKI_ORIGIN = "https://deadbydaylight.fandom.com"
ROLE_BY_HEADING = {"survivor": "survivor", "killer": "killer"}
CELLS_PER_ROW = 4  # icon, name, description, character
_NOISE_SELECTORS = ["img", ".tooltiptext", ".mobileView", ".display-none"]
_BLOCK_TAGS = ["p", "div", "ul", "ol"]


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class RawPerk:
    id: str
    name: str
    role: str
    owner: str
    icon_file: str
    icon_url: str
    description: str
    flavor: str
    description_patch: str | None
    wiki_url: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "owner": self.owner,
            "iconFile": self.icon_file,
            "iconUrl": self.icon_url,
            "description": self.description,
            "flavor": self.flavor,
            "descriptionPatch": self.description_patch,
            "wikiUrl": self.wiki_url,
        }


def _role_for_table(table: Tag) -> str:
    heading = table.find_previous(["h2", "h3"])
    text = heading.get_text(" ", strip=True).lower() if heading else ""
    for keyword, role in ROLE_BY_HEADING.items():
        if text.startswith(keyword):
            return role
    raise ParseError(f"Cannot tell survivor/killer from heading: {text!r}")


def _description_text(cell: Tag) -> tuple[list[str], str | None]:
    """Return (cleaned lines, patch version of the 'based on Patch X' banner)."""
    cell = copy.copy(cell)
    patch = None
    for banner in cell.select(".dynamicTitle"):
        patch = extract_patch_version(banner.get_text(" ", strip=True))
        banner.decompose()
    for selector in _NOISE_SELECTORS:
        for tag in cell.select(selector):
            tag.decompose()
    for br in cell.find_all("br"):
        br.replace_with("\n")
    for li in cell.find_all("li"):
        li.insert_before("\n• ")
    for block in cell.find_all(_BLOCK_TAGS):
        block.insert_before("\n")
        block.insert_after("\n")
    return clean_lines(cell.get_text()), patch


def _icon(cell: Tag) -> tuple[str, str]:
    image = cell.find("img")
    link = cell.select_one("a.mw-file-description")
    if image is None or link is None or not image.get("data-image-key"):
        raise ParseError("Row has no icon image")
    return image["data-image-key"], link["href"]


def _owner(cell: Tag) -> str:
    titles = [a["title"] for a in cell.find_all("a") if a.get("title")]
    return clean_line(titles[-1]) if titles else ""


def _parse_row(cells: list[Tag], role: str) -> RawPerk:
    icon_file, icon_url = _icon(cells[0])
    name_link = cells[1].find("a")
    name = clean_line(cells[1].get_text())
    if not name:
        raise ParseError("Row has no perk name")
    lines, patch = _description_text(cells[2])
    body, flavor = split_flavor(lines)
    return RawPerk(
        id=slugify(name),
        name=name,
        role=role,
        owner=_owner(cells[3]),
        icon_file=icon_file,
        icon_url=icon_url,
        description="\n".join(body),
        flavor=flavor,
        description_patch=patch,
        wiki_url=WIKI_ORIGIN + name_link["href"] if name_link and name_link.get("href") else "",
    )


def parse_perks_page(html: str) -> tuple[list[RawPerk], list[str]]:
    """Parse the wiki 'Perks' page. Returns (perks, per-row error messages)."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select("table.wikitable.sortable")
    if not tables:
        raise ParseError("No perk tables found; the wiki layout may have changed")

    perks: list[RawPerk] = []
    errors: list[str] = []
    for table in tables:
        role = _role_for_table(table)
        for row in table.select("tr")[1:]:
            cells = row.find_all(["th", "td"], recursive=False)
            if len(cells) != CELLS_PER_ROW:
                errors.append(f"{role}: unexpected cell count {len(cells)}")
                continue
            try:
                perks.append(_parse_row(cells, role))
            except ParseError as error:
                errors.append(f"{role}: {error}")
    return perks, errors
