import os, requests, json, time, random, re
import json
from bs4 import BeautifulSoup

def get_description():
    data = requests.get(
        "https://deadbydaylight.fandom.com/api.php?action=parse&page=Perks&prop=text&format=json"
    ).json()

    html = data["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")

    perks = []

    # Each perk is inside a table row <tr> under .wikitable
    for row in soup.select("table.wikitable.sortable tbody tr"):
        cols = row.find_all("th") + row.find_all("td")      

        try:
            # Perk name & link
            perk_link_tag = cols[1].find("a")
            perk_name = perk_link_tag.text.strip() if perk_link_tag else ""
            perk_url = "https://deadbydaylight.fandom.com" + perk_link_tag["href"] if perk_link_tag else ""

            # Description
            description = cols[3].get_text(separator="\n", strip=True)

            perks.append({
                "name": perk_name,
                "url": perk_url,
                "description": description,
            })
        except Exception as e:
            print(f"Error parsing row: {e}")
    return perks

for perk in get_description():
    print(perk)
    perk_name = perk["name"].lower()
    if perk_name == "adrenaline":
        print(perk)

print(f"✅ Found {len(get_description())} perks")
print(json.dumps(get_description()[:3], indent=2))
