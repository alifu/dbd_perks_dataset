import os, requests, json, time, random, re
from bs4 import BeautifulSoup, NavigableString
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

# ---- CONFIG ----
BASE_URL = "https://deadbydaylight.fandom.com/api.php"
OUT_DIR = "dbd_perks_dataset"
AUGMENT_PER_PERK = 50
REFETCH_IMAGES = False  # 👈 Set to True to force re-download all images
HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---- FOLDER SETUP ----
os.makedirs(OUT_DIR, exist_ok=True)
RAW_DIR = os.path.join(OUT_DIR, "raw")
AUG_DIR = os.path.join(OUT_DIR, "augmented")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(AUG_DIR, exist_ok=True)

metadata_path = os.path.join(OUT_DIR, "metadata.json")
metadata = {}

# ---- LOAD EXISTING METADATA ----
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"📂 Loaded {len(metadata)} existing metadata entries.")

def get_description_perks():
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

            # 2️⃣ Perk name & link
            perk_link_tag = cols[1].find("a")
            perk_name = perk_link_tag.text.strip() if perk_link_tag else ""
            perk_url = "https://deadbydaylight.fandom.com" + perk_link_tag["href"] if perk_link_tag else None

            # 3️⃣ Description
            for tag in cols[3].find_all(["div", "p", "li"]):
                tag.insert_before(NavigableString("\n"))

            description = cols[3].get_text()


            perks.append({
                "name": perk_name,
                "url": perk_url,
                "description": description,
            })
        except Exception as e:
            print(f"Error parsing row: {e}")
    return perks

perks_data = get_description_perks()

def extract_description(key: str):
    
    for perk in perks_data:
        perk_name = perk["name"].lower()
        if perk_name == key.lower().replace("_", " "):
            return perk["description"]
    return ""

def extract_perk_title(file_name: str) -> str:
    """
    Convert a DBD perk file name like:
        'File:IconPerks_forcedHesitation.png'
    → 'Forced_Hesitation'
    """
    # 1️⃣ Remove prefix/suffix
    base = file_name.replace("File:IconPerks", "").replace(".png", "")

    # 2️⃣ Normalize capitalization:
    #    Split camelCase and capitalize each word properly
    base = re.sub(r'(?<!^)(?=[A-Z])', ' ', base)  # insert space before capital letters
    base = "_".join(word.capitalize() for word in base.split())

    return base

def get_perk_description(perk_name: str):
    """Fetch perk description from Fandom API."""
    params = {
        "action": "query",
        "format": "json",
        "titles": perk_name,
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1
    }
    try:
        r = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=10).json()
        pages = r.get("query", {}).get("pages", {})
        for p in pages.values():
            extract = p.get("extract")
            if extract and len(extract.strip()) > 0:
                return extract.strip()
    except Exception as e:
        print(f"⚠️ Failed API description for {perk_name}: {e}")
    return "No description found."

# ---- STEP 1: Fetch perk image files ----
params = {
    "action": "query",
    "format": "json",
    "list": "categorymembers",
    "cmtitle": "Category:Perk_images",
    "cmlimit": "max"
}

perk_items = []
print("🔍 Fetching perk image list...")
while True:
    r = requests.get(BASE_URL, params=params).json()
    for item in r["query"]["categorymembers"]:
        title = item["title"]
        if title.lower().endswith(".png"):
            perk_items.append(title)
    if "continue" in r:
        params.update(r["continue"])
    else:
        break

print(f"✅ Found {len(perk_items)} perk images.\n")

# ---- STEP 2: Download images + fetch descriptions ----
for i, perk in enumerate(perk_items, 1):
    clean_name = perk.replace("File:IconPerks_", "").replace(".png", "")
    safe_name = clean_name.replace(" ", "_")
    image_path = os.path.join(RAW_DIR, f"{safe_name}.png")

    # --- Image download logic ---
    if os.path.exists(image_path) and not REFETCH_IMAGES:
        print(f"[{i}/{len(perk_items)}] ⏩ Skipping image for {safe_name}")
    else:
        img_params = {
            "action": "query",
            "format": "json",
            "titles": perk,
            "prop": "imageinfo",
            "iiprop": "url"
        }
        img_data = requests.get(BASE_URL, params=img_params).json()
        pages = img_data.get("query", {}).get("pages", {})
        img_url = None
        for p in pages.values():
            info = p.get("imageinfo", [])
            if info:
                img_url = info[0]["url"]

        if img_url:
            try:
                img_bytes = requests.get(img_url, headers=HEADERS).content
                with open(image_path, "wb") as f:
                    f.write(img_bytes)
                print(f"[{i}/{len(perk_items)}] 💾 Downloaded {safe_name}")
            except Exception as e:
                print(f"⚠️ Failed to download {safe_name}: {e}")
                continue
            time.sleep(0.2)
        else:
            print(f"❌ No image URL for {perk}")
            continue

    # --- Description scraping ---
    desc = extract_description(extract_perk_title(perk))

    metadata[safe_name] = {
        "name": clean_name.replace("_", " ").title(),
        "image": image_path,
        "description": desc,
    }

    # Save every 10 items
    if i % 10 == 0:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print("💾 Intermediate metadata saved.")

# ---- Final save ----
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"\n💾 Saved metadata for {len(metadata)} perks.")

# ---- STEP 3: Augment images ----
def augment_image(img: Image.Image):
    img = img.copy()
    img = img.rotate(random.uniform(-15, 15))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.5))
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
    return img

print("\n🎨 Generating augmented samples...")
for fname in os.listdir(RAW_DIR):
    perk_name = os.path.splitext(fname)[0]
    src = Image.open(os.path.join(RAW_DIR, fname)).convert("RGB").resize((128, 128))
    perk_dir = os.path.join(AUG_DIR, perk_name)
    os.makedirs(perk_dir, exist_ok=True)

    existing_aug = len(os.listdir(perk_dir))
    if existing_aug >= AUGMENT_PER_PERK:
        print(f"⏩ Skipping {perk_name}, already has {existing_aug} augmented images.")
        continue

    for j in range(existing_aug, AUGMENT_PER_PERK):
        out_path = os.path.join(perk_dir, f"{perk_name}_{j}.png")
        augment_image(src).save(out_path)

print("\n✅ Dataset complete!")
print(f"Total perks: {len(metadata)}")
print(f"Images per perk: {AUGMENT_PER_PERK}")
print(f"Output folder: {os.path.abspath(OUT_DIR)}")
