SCHEMA_VERSION = 1
SOURCE_NOTE = "https://deadbydaylight.fandom.com (CC BY-SA 3.0, icons (c) Behaviour Interactive). Educational use only."
MAX_ALLOWED_COUNT_DROP = 0.05

PERK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id", "name", "role", "owner", "iconFile",
        "description", "flavor", "descriptionPatch", "wikiUrl",
    ],
    "properties": {
        "id": {"type": "string", "pattern": "^[a-z0-9]+(_[a-z0-9]+)*$"},
        "name": {"type": "string", "minLength": 1},
        "role": {"enum": ["survivor", "killer"]},
        "owner": {"type": "string", "description": "Empty string means a generic (teachable) perk."},
        "iconFile": {"type": "string", "pattern": r"\.png$"},
        "iconUrl": {"type": "string", "pattern": "^https://", "description": "Absent for perks only in the game data (no wiki icon)."},
        "description": {"type": "string", "minLength": 1},
        "flavor": {"type": "string"},
        "descriptionPatch": {"type": ["string", "null"]},
        "wikiUrl": {"type": "string", "pattern": "^https://"},
    },
}

DOCUMENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schemaVersion", "generatedAt", "source", "perks"],
    "properties": {
        "schemaVersion": {"const": SCHEMA_VERSION},
        "generatedAt": {"type": "string"},
        "source": {"type": "string"},
        "wikiRevisionId": {"type": "integer"},
        "perks": {"type": "array", "minItems": 1, "items": PERK_SCHEMA},
    },
}
