"""D1 order-invariant address component parser (view B)."""
from __future__ import annotations

import re

PART_CLEAN = re.compile(r"[^0-9a-z/\- ]+")
DASH_SLASH = re.compile(r"(?<![0-9])[-/]|[-/](?![0-9])")
WS = re.compile(r"\s+")
DIGITS = re.compile(r"\d+")
# spec D1 step 7 tag vocabularies (listed in the spec itself)
LANDMARK = re.compile(r"^(?:near|opp|opposite|behind|next to|beside)\b|\bke paa?s\b")
UNIT = re.compile(r"^(?:unit|suite|fl|flr|floor|ofc|office)\b")
HOUSE = re.compile(r"^(\d+)(?:\s*(bis|ter)\b|([a-z])\b)?")

FIELDS = ("addr_b", "addr_core", "house", "house_sfx", "street", "city", "region", "unit", "landmark",
          "nums", "n_parts")


def clean_part(p: str) -> str:
    p = PART_CLEAN.sub(" ", p.replace(".", " ").replace("'", ""))
    p = DASH_SLASH.sub(" ", p)
    return WS.sub(" ", p).strip()


def strip_zeros(d: str) -> str:
    return d.lstrip("0") or "0"


def parse_address(folded: str, regions: dict[str, str]) -> tuple:
    """``folded``: lower-cased, accent-folded address. ``regions``: part text -> canonical region.

    Returns a tuple aligned with FIELDS.
    """
    parts = [c for c in (clean_part(p) for p in folded.split(",")) if c]
    if not parts:
        return ("", "", "", "", "", "", "", "", "", "", 0)
    tags = []
    for p in parts:
        if LANDMARK.search(p):
            tags.append("L")
        elif UNIT.match(p):
            tags.append("U")
        elif p in regions:
            tags.append("R")
        elif p[0].isdigit():
            tags.append("H")
        else:
            tags.append("O")
    region, ridx = "", -1
    for i in range(len(parts) - 1, -1, -1):
        if tags[i] == "R":
            region, ridx = regions[parts[i]], i
            break
    city = ""
    if ridx >= 0:
        if ridx > 0 and tags[ridx - 1] == "O":
            city = parts[ridx - 1]
        elif ridx + 1 < len(parts) and tags[ridx + 1] == "O":
            city = parts[ridx + 1]
    if not city:
        if len(parts) >= 2 and tags[-2] == "O":
            city = parts[-2]
        else:
            others = [p for p, t in zip(parts, tags) if t == "O"]
            city = others[-1] if others else ""
    house, sfx, street = "", "", ""
    hidx = tags.index("H") if "H" in tags else -1
    if hidx >= 0:
        m = HOUSE.match(parts[hidx])
        house = strip_zeros(m.group(1))
        sfx = m.group(2) or m.group(3) or ""
        street = parts[hidx][m.end():].strip()
    else:
        for p, t in zip(parts, tags):
            if t in ("O",) and (m := DIGITS.search(p)):
                house = strip_zeros(m.group(0))
                break
    unit = " ".join(p for p, t in zip(parts, tags) if t == "U")
    landmark = " ".join(p for p, t in zip(parts, tags) if t == "L")
    core = " ".join(p for p, t in zip(parts, tags) if t in ("H", "O"))
    nums = " ".join(dict.fromkeys(strip_zeros(d) for d in DIGITS.findall(" ".join(parts))))
    return (", ".join(parts), core, house, sfx, street, city, region, unit, landmark, nums, len(parts))
