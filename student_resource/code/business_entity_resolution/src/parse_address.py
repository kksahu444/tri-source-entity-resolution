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
MARKER = re.compile(r"^([a-z]{1,4}) ?(?=\d)")
LEAD_ZEROS = re.compile(r"(?<![0-9])0+(?=[0-9])")

FIELDS = ("addr_b", "addr_core", "house", "house_sfx", "street", "city", "region", "unit", "landmark",
          "nums", "n_parts")


def clean_part(p: str) -> str:
    p = PART_CLEAN.sub(" ", p.replace(".", " ").replace("'", ""))
    p = DASH_SLASH.sub(" ", p)
    return WS.sub(" ", p).strip()


def strip_zeros(d: str) -> str:
    return d.lstrip("0") or "0"


def strip_marker(part: str, markers: frozenset) -> str:
    """Drop a numbering marker in front of the house number ("no 5 r x" / "ndeg5 r x" -> "5 r x").

    ``markers`` is mined per country (prep.numbering_markers); without it a part like "no 5 rue x" is not
    read as the house part, so its street is lost and the part can even be taken for the city.
    """
    if markers and (m := MARKER.match(part)) and m.group(1) in markers:
        return part[m.end():]
    return part


def parse_address(folded: str, regions: dict[str, str], markers: frozenset = frozenset(),
                  zero_strip: bool = False) -> tuple:
    """``folded``: lower-cased, accent-folded address. ``regions``: part text -> canonical region.
    ``markers``: numbering-marker tokens stripped from the start of a part. ``zero_strip``: drop leading zeros
    of numbers in the address text too ("0011 rue x" -> "11 rue x"), as house and nums already do, so the
    word-level address index matches the house number (3-5% of records in every country are zero-padded).

    Returns a tuple aligned with FIELDS.
    """
    parts = [strip_marker(c, markers) for c in (clean_part(p) for p in folded.split(",")) if c]
    if zero_strip:
        parts = [LEAD_ZEROS.sub("", p) for p in parts]
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
