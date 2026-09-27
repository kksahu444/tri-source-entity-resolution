"""Text normalisation for business names and addresses.

Design rules
------------
* **Open-set countries.** France appears only in the test set, so nothing here branches on a
  hard-coded country list. Dictionaries are merged across US / India / French conventions and
  applied to every record.
* **Canonical short forms.** Abbreviations map to one *short* token ("street"/"str"/"st" -> "st")
  instead of being expanded. Expansion is ambiguous ("st" = street or saint); collapsing both
  sides to the same short token is not.
* **Only the provided data.** These are hand-written normalisation rules, not external lookups.
"""
from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache

import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz.fuzz import ratio

# ----------------------------------------------------------------------------------------
# generic cleaning
# ----------------------------------------------------------------------------------------
_SPECIAL_CHARS = str.maketrans(
    {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "đ": "d", "ł": "l", "ı": "i", "’": "'", "‘": "'", "`": "'", "´": "'"}
)
_NON_WORD = re.compile(r"[^\w\s]|_")


def strip_accents(text: str) -> str:
    """Lower-case and fold accents: ``"Lefèvre Époque"`` -> ``"lefevre epoque"``."""
    text = unicodedata.normalize("NFKD", text.lower().translate(_SPECIAL_CHARS))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def _join_single_letters(tokens: list[str]) -> list[str]:
    """Merge runs of single letters so ``"a b c traders"`` == ``"abc traders"`` (from "A.B.C.")."""
    out: list[str] = []
    run: list[str] = []
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(tok)
    if run:
        out.append("".join(run))
    return out


# Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam)
# share one layout: the same letter sits at the same offset inside each 128-code-point block, so
# one offset table transliterates all nine. Romanisation follows common Indian-English spelling
# ("ch", "sh", short vowels) because that is how the Latin-script sources write the same names.
_INDIC_START, _INDIC_END = 0x0900, 0x0D7F
_INDIC_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh",
    0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
    0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y",
    0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s",
    0x39: "h", 0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_INDIC_VOWELS = {  # independent vowel letters
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li", 0x0D: "e",
    0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
_INDIC_SIGNS = {  # dependent vowel signs, replacing a consonant's inherent "a"
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e",
    0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "li",
}
_INDIC_OTHER = {0x01: "n", 0x02: "n", 0x03: "h", 0x50: "om", 0x70: "n"} | {0x66 + d: str(d) for d in range(10)}
_VIRAMA = 0x4D
_HAS_INDIC = re.compile(r"[\u0900-\u0d7f]")


def transliterate_indic(text: str) -> str:
    """Romanise Indic-script text: ``राम मार्केटिंग प्राइवेट लिमिटेड`` -> ``ram marketing praivet limited``.

    A consonant carries an inherent "a" unless a vowel sign or virama follows; the inherent "a" is
    dropped at the end of a word, as Hindi pronunciation (and Latin spelling) does.
    """
    out: list[str] = []
    pending_a = False
    for ch in text:
        cp = ord(ch)
        if _INDIC_START <= cp <= _INDIC_END:
            off = cp & 0x7F
            if off in _INDIC_SIGNS:
                out.append(_INDIC_SIGNS[off])
                pending_a = False
                continue
            if off == _VIRAMA:
                pending_a = False
                continue
            if off in (0x3C, 0x3D, 0x71) or 0x51 <= off <= 0x57:  # nukta, avagraha, addak, accents
                continue
            if pending_a:
                out.append("a")
            if off in _INDIC_CONSONANTS:
                out.append(_INDIC_CONSONANTS[off])
                pending_a = True
            else:
                out.append(_INDIC_VOWELS.get(off) or _INDIC_OTHER.get(off, ""))
                pending_a = False
            continue
        pending_a = False  # word end: the inherent vowel is silent
        out.append(ch)
    return "".join(out)


def clean_text(text: object) -> str:
    """Romanise Indic scripts, ASCII-fold, lower-case, ``&`` -> ``and``, drop apostrophes,
    punctuation -> space."""
    if not isinstance(text, str) or not text:
        return ""
    if not text.isascii() and _HAS_INDIC.search(text):
        text = transliterate_indic(text)
    t = strip_accents(text)
    t = t.replace("&", " and ").replace("@", " at ").replace("'", "")
    t = _NON_WORD.sub(" ", t)
    return " ".join(_join_single_letters(t.split()))


def _phrase_regex(phrases: dict[str, str]) -> re.Pattern:
    keys = sorted(phrases, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")


def _replace_phrases(text: str, pattern: re.Pattern, phrases: dict[str, str]) -> str:
    return pattern.sub(lambda m: phrases[m.group(0)], text)


# ----------------------------------------------------------------------------------------
# business names
# ----------------------------------------------------------------------------------------
NAME_PHRASES = {
    "limited liability company": "llc",
    "limited liability partnership": "llp",
    "limited partnership": "lp",
    "public limited company": "plc",
    "one person company": "opc",
    "pra li": "pvt ltd",
    "doing business as": "dba",
    "trading as": "dba",
    "also known as": "aka",
    "formerly known as": "fka",
    "societe anonyme": "sa",
    "societe par actions simplifiee unipersonnelle": "sasu",
    "societe par actions simplifiee": "sas",
    "societe a responsabilite limitee": "sarl",
    "entreprise unipersonnelle a responsabilite limitee": "eurl",
    "societe civile immobiliere": "sci",
    "societe en nom collectif": "snc",
}

NAME_CANON = {
    # legal forms
    "incorporated": "inc", "incorporation": "inc", "incorp": "inc",
    "corporation": "corp", "corpn": "corp", "corporations": "corp",
    "company": "co", "companies": "co", "cos": "co", "compnay": "co",
    "limited": "ltd", "ltda": "ltd", "lmtd": "ltd", "ltd": "ltd",
    "private": "pvt", "praivet": "pvt", "privet": "pvt", "pvte": "pvt", "prv": "pvt", "pte": "pvt", "priv": "pvt",
    # romanised Indic spellings (Tamil, Bengali, Malayalam, Hindi LLP spelt out)
    "limitet": "ltd", "limirrad": "ltd", "piraivet": "pvt", "praibhet": "pvt", "praivarr": "pvt",
    "elaelapi": "llp", "elelpi": "llp",
    "public": "pub",
    "et": "and", "und": "and",
    # frequent business words
    "international": "intl", "internationals": "intl",
    "manufacturing": "mfg", "manufacturers": "mfrs", "manufacturer": "mfrs",
    "services": "svc", "service": "svc", "svcs": "svc", "srvcs": "svc", "servs": "svc",
    "brothers": "bros", "bro": "bros",
    "associates": "assoc", "associate": "assoc", "assocs": "assoc",
    "association": "assn",
    "technologies": "tech", "technology": "tech", "techs": "tech", "technologys": "tech",
    "enterprises": "ent", "enterprise": "ent", "entp": "ent", "enterprize": "ent", "enterprizes": "ent",
    "industries": "inds", "industry": "inds", "industrial": "indl",
    "solutions": "soln", "solution": "soln", "solns": "soln",
    "systems": "sys", "system": "sys",
    "group": "grp", "groups": "grp",
    "holdings": "hldg", "holding": "hldg",
    "department": "dept",
    "national": "natl",
    "center": "ctr", "centre": "ctr",
    "university": "univ",
    "hospital": "hosp",
    "laboratories": "labs", "laboratory": "labs", "lab": "labs",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma",
    "engineering": "engg", "engineers": "engrs",
    "construction": "constr", "constructions": "constr",
    "electricals": "elec", "electrical": "elec", "electric": "elec", "electronics": "elecs",
    "agency": "agcy", "agencies": "agcy",
    "saint": "st", "sainte": "ste", "mount": "mt", "fort": "ft",
    # Indian honorifics / transliteration variants
    "shri": "sri", "shree": "sri", "sree": "sri", "shrii": "sri", "sriee": "sri",
    "messrs": "ms",
}

# tokens that never identify a business on their own
LEGAL_TOKENS = frozenset(
    "inc corp co ltd llc llp lp plc pvt pub opc pllc lllp pc gmbh ag sa sas sasu sarl eurl sci snc bv nv pty srl spa".split()
)
_STRONG_LEGAL = frozenset("inc corp ltd llc llp plc pvt opc pllc lllp gmbh sarl sas sasu eurl".split())
_NAME_STOP = frozenset("the a an of and le la les l de du des d ms".split())
_DBA_MARKERS = frozenset("dba aka fka ta".split())
# honorifics and the "Center" tag the sources add around names; dropped only at the ends of the core
_NAME_AFFIX = frozenset("mr mrs smt dr sri ctr".split())
_NAME_PHRASE_RE = _phrase_regex(NAME_PHRASES)
_DOMAIN = re.compile(r"\b(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:co\.in|com|net|org|in|co|biz|info|fr|us|io)\b", re.I)


@lru_cache(maxsize=500_000)
def skeleton(token: str) -> str:
    """Phonetic-ish consonant skeleton that survives common transliteration noise.

    ``agarwal``/``aggarwal`` -> ``agrvl``; ``lakshmi``/``laxmi`` -> ``lksm``. Digits are kept.
    """
    if not token or token.isdigit():
        return token
    t = token
    for src, dst in (("x", "ks"), ("ph", "f"), ("kh", "k"), ("gh", "g"), ("bh", "b"), ("dh", "d"),
                     ("th", "t"), ("sh", "s"), ("ch", "c"), ("ck", "k"), ("q", "k"), ("z", "s"), ("w", "v")):
        t = t.replace(src, dst)
    head, tail = t[0], re.sub(r"[aeiouyh]", "", t[1:])
    t = head + tail
    return re.sub(r"(.)\1+", r"\1", t)


def _core_tokens(tokens: list[str]) -> list[str]:
    """Drop legal forms, stop words and DBA markers; fall back to all tokens if nothing is left."""
    toks = [t for t in tokens if t not in _STRONG_LEGAL and t not in _DBA_MARKERS]
    while toks and toks[-1] in LEGAL_TOKENS:  # trailing "... & co", "... sa"
        toks.pop()
    toks = [t for t in toks if t not in _NAME_STOP]
    while len(toks) > 1 and toks[0] in _NAME_AFFIX:
        toks.pop(0)
    while len(toks) > 1 and toks[-1] in _NAME_AFFIX:
        toks.pop()
    return toks or [t for t in tokens if t not in _DBA_MARKERS] or tokens


def _fix_legal_typos(tokens: list[str]) -> list[str]:
    """``prvsiare ltd`` -> ``pvt ltd``: noisy sources corrupt "Private" at random, but the token
    right before "Limited" is still close to it. Real words that land near "private" start with
    "pa" (parvati, parivar) or "print", and are left alone."""
    out = list(tokens)
    for i in range(len(out) - 1):
        t = out[i]
        if (out[i + 1] == "ltd" and len(t) >= 6 and t[:2] in ("pr", "pi", "pv") and not t.startswith("print")
                and t not in NAME_CANON):
            r = ratio(t, "private")
            if r >= 70 or (r >= 64 and "v" in t):
                out[i] = "pvt"
    return out


def _split_dba(tokens: list[str]) -> list[list[str]]:
    """``acme hldg llc dba acme robotics`` -> ``[[acme, hldg, llc], [acme, robotics]]``."""
    parts, cur = [], []
    for tok in tokens:
        if tok in _DBA_MARKERS:
            if cur:
                parts.append(cur)
            cur = []
        else:
            cur.append(tok)
    if cur:
        parts.append(cur)
    return parts


def normalize_name(raw: object) -> dict:
    """Normalise a business name into the representations used by blocking and features."""
    if isinstance(raw, str) and "." in raw:
        raw = _DOMAIN.sub(r"\1", raw)
    text = _replace_phrases(clean_text(raw), _NAME_PHRASE_RE, NAME_PHRASES)
    tokens = _fix_legal_typos([NAME_CANON.get(t, t) for t in text.split()])
    core = _core_tokens(tokens)
    parts = _split_dba(tokens)
    variants = [" ".join(_core_tokens(p)) for p in parts] if len(parts) > 1 else []
    alpha = [t for t in core if t[0].isalpha()]
    return {
        "name_norm": " ".join(tokens),
        "name_core": " ".join(core),
        "name_skel": " ".join(skeleton(t) for t in core),
        "name_acronym": "".join(t[0] for t in alpha) if len(alpha) >= 2 else "",
        "name_legal": " ".join(sorted({t for t in tokens if t in LEGAL_TOKENS})),
        "name_digits": " ".join(sorted({t for t in core if t.isdigit()})),
        "name_variants": "|".join(variants),
    }


# ----------------------------------------------------------------------------------------
# addresses
# ----------------------------------------------------------------------------------------
ADDR_PHRASES = {
    "post office": "po", "p o box": "pobox", "po box": "pobox", "post box": "pobox", "boite postale": "pobox",
    "next to": "nr", "close to": "nr", "near to": "nr", "in front of": "opp", "opposite to": "opp",
    "opposite of": "opp", "adjacent to": "adj", "land mark": "landmark",
    # multi-word US states / Indian states (long form -> short code)
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "south carolina": "sc", "south dakota": "sd",
    "rhode island": "ri", "west virginia": "wv", "district of columbia": "dc",
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "himachal pradesh": "hp", "madhya pradesh": "mp",
    "uttar pradesh": "up", "tamil nadu": "tn", "west bengal": "wb", "jammu and kashmir": "jk",
    "national capital territory of delhi": "delhi", "nct of delhi": "delhi", "nct delhi": "delhi",
}

ADDR_CANON = {
    # street types
    "street": "st", "str": "st", "strt": "st", "saint": "st",
    "road": "rd", "raod": "rd",
    "avenue": "ave", "av": "ave", "avn": "ave", "aven": "ave",
    "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "bld": "blvd", "blv": "blvd",
    "drive": "dr", "drv": "dr",
    "lane": "ln",
    "court": "ct", "crt": "ct",
    "place": "pl", "plc": "pl",
    "square": "sq", "sqr": "sq",
    "highway": "hwy", "hiway": "hwy",
    "parkway": "pkwy", "pky": "pkwy",
    "expressway": "expy",
    "terrace": "ter", "terr": "ter",
    "circle": "cir", "crcl": "cir",
    "crossing": "xing",
    "cross": "crs",
    "extension": "ext", "extn": "ext",
    # units / buildings
    "suite": "ste", "apartment": "apt", "appt": "apt", "building": "bldg", "bldng": "bldg",
    "floor": "fl", "flr": "fl", "room": "rm", "unit": "unit", "number": "no", "num": "no",
    "plot": "plot", "shop": "shop", "house": "h", "hno": "h",
    # directions
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    # landmarks
    "near": "nr", "nearby": "nr", "opposite": "opp", "oppo": "opp", "behind": "bhd", "beside": "bsd",
    "besides": "bsd", "adjacent": "adj",
    # Indian address words
    "sector": "sec", "sect": "sec", "phase": "ph", "nagar": "ngr", "ngr": "ngr", "marg": "marg",
    "chowk": "chowk", "colony": "col", "cly": "col", "layout": "lyt",
    "industrial": "indl", "estate": "est", "district": "dist", "dist": "dist", "village": "vill",
    "vil": "vill", "junction": "jn", "jct": "jn", "station": "stn", "center": "ctr", "centre": "ctr",
    "complex": "cmplx", "tower": "twr", "towers": "twr", "bazar": "bazaar", "gali": "gali",
    "mount": "mt", "fort": "ft",
    # French address words
    "rue": "rue", "chemin": "chem", "impasse": "imp", "allee": "allee", "route": "rte",
    "faubourg": "fbg", "quai": "quai", "cours": "crs", "residence": "res", "batiment": "bat", "bat": "bat",
    "etage": "fl", "cedex": "",
    # single-word US states (long -> short)
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "wisconsin": "wi", "wyoming": "wy",
    # single-word Indian states / variants (long -> short)
    "maharashtra": "mh", "karnataka": "ka", "telangana": "tg", "gujarat": "gj", "rajasthan": "rj",
    "kerala": "kl", "punjab": "pb", "haryana": "hr", "bihar": "br", "odisha": "od", "orissa": "od",
    "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk", "uttaranchal": "uk",
    "goa": "goa",
    # frequent Indian city spelling variants
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "gurgaon": "gurugram", "poona": "pune", "trivandrum": "thiruvananthapuram", "baroda": "vadodara",
}

LANDMARK_MARKERS = frozenset({"nr", "opp", "bhd", "bsd", "adj", "landmark"})
_ADDR_PHRASE_RE = _phrase_regex(ADDR_PHRASES)
_ZIP4 = re.compile(r"\b(\d{5})-\d{4}\b")
_SPACED_PIN = re.compile(r"\b(\d{3})\s(\d{3})\b")
_POSTCODE = re.compile(r"\b\d{5,6}\b")
_LEADING_DIGITS = re.compile(r"^(\d+)")
_SEGMENT_SPLIT = re.compile(r"[,;\n|]")
# street types, unit words and directions: never the identifying word of a street name
_ADDR_GENERIC = frozenset(
    "st rd ave blvd dr ln ct pl sq hwy pkwy expy ter cir xing crs ext rue chem imp allee rte fbg quai res bat "
    "no plot h unit ste apt bldg fl rm shop sec ph n s e w ne nw se sw".split()
)


# region codes after ADDR_CANON / ADDR_PHRASES (US states + DC, Indian states / UTs) and filler
# segments: a segment made only of these is never the city
_REGIONS = frozenset(
    "al ak az ar ca co ct de dc fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv nh nj nm ny nc nd "
    "oh ok or pa ri sc sd tn tx ut vt va wa wv wi wy "
    "mh ka tg gj rj kl pb hr br od as jh cg uk goa ap hp mp up wb jk ga ml mn mz nl sk tr py ch "
    "n a na nil none".split()
)
_CITY_FILLER = frozenset("city of the town twp village vill dist district borough boro township cdp".split())


def _city(segments: list[list[str]]) -> str:
    """Last segment that looks like a place name: no digits, not a region code, not a street line.

    ``500 Market St, San Jose, CA`` -> ``san jose``; ``SC, York, 191 Bezelle Ave`` -> ``york``;
    ``..., City of El Paso`` and ``Ticonderoga Townshiip`` lose the administrative words.
    """
    for toks in reversed(segments):
        words = [t for t in toks if t not in _CITY_FILLER and not t.startswith(("townsh", "towsh"))
                 and not _POSTCODE.fullmatch(t)]
        if not words or any(ch.isdigit() for t in words for ch in t) or all(t in _REGIONS for t in words):
            continue
        if words[-1] in _ADDR_GENERIC and len(words) > 1:  # "mg rd", "oak ave"; "st paul" is a city
            continue
        return " ".join(words)
    return ""


def _house_street(core: list[str]) -> str:
    """First house number plus the next identifying word: ``500 market st san jose`` -> ``500 market``."""
    for i, tok in enumerate(core):
        if m := _LEADING_DIGITS.match(tok):
            word = next((t for t in core[i + 1:] if t.isalpha() and t not in _ADDR_GENERIC), "")
            return f"{m.group(1)} {word}" if word else ""
    return ""


def normalize_address(raw: object, country: object = "") -> dict:
    """Split an address into core tokens, landmark tokens, postcodes and house numbers.

    ``"Plot 12, Near SBI ATM, MG Road, Bengaluru 560 001"`` ->
    core ``"plot 12 mg rd bengaluru"``, landmark ``"sbi atm"``, postcodes ``"560001"``, numbers ``"12"``.
    """
    raw = raw if isinstance(raw, str) else ""
    raw = _SPACED_PIN.sub(r"\1\2", _ZIP4.sub(r"\1", raw))
    postcodes = sorted(set(_POSTCODE.findall(raw)))
    country_clean = clean_text(country)

    core: list[str] = []
    landmark: list[str] = []
    segments: list[list[str]] = []
    for segment in _SEGMENT_SPLIT.split(raw):
        seg = clean_text(segment)
        if not seg or seg == country_clean:
            continue
        seg = _replace_phrases(seg, _ADDR_PHRASE_RE, ADDR_PHRASES)
        toks = [ADDR_CANON.get(t, t) for t in seg.split()]
        toks = [t for t in toks if t]
        cut = next((i for i, t in enumerate(toks) if t in LANDMARK_MARKERS), None)
        if cut is not None:
            landmark.extend(toks[cut + 1:])
            toks = toks[:cut]
        core.extend(toks)
        segments.append(toks)

    pc_set = set(postcodes)
    if core and country_clean and core[-1] == country_clean:
        core = core[:-1]
    core = [t for t in core if t not in pc_set]
    numbers = list(dict.fromkeys(m.group(1) for t in core if (m := _LEADING_DIGITS.match(t))))  # order kept
    return {
        "addr_norm": " ".join(core + landmark + postcodes),
        "addr_core": " ".join(core),
        "addr_landmark": " ".join(landmark),
        "addr_postcodes": " ".join(postcodes),
        "addr_numbers": " ".join(numbers),
        "addr_street": _house_street(core),
        "addr_city": _city(segments),
        "has_landmark": int(bool(landmark)),
    }


def normalize_country(raw: object) -> str:
    """Case/accent-insensitive country label; the label set stays open (no mapping to a fixed list)."""
    return clean_text(raw)


def _normalize_chunk(records: pd.DataFrame) -> pd.DataFrame:
    names = pd.DataFrame([normalize_name(v) for v in records["business_name"].tolist()], index=records.index)
    addrs = pd.DataFrame(
        [normalize_address(a, c) for a, c in zip(records["business_address"].tolist(), records["country"].tolist())],
        index=records.index,
    )
    out = pd.concat([records, names, addrs], axis=1)
    out["country_norm"] = [normalize_country(c) for c in records["country"].tolist()]
    return out


def normalize_records(records: pd.DataFrame, chunk: int = 250_000) -> pd.DataFrame:
    """Return ``records`` with all normalised columns appended (row order preserved).

    Pure Python per record, so partitions of millions of rows are split across worker processes.
    """
    if len(records) <= chunk:
        return _normalize_chunk(records)
    parts = Parallel(n_jobs=os.cpu_count() or 1)(
        delayed(_normalize_chunk)(records.iloc[s:s + chunk]) for s in range(0, len(records), chunk))
    return pd.concat(parts)
