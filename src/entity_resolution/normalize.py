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

import re
import unicodedata
from functools import lru_cache

import pandas as pd

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


def clean_text(text: object) -> str:
    """ASCII-fold, lower-case, ``&`` -> ``and``, drop apostrophes, punctuation -> space."""
    if not isinstance(text, str) or not text:
        return ""
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
    "private": "pvt", "pvte": "pvt", "prv": "pvt", "pte": "pvt", "priv": "pvt",
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
_NAME_PHRASE_RE = _phrase_regex(NAME_PHRASES)


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
    return toks or [t for t in tokens if t not in _DBA_MARKERS] or tokens


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
    text = _replace_phrases(clean_text(raw), _NAME_PHRASE_RE, NAME_PHRASES)
    tokens = [NAME_CANON.get(t, t) for t in text.split()]
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
        "has_landmark": int(bool(landmark)),
    }


def normalize_country(raw: object) -> str:
    """Case/accent-insensitive country label; the label set stays open (no mapping to a fixed list)."""
    return clean_text(raw)


def normalize_records(records: pd.DataFrame) -> pd.DataFrame:
    """Return ``records`` with all normalised columns appended (row order preserved)."""
    names = pd.DataFrame([normalize_name(v) for v in records["business_name"].tolist()], index=records.index)
    addrs = pd.DataFrame(
        [normalize_address(a, c) for a, c in zip(records["business_address"].tolist(), records["country"].tolist())],
        index=records.index,
    )
    out = pd.concat([records, names, addrs], axis=1)
    out["country_norm"] = [normalize_country(c) for c in records["country"].tolist()]
    return out
