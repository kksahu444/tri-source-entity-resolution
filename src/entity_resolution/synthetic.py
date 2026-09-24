"""Synthetic data in the exact challenge format - for smoke tests and CI only.

The generator mimics the documented noise (abbreviations, legal suffixes, typos, transliteration,
word swaps, missing postcode/state, landmark-only addresses) and adds look-alike distractors.
Scores on it say nothing about the real leaderboard; it exists so the pipeline can be run end to
end before (or without) the real dataset.
"""
from __future__ import annotations

import random
from pathlib import Path

from .io import write_id_lists

US = {
    "first": ["Acme", "Summit", "Pioneer", "Blue Ridge", "Golden Gate", "Liberty", "Evergreen", "Silver Oak",
              "Redwood", "Horizon", "Keystone", "Northstar", "Riverbend", "Lakeside", "Iron Horse", "Eagle",
              "Patriot", "Cedar Point", "Sunrise", "Bright"],
    "second": ["Robotics", "Foods", "Logistics", "Dental Care", "Auto Repair", "Consulting", "Bakery", "Hardware",
               "Plumbing", "Realty", "Pharmacy", "Motors", "Printing", "Fitness", "Coffee Roasters", "Landscaping",
               "Electric", "Insurance Agency", "Veterinary Clinic", "Books"],
    "legal": ["Inc", "Inc.", "Incorporated", "LLC", "L.L.C.", "Corp", "Corporation", "Co", "Company", ""],
    "streets": ["Market", "Main", "Oak", "Pine", "Maple", "Elm", "Washington", "Lake", "Hill", "Park", "Sunset",
                "Broadway", "Cedar", "Highland", "River"],
    "types": [("Street", "St"), ("Road", "Rd"), ("Avenue", "Ave"), ("Boulevard", "Blvd"), ("Drive", "Dr"),
              ("Lane", "Ln")],
    "cities": [("San Jose", "CA", "California", "95"), ("Austin", "TX", "Texas", "78"), ("Reno", "NV", "Nevada", "89"),
               ("Boise", "ID", "Idaho", "83"), ("Fargo", "ND", "North Dakota", "58"),
               ("Seattle", "WA", "Washington", "98"), ("Denver", "CO", "Colorado", "80"),
               ("Chicago", "IL", "Illinois", "60")],
}
IN = {
    "first": ["Shree Ganesh", "Laxmi", "Sai", "Balaji", "Agarwal", "Krishna", "Om Sai", "Maa Durga", "Shiv Shakti",
              "Jai Hind", "Mahalaxmi", "Siddhi Vinayak", "Gurukripa", "Annapurna", "Hanuman", "Sri Venkateswara",
              "Bharat", "Navkar", "Ambika", "Parvati"],
    "second": ["Traders", "Enterprises", "Textiles", "Steel", "Pharma", "Electricals", "Jewellers", "Sweets",
               "Hardware", "Automobiles", "Agencies", "Constructions", "Motors", "Garments", "Medicals", "Tiles",
               "Plastics", "Foods", "Logistics", "Printers"],
    "legal": ["Pvt Ltd", "Private Limited", "Pvt. Ltd.", "LLP", "Ltd", "Limited", ""],
    "areas": ["MG Road", "Station Road", "Gandhi Nagar", "Shivaji Nagar", "Civil Lines", "Sector 17", "Anna Nagar",
              "Salt Lake", "Banjara Hills", "Koramangala", "Andheri East", "Rajaji Nagar", "Lajpat Nagar", "Model Town"],
    "cities": [("Bengaluru", "Karnataka", "560"), ("Mumbai", "Maharashtra", "400"), ("Chennai", "Tamil Nadu", "600"),
               ("Hyderabad", "Telangana", "500"), ("New Delhi", "Delhi", "110"), ("Pune", "Maharashtra", "411"),
               ("Kolkata", "West Bengal", "700"), ("Jaipur", "Rajasthan", "302")],
    "landmarks": ["SBI ATM", "Bus Stand", "City Hospital", "Hanuman Temple", "Railway Station", "Big Bazaar",
                  "Post Office", "Govt School"],
}
FR = {
    "first": ["Boulangerie", "Atelier", "Maison", "Garage", "Pharmacie", "Librairie", "Cabinet", "Café",
              "Fromagerie", "Brasserie"],
    "second": ["du Centre", "Dupont", "Saint-Michel", "Lefèvre", "des Arts", "de la Gare", "Moreau", "Belle Époque",
               "du Marché", "Rousseau"],
    "legal": ["SARL", "SAS", "SA", "EURL", ""],
    "types": [("Rue", "R."), ("Avenue", "Av."), ("Boulevard", "Bd"), ("Place", "Pl."), ("Chemin", "Ch.")],
    "streets": ["de la République", "Victor Hugo", "Jean Jaurès", "des Lilas", "Saint-Honoré",
                "du Général de Gaulle", "Pasteur", "de Paris"],
    "cities": [("Paris", "75"), ("Lyon", "69"), ("Marseille", "13"), ("Toulouse", "31"), ("Nantes", "44"),
               ("Lille", "59")],
}
ABBREV = {"Corporation": "Corp", "Incorporated": "Inc", "Private": "Pvt", "Limited": "Ltd", "Company": "Co",
          "Street": "St", "Road": "Rd", "Avenue": "Ave", "Boulevard": "Blvd", "Drive": "Dr", "Lane": "Ln",
          " and ": " & ", "Saint": "St", "Near": "Nr."}
TRANSLIT = [("ee", "i"), ("sh", "s"), ("aa", "a"), ("v", "w"), ("ksh", "x"), ("Shree", "Shri"), ("Laxmi", "Lakshmi"),
            ("Agarwal", "Aggarwal"), ("Sai", "Saii")]


def _typo(rng: random.Random, s: str) -> str:
    if len(s) < 4:
        return s
    i = rng.randrange(1, len(s) - 1)
    op = rng.choice(["drop", "swap", "dup"])
    if op == "drop":
        return s[:i] + s[i + 1:]
    if op == "swap":
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + s[i] + s[i:]


def _entity(rng: random.Random, country: str) -> dict:
    """Canonical (clean) name/address parts for one business."""
    if country == "US":
        city, st, st_long, zp = rng.choice(US["cities"])
        typ = rng.choice(US["types"])
        return {"country": country, "core": f"{rng.choice(US['first'])} {rng.choice(US['second'])}",
                "legal": rng.choice(US["legal"]), "num": str(rng.randint(1, 9999)),
                "street": rng.choice(US["streets"]), "type": typ, "city": city, "state": (st, st_long),
                "pc": zp + f"{rng.randint(0, 999):03d}"}
    if country == "India":
        city, state, pin = rng.choice(IN["cities"])
        return {"country": country, "core": f"{rng.choice(IN['first'])} {rng.choice(IN['second'])}",
                "legal": rng.choice(IN["legal"]), "num": f"{rng.randint(1, 400)}", "area": rng.choice(IN["areas"]),
                "city": city, "state": state, "pc": pin + f"{rng.randint(0, 99):03d}",
                "landmark": rng.choice(IN["landmarks"])}
    city, dep = rng.choice(FR["cities"])
    return {"country": country, "core": f"{rng.choice(FR['first'])} {rng.choice(FR['second'])}",
            "legal": rng.choice(FR["legal"]), "num": str(rng.randint(1, 200)), "type": rng.choice(FR["types"]),
            "street": rng.choice(FR["streets"]), "city": city, "pc": dep + f"{rng.randint(0, 999):03d}"}


def _render(rng: random.Random, e: dict, noise: float) -> tuple[str, str]:
    """One noisy (name, address) rendering of entity ``e``; ``noise`` in [0, 1]."""
    def maybe(p: float) -> bool:
        return rng.random() < p * noise

    legal = e["legal"]
    if maybe(0.5):
        legal = rng.choice({"US": US, "India": IN}.get(e["country"], FR)["legal"])
    core = e["core"]
    if e["country"] == "India" and maybe(0.5):
        a, b = rng.choice(TRANSLIT)
        core = core.replace(a, b)
    if maybe(0.15) and " " in core:
        w = core.split()
        core = " ".join(w[1:] + w[:1])
    if maybe(0.35):
        core = _typo(rng, core)
    name = f"{core} {legal}".strip()
    if e["country"] == "France" and maybe(0.5):
        name = name.replace("é", "e").replace("è", "e").replace("É", "E")

    c = e["country"]
    if c == "US":
        typ = e["type"][1] if maybe(0.6) else e["type"][0]
        state = e["state"][1] if maybe(0.4) else e["state"][0]
        parts = [f"{e['num']} {e['street']} {typ}", e["city"]]
        if not maybe(0.3):
            parts.append(f"{state} {e['pc']}" if not maybe(0.3) else state)
        addr = ", ".join(parts)
    elif c == "India":
        parts = [f"{'Plot No. ' if maybe(0.5) else ''}{e['num']}", e["area"]]
        if maybe(0.4):
            parts.insert(1, f"Near {e['landmark']}")
        parts.append(e["city"])
        if not maybe(0.4):
            parts.append(f"{e['state']} {e['pc'][:3]} {e['pc'][3:]}" if maybe(0.3) else e["pc"])
        addr = ", ".join(parts)
        if maybe(0.12):
            addr = f"Near {e['landmark']}, {e['city']}"
    else:
        typ = e["type"][1] if maybe(0.5) else e["type"][0]
        addr = f"{e['num']} {typ} {e['street']}, {'' if maybe(0.3) else e['pc'] + ' '}{e['city']}"
    for long, short in ABBREV.items():
        if long in addr and maybe(0.4):
            addr = addr.replace(long, short)
    if maybe(0.15):
        addr = _typo(rng, addr)
    if maybe(0.2):
        name, addr = name.upper(), addr.upper()
    return name, addr


def _lookalike(rng: random.Random, e: dict) -> dict:
    """A different business that shares the name stem or the exact address of ``e``."""
    other = _entity(rng, e["country"])
    if rng.random() < 0.5:  # same first word, other trade, same city
        other["core"] = e["core"].split()[0] + " " + other["core"].split()[-1]
        other["city"] = e["city"]
    else:  # same address, different business
        for key in ("num", "street", "type", "area", "city", "state", "pc"):
            if key in e:
                other[key] = e[key]
    return other


def generate_split(rng: random.Random, split: str, n: int, countries: list[str], out_dir: Path) -> None:
    """Write ``<split>_source{1,2,3}.tsv`` (+ ground truth for train) under ``out_dir/split``."""
    entities = [_entity(rng, rng.choice(countries)) for _ in range(n)]
    ids = rng.sample(range(100000, 999999), 3 * n * 3)
    it = iter(ids)
    s1, s2, s3, gt = [], [], [], {}
    for e in entities:
        sid = f"S1-{next(it)}"
        s1.append((sid, *_render(rng, e, 0.3), e["country"]))
        gt[sid] = []
        for rows, prefix, p1, p2 in ((s2, "S2", 0.55, 0.12), (s3, "S3", 0.5, 0.08)):
            r = rng.random()
            copies = 2 if r < p2 else 1 if r < p1 else 0
            for _ in range(copies):
                cid = f"{prefix}-{next(it)}"
                rows.append((cid, *_render(rng, e, 1.0), e["country"]))
                gt[sid].append(cid)
    for rows, prefix in ((s2, "S2"), (s3, "S3")):  # distractors: look-alikes + unrelated records
        for _ in range(int(0.35 * n)):
            base = rng.choice(entities)
            other = _lookalike(rng, base) if rng.random() < 0.6 else _entity(rng, base["country"])
            rows.append((f"{prefix}-{next(it)}", *_render(rng, other, 1.0), other["country"]))
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for rows, k in ((s1, 1), (s2, 2), (s3, 3)):
        rng.shuffle(rows)
        with open(split_dir / f"{split}_source{k}.tsv", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for row in rows:
                fh.write("\t".join(x.replace("\t", " ") for x in row) + "\n")
    if split == "train":
        write_id_lists(split_dir / "train_ground_truth.tsv", [r[0] for r in s1], gt, "matched_entity_ids")
    else:  # hidden labels kept outside test/ so the pipeline treats test as unlabelled
        write_id_lists(out_dir / "hidden_labels" / f"{split}_ground_truth.tsv", [r[0] for r in s1], gt,
                       "matched_entity_ids")


def generate(out_dir: str | Path = "data_synthetic", n_train: int = 1500, n_test: int = 900, seed: int = 7) -> Path:
    """Train covers US + India; test adds France (unseen country), like the real challenge."""
    rng = random.Random(seed)
    out = Path(out_dir)
    generate_split(rng, "train", n_train, ["US", "India"], out)
    generate_split(rng, "test", n_test, ["US", "India", "France"], out)
    return out
