"""Golden view-B cases, taken from test_source1.tsv / train files (junk, reordered, uppercase,
bis/ter, (France), Dr Ambedkar Road, native scripts)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.normalize import fold, name_tokens, skeleton, split_legal, view_a  # noqa: E402
from src.parse_address import parse_address  # noqa: E402

LEGAL = frozenset({"llc", "inc", "pc", "corp", "ltd", "pvt", "llp", "co", "sarl", "sas", "eurl", "sa", "sasu", "sci", "ei"})
REG = {"nouvelle aquitaine": "nouvelle aquitaine", "ia": "ia", "uttar pradesh": "uttar pradesh",
       "tx": "tx", "maharashtra": "maharashtra", "texas": "tx", "hauts de france": "hauts de france"}


def core_legal(raw):
    main, alt, flags = name_tokens(fold(raw))
    core, legal = split_legal(main, LEGAL)
    return " ".join(core), " ".join(sorted(set(legal))), flags, " ".join(alt)


def test_view_a_only_nfkc_and_spaces():
    assert view_a("  Garza  Émpire\tLLC ") == "Garza Émpire LLC"


def test_fold_accents_and_scripts():
    assert fold("Écheevrr Léarning") == "echeevrr learning"
    assert fold("राम मार्केटिंग प्राइवेट लिमिटेड") == "ram marketimg praivet limited"


def test_junk_flags():
    core, legal, flags, _ = core_legal("<< Team Ecole")
    assert core == "team ecole" and flags["f_lead"] == 1
    core, legal, flags, _ = core_legal("Garza Empire LLC!")
    assert core == "garza empire" and legal == "llc" and flags["f_bang"] == 1
    core, legal, flags, _ = core_legal("(India) Meenakshi Supplies Private Limited")
    assert core == "meenakshi supplies" and legal == "ltd pvt" and flags["f_tag"] == 1
    core, legal, flags, _ = core_legal("Marina Ecole (France) Sarl")
    assert core == "marina ecole" and legal == "sarl" and flags["f_tag"] == 1
    core, _, flags, _ = core_legal('"ehpad Club SAS')
    assert core == "ehpad club" and flags["f_lead"] == 1


def test_legal_moved_to_front_and_dotted():
    assert core_legal("LLC Dalton and Woodworth Legato")[:2] == ("dalton and woodworth legato", "llc")
    assert core_legal("Clean Realty Services, L.L.C.")[:2] == ("clean realty services", "llc")
    assert core_legal("Pvt. EFS Print Ventures Ltd.")[:2] == ("efs print ventures", "ltd pvt")
    assert core_legal("Johnsen, Hicks & Fogarty P.C.")[:2] == ("johnsen hicks and fogarty", "pc")
    assert core_legal("विजन इंफोटेक प्राइवेट लिमिटेड")[1] == "ltd pvt"


def test_long_digit_runs_removed_and_dba_split():
    core, _, flags, _ = core_legal("Dynamic Consultants Private Ltd - 7648973034")
    assert core == "dynamic consultants" and flags["f_digits"] == 1
    core, _, _, alt = core_legal("Acme Holdings LLC dba Acme Robotics")
    assert core == "acme holdings" and alt == "acme robotics"


def test_skeleton():
    assert skeleton("private") == skeleton("praivet") == "prvt"
    assert skeleton("aggarwal") == "agrwl"


def P(addr):
    return dict(zip(("addr_b", "addr_core", "house", "house_sfx", "street", "city", "region", "unit",
                     "landmark", "nums", "n_parts"), parse_address(fold(addr), REG)))


def test_address_france_bis_and_region():
    a = P("Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac")
    assert (a["house"], a["house_sfx"], a["street"], a["region"], a["city"]) == \
        ("5", "bis", "rue pierre dignac", "nouvelle aquitaine", "la teste de buch")


def test_address_reordered_us():
    a = P("IA, Iowa City, 1064 Newton Rd, Unit 11")
    assert (a["house"], a["street"], a["city"], a["region"], a["unit"]) == ("1064", "newton rd", "iowa city", "ia", "unit 11")


def test_address_uppercase_and_alias():
    a = P("2621 COTTEN ROAD, TYLER, Texas")
    assert (a["house"], a["street"], a["city"], a["region"]) == ("2621", "cotten road", "tyler", "tx")


def test_address_dr_ambedkar_not_rewritten():
    a = P("12, Dr Ambedkar Road, Near SBI ATM, Pune, Maharashtra")
    assert "dr ambedkar road" in a["addr_core"] and a["landmark"] == "near sbi atm"
    assert a["house"] == "12" and a["region"] == "maharashtra" and a["city"] == "pune"


def test_address_leading_zeros_and_empty():
    assert P("02243 KEELER AVE, CHICAGO, IL")["house"] == "2243"
    assert P("")["n_parts"] == 0


def test_numbering_marker_france():
    from src.parse_address import strip_marker
    reg = {**REG, "nord": "hauts de france"}
    a = dict(zip(("addr_b", "addr_core", "house", "house_sfx", "street", "city", "region", "unit", "landmark", "nums",
                  "n_parts"), parse_address(fold("Lille, N° 94 R. Jules Valles, Nord"), reg, frozenset({"no", "ndeg"}))))
    assert (a["house"], a["street"], a["city"], a["region"]) == ("94", "r jules valles", "lille", "hauts de france")
    a = dict(zip(("addr_b", "addr_core", "house", "house_sfx", "street", "city"),
                 parse_address(fold("No 5 R Etienne Marcel, Pessac, Nouvelle-Aquitaine"), REG, frozenset({"no"}))))
    assert (a["house"], a["street"], a["city"]) == ("5", "r etienne marcel", "pessac")
    # without markers the street is lost (the behaviour config_v6 and earlier had)
    assert parse_address(fold("No 5 R Etienne Marcel, Pessac, Nouvelle-Aquitaine"), REG)[4] == ""
    assert strip_marker("ndeg4624 r anatole france", frozenset({"ndeg"})) == "4624 r anatole france"
    assert strip_marker("apt 5", frozenset({"no"})) == "apt 5"


def test_address_vocab_miners():
    from src.prep import numbering_markers, open_region_alias
    recs = [["no 5 rue etienne marcel", "pessac"]] * 250 + [["apt 5", "12 rue x", "lille"]] * 300
    assert numbering_markers(recs) == ["no"]
    regions = {"hauts de france", "nouvelle aquitaine"}
    s1 = [["1 rue a", "lille", "hauts de france"]] * 10 + [["2 rue b", "pessac", "nouvelle aquitaine"]] * 10
    rec = ([["3 rue a", "lille", "nord"]] * 40 + [["4 rue b", "pessac", "gironde"]] * 40
           + [["5 rue c", "lille", "hauts de france"]] * 40)
    assert open_region_alias(s1, rec, regions) == {"nord": "hauts de france", "gironde": "nouvelle aquitaine"}


def test_zero_strip_address_text():
    out = parse_address(fold("0011 Rue Nerton, Calais"), REG, frozenset(), True)
    assert (out[0], out[2], out[4]) == ("11 rue nerton, calais", "11", "rue nerton")
    assert parse_address(fold("0011 Rue Nerton, Calais"), REG)[0] == "0011 rue nerton, calais"
    assert parse_address(fold("100 Rue X, Calais"), REG, frozenset(), True)[0] == "100 rue x, calais"
