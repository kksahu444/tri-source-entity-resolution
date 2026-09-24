"""Normalisation must collapse the documented noise patterns onto the same representation."""
from entity_resolution.normalize import normalize_address, normalize_name, skeleton


def core(name):
    return normalize_name(name)["name_core"]


def test_legal_suffixes_and_abbreviations():
    assert core("Acme Robotics Inc.") == core("ACME ROBOTICS INCORPORATED") == "acme robotics"
    assert core("Shree Laxmi Traders Pvt. Ltd.") == core("Shree Laxmi Traders Private Limited")
    assert core("Boulangerie Lefèvre SARL") == "boulangerie lefevre"


def test_punctuation_ampersand_and_initials():
    assert core("A.B.C. Enterprises") == core("ABC Enterprises")
    assert normalize_name("Smith & Sons")["name_norm"] == normalize_name("Smith and Sons")["name_norm"]


def test_dba_variants():
    out = normalize_name("Acme Holdings LLC d/b/a Acme Robotics")
    assert out["name_variants"].split("|") == ["acme hldg", "acme robotics"]


def test_transliteration_skeleton():
    assert skeleton("agarwal") == skeleton("aggarwal")
    assert skeleton("lakshmi") == skeleton("laxmi")


def test_address_abbreviations_and_postcodes():
    a = normalize_address("500 Market St, San Jose, CA 95113", "US")
    b = normalize_address("500 Market Street, San Jose California", "US")
    assert a["addr_core"] == b["addr_core"]
    assert a["addr_postcodes"] == "95113" and b["addr_postcodes"] == ""
    assert a["addr_numbers"] == "500"


def test_landmarks_and_spaced_pin():
    out = normalize_address("Plot 12, Near SBI ATM, MG Road, Bengaluru 560 001", "India")
    assert out["addr_landmark"] == "sbi atm"
    assert "sbi" not in out["addr_core"]
    assert out["addr_postcodes"] == "560001"
    assert out["has_landmark"] == 1


def test_country_label_removed_from_address_only_when_it_is_the_label():
    out = normalize_address("12 Rue Saint-Honoré, 75001 Paris, France", "France")
    assert out["addr_core"] == "12 rue st honore paris"
