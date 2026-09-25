"""
tests/test_normalize.py: Comprehensive test suite for normalization pipeline.
Tests edge cases discovered in the AWS ML Hackathon 2026 ER datasets.
"""

import pytest
from src.normalize import (
    transliterate_indic,
    strip_accents,
    normalize_name,
    normalize_name_clean,
    strip_legal_suffixes,
    extract_dba_name,
    clean_domain_name,
    normalize_address,
    extract_postal_code,
    extract_numeric_tokens,
    get_derived_record,
)

# ---------------------------------------------------------------------------
# Test Indic Transliteration
# ---------------------------------------------------------------------------

def test_transliterate_indic_devanagari():
    text = "श्री सिस्टम्स प्रा. लि."
    result = transliterate_indic(text)
    assert "shree" in result
    assert "sistams" in result

def test_transliterate_indic_tamil():
    text = "அரிஹந்த்"
    result = transliterate_indic(text)
    assert len(result) > 0
    assert result.isascii()

def test_transliterate_indic_kannada():
    text = "ಕರ್ನಾಟಕ"
    result = transliterate_indic(text)
    assert "karnaatak" in result or "karnataka" in result

def test_transliterate_indic_malayalam():
    text = "സിറ്റി ലോജിസ്റ്റിക്സ്"
    result = transliterate_indic(text)
    assert "sii" in result or "siti" in result

def test_transliterate_indic_passthrough_ascii():
    assert transliterate_indic("Hello World 123") == "Hello World 123"
    assert transliterate_indic("") == ""
    assert transliterate_indic(None) is None

# ---------------------------------------------------------------------------
# Test Accent Stripping
# ---------------------------------------------------------------------------

def test_strip_accents_french():
    assert strip_accents("École primaire Sainte Pierre") == "Ecole primaire Sainte Pierre"
    assert strip_accents("Maison de Santé") == "Maison de Sante"
    assert strip_accents("Thénard, Clément, Réunis, Kléber") == "Thenard, Clement, Reunis, Kleber"
    assert strip_accents("Allée des Hêtres") == "Allee des Hetres"

def test_strip_accents_empty():
    assert strip_accents("") == ""
    assert strip_accents(None) == ""

# ---------------------------------------------------------------------------
# Test Business Name Normalization
# ---------------------------------------------------------------------------

def test_normalize_name_basic():
    assert normalize_name("Orelee's Barbershop") == "orelee s barbershop"
    assert normalize_name("  Apple   Inc.  ") == "apple inc"

def test_normalize_name_punctuation_and_symbols():
    assert normalize_name("Thermal & Fils SASU") == "thermal and fils sasu"
    assert normalize_name("U.S. Steel Corp.") == "us steel corp"
    assert normalize_name("P.V.T. Ltd.") == "pvt ltd"
    assert normalize_name("#centraleducation") == "centraleducation"

def test_normalize_name_decorative_symbols():
    assert normalize_name("*** Sai Tech Private Limited") == "sai tech private limited"
    assert normalize_name("<< Team Ecole >>") == "team ecole"
    assert normalize_name("2253 Park Plaza [Realty]") == "2253 park plaza realty"
    assert normalize_name("Sri *** Jadeja Services (India) Prívate Limited") == "sri jadeja services private limited"

def test_normalize_name_dba():
    assert normalize_name("Fluxkor DBA: Premier Star Payment L.L.C.") == "premier star payment llc"
    assert normalize_name("Orbicalo DBA: Cho, Weinrich and Fowler") == "cho weinrich and fowler"
    assert normalize_name("Synecto F/K/A Basalt Safe") == "basalt safe"

def test_normalize_name_domain():
    assert normalize_name("Physicaltherapycare.Com") == "physicaltherapycare"
    assert normalize_name("heartinstitutecity.com") == "heartinstitutecity"
    assert normalize_name("swapnaindiaom.co.in") == "swapnaindiaom"

def test_normalize_name_null_handling():
    assert normalize_name(None) == ""
    assert normalize_name("") == ""
    assert normalize_name("nan") == ""
    assert normalize_name("NaN") == ""
    assert normalize_name("null") == ""
    assert normalize_name("<null>") == ""
    assert normalize_name("None") == ""
    assert normalize_name("-") == ""

def test_strip_legal_suffixes():
    # End suffixes
    assert strip_legal_suffixes("premier star payment llc") == "premier star payment"
    assert strip_legal_suffixes("gsk sources private limited") == "gsk sources"
    assert strip_legal_suffixes("znb club sarl") == "znb club"
    # Inverted start suffixes
    assert strip_legal_suffixes("llc hernandez colonial redwood") == "hernandez colonial redwood"
    assert strip_legal_suffixes("pvt efs print ventures ltd") == "efs print ventures"

def test_normalize_name_clean_alignment():
    # Both forward and inverted suffix align to the exact same clean name
    forward = normalize_name_clean("Hernandez Colonial Redwood LLC")
    inverted = normalize_name_clean("LLC Hernandez Colonial Redwood")
    assert forward == inverted == "hernandez colonial redwood"

# ---------------------------------------------------------------------------
# Test Business Address Normalization
# ---------------------------------------------------------------------------

def test_normalize_address_road_abbreviations():
    assert normalize_address("59 Robbins Street, Spencer, TN") == "59 robbins street spencer tn"
    assert normalize_address("Robbins St, Spence CDP, Tennessee") == "robbins street spence cdp tennessee"
    assert normalize_address("421 6th Avenue, Washburn, WI") == "421 6th avenue washburn wi"
    assert normalize_address("Mack Rd, Haltom City, Texas") == "mack road haltom city texas"

def test_normalize_address_french():
    assert "rue de dieppe" in normalize_address("63 R. DE DIEPPE, LILLE, Hauts-de-France")
    assert "boulevard du president" in normalize_address("154 BD du President Wilson, Nouvelle-Aquitaine")
    assert "allee des hetres" in normalize_address("NO. 5 ALLÉE DES HÊTRES, Pornic")

def test_normalize_address_unit_and_landmark():
    assert "suite 9" in normalize_address("665 Locust Street, Ste 9, Canton, IL")
    assert "apartment 4" in normalize_address("12 Main St, Apt 4, Boston, MA")
    assert "opposite sbi" in normalize_address("Opp SBI, MG Road, Pune, MH")
    assert "phase 2" in normalize_address("Sector 19, Ph 2, Gurugram, HR")

def test_normalize_address_synthetic_noise():
    # Preceding hashes before numbers
    assert "8 willow oak lane" in normalize_address("##8 Willow Oak Lane, Saint Louis, Missouri")
    assert "278" in normalize_address("###278, Udyog Vihar, Gurgaon")
    assert "d-51" in normalize_address("D-##51, Sector 19, Vashi")

def test_normalize_address_literal_null_tokens():
    raw = "N335 BEAR TRAIL RD, <NULL>, POYNETTE, WI"
    cleaned = normalize_address(raw)
    assert "<null>" not in cleaned
    assert "null" not in cleaned
    assert "bear trail road" in cleaned

def test_normalize_address_empty_and_null():
    assert normalize_address(None) == ""
    assert normalize_address("") == ""
    assert normalize_address("nan") == ""
    assert normalize_address("<null>") == ""
    assert normalize_address("N/A") == ""

# ---------------------------------------------------------------------------
# Test Numeric & Postal Code Extraction
# ---------------------------------------------------------------------------

def test_extract_postal_code():
    assert extract_postal_code("1795 Westchester Drive, High Point, NC 27262") == "27262"
    assert extract_postal_code("Dayton, OH 45402-1234") == "45402-1234"
    assert extract_postal_code("Andheri West, Mumbai 400053, Maharashtra") == "400053"
    assert extract_postal_code("75001 Paris, France") == "75001"
    assert extract_postal_code("No postal code here") == ""
    assert extract_postal_code(None) == ""

def test_extract_numeric_tokens():
    tokens = extract_numeric_tokens("Plot No 14, 41st Cross, 22nd Main 9th Block, PIN 560041")
    assert "14" in tokens
    assert "41st" in tokens or "41" in tokens
    assert "560041" in tokens

# ---------------------------------------------------------------------------
# Test Full Derived Record Generation
# ---------------------------------------------------------------------------

def test_get_derived_record():
    rec = get_derived_record(
        entity_id="S1-00001",
        business_name="Premier Star Payment L.L.C.",
        business_address="665 Locust Street, Ste 9, Canton, IL 61520",
        country="US",
    )
    assert rec["entity_id"] == "S1-00001"
    assert rec["raw_name"] == "Premier Star Payment L.L.C."
    assert rec["country"] == "US"
    assert rec["norm_name"] == "premier star payment llc"
    assert rec["norm_name_clean"] == "premier star payment"
    assert "locust street" in rec["norm_address"]
    assert rec["postal_code"] == "61520"
    assert not rec["has_missing_address"]
    assert "premier" in rec["norm_name_clean"]

def test_get_derived_record_missing_address():
    rec = get_derived_record(
        entity_id="S2-99999",
        business_name="Twisted Diner LLC Center",
        business_address="nan",
        country="US",
    )
    assert rec["has_missing_address"] is True
    assert rec["norm_address"] == ""
    assert rec["postal_code"] == ""
