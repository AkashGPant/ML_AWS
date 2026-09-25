"""
src/normalize.py: Deterministic, country-agnostic normalization pipeline
for the AWS ML Hackathon 2026 Business Entity Resolution Challenge.

Provides evidence-driven normalization functions:
- transliterate_indic: Universal Indic-to-Latin transliteration for 9 scripts
- strip_accents: NFKD decomposition removing combining marks (French, etc.)
- normalize_name: Deterministic business name normalization
- strip_legal_suffixes: Legal suffix stripping & standardization (front or back)
- extract_dba_name: Trade name / DBA parsing
- clean_domain_name: Web domain artifact stripping (.com, .org, www, etc.)
- normalize_address: Deterministic address normalization & abbreviation expansion
- extract_numeric_tokens: House/plot/unit number extraction
- extract_postal_code: Country-agnostic postal / PIN code extraction
- get_derived_record: Full record derivation for fast Parquet storage
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# 1. Indic Scripts Unicode Mapping & Transliteration
# ---------------------------------------------------------------------------

# Unicode offsets relative to script base (e.g. 0x0900 for Devanagari)
INDIC_OFFSET_TO_LATIN = {
    0x02: "n",   # Anusvara
    0x03: "h",   # Visarga
    0x05: "a",   # Short A
    0x06: "aa",  # AA
    0x07: "i",   # I
    0x08: "ee",  # II
    0x09: "u",   # U
    0x0A: "oo",  # UU
    0x0B: "ri",  # Vocalic R
    0x0E: "e",   # Short E
    0x0F: "e",   # E
    0x10: "ai",  # AI
    0x12: "o",   # Short O
    0x13: "o",   # O
    0x14: "au",  # AU
    0x15: "k",
    0x16: "kh",
    0x17: "g",
    0x18: "gh",
    0x19: "ng",
    0x1A: "ch",
    0x1B: "chh",
    0x1C: "j",
    0x1D: "jh",
    0x1E: "ny",
    0x1F: "t",
    0x20: "th",
    0x21: "d",
    0x22: "dh",
    0x23: "n",
    0x24: "t",
    0x25: "th",
    0x26: "d",
    0x27: "dh",
    0x28: "n",
    0x2A: "p",
    0x2B: "ph",
    0x2C: "b",
    0x2D: "bh",
    0x2E: "m",
    0x2F: "y",
    0x30: "r",
    0x32: "l",
    0x33: "l",
    0x35: "v",
    0x36: "sh",
    0x37: "sh",
    0x38: "s",
    0x39: "h",
    # Matras (vowel signs)
    0x3E: "aa",
    0x3F: "i",
    0x40: "ee",
    0x41: "u",
    0x42: "oo",
    0x43: "ri",
    0x46: "e",
    0x47: "e",
    0x48: "ai",
    0x4A: "o",
    0x4B: "o",
    0x4C: "au",
    0x4D: "",    # Virama / Halant
}

INDIC_CONSONANTS = set(range(0x15, 0x3A))
INDIC_MATRAS = set(range(0x3E, 0x4D)) | {0x02, 0x03}

SCRIPT_BASES = (
    0x0900,  # Devanagari (Hindi, Marathi, Sanskrit)
    0x0980,  # Bengali / Assamese
    0x0A00,  # Gurmukhi (Punjabi)
    0x0A80,  # Gujarati
    0x0B00,  # Oriya / Odia
    0x0B80,  # Tamil
    0x0C00,  # Telugu
    0x0C80,  # Kannada
    0x0D00,  # Malayalam
)

INDIC_RANGE_CHECK = re.compile(r"[\u0900-\u0D7F]")

def transliterate_indic(text: str) -> str:
    """Transliterates text containing any of 9 Indic scripts into Latin alphabet."""
    if not text or not INDIC_RANGE_CHECK.search(text):
        return text

    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        cp = ord(text[i])
        matched_base = None
        for b in SCRIPT_BASES:
            if b <= cp < b + 0x80:
                matched_base = b
                break

        if matched_base is None:
            out.append(text[i])
            i += 1
            continue

        offset = cp - matched_base
        char_trans = INDIC_OFFSET_TO_LATIN.get(offset, "")

        if offset in INDIC_CONSONANTS:
            next_is_virama = False
            next_is_matra = False
            if i + 1 < n:
                next_cp = ord(text[i + 1])
                if matched_base <= next_cp < matched_base + 0x80:
                    next_offset = next_cp - matched_base
                    if next_offset == 0x4D:  # Virama
                        next_is_virama = True
                        i += 1
                    elif next_offset in INDIC_MATRAS:
                        next_is_matra = True

            out.append(char_trans)
            if not next_is_virama and not next_is_matra:
                if i + 1 < n and (text[i + 1].isalpha() or ord(text[i + 1]) >= 0x0900):
                    out.append("a")
        else:
            out.append(char_trans)
        i += 1

    return "".join(out)

# ---------------------------------------------------------------------------
# 2. Accent & Diacritics Stripping
# ---------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """Removes combining diacritical marks (e.g. accents in French: é, à, ç, etc.)."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))

# ---------------------------------------------------------------------------
# 3. Clean Known Noise Patterns & Artifacts
# ---------------------------------------------------------------------------

RE_LITERAL_NULL = re.compile(r"^(?:<null>|<nan>|null|none|nan|n/a|na|-|undefined|\s*)$", re.IGNORECASE)
RE_COUNTRY_TAGS = re.compile(r"\s*\((?:india|france|us|usa|u\.s\.|u\.s\.a\.)\)\s*", re.IGNORECASE)
RE_LEADING_DECORATIVE = re.compile(r"^[\s*#~<>\-=_+!]+|[\s*#~<>\-=_+!]+$")
RE_DOUBLE_HASH = re.compile(r"[#]{2,}")
RE_MULTI_STAR = re.compile(r"[*]{2,}")
RE_MULTI_DASH = re.compile(r"[-]{2,}")
RE_BRACKETS_AROUND_WORD = re.compile(r"\[([a-zA-Z0-9\s.&'-]+)\]")
RE_ANGLE_BRACKETS = re.compile(r"<<|>>")
RE_EXTRA_SPACES = re.compile(r"\s+")

# DBA / Trade Name Pattern
RE_DBA = re.compile(
    r"\b(?:dba|d/b/a|d\.b\.a\.|t/a|trading as|fka|f/k/a|aka|a/k/a)\b[:\s]*",
    re.IGNORECASE,
)

# Web Domain Pattern in Business Names
RE_DOMAIN = re.compile(
    r"\b(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+)\.(?:com|org|net|in|co\.in|fr|io|biz|info|co)\b",
    re.IGNORECASE,
)

# Legal Suffixes across US, India, France
LEGAL_SUFFIX_TOKENS = [
    # Multi-token legal phrases
    r"private\s+limited",
    r"pvt\s+ltd",
    r"pvt\s+limited",
    r"private\s+ltd",
    r"praivet\s+limited",
    r"pra\s+li",
    # Single legal terms
    r"incorporated",
    r"corporation",
    r"company",
    r"limited",
    r"corp",
    r"inc",
    r"llc",
    r"pllc",
    r"ltd",
    r"pvt",
    r"llp",
    r"sarl",
    r"sasu",
    r"sas",
    r"eurl",
    r"snc",
    r"sci",
    r"gie",
    r"selarl",
]

# Regex for stripping suffixes at end
RE_SUFFIX_END = re.compile(
    r"\b(?:" + "|".join(LEGAL_SUFFIX_TOKENS) + r")\.?\s*$",
    re.IGNORECASE,
)

# Regex for stripping inverted suffixes at beginning (e.g. "LLC Hernandez Colonial Redwood")
RE_SUFFIX_START = re.compile(
    r"^\s*(?:" + "|".join(LEGAL_SUFFIX_TOKENS) + r")\.?\b[\s:-]*",
    re.IGNORECASE,
)

def extract_dba_name(name: str) -> str:
    """If a DBA / trade name pattern is present, returns the entity name after the marker."""
    if not name:
        return ""
    m = RE_DBA.search(name)
    if m:
        after = name[m.end():].strip()
        if after:
            return after
    return name

def clean_domain_name(name: str) -> str:
    """If a name is or contains a domain (e.g. 'heartinstitutecity.com'), cleans it to root."""
    if not name:
        return ""
    m = RE_DOMAIN.search(name)
    if m:
        # Replace the domain with its root name token
        return RE_DOMAIN.sub(r"\1", name)
    return name

def strip_legal_suffixes(name: str) -> str:
    """Removes legal suffixes whether they appear at the end or inverted at the beginning."""
    if not name:
        return ""
    s = name.strip()
    # Strip at end repeatedly (handles cases like "Co Ltd" or "Pvt Ltd")
    for _ in range(2):
        s = RE_SUFFIX_END.sub("", s).strip()
    # Strip inverted suffix at start
    s = RE_SUFFIX_START.sub("", s).strip()
    return s

def normalize_name(name: Optional[str]) -> str:
    """
    Deterministic, country-agnostic business name normalization:
    1. Null & literal-null detection
    2. Indic transliteration
    3. Accent / diacritics stripping via NFKD
    4. Decorative noise removal (***, ###, <<>>, leading #)
    5. Brackets & country artifact removal
    6. DBA / domain cleaning
    7. Punctuation standardization (& -> and, dots removed from acronyms)
    8. Whitespace collapsing & lowercasing
    """
    if name is None:
        return ""
    name_str = str(name).strip()
    if not name_str or RE_LITERAL_NULL.match(name_str):
        return ""

    # 1. Transliterate Indic scripts
    s = transliterate_indic(name_str)

    # 2. Strip Latin accents / diacritics
    s = strip_accents(s)

    # 3. Clean decorative characters & artifacts
    s = RE_ANGLE_BRACKETS.sub(" ", s)
    s = RE_BRACKETS_AROUND_WORD.sub(r" \1 ", s)
    s = RE_COUNTRY_TAGS.sub(" ", s)
    s = RE_MULTI_STAR.sub(" ", s)
    s = RE_MULTI_DASH.sub(" ", s)
    s = RE_DOUBLE_HASH.sub(" ", s)

    # 4. Handle DBA & domains
    s = extract_dba_name(s)
    s = clean_domain_name(s)

    # 5. Punctuation: & -> and, remove leading # from words
    s = re.sub(r"&", " and ", s)
    s = re.sub(r"#([a-zA-Z0-9])", r"\1", s)

    # 6. Lowercase
    s = s.lower()

    # 7. Standardize acronym dots: p.v.t. -> pvt, l.l.c. -> llc, u.s. -> us
    s = re.sub(r"\b([a-z])\.([a-z])\.([a-z])\.", r"\1\2\3", s)
    s = re.sub(r"\b([a-z])\.([a-z])\.", r"\1\2", s)

    # 8. Clean non-alphanumeric punctuation except space
    s = re.sub(r"[^\w\s]", " ", s)

    # 9. Collapse whitespace
    s = RE_EXTRA_SPACES.sub(" ", s).strip()
    return s

def normalize_name_clean(name: Optional[str]) -> str:
    """Normalizes name and strips legal suffixes at start/end for core entity alignment."""
    norm = normalize_name(name)
    return strip_legal_suffixes(norm)

# ---------------------------------------------------------------------------
# 4. Address Normalization & Expansion
# ---------------------------------------------------------------------------

# Common road & unit abbreviations mapping across US, India, France
ROAD_ABBREVIATIONS = {
    # Road types
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbd\b": "boulevard",
    r"\bbvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bcir\b": "circle",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b": "highway",
    r"\br\b": "rue",
    r"\ball\b": "allee",
    r"\brte\b": "route",
    r"\bimp\b": "impasse",
    r"\bche\b": "chemin",
    # Units / Floors / Building
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\baptt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bbldg\b": "building",
    r"\bbat\b": "building",
    r"\brm\b": "room",
    r"\bno\b": "number",
    # India address markers
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\btq\b": "taluk",
    r"\bdist\b": "district",
    r"\bsec\b": "sector",
    r"\bph\b": "phase",
    r"\bmkt\b": "market",
    r"\bp\.?o\.?\s*box\b": "pobox",
}

# NOTE (STATE_MAPPING — intentionally NOT applied inside normalize_address):
# A naive token-substitution of two-letter state abbreviations inside a full
# address string would produce false positives: "in" matches INDIANA inside
# words like "Main" or "Building", "or" matches OREGON inside "floor", etc.
# The COMPILED_ROAD_ABBR patterns already use \b word-boundary anchors which
# mitigates the worst cases for road/unit abbreviations, but the same trick
# cannot be applied safely to arbitrary two-letter state codes without a full
# address parser.
#
# Decision: STATE_MAPPING is preserved here as a reference lookup table for
# Phase 1 feature engineering (e.g. canonicalising the last token of a
# pre-split address field).  It is NOT used in normalize_address() so that
# existing audit-verified normalisation outputs remain unchanged.
STATE_MAPPING = {
    # US States
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey",
    "nm": "new mexico", "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    # India States
    "mh": "maharashtra", "ka": "karnataka", "tn": "tamil nadu", "tg": "telangana", "ts": "telangana",
    "ap": "andhra pradesh", "gj": "gujarat", "wb": "west bengal", "up": "uttar pradesh", "dl": "delhi",
    "hr": "haryana", "rj": "rajasthan", "kl": "kerala", "mp": "madhya pradesh", "pb": "punjab",
    "br": "bihar", "od": "odisha", "as": "assam", "jh": "jharkhand", "ch": "chandigarh",
    "uk": "uttarakhand", "hp": "himachal pradesh", "ga": "goa",
}

COMPILED_ROAD_ABBR = [
    (re.compile(pattern, re.IGNORECASE), repl)
    for pattern, repl in ROAD_ABBREVIATIONS.items()
]

RE_HASH_NUMBERS = re.compile(r"[#]{1,}(\d+)")

def normalize_address(address: Optional[str]) -> str:
    """
    Deterministic, country-agnostic business address normalization:
    1. Null & literal-null detection (<null>, nan, etc.)
    2. Indic transliteration (converts Kannada/Marathi/Hindi scripts to Latin)
    3. French accent removal via NFKD
    4. Clean synthetic hashes on numbers (##8 -> 8, ###278 -> 278, D-##51 -> D-51)
    5. Clean noise tokens (<null>, ***)
    6. Expand road / unit / landmark abbreviations
    7. Standardize state abbreviations
    8. Whitespace collapsing & lowercasing
    """
    if address is None:
        return ""
    addr_str = str(address).strip()
    if not addr_str or RE_LITERAL_NULL.match(addr_str):
        return ""

    # 1. Transliterate Indic scripts
    s = transliterate_indic(addr_str)

    # 2. Strip Latin accents / diacritics
    s = strip_accents(s)

    # 3. Clean synthetic hashes before numbers: ##8 -> 8, ###278 -> 278, D-##51 -> D-51
    s = RE_HASH_NUMBERS.sub(r"\1", s)
    s = RE_DOUBLE_HASH.sub(" ", s)
    s = RE_MULTI_STAR.sub(" ", s)

    # 4. Remove literal <null> or <nan> tokens inside addresses
    s = re.sub(r"<\s*null\s*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<\s*nan\s*>", " ", s, flags=re.IGNORECASE)

    # 5. Lowercase
    s = s.lower()

    # 6. Expand road and unit abbreviations
    for pattern, repl in COMPILED_ROAD_ABBR:
        s = pattern.sub(repl, s)

    # 7. Clean non-alphanumeric punctuation except space and hyphen in numbers
    s = re.sub(r"[^\w\s-]", " ", s)
    # Separate standalone hyphens
    s = re.sub(r"(?<!\w)-|-(?!\w)", " ", s)

    # 9. Collapse whitespace
    s = RE_EXTRA_SPACES.sub(" ", s).strip()
    return s

# ---------------------------------------------------------------------------
# 5. Numeric & Postal Code Extraction
# ---------------------------------------------------------------------------

RE_POSTAL_PIN = re.compile(r"\b([1-9]\d{5})\b")
RE_POSTAL_US = re.compile(
    r"(?:,\s*|\b[a-zA-Z]{2}\s+|\b(?:zip|pin|postal|cedex)\s*[:#-]?\s*)(\d{5}(?:-\d{4})?)\b",
    re.IGNORECASE,
)
RE_POSTAL_END = re.compile(r"\b(\d{5})\s*$")
RE_POSTAL_CITY = re.compile(r"\b(\d{5})\s+([a-zA-ZÀ-ÿ-]+)\b", re.IGNORECASE)
RE_ROAD_WORDS = re.compile(r"^(?:st|street|rd|road|ave|avenue|dr|drive|ln|lane|blvd|boulevard|ct|court|hwy|highway|way|pl|place)$", re.IGNORECASE)
RE_ALL_NUMERICS = re.compile(r"\b(?:\d+(?:st|nd|rd|th|[a-zA-Z])?|\d+/\d+)\b", re.IGNORECASE)

def extract_postal_code(address: Optional[str]) -> str:
    """
    Extracts 5-digit (US/France) or 6-digit (India PIN) postal codes.
    Avoids misclassifying 5-digit leading street numbers (e.g. '17560 Ellis Road').
    """
    if not address:
        return ""
    addr = str(address).strip()
    # 1. Indian 6-digit PIN code
    m_pin = RE_POSTAL_PIN.search(addr)
    if m_pin:
        return m_pin.group(1)
    # 2. US ZIP code after state code or comma
    m_zip = RE_POSTAL_US.search(addr)
    if m_zip:
        return m_zip.group(1)
    # 3. Trailing 5-digit postal code at end of address
    m_end = RE_POSTAL_END.search(addr)
    if m_end:
        return m_end.group(1)
    # 4. 5-digit postal code before city name (e.g. '75001 Paris', '62100 Calais')
    for m in RE_POSTAL_CITY.finditer(addr):
        next_word = m.group(2)
        if not RE_ROAD_WORDS.match(next_word):
            return m.group(1)
    return ""

def extract_numeric_tokens(text: Optional[str]) -> List[str]:
    """Extracts house numbers, plot numbers, PIN codes, and alphanumeric digits."""
    if not text:
        return []
    return RE_ALL_NUMERICS.findall(str(text))

# ---------------------------------------------------------------------------
# 6. Structured Tokenization & Derived Record Generation
# ---------------------------------------------------------------------------

STOP_WORDS_NAME: Set[str] = {
    "and", "the", "of", "in", "for", "at", "by", "on", "a", "an", "to",
    "co", "ltd", "inc", "corp", "llc", "pvt", "limited", "company"
}

def tokenize_string(s: str, min_len: int = 2) -> List[str]:
    """Extracts unique sorted significant tokens."""
    if not s:
        return []
    tokens = [w for w in s.split() if len(w) >= min_len and w not in STOP_WORDS_NAME]
    return sorted(set(tokens))

def get_derived_record(
    entity_id: str,
    business_name: Optional[str],
    business_address: Optional[str],
    country: Optional[str],
) -> Dict[str, object]:
    """
    Produces complete raw + derived representation for Parquet export.
    Preserves exact raw inputs while providing fast, deterministic normalized fields.
    """
    raw_name = "" if business_name is None or str(business_name) == "nan" else str(business_name)
    raw_addr = "" if business_address is None or str(business_address) == "nan" else str(business_address)
    raw_ctry = "" if country is None or str(country) == "nan" else str(country)

    norm_name = normalize_name(raw_name)
    norm_name_clean = normalize_name_clean(raw_name)
    norm_addr = normalize_address(raw_addr)

    # name_tokens / addr_tokens were previously computed here but never included
    # in the returned dict or the Parquet schema.  Removed to eliminate ~2×
    # tokenize_string() overhead across every row in the hot loop.

    postal_code = extract_postal_code(raw_addr)

    return {
        "entity_id": str(entity_id),
        "raw_name": raw_name,
        "raw_address": raw_addr,
        "country": raw_ctry,
        "norm_name": norm_name,
        "norm_name_clean": norm_name_clean,
        "norm_address": norm_addr,
        "postal_code": postal_code,
        "has_missing_address": (len(raw_addr.strip()) == 0 or bool(RE_LITERAL_NULL.match(raw_addr))),
    }
