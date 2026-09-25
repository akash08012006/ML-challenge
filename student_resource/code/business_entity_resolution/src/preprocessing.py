"""Text normalization - unicode-safe, open-set, preserves distinguishing info."""
import re
import pandas as pd

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WS_RE = re.compile(r"\s+")

# Abbreviation normalisation (applied on token level, after lowercasing).
_ABBR = {
    "corporation": "corp",
    "incorporated": "inc",
    "company": "co",
    "limited": "ltd",
    "private": "pvt",
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "hospital": "hosp",  # applied both sides so safe
}

# precompiled word-boundary regexes for vectorized abbr mapping
_ABBR_RES = [(re.compile(r"\b%s\b" % re.escape(k)), v) for k, v in _ABBR.items()]

# NOTE: we intentionally keep numbers (e.g. "ABC Hospital 2" vs "ABC Hospital")
# and do NOT strip legal suffixes entirely - only normalise variants.


def normalize_text(s: str) -> str:
    if s is None or (isinstance(s, float) and str(s) == "nan"):
        return ""
    s = str(s).lower()
    s = s.replace("&", " and ")
    # unicode-safe: replace ascii punctuation only, keep Devanagari etc.
    s = re.sub(r"[!\"#$%'()*+,\-./:;<=>?@\[\\\]^_`{|}~]", " ", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return ""
    toks = s.split(" ")
    toks = [_ABBR.get(t, t) for t in toks]
    return " ".join(t for t in toks if t)


def normalize_series(s: pd.Series) -> pd.Series:
    """Vectorized normalization (C-level string ops, unicode-safe)."""
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace("&", " and ", regex=False)
    # ascii punctuation -> space (keeps Hindi/unicode letters intact)
    s = s.str.replace(r"[\(\)\[\]!\"#\$%'\(\)\*\+,\-\./:;<=>\?@\\\^_`\{\|\}~]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    for pat, rep in _ABBR_RES:
        s = s.str.replace(pat, rep, regex=True)
    return s


def normalize_country(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip().lower()


def normalize_country_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower()


def blocking_text(name: str, address: str) -> str:
    """Text used for TF-IDF blocking. Name weighted 2x."""
    n = normalize_text(name)
    a = normalize_text(address)
    if n and a:
        return f"{n} {n} {a}"
    return n or a
