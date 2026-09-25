"""Pairwise similarity features (fast, dependency-light).

Uses rapidfuzz (C++ backend) + token Jaccard + length stats.
All features in [0,1] or small ints; no one-hot on country (open set).
"""
import numpy as np
from rapidfuzz import fuzz

from .preprocessing import normalize_text, normalize_country


def jaccard(a: str, b: str) -> float:
    sa = set(a.split()) if a else set()
    sb = set(b.split()) if b else set()
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def common_tokens(a: str, b: str) -> int:
    if not a or not b:
        return 0
    return len(set(a.split()) & set(b.split()))


FEATURE_NAMES = [
    "name_exact",
    "name_jaccard",
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "addr_exact",
    "addr_jaccard",
    "addr_ratio",
    "addr_token_set",
    "country_exact",
    "name_len_diff",
    "addr_len_diff",
    "common_name_tokens",
    "common_addr_tokens",
    "tfidf_cosine",
]


def pair_features_norm(nn1, na1, cc1, nn2, na2, cc2, cosine=0.0):
    """Inputs already normalized (nn/na lowercased, cc lowercased)."""
    try:
        name_ratio = fuzz.ratio(nn1, nn2) / 100.0
    except Exception:
        name_ratio = 0.0
    try:
        name_ts = fuzz.token_sort_ratio(nn1, nn2) / 100.0
    except Exception:
        name_ts = 0.0
    try:
        name_tset = fuzz.token_set_ratio(nn1, nn2) / 100.0
    except Exception:
        name_tset = 0.0
    try:
        name_part = fuzz.partial_ratio(nn1, nn2) / 100.0
    except Exception:
        name_part = 0.0
    try:
        addr_ratio = fuzz.ratio(na1, na2) / 100.0
    except Exception:
        addr_ratio = 0.0
    try:
        addr_tset = fuzz.token_set_ratio(na1, na2) / 100.0
    except Exception:
        addr_tset = 0.0
    return [
        1.0 if nn1 == nn2 and nn1 != "" else 0.0,
        jaccard(nn1, nn2),
        name_ratio,
        name_ts,
        name_tset,
        name_part,
        1.0 if na1 == na2 and na1 != "" else 0.0,
        jaccard(na1, na2),
        addr_ratio,
        addr_tset,
        1.0 if cc1 == cc2 else 0.0,
        abs(len(nn1) - len(nn2)),
        abs(len(na1) - len(na2)),
        float(common_tokens(nn1, nn2)),
        float(common_tokens(na1, na2)),
        float(cosine),
    ]


def pair_features(n1, a1, c1, n2, a2, c2, cosine=0.0):
    from .preprocessing import normalize_text, normalize_country
    nn1, nn2 = normalize_text(n1), normalize_text(n2)
    na1, na2 = normalize_text(a1), normalize_text(a2)
    cc1, cc2 = normalize_country(c1), normalize_country(c2)
    return pair_features_norm(nn1, na1, cc1, nn2, na2, cc2, cosine)


def pairs_to_matrix(pairs, s1_lookup, s23_lookup, cos_map):
    """pairs: list of (s1_id, c_id). lookups: id -> (nn, na, cc) NORMALIZED.

    For backward compat, if lookup values look raw (contain uppercase/punct),
    they are used as-is (caller should pre-normalize for speed).
    """
    X = np.zeros((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
    for i, (s1, c) in enumerate(pairs):
        nn1, na1, cc1 = s1_lookup[s1]
        nn2, na2, cc2 = s23_lookup[c]
        X[i, :] = pair_features_norm(nn1, na1, cc1, nn2, na2, cc2, cos_map.get((s1, c), 0.0))
    return X
