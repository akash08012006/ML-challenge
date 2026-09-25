"""I/O helpers - always sep='\t', utf-8."""
import pandas as pd


def load_source(path, nrows=None):
    return pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False, nrows=nrows)


def load_ground_truth(path, only_ids=None):
    """only_ids: set of S1 to keep (saves RAM when sampling)."""
    if only_ids is None:
        df = pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False)
        out = {}
        for _, row in df.iterrows():
            s1 = str(row["source1_entity_id"]).strip()
            raw = str(row.get("matched_entity_ids", "")).strip()
            out[s1] = {x.strip() for x in raw.split(",") if x.strip()} if raw else set()
        return out
    # streaming to avoid holding 2.2M rows when only sample needed
    out = {}
    want = set(only_ids)
    for chunk in pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False, chunksize=200000):
        for _, row in chunk.iterrows():
            s1 = str(row["source1_entity_id"]).strip()
            if s1 in want:
                raw = str(row.get("matched_entity_ids", "")).strip()
                out[s1] = {x.strip() for x in raw.split(",") if x.strip()} if raw else set()
        if len(out) >= len(want):
            # may still need to scan? break early if all found
            pass
    # ensure all wanted present (singletons missing? gt has row per S1 so should be present)
    for sid in want:
        out.setdefault(sid, set())
    return out


def collect_s23_for_country(s2_path, s3_path, country_norm, chunksize=200000):
    """Stream S2+S3, return DataFrame filtered to one normalized country."""
    from .preprocessing import normalize_country
    parts = []
    for path in (s2_path, s3_path):
        for chunk in pd.read_csv(path, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False, chunksize=chunksize):
            mask = chunk["country"].map(normalize_country) == country_norm
            sub = chunk.loc[mask]
            if len(sub):
                parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    import pandas as _pd
    return _pd.concat(parts, ignore_index=True)
