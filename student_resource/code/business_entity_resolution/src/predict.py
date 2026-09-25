"""Inference per-country streaming: blocking -> chunked predict -> TSVs."""
import os
import joblib
import numpy as np
import pandas as pd

from . import config
from .data_loader import load_source, collect_s23_for_country
from .blocking import block_all
from .features import pairs_to_matrix
from .preprocessing import normalize_series, normalize_country_series


def _load_model():
    bundle = joblib.load(config.MODEL_PATH)
    return bundle["model"], float(bundle.get("threshold", config.THRESHOLD))


def main():
    print(f"Test dir: {config.TEST_DIR}", flush=True)
    s1 = load_source(os.path.join(config.TEST_DIR, "test_source1.tsv"))
    print(f"S1={len(s1)}", flush=True)
    s1["_cc"] = normalize_country_series(s1["country"])
    s1["_nn"] = normalize_series(s1["business_name"])
    s1["_na"] = normalize_series(s1["business_address"])
    s1_order = s1["entity_id"].astype(str).tolist()
    s1_lookup_all = {str(e): (nn, na, cc) for e, nn, na, cc in zip(s1["entity_id"].astype(str), s1["_nn"], s1["_na"], s1["_cc"])}

    model, thr = _load_model()
    print(f"Model thr={thr}", flush=True)

    s2_path = os.path.join(config.TEST_DIR, "test_source2.tsv")
    s3_path = os.path.join(config.TEST_DIR, "test_source3.tsv")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    match_path = os.path.join(config.OUTPUT_DIR, "matching_results.tsv")
    cand_path = os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv")

    # open outputs, write headers, then per-country append (keep order for final? collect then sort by s1_order)
    final_cands = {}
    final_proba = {}  # sid -> {cid: prob}

    for cc in sorted(s1["_cc"].unique().tolist()):
        sub1 = s1[s1["_cc"] == cc]
        print(f"Country '{cc}': S1={len(sub1)}", flush=True)
        s23_c = collect_s23_for_country(s2_path, s3_path, cc)
        print(f"  S23={len(s23_c)}", flush=True)
        s23_c["_nn"] = normalize_series(s23_c["business_name"])
        s23_c["_na"] = normalize_series(s23_c["business_address"])
        s23_c["_ccn"] = normalize_country_series(s23_c["country"])
        cands, cos_map = block_all(sub1[["entity_id", "business_name", "business_address", "country"]].copy(), s23_c[["entity_id", "business_name", "business_address", "country"]].copy(), top_k=config.TOP_K, batch_size=config.BATCH_SIZE)
        print(f"  cands={sum(len(v) for v in cands.values())}", flush=True)
        s23_lookup = {str(e): (nn, na, ccn) for e, nn, na, ccn in zip(s23_c["entity_id"].astype(str), s23_c["_nn"], s23_c["_na"], s23_c["_ccn"])}
        s1_lookup = {str(e): (nn, na, ccn) for e, nn, na, ccn in zip(sub1["entity_id"].astype(str), sub1["_nn"], sub1["_na"], sub1["_cc"])}
        pairs = [(sid, cid) for sid, clist in cands.items() for cid in clist]
        print(f"  pairs={len(pairs)} inferring...", flush=True)
        CHUNK = 200000
        proba = np.zeros(len(pairs), dtype=np.float32)
        for st in range(0, len(pairs), CHUNK):
            en = min(len(pairs), st + CHUNK)
            X = pairs_to_matrix(pairs[st:en], s1_lookup, s23_lookup, cos_map)
            proba[st:en] = model.predict_proba(X)[:, 1]
            if (st // CHUNK) % 5 == 0:
                print(f"    {en}/{len(pairs)}", flush=True)
            del X
        for (sid, cid), p in zip(pairs, proba.tolist()):
            final_proba.setdefault(sid, {})[cid] = float(p)
        for sid, clist in cands.items():
            final_cands[sid] = clist
        del s23_c, s23_lookup, cands, cos_map, pairs, proba

    with open(match_path, "w", encoding="utf-8", newline="") as fm, open(cand_path, "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_order:
            clist = final_cands.get(sid, [])
            fc.write(f"{sid}\t{','.join(clist)}\n")
            d = final_proba.get(sid, {})
            matched = [cid for cid in clist if d.get(cid, 0.0) >= thr]
            fm.write(f"{sid}\t{','.join(matched)}\n")
    print(f"Wrote {match_path} and {cand_path}", flush=True)


if __name__ == "__main__":
    main()
