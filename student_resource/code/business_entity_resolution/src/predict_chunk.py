"""Chunked inference: process one slice of test S1 to fit in timeout.

Usage:
  python -X utf8 -m code.business_entity_resolution.src.predict_chunk --country india --start 0 --end 100000
Saves output/parts/matching_<country>_<start>_<end>.tsv and candidate_*.tsv
Merge with merge_parts.py
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd

from . import config
from .data_loader import load_source, collect_s23_for_country
from .blocking import block_all
from .features import pairs_to_matrix
from .preprocessing import normalize_series, normalize_country_series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True, help="normalized country, e.g. india, us, france")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=100000)
    args = ap.parse_args()

    print(f"Chunk {args.country} [{args.start}:{args.end}]", flush=True)
    s1 = load_source(os.path.join(config.TEST_DIR, "test_source1.tsv"))
    s1["_cc"] = normalize_country_series(s1["country"])
    sub_all = s1[s1["_cc"] == args.country].copy()
    print(f"Country {args.country}: total S1={len(sub_all)}", flush=True)
    sub = sub_all.iloc[args.start:args.end].copy()
    if len(sub) == 0:
        print("empty slice, nothing to do", flush=True)
        return
    print(f"Slice S1={len(sub)}", flush=True)
    sub["_nn"] = normalize_series(sub["business_name"])
    sub["_na"] = normalize_series(sub["business_address"])

    bundle = joblib.load(config.MODEL_PATH)
    model = bundle["model"]
    thr = float(bundle.get("threshold", config.THRESHOLD))
    print(f"thr={thr}", flush=True)

    s2_path = os.path.join(config.TEST_DIR, "test_source2.tsv")
    s3_path = os.path.join(config.TEST_DIR, "test_source3.tsv")
    print("Collecting S23...", flush=True)
    s23_c = collect_s23_for_country(s2_path, s3_path, args.country)
    print(f"S23={len(s23_c)}", flush=True)
    s23_c["_nn"] = normalize_series(s23_c["business_name"])
    s23_c["_na"] = normalize_series(s23_c["business_address"])
    s23_c["_ccn"] = normalize_country_series(s23_c["country"])

    print("Blocking...", flush=True)
    cands, cos_map = block_all(
        sub[["entity_id", "business_name", "business_address", "country"]].copy(),
        s23_c[["entity_id", "business_name", "business_address", "country"]].copy(),
        top_k=config.TOP_K, batch_size=config.BATCH_SIZE,
    )
    print(f"cands={sum(len(v) for v in cands.values())}", flush=True)

    s23_lookup = {str(e): (nn, na, ccn) for e, nn, na, ccn in zip(s23_c["entity_id"].astype(str), s23_c["_nn"], s23_c["_na"], s23_c["_ccn"])}
    s1_lookup = {str(e): (nn, na, ccn) for e, nn, na, ccn in zip(sub["entity_id"].astype(str), sub["_nn"], sub["_na"], sub["_cc"])}
    del s23_c

    pairs = [(sid, cid) for sid, clist in cands.items() for cid in clist]
    print(f"pairs={len(pairs)} inferring...", flush=True)
    CHUNK = 200000
    proba = np.zeros(len(pairs), dtype=np.float32)
    for st in range(0, len(pairs), CHUNK):
        en = min(len(pairs), st + CHUNK)
        X = pairs_to_matrix(pairs[st:en], s1_lookup, s23_lookup, cos_map)
        proba[st:en] = model.predict_proba(X)[:, 1]
        print(f"  {en}/{len(pairs)}", flush=True)
        del X

    proba_map = {}
    idx = 0
    # need mapping from pair order: pairs list order is cands insertion order; rebuild per sid
    # pairs built as [(sid,cid)...] in cands order, proba aligned
    for (sid, cid), p in zip(pairs, proba.tolist()):
        proba_map.setdefault(sid, {})[cid] = float(p)

    os.makedirs(os.path.join(config.OUTPUT_DIR, "parts"), exist_ok=True)
    mp = os.path.join(config.OUTPUT_DIR, "parts", f"matching_{args.country}_{args.start}_{args.end}.tsv")
    cp = os.path.join(config.OUTPUT_DIR, "parts", f"candidate_{args.country}_{args.start}_{args.end}.tsv")
    order = sub["entity_id"].astype(str).tolist()
    with open(mp, "w", encoding="utf-8", newline="") as fm, open(cp, "w", encoding="utf-8", newline="") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in order:
            clist = cands.get(sid, [])
            fc.write(f"{sid}\t{','.join(clist)}\n")
            d = proba_map.get(sid, {})
            matched = [cid for cid in clist if d.get(cid, 0.0) >= thr]
            fm.write(f"{sid}\t{','.join(matched)}\n")
    print(f"Wrote {mp} {cp}", flush=True)


if __name__ == "__main__":
    main()
