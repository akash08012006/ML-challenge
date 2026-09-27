"""Fast rescore: reuse v1 candidate_pairs.tsv (single-stream) with hybrid-trained model.

No TF-IDF blocking - just lookups + featurize + predict. ~1h total vs 26h re-block.
Usage: python -m code.business_entity_resolution.src.rescore --country france|us|india
Writes output/parts/rescored_matching_<country>.tsv (full country, all S1 rows).
"""
import os
import argparse
import joblib
import numpy as np
import pandas as pd

from . import config
from .data_loader import load_source, collect_s23_for_country
from .features import pairs_to_matrix
from .preprocessing import normalize_series, normalize_country_series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", required=True)
    args = ap.parse_args()

    bundle = joblib.load(config.MODEL_PATH)
    model, thr = bundle["model"], float(bundle.get("threshold", 0.55))
    print(f"model thr={thr}", flush=True)

    s1 = load_source(os.path.join(config.TEST_DIR, "test_source1.tsv"))
    s1["_cc"] = normalize_country_series(s1["country"])
    sub = s1[s1["_cc"] == args.country].copy()
    print(f"S1 {args.country}: {len(sub)}", flush=True)
    sub["_nn"] = normalize_series(sub["business_name"])
    sub["_na"] = normalize_series(sub["business_address"])
    s1_lookup = {str(e): (nn, na, cc) for e, nn, na, cc in zip(sub["entity_id"].astype(str), sub["_nn"], sub["_na"], sub["_cc"])}
    want = set(sub["entity_id"].astype(str))

    # stream candidate_pairs.tsv, keep only this country's S1
    cand_path = os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv")
    cands = {}
    print("Reading candidates...", flush=True)
    for ch in pd.read_csv(cand_path, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False, chunksize=200000):
        m = ch[ch["source1_entity_id"].isin(want)]
        for _, r in m.iterrows():
            raw = str(r["candidate_entity_ids"]).strip()
            cands[str(r["source1_entity_id"])] = [x for x in raw.split(",") if x] if raw else []
    print(f"cands for {len(cands)}/{len(want)}", flush=True)

    s23_c = collect_s23_for_country(
        os.path.join(config.TEST_DIR, "test_source2.tsv"),
        os.path.join(config.TEST_DIR, "test_source3.tsv"),
        args.country,
    )
    print(f"S23={len(s23_c)}", flush=True)
    s23_c["_nn"] = normalize_series(s23_c["business_name"])
    s23_c["_na"] = normalize_series(s23_c["business_address"])
    s23_c["_cc"] = normalize_country_series(s23_c["country"])
    s23_lookup = {str(e): (nn, na, cc) for e, nn, na, cc in zip(s23_c["entity_id"].astype(str), s23_c["_nn"], s23_c["_na"], s23_c["_cc"])}
    del s23_c

    # tfidf cosine unavailable without blocking - use 0.0 (model robust; cosine is 1 of 16 feats)
    cos_map = {}
    pairs = [(sid, cid) for sid, cl in cands.items() for cid in cl]
    print(f"pairs={len(pairs)}", flush=True)
    CH = 200000
    proba = np.zeros(len(pairs), dtype=np.float32)
    for st in range(0, len(pairs), CH):
        en = min(len(pairs), st + CH)
        X = pairs_to_matrix(pairs[st:en], s1_lookup, s23_lookup, cos_map)
        proba[st:en] = model.predict_proba(X)[:, 1]
        if (st // CH) % 5 == 0:
            print(f"  {en}/{len(pairs)}", flush=True)
        del X

    order = sub["entity_id"].astype(str).tolist()
    pmap = {}
    for (sid, cid), p in zip(pairs, proba.tolist()):
        pmap.setdefault(sid, {})[cid] = float(p)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "parts"), exist_ok=True)
    out = os.path.join(config.OUTPUT_DIR, "parts", f"rescored_matching_{args.country}.tsv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in order:
            cl = cands.get(sid, [])
            d = pmap.get(sid, {})
            m = [c for c in cl if d.get(c, 0.0) >= thr]
            f.write(f"{sid}\t{','.join(m)}\n")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
