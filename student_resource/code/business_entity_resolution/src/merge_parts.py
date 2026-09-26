"""Merge output/parts/*.tsv into output/matching_results.tsv + candidate_pairs.tsv in test S1 order."""
import os
import glob
import pandas as pd

from . import config
from .data_loader import load_source


def main():
    s1 = load_source(os.path.join(config.TEST_DIR, "test_source1.tsv"))
    order = s1["entity_id"].astype(str).tolist()
    print(f"S1={len(order)}", flush=True)

    m_parts = glob.glob(os.path.join(config.OUTPUT_DIR, "parts", "matching_*.tsv"))
    c_parts = glob.glob(os.path.join(config.OUTPUT_DIR, "parts", "candidate_*.tsv"))
    print(f"found {len(m_parts)} matching parts, {len(c_parts)} candidate parts", flush=True)

    m_map, c_map = {}, {}
    for p in m_parts:
        df = pd.read_csv(p, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False)
        for _, r in df.iterrows():
            m_map[str(r["source1_entity_id"])] = str(r["matched_entity_ids"])
    for p in c_parts:
        df = pd.read_csv(p, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False)
        for _, r in df.iterrows():
            c_map[str(r["source1_entity_id"])] = str(r["candidate_entity_ids"])
    print(f"collected m={len(m_map)} c={len(c_map)}", flush=True)
    missing = [s for s in order if s not in m_map]
    print(f"missing={len(missing)}", flush=True)
    if missing:
        print(f"e.g. {missing[:5]}", flush=True)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "matching_results.tsv"), "w", encoding="utf-8", newline="") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in order:
            fm.write(f"{sid}\t{m_map.get(sid, '')}\n")
    with open(os.path.join(config.OUTPUT_DIR, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in order:
            fc.write(f"{sid}\t{c_map.get(sid, '')}\n")
    print("merged", flush=True)


if __name__ == "__main__":
    main()
