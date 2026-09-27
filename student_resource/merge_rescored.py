import pandas as pd

s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id"], encoding="utf-8", dtype=str)
order = s1["entity_id"].astype(str).tolist()
mmap = {}
for c in ["france", "us", "india"]:
    p = "output/parts/rescored_matching_" + c + ".tsv"
    for ch in pd.read_csv(p, sep="\t", encoding="utf-8", dtype=str, keep_default_na=False, chunksize=200000):
        for _, r in ch.iterrows():
            mmap[str(r["source1_entity_id"])] = str(r["matched_entity_ids"])
print("collected", len(mmap))
assert len(mmap) == len(order), (len(mmap), len(order))
with open("output/matching_results.tsv", "w", encoding="utf-8", newline="") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in order:
        f.write(sid + "\t" + mmap.get(sid, "") + "\n")
print("wrote matching_results.tsv")
