"""Train: per-country streaming blocking -> features -> HGB -> threshold tune."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from . import config
from .data_loader import load_source, load_ground_truth, collect_s23_for_country
from .blocking import block_all
from .features import pairs_to_matrix, FEATURE_NAMES
from .evaluation import threshold_sweep
from .preprocessing import normalize_series, normalize_country_series


def main():
    print(f"Dataset: {config.TRAIN_DIR}", flush=True)
    s1 = load_source(os.path.join(config.TRAIN_DIR, "train_source1.tsv"))
    print(f"S1={len(s1)}", flush=True)

    all_ids = s1["entity_id"].astype(str).tolist()
    if len(all_ids) > config.MAX_TRAIN_S1:
        cmap = dict(zip(s1["entity_id"].astype(str), s1["country"].astype(str)))
        strat = [cmap[i] for i in all_ids]
        sample_ids, _ = train_test_split(
            all_ids, train_size=config.MAX_TRAIN_S1, random_state=config.RANDOM_STATE, stratify=strat
        )
        sample_set = set(sample_ids)
        s1_sub = s1[s1["entity_id"].astype(str).isin(sample_set)].copy()
        print(f"Sampled {len(s1_sub)} S1", flush=True)
    else:
        s1_sub = s1
        sample_set = set(all_ids)

    gt_sub = load_ground_truth(os.path.join(config.TRAIN_DIR, "train_ground_truth.tsv"), only_ids=sample_set)
    print(f"GT loaded for {len(gt_sub)} sampled S1", flush=True)

    s2_path = os.path.join(config.TRAIN_DIR, "train_source2.tsv")
    s3_path = os.path.join(config.TRAIN_DIR, "train_source3.tsv")

    s1_sub["_cc"] = normalize_country_series(s1_sub["country"])
    s1_sub["_nn"] = normalize_series(s1_sub["business_name"])
    s1_sub["_na"] = normalize_series(s1_sub["business_address"])
    all_cands, all_cos = {}, {}
    s1_lookup = {str(e): (nn, na, cc) for e, nn, na, cc in zip(s1_sub["entity_id"].astype(str), s1_sub["_nn"], s1_sub["_na"], s1_sub["_cc"])}
    s23_lookup = {}

    for cc in sorted(s1_sub["_cc"].unique().tolist()):
        sub1 = s1_sub[s1_sub["_cc"] == cc]
        print(f"Country '{cc}': S1={len(sub1)} - collecting S2/S3...", flush=True)
        s23_c = collect_s23_for_country(s2_path, s3_path, cc)
        print(f"  S23={len(s23_c)}", flush=True)
        s23_c["_nn"] = normalize_series(s23_c["business_name"])
        s23_c["_na"] = normalize_series(s23_c["business_address"])
        s23_c["_cc"] = normalize_country_series(s23_c["country"])
        for e, nn, na, ccn in zip(s23_c["entity_id"].astype(str), s23_c["_nn"], s23_c["_na"], s23_c["_cc"]):
            s23_lookup[str(e)] = (nn, na, ccn)
        print("  blocking...", flush=True)
        # block_all expects raw cols; pass raw business_name/address/country (it normalizes internally vectorized)
        raw_sub1 = sub1[["entity_id", "business_name", "business_address", "country"]].copy()
        cands, cos_map = block_all(raw_sub1, s23_c[["entity_id", "business_name", "business_address", "country"]].copy(), top_k=config.TOP_K, batch_size=config.BATCH_SIZE)
        all_cands.update(cands)
        all_cos.update(cos_map)
        tp = fp = 0
        for sid, clist in cands.items():
            truth = gt_sub.get(sid, set())
            if not truth:
                continue
            tp += len(set(clist) & truth)
            fp += len(truth)
        print(f"  recall={tp/max(1,fp):.4f} cand_per={sum(len(v) for v in cands.values())/max(1,len(cands)):.1f}", flush=True)
        del s23_c, cands, cos_map

    n_cand = sum(len(v) for v in all_cands.values())
    print(f"Total candidates: {n_cand} ({n_cand/max(1,len(all_cands)):.1f} per S1)", flush=True)

    pairs, y = [], []
    for sid, clist in all_cands.items():
        truth = gt_sub.get(sid, set())
        for cid in clist:
            pairs.append((sid, cid))
            y.append(1 if cid in truth else 0)
    y = np.array(y, dtype=np.int8)
    print(f"Pairs: {len(pairs)} pos={int(y.sum())}", flush=True)

    # featurize in chunks to bound RAM
    CH = 300000
    Xs = []
    for st in range(0, len(pairs), CH):
        en = min(len(pairs), st + CH)
        Xs.append(pairs_to_matrix(pairs[st:en], s1_lookup, s23_lookup, all_cos))
        print(f"  feat {en}/{len(pairs)}", flush=True)
    import numpy as _np
    X = _np.vstack(Xs)
    del Xs
    print(f"X {X.shape}", flush=True)

    uniq = np.array(list(all_cands.keys()))
    tr_ids, va_ids = train_test_split(uniq, test_size=config.VAL_FRACTION, random_state=config.RANDOM_STATE)
    tr_set = set(tr_ids)
    s1_arr = np.array([p[0] for p in pairs])
    tr_mask = np.array([s in tr_set for s in s1_arr])

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=8, max_leaf_nodes=63,
        min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
        random_state=config.RANDOM_STATE,
    )
    print("Training...", flush=True)
    clf.fit(X[tr_mask], y[tr_mask])
    print(f"train acc={clf.score(X[tr_mask], y[tr_mask]):.4f}", flush=True)

    va_mask = ~tr_mask
    va_pairs = [p for p, m in zip(pairs, va_mask) if m]
    va_proba = clf.predict_proba(X[va_mask])[:, 1]
    proba_map = {}
    for (sid, cid), pr in zip(va_pairs, va_proba):
        proba_map.setdefault(sid, {})[cid] = float(pr)
    for sid in va_ids:
        proba_map.setdefault(str(sid), {})
    truth_va = {sid: gt_sub.get(str(sid), set()) for sid in proba_map}
    best_t, best_s, sweep = threshold_sweep(proba_map, truth_va)
    for t, s in sweep:
        print(f"  thr={t:.2f} F0.5={s:.4f}", flush=True)
    print(f"Best thr={best_t} F0.5={best_s:.4f}", flush=True)

    print("Refitting on full sample...", flush=True)
    clf_final = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=8, max_leaf_nodes=63,
        min_samples_leaf=50, l2_regularization=1.0, early_stopping=False,
        random_state=config.RANDOM_STATE,
    )
    clf_final.fit(X, y)
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    joblib.dump({"model": clf_final, "features": FEATURE_NAMES, "threshold": best_t, "val_f05": best_s}, config.MODEL_PATH)
    with open(config.MODEL_PATH + ".threshold", "w") as f:
        f.write(str(best_t))
    print(f"Saved {config.MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
