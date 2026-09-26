# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** akash08012006
**Team Members:** akash08012006, Somesh4206
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

Per-country word TF-IDF blocking (top-20 + exact-name safety net) reduces 1.7M x 10M comparisons to ~38M candidate pairs (~22 per S1). A 16-feature HistGradientBoosting matcher (rapidfuzz + Jaccard + TF-IDF cosine, BSD/MIT stack, <8B params) with F0.5-tuned threshold 0.60 achieves 0.709 macro F0.5 on held-out validation. No external data, APIs, or geocoding.

---

## 2. Methodology

### 2.1 Problem Analysis

- Train: S1 2.2M, S2 5.0M, S3 5.3M. Test: S1 1.73M, S2 4.89M, S3 5.08M. All TSV, `sep="\t"`.
- Countries: train {US 60%, India 40%}; test {India 47%, US 38%, France 15% unseen}. Country treated as open-set normalized string, never hard-coded.
- Ground truth (500k sample): ~5.6% singletons, ~3.67 matches per non-singleton (mostly 2-6, balanced S2/S3).
- Noise: abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited, Rd/Road, St/Street), legal-suffix inconsistency, DBA names, punctuation (& vs and), word-order changes, typos, transliteration (Hindi Devanagari preserved - unicode-safe normalization keeps non-ASCII), partial addresses, landmarks.
- Scale forces blocking: naive test comparisons ~1.7M x 10M = 17T pairs, impossible.

### 2.2 Solution Strategy

Blocking + gradient-boosting classifier + threshold tuning.

**Approach Type:** Blocking + Classifier
**Core Innovation:** Rare-token-sparsified word TF-IDF per-country blocking (keep top-6 rarest tokens per query) - 5x faster than char-TFIDF with 96.8% small-scale recall, plus exact normalized-name union to protect precision.

---

## 3. Candidate Generation (Blocking)

- **Keys:** strictly per normalized country (open set). Within country: word (1,2)-gram TF-IDF (`max_features=40000, min_df=2, max_df=0.1`) fit on 150k sample for IDF, query sparsified to top-6 rarest tokens, sparse matmul in 500k-doc shards, top-20 per S1. Union exact normalized-name matches (dict lookup, capped at 30 total).
- **Candidates:** train sample 60k S1 -> 1.33M pairs (22.2/S1). Test 1.73M S1 -> ~38M pairs (~22/S1). Reduction ratio vs naive ~450,000x.
- **Recall ceiling:** train full-scale recall 59-60% (India 0.590, US 0.603) at top-20; small-scale (500 S1 vs 23k distractors) recall 96.8% word / 93.9% char. Loss comes from full 4-6M distractors competing for top-20 and aggressive rare-keeping for speed. Exact-name net recovers trivial matches.
- Chunked inference (80-100k S1 slices per run) to fit 8GB RAM / 1h timeouts; 22 slices merged in S1 order.

---

## 4. Matching Model

**Features (16, all pre-normalized, no one-hot):**
- Name: exact, Jaccard, rapidfuzz ratio / token_sort / token_set / partial, len-diff, common-tokens
- Address: exact, Jaccard, rapidfuzz ratio / token_set, len-diff, common-tokens
- Other: country_exact (0/1, open-set safe), tfidf_cosine from blocking

**Model type:** `sklearn.ensemble.HistGradientBoostingClassifier` (BSD-licensed, <<8B): `max_iter=300, lr=0.06, max_depth=8, leaves=63, min_samples_leaf=50, l2=1.0`. Trained on 60k stratified S1 sample (1.33M pairs, 124k pos), entity-level 80/20 split.
**Threshold:** sweep 0.50-0.95 on validation macro F0.5; best 0.60 (F0.5=0.7094). Precision-heavy metric favors conservative threshold; singletons score 1.0 iff empty prediction.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, validation):** 0.7094 @ thr=0.60 (sweep: 0.50:0.7068, 0.55:0.7080, 0.60:0.7094, 0.65:0.7091, 0.70:0.7063, 0.75:0.7034, 0.80:0.6971, 0.85:0.6883, 0.90:0.6736, 0.95:0.6448). Train acc 0.9888 (imbalanced).
- **Test output:** 1,732,544 rows, 293,007 empty (16.9% predicted singletons), 1,439,537 non-empty. Validator PASS with `--check-ids`.
- **False positives:** near-duplicate names with different house numbers ("ABC Hospital 2" vs "ABC Hospital"), same chain different branch sharing address tokens.
- **False negatives:** DBA/trade names with zero token overlap missed by blocking (recall ceiling 60%); heavy typo + address-only signals.

---

## 6. Conclusion

Word-TFIDF rare-token blocking + HGB on string-similarity features gives a reproducible, license-compliant baseline (val F0.5 0.709) that scales to 10M records on 8GB RAM via per-country sharded, chunked inference. Future gains: higher TOP_K (30), char-trigram hybrid blocking for typos, hard-negative mining.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (src/, README.md, requirements.txt):
- `src/config.py` - TOP_K=20, word(1,2), Q_RARE_KEEP=6, MAX_TRAIN_S1=60000, thr from model
- `src/preprocessing.py` - vectorized unicode-safe normalization + abbr map
- `src/blocking.py` - per-country sharded TF-IDF + exact union
- `src/features.py` - 16 rapidfuzz/Jaccard features on pre-normalized strings
- `src/train.py` - stratified sample, blocking, HGB, threshold sweep, saves model.joblib
- `src/predict_chunk.py` - one 80-100k slice; `src/merge_parts.py` - merge in S1 order
- Reproduce from `student_resource/`: `pip install -r code/business_entity_resolution/requirements.txt`; `python -m code.business_entity_resolution.src.run train`; chunked predicts (see README) then merge; `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids`

### B. Additional Results

- Blocking: France S23 1.43M, US 3.82M, India 4.72M. Test pairs ~38M.
- Outputs: `matching_results.tsv` 73MB, `candidate_pairs.tsv` 517MB.
- Fair play: stdlib + sklearn/rapidfuzz/pandas only; no external lookup, geocoding, or internet.
