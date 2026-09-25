# Business Entity Resolution

MIT/Apache-2.0 pipeline: TF-IDF blocking + HistGradientBoosting matching.
Model: `sklearn.ensemble.HistGradientBoostingClassifier` (BSD-licensed, <8B params).
No external data, APIs, geocoding, or internet lookup.

## Layout
```
code/business_entity_resolution/
  src/config.py         # paths, TOP_K=20, sampling, threshold
  src/preprocessing.py  # unicode-safe normalization, abbr mapping
  src/blocking.py       # per-country char TF-IDF top-K + exact-name safety net
  src/features.py       # rapidfuzz + jaccard + tfidf_cosine (16 features)
  src/train.py          # stratified 300k S1 sample, 80/20 val, threshold sweep for macro F0.5
  src/predict.py        # test blocking + chunked inference -> output/*.tsv
  src/evaluation.py     # macro F0.5
  requirements.txt
```

## Reproduce
From `student_resource/`:

```bash
pip install -r code/business_entity_resolution/requirements.txt
python -m code.business_entity_resolution.src.run train
python -m code.business_entity_resolution.src.run predict
# outputs: output/matching_results.tsv, output/candidate_pairs.tsv
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Or single step: `python -m code.business_entity_resolution.src.run all`

## Notes
- Country is open-set: grouped dynamically via normalized string, never hard-coded to {US, India}.
- Blocking is strictly per-country (same normalized country), then char_wb(3,3) TF-IDF top-20 + exact normalized-name union (cap 30).
- Inference is chunked (200k pairs) to stay under 8GB RAM.
- Threshold tuned on held-out 20% of sampled S1 for macro F0.5 (precision-heavy).
