"""Macro F0.5 as defined by the challenge (per-S1 average, singletons included)."""


def f05_single(pred: set, truth: set) -> float:
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec = tp / len(truth)
    return (1.25 * prec * rec) / (0.25 * prec + rec)


def macro_f05(pred_map: dict, truth_map: dict) -> float:
    scores = []
    for s1, truth in truth_map.items():
        pred = pred_map.get(s1, set())
        scores.append(f05_single(set(pred), set(truth)))
    return sum(scores) / len(scores) if scores else 0.0


def threshold_sweep(proba_map: dict, truth_map: dict, thresholds=None):
    if thresholds is None:
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    best_t, best_s = thresholds[0], -1
    results = []
    for t in thresholds:
        pred = {s1: {c for c, p in d.items() if p >= t} for s1, d in proba_map.items()}
        s = macro_f05(pred, truth_map)
        results.append((t, s))
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s, results
