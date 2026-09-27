"""Scalable per-country TF-IDF blocking.

For each country value (open set - grouped dynamically, no hard-coding):
  - build char_wb TF-IDF on S2+S3 blocking_text
  - query S1 in batches via sparse matmul, keep top-K per S1
Also unions exact normalized-name matches (dict lookup) to protect recall.

Memory-friendly: processes one country at a time, batches S1,
uses CSR sparse matrices, releases per-country matrices after use.
"""
import numpy as np
from collections import defaultdict
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import normalize_series, normalize_country_series
from . import config


def _build_vectorizer():
    return TfidfVectorizer(
        analyzer=config.TFIDF_ANALYZER,
        ngram_range=config.TFIDF_NGRAM,
        min_df=config.TFIDF_MIN_DF,
        max_features=config.TFIDF_MAX_FEATURES,
        max_df=getattr(config, "TFIDF_MAX_DF", 1.0),
        lowercase=False,
    )


def _sparsify_queries(Q, idf, keep):
    """Keep only top-`keep` rarest (highest idf) ngrams per query row.

    Q: CSR (batch x vocab). idf: array vocab. Returns CSR with <=keep nnz/row.
    """
    Q = Q.tocsr()
    indptr = Q.indptr
    indices = Q.indices
    data = Q.data
    new_indptr = [0]
    new_indices = []
    new_data = []
    # idf lookup fast
    for i in range(Q.shape[0]):
        s, e = indptr[i], indptr[i + 1]
        row_idx = indices[s:e]
        row_data = data[s:e]
        if len(row_idx) <= keep:
            new_indices.extend(row_idx.tolist())
            new_data.extend(row_data.tolist())
        else:
            # rarest = highest idf
            w = idf[row_idx]
            part = np.argpartition(-w, keep - 1)[:keep]
            # keep original tfidf weights for selected
            sel_idx = row_idx[part]
            sel_data = row_data[part]
            new_indices.extend(sel_idx.tolist())
            new_data.extend(sel_data.tolist())
        new_indptr.append(len(new_indices))
    import scipy.sparse as sp
    return sp.csr_matrix((np.array(new_data, dtype=np.float32), np.array(new_indices, dtype=np.int32), np.array(new_indptr, dtype=np.int64)), shape=Q.shape)


def block_country(s1_ids, s1_texts, c_ids, c_texts, top_k, batch_size):
    """Fast sharded TF-IDF top-K.

    - Fit on sample for vocab/IDF.
    - Q sparsified to rarest K_RARE ngrams/row -> sparse matmul touches only
      postings of rare ngrams (10-20x faster, still typo-robust for char ngrams).
    - Per shard-batch keep top-K per row (numpy), merge across shards.
    """
    C_SHARD = 1000000
    FIT_SAMPLE = 100000
    keep = getattr(config, "Q_RARE_KEEP", 25)
    vec = _build_vectorizer()
    if len(c_texts) > FIT_SAMPLE:
        idx = np.random.RandomState(0).choice(len(c_texts), FIT_SAMPLE, replace=False)
        fit_texts = [c_texts[i] for i in idx]
    else:
        fit_texts = c_texts
    try:
        vec.fit(fit_texts)
    except ValueError:
        return {sid: [] for sid in s1_ids}, {}
    if not getattr(vec, "vocabulary_", None):
        return {sid: [] for sid in s1_ids}, {}
    idf = np.array(vec.idf_, dtype=np.float32)

    c_ids_arr = np.array(c_ids)
    n_c = len(c_ids)
    n_shards = (n_c + C_SHARD - 1) // C_SHARD
    print(f"    [blocking] n_c={n_c} shards={n_shards} n_s1={len(s1_ids)}", flush=True)

    # acc per sid: dict of cid->best score, but only from top-K per shard (small)
    acc = {sid: {} for sid in s1_ids}

    for sh in range(n_shards):
        s, e = sh * C_SHARD, min(n_c, (sh + 1) * C_SHARD)
        C = vec.transform(c_texts[s:e])
        for start in range(0, len(s1_ids), batch_size):
            end = min(len(s1_ids), start + batch_size)
            Q = vec.transform(s1_texts[start:end])
            Qr = _sparsify_queries(Q, idf, keep)
            del Q
            S = (Qr @ C.T).tocsr()
            del Qr
            # top-K per row via numpy (no Python per-nnz loop)
            for i in range(S.shape[0]):
                row = S.getrow(i)
                if row.nnz == 0:
                    continue
                data = row.data
                idxs = row.indices
                k = top_k if row.nnz > top_k else row.nnz
                part = np.argpartition(-data, k - 1)[:k]
                order = part[np.argsort(-data[part])]
                sid = s1_ids[start + i]
                d = acc[sid]
                for pj in order.tolist():
                    gj = s + int(idxs[pj])
                    cid = str(c_ids_arr[gj])
                    sc = float(data[pj])
                    if cid not in d or sc > d[cid]:
                        d[cid] = sc
        del C
    out = {}
    scores_out = {}
    for sid, d in acc.items():
        if not d:
            out[sid] = []
            continue
        ranked = sorted(d.items(), key=lambda x: -x[1])[:top_k]
        out[sid] = ranked
        scores_out[sid] = dict(ranked)
    return out, scores_out


def block_all(s1_df, s23_df, top_k=None, batch_size=None):
    """Block S1 against S2+S3.

    s1_df: DataFrame with entity_id, business_name, business_address, country
    s23_df: DataFrame with entity_id, business_name, business_address, country
    Returns: dict s1_id -> list of candidate ids (deduped, order by score),
             dict (s1_id, c_id) -> cosine score
    """
    top_k = top_k or config.TOP_K
    batch_size = batch_size or config.BATCH_SIZE

    # group by normalized country (open set) - vectorized
    s1_df = s1_df.copy()
    s23_df = s23_df.copy()
    s1_df["_cc"] = normalize_country_series(s1_df["country"])
    s23_df["_cc"] = normalize_country_series(s23_df["country"])

    s1_df["_nn"] = normalize_series(s1_df["business_name"])
    s23_df["_nn"] = normalize_series(s23_df["business_name"])
    s1_df["_na"] = normalize_series(s1_df["business_address"])
    s23_df["_na"] = normalize_series(s23_df["business_address"])
    s1_df["_bt"] = (s1_df["_nn"] + " " + s1_df["_nn"] + " " + s1_df["_na"]).str.strip()
    s23_df["_bt"] = (s23_df["_nn"] + " " + s23_df["_nn"] + " " + s23_df["_na"]).str.strip()

    candidates = {}
    cos_map = {}

    countries = set(s1_df["_cc"].unique().tolist()) | set(s23_df["_cc"].unique().tolist())
    name_k = getattr(config, "NAME_K", top_k)
    addr_k = getattr(config, "ADDR_K", top_k)
    for cc in countries:
        sub1 = s1_df[s1_df["_cc"] == cc]
        sub23 = s23_df[s23_df["_cc"] == cc]
        if len(sub1) == 0 or len(sub23) == 0:
            for sid in sub1["entity_id"].tolist():
                candidates[str(sid)] = []
            continue
        s1_ids = sub1["entity_id"].astype(str).tolist()
        c_ids = sub23["entity_id"].astype(str).tolist()
        s1_names = sub1["_nn"].tolist()
        c_names = sub23["_nn"].tolist()
        s1_addrs = sub1["_na"].tolist()
        c_addrs = sub23["_na"].tolist()

        # exact-name index for this country
        exact = defaultdict(list)
        for cid, nn in zip(c_ids, sub23["_nn"].tolist()):
            if nn:
                exact[nn].append(str(cid))

        # Stream 1: name-only TF-IDF (catches word-order/abbr/typo-light matches)
        blk_n, sc_n = block_country(s1_ids, s1_names, c_ids, c_names, name_k, batch_size)
        # Stream 2: address-only TF-IDF (catches transliteration + DBA same-address)
        blk_a, sc_a = block_country(s1_ids, s1_addrs, c_ids, c_addrs, addr_k, batch_size)
        # merge exact + name + address (dedupe, name-first ordering)
        s1_nn = dict(zip(s1_ids, sub1["_nn"].tolist()))
        for sid in s1_ids:
            ex = exact.get(s1_nn.get(sid, ""), [])
            n_ids = [cid for cid, _ in blk_n.get(sid, [])]
            a_ids = [cid for cid, _ in blk_a.get(sid, [])]
            merged = []
            seen = set()
            for cid in ex + n_ids + a_ids:
                if cid not in seen:
                    seen.add(cid)
                    merged.append(cid)
            cap = name_k + addr_k + 10
            merged = merged[:cap]
            candidates[sid] = merged
            for cid, s in blk_n.get(sid, []):
                cos_map[(sid, cid)] = s
            for cid, s in blk_a.get(sid, []):
                # keep max cosine across streams; address cosine on different scale
                # store with 0.9 weight so name matches rank slightly higher in features
                w = float(s) * 0.9
                if (sid, cid) not in cos_map or w > cos_map[(sid, cid)]:
                    # don't overwrite a strong name score with weaker addr score
                    if (sid, cid) not in cos_map:
                        cos_map[(sid, cid)] = w
            for cid in ex:
                if (sid, cid) not in cos_map:
                    cos_map[(sid, cid)] = 1.0
        # free memory
        del blk_n, sc_n, blk_a, sc_a
    # ensure every S1 present
    for sid in s1_df["entity_id"].astype(str).tolist():
        candidates.setdefault(sid, [])
    return candidates, cos_map
