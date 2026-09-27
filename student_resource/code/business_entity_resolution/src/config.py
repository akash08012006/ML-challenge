"""Central config - open-set country handling, no hard-coded country list."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# When running from student_resource/, dataset lives at dataset/
# Fall back to absolute locations.
def _find_dataset():
    candidates = [
        os.path.join(os.getcwd(), "dataset"),
        os.path.join(BASE_DIR, "dataset"),
        r"C:\ML-Challenge\student_resource\dataset",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]

DATASET_DIR = _find_dataset()
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

OUTPUT_DIR = os.path.join(os.getcwd(), "output")

# Blocking params (hybrid name + address streams)
TOP_K = 20  # fallback single-stream cap
NAME_K = 18  # name-stream top-K per S1
ADDR_K = 14  # address-stream top-K per S1 (catches transliteration/DBA)
BATCH_SIZE = 10000  # S1 queries per batch for TF-IDF blocking
TFIDF_NGRAM = (1, 2)
TFIDF_ANALYZER = "word"
TFIDF_MIN_DF = 2
TFIDF_MAX_FEATURES = 40000
TFIDF_MAX_DF = 0.1  # drop overly common tokens to sparsify
Q_RARE_KEEP = 6  # keep top-N rarest ngrams per query for speed

# Training params
VAL_FRACTION = 0.2
RANDOM_STATE = 42
MAX_TRAIN_S1 = 24000  # hybrid 2-stream blocking is 2x cost; smaller sample still ~800k pairs
MODEL_PATH = os.path.join(os.getcwd(), "code", "business_entity_resolution", "model.joblib")

# Inference
THRESHOLD = 0.65  # tuned on validation; overwritten by train.py
