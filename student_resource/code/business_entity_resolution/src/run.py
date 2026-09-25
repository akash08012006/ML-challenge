"""End-to-end entry point: python -m src.run [train|predict|all]"""
import sys
from .train import main as train_main
from .predict import main as predict_main

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("train", "all"):
        train_main()
    if mode in ("predict", "all"):
        predict_main()
