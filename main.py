"""
main.py
-------
Entry point for the BigVul vulnerability detection pipeline.

Runs the full sequence:
    1. Preprocessing  — tokenise, build vocab, pad sequences
    2. Random Forest  — BoW + RandomizedSearchCV
    3. SVM            — BoW + RandomizedSearchCV
    4. CNN            — trainable embeddings + Conv1D

Usage:
    python main.py                  # run all stages
    python main.py --skip-preprocess  # skip step 1 if data/ already exists
    python main.py --model cnn      # run a single model (rf | svm | cnn)
"""

import argparse
import subprocess
import sys
import time


STAGES = {
    "preprocess": "1_preprocess.py",
    "rf":         "2_model_rf.py",
    "svm":        "3_model_svm.py",
    "cnn":        "4_model_cnn.py",
}


def run(script: str):
    print(f"\n{'='*60}")
    print(f"  Running: {script}")
    print(f"{'='*60}\n")
    start = time.time()
    result = subprocess.run([sys.executable, script], check=True)
    elapsed = time.time() - start
    print(f"\n  Completed in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="BigVul vulnerability detection pipeline")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing if data/ already exists")
    parser.add_argument("--model", choices=["rf", "svm", "cnn"], default=None,
                        help="Run a single model instead of all three")
    args = parser.parse_args()

    if not args.skip_preprocess:
        run(STAGES["preprocess"])

    if args.model:
        run(STAGES[args.model])
    else:
        for name in ["rf", "svm", "cnn"]:
            run(STAGES[name])

    print("\nPipeline complete. Results saved to outputs/")


if __name__ == "__main__":
    main()
