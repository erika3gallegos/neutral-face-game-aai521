"""`python -m src.models.train --model baseline|improved` (README entry point).

baseline -> src.models.train_baseline (Person 1)
improved -> not wired yet; Person 2 plugs their trainer in here.
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["baseline", "improved"])
    a, rest = ap.parse_known_args()
    if a.model == "baseline":
        from src.models.train_baseline import main as run
        run(rest)
    else:
        sys.exit("Improved detector trainer not connected yet (owned by Person 2).")


if __name__ == "__main__":
    main()
