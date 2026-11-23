"""Check the Kaggle downloads before running the pipeline.

    python -m src.data.check_raw

Prints which folder layout was found and compares counts with the published
dataset sizes, so a wrong or partial download shows up in seconds instead of
after preprocessing.
"""
from __future__ import annotations

from .config import FER_RAW, GENKI_RAW
from .datasets import index_fer, index_genki

EXPECTED = {
    "genki": {"total": 4000, "smile": 2162, "non_smile": 1838},
    "fer": {"total": 35887, "Training": 28709, "PublicTest": 3589, "PrivateTest": 3589},
}


def _line(name, got, want):
    ok = "OK " if got == want else "!! "
    print(f"  {ok}{name:<12} {got:>6}   (expected {want})")
    return got == want


def main():
    all_ok = True
    print(f"GENKI-4K  ({GENKI_RAW})")
    try:
        g = index_genki(GENKI_RAW)
        print(f"  layout: {g.attrs['layout']}   head pose: {'yes' if g['yaw'].notna().any() else 'no'}")
        e = EXPECTED["genki"]
        all_ok &= _line("images", len(g), e["total"])
        for lab in ("smile", "non_smile"):
            all_ok &= _line(lab, int((g["raw_label"] == lab).sum()), e[lab])
    except Exception as exc:  # noqa: BLE001
        all_ok = False
        print(f"  !! {exc}")

    print(f"\nFER-2013  ({FER_RAW})")
    try:
        f = index_fer(FER_RAW)
        print(f"  layout: {f.attrs['layout']}   official PublicTest/PrivateTest split: "
              f"{'recovered' if f.attrs['official_split'] else 'NOT available'}")
        e = EXPECTED["fer"]
        all_ok &= _line("images", len(f), e["total"])
        if f.attrs["official_split"]:
            for u in ("Training", "PublicTest", "PrivateTest"):
                all_ok &= _line(u, int((f["origin"] == u).sum()), e[u])
        else:
            all_ok = False
            print("  !! file names have no usage prefix: FER val/test will differ from the report")
        print("  classes:", f["raw_label"].value_counts().to_dict())
    except Exception as exc:  # noqa: BLE001
        all_ok = False
        print(f"  !! {exc}")

    print("\nAll checks passed." if all_ok else "\nSome checks failed: see lines marked !!")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
