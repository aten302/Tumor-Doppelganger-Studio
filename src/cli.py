from __future__ import annotations

import argparse

from . import config
from .data_prep import prepare_and_save
from .similarity import build_and_save_index


def cmd_prepare_data(_: argparse.Namespace) -> None:
    prepared = prepare_and_save()
    print(f"Saved cleaned dataset -> {config.DATA_PROCESSED_PATH}")
    print(f"Rows: {len(prepared.df):,} | Features: {len(prepared.feature_cols)}")


def cmd_build_index(args: argparse.Namespace) -> None:
    path = build_and_save_index(n_neighbors=args.neighbors)
    print(f"Saved twin index -> {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tumor Doppelgänger Studio CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prepare-data", help="Clean dataset into data/processed/clean.csv")
    p1.set_defaults(func=cmd_prepare_data)

    p2 = sub.add_parser("build-index", help="Build and save similarity index (kNN)")
    p2.add_argument("--neighbors", type=int, default=50)
    p2.set_defaults(func=cmd_build_index)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
