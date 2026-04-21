"""UCI dataset fetcher used by the DVC `download` and `validate` stages.

The function `fetch_uci()` downloads the official "Predict Students' Dropout
and Academic Success" archive from the UCI repository and writes a clean CSV
to disk. We prefer `ucimlrepo` (official UCI helper) and fall back to the
direct URL if the package is unavailable in restricted environments.
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

from backend.app.ml.schemas import validate_raw

UCI_DATASET_ID = 697
UCI_DIRECT_ZIP = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
EXPECTED_COLUMNS_MIN = 35

log = logging.getLogger(__name__)


def fetch_uci() -> pd.DataFrame:
    """Return the UCI dropout dataset as a clean DataFrame.

    Tries `ucimlrepo` first (official, cached, polite), then falls back to a
    direct ZIP download. Raises if neither path works — by design, the DVC
    pipeline should fail loudly rather than silently train on stale data.
    """
    try:
        from ucimlrepo import fetch_ucirepo

        log.info("Fetching dataset %s via ucimlrepo", UCI_DATASET_ID)
        repo = fetch_ucirepo(id=UCI_DATASET_ID)
        x = repo.data.features
        y = repo.data.targets
        df = pd.concat([x, y], axis=1)
        # ucimlrepo returns Target as a single-column DataFrame; flatten it.
        if "Target" not in df.columns and y is not None:
            df = df.rename(columns={y.columns[0]: "Target"})
        return df
    except Exception as exc:  # noqa: BLE001 — fall back to URL
        log.warning("ucimlrepo failed (%s); falling back to direct download", exc)

    log.info("Downloading %s", UCI_DIRECT_ZIP)
    response = requests.get(UCI_DIRECT_ZIP, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # The archive contains a single CSV named "data.csv" (semicolon-delim).
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as fh:
            return pd.read_csv(fh, sep=";")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def cmd_download(out: Path) -> None:
    df = fetch_uci()
    if len(df.columns) < EXPECTED_COLUMNS_MIN:
        raise RuntimeError(
            f"UCI dataset returned only {len(df.columns)} columns; "
            f"expected at least {EXPECTED_COLUMNS_MIN}. Refusing to proceed."
        )
    _ensure_parent(out)
    df.to_csv(out, index=False)
    log.info("Wrote %s rows × %s cols to %s", len(df), len(df.columns), out)


def cmd_validate(in_path: Path) -> None:
    df = pd.read_csv(in_path)
    validated = validate_raw(df)
    out = in_path.parent.parent / "processed" / "validated.parquet"
    _ensure_parent(out)
    validated.to_parquet(out, index=False)
    log.info("Validated %s rows; wrote %s", len(validated), out)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="UCI dropout dataset utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download UCI dataset to CSV")
    p_dl.add_argument("--out", type=Path, required=True)

    p_val = sub.add_parser("validate", help="Validate a raw CSV against Pandera schema")
    p_val.add_argument("--in", dest="in_path", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "download":
        cmd_download(args.out)
    elif args.cmd == "validate":
        cmd_validate(args.in_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
