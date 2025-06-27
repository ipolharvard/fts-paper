import argparse
from pathlib import Path

import pandas as pd
import polars as pl
from joblib import Parallel, delayed
from loguru import logger

DEFAULT_DATA_FORMAT = ".csv.gz"


def _convert_csv_to_parquet(orig_path: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(orig_path, low_memory=False)

    out_path = (out_dir / orig_path.name.split(".")[0]).with_suffix(".parquet")

    pl.from_pandas(df).write_parquet(out_path, use_pyarrow=True)


def convert_csv_to_parquet(path, data_format=DEFAULT_DATA_FORMAT, n_jobs=1):
    path = Path(path).resolve()
    assert path.is_dir(), f"Path is not a directory: {path}"

    out_dir = path.with_name(f"{path.name}_parquet")
    out_dir.mkdir(exist_ok=True)

    subset_paths = list(path.rglob(f"*{data_format}"))
    logger.info(f"Found {len(subset_paths)} subsets in the directory.")

    Parallel(n_jobs=n_jobs, verbose=2000)(
        delayed(_convert_csv_to_parquet)(orig_path, out_dir / orig_path.relative_to(path).parent)
        for orig_path in subset_paths
    )
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all CSV files to parquet preserving the original file hierarchy. "
        "The parquet dataset is created in the same directory as the CSV dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the directory with the CSV files to be converted to parquet.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default=DEFAULT_DATA_FORMAT,
        help="Format suffix of the CSV files.",
    )
    parser.add_argument(
        "-j", "--n_jobs", type=int, default=1, help="Number of parallel jobs to run."
    )
    args = parser.parse_args()
    convert_csv_to_parquet(args.path, args.data_format, args.n_jobs)
