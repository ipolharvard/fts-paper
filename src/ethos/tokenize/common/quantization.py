import json
from pathlib import Path

import numpy as np
import polars as pl

from ..patterns import ScanAndAggregate
from ..utils import create_prefix_or_chain, static_class


@static_class
class Quantizator(ScanAndAggregate):
    def __call__(self, df: pl.DataFrame, *, code_prefixes: list[str]) -> pl.DataFrame:
        expr = create_prefix_or_chain(code_prefixes)
        return df.filter(expr).group_by("code").agg("numeric_value")

    def agg(
        self,
        in_fps: list[str | Path],
        out_fp: str | Path,
        *,
        num_quantiles: int = 10,
    ) -> None:
        dfs = [
            pl.scan_parquet(fp).rename({"numeric_value": f"numeric_value/{i}"})
            for i, fp in enumerate(in_fps)
        ]
        df = dfs[0]
        for i, rdf in enumerate(dfs[1:], 1):
            df = df.join(rdf, on="code", how="full", coalesce=True, join_nulls=True)
        df = df.with_columns(
            pl.concat_list(pl.col("^numeric_value/.*$").fill_null([])).alias("numeric_value")
        )

        quantiles = np.linspace(0, 1, num_quantiles + 1)[1:-1]
        quantile_df = df.select(
            "code",
            pl.concat_list(
                pl.col("numeric_value").list.eval(pl.col("").quantile(q)) for q in quantiles
            )
            .list.unique()
            .alias("quantiles"),
        ).collect()
        quantile_dict = dict(zip(quantile_df["code"], quantile_df["quantiles"].to_list()))
        with Path(out_fp).open("w") as f:
            json.dump(quantile_dict, f)


def transform_to_quantiles(df: pl.DataFrame, *, code_quantiles: str | Path | dict) -> pl.DataFrame:
    """Transform numeric values in the DataFrame to quantiles based on provided code quantiles.

    Examples
    --------
    Basic usage with two breaks:
    >>> df = pl.DataFrame({
    ...     "code": ["A", "A", "B", "B"],
    ...     "numeric_value": [1, 2, 3, 4]
    ... })
    >>> code_quantiles = {
    ...     "A": [1.1, 2.],
    ...     "B": [3., 3.9],
    ... }
    >>> transform_to_quantiles(df, code_quantiles=code_quantiles)
    shape: (4, 2)
    ┌──────┬───────────────┐
    │ code ┆ numeric_value │
    │ ---  ┆ ---           │
    │ str  ┆ i64           │
    ╞══════╪═══════════════╡
    │ Q1   ┆ 1             │
    │ Q2   ┆ 2             │
    │ Q1   ┆ 3             │
    │ Q3   ┆ 4             │
    └──────┴───────────────┘
    >>> transform_to_quantiles(df.lazy(), code_quantiles=code_quantiles).collect()
    shape: (4, 2)
    ┌──────┬───────────────┐
    │ code ┆ numeric_value │
    │ ---  ┆ ---           │
    │ str  ┆ i64           │
    ╞══════╪═══════════════╡
    │ Q1   ┆ 1             │
    │ Q2   ┆ 2             │
    │ Q1   ┆ 3             │
    │ Q3   ┆ 4             │
    └──────┴───────────────┘
    >>> transform_to_quantiles(df, code_quantiles={"A": [1.5, 2.5]})
    shape: (4, 2)
    ┌──────┬───────────────┐
    │ code ┆ numeric_value │
    │ ---  ┆ ---           │
    │ str  ┆ i64           │
    ╞══════╪═══════════════╡
    │ Q1   ┆ 1             │
    │ Q2   ┆ 2             │
    │ B    ┆ 3             │
    │ B    ┆ 4             │
    └──────┴───────────────┘
    """
    if not isinstance(code_quantiles, dict):
        with Path(code_quantiles).open("r") as f:
            code_quantiles = json.load(f)

    max_length = max(len(v) for v in code_quantiles.values())
    tmp_cols = [f"field_{i}" for i in range(max_length)]
    return (
        df.with_columns(
            pl.col("code")
            .replace_strict(code_quantiles, default=None, return_dtype=pl.List(pl.Float64))
            .list.to_struct(fields=tmp_cols)
            .struct.unnest()
        )
        .with_columns(
            code=pl.when(pl.col(tmp_cols[0]).is_not_null())
            .then(
                "Q"
                + pl.sum_horizontal(
                    pl.col(tmp_cols) < pl.col("numeric_value"),
                    1,
                )
            )
            .otherwise("code")
        )
        .drop(tmp_cols)
    )
