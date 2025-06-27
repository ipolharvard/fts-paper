import os
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from ...constants import PATIENT_ID_COL, STATIC_DATA_FN
from ...constants import SpecialToken as ST
from ...vocabulary import Vocabulary
from ..patterns import ScanAndAggregate
from ..utils import create_prefix_or_chain, static_class


def filter_codes(
    df: pl.DataFrame, *, codes_to_remove: Sequence[str], is_prefix: bool = False
) -> pl.DataFrame:
    expr = pl.col("code").cast(str).is_in(codes_to_remove)
    if is_prefix:
        expr = create_prefix_or_chain(codes_to_remove)
    return df.filter(~expr)


def filter_out_incorrectly_dated_events(df: pl.DataFrame) -> pl.DataFrame:
    birth_dates = df.filter(code=ST.DOB)[PATIENT_ID_COL, "time"]
    return df.filter(
        pl.col("time").is_null()
        | (
            pl.col("time")
            >= pl.col(PATIENT_ID_COL)
            .replace_strict(birth_dates[PATIENT_ID_COL], birth_dates["time"], default=None)
            .over(PATIENT_ID_COL)
        )
    )


def apply_vocab(df: pl.DataFrame, *, vocab: str | list[str] | None = None) -> pl.DataFrame:
    if vocab is None:
        return df
    elif isinstance(vocab, str):
        vocab = list(Vocabulary.from_path(vocab))
    return df.filter(pl.col("code").is_in(vocab))


@static_class
class CodeCounter(ScanAndAggregate):
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.select(pl.col("code").value_counts()).unnest("code")

    def agg(self, in_fps: list, out_fp: str | Path) -> None:
        dfs = (pl.scan_parquet(fp) for fp in in_fps)
        df = next(dfs)
        for rdf in dfs:
            df = df.join(rdf, on="code", how="full", coalesce=True, join_nulls=True).select(
                "code", pl.sum_horizontal(pl.exclude("code"))
            )
        df.sort("count", "code", descending=[True, False]).collect().write_csv(out_fp)


@static_class
class StaticDataCollector(ScanAndAggregate):
    def __init__(self):
        os.environ["STATIC_DATA_FN"] = STATIC_DATA_FN

    def __call__(self, df: pl.DataFrame, *, static_code_prefixes: list[str]) -> pl.DataFrame:
        return (
            df.select(PATIENT_ID_COL, "code", pl.col("time").cast(pl.Int64))
            .filter(create_prefix_or_chain(static_code_prefixes))
            .with_columns(prefix=pl.col("code").str.split("//"))
            .group_by(
                PATIENT_ID_COL,
                pl.when(pl.col("prefix").list.len() > 1)
                .then(
                    pl.col("prefix")
                    .list.slice(0, pl.col("prefix").list.len() - 1)
                    .list.join(separator="//")
                )
                .otherwise(pl.col("prefix").list[0]),
            )
            .agg("code", "time")
            .with_columns(pl.struct(code="code", time="time"))
            .pivot(index=PATIENT_ID_COL, on="prefix", values="code")
            .with_columns(
                pl.when(pl.col(col_name).struct[0].is_null())
                .then(pl.struct(code=pl.lit([f"{col_name}//UNKNOWN"])))
                .otherwise(col_name)
                .alias(col_name)
                for col_name in static_code_prefixes
                if col_name != ST.DOB
            )
        )
