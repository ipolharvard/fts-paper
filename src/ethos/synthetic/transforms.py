from collections.abc import Sequence

import polars as pl

from ..constants import SpecialToken as ST
from ..vocabulary import Vocabulary


def remove_rows_without_saved_tokens(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.filter(pl.col("generated_tokens").is_not_null())


def add_subject_id_col(df: pl.LazyFrame, offset: int = 0) -> pl.LazyFrame:
    return df.select(
        pl.int_range(offset, offset + pl.len())
        .cast(pl.String)
        .str.zfill(8)
        .cast(pl.Int64)
        .alias("subject_id"),
        pl.all(),
    )


def filter_subjects(df: pl.LazyFrame, subject_ids: Sequence[int]) -> pl.LazyFrame:
    return df.filter(pl.col("subject_id").is_in(subject_ids))


def convert_to_meds_format(df: pl.LazyFrame, vocab: Vocabulary) -> pl.LazyFrame:
    interval_estimates = vocab.interval_estimates["mean"]
    interval_estimates = {vocab.encode(k): v for k, v in interval_estimates.items()}

    return (
        df.with_columns(
            # This is probably costly, but we have to replace the last token in case
            # of hitting time_limit or key_error, so we can add TOS by the way
            pl.concat_list(
                pl.lit(vocab.encode(ST.TIMELINE_START)),
                pl.col("generated_tokens").list.slice(0, pl.col("token_dist") - 1),
                pl.lit(vocab.encode(ST.TIMELINE_END)),
            ).alias("generated_tokens")
        )
        .with_columns(
            time=pl.col("generated_tokens").list.eval(
                pl.element()
                .replace_strict(interval_estimates, default=0, return_dtype=pl.Float64)
                .cum_sum()
                .cast(pl.Duration)
            )
        )
        .explode("generated_tokens", "time")
        .select(
            "subject_id",
            time=pl.col("timeline_start_date") + pl.col("time"),
            code=pl.col("generated_tokens").replace_strict(vocab.itos, return_dtype=pl.String),
        )
    )
