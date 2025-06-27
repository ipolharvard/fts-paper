import json
from datetime import timedelta
from pathlib import Path

import polars as pl
import polars.selectors as cs

from ...constants import SpecialToken as ST
from ..patterns import MatchAndRevise, ScanAndAggregate
from ..utils import static_class


def _parse_time_interval_spec(spec: dict[str, dict]) -> dict[str, timedelta]:
    return {label: timedelta(**lower_bound) for label, lower_bound in spec.items()}


def inject_time_intervals(
    df: pl.DataFrame, *, time_intervals_spec: dict[str, dict]
) -> pl.DataFrame:
    patient_id_col = MatchAndRevise.sort_cols[0]

    time_intervals = _parse_time_interval_spec(time_intervals_spec)
    interval_names = sorted(time_intervals.keys(), reverse=True, key=lambda v: time_intervals[v])

    time_diff_col, timeline_start_col, timeline_end_col = (
        "time_diff",
        ST.TIMELINE_START,
        ST.TIMELINE_END,
    )
    largest_interval = interval_names.pop(0)

    # the largest interval is repeated as many times as fits
    intervals_expr = pl.when(pl.col(time_diff_col) >= time_intervals[largest_interval]).then(
        pl.lit(largest_interval).repeat_by(
            (pl.col(time_diff_col) / time_intervals[largest_interval]).round()
        )
    )
    for name in interval_names:
        intervals_expr = intervals_expr.when(pl.col(time_diff_col) >= time_intervals[name]).then(
            pl.concat_list(pl.lit(name))
        )
    intervals_expr = intervals_expr.otherwise([])

    return (
        df.with_columns(
            pl.col("time").diff().alias(time_diff_col),
            *[
                (
                    pl.col(patient_id_col) != pl.col(patient_id_col).shift(shift_n, fill_value=True)
                ).alias(col)
                for col, shift_n in [(timeline_start_col, 1), (timeline_end_col, -1)]
            ],
        )
        # zero out the time_diff when the timeline starts
        .with_columns(
            pl.when(timeline_start_col).then(None).otherwise(time_diff_col).alias(time_diff_col)
        )
        .with_columns(code=pl.concat_list(intervals_expr, "code"))
        .with_columns(
            code=pl.when(timeline_start_col)
            .then(pl.concat_list(pl.lit(ST.TIMELINE_START), "code"))
            .otherwise("code")
        )
        .with_columns(
            code=pl.when(timeline_end_col)
            .then(pl.concat_list("code", pl.lit(ST.TIMELINE_END)))
            .otherwise("code")
        )
        .drop(time_diff_col, timeline_start_col, timeline_end_col)
        .explode("code")
    )


@static_class
class IntervalEstimator(ScanAndAggregate):
    aggregations = ["min", "q1", "mean", "median", "q3", "max"]
    patient_id_col = MatchAndRevise.sort_cols[0]

    def __call__(self, df: pl.DataFrame, *, time_intervals_spec: dict[str, dict]) -> pl.DataFrame:
        time_intervals = _parse_time_interval_spec(time_intervals_spec)
        return (
            df.select(self.patient_id_col, "code", "time", pl.col("time").diff().alias("diff"))
            .filter(pl.col("code").is_in(list(time_intervals)))
            .set_sorted(self.patient_id_col)
            .with_columns(n=pl.count("code").over(self.patient_id_col, "time"))
            .filter(pl.col("diff") > 0)
            .with_columns(diff=(pl.col("diff") / pl.col("n")).repeat_by(pl.col("n")))
            .explode("diff")
            .group_by("code")
            .agg(
                min=pl.col("diff").min(),
                q1=pl.col("diff").quantile(0.25),
                mean=pl.col("diff").mean(),
                median=pl.col("diff").median(),
                q3=pl.col("diff").quantile(0.75),
                max=pl.col("diff").max(),
            )
            .sort(pl.col("code").replace_strict(time_intervals, return_dtype=pl.Duration))
        )

    def agg(self, in_fps: list, out_fp: str | Path) -> None:
        """Naively averages statistics pulled from all shards."""
        dfs = [pl.scan_parquet(fp) for fp in in_fps]
        df = dfs[0]
        for i, rdf in enumerate(dfs[1:], 1):
            rdf = rdf.select(
                "code", *[pl.col(agg).alias(f"{agg}/{i}") for agg in self.aggregations]
            )
            df = df.join(rdf, on="code", how="full", coalesce=True, join_nulls=True)
        df = (
            df.select(
                "code",
                *[
                    pl.mean_horizontal(cs.starts_with(agg).cast(int)).alias(agg)
                    for agg in self.aggregations
                ],
            )
            .melt(id_vars="code", value_vars=cs.exclude("code"), variable_name="stats")
            .collect()
            .pivot(index="stats", columns="code", values="value")
        )
        interval_stats_dict = dict(zip(df["stats"], df.select(pl.exclude("stats")).to_dicts()))
        with Path(out_fp).open("w") as f:
            json.dump(interval_stats_dict, f, indent=4)
