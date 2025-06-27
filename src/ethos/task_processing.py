from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from .constants import MAPPINGS_DIR
from .constants import SpecialToken as ST
from .inference.constants import Reason, Task


def load_results(input_dir: str | Path) -> pl.DataFrame:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {input_dir}")

    parquet_dfs = []
    if parquet_fps := list(input_dir.rglob("*.parquet")):
        parquet_dfs = [
            pl.read_parquet(res_path, glob=False).with_columns(pl.col("^.*time$").cast(pl.Duration))
            for res_path in parquet_fps
        ]

    # To be removed in the future
    json_dfs = []
    if json_fps := list(input_dir.rglob("*.json")):
        json_dfs = [
            pl.read_json(res_path, infer_schema_length=None).with_columns(
                pl.col("^.*time$").cast(pl.Duration)
            )
            for res_path in json_fps
            if res_path.stem != "metadata"
        ]

    if not parquet_dfs and not json_dfs:
        raise FileNotFoundError(f"No results found in {input_dir}")

    return pl.concat((*parquet_dfs, *json_dfs), how="diagonal")


def process_inference_results(
    input_dir: str | Path,
    actual_expr: pl.Expr,
    expected_expr: pl.Expr = None,
    filter_ambiguous: pl.Expr = None,
    additional_columns: list[str] | None = None,
    max_rep_num: int | None = None,
    group_by_col: str | pl.Expr = "data_idx",
    warn_on_dropped: bool = True,
) -> pl.DataFrame:
    df = load_results(input_dir)

    prev_len = len(df)
    df = df.filter(pl.col("stop_reason").is_in([Reason.GOT_TOKEN, Reason.TIME_LIMIT]))
    if warn_on_dropped and (dropped := prev_len - len(df)):
        logger.warning(f"Dropped {dropped:,} results due to stop reason: {Reason.KEY_ERROR}.")

    if filter_ambiguous is not None:
        prev_len = len(df)
        df = df.filter(filter_ambiguous)
        if warn_on_dropped and (dropped := prev_len - len(df)):
            logger.warning(f"Dropped {dropped:,} ({dropped / prev_len:.2%}) ambiguous results.")

    optional_columns = [col for col in ("icu_stay_id", "hadm_id", "stay_id") if col in df.columns]
    aggregations = [
        ("expected", "first"),
        ("actual", "mean"),
        ("true_token_dist", "first"),
        ("token_dist", "mean"),
        ("true_token_time", "first"),
        ("token_time", "mean"),
        *[
            (col_name, "first")
            for col_name in ["patient_id", *optional_columns, *(additional_columns or [])]
        ],
    ]
    max_rep_num_expr = pl.min_horizontal(max_rep_num, pl.len())

    return (
        df.lazy()
        .with_columns(
            actual=actual_expr,
            expected=expected_expr if expected_expr is not None else "expected",
        )
        .group_by(group_by_col)
        .agg(
            *[
                (
                    getattr(pl.col(col_name), op)()
                    if max_rep_num is None
                    else getattr(pl.col(col_name).sample(max_rep_num_expr), op)()
                )
                for col_name, op in aggregations
            ],
            counts=max_rep_num_expr,
        )
        .collect()
    )


def join_metadata(df: pl.DataFrame, input_dir: Path, fn_col: str = "name") -> pl.DataFrame:
    return pl.concat(
        (
            pl.read_json(fp).with_columns(pl.Series(fn_col, [fn]))
            for fn in df[fn_col]
            if (fp := input_dir / fn / "metadata.json").exists()
        ),
        how="vertical_relaxed",
    ).join(df, on=fn_col, how="right")


def process_hospital_mortality_results(input_dir: Path, **kwargs) -> pl.DataFrame:
    return process_inference_results(
        input_dir,
        actual_expr=pl.col("actual") == ST.DEATH,
        expected_expr=pl.col("expected") == ST.DEATH,
        filter_ambiguous=(
            pl.col("actual").is_in([ST.DEATH, ST.DISCHARGE])
            & (pl.col("stop_reason") == Reason.GOT_TOKEN)
        ),
        **kwargs,
    )


def process_icu_admission_results(input_dir: Path, **kwargs) -> pl.DataFrame:
    return process_inference_results(
        input_dir,
        actual_expr=pl.col("actual").is_in([ST.DEATH, ST.ICU_ADMISSION]),
        expected_expr=pl.col("expected").is_in([ST.DEATH, ST.ICU_ADMISSION]),
        filter_ambiguous=pl.col("stop_reason") == Reason.GOT_TOKEN,
        **kwargs,
    )


def process_readmission_results(input_dir: Path, **kwargs) -> pl.DataFrame:
    outcome_tokens = [ST.ADMISSION, ST.DEATH]
    return process_inference_results(
        input_dir,
        actual_expr=pl.col("actual").is_in(outcome_tokens),
        expected_expr=(
            pl.col("expected").is_in(outcome_tokens)
            & (pl.col("true_token_time") <= pl.duration(days=30))
        ),
        **kwargs,
    )


def process_ed_hospitalization_results(input_dir: Path, **kwargs) -> pl.DataFrame:
    return process_inference_results(
        input_dir, actual_expr=pl.col("actual") == ST.ADMISSION, **kwargs
    )


def process_critical_outcome_results(input_dir: Path, **kwargs) -> pl.DataFrame:
    return process_inference_results(
        input_dir,
        actual_expr=pl.col("actual").is_in([ST.ICU_ADMISSION, ST.DEATH]),
        expected_expr=pl.col("expected") & (pl.col("true_token_time") <= pl.duration(hours=12)),
        **kwargs,
    )


def process_ed_representation_results(input_dir: Path, **kwargs) -> pl.DataFrame:
    return process_inference_results(
        input_dir,
        actual_expr=pl.col("actual").is_in([ST.ED_ADMISSION]),
        expected_expr=pl.col("expected") & (pl.col("true_token_time") <= pl.duration(hours=72)),
        **kwargs,
    )


def process_sofa_results(input_dir: Path) -> pl.DataFrame:
    sofa_scores = pl.read_csv(
        MAPPINGS_DIR / "mimic-iv_derived.csv.gz", columns=["stay_id", "first_day_sofa"]
    )
    df = (
        load_results(input_dir)
        .with_columns(
            pl.col("expected").str.slice(1).cast(pl.Int8),
            pl.col("actual").str.slice(1).cast(pl.Int8, strict=False),
        )
        .join(sofa_scores, left_on="icu_stay_id", right_on="stay_id", validate="m:1")
        .rename({"expected": "true_bin", "actual": "pred_bin", "first_day_sofa": "true_sofa"})
    )
    num_quantiles = sum(1 for col in df.columns if col.startswith("Q"))
    q_cols = [f"Q{i}" for i in range(1, num_quantiles + 1)]
    discrete_exps = np.arange(1, num_quantiles + 1)

    bin_ranges = np.linspace(0, 23, num_quantiles + 1)
    exps = np.fromiter(
        (
            np.arange(np.ceil(bottom), top + 1e-6).mean()
            for bottom, top in zip(bin_ranges[:-1], bin_ranges[1:])
        ),
        dtype=float,
    )
    tmp_col = "probs_sum"
    return (
        df.with_columns(pl.sum_horizontal(q_cols).alias(tmp_col))
        .with_columns(
            (
                pl.sum_horizontal(
                    pl.col(q_col) / pl.col(tmp_col) * expectations[i]
                    for i, q_col in enumerate(q_cols)
                )
            ).alias(col)
            for col, expectations in [("pred_sofa", exps), ("pred_bin", discrete_exps)]
        )
        .group_by("data_idx")
        .agg(
            pl.mean("true_sofa", "pred_sofa", "true_bin", "pred_bin"),
            pl.std("pred_sofa").alias("pred_sofa_std"),
            pl.first("icu_stay_id"),
        )
    )


def process_drg_results(
    input_dir: Path,
    top_k: int | Iterable[int] | None = None,
) -> pl.DataFrame:
    df = load_results(input_dir)
    drg_cols = [col for col in df.columns if col.startswith("DRG//")]
    if isinstance(top_k, int):
        top_k = [top_k]
    elif top_k is None:
        top_k = [len(drg_cols)]
    else:
        top_k = list(top_k)

    top_k_cols = {f"acc_top_{k}" if k <= len(drg_cols) else "sorted_drgs": k for k in top_k}

    df_drg_probs = df.select(drg_cols)
    df = df.drop(drg_cols)

    df_drg_probs = (
        df_drg_probs.transpose()
        .select(pl.all().arg_sort(descending=True))
        .head(max(top_k))
        .transpose()
        .lazy()
        .select(
            pl.concat_list(
                pl.all().replace_strict(pl.int_range(len(drg_cols)), drg_cols, return_dtype=pl.Utf8)
            ).alias("top")
        )
        .select(pl.col("top").list.slice(0, k).alias(col) for col, k in top_k_cols.items())
        .collect()
    )

    return df.with_columns(df_drg_probs).select(
        "expected",
        *df_drg_probs.columns,
        "data_idx",
        "patient_id",
        "hadm_id",
    )


TASK_RESULTS_PROCESSING_FUNC = {
    Task.HOSPITAL_MORTALITY: process_hospital_mortality_results,
    Task.READMISSION: process_readmission_results,
    Task.ICU_ADMISSION: process_icu_admission_results,
    Task.SOFA_PREDICTION: process_sofa_results,
    Task.DRG_PREDICTION: process_drg_results,
    Task.ED_HOSPITALIZATION: process_ed_hospitalization_results,
    Task.ED_CRITICAL_OUTCOME: process_critical_outcome_results,
    Task.ED_REPRESENTATION: process_ed_representation_results,
}
