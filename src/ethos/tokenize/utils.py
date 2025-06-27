import time
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import polars as pl
from loguru import logger


def create_prefix_or_chain(prefixes: list[str]) -> pl.Expr:
    expr = pl.lit(False)
    for prefix in prefixes:
        expr = expr | pl.col("code").str.starts_with(prefix)
    return expr


def process_icd_cm_10(
    df: pl.DataFrame,
    icd_col: pl.Expr,
    vocab: list[str] | None = None,
) -> pl.DataFrame:
    from .mappings import get_icd_cm_code_to_name_mapping

    code_to_name = get_icd_cm_code_to_name_mapping()
    code_prefixes = ["", "3-6//", "SFX//"]
    code_slices = [(0, 3), (3, 3), (6,)]
    temp_cols = [f"part{i}" for i in enumerate(code_prefixes)]

    df = (
        df.with_columns(
            icd_col.str.slice(*code_slice).alias(col)
            for col, code_slice in zip(temp_cols, code_slices)
        )
        .with_columns(pl.col(temp_cols[0]).replace_strict(code_to_name, default=None))
        .with_columns(
            pl.when(pl.col(col) != "").then(pl.lit(f"ICD//CM//{prefix}") + pl.col(col)).alias(col)
            for col, prefix in zip(temp_cols, code_prefixes)
        )
        .with_columns(unify_code_names(pl.col(temp_cols)))
    )

    if vocab is not None:
        df = apply_vocab_to_multitoken_codes(df, temp_cols, vocab)

    return (
        df.with_columns(code=pl.concat_list(temp_cols))
        .drop(temp_cols)
        .explode("code")
        .drop_nulls("code")
    )


def apply_vocab_to_multitoken_codes(
    df: pl.DataFrame, cols: list[str], vocab: list[str]
) -> pl.DataFrame:
    df = df.with_columns(pl.when(pl.col(col).is_in(vocab)).then(col).alias(col) for col in cols)
    for l_col, r_col in zip(cols, cols[1:]):
        df = df.with_columns(pl.when(pl.col(l_col).is_not_null()).then(r_col).alias(r_col))
    return df


def unify_code_names(col: pl.Expr) -> pl.Expr:
    return (
        col.str.to_uppercase().str.replace_all(r"[,.]", "").str.replace_all(" ", "_", literal=True)
    )


def static_class(cls):
    return cls()


def load_function(function_name: str, module_name: str) -> Callable:
    module = import_module(module_name)
    if "." in function_name:
        cls_name, function_name = function_name.split(".")
        module = getattr(module, cls_name)
    return getattr(module, function_name)


def wait_for_workers(output_dir: str | Path, sleep_time: int = 2):
    time_slept = 0
    output_dir = Path(output_dir)
    while any(output_dir.glob(".*.parquet_cache/locks/*.json")):
        time.sleep(sleep_time)
        time_slept += sleep_time
        if time_slept > 30:
            logger.warning(
                "Waiting for: {}".format(
                    [str(fp) for fp in output_dir.glob(".*.parquet_cache/locks/*.json")]
                )
            )
