import polars as pl
from matplotlib import pyplot as plt

from ethos.inference.constants import Task

tasks = [
    (Task.DRG_PREDICTION, "acc_top_1"),
    (Task.SOFA_PREDICTION, "r2"),
    (Task.READMISSION, "fitted_auc"),
    (Task.ICU_ADMISSION, "fitted_auc"),
    (Task.HOSPITAL_MORTALITY, "fitted_auc"),
]

col_to_title = {
    Task.DRG_PREDICTION: "DRG Classification",
    Task.SOFA_PREDICTION: "SOFA Score Prediction",
    Task.READMISSION: "30-day Readmission",
    Task.ICU_ADMISSION: "ICU Admission",
    Task.HOSPITAL_MORTALITY: "In-Hospital Mortality",
    "overall_score": "Overall Score",
}


def join_results(results: dict[Task, pl.DataFrame], on: str | list[str], sort=None) -> pl.DataFrame:
    result_generator = (
        results[task].select(
            pl.col(on).cast(str),
            pl.struct(
                score=score_col,
                ci_low=pl.col(f"{score_col}_ci").list[0],
                ci_high=pl.col(f"{score_col}_ci").list[1],
            ).alias(task),
        )
        for task, score_col in tasks
        if task in results
    )

    df = next(result_generator)
    for rdf in result_generator:
        df = df.join(rdf, on=on, how="left")

    if sort is not None:
        df = df.sort(sort, nulls_last=True)
    return df


def eval_overall_results(df: pl.DataFrame) -> pl.DataFrame:
    def get_variance(ci_high: pl.Expr, ci_low: pl.Expr) -> pl.Expr:
        return ((ci_high - ci_low) / (2 * 1.96)) ** 2

    from itertools import chain

    score_columns = df.select(pl.exclude(pl.Utf8)).columns

    df = (
        df.with_columns(
            chain.from_iterable(
                (
                    pl.col(col).struct["score"].alias(col),
                    get_variance(pl.col(col).struct["ci_high"], pl.col(col).struct["ci_low"]).alias(
                        f"{col}_var"
                    ),
                )
                for col in score_columns
            )
        )
        .with_columns(
            *[
                ((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())).alias(
                    f"{col}_hat"
                )
                for col in score_columns
            ],
            *[
                (pl.col(f"{col}_var") / (pl.col(col).max() - pl.col(col).min()) ** 2).alias(
                    f"{col}_var_hat"
                )
                for col in score_columns
            ],
        )
        .with_columns(
            (1 / pl.col(f"{col}_var_hat")).alias(f"{col}_inv_var_hat") for col in score_columns
        )
        .with_columns(
            pl.sum_horizontal(pl.selectors.ends_with("inv_var_hat")).alias("inv_var_hat_sum")
        )
        .with_columns(
            (pl.col(f"{col}_inv_var_hat") / pl.col("inv_var_hat_sum")).alias(f"{col}_w")
            for col in score_columns
        )
        .with_columns(
            (1.96 * (1 / pl.col("inv_var_hat_sum")).sqrt()).alias("overall_score_ci_half")
        )
        .with_columns(
            overall_score=pl.sum_horizontal(
                pl.col(f"{col}_hat") * pl.col(f"{col}_w") for col in score_columns
            ),
        )
        .select(
            pl.col(pl.Utf8),
            overall_score=pl.struct(
                score=pl.col("overall_score").round(10),
                ci_low=(pl.col("overall_score") - pl.col("overall_score_ci_half")).clip(
                    lower_bound=0
                ),
                ci_high=(pl.col("overall_score") + pl.col("overall_score_ci_half")).clip(
                    upper_bound=1
                ),
            ),
        )
    )
    return df


def score_to_str(v: dict) -> str:
    return f"{v['score']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"


def print_overall_score(df: pl.DataFrame, figsize=None):
    df_score = df.select(
        pl.col(pl.Utf8),
        pl.col("overall_score").struct.unnest(),
    )

    error_low = (df_score["score"] - df_score["ci_low"]).to_numpy()
    error_high = (df_score["ci_high"] - df_score["score"]).to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        x=df_score["score"],
        y=df_score[:, 0],
        xerr=[error_low, error_high],
        fmt="x",
        capsize=4,
        capthick=2,
        markersize=6,
    )

    xmin = df_score.select("ci_low").min().item() - 0.01
    xmax = df_score.select("ci_high").max().item() + 0.01
    ax.set_xlim(xmin, xmax)
    ax.invert_yaxis()

    for direction in ["top", "right"]:
        ax.spines[direction].set_visible(False)

    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    return ax
