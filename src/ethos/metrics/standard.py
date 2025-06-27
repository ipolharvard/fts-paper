from collections import defaultdict

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.calibration import calibration_curve
from sklearn.metrics import r2_score, roc_auc_score, roc_curve


def compute_basic_metrics(y_true, y_pred):
    return {
        "n": len(y_true),
        "prevalence": y_true.mean(),
        "auc": roc_auc_score(y_true, y_pred),
        "auprc": -np.trapezoid(*roc_curve(y_true, y_pred)[:2]),
    }


def objective_function(std, points, equal_variance=False):
    thresholds = np.linspace(-10, 11, num=10000)
    std2 = std[0] if equal_variance else std[1]
    cdf_hypothesis_1 = norm.cdf(thresholds, loc=0, scale=std[0])
    cdf_hypothesis_2 = norm.cdf(thresholds, loc=1, scale=std2)
    # Calculate the True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr_values = 1 - cdf_hypothesis_2
    fpr_values = 1 - cdf_hypothesis_1
    val = sum(np.min((tpr_values - tpr) ** 2 + (fpr_values - fpr) ** 2) for tpr, fpr in points)
    return val


def compute_fitted_metrics(
    y_true,
    y_pred,
    equal_variance: bool = False,
    operating_point: float | None = None,
    operating_point_type: str = "01",
    auc_interpolation_type="gaussian",  # 'quadratic','linear'
) -> dict:
    fpr_points, tpr_points, thresholds = roc_curve(y_true, y_pred)
    points = np.stack((tpr_points, fpr_points), axis=1).tolist()

    # number of samples in interpolation curves
    samples = 10000
    tpr_values = np.linspace(0, 1, samples)
    fpr_values = np.linspace(0, 1, samples)
    if auc_interpolation_type == "gaussian":
        delta = 1e-6
        upper_const = 10
        # Define the range constraints for x and y
        std_constraint = {
            "type": "ineq",
            "fun": lambda x: np.array([x[0] - delta, upper_const - x[0]]),
        }
        std2_constraint = {
            "type": "ineq",
            "fun": lambda x: np.array([x[1] - delta, upper_const - x[1]]),
        }
        if equal_variance:
            constraints = [std_constraint]
            x0 = np.array([1.0])
        else:
            constraints = [std_constraint, std2_constraint]
            x0 = np.array([0.5, 0.5])

        result = minimize(objective_function, x0, args=(points,), constraints=constraints)

        if equal_variance:
            optimal_x = result.x
        else:
            optimal_x, optimal_y = result.x

        # Parameters for hypothesis 1 (mean and standard deviation)
        mean_hypothesis_1 = 0.0
        std_dev_hypothesis_1 = optimal_x

        # Parameters for hypothesis 2 (mean and standard deviation)
        mean_hypothesis_2 = 1
        std_dev_hypothesis_2 = optimal_x
        if not equal_variance:
            std_dev_hypothesis_2 = optimal_y

        # Calculate the cumulative distribution functions (CDFs) for the two distributions
        min_v = -5
        max_v = 10
        lattice_points = np.linspace(min_v, max_v, num=samples)

        cdf_hypothesis_1 = norm.cdf(
            lattice_points, loc=mean_hypothesis_1, scale=std_dev_hypothesis_1
        )
        cdf_hypothesis_2 = norm.cdf(
            lattice_points, loc=mean_hypothesis_2, scale=std_dev_hypothesis_2
        )
        # Calculate the True Positive Rate (TPR) and False Positive Rate (FPR)
        tpr_values = 1 - cdf_hypothesis_2
        fpr_values = 1 - cdf_hypothesis_1
    elif auc_interpolation_type == "linear" or auc_interpolation_type == "quadratic":
        unique_fpr_points = []
        unique_tpr_points = []

        fpr_dict = defaultdict(list)
        for x, y in zip(fpr_points, tpr_points):
            fpr_dict[x].append(y)

        for x, y_values in fpr_dict.items():
            unique_fpr_points.append(x)
            unique_tpr_points.append(np.mean(y_values))  # Take the mean of Y values for duplicates

        # Generate 10,000 equidistant points in the range of X
        # Note this is not ideal as interpolation point should be equdistanmt in 2D
        fpr_values = np.linspace(
            np.array(unique_fpr_points).max(), np.array(unique_fpr_points).min(), samples
        )
        # Create the interpolation function
        interpolation_function = interp1d(
            unique_fpr_points, unique_tpr_points, kind=auc_interpolation_type
        )
        # Calculate interpolated y values
        tpr_values = interpolation_function(fpr_values)

    lattice_idx = [
        np.argmin((tpr_values - tpr) ** 2 + (fpr_values - fpr) ** 2) for tpr, fpr in points
    ]

    if operating_point is None:
        match operating_point_type:
            case "01":
                # Find the best operating point defined as the closest point to (0,1)
                min_idx = np.argmin(fpr_values**2 + (tpr_values - 1) ** 2)
            case "Youden":
                # Find the best operating point defined as the point with the maximum Youden index
                min_idx = np.argmax(tpr_values - fpr_values)
            case "maxF1":
                # Find the best operating point defined as the point with the maximum F1 score
                min_idx = np.argmax(
                    2 * tpr_values * (1 - fpr_values) / (tpr_values + (1 - fpr_values))
                )
            case _:
                raise ValueError("operating_point_type must be one of: '01', 'Youden', 'maxF1'")

        # fit interpolation function between y and x points,
        # where y is thresholds and x is lattice_points
        f = np.poly1d(np.polyfit(lattice_idx[1:-1], thresholds[1:-1], 3))
        operating_point = f(min_idx)
    else:  # use provided operating point
        f = np.poly1d(np.polyfit(lattice_idx[1:-1], thresholds[1:-1], 3))
        min_idx_start = int((0 - min_v) / (max_v - min_v) * samples)
        max_idx_start = int((1 - min_v) / (max_v - min_v) * samples)
        # find the closest point to the operating point
        diff = f(np.arange(min_idx_start, max_idx_start - 1)) - operating_point
        min_idx = np.argmin(diff**2) + min_idx_start

    fpr = fpr_values[min_idx]
    tpr = tpr_values[min_idx]

    positives = np.sum(np.asarray(y_true) == 1)
    negatives = len(y_true) - positives

    denominator = tpr_values * positives + fpr_values * negatives
    recall_values = tpr_values
    more_than_zero = denominator > 0
    precision_values = np.divide(tpr_values * positives, denominator, where=more_than_zero)
    precision_values[~more_than_zero] = 1
    # =======================================================
    # Compute metrics now using the operating point
    # =======================================================

    tp = tpr * positives
    fn = (1 - tpr) * positives
    tn = (1 - fpr) * negatives
    fp = fpr * negatives

    # tier 1
    auprc = -np.trapezoid(precision_values, recall_values)
    auc = -np.trapezoid(tpr_values, fpr_values)
    accuracy = (tp + tn) / (positives + negatives)
    sensitivity = tpr
    specificity = 1 - fpr

    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    plr = sensitivity / (1 - specificity)
    nlr = (1 - sensitivity) / specificity
    f1 = tp / (tp + (fp + fn) / 2)

    return {
        "auc": auc,
        "auprc": auprc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "plr": plr,
        "nlr": nlr,
        "f1": f1,
        "recall": tpr,
        "precision_values": precision_values,
        "recall_values": recall_values,
        "tpr_points": tpr_points,
        "fpr_points": fpr_points,
        "tpr_values": tpr_values[::-1],
        "fpr_values": fpr_values[::-1],
        "operating_point": operating_point,
        "operating_point_index_in_values": samples - 1 - min_idx,
    }


def print_auc_roc_plot(res, fitted_res, title="AUC-ROC", lw=2, clinical=False):
    plt.plot([0, 1], [0, 1], color="grey", lw=lw, linestyle="--", label="Random Guess")
    plt.plot(
        fitted_res["fpr_values"],
        fitted_res["tpr_values"],
        color="darkorange",
        lw=lw,
        label="AUC-ROC Fitted",
    )
    plt.scatter(fitted_res["fpr_points"], fitted_res["tpr_points"])
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)

    text = [
        f"N={res['n']:,}",
        f"prevalence={res['prevalence']:.2%}",
    ]
    if clinical:
        text.extend(
            [
                f"auc={fitted_res['auc']:.3f}",
            ]
        )
    else:
        text.extend(
            [
                f"auc={res['auc']:.3f}",
                f"fitted_auc={fitted_res['auc']:.3f}",
                f"fitted_f1-score={fitted_res['f1']:.3f}",
                f"fitted_precision={fitted_res['precision']:.3f}",
                f"fitted_recall={fitted_res['recall']:.3f}",
                f"fitted_accuracy={fitted_res['accuracy']:.3f}",
            ]
        )
    anc = AnchoredText(
        "\n".join(text),
        loc="lower right",
        frameon=True,
        pad=0.3,
        prop=dict(size=12),
    )
    anc.patch.set_boxstyle("round,pad=0.2")
    plt.gca().add_artist(anc)


def compute_metrics(
    y_true,
    y_pred,
    metrics: list[str] | None = (
        "fitted_auc",
        "fitted_auc_ci",
        "auc",
        "fitted_f1",
        "n",
        "prevalence",
    ),
    n_bootstraps: int | None = None,
    **kwargs,
) -> dict:
    df = pl.DataFrame({"expected": y_true, "actual": y_pred})

    res = compute_basic_metrics(*df)
    res.update({f"fitted_{k}": v for k, v in compute_fitted_metrics(*df, **kwargs).items()})

    if n_bootstraps is not None and "fitted_auc_ci" in metrics:
        ci_low, ci_high = (
            pl.DataFrame(
                [
                    compute_fitted_metrics(*df.sample(fraction=1, with_replacement=True, seed=i))[
                        "auc"
                    ]
                    for i in range(n_bootstraps)
                ],
                schema=["auc"],
            ).select(pl.concat_list(pl.quantile("auc", q) for q in [0.025, 0.975]))
        ).item()
        res["fitted_auc_ci"] = (ci_low, ci_high)

    if metrics is not None:
        return {metric: res[metric] for metric in metrics if metric in res}
    return res


def compute_and_print_metrics(y_true, y_pred, figure_title: str, **kwargs) -> dict:
    basic_res = compute_basic_metrics(y_true, y_pred)
    fitted_res = compute_fitted_metrics(y_true, y_pred, **kwargs)
    print_auc_roc_plot(basic_res, fitted_res, figure_title)
    fitted_res = {f"fitted_{k}": v for k, v in fitted_res.items()}
    return basic_res | fitted_res


def plot_calibration_curve(y_true, y_pred, n_bins: int = 10):
    """Plots the calibration curve for given true labels and predicted probabilities.

    Parameters:
    y_true (array-like): True binary labels (0 or 1).
    y_pred (array-like): Predicted probabilities or scores.
    n_bins (int): Number of bins to use for calibration curve.

    Returns:
    None
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")

    # Plot calibration curve
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.show()


def get_auc_vs_fraction(
    df: pl.DataFrame,
    num_fractions: int = 10,
    frac_start: float = 0.01,
    frac_end: float = 1,
    num_fit_reps: int = 10,
) -> pl.DataFrame:
    """Computes the AUC for different fractions of the data."""
    res = []
    for frac in np.logspace(np.log10(frac_start), np.log10(frac_end), num=num_fractions):
        scores = [
            compute_fitted_metrics(*df.sample(fraction=frac, seed=i)["expected", "actual"])["auc"]
            for i in range(num_fit_reps)
        ]
        res.append(
            {
                "fraction": frac,
                "n_samples": round(len(df) * frac),
                "auc": np.median(scores),
            }
        )
    return pl.from_dicts(res)


def compute_sofa_results(y_true: pl.Series, y_pred: pl.Series, n_bootstraps: int = 10) -> dict:
    errors = pl.select(error=(y_true - y_pred).abs())
    mae_ci_low, mae_ci_high = (
        errors.select(
            pl.concat(
                pl.col("error").sample(fraction=1, with_replacement=True, seed=i).mean()
                for i in range(n_bootstraps)
            ).alias("errors")
        ).select(pl.concat_list(pl.quantile("errors", q) for q in [0.025, 0.975]))
    ).item()

    mae_result = {"score": errors.mean().item(), "ci_low": mae_ci_low, "ci_high": mae_ci_high}

    df = pl.select(y_true=y_true, y_pred=y_pred)
    r2_low, r2_high = (
        pl.DataFrame(
            [
                r2_score(*df.sample(fraction=1, with_replacement=True, seed=i))
                for i in range(n_bootstraps)
            ],
            schema=["r2"],
        ).select(pl.concat_list(pl.quantile("r2", q) for q in [0.025, 0.975]))
    ).item()

    r2_result = {"score": r2_score(y_true, y_pred), "ci_low": r2_low, "ci_high": r2_high}

    return {"r2": r2_result, "mae": mae_result}


def compute_drg_results(df: pl.DataFrame, n_bootstraps: int | None = None) -> pl.DataFrame:
    top_k_cols = [col for col in df.columns if col.startswith("acc_top_")]
    if not top_k_cols:
        raise ValueError("No top-k columns found in the DataFrame.")

    df = df.lazy().select(pl.col("expected").is_in(col).alias(col) for col in top_k_cols)
    exprs = [pl.col(col).mean().alias(col) for col in top_k_cols]

    if n_bootstraps is not None:
        exprs.extend(
            pl.concat_list(
                pl.col(col).sample(fraction=1, with_replacement=True, seed=i).mean()
                for i in range(n_bootstraps)
            ).alias(f"{col}_ci")
            for col in top_k_cols
        )
    df = df.select(exprs)

    if n_bootstraps is not None:
        df = df.with_columns(
            pl.concat_list(
                pl.col(f"{col}_ci").list.eval(pl.col("").quantile(q)) for q in [0.025, 0.975]
            )
            for col in top_k_cols
        )

    return df.collect()
