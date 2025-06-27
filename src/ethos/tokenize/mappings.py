import polars as pl

from ..constants import MAPPINGS_DIR


def get_icd_9_to_10_mapping(icd_type="cm") -> dict[str, str]:
    match icd_type:
        case "cm":
            conv_strategy = (
                pl.when(pl.col("icd_10").list.len() > 1)
                .then(pl.col("icd_10").list.sort().list[0].str.slice(0, 3))
                .otherwise(pl.col("icd_10").list[0])
            )
        case "pcs":
            conv_strategy = pl.col("icd_10").list.sort().list[0]
        case _:
            raise ValueError(f"Invalid ICD type: {icd_type}")

    df = (
        pl.read_csv(
            MAPPINGS_DIR / f"icd_{icd_type}_9_to_10_mapping.csv.gz",
            schema={"icd_9": pl.String, "icd_10": pl.String},
            null_values=["NoDx", "NoPCS"],
            truncate_ragged_lines=True,
        )
        .filter(pl.col("icd_10").is_not_null())
        .group_by("icd_9")
        .agg("icd_10")
        .with_columns(conv_strategy.alias("icd_10"))
    )
    return dict(zip(df["icd_9"], df["icd_10"]))


def get_icd_cm_9_to_10_mapping() -> dict[str, str]:
    return get_icd_9_to_10_mapping(icd_type="cm")


def get_icd_pcs_9_to_10_mapping() -> dict[str, str]:
    return get_icd_9_to_10_mapping(icd_type="pcs")


_COMPLEMENTARY_CODE_TO_NAME = {
    "F32": "Major depressive disorder, single episode",
    "F32A": "Depression, unspecified",
    "G928": "Other toxic encephalopathy",
    "G929": "Unspecified toxic encephalopathy",
    "I5A": "Non-ischemic myocardial injury (non-traumatic)",
    "K31A": "Gastric intestinal metaplasia",
    "L24A9": "Irritant contact dermatitis due to friction or contact with body fluids",
    "L24B": "Irritant contact dermatitis related to stoma or fistula",
    "M350": "Sicca syndrome [Sjogren]",
    "P099": "Abnormal findings on neonatal screening, unspecified",
    "R051": "Acute cough",
    "R052": "Subacute cough",
    "R053": "Chronic cough",
    "R054": "Cough syncope",
    "R058": "Other specified cough",
    "R059": "Cough, unspecified",
    "U09": "Post COVID-19 condition",
    "U099": "Post COVID-19 condition, unspecified",
    "W458X": "Lid of can entering through skin",
    "Z09": "Encounter for follow-up examination after completed treatment",
    "Z2252": "Carrier of viral hepatitis",
}


def get_icd_cm_code_to_name_mapping():
    df = (
        pl.read_csv(MAPPINGS_DIR / "icd10cm-order-Jan-2021.csv.gz")
        .group_by("code")
        .agg(pl.col("long").first())
    )
    out_dict = dict(zip(df["code"], df["long"]))
    out_dict.update(_COMPLEMENTARY_CODE_TO_NAME)
    return out_dict


def get_mimic_drug_name_to_atc_mapping() -> dict[str, list[str]]:
    df = (
        pl.read_csv(MAPPINGS_DIR / "mimic_drug_to_atc.csv.gz")
        .with_columns(pl.col("drug").str.to_lowercase().str.strip_chars(" "))
        .group_by("drug", maintain_order=True)
        .agg(pl.col("atc_code").unique(maintain_order=True))
    )
    return dict(zip(df["drug"], df["atc_code"].to_list()))


def get_atc_code_to_desc() -> dict[str, str]:
    df = pl.read_csv(MAPPINGS_DIR / "atc_coding.csv.gz").filter(
        pl.col("atc_code").is_last_distinct()
    )
    return dict(zip(df["atc_code"], df["atc_name"]))


def get_stay_id_to_sofa_mapping() -> dict[int, int]:
    df = pl.read_csv(
        MAPPINGS_DIR / "mimic-iv_derived.csv.gz", columns=["stay_id", "first_day_sofa"]
    )
    return dict(zip(df["stay_id"], df["first_day_sofa"]))


def get_cpt_to_level_names() -> dict[str, tuple[str | None]]:
    return (
        pl.scan_parquet(MAPPINGS_DIR / "cpt_code_ranges.parquet")
        .with_columns(last_letter=pl.col("lower_bound").str.contains(r"[a-zA-Z]$"))
        .with_columns(
            [
                pl.when("last_letter")
                .then(pl.col(col).str.slice(0, pl.col(col).str.len_chars() - 1))
                .otherwise(col)
                .cast(pl.Int64)
                .alias(col)
                for col in ("lower_bound", "upper_bound")
            ],
            last_letter=pl.when("last_letter").then(pl.col("lower_bound").str.slice(-1)),
        )
        .select(
            pl.int_ranges("lower_bound", pl.col("upper_bound") + 1)
            .cast(pl.List(pl.Utf8))
            .alias("cpt"),
            "desc",
            "level",
            "last_letter",
        )
        .explode("cpt")
        .with_columns(
            pl.when(pl.col("last_letter").is_null())
            .then(pl.col("cpt").str.zfill(5))
            .otherwise(pl.col("cpt").str.zfill(4) + pl.col("last_letter"))
        )
        .collect()
        .pivot(on="level", index="cpt", values="desc", aggregate_function="last")
        .rows_by_key("cpt", unique=True)
    )


def get_zip_code_to_median_income() -> dict:
    df = (
        pl.scan_parquet(MAPPINGS_DIR / "zip-code-median-income-11-23.parquet")
        .select("zip_code", pl.mean_horizontal(pl.exclude("zip_code")).alias("median_income"))
        .collect()
    )
    return dict(zip(df["zip_code"], df["median_income"]))


def get_mgb_coding_to_atc_mapping() -> pl.DataFrame:
    return pl.read_parquet(MAPPINGS_DIR / "mgb_drug_code_to_atc.parquet").select(
        "code_type", "code", "atc_code"
    )
