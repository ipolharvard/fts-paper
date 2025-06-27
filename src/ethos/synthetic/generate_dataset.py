import functools
import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from loguru import logger
from MEDS_transforms.mapreduce.utils import rwlock_wrap
from omegaconf import DictConfig

from ethos.tokenize.common import CodeCounter

from ..constants import STATIC_DATA_FN
from ..constants import SpecialToken as ST
from ..datasets import TimelineDataset
from ..vocabulary import Vocabulary
from .transforms import (
    add_subject_id_col,
    convert_to_meds_format,
    filter_subjects,
    remove_rows_without_saved_tokens,
)

FILES_TO_COPY = [
    "interval_estimates.json",
    "quantiles.json",
]


def apply_transforms(in_fp: Path | list[Path], out_fp: Path, *transforms: Callable):
    rwlock_wrap(
        in_fp,
        out_fp,
        partial(pl.scan_parquet, glob=False),
        lambda df, out_: df.collect().write_parquet(out_, use_pyarrow=True),
        compute_fn=lambda df: functools.reduce(lambda df, fn: fn(df), transforms, df),
    )


def retrieve_patient_static_data(in_fp: Path) -> pl.DataFrame:
    df = pl.read_parquet(in_fp)
    static_data_columns = df["expected"].struct.fields
    return df.select("subject_id", pl.col("expected").struct.unnest()).with_columns(
        pl.struct(
            code=pl.concat_list(pl.lit(ST.DOB)),
            time=pl.concat_list(pl.col(ST.DOB).cast(pl.Int64)),
        ).alias(ST.DOB),
        *[
            pl.struct(code=pl.concat_list(col), time=None).alias(col)
            for col in static_data_columns
            if col != ST.DOB
        ],
    )


@hydra.main(version_base=None, config_path="../configs", config_name="synthetic")
def main(cfg: DictConfig):
    in_fps = list(Path(cfg.input_dir).rglob("*.parquet"))
    if not in_fps:
        raise FileNotFoundError(f"No parquet files found in '{cfg.input_dir}'.")
    logger.info(f"Found {len(in_fps)} input files in '{cfg.input_dir}'.")

    dataset_dir = Path(cfg.dataset_dir)
    output_dir = dataset_dir
    if cfg.output_dir is not None:
        output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting generating the new dataset at {output_dir}")

    vocab = Vocabulary.from_path(dataset_dir)

    subject_id_offset, shard_offset = 0, 0
    if output_dir.samefile(dataset_dir):
        subject_id_offset = max(TimelineDataset(dataset_dir).static_data.keys()) + 1
        shard_offset = len(list(dataset_dir.glob("*.safetensors")))
        logger.info("Skipping copying some files as `output_dir` == `dataset_dir`")
    else:
        logger.info("Copying vocab...")
        vocab.dump(output_dir)

        for fn in FILES_TO_COPY:
            if (fp := dataset_dir / fn).exists():
                logger.info(f"Copying {fp} to {output_dir}")
                shutil.copy(fp, output_dir / fn)
            else:
                raise FileNotFoundError(f"Expected, but not found: {fp}")

    out_fps, subject_ids = [], []
    output_dir /= "01_add_subject_id"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating shards with synthetic subject ids...")
    for i, in_fp in enumerate(in_fps, shard_offset):
        out_fps.append(output_dir / f"{i}.parquet")
        apply_transforms(
            in_fp,
            out_fps[-1],
            remove_rows_without_saved_tokens,
            functools.partial(add_subject_id_col, offset=subject_id_offset),
        )
        new_subject_ids = pl.read_parquet(out_fps[-1], columns=["subject_id"]).to_series()
        subject_id_offset += len(new_subject_ids)
        subject_ids.extend(new_subject_ids)
    in_fps = out_fps
    logger.info(f"Records of {len(subject_ids):,} subjects.")

    logger.info("Retrieving static data of the patients...")
    static_data_fn = STATIC_DATA_FN
    if output_dir.parent.samefile(dataset_dir):
        static_data_pfx = STATIC_DATA_FN.split(".")[0]
        static_data_file_num = len(list(dataset_dir.glob(static_data_pfx + "*")))
        static_data_fn = f"{static_data_pfx}_{static_data_file_num + 1}.parquet"
    pl.concat(retrieve_patient_static_data(in_fp) for in_fp in in_fps).write_parquet(
        output_dir.parent / static_data_fn, use_pyarrow=True
    )

    subject_partitions = np.array_split(
        subject_ids, indices_or_sections=np.ceil(len(subject_ids) / cfg.n_subjects_per_shard)
    )
    logger.info(f"Subjects will be repartitioned into {len(subject_partitions)} shards.")

    n_jobs = (
        len(subject_partitions)
        if cfg.max_n_jobs is None
        else min(cfg.max_n_jobs, len(subject_partitions))
    )
    output_dir = output_dir.with_name("02_meds")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_fps = [output_dir / f"{i}.parquet" for i, _ in enumerate(subject_partitions, shard_offset)]

    logger.info("Resharding and converting to MEDS...")
    Parallel(n_jobs=n_jobs, verbose=False)(
        delayed(apply_transforms)(
            in_fps,
            out_fp,
            functools.partial(filter_subjects, subject_ids=subject_id_partition),
            functools.partial(convert_to_meds_format, vocab=vocab),
        )
        for out_fp, subject_id_partition in zip(out_fps, subject_partitions)
    )
    in_fps = out_fps

    logger.info("Tensorizing MEDS shards...")
    Parallel(n_jobs=n_jobs, verbose=False)(
        delayed(TimelineDataset.tensorize)(in_fp, in_fp.parent.with_name(in_fp.name), vocab)
        for in_fp in in_fps
    )

    logger.info("Counting codes...")
    output_dir = output_dir.with_name("03_CodeCounter")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_fps = [output_dir / f"{i}.parquet" for i, _ in enumerate(subject_partitions, shard_offset)]
    Parallel(n_jobs=n_jobs, verbose=False)(
        delayed(apply_transforms)(in_fp, out_fp, CodeCounter)
        for in_fp, out_fp in zip(in_fps, out_fps)
    )
    CodeCounter.agg(in_fps=out_fps, out_fp=output_dir.with_name("code_counts.csv"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
