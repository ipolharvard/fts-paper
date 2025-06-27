import shutil
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from ..constants import STATIC_DATA_FN
from ..constants import SpecialToken as ST
from ..datasets import TimelineDataset
from ..vocabulary import Vocabulary
from .run_stage import run_stage
from .utils import load_function, wait_for_workers

OmegaConf.register_new_resolver("default_if_null", lambda a, b: b if a is None else a)
OmegaConf.register_new_resolver("is_not_null", lambda v: v is not None)


@hydra.main(version_base=None, config_path="../configs", config_name="tokenization")
def main(cfg: DictConfig):
    if not (input_dir := Path(cfg.input_dir)).is_dir():
        raise FileNotFoundError(f"Data directory '{input_dir}' not found.")

    dataset = cfg.dataset.name
    out_fn = f"{dataset}_{input_dir.name}" if cfg.out_fn is None else cfg.out_fn
    output_dir = Path(cfg.output_dir) / out_fn
    # Update the real output directory path
    cfg.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Tokenizing '{input_dir}' using the {dataset.upper()} preprocessing pipeline.")

    in_fps = list(input_dir.glob("*.parquet"))
    logger.info(f"Found {len(in_fps)} patient splits in '{input_dir}'.")
    assert in_fps, "No parquet files found!"

    i = 1
    for stage_cfg in cfg.dataset.stages:
        if stage_cfg.get("skip", False):
            logger.info(f"Skipping stage {stage_cfg.name}")
            continue

        stage_dir = output_dir / f"{i:02}_{stage_cfg.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        out_fps = [stage_dir / fp.name for fp in in_fps]

        transform_fns = []
        if "transforms" in stage_cfg:
            for transform_name in stage_cfg.transforms:
                func = load_function(transform_name, f"ethos.tokenize.{dataset}")
                transform_fns.append(func)
        else:
            func = load_function(stage_cfg.name, "ethos.tokenize.common")
            transform_fns.append(func)

        run_stage(in_fps, out_fps, *transform_fns, worker=cfg.worker, **stage_cfg)

        # Make workers wait for lock file deletion to avoid moving on prematurely
        wait_for_workers(stage_dir)

        if "agg_to" not in stage_cfg:
            in_fps = out_fps

        i += 1

    if cfg.worker == 0:
        if cfg.vocab is None:
            vocab = Vocabulary()
            static_codes = (
                pl.scan_parquet(output_dir / STATIC_DATA_FN)
                .select(
                    pl.concat_list(pl.exclude("subject_id").struct["code"])
                    .explode()
                    .unique()
                    .sort()
                )
                .filter(pl.col("code") != ST.DOB)
                .collect()
                .to_series()
            )
            vocab.add_words(static_codes)

            codes = pl.read_csv(output_dir / cfg.code_counts_fn, columns="code")["code"].to_list()
            vocab.add_words(codes)
        else:
            vocab = Vocabulary.from_path(cfg.vocab)
            if (quantile_fp := Path(cfg.vocab) / cfg.quantiles_fn).exists():
                logger.info(f"Copying quantile breaks from {quantile_fp} to {output_dir}")
                shutil.copy(quantile_fp, output_dir / cfg.quantiles_fn)

            if (intervals_fp := Path(cfg.vocab) / cfg.intervals_fn).exists():
                logger.info(f"Copying intervals from {intervals_fp} to {output_dir}")
                shutil.copy(intervals_fp, output_dir / cfg.intervals_fn)
            else:
                logger.warning(f"Intervals file not found at {intervals_fp}")

        vocab.dump(output_dir)
        for in_fp in in_fps:
            TimelineDataset.tensorize(in_fp, output_dir / in_fp.name, vocab)


if __name__ == "__main__":
    main()
