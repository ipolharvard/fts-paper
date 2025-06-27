import functools
import random
import time
from pathlib import Path

import polars as pl
from MEDS_transforms.mapreduce.utils import rwlock_wrap

from ..vocabulary import Vocabulary
from .utils import wait_for_workers


def run_stage(
    in_fps,
    out_fps,
    *transform_fns,
    params={},
    vocab=None,
    agg_to=None,
    agg_params=None,
    worker: int = 0,
    **kwargs,
):
    """This function can be run in parallel by multiple workers."""

    if vocab is not None:
        params = {"vocab": Vocabulary.from_path(vocab), **params}

    transforms_to_run = [
        functools.partial(transform_fn, **params) for transform_fn in transform_fns
    ]

    fps = list(zip(in_fps, out_fps))
    random.shuffle(fps)

    for in_fp, out_fp in fps:
        rwlock_wrap(
            in_fp,
            out_fp,
            functools.partial(pl.read_parquet, use_pyarrow=True),
            lambda df, out_: df.write_parquet(out_, use_pyarrow=True),
            compute_fn=lambda df: functools.reduce(lambda df, fn: fn(df), transforms_to_run, df),
        )

    if agg_to is not None:
        agg_to = Path(agg_to)
        if worker == 0:
            wait_for_workers(out_fps[0].parent)
            transform_fns[-1].agg(in_fps=out_fps, out_fp=agg_to, **(agg_params or {}))
        else:
            while not agg_to.exists():
                time.sleep(2)
