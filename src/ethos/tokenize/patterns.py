import warnings
from collections.abc import Callable
from pathlib import Path

import polars as pl

from ..constants import PATIENT_ID_COL
from ..vocabulary import Vocabulary
from .utils import create_prefix_or_chain, unify_code_names


class MatchAndRevise:
    sort_cols = [PATIENT_ID_COL, "time"]
    index_col = "__idx"

    def __init__(
        self,
        *,
        prefix: str | list[str],
        needs_resorting: bool = False,
        apply_vocab: bool = False,
        needs_counts: bool = False,
        needs_vocab: bool = False,
    ):
        """Match&Revise pattern implementation for the convenient transformations of MEDS datasets.

        Args:
            prefix: Prefixes of codes that will be passed to the transform function to work with.
            needs_resorting: If True, the resulting DataFrame will be resorted by `self.sort_cols`.
            Otherwise, the initial order will be preserved by putting the processed event back in
            the rows they originated from.
            apply_vocab: If True, and vocabulary is provided, only the codes preset in the
            vocabulary will be retained in the processed subset of data.
            needs_counts: If True, the transform function will require a 'counts' argument to be
            passed into it, but only if vocabulary is not provided. Otherwise, the 'counts'
            argument will always be `None`.
        """
        self._prefix = prefix if isinstance(prefix, list) else [prefix]
        self._needs_resorting = needs_resorting
        self._apply_vocab = apply_vocab
        self._needs_vocab = needs_vocab
        self._needs_counts = needs_counts

    def __call__(
        self, fn: Callable[[pl.DataFrame, ...], pl.DataFrame]
    ) -> Callable[[pl.DataFrame, ...], pl.DataFrame]:
        def out_fn(
            df: pl.DataFrame, *, vocab: dict[str, int] | str | None = None, **kwargs
        ) -> pl.DataFrame:
            if self._needs_vocab:
                if isinstance(vocab, str):
                    vocab = Vocabulary.from_path(vocab)
                kwargs["vocab"] = list(vocab) if vocab is not None else None

            if self._needs_counts:
                if vocab is None:
                    if (counts := kwargs.get("counts", None)) is None:
                        warnings.warn("Expected `counts` is not None if `vocab` is None")
                    else:
                        if not isinstance(counts, dict):
                            counts = dict(zip(*pl.read_csv(counts)[:, [0, 1]]))
                        kwargs["counts"] = {
                            code: counts[code]
                            for code in counts.keys()
                            if any(code.startswith(p) for p in self._prefix)
                        }
                else:
                    kwargs["counts"] = None

            df = df.with_row_index(self.index_col)

            input_row_mask = create_prefix_or_chain(self._prefix)
            in_df = df.filter(input_row_mask)
            df = df.filter(~input_row_mask)

            new_events = (
                fn(in_df, **kwargs)
                .cast(df.collect_schema())
                .with_columns(unify_code_names(pl.col("code")))
            )

            if isinstance(new_events, pl.DataFrame) and not new_events[self.index_col].is_sorted():
                raise ValueError(
                    "The resulting DataFrame is expected to be sorted by the index"
                    f" column: '{self.index_col}'"
                )

            if vocab is not None and self._apply_vocab:
                new_events = new_events.filter(pl.col("code").is_in(list(vocab)))

            df = df.merge_sorted(new_events, key=self.index_col)
            if self._needs_resorting:
                df = df.sort(by=self.sort_cols, maintain_order=True)

            return df.drop(self.index_col)

        return out_fn


class ScanAndAggregate:
    def __call__(self, *args, **kwargs) -> pl.DataFrame:
        pass

    def agg(self, in_fps: list[str | Path], out_fp: str | Path) -> None:
        pl.scan_parquet(in_fps).sink_parquet(out_fp)
