from datetime import timedelta
from pathlib import Path

import polars as pl
import torch as th

from ..constants import SpecialToken as ST
from ..datasets.base import InferenceDataset


class SyntheticDataset(InferenceDataset):

    def __init__(self, input_dir: str | Path, n_positions: int = 2048, **kwargs):
        super().__init__(input_dir, n_positions, **kwargs)
        self.stop_stokens = [ST.TIMELINE_END]
        self.time_limit = timedelta(days=365 * 10)

        static_data_df = pl.from_dicts(
            {
                "subject_id": self.static_data.keys(),
                "data": self.static_data.values(),
            }
        )
        static_data_cols = static_data_df["data"].struct.fields
        self.static_data_df = (
            static_data_df.lazy()
            .unnest("data")
            .select(
                pl.col(ST.DOB).struct[1].list.first().cast(pl.Datetime).alias(ST.DOB),
                *[
                    pl.col(col).struct[0].list.first().alias(col)
                    for col in static_data_cols
                    if col != ST.DOB
                ],
                pl.col("subject_id")
                .replace_strict(
                    {
                        subject_id.item(): self.times[idx].item()
                        for subject_id, idx in zip(self.patient_ids, self.patient_offsets)
                    },
                    default=None,
                    return_dtype=pl.Datetime,
                )
                .alias("timeline_start_date"),
            )
            .filter(pl.col("timeline_start_date").is_not_null())
            .select(
                # the order of columns matters
                *static_data_cols,
                "timeline_start_date",
                age=(pl.col("timeline_start_date") - pl.col(ST.DOB)).dt.total_days() / 365.25,
            )
            .collect()
        )

    def __len__(self) -> int:
        return len(self.static_data_df)

    def __getitem__(self, idx: int) -> tuple[pl.Series, dict]:
        static_data = self.static_data_df.row(idx, named=True)
        timeline_start_date = static_data.pop("timeline_start_date")
        age = static_data.pop("age")

        timeline = []
        for token_type, token in static_data.items():
            if token_type == ST.DOB:
                timeline.extend(self._age_to_tokens(age))
            else:
                timeline.append(token)
        timeline.append(str(ST.TIMELINE_START))

        return th.tensor(self.vocab.encode(timeline)), {
            "expected": static_data,
            "timeline_start_date": timeline_start_date,
        }
