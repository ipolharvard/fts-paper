import abc
import pickle
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import polars as pl
import torch as th
from safetensors.torch import save_file

from ..constants import PATIENT_ID_COL, STATIC_DATA_FN
from ..constants import SpecialToken as ST
from ..vocabulary import Vocabulary
from ._sharded_data import ShardedData


class TimelineDataset(th.utils.data.Dataset):
    def __init__(
        self, input_dir: str | Path, n_positions: int = 2048, is_encoder_decoder: bool = False
    ):
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        self._data = ShardedData(input_dir)

        self.vocab = Vocabulary.from_path(input_dir)
        self._num_quantiles = len(self.vocab.quantile_stokens)

        static_data_fps = list(input_dir.glob(f"{STATIC_DATA_FN.split('.')[0]}*.parquet"))
        if static_data_fps:
            sorted_cols = sorted(pl.read_parquet_schema(static_data_fps[0]).keys())
            self.static_data = (
                pl.concat(
                    [pl.scan_parquet(fp).select(sorted_cols) for fp in static_data_fps],
                    how="vertical_relaxed",
                )
                .collect()
                .rows_by_key(PATIENT_ID_COL, named=True, unique=True)
            )
        else:
            # Legacy
            self.static_data = {}
            for fp in input_dir.glob(f"{STATIC_DATA_FN.split('.')[0]}*.pickle"):
                self.static_data.update(pickle.load(fp.open("rb")))

        # plus one, because DOB takes 2 spots
        self.context_size = len(next(iter(self.static_data.values()))) + 1
        self.timeline_size = n_positions - self.context_size

        self.is_encoder_decoder = is_encoder_decoder
        if is_encoder_decoder:
            self.timeline_size = n_positions

        self.modality_tokens = th.tensor(
            self.vocab.encode(
                [
                    stoken
                    for stoken in self.vocab
                    if stoken.startswith("CXR//") or stoken.startswith("NOTE//")
                ]
            )
        )

    @property
    def tokens(self):
        return self._data.tokens

    @property
    def times(self):
        return self._data.times

    @property
    def patient_ids(self) -> th.Tensor:
        return th.cat([shard["patient_ids"] for shard in self._data.shards])

    @property
    def patient_id_at_idx(self):
        return self._data.patient_id_at_idx

    @property
    def patient_offsets(self) -> th.Tensor:
        return th.cat([shard["patient_offsets"] + shard["offset"] for shard in self._data.shards])

    @property
    def patient_offset_at_idx(self):
        """Aka patient data start at idx."""
        return self._data.patient_offset_at_idx

    @property
    def patient_data_end_at_idx(self):
        return self._data.patient_data_end_at_idx

    @property
    def is_mimic(self):
        return "hadm_id" in self._data.shards[0]

    @property
    def hadm_id(self):
        if not self.is_mimic:
            raise AttributeError("It's not MIMIC, no 'hadm_id' available.")
        return self._data.hadm_id

    @property
    def icu_stay_id(self):
        if not self.is_mimic:
            raise AttributeError("It's not MIMIC, no 'icustay_id' available.")
        return self._data.icu_stay_id

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(len={len(self):,}, "
            f"patient_num={len(self.patient_ids):,}, "
            f"vocab_size={len(self.vocab):,})"
        )

    def __len__(self) -> int:
        return len(self.tokens) - self.timeline_size

    def __getitem__(self, idx: int) -> tuple[th.Tensor | tuple, th.Tensor]:
        pt_ctx = self._get_patient_context(idx)
        timeline = self.tokens[idx : idx + self.timeline_size + 1]

        if self.is_encoder_decoder:
            return (pt_ctx, timeline[:-1]), timeline[1:]

        x = th.cat((pt_ctx, timeline[:-1]))
        y = th.cat((pt_ctx, timeline[1:]))
        y[: self.context_size] = -100
        return x, y

    def _get_patient_context(self, idx: int) -> th.Tensor:
        patient_id = self.patient_id_at_idx[idx].item()
        time_at_start = self.times[idx].item()

        static_tokens = []
        for token in self.static_data[patient_id].values():
            if token["code"][0] == ST.DOB:
                age = timedelta(microseconds=time_at_start - token["time"][0])
                static_tokens.extend(self._age_to_tokens(age.days / 365.25))
            elif len(token["code"]) == 1:
                static_tokens.append(token["code"][0])
            else:
                idx = self._find_closest_index(token["time"], time_at_start)
                static_tokens.append(token["code"][idx])
        return th.tensor(self.vocab.encode(static_tokens))

    def _age_to_tokens(self, age_years: float) -> tuple[str, str]:
        age_scaled = age_years * self._num_quantiles**2 / 100
        age_scaled = min(age_scaled, self._num_quantiles**2 - 1)

        age_t1 = int(age_scaled // self._num_quantiles)
        age_t2 = round(age_scaled % self._num_quantiles)
        if age_t2 == self._num_quantiles:
            age_t1 += 1
            age_t2 = 0

        return f"Q{age_t1 + 1}", f"Q{age_t2 + 1}"

    @staticmethod
    def _find_closest_index(ll: list, target_value: float) -> int:
        return min(range(len(ll)), key=lambda i: abs(ll[i] - target_value))

    @staticmethod
    def tensorize(in_fp: str | Path | list, out_fp: str | Path, vocab: Vocabulary):
        df = (
            pl.scan_parquet(in_fp)
            .with_columns(
                tokens=pl.col("code").replace_strict(
                    vocab.stoi, default=None, return_dtype=pl.Int64
                ),
                times=pl.col("time").cast(pl.Int64),
            )
            # Filter out negative times, dates before 1970 are most likely errors
            .filter(pl.col("time") >= 0)
            .collect()
        )

        if df.select(pl.col("tokens").is_null().any()).item():
            not_translated = (
                pl.scan_parquet(in_fp)
                .select(
                    "code",
                    translated=pl.col("code").replace_strict(
                        vocab.stoi, default=None, return_dtype=pl.Int64
                    ),
                )
                .filter(pl.col("translated").is_null())
                .select(pl.col("code").unique())
                .collect()
                .get_column("code")
                .to_list()
            )
            raise RuntimeError(
                f"The following codes are not present in the provided vocab: {not_translated}"
            )

        patient_df = (
            df.set_sorted(PATIENT_ID_COL)
            .group_by(PATIENT_ID_COL, maintain_order=True)
            .agg(len=pl.len())
            .select(PATIENT_ID_COL, offsets=pl.col("len").cum_sum().shift(1, fill_value=0))
        )
        tensors = {
            "tokens": df["tokens"].to_torch(),
            "times": df["times"].to_torch(),
            "patient_ids": patient_df[PATIENT_ID_COL].to_torch(),
            "patient_offsets": patient_df["offsets"].to_torch(),
        }

        for mimic_col in ["hadm_id", "icustay_id"]:
            if mimic_col in df.columns:
                tensors[mimic_col] = df[mimic_col].to_torch()

        save_file(tensors, Path(out_fp).with_suffix(".safetensors"))


class InferenceDataset(TimelineDataset, abc.ABC):
    # INFERENCE DEFAULT CONSTRAINTS
    stop_stokens: list[ST] = [ST.DEATH, ST.TIMELINE_END]  # Default inference stop tokens
    time_limit: timedelta = timedelta(days=365.25 * 2)  # Inference time constraint

    def _get_hadm_id(self, idx: int) -> int | None:
        return None if th.isnan(hadm_id := self.hadm_id[idx]) else int(hadm_id)

    def _get_icu_stay_id(self, idx: int) -> int | None:
        return None if th.isnan(icu_stay_id := self.icu_stay_id[idx]) else int(icu_stay_id)

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, idx: int) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        data_start_idx = self.patient_offset_at_idx[idx]
        if idx - data_start_idx + 1 > self.timeline_size:
            data_start_idx = idx + 1 - self.timeline_size

        pt_ctx = self._get_patient_context(data_start_idx)
        timeline = self.tokens[data_start_idx : idx + 1]
        if self.is_encoder_decoder:
            return (pt_ctx, timeline)
        return th.cat((pt_ctx, timeline))

    def _get_indices_of_stokens(self, stokens: str | Sequence[str]) -> th.Tensor:
        if isinstance(stokens, str):
            stokens = [stokens]
        tokens_of_interest = th.tensor(self.vocab.encode(stokens))
        shard_indices = []
        token_offset = 0
        for token_chunk in self.tokens:
            new_indices = th.nonzero(th.isin(token_chunk, tokens_of_interest)).view(-1)
            new_indices += token_offset
            shard_indices.append(new_indices)
            token_offset += len(token_chunk)

        return th.cat(shard_indices)

    def _match(
        self,
        ordered_sequence: th.Tensor,
        input: th.Tensor,
        *,
        fill_unmatched: int | None = None,
        shift: int = 0,
    ) -> th.Tensor:
        """TODO: Write a docstring, because this function is hell."""
        ordered_sequence_indices = th.searchsorted(ordered_sequence, input, right=True)
        if shift:
            ordered_sequence_indices += shift
        if fill_unmatched is None:
            return ordered_sequence[ordered_sequence_indices]
        else:
            out = th.full_like(input, fill_value=fill_unmatched)
            mask = ordered_sequence_indices < len(ordered_sequence)
            out[mask] = ordered_sequence[ordered_sequence_indices[mask]]
            return out

    def _move_idx_to_last_same_time(self, token_index: th.Tensor) -> th.Tensor:
        """Shifts index to the last token with the same time of the token at the index."""
        data_end_idx = self.patient_data_end_at_idx[token_index]
        times = self.times[token_index : data_end_idx - 1]
        idx_offset = th.searchsorted(times[1:], times[0], right=True)
        return token_index + idx_offset

    def _move_indices_to_last_same_time(self, token_indices: th.Tensor) -> th.Tensor:
        """Shifts indices to the last token with the same time of the token at the index."""
        new_indices = th.empty_like(token_indices)
        for i, idx in enumerate(token_indices):
            new_indices[i] = self._move_idx_to_last_same_time(idx)
        return new_indices
