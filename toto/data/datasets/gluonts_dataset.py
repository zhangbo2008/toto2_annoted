# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from __future__ import annotations

from functools import cached_property
from itertools import islice
from math import ceil
from typing import Any, Callable, Iterable, List, Mapping, cast

import datasets as hfds
import numpy as np
import torch
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TrainingDataset, split
from gluonts.itertools import Map
from torch.utils.data import Dataset

from ..util.dataset import CausalMaskedTimeseries
from ..util.helpers import (
    build_id_mask,
    ensure_variate_first,
    itemize_start,
    preprocess_exogenous_features,
    transform_fev_dataset,
)


class GluonTSDatasetView:
    """
    Minimal wrapper that adapts a HuggingFace dataset into the data format
        expected by **GluonTS**.


        This class performs three main tasks:
          • Accept a HuggingFace dataset with custom fields

          • Transforms it to the format expected by GluonTS, where each item has the following fields:
            time-series fields such as:
                - "target": numpy array shaped (D, T) or (T,)
                - "feat_dynamic_real": numpy array of real-valued dynamic features,
                                       shaped (K, T)
                - "freq": string representing the sampling frequency (e.g. "5min")
                - "start": timestamp of the first time step

          • Convert each item to a GluonTS-compatible data entry using
            `ProcessDataEntry` after applying optional transformations to the
            target or dynamic features.

          • Provide train/validation/test splits by applying `gluonts.split`,
            where the number of rolling windows for evaluation is computed as
            ~10% of the minimal series length (unless overridden).

        The view exposes three datasets:
          - `training_dataset`
          - `validation_dataset`
          - `test_data`

        All of them follow GluonTS's iterables of dict-like entries.
    """

    def __init__(
        self,
        dataset: hfds.Dataset,
        patch_size: int,
        max_context_length: int,
        prediction_horizon: int,
        test_windows: int | None = None,
        target_fields: List[str] = ["target"],
        target_transform_fns: List[Callable[[np.ndarray], np.ndarray]] | None = None,
        ev_fields: List[str] = ["feat_dynamic_real"],
        ev_transform_fns: List[Callable[[np.ndarray], np.ndarray]] | None = None,
        test_split_fraction: float = 0.1,
        max_windows: int = 20,
        max_val_series: int = 100,
        max_rows: int | None = None,
    ):
        self.dataset = dataset
        self._patch_size = patch_size
        self._max_context_length = max_context_length
        self._target_fields = target_fields
        self._target_transform_fns = target_transform_fns
        self._ev_fields = ev_fields
        self._ev_transform_fns = ev_transform_fns
        self._test_split_fraction = test_split_fraction
        self._max_windows = max_windows
        self._max_val_series = max_val_series
        self._prediction_horizon = prediction_horizon
        self._max_rows = max_rows
        # Preprocess the HF dataset: apply optional transformations and ensure
        # that each example contains GluonTS-style fields (target, dynamic feats...).
        self.hf_dataset = transform_fev_dataset(
            dataset, target_fields, target_transform_fns, ev_fields, ev_transform_fns
        )

        def _series_length(ex):
            t = ex["target"]
            # t is typically np.ndarray with shape (D, T) or (T,)
            if hasattr(t, "shape"):
                return t.shape[-1]
            return len(t)

        def assess_series_length(ex):
            test_length = ceil(_series_length(ex) * self._test_split_fraction)
            train_length = _series_length(ex) - test_length
            # training split must at least have one window for training and one for validation
            train_bool = train_length >= 3 * self._patch_size
            # test split must at least have one window for testing
            test_bool = test_length >= self._prediction_horizon
            return train_bool and test_bool

        self.hf_dataset = self.hf_dataset.filter(assess_series_length)

        assert len(self.hf_dataset) > 0, "No series left after filtering by minimum length"

        # clip dataset to max rows if specified
        if self._max_rows is not None and len(self.hf_dataset) > self._max_rows:
            self.hf_dataset = self.hf_dataset.select(range(self._max_rows))

        # Build a GluonTS dataset
        process = ProcessDataEntry(
            freq=self.freq,
            one_dim_target=self.target_dim == 1,
        )
        self.gluonts_dataset = Map(lambda x: process(itemize_start(x)), self.hf_dataset)

        # Number of rolling windows is either user-provided or will be computed lazily.
        self._windows_override = test_windows

    @cached_property
    def freq(self) -> str:
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        target = self.hf_dataset[0]["target"]
        return target.shape[0] if getattr(target, "ndim", 1) > 1 else 1

    @property
    def prediction_length(self) -> int:
        return self._patch_size

    @cached_property
    def _min_series_length(self) -> int:
        """
        Minimal series length over all HF items, measured along time axis.
        """
        lengths: List[int] = []
        for i in range(len(self.hf_dataset)):
            # this takes in the already transformed target
            t_np = self.hf_dataset[i]["target"]
            t = torch.as_tensor(t_np)
            t_vt = cast(torch.Tensor, t)
            lengths.append(t_vt.shape[-1])
        return min(lengths)

    @property
    def _context_length(self) -> int:
        # possible context length is at max the length of the training data minus 2 windows for training and validation
        possible_context_length = (1 - self._test_split_fraction) * self._min_series_length - 2 * self._patch_size

        assert possible_context_length >= self._patch_size, "Possible context length is less than patch size"

        # clip context length to the nearest multiple of the patch size
        clipped_context_length = (possible_context_length // self._patch_size) * self._patch_size

        return min(self._max_context_length, int(clipped_context_length))

    @property
    def training_dataset(self) -> TrainingDataset:
        """
        Training region:
        Everything before the last `test_split_fraction * min_series_length + patch_size` time steps.
        """
        train_ds, _ = split(
            self.gluonts_dataset,
            offset=-ceil(self._test_split_fraction * self._min_series_length + self._patch_size),
        )
        return train_ds

    @property
    def validation_dataset(self) -> TrainingDataset:
        """
        Validation region:
        Boundary at `-test_split_fraction * min_series_length`.
        This means validation ends exactly before the test windows begin.
        """
        val_ds, _ = split(
            self.gluonts_dataset,
            offset=-ceil(self._test_split_fraction * self._min_series_length),
        )
        # clip validation dataset to max series (rows) if specified
        if self._max_val_series > 0:
            val_ds = list(islice(val_ds, self._max_val_series))

        return val_ds


# ---------------------------------------------------------------------------
# GluonTS instance → CausalMaskedTimeseries Dataset
# ---------------------------------------------------------------------------


def _extract_targets_and_exogenous(
    instance: Mapping[str, Any],
    target_field: str,
    ev_field: str,
    context_length: int,
    patch_size: int,
    add_exogenous_features: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Extract past/future target arrays (and optionally exogenous features)
    from a GluonTS-style instance and ensure (variates, time) layout.
    """
    # Targets
    past = torch.as_tensor(instance[f"past_{target_field}"], dtype=torch.float32)
    future = torch.as_tensor(instance[f"future_{target_field}"], dtype=torch.float32)

    past_vt = cast(torch.Tensor, ensure_variate_first(past))
    future_vt = cast(torch.Tensor, ensure_variate_first(future))

    exogenous_features: torch.Tensor | None = None
    if add_exogenous_features and f"past_{ev_field}" in instance:
        past_exog = torch.as_tensor(instance[f"past_{ev_field}"], dtype=torch.float32)
        future_exog = torch.as_tensor(instance[f"future_{ev_field}"], dtype=torch.float32)
        exogenous_features = preprocess_exogenous_features(
            past_exog,
            future_exog,
            context_length=context_length,
            patch_size=patch_size,
        )

    return past_vt, future_vt, exogenous_features


def _stack_targets_and_exogenous(
    past_vt: torch.Tensor,
    future_vt: torch.Tensor,
    exogenous_features: torch.Tensor | None,
    add_exogenous_features: bool,
) -> tuple[torch.Tensor, int, int, int]:
    """
    Build the full (variates, time) window tensor, optionally stacking
    exogenous features along the variate dimension.
    """
    # Compose target window: [num_target_variates, time_steps]
    window = torch.cat([past_vt, future_vt], dim=-1)

    num_target_variates, time_steps = window.shape[-2], window.shape[-1]
    num_exogenous_variates = 0

    if add_exogenous_features and exogenous_features is not None:
        # Sanity check that time dimension matches
        assert exogenous_features.shape[-1] == time_steps, (
            f"Exogenous features shape last dimension {exogenous_features.shape[-1]} "
            f"does not match window shape last dimension {time_steps}"
        )
        num_exogenous_variates = exogenous_features.shape[-2]
        # Stack exogenous features below target features along variates
        window = torch.cat([window, exogenous_features], dim=-2)

    return window, num_target_variates, num_exogenous_variates, time_steps


def _build_masks_and_metadata(
    num_target_variates: int,
    num_exogenous_variates: int,
    time_steps: int,
    patch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct padding mask, id mask, timestamps, and time intervals for
    a CausalMaskedTimeseries.
    """
    num_variates = num_target_variates + num_exogenous_variates

    # Construct padding mask
    padding_mask = torch.ones((num_variates, time_steps), dtype=torch.bool, device=device)
    if num_exogenous_variates > 0:
        padding_mask[num_target_variates : num_target_variates + num_exogenous_variates, -patch_size:] = False

    # Group/series ids; exogenous features currently share the same grouping logic
    id_mask = build_id_mask(
        num_variates=num_variates,
        time_steps=time_steps,
        device=device,
        dtype=torch.long,
    )

    # Simple index-based timestamps and unit time intervals
    timestamps = torch.arange(time_steps, dtype=torch.long, device=device).unsqueeze(0).expand(num_variates, time_steps)
    time_interval_seconds = torch.ones((num_variates,), dtype=torch.long, device=device)

    return padding_mask, id_mask, timestamps, time_interval_seconds


def instance_to_causal(
    instance,
    context_length: int,
    patch_size: int,
    add_exogenous_features: bool = False,
    target_field: str = "target",
    ev_field: str = "feat_dynamic_real",
) -> CausalMaskedTimeseries:
    """
    Convert a single GluonTS-style instance into a `CausalMaskedTimeseries`.

    This helper:
      1. Extracts past and future target values (and optionally exogenous features).
      2. Builds a (variates, time) window stacking past and future segments.
      3. Optionally stacks exogenous covariates along the variate dimension.
      4. Constructs padding / id masks and simple timestamp metadata.
      5. Defines input and target slices for autoregressive training.

    Parameters
    ----------
    instance:
        A GluonTS-style instance or (input, label) pair produced by
        an `InstanceSplitter` or a test template.
    context_length:
        Number of time steps used as model input context.
    patch_size:
        Prediction length (number of future time steps).
    add_exogenous_features:
        If True, attempt to include exogenous covariates under `ev_field`.
    target_field:
        Name of the target field in the instance (default: "target").
    ev_field:
        Name of the exogenous (covariate) field in the instance
        (default: "feat_dynamic_real").

    Returns
    -------
    CausalMaskedTimeseries
        A fully constructed causal timeseries object ready to be batched
        by `collate_causal`.
    """
    # 1) Extract target & exogenous tensors from the raw instance
    past_vt, future_vt, exogenous_features = _extract_targets_and_exogenous(
        instance=instance,
        target_field=target_field,
        ev_field=ev_field,
        context_length=context_length,
        patch_size=patch_size,
        add_exogenous_features=add_exogenous_features,
    )

    # 2) Build full window and optionally stack exogenous channels
    window, num_target_variates, num_exogenous_variates, time_steps = _stack_targets_and_exogenous(
        past_vt=past_vt,
        future_vt=future_vt,
        exogenous_features=exogenous_features,
        add_exogenous_features=add_exogenous_features,
    )

    device = window.device

    # 3) Build masks and simple time metadata
    padding_mask, id_mask, timestamps, time_interval_seconds = _build_masks_and_metadata(
        num_target_variates=num_target_variates,
        num_exogenous_variates=num_exogenous_variates,
        time_steps=time_steps,
        patch_size=patch_size,
        device=device,
    )

    # 4) Define slices for input and target regions
    context_len = past_vt.shape[-1]
    pred_len = future_vt.shape[-1]

    # NOTE: targets are a shifted version of the input series to the future
    input_slice = slice(0, context_len)
    target_slice = slice(pred_len, context_len + pred_len)

    return CausalMaskedTimeseries(
        series=window,
        padding_mask=padding_mask,
        id_mask=id_mask,
        timestamp_seconds=timestamps,
        time_interval_seconds=time_interval_seconds,
        input_slice=input_slice,
        target_slice=target_slice,
        num_exogenous_variables=num_exogenous_variates,
    )


class GluonTSInstanceDataset(Dataset):
    """
    Materializes gluonts instances into a list of CausalMaskedTimeseries.

    - For train/val: instances are dicts from InstanceSplitter (past_target/future_target).
    - For test: instances are (input_entry, label_entry) tuples from TestData.
    """

    def __init__(
        self,
        instances: Iterable[DataEntry] | Iterable[tuple[DataEntry, DataEntry]],
        context_length: int,
        patch_size: int,
        add_exogenous_features: bool = False,
        target_field: str = "target",
        ev_field: str = "feat_dynamic_real",
    ):
        self._instances: List[CausalMaskedTimeseries] = []
        for inst in instances:
            ts = instance_to_causal(inst, context_length, patch_size, add_exogenous_features, target_field, ev_field)
            self._instances.append(ts)

    def __len__(self) -> int:
        return len(self._instances)

    def __getitem__(self, idx: int) -> CausalMaskedTimeseries:
        return self._instances[idx]
