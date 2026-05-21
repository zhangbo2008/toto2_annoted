# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from __future__ import annotations

from typing import Callable, List, overload

import datasets as hfds
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int

from .dataset import CausalMaskedTimeseries

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@overload
def ensure_variate_first(target: torch.Tensor) -> torch.Tensor: ...
@overload
def ensure_variate_first(target: np.ndarray) -> np.ndarray: ...


def ensure_variate_first(target: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Convert a target array to shape [variates, time].
    Accepts shapes:
      - [time] → [1, time]
      - [time, variate] → [variate, time]

    (If your HF data uses [time, variates], adjust here accordingly.)
    """
    if isinstance(target, torch.Tensor):
        if target.ndim == 1:
            # [time] -> [1, time]
            return target.unsqueeze(0)
        if target.ndim == 2:
            # Assume [time, variate] → transpose to [variate, time] for GluonTS compatibility
            return target.transpose(1, 0)
        raise ValueError(f"Expected target with ndim in {{1,2}}, got {target.shape}")

    if isinstance(target, np.ndarray):
        if target.ndim == 1:
            # [time] -> [1, time]
            return np.expand_dims(target, axis=0)
        if target.ndim == 2:
            # Assume [time, variate] → transpose to [variate, time] for GluonTS compatibility
            return np.transpose(target, (1, 0))
        raise ValueError(f"Expected target with ndim in {{1,2}}, got {target.shape}")

    # This should be unreachable given the type hints, but kept for safety.
    raise TypeError(f"ensure_variate_first expects torch.Tensor or np.ndarray, got {type(target)}")


def build_id_mask(num_variates: int, time_steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create an id_mask tensor of shape [variates, time] assigning the SAME id to all variates.
    """
    return torch.zeros((num_variates, time_steps), dtype=dtype, device=device)


def itemize_start(data_entry):
    # GluonTS wants scalar start, but HF stores it as 0-d array sometimes
    data_entry = dict(data_entry)
    if "start" in data_entry and hasattr(data_entry["start"], "item"):
        data_entry["start"] = data_entry["start"].item()
    return data_entry


def _check_dataset_fields(hf_dataset: hfds.Dataset) -> bool:
    """
    Check if the dataset conforms to the format expected by the GluonTS dataset.
    """
    # check if the dataset has a start field
    if "start" not in hf_dataset.features:
        return False
    # check if the dataset has a freq field
    if "freq" not in hf_dataset.features:
        return False
    # check if the dataset has a target field
    if "target" not in hf_dataset.features:
        return False
    return True


def _assert_fev_compatible(
    ds: hfds.Dataset,
    target_fields: List[str],
    ev_fields: List[str],
) -> None:
    """
    Verify that `ds` is compatible with the FEV input scheme:

      - Must contain "timestamp" of shape (T,)
      - Each target field in `target_fields` must exist and have shape (T,) or (1, T)
      - Each exogenous field in `ev_fields` must exist and have shape (T,) or (1, T)
      - All series must share the same time length T
    """

    # --- Check timestamp structure ---
    if "timestamp" not in ds.features:
        raise ValueError("Dataset missing required field 'timestamp'.")

    ts0 = ds[0]["timestamp"]
    if not isinstance(ts0, (list, np.ndarray)):
        raise ValueError("'timestamp' must be an array-like of datetimes or numbers.")

    ts0 = np.asarray(ts0)
    if ts0.ndim != 1:
        raise ValueError("'timestamp' must be a 1-D array.")
    T = len(ts0)

    def _check_series(name: str):
        if name not in ds.features:
            raise ValueError(f"Dataset missing required field '{name}'.")

        arr = np.asarray(ds[0][name])
        if arr.ndim == 1:
            if arr.shape[0] != T:
                raise ValueError(f"Field '{name}' must have length T={T}, " f"but got {arr.shape}.")
        elif arr.ndim == 2:
            # Accept shapes (1, T) or (D, T)
            if arr.shape[-1] != T:
                raise ValueError(
                    f"Field '{name}' must have time dimension T={T} in last axis, " f"but got {arr.shape}."
                )
        else:
            raise ValueError(f"Field '{name}' must be 1-D or 2-D time series, but got {arr.ndim} dims.")

    # --- Check all target fields ---
    for f in target_fields:
        _check_series(f)

    # --- Check all ev (exogenous) fields ---
    for f in ev_fields:
        _check_series(f)


def transform_fev_dataset(
    hf_dataset: hfds.Dataset,
    target_fields: List[str] = ["target"],
    target_transform_fns: List[Callable[[np.ndarray], np.ndarray]] | None = None,
    ev_fields: List[str] = ["feat_dynamic_real"],
    ev_transform_fns: List[Callable[[np.ndarray], np.ndarray]] | None = None,
) -> hfds.Dataset:
    """
    Transform HF datasets from fev_benchmark to the format expected by the GluonTS dataset.

    Input format (fev benchmark datasets):
      Each item in `hf_dataset` is expected to contain:
        - "timestamp": 1D array-like of shape (T,)
                       Monotonic sequence of datetimes or timestamp-like values.
        - One univariate series per target field in `target_fields`:
            For each name `f` in `target_fields`, there is:
              x[f]: array-like of shape (T,) or (1, T)
                    Univariate target time series aligned with `timestamp`.
        - One univariate series per exogenous field in `ev_fields`:
            For each name `g` in `ev_fields`, there is:
              x[g]: array-like of shape (T,) or (1, T)
                    Univariate exogenous (dynamic) time series aligned with `timestamp`.

      Optionally, the dataset may already contain:
        - "start": the start time of the series
        - "freq": the frequency string (e.g. "5min", "1H")

    This function:
      - Ensures `start` exists (using the first element of "timestamp" if missing).
      - Infers `freq` from the "timestamp" sequence if missing.
      - Applies optional `target_transform_fns` to each target field and stacks all
        target variates into a single multivariate array under the "target" key.
      - Applies optional `ev_transform_fns` to each exogenous field and stacks all
        exogenous variates into a single multivariate array under the
        "feat_dynamic_real" key.
      - Uses `ensure_variate_first` so that the variate dimension is always before
        the time dimension.

    Final dataset format (GluonTS-style):
      Each item will contain:
        - "start": the start time of the series (scalar timestamp)
        - "freq": string representing the frequency of the series
        - "target": np.ndarray of shape (D, T)
                    D target variates, time along the last dimension.
        - "feat_dynamic_real": np.ndarray of shape (K, T)
                               K dynamic real-valued exogenous features, time along
                               the last dimension.
    """

    # Build non-optional transform fn lists so indexing is always valid in lambdas.
    def _identity(arr: np.ndarray) -> np.ndarray:
        return arr

    _assert_fev_compatible(hf_dataset, target_fields, ev_fields)

    if target_transform_fns is None:
        t_fns: list[Callable[[np.ndarray], np.ndarray]] = [_identity for _ in target_fields]
    else:
        t_fns = target_transform_fns

    if ev_transform_fns is None:
        ev_fns: list[Callable[[np.ndarray], np.ndarray]] = [_identity for _ in ev_fields]
    else:
        ev_fns = ev_transform_fns

    # add a start field to the dataset
    hf_dataset = hf_dataset.map(lambda item: {"start": item["timestamp"][0]})
    # add a freq field to the dataset
    hf_dataset = hf_dataset.map(lambda item: {"freq": pd.infer_freq(pd.to_datetime(item["timestamp"]))})
    # stack the target variates under the same key 'target'
    hf_dataset = hf_dataset.map(
        lambda item: {
            "target": np.concatenate(
                [ensure_variate_first(t_fns[i](item[target])) for i, target in enumerate(target_fields)], axis=-2
            )
        }
    )
    if len(target_fields) == 1:
        # make the target field univariate
        hf_dataset = hf_dataset.map(lambda item: {"target": item["target"][0]})
    # apply the transformation functions to the exogenous features and stack them under 'feat_dynamic_real'
    hf_dataset = hf_dataset.map(
        lambda item: {
            "feat_dynamic_real": np.concatenate(
                [ensure_variate_first(ev_fns[i](item[ev])) for i, ev in enumerate(ev_fields)], axis=-2
            )
        }
    )

    return hf_dataset


def preprocess_exogenous_features(
    past_exogenous_features: Float[torch.Tensor, "n_exog past_len"],
    future_exogenous_features: Float[torch.Tensor, "n_exog future_len"],
    context_length: int,
    patch_size: int,
) -> torch.Tensor:
    """
    Prepare exogenous covariates (EVs) for Toto's decoder-only architecture.

    Toto can only attend to past positions, but forecasting with "known future"
    covariates (holidays, promotions, etc.) requires access to future EV values.
    We solve this by shifting the EV timeline backward by one patch: future EV
    values are moved into the context window so the model can attend to them when
    predicting. After this function, EVs are stacked with targets in
    ``instance_to_causal``.
    The last patch of padding does not effect training since we don't finetune Toto
    to predict EVs nor does it effect inference since we don't use that last patch during
    autoregressive inference.


    Parameters
    ----------
    past_exogenous_features : Tensor, shape (past_len,) or (past_len, n_exog)
        Historical exogenous covariate values.
    future_exogenous_features : Tensor, shape (future_len,) or (future_len, n_exog)
        Known future exogenous covariate values (e.g., planned promotions).
    context_length : int
        The context window size used by the model.
    patch_size : int
        The prediction horizon (and shift amount).

    Returns
    -------
    Tensor, shape (n_exog, context_length + patch_size)
        Shifted EVs with zero-padding, ready to be stacked with targets.
    """
    past_exogenous_features_vt = ensure_variate_first(past_exogenous_features)
    future_exogenous_features_vt = ensure_variate_first(future_exogenous_features)

    # Step 1: Concatenate past and known future EVs along time
    # Result shape: (n_exog, past_len + future_len)
    exogenous_features = torch.cat(
        [past_exogenous_features_vt, future_exogenous_features_vt],
        dim=-1,
    )

    # Step 2: Trim to last `context_length` timesteps - this is the 'shift' operation.
    # By taking [-context_length:], we include future_ev values in what will become
    # the model's "context" window, effectively shifting EVs backward in time.
    exogenous_features = exogenous_features[..., -context_length:]

    # Step 3: Append zero padding for the prediction horizon.
    # This aligns with target_slice where predictions happen. The zeros don't matter
    # because EV variates have padding_mask=False and are excluded from loss.
    exogenous_features = torch.cat(
        [
            exogenous_features,
            torch.zeros((exogenous_features.shape[0], patch_size)).to(
                exogenous_features.device, dtype=exogenous_features.dtype
            ),
        ],
        dim=-1,
    )

    return exogenous_features


def collate_causal(batch: list[CausalMaskedTimeseries]) -> CausalMaskedTimeseries:
    """
    Stack a list of CausalMaskedTimeseries into a batched CausalMaskedTimeseries.
    """
    series = torch.stack([b.series for b in batch], dim=0)
    padding_mask = torch.stack([b.padding_mask for b in batch], dim=0)
    id_mask = torch.stack([b.id_mask for b in batch], dim=0)
    timestamp_seconds = torch.stack([b.timestamp_seconds for b in batch], dim=0)
    time_interval_seconds = torch.stack([b.time_interval_seconds for b in batch], dim=0)
    input_slice = batch[0].input_slice
    target_slice = batch[0].target_slice
    num_exogenous_variables = batch[0].num_exogenous_variables
    return CausalMaskedTimeseries(
        series=series,
        padding_mask=padding_mask,
        id_mask=id_mask,
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
        input_slice=input_slice,
        target_slice=target_slice,
        num_exogenous_variables=num_exogenous_variables,
    )
