# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from math import ceil
from typing import List, NamedTuple, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from gluonts.evaluation import Evaluator
from gluonts.model.forecast import SampleForecast

from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import Forecast, TotoForecaster
from toto.model.lightning_module import TotoForFinetuning

TEST_SPLIT_FRACTION = 0.1
METRICS = ["abs_error", "mean_wQuantileLoss", "MASE"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets that are not contaminated by LOTSA and are thus safe to include
DATASETS = {
    "uci_air_quality_1D": True,
    "uci_air_quality_1H": True,
    "epf_pjm": True,
    "epf_np": True,
    "epf_fr": True,
    "epf_de": True,
    "epf_be": True,
    "entsoe_15T": True,
    "entsoe_30T": True,
    "entsoe_1H": True,
    "solar_with_weather_15T": True,
    "solar_with_weather_1H": True,
    "rohlik_orders_1D": True,
    "rohlik_sales_1D": True,
    "m5_1D": False,
    "m5_1W": False,
    "favorita_transactions_1D": False,
    "favorita_stores_1D": False,
    "favorita_stores_1W": False,
    "rossmann_1D": False,
    "proenfo_gfc12": False,
    "proenfo_gfc14": False,
    "proenfo_gfc17": False,
}


class TaskWindow(NamedTuple):
    inputs: MaskedTimeseries  # single-series inputs (no batch dim)
    future_known_features: torch.Tensor | None  # shape (1, V_ev, T_future) or None
    target_slice: np.ndarray  # shape (V_target, T_total)
    context_start: np.datetime64
    future_start: np.datetime64


def generate_test_window(
    data,
    patch_size,
    add_exogenous_variables: bool = True,
    offset: int = 1,
    context_length: int = 1024,
    max_pred_length: int = 512,
    idx: int = 0,
    device=DEVICE,
) -> TaskWindow | None:
    """
    Create a TaskWindow for evaluation from a specific offset and dataset row.
    """
    series = np.array(data["target"][idx])
    if len(series.shape) == 1:
        series = series.reshape(1, -1)
    timestamps = np.array(data["timestamp"][idx])

    total_length = series.shape[-1]

    if add_exogenous_variables:
        ev_features = np.array(data["feat_dynamic_real"][idx])  # (V_ev, T_total)
        if len(ev_features.shape) == 1:
            ev_features = ev_features.reshape(1, -1)
        V_ev = ev_features.shape[0]
        # shift exogenous features
        ev_features = ev_features[:, patch_size:]
        ev_features = np.concatenate([ev_features, np.zeros((V_ev, patch_size))], axis=-1)
        mv_series = np.concatenate([series, ev_features], axis=0)  # (V_target + V_ev, T_total)
    else:
        V_ev = 0
        mv_series = series

    V_target = series.shape[0]

    test_split = ceil(TEST_SPLIT_FRACTION * total_length - offset)

    # slicing context and future series
    target_slice = mv_series[:, -(test_split + context_length) :][:, : context_length + max_pred_length]
    timestamps_slice = timestamps[-(test_split + context_length) :][: context_length + max_pred_length]

    # if not enough points left, return None
    if target_slice.shape[1] < context_length + max_pred_length:
        return None

    context_series = target_slice[:, :context_length]
    context_start = timestamps_slice[0]

    future_series = target_slice[:, context_length:]
    future_start = timestamps_slice[context_length]

    # convert to tensors
    context_series = torch.tensor(context_series, dtype=torch.float32)  # (V_total, context_length)
    future_series = torch.tensor(future_series, dtype=torch.float32)  # (V_total, T_future)

    # create padding mask
    padding_mask = torch.ones_like(context_series, dtype=torch.bool)
    if add_exogenous_variables and V_ev > 0:
        padding_mask[V_target : V_target + V_ev, -patch_size:] = False

    # constructing id mask
    id_mask = torch.ones_like(context_series, dtype=torch.long)

    # constructing timestamps (relative index, not wall-clock)
    num_variates = V_target + V_ev
    timestamp_seconds = torch.arange(context_length).expand((num_variates, context_length))
    time_interval_seconds = torch.full((num_variates,), 1)

    # single-series MaskedTimeseries (no batch dim yet)
    inputs = MaskedTimeseries(
        series=context_series.to(device),
        padding_mask=padding_mask.to(device),
        id_mask=id_mask.to(device),
        timestamp_seconds=timestamp_seconds.to(device),
        time_interval_seconds=time_interval_seconds.to(device),
        num_exogenous_variables=V_ev,
    )

    # shape (1, V_ev, T_future) or None
    future_known_features = (
        future_series[-V_ev:, :].unsqueeze(0).to(device) if (add_exogenous_variables and V_ev > 0) else None
    )

    return TaskWindow(
        inputs=inputs,
        future_known_features=future_known_features,
        target_slice=target_slice[:V_target, :],
        context_start=context_start,
        future_start=future_start,
    )


class BatchedTaskWindow(NamedTuple):
    """
    Holds a batched MaskedTimeseries plus metadata for each series.
    """

    inputs: MaskedTimeseries  # fields are batched
    future_known_features: torch.Tensor | None  # (B, V_ev, T_future) or None
    windows: List[TaskWindow]  # per-series metadata (targets, timestamps)


def batch_task_windows(
    windows: List[TaskWindow],
    device=DEVICE,
) -> BatchedTaskWindow:
    """
    Collate a list of single-series TaskWindows into a single batched MaskedTimeseries.

    Assumptions:
      - all windows share same num_variates, context_length, and future length
      - MaskedTimeseries has attributes: series, padding_mask, id_mask,
        timestamp_seconds, time_interval_seconds
    """
    # check if all the windows have future known features
    has_future = all(w.future_known_features is not None for w in windows)

    # Stack tensors directly along batch dimension -> (B, V_total, T_context) etc.
    batched_inputs = MaskedTimeseries(
        series=torch.stack([w.inputs.series for w in windows], dim=0).to(device),
        padding_mask=torch.stack([w.inputs.padding_mask for w in windows], dim=0).to(device),
        id_mask=torch.stack([w.inputs.id_mask for w in windows], dim=0).to(device),
        timestamp_seconds=torch.stack([w.inputs.timestamp_seconds for w in windows], dim=0).to(device),
        time_interval_seconds=torch.stack([w.inputs.time_interval_seconds for w in windows], dim=0).to(device),
        num_exogenous_variables=windows[0].inputs.num_exogenous_variables,
    )

    future_known_features = None
    if has_future:
        # (B, V_ev, T_future) - squeeze removes leading dim from each window
        future_known_features = torch.stack(
            [w.future_known_features.squeeze(0) for w in windows], dim=0  # type: ignore[union-attr]
        ).to(device)

    return BatchedTaskWindow(
        inputs=batched_inputs,
        future_known_features=future_known_features,
        windows=windows,
    )


def predict_batched(
    batched_window: BatchedTaskWindow,
    forecaster: TotoForecaster,
    add_exogenous_variables: bool,
    prediction_length: int,
    device=DEVICE,
    num_samples: int = 256,
    samples_per_batch: int = 256,
) -> Forecast:
    """
    Run Toto inference on a BatchedTaskWindow and return a batched Forecast.

    Assumes TotoForecaster.forecast supports a batched MaskedTimeseries where
    inputs.series has shape (B, V, T).
    """
    inputs = batched_window.inputs
    forecaster.model.eval()
    while True:
        try:
            forecast = forecaster.forecast(
                inputs.to(device),
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=samples_per_batch,
                use_kv_cache=True,
                future_exogenous_variables=(batched_window.future_known_features if add_exogenous_variables else None),
            )
            break
        except Exception as e:
            print(f"Error in predict_batched: {e} attempting to reduce samples_per_batch to {samples_per_batch // 2}")
            samples_per_batch = samples_per_batch // 2
            if samples_per_batch < 1:
                raise RuntimeError(f"Error in predict_batched: {e} and samples_per_batch is {samples_per_batch}")

    # Narrow Optional[Tensor] → Tensor for mypy
    assert forecast.mean is not None, "Forecast.mean is None despite num_samples > 0"
    assert forecast.samples is not None, "Forecast.samples is None despite num_samples > 0"

    # When exogenous variables are used, they're stacked as extra variates after targets.
    # The model predicts all variates, but we only want target predictions—strip out EVs.
    if add_exogenous_variables and batched_window.future_known_features is not None:
        V_ev = batched_window.future_known_features.shape[1]
        return Forecast(
            mean=forecast.mean[:, :-V_ev],
            samples=forecast.samples[:, :-V_ev, :, :],
        )
    return forecast


def compute_offsets(
    series_length: int,
    prediction_length: int,
    stride: int,
    max_windows_per_series: int,
    test_split_fraction: float = TEST_SPLIT_FRACTION,
) -> np.ndarray:
    """
    Compute rolling-window offsets within the test split.

    Ensures at least one prediction window fits in the test region.
    """
    assert (
        ceil(test_split_fraction * series_length) >= prediction_length
    ), "Not enough points for at least one-step future."

    raw_offsets = np.arange(
        0,
        test_split_fraction * series_length - prediction_length,
        stride,
    )

    if len(raw_offsets) > max_windows_per_series:
        raw_offsets = raw_offsets[:max_windows_per_series]

    return raw_offsets


def setup_toto_forecaster(
    model: TotoForFinetuning,
    add_exogenous_variables: bool,
    device=DEVICE,
) -> Tuple[TotoForecaster, int]:
    """
    Move model to device, set eval mode, construct TotoForecaster and return:
      - forecaster
      - patch_size (stride of patch_embed)
    """
    model.model.to(device)
    model.model.eval()

    forecaster = TotoForecaster(model.model)
    patch_size = model.model.patch_embed.stride

    return forecaster, patch_size


def build_batched_window_for_batch(
    batch_indices: List[int],
    data: HFDataset,
    patch_size: int,
    add_exogenous_variables: bool,
    context_length: int,
    prediction_length: int,
    offset: int,
    device=DEVICE,
) -> BatchedTaskWindow | None:
    """
    For a list of series indices, build a BatchedTaskWindow.

    Returns None if no valid windows are found (e.g., too short series).
    """
    windows: List[TaskWindow] = []

    for idx in batch_indices:
        w = generate_test_window(
            data=data,
            patch_size=patch_size,
            add_exogenous_variables=add_exogenous_variables,
            context_length=context_length,
            max_pred_length=prediction_length,
            offset=offset,
            idx=idx,
            device=device,
        )
        if w is not None:
            windows.append(w)

    if len(windows) == 0:
        return None

    return batch_task_windows(windows, device=device)


def collect_gluonts_iterators_for_batch(
    batched_window: BatchedTaskWindow,
    forecast: Forecast,
    batch_indices: List[int],
    frequency: str,
    ts_iterator: List[pd.Series],
    forecasts_iterator: List[SampleForecast],
) -> None:
    """
    Given a BatchedTaskWindow and corresponding batched Forecast, extend
    ts_iterator & forecasts_iterator with one time series and one forecast
    per variate, per series.
    """
    assert forecast.samples is not None, "Forecast.samples is None"

    samples = forecast.samples.cpu().numpy()  # (B, V_target, T_future, S)
    B, V_target, T_future, S = samples.shape

    assert B == len(batched_window.windows), "Batch size mismatch between forecasts and windows."

    for b, window in enumerate(batched_window.windows):
        target = window.target_slice  # (V_target, T_total)
        V, T_total = target.shape
        assert V == V_target, "Target variate dimension mismatch."

        context_start = pd.Period(window.context_start, freq=frequency)
        future_start = pd.Period(window.future_start, freq=frequency)

        # full index for context + future
        index = pd.period_range(
            start=context_start,
            periods=T_total,
            freq=frequency,
        )

        for v in range(V_target):
            ts_series = pd.Series(target[v], index=index)
            ts_iterator.append(ts_series)

            # samples[b, v] is (T_future, S) -> need (S, T_future)
            samples_v = samples[b, v].T  # (S, T_future)
            fc = SampleForecast(
                samples=samples_v,
                start_date=future_start,
                item_id=f"series_{batch_indices[b]}_var_{v}",
            )
            forecasts_iterator.append(fc)


def _evaluate_single_offset(
    offset: int,
    data: HFDataset,
    forecaster: TotoForecaster,
    patch_size: int,
    context_length: int,
    prediction_length: int,
    seasonality: int,
    add_exogenous_variables: bool,
    base_batch_size: int,
    max_series: int,
    frequency: str,
    num_samples: int = 256,
    samples_per_batch: int = 256,
    device=DEVICE,
) -> dict[str, float]:
    """
    Evaluate a single offset across up to `max_series` series.

    Handles mini-batching and batch-size backoff on errors.
    Returns a dict of aggregated metrics for this offset.
    """
    ts_iterator: List[pd.Series] = []
    forecasts_iterator: List[SampleForecast] = []

    current_batch_size = base_batch_size
    num_series = min(len(data), max_series)

    while True:
        try:
            ts_iterator.clear()
            forecasts_iterator.clear()

            # iterate over dataset rows in mini-batches
            for start_idx in range(0, num_series, current_batch_size):
                end_idx = min(start_idx + current_batch_size, num_series)
                batch_indices = list(range(start_idx, end_idx))

                # 1) build batched window
                batched_window = build_batched_window_for_batch(
                    batch_indices=batch_indices,
                    data=data,
                    patch_size=patch_size,
                    add_exogenous_variables=add_exogenous_variables,
                    context_length=context_length,
                    prediction_length=prediction_length,
                    offset=offset,
                    device=device,
                )
                if batched_window is None:
                    continue

                # 2) run batched forecast
                forecast = predict_batched(
                    batched_window=batched_window,
                    forecaster=forecaster,
                    add_exogenous_variables=add_exogenous_variables,
                    prediction_length=prediction_length,
                    device=device,
                    num_samples=num_samples,
                    samples_per_batch=samples_per_batch,
                )

                if forecast is None:
                    continue

                # 3) extend GluonTS iterators
                collect_gluonts_iterators_for_batch(
                    batched_window=batched_window,
                    forecast=forecast,
                    batch_indices=batch_indices,
                    frequency=frequency,
                    ts_iterator=ts_iterator,
                    forecasts_iterator=forecasts_iterator,
                )

            break  # successful run
        except Exception as e:
            print(
                f"Error in _evaluate_single_offset (offset={offset}): {e}, "
                f"reducing batch size to {current_batch_size // 2}"
            )
            current_batch_size = current_batch_size // 2
            if current_batch_size < 1:
                raise RuntimeError(f"Error in _evaluate_single_offset: {e} and batch_size is {current_batch_size}")

    evaluator = Evaluator(seasonality=seasonality)
    metrics, _ = evaluator(ts_iterator, forecasts_iterator)
    return metrics


def evaluate_model(
    model: TotoForFinetuning,
    data: HFDataset,
    context_length: int,
    prediction_length: int,
    seasonality: int,
    stride: int,
    add_exogenous_variables: bool,
    batch_size: int = 32,
    max_windows_per_series: int = 10,
    max_series: int = 200,
    num_samples: int = 256,
    samples_per_batch: int = 256,
) -> pd.DataFrame:
    """
    Evaluate the model across all rows in `data` and rolling windows,
    using batched forecasting.

    Returns a DataFrame with one row per offset (rolling window),
    metrics aggregated over all series and variates for that offset.
    """
    series_length = data["target"][0].shape[-1]

    # infer frequency from first series timestamps
    timestamps = np.array(data["timestamp"][0])
    frequency = pd.infer_freq(pd.DatetimeIndex(timestamps))

    # determine offsets within test split
    offsets = compute_offsets(
        series_length=series_length,
        prediction_length=prediction_length,
        stride=stride,
        max_windows_per_series=max_windows_per_series,
        test_split_fraction=TEST_SPLIT_FRACTION,
    )

    # setup model + forecaster
    forecaster, patch_size = setup_toto_forecaster(
        model=model,
        add_exogenous_variables=add_exogenous_variables,
        device=DEVICE,
    )

    # aggregate metrics per offset
    results: dict[str, list[float]] = {metric: [] for metric in METRICS}

    for offset in offsets:
        metrics = _evaluate_single_offset(
            offset=offset,
            data=data,
            forecaster=forecaster,
            patch_size=patch_size,
            context_length=context_length,
            prediction_length=prediction_length,
            seasonality=seasonality,
            add_exogenous_variables=add_exogenous_variables,
            base_batch_size=batch_size,
            max_series=max_series,
            frequency=frequency,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch,
            device=DEVICE,
        )

        for metric in results.keys():
            results[metric].append(metrics[metric])

    return pd.DataFrame(results)


# -----------------------------
# aggregation helpers
# -----------------------------
def gmean_1d(x: np.ndarray) -> float:
    """
    Geometric mean for finite, non-negative values.
    - returns NaN if empty
    - returns 0 if any zero
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.nan
    if np.any(x < 0):
        raise ValueError("Geometric mean requires non-negative values.")
    if np.any(x == 0):
        return 0.0
    return float(np.exp(np.mean(np.log(x))))


def gmean_finite_metric(series: pd.Series) -> tuple[float, int]:
    """
    Compute gmean on a single metric, dropping only non-finite entries for that metric.
    Returns (gmean_value, n_used).
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s.to_numpy())]
    val = gmean_1d(s.to_numpy(dtype=np.float64))
    return val, int(s.shape[0])
