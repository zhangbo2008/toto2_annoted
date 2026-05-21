# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

from __future__ import annotations

from typing import Any, List

import datasets as hfds
import numpy as np  # noqa: F401
from gluonts.transform import InstanceSplitter
from gluonts.transform.sampler import ExpectedNumInstanceSampler, ValidationSplitSampler
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..datasets.gluonts_dataset import GluonTSDatasetView, GluonTSInstanceDataset
from ..util.helpers import collate_causal


class FinetuneDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for finetuning the Toto time series model.

    Overview
    --------
    This module orchestrates the complete data pipeline for finetuning Toto on
    custom time series datasets. It bridges three distinct ecosystems, handling
    format conversions, temporal splitting, and batching automatically:

    1. **Hugging Face Datasets** — The input format. Your raw time series data
       arrives as a HuggingFace `Dataset` where each row represents one time
       series with fields like ``timestamp``, ``target``, and optionally
       ``feat_dynamic_real`` for exogenous covariates.

    2. **GluonTS** — The splitting and sampling engine. GluonTS provides
       battle-tested utilities for time series cross-validation. This module
       uses GluonTS to split each series into train/validation regions and to
       sample fixed-length training instances from those regions.

    3. **PyTorch Lightning** — The training framework. The final output is
       ``CausalMaskedTimeseries`` tensors that Toto's backbone consumes during
       forward passes and loss computation.

    Data Pipeline in Detail
    -----------------------
    When PyTorch Lightning calls ``setup()``, the following transformations occur:

    **Step 1: HuggingFace → GluonTS Format Conversion**

    The input HuggingFace dataset is wrapped in a ``GluonTSDatasetView``. This
    view performs several transformations:

    - Infers the frequency (``freq``) from the timestamp column if not provided
    - Extracts the ``start`` timestamp for each series
    - Applies optional transform functions to target and exogenous fields
    - Stacks multiple target/exogenous fields into multivariate arrays
    - Filters out series that are too short for the requested context/prediction lengths
    - Converts each row into a GluonTS-compatible dict with keys like ``target``,
      ``start``, ``freq``, and ``feat_dynamic_real``

    **Step 2: Temporal Train/Validation Splitting**

    Each time series is split temporally (not by row, but by time):

    - The last ~10% of each series is reserved for testing
    - The training region excludes one additional ``patch_size`` window before test
    - The validation region ends exactly where the test region begins

    **Step 3: Instance Sampling with InstanceSplitter**

    GluonTS ``InstanceSplitter`` generates training instances by sampling windows
    from each series. For each window, it produces:

    - ``past_target``: The context window of shape ``(variates, context_length)``
    - ``future_target``: The prediction window of shape ``(variates, patch_size)``
    - Corresponding past/future exogenous features if present

    Training uses ``ExpectedNumInstanceSampler`` which randomly samples windows,
    while validation uses ``ValidationSplitSampler`` which takes the last window.

    **Step 4: Conversion to CausalMaskedTimeseries**

    Each GluonTS instance is converted into a ``CausalMaskedTimeseries`` tensor.
    This structure contains:

    - ``series``: The full window ``[past_target | future_target]`` concatenated
      along time, shape ``(variates, context_length + patch_size)``
    - ``padding_mask``: Boolean mask indicating valid (non-padding) positions
    - ``id_mask``: Integer mask for grouping variates (for attention masking)
    - ``input_slice``: Python slice ``[0:context_length]`` marking the input region
    - ``target_slice``: Python slice ``[patch_size:context_length+patch_size]``
      marking the target region (shifted by patch_size for autoregressive training)

    The ``target_slice`` being shifted means the model learns to predict future
    values: given input at positions 0..C, predict values at positions P..C+P.

    **Step 5: Batching with collate_causal**

    The DataLoader uses a custom collate function that stacks individual
    ``CausalMaskedTimeseries`` objects into batched tensors.

    Parameters
    ----------
    dataset : hfds.Dataset
        Input HuggingFace Dataset. Each row should contain at minimum a
        ``timestamp`` array and one or more target series fields. The expected
        format follows the FEV benchmark convention.
    max_context_length : int
        Upper bound on the context window size. The actual context length may be
        smaller if your shortest series cannot support this length after accounting
        for train/validation/test splits.
    prediction_horizon : int
        The number of future timesteps to predict during evaluation. This affects
        how much data is reserved for the test split.
    patch_size : int
        The patch size of the Toto model being fine-tuned. During training,
        the prediction horizon is set to the size of a single patch.
    train_batch_size : int, default=32
        Number of training instances per batch.
    val_batch_size : int, default=32
        Number of validation instances per batch.
    test_windows : int | None, default=None
        Number of rolling test windows per series. If ``None``, computed as
        approximately 10% of the minimum series length divided by patch_size,
        capped at 20 windows.
    num_workers : int, default=0
        Number of parallel workers for data loading.
    num_train_samples : int, default=1
        Expected number of training instances to sample from each time series
        per epoch. Higher values provide more data augmentation by sampling
        different windows from the same series.
    add_exogenous_features : bool, default=False
        If ``True``, include exogenous covariates in the model input. Exogenous
        features are stacked as additional variates in the ``series`` tensor but
        marked as padding in the ``padding_mask`` so they don't contribute to the
        prediction loss.
    target_fields : List[str], default=["target"]
        Column names in the HuggingFace dataset to treat as target series. Multiple
        fields are stacked along the variate dimension to form a multivariate target.
    target_transform_fns : List[Callable] | None, default=None
        Optional preprocessing functions applied to each target field before
        stacking. Useful for normalization or feature engineering.
    ev_fields : List[str], default=["feat_dynamic_real"]
        Column names for exogenous (covariate) features. Like targets, multiple
        fields are stacked along the variate dimension.
    ev_transform_fns : List[Callable] | None, default=None
        Optional preprocessing functions applied to each exogenous field.
    max_rows : int | None, default=None
        If provided, limit the dataset to this many time series (useful for
        debugging or quick experiments).
    """

    def __init__(
        self,
        *,
        dataset: hfds.Dataset,
        max_context_length: int,
        prediction_horizon: int,
        patch_size: int = 64,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_windows: int | None = None,
        num_workers: int = 0,
        num_train_samples: int = 1,
        add_exogenous_features: bool = False,
        target_fields: List[str] = ["target"],
        target_transform_fns: List[Any] | None = None,
        ev_fields: List[str] = ["feat_dynamic_real"],
        ev_transform_fns: List[Any] | None = None,
        max_rows: int | None = None,
    ):
        super().__init__()
        # Store configuration / hyperparameters
        self.dataset = dataset
        self.max_context_length = max_context_length
        self.prediction_horizon = int(prediction_horizon)
        self.patch_size = patch_size
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_windows = test_windows
        self.num_workers = num_workers
        self.num_train_samples = num_train_samples
        self.add_exogenous_features = add_exogenous_features
        self.target_fields = target_fields
        self.target_transform_fns = target_transform_fns
        self.ev_fields = ev_fields
        self.ev_transform_fns = ev_transform_fns
        self.max_rows = max_rows
        # Lazily constructed GluonTS view and derived datasets
        self._view: GluonTSDatasetView | None = None
        self._train_ds: GluonTSInstanceDataset | None = None
        self._val_ds: GluonTSInstanceDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        # `setup` is called once per trainer stage; guard against re-creation.
        if self._view is None:
            # Create a GluonTS view
            self._view = GluonTSDatasetView(
                dataset=self.dataset,
                max_context_length=self.max_context_length,
                prediction_horizon=self.prediction_horizon,
                patch_size=self.patch_size,
                test_windows=self.test_windows,
                target_fields=self.target_fields,
                target_transform_fns=self.target_transform_fns,
                ev_fields=self.ev_fields,
                ev_transform_fns=self.ev_transform_fns,
                max_rows=self.max_rows,
            )

            # --------------------------
            # InstanceSplitters for Train / Val
            # --------------------------
            # Training sampler: randomly sample windows rather than using a fixed rolling window.
            # This provides data augmentation by exposing the model to different temporal
            # positions each epoch, improving generalization when training data is limited.
            splitter_train = InstanceSplitter(
                target_field="target",
                is_pad_field="is_pad",
                start_field="start",
                forecast_start_field="forecast_start",
                instance_sampler=ExpectedNumInstanceSampler(
                    num_instances=self.num_train_samples,
                    min_future=self.patch_size,
                ),
                # here we pre-computed optimal context length
                past_length=self._view._context_length,
                future_length=self.patch_size,
                time_series_fields=["feat_dynamic_real"],
            )

            # Validation sampler: deterministically takes the last window from each series
            # for reproducible metrics (in contrast to random training sampling above).
            splitter_val = InstanceSplitter(
                target_field="target",
                is_pad_field="is_pad",
                start_field="start",
                forecast_start_field="forecast_start",
                instance_sampler=ValidationSplitSampler(
                    min_future=self.patch_size,
                ),
                # here we pre-computed optimal context length
                past_length=self._view._context_length,
                future_length=self.patch_size,
                time_series_fields=["feat_dynamic_real"],
            )

            # GluonTS training/validation datasets
            train_gts = self._view.training_dataset  # GluonTS TrainingDataset
            val_gts = self._view.validation_dataset  # GluonTS TrainingDataset

            # Convert each time series into multiple train/val instances
            # (dict-like DataEntry with past_target / future_target, plus covariates)
            train_instances = splitter_train(train_gts, is_train=True)
            val_instances = splitter_val(val_gts, is_train=False)

            # Wrap GluonTS instances into PyTorch Datasets producing CausalMaskedTimeseries
            self._train_ds = GluonTSInstanceDataset(
                train_instances,
                context_length=self._view._context_length,
                patch_size=self.patch_size,
                add_exogenous_features=self.add_exogenous_features,
            )
            self._val_ds = GluonTSInstanceDataset(
                val_instances,
                context_length=self._view._context_length,
                patch_size=self.patch_size,
                add_exogenous_features=self.add_exogenous_features,
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "Train dataset not created"
        return DataLoader(
            self._train_ds,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,  # shuffle training instances each epoch
            drop_last=False,
            collate_fn=collate_causal,  # builds CausalMaskedTimeseries batches
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None, "Validation dataset not created"
        return DataLoader(
            self._val_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # deterministic evaluation
            drop_last=False,
            collate_fn=collate_causal,
            pin_memory=True,
        )
