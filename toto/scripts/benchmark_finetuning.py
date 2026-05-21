# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import argparse
import gc
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, MutableMapping, cast

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset

from toto.evaluation.fev.evaluate import DATASETS, evaluate_model
from toto.scripts import finetune_toto as finetune

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

Config = Dict[str, object]


def load_config(config_path: str) -> Config:
    """
    Load the Toto finetuning/evaluation configuration from a YAML file.

    The YAML is expected to contain the keys used by `toto.scripts.finetune_toto`,
    e.g.:
      - seed
      - pretrained_model
      - model: { ... }
      - data: { ... }
      - trainer: { ... }
      - checkpoint: { ... }
      - logging: { ... }

    Args:
        config_path: Path to a YAML config file.

    Returns:
        Parsed config as a nested dict.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config (expected mapping at root): {config_path}")

    return cast(Config, cfg)


def get_config(
    base_config: Config,
    model_name: str,
    dataset_name: str,
    add_exogenous_features: bool,
    horizon: int | None,
) -> Config:
    """
    Build the final runtime config for a specific (dataset, model variant) run.

    We start from a base config loaded from YAML, then fill in run-specific fields:
      - data.add_exogenous_features
      - data.prediction_horizon
      - logging.name
      - checkpoint.dirpath

    Args:
        base_config: Base config loaded from YAML.
        model_name: Name of the model variant used for logging/ckpt paths.
        dataset_name: Dataset identifier (used for logging/ckpt paths).
        add_exogenous_features: Whether to enable known exogenous variables.
        horizon: Forecast horizon (overrides data.prediction_horizon).

    Returns:
        A config dict ready to be passed to finetune/evaluation helpers.
    """
    config: Config = deepcopy(base_config)

    data_cfg = cast(MutableMapping[str, Any], config.get("data", {}))
    logging_cfg = cast(MutableMapping[str, Any], config.get("logging", {}))
    ckpt_cfg = cast(MutableMapping[str, Any], config.get("checkpoint", {}))

    # Override task-specific settings
    data_cfg["add_exogenous_features"] = add_exogenous_features
    data_cfg["prediction_horizon"] = horizon

    # Set run-identifying metadata (used by Lightning loggers and checkpointing)
    logging_cfg["name"] = Path(dataset_name) / model_name
    ckpt_cfg["dirpath"] = Path("checkpoints") / dataset_name / model_name

    # Write the mutated sub-configs back (in case the keys were missing)
    config["data"] = data_cfg
    config["logging"] = logging_cfg
    config["checkpoint"] = ckpt_cfg

    return config


def drop_nan_fn(x: np.ndarray, encode_categorical: bool = True) -> np.ndarray:
    """Replace NaNs (or None for object arrays) with safe defaults and return a cleaned copy."""
    if x.dtype == np.dtype("O"):
        # For object arrays, None can appear for missing values.
        x = np.array([i if i is not None else "none" for i in x])
        if encode_categorical:
            x = encode_categorical_fn(x)
        return x
    else:
        x = x.astype(float, copy=True)
        x[np.isnan(x)] = 0
        return x


def encode_categorical_fn(x: np.ndarray) -> np.ndarray:
    """Encode categorical features as integer ids (stable within the provided array)."""
    return np.unique(x, return_inverse=True)[1]


def prepare_dataset(dataset_name: str, target_fields: list[str], ev_fields: list[str]) -> dict[str, Any]:
    """
    Load a FEV dataset from HuggingFace and build a custom structure expected by Toto datamodule helpers.

    Args:
        dataset_name: Name of the dataset config in autogluon/fev_datasets.
        target_fields: Which columns represent the target series.
        ev_fields: Which columns are known dynamic / exogenous variables.

    Returns:
        A dictionary with dataset + transformation hooks.
    """
    dataset = load_dataset("autogluon/fev_datasets", dataset_name, split="train")
    dataset.set_format("numpy")

    # One transform per field, as expected by the datamodule.
    target_transform_fns = [drop_nan_fn] * len(target_fields)
    ev_transform_fns = [drop_nan_fn] * len(ev_fields)

    return {
        "dataset_name": dataset_name,
        "dataset": dataset,
        "target_fields": target_fields,
        "target_transform_fns": target_transform_fns,
        "ev_fields": ev_fields,
        "ev_transform_fns": ev_transform_fns,
    }


def run_pipeline(
    base_config: Config,
    custom_dataset: dict[str, Any],
    horizon: int,
    seasonality: int,
    enable_finetuning: bool,
    enable_exogenous_features: bool,
    results_dir: str,
) -> None:
    """
    Run one experiment variant (zero-shot or finetune; with or without exogenous features)
    on a single dataset.

    Important: This is an internal evaluation pipeline, NOT the official FEV benchmark.
    We use publicly available datasets from FEV (via HuggingFace) but apply our own
    evaluation protocol (e.g., custom train/val/test splits, stride settings, metrics).
    Results from this script are not directly comparable to official FEV leaderboard scores.

    Pipeline steps:
      1) Builds a run-specific config from the YAML base config
      2) Initializes Toto Lightning module + datamodule
      3) Optionally finetunes and loads the best checkpoint
      4) Runs evaluation with our internal evaluator
      5) Saves per-series results to CSV
      6) Frees GPU/CPU memory

    Note: We use `stride=horizon` for evaluation (non-overlapping windows by default).
    """
    # Build a human-readable model variant name.
    base_model_name = "toto"
    if enable_finetuning:
        base_model_name = f"{base_model_name}_finetuning"
    if enable_exogenous_features:
        base_model_name = f"{base_model_name}_exogenous"

    # Build config for the specific model + dataset run.
    config = get_config(
        base_config=base_config,
        model_name=base_model_name,
        dataset_name=custom_dataset["dataset_name"],
        add_exogenous_features=enable_exogenous_features,
        horizon=horizon,
    )

    print(f"Running {base_model_name} on {custom_dataset['dataset_name']}")

    # Initialize Lightning module and datamodule
    lightning_module, patch_size = finetune.init_lightning(config)
    datamodule = finetune.get_datamodule(config, patch_size, custom_dataset, setup=True)

    assert datamodule._view is not None, "Datamodule view is not setup yet"

    # Train or run zero-shot
    if enable_finetuning:
        _, best_ckpt_path, best_val_loss = finetune.train(lightning_module, datamodule, config)

        if best_ckpt_path is None:
            raise RuntimeError("No checkpoint was saved during training. Check checkpoint config.")

        pretrained_model = cast(str, config["pretrained_model"])

        # Load best finetuned model checkpoint
        trained_model = finetune.load_finetuned_toto(
            pretrained_model,
            best_ckpt_path,
            lightning_module.device,
        )
    else:
        trained_model = lightning_module
        best_val_loss = None

    print("Best validation loss: ", best_val_loss)

    # Evaluate model
    results = evaluate_model(
        trained_model,
        datamodule._view.hf_dataset,
        datamodule._view._context_length,
        horizon,
        seasonality,
        stride=horizon,
        add_exogenous_variables=enable_exogenous_features,
    )

    # Save results
    out_dir = Path(results_dir) / custom_dataset["dataset_name"] / base_model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv")

    # Aggregate results
    aggregated_results = aggregate_results(results)
    aggregated_results.to_csv(out_dir / "aggregated_results.csv")


def aggregate_results(results, eps: float = 1e-12):
    """
    Aggregate results across all metrics using geometric mean over series (rows).

    Notes:
      - Geometric mean is only defined for strictly positive values.
      - We clamp values to >= eps to avoid log(0) / negative issues.
      - If your metrics can be negative (rare), consider a different aggregation.
    """
    # results is typically a pandas DataFrame; this works for numpy arrays too.
    x = np.asarray(results, dtype=float)

    # Clamp to avoid zeros/negatives breaking log. (You can also choose to raise instead.)
    x = np.clip(x, eps, None)

    gmean = np.exp(np.mean(np.log(x), axis=0))

    # Return same type/labels as the original aggregator (Series with column names)
    return pd.Series(gmean, index=results.columns)


def parse_field(task: dict[str, Any], field_name: str, default_value=None):
    """
    Normalize a YAML task field into a list.

    Some task fields can be provided as either:
      - a scalar (single field name), or
      - a list of field names.

    This helper makes downstream code simpler by always returning a list.
    """
    if field_name in task:
        if isinstance(task[field_name], list):
            return task[field_name]
        else:
            return [task[field_name]]
    else:
        return default_value


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for config/task file paths and output directory."""
    parser = argparse.ArgumentParser(description="Run Toto on FEV datasets with YAML-driven config.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the Toto config YAML (replaces the hard-coded GENERAL_CONFIG).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="toto/evaluation/fev/tasks.yaml",
        help="Path to the FEV tasks YAML.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where CSV outputs will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load base Toto config (previously hard-coded in the script).
    base_config = load_config(args.config)

    # Load evaluation tasks.
    with open(args.tasks, "r") as f:
        tasks = yaml.safe_load(f)

    for task in tasks["tasks"]:
        dataset_name = task["dataset_config"]

        if not DATASETS[dataset_name]:
            continue

        target_fields = parse_field(task, "target", default_value=["target"])
        ev_fields = parse_field(task, "known_dynamic_columns", default_value=[])
        horizon = task["horizon"]
        seasonality = task["seasonality"]

        # Prepare dataset (HF load + set transforms)
        custom_dataset = prepare_dataset(dataset_name, target_fields, ev_fields)

        # We evaluate only on datasets with known exogenous features
        if len(ev_fields) > 0:
            # Zero-shot baseline
            run_pipeline(
                base_config=base_config,
                custom_dataset=custom_dataset,
                horizon=horizon,
                seasonality=seasonality,
                enable_finetuning=False,
                enable_exogenous_features=False,
                results_dir=args.results_dir,
            )
            # Finetuned Toto (no exogenous variables)
            run_pipeline(
                base_config=base_config,
                custom_dataset=custom_dataset,
                horizon=horizon,
                seasonality=seasonality,
                enable_finetuning=True,
                enable_exogenous_features=False,
                results_dir=args.results_dir,
            )
            # Finetuned Toto with exogenous features (variate labels auto-enabled)
            run_pipeline(
                base_config=base_config,
                custom_dataset=custom_dataset,
                horizon=horizon,
                seasonality=seasonality,
                enable_finetuning=True,
                enable_exogenous_features=True,
                results_dir=args.results_dir,
            )
