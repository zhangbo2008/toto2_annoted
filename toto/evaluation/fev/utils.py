# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

import os

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset

from toto.evaluation.fev.evaluate import gmean_1d, gmean_finite_metric
from toto.scripts.benchmark_finetuning import parse_field

# -----------------------------
# Functions for aggregation and ranking
# -----------------------------


def aggregate_per_dataset_model(results_path, datasets, models_to_include, metrics):
    """
    Aggregates results from results_path for each dataset and model, calculating
    the geometric mean for each metric over all valid data windows.

    Args:
        results_path (str): Path to the root results directory.
        datasets (dict): Dictionary of dataset keys to include.
        models_to_include (list): List of model names to aggregate.
        metrics (list): List of metric names to aggregate.

    Returns:
        pd.DataFrame: Aggregated dataframe (one row per dataset & model).
    """
    rows = []
    for dataset in sorted(os.listdir(results_path)):
        dataset_path = os.path.join(results_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        # Exclude by default for unknown datasets
        if not datasets.get(dataset, False):
            continue

        for model in sorted(os.listdir(dataset_path)):
            model_path = os.path.join(dataset_path, model)
            if not os.path.isdir(model_path):
                continue
            if model not in models_to_include:
                continue

            csv_path = os.path.join(model_path, "results.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)

            agg = {"dataset": dataset, "model": model, "n_windows_total": len(df)}
            for m in metrics:
                gm, n_used = gmean_finite_metric(df[m]) if m in df.columns else (np.nan, 0)
                agg[m] = gm
                agg[f"n_windows_used_{m}"] = n_used

            rows.append(agg)

    return pd.DataFrame(rows)


def compute_overall_model_agg(dataset_model_agg: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Compute overall aggregated performance per model.
    Returns a DataFrame with one row per model.
    """
    overall_rows = []
    for model, g in dataset_model_agg.groupby("model"):
        r = {"model": model}
        for m in metrics:
            vals = g[m]
            vals = vals[pd.notna(vals) & np.isfinite(vals.to_numpy())]
            r[m] = gmean_1d(vals.to_numpy(dtype=np.float64))
            r[f"n_datasets_used_{m}"] = int(vals.shape[0])
        overall_rows.append(r)
    overall_model_agg = pd.DataFrame(overall_rows).sort_values("model").reset_index(drop=True)
    return overall_model_agg


def compute_rank_summary(dataset_model_agg: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Compute model ranks per metric per dataset, then gmean of ranks across datasets.
    Returns a summary DataFrame sorted by overall gmean rank (lower is better).
    """
    rank_frames = []
    for dataset, g in dataset_model_agg.groupby("dataset"):
        base = g[["dataset", "model"]].copy()
        for m in metrics:
            gm_vals = g[["model", m]].copy()
            gm_vals = gm_vals[pd.notna(gm_vals[m]) & np.isfinite(gm_vals[m].to_numpy())]
            # If no one has a valid value for this metric on this dataset, skip ranking it.
            if gm_vals.empty:
                base[f"rank_{m}"] = np.nan
                continue
            # Rank only among valid models for that metric on that dataset
            gm_vals[f"rank_{m}"] = gm_vals[m].rank(ascending=True, method="average")
            base = base.merge(gm_vals[["model", f"rank_{m}"]], on="model", how="left")
        rank_frames.append(base)
    ranks = pd.concat(rank_frames, ignore_index=True)

    rank_summary_rows = []
    for model, g in ranks.groupby("model"):
        rr = {"model": model}
        for m in metrics:
            s = g[f"rank_{m}"].dropna()
            rr[f"gmean_rank_{m}"] = gmean_1d(s.to_numpy(dtype=np.float64))
            rr[f"n_datasets_ranked_{m}"] = int(s.shape[0])
        # optional: combine metric-wise rank summaries into one overall number
        metric_rank_vals = np.array([rr[f"gmean_rank_{m}"] for m in metrics], dtype=np.float64)
        metric_rank_vals = metric_rank_vals[np.isfinite(metric_rank_vals)]
        rr["gmean_rank_overall"] = gmean_1d(metric_rank_vals)
        rank_summary_rows.append(rr)
    rank_summary = pd.DataFrame(rank_summary_rows).sort_values("gmean_rank_overall").reset_index(drop=True)
    return rank_summary


# -----------------------------
# Functions for metadata collection
# -----------------------------


def get_median_series_length(dataset, target_field):
    series_lengths = []
    for i in range(len(dataset)):
        series_lengths.append(dataset[i][target_field].shape[-1])
    return np.median(series_lengths)


def get_datasets_metadata(tasks_yaml_path, datasets):
    datasets_metadata = {}

    with open(tasks_yaml_path, "r") as f:
        tasks = yaml.safe_load(f)

    for task in tasks["tasks"]:
        dataset_name = task["dataset_config"]

        if not datasets[dataset_name]:
            continue

        target_fields = parse_field(task, "target", default_value=["target"])
        ev_fields = parse_field(task, "known_dynamic_columns", default_value=[])
        horizon = task["horizon"]

        dataset = load_dataset("autogluon/fev_datasets", dataset_name, split="train")
        dataset.set_format("numpy")

        datasets_metadata[dataset_name] = {
            "num_rows": len(dataset),
            "series_length": get_median_series_length(dataset, target_fields[0]),
            "num_evs": len(ev_fields),
            "horizon": horizon,
        }
    return datasets_metadata


# -----------------------------
# Functions for latex table formatting
# -----------------------------


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def format_value(v: float, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if np.isinf(v):
        return r"$\infty$"
    return f"{v:.{decimals}f}"


def best_second_mask(values: list[float]):
    """
    Returns indices (best, second) by min. Ignores NaN; inf loses to finite.
    """
    arr = np.array(values, dtype=float)
    arr = np.where(np.isnan(arr), np.inf, arr)

    finite_idx = np.where(np.isfinite(arr))[0]
    if finite_idx.size == 0:
        return None, None

    sorted_idx = finite_idx[np.argsort(arr[finite_idx], kind="mergesort")]
    best = int(sorted_idx[0])
    second = int(sorted_idx[1]) if sorted_idx.size >= 2 else None
    return best, second


def style_cell(text: str, is_best: bool, is_second: bool) -> str:
    if is_best:
        return r"\textbf{" + text + "}"
    if is_second:
        return r"\underline{" + text + "}"
    return text


def format_meta(v):
    # match your example: floats printed with .1f
    if v is None:
        return "--"
    if isinstance(v, (int, np.integer)):
        return f"{float(v):.1f}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.1f}"
    return latex_escape(str(v))


def make_latex_table_from_dict(
    dataset_model_agg: pd.DataFrame,
    datasets_metadata: dict,
    model_key: dict,
    metric_display: dict,
    meta_map: list[tuple[str, str]],
    out_tex_path: str = "results_table.tex",
    section_title: str = "Aggregated results on datasets",
    decimals: int = 3,
    dataset_order: list[str] | None = None,
):
    """
    dataset_model_agg: long df with columns ['dataset','model', ...metrics...]
    datasets_metadata: dict[str, dict] like you showed
    """

    # restrict to the models we want in the table
    use_model_keys = list(model_key.values())
    df = dataset_model_agg[dataset_model_agg["model"].isin(use_model_keys)].copy()

    # pivot to (dataset) x (metric, model_key)
    wide = df.pivot_table(index="dataset", columns="model", values=list(metric_display.keys()), aggfunc="first")

    # datasets to show: intersection of agg and metadata
    available = sorted(set(wide.index).intersection(datasets_metadata.keys()))
    if dataset_order is None:
        datasets = available
    else:
        datasets = [d for d in dataset_order if d in available]

    n_meta = len(meta_map)
    n_models = len(model_key)
    colspec = "l" + "c" * n_meta + "|" + "c" * n_models + "|" + "c" * n_models

    lines = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[a4paper,margin=1in]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{amsmath}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\begin{document}")
    lines.append("")
    lines.append(r"\section*{" + section_title + "}")
    lines.append(r"\noindent \textbf{Metrics:} " + ", ".join(list(metric_display.values())) + r"\\")
    lines.append(r"\textbf{Models:} " + ", ".join(m.replace("&", r"\&") for m in list(model_key.keys())))
    lines.append("")
    lines.append(r"\vspace{0.5em}")
    lines.append("")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")

    # header row 1
    lines.append(
        r"Group & \multicolumn{" + str(n_meta) + r"}{c}{Metadata}"
        r" & \multicolumn{" + str(n_models) + r"}{c}{" + metric_display[list(metric_display.keys())[0]] + r"}"
        r" & \multicolumn{" + str(n_models) + r"}{c}{" + metric_display[list(metric_display.keys())[1]] + r"} \\"
    )

    # header row 2
    meta_hdr = " & ".join(latex_escape(col_name) for _, col_name in meta_map)
    model_hdr = " & ".join(m.replace("&", r"\&") for m in list(model_key.keys()))
    lines.append(r"Subcol & " + meta_hdr + " & " + model_hdr + " & " + model_hdr + r" \\")
    lines.append(r"\midrule")

    # rows
    for dataset in datasets:
        md = datasets_metadata.get(dataset, {})
        meta_vals = [format_meta(md.get(k, None)) for k, _ in meta_map]

        metric_cells = {}
        for metric in list(metric_display.keys()):
            vals = []
            for disp_model in list(model_key.keys()):
                key = model_key[disp_model]
                try:
                    v = wide.loc[dataset, (metric, key)]
                except KeyError:
                    v = np.nan
                vals.append(float(v) if v is not None else np.nan)

            best_i, second_i = best_second_mask(vals)

            cells = []
            for i, v in enumerate(vals):
                txt = format_value(v, decimals=decimals)
                cells.append(style_cell(txt, is_best=(best_i == i), is_second=(second_i == i)))
            metric_cells[metric] = cells

        row = (
            latex_escape(dataset)
            + " & "
            + " & ".join(meta_vals)
            + " & "
            + " & ".join(metric_cells[list(metric_display.keys())[0]])
            + " & "
            + " & ".join(metric_cells[list(metric_display.keys())[1]])
            + r" \\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}%")
    lines.append("")
    lines.append(r"\end{document}")

    tex = "\n".join(lines)
    with open(out_tex_path, "w") as f:
        f.write(tex)

    return tex
