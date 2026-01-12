#!/usr/bin/env python
import json
import argparse
from typing import Tuple, Dict, Any, List

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def _read_cachefile(cache_file: str) -> Dict[str, Any]:
    # Read file and handle partial/trailing content
    with open(cache_file, "r") as fh:
        filestr = fh.read().strip()

    if filestr == "":
        raise ValueError("Cache file is empty")

    # Try to be permissive: if file ends with a stray comma or missing closing braces, try to fix
    if not filestr.endswith("}\n}") and not filestr.endswith("}\n}"):
        # remove trailing comma if present
        if filestr[-1] == ",":
            filestr = filestr[:-1]
        # attempt to close
        if not filestr.endswith("}\n}"):
            filestr = filestr + "}\n}"

    cached_data = json.loads(filestr)
    return cached_data


def prepare_dataframe(
    cached_data: Dict[str, Any], objective: str = None
) -> Tuple[pd.DataFrame, Dict[str, List[Any]]]:
    if objective is None:
        objective = cached_data.get("objective", "time")

    data = list(cached_data["cache"].values())
    data = [
        d
        for d in data
        if d.get(objective) != 1e20 and not isinstance(d.get(objective), str)
    ]

    tune_params_keys = cached_data["tune_params_keys"]
    all_tune_params = {}
    for key in tune_params_keys:
        values = cached_data["tune_params"][key]
        for row in data:
            if row[key] not in values:
                values = sorted(values + [row[key]])
        all_tune_params[key] = values

    # figure out which keys are interesting
    single_value_tune_param_keys = [
        key for key in tune_params_keys if len(all_tune_params[key]) == 1
    ]
    tune_param_keys = [
        key for key in tune_params_keys if key not in single_value_tune_param_keys
    ]
    scalar_value_keys = [
        key
        for key in data[0].keys()
        if not isinstance(data[0][key], list)
        and key not in single_value_tune_param_keys
    ]
    output_keys = [key for key in scalar_value_keys if key not in tune_param_keys]

    df = pd.DataFrame(data)[scalar_value_keys]

    # Add 'index' column (numeric) so the UI can select "index" as an axis
    df = df.reset_index(drop=True)
    df.insert(0, "index", df.index.astype(int))

    # Convert tune params to categorical where appropriate to preserve ordering
    for key in tune_param_keys:
        if key in df.columns:
            df[key] = pd.Categorical(
                df[key], categories=all_tune_params[key], ordered=True
            )

    return df, {
        "tune_param_keys": tune_param_keys,
        "all_tune_params": all_tune_params,
        "scalar_value_keys": scalar_value_keys,
        "output_keys": output_keys,
    }


def filter_dataframe(
    df: pd.DataFrame, selections: Dict[str, List[Any]]
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for k, v in selections.items():
        if v:
            mask &= df[k].isin(v)
    return df[mask]


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    xscale: str,
    yscale: str,
    palette: str = "Viridis",
) -> Any:
    # For categorical axes, we can map categories to numbers and add jitter for visual separation
    df_plot = df.copy()

    def jitter(col):
        dtype = df_plot[col].dtype
        if isinstance(dtype, pd.CategoricalDtype) or dtype == object:
            categories = list(pd.Categorical(df_plot[col]).categories)
            mapping = {c: i for i, c in enumerate(categories)}
            arr = df_plot[col].map(mapping).astype(float)
            arr += np.random.normal(scale=0.15, size=len(arr))
            return arr, categories
        else:
            return df_plot[col], None

    x_vals, x_cats = jitter(x)
    y_vals, y_cats = jitter(y)

    df_plot["_x"] = x_vals
    df_plot["_y"] = y_vals

    color_arg = color if color in df_plot.columns else None

    # Determine palette lists from plotly (sequential palettes only)
    seq = getattr(px.colors.sequential, palette, None)
    color_kwargs = {"color_continuous_scale": seq}

    fig = px.scatter(
        df_plot,
        x="_x",
        y="_y",
        color=color_arg,
        hover_data=df_plot.columns,
        height=600,
        width=900,
        labels={"_x": x, "_y": y},
        **color_kwargs,
    )

    # If axis corresponded to categories, set tick labels
    if x_cats is not None:
        fig.update_xaxes(
            tickmode="array", tickvals=list(range(len(x_cats))), ticktext=x_cats
        )
    if y_cats is not None:
        fig.update_yaxes(
            tickmode="array", tickvals=list(range(len(y_cats))), ticktext=y_cats
        )

    # Set log scales if requested
    if xscale == "log":
        fig.update_xaxes(type="log")
    if yscale == "log":
        fig.update_yaxes(type="log")

    return fig


def main():
    parser = argparse.ArgumentParser(description="Streamlit Kernel Tuner Dashboard")
    parser.add_argument("cachefile", nargs="?", help="Path to cache file (JSON)")
    args = parser.parse_args()

    st.set_page_config(layout="wide", page_title="Kernel Tuner Dashboard")

    st.title("Kernel Tuner Dashboard")

    # Allow providing file via arg or file uploader
    cachefile = args.cachefile

    if not cachefile:
        uploaded = st.sidebar.file_uploader("Upload cache JSON file", type=["json"])
        if uploaded is not None:
            # save to a temp file so we can read locations
            import tempfile

            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            tf.write(uploaded.read())
            tf.flush()
            cachefile = tf.name

    if not cachefile:
        st.info(
            "Provide a cache file via command-line (streamlit run ... -- <cachefile>) or upload one in the sidebar."
        )
        return

    try:
        cached_data = _read_cachefile(cachefile)
    except Exception as exc:
        st.error(f"Failed to read cache file: {exc}")
        return

    kernel_name = cached_data.get("kernel_name", "<unknown>")
    device_name = cached_data.get("device_name", "<unknown>")

    st.sidebar.markdown(f"**Kernel:** {kernel_name}")
    st.sidebar.markdown(f"**Device:** {device_name}")

    df, meta = prepare_dataframe(cached_data)

    scalar_value_keys = meta["scalar_value_keys"]
    tune_param_keys = meta["tune_param_keys"]
    all_tune_params = meta["all_tune_params"]

    default_key = (
        "GFLOP/s"
        if "GFLOP/s" in scalar_value_keys
        else ("time" if "time" in scalar_value_keys else scalar_value_keys[0])
    )

    yvariable = st.sidebar.selectbox(
        "Y", options=scalar_value_keys, index=scalar_value_keys.index(default_key)
    )
    xvariable = st.sidebar.selectbox(
        "X", options=["index"] + scalar_value_keys, index=0
    )
    colorvariable = st.sidebar.selectbox(
        "Color By",
        options=scalar_value_keys,
        index=scalar_value_keys.index(default_key),
    )
    xscale = st.sidebar.radio("X axis scale", options=["linear", "log"], index=0)
    yscale = st.sidebar.radio("Y axis scale", options=["linear", "log"], index=0)

    # Color palette chooser (sequential palettes only)
    seq_names = [
        name
        for name in dir(px.colors.sequential)
        if not name.startswith("_")
        and isinstance(getattr(px.colors.sequential, name), list)
    ]
    seq_names = sorted(seq_names)
    default_idx = seq_names.index("Viridis") if "Viridis" in seq_names else 0
    palette = st.sidebar.selectbox(
        "Color palette", options=seq_names, index=default_idx
    )

    # tune param multi-selects
    selections = {}
    for tp in tune_param_keys:
        selections[tp] = st.sidebar.multiselect(
            tp, options=all_tune_params[tp], default=list(all_tune_params[tp])
        )

    filtered_df = filter_dataframe(df, selections)

    st.markdown(f"## Auto-tuning {kernel_name} on {device_name}")

    fig = plot_scatter(
        filtered_df,
        xvariable,
        yvariable,
        colorvariable,
        xscale,
        yscale,
        palette=palette,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.markdown("### Top results")

    # Show best by selected y (if numeric)
    if pd.api.types.is_numeric_dtype(df[yvariable]):
        best = df.sort_values(yvariable).head(10)
        st.dataframe(best)
    else:
        st.write("Y variable is non-numeric; showing raw filtered data")
        st.dataframe(filtered_df)


if __name__ == "__main__":
    main()
