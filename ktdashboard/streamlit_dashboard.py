import argparse
from typing import Any
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from dashboard import Dashboard


class StreamlitDashboard:
    def __init__(self, cachefile: str):
        self.model = Dashboard(cachefile)
        self.df = self.model.data_df
        self.plot_height = 600

    def plot_scatter(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color: str,
        xscale: str,
        yscale: str,
        palette: str = "Viridis",
    ) -> Any:
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

        seq = getattr(px.colors.sequential, palette, None)
        color_kwargs = {"color_continuous_scale": seq}

        fig = px.scatter(
            df_plot,
            x="_x",
            y="_y",
            color=color_arg,
            hover_data=df_plot.columns,
            height=self.plot_height,
            labels={"_x": x, "_y": y},
            **color_kwargs,
        )

        if x_cats is not None:
            fig.update_xaxes(
                tickmode="array", tickvals=list(range(len(x_cats))), ticktext=x_cats
            )
        if y_cats is not None:
            fig.update_yaxes(
                tickmode="array", tickvals=list(range(len(y_cats))), ticktext=y_cats
            )

        if xscale == "log":
            fig.update_xaxes(type="log")
        if yscale == "log":
            fig.update_yaxes(type="log")

        return fig

    def render(self):
        st.set_page_config(layout="wide", page_title="Kernel Tuner Dashboard")
        st.title("Kernel Tuner Dashboard")

        kernel_name = self.model.kernel_name
        device_name = self.model.device_name

        st.sidebar.markdown(f"**Kernel:** {kernel_name}")
        st.sidebar.markdown(f"**Device:** {device_name}")

        scalar_value_keys = self.model.scalar_value_keys
        tune_param_keys = self.model.tune_param_keys
        all_tune_params = self.model.all_tune_params

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

        # Show table control
        show_table = st.sidebar.checkbox("Show table", value=True)

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
            self.model.update_selection(tp, selections[tp])

        filtered_df = self.model.get_filtered_df()

        st.markdown(f"## Auto-tuning {kernel_name} on {device_name}")

        plot_height = self.plot_height
        if not show_table:
            plot_height = int(plot_height * 1.5)

        fig = self.plot_scatter(
            filtered_df,
            xvariable,
            yvariable,
            colorvariable,
            xscale,
            yscale,
            palette=palette,
        )

        st.plotly_chart(fig, height=plot_height, width="stretch")

        if show_table:
            st.markdown("---")

            if pd.api.types.is_numeric_dtype(filtered_df[yvariable]):
                sorted_df = filtered_df.sort_values(yvariable)
                st.dataframe(sorted_df)
            else:
                st.dataframe(filtered_df)


def serve_streamlit(cachefile: str) -> None:
    sd = StreamlitDashboard(cachefile)
    sd.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ktdashboard")
    parser.add_argument("filename", help="Path to cache JSON file")
    args = parser.parse_args()
    serve_streamlit(args.filename)
