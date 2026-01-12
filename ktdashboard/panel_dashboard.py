from typing import Optional
import panel as pn
import panel.widgets as pnw
import pandas as pd
import bokeh.palettes
from bokeh.models.ranges import FactorRange
from bokeh.transform import jitter
from bokeh.models import HoverTool, LinearColorMapper, CategoricalColorMapper
from bokeh.plotting import ColumnDataSource, figure

from dashboard import Dashboard


class PanelDashboard:
    def __init__(self, cachefile: str, default_key: Optional[str] = None):
        self.model = Dashboard(cachefile)

        # local copies for UI
        self.data_df = self.model.data_df
        self.scalar_value_keys = self.model.scalar_value_keys
        self.tune_param_keys = self.model.tune_param_keys
        self.all_tune_params = self.model.all_tune_params
        self.source = ColumnDataSource(data=self._df_categorical_to_str(self.data_df))
        self.selected_tune_params = {
            key: self.all_tune_params[key].copy() for key in self.tune_param_keys
        }

        # layout parameters
        self.plot_height = 600
        plot_options = dict(
            height=self.plot_height,
            min_height=self.plot_height,
            sizing_mode="stretch_width",
        )
        float_keys = [k for k in self.scalar_value_keys if k in self.model.float_keys]
        plot_options["tools"] = [
            HoverTool(
                tooltips=[
                    (k, "@{" + k + "}" + ("{0.00}" if k in float_keys else ""))
                    for k in self.scalar_value_keys
                ]
            ),
            "box_select,box_zoom,save,reset",
        ]
        self.plot_options = plot_options

        # find default key
        if default_key is None:
            default_key = "GFLOP/s"
            if default_key not in self.scalar_value_keys:
                default_key = (
                    "time"
                    if "time" in self.scalar_value_keys
                    else (self.scalar_value_keys[0] if self.scalar_value_keys else None)
                )

        # Widgets
        self.yvariable = pnw.Select(
            name="Y", value=default_key, options=self.scalar_value_keys
        )
        self.xvariable = pnw.Select(
            name="X", value="index", options=["index"] + self.scalar_value_keys
        )
        self.colorvariable = pnw.Select(
            name="Color By", value=default_key, options=self.scalar_value_keys
        )
        self.xscale = pnw.RadioButtonGroup(name="xscale", options=["linear", "log"])
        self.yscale = pnw.RadioButtonGroup(name="yscale", options=["linear", "log"])

        # connect widgets
        self.scatter = pn.bind(
            self.make_scatter,
            xvariable=self.xvariable,
            yvariable=self.yvariable,
            color_by=self.colorvariable,
            xscale=self.xscale,
            yscale=self.yscale,
        )

        # build up the dashboard
        self.dashboard = pn.template.BootstrapTemplate(title="Kernel Tuner Dashboard")
        self.dashboard.main.append(self.scatter)
        self.dashboard.sidebar.append(
            pn.Column(self.yvariable, self.xvariable, self.colorvariable)
        )
        self.dashboard.sidebar.append(pn.layout.Divider())
        self.dashboard.sidebar.append(pn.Row(pn.pane.Markdown("X axis"), self.xscale))
        self.dashboard.sidebar.append(pn.Row(pn.pane.Markdown("Y axis"), self.yscale))
        self.dashboard.sidebar.append(pn.layout.Divider())

        for tune_param in self.tune_param_keys:
            values = self.all_tune_params[tune_param]
            multi_choice = pnw.MultiChoice(
                name=tune_param, value=values, options=values
            )
            self.dashboard.sidebar.append(multi_choice)
            row = pn.bind(self.update_data_selection, tune_param, multi_choice)
            self.dashboard.sidebar.append(row)

    def _df_categorical_to_str(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `df` where categorical columns are converted to strings."""
        df2 = df.copy()
        for c in df2.columns:
            if pd.api.types.is_categorical_dtype(df2[c]):
                df2[c] = df2[c].astype(str)
        return df2

    def _convert_stream_dict(self, sd: dict) -> dict:
        """Convert values inside a stream-dict to string for categorical columns."""
        sd2 = {}
        for k, v in sd.items():
            if k in self.data_df.columns and pd.api.types.is_categorical_dtype(
                self.data_df[k]
            ):
                sd2[k] = [str(x) for x in v]
            else:
                sd2[k] = v
        return sd2

    def update_data_selection(self, tune_param, multi_choice):
        self.selected_tune_params[tune_param] = multi_choice
        # Cross selection based on all selections in all tunable parameters
        mask = pd.Series(True, index=self.data_df.index)
        for k, v in self.selected_tune_params.items():
            mask &= self.data_df[k].isin(v)
        data_df = self.data_df[mask]
        self.source.data = self._df_categorical_to_str(data_df)

    def update_colors(self, color_by):
        dtype = self.data_df.dtypes[color_by]

        if dtype == "category":
            factors = [str(f) for f in dtype.categories]
            if len(factors) < 10:
                palette = bokeh.palettes.Category10[10]
            else:
                palette = bokeh.palettes.Category20[20]
            color_mapper = CategoricalColorMapper(palette=palette, factors=factors)
        else:
            color_mapper = LinearColorMapper(
                palette="Viridis256",
                low=min(self.data_df[color_by]),
                high=max(self.data_df[color_by]),
            )

        color = {"field": color_by, "transform": color_mapper}
        return color

    def make_scatter(self, xvariable, yvariable, color_by, xscale, yscale):
        color = self.update_colors(color_by)

        x = xvariable
        y = yvariable

        plot_options = dict(self.plot_options)
        plot_options["x_axis_type"] = xscale
        plot_options["y_axis_type"] = yscale

        dtype = self.data_df.dtypes.get(xvariable)
        if pd.api.types.is_categorical_dtype(dtype):
            x_factors = [str(f) for f in dtype.categories]
            plot_options["x_range"] = x_factors
            x = jitter(
                xvariable,
                width=0.02,
                distribution="normal",
                range=FactorRange(*x_factors),
            )

        dtype = self.data_df.dtypes.get(yvariable)
        if pd.api.types.is_categorical_dtype(dtype):
            y_factors = [str(f) for f in dtype.categories]
            plot_options["y_range"] = y_factors
            y = jitter(
                yvariable,
                width=0.02,
                distribution="normal",
                range=FactorRange(*y_factors),
            )

        f = figure(**plot_options)
        f.scatter(x, y, size=5, color=color, alpha=0.5, source=self.source)
        f.xaxis.axis_label = xvariable
        f.yaxis.axis_label = yvariable

        bokeh_pane = pn.pane.Bokeh(
            object=f,
            min_width=self.plot_width,
            min_height=self.plot_height,
            max_width=self.plot_width,
            max_height=self.plot_height,
        )
        pane = pn.Column(
            pn.pane.Markdown(
                f"## Auto-tuning {self.model.kernel_name} on {self.model.device_name}"
            ),
            bokeh_pane,
        )
        return pane

    def update_plot(self, i):
        sd = self.model.get_stream_for_index(i)
        self.source.stream(self._convert_stream_dict(sd))

    def update_data(self):
        stream_dicts = self.model.read_new_contents()
        for sd in stream_dicts:
            self.source.stream(self._convert_stream_dict(sd))


def serve_panel(cachefile: str) -> None:
    ui = PanelDashboard(cachefile)

    ui.dashboard.servable()

    def dashboard_f():
        pn.state.add_periodic_callback(ui.update_data, 1000)
        return ui.dashboard

    pn.serve(dashboard_f, show=False)
