from typing import Dict, Any, List
import json
import pandas as pd


class Dashboard:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file

        # Open cachefile for reading appended results
        self.cache_file_handle = open(cache_file, "r")
        filestr = self.cache_file_handle.read().strip()

        # Try to be tolerant for trailing missing brackets or commas
        if filestr and not filestr.endswith("}\n}"):
            if filestr[-1] == ",":
                filestr = filestr[:-1]
            filestr = filestr + "}\n}"

        cached_data = json.loads(filestr) if filestr else {}

        self.kernel_name = cached_data.get("kernel_name", "<unknown>")
        self.device_name = cached_data.get("device_name", "<unknown>")
        self.objective = cached_data.get("objective", "time")

        # raw performance records
        data = list(cached_data.get("cache", {}).values())
        data = [
            d
            for d in data
            if d.get(self.objective) != 1e20
            and not isinstance(d.get(self.objective), str)
        ]

        self.index = len(data)
        self._all_data = data

        # tune parameters
        self.tune_params_keys = cached_data.get("tune_params_keys", [])
        self.all_tune_params: Dict[str, List[Any]] = {}
        for key in self.tune_params_keys:
            values = cached_data.get("tune_params", {}).get(key, [])
            for row in data:
                if row.get(key) not in values:
                    values = sorted(values + [row.get(key)])
            self.all_tune_params[key] = values

        # find keys
        self.single_value_tune_param_keys = [
            k for k in self.tune_params_keys if len(self.all_tune_params[k]) == 1
        ]
        self.tune_param_keys = [
            k
            for k in self.tune_params_keys
            if k not in self.single_value_tune_param_keys
        ]

        scalar_value_keys = [
            k
            for k in (data[0].keys() if data else [])
            if not isinstance(data[0][k], list)
            and k not in self.single_value_tune_param_keys
        ]
        self.scalar_value_keys = scalar_value_keys
        self.output_keys = [
            k for k in scalar_value_keys if k not in self.tune_param_keys
        ]
        self.float_keys = [
            k for k in self.output_keys if isinstance(data[0].get(k), float) if data
        ]

        # prepare DataFrame
        self.data_df = (
            pd.DataFrame(data)[self.scalar_value_keys]
            if len(self.scalar_value_keys) > 0
            else pd.DataFrame()
        )
        self.data_df = self.data_df.reset_index(drop=True)
        if not self.data_df.empty:
            self.data_df.insert(0, "index", self.data_df.index.astype(int))

        # categorical conversion for tune params
        for key in self.tune_param_keys:
            if key in self.data_df.columns:
                self.data_df[key] = pd.Categorical(
                    self.data_df[key],
                    categories=self.all_tune_params[key],
                    ordered=True,
                )

        # selections for filtering
        self.selected_tune_params = {
            key: self.all_tune_params[key].copy() for key in self.tune_param_keys
        }

    def close(self):
        try:
            self.cache_file_handle.close()
        except Exception:
            pass

    def get_filtered_df(self) -> pd.DataFrame:
        """Return filtered DataFrame based on selected_tune_params."""
        mask = pd.Series(True, index=self.data_df.index)
        for k, v in self.selected_tune_params.items():
            mask &= self.data_df[k].isin(v)
        return self.data_df[mask]

    def update_selection(self, key: str, values: List[Any]):
        """Update selection for a tune parameter."""
        self.selected_tune_params[key] = values

    def get_stream_for_index(self, i: int) -> Dict[str, List[Any]]:
        """Return a stream dict for a single element at index i."""
        element = self._all_data[i]
        stream_dict = {
            k: [v]
            for k, v in dict(element, index=i).items()
            if k in ["index"] + self.scalar_value_keys
        }
        return stream_dict

    def read_new_contents(self) -> List[Dict[str, List[Any]]]:
        """Read appended JSON from the cachefile and return list of stream_dicts for new entries."""
        new_contents = self.cache_file_handle.read().strip()
        stream_dicts = []
        if new_contents:
            # process new contents (parse as JSON, make into dict that goes into source.stream)
            new_contents_json = "{" + new_contents[:-1] + "}"
            new_data = list(json.loads(new_contents_json).values())
            for i, element in enumerate(new_data):
                stream_dict = {
                    k: [v]
                    for k, v in dict(element, index=self.index + i).items()
                    if k in ["index"] + self.scalar_value_keys
                }
                stream_dicts.append(stream_dict)
            self.index += len(new_data)
        return stream_dicts
