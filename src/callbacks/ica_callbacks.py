import os
import dash
from dash import Input, Output, State, callback
import mne
import config
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import path_utils as dpu
from callbacks.utils import history_utils as hu


def register_compute_ica():
    @callback(
        Output("ica-status", "children"),
        Output("compute-ica-button", "n_clicks"),
        Output("ica-store", "data"),
        Output("history-store", "data", allow_duplicate=True),
        Output("ica-components-selection", "options"),
        Input("compute-ica-button", "n_clicks"),
        State("data-path-store", "data"),
        State("chunk-limits-store", "data"),
        State("n-components", "value"),
        State("ica-method", "value"),
        State("max-iter", "value"),
        State("decim", "value"),
        State("frequency-store", "data"),
        State("history-store", "data"),
        State("ica-store", "data"),
        State("channel-store", "data"),
        prevent_initial_call=True,
    )
    def _compute_ica(
        n_clicks,
        data_path,
        chunk_limits,
        n_components,
        ica_method,
        max_iter,
        decim,
        freq_data,
        history_data,
        ica_store,
        channel_store,
    ):
        """Update ICA signal visualization."""

        if n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Validation: Check if all required fields are filled
        if not data_path:
            error_message = "⚠️ Please choose a subject to display on Home page."
            return error_message, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if not chunk_limits:
            error_message = "⚠️ You have a subject in memory but its recording has not been preprocessed yet. Please go back on Home page to reprocess the signal."
            return error_message, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        missing_fields = []
        if not n_components:
            missing_fields.append("N components")
        if not ica_method:
            missing_fields.append("ICA method")
        if not max_iter:
            missing_fields.append("Max iterations")
        if not decim:
            missing_fields.append("Temporal decimation")
        if missing_fields:
            error_message = (
                f"⚠️ Please fill in all required fields: {', '.join(missing_fields)}"
            )
            return (
                error_message,
                dash.no_update,
                dash.no_update,
            )

        cache_dir = config.CACHE_DIR
        ica_result_path = (
            cache_dir / f"{n_components}_{ica_method}_{max_iter}_{decim}-ica.fif"
        )
        if ica_result_path.exists() and str(ica_result_path) in ica_store:
            options = [{"label": f"ICA {i}", "value": i} for i in range(n_components)]
            return "✅ Reusing existing ICA results", 0, dash.no_update, dash.no_update, options

        raw = dpu.read_raw(
            data_path,
            preload=True,
            verbose=False,
            bad_channels=channel_store.get("bad", []),
        ).pick_types(meg=True)
        raw = raw.filter(l_freq=1.0, h_freq=None)

        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=ica_method,
            max_iter=max_iter,
            random_state=97,
        )
        ica.fit(raw, decim=decim)
        ica.save(ica_result_path, overwrite=True)

        new_options = [{"label": f"ICA {i:02d}", "value": i} for i in range(ica.n_components_)]

        for chunk_idx in chunk_limits:
            start_time, end_time = chunk_idx
            pu.get_ica_dataframe_dask(
                data_path, start_time, end_time, ica_result_path, raw
            )

        ica_store = [str(ica_result_path)]

        action = f"Computed ICA with <n_components = {n_components}, method: {ica_method}, max_iter: {max_iter}, decim: {decim}> as parameters.\n"
        history_data = hu.fill_history_data(history_data, "ICA", action)
        return None, 0, ica_store, history_data, new_options


def register_fill_ica_results(ica_result_radio_id):
    @callback(
        Output(ica_result_radio_id, "options"),
        Input("sidebar-tabs-ica", "active_tab"),
        Input("ica-store", "data"),
        prevent_initial_call=False,
    )
    def fill_ica_results(pathname, ica_store):
        if ica_store is None:
            return dash.no_update

        return [{"label": os.path.basename(k), "value": k} for k in ica_store]
