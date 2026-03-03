import os
import dash
from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
import config
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import path_utils as dpu
from callbacks.utils import history_utils as hu
from callbacks.utils import graph_utils as gu


def register_compute_ica():
    @callback(
        Output("ica-status", "children"),
        Output("compute-ica-button", "n_clicks"),
        Output("ica-store", "data"),
        Output("history-store", "data", allow_duplicate=True),
        Output("ica-components-dir-store", "data"),
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
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        try: 
            ica_path, components_dir, is_from_cache, n_components, explained_var = pu.run_ica_processing(
                data_path, n_components, ica_method, max_iter, decim, 
                channel_store, config.CACHE_DIR, ica_store
            )

            status_msg = "✅ Reusing existing ICA results" if is_from_cache else "✅ Calcul ICA terminé"

            for start_time, end_time in chunk_limits:
                pu.get_ica_dataframe_dask(data_path, start_time, end_time, ica_path)

            action = f"Computed ICA with <n_components = {n_components}, method: {ica_method}, max_iter: {max_iter}, decim: {decim}> as parameters.\n"
            history_data = hu.fill_history_data(history_data, "ICA", action, n_components, explained_var)
            history_data["excluded_ica_components"] = []

            return status_msg, 0, [str(ica_path)], history_data, str(components_dir)

        except Exception as e:
            return f"Erreur : {str(e)}", dash.no_update, dash.no_update, dash.no_update, dash.no_update

def register_apply_ica_exclusion():
    @callback(
        Output("history-store", "data", allow_duplicate=True),
        Output("exclusion-status", "children"),
        Output("ica-components-selection", "value"),
        Input("apply-ica-exclusion-button", "n_clicks"),
        State("ica-components-selection", "value"),
        State("history-store", "data"),
        State("data-path-store", "data"),
        State("chunk-limits-store", "data"),
        State("ica-result-radio", "value"),
        State("frequency-store", "data"),
        State("channel-store", "data"),
        prevent_initial_call=True,
    )
    def _apply_ica_exclusion(
        n_clicks, selected, history_data, data_path, 
        chunk_limits, ica_result_path, freq_data, channel_store):

        if not ica_result_path:
            return dash.no_update, dbc.Alert("No ICA result selected.", color="warning"), []
        
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update

        if not selected:
            return (
                dash.no_update,
                dbc.Alert("No components selected.", color="warning", duration=3000),
                [],
            )
    
        history_data = history_data or {}
        already_excluded = set(history_data.get("excluded_ica_components", []))
        all_excluded = sorted(already_excluded | set(selected))

        prep_raw = pu.sort_filter_resample(data_path, freq_data, channel_store)

        for start_time, end_time in chunk_limits:
            
            pu.get_ica_cleaned_dataframe_dask(
                data_path,
                freq_data,
                start_time,
                end_time,
                ica_result_path,
                all_excluded,
                prep_raw,
            )

        history_data["excluded_ica_components"] = all_excluded
        action = f"Excluded ICA components {all_excluded} from signal.\n"
        history_data = hu.fill_history_data(history_data, "ICA", action)

        status = dbc.Alert(
            f"{len(all_excluded)} component(s) permanently excluded "
            f"({len(all_excluded) - len(already_excluded)} new).",
            color="danger",
            duration=4000,
        )

        return history_data, status, []


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
    
def register_plot_ica_maps():
    @callback(
        Output("components-window-ica", "style"),
        Output("components-content-ica", "children"),
        Input("nav-components-ica", "n_clicks"),
        Input("close-components-ica", "n_clicks"),
        State("components-window-ica", "style"),
        State("ica-components-dir-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_components_window(
        open_clicks, close_clicks, current_style, components_dir
    ):
        triggered = dash.ctx.triggered_id

        base_style = {
            "position": "fixed",
            "top": "70px",
            "right": "20px",
            "zIndex": 2000,
        }

        if triggered == "close-components-ica":
            return {**base_style, "display": "none"}, dash.no_update

        if triggered == "nav-components-ica":

            if not components_dir:
                content = html.Div(
                    "⚠️ Aucune composante disponible. Veuillez d'abord calculer l'ICA.",
                    style={"padding": "12px", "color": "orange"},
                )
                return {**base_style, "display": "block"}, content

            images = gu.get_ica_components_figures(components_dir)

            content = html.Div(
                [
                    html.Img(
                        src=src,
                        style={"width": "100%", "marginBottom": "8px"},
                    )
                    for src in images
                ],
                style={"overflowY": "auto", "maxHeight": "60vh", "padding": "8px"},
            )
            return {**base_style, "display": "block"}, content

        return current_style, dash.no_update
    