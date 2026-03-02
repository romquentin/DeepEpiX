import pickle
import traceback
import dash
from dash import Input, Output, State, callback

from callbacks.utils import graph_utils as gu
from layout.config_layout import ERROR


def register_update_graph_raw_signal():
    @callback(
        Output("signal-graph", "figure"),
        Output("python-error", "children"),
        Output("python-error", "style"),
        Input("update-button", "n_clicks"),
        Input("page-selector", "value"),
        State("signal-graph", "figure"),
        State("montage-radio", "value"),
        State("channel-region-checkboxes", "value"),
        State("data-path-store", "data"),
        State("offset-display", "value"),
        State("colors-radio", "value"),
        State("chunk-limits-store", "data"),
        State("frequency-store", "data"),
        State("montage-store", "data"),
        State("channel-store", "data"),
        State("sensitivity-analysis-store", "data"),
        running=[(Output("update-button", "disabled"), True, False)],
        prevent_initial_call=True,
    )
    def _update_graph_raw_signal(
        n_clicks,
        page_selection,
        graph,
        montage_selection,
        channel_selection,
        data_path,
        offset_selection,
        color_selection,
        chunk_limits,
        freq_data,
        montage_store,
        channel_store,
        sensitivity_analysis_store,
    ):
        """Update M/EEG signal visualization based on time and channel selection."""

        if n_clicks == 0:
            return dash.no_update, dash.no_update, dash.no_update

        if not data_path:
            return (
                dash.no_update,
                "Please choose a subject to display on Home page.",
                ERROR,
            )

        if (
            None
            in (
                page_selection,
                offset_selection,
                color_selection,
                freq_data,
                channel_store,
            )
            or not chunk_limits
        ):
            return (
                dash.no_update,
                "You have a subject in memory but its recording has not been preprocessed yet. Please go back on Home page to reprocess the signal.",
                ERROR,
            )

        if montage_selection == "channel selection" and not channel_selection:
            return (
                dash.no_update,
                "Missing channel selection for graph rendering.",
                ERROR,
            )

        # Get the selected channels based on region
        if montage_selection == "channel selection":
            selected_channels = [
                channel
                for region_code in channel_selection
                if region_code in channel_store
                for channel in channel_store[region_code]
            ]

            if not selected_channels:
                return (
                    dash.no_update,
                    "No channels selected from the given regions",
                    ERROR,
                )

        # If montage selection is not "channel selection", use montage's corresponding channels
        elif montage_selection != "montage selection":
            selected_channels = montage_store.get(montage_selection, [])

            if not selected_channels:
                return (
                    dash.no_update,
                    f"No channels available for the selected montage: {montage_selection}",
                    ERROR,
                )

        time_range = chunk_limits[int(page_selection)]

        # Get the current x-axis center
        xaxis_range = graph["layout"]["xaxis"].get("range", [])
        if xaxis_range[1] <= time_range[0] or xaxis_range[0] >= time_range[1]:
            xaxis_range = [time_range[0], time_range[0] + 20]

        filter = {}

        if "smoothGrad" in color_selection:
            try:
                with open(sensitivity_analysis_store[0], "rb") as f:
                    filter = pickle.load(f)
            except KeyError:
                return dash.no_update, "No color selected for graph traces.", ERROR

        try:
            fig, error, error_style = gu.generate_graph_time_channel(
                selected_channels,
                float(offset_selection),
                time_range,
                data_path,
                freq_data,
                color_selection,
                xaxis_range,
                channel_store,
                filter,
            )

            return fig, error, error_style

        except FileNotFoundError:
            return dash.no_update, "⚠️ Error: Folder not found.", ERROR
        except ValueError as ve:
            return (
                dash.no_update,
                f"⚠️ Error: {str(ve)}.\n Details: {traceback.format_exc()}",
                ERROR,
            )
        except Exception as e:
            return (
                dash.no_update,
                f"⚠️ Error: Unexpected error {str(e)}.\n Details: {traceback.format_exc()}",
                ERROR,
            )


def register_update_graph_ica(ica_result_radio_id):
    @callback(
        Output("graph-ica", "figure"),
        Output("python-error-ica", "children"),
        Input("update-button-ica", "n_clicks"),
        Input("page-selector-ica", "value"),
        Input("ica-components-selection", "value"),
        State(ica_result_radio_id, "value"),
        State("data-path-store", "data"),
        State("offset-display-ica", "value"),
        State("colors-radio-ica", "value"),
        State("chunk-limits-store", "data"),
        State("n-components", "value"),
        State("ica-method", "value"),
        State("max-iter", "value"),
        State("decim", "value"),
        State("graph-ica", "figure"),
        State("history-store", "data"),
        prevent_initial_call=True,
    )
    def _update_graph_ica(
        n_clicks,
        page_selection,
        selected_indices,
        ica_result_path,
        data_path,
        offset_selection,
        color_selection,
        chunk_limits,
        n_components,
        ica_method,
        max_iter,
        decim,
        graph,
        history_data,
    ):
        """Update ICA signal visualization."""
        if n_clicks == 0:
            return dash.no_update, dash.no_update

        if not data_path:
            return dash.no_update, "Please choose a subject to display on Home page."

        if (
            None in (page_selection, offset_selection, color_selection)
            or not chunk_limits
        ):
            return (
                dash.no_update,
                "You have a subject in memory but its recording has not been preprocessed yet. Please go back on Home page to reprocess the signal.",
            )

        if None in (n_components, ica_method, max_iter, decim):
            return dash.no_update, "You haven't compt"
        
        permanently_excluded: set = set(
            (history_data or {}).get("excluded_ica_components", [])
        )
        all_grayed: list = sorted(permanently_excluded | set(selected_indices or []))

        time_range = chunk_limits[int(page_selection)]

        # Get the current x-axis center
        xaxis_range = graph["layout"]["xaxis"].get("range", [])
        if xaxis_range[1] < time_range[0] or xaxis_range[0] > time_range[1]:
            xaxis_range = [time_range[0], time_range[0] + 20]

        try:
            fig, error = gu.generate_graph_time_ica(
                offset_selection,
                time_range,
                data_path,
                ica_result_path,
                color_selection,
                xaxis_range,
                all_grayed,
            )

            return fig, error

        except FileNotFoundError:
            return dash.no_update, "⚠️ Error: Folder not found."
        except ValueError as ve:
            return (
                dash.no_update,
                f"⚠️ Error: {str(ve)}.\n Details: {traceback.format_exc()}",
            )
        except Exception as e:
            return (
                dash.no_update,
                f"⚠️ Error: Unexpected error {str(e)}.\n Details: {traceback.format_exc()}",
            )
