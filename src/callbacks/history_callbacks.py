from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
from callbacks.utils import history_utils as hu


def register_update_annotation_history():
    @callback(
        Output("history-log", "children"),
        Input("url", "pathname"),
        Input("history-store", "data"),
        prevent_initial_call=False,
    )
    def update_history(pathname, history_data):
        category = "annotations"
        return html.Div(
            [
                (
                    dbc.ListGroup(
                        [
                            dbc.ListGroupItem(entry)
                            for entry in hu.read_history_data_by_category(
                                history_data, category
                            )
                        ]
                    )
                    if hu.read_history_data_by_category(history_data, category)
                    else html.P("No entries yet.")
                )
            ]
        )


def register_clean_annotation_history():
    @callback(
        Output("history-store", "clear_data", allow_duplicate=True),
        Input("clean-history-button", "n_clicks"),
        prevent_initial_call=True,
    )
    def clean_history(n_clicks):
        if n_clicks > 0:
            return True


def register_update_ica_history():
    @callback(
        Output("history-log-ica", "children"),
        Input("sidebar-tabs-ica", "active_tab"),
        Input("history-store", "data"),
        prevent_initial_call=False,
    )
    def update_history(pathname, history_data):
        category = "ICA"
        return html.Div(
            [
                (
                    dbc.ListGroup(
                        [
                            dbc.ListGroupItem(entry)
                            for entry in hu.read_history_data_by_category(
                                history_data, category
                            )
                        ]
                    )
                    if hu.read_history_data_by_category(history_data, category)
                    else html.P("No entries yet.")
                )
            ]
        )

def register_restore_ica_channels():
    @callback(
        Output("ica-components-selection", "options"),
        Input("sidebar-tabs-ica", "active_tab"),
        Input("ica-store", "data"),
        State("n-components", "value"),
        prevent_initial_call=False,
    )
    def restore_channels(active_tab, ica_store_data, n_components):
        if not ica_store_data:
            return []

        options = [
            {"label": f"ICA {i:03d}", "value": i} 
            for i in range(n_components)
        ]
        return options