from dash import Input, Output, callback, html, State, no_update
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
        Output("history-store", "data", allow_duplicate=True),
        Input("clean-history-button-ica", "n_clicks"),
        State("history-store", "data"),
        prevent_initial_call=True,
    )
    def clean_history(n_clicks, history_data):
        if not n_clicks:
            return no_update
        
        print("COUCOU")
        
        history_data = history_data or {}
        HISTORY_CATEGORIES = ["ICA"]

        for category in HISTORY_CATEGORIES:
            history_data.pop(category, None)

        return history_data


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

def register_update_ica_components():
    @callback(
        Output("ica-components-selection", "options"),
        Input("sidebar-tabs-ica", "active_tab"),
        Input("history-store", "data"),
        prevent_initial_call=False,
    )
    def _update_ica_options(active_tab, history_data):
        if not history_data or "metadata" not in history_data:
            return []
        
        n_components = history_data["metadata"].get("last_ica_count")
        if n_components is None:
            return []
        
        excluded: set = set(history_data.get("excluded_ica_components", []))

        options = []
        for i in range(n_components):
            if i in excluded:
                options.append({
                    "label": f"ICA {i:02d}  ✗ excluded",
                    "value": i,
                    "disabled": True,               # grayed out in Dash Checklist
                })
            else:
                options.append({
                    "label": f"ICA {i:02d}",
                    "value": i,
                    "disabled": False,
                })

        return options
        