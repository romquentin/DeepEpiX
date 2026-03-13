# view.py
import dash
from dash import html, dcc
from dash_extensions import Keyboard

import dash_bootstrap_components as dbc

from callbacks.selection_callbacks import (
    register_toggle_sidebar,
    register_navigate_tabs_ica,
    register_cancel_or_confirm_annotation_suppression,
    register_annotation_checkboxes_options,
    register_annotation_dropdown_options,
    register_clear_check_all_annotation_checkboxes,
    register_offset_display,
    register_page_buttons_display,
    register_modal_annotation_suppression,
    register_toggle_intersection_modal,
    register_create_intersection,
    register_fill_signal_versions,
)

from callbacks.annotation_callbacks import (
    register_move_to_next_annotation,
    register_update_annotation_graph,
    register_update_annotations_on_graph,
)

from callbacks.history_callbacks import register_update_ica_history, register_update_ica_components

from callbacks.ica_callbacks import register_compute_ica, register_fill_ica_results, register_apply_ica_exclusion, register_plot_ica_maps

from callbacks.graph_callbacks import register_update_graph_ica

# Layout imports
import layout.graph_layout as gl
from layout.ica_sidebar_layout import create_sidebar

dash.register_page(__name__, name="ICA", path="/viz/ica")


layout = html.Div(
    [
        Keyboard(id="keyboard-ica", captureKeys=["ArrowRight", "ArrowLeft", "+", "-"]),
        html.Div(
            [
                # Sidebar container
                html.Div(
                    [
                        # Collapsible sidebar
                        dbc.Collapse(
                            create_sidebar(),
                            id="sidebar-collapse-ica",
                            is_open=True,
                            dimension="width",
                            className="sidebar-collapse",
                        ),
                        # Button stack on the left
                        html.Div(
                            [
                                dbc.Button(
                                    html.I(
                                        id="sidebar-toggle-icon-ica",
                                        className="bi bi-x-lg",
                                    ),
                                    id="toggle-sidebar-ica",
                                    color="danger",
                                    size="sm",
                                    className="mb-2 shadow-sm",
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-noise-reduction"),
                                    id="nav-compute-ica",
                                    color="warning",
                                    size="sm",
                                    className="mb-2",
                                    title="Compute ICA",
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-hand-index-thumb"),
                                    id="nav-select-ica",
                                    color="primary",
                                    size="sm",
                                    className="mb-2",
                                    title="Select",
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-grid-3x3-gap"),
                                    id="nav-components-ica",
                                    color="info",
                                    size="sm",
                                    className="mb-2",
                                    title="View ICA Scalp Field Topographies",
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "alignItems": "center",
                                "marginTop": "10px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "row",  # button -> collapse (left to right)
                        "alignItems": "flex-start",
                        "zIndex": 1000,
                    },
                ),
                gl.create_graph_container(
                    update_button_id="update-button-ica",
                    page_selector_id="page-selector-ica",
                    next_spike_buttons_container_id="next-spike-buttons-container-ica",
                    prev_spike_id="prev-spike-ica",
                    next_spike_id="next-spike-ica",
                    annotation_dropdown_id="annotation-dropdown-ica",
                    loading_id="loading-graph-ica",
                    signal_graph_id="graph-ica",
                    annotation_graph_id="annotation-graph-ica",
                ),
            ],
            style={
                "display": "flex",  # Horizontal layout
                "flexDirection": "row",
                "height": "85vh",  # Use the full height of the viewport
                "width": "95vw",  # Use the full width of the viewport
                "overflow": "hidden",  # Prevent overflow in case of resizing
                "boxSizing": "border-box",
                "gap": "20px",
            },
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(
                                    "Scalp Field Topographies of each component",
                                    style={"fontWeight": "bold", "fontSize": "14px"},
                                ),
                                dbc.Button(
                                    html.I(className="bi bi-x-lg"),
                                    id="close-components-ica",
                                    color="link",
                                    size="sm",
                                    style={"color": "white", "padding": "0"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "backgroundColor": "#2eafc9",
                                "color": "white",
                                "padding": "6px 10px",
                                "borderRadius": "6px 6px 0 0",
                                "cursor": "move",
                            },
                            id="components-window-header-ica",
                        ),
                        dbc.Collapse(
                            html.Div(
                                [
                                    dcc.Loading(
                                        id="loading-components-ica",
                                        children=html.Div(
                                            id="components-content-ica",
                                            style={
                                                "padding": "10px",
                                                "overflowY": "auto",
                                                "maxHeight": "60vh",
                                            },
                                        ),
                                        type="circle",
                                    ),
                                ],
                            ),
                            id="components-body-collapse-ica",
                            is_open=True,
                        ),
                    ],
                    style={
                        "backgroundColor": "white",
                        "border": "1px solid #dee2e6",
                        "borderRadius": "6px",
                        "boxShadow": "0 4px 15px rgba(0,0,0,0.2)",
                        "width": "500px",
                        "minWidth": "600px",
                    },
                ),
            ],
            id="components-window-ica",
            style={
                "position": "fixed",
                "top": "70px",
                "right": "20px",
                "zIndex": 2000,
                "display": "none",
            },
        ),
        html.Div(id="python-error-ica"),
    ]
)

# --- Siderbar dynamic ---
register_toggle_sidebar(
    collapse_id="sidebar-collapse-ica",
    icon_id="sidebar-toggle-icon-ica",
    toggle_id="toggle-sidebar-ica",
)

register_navigate_tabs_ica(
    collapse_id="sidebar-collapse-ica",
    sidebar_tabs_id="sidebar-tabs-ica",
    icon_id="sidebar-toggle-icon-ica",
)


# specific to ica.py
register_compute_ica()

register_update_graph_ica(ica_result_radio_id="ica-result-radio")

register_update_ica_history()

register_update_ica_components(ica_result_radio_id="ica-result-radio")

register_fill_ica_results(ica_result_radio_id="ica-result-radio")

register_apply_ica_exclusion()

register_plot_ica_maps(ica_result_radio_id="ica-result-radio")

register_fill_signal_versions()

# same as (raw) viz.py
register_cancel_or_confirm_annotation_suppression(
    confirm_btn_id="confirm-delete-btn-ica",
    cancel_btn_id="cancel-delete-btn-ica",
    checkboxes_id="annotation-checkboxes-ica",
    modal_id="delete-confirmation-modal-ica",
)

register_annotation_checkboxes_options(
    checkboxes_id="annotation-checkboxes-ica",
)
register_annotation_dropdown_options(
    dropdown_id="annotation-dropdown-ica", checkboxes_id="annotation-checkboxes-ica"
)
register_clear_check_all_annotation_checkboxes(
    check_all_btn_id="check-all-annotations-btn-ica",
    clear_all_btn_id="clear-all-annotations-btn-ica",
    checkboxes_id="annotation-checkboxes-ica",
)
register_offset_display(
    offset_decrement_id="offset-decrement-ica",
    offset_increment_id="offset-increment-ica",
    offset_display_id="offset-display-ica",
    keyboard_id="keyboard-ica",
)
register_page_buttons_display(page_selector_id="page-selector-ica")

register_modal_annotation_suppression(
    btn_id="delete-annotations-btn-ica",
    checkboxes_id="annotation-checkboxes-ica",
    modal_id="delete-confirmation-modal-ica",
    modal_body_id="delete-modal-body-ica",
)

register_toggle_intersection_modal(
    btn_id="create-intersection-btn-ica",
    checkboxes_id="annotation-checkboxes-ica",
    modal_id="create-intersection-modal-ica",
    modal_body_id="create-intersection-modal-body-ica",
)

register_create_intersection(
    confirm_btn_id="confirm-intersection-btn-ica",
    cancel_btn_id="cancel-intersection-btn-ica",
    checkboxes_id="annotation-checkboxes-ica",
    tolerance_id="intersection-tolerance-ica",
    modal_id="create-intersection-modal-ica",
)

register_move_to_next_annotation(
    prev_spike_id="prev-spike-ica",
    next_spike_id="next-spike-ica",
    graph_id="graph-ica",
    dropdown_id="annotation-dropdown-ica",
    checkboxes_id="annotation-checkboxes-ica",
    page_selector_id="page-selector-ica",
)

register_update_annotation_graph(
    update_button_id="update-button-ica",
    page_selector_id="page-selector-ica",
    checkboxes_id="annotation-checkboxes-ica",
    annotation_graph_id="annotation-graph-ica",
)

register_update_annotations_on_graph(
    graph_id="graph-ica",
    checkboxes_id="annotation-checkboxes-ica",
    page_selector_id="page-selector-ica",
)
