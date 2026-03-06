from dash import html, dcc
import dash_bootstrap_components as dbc
from layout.config_layout import (
    INPUT_STYLES,
    BOX_STYLES,
    BUTTON_STYLES,
    LABEL_STYLES,
    ICON,
)
from layout.selection_sidebar_layout import create_selection


def create_compute():
    return html.Div(
        [
            html.Div(
                [
                    html.Label(
                        "Number of Components:", style={**LABEL_STYLES["classic"]}
                    ),
                    dbc.Input(
                        id="n-components",  # Unique ID for each input
                        type="number",
                        placeholder="n-components ...",
                        size="sm",
                        persistence=True,
                        persistence_type="session",
                        style={**INPUT_STYLES["small-number"]},
                    ),
                    dbc.Tooltip(
                        """
                n_components : int | float\n

                Number of principal components (from the pre-whitening PCA step) 
                that are passed to the ICA algorithm during fitting.

                - int: Must be >1 and ≤ number of channels.
                - float (0 < x < 1): Selects smallest number of components needed 
                to exceed this cumulative variance threshold.
                """,
                        target="n-components",
                        placement="right",
                        class_name="custom-tooltip",
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Label("ICA Method:", style={**LABEL_STYLES["classic"]}),
                    dcc.Dropdown(
                        id="ica-method",
                        options=[
                            {"label": "fastica", "value": "fastica"},
                            {"label": "infomax", "value": "infomax"},
                            {"label": "picard", "value": "picard"},
                        ],
                        placeholder="Select...",
                        persistence=True,
                        persistence_type="session",
                    ),
                    dbc.Tooltip(
                        """
                ICA Method:

                - fastica: A FastICA algorithm from sklearn.
                - infomax: Infomax ICA from MNE (uses CUDA if available).
                - picard: Picard algorithm, optimized for speed and stability.

                Choose depending on speed vs. accuracy needs.
                """,
                        target="ica-method",
                        placement="right",
                        class_name="custom-tooltip",
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Label(
                        "Max Iterations for Convergence:",
                        style={**LABEL_STYLES["classic"]},
                    ),
                    dbc.Input(
                        id="max-iter",  # Unique ID for each input
                        type="number",
                        placeholder="max-iter",
                        step=1,
                        min=1,
                        max=2000,
                        size="sm",
                        persistence=True,
                        persistence_type="session",
                        style={**INPUT_STYLES["small-number"]},
                    ),
                    dbc.Tooltip(
                        """
                Maximum number of iterations during ICA fitting.

                - If 'auto':
                • Sets max_iter = 1000 for 'fastica'
                • Sets max_iter = 500 for 'infomax' or 'picard'
                - The actual number of iterations used will be stored in `n_iter_`.

                Increase this value if ICA does not converge.
                """,
                        target="max-iter",
                        placement="right",
                        class_name="custom-tooltip",
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Label(
                        "Temporal decimation (s):", style={**LABEL_STYLES["classic"]}
                    ),
                    dbc.Input(
                        id="decim",  # Unique ID for each input
                        type="number",
                        placeholder="decim",
                        step=1,
                        min=1,
                        max=50,
                        size="sm",
                        persistence=True,
                        persistence_type="session",
                        style={**INPUT_STYLES["small-number"]},
                    ),
                    dbc.Tooltip(
                        """
                Decimation factor used when fitting ICA.

                - Reduces the number of time points by selecting every N-th sample.
                - Helps speed up ICA computation, especially on long recordings.
                - Must be ≥ 1 (1 = no decimation).
                - Typical values: 1 (no decimation), 2, 5, 10.

                Use higher values for faster computation at the cost of temporal precision.
                """,
                        target="decim",
                        placement="right",
                        class_name="custom-tooltip",
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Compute",
                                id="compute-ica-button",
                                color="warning",
                                outline=True,
                                size="sm",
                                n_clicks=0,
                                disabled=False,
                                style=BUTTON_STYLES["big"],
                            ),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                ]
            ),
            dcc.Loading(
                id="ica-loading",
                type="default",
                children=[
                    html.Div(
                        id="ica-status",
                        style={"marginTop": "10px", "marginBottom": "20px"},
                    )
                ],
            ),
            html.Div(
                [
                    html.H6(
                        [html.I(className=f"bi {ICON['ICA']}"), " ICA History"],
                        style={"fontWeight": "bold", "marginBottom": "10px"},
                    ),  # Title for the history section
                    html.Div(
                        id="history-log-ica",  # Dynamic log area
                        style={
                            "height": "200px",  # Adjust the height as needed
                            "overflowY": "auto",  # Scrollable if content exceeds height
                        },
                    ),
                    dbc.Button(
                        "Clean",
                        id="clean-history-button-ica",
                        color="danger",
                        outline=True,
                        size="sm",
                        n_clicks=0,
                        style=BUTTON_STYLES["big"],
                    ),
                ],
                style=BOX_STYLES["classic"],
            ),
        ]
    )


def create_sidebar():
    return html.Div(
        [
            dbc.Tabs(
                [
                    dbc.Tab(
                        create_compute(),
                        labelClassName="bi bi-noise-reduction",
                        tab_id="compute-tab-ica",
                    ),
                    dbc.Tab(
                        create_selection(
                            check_all_annotations_btn_id="check-all-annotations-btn-ica",
                            clear_all_annotations_btn_id="clear-all-annotations-btn-ica",
                            delete_annotations_btn_id="delete-annotations-btn-ica",
                            annotation_checkboxes_id="annotation-checkboxes-ica",
                            delete_confirmation_modal_id="delete-confirmation-modal-ica",
                            delete_modal_body_id="delete-modal-body-ica",
                            cancel_delete_btn_id="cancel-delete-btn-ica",
                            confirm_delete_btn_id="confirm-delete-btn-ica",
                            create_intersection_btn_id="create-intersection-btn-ica",
                            create_intersection_modal_id="create-intersection-modal-ica",
                            create_intersection_modal_body_id="create-intersection-modal-body-ica",
                            intersection_tolerance_id="intersection-tolerance-ica",
                            cancel_intersection_btn_id="cancel-intersection-btn-ica",
                            confirm_intersection_btn_id="confirm-intersection-btn-ica",
                            offset_decrement_id="offset-decrement-ica",
                            offset_display_id="offset-display-ica",
                            offset_increment_id="offset-increment-ica",
                            colors_radio_id="colors-radio-ica",
                            ica_result_radio_id="ica-result-radio",
                        ),
                        labelClassName="bi bi-hand-index-thumb",
                        tab_id="select-tab-ica",
                    ),
                ],
                id="sidebar-tabs-ica",
                persistence=True,
                persistence_type="memory",
                className="custom-sidebar",
            ),
        ],
        style={
            "padding": "0px",
            "height": "90vh",
            "display": "flex",
            "flexDirection": "column",
            "overflowY": "auto",
            "justifyContent": "flex-start",  # Align content at the top
            "width": "250px",  # Sidebar width is now fixed
            "boxSizing": "border-box",
            "fontSize": "12px",
            "borderRadius": "10px",  # Rounded corners for the sidebar itself
        },
    )
