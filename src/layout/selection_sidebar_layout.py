from dash import html, dcc
import dash_bootstrap_components as dbc
from layout.config_layout import BOX_STYLES, BUTTON_STYLES, INPUT_STYLES


def create_selection(
    check_all_annotations_btn_id,
    clear_all_annotations_btn_id,
    delete_annotations_btn_id,
    annotation_checkboxes_id,
    delete_confirmation_modal_id,
    delete_modal_body_id,
    cancel_delete_btn_id,
    confirm_delete_btn_id,
    create_intersection_btn_id,
    create_intersection_modal_id,
    create_intersection_modal_body_id,
    intersection_tolerance_id,
    cancel_intersection_btn_id,
    confirm_intersection_btn_id,
    offset_decrement_id,
    offset_display_id,
    offset_increment_id,
    colors_radio_id,
    montage_radio_id=None,
    check_all_button_id=None,
    clear_all_button_id=None,
    channel_region_checkboxes_id=None,
    ica_result_radio_id=None,
):

    layout = []

    if montage_radio_id:
        layout.extend(
            [
                html.Div(
                    [
                        html.Label(
                            "Select Montage:",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "14px",
                                "marginBottom": "8px",
                            },
                        ),
                        dcc.RadioItems(
                            id=montage_radio_id,
                            options=[],
                            inline=False,
                            style={"margin": "10px 0", "fontSize": "12px"},
                            persistence=True,
                            persistence_type="session",
                        ),
                    ],
                    style=BOX_STYLES["classic"],
                ),
                html.Div(
                    [
                        html.Label(
                            "Select Channels:",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "14px",
                                "marginBottom": "8px",
                            },
                        ),
                        html.Div(
                            [
                                dbc.Button(
                                    "Check All",
                                    id=check_all_button_id,
                                    color="success",
                                    outline=True,
                                    size="sm",
                                    n_clicks=0,
                                    style=BUTTON_STYLES["tiny"],
                                ),
                                dbc.Button(
                                    "Clear All",
                                    id=clear_all_button_id,
                                    color="warning",
                                    outline=True,
                                    size="sm",
                                    n_clicks=0,
                                    style=BUTTON_STYLES["tiny"],
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "gap": "4%",
                            },
                        ),
                        dcc.Checklist(
                            id=channel_region_checkboxes_id,
                            options=[],
                            value=[],
                            inline=False,
                            style={
                                "margin": "10px 0",
                                "fontSize": "12px",
                                "borderRadius": "5px",
                                "padding": "8px",
                                "border": "1px solid #ddd",
                            },
                            persistence=True,
                            persistence_type="session",
                        ),
                    ],
                    style=BOX_STYLES["classic"],
                ),
            ]
        )

    if ica_result_radio_id:
        layout.extend(
            [
                html.Div(
                    [
                        html.Label(
                            "Select ICA Result:",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "14px",
                                "marginBottom": "8px",
                            },
                        ),
                        dcc.RadioItems(
                            id=ica_result_radio_id,
                            options=[],
                            inline=False,
                            style={"margin": "10px 0", "fontSize": "12px"},
                            persistence=True,
                            persistence_type="session",
                        ),
                    ],
                    style=BOX_STYLES["classic"],
                ),
                html.Div(
                [
                    html.Label(
                        "Select ICA components to remove (last computed):",
                        style={
                            "fontWeight": "bold",
                            "fontSize": "14px",
                            "marginBottom": "8px",
                        },
                    ),
                    dcc.Checklist(
                        id="ica-components-selection",
                        options=[], 
                        value=[],
                        inline=False,
                        persistence=True,
                        persistence_type="session",
                    ),
                    dbc.Button(
                        "Apply Exclusion",
                        id="apply-ica-exclusion-button",
                        color="danger",
                        outline=True,
                        size="sm",
                        n_clicks=0,
                        style=BUTTON_STYLES["big"],
                    ),
                    html.Div(id="exclusion-status", style={"marginTop": "10px"})
                ],
                style=BOX_STYLES["classic"],
                ),
            ]
        )

    layout.extend(
        [
            html.Div(
                [
                    html.Label(
                        "Select Annotations:",
                        style={
                            "fontWeight": "bold",
                            "fontSize": "14px",
                            "marginBottom": "8px",
                        },
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Check All",
                                id=check_all_annotations_btn_id,
                                color="success",
                                outline=True,
                                size="sm",
                                n_clicks=0,
                                style=BUTTON_STYLES["tiny"],
                            ),
                            dbc.Button(
                                "Clear All",
                                id=clear_all_annotations_btn_id,
                                color="warning",
                                outline=True,
                                size="sm",
                                n_clicks=0,
                                style=BUTTON_STYLES["tiny"],
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "gap": "4%",
                            "marginBottom": "6px",
                        },
                    ),
                    dcc.Checklist(
                        id=annotation_checkboxes_id,
                        inline=False,
                        style={"margin": "10px 0", "fontSize": "12px"},
                        persistence=True,
                        persistence_type="session",
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Create ∩",
                                id=create_intersection_btn_id,
                                color="warning",
                                outline=True,
                                size="sm",
                                n_clicks=0,
                                style=BUTTON_STYLES["tiny"],
                            ),
                            dbc.Button(
                                "Delete Selected",
                                id=delete_annotations_btn_id,
                                color="danger",
                                outline=True,
                                size="sm",
                                n_clicks=0,
                                style=BUTTON_STYLES["tiny"],
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "gap": "4%",
                            "marginBottom": "6px",
                        },
                    ),
                    dbc.Modal(
                        [
                            dbc.ModalHeader("Confirm Deletion"),
                            dbc.ModalBody(
                                id=delete_modal_body_id,
                                children="Are you sure you want to delete the selected annotations?",
                            ),
                            dbc.ModalFooter(
                                [
                                    dbc.Button(
                                        "Cancel",
                                        id=cancel_delete_btn_id,
                                        color="secondary",
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Delete",
                                        id=confirm_delete_btn_id,
                                        color="danger",
                                        n_clicks=0,
                                    ),
                                ]
                            ),
                        ],
                        id=delete_confirmation_modal_id,
                        is_open=False,
                    ),
                    dbc.Modal(
                        [
                            dbc.ModalHeader("Create New Annotation from Intersection"),
                            dbc.ModalBody(id=create_intersection_modal_body_id),
                            html.Label(
                                "Select Tolerance:",
                                style={
                                    "fontWeight": "bold",
                                    "fontSize": "14px",
                                    "marginBottom": "8px",
                                    "padding": "20px",
                                },
                            ),
                            dbc.Input(
                                id=intersection_tolerance_id,  # Unique ID for each input
                                type="number",
                                placeholder="Tolerance (s) ...",
                                step=0.01,
                                min=0,
                                max=0.300,
                                size="sm",
                                value=0.100,
                                persistence=True,
                                persistence_type="session",
                                style={
                                    **INPUT_STYLES["small-number"],
                                    "width": "30%",
                                    "marginLeft": "50px",
                                },
                            ),
                            dbc.ModalFooter(
                                [
                                    dbc.Button(
                                        "Cancel",
                                        id=cancel_intersection_btn_id,
                                        color="secondary",
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Create",
                                        id=confirm_intersection_btn_id,
                                        color="primary",
                                        n_clicks=0,
                                    ),
                                ]
                            ),
                        ],
                        id=create_intersection_modal_id,
                        is_open=False,
                    ),
                ],
                style=BOX_STYLES["classic"],
            ),
            html.Div(
                [
                    html.Label(
                        "Select Amplitude:",
                        style={
                            "fontWeight": "bold",
                            "fontSize": "14px",
                            "marginBottom": "8px",
                        },
                    ),
                    html.Div(
                        children=[
                            dbc.Button(
                                "-",
                                id=offset_decrement_id,
                                color="primary",
                                size="sm",
                                n_clicks=0,
                            ),
                            dbc.Input(
                                id=offset_display_id,
                                type="number",
                                value=5,
                                persistence=True,
                                persistence_type="session",
                                style={
                                    "width": "60px",
                                    "fontWeight": "bold",
                                    "fontSize": "12px",
                                },
                            ),
                            dbc.Button(
                                "+",
                                id=offset_increment_id,
                                color="primary",
                                size="sm",
                                n_clicks=0,
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "gap": "12px",
                        },
                    ),
                ],
                style=BOX_STYLES["classic"],
            ),
            html.Div(
                [
                    html.Label(
                        "Select Colors:",
                        style={
                            "fontWeight": "bold",
                            "fontSize": "14px",
                            "marginBottom": "8px",
                        },
                    ),
                    dcc.RadioItems(
                        id=colors_radio_id,
                        options=[
                            {"label": "blue", "value": "blue"},
                            {"label": "white", "value": "white"},
                            {"label": "rainbow", "value": "rainbow"},
                        ],
                        value="rainbow",
                        inline=False,
                        style={"margin": "10px 0", "fontSize": "12px"},
                        persistence=True,
                        persistence_type="session",
                    ),
                ],
                style=BOX_STYLES["classic"],
            ),
        ]
    )

    return html.Div(layout)
