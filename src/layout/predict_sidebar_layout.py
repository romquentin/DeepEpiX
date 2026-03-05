from dash import html, dcc
from layout.config_layout import (
    INPUT_STYLES,
    BUTTON_STYLES,
    LABEL_STYLES,
    FLEXDIRECTION,
    BOX_STYLES
)
import dash_bootstrap_components as dbc
from callbacks.utils import predict_utils as pu


def create_predict():
    layout = (
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Select signal to use for prediction:",
                            style={
                                "fontWeight": "bold",
                                "fontSize": "14px",
                                "marginBottom": "8px",
                            },
                        ),
                        dcc.RadioItems(
                            id="signal-version-predict",
                            options=[{"label": "Filtered signal", "value": "__raw__"}],
                            value="__raw__",
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
                            "Available Models:", style={**LABEL_STYLES["classic"]}
                        ),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=pu.get_model_options(),
                            placeholder="Select ...",
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
                html.Div(
                    [
                        html.Label("Environment:", style={**LABEL_STYLES["classic"]}),
                        dbc.Input(
                            id="venv",
                            type="text",
                            value="",
                            disabled=True,
                            style={**INPUT_STYLES["small-number"]},
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.Label("Threshold:", style={**LABEL_STYLES["classic"]}),
                        dbc.Input(
                            id="initial-threshold",
                            type="number",
                            value=0.5,
                            step=0.01,
                            min=0,
                            max=1,
                            style=INPUT_STYLES["small-number"],
                        ),
                        dbc.Tooltip(
                            "Controls how confident the model must be to detect a spike.",
                            target="threshold",
                            placement="left",
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Sensitivity Analysis (smoothGrad):",
                            style={**LABEL_STYLES["classic"]},
                        ),
                        dbc.RadioItems(
                            id="sensitivity-analysis",
                            options=[
                                {"label": "Yes", "value": True},
                                {"label": "No", "value": False},
                            ],
                            value=False,  # Default selection
                            inline=True,  # Display buttons in a row
                            style={"margin-left": "10px"},
                        ),
                        dbc.Tooltip(
                            "SmoothGrad is an explainability technique that averages the gradients of the loss with respect to the input, over multiple noisy versions of the input. This reduces noise and highlights the regions in the time series that have the greatest influence on the model's predictions. It is available only for simple tensorflow models.",
                            target="sensitivity-analysis",
                            placement="left",
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "Adjust onset (Global Field Power):",
                            style={**LABEL_STYLES["classic"]},
                        ),
                        dbc.RadioItems(
                            id="adjust-onset",
                            options=[
                                {"label": "Yes", "value": True},
                                {"label": "No", "value": False},
                            ],
                            value=True,
                            inline=True,
                            style={"margin-left": "10px"},
                        ),
                        dbc.Tooltip(
                            "Instead of setting the spike prediction to the middle of the window, this method adjusts the onset to the exact timestep where the Global Field Power (GFP) is at its maximum. This ensures more precise alignment with the probable spikes.",
                            target="adjust-onset",
                            placement="left",
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Run Prediction",
                            id="run-prediction-button",
                            color="warning",
                            outline=True,
                            size="sm",
                            n_clicks=0,
                            disabled=False,
                            style=BUTTON_STYLES["big"],
                        ),
                    ]
                ),
                dcc.Loading(
                    id="loading",
                    type="default",
                    children=[
                        html.Div(
                            id="prediction-status",
                            style={"marginTop": "10px", "marginBottom": "20px"},
                        )
                    ],
                ),
                # Prediction Output
                html.Div(
                    id="store-display-div",
                    children=[
                        html.Div(
                            [
                                html.Label(
                                    "Threshold Refinement:",
                                    style={**LABEL_STYLES["classic"]},
                                ),
                                dbc.Input(
                                    id="adjusted-threshold",
                                    type="number",
                                    value=0.5,
                                    step=0.01,
                                    min=0,
                                    max=1,
                                    style=INPUT_STYLES["small-number"],
                                ),
                                dbc.Tooltip(
                                    "Controls how confident the model must be to detect a spike.",
                                    target="threshold",
                                    placement="left",
                                ),
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        html.Div(
                            id="prediction-output",
                            children=[
                                dbc.Tabs(
                                    id="prediction-output-tabs",
                                    active_tab="prediction-output-summary",
                                    children=[
                                        dbc.Tab(
                                            label="Summary",
                                            tab_id="prediction-output-summary",
                                            children=[
                                                html.Div(
                                                    id="prediction-output-summary-div"
                                                )
                                            ],
                                            style={
                                                "margin-top": "10px",
                                                "width": "100%",
                                                "maxHeight": "300px",
                                                "overflowY": "auto",
                                            },
                                        ),
                                        dbc.Tab(
                                            label="Distribution",
                                            tab_id="prediction-output-distribution",
                                            children=[
                                                html.Div(
                                                    id="prediction-output-distribution-div"
                                                )
                                            ],
                                            style={
                                                "margin-top": "10px",
                                                "width": "100%",
                                                "maxHeight": "300px",
                                                "overflowY": "auto",
                                            },
                                        ),
                                        dbc.Tab(
                                            label="Table",
                                            tab_id="prediction-output-table",
                                            children=[
                                                html.Div(
                                                    id="prediction-output-table-div"
                                                )
                                            ],
                                            style={
                                                "margin-top": "10px",
                                                "width": "95%",
                                                "maxHeight": "300px",
                                                "overflowY": "auto",
                                                "overflowX": "hidden",
                                                "maxWidth": "100%",
                                            },
                                        ),
                                    ],
                                    style={
                                        **FLEXDIRECTION["row-tabs"],
                                        "width": "100%",
                                    },
                                ),
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Detected Spike Name:",
                                    style={**LABEL_STYLES["classic"]},
                                ),
                                dbc.Input(
                                    id="model-spike-name",
                                    type="text",
                                    value="detected_spikes_name",
                                    style={**INPUT_STYLES["small-number"]},
                                ),
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        dbc.Button(
                            "Store & Display",
                            id="store-display-button",
                            color="success",
                            outline=True,
                            disabled=False,
                            n_clicks=0,
                            style=BUTTON_STYLES["big"],
                        ),
                    ],
                    style={"display": "none"},
                ),
            ]
        ),
    )
    return layout
