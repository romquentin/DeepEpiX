from dash import html, dcc
import dash_bootstrap_components as dbc

from layout.config_layout import BOX_STYLES, BUTTON_STYLES, LABEL_STYLES


def create_save():
    layout = html.Div(
        [
            html.Div(
                [
                    html.Label(
                        "Select Annotations:", style={**LABEL_STYLES["classic"]}
                    ),
                    html.Div(
                        [
                            dbc.Button(
                                "Check All",
                                id="check-all-annotations-to-save-btn",
                                color="success",
                                outline=True,
                                size="sm",
                                n_clicks=0,
                                style=BUTTON_STYLES["tiny"],
                            ),
                            dbc.Button(
                                "Clear All",
                                id="clear-all-annotations-to-save-btn",
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
                        id="annotations-to-save-checkboxes",
                        inline=False,
                        style={"margin": "10px 0", "fontSize": "12px"},
                        persistence=True,
                        persistence_type="memory",
                    ),
                ],
                style=BOX_STYLES["classic"],
            ),
            html.Div(
                [
                    html.Label(
                        "Select Bad Channels:", style={**LABEL_STYLES["classic"]}
                    ),
                    dcc.Checklist(
                        id="bad-channels-to-save-checkboxes",
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
                        html.Span(
                            [
                                "Select Saving Format: ",
                                html.I(
                                    className="bi bi-info-circle-fill",
                                    id="help-saving-format",
                                ),
                            ]
                        ),
                        style={**LABEL_STYLES["classic"]},
                    ),
                    html.Div(
                        [
                            dbc.RadioItems(
                                id="saving-format-radio",
                                options=[
                                    {
                                        "label": "Original",
                                        "value": "original",
                                        "id": "radio-original",
                                    },
                                    {
                                        "label": "FIF (recommended for full metadata)",
                                        "value": "fif",
                                        "id": "radio-fif",
                                    },
                                    {
                                        "label": "CSV (save annotations only)",
                                        "value": "csv",
                                        "id": "radio-csv",
                                    },
                                ],
                                value="fif",
                                inline=False,
                                labelStyle={
                                    "display": "block",
                                    "margin-bottom": "0.5em",
                                },
                            ),
                            dbc.Tooltip(
                                """
                    - If the original format is .ds, only the new markerfile is saved; bad channels cannot be saved. By default, the old marker file is renamed to 'OldMarkerFile_{date}.mrk', and the new one is saved as 'MarkerFile.mrk' in the subject folder to ensure backward compatibility.\n

                    - Use the FIF format to include annotations and bad channels. This is recommended for preserving full metadata. By default, keeps a trace of the old .fif file renamed to {original_name}_{date}.fif.

                    - Used the CSV format to save the annotations so they can be easily reused.
                    """,
                                target="help-saving-format",
                                placement="left",
                                class_name="custom-tooltip",
                            ),
                        ]
                    ),
                    dbc.Button(
                        "Save",
                        id="save-annotation-button",
                        color="warning",
                        outline=True,
                        size="sm",
                        n_clicks=0,
                        disabled=False,
                        style=BUTTON_STYLES["big"],
                    ),
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=[
                            html.Div(
                                id="saving-mrk-status", style={"margin-top": "10px"}
                            )
                        ],
                    ),
                ]
            ),
        ]
    )

    return layout
