from dash import html, dcc
import dash_bootstrap_components as dbc
from layout.config_layout import DEFAULT_FIG_LAYOUT
import config


def create_graph_container(
    update_button_id="update-button",
    page_selector_id="page-selector",
    next_spike_buttons_container_id="next-spike-buttons-container",
    prev_spike_id="prev-spike",
    next_spike_id="next-spike",
    annotation_dropdown_id="annotation-dropdown",
    loading_id="loading-graph",
    signal_graph_id="signal-graph",
    annotation_graph_id="annotation-graph",
):
    return html.Div(
        [
            # Update Button
            html.Div(
                style={
                    "position": "absolute",
                    "top": "10px",
                    "left": "20px",
                    "height": "30px",
                    "backgroundColor": "rgba(0, 0, 0, 0.5)",
                    "borderRadius": "2px",
                    "zIndex": "1000",
                    "padding": "2px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
                children=[
                    dbc.Button(
                        html.I(
                            className="bi bi-arrow-clockwise", style={"color": "white"}
                        ),
                        id=update_button_id,
                        n_clicks=0,
                        outline=True,
                        style={
                            "width": "30px",
                            "height": "20px",
                            "padding": "0",
                            "border": "none",
                            "fontSize": "1rem",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                        },
                    ),
                    dbc.Tooltip(
                        "Refresh the graph after modifying the display parameters",
                        target=update_button_id,
                        placement="bottom",
                    ),
                    dbc.RadioItems(
                        id=page_selector_id,
                        options=[],
                        value=0,
                        inline=True,
                        labelClassName="page-radio-label",
                        inputClassName="page-radio-input",
                        style={
                            "display": "flex",
                            "gap": "1px",  # Reduce space between buttons
                            "alignItems": "center",
                        },
                    ),
                    dbc.Tooltip(
                        "The graph is divided in multiple page.",
                        target=page_selector_id,
                        placement="bottom",
                    ),
                    html.Div(
                        id=next_spike_buttons_container_id,
                        style={"display": "flex", "alignItems": "center"},
                        children=[
                            dbc.Button(
                                html.I(
                                    className="bi bi-arrow-left-circle",
                                    style={"color": "white"},
                                ),
                                id=prev_spike_id,
                                n_clicks=0,
                                outline=True,
                            ),
                            dbc.Tooltip(
                                "Move graph to previous event",
                                target=prev_spike_id,
                                placement="bottom",
                            ),
                            dcc.Dropdown(
                                id=annotation_dropdown_id,
                                value="__ALL__",
                                persistence=True,
                                persistence_type="session",
                                clearable=False,
                                className="custom-dropdown",
                                style={
                                    "width": "100px",
                                    "height": "25px",
                                    "fontSize": "10px",
                                    "backgroundColor": "rgb(175, 175, 175, 0.4)",
                                    "border": "none",
                                    "color": "white",
                                },
                            ),
                            dbc.Button(
                                html.I(
                                    className="bi bi-arrow-right-circle",
                                    style={"color": "white"},
                                ),
                                id=next_spike_id,
                                n_clicks=0,
                                outline=True,
                            ),
                            dbc.Tooltip(
                                "Move graph to next event",
                                target=next_spike_id,
                                placement="bottom",
                            ),
                        ],
                    ),
                ],
            ),
            # Prev / Next Spike Buttons
            html.Div(
                id="cursor",
                style={
                    "position": "absolute",
                    "top": "10px",
                    "left": "49.7%",
                    "z-index": "500",
                    "opacity": 0.8,
                    "display": "flex",
                },
                children=html.Span(
                    html.I(className="bi bi-caret-down-fill"),
                ),
            ),
            dcc.Loading(
                id=loading_id,
                type="circle",
                children=[
                    html.Div(
                        children=[
                            dcc.Graph(
                                id=signal_graph_id,
                                figure={
                                    "data": [],
                                    "layout": {
                                        **DEFAULT_FIG_LAYOUT,
                                        "xaxis": {
                                            **DEFAULT_FIG_LAYOUT.get("xaxis", {}), #type: ignore
                                            "range": [0, 20],
                                            "minallowed": 0,
                                            "maxallowed": config.CHUNK_RECORDING_DURATION,
                                        },
                                        "yaxis": {
                                            **DEFAULT_FIG_LAYOUT.get("yaxis", {}), #type: ignore
                                        },
                                        "title": {
                                            **DEFAULT_FIG_LAYOUT.get("title", {}), #type: ignore
                                        },
                                    },
                                },
                                config={
                                    "responsive": True,
                                    "doubleClick": "autosize",
                                    "displayModeBar": True,
                                    "modeBarButtonsToRemove": ["resetScale2d"],
                                },
                            ),
                        ],
                        id="graph-container",
                        style={
                            "width": "100%",
                            "height": "82vh",
                            "borderRadius": "10px",
                            "overflowX": "hidden",
                            "overflowY": "scroll",
                            "boxShadow": "none",
                        },
                    ),
                ],
                overlay_style={"pointerEvents": "none", "visibility": "visible"},
            ),
            dcc.Graph(
                id=annotation_graph_id,
                figure={
                    "data": [],
                    "layout": {
                        "xaxis": {
                            "title": "",
                            "showgrid": False,
                            "zeroline": False,
                            "showticklabels": False,
                        },
                        "yaxis": {
                            "title": "",
                            "showgrid": False,
                            "zeroline": False,
                            "tickvals": [0],
                            "tickfont": {"color": "rgba(0, 0, 0, 0)"},
                            "range": [0, 1],
                        },
                        "paper_bgcolor": "rgba(0,0,0,0)",
                        "margin": {"l": 0, "r": 0, "t": 0, "b": 10},
                    },
                },
                config={"staticPlot": True},
                style={
                    "width": "100%",
                    "height": "2vh",
                    "pointerEvents": "none",
                    "borderRadius": "10px",
                    "overflow": "hidden",
                    "backgroundColor": "transparent",
                    "boxShadow": "none",
                },
            ),
        ],
        style={
            "position": "relative",
            "display": "flex",
            "flexDirection": "column",
            "width": "100%",
            "height": "100%",
        },
    )
