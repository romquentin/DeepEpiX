import dash
from dash import html

dash.register_page(__name__, name="Help", path="/settings/help")

layout = html.Div(
    [
        html.Iframe(
            src="https://MEL-Eduwell-lab.github.io/deepepix-docs/",
            style={"width": "100%", "height": "1600px", "border": "none"},
        )
    ]
)
