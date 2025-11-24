# Dash & Plotly
from dash import Input, Output, State, callback

# Local Imports
from callbacks.utils import preprocessing_utils as pu


def register_display_psd():
    @callback(
        Output("psd-status", "children"),
        Input("compute-display-psd-button", "n_clicks"),
        State("data-path-store", "data"),
        State("resample-freq", "value"),
        State("high-pass-freq", "value"),
        State("low-pass-freq", "value"),
        State("notch-freq", "value"),
        State("theme-store", "data"),
        running=[(Output("compute-display-psd-button", "disabled"), True, False)],
        prevent_initial_call=True,
    )
    def display_psd(
        n_clicks,
        data_path,
        resample_freq,
        high_pass_freq,
        low_pass_freq,
        notch_freq,
        theme,
    ):
        """Compute and display power spectrum decomposition depending on the frequency parameters stored."""
        if None in (
            data_path,
            resample_freq,
            high_pass_freq,
            low_pass_freq,
        ):
            return "⚠️ Please fill in high and low frequency parameters."

        if n_clicks > 0:
            freq_data = {
                "resample_freq": resample_freq,
                "low_pass_freq": low_pass_freq,
                "high_pass_freq": high_pass_freq,
                "notch_freq": notch_freq,
            }
            return pu.compute_power_spectrum_decomposition(data_path, freq_data, theme)
