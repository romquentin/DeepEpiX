import os
import time
import subprocess

import pandas as pd
import dash
from dash import Input, Output, State, html, callback
import dash_bootstrap_components as dbc
import json
import hashlib

import config
from callbacks.utils import annotation_utils as au
from callbacks.utils import history_utils as hu
from callbacks.utils import preprocessing_utils as pu
from callbacks.utils import predict_utils as pru


def register_update_selected_model():
    @callback(
        Output("venv", "value"),
        Input("model-dropdown", "value"),
        prevent_initial_call=True,
    )
    def update_selected_model(selected_value):
        """Update the selected model path and detect the environment."""
        if not selected_value:
            return dash.no_update

        # Detect environment
        if selected_value.endswith((".keras", ".h5")):
            environment = "TensorFlow (.tfenv)"
        elif selected_value.endswith((".pth", ".ckpt")):
            environment = "PyTorch (.torchenv)"
        else:
            environment = "Unknown"

        return environment

def register_fill_signal_versions_predict():
    @callback(
        Output("signal-version-predict", "options"),
        Input("history-store", "data"),
        Input("url", "pathname"),
        prevent_initial_call=False,
    )
    def _fill_signal_versions_predict(history_data, url):
        """
        This function populate signal version options for prediction.

        Parameters
        ----------
        history_data : dict or None
            The data dictionary from the 'history-store' component. 
            Expected to have structure: 
            `metadata -> ica_results -> {path: {excluded_components: [...]}}`.
        url : str
            The current page path from the 'url' component.

        Returns
        -------
        list of dict
            A list of option dictionaries for the dcc.Dropdown component.
        """
        options = [{"label": "Filtered signal", "value": "__raw__"}]
        ica_results = (
            (history_data or {})
            .get("metadata", {})
            .get("ica_results", {})
        )
        for ica_path, ica_meta in ica_results.items():
            excluded = ica_meta.get("excluded_components", [])
            if not excluded:
                continue
            options.append({
                "label": f"ICA · {os.path.basename(ica_path)} — excl. {excluded}",
                "value": ica_path,
            })
        return options 

def register_execute_predict_script():
    @callback(
        Output("prediction-status", "children"),
        Output("run-prediction-button", "n_clicks"),
        Output("store-display-div", "style"),
        Output("model-probabilities-store", "data", allow_duplicate=True),
        Output("sensitivity-analysis-store", "data", allow_duplicate=True),
        Input("run-prediction-button", "n_clicks"),
        State("data-path-store", "data"),
        State("frequency-store", "data"),
        State("channel-store", "data"),
        State("chunk-limits-store", "data"),
        State("history-store", "data"),
        State("signal-version-predict", "value"),
        State("model-dropdown", "value"),
        State("venv", "value"),
        State("initial-threshold", "value"),
        State("sensitivity-analysis", "value"),
        State("adjust-onset", "value"),
        State("model-probabilities-store", "data"),
        State("sensitivity-analysis-store", "data"),
        State("channel-store", "data"),
        prevent_initial_call=True,
    )
    def _execute_predict_script(
        n_clicks,
        data_path,
        freq_data,
        channels_dict,
        chunk_limits,
        history_data,
        signal_version,
        model_path,
        venv,
        threshold,
        sensitivity_analysis,
        adjust_onset,
        model_probabilities_store,
        sensitivity_analysis_store,
        channel_store,
    ):
        """
        Execute model inference and sensitivity analysis via external subprocesses.

        This callback reconstructs the signal of interest by aggregating the packets 
        (Parquet files) associated with the previously used time windows in the app 
        or reuses an existing global cache. It then retrieves the MNE metadata (JSON format)
        before launching the prediction and SmoothGrad scripts in the appropriate virtual environment.

        Parameters
        ----------
        n_clicks : int
            The number of times the 'run-prediction-button' has been clicked.
        data_path : str
            File system path to the input dataset or subject directory.
        freq_data : dict
            Frequency-related parameters for signal preprocessing.
        channels_dict : dict
            Mapping of channel names and indices to be used for extraction.
        chunk_limits : list of int
            Start and end indices for data slicing.
        history_data : dict
            Application session history containing metadata.
        signal_version : str
            The specific ICA version to use or '__raw__' for 
            filtered data.
        model_path : str
            Path to the selected model file (e.g., .h5 or .pth).
        venv : str
            The selected environment type (e.g., "TensorFlow" or "PyTorch").
        threshold : float
            Confidence threshold for prediction onset.
        sensitivity_analysis : bool
            Whether to execute the SmoothGrad attribution script.
        adjust_onset : bool
            Whether to apply post-processing logic to adjust predicted onset 
            times.
        model_probabilities_store : list
            Cached paths to existing prediction CSV files.
        sensitivity_analysis_store : list
            Cached paths to existing SmoothGrad pickle files.
        channel_store : str
            Serialized string representation of the selected channels.

        Returns
        -------
        prediction_status : str or bool
            Feedback message for the user, or True if the process completed 
            successfully.
        n_clicks : int
            Resets the button click count (returns 0 or dash.no_update).
        display_style : dict
            CSS style dictionary (e.g., {'display': 'block'}) to show/hide 
            result containers.
        model_probabilities_store : list
            Updated list containing the path to the new prediction CSV.
        sensitivity_analysis_store : list
            Updated list containing the path to the new SmoothGrad .pkl file.

        Notes
        -----
        The function follows a strict execution pipeline:
        1. **Cache Check**: If a prediction with the same model already exists 
           in the store, it returns early.
        2. **Signal Management**: The function checks for the existence of a global cache 
           via an MD5 hash (including data_path, freq_data, and excluded ICA components).
           If absent, it recreates the signal from the preprocessed time windows 
           and saves it in Parquet format.
        3. **MNE Metadata**: Retrieves signal structure information 
           from a JSON file (`_mne_meta.json`).
        4. **Inference**: Runs `main.py` using the `subprocess` module in the 
           environment specified by `venv`.
        5. **Attribution**: If `sensitivity_analysis` is True, runs 
           `run_smoothgrad.py`.
        """
        if not n_clicks or n_clicks == 0:
            return None, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if not data_path:
            error_message = "⚠️ Please choose a subject to display on Home page."
            return (
                error_message,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        missing_fields = []
        if not model_path:
            missing_fields.append("Model")
        if not venv:
            missing_fields.append("Environment")
        if threshold is None:
            missing_fields.append("Threshold")
        if missing_fields:
            error_message = (
                f"⚠️ Please fill in all required fields: {', '.join(missing_fields)}"
            )
            return (
                error_message,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        cache_dir = config.CACHE_DIR
        predictions_csv_path = (
            cache_dir / f"{os.path.basename(model_path)}_predictions.csv"   # Modifier le cache ici pour ajouter info sur signal utilisé
        )
        smoothgrad_path = cache_dir / f"{os.path.basename(model_path)}_smoothGrad.pkl"

        # If already exists, skip execution
        if (
            predictions_csv_path.exists()
            and str(predictions_csv_path) in model_probabilities_store
        ):
            if (
                sensitivity_analysis
                and smoothgrad_path.exists()
                and str(smoothgrad_path) in model_probabilities_store
            ):
                return (
                    "✅ Reusing existing model predictions",
                    0,
                    {"display": "block"},
                    dash.no_update,
                    dash.no_update,
                )
            elif not sensitivity_analysis:
                return (
                    "✅ Reusing existing model predictions",
                    0,
                    {"display": "block"},
                    dash.no_update,
                    dash.no_update,
                )
            
        meta = (history_data or {}).get("metadata", {})
        ica_results = meta.get("ica_results", {})
        if signal_version and signal_version != "__raw__" and signal_version in ica_results:
            excluded_ica = ica_results[signal_version].get("excluded_components", [])
        else:
            excluded_ica = None

        print(f"Datapath : {data_path}")
        print(f"Freq data : {freq_data}")
        print(f"Excluded ica : {excluded_ica}")

        signal_cache_path = os.path.join(
            cache_dir, 
            f"signal_{hashlib.md5(f'{data_path}_{json.dumps(freq_data, sort_keys=True)}_{sorted(excluded_ica) if excluded_ica else []}'.encode()).hexdigest()}.parquet"
        )

        mne_info_path = pu.get_cache_filename(data_path, freq_data).replace(".parquet", "_mne_meta.json")

        if not os.path.exists(signal_cache_path):
            signal = pru.extract_preprocess_signal(
                data_path, freq_data, channels_dict, chunk_limits, excluded_ica
            )
            clean_variance = signal.var().mean()
            print(f"Variance : {clean_variance}")
            print(f"Signal shape : {signal.shape}")
            signal.to_parquet(signal_cache_path)

            print(f"Signal cache in DASH : {signal_cache_path}")
            
        else:
            print(f"✅ Signal cache déjà existant — skip extraction ({signal_cache_path})")

        # Otherwise, execute model
        if "TensorFlow" in venv:
            ACTIVATE_ENV = str(config.TENSORFLOW_ENV / "bin/python")
        elif "PyTorch" in venv:
            ACTIVATE_ENV = str(config.TORCH_ENV / "bin/python")

        command = [
            ACTIVATE_ENV,
            str(config.MODEL_PIPELINE_DIR / "main.py"),
            str(model_path),
            str(venv),
            str(data_path),
            str(cache_dir),
            str(threshold),  # Ensure threshold is passed as a string
            str(adjust_onset),
            str(channel_store),
            str(signal_cache_path),
            str(mne_info_path),
        ]

        working_dir = str(config.APP_ROOT)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(working_dir)

        try:
            start_time = time.time()
            subprocess.run(
                command, env=env, text=True, cwd=str(config.MODEL_PIPELINE_DIR)
            )  # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            model_probabilities_store = [str(predictions_csv_path)]
            print(f"Model testing executed in {time.time()-start_time:.2f} seconds")

        except Exception as e:
            return (
                f"⚠️ Error running model: {e}",
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
            )

        if sensitivity_analysis:
            command = [
                ACTIVATE_ENV,
                str(config.MODEL_PIPELINE_DIR / "run_smoothgrad.py"),
                str(model_path),
                str(venv),
                str(cache_dir),
                str(predictions_csv_path),
                str(threshold),  # Ensure threshold is passed as a string
            ]

            try:
                # Start timing for the second subprocess
                start_time = time.time()
                subprocess.run(
                    command, env=env, text=True, cwd=str(config.MODEL_PIPELINE_DIR)
                )  # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"Smoothgrad executed in {time.time()-start_time:.2f} seconds")

            except Exception as e:
                return (
                    f"⚠️ Error running smoothgrad: {e}",
                    0,
                    {"display": "block"},
                    model_probabilities_store,
                    dash.no_update,
                )

            sensitivity_analysis_store = [str(smoothgrad_path)]
            if not smoothgrad_path.exists():
                return (
                    "⚠️ Error running smoothgrad.",
                    0,
                    {"display": "block"},
                    model_probabilities_store,
                    dash.no_update,
                )
            return (
                True,
                0,
                {"display": "block"},
                model_probabilities_store,
                sensitivity_analysis_store,
            )

        if not predictions_csv_path.exists():
            return (
                "⚠️ Error running model.",
                0,
                {"display": "none"},
                dash.no_update,
                dash.no_update,
            )
        return True, 0, {"display": "block"}, model_probabilities_store, dash.no_update


@callback(
    Output("prediction-output-summary-div", "children"),
    Output("prediction-output-distribution-div", "children"),
    Output("prediction-output-table-div", "children"),
    Input("store-display-div", "style"),
    Input("adjusted-threshold", "value"),
    State("model-probabilities-store", "data"),
    prevent_initial_call=True,
)
def update_prediction_table(style, threshold, prediction_csv_path):
    if style["display"] == "none":
        return None, None, None
    if not prediction_csv_path or threshold is None:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        df = pd.read_csv(prediction_csv_path[0])
        df_filtered = df[df["probas"] > threshold]
        df_filtered["probas"] = df_filtered["probas"].round(2)

        if df_filtered.empty:
            msg = html.P("No events found in this recording.")
            return msg, msg, msg

        table = dbc.Table.from_dataframe(
            df_filtered.copy(),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            dark=False,
        )

        return (
            au.build_table_prediction_statistics(df_filtered, len(df)),
            au.build_prediction_distribution_statistics(df, threshold),
            table,
        )
    except Exception as e:
        error_msg = html.P(f"⚠️ Error loading predictions: {e}")
        return error_msg, error_msg, error_msg


@callback(
    Output("model-spike-name", "value"),
    Input("model-dropdown", "value"),
    Input("adjusted-threshold", "value"),
    prevent_initial_call=True,
)
def update_spike_name(model_path, threshold_value):
    if model_path is None or threshold_value is None:
        return dash.no_update
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    return f"{model_name}_{threshold_value}"


def register_store_display_prediction():
    @callback(
        Output("annotation-store", "data", allow_duplicate=True),
        Output("sidebar-tabs", "active_tab"),
        Output("store-display-div", "style", allow_duplicate=True),
        Output("history-store", "data"),
        Input("store-display-button", "n_clicks"),
        State("annotation-store", "data"),
        State("model-probabilities-store", "data"),
        State("adjusted-threshold", "value"),
        State("model-spike-name", "value"),
        State("history-store", "data"),
        prevent_initial_call=True,
    )
    def store_display_prediction(
        n_clicks,
        annotation_data,
        prediction_csv_path,
        threshold,
        spike_name,
        history_data,
    ):
        if not n_clicks or n_clicks == 0 or prediction_csv_path is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        if not annotation_data:
            annotation_data = []

        df = pd.read_csv(prediction_csv_path[0])
        prediction_df = df[df["probas"] > threshold]
        new_annotations = prediction_df[["onset", "duration"]].copy()
        new_annotations["description"] = spike_name  # Set spike name as description
        new_annotations_dict = new_annotations.to_dict(orient="records")
        annotation_data.extend(new_annotations_dict)

        action = f"Tested model with <{spike_name}> as the predicted event name.\n"
        history_data = hu.fill_history_data(history_data, "models", action)

        # Return updated annotations and switch tab
        return annotation_data, "selection-tab", {"display": "none"}, history_data
