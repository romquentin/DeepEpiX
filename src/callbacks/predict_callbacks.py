import os
import time
import subprocess

import pandas as pd
import dash
from dash import Input, Output, State, html, callback
import dash_bootstrap_components as dbc

import config
from callbacks.utils import annotation_utils as au
from callbacks.utils import history_utils as hu


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
        Register the Dash callback for executing model prediction and sensitivity scripts.

        This function defines a callback that triggers a backend subprocess to run
        machine learning inference and/or SmoothGrad analysis based on the selected
        virtual environment (TensorFlow or PyTorch). It handles input validation,
        caching logic to avoid redundant computations, and updates the application
        state with file paths to the generated results.

        Parameters
        ----------
        n_clicks : int
            Trigger count from the "Run Prediction" button.
        data_path : str
            Path to the input data file/subject directory.
        model_path : str
            Path to the selected model file (.h5, .pth, etc.).
        venv : str
            The selected environment type (e.g., "TensorFlow" or "PyTorch").
        threshold : float
            Confidence threshold for prediction onset.
        sensitivity_analysis : bool
            Flag to determine if SmoothGrad analysis should be executed.
        adjust_onset : bool
            Flag to determine if onset adjustment logic should be applied.
        model_probabilities_store : list
            Store containing paths to existing prediction CSVs.
        sensitivity_analysis_store : list
            Store containing paths to existing SmoothGrad PKL files.
        channel_store : str
            Serialized data representing the selected data channels.

        Returns
        -------
        prediction_status : str or bool
            Success status, error message, or True if successful.
        n_clicks : int
            Resets the button click count to 0.
        display_style : dict
            CSS style dictionary to toggle visibility of the results container.
        model_probabilities_store : list
            Updated list of file paths to prediction results.
        sensitivity_analysis_store : list
            Updated list of file paths to sensitivity analysis results.

        Notes
        -----
        The function uses 'subprocess.run' to execute external Python scripts:
        1. 'main.py': Handles the primary model inference.
        2. 'run_smoothgrad.py': Conducts sensitivity analysis (optional).
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
            cache_dir / f"{os.path.basename(model_path)}_predictions.csv"
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

        signal = extract_preprocess_signal(data_path, freq_data, channels_dict, chunk_limits, excluded_ica)

        clean_variance = signal.var().mean()
        print(f"Variance : {clean_variance}")
        print(f"Signal shape : {signal.shape}")
        ls
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

def extract_preprocess_signal(
    data_path,
    freq_data,
    channels_dict,
    chunk_limits,
    excluded_ica_components=None,
    cache_dir=f"{config.CACHE_DIR}",
):
    """
    Reconstruit le signal complet préprocessé depuis les segments en cache.
    
    Tente de retrouver et concaténer tous les segments Parquet correspondant
    à la configuration (data_path + freq_data + excluded_ica_components).
    Si aucun segment n'est trouvé, déclenche un preprocessing de rattrapage
    sur la plage fallback_time_range.

    Parameters
    ----------
    data_path : str
        Chemin vers le fichier M/EEG source.
    freq_data : dict
        Paramètres de filtrage : 'resample_freq', 'low_pass_freq', etc.
    channels_dict : dict
        Mapping des groupes de canaux vers leurs noms.
    excluded_ica_components : list of int, optional
        Composantes ICA à exclure du signal.
    cache_dir : str, optional
        Répertoire des fichiers Parquet. Défaut : config.CACHE_DIR.
    segment_duration : float, optional
        Durée de chaque segment en secondes. Défaut : 300s.
    fallback_time_range : tuple of float, optional
        (start, end) en secondes à utiliser si aucun cache n'est disponible.

    Returns
    -------
    pd.DataFrame
        Signal complet reconstruit, indexé par temps.
    
    Raises
    ------
    RuntimeError
        Si aucun cache n'est trouvé et qu'aucun fallback n'est fourni.
    """
    import dask.dataframe as dd
    from callbacks.utils import preprocessing_utils as pu

    os.makedirs(cache_dir, exist_ok=True)

    found_segments, missing_segments = _find_cached_segments(
            data_path, freq_data, excluded_ica_components,
            cache_dir, chunk_limits
        )

    if missing_segments:
        print(f"⚠️ {len(missing_segments)} segment(s) manquant(s) — preprocessing ciblé...")
        for chunk in missing_segments:
            pu.get_preprocessed_dataframe_dask(
                data_path=data_path,
                freq_data=freq_data,
                start_time=chunk["start"],
                end_time=chunk["end"],
                channels_dict=channels_dict,
                excluded_ica_components=excluded_ica_components,
                cache_dir=cache_dir,
            )

        # Relecture après preprocessing ciblé
        found_segments, _ = _find_cached_segments(
            data_path, freq_data, excluded_ica_components,
            cache_dir, chunk_limits
        )

    # === 3. Aucun cache → fallback preprocessing ===
    if found_segments:
        ddfs = [dd.read_parquet(seg_path) for seg_path in found_segments]
        return dd.concat(ddfs).compute()


def _find_cached_segments(
    data_path, freq_data, excluded_ica_components,
    cache_dir, chunk_limits 
):
    """
    Retrouve les fichiers cache en utilisant les bornes exactes des segments.
    
    Parameters
    ----------
    chunk_limits : list of [float, float]
        Liste de [start, end] depuis chunk-limits-store.
    """
    from callbacks.utils import preprocessing_utils as pu
    found_segments = []
    missing_segments = []

    for chunk in sorted(chunk_limits, key=lambda x: x[0]):
        start, end = chunk[0], chunk[1]

        candidate = pu.get_cache_filename(
            data_path, freq_data, start, end,
            cache_dir, excluded_ica_components or None
        )

        if os.path.exists(candidate):
            found_segments.append((start, candidate))
        else:
            missing_segments.append(chunk)

    return (
        [path for _, path in found_segments],
        missing_segments
    )




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
