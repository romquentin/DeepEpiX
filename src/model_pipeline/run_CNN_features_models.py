"""
Author: Agnès GUINARD
Created: 2025-09-30
Description: Testing Pipeline for model_CNN.keras & model_features_only.keras (trained by Pauline Mouchès)
"""

import os
import gc
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from model_pipeline.features_utils import get_win_data_feat
from model_pipeline.sliding_windows_utils import (
    save_data_matrices,
    create_windows,
    generate_database,
    get_win_data_signal,
)
from model_pipeline.utils import (
    read_raw,
    load_obj,
    compute_gfp,
    find_peak_gfp,
    load_raw_from_parquet,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# === Helper functions specific to models ===
def prepare_data(
    signal_cache_path,
    mne_info_cache_path,
    data_path,
    preprocessing_option,
    output_path,
    channel_groups,
    sfreq,
    window_size,
    spike_spacing_from_borders,
):
    """Prepare data matrices, create windows, and generate test IDs."""
    with open("good_channels_dict.pkl", "rb") as f:
        good_channels = pickle.load(f)

    if preprocessing_option == 'custom':
        raw, metadata = load_raw_from_parquet(signal_cache_path, mne_info_cache_path)

        sfreq_orig = metadata['sfreq']

        if sfreq_orig != sfreq:
            raw.resample(sfreq)
    
    else:
        raw = read_raw(
            data_path,
            preload=True,
            verbose=False,
            bad_channels=channel_groups.get("bad", []),
        )
        raw.filter(0.5, 50, n_jobs=8)
        raw.resample(sfreq)

    save_data_matrices(
        raw,             #type: ignore
        output_path,
        channel_groups,
        good_channels,
        "mag",
    )

    total_nb_windows = create_windows(
        output_path, window_size, False, sfreq, spike_spacing_from_borders
    )

    return generate_database(total_nb_windows)


def load_model(model_name):
    """Load and compile a Keras model."""
    model = keras.models.load_model(model_name, compile=False)
    model.compile()
    print(model.summary())
    return model


def predict_windows(model, X_test_ids, model_name, output_path, dim):
    """Predict probabilities for all windows using the given model."""
    file_path = os.path.join(output_path, "data_raw_windows_bi")
    with open(file_path, "rb") as f:
        y_pred_probas = []

        device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        with tf.device(device):
            for cur_win in X_test_ids:
                sample = get_win_data_signal(f, cur_win, dim)

                if "features" in model_name:
                    sample = get_win_data_feat(sample)

                y_pred_probas.append(model(sample).numpy()[0][0])
                del sample

    return y_pred_probas


def get_adjusted_onsets(X_test_ids, output_path, dim, sfreq):
    """Adjust onset times based on GFP peaks."""
    y_timing_data = load_obj("data_raw_timing.pkl", output_path)
    onsets = []

    for win in X_test_ids:
        cur_win = X_test_ids[win]
        window = get_win_data_signal(
            open(
                os.path.join(
                    output_path,
                    "data_raw_windows_bi",
                ),
                "rb",
            ),
            cur_win,
            dim,
        ).squeeze()

        gfp = compute_gfp(window.T)
        times = np.linspace(0, window.shape[0] / sfreq, window.shape[0])
        peak_time = find_peak_gfp(gfp, times)

        onset = ((y_timing_data[win] - window.shape[0] / 2) / sfreq) + peak_time
        onsets.append(round(onset, 3))

    return onsets


def get_onsets(output_path, sfreq):
    """Get raw timing data."""
    y_timing_data = load_obj("data_raw_timing.pkl", output_path)
    onsets = (y_timing_data / sfreq).round(3).tolist()
    return onsets


def save_predictions(output_path, signal_name, model_name, onsets, y_pred_probas):
    """Save predictions into a CSV file compatible with MNE annotations."""
    df = pd.DataFrame(
        {
            "onset": onsets,
            "duration": 0,
            "probas": y_pred_probas,
        }
    )
    if signal_name is not None:
        output_file = os.path.join(
            output_path, f"{os.path.basename(model_name)}_{signal_name}_predictions.csv"
        )
    else:
        output_file = os.path.join(
            output_path, f"{os.path.basename(model_name)}_predictions.csv"
        )
    df.to_csv(output_file, index=False)
    return output_file


# === Main function ===
def test_model(
    model_name,
    output_path,
    signal_cache_path,
    mne_info_cache_path,
    data_path,
    preprocessing_option,
    adjust_onset=True,
    channel_groups=None,
    signal_name=None,
):
    """Run the full pipeline: prepare data, predict, adjust onsets, and save results."""
    # Params
    window_size = 0.2
    sfreq_model = 150
    dim = (int(sfreq_model * window_size), 275, 1)
    spike_spacing_from_borders = 0.03

    # 1. Data preparation
    X_test_ids = prepare_data(
        signal_cache_path,
        mne_info_cache_path,
        data_path,
        preprocessing_option,
        output_path,
        channel_groups,
        sfreq_model,
        window_size,
        spike_spacing_from_borders,
    )

    # 2. Load model
    model = load_model(model_name)

    # 3. Predictions
    y_pred_probas = predict_windows(model, X_test_ids, model_name, output_path, dim)

    # 4. Cleanup model & GPU memory
    del model
    gc.collect()
    keras.backend.clear_session()

    # 5. Adjust onset times
    if adjust_onset:
        onsets = get_adjusted_onsets(X_test_ids, output_path, dim, sfreq_model)
    else:
        onsets = get_onsets(output_path, sfreq_model)

    # 6. Save predictions
    return save_predictions(output_path, signal_name, model_name, onsets, y_pred_probas)
