
"""
Model Pipeline Template
Author: Your Name
Created: Date
Description: Standard template for running model pipelines with test_model.
"""

# === Import ===
import os
import pandas as pd
import gc
from tensorflow import keras

from model_pipeline.utils import load_raw_from_parquet


# === Helper functions specific to your model ===
def prepare_data(raw):
    """
    Prepare dataset for the model.
    Replace with your actual preprocessing pipeline.
    """
    # TODO


def load_model(model_name):
    """
    Load a model from disk.
    Args:
        model_name: Path to the trained model file.
    Returns:
        Compiled model ready for inference.
    """
    # TODO


def predict_windows(model, X_test_ids, model_name, output_path) -> list[float]:
    """
    Run predictions on the test dataset.
    Args:
        ?
    Returns:
        List of predicted probabilities.
    """
    # TODO


def get_adjusted_onsets(X_test_ids, output_path) -> list[float]:
    """
    Compute adjusted onsets (e.g. aligning to GFP peaks).
    Args:
        ?
    Returns:
        List of adjusted onsets.
    """
    # TODO


def get_onsets(output_path):
    """
    Compute raw timing windows.
    Args:
        ?
    Returns:
        List of timing windows.
    """
    # TODO


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
    adjust_onset=True,
    channel_groups=None,
    signal_name=None,
):
    """Run the full pipeline: prepare data, predict, adjust onsets, and save results."""
    raw, metadata = load_raw_from_parquet(signal_cache_path, mne_info_cache_path)

    # 1. Data preparation
    X_test_ids = prepare_data(raw)

    # 2. Load model
    model = load_model(model_name)

    # 3. Predictions
    y_pred_probas = predict_windows(model, X_test_ids, model_name, output_path)

    # 4. Cleanup model & GPU memory
    del model
    gc.collect()
    keras.backend.clear_session()

    # 5. Adjust onset times
    if adjust_onset:
        onsets = get_adjusted_onsets(X_test_ids, y_pred_probas)
    else:
        onsets = get_onsets(output_path)

    # 6. Save predictions
    return save_predictions(output_path, signal_name, model_name, onsets, y_pred_probas)


