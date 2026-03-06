import os.path as op
from pathlib import Path
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
import mne
import json
import pandas as pd


# read and write pickle files
def save_obj(obj, name, path):
    with open(op.join(path, name + ".pkl"), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path):
    with open(op.join(path, name), "rb") as f:
        return pickle.load(f)


# center scale each window using all window mean and std
def standardize(X, mean=False, std=False):
    if not mean:
        mean = np.mean(X, axis=(1, 2))
    if not std:
        std = np.std(X, axis=(1, 2))
    X_stand = np.zeros(X.shape)
    nb_data = X.shape[0]
    for i in range(0, nb_data, 1):
        X_stand[i, :, :] = (X[i, :, :] - mean[i]) / std[i]
    return X_stand


# read raw from different acquisition systems
def read_raw(data_path, preload, verbose, bad_channels=None):
    data_path = Path(data_path)

    if data_path.suffix == ".ds":
        raw = mne.io.read_raw_ctf(str(data_path), preload=preload, verbose=verbose)

    elif data_path.suffix == ".fif":
        raw = mne.io.read_raw_fif(str(data_path), preload=preload, verbose=verbose)

    elif data_path.is_dir():
        # Assume BTi/4D format: folder must contain 3 specific files
        files = list(data_path.glob("*"))
        # Try to identify the correct files by names
        raw_fname = next(
            (f for f in files if "rfDC" in f.name and f.suffix == ""), None
        )
        config_fname = next((f for f in files if "config" in f.name.lower()), None)
        hs_fname = next((f for f in files if "hs" in f.name.lower()), None)

        if not all([raw_fname, config_fname, hs_fname]):
            raise ValueError(
                "Could not identify raw, config, or hs file in BTi folder."
            )

        raw = mne.io.read_raw_bti(
            pdf_fname=str(raw_fname),
            config_fname=str(config_fname),
            head_shape_fname=str(hs_fname),
            preload=preload,
            verbose=verbose,
        )

    else:
        raise ValueError("Unrecognized file or folder type for MEG data.")

    if bad_channels:
        raw.drop_channels(bad_channels)

    return raw

def load_raw_from_parquet(parquet_path: str, json_path: str) -> mne.io.RawArray:
    """
    Reconstruct an MNE RawArray using .parquet & .jsons files.

    Parquet file must contain channels as columns & samples as rows.
    Json file must contain : sfreq, ch_names, ch_types & bads.

    Parameters
    ----------
    parquet_path: str
        Path to the .parquet file containing signal values.
    json_path: str
        Path to the .json file containing mne metadata.

    Returns
    -------
    mne.io.RawArray
        MNE instance of the reconstructed signal.
    metadata: dict
        Dictionnary containing signal metadatas.
    """
    df = pd.read_parquet(parquet_path)
    data = df.values.T  # shape: (n_channels, n_times)

    with open(json_path, "r") as f:
        metadata = json.load(f)

    ch_names = metadata["ch_names"]
    channel_types_dict = metadata["channel_types"]
    ch_types = [channel_types_dict[ch] for ch in ch_names]

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=metadata["sfreq"],
        ch_types=ch_types,
    )
    info["bads"] = metadata.get("bads", [])

    return mne.io.RawArray(data, info, verbose=False), metadata


# tentative function to interpolate missing channels using mne
def interpolate_missing_channels(raw, good_channels):

    def get_base(name):
        return name.split()[0].split("-")[0].strip()

    raw.rename_channels(get_base)
    existing_channels = raw.info["ch_names"]
    good_basenames = [name for name in good_channels.keys()]

    # Figure out missing channels by base name
    missing_basenames = list(set(good_basenames) - set(existing_channels))
    new_raw = raw.copy()

    # Create fake channels for missing ones
    for miss in missing_basenames:

        to_copy = raw.info["ch_names"][71]  # just an existing template channel
        new_channel = raw.copy().pick([to_copy])
        new_channel.rename_channels({to_copy: miss})
        new_raw.add_channels([new_channel], force_update_info=True)

        # specifies the location of the missing channel
        for i in range(len(new_raw.info["chs"])):
            if new_raw.info["chs"][i]["ch_name"] == miss:
                new_raw.info["chs"][i]["loc"] = good_channels[miss]

    # Reorder based on full good_channels list (with no suffixes now)
    new_raw.reorder_channels(good_basenames)
    new_raw.info["bads"] = missing_basenames

    # Interpolate
    new_raw.interpolate_bads(origin=(0, 0, 0.04), reset_bads=True)
    return new_raw


def fill_missing_channels(raw, target_channel_count):
    """
    Fills missing channels by duplicating existing channels at regular intervals
    and inserting them next to the originals they are copied from.

    Parameters:
    - raw (mne.io.Raw): The original raw object.
    - target_channel_count (int): Desired total number of channels.

    Returns:
    - numpy.ndarray: Data with inserted channels (shape: target_channel_count, n_times).
    """
    data = raw.get_data()
    current_count = data.shape[0]

    if current_count >= target_channel_count:
        return data  # Nothing to add

    n_missing = target_channel_count - current_count

    # Get evenly spaced indices from the existing channels to duplicate
    duplicate_indices = np.linspace(0, current_count - 1, n_missing, dtype=int)

    new_data = []

    for i in range(current_count):
        new_data.append(data[i])  # Original channel
        if i in duplicate_indices:
            new_data.append(data[i])  # Insert duplicate right after

    full_data = np.stack(new_data, axis=0)
    return full_data


def compute_gfp(window):
    """Compute Global Field Power (GFP) as standard deviation across channels."""
    return np.std(window, axis=0)


def find_peak_gfp(gfp, times, smoothing_sigma=2, percentile=90):
    """Find the peak GFP time within a window after smoothing."""
    gfp_smooth = gaussian_filter1d(gfp, sigma=smoothing_sigma)

    # Thresholding: Find first peak above percentile threshold
    threshold = np.percentile(gfp_smooth, percentile)
    peak_indices = np.where(gfp_smooth >= threshold)[0]

    if len(peak_indices) > 0:
        return times[peak_indices[0]]  # First peak above threshold
    else:
        return times[np.argmax(gfp_smooth)]  # Default to max if no peak found
