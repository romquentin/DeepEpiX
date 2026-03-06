from pathlib import Path
from tqdm import tqdm
import numpy as np
import mne
from model_pipeline.utils import (
    interpolate_missing_channels,
    fill_missing_channels,
    save_obj,
    load_obj,
    standardize,
)

def save_data_matrices(
    raw: mne.io.RawArray,
    output_dir: str,
    channel_groups: dict,
    good_channels: dict,
    channel_type: str = "mag",
) -> None:
    """
    Extract MEG/EEG data from a RawArray and save it as a pickle file.

    Parameters
    ----------
    raw : mne.io.RawArray
        Preprocessed MNE Raw object.
    output_dir : str
        Directory where processed data will be stored.
    channel_groups : dict
        Channel groups mapping group names to channel lists.
        Must include a "bad" key if bad channels are present.
    good_channels : dict
        Reference channel layout mapping base names to sensor locations.
    channel_type : str, optional
        Channel type to process, either "mag" or "eeg". Defaults to "mag".

    Raises
    ------
    ValueError
        If channel_type is not "mag" or "eeg".
    """
    output_dir = Path(output_dir)

    if channel_type == "mag":
        first_key = next(iter(channel_groups), None)
        base_name = first_key.split("-")[0]
        if base_name in good_channels:
            print("letsgo here")
            raw = interpolate_missing_channels(raw, good_channels)
            data = {"m/eeg": [raw.get_data()]}

        else:
            channels_order = [
                ch
                for group, chans in channel_groups.items()
                if group not in ("bad", "EEG")
                for ch in chans
            ]
            raw.reorder_channels(channels_order)
            meg_data = fill_missing_channels(raw, len(good_channels))
            data = {"m/eeg": [meg_data]}

    elif channel_type == "eeg":
        channels_order = [
            ch
            for group, chans in channel_groups.items()
            if (group != "bad")
            for ch in chans
        ]
        raw.reorder_channels(channels_order)

        meg_data = fill_missing_channels(raw, len(good_channels))
        data = {"m/eeg": [meg_data]}

    else:
        raise ValueError(f"Unsupported channel_type: {channel_type}")

    save_obj(data, "data_raw", output_dir)


def create_windows(
    output_dir: str,
    window_size_s: int,
    stand: bool,
    sfreq: int,
    spike_spacing_from_border_s: float,
) -> int:
    """
    Crop windows from the pickle file and save them in a binary file.

    Parameters
    ----------
    output_dir : str
        Directory where processed data is stored.
    window_size_s : int
        Window size in seconds.
    stand : bool
        If True, standardize the data before saving.
    sfreq : int
        Sampling frequency in Hz.
    spike_spacing_from_border_s : float
        Minimum spacing from window borders in seconds,
        used to compute the stride between window centers.

    Returns
    -------
    int
        Total number of windows created.

    Raises
    ------
    RuntimeError
        If no valid windows could be created with the given parameters.
    """
    output_dir = Path(output_dir)

    window_size = int(window_size_s * sfreq)
    window_spacing = int((window_size_s - 2 * spike_spacing_from_border_s) * sfreq)

    data = load_obj("data_raw.pkl", output_dir)

    all_windows = []
    window_centers_all = []
    block_indices_all = []

    for block_idx, block_data in enumerate(data["m/eeg"]):
        window_centers = np.arange(window_size / 2, block_data.shape[1], window_spacing)

        block_windows = []
        for center in tqdm(window_centers, desc=f"Block {block_idx}"):
            if window_size / 2 <= center <= block_data.shape[1] - window_size / 2:
                low = int(center - window_size / 2)
                high = int(center + window_size / 2 + 0.1)  # Handle odd sizes
                block_windows.append(block_data[:, low:high])

                window_centers_all.append(center)
                block_indices_all.append(block_idx)

        if block_windows:
            all_windows.extend(block_windows)

    if not all_windows:
        raise RuntimeError("No valid windows were created. Check your parameters.")

    X_all = np.stack(all_windows).astype("float32")

    if stand:
        X_all = standardize(X_all)

    (output_dir / "data_raw_windows_bi").write_bytes(X_all.tobytes())

    # Save metadata
    save_obj(np.array(window_centers_all), "data_raw_timing", output_dir)
    save_obj(np.array(block_indices_all), "data_raw_blocks", output_dir)

    return len(X_all)

def generate_database(total_nb_windows: int) -> np.ndarray:
    """
    Generate a database of test window IDs.

    Args:
        total_nb_windows: Total number of windows.

    Returns:
        Array of shape (N, 1), for window index
    """
    X_test_ids = np.arange(total_nb_windows, dtype=int)
    return X_test_ids


def get_win_data_signal(f, win, dim):
    """
    Load and normalize a single window from a binary MEG data file.

    Parameters
    ----------
    f : file object
        Opened binary file containing MEG windows in float32 format.
    win : int
        Index of the window to retrieve.
    dim : tuple of int
        Shape of a single window as (n_channels, n_times).

    Returns
    -------
    numpy.ndarray
        Normalized window of shape (1, n_channels, n_times, 1).
    """
    f.seek(dim[0] * dim[1] * win * 4) 
    sample = np.fromfile(f, dtype="float32", count=dim[0] * dim[1])
    sample = sample.reshape(dim[1], dim[0])
    sample = np.swapaxes(sample, 0, 1)
    sample = np.expand_dims(sample, axis=-1)
    sample = np.expand_dims(sample, axis=0)

    mean = np.mean(sample)
    std = np.std(sample)
    sample_norm = (sample - mean) / std

    return sample_norm
