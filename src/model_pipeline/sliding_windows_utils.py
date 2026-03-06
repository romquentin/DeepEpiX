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
    Apply preprocessing, extract MEG/EEG data, and save it in a pickle file.

    Args:
        subject_path: Path to raw MEG/EEG data file (.ds, .fif, or directory).
        output_dir: Directory where processed data will be stored.
        channel_groups: Dict of channel groups (must include "bad" if applicable).
        good_channels: List of known good channels.
        channel_type: Channel type to pick ("mag" or "eeg").
        freq: Tuple (low_cutoff, high_cutoff) for bandpass filter.
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
                if (group != "bad" and group != "EEG")
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
    window_size_ms: int,
    stand: bool,
    sfreq: int,
    spike_spacing_from_border_ms: float,
) -> int:
    """
    Crop windows from the pickle file and save them in a binary file.

    Args:
        output_dir: Directory where processed data is stored.
        window_size_ms: Window size in milliseconds.
        stand: If True, standardize the data.

    Returns:
        Total number of windows created.
    """
    output_dir = Path(output_dir)

    # Window size in samples (ms × sampling frequency)
    window_size = int(window_size_ms * sfreq)
    # Spacing between window centers (samples)
    window_spacing = int((window_size_ms - 2 * spike_spacing_from_border_ms) * sfreq)

    # Load preprocessed data
    data = load_obj("data_raw.pkl", output_dir)

    all_windows = []
    window_centers_all = []
    block_indices_all = []

    for block_idx, block_data in enumerate(data["m/eeg"]):
        # Compute window centers (in samples)
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

    # Stack and convert to float32 for binary saving
    X_all = np.stack(all_windows).astype("float32")

    if stand:
        X_all = standardize(X_all)

    # Save binary MEG windows
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

    # Store sample
    f.seek(dim[0] * dim[1] * win * 4)  # 4 because its float32 and dtype.itemsize = 4
    sample = np.fromfile(f, dtype="float32", count=dim[0] * dim[1])
    sample = sample.reshape(dim[1], dim[0])
    sample = np.swapaxes(sample, 0, 1)
    sample = np.expand_dims(sample, axis=-1)
    sample = np.expand_dims(sample, axis=0)

    mean = np.mean(sample)
    std = np.std(sample)
    sample_norm = (sample - mean) / std

    return sample_norm
