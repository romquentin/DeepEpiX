# Dash & Plotly
import dash
from dash import dcc
import plotly.graph_objects as go

# Standard Library
import os
import hashlib
from pathlib import Path
import json

# Third-party Libraries
import numpy as np
import mne
import dask.dataframe as dd
from dask import delayed
from flask_caching import Cache

# Local Modules
import config
from callbacks.utils import path_utils as dpu
from callbacks.utils import cache_utils as cu

app = dash.get_app()

cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": config.CACHE_TYPE,
        "CACHE_DIR": config.CACHE_DIR,
        "CACHE_DEFAULT_TIMEOUT": config.CACHE_DEFAULT_TIMEOUT,
    },
)

# RAW DATA PREPROCESSING (FILTERING, SUBSAMPLING) #################################################################


def sort_filter_resample(data_path, freq_data, channels_dict):

    raw = dpu.read_raw(data_path, preload=True, verbose=False)

    channels_order = [ch for group in channels_dict.values() for ch in group]
    raw.reorder_channels(channels_order)

    resample_freq = freq_data.get("resample_freq")
    low_pass_freq = freq_data.get("low_pass_freq")
    high_pass_freq = freq_data.get("high_pass_freq")
    notch_freq = freq_data.get("notch_freq")

    # Apply filtering and resampling
    raw.filter(l_freq=high_pass_freq, h_freq=low_pass_freq, n_jobs=-1)
    if notch_freq:
        raw.notch_filter(freqs=notch_freq, n_jobs=-1)
    raw.resample(resample_freq, n_jobs=-1)

    # raw.rename_channels({ch['ch_name']: ch['ch_name'].split('-')[0] for ch in raw.info['chs']})

    return raw


# MAIN CACHED FUNCTIONS (PREPROCESSING) ##############################################


def get_max_length(raw, resample_freq):
    return raw.times[-1] - 1 / resample_freq


def update_chunk_limits(total_duration):
    chunk_duration = config.CHUNK_RECORDING_DURATION
    chunk_limits = [
        [start, min(start + chunk_duration, total_duration)]
        for start in range(0, int(total_duration), chunk_duration)
    ]
    return chunk_limits


def get_cache_filename(
    data_path, freq_data, start_time, end_time, cache_dir=f"{config.CACHE_DIR}"
):
    # Create a unique hash key
    hash_input = f"{data_path}_{json.dumps(freq_data, sort_keys=True)}_{start_time}_{end_time}"
    hash_key = hashlib.md5(hash_input.encode()).hexdigest()
    return os.path.join(cache_dir, f"{hash_key}.parquet")

def get_cache_filename_cleaned(data_path, freq_data, start_time, end_time, 
                                excluded_components, cache_dir):
    import json
    hash_input = (
        f"{data_path}_{json.dumps(freq_data, sort_keys=True)}"
        f"_{start_time}_{end_time}"
        f"_ica_excluded_{sorted(excluded_components)}"
    )
    hash_key = hashlib.md5(hash_input.encode()).hexdigest()
    return os.path.join(cache_dir, f"{hash_key}.parquet")

def get_preprocessed_dataframe_dask(
    data_path,
    freq_data,
    start_time,
    end_time,
    channels_dict,
    excluded_ica_components=None,
    prep_raw=None,
    cache_dir=f"{config.CACHE_DIR}",
):
    os.makedirs(cache_dir, exist_ok=True)
    cu.clear_old_cache_files(cache_dir)

    if excluded_ica_components:
        print(f"Composants ICA exclus demandés : {excluded_ica_components}")
        cleaned_cache = get_cache_filename_cleaned(
            data_path, freq_data, start_time, end_time, 
            excluded_ica_components, cache_dir
        )

        if os.path.exists(cleaned_cache):
            return dd.read_parquet(cleaned_cache)
        
    cache_file = get_cache_filename(
        data_path, freq_data, start_time, end_time, cache_dir
    )

    if os.path.exists(cache_file):
        return dd.read_parquet(cache_file)

    # Otherwise, compute and save
    @delayed
    def load_sort_filter():
        return sort_filter_resample(data_path, freq_data, channels_dict)

    @delayed
    def crop_and_to_df(prep_raw):
        raw_chunk = prep_raw.copy().crop(tmin=start_time, tmax=end_time)
        return raw_chunk.to_data_frame(picks="all", index="time")

    @delayed
    def standardize(raw_df):
        return raw_df - raw_df.mean(axis=0)

    if prep_raw is None:
        prep_raw = load_sort_filter()
    raw_df = crop_and_to_df(prep_raw)
    raw_df_std = standardize(raw_df)

    df = raw_df_std.compute()
    ddf = dd.from_pandas(df, npartitions=10)
    ddf.to_parquet(cache_file)

    return ddf


def get_preprocessed_dataframe(data_path, freq_data, start_time, end_time, raw=None):
    """
    Preprocess the M/EEG data in chunks and cache them.

    :param data_path: Path to the raw data file.
    :param freq_data: Dictionary containing frequency parameters for preprocessing.
    :param chunk_duration: Duration of each chunk in seconds (default is 3 minutes).
    :param cache: Cache object to store preprocessed chunks.
    :return: Processed dataframe in pandas format.
    """

    @cache.memoize(make_name=f"{data_path}:{freq_data}:{start_time}:{end_time}")
    def process_data_in_chunks(
        data_path, freq_data, start_time, end_time, prep_raw=None
    ):
        try:
            if prep_raw is None:
                prep_raw = sort_filter_resample(data_path, freq_data)

            raw_chunk = prep_raw.copy().crop(tmin=start_time, tmax=end_time)
            raw_df = raw_chunk.to_data_frame(picks="meg", index="time")
            raw_df_standardized = raw_df - raw_df.mean(axis=0)

            return raw_df_standardized

        except Exception as e:
            return f"⚠️ Error during processing: {str(e)}"

    # Process and return the result in JSON format
    return process_data_in_chunks(data_path, freq_data, start_time, end_time, raw)


# ICA ########################################################################


def get_cache_filename_ica(data_path, start_time, end_time, ica_result_path, cache_dir):
    hash_input = f"{data_path}_{start_time}_{end_time}_{ica_result_path}"
    hash_key = hashlib.md5(hash_input.encode()).hexdigest()
    return os.path.join(cache_dir, f"{hash_key}.parquet")

def run_ica_processing(data_path, n_components, ica_method, max_iter, decim, channel_store, cache_dir, ica_store):

    ica_result_path = Path(cache_dir) / f"{n_components}_{ica_method}_{max_iter}_{decim}-ica.fif"
    components_dir = Path(cache_dir) / f"{n_components}_{ica_method}_{max_iter}_{decim}-ica-components"

    print(components_dir)

    if ica_result_path.exists() and str(ica_result_path) in ica_store:
        ica = mne.preprocessing.read_ica(ica_result_path)
        explained_variance = np.sum(ica.pca_explained_variance_[:ica.n_components_]) / np.sum(ica.pca_explained_variance_)
        return ica_result_path, components_dir, True, ica.n_components_, explained_variance

    raw = dpu.read_raw(
        data_path,
        preload=True,
        verbose=False,
        bad_channels=channel_store.get("bad", []),
    ).pick_types(meg=True)
    raw.filter(l_freq=1.0, h_freq=None)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=ica_method,
        max_iter=max_iter,
        random_state=97,
    )
    ica.fit(raw, decim=decim)
    ica.save(ica_result_path, overwrite=True)

    _save_ica_component_plots(ica, components_dir)

    total_var = np.sum(ica.pca_explained_variance_)
    explained_var_sum = np.sum(ica.pca_explained_variance_[:ica.n_components_])
    explained_variance = explained_var_sum / total_var

    return ica_result_path, components_dir, False, ica.n_components_, explained_variance

def _save_ica_component_plots(ica, components_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components_dir.mkdir(parents=True, exist_ok=True)

    figs = ica.plot_components(show=False)
    if not isinstance(figs, list):
        figs = [figs]

    for i, fig in enumerate(figs):
        fig.savefig(components_dir / f"page_{i}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

def get_ica_dataframe_dask(
    data_path,
    start_time,
    end_time,
    ica_result_path,
    raw=None,
    cache_dir=f"{config.CACHE_DIR}",
):
    os.makedirs(cache_dir, exist_ok=True)
    cu.clear_old_cache_files(cache_dir)

    cache_file = get_cache_filename_ica(
        data_path, start_time, end_time, ica_result_path, cache_dir
    )

    if os.path.exists(cache_file):
        return dd.read_parquet(cache_file)

    if raw is None:
        raw = dpu.read_raw(data_path, preload=True, verbose=False).pick_types(meg=True)
        raw.filter(l_freq=1.0, h_freq=None)

    raw = raw.copy().crop(tmin=start_time, tmax=end_time)

    ica = mne.preprocessing.read_ica(ica_result_path)
    sources = ica.get_sources(raw)

    data = sources.get_data()
    sfreq = sources.info["sfreq"]
    ch_names = sources.ch_names
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="misc")

    new_sources = mne.io.RawArray(data, info)
    sources = new_sources.resample(300)
    sources_df = sources.to_data_frame(index="time")

    ddf = dd.from_pandas(sources_df, npartitions=10)
    ddf = ddf - ddf.mean(axis=0)
    ddf.to_parquet(cache_file)

    return ddf

def get_ica_cleaned_dataframe_dask(
        data_path,
        freq_data,
        start_time,
        end_time,
        ica_result_path,
        excluded_components,
        raw,
        cache_dir=f"{config.CACHE_DIR}",
    ):
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = get_cache_filename_cleaned(
        data_path, freq_data, start_time, end_time,
        excluded_components, cache_dir
    )
    
    if raw is None:
        raw = dpu.read_raw(data_path, preload=True, verbose=False).pick_types(meg=True)

    raw_chunk = raw.copy().crop(tmin=start_time, tmax=end_time)

    ica = mne.preprocessing.read_ica(ica_result_path)
    ica.exclude = list(excluded_components)
    ica.apply(raw_chunk)

    cleaned_df = raw_chunk.to_data_frame(index="time")
    ddf = dd.from_pandas(cleaned_df, npartitions=10)
    ddf.to_parquet(cache_file, overwrite=True)

    return ddf


# POWER SPECTRUM DECOMPOSITION ######################################################


def compute_power_spectrum_decomposition(data_path, freq_data, theme="light"):
    raw = dpu.read_raw(data_path, preload=True, verbose=False)

    low_pass_freq = freq_data.get("low_pass_freq")
    high_pass_freq = freq_data.get("high_pass_freq")
    notch_freq = freq_data.get("notch_freq")

    if not low_pass_freq or not high_pass_freq:
        return dash.no_update

    if notch_freq:
        raw.notch_filter(freqs=notch_freq)

    psd_data = raw.compute_psd(
        method="welch",
        fmin=high_pass_freq,
        fmax=low_pass_freq,
        n_fft=2048,
        picks="meg",
        n_jobs=-1,
    )
    psd, freqs = psd_data.get_data(return_freqs=True)

    psd_dB = 10 * np.log10(psd)

    text_color = "#000" if theme == "light" else "#FFF"
    grid_color = (
        "rgba(200,200,200,0.3)" if theme == "light" else "rgba(255,255,255,0.1)"
    )

    psd_fig = go.Figure()

    for ch_idx, ch_name in enumerate(psd_data.ch_names):
        psd_fig.add_trace(
            go.Scatter(
                x=freqs,
                y=psd_dB[ch_idx],
                mode="lines",
                line=dict(width=1),
                name=ch_name,
            )
        )

    psd_fig.update_layout(
        title=dict(text="Power Spectral Density (PSD)", font=dict(color=text_color)),
        xaxis=dict(
            title="Frequency (Hz)",
            type="linear",
            showgrid=True,
            gridcolor=grid_color,
            color=text_color,
        ),
        yaxis=dict(
            title="Power (dB)",
            type="linear",
            showgrid=True,
            gridcolor=grid_color,
            color=text_color,
        ),
        font=dict(color=text_color),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=text_color)),
    )

    return dcc.Graph(figure=psd_fig)
