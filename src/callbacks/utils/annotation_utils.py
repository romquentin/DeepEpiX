import os
from collections import Counter
from dash import dcc, html
import dash_bootstrap_components as dbc
import mne
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def get_annotation_descriptions(annotations_store):
    """
    Extracts the list of unique annotation description names
    from the annotation-store.

    Parameters
    ----------
    annotations_store : list 
        A list of dictionaries representing annotations.

    Returns
    -------
    list 
        A list of unique description names.
    """
    if not annotations_store or not isinstance(annotations_store, list):
        return Counter()
    descriptions = [
        annotation.get("description")
        for annotation in annotations_store
        if "description" in annotation
    ]
    return Counter(descriptions)


def get_heartbeat_event(raw, ch_name):
    # Find ECG events using the `find_ecg_events` function
    events, _, _ = mne.preprocessing.find_ecg_events(raw, ch_name=ch_name)

    sfreq = raw.info["sfreq"]
    event_list = []
    for event in events:
        onset_sample = event[0]
        onset_sec = onset_sample / sfreq
        description = "ECG Event"
        duration = 0
        event_list.append(
            {"onset": onset_sec, "description": description, "duration": duration}
        )

    return pd.DataFrame(event_list)


# def time_to_seconds(time_str):
#     """
#     Convert a time string 'HH:MM:SS.ssssss' into seconds as float.
#     """
#     t = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
#     delta = datetime.timedelta(
#         hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond
#     )
#     return round(delta.total_seconds(), 3)


def get_annotations_dataframe(raw, heartbeat_ch_name, modality):

    annotations_df = raw.annotations.to_data_frame()
    origin_time = pd.Timestamp(raw.annotations.orig_time)

    if pd.isna(origin_time):

        annotations_df["onset"] = (
            annotations_df["onset"].dt.hour * 3600
            + annotations_df["onset"].dt.minute * 60
            + annotations_df["onset"].dt.second
            + annotations_df["onset"].dt.microsecond / 1e6
        )
    else:
        annotations_df["onset"] = pd.to_datetime(
            annotations_df["onset"]
        ).dt.tz_localize("UTC")
        annotations_df["onset"] = (
            annotations_df["onset"] - origin_time
        ).dt.total_seconds()

    if modality in ["meg", "mixed"]:
        try:
            heartbeat_df = get_heartbeat_event(raw, heartbeat_ch_name)
            annotations_df = pd.concat(
                [annotations_df, heartbeat_df], ignore_index=True
            )
        except Exception as e:
            print(f"Warning: Could not extract heartbeat events: {e}")

    annotations_dict = annotations_df.to_dict(orient="records")

    return annotations_dict


def extract_meg_timepoints_from_mrk(file_path):
    """
    Extracts timepoints labeled 'MEG' from a .mrk file (AnyWave marker file).

    Args:
        file_path (str): Path to the .mrk file.

    Returns:
        List[float]: List of MEG timepoints.
    """
    meg_timepoints = []

    with open(file_path, "r") as f:
        for line in f:
            if line.strip().startswith("//") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == "MEG":
                try:
                    timepoint = float(parts[2])
                    meg_timepoints.append(timepoint)
                except ValueError:
                    continue
    return meg_timepoints


def get_mrk_annotations_dataframe(data_path, annotations_dict):
    mrk_file_path = None
    if os.path.isdir(data_path):
        for file in os.listdir(data_path):
            if file.lower().endswith(".mrk"):
                mrk_file_path = os.path.join(data_path, file)
                break

    # --- If .mrk file found, extract MEG timepoints and add to annotations ---
    if mrk_file_path and os.path.exists(mrk_file_path):
        meg_timepoints = extract_meg_timepoints_from_mrk(mrk_file_path)

        # Add MEG markers as new annotations
        meg_annotations = [
            {"onset": tp, "duration": 0, "description": "MEG"} for tp in meg_timepoints
        ]

        annotations_dict = (
            annotations_dict + meg_annotations
        )  # Extend list of annotations
        return annotations_dict


def get_annotations(prediction_or_truth, annotations_df):
    """
    Function to retrieve annotation onsets (timestamps) based on the selected prediction or ground truth.

    Parameters:
    - prediction_or_truth (str): Either the model's prediction or the ground truth label.
    - annotations_df (pandas.DataFrame): DataFrame containing the annotations with onset times and other info.

    Returns:
    - List of onset times for the selected annotation type.
    """
    if isinstance(prediction_or_truth, str):
        prediction_or_truth = [prediction_or_truth]

    filtered_annotations = annotations_df[
        annotations_df["description"].isin(prediction_or_truth)
    ]
    return filtered_annotations.index.tolist()


def build_table_events_statistics(annotations):
    descriptions = [ann["description"] for ann in annotations]
    onsets = [ann["onset"] for ann in annotations]
    durations = [ann["duration"] for ann in annotations]

    description_counts = Counter(descriptions)

    table_header = html.Thead(html.Tr([html.Th("Event Name"), html.Th("Count")]))
    table_body = html.Tbody(
        [
            html.Tr([html.Td(desc), html.Td(count)])
            for desc, count in description_counts.items()
        ]
    )

    annotation_table = dbc.Table(
        [table_header, table_body], bordered=True, striped=True, hover=True, size="sm"
    )

    stats_summary = html.Div(
        [
            html.Span(
                [
                    html.I(
                        className="bi bi-bar-chart-line",
                        style={"marginRight": "10px", "fontSize": "1.2em"},
                    ),
                    "Event Summary",
                ],
                className="card-title",
            ),
            dbc.ListGroup(
                [
                    dbc.ListGroupItem(f"Total annotations: {len(annotations)}"),
                    dbc.ListGroupItem(f"Unique event types: {len(description_counts)}"),
                    dbc.ListGroupItem(f"First event starts at {min(onsets):.2f} s"),
                    dbc.ListGroupItem(
                        f"Last event ends at {max([o + d for o, d in zip(onsets, durations)]):.2f} s"
                    ),
                ],
                style={"marginBottom": "15px"},
            ),
            annotation_table,
        ]
    )

    return stats_summary


def build_table_prediction_statistics(df_spike, total_windows):
    spike_count = len(df_spike)
    spike_ratio = (spike_count / total_windows) * 100 if total_windows else 0

    if spike_count > 0:
        min_prob = df_spike["probas"].min()
        max_prob = df_spike["probas"].max()
        mean_prob = df_spike["probas"].mean()
        median_prob = df_spike["probas"].median()
    else:
        min_prob = max_prob = mean_prob = median_prob = 0

    stats_summary = dbc.ListGroup(
        [
            dbc.ListGroupItem(f"Total Windows: {total_windows}"),
            dbc.ListGroupItem(f"Spike Events Detected): {spike_count}"),
            dbc.ListGroupItem(f"Spike Ratio: {spike_ratio:.2f}%"),
            dbc.ListGroupItem(f"Min Spike Probability: {min_prob:.2f}"),
            dbc.ListGroupItem(f"Max Spike Probability: {max_prob:.2f}"),
            dbc.ListGroupItem(f"Mean Spike Probability: {mean_prob:.2f}"),
            dbc.ListGroupItem(f"Median Spike Probability: {median_prob:.2f}"),
        ]
    )

    return stats_summary


def build_prediction_distribution_statistics(df, threshold):
    df_below_threshold = df[df["probas"] <= threshold]
    df_above_threshold = df[df["probas"] > threshold]

    hist_below = go.Histogram(
        x=df_below_threshold["probas"],
        nbinsx=10,
        name=f"Below {threshold}",
        marker=dict(color="yellow"),
        opacity=0.7,
        showlegend=False,
    )

    hist_above = go.Histogram(
        x=df_above_threshold["probas"],
        nbinsx=10,
        name=f"Above {threshold}",
        marker=dict(color="red"),
        opacity=0.7,
        showlegend=False,
    )

    hist_fig = go.Figure(data=[hist_below, hist_above])

    hist_fig.update_layout(
        barmode="overlay",
        bargap=0.2,
        xaxis_title="Probability",
        yaxis_title="Count",
        xaxis=dict(title_font=dict(size=10), tickfont=dict(size=9)),
        yaxis=dict(title_font=dict(size=10), tickfont=dict(size=9)),
        margin=dict(t=0, b=0, l=0, r=0),
    )

    return dcc.Graph(
        figure=hist_fig,
        config={"staticPlot": False, "displayModeBar": True, "scrollZoom": False},
        style={"height": "300px"},
    )


def compute_multiway_intersections(selected, tolerance):
    """
    Compute intersections across N annotation descriptions.

    Parameters
    ----------
    selected : list of dicts with {"description", "onset", "duration"}
    tolerance : float, max time diff in seconds

    Returns
    -------
    list of dicts (intersection annotations)
    """
    if not selected:
        return []

    # Group by description (to know how many types we need to cover)
    descs = {a["description"] for a in selected}
    n_groups = len(descs)

    # Flatten into (onset, duration, description) and sort
    events = sorted(
        [(a["onset"], a["duration"], a["description"]) for a in selected],
        key=lambda x: x[0],
    )

    results = []
    cluster = []

    def finalize_cluster(cluster):
        """Check cluster and return intersection annotation if valid."""
        descs_in_cluster = {d for _, _, d in cluster}
        if len(descs_in_cluster) == n_groups:
            return {
                "description": f"{'∩'.join(sorted(descs))}",
                "onset": float(np.mean([c[0] for c in cluster])),
                "duration": float(np.min([c[1] for c in cluster])),
            }
        return None

    for onset, duration, desc in events:
        if not cluster:
            cluster = [(onset, duration, desc)]
            continue

        cluster_onsets = [c[0] for c in cluster]
        if abs(onset - np.mean(cluster_onsets)) <= tolerance:
            cluster.append((onset, duration, desc))
        else:
            # finalize old cluster
            ann = finalize_cluster(cluster)
            if ann:
                results.append(ann)
            cluster = [(onset, duration, desc)]

    # Final cluster check
    ann = finalize_cluster(cluster)
    if ann:
        results.append(ann)

    return results
