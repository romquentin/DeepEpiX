import time
from dash import html


def fill_history_data(history_data, category, action, n_components=None, explained_var=None, ica_key=None):
    if not isinstance(history_data, dict):
        history_data = {"annotations": [], "models": [], "ICA": [], "metadata": {}}

    if "metadata" not in history_data:
        history_data["metadata"] = {}

    if category not in history_data:
        history_data[category] = []
    
    if category == "ICA":
        if ica_key is not None:
            if "ica_results" not in history_data["metadata"]:
                history_data["metadata"]["ica_results"] = {}
            if ica_key not in history_data["metadata"]["ica_results"]:
                history_data["metadata"]["ica_results"][ica_key] = {
                    "n_components": n_components,
                    "explained_var": explained_var,
                    "excluded_components": [],
                }
            else:
                entry = history_data["metadata"]["ica_results"][ica_key]
                if n_components is not None:
                    entry["n_components"] = n_components
                if explained_var is not None:
                    entry["explained_var"] = explained_var
            
    if action is None:
        return history_data

    if isinstance(action, str):
        current_time = time.strftime("%I:%M %p")  # e.g., "03:45 PM"
        entry = f"{current_time} - {action}"
        if explained_var is not None:
            entry = f"{entry} Explained Var : {explained_var * 100:.2f}%"
        history_data[category].insert(0, entry)  # Prepend to the list
        return history_data

    raise ValueError("Action must be a string or None")


def read_history_data_by_category(history_data, category):
    if not isinstance(history_data, dict) or category not in history_data:
        return []
    return [html.Span(entry) for entry in history_data.get(category, [])]
