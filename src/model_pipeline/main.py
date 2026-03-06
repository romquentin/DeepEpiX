import sys
import ast
import importlib
import os


def run_model_pipeline(
    model_name,
    model_type,
    subject,
    output_path,
    threshold,
    adjust_onset,
    channel_groups,
    signal_cache_path,
    mne_info_cache_path,
    signal_name,
):
    
    # === Determine module based on model_name ===
    MODEL_MODULES = {
        "model_CNN.keras": "model_pipeline.run_CNN_features_models",
        "model_features_only.keras": "model_pipeline.run_CNN_features_models",
        "transformer.ckpt": "model_pipeline.run_hbiot",
        "hbiot.ckpt": "model_pipeline.run_hbiot",
        "model_CNN_EEG.keras": "model_pipeline.run_CNN_EEG_model",
        # "new_model": "model_pipeline.run_new_model",  # example
    }

    module_name = MODEL_MODULES.get(os.path.basename(model_name))
    if module_name is None:
        raise ValueError(f"Cannot determine backend for model '{model_name}'")

    model_module = importlib.import_module(module_name)
    test_model = getattr(model_module, "test_model")

    # === Run the model ===
    return test_model(
        model_name=model_name,
        output_path=output_path,
        signal_cache_path=signal_cache_path,
        mne_info_cache_path=mne_info_cache_path,
        adjust_onset=adjust_onset,
        channel_groups=channel_groups,
        signal_name=signal_name,
    )


if __name__ == "__main__":
    model_path = sys.argv[1]
    model_type = sys.argv[2]  # TensorFlow or Pytorch
    data_path = sys.argv[3]
    results_path = sys.argv[4]
    threshold = float(sys.argv[5])  # Convert back to float
    adjust_onset = str(sys.argv[6]).lower() == "true"  # Bool
    channel_groups = ast.literal_eval(sys.argv[7])
    signal_cache_path = sys.argv[8] if len(sys.argv) > 8 else None
    mne_info_cache_path = sys.argv[9] if len(sys.argv) > 6 else None
    signal_name = sys.argv[10]

    run_model_pipeline(
        model_path,
        model_type,
        data_path,
        results_path,
        threshold,
        adjust_onset,
        channel_groups,
        signal_cache_path,
        mne_info_cache_path,
        signal_name,
    )
