import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from mne.io.base import BaseRaw
import os
from scipy.ndimage import median_filter
import torch
from scipy import stats
import logging
import lightning as L
import pickle
from pathlib import Path
import yaml
import traceback

from model_pipeline.utils import load_raw_from_parquet
from utils_biot.models import BIOTClassifier, BIOTHierarchicalClassifier


logger = logging.getLogger(__name__)

class PredictionDataModule(L.LightningDataModule):
    """Lightning DataModule for prediction on single MEG files.

    Note: At inference time, channel selection is handled automatically by PredictDataset
    based on the available channels in the MEG file. No reference channels are needed.
    """

    def __init__(
        self,
        signal_path: str,
        mne_info_path: str,
        dataset_config: Dict[str, Any],
        dataloader_config: Dict[str, Any],
        reference_channels_path: Optional[str] = None,
        num_workers_ratio: float = 0.5,
        **kwargs
    ):
        """Initialize prediction data module.

        Args:
            signal_path: Path to the preprocessed file
            dataset_config: Configuration for data processing
            dataloader_config: Configuration for data loaders
            reference_channels_path: Path to reference channels pickle file
            num_workers_ratio: Ratio of CPU cores to use for workers (default: 0.5)
            **kwargs: Additional parameters for compatibility (unused)
        """
        super().__init__()
        self.signal_path = signal_path
        self.mne_info_path = mne_info_path
        self.dataset_config = dataset_config
        self.dataloader_config = dataloader_config
        self.reference_channels_path = reference_channels_path
        self.num_workers_ratio = num_workers_ratio

        self.predict_dataset: Optional[PredictDataset] = None
        self.input_shape: Optional[torch.Size] = None
        self.output_shape: Optional[torch.Size] = None
        
    def prepare_data(self):
        """Prepare data - verify file exists."""
        if not os.path.exists(self.signal_path):
            raise FileNotFoundError(f"Preprocessed signal not found: {self.signal_path}")
            
    def setup(self, stage: Optional[str] = None):
        """Set up the prediction dataset."""
        if stage == 'predict' or stage is None:
            self.predict_dataset = PredictDataset(
                signal_path=self.signal_path,
                mne_info_path=self.mne_info_path,
                dataset_config=self.dataset_config,
                reference_channels_path=self.reference_channels_path,
            )
           
            # Set shapes
            if len(self.predict_dataset) > 0:
                sample = self.predict_dataset[0]
                data = sample[0]  # chunk data
                self.input_shape = data.shape
                self.output_shape = torch.Size([data.shape[0]])  # n_windows

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Create the prediction dataloader with dynamic num_workers."""
        if self.predict_dataset is None:
            raise RuntimeError("Call setup() before getting prediction dataloader")

        predict_config = self.dataloader_config.get('predict', self.dataloader_config.get('test', {})).copy()

        # Check that shuffle is False for prediction
        if predict_config.get('shuffle', True):
            predict_config['shuffle'] = False

        if 'num_workers' not in predict_config or predict_config['num_workers'] == 0:
            optimal_workers = get_optimal_num_workers(
                ratio=self.num_workers_ratio,
                min_workers=0,
                max_workers=None
            )
            predict_config['num_workers'] = optimal_workers

        return torch.utils.data.DataLoader(
            self.predict_dataset,
            **predict_config,
            collate_fn=predict_collate_fn,
        )
    
    def get_input_shape(self) -> torch.Size:
        """Get the input shape for model initialization."""
        if self.input_shape is None:
            raise RuntimeError("Call setup() before getting input shape")
        return self.input_shape
    
    def get_output_shape(self) -> torch.Size:
        """Get the output shape for model initialization."""
        if self.output_shape is None:
            raise RuntimeError("Call setup() before getting output shape")
        return self.output_shape

class PredictDataset(torch.utils.data.Dataset):
    """Dataset for prediction using sequential chunk extraction.

    Returns:
        Tuple of (chunk_data, metadata) with unified metadata convention including
        chunk_onset_sample, chunk_idx, window_times, etc.
    """

    def __init__(
        self,
        signal_path: str,
        mne_info_path: str,
        dataset_config: Dict[str, Any],
        n_channels: int = 275,
        reference_channels_path: Optional[str] = None,
    ):
        """Initialize prediction dataset with sequential chunk extraction.

        Args:
            signal_path: Path to the preprocessed signal file (.parquet).
            dataset_config: Configuration for data processing.
            n_channels: Number of MEG channels (default: 275) for consistent input size.
        """
        self.signal_path = signal_path
        self.mne_info_path = mne_info_path
        self.dataset_config = dataset_config
        self.n_channels = n_channels
        if reference_channels_path is not None:
            with open(reference_channels_path, 'rb') as f:
                self.reference_channels = pickle.load(f)
        else:
            self.reference_channels = None

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing PredictDataset for {signal_path}")

        # Load and preprocess the recording once
        self.meg_data = None
        self.channel_info = None
        self.sampling_rate = None
        self.n_chunks = 0

        self._load_recording()

    def _load_recording(self):
        """Load and preprocess the MEG recording once."""
        try:
            raw, self.meg_data, self.channel_info = load_and_process_meg_data(
                self.signal_path,
                self.mne_info_path,
                self.dataset_config,
                good_channels=self.reference_channels,
                n_channels=self.n_channels,
            )

            self.sampling_rate = raw.info['sfreq']
            raw.close()

            self.all_windows = create_windows(
                self.meg_data,
                self.sampling_rate,
                self.dataset_config['window_duration_s'],
                self.dataset_config.get('window_overlap', 0.0),
            )

            num_context_windows = self.dataset_config['n_windows']
            total_windows = len(self.all_windows)
            self.n_chunks = (total_windows + num_context_windows - 1) // num_context_windows

            print(f"Loaded recording: {self.meg_data.shape[1]} samples, "
                           f"{total_windows} windows, {self.n_chunks} chunks")

        except Exception as e:
            self.logger.error(f"Error loading file {self.signal_path}: {e}")
            raise

    def __len__(self) -> int:
        """Return number of chunks."""
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Extract a chunk sequentially for prediction.

        Args:
            idx: Chunk index (0-based).

        Returns:
            Tuple of (chunk_data, metadata) with chunk_data as tensor of shape
            (n_windows, n_channels, window_samples) and metadata dictionary.
        """
        num_context_windows = self.dataset_config['n_windows']
        window_duration_samples = int(self.dataset_config['window_duration_s'] * self.sampling_rate)
        window_overlap = self.dataset_config.get('window_overlap', 0.0)
        window_step = max(1, int(window_duration_samples * (1 - window_overlap)))

        start_window_idx = idx * num_context_windows
        end_window_idx = min(start_window_idx + num_context_windows, len(self.all_windows))

        windows = self.all_windows[start_window_idx:end_window_idx]

        chunk_onset_sample = start_window_idx * window_step

        window_times = []
        for local_idx, global_idx in enumerate(range(start_window_idx, end_window_idx)):
            window_start = global_idx * window_step
            window_end = window_start + window_duration_samples
            window_center = window_start + window_duration_samples // 2

            peak_sample, peak_time = find_gfp_peak_in_window(
                self.meg_data, window_start, window_end, self.sampling_rate
            )

            window_times.append({
                'start_sample': int(window_start),
                'end_sample': int(window_end),
                'center_sample': int(window_center),
                'peak_sample': int(peak_sample),
                'start_time': float(window_start / self.sampling_rate),
                'end_time': float(window_end / self.sampling_rate),
                'center_time': float(window_center / self.sampling_rate),
                'peak_time': float(peak_time),
                'window_idx_in_chunk': local_idx,
                'global_window_idx': global_idx,
            })

        metadata = {
            'chunk_onset_sample': chunk_onset_sample,
            'chunk_offset_sample': chunk_onset_sample + len(windows) * window_step + (window_duration_samples - window_step),
            'chunk_duration_samples': len(windows) * window_step + (window_duration_samples - window_step),
            'chunk_idx': idx,
            'start_window_idx': start_window_idx,
            'end_window_idx': end_window_idx,
            'n_windows': len(windows),
            'window_times': window_times,
            'window_duration_s': self.dataset_config['window_duration_s'],
            'window_duration_samples': window_duration_samples,
            'signal_path': self.signal_path,
            'channel_mask': self.channel_info.get('channel_mask', None) if self.channel_info else None,
            'selected_channels': self.channel_info.get('selected_channels', []) if self.channel_info else [],
            'n_selected_channels': len(self.channel_info.get('selected_channels', [])) if self.channel_info else 0,
            'USE_REFERENCE_CHANNELS': self.channel_info.get('USE_REFERENCE_CHANNELS', False) if self.channel_info else False,
            'sampling_rate': self.sampling_rate,
            'is_test_set': False,
            'extraction_mode': 'sequential',
        }

        return torch.tensor(windows, dtype=torch.float32), metadata
    
class MEGSpikeDetector(L.LightningModule):
    """Lightning module for spike detection in MEG data.

    This module handles training, validation, and testing of MEG spike detection models.
    All metrics computation and reporting is handled by the MetricsEvaluationCallback.

    Attributes:
        config: Configuration dictionary containing all component settings
        model: The neural network model for spike detection
        loss_fn: The loss function for training
        threshold: Classification threshold for binary predictions (updated by callback)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        input_shape: Tuple[int, int, int],
        log_dir: str,
        **_kwargs,
    ) -> None:
        """Initialize the Lightning module with configuration.

        Args:
            config: Configuration dictionary containing model, loss, optimizer settings
            input_shape: Shape of the input data (channels, time_points)
            log_dir: Directory for logging
            **kwargs: Additional keyword arguments

        Raises:
            ValueError: If required configuration keys are missing
            TypeError: If input_shape is not a tuple
        """
        # Input validation
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dictionary, got {type(config)}")

        required_keys = ["model", "data", "evaluation"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise ValueError(
                f"input_shape must be a tuple of length 3, got {input_shape}"
            )
        super().__init__()
        logger.info("Initializing ConfigurableLightningModule")
        self.config = config
        self.log_dir = log_dir
        self.input_shape = input_shape
        config["model"][config["model"]["name"]]["input_shape"] = list(input_shape)
        config["model"][config["model"]["name"]]["log_dir"] = log_dir
        self.save_hyperparameters(config)

        # Create model and processing flags
        self.contextual = config["model"][config["model"]["name"]].get(
            "contextual", False
        )
        self.sequential_processing = config["model"][config["model"]["name"]].get(
            "sequential_processing", False
        )
        if config["model"]["name"] == "BIOT":
            self.model = BIOTClassifier(**config["model"]["BIOT"])    
        elif config["model"]["name"] == "BIOTHierarchical":
            self.model = BIOTHierarchicalClassifier(
                **config["model"]["BIOTHierarchical"]
            )
        else:
            raise ValueError(f"Unsupported model name: {config['model']['name']}")

        # Temperature scaling configuration and validation
        self.temperature_scaling_enabled = config["evaluation"].get(
            "temperature_scaling", False
        )
        # Classification threshold (can be updated by MetricsEvaluationCallback if threshold_optimization=True)
        self.threshold = config["evaluation"].get("default_threshold", 0.5)

        # Temperature scaling for calibrated predictions (1.0 = no scaling)
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
        self.temperature.requires_grad = (
            False  # Only optimized during temperature scaling phase
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore threshold and temperature from checkpoint if available."""
        super().on_load_checkpoint(checkpoint)
        if "hyper_parameters" in checkpoint:
            if "threshold" in checkpoint["hyper_parameters"]:
                self.threshold = checkpoint["hyper_parameters"]["threshold"]
                print(f"Restored threshold from checkpoint: {self.threshold:.4f}")
            if "temperature" in checkpoint["hyper_parameters"]:
                temp_value = checkpoint["hyper_parameters"]["temperature"]
                if isinstance(temp_value, torch.Tensor):
                    self.temperature.data = temp_value.to(self.temperature.device)
                else:
                    self.temperature.data = torch.tensor(
                        [temp_value], device=self.temperature.device
                    )
                print(
                    f"Restored temperature from checkpoint: {self.temperature.item():.4f}"
                )

    def apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits for calibrated predictions.

        Temperature scaling divides logits by a learned temperature parameter T:
        - T > 1: Makes predictions less confident (smoother probabilities)
        - T = 1: No scaling (default)
        - T < 1: Makes predictions more confident (sharper probabilities)

        Args:
            logits: Raw model logits [batch_size, n_windows, n_classes] or [batch_size, n_windows]

        Returns:
            Temperature-scaled logits of the same shape
        """
        return logits / self.temperature

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor],
        window_mask: Optional[torch.Tensor] = None,
        force_sequential: bool = False,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of the model with contextual and sequential processing support.

        Handles different processing modes:
        - Contextual: Pass full sequence [batch_size, n_windows, n_channels, n_timepoints] to model
        - Non-contextual + batch mode: Reshape to [BxN_window, n_channels, n_timepoints]
        - Non-contextual + sequential: Loop through windows individually

        Args:
            x: Input tensor of shape [batch_size, n_windows, n_channels, n_timepoints]
            channel_mask: Optional channel mask tensor (B, C) where True=valid, False=masked.
            window_mask: Optional window mask tensor (B, N) where True=valid, False=masked.
            force_sequential: Whether to force sequential processing mode.
            *args: Additional positional arguments to pass to the model.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            torch.Tensor: Output logits of shape [batch_size, n_windows, n_classes]
        """
        if self.contextual:
            # Contextual models process the full sequence with temporal context
            return self.model(x, channel_mask, window_mask, *args, **kwargs)

        # Non-contextual processing for window-level models
        batch_size, n_windows, n_channels, n_timepoints = x.shape
        if self.sequential_processing or force_sequential:
            # Sequential mode: Process each window individually in a loop
            window_outputs = []
            for seg_idx in range(n_windows):
                window = x[:, seg_idx, :, :]  # [batch_size, n_channels, n_timepoints]
                window_output = self.model(
                    window, channel_mask, *args, **kwargs
                )  # [batch_size, n_classes]
                window_outputs.append(
                    window_output.unsqueeze(1)
                )  # [batch_size, 1, n_classes]

            return torch.cat(
                window_outputs, dim=1
            )  # [batch_size, n_windows, n_classes]
        else:
            # Batch mode: Reshape to process all windows simultaneously
            x = x.view(
                batch_size * n_windows, n_channels, n_timepoints
            )  # [B×N_window, n_channels, n_timepoints]
            channel_mask = (
                channel_mask.repeat_interleave(n_windows, dim=0)
                if channel_mask is not None
                else None
            )  # [B×N_window, n_channels]
            if "unknown_mask" in kwargs and kwargs["unknown_mask"] is not None:
                unknown_mask = kwargs["unknown_mask"].repeat_interleave(
                    n_windows, dim=0
                )
                kwargs["unknown_mask"] = unknown_mask  # [B×N_window, n_channels]
            result = self.model(
                x, channel_mask, *args, **kwargs
            )  # [B×N_window, n_classes]
            return result.view(
                batch_size, n_windows, -1
            )  # [batch_size, n_windows, n_classes]

    def predict_step(self, batch, batch_idx):
        """Perform a single prediction step.

        Args:
            batch: Batch data (X, window_mask, channel_mask, metadata) where:
                - X: Input MEG data [batch_size, n_windows, n_channels, n_timepoints]
                - window_mask: Valid window mask [batch_size, n_windows] - 1=valid, 0=padded
                - channel_mask: Valid channel mask [batch_size, n_channels] - 1=valid, 0=masked
                - metadata: Sample metadata for result export
            batch_idx: Index of the batch

        Returns:
            Dictionary containing predictions, probabilities, and metadata
        """
        X, window_mask, channel_mask, metadata = batch

        unknown_mask = None
        if not metadata[0]["USE_REFERENCE_CHANNELS"] and channel_mask is not None:
            # Channel mask is actually true everywhere but for padded channels
            # We actually don't know if good channels are really good at inference time, we just know that this is real data
            # So we use an unknown mask that is all True where channel_mask is given
            unknown_mask = torch.ones_like(channel_mask, dtype=torch.bool)

        # Forward pass with batch-aware channel mask
        force_sequential = not self.config["model"]["name"] == "BIOTHierarchical"
        logits = self.forward(
            X,
            channel_mask=channel_mask,
            window_mask=window_mask,
            unknown_mask=unknown_mask,
            force_sequential=force_sequential,
        )

        # Apply temperature scaling and compute calibrated probabilities
        scaled_logits = self.apply_temperature_scaling(logits)
        probs = torch.sigmoid(scaled_logits).cpu().detach()

        # Apply threshold for binary predictions
        predictions = (probs >= self.threshold).float()

        # Prepare outputs
        outputs = {
            "logits": logits.cpu().detach(),
            "probs": probs,
            "predictions": predictions,
            "batch_size": X.shape[0],
            "n_windows": X.shape[1] if len(X.shape) > 2 else 1,
            "batch_idx": batch_idx,
            "metadata": metadata if metadata else {},
            "channel_mask": (
                channel_mask.cpu().detach().float().numpy()
                if channel_mask is not None
                else None
            ),
            "window_mask": (
                window_mask.cpu().detach().float().numpy()
                if window_mask is not None
                else None
            ),
        }
        return outputs

    def _collect_batch_outputs(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict],
        logits: torch.Tensor,
    ) -> Dict[str, Any]:
        """Collect outputs for a single batch.

        Args:
            batch: Input batch (X, y, window_mask, channel_mask, metadata)
            logits: Model output logits for the batch

        Returns:
            Dictionary with batch outputs including per-window losses
        """
        X, y, window_mask, _channel_mask, metadata = batch

        # Apply temperature scaling for calibrated probabilities
        scaled_logits = self.apply_temperature_scaling(logits)

        # Compute per-window BCE loss without reduction for analysis (using scaled logits)
        # Note: Both scaled_logits and y are [B, N] for binary classification
        per_window_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scaled_logits, y, reduction="none"
        )
        probs = torch.sigmoid(scaled_logits)

        return {
            "logits": logits.cpu().detach().float().numpy(),
            "probs": probs.cpu().detach().float().numpy(),
            "predictions": (probs >= self.threshold).float().cpu().detach().numpy(),
            "gt": y.cpu().detach().float().numpy(),
            "mask": window_mask.cpu().detach().float().numpy(),
            "losses": per_window_loss.cpu()
            .detach()
            .float()
            .numpy(),  # Per-window losses
            "metadata": metadata if metadata else {},
            "batch_size": X.shape[0],
            "n_windows": X.shape[1],
        }

def get_optimal_num_workers(ratio: float = 0.5, min_workers: int = 0, max_workers: Optional[int] = None) -> int:
    """Dynamically determine the optimal number of workers for data loading.

    Args:
        ratio: Conservative ratio to multiply CPU count by (default: 0.5 for 50% of CPUs)
        min_workers: Minimum number of workers (default: 0)
        max_workers: Maximum number of workers (default: None, no limit)

    Returns:
        Optimal number of workers as an integer

    Example:
        # Use 50% of available CPUs
        num_workers = get_optimal_num_workers(ratio=0.5)

        # Use 75% of available CPUs, but at least 2 and at most 8
        num_workers = get_optimal_num_workers(ratio=0.75, min_workers=2, max_workers=8)
    """
    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1

    # Calculate optimal workers with conservative ratio
    optimal_workers = max(min_workers, int(cpu_count * ratio))

    # Apply maximum limit if specified
    if max_workers is not None:
        optimal_workers = min(optimal_workers, max_workers)

    logger.info(f"Dynamically determined num_workers: {optimal_workers} "
                f"(CPU count: {cpu_count}, ratio: {ratio}, "
                f"min: {min_workers}, max: {max_workers})")

    return optimal_workers

def load_and_process_meg_data(
    signal_cache_path: str,
    mne_info_cache_path: str,
    config: Dict[str, Any],
    good_channels: Optional[List[str]] = None,
    n_channels: int = 275,
    close_raw: bool = True
) -> Tuple[BaseRaw, np.ndarray, Dict[str, Any]]:
    """Load and process MEG data for prediction.
    
    Args:
        file_path: Path to the MEG data file.
        config: Configuration dictionary with preprocessing parameters.
        good_channels: List of channels that should be present. If None, use all available channels (useful for inference on new systems).
        n_channels: Number of MEG channels to use (default: 275) to enforce consistent input size
        close_raw: Whether to close the MNE Raw object after processing to free memory.
            
    Returns:
        Tuple containing:
            - raw: MNE Raw object after processing.
            - data: Processed MEG data array (n_channels, n_timepoints).
            - channel_info: loc information and channel mask.
    """
    USE_REFERENCE_CHANNELS = False
    try:
        raw, metadata = load_raw_from_parquet(signal_cache_path, mne_info_cache_path)

        if raw.info['sfreq'] != config['sampling_rate']:
            raw.resample(sfreq=config['sampling_rate'])

        if good_channels is None or not USE_REFERENCE_CHANNELS:
            good_channels = list(raw.ch_names)  # Use all available channels if no reference provided
            # sample n_channels from good_channels if more than n_channels are available

            if len(good_channels) > n_channels:
                good_channels = good_channels[:n_channels]
        
        # Select channels based on good channels and location information
        raw, channel_info = select_channels(raw, good_channels)
        
        # Get raw data from MNE (in order of selected_channels)
        raw_data = np.array(raw.get_data())  # Shape: (n_selected_channels, n_timepoints)
        n_timepoints = raw_data.shape[1]

        # Now normalize and filter
        raw_data = normalize_data(raw_data, config.get('normalization', {'method': 'robust_zscore', 'axis': None}))

        if config.get('median_filter_temporal_window_ms', 0) > 0:
            raw_data = apply_median_filter(raw_data, config['sampling_rate'], config['median_filter_temporal_window_ms'])

        if close_raw:
            raw.close()
        
        # Reorder data to match good_channels exactly
        # This ensures all samples in batch have data at same positions
        # Position i in data array ALWAYS represents good_channels[i]
        num_channels = max(n_channels, len(good_channels))
        data = np.zeros((num_channels, n_timepoints), dtype=raw_data.dtype)
        channel_mask = torch.zeros(num_channels, dtype=torch.bool)

        # Create index mapping for efficiency
        good_channels_index = {ch: i for i, ch in enumerate(good_channels)}

        # Place each channel's data at its correct position
        for ch_idx, ch_name in enumerate(channel_info['selected_channels']):
            if ch_name in good_channels_index:
                target_idx = good_channels_index[ch_name]
                data[target_idx, :] = raw_data[ch_idx, :]
                channel_mask[target_idx] = True
            else:
                logger.warning(f"Channel {ch_name} not in good_channels reference - skipping")

        # Store channel mask for batch collation
        channel_info['channel_mask'] = channel_mask
        channel_info['USE_REFERENCE_CHANNELS'] = USE_REFERENCE_CHANNELS and (good_channels is not None)

        return raw, data, channel_info

    except Exception as e:
        logger.error(f"Error processing {signal_cache_path}: {e}")
        raise


def select_channels(raw: BaseRaw, good_channels: List[str]) -> Tuple[BaseRaw, Dict[str, Any]]:
    """Select channels ensuring consistent ordering across all samples for batch compatibility.

    Args:
        raw: MNE Raw object containing MEG data
        good_channels: ORDERED list of reference channel names (defines canonical ordering)

    Returns:
        Tuple of (processed_raw, channel_info) where channel_info contains:
            - 'loc': Dictionary mapping selected channel names to coordinates (legacy)
            - 'selected_channels': ORDERED list of channel names matching good_channels order
            - 'n_selected': Number of channels actually present in raw data
            - 'n_with_coordinates': Number of channels with coordinate info (legacy)
    """
    logger.debug(f"Raw channels available: {len(raw.ch_names)} channels")
    logger.debug(f"Good channels reference: {len(good_channels)} channels")
    
    # Get available MEG channels from the raw data
    available_channels = set(raw.ch_names)  # Use set for O(1) lookup
    logger.debug(f"Available MEG channels: {len(available_channels)}")

    # Ensure batch consistency - all samples have same channel ordering
    selected_channels = [ch for ch in good_channels if ch in available_channels]

    if len(selected_channels) == 0:
        raise ValueError(f"No channels from good_channels found in raw data! "
                        f"Raw has: {list(raw.ch_names)[:10]}..., "
                        f"Expected: {good_channels[:10]}...")

    logger.debug(f"Selected {len(selected_channels)}/{len(good_channels)} channels from reference list")

    # Pick only the selected channels in the raw object
    raw = raw.pick_channels(selected_channels)
    channel_info = {
        'ch_info': raw.info['chs'],  # Full channel info from MNE},
        'selected_channels': selected_channels,
    }
    return raw, channel_info


def normalize_data(data: np.ndarray, norm_config: Dict, eps: Optional[float] = None) -> np.ndarray:
    """Normalize data using specified method."""
    if eps is None:
        eps = norm_config.get('epsilon', 1e-20)
    
    method = norm_config.get('method', 'robust_zscore')
    axis = norm_config.get('axis', None)
    
    if method == 'percentile':
        percentile = norm_config.get('percentile', 95)
        if not (0 < percentile < 100):
            raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
        q = np.percentile(np.abs(data), percentile, axis=axis, keepdims=True)
        return data / (q + eps)
    
    elif method == 'robust_normalize':
        median = np.median(data, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        iqr = q75 - q25
        return (data - median) / (iqr + eps)
    
    elif method == 'robust_zscore':
        median = np.median(data, axis=axis, keepdims=True)
        mad = stats.median_abs_deviation(data, axis=axis)  # type: ignore
        if axis is not None:
            mad = np.expand_dims(mad, axis=axis)
        return (data - median) / (mad + eps)  # type: ignore
    
    elif method == 'zscore':
        return (data - np.mean(data, axis=axis, keepdims=True)) / (np.std(data, axis=axis, keepdims=True) + eps)
    
    elif method == 'minmax':
        min_v = np.min(data, axis=axis, keepdims=True)
        max_v = np.max(data, axis=axis, keepdims=True)
        return (data - min_v) / (max_v - min_v + eps)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}. Supported methods: percentile, zscore, minmax, robust_normalize, robust_zscore")


def apply_median_filter(data: np.ndarray, sfreq: float, temporal_window_ms: float) -> np.ndarray:
    """Apply median filter with adaptive kernel size based on sampling frequency.
    
    Args:
        data: MEG data array of shape (n_channels, n_timepoints)
        sfreq: Sampling frequency in Hz
        temporal_window_ms: Temporal smoothing window in milliseconds
        
    Returns:
        Filtered data with same shape as input
    """
    if temporal_window_ms <= 0:
        return data
    
    # Calculate kernel size based on sampling frequency and temporal window
    kernel_samples = int(temporal_window_ms * sfreq / 1000)
    # Ensure odd kernel size for symmetric filtering
    kernel_size = kernel_samples if kernel_samples % 2 == 1 else kernel_samples + 1
    
    # Apply median filter along time axis (axis=1) for each channel
    return median_filter(data, size=(1, kernel_size))


def create_windows(
    meg_data: np.ndarray,
    sampling_rate: float,
    window_duration_s: float,
    window_overlap: float,
) -> np.ndarray:
    """Create windows from MEG data.
    
    Args:
        meg_data: MEG data array (n_channels, n_timepoints)
        sampling_rate: Sampling rate in Hz
        window_duration_s: Duration of each window in seconds
        window_overlap: Overlap between windows (0.0 to 1.0)
        
    Returns:
        Array of windows with shape (n_windows, n_channels, n_samples_per_window)
    """
    window_duration_samples = int(window_duration_s * sampling_rate)
    window_step = max(1, int(window_duration_samples * (1 - window_overlap)))
    
    windows = []
    seg_start = 0
    
    while seg_start + window_duration_samples <= meg_data.shape[1]:
        seg_end = seg_start + window_duration_samples
        windows.append(meg_data[:, seg_start:seg_end])
        seg_start += window_step
    
    return np.array(windows)


def compute_gfp(meg_data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute Global Field Power (GFP) from MEG data.
    
    Args:
        meg_data: MEG data array, shape (n_channels, n_timepoints) or (n_timepoints, n_channels)
        axis: Axis along which channels are located (0 for first dim, 1 for second dim)
        
    Returns:
        GFP values, shape (n_timepoints,)
    """
    gfp = np.std(meg_data, axis=axis)
    return gfp


def find_gfp_peak_in_window(
    meg_data: np.ndarray,
    window_start: int,
    window_end: int,
    sampling_rate: float
) -> Tuple[int, float]:
    """Find the peak GFP within a window.
    
    Args:
        meg_data: MEG data array, shape (n_channels, n_timepoints)
        window_start: Start sample of the window
        window_end: End sample of the window
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (peak_sample, peak_time_in_seconds)
    """
    # Extract window
    window_data = meg_data[:, window_start:window_end]
    
    # Compute GFP
    gfp = compute_gfp(window_data, axis=0)
    
    # Find peak
    peak_idx = np.argmax(gfp)
    peak_sample = window_start + peak_idx
    peak_time = peak_sample / sampling_rate
    
    return int(peak_sample), float(peak_time)

def predict_collate_fn(batch):
    """Collate function for prediction batches with padding and masking.

    Handles batches with (data, metadata) tuples from PredictDataset.
    Pads variable-length sequences and extracts channel masks from metadata.

    Args:
        batch: List of (data, metadata) tuples from dataset

    Returns:
        Tuple of (batch_data, batch_window_mask, batch_channel_mask, metadata_list)
    """
    # For chunked prediction: (data, metadata) - use padded collate for consistency with training
    data_list = [item[0] for item in batch]
    metadata_list = [item[1] for item in batch]

    # Pad data to same length as training (handles variable chunk sizes)
    seg_counts = [d.shape[0] for d in data_list]
    max_segs = max(seg_counts)

    padded_data, window_mask_list = [], []
    channel_mask_list = []

    for i, data in enumerate(data_list):
        n = data.shape[0]
        pad = max_segs - n
        padded_data.append(torch.cat([data, torch.zeros(pad, *data.shape[1:])]))
        window_mask_list.append(torch.cat([torch.ones(n), torch.zeros(pad)]))

        # Extract channel mask from metadata
        if metadata_list and i < len(metadata_list):
            ch_mask = metadata_list[i].get('channel_mask', None)
            if ch_mask is not None:
                if isinstance(ch_mask, list):
                    ch_mask = torch.tensor(ch_mask, dtype=torch.bool)
                elif not isinstance(ch_mask, torch.Tensor):
                    ch_mask = torch.tensor(ch_mask, dtype=torch.bool)
                channel_mask_list.append(ch_mask)
            else:
                n_channels = data.shape[1] if len(data.shape) > 1 else 1
                channel_mask_list.append(torch.ones(n_channels, dtype=torch.bool))
        else:
            n_channels = data.shape[1] if len(data.shape) > 1 else 1
            channel_mask_list.append(torch.ones(n_channels, dtype=torch.bool))

    batch_data = torch.stack(padded_data, dim=0)
    batch_window_mask = torch.stack(window_mask_list, dim=0)  # 1=real, 0=padded
    batch_channel_mask = torch.stack(channel_mask_list, dim=0) if channel_mask_list else None

    return batch_data, batch_window_mask, batch_channel_mask, metadata_list


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a YAML file with optional validation.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If validation fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing failed for {config_path}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise yaml.YAMLError(f"Failed to parse YAML file {config_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading config from {config_path}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Failed to load configuration: {e}")

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    def expand_env_vars(obj: Any, max_recursion_depth: int = 20) -> Any:
        """Recursively expand environment variables in strings within the config."""
        if max_recursion_depth <= 0:
            raise ValueError(
                "Maximum recursion depth reached while expanding environment variables. Default is 20."
            )

        # if this is a dict, recurse into values
        if isinstance(obj, dict):
            return {
                k: expand_env_vars(v, max_recursion_depth - 1) for k, v in obj.items()
            }
        # if this is a list, recurse into items
        elif isinstance(obj, list):
            return [expand_env_vars(i, max_recursion_depth - 1) for i in obj]
        # if this is a string, expand env vars and user (~)
        elif isinstance(obj, str):
            expanded = os.path.expandvars(os.path.expanduser(obj))
            return expanded
        else:
            return obj

    config = expand_env_vars(config)
    return config