# Copyright contributors to the TSFM project
# Cherry-picked from https://github.com/ibm-granite/granite-tsfm/commit/2c7d487fb83c80ce29c144f5bdcbaa13eb668eb0#diff-dd5aba95a6a7930ce51897d898aa12f61234418aec1f5dc5b4b10d94686aba6dR10
# with minor modifications

"""Tools for building TTM Predictor that works with GluonTS datasets"""
import logging
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.split import InputDataset
from gluonts.itertools import batcher
from gluonts.model import Predictor
from gluonts.model.forecast import SampleForecast
from gluonts.transform import LastValueImputation
from tqdm.auto import tqdm
from tsfm_public.toolkit.get_model import get_model

# TTM Constants
TTM_MAX_FORECAST_HORIZON = 720
logger = logging.getLogger(__file__)

def impute_series(target):
    if np.isnan(target).any():
        target = target.copy()
        if len(target.shape) == 2:
            for i in range(target.shape[0]):
                target[i, ...] = LastValueImputation()(target[i, ...])
        elif len(target.shape) == 1:
            target = LastValueImputation()(target)
        else:
            raise Exception(
                "Only 1D and 2D arrays are accepted by the impute_series() function."
            )
    return target


class TTMGluonTSPredictor(Predictor):
    """Wrapper to TTM that can be directly trained, validated, and tested with GluonTS datasets."""

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        model_path: str = "ibm-granite/granite-timeseries-ttm-r2",
        random_seed: int = 42,
        **kwargs,
    ):
        """Initialize a TTMGluonTSPredictor object.
        Args:
            context_length (int): Context length.
            prediction_length (int): Forecast length.
            model_path (str, optional): TTM Model path.. Defaults to "ibm-granite/granite-timeseries-ttm-r2".
            random_seed (int, optional): Seed. Defaults to 42.
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.random_seed = random_seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if "dropout" in kwargs and kwargs["dropout"] is None:
            del kwargs["dropout"]
        if "head_dropout" in kwargs and kwargs["head_dropout"] is None:
            del kwargs["head_dropout"]
        # Call get_model() function to load TTM model automatically.
        self.ttm = get_model(
            model_path,
            context_length=self.context_length,
            prediction_length=min(self.prediction_length, TTM_MAX_FORECAST_HORIZON),
            **kwargs,
        ).to(self.device)

    def predict(
        self,
        dataset: InputDataset,
        batch_size: int = 512,
        **kwargs,
    ):
        """Predict.
        Args:
            test_data_input (InputDataset): Test input dataset.
            batch_size (int, optional): Batch size. Defaults to 256.
        """
        while True:
            try:
                # Generate forecast samples
                forecast_samples = []
                metadata = []  # Capture metadata during first iteration
                for batch in batcher(dataset, batch_size=batch_size):
                    batch_ttm = {}
                    adjusted_batch_raw = []
                    for entry in batch:
                        metadata.append(entry)  # Store metadata here
                        # univariate array of shape (time,)
                        # multivariate array of shape (var, time)
                        # TTM supports multivariate time series
                        if len(entry["target"].shape) == 1:
                            entry["target"] = entry["target"].reshape(1, -1)
                        entry_context_length = entry["target"].shape[1]
                        num_channels = entry["target"].shape[0]
                        # Pad
                        if entry_context_length < self.ttm.config.context_length:
                            padding = torch.zeros(
                                (
                                    num_channels,
                                    self.ttm.config.context_length
                                    - entry_context_length,
                                )
                            )
                            adjusted_entry = torch.cat(
                                (padding, torch.tensor(impute_series(entry["target"]))),
                                dim=1,
                            )
                            # observed_mask[idx, :, :(ttm.config.context_length - entry_context_length)] = 0
                        # Truncate
                        elif entry_context_length > self.ttm.config.context_length:
                            adjusted_entry = torch.tensor(
                                impute_series(
                                    entry["target"][
                                        :, -self.ttm.config.context_length :
                                    ]
                                )
                            )
                        # Take full context
                        else:
                            adjusted_entry = torch.tensor(
                                impute_series(entry["target"])
                            )
                        adjusted_batch_raw.append(adjusted_entry)
                    # For TTM channel dimension comes at the end
                    batch_ttm["past_values"] = (
                        torch.stack(adjusted_batch_raw).permute(0, 2, 1).to(self.device)
                    )
                    # This statment won't ever be true in experiment settings
                    if self.prediction_length > TTM_MAX_FORECAST_HORIZON:
                        logger.error("It should not happen")
                        recursive_steps = int(
                            np.ceil(
                                self.prediction_length
                                / self.ttm.config.prediction_length
                            )
                        )
                        predict_outputs = torch.empty(len(batch), 0, num_channels).to(self.device)  # type: ignore
                        with torch.no_grad():
                            for i in range(recursive_steps):
                                model_outputs = self.ttm(**batch_ttm)
                                batch_ttm["past_values"] = torch.cat(
                                    [
                                        batch_ttm["past_values"],
                                        model_outputs["prediction_outputs"],
                                    ],
                                    dim=1,
                                )[:, -self.ttm.config.context_length :, :]
                                predict_outputs = torch.cat(
                                    [
                                        predict_outputs,
                                        model_outputs["prediction_outputs"][
                                            :, : self.ttm.config.prediction_length, :
                                        ],
                                    ],
                                    dim=1,
                                )
                    else:
                        model_outputs = self.ttm(**batch_ttm)
                        predict_outputs = model_outputs.prediction_outputs
                    # Trim prediction length to the desired length for experiment
                    predict_outputs = predict_outputs[
                        :, : self.prediction_length, :
                    ]
                    # Accumulate all forecasts
                    forecast_samples.append(predict_outputs.detach().cpu().numpy())
                # list to np.ndarray
                forecast_samples = np.concatenate(forecast_samples)
                if forecast_samples.shape[2] == 1:
                    forecast_samples = np.squeeze(forecast_samples, axis=2)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                )
                batch_size //= 2
                metadata = []  # Reset metadata on retry
        # Convert forecast samples into gluonts SampleForecast objects
        #   Array of size (num_samples, prediction_length) (1D case) or
        #   (num_samples, prediction_length, target_dim) (multivariate case)
        sample_forecasts: list[SampleForecast] = []
        for item, ts in zip(forecast_samples, metadata):
            sample_forecasts.append(
                SampleForecast(
                    item_id=ts["item_id"],
                    samples=np.expand_dims(item, axis=0),
                    start_date=ts["start"] + len(ts["target"]),
                )
            )
        return iter(sample_forecasts)
