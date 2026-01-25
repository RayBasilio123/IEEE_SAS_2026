from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils_exp.fixed_window_splitter import FixedWindowSplitter
import mlflow
import mlflow.client
import mlflow.experiments
from gluonts.dataset.common import FileDataset
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss, MSIS, NRMSE, MAE, SMAPE, RMSE, MAPE
from gluonts.itertools import batcher
from utils_exp.metrics import R2Score
from utils_exp.splitter import TSFMExperimentSplitter
import numpy as np
import pandas as pd
if TYPE_CHECKING:
    from collections.abc import Iterator

    from gluonts.dataset.common import Dataset
    from gluonts.dataset.split import AbstractBaseSplitter, TestData
    from gluonts.model.forecast import Forecast
    from gluonts.model.predictor import Predictor
    from pandas import DataFrame

import time

logger = logging.getLogger(__name__)


class MLExperimentFacade:
    def __init__(
        self,
        experiment_name: str,
        artifacts_path: str,
        prediction_length: int,
        context_length: int,
        frequency: str,
        splitter: AbstractBaseSplitter | None = None,
    ):
        self.experiment_name = experiment_name
        self.artifacts_path = artifacts_path
        self.context_length = context_length
        # self.splitter = splitter or TSFMExperimentSplitter(
        #     context_length=context_length
        # )
        self.splitter = splitter or FixedWindowSplitter(
            window_size=context_length,
            step_size=prediction_length,
        )
        self.frequency = frequency
        self.prediction_length = prediction_length
        if (exp := mlflow.get_experiment_by_name(self.experiment_name)) is None:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name, artifact_location=self.artifacts_path
            )
        else:
            self.experiment_id: str = exp.experiment_id

    def run_experiment(
        self, model_name: str, dataset_path, predictor: Predictor, num_samples: int, save_predictions: bool = True
    ) -> None:
        self.model_name = model_name
        mlflow.start_run(run_name=model_name, experiment_id=self.experiment_id)
        logger.info(f"Starting experiment {model_name}")
        mlflow.log_params(self.__dict__)
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_path": dataset_path,
                "num_samples": num_samples,
                "save_predictions": save_predictions,
            }
        )

        dataset = self.get_data(dataset_path, frequency=self.frequency)
        logger.info(f"Data loaded from {dataset_path}")
        forecast_it, ts_it = self.make_evaluation_predictions(
            predictor=predictor, dataset=dataset, num_samples=num_samples
        )
        forecast = list(forecast_it)
        
        # Save predictions before evaluation (test_data can be reused)
        if save_predictions:
            predictions_path = self._save_predictions(
                forecasts=forecast,
                test_data=ts_it,
                model_name=model_name,
                experiment_name=self.experiment_name,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
            )
            mlflow.log_artifact(predictions_path)
        
        metrics_disease = self._evaluate(forecast, ts_it, model_name, self.experiment_name, self.context_length, self.prediction_length, axis=None)
        metrics_all = self._evaluate(forecast, ts_it, model_name, self.experiment_name, self.context_length, self.prediction_length,  axis=1)
        logger.info(f"Metrics calculated")
        mlflow.log_artifact(metrics_disease)
        mlflow.log_artifact(metrics_all)
        mlflow.end_run()

    def get_data(self, path: str, frequency: str) -> Dataset:
        return FileDataset(
            path=Path(path), 
            freq=frequency
        )

    def make_evaluation_predictions(
        self,
        predictor: Predictor,
        dataset: Dataset,
        num_samples: int,
        distance: int | None = None,
    ) -> tuple[Iterator[Forecast], Any]:
        """
        Generate forecasts and time series from a dataset.

        This method was implemented because the default method in the GluonTS library assumes splitters that do not meet the requirements of this evaluation.

        Parameters
        ----------
        predictor
            Predictor to use.
        dataset
            Dataset to generate forecasts from.
        num_samples
            Number of samples to use when generating sample-based forecasts.
        distance
            Distance between windows. The default is the context length.

            Returns
            -------
            A tuple of forecasts and time series obtained by predicting `dataset` with `predictor`.
        """
        _, test_template = self.splitter.split(dataset)
        test_data: TestData = test_template.generate_instances(
            prediction_length=self.prediction_length,
            distance=distance,
        )
        return (
            predictor.predict(test_data.input, num_samples=num_samples),
            test_data,
        )

    def _evaluate(self, forecasts, ts, model_name: str, disease_code: str, contex_length: int, prediction_length: int, axis=None) -> str:
        result_rows = []
        logger.info(f"Evaluating forecasts")
        metrics = (
            evaluate_forecasts(
                forecasts,
                test_data=ts,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
                    MSIS(),
                    NRMSE(forecast_type="0.5"),
                    R2Score(forecast_type="0.5"),
                    MAE(), SMAPE(), RMSE(), MAPE()
                ],
                axis=axis,
            )
        )
        metrics["Model"] = model_name
        metrics["Disease"] = disease_code
        metrics["Context"] = contex_length
        metrics["Prediction"] = prediction_length
        metrics = metrics.rename(
            columns={"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "CPRS", "NRMSE[0.5]": "NRMSE", "R2[0.5]": "R2"}
        )
        if axis is not None:
            metrics.reset_index(inplace=True)
            metrics.rename(columns={"level_0": "item_id", "level_1": "forecast_start"}, inplace=True)
        sufix = "agg" if axis is None else "per_window"
        path = f"{self.artifacts_path}/metric_{sufix}.csv"
        metrics.to_csv(path, mode='a', index=False, header=not os.path.exists(path))
        return path

    def _save_predictions(
        self,
        forecasts: list[Forecast],
        test_data: TestData,
        model_name: str,
        experiment_name: str,
        context_length: int,
        prediction_length: int,
        batch_size: int = 100,
    ) -> str:
        """
        Save predictions along with actual values and metadata to a parquet file.

        Parameters
        ----------
        forecasts
            List of forecast objects from model predictions.
        test_data
            TestData object containing input and label data.
        model_name
            Name of the model.
        experiment_name
            Name of the experiment.
        context_length
            Length of the context window.
        prediction_length
            Length of the prediction horizon.
        batch_size
            Batch size for processing forecasts.

        Returns
        -------
        Path to the saved predictions file.
        """
        logger.info("Saving predictions with metadata")
        
        # Standard quantile levels used across experiments
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        rows = []
        window_index = 0
        
        # Batch iteration matching GluonTS evaluation pattern
        input_batches = batcher(test_data.input, batch_size=batch_size)
        label_batches = batcher(test_data.label, batch_size=batch_size)
        forecast_batches = batcher(forecasts, batch_size=batch_size)
        
        for input_batch, label_batch, forecast_batch in zip(
            input_batches, label_batches, forecast_batches
        ):
            # Process each window in the batch
            for input_entry, label_entry, forecast in zip(
                input_batch, label_batch, forecast_batch
            ):
                item_id = forecast.item_id
                forecast_start = forecast.start_date
                actual_values = label_entry["target"]
                
                # Extract forecast values
                predicted_mean = forecast.mean
                
                # Extract quantiles - handle both QuantileForecast and SampleForecast
                quantile_values = {}
                for q in quantile_levels:
                    try:
                        quantile_values[f"p{int(q*100)}"] = forecast.quantile(q)
                    except (AttributeError, KeyError):
                        # If quantile not available, set to None
                        quantile_values[f"p{int(q*100)}"] = None
                
                # Create a row for each horizon step
                for step in range(prediction_length):
                    row = {
                        "model_name": model_name,
                        "experiment_name": experiment_name,
                        "item_id": item_id,
                        "context_length": context_length,
                        "prediction_length": prediction_length,
                        "window_index": window_index,
                        "forecast_start": str(forecast_start),
                        "horizon_step": step + 1,  # 1-indexed for clarity
                        "actual_value": float(actual_values[step]) if step < len(actual_values) else None,
                        "predicted_mean": float(predicted_mean[step]),
                    }
                    
                    # Add quantile predictions
                    for q_name, q_values in quantile_values.items():
                        if q_values is not None:
                            row[f"predicted_{q_name}"] = float(q_values[step])
                        else:
                            row[f"predicted_{q_name}"] = None
                    
                    rows.append(row)
                
                window_index += 1
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Ensure predictions directory exists
        predictions_dir = Path(self.artifacts_path) / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to parquet with append logic
        predictions_path = predictions_dir / "predictions.parquet"
        
        if predictions_path.exists():
            # Append to existing file
            existing_df = pd.read_parquet(predictions_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_parquet(predictions_path, index=False)
        logger.info(f"Saved {len(rows)} prediction rows to {predictions_path}")
        
        return str(predictions_path)
