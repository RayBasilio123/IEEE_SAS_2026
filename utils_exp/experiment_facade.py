from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.client
import mlflow.experiments
from gluonts.dataset.common import FileDataset
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss, MSIS, NRMSE
from utils_exp.splitter import TSFMExperimentSplitter
import numpy as np
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
        splitter: AbstractBaseSplitter | None = None,
    ):
        self.experiment_name = experiment_name
        self.artifacts_path = artifacts_path
        self.context_length = context_length
        self.splitter = splitter or TSFMExperimentSplitter(
            context_length=context_length
        )
        self.prediction_length = prediction_length
        if (exp := mlflow.get_experiment_by_name(self.experiment_name)) is None:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name, artifact_location=self.artifacts_path
            )
        else:
            self.experiment_id: str = exp.experiment_id

    def run_experiment(
        self, model_name: str, dataset_path, predictor: Predictor, num_samples: int
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
            }
        )

        dataset = self.get_data(dataset_path)
        logger.info(f"Data loaded from {dataset_path}")
        forecast_it, ts_it = self.make_evaluation_predictions(
            predictor=predictor, dataset=dataset, num_samples=num_samples
        )
        forecast = list(forecast_it)
        metrics_disease = self._evaluate(forecast, ts_it, model_name, self.experiment_name, self.context_length, self.prediction_length, axis=None)
        metrics_all = self._evaluate(forecast, ts_it, model_name, self.experiment_name, self.context_length, self.prediction_length,  axis=1)
        logger.info(f"Metrics calculated")
        mlflow.log_artifact(metrics_disease)
        mlflow.log_artifact(metrics_all)
        mlflow.end_run()

    def get_data(self, path: str) -> Dataset:
        return FileDataset(
            path=Path(path), 
            freq="M"
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
                ],
                axis=axis,
            )
        )
        metrics["Model"] = model_name
        metrics["Disease"] = disease_code
        metrics["Context"] = contex_length
        metrics["Prediction"] = prediction_length
        metrics = metrics.rename(
                columns={"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "CPRS", "NRMSE[0.5]": "NRMSE"}
        )
        if axis is not None:
            metrics.reset_index(inplace=True)
            metrics.rename(columns={"level_0": "item_id", "level_1": "forecast_start"}, inplace=True)
        sufix = "agg" if axis is None else "per_window"
        path = f"{self.artifacts_path}/metric_{sufix}.csv"
        metrics.to_csv(path, mode='a', index=False, header=not os.path.exists(path))
        return path