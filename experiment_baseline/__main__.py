# based on https://github.com/abdulfatir/gift-eval/blob/9ad8d03e3292ecd869e23c03d6f285ed9ffa5370/notebooks/naive.ipynb

import inspect
from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Type
import logging

from gluonts.time_feature import get_seasonality
import numpy as np
import pandas as pd
from gluonts.core.component import validated
from gluonts.dataset import Dataset
from gluonts.dataset.util import forecast_start
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.transform.feature import LastValueImputation, MissingValueImputation

from utils_exp import Args, MLExperimentFacade, get_parser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)
from gluonts.ext.statsforecast import (
    AutoARIMAPredictor,
    AutoETSPredictor,
    AutoThetaPredictor,
    NaivePredictor,
    SeasonalNaivePredictor,
)

def main():
    logger.info("Initializing experiment")
    parser = get_parser()

    args = parser.parse_args(namespace=Args)

    season_length = get_seasonality('M')
    matching_models = {
        "naive": NaivePredictor,
        "seasonal_naive": SeasonalNaivePredictor,
        "auto_arima": AutoARIMAPredictor,
        "auto_ets": AutoETSPredictor,
        "auto_theta": AutoThetaPredictor,
    }
    selected_model = matching_models.get(args.model_name)
    assert selected_model
    predictor_args = {
        "prediction_length": args.prediction_length,
        "season_length": season_length,
        "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }

    if args.model_name == "auto_ets":
        predictor_args["model"] = "AZN"

    predictor = selected_model(**predictor_args)
    exp = MLExperimentFacade(
        experiment_name=args.experiment_name,
        artifacts_path=args.artifacts_path,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
    )

    exp.run_experiment(
        model_name=args.model_name,
        dataset_path=args.data_path,
        predictor=predictor,
        num_samples=args.num_samples,
    )

if __name__ == "__main__":
    main()