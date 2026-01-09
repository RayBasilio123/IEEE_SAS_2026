# Get predictor from https://github.com/abdulfatir/gift-eval/blob/9ad8d03e3292ecd869e23c03d6f285ed9ffa5370/notebooks/chronos.ipynb
# with minor modifications
import logging
from dataclasses import dataclass, field
from typing import Iterator

from gluonts.model.predictor import RepresentablePredictor
import numpy as np
import torch
from chronos import BaseChronosPipeline, ForecastType
from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast, SampleForecast
from tqdm.auto import tqdm

from utils_exp import Args, MLExperimentFacade, get_parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)

class ChronosPredictor(RepresentablePredictor):
    def __init__(
        self,
        model_path,
        num_samples: int,
        prediction_length: int,
        *args,
        **kwargs,
    ):
        self.pipeline = BaseChronosPipeline.from_pretrained(
            f"/models/{model_path}",
            *args,
            **kwargs,
        )
        self.prediction_length = prediction_length
        self.num_samples = num_samples

    def predict(
        self, dataset, batch_size: int = 256, **kwargs
    ) -> Iterator[Forecast]:
        pipeline = self.pipeline
        predict_kwargs = (
            {"num_samples": self.num_samples}
            if pipeline.forecast_type == ForecastType.SAMPLES
            else {}
        )
        while True:
            try:
                # Generate forecast samples
                forecast_outputs = []
                metadata = []  # Capture metadata during first iteration
                for batch in batcher(dataset, batch_size=batch_size):
                    context = [torch.tensor(entry["target"]) for entry in batch]
                    for entry in batch:
                        metadata.append(entry)  # Store metadata here
                    forecast_outputs.append(
                        pipeline.predict(
                            context,
                            prediction_length=self.prediction_length,
                            **predict_kwargs,
                        ).numpy()
                    )
                forecast_outputs = np.concatenate(forecast_outputs)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {batch_size}, reducing to {batch_size // 2}"
                )
                batch_size //= 2
                metadata = []  # Reset metadata on retry

        # Convert forecast samples into gluonts Forecast objects using captured metadata
        forecasts = []
        for item, ts in zip(forecast_outputs, metadata):
            if pipeline.forecast_type == ForecastType.SAMPLES:
                forecasts.append(
                    SampleForecast(samples=item, start_date=ts["start"] + len(ts["target"]), item_id=ts["item_id"])
                )
            elif pipeline.forecast_type == ForecastType.QUANTILES:
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=item,
                        forecast_keys=list(map(str, pipeline.quantiles)),
                        start_date=ts["start"] + len(ts["target"]),
                        item_id=ts["item_id"],
                    )
                )

        return iter(forecasts)


def main():
    logger.info("Initializing experiment")
    parser = get_parser()
    args = parser.parse_args(namespace=Args)

    chronos_predictor = ChronosPredictor(
        model_path=args.model_path,
        prediction_length=args.prediction_length,
        num_samples=args.num_samples,
        device_map="cuda:0",
    )
    exp = MLExperimentFacade(
        experiment_name=args.experiment_name,
        artifacts_path=args.artifacts_path,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
    )
    exp.run_experiment(
        model_name=args.model_name,
        dataset_path=args.data_path,
        predictor=chronos_predictor,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()

