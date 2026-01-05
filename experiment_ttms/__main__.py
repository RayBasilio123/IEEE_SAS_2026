import logging

from experiment_ttms.ttm_gluonts_predictor import TTMGluonTSPredictor
from utils_exp import Args, MLExperimentFacade, get_parser

logger = logging.getLogger(__package__)


def main():
    logger.info("Initializing experiment")
    parser = get_parser()
    args = parser.parse_args(namespace=Args)

    logger.info("Creating TTM GluonTS predictor")
    logger.warning(
        "The TTM model requires at least 512 context length and 96 prediction length. The padding will be done automatically."
    )
    ttm_predictor = TTMGluonTSPredictor(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        model_path=f"/models/{args.model_path}",
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
        predictor=ttm_predictor,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
