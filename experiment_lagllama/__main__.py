import logging
import torch
from lag_llama.gluon.estimator import LagLlamaEstimator

from utils_exp.cli import get_parser
from utils_exp.dummy_module import create_dummy_gluonts_torch_module
from utils_exp.experiment_facade import MLExperimentFacade

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)

if __name__ == "__main__":
    logger.info("Initializing experiment")
    parser = get_parser()
    args = parser.parse_args()

    create_dummy_gluonts_torch_module()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prediction_length = args.prediction_length

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (args.context_length + prediction_length) / estimator_args["context_length"]),
    }
    estimator = LagLlamaEstimator(
        ckpt_path=args.model_path,
        prediction_length=prediction_length,
        context_length=args.context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        batch_size=1,
        num_parallel_samples=args.num_samples,
        device=device,
        rope_scaling=rope_scaling_arguments
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    exp = MLExperimentFacade(
        experiment_name=args.experiment_name,
        artifacts_path=args.artifacts_path,
        prediction_length=prediction_length,
        context_length=args.context_length,
        frequency=args.frequency,
    )
    exp.run_experiment(
        args.model_name,
        args.data_path,
        predictor,
        args.num_samples,
        save_predictions=args.save_predictions,
    )
