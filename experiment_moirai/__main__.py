import logging

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

from utils_exp import Args, MLExperimentFacade, get_parser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__package__)

class MoiraiArgs(Args):
    patch_size: int

def main():
    logger.info("Initializing experiment")
    parser = get_parser()

    parser.add_argument(
        "--patch_size",
        choices=["auto", 8, 16, 32, 64, 128],
        default="auto",
        help="Patch size for Moirai model",
    )


    args = parser.parse_args(namespace=MoiraiArgs)

    exp = MLExperimentFacade(
        experiment_name=args.experiment_name,
        artifacts_path=args.artifacts_path,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        frequency=args.frequency,
    )


    # Prepare pre-trained model by downloading model weights from huggingface hub
    if args.model_name.startswith("moirai-moe"):
        model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"/models/{args.model_path}"),
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            patch_size=8 if args.patch_size == "auto" else args.patch_size,
            num_samples=args.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    elif args.model_name.startswith("moirai"):
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"/models/{args.model_path}"),
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            patch_size=args.patch_size,
            num_samples=args.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    else:
        raise ValueError(f"Model {args.model_name} not supported")
    logger.info(f"Creating predicor")
    predictor = model.create_predictor(batch_size=args.batch_size)

    exp.run_experiment(
        model_name=args.model_name,
        dataset_path=args.data_path,
        predictor=predictor,
        num_samples=args.num_samples,
        save_predictions=args.save_predictions,
    )


if __name__ == "__main__":
    main()