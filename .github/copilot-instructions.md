# Copilot Instructions for Ribeiro2026 Codebase

## Project Overview
This is a Time Series Forecasting (TSFM) experiment framework comparing multiple foundation models (Chronos, LagLLaMA, Moirai, TimesFM, TTMs) and baseline statistical methods against epidemiological data from SINAN/BR (Brazilian dengue, zika, chikungunya, and other diseases).

### Architecture
- **Separated Experiment Folders**: Each TSFM lives in `experiment_*` because of conflicting package dependencies. Each has its own `requirements.txt`, `setup.sh` (downloads HF model weights), and `__main__.py` entry point.
- **Unified Evaluation**: `utils_exp/` provides shared evaluation logic (`MLExperimentFacade`, `TSFMExperimentSplitter`) and CLI infrastructure.
- **Containerized Execution**: All experiments run in Docker with build-arg `EXPERIMENT_PATH` selecting which experiment to build (see [Dockerfile](Dockerfile#L1-L10)).
- **Data Pipeline**: Raw data from SINAN → extracted by `data_extractor/` → processed `.parquet` files → fed to experiments.

## Experiment Pattern
Every experiment follows this flow ([experiment_baseline/__main__.py](experiment_baseline/__main__.py) as reference):
1. Parse args via [utils_exp/cli.py](utils_exp/cli.py)
2. Instantiate predictor (GluonTS-compatible, wraps TSFM model)
3. Create `MLExperimentFacade` with experiment name, artifacts path, prediction/context lengths
4. Call `run_experiment()` which orchestrates:
   - Load parquet dataset
   - Split using `TSFMExperimentSplitter` (context window + rolling horizon validation)
   - Generate forecasts
   - Log metrics to MLflow

**Key Wrappers**: Some models need a GluonTS `RepresentablePredictor` wrapper (e.g., [ChronosPredictor](experiment_chronos/__main__.py#L19) for Chronos model).

## Data & Evaluation
- **Dataset Format**: `.parquet` files with pandas-style time series. Loaded via `FileDataset(path, freq="M")` (monthly frequency).
- **Splitter Logic** ([utils_exp/splitter.py](utils_exp/splitter.py)): Rolling window validation—maximizes context window up to `context_length`, then rolls forward by `prediction_length`.
- **Metrics**: GluonTS metrics (MASE, NRMSE, MSIS) computed per disease and aggregated; logged via MLflow to artifacts path.

## Running Experiments
```bash
# All experiments (set datasets, models, prediction/context lengths in run-experiments.sh)
export local_artifacts=CHOOSE_A_PATH data_path=./data
chmod +x ./run-experiments.sh && ./run-experiments.sh

# Single experiment via Docker
export local_artifacts=... data_path=... prediction_length=24 context_length=32
docker build -t chronos --build-arg EXPERIMENT_PATH="experiment_chronos" .
docker run --gpus all -v ~/data:/data -v ${local_artifacts}:/artifacts chronos \
  --experiment_name="my_exp" --artifacts_path="/artifacts" \
  --prediction_length=24 --context_length=32 \
  --model_name="chronos-t5-large" --data_path="/data/DENGBR.parquet" \
  --num_samples=20 --model_path="chronos-t5-large"
```

## Critical Conventions
- **Model Names**: Folder suffix becomes model key in experiments (e.g., `experiment_lagllama` → model name "lagllama" in CLI).
- **Quantile Levels**: Baseline uses `[0.1, 0.2, ..., 0.9]`; ensure consistency across wrappers.
- **Seasonality**: Computed via `get_seasonality('M')` for monthly data (currently hardcoded—see [experiment_baseline/__main__.py](experiment_baseline/__main__.py#L34)).
- **MLflow Experiment ID**: Reused if experiment exists; artifacts stored in specified path.
- **Evaluation Axis**: Computed both per-disease (`axis=None`) and aggregated (`axis=1`).

## Key Files
- [utils_exp/cli.py](utils_exp/cli.py): Standard CLI interface; add args here for cross-experiment features.
- [utils_exp/experiment_facade.py](utils_exp/experiment_facade.py): MLflow orchestration, evaluation entry point.
- [utils_exp/splitter.py](utils_exp/splitter.py): Custom splitter—modify here to change train/test split logic.
- [experiment_baseline/__main__.py](experiment_baseline/__main__.py): Template for new experiment implementations.
- [Dockerfile](Dockerfile): Multi-stage build; uses `EXPERIMENT_PATH` to select experiment code.
- [run-experiments.sh](run-experiments.sh): Master script; defines datasets, model configs, prediction/context lengths.

## Development Workflow
1. **Adding a new TSFM**: Create `experiment_newmodel/` with `__main__.py`, `setup.sh`, `requirements.txt`.
2. **Modifying evaluation**: Centralize in `MLExperimentFacade._evaluate()`.
3. **Testing locally** (not recommended but possible): Install `utils_exp` + experiment deps, mock dataset path.
4. **Debugging Docker builds**: Check `logs/build_logs_*.txt` and use `docker run -it <image> bash`.
