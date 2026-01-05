# Paper Code

This repository contains the implementation of the experiments for Time Series Foundation Models (TSFM) as described in the associated research paper. It includes all necessary scripts and dependencies to replicate the experiments and evaluate the models.

# Overall Structure

The project is organized into folders of each TSFM due to the conflict of package dependencies between TSFM libraries. Each TSFM is in a folder named "experiment_TSFM_NAME" and has its specific package dependencies and a script to download the model weights from Hugging Face. These folders contain an experiment pipeline and, when necessary, a class wrapper that creates a predictor class based on the GluonTS package to standardize experiment inputs and outputs.

The `utils_exp` package contains logic to split data and evaluate the models. Each model has its own Docker image, built for its specific needs using the `EXPERIMENT_NAME` argument in the Dockerfile, which selects the experiment to run and builds an image for it.

The Data Extractor has a distinct structure from the others and has its own Dockerfile.

## Dataset

The dataset is extracted from SINAN/BR. The code to fetch and manipulate the data is located in the `data_provider` folder. The processed dataset is available in the `data` folder.

## Usage

### Run All Experiments

The processed datasets are already in the project folder. If you want to run the Data Extractor, refer to the README inside its directory for instructions.

Make sure you have Docker installed and run the shell script as follows:

```sh
export local_artifacts=CHOOSE_A_PATH
export data_path=./data
chmod +x ./run-experiments.sh
./run-experiments.sh
```

### Run a Single Experiment

```sh
export local_artifacts=CHOOSE_A_PATH
export prediction_length=SELECTED_PREDICTION
export model_name=A_MODELNAME_IS_THE_SUFFIX_OF_THE_FOLDER_experiment
export model_key=SEE_RUN_EXPERIMENTS_TO_GET_KEY
export file=filename
docker build -t ${model_name} --build-arg EXPERIMENT_PATH="experiment_${model_name}" .
docker run --gpus all -v ~/data:/data -v ${local_artifacts}:/artifacts -e LOG_LEVEL='INFO' ${model_name} \
                    --experiment_name "${file}" \
                    --artifacts_path "/artifacts" \
                    --prediction_length "$prediction_length" \
                    --context_length "$context_length" \
                    --model_name "$model_key" \
                    --data_path "/data/${file}.parquet" \
                    --num_samples 20 \
                    --model_path "$model_path"
```

The CSV file with the results will be in the artifacts folder. You can use the `report` scripts to generate tables and figures.

### Run Outside Docker

We do not recommend this, so we do not explicitly explain how to do it. However, you can read the Dockerfiles and figure it out on your own.

### Using Dev Containers

You can also use dev containers to replicate the development environment and make changes. Dev containers provide a consistent and isolated environment for development, ensuring that all dependencies and tools are available. Refer to the `.devcontainer` folder for configuration details.

