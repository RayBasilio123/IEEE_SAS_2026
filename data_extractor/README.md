# Data Extractor

This folder contains scripts to extract raw data from SINAN and save the processed data in selected folders.

## Running the Extractor

To run the extractor, ensure you have Docker installed and build the image using the following commands:

```sh
cd sinan_extractor
docker build -t sinan:latest .
```

To execute the extractor, use the following command:

```sh
export data_path=SELEC_A_PATH
docker run -v ${data_path}:/data -v ${local_artifacts}:/artifacts sinan:latest \
    --path=/data
```
