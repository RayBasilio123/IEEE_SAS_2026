#!/bin/bash
set -e

sizes=("small" "base" "large")
for size in "${sizes[@]}"; do
    huggingface-cli download --repo-type model --cache-dir /root/.cache/huggingface --local-dir /models/moirai-1.1-R-$size Salesforce/moirai-1.1-R-$size
    if [ "$size" != "large" ]; then
        huggingface-cli download --repo-type model --cache-dir /root/.cache/huggingface --local-dir /models/moirai-moe-1.0-R-$size Salesforce/moirai-moe-1.0-R-$size
    fi
done