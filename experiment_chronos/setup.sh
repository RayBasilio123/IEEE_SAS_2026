#!/bin/bash
set -e
sizes=("tiny" "mini" "small" "base" "large")

for size in "${sizes[@]}"; do
    huggingface-cli download --repo-type model --cache-dir /root/.cache/huggingface --local-dir /models/chronos-t5-${size} amazon/chronos-t5-${size}
    
    # if [ "$size" != "large" ]; then
    #     huggingface-cli download --repo-type model --cache-dir /root/.cache/huggingface --local-dir /models/chronos-bolt-${size} amazon/chronos-bolt-${size}
    # fi
done