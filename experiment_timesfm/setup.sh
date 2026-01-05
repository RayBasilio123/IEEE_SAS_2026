#!/bin/bash
set -e

huggingface-cli download --repo-type model --cache-dir /root/.cache/huggingface --local-dir /models/timesfm-1.0 google/timesfm-1.0-200m
huggingface-cli download --repo-type model --cache-dir /root/.cache/huggingface --local-dir /models/timesfm-2.0 google/timesfm-2.0-500m-jax