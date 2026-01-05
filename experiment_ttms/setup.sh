#!/bin/bash
set -e

huggingface-cli download --repo-type model --cache-dir /root/.cache/huggingface --local-dir /models/granite-timeseries-ttm-r1 ibm-granite/granite-timeseries-ttm-r1
huggingface-cli download --repo-type model --cache-dir /root/.cache/huggingface --local-dir /models/granite-timeseries-ttm-r2 ibm-granite/granite-timeseries-ttm-r2