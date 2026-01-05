#!/bin/bash
set -e
huggingface-cli download --local-dir /models time-series-foundation-models/Lag-Llama lag-llama.ckpt