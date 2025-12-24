#!/bin/bash

# Full pipeline script
# Sử dụng: bash scripts/run.sh

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo Python path: $PYTHONPATH

set -e

python -m code.Raindrop
