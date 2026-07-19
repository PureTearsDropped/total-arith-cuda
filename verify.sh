#!/usr/bin/env bash
# GPU total-arithmetic self-test (needs CUDA; CPU fallback keeps correctness).
cd "$(dirname "$0")"; exec "${PY:-python3}" cuda_total.py
