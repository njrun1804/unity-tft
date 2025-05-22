#!/usr/bin/env bash
export OMP_NUM_THREADS=$(sysctl -n hw.physicalcpu)
export MKL_NUM_THREADS=$OMP_NUM_THREADS
python "$@"
