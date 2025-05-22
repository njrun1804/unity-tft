#!/bin/zsh
# scripts/auto_resume_train.sh
# Usage: zsh scripts/auto_resume_train.sh [max_retries]

MAX_RETRIES=${1:-3}
COUNT=0
SUCCESS=0

while [[ $COUNT -lt $MAX_RETRIES ]]; do
    echo "[Auto-Resume] Training attempt $((COUNT+1))/$MAX_RETRIES..."
    python train_tft.py && SUCCESS=1 && break
    echo "[Auto-Resume] Training failed. Retrying in 10 seconds..."
    sleep 10
    COUNT=$((COUNT+1))
done

if [[ $SUCCESS -eq 1 ]]; then
    echo "[Auto-Resume] Training completed successfully."
else
    echo "[Auto-Resume] Training failed after $MAX_RETRIES attempts."
    exit 1
fi
