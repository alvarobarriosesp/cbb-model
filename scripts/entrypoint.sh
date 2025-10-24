#!/usr/bin/env bash
set -euo pipefail
echo "TASK=${TASK:-train}"
if [[ "${TASK:-train}" == "train" ]]; then
  python -u src/steps/train.py
elif [[ "${TASK}" == "predict" ]]; then
  python -u src/steps/predict.py
else
  echo "TASK desconocido: ${TASK} (usa train|predict)"; exit 2
fi
