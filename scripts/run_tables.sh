#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for arch in ${ARCHES:-ViT-B/32 ViT-B/16}; do
  for dataset in ${DATASETS:-Caltech101 Pets Cars Flower102 Aircraft DTD eurosat UCF101}; do
    ARCH="$arch" DATASET="$dataset" bash "$ROOT/scripts/run_ttp_shards.sh"
  done
done
