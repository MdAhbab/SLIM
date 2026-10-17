#!/usr/bin/env bash
# One-time setup of the data-processing environment, inside WSL Ubuntu or any
# Linux machine. Installs Miniforge into ~/miniforge3 (skipped if present) and
# creates the conda environment `slimdata` with everything the scripts under
# scripts/data/ need: pyBigWig and bedtools for the tracks and pairs, UCSC
# liftOver and CrossMap for converting GRCh38 files to hg19, and a CPU build of
# PyTorch to write the .pt track files and run the tests.
#
#   bash scripts/data/setup_wsl.sh
#   conda activate slimdata        # in every new shell afterwards
#
# Safe to run again: existing pieces are kept.
set -euo pipefail

PREFIX="${HOME}/miniforge3"
ENV=slimdata

if [ ! -x "${PREFIX}/bin/conda" ]; then
    installer="$(mktemp -d)/Miniforge3.sh"
    curl -fsSL -o "${installer}" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash "${installer}" -b -p "${PREFIX}"
    rm -f "${installer}"
fi
# shellcheck disable=SC1091
source "${PREFIX}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV}"; then
    conda create -y -n "${ENV}" -c conda-forge -c bioconda \
        python=3.11 numpy scipy pandas scikit-learn pyyaml requests \
        matplotlib pytest pyfaidx pybigwig bedtools ucsc-liftover crossmap
fi
conda activate "${ENV}"

# The CPU wheel is far smaller than the CUDA one; this machine only processes
# data and runs the tests.
python -c "import torch" 2>/dev/null || \
    pip install --index-url https://download.pytorch.org/whl/cpu torch

# Shell start-up: make `conda activate slimdata` available in new shells.
grep -q "miniforge3/etc/profile.d/conda.sh" "${HOME}/.bashrc" || \
    echo "source ${PREFIX}/etc/profile.d/conda.sh" >> "${HOME}/.bashrc"

python - <<'EOF'
import pyBigWig, torch, yaml, numpy, shutil
print("pyBigWig", pyBigWig.numpy, "torch", torch.__version__)
for tool in ("bedtools", "liftOver", "CrossMap"):
    print(f"{tool:9s}", shutil.which(tool) or "MISSING")
EOF
