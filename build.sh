#!/bin/bash
# Azure Deployment Build Script for TensorFlow 2.12.1
# Special optimizations for .keras model loading

set -e  # Exit immediately on error

echo "=== Initializing Azure-Compatible Build ==="

# 1. System-level optimizations
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y \
    python3-dev \
    build-essential \
    libpython3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Python environment setup
python -m pip install --upgrade pip==23.1.2
pip install setuptools==65.5.0 wheel==0.37.1

# 3. Install with Azure-optimized flags
pip install --no-cache-dir \
    --default-timeout=300 \
    --implementation=py \
    --only-binary=:all: \
    -r requirements.txt

# 4. Post-install verification
echo "=== Verifying Critical Packages ==="
python -c "
import tensorflow as tf;
assert tf.__version__ == '2.12.1', f'Wrong TF version: {tf.__version__}';
from tensorflow.keras.models import load_model;
print('✓ TensorFlow', tf.__version__, 'verified');
import flask;
assert flask.__version__ == '2.3.2';
print('✓ Flask', flask.__version__, 'verified');
import numpy as np;
assert np.__version__ == '1.23.5';
print('✓ NumPy', np.__version__, 'verified')
"

# 5. Cleanup to reduce image size
find /usr/local/lib/python3.10 -type d -name '__pycache__' -exec rm -rf {} +
rm -rf /root/.cache/pip

echo "=== Build Successful - Azure Ready ==="