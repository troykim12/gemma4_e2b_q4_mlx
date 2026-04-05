#!/bin/bash
# Gemma 4 for mlx-lm — Install Script
# Usage: bash install.sh

set -e

echo "========================================"
echo "  Gemma 4 for mlx-lm — Installer"
echo "========================================"

# Find mlx-lm install path
MODELS_DIR=$(python3 -c "
import mlx_lm
from pathlib import Path
print(Path(mlx_lm.__file__).parent / 'models')
" 2>/dev/null)

if [ -z "$MODELS_DIR" ]; then
    echo "[FAIL] mlx-lm is not installed."
    echo "       Run: pip install mlx-lm"
    exit 1
fi

echo "[INFO] Target directory: $MODELS_DIR"

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Copy model files
cp "$SCRIPT_DIR/gemma4_text.py" "$MODELS_DIR/"
cp "$SCRIPT_DIR/gemma4.py" "$MODELS_DIR/"

echo "[OK]   Files copied successfully."

# Verify imports
python3 -c "from mlx_lm.models import gemma4_text, gemma4; print('[OK]   Import verification passed.')"

echo ""
echo "========================================"
echo "  Installation complete!"
echo "========================================"
echo ""
echo "Usage:"
echo "  python3 -m mlx_lm.generate --model <gemma4-model-path> --prompt 'Hello'"
echo ""
echo "Validation:"
echo "  python3 validate_gemma4.py --model <gemma4-model-path>"
