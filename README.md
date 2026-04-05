# Gemma 4 for mlx-lm

Run Gemma 4 models on Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm).

## Supported Models

| Model | model_type | PLE | MoE | KV Sharing | k_eq_v |
|-------|-----------|-----|-----|------------|--------|
| gemma-4-E2B | `gemma4` | Yes | No  | No  | Yes |
| gemma-4-E4B | `gemma4` | Yes | No  | No  | Yes |
| gemma-4-31B | `gemma4` | No  | No  | Yes | Yes |
| gemma-4-26B-A4B | `gemma4` | No  | Yes | Yes | Yes |

## Installation

### 1. Environment Setup

```bash
# Create conda environment (recommended)
conda create -n mlx_gemma python=3.11
conda activate mlx_gemma

# Install required packages
pip install mlx mlx-lm
```

### 2. Install Model Files

```bash
# Find the mlx-lm models directory
MODELS_DIR=$(python3 -c "import mlx_lm; from pathlib import Path; print(Path(mlx_lm.__file__).parent / 'models')")

# Copy the files
cp gemma4_text.py gemma4.py "$MODELS_DIR/"

# Verify installation
python3 -c "from mlx_lm.models import gemma4_text, gemma4; print('Installation successful')"
```

Or use the install script:

```bash
bash install.sh
```

### 3. Download and Convert a Model

```bash
# Option A: Use a pre-converted model from mlx-community (if available)
python3 -m mlx_lm.generate --model mlx-community/gemma-4-E2B-it-4bit --prompt "Hello"

# Option B: Convert from HuggingFace
python3 -m mlx_lm.convert \
    --hf-path google/gemma-4-E2B-it \
    --mlx-path ./gemma-4-e2b-q4 \
    -q
```

## Usage

### CLI

```bash
# Text generation
python3 -m mlx_lm.generate \
    --model ./gemma-4-e2b-q4 \
    --max-tokens 500 \
    --prompt "Explain quantum computing in simple terms."

# Interactive chat
python3 -m mlx_lm.chat --model ./gemma-4-e2b-q4
```

### Python API

```python
from mlx_lm import load, generate

model, tokenizer = load("./gemma-4-e2b-q4")

prompt = "What is machine learning?"
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)

response = generate(model, tokenizer, prompt=formatted, max_tokens=500)
print(response)
```

## Validation

### Basic validation (weight mapping + architecture)

```bash
python3 validate_gemma4.py --model ./gemma-4-e2b-q4
```

### Weight key analysis

```bash
python3 validate_gemma4.py --model ./gemma-4-e2b-q4 --keys
```

### Inference validation

```bash
python3 validate_gemma4.py --model ./gemma-4-e2b-q4 --inference
```

### Numerical comparison against HF (optional)

```bash
# Requires transformers + torch
pip install transformers torch

python3 validate_gemma4.py \
    --model ./gemma-4-e2b-q4 \
    --numerical \
    --hf-model google/gemma-4-E2B-it
```

## Validation Checklist

Use this checklist when reproducing the setup on a new machine:

### Required

- [ ] `python3 -c "from mlx_lm.models import gemma4; print('OK')"` succeeds
- [ ] `validate_gemma4.py --model <path>` weight mapping passes
- [ ] `validate_gemma4.py --model <path> --inference` inference passes

### Numerical Accuracy

- [ ] Top-1 token matches HF transformers (bf16 baseline)
- [ ] Cosine similarity > 0.95 (bf16) or > 0.85 (q4)

### Architecture Mapping

| HuggingFace (PyTorch) | MLX (this implementation) |
|---|---|
| `Gemma4TextScaledWordEmbedding` | `nn.Embedding` + `* hidden_size**0.5` scaling |
| `Gemma4RMSNorm(with_scale=True)` | `nn.RMSNorm` (1+weight convention) |
| `Gemma4RMSNorm(with_scale=False)` | `RMSNoScale` (no learnable weight) |
| `Gemma4TextAttention` | `Attention` (auto sliding/full dispatch) |
| `Gemma4TextMLP` | `MLP` (auto double-wide dispatch) |
| `Gemma4TextDecoderLayer` | `TransformerBlock` |
| `Gemma4TextRouter` + `Gemma4TextExperts` | `Router` + `Experts` |
| `per_layer_input_gate` + `per_layer_projection` | Same names |
| `layer_scalar` | Same name (buffer -> parameter) |

## Known Limitations

1. **Text-only**: Vision and audio towers are stripped. Image/audio inputs are not supported.
2. **RMSNorm convention**: MLX's `nn.RMSNorm` uses `(1 + weight) * x` internally via `mx.fast.rms_norm`.
   This matches how mlx-lm handles Gemma 3, so weight loading should be consistent.
3. **Quantization differences**: Q4-quantized models may produce slightly different outputs compared to bf16.

## Troubleshooting

### "Model type gemma4 not supported"
Ensure `gemma4.py` and `gemma4_text.py` are correctly copied into `mlx_lm/models/`.

### "Model type gemma4_text not supported"
Some text-only checkpoints set `model_type` to `gemma4_text` in config.json. Both files must be installed.

### Output is garbled or repetitive
Check the RMSNorm weight convention. Run `validate_gemma4.py --numerical` for numerical verification.

### Out of memory
Use `--max-kv-size 512` to limit KV cache size.

## File Structure

```
gemma4_text.py      # Core text model (Attention, MLP, MoE, PLE, etc.)
gemma4.py           # Multimodal wrapper (text_config extraction, weight sanitization)
validate_gemma4.py  # Validation script
install.sh          # One-command installer
README.md           # This file
```

## Tested Environments

### Hardware

| Machine | Chip | Unified Memory | Model | Quant | Peak Memory | Generation Speed |
|---------|------|---------------|-------|-------|-------------|-----------------|
| MacBook Air M5 | Apple M5 | 24 GB | gemma-4-E2B-it | Q4 | 2.66 GB | ~80 tok/s |
| MacBook Pro M5 | Apple M5 Pro | 36 GB | gemma-4-E2B-it | Q4 | — | — |

### Software

- macOS 25 (Tahoe) / macOS 15.3+ (Sequoia)
- Python 3.11+
- mlx >= 0.22.0
- mlx-lm >= 0.31.0

## License

This implementation is licensed under the Apache License 2.0.
The Gemma 4 model weights are subject to [Google's Gemma license](https://ai.google.dev/gemma/terms).
