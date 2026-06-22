#!/bin/bash

# ─────────────────────────────────────────────
#  setup.sh — one-time environment setup
#  usage: bash setup.sh
# ─────────────────────────────────────────────

set -e  # exit immediately on any error

HF_REPO="your-username/my-transformer"   # <-- change this to your HF repo
MODEL_PATH="bin/model/model.pt"
CONFIG_PATH="bin/data/config.pkl"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Transformer Training Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: Install dependencies ─────────────
echo "[1/4] Installing dependencies..."
pip install -r data/requirements.txt --break-system-packages -q
pip install huggingface_hub tiktoken rich --break-system-packages -q
echo "      ✓ Dependencies installed"
echo ""

# ── Step 2: Create local directories ─────────
echo "[2/4] Creating local directories..."
mkdir -p bin/model
mkdir -p bin/data
mkdir -p data
echo "      ✓ Directories ready"
echo ""

# ── Step 3: Hugging Face login ────────────────
echo "[3/4] Hugging Face authentication..."
echo "      Paste your HF write token when prompted."
echo "      Get it from: https://huggingface.co/settings/tokens"
echo ""
huggingface-cli login
echo ""

# ── Step 4: Pull latest checkpoint from HF ───
echo "[4/4] Downloading latest checkpoint from Hugging Face..."
echo ""

python3 - <<EOF
import os
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

REPO_ID = "${HF_REPO}"
MODEL_PATH = "${MODEL_PATH}"
CONFIG_PATH = "${CONFIG_PATH}"

def try_download(filename, local_path):
    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=os.path.dirname(local_path),
        )
        print(f"      ✓ {filename} downloaded → {local_path}")
    except EntryNotFoundError:
        print(f"      ⚠  {filename} not found in repo — skipping (first run?)")
    except RepositoryNotFoundError:
        print(f"      ✗  Repo '{REPO_ID}' not found — check HF_REPO in setup.sh")
        raise
    except Exception as e:
        print(f"      ✗  Failed to download {filename}: {e}")

try_download("model.pt", MODEL_PATH)
try_download("config.pkl", CONFIG_PATH)
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete."
echo ""
echo "  To start training:"
echo "    python -m main.train"
echo ""
echo "  To run inference:"
echo "    python -m main.load"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
