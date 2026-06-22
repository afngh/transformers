#!/bin/bash

# ─────────────────────────────────────────────
#  setup.sh — one-time environment setup
#  usage: bash setup/setup.sh
# ─────────────────────────────────────────────

HF_REPO="afnhf/my-transformer"   # your HF repo

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Transformer Training Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Step 1: Install dependencies ─────────────
echo "[1/3] Installing dependencies..."
pip install -r data/requirements.txt --break-system-packages -q
echo "      ✓ Done"
echo ""

# ── Step 2: Create local directories ─────────
echo "[2/3] Creating directories..."
mkdir -p bin/model
mkdir -p bin/data
mkdir -p data
echo "      ✓ Done"
echo ""

# ── Step 3: Hugging Face login ────────────────
echo "[3/3] Hugging Face login..."
echo "      Get your write token from: https://huggingface.co/settings/tokens"
echo ""
huggingface-cli login

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete."
echo ""
echo "  To train:    python -m main.train"
echo "  To generate: python -m main.load"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""