# Transformers From Scratch

> Because using Hugging Face was too easy and we don't do easy things here.

A **decoder-only Transformer language model built entirely from scratch** in PyTorch — no pretrained weights, no `from transformers import BertForEverything`. Just raw math, a clean modular codebase, and an incremental daily pretraining workflow designed to accumulate knowledge across datasets over time.

---

## Architecture

| Component | Implementation |
|---|---|
| Tokenizer | GPT-2 BPE via `tiktoken` — fixed vocab of 50,257, never rebuilt |
| Embedding | `nn.Embedding(vocab_size, dim)` with weight tying to output projection |
| Positional Encoding | Sinusoidal, sized to `max_seq_len=256` |
| Attention | Causal multi-head self-attention with scaled dot-product + upper-triangular mask |
| Normalization | `RMSNorm` (pre-norm — applied before each sublayer, not after) |
| FFN | SwiGLU — `w_down(silu(w_gate(x)) * w_value(x))`, no bias |
| Depth | 6 stacked `TransformerBlock` layers via `nn.ModuleList` |
| Output | `nn.Linear(dim, vocab_size)` — weight-tied to embedding table |
| Optimizer | AdamW — `lr=1e-3`, `weight_decay=0.1`, `betas=(0.9, 0.95)` |
| Grad clipping | `clip_grad_norm_` at `1.0` |
| Scheduler | `CosineAnnealingLR(T_max=100, eta_min=1e-5)` |

**Config:** `dim=256`, `head=4`, `ntbl=6`, `dropout=0.1`, `seq_len=256`, `vocab_size=50257` — ~17M parameters

---

## How Pre-norm Works

```
x → RMSNorm → Attention → + residual → x
x → RMSNorm → SwiGLU FFN → + residual → x
```

Modern transformers (GPT-2 onward) normalize *before* each sublayer, not after. Produces more stable gradients as depth increases — especially important when stacking 6 blocks.

---

## Project Structure

```
transformers/
├── main/
│   ├── train.py                         # Daily training entry point
│   ├── load.py                          # Inference entry point
│   ├── transformer_orch/
│   │   ├── _attention.py                # Causal multi-head self-attention
│   │   ├── _embedding.py                # Token embedding table
│   │   ├── _positional_embedding.py     # Sinusoidal positional encoding
│   │   ├── _post_attention.py           # Pre-norm block (RMSNorm + Attention + SwiGLU)
│   │   ├── _swiglu_activation.py        # SwiGLU FFN
│   │   ├── _transformer.py              # Output projection head
│   │   ├── _transformer_block.py        # Single transformer block wrapper
│   │   └── _model_orc.py               # Model orchestrator + weight tying
│   ├── seq2seq/
│   │   └── _gpt2_tokenizer.py           # TokenCodec wraps tiktoken GPT-2 encoding
│   ├── config/
│   │   └── _model_config.py             # All config classes
│   ├── fine_tune/
│   │   └── _fine_tune_model.py          # FineTuneModel — load, train, save, HF sync
│   └── generator_config/
│       ├── _generator_api.py            # Generator — top-p sampling + EOS stopping
│       └── _load_config_and_model.py    # PretrainedHandler — load for inference
│
├── data/
│   ├── shakespeare.txt                  # Default training corpus
│   └── requirements.txt
│
├── setup/
│   └── setup.sh                         # One-time environment setup
│
└── bin/
    ├── model/                           # Local checkpoint cache (gitignored)
    └── data/                            # config.pkl
```

---

## Daily Pretraining Workflow

The core idea: train on one text file per day, save a checkpoint, continue from it the next day on a new file. The GPT-2 tokenizer's fixed vocabulary means checkpoints are always compatible regardless of which dataset you switch to.

```
Day 1 — no checkpoint exists:
  python -m main.train   →  initializes fresh model, trains, saves to HF

Day N — checkpoint exists on HF:
  python -m main.train   →  downloads from HF, continues training, pushes updated checkpoint
```

To change today's dataset, edit one line in `main/train.py`:
```python
FILE_PATH = 'data/your_new_file.txt'
```

Checkpoints are stored on Hugging Face Hub — not GitHub, since model files are 400MB+. Every upload creates a new commit so you can roll back to any previous session by commit hash.

---

## Quick Start

```bash
git clone https://github.com/afngh/transformers.git
cd transformers

bash setup/setup.sh   # install deps + HF login (once per machine)

python -m main.train  # start training
python -m main.load   # run inference
```

---

## Inference

```python
from main.generator_config._load_config_and_model import PretrainedHandler

handler = PretrainedHandler('bin/model/model.pt', 'bin/data/config.pkl')
model, config = handler.load()
client = handler.client(model, config, require_params=True, temperature=0.8, max_tokens=100)

print(client.generate_response("to be or not to be"))
```

Generation uses top-p nucleus sampling with temperature scaling. Stops at `<|endoftext|>` (GPT-2 EOS token, id=50256) or `max_tokens`, whichever comes first.

---

## Why GPT-2 Tokenizer

The original implementation used word-level tokenization built from whatever file was currently loaded — meaning every new dataset produced a different vocabulary and a structurally incompatible model. Switching files meant retraining from scratch.

GPT-2's BPE tokenizer has a fixed vocabulary of 50,257 tokens trained once on a large web corpus. Any English text encodes into the same token ID space, making checkpoint continuity across datasets possible by design.

---

## Key Design Decisions

**Weight tying** — the output projection shares its weight matrix with the token embedding table. Saves ~13M parameters at this vocab size and improves small-model quality by learning one consistent token representation space instead of two.

**Pre-norm over post-norm** — normalization before each sublayer produces more stable gradients at depth, especially during the first few hundred training steps of each session when Adam's momentum estimates are cold.

**Full-sequence loss** — targets are input shifted by one position, so every token contributes a loss signal simultaneously. Gives ~128x more gradient signal per batch compared to predicting only the final token.

---

## Requirements

```
torch
tiktoken
rich
huggingface_hub
```

---

## License

MIT — see `LICENSE` file.

---

> *"What a piece of work is a man, how noble in reason... also please predict the next token."*
> — Shakespeare, probably, if he knew about transformers