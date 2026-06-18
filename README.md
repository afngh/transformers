# 🤖 Transformers From Scratch
### Because using Hugging Face was too easy and we don't do easy things here.

---

## What Is This?

A **decoder-only Transformer language model built entirely from scratch** in PyTorch, trained on Shakespeare's text. No pretrained weights. No `from transformers import BertForEverything`. Just raw math, a proper project structure, and an increasingly sophisticated understanding of what "from scratch" actually means.

This isn't the notebook anymore. This is the **refactored, modular, production-structured version** — every component lives in its own file, every config has its own class, and the whole thing imports like a real Python package.

---

## What Changed (The Glow-Up)

| Before | After |
|---|---|
| Single notebook, everything in one cell | Proper package structure under `main/` |
| Character-level tokenization | Word-level tokenization with special tokens |
| ReLU + LayerNorm | **SwiGLU + RMSNorm** (yes, like LLaMA) |
| Broken attention scaling, no mask | Fixed scaling + causal mask ✓ |
| Raw `generate_response()` function | `Generator` class with EOS stopping |
| Magic numbers scattered everywhere | Config classes for everything |
| `shakespeare.txt` in root | Clean `data/` directory with `requirements.txt` |

---

## Architecture

Because reading "Attention Is All You Need" once wasn't enough — we went ahead and *implemented* it. Then we upgraded it.

| Component | What It Does | What It Actually Does |
|---|---|---|
| `Embedding` | Maps tokens to dense vectors | Turns words into vibes |
| `PositionalEmbedding` | Injects position via sinusoids | Tells the model where in the sentence it is, because apparently it can't count |
| `Attention` | Causal multi-head self-attention | Every token gossips about every *past* token. Future tokens are none of its business. |
| `SwiGLUFFN` | Gated feed-forward network | The activation function LLaMA uses. We also use it now. No big deal. |
| `PostAttention` | RMSNorm + SwiGLU FFN | Cleans up the mess attention made, with modern normalization |
| `Transformer` | Output projection | The bouncer who decides what word comes next |
| `Model` | Orchestrates everything | The guy who says "trust me, it all works together" |

---

## How It Works

```
Input Words → Embedding → Positional Encoding → Causal Multi-Head Attention
→ RMSNorm → SwiGLU FFN → RMSNorm → Output Logits → Next Word
```

Or in plain English:

> "Give me Shakespeare. I will stare at 10 words at a time, very hard, with a causal mask so I don't cheat.
> Then I will guess what word comes next. Repeat. Profit."

---

## Project Structure

```
transformers/
├── main/
│   ├── model.py                        # The entry point — training loop, configs, inference
│   ├── transformer_orch/               # Every transformer component, one file each
│   │   ├── _attention.py               # Causal multi-head self-attention
│   │   ├── _embedding.py               # Token embeddings
│   │   ├── _positional_embedding.py    # Sinusoidal positional encoding
│   │   ├── _post_attention.py          # RMSNorm + SwiGLU FFN (the modern stack)
│   │   ├── _swiglu_activation.py       # SwiGLU: w_gate ⊙ silu(w_value) → w_down
│   │   ├── _transformer.py             # Output projection head
│   │   └── _model_orc.py              # Model orchestrator — wires everything together
│   ├── seq2seq/
│   │   └── _enc_dec.py                 # EncDec: encode words → indices, decode back
│   └── generator_config/
│       └── _generator_api.py           # Generator class with top-p sampling + EOS stop
│
├── data/
│   ├── shakespeare.txt                 # The corpus. 37 plays. 1 training set.
│   ├── requirements.txt               # torch, matplotlib, rich
│   └── resources.txt                  # Papers and docs referenced
│
└── transformers.py                     # The original flat script, still lives here
```

---

## The Modern Upgrades

### SwiGLU Activation (used in LLaMA, PaLM, Gemini)
Replaced the plain ReLU FFN with SwiGLU — the same activation function used in frontier models:

```python
class SwiGLUFFN(nn.Module):
    def forward(self, x):
        gated_flow = F.silu(self.w_gate(x)) * self.w_value(x)
        return self.w_down(gated_flow)
```

Two linear projections, a gating mechanism, and a SiLU activation. No bias. Clean.

### RMSNorm (also used in LLaMA)
Swapped `LayerNorm` → `RMSNorm`. Faster, simpler, and what modern LLMs actually use.

### Proper Causal Masking
The model can no longer cheat by looking at future tokens:

```python
mask = torch.triu(torch.ones(S, S), diagonal=1).bool().to(x.device)
scores = scores.masked_fill(mask, float('-inf'))
```

### Word-Level Tokenization with Special Tokens
Moved from character-level to word-level, with `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`:

```python
spcl = ['<BOS>', '<EOS>', '<PAD>', '<UNK>']
words = spcl + sorted(list(set(data)))
```

Generation now stops cleanly at `<EOS>` instead of running for a fixed token count.

---

## Config-Driven Training

Everything is a config class. No magic numbers, no hunting through 200 lines of code:

```python
class ModelConfig:
    vocab_size = len(words)
    dim = 64
    head = 4

class TrainConfig:
    EPOCHS = 30
    NORM = 1.0

class InferenceConfig:
    max_tokens = 1000
    temperature = 0.8
    top_k = 0
    top_p = 0.75
```

---

## Text Generation

```python
client = Generator(
    model=model,
    max_tokens=1000,
    temperature=0.8,
    top_p=0.75,
    EOS_token='<EOS>',
    ...
)

print(client.generate_response("to be or not to be"))
```

The model supports:
- **Temperature scaling** — How spicy do you want your predictions? 🌶️
- **Top-p (nucleus) sampling** — Only tokens that matter make the cut
- **EOS stopping** — Generation ends when the model says it ends

---

## Quick Start

```bash
git clone https://github.com/afngh/transformers.git
cd transformers

pip install -r data/requirements.txt

python -m main.model
```

---

## Requirements

```
torch
matplotlib
rich
```

---

## Why?

Because:

1. Everyone imports libraries. **We build libraries.**
2. Understanding SwiGLU hits different when you've written `F.silu(self.w_gate(x)) * self.w_value(x)` yourself.
3. Refactoring a notebook into a real project structure, with no AI assistance, at this pace — that's the rep that compounds.

---

> *"What a piece of work is a man, how noble in reason... also please predict the next token."*
> — Shakespeare, probably, if he knew about transformers
