# 🤖 Transformers From Scratch
### Because using Hugging Face was too easy and we don't do easy things here.

---

## What Is This?

This is a **Transformer language model built entirely from scratch** in PyTorch, trained on Shakespeare's text, because apparently predicting what a 16th-century playwright would say next is the pinnacle of modern AI research.

No pretrained weights. No `from transformers import BertForEverything`. Just raw math, sweat, and an unhealthy relationship with matrix multiplication.

---

## Architecture

Because reading "Attention Is All You Need" once wasn't enough — we went ahead and *implemented* it.

| Component | What It Does | What It Actually Does |
|---|---|---|
| `Embedding` | Maps tokens to dense vectors | Turns letters into vibes |
| `PositionalEmbedding` | Injects position info via sinusoids | Tells the model where in the sentence it is, because apparently it can't count |
| `Attention` | Multi-head self-attention | Every token gossips about every other token simultaneously |
| `PostAttention` | Add & Norm + FFN | Cleans up the mess attention made |
| `Transformer` | Output projection | The bouncer who decides what word comes next |
| `Model` | Orchestrates everything | The guy who says "trust me, it all works together" |

---

## How It Works

```
Input Text → Embeddings → Positional Encoding → Multi-Head Attention
→ Add & Norm → Feed Forward Network → Add & Norm → Output Logits → Next Character
```

Or in plain English:

> "Give me Shakespeare. I will stare at it very hard. Then I will guess what letter comes next. Repeat 200 times. Profit."

---

## Training

```python
EPOCHS = 200          # Not 30. We meant 200. Comments lie.
batch_size = 32       # Classic. Timeless. Ours.
lr = 0.001            # Adam optimizer because SGD is for people who enjoy suffering
scheduler_gamma = 0.9 # We do decay here. Like fine cheese.
token_size = 3        # The model reads 3 characters at a time. Shakespeare wrote 5 acts. We start small.
```

Trained on the complete works of Shakespeare — a dataset containing approximately **zero** modern slang and **infinite** dramatic monologues about death.

---

## Text Generation

The model supports:

- **Temperature scaling** — How spicy do you want your predictions? 🌶️
- **Top-p (nucleus) sampling** — We only consider the tokens that matter. The rest are fired.
- **Top-k sampling** — Listed in the code. Not implemented. We believe in honesty.

```python
output = generate_response(model, "From", max_tokens=20, temperature=1, top_k=0, top_p=0.75)
# Output: Something that sounds vaguely Shakespearean and completely unhinged
```

---

## Requirements

```bash
pip install torch matplotlib rich
# Also: patience, a GPU if you have one, and the will to watch loss curves descend
```

---

## Results

The model learns to generate text in the style of Shakespeare. Whether Shakespeare himself would be flattered or horrified is a question for philosophers. We are engineers. We look at the loss go down and we feel things.

---

## Why?

Because:

1. Everyone imports libraries. **We build libraries** (well, models).
2. Understanding attention from the ground up hits different when you've written every `torch.matmul` yourself.
3. "I built a transformer from scratch" is objectively the best thing to say at a dinner party.

---

## File Structure

```
.
├── transformers.ipynb   # The whole thing. Notebook life.
└── shakespeare.txt      # 37 plays, 154 sonnets, 1 training corpus
```

---

## Author

Built by someone who read the Attention paper and thought *"yeah I can do that"* — and then did.

---

> *"What a piece of work is a man, how noble in reason... also please predict the next token."*
> — Shakespeare, probably, if he knew about transformers