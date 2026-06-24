# pretokenize_wiki103.py
from datasets import load_dataset
import tiktoken
import torch

print("Loading WikiText-103...")
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

codec = tiktoken.get_encoding("gpt2")
EOS = codec.eot_token

all_ids = []
for i, row in enumerate(ds):
    text = row["text"].strip()
    if not text:
        continue
    ids = codec.encode_ordinary(text)
    all_ids.extend(ids)
    all_ids.append(EOS)
    if i % 10000 == 0:
        print(f"  {i:,} articles — {len(all_ids):,} tokens so far")

all_ids = torch.tensor(all_ids, dtype=torch.long)
print(f"Total tokens: {len(all_ids):,}")  # expect ~103M tokens

# 90/10 split
split = int(len(all_ids) * 0.9)
train_ids = all_ids[:split]
val_ids   = all_ids[split:]

import os
os.makedirs("bin/data", exist_ok=True)

torch.save(train_ids, "bin/data/wiki103_train.pt")
torch.save(val_ids,   "bin/data/wiki103_val.pt")
print("Saved wiki103_train.pt and wiki103_val.pt")
