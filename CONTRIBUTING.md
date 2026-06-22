# Contributing

Thanks for your interest in contributing. Before you do anything, read this.

---

## This is a learning project. That changes everything.

This repository exists for one reason — to understand how transformers work by building every part by hand. That goal shapes every rule below.

**No AI-generated code. No exceptions.**

Not GitHub Copilot. Not Claude. Not ChatGPT. Not Cursor, Tabnine, or any other agent or autocomplete tool that writes code for you. If you didn't type it and understand every line of it, it doesn't belong here.

This isn't about gatekeeping — it's about the point of the project. A transformer block you copy-pasted from an AI suggestion teaches you nothing. A transformer block you wrote, broke, debugged, and fixed teaches you everything. That's the only kind of contribution this repo accepts.

---

## What you can contribute

- Bug fixes — genuine bugs, with an explanation of why it's wrong and what the correct behavior should be
- Architecture improvements — only if you understand the math behind the change and can explain it
- New training datasets or data cleaning scripts
- Documentation improvements — clearer explanations, better comments, corrected errors in the README
- New features that fit the "from scratch" philosophy — no wrapping existing library components and calling it an implementation

---

## What you cannot contribute

- Code written or completed by any AI assistant or agent
- Wrapper classes around Hugging Face Transformers, PyTorch Lightning, or similar libraries
- Anything you can't explain line by line if asked
- Pretrained weights or checkpoints from other models

---

## How to contribute

1. Fork the repo
2. Create a branch — `fix/your-bug-name` or `feature/your-feature-name`
3. Write your code yourself, by hand
4. Test it — make sure training runs and loss goes down
5. Open a PR with a clear description of what changed and why

PR descriptions should answer three things:
- What is the problem or improvement?
- What did you change?
- Did you verify it works?

---

## Code style

- One class or concept per file — match the existing structure under `main/transformer_orch/`
- Config classes go in `main/config/_model_config.py`
- No magic numbers — if it's a hyperparameter, it belongs in a config class
- Comments should explain *why*, not *what* — the code already shows what

---

## Questions

Open an issue. If something in the architecture is unclear or you think something is wrong, say so — that kind of discussion is exactly what this project is for.

---

> *This project was built hands + keyboard. Contributions should be too.*