import torch

class Generator:
    def __init__(self, model, max_tokens=20, temperature=0.8, top_k=0, top_p=0.75, transform=None, config=None, seq_len=None, device=None):
        self.model = model.module if hasattr(model, 'module') else model
        self.transform = transform
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seq_len = seq_len
        self.config = config

    def generate_response(self, prompt):
        self.model.eval()
        ids = self.transform.encode(prompt)

        with torch.no_grad():
            for i in range(self.max_tokens):
                window = ids[-self.seq_len:]
                data = torch.tensor(window).unsqueeze(0).to(self.device)

                output = self.model(data)[:, -1, :]

                probabilities = torch.softmax(output / self.temperature, dim=-1)

                if self.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    num_to_keep = (cumulative_probs < self.top_p).sum(dim=-1) + 1
                    mask = torch.arange(probabilities.shape[-1], device=self.device).unsqueeze(0) < num_to_keep.unsqueeze(1)
                    filtered_sorted_probs = sorted_probs * mask
                    probabilities = torch.zeros_like(probabilities).scatter_(-1, sorted_indices, filtered_sorted_probs)
                    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

                next_id = torch.multinomial(probabilities.squeeze(0), num_samples=1).item()

                if next_id == self.transform.EOS_IDX:
                    break

                ids.append(next_id)

        return self.transform.decode(ids)