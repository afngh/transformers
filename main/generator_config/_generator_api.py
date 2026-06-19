import torch

class Generator:
    def __init__(self, model, max_tokens=20, temperature=0.8, top_k=0, top_p=0.75, transform=None,itw=None, config=None, seq_len=None, device=None, EOS_token='<EOS>'):
        self.model = model
        self.transform = transform
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.itw = itw
        self.seq_len = seq_len
        self.EOS_token = EOS_token
        self.config = config

    def generate_response(self, prompt):
        self.model.eval()
        final_data_words = [prompt]
        current_words = prompt.split()

        with torch.no_grad():
            for i in range(self.max_tokens):
                window = current_words[-self.seq_len:]
                data = self.transform.encode(window)
                data = [min(max(idx, 0), self.config.vocab_size - 1) for idx in data]
                data = torch.tensor(data).unsqueeze(0).to(self.device)

                output = self.model(data)

                probabilities = torch.softmax(output / self.temperature, dim=-1)

                if self.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    num_to_keep = (cumulative_probs < self.top_p).sum(dim=-1) + 1
                    mask = torch.arange(probabilities.shape[-1], device=self.device).unsqueeze(0) < num_to_keep.unsqueeze(1)
                    filtered_sorted_probs = sorted_probs * mask
                    probabilities = torch.zeros_like(probabilities).scatter_(-1, sorted_indices, filtered_sorted_probs)
                    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

                word_idx = torch.multinomial(probabilities.squeeze(0), num_samples=1).item()
                predicted_word = self.itw[word_idx]

                if predicted_word == self.EOS_token:
                    break

                final_data_words.append(predicted_word)
                current_words.append(predicted_word)

        return ' '.join(final_data_words)