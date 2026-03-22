import torch
from torch import nn

from mini_vllm.utils.compile import safe_compile


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @safe_compile
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
    ):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        logits = self._apply_top_k(logits, top_ks)
        logits = self._apply_top_p(logits, top_ps)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        valid_top_ks = torch.where(top_ks > 0, torch.minimum(top_ks, torch.full_like(top_ks, vocab_size)), vocab_size)
        if torch.all(valid_top_ks == vocab_size):
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        ranks = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
        keep_sorted = ranks < valid_top_ks.unsqueeze(1)
        masked_sorted_logits = sorted_logits.masked_fill(~keep_sorted, float("-inf"))
        return torch.scatter(torch.empty_like(logits), 1, sorted_indices, masked_sorted_logits)

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
        if torch.all(top_ps >= 1.0):
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        keep_sorted = cumulative_probs < top_ps.unsqueeze(1)
        keep_sorted[:, 0] = True
        masked_sorted_logits = sorted_logits.masked_fill(~keep_sorted, float("-inf"))
        return torch.scatter(torch.empty_like(logits), 1, sorted_indices, masked_sorted_logits)
