from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_k: int = -1
    top_p: float = 1.0

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
        assert self.top_k == -1 or self.top_k > 0, "top_k must be -1 or > 0"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
