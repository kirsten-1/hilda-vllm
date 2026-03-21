import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    enable_chunked_prefill: bool = True
    chunked_prefill_size: int = 512
    chunked_prefill_min_size: int = 128
    chunked_prefill_tile_size: int = 128
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kv_cache_dtype: str = "auto"
    spec_decode_model: str = ""
    spec_decode_gamma: int = 5

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.chunked_prefill_size > 0
        assert self.chunked_prefill_min_size > 0
        assert self.chunked_prefill_tile_size > 0
        assert self.chunked_prefill_min_size <= self.chunked_prefill_size
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.kv_cache_dtype in {"auto", "fp8"}
        assert self.spec_decode_gamma >= 1
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        if self.spec_decode_model:
            assert os.path.isdir(self.spec_decode_model), f"Draft model not found: {self.spec_decode_model}"
            self.draft_hf_config = AutoConfig.from_pretrained(self.spec_decode_model)
        else:
            self.draft_hf_config = None
