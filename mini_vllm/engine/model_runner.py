import os
import pickle
from time import perf_counter

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from mini_vllm.config import Config
from mini_vllm.engine.scheduler import ScheduledSequence
from mini_vllm.engine.sequence import Sequence
from mini_vllm.layers.attention import copy_kvcache_fp8
from mini_vllm.layers.sampler import Sampler
from mini_vllm.models.qwen3 import Qwen3ForCausalLM
from mini_vllm.models.qwen2 import Qwen2ForCausalLM
from mini_vllm.utils.context import get_context, reset_context, set_context
from mini_vllm.utils.loader import load_model

MODEL_REGISTRY = {
    "qwen3": Qwen3ForCausalLM,
    "qwen2": Qwen2ForCausalLM,
}


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.max_num_decode_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.use_fp8_kv_cache = config.kv_cache_dtype == "fp8"
        self.use_spec_decode = bool(config.spec_decode_model)
        self.gamma = config.spec_decode_gamma
        self.last_verify_breakdown = {
            "prep_time_s": 0.0,
            "forward_time_s": 0.0,
            "lmhead_time_s": 0.0,
            "sampling_time_s": 0.0,
        }

        self.dist_port = int(os.getenv("MINI_VLLM_DIST_PORT", "2333"))
        dist.init_process_group("nccl", f"tcp://localhost:{self.dist_port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        target_cls = MODEL_REGISTRY.get(hf_config.model_type, Qwen3ForCausalLM)
        self.model = target_cls(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # Load draft model if speculative decoding is enabled
        self.draft_model = None
        if self.use_spec_decode:
            draft_hf_config = config.draft_hf_config
            torch.set_default_dtype(draft_hf_config.torch_dtype)
            draft_cls = MODEL_REGISTRY.get(draft_hf_config.model_type, Qwen3ForCausalLM)
            self.draft_model = draft_cls(draft_hf_config)
            load_model(self.draft_model, config.spec_decode_model)
            torch.set_default_dtype(hf_config.torch_dtype)

        self.warmup_model()
        self.allocate_kv_cache()
        if self.use_spec_decode:
            self.allocate_draft_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
            if self.use_spec_decode:
                self.capture_draft_cudagraph()
        self.allocate_decode_runtime_buffers()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="mini_vllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="mini_vllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [ScheduledSequence(Sequence([0] * max_model_len), max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        prefill_dtype = hf_config.torch_dtype
        decode_dtype = torch.float8_e4m3fn

        # Calculate per-block bytes for target model
        if self.use_fp8_kv_cache:
            assert prefill_dtype == torch.bfloat16, "FP8 KV cache currently requires BF16 model weights"
            assert head_dim == 128, "FP8 KV cache currently requires head_dim=128"
            block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * (
                torch.empty((), dtype=prefill_dtype).element_size() + torch.empty((), dtype=decode_dtype).element_size()
            )
        else:
            block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * torch.empty((), dtype=prefill_dtype).element_size()

        # If spec decode is enabled, also account for draft KV cache per block
        draft_block_bytes = 0
        if self.use_spec_decode:
            draft_hf_config = config.draft_hf_config
            draft_num_kv_heads = draft_hf_config.num_key_value_heads // self.world_size
            draft_head_dim = getattr(draft_hf_config, "head_dim", draft_hf_config.hidden_size // draft_hf_config.num_attention_heads)
            draft_dtype = draft_hf_config.torch_dtype
            draft_block_bytes = 2 * draft_hf_config.num_hidden_layers * self.block_size * draft_num_kv_heads * draft_head_dim * torch.empty((), dtype=draft_dtype).element_size()

        total_block_bytes = block_bytes + draft_block_bytes
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // total_block_bytes
        assert config.num_kvcache_blocks > 0

        if self.use_fp8_kv_cache:
            self.prefill_kv_cache = torch.empty(
                2,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                dtype=prefill_dtype,
            )
            self.decode_kv_cache = torch.empty(
                2,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                dtype=decode_dtype,
            )
        else:
            self.kv_cache = torch.empty(
                2,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                dtype=prefill_dtype,
            )

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.kv_cache_dtype = config.kv_cache_dtype
                if self.use_fp8_kv_cache:
                    module.prefill_k_cache = self.prefill_kv_cache[0, layer_id]
                    module.prefill_v_cache = self.prefill_kv_cache[1, layer_id]
                    module.decode_k_cache = self.decode_kv_cache[0, layer_id]
                    module.decode_v_cache = self.decode_kv_cache[1, layer_id]
                    module.k_cache = module.prefill_k_cache
                    module.v_cache = module.prefill_v_cache
                else:
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def allocate_draft_kv_cache(self):
        """Allocate separate KV cache for the draft model."""
        config = self.config
        draft_hf_config = config.draft_hf_config
        num_kv_heads = draft_hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(draft_hf_config, "head_dim", draft_hf_config.hidden_size // draft_hf_config.num_attention_heads)
        dtype = draft_hf_config.torch_dtype
        # Draft model shares the same number of blocks as the target model
        num_blocks = config.num_kvcache_blocks
        self.draft_kv_cache = torch.empty(
            2,
            draft_hf_config.num_hidden_layers,
            num_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
        )
        layer_id = 0
        for module in self.draft_model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.kv_cache_dtype = "auto"
                module.k_cache = self.draft_kv_cache[0, layer_id]
                module.v_cache = self.draft_kv_cache[1, layer_id]
                layer_id += 1

    @torch.inference_mode()
    def run_draft_prefill(self, seqs: list[Sequence]):
        """Run draft model prefill for sequences that just completed target prefill."""
        if not self.draft_model or not seqs:
            return
        all_input_ids = []
        all_positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        all_slot_mapping = []
        for seq in seqs:
            prompt_len = seq.num_prompt_tokens
            all_input_ids.extend(seq.prompt_token_ids)
            all_positions.extend(range(prompt_len))
            cu_seqlens_q.append(cu_seqlens_q[-1] + prompt_len)
            cu_seqlens_k.append(cu_seqlens_k[-1] + prompt_len)
            max_seqlen_q = max(prompt_len, max_seqlen_q)
            max_seqlen_k = max(prompt_len, max_seqlen_k)
            for position in range(prompt_len):
                block_id = seq.block_table[position // self.block_size]
                all_slot_mapping.append(block_id * self.block_size + position % self.block_size)
        input_ids = torch.tensor(all_input_ids, dtype=torch.int64, device="cuda")
        positions = torch.tensor(all_positions, dtype=torch.int64, device="cuda")
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cuda")
        slot_mapping = torch.tensor(all_slot_mapping, dtype=torch.int32, device="cuda")
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, None)
        self.draft_model(input_ids, positions)
        reset_context()

    @torch.inference_mode()
    def run_draft_decode(self, seqs: list[Sequence]) -> tuple[list[list[int]], list[list[torch.Tensor]]]:
        """Generate gamma draft tokens per sequence autoregressively using decode path."""
        if not self.draft_model:
            return [[] for _ in seqs], [[] for _ in seqs]
        bs = len(seqs)
        cpu = self.decode_cpu_staging
        gpu = self.decode_gpu_buffers
        block_tables = self._prepare_spec_block_tables(seqs)
        temperatures, top_ks, top_ps = self.prepare_sample(seqs)
        all_draft_tokens = [[] for _ in range(bs)]
        all_draft_probs = [[] for _ in range(bs)]
        tokens_added = [0] * bs
        for _ in range(self.gamma):
            for i, seq in enumerate(seqs):
                cur_len = len(seq) + tokens_added[i]
                last_tok = all_draft_tokens[i][-1] if all_draft_tokens[i] else seq.last_token
                num_blocks_needed = (cur_len + self.block_size - 1) // self.block_size
                if num_blocks_needed > len(seq.block_table):
                    num_blocks_needed = len(seq.block_table)
                last_block_tokens = cur_len - (num_blocks_needed - 1) * self.block_size
                cpu["input_ids"][i] = last_tok
                cpu["positions"][i] = cur_len - 1
                cpu["context_lens"][i] = cur_len
                cpu["slot_mapping"][i] = seq.block_table[num_blocks_needed - 1] * self.block_size + last_block_tokens - 1
            gpu["input_ids"][:bs].copy_(cpu["input_ids"][:bs], non_blocking=True)
            gpu["positions"][:bs].copy_(cpu["positions"][:bs], non_blocking=True)
            gpu["context_lens"][:bs].copy_(cpu["context_lens"][:bs], non_blocking=True)
            gpu["slot_mapping"][:bs].copy_(cpu["slot_mapping"][:bs], non_blocking=True)
            set_context(False, slot_mapping=gpu["slot_mapping"][:bs], context_lens=gpu["context_lens"][:bs], block_tables=block_tables)
            logits = self.run_draft_model(gpu["input_ids"][:bs], gpu["positions"][:bs])
            probs = torch.softmax(logits.float() / temperatures.unsqueeze(1), dim=-1)
            draft_token_ids = self.sampler(logits, temperatures, top_ks, top_ps).tolist()
            for i, token_id in enumerate(draft_token_ids):
                all_draft_tokens[i].append(token_id)
                all_draft_probs[i].append(probs[i])
                tokens_added[i] += 1
            reset_context()
        return all_draft_tokens, all_draft_probs

    @torch.inference_mode()
    def run_verify(self, seqs: list[Sequence], draft_tokens: list[list[int]], draft_probs: list[list[torch.Tensor]]) -> list[tuple[list[int], int]]:
        """
        Verify draft tokens using the target model.
        Runs target model on draft tokens in a single prefill-like pass
        using block_tables to read existing cached KV.
        Returns list of (accepted_token_ids, num_accepted) per sequence.
        """
        cpu = self.spec_cpu_staging
        gpu = self.spec_gpu_buffers
        total_tokens = 0
        max_seqlen_q = 0
        max_seqlen_k = 0
        seq_verify_lens = []
        prep_t0 = perf_counter()
        cpu["cu_seqlens_q"][0] = 0
        cpu["cu_seqlens_k"][0] = 0

        for i, seq in enumerate(seqs):
            verify_tokens = [seq.last_token] + draft_tokens[i]
            num_verify = len(verify_tokens)
            seq_verify_lens.append(num_verify)
            start_pos = len(seq) - 1
            base_offset = total_tokens

            for j, token_id in enumerate(verify_tokens):
                pos = start_pos + j
                cpu["input_ids"][base_offset + j] = token_id
                cpu["positions"][base_offset + j] = pos
                block_idx = pos // self.block_size
                if block_idx >= len(seq.block_table):
                    cpu["slot_mapping"][base_offset + j] = -1
                else:
                    block_id = seq.block_table[block_idx]
                    cpu["slot_mapping"][base_offset + j] = block_id * self.block_size + pos % self.block_size

            total_tokens += num_verify
            seqlen_k = start_pos + num_verify
            cpu["cu_seqlens_q"][i + 1] = total_tokens
            cpu["cu_seqlens_k"][i + 1] = cpu["cu_seqlens_k"][i] + seqlen_k
            max_seqlen_q = max(num_verify, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

        gpu["input_ids"][:total_tokens].copy_(cpu["input_ids"][:total_tokens], non_blocking=True)
        gpu["positions"][:total_tokens].copy_(cpu["positions"][:total_tokens], non_blocking=True)
        gpu["slot_mapping"][:total_tokens].copy_(cpu["slot_mapping"][:total_tokens], non_blocking=True)
        gpu["cu_seqlens_q"][:len(seqs) + 1].copy_(cpu["cu_seqlens_q"][:len(seqs) + 1], non_blocking=True)
        gpu["cu_seqlens_k"][:len(seqs) + 1].copy_(cpu["cu_seqlens_k"][:len(seqs) + 1], non_blocking=True)
        block_tables = self._prepare_spec_block_tables(seqs)
        torch.cuda.synchronize()
        prep_time_s = perf_counter() - prep_t0

        forward_t0 = perf_counter()
        set_context(True, gpu["cu_seqlens_q"][:len(seqs) + 1], gpu["cu_seqlens_k"][:len(seqs) + 1], max_seqlen_q, max_seqlen_k, gpu["slot_mapping"][:total_tokens], None, block_tables)
        hidden_states = self.model(gpu["input_ids"][:total_tokens], gpu["positions"][:total_tokens])
        torch.cuda.synchronize()
        forward_time_s = perf_counter() - forward_t0

        lmhead_t0 = perf_counter()
        # Do NOT use compute_logits here because ParallelLMHead in prefill mode
        # only returns last-token logits per seq. We need logits for ALL verify tokens.
        logits = torch.nn.functional.linear(hidden_states, self.model.lm_head.weight)
        torch.cuda.synchronize()
        lmhead_time_s = perf_counter() - lmhead_t0
        reset_context()

        sampling_t0 = perf_counter()
        results = []
        offset = 0
        gamma = self.gamma
        for i, seq in enumerate(seqs):
            num_verify = seq_verify_lens[i]
            num_draft = len(draft_tokens[i])
            seq_logits = logits[offset:offset + num_verify]
            offset += num_verify

            temperature = seq.temperature
            target_probs = torch.softmax(seq_logits.float() / temperature, dim=-1)

            draft_tok_ids = torch.tensor(draft_tokens[i], dtype=torch.int64, device=target_probs.device)
            draft_probs_stacked = torch.stack(draft_probs[i])

            p_target = target_probs[:num_draft].gather(1, draft_tok_ids.unsqueeze(1)).squeeze(1)
            p_draft = draft_probs_stacked.gather(1, draft_tok_ids.unsqueeze(1)).squeeze(1)

            accept_probs = torch.clamp(p_target / p_draft.clamp(min=1e-10), max=1.0)
            rand_vals = torch.rand(num_draft, device=target_probs.device)
            accepted_mask = rand_vals < accept_probs

            if accepted_mask.all():
                accepted_tokens = draft_tok_ids.tolist()
                bonus_token = torch.multinomial(target_probs[num_draft], 1).item()
                accepted_tokens.append(bonus_token)
                draft_accepted_count = num_draft
            else:
                first_reject = (~accepted_mask).nonzero(as_tuple=True)[0][0].item()
                accepted_tokens = draft_tok_ids[:first_reject].tolist()
                draft_accepted_count = first_reject
                adjusted = torch.clamp(target_probs[first_reject] - draft_probs_stacked[first_reject], min=0)
                adjusted_sum = adjusted.sum()
                if adjusted_sum > 1e-10:
                    adjusted = adjusted / adjusted_sum
                else:
                    adjusted = target_probs[first_reject]
                bonus_token = torch.multinomial(adjusted, 1).item()
                accepted_tokens.append(bonus_token)

            results.append((accepted_tokens, draft_accepted_count))
        torch.cuda.synchronize()
        sampling_time_s = perf_counter() - sampling_t0
        self.last_verify_breakdown = {
            "prep_time_s": prep_time_s,
            "forward_time_s": forward_time_s,
            "lmhead_time_s": lmhead_time_s,
            "sampling_time_s": sampling_time_s,
        }
        return results

    @torch.inference_mode()
    def cleanup_rejected_kv_slots(self, seqs: list[Sequence], verify_results: list[tuple[list[int], int]], gamma: int):
        """Zero out KV cache slots for rejected draft tokens to prevent dirty reads.
        After verify, positions [len(seq)..len(seq)+gamma-num_accepted-1] may have dirty KV
        in both target and draft caches. Since context_lens prevents reading them, this is
        a defensive cleanup for correctness safety."""
        dirty_slots = []
        for seq, (accepted, draft_accepted_count) in zip(seqs, verify_results):
            # accepted includes bonus token, so total positions written = len(accepted)
            # but seq.len was len(seq) at verify time. After appending accepted tokens,
            # seq.len = old_len + len(accepted). The dirty slots are from
            # old_len + len(accepted) to old_len + gamma (the slots written during verify
            # but beyond what was accepted).
            num_written = gamma + 1  # verify wrote gamma+1 positions (including last_token re-write)
            num_kept = len(accepted)  # tokens actually accepted + bonus
            # Dirty positions: from (old_len - 1 + num_kept + 1) to (old_len - 1 + num_written)
            # But old_len already advanced by num_kept via append_token, so:
            cur_len = len(seq)
            old_len = cur_len - num_kept
            for pos in range(old_len + num_kept, old_len + num_written):
                if pos < 0:
                    continue
                block_idx = pos // self.block_size
                if block_idx < len(seq.block_table):
                    block_id = seq.block_table[block_idx]
                    dirty_slots.append(block_id * self.block_size + pos % self.block_size)
        if not dirty_slots:
            return
        # Zero the dirty slots in target KV cache
        slot_tensor = torch.tensor(dirty_slots, dtype=torch.int64, device="cuda")
        if self.use_fp8_kv_cache:
            for layer_id in range(self.decode_kv_cache.shape[1]):
                self.decode_kv_cache[0, layer_id].view(-1, self.decode_kv_cache.shape[-1])[slot_tensor] = 0
                self.decode_kv_cache[1, layer_id].view(-1, self.decode_kv_cache.shape[-1])[slot_tensor] = 0
        elif hasattr(self, 'kv_cache'):
            for layer_id in range(self.kv_cache.shape[1]):
                self.kv_cache[0, layer_id].view(-1, self.kv_cache.shape[-1])[slot_tensor] = 0
                self.kv_cache[1, layer_id].view(-1, self.kv_cache.shape[-1])[slot_tensor] = 0
        # Zero the dirty slots in draft KV cache
        if hasattr(self, 'draft_kv_cache'):
            for layer_id in range(self.draft_kv_cache.shape[1]):
                self.draft_kv_cache[0, layer_id].view(-1, self.draft_kv_cache.shape[-1])[slot_tensor] = 0
                self.draft_kv_cache[1, layer_id].view(-1, self.draft_kv_cache.shape[-1])[slot_tensor] = 0

    @torch.inference_mode()
    def convert_prefill_to_decode_cache(self, seqs: list[Sequence]):
        if not self.use_fp8_kv_cache or not seqs:
            return
        slot_mapping = []
        for seq in seqs:
            for position in range(seq.num_prompt_tokens):
                block_id = seq.block_table[position // self.block_size]
                slot_mapping.append(block_id * self.block_size + position % self.block_size)
        if not slot_mapping:
            return
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, device='cuda')
        for layer_id in range(self.prefill_kv_cache.shape[1]):
            copy_kvcache_fp8(
                self.prefill_kv_cache[0, layer_id],
                self.prefill_kv_cache[1, layer_id],
                self.decode_kv_cache[0, layer_id],
                self.decode_kv_cache[1, layer_id],
                slot_mapping,
            )

    @torch.inference_mode()
    def allocate_decode_runtime_buffers(self):
        max_bs = self.config.max_num_seqs
        max_spec_tokens = max_bs * (self.gamma + 1)
        self.decode_cpu_staging = dict(
            input_ids=torch.empty(max_bs, dtype=torch.int64, device="cpu", pin_memory=True),
            positions=torch.empty(max_bs, dtype=torch.int64, device="cpu", pin_memory=True),
            slot_mapping=torch.empty(max_bs, dtype=torch.int32, device="cpu", pin_memory=True),
            context_lens=torch.empty(max_bs, dtype=torch.int32, device="cpu", pin_memory=True),
            block_table_row=torch.empty(self.max_num_decode_blocks, dtype=torch.int32, device="cpu", pin_memory=True),
        )
        self.spec_cpu_staging = dict(
            input_ids=torch.empty(max_spec_tokens, dtype=torch.int64, device="cpu", pin_memory=True),
            positions=torch.empty(max_spec_tokens, dtype=torch.int64, device="cpu", pin_memory=True),
            slot_mapping=torch.empty(max_spec_tokens, dtype=torch.int32, device="cpu", pin_memory=True),
            cu_seqlens_q=torch.empty(max_bs + 1, dtype=torch.int32, device="cpu", pin_memory=True),
            cu_seqlens_k=torch.empty(max_bs + 1, dtype=torch.int32, device="cpu", pin_memory=True),
        )
        self.decode_slot_seq_ids = [-1] * max_bs
        self.decode_slot_block_table_lens = [0] * max_bs
        self.decode_uses_graph_buffers = not self.enforce_eager and max_bs <= 512

        if self.decode_uses_graph_buffers:
            self.graph_vars["slot_mapping"].fill_(-1)
            self.graph_vars["context_lens"].zero_()
            self.graph_vars["block_tables"].fill_(-1)
            self.decode_gpu_buffers = {
                key: self.graph_vars[key]
                for key in ("input_ids", "positions", "slot_mapping", "context_lens", "block_tables")
            }
        else:
            self.decode_gpu_buffers = dict(
                input_ids=torch.empty(max_bs, dtype=torch.int64, device="cuda"),
                positions=torch.empty(max_bs, dtype=torch.int64, device="cuda"),
                slot_mapping=torch.empty(max_bs, dtype=torch.int32, device="cuda"),
                context_lens=torch.empty(max_bs, dtype=torch.int32, device="cuda"),
                block_tables=torch.full((max_bs, self.max_num_decode_blocks), -1, dtype=torch.int32, device="cuda"),
            )

        self.spec_gpu_buffers = dict(
            input_ids=torch.empty(max_spec_tokens, dtype=torch.int64, device="cuda"),
            positions=torch.empty(max_spec_tokens, dtype=torch.int64, device="cuda"),
            slot_mapping=torch.empty(max_spec_tokens, dtype=torch.int32, device="cuda"),
            cu_seqlens_q=torch.empty(max_bs + 1, dtype=torch.int32, device="cuda"),
            cu_seqlens_k=torch.empty(max_bs + 1, dtype=torch.int32, device="cuda"),
        )

    def _clear_decode_block_table_slot(self, slot_idx: int):
        if self.decode_slot_seq_ids[slot_idx] == -1 and self.decode_slot_block_table_lens[slot_idx] == 0:
            return
        self.decode_gpu_buffers["block_tables"][slot_idx].fill_(-1)
        self.decode_slot_seq_ids[slot_idx] = -1
        self.decode_slot_block_table_lens[slot_idx] = 0

    def _copy_decode_block_table_row(self, slot_idx: int, block_table: list[int]):
        staging_row = self.decode_cpu_staging["block_table_row"]
        staging_row.fill_(-1)
        for i, block_id in enumerate(block_table):
            staging_row[i] = block_id
        self.decode_gpu_buffers["block_tables"][slot_idx].copy_(staging_row, non_blocking=True)

    def _sync_decode_block_table_row(self, slot_idx: int, seq: Sequence):
        block_table = seq.block_table
        block_table_len = len(block_table)
        prev_seq_id = self.decode_slot_seq_ids[slot_idx]
        prev_block_table_len = self.decode_slot_block_table_lens[slot_idx]

        if prev_seq_id != seq.seq_id or block_table_len < prev_block_table_len:
            self._copy_decode_block_table_row(slot_idx, block_table)
        elif block_table_len > prev_block_table_len:
            staging_row = self.decode_cpu_staging["block_table_row"]
            tail_len = block_table_len - prev_block_table_len
            for i, block_id in enumerate(block_table[prev_block_table_len:block_table_len]):
                staging_row[i] = block_id
            self.decode_gpu_buffers["block_tables"][slot_idx, prev_block_table_len:block_table_len].copy_(
                staging_row[:tail_len],
                non_blocking=True,
            )

        self.decode_slot_seq_ids[slot_idx] = seq.seq_id
        self.decode_slot_block_table_lens[slot_idx] = block_table_len

    def _prepare_spec_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        slot_indices = []
        for seq in seqs:
            slot_idx = seq.decode_slot_index
            if slot_idx < 0:
                return self.prepare_block_tables(seqs)
            self._sync_decode_block_table_row(slot_idx, seq)
            slot_indices.append(slot_idx)

        first_slot = slot_indices[0]
        if slot_indices == list(range(first_slot, first_slot + len(slot_indices))):
            return self.decode_gpu_buffers["block_tables"][first_slot:first_slot + len(slot_indices), :max_len]

        index_tensor = torch.tensor(slot_indices, dtype=torch.int64, device="cuda")
        return self.decode_gpu_buffers["block_tables"].index_select(0, index_tensor)[:, :max_len]

    def _decode_graph_batch_size(self, bs: int) -> int:
        if self.enforce_eager or bs > 512:
            return bs
        return next(graph_bs for graph_bs in self.graph_bs if graph_bs >= bs)

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[ScheduledSequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        has_prefix = False
        for scheduled_seq in seqs:
            seq = scheduled_seq.seq
            start = seq.num_computed_tokens
            end = start + scheduled_seq.token_chunk_size
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            seqlen_q = end - start
            seqlen_k = end
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            has_prefix = has_prefix or start > 0
            if not seq.block_table:
                continue
            for position in range(start, end):
                block_id = seq.block_table[position // self.block_size]
                slot_mapping.append(block_id * self.block_size + position % self.block_size)
        if has_prefix:
            block_tables = self.prepare_block_tables([scheduled_seq.seq for scheduled_seq in seqs])
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    @torch.inference_mode()
    def prepare_decode(self, scheduled_seqs: list[ScheduledSequence]):
        bs = len(scheduled_seqs)
        graph_bs = self._decode_graph_batch_size(bs)
        cpu = self.decode_cpu_staging
        gpu = self.decode_gpu_buffers

        cpu["input_ids"][:bs].zero_()
        cpu["positions"][:bs].zero_()
        cpu["context_lens"][:bs].zero_()
        cpu["slot_mapping"][:bs].fill_(-1)

        for i, scheduled_seq in enumerate(scheduled_seqs):
            slot_idx = scheduled_seq.slot_index if scheduled_seq.slot_index >= 0 else i
            assert slot_idx == i, "decode scheduling is expected to follow slot order"
            if scheduled_seq.is_padding:
                self._clear_decode_block_table_slot(slot_idx)
                continue

            seq = scheduled_seq.seq
            context_len = len(seq)
            cpu["input_ids"][i] = seq.last_token
            cpu["positions"][i] = context_len - 1
            cpu["context_lens"][i] = context_len
            cpu["slot_mapping"][i] = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            self._sync_decode_block_table_row(slot_idx, seq)

        gpu["input_ids"][:bs].copy_(cpu["input_ids"][:bs], non_blocking=True)
        gpu["positions"][:bs].copy_(cpu["positions"][:bs], non_blocking=True)
        gpu["context_lens"][:bs].copy_(cpu["context_lens"][:bs], non_blocking=True)
        gpu["slot_mapping"][:bs].copy_(cpu["slot_mapping"][:bs], non_blocking=True)
        if graph_bs > bs:
            gpu["positions"][bs:graph_bs].zero_()
            gpu["context_lens"][bs:graph_bs].zero_()
            gpu["slot_mapping"][bs:graph_bs].fill_(-1)

        set_context(
            False,
            slot_mapping=gpu["slot_mapping"][:bs],
            context_lens=gpu["context_lens"][:bs],
            block_tables=gpu["block_tables"][:bs],
        )
        return gpu["input_ids"][:bs], gpu["positions"][:bs]

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        top_ks = []
        top_ps = []
        for seq in seqs:
            temperatures.append(seq.temperature)
            top_ks.append(seq.top_k)
            top_ps.append(seq.top_p)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ks = torch.tensor(top_ks, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures, top_ks, top_ps

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        if input_ids.data_ptr() != graph_vars["input_ids"].data_ptr():
            graph_vars["input_ids"][:bs].copy_(input_ids, non_blocking=True)
        if positions.data_ptr() != graph_vars["positions"].data_ptr():
            graph_vars["positions"][:bs].copy_(positions, non_blocking=True)
        if context.slot_mapping.data_ptr() != graph_vars["slot_mapping"].data_ptr():
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs].copy_(context.slot_mapping, non_blocking=True)
        if context.context_lens.data_ptr() != graph_vars["context_lens"].data_ptr():
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs].copy_(context.context_lens, non_blocking=True)
        if context.block_tables.data_ptr() != graph_vars["block_tables"].data_ptr():
            graph_vars["block_tables"][:bs].copy_(context.block_tables, non_blocking=True)
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, scheduled_seqs: list[ScheduledSequence], is_prefill: bool) -> list[int | None]:
        if is_prefill:
            seqs = [scheduled_seq.seq for scheduled_seq in scheduled_seqs]
            input_ids, positions = self.prepare_prefill(scheduled_seqs)
            logits = self.run_model(input_ids, positions, True)
            if self.rank == 0:
                sample_indices = [
                    i for i, scheduled_seq in enumerate(scheduled_seqs)
                    if scheduled_seq.requires_sampling
                ]
                token_ids = [None] * len(scheduled_seqs)
                if sample_indices:
                    sample_logits = logits[sample_indices]
                    temperatures, top_ks, top_ps = self.prepare_sample([seqs[i] for i in sample_indices])
                    sampled_token_ids = self.sampler(sample_logits, temperatures, top_ks, top_ps).tolist()
                    for i, token_id in zip(sample_indices, sampled_token_ids):
                        token_ids[i] = token_id
            else:
                token_ids = None
            reset_context()
            return token_ids

        seqs = [scheduled_seq.seq for scheduled_seq in scheduled_seqs]
        input_ids, positions = self.prepare_decode(scheduled_seqs)
        logits = self.run_model(input_ids, positions, False)
        if self.rank == 0:
            real_indices = [i for i, sseq in enumerate(scheduled_seqs) if not sseq.is_padding]
            real_seqs = [scheduled_seqs[i].seq for i in real_indices]
            temperatures, top_ks, top_ps = self.prepare_sample(real_seqs)
            real_logits = logits[real_indices]
            sampled = self.sampler(real_logits, temperatures, top_ks, top_ps).tolist()
            token_ids = [None] * len(scheduled_seqs)
            for idx, token_id in zip(real_indices, sampled):
                token_ids[idx] = token_id
        else:
            token_ids = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = self.max_num_decode_blocks
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_draft_cudagraph(self):
        """Capture CUDA graphs for the draft model decode path."""
        config = self.config
        draft_hf_config = config.draft_hf_config
        max_bs = min(config.spec_decode_max_batch_size, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, draft_hf_config.hidden_size)
        self.draft_graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.draft_graphs = {}
        self.draft_graph_pool = None

        for bs in reversed(self.draft_graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.draft_model(input_ids[:bs], positions[:bs])
            with torch.cuda.graph(graph, self.draft_graph_pool):
                outputs[:bs] = self.draft_model(input_ids[:bs], positions[:bs])
            if self.draft_graph_pool is None:
                self.draft_graph_pool = graph.pool()
            self.draft_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.draft_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def run_draft_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        """Run draft model with CUDA graph support."""
        if self.enforce_eager or not hasattr(self, 'draft_graphs') or input_ids.size(0) > 512:
            return self.draft_model.compute_logits(self.draft_model(input_ids, positions))
        bs = input_ids.size(0)
        context = get_context()
        graph = self.draft_graphs[next(x for x in self.draft_graph_bs if x >= bs)]
        gv = self.draft_graph_vars
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = context.slot_mapping
        gv["context_lens"].zero_()
        gv["context_lens"][:bs] = context.context_lens
        gv["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.draft_model.compute_logits(gv["outputs"][:bs])
