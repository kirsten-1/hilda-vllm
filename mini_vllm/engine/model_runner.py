import pickle
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
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.use_fp8_kv_cache = config.kv_cache_dtype == "fp8"
        self.use_spec_decode = bool(config.spec_decode_model)
        self.gamma = config.spec_decode_gamma

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
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
        if self.use_fp8_kv_cache:
            assert prefill_dtype == torch.bfloat16, "FP8 KV cache currently requires BF16 model weights"
            assert head_dim == 128, "FP8 KV cache currently requires head_dim=128"
            block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * (
                torch.empty((), dtype=prefill_dtype).element_size() + torch.empty((), dtype=decode_dtype).element_size()
            )
        else:
            block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * torch.empty((), dtype=prefill_dtype).element_size()
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
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
        all_draft_tokens = [[] for _ in range(len(seqs))]
        all_draft_probs = [[] for _ in range(len(seqs))]
        tokens_added = [0] * len(seqs)
        for step in range(self.gamma):
            input_ids_list = []
            context_lens_list = []
            slot_mapping_list = []
            for i, seq in enumerate(seqs):
                cur_len = len(seq) + tokens_added[i]
                last_tok = all_draft_tokens[i][-1] if all_draft_tokens[i] else seq.last_token
                input_ids_list.append(last_tok)
                context_lens_list.append(cur_len)
                num_blocks_needed = (cur_len + self.block_size - 1) // self.block_size
                last_block_tokens = cur_len - (num_blocks_needed - 1) * self.block_size
                slot_mapping_list.append(seq.block_table[num_blocks_needed - 1] * self.block_size + last_block_tokens - 1)
            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device="cuda")
            positions = (context_lens - 1).to(dtype=torch.int64)
            slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int32, device="cuda")
            block_tables = self.prepare_block_tables(seqs)
            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
            logits = self.draft_model.compute_logits(self.draft_model(input_ids, positions))
            temperatures, top_ks, top_ps = self.prepare_sample(seqs)
            probs = torch.softmax(logits.float() / temperatures.unsqueeze(1), dim=-1)
            draft_token_ids = self.sampler(logits, temperatures, top_ks, top_ps).tolist()
            for i, token_id in enumerate(draft_token_ids):
                all_draft_tokens[i].append(token_id)
                all_draft_probs[i].append(probs[i].cpu())
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
        # Build verification inputs: for each seq, feed [draft_tok_0, ..., draft_tok_(gamma-1)]
        # The KV for positions before these draft tokens is already in the target cache.
        # We also need the logit at position (len(seq)-1) to verify the first draft token,
        # but that token's KV is already cached, so we re-feed it to get its logit.
        all_input_ids = []
        all_positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        all_slot_mapping = []
        seq_verify_lens = []

        for i, seq in enumerate(seqs):
            # We need gamma+1 logits: one for each draft token verification + bonus
            # Feed: [last_token_already_decoded, draft_0, ..., draft_(gamma-1)]
            # last_token is at position len(seq)-1 (already in cache, will be overwritten - that's ok)
            verify_tokens = [seq.last_token] + draft_tokens[i]
            num_verify = len(verify_tokens)
            seq_verify_lens.append(num_verify)
            start_pos = len(seq) - 1

            all_input_ids.extend(verify_tokens)
            all_positions.extend(range(start_pos, start_pos + num_verify))

            seqlen_q = num_verify
            seqlen_k = start_pos + num_verify  # total context including cached KV
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            for pos in range(start_pos, start_pos + num_verify):
                block_idx = pos // self.block_size
                block_id = seq.block_table[block_idx]
                all_slot_mapping.append(block_id * self.block_size + pos % self.block_size)

        input_ids = torch.tensor(all_input_ids, dtype=torch.int64, device="cuda")
        positions = torch.tensor(all_positions, dtype=torch.int64, device="cuda")
        cu_seqlens_q_t = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
        cu_seqlens_k_t = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cuda")
        slot_mapping = torch.tensor(all_slot_mapping, dtype=torch.int32, device="cuda")
        block_tables = self.prepare_block_tables(seqs)

        set_context(True, cu_seqlens_q_t, cu_seqlens_k_t, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        hidden_states = self.model(input_ids, positions)
        # Do NOT use compute_logits here because ParallelLMHead in prefill mode
        # only returns last-token logits per seq. We need logits for ALL verify tokens.
        logits = torch.nn.functional.linear(hidden_states, self.model.lm_head.weight)
        reset_context()

        # Apply rejection sampling per sequence
        results = []
        offset = 0
        for i, seq in enumerate(seqs):
            num_verify = seq_verify_lens[i]
            seq_logits = logits[offset:offset + num_verify]  # (gamma+1, vocab)
            offset += num_verify

            temperature = seq.temperature
            target_probs = torch.softmax(seq_logits.float() / temperature, dim=-1)

            accepted_tokens = []
            for j in range(len(draft_tokens[i])):
                draft_tok = draft_tokens[i][j]
                p_target = target_probs[j, draft_tok].item()
                p_draft = draft_probs[i][j][draft_tok].item()
                accept_prob = min(1.0, p_target / max(p_draft, 1e-10))
                if torch.rand(1).item() < accept_prob:
                    accepted_tokens.append(draft_tok)
                else:
                    # Rejection: sample from adjusted distribution max(0, p_target - p_draft)
                    adjusted = torch.clamp(target_probs[j] - draft_probs[i][j].to(target_probs.device), min=0)
                    adjusted_sum = adjusted.sum()
                    if adjusted_sum > 1e-10:
                        adjusted = adjusted / adjusted_sum
                    else:
                        adjusted = target_probs[j]
                    bonus_token = torch.multinomial(adjusted, 1).item()
                    accepted_tokens.append(bonus_token)
                    break
            else:
                # All draft tokens accepted, sample bonus token from last target position
                bonus_token = torch.multinomial(target_probs[len(draft_tokens[i])], 1).item()
                accepted_tokens.append(bonus_token)

            results.append((accepted_tokens, len(accepted_tokens)))
        return results

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

    def prepare_decode(self, scheduled_seqs: list[ScheduledSequence]):
        bs = len(scheduled_seqs)
        input_ids_list = [0] * bs
        context_lens_list = [0] * bs
        slot_mapping_list = [-1] * bs
        real_seqs_with_idx = []
        max_block_table_len = 1

        for i, sseq in enumerate(scheduled_seqs):
            if sseq.is_padding:
                continue
            seq = sseq.seq
            input_ids_list[i] = seq.last_token
            context_lens_list[i] = len(seq)
            slot_mapping_list[i] = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            real_seqs_with_idx.append((i, seq))
            max_block_table_len = max(max_block_table_len, len(seq.block_table))

        block_tables_list = [[0] * max_block_table_len for _ in range(bs)]
        for i, seq in real_seqs_with_idx:
            bt = seq.block_table
            for j, bid in enumerate(bt):
                block_tables_list[i][j] = bid
            for j in range(len(bt), max_block_table_len):
                block_tables_list[i][j] = -1

        input_ids = torch.tensor(input_ids_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        positions = (context_lens - 1).to(dtype=torch.int64)
        positions.clamp_min_(0)
        slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = torch.tensor(block_tables_list, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

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
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
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
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
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
