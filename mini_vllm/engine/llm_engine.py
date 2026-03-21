import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from mini_vllm.config import Config
from mini_vllm.engine.model_runner import ModelRunner
from mini_vllm.engine.scheduler import Scheduler
from mini_vllm.engine.sequence import Sequence
from mini_vllm.engine.sequence import SequenceStatus
from mini_vllm.sampling_params import SamplingParams


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        self.last_generate_stats = None
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if is_prefill and self.config.kv_cache_dtype == "fp8":
            ready_for_decode = [
                scheduled_seq.seq
                for scheduled_seq in seqs
                if scheduled_seq.seq.is_prefill_done and not scheduled_seq.seq.is_finished and not scheduled_seq.seq.decode_cache_ready
            ]
            if ready_for_decode:
                self.model_runner.call("convert_prefill_to_decode_cache", ready_for_decode)
                for seq in ready_for_decode:
                    seq.decode_cache_ready = True

        # Speculative decoding: after normal decode, run draft+verify
        num_spec_tokens = 0
        if not is_prefill and self.config.spec_decode_model:
            running_seqs = [scheduled_seq.seq for scheduled_seq in seqs if not scheduled_seq.seq.is_finished]
            if running_seqs:
                gamma = self.config.spec_decode_gamma
                bm = self.scheduler.block_manager
                bs = self.config.kvcache_block_size

                # Step 1: Pre-allocate blocks for gamma draft tokens + 1 bonus token
                for seq in running_seqs:
                    current_len = len(seq)
                    needed_len = current_len + gamma + 1
                    current_blocks = len(seq.block_table)
                    needed_blocks = (needed_len + bs - 1) // bs
                    for _ in range(needed_blocks - current_blocks):
                        if bm.free_block_ids:
                            block_id = bm.free_block_ids[0]
                            block = bm.blocks[block_id]
                            block.reset()
                            bm.free_block_ids.remove(block_id)
                            bm.used_block_ids.add(block_id)
                            seq.block_table.append(block_id)

                # Step 2: Generate draft tokens
                draft_tokens, draft_probs = self.model_runner.call("run_draft_decode", running_seqs)

                # Step 3: Verify with target model
                verify_results = self.model_runner.call("run_verify", running_seqs, draft_tokens, draft_probs)

                # Step 4: Apply accepted tokens to sequences
                eos = self.config.eos
                for seq, (accepted, num_accepted) in zip(running_seqs, verify_results):
                    for tok in accepted:
                        seq.append_token(tok)
                        seq.num_computed_tokens += 1
                        num_spec_tokens += 1
                        if (not seq.ignore_eos and tok == eos) or seq.num_completion_tokens == seq.max_tokens:
                            seq.status = SequenceStatus.FINISHED
                            bm.deallocate(seq)
                            if seq in self.scheduler.running:
                                self.scheduler.running.remove(seq)
                            break

                # Step 5: Trim unused blocks and fix hash state for remaining sequences
                for seq in running_seqs:
                    if not seq.is_finished:
                        needed_blocks = (len(seq) + bs - 1) // bs
                        while len(seq.block_table) > needed_blocks:
                            block_id = seq.block_table.pop()
                            block = bm.blocks[block_id]
                            block.ref_count -= 1
                            if block.ref_count == 0:
                                bm.used_block_ids.discard(block_id)
                                bm.free_block_ids.append(block_id)
                        # Fix hash state on blocks:
                        # - Full blocks (not the last, or last when len%bs==0) should have hash set
                        # - Partial last block (len%bs != 0) should have hash == -1
                        for bi in range(len(seq.block_table)):
                            block = bm.blocks[seq.block_table[bi]]
                            is_last = (bi == len(seq.block_table) - 1)
                            block_is_full = (not is_last) or (len(seq) % bs == 0)
                            if block_is_full and block.hash == -1:
                                token_ids = seq.block(bi)
                                prefix = bm.blocks[seq.block_table[bi - 1]].hash if bi > 0 else -1
                                h = bm.compute_hash(token_ids, prefix)
                                block.update(h, token_ids)
                                bm.hash_to_block_id[h] = block.block_id
                            elif not block_is_full and block.hash != -1:
                                block.hash = -1
                                block.token_ids = []

        outputs = [(scheduled_seq.seq.seq_id, scheduled_seq.seq.completion_token_ids) for scheduled_seq in seqs if scheduled_seq.seq.is_finished]
        num_prefill_tokens = sum(scheduled_seq.token_chunk_size for scheduled_seq in seqs) if is_prefill else 0
        num_decode_tokens = sum(token_id is not None for token_id in token_ids) if is_prefill else len(token_ids)
        num_decode_tokens += num_spec_tokens
        return outputs, num_prefill_tokens, num_decode_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        seqs = []
        for prompt, sp in zip(prompts, sampling_params):
            seqs.append(self.add_request(prompt, sp))
        outputs = {}
        total_prompt_tokens = sum(seq.num_prompt_tokens for seq in seqs)
        prefill_throughput = decode_throughput = 0.0
        prefill_tokens = decode_tokens = 0
        prefill_time = decode_time = 0.0
        total_start = perf_counter()
        while not self.is_finished():
            t = perf_counter()
            output, num_prefill_tokens, num_decode_tokens = self.step()
            elapsed = perf_counter() - t
            if num_prefill_tokens > 0:
                prefill_tokens += num_prefill_tokens
                prefill_time += elapsed
                prefill_throughput = num_prefill_tokens / elapsed if elapsed else 0.0
            if num_decode_tokens > 0:
                decode_tokens += num_decode_tokens
                decode_time += elapsed
                decode_throughput = num_decode_tokens / elapsed if elapsed else 0.0
            if use_tqdm:
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        total_time = perf_counter() - total_start
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        total_output_tokens = sum(len(output["token_ids"]) for output in outputs)
        self.last_generate_stats = {
            "num_requests": len(prompts),
            "prompt_tokens": total_prompt_tokens,
            "output_tokens": total_output_tokens,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "total_time_s": total_time,
            "prefill_throughput_toks": prefill_tokens / prefill_time if prefill_time else 0.0,
            "decode_throughput_toks": decode_tokens / decode_time if decode_time else 0.0,
            "total_throughput_toks": total_output_tokens / total_time if total_time else 0.0,
        }
        if use_tqdm:
            pbar.close()
        return outputs
