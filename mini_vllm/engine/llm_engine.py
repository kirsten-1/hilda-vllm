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
from mini_vllm.sampling_params import SamplingParams


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
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
        outputs = [(scheduled_seq.seq.seq_id, scheduled_seq.seq.completion_token_ids) for scheduled_seq in seqs if scheduled_seq.seq.is_finished]
        num_prefill_tokens = sum(scheduled_seq.token_chunk_size for scheduled_seq in seqs) if is_prefill else 0
        num_decode_tokens = sum(token_id is not None for token_id in token_ids)
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
