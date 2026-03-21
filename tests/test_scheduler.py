import itertools
from types import SimpleNamespace

import pytest

from mini_vllm.engine.scheduler import Scheduler
from mini_vllm.engine.sequence import Sequence, SequenceStatus
from mini_vllm.sampling_params import SamplingParams


@pytest.fixture(autouse=True)
def reset_sequence_state():
    original_block_size = Sequence.block_size
    original_counter = Sequence.counter
    Sequence.block_size = 2
    Sequence.counter = itertools.count()
    yield
    Sequence.block_size = original_block_size
    Sequence.counter = original_counter


def make_config(**overrides):
    config = {
        "max_num_seqs": 2,
        "max_num_batched_tokens": 8,
        "eos": 99,
        "num_kvcache_blocks": 6,
        "kvcache_block_size": 2,
        "enable_chunked_prefill": True,
        "chunked_prefill_size": 4,
        "chunked_prefill_min_size": 2,
        "chunked_prefill_tile_size": 2,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def test_prefill_moves_waiting_sequences_to_running():
    scheduler = Scheduler(make_config())
    seq1 = Sequence([10, 11], SamplingParams(max_tokens=4))
    seq2 = Sequence([20], SamplingParams(max_tokens=4))

    scheduler.add(seq1)
    scheduler.add(seq2)

    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [1, 2], is_prefill)

    assert is_prefill is True
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq1, seq2]
    assert seq1.status is SequenceStatus.RUNNING
    assert seq2.status is SequenceStatus.RUNNING


def test_postprocess_marks_finished_sequence_on_eos():
    scheduler = Scheduler(make_config())
    seq = Sequence([1, 2], SamplingParams(max_tokens=4))
    scheduler.add(seq)
    scheduled, is_prefill = scheduler.schedule()

    scheduler.postprocess(scheduled, [99], is_prefill)

    assert seq.status is SequenceStatus.FINISHED
    assert list(scheduler.running) == []
    assert seq.block_table == []


def test_chunked_prefill_rotates_partial_sequence_to_waiting_tail():
    scheduler = Scheduler(make_config(max_num_seqs=1, max_num_batched_tokens=4, chunked_prefill_size=2))
    long_seq = Sequence([10, 11, 12, 13, 14, 15], SamplingParams(max_tokens=4))
    short_seq = Sequence([20, 21], SamplingParams(max_tokens=4))
    scheduler.add(long_seq)
    scheduler.add(short_seq)

    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [None], is_prefill)

    assert is_prefill is True
    assert long_seq.num_computed_tokens == 4
    assert list(scheduler.waiting) == [short_seq, long_seq]
    assert list(scheduler.running) == []


def test_decode_runs_between_prefill_chunks_when_running_queue_is_non_empty():
    scheduler = Scheduler(make_config(max_num_batched_tokens=4, chunked_prefill_size=2))
    decode_seq = Sequence([1, 2], SamplingParams(max_tokens=4))
    scheduler.add(decode_seq)
    warmup_scheduled, warmup_is_prefill = scheduler.schedule()
    scheduler.postprocess(warmup_scheduled, [42], warmup_is_prefill)

    waiting_seq = Sequence([10, 11, 12, 13], SamplingParams(max_tokens=4))
    scheduler.add(waiting_seq)
    scheduler.last_step_was_prefill = False

    first_scheduled, first_is_prefill = scheduler.schedule()
    scheduler.postprocess(first_scheduled, [None], first_is_prefill)
    second_scheduled, second_is_prefill = scheduler.schedule()

    assert first_is_prefill is True
    assert second_is_prefill is False
    assert second_scheduled[0].seq is decode_seq
