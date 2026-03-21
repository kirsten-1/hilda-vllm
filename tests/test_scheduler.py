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

    assert is_prefill is True
    assert scheduled == [seq1, seq2]
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq1, seq2]
    assert seq1.status is SequenceStatus.RUNNING
    assert seq2.status is SequenceStatus.RUNNING


def test_postprocess_marks_finished_sequence_on_eos():
    scheduler = Scheduler(make_config())
    seq = Sequence([1, 2], SamplingParams(max_tokens=4))
    scheduler.add(seq)
    scheduled, _ = scheduler.schedule()

    scheduler.postprocess(scheduled, [99])

    assert seq.status is SequenceStatus.FINISHED
    assert list(scheduler.running) == []
    assert seq.block_table == []
