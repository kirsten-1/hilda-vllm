import itertools
import pickle

import pytest

from mini_vllm.engine.sequence import Sequence
from mini_vllm.sampling_params import SamplingParams


@pytest.fixture(autouse=True)
def reset_sequence_state():
    original_block_size = Sequence.block_size
    original_counter = Sequence.counter
    Sequence.block_size = 4
    Sequence.counter = itertools.count()
    yield
    Sequence.block_size = original_block_size
    Sequence.counter = original_counter


def test_sequence_tracks_prompt_and_completion_tokens():
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=5))

    seq.append_token(4)
    seq.append_token(5)

    assert seq.prompt_token_ids == [1, 2, 3]
    assert seq.completion_token_ids == [4, 5]
    assert seq.num_completion_tokens == 2
    assert seq.num_blocks == 2
    assert seq.last_block_num_tokens == 1


def test_prompt_only_sequence_survives_pickle_roundtrip():
    seq = Sequence([7, 8, 9], SamplingParams(max_tokens=3, top_k=8, top_p=0.9))
    seq.num_computed_tokens = 2

    restored = pickle.loads(pickle.dumps(seq))

    assert restored.token_ids == [7, 8, 9]
    assert restored.num_prompt_tokens == 3
    assert restored.num_completion_tokens == 0
    assert restored.num_computed_tokens == 2
    assert restored.top_k == 8
    assert restored.top_p == 0.9


def test_sequence_exposes_prefill_progress():
    seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=3))
    seq.num_computed_tokens = 3

    assert seq.num_prompt_tokens_remaining == 1
    assert seq.num_uncomputed_tokens == 1
    assert seq.is_prefill_done is False


def test_sampling_params_validate_top_k_and_top_p():
    with pytest.raises(AssertionError):
        SamplingParams(top_k=0)
    with pytest.raises(AssertionError):
        SamplingParams(top_p=0.0)
    with pytest.raises(AssertionError):
        SamplingParams(top_p=1.1)


def test_sampling_params_rejects_zero_temperature():
    with pytest.raises(AssertionError):
        SamplingParams(temperature=0.0)
