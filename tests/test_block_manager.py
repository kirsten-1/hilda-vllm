import itertools

import pytest

from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.sequence import Sequence


@pytest.fixture(autouse=True)
def reset_sequence_state():
    original_block_size = Sequence.block_size
    original_counter = Sequence.counter
    Sequence.block_size = 2
    Sequence.counter = itertools.count()
    yield
    Sequence.block_size = original_block_size
    Sequence.counter = original_counter


def test_allocate_reuses_cached_full_prefix_block():
    manager = BlockManager(num_blocks=4, block_size=2)
    seq1 = Sequence([1, 2, 3])
    seq2 = Sequence([1, 2, 9])

    manager.allocate(seq1)
    manager.allocate(seq2)

    shared_block_id = seq1.block_table[0]
    assert seq2.block_table[0] == shared_block_id
    assert seq2.num_cached_tokens == 2
    assert manager.blocks[shared_block_id].ref_count == 2


def test_deallocate_returns_blocks_to_free_pool():
    manager = BlockManager(num_blocks=3, block_size=2)
    seq = Sequence([1, 2, 3])

    manager.allocate(seq)
    assert len(manager.free_block_ids) == 1

    manager.deallocate(seq)

    assert len(manager.free_block_ids) == 3
    assert manager.used_block_ids == set()
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
