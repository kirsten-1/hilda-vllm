import heapq
import os
from collections import deque
from dataclasses import dataclass, field

from mini_vllm.config import Config
from mini_vllm.engine.block_manager import BlockManager
from mini_vllm.engine.sequence import Sequence, SequenceStatus


@dataclass
class ScheduledSequence:
    seq: Sequence | None  # None = padding slot
    token_chunk_size: int
    slot_index: int = -1

    @property
    def is_padding(self):
        return self.seq is None

    @property
    def num_computed_tokens_after(self):
        return self.seq.num_computed_tokens + self.token_chunk_size

    @property
    def requires_sampling(self):
        return self.num_computed_tokens_after >= len(self.seq)


class Scheduler:

    def __init__(self, config: Config):
        Sequence.block_size = config.kvcache_block_size
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.chunked_prefill_size = config.chunked_prefill_size
        self.chunked_prefill_min_size = config.chunked_prefill_min_size
        self.chunked_prefill_tile_size = config.chunked_prefill_tile_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: set[Sequence] = set()
        self.last_step_was_prefill = False

        # Persistent batching slots
        self.decode_slots: list[Sequence | None] = [None] * self.max_num_seqs
        self.free_slot_indices: list[int] = list(range(self.max_num_seqs))
        self.persistent_batch_size: int = 0
        self.debug_persistent_batching = os.getenv("HILDA_DEBUG_PB", "0") == "1"
        self.debug_prefill = os.getenv("HILDA_DEBUG_PREFILL", "0") == "1"

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # --- Slot management ---

    def _assign_decode_slot(self, seq: Sequence):
        #  从空闲 slot 列表弹出一个索引（最小堆）
        slot_idx = heapq.heappop(self.free_slot_indices)
        # 在 decode_slots 中放入请求
        self.decode_slots[slot_idx] = seq
        # 记录请求的 slot 索引
        seq.decode_slot_index = slot_idx
        #  persistent_batch_size 直接决定 GPU kernel 的计算范围。
        # batch size是一个“最高的水位”
        if slot_idx + 1 > self.persistent_batch_size:
            self.persistent_batch_size = slot_idx + 1
#   举个具体例子：

#   decode_slots = [SeqA, None, SeqC, None, SeqE]
#                    0      1     2     3     4

#   - 有效 slot 数量 = 3
#   - 但 kernel 必须处理 index 0~4，因为 SeqE 在 index 4

#   如果你把 batch dimension 设成 3（有效数量），kernel 只处理 [0, 1, 2]，SeqE 就被漏掉了。
# 所以必须设成 5（最高水位 = 4 + 1）。
# kernel 内部遇到 None slot 跳过就行，但range 必须覆盖所有有效 slot。

    # 把尾部的序列搬到中间的空洞处，使尾部变空，从而让 _maybe_shrink_batch 能收缩水位线
    def _free_decode_slot(self, seq: Sequence):
        slot_idx = seq.decode_slot_index
        if slot_idx >= 0:
            last_idx = self.persistent_batch_size - 1
            # while：找到最后一个非空 slot
            while last_idx > slot_idx and self.decode_slots[last_idx] is None:
                last_idx -= 1
            if last_idx > slot_idx:
                # 把尾部的请求搬到被释放的位置，填补空洞
                moved_seq = self.decode_slots[last_idx]
                assert moved_seq is not None
                self.decode_slots[slot_idx] = moved_seq
                moved_seq.decode_slot_index = slot_idx
                self.decode_slots[last_idx] = None
                heapq.heappush(self.free_slot_indices, last_idx)
            else:
                self.decode_slots[slot_idx] = None
                heapq.heappush(self.free_slot_indices, slot_idx)
            seq.decode_slot_index = -1

    def _maybe_shrink_batch(self):
        # 最后都是None,那么“最高水位”收缩
        while self.persistent_batch_size > 0 and self.decode_slots[self.persistent_batch_size - 1] is None:
            self.persistent_batch_size -= 1
        # 如果没有 running 请求，重置所有 slots
        if not self.running:
            assert all(slot is None for slot in self.decode_slots)
            self.persistent_batch_size = 0
            self.free_slot_indices = list(range(self.max_num_seqs))

    # --- Chunked prefill helpers ---

    def _compute_chunk_limit(self):
        if not self.enable_chunked_prefill:
            return self.max_num_batched_tokens
        if not self.running:
            return self.max_num_batched_tokens
        decode_pressure = len(self.running) / max(1, self.max_num_seqs)
        chunk_limit = self.chunked_prefill_size
        if decode_pressure >= 0.75:
            chunk_limit = max(self.chunked_prefill_min_size, chunk_limit // 4)
        elif decode_pressure >= 0.5:
            chunk_limit = max(self.chunked_prefill_min_size, chunk_limit // 2)
        return min(chunk_limit, self.max_num_batched_tokens)

    def _align_chunk_size(self, chunk_size: int, remaining_tokens: int):
        if not self.enable_chunked_prefill:
            return chunk_size
        if remaining_tokens <= chunk_size:
            return remaining_tokens
        tile_size = self.chunked_prefill_tile_size
        if chunk_size <= tile_size:
            return chunk_size
        aligned = (chunk_size // tile_size) * tile_size
        return aligned if aligned > 0 else chunk_size

    # --- Scheduling ---

    def _schedule_prefill(self) -> list[ScheduledSequence]:
        scheduled_seqs = []
        num_seqs = 0
        active_seqs = len(self.running)
        num_batched_tokens = 0
        chunk_limit = self._compute_chunk_limit()
        for seq in list(self.waiting):
            if active_seqs + num_seqs >= self.max_num_seqs:
                break
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    if self.debug_prefill:
                        print(
                            "[PREFILL] blocked on allocation: "
                            f"seq_id={seq.seq_id} len={len(seq)} "
                            f"free_blocks={len(self.block_manager.free_block_ids)} "
                            f"required_blocks={seq.num_blocks}"
                        )
                    break
                self.block_manager.allocate(seq)
                seq.num_computed_tokens = seq.num_cached_tokens
            remaining_tokens = seq.num_uncomputed_tokens
            if remaining_tokens <= 0:
                if self.debug_prefill:
                    print(
                        "[PREFILL] rotating anomalous seq with no remaining tokens: "
                        f"seq_id={seq.seq_id} len={len(seq)} "
                        f"num_computed_tokens={seq.num_computed_tokens}"
                    )
                self._append_waiting(seq)
                continue
            available_tokens = min(chunk_limit, self.max_num_batched_tokens - num_batched_tokens)
            if available_tokens <= 0:
                break
            chunk_size = min(remaining_tokens, available_tokens)
            chunk_size = self._align_chunk_size(chunk_size, remaining_tokens)
            if chunk_size <= 0:
                if self.debug_prefill:
                    print(
                        "[PREFILL] rotating anomalous seq with non-positive chunk: "
                        f"seq_id={seq.seq_id} remaining_tokens={remaining_tokens} "
                        f"available_tokens={available_tokens} chunk_limit={chunk_limit}"
                    )
                self._append_waiting(seq)
                continue
            num_seqs += 1
            num_batched_tokens += chunk_size
            scheduled_seqs.append(ScheduledSequence(seq, chunk_size))
        return scheduled_seqs

    def _schedule_decode(self) -> list[ScheduledSequence]:
        if self.persistent_batch_size == 0:
            return []
        scheduled_seqs = []
        has_real_seq = False
        for i in range(self.persistent_batch_size):
            seq = self.decode_slots[i]
            if seq is None:
                # Padding slot
                scheduled_seqs.append(ScheduledSequence(None, 0, slot_index=i))
                continue
            if not seq.block_table:
                # Keep decode slots self-healing under high churn.
                self.preempt(seq)
                scheduled_seqs.append(ScheduledSequence(None, 0, slot_index=i))
                continue
            # Check block availability
            if not self.block_manager.can_append(seq):
                # Try to preempt another seq to free blocks
                victim = self._find_preempt_victim(exclude=seq)
                if victim is not None:
                    self.preempt(victim)
                if not self.block_manager.can_append(seq):
                    # Can't allocate for this seq either, preempt it
                    self.preempt(seq)
                    scheduled_seqs.append(ScheduledSequence(None, 0, slot_index=i))
                    continue
            self.block_manager.may_append(seq)
            scheduled_seqs.append(ScheduledSequence(seq, 1, slot_index=i))
            has_real_seq = True
        return scheduled_seqs if has_real_seq else []

    def _find_preempt_victim(self, exclude: Sequence | None = None) -> Sequence | None:
        """Find a running seq to preempt (longest first), excluding the given seq."""
        best = None
        for seq in self.running:
            if seq is exclude:
                continue
            if best is None or len(seq) > len(best):
                best = seq
        return best

    def schedule(self) -> tuple[list[ScheduledSequence], bool]:
        decode_pressure = len(self.running) / max(1, self.max_num_seqs)
        prefer_decode = (
            self.enable_chunked_prefill
            and self.running
            and self.waiting
            and self.last_step_was_prefill
            and decode_pressure >= 0.5
        )
        if prefer_decode:
            scheduled = self._schedule_decode()
            if scheduled:
                if self.debug_persistent_batching:
                    active_slots = sum(slot is not None for slot in self.decode_slots[:self.persistent_batch_size])
                    print(
                        "[PB] schedule decode(preferred): "
                        f"running={len(self.running)} waiting={len(self.waiting)} "
                        f"persistent_batch_size={self.persistent_batch_size} "
                        f"active_slots={active_slots} scheduled={len(scheduled)}"
                    )
                self.last_step_was_prefill = False
                return scheduled, False
        scheduled = self._schedule_prefill()
        if scheduled:
            self.last_step_was_prefill = True
            return scheduled, True
        if self.debug_prefill and self.waiting:
            head = self.waiting[0]
            print(
                "[PREFILL] no schedulable prefill work: "
                f"waiting={len(self.waiting)} running={len(self.running)} "
                f"persistent_batch_size={self.persistent_batch_size} "
                f"head_seq_id={head.seq_id} head_len={len(head)} "
                f"head_num_computed_tokens={head.num_computed_tokens} "
                f"head_blocks={len(head.block_table)}"
            )
        scheduled = self._schedule_decode()
        if not scheduled:
            raise RuntimeError(
                "scheduler found no runnable prefill or decode work: "
                f"running={len(self.running)} waiting={len(self.waiting)} "
                f"persistent_batch_size={self.persistent_batch_size}"
            )
        if self.debug_persistent_batching:
            active_slots = sum(slot is not None for slot in self.decode_slots[:self.persistent_batch_size])
            print(
                "[PB] schedule decode(fallback): "
                f"running={len(self.running)} waiting={len(self.waiting)} "
                f"persistent_batch_size={self.persistent_batch_size} "
                f"active_slots={active_slots} scheduled={len(scheduled)}"
            )
        self.last_step_was_prefill = False
        return scheduled, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        seq.num_computed_tokens = 0
        self.running.discard(seq)
        self._free_decode_slot(seq)
        self._maybe_shrink_batch()
        self.waiting.appendleft(seq)

    def _append_waiting(self, seq: Sequence):
        self.waiting.remove(seq)
        self.waiting.append(seq)

    def postprocess(self, seqs: list[ScheduledSequence], token_ids: list[int | None], is_prefill: bool):
        if not is_prefill:
            for scheduled_seq, token_id in zip(seqs, token_ids):
                if scheduled_seq.is_padding:
                    continue
                seq = scheduled_seq.seq
                assert token_id is not None
                seq.num_computed_tokens += 1
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.discard(seq)
                    self._free_decode_slot(seq)
            self._maybe_shrink_batch()
            return

        for scheduled_seq, token_id in zip(seqs, token_ids):
            seq = scheduled_seq.seq
            reached_sequence_end = seq.num_computed_tokens + scheduled_seq.token_chunk_size >= len(seq)
            seq.num_computed_tokens = min(len(seq), seq.num_computed_tokens + scheduled_seq.token_chunk_size)
            if reached_sequence_end:
                self.waiting.remove(seq)
                seq.status = SequenceStatus.RUNNING
                self._assign_decode_slot(seq)
                self.running.add(seq)
            else:
                self._append_waiting(seq)
                continue
            assert token_id is not None
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.discard(seq)
                self._free_decode_slot(seq)
                self._maybe_shrink_batch()
