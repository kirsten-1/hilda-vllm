import torch

from mini_vllm.layers.sampler import Sampler


def test_top_k_masks_out_lower_ranked_logits():
    sampler = Sampler()
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0]], dtype=torch.float32)
    temperatures = torch.tensor([1.0], dtype=torch.float32)
    top_ks = torch.tensor([2], dtype=torch.int32)
    top_ps = torch.tensor([1.0], dtype=torch.float32)

    masked_logits = sampler._apply_top_k(logits.clone(), top_ks)

    assert torch.isneginf(masked_logits[0, 2])
    assert torch.isneginf(masked_logits[0, 3])
    assert masked_logits[0, 0] == logits[0, 0]
    assert masked_logits[0, 1] == logits[0, 1]


def test_top_p_masks_tail_outside_probability_mass():
    sampler = Sampler()
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float32))
    top_ps = torch.tensor([0.81], dtype=torch.float32)

    masked_logits = sampler._apply_top_p(logits.clone(), top_ps)

    assert not torch.isneginf(masked_logits[0, 0])
    assert not torch.isneginf(masked_logits[0, 1])
    assert torch.isneginf(masked_logits[0, 2])
    assert torch.isneginf(masked_logits[0, 3])


def test_top_p_uses_strict_boundary():
    sampler = Sampler()
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float32))
    top_ps = torch.tensor([0.8], dtype=torch.float32)

    masked_logits = sampler._apply_top_p(logits.clone(), top_ps)

    assert not torch.isneginf(masked_logits[0, 0])
    assert torch.isneginf(masked_logits[0, 1])
    assert torch.isneginf(masked_logits[0, 2])
    assert torch.isneginf(masked_logits[0, 3])
