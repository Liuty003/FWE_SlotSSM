import torch

from scripts.train_task6_macro_candidate_scorer import (
    MacroCandidateScorer,
    pairwise_ranking_loss,
)


def test_macro_candidate_scorer_is_permutation_equivariant():
    model = MacroCandidateScorer(context_dim=16, chunk_steps=4)
    context = torch.randn(2, 16)
    chunks = torch.randn(2, 3, 4, 7)
    permutation = torch.tensor([2, 0, 1])
    expected = model(context, chunks)[:, permutation]
    actual = model(context, chunks[:, permutation])
    assert torch.allclose(actual, expected, atol=1e-6)


def test_pairwise_ranking_rewards_correct_order():
    target = torch.tensor([[1.0, 0.0, -1.0]])
    correct = torch.tensor([[2.0, 0.0, -2.0]])
    reversed_order = -correct
    assert pairwise_ranking_loss(correct, target) < pairwise_ranking_loss(
        reversed_order, target
    )
