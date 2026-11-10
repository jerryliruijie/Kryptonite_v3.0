"""Checks that LoRA starts from the pretrained function and only trains what it should."""

import unittest

import torch
import torch.nn.functional as F

from kryptonet import kryptonet
from lora import HEAD, HIDDEN_LINEAR, LoRALinear, apply_lora, count_trainable, expected_lora_count, merge_lora, trainable_parameters


def _base(seed=0):
    torch.manual_seed(seed)
    return kryptonet().eval()


def _steps(model, n=5, seed=1):
    """Run a few Adam steps on random data so that every trainable parameter moves."""
    torch.manual_seed(seed)
    x, y = torch.randn(64, 40), torch.randint(0, 2, (64,)).float()
    optimiser = torch.optim.Adam(trainable_parameters(model), lr=1e-2)
    for _ in range(n):
        loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


class TestLoRA(unittest.TestCase):
    def test_initial_function_is_unchanged(self):
        base = _base()
        adapted = apply_lora(base, rank=8, alpha=8)
        x = torch.randn(256, 40)
        with torch.no_grad():
            self.assertTrue(torch.equal(base(x), adapted.eval()(x)))
        for index in HIDDEN_LINEAR:
            self.assertTrue(torch.equal(adapted[index].B, torch.zeros_like(adapted[index].B)))
            self.assertGreater(adapted[index].A.abs().sum().item(), 0)

    def test_only_intended_parameters_update(self):
        for train_head in (True, False):
            with self.subTest(train_head=train_head):
                base = _base()
                adapted = apply_lora(base, rank=4, alpha=8, train_head=train_head)
                before = {name: value.clone() for name, value in adapted.state_dict().items()}
                _steps(adapted.train())
                for name, value in adapted.state_dict().items():
                    is_lora = name.endswith(".A") or name.endswith(".B")
                    is_head = name.startswith(f"{HEAD}.")
                    should_move = is_lora or (is_head and train_head)
                    self.assertEqual(not torch.equal(before[name], value), should_move, name)

    def test_original_model_is_not_modified(self):
        base = _base()
        snapshot = {name: value.clone() for name, value in base.state_dict().items()}
        _steps(apply_lora(base, rank=2, alpha=8).train())
        for name, value in base.state_dict().items():
            self.assertTrue(torch.equal(snapshot[name], value), name)
        self.assertTrue(all(parameter.requires_grad for parameter in base.parameters()))

    def test_parameter_count_matches_formula(self):
        base = _base()
        self.assertEqual(sum(p.numel() for p in base.parameters()), 809_473)
        for rank in (1, 2, 4, 8, 16):
            for train_head in (True, False):
                adapted = apply_lora(base, rank=rank, alpha=8, train_head=train_head)
                closed_form = 3624 * rank + (513 if train_head else 0)
                self.assertEqual(count_trainable(adapted), closed_form)
                self.assertEqual(expected_lora_count(base, rank, train_head=train_head), closed_form)

    def test_update_rank_is_at_most_r(self):
        adapted = apply_lora(_base(), rank=3, alpha=8)
        _steps(adapted.train())
        for index in HIDDEN_LINEAR:
            self.assertLessEqual(int(torch.linalg.matrix_rank(adapted[index].delta_weight().detach())), 3)

    def test_merged_model_matches_adapted_model(self):
        adapted = apply_lora(_base(), rank=4, alpha=8)
        _steps(adapted.train())
        merged = merge_lora(adapted).eval()
        self.assertFalse(any(isinstance(module, LoRALinear) for module in merged))
        x = torch.randn(128, 40)
        with torch.no_grad():
            torch.testing.assert_close(merged(x), adapted.eval()(x), rtol=1e-5, atol=1e-5)

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            LoRALinear(torch.nn.Linear(4, 4), rank=0, alpha=1)
        with self.assertRaises(ValueError):
            apply_lora(_base(), rank=2, alpha=8, layers=(HEAD,))


if __name__ == "__main__":
    unittest.main()
