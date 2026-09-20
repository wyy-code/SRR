"""Small, checkpoint-free checks of the manuscript's SRR equations."""

import unittest

import torch

from srr.build import (
    build_layer_system,
    candidate_edges,
    centered_lift,
    pair_row_update,
    select_edges,
    split_summary,
)
from srr.solver import batched_weighted_functional_pullback


class SRRCoreTests(unittest.TestCase):
    def test_centered_log_probability_lift_is_centered_logit_difference(self):
        source = torch.tensor([[2.0, -1.0, 0.5], [0.2, 0.3, -0.8]])
        base = torch.tensor([[0.5, 0.0, -1.0], [0.1, 0.4, 0.7]])
        difference = source - base
        expected = difference - difference.mean(dim=-1, keepdim=True)
        torch.testing.assert_close(centered_lift(source, base), expected, atol=1e-6, rtol=0)

    def test_split_stable_pairs_and_disjoint_round_robin(self):
        profiles = [
            (torch.tensor([2.0, -1.0, 1.0, -0.5]), torch.ones(4) * 0.25),
            (torch.tensor([1.0, -2.0, 1.0, -1.0]), torch.ones(4) * 0.25),
        ]
        summary = split_summary(profiles)
        edges = candidate_edges(summary, summary, consistency=0.75, min_activity=0.002)
        self.assertEqual((edges[0]["promote"], edges[0]["demote"]), (0, 1))
        selected, used = select_edges({"math": edges, "code": edges}, ["math", "code"], 1)
        self.assertEqual(len(selected["math"]), 1)
        self.assertEqual(len(selected["code"]), 1)
        self.assertEqual(len(used), 4)

    def test_matching_domain_positive_advantage_and_source_base_clipping(self):
        key = "model.layers.7.mlp.gate.weight"
        rows = [
            {"uuid": "a", "role": "math", "continuation_ids": [1]},
            {"uuid": "b", "role": "code", "continuation_ids": [2]},
        ]
        def trace(nll, logits, hidden=None):
            value = {"nll": torch.tensor([nll]), "route_logits": {key: torch.tensor([logits])}}
            if hidden is not None:
                value["hidden"] = {key: torch.tensor([hidden])}
            return value

        base = {"a": trace(2, [0, 0]), "b": trace(2, [0, 0])}
        source = {"a": trace(1, [2, -2]), "b": trace(3, [2, -2])}
        parent = {"a": trace(2, [-3, 3], [1, 0]), "b": trace(2, [-3, 3], [0, 1])}
        edges = {"math": [{"promote": 0, "demote": 1}], "code": []}
        system = build_layer_system(rows, key, ["math", "code"], edges, base, source, parent, 0.5)
        self.assertEqual(system["response"].tolist(), [[4.0], [0.0]])
        self.assertGreater(float(system["weights"][0, 0]), 0)
        self.assertEqual(float(system["weights"][1, 0]), 0)

    def test_weighted_ridge_and_equal_opposite_sparse_writeback(self):
        hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        response = torch.tensor([[1.0], [2.0], [3.0]])
        weights = torch.tensor([[1.0], [0.5], [1.0]])
        result = batched_weighted_functional_pullback(
            hidden=hidden, response=response, weights=weights,
            owned_rows=torch.tensor([True]), ridge_relative=1e-3,
            max_iterations=100, tolerance=1e-8,
        )
        design = hidden.double()
        weight = weights[:, 0].double()
        ridge = 1e-3 * (weight * design.square().sum(1)).sum() / len(hidden)
        expected = torch.linalg.solve(
            design.T @ (weight[:, None] * design) + ridge * torch.eye(2, dtype=torch.float64),
            design.T @ (weight * response[:, 0].double()),
        )
        torch.testing.assert_close(result.delta_weight[0].double(), expected, atol=1e-5, rtol=0)
        self.assertEqual(result.audit["converged_rows"], 1)
        raw, scaled, used = pair_row_update(
            result.delta_weight, [{"promote": 0, "demote": 2}], (4, 2), 0.0625
        )
        self.assertEqual(used, {0, 2})
        torch.testing.assert_close(raw[0] + raw[2], torch.zeros(2))
        torch.testing.assert_close(scaled[0] - scaled[2], 0.0625 * result.delta_weight[0])
        self.assertTrue(torch.equal(scaled[[1, 3]], torch.zeros(2, 2)))


if __name__ == "__main__":
    unittest.main()
