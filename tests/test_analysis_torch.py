"""Small tensor tests; skipped on CPU-only machines without PyTorch."""

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RoutingTensorTests(unittest.TestCase):
    def test_crossed_route_partition(self):
        from srr.analysis.routing import crossed_routes, route_origin_masks, topk_set_distance

        source_hidden = torch.tensor([[2.0, 0.0]])
        merged_hidden = torch.tensor([[0.0, 2.0]])
        source_weight = torch.eye(2)
        merged_weight = torch.eye(2)
        routes = crossed_routes(source_hidden, merged_hidden, source_weight, merged_weight, 1)
        masks = route_origin_masks(routes)
        self.assertTrue(masks["representation_only"].item())
        self.assertEqual(topk_set_distance(routes.source_source, routes.merged_merged).item(), 1)

    def test_mixture_retains_native_mass(self):
        from srr.analysis.mixture import routed_mixture, output_comparison, max_entering_leaving_cosine

        outputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        before = routed_mixture(outputs, torch.tensor([0, 1]), torch.tensor([0.2, 0.3]))
        after = routed_mixture(outputs, torch.tensor([0, 2]), torch.tensor([0.2, 0.3]))
        torch.testing.assert_close(before, torch.tensor([0.2, 0.3]))
        self.assertGreater(output_comparison(before, after)["relative_l2"].item(), 0)
        self.assertGreaterEqual(max_entering_leaving_cosine(outputs, torch.tensor([0, 1]), torch.tensor([0, 2])).item(), 0)

    def test_native_route_mass_is_explicit(self):
        from srr.analysis.native_intervention import route_from_logits, route_from_selected_weights

        logits = torch.tensor([2.0, 1.0, 0.0])
        native = route_from_logits(logits, 2, norm_topk_prob=False)
        self.assertLess(native.mass.item(), 1)
        restored = route_from_selected_weights(native.selected, native.weights)
        torch.testing.assert_close(restored.mass, native.mass)


if __name__ == "__main__":
    unittest.main()
