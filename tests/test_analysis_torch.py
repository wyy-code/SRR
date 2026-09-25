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
        from srr.analysis.route_intervention import route_from_logits, route_from_selected_weights

        logits = torch.tensor([2.0, 1.0, 0.0])
        native = route_from_logits(logits, 2, norm_topk_prob=False)
        self.assertLess(native.mass.item(), 1)
        restored = route_from_selected_weights(native.selected, native.weights)
        torch.testing.assert_close(restored.mass, native.mass)

    def test_native_hook_noop_and_selected_route_change(self):
        from types import SimpleNamespace
        from torch import nn
        from srr.analysis.route_intervention import NativeRouteIntervention, route_from_logits
        from srr.analysis.mixture import routed_mixture

        class TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(2, 3, bias=False)
                self.experts = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(3)])
                self.top_k = 1
                self.norm_topk_prob = True
                with torch.no_grad():
                    self.gate.weight.copy_(torch.tensor([[2., 0.], [0., 2.], [-2., 0.]]))
                    self.experts[0].weight.copy_(torch.eye(2))
                    self.experts[1].weight.copy_(2 * torch.eye(2))
                    self.experts[2].weight.copy_(-torch.eye(2))

            def forward(self, hidden):
                logits = self.gate(hidden)
                route = route_from_logits(logits, self.top_k, self.norm_topk_prob,
                                          weight_dtype=hidden.dtype)
                outputs = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
                return routed_mixture(outputs, route.selected, route.weights), logits

        block = TinyBlock()
        model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=block)]))
        hidden = torch.tensor([[1.0, 0.0]])
        from srr.analysis.router_capture import RouterInputCapture

        with RouterInputCapture(model, [0]) as capture:
            block(hidden)
            self.assertEqual(capture.capture_modes[0], "native_full_logits")
            torch.testing.assert_close(capture.inputs[0], hidden)
            self.assertEqual(tuple(capture.outputs[0].shape), (1, 3))
        native_output = block(hidden)[0]
        operator = NativeRouteIntervention(model, "olmoe", [0])
        try:
            operator.set({"layer_id": 0, "position": 0, "condition": "native_noop",
                          "assert_native_recompute_rtol": 1e-5})
            no_op_output = block(hidden)[0]
            torch.testing.assert_close(no_op_output, native_output, atol=1e-6, rtol=0)
            self.assertEqual(len(operator.records), 1)
            self.assertLess(operator.records[0]["native_routed_recompute_relative_error"], 1e-5)

            operator.set({"layer_id": 0, "position": 0,
                          "condition": "source_set_mass_matched", "source_topk": [1],
                          "assert_mass": "native", "assert_native_recompute_rtol": 1e-5})
            patched_output = block(hidden)[0]
            self.assertEqual(operator.records[0]["topk_after"], [1])
            self.assertFalse(torch.allclose(patched_output, native_output))
        finally:
            operator.close()

        from srr.analysis.batched_route_intervention import BatchedNativeRouteIntervention

        batched = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
        native_batch = block(batched)[0]
        batch_operator = BatchedNativeRouteIntervention(model, "olmoe", [0])
        try:
            batch_operator.set_many([{
                "batch_index": 1, "sample_id": "second-row", "layer_id": 0,
                "position": 0, "condition": "source_set_mass_matched",
                "source_topk": [1], "assert_mass": "native",
                "assert_native_recompute_rtol": 1e-5,
            }])
            patched_batch = block(batched)[0]
            torch.testing.assert_close(patched_batch[0], native_batch[0], atol=1e-6, rtol=0)
            self.assertFalse(torch.allclose(patched_batch[1], native_batch[1]))
            self.assertEqual(batch_operator.records[0]["sample_id"], "second-row")
        finally:
            batch_operator.close()

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_crossed_routes_and_mixture(self):
        from srr.analysis.routing import crossed_routes, route_origin_masks
        from srr.analysis.mixture import routed_mixture

        device = torch.device("cuda:0")
        source = torch.tensor([[2.0, 0.0]], device=device)
        merged = torch.tensor([[0.0, 2.0]], device=device)
        weight = torch.eye(2, device=device)
        routes = crossed_routes(source, merged, weight, weight, 1)
        self.assertTrue(route_origin_masks(routes)["representation_only"].item())
        outputs = torch.eye(2, device=device)
        mixture = routed_mixture(outputs, routes.merged_merged[0], torch.ones(1, device=device))
        torch.testing.assert_close(mixture, torch.tensor([0.0, 1.0], device=device))


if __name__ == "__main__":
    unittest.main()
