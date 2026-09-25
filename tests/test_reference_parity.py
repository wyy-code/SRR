"""Optional numerical parity check against a reference implementation."""

import importlib.util
import os
import unittest

import torch

from srr import build


class ReferenceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("SRR_REFERENCE_MODULE")
        if not path:
            raise unittest.SkipTest("set SRR_REFERENCE_MODULE for numerical parity")
        spec = importlib.util.spec_from_file_location("srr_reference_builder", path)
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)

    def test_core_functions_match_reference(self):
        key = "model.layers.4.mlp.gate.weight"
        def trace(nll, logits, hidden=None):
            value = {"nll": torch.tensor(nll), "route_logits": {key: torch.tensor(logits)}}
            if hidden is not None:
                value["hidden"] = {key: torch.tensor(hidden)}
            return value

        base = trace([2.0, 2.5], [[0.1, 0.2], [0.1, 0.2]])
        source = trace([1.0, 2.0], [[0.8, -0.3], [0.7, -0.2]])
        parent = trace([1.8, 2.4], [[0.3, 0.1], [0.2, 0.2]], [[1.0, 0.0], [0.0, 1.0]])
        for function in ("prompt_profile",):
            got = getattr(build, function)(source, base, key, 0.5)
            expected = getattr(self.reference, function)(source, base, key, 0.5)
            for left, right in zip(got, expected):
                torch.testing.assert_close(left, right, atol=0, rtol=0)

        profiles = [build.prompt_profile(source, base, key, 0.5)] * 2
        got_summary = build.split_summary(profiles)
        ref_summary = self.reference.split_summary(profiles)
        for name in got_summary:
            torch.testing.assert_close(got_summary[name], ref_summary[name], atol=0, rtol=0)
        self.assertEqual(
            build.candidate_edges(got_summary, got_summary, 0.75, 0.002),
            self.reference.candidate_edges(ref_summary, ref_summary, 0.75, 0.002),
        )

        rows = [{"uuid": "one", "role": "math", "continuation_ids": [7, 8]}]
        edges = {"math": [{"promote": 0, "demote": 1}]}
        args = (rows, key, ["math"], edges, {"one": base}, {"one": source}, {"one": parent}, 0.5)
        got = build.build_layer_system(*args)
        expected = self.reference.build_layer_system(*args)
        for name in ("hidden", "parent_logits", "response", "weights"):
            torch.testing.assert_close(got[name], expected[name], atol=0, rtol=0)
        self.assertEqual(got["flat_edges"], expected["flat_edges"])


if __name__ == "__main__":
    unittest.main()
