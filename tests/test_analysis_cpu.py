"""Checkpoint-free tests for paired routing analysis contracts."""

import json
from pathlib import Path
import tempfile
import unittest

from srr.analysis.cli import main, summarize_diagnosis, summarize_routes, summarize_task
from srr.analysis.controls import matched_random_expert_sets
from srr.analysis.statistics import (
    binary_auc, choice_accuracy, choice_margin, cluster_bootstrap_mean, holm_adjust,
)


class StatisticsTests(unittest.TestCase):
    def test_choice_scores_and_transitions(self):
        self.assertEqual(choice_margin([-2, -1, -3], 1), 1)
        self.assertEqual(choice_accuracy([-2, -1, -3], 1), 1)
        rows = [
            {"item_id": "a", "cluster_id": "p1", "gold_index": 1,
             "baseline_choice_scores": [0, -1], "intervention_choice_scores": [-1, 0]},
            {"item_id": "b", "cluster_id": "p1", "gold_index": 0,
             "baseline_choice_scores": [0, -1], "intervention_choice_scores": [0, -1]},
            {"item_id": "c", "cluster_id": "p2", "gold_index": 1,
             "baseline_choice_scores": [0, -1], "intervention_choice_scores": [-1, 0]},
        ]
        result = summarize_task(rows, 100, 1)
        self.assertAlmostEqual(result["accuracy_gain_fraction"]["estimate"], 2 / 3)
        self.assertEqual(result["changed_predictions"], 2)
        self.assertEqual(result["baseline_correct"], 1)
        self.assertEqual(result["intervention_correct"], 3)

    def test_cluster_bootstrap_is_item_weighted_and_deterministic(self):
        a = cluster_bootstrap_mean([1, 1, 0], ["long", "long", "short"], repetitions=100, seed=3)
        b = cluster_bootstrap_mean([1, 1, 0], ["long", "long", "short"], repetitions=100, seed=3)
        self.assertEqual(a, b)
        self.assertEqual(a["estimate"], 2 / 3)
        self.assertEqual(a["clusters"], 2)

    def test_multiplicity_and_auc(self):
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(binary_auc([0.0, 1.0, 1.0], [False, True, False]), 0.75)

    def test_route_partition_and_input_hash(self):
        rows = [
            {"event_id": "a", "cluster_id": "p1", "source_source": [0, 1],
             "merged_source": [0, 2], "source_merged": [0, 1], "merged_merged": [0, 2]},
            {"event_id": "b", "cluster_id": "p2", "source_source": [0, 1],
             "merged_source": [0, 1], "source_merged": [0, 2], "merged_merged": [0, 2]},
        ]
        result = summarize_routes(rows, 50, 2)
        self.assertEqual(result["changed_events"], 2)
        self.assertEqual(result["representation_only_among_changed"]["estimate"], 0.5)
        self.assertEqual(result["gate_only_among_changed"]["estimate"], 0.5)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            output = Path(directory) / "output.json"
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            from unittest.mock import patch
            with patch("sys.argv", ["srr-analyze", "routes", "--input", str(source), "--output", str(output), "--repetitions", "50"]):
                main()
                with self.assertRaises(FileExistsError):
                    main()
            self.assertEqual(len(json.loads(output.read_text())["input_sha256"]), 64)

    def test_changed_route_diagnostic_auc(self):
        rows = [
            {"event_id": "a", "cluster_id": "p1", "changed_route": True,
             "source_replay_nll_gain": 1, "metrics": {"js": 0.3, "set_distance": 1}},
            {"event_id": "b", "cluster_id": "p2", "changed_route": True,
             "source_replay_nll_gain": -1, "metrics": {"js": 0.1, "set_distance": 1}},
            {"event_id": "c", "cluster_id": "p3", "changed_route": False,
             "source_replay_nll_gain": 2, "metrics": {"js": 0.9, "set_distance": 1}},
        ]
        result = summarize_diagnosis(rows, 100, 4)
        self.assertEqual(result["changed_events"], 2)
        self.assertEqual(result["metrics"]["js"]["auc"], 1)
        self.assertEqual(result["metrics"]["set_distance"]["auc"], 0.5)

    def test_matched_random_controls(self):
        controls = matched_random_expert_sets([0, 1], [0, 2], 8, count=4, seed=3)
        self.assertEqual(len(controls), 4)
        self.assertTrue(all(len(set(route) & {0, 1}) == 1 for route in controls))
        self.assertNotIn([0, 2], controls)
        self.assertEqual(controls, matched_random_expert_sets([0, 1], [0, 2], 8, count=4, seed=3))
        with self.assertRaises(ValueError):
            matched_random_expert_sets([0], [1], 2, count=2)


if __name__ == "__main__":
    unittest.main()
