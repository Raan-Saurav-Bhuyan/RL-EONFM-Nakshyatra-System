"""
Unit tests for SDM-EON Digital Twin Web Application.
"""

import os
import unittest
from web_app.backend.topology_parser import validate_and_parse_topology
from web_app.backend.trainer_runner import TrainingRunner

class TestWebAppBackend(unittest.TestCase):

    def test_topology_parser(self):
        nsfnet_path = "nsfnet.json"
        self.assertTrue(os.path.exists(nsfnet_path), "nsfnet.json should exist in workspace root")

        is_valid, msg, graph = validate_and_parse_topology(nsfnet_path)
        self.assertTrue(is_valid, f"Topology parsing failed: {msg}")
        self.assertEqual(graph["num_nodes"], 14, "NSFNet should have 14 nodes")
        self.assertTrue(graph["num_edges"] > 0, "NSFNet should have edges")

    def test_runner_initialization(self):
        runner = TrainingRunner()
        state = runner.get_state()
        self.assertEqual(state["status"], "idle")
        self.assertEqual(state["mode"], "integrated")

if __name__ == '__main__':
    unittest.main()
