import unittest

import torch_iris

from parity.operators import OPERATOR_SCOPE, run_operator_case
from parity.training import TRAINING_SCOPE, run_training_case
from parity.workloads import SCOPE, compare, run_case


class IrisParityTests(unittest.TestCase):
    def test_mlp_forward_scope(self):
        for case in SCOPE["cases"]:
            with self.subTest(case=case["id"]):
                count = torch_iris._C._kernel_submissions()
                actual = run_case(case, "iris:0")
                if case["id"] in {"mlp_forward_2d", "mlp_forward_3d"}:
                    # Two fused linear kernels, ReLU, and the final reduction.
                    self.assertEqual(torch_iris._C._kernel_submissions() - count, 4)
                compare(actual, run_case(case, "cpu"))

    def test_mlp_training_scope(self):
        for case in TRAINING_SCOPE["cases"]:
            with self.subTest(case=case["id"]):
                compare(run_training_case(case, "iris:0"), run_training_case(case, "cpu"),
                        TRAINING_SCOPE["tolerances"])

    def test_fp32_operator_scope(self):
        for case in OPERATOR_SCOPE["cases"]:
            with self.subTest(case=case["id"]):
                compare(run_operator_case(case, "iris:0"), run_operator_case(case, "cpu"),
                        OPERATOR_SCOPE["tolerances"])
