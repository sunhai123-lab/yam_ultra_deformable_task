"""不启动 Kit：验证真实脚本中的方位筛选、边界 IK 和采样约束。

脚本顶层会启动 AppLauncher，因此通过 AST 仅加载被测纯函数。
使用 Isaac Lab 的 Python 直接运行本文件即可，不需要 GUI/GPU；放在 scripts 下避免被模板忽略。
"""

import ast
import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/yam_ultra_deformable_place"))
from yam_ultra_deformable_place.yam_kinematics import (
    damped_least_squares_pose_step,
    forward_kinematics_and_jacobian,
)


class WorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        namespace = dict(math=math, random=random, torch=torch,
                         forward_kinematics_and_jacobian=forward_kinematics_and_jacobian,
                         damped_least_squares_pose_step=damped_least_squares_pose_step)
        constants = {
            "TABLE_SIZE", "TABLE_CENTER", "TABLE_TOP_Z", "ROBOT_BASE_XY", "OBJECT_RADIUS_RANGE",
            "OBJECT_BEARING_RANGE", "MIN_OBJECT_PLANAR_DISTANCE", "SAMPLING_MAX_ATTEMPTS",
            "MAX_TRANSFER_BEARING_DELTA",
            "BLOCK_SIZE", "BLOCK_RESET_CLEARANCE", "BALL_RADIUS", "BALL_PARTICLE_RADIUS",
            "GRASP_POINT_LOCAL", "GRASP_ROTATION_W", "APPROACH_IK_SEED",
        }
        functions = {"sample_object_positions", "grasp_point_kinematics", "radial_grasp_rotation",
                     "joint1_compatible_bearing", "solve_approach_joint_target"}
        tree = ast.parse((ROOT / "scripts/yam_pick_ball.py").read_text())
        selected = [node for node in tree.body if
                    (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants
                                                         for t in node.targets)) or
                    (isinstance(node, ast.FunctionDef) and node.name in functions)]
        exec(compile(ast.Module(body=selected, type_ignores=[]), "yam_pick_pure_functions", "exec"), namespace)
        cls.ns = namespace
        # 来自当前 YAM 资产的 rad 限位；此测试只针对当前固定基座版本。
        limits = torch.tensor([[[-2.618, 3.14159], [0., 3.665], [0., 3.14159],
                                [-1.693, 1.571], [-1.571, 1.571], [-2.094, 2.094]]])
        cls.robot = SimpleNamespace(device="cpu", data=SimpleNamespace(
            root_link_pose_w=SimpleNamespace(torch=torch.tensor([[0., 0., .04, 0., 0., 0., 1.]])),
            joint_pos=SimpleNamespace(torch=torch.zeros(1, 8)),
            joint_pos_limits=SimpleNamespace(torch=limits)))

    def test_radial_ik_boundaries(self):
        for angle in (*self.ns["OBJECT_BEARING_RANGE"], 0.0):
            for radius in self.ns["OBJECT_RADIUS_RANGE"]:
                for height in (.084, .184, .20):
                    with self.subTest(angle=angle, radius=radius, height=height):
                        point = torch.tensor([[radius * math.cos(angle), radius * math.sin(angle), height]])
                        rotation = self.ns["radial_grasp_rotation"](self.robot, point)
                        _, position_error, rotation_error = self.ns["solve_approach_joint_target"](
                            self.robot, point, rotation)
                        self.assertLess(position_error, .002)
                        self.assertLess(rotation_error, .02)

    def test_old_unreachable_bearing_is_rejected(self):
        point = torch.tensor([[.25 * math.cos(3.54), .25 * math.sin(3.54), .184]])
        with self.assertRaisesRegex(RuntimeError, "UNREACHABLE_BEARING"):
            self.ns["joint1_compatible_bearing"](self.robot, point)

    def test_sampling_separation_and_reproducibility(self):
        rng1, rng2 = random.Random(7), random.Random(7)
        for _ in range(100):
            first = self.ns["sample_object_positions"](rng1)
            self.assertEqual(first, self.ns["sample_object_positions"](rng2))
            block, ball = first
            self.assertGreaterEqual(math.dist(block[:2], ball[:2]), self.ns["MIN_OBJECT_PLANAR_DISTANCE"])
            block_bearing = math.atan2(block[1], block[0])
            ball_bearing = math.atan2(ball[1], ball[0])
            self.assertLessEqual(
                abs(block_bearing - ball_bearing), self.ns["MAX_TRANSFER_BEARING_DELTA"] + 1.0e-12
            )
            for point in first:
                radius = math.hypot(*point[:2])
                self.assertGreaterEqual(radius, .20)
                self.assertLessEqual(radius, .30)


if __name__ == "__main__":
    unittest.main()
