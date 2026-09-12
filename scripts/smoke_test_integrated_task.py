"""YAM Ultra 抓取并抬升可变形球的 Isaac Lab/Newton 集成测试。

脚本直接使用 Isaac Lab API 构建桌面、刚体方块、VBD 可变形球和 YAM Ultra
机械臂，不依赖其他机器人任务或策略代码。任务控制不是强化学习策略，而是一个
确定性的硬编码流程：

1. 在桌面可达区域随机化方块和球的位置；
2. 使用项目内基于 URDF 的正运动学、雅可比矩阵和 DLS 逆运动学移动夹爪；
3. 夹爪依次执行预抓取、下降、闭合和抬升；
4. 使用原始夹指碰撞网格与软球产生法向接触和 Coulomb 摩擦；
5. 通过有限值、FK 对齐误差和球心实际抬升量判断测试是否成功。

保留原始 YAM 视觉与碰撞资产，不添加夹指板，不使用节点绑定。
开合方向由原始指尖内侧几何验证；接触与完整抓取尚需分阶段运行验收。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=10, help="随机复位并测试的 episode 数量。")
parser.add_argument(
    "--steps-per-episode",
    type=int,
    default=30,
    help="每次复位后至少推进的物理步数；随后继续等待刚体方块和软球同时满足稳定判据。",
)
parser.add_argument("--seed", type=int, default=7, help="物体位置随机化使用的可复现随机种子。")
parser.add_argument("--pick-lift", action="store_true", help="执行硬编码抓取和抬升流程。")
parser.add_argument("--motion-steps", type=int, default=240, help="每个机械臂运动阶段使用的物理步数。")
parser.add_argument("--lift-height", type=float, default=0.03, help="球心期望沿世界 Z 轴抬升的距离，单位 m。")
parser.add_argument("--keep-open", action="store_true", help="测试结束后保持 Kit GUI，直到手动关闭窗口。")
parser.add_argument("--stop-after-hold", action="store_true", help="只验收到闭合保持阶段，不抬升。")
parser.add_argument("--gripper-check", action="store_true", help="固定机械臂，检查原始夹指张开—闭合—张开，不接近球。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_newton.assets import Articulation, RigidObject
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg, NewtonMaterialPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from yam_ultra_deformable_place.yam_kinematics import (
    damped_least_squares_position_step,
    damped_least_squares_pose_step,
    forward_kinematics_and_jacobian,
)
from isaaclab_contrib.deformable import CoupledMJWarpVBDSolverCfg, DeformableObject, VBDSolverCfg
from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAM_USD = PROJECT_ROOT / "assets/generated/yam_ultra_2/yam_ultra/yam_ultra.usda"

TABLE_SIZE = (1.10, 0.75, 0.08)
TABLE_CENTER = (0.12, 0.0, 0.40)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0
BLOCK_SIZE = (0.07, 0.07, 0.05)
BLOCK_RESET_CLEARANCE = 5.0e-4
RIGID_STATIC_FRICTION = 0.6
RIGID_DYNAMIC_FRICTION = 0.5
RIGID_RESTITUTION = 0.0

BLOCK_SETTLE_LINEAR_SPEED = 5.0e-4
BLOCK_SETTLE_ANGULAR_SPEED = 1.0e-2
BLOCK_SETTLE_CLEARANCE = 1.0e-3

BALL_RADIUS = 0.04
BALL_DENSITY = 500.0
BALL_YOUNGS_MODULUS = 5.0e4
BALL_POISSONS_RATIO = 0.35
BALL_PARTICLE_RADIUS = 0.006
BALL_RESET_CLEARANCE = 5.0e-4
BALL_INTERNAL_DAMPING = 0.01
# 允许重力沉降，但只有连续窗口内横向漂移小于 0.25 mm 才开始接近。
BALL_SETTLE_WINDOW_DRIFT = 2.5e-4
BALL_SETTLE_ROOT_SPEED = 1.0e-3
BALL_SETTLE_MAX_NODAL_SPEED = 1.0e-2
BALL_SETTLE_CONTACT_CLEARANCE = 1.5e-3

SCENE_SETTLE_REQUIRED_STEPS = 30
SCENE_SETTLE_TIMEOUT_STEPS = 480
SCENE_DIAGNOSTIC_STEPS = (30, 60, 120, 240, 360)

GRASP_POINT_LOCAL = (0.0, 0.0, -0.125)
GRASP_ROTATION_W = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
LEFT_PAD_LOCAL_POS = (0.014, 0.0469184183, -0.1980911)
RIGHT_PAD_LOCAL_POS = (0.014, 0.0458805300, -0.1981001)
LEFT_TIP_PATH = "/World/env_0/Robot/Geometry/base/link1/link2/link3/link4/link5/gripper/tip_left"
RIGHT_TIP_PATH = "/World/env_0/Robot/Geometry/base/link1/link2/link3/link4/link5/gripper/tip_right"

GRIPPER_OPEN_POSITION = -0.04695
GRIPPER_CLOSED_GAP = 0.00006211
GRIPPER_OPEN_GAP = GRIPPER_CLOSED_GAP - 2.0 * GRIPPER_OPEN_POSITION
NOMINAL_GRIP_STRESS = 5.0e3
BALL_EFFECTIVE_MODULUS = BALL_YOUNGS_MODULUS / (1.0 - BALL_POISSONS_RATIO**2)
BALL_COMPRESSION_RATIO = min(0.12, NOMINAL_GRIP_STRESS / BALL_EFFECTIVE_MODULUS)
GRIPPER_TARGET_GAP = 2.0 * BALL_RADIUS * (1.0 - BALL_COMPRESSION_RATIO)
GRIPPER_GRASP_POSITION = -0.5 * (GRIPPER_TARGET_GAP - GRIPPER_CLOSED_GAP)
MIN_LIFT_RATIO = 0.85
APPROACH_IK_SEED = (-0.48, 1.97, 1.63, -1.23, 0.0, -0.48)


@configclass
class DeformableNewtonCfg(NewtonCfg):
    model_cfg: NewtonModelCfg | None = None


def make_sim() -> SimulationContext:
    solver_cfg = CoupledMJWarpVBDSolverCfg(
        rigid_solver_cfg=MJWarpSolverCfg(
            njmax=256,
            nconmax=2048,
            ls_iterations=20,
            ls_parallel=False,
            impratio=1,
            cone="pyramidal",
            integrator="implicitfast",
            ccd_iterations=100,
        ),
        soft_solver_cfg=VBDSolverCfg(
            iterations=20,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,
            particle_collision_detection_interval=-1,
        ),
        coupling_mode="two_way",
    )
    return SimulationContext(
        SimulationCfg(
            dt=1.0 / 120.0,
            device=args_cli.device,
            physics=DeformableNewtonCfg(
                solver_cfg=solver_cfg,
                model_cfg=NewtonModelCfg(
                    soft_contact_ke=1.0e4,
                    soft_contact_kd=1.0e-2,
                    soft_contact_mu=5.0,
                ),
                num_substeps=10,
                use_cuda_graph=False,
            ),
        )
    )


def rigid_surface_material() -> NewtonMaterialPropertiesCfg:
    return NewtonMaterialPropertiesCfg(
        static_friction=RIGID_STATIC_FRICTION,
        dynamic_friction=RIGID_DYNAMIC_FRICTION,
        restitution=RIGID_RESTITUTION,
    )


def spawn_scene() -> tuple[Articulation, RigidObject, DeformableObject]:
    if not YAM_USD.is_file():
        raise FileNotFoundError(f"Converted YAM asset is missing: {YAM_USD}")

    sim_utils.create_prim("/World/env_0", "Xform")
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    table_cfg = sim_utils.CuboidCfg(
        size=TABLE_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=rigid_surface_material(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.22, 0.12)),
    )
    table_cfg.func("/World/env_0/Table", table_cfg, translation=TABLE_CENTER)

    robot_cfg = ArticulationCfg(
        prim_path="/World/env_0/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(YAM_USD),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_force=80.0, max_joint_velocity=3.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.22, 0.0, TABLE_TOP_Z),
            joint_pos={
                "joint1": 0.0,
                "joint2": 1.0,
                "joint3": 1.4,
                "joint4": -0.8,
                "joint5": 0.0,
                "joint6": 0.0,
                "joint7": GRIPPER_OPEN_POSITION,
                "joint8": GRIPPER_OPEN_POSITION,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=120.0,
                velocity_limit_sim=3.0,
                stiffness=3000.0,
                damping=100.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint[7-8]"],
                effort_limit_sim=15.0,
                velocity_limit_sim=0.01,
                stiffness=1000.0,
                damping=50.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)

    block = RigidObject(
        RigidObjectCfg(
            prim_path="/World/env_0/TargetBlock",
            spawn=sim_utils.CuboidCfg(
                size=BLOCK_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=rigid_surface_material(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.16, TABLE_TOP_Z + BLOCK_SIZE[2] / 2.0)),
        )
    )

    ball = DeformableObject(
        DeformableObjectCfg(
            prim_path="/World/env_0/DeformableBall",
            spawn=sim_utils.MeshSphereCfg(
                radius=BALL_RADIUS,
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.18, 0.12)),
                physics_material=NewtonDeformableBodyMaterialCfg(
                    density=BALL_DENSITY,
                    k_mu=BALL_YOUNGS_MODULUS / (2.0 * (1.0 + BALL_POISSONS_RATIO)),
                    k_lambda=(
                        BALL_YOUNGS_MODULUS
                        * BALL_POISSONS_RATIO
                        / ((1.0 + BALL_POISSONS_RATIO) * (1.0 - 2.0 * BALL_POISSONS_RATIO))
                    ),
                    k_damp=BALL_INTERNAL_DAMPING,
                    particle_radius=BALL_PARTICLE_RADIUS,
                ),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.15, -0.16, TABLE_TOP_Z + BALL_RADIUS)),
        )
    )
    return robot, block, ball


def sample_object_positions(rng: random.Random) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    block_pos = (
        rng.uniform(0.02, 0.16),
        rng.uniform(0.08, 0.18),
        TABLE_TOP_Z + BLOCK_SIZE[2] / 2.0 + BLOCK_RESET_CLEARANCE,
    )
    ball_pos = (
        rng.uniform(0.02, 0.16),
        rng.uniform(-0.18, -0.08),
        TABLE_TOP_Z + BALL_RADIUS,
    )
    planar_distance = ((block_pos[0] - ball_pos[0]) ** 2 + (block_pos[1] - ball_pos[1]) ** 2) ** 0.5
    if planar_distance < 0.13:
        raise RuntimeError(f"Sampling regions unexpectedly overlap: distance={planar_distance:.3f} m")
    return block_pos, ball_pos


def reset_episode(
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    block_pos: tuple[float, float, float],
    ball_pos: tuple[float, float, float],
) -> tuple[float, float, float]:
    joint_pos = robot.data.default_joint_pos.torch.clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
    robot.set_joint_position_target_index(target=joint_pos)

    block_pose = block.data.default_root_pose.torch.clone()
    block_pose[0, :3] = torch.tensor(block_pos, device=block_pose.device)
    block_pose[0, 3:] = torch.tensor((0.0, 0.0, 0.0, 1.0), device=block_pose.device)
    block.write_root_pose_to_sim_index(root_pose=block_pose)
    block.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=block_pose.device))

    nodal_state = ball.data.default_nodal_state_w.torch.clone()
    positions = nodal_state[..., :3]
    current_center = positions.mean(dim=1)
    desired_xy = torch.tensor(ball_pos[:2], device=positions.device, dtype=positions.dtype).unsqueeze(0)
    positions[..., :2] += (desired_xy - current_center[:, :2]).unsqueeze(1)
    current_lowest_node_z = positions[..., 2].amin(dim=1)
    desired_lowest_node_z = TABLE_TOP_Z + BALL_PARTICLE_RADIUS + BALL_RESET_CLEARANCE
    positions[..., 2] += (desired_lowest_node_z - current_lowest_node_z).unsqueeze(1)
    nodal_state[..., 3:] = 0.0
    ball.write_nodal_state_to_sim_index(nodal_state)

    kinematic_target = ball.data.nodal_kinematic_target.torch.clone()
    kinematic_target[..., :3] = nodal_state[..., :3]
    kinematic_target[..., 3] = 1.0
    ball.write_nodal_kinematic_target_to_sim_index(kinematic_target)

    robot.reset()
    block.reset()
    ball.reset()

    placed_center = nodal_state[0, :, :3].mean(dim=0)
    return tuple(float(v) for v in placed_center.tolist())


def block_support_metrics(
    block: RigidObject,
) -> tuple[float, tuple[float, float, float], tuple[float, float, float], bool]:
    pose = block.data.root_link_pose_w.torch[0]
    corners = torch.tensor(
        [
            (x * BLOCK_SIZE[0] / 2, y * BLOCK_SIZE[1] / 2, z * BLOCK_SIZE[2] / 2)
            for x in (-1, 1)
            for y in (-1, 1)
            for z in (-1, 1)
        ],
        device=pose.device,
        dtype=pose.dtype,
    )
    corners_w = quat_apply(pose[3:].expand(8, -1), corners) + pose[:3]
    clearance = corners_w[:, 2].min().item() - TABLE_TOP_Z
    linear_velocity = tuple(float(v) for v in block.data.root_com_lin_vel_w.torch[0].tolist())
    angular_velocity = tuple(float(v) for v in block.data.root_com_ang_vel_w.torch[0].tolist())
    half_x = TABLE_SIZE[0] / 2.0 - BLOCK_SIZE[0] / 2.0
    half_y = TABLE_SIZE[1] / 2.0 - BLOCK_SIZE[1] / 2.0
    on_table_xy = abs(pose[0].item() - TABLE_CENTER[0]) <= half_x and abs(pose[1].item() - TABLE_CENTER[1]) <= half_y
    return clearance, linear_velocity, angular_velocity, on_table_xy


def check_block_support(block: RigidObject, *, stage: str) -> None:
    pose = block.data.root_link_pose_w.torch[0]
    clearance, linear_velocity, angular_velocity, on_table_xy = block_support_metrics(block)
    linear_speed = sum(v * v for v in linear_velocity) ** 0.5
    angular_speed = sum(v * v for v in angular_velocity) ** 0.5
    print(
        f"BLOCK_SUPPORT stage={stage} center={pose[:3].tolist()} clearance_m={clearance:.6f} "
        f"linear_velocity_mps={[round(v, 7) for v in linear_velocity]} linear_speed_mps={linear_speed:.6f} "
        f"angular_velocity_radps={[round(v, 7) for v in angular_velocity]} angular_speed_radps={angular_speed:.6f} "
        f"on_table_xy={on_table_xy}",
        flush=True,
    )
    if not on_table_xy or clearance < -0.01:
        raise RuntimeError(f"BLOCK_INSTABILITY: dynamic target block left the tabletop at stage={stage}")


def block_state_is_settled(clearance, linear_velocity, angular_velocity, on_table_xy) -> bool:
    linear_speed = sum(v * v for v in linear_velocity) ** 0.5
    angular_speed = sum(v * v for v in angular_velocity) ** 0.5
    return (
        on_table_xy
        and abs(clearance) <= BLOCK_SETTLE_CLEARANCE
        and linear_speed <= BLOCK_SETTLE_LINEAR_SPEED
        and angular_speed <= BLOCK_SETTLE_ANGULAR_SPEED
    )


def ball_center(ball: DeformableObject) -> torch.Tensor:
    return ball.data.nodal_pos_w.torch.mean(dim=1)


def ball_settle_metrics(ball: DeformableObject):
    nodal_state = ball.data.nodal_state_w.torch[0]
    positions = nodal_state[:, :3]
    velocities = nodal_state[:, 3:]
    center = positions.mean(dim=0)
    root_velocity = velocities.mean(dim=0)
    max_nodal_speed = torch.linalg.vector_norm(velocities, dim=-1).max().item()
    contact_clearance = positions[:, 2].min().item() - BALL_PARTICLE_RADIUS - TABLE_TOP_Z
    half_x = TABLE_SIZE[0] / 2.0 - BALL_RADIUS
    half_y = TABLE_SIZE[1] / 2.0 - BALL_RADIUS
    on_table_xy = (
        abs(center[0].item() - TABLE_CENTER[0]) <= half_x and abs(center[1].item() - TABLE_CENTER[1]) <= half_y
    )
    return (
        tuple(float(v) for v in center.tolist()),
        contact_clearance,
        tuple(float(v) for v in root_velocity.tolist()),
        max_nodal_speed,
        on_table_xy,
    )


def check_ball_support(ball: DeformableObject, *, stage: str, reset_center: tuple[float, float, float]) -> None:
    center, clearance, root_velocity, max_nodal_speed, on_table_xy = ball_settle_metrics(ball)
    root_speed = sum(v * v for v in root_velocity) ** 0.5
    xy_drift = ((center[0] - reset_center[0]) ** 2 + (center[1] - reset_center[1]) ** 2) ** 0.5
    print(
        f"BALL_SUPPORT stage={stage} center={[round(v, 7) for v in center]} contact_clearance_m={clearance:.6f} "
        f"xy_drift_m={xy_drift:.6f} root_velocity_mps={[round(v, 7) for v in root_velocity]} "
        f"root_speed_mps={root_speed:.6f} max_nodal_speed_mps={max_nodal_speed:.6f} on_table_xy={on_table_xy}",
        flush=True,
    )
    if not on_table_xy or clearance < -0.01:
        raise RuntimeError(
            f"BALL_INSTABILITY: deformable ball left the tabletop or penetrated excessively at stage={stage}"
        )


def ball_state_is_settled(ball: DeformableObject) -> bool:
    _, clearance, root_velocity, max_nodal_speed, on_table_xy = ball_settle_metrics(ball)
    root_speed = sum(v * v for v in root_velocity) ** 0.5
    return (
        on_table_xy
        and abs(clearance) <= BALL_SETTLE_CONTACT_CLEARANCE
        and root_speed <= BALL_SETTLE_ROOT_SPEED
        and max_nodal_speed <= BALL_SETTLE_MAX_NODAL_SPEED
    )


def step_scene(sim, robot, block, ball, joint_target) -> None:
    robot.set_joint_position_target_index(target=joint_target)
    robot.write_data_to_sim()
    ball.write_data_to_sim()
    sim.step(render=not args_cli.headless)
    dt = sim.get_physics_dt()
    robot.update(dt)
    block.update(dt)
    ball.update(dt)
    for name, value in (
        ("robot_joint_positions", robot.data.joint_pos.torch),
        ("robot_joint_velocities", robot.data.joint_vel.torch),
        ("block_position", block.data.root_pos_w.torch),
        ("block_velocity", block.data.root_com_vel_w.torch),
        ("ball_nodes", ball.data.nodal_state_w.torch),
    ):
        if not torch.isfinite(value).all():
            invalid = (~torch.isfinite(value)).nonzero()[:8].cpu().tolist()
            raise RuntimeError(f"FAILURE: non-finite {name}; first_indices={invalid}")


def wait_until_scene_settled(
    sim, robot, block, ball, joint_target, *, episode: int, minimum_steps: int, ball_reset_center
) -> int:
    stable_steps = 0
    stable_centers = []
    timeout_steps = max(SCENE_SETTLE_TIMEOUT_STEPS, minimum_steps + SCENE_SETTLE_REQUIRED_STEPS)
    for step_index in range(timeout_steps):
        step_scene(sim, robot, block, ball, joint_target)
        step_count = step_index + 1
        bc, blv, bav, bot = block_support_metrics(block)
        ball_now, bclear, _, _, bon = ball_settle_metrics(ball)
        if not bot or bc < -0.01:
            check_block_support(block, stage=f"episode_{episode}_step_{step_count}")
            raise RuntimeError(f"BLOCK_INSTABILITY: target block became unsupported in episode {episode}")
        if not bon or bclear < -0.01:
            check_ball_support(ball, stage=f"episode_{episode}_step_{step_count}", reset_center=ball_reset_center)
            raise RuntimeError(
                f"BALL_INSTABILITY: deformable ball became unsupported in episode {episode}, center={ball_now}"
            )

        if block_state_is_settled(bc, blv, bav, bot) and ball_state_is_settled(ball):
            stable_centers.append(ball_now[:2])
            if len(stable_centers) > SCENE_SETTLE_REQUIRED_STEPS:
                stable_centers.pop(0)
            drift = sum((max(p[i] for p in stable_centers) - min(p[i] for p in stable_centers)) ** 2
                        for i in (0, 1)) ** 0.5
            if drift > BALL_SETTLE_WINDOW_DRIFT:
                stable_centers = [ball_now[:2]]
                stable_steps = 0
            stable_steps += 1
        else:
            stable_steps = 0
            stable_centers.clear()

        if step_count in SCENE_DIAGNOSTIC_STEPS or step_count == minimum_steps:
            stage = f"episode_{episode}_settling_step_{step_count}"
            check_block_support(block, stage=stage)
            check_ball_support(ball, stage=stage, reset_center=ball_reset_center)

        if step_count >= minimum_steps and stable_steps >= SCENE_SETTLE_REQUIRED_STEPS:
            stage = f"episode_{episode}_settled_step_{step_count}"
            check_block_support(block, stage=stage)
            check_ball_support(ball, stage=stage, reset_center=ball_reset_center)
            print(
                f"SCENE_SETTLED episode={episode} steps={step_count} stable_steps={stable_steps} "
                f"block_linear_threshold_mps={BLOCK_SETTLE_LINEAR_SPEED:.6f} "
                f"ball_root_threshold_mps={BALL_SETTLE_ROOT_SPEED:.6f} "
                f"ball_max_nodal_threshold_mps={BALL_SETTLE_MAX_NODAL_SPEED:.6f}",
                flush=True,
            )
            return step_count

    check_block_support(block, stage=f"episode_{episode}_settle_timeout")
    check_ball_support(ball, stage=f"episode_{episode}_settle_timeout", reset_center=ball_reset_center)
    raise RuntimeError(
        f"SCENE_SETTLE_TIMEOUT: episode={episode} did not keep block and ball stable for {SCENE_SETTLE_REQUIRED_STEPS} consecutive steps within {timeout_steps} physics steps"
    )


def finger_surface_points(robot: Articulation) -> torch.Tensor:
    tip_indices = [robot.body_names.index("tip_left"), robot.body_names.index("tip_right")]
    tip_poses_w = robot.data.body_link_pose_w.torch[0, tip_indices]
    local_centers = torch.tensor(
        (LEFT_PAD_LOCAL_POS, RIGHT_PAD_LOCAL_POS), device=robot.device, dtype=tip_poses_w.dtype
    )
    return tip_poses_w[:, :3] + quat_apply(tip_poses_w[:, 3:], local_centers)


def grasp_geometry_metrics(robot, ball):
    pad_centers = finger_surface_points(robot)
    center_delta = pad_centers[1] - pad_centers[0]
    center_distance = torch.linalg.vector_norm(center_delta)
    grasp_axis = center_delta / center_distance.clamp_min(1.0e-8)
    node_projections = ball.data.nodal_pos_w.torch[0] @ grasp_axis
    return center_distance.item(), (node_projections.max() - node_projections.min()).item()


def grasp_point_kinematics(root_pose_w, arm_pos):
    gripper_pos_w, gripper_rot_w, geometric_jacobian_w = forward_kinematics_and_jacobian(root_pose_w, arm_pos)
    local_offset = torch.tensor(GRASP_POINT_LOCAL, device=arm_pos.device, dtype=arm_pos.dtype)
    offset_w = (gripper_rot_w @ local_offset.expand(arm_pos.shape[0], 3).unsqueeze(-1)).squeeze(-1)
    grasp_pos_w = gripper_pos_w + offset_w
    angular_columns = geometric_jacobian_w[:, 3:, :].transpose(1, 2)
    point_velocity_columns = torch.linalg.cross(
        angular_columns, offset_w.unsqueeze(1).expand_as(angular_columns), dim=-1
    )
    point_jacobian_w = geometric_jacobian_w[:, :3, :] + point_velocity_columns.transpose(1, 2)
    return (
        gripper_pos_w,
        gripper_rot_w,
        grasp_pos_w,
        torch.cat((point_jacobian_w, geometric_jacobian_w[:, 3:, :]), dim=1),
    )


def move_grasp_point(
    sim, robot, block, ball, target_pos_w, gripper_target, num_steps, maintain_orientation=True, target_rotation_w=None
):
    ee_body_idx = robot.body_names.index("gripper")
    limits = robot.data.joint_pos_limits.torch[0, :6]
    initial_fk_error = 0.0
    target_rot_w = (
        torch.tensor(GRASP_ROTATION_W, device=robot.device, dtype=robot.data.joint_pos.torch.dtype).unsqueeze(0)
        if target_rotation_w is None
        else target_rotation_w
    )
    _, _, start_pos, _ = grasp_point_kinematics(robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6])
    travel_steps = max(num_steps, int(torch.linalg.vector_norm(target_pos_w - start_pos).item() / 0.0005) + 1)
    for step in range(travel_steps + 60):
        waypoint = start_pos + min(1.0, (step + 1) / travel_steps) * (target_pos_w - start_pos)
        root_pose_w = robot.data.root_link_pose_w.torch
        arm_pos = robot.data.joint_pos.torch[:, :6]
        gripper_pos_w, gripper_rot_w, grasp_pos_w, jac = grasp_point_kinematics(root_pose_w, arm_pos)
        if step == 0:
            simulated_pos_w = robot.data.body_link_pose_w.torch[:, ee_body_idx, :3]
            initial_fk_error = torch.linalg.vector_norm(gripper_pos_w - simulated_pos_w, dim=-1).item()
        if maintain_orientation:
            joint_delta = damped_least_squares_pose_step(
                grasp_pos_w, waypoint, gripper_rot_w, target_rot_w, jac, damping=0.05
            )
        else:
            joint_delta = damped_least_squares_position_step(grasp_pos_w, waypoint, jac[:, :3, :], damping=0.05)
        arm_target = torch.clamp(
            arm_pos + torch.clamp(joint_delta, min=-0.02, max=0.02), min=limits[:, 0], max=limits[:, 1]
        )
        full_target = robot.data.joint_pos.torch.clone()
        full_target[:, :6] = arm_target
        full_target[:, 6:] = gripper_target
        step_scene(sim, robot, block, ball, full_target)
    _, _, final_grasp_pos_w, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
    )
    return torch.linalg.vector_norm(final_grasp_pos_w - target_pos_w, dim=-1).item(), initial_fk_error


def solve_approach_joint_target(robot, target_pos_w, target_rotation_w, iterations=300):
    arm_target = torch.tensor((APPROACH_IK_SEED,), device=robot.device, dtype=robot.data.joint_pos.torch.dtype)
    limits = robot.data.joint_pos_limits.torch[0, :6]
    root_pose_w = robot.data.root_link_pose_w.torch
    for _ in range(iterations):
        _, current_rotation_w, _ = forward_kinematics_and_jacobian(root_pose_w, arm_target)
        _, _, grasp_pos_w, grasp_jacobian_w = grasp_point_kinematics(root_pose_w, arm_target)
        joint_delta = damped_least_squares_pose_step(
            grasp_pos_w, target_pos_w, current_rotation_w, target_rotation_w, grasp_jacobian_w, damping=0.03
        )
        arm_target = torch.clamp(
            arm_target + torch.clamp(joint_delta, min=-0.05, max=0.05), min=limits[:, 0], max=limits[:, 1]
        )
    _, solved_rotation_w, solved_pos_w, _ = grasp_point_kinematics(root_pose_w, arm_target)
    return (
        arm_target,
        torch.linalg.vector_norm(solved_pos_w - target_pos_w, dim=-1).item(),
        torch.linalg.matrix_norm(solved_rotation_w - target_rotation_w, dim=(-2, -1)).item(),
    )


def execute_arm_trajectory(sim, robot, block, ball, target_arm_pos, gripper_target, num_steps):
    start_arm_pos = robot.data.joint_pos.torch[:, :6].clone()
    for step in range(num_steps):
        phase = (step + 1) / num_steps
        alpha = phase * phase * (3.0 - 2.0 * phase)
        full_target = robot.data.joint_pos.torch.clone()
        full_target[:, :6] = start_arm_pos + alpha * (target_arm_pos - start_arm_pos)
        full_target[:, 6:] = gripper_target
        step_scene(sim, robot, block, ball, full_target)


def hold_grasp(sim, robot, block, ball, target_pos_w, gripper_start, gripper_end, num_steps, target_rotation_w):
    target = robot.data.joint_pos.torch.clone()
    steps = max(num_steps, int(abs(gripper_end - gripper_start) / (0.005 * sim.get_physics_dt())) + 1)
    for step in range(steps):
        target[:, 6:] = gripper_start + (gripper_end - gripper_start) * (step + 1) / steps
        step_scene(sim, robot, block, ball, target)


def check_gripper_motion(sim, robot, block, ball):
    print("STATE=GRIPPER_CHECK", flush=True)
    gaps = []
    for start, end in ((GRIPPER_OPEN_POSITION, 0.0), (0.0, GRIPPER_OPEN_POSITION)):
        hold_grasp(sim, robot, block, ball, ball_center(ball), start, end, 240, None)
        target = robot.data.joint_pos.torch.clone()
        target[:, 6:] = end
        for _ in range(120):
            step_scene(sim, robot, block, ball, target)
        gap, _ = grasp_geometry_metrics(robot, ball)
        gaps.append(gap)
        print(f"FINGER_TARGET={end:.5f} ACTUAL={robot.data.joint_pos.torch[0,6:].tolist()} GAP={gap:.6f}", flush=True)
    if gaps[0] > 0.005 or gaps[1] < 0.085:
        raise RuntimeError(f"FAILURE: finger open/close geometry: {gaps}")
    print("GRIPPER_CHECK_OK", flush=True)


def check_ball_in_gripper(robot, ball):
    middle = finger_surface_points(robot).mean(dim=0)
    if torch.linalg.vector_norm(ball_center(ball)[0] - middle).item() > 0.025:
        raise RuntimeError("FAILURE: ball left finger region")
    extent = ball.data.nodal_pos_w.torch[0].amax(dim=0) - ball.data.nodal_pos_w.torch[0].amin(dim=0)
    if extent.min().item() < BALL_RADIUS * 0.6 or extent.max().item() > BALL_RADIUS * 3.0:
        raise RuntimeError("FAILURE: excessive deformation")


def pick_and_lift_ball(sim, robot, block, ball, initial_ball_pos, motion_steps, lift_height):
    print("STATE=PRE_GRASP", flush=True)
    device = robot.device
    dtype = robot.data.joint_pos.torch.dtype
    initial_ball_target = torch.tensor((initial_ball_pos,), device=device, dtype=dtype)
    pregrasp_offset = torch.tensor(((0.0, 0.0, 0.10),), device=device, dtype=dtype)
    pregrasp_target = initial_ball_target + pregrasp_offset
    approach_rotation_w = torch.tensor(GRASP_ROTATION_W, device=device, dtype=dtype).unsqueeze(0)
    pregrasp_joint_target, solved_position_error, solved_rotation_error = solve_approach_joint_target(
        robot, pregrasp_target, approach_rotation_w
    )
    if solved_position_error > 0.002 or solved_rotation_error > 0.02:
        raise RuntimeError(
            f"Pregrasp IK did not converge: position_error={solved_position_error:.6f}, rotation_error={solved_rotation_error:.6f}"
        )
    execute_arm_trajectory(sim, robot, block, ball, pregrasp_joint_target, GRIPPER_OPEN_POSITION, motion_steps)

    # 机械臂到球上方后重新确认沉降完成；使用保持命令，不重置球或清零节点速度。
    approach_hold = robot.data.joint_pos.torch.clone()
    approach_hold[:, :6] = pregrasp_joint_target
    approach_hold[:, 6:] = GRIPPER_OPEN_POSITION
    wait_until_scene_settled(sim, robot, block, ball, approach_hold,
                            episode=0, minimum_steps=30, ball_reset_center=initial_ball_pos)

    tracked_pregrasp_target = ball_center(ball).clone() + pregrasp_offset
    tracked_joint_target, solved_position_error, solved_rotation_error = solve_approach_joint_target(
        robot, tracked_pregrasp_target, approach_rotation_w
    )
    if solved_position_error > 0.002 or solved_rotation_error > 0.02:
        raise RuntimeError(
            f"Tracked pregrasp IK did not converge: position_error={solved_position_error:.6f}, rotation_error={solved_rotation_error:.6f}"
        )
    execute_arm_trajectory(
        sim, robot, block, ball, tracked_joint_target, GRIPPER_OPEN_POSITION, max(60, motion_steps // 2)
    )
    _, _, actual_pregrasp_pos_w, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
    )
    pregrasp_error = torch.linalg.vector_norm(actual_pregrasp_pos_w - tracked_pregrasp_target, dim=-1).item()
    gripper_index = robot.body_names.index("gripper")
    fk_error = torch.linalg.vector_norm(
        forward_kinematics_and_jacobian(robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6])[0]
        - robot.data.body_link_pose_w.torch[:, gripper_index, :3],
        dim=-1,
    ).item()
    print(
        f"PICK_LIFT_STAGE=pregrasp error_m={pregrasp_error:.6f} ball_center={ball_center(ball)[0].cpu().tolist()}",
        flush=True,
    )
    print(f"FINGER_SURFACE_POINTS_PREGRASP={finger_surface_points(robot).cpu().tolist()}", flush=True)
    if pregrasp_error > 0.005:
        raise RuntimeError(f"FAILURE PRE_GRASP: tracking error {pregrasp_error}")
    gap, width = grasp_geometry_metrics(robot, ball)
    # 软体接触包络包含两侧 particle_radius，不能只按可视网格宽度判断能否进入。
    required_gap = width + 2 * BALL_PARTICLE_RADIUS + 0.001
    if gap < required_gap:
        raise RuntimeError(f"FAILURE OPEN: gap={gap}, required_contact_envelope={required_gap}")

    print("STATE=DESCEND", flush=True)
    grasp_target = ball_center(ball).clone()
    grasp_error, _ = move_grasp_point(
        sim,
        robot,
        block,
        ball,
        grasp_target,
        GRIPPER_OPEN_POSITION,
        motion_steps,
        target_rotation_w=approach_rotation_w,
    )
    print(
        f"PICK_LIFT_STAGE=grasp error_m={grasp_error:.6f} ball_center={ball_center(ball)[0].cpu().tolist()}", flush=True
    )
    print(f"FINGER_SURFACE_POINTS_OPEN={finger_surface_points(robot).cpu().tolist()}", flush=True)
    if grasp_error > 0.005:
        raise RuntimeError(f"FAILURE DESCEND: tracking error {grasp_error}")
    lateral_error = torch.linalg.vector_norm(ball_center(ball)[0, :2] - grasp_target[0, :2]).item()
    if lateral_error > 0.003:
        raise RuntimeError(f"FAILURE DESCEND: ball moved sideways {lateral_error:.6f} m; do not close off-center")
    check_ball_in_gripper(robot, ball)

    print("STATE=CLOSE_GRIPPER", flush=True)
    hold_grasp(
        sim,
        robot,
        block,
        ball,
        grasp_target,
        GRIPPER_OPEN_POSITION,
        GRIPPER_GRASP_POSITION,
        max(60, motion_steps // 2),
        approach_rotation_w,
    )
    print(
        f"PICK_LIFT_STAGE=closed ball_center={ball_center(ball)[0].cpu().tolist()} finger_joints={robot.data.joint_pos.torch[0, 6:].cpu().tolist()}",
        flush=True,
    )
    print(f"FINGER_SURFACE_POINTS_CLOSED={finger_surface_points(robot).cpu().tolist()}", flush=True)
    measured_gap, ball_width_after_grasp = grasp_geometry_metrics(robot, ball)
    print(
        f"PHYSICAL_GRASP target_gap_m={GRIPPER_TARGET_GAP:.6f} measured_gap_m={measured_gap:.6f} ball_width_on_grasp_axis_m={ball_width_after_grasp:.6f}",
        flush=True,
    )
    closed_target = robot.data.joint_pos.torch.clone()
    closed_target[:, 6:] = GRIPPER_GRASP_POSITION
    print("STATE=HOLD", flush=True)
    for _ in range(120):
        step_scene(sim, robot, block, ball, closed_target)
        check_ball_in_gripper(robot, ball)
    if args_cli.stop_after_hold:
        print("CLOSE_HOLD_COMPLETED_NOT_LIFT_SUCCESS", flush=True)
        return {"hold_only": 1.0}

    print("STATE=LIFT", flush=True)
    center_before_lift = ball_center(ball).clone()
    _, _, grasp_before_lift, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
    )
    lift_target = grasp_before_lift.clone()
    lift_target[:, 2] += lift_height
    lift_error, _ = move_grasp_point(
        sim,
        robot,
        block,
        ball,
        lift_target,
        GRIPPER_GRASP_POSITION,
        motion_steps * 2,
        target_rotation_w=approach_rotation_w,
    )
    print(
        f"PICK_LIFT_STAGE=lift error_m={lift_error:.6f} ball_center={ball_center(ball)[0].cpu().tolist()}", flush=True
    )
    final_target = robot.data.joint_pos.torch.clone()
    final_target[:, 6:] = GRIPPER_GRASP_POSITION
    for _ in range(120):
        step_scene(sim, robot, block, ball, final_target)
        check_ball_in_gripper(robot, ball)
        lowest = ball.data.nodal_pos_w.torch[..., 2].min().item() - BALL_PARTICLE_RADIUS
        rise = (ball_center(ball)[:, 2] - center_before_lift[:, 2]).item()
        if lowest < TABLE_TOP_Z + 0.002 or rise < lift_height * MIN_LIFT_RATIO:
            raise RuntimeError(f"FAILURE LIFT_HOLD: bottom={lowest}, rise={rise}")
    center_after_lift = ball_center(ball)
    actual_lift = (center_after_lift[:, 2] - center_before_lift[:, 2]).item()
    return {
        "pregrasp_error_m": pregrasp_error,
        "grasp_error_m": grasp_error,
        "lift_target_error_m": lift_error,
        "fk_alignment_error_m": fk_error,
        "commanded_lift_m": lift_height,
        "actual_ball_lift_m": actual_lift,
        "ball_height_before_m": center_before_lift[:, 2].item(),
        "ball_height_after_m": center_after_lift[:, 2].item(),
    }


def main() -> None:
    if args_cli.episodes < 1 or args_cli.steps_per_episode < 1:
        raise ValueError("--episodes and --steps-per-episode must both be positive")
    if args_cli.motion_steps < 1:
        raise ValueError("--motion-steps must be positive")
    if args_cli.lift_height <= 0.0:
        raise ValueError("--lift-height must be positive")
    if args_cli.pick_lift and args_cli.episodes != 1:
        raise ValueError("--pick-lift currently supports exactly one episode")
    if args_cli.gripper_check and (args_cli.pick_lift or args_cli.episodes != 1):
        raise ValueError("--gripper-check requires --episodes 1 and no --pick-lift")
    if args_cli.stop_after_hold and not args_cli.pick_lift:
        raise ValueError("--stop-after-hold requires --pick-lift")

    mode = (
        "GRIPPER_CHECK (arm fixed)"
        if args_cli.gripper_check
        else ("PICK_LIFT" if args_cli.pick_lift else "RESET_ONLY (arm fixed)")
    )
    print(f"RUN_MODE={mode}", flush=True)
    sim = make_sim()
    sim.set_camera_view(eye=(1.25, -1.20, 1.15), target=(0.12, 0.0, TABLE_TOP_Z))
    robot, block, ball = spawn_scene()
    sim.reset()
    if robot.num_joints != 8:
        raise RuntimeError(f"Expected 8 YAM joints, got {robot.num_joints}: {robot.joint_names}")

    rng = random.Random(args_cli.seed)
    results = []
    pick_lift_result = None
    for episode in range(args_cli.episodes):
        block_pos, ball_pos = sample_object_positions(rng)
        ball_reset_center = reset_episode(robot, block, ball, block_pos, ball_pos)
        settle_steps = wait_until_scene_settled(
            sim,
            robot,
            block,
            ball,
            robot.data.default_joint_pos.torch,
            episode=episode,
            minimum_steps=args_cli.steps_per_episode,
            ball_reset_center=ball_reset_center,
        )

        if args_cli.gripper_check:
            check_gripper_motion(sim, robot, block, ball)
        if args_cli.pick_lift:
            settled_ball_pos = tuple(ball_center(ball)[0].cpu().tolist())
            pick_lift_result = pick_and_lift_ball(
                sim, robot, block, ball, settled_ball_pos, args_cli.motion_steps, args_cli.lift_height
            )

        check_block_support(block, stage=f"episode_{episode}_complete")
        check_ball_support(ball, stage=f"episode_{episode}_complete", reset_center=ball_reset_center)

        tensors = (
            robot.data.joint_pos.torch,
            block.data.root_pos_w.torch,
            block.data.root_com_vel_w.torch,
            ball.data.nodal_state_w.torch,
        )
        if not all(torch.isfinite(value).all() for value in tensors):
            raise RuntimeError(f"Non-finite simulation state in episode {episode}")

        measured_block = block.data.root_pos_w.torch[0].cpu().tolist()
        measured_ball = ball.data.nodal_pos_w.torch[0].mean(dim=0).cpu().tolist()
        # 这是整次流程位移，包含抓取；不能当作纯初始化漂移。
        ball_xy_drift = (
            (measured_ball[0] - ball_reset_center[0]) ** 2 + (measured_ball[1] - ball_reset_center[1]) ** 2
        ) ** 0.5
        results.append(
            {
                "episode": episode,
                "settle_steps": settle_steps,
                "sampled_block": [round(v, 4) for v in block_pos],
                "sampled_ball": [round(v, 4) for v in ball_pos],
                "reset_ball_center": [round(v, 4) for v in ball_reset_center],
                "measured_block": [round(v, 4) for v in measured_block],
                "measured_ball_center": [round(v, 4) for v in measured_ball],
                "ball_xy_drift_m": round(ball_xy_drift, 6),
            }
        )

    print(f"NEWTON_SOLVER={type(sim.cfg.physics.solver_cfg).__name__}")
    print(f"YAM_JOINT_NAMES={robot.joint_names}")
    print(f"YAM_BODY_NAMES={robot.body_names}")
    gripper_index = robot.body_names.index("gripper")
    print(f"YAM_GRIPPER_POSE={robot.data.body_link_pose_w.torch[0, gripper_index].cpu().tolist()}")
    print("YAM_KINEMATICS=project URDF-derived Torch FK/Jacobian")
    for result in results:
        print(f"RESET_RESULT={result}")
    if pick_lift_result is not None and not pick_lift_result.get("hold_only"):
        print(f"PICK_LIFT_RESULT={pick_lift_result}")
        if pick_lift_result["fk_alignment_error_m"] > 0.002:
            raise RuntimeError(f"URDF FK does not align with the simulated gripper: {pick_lift_result}")
        minimum_lift = args_cli.lift_height * MIN_LIFT_RATIO
        if pick_lift_result["actual_ball_lift_m"] < minimum_lift:
            raise RuntimeError(f"Ball was not lifted high enough; required {minimum_lift:.3f} m: {pick_lift_result}")
        print("YAM_DEFORMABLE_BALL_LIFT_OK")
    print("YAM_DEFORMABLE_TASK_INTEGRATION_OK")

    if args_cli.keep_open:
        if args_cli.headless:
            raise ValueError("--keep-open requires the Kit visualizer; use --visualizer kit.")
        print("GUI_READY_CLOSE_WINDOW_TO_EXIT", flush=True)
        hold_target = robot.data.joint_pos.torch.clone()
        while simulation_app.is_running():
            step_scene(sim, robot, block, ball, hold_target)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GRASP_FAILURE={type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        simulation_app.close()
