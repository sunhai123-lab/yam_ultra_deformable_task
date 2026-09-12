"""Newton integration smoke test for the YAM Ultra deformable-ball task.

This is the first executable milestone of the task.  It deliberately contains no
policy or task code copied from another robotics project: the scene, reset logic,
and checks are built directly with Isaac Lab APIs.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from isaaclab.app import AppLauncher

# Command-line arguments
# --episodes：测试多少个 episode；
# --steps-per-episode：每个 episode 运行多少个物理步；
# --seed：随机种子，用于保证随机位置可复现；
# --pick-lift：执行硬编码抓取和抬升；
# --motion-steps：每个机械臂移动阶段的物理步数；
# --lift-height：抬升高度；
# --keep-open：测试结束后是否保持 GUI 打开。
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=3, help="Number of randomized resets to exercise.")
parser.add_argument("--steps-per-episode", type=int, default=30, help="Physics steps after each reset.")
parser.add_argument("--seed", type=int, default=7, help="Seed for deterministic object randomization.")
parser.add_argument("--pick-lift", action="store_true", help="Grasp the deformable ball and lift it.")
parser.add_argument("--motion-steps", type=int, default=240, help="Physics steps for each arm-motion phase.")
parser.add_argument("--lift-height", type=float, default=0.4, help="Commanded world-Z lift distance in meters.")
parser.add_argument("--keep-open", action="store_true", help="Keep the Kit GUI open after the checks finish.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

#启动 Isaac Lab应用程序
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_newton.assets import Articulation, RigidObject
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext
#导入 YAM Ultra 机器人运动学工具
from yam_ultra_deformable_place.yam_kinematics import (
    damped_least_squares_position_step,
    damped_least_squares_pose_step,
    forward_kinematics_and_jacobian,
)
from isaaclab_contrib.deformable import CoupledMJWarpVBDSolverCfg, DeformableObject, VBDSolverCfg

#全局变量
PROJECT_ROOT = Path(__file__).resolve().parents[1]
YAM_USD = PROJECT_ROOT / "assets/generated/yam_ultra_2/yam_ultra/yam_ultra.usda"

TABLE_SIZE = (1.10, 0.75, 0.08)
TABLE_CENTER = (0.12, 0.0, 0.40)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0
BLOCK_SIZE = (0.07, 0.07, 0.05)
BALL_RADIUS = 0.04
GRASP_POINT_LOCAL = (0.0, 0.0, -0.10)
GRASP_ROTATION_W = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
GRIPPER_OPEN_POSITION = 0.0
GRIPPER_GRASP_POSITION = -0.004
MIN_LIFT_RATIO = 0.85
LIFT_IK_SEED = (-0.37, 1.86, 2.73, -1.53, 0.30, -2.02)
BallAttachment = tuple[torch.Tensor, torch.Tensor]


def make_sim() -> SimulationContext:
    """Create a two-way coupled MJWarp rigid + VBD deformable Newton simulation."""
    solver_cfg = CoupledMJWarpVBDSolverCfg(
        rigid_solver_cfg=MJWarpSolverCfg(
            njmax=256,
            nconmax=2048,
            ls_iterations=20,
            ls_parallel=False,
            impratio=1,
            cone="pyramidal",
            integrator="implicitfast",
        ),
        soft_solver_cfg=VBDSolverCfg(
            iterations=5,
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
            physics=NewtonCfg(solver_cfg=solver_cfg, num_substeps=4, use_cuda_graph=False),
        )
    )


def spawn_scene() -> tuple[Articulation, RigidObject, DeformableObject]:
    """Spawn the robot, static workspace, rigid target block, and soft ball."""
    if not YAM_USD.is_file():
        raise FileNotFoundError(f"Converted YAM asset is missing: {YAM_USD}")

    # Create a simple scene with a table, light, and ground plane.
    sim_utils.create_prim("/World/env_0", "Xform")
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    table_cfg = sim_utils.CuboidCfg(
        size=TABLE_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.22, 0.12)),
    )
    table_cfg.func("/World/env_0/Table", table_cfg, translation=TABLE_CENTER)

    # Spawn the YAM Ultra robot with a simple joint drive model and initial joint configuration.
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
                "joint7": -0.02,
                "joint8": -0.02,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                effort_limit_sim=120.0,
                velocity_limit_sim=3.0,
                stiffness=500.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint[7-8]"],
                effort_limit_sim=60.0,
                velocity_limit_sim=1.0,
                stiffness=500.0,
                damping=30.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    sim_utils.make_uninstanceable("/World/env_0/Robot")
    sim_utils.modify_collision_properties(
        "/World/env_0/Robot",
        sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )

    block = RigidObject(
        RigidObjectCfg(
            prim_path="/World/env_0/TargetBlock",
            spawn=sim_utils.CuboidCfg(
                size=BLOCK_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.16, TABLE_TOP_Z + BLOCK_SIZE[2] / 2.0)),
        )
    )
    #Young's modulus：描述材料刚硬程度；
    #Poisson's ratio：描述材料受压后横向变形特性；
    youngs_modulus = 5.0e4
    poissons_ratio = 0.35
    ball = DeformableObject(
        DeformableObjectCfg(
            prim_path="/World/env_0/DeformableBall",
            spawn=sim_utils.MeshSphereCfg(
                radius=BALL_RADIUS,
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.18, 0.12)),
                physics_material=NewtonDeformableBodyMaterialCfg(
                    #density：密度；
                    #k_mu、k_lambda：由材料参数换算得到的弹性参数；
                    #particle_radius：软体内部粒子相关尺寸。
                    density=500.0,
                    k_mu=youngs_modulus / (2.0 * (1.0 + poissons_ratio)),
                    k_lambda=(
                        youngs_modulus
                        * poissons_ratio
                        / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
                    ),
                    particle_radius=0.006,
                ),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.15, -0.16, TABLE_TOP_Z + BALL_RADIUS + 0.01)),
        )
    )
    return robot, block, ball

#
def sample_object_positions(rng: random.Random) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Sample separated start positions from two reachable tabletop regions."""
    block_pos = (
        rng.uniform(0.02, 0.16),
        rng.uniform(0.08, 0.18),
        TABLE_TOP_Z + BLOCK_SIZE[2] / 2.0 + 0.004,
    )
    ball_pos = (
        rng.uniform(0.02, 0.16),
        rng.uniform(-0.18, -0.08),
        TABLE_TOP_Z + BALL_RADIUS + 0.012,
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
    prepare_grasp: bool = False,
) -> torch.Tensor | None:
    """Reset every dynamic state, including all deformable nodes."""
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
    current_center = nodal_state[..., :3].mean(dim=1)
    desired_center = torch.tensor(ball_pos, device=nodal_state.device).unsqueeze(0)
    nodal_state[..., :3] += (desired_center - current_center).unsqueeze(1)
    nodal_state[..., 3:] = 0.0
    ball.write_nodal_state_to_sim_index(nodal_state)

    kinematic_target = ball.data.nodal_kinematic_target.torch.clone()
    kinematic_target[..., :3] = nodal_state[..., :3]
    kinematic_target[..., 3] = 1.0
    pinned_node_indices = None
    if prepare_grasp:
        centered_y = nodal_state[0, :, 1] - nodal_state[0, :, 1].mean()
        nodes_per_side = min(6, nodal_state.shape[1] // 2)
        positive_indices = torch.topk(centered_y, k=nodes_per_side).indices
        negative_indices = torch.topk(-centered_y, k=nodes_per_side).indices
        pinned_node_indices = torch.cat((positive_indices, negative_indices))
        kinematic_target[0, pinned_node_indices, 3] = 0.0
    ball.write_nodal_kinematic_target_to_sim_index(kinematic_target)

    robot.reset()
    block.reset()
    ball.reset()
    return pinned_node_indices


def step_scene(
    sim: SimulationContext,
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    joint_target: torch.Tensor,
    attachment: BallAttachment | None = None,
) -> None:
    """Apply one joint target and advance every scene object by one physics step."""
    robot.set_joint_position_target_index(target=joint_target)
    robot.write_data_to_sim()
    if attachment is not None:
        update_ball_attachment(robot, ball, attachment)
    ball.write_data_to_sim()
    sim.step(render=not args_cli.headless)
    dt = sim.get_physics_dt()
    robot.update(dt)
    block.update(dt)
    ball.update(dt)


def ball_center(ball: DeformableObject) -> torch.Tensor:
    """Return the current world-space center of the deformable ball."""
    return ball.data.nodal_pos_w.torch.mean(dim=1)


def create_ball_attachment(
    robot: Articulation,
    ball: DeformableObject,
    pinned_node_indices: torch.Tensor,
) -> BallAttachment:
    """Store the pre-pinned surface patches in the current gripper frame."""
    gripper_pos_w, gripper_rot_w, _ = forward_kinematics_and_jacobian(
        robot.data.root_link_pose_w.torch,
        robot.data.joint_pos.torch[:, :6],
    )
    pinned_nodes_w = ball.data.nodal_pos_w.torch[0, pinned_node_indices]
    pinned_nodes_local = (
        gripper_rot_w[0].transpose(0, 1) @ (pinned_nodes_w - gripper_pos_w[0]).transpose(0, 1)
    ).transpose(0, 1)
    return pinned_node_indices, pinned_nodes_local.clone()


def update_ball_attachment(robot: Articulation, ball: DeformableObject, attachment: BallAttachment) -> None:
    """Move the already-kinematic surface patches with the current gripper frame."""
    pinned_node_indices, pinned_nodes_local = attachment
    gripper_pos_w, gripper_rot_w, _ = forward_kinematics_and_jacobian(
        robot.data.root_link_pose_w.torch,
        robot.data.joint_pos.torch[:, :6],
    )
    attached_nodes_w = (
        gripper_rot_w[0] @ pinned_nodes_local.transpose(0, 1)
    ).transpose(0, 1) + gripper_pos_w[0]
    targets = ball.data.nodal_kinematic_target.torch.clone()
    targets[0, pinned_node_indices, :3] = attached_nodes_w
    ball.write_nodal_kinematic_target_to_sim_index(targets)


def grasp_point_kinematics(
    root_pose_w: torch.Tensor,
    arm_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gripper-link position, grasp-point position, and grasp-point Jacobian."""
    gripper_pos_w, gripper_rot_w, geometric_jacobian_w = forward_kinematics_and_jacobian(root_pose_w, arm_pos)
    local_offset = torch.tensor(GRASP_POINT_LOCAL, device=arm_pos.device, dtype=arm_pos.dtype)
    offset_w = (gripper_rot_w @ local_offset.expand(arm_pos.shape[0], 3).unsqueeze(-1)).squeeze(-1)
    grasp_pos_w = gripper_pos_w + offset_w

    angular_columns = geometric_jacobian_w[:, 3:, :].transpose(1, 2)
    point_velocity_columns = torch.linalg.cross(
        angular_columns,
        offset_w.unsqueeze(1).expand_as(angular_columns),
        dim=-1,
    )
    point_jacobian_w = geometric_jacobian_w[:, :3, :] + point_velocity_columns.transpose(1, 2)
    grasp_geometric_jacobian_w = torch.cat((point_jacobian_w, geometric_jacobian_w[:, 3:, :]), dim=1)
    return gripper_pos_w, gripper_rot_w, grasp_pos_w, grasp_geometric_jacobian_w


def move_grasp_point(
    sim: SimulationContext,
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    target_pos_w: torch.Tensor,
    gripper_target: float,
    num_steps: int,
    maintain_orientation: bool = True,
    attachment: BallAttachment | None = None,
) -> tuple[float, float]:
    """Move the point between the fingers to a world-space target using DLS IK."""
    ee_body_idx = robot.body_names.index("gripper")
    limits = robot.data.joint_pos_limits.torch[0, :6]
    initial_fk_error = 0.0
    target_rot_w = torch.tensor(GRASP_ROTATION_W, device=robot.device, dtype=robot.data.joint_pos.torch.dtype).unsqueeze(0)

    for step in range(num_steps):
        root_pose_w = robot.data.root_link_pose_w.torch
        arm_pos = robot.data.joint_pos.torch[:, :6]
        gripper_pos_w, gripper_rot_w, grasp_pos_w, grasp_geometric_jacobian_w = grasp_point_kinematics(
            root_pose_w, arm_pos
        )
        if step == 0:
            simulated_pos_w = robot.data.body_link_pose_w.torch[:, ee_body_idx, :3]
            initial_fk_error = torch.linalg.vector_norm(gripper_pos_w - simulated_pos_w, dim=-1).item()

        if maintain_orientation:
            joint_delta = damped_least_squares_pose_step(
                grasp_pos_w,
                target_pos_w,
                gripper_rot_w,
                target_rot_w,
                grasp_geometric_jacobian_w,
                damping=0.05,
            )
        else:
            joint_delta = damped_least_squares_position_step(
                grasp_pos_w,
                target_pos_w,
                grasp_geometric_jacobian_w[:, :3, :],
                damping=0.05,
            )
        arm_target = arm_pos + torch.clamp(joint_delta, min=-0.02, max=0.02)
        arm_target = torch.clamp(arm_target, min=limits[:, 0], max=limits[:, 1])

        full_target = robot.data.joint_pos.torch.clone()
        full_target[:, :6] = arm_target
        full_target[:, 6:] = gripper_target
        step_scene(sim, robot, block, ball, full_target, attachment)

    _, _, final_grasp_pos_w, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch,
        robot.data.joint_pos.torch[:, :6],
    )
    final_error = torch.linalg.vector_norm(final_grasp_pos_w - target_pos_w, dim=-1).item()
    return final_error, initial_fk_error


def solve_lift_joint_target(
    robot: Articulation,
    target_pos_w: torch.Tensor,
    iterations: int = 300,
) -> tuple[torch.Tensor, float]:
    """Solve the high lift waypoint from a known reachable YAM configuration branch."""
    arm_target = torch.tensor((LIFT_IK_SEED,), device=robot.device, dtype=robot.data.joint_pos.torch.dtype)
    limits = robot.data.joint_pos_limits.torch[0, :6]
    root_pose_w = robot.data.root_link_pose_w.torch

    for _ in range(iterations):
        _, _, grasp_pos_w, grasp_jacobian_w = grasp_point_kinematics(root_pose_w, arm_target)
        joint_delta = damped_least_squares_position_step(
            grasp_pos_w,
            target_pos_w,
            grasp_jacobian_w[:, :3, :],
            damping=0.03,
        )
        arm_target = torch.clamp(
            arm_target + torch.clamp(joint_delta, min=-0.05, max=0.05),
            min=limits[:, 0],
            max=limits[:, 1],
        )

    _, _, solved_pos_w, _ = grasp_point_kinematics(root_pose_w, arm_target)
    solved_error = torch.linalg.vector_norm(solved_pos_w - target_pos_w, dim=-1).item()
    return arm_target, solved_error


def execute_joint_trajectory(
    sim: SimulationContext,
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    target_arm_pos: torch.Tensor,
    num_steps: int,
    attachment: BallAttachment,
) -> None:
    """Track a smooth joint-space path while keeping the closed grasp attached."""
    start_arm_pos = robot.data.joint_pos.torch[:, :6].clone()
    for step in range(num_steps):
        phase = (step + 1) / num_steps
        alpha = phase * phase * (3.0 - 2.0 * phase)
        full_target = robot.data.joint_pos.torch.clone()
        full_target[:, :6] = start_arm_pos + alpha * (target_arm_pos - start_arm_pos)
        full_target[:, 6:] = GRIPPER_GRASP_POSITION
        step_scene(sim, robot, block, ball, full_target, attachment)


def hold_grasp(
    sim: SimulationContext,
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    target_pos_w: torch.Tensor,
    gripper_start: float,
    gripper_end: float,
    num_steps: int,
    attachment: BallAttachment | None = None,
) -> None:
    """Hold the grasp point fixed while smoothly moving both finger joints."""
    for step in range(num_steps):
        alpha = (step + 1) / num_steps
        gripper_target = gripper_start + alpha * (gripper_end - gripper_start)
        move_grasp_point(
            sim,
            robot,
            block,
            ball,
            target_pos_w,
            gripper_target,
            num_steps=1,
            attachment=attachment,
        )


def pick_and_lift_ball(
    sim: SimulationContext,
    robot: Articulation,
    block: RigidObject,
    ball: DeformableObject,
    initial_ball_pos: tuple[float, float, float],
    pinned_node_indices: torch.Tensor,
    motion_steps: int,
    lift_height: float,
) -> dict[str, float]:
    """Open, approach, grasp, and lift the deformable ball with a hard-coded sequence."""
    device = robot.device
    dtype = robot.data.joint_pos.torch.dtype
    ball_target = torch.tensor((initial_ball_pos,), device=device, dtype=dtype)
    grasp_target = ball_target + torch.tensor(((0.0, 0.0, 0.02),), device=device, dtype=dtype)
    pregrasp_target = grasp_target + torch.tensor(((0.0, 0.0, 0.12),), device=device, dtype=dtype)

    pregrasp_error, fk_error = move_grasp_point(
        sim, robot, block, ball, pregrasp_target, GRIPPER_OPEN_POSITION, motion_steps
    )
    print(
        f"PICK_LIFT_STAGE=pregrasp error_m={pregrasp_error:.6f} "
        f"ball_center={ball_center(ball)[0].cpu().tolist()}",
        flush=True,
    )
    grasp_error, _ = move_grasp_point(
        sim, robot, block, ball, grasp_target, GRIPPER_OPEN_POSITION, motion_steps
    )
    print(
        f"PICK_LIFT_STAGE=grasp error_m={grasp_error:.6f} "
        f"ball_center={ball_center(ball)[0].cpu().tolist()}",
        flush=True,
    )
    hold_grasp(
        sim,
        robot,
        block,
        ball,
        grasp_target,
        GRIPPER_OPEN_POSITION,
        GRIPPER_GRASP_POSITION,
        max(60, motion_steps // 2),
    )
    print(
        f"PICK_LIFT_STAGE=closed ball_center={ball_center(ball)[0].cpu().tolist()} "
        f"finger_joints={robot.data.joint_pos.torch[0, 6:].cpu().tolist()}",
        flush=True,
    )

    attachment = create_ball_attachment(robot, ball, pinned_node_indices)
    print(f"PICK_LIFT_ATTACHMENT_NODES={pinned_node_indices.cpu().tolist()}", flush=True)
    closed_target = robot.data.joint_pos.torch.clone()
    closed_target[:, 6:] = GRIPPER_GRASP_POSITION
    for _ in range(60):
        step_scene(sim, robot, block, ball, closed_target, attachment)

    center_before_lift = ball_center(ball).clone()
    lift_target = grasp_target + torch.tensor(((0.0, 0.0, lift_height),), device=device, dtype=dtype)
    lift_joint_target, solved_lift_error = solve_lift_joint_target(robot, lift_target)
    if solved_lift_error > 0.01:
        raise RuntimeError(f"Lift IK did not converge: error={solved_lift_error:.6f} m")
    execute_joint_trajectory(
        sim,
        robot,
        block,
        ball,
        lift_joint_target,
        num_steps=motion_steps * 2,
        attachment=attachment,
    )
    _, _, final_grasp_pos_w, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch,
        robot.data.joint_pos.torch[:, :6],
    )
    lift_error = torch.linalg.vector_norm(final_grasp_pos_w - lift_target, dim=-1).item()
    print(
        f"PICK_LIFT_STAGE=lift solved_error_m={solved_lift_error:.6f} actual_error_m={lift_error:.6f} "
        f"ball_center={ball_center(ball)[0].cpu().tolist()}",
        flush=True,
    )
    final_target = robot.data.joint_pos.torch.clone()
    final_target[:, :6] = lift_joint_target
    final_target[:, 6:] = GRIPPER_GRASP_POSITION
    for _ in range(120):
        step_scene(sim, robot, block, ball, final_target, attachment)

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

    sim = make_sim()
    sim.set_camera_view(eye=(1.25, -1.20, 1.15), target=(0.12, 0.0, TABLE_TOP_Z))
    robot, block, ball = spawn_scene()
    sim.reset()

    if robot.num_joints != 8:
        raise RuntimeError(f"Expected 8 YAM joints, got {robot.num_joints}: {robot.joint_names}")

    rng = random.Random(args_cli.seed)
    results: list[dict[str, object]] = []
    pick_lift_result: dict[str, float] | None = None
    for episode in range(args_cli.episodes):
        block_pos, ball_pos = sample_object_positions(rng)
        pinned_node_indices = reset_episode(
            robot,
            block,
            ball,
            block_pos,
            ball_pos,
            prepare_grasp=args_cli.pick_lift,
        )

        for _ in range(args_cli.steps_per_episode):
            step_scene(sim, robot, block, ball, robot.data.default_joint_pos.torch)

        if args_cli.pick_lift:
            if pinned_node_indices is None:
                raise RuntimeError("Grasp nodes were not prepared during reset")
            settled_ball_pos = tuple(ball_center(ball)[0].cpu().tolist())
            pick_lift_result = pick_and_lift_ball(
                sim,
                robot,
                block,
                ball,
                settled_ball_pos,
                pinned_node_indices,
                args_cli.motion_steps,
                args_cli.lift_height,
            )

        tensors = (robot.data.joint_pos.torch, block.data.root_pos_w.torch, ball.data.nodal_pos_w.torch)
        if not all(torch.isfinite(value).all() for value in tensors):
            raise RuntimeError(f"Non-finite simulation state in episode {episode}")

        measured_block = block.data.root_pos_w.torch[0].cpu().tolist()
        measured_ball = ball.data.nodal_pos_w.torch[0].mean(dim=0).cpu().tolist()
        results.append(
            {
                "episode": episode,
                "sampled_block": [round(v, 4) for v in block_pos],
                "sampled_ball": [round(v, 4) for v in ball_pos],
                "measured_block": [round(v, 4) for v in measured_block],
                "measured_ball_center": [round(v, 4) for v in measured_ball],
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
    if pick_lift_result is not None:
        print(f"PICK_LIFT_RESULT={pick_lift_result}")
        if pick_lift_result["fk_alignment_error_m"] > 0.002:
            raise RuntimeError(f"URDF FK does not align with the simulated gripper: {pick_lift_result}")
        minimum_lift = args_cli.lift_height * MIN_LIFT_RATIO
        if pick_lift_result["actual_ball_lift_m"] < minimum_lift:
            raise RuntimeError(
                f"Ball was not lifted high enough; required {minimum_lift:.3f} m: {pick_lift_result}"
            )
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
    finally:
        simulation_app.close()
