"""YAM Ultra 抓取并抬升可变形球 —— 100% 学习注释版。

本文件基于仓库分支 ``fix/rigid-block-stability`` 中最新的
``scripts/yam_pick_ball.py``（读取时分支 HEAD: 05874a5，commit message: ``learn to change the range``）整理。

【重要】
1. 这是桌心环形采样学习版，包含方位限位筛选、姿态平滑与实际运动收敛检查。
2. 你现在主要在学“场景参数、坐标、机器人参数、运动控制”，因此所有这类参数都标注了
   “改它会发生什么”和“它依赖谁”。
3. World 坐标统一采用米 m；关节转角采用弧度 rad；线速度 m/s；角速度 rad/s。
4. 软球不是刚体，代码中的 ``ball_center`` 是所有 VBD 节点位置的算术平均，不等同于严格的连续体质心。
5. 当前学习版使用桌心机器人和环形随机工作区；相关参数集中在第 4～5 节。

任务控制流程：
    随机/复位物体
      -> 等场景沉降
      -> 预抓取（pre-grasp）
      -> 下降（descend）
      -> 闭合夹爪（close）
      -> 快速抓取稳定性检查
      -> 抬升（lift）
      -> 可选：放回桌面 / 放到方块上

核心运动学链路：
    当前关节角 q
      -> FK 得到末端位置/姿态
      -> Jacobian J
      -> 目标误差 e
      -> DLS IK 求 Δq
      -> 下发新的 joint position target
      -> Newton 物理推进一步
      -> 读取新的真实状态，继续闭环
"""

from __future__ import annotations

# argparse：解析命令行参数，例如 --episodes 1 --pick-lift。
import argparse

# math：环形随机采样时把半径/方位角转换成世界 XY 坐标。
import math

# random：只用于场景中方块/球的随机位置采样；Random(seed) 让结果可复现。
import random

# time：用于记录真实 wall-clock 时间，与仿真时间区分。
import time

# Path：用相对当前脚本的路径找到项目根目录和机器人 USD。
from pathlib import Path

# ------------------------------------------------------------
# AppLauncher 必须尽量早启动。
# Isaac Lab / Isaac Sim 很多模块依赖 Omniverse Kit 已经初始化，
# 因此不要为了“import 排版整齐”把所有 Isaac 模块都挪到它前面。
# ------------------------------------------------------------
from isaaclab.app import AppLauncher

# ============================================================
# 1. 命令行参数
# ============================================================

# description=__doc__：把文件顶部 docstring 作为 --help 的说明文字。
parser = argparse.ArgumentParser(description=__doc__)
# episode = 一次“复位 -> 沉降 -> 可选抓取/放置 -> 检查”的完整实验。
# default=10 表示不额外传参时连续跑 10 次。
parser.add_argument("--episodes", type=int, default=10, help="随机复位并测试的 episode 数量。")
# 每次 reset 后至少推进多少个 physics step。
# 注意：这不是“固定只跑 30 步”；wait_until_scene_settled() 后面还会继续等，直到方块和球满足连续稳定判据。
parser.add_argument(
    "--steps-per-episode",
    type=int,
    default=30,
    help="每次复位后至少推进的物理步数；随后继续等待刚体方块和软球同时满足稳定判据。",
)
# 随机种子。相同 seed + 相同采样逻辑，应得到相同的随机目标序列。
parser.add_argument("--seed", type=int, default=7, help="物体位置随机化使用的可复现随机种子。")
# action="store_true"：命令行出现 --pick-lift 时为 True；不出现就是 False。
parser.add_argument("--pick-lift", action="store_true", help="执行硬编码抓取和抬升流程。")
# CUDA Graph 是性能优化；调试崩溃、堆栈和动态行为时可以关掉。
parser.add_argument("--disable-cuda-graph", action="store_true", help="调试用：关闭 Newton GPU 执行图加速。")
# 抓起来以后再放回最初位置。
parser.add_argument(
    "--put-back", action="store_true", help="抓起验收后平稳放回本次抓取前的位置；需同时使用 --pick-lift。"
)
# 抓起来以后搬到刚体方块顶面；与 --put-back 互斥。
parser.add_argument(
    "--place-on-block", action="store_true", help="抓起后搬运至刚体方块顶面并释放；需 --pick-lift，与 --put-back 互斥。"
)
# motion-steps 现在不是“每段固定 240 步”，而是时间倍率：240=1x，480≈半速。
parser.add_argument(
    "--motion-steps", type=int, default=240, help="运动时间倍率基准：240=默认速度，480=半速；不再是每阶段固定步数。"
)
# 默认抬升 0.03m=3cm。
parser.add_argument("--lift-height", type=float, default=0.03, help="球心期望沿世界 Z 轴抬升的距离，单位 m。")
# 任务结束后保持 GUI 打开。
parser.add_argument("--keep-open", action="store_true", help="测试结束后保持 Kit GUI，直到手动关闭窗口。")
# 只执行到闭爪稳定，不进入 lift。
parser.add_argument("--stop-after-hold", action="store_true", help="只验收到闭合保持阶段，不抬升。")
# 机械臂不靠近球，只测试夹爪开合。
parser.add_argument("--gripper-check", action="store_true", help="固定机械臂，检查原始夹指张开—闭合—张开，不接近球。")
# Isaac Lab 自带参数，例如 --device、--headless、--visualizer。
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================
# 2. Kit 启动后再导入仿真相关模块
# ============================================================

import torch
from pxr import Sdf, Usd, UsdPhysics  # USD 属性类型、Prim 遍历与碰撞 API。
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
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_from_matrix, quat_slerp
from yam_ultra_deformable_place.yam_kinematics import (
    damped_least_squares_position_step,
    damped_least_squares_pose_step,
    forward_kinematics_and_jacobian,
)
from isaaclab_contrib.deformable import CoupledMJWarpVBDSolverCfg, DeformableObject, VBDSolverCfg
from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg

# ============================================================
# 3. 文件路径
# ============================================================

# 当前脚本位于 <project>/scripts，因此 parents[1] 是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 仿真实际加载的 YAM USD。
YAM_USD = PROJECT_ROOT / "assets/generated/yam_ultra_2/yam_ultra/yam_ultra.usda"

# ============================================================
# 4. 场景几何参数
# ============================================================

# TABLE_SIZE=(X长度,Y宽度,Z厚度)，单位 m。
TABLE_SIZE = (1.50, 1.50, 0.08)
# TABLE_CENTER 是 Cuboid 几何中心，不是桌面高度；当前桌体 z 范围约 [-0.04,+0.04]。
TABLE_CENTER = (0.0, 0.0, 0.0)
# 桌面顶部世界 Z。
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0

# ============================================================
# 5. 方块/球随机采样区域
# ============================================================

# 机器人 base 固定在桌面中心；XY 不能只在这里改，spawn_scene() 也必须引用它。
ROBOT_BASE_XY = TABLE_CENTER[:2]
# 球和方块到机器人中心的随机半径。下限避开底座，上限限制在已验证可达区域。
OBJECT_RADIUS_RANGE = (0.30, 0.45)
# 随机方位角（rad）。径向姿态下 joint1 约等于方位；保留实际限位余量。
OBJECT_BEARING_RANGE = (-2.40, 3.00)
# 方块与球水平距离至少 18cm，避免初始化重叠或任务距离太短。
MIN_OBJECT_PLANAR_DISTANCE = 0.18
# 最多随机尝试 100 次。
SAMPLING_MAX_ATTEMPTS = 100
# 球和方块的方位差最多为 180°。超过此值时，受 joint1 限位影响，机械臂只能
# 绕更长的一侧搬运；这既慢，也会显著增加软球在夹爪内滑动的风险，因此重采样。
MAX_TRANSFER_BEARING_DELTA = math.pi
# 下放前使用实际软球球心闭环对正，而不只检查夹爪 IK 误差。
TRANSFER_BALL_ALIGNMENT_TOLERANCE = 0.002
# 方块 X/Y/Z 尺寸。
BLOCK_SIZE = (0.07, 0.07, 0.05)
# reset 时方块额外离桌 0.5mm。
BLOCK_RESET_CLEARANCE = 5.0e-4

# ============================================================
# 6. 刚体材料与方块稳定判据
# ============================================================

RIGID_STATIC_FRICTION = 0.6  # 静摩擦系数。
RIGID_DYNAMIC_FRICTION = 0.5  # 动摩擦系数。
RIGID_RESTITUTION = 0.0  # 恢复系数；0 表示不希望明显弹跳。
BLOCK_SETTLE_LINEAR_SPEED = 5.0e-4  # m/s。
BLOCK_SETTLE_ANGULAR_SPEED = 1.0e-2  # rad/s。
BLOCK_SETTLE_CLEARANCE = 1.0e-3  # m，底面与桌面的容差。
# Newton 将形状 ke/kd 转换成 MuJoCo solref。保持 ke=2500，增加 kd 至 200，
# 抑制接触法向振荡。这是约束恢复参数，不能当成软球杨氏模量。
SUPPORT_CONTACT_STIFFNESS = 2500.0
SUPPORT_CONTACT_DAMPING = 200.0

# ============================================================
# 7. VBD 软球参数与稳定判据
# ============================================================

BALL_RADIUS = 0.04  # 几何半径 4cm。
BALL_DENSITY = 500.0  # kg/m^3。
BALL_YOUNGS_MODULUS = 5.0e4  # Pa，杨氏模量 E=50kPa。
BALL_POISSONS_RATIO = 0.35  # 泊松比 ν。
BALL_PARTICLE_RADIUS = 0.006  # m，VBD 粒子接触半径 6mm。
BALL_RESET_CLEARANCE = 5.0e-4  # m，reset 额外净空 0.5mm。
BALL_INTERNAL_DAMPING = 0.01  # 软体内部材料阻尼，不是接触阻尼。
BALL_SETTLE_WINDOW_DRIFT = 2.5e-4  # m，稳定窗口内球心 XY 最大漂移。
BALL_SETTLE_ROOT_SPEED = 1.0e-3  # m/s，节点平均速度阈值。
BALL_SETTLE_MAX_NODAL_SPEED = 1.0e-2  # m/s，任一节点最大速度阈值。
BALL_SETTLE_CONTACT_CLEARANCE = 1.5e-3  # m，球底接触包络距桌面阈值。


SCENE_SETTLE_REQUIRED_STEPS = 30  # reset-only 连续稳定步数。
SCENE_SETTLE_TIMEOUT_STEPS = 480  # 最大等待步数，120Hz 下约 4s。
SCENE_DIAGNOSTIC_STEPS = (30, 60, 120, 240, 360)

GRASP_READY_REQUIRED_STEPS = 12
GRASP_READY_ROOT_SPEED = 1.5e-3
GRASP_READY_MAX_NODAL_SPEED = 3.0e-3
GRASP_READY_WINDOW_DRIFT = 5.0e-4

# ============================================================
# 8. 抓取几何
# ============================================================

# 真正抓取点相对 gripper link 原点的局部偏移 [m]。
GRASP_POINT_LOCAL = (0.0, 0.0, -0.125)
# 抓取时希望保持的世界旋转矩阵。
GRASP_ROTATION_W = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
# 左右夹指内侧代表点的 tip-local 坐标 [m]。
LEFT_PAD_LOCAL_POS = (0.014, 0.0469184183, -0.1980911)
RIGHT_PAD_LOCAL_POS = (0.014, 0.0458805300, -0.1981001)
# USD 中指尖路径，用来运行时修改碰撞近似。
LEFT_TIP_PATH = "/World/env_0/Robot/Geometry/base/link1/link2/link3/link4/link5/gripper/tip_left"
RIGHT_TIP_PATH = "/World/env_0/Robot/Geometry/base/link1/link2/link3/link4/link5/gripper/tip_right"

# ============================================================
# 9. 机器人/夹爪速度与抓取目标
# ============================================================

GRIPPER_OPEN_POSITION = -0.04695  # m；YAM prismatic joint 最大张开位置。
GRIPPER_COMMAND_SPEED = 0.04  # m/s，空载夹爪测试。
GRIPPER_CONTACT_SPEED = 0.030  # m/s，碰球时慢速闭合。0.020
ARM_CARTESIAN_SPEED = 0.35  # m/s，非接触末端移动峰值速度基准。
ARM_JOINT_SPEED = 3.0  # rad/s，首次 joint-space approach 峰值速度。
ARM_ORIENTATION_SPEED = 1.2  # rad/s，末端姿态插值的峰值角速度，与关节速度不同。
MOTION_SETTLE_STEPS = 120  # 轨迹走完后最多额外闭环 1 s；超时明确报错。
DESCEND_SPEED = 0.25  # m/s，最后接近球的敏感下降段。
LIFT_SPEED = ARM_CARTESIAN_SPEED
# 对侧长弧搬运时正夹持软球；单独限速，避免旋转惯性在夹爪内积累偏移。
TRANSFER_SPEED = min(ARM_CARTESIAN_SPEED, 0.30)
# 连续搬运时，驱动目标需要适度领先实际关节，避免仅用一步 DLS 增量造成持续滞后。
TRANSFER_DLS_GAIN = 3.0
TRANSFER_MAX_JOINT_COMMAND_STEP = 0.04  # rad/physics-step，只限制目标领先量，不是关节瞬时跳变。
PLACE_APPROACH_SPEED = ARM_CARTESIAN_SPEED
PLACE_SPEED = 0.030  # m/s，最后接触放球阶段。
PLACE_RELEASE_SPEED = 0.012  # m/s，单指卸载释放速度。
SCENE_STEP_COUNT = 0  # 全局累计 physics step 数。
QUICK_GRASP_STABLE_STEPS = 12  # 连续约 0.1s 稳定才通过。
QUICK_GRASP_TIMEOUT_STEPS = 120  # 最多等约 1s。

PLACE_CONTACT_CLEARANCE = 0.0005
PLACE_POSITION_TOLERANCE = 0.005
PLACE_RELEASE_DRIFT = 0.003
BLOCK_PLACEMENT_MAX_DISPLACEMENT = 0.004
GRIPPER_CLOSED_GAP = 0.00006211

GRIPPER_OPEN_GAP = GRIPPER_CLOSED_GAP - 2.0 * GRIPPER_OPEN_POSITION
NOMINAL_GRIP_STRESS = 5.0e3

# ?
BALL_EFFECTIVE_MODULUS = BALL_YOUNGS_MODULUS / (1.0 - BALL_POISSONS_RATIO**2)
BALL_COMPRESSION_RATIO = min(0.12, NOMINAL_GRIP_STRESS / BALL_EFFECTIVE_MODULUS)
GRIPPER_TARGET_GAP = 2.0 * BALL_RADIUS * (1.0 - BALL_COMPRESSION_RATIO)
GRIPPER_GRASP_POSITION = -0.5 * (GRIPPER_TARGET_GAP - GRIPPER_CLOSED_GAP)
MIN_LIFT_RATIO = 0.85
APPROACH_IK_SEED = (-0.48, 1.97, 1.63, -1.23, 0.0, -0.48)

# ============================================================
# 10. Newton physics
# ============================================================


@configclass
class DeformableNewtonCfg(NewtonCfg):
    """NewtonCfg 增加 model-level 刚软接触参数。"""

    model_cfg: NewtonModelCfg | None = None


def make_sim() -> SimulationContext:
    """创建 MJWarp 刚体 + VBD 软体的双向耦合仿真。"""
    solver_cfg = CoupledMJWarpVBDSolverCfg(
        rigid_solver_cfg=MJWarpSolverCfg(
            njmax=256,  # 每个 world/env 的约束容量，不是机器人关节数。
            nconmax=2048,  # 接触点容量。
            ls_iterations=20,  # line-search 最大迭代，不是主 solver iteration。
            ls_parallel=False,
            impratio=1,  # 摩擦/法向约束 impedance 比例。
            cone="pyramidal",  # 多面锥摩擦近似。
            integrator="implicitfast",  # 隐式积分器。
            ccd_iterations=100,  # 凸碰撞/CCD 的迭代预算。
        ),
        soft_solver_cfg=VBDSolverCfg(
            iterations=20,  # VBD 每步迭代数。
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,  # 关闭软球内部自接触。
            particle_collision_detection_interval=-1,
        ),
        coupling_mode="two_way",  # 刚体影响软体，软体也反作用刚体。
    )
    return SimulationContext(
        SimulationCfg(
            dt=1.0 / 120.0,  # 主物理频率 120Hz。
            device=args_cli.device,
            physics=DeformableNewtonCfg(
                solver_cfg=solver_cfg,
                model_cfg=NewtonModelCfg(
                    soft_contact_ke=1.0e4,  # N/m，刚软接触法向刚度。
                    soft_contact_kd=1.0e-2,  # N*s/m，接触阻尼。
                    soft_contact_mu=5.0,  # 高摩擦设置，增强物理抓取鲁棒性。
                ),
                num_substeps=10,  # 每个主步再细分 10 个物理子步。
                use_cuda_graph=not args_cli.disable_cuda_graph,
            ),
        )
    )


def rigid_surface_material() -> NewtonMaterialPropertiesCfg:
    """桌面和方块共用的刚体材料。"""
    return NewtonMaterialPropertiesCfg(
        static_friction=RIGID_STATIC_FRICTION,
        dynamic_friction=RIGID_DYNAMIC_FRICTION,
        restitution=RIGID_RESTITUTION,
    )


# ============================================================
# 11. 创建场景
# ============================================================


def spawn_scene() -> tuple[Articulation, RigidObject, DeformableObject]:
    """生成地面、灯光、桌子、YAM、动态方块和 VBD 软球。"""
    if not YAM_USD.is_file():
        raise FileNotFoundError(f"Converted YAM asset is missing: {YAM_USD}")

    sim_utils.create_prim("/World/env_0", "Xform")
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    # 静态桌子：只有碰撞，没有动态刚体 schema。
    table_cfg = sim_utils.CuboidCfg(
        size=TABLE_SIZE,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=rigid_surface_material(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.22, 0.12)),
    )
    table_cfg.func("/World/env_0/Table", table_cfg, translation=TABLE_CENTER)
    # 显式设置桌面接触的恢复时间尺度，避免默认硬接触对微小几何误差过度响应。
    table_prim = sim_utils.get_current_stage().GetPrimAtPath("/World/env_0/Table/geometry/mesh")
    table_prim.CreateAttribute("newton:contact_ke", Sdf.ValueTypeNames.Float).Set(SUPPORT_CONTACT_STIFFNESS)
    table_prim.CreateAttribute("newton:contact_kd", Sdf.ValueTypeNames.Float).Set(SUPPORT_CONTACT_DAMPING)

    robot_cfg = ArticulationCfg(
        prim_path="/World/env_0/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(YAM_USD),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_force=80.0, max_joint_velocity=3.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # 机器人 base 世界位置；Z 跟随桌面顶部。
            pos=(*ROBOT_BASE_XY, TABLE_TOP_Z),
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
                velocity_limit_sim=0.05,
                stiffness=1000.0,
                damping=50.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)

    # 解除实例化后，运行时把左右指尖三角网格碰撞改为 convex decomposition。
    sim_utils.make_uninstanceable("/World/env_0/Robot")
    stage = sim_utils.get_current_stage()
    for tip_path in (LEFT_TIP_PATH, RIGHT_TIP_PATH):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(tip_path)):
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set("convexDecomposition")

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

    block_prim = stage.GetPrimAtPath("/World/env_0/TargetBlock/geometry/mesh")
    block_prim.CreateAttribute("newton:contact_ke", Sdf.ValueTypeNames.Float).Set(SUPPORT_CONTACT_STIFFNESS)
    block_prim.CreateAttribute("newton:contact_kd", Sdf.ValueTypeNames.Float).Set(SUPPORT_CONTACT_DAMPING)

    # E,ν 通过 Lamé 参数 μ、λ 传给 VBD 材料。
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


# ============================================================
# 12. 随机位置采样
# ============================================================


def sample_object_positions(rng: random.Random) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """在桌心机器人周围的环形工作区采样，并过滤过近组合。"""
    if not 0.0 < OBJECT_RADIUS_RANGE[0] <= OBJECT_RADIUS_RANGE[1]:
        raise ValueError("OBJECT_RADIUS_RANGE must contain positive, ordered radii")
    if not -2.40 <= OBJECT_BEARING_RANGE[0] < OBJECT_BEARING_RANGE[1] <= 3.00:
        raise ValueError("OBJECT_BEARING_RANGE exceeds the checked radial-grasp sector [-2.40, 3.00]")

    def sample_on_annulus(z: float) -> tuple[tuple[float, float, float], float]:
        # 极坐标采样比在正方形采样后再过滤更直接，也明确保留基座避碰内圈。
        radius = rng.uniform(*OBJECT_RADIUS_RANGE)
        bearing = rng.uniform(*OBJECT_BEARING_RANGE)
        position = (
            ROBOT_BASE_XY[0] + radius * math.cos(bearing),
            ROBOT_BASE_XY[1] + radius * math.sin(bearing),
            z,
        )
        return position, bearing

    for _ in range(SAMPLING_MAX_ATTEMPTS):
        block_pos, block_bearing = sample_on_annulus(TABLE_TOP_Z + BLOCK_SIZE[2] / 2.0 + BLOCK_RESET_CLEARANCE)
        ball_pos, ball_bearing = sample_on_annulus(TABLE_TOP_Z + BALL_RADIUS)
        planar_distance = ((block_pos[0] - ball_pos[0]) ** 2 + (block_pos[1] - ball_pos[1]) ** 2) ** 0.5
        # 为旋转方块留半对角线余量，为软球留粒子接触包络余量。
        block_margin = math.hypot(*BLOCK_SIZE[:2]) / 2 + 0.01
        ball_margin = BALL_RADIUS + BALL_PARTICLE_RADIUS + 0.01
        inside_table = all(
            abs(pos[axis] - TABLE_CENTER[axis]) + margin <= TABLE_SIZE[axis] / 2
            for pos, margin in ((block_pos, block_margin), (ball_pos, ball_margin))
            for axis in (0, 1)
        )
        # 当前 joint1 的可用方位区间不是闭环。若直接角差超过 pi，机械臂不能简单地
        # 选择另一条短弧跨过限位缺口，只能走大于 180° 的长弧，因此拒绝该组合。
        transfer_bearing_ok = abs(block_bearing - ball_bearing) <= MAX_TRANSFER_BEARING_DELTA
        if planar_distance >= MIN_OBJECT_PLANAR_DISTANCE and inside_table and transfer_bearing_ok:
            return block_pos, ball_pos
    raise RuntimeError("Could not sample separated object positions; check radius, bearing range, and minimum distance")


# ============================================================
# 13. Episode reset
# ============================================================


def reset_episode(robot, block, ball, block_pos, ball_pos, ball_template=None) -> tuple[float, float, float]:
    """复位机器人、刚体方块与软球节点状态。"""
    joint_pos = robot.data.default_joint_pos.torch.clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel)
    robot.set_joint_position_target_index(target=joint_pos)

    block_pose = block.data.default_root_pose.torch.clone()
    block_pose[0, :3] = torch.tensor(block_pos, device=block_pose.device)
    block_pose[0, 3:] = torch.tensor((0.0, 0.0, 0.0, 1.0), device=block_pose.device)
    block.write_root_pose_to_sim_index(root_pose=block_pose)
    block.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=block_pose.device))

    # 软球是节点系统：只把 XY 对齐 ball_pos；Z 按最低节点+粒子半径对齐桌面。
    nodal_state = ball.data.default_nodal_state_w.torch.clone() if ball_template is None else ball_template.clone()
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


# ============================================================
# 14. 支撑/稳定性指标
# ============================================================


def block_support_metrics(block):
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


def check_block_support(block, *, stage: str) -> None:
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
    """软球所有节点世界位置均值，shape=(N,3)。"""
    return ball.data.nodal_pos_w.torch.mean(dim=1)


def ball_settle_metrics(ball):
    nodal_state = ball.data.nodal_state_w.torch[0]
    positions, velocities = nodal_state[:, :3], nodal_state[:, 3:]
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


def check_ball_support(ball, *, stage: str, reset_center) -> None:
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


# ============================================================
# 15. 单步物理推进与场景沉降
# ============================================================


def step_scene(sim, robot, block, ball, joint_target) -> None:
    """发送 8 关节 position target，推进一个 Newton 主步，并刷新所有状态。"""
    global SCENE_STEP_COUNT
    robot.set_joint_position_target_index(target=joint_target)
    robot.write_data_to_sim()
    ball.write_data_to_sim()
    sim.step(render=not args_cli.headless)
    SCENE_STEP_COUNT += 1
    dt = sim.get_physics_dt()
    robot.update(dt)
    block.update(dt)
    ball.update(dt)
    # 每步检查 NaN/Inf，数值爆炸时尽早停止。
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
    sim,
    robot,
    block,
    ball,
    joint_target,
    *,
    episode: int,
    minimum_steps: int,
    ball_reset_center,
    on_step=None,
    timeout_limit=SCENE_SETTLE_TIMEOUT_STEPS,
    support_height=None,
    required_steps=SCENE_SETTLE_REQUIRED_STEPS,  # 30
    ball_root_threshold=BALL_SETTLE_ROOT_SPEED,
    ball_nodal_threshold=BALL_SETTLE_MAX_NODAL_SPEED,
    ball_window_drift=BALL_SETTLE_WINDOW_DRIFT,
) -> int:
    """连续满足几何/速度/漂移阈值 required_steps 帧后才认为场景沉降完成。"""
    stable_steps, stable_centers = 0, []
    timeout_steps = max(timeout_limit, minimum_steps + required_steps)
    for step_index in range(timeout_steps):
        step_scene(sim, robot, block, ball, joint_target)
        if on_step is not None:
            on_step()
        step_count = step_index + 1
        bc, blv, bav, bot = block_support_metrics(block)
        ball_now, bclear, ball_velocity, nodal_speed, bon = ball_settle_metrics(ball)
        if support_height is not None:
            bclear -= support_height() - TABLE_TOP_Z
        if not bot or bc < -0.01:
            check_block_support(block, stage=f"episode_{episode}_step_{step_count}")
            raise RuntimeError(f"BLOCK_INSTABILITY: target block became unsupported in episode {episode}")
        if not bon or bclear < -0.01:
            check_ball_support(ball, stage=f"episode_{episode}_step_{step_count}", reset_center=ball_reset_center)
            raise RuntimeError(
                f"BALL_INSTABILITY: deformable ball became unsupported in episode {episode}, center={ball_now}"
            )

        ball_stable = (
            bon
            and abs(bclear) <= BALL_SETTLE_CONTACT_CLEARANCE
            and sum(v * v for v in ball_velocity) ** 0.5 <= ball_root_threshold
            and nodal_speed <= ball_nodal_threshold
        )
        if block_state_is_settled(bc, blv, bav, bot) and ball_stable:
            stable_centers.append(ball_now[:2])
            if len(stable_centers) > required_steps:
                stable_centers.pop(0)
            drift = (
                sum((max(p[i] for p in stable_centers) - min(p[i] for p in stable_centers)) ** 2 for i in (0, 1)) ** 0.5
            )
            if drift > ball_window_drift:
                stable_centers, stable_steps = [ball_now[:2]], 0
            stable_steps += 1
        else:
            stable_steps = 0
            stable_centers.clear()

        if step_count in SCENE_DIAGNOSTIC_STEPS or step_count == minimum_steps:
            stage = f"episode_{episode}_settling_step_{step_count}"
            check_block_support(block, stage=stage)
            check_ball_support(ball, stage=stage, reset_center=ball_reset_center)
        if step_count >= minimum_steps and stable_steps >= required_steps:
            stage = f"episode_{episode}_settled_step_{step_count}"
            check_block_support(block, stage=stage)
            check_ball_support(ball, stage=stage, reset_center=ball_reset_center)
            print(
                f"SCENE_SETTLED episode={episode} steps={step_count} stable_steps={stable_steps} "
                f"block_linear_threshold_mps={BLOCK_SETTLE_LINEAR_SPEED:.6f} ball_root_threshold_mps={ball_root_threshold:.6f} "
                f"ball_max_nodal_threshold_mps={ball_nodal_threshold:.6f}",
                flush=True,
            )
            return step_count
    check_block_support(block, stage=f"episode_{episode}_settle_timeout")
    check_ball_support(ball, stage=f"episode_{episode}_settle_timeout", reset_center=ball_reset_center)
    raise RuntimeError(
        f"SCENE_SETTLE_TIMEOUT: episode={episode} stable_steps={stable_steps}/{required_steps} "
        f"within {timeout_steps} physics steps; "
        f"block_ok={block_state_is_settled(bc, blv, bav, bot)} ball_ok={ball_stable}; "
        f"block_speed={math.sqrt(sum(v*v for v in blv)):.6f}/{BLOCK_SETTLE_LINEAR_SPEED:.6f} m/s; "
        f"ball_speed={math.sqrt(sum(v*v for v in ball_velocity)):.6f}/{ball_root_threshold:.6f} m/s; "
        f"ball_max_nodal_speed={nodal_speed:.6f}/{ball_nodal_threshold:.6f} m/s"
    )


# ============================================================
# 16. 抓取几何和 grasp-point kinematics
# ============================================================


def finger_surface_points(robot: Articulation) -> torch.Tensor:
    """左右夹指内侧代表点的世界坐标，shape=(2,3)。"""
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
    """把 FK/Jacobian 从 gripper origin 平移到真正的 grasp point。"""
    # 机械臂前 6 个关节是末端执行器，7、8 是夹爪。
    gripper_pos_w, gripper_rot_w, geometric_jacobian_w = forward_kinematics_and_jacobian(root_pose_w, arm_pos)
    # 把局部偏移转换到世界坐标
    local_offset = torch.tensor(GRASP_POINT_LOCAL, device=arm_pos.device, dtype=arm_pos.dtype)
    offset_w = (gripper_rot_w @ local_offset.expand(arm_pos.shape[0], 3).unsqueeze(-1)).squeeze(-1)
    # 世界坐标下，真正用于抓取的点位置；gripper_pos_w 是机械臂末端 gripper origin。
    grasp_pos_w = gripper_pos_w + offset_w

    angular_columns = geometric_jacobian_w[:, 3:, :].transpose(1, 2)
    # 刚体点速度 v_point=v_origin+ω×r，因此 Jv_point=Jv_origin+Jω×r。
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


# ============================================================
# 17. 运动控制
# ============================================================


def move_grasp_point(
    sim,
    robot,
    block,
    ball,
    target_pos_w,
    gripper_target,
    maintain_orientation=True,
    target_rotation_w=None,
    speed=None,
    smooth=False,
    on_step=None,
):
    """位置与 SLERP 姿态同步插值，再用 FK/Jacobian/DLS 跟踪；超时明确失败。
    slerp 插值规划路径
    """
    ee_body_idx = robot.body_names.index("gripper")
    limits = robot.data.joint_pos_limits.torch[0, :6]
    initial_fk_error = 0.0
    target_rot_w = (
        torch.tensor(GRASP_ROTATION_W, device=robot.device, dtype=robot.data.joint_pos.torch.dtype).unsqueeze(0)
        if target_rotation_w is None
        else target_rotation_w
    )
    _, start_rot, start_pos, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
    )
    # 本任务只有一个机械臂实例；quat_slerp 接受 (4,) 而不是 (N,4)。
    start_quat = quat_from_matrix(start_rot)[0]
    target_quat = quat_from_matrix(target_rot_w)[0]
    # 计算时长
    rotation_distance = 2.0 * torch.acos(torch.dot(start_quat, target_quat).abs().clamp(0.0, 1.0)).item()
    linear_duration = torch.linalg.vector_norm(target_pos_w - start_pos).item() / (
        ARM_CARTESIAN_SPEED if speed is None else speed
    )
    angular_duration = rotation_distance / ARM_ORIENTATION_SPEED if maintain_orientation else 0.0
    # 平移和旋转同时插值，取较长时间；纯旋转也必须获得足够物理步数。
    time_scale = max(1.0, args_cli.motion_steps / 240)
    travel_steps = max(
        1,
        math.ceil(
            max(linear_duration, angular_duration) * (1.5 if smooth else 1.0) * time_scale / sim.get_physics_dt()
        ),
    )
    stable_steps = 0
    for step in range(travel_steps + MOTION_SETTLE_STEPS):
        phase = min(1.0, (step + 1) / travel_steps)
        alpha = phase * phase * (3.0 - 2.0 * phase) if smooth else phase
        # smooth step 0.0->1.0: 3α²-2α³, smooth step derivative 6α(1-α)
        waypoint = start_pos + alpha * (target_pos_w - start_pos)
        root_pose_w = robot.data.root_link_pose_w.torch
        arm_pos = robot.data.joint_pos.torch[:, :6]
        gripper_pos_w, gripper_rot_w, grasp_pos_w, jac = grasp_point_kinematics(root_pose_w, arm_pos)
        if step == 0:
            simulated_pos_w = robot.data.body_link_pose_w.torch[:, ee_body_idx, :3]
            initial_fk_error = torch.linalg.vector_norm(gripper_pos_w - simulated_pos_w, dim=-1).item()
        if maintain_orientation:
            # SLERP 沿单位四元数球面插值，避免原代码在段首突然旋转约 40°。
            # 当前 IsaacLab 的 quat_slerp 会原地调整 q2 的符号，所以传入 clone。
            waypoint_rotation = matrix_from_quat(quat_slerp(start_quat, target_quat.clone(), alpha)).unsqueeze(0)
            joint_delta = damped_least_squares_pose_step(
                grasp_pos_w, waypoint, gripper_rot_w, waypoint_rotation, jac, damping=0.05
            )
        else:
            joint_delta = damped_least_squares_position_step(grasp_pos_w, waypoint, jac[:, :3, :], damping=0.05)
        # 单步 DLS 增量限制 ±0.02rad，再做关节限位。
        arm_target = torch.clamp(
            arm_pos + torch.clamp(joint_delta, min=-0.02, max=0.02), min=limits[:, 0], max=limits[:, 1]
        )
        full_target = robot.data.joint_pos.torch.clone()
        full_target[:, :6] = arm_target
        full_target[:, 6:] = gripper_target
        step_scene(sim, robot, block, ball, full_target)
        if on_step is not None:
            on_step()
        if step >= travel_steps - 1:
            _, final_rot, current_pos, _ = grasp_point_kinematics(
                robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
            )
            pos_ok = torch.linalg.vector_norm(current_pos - target_pos_w).item() <= 0.002
            rot_ok = not maintain_orientation or torch.linalg.matrix_norm(final_rot - target_rot_w).item() <= 0.02
            stopped = robot.data.joint_vel.torch[:, :6].abs().max().item() <= 0.05
            stable_steps = stable_steps + 1 if pos_ok and rot_ok and stopped else 0
            if stable_steps >= 3:
                break
    else:
        position_error = torch.linalg.vector_norm(current_pos - target_pos_w).item()
        rotation_error = torch.linalg.matrix_norm(final_rot - target_rot_w).item()
        raise RuntimeError(
            f"MOTION_TIMEOUT: target={target_pos_w[0].tolist()} position_error_m={position_error:.6f} "
            f"rotation_matrix_error={rotation_error:.6f} joint_speed={robot.data.joint_vel.torch[:, :6].abs().max().item():.6f}"
        )
    _, _, final_grasp_pos_w, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
    )
    return torch.linalg.vector_norm(final_grasp_pos_w - target_pos_w, dim=-1).item(), initial_fk_error


def radial_grasp_rotation(robot, target_pos_w):
    """让夹爪随目标方位绕世界 Z 轴旋转，同时保持从上向下抓取。"""
    root_xy = robot.data.root_link_pose_w.torch[:, :2]
    bearing = torch.atan2(target_pos_w[:, 1] - root_xy[:, 1], target_pos_w[:, 0] - root_xy[:, 0])
    cosine, sine = torch.cos(bearing), torch.sin(bearing)
    zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
    yaw_rotation = torch.stack((cosine, -sine, zero, sine, cosine, zero, zero, zero, one), dim=-1).reshape(-1, 3, 3)
    reference = torch.tensor(GRASP_ROTATION_W, device=robot.device, dtype=target_pos_w.dtype).unsqueeze(0)
    return yaw_rotation @ reference


def joint1_compatible_bearing(robot, target_pos_w):
    """选择落在首关节安全区内的方位等价角；完整姿态可达性仍由 IK 判断。
    选择合适的初始角度

    径向竖直抓取的 joint1 约等于目标方位（还有小的机构偏置），不能把
    APPROACH_IK_SEED[0] 当成固定几何偏置加到限位判断里。
    """
    root_xy = robot.data.root_link_pose_w.torch[:, :2]
    bearing = torch.atan2(target_pos_w[:, 1] - root_xy[:, 1], target_pos_w[:, 0] - root_xy[:, 0])
    limits = robot.data.joint_pos_limits.torch[0, 0]
    candidates = bearing[:, None] + torch.tensor((-2 * math.pi, 0.0, 2 * math.pi), device=robot.device)
    valid = (candidates >= limits[0] + 0.10) & (candidates <= limits[1] - 0.10)
    if not valid.any(dim=1).all():
        raise RuntimeError(f"UNREACHABLE_BEARING: bearing={bearing.tolist()} joint1_limits={limits.tolist()}")
    selected = valid.to(torch.int64).argmax(dim=1, keepdim=True)
    return candidates.gather(1, selected).squeeze(1)


def move_ball_on_continuous_polar_path(sim, robot, block, ball, destination_center_w, gripper_target, on_step=None):
    """沿一条连续极坐标曲线搬运球，全程只使用一次 smoothstep。

    旧实现把圆弧拆成多个路点，并让每一段各自执行 smoothstep。由于 smoothstep
    在每段首尾的速度都为零，机械臂会反复慢—快—慢。这里改用全局进度 ``u``：
    方位角和半径共享同一个 ``alpha=3u²-2u³``，所以只在整条路径的起点和终点减速。
    """
    limits = robot.data.joint_pos_limits.torch[0, :6]
    root_xy = robot.data.root_link_pose_w.torch[:, :2]
    start_ball_center = ball_center(ball).clone()
    start_bearing = joint1_compatible_bearing(robot, start_ball_center)
    destination_bearing = joint1_compatible_bearing(robot, destination_center_w)
    bearing_delta = destination_bearing - start_bearing
    bearing_travel = abs(bearing_delta.item())
    if bearing_travel > MAX_TRANSFER_BEARING_DELTA + 1.0e-6:
        raise RuntimeError(
            f"TRANSFER_BEARING_TOO_LARGE: delta_rad={bearing_travel:.6f} " f"limit_rad={MAX_TRANSFER_BEARING_DELTA:.6f}"
        )

    start_radius = torch.linalg.vector_norm(start_ball_center[:, :2] - root_xy, dim=-1)
    destination_radius = torch.linalg.vector_norm(destination_center_w[:, :2] - root_xy, dim=-1)
    radius_delta = destination_radius - start_radius
    _, _, start_grasp_pos, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
    )
    transfer_height = start_grasp_pos[:, 2:].clone()

    # Archimedean-like spiral 的长度估计。半径变化较小时，平均半径乘角差近似弧长；
    # 与径向位移做勾股合成，用于把用户给出的 m/s 换算成物理步数。
    mean_radius = 0.5 * (start_radius + destination_radius)
    path_length = math.hypot(radius_delta.item(), mean_radius.item() * bearing_travel)
    linear_duration = path_length / TRANSFER_SPEED
    angular_duration = bearing_travel / ARM_ORIENTATION_SPEED
    time_scale = max(1.0, args_cli.motion_steps / 240)
    # cubic smoothstep 的峰值导数是平均值的 1.5 倍，因此时长乘 1.5 后，峰值速度
    # 才大致等于 TRANSFER_SPEED/ARM_ORIENTATION_SPEED，而不是超过设定值。
    travel_steps = max(
        1,
        math.ceil(max(linear_duration, angular_duration) * 1.0 * time_scale / sim.get_physics_dt()),
    )
    print(
        f"TRANSFER_PATH bearing_delta_rad={bearing_delta.item():.3f} "
        f"estimated_length_m={path_length:.3f} travel_steps={travel_steps}",
        flush=True,
    )

    stable_steps = 0
    current_rotation_w = None
    ball_alignment_error = float("inf")
    for step in range(travel_steps + MOTION_SETTLE_STEPS):
        # travel_steps 之后 alpha 固定为 1，额外物理步只用于闭环消除末端弹性偏移。
        phase = min(1.0, (step + 1) / travel_steps)
        alpha = phase * phase * (3.0 - 2.0 * phase)
        bearing = start_bearing + alpha * bearing_delta
        radius = start_radius + alpha * radius_delta
        desired_ball_xy = root_xy + torch.stack((radius * torch.cos(bearing), radius * torch.sin(bearing)), dim=-1)

        arm_pos = robot.data.joint_pos.torch[:, :6]
        _, current_rotation_w, current_grasp_pos, jacobian = grasp_point_kinematics(
            robot.data.root_link_pose_w.torch, arm_pos
        )
        # 以实际软球球心反馈修正夹爪目标。XY 跟随极坐标轨迹，Z 始终保持抬起高度。
        target_grasp_pos = torch.cat(
            (current_grasp_pos[:, :2] + desired_ball_xy - ball_center(ball)[:, :2], transfer_height), dim=-1
        )
        orientation_reference = torch.cat((desired_ball_xy, destination_center_w[:, 2:]), dim=-1)
        target_rotation_w = radial_grasp_rotation(robot, orientation_reference)
        joint_delta = damped_least_squares_pose_step(
            current_grasp_pos,
            target_grasp_pos,
            current_rotation_w,
            target_rotation_w,
            jacobian,
            damping=0.05,
        )
        # DLS 给出“消除当前位姿误差”的关节增量，不是已经考虑执行器滞后的速度命令。
        # 让驱动目标适度领先实际关节，才能连续跟上移动参考；真实运动仍由 PD 驱动和
        # 资产中的速度/力矩限制约束。
        arm_target = torch.clamp(
            arm_pos
            + torch.clamp(
                TRANSFER_DLS_GAIN * joint_delta,
                min=-TRANSFER_MAX_JOINT_COMMAND_STEP,
                max=TRANSFER_MAX_JOINT_COMMAND_STEP,
            ),
            min=limits[:, 0],
            max=limits[:, 1],
        )
        full_target = robot.data.joint_pos.torch.clone()
        full_target[:, :6] = arm_target
        full_target[:, 6:] = gripper_target
        step_scene(sim, robot, block, ball, full_target)
        if on_step is not None:
            on_step()

        if step >= travel_steps - 1:
            _, current_rotation_w, _, _ = grasp_point_kinematics(
                robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
            )
            ball_alignment_error = torch.linalg.vector_norm(
                ball_center(ball)[:, :2] - destination_center_w[:, :2]
            ).item()
            final_rotation_w = radial_grasp_rotation(robot, destination_center_w)
            rotation_error = torch.linalg.matrix_norm(current_rotation_w - final_rotation_w).item()
            stopped = robot.data.joint_vel.torch[:, :6].abs().max().item() <= 0.05
            stable_steps = (
                stable_steps + 1
                if ball_alignment_error <= TRANSFER_BALL_ALIGNMENT_TOLERANCE and rotation_error <= 0.02 and stopped
                else 0
            )
            if stable_steps >= 3:
                return ball_alignment_error

    raise RuntimeError(
        f"TRANSFER_TIMEOUT: ball_alignment_error_m={ball_alignment_error:.6f} "
        f"rotation_matrix_error={rotation_error:.6f} "
        f"joint_speed={robot.data.joint_vel.torch[:, :6].abs().max().item():.6f}"
    )


def solve_approach_joint_target(robot, target_pos_w, target_rotation_w, iterations=300):
    """离线迭代 DLS 求预抓取关节解，不推进 physics。"""
    arm_target = torch.tensor((APPROACH_IK_SEED,), device=robot.device, dtype=robot.data.joint_pos.torch.dtype)
    limits = robot.data.joint_pos_limits.torch[0, :6]
    # 固定初值只适用于一个方向；根据目标方位旋转 joint1 初值，保留其余关节的肘形。
    arm_target[:, 0] = joint1_compatible_bearing(robot, target_pos_w)
    arm_target = torch.clamp(arm_target, min=limits[:, 0], max=limits[:, 1])
    root_pose_w = robot.data.root_link_pose_w.torch
    for _ in range(iterations):
        #  当前候选关节的gripper link状态，只保留姿态
        _, current_rotation_w, _ = forward_kinematics_and_jacobian(root_pose_w, arm_target)
        # 加上夹爪修正，获得真正抓取点的位置和jocabin
        _, _, grasp_pos_w, grasp_jacobian_w = grasp_point_kinematics(root_pose_w, arm_target)
        joint_delta = damped_least_squares_pose_step(
            # 位置需要加上夹爪的偏移，但是姿态和gripper link一致，所以不需要加上夹爪的偏移
            grasp_pos_w,
            target_pos_w,
            current_rotation_w,
            target_rotation_w,
            grasp_jacobian_w,
            damping=0.03,
        )
        arm_target = torch.clamp(
            arm_target + torch.clamp(joint_delta, min=-0.05, max=0.05), min=limits[:, 0], max=limits[:, 1]
        )
    _, solved_rotation_w, solved_pos_w, _ = grasp_point_kinematics(root_pose_w, arm_target)
    return (
        arm_target,
        torch.linalg.vector_norm(solved_pos_w - target_pos_w, dim=-1).item(),
        # Frobenius norm of the difference between the solved rotation and the target rotation
        torch.linalg.matrix_norm(solved_rotation_w - target_rotation_w, dim=(-2, -1)).item(),
    )


def execute_arm_trajectory(sim, robot, block, ball, target_arm_pos, gripper_target):
    """首次 approach：在 joint space 用 smoothstep 从当前 q 插值到离线 IK q。
    运动到小球上方，预抓取
    """
    start_arm_pos = robot.data.joint_pos.torch[:, :6].clone()
    steps = max(
        30,
        int(
            1.5  # 放大系数，比理论更慢点
            * (target_arm_pos - start_arm_pos).abs().max().item()
            / sim.get_physics_dt()
            / ARM_JOINT_SPEED
            * args_cli.motion_steps
            / 240
        )
        + 1,
    )
    for step in range(steps):
        phase = (step + 1) / steps
        alpha = phase * phase * (3.0 - 2.0 * phase)
        full_target = robot.data.joint_pos.torch.clone()
        full_target[:, :6] = start_arm_pos + alpha * (target_arm_pos - start_arm_pos)
        full_target[:, 6:] = gripper_target
        step_scene(sim, robot, block, ball, full_target)


def hold_grasp(sim, robot, block, ball, gripper_start, gripper_end, *, release=False, on_step=None):
    """机械臂维持最后 drive target，只线性移动 joint7/8。
    它主要用于：闭合夹爪抓球  张开夹爪释放球  单独测试夹爪开合
    """
    target = robot.data.joint_pos_target.torch.clone()
    # 选择速度
    speed = GRIPPER_COMMAND_SPEED if args_cli.gripper_check else GRIPPER_CONTACT_SPEED
    if release:
        speed = PLACE_RELEASE_SPEED

    time_scale = max(1.0, args_cli.motion_steps / 240) if release else args_cli.motion_steps / 240
    steps = max(1, int(abs(gripper_end - gripper_start) / (speed * sim.get_physics_dt()) * time_scale) + 1)
    for step in range(steps):
        target[:, 6:] = gripper_start + (gripper_end - gripper_start) * (step + 1) / steps
        step_scene(sim, robot, block, ball, target)
        if on_step is not None:
            on_step()


# ============================================================
# 18. 抓取检查
# ============================================================


# 检测开合 正式不用
def check_gripper_motion(sim, robot, block, ball):
    print("STATE=GRIPPER_CHECK", flush=True)
    gaps = []
    for start, end in ((GRIPPER_OPEN_POSITION, 0.0), (0.0, GRIPPER_OPEN_POSITION)):
        hold_grasp(sim, robot, block, ball, start, end)
        target = robot.data.joint_pos_target.torch.clone()
        target[:, 6:] = end
        for _ in range(30):
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
    # 检查球的形变是否过大，避免夹爪把球压扁。
    # 软体小球包围盒： 用一个与世界坐标轴平行的长方体，把小球的所有节点包住，这个长方体在 X、Y、Z 三个方向的长度。
    extent = ball.data.nodal_pos_w.torch[0].amax(dim=0) - ball.data.nodal_pos_w.torch[0].amin(dim=0)
    if extent.min().item() < BALL_RADIUS * 0.6 or extent.max().item() > BALL_RADIUS * 3.0:
        raise RuntimeError("FAILURE: excessive deformation")


def quick_grasp_check(sim, robot, block, ball, *, center_before_lift=None, required_lift=0.0):
    """有限窗口检查夹持包络、相对速度、离桌和实际升高。
    保持夹爪闭合
    -> 每帧推进物理
    -> 检查球是否还在夹爪中
    -> 检查夹持包络
    -> 检查夹爪是否停止
    -> 检查球相对夹爪是否稳定
    -> 可选检查是否已经抬离桌面
    """
    target = robot.data.joint_pos_target.torch.clone()
    previous_relative = ball_center(ball) - finger_surface_points(robot).mean(dim=0)
    stable = 0
    for index in range(QUICK_GRASP_TIMEOUT_STEPS):
        step_scene(sim, robot, block, ball, target)
        check_ball_in_gripper(robot, ball)
        # gap：左右夹指内侧点之间的距离；
        # width：小球沿夹持轴方向的投影宽度
        gap, width = grasp_geometry_metrics(robot, ball)

        # 计算小球相对于夹爪的速度，判断是否稳定
        relative = ball_center(ball) - finger_surface_points(robot).mean(dim=0)
        relative_speed = torch.linalg.vector_norm(relative - previous_relative).item() / sim.get_physics_dt()
        previous_relative = relative.clone()
        finger_stopped = robot.data.joint_vel.torch[:, 6:].abs().max().item() < 0.002
        enveloped = gap < GRIPPER_OPEN_GAP - 0.002 and gap <= width + 2 * BALL_PARTICLE_RADIUS + 0.001
        lift_ok = True
        # 抬起小球检测
        if center_before_lift is not None:
            lowest = ball.data.nodal_pos_w.torch[..., 2].min().item() - BALL_PARTICLE_RADIUS
            rise = (ball_center(ball)[:, 2] - center_before_lift[:, 2]).item()
            lift_ok = lowest >= TABLE_TOP_Z + 0.002 and rise >= required_lift * MIN_LIFT_RATIO
        stable = stable + 1 if enveloped and finger_stopped and relative_speed <= 0.003 and lift_ok else 0
        # 判断12个物理帧，0.1s
        if stable >= QUICK_GRASP_STABLE_STEPS:
            print(
                f"QUICK_GRASP_OK lifted={center_before_lift is not None} steps={index + 1} sim_s={(index + 1) * sim.get_physics_dt():.3f}",
                flush=True,
            )
            return
    raise RuntimeError("FAILURE QUICK_GRASP: grip/relative motion/lift did not stabilize before timeout")


# ============================================================
# 19. Pick + lift
# ============================================================


def pick_and_lift_ball(sim, robot, block, ball, initial_ball_pos, lift_height, episode):
    """pregrasp -> settle -> descend -> close -> verify -> lift。"""
    print("STATE=PRE_GRASP", flush=True)
    device, dtype = robot.device, robot.data.joint_pos.torch.dtype
    initial_ball_target = torch.tensor((initial_ball_pos,), device=device, dtype=dtype)
    # 目标位置，小球上方1cm
    pregrasp_offset = torch.tensor(((0.0, 0.0, 0.10),), device=device, dtype=dtype)
    pregrasp_target = initial_ball_target + pregrasp_offset
    # 机器人位于桌心后，夹爪朝向应随球相对基座的方位变化。
    approach_rotation_w = radial_grasp_rotation(robot, pregrasp_target)
    # 计算小球上方目标
    # 离线300计算目标位置和姿态
    pregrasp_joint_target, solved_position_error, solved_rotation_error = solve_approach_joint_target(
        robot, pregrasp_target, approach_rotation_w
    )
    if solved_position_error > 0.002 or solved_rotation_error > 0.02:
        raise RuntimeError(
            f"Pregrasp IK did not converge: position_error={solved_position_error:.6f}, rotation_error={solved_rotation_error:.6f}"
        )
    # 执行操作
    execute_arm_trajectory(sim, robot, block, ball, pregrasp_joint_target, GRIPPER_OPEN_POSITION)

    approach_hold = robot.data.joint_pos_target.torch.clone()
    approach_hold[:, :6] = pregrasp_joint_target
    approach_hold[:, 6:] = GRIPPER_OPEN_POSITION
    wait_until_scene_settled(  # 先等待60帧，只要12帧连续稳定就settle了
        sim,
        robot,
        block,
        ball,
        approach_hold,
        episode=episode,
        minimum_steps=12,
        ball_reset_center=initial_ball_pos,
        required_steps=GRASP_READY_REQUIRED_STEPS,
        ball_root_threshold=GRASP_READY_ROOT_SPEED,
        ball_nodal_threshold=GRASP_READY_MAX_NODAL_SPEED,
        ball_window_drift=GRASP_READY_WINDOW_DRIFT,
    )

    # 球可能自然沉降，所以用最新球心重新求一次 pregrasp。
    # 重新计算预抓取目标
    tracked_pregrasp_target = ball_center(ball).clone() + pregrasp_offset
    # 离线300计算目标位置和姿态
    tracked_joint_target, solved_position_error, solved_rotation_error = solve_approach_joint_target(
        robot, tracked_pregrasp_target, approach_rotation_w
    )
    # 误差判断
    if solved_position_error > 0.002 or solved_rotation_error > 0.02:
        raise RuntimeError(
            f"Tracked pregrasp IK did not converge: position_error={solved_position_error:.6f}, rotation_error={solved_rotation_error:.6f}"
        )

    # 读取当前真实关节角，通过FK计算抓取点位置
    _, _, current_pregrasp, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
    )
    # 计算当前抓取点与追踪目标的距离，如果超过阈值则重新执行轨迹
    if torch.linalg.vector_norm(current_pregrasp - tracked_pregrasp_target).item() > 0.003:
        execute_arm_trajectory(sim, robot, block, ball, tracked_joint_target, GRIPPER_OPEN_POSITION)

    # 检查实际抓取点与追踪目标的误差
    _, _, actual_pregrasp_pos_w, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
    )
    # tracked_pregrasp_target：预测抓取点的目标位置
    pregrasp_error = torch.linalg.vector_norm(actual_pregrasp_pos_w - tracked_pregrasp_target, dim=-1).item()
    gripper_index = robot.body_names.index("gripper")
    # 检查仿真器姿态是否一致
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
    required_gap = width + 2 * BALL_PARTICLE_RADIUS + 0.001
    if gap < required_gap:
        raise RuntimeError(f"FAILURE OPEN: gap={gap}, required_contact_envelope={required_gap}")

    print("STATE=DESCEND", flush=True)
    # 先到球上方 2.5cm，再低速进入夹持区域。
    above_ball = ball_center(ball).clone()
    above_ball[:, 2] += 0.025
    move_grasp_point(
        sim,
        robot,
        block,
        ball,
        above_ball,
        GRIPPER_OPEN_POSITION,
        speed=ARM_CARTESIAN_SPEED,
        smooth=True,
        target_rotation_w=approach_rotation_w,
    )
    # 移动到球心位置，夹爪张开，低速下降
    grasp_target = ball_center(ball).clone()
    grasp_error, _ = move_grasp_point(
        sim,
        robot,
        block,
        ball,
        grasp_target,
        GRIPPER_OPEN_POSITION,
        target_rotation_w=approach_rotation_w,
        speed=DESCEND_SPEED,
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

    # 夹爪闭合，保持夹持，检查夹持包络、相对速度、离桌和实际升高。
    hold_grasp(sim, robot, block, ball, GRIPPER_OPEN_POSITION, GRIPPER_GRASP_POSITION)

    print(
        f"PICK_LIFT_STAGE=closed ball_center={ball_center(ball)[0].cpu().tolist()} finger_joints={robot.data.joint_pos.torch[0, 6:].cpu().tolist()}",
        flush=True,
    )
    print(f"FINGER_SURFACE_POINTS_CLOSED={finger_surface_points(robot).cpu().tolist()}", flush=True)
    # measured_gap： 夹爪实际开开距离 ball_width_after_grasp：小球投影在该方向上的宽度
    measured_gap, ball_width_after_grasp = grasp_geometry_metrics(robot, ball)
    print(
        f"PHYSICAL_GRASP target_gap_m={GRIPPER_TARGET_GAP:.6f} measured_gap_m={measured_gap:.6f} ball_width_on_grasp_axis_m={ball_width_after_grasp:.6f}",
        flush=True,
    )
    print("STATE=QUICK_GRASP_CHECK", flush=True)

    # 夹爪闭合后，快速检查夹持包络、相对速度、离桌和实际升高。
    quick_grasp_check(sim, robot, block, ball)
    if args_cli.stop_after_hold:
        print("CLOSE_HOLD_COMPLETED_NOT_LIFT_SUCCESS", flush=True)
        return {"hold_only": 1.0}

    # 抬起阶段
    print("STATE=LIFT", flush=True)
    center_before_lift = ball_center(ball).clone()
    _, _, grasp_before_lift, _ = grasp_point_kinematics(
        robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
    )
    lift_target = grasp_before_lift.clone()
    if args_cli.place_on_block:
        ball_bottom = ball.data.nodal_pos_w.torch[..., 2].min().item() - BALL_PARTICLE_RADIUS
        block_top = block.data.root_pos_w.torch[0, 2].item() + BLOCK_SIZE[2] / 2
        lift_height = max(lift_height, block_top + 0.065 - ball_bottom)
    lift_target[:, 2] += lift_height
    lift_error, _ = move_grasp_point(
        sim,
        robot,
        block,
        ball,
        lift_target,
        GRIPPER_GRASP_POSITION,
        target_rotation_w=approach_rotation_w,
        speed=LIFT_SPEED,
        smooth=True,
        on_step=lambda: check_ball_in_gripper(robot, ball),
    )
    print(
        f"PICK_LIFT_STAGE=lift error_m={lift_error:.6f} ball_center={ball_center(ball)[0].cpu().tolist()}", flush=True
    )
    quick_grasp_check(sim, robot, block, ball, center_before_lift=center_before_lift, required_lift=lift_height)
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


# ============================================================
# 20. 放回桌面 / 放到方块
# ============================================================


def put_ball_back(sim, robot, block, ball, original_center, episode, *, on_block=False):
    """水平搬运 -> 接近支撑面 -> 低速接地 -> 松爪 -> 撤离 -> 再沉降。"""
    origin = torch.tensor((original_center,), device=robot.device, dtype=robot.data.joint_pos.torch.dtype)
    initial_block_pos = block.data.root_pos_w.torch.clone()
    if on_block:
        origin[:, :2] = initial_block_pos[:, :2]
        origin[:, 2] = initial_block_pos[:, 2] + BLOCK_SIZE[2] / 2 + BALL_RADIUS
    destination_rotation_w = radial_grasp_rotation(robot, origin)

    def support_height():
        if not on_block:
            return TABLE_TOP_Z
        q = block.data.root_quat_w.torch[0]
        up_z = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2).item()
        displacement = torch.linalg.vector_norm(block.data.root_pos_w.torch - initial_block_pos).item()
        if up_z < 0.9998 or displacement > BLOCK_PLACEMENT_MAX_DISPLACEMENT:
            raise RuntimeError(f"FAILURE BLOCK_SUPPORT: tilt_or_motion up_z={up_z}, displacement={displacement}")
        return block.data.root_pos_w.torch[0, 2].item() + BLOCK_SIZE[2] / 2

    def grasp_position():
        return grasp_point_kinematics(robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6])[2].clone()

    def clearance():
        return ball.data.nodal_pos_w.torch[..., 2].min().item() - BALL_PARTICLE_RADIUS - support_height()

    if on_block and clearance() < 0.05:
        print("STATE=RAISE_FOR_TRANSFER", flush=True)
        target = grasp_position()
        current_rotation_w = grasp_point_kinematics(
            robot.data.root_link_pose_w.torch, robot.data.joint_pos.torch[:, :6]
        )[1].clone()
        target[:, 2] += max(0.0, 0.06 - clearance())
        error, _ = move_grasp_point(
            sim,
            robot,
            block,
            ball,
            target,
            GRIPPER_GRASP_POSITION,
            target_rotation_w=current_rotation_w,
            speed=LIFT_SPEED,
            smooth=True,
            on_step=lambda: check_ball_in_gripper(robot, ball),
        )
        if error > 0.005 or clearance() < 0.05:
            raise RuntimeError(f"FAILURE TRANSFER_HEIGHT: error={error}, clearance={clearance()}")

    print("STATE=TRANSFER_TO_BLOCK" if on_block else "STATE=RETURN_ALIGN", flush=True)
    # 两个物体位于机器人对侧时不能直线穿过底座。沿环形工作区连续绕行，整条
    # 轨迹只执行一次 smoothstep，不再在中间路点反复减速到零。
    alignment_error = move_ball_on_continuous_polar_path(
        sim,
        robot,
        block,
        ball,
        origin,
        GRIPPER_GRASP_POSITION,
        on_step=lambda: check_ball_in_gripper(robot, ball),
    )

    # 连续轨迹本身已经闭环到 2 mm；这里保留最多三次修正作为软球突发滑移的恢复路径。
    for correction in range(3):
        ball_alignment_error = torch.linalg.vector_norm(ball_center(ball)[:, :2] - origin[:, :2]).item()
        if ball_alignment_error <= TRANSFER_BALL_ALIGNMENT_TOLERANCE:
            break
        print(
            f"STATE=BALL_ALIGNMENT_CORRECTION pass={correction + 1} error_m={ball_alignment_error:.6f}",
            flush=True,
        )
        target = grasp_position()
        target[:, :2] += origin[:, :2] - ball_center(ball)[:, :2]
        move_grasp_point(
            sim,
            robot,
            block,
            ball,
            target,
            GRIPPER_GRASP_POSITION,
            target_rotation_w=destination_rotation_w,
            speed=TRANSFER_SPEED,
            smooth=True,
            on_step=lambda: check_ball_in_gripper(robot, ball),
        )
    else:
        ball_alignment_error = torch.linalg.vector_norm(ball_center(ball)[:, :2] - origin[:, :2]).item()
    if ball_alignment_error > TRANSFER_BALL_ALIGNMENT_TOLERANCE:
        raise RuntimeError(f"FAILURE TRANSFER_BALL_ALIGNMENT: error={ball_alignment_error}")

    print("STATE=LOWER_TO_BLOCK" if on_block else "STATE=LOWER_TO_TABLE", flush=True)
    target = grasp_position()
    target[:, 2] -= max(0.0, clearance() - 0.015)
    move_grasp_point(
        sim,
        robot,
        block,
        ball,
        target,
        GRIPPER_GRASP_POSITION,
        target_rotation_w=destination_rotation_w,
        speed=PLACE_APPROACH_SPEED,
        smooth=True,
        on_step=lambda: check_ball_in_gripper(robot, ball),
    )
    for _ in range(8):
        gap = clearance()
        if -BALL_SETTLE_CONTACT_CLEARANCE <= gap <= PLACE_CONTACT_CLEARANCE:
            break
        if gap < -BALL_SETTLE_CONTACT_CLEARANCE:
            raise RuntimeError(f"FAILURE PLACE: excessive table penetration {gap:.6f} m")
        target = grasp_position()
        target[:, 2] -= min(0.015, gap)
        move_grasp_point(
            sim,
            robot,
            block,
            ball,
            target,
            GRIPPER_GRASP_POSITION,
            target_rotation_w=destination_rotation_w,
            speed=PLACE_SPEED,
            smooth=True,
            on_step=lambda: check_ball_in_gripper(robot, ball),
        )
    if not -BALL_SETTLE_CONTACT_CLEARANCE <= clearance() <= PLACE_CONTACT_CLEARANCE:
        raise RuntimeError(f"FAILURE PLACE: could not reach release height, clearance={clearance():.6f}")

    held_target = robot.data.joint_pos_target.torch.clone()
    for _ in range(6):
        step_scene(sim, robot, block, ball, held_target)
        check_ball_in_gripper(robot, ball)
    if not -BALL_SETTLE_CONTACT_CLEARANCE <= clearance() <= PLACE_CONTACT_CLEARANCE:
        raise RuntimeError("FAILURE PLACE: release height changed during hold")
    release_center = ball_center(ball).clone()
    max_drift = 0.0

    def monitor_release():
        nonlocal max_drift
        center = ball_center(ball)
        error = torch.linalg.vector_norm(center[:, :2] - origin[:, :2]).item()
        max_drift = max(max_drift, torch.linalg.vector_norm(center[:, :2] - release_center[:, :2]).item())
        if error > PLACE_POSITION_TOLERANCE or max_drift > PLACE_RELEASE_DRIFT:
            raise RuntimeError(f"FAILURE PLACE_DRIFT: position_error={error:.6f}, max_release_drift={max_drift:.6f}")
        if on_block:
            offset = (center[:, :2] - block.data.root_pos_w.torch[:, :2]).abs()
            half_size = torch.tensor(BLOCK_SIZE[:2], device=robot.device) / 2 - 0.005
            if (offset > half_size).any():
                raise RuntimeError("FAILURE BLOCK_SUPPORT: ball center left block footprint")
        if clearance() < -BALL_SETTLE_CONTACT_CLEARANCE:
            raise RuntimeError("FAILURE PLACE: ball penetrated table during release/retreat")

    monitor_release()
    print(f"STATE=RELEASE clearance_m={clearance():.6f}", flush=True)
    hold_grasp(
        sim, robot, block, ball, GRIPPER_GRASP_POSITION, GRIPPER_OPEN_POSITION, release=True, on_step=monitor_release
    )
    open_target = robot.data.joint_pos_target.torch.clone()
    clearance_steps = 0
    for _ in range(60):
        gap, width = grasp_geometry_metrics(robot, ball)
        if gap >= width + 2 * BALL_PARTICLE_RADIUS + 0.001:
            break
        step_scene(sim, robot, block, ball, open_target)
        monitor_release()
        clearance_steps += 1
    else:
        raise RuntimeError("FAILURE PLACE: fingers not fully clear of ball; refusing retreat")

    print("STATE=RETREAT", flush=True)
    target = grasp_position()
    target[:, 2] += 0.10
    move_grasp_point(
        sim,
        robot,
        block,
        ball,
        target,
        GRIPPER_OPEN_POSITION,
        target_rotation_w=destination_rotation_w,
        smooth=True,
        on_step=monitor_release,
    )
    final_steps = wait_until_scene_settled(
        sim,
        robot,
        block,
        ball,
        robot.data.joint_pos_target.torch.clone(),
        episode=episode,
        minimum_steps=60,
        ball_reset_center=original_center,
        on_step=monitor_release,
        timeout_limit=240,
        support_height=support_height if on_block else None,
    )
    result = {
        "episode": episode,
        "original_center_m": list(original_center),
        "final_center_m": ball_center(ball)[0].cpu().tolist(),
        "xy_error_m": torch.linalg.vector_norm(ball_center(ball)[:, :2] - origin[:, :2]).item(),
        "max_release_drift_m": max_drift,
        "contact_clearance_m": clearance(),
        "clearance_wait_after_open_s": clearance_steps * sim.get_physics_dt(),
        "post_retreat_check_s": final_steps * sim.get_physics_dt(),
    }
    result["target_center_m"] = origin[0].cpu().tolist()
    print(f"{'PLACE_ON_BLOCK_RESULT' if on_block else 'PUT_BACK_RESULT'}={result}", flush=True)
    marker = "YAM_DEFORMABLE_BALL_ON_BLOCK_OK" if on_block else "YAM_DEFORMABLE_BALL_PUT_BACK_OK"
    print(f"{marker} episode={episode}", flush=True)
    return result


# ============================================================
# 21. Main
# ============================================================


def main() -> None:
    """参数校验 -> 建场景 -> episode reset/settle -> 可选抓取/放置 -> 汇总。"""
    if args_cli.episodes < 1 or args_cli.steps_per_episode < 1:
        raise ValueError("--episodes and --steps-per-episode must both be positive")
    if args_cli.motion_steps < 240:
        raise ValueError("--motion-steps must be >= 240; adjust speed constants to tune faster motion")
    if args_cli.lift_height <= 0.0:
        raise ValueError("--lift-height must be positive")
    if args_cli.gripper_check and (args_cli.pick_lift or args_cli.episodes != 1):
        raise ValueError("--gripper-check requires --episodes 1 and no --pick-lift")
    if args_cli.stop_after_hold and not args_cli.pick_lift:
        raise ValueError("--stop-after-hold requires --pick-lift")
    if args_cli.put_back and (not args_cli.pick_lift or args_cli.stop_after_hold):
        raise ValueError("--put-back requires --pick-lift and cannot be combined with --stop-after-hold")
    if args_cli.place_on_block and (not args_cli.pick_lift or args_cli.put_back or args_cli.stop_after_hold):
        raise ValueError("--place-on-block requires --pick-lift; cannot combine with --put-back/--stop-after-hold")

    mode = (
        "GRIPPER_CHECK (arm fixed)"
        if args_cli.gripper_check
        else (
            "PICK_LIFT_PUT_BACK"
            if args_cli.put_back
            else (
                "PICK_LIFT_ON_BLOCK"
                if args_cli.place_on_block
                else "PICK_LIFT" if args_cli.pick_lift else "RESET_ONLY (arm fixed)"
            )
        )
    )
    print(f"RUN_MODE={mode} seed={args_cli.seed} episodes={args_cli.episodes}", flush=True)
    print(f"WORKSPACE base={ROBOT_BASE_XY} radius={OBJECT_RADIUS_RANGE} bearing={OBJECT_BEARING_RANGE}", flush=True)
    sim = make_sim()
    # eye=观察相机世界位置；target=镜头朝向点。
    sim.set_camera_view(eye=(1.25, -1.20, 1.15), target=(*ROBOT_BASE_XY, TABLE_TOP_Z))
    robot, block, ball = spawn_scene()
    sim.reset()
    if robot.num_joints != 8:
        raise RuntimeError(f"Expected 8 YAM joints, got {robot.num_joints}: {robot.joint_names}")

    rng = random.Random(args_cli.seed)
    results, pick_lift_results, placement_results, motion_timings = [], [], [], []
    settled_ball_template = None

    for episode in range(args_cli.episodes):
        print(f"EPISODE_START index={episode} total={args_cli.episodes}", flush=True)
        block_pos, ball_pos = sample_object_positions(rng)
        print(f"EPISODE_RANDOM_TARGETS episode={episode} block={block_pos} ball={ball_pos}", flush=True)
        ball_reset_center = reset_episode(robot, block, ball, block_pos, ball_pos, ball_template=settled_ball_template)
        settle_steps = wait_until_scene_settled(
            sim,
            robot,
            block,
            ball,
            robot.data.default_joint_pos.torch,
            episode=episode,
            minimum_steps=args_cli.steps_per_episode,
            ball_reset_center=ball_reset_center,
            required_steps=GRASP_READY_REQUIRED_STEPS if args_cli.pick_lift else SCENE_SETTLE_REQUIRED_STEPS,
            ball_root_threshold=GRASP_READY_ROOT_SPEED if args_cli.pick_lift else BALL_SETTLE_ROOT_SPEED,
            ball_nodal_threshold=GRASP_READY_MAX_NODAL_SPEED if args_cli.pick_lift else BALL_SETTLE_MAX_NODAL_SPEED,
            ball_window_drift=GRASP_READY_WINDOW_DRIFT if args_cli.pick_lift else BALL_SETTLE_WINDOW_DRIFT,
        )
        if settled_ball_template is None:
            settled_ball_template = ball.data.nodal_state_w.torch.clone()
            settled_ball_template[..., 3:] = 0.0
            print("BALL_SETTLED_TEMPLATE_CAPTURED episode=0", flush=True)

        if args_cli.gripper_check:
            check_gripper_motion(sim, robot, block, ball)
        if args_cli.pick_lift:
            motion_start_step = SCENE_STEP_COUNT
            motion_start_wall = time.perf_counter()
            settled_ball_pos = tuple(ball_center(ball)[0].cpu().tolist())
            pick_lift_result = pick_and_lift_ball(
                sim, robot, block, ball, settled_ball_pos, args_cli.lift_height, episode
            )
            pick_lift_result["episode"] = episode
            pick_lift_results.append(pick_lift_result)
            if not pick_lift_result.get("hold_only"):
                if pick_lift_result["fk_alignment_error_m"] > 0.002:
                    raise RuntimeError(f"URDF FK does not align with the simulated gripper: {pick_lift_result}")
                minimum_lift = pick_lift_result["commanded_lift_m"] * MIN_LIFT_RATIO
                if pick_lift_result["actual_ball_lift_m"] < minimum_lift:
                    raise RuntimeError(
                        f"Ball was not lifted high enough; required {minimum_lift:.3f} m: {pick_lift_result}"
                    )
                print(f"YAM_DEFORMABLE_BALL_LIFT_OK episode={episode}", flush=True)
            if args_cli.put_back:
                placement_results.append(put_ball_back(sim, robot, block, ball, settled_ball_pos, episode))
            elif args_cli.place_on_block:
                placement_results.append(
                    put_ball_back(sim, robot, block, ball, settled_ball_pos, episode, on_block=True)
                )
            timing = {
                "episode": episode,
                "sim_s": (SCENE_STEP_COUNT - motion_start_step) * sim.get_physics_dt(),
                "wall_s": time.perf_counter() - motion_start_wall,
            }
            motion_timings.append(timing)
            print(f"MOTION_TIMING={timing}", flush=True)

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
        print(f"EPISODE_COMPLETE index={episode} total={args_cli.episodes}", flush=True)

    print(f"NEWTON_SOLVER={type(sim.cfg.physics.solver_cfg).__name__}")
    print(f"YAM_JOINT_NAMES={robot.joint_names}")
    print(f"YAM_BODY_NAMES={robot.body_names}")
    gripper_index = robot.body_names.index("gripper")
    print(f"YAM_GRIPPER_POSE={robot.data.body_link_pose_w.torch[0, gripper_index].cpu().tolist()}")
    print("YAM_KINEMATICS=project URDF-derived Torch FK/Jacobian")
    for result in results:
        print(f"RESET_RESULT={result}")
    for pick_lift_result in pick_lift_results:
        print(f"PICK_LIFT_RESULT={pick_lift_result}")
    print(
        f"EPISODE_SUMMARY requested={args_cli.episodes} reset_ok={len(results)} "
        f"pick_ok={len([r for r in pick_lift_results if not r.get('hold_only')])} "
        f"placement_ok={len(placement_results)} timings={motion_timings}",
        flush=True,
    )
    print("YAM_DEFORMABLE_TASK_INTEGRATION_OK")

    if args_cli.keep_open:
        if args_cli.headless:
            raise ValueError("--keep-open requires the Kit visualizer; use --visualizer kit.")
        print("GUI_READY_CLOSE_WINDOW_TO_EXIT", flush=True)
        hold_target = robot.data.joint_pos_target.torch.clone()
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
