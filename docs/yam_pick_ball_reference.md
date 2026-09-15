# `yam_pick_ball.py` 参数与函数参考

本文档只对应当前主运行文件 [`scripts/yam_pick_ball.py`](../scripts/yam_pick_ball.py)。以后如果源码参数变化，以源码为准；本文档用于解释“这个参数/函数是什么、为什么存在、改了会怎样”。

## 1. 程序边界

这是一个单机械臂、单场景、按 episode 重复执行的确定性任务控制器。它不是强化学习策略，没有奖励函数、训练循环或神经网络。每个 episode 的典型流程是：采样物体位置 → reset → 等待稳定 → 预抓取 → 下降 → 闭爪 → 抓取检查 → 抬升 → 可选搬运/放置 → 释放 → 撤离 → 再次稳定检查。

机器人运动依赖项目内 [`yam_kinematics.py`](../source/yam_ultra_deformable_place/yam_ultra_deformable_place/yam_kinematics.py) 的 FK、几何 Jacobian 和 DLS IK。软球不是刚体，`ball_center()` 使用所有 VBD 节点世界位置的算术平均估计球心。

## 2. 坐标、单位与常见张量

世界坐标使用右手系，Z 向上。位置单位 m，时间 s，角度 rad，线速度 m/s，角速度 rad/s。变量后缀 `_w` 表示世界坐标，`local` 表示对应 link 的局部坐标。

| 数据 | 常见 shape | 含义 |
| --- | --- | --- |
| `root_link_pose_w.torch` | `(1, 7)` | 机器人根 link 的 XYZ + 四元数 XYZW |
| `joint_pos.torch` | `(1, 8)` | joint1~6 机械臂，joint7~8 夹爪 |
| `arm_pos` | `(1, 6)` | 六个机械臂关节角 |
| `target_pos_w` | `(1, 3)` | 目标世界 XYZ |
| `gripper_rot_w` | `(1, 3, 3)` | gripper 局部坐标轴在世界中的旋转矩阵 |
| `jac` | `(1, 6, 6)` | 抓取点 6D 几何 Jacobian |
| `nodal_state` | `(1, V, 6)` | V 个软体节点位置 XYZ + 速度 XYZ |
| `finger_surface_points()` | `(2, 3)` | 左右夹指内侧代表点世界坐标 |

当前项目的四元数接口使用 XYZW，单位四元数为 `(0, 0, 0, 1)`。

## 3. CLI 参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--episodes` | `10` | 完整 episode 数量 |
| `--steps-per-episode` | `30` | reset 后至少推进多少主物理步；不是总步数 |
| `--seed` | `7` | Python 随机位置采样种子 |
| `--pick-lift` | False | 执行抓取与抬升 |
| `--put-back` | False | 抓起后放回原位置，要求同时 `--pick-lift` |
| `--place-on-block` | False | 抓起后搬到方块上，要求同时 `--pick-lift` |
| `--stop-after-hold` | False | 闭爪并稳定后停止，不抬升 |
| `--gripper-check` | False | 固定机械臂，只测试夹爪开合 |
| `--motion-steps` | `240` | 运动时间倍率基准；当前要求 `>=240`，240 为 1× 时间，480 约为半速 |
| `--lift-height` | `0.03` | 抬升高度，单位 m |
| `--disable-cuda-graph` | False | 调试时关闭 CUDA Graph |
| `--keep-open` | False | GUI 模式结束后保持窗口 |

## 4. 场景几何与随机区域

```python
TABLE_SIZE = (1.50, 1.50, 0.08)
TABLE_CENTER = (0.0, 0.0, 0.0)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0
ROBOT_BASE_XY = TABLE_CENTER[:2]
OBJECT_RADIUS_RANGE = (0.30, 0.45)
OBJECT_BEARING_RANGE = (-2.40, 3.00)
MIN_OBJECT_PLANAR_DISTANCE = 0.18
SAMPLING_MAX_ATTEMPTS = 100
MAX_TRANSFER_BEARING_DELTA = math.pi
TRANSFER_BALL_ALIGNMENT_TOLERANCE = 0.002
BLOCK_SIZE = (0.07, 0.07, 0.05)
BLOCK_RESET_CLEARANCE = 5.0e-4
```

`TABLE_CENTER` 是桌体几何中心，不是桌面位置。当前桌面顶部 `TABLE_TOP_Z=0.04 m`。机器人 XY 与桌心绑定，Z 在 `spawn_scene()` 中取 `TABLE_TOP_Z`。

随机位置采用极坐标：

```text
x = robot_x + r cos(theta)
y = robot_y + r sin(theta)
```

`OBJECT_RADIUS_RANGE` 控制物体距离机器人中心的范围；`OBJECT_BEARING_RANGE` 控制物体绕机器人分布的方向。采样后还要检查：物体完整包络在桌面内、球和方块至少相距 0.18 m、两者方位差不超过 π。扩大范围不等于工作空间一定可达，最终仍由 IK 与运行时误差检查决定。

## 5. 刚体材料与稳定阈值

```python
RIGID_STATIC_FRICTION = 0.6
RIGID_DYNAMIC_FRICTION = 0.5
RIGID_RESTITUTION = 0.0
BLOCK_SETTLE_LINEAR_SPEED = 5.0e-4
BLOCK_SETTLE_ANGULAR_SPEED = 1.0e-2
BLOCK_SETTLE_CLEARANCE = 1.0e-3
SUPPORT_CONTACT_STIFFNESS = 2500.0
SUPPORT_CONTACT_DAMPING = 200.0
```

前三项是真正的刚体表面物理参数。`BLOCK_SETTLE_*` 是“什么时候认为方块稳定”的验收阈值，不会直接改变物体运动。`SUPPORT_CONTACT_STIFFNESS/DAMPING` 写到桌面和方块的 Newton 接触属性，控制支撑接触约束恢复；它们不是软球材料的杨氏模量。

## 6. VBD 软球参数

```python
BALL_RADIUS = 0.04
BALL_DENSITY = 500.0
BALL_YOUNGS_MODULUS = 5.0e4
BALL_POISSONS_RATIO = 0.35
BALL_PARTICLE_RADIUS = 0.006
BALL_RESET_CLEARANCE = 5.0e-4
BALL_INTERNAL_DAMPING = 0.01
```

`BALL_RADIUS` 是可视/几何半径。`BALL_PARTICLE_RADIUS` 是 VBD 粒子接触包络半径，两者不能混用。杨氏模量 `E` 和泊松比 `ν` 在 `spawn_scene()` 中转为 Lamé 参数：

```text
mu = E / [2(1+nu)]
lambda = E*nu / [(1+nu)(1-2nu)]
```

`BALL_INTERNAL_DAMPING` 是软体内部材料阻尼，不是刚软接触阻尼。

软球稳定判据：

```python
BALL_SETTLE_WINDOW_DRIFT = 2.5e-4
BALL_SETTLE_ROOT_SPEED = 1.0e-3
BALL_SETTLE_MAX_NODAL_SPEED = 1.0e-2
BALL_SETTLE_CONTACT_CLEARANCE = 1.5e-3
SCENE_SETTLE_REQUIRED_STEPS = 30
SCENE_SETTLE_TIMEOUT_STEPS = 480
GRASP_READY_REQUIRED_STEPS = 12
GRASP_READY_ROOT_SPEED = 1.5e-3
GRASP_READY_MAX_NODAL_SPEED = 3.0e-3
GRASP_READY_WINDOW_DRIFT = 5.0e-4
```

抓取前使用 `GRASP_READY_*`，普通 reset-only 使用 `SCENE_SETTLE_*`。稳定需要连续多步满足几何、节点速度和漂移阈值。

## 7. 抓取点与夹爪几何

```python
GRASP_POINT_LOCAL = (0.0, 0.0, -0.125)
GRASP_ROTATION_W = ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
LEFT_PAD_LOCAL_POS = (0.014, 0.0469184183, -0.1980911)
RIGHT_PAD_LOCAL_POS = (0.014, 0.0458805300, -0.1981001)
```

FK 返回的是 gripper link 原点。真正抓球的控制点位于 gripper 局部 Z 负方向 0.125 m：

```text
p_grasp = p_gripper + R_gripper * r_local
```

把 Jacobian 从 gripper 原点转到抓取点时使用刚体点速度关系：

```text
v_point = v_origin + omega × r
Jv_point = Jv_origin + Jw × r
```

`LEFT_PAD_LOCAL_POS/RIGHT_PAD_LOCAL_POS` 是夹指内侧代表点的 tip-local 坐标，用于测量真实夹爪 gap 和球在夹指轴向的宽度。

## 8. 机器人速度与抓取目标

当前源码：

```python
GRIPPER_OPEN_POSITION = -0.04695
GRIPPER_COMMAND_SPEED = 0.04
GRIPPER_CONTACT_SPEED = 0.030
ARM_CARTESIAN_SPEED = 0.35
ARM_JOINT_SPEED = 3.0
ARM_ORIENTATION_SPEED = 1.2
MOTION_SETTLE_STEPS = 120
DESCEND_SPEED = 0.25
LIFT_SPEED = ARM_CARTESIAN_SPEED
TRANSFER_SPEED = min(ARM_CARTESIAN_SPEED, 0.30)
TRANSFER_DLS_GAIN = 3.0
TRANSFER_MAX_JOINT_COMMAND_STEP = 0.04
PLACE_APPROACH_SPEED = ARM_CARTESIAN_SPEED
PLACE_SPEED = 0.030
PLACE_RELEASE_SPEED = 0.012
```

`ARM_CARTESIAN_SPEED` 是一般末端平移速度基准；`ARM_ORIENTATION_SPEED` 是姿态插值角速度；`ARM_JOINT_SPEED` 只用于首次 joint-space approach。`TRANSFER_SPEED` 单独限制搬运速度，因为此时夹爪已夹住软球。

`MOTION_SETTLE_STEPS` 允许轨迹结束后继续闭环收敛，超时后明确报错，而不是无限等待。

## 9. 抓取压缩启发式

```python
GRIPPER_CLOSED_GAP = 0.00006211
GRIPPER_OPEN_GAP = GRIPPER_CLOSED_GAP - 2.0 * GRIPPER_OPEN_POSITION
NOMINAL_GRIP_STRESS = 5.0e3
BALL_EFFECTIVE_MODULUS = BALL_YOUNGS_MODULUS / (1.0 - BALL_POISSONS_RATIO**2)
BALL_COMPRESSION_RATIO = min(0.12, NOMINAL_GRIP_STRESS / BALL_EFFECTIVE_MODULUS)
GRIPPER_TARGET_GAP = 2.0 * BALL_RADIUS * (1.0 - BALL_COMPRESSION_RATIO)
GRIPPER_GRASP_POSITION = -0.5 * (GRIPPER_TARGET_GAP - GRIPPER_CLOSED_GAP)
MIN_LIFT_RATIO = 0.85
```

这里不是实时力控制。代码用名义应力与有效模量估算压缩比例，并限制最大压缩为 12%，再把目标双指间距换算成两个对称 prismatic joint 的目标位置。

## 10. Newton 求解器参数

`make_sim()` 创建 `CoupledMJWarpVBDSolverCfg`：MJWarp 处理机器人和刚体，VBD 处理软球，`coupling_mode="two_way"` 让软球反作用到夹爪。主物理步长为 `1/120 s`，每个主步有 10 个 Newton substeps。

主要参数：

```text
MJWarp: njmax=256, nconmax=2048, ls_iterations=20,
        cone='pyramidal', integrator='implicitfast', ccd_iterations=100
VBD:    iterations=20, integrate_with_external_rigid_solver=True,
        particle_enable_self_contact=False
contact: soft_contact_ke=1e4, soft_contact_kd=1e-2, soft_contact_mu=5.0
```

`soft_contact_ke/kd/mu` 是刚体-软体接触界面参数，不是软球内部材料参数。

## 11. Robot actuator

机器人有 8 个 joint：joint1~6 为机械臂旋转关节，joint7~8 为夹爪移动关节。当前 ImplicitActuator：

```text
arm:     effort_limit=120, velocity_limit=3.0, stiffness=3000, damping=100
gripper: effort_limit=15,  velocity_limit=0.05, stiffness=1000, damping=50
```

这些 stiffness/damping 是关节位置驱动器参数，不是接触刚度，也不是 VBD 材料刚度。

## 12. 主要函数

### `make_sim()`
创建 Newton 仿真上下文，配置 MJWarp、VBD、two-way coupling、substeps、接触参数和 CUDA Graph。

### `rigid_surface_material()`
返回桌面和方块共用的刚体摩擦/恢复材料。

### `spawn_scene()`
创建环境根节点、地面、灯光、桌子、YAM、动态方块和 VBD 球。机器人 base 放在 `(*ROBOT_BASE_XY, TABLE_TOP_Z)`。运行时将左右指尖碰撞近似改为 `convexDecomposition`，不修改磁盘 USD。

### `sample_object_positions(rng)`
在环形工作区采样方块和球，检查桌面边界、最小物体间距和最大方位差。返回 `(block_pos, ball_pos)`。

### `reset_episode(...)`
复位机器人关节、方块 root pose/velocity 和软球 nodal state。软球 Z 不是直接使用 `ball_pos[2]`，而是根据最低节点、粒子半径和 reset clearance 对齐桌面。

### `block_support_metrics()` / `check_block_support()` / `block_state_is_settled()`
计算方块最低角点离桌面高度、线速度、角速度和是否仍在桌面范围内；用于稳定验收与错误报告。

### `ball_center()` / `ball_settle_metrics()` / `check_ball_support()`
计算节点平均球心、球底接触净空、平均节点速度、最大节点速度和桌面范围检查。

### `step_scene(...)`
下发 8 关节 position target，写入 robot/ball 数据，推进一个主物理步，然后更新 robot、block、ball 缓存。每步额外检查 NaN/Inf。

### `wait_until_scene_settled(...)`
连续推进物理，只有方块和软球连续 `required_steps` 步同时通过稳定条件才返回。支持自定义支撑高度和每步回调。

### `finger_surface_points(robot)`
把左右夹指内侧代表点从 tip-local 坐标变换到世界坐标。

### `grasp_geometry_metrics(robot, ball)`
计算真实夹爪内侧 gap，以及软球沿当前抓取轴的投影宽度。

### `grasp_point_kinematics(root_pose_w, arm_pos)`
调用项目 FK/Jacobian，随后把 gripper origin 平移到真正抓取点，并同步修正线速度 Jacobian。

### `move_grasp_point(...)`
Cartesian 闭环运动核心。根据起终位置距离、平移速度、姿态角速度和 `motion_steps` 时间倍率计算轨迹长度；每个 physics step 重新读取真实 q、计算 FK/Jacobian、DLS IK、关节目标并推进仿真。支持位置-only/pose IK、smoothstep、SLERP 姿态插值、夹爪目标和 `on_step` 监控。

### `radial_grasp_rotation(robot, target_pos_w)`
根据目标相对机器人 base 的方位生成绕世界 Z 轴旋转后的抓取姿态，使夹爪始终从上向下抓，但水平朝向跟随目标。

### `joint1_compatible_bearing(robot, target_pos_w)`
计算目标方位的等价角，并选择落在 joint1 实际限位安全区内的候选角。它只是给 IK 一个合理初值，最终可达性仍由完整 IK 判断。

### `move_ball_on_continuous_polar_path(...)`
抓住球后沿连续极坐标路径搬运。半径和方位共享同一个全局 smoothstep，避免旧式多路点轨迹在每个中间点反复减速到零。目标位置使用实际球心反馈修正。

### `solve_approach_joint_target(...)`
离线迭代 DLS 求预抓取关节目标，不推进 physics。joint1 初值先由目标方位修正，其余关节沿用 `APPROACH_IK_SEED` 的肘形。

### `execute_arm_trajectory(...)`
第一次预抓取使用 joint-space smoothstep 从当前 q 插值到离线 IK 的 q。所需步数由最大关节角差、`ARM_JOINT_SPEED`、physics dt 和 `motion_steps` 计算。

### `hold_grasp(...)`
保持机械臂当前 drive target，只线性移动 joint7/8。用于闭爪、张爪和 gripper-check。

### `check_gripper_motion(...)`
独立测试夹爪从张开到闭合再张开，并用真实几何 gap 验证方向和行程。

### `check_ball_in_gripper(robot, ball)`
检查球心是否仍在两指区域，并检查软球包围盒尺寸是否出现明显异常变形。

### `quick_grasp_check(...)`
在有限时间窗口中检查夹爪是否停止、球和夹爪相对运动是否足够小、夹持包络是否成立；抬升后还检查球底离台和实际升高量。

### `pick_and_lift_ball(...)`
抓取状态机：预抓取 IK → joint-space approach → 再次等待稳定 → 跟踪球心 → 两阶段下降 → 闭爪 → 快速抓取检查 → 抬升 → 实际升高验收。返回关键误差和高度数据。

### `put_ball_back(...)`
统一处理“放回桌面”和“放到方块”。主要阶段：必要时先抬高 → 连续极坐标搬运 → 球心 XY 对正 → 接近支撑面 → 低速下放 → 保持 → 慢速松爪 → 确认夹指脱离 → 垂直撤离 → 最终沉降验收。

### `main()`
检查 CLI 组合是否合法，创建仿真和场景，按 episode 执行采样/reset/settle/抓取/放置，打印结果和最终汇总。GUI `--keep-open` 时最终保持最后 drive target。

## 13. 运动学核心关系

位置 DLS：

```text
e_p = p_target - p_current
Delta q = Jv^T (Jv Jv^T + lambda^2 I)^-1 e_p
```

Pose DLS 将位置误差和姿态误差合成 6D 误差，再使用完整 6×6 Jacobian。当前姿态误差由旋转矩阵轴向叉乘的小角度近似构造。DLS 是迭代局部方法，接近奇异位形时 damping 提高稳定性，但过大会降低响应速度。

## 14. 修改参数时的建议顺序

改场景几何时先改 `TABLE_*` 和 `ROBOT_BASE_XY`，再检查随机区域是否仍在桌面内；改随机工作区时优先逐步调整 `OBJECT_RADIUS_RANGE`，再调整 `OBJECT_BEARING_RANGE`，并用多个 seed 验证；改机械臂速度时优先修改 `ARM_CARTESIAN_SPEED/TRANSFER_SPEED/DESCEND_SPEED/PLACE_SPEED`，不要把 `motion_steps` 当固定阶段步数；改抓取强度时区分软体内部材料、刚软接触参数、夹爪关节驱动和启发式目标 gap，四者不是同一层参数。
