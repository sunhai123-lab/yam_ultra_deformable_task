# `yam_pick_ball.py` 100% 学习说明

> 基于 `fix/rigid-block-stability` 分支的 `scripts/yam_pick_ball.py`，读取时分支 HEAD 为 `05874a5`（`learn to change the range`）。本说明按“你改哪个参数，会影响什么”的角度整理。

## 1. 先看清整个程序在干什么

这份脚本不是 RL policy，而是一个确定性硬编码任务：创建 Newton 刚体+VBD 软体场景，随机复位方块/球，等待稳定，用项目自写 FK/Jacobian/DLS IK 控制 YAM 到预抓取位，下降、闭爪、抬升，并通过几何、速度、FK 对齐和软球真实抬升量判断成功；可选再放回桌面或放到方块上。

程序最重要的数据流是：

```text
场景参数
  ↓
spawn_scene()
  ↓
reset_episode()
  ↓
wait_until_scene_settled()
  ↓
ball_center() 得到真实球心
  ↓
solve_approach_joint_target()   离线 DLS 求预抓取 q
  ↓
execute_arm_trajectory()        joint-space 到预抓取
  ↓
move_grasp_point()              Cartesian 闭环下降
  ↓
hold_grasp()                    物理闭爪
  ↓
quick_grasp_check()             抓取稳定性验收
  ↓
move_grasp_point()              抬升
  ↓
实际球心升高量验收
```

## 2. 当前你已经改过的场景坐标

当前源码是：

```python
TABLE_SIZE = (1.50, 1.50, 0.08)
TABLE_CENTER = (0.0, 0.0, 0.0)
TABLE_TOP_Z = 0.04

BLOCK_X_RANGE = (-0.40, 0.40)
BLOCK_Y_RANGE = (0.40, 0.40)
BALL_X_RANGE = (-0.40, 0.40)
BALL_Y_RANGE = (-0.40, -0.40)
```

`TABLE_SIZE=(X,Y,Z)` 是桌子尺寸，单位 m。`TABLE_CENTER` 是桌子 Cuboid 的几何中心，不是桌面位置；因此现在桌面顶面是 `z=+0.04`。机器人 spawn 的 Z 也是 `TABLE_TOP_Z`，所以机器人基座会跟桌面高度联动。

`BLOCK_Y_RANGE=(0.40,0.40)` 并不是随机范围，而是 **Y 永远等于 +0.40**；同理 `BALL_Y_RANGE=(-0.40,-0.40)` 代表球 Y 永远是 `-0.40`。如果你希望 Y 真正随机，可以改成例如：

```python
BLOCK_Y_RANGE = (0.10, 0.35)
BALL_Y_RANGE = (-0.35, -0.10)
```

但最终是否能用，还要同时通过 `SAMPLE_REACH_RADIUS` 和 `MIN_OBJECT_PLANAR_DISTANCE` 的过滤。

## 3. CLI 参数

| 参数 | 默认 | 真正含义 |
|---|---:|---|
| `--episodes` | 10 | 完整 reset/测试轮数 |
| `--steps-per-episode` | 30 | 每轮 reset 后最少推进物理步数；不是总步数 |
| `--seed` | 7 | Python 随机位置采样种子 |
| `--pick-lift` | False | 是否真正执行抓球+抬升 |
| `--disable-cuda-graph` | False | 调试时关闭 CUDA Graph |
| `--put-back` | False | 抓起后放回原位置 |
| `--place-on-block` | False | 抓起后放到方块顶面 |
| `--motion-steps` | 240 | **时间倍率基准**，240=1×，480≈半速；不是每段固定 240 步 |
| `--lift-height` | 0.03 | 抓住后期望抬升高度，m |
| `--keep-open` | False | GUI 最后保持打开 |
| `--stop-after-hold` | False | 闭爪稳定后停，不 lift |
| `--gripper-check` | False | 只测试夹爪开/关，不接近球 |

## 4. 路径参数

`PROJECT_ROOT = Path(__file__).resolve().parents[1]` 假设脚本位于 `<project>/scripts/`。`YAM_USD` 指向已转换好的 YAM USD。这里如果路径错，不是运动学问题，而是资产加载问题。

## 5. 随机位置和可达性参数

`SAMPLE_REACH_RADIUS=(0.20,0.40)` 是从 `ROBOT_BASE_XY=(-0.22,0)` 到物体 XY 的粗略水平距离范围。它只是初筛，不证明完整姿态 IK 一定可达。

`MIN_OBJECT_PLANAR_DISTANCE=0.18` 强制方块与球 XY 至少相距 18 cm。`SAMPLING_MAX_ATTEMPTS=100` 表示最多尝试 100 次；如果你把随机范围、可达半径和最小间距设得互相矛盾，就会报 `Could not sample reachable object positions`。

## 6. 方块参数

`BLOCK_SIZE=(0.07,0.07,0.05)` 是 7×7×5 cm。`BLOCK_RESET_CLEARANCE=0.0005` reset 时多抬 0.5 mm，避免一开始深度穿透桌面。

`RIGID_STATIC_FRICTION=0.6`、`RIGID_DYNAMIC_FRICTION=0.5`、`RIGID_RESTITUTION=0` 是刚体表面材料。静摩擦影响开始滑动，动摩擦影响已经滑动后的阻力，恢复系数 0 代表不希望弹跳。

`BLOCK_SETTLE_LINEAR_SPEED=5e-4`、`BLOCK_SETTLE_ANGULAR_SPEED=1e-2`、`BLOCK_SETTLE_CLEARANCE=1e-3` 不是物理材料，而是“什么时候认为方块已经稳定”的验收阈值。

## 7. 软球材料参数

`BALL_RADIUS=0.04` 是几何半径 4 cm。`BALL_DENSITY=500 kg/m³` 决定质量。`BALL_YOUNGS_MODULUS=5e4 Pa` 与 `BALL_POISSONS_RATIO=0.35` 被转换成 Lamé 参数：

```text
μ = E / [2(1+ν)]
λ = Eν / [(1+ν)(1-2ν)]
```

它们描述的是**球内部材料**，不是接触材料。

`BALL_PARTICLE_RADIUS=0.006` 是 VBD 节点接触包络半径，6 mm。很多抓取 gap 判断都必须加上两侧 `2*particle_radius`，所以它会直接影响“夹爪是否真的足够打开”和“球底是否已经接触桌面”。

`BALL_INTERNAL_DAMPING=0.01` 是内部材料阻尼，抑制球体内部振动，不等于 `soft_contact_kd`。

## 8. 软球稳定阈值

`BALL_SETTLE_WINDOW_DRIFT=0.25mm`：一段连续窗口中球心 XY 漂移必须很小。`BALL_SETTLE_ROOT_SPEED=1mm/s`：所有节点平均速度很小。`BALL_SETTLE_MAX_NODAL_SPEED=1cm/s`：任何单节点也不能振得太快。`BALL_SETTLE_CONTACT_CLEARANCE=1.5mm`：球底接触包络要靠近桌面。

这四个条件同时存在，是因为只看球心速度会漏掉“球心不动但内部还在抖”的情况。

## 9. 场景稳定窗口

`SCENE_SETTLE_REQUIRED_STEPS=30` 要连续 30 帧满足稳定；`SCENE_SETTLE_TIMEOUT_STEPS=480` 最多等 480 帧。dt=1/120s 时约 4 秒仿真时间。

抓取前的 `GRASP_READY_*` 更强调“准备接触前球内部真的安静”，其中 `GRASP_READY_MAX_NODAL_SPEED=0.003 m/s` 比普通 settle 的 0.01 更严格。

## 10. 抓取点和夹爪几何

`GRASP_POINT_LOCAL=(0,0,-0.125)` 表示真正要对准球心的点，在 gripper link 原点局部 Z 负方向 12.5 cm。你的 FK 原本算的是 gripper origin，因此 `grasp_point_kinematics()` 做：

```text
p_grasp = p_gripper + R_gripper * r_local
```

并把 Jacobian 从 origin 转到该点：

```text
v_point = v_origin + ω × r
Jv_point = Jv_origin + Jω × r
```

`GRASP_ROTATION_W=diag(-1,-1,+1)` 是抓球时末端期望固定姿态。

`LEFT_PAD_LOCAL_POS` / `RIGHT_PAD_LOCAL_POS` 是各夹指内侧代表点在 tip frame 中的位置；它们不是 joint 坐标。代码把这两个点变换到世界系，直接测真实 gap。

## 11. 夹爪关节

`GRIPPER_OPEN_POSITION=-0.04695 m` 是每根夹指最大张开位置。两个 joint 都是负值打开，是因为左右 prismatic axis 方向相反。

`GRIPPER_COMMAND_SPEED=0.04 m/s` 用于空载夹爪测试；`GRIPPER_CONTACT_SPEED=0.020 m/s` 用于真实接触球，降低冲击；`PLACE_RELEASE_SPEED=0.012 m/s` 用于放球时慢慢卸载。

## 12. 机械臂运动速度

`ARM_CARTESIAN_SPEED=0.30 m/s` 是普通 Cartesian 移动峰值速度，`DESCEND_SPEED=0.14 m/s` 是最后靠近球的慢速，`PLACE_SPEED=0.03 m/s` 是最后接触支撑面的低速。

`ARM_JOINT_SPEED=3 rad/s` 只用于第一次从当前姿态到“离线求好的预抓取 q”的 joint-space 轨迹。

`move_grasp_point()` 的 `travel_steps` 是由距离自动算的：

```text
steps ≈ distance / (speed * dt)
```

如果 `smooth=True`，再乘 1.5，因为 cubic smoothstep 的峰值速度是平均速度的 1.5 倍。`args_cli.motion_steps/240` 再作为整体时间倍率。

## 13. 抓取目标 gap 的材料近似

代码先用：

```text
E_eff = E/(1-ν²)
compression ≈ stress/E_eff
```

再限制最多 12% 压缩。`NOMINAL_GRIP_STRESS=5kPa` 是人为的名义压缩应力，不是仿真实测接触压力。最后得到 `GRIPPER_TARGET_GAP` 和两个对称 prismatic joint 的 `GRIPPER_GRASP_POSITION`。

这组参数影响“夹多紧”，与 VBD 内部 `E/ν`、刚软接触 `soft_contact_ke/kd/mu` 共同决定实际变形和是否滑落。

## 14. Newton solver 参数

刚体 `MJWarpSolverCfg`：`njmax=256` 是约束容量，不是关节数；`nconmax=2048` 是接触容量；`ls_iterations=20` 是 line-search 预算；`cone='pyramidal'` 是摩擦锥近似；`integrator='implicitfast'` 是隐式积分；`ccd_iterations=100` 是较保守的凸碰撞迭代预算。

软体 `VBDSolverCfg(iterations=20)` 每子步做 20 次 VBD 迭代，`integrate_with_external_rigid_solver=True` 表示刚体交给 MJWarp。`coupling_mode='two_way'` 是物理抓取必须的：夹爪推球，球也反作用夹爪。

`SimulationCfg(dt=1/120)` 是 120Hz 主物理步；`num_substeps=10` 又把每个主步细分 10 个 Newton 子步。

`soft_contact_ke=1e4`、`soft_contact_kd=1e-2`、`soft_contact_mu=5` 是**刚体/软体接触界面**参数，不是软球内部材料。`mu=5` 很高，主要服务鲁棒抓持。

## 15. Robot actuator 参数

USD 载入阶段 `max_force=80`、`max_joint_velocity=3` 是 joint drive 基础属性。随后 `ImplicitActuatorCfg` 又设置仿真执行器：

```text
arm: effort_limit=120, velocity_limit=3, stiffness=3000, damping=100
gripper: effort_limit=15, velocity_limit=0.05, stiffness=1000, damping=50
```

`stiffness/damping` 属于关节 position controller，不是软体弹性，也不是接触刚度。

## 16. 为什么指尖改成 convexDecomposition

原指尖三角网格带凹部。如果直接做单一 convex hull，会把凹陷“填满”，导致软球接触形状与视觉模型不一致。代码运行时把 MeshCollision approximation 改成 `convexDecomposition`，只是修改当前 Stage，不改磁盘 USD。

## 17. `reset_episode()` 最容易忽略的一点

方块用 root pose reset；软球没有简单 rigid root pose，所以直接改所有 nodal state。

`ball_pos[2]` 实际没有直接用来放置软球。代码用当前节点最低 Z，把整个球平移到：

```text
TABLE_TOP_Z + BALL_PARTICLE_RADIUS + BALL_RESET_CLEARANCE
```

所以你想改球初始高度，不能只盯着 `sample_object_positions()` 的 `ball_pos.z`；要看 `reset_episode()` 的最低节点对齐逻辑。

第 0 episode 沉降后还会保存 `settled_ball_template`，后续 episode 复用已沉降形状，从而避免每轮重新从理想球形滚动很久。

## 18. `wait_until_scene_settled()`

它不是简单 sleep。每个 physics step 都重新测方块和球，然后要求连续若干帧同时满足条件。只要某一帧不稳定，`stable_steps` 清零。

球还额外维护 `stable_centers`，看最近窗口内 XY 的 max-min 漂移，所以“瞬间速度很小但还在缓慢爬动”也不会轻易误判稳定。

## 19. `move_grasp_point()`

这是整个 Cartesian 控制核心。每一步：读取真实 q -> FK/J -> 算 waypoint -> DLS -> clamp Δq -> clamp joint limit -> 写 8 关节 target -> physics step -> 再读取真实状态。

DLS 实时更新限制 ±0.02rad/step；离线预抓取 IK 则允许 ±0.05rad/iteration。完成后要求位置 <=2mm、姿态矩阵误差 <=0.02、关节最大速度 <=0.05rad/s，连续 3 帧才退出。

## 20. `solve_approach_joint_target()` 与 `execute_arm_trajectory()` 为什么分开

前者是“纯数学离线 IK”，不推进 physics，从固定 seed 迭代 300 次求一个 q。后者才真正把物理机器人从当前 q 平滑插值到这个 q。

这样第一次大范围接近球时不需要边走边从远处做 Cartesian DLS，通常更稳定；到达预抓取附近后，再切换成 `move_grasp_point()` 的实时 Cartesian 闭环。

## 21. `hold_grasp()` 为什么保持 `joint_pos_target` 而不是实际 `joint_pos`

实际关节角会因为重力、接触、控制误差与 target 有小偏差。如果每一帧把实际 q 当成新的 target，相当于“认可了偏差”，机械臂 target 会逐步漂走。保存最后的 drive target 才是真正的“保持姿态”。

## 22. 抓取流程中的关键距离

预抓取：球心上方 `0.10m`。快速下降终点：球心上方 `0.045m`。最后 4.5cm 用 `DESCEND_SPEED`。下降后球的横向漂移若 >3mm 就拒绝闭爪。

闭爪后 `quick_grasp_check()` 不只看“夹爪关了”，还看真实 gap、球宽、particle envelope、夹指停止、球相对夹爪速度。lift 后再增加球底离台 >=2mm 和实际升高 >=命令高度的 85%。

## 23. 放球流程

先把球抬到安全高度，再水平搬到目标 XY；高速下降到约 15mm 净空，再用 0.03m/s 慢速根据实际 clearance 下探；停 6 步后才慢慢松爪。夹指真实 gap 足够离开球以后才向上撤离 10cm，最后再等整个场景稳定。

放到方块上时还要求方块几乎水平 `up_z>=0.9998`，且从放置前位置移动 <=4mm。

## 24. `main()` 中相机

```python
sim.set_camera_view(eye=(1.25,-1.20,1.15), target=(0.12,0.0,TABLE_TOP_Z))
```

`eye` 是 GUI 相机世界位置；`target` 是看向的世界点。因为 target Z 使用 `TABLE_TOP_Z`，你改桌面高度时镜头关注高度会自动跟着变。

## 25. 你现在最建议优先改的参数

学习阶段建议按这个顺序：先只动 `TABLE_SIZE/TABLE_CENTER` 看世界坐标；再动 `BLOCK/BALL_X/Y_RANGE` 学随机采样；再动机器人 `init_state.pos` 和 `joint_pos` 看机器人位姿；再动相机；最后才动 `ARM_CARTESIAN_SPEED/DESCEND_SPEED/GRIPPER_CONTACT_SPEED`。在你还没把场景坐标和 IK 彻底弄明白前，不建议同时改 solver、soft contact、E/ν 和 actuator stiffness，因为那会把“几何/运动问题”与“物理接触问题”混在一起。
