# YAM 软球抓取：逐函数、变量、数学与语法学习手册

配套代码：[yam_pick_ball.py](../scripts/yam_pick_ball.py)。运动学实现：[yam_kinematics.py](../source/yam_ultra_deformable_place/yam_ultra_deformable_place/yam_kinematics.py)。本手册以 2026-09-15 的桌心布局学习版为准。旧参数手册是历史快照，不能直接用其中的随机区域替换当前配置。

## 1. 先理解程序的边界

这是单机械臂、单场景、按 episode 重复的硬编码控制器。没有训练、奖励函数或神经网络。episode 指一次复位到验收的完整实验，不能等同于训练 epoch。

机器人关节通过驱动器产生力，软球靠夹指接触与摩擦抬升。读取球节点估计球的位置是观测；episode 复位时写节点位置是初始化；抓取中没有把节点绑在夹爪上。物体坐标来自仿真真值，因此尚不包括相机识别误差。

执行顺序：

```text
解析 CLI → AppLauncher → 创建场景 → sim.reset
  → 每轮采样 → 复位 → 等待沉降
  → 球上方预抓取 → 下降 → 闭爪 → 快速检查 → 抬升检查
  → 绕行搬运 → 实际球心对正 → 下放 → 松爪 → 撤离 → 稳定验收
  → 下一轮 / 汇总 / 可选保持 GUI
```

关键区别：IK 解出角度不代表运动路径无碰撞；下发角度不代表真实关节已经到达；夹爪闭合不代表球已抓牢；球在方块附近不代表已经稳定放好。

## 2. 坐标、单位、数组形状

世界坐标右手系，Z 向上。位置单位 m、时间 s、角度 rad、速度 m/s、角速度 rad/s。`_w` 表示世界系，`local` 表示所属 link 局部系。

| 数据 | 形状 | 含义 |
| --- | --- | --- |
| `root_link_pose_w.torch` | `(1,7)` | 基座位置 XYZ 加四元数 XYZW |
| `joint_pos.torch` | `(1,8)` | 前六个旋转关节，后两个夹指移动关节 |
| `arm_pos` | `(1,6)` | 当前机械臂关节角 |
| `target_pos_w` | `(1,3)` | 抓取点目标世界位置 |
| `gripper_rot_w` | `(1,3,3)` | gripper 局部向量到世界的旋转矩阵 |
| `jac` | `(1,6,6)` | 抓取点位姿对六个关节的雅可比 |
| `nodal_state` | `(1,V,6)` | V 个软体节点的 XYZ 与速度 XYZ |
| `nodal_kinematic_target` | `(1,V,4)` | 节点目标 XYZ 与自由/运动学标志 |
| `finger_surface_points()` | `(2,3)` | 两个夹指内侧代表点 |

这里的 Isaac Lab 3 beta 路径和项目运动学使用 **XYZW** 四元数。不要把其他版本的 WXYZ 教程直接套进来。单位四元数是 `(0,0,0,1)`。

`ball_center()` 是节点位置算术平均，并非严格质量加权质心。节点分布不均时二者有差异；代码中所有球心误差沿用同一估计，避免混用定义。

## 3. 文件开头与 Python 常用语法

- 三引号是文档字符串；放在文件或函数开头会成为 `__doc__`，不是可执行控制逻辑。
- `from __future__ import annotations` 延迟求值类型注解。`x: torch.Tensor`、`-> int` 给编辑器和读者信息，不会自动检查运行时类型。
- `argparse` 处理命令行；`math` 提供三角函数和向上取整；`random.Random(seed)` 是独立随机数生成器；`time.perf_counter()` 测真实耗时；`Path` 拼接文件路径。
- `AppLauncher` 启动 Kit，之后才导入依赖仿真运行环境的模块。普通系统 Python 可以检查语法，但不能据此断言 Isaac Lab 模块能正常加载。
- `@configclass` 让配置类遵循 Isaac Lab 配置机制。`class DeformableNewtonCfg(NewtonCfg)` 表示继承，并增加 `model_cfg` 字段。
- `a if condition else b` 是条件表达式；`None` 表示未指定，区别于数值零。
- `*ROBOT_BASE_XY` 把二元组展开为两个实参；`rng.uniform(*RANGE)` 相当于 `uniform(RANGE[0], RANGE[1])`。
- 函数参数中的单独 `*` 表示后续参数只能按名字传，如 `stage="reset"`。
- `lambda: check_ball_in_gripper(robot, ball)` 是延迟调用的无参函数，用作每步回调。
- `global SCENE_STEP_COUNT` 修改模块变量；`nonlocal max_drift` 修改外层函数的局部变量。
- `for ... else` 的 `else` 在循环没有通过 `break` 退出时执行，适合表达“所有尝试耗尽”。
- `try/except/finally` 捕获错误并保证清理；`raise` 重新抛出错误，让终端获得失败退出码。
- `if __name__ == "__main__"` 只在直接执行脚本时运行入口。但本脚本的 AppLauncher 在模块顶层，不能把它当普通纯函数库随意 import。

PyTorch 语法：`[:, :6]` 取所有环境的前六关节；`[..., :3]` 保留前面所有维度、取最后一维前三项；`unsqueeze(0)` 增加 batch 维；`squeeze(-1)` 去掉末尾大小为 1 的维度；`stack` 新增维度，`cat` 沿已有维度拼接；`@` 是矩阵乘法，`*` 通常为逐元素乘法；`clone()` 创建独立存储，防止修改观测缓存；`expand()` 使用广播视图，不能把它理解为多份独立内存；`.item()` 取单个 Python 标量；`.cpu().tolist()` 用于日志，会产生设备同步开销；`.torch` 是当前 Newton 数据接口提供的 Torch 视图。

## 4. 配置变量字典

### 4.1 几何与采样

| 名称 | 意义与影响 |
| --- | --- |
| `PROJECT_ROOT` | 当前文件的上两级目录，即工程根；`parents[1]` 是从 0 开始索引 |
| `YAM_USD` | 已转换机器人资产路径，改变它可能使运动学与资产不再匹配 |
| `TABLE_SIZE` | 桌体 X/Y/Z 尺寸 `(1.50,1.50,0.08)` |
| `TABLE_CENTER` | 桌体几何中心 `(0,0,0)`；这是一块台面，底半部位于地平面以下 |
| `TABLE_TOP_Z` | `center_z + thickness/2 = 0.04`，所有支撑高度的基准 |
| `ROBOT_BASE_XY` | `TABLE_CENTER[:2]`，生成机器人与采样共同使用 |
| `OBJECT_RADIUS_RANGE` | `(0.20,0.30)`，环形候选区内外半径，不是完整碰撞验证 |
| `OBJECT_BEARING_RANGE` | `(-2.40,3.00)`，约 -137.5°～171.9°；避开当前径向姿态的首关节限位 |
| `MIN_OBJECT_PLANAR_DISTANCE` | 球与方块中心最少间距 0.18 m |
| `SAMPLING_MAX_ATTEMPTS` | 最大重采样次数 100，参数冲突时有限时间内报错 |
| `MAX_TRANSFER_BEARING_DELTA` | 起终点最大方位差 π rad；超过时重采样，避免 joint1 绕限位缺口走长弧 |
| `TRANSFER_BALL_ALIGNMENT_TOLERANCE` | 开始下放前球心 XY 对正阈值 0.002 m |
| `BLOCK_SIZE` | 方块尺寸 `(0.07,0.07,0.05)` |
| `BLOCK_RESET_CLEARANCE` | 初始化方块底面高出台面 0.0005 m，避免初始深穿透 |

### 4.2 材料、接触与验收

| 名称 | 意义 |
| --- | --- |
| `RIGID_STATIC_FRICTION` / `RIGID_DYNAMIC_FRICTION` | 刚体表面的静/动摩擦系数 0.6/0.5 |
| `RIGID_RESTITUTION` | 恢复系数 0，减少碰撞反弹 |
| `BLOCK_SETTLE_LINEAR_SPEED` / `BLOCK_SETTLE_ANGULAR_SPEED` | 方块稳定阈值 0.0005 m/s、0.01 rad/s |
| `BLOCK_SETTLE_CLEARANCE` | 方块底面支撑高度容差 0.001 m |
| `BALL_RADIUS` / `BALL_DENSITY` | 球几何半径 0.04 m、密度 500 kg/m³ |
| `BALL_YOUNGS_MODULUS` / `BALL_POISSONS_RATIO` | 杨氏模量 E=50000 Pa、泊松比 ν=0.35 |
| `BALL_PARTICLE_RADIUS` | 节点接触包络半径 0.006 m；并非球的几何半径 |
| `BALL_RESET_CLEARANCE` | 初始化软体最低接触包络与台面的额外间距 |
| `BALL_INTERNAL_DAMPING` | 内部材料阻尼 0.01，不能等同接触阻尼 |
| `BALL_SETTLE_WINDOW_DRIFT` | 稳定窗口内 XY 包围盒对角线阈值 0.00025 m |
| `BALL_SETTLE_ROOT_SPEED` | 节点平均速度模长阈值 0.001 m/s |
| `BALL_SETTLE_MAX_NODAL_SPEED` | 任一节点最大速度阈值 0.01 m/s |
| `BALL_SETTLE_CONTACT_CLEARANCE` | 球支撑接触包络高度容差 0.0015 m |
| `SCENE_SETTLE_REQUIRED_STEPS` | 连续 30 主步通过稳定判据 |
| `SCENE_SETTLE_TIMEOUT_STEPS` | 一般沉降上限 480 主步 |
| `SCENE_DIAGNOSTIC_STEPS` | 打印中间诊断的步数集合 |
| `GRASP_READY_REQUIRED_STEPS` | 抓取前只需连续 12 步通过专用判据 |
| `GRASP_READY_ROOT_SPEED` | 抓取前均速上限 0.0015 m/s |
| `GRASP_READY_MAX_NODAL_SPEED` | 抓取前节点最大速度 0.003 m/s |
| `GRASP_READY_WINDOW_DRIFT` | 抓取前短窗口漂移阈值 0.0005 m |
| `PLACE_CONTACT_CLEARANCE` | 释放前允许的接触包络间隙 0.0005 m |
| `PLACE_POSITION_TOLERANCE` | 放置球心相对目标 XY 误差上限 0.005 m |
| `PLACE_RELEASE_DRIFT` | 松爪到验收相对释放起点最大漂移 0.003 m |
| `BLOCK_PLACEMENT_MAX_DISPLACEMENT` | 放置期间方块位置变化上限 0.004 m，实际使用三维位移模长 |
| `MIN_LIFT_RATIO` | 球实际升高至少达到命令升高量的 85% |

这些稳定阈值是验收规则，不会直接改变物体运动。增大阈值使验收更宽松，不等于物理更稳定。

### 4.3 夹爪与运动

| 名称 | 意义 |
| --- | --- |
| `GRASP_POINT_LOCAL` | 抓取控制点相对 gripper 原点 `(0,0,-0.125)` m |
| `GRASP_ROTATION_W` | 径向姿态参考 `diag(-1,-1,1)`；结合负 Z 抓取偏移实现自上而下接近 |
| `LEFT_PAD_LOCAL_POS` / `RIGHT_PAD_LOCAL_POS` | 夹指内侧代表点的局部坐标，属于 tip link；不是添加的碰撞板 |
| `LEFT_TIP_PATH` / `RIGHT_TIP_PATH` | USD 中左右夹指路径，运行时修改网格碰撞近似 |
| `GRIPPER_OPEN_POSITION` | 两个夹指关节各取 -0.04695 m，负方向表示张开 |
| `GRIPPER_CLOSED_GAP` | 关节零位时两代表点的标定间隙 |
| `GRIPPER_OPEN_GAP` | `closed_gap - 2*open_position` |
| `NOMINAL_GRIP_STRESS` | 选压缩比例的经验应力尺度 5000 Pa，非实时测量力 |
| `BALL_EFFECTIVE_MODULUS` | `E/(1-ν²)`，用于经验压缩估计 |
| `BALL_COMPRESSION_RATIO` | `min(0.12, nominal_stress/effective_modulus)` |
| `GRIPPER_TARGET_GAP` | `2*radius*(1-compression_ratio)` |
| `GRIPPER_GRASP_POSITION` | `-(target_gap-closed_gap)/2`，由双指间距换算单指位置 |
| `GRIPPER_COMMAND_SPEED` | 空载开合单指速度 0.04 m/s |
| `GRIPPER_CONTACT_SPEED` | 实际闭合单指目标速度 0.03 m/s |
| `PLACE_RELEASE_SPEED` | 释放单指速度 0.012 m/s |
| `ARM_CARTESIAN_SPEED` | 普通末端平移参考峰值速度 0.35 m/s |
| `ARM_JOINT_SPEED` | 初始关节插值参考峰值速度 3 rad/s |
| `ARM_ORIENTATION_SPEED` | 末端姿态参考峰值角速度 1.2 rad/s，与单关节角速度不同 |
| `MOTION_SETTLE_STEPS` | 轨迹结束后的最大闭环收敛步数 120 |
| `DESCEND_SPEED` | 最后接近球的平移速度 0.25 m/s |
| `LIFT_SPEED` / `PLACE_APPROACH_SPEED` | 抬升/下放粗接近速度，跟随普通平移速度 |
| `TRANSFER_SPEED` | `min(ARM_CARTESIAN_SPEED,0.30)`，持球转向速度上限 |
| `TRANSFER_DLS_GAIN` | 连续搬运 DLS 伺服增益 3.0，用于补偿驱动器跟踪滞后 |
| `TRANSFER_MAX_JOINT_COMMAND_STEP` | 搬运时驱动目标每步最多领先实际关节 0.04 rad |
| `PLACE_SPEED` | 最后接触下放速度 0.03 m/s |
| `QUICK_GRASP_STABLE_STEPS` / `QUICK_GRASP_TIMEOUT_STEPS` | 快速检查要求连续 12 步、最多 120 步 |
| `SCENE_STEP_COUNT` | 实际推进主步的累加器，用于仿真时间统计 |
| `APPROACH_IK_SEED` | 离线 IK 迭代初值，不是几何参数，也不是固定首关节方位偏差 |

经验压缩公式只是给出初始闭爪目标。真实球形弹性接触、夹指几何、摩擦、接触刚度和驱动力共同决定实际间隙，不能由 `gap=diameter*(1-strain)` 严格预测。PD 驱动受力平衡后，实际关节可停在比目标更张开的位置。

## 5. CLI 与仿真配置

`parser` 是解析器；`args_cli` 是解析结果；`app_launcher` 负责启动；`simulation_app` 是应用句柄。

| 参数 | 默认/作用 |
| --- | --- |
| `--episodes` | 10；完整实验次数 |
| `--steps-per-episode` | 30；复位后至少等待主步数，不是动作总时长 |
| `--seed` | 7；物体坐标随机序列种子 |
| `--pick-lift` | 启用抓取和抬升，`store_true` 表示出现此参数才为 True |
| `--place-on-block` / `--put-back` | 放方块/放原地，互斥且依赖抓取 |
| `--stop-after-hold` | 闭爪检查后停止，不能宣称抬升成功 |
| `--gripper-check` | 单轮空载夹爪开合测试 |
| `--motion-steps` | 240 为默认时间倍率，480 约慢一倍；当前要求不少于 240 |
| `--lift-height` | 0.03 m；放方块模式会按净空要求增加实际命令升高 |
| `--disable-cuda-graph` | 调试时关闭 Newton 执行图 |
| `--keep-open` | 结束后保持 GUI 和物理推进 |
| `--visualizer kit/none` | AppLauncher 参数，选择 GUI/无 GUI |
| `--device`、`--headless` 等 | 由 `AppLauncher.add_app_launcher_args` 添加，具体值以本地版本帮助为准 |

`--seed` 只保证相同采样算法下坐标序列重现。Isaac Lab 在 `sim/schemas/schemas.py` 自动调用 `pytetwild.tetrahedralize`，没有把这里的 seed 传入；历史相同位置种子的运行出现过不同节点数。因此它不保证网格相同或物理逐位确定性。

## 6. 创建场景的三个函数

### `make_sim()`

无输入，返回 `SimulationContext`。`solver_cfg` 是刚柔耦合求解器配置：刚体侧 `MJWarpSolverCfg`、软体侧 `VBDSolverCfg`，双向耦合使软球也能反推夹爪和方块。

`dt=1/120` 是主步；`num_substeps=10` 把主步进一步细分。理想均匀细分时子步时长为 `1/1200 s`。VBD 的 `iterations=20` 是求解迭代次数，不能当作主步数。CUDA Graph 优化执行开销，不改变脚本动作含义。

`njmax`/`nconmax` 是约束与接触容量；`ls_iterations` 是线搜索预算；`ls_parallel` 控制相应执行方式；`impratio` 调整摩擦/法向约束关系；`cone` 选择摩擦锥近似；`integrator` 选择积分算法；`ccd_iterations` 是碰撞算法迭代预算。它们不是抓取成功率开关。

`soft_contact_ke`、`soft_contact_kd`、`soft_contact_mu` 是软接触刚度、阻尼与摩擦配置。线性弹簧阻尼近似可理解为 `Fn≈ke*penetration+kd*closing_speed`，切向摩擦满足库仑约束 `|Ft|≤μFn`；具体求解还受 Newton 接触实现影响。本任务软接触摩擦设置偏高，不能由仿真成功直接推断真实材料也抓得住。

### `rigid_surface_material()`

无输入，返回 `NewtonMaterialPropertiesCfg`，集中复用桌面和方块表面材料，避免多处写值后不一致。材料对象是配置；真正接触响应由后端组合双方参数求解。

### `spawn_scene()`

无输入，返回 `(robot, block, ball)`。

`light_cfg` 是环境灯；`table_cfg` 是静态立方体台面，具有碰撞而没有动态刚体配置；`robot_cfg` 是机械臂初始位置、关节角、驱动器配置。`joint_pos` 的键是资产中的关节名，前六旋转、后两平移。

`ImplicitActuatorCfg` 的目标可以用 PD 理解：`τ≈Kp(q_target-q)-Kd*q_dot`，并受力矩/速度限制。`stiffness`、`damping` 是驱动参数，不是球的杨氏模量。

`stage` 是 USD 场景；`tip_path` 指向夹指；`prim` 是遍历到的场景元素。`make_uninstanceable` 允许独立覆盖属性；`convexDecomposition` 把原夹指碰撞分解为凸块，更好保留凹形，未添加可见夹指板，也未修改磁盘原始网格。

`block` 为动态方块、质量 0.12 kg；`ball` 是体积软体球。球材料由 E、ν 转成 Lamé 参数：

$$\mu=\frac{E}{2(1+\nu)},\qquad\lambda=\frac{E\nu}{(1+\nu)(1-2\nu)}.$$

小应变各向同性线弹性满足 `σ=2με+λ tr(ε)I`，这是参数换算的理论基础，不意味着后端只使用小变形线性模型。`ν→0.5` 时 λ 急剧变大，会增加数值求解难度。

## 7. 随机、复位与基础观测

### `sample_object_positions(rng)` 和内部 `sample_on_annulus(z)`

输入 `rng` 是随机数生成器；输出方块和球的世界 XYZ 元组。内部函数输入支撑中心高度 `z`，生成 `radius=r`、`bearing=θ`，计算 `x=base_x+r cosθ`、`y=base_y+r sinθ`。

`block_pos`、`ball_pos` 为候选点；`planar_distance` 为中心水平欧氏距离；`block_margin` 为方块半对角线加余量；`ball_margin` 为球半径、接触半径及额外余量；`inside_table` 通过 `all(...)` 确认两个物体在 X/Y 方向都留在台面内；`pos`、`margin`、`axis` 是生成器表达式的临时变量。

当前半径直接均匀采样，因此不是面积均匀分布。若需要环带内面积均匀，应用 `r=sqrt(r_min²+u*(r_max²-r_min²))`；这是因为极坐标面积元为 `r dr dθ`。不要在学习时把两种分布混为一谈。

拒绝不合格组合意味着最终两个物体位置不是完全统计独立。除间距和桌面边界外，
还要求 `abs(block_bearing-ball_bearing) <= MAX_TRANSFER_BEARING_DELTA`。这是因为 joint1
不是连续旋转关节；超过 π 的组合会迫使机械臂沿大于 180° 的长弧绕开限位缺口。
几何过滤也不代表全路径无碰撞。角度范围验证防止重新放入已经证实不可达的末端姿态样本。

### `reset_episode(robot, block, ball, block_pos, ball_pos, ball_template=None)`

返回实际重置后的节点平均中心。`joint_pos` 是默认关节姿态副本，`joint_vel` 清零；`block_pose` 保存位置和单位姿态，同时清零方块根速度。

`nodal_state` 取默认网格或之前沉降的自由形状；`positions` 是其中的位置视图；`current_center` 是原平均中心；`desired_xy` 是目标 XY；所有节点加同一个水平平移，保持相对形状。

`current_lowest_node_z` 取最小节点高度；`desired_lowest_node_z=table_top+particle_radius+reset_clearance`。两者之差加到所有节点 Z，令接触包络落在台面上方。这里 `ball_pos[2]` 不直接控制最终高度，所以返回的 `placed_center` 比名义 `table_top+radius` 更可靠。

`kinematic_target[...,3]=1` 在本后端表示自由节点。前面写目标 XYZ 是清理初始化状态，不是抓取中绑定节点。最后各对象 `.reset()` 重置内部缓存。`ball_template` 是自由沉降网格，不能取正在受夹爪挤压的状态作为下次原形。

### `block_support_metrics(block)`

返回 `(clearance, linear_velocity, angular_velocity, on_table_xy)`。`pose` 为根 link 位姿；`corners` 用 x/y/z 各取 ±1 组合八个局部角点；`corners_w=R*corners+p`；`clearance=min(corners_z)-table_top`。

`half_x/half_y` 从台面半尺寸减去方块半尺寸，检查中心是否在允许区。该 XY 检查按轴对齐尺寸近似，不能取代旋转方块完整足迹碰撞检查。Z 使用实际旋转后的角点。

### `check_block_support(block, *, stage)`

`stage` 是日志标签；`pose`、`clearance`、`linear_velocity`、`angular_velocity`、`on_table_xy` 是当前指标；`linear_speed`、`angular_speed` 为向量模长。打印后在离桌或深穿透时抛错。它是异常筛查，不是严格稳定判据。

### `block_state_is_settled(clearance, linear_velocity, angular_velocity, on_table_xy)`

输入上面的观测，返回布尔值。要求在台面范围、底面间隙小、线速度小、角速度小。速度模长按 `sqrt(sum(v_i²))` 求，`v` 为生成器中的分量。

### `ball_center(ball)`

返回 `(1,3)` 节点平均：`c=(1/V)Σp_i`。`mean(dim=1)` 消去节点维，不消去环境维。

### `ball_settle_metrics(ball)`

返回 `(center, contact_clearance, root_velocity, max_nodal_speed, on_table_xy)`。`nodal_state` 取第一个环境；`positions/velocities` 分开 XYZ 与速度；`root_velocity` 是节点平均速度；`max_nodal_speed=max_i ||v_i||` 排除内部仍振荡的情况。

`contact_clearance=min_i(z_i)-particle_radius-table_top`。`half_x/half_y` 留球半径余量，但仍是球形尺寸近似，严重形变由其他检查拦截。

### `check_ball_support(ball, *, stage, reset_center)`

打印上述数据。`root_speed` 是平均速度模；`xy_drift` 是相对 `reset_center` 的水平距离，整个搬运结束后这个值很大是正常的，不能当释放漂移。离台或深穿透才报异常；球在方块上方时打印的台面净空约为方块高度，不是悬空故障。

## 8. 仿真推进和稳定判定

### `step_scene(sim, robot, block, ball, joint_target)`

发送 8 个关节目标，写待提交数据，执行一次 `sim.step`，再用 `dt` 更新对象缓存。`SCENE_STEP_COUNT` 累加主步。`name/value` 是逐项有限性检查的标签和张量；`invalid` 给出最早几个 NaN/Inf 索引。张量有限不代表物理合理，只是必要条件。

物理求解发生在 `sim.step`，`robot.update(dt)` 是更新读出的状态，不能当成又模拟了一步。渲染开关受 `headless` 影响；GUI 的帧率不能直接当物理频率。

### `wait_until_scene_settled(...)`

必需输入场景对象、保持的 `joint_target`、日志 `episode`、最少步数 `minimum_steps`、`ball_reset_center`；可选 `on_step` 做额外验收，`support_height` 把球支撑参考改为方块顶面；其余参数是超时、连续步数和速度/漂移阈值。返回实际推进步数。

`stable_steps` 记录连续合格步；`stable_centers` 保存短窗口 XY；`timeout_steps` 保证至少有一段验收窗口；`step_index` 从 0 开始，`step_count` 从 1 开始。

`bc/blv/bav/bot` 依次是方块间隙、线速度、角速度、在台面标志；`ball_now/bclear/bon` 是球中心、间隙、范围标志；`ball_velocity/nodal_speed` 是平均速度与最大节点速度。有 `support_height` 时，从台面间隙中减去方块顶面高出台面的量。

`ball_stable` 合并高度、速度等条件。所有条件合格才向窗口加入中心。`drift=sqrt((max x-min x)²+(max y-min y)²)` 是包围盒对角线，对窗口中任意两点距离给出上界，不是累计路径长度。超过阈值重开窗口；否则连续计数增长。达标返回，耗尽上限抛 `SCENE_SETTLE_TIMEOUT`。

因此“多等一会”只可能帮助本来会沉降的系统；持续滑移不会因等待自动变成正确。`on_step` 可在等待中发现漂移并及时停止。

## 9. 抓取几何和运动学

### `finger_surface_points(robot)`

`tip_indices` 在 body 名称中找左右 tip；`tip_poses_w` 是它们位姿；`local_centers` 是标定局部点。返回 `p_world=p_tip+R_tip*p_local`。两个点只是夹指代表位置，不是完整接触面、力传感器或碰撞检测器。

### `grasp_geometry_metrics(robot, ball)`

`pad_centers` 为两点；`center_delta` 是两点差；`center_distance=||delta||` 是估计间隙；`grasp_axis=delta/max(distance,1e-8)` 是夹持轴；`node_projections=p_i·axis` 是节点投影。返回间隙和球投影宽度 `max(projection)-min(projection)`。投影会随夹爪转动自适应。

### `grasp_point_kinematics(root_pose_w, arm_pos)`

调用基础 FK 得到 gripper 原点位置 `gripper_pos_w`、旋转 `gripper_rot_w` 和 `geometric_jacobian_w`。`local_offset` 为控制点偏移，`offset_w=R*r_local`，`grasp_pos_w=p_origin+offset_w`。

刚体点速度关系为 `v_point=v_origin+ω×r`。因此每个关节的线速度雅可比列都要增加 `Jω_i×r`。`angular_columns` 转置后让列变成可批量叉乘的向量；`point_velocity_columns` 是偏移贡献；`point_jacobian_w` 是新的前三行。最后拼回角速度雅可比。返回 `(gripper_pos_w, gripper_rot_w, grasp_pos_w, grasp_jacobian)`。

若只改控制点位置而不改雅可比，IK 的线性化模型会控制错点，表现为位置/姿态耦合误差。

## 10. 运动控制函数

### `move_grasp_point(...)`

输入场景对象、`target_pos_w`、`gripper_target`；`maintain_orientation` 选择位姿或仅位置 IK；`target_rotation_w=None` 使用参考姿态；`speed` 覆盖平移速度；`smooth` 选择平滑时间曲线；`on_step` 是每步监测回调。返回位置误差和首次 FK 对齐误差，超时直接报错。

| 局部变量 | 意义 |
| --- | --- |
| `ee_body_idx` / `limits` | gripper body 索引 / 六关节限位 |
| `initial_fk_error` / `simulated_pos_w` | 首步解析 FK 与仿真 gripper 原点的偏差 / 仿真原点 |
| `target_rot_w` | 真正采用的目标旋转矩阵 |
| `start_rot/start_pos` | 运动开始时的真实旋转和抓取点位置 |
| `start_quat/target_quat` | 起终点单位四元数，形状 `(4,)` |
| `rotation_distance` | 起终姿态最短转角 `2 acos(|q0·q1|)` |
| `linear_duration/angular_duration` | 平移距离/平移速度、转角/角速度 |
| `time_scale/travel_steps` | 命令时间倍率 / 轨迹主步数 |
| `step/phase/alpha` | 步索引、归一化进度、时间插值权重 |
| `waypoint/waypoint_rotation` | 当前应追踪的位置与旋转 |
| `root_pose_w/arm_pos` | 本物理步的真实基座位姿与关节角 |
| `gripper_pos_w/gripper_rot_w/grasp_pos_w/jac` | 当前 FK 和抓取点雅可比 |
| `joint_delta/arm_target/full_target` | DLS 增量、限幅后的六关节目标、带夹爪的八关节目标 |
| `stable_steps/pos_ok/rot_ok/stopped` | 连续稳定计数及三个终止条件 |
| `final_rot/current_pos` | 最新末端实际姿态/位置 |
| `position_error/rotation_error` | 超时日志的残差 |
| `final_grasp_pos_w` | 正常返回时的抓取点位置 |

轨迹时间取 `T=max(distance/v, angle/ω)*time_scale`，平滑时再乘 1.5。纯旋转也有足够时间。

位置轨迹 `p(s)=p0+α(s)(p1-p0)`，平滑函数 `α=3s²-2s³`。导数 `α'=6s(1-s)` 的最大值是 1.5，所以若不延长时间，实际参考峰速会比设定值高 50%。起终点速度为零，但加速度仍有跳变，它不等同五次最小 jerk 曲线。

姿态采用 SLERP：令 `Ω=acos(q0·q1)`，则 `q(α)=sin((1-α)Ω)/sinΩ*q0+sin(αΩ)/sinΩ*q1`。先选择四元数符号以走较短弧。普通旋转矩阵逐元素线性插值可能不再正交，因此这里用四元数球面插值。当前工具函数可能原地翻转终点四元数符号，传 `clone()` 保护缓存。

每步用最新观测计算 DLS，而不是一次算出目标后盲目播放。`joint_delta` 限 ±0.02 rad 后再限制机械关节范围；驱动器仍决定真实跟随动态。参考峰速不是对真实接触加速度的硬保证。

轨迹结束后最多追加 `MOTION_SETTLE_STEPS`。要求连续三步位置误差 ≤2 mm、矩阵姿态误差 ≤0.02、关节速度 ≤0.05 rad/s。`for ... else` 确保超时不会被当成正常到达后继续下放。

### `radial_grasp_rotation(robot, target_pos_w)`

`root_xy` 是基座位置，`bearing=atan2(dy,dx)`。`cosine/sine/zero/one` 组出 `yaw_rotation=Rz(bearing)`；`reference` 是原参考矩阵；返回 `Rz(bearing)@reference`。左乘表示绕世界 Z 转，局部负 Z 抓取偏移仍指向下方。本实现假定基座安装旋转为单位姿态。

### `joint1_compatible_bearing(robot, target_pos_w)`

取 `bearing=atan2(...)`，建立 `candidates=θ+[-2π,0,2π]`，用实际首关节 `limits` 和 0.10 rad 余量计算 `valid`。`selected` 是有效项的索引，`gather` 提取结果；没有合法值抛 `UNREACHABLE_BEARING`。

这仅是径向方位候选筛查。首关节真实值与方位还存在小机构偏差，其他关节可达性仍需求 IK。旧版把 `APPROACH_IK_SEED[0]` 加入限位判断是错误的：初值是数值求解起点，不是机构固有几何偏置。

### `move_ball_on_continuous_polar_path(...)`

这是球从抓取位置搬到方块上方的连续路径控制器。`start_ball_center` 是实际软球球心，
`start_bearing/destination_bearing` 是起终方位，`start_radius/destination_radius` 是起终半径。
路径采用一个全局进度：

$$u\in[0,1],\qquad \alpha(u)=3u^2-2u^3,$$
$$\theta(u)=\theta_0+\alpha(u)(\theta_1-\theta_0),\qquad
r(u)=r_0+\alpha(u)(r_1-r_0),$$
$$p_{xy}(u)=p_{base}+r(u)[\cos\theta(u),\sin\theta(u)].$$

旧实现将方位拆成多个路点，每个路点单独调用 `move_grasp_point(..., smooth=True)`。
每段的 `α'(0)=α'(1)=0`，所以每到一个路点机械臂都会减速到接近零，再重新加速。
新实现只有一个 `u`，因此只在整条路径的开头和结尾减速。

`path_length` 用径向变化和平均半径弧长估算：
`hypot(Δr, r_mean*abs(Δθ))`。平移时长与转向时长分别为 `L/v`、`abs(Δθ)/ω`，
取两者最大值并乘 1.5，以限制 cubic smoothstep 的峰值速度。

每个物理步都用实际球心修正夹爪 XY：
`target_grasp_xy=current_grasp_xy+desired_ball_xy-actual_ball_xy`。这不是把球节点绑到夹爪，
而是机械臂通过真实接触夹持球，同时对软球在夹爪内的毫米级弹性偏移进行反馈补偿。
夹爪高度在搬运期间保持不变，姿态随当前连续方位径向旋转。到达后要求球心 XY 误差
不超过 2 mm、姿态误差和关节速度同时合格，才进入下放阶段。

连续移动参考下，单纯命令 `q_target=q_actual+Δq_DLS` 会让 PD 驱动目标始终只领先很小距离，
执行器产生持续滞后。搬运控制因此使用
`q_target=q_actual+clamp(TRANSFER_DLS_GAIN*Δq_DLS, ±0.04)`；这只是提高目标跟踪增益，
并不会直接瞬移关节，真实速度和力矩仍受 USD 资产中的 actuator 限制。

### `solve_approach_joint_target(robot, target_pos_w, target_rotation_w, iterations=300)`

`arm_target` 从经验肘形初值开始，首关节改为目标安全方位；`limits/root_pose_w` 是限位和基座。每次计算 `current_rotation_w/grasp_pos_w/grasp_jacobian_w`，求 `joint_delta`，限 ±0.05 rad 并投影到关节范围。

`solved_rotation_w/solved_pos_w` 是最后 FK。返回六关节解、位置残差、矩阵旋转残差。它不推进物理、不检查整条路径碰撞。`iterations` 是数值迭代上限，增加它不能让物理不可达点变得可达。局部 DLS 也可能错过其他肘形的解。

### `execute_arm_trajectory(sim, robot, block, ball, target_arm_pos, gripper_target)`

`start_arm_pos` 取当前角度；`steps=max(30, int(1.5*max|Δq|/(dt*joint_speed)*scale)+1)`。每步 `phase/alpha` 用三次曲线生成 `full_target`。最慢关节决定共同时间，所有关节同步结束。

关节角平滑不等于末端直线，也不保证避碰。当前用于空载预抓取接近，后续阶段另查实际误差。

### `hold_grasp(sim, robot, block, ball, gripper_start, gripper_end, *, release=False, on_step=None)`

`target` 从上一次驱动目标复制，保留机械臂保持力；不能用受重力偏移的实际关节角反复替代保持目标。`speed` 根据空载检查/闭合/释放选择；`time_scale/steps` 用距离除速度决定时长。循环只修改后两个夹指目标并推进物理；`on_step` 可持续检查放置漂移。

## 11. 抓取检查与抓起流程

### `check_gripper_motion(sim, robot, block, ball)`

`gaps` 存闭合/张开的实测间距；`start/end` 依次表示从全开到零位、再从零位到全开；`target` 在终点保持 30 步；`gap` 由几何函数返回。要求闭合间距 ≤5 mm、张开 ≥85 mm。只验证开合，不说明带球夹持有效。

### `check_ball_in_gripper(robot, ball)`

`middle` 为两个代表点中点，球心到中点距离要求 ≤25 mm；`extent` 为节点世界轴对齐包围盒尺寸，过扁或过大报错。这是有限几何检查，不是法向力、摩擦裕度或全网格碰撞检查。

### `quick_grasp_check(..., center_before_lift=None, required_lift=0.0)`

`target` 固定驱动目标；`previous_relative/relative` 是球心相对夹指中点的位置；差分除 `dt` 得 `relative_speed`。这样机器人整体运动不被误算成球在夹爪内滑动。

`gap/width` 是间距和球投影宽；`finger_stopped` 检查夹指速度；`enveloped` 检查夹爪确已收拢且覆盖接触包络；`lift_ok` 默认 True，传入抬升前中心时再检查 `lowest` 离桌、`rise` 达到 `required_lift*MIN_LIFT_RATIO`。`stable/index` 是连续计数和尝试索引。闭合检查只能提供夹持迹象，离桌后的真实升高检查才提供抬升证据。

### `pick_and_lift_ball(sim, robot, block, ball, initial_ball_pos, lift_height, episode)`

1. `device/dtype` 确保新张量与观测一致；`initial_ball_target` 是球初始观测；`pregrasp_offset=(0,0,0.10)`；`pregrasp_target` 为球上方；`approach_rotation_w` 为径向姿态。
2. `pregrasp_joint_target/solved_position_error/solved_rotation_error` 是离线解与残差，通过才执行空载关节轨迹；`approach_hold` 保持该目标等球沉降。
3. `tracked_pregrasp_target/tracked_joint_target` 使用沉降后的球重求目标；`current_pregrasp` 判断是否需要微调；`actual_pregrasp_pos_w/pregrasp_error` 检查真实到位。
4. `gripper_index/fk_error` 对比解析与仿真原点；`gap/width/required_gap` 检查张开间隙覆盖球加两侧粒子半径及余量。
5. `above_ball` 位于球上 4.5 cm，随后 `grasp_target` 到球心；`grasp_error/lateral_error` 检查末端到位以及球是否被碰偏。
6. 闭合到 `GRIPPER_GRASP_POSITION`；`measured_gap/ball_width_after_grasp` 用于日志；执行快速夹持检查。
7. `center_before_lift/grasp_before_lift` 保存抬升前状态；`lift_target` 增加 Z。放方块模式用 `ball_bottom/block_top` 求足够净空，可能覆盖用户给出的较小 `lift_height`。
8. `lift_error` 为机械臂跟踪误差；抬升检查通过后用 `center_after_lift/actual_lift` 生成结果字典。`hold_only` 分支不计算抬升成功。

返回的 `commanded_lift_m` 是实际采用的抬升距离，不一定等于 CLI 的 0.03 m。不要把大约 0.11 m 的抬升误认为忽略参数。

## 12. 放置流程及四个闭包

### `put_ball_back(..., original_center, episode, *, on_block=False)`

`origin` 是目标球心参考；原地模式用 `original_center`，方块模式用 `initial_block_pos` 的 XY 和名义支撑高度。`destination_rotation_w` 面向目标；真正下放停止由软球最低节点包络决定，不能仅用名义球半径。

| 变量组 | 意义 |
| --- | --- |
| `target/current_rotation_w/error` | 当前阶段末端目标、上抬保持姿态、误差 |
| `alignment_error` | 连续搬运结束时的实际球心 XY 对正误差 |
| `correction/ball_alignment_error` | 最终球心闭环对正次数及实测误差 |
| `gap` | 下放循环里的支撑间隙；在释放检查段同名变量表示夹爪间距，需按局部上下文读 |
| `held_target/release_center/max_drift` | 释放前保持目标、释放起点、最大释放漂移 |
| `open_target/clearance_steps/width` | 全开保持目标、等实际张开所用步数、球投影宽 |
| `final_steps/result/marker` | 最后沉降步数、结果字典、成功日志标签 |

绕行由 `move_ball_on_continuous_polar_path()` 完成；它直接逐物理步生成极坐标曲线，
不再用多段直线弦近似，也不在中间路点停车。当前路径仍不是带完整机器人碰撞模型的
全局规划器，因此不能承诺任意随机布局都无碰撞。

转向后软球相对夹爪可能产生毫米级弹性偏移。最终用 `target_xy += desired_ball_xy-actual_ball_xy` 做球心闭环，最多三次，要求 ≤2 mm，给后续接触留出误差余量。

下放先快速到包络距支撑面约 15 mm，再最多八次小步接近。每次使用实际包络差，而非把球当刚体中心。接触后保持六步、记录释放起点、慢慢张开、确认实际间隙足够再撤离 10 cm。

### 内部 `support_height()`

原地返回台面高度。方块模式：`q` 为 XYZW 姿态，`up_z=1-2(qx²+qy²)` 是方块局部 Z 轴在世界 Z 的投影；`displacement` 是相对初始方块中心三维位移。过度倾斜或移位报错；否则返回方块中心 Z 加半高。该顶面模型适用于近水平方块。

### 内部 `grasp_position()`

调用 FK 返回真实抓取点位置的副本。每次获取最新状态，避免沿用陈旧名义位置。

### 内部 `clearance()`

返回 `min(node_z)-particle_radius-support_height()`。正值为空隙，接近零为接触，过负为穿透。这里的支撑参考随放置模式变化。

### 内部 `monitor_release()`

`center` 为当前球心；`error` 为相对放置目标的 XY 误差；`max_drift` 用 `nonlocal` 累计相对 `release_center` 的最大距离。方块模式还检查 `offset` 是否在 `half_size=block_half_size-5mm` 足迹内，以及包络是否深穿透。它在松爪、撤离和最终稳定等待中反复运行。

## 13. `main()` 与程序退出

校验参数组合后生成 `mode` 日志；创建 `sim/robot/block/ball`；`sim.reset()` 初始化后端；确认资产恰有八个关节。`rng` 跨 episode 使用同一实例，不能每轮用同一 seed 重新创建，否则每轮坐标相同。

`results` 收集复位与最终状态；`pick_lift_results` 收集抬升结果；`placement_results` 收集放置结果；`motion_timings` 收集耗时；`settled_ball_template` 只在第一轮自由沉降后记录。

第 2 轮及以后会先调用 `clear_contact_history_before_reset()`，再由 `reset_episode()` 写入
新的随机状态。原因是资产级 `block.reset()` 只清外力，`DeformableObject.reset()` 在当前实现
中是 no-op；它们不会清除 MJWarp/VBD solver 的接触约束和 warm-start 缓存。如果上一轮球
停在方块上，仅传送位姿会把旧接触冲量带入新场景，表现为方块在 Z 方向持续上下振荡。

清理函数先把方块和球移到空中两侧，把软球节点暂时设为 kinematic，清零全部速度，并推进
两个不渲染的无接触物理步；随后正式 reset 恢复球节点为自由状态。不能在每轮直接调用完整
`sim.reset()`：当前 Newton CUDA Graph 在运行中重新构建可能触发 illegal memory access。

循环变量 `episode` 为轮次；`block_pos/ball_pos/ball_reset_center/settle_steps` 记录采样与实际重置。`motion_start_step/motion_start_wall` 为计时起点；`settled_ball_pos` 为真正接近前的观测；`pick_lift_result/minimum_lift` 用于抬升验收；`timing` 中仿真时间为主步差乘 dt，真实时间来自 `perf_counter`。

`tensors/value` 做最后有限性检查；`measured_block/measured_ball/ball_xy_drift` 用于结果记录。最终各结果列表的长度决定成功计数，只有全部流程完成才输出 `YAM_DEFORMABLE_TASK_INTEGRATION_OK`。

`hold_target` 在 `--keep-open` 时保持最后关节目标，GUI 保持期间物理仍继续运行。当前最终验收是有限时间窗口，并没有在保持 GUI 的每一步重新执行完整放置验收，所以“最终通过”不是永久静止承诺。

入口 `except Exception as exc` 输出异常类型与内容后重新抛出，`finally` 关闭应用。不要只依据窗口是否出现或进程退出码判断成功，应同时看最终成功标记与 episode 汇总。

## 14. 运动学模块中的全部函数

### `_rpy_matrix(rpy)`

输入最后一维 `(roll,pitch,yaw)`；`cr/sr/cp/sp/cy/sy` 为对应余弦/正弦；返回 `Rz(yaw) Ry(pitch) Rx(roll)`。矩阵最右侧先作用于向量。变量来自 URDF 的固定 joint origin 旋转，与动态关节转角不同。

### `_axis_angle_matrix(axis, angle)`

`axis` 先归一化；`x/y/z` 为轴分量；`skew` 为叉乘矩阵，使 `skew*v=axis×v`；`eye` 为单位矩阵；`outer=axis*axisᵀ`；`c/s` 是转角余弦正弦。

Rodrigues 公式：`R=cosθ I+(1-cosθ)aaᵀ+sinθ[a]×`。沿旋转轴的分量不变，垂直轴的平面分量旋转。

### `_quat_xyzw_matrix(quat)`

先单位化四元数；`x,y,z,w` 从末维拆出；返回标准四元数旋转矩阵。例如 `Rzz=1-2(x²+y²)`，正是支撑倾斜检查所用项。归一化可减轻浮点误差，但零四元数不是合法输入。

### `forward_kinematics_and_jacobian(root_pose_w, arm_joint_pos)`

输入 `(N,7)` 基座位姿和 `(N,6)` 关节角。`batch_size/device/dtype` 控制批量、设备和类型；`origins_xyz/origins_rpy/axes` 来自 URDF 常量 `JOINT_ORIGINS_XYZ/JOINT_ORIGINS_RPY/JOINT_AXES`；`origin_rotations` 为固定旋转。

`transform` 从基座齐次变换开始。每个 `joint_id` 先乘 `origin_transform` 得 `joint_transform`，记录关节原点和 `axis_w`；再乘随 q 变化的 `motion_transform`。

$$T_{world,ee}=T_{world,base}\prod_{i=1}^6 T_{origin,i}T_{rotation,i}(q_i).$$

齐次变换 `T=[[R,p],[0,1]]` 同时表达位置与旋转。FK 返回最终 `gripper_pos_w/gripper_rot_w`。

`joint_positions_w/joint_axes_w` 保存世界系关节轴与原点。旋转关节雅可比列：`Jv_i=a_i×(p_ee-p_i)`、`Jω_i=a_i`。`linear_columns` 存线速度列；`jacobian` 将前三行线速度与后三行角速度拼接。它是当前构型的一阶局部模型：`[v;ω]=J*q_dot`。

### `damped_least_squares_position_step(...)`

输入 `current_pos_w/target_pos_w/position_jacobian_w` 与阻尼 `damping`。`error=target-current`；`jacobian_t=Jᵀ`；`regularizer=λ²I`。

求解 `min_Δq ||JΔq-e||²+λ²||Δq||²`，得到 `Δq=Jᵀ(JJᵀ+λ²I)⁻¹e`。程序用 `torch.linalg.solve` 解线性系统而不显式求逆，数值和计算上更合适。阻尼让接近奇异构型时增量有限，但阻尼过大也会降低收敛速度。

### `damped_least_squares_pose_step(...)`

增加 `current_rot_w/target_rot_w/geometric_jacobian_w`。`position_error` 为位置差；`orientation_error=0.5*Σ(current_column_i×target_column_i)`；`pose_error` 拼成六维，然后用同样的 DLS 公式。

该姿态误差等价于旋转误差矩阵反对称部分的向量，约为 `sinθ*a`；小角度时近似 `θ*a`。它在接近 180° 时会退化，不能当全局无奇异的旋转对数。当前逐段姿态平滑和径向初值减少大角度跳变，但并没有让局部 IK 成为全局规划器。

位置误差单位 m、角误差近似 rad，当前直接拼接等价于采用隐含权重；更一般的控制器应显式设计位置/姿态权重。不要把这个简化当成唯一正确的 IK 写法。

## 15. 本轮完善、验证与调试顺序

确认的问题与改动：

1. 旧方位上限 3.54 rad 在径向竖直姿态下不可达；离线边界扫描位置误差约 7.6～11.5 cm。修正为有余量的 `[-2.40,3.00]`，移除把数值初值当几何偏置的展开规则。
2. 原轨迹只平滑位置，姿态直接跳到下一段终点；现在位置和 SLERP 姿态一起插值，时间取平移与旋转要求的较大值。
3. 原运动循环耗尽后仍返回位置误差，部分调用方忽略它；现在必须连续通过到位/姿态/关节速度判据，否则抛出带目标与残差的 `MOTION_TIMEOUT`。
4. 增加物体完整尺寸余量的桌边检查和非法随机参数检查；统一要求 `--motion-steps>=240`，避免各阶段对小于 240 的倍率解释不一致。

用户所述的那一次失败尚无对应终端 traceback，不能声称已确定其唯一原因。修改前默认 seed 7 三轮复现全部成功；这不排除其他随机位置、不同体网格或 GUI 后续保持阶段出现失败。

遇到错误先找末尾 `GRASP_FAILURE`，再向上找最后的 `STATE`、`EPISODE_RANDOM_TARGETS`。`UNREACHABLE_BEARING` 是方位范围问题；预抓取 IK 残差过大是位姿求解问题；`MOTION_TIMEOUT` 是实际运动没收敛；`FAILURE OPEN` 是间隙不足；`QUICK_GRASP` 是夹持/抬升检查；`PLACE_DRIFT` 是松爪位置或漂移超限；`SCENE_SETTLE_TIMEOUT` 是连续稳定窗口没建立。

不要把所有失败都归为速度过快。目标不可达、姿态不连续、物体碰撞、接触参数、网格差异、稳定阈值都可能产生不同失败。一次更换多个参数后通过，也不足以证明其中某一项就是唯一根因。

## 16. 如何运行并保存日志

离线回归检查 [test_yam_pick_workspace.py](../scripts/test_yam_pick_workspace.py) 不启动 Kit，直接提取学习脚本的被测纯函数，验证 18 个边界 IK 组合、旧不可达方位拒绝，以及 100 组随机位置的间距与坐标序列复现：

```bash
../isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/env_isaaclab/bin/python \
  scripts/test_yam_pick_workspace.py
```

这是局部数值与采样回归，不替代接触动力学测试。三个测试方法均已通过。

在工程根目录使用对应 Isaac Lab 环境：

```bash
set -o pipefail
../isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
  -p scripts/yam_pick_ball.py --visualizer none \
  --pick-lift --place-on-block --episodes 3 \
  --steps-per-episode 60 --seed 7 2>&1 | tee /tmp/yam_pick_run.log
```

`2>&1` 合并 stderr 到 stdout；`tee` 同时显示并保存；`pipefail` 保留管道前端失败状态。GUI 改用 `--visualizer kit --keep-open`。默认放球任务 `Task: Place ball on block GUI` 的 F5 与 `Ctrl+Shift+B` 已统一运行学习版；其他诊断配置仍可能指向集成脚本，先看终端实际命令路径。

本手册是针对当前单实例教学程序的解释。若以后扩为批量并行环境、多机器人或任意安装朝向，需要重新检查 `[0]`、`.item()`、SLERP 批处理及世界/基座旋转假设。
