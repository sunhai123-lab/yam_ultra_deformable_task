# YAM 可变形球任务：运行、流程、验收与调试指南

本文档对应唯一主运行入口 [`scripts/yam_pick_ball.py`](../scripts/yam_pick_ball.py)。参数和函数定义请看 [`yam_pick_ball_reference.md`](yam_pick_ball_reference.md)。

## 1. 当前项目结构

任务运行只保留一个主入口：

```text
scripts/yam_pick_ball.py
```

运动学实现独立放在：

```text
source/yam_ultra_deformable_place/yam_ultra_deformable_place/yam_kinematics.py
```

资产转换工具 `scripts/convert_yam_ultra_2.py` 只用于 URDF→USD 重新生成，不是任务运行版本。正常调试、抓取、抬升、放置、reset-only 和 gripper-check 都由 `yam_pick_ball.py` 的命令行参数切换。

## 2. 环境与一次性安装

当前验证环境见仓库根目录 [`VERSIONS.md`](../VERSIONS.md)。项目包需要以 editable 模式安装，使主脚本能够导入 `yam_ultra_deformable_place.yam_kinematics`：

```bash
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/env_isaaclab/bin/python \
-m pip install -e source/yam_ultra_deformable_place
```

只有 vendored URDF 或 mesh 改变时才需要重新生成 USD：

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/convert_yam_ultra_2.py
```

## 3. 常用运行命令

### 3.1 只做 reset 和沉降检查

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer none \
--episodes 3 --steps-per-episode 60 --seed 7
```

### 3.2 只测试夹爪开合

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer kit --keep-open \
--gripper-check --episodes 1 --steps-per-episode 60 --seed 7
```

### 3.3 抓取并抬升

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer none \
--pick-lift --episodes 1 --steps-per-episode 60 --lift-height 0.03 --seed 7
```

### 3.4 抓起后放回原位置

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer none \
--pick-lift --put-back --episodes 1 --steps-per-episode 60 --lift-height 0.03 --seed 7
```

### 3.5 抓起后放到方块上

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/yam_pick_ball.py --visualizer kit --keep-open \
--pick-lift --place-on-block --episodes 3 --steps-per-episode 60 --lift-height 0.03 --seed 7
```

`--motion-steps 240` 是默认时间倍率，可显式传入。当前主程序不允许小于 240；想提高速度应优先调整源码中的速度常量，而不是把它减到 120。

## 4. 整体状态机

```text
AppLauncher
  ↓
make_sim()
  ↓
spawn_scene()
  ↓
sim.reset()
  ↓
EPISODE_START
  ↓
sample_object_positions()
  ↓
reset_episode()
  ↓
wait_until_scene_settled()
  ↓
[RESET_ONLY 到此即可]
  ↓
pick_and_lift_ball()
  ├─ PRE_GRASP
  ├─ DESCEND
  ├─ CLOSE_GRIPPER
  ├─ QUICK_GRASP_CHECK
  └─ LIFT
  ↓
[可选]
put_ball_back(on_block=False/True)
  ├─ RAISE_FOR_TRANSFER（需要时）
  ├─ TRANSFER / RETURN_ALIGN
  ├─ BALL_ALIGNMENT_CORRECTION（需要时）
  ├─ LOWER_TO_TABLE / LOWER_TO_BLOCK
  ├─ RELEASE
  ├─ RETREAT
  └─ final settle
  ↓
EPISODE_COMPLETE
  ↓
下一轮 / 汇总
```

## 5. 为什么当前采用桌心+环形采样

机器人 base 的 XY 与 `TABLE_CENTER[:2]` 绑定。方块和球独立在机器人周围的环形区域采样，而不是固定分布在左右两个矩形区域。

这样可以学习和测试：

- 世界坐标与机器人基座坐标的关系；
- joint1 面向不同方位时的 IK 初值；
- 末端抓取姿态随方位旋转；
- 物体位于机器人不同侧时的搬运轨迹；
- 随机范围扩大后“几何上在桌面内”与“机械臂完整 pose 可达”之间的区别。

采样只是候选生成。真正可达仍必须通过 `solve_approach_joint_target()` 的 IK 误差和后续真实运动误差检查。

## 6. 为什么搬运使用连续极坐标路径

如果球和方块分处机器人两侧，世界 XY 直线可能穿过机器人 base。当前 `move_ball_on_continuous_polar_path()` 在机器人中心周围插值半径 `r` 和方位 `theta`：

```text
alpha(u) = 3u² - 2u³
r(u)     = r0 + alpha(r1-r0)
theta(u) = theta0 + alpha(theta1-theta0)
```

整条路径只使用一次 smoothstep，因此只有整段运动的起点和终点速度降到零。旧式多 waypoint 方案如果每一小段都做 smoothstep，会在每个中间点不断“慢—快—慢”。

搬运不是只跟踪夹爪参考点；代码持续读取实际软球球心，并用球心误差修正夹爪目标，使最终对正依据物体真实位置而不是只看机械臂 IK。

## 7. 为什么姿态要随目标方位变化

固定一个世界抓取旋转只适用于目标都位于机器人同一侧。当前 `radial_grasp_rotation()` 根据目标相对 base 的 bearing 绕世界 Z 轴旋转参考抓取姿态，同时保持垂直下抓。

`joint1_compatible_bearing()` 再从 `theta + k*2π` 的候选中选择一个落入 joint1 实际限位安全区的等价角，作为 IK 初值。这样 base joint 先“面向”目标，其余关节不需要用腕部扭曲去补偿很大的水平朝向差。

## 8. 预抓取为什么先离线求 IK，再走 joint-space

第一次从 reset 姿态移动到球上方时，代码先用 `solve_approach_joint_target()` 在数学层面反复做 FK/Jacobian/DLS，得到一个完整的目标关节配置。这个阶段不推进物理。

之后 `execute_arm_trajectory()` 再用 smoothstep 将真实机器人从当前 q 插值到目标 q。这样把“能不能找到一个合适的预抓取构型”和“真实执行器如何走过去”分开，便于定位问题。

后续下降和抬升则使用 `move_grasp_point()` 的 Cartesian 闭环，因为这些阶段更需要直接约束抓取点的空间轨迹。

## 9. 软球 reset 与多 episode

方块是刚体，可以直接写 root pose 和 root velocity。软球是 VBD 节点系统，reset 时写所有节点位置/速度。

软球 Z 放置不直接使用名义球心高度。代码先找到当前节点最低 Z，再让：

```text
lowest_node_z = TABLE_TOP_Z + BALL_PARTICLE_RADIUS + BALL_RESET_CLEARANCE
```

这样会考虑粒子接触半径和当前已沉降/变形后的节点几何。

第一轮自然沉降完成后，可以保存一份自由节点形状作为后续 episode 的 reset template，并清零速度。这样后续轮次避免每次都从原始完美网格重新经历较长滚动/变形过程。

多 episode 调试时，要特别关注上一轮的接触历史、warm-start、球节点速度、方块速度和机器人 drive target 是否真正被清理。日志中的 episode 边界和 reset/settle 诊断应保留。

## 10. 抓取成功不是“夹爪闭上了”

当前验收分阶段进行：

第一层是几何包络。球必须仍位于两指区域，夹爪 gap 与球沿抓取轴宽度要合理。

第二层是相对运动。`quick_grasp_check()` 要求夹指基本停止，球相对夹爪的短时速度足够小。

第三层是实际抬升。抬升后球底需要离开桌面，且球心实际升高量至少达到命令高度的 `MIN_LIFT_RATIO=0.85`。

因此“机器人末端升高”不等于“球被抓起”，最终必须检查软球真实节点位置。

## 11. 放置成功判据

放到桌面或方块上时，代码先把球送到目标 XY，再分两阶段下降：先较快接近支撑面，再用 `PLACE_SPEED` 做低速末段。

释放前要求球底接触包络的 clearance 落入允许区间。松爪过程中持续监控：

- 球心相对目标 XY 误差；
- 相对松爪起点的最大漂移；
- 球是否过度穿透支撑面；
- 放到方块时，球心投影是否仍在方块 footprint 内；
- 方块是否倾斜或移动超过容差。

夹爪完全清出球体后才允许 RETREAT，撤离后还要重新做场景稳定检查。

## 12. 常见终端标记

正常调试时优先关注：

```text
RUN_MODE
WORKSPACE
EPISODE_START
EPISODE_RANDOM_TARGETS
SCENE_SETTLED
STATE=PRE_GRASP
STATE=DESCEND
STATE=CLOSE_GRIPPER
STATE=QUICK_GRASP_CHECK
STATE=LIFT
TRANSFER_PATH
STATE=RELEASE
STATE=RETREAT
YAM_DEFORMABLE_BALL_LIFT_OK
YAM_DEFORMABLE_BALL_PUT_BACK_OK
YAM_DEFORMABLE_BALL_ON_BLOCK_OK
EPISODE_COMPLETE
EPISODE_SUMMARY
YAM_DEFORMABLE_TASK_INTEGRATION_OK
```

出现 `*_TIMEOUT`、`FAILURE ...`、`BLOCK_INSTABILITY`、`BALL_INSTABILITY` 时，不要只提高阈值掩盖问题；先判断是运动学、执行器跟踪、接触物理还是验收阈值哪一层失败。

## 13. 推荐调试顺序

当任务失败时按层排查最省时间：

```text
1. reset-only
2. gripper-check
3. FK 与仿真 gripper 对齐
4. 单个固定位置 pre-grasp
5. 随机位置 pre-grasp
6. descend，不闭爪
7. close + quick grasp
8. lift
9. 单侧短距离 transfer
10. 对侧 polar transfer
11. lower + release
12. multi-episode
```

如果 pre-grasp 就有厘米级误差，不要去调软体摩擦；如果下降位置正确但闭爪后球弹走，优先检查 fingertip collision、夹爪速度、目标 gap、接触刚度/阻尼和摩擦；如果抓起稳定但搬运时滑动，再看 transfer 速度、姿态路径和球心闭环。

## 14. 参数调试原则

不要把下面几种“刚度/阻尼”混为一谈：

```text
BALL_YOUNGS_MODULUS / POISSONS_RATIO  -> 软球内部材料
BALL_INTERNAL_DAMPING                 -> 软体内部耗散
soft_contact_ke/kd/mu                 -> 刚体-软体接触
SUPPORT_CONTACT_STIFFNESS/DAMPING     -> 桌面/方块支撑接触
arm/gripper stiffness/damping         -> 关节 position drive
```

同理，稳定阈值只决定“是否通过验收”，不会让真实物理自动更稳定。

## 15. 仓库维护约定

为了避免同一个功能存在多个脚本版本，任务行为只在 `scripts/yam_pick_ball.py` 维护。新增实验模式优先做成 CLI 参数或内部函数，不再复制新的 `*_test.py`、`*_integrated.py` 主任务版本。

`docs/` 只保留两份文档：

- `yam_pick_ball_reference.md`：参数、变量、函数、数学关系；
- `yam_pick_ball_workflow.md`：运行方式、状态机、验收、调试和维护约定。

根目录 README 只负责项目入口和最常用命令，不再重复大段实现细节。

## 16. 当前能力边界

当前成功意味着脚本在配置好的仿真参数、球尺寸/材料、夹爪几何和随机工作区中通过了既定检查，不等于一般化机器人抓取能力。当前系统仍依赖仿真真值位置，没有视觉定位误差；使用硬编码状态机和 DLS IK，不是通用 motion planner；抓取目标 gap 使用材料启发式而非力传感器闭环；软体和接触参数是仿真任务调参，不应直接当成真实材料标定值。
