# Deformable Ball 抓取测试建议与开发计划

## 1. 当前阶段目标

当前阶段建议暂时不要扩展到完整的 pick-and-place、RL 或正式 `DirectRLEnv`，而是把问题压缩成一个可重复、可量化的抓取实验：

> **YAM Ultra 刚体机械臂通过 IK 到达抓取位姿，夹爪缓慢闭合并稳定夹持 Newton/VBD 模拟的 deformable ball，随后将球从桌面抬起。**

当前项目中：

- 刚体机械臂：使用 YAM Ultra 2。
- 机械臂控制：使用自定义 FK、Jacobian 和 DLS IK。
- 刚体动力学：Newton / MJWarp。
- 形变球动力学：Newton / VBD。
- 刚体与形变体：使用 `CoupledMJWarpVBDSolverCfg` 的 `two_way` 双向耦合。
- 主要实验代码目前集中在 `scripts/smoke_test_integrated_task.py`。

当前优先目标不是“完成任务”，而是回答以下几个问题：

1. IK 是否能稳定把夹爪送到正确的 pre-grasp 位姿？
2. 夹爪与 deformable ball 的接触是否稳定？
3. 夹爪闭合时是否会穿透、弹飞或产生数值不稳定？
4. deformable ball 是否能被稳定夹持？
5. 保持夹持后，机械臂是否能把球稳定抬离桌面？

只有这些问题逐项通过后，再进入完整的 grasp / lift / place / release 任务。

---

## 2. 推荐总体流程

建议将抓取流程拆成明确的有限状态机（FSM）：

```text
RESET
  ↓
PRE_GRASP
  ↓
DESCEND
  ↓
CLOSE_GRIPPER
  ↓
HOLD
  ↓
LIFT
  ↓
SUCCESS / FAILURE
```

每个状态只负责一件事，不要把机械臂运动、夹爪闭合和抓取判定混在同一个控制循环中。

这样做的最大好处是：一旦失败，可以非常明确地知道问题发生在哪个阶段。

---

## 3. PRE_GRASP：IK 只控制机械臂

当前已经有：

- `yam_kinematics.py`
- forward kinematics
- geometric Jacobian
- damped least-squares position IK

因此建议继续使用当前 IK 方案。

第一阶段只需要让夹爪到达球正上方，例如：

```text
target = ball_center + [0, 0, 0.10 ~ 0.14 m]
```

对于球体抓取，初期可以继续使用 position-only IK，因为球本身具有旋转对称性。

但建议固定夹爪姿态，不要允许手腕在接近球时产生较大的姿态变化。

### 建议

IK 控制与夹爪控制彻底解耦：

```text
Arm:
IK → pre-grasp
IK → approach
IK → lift

Gripper:
OPEN
OPEN
CLOSE
HOLD
HOLD
```

不要在机械臂每个 IK step 中顺便改变 finger joint target。

---

## 4. DESCEND：缓慢下降到抓取高度

不要从 pre-grasp 位姿一步直接移动到球两侧。

建议采用缓慢、分阶段下降：

```text
pre-grasp
    ↓
每个 control step 下降约 0.5 ~ 1 mm
    ↓
grasp height
```

下降期间：

- 夹爪保持完全张开。
- 不做夹持动作。
- 持续检查球的位置和仿真状态是否 finite。

这样可以减少高速刚体接触导致的 deformable body 抖动、弹飞和穿透。

---

## 5. CLOSE_GRIPPER：不要瞬间闭合

这是整个实验最关键的阶段。

不建议：

```text
OPEN → CLOSED
```

一次性把 finger target 跳到闭合位置。

建议采用渐进式 ramp：

```text
OPEN
 ↓
90%
 ↓
80%
 ↓
70%
 ↓
...
 ↓
CLOSED / FORCE-LIMITED HOLD
```

每次只改变很小的 finger joint target，然后运行若干 physics steps。

例如伪代码：

```python
for target in closing_targets:
    set_gripper_target(target)

    for _ in range(physics_steps_per_target):
        step_simulation()
        update_assets()
        check_finite_state()
```

### 核心原则

对于 deformable grasp：

> **慢慢压，比瞬间夹住更重要。**

初期建议宁可抓得慢，也不要追求动作速度。

---

## 6. HOLD：夹住以后先不要抬

夹爪闭合后，建议增加一个单独的 HOLD 阶段。

例如：

```text
0.3 ~ 1.0 s simulated time
```

这段时间机械臂位置固定，夹爪维持当前 target。

重点观察：

- 球是否稳定停留在两个 finger 之间。
- 球是否被慢慢挤出去。
- finger 是否逐渐穿透球。
- 球是否产生异常大的形变。
- 是否出现 NaN / Inf。
- 接触是否导致机械臂明显抖动。

如果 HOLD 阶段不能稳定，就不应该继续做 LIFT。

---

## 7. LIFT：先做最简单的垂直抬升

HOLD 稳定以后，再进入 LIFT。

不要一开始就做复杂轨迹。

建议：

```text
保持 gripper target 不变
        ↓
机械臂末端沿 +Z 缓慢移动
        ↓
先测试 3 cm
        ↓
再测试 5 cm
        ↓
最后测试 10 cm
```

可以继续使用当前 IK：

```text
lift_target = current_gripper_position + [0, 0, lift_height]
```

整个 lift 过程中应持续监控：

- ball center / COM
- finger positions
- deformation
- finite state
- ball 是否掉落

---

## 8. 抓取成功判据

不要只通过 GUI 观察“看起来抓住了”。

建议至少定义以下自动判据。

### 8.1 球离开桌面

例如：

```text
ball_center_z > table_top_z + threshold
```

其中 threshold 可以先取：

```text
2 ~ 3 cm
```

### 8.2 持续时间

球不能只是瞬间被弹起来。

建议要求：

```text
连续 N 个 simulation steps 保持在离桌高度以上
```

### 8.3 球仍处于夹爪区域

检查 ball center 是否仍位于两个 finger 中间附近。

### 8.4 数值稳定

必须保证：

```text
robot state finite
rigid object state finite
deformable nodal position finite
deformable nodal velocity finite
```

即不能出现：

```text
NaN
Inf
```

### 8.5 形变不过度

可以保存球 reset 时的节点位置，并测量当前节点相对于参考形状的最大或 RMS deformation。

如果 deformation 超过不合理范围，应判为失败或物理参数异常。

---

## 9. 强烈建议先做“隔离实验”

在把 IK、机器人运动、夹爪和 deformable coupling 全部一起调之前，先把问题拆开。

### Test A：只测试夹爪与 deformable ball

```text
机械臂固定
   ↓
只控制 joint7 / joint8
   ↓
缓慢关闭夹爪
   ↓
观察 ball deformation / contact
```

这个测试回答：

> Newton/VBD 的刚体-软体接触本身是否稳定？

如果这个实验都不稳定，则暂时没有必要继续调整 IK。

### Test B：固定机械臂 + close + hold

验证：

```text
夹爪闭合
→ deformable ball 压缩
→ 系统保持稳定
```

### Test C：IK pre-grasp + close

加入机械臂运动，但暂时不 lift。

### Test D：IK + close + lift

最终才进入完整抓起测试。

推荐调试顺序：

```text
固定机械臂 + close
        ↓
固定机械臂 + close + hold
        ↓
机械臂 IK + close
        ↓
机械臂 IK + close + hold
        ↓
机械臂 IK + close + lift
```

---

## 10. Newton / VBD 调参建议

当前已有参数大致包括：

```text
dt = 1 / 120
num_substeps = 4
VBD iterations = 5
Young's modulus ≈ 5e4 Pa
Poisson ratio = 0.35
particle radius = 0.006
```

初期不要同时修改所有参数。

建议按照问题类型逐步调整。

### 如果 finger 穿透球

优先顺序：

```text
1. 降低 gripper closing speed
2. 增加 num_substeps
3. 增加 VBD iterations
4. 检查/调整 soft contact 参数
5. 降低 gripper actuator stiffness
6. 最后再考虑材料参数
```

不要一看到穿透就首先修改 Young's modulus。

### 推荐 debug 配置

可以专门准备一个更保守的抓取 debug 配置：

```text
dt = 1 / 120
num_substeps = 8
VBD iterations = 10
```

如果：

```text
substeps=8 / iterations=10  可以稳定
substeps=4 / iterations=5   不稳定
```

则基本可以判断主要问题来自数值求解精度，而不是 IK 或抓取几何本身。

之后再逐步降低 solver cost。

---

## 11. Gripper actuator 建议

对于 deformable object，不建议让 gripper 成为无限刚性的 position controller。

需要重点关注：

- stiffness
- damping
- effort limit
- closing velocity

如果夹爪 position target 过于激进，solver 会尝试强行达到目标位置，容易导致：

- deformable penetration
- 极端压缩
- 数值不稳定
- 接触震荡

更合理的思路是：

```text
position target 负责“想闭合到哪里”
effort limit   负责“最多能夹多用力”
```

因此 deformable grasp 的最终稳定性通常不仅依赖 soft-body material，还依赖 gripper actuator 参数。

---

## 12. 建议新增独立测试脚本

推荐新增：

```text
scripts/test_deformable_grasp.py
```

不要一开始修改正式 RL environment。

该脚本只负责抓起测试。

### 推荐执行流程

```text
spawn scene
    ↓
reset ball / robot
    ↓
PRE_GRASP
    ↓
DESCEND
    ↓
CLOSE_GRIPPER
    ↓
HOLD
    ↓
LIFT
    ↓
SUCCESS / FAILURE
```

### 推荐输出

```text
BALL_INITIAL_CENTER
BALL_CURRENT_CENTER
BALL_LIFT_HEIGHT
LEFT_FINGER_POSITION
RIGHT_FINGER_POSITION
MAX_DEFORMATION
IK_FINAL_ERROR
FINITE_STATE
GRASP_SUCCESS
```

如果后续能够获得可靠 contact 数据，可以再增加：

```text
LEFT_CONTACT
RIGHT_CONTACT
CONTACT_FORCE
```

---

## 13. 随机化测试

单次成功不能说明抓取已经稳定。

当基础测试跑通后，建议固定随机种子范围测试 10 次或更多：

```text
episode 0 → PASS
episode 1 → PASS
episode 2 → FAIL
...
```

最终打印：

```text
GRASP_SUCCESS_RATE = 8 / 10
```

随机项初期只建议包括：

- ball x/y position
- 少量 ball initial orientation（如果适用）

暂时不要同时随机：

- material
- friction
- robot pose
- grasp orientation
- solver parameters

先把最基本的物理抓取做稳定。

---

## 14. 当前代码结构建议

目前建议保持：

```text
scripts/smoke_test_integrated_task.py
    → 用于已有 Newton 场景与 reset / IK smoke test

scripts/test_deformable_grasp.py
    → 专门用于 deformable grasp FSM 和抓起验证

source/.../yam_kinematics.py
    → FK / Jacobian / DLS IK
```

暂时不要把抓取逻辑直接搬进：

```text
source/.../tasks/direct/.../yam_ultra_deformable_place_env.py
```

因为当前 `DirectRLEnv` 代码仍主要是 Cartpole 模板骨架。

等独立抓取测试稳定以后，再将已经验证过的逻辑迁移到正式 environment。

---

## 15. 推荐开发顺序

### Milestone 1：接触稳定

目标：

```text
固定机械臂
夹爪闭合
球稳定变形
无穿透 / 无 NaN / 无爆炸
```

### Milestone 2：稳定夹持

目标：

```text
close
→ hold 0.5 s
→ ball 不滑出
```

### Milestone 3：IK 抓取

目标：

```text
IK pre-grasp
→ descend
→ close
→ hold
```

### Milestone 4：抓起

目标：

```text
IK
→ close
→ hold
→ lift 3 cm
```

### Milestone 5：提高 lift height

```text
3 cm
→ 5 cm
→ 10 cm
```

### Milestone 6：随机位置成功率

```text
10+ episodes
→ 统计成功率
```

### Milestone 7：迁移正式任务

只有此时再进入：

```text
grasp
→ lift
→ move
→ place
→ release
```

以及后续 RL / policy 训练。

---

## 16. 调试原则

这个阶段最重要的原则是：

> **一次只增加一个复杂度。**

不要同时调：

```text
IK
+ gripper
+ material
+ solver
+ contact
+ trajectory
+ reset
```

否则失败后很难判断根因。

推荐始终按下面的层级验证：

```text
soft contact
    ↓
gripper closing
    ↓
gripper hold
    ↓
IK approach
    ↓
lift
    ↓
randomized grasp
    ↓
full task
```

---

## 17. 当前最推荐的下一步

下一步最值得实现的是：

```text
scripts/test_deformable_grasp.py
```

包含五个核心状态：

```text
PRE_GRASP
DESCEND
CLOSE_GRIPPER
HOLD
LIFT
```

第一版只需要做到：

1. 固定一个球位置。
2. IK 到 pre-grasp。
3. 缓慢下降。
4. 缓慢闭合 gripper。
5. hold 一段时间。
6. 垂直 lift 3 cm。
7. 输出自动成功判据。

当这个版本稳定后，再增加随机 ball position 和更高的 lift height。

---

## 参考资料

- Isaac Lab Newton / VBD documentation:
  https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/physical-backends/newton/index.html

- Isaac Lab VBD solver tuning:
  https://isaac-sim.github.io/IsaacLab/main/source/concepts/solver-tuning/tune_vbd.html

建议后续同时参考 Isaac Lab 官方 deformable manipulation 示例中的 coupling、contact、actuator 和 VBD 参数配置，但实际参数仍应以 YAM Ultra 当前抓取实验为准。
