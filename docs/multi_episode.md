# 多轮随机抓取与方块放置

随机范围和调整方法见 [物体位置随机化](randomization.md)。

方块放置流程已支持连续多个 episode。每轮都会执行：

1. 使用同一个伪随机数生成器继续采样新的方块和球坐标；
2. 从第 2 轮开始把球和方块短暂移到隔离位置并推进两个不渲染的物理步，清除上一轮接触约束和 warm-start 状态；
3. 机械臂关节复位到默认位置并清零关节速度；
4. 方块写入新的位姿并清零刚体速度；
5. 第 1 轮自然沉降后保存落桌形状；后续轮次把该自由节点形状平移到新坐标并清零复位速度；
6. 等待本轮场景稳定，再完成抓取、抬升、搬运、放置和撤离；
7. 独立检查本轮的抬升、放置误差、释放漂移和最终稳定状态。

F5 的 `Task: Place ball on block GUI` 和 Ctrl+Shift+B 默认运行 3 轮：

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
../isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py \
--visualizer kit --keep-open --episodes 3 --steps-per-episode 60 \
--pick-lift --place-on-block --lift-height 0.03 --seed 7
```

`--seed` 控制整组采样序列，而不是让每轮得到同一坐标。同一个 seed 和 episode 数会产生相同的位置采样序列；软体网格生成和物理结果仍可能存在细微差异。

沉降模板只保存第 1 轮落在水平桌面后的节点形状和零速度，不保存抓取或方块上的形变。每轮仍会重新等待稳定，节点运动标志保持自由；它用于避免把未经重力沉降的离散球网格反复放回桌面后产生相同的偏心滚动。

主要日志：

- `EPISODE_START` / `EPISODE_COMPLETE`：本轮边界；
- `STATE=CLEAR_CONTACT_HISTORY`：第 2 轮及以后先执行无接触清理阶段；
- `EPISODE_RANDOM_TARGETS`：本轮采样的方块和球目标坐标；
- `YAM_DEFORMABLE_BALL_LIFT_OK episode=N`：本轮抓起验收通过；
- `PLACE_ON_BLOCK_RESULT`：带有本轮编号的放置结果；
- `MOTION_TIMING`：本轮从抓取开始到放置结束的仿真和实际时间；
- `EPISODE_SUMMARY`：请求、复位、抓取和放置的成功数量。

任意一轮失败会立即抛出异常并停止后续轮次，避免最终汇总掩盖中间失败。`--keep-open` 只在全部轮次通过后保持最后一轮画面。

## 三轮验证（seed=7）

无 GUI 完整测试输出：`requested=3 reset_ok=3 pick_ok=3 placement_ok=3`。三轮采样坐标如下：

| Episode | 方块 XY (m) | 球 XY (m) | 动作仿真时间 (s) | 实际计算时间 (s) |
| --- | --- | --- | ---: | ---: |
| 0 | (0.0653, 0.0951) | (0.1111, -0.1728) | 8.517 | 34.859 |
| 1 | (0.0950, 0.1166) | (0.0281, -0.1293) | 8.392 | 34.543 |
| 2 | (0.0252, 0.1234) | (0.0298, -0.1709) | 8.867 | 36.923 |

三轮实际抬升分别为 111.83、112.42、112.29 mm，均通过各自的最低抬升要求。第 2 轮初始化就绪用了 159 步，其余两轮为 60 步；最终放置仍使用严格的稳定判据。

高速横移在误差超过 5 mm 时会保持 0.30 m/s 目标速度追加一次短程闭环收敛。它使边缘随机位置也完成放置，同时保留 5 mm 的最终搬运误差门槛。
