# YAM Ultra 基于 Isaac Lab Newton 的可变形球放置任务

[English](README.md) | 简体中文

## 当前实现状态

第一个 Newton 任务里程碑已经可以运行：

- `scripts/smoke_test_integrated_task.py` 直接通过 Isaac Lab API 构建一个场景，包含 YAM Ultra 2、静态桌面、刚性目标方块以及四面体可变形球。
- Newton 在 `two_way` 模式下使用 `CoupledMJWarpVBDSolverCfg`：MJWarp 负责刚体与关节系统动力学，VBD 负责可变形球。
- 每个 episode 都会基于确定性随机种子采样彼此分离的方块和球位置，并重置刚体位姿与速度、所有软体节点的位置与速度、自由节点目标以及机器人关节状态。
- `yam_kinematics.py` 使用纯 PyTorch，并基于项目中内置的官方 URDF，实现了 YAM 的正向运动学、几何雅可比矩阵和位置 DLS（阻尼最小二乘）求解。
- 随机化预抓取 waypoint 的最终误差达到 5.6 mm；自定义 FK 计算结果与仿真中的夹爪位置在打印精度范围内一致。

基础 reset 测试：

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --viz none --episodes 3 --steps-per-episode 60 --seed 7
```

当前单 episode IK 测试：

```bash
OMNI_KIT_ACCEPT_EULA=yes ACCEPT_EULA=Y \
/home/lightwheel-laure/embodied_ai_learning/isaac_versions/new/IsaacLab-v3.0.0-beta2.patch1/isaaclab.sh \
-p scripts/smoke_test_integrated_task.py --viz none --episodes 1 --steps-per-episode 1 \
--ik-smoke-steps 240 --seed 7
```

当前限制：完整的抓取、抬升、放置、释放 FSM（有限状态机）以及成功判定指标尚未实现。纯多 episode reset 已经通过，一个 reset 加一次 IK 也能够通过。但机器人运动之后再次执行 reset，目前可能会触发 beta 版耦合求解器发生原生进程退出；在进行多 episode 任务评估之前，需要先解决这一问题。

# Isaac Lab 项目模板

## 概述

该项目/仓库可作为基于 Isaac Lab 构建项目或扩展的模板。
它允许你在 Isaac Lab 核心仓库之外的独立环境中进行开发。

**主要特性：**

- `隔离性`：在 Isaac Lab 核心仓库之外工作，使项目开发保持独立、自包含。
- `灵活性`：该模板支持将代码作为 Omniverse 扩展运行。

**关键词：** extension、template、isaaclab

## 安装

- 按照 [Isaac Lab 安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。
  推荐使用 conda 或 uv 安装，这样可以更方便地从终端调用 Python 脚本。

- 将本项目/仓库单独 clone 或复制到 Isaac Lab 安装目录之外，也就是说不要放在 `IsaacLab` 目录内部。

- 使用已经安装 Isaac Lab 的 Python 解释器，以 editable 模式安装本项目：

```bash
# 如果 Isaac Lab 并非安装在 Python venv 或 conda 中，请使用 'PATH_TO_isaaclab.sh|bat -p' 替代 'python'
python -m pip install -e source/yam_ultra_deformable_place
```

- 验证扩展是否正确安装：

  - 列出可用任务：

    注意：如果任务名称发生变化，可能需要更新 `scripts/list_envs.py` 中的搜索模式 `"Template-"`，否则任务可能不会被列出。

    ```bash
    # 如果 Isaac Lab 并非安装在 Python venv 或 conda 中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
    python scripts/list_envs.py
    ```

  - 运行一个任务：

    ```bash
    # 如果 Isaac Lab 并非安装在 Python venv 或 conda 中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
    python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
    ```

  - 使用 dummy agent 运行任务：

    这些 dummy agent 会输出全零动作或随机动作，可用于确认环境配置是否正确。

    - Zero-action agent：

      ```bash
      # 如果 Isaac Lab 并非安装在 Python venv 或 conda 中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
      python scripts/zero_agent.py --task=<TASK_NAME>
      ```

    - Random-action agent：

      ```bash
      # 如果 Isaac Lab 并非安装在 Python venv 或 conda 中，请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 替代 'python'
      python scripts/random_agent.py --task=<TASK_NAME>
      ```

### 配置 IDE（可选）

如需配置 IDE，请按以下步骤操作：

- 按 `Ctrl+Shift+P` 打开 VSCode 命令面板，选择 `Tasks: Run Task`，然后运行下拉菜单中的 `setup_python_env`。
- 运行该任务时，系统会要求输入 Isaac Sim 安装目录的绝对路径。

如果执行正常，`.vscode` 目录中应该会生成 `.python.env` 文件。
该文件包含 Isaac Sim 和 Omniverse 所提供扩展的 Python 路径，有助于 IDE 索引所有 Python 模块，并在编写代码时提供智能补全与提示。

### 作为 Omniverse Extension 使用（可选）

项目提供了一个示例 UI 扩展，启用扩展后会加载：

`source/yam_ultra_deformable_place/yam_ultra_deformable_place/ui_extension_example.py`

启用扩展的步骤如下：

1. **将本项目/仓库的搜索路径添加到 Extension Manager：**
   - 通过 `Window` -> `Extensions` 打开 Extension Manager。
   - 点击 **Hamburger Icon（三横线菜单）**，然后进入 `Settings`。
   - 在 `Extension Search Paths` 中添加本项目 `source` 目录的绝对路径。
   - 如果尚未添加，还需要在 `Extension Search Paths` 中加入 Isaac Lab 扩展目录路径，即 `IsaacLab/source`。
   - 点击 **Hamburger Icon（三横线菜单）**，然后点击 `Refresh`。

2. **搜索并启用扩展：**
   - 在 `Third Party` 分类下找到你的扩展。
   - 切换开关以启用该扩展。

## 代码格式化

项目包含 pre-commit 模板，可用于自动格式化代码。

安装 pre-commit：

```bash
pip install pre-commit
```

然后运行：

```bash
pre-commit run --all-files
```

## 故障排查

### Pylance 无法完整索引扩展

在部分 VSCode 版本中，某些扩展可能不会被 Pylance 正确索引。

如果出现这种情况，请在 `.vscode/settings.json` 中的 `"python.analysis.extraPaths"` 配置项里加入你的扩展路径：

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/yam_ultra_deformable_place"
    ]
}
```

### Pylance 崩溃

如果 `pylance` 发生崩溃，很可能是由于索引文件过多，从而导致内存耗尽。

一种解决方法是排除项目中没有使用到的部分 Omniverse package。
可以修改 `.vscode/settings.json`，并在 `"python.analysis.extraPaths"` 中注释掉不需要的 package。

例如，以下类型的 package 通常可以考虑排除：

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
