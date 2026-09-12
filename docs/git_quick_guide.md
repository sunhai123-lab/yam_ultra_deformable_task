# Git 简洁使用指南

本项目常用场景只有两个方向：

- **本地 VS Code → GitHub**：上传修改
- **GitHub → 本地 VS Code**：拉取更新

下面以当前分支 `fix/rigid-block-stability` 和文件 `scripts/smoke_test_integrated_task.py` 为例。

## 1. 先确认当前状态

```bash
git branch --show-current
git status
```

当前开发分支应为：

```text
fix/rigid-block-stability
```

---

## 2. 上传：本地 VS Code → GitHub

### 2.1 上传全部修改

```bash
git add -A
git commit -m "update"
git push

git pull
```

其中：

- `git add .`：加入当前目录及子目录中的修改、新文件；通常项目根目录执行即可。
- `git add -A`：加入整个仓库里的所有修改、删除和新文件，范围更彻底。
- `git commit`：在本地创建一次提交。
- `git push`：把本地提交上传到 GitHub 当前分支。

想明确指定远端和分支，也可以：

```bash
git push origin fix/rigid-block-stability
```

### 2.2 只上传一个文件

例如只上传：

```text
scripts/smoke_test_integrated_task.py
```

执行：

```bash
git add scripts/smoke_test_integrated_task.py
git commit -m "update integrated smoke test"
git push
```

其他未 `git add` 的文件不会进入这次 commit。

### 2.3 只上传一个文件夹

例如只上传 `scripts/` 下的修改：

```bash
git add scripts/
git commit -m "update scripts"
git push
```

---

## 3. 下拉：GitHub → 本地 VS Code

### 3.1 拉取当前分支全部最新内容

最简单：

```bash
git pull
```

如果要明确指定：

```bash
git pull origin fix/rigid-block-stability
```

如果 HTTPS 网络偶尔有问题，可以尝试：

```bash
git -c http.version=HTTP/1.1 pull origin fix/rigid-block-stability
```

拉取完成后，VS Code 中的文件会更新。

### 3.2 只从 GitHub 更新一个文件

Git 没有常用的“只 pull 一个文件”命令。做法是先更新远端索引，再把指定文件恢复成远端版本。

例如只更新：

```text
scripts/smoke_test_integrated_task.py
```

执行：

```bash
git fetch origin
git restore --source=origin/fix/rigid-block-stability --worktree \
  scripts/smoke_test_integrated_task.py
```

**注意：这会覆盖该文件当前未提交的本地修改。**

如果想先备份：

```bash
cp scripts/smoke_test_integrated_task.py \
   /tmp/smoke_test_integrated_task.backup.py
```

然后再执行 `git restore`。

---

## 4. 最常用的日常流程

开始工作前：

```bash
git pull
```

修改代码并 `Ctrl+S` 保存后：

```bash
git status
git add .
git commit -m "说明这次修改"
git push
```

如果只想提交一个文件：

```bash
git add scripts/smoke_test_integrated_task.py
git commit -m "update smoke test"
git push
```

---

## 5. `Ctrl+S`、commit、push 的区别

```text
Ctrl+S
  ↓
只保存到本地磁盘

 git add
  ↓
选择哪些修改进入本次提交

 git commit
  ↓
创建本地 Git 提交

 git push
  ↓
真正上传到 GitHub
```

所以：**Ctrl+S 不等于上传 GitHub。**

---

## 6. 快速检查本地是否和 GitHub 对齐

查看当前本地提交：

```bash
git rev-parse --short HEAD
```

先更新远端信息：

```bash
git fetch origin
```

再查看远端分支提交：

```bash
git rev-parse --short origin/fix/rigid-block-stability
```

两个 SHA 相同，说明当前提交位置一致。
