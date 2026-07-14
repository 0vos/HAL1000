# 可移植性指南（PORTABILITY）

> 让本仓库可以在 **Linux（服务器）**、**macOS**、**Windows** 三平台上一致运行，无需修改任何代码。

---

## 1. 设计目标

本项目的所有路径都通过 `common/path_utils.py` 统一解析，目标是：

1. **服务器（Linux / `/opt/conda/envs/hal`）**：开箱即用，无需任何环境变量。
2. **macOS**：conda 环境 + 本地模型目录。
3. **Windows**：conda 或 venv + Windows 风格路径。

无论哪种平台，`python b2_run_skill.py ...` / `python b3_tool_layer.py ...` 等命令的相对路径都基于 `cwd`（CLI 调用时所在的目录），因此推荐统一从 `agent/code` 目录调用。

---

## 2. 路径解析的三档兜底

`common/path_utils.resolve_model_path(raw, base_dir)` 的解析顺序：

| 序号 | 输入形态 | 解析方式 |
|---|---|---|
| 1 | `${HAL_MODEL_PATH:-/abs/path}` 占位符 | 先读 `HAL_MODEL_PATH` 环境变量；为空则用 `:-` 后的默认值 |
| 2 | 绝对路径 | 直接使用，必须存在 |
| 3 | 相对路径（相对 `configs/model.yaml`） | 拼成 `configs/<path>`，必须存在 |
| 4 | `os.environ['HAL_MODEL_PATH']` | 若 1-3 都失败，再读一次环境变量 |
| 5 | `PROJECT_ROOT.parent/<basename>` | 项目父目录下的同名目录（即 `/root/siton-tmp/HAL1000/Qwen3.5-4B`） |
| 6 | `PROJECT_ROOT/models/<basename>` | 项目内 models/ 目录 |
| 7 | 全部失败 | 抛 `FileNotFoundError` 并列出已搜索的所有位置 |

`configs/model.yaml` 默认配置：

```yaml
model:
  model_name_or_path: ${HAL_MODEL_PATH:-../Qwen3.5-4B}
  tokenizer_name_or_path: ${HAL_MODEL_PATH:-../Qwen3.5-4B}
```

> 默认值 `../Qwen3.5-4B` 是相对 `configs/model.yaml` 的，会解析为 `<agent root>/Qwen3.5-4B`，刚好覆盖服务器上的 `/root/siton-tmp/HAL1000/Qwen3.5-4B`。

---

## 3. 三平台运行步骤

### 3.1 Linux（服务器，已验证）

```bash
# 1. 连接服务器
ssh -p 20021 root@202.199.13.141     # 密码：HNxgk1pswv

# 2. 激活 hal 环境
source /opt/conda/etc/profile.d/conda.sh
conda activate hal
# （或直接用绝对路径 /opt/conda/envs/hal/bin/python）

# 3. 进入代码目录
cd /root/siton-tmp/HAL1000/agent/code

# 4. 跑 B2 个人演示
python b2_run_skill.py --skill calculator \
    --input ../data/tool_inputs/tool_input_calculator.json \
    --outdir /tmp/demo/calculator_ok

# 5. 跑 B3 个人演示
python b3_tool_layer.py --tools_config ../configs/tools.yaml \
    --toolset basic_tools --export_schema --outdir /tmp/demo/schema

# 6. 一键全跑
cd /root/siton-tmp/HAL1000/agent
bash scripts/run_all_demos.sh        # mock
bash scripts/run_all_demos.sh pj     # 含 prompt_json
```

### 3.2 macOS（本地开发）

```bash
# 1. 创建 conda 环境（首次）
conda create -n hal python=3.10 -y
conda activate hal

# 2. 安装依赖
cd HAL1000/agent
pip install -r requirements.txt

# 3. 设置模型路径（推荐用环境变量）
export HAL_MODEL_PATH=$HOME/models/Qwen3.5-4B

# 4. 进入代码目录
cd code

# 5. 跑 B2 / B3（路径全部相对项目根，相对路径解析照常工作）
python b2_run_skill.py --skill calculator \
    --input ../data/tool_inputs/tool_input_calculator.json \
    --outdir /tmp/demo/calculator_ok

python b3_tool_layer.py --tools_config ../configs/tools.yaml \
    --toolset basic_tools --export_schema --outdir /tmp/demo/schema
```

如果模型放在仓库内的 `../Qwen3.5-4B`（即 `HAL1000/Qwen3.5-4B`），无需设置 `HAL_MODEL_PATH`，会自动找到。

### 3.3 Windows（PowerShell）

```powershell
# 1. 创建 conda 环境（首次）
conda create -n hal python=3.10 -y
conda activate hal

# 2. 安装依赖
cd HAL1000\agent
pip install -r requirements.txt

# 3. 设置模型路径（PowerShell 风格）
$env:HAL_MODEL_PATH = "E:\models\Qwen3.5-4B"
# 或者放到 cmd 文件里： set HAL_MODEL_PATH=E:\models\Qwen3.5-4B

# 4. 进入代码目录
cd code

# 5. 跑 B2 / B3
python b2_run_skill.py --skill calculator `
    --input ..\data\tool_inputs\tool_input_calculator.json `
    --outdir .\out\calculator_ok

python b3_tool_layer.py --tools_config ..\configs\tools.yaml `
    --toolset basic_tools --export_schema --outdir .\out\schema
```

Windows 下 PowerShell 路径分隔符可用 `\` 或 `/`，两种都能被 `pathlib` 正确处理。

---

## 4. SSH 连接（仅本项目作者使用）

服务器要求 `ssh -p 20021`，Windows OpenSSH 不支持交互式密码登录。本项目使用 **paramiko** 封装：

```bash
# 任意命令
python E:\MyCode\Python\PycharmProjects\HAL1000\ssh_run.py "echo hello"

# 传 stdin 命令（避免 PowerShell 引号转义问题）
type cmd.txt | python E:\MyCode\Python\PycharmProjects\HAL1000\ssh_run.py pipe

# 上传/下载文件
python E:\MyCode\Python\PycharmProjects\HAL1000\ssh_run.py put <local> <remote>
python E:\MyCode\Python\PycharmProjects\HAL1000\ssh_run.py get <remote> <local>

# 递归下载目录
python E:\MyCode\Python\PycharmProjects\HAL1000\ssh_run.py get-tree <remote_dir> <local_dir>
```

> `ssh_run.py` 内置了 `HNxgk1pswv` 密码和 `202.199.13.141:20021` 端口。请勿在公共网络明文传输。

---

## 5. 模型要求

- **Qwen3.5-4B**（约 8.3 GB）safetensors 格式
- 必须存在 `config.json` / `tokenizer.json` / `tokenizer_config.json` / `model.safetensors.index.json`
- 显存：bfloat16 需要 ~10 GB；H200 / A100 / 3090 / 4090 等都可以跑

如果使用其他模型（如 `Qwen2.5-7B-Instruct`），只需：

```bash
# 1. 放到任何位置
export HAL_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct

# 2. 确保 tokenizer 的 chat_template 支持 tool_calls JSON 输出
#    （Qwen 官方 tokenizer 模板已支持；其他模型可能需要自定义 prompt）
```

---

## 6. 常见问题

### Q1：报错 "local model path does not exist: /root/siton-tmp/HAL1000/Qwen3.5-4B"（在本地 Windows）

**A**：本地模型不在服务器路径下。设置：

```powershell
$env:HAL_MODEL_PATH = "E:\models\Qwen3.5-4B"
```

或在 `configs/model.yaml` 中改默认值为本地路径：

```yaml
model_name_or_path: ${HAL_MODEL_PATH:-E:/models/Qwen3.5-4B}
```

### Q2：PowerShell 命令太长导致换行后无法运行

**A**：用 PowerShell 的换行符 `` ` `` (反引号)。或在本地用 `.cmd` 脚本：

```cmd
@echo off
set HAL_MODEL_PATH=E:\models\Qwen3.5-4B
cd /d %~dp0code
python b2_run_skill.py --skill calculator ^
    --input ..\data\tool_inputs\tool_input_calculator.json ^
    --outdir .\out
```

### Q3：服务器端 locale 是 POSIX（zh_CN.UTF-8 未启用），终端输出中文乱码

**A**：中文乱码只影响终端 print，文件内容都是 UTF-8 编码正常。用 `iconv` 转码查看或直接 `sftp_get` 下载到本地查看：

```bash
LANG=zh_CN.UTF-8 python b2_run_skill.py ...    # 临时切换 locale
```

### Q4：模型加载慢（约 5-10 秒）怎么办

**A**：`_MODEL_CACHE` 缓存已经做在 `b4_local_agent_llm.py` 里，同一进程内第二次调用 `generate_ai_message` 直接命中（输出 `model_cache=hit`）。如果跨进程，重新加载是正常的。

### Q5：服务器没 GPU，能否跑 prompt_json？

**A**：可以，CPU 推理可行但慢（每次 30-90 秒）。建议：

1. 优先用 `mock` 模式验证流程
2. 需要展示真实模型时，至少跑 5 条 prompt_json 抽样

---

## 7. 一键演示脚本

服务器上：

```bash
bash /root/siton-tmp/HAL1000/agent/scripts/run_all_demos.sh        # mock
bash /root/siton-tmp/HAL1000/agent/scripts/run_all_demos.sh pj     # 含 prompt_json
```

本地（macOS / Windows），把脚本里的 `python` 替换为 `python` 即可（无需替换 `conda activate`，因为已通过环境变量 `HAL_MODEL_PATH` 配置模型）。

---

**PORTABILITY.md 结束**。本项目的核心路径解析能力在 `common/path_utils.py`，任何新增路径相关逻辑都应该复用 `resolve_path / resolve_cli_path / resolve_from_file / resolve_model_path` 四个函数。