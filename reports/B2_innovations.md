# B2 创新改进：可信智能文档处理与跨平台受限执行环境

负责人：杨贺淳  
服务器路径：`/root/siton-tmp/HAL1000/agent`  
完成日期：2026-07-13

## 1. 改动边界

本次仅修改或新增 B2 自有文件，没有修改 B1、B3、B4、B5 源码，也没有修改共享的 `configs/tools.yaml`。

覆盖的已有文件：

- `code/b2_run_skill.py`
- `code/b2_advanced.py`
- `code/safe_python_exec.py`

新增文件：

- `skills/document_inspector.py`
- `tests_b2/test_b2_innovations.py`
- `scripts/run_b2_innovations.sh`
- `data/tool_inputs/advanced/document_inspector_*.json`
- `data/b2_samples/poster.pdf`

服务器原文件备份位于 `outputs/B2_backups/20260713_224539/`。

## 2. 跨平台受限执行环境

### 2.1 改进前问题

- 使用 `signal.SIGALRM`，Windows 不支持。
- 在 Agent 主进程内调用 `exec`，隔离边界不足。
- 主要依赖字符串黑名单，容易出现等价写法绕过。
- 没有跨平台内存监控、进程树清理和输出上限元数据。

### 2.2 新执行链路

```text
source
  -> AST allowlist validation
  -> minimal environment
  -> fresh subprocess
  -> JSON stdin/stdout protocol
  -> timeout + RSS memory monitor
  -> bounded stdout/stderr
  -> process-tree cleanup
  -> structured result
```

主要能力：

- Windows 和 Linux 使用同一套 `subprocess` 协议。
- 禁止 import、文件访问、网络访问、动态调用、私有属性和用户自定义函数。
- 允许安全算术、容器、循环以及 `math`、`statistics` 白名单函数。
- 默认超时 5 秒、内存 128 MB、输出 4000 字符。
- Linux 额外使用 `RLIMIT_AS` 和 `RLIMIT_CPU`；Windows/Linux 均使用 `psutil` 监控 RSS。
- 超时或超内存时终止工作进程及其进程树。
- 返回 `security` 元数据，记录平台、策略、限制和峰值内存。

成功结果示意：

```json
{
  "status": "success",
  "result": 45,
  "stdout": "",
  "stderr": "",
  "security": {
    "isolation": "subprocess-json-stdio",
    "ast_policy": "allowlist-v1",
    "timeout_seconds": 5.0,
    "memory_limit_mb": 128
  }
}
```

### 2.3 安全边界

该功能应称为“受限执行环境”，不宣称为不可突破的绝对沙箱。当前文件和网络访问由 AST 策略与最小内置函数阻断，并非容器级系统调用过滤。若未来执行更复杂的不可信代码，应继续接入容器、seccomp、WASM 或专用沙箱服务。

## 3. 智能文档处理 Skill

新增 `document_inspector`，保留原 `pdf_reader`、`docx_reader` 行为不变。

### 3.1 支持格式

- TXT / Markdown
- PDF
- DOCX
- CSV / TSV

### 3.2 结构化输出

- 文档类型、标题、大小、SHA-256、页数和截断状态。
- 按页码、段落号、行号或表格行号定位的证据块。
- PDF/DOCX/CSV 表格抽取。
- 查询词相关证据排序和 `top_k` 返回。
- PDF 文本不足时的 OCR 自动降级接口与能力状态。
- 文件大小、字符数、分块大小、重叠长度和返回数量限制。
- Skill 内部再次校验 data root，避免依赖不同版本公共函数时出现路径越界。

典型返回：

```json
{
  "document": {
    "source": "b2_samples/poster.pdf",
    "file_type": "pdf",
    "num_pages": 1,
    "num_chunks": 2,
    "sha256": "adb83b0d..."
  },
  "matches": [
    {
      "chunk_id": "chunk_0001",
      "score": 3,
      "location": {"page": 1}
    }
  ]
}
```

### 3.3 Poster 实测

输入：`data/b2_samples/poster.pdf`，查询：`B2 Skill`。

| 指标 | 结果 |
|---|---:|
| 文件大小 | 2,060,280 bytes |
| 页数 | 1 |
| 提取字符 | 1,244 |
| 证据块 | 2 |
| 命中位置 | 第 1 页 |
| 服务器耗时 | 313.556 ms |
| OCR | 未安装，可插拔降级状态返回正常 |

原始 UTF-8 提取结果已确认包含“杨贺淳”和“B2”。Windows PowerShell 5 直接读取 UTF-8 JSON 时可能显示乱码，应使用 Python `json.load(..., encoding="utf-8")` 检查结果。

## 4. 测试结果

服务器命令：

```bash
bash /root/siton-tmp/HAL1000/agent/scripts/run_b2_innovations.sh
bash /root/siton-tmp/HAL1000/agent/scripts/run_b2_baseline.sh
bash /root/siton-tmp/HAL1000/agent/scripts/run_b2_advanced.sh
```

| 测试组 | 结果 |
|---|---:|
| 新增 B2 创新测试 | 10/10 通过 |
| 原 B2 基础回归 | 10 个场景全部符合预期 |
| 原 B2 进阶回归 | 12 个场景全部符合预期 |
| Windows 本地新测试 | 通过；DOCX 因本机缺依赖跳过 |
| Linux 服务器新测试 | 10/10 通过，包含 DOCX |

新增安全场景包括：导入阻断、私有属性阻断、文件访问阻断、动态属性阻断、死循环终止、内存超限终止、输出截断和除零错误结构化。

服务器原进阶脚本中的沙箱指标：

| 场景 | 状态 | 耗时 |
|---|---|---:|
| `result = sum(range(10))` | success | 74.705 ms |
| `import os` | PERMISSION_DENIED | 28.603 ms |
| `while True: pass` | EXECUTION_TIMEOUT | 5047.683 ms |

## 5. 尚未修改的共享集成点

`document_inspector` 已能通过 B2 CLI 独立运行，但尚未加入 `configs/tools.yaml`，因此 B3/B4 还看不到它。这是有意保留的边界。后续与 B3 负责人确认后，只需在共享工具配置中增加 schema 和目标 toolset，不需要再修改 Skill 实现。

服务器当前没有 `tesseract`、`pdftoppm` 或 `pytesseract`。OCR 接口已支持 `off/auto/force` 和能力状态返回，但扫描版 PDF 的真实 OCR 需要后续单独安装依赖并增加 OCR 精度实验。
