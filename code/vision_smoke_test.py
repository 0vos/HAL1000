"""
vision_smoke_test.py — Qwen3.5 视觉能力最小验证脚本

只测三件事，逐步排查问题出在哪一层：
  1. AutoProcessor 能否加载（tokenizer + image processor）
  2. AutoModelForImageTextToText 能否识别 Qwen3_5ForConditionalGeneration 架构并加载权重
  3. 一次完整的图文问答能否跑通（不经过 image_qa.py，直接调用最底层 API）

模型路径解析优先级（无需每次手动 export）：
  1. --model_path 显式传参
  2. HAL_MODEL_PATH 环境变量（如果设置了）
  3. configs/model.yaml 里的 model_name_or_path（自动展开 ${HAL_MODEL_PATH:-../Qwen3.5-4B}
     占位符，找不到环境变量时 fallback 到与 agent/ 同级目录下的 Qwen3.5-4B —— 这与
     hal_chat.py / b4_local_agent_llm.py 使用的是同一套 resolve_model_path 逻辑）

用法（在服务器上，agent/code 目录下）：
    python vision_smoke_test.py                                    # 只测加载，不需要任何 export
    python vision_smoke_test.py --image /path/to/test.jpg          # 测完整问答
    python vision_smoke_test.py --model_path /custom/path          # 显式指定模型路径
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common.path_utils import resolve_model_path, resolve_cli_path, expand_placeholders  # noqa: E402
from common.io_utils import read_yaml  # noqa: E402


def _step(name: str):
    print(f"\n{'='*60}\n[STEP] {name}\n{'='*60}")


def _resolve_model_path(args: argparse.Namespace) -> Path:
    """按优先级解析模型路径，无需每次手动 export。"""
    if args.model_path:
        return Path(args.model_path).expanduser().resolve()

    model_config_path = args.model_config or str(_HERE.parent / "configs" / "model.yaml")
    cfg_path = resolve_cli_path(model_config_path)
    cfg = read_yaml(cfg_path)
    setting = expand_placeholders(cfg.get("model", {}).get("model_name_or_path", ""))
    try:
        return resolve_model_path(setting, cfg_path)
    except FileNotFoundError as exc:
        print(f"❌ 无法自动解析模型路径：{exc}")
        print("请手动指定 --model_path /path/to/Qwen3.5-4B，或确保 model.yaml 里的路径可解析")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", default=None,
        help="Qwen3.5-4B 权重目录。不传时自动从 model.yaml 解析（无需 export HAL_MODEL_PATH）",
    )
    parser.add_argument("--model_config", default=None, help="model.yaml 路径，默认 ../configs/model.yaml")
    parser.add_argument("--image", default=None, help="测试图片路径，不传则只测加载不测推理")
    parser.add_argument("--question", default="这张图里有什么？")
    args = parser.parse_args()

    model_path = _resolve_model_path(args)
    auto_note = "" if os.environ.get("HAL_MODEL_PATH") or args.model_path else "  （自动解析，无需 export HAL_MODEL_PATH）"
    print(f"模型路径: {model_path}{auto_note}")

    # ── STEP 0: 基础依赖检查 ──────────────────────────────────
    _step("0. 基础依赖检查")
    try:
        import torch
        print(f"✅ torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ torch 未安装: {e}")
        sys.exit(1)
    try:
        import transformers
        print(f"✅ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ transformers 未安装: {e}")
        sys.exit(1)
    try:
        from PIL import Image
        print("✅ Pillow 已安装")
    except ImportError as e:
        print(f"❌ Pillow 未安装（pip install pillow）: {e}")
        sys.exit(1)

    # ── STEP 1: AutoProcessor 加载 ────────────────────────────
    _step("1. AutoProcessor.from_pretrained")
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=True,
        )
        print(f"✅ processor 加载成功: {type(processor).__name__}")
    except Exception:
        print("❌ AutoProcessor 加载失败：")
        traceback.print_exc()
        sys.exit(1)

    # ── STEP 2: AutoModelForImageTextToText 加载 ─────────────
    _step("2. AutoModelForImageTextToText.from_pretrained（这是最可能失败的一步）")
    model = None
    load_errors = []
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        try:
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                print(f"⚠️  transformers 里没有 {cls_name} 这个类，跳过")
                continue
            print(f"尝试用 {cls_name} 加载...")
            model = cls.from_pretrained(
                str(model_path),
                local_files_only=True,
                trust_remote_code=True,
                dtype="auto",
                device_map="auto",
            )
            print(f"✅ 用 {cls_name} 加载成功: {type(model).__name__}")
            break
        except Exception as exc:
            print(f"❌ {cls_name} 加载失败: {exc}")
            load_errors.append((cls_name, exc))

    if model is None:
        print("\n❌❌❌ 两个 Auto 类都加载失败，视觉模型无法使用。")
        print("常见原因：transformers 版本太旧，不认识 Qwen3_5ForConditionalGeneration 架构。")
        print("解决办法：pip install -U transformers （Qwen3.5 需要较新版本，建议 >= 5.12）")
        for name, exc in load_errors:
            print(f"\n--- {name} 完整报错 ---")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        sys.exit(1)

    if args.image is None:
        print("\n✅ 加载测试全部通过（未指定 --image，跳过真实推理测试）")
        print("如需测试完整问答，重新运行并加上: --image /path/to/test.jpg")
        return

    # ── STEP 3: 完整图文问答 ──────────────────────────────────
    _step("3. 完整图文问答（apply_chat_template + generate）")
    if not os.path.isfile(args.image):
        print(f"❌ 图片不存在: {args.image}")
        sys.exit(1)

    try:
        image = Image.open(args.image).convert("RGB")
        print(f"✅ 图片加载成功: {image.size}")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": args.question},
                ],
            }
        ]
        prompt_text = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )
        print(f"✅ apply_chat_template 成功，prompt 长度: {len(prompt_text)} 字符")
        print(f"   prompt 预览: {prompt_text[:200]}...")

        inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
        print(f"✅ processor(text=, images=) 成功，返回 keys: {list(inputs.keys())}")

        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        print(f"✅ 输入已移到设备 {device}，input token 长度: {input_length}")

        print("正在生成回答（可能需要几秒到几十秒）...")
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        new_tokens = generated[0][input_length:]
        answer = processor.decode(new_tokens, skip_special_tokens=True)

        print(f"\n{'='*60}")
        print("✅✅✅ 全部通过！模型回答:")
        print(f"{'='*60}")
        print(answer)

    except Exception:
        print("❌ 推理过程失败：")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
