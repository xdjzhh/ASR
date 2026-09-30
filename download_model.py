"""
模型预下载脚本（sherpa-onnx 版本）
首次运行会自动下载模型，也可通过此脚本预下载
"""
import os
import sys
import argparse


def download_models(mode: str = "tiny"):
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    model_info = {
        "tiny": ("SenseVoice INT8", "约 230MB", "SenseVoice 中文优先（5语种）"),
        "mini": ("SenseVoice INT8", "约 230MB", "SenseVoice 多语种"),
        "standard": ("Paraformer-zh INT8", "约 227MB", "Paraformer 中文精度最高"),
        "paraformer": ("Paraformer-zh INT8", "约 227MB", "Paraformer 纯ASR"),
    }

    print("=" * 60)
    print("VocoType ASR - 模型下载工具 (sherpa-onnx)")
    print("=" * 60)
    if mode in model_info:
        model_id, size, desc = model_info[mode]
        print(f"模式: {mode} - {desc}")
        print(f"模型: {model_id}")
        print(f"大小: {size}")
    print(f"模型目录: {models_dir}")
    print()

    # 检查已有模型
    existing = [f for f in os.listdir(models_dir) if f != '.gitkeep']
    if existing:
        print("检测到 models 目录已有内容:")
        for f in existing:
            print(f"   - {f}")
        print("   如需重新下载，请先删除 models 目录下对应文件夹")
        print()

    print("正在下载模型（首次运行）...")
    print()

    try:
        from asr_engine import get_asr_engine
        # 加载模型会触发自动下载
        engine = get_asr_engine(mode=mode)
        print()
        print(f"✅ 模型下载完成！模型目录: {engine.model_dir_name}")
        print()
        print("启动服务:")
        print(f"  python server.py --mode {mode}")

    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        print()
        print("可能的原因:")
        print("  1. 网络问题 - 请检查网络或稍后重试")
        print("  2. 磁盘空间不足 - 确保有足够空间")
        print("  3. sherpa-onnx 未安装 - 运行: pip install sherpa-onnx")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VocoType ASR 模型下载工具")
    parser.add_argument("--mode", type=str, default="tiny",
                        choices=["tiny", "mini", "standard", "paraformer"],
                        help="tiny(默认)/mini/standard/paraformer")
    args = parser.parse_args()
    download_models(mode=args.mode)
