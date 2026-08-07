"""
模型预下载脚本
使用 FunASR 内置别名，避免模型ID错误
"""
import os
import sys
import argparse


def download_models(mode: str = "tiny"):
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = models_dir
    
    model_info = {
        "tiny": ("iic/SenseVoiceSmall", "约 300MB", "SenseVoice 中文优先（最小）"),
        "mini": ("iic/SenseVoiceSmall + fsmn-vad", "约 350MB", "SenseVoice + VAD"),
        "standard": ("paraformer-zh + fsmn-vad + ct-punc", "约 400MB", "Paraformer + VAD + 标点（精度最高）"),
        "paraformer": ("paraformer-zh", "约 220MB", "Paraformer 纯ASR（无VAD/标点）"),
    }
    
    print("=" * 60)
    print("🎤 VocoType ASR - 模型下载工具")
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
        print("⚠️  检测到 models 目录已有文件:")
        for f in existing:
            print(f"   - {f}")
        print("   如需重新下载，请先删除 models 目录下对应文件夹")
        print()
    
    print("正在导入 FunASR...")
    try:
        from funasr import AutoModel
    except ImportError:
        print("❌ 错误: FunASR 未安装")
        print("请先运行: pip install -r requirements.txt")
        sys.exit(1)
    
    print()
    print("开始下载模型，请耐心等待...")
    print()
    
    try:
        kwargs = {"device": "cpu", "ncpu": 4, "disable_pbar": False}
        
        if mode == "tiny":
            kwargs["model"] = "iic/SenseVoiceSmall"
            kwargs["trust_remote_code"] = True
        elif mode == "mini":
            kwargs["model"] = "iic/SenseVoiceSmall"
            kwargs["vad_model"] = "fsmn-vad"
            kwargs["trust_remote_code"] = True
        elif mode == "standard":
            kwargs["model"] = "paraformer-zh"
            kwargs["vad_model"] = "fsmn-vad"
            kwargs["punc_model"] = "ct-punc"
        elif mode == "paraformer":
            kwargs["model"] = "paraformer-zh"
        
        print(f"配置: {kwargs}")
        model = AutoModel(**kwargs)
        
        print()
        print("✅ 模型下载完成！")
        print(f"📂 模型位置: {models_dir}")
        print()
        print("启动服务:")
        print(f"  python server.py --mode {mode}")
        
    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        print()
        print("可能的原因:")
        print("  1. 网络问题 - 请检查网络或稍后重试")
        print("  2. 磁盘空间不足 - 确保有足够空间")
        print("  3. 依赖问题 - 重新运行: pip install -U funasr modelscope")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VocoType ASR 模型下载工具")
    parser.add_argument("--mode", type=str, default="tiny",
                        choices=["tiny", "mini", "standard", "paraformer"],
                        help="tiny(最小~300MB,默认)/mini/standard/paraformer")
    args = parser.parse_args()
    download_models(mode=args.mode)
