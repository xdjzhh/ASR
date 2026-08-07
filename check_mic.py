"""检测麦克风设备和 sounddevice 安装"""
import sys
import importlib

def check_sounddevice():
    try:
        import sounddevice as sd
        print(f"✅ sounddevice 已安装，版本: {sd.__version__}")
        
        # 获取 PortAudio 版本
        try:
            pa_version = sd.get_portaudio_version()
            print(f"✅ PortAudio 版本: {pa_version}")
        except Exception as e:
            print(f"⚠️  PortAudio 信息获取失败: {e}")
        
        # 查询默认设备
        try:
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            print(f"默认输入设备ID: {default_input}")
            print(f"默认输出设备ID: {default_output}")
        except Exception as e:
            print(f"⚠️  默认设备获取失败: {e}")
            default_input = None
        
        # 列出所有设备
        print("\n可用音频设备:")
        print("-" * 70)
        devices = sd.query_devices()
        input_devices = []
        for i, dev in enumerate(devices):
            in_ch = dev['max_input_channels']
            out_ch = dev['max_output_channels']
            if in_ch > 0:
                marker = " ← 默认输入" if i == default_input else ""
                print(f"  [{i:2d}] 输入({in_ch}ch) 输出({out_ch}ch) | {dev['name']}{marker}")
                input_devices.append((i, dev['name']))
        
        if not input_devices:
            print("\n❌ 未检测到任何音频输入设备！")
            print("   请检查:")
            print("   1. 麦克风是否已连接")
            print("   2. Windows 隐私设置中是否允许应用访问麦克风")
            print("   3. 设备管理器中音频驱动是否正常")
        else:
            print(f"\n✅ 共检测到 {len(input_devices)} 个音频输入设备")
            
            # 尝试打开默认输入设备进行短暂录音测试
            print("\n正在测试麦克风录音（3秒）...")
            try:
                duration = 3  # 秒
                samplerate = 16000
                channels = 1
                
                print(f"   采样率: {samplerate}Hz, 声道: {channels}, 时长: {duration}s")
                
                import numpy as np
                recording = sd.rec(
                    int(duration * samplerate),
                    samplerate=samplerate,
                    channels=channels,
                    dtype='float32',
                    blocking=True
                )
                
                # 检查录音电平
                rms = np.sqrt(np.mean(recording ** 2))
                peak = np.max(np.abs(recording))
                
                print(f"   录音完成! RMS电平: {rms:.6f}, 峰值: {peak:.6f}")
                
                if peak < 0.001:
                    print("   ⚠️  录音信号极弱，请检查麦克风是否静音或音量是否太小")
                elif rms > 0.01:
                    print("   ✅ 麦克风输入正常！")
                else:
                    print("   ⚠️  检测到信号但电平较低，请对着麦克风说话测试")
                    
            except sd.PortAudioError as e:
                print(f"   ❌ 录音测试失败: {e}")
            except Exception as e:
                print(f"   ❌ 录音测试异常: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ sounddevice 未安装: {e}")
        print("\n请安装:")
        print("   pip install sounddevice")
        return False
    except Exception as e:
        print(f"❌ sounddevice 加载失败: {e}")
        print("\n可能需要安装 PortAudio 库:")
        print("   Windows: pip install sounddevice (通常自带)")
        print("   如果仍有问题，尝试: pip install pyaudiowpatch")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("🎤 VocoType ASR - 麦克风设备检测")
    print("=" * 70)
    print()
    
    ok = check_sounddevice()
    
    print()
    print("=" * 70)
    if ok:
        print("✅ 检测完成")
    else:
        print("❌ 存在问题，请根据上述提示修复")
        sys.exit(1)
