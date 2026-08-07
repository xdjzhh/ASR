"""
VocoType ASR 服务测试客户端
使用方法: python test_client.py <音频文件路径>
"""
import sys
import requests
import os
import argparse


def transcribe_file(server_url: str, file_path: str, use_itn: bool = True):
    """
    调用 ASR 服务识别音频文件
    
    Args:
        server_url: 服务地址，如 http://localhost:8000
        file_path: 音频文件路径
        use_itn: 是否启用数字转写
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return None
    
    url = f"{server_url.rstrip('/')}/transcribe"
    
    print(f"正在上传文件: {file_path}")
    print(f"服务地址: {url}")
    print("-" * 60)
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            params = {"use_itn": use_itn}
            
            response = requests.post(url, files=files, params=params, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            if result["success"]:
                print("✅ 识别成功！")
                print("-" * 60)
                print("识别结果:")
                print(result["text"])
                print("-" * 60)
                
                if result.get("sentences"):
                    print("句子详情:")
                    for i, sent in enumerate(result["sentences"], 1):
                        ts = sent.get("timestamp", "")
                        if ts:
                            print(f"  [{i}] {ts}: {sent['text']}")
                        else:
                            print(f"  [{i}] {sent['text']}")
                
                return result["text"]
            else:
                print(f"❌ 识别失败: {result.get('message', '未知错误')}")
        else:
            print(f"❌ 请求失败: HTTP {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务，请确认服务已启动")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    return None


def check_health(server_url: str):
    """检查服务健康状态"""
    url = f"{server_url.rstrip('/')}/health"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ 服务运行中")
            print(f"   模型: {data['model']}")
            print(f"   设备: {data['device']}")
            return True
        else:
            print("❌ 服务异常")
            return False
    except:
        print("❌ 无法连接到服务")
        return False


def main():
    parser = argparse.ArgumentParser(description="VocoType ASR 测试客户端")
    parser.add_argument("file", nargs="?", help="要识别的音频文件路径")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="服务地址")
    parser.add_argument("--no-itn", action="store_true", help="禁用数字转写")
    parser.add_argument("--check", action="store_true", help="仅检查服务状态")
    
    args = parser.parse_args()
    
    print("🎤 VocoType ASR 测试客户端")
    print("=" * 60)
    
    if args.check or not args.file:
        check_health(args.server)
        if not args.file:
            print("\n使用方法: python test_client.py <音频文件路径>")
            print("示例: python test_client.py test.wav")
            return
    
    if args.file:
        transcribe_file(args.server, args.file, use_itn=not args.no_itn)


if __name__ == "__main__":
    main()
