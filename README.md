# 🎤 VocoType ASR - 本地语音转文字服务

基于 Paraformer 模型的本地离线语音识别服务，类似 VocoType 的核心能力。完全在本地运行，无需联网，保护隐私。支持中文、英文及中英文混合识别。

## ✨ 特性

- **100% 本地离线**：所有计算在本地完成，无需上传音频到云端
- **高精度识别**：基于阿里达摩院 Paraformer 模型，中文识别准确率高
- **自动标点**：内置标点符号预测，输出直接可读
- **VAD 静音检测**：自动跳过静音和杂音段落
- **数字格式化**：自动将语音数字转为阿拉伯数字
- **多格式支持**：支持 WAV, MP3, M4A, AAC, FLAC 等主流音频格式
- **Web 界面**：内置网页演示，支持麦克风录音和文件上传
- **HTTP API**：提供 RESTful API，方便集成到其他应用
- **GPU 加速**：支持 CUDA 加速（如有 NVIDIA 显卡）

## 📁 项目结构

```
ASR/
├── server.py          # FastAPI 服务主程序
├── asr_engine.py      # ASR 识别引擎封装
├── requirements.txt   # Python 依赖
├── start.bat          # Windows 一键启动脚本
├── test_client.py     # Python 测试客户端
├── static/
│   └── demo.html      # Web 演示页面
└── models/            # 模型存放目录（自动下载）
```

## 🚀 快速开始

### 环境要求

- Python 3.8 ~ 3.12（推荐 3.10 或 3.11）
- Windows / macOS / Linux
- 内存：至少 4GB RAM（纯 CPU 推理）
- 可选：NVIDIA 显卡 + CUDA（用于加速）

### 方式一：Windows 一键启动

双击 `start.bat` 即可自动安装依赖并启动服务。

### 方式二：手动启动

1. **安装依赖**

```bash
pip install -r requirements.txt
```

2. **启动服务**

```bash
# 默认 CPU 模式
python server.py

# 指定端口
python server.py --port 8080

# 使用 GPU 加速（需要 CUDA）
python server.py --device cuda:0

# 指定 CPU 线程数
python server.py --ncpu 8
```

3. **访问服务**

- 演示页面：http://localhost:8000/demo
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

首次启动会自动下载 Paraformer 模型（约 200-300MB），请耐心等待。

## 📡 API 接口

### 1. 健康检查

```http
GET /health
```

响应示例：
```json
{
  "status": "running",
  "model": "paraformer-zh",
  "device": "cpu"
}
```

### 2. 上传音频文件识别

```http
POST /transcribe
Content-Type: multipart/form-data
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | 音频文件 |
| use_itn | bool | 否 | 是否启用数字格式化，默认 true |
| batch_size_s | int | 否 | 长音频分段时长（秒），默认 300 |

响应示例：
```json
{
  "success": true,
  "text": "你好，欢迎使用语音转文字服务。",
  "sentences": [
    {"text": "你好，欢迎使用语音转文字服务。"}
  ],
  "message": "识别成功"
}
```

### 3. 通过本地文件路径识别

```http
POST /transcribe/path?file_path=/path/to/audio.wav
```

## 💻 使用示例

### curl 命令

```bash
# 识别音频文件
curl -X POST http://localhost:8000/transcribe ^
  -F "file=@test.wav"

# 禁用数字格式化
curl -X POST "http://localhost:8000/transcribe?use_itn=false" ^
  -F "file=@test.mp3"
```

### Python 调用

```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f}
    )
    result = response.json()
    print(result["text"])
```

### 使用测试客户端

```bash
# 检查服务状态
python test_client.py --check

# 识别音频文件
python test_client.py audio.wav
```

## 🌐 Web 演示

访问 http://localhost:8000/demo 可以使用网页演示界面：

- 🎙️ **麦克风录音**：点击麦克风按钮开始录音，停止后自动识别
- 📁 **文件上传**：拖拽或点击上传音频文件
- 📋 **一键复制**：识别结果支持一键复制

## 🔧 模型选择

默认使用 `paraformer-zh`（中文模型），如需其他模型：

```bash
# 非实时中文模型（默认，精度高）
python server.py --model paraformer-zh

# 流式模型（低延迟，适合实时场景）
python server.py --model paraformer-zh-streaming

# 英语模型
python server.py --model paraformer-en
```

## ❓ 常见问题

### Q: 模型下载慢怎么办？

A: 模型会自动从 ModelScope 下载。如果网络不佳，可以手动下载模型放到 `models/` 目录。

### Q: 识别速度慢？

A: 
- CPU 模式下，1小时音频约需 10-20 分钟（取决于 CPU 性能）
- 使用 GPU（CUDA）可提速 5-10 倍
- 增加 `--ncpu` 参数使用更多 CPU 线程

### Q: 支持哪些音频格式？

A: 支持 WAV, MP3, M4A, AAC, FLAC, OGG 等 libsndfile 支持的格式。音频会自动重采样到 16kHz。

### Q: 支持实时流式识别吗？

A: 当前版本支持文件和短音频识别。如需实时流式识别，可以使用 `paraformer-zh-streaming` 模型。

## 📝 与 VocoType 的关系

本项目使用与 VocoType 相同的底层技术栈（FunASR + Paraformer），提供类似的核心语音识别能力，但是以 API 服务的形式呈现，方便开发者集成到自己的应用中。VocoType 是一款优秀的桌面语音输入软件，如果你需要系统级的语音输入功能，推荐直接使用 VocoType。

## 📄 开源协议

本项目仅供学习和研究使用。Paraformer 模型遵循其相应的开源协议。
