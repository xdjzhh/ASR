#!/usr/bin/env python3
"""
VocoType ASR 本地语音转文字服务
基于 Paraformer/SenseVoice (FunASR) 模型
提供 HTTP API 和 WebSocket 实时识别
"""
# ===== 在导入任何库之前，先设置环境变量，加速启动并禁止联网检查 =====
import os
import sys

# === 内存优化：限制 PyTorch/OpenMP/MKL 线程数，防止内存爆炸 ===
cpu_count = os.cpu_count() or 4
_opt_threads = str(min(4, cpu_count))  # 最多4线程，足够CPU推理且大幅减少内存
os.environ.setdefault("OMP_NUM_THREADS", _opt_threads)
os.environ.setdefault("MKL_NUM_THREADS", _opt_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _opt_threads)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _opt_threads)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _opt_threads)
os.environ.setdefault("KMP_BLOCKTIME", "0")  # 减少Intel OpenMP空闲等待内存
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
# 禁用 PyTorch 的 oneDNN 自动调优（减少临时内存）
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX2")  # 限制最高指令集，减少内部workspace

# 模型缓存目录
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
# 离线模式（模型已下载后使用，禁止联网检查更新）
os.environ.setdefault("MODELSCOPE_OFFLINE", "0")  # 启动时先允许加载，模型加载后自动设为1
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
# 减少日志输出
os.environ.setdefault("MODELSCOPE_LOG_LEVEL", "40")  # ERROR级别
os.environ.setdefault("FUNASR_VERBOSE", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 避免tokenizers警告

# ===== 标准库 =====
import json
import asyncio
import threading
import time
from typing import Optional, List
from contextlib import asynccontextmanager

# ===== Web框架 =====
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 全局状态
asr_engine: Optional["ASREngine"] = None
mic_listener = None
engine_ready = False
engine_loading = False
engine_error: Optional[str] = None
ws_clients: List[WebSocket] = []
ws_loop = None


# ===== 生命周期管理：先启动HTTP服务，再后台加载模型 =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ws_loop
    ws_loop = asyncio.get_running_loop()
    
    # 启动定期状态广播
    broadcast_task = asyncio.create_task(periodic_status_broadcast())
    
    # 后台线程加载模型（不阻塞HTTP服务启动）
    def bg_load():
        global asr_engine, engine_ready, engine_loading, engine_error, mic_listener
        
        try:
            device = os.environ.get("ASR_DEVICE", "cpu")
            ncpu = int(os.environ.get("ASR_NCPU", "4"))
            mode = os.environ.get("ASR_MODE", "tiny")
            
            broadcast_sync({"type": "loading", "stage": "init", "message": f"正在初始化 ASR 引擎 ({mode})..."})
            logger.info(f"正在初始化 ASR 引擎，模式: {mode}, 设备: {device}")
            
            # 延迟导入 ASR 引擎（避免启动时就加载 torch/funasr）
            from asr_engine import ASREngine, get_asr_engine
            asr_engine = get_asr_engine(mode=mode, device=device, ncpu=ncpu)
            
            engine_ready = True
            engine_loading = False
            logger.info("✅ ASR 引擎初始化完成！")
            broadcast_sync({"type": "ready", "message": "ASR引擎已就绪"})
            
            # 模型加载完成后设置离线模式
            os.environ["MODELSCOPE_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            # 自动启动麦克风监听
            auto_mic = os.environ.get("AUTO_START_MIC", "1") != "0"
            if auto_mic:
                logger.info("正在自动启动麦克风监听...")
                broadcast_sync({"type": "loading", "stage": "mic", "message": "正在启动麦克风..."})
                try:
                    from microphone_listener import MicrophoneListener
                    
                    default_ww = os.environ.get("DEFAULT_WAKE_WORD", "jason")
                    wake_words = [default_ww]
                    if default_ww.lower() == "jason":
                        wake_words.extend(["杰森", "嘉森", "杰逊"])
                    
                    # 读取指定的麦克风设备
                    mic_device = os.environ.get("MIC_DEVICE")
                    mic_device_idx = int(mic_device) if mic_device and mic_device.isdigit() else None
                    
                    listener = MicrophoneListener(
                        asr_engine=asr_engine,
                        wake_words=wake_words,
                        device=mic_device_idx,
                        task_callback_url=os.environ.get("TASK_CALLBACK_URL"),
                    )
                    
                    listener.on_wake_word = lambda word: broadcast_sync({"type": "wake", "wake_word": word})
                    listener.on_result = lambda text: broadcast_sync({"type": "result", "text": text})
                    listener.on_partial = lambda text: broadcast_sync({"type": "partial", "text": text})
                    listener.on_state_change = lambda state: broadcast_sync({"type": "state", "state": state})
                    listener.on_error = lambda e: broadcast_sync({"type": "error", "message": str(e)})
                    
                    mic_listener = listener
                    listener.start()
                    logger.info("✅ 麦克风监听已自动启动，等待唤醒词...")
                    broadcast_sync({"type": "mic_started", **listener.get_status()})
                except Exception as e:
                    logger.error(f"自动启动麦克风失败: {e}")
                    broadcast_sync({"type": "error", "message": f"麦克风启动失败: {e}"})
            
        except Exception as e:
            engine_error = str(e)
            engine_loading = False
            logger.error(f"❌ ASR 引擎初始化失败: {e}")
            broadcast_sync({"type": "error", "message": f"引擎加载失败: {e}"})
    
    engine_loading = True
    threading.Thread(target=bg_load, daemon=True).start()
    
    yield  # HTTP 服务在这里运行
    
    # 关闭时清理
    broadcast_task.cancel()
    if mic_listener and mic_listener.running:
        mic_listener.stop()


app = FastAPI(
    title="VocoType ASR Service - 本地语音转文字服务",
    description="基于 Paraformer/SenseVoice 模型的本地离线语音识别服务，支持文件上传、麦克风实时监听、唤醒词检测",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ===== Pydantic 模型 =====
class TranscribeResponse(BaseModel):
    text: str
    duration: float
    model: str
    mode: str


class HealthResponse(BaseModel):
    status: str
    model: Optional[str] = None
    mode: Optional[str] = None
    device: Optional[str] = None
    engine_ready: bool = False
    engine_error: Optional[str] = None
    microphone_running: bool = False


class WakeWordRequest(BaseModel):
    words: List[str]


class SensitivityRequest(BaseModel):
    level: float


class VolumeRequest(BaseModel):
    gain: float


class DeviceSelectRequest(BaseModel):
    device_index: Optional[int] = None  # None = 使用默认设备


class MicControlRequest(BaseModel):
    device: Optional[int] = None
    samplerate: Optional[int] = None
    blocksize: Optional[int] = None
    wake_words: Optional[List[str]] = None
    silence_duration: Optional[float] = None
    silence_threshold: Optional[float] = None


# ===== WebSocket 辅助函数 =====
def run_broadcast(coro_data):
    """在主线程事件循环中安全执行广播"""
    if ws_loop is None or not ws_clients:
        return
    asyncio.run_coroutine_threadsafe(broadcast(coro_data), ws_loop)


def broadcast_sync(data):
    """同步版本广播（供非async线程调用）"""
    run_broadcast(data)


async def broadcast(data: dict):
    """向所有 WebSocket 客户端广播消息"""
    data["timestamp"] = time.time()
    dead_clients = []
    for client in ws_clients:
        try:
            await client.send_json(data)
        except:
            dead_clients.append(client)
    for c in dead_clients:
        if c in ws_clients:
            ws_clients.remove(c)


async def periodic_status_broadcast():
    """定期广播麦克风音量电平（用于前端电平表）"""
    global mic_listener
    while True:
        await asyncio.sleep(0.2)  # 5Hz
        if mic_listener and mic_listener.running and not mic_listener.paused:
            status = mic_listener.get_status()
            await broadcast({
                "type": "audio_level",
                "rms": status.get("current_rms", 0),
                "state": status.get("state", "idle"),
                "threshold": status.get("silence_threshold", 0.012),
            })


# ===== API 路由 =====
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/demo")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    mic_running = mic_listener is not None and mic_listener.running if mic_listener else False
    
    if engine_error:
        return HealthResponse(
            status="error",
            engine_ready=False,
            engine_error=engine_error,
            microphone_running=mic_running,
        )
    if not engine_ready:
        return HealthResponse(
            status="loading",
            engine_ready=False,
            microphone_running=mic_running,
        )
    
    return HealthResponse(
        status="ok",
        model=getattr(asr_engine, 'model_name', 'unknown'),
        mode=getattr(asr_engine, 'mode', 'unknown'),
        device=getattr(asr_engine, 'device', 'cpu'),
        engine_ready=True,
        microphone_running=mic_running,
    )


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """上传音频文件进行语音转文字"""
    if not engine_ready:
        raise HTTPException(status_code=503, detail="ASR 引擎尚未就绪，请稍候...")
    
    start_time = time.time()
    
    try:
        audio_bytes = await file.read()
        text = asr_engine.transcribe_bytes(audio_bytes)
        duration = time.time() - start_time
        
        return TranscribeResponse(
            text=text,
            duration=duration,
            model=asr_engine.model_name,
            mode=asr_engine.mode,
        )
    except Exception as e:
        logger.error(f"识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


@app.get("/demo")
async def demo_page():
    """演示页面"""
    demo_path = os.path.join(static_dir, "demo.html")
    if os.path.exists(demo_path):
        with open(demo_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>演示页面未找到</h1>")


# ===== 麦克风监听 API =====
def _get_mic_listener() -> "MicrophoneListener":
    """获取或创建麦克风监听器"""
    global mic_listener
    
    if mic_listener is None:
        if not engine_ready:
            raise HTTPException(status_code=503, detail="ASR 引擎尚未就绪")
        from microphone_listener import MicrophoneListener
        default_ww = os.environ.get("DEFAULT_WAKE_WORD", "jason")
        wake_words = [default_ww]
        if default_ww.lower() == "jason":
            wake_words.extend(["杰森", "嘉森", "杰逊"])
        
        # 读取指定的麦克风设备（环境变量）
        mic_device = os.environ.get("MIC_DEVICE")
        mic_device_idx = int(mic_device) if mic_device and mic_device.isdigit() else None
        
        listener = MicrophoneListener(
            asr_engine=asr_engine,
            wake_words=wake_words,
            device=mic_device_idx,
            task_callback_url=os.environ.get("TASK_CALLBACK_URL"),
        )
        
        listener.on_wake_word = lambda word: run_broadcast({"type": "wake", "wake_word": word})
        listener.on_result = lambda text: run_broadcast({"type": "result", "text": text})
        listener.on_partial = lambda text: run_broadcast({"type": "partial", "text": text})
        listener.on_state_change = lambda state: run_broadcast({"type": "state", "state": state})
        listener.on_error = lambda e: run_broadcast({"type": "error", "message": str(e)})
        
        mic_listener = listener
    
    return mic_listener


@app.post("/mic/start")
async def start_mic(req: MicControlRequest = None):
    """启动麦克风监听"""
    listener = _get_mic_listener()
    if listener.running:
        return {"success": True, "message": "麦克风已在监听中", "status": listener.get_status()}
    
    try:
        # 如果指定了设备，先设置
        if req and req.device is not None:
            ok = listener.set_device(req.device)
            if not ok:
                raise HTTPException(status_code=400, detail=f"无效的麦克风设备ID: {req.device}")
        if req and req.wake_words is not None:
            listener.set_wake_words(req.wake_words)
        if req and req.silence_threshold is not None:
            listener.silence_threshold = req.silence_threshold
        
        def start_sync():
            listener.start()
        
        await asyncio.get_event_loop().run_in_executor(None, start_sync)
        return {"success": True, "message": "麦克风监听已启动", "status": listener.get_status()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动麦克风失败: {e}")


@app.post("/mic/stop")
async def stop_mic():
    """停止麦克风监听"""
    global mic_listener
    if not mic_listener or not mic_listener.running:
        return {"success": True, "message": "麦克风未在监听"}
    
    try:
        mic_listener.stop()
        return {"success": True, "message": "麦克风监听已停止"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"停止失败: {e}")


@app.post("/mic/pause")
async def pause_mic():
    """暂停监听（不关闭设备）"""
    if not mic_listener or not mic_listener.running:
        raise HTTPException(status_code=400, detail="麦克风未启动")
    mic_listener.pause()
    return {"success": True, "message": "已暂停"}


@app.post("/mic/resume")
async def resume_mic():
    """恢复监听"""
    if not mic_listener:
        raise HTTPException(status_code=400, detail="麦克风未启动")
    mic_listener.resume()
    return {"success": True, "message": "已恢复"}


@app.get("/mic/status")
async def mic_status():
    """获取麦克风监听状态"""
    if not mic_listener:
        return {"running": False, "paused": False, "state": "stopped", "wake_words": ["jason"]}
    return mic_listener.get_status()


@app.post("/mic/wake-words")
async def set_wake_words(req: WakeWordRequest):
    """设置唤醒词"""
    listener = _get_mic_listener()
    listener.set_wake_words(req.words)
    return {"success": True, "wake_words": req.words}


@app.post("/mic/sensitivity")
async def set_sensitivity(req: SensitivityRequest):
    """设置麦克风灵敏度 (0.0-1.0)"""
    listener = _get_mic_listener()
    listener.set_sensitivity(req.level)
    return {"success": True, "level": req.level, "threshold": listener.silence_threshold}


@app.post("/mic/volume")
async def set_volume(req: VolumeRequest):
    """设置输入音量增益 (0.1-5.0)"""
    listener = _get_mic_listener()
    listener.set_volume_gain(req.gain)
    return {"success": True, "gain": req.gain}


@app.get("/mic/devices")
async def list_audio_devices():
    """列出可用音频设备（麦克风）"""
    try:
        from microphone_listener import MicrophoneListener
        input_devices = MicrophoneListener.list_devices()
        default_idx = next((d["index"] for d in input_devices if d["is_default"]), None)
        return {"devices": input_devices, "default": default_idx}
    except ImportError:
        raise HTTPException(status_code=500, detail="sounddevice 未安装，请运行: pip install sounddevice")


@app.post("/mic/device")
async def select_mic_device(req: DeviceSelectRequest):
    """选择/切换麦克风设备（运行时切换，无需重启）"""
    listener = _get_mic_listener()
    try:
        ok = listener.set_device(req.device_index)
        if ok:
            status = listener.get_status()
            return {"success": True, "message": f"已切换到设备 {req.device_index}", "status": status}
        else:
            raise HTTPException(status_code=400, detail=f"无法切换到设备 {req.device_index}，设备无效或不可用")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"切换设备失败: {e}")


# ===== WebSocket 实时识别 =====
@app.websocket("/ws/mic")
async def websocket_mic_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.append(websocket)
    
    # 连接后立即发送当前状态
    try:
        status = {
            "type": "status",
            "engine_ready": engine_ready,
            "engine_error": engine_error,
            "mic_running": mic_listener.running if mic_listener else False,
        }
        if mic_listener:
            status.update(mic_listener.get_status())
        await websocket.send_json(status)
    except:
        pass
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            action = msg.get("action")
            
            if action == "start":
                try:
                    listener = _get_mic_listener()
                    if not listener.running:
                        # 如果指定了设备，先设置
                        if "device" in msg and msg["device"] is not None:
                            ok = listener.set_device(msg["device"])
                            if not ok:
                                await websocket.send_json({"type": "error", "message": f"无效的设备ID: {msg['device']}"})
                                continue
                        if msg.get("wake_words"):
                            listener.set_wake_words(msg["wake_words"])
                        def start_sync():
                            listener.start()
                        await asyncio.get_event_loop().run_in_executor(None, start_sync)
                    await websocket.send_json({"type": "started", **listener.get_status()})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
            
            elif action == "stop":
                if mic_listener:
                    mic_listener.stop()
                    await websocket.send_json({"type": "stopped"})
            
            elif action == "set_device":
                # WebSocket切换麦克风设备
                try:
                    from microphone_listener import MicrophoneListener
                    device_idx = msg.get("device_index")
                    listener = _get_mic_listener()
                    ok = listener.set_device(device_idx)
                    if ok:
                        await websocket.send_json({"type": "device_changed", **listener.get_status()})
                    else:
                        await websocket.send_json({"type": "error", "message": f"切换设备失败: {device_idx}"})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
            
            elif action == "get_devices":
                # WebSocket获取设备列表
                try:
                    from microphone_listener import MicrophoneListener
                    devices = MicrophoneListener.list_devices()
                    await websocket.send_json({"type": "devices", "devices": devices})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
            
            elif action == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)


# ===== 命令行入口 =====
def main():
    parser = argparse.ArgumentParser(description="VocoType ASR 本地语音转文字服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8081, help="服务端口")
    parser.add_argument("--mode", type=str, default="tiny", 
                        choices=["tiny", "mini", "standard", "paraformer"], 
                        help="模型模式: tiny(SenseVoice~300MB,默认)/mini/standard(VAD+标点~400MB)/paraformer")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备 (cpu 或 cuda:0)")
    parser.add_argument("--ncpu", type=int, default=4, help="CPU 线程数")
    parser.add_argument("--wake-word", type=str, default="jason", help="默认唤醒词")
    parser.add_argument("--mic-device", type=int, default=None, help="指定麦克风设备ID（通过API /mic/devices 查看）")
    parser.add_argument("--no-auto-mic", action="store_true", help="启动时不自动开启麦克风监听")
    parser.add_argument("--callback-url", type=str, default=None, 
                        help="识别结果回调URL，POST {\"task\": text} 到此地址（默认 http://localhost:8080/run）")
    args = parser.parse_args()
    
    os.environ["ASR_MODE"] = args.mode
    os.environ["ASR_DEVICE"] = args.device
    os.environ["ASR_NCPU"] = str(args.ncpu)
    os.environ["AUTO_START_MIC"] = "0" if args.no_auto_mic else "1"
    os.environ["DEFAULT_WAKE_WORD"] = args.wake_word
    if args.mic_device is not None:
        os.environ["MIC_DEVICE"] = str(args.mic_device)
    if args.callback_url:
        os.environ["TASK_CALLBACK_URL"] = args.callback_url
    elif "TASK_CALLBACK_URL" not in os.environ:
        os.environ["TASK_CALLBACK_URL"] = "http://localhost:8080/run"
    
    mode_descs = {
        "tiny": "SenseVoiceSmall (~300MB)",
        "mini": "SenseVoice + VAD (~350MB)",
        "standard": "Paraformer-zh + VAD + 标点 (~400MB)",
        "paraformer": "Paraformer-zh 纯ASR",
    }
    mode_desc = mode_descs.get(args.mode, args.mode)
    
    print()
    print("=" * 60)
    print("🎤 VocoType ASR 本地语音转文字服务")
    print("=" * 60)
    print(f"模式: {mode_desc}")
    print(f"设备: {args.device}")
    print(f"默认唤醒词: {args.wake_word}")
    print(f"自动启动麦克风: {'是' if not args.no_auto_mic else '否'}")
    callback_url = os.environ.get("TASK_CALLBACK_URL", "")
    print(f"任务回调URL: {callback_url if callback_url else '（未设置）'}")
    print(f"监听: http://{args.host}:{args.port}")
    print(f"演示页面: http://localhost:{args.port}/demo")
    print(f"API 文档: http://localhost:{args.port}/docs")
    print("=" * 60)
    print("⏳ HTTP 服务正在启动...（模型在后台加载，可先打开页面）")
    print("=" * 60)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
