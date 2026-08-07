"""
实时麦克风监听器
- 持续监控麦克风输入
- VAD 语音活动检测
- 唤醒词检测（默认 "jason"）
- 唤醒后持续识别语音并转文字
"""
import os
import time
import threading
import queue
import numpy as np
import logging
from typing import Optional, Callable, List
from collections import deque

logger = logging.getLogger(__name__)


class MicrophoneListener:
    """
    实时麦克风监听器
    
    工作流程：
    1. 持续采集麦克风音频
    2. VAD 检测语音段
    3. 对语音段进行识别
    4. 检测唤醒词，唤醒后进入指令识别模式
    5. 通过回调函数推送识别结果
    """
    
    # 状态常量
    STATE_IDLE = "idle"           # 待机中，等待唤醒词
    STATE_LISTENING = "listening"  # 已唤醒，正在听指令
    STATE_SPEAKING = "speaking"    # 检测到说话，正在录音
    
    def __init__(
        self,
        asr_engine,
        wake_words: Optional[List[str]] = None,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype = np.float32,
        block_duration: float = 0.5,   # 每个音频块时长（秒）
        silence_threshold: float = 0.008,  # 静音阈值（RMS能量），值越大越不灵敏
        silence_duration: float = 1.0,    # 唤醒后多久静音认为说完了（秒）
        wake_silence_duration: float = 0.8,  # 待机时语音段最大间隔
        input_gain: float = 1.0,  # 输入音量增益（<1衰减，>1放大）
        device: Optional[int] = None,
        task_callback_url: Optional[str] = None,  # 识别结果回调URL，POST {"task": text}
    ):
        self.asr_engine = asr_engine
        self.wake_words = wake_words or ["jason", "杰森", "嘉森", "杰逊"]
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.block_size = int(sample_rate * block_duration)
        self.block_duration = block_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.wake_silence_duration = wake_silence_duration
        self.input_gain = input_gain
        self.device = device
        self.task_callback_url = task_callback_url or os.environ.get("TASK_CALLBACK_URL", None)
        
        # 当前RMS电平（用于UI显示）
        self.current_rms = 0.0
        
        self.state = self.STATE_IDLE
        self.running = False
        self.paused = False
        
        # 音频队列和缓冲区
        self.audio_queue: queue.Queue = queue.Queue()
        self.speech_buffer: List[np.ndarray] = []  # 当前语音段缓冲区
        
        # 唤醒后的语音缓冲区（完整指令）
        self.command_buffer: List[np.ndarray] = []
        self.silence_blocks = 0  # 连续静音块计数
        self.silence_blocks_threshold = int(silence_duration / block_duration)
        self.wake_silence_blocks_threshold = int(wake_silence_duration / block_duration)
        
        # 线程
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._stream = None
        
        # 回调函数
        self.on_wake_word: Optional[Callable[[str], None]] = None  # 检测到唤醒词
        self.on_partial: Optional[Callable[[str], None]] = None     # 实时部分结果
        self.on_result: Optional[Callable[[str], None]] = None      # 最终识别结果
        self.on_state_change: Optional[Callable[[str], None]] = None  # 状态变化
        self.on_error: Optional[Callable[[Exception], None]] = None    # 错误回调
        
        # 最近识别文本（用于唤醒词检测）
        self.recent_texts = deque(maxlen=5)
        
        # 唤醒词后需要忽略的前缀（唤醒词本身）
        self._wake_word_detected_at = 0
        self._wake_word_cooldown = 1.0  # 唤醒词检测后的冷却时间（秒）
    
    def _set_state(self, new_state: str):
        """切换状态"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            logger.info(f"状态变化: {old_state} -> {new_state}")
            if self.on_state_change:
                try:
                    self.on_state_change(new_state)
                except Exception as e:
                    logger.error(f"状态回调错误: {e}")
    
    def _is_silence(self, audio_block: np.ndarray) -> bool:
        """判断音频块是否为静音"""
        rms = np.sqrt(np.mean(audio_block ** 2))
        self.current_rms = float(rms)
        return rms < self.silence_threshold
    
    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice 音频采集回调（在音频线程中调用）"""
        if status:
            logger.warning(f"音频状态: {status}")
        
        if self.paused:
            return
        
        # 将音频数据放入队列（只做轻量操作，不阻塞）
        try:
            # 转单声道、float32
            if indata.ndim > 1:
                audio = indata[:, 0].astype(self.dtype).copy()
            else:
                audio = indata.astype(self.dtype).copy()
            
            # 应用输入音量增益
            if self.input_gain != 1.0:
                audio = np.clip(audio * self.input_gain, -1.0, 1.0)
            
            self.audio_queue.put_nowait(audio)
        except queue.Full:
            pass  # 队列满了就丢弃，避免阻塞音频线程
    
    def _recognize_audio(self, audio_chunks: List[np.ndarray]) -> str:
        """将累积的音频块拼接后进行识别"""
        if not audio_chunks:
            return ""
        
        audio = np.concatenate(audio_chunks)
        
        try:
            result = self.asr_engine.transcribe_waveform(audio)
            text = result.get("text", "").strip()
            return text
        except Exception as e:
            logger.error(f"识别失败: {e}")
            if self.on_error:
                try:
                    self.on_error(e)
                except:
                    pass
            return ""
    
    def _check_wake_word(self, text: str) -> Optional[str]:
        """检查文本中是否包含唤醒词，返回匹配的唤醒词"""
        text_lower = text.lower().replace(" ", "").replace("，", "").replace("。", "").replace(",", "").replace(".", "")
        for wake_word in self.wake_words:
            ww_lower = wake_word.lower().replace(" ", "")
            if ww_lower in text_lower:
                return wake_word
        return None
    
    def _process_loop(self):
        """音频处理主循环（在独立线程中运行）"""
        logger.info("音频处理线程启动")
        
        while self.running:
            try:
                # 获取音频块（超时1秒，便于检查running状态）
                try:
                    audio_block = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                is_silent = self._is_silence(audio_block)
                
                if self.state == self.STATE_IDLE:
                    # ========== 待机模式：持续监听唤醒词 ==========
                    if not is_silent:
                        # 检测到语音，开始累积
                        self.speech_buffer.append(audio_block)
                        self.silence_blocks = 0
                    else:
                        self.silence_blocks += 1
                        if self.speech_buffer:
                            self.speech_buffer.append(audio_block)
                        
                        # 静音足够长时间，处理累积的语音
                        if self.silence_blocks >= self.wake_silence_blocks_threshold and self.speech_buffer:
                            # 检查语音段长度（至少0.5秒）
                            total_duration = len(self.speech_buffer) * self.block_duration
                            if total_duration >= 0.4:
                                text = self._recognize_audio(self.speech_buffer)
                                if text:
                                    logger.debug(f"待机检测: {text}")
                                    matched_ww = self._check_wake_word(text)
                                    if matched_ww:
                                        # 检测到唤醒词！
                                        logger.info(f"🎤 检测到唤醒词: {matched_ww}")
                                        
                                        # 从文本中去掉唤醒词，得到后续指令前缀
                                        command_text = text
                                        for ww in self.wake_words:
                                            command_text = command_text.lower().replace(ww.lower(), "", 1).strip()
                                            command_text = command_text.lstrip("，。,. ")
                                        
                                        self._set_state(self.STATE_LISTENING)
                                        self.command_buffer = []
                                        self.silence_blocks = 0
                                        
                                        if self.on_wake_word:
                                            try:
                                                self.on_wake_word(matched_ww)
                                            except Exception as e:
                                                logger.error(f"唤醒词回调错误: {e}")
                                        
                                        # 如果唤醒词后面已经跟着指令了
                                        if command_text:
                                            self.command_buffer.extend(self.speech_buffer)
                                            if self.on_partial:
                                                try:
                                                    self.on_partial(command_text)
                                                except:
                                                    pass
                            
                            # 清空缓冲区
                            self.speech_buffer = []
                            self.silence_blocks = 0
                
                elif self.state == self.STATE_LISTENING:
                    # ========== 唤醒后：监听完整指令 ==========
                    self.command_buffer.append(audio_block)
                    
                    if is_silent:
                        self.silence_blocks += 1
                    else:
                        self.silence_blocks = 0
                    
                    # 静音持续一段时间，认为指令说完了
                    if self.silence_blocks >= self.silence_blocks_threshold:
                        total_duration = len(self.command_buffer) * self.block_duration
                        if total_duration >= 0.3:
                            text = self._recognize_audio(self.command_buffer)
                            if text:
                                # 去掉可能残留的唤醒词
                                for ww in self.wake_words:
                                    text = text.replace(ww, "").replace(ww.lower(), "").strip()
                                    text = text.lstrip("，。,. ")
                                
                                if text:
                                    logger.info(f"📝 指令识别: {text}")
                                    if self.on_result:
                                        try:
                                            self.on_result(text)
                                        except Exception as e:
                                            logger.error(f"结果回调错误: {e}")
                                    # 调用 HTTP 回调接口
                                    if self.task_callback_url:
                                        self._post_task_callback(text)
                        
                        # 回到待机状态
                        self.command_buffer = []
                        self.speech_buffer = []
                        self.silence_blocks = 0
                        self._set_state(self.STATE_IDLE)
                
            except Exception as e:
                logger.error(f"处理循环错误: {e}", exc_info=True)
                if self.on_error:
                    try:
                        self.on_error(e)
                    except:
                        pass
                time.sleep(0.1)
    
    def start(self):
        """启动麦克风监听"""
        if self.running:
            logger.warning("监听器已在运行")
            return
        
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("请先安装 sounddevice: pip install sounddevice")
        
        logger.info("启动麦克风监听器...")
        logger.info(f"唤醒词: {self.wake_words}")
        logger.info(f"采样率: {self.sample_rate}Hz, 块大小: {self.block_duration}s")
        
        self.running = True
        self.paused = False
        
        # 列出音频设备
        logger.info("可用音频设备:")
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    marker = " ← 默认输入" if i == default_input else ""
                    logger.info(f"  [{i}] {dev['name']}{marker}")
        except Exception as e:
            logger.warning(f"无法列出音频设备: {e}")
        
        # 启动音频流
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.block_size,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info("麦克风音频流已启动")
        except Exception as e:
            self.running = False
            raise RuntimeError(f"无法打开麦克风: {e}\n请检查麦克风是否连接并授权。")
        
        # 启动处理线程
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        self._set_state(self.STATE_IDLE)
        logger.info("✅ 麦克风监听器已启动，等待唤醒词...")
    
    def stop(self):
        """停止麦克风监听"""
        if not self.running:
            return
        
        logger.info("停止麦克风监听器...")
        self.running = False
        
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.error(f"关闭音频流错误: {e}")
            self._stream = None
        
        if self._process_thread:
            self._process_thread.join(timeout=3.0)
            self._process_thread = None
        
        # 清空队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        self.command_buffer = []
        self.speech_buffer = []
        logger.info("麦克风监听器已停止")
    
    def pause(self):
        """暂停监听"""
        self.paused = True
        logger.info("麦克风监听已暂停")
    
    def resume(self):
        """恢复监听"""
        self.paused = False
        logger.info("麦克风监听已恢复")
    
    def set_wake_words(self, wake_words: List[str]):
        """设置唤醒词"""
        self.wake_words = wake_words
        logger.info(f"唤醒词已更新: {wake_words}")
    
    def set_sensitivity(self, level: float):
        """
        设置灵敏度等级 (0.0 - 1.0)
        - 0.0: 最不灵敏（需要很大声才触发，适合嘈杂环境）
        - 0.5: 默认（普通说话音量）
        - 1.0: 最灵敏（小声说话也能触发，适合安静环境）
        """
        level = max(0.0, min(1.0, level))
        # 将 0-1 映射到 silence_threshold: 
        # level=1.0 -> threshold=0.003 (很灵敏，耳语也能识别)
        # level=0.5 -> threshold=0.012 (默认，普通说话)
        # level=0.0 -> threshold=0.04  (很不灵敏，需要大声)
        self.silence_threshold = 0.04 - level * 0.037
        logger.info(f"灵敏度设置为 {level:.2f}, 静音阈值: {self.silence_threshold:.4f}")
    
    def set_volume_gain(self, gain: float):
        """
        设置输入音量增益
        - 0.5: 音量减半（输入太大声时）
        - 1.0: 原始音量（默认）
        - 2.0: 音量加倍（麦克风太小声时）
        """
        gain = max(0.1, min(5.0, gain))
        self.input_gain = gain
        logger.info(f"音量增益设置为: {gain:.2f}")
    
    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            "running": self.running,
            "paused": self.paused,
            "state": self.state,
            "wake_words": self.wake_words,
            "silence_threshold": self.silence_threshold,
            "input_gain": self.input_gain,
            "current_rms": self.current_rms,
            "task_callback_url": self.task_callback_url,
        }
    
    def _post_task_callback(self, text: str):
        """
        在后台线程中发送 HTTP POST 回调，不阻塞音频处理
        POST {task_callback_url}  body: {"task": text}
        """
        url = self.task_callback_url
        if not url:
            return
        
        def _do_post():
            try:
                import urllib.request
                import json as _json
                
                payload = _json.dumps({"task": text}, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json; charset=utf-8"},
                    method="POST",
                )
                # 设置超时，避免长时间阻塞
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resp_body = resp.read().decode("utf-8", errors="replace")
                    logger.info(f"📤 任务回调已发送 [{resp.status}]: {text} -> {url}")
                    if resp_body:
                        logger.debug(f"回调响应: {resp_body[:200]}")
            except Exception as e:
                logger.error(f"❌ 任务回调失败 ({url}): {e}")
        
        # 在独立守护线程中发送，不阻塞音频处理循环
        t = threading.Thread(target=_do_post, daemon=True, name="task-callback")
        t.start()


# ============ 便捷函数：命令行直接运行麦克风监听 ============

def run_standalone():
    """独立运行模式：命令行麦克风监听"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="VocoType 实时麦克风监听（唤醒词模式）")
    parser.add_argument("--wake-word", type=str, default="jason", help="唤醒词（默认: jason）")
    parser.add_argument("--mode", type=str, default="tiny", help="ASR模型模式")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备")
    parser.add_argument("--mic-device", type=int, default=None, help="麦克风设备ID")
    parser.add_argument("--callback-url", type=str, default=None,
                        help="识别结果回调URL（默认 http://localhost:8080/run）")
    args = parser.parse_args()
    
    # 设置回调URL环境变量
    callback_url = args.callback_url or os.environ.get("TASK_CALLBACK_URL", "http://localhost:8080/run")
    os.environ["TASK_CALLBACK_URL"] = callback_url
    
    print("=" * 60)
    print("🎤 VocoType - 实时麦克风监听模式")
    print("=" * 60)
    
    # 加载 ASR 引擎
    print(f"正在加载 ASR 模型 (mode={args.mode})...")
    from asr_engine import get_asr_engine
    asr = get_asr_engine(mode=args.mode, device=args.device)
    print("✅ ASR 模型加载完成")
    print()
    
    # 唤醒词列表（包含唤醒词本身和常见变体）
    wake_words = [args.wake_word]
    # 为 "jason" 添加常见中文变体
    if args.wake_word.lower() == "jason":
        wake_words.extend(["杰森", "嘉森", "杰逊"])
    
    listener = MicrophoneListener(
        asr_engine=asr,
        wake_words=wake_words,
        device=args.mic_device,
        task_callback_url=callback_url,
    )
    
    # 设置回调
    def on_wake(word):
        print(f"\n🎤 听到唤醒词 [{word}]，请说指令...")
    
    def on_result(text):
        print(f"📝 识别结果: {text}")
        print(f"   （继续说唤醒词 {wake_words} 可再次触发）")
        print()
    
    def on_state(state):
        state_emoji = {
            "idle": "💤",
            "listening": "👂",
            "speaking": "🗣️",
        }
        emoji = state_emoji.get(state, "❓")
    
    def on_error(e):
        print(f"❌ 错误: {e}")
    
    listener.on_wake_word = on_wake
    listener.on_result = on_result
    listener.on_state_change = on_state
    listener.on_error = on_error
    
    print(f"唤醒词: {wake_words}")
    print(f"任务回调: {callback_url}")
    print("提示: 对着麦克风说唤醒词即可开始识别")
    print("按 Ctrl+C 退出")
    print("=" * 60)
    print()
    
    try:
        listener.start()
        
        # 主线程等待
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\n正在退出...")
    finally:
        listener.stop()
        print("已退出")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    run_standalone()
