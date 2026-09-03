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
import difflib
import numpy as np
import logging
from typing import Optional, Callable, List, Tuple, Dict
from collections import deque

logger = logging.getLogger(__name__)

# 拼音模糊匹配为可选能力：pypinyin 缺失时自动降级为纯精确匹配
try:
    from pypinyin import lazy_pinyin, Style
    HAS_PYPINYIN = True
except Exception:
    HAS_PYPINYIN = False


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
        # 唤醒词拼音缓存（模糊匹配用，唤醒词更新时重建）
        self._wake_py: List[Dict] = []
        self._rebuild_wake_pinyin_cache()
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
        
        # 任务回调专用线程池（即发即弃：提交后立即返回，后台线程完成HTTP请求）
        from concurrent.futures import ThreadPoolExecutor
        self._callback_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="task-callback"
        )
        
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
    
    @staticmethod
    def _strip_punct(s: str) -> str:
        """去除中英文标点（保留英文单词之间的空格以便 ASCII 词边界判断），转小写、strip两端空白"""
        # 中文标点/符号（含全角空格）全删
        for ch in ("\t", "\n", "，", "。", ",", ".", "！", "!", "？", "?",
                   "、", "；", ";", "：", ":", "“", "”", "\"", "‘", "’", "'",
                   "（", "）", "(", ")", "…", "—", "-", "　"):
            s = s.replace(ch, "")
        # 英文半角空格：两端 strip，内部压缩但保留（词边界判断要用）
        s = " ".join(s.split()).lower()
        return s

    def _rebuild_wake_pinyin_cache(self):
        """预计算每个唤醒词的拼音特征（全拼/首字母/首音节/第二音节首字母/字数），唤醒词更新时调用"""
        self._wake_py = []
        if not HAS_PYPINYIN:
            return
        for ww in self.wake_words:
            feats = self._pinyin_features(ww)
            if not feats:
                continue
            entry = {
                "ww": ww,
                "full": "".join(f[1] for f in feats),
                "first": "".join(f[2] for f in feats),
                "syl0": feats[0][1],
                "n": len(feats),
                # 多字唤醒词：第二音节首字母（L≥2 时用来硬约束 R3，区分 s/sh/shen/shu/shi 类）
                # 例：杰森 → ['jie','sen'] → init1='s'; 嘉森 → jia+sen → init1='s'; 杰逊 → jie+xun → init1='x'
                "init1": feats[1][2] if len(feats) >= 2 else None,
            }
            self._wake_py.append(entry)

    @staticmethod
    def _pinyin_features(s: str) -> List[Tuple[str, str, str]]:
        """
        提取字符级拼音特征：[(原字符, 全拼无声调, 首字母)]
        - 汉字 → pypinyin 全拼/首字母（小写）
        - 英文字母 → 小写保留（兼容英文唤醒词如 jason）
        - 标点/数字/其他 → 跳过
        """
        feats: List[Tuple[str, str, str]] = []
        if not HAS_PYPINYIN or not s:
            return feats
        full_list = lazy_pinyin(s, style=Style.NORMAL, errors="default")
        first_list = lazy_pinyin(s, style=Style.FIRST_LETTER, errors="default")
        if len(full_list) != len(s) or len(first_list) != len(s):
            # 对齐异常时退化为逐字转换
            full_list = []
            first_list = []
            for ch in s:
                f = lazy_pinyin(ch, style=Style.NORMAL, errors="ignore")
                i = lazy_pinyin(ch, style=Style.FIRST_LETTER, errors="ignore")
                full_list.append(f[0] if f else "")
                first_list.append(i[0] if i else "")
        for ch, fpy, ipy in zip(s, full_list, first_list):
            if "\u4e00" <= ch <= "\u9fff":
                feats.append((ch, (fpy or "").lower(), (ipy or "").lower()))
            elif ch.isascii() and ch.isalpha():
                feats.append((ch, ch.lower(), ch.lower()))
        return feats

    def _fuzzy_match_pinyin(self, text_clean: str) -> Optional[Tuple[float, str, str, str]]:
        """
        拼音模糊匹配（仅在精确匹配未命中时调用）。

        ASR 常把唤醒词识别成近音/同音字，例如 "杰森"(jié sēn) 被识别成
        "接是"(jiē shì)、"节审"(jié shěn)、"杰深"(jié shēn) 等。
        对文本做与唤醒词字数相同的滑窗（首音节全拼必须一致才进入判定），
        按以下规则判定（取最高分）：
          规则1：窗口全拼 == 唤醒词全拼（严格同音字，如 借森→jiesen）
          规则2：全拼相似度 ≥ 0.80（强信号，允许跟指令连说，如 节审 jieshen ≈ 0.92）
          规则3：首字母序列吻合 + 全拼相似度 ≥ 0.7 且整段字数 ≤ 唤醒词字数+3
                 （如 接是打开灯 中「接是」jieshi：首音节 jie 相同、首字母 js 吻合、相似度 0.67）
        返回 (best, diag)：
          - best: (score, ww, win_text, win_full, win_start) | None  ——命中或未命中
          - diag: str | None  ——当 best==None 且存在 ≥ 0.50 的强候选时，给出一行「未命中原因 + 原文」的诊断日志文本，供调用方打 INFO。
        早退分支（无 pypinyin / 无唤醒词 / 无拼音特征）统一返回 (None, None)。
        """
        if not HAS_PYPINYIN or not self._wake_py:
            return None, None
        feats = self._pinyin_features(text_clean)
        if not feats:
            return None, None
        n = len(feats)
        best: Optional[Tuple[float, str, str, str]] = None
        # 用于未命中诊断：记录"首音节吻合但规则没通过"的最强候选（观察型日志用）
        top_near: Optional[Tuple[float, str, str, str, str]] = None  # (ratio, ww, win_text, win_full, why)
        for w in self._wake_py:
            # 仅取与唤醒词字数相同的窗口（字数不同的高相似度多为巧合，如「件 jian」≈「嘉森 jiasen」）
            L = w["n"]
            if L > n:
                continue
            # 规则1(R1:全拼完全相同，score=1.0) 与 规则2(R2:相似度≥0.80) 属于强信号：
            #   2个字符窗口的拼音≥0.80，说明至少5~6个字母匹配（如 jieshen≈jiesen），
            #   即使后面跟了完整指令（"节审今天天气"，段n=6）也应放行；
            #   之前 n<=L+2 的整段长度限制是"只说唤醒词"的假设，与用户"唤醒+指令连说"的
            #   真实输入形态矛盾，导致模糊匹配永无命中（bug）。
            allow_high = True
            # 规则3(R3:0.7≤相似度<0.80，专门匹配"接是 jieshi"这类首音节+首字母都吻合
            #   但元音差较大的弱信号)：放宽到 段长≤L+3，既能放过"接是打开灯"(n=5,L=2)，
            #   也能把"结实耐用"(n=6)、"请解释一下这个问题"(n=8) 等长句中的假阳性
            #   jieshi 类词继续拦在外面。
            allow_low = n <= L + 3
            for i in range(0, n - L + 1):
                win_text_i = "".join(f[0] for f in feats[i:i + L])
                # 窗口内是纯ASCII字母（英文 jason 这类）→ 走「ASCII词边界」；否则走中文句首+填充字约束
                ascii_win = self._is_ascii_word(win_text_i)
                if not self._wake_word_ok_at(text_clean, i, L, ascii_word=ascii_win):
                    continue
                win = feats[i:i + L]
                win_text = win_text_i
                win_full = "".join(f[1] for f in win)
                win_first = "".join(f[2] for f in win)
                # 首音节全拼必须一致（jie 开头的词不可能匹配 wen/jiu 开头的词）
                if win[0][1] != w["syl0"]:
                    continue
                score = 0.0
                why = ""
                ratio = difflib.SequenceMatcher(None, win_full, w["full"]).ratio()
                if win_full == w["full"] and allow_high:
                    score = 1.0  # 严格同音/近形字，如 借森→jiesen
                else:
                    if ratio >= 0.80 and allow_high:
                        score = ratio  # 高相似度，如 节审 jieshen vs jiesen ≈ 0.92
                    elif ratio >= 0.70 and allow_low:
                        first_ok = (
                            win_first == w["first"]
                            or w["first"] in win_first
                            or win_first in w["first"]
                        )
                        # R3 低置信门槛三重硬约束（专防 jieshi 形态的 结实/届时/节食/结识 误报）：
                        #   a) 首音节一致（jie）——已在 L285 之前的 continue 保证
                        #   b) 首字母序列吻合（如 接是 js == 杰森 js）
                        #   c) 多字唤醒词：窗口第二音节首字母 == 唤醒词第二音节首字母
                        #      （如 杰森 init1='s'，所有 shi/shu/sha...「sh声母」字在此全部被拦）
                        if first_ok:
                            if L >= 2 and w.get("init1") is not None:
                                win_init1 = win[1][2]  # 窗口第二字 FIRST_LETTER
                                if win_init1 != w["init1"]:
                                    why = (
                                        f"第二声母{win_init1}≠{w['init1']}"
                                        f"（候选「{win_text}」第二字为{win[1][1]}，"
                                        f"唤醒词「{w['ww']}」第二字拼音含{w['init1']}声母）"
                                    )
                                else:
                                    score = ratio
                            else:
                                score = ratio
                        else:
                            why = f"首字母{win_first}≠{w['first']}"
                    elif ratio >= 0.5:  # 未命中但接近，收集到 near miss
                        if ratio < 0.70:
                            why = f"相似{ratio:.2f}<0.70"
                        elif not allow_low:
                            why = f"段长{n}>L+3={L+3}"
                        else:
                            why = "首字母/第二声母不吻合"
                if score > 0.0 and (best is None or score > best[0]):
                    best = (score, w["ww"], win_text, win_full, i)
                elif score == 0.0 and ratio >= 0.50 and (
                    top_near is None or ratio > top_near[0]
                ):
                    top_near = (ratio, w["ww"], win_text, win_full, why or "")
        # 未命中但存在 ≥0.50 的 near-miss：返回 (None, near_miss_log_line) 给调用方打诊断日志
        diag = None
        if best is None and top_near is not None:
            r, ww, wt, wf, why = top_near
            diag = (
                f"候选「{wt}」({wf})≈「{ww}」 相似{r:.2f} 未通过原因: {why or '未知'}")
        return best, diag

    # 句首约束：用户唤醒时几乎总是先说唤醒词再说指令，
    # 极少出现在句中（"帮我问杰芬那个事情"）。允许唤醒词前最多 1 个单字语气词
    # （嗯/啊/哦/呃/哎/唉/唔），模拟"嗯杰森打开空调"这类真实起手。
    # 英文（如 jason）按词边界匹配，不参与句首限制（ASR 常把中文+英文连写成 HelloJason）。
    _FILLER_CHARS = frozenset("嗯啊哦呃哎唉唔")
    _FILLER_TOLERANCE = 1  # 允许的前导填充字数量

    @classmethod
    def _is_ascii_word(cls, s: str) -> bool:
        return bool(s) and all(("\x00" <= c <= "\x7f" and c.isalpha()) for c in s)

    @classmethod
    def _wake_word_ok_at(cls, text_clean: str, start: int, ww_len: int, *, ascii_word: bool = False) -> bool:
        """
        判断唤醒词窗口起始在 start 是否符合位置约束：
          - 英文（ascii_word=True）：按词边界（前后必须是文本边界或非字母）
          - 中文（ascii_word=False）：句首 or 前导字全是填充字 & 数量 ≤ TOLERANCE
        ww_len 仅用于完整性，实际判断由 start 主导。
        """
        end = start + ww_len
        if ascii_word:
            # 前边界：start==0 或 text_clean[start-1] 不是 ASCII 字母
            ok_bef = (start == 0) or not (
                "\x00" <= text_clean[start - 1] <= "\x7f" and text_clean[start - 1].isalpha()
            )
            ok_aft = (end >= len(text_clean)) or not (
                "\x00" <= text_clean[end] <= "\x7f" and text_clean[end].isalpha()
            )
            return ok_bef and ok_aft
        # 中文：句首 or 前导字都是填充字（数量 ≤ FILLER_TOLERANCE）
        if start == 0:
            return True
        if start > cls._FILLER_TOLERANCE:
            return False
        return all(text_clean[k] in cls._FILLER_CHARS for k in range(start))

    def _check_wake_word(self, text: str) -> Optional[str]:
        """检查文本中是否包含唤醒词，返回匹配的唤醒词（精确匹配优先，未命中走拼音模糊匹配）"""
        text_clean = self._strip_punct(text)
        # 1) 精确子串匹配（汉字/英文均适用）：
        #    中文必须句首/填充字起手；英文必须 ASCII 词边界（Hello Jason 中的 jason 可命中）
        for wake_word in self.wake_words:
            ww_clean = self._strip_punct(wake_word)
            if not ww_clean:
                continue
            ascii_ww = self._is_ascii_word(ww_clean)
            idx = text_clean.find(ww_clean)
            while idx >= 0:
                if self._wake_word_ok_at(text_clean, idx, len(ww_clean), ascii_word=ascii_ww):
                    return wake_word
                idx = text_clean.find(ww_clean, idx + 1)
        # 2) 拼音模糊匹配（同音/近音字，如 接是/节审/杰深 ≈ 杰森）
        if HAS_PYPINYIN:
            hit, diag = self._fuzzy_match_pinyin(text_clean)
            if hit is not None:
                score, ww, win_text, win_full, win_start = hit  # win_start 是窗口起始index
                ww_full = next((w["full"] for w in self._wake_py if w["ww"] == ww), "")
                logger.info(
                    f"🎯 拼音模糊匹配命中: 片段「{win_text}」({win_full}) ≈ "
                    f"唤醒词「{ww}」({ww_full})，相似度={score:.2f}，位置i={win_start}"
                )
                return ww
            if diag:
                logger.info(f"🔍 模糊匹配未命中: {diag}  原文: {text!r}")
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
                                seg_rms = float(np.sqrt(np.mean(
                                    np.concatenate(self.speech_buffer).astype(np.float32) ** 2)))
                                # 可观测性：待机段无论是否命中唤醒词，都打印 ASR 原文
                                logger.info(
                                    f"🎧 待机语音段转写 ({total_duration:.1f}s, "
                                    f"段RMS={seg_rms:.4f}, 静音阈值={self.silence_threshold:.4f}): {text!r}"
                                )
                                if text:
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
                            logger.info(f"🎧 指令语音段转写 ({total_duration:.1f}s): {text!r}")
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
        
        # 若线程池已被 stop() 关闭，则重建（支持停止后再次启动）
        if self._callback_executor is None or getattr(self._callback_executor, "_shutdown", False):
            from concurrent.futures import ThreadPoolExecutor
            self._callback_executor = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="task-callback"
            )
        
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
        
        # 未显式指定设备时，自动扫描并跳过"恒定满幅噪声"的坏设备/虚拟声卡
        if self.device is None:
            picked = self._auto_select_input_device()
            if picked is not None:
                self.device = picked

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
            logger.info(f"麦克风音频流已启动（设备 index={self.device}）")
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
        
        # 关闭回调线程池（不等待正在发送的回调，守护线程会随进程退出）
        try:
            self._callback_executor.shutdown(wait=False)
        except Exception:
            pass
        
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
        self._rebuild_wake_pinyin_cache()
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
    
    @staticmethod
    def list_devices() -> List[dict]:
        """列出所有可用的麦克风（输入）设备，返回设备列表"""
        try:
            import sounddevice as sd
        except ImportError:
            raise ImportError("请先安装 sounddevice: pip install sounddevice")
        
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        input_devices = []
        
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append({
                    "index": i,
                    "name": dev['name'],
                    "channels": dev['max_input_channels'],
                    "default_samplerate": dev['default_samplerate'],
                    "is_default": (i == default_input),
                })
        
        return input_devices

    # 虚拟声卡 / 远程桌面音频设备名关键词（不能作为本地语音采集设备）
    _VIRTUAL_DEVICE_KEYWORDS = (
        "virtual", "虚拟", "cable", "voicemeeter", "vb-audio", "todesk",
        "声音映射器", "sound mapper",
    )

    @staticmethod
    def _auto_select_input_device() -> Optional[int]:
        """
        启动时自动选择"真正能拾音"的麦克风。

        背景：部分机器的系统默认输入设备是虚接麦克风口（增益被AGC拉满，
        输出恒定满幅宽带噪声、对真实声音无响应）或远程桌面虚拟声卡，
        会导致 VAD 永远判定"有人说话"、语音段永不切分、识别永不触发。

        做法：对 MME 输入设备各采集约 0.8 秒测本底噪声：
          - 系统默认设备噪声 > 0.05（异常高）且存在噪声 < 0.02（正常）的物理设备
            → 自动切换到噪声最低的设备；
          - 其余情况（默认设备正常 / 无法判断）→ 返回 None，沿用系统默认。
        用户显式指定的设备（--mic-device / 环境变量 / 网页手动切换）不走此逻辑。
        """
        import sounddevice as sd
        try:
            hostapis = sd.query_hostapis()
            mme = next((h for h in hostapis if "MME" in h["name"]), None)
            scan_indices = list(mme["devices"]) if mme else []
            all_devices = sd.query_devices()
            default_idx = sd.default.device[0]

            results = []  # (index, name, noise_rms)
            for idx in scan_indices:
                dev = all_devices[idx]
                if dev["max_input_channels"] <= 0:
                    continue
                name = dev["name"]
                try:
                    rate = int(dev["default_samplerate"] or 44100)
                    vals: List[float] = []
                    q: "queue.Queue[np.ndarray]" = queue.Queue()

                    def _probe_cb(indata, frames, t_info, status):
                        x = indata[:, 0] if indata.ndim > 1 else indata
                        q.put(x.astype(np.float32).copy())

                    st = sd.InputStream(
                        samplerate=rate, channels=1, dtype="float32",
                        blocksize=int(rate * 0.2), device=idx, callback=_probe_cb,
                    )
                    st.start()
                    t0 = time.time()
                    while time.time() - t0 < 0.8:
                        try:
                            blk = q.get(timeout=0.5)
                            vals.append(float(np.sqrt(np.mean(blk ** 2))))
                        except queue.Empty:
                            break
                    st.stop()
                    st.close()

                    if vals:
                        # 跳过第1块（开流暂态），其余取中位数作为本底噪声
                        noise = float(np.median(vals[1:])) if len(vals) > 1 else float(vals[0])
                    else:
                        noise = None
                    results.append((idx, name, noise))
                    logger.info(
                        f"麦克风扫描 [{idx}] {name}: 本底噪声 RMS="
                        f"{f'{noise:.4f}' if noise is not None else '无数据'}"
                    )
                except Exception as e:
                    results.append((idx, name, None))
                    logger.warning(f"麦克风扫描 [{idx}] {name}: 打开失败 ({e})")

            def _is_virtual(n: str) -> bool:
                nl = n.lower()
                return any(k in nl for k in MicrophoneListener._VIRTUAL_DEVICE_KEYWORDS)

            default_noise = next((r for i, n, r in results if i == default_idx), None)
            candidates = [(i, n, r) for i, n, r in results
                          if r is not None and not _is_virtual(n)]

            if candidates:
                best = min(candidates, key=lambda x: x[2])
                if default_noise is not None and default_noise > 0.05 and best[2] < 0.02:
                    logger.warning(
                        f"⚠️ 系统默认麦克风 [{default_idx}] 本底噪声异常 "
                        f"(RMS={default_noise:.4f})，疑似虚接/虚拟设备；"
                        f"自动改用 [{best[0]}] {best[1]} (RMS={best[2]:.4f})。"
                        f"如不正确可在网页下拉框手动选择麦克风。"
                    )
                    return best[0]
                if default_noise is not None:
                    logger.info(
                        f"系统默认麦克风 [{default_idx}] 本底噪声 RMS={default_noise:.4f}，"
                        f"状态正常，沿用默认设备"
                    )
            return None
        except Exception as e:
            logger.warning(f"麦克风自动扫描失败，沿用系统默认设备: {e}")
            return None

    def get_device(self) -> Optional[int]:
        """获取当前使用的麦克风设备ID"""
        return self.device
    
    def set_device(self, device_index: Optional[int]) -> bool:
        """
        切换麦克风设备（运行时切换，需要重启音频流）
        - device_index: 设备ID（从 list_devices() 获取），None 使用默认设备
        返回是否切换成功
        """
        # 验证设备有效性
        if device_index is not None:
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                if device_index < 0 or device_index >= len(devices):
                    logger.error(f"设备ID {device_index} 超出范围")
                    return False
                dev = devices[device_index]
                if dev['max_input_channels'] <= 0:
                    logger.error(f"设备 [{device_index}] {dev['name']} 不是输入设备")
                    return False
            except Exception as e:
                logger.error(f"验证设备失败: {e}")
                return False
        
        old_device = self.device
        self.device = device_index
        
        # 如果正在运行，需要重启音频流
        if self.running:
            logger.info(f"切换麦克风设备: {old_device} -> {device_index}")
            try:
                # 关闭旧流
                if self._stream:
                    try:
                        self._stream.stop()
                        self._stream.close()
                    except Exception as e:
                        logger.warning(f"关闭旧音频流时出错（可忽略）: {e}")
                    self._stream = None
                
                # 清空队列中的旧数据
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except:
                        break
                
                # 启动新流
                import sounddevice as sd
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=self.dtype,
                    blocksize=self.block_size,
                    device=self.device,
                    callback=self._audio_callback,
                )
                self._stream.start()
                logger.info(f"✅ 已切换到麦克风设备 [{device_index}]")
                return True
            except Exception as e:
                logger.error(f"切换麦克风失败: {e}")
                # 回滚到旧设备
                self.device = old_device
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
                    logger.info(f"已回滚到旧设备 [{old_device}]")
                except Exception as e2:
                    logger.error(f"回滚也失败: {e2}")
                    self.running = False
                return False
        else:
            logger.info(f"麦克风设备已设置为: {device_index}（将在启动时生效）")
            return True
    
    def get_status(self) -> dict:
        """获取当前状态"""
        # 尝试获取当前设备名称
        device_name = None
        try:
            import sounddevice as sd
            if self.device is not None:
                dev = sd.query_devices(self.device)
                device_name = dev.get('name', None)
            else:
                default_idx = sd.default.device[0]
                dev = sd.query_devices(default_idx)
                device_name = f"{dev.get('name', '默认')} (默认)"
        except Exception:
            pass
        
        return {
            "running": self.running,
            "paused": self.paused,
            "state": self.state,
            "wake_words": self.wake_words,
            "silence_threshold": self.silence_threshold,
            "input_gain": self.input_gain,
            "current_rms": self.current_rms,
            "task_callback_url": self.task_callback_url,
            "device_index": self.device,
            "device_name": device_name,
        }
    
    def _post_task_callback(self, text: str):
        """
        即发即弃（Fire-and-Forget）任务回调
        - 提交到后台线程池后【立即返回】，不阻塞识别主循环
        - 不等待、不使用返回值；成功/失败仅写日志
        - 后台线程完整完成 HTTP 事务（读取并丢弃响应体），保证连接优雅关闭
        POST {task_callback_url}  body: {"task": text}
        """
        url = self.task_callback_url
        if not url:
            return
        
        try:
            # submit() 微秒级返回，调用方绝不等待
            self._callback_executor.submit(self._do_task_callback, url, text)
        except Exception as e:
            logger.error(f"❌ 任务回调提交失败 ({url}): {e}")
    
    @staticmethod
    def _do_task_callback(url: str, text: str):
        """后台线程中执行：发送 POST 回调。不读取业务返回值，仅确认请求完成。"""
        try:
            import json as _json
            import urllib.request
            
            payload = _json.dumps({"task": text}, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            # 连接/读取超时（仅作用于后台线程）：避免任务服务器无响应时线程永久挂起
            with urllib.request.urlopen(req, timeout=10) as resp:
                # 读完响应体后连接优雅关闭（服务器不会收到连接重置）
                # 返回值不使用，直接丢弃
                resp.read()
                logger.info(f"📤 任务回调已发送 [{resp.status}]: {text} -> {url}")
        except Exception as e:
            logger.error(f"❌ 任务回调发送失败 ({url}): {e}")


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
