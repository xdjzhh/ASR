import os
import io
import glob
import numpy as np
import soundfile as sf
from funasr import AutoModel
from typing import Optional, List, Dict, Any
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_local_model_path(cache_dir: str, model_id: str) -> Optional[str]:
    """
    检查模型是否已存在本地缓存，返回本地路径。
    ModelScope 缓存结构: {cache_dir}/hub/{owner}--{model_name}/snapshots/{revision}/
    或者旧结构: {cache_dir}/models/{owner}--{model_name}/snapshots/{revision}/
    """
    # 处理 model_id 格式: "iic/SenseVoiceSmall" -> "iic--SenseVoiceSmall"
    folder_name = model_id.replace("/", "--")
    
    # 尝试多种可能的缓存路径结构
    possible_paths = [
        os.path.join(cache_dir, "hub", folder_name, "snapshots"),
        os.path.join(cache_dir, "models", folder_name, "snapshots"),
        os.path.join(cache_dir, folder_name, "snapshots"),
    ]
    
    for snapshots_dir in possible_paths:
        if os.path.isdir(snapshots_dir):
            # 找到最新的 snapshot（通常是 "master" 或版本hash）
            versions = [d for d in os.listdir(snapshots_dir) 
                       if os.path.isdir(os.path.join(snapshots_dir, d))]
            if versions:
                # 优先用 master，否则用第一个
                if "master" in versions:
                    return os.path.join(snapshots_dir, "master")
                return os.path.join(snapshots_dir, sorted(versions)[-1])
    
    return None


class ASREngine:
    """Paraformer 语音识别引擎，基于 FunASR 框架"""
    
    # 预定义模型配置（按大小排序，使用 FunASR 内置别名，避免模型ID错误）
    MODEL_CONFIGS = {
        # 超小模式：SenseVoiceSmall，~234M参数，约300MB，支持中文普通话，速度快
        "tiny": {
            "model": "iic/SenseVoiceSmall",
            "vad_model": None,
            "punc_model": None,
            "trust_remote_code": True,
            "language": "zh",
            "description": "SenseVoice最小模型，约300MB，中文优先"
        },
        # 轻量模式：SenseVoiceSmall + VAD，约350MB
        "mini": {
            "model": "iic/SenseVoiceSmall",
            "vad_model": "fsmn-vad",
            "punc_model": None,
            "trust_remote_code": True,
            "language": "zh",
            "description": "SenseVoice轻量模式，约350MB"
        },
        # 标准模式：Paraformer-zh（内置别名）+ VAD + 标点，约400MB，中文精度最高
        "standard": {
            "model": "paraformer-zh",
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc",
            "trust_remote_code": False,
            "language": None,
            "description": "Paraformer标准模式，含VAD+标点，约400MB"
        },
        # Paraformer 纯ASR模式（无VAD/标点）
        "paraformer": {
            "model": "paraformer-zh",
            "vad_model": None,
            "punc_model": None,
            "trust_remote_code": False,
            "language": None,
            "description": "Paraformer纯ASR，约220M参数"
        },
    }
    
    # 辅助模型的 ModelScope ID（用于本地路径解析）
    AUX_MODEL_IDS = {
        "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "ct-punc": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    }
    
    # 模型加载回退顺序
    FALLBACK_MODELS = [
        {"model": "iic/SenseVoiceSmall", "trust_remote_code": True, "vad_model": "fsmn-vad"},
        {"model": "paraformer-zh", "trust_remote_code": False, "vad_model": None},
    ]
    
    def __init__(
        self,
        model: str = None,
        model_revision: str = None,
        vad_model: Optional[str] = None,
        vad_revision: Optional[str] = None,
        punc_model: Optional[str] = None,
        punc_revision: Optional[str] = None,
        device: str = "cpu",
        ncpu: int = 4,
        mode: str = "tiny",
        trust_remote_code: bool = False,
        language: str = "zh",
    ):
        self.device = device
        self.ncpu = ncpu
        self.model = None
        self.mode = mode
        self.language = language
        self.is_sensevoice = False
        
        # 加载预定义配置
        if mode in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[mode].copy()
            desc = config.pop("description", "")
            
            if model is None:
                model = config.pop("model", "iic/SenseVoiceSmall")
            if vad_model is None:
                vad_model = config.pop("vad_model", None)
            if punc_model is None:
                punc_model = config.pop("punc_model", None)
            if "trust_remote_code" in config:
                trust_remote_code = config["trust_remote_code"]
            if "language" in config and config["language"] is not None:
                self.language = config["language"]
        
        self.model_name = model
        self.vad_model_name = vad_model
        self.punc_model_name = punc_model
        self.trust_remote_code = trust_remote_code
        
        # 判断是否是 SenseVoice
        self.is_sensevoice = "SenseVoice" in str(model)
        
        # 模型存储目录
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 设置 ModelScope 缓存目录（使用 hub 子目录，避免嵌套）
        # ModelScope 新版本使用 {cache_dir}/hub/ 结构，旧版本用 {cache_dir}/models/
        self.cache_dir = self.models_dir
        os.environ["MODELSCOPE_CACHE"] = self.cache_dir
        # 禁止 ModelScope 联网检查更新（如果模型已存在本地）
        os.environ.setdefault("MODELSCOPE_OFFLINE", "0")  # 默认允许下载，等确认本地有模型后再设为1
        
        logger.info(f"模型目录: {self.models_dir}")
        logger.info(f"模式: {mode} - {self.MODEL_CONFIGS.get(mode, {}).get('description', '自定义')}")
        logger.info(f"模型: {model}")
        if vad_model:
            logger.info("包含 VAD 语音端点检测")
        if punc_model:
            logger.info("包含自动标点")
        
        self._load_model_with_fallback(
            model=model,
            vad_model=vad_model,
            punc_model=punc_model,
        )
    
    def _resolve_model_path(self, model_id: str) -> str:
        """
        解析模型路径：如果本地已有模型，返回本地路径；否则返回远程model_id。
        对于FunASR内置别名(paraformer-zh, fsmn-vad, ct-punc)，保持原样让FunASR自己处理。
        """
        # FunASR 内置别名不需要解析路径
        builtin_aliases = {"paraformer-zh", "paraformer", "fsmn-vad", "ct-punc", 
                          "paraformer-zh-streaming", "fsmn-vad-8k"}
        if model_id in builtin_aliases:
            return model_id
        
        # 检查本地缓存
        local_path = _resolve_local_model_path(self.cache_dir, model_id)
        if local_path and os.path.exists(os.path.join(local_path, "model.pt")):
            logger.info(f"使用本地模型: {local_path}")
            return local_path
        
        # 对于辅助模型，也尝试从 AUX_MODEL_IDS 查找
        if model_id in self.AUX_MODEL_IDS:
            ms_id = self.AUX_MODEL_IDS[model_id]
            local_path = _resolve_local_model_path(self.cache_dir, ms_id)
            if local_path:
                # 检查辅助模型的关键文件
                has_model = any(glob.glob(os.path.join(local_path, "*.pt")) + 
                              glob.glob(os.path.join(local_path, "*.onnx")))
                if has_model or os.path.exists(os.path.join(local_path, "config.yaml")):
                    logger.info(f"使用本地辅助模型: {local_path}")
                    return local_path
        
        # 本地不存在，返回远程ID让FunASR下载
        logger.info(f"本地未找到模型 {model_id}，将从 ModelScope 下载")
        return model_id
    
    def _load_model_with_fallback(self, model, vad_model, punc_model):
        """加载模型，失败时自动回退到备选模型"""
        
        # 先尝试用户指定的模型
        try:
            self._try_load_model(model, vad_model, punc_model)
            logger.info("✅ 模型加载成功！")
            return
        except Exception as e:
            logger.warning(f"模型 {model} 加载失败: {e}")
        
        # 回退到备选模型
        for fallback in self.FALLBACK_MODELS:
            if fallback["model"] == model:
                continue  # 跳过已尝试的
            try:
                logger.info(f"尝试回退模型: {fallback['model']}")
                self.model_name = fallback["model"]
                self.vad_model_name = fallback.get("vad_model")
                self.punc_model_name = None
                self.trust_remote_code = fallback.get("trust_remote_code", False)
                self.is_sensevoice = "SenseVoice" in fallback["model"]
                if self.is_sensevoice:
                    self.language = "zh"
                
                self._try_load_model(
                    fallback["model"],
                    fallback.get("vad_model"),
                    None,
                )
                logger.info(f"✅ 回退模型 {fallback['model']} 加载成功！")
                return
            except Exception as e2:
                logger.warning(f"回退模型 {fallback['model']} 也失败: {e2}")
        
        raise RuntimeError("所有模型都加载失败，请检查网络连接或依赖安装")
    
    def _try_load_model(self, model, vad_model, punc_model):
        """尝试加载单个模型配置"""
        
        # 解析为本地路径（如果已下载）
        model_path = self._resolve_model_path(model)
        vad_path = self._resolve_model_path(vad_model) if vad_model else None
        punc_path = self._resolve_model_path(punc_model) if punc_model else None
        
        # 判断是否全部本地
        main_local = os.path.isabs(model_path) and os.path.exists(os.path.join(model_path, "model.pt"))
        
        kwargs = {
            "model": model_path,
            "device": self.device,
            "ncpu": self.ncpu,
            "disable_pbar": False,
            "disable_update": True,  # 禁止在线检查更新
        }
        
        if main_local:
            logger.info("主模型从本地路径加载，不联网检查更新")
        
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = self.trust_remote_code
        if vad_path:
            kwargs["vad_model"] = vad_path
        if punc_path:
            kwargs["punc_model"] = punc_path
        
        logger.info(f"正在加载模型: {model_path}")
        self.model = AutoModel(**kwargs)
        
        # === 内存优化 ===
        import torch
        import gc
        
        # 1. 设置为评估模式（递归遍历所有子模块）
        if hasattr(self.model, 'model'):
            self.model.model.eval()
        for attr_name in dir(self.model):
            attr = getattr(self.model, attr_name, None)
            if attr is not None and hasattr(attr, 'eval') and hasattr(attr, 'parameters'):
                try:
                    attr.eval()
                except:
                    pass
        
        # 2. 全局禁用梯度计算
        torch.set_grad_enabled(False)
        
        # 3. 设置PyTorch线程数（与环境变量一致）
        ncpu = self.ncpu if self.ncpu else min(4, os.cpu_count() or 4)
        torch.set_num_threads(ncpu)
        try:
            torch.set_num_interop_threads(min(2, ncpu))
        except RuntimeError:
            pass
        
        # 4. 动态INT8量化（CPU推理专用，可减少约50%内存，同时加速推理）
        if self.device == "cpu" and hasattr(self.model, 'model'):
            try:
                self.model.model = torch.quantization.quantize_dynamic(
                    self.model.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("✅ 已启用INT8动态量化（内存优化）")
            except Exception as qe:
                logger.warning(f"INT8量化失败（不影响使用）: {qe}")
        
        # 5. 清理加载过程中产生的临时对象
        gc.collect()
    
    def _load_audio(self, audio_data: bytes, sample_rate: int = 16000) -> np.ndarray:
        """从字节数据加载音频，自动重采样到 16kHz（全程内存处理）"""
        try:
            data, orig_sr = sf.read(io.BytesIO(audio_data))
        except Exception:
            try:
                import librosa
                data, orig_sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
            except Exception as e:
                raise RuntimeError(f"无法解码音频，请尝试转换为 WAV 格式。错误: {e}")
        
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        if orig_sr != sample_rate:
            import librosa
            data = librosa.resample(data.astype(np.float32), orig_sr=orig_sr, target_sr=sample_rate)
        
        return data.astype(np.float32)
    
    def transcribe_bytes(
        self,
        audio_data: bytes,
        batch_size_s: int = 300,
        use_itn: bool = True,
        language: str = None,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        try:
            waveform = self._load_audio(audio_data)
            return self._transcribe_waveform(waveform, batch_size_s, use_itn, language)
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            raise
    
    def transcribe_file(
        self,
        file_path: str,
        batch_size_s: int = 300,
        use_itn: bool = True,
        language: str = None,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        try:
            import librosa
            waveform, sr = librosa.load(file_path, sr=16000, mono=True)
            waveform = waveform.astype(np.float32)
            return self._transcribe_waveform(waveform, batch_size_s, use_itn, language)
        except Exception as e1:
            logger.warning(f"librosa 读取失败，尝试直接传入文件路径: {e1}")
            try:
                kwargs = {
                    "input": file_path,
                    "batch_size_s": batch_size_s,
                    "use_itn": use_itn,
                }
                if self.is_sensevoice:
                    kwargs["language"] = language or self.language or "zh"
                
                result = self.model.generate(**kwargs)
                return self._postprocess_result(result)
            except Exception as e2:
                logger.error(f"识别失败: {e2}")
                raise
    
    def transcribe_waveform(
        self,
        waveform: np.ndarray,
        batch_size_s: int = 300,
        use_itn: bool = True,
    ) -> Dict[str, Any]:
        return self._transcribe_waveform(waveform, batch_size_s, use_itn, None)
    
    def _transcribe_waveform(
        self,
        waveform: np.ndarray,
        batch_size_s: int = 300,
        use_itn: bool = True,
        language: str = None,
    ) -> Dict[str, Any]:
        import torch
        
        kwargs = {
            "input": waveform,
            "batch_size_s": batch_size_s,
            "use_itn": use_itn,
        }
        
        if self.is_sensevoice:
            kwargs["language"] = language or self.language or "zh"
        
        # 使用 inference_mode（比 no_grad 更激进，完全禁用autograd追踪）
        with torch.inference_mode():
            result = self.model.generate(**kwargs)
        return self._postprocess_result(result)
    
    def _postprocess_result(self, result: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理识别结果，SenseVoice 需要特殊后处理"""
        if not result:
            return {"text": "", "sentences": []}
        
        if self.is_sensevoice:
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                text = rich_transcription_postprocess(result[0]["text"])
                return {
                    "text": text,
                    "sentences": [{"text": text}],
                }
            except ImportError:
                pass
        
        return self._format_result(result)
    
    def _format_result(self, result: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not result:
            return {"text": "", "sentences": []}
        
        full_text = ""
        sentences = []
        
        for res in result:
            text = res.get("text", "")
            full_text += text
            sent_info = {"text": text}
            if "timestamp" in res:
                sent_info["timestamp"] = res["timestamp"]
            sentences.append(sent_info)
        
        return {
            "text": full_text.strip(),
            "sentences": sentences,
        }


# 全局单例
_asr_engine: Optional[ASREngine] = None


def get_asr_engine(
    mode: str = "tiny",
    device: str = "cpu",
    ncpu: int = 4,
    **kwargs
) -> ASREngine:
    global _asr_engine
    if _asr_engine is None:
        _asr_engine = ASREngine(mode=mode, device=device, ncpu=ncpu, **kwargs)
    return _asr_engine
