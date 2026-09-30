import os
import io
import tarfile
import urllib.request
import numpy as np
import soundfile as sf
from typing import Optional, List, Dict, Any
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# sherpa-onnx 模型仓库基础地址（GitHub Releases）
GITHUB_RELEASES_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
# ModelScope 镜像地址（国内下载更快）
MODELSCOPE_URL = "https://www.modelscope.cn/models/csukuangfj/asr-models/resolve/master"


class ASREngine:
    """语音识别引擎，基于 sherpa-onnx（ONNX Runtime 推理）"""

    # 预定义模型配置（统一使用 Paraformer-zh INT8）
    MODEL_CONFIGS = {
        # tiny/mini/standard/paraformer 均指向 Paraformer-zh INT8
        "tiny": {
            "model_type": "paraformer",
            "model_archive": "sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2",
            "model_dir": "sherpa-onnx-paraformer-zh-2023-03-28",
            "model_file": "model.int8.onnx",
            "tokens_file": "tokens.txt",
            "description": "Paraformer-zh INT8，约213MB，中文通用精度最优"
        },
        "mini": {
            "model_type": "paraformer",
            "model_archive": "sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2",
            "model_dir": "sherpa-onnx-paraformer-zh-2023-03-28",
            "model_file": "model.int8.onnx",
            "tokens_file": "tokens.txt",
            "description": "Paraformer-zh INT8，约213MB"
        },
        "standard": {
            "model_type": "paraformer",
            "model_archive": "sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2",
            "model_dir": "sherpa-onnx-paraformer-zh-2023-03-28",
            "model_file": "model.int8.onnx",
            "tokens_file": "tokens.txt",
            "description": "Paraformer-zh INT8，约213MB，中文通用精度最优"
        },
        "paraformer": {
            "model_type": "paraformer",
            "model_archive": "sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2",
            "model_dir": "sherpa-onnx-paraformer-zh-2023-03-28",
            "model_file": "model.int8.onnx",
            "tokens_file": "tokens.txt",
            "description": "Paraformer-zh INT8，约213MB"
        },
    }

    def __init__(
        self,
        model: str = None,
        vad_model: Optional[str] = None,
        punc_model: Optional[str] = None,
        device: str = "cpu",
        ncpu: int = 4,
        mode: str = "standard",
        language: str = "zh",
        **kwargs,
    ):
        self.device = device
        self.ncpu = ncpu
        self.mode = mode
        self.language = language
        self.recognizer = None
        self.model_type = None

        # 加载预定义配置
        if mode in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[mode].copy()
            desc = config.pop("description", "")
            self.model_type = config.pop("model_type")
            self.model_archive = config.pop("model_archive")
            self.model_dir_name = config.pop("model_dir")
            self.model_file_name = config.pop("model_file")
            self.tokens_file_name = config.pop("tokens_file")
            self.language = config.pop("language", language)
            self.use_itn = config.pop("use_itn", True)
            self.model_name = self.model_dir_name
        else:
            raise ValueError(f"不支持的模式: {mode}，可选: {list(self.MODEL_CONFIGS.keys())}")

        # 模型存储目录
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.models_dir, exist_ok=True)

        logger.info(f"模式: {mode} - {desc}")
        logger.info(f"模型类型: {self.model_type}")
        logger.info(f"模型目录: {self.models_dir}")

        self._load_model()

    def _get_model_dir(self) -> str:
        """返回模型解压后的目录路径"""
        return os.path.join(self.models_dir, self.model_dir_name)

    def _ensure_model_files(self) -> str:
        """确保模型文件存在，不存在则下载解压。返回模型目录路径。"""
        model_dir = self._get_model_dir()
        model_path = os.path.join(model_dir, self.model_file_name)
        tokens_path = os.path.join(model_dir, self.tokens_file_name)

        if os.path.exists(model_path) and os.path.exists(tokens_path):
            logger.info(f"模型文件已存在: {model_dir}")
            return model_dir

        logger.info(f"模型文件不存在，开始下载: {self.model_archive}")
        archive_path = os.path.join(self.models_dir, self.model_archive)

        # 尝试多个下载源
        urls = [
            f"{GITHUB_RELEASES_URL}/{self.model_archive}",
            f"{MODELSCOPE_URL}/{self.model_archive}",
        ]

        downloaded = False
        for url in urls:
            try:
                logger.info(f"尝试下载: {url}")
                self._download_file(url, archive_path)
                downloaded = True
                break
            except Exception as e:
                logger.warning(f"下载失败 ({url}): {e}")
                if os.path.exists(archive_path):
                    os.remove(archive_path)

        if not downloaded:
            raise RuntimeError(
                f"模型下载失败，请手动下载 {self.model_archive} 并解压到 {self.models_dir}"
            )

        # 解压
        logger.info(f"解压模型: {archive_path}")
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(self.models_dir)

        # 删除压缩包
        os.remove(archive_path)
        logger.info("模型解压完成")

        if not os.path.exists(model_path):
            raise RuntimeError(f"模型解压后未找到 {model_path}")

        return model_dir

    def _download_file(self, url: str, dest: str):
        """下载文件，带简单进度日志"""
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 256  # 256KB
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0 and downloaded % (1024 * 1024 * 10) < chunk_size:
                        pct = downloaded * 100 / total
                        logger.info(f"下载进度: {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB ({pct:.1f}%)")

    def _load_model(self):
        """加载 sherpa-onnx 识别器"""
        import sherpa_onnx

        model_dir = self._ensure_model_files()
        model_path = os.path.join(model_dir, self.model_file_name)
        tokens_path = os.path.join(model_dir, self.tokens_file_name)

        logger.info(f"正在加载识别器: {model_path}")

        # 加载 Paraformer-zh INT8 识别器
        if self.model_type == "paraformer":
            recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=model_path,
                tokens=tokens_path,
                num_threads=self.ncpu,
            )
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

        self.recognizer = recognizer
        logger.info(f"✅ sherpa-onnx 识别器加载成功 (model_type={self.model_type})")

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
        **kwargs,
    ) -> Dict[str, Any]:
        if self.recognizer is None:
            raise RuntimeError("模型未加载")

        try:
            waveform = self._load_audio(audio_data)
            return self._transcribe_waveform(waveform)
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            raise

    def transcribe_file(
        self,
        file_path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        if self.recognizer is None:
            raise RuntimeError("模型未加载")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            import librosa
            waveform, sr = librosa.load(file_path, sr=16000, mono=True)
            waveform = waveform.astype(np.float32)
            return self._transcribe_waveform(waveform)
        except Exception as e1:
            logger.warning(f"librosa 读取失败，尝试 soundfile: {e1}")
            try:
                data, sr = sf.read(file_path)
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                if sr != 16000:
                    import librosa
                    data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
                return self._transcribe_waveform(data.astype(np.float32))
            except Exception as e2:
                logger.error(f"识别失败: {e2}")
                raise

    def transcribe_waveform(
        self,
        waveform: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        return self._transcribe_waveform(waveform)

    def _transcribe_waveform(self, waveform: np.ndarray) -> Dict[str, Any]:
        """使用 sherpa-onnx 识别波形"""
        if waveform is None or len(waveform) == 0:
            return {"text": "", "sentences": []}

        # 确保是 float32 一维数组
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=waveform)
        self.recognizer.decode_stream(stream)

        text = stream.result.text.strip()
        return {
            "text": text,
            "sentences": [{"text": text}] if text else [],
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
