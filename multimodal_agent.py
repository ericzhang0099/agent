import os
import base64
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import aiohttp
import asyncio


@dataclass
class VisionInput:
    """视觉输入数据类"""
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    prompt: str = "描述这张图片"
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为OpenAI API格式"""
        if self.image_base64:
            image_url = f"data:image/jpeg;base64,{self.image_base64}"
        elif self.image_path:
            image_url = f"data:image/jpeg;base64,{self._encode_image(self.image_path)}"
        elif self.image_url:
            image_url = self.image_url
        else:
            raise ValueError("必须提供image_path、image_url或image_base64之一")
        
        return {
            "type": "image_url",
            "image_url": {"url": image_url}
        }
    
    @staticmethod
    def _encode_image(image_path: str) -> str:
        """将图片编码为base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


@dataclass
class AudioInput:
    """音频输入数据类"""
    audio_path: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    language: Optional[str] = "zh"
    
    def get_audio_data(self) -> bytes:
        """获取音频数据"""
        if self.audio_bytes:
            return self.audio_bytes
        elif self.audio_path:
            with open(self.audio_path, "rb") as f:
                return f.read()
        else:
            raise ValueError("必须提供audio_path或audio_bytes")


@dataclass
class TTSOutput:
    """TTS输出数据类"""
    audio_bytes: bytes
    text: str
    format: str = "mp3"


class MultimodalAgent:
    """
    多模态Agent核心类
    支持：视觉理解(GPT-4V)、语音转录(Whisper)、语音合成(TTS)
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_base_url: str = "https://api.openai.com/v1",
        vision_model: str = "gpt-4o",
        whisper_model: str = "whisper-1",
        tts_model: str = "tts-1",
        tts_voice: str = "alloy"
    ):
        """
        初始化多模态Agent
        
        Args:
            openai_api_key: OpenAI API密钥
            openai_base_url: API基础URL
            vision_model: 视觉模型名称
            whisper_model: 语音转录模型
            tts_model: 语音合成模型
            tts_voice: TTS声音类型 (alloy, echo, fable, onyx, nova, shimmer)
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("必须提供openai_api_key或设置OPENAI_API_KEY环境变量")
        
        self.base_url = openai_base_url
        self.vision_model = vision_model
        self.whisper_model = whisper_model
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建HTTP会话"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self.session
    
    async def vision_understand(
        self,
        vision_input: VisionInput,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        视觉理解 - 使用GPT-4V分析图片
        
        Args:
            vision_input: 视觉输入
            max_tokens: 最大生成token数
            temperature: 温度参数
            
        Returns:
            图片描述文本
        """
        session = await self._get_session()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_input.prompt},
                    vision_input.to_openai_format()
                ]
            }
        ]
        
        payload = {
            "model": self.vision_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with session.post(
            f"{self.base_url}/chat/completions",
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Vision API错误: {response.status} - {error_text}")
            
            result = await response.json()
            return result["choices"][0]["message"]["content"]
    
    async def speech_to_text(
        self,
        audio_input: AudioInput,
        prompt: Optional[str] = None
    ) -> str:
        """
        语音转录 - 使用Whisper将语音转为文本
        
        Args:
            audio_input: 音频输入
            prompt: 可选的提示文本
            
        Returns:
            转录的文本
        """
        session = await self._get_session()
        
        audio_data = audio_input.get_audio_data()
        
        # 构建multipart表单数据
        data = aiohttp.FormData()
        data.add_field('file', audio_data, filename='audio.mp3', content_type='audio/mpeg')
        data.add_field('model', self.whisper_model)
        if audio_input.language:
            data.add_field('language', audio_input.language)
        if prompt:
            data.add_field('prompt', prompt)
        
        async with session.post(
            f"{self.base_url}/audio/transcriptions",
            data=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Whisper API错误: {response.status} - {error_text}")
            
            result = await response.json()
            return result["text"]
    
    async def text_to_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        output_format: str = "mp3"
    ) -> TTSOutput:
        """
        语音合成 - 使用TTS将文本转为语音
        
        Args:
            text: 要合成的文本
            voice: 声音类型，默认使用初始化时的voice
            speed: 语速 (0.25-4.0)
            output_format: 输出格式 (mp3, opus, aac, flac, wav, pcm)
            
        Returns:
            TTSOutput对象，包含音频字节
        """
        session = await self._get_session()
        
        payload = {
            "model": self.tts_model,
            "input": text,
            "voice": voice or self.tts_voice,
            "speed": speed,
            "response_format": output_format
        }
        
        async with session.post(
            f"{self.base_url}/audio/speech",
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"TTS API错误: {response.status} - {error_text}")
            
            audio_bytes = await response.read()
            return TTSOutput(
                audio_bytes=audio_bytes,
                text=text,
                format=output_format
            )
    
    async def multimodal_chat(
        self,
        text: Optional[str] = None,
        vision_input: Optional[VisionInput] = None,
        audio_input: Optional[AudioInput] = None,
        enable_tts: bool = False
    ) -> Dict[str, Any]:
        """
        多模态对话 - 整合视觉、语音、文本
        
        Args:
            text: 文本输入
            vision_input: 视觉输入
            audio_input: 音频输入
            enable_tts: 是否启用TTS回复
            
        Returns:
            包含响应文本和可选TTS音频的字典
        """
        result = {"text": "", "audio": None}
        
        # 如果有音频输入，先转录为文本
        if audio_input:
            transcribed_text = await self.speech_to_text(audio_input)
            if text:
                text = f"{text}\n[用户语音]: {transcribed_text}"
            else:
                text = transcribed_text
            result["transcribed_text"] = transcribed_text
        
        # 如果有视觉输入，使用视觉理解
        if vision_input:
            if not vision_input.prompt and text:
                vision_input.prompt = text
            vision_response = await self.vision_understand(vision_input)
            result["text"] = vision_response
        elif text:
            # 纯文本对话
            result["text"] = text
        
        # 如果需要TTS
        if enable_tts and result["text"]:
            tts_output = await self.text_to_speech(result["text"])
            result["audio"] = tts_output
        
        return result
    
    async def close(self):
        """关闭会话"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
