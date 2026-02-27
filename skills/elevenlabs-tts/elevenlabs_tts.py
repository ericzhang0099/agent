#!/usr/bin/env python3
"""
ElevenLabs语音合成系统 v1.1
高质量TTS + 多声音 + 16种情感控制 + 跨模态一致性验证
适配 elevenlabs Python SDK v2.x
"""

import os
import json
import hashlib
from typing import Optional, Dict, Any, Iterator, List
from datetime import datetime

# 尝试导入elevenlabs v2.x
try:
    from elevenlabs import ElevenLabs, VoiceSettings, play, save
    from elevenlabs.client import ElevenLabs as ElevenLabsClient
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("⚠️ 警告: elevenlabs 未安装，运行: pip install elevenlabs")

# 尝试加载.env文件
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

class ElevenLabsTTS:
    """ElevenLabs语音合成系统 - 支持16种情感映射"""
    
    # 预设声音配置 - 12种声音（6男6女）
    VOICES = {
        # 女声
        'Bella': {'name': 'Bella', 'gender': 'female', 'style': '温暖、自然', 'locale': 'en-US'},
        'Rachel': {'name': 'Rachel', 'gender': 'female', 'style': '友好、活泼', 'locale': 'en-US'},
        'Elli': {'name': 'Elli', 'gender': 'female', 'style': '年轻、活力', 'locale': 'en-US'},
        'Alice': {'name': 'Alice', 'gender': 'female', 'style': '优雅、知性', 'locale': 'en-GB'},
        'Domi': {'name': 'Domi', 'gender': 'female', 'style': '专业、自信', 'locale': 'en-US'},
        'Grace': {'name': 'Grace', 'gender': 'female', 'style': '温柔、亲切', 'locale': 'en-US'},
        # 男声
        'Adam': {'name': 'Adam', 'gender': 'male', 'style': '专业、清晰', 'locale': 'en-US'},
        'Antoni': {'name': 'Antoni', 'gender': 'male', 'style': '深沉、稳重', 'locale': 'en-US'},
        'Josh': {'name': 'Josh', 'gender': 'male', 'style': '随意、亲切', 'locale': 'en-US'},
        'Sam': {'name': 'Sam', 'gender': 'male', 'style': '年轻、活力', 'locale': 'en-US'},
        'Thomas': {'name': 'Thomas', 'gender': 'male', 'style': '权威、正式', 'locale': 'en-GB'},
        'Michael': {'name': 'Michael', 'gender': 'male', 'style': '温暖、可信', 'locale': 'en-US'},
    }
    
    # 16种情感风格映射（通过stability/similarity/style调整）
    EMOTIONS = {
        # 基础情感
        'neutral': {
            'name': '中性',
            'name_en': 'neutral',
            'stability': 0.50, 
            'similarity_boost': 0.75, 
            'style': 0.00,
            'description': '标准、自然的语音风格'
        },
        'calm': {
            'name': '平静',
            'name_en': 'calm',
            'stability': 0.80, 
            'similarity_boost': 0.60, 
            'style': -0.30,
            'description': '放松、舒缓的语调'
        },
        'gentle': {
            'name': '温和',
            'name_en': 'gentle',
            'stability': 0.70, 
            'similarity_boost': 0.65, 
            'style': -0.20,
            'description': '柔和、友善的表达'
        },
        # 积极情感
        'happy': {
            'name': '开心',
            'name_en': 'happy',
            'stability': 0.40, 
            'similarity_boost': 0.80, 
            'style': 0.40,
            'description': '愉悦、快乐的情绪'
        },
        'excited': {
            'name': '兴奋',
            'name_en': 'excited',
            'stability': 0.30, 
            'similarity_boost': 0.80, 
            'style': 0.60,
            'description': '激动、热情的表达'
        },
        'optimistic': {
            'name': '乐观',
            'name_en': 'optimistic',
            'stability': 0.45, 
            'similarity_boost': 0.78, 
            'style': 0.35,
            'description': '积极、向上的态度'
        },
        'friendly': {
            'name': '友好',
            'name_en': 'friendly',
            'stability': 0.40, 
            'similarity_boost': 0.80, 
            'style': 0.20,
            'description': '亲切、热情的语气'
        },
        'humorous': {
            'name': '幽默',
            'name_en': 'humorous',
            'stability': 0.35, 
            'similarity_boost': 0.75, 
            'style': 0.45,
            'description': '轻松、诙谐的风格'
        },
        # 专业情感
        'serious': {
            'name': '严肃',
            'name_en': 'serious',
            'stability': 0.75, 
            'similarity_boost': 0.70, 
            'style': -0.20,
            'description': '正式、庄重的语调'
        },
        'professional': {
            'name': '专业',
            'name_en': 'professional',
            'stability': 0.70, 
            'similarity_boost': 0.72, 
            'style': 0.00,
            'description': '商务、权威的表达'
        },
        'authoritative': {
            'name': '权威',
            'name_en': 'authoritative',
            'stability': 0.80, 
            'similarity_boost': 0.75, 
            'style': -0.10,
            'description': '命令、领导的语气'
        },
        'confident': {
            'name': '自信',
            'name_en': 'confident',
            'stability': 0.60, 
            'similarity_boost': 0.80, 
            'style': 0.15,
            'description': '坚定、确信的表达'
        },
        # 特殊情感
        'mysterious': {
            'name': '神秘',
            'name_en': 'mysterious',
            'stability': 0.55, 
            'similarity_boost': 0.55, 
            'style': 0.10,
            'description': '悬疑、暗示的风格'
        },
        'sad': {
            'name': '悲伤',
            'name_en': 'sad',
            'stability': 0.65, 
            'similarity_boost': 0.50, 
            'style': -0.40,
            'description': '低沉、忧郁的语调'
        },
        'angry': {
            'name': '愤怒',
            'name_en': 'angry',
            'stability': 0.25, 
            'similarity_boost': 0.85, 
            'style': 0.50,
            'description': '强烈、激动的情绪'
        },
        'fearful': {
            'name': '恐惧',
            'name_en': 'fearful',
            'stability': 0.20, 
            'similarity_boost': 0.60, 
            'style': 0.30,
            'description': '紧张、不安的表达'
        },
        'whisper': {
            'name': '耳语',
            'name_en': 'whisper',
            'stability': 0.60, 
            'similarity_boost': 0.50, 
            'style': -0.50,
            'description': '私密、轻声的风格'
        },
    }
    
    # 情感分类
    EMOTION_CATEGORIES = {
        'basic': ['neutral', 'calm', 'gentle'],
        'positive': ['happy', 'excited', 'optimistic', 'friendly', 'humorous'],
        'professional': ['serious', 'professional', 'authoritative', 'confident'],
        'special': ['mysterious', 'sad', 'angry', 'fearful', 'whisper']
    }
    
    # 默认模型
    DEFAULT_MODEL = "eleven_multilingual_v2"
    
    def __init__(self, 
                 api_key: str = None,
                 voice: str = None,
                 model: str = None,
                 cache_dir: str = None,
                 auto_cache: bool = True,
                 default_emotion: str = "neutral"):
        """
        初始化ElevenLabs TTS
        
        Args:
            api_key: API密钥（如不提供，从环境变量读取）
            voice: 默认声音（默认从环境变量或Bella）
            model: TTS模型
            cache_dir: 缓存目录
            auto_cache: 是否自动缓存
            default_emotion: 默认情感
        """
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        self.default_voice = voice or os.getenv('ELEVENLABS_DEFAULT_VOICE', 'Bella')
        self.default_emotion = default_emotion or os.getenv('ELEVENLABS_DEFAULT_EMOTION', 'neutral')
        self.model = model or os.getenv('ELEVENLABS_MODEL', self.DEFAULT_MODEL)
        self.cache_dir = cache_dir or os.getenv('ELEVENLABS_CACHE_DIR', './tts_cache')
        self.auto_cache = auto_cache
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 状态
        self.status = "configured"
        self.last_error = None
        self.stats = {
            'synthesis_count': 0,
            'cache_hits': 0,
            'errors': 0
        }
        
        # 初始化客户端
        self.client = None
        if ELEVENLABS_AVAILABLE and self.api_key:
            try:
                self.client = ElevenLabsClient(api_key=self.api_key)
                self.status = "ready"
            except Exception as e:
                self.status = "error"
                self.last_error = str(e)
        elif not ELEVENLABS_AVAILABLE:
            self.status = "missing_dependency"
        elif not self.api_key:
            self.status = "missing_api_key"
    
    def synthesize(self, 
                   text: str,
                   voice: str = None,
                   emotion: str = None,
                   speed: float = 1.0,
                   output_path: str = None) -> Dict[str, Any]:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            voice: 声音ID（默认使用初始化设置）
            emotion: 情感风格（16种之一）
            speed: 语速（0.5-2.0）
            output_path: 输出文件路径（可选）
            
        Returns:
            dict: 包含音频路径和元数据
        """
        if self.status != "ready":
            return {
                'success': False,
                'error': f"TTS未就绪: {self.status}",
                'text': text
            }
        
        voice = voice or self.default_voice
        emotion = emotion or self.default_emotion
        
        # 验证情感
        if emotion not in self.EMOTIONS:
            return {
                'success': False,
                'error': f"无效的情感: {emotion}，可用: {list(self.EMOTIONS.keys())}",
                'text': text
            }
        
        # 检查缓存
        if self.auto_cache:
            cached = self._get_cache(text, voice, emotion, speed)
            if cached:
                self.stats['cache_hits'] += 1
                return {
                    'success': True,
                    'audio_path': cached,
                    'cached': True,
                    'text': text,
                    'voice': voice,
                    'emotion': emotion,
                    'emotion_params': self.EMOTIONS[emotion]
                }
        
        # 获取情感参数
        emotion_params = self.EMOTIONS[emotion]
        
        try:
            # 生成音频 (elevenlabs v2.x API)
            voice_settings = VoiceSettings(
                stability=emotion_params['stability'],
                similarity_boost=emotion_params['similarity_boost'],
                style=emotion_params['style'] + (speed - 1.0) * 0.3
            )
            
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice,
                model_id=self.model,
                voice_settings=voice_settings
            )
            
            # 确定输出路径
            if output_path is None:
                output_path = self._generate_cache_path(text, voice, emotion, speed)
            
            # 保存音频
            save(audio, output_path)
            
            self.stats['synthesis_count'] += 1
            
            return {
                'success': True,
                'audio_path': output_path,
                'cached': False,
                'text': text,
                'voice': voice,
                'emotion': emotion,
                'emotion_params': emotion_params,
                'duration_estimate': len(text) * 0.3  # 粗略估计
            }
            
        except Exception as e:
            self.stats['errors'] += 1
            self.last_error = str(e)
            return {
                'success': False,
                'error': str(e),
                'text': text
            }
    
    def synthesize_stream(self, 
                         text: str,
                         voice: str = None,
                         emotion: str = None) -> Iterator[bytes]:
        """
        流式合成（适合长文本）
        
        Args:
            text: 要合成的文本
            voice: 声音ID
            emotion: 情感风格
            
        Yields:
            bytes: 音频数据块
        """
        if self.status != "ready":
            raise RuntimeError(f"TTS未就绪: {self.status}")
        
        voice = voice or self.default_voice
        emotion = emotion or self.default_emotion
        emotion_params = self.EMOTIONS.get(emotion, self.EMOTIONS['neutral'])
        
        # 流式生成 (elevenlabs v2.x API)
        voice_settings = VoiceSettings(
            stability=emotion_params['stability'],
            similarity_boost=emotion_params['similarity_boost'],
            style=emotion_params['style']
        )
        
        audio_stream = self.client.text_to_speech.convert_as_stream(
            text=text,
            voice_id=voice,
            model_id=self.model,
            voice_settings=voice_settings
        )
        
        for chunk in audio_stream:
            yield chunk
    
    def synthesize_batch(self,
                        texts: List[str],
                        voice: str = None,
                        emotion: str = None) -> List[Dict[str, Any]]:
        """
        批量合成
        
        Args:
            texts: 文本列表
            voice: 声音ID
            emotion: 情感风格
            
        Returns:
            list: 合成结果列表
        """
        results = []
        for text in texts:
            result = self.synthesize(text, voice=voice, emotion=emotion)
            results.append(result)
        return results
    
    def validate_emotion_mapping(self, emotion: str) -> Dict[str, Any]:
        """
        验证情感映射的跨模态一致性
        
        Args:
            emotion: 情感名称
            
        Returns:
            dict: 验证结果
        """
        if emotion not in self.EMOTIONS:
            return {
                'emotion': emotion,
                'valid': False,
                'error': f'无效情感，可用: {list(self.EMOTIONS.keys())}'
            }
        
        params = self.EMOTIONS[emotion]
        
        # 验证参数范围
        validations = {
            'stability_range': 0 <= params['stability'] <= 1,
            'similarity_range': 0 <= params['similarity_boost'] <= 1,
            'style_range': -1 <= params['style'] <= 1
        }
        
        # 情感一致性检查
        consistency_checks = self._check_emotion_consistency(emotion, params)
        
        return {
            'emotion': emotion,
            'valid': all(validations.values()),
            'params': params,
            'validations': validations,
            'consistency': consistency_checks,
            'category': self._get_emotion_category(emotion)
        }
    
    def _check_emotion_consistency(self, emotion: str, params: Dict) -> Dict[str, Any]:
        """检查情感参数一致性"""
        checks = {}
        
        # 稳定性与情感类型的关系
        if emotion in ['calm', 'serious', 'authoritative']:
            checks['stability_match'] = params['stability'] >= 0.6
        elif emotion in ['excited', 'angry', 'fearful']:
            checks['stability_match'] = params['stability'] <= 0.4
        else:
            checks['stability_match'] = True
        
        # 风格值与情感类型的关系
        if emotion in ['happy', 'excited', 'humorous', 'angry']:
            checks['style_match'] = params['style'] > 0
        elif emotion in ['calm', 'sad', 'whisper', 'serious']:
            checks['style_match'] = params['style'] < 0
        else:
            checks['style_match'] = True
        
        return checks
    
    def _get_emotion_category(self, emotion: str) -> str:
        """获取情感分类"""
        for category, emotions in self.EMOTION_CATEGORIES.items():
            if emotion in emotions:
                return category
        return 'unknown'
    
    def test_all_emotions(self, text: str = "这是一段测试语音，用于验证不同情感风格的效果。") -> List[Dict[str, Any]]:
        """
        测试所有16种情感
        
        Args:
            text: 测试文本
            
        Returns:
            list: 所有情感的测试结果
        """
        results = []
        print(f"🧪 测试所有16种情感，文本: {text}")
        print("=" * 60)
        
        for emotion in self.EMOTIONS.keys():
            print(f"  测试情感: {emotion}...", end=" ")
            
            # 验证映射
            validation = self.validate_emotion_mapping(emotion)
            
            # 尝试合成（如果API已配置）
            synthesis_result = None
            if self.status == "ready":
                synthesis_result = self.synthesize(text, emotion=emotion)
            
            result = {
                'emotion': emotion,
                'validation': validation,
                'synthesis': synthesis_result
            }
            results.append(result)
            
            status = "✅" if validation['valid'] else "❌"
            print(f"{status}")
        
        print("=" * 60)
        print(f"✅ 测试完成: {len(results)} 种情感")
        return results
    
    def _generate_cache_path(self, text: str, voice: str, emotion: str, speed: float) -> str:
        """生成缓存文件路径"""
        content = f"{text}|{voice}|{emotion}|{speed}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
        filename = f"{voice}_{emotion}_{hash_val}.mp3"
        return os.path.join(self.cache_dir, filename)
    
    def _get_cache(self, text: str, voice: str, emotion: str, speed: float) -> Optional[str]:
        """检查缓存"""
        cache_path = self._generate_cache_path(text, voice, emotion, speed)
        if os.path.exists(cache_path):
            return cache_path
        return None
    
    def get_voices(self) -> Dict[str, Any]:
        """获取可用声音列表"""
        if self.status == "ready" and self.client:
            try:
                available_voices = self.client.voices.get_all()
                voice_names = [v.name for v in available_voices.voices] if hasattr(available_voices, 'voices') else []
                return {
                    'preset': self.VOICES,
                    'preset_count': len(self.VOICES),
                    'available': voice_names,
                    'available_count': len(voice_names)
                }
            except Exception as e:
                return {
                    'preset': self.VOICES,
                    'preset_count': len(self.VOICES),
                    'available': [],
                    'error': str(e)
                }
        return {
            'preset': self.VOICES,
            'preset_count': len(self.VOICES),
            'available': [],
            'status': self.status
        }
    
    def get_emotions(self) -> Dict[str, Any]:
        """获取可用情感列表"""
        return {
            'emotions': self.EMOTIONS,
            'count': len(self.EMOTIONS),
            'categories': self.EMOTION_CATEGORIES
        }
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'status': self.status,
            'api_key_configured': bool(self.api_key),
            'api_key_preview': self.api_key[:8] + '...' if self.api_key else None,
            'default_voice': self.default_voice,
            'default_emotion': self.default_emotion,
            'model': self.model,
            'cache_dir': self.cache_dir,
            'last_error': self.last_error,
            'elevenlabs_available': ELEVENLABS_AVAILABLE,
            'stats': self.stats,
            'voices_count': len(self.VOICES),
            'emotions_count': len(self.EMOTIONS)
        }
    
    def configure_api_key(self, api_key: str) -> Dict[str, Any]:
        """配置API密钥"""
        self.api_key = api_key
        os.environ['ELEVENLABS_API_KEY'] = api_key
        
        if ELEVENLABS_AVAILABLE:
            try:
                self.client = ElevenLabsClient(api_key=api_key)
                self.status = "ready"
                return {'success': True, 'status': 'ready'}
            except Exception as e:
                self.status = "error"
                self.last_error = str(e)
                return {'success': False, 'error': str(e)}
        else:
            return {'success': False, 'error': 'elevenlabs库未安装'}
    
    def clear_cache(self) -> Dict[str, Any]:
        """清除缓存"""
        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.mp3'):
                os.remove(os.path.join(self.cache_dir, filename))
                count += 1
        return {'success': True, 'cleared_files': count}

# 全局实例
elevenlabs_tts = ElevenLabsTTS()

def main():
    """主函数 - CLI入口"""
    import sys
    
    if len(sys.argv) < 2:
        # 显示状态
        status = elevenlabs_tts.get_status()
        print("=" * 60)
        print("🎙️ ElevenLabs语音合成系统 v1.1")
        print("=" * 60)
        print(f"状态: {status['status']}")
        print(f"API密钥: {'已配置' if status['api_key_configured'] else '未配置'}")
        print(f"默认声音: {status['default_voice']}")
        print(f"默认情感: {status['default_emotion']}")
        print(f"模型: {status['model']}")
        print(f"缓存目录: {status['cache_dir']}")
        print(f"声音数: {status['voices_count']} | 情感数: {status['emotions_count']}")
        print("=" * 60)
        print("\n可用声音:")
        for vid, vinfo in elevenlabs_tts.VOICES.items():
            gender_emoji = "👩" if vinfo['gender'] == 'female' else "👨"
            print(f"  {gender_emoji} {vid}: {vinfo['style']}")
        print("\n16种情感映射:")
        for category, emotions in elevenlabs_tts.EMOTION_CATEGORIES.items():
            emoji = {'basic': '🔵', 'positive': '🟢', 'professional': '🟠', 'special': '🟣'}[category]
            print(f"  {emoji} {category}: {', '.join(emotions)}")
        print("=" * 60)
        print("\n用法:")
        print("  python elevenlabs_tts.py status")
        print("  python elevenlabs_tts.py configure <api_key>")
        print("  python elevenlabs_tts.py synthesize '文本' [--voice Bella] [--emotion excited] [--output file.mp3]")
        print("  python elevenlabs_tts.py test-emotions")
        print("  python elevenlabs_tts.py clear-cache")
        return
    
    command = sys.argv[1]
    
    if command == "status":
        print(json.dumps(elevenlabs_tts.get_status(), indent=2, ensure_ascii=False))
    
    elif command == "configure":
        if len(sys.argv) < 3:
            print("❌ 错误: 需要提供API密钥")
            print("   获取地址: https://elevenlabs.io/app/settings/api-keys")
            return
        api_key = sys.argv[2]
        result = elevenlabs_tts.configure_api_key(api_key)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == "synthesize":
        if len(sys.argv) < 3:
            print("❌ 错误: 需要提供文本")
            return
        text = sys.argv[2]
        
        # 解析可选参数
        voice = elevenlabs_tts.default_voice
        emotion = elevenlabs_tts.default_emotion
        output = None
        
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--voice" and i + 1 < len(sys.argv):
                voice = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--emotion" and i + 1 < len(sys.argv):
                emotion = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--output" and i + 1 < len(sys.argv):
                output = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        result = elevenlabs_tts.synthesize(text, voice=voice, emotion=emotion, output_path=output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == "test-emotions":
        text = "这是一段测试语音，用于验证不同情感风格的效果。"
        if len(sys.argv) > 2:
            text = sys.argv[2]
        results = elevenlabs_tts.test_all_emotions(text)
        print("\n详细结果:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    
    elif command == "clear-cache":
        result = elevenlabs_tts.clear_cache()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == "voices":
        voices_info = elevenlabs_tts.get_voices()
        print(json.dumps(voices_info, indent=2, ensure_ascii=False))
    
    elif command == "emotions":
        emotions_info = elevenlabs_tts.get_emotions()
        print(json.dumps(emotions_info, indent=2, ensure_ascii=False))
    
    else:
        print(f"❌ 未知命令: {command}")
        print("可用命令: status, configure, synthesize, test-emotions, clear-cache, voices, emotions")

if __name__ == '__main__':
    main()
