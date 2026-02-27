#!/usr/bin/env python3
"""
多模态Agent基础测试
验证视觉理解、语音转录、TTS功能
"""

import asyncio
import os
import tempfile
from multimodal_agent import MultimodalAgent, VisionInput, AudioInput


async def test_vision():
    """测试视觉理解功能"""
    print("=" * 50)
    print("🖼️  测试视觉理解 (GPT-4V)")
    print("=" * 50)
    
    # 创建一个简单的测试图片URL（使用公开的图片）
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    
    async with MultimodalAgent() as agent:
        try:
            vision_input = VisionInput(
                image_url=test_image_url,
                prompt="请详细描述这张图片中的场景"
            )
            
            response = await agent.vision_understand(vision_input)
            print(f"✅ 视觉理解成功")
            print(f"📝 响应: {response[:200]}...")
            return True
        except Exception as e:
            print(f"❌ 视觉理解失败: {e}")
            return False


async def test_tts():
    """测试TTS语音合成功能"""
    print("\n" + "=" * 50)
    print("🔊 测试TTS语音合成")
    print("=" * 50)
    
    async with MultimodalAgent() as agent:
        try:
            test_text = "你好，我是多模态AI助手，我可以理解图片、语音和文字。"
            
            tts_output = await agent.text_to_speech(
                text=test_text,
                voice="nova",
                speed=1.0
            )
            
            # 保存音频文件
            output_path = "test_output.mp3"
            with open(output_path, "wb") as f:
                f.write(tts_output.audio_bytes)
            
            print(f"✅ TTS合成成功")
            print(f"📁 音频已保存: {output_path}")
            print(f"📊 音频大小: {len(tts_output.audio_bytes)} bytes")
            return True
        except Exception as e:
            print(f"❌ TTS合成失败: {e}")
            return False


async def test_whisper():
    """测试Whisper语音转录功能"""
    print("\n" + "=" * 50)
    print("🎤 测试Whisper语音转录")
    print("=" * 50)
    
    # 注意：这里需要一个真实的音频文件来测试
    # 如果没有，会显示跳过信息
    test_audio_path = "test_audio.mp3"
    
    if not os.path.exists(test_audio_path):
        print(f"⚠️  跳过测试: 未找到测试音频文件 {test_audio_path}")
        print(f"💡 提示: 请提供一个MP3格式的音频文件进行测试")
        return None
    
    async with MultimodalAgent() as agent:
        try:
            audio_input = AudioInput(
                audio_path=test_audio_path,
                language="zh"
            )
            
            transcribed_text = await agent.speech_to_text(audio_input)
            print(f"✅ 语音转录成功")
            print(f"📝 转录结果: {transcribed_text}")
            return True
        except Exception as e:
            print(f"❌ 语音转录失败: {e}")
            return False


async def test_multimodal_chat():
    """测试多模态对话功能"""
    print("\n" + "=" * 50)
    print("🤖 测试多模态对话")
    print("=" * 50)
    
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    
    async with MultimodalAgent() as agent:
        try:
            # 测试图文对话
            vision_input = VisionInput(
                image_url=test_image_url,
                prompt="这张图片展示了什么？"
            )
            
            result = await agent.multimodal_chat(
                text="请分析这张图片",
                vision_input=vision_input,
                enable_tts=False
            )
            
            print(f"✅ 多模态对话成功")
            print(f"📝 响应: {result['text'][:200]}...")
            return True
        except Exception as e:
            print(f"❌ 多模态对话失败: {e}")
            return False


async def test_sync_vision():
    """测试同步视觉理解（用于非异步环境）"""
    print("\n" + "=" * 50)
    print("🖼️  测试同步视觉理解")
    print("=" * 50)
    
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    
    try:
        agent = MultimodalAgent()
        
        vision_input = VisionInput(
            image_url=test_image_url,
            prompt="请描述这张图片"
        )
        
        response = await agent.vision_understand(vision_input)
        await agent.close()
        
        print(f"✅ 同步视觉理解成功")
        print(f"📝 响应: {response[:200]}...")
        return True
    except Exception as e:
        print(f"❌ 同步视觉理解失败: {e}")
        return False


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "🚀" * 25)
    print("   多模态Agent核心测试")
    print("🚀" * 25 + "\n")
    
    results = {}
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  警告: 未设置OPENAI_API_KEY环境变量")
        print("💡 请设置: export OPENAI_API_KEY='your-api-key'")
        print("\n继续运行测试（预计会失败）...\n")
    
    # 运行测试
    results["vision"] = await test_vision()
    results["tts"] = await test_tts()
    results["whisper"] = await test_whisper()
    results["multimodal_chat"] = await test_multimodal_chat()
    
    # 打印测试结果汇总
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "⏭️  跳过"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    print(f"\n总计: {passed} 通过, {failed} 失败, {skipped} 跳过")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_all_tests())
