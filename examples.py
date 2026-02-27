#!/usr/bin/env python3
"""
多模态Agent使用示例
展示各种使用场景
"""

import asyncio
import os
from multimodal_agent import MultimodalAgent, VisionInput, AudioInput


async def example_1_basic_vision():
    """示例1: 基础视觉理解"""
    print("\n" + "="*50)
    print("示例1: 基础视觉理解")
    print("="*50)
    
    async with MultimodalAgent() as agent:
        # 使用网络图片
        vision_input = VisionInput(
            image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            prompt="请描述这张图片中的自然风景"
        )
        
        result = await agent.vision_understand(vision_input)
        print(f"🖼️  图片描述: {result}")


async def example_2_local_image():
    """示例2: 本地图片分析"""
    print("\n" + "="*50)
    print("示例2: 本地图片分析")
    print("="*50)
    
    # 检查本地图片是否存在
    local_image = "test_image.jpg"
    if not os.path.exists(local_image):
        print(f"⚠️  本地图片 {local_image} 不存在，跳过此示例")
        return
    
    async with MultimodalAgent() as agent:
        vision_input = VisionInput(
            image_path=local_image,
            prompt="详细分析这张图片的内容"
        )
        
        result = await agent.vision_understand(vision_input)
        print(f"🖼️  图片分析: {result}")


async def example_3_tts_various_voices():
    """示例3: 不同声音的TTS"""
    print("\n" + "="*50)
    print("示例3: 不同声音的TTS")
    print("="*50)
    
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    test_text = "你好，我是多模态AI助手。"
    
    async with MultimodalAgent() as agent:
        for voice in voices:
            tts_output = await agent.text_to_speech(
                text=test_text,
                voice=voice
            )
            
            # 保存音频文件
            output_file = f"tts_{voice}.mp3"
            with open(output_file, "wb") as f:
                f.write(tts_output.audio_bytes)
            
            print(f"🔊 声音 {voice}: 已保存到 {output_file}")


async def example_4_multimodal_conversation():
    """示例4: 多模态对话"""
    print("\n" + "="*50)
    print("示例4: 多模态对话")
    print("="*50)
    
    async with MultimodalAgent() as agent:
        # 图文对话
        vision_input = VisionInput(
            image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            prompt="这张图片展示了什么场景？"
        )
        
        result = await agent.multimodal_chat(
            text="请详细描述",
            vision_input=vision_input,
            enable_tts=True  # 同时生成语音回复
        )
        
        print(f"📝 文本回复: {result['text'][:200]}...")
        
        if result['audio']:
            with open("multimodal_response.mp3", "wb") as f:
                f.write(result['audio'].audio_bytes)
            print(f"🔊 语音回复已保存到 multimodal_response.mp3")


async def example_5_voice_to_text():
    """示例5: 语音转文字"""
    print("\n" + "="*50)
    print("示例5: 语音转文字")
    print("="*50)
    
    # 检查音频文件
    audio_file = "test_audio.mp3"
    if not os.path.exists(audio_file):
        print(f"⚠️  音频文件 {audio_file} 不存在，跳过此示例")
        print("💡 请准备一个MP3格式的音频文件")
        return
    
    async with MultimodalAgent() as agent:
        audio_input = AudioInput(
            audio_path=audio_file,
            language="zh"  # 指定语言为中文
        )
        
        text = await agent.speech_to_text(audio_input)
        print(f"🎤 转录结果: {text}")


async def example_6_complete_workflow():
    """示例6: 完整工作流程 - 语音+视觉+语音回复"""
    print("\n" + "="*50)
    print("示例6: 完整工作流程")
    print("="*50)
    
    async with MultimodalAgent() as agent:
        # 模拟用户语音提问
        print("1️⃣ 用户语音提问: '这张图片里有什么？'")
        
        # 这里假设我们已经有了语音转录的文本
        user_question = "这张图片里有什么？"
        
        # 分析图片
        vision_input = VisionInput(
            image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            prompt=user_question
        )
        
        # 获取视觉理解结果
        vision_result = await agent.vision_understand(vision_input)
        print(f"2️⃣ AI视觉分析: {vision_result[:150]}...")
        
        # 生成语音回复
        tts_output = await agent.text_to_speech(
            text=vision_result,
            voice="nova"
        )
        
        with open("workflow_response.mp3", "wb") as f:
            f.write(tts_output.audio_bytes)
        
        print(f"3️⃣ 语音回复已生成: workflow_response.mp3")


async def run_all_examples():
    """运行所有示例"""
    print("\n" + "🚀" * 25)
    print("   多模态Agent使用示例")
    print("🚀" * 25)
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  警告: 未设置OPENAI_API_KEY环境变量")
        print("💡 请设置: export OPENAI_API_KEY='your-api-key'")
        return
    
    examples = [
        example_1_basic_vision,
        example_2_local_image,
        example_3_tts_various_voices,
        example_4_multimodal_conversation,
        example_5_voice_to_text,
        example_6_complete_workflow,
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"❌ 示例失败: {e}")
    
    print("\n" + "="*50)
    print("✅ 所有示例执行完毕")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(run_all_examples())
