#!/usr/bin/env python3
"""
VLA（视觉-语言-动作）推理示例

演示如何使用VLA模型进行端到端的视觉-语言-动作推理，包括：
- 加载预训练VLA模型
- 处理视觉和语言输入
- 生成机器人动作
- 执行复杂任务

作者: Embodied AI Research Team
版本: 1.0.0
"""

import numpy as np
import torch
import sys
import time
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, '..')

from embodied_agent import (
    EmbodiedAgent,
    PerceptionData,
    ActionCommand,
    VLAPolicy,
    create_agent
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VLAConfig:
    """VLA模型配置"""
    model_name: str = "openvla"
    checkpoint_path: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    image_size: int = 224
    action_dim: int = 7
    num_action_bins: int = 256
    max_seq_length: int = 512


class VLAInference:
    """
    VLA推理引擎
    
    封装VLA模型的推理流程，支持多种VLA架构。
    """
    
    def __init__(self, config: VLAConfig):
        """
        初始化VLA推理引擎
        
        Args:
            config: VLA配置
        """
        self.config = config
        self.device = config.device
        
        # 加载模型
        self.model = self._load_model()
        
        logger.info(f"VLA Inference Engine initialized")
        logger.info(f"  Model: {config.model_name}")
        logger.info(f"  Device: {config.device}")
    
    def _load_model(self):
        """加载VLA模型"""
        if self.config.model_name == "openvla":
            # 使用OpenVLA模型
            return self._load_openvla()
        elif self.config.model_name == "rt2":
            # 使用RT-2模型
            return self._load_rt2()
        elif self.config.model_name == "octo":
            # 使用Octo模型
            return self._load_octo()
        else:
            # 使用本地VLA策略
            return VLAPolicy(
                action_dim=self.config.action_dim,
                num_action_bins=self.config.num_action_bins,
                device=self.config.device
            )
    
    def _load_openvla(self):
        """加载OpenVLA模型"""
        try:
            # 实际应用中应加载真实的OpenVLA模型
            # from transformers import AutoModelForVision2Seq
            # model = AutoModelForVision2Seq.from_pretrained(
            #     "openvla/openvla-7b",
            #     torch_dtype=torch.bfloat16,
            #     low_cpu_mem_usage=True
            # )
            logger.info("Loading OpenVLA model (mock)")
            return VLAPolicy(
                action_dim=self.config.action_dim,
                num_action_bins=self.config.num_action_bins,
                device=self.config.device
            )
        except Exception as e:
            logger.warning(f"Failed to load OpenVLA: {e}, using mock")
            return VLAPolicy(
                action_dim=self.config.action_dim,
                num_action_bins=self.config.num_action_bins,
                device=self.config.device
            )
    
    def _load_rt2(self):
        """加载RT-2模型"""
        logger.info("Loading RT-2 model (mock)")
        return VLAPolicy(
            action_dim=self.config.action_dim,
            num_action_bins=self.config.num_action_bins,
            device=self.config.device
        )
    
    def _load_octo(self):
        """加载Octo模型"""
        logger.info("Loading Octo model (mock)")
        return VLAPolicy(
            action_dim=self.config.action_dim,
            num_action_bins=self.config.num_action_bins,
            device=self.config.device
        )
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: 输入图像 [H, W, C]
            
        Returns:
            torch.Tensor: 预处理后的图像 [1, C, H, W]
        """
        # 调整大小
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        tensor = transform(image).unsqueeze(0).to(self.device)
        return tensor
    
    def preprocess_language(self, instruction: str) -> torch.Tensor:
        """
        预处理语言指令
        
        Args:
            instruction: 语言指令
            
        Returns:
            torch.Tensor: 语言token
        """
        # 简化的tokenization
        # 实际应使用tokenizer
        tokens = torch.randint(0, 30000, (1, 20)).to(self.device)
        return tokens
    
    def predict_action(
        self,
        image: np.ndarray,
        instruction: str,
        proprioception: Optional[np.ndarray] = None
    ) -> ActionCommand:
        """
        预测动作
        
        Args:
            image: 视觉输入
            instruction: 语言指令
            proprioception: 本体感觉数据（可选）
            
        Returns:
            ActionCommand: 预测的动作
        """
        # 预处理输入
        image_tensor = self.preprocess_image(image)
        language_tokens = self.preprocess_language(instruction)
        
        # 创建感知数据
        perception = PerceptionData(
            rgb_image=image,
            joint_positions=proprioception,
            language_instruction=instruction,
            timestamp=time.time()
        )
        
        # 模型推理
        with torch.no_grad():
            action = self.model.predict(perception)
        
        return action
    
    def predict_action_sequence(
        self,
        image: np.ndarray,
        instruction: str,
        sequence_length: int = 10,
        proprioception: Optional[np.ndarray] = None
    ) -> List[ActionCommand]:
        """
        预测动作序列（动作分块）
        
        Args:
            image: 视觉输入
            instruction: 语言指令
            sequence_length: 序列长度
            proprioception: 本体感觉数据
            
        Returns:
            List[ActionCommand]: 动作序列
        """
        actions = []
        
        for i in range(sequence_length):
            action = self.predict_action(image, instruction, proprioception)
            actions.append(action)
            
            # 更新本体感觉（模拟）
            if action.joint_positions is not None:
                proprioception = action.joint_positions
        
        return actions


class VLATaskExecutor:
    """
    VLA任务执行器
    
    使用VLA模型执行复杂的机器人任务。
    """
    
    def __init__(
        self,
        agent: EmbodiedAgent,
        vla_config: Optional[VLAConfig] = None
    ):
        """
        初始化任务执行器
        
        Args:
            agent: 具身智能体
            vla_config: VLA配置
        """
        self.agent = agent
        self.vla = VLAInference(vla_config or VLAConfig())
        
        logger.info("VLATaskExecutor initialized")
    
    def execute_instruction(
        self,
        instruction: str,
        max_steps: int = 100,
        step_frequency: float = 10.0
    ) -> bool:
        """
        执行语言指令
        
        Args:
            instruction: 语言指令
            max_steps: 最大执行步数
            step_frequency: 控制频率（Hz）
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"Executing instruction: {instruction}")
        
        step_duration = 1.0 / step_frequency
        
        for step in range(max_steps):
            # 获取感知
            perception = self.agent.perceive()
            
            # 检查是否有图像数据
            if perception.rgb_image is None:
                logger.warning("No image data available, using mock")
                perception.rgb_image = np.random.randint(
                    0, 255, (224, 224, 3), dtype=np.uint8
                )
            
            # VLA推理
            action = self.vla.predict_action(
                image=perception.rgb_image,
                instruction=instruction,
                proprioception=perception.joint_positions
            )
            
            # 执行动作
            success = self.agent.act(action)
            
            if not success:
                logger.error("Action execution failed")
                return False
            
            # 检查任务完成（简化版）
            # 实际应使用成功检测模型
            if step > 50:  # 假设50步后完成
                logger.info("Task completed")
                return True
            
            time.sleep(step_duration)
        
        logger.warning("Task timed out")
        return False
    
    def execute_instruction_with_chunking(
        self,
        instruction: str,
        chunk_size: int = 10,
        max_chunks: int = 10
    ) -> bool:
        """
        使用动作分块执行指令
        
        Args:
            instruction: 语言指令
            chunk_size: 动作块大小
            max_chunks: 最大块数
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"Executing with chunking: {instruction}")
        
        for chunk_idx in range(max_chunks):
            logger.info(f"Executing chunk {chunk_idx + 1}/{max_chunks}")
            
            # 获取感知
            perception = self.agent.perceive()
            
            if perception.rgb_image is None:
                perception.rgb_image = np.random.randint(
                    0, 255, (224, 224, 3), dtype=np.uint8
                )
            
            # 预测动作序列
            actions = self.vla.predict_action_sequence(
                image=perception.rgb_image,
                instruction=instruction,
                sequence_length=chunk_size,
                proprioception=perception.joint_positions
            )
            
            # 执行动作序列
            for i, action in enumerate(actions):
                logger.debug(f"Executing action {i + 1}/{len(actions)}")
                
                success = self.agent.act(action)
                if not success:
                    logger.error("Action execution failed")
                    return False
                
                time.sleep(0.1)
        
        logger.info("Task completed with chunking")
        return True
    
    def execute_task_plan(
        self,
        high_level_instruction: str,
        subtasks: List[str]
    ) -> bool:
        """
        执行分层任务计划
        
        Args:
            high_level_instruction: 高层指令
            subtasks: 子任务列表
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"Executing task plan: {high_level_instruction}")
        logger.info(f"Subtasks: {subtasks}")
        
        for i, subtask in enumerate(subtasks):
            logger.info(f"Executing subtask {i + 1}/{len(subtasks)}: {subtask}")
            
            success = self.execute_instruction(subtask, max_steps=50)
            
            if not success:
                logger.error(f"Subtask failed: {subtask}")
                return False
            
            time.sleep(0.5)
        
        logger.info("Task plan completed successfully")
        return True


def demo_single_instruction():
    """演示单指令执行"""
    logger.info("=" * 50)
    logger.info("Demo: Single Instruction Execution")
    logger.info("=" * 50)
    
    # 创建智能体
    agent = create_agent(
        robot_name="vla_robot",
        use_simulator=False,
        policy_type="vla"
    )
    
    # 创建VLA执行器
    config = VLAConfig(model_name="mock")
    executor = VLATaskExecutor(agent, config)
    
    # 执行指令
    instruction = "pick up the red cube from the table"
    executor.execute_instruction(instruction, max_steps=20)
    
    logger.info("Single instruction demo completed")


def demo_action_chunking():
    """演示动作分块"""
    logger.info("=" * 50)
    logger.info("Demo: Action Chunking")
    logger.info("=" * 50)
    
    # 创建智能体
    agent = create_agent(
        robot_name="vla_robot",
        use_simulator=False,
        policy_type="vla"
    )
    
    # 创建VLA执行器
    config = VLAConfig(model_name="mock")
    executor = VLATaskExecutor(agent, config)
    
    # 执行指令（使用动作分块）
    instruction = "move the blue block to the left side"
    executor.execute_instruction_with_chunking(
        instruction,
        chunk_size=5,
        max_chunks=3
    )
    
    logger.info("Action chunking demo completed")


def demo_task_planning():
    """演示任务规划"""
    logger.info("=" * 50)
    logger.info("Demo: Task Planning with VLA")
    logger.info("=" * 50)
    
    # 创建智能体
    agent = create_agent(
        robot_name="vla_robot",
        use_simulator=False,
        policy_type="vla"
    )
    
    # 创建VLA执行器
    config = VLAConfig(model_name="mock")
    executor = VLATaskExecutor(agent, config)
    
    # 定义任务计划
    high_level_instruction = "prepare a cup of coffee"
    subtasks = [
        "pick up the coffee cup",
        "move to the coffee machine",
        "place cup under the dispenser",
        "press the brew button",
        "wait for coffee to finish",
        "remove the cup",
    ]
    
    # 执行任务计划
    executor.execute_task_plan(high_level_instruction, subtasks)
    
    logger.info("Task planning demo completed")


def demo_multi_modal_fusion():
    """演示多模态融合"""
    logger.info("=" * 50)
    logger.info("Demo: Multi-modal Fusion")
    logger.info("=" * 50)
    
    # 创建VLA推理引擎
    config = VLAConfig(model_name="mock")
    vla = VLAInference(config)
    
    # 创建模拟的多模态输入
    test_cases = [
        {
            "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "instruction": "pick up the red cube",
            "proprioception": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        },
        {
            "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "instruction": "place the object on the shelf",
            "proprioception": np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        },
        {
            "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "instruction": "push the button",
            "proprioception": np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        logger.info(f"Test case {i + 1}: {test_case['instruction']}")
        
        action = vla.predict_action(
            image=test_case["image"],
            instruction=test_case["instruction"],
            proprioception=test_case["proprioception"]
        )
        
        logger.info(f"Predicted action: {action.joint_positions}")
    
    logger.info("Multi-modal fusion demo completed")


def demo_different_vla_models():
    """演示不同VLA模型"""
    logger.info("=" * 50)
    logger.info("Demo: Different VLA Models")
    logger.info("=" * 50)
    
    models = ["mock", "openvla", "rt2", "octo"]
    
    for model_name in models:
        logger.info(f"\nTesting model: {model_name}")
        
        try:
            config = VLAConfig(model_name=model_name)
            vla = VLAInference(config)
            
            # 测试推理
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            instruction = "test instruction"
            
            action = vla.predict_action(image, instruction)
            logger.info(f"  Action shape: {action.joint_positions.shape if action.joint_positions is not None else None}")
            
        except Exception as e:
            logger.error(f"  Failed to test {model_name}: {e}")
    
    logger.info("\nDifferent VLA models demo completed")


def run_all_demos():
    """运行所有演示"""
    logger.info("\n" + "=" * 50)
    logger.info("VLA Inference Examples")
    logger.info("=" * 50 + "\n")
    
    try:
        demo_single_instruction()
        print()
        
        demo_action_chunking()
        print()
        
        demo_task_planning()
        print()
        
        demo_multi_modal_fusion()
        print()
        
        demo_different_vla_models()
        print()
        
        logger.info("=" * 50)
        logger.info("All VLA demos completed successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLA Inference Examples")
    parser.add_argument(
        "--demo",
        type=str,
        choices=["single", "chunking", "planning", "multimodal", "models", "all"],
        default="all",
        help="Which demo to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock",
        help="VLA model to use"
    )
    
    args = parser.parse_args()
    
    if args.demo == "single":
        demo_single_instruction()
    elif args.demo == "chunking":
        demo_action_chunking()
    elif args.demo == "planning":
        demo_task_planning()
    elif args.demo == "multimodal":
        demo_multi_modal_fusion()
    elif args.demo == "models":
        demo_different_vla_models()
    else:
        run_all_demos()
