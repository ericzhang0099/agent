#!/usr/bin/env python3
"""
具身智能核心智能体类 (Embodied Agent Core Class)

该类实现了具身智能体的核心功能，包括：
- 多模态感知（视觉、语言、本体感觉）
- 动作生成与控制
- 任务规划与执行
- 与ROS2和模拟器的接口

作者: Embodied AI Research Team
版本: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """智能体状态枚举"""
    IDLE = "idle"
    PERCEIVING = "perceiving"
    PLANNING = "planning"
    EXECUTING = "executing"
    ERROR = "error"


@dataclass
class PerceptionData:
    """感知数据结构"""
    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    end_effector_pose: Optional[np.ndarray] = None
    language_instruction: Optional[str] = None
    timestamp: float = field(default_factory=lambda: 0.0)


@dataclass
class ActionCommand:
    """动作命令数据结构"""
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    joint_torques: Optional[np.ndarray] = None
    end_effector_pose: Optional[np.ndarray] = None
    gripper_state: Optional[float] = None  # 0.0=关闭, 1.0=打开
    timestamp: float = field(default_factory=lambda: 0.0)


@dataclass
class TaskPlan:
    """任务计划数据结构"""
    task_description: str = ""
    subtasks: List[str] = field(default_factory=list)
    current_step: int = 0
    is_complete: bool = False


class BasePolicy(ABC):
    """策略基类"""
    
    @abstractmethod
    def predict(self, perception: PerceptionData) -> ActionCommand:
        """根据感知数据预测动作"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置策略状态"""
        pass


class VLAPolicy(BasePolicy):
    """
    视觉-语言-动作策略 (Vision-Language-Action Policy)
    
    实现基于Transformer的VLA策略，支持：
    - 视觉编码 (ViT/ResNet)
    - 语言编码 (BERT/GPT)
    - 多模态融合
    - 动作解码
    """
    
    def __init__(
        self,
        vision_encoder: str = "vit",
        language_encoder: str = "bert",
        action_dim: int = 7,
        hidden_dim: int = 512,
        num_action_bins: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.num_action_bins = num_action_bins
        
        # 视觉编码器 (简化版)
        self.vision_encoder = self._build_vision_encoder(vision_encoder, hidden_dim)
        
        # 语言编码器 (简化版)
        self.language_encoder = self._build_language_encoder(language_encoder, hidden_dim)
        
        # 多模态融合层
        self.fusion_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        
        # 动作解码器
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * num_action_bins)
        )
        
        self.to(device)
        logger.info(f"VLA Policy initialized on {device}")
    
    def _build_vision_encoder(self, arch: str, hidden_dim: int) -> nn.Module:
        """构建视觉编码器"""
        if arch == "vit":
            # 简化的ViT实现
            return nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((14, 14)),
                nn.Flatten(),
                nn.Linear(64 * 14 * 14, hidden_dim)
            )
        else:  # resnet
            return nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, hidden_dim)
            )
    
    def _build_language_encoder(self, arch: str, hidden_dim: int) -> nn.Module:
        """构建语言编码器"""
        # 简化的语言编码器
        return nn.Sequential(
            nn.Embedding(30000, 128),  # 假设词汇表大小
            nn.LSTM(128, hidden_dim, batch_first=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
    
    def forward(
        self,
        rgb_image: torch.Tensor,
        language_tokens: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        # 视觉编码
        visual_features = self.vision_encoder(rgb_image)
        
        # 语言编码
        language_features = self.language_encoder(language_tokens)
        
        # 多模态融合
        combined = torch.stack([visual_features, language_features], dim=1)
        fused = self.fusion_layer(combined)
        fused = fused.mean(dim=1)  # 平均池化
        
        # 动作解码
        action_logits = self.action_decoder(fused)
        action_logits = action_logits.view(-1, self.action_dim, self.num_action_bins)
        
        return action_logits
    
    def predict(self, perception: PerceptionData) -> ActionCommand:
        """预测动作"""
        self.eval()
        with torch.no_grad():
            # 预处理输入
            rgb = torch.from_numpy(perception.rgb_image).float().unsqueeze(0).to(self.device)
            rgb = rgb.permute(0, 3, 1, 2) / 255.0  # [B, H, W, C] -> [B, C, H, W]
            
            # 简化的语言tokenization
            lang_tokens = torch.randint(0, 30000, (1, 10)).to(self.device)
            
            # 推理
            action_logits = self.forward(rgb, lang_tokens)
            
            # 选择最可能的动作
            action_bins = action_logits.argmax(dim=-1).squeeze(0).cpu().numpy()
            
            # 反归一化到实际动作空间
            actions = (action_bins / self.num_action_bins) * 2 - 1  # [-1, 1]
            
            return ActionCommand(
                joint_positions=actions,
                timestamp=perception.timestamp
            )
    
    def reset(self):
        """重置策略"""
        pass


class EmbodiedAgent:
    """
    具身智能核心类
    
    该类是具身智能系统的核心，负责协调感知、规划、控制等模块。
    """
    
    def __init__(
        self,
        robot_name: str = "default_robot",
        action_dim: int = 7,
        use_ros2: bool = False,
        use_simulator: bool = True,
        simulator_type: str = "isaac_sim",
        policy_type: str = "vla",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化具身智能体
        
        Args:
            robot_name: 机器人名称
            action_dim: 动作维度（关节数）
            use_ros2: 是否使用ROS2
            use_simulator: 是否使用模拟器
            simulator_type: 模拟器类型 (isaac_sim, gazebo, mujoco)
            policy_type: 策略类型 (vla, mpc, rl)
            device: 计算设备
        """
        self.robot_name = robot_name
        self.action_dim = action_dim
        self.use_ros2 = use_ros2
        self.use_simulator = use_simulator
        self.simulator_type = simulator_type
        self.device = device
        
        # 状态
        self.state = AgentState.IDLE
        self.current_perception: Optional[PerceptionData] = None
        self.current_plan: Optional[TaskPlan] = None
        
        # 初始化策略
        self.policy = self._init_policy(policy_type)
        
        # 初始化接口
        self.ros2_interface = None
        self.simulator = None
        
        if use_ros2:
            self._init_ros2()
        
        if use_simulator:
            self._init_simulator()
        
        logger.info(f"EmbodiedAgent '{robot_name}' initialized")
        logger.info(f"  - ROS2: {use_ros2}")
        logger.info(f"  - Simulator: {simulator_type if use_simulator else 'None'}")
        logger.info(f"  - Policy: {policy_type}")
        logger.info(f"  - Device: {device}")
    
    def _init_policy(self, policy_type: str) -> BasePolicy:
        """初始化策略"""
        if policy_type == "vla":
            return VLAPolicy(action_dim=self.action_dim, device=self.device)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    def _init_ros2(self):
        """初始化ROS2接口"""
        try:
            from ros2_interface import ROS2Interface
            self.ros2_interface = ROS2Interface(self.robot_name)
            logger.info("ROS2 interface initialized")
        except ImportError:
            logger.warning("ROS2 not available, running in standalone mode")
            self.use_ros2 = False
    
    def _init_simulator(self):
        """初始化模拟器连接"""
        try:
            from simulator_connector import SimulatorConnector
            self.simulator = SimulatorConnector(
                simulator_type=self.simulator_type,
                robot_name=self.robot_name
            )
            logger.info(f"{self.simulator_type} simulator connected")
        except ImportError as e:
            logger.warning(f"Simulator not available: {e}")
            self.use_simulator = False
    
    def perceive(self) -> PerceptionData:
        """
        获取感知数据
        
        Returns:
            PerceptionData: 感知数据
        """
        self.state = AgentState.PERCEIVING
        
        perception = PerceptionData()
        
        # 从模拟器获取感知
        if self.simulator is not None:
            sim_data = self.simulator.get_observation()
            perception.rgb_image = sim_data.get("rgb")
            perception.depth_image = sim_data.get("depth")
            perception.joint_positions = sim_data.get("joint_positions")
            perception.joint_velocities = sim_data.get("joint_velocities")
        
        # 从ROS2获取感知
        if self.ros2_interface is not None:
            ros_data = self.ros2_interface.get_observation()
            if perception.joint_positions is None:
                perception.joint_positions = ros_data.get("joint_states")
        
        self.current_perception = perception
        self.state = AgentState.IDLE
        
        return perception
    
    def plan(self, instruction: str) -> TaskPlan:
        """
        任务规划
        
        Args:
            instruction: 自然语言指令
            
        Returns:
            TaskPlan: 任务计划
        """
        self.state = AgentState.PLANNING
        
        # 简化的任务规划（实际应使用LLM或规划算法）
        plan = TaskPlan(
            task_description=instruction,
            subtasks=self._parse_instruction(instruction),
            current_step=0,
            is_complete=False
        )
        
        self.current_plan = plan
        self.state = AgentState.IDLE
        
        logger.info(f"Task plan created: {plan.subtasks}")
        return plan
    
    def _parse_instruction(self, instruction: str) -> List[str]:
        """解析指令为子任务列表（简化版）"""
        # 实际应用中应使用LLM进行任务分解
        instruction_lower = instruction.lower()
        
        if "pick" in instruction_lower or "grab" in instruction_lower:
            return ["approach_object", "open_gripper", "lower_arm", "close_gripper", "lift_object"]
        elif "place" in instruction_lower or "put" in instruction_lower:
            return ["move_to_target", "lower_object", "open_gripper", "retract_arm"]
        elif "move" in instruction_lower or "go" in instruction_lower:
            return ["plan_path", "execute_movement"]
        else:
            return ["execute_action"]
    
    def act(self, action: Optional[ActionCommand] = None) -> bool:
        """
        执行动作
        
        Args:
            action: 动作命令，如果为None则使用策略生成
            
        Returns:
            bool: 执行是否成功
        """
        self.state = AgentState.EXECUTING
        
        try:
            # 如果没有提供动作，使用策略生成
            if action is None:
                if self.current_perception is None:
                    self.perceive()
                action = self.policy.predict(self.current_perception)
            
            # 发送动作到执行器
            if self.simulator is not None:
                self.simulator.apply_action({
                    "joint_positions": action.joint_positions,
                    "gripper_state": action.gripper_state
                })
            
            if self.ros2_interface is not None:
                self.ros2_interface.send_action(action)
            
            self.state = AgentState.IDLE
            return True
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            self.state = AgentState.ERROR
            return False
    
    def step(self, instruction: Optional[str] = None) -> Tuple[PerceptionData, ActionCommand]:
        """
        执行一个完整的感知-规划-动作循环
        
        Args:
            instruction: 可选的指令，如果提供则重新规划
            
        Returns:
            Tuple[PerceptionData, ActionCommand]: 感知和动作
        """
        # 感知
        perception = self.perceive()
        
        # 规划（如果需要）
        if instruction is not None:
            self.plan(instruction)
        
        # 动作
        action = self.policy.predict(perception)
        self.act(action)
        
        return perception, action
    
    def run_task(self, instruction: str, max_steps: int = 100) -> bool:
        """
        执行完整任务
        
        Args:
            instruction: 任务指令
            max_steps: 最大执行步数
            
        Returns:
            bool: 任务是否成功完成
        """
        logger.info(f"Starting task: {instruction}")
        
        # 规划任务
        plan = self.plan(instruction)
        
        # 执行每个子任务
        for step in range(max_steps):
            if plan.current_step >= len(plan.subtasks):
                logger.info("Task completed successfully")
                return True
            
            current_subtask = plan.subtasks[plan.current_step]
            logger.info(f"Executing subtask {plan.current_step + 1}/{len(plan.subtasks)}: {current_subtask}")
            
            # 执行一步
            perception, action = self.step()
            
            # 检查是否完成（简化版，实际应使用成功检测）
            plan.current_step += 1
            
            # 模拟延迟
            import time
            time.sleep(0.1)
        
        logger.warning(f"Task timed out after {max_steps} steps")
        return False
    
    def reset(self):
        """重置智能体状态"""
        self.state = AgentState.IDLE
        self.current_perception = None
        self.current_plan = None
        self.policy.reset()
        
        if self.simulator is not None:
            self.simulator.reset()
        
        logger.info("Agent reset")
    
    def shutdown(self):
        """关闭智能体"""
        if self.ros2_interface is not None:
            self.ros2_interface.shutdown()
        
        if self.simulator is not None:
            self.simulator.close()
        
        logger.info("Agent shutdown")


# 辅助函数
def create_agent(
    robot_name: str = "default_robot",
    use_simulator: bool = True,
    simulator_type: str = "isaac_sim",
    **kwargs
) -> EmbodiedAgent:
    """
    创建具身智能体的工厂函数
    
    Args:
        robot_name: 机器人名称
        use_simulator: 是否使用模拟器
        simulator_type: 模拟器类型
        **kwargs: 其他参数
        
    Returns:
        EmbodiedAgent: 智能体实例
    """
    return EmbodiedAgent(
        robot_name=robot_name,
        use_simulator=use_simulator,
        simulator_type=simulator_type,
        **kwargs
    )


if __name__ == "__main__":
    # 测试代码
    logger.info("Testing EmbodiedAgent...")
    
    # 创建智能体（不使用ROS2和模拟器，仅测试核心逻辑）
    agent = EmbodiedAgent(
        robot_name="test_robot",
        use_ros2=False,
        use_simulator=False,
        policy_type="vla"
    )
    
    # 创建模拟的感知数据
    test_perception = PerceptionData(
        rgb_image=np.random.rand(224, 224, 3).astype(np.float32),
        joint_positions=np.random.rand(7).astype(np.float32),
        language_instruction="pick up the red cube"
    )
    
    # 测试策略推理
    action = agent.policy.predict(test_perception)
    logger.info(f"Predicted action shape: {action.joint_positions.shape if action.joint_positions is not None else None}")
    
    logger.info("Test completed successfully!")
