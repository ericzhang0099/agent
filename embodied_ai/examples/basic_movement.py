#!/usr/bin/env python3
"""
基础运动控制示例

演示如何使用具身智能框架进行基础运动控制，包括：
- 关节位置控制
- 末端执行器控制
- 基础导航
- 抓取动作

作者: Embodied AI Research Team
版本: 1.0.0
"""

import numpy as np
import sys
import time
import logging
from typing import List, Optional

# 添加父目录到路径
sys.path.insert(0, '..')

from embodied_agent import (
    EmbodiedAgent, 
    PerceptionData, 
    ActionCommand,
    create_agent
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicMovementController:
    """
    基础运动控制器
    
    提供基础的运动控制功能。
    """
    
    def __init__(self, agent: EmbodiedAgent):
        """
        初始化控制器
        
        Args:
            agent: 具身智能体实例
        """
        self.agent = agent
        self.joint_names = agent.get_joint_names() if hasattr(agent, 'get_joint_names') else None
        
        logger.info("BasicMovementController initialized")
    
    def move_to_joint_positions(
        self,
        target_positions: np.ndarray,
        duration: float = 2.0,
        steps: int = 50
    ) -> bool:
        """
        移动到目标关节位置
        
        Args:
            target_positions: 目标关节位置
            duration: 运动持续时间
            steps: 插值步数
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"Moving to joint positions: {target_positions}")
        
        # 获取当前位置
        perception = self.agent.perceive()
        current_positions = perception.joint_positions
        
        if current_positions is None:
            logger.warning("No joint position data available")
            current_positions = np.zeros_like(target_positions)
        
        # 轨迹插值
        for i in range(steps + 1):
            t = i / steps
            # 使用平滑插值 (cubic)
            t_smooth = 3 * t**2 - 2 * t**3
            
            interpolated = current_positions + t_smooth * (target_positions - current_positions)
            
            # 发送动作
            action = ActionCommand(
                joint_positions=interpolated,
                timestamp=time.time()
            )
            self.agent.act(action)
            
            # 等待
            time.sleep(duration / steps)
        
        logger.info("Movement completed")
        return True
    
    def move_end_effector_to_pose(
        self,
        target_pose: np.ndarray,
        duration: float = 2.0
    ) -> bool:
        """
        移动末端执行器到目标位姿（使用简单的逆运动学）
        
        Args:
            target_pose: 目标位姿 [x, y, z, roll, pitch, yaw]
            duration: 运动持续时间
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"Moving end effector to pose: {target_pose}")
        
        # 简化的逆运动学（实际应使用专业的IK求解器）
        # 这里使用简单的启发式方法
        target_joints = self._simple_ik(target_pose)
        
        return self.move_to_joint_positions(target_joints, duration)
    
    def _simple_ik(self, pose: np.ndarray) -> np.ndarray:
        """
        简化的逆运动学求解
        
        Args:
            pose: 目标位姿
            
        Returns:
            np.ndarray: 关节角度
        """
        # 这是一个简化的示例，实际应使用专业IK库如trac_ik或pybullet的IK
        x, y, z = pose[:3]
        
        # 简化的2D平面机械臂IK（假设前3个关节控制位置）
        L1 = 0.3  # 连杆1长度
        L2 = 0.3  # 连杆2长度
        
        # 计算关节角度
        cos_q2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)
        q2 = np.arccos(cos_q2)
        
        q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        
        # 返回7个关节的角度（其余关节设为0）
        joints = np.zeros(7)
        joints[0] = q1
        joints[1] = q2
        joints[2] = -q1 - q2  # 保持末端朝向
        
        return joints
    
    def execute_predefined_motion(self, motion_name: str) -> bool:
        """
        执行预定义动作
        
        Args:
            motion_name: 动作名称
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"Executing predefined motion: {motion_name}")
        
        # 预定义动作库
        motions = {
            "home": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "reach": np.array([0.5, -0.5, 0.5, -0.5, 0.0, 0.0, 0.0]),
            "grasp_ready": np.array([0.3, -0.3, 0.3, -0.3, 0.0, 0.0, 0.0]),
            "wave": np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0]),
        }
        
        if motion_name not in motions:
            logger.error(f"Unknown motion: {motion_name}")
            return False
        
        return self.move_to_joint_positions(motions[motion_name])
    
    def gripper_control(self, open_gripper: bool, force: float = 0.5) -> bool:
        """
        控制夹爪
        
        Args:
            open_gripper: 是否打开夹爪
            force: 夹持力
            
        Returns:
            bool: 是否成功
        """
        gripper_state = 1.0 if open_gripper else 0.0
        logger.info(f"Setting gripper to: {'open' if open_gripper else 'close'}")
        
        action = ActionCommand(
            gripper_state=gripper_state,
            timestamp=time.time()
        )
        
        return self.agent.act(action)
    
    def perform_pick_and_place(
        self,
        pick_pose: np.ndarray,
        place_pose: np.ndarray
    ) -> bool:
        """
        执行抓取放置任务
        
        Args:
            pick_pose: 抓取位姿
            place_pose: 放置位姿
            
        Returns:
            bool: 是否成功
        """
        logger.info("Starting pick and place task")
        
        try:
            # 1. 移动到预抓取位置
            pre_pick = pick_pose.copy()
            pre_pick[2] += 0.1  # 在抓取位置上方10cm
            self.move_end_effector_to_pose(pre_pick, duration=1.5)
            
            # 2. 打开夹爪
            self.gripper_control(open_gripper=True)
            time.sleep(0.5)
            
            # 3. 下降到抓取位置
            self.move_end_effector_to_pose(pick_pose, duration=1.0)
            
            # 4. 关闭夹爪
            self.gripper_control(open_gripper=False)
            time.sleep(0.5)
            
            # 5. 抬起
            self.move_end_effector_to_pose(pre_pick, duration=1.0)
            
            # 6. 移动到预放置位置
            pre_place = place_pose.copy()
            pre_place[2] += 0.1
            self.move_end_effector_to_pose(pre_place, duration=1.5)
            
            # 7. 下降到放置位置
            self.move_end_effector_to_pose(place_pose, duration=1.0)
            
            # 8. 打开夹爪
            self.gripper_control(open_gripper=True)
            time.sleep(0.5)
            
            # 9. 抬起
            self.move_end_effector_to_pose(pre_place, duration=1.0)
            
            # 10. 回到home位置
            self.execute_predefined_motion("home")
            
            logger.info("Pick and place completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pick and place failed: {e}")
            return False
    
    def navigate_to(self, target_position: np.ndarray, tolerance: float = 0.1) -> bool:
        """
        导航到目标位置（适用于移动底盘）
        
        Args:
            target_position: 目标位置 [x, y]
            tolerance: 到达容差
            
        Returns:
            bool: 是否成功
        """
        logger.info(f"Navigating to: {target_position}")
        
        # 简化的导航控制
        max_steps = 100
        step = 0
        
        while step < max_steps:
            # 获取当前位置
            perception = self.agent.perceive()
            
            # 计算距离
            # 注意：实际应从perception中获取机器人位置
            distance = np.linalg.norm(target_position)
            
            if distance < tolerance:
                logger.info("Navigation target reached")
                return True
            
            # 计算速度命令（简单的P控制）
            kp = 0.5
            velocity = kp * target_position / (distance + 0.01)
            
            # 发送速度命令
            action = ActionCommand(
                joint_velocities=np.array([velocity[0], velocity[1], 0.0]),
                timestamp=time.time()
            )
            self.agent.act(action)
            
            time.sleep(0.1)
            step += 1
        
        logger.warning("Navigation timeout")
        return False


def demo_joint_control():
    """演示关节控制"""
    logger.info("=" * 50)
    logger.info("Demo: Joint Position Control")
    logger.info("=" * 50)
    
    # 创建智能体
    agent = create_agent(
        robot_name="demo_robot",
        use_simulator=False,  # 使用mock模式
        policy_type="vla"
    )
    
    # 创建控制器
    controller = BasicMovementController(agent)
    
    # 执行预定义动作
    controller.execute_predefined_motion("home")
    time.sleep(1.0)
    
    controller.execute_predefined_motion("reach")
    time.sleep(1.0)
    
    controller.execute_predefined_motion("home")
    
    logger.info("Joint control demo completed")


def demo_end_effector_control():
    """演示末端执行器控制"""
    logger.info("=" * 50)
    logger.info("Demo: End Effector Control")
    logger.info("=" * 50)
    
    # 创建智能体
    agent = create_agent(
        robot_name="demo_robot",
        use_simulator=False,
        policy_type="vla"
    )
    
    # 创建控制器
    controller = BasicMovementController(agent)
    
    # 定义目标位姿
    target_poses = [
        np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0]),
        np.array([0.4, 0.2, 0.3, 0.0, 0.0, 0.0]),
        np.array([0.4, -0.2, 0.3, 0.0, 0.0, 0.0]),
        np.array([0.3, 0.0, 0.5, 0.0, 0.0, 0.0]),
    ]
    
    for i, pose in enumerate(target_poses):
        logger.info(f"Moving to pose {i + 1}/{len(target_poses)}")
        controller.move_end_effector_to_pose(pose, duration=1.5)
        time.sleep(0.5)
    
    # 回到home
    controller.execute_predefined_motion("home")
    
    logger.info("End effector control demo completed")


def demo_pick_and_place():
    """演示抓取放置"""
    logger.info("=" * 50)
    logger.info("Demo: Pick and Place")
    logger.info("=" * 50)
    
    # 创建智能体
    agent = create_agent(
        robot_name="demo_robot",
        use_simulator=False,
        policy_type="vla"
    )
    
    # 创建控制器
    controller = BasicMovementController(agent)
    
    # 定义抓取和放置位姿
    pick_pose = np.array([0.4, 0.1, 0.1, 0.0, 0.0, 0.0])
    place_pose = np.array([0.4, -0.1, 0.1, 0.0, 0.0, 0.0])
    
    # 执行抓取放置
    controller.perform_pick_and_place(pick_pose, place_pose)
    
    logger.info("Pick and place demo completed")


def demo_vla_control():
    """演示VLA策略控制"""
    logger.info("=" * 50)
    logger.info("Demo: VLA Policy Control")
    logger.info("=" * 50)
    
    # 创建智能体
    agent = create_agent(
        robot_name="demo_robot",
        use_simulator=False,
        policy_type="vla"
    )
    
    # 执行任务
    instructions = [
        "move to home position",
        "reach forward",
        "prepare for grasping",
    ]
    
    for instruction in instructions:
        logger.info(f"Executing instruction: {instruction}")
        
        # 创建感知数据
        perception = PerceptionData(
            rgb_image=np.random.rand(224, 224, 3).astype(np.float32),
            joint_positions=np.random.rand(7).astype(np.float32),
            language_instruction=instruction,
            timestamp=time.time()
        )
        
        # 使用VLA策略生成动作
        action = agent.policy.predict(perception)
        logger.info(f"Generated action: {action.joint_positions}")
        
        # 执行动作
        agent.act(action)
        time.sleep(1.0)
    
    logger.info("VLA control demo completed")


def run_all_demos():
    """运行所有演示"""
    logger.info("\n" + "=" * 50)
    logger.info("Embodied AI Basic Movement Examples")
    logger.info("=" * 50 + "\n")
    
    try:
        demo_joint_control()
        print()
        
        demo_end_effector_control()
        print()
        
        demo_pick_and_place()
        print()
        
        demo_vla_control()
        print()
        
        logger.info("=" * 50)
        logger.info("All demos completed successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Basic Movement Examples")
    parser.add_argument(
        "--demo",
        type=str,
        choices=["joint", "ee", "pickplace", "vla", "all"],
        default="all",
        help="Which demo to run"
    )
    
    args = parser.parse_args()
    
    if args.demo == "joint":
        demo_joint_control()
    elif args.demo == "ee":
        demo_end_effector_control()
    elif args.demo == "pickplace":
        demo_pick_and_place()
    elif args.demo == "vla":
        demo_vla_control()
    else:
        run_all_demos()
