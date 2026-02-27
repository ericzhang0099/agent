#!/usr/bin/env python3
"""
ROS2接口封装模块

提供与ROS2机器人操作系统的接口，包括：
- 节点管理
- 话题订阅与发布
- 服务调用
- 动作客户端
- 参数管理

作者: Embodied AI Research Team
版本: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging
import threading
import queue

logger = logging.getLogger(__name__)

# 尝试导入ROS2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, JointState, CameraInfo
    from geometry_msgs.msg import Pose, Twist, PoseStamped
    from std_msgs.msg import Float64MultiArray, String
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger.warning("ROS2 not available. Running in mock mode.")


@dataclass
class ROS2Config:
    """ROS2配置"""
    node_name: str = "embodied_agent_node"
    namespace: str = ""
    robot_description_topic: str = "/robot_description"
    joint_states_topic: str = "/joint_states"
    cmd_vel_topic: str = "/cmd_vel"
    image_topic: str = "/camera/image_raw"
    depth_topic: str = "/camera/depth/image_raw"
    action_server: str = "/joint_trajectory_controller/follow_joint_trajectory"
    qos_profile: str = "default"


class ROS2Interface:
    """
    ROS2接口类
    
    封装ROS2功能，提供简化的API与机器人交互。
    """
    
    def __init__(self, robot_name: str, config: Optional[ROS2Config] = None):
        """
        初始化ROS2接口
        
        Args:
            robot_name: 机器人名称
            config: ROS2配置
        """
        if not ROS2_AVAILABLE:
            logger.warning("ROS2 is not available. Interface will run in mock mode.")
            self.mock_mode = True
        else:
            self.mock_mode = False
        
        self.robot_name = robot_name
        self.config = config or ROS2Config(node_name=f"{robot_name}_node")
        
        # ROS2节点
        self.node: Optional[Node] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False
        
        # 发布者和订阅者
        self._publishers: Dict[str, Any] = {}
        self._subscribers: Dict[str, Any] = {}
        
        # 数据缓存
        self._joint_states: Optional[JointState] = None
        self._rgb_image: Optional[np.ndarray] = None
        self._depth_image: Optional[np.ndarray] = None
        self._robot_pose: Optional[Pose] = None
        
        # 回调函数
        self._joint_state_callbacks: List[Callable] = []
        self._image_callbacks: List[Callable] = []
        
        # 动作客户端
        self._action_client: Optional[ActionClient] = None
        
        if not self.mock_mode:
            self._init_ros2()
        else:
            self._init_mock()
    
    def _init_ros2(self):
        """初始化ROS2节点"""
        if not ROS2_AVAILABLE:
            return
        
        # 检查ROS2上下文
        if not rclpy.ok():
            rclpy.init()
        
        # 创建节点
        self.node = Node(self.config.node_name, namespace=self.config.namespace)
        
        # 创建QoS配置
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # 创建订阅者
        self._subscribers['joint_states'] = self.node.create_subscription(
            JointState,
            self.config.joint_states_topic,
            self._joint_states_callback,
            qos
        )
        
        self._subscribers['rgb_image'] = self.node.create_subscription(
            Image,
            self.config.image_topic,
            self._rgb_image_callback,
            qos
        )
        
        self._subscribers['depth_image'] = self.node.create_subscription(
            Image,
            self.config.depth_topic,
            self._depth_image_callback,
            qos
        )
        
        # 创建发布者
        self._publishers['cmd_vel'] = self.node.create_publisher(
            Twist,
            self.config.cmd_vel_topic,
            qos
        )
        
        self._publishers['joint_command'] = self.node.create_publisher(
            Float64MultiArray,
            f"/{self.robot_name}/joint_commands",
            qos
        )
        
        self._publishers['trajectory'] = self.node.create_publisher(
            JointTrajectory,
            f"/{self.robot_name}/joint_trajectory",
            qos
        )
        
        # 创建动作客户端
        self._action_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            self.config.action_server
        )
        
        # 启动spin线程
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin)
        self._spin_thread.start()
        
        logger.info(f"ROS2 node '{self.config.node_name}' initialized")
    
    def _init_mock(self):
        """初始化模拟模式"""
        self._mock_joint_states = np.zeros(7)
        self._mock_joint_velocities = np.zeros(7)
        logger.info("ROS2 mock mode initialized")
    
    def _spin(self):
        """ROS2 spin循环"""
        while self._running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
    
    def _joint_states_callback(self, msg: JointState):
        """关节状态回调"""
        self._joint_states = msg
        for callback in self._joint_state_callbacks:
            callback(msg)
    
    def _rgb_image_callback(self, msg: Image):
        """RGB图像回调"""
        self._rgb_image = self._image_msg_to_numpy(msg)
        for callback in self._image_callbacks:
            callback(self._rgb_image)
    
    def _depth_image_callback(self, msg: Image):
        """深度图像回调"""
        self._depth_image = self._image_msg_to_numpy(msg)
    
    def _image_msg_to_numpy(self, msg: Image) -> np.ndarray:
        """将ROS图像消息转换为numpy数组"""
        if msg.encoding == "rgb8":
            dtype = np.uint8
            channels = 3
        elif msg.encoding == "bgr8":
            dtype = np.uint8
            channels = 3
        elif msg.encoding == "32FC1":
            dtype = np.float32
            channels = 1
        elif msg.encoding == "16UC1":
            dtype = np.uint16
            channels = 1
        else:
            raise ValueError(f"Unsupported encoding: {msg.encoding}")
        
        image = np.frombuffer(msg.data, dtype=dtype)
        if channels > 1:
            image = image.reshape((msg.height, msg.width, channels))
        else:
            image = image.reshape((msg.height, msg.width))
        
        return image
    
    def get_observation(self) -> Dict[str, Any]:
        """
        获取当前观测数据
        
        Returns:
            Dict: 包含关节状态、图像等数据的字典
        """
        if self.mock_mode:
            return {
                "joint_states": self._mock_joint_states.copy(),
                "joint_velocities": self._mock_joint_velocities.copy(),
                "rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "depth": np.random.rand(480, 640).astype(np.float32)
            }
        
        observation = {
            "joint_states": None,
            "joint_velocities": None,
            "joint_names": None,
            "rgb": self._rgb_image,
            "depth": self._depth_image,
            "pose": self._robot_pose
        }
        
        if self._joint_states is not None:
            observation["joint_states"] = np.array(self._joint_states.position)
            observation["joint_velocities"] = np.array(self._joint_states.velocity)
            observation["joint_names"] = list(self._joint_states.name)
        
        return observation
    
    def send_action(self, action: Any):
        """
        发送动作命令
        
        Args:
            action: 动作命令
        """
        if self.mock_mode:
            logger.debug(f"Mock action: {action}")
            return
        
        # 根据动作类型发送
        if hasattr(action, 'joint_positions') and action.joint_positions is not None:
            self._send_joint_positions(action.joint_positions)
        
        if hasattr(action, 'joint_velocities') and action.joint_velocities is not None:
            self._send_joint_velocities(action.joint_velocities)
        
        if hasattr(action, 'joint_torques') and action.joint_torques is not None:
            self._send_joint_torques(action.joint_torques)
        
        if hasattr(action, 'gripper_state') and action.gripper_state is not None:
            self._send_gripper_command(action.gripper_state)
    
    def _send_joint_positions(self, positions: np.ndarray):
        """发送关节位置命令"""
        if self.mock_mode:
            self._mock_joint_states = positions.copy()
            return
        
        msg = Float64MultiArray()
        msg.data = positions.tolist()
        self._publishers['joint_command'].publish(msg)
    
    def _send_joint_velocities(self, velocities: np.ndarray):
        """发送关节速度命令"""
        if self.mock_mode:
            self._mock_joint_velocities = velocities.copy()
            return
        
        # 实现速度控制
        pass
    
    def _send_joint_torques(self, torques: np.ndarray):
        """发送关节力矩命令"""
        if self.mock_mode:
            return
        
        # 实现力矩控制
        pass
    
    def _send_gripper_command(self, state: float):
        """发送夹爪命令"""
        if self.mock_mode:
            return
        
        # 实现夹爪控制
        pass
    
    def send_trajectory(
        self,
        joint_names: List[str],
        positions: List[List[float]],
        times: List[float]
    ):
        """
        发送关节轨迹
        
        Args:
            joint_names: 关节名称列表
            positions: 位置序列
            times: 时间序列
        """
        if self.mock_mode:
            logger.info(f"Mock trajectory for joints: {joint_names}")
            return
        
        msg = JointTrajectory()
        msg.joint_names = joint_names
        
        for pos, t in zip(positions, times):
            point = JointTrajectoryPoint()
            point.positions = pos
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1) * 1e9)
            msg.points.append(point)
        
        self._publishers['trajectory'].publish(msg)
    
    def send_velocity_command(self, linear: List[float], angular: List[float]):
        """
        发送速度命令（用于移动底盘）
        
        Args:
            linear: 线速度 [x, y, z]
            angular: 角速度 [x, y, z]
        """
        if self.mock_mode:
            logger.debug(f"Mock velocity: linear={linear}, angular={angular}")
            return
        
        msg = Twist()
        msg.linear.x = linear[0]
        msg.linear.y = linear[1]
        msg.linear.z = linear[2]
        msg.angular.x = angular[0]
        msg.angular.y = angular[1]
        msg.angular.z = angular[2]
        
        self._publishers['cmd_vel'].publish(msg)
    
    def register_joint_state_callback(self, callback: Callable):
        """注册关节状态回调"""
        self._joint_state_callbacks.append(callback)
    
    def register_image_callback(self, callback: Callable):
        """注册图像回调"""
        self._image_callbacks.append(callback)
    
    def get_joint_names(self) -> Optional[List[str]]:
        """获取关节名称列表"""
        if self.mock_mode:
            return [f"joint_{i}" for i in range(7)]
        
        if self._joint_states is not None:
            return list(self._joint_states.name)
        return None
    
    def get_joint_limits(self) -> Dict[str, Dict[str, float]]:
        """
        获取关节限制
        
        Returns:
            Dict: 关节限制字典
        """
        # 实际应从URDF或参数服务器获取
        return {
            "joint_0": {"min": -3.14, "max": 3.14},
            "joint_1": {"min": -1.57, "max": 1.57},
            "joint_2": {"min": -3.14, "max": 3.14},
            "joint_3": {"min": -1.57, "max": 1.57},
            "joint_4": {"min": -3.14, "max": 3.14},
            "joint_5": {"min": -1.57, "max": 1.57},
            "joint_6": {"min": -3.14, "max": 3.14},
        }
    
    def shutdown(self):
        """关闭ROS2接口"""
        self._running = False
        
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
        
        if self.node is not None:
            self.node.destroy_node()
        
        logger.info("ROS2 interface shutdown")


# 辅助函数
def create_ros2_interface(
    robot_name: str,
    config: Optional[ROS2Config] = None
) -> ROS2Interface:
    """
    创建ROS2接口的工厂函数
    
    Args:
        robot_name: 机器人名称
        config: ROS2配置
        
    Returns:
        ROS2Interface: ROS2接口实例
    """
    return ROS2Interface(robot_name, config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Testing ROS2Interface...")
    
    # 创建接口（模拟模式）
    interface = ROS2Interface("test_robot")
    
    # 测试获取观测
    obs = interface.get_observation()
    logger.info(f"Observation keys: {obs.keys()}")
    
    # 测试发送动作
    class MockAction:
        def __init__(self):
            self.joint_positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            self.gripper_state = 0.5
    
    interface.send_action(MockAction())
    
    logger.info("Test completed!")
