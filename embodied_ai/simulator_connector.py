#!/usr/bin/env python3
"""
模拟器连接模块

提供与主流机器人模拟器的连接接口，支持：
- NVIDIA Isaac Sim
- Gazebo
- MuJoCo
- PyBullet

作者: Embodied AI Research Team
版本: 1.0.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SimulatorType(Enum):
    """模拟器类型枚举"""
    ISAAC_SIM = "isaac_sim"
    GAZEBO = "gazebo"
    MUJOCO = "mujoco"
    PYBULLET = "pybullet"
    MOCK = "mock"


@dataclass
class SimulatorConfig:
    """模拟器配置"""
    headless: bool = False
    render: bool = True
    physics_dt: float = 1.0 / 60.0
    rendering_dt: float = 1.0 / 60.0
    device: str = "cuda"
    
    # Isaac Sim特定配置
    isaac_sim_path: Optional[str] = None
    
    # Gazebo特定配置
    gazebo_world: str = "empty.world"
    
    # MuJoCo特定配置
    mujoco_model_path: Optional[str] = None
    
    # PyBullet特定配置
    pybullet_mode: str = "gui"  # 或 "direct"


class BaseSimulator(ABC):
    """模拟器基类"""
    
    @abstractmethod
    def reset(self):
        """重置环境"""
        pass
    
    @abstractmethod
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行一步仿真"""
        pass
    
    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """获取观测"""
        pass
    
    @abstractmethod
    def close(self):
        """关闭模拟器"""
        pass


class MockSimulator(BaseSimulator):
    """
    模拟模拟器（用于测试）
    
    当真实模拟器不可用时提供模拟环境。
    """
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.step_count = 0
        self.joint_positions = np.zeros(7)
        self.joint_velocities = np.zeros(7)
        self.rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.depth_image = np.random.rand(480, 640).astype(np.float32)
        
        logger.info("MockSimulator initialized")
    
    def reset(self):
        """重置环境"""
        self.step_count = 0
        self.joint_positions = np.zeros(7)
        self.joint_velocities = np.zeros(7)
        logger.info("MockSimulator reset")
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行一步仿真"""
        self.step_count += 1
        
        # 模拟关节运动
        if "joint_positions" in action:
            target = np.array(action["joint_positions"])
            # 简单的PD控制模拟
            error = target - self.joint_positions
            self.joint_velocities = error * 10.0  # P gain
            self.joint_positions += self.joint_velocities * self.config.physics_dt
        
        # 模拟图像变化
        self.rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        return self.get_observation()
    
    def get_observation(self) -> Dict[str, Any]:
        """获取观测"""
        return {
            "rgb": self.rgb_image.copy(),
            "depth": self.depth_image.copy(),
            "joint_positions": self.joint_positions.copy(),
            "joint_velocities": self.joint_velocities.copy(),
            "step": self.step_count
        }
    
    def close(self):
        """关闭模拟器"""
        logger.info("MockSimulator closed")


class IsaacSimConnector(BaseSimulator):
    """
    NVIDIA Isaac Sim连接器
    
    提供与Isaac Sim的接口，支持高保真物理仿真和渲染。
    """
    
    def __init__(self, robot_name: str, config: SimulatorConfig):
        self.robot_name = robot_name
        self.config = config
        self._simulation_app = None
        self._world = None
        self._robot = None
        self._camera = None
        
        self._init_isaac_sim()
    
    def _init_isaac_sim(self):
        """初始化Isaac Sim"""
        try:
            # 尝试导入Isaac Sim模块
            from omni.isaac.kit import SimulationApp
            from omni.isaac.core import World
            from omni.isaac.core.robots import Robot
            from omni.isaac.sensor import Camera
            
            # 启动SimulationApp
            self._simulation_app = SimulationApp({
                "headless": self.config.headless,
                "width": 1280,
                "height": 720
            })
            
            # 创建World
            self._world = World(
                physics_dt=self.config.physics_dt,
                rendering_dt=self.config.rendering_dt,
                stage_units_in_meters=1.0
            )
            
            logger.info("Isaac Sim initialized successfully")
            
        except ImportError:
            logger.warning("Isaac Sim not available, falling back to mock")
            raise RuntimeError("Isaac Sim not installed")
    
    def load_robot(self, robot_usd_path: str, position: Optional[List[float]] = None):
        """
        加载机器人
        
        Args:
            robot_usd_path: 机器人USD文件路径
            position: 初始位置
        """
        if position is None:
            position = [0, 0, 0]
        
        try:
            from omni.isaac.core.robots import Robot
            
            self._robot = Robot(
                prim_path=f"/World/{self.robot_name}",
                name=self.robot_name,
                usd_path=robot_usd_path,
                position=position
            )
            
            self._world.scene.add(self._robot)
            logger.info(f"Robot '{self.robot_name}' loaded from {robot_usd_path}")
            
        except Exception as e:
            logger.error(f"Failed to load robot: {e}")
            raise
    
    def add_camera(
        self,
        camera_name: str = "camera",
        position: List[float] = [1.0, 0.0, 0.5],
        orientation: List[float] = [0.0, 0.0, 0.0],
        resolution: Tuple[int, int] = (640, 480)
    ):
        """
        添加相机
        
        Args:
            camera_name: 相机名称
            position: 相机位置
            orientation: 相机朝向
            resolution: 分辨率
        """
        try:
            from omni.isaac.sensor import Camera
            
            self._camera = Camera(
                prim_path=f"/World/{self.robot_name}/{camera_name}",
                position=position,
                orientation=orientation,
                resolution=resolution
            )
            
            self._world.scene.add(self._camera)
            logger.info(f"Camera '{camera_name}' added")
            
        except Exception as e:
            logger.error(f"Failed to add camera: {e}")
            raise
    
    def reset(self):
        """重置环境"""
        if self._world is not None:
            self._world.reset()
            logger.info("Isaac Sim world reset")
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行一步仿真"""
        # 应用动作
        if self._robot is not None and "joint_positions" in action:
            self._robot.set_joint_positions(action["joint_positions"])
        
        # 步进仿真
        if self._world is not None:
            self._world.step(render=self.config.render)
        
        return self.get_observation()
    
    def get_observation(self) -> Dict[str, Any]:
        """获取观测"""
        observation = {
            "rgb": None,
            "depth": None,
            "joint_positions": None,
            "joint_velocities": None
        }
        
        # 获取相机图像
        if self._camera is not None:
            observation["rgb"] = self._camera.get_rgba()
            observation["depth"] = self._camera.get_depth()
        
        # 获取关节状态
        if self._robot is not None:
            observation["joint_positions"] = self._robot.get_joint_positions()
            observation["joint_velocities"] = self._robot.get_joint_velocities()
        
        return observation
    
    def close(self):
        """关闭模拟器"""
        if self._simulation_app is not None:
            self._simulation_app.close()
            logger.info("Isaac Sim closed")


class GazeboConnector(BaseSimulator):
    """
    Gazebo模拟器连接器
    
    通过ROS2与Gazebo通信。
    """
    
    def __init__(self, robot_name: str, config: SimulatorConfig):
        self.robot_name = robot_name
        self.config = config
        
        # 使用ROS2接口与Gazebo通信
        try:
            from ros2_interface import ROS2Interface
            self._ros2_interface = ROS2Interface(robot_name)
            logger.info("Gazebo connector initialized via ROS2")
        except ImportError:
            logger.warning("ROS2 not available, using mock")
            self._ros2_interface = None
    
    def reset(self):
        """重置环境"""
        # 通过ROS2发送重置命令
        logger.info("Gazebo reset requested")
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行一步"""
        # 通过ROS2发送动作
        if self._ros2_interface is not None:
            class ActionWrapper:
                pass
            wrapper = ActionWrapper()
            wrapper.joint_positions = action.get("joint_positions")
            wrapper.gripper_state = action.get("gripper_state")
            self._ros2_interface.send_action(wrapper)
        
        return self.get_observation()
    
    def get_observation(self) -> Dict[str, Any]:
        """获取观测"""
        if self._ros2_interface is not None:
            return self._ros2_interface.get_observation()
        
        # 返回模拟数据
        return {
            "rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "depth": np.random.rand(480, 640).astype(np.float32),
            "joint_positions": np.zeros(7),
            "joint_velocities": np.zeros(7)
        }
    
    def close(self):
        """关闭连接器"""
        if self._ros2_interface is not None:
            self._ros2_interface.shutdown()
        logger.info("Gazebo connector closed")


class MuJoCoConnector(BaseSimulator):
    """
    MuJoCo模拟器连接器
    """
    
    def __init__(self, robot_name: str, config: SimulatorConfig):
        self.robot_name = robot_name
        self.config = config
        self._model = None
        self._data = None
        self._viewer = None
        
        self._init_mujoco()
    
    def _init_mujoco(self):
        """初始化MuJoCo"""
        try:
            import mujoco
            import mujoco.viewer
            
            # 加载模型
            if self.config.mujoco_model_path:
                self._model = mujoco.MjModel.from_xml_path(self.config.mujoco_model_path)
            else:
                # 使用默认模型
                self._model = mujoco.MjModel.from_xml_string(self._get_default_model())
            
            self._data = mujoco.MjData(self._model)
            
            logger.info("MuJoCo initialized")
            
        except ImportError:
            logger.error("MuJoCo not installed")
            raise
    
    def _get_default_model(self) -> str:
        """获取默认模型XML"""
        return """
        <mujoco>
          <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
            <body pos="0 0 1" name="robot">
              <freejoint/>
              <geom type="box" size=".1 .1 .1" rgba="0 .9 0 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    
    def reset(self):
        """重置环境"""
        if self._data is not None:
            mujoco.mj_resetData(self._model, self._data)
            logger.info("MuJoCo reset")
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行一步仿真"""
        if self._model is not None and self._data is not None:
            # 应用控制
            if "joint_positions" in action:
                self._data.ctrl[:] = action["joint_positions"]
            
            # 步进
            mujoco.mj_step(self._model, self._data)
        
        return self.get_observation()
    
    def get_observation(self) -> Dict[str, Any]:
        """获取观测"""
        observation = {
            "rgb": None,
            "depth": None,
            "joint_positions": None,
            "joint_velocities": None
        }
        
        if self._data is not None:
            observation["joint_positions"] = self._data.qpos.copy()
            observation["joint_velocities"] = self._data.qvel.copy()
        
        return observation
    
    def close(self):
        """关闭模拟器"""
        logger.info("MuJoCo closed")


class PyBulletConnector(BaseSimulator):
    """
    PyBullet模拟器连接器
    """
    
    def __init__(self, robot_name: str, config: SimulatorConfig):
        self.robot_name = robot_name
        self.config = config
        self._physics_client = None
        self._robot_id = None
        
        self._init_pybullet()
    
    def _init_pybullet(self):
        """初始化PyBullet"""
        try:
            import pybullet as p
            import pybullet_data
            
            # 连接服务器
            if self.config.pybullet_mode == "gui":
                self._physics_client = p.connect(p.GUI)
            else:
                self._physics_client = p.connect(p.DIRECT)
            
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            
            # 加载地面
            p.loadURDF("plane.urdf")
            
            logger.info("PyBullet initialized")
            
        except ImportError:
            logger.error("PyBullet not installed")
            raise
    
    def load_robot(self, urdf_path: str, base_position: List[float] = [0, 0, 0.5]):
        """
        加载机器人URDF
        
        Args:
            urdf_path: URDF文件路径
            base_position: 初始位置
        """
        try:
            import pybullet as p
            
            self._robot_id = p.loadURDF(
                urdf_path,
                basePosition=base_position,
                useFixedBase=False
            )
            
            logger.info(f"Robot loaded from {urdf_path}")
            
        except Exception as e:
            logger.error(f"Failed to load robot: {e}")
            raise
    
    def reset(self):
        """重置环境"""
        try:
            import pybullet as p
            
            if self._robot_id is not None:
                p.resetBasePositionAndOrientation(
                    self._robot_id,
                    [0, 0, 0.5],
                    [0, 0, 0, 1]
                )
            
            logger.info("PyBullet reset")
            
        except Exception as e:
            logger.error(f"Reset failed: {e}")
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行一步仿真"""
        try:
            import pybullet as p
            
            # 应用关节控制
            if self._robot_id is not None and "joint_positions" in action:
                positions = action["joint_positions"]
                for i, pos in enumerate(positions):
                    p.setJointMotorControl2(
                        self._robot_id,
                        i,
                        p.POSITION_CONTROL,
                        targetPosition=pos
                    )
            
            # 步进
            p.stepSimulation()
            
        except Exception as e:
            logger.error(f"Step failed: {e}")
        
        return self.get_observation()
    
    def get_observation(self) -> Dict[str, Any]:
        """获取观测"""
        observation = {
            "rgb": None,
            "depth": None,
            "joint_positions": None,
            "joint_velocities": None
        }
        
        try:
            import pybullet as p
            
            if self._robot_id is not None:
                # 获取关节状态
                joint_states = []
                joint_velocities = []
                
                for i in range(p.getNumJoints(self._robot_id)):
                    state = p.getJointState(self._robot_id, i)
                    joint_states.append(state[0])
                    joint_velocities.append(state[1])
                
                observation["joint_positions"] = np.array(joint_states)
                observation["joint_velocities"] = np.array(joint_velocities)
                
                # 获取相机图像（如果配置了相机）
                # ...
                
        except Exception as e:
            logger.error(f"Get observation failed: {e}")
        
        return observation
    
    def close(self):
        """关闭模拟器"""
        try:
            import pybullet as p
            
            if self._physics_client is not None:
                p.disconnect()
            
            logger.info("PyBullet closed")
            
        except Exception as e:
            logger.error(f"Close failed: {e}")


class SimulatorConnector:
    """
    模拟器连接器工厂类
    
    统一接口，根据配置自动选择合适的模拟器。
    """
    
    def __init__(
        self,
        simulator_type: str,
        robot_name: str,
        config: Optional[SimulatorConfig] = None
    ):
        """
        初始化连接器
        
        Args:
            simulator_type: 模拟器类型
            robot_name: 机器人名称
            config: 模拟器配置
        """
        self.simulator_type = simulator_type
        self.robot_name = robot_name
        self.config = config or SimulatorConfig()
        
        # 创建具体的模拟器连接
        self._simulator = self._create_simulator()
    
    def _create_simulator(self) -> BaseSimulator:
        """创建模拟器实例"""
        sim_type = SimulatorType(self.simulator_type)
        
        if sim_type == SimulatorType.ISAAC_SIM:
            try:
                return IsaacSimConnector(self.robot_name, self.config)
            except Exception as e:
                logger.warning(f"Isaac Sim not available: {e}, using mock")
                return MockSimulator(self.config)
        
        elif sim_type == SimulatorType.GAZEBO:
            return GazeboConnector(self.robot_name, self.config)
        
        elif sim_type == SimulatorType.MUJOCO:
            try:
                return MuJoCoConnector(self.robot_name, self.config)
            except Exception as e:
                logger.warning(f"MuJoCo not available: {e}, using mock")
                return MockSimulator(self.config)
        
        elif sim_type == SimulatorType.PYBULLET:
            try:
                return PyBulletConnector(self.robot_name, self.config)
            except Exception as e:
                logger.warning(f"PyBullet not available: {e}, using mock")
                return MockSimulator(self.config)
        
        else:
            return MockSimulator(self.config)
    
    def reset(self):
        """重置环境"""
        self._simulator.reset()
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行一步"""
        return self._simulator.step(action)
    
    def get_observation(self) -> Dict[str, Any]:
        """获取观测"""
        return self._simulator.get_observation()
    
    def apply_action(self, action: Dict[str, Any]):
        """应用动作"""
        self._simulator.step(action)
    
    def close(self):
        """关闭模拟器"""
        self._simulator.close()


# 辅助函数
def create_simulator(
    simulator_type: str,
    robot_name: str,
    **kwargs
) -> SimulatorConnector:
    """
    创建模拟器连接器的工厂函数
    
    Args:
        simulator_type: 模拟器类型
        robot_name: 机器人名称
        **kwargs: 其他配置参数
        
    Returns:
        SimulatorConnector: 模拟器连接器实例
    """
    config = SimulatorConfig(**kwargs)
    return SimulatorConnector(simulator_type, robot_name, config)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Testing SimulatorConnector...")
    
    # 创建模拟器连接（使用mock）
    connector = SimulatorConnector("mock", "test_robot")
    
    # 测试获取观测
    obs = connector.get_observation()
    logger.info(f"Observation keys: {obs.keys()}")
    
    # 测试执行动作
    action = {"joint_positions": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])}
    obs = connector.step(action)
    logger.info(f"Joint positions after step: {obs['joint_positions']}")
    
    connector.close()
    
    logger.info("Test completed!")
