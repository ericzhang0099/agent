# Embodied AI Framework

具身智能（Embodied AI）研究与开发框架，提供完整的ROS2接口、模拟器连接和VLA（视觉-语言-动作）模型实现。

## 功能特性

- **核心智能体**: 具身智能体类，支持多模态感知和动作生成
- **ROS2接口**: 完整的ROS2封装，支持话题、服务、动作
- **模拟器连接**: 支持Isaac Sim、Gazebo、MuJoCo、PyBullet
- **VLA模型**: 视觉-语言-动作模型实现和推理
- **运动控制**: MPC、WBC、RL等多种控制方法
- **Sim-to-Real**: 域随机化、域适应等迁移技术

## 项目结构

```
embodied_ai/
├── embodied_agent.py       # 核心智能体类
├── ros2_interface.py       # ROS2接口封装
├── simulator_connector.py  # 模拟器连接
├── controllers/            # 控制器实现
├── perception/             # 感知模块
├── planning/               # 规划模块
├── sim2real/               # Sim-to-Real技术
├── examples/               # 示例代码
│   ├── basic_movement.py   # 基础运动控制
│   └── vla_inference.py    # VLA推理示例
├── config/                 # 配置文件
├── research_document.md    # 研究文档
└── requirements.txt        # 依赖列表
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础使用

```python
from embodied_agent import create_agent

# 创建智能体
agent = create_agent(
    robot_name="my_robot",
    use_simulator=True,
    simulator_type="isaac_sim"
)

# 执行任务
agent.run_task("pick up the red cube")
```

### 运行示例

```bash
# 基础运动控制示例
python examples/basic_movement.py

# VLA推理示例
python examples/vla_inference.py
```

## 研究内容

### 1. ROS2机器人操作系统架构
- 去中心化架构设计
- DDS通信中间件
- QoS策略配置
- 多机器人系统支持

### 2. 模拟器环境
- NVIDIA Isaac Sim
- Gazebo
- MuJoCo
- PyBullet

### 3. 视觉-动作-语言多模态融合
- VLA模型架构
- 多模态融合方法
- 动作表示与生成

### 4. 物理约束下的任务规划
- MPC模型预测控制
- WBC全身控制
- 强化学习方法

### 5. Sim-to-Real迁移技术
- 域随机化
- 域适应
- 系统辨识

## 参考文献

1. RT-2: Vision-Language-Action Models, Google DeepMind, 2023
2. OpenVLA: An Open-Source Vision-Language-Action Model, 2024
3. MPC and WBC for Humanoid Robots, OpenLoong, 2024
4. Sim-to-Real Reinforcement Learning for Robotics, UC Berkeley, 2024

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

- 项目主页: https://github.com/your-org/embodied-ai
- 问题反馈: https://github.com/your-org/embodied-ai/issues
