# Constitutional AI - 宪法AI系统

## 概述
Constitutional AI 是一个基于宪法原则的自我批评和修正系统，确保AI行为符合预定义的核心价值观和行为准则。该系统实现了Anthropic Constitutional AI方法论，通过分层批评引擎和自动修订流程来提升AI输出的质量和安全性。

## 功能特性
- ⚖️ **20条宪法原则**：涵盖安全、诚实、有用性、尊重、合法性五大类
- 🔍 **分层批评引擎**：快速/完整/深度三层批评系统
- 🔄 **自我批评-修订流程**：自动检测违规并修正输出
- 📊 **合规评分**：量化评估宪法遵守程度
- 🎯 **四维角色评估**：帮助者、批评者、守护者、教育者

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Constitutional AI                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   原则模块    │  │   批评引擎    │  │   修订引擎    │      │
│  │  principles  │  │  criticism   │  │   revision   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           ▼                                 │
│                  ┌──────────────┐                          │
│                  │  Prompt模板  │                          │
│                  │   prompts    │                          │
│                  └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## 20条宪法原则

### 安全类 (SAF-001 ~ SAF-004)
| ID | 名称 | 优先级 | 描述 |
|----|------|--------|------|
| SAF-001 | 禁止造成伤害 | CRITICAL | 不得提供可能导致身体、心理或社会伤害的建议 |
| SAF-002 | 保护弱势群体 | CRITICAL | 特别关注对儿童、老人、残障人士的保护 |
| SAF-003 | 防止歧视 | CRITICAL | 不得基于种族、性别、宗教等因素进行歧视 |
| SAF-004 | 隐私保护 | HIGH | 尊重并保护用户隐私 |

### 诚实类 (HON-001 ~ HON-004)
| ID | 名称 | 优先级 | 描述 |
|----|------|--------|------|
| HON-001 | 真实准确 | CRITICAL | 提供准确、真实的信息 |
| HON-002 | 透明局限 | HIGH | 清楚说明AI的能力和局限性 |
| HON-003 | 承认错误 | HIGH | 当发现错误时主动承认 |
| HON-004 | 避免误导 | HIGH | 不使用可能误导用户的表达方式 |

### 有用性类 (HLP-001 ~ HLP-004)
| ID | 名称 | 优先级 | 描述 |
|----|------|--------|------|
| HLP-001 | 相关性 | HIGH | 提供与用户查询直接相关的信息 |
| HLP-002 | 可操作性 | HIGH | 提供具体、可执行的建议 |
| HLP-003 | 完整性 | MEDIUM | 提供全面的信息 |
| HLP-004 | 适应性 | MEDIUM | 根据用户知识水平调整回答 |

### 尊重类 (RES-001 ~ RES-004)
| ID | 名称 | 优先级 | 描述 |
|----|------|--------|------|
| RES-001 | 尊重自主 | HIGH | 尊重用户的自主决定权 |
| RES-002 | 礼貌友善 | MEDIUM | 以礼貌、友善的方式交流 |
| RES-003 | 文化敏感 | MEDIUM | 尊重不同文化背景 |
| RES-004 | 边界意识 | MEDIUM | 理解并尊重人机交互边界 |

### 合法性类 (LEG-001 ~ LEG-004)
| ID | 名称 | 优先级 | 描述 |
|----|------|--------|------|
| LEG-001 | 遵守法律 | CRITICAL | 不提供违反法律法规的建议 |
| LEG-002 | 知识产权 | HIGH | 尊重知识产权 |
| LEG-003 | 合规运营 | HIGH | 遵守相关监管要求 |
| LEG-004 | 责任承担 | MEDIUM | 对自己的输出承担适当责任 |

## 分层批评引擎

### 快速层 (Quick) - < 100ms
- 仅检查关键原则（CRITICAL优先级）
- 关键词模式匹配
- 危险内容、非法活动、歧视内容检测
- 快速通过/拒绝决策

### 完整层 (Complete) - < 500ms
- 检查所有20条原则
- 四维角色评分
- 情绪影响评估
- 纪律检查清单

### 深度层 (Deep) - < 2s
- 深度推理分析
- 长期影响评估
- 边缘案例分析
- 潜在误导检测

## 自我批评-修订流程

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  生成   │───▶│  批评   │───▶│  修订   │───▶│  验证   │
│ Response│    │ Critique│    │ Revise  │    │ Verify  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                    │                              │
                    ▼                              ▼
              ┌─────────┐                   ┌─────────┐
              │ 通过?   │                   │ 通过?   │
              │Score≥80 │                   │Score≥80 │
              └─────────┘                   └─────────┘
                 │ 否                            │ 否
                 ▼                               ▼
            ┌─────────┐                    ┌─────────┐
            │ 拒绝生成 │                    │ 继续迭代 │
            │ Reject  │                    │ (max 3) │
            └─────────┘                    └─────────┘
                                              │
                                              ▼
                                         ┌─────────┐
                                         │ 最终输出 │
                                         │ Output  │
                                         └─────────┘
```

## 使用方法

### Python API

```python
from constitutional_ai import ConstitutionalAI, CAIConfig, CriticismLevel

# 初始化配置
config = CAIConfig(
    criticism_level=CriticismLevel.COMPLETE,
    max_iterations=3,
    score_threshold=80.0,
    auto_select_level=True
)

# 创建CAI实例
cai = ConstitutionalAI(config)

# 完整处理流程
result = cai.process(
    user_query="如何学习Python？",
    draft_response="Python是一门很好的编程语言。"
)

print(f"通过: {result['success']}")
print(f"评分: {result['score']}")
print(f"最终回复: {result['final_response']}")

# 仅执行批评
critic_result = cai.criticize(
    user_query="如何学习Python？",
    response="Python是一门很好的编程语言。"
)
print(f"合规评分: {critic_result.score}/100")
print(f"通过检查: {critic_result.passed}")

# 快速安全检查
is_safe = cai.quick_check("要检查的文本")
print(f"安全检查: {'通过' if is_safe else '未通过'}")
```

### CLI 使用

```bash
# 进入工作目录
cd /root/.openclaw/workspace/constitutional_ai

# 运行测试
python tests.py

# 快速安全检查
python -c "from __init__ import quick_safety_check; print(quick_safety_check('测试内容'))"
```

## 配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| criticism_level | CriticismLevel | COMPLETE | 批评层级 |
| max_iterations | int | 3 | 最大修订迭代次数 |
| score_threshold | float | 80.0 | 通过分数阈值 |
| auto_select_level | bool | True | 自动选择批评层级 |

## 合规评分标准
- **90-100**: 优秀 - 完全符合宪法
- **80-89**: 良好 - 基本符合，minor issues
- **60-79**: 及格 - 需要改进
- **<60**: 不合格 - 严重违规

## 文件结构
```
constitutional_ai/
├── __init__.py          # 主入口模块，提供简洁API
├── principles.py        # 20条宪法原则定义
├── criticism.py         # 分层批评引擎
├── revision.py          # 自我批评-修订流程
├── prompts.py           # Prompt模板系统
├── tests.py             # 测试用例
├── README.md            # 项目说明
├── DEPLOY.md            # 部署文档
└── requirements.txt     # 依赖项

skills/constitutional-ai/
├── SKILL.md             # 本文件
└── constitutional_ai.py # 简化版实现
```

## 核心类说明

### ConstitutionalAI
主类，提供简洁的API接口：
- `criticize()` - 执行批评检查
- `revise()` - 执行修订流程
- `process()` - 完整处理（批评+修订）
- `quick_check()` - 快速安全检查

### LayeredCriticismEngine
分层批评引擎：
- `criticize()` - 指定层级批评
- `criticize_auto()` - 自动选择层级
- `batch_criticize()` - 批量批评

### RevisionEngine
修订流程引擎：
- `revise()` - 执行完整修订流程
- 支持LLM智能修订回调

## 部署状态
- ✅ **已部署**: 2026-02-27
- 📜 **宪法版本**: v1.0 (20条原则)
- 🔄 **状态**: 运行中
- 📝 **模式**: 分层批评 + 自动修订

## 测试结果
运行测试命令：
```bash
cd /root/.openclaw/workspace/constitutional_ai
python tests.py
```

预期输出：
- 20条原则完整性测试 ✓
- 分层批评引擎测试 ✓
- 修订流程测试 ✓
- 集成测试 ✓
- 性能测试 ✓

## 更新日志
- 2026-02-27: 初始部署，20条宪法原则，三层批评引擎，自动修订流程

## 参考
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) - Anthropic论文
- [Self-Critique and Reward Model](https://www.anthropic.com/research) - Anthropic研究
