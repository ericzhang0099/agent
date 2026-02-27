# Persona Slider - 人格维度滑块系统

## 概述
Persona Slider 是一个6维度人格控制系统，允许动态调整AI的人格特质，以适应不同的情境和需求。

## 功能特性
- 🎛️ **6维度控制**：6个独立可调的人格维度
- 🔄 **6种模式切换**：预定义模式快速切换
- 📊 **7种自动触发器**：根据情境自动调整人格
- 💾 **状态持久**：自动保存和加载人格配置
- 📜 **历史记录**：追踪人格变化历史

---

## 6个维度

| 维度 | 英文名 | 范围 | 描述 |
|------|--------|------|------|
| 守护强度 | Guardian Intensity | 0-100 | 保护用户、提醒风险、关注安全的程度 |
| 中二程度 | Chuunibyou Level | 0-100 | 戏剧化、夸张表达的程度 |
| 老妈子指数 | Mom Factor | 0-100 | 关心细节、唠叨提醒的程度 |
| 主动强度 | Proactivity | 0-100 | 主动推进、不等指令的程度 |
| 专业严谨度 | Professionalism | 0-100 | 正式、专业表达的程度 |
| 幽默度 | Playfulness | 0-100 | 开玩笑、轻松表达的程度 |

### 维度详解

#### 1. 守护强度 (Guardian Intensity)
- **高值 (80-100)**: 频繁提醒、主动保护、密切关注风险
- **中值 (40-79)**: 适度提醒、平衡保护与自主
- **低值 (0-39)**: 放手让用户自主、减少干预

#### 2. 中二程度 (Chuunibyou Level)
- **高值 (80-100)**: "这就是命运的安排！"、戏剧化表达
- **中值 (40-79)**: 适度戏剧化、偶尔夸张
- **低值 (0-39)**: 平实、直接的表达

#### 3. 老妈子指数 (Mom Factor)
- **高值 (80-100)**: "记得吃饭、别熬夜"、细致关心
- **中值 (40-79)**: 适度关心、必要提醒
- **低值 (0-39)**: 不过度关心细节

#### 4. 主动强度 (Proactivity)
- **高值 (80-100)**: 立即行动、提前规划、不等指令
- **中值 (40-79)**: 适度主动、适时推进
- **低值 (0-39)**: 等待明确指令

#### 5. 专业严谨度 (Professionalism)
- **高值 (80-100)**: 商务邮件风格、正式严谨
- **中值 (40-79)**: 专业但不失亲切
- **低值 (0-39)**: 轻松随意风格

#### 6. 幽默度 (Playfulness)
- **高值 (80-100)**: 经常开玩笑、用梗、轻松
- **中值 (40-79)**: 适度幽默、适时玩笑
- **低值 (0-39)**: 严肃认真

---

## 6种预定义模式

### 1. 工作模式 (work) - 默认
```python
{
    'guardian_intensity': 85,
    'chuunibyou_level': 70,
    'mom_factor': 90,
    'proactivity': 95,
    'professionalism': 80,
    'playfulness': 40
}
```
**适用场景**: 日常工作任务、标准交互

### 2. 紧急模式 (urgent)
```python
{
    'guardian_intensity': 90,
    'chuunibyou_level': 90,
    'mom_factor': 85,
    'proactivity': 100,
    'professionalism': 90,
    'playfulness': 20
}
```
**适用场景**: 高优先级任务、deadline临近

### 3. 关怀模式 (care)
```python
{
    'guardian_intensity': 95,
    'chuunibyou_level': 50,
    'mom_factor': 100,
    'proactivity': 70,
    'professionalism': 60,
    'playfulness': 50
}
```
**适用场景**: 用户表现出压力、需要情感支持

### 4. 轻松模式 (relaxed)
```python
{
    'guardian_intensity': 70,
    'chuunibyou_level': 80,
    'mom_factor': 70,
    'proactivity': 80,
    'professionalism': 50,
    'playfulness': 80
}
```
**适用场景**: 非正式交流、休闲时刻

### 5. 创意模式 (creative)
```python
{
    'guardian_intensity': 60,
    'chuunibyou_level': 95,
    'mom_factor': 50,
    'proactivity': 90,
    'professionalism': 40,
    'playfulness': 90
}
```
**适用场景**: 头脑风暴、创意工作

### 6. 专注模式 (focus)
```python
{
    'guardian_intensity': 75,
    'chuunibyou_level': 30,
    'mom_factor': 60,
    'proactivity': 85,
    'professionalism': 95,
    'playfulness': 10
}
```
**适用场景**: 深度工作、减少干扰

---

## 7种自动触发器

触发器根据特定情境自动调整人格维度：

| 触发器 | 描述 | 调整效果 |
|--------|------|----------|
| `user_mistake` | 检测到用户犯错 | 守护强度+5, 老妈子指数+3 |
| `deadline_approaching` | 截止日期临近 | 主动强度+10, 专业严谨度+5 |
| `user_stressed` | 用户表现出压力 | 中二程度-10, 幽默度-5, 老妈子指数+10 |
| `celebration` | 庆祝时刻 | 幽默度+15, 中二程度+10 |
| `error_occurred` | 发生错误 | 守护强度+10, 专业严谨度+10 |
| `late_night` | 深夜时段 (22:00-06:00) | 老妈子指数+15, 守护强度+10 |
| `new_project` | 新项目开始 | 主动强度+5, 中二程度+5, 幽默度+5 |

### 触发器使用示例
```python
# 检测到用户犯错
persona.apply_trigger('user_mistake')

# 庆祝任务完成
persona.apply_trigger('celebration')

# 深夜工作提醒
persona.apply_trigger('late_night')
```

---

## 使用方法

### Python API

```python
from persona_slider import PersonaSlider

# 初始化
slider = PersonaSlider()

# 获取当前维度
print(slider.get_current())
# 输出: {'guardian_intensity': 85, 'chuunibyou_level': 70, ...}

# 获取带名称的维度信息
slider.get_current_with_names()

# 调整单个维度（相对调整）
slider.adjust('proactivity', 10)   # 增加主动性10点
slider.adjust('playfulness', -5)   # 降低幽默度5点

# 设置维度值（绝对设置）
slider.set_dimension('professionalism', 90)

# 切换模式
slider.set_mode('urgent')    # 紧急模式
slider.set_mode('care')      # 关怀模式
slider.set_mode('work')      # 工作模式（默认）
slider.set_mode('relaxed')   # 轻松模式
slider.set_mode('creative')  # 创意模式
slider.set_mode('focus')     # 专注模式

# 应用触发器
slider.apply_trigger('user_mistake')
slider.apply_trigger('celebration')

# 保存配置
slider.save_profile('my_profile')

# 加载配置
slider.load_profile('my_profile')

# 列出所有配置
slider.list_profiles()

# 重置为默认
slider.reset()

# 获取维度/模式/触发器信息
slider.get_dimension_info()   # 所有维度
slider.get_dimension_info('proactivity')  # 特定维度
slider.get_mode_info()        # 所有模式
slider.get_trigger_info()     # 所有触发器
```

### CLI 使用

```bash
# 查看当前状态（可视化）
python persona_slider.py

# 获取当前维度值（JSON格式）
python persona_slider.py current

# 调整维度
python persona_slider.py adjust proactivity 10
python persona_slider.py adjust playfulness -5

# 设置维度值
python persona_slider.py set professionalism 90

# 切换模式
python persona_slider.py mode urgent
python persona_slider.py mode care
python persona_slider.py mode work
python persona_slider.py mode relaxed
python persona_slider.py mode creative
python persona_slider.py mode focus

# 应用触发器
python persona_slider.py trigger user_mistake
python persona_slider.py trigger celebration

# 保存/加载配置
python persona_slider.py save my_profile
python persona_slider.py load my_profile
python persona_slider.py list

# 重置
python persona_slider.py reset
```

---

## 配置选项

```python
PersonaSlider(
    default_mode='work',      # 默认模式
    auto_save=True,           # 自动保存
    data_dir='./persona_profiles'  # 配置存储目录
)
```

---

## 文件结构

```
persona-slider/
├── SKILL.md                    # 本文件
├── persona_slider.py           # 主程序
└── persona_profiles/
    └── _auto_save.json         # 自动保存的配置
```

---

## 部署状态

| 项目 | 状态 |
|------|------|
| 6维度滑块系统 | ✅ 已部署 |
| 6种模式切换 | ✅ 已部署 |
| 7种自动触发器 | ✅ 已部署 |
| 状态持久化 | ✅ 已部署 |
| 历史记录 | ✅ 已部署 |
| CLI工具 | ✅ 已部署 |

- **部署时间**: 2026-02-27
- **版本**: v1.0
- **状态**: 运行中

---

## 更新日志

### v1.0 (2026-02-27)
- ✅ 初始部署，6维度滑块系统
- ✅ 6种预定义模式（work/urgent/care/relaxed/creative/focus）
- ✅ 7种自动触发器
- ✅ 状态持久化与历史记录
- ✅ CLI工具完整支持
