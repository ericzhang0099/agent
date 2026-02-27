# Meeting Minutes Generator

自动会议纪要生成器 - 将会议录音或文本转换为结构化会议纪要。

## 快速开始

### 安装

```bash
# 克隆到 skills 目录
cp -r meeting-minutes ~/.openclaw/skills/

# 安装依赖
pip install openai requests
```

### 使用

```bash
# 处理文本
python meeting_minutes.py --text "会议内容..."

# 处理文本文件
python meeting_minutes.py --text meeting.txt

# 处理音频 (需要OpenAI API Key)
python meeting_minutes.py --audio meeting.mp3

# 保存到文件
python meeting_minutes.py --text meeting.txt --output minutes.md
```

### 输入格式示例

```
2026年2月27日 团队周会

参与人：张三、李四、王五

议题：
1. 上周工作回顾
2. 本周计划讨论

决定：
- 确定采用微服务架构
- 下周发布v1.0版本

行动项：
张三负责完成API文档，3月5日前
李四负责部署脚本，下周二前
```

### 输出格式

生成包含以下内容的Markdown格式会议纪要：
- 📋 会议信息（日期、时长、参与人）
- 📌 会议议题
- ✅ 关键决策
- 📝 行动项（TODO）
- 🔍 待跟进事项

## 测试

```bash
# 运行单元测试
python test_meeting_minutes.py

# 运行演示
python test_meeting_minutes.py --demo
```

## 配置

设置环境变量：
```bash
export OPENAI_API_KEY="your-api-key"  # 用于音频转录
export FEISHU_APP_ID="your-app-id"     # 用于飞书导出
export FEISHU_APP_SECRET="your-secret"
```

## 功能特性

- ✅ 智能提取参与人、议题、决策、行动项
- ✅ 支持中文和英文
- ✅ 自动识别截止日期
- ✅ 多种输出格式（Markdown/JSON/Text）
- ✅ 音频转录（Whisper API）
- 🚧 飞书文档导出（开发中）

## 适用场景

- 团队周会/月会
- 项目评审会
- 1对1沟通
- 客户会议
- 线上讨论

## License

MIT
