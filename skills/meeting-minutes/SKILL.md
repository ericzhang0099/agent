# Meeting Minutes Generator - 自动会议纪要生成器

## 概述

自动将会议录音/文本转换为结构化会议纪要，支持飞书文档输出。

## 功能

- 支持音频转录（Whisper API）
- 支持文本输入直接处理
- 自动提取：议题、决策、行动项、负责人、截止时间
- 输出到飞书文档
- 支持中英文

## 使用方法

### 1. 处理音频文件
```bash
openclaw run meeting-minutes --audio meeting.mp3
```

### 2. 处理文本内容
```bash
openclaw run meeting-minutes --text "会议内容..."
```

### 3. 从飞书妙记导入
```bash
openclaw run meeting-minutes --feishu-minutes <meeting_url>
```

## 配置

在 `~/.openclaw/config.toml` 中添加：
```toml
[meeting-minutes]
feishu_folder_token = "your_folder_token"
whisper_api_key = "your_openai_api_key"
```

## 输出格式

生成的会议纪要包含：
1. 会议基本信息（时间、参与人、时长）
2. 会议议题列表
3. 关键决策点
4. 行动项（TODO）- 含负责人和截止时间
5. 待跟进事项
6. 原始录音链接（如有）

## 依赖

- Python 3.9+
- openai-whisper
- feishu-api-sdk

## 作者

KCGS自动生成 | Kimi Claw团队
