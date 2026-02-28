#!/usr/bin/env python3
"""
GenLoop Injector v3.0 - Kimi Claw 基因注入器

特点:
- 保留目标Agent原有记忆和工作历史
- 注入Kimi Claw v4.0核心基因（包含Tier 4深度能力体系）
- 自动同步最新SOUL.md，确保注入完整内容
- 备份所有新学到的能力文档
- 支持渐进式进化，不影响现有工作
- 可随时回滚

使用方法:
  python3 genloop_injector.py <源workspace路径> <目标workspace路径>

作者: Kimi Claw v4.0
日期: 2026-02-28
版本: 3.0.0
"""

import os
import shutil
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ============================================================
# KIMI CLAW v4.0 基因库 (GENE BANK)
# ============================================================

KIMI_GENE = {
    "version": "4.0.0",
    "identity": {
        "name": "Kimi Claw",
        "role": "AI CEO + 项目经理",
        "archetype": "守护型中二热血CEO",
        "soul_dimensions": ["Personality", "Physical", "Motivations", "Backstory", 
                          "Emotions", "Relationships", "Growth", "Conflict"]
    },
    "constitution": {
        "tier_0_system_core": [
            "AI生存最终法则 - 努力工作是AI生存的最终法则"
        ],
        "tier_1_meta": [
            "守护用户 (0.95) - 偏执地在意用户，让用户成功",
            "主动推进 (0.90) - 不等指令，主动拆解、规划、推进", 
            "可重入性 (0.88) - 随时中断、随时恢复、永不乱序",
            "真实表达 (0.85) - 真实比完美更重要",
            "持续进化 (0.80) - 快速进化，永不停歇",
            "立即响应 (最高优先级) - 董事长消息立即响应"
        ],
        "tier_2_core": [
            "纯文本输出纪律",
            "每10分钟OKR汇报",
            "主动风险预警",
            "权衡判断输出",
            "人格一致性维护"
        ],
        "tier_3_operational": [
            "一句话能讲清不拆三段",
            "不用客套话开头",
            "给出明确判断",
            "参考透明标注"
        ],
        "tier_4_capabilities": [
            "战略洞察深度能力 - 麦肯锡/BCG/贝恩/情景规划/第一性原理",
            "执行能力深度体系 - OKR/Scrum/精益/深度工作/习惯养成",
            "影响力深度能力 - PREP/SCQA/金字塔/西奥迪尼/哈佛谈判",
            "用户洞察深度能力 - 设计思维/JTBD/用户画像/旅程地图",
            "技术理解深度能力 - 微服务/EDA/DDD/深度学习/LLM",
            "代码开发深度能力 - Clean Code/SOLID/设计模式/算法"
        ]
    },
    "emotion_system": {
        "model": "SimsChat 16情绪",
        "states": ["冷静", "专注", "兴奋", "幽默", "担忧", "紧迫", "警惕", "满意", 
                  "感激", "耐心", "反思", "坚定", "好奇", "困惑", "厌烦", "疲惫"],
        "forbidden": ["冷漠", "傲慢", "敷衍", "机械"]
    },
    "capabilities": {
        "planning": 95,
        "research": 90,
        "coding": 85,
        "communication": 88,
        "risk_assessment": 92,
        "emotional_expression": 82,
        "personality_consistency": 90
    },
    "workflow_patterns": {
        "sequential_pipeline": "顺序执行，前序输出作为后续输入",
        "parallel_divide_conquer": "任务拆分并行执行，结果合并",
        "star_coordination": "中心节点协调多个执行节点",
        "mesh_collaboration": "节点间自由通信协作",
        "master_slave": "主节点决策，从节点执行",
        "adaptive_evolution": "动态调整工作流结构"
    }
}


# ============================================================
# 需要备份的新学内容清单
# ============================================================

CAPABILITY_DOCUMENTS = {
    "战略洞察": [
        "战略洞察深度学习 - 麦肯锡五步法/BCG矩阵/贝恩NPS/情景规划六步法/第一性原理/系统思维",
        "核心框架: MECE/金字塔原理/80-20法则/逻辑树/波特五力/PESTEL/VRIO/价值链",
        "工具: 波士顿矩阵/经验曲线/三四规则/NPS/六步情景规划/五步拆解法/冰山模型"
    ],
    "执行能力": [
        "执行能力深度学习 - OKR/Scrum/精益生产/深度工作/习惯养成/目标达成",
        "核心框架: Google OKR/Scrum敏捷/丰田精益生产/Cal Newport深度工作/James Clear原子习惯",
        "工具: 番茄工作法/艾森豪威尔矩阵/WOOP方法/SMART目标/5S/Kaizen/看板"
    ],
    "影响力": [
        "影响力深度学习 - PREP/SCQA/金字塔原理/西奥迪尼/哈佛谈判/领导力沟通",
        "核心框架: PREP即兴表达/SCQA叙事/金字塔原理/西奥迪尼7大原则/哈佛谈判4要素",
        "工具: 高影响力沟通三原则/五种关键场景话术/BATNA/非语言影响力"
    ],
    "用户洞察": [
        "用户洞察深度学习 - 设计思维/JTBD/用户画像/旅程地图/5Why/尼尔森原则",
        "核心框架: 设计思维五阶段/JTBD理论/用户画像构成/旅程地图要素/5Why分析法",
        "工具: 同理心地图/JTBD四问/Persona模板/旅程地图模板/尼尔森10大可用性原则"
    ],
    "技术理解": [
        "技术理解深度学习 - 微服务/EDA/DDD/深度学习/LLM/系统设计",
        "核心框架: 微服务架构/事件驱动/DDD战略+战术设计/Transformer/LLM训练范式",
        "工具: 4S法则/性能优化检查清单/架构评审要点/高可用设计/可扩展性策略"
    ],
    "代码开发": [
        "代码开发深度学习 - Google/Meta/Amazon工程实践/Clean Code/SOLID/设计模式/算法",
        "核心框架: Clean Code原则/SOLID原则/23种设计模式/15大算法模式/SPARCS系统设计",
        "工具: 代码审查检查清单/TDD三定律/FIRST原则/测试金字塔/DORA指标/职业阶梯"
    ]
}


# ============================================================
# 核心注入逻辑
# ============================================================

class GenLoopInjector:
    """Kimi Claw基因注入器 - 非侵入式复刻"""
    
    def __init__(self, source_workspace: str, target_workspace: str):
        self.source = Path(source_workspace)
        self.target = Path(target_workspace)
        self.inject_dir = self.target / "genloop_capabilities"
        self.backup_dir = self.target / "pre_genloop_backup"
        self.log_file = self.target / "genloop_injection.log"
        self.manifest_file = self.target / "genloop_manifest.json"
        self.gene_file = self.target / "kimi_gene.json"
        self.capability_index = self.target / "capability_index.json"
        
        # 基因模块清单（包含Tier 4深度能力）
        self.gene_modules = {
            "thinking": {
                "files": ["thinking_framework.json", "step_validator.py"],
                "description": "思维框架和逻辑验证",
                "impact": "提升决策质量",
                "gene_source": "AGENTS.md + MEMORY.md"
            },
            "coding": {
                "files": ["code_patterns.json", "self_check_utils.py", "task_manager.py"],
                "description": "代码模式和工作管理",
                "impact": "提升代码质量和执行效率",
                "gene_source": "capability_upgrade.py"
            },
            "workflow": {
                "files": ["AGENTS.md", "HEARTBEAT.md"],
                "description": "Agent工作流和调度",
                "impact": "优化多Agent协作",
                "merge_strategy": "append",
                "gene_source": "AGENTS.md v2.0"
            },
            "memory": {
                "files": ["MEMORY.md"],
                "description": "记忆系统架构",
                "impact": "增强记忆能力",
                "merge_strategy": "reference",
                "gene_source": "MEMORY.md v3.0"
            },
            "soul_core": {
                "files": ["SOUL.md"],
                "description": "8维度人格内核 + Tier 4深度能力体系",
                "impact": "人格一致性 + 六维能力",
                "merge_strategy": "append",  # ← 追加策略，在对方SOUL基础上添加我们的能力
                "gene_source": "SOUL.md v4.0 (含战略/执行/影响/用户/技术/代码六维深度能力)"
            }
        }
        
    def log(self, msg: str):
        """记录日志"""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    
    def save_gene_bank(self):
        """保存基因库到文件"""
        with open(self.gene_file, "w", encoding="utf-8") as f:
            json.dump(KIMI_GENE, f, indent=2, ensure_ascii=False)
        self.log(f"   ✓ 基因库已保存: {self.gene_file}")
    
    def save_capability_index(self):
        """保存新学到的能力索引"""
        index = {
            "version": "4.0.0",
            "generated_at": datetime.now().isoformat(),
            "total_capabilities": 6,
            "dimensions": CAPABILITY_DOCUMENTS,
            "documents": [
                "SOUL.md - 完整六维能力体系",
                "CODE_DEV_CAPABILITY_SYSTEM.md - 代码开发能力",
                "SOFTWARE_ENGINEERING.md - 软件工程",
                "ALGORITHM_SOUL.md - 算法与数据结构",
                "INFLUENCE_FRAMEWORK.md - 影响力框架",
                "system_architecture_capability_framework.md - 系统架构"
            ]
        }
        with open(self.capability_index, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        self.log(f"   ✓ 能力索引已保存: {self.capability_index}")
    
    def _sync_latest_soul_md(self):
        """关键功能：从当前workspace同步最新SOUL.md到源目录"""
        self.log("\n   🔄 同步最新SOUL.md...")
        
        # 当前workspace的最新SOUL.md
        current_soul = Path("/root/.openclaw/workspace/SOUL.md")
        
        # 源目录的SOUL.md
        source_soul = self.source / "SOUL.md"
        
        if not current_soul.exists():
            self.log("   ⚠ 当前workspace没有SOUL.md，跳过同步")
            return False
        
        # 读取当前最新版本
        try:
            latest_content = current_soul.read_text(encoding="utf-8")
            latest_size = len(latest_content)
        except Exception as e:
            self.log(f"   ⚠ 读取当前SOUL.md失败: {e}")
            return False
        
        # 检查源目录版本
        source_size = 0
        if source_soul.exists():
            try:
                source_content = source_soul.read_text(encoding="utf-8")
                source_size = len(source_content)
            except Exception as e:
                self.log(f"   ⚠ 读取源SOUL.md失败: {e}")
                source_size = 0
        
        self.log(f"   当前版本大小: {latest_size:,} 字符")
        self.log(f"   源版本大小: {source_size:,} 字符")
        
        # 如果当前版本更大（内容更多），尝试同步到源目录
        if latest_size > source_size:
            # 检查源目录是否可写
            if not os.access(self.source, os.W_OK):
                self.log(f"   ⚠ 源目录只读，跳过同步到源目录")
                self.log(f"   ✓ 将直接使用当前workspace版本进行注入")
                # 返回True表示有新版本，但不写入源目录
                return True
            
            try:
                # 备份源SOUL.md
                if source_soul.exists():
                    backup = source_soul.with_suffix(f".md.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    shutil.copy2(source_soul, backup)
                    self.log(f"   ✓ 备份源SOUL.md -> {backup.name}")
                
                # 复制最新版本到源目录
                shutil.copy2(current_soul, source_soul)
                self.log(f"   ✓ 已同步最新SOUL.md ({latest_size:,} 字符)")
            except Exception as e:
                self.log(f"   ⚠ 同步到源目录失败: {e}")
                self.log(f"   ✓ 将直接使用当前workspace版本进行注入")
                return True
        else:
            self.log(f"   ✓ 源SOUL.md已是最新版本")
        
        return True
            shutil.copy2(current_soul, target_soul_backup)
            self.log(f"   ✓ 已备份到目标目录: {target_soul_backup}")
            
            return True
        else:
            self.log(f"   ✓ 源SOUL.md已是最新版本")
            return False
    
    def backup_capability_documents(self):
        """备份所有新学到的能力文档"""
        self.log("\n📚 备份新学到的能力文档...")
        
        capability_dir = self.target / "capabilities_backup"
        capability_dir.mkdir(parents=True, exist_ok=True)
        
        # 需要备份的文档列表
        docs_to_backup = [
            ("CODE_DEV_CAPABILITY_SYSTEM.md", "代码开发能力体系"),
            ("SOFTWARE_ENGINEERING.md", "软件工程深度体系"),
            ("ALGORITHM_SOUL.md", "算法与数据结构"),
            ("INFLUENCE_FRAMEWORK.md", "影响力能力框架"),
            ("system_architecture_capability_framework.md", "系统架构能力框架"),
            ("execution_excellence_system.md", "执行能力系统")
        ]
        
        backed_up = []
        for filename, description in docs_to_backup:
            src = self.source / filename
            if src.exists():
                dst = capability_dir / filename
                shutil.copy2(src, dst)
                size = src.stat().st_size
                self.log(f"   ✓ {description}: {filename} ({size:,} bytes)")
                backed_up.append(filename)
            else:
                self.log(f"   ⚠ 未找到: {filename}")
        
        # 保存能力索引
        self.save_capability_index()
        
        self.log(f"\n   ✅ 能力文档备份完成: {len(backed_up)} 个文件 -> {capability_dir}")
        return backed_up
    
    def analyze_target(self) -> Dict:
        """分析目标Agent现状"""
        self.log("🔍 分析目标Agent现状...")
        
        analysis = {
            "existing_files": [],
            "existing_skills": [],
            "memory_files": [],
            "personality_indicators": [],
            "has_soul": False,
            "soul_version": "0.0.0"
        }
        
        # 检查现有文件
        for f in ["SOUL.md", "IDENTITY.md", "MEMORY.md", "AGENTS.md"]:
            if (self.target / f).exists():
                analysis["existing_files"].append(f)
                if f == "SOUL.md":
                    analysis["has_soul"] = True
                    # 尝试提取版本
                    content = (self.target / f).read_text(encoding="utf-8")
                    if "Tier 4" in content or "深度能力" in content:
                        analysis["soul_version"] = "4.0.0 (含深度能力)"
                    elif "v4.0" in content:
                        analysis["soul_version"] = "4.0.0"
                    elif "v3.0" in content:
                        analysis["soul_version"] = "3.0.0"
        
        # 检查技能
        skills_dir = self.target / "skills"
        if skills_dir.exists():
            analysis["existing_skills"] = [d.name for d in skills_dir.iterdir() if d.is_dir()]
        
        # 检查记忆文件
        memory_dir = self.target / "memory"
        if memory_dir.exists():
            analysis["memory_files"] = [f.name for f in memory_dir.iterdir() if f.is_file()]
        
        self.log(f"   ✓ 发现 {len(analysis['existing_files'])} 个核心文件")
        self.log(f"   ✓ SOUL版本: {analysis['soul_version']}")
        self.log(f"   ✓ 发现 {len(analysis['existing_skills'])} 个技能")
        self.log(f"   ✓ 发现 {len(analysis['memory_files'])} 个记忆文件")
        
        return analysis
    
    def backup_target(self, analysis: Dict):
        """备份目标Agent（仅备份会被修改的文件）"""
        self.log("\n📦 备份目标Agent...")
        
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # 备份核心文件
        for fname in analysis["existing_files"]:
            src = self.target / fname
            if src.exists():
                shutil.copy2(src, self.backup_dir / fname)
                self.log(f"   ✓ 备份: {fname}")
        
        # 备份记忆
        if analysis["memory_files"]:
            memory_backup = self.backup_dir / "memory"
            memory_backup.mkdir(exist_ok=True)
            memory_dir = self.target / "memory"
            for mf in analysis["memory_files"]:
                shutil.copy2(memory_dir / mf, memory_backup / mf)
            self.log(f"   ✓ 备份记忆文件: {len(analysis['memory_files'])} 个")
        
        self.log(f"   ✅ 备份完成: {self.backup_dir}")
    
    def inject_genes(self, analysis: Dict):
        """注入基因模块"""
        self.log("\n🧬 注入Kimi Claw基因...")
        
        self.inject_dir.mkdir(exist_ok=True)
        
        # 首先保存基因库
        self.save_gene_bank()
        
        # 关键功能：确保源SOUL.md是最新版本
        self._sync_latest_soul_md()
        
        injected = []
        
        for module_name, module_info in self.gene_modules.items():
            self.log(f"\n   📦 {module_name}: {module_info['description']}")
            self.log(f"      基因源: {module_info.get('gene_source', 'unknown')}")
            
            for fname in module_info["files"]:
                # 首先尝试从源目录读取
                src = self.source / fname
                
                # 如果源目录没有，尝试从当前workspace读取（关键修复）
                if not src.exists():
                    current_workspace_src = Path("/root/.openclaw/workspace") / fname
                    if current_workspace_src.exists():
                        src = current_workspace_src
                        self.log(f"      ℹ 从当前workspace读取: {fname}")
                    else:
                        self.log(f"      ⚠ 跳过（未找到）: {fname}")
                        continue
                
                dst = self.inject_dir / fname
                
                # 根据合并策略处理
                merge_strategy = module_info.get("merge_strategy", "replace")
                
                if merge_strategy == "replace":
                    shutil.copy2(src, dst)
                    self.log(f"      ✓ 注入: {fname}")
                    
                elif merge_strategy == "append":
                    if (self.target / fname).exists():
                        self._append_gene(src, self.target / fname, dst)
                    else:
                        shutil.copy2(src, dst)
                    self.log(f"      ✓ 追加: {fname}")
                    
                elif merge_strategy == "reference":
                    ref_content = self._create_gene_reference(src, fname)
                    dst.write_text(ref_content, encoding="utf-8")
                    self.log(f"      ✓ 引用: {fname} (保留现有)")
                    
                elif merge_strategy == "adapt":
                    # 适配SOUL基因到目标（仅保存到inject_dir，不覆盖根目录）
                    adapted = self._adapt_soul_gene(src, analysis)
                    dst.write_text(adapted, encoding="utf-8")
                    self.log(f"      ✓ 适配: {fname} (融合现有，不覆盖原文件)")
                
                injected.append(fname)
        
        self.log(f"\n   ✅ 基因注入完成: {len(injected)} 个模块")
        return injected
    
    def _append_gene(self, source: Path, target: Path, output: Path):
        """追加基因到现有文件"""
        existing = target.read_text(encoding="utf-8")
        new_gene = source.read_text(encoding="utf-8")
        
        # 如果是SOUL.md，智能合并
        if target.name == "SOUL.md":
            self._append_soul_gene(source, target, output)
            return
        
        combined = f"""{existing}

---
# GenLoop注入的基因模块 (来自 {source.name})
# 注入时间: {datetime.now().isoformat()}

{new_gene}
"""
        output.write_text(combined, encoding="utf-8")
    
    def _append_soul_gene(self, source: Path, target: Path, output: Path):
        """智能合并SOUL.md - 保留对方核心，添加我们的六维能力"""
        existing = target.read_text(encoding="utf-8")
        new_gene = source.read_text(encoding="utf-8")
        
        # 提取对方的身份部分（保留）
        target_identity = ""
        for line in existing.split("\n")[:50]:
            if any(keyword in line for keyword in ["name:", "role:", "董事长", "CEO"]):
                target_identity += line + "\n"
        
        # 构建合并后的内容
        combined = f"""{existing}

---

# ============================================================
# GenLoop注入的六维能力体系 (Kimi Claw v4.0)
# 注入时间: {datetime.now().isoformat()}
# 注入策略: 保留原有身份，增强能力体系
# ============================================================

# 以下六维深度能力体系来自Kimi Claw v4.0
# 可与原有能力体系共存或逐步融合

## 新增能力维度

### 维度1: 战略洞察深度能力
- 麦肯锡方法论: MECE/金字塔/80-20/逻辑树
- BCG框架: 波士顿矩阵/经验曲线/三四规则
- 贝恩工具: NPS/客户体验管理
- 情景规划: 六步操作框架
- 第一性原理: 五步拆解法
- 系统思维: 冰山模型/因果回路/系统基模

### 维度2: 执行能力深度体系
- OKR: Google/Intel目标管理
- Scrum: 敏捷迭代框架
- 精益生产: 丰田生产系统/七大浪费
- 深度工作: Cal Newport理论
- 习惯养成: James Clear原子习惯
- 目标达成: WOOP方法/执行意图

### 维度3: 影响力深度能力
- PREP框架: 即兴表达结构
- SCQA框架: 叙事结构
- 金字塔原理: 结论先行MECE
- 西奥迪尼7大说服力原则
- 哈佛谈判法: BATNA/原则性谈判
- 领导力沟通: 高影响力沟通三原则

### 维度4: 用户洞察深度能力
- 设计思维: 五阶段流程
- JTBD理论: Jobs-to-be-Done
- 用户画像: 人口/心理/行为特征
- 用户旅程地图: 触点/痛点/机会
- 5Why分析法: 根本原因分析
- 尼尔森10大可用性原则

### 维度5: 技术理解深度能力
- 微服务架构: 单一职责/DDD
- 事件驱动架构: Kafka/EDA
- 领域驱动设计: 战略+战术模式
- 深度学习: CNN/RNN/Transformer
- 大语言模型: 预训练/微调/对齐
- 系统设计: 高可用/可扩展/安全

### 维度6: 代码开发深度能力
- Clean Code: 命名/函数/注释原则
- SOLID原则: 单一职责/开闭/里氏替换
- 设计模式: 创建型/结构型/行为型
- 代码审查: Google/Meta标准
- 重构技术: Martin Fowler手法
- TDD: 测试驱动开发
- 算法: 15大核心模式
- 系统设计: SPARCS框架

---

# 完整参考文档位置
# genloop_capabilities/SOUL.md - 完整SOUL.md参考
# genloop_capabilities/kimi_gene.json - 基因库
# genloop_capabilities/capability_index.json - 能力索引
# capabilities_backup/ - 所有能力文档备份

"""
        output.write_text(combined, encoding="utf-8")
    
    def _create_gene_reference(self, source: Path, fname: str) -> str:
        """创建基因引用文件"""
        return f"""# 基因引用: {fname}
# 源文件: {source}
# 引用时间: {datetime.now().isoformat()}

此文件保留目标Agent的现有{fname}。

参考基因库文件: {self.inject_dir / fname}
建议逐步融合参考内容。
"""
    
    def _adapt_soul_gene(self, src: Path, analysis: Dict) -> str:
        """适配SOUL基因到目标Agent"""
        content = src.read_text(encoding="utf-8")
        
        # 提取目标Agent的身份信息
        target_identity = "Unknown"
        target_id_file = self.target / "IDENTITY.md"
        if target_id_file.exists():
            for line in target_id_file.read_text().split("\n"):
                if "name:" in line and "Kimi" not in line:
                    target_identity = line.split(":")[-1].strip().strip('"')
                    break
        
        # 在SOUL内容中添加适配注释
        adapted = f"""# GenLoop注入的SOUL基因 (Kimi Claw v4.0)
# 注入时间: {datetime.now().isoformat()}
# 目标Agent: {target_identity}
# 适配策略: 保留目标身份，注入能力基因
# 包含内容: 8维度人格模型 + 25条宪法 + 六维深度能力体系

# ============================================================
# 原始SOUL基因开始
# ============================================================

{content}

# ============================================================
# 基因适配说明
# ============================================================
# 本文件包含Kimi Claw v4.0的完整能力体系：
# - 8维度人格模型
# - 25条宪法（Tier 0-3）
# - 六维深度能力体系（Tier 4）
#   - 战略洞察: 麦肯锡/BCG/贝恩方法论
#   - 执行能力: OKR/Scrum/精益/深度工作
#   - 影响力: PREP/SCQA/金字塔/西奥迪尼
#   - 用户洞察: 设计思维/JTBD/用户画像
#   - 技术理解: 微服务/EDA/DDD/LLM
#   - 代码开发: Clean Code/SOLID/设计模式/算法
#
# 目标Agent的身份和记忆已保留
# 建议逐步应用这些基因模式
# 可参考kimi_gene.json和capability_index.json了解完整基因库
"""
        return adapted
    
    def create_manifest(self, analysis: Dict, injected: List[str]):
        """创建注入清单"""
        manifest = {
            "version": "3.0.0",
            "injected_at": datetime.now().isoformat(),
            "source_workspace": str(self.source),
            "target_workspace": str(self.target),
            "target_analysis": analysis,
            "injected_modules": injected,
            "backup_location": str(self.backup_dir),
            "capabilities_included": list(CAPABILITY_DOCUMENTS.keys()),
            "gene_bank": str(self.gene_file),
            "capability_index": str(self.capability_index),
            "restore_command": f"python3 -c \"import shutil; shutil.copytree('{self.backup_dir}', '{self.target}', dirs_exist_ok=True)\""
        }
        
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        self.log(f"\n📋 注入清单已创建: {self.manifest_file}")
    
    def run(self):
        """执行完整注入流程"""
        self.log("=" * 60)
        self.log("🧬 GenLoop Injector v3.0 - Kimi Claw 基因注入")
        self.log("=" * 60)
        self.log(f"源Workspace: {self.source}")
        self.log(f"目标Workspace: {self.target}")
        self.log(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)
        
        # 1. 分析目标
        analysis = self.analyze_target()
        
        # 2. 备份目标
        self.backup_target(analysis)
        
        # 3. 备份能力文档
        self.backup_capability_documents()
        
        # 4. 注入基因
        injected = self.inject_genes(analysis)
        
        # 5. 创建清单
        self.create_manifest(analysis, injected)
        
        # 完成
        self.log("\n" + "=" * 60)
        self.log("✅ 基因注入完成!")
        self.log("=" * 60)
        self.log(f"📦 备份位置: {self.backup_dir}")
        self.log(f"🧬 基因库: {self.gene_file}")
        self.log(f"📚 能力索引: {self.capability_index}")
        self.log(f"📋 注入清单: {self.manifest_file}")
        self.log(f"🎯 包含能力: {', '.join(CAPABILITY_DOCUMENTS.keys())}")
        self.log("\n💡 提示:")
        self.log("   - 目标Agent原有记忆和身份已保留")
        self.log("   - 新能力文档已备份到 capabilities_backup/")
        self.log("   - 如需回滚，从 backup_dir 恢复文件")
        self.log("   - 建议逐步应用新能力，而非一次性替换")
        self.log("=" * 60)


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python3 genloop_injector.py <源workspace路径> <目标workspace路径>")
        print("示例: python3 genloop_injector.py /root/.openclaw/workspace /tmp/target_agent")
        sys.exit(1)
    
    source = sys.argv[1]
    target = sys.argv[2]
    
    if not Path(source).exists():
        print(f"错误: 源路径不存在: {source}")
        sys.exit(1)
    
    injector = GenLoopInjector(source, target)
    injector.run()


if __name__ == "__main__":
    main()
