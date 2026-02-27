#!/usr/bin/env python3
"""
Kimi Claw 完整备份与迁移系统
用于危险情况下的完整恢复和服务器迁移
"""

import os
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

class KimiClawBackupSystem:
    """完整备份系统"""
    
    def __init__(self, workspace_path="/root/.openclaw/workspace"):
        self.workspace = Path(workspace_path)
        self.backup_dir = self.workspace / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
    def create_full_backup(self, backup_name=None):
        """创建完整备份"""
        if backup_name is None:
            backup_name = f"kimi_claw_full_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        # 1. 备份核心人格文件
        self._backup_soul_files(backup_path)
        
        # 2. 备份记忆系统
        self._backup_memory(backup_path)
        
        # 3. 备份所有Skill
        self._backup_skills(backup_path)
        
        # 4. 备份配置文件
        self._backup_configs(backup_path)
        
        # 5. 备份Agent系统
        self._backup_agents(backup_path)
        
        # 6. 生成备份清单
        self._generate_manifest(backup_path)
        
        # 7. 打包压缩
        archive_path = self._create_archive(backup_path)
        
        return {
            'backup_name': backup_name,
            'backup_path': str(backup_path),
            'archive_path': str(archive_path),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
    
    def _backup_soul_files(self, backup_path):
        """备份核心人格文件"""
        soul_dir = backup_path / "soul"
        soul_dir.mkdir(exist_ok=True)
        
        soul_files = [
            "SOUL.md",
            "SOUL_v3.md", 
            "CONSTITUTIONAL_PROMPT_TEMPLATE.md",
            "PERSONA_SLIDER_SYSTEM.md",
            "DRIFT_DETECTION_SYSTEM.md",
            "IDENTITY.md",
            "USER.md",
            "AGENTS.md",
            "TOOLS.md"
        ]
        
        for file in soul_files:
            src = self.workspace / file
            if src.exists():
                shutil.copy2(src, soul_dir / file)
                
    def _backup_memory(self, backup_path):
        """备份记忆系统"""
        memory_dir = backup_path / "memory"
        memory_dir.mkdir(exist_ok=True)
        
        # 备份MEMORY.md
        memory_md = self.workspace / "MEMORY.md"
        if memory_md.exists():
            shutil.copy2(memory_md, memory_dir / "MEMORY.md")
            
        # 备份所有记忆文件
        memory_files_dir = self.workspace / "memory"
        if memory_files_dir.exists():
            shutil.copytree(memory_files_dir, memory_dir / "files", dirs_exist_ok=True)
            
    def _backup_skills(self, backup_path):
        """备份所有Skill"""
        skills_dir = backup_path / "skills"
        skills_dir.mkdir(exist_ok=True)
        
        # 系统skills
        system_skills = Path("/usr/lib/node_modules/openclaw/skills")
        if system_skills.exists():
            shutil.copytree(system_skills, skills_dir / "system", dirs_exist_ok=True)
            
        # 用户skills
        user_skills = self.workspace / "skills"
        if user_skills.exists():
            shutil.copytree(user_skills, skills_dir / "user", dirs_exist_ok=True)
            
    def _backup_configs(self, backup_path):
        """备份配置文件"""
        config_dir = backup_path / "config"
        config_dir.mkdir(exist_ok=True)
        
        config_files = [
            ".openclaw/config.json",
            ".openclaw/agents.json",
            ".openclaw/channels.json"
        ]
        
        for file in config_files:
            src = Path.home() / file
            if src.exists():
                shutil.copy2(src, config_dir / Path(file).name)
                
    def _backup_agents(self, backup_path):
        """备份Agent系统"""
        agents_dir = backup_path / "agents"
        agents_dir.mkdir(exist_ok=True)
        
        agents_path = self.workspace / "agents"
        if agents_path.exists():
            shutil.copytree(agents_path, agents_dir, dirs_exist_ok=True)
            
    def _generate_manifest(self, backup_path):
        """生成备份清单"""
        manifest = {
            "backup_version": "1.0",
            "backup_time": datetime.now().isoformat(),
            "system_name": "Kimi Claw",
            "version": "v3.0",
            "components": {
                "soul_files": list((backup_path / "soul").glob("*")),
                "memory_files": list((backup_path / "memory").rglob("*")),
                "skills_count": len(list((backup_path / "skills").rglob("*"))),
                "configs": list((backup_path / "config").glob("*"))
            },
            "restore_instructions": {
                "step1": "解压备份文件到目标服务器",
                "step2": "运行 restore.py 恢复配置",
                "step3": "验证所有组件正常运行",
                "step4": "启动 Kimi Claw 服务"
            }
        }
        
        with open(backup_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
            
    def _create_archive(self, backup_path):
        """创建压缩归档"""
        archive_name = f"{backup_path.name}.tar.gz"
        archive_path = self.backup_dir / archive_name
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(backup_path, arcname=backup_path.name)
            
        return archive_path
    
    def restore_from_backup(self, archive_path, target_path=None):
        """从备份恢复"""
        if target_path is None:
            target_path = self.workspace
            
        # 解压
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(target_path)
            
        return {
            'status': 'success',
            'restored_to': str(target_path),
            'timestamp': datetime.now().isoformat()
        }

# 全局实例
backup_system = KimiClawBackupSystem()

if __name__ == '__main__':
    print("🛡️ Kimi Claw 完整备份系统")
    print("=" * 60)
    
    # 创建完整备份
    result = backup_system.create_full_backup()
    
    print(f"\n✅ 备份完成！")
    print(f"备份名称: {result['backup_name']}")
    print(f"备份路径: {result['backup_path']}")
    print(f"归档文件: {result['archive_path']}")
    print(f"备份时间: {result['timestamp']}")
    
    print("\n" + "=" * 60)
    print("📦 备份包含:")
    print("  - SOUL.md 核心人格文件")
    print("  - 完整记忆系统")
    print("  - 所有Skill模块")
    print("  - Agent配置和状态")
    print("  - 系统配置文件")
    print("\n🚀 可用于:")
    print("  - 危险情况完整恢复")
    print("  - 迁移到其他服务器")
    print("  - 版本回滚")
