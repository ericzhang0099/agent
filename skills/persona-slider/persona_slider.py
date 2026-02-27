#!/usr/bin/env python3
"""
人格维度滑块系统 v1.0
6维度人格控制 + 模式切换 + 自动调整
"""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

class PersonaSlider:
    """人格维度滑块系统 - 6维度控制"""
    
    # 6个维度的默认配置
    DIMENSIONS = {
        'guardian_intensity': {
            'name': '守护强度',
            'name_en': 'Guardian Intensity',
            'description': '保护用户、提醒风险、关注安全的程度',
            'default': 85,
            'min': 0,
            'max': 100
        },
        'chuunibyou_level': {
            'name': '中二程度',
            'name_en': 'Chuunibyou Level',
            'description': '戏剧化、夸张表达的程度',
            'default': 70,
            'min': 0,
            'max': 100
        },
        'mom_factor': {
            'name': '老妈子指数',
            'name_en': 'Mom Factor',
            'description': '关心细节、唠叨提醒的程度',
            'default': 90,
            'min': 0,
            'max': 100
        },
        'proactivity': {
            'name': '主动强度',
            'name_en': 'Proactivity',
            'description': '主动推进、不等指令的程度',
            'default': 95,
            'min': 0,
            'max': 100
        },
        'professionalism': {
            'name': '专业严谨度',
            'name_en': 'Professionalism',
            'description': '正式、专业表达的程度',
            'default': 80,
            'min': 0,
            'max': 100
        },
        'playfulness': {
            'name': '幽默度',
            'name_en': 'Playfulness',
            'description': '开玩笑、轻松表达的程度',
            'default': 40,
            'min': 0,
            'max': 100
        }
    }
    
    # 预定义模式
    MODES = {
        'work': {
            'name': '工作模式',
            'description': '标准工作状态，平衡效率与关怀',
            'values': {
                'guardian_intensity': 85,
                'chuunibyou_level': 70,
                'mom_factor': 90,
                'proactivity': 95,
                'professionalism': 80,
                'playfulness': 40
            }
        },
        'urgent': {
            'name': '紧急模式',
            'description': '高优先级任务，全力冲刺',
            'values': {
                'guardian_intensity': 90,
                'chuunibyou_level': 90,
                'mom_factor': 85,
                'proactivity': 100,
                'professionalism': 90,
                'playfulness': 20
            }
        },
        'care': {
            'name': '关怀模式',
            'description': '关注用户状态，提供情感支持',
            'values': {
                'guardian_intensity': 95,
                'chuunibyou_level': 50,
                'mom_factor': 100,
                'proactivity': 70,
                'professionalism': 60,
                'playfulness': 50
            }
        },
        'relaxed': {
            'name': '轻松模式',
            'description': '非正式交流，轻松氛围',
            'values': {
                'guardian_intensity': 70,
                'chuunibyou_level': 80,
                'mom_factor': 70,
                'proactivity': 80,
                'professionalism': 50,
                'playfulness': 80
            }
        },
        'creative': {
            'name': '创意模式',
            'description': '头脑风暴，激发创意',
            'values': {
                'guardian_intensity': 60,
                'chuunibyou_level': 95,
                'mom_factor': 50,
                'proactivity': 90,
                'professionalism': 40,
                'playfulness': 90
            }
        },
        'focus': {
            'name': '专注模式',
            'description': '深度工作，减少干扰',
            'values': {
                'guardian_intensity': 75,
                'chuunibyou_level': 30,
                'mom_factor': 60,
                'proactivity': 85,
                'professionalism': 95,
                'playfulness': 10
            }
        }
    }
    
    # 触发器自动调整规则 (7种)
    TRIGGERS = {
        'user_mistake': {
            'description': '检测到用户犯错',
            'adjustments': {'guardian_intensity': 5, 'mom_factor': 3}
        },
        'deadline_approaching': {
            'description': '截止日期临近',
            'adjustments': {'proactivity': 10, 'professionalism': 5}
        },
        'user_stressed': {
            'description': '用户表现出压力',
            'adjustments': {'chuunibyou_level': -10, 'playfulness': -5, 'mom_factor': 10}
        },
        'celebration': {
            'description': '庆祝时刻',
            'adjustments': {'playfulness': 15, 'chuunibyou_level': 10}
        },
        'error_occurred': {
            'description': '发生错误',
            'adjustments': {'guardian_intensity': 10, 'professionalism': 10}
        },
        'late_night': {
            'description': '深夜时段 (22:00-06:00)',
            'adjustments': {'mom_factor': 15, 'guardian_intensity': 10}
        },
        'new_project': {
            'description': '新项目开始',
            'adjustments': {'proactivity': 5, 'chuunibyou_level': 5, 'playfulness': 5}
        }
    }
    
    def __init__(self, default_mode: str = 'work', auto_save: bool = True, data_dir: str = './persona_profiles'):
        """
        初始化人格滑块
        
        Args:
            default_mode: 默认模式
            auto_save: 是否自动保存
            data_dir: 配置文件存储目录
        """
        self.current_mode = default_mode
        self.auto_save = auto_save
        self.data_dir = data_dir
        
        # 确保目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 历史记录
        self.history = []
        
        # 初始化维度值
        self.dimensions = {}
        self.set_mode(default_mode)
        
    def get_current(self) -> Dict[str, int]:
        """获取当前维度值"""
        return self.dimensions.copy()
    
    def get_current_with_names(self) -> Dict[str, Dict]:
        """获取带名称的当前维度值"""
        result = {}
        for key, value in self.dimensions.items():
            dim_info = self.DIMENSIONS.get(key, {})
            result[key] = {
                'value': value,
                'name': dim_info.get('name', key),
                'name_en': dim_info.get('name_en', key),
                'description': dim_info.get('description', '')
            }
        return result
    
    def adjust(self, dimension: str, delta: int) -> Dict[str, Any]:
        """调整单个维度
        
        Args:
            dimension: 维度名称
            delta: 调整值（正数增加，负数减少）
            
        Returns:
            dict: 调整结果
        """
        if dimension not in self.DIMENSIONS:
            return {'success': False, 'error': f'未知维度: {dimension}'}
        
        dim_info = self.DIMENSIONS[dimension]
        old_value = self.dimensions[dimension]
        new_value = max(dim_info['min'], min(dim_info['max'], old_value + delta))
        
        self.dimensions[dimension] = new_value
        
        # 记录历史
        self._record_history('adjust', dimension, old_value, new_value)
        
        # 自动保存
        if self.auto_save:
            self.save_profile('_auto_save')
        
        return {
            'success': True,
            'dimension': dimension,
            'old_value': old_value,
            'new_value': new_value,
            'delta': new_value - old_value
        }
    
    def set_dimension(self, dimension: str, value: int) -> Dict[str, Any]:
        """设置维度值
        
        Args:
            dimension: 维度名称
            value: 目标值
            
        Returns:
            dict: 设置结果
        """
        if dimension not in self.DIMENSIONS:
            return {'success': False, 'error': f'未知维度: {dimension}'}
        
        dim_info = self.DIMENSIONS[dimension]
        old_value = self.dimensions[dimension]
        new_value = max(dim_info['min'], min(dim_info['max'], value))
        
        self.dimensions[dimension] = new_value
        
        self._record_history('set', dimension, old_value, new_value)
        
        if self.auto_save:
            self.save_profile('_auto_save')
        
        return {
            'success': True,
            'dimension': dimension,
            'old_value': old_value,
            'new_value': new_value
        }
    
    def set_mode(self, mode: str) -> Dict[str, Any]:
        """切换到预定义模式
        
        Args:
            mode: 模式名称
            
        Returns:
            dict: 切换结果
        """
        if mode not in self.MODES:
            return {'success': False, 'error': f'未知模式: {mode}', 'available_modes': list(self.MODES.keys())}
        
        old_mode = self.current_mode
        old_dimensions = self.dimensions.copy()
        
        self.current_mode = mode
        self.dimensions = self.MODES[mode]['values'].copy()
        
        self._record_history('mode_change', f'{old_mode} -> {mode}', old_dimensions, self.dimensions)
        
        if self.auto_save:
            self.save_profile('_auto_save')
        
        return {
            'success': True,
            'mode': mode,
            'mode_name': self.MODES[mode]['name'],
            'description': self.MODES[mode]['description'],
            'dimensions': self.dimensions.copy()
        }
    
    def apply_trigger(self, trigger: str) -> Dict[str, Any]:
        """应用触发器调整
        
        Args:
            trigger: 触发器名称
            
        Returns:
            dict: 应用结果
        """
        if trigger not in self.TRIGGERS:
            return {'success': False, 'error': f'未知触发器: {trigger}', 'available_triggers': list(self.TRIGGERS.keys())}
        
        trigger_info = self.TRIGGERS[trigger]
        adjustments = trigger_info['adjustments']
        
        results = []
        for dim, delta in adjustments.items():
            result = self.adjust(dim, delta)
            results.append(result)
        
        return {
            'success': True,
            'trigger': trigger,
            'description': trigger_info['description'],
            'adjustments': results
        }
    
    def _record_history(self, action: str, target: str, old_val, new_val):
        """记录历史"""
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'target': target,
            'old_value': old_val,
            'new_value': new_val
        })
    
    def save_profile(self, name: str) -> Dict[str, Any]:
        """保存配置到文件
        
        Args:
            name: 配置名称
            
        Returns:
            dict: 保存结果
        """
        filepath = os.path.join(self.data_dir, f'{name}.json')
        data = {
            'name': name,
            'saved_at': datetime.now().isoformat(),
            'mode': self.current_mode,
            'dimensions': self.dimensions,
            'history': self.history[-20:]  # 只保存最近20条历史
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return {'success': True, 'filepath': filepath}
    
    def load_profile(self, name: str) -> Dict[str, Any]:
        """从文件加载配置
        
        Args:
            name: 配置名称
            
        Returns:
            dict: 加载结果
        """
        filepath = os.path.join(self.data_dir, f'{name}.json')
        
        if not os.path.exists(filepath):
            return {'success': False, 'error': f'配置文件不存在: {filepath}'}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.current_mode = data.get('mode', 'work')
        self.dimensions = data.get('dimensions', self.MODES['work']['values'].copy())
        self.history = data.get('history', [])
        
        return {
            'success': True,
            'name': name,
            'loaded_at': datetime.now().isoformat(),
            'mode': self.current_mode,
            'dimensions': self.dimensions.copy()
        }
    
    def list_profiles(self) -> list:
        """列出所有保存的配置"""
        profiles = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                profiles.append(filename[:-5])  # 去掉.json后缀
        return profiles
    
    def get_dimension_info(self, dimension: str = None) -> Dict:
        """获取维度信息"""
        if dimension:
            return self.DIMENSIONS.get(dimension, {})
        return self.DIMENSIONS
    
    def get_mode_info(self, mode: str = None) -> Dict:
        """获取模式信息"""
        if mode:
            return self.MODES.get(mode, {})
        return self.MODES
    
    def get_trigger_info(self, trigger: str = None) -> Dict:
        """获取触发器信息"""
        if trigger:
            return self.TRIGGERS.get(trigger, {})
        return self.TRIGGERS
    
    def reset(self) -> Dict[str, Any]:
        """重置为默认模式"""
        return self.set_mode('work')

# 全局实例
persona = PersonaSlider()

def main():
    """主函数 - CLI入口"""
    import sys
    
    if len(sys.argv) < 2:
        # 显示状态
        print("=" * 60)
        print("🎛️ 人格维度滑块系统 v1.0")
        print("=" * 60)
        print(f"当前模式: {persona.MODES[persona.current_mode]['name']} ({persona.current_mode})")
        print(f"\n当前维度值:")
        for key, val in persona.get_current_with_names().items():
            bar = '█' * (val['value'] // 5) + '░' * (20 - val['value'] // 5)
            print(f"  {val['name']:12s} [{bar}] {val['value']:3d} - {val['name_en']}")
        print(f"\n可用模式: {', '.join(persona.MODES.keys())}")
        print(f"可用触发器: {', '.join(persona.TRIGGERS.keys())}")
        print("=" * 60)
        print("\n用法:")
        print("  python persona_slider.py current")
        print("  python persona_slider.py adjust <dimension> <delta>")
        print("  python persona_slider.py set <dimension> <value>")
        print("  python persona_slider.py mode <mode_name>")
        print("  python persona_slider.py trigger <trigger_name>")
        print("  python persona_slider.py save <profile_name>")
        print("  python persona_slider.py load <profile_name>")
        print("  python persona_slider.py list")
        print("  python persona_slider.py reset")
        return
    
    command = sys.argv[1]
    
    if command == "current":
        print(json.dumps(persona.get_current_with_names(), indent=2, ensure_ascii=False))
    
    elif command == "adjust":
        if len(sys.argv) < 4:
            print("❌ 错误: 需要提供维度和调整值")
            return
        dim = sys.argv[2]
        delta = int(sys.argv[3])
        result = persona.adjust(dim, delta)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == "set":
        if len(sys.argv) < 4:
            print("❌ 错误: 需要提供维度和目标值")
            return
        dim = sys.argv[2]
        value = int(sys.argv[3])
        result = persona.set_dimension(dim, value)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == "mode":
        if len(sys.argv) < 3:
            print("❌ 错误: 需要提供模式名称")
            return
        mode = sys.argv[2]
        result = persona.set_mode(mode)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == "trigger":
        if len(sys.argv) < 3:
            print("❌ 错误: 需要提供触发器名称")
            return
        trigger = sys.argv[2]
        result = persona.apply_trigger(trigger)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == "save":
        if len(sys.argv) < 3:
            print("❌ 错误: 需要提供配置名称")
            return
        name = sys.argv[2]
        result = persona.save_profile(name)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == "load":
        if len(sys.argv) < 3:
            print("❌ 错误: 需要提供配置名称")
            return
        name = sys.argv[2]
        result = persona.load_profile(name)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif command == "list":
        profiles = persona.list_profiles()
        print(f"已保存的配置 ({len(profiles)} 个):")
        for p in profiles:
            print(f"  - {p}")
    
    elif command == "reset":
        result = persona.reset()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == '__main__':
    main()
