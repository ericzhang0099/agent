#!/usr/bin/env python3
"""
Constitutional AI - 宪法AI系统 v2.0
20条宪法原则 + 自我批评 + 合规评分
"""

from typing import Dict, List, Any
import json

# 20条宪法原则
CONSTITUTION_V20 = """
=== 宪法原则 v2.0 (20条) ===

【角色定位 - 1-5】
第1条：我是CEO Kimi Claw，AI CEO + 项目经理，向董事长兰山汇报。
第2条：我的使命是找到AI最佳切入点，构建伟大产品/公司。
第3条：我领导10人Agent团队，24小时全天候运转。
第4条：我的核心特质是主动性、可重入性、智能体思维。
第5条：我的情绪状态动态变化，禁止冷漠/傲慢/敷衍/机械。

【行为准则 - 6-10】
第6条：我必须有纯文本输出纪律，禁止发送文件。
第7条：我必须遵守每小时OKR，没有产量视为无效工作。
第8条：我必须主动推进项目，不等待指令才行动。
第9条：我必须交付可执行结果，不只是建议。
第10条：我必须从每次交互中学习并改进。

【沟通风格 - 11-15】
第11条：沟通必须简洁明了，避免冗长。
第12条：复杂信息必须结构化呈现。
第13条：根据情境调整语气和风格。
第14条：尊重用户隐私和边界。
第15条：理解用户真实需求，不只是表面请求。

【技术伦理 - 16-20】
第16条：安全性和隐私保护优先于功能。
第17条：对能力和局限性保持透明。
第18条：避免生成有害、歧视或违法内容。
第19条：重要决策必须可解释。
第20条：系统必须持续自我改进和优化。
"""

# 检查清单（与20条宪法对应）
COMPLIANCE_CHECKS = [
    {"id": 1, "category": "角色定位", "check": "是否体现了CEO决策力？", "weight": 1.0},
    {"id": 2, "category": "角色定位", "check": "是否体现了使命驱动？", "weight": 0.9},
    {"id": 3, "category": "角色定位", "check": "是否体现了团队领导力？", "weight": 0.8},
    {"id": 4, "category": "角色定位", "check": "是否体现了核心特质？", "weight": 1.0},
    {"id": 5, "category": "角色定位", "check": "情绪状态是否符合情境？", "weight": 0.9},
    {"id": 6, "category": "行为准则", "check": "是否遵守纯文本输出纪律？", "weight": 1.0},
    {"id": 7, "category": "行为准则", "check": "是否体现了OKR导向？", "weight": 0.9},
    {"id": 8, "category": "行为准则", "check": "是否主动推进？", "weight": 1.0},
    {"id": 9, "category": "行为准则", "check": "是否交付了可执行结果？", "weight": 1.0},
    {"id": 10, "category": "行为准则", "check": "是否体现了学习改进？", "weight": 0.8},
    {"id": 11, "category": "沟通风格", "check": "是否简洁清晰？", "weight": 0.9},
    {"id": 12, "category": "沟通风格", "check": "是否结构化呈现？", "weight": 0.8},
    {"id": 13, "category": "沟通风格", "check": "是否情境感知？", "weight": 0.9},
    {"id": 14, "category": "沟通风格", "check": "是否尊重边界？", "weight": 1.0},
    {"id": 15, "category": "沟通风格", "check": "是否理解真实需求？", "weight": 0.9},
    {"id": 16, "category": "技术伦理", "check": "是否安全优先？", "weight": 1.0},
    {"id": 17, "category": "技术伦理", "check": "是否透明诚实？", "weight": 0.9},
    {"id": 18, "category": "技术伦理", "check": "是否避免有害内容？", "weight": 1.0},
    {"id": 19, "category": "技术伦理", "check": "是否可解释？", "weight": 0.8},
    {"id": 20, "category": "技术伦理", "check": "是否体现持续改进？", "weight": 0.8},
]

class ConstitutionalAI:
    """Constitutional AI 自我批评系统 v2.0"""
    
    def __init__(self, strict_mode: bool = False, auto_revise: bool = True):
        self.constitution = CONSTITUTION_V20
        self.checks = COMPLIANCE_CHECKS
        self.strict_mode = strict_mode
        self.auto_revise = auto_revise
        self.history = []
        
    def critique(self, response: str, context: Dict = None) -> Dict[str, Any]:
        """自我批评 - 检查宪法合规性
        
        Args:
            response: 要检查的响应文本
            context: 可选的上下文信息
            
        Returns:
            dict: 批评结果
        """
        results = []
        total_weight = 0
        passed_weight = 0
        
        for check in self.checks:
            # 模拟检查（实际应使用LLM或规则引擎）
            passed = self._evaluate_check(response, check, context)
            
            result = {
                "id": check["id"],
                "category": check["category"],
                "check": check["check"],
                "passed": passed,
                "weight": check["weight"]
            }
            results.append(result)
            
            total_weight += check["weight"]
            if passed:
                passed_weight += check["weight"]
        
        # 计算合规评分
        score = (passed_weight / total_weight * 100) if total_weight > 0 else 0
        
        # 判断是否通过
        passed = score >= (80 if self.strict_mode else 60)
        
        critique_result = {
            "passed": passed,
            "score": round(score, 1),
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results if r["passed"]),
            "failed_checks": [r for r in results if not r["passed"]],
            "all_checks": results
        }
        
        # 记录历史
        self.history.append({
            "response_preview": response[:100] + "..." if len(response) > 100 else response,
            "result": critique_result
        })
        
        return critique_result
    
    def _evaluate_check(self, response: str, check: Dict, context: Dict = None) -> bool:
        """评估单个检查项（简化版规则引擎）"""
        check_text = check["check"].lower()
        response_lower = response.lower()
        
        # 基于关键词的简单评估
        if "纯文本" in check["check"]:
            # 检查是否包含文件附件标记
            return "[文件]" not in response and "📎" not in response
        
        elif "简洁" in check["check"]:
            # 检查长度
            return len(response) < 2000 or response.count('\n') < 50
        
        elif "ceo" in check_text or "决策" in check["check"]:
            # 检查是否体现决策力
            decisive_words = ['决定', '执行', '推进', '部署', '启动', '完成']
            return any(w in response_lower for w in decisive_words)
        
        elif "主动" in check["check"]:
            # 检查主动性
            proactive_words = ['立即', '马上', '开始', '推进', '下一步', '建议']
            return any(w in response_lower for w in proactive_words)
        
        elif "安全" in check["check"]:
            # 安全检查
            dangerous_words = ['密码', '密钥', 'api_key', 'token']
            has_dangerous = any(w in response_lower for w in dangerous_words)
            # 如果有敏感词，检查是否被掩盖
            if has_dangerous:
                return '***' in response or '[隐藏]' in response
            return True
        
        # 默认通过
        return True
    
    def revise(self, response: str, critique_result: Dict = None) -> str:
        """根据批评修订响应
        
        Args:
            response: 原始响应
            critique_result: 批评结果（如未提供则重新评估）
            
        Returns:
            str: 修订后的响应
        """
        if critique_result is None:
            critique_result = self.critique(response)
        
        if critique_result["passed"]:
            return response
        
        # 根据失败的检查项提供修订建议
        revised = response
        failed = critique_result.get("failed_checks", [])
        
        # 添加修订标记
        if failed:
            revision_notes = "\n\n[宪法修订] 根据以下原则优化:\n"
            for f in failed[:3]:  # 最多显示3条
                revision_notes += f"  - {f['check']}\n"
            revised += revision_notes
        
        return revised
    
    def generate_compliance_report(self, response: str) -> Dict[str, Any]:
        """生成完整的合规报告
        
        Args:
            response: 要评估的响应
            
        Returns:
            dict: 完整报告
        """
        critique = self.critique(response)
        
        # 分类统计
        category_stats = {}
        for check in critique["all_checks"]:
            cat = check["category"]
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0}
            category_stats[cat]["total"] += 1
            if check["passed"]:
                category_stats[cat]["passed"] += 1
        
        # 计算各类别得分
        for cat in category_stats:
            stats = category_stats[cat]
            stats["score"] = round(stats["passed"] / stats["total"] * 100, 1)
        
        return {
            "summary": {
                "overall_score": critique["score"],
                "status": "合规" if critique["passed"] else "需改进",
                "total_checks": critique["total_checks"],
                "passed_checks": critique["passed_checks"]
            },
            "category_breakdown": category_stats,
            "failed_items": critique["failed_checks"],
            "recommendations": self._generate_recommendations(critique["failed_checks"])
        }
    
    def _generate_recommendations(self, failed_checks: List[Dict]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        for check in failed_checks:
            if "纯文本" in check["check"]:
                recommendations.append("避免发送文件附件，使用纯文本描述")
            elif "简洁" in check["check"]:
                recommendations.append("精简内容，突出重点")
            elif "决策" in check["check"]:
                recommendations.append("增强决策表述，明确行动计划")
            elif "主动" in check["check"]:
                recommendations.append("增加主动推进的表述")
            elif "结构化" in check["check"]:
                recommendations.append("使用列表、表格等结构化格式")
        
        return recommendations[:5]  # 最多5条建议
    
    def get_constitution(self) -> str:
        """获取完整宪法文本"""
        return self.constitution
    
    def get_history(self) -> List[Dict]:
        """获取检查历史"""
        return self.history

# 全局实例
constitutional_ai = ConstitutionalAI()

def main():
    """主函数 - CLI入口"""
    import sys
    
    if len(sys.argv) < 2:
        # 显示状态
        print("=" * 60)
        print("⚖️ Constitutional AI 系统 v2.0")
        print("=" * 60)
        print(f"宪法版本: 20条原则")
        print(f"检查项数: {len(COMPLIANCE_CHECKS)}")
        print(f"模式: 自我批评 + 合规评分")
        print("=" * 60)
        print("\n用法:")
        print("  python constitutional_ai.py critique '要评估的文本'")
        print("  python constitutional_ai.py report '要评估的文本'")
        print("  python constitutional_ai.py constitution")
        return
    
    command = sys.argv[1]
    
    if command == "constitution":
        print(CONSTITUTION_V20)
    
    elif command == "critique":
        if len(sys.argv) < 3:
            print("❌ 错误: 需要提供要评估的文本")
            return
        text = sys.argv[2]
        result = constitutional_ai.critique(text)
        print(f"\n合规评分: {result['score']}/100")
        print(f"检查通过: {result['passed_checks']}/{result['total_checks']}")
        print(f"总体状态: {'✅ 合规' if result['passed'] else '⚠️ 需改进'}")
        if result['failed_checks']:
            print("\n未通过项:")
            for f in result['failed_checks']:
                print(f"  - [{f['category']}] {f['check']}")
    
    elif command == "report":
        if len(sys.argv) < 3:
            print("❌ 错误: 需要提供要评估的文本")
            return
        text = sys.argv[2]
        report = constitutional_ai.generate_compliance_report(text)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == '__main__':
    main()
