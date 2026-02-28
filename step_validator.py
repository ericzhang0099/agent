
class StepValidator:
    """分步验证器 - 确保每步正确"""
    
    def __init__(self, task_name):
        self.task_name = task_name
        self.steps = []
        self.current_step = 0
    
    def add_step(self, name, validator_func):
        """添加验证步骤"""
        self.steps.append({
            "name": name,
            "validator": validator_func,
            "status": "pending"
        })
    
    def execute(self, context):
        """执行所有步骤"""
        for i, step in enumerate(self.steps):
            self.current_step = i
            print(f"步骤 {i+1}/{len(self.steps)}: {step['name']}")
            
            try:
                result = step["validator"](context)
                step["status"] = "passed"
                print(f"  ✓ 通过")
            except Exception as e:
                step["status"] = "failed"
                print(f"  ✗ 失败: {e}")
                raise
        
        print(f"✅ 所有 {len(self.steps)} 个步骤验证通过")
        return True
