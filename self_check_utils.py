
def self_check(func):
    """函数自检装饰器"""
    def wrapper(*args, **kwargs):
        print(f"▶️  执行: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            print(f"✅ 成功: {func.__name__}")
            return result
        except Exception as e:
            print(f"❌ 失败: {func.__name__} - {e}")
            raise
    return wrapper

def validate_output(output, expected_type=None, validator=None):
    """输出验证"""
    if expected_type and not isinstance(output, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(output)}")
    
    if validator and not validator(output):
        raise ValueError("Output validation failed")
    
    return True
