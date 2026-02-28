
#!/usr/bin/env python3
"""
任务管理器 - 防止超时、提高成功率
"""

import time
import signal
from datetime import datetime
from pathlib import Path

class TaskManager:
    """智能任务管理器"""
    
    def __init__(self, default_timeout=300):
        self.default_timeout = default_timeout
        self.active_tasks = {}
        self.completed_tasks = []
    
    def run_with_timeout(self, func, args=(), kwargs=None, timeout=None):
        """带超时的任务执行"""
        timeout = timeout or self.default_timeout
        
        # 设置超时信号
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Task exceeded {timeout} seconds")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            start = time.time()
            result = func(*args, **(kwargs or {}))
            elapsed = time.time() - start
            
            signal.alarm(0)  # 取消超时
            
            return {
                "success": True,
                "result": result,
                "elapsed": elapsed
            }
            
        except TimeoutError as e:
            return {
                "success": False,
                "error": str(e),
                "timeout": True
            }
        except Exception as e:
            signal.alarm(0)
            return {
                "success": False,
                "error": str(e),
                "timeout": False
            }
    
    def chunked_execution(self, items, processor, chunk_size=10):
        """分块执行，避免超时"""
        results = []
        total = len(items)
        
        for i in range(0, total, chunk_size):
            chunk = items[i:i+chunk_size]
            print(f"处理块 {i//chunk_size + 1}/{(total-1)//chunk_size + 1}")
            
            for item in chunk:
                try:
                    result = processor(item)
                    results.append({"item": item, "result": result, "success": True})
                except Exception as e:
                    results.append({"item": item, "error": str(e), "success": False})
            
            # 保存检查点
            self._save_checkpoint(results)
        
        return results
    
    def _save_checkpoint(self, data):
        """保存进度检查点"""
        checkpoint_file = Path(f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        import json
        with open(checkpoint_file, "w") as f:
            json.dump(data, f)
