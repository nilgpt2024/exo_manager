"""
EXO Manager - 系统日志收集器
============================

内存中的系统事件日志收集器，供管理端页面展示。
收集关键事件：模型加载/卸载、节点上下线、故障恢复、智能分配决策等。

使用方式:
    from sys_logger import sys_log
    sys_log.log(sys_log.INFO, "category", "消息内容", {"key": "value"})
    logs = sys_log.get_logs(level="warning", limit=50)
"""

import time
import uuid
from typing import Optional, Dict, List, Any


class SystemLogCollector:
    """
    内存中的系统事件日志收集器

    收集关键系统事件（模型加载/卸载、节点上下线、故障恢复等），
    通过 API 暴露给前端展示。

    日志级别:
        info     - 一般信息（模型预览、分配方案）
        warning  - 警告（安全余量不足、节点异常）
        error    - 错误（节点掉线、加载失败）
        success  - 成功（模型加载完成、故障恢复成功）
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"

    MAX_LOGS = 500

    _instance = None
    _logs: List[Dict] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True

    def log(self, level: str, category: str, message: str, detail: Dict = None):
        """记录一条系统日志"""
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.strftime("%H:%M:%S"),
            "full_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "category": category,
            "message": message,
            "detail": detail or {}
        }
        self._logs.append(entry)

        if len(self._logs) > self.MAX_LOGS:
            self._logs = self._logs[-self.MAX_LOGS:]

    def get_logs(
        self,
        level: str = None,
        category: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict:
        """查询日志（返回最新的在前）"""
        result = list(self._logs)

        if level:
            result = [l for l in result if l["level"] == level]
        if category:
            result = [l for l in result if l["category"] == category]

        total = len(result)
        page = result[offset:offset + limit]

        return {
            "total": total,
            "logs": page[::-1],  # 最新的在前
            "categories": sorted(set(l["category"] for l in self._logs))
        }

    def clear(self):
        """清空日志"""
        self._logs.clear()


# 全局单例
sys_log = SystemLogCollector()
