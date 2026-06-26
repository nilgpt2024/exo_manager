"""
EXO Manager - 系统日志收集器
============================

内存中的系统事件日志收集器，供管理端页面展示。
收集关键事件：模型加载/卸载、节点上下线、故障恢复、智能分配决策等。

使用方式:
    from sys_logger import sys_log, setup_logging
    setup_logging()
    sys_log.log(sys_log.INFO, "category", "消息内容", {"key": "value"})
    logs = sys_log.get_logs(level="warning", limit=50)
"""

import time
import uuid
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, List, Any


def setup_logging(log_level: str = None, log_dir: str = None):
    """
    初始化全局日志配置

    配置两个处理器：
    1. StreamHandler: 输出到控制台（stdout）
    2. RotatingFileHandler: 输出到文件，自动轮转

    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                   默认从环境变量 EXO_LOG_LEVEL 读取，未设置则为 INFO
        log_dir: 日志文件目录，默认使用项目根目录下的 logs/
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level_str = log_level or os.getenv("EXO_LOG_LEVEL", "INFO").upper()
    log_level = level_map.get(log_level_str, logging.INFO)

    if log_dir is None:
        log_dir = str(Path(__file__).parent / "logs")

    os.makedirs(log_dir, exist_ok=True)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    log_file_path = os.path.join(log_dir, "exo_manager.log")
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.info(f"📝 日志系统已初始化 (级别: {log_level_str}, 文件: {log_file_path})")


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
            "logs": page[::-1],
            "categories": sorted(set(l["category"] for l in self._logs))
        }

    def clear(self):
        """清空日志"""
        self._logs.clear()


sys_log = SystemLogCollector()
