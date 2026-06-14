"""
EXO Cluster Manager - 安全审计日志模块
========================================

功能:
- 记录关键安全事件 (登录/登出/权限变更)
- 记录敏感操作 (API Key 管理/用户管理)
- 记录异常行为 (多次失败登录/速率限制触发)
- 支持结构化日志输出 (JSON 格式)
- 自动轮转和归档

日志级别:
- CRITICAL: 安全事件 (未授权访问尝试, 权限提升)
- ERROR: 异常行为 (暴力破解迹象)
- WARNING: 可疑活动 (频繁失败)
- INFO: 关键操作 (登录成功, 密码修改)

配置环境变量:
- EXO_AUDIT_ENABLED: 是否启用 (默认 true)
- EXO_AUDIT_LOG_FILE: 日志文件路径
- EXO_AUDIT_MAX_SIZE: 单个日志文件最大大小 (MB, 默认 10)
- EXO_AUDIT_BACKUP_COUNT: 保留的备份文件数 (默认 5)
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field
from logging.handlers import RotatingFileHandler
import threading

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """审计事件数据模型"""
    event_id: str = ""
    timestamp: float = 0.0
    level: str = "INFO"  # CRITICAL / ERROR / WARNING / INFO
    category: str = ""   # auth / api_key / user_admin / system / security
    action: str = ""
    user_id: str = ""
    user_role: str = ""
    ip_address: str = ""
    user_agent: str = ""
    resource: str = ""
    resource_id: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""
    session_token: str = ""  # 仅存储前8位用于关联

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            **asdict(self),
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
        }

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class AuditLogger:
    """
    安全审计日志管理器

    特性:
    - 结构化 JSON 日志格式
    - 自动文件轮转
    - 线程安全
    - 多级日志分类
    - 敏感信息自动脱敏
    """

    # 实例 (单例模式)
    _instance: Optional["AuditLogger"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        log_file_path: Optional[str] = None,
        enabled: Optional[bool] = None,
        max_size_mb: int = 10,
        backup_count: int = 5
    ):
        # 避免重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            return

        # 从环境变量读取配置
        self.enabled = enabled if enabled is not None else \
            os.getenv("EXO_AUDIT_ENABLED", "true").lower() in ("true", "1", "yes")

        if log_file_path is None:
            base_dir = Path(__file__).parent / "logs"
            base_dir.mkdir(exist_ok=True)
            log_file_path = str(base_dir / "audit.log")

        self.log_file_path = Path(log_file_path)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.backup_count = backup_count

        # 内存缓存 (用于快速查询最近的事件)
        self._recent_events: List[AuditEvent] = []
        self._max_recent_events = 1000
        self._cache_lock = threading.Lock()

        # 初始化日志处理器
        self._setup_logger()

        self._initialized = True

        if self.enabled:
            logger.info(f"✅ 安全审计日志已启用 (文件: {self.log_file_path})")
        else:
            logger.warning("⚠️ 安全审计日志已禁用")

    def _setup_logger(self):
        """配置日志处理器"""
        self.audit_log = logging.getLogger("exo_audit")
        self.audit_log.setLevel(logging.DEBUG)
        self.audit_log.propagate = False  # 防止传播到根日志

        # 清除已有处理器 (避免重复添加)
        self.audit_log.handlers.clear()

        # 文件处理器 (JSON 格式 + 轮转)
        try:
            file_handler = RotatingFileHandler(
                filename=str(self.log_file_path),
                maxBytes=self.max_size_bytes,
                backupCount=self.backup_count,
                encoding='utf-8'
            )

            # 自定义 JSON 格式化器
            class JsonFormatter(logging.Formatter):
                def format(self, record):
                    if hasattr(record, 'audit_event'):
                        return record.audit_event.to_json()
                    return json.dumps({
                        "level": record.levelname,
                        "message": record.getMessage(),
                        "timestamp": datetime.now().isoformat()
                    })

            file_handler.setFormatter(JsonFormatter())
            self.audit_log.addHandler(file_handler)

        except Exception as e:
            logger.error(f"初始化审计日志文件失败: {e}")
            self.enabled = False

    @staticmethod
    def get_instance() -> "AuditLogger":
        """获取单例实例"""
        if AuditLogger._instance is None:
            AuditLogger()
        return AuditLogger._instance

    def _sanitize_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        脱敏敏感信息

        处理字段:
        - password: 完全移除
        - token / session_token: 仅保留前8位
        - api_key: 仅保留前12位
        - app_secret: 完全移除
        """
        sensitive_keys = [
            'password', 'passwd', 'pwd',
            'app_secret', 'secret_key', 'private_key'
        ]

        sanitized = data.copy()
        for key in list(sanitized.keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif 'token' in key.lower() or 'session' in key.lower():
                value = str(sanitized[key])
                sanitized[key] = f"{value[:8]}..." if len(value) > 8 else "***"
            elif 'api_key' in key.lower() or 'key' in key.lower():
                value = str(sanitized[key])
                sanitized[key] = f"{value[:12]}..." if len(value) > 12 else "***"

        return sanitized

    def log(
        self,
        level: str,
        category: str,
        action: str,
        user_id: str = "",
        user_role: str = "",
        ip_address: str = "",
        user_agent: str = "",
        resource: str = "",
        resource_id: str = "",
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: str = "",
        session_token: str = ""
    ):
        """
        记录审计事件

        Args:
            level: 日志级别 (CRITICAL / ERROR / WARNING / INFO)
            category: 事件类别 (auth / api_key / user_admin / system / security)
            action: 操作描述
            user_id: 用户ID
            user_role: 用户角色
            ip_address: 客户端IP
            user_agent: 用户代理字符串
            resource: 资源类型
            resource_id: 资源ID
            details: 附加详情 (会自动脱敏)
            success: 是否成功
            error_message: 错误信息
            session_token: 会话令牌 (仅存储前8位)
        """
        if not self.enabled:
            return

        # 创建事件对象
        event = AuditEvent(
            event_id=uuid.uuid4().hex[:16],
            timestamp=time.time(),
            level=level.upper(),
            category=category,
            action=action,
            user_id=user_id,
            user_role=user_role,
            ip_address=ip_address,
            user_agent=user_agent[:200] if user_agent else "",  # 截断过长的UA
            resource=resource,
            resource_id=resource_id,
            details=self._sanitize_sensitive_data(details or {}),
            success=success,
            error_message=error_message,
            session_token=session_token[:8] if session_token else "",
        )

        # 写入日志文件
        log_level = getattr(logging, event.level.upper(), logging.INFO)
        audit_record = logging.LogRecord(
            name="exo_audit",
            level=log_level,
            pathname="",
            lineno=0,
            msg=event.action,
            args=(),
            exc_info=None
        )
        audit_record.audit_event = event
        self.audit_log.handle(audit_record)

        # 添加到内存缓存
        with self._cache_lock:
            self._recent_events.append(event)
            if len(self._recent_events) > self._max_recent_events:
                self._recent_events = self._recent_events[-self._max_recent_events:]

        # 同时输出到标准日志 (便于实时监控)
        log_func = getattr(logger, event.level.lower(), logger.info)
        prefix = "✅" if success else "❌"
        log_func(
            f"{prefix} [Audit] [{event.category}] {event.action} "
            f"(user={event.user_id or 'anonymous'}, IP={event.ip_address})"
        )

    # ==================== 便捷方法 ====================

    def log_auth_login_success(self, user_id: str, role: str, ip: str, method: str = "password"):
        """记录登录成功"""
        self.log(
            level="INFO",
            category="auth",
            action=f"Login success ({method})",
            user_id=user_id,
            user_role=role,
            ip_address=ip,
            details={"login_method": method}
        )

    def log_auth_login_failure(self, username: str, ip: str, reason: str):
        """记录登录失败"""
        self.log(
            level="WARNING",
            category="auth",
            action="Login failed",
            ip_address=ip,
            details={"username": username, "reason": reason},
            success=False,
            error_message=reason
        )

    def log_auth_logout(self, user_id: str, ip: str):
        """记录登出"""
        self.log(
            level="INFO",
            category="auth",
            action="Logout",
            user_id=user_id,
            ip_address=ip
        )

    def log_auth_password_change(self, user_id: str, ip: str):
        """记录密码修改"""
        self.log(
            level="INFO",
            category="auth",
            action="Password changed",
            user_id=user_id,
            ip_address=ip
        )

    def log_api_key_created(self, user_id: str, key_name: str, ip: str):
        """记录 API Key 创建"""
        self.log(
            level="INFO",
            category="api_key",
            action="API Key created",
            user_id=user_id,
            ip_address=ip,
            resource="api_key",
            details={"key_name": key_name}
        )

    def log_api_key_revoked(self, user_id: str, key_prefix: str, revoked_by: str, ip: str):
        """记录 API Key 吊销"""
        self.log(
            level="WARNING",
            category="api_key",
            action="API Key revoked",
            user_id=user_id,
            ip_address=ip,
            resource="api_key",
            resource_id=key_prefix,
            details={"revoked_by": revoked_by}
        )

    def log_user_role_change(self, target_user_id: str, old_role: str, new_role: str, operator_id: str, ip: str):
        """记录角色变更"""
        self.log(
            level="WARNING",
            category="user_admin",
            action=f"User role changed: {old_role} → {new_role}",
            user_id=target_user_id,
            ip_address=ip,
            resource="user",
            resource_id=target_user_id,
            details={
                "old_role": old_role,
                "new_role": new_role,
                "operator_id": operator_id
            }
        )

    def log_user_disabled(self, target_user_id: str, operator_id: str, ip: str, reason: str = ""):
        """记录用户禁用"""
        self.log(
            level="WARNING",
            category="user_admin",
            action="User disabled",
            user_id=target_user_id,
            ip_address=ip,
            resource="user",
            resource_id=target_user_id,
            details={"operator_id": operator_id, "reason": reason}
        )

    def log_security_event(self, event_type: str, ip: str, details: Dict, severity: str = "ERROR"):
        """记录安全事件"""
        self.log(
            level=severity,
            category="security",
            action=event_type,
            ip_address=ip,
            details=details,
            success=False
        )

    def log_rate_limit_exceeded(self, ip: str, path: str, limit_type: str):
        """记录速率限制触发"""
        self.log(
            level="WARNING",
            category="security",
            action="Rate limit exceeded",
            ip_address=ip,
            details={
                "path": path,
                "limit_type": limit_type
            },
            success=False
        )

    # ==================== 查询方法 ====================

    def get_recent_events(
        self,
        limit: int = 50,
        level: Optional[str] = None,
        category: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict]:
        """
        查询最近的审计事件

        Args:
            limit: 返回的最大数量
            level: 过滤日志级别
            category: 过滤事件类别
            user_id: 过滤用户ID

        Returns:
            事件列表 (字典格式)
        """
        with self._cache_lock:
            events = list(self._recent_events)

        # 应用过滤器
        if level:
            events = [e for e in events if e.level == level.upper()]
        if category:
            events = [e for e in events if e.category == category]
        if user_id:
            events = [e for e in events if e.user_id == user_id]

        # 返回最新的 N 条
        events = events[-limit:]
        return [e.to_dict() for e in reversed(events)]

    def get_security_summary(self, hours: int = 24) -> Dict:
        """
        获取安全摘要统计

        Args:
            hours: 统计时间范围 (小时)

        Returns:
            统计数据字典
        """
        cutoff_time = time.time() - (hours * 3600)

        with self._cache_lock:
            recent = [e for e in self._recent_events if e.timestamp >= cutoff_time]

        total = len(recent)
        failures = sum(1 for e in recent if not e.success)
        security_events = sum(1 for e in recent if e.category == "security")
        auth_failures = sum(1 for e in recent if e.category == "auth" and not e.success)

        # 统计最活跃的 IP
        ip_counts: Dict[str, int] = {}
        for e in recent:
            ip = e.ip_address or "unknown"
            ip_counts[ip] = ip_counts.get(ip, 0) + 1
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "time_range_hours": hours,
            "total_events": total,
            "failed_events": failures,
            "security_events": security_events,
            "auth_failures": auth_failures,
            "success_rate": round((total - failures) / total * 100, 1) if total > 0 else 0,
            "top_active_ips": [{"ip": ip, "count": count} for ip, count in top_ips],
            "generated_at": datetime.now().isoformat(),
        }


# 全局实例 (延迟初始化)
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """获取全局审计日志器实例"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def init_audit_logger(**kwargs) -> AuditLogger:
    """显式初始化审计日志器"""
    global _audit_logger
    _audit_logger = AuditLogger(**kwargs)
    return _audit_logger
