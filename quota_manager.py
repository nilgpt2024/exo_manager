"""
EXO Cluster Manager - Token 额度管理模块 (SQLite 版)
====================================================

使用 SQLite 数据库存储用户 Token 额度数据，支持：
- 并发安全的事务操作
- 原子性读写
- 使用历史记录追踪
- 管理员充值/调整额度

数据库表结构:
- users_quota: 用户额度主表
- usage_history: 使用历史记录表
"""

import sqlite3
import json
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class QuotaInfo:
    """用户额度信息"""
    user_id: str
    total_quota: int = 100000
    used_tokens: int = 0
    remaining: int = 100000
    is_unlimited: bool = False
    last_usage: float = 0
    created_at: float = 0

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "total_quota": self.total_quota,
            "used_tokens": self.used_tokens,
            "remaining": self.remaining if not self.is_unlimited else -1,
            "is_unlimited": self.is_unlimited,
            "usage_percent": round(self.used_tokens / self.total_quota * 100, 1) if self.total_quota > 0 and not self.is_unlimited else 0,
        }


@dataclass
class UsageRecord:
    """使用记录"""
    id: int = 0
    timestamp: float = 0
    tokens: int = 0
    model: str = ""
    request_type: str = ""


class QuotaManager:
    """
    Token 额度管理器 (SQLite 版)

    特点:
    - 线程安全 (使用连接池 + 锁)
    - 事务支持 (原子性操作)
    - 自动重连 (处理连接断开)
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path(__file__).parent
            db_path = str(base_dir / "quota.db")

        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._local = threading.local()

        # 配置
        self._config = {
            "default_quota": 100_000,
            "admin_unlimited": True,
            "warn_threshold": 0.8,
            "max_history_records": 500,
        }

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _close_conn(self):
        """关闭当前线程的连接"""
        if hasattr(self._local, 'conn') and self._local.conn:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None

    def _init_db(self):
        """初始化数据库表结构"""
        conn = self._get_conn()

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users_quota (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                total_quota INTEGER DEFAULT 100000,
                used_tokens INTEGER DEFAULT 0,
                remaining INTEGER DEFAULT 100000,
                is_unlimited INTEGER DEFAULT 0,
                last_usage REAL DEFAULT 0,
                created_at REAL DEFAULT 0,
                updated_at REAL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_user_id ON users_quota(user_id);

            CREATE TABLE IF NOT EXISTS usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                tokens INTEGER NOT NULL,
                model TEXT DEFAULT '',
                request_type TEXT DEFAULT '',
                created_at REAL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users_quota(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_history_user ON usage_history(user_id);
            CREATE INDEX IF NOT EXISTS idx_history_time ON usage_history(created_at DESC);

            CREATE TABLE IF NOT EXISTS api_call_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                api_key TEXT DEFAULT '',
                model_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0,
                status TEXT DEFAULT 'success',
                error_message TEXT DEFAULT '',
                request_id TEXT DEFAULT '',
                created_at REAL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_api_stats_user ON api_call_stats(user_id);
            CREATE INDEX IF NOT EXISTS idx_api_stats_time ON api_call_stats(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_api_stats_model ON api_call_stats(model_id);

            CREATE TABLE IF NOT EXISTS quota_config (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)

        # 初始化默认配置
        defaults = [
            ("default_quota", str(self._config["default_quota"])),
            ("admin_unlimited", str(int(self._config["admin_unlimited"]))),
            ("warn_threshold", str(self._config["warn_threshold"])),
            ("max_history_records", str(self._config["max_history_records"])),
        ]

        for key, value in defaults:
            conn.execute(
                "INSERT OR IGNORE INTO quota_config (key, value) VALUES (?, ?)",
                (key, value)
            )

        conn.commit()
        logger.info(f"SQLite 额度数据库已初始化: {self.db_path}")

    # ==================== 公共接口 ====================

    def get_quota(self, user_id: str) -> QuotaInfo:
        """获取用户额度信息"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM users_quota WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if not row:
            return self._create_default_quota(user_id)

        return QuotaInfo(
            user_id=row["user_id"],
            total_quota=row["total_quota"],
            used_tokens=row["used_tokens"],
            remaining=row["remaining"],
            is_unlimited=bool(row["is_unlimited"]),
            last_usage=row["last_usage"] or 0,
            created_at=row["created_at"] or 0,
        )

    def _create_default_quota(self, user_id: str, is_admin: bool = False) -> QuotaInfo:
        """创建默认额度记录"""
        now = time.time()
        default = int(self._get_config_value("default_quota", "100000"))
        admin_unlimited = bool(int(self._get_config_value("admin_unlimited", "1")))

        conn = self._get_conn()
        with self._lock:
            conn.execute("""
                INSERT OR IGNORE INTO users_quota 
                (user_id, total_quota, used_tokens, remaining, is_unlimited, last_usage, created_at, updated_at)
                VALUES (?, ?, 0, ?, ?, ?, ?, ?)
            """, (
                user_id,
                default if not is_admin else 0,
                default if not is_admin else -1,
                1 if (is_admin and admin_unlimited) else 0,
                now,
                now,
                now
            ))
            conn.commit()

        logger.info(f"创建用户额度记录: {user_id} (admin={is_admin})")
        return self.get_quota(user_id)

    def check_quota(self, user_id: str, requested_tokens: int = 0) -> Tuple[bool, str, int]:
        """检查用户是否有足够额度"""
        quota = self.get_quota(user_id)

        if quota.is_unlimited:
            return True, "unlimited", -1

        remaining = quota.remaining
        used = quota.used_tokens
        total = quota.total_quota

        if remaining <= 0:
            return False, f"额度已用完 ({used}/{total} tokens)", 0

        if requested_tokens > 0 and remaining < requested_tokens:
            return (
                False,
                f"额度不足 (剩余 {remaining} tokens，需要 {requested_tokens} tokens)",
                remaining
            )

        threshold = float(self._get_config_value("warn_threshold", "0.8"))
        usage_ratio = used / total if total > 0 else 0

        if usage_ratio >= threshold:
            return True, f"额度即将用尽 ({usage_ratio*100:.0f}% 已使用)", remaining

        return True, "ok", remaining

    def deduct_tokens(
        self,
        user_id: str,
        tokens: int,
        model: str = "",
        request_type: str = "inference"
    ) -> Tuple[bool, int]:
        """扣减用户额度（原子操作）"""
        if tokens <= 0:
            return True, self.get_quota(user_id).remaining

        conn = self._get_conn()
        now = time.time()

        with self._lock:
            try:
                # 使用 BEGIN IMMEDIATE 获取排他锁
                conn.execute("BEGIN IMMEDIATE")

                row = conn.execute(
                    "SELECT * FROM users_quota WHERE user_id = ?",
                    (user_id,)
                ).fetchone()

                if not row:
                    self._create_default_quota(user_id)
                    row = conn.execute(
                        "SELECT * FROM users_quota WHERE user_id = ?", (user_id,)
                    ).fetchone()

                if not row:
                    return False, 0

                is_unlimited = bool(row["is_unlimited"])
                remaining = row["remaining"]

                if is_unlimited:
                    # 无限额度只记录，不扣减
                    conn.execute(
                        "INSERT INTO usage_history (user_id, tokens, model, request_type, created_at) VALUES (?, ?, ?, ?, ?)",
                        (user_id, tokens, model, request_type, now)
                    )
                    conn.execute(
                        "UPDATE users_quota SET last_usage = ?, updated_at = ? WHERE user_id = ?",
                        (now, now, user_id)
                    )
                    conn.commit()
                    return True, -1

                if remaining < tokens:
                    logger.warning(f"用户 {user_id} 额度不足: 需要 {tokens}, 剩余 {remaining}")
                    return False, remaining

                new_used = row["used_tokens"] + tokens
                new_remaining = remaining - tokens

                conn.execute("""
                    UPDATE users_quota 
                    SET used_tokens = ?, remaining = ?, last_usage = ?, updated_at = ?
                    WHERE user_id = ?
                """, (new_used, new_remaining, now, now, user_id))

                conn.execute(
                    "INSERT INTO usage_history (user_id, tokens, model, request_type, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, tokens, model, request_type, now)
                )
                conn.commit()

                logger.info(f"用户 {user_id} 扣减 {tokens} tokens (剩余 {new_remaining})")
                return True, new_remaining

            except Exception as e:
                conn.rollback()
                logger.error(f"扣减额度失败: {e}")
                raise

    def add_quota(self, user_id: str, amount: int, reason: str = "") -> Tuple[bool, int]:
        """为用户增加额度（管理员操作）"""
        if amount <= 0:
            return False, self.get_quota(user_id).remaining

        conn = self._get_conn()
        now = time.time()

        with self._lock:
            row = conn.execute(
                "SELECT * FROM users_quota WHERE user_id = ?", (user_id,)
            ).fetchone()

            if not row:
                self._create_default_quota(user_id)
                row = conn.execute(
                    "SELECT * FROM users_quota WHERE user_id = ?", (user_id,)
                ).fetchone()

            if not row or bool(row["is_unlimited"]):
                return True, -1

            new_total = row["total_quota"] + amount
            new_remaining = row["remaining"] + amount

            conn.execute("""
                UPDATE users_quota 
                SET total_quota = ?, remaining = ?, updated_at = ?
                WHERE user_id = ?
            """, (new_total, new_remaining, now, user_id))

            conn.execute(
                "INSERT INTO usage_history (user_id, tokens, model, request_type, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, -amount, "", f"recharge:{reason}", now)
            )
            conn.commit()

            logger.info(f"管理员为用户 {user_id} 增加 {amount} tokens")
            return True, new_remaining

    def set_quota(self, user_id: str, new_total: int) -> bool:
        """设置用户总额度"""
        if new_total < 0:
            return False

        conn = self._get_conn()
        now = time.time()

        with self._lock:
            row = conn.execute(
                "SELECT * FROM users_quota WHERE user_id = ?", (user_id,)
            ).fetchone()

            if not row:
                self._create_default_quota(user_id)
                row = conn.execute(
                    "SELECT * FROM users_quota WHERE user_id = ?", (user_id,)
                ).fetchone()

            if not row:
                return False

            used = row["used_tokens"]
            new_remaining = max(0, new_total - used)

            conn.execute("""
                UPDATE users_quota 
                SET total_quota = ?, remaining = ?, updated_at = ?
                WHERE user_id = ?
            """, (new_total, new_remaining, now, user_id))
            conn.commit()

            logger.info(f"设置用户 {user_id} 额度: {new_total}")
            return True

    def get_usage_history(self, user_id: str, limit: int = 20) -> List[Dict]:
        """获取用户使用历史"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM usage_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit)
        ).fetchall()

        return [dict(r) for r in rows]

    def get_all_quotas(self) -> List[QuotaInfo]:
        """获取所有用户的额度信息（管理员）"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM users_quota ORDER BY last_usage DESC"
        ).fetchall()

        return [
            QuotaInfo(
                user_id=r["user_id"],
                total_quota=r["total_quota"],
                used_tokens=r["used_tokens"],
                remaining=r["remaining"],
                is_unlimited=bool(r["is_unlimited"]),
                last_usage=r["last_usage"] or 0,
                created_at=r["created_at"] or 0,
            ) for r in rows
        ]

    def reset_user_quota(self, user_id: str) -> bool:
        """重置用户额度到默认值"""
        conn = self._get_conn()
        now = time.time()
        default = int(self._get_config_value("default_quota", "100000"))

        with self._lock:
            row = conn.execute(
                "SELECT * FROM users_quota WHERE user_id = ?", (user_id,)
            ).fetchone()

            if not row or bool(row["is_unlimited"]):
                return True

            conn.execute("""
                UPDATE users_quota 
                SET total_quota = ?, used_tokens = 0, remaining = ?, updated_at = ?
                WHERE user_id = ?
            """, (default, default, now, user_id))

            conn.execute(
                "DELETE FROM usage_history WHERE user_id = ?", (user_id,)
            )
            conn.commit()

            logger.info(f"重置用户 {user_id} 额度: {default}")
            return True

    def update_config(self, **kwargs):
        """更新配置"""
        conn = self._get_conn()
        for key, value in kwargs.items():
            conn.execute(
                "INSERT OR REPLACE INTO quota_config (key, value) VALUES (?, ?)",
                (key, str(value))
            )
        conn.commit()
        self._config.update(kwargs)

    def _get_config_value(self, key: str, default: str) -> str:
        """获取配置值"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM quota_config WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def record_api_call(
        self,
        user_id: str,
        model_id: str,
        endpoint: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: float = 0,
        status: str = "success",
        error_message: str = "",
        request_id: str = "",
        api_key: str = ""
    ):
        """记录一次 API 调用"""
        conn = self._get_conn()
        now = time.time()

        try:
            conn.execute("""
                INSERT INTO api_call_stats
                (user_id, api_key, model_id, endpoint, input_tokens, output_tokens,
                 total_tokens, latency_ms, status, error_message, request_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, api_key, model_id, endpoint, input_tokens, output_tokens,
                total_tokens, latency_ms, status, error_message[:500], request_id, now
            ))
            conn.commit()
            logger.debug(f"记录API调用: user={user_id} model={model_id} tokens={total_tokens} status={status}")
        except Exception as e:
            logger.error(f"记录API调用失败: {e}")

    def get_user_api_stats(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict:
        """获取用户的 API 调用统计信息"""
        conn = self._get_conn()
        since = time.time() - (days * 86400)

        total_calls = conn.execute(
            "SELECT COUNT(*) as c FROM api_call_stats WHERE user_id = ? AND created_at > ?",
            (user_id, since)
        ).fetchone()["c"]

        successful_calls = conn.execute(
            "SELECT COUNT(*) as c FROM api_call_stats WHERE user_id = ? AND created_at > ? AND status = 'success'",
            (user_id, since)
        ).fetchone()["c"]

        failed_calls = total_calls - successful_calls

        total_tokens = conn.execute(
            "SELECT COALESCE(SUM(total_tokens), 0) as s FROM api_call_stats WHERE user_id = ? AND created_at > ?",
            (user_id, since)
        ).fetchone()["s"]

        total_input_tokens = conn.execute(
            "SELECT COALESCE(SUM(input_tokens), 0) as s FROM api_call_stats WHERE user_id = ? AND created_at > ?",
            (user_id, since)
        ).fetchone()["s"]

        total_output_tokens = conn.execute(
            "SELECT COALESCE(SUM(output_tokens), 0) as s FROM api_call_stats WHERE user_id = ? AND created_at > ?",
            (user_id, since)
        ).fetchone()["s"]

        avg_latency = conn.execute(
            "SELECT COALESCE(AVG(latency_ms), 0) as a FROM api_call_stats WHERE user_id = ? AND created_at > ? AND status = 'success'",
            (user_id, since)
        ).fetchone()["a"]

        model_stats = conn.execute("""
            SELECT model_id, COUNT(*) as calls, SUM(total_tokens) as tokens, AVG(latency_ms) as avg_latency
            FROM api_call_stats
            WHERE user_id = ? AND created_at > ?
            GROUP BY model_id
            ORDER BY calls DESC
        """, (user_id, since)).fetchall()

        daily_stats = conn.execute("""
            SELECT DATE(created_at, 'unixepoch', 'localtime') as day,
                   COUNT(*) as calls,
                   SUM(total_tokens) as tokens
            FROM api_call_stats
            WHERE user_id = ? AND created_at > ?
            GROUP BY day
            ORDER BY day DESC
            LIMIT ?
        """, (user_id, since, days)).fetchall()

        return {
            "user_id": user_id,
            "period_days": days,
            "total_api_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": round(successful_calls / total_calls * 100, 1) if total_calls > 0 else 0,
            "total_tokens_consumed": total_tokens,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "avg_latency_ms": round(avg_latency, 1),
            "by_model": [
                {
                    "model_id": r["model_id"],
                    "call_count": r["calls"],
                    "tokens_consumed": r["tokens"],
                    "avg_latency_ms": round(r["avg_latency"], 1) if r["avg_latency"] else 0
                } for r in model_stats
            ],
            "daily_summary": [
                {
                    "date": r["day"],
                    "api_calls": r["calls"],
                    "tokens_consumed": r["tokens"]
                } for r in daily_stats
            ]
        }

    def get_user_api_history(
        self,
        user_id: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict]:
        """获取用户的 API 调用历史记录"""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT * FROM api_call_stats
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (user_id, limit, offset)).fetchall()

        return [dict(r) for r in rows]

    def get_all_users_api_stats(self, days: int = 30) -> List[Dict]:
        """获取所有用户的 API 调用统计（管理员用）"""
        conn = self._get_conn()
        since = time.time() - (days * 86400)

        rows = conn.execute("""
            SELECT
                user_id,
                COUNT(*) as total_calls,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_calls,
                SUM(total_tokens) as total_tokens,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                AVG(CASE WHEN status = 'success' THEN latency_ms ELSE NULL END) as avg_latency
            FROM api_call_stats
            WHERE created_at > ?
            GROUP BY user_id
            ORDER BY total_calls DESC
        """, (since,)).fetchall()

        return [
            {
                "user_id": r["user_id"],
                "total_api_calls": r["total_calls"],
                "successful_calls": r["successful_calls"],
                "failed_calls": r["total_calls"] - r["successful_calls"],
                "total_tokens_consumed": r["total_tokens"] or 0,
                "input_tokens": r["input_tokens"] or 0,
                "output_tokens": r["output_tokens"] or 0,
                "avg_latency_ms": round(r["avg_latency"], 1) if r["avg_latency"] else 0
            } for r in rows
        ]

    def get_system_api_overview(self, days: int = 30) -> Dict:
        """获取系统级别的 API 调用概览（管理员用）"""
        conn = self._get_conn()
        since = time.time() - (days * 86400)

        total_calls = conn.execute(
            "SELECT COUNT(*) as c FROM api_call_stats WHERE created_at > ?",
            (since,)
        ).fetchone()["c"]

        total_tokens = conn.execute(
            "SELECT COALESCE(SUM(total_tokens), 0) as s FROM api_call_stats WHERE created_at > ?",
            (since,)
        ).fetchone()["s"]

        active_users = conn.execute(
            "SELECT COUNT(DISTINCT user_id) as c FROM api_call_stats WHERE created_at > ?",
            (since,)
        ).fetchone()["c"]

        popular_models = conn.execute("""
            SELECT model_id, COUNT(*) as calls, SUM(total_tokens) as tokens
            FROM api_call_stats
            WHERE created_at > ?
            GROUP BY model_id
            ORDER BY calls DESC
            LIMIT 10
        """, (since,)).fetchall()

        hourly_distribution = conn.execute("""
            SELECT strftime('%H', datetime(created_at, 'unixepoch', 'localtime')) as hour,
                   COUNT(*) as calls
            FROM api_call_stats
            WHERE created_at > ?
            GROUP BY hour
            ORDER BY hour
        """, (since,)).fetchall()

        return {
            "period_days": days,
            "total_api_calls": total_calls,
            "total_tokens_consumed": total_tokens,
            "active_users": active_users,
            "popular_models": [
                {
                    "model_id": r["model_id"],
                    "call_count": r["calls"],
                    "tokens_consumed": r["tokens"]
                } for r in popular_models
            ],
            "hourly_distribution": [
                {"hour": int(r["hour"]), "call_count": r["calls"]} for r in hourly_distribution
            ]
        }

    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        conn = self._get_conn()

        total_users = conn.execute("SELECT COUNT(*) as c FROM users_quota").fetchone()["c"]
        active_users = conn.execute(
            "SELECT COUNT(*) as c FROM users_quota WHERE last_usage > ?",
            (time.time() - 86400 * 7,)
        ).fetchone()["c"]

        total_used = conn.execute("SELECT COALESCE(SUM(used_tokens), 0) as s FROM users_quota").fetchone()["s"]
        total_remaining = conn.execute(
            "SELECT COALESCE(SUM(remaining), 0) as s FROM users_quota WHERE is_unlimited = 0"
        ).fetchone()["s"]

        return {
            "total_users": total_users,
            "active_users_7d": active_users,
            "total_tokens_used": total_used,
            "total_tokens_remaining": total_remaining,
            "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
        }


# 全局实例
_quota_manager: Optional[QuotaManager] = None


def get_quota_manager() -> QuotaManager:
    """获取全局额度管理器实例"""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
    return _quota_manager
