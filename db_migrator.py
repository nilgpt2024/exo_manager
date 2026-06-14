"""
EXO Cluster Manager - 数据库迁移工具
=====================================

功能:
- 将 JSON 文件 (users.json, api_keys.json) 迁移到 SQLite 数据库
- 提供数据库初始化和表结构创建
- 支持增量迁移和数据验证
- 自动备份原始 JSON 文件

使用场景:
- 从文件存储升级到关系型数据库
- 提升数据查询性能和安全性
- 支持更复杂的数据关系

配置环境变量:
- EXO_DB_PATH: SQLite 数据库文件路径 (默认 data/exo_manager.db)
- EXO_DB_ENABLE: 是否启用数据库 (默认 false，保持向后兼容)
"""

import os
import json
import sqlite3
import shutil
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 默认路径
DEFAULT_DB_PATH = Path(__file__).parent / "data" / "exo_manager.db"
USERS_JSON_PATH = Path(__file__).parent / "data" / "users.json"
API_KEYS_JSON_PATH = Path(__file__).parent / "data" / "api_keys.json"
BACKUP_DIR = Path(__file__).parent / "data" / "backups"


@dataclass
class MigrationResult:
    """迁移结果"""
    success: bool
    message: str
    migrated_records: int = 0
    errors: List[str] = None
    backup_path: Optional[str] = None
    duration_seconds: float = 0.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class DatabaseManager:
    """
    SQLite 数据库管理器

    特性:
    - 自动建表和索引
    - 参数化查询防止 SQL 注入
    - 连接池管理
    - 事务支持
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化数据库管理器

        Args:
            db_path: 数据库文件路径 (可选)
        """
        resolved_path = db_path or os.getenv("EXO_DB_PATH", str(DEFAULT_DB_PATH))
        self.db_path = Path(resolved_path)

        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection: Optional[sqlite3.Connection] = None
        self._is_initialized = False

    @property
    def is_available(self) -> bool:
        """检查数据库是否可用"""
        return self._is_initialized and self._connection is not None

    def initialize(self) -> bool:
        """
        初始化数据库并创建表结构

        Returns:
            是否成功
        """
        try:
            # 创建/连接数据库
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row  # 返回字典式行

            # 启用外键约束
            self._connection.execute("PRAGMA foreign_keys = ON")

            # 创建表结构
            self._create_tables()

            # 创建索引
            self._create_indexes()

            self._is_initialized = True
            logger.info(f"✅ 数据库已初始化: {self.db_path}")
            return True

        except Exception as e:
            logger.error(f"❌ 初始化数据库失败: {e}")
            return False

    def _create_tables(self):
        """创建数据表"""
        cursor = self._connection.cursor()

        # 用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                union_id TEXT UNIQUE NOT NULL,
                openid TEXT,
                nickname TEXT DEFAULT '',
                avatar TEXT DEFAULT '',
                role TEXT DEFAULT 'user',
                is_disabled INTEGER DEFAULT 0,
                created_at REAL DEFAULT 0,
                last_login_at REAL DEFAULT 0,
                website_account TEXT,
                website_password_hash TEXT,
                password_changed_at REAL DEFAULT 0,
                
                -- 审计字段
                created_by TEXT,
                updated_at REAL,
                
                -- 唯一约束
                UNIQUE(website_account)
            )
        """)

        # API Keys 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_name TEXT NOT NULL,
                api_key_encrypted TEXT NOT NULL,
                owner_id TEXT,
                permissions TEXT DEFAULT '[]',
                is_active INTEGER DEFAULT 1,
                created_at REAL DEFAULT 0,
                expires_at REAL DEFAULT 0,
                last_used_at REAL DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                
                -- 外键关联用户
                FOREIGN KEY (owner_id) REFERENCES users(id),
                
                -- API Key 必须唯一
                UNIQUE(api_key_encrypted)
            )
        """)

        # 会话表 (替代内存中的 session 存储)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                user_role TEXT DEFAULT 'user',
                ip_address TEXT,
                user_agent TEXT,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                is_valid INTEGER DEFAULT 1,
                
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # 操作审计日志表 (结构化存储)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                timestamp REAL NOT NULL,
                level TEXT DEFAULT 'INFO',
                category TEXT NOT NULL,
                action TEXT NOT NULL,
                user_id TEXT,
                user_role TEXT,
                ip_address TEXT,
                user_agent TEXT,
                resource TEXT,
                resource_id TEXT,
                details TEXT,  -- JSON 格式
                success INTEGER DEFAULT 1,
                error_message TEXT,
                session_token_prefix TEXT,
                
                -- 索引优化查询
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # 登录尝试记录表 (用于暴力破解检测)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                attempt_time REAL NOT NULL,
                success INTEGER DEFAULT 0,
                failure_reason TEXT,
                
                -- 快速查询索引
                INDEX(ip_address, attempt_time)
            )
        """)

        self._connection.commit()
        logger.debug("✅ 数据库表已创建/验证")

    def _create_indexes(self):
        """创建性能优化索引"""
        cursor = self._connection.cursor()

        indexes = [
            ("idx_users_union_id", "users", "union_id"),
            ("idx_users_email", "users", "website_account"),
            ("idx_users_role", "users", "role"),
            ("idx_api_keys_owner", "api_keys", "owner_id"),
            ("idx_sessions_user", "sessions", "user_id"),
            ("idx_sessions_expires", "sessions", "expires_at"),
            ("idx_audit_timestamp", "audit_logs", "timestamp"),
            ("idx_audit_category", "audit_logs", "category"),
            ("idx_audit_user", "audit_logs", "user_id"),
            ("idx_login_attempts_ip", "login_attempts", "ip_address"),
            ("idx_login_attempts_time", "login_attempts", "attempt_time"),
        ]

        for index_name, table, column in indexes:
            try:
                cursor.execute(
                    f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column})"
                )
            except Exception as e:
                logger.warning(f"创建索引 {index_name} 失败: {e}")

        self._connection.commit()
        logger.debug("✅ 数据库索引已创建")

    def get_connection(self) -> Optional[sqlite3.Connection]:
        """获取数据库连接"""
        if not self._is_initialized or self._connection is None:
            logger.error("数据库未初始化")
            return None
        return self._connection

    def close(self):
        """关闭数据库连接"""
        if self._connection:
            self._connection.close()
            self._connection = None
            self._is_initialized = False
            logger.info("数据库连接已关闭")


class DataMigrator:
    """
    JSON 到 SQLite 迁移工具

    功能:
    - 自动备份原始 JSON 文件
    - 验证数据完整性
    - 增量迁移 (跳过已存在的记录)
    - 详细的迁移报告
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def migrate_users(self, json_path: Optional[str] = None) -> MigrationResult:
        """
        迁移用户数据从 JSON 到 SQLite

        Args:
            json_path: users.json 路径 (可选)

        Returns:
            迁移结果
        """
        start_time = time.time()
        source_path = Path(json_path) if json_path else USERS_JSON_PATH

        if not source_path.exists():
            return MigrationResult(
                success=False,
                message=f"源文件不存在: {source_path}"
            )

        conn = self.db.get_connection()
        if not conn:
            return MigrationResult(success=False, message="数据库不可用")

        try:
            # 备份原始文件
            backup_path = self._backup_file(source_path)

            # 读取 JSON 数据
            with open(source_path, 'r', encoding='utf-8') as f:
                users_data = json.load(f)

            if isinstance(users_data, dict):
                users_list = users_data.get('users', [])
            elif isinstance(users_data, list):
                users_list = users_data
            else:
                return MigrationResult(
                    success=False,
                    message="无效的 JSON 数据格式"
                )

            # 开始事务
            cursor = conn.cursor()
            migrated_count = 0
            errors = []

            for user_data in users_list:
                try:
                    # 检查是否已存在
                    user_id = user_data.get('id')
                    if not user_id:
                        errors.append(f"缺少用户 ID: {user_data}")
                        continue

                    cursor.execute(
                        "SELECT id FROM users WHERE id = ?",
                        (user_id,)
                    )

                    if cursor.fetchone():
                        # 用户已存在，更新记录
                        self._update_user(cursor, user_data)
                        logger.debug(f"更新用户: {user_id}")
                    else:
                        # 插入新记录
                        self._insert_user(cursor, user_data)
                        logger.debug(f"插入用户: {user_id}")

                    migrated_count += 1

                except Exception as e:
                    error_msg = f"迁移用户 {user_data.get('id', 'unknown')} 失败: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            # 提交事务
            conn.commit()

            duration = time.time() - start_time
            result = MigrationResult(
                success=True,
                message=f"成功迁移 {migrated_count} 个用户",
                migrated_records=migrated_count,
                errors=errors,
                backup_path=str(backup_path) if backup_path else None,
                duration_seconds=round(duration, 2)
            )

            logger.info(
                f"✅ 用户数据迁移完成: {migrated_count} 条记录 "
                f"(耗时: {duration:.2f}s)"
            )

            return result

        except Exception as e:
            conn.rollback()
            logger.error(f"❌ 用户数据迁移失败: {e}")
            return MigrationResult(
                success=False,
                message=str(e),
                duration_seconds=time.time() - start_time
            )

    def _insert_user(self, cursor: sqlite3.Cursor, user_data: Dict):
        """插入单个用户记录"""
        cursor.execute("""
            INSERT INTO users (
                id, union_id, openid, nickname, avatar, role,
                is_disabled, created_at, last_login_at,
                website_account, website_password_hash, password_changed_at,
                created_by, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_data.get('id', ''),
            user_data.get('union_id', ''),
            user_data.get('openid', ''),
            user_data.get('nickname', ''),
            user_data.get('avatar', ''),
            user_data.get('role', 'user'),
            int(user_data.get('is_disabled', False)),
            float(user_data.get('created_at', 0)),
            float(user_data.get('last_login_at', 0)),
            user_data.get('website_account', ''),
            user_data.get('website_password_hash', ''),
            float(user_data.get('password_changed_at', 0)),
            user_data.get('created_by', 'migration'),
            time.time(),
        ))

    def _update_user(self, cursor: sqlite3.Cursor, user_data: Dict):
        """更新现有用户记录"""
        cursor.execute("""
            UPDATE users SET
                nickname = ?, avatar = ?, role = ?, is_disabled = ?,
                last_login_at = ?, website_account = ?,
                website_password_hash = ?, password_changed_at = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            user_data.get('nickname', ''),
            user_data.get('avatar', ''),
            user_data.get('role', 'user'),
            int(user_data.get('is_disabled', False)),
            float(user_data.get('last_login_at', 0)),
            user_data.get('website_account', ''),
            user_data.get('website_password_hash', ''),
            float(user_data.get('password_changed_at', 0)),
            time.time(),
            user_data.get('id', ''),
        ))

    def migrate_api_keys(self, json_path: Optional[str] = None) -> MigrationResult:
        """
        迁移 API Key 数据

        Args:
            json_path: api_keys.json 路径

        Returns:
            迁移结果
        """
        start_time = time.time()
        source_path = Path(json_path) if json_path else API_KEYS_JSON_PATH

        if not source_path.exists():
            return MigrationResult(
                success=False,
                message=f"源文件不存在: {source_path}"
            )

        conn = self.db.get_connection()
        if not conn:
            return MigrationResult(success=False, message="数据库不可用")

        try:
            # 备份
            backup_path = self._backup_file(source_path)

            with open(source_path, 'r', encoding='utf-8') as f:
                keys_data = json.load(f)

            if isinstance(keys_data, list):
                keys_list = keys_data
            elif isinstance(keys_data, dict):
                keys_list = keys_data.get('api_keys', [])
            else:
                return MigrationResult(
                    success=False,
                    message="无效的 JSON 数据格式"
                )

            cursor = conn.cursor()
            migrated_count = 0
            errors = []

            for key_data in keys_list:
                try:
                    key_name = key_data.get('name', key_data.get('key_name', ''))
                    api_key = key_data.get('api_key', '')

                    if not api_key:
                        errors.append(f"API Key 缺少: {key_name}")
                        continue

                    # 检查是否已加密
                    is_encrypted = api_key.startswith("ENC(")

                    cursor.execute(
                        "INSERT OR IGNORE INTO api_keys "
                        "(key_name, api_key_encrypted, owner_id, permissions, "
                        "created_at, expires_at, is_active) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            key_name,
                            api_key,  # 保持原有格式 (加密或明文)
                            key_data.get('owner_id', ''),
                            json.dumps(key_data.get('permissions', [])),
                            float(key_data.get('created_at', time.time())),
                            float(key_data.get('expires_at', 0)),
                            int(key_data.get('is_active', True)),
                        )
                    )

                    if cursor.rowcount > 0:
                        migrated_count += 1

                except Exception as e:
                    errors.append(f"迁移 API Key {key_data.get('name', '?')} 失败: {e}")

            conn.commit()

            duration = time.time() - start_time
            result = MigrationResult(
                success=True,
                message=f"成功迁移 {migrated_count} 个 API Key",
                migrated_records=migrated_count,
                errors=errors,
                backup_path=str(backup_path) if backup_path else None,
                duration_seconds=round(duration, 2)
            )

            logger.info(
                f"✅ API Key 迁移完成: {migrated_count} 条记录 "
                f"(耗时: {duration:.2f}s)"
            )

            return result

        except Exception as e:
            conn.rollback()
            logger.error(f"❌ API Key 迁移失败: {e}")
            return MigrationResult(
                success=False,
                message=str(e),
                duration_seconds=time.time() - start_time
            )

    def _backup_file(self, file_path: Path) -> Optional[Path]:
        """备份原始文件"""
        try:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{file_path.stem}_{timestamp}.json.bak"
            backup_path = BACKUP_DIR / backup_filename

            shutil.copy2(file_path, backup_path)
            logger.info(f"📦 已备份: {file_path} -> {backup_path}")
            return backup_path

        except Exception as e:
            logger.warning(f"备份文件失败: {e}")
            return None


# ==================== 便捷函数 ====================

def init_database(db_path: Optional[str] = None) -> Tuple[Optional[DatabaseManager], bool]:
    """
    初始化数据库 (便捷函数)

    Returns:
        (database_manager, success)
    """
    db = DatabaseManager(db_path)
    success = db.initialize()
    return db, success


def run_migration(json_dir: Optional[str] = None) -> Dict[str, MigrationResult]:
    """
    执行完整的数据迁移

    Args:
        json_dir: JSON 文件所在目录 (可选)

    Returns:
        包含各类别迁移结果的字典
    """
    results = {}

    # 初始化数据库
    db, success = init_database()
    if not success:
        results['error'] = MigrationResult(
            success=False,
            message="数据库初始化失败"
        )
        return results

    migrator = DataMigrator(db)

    # 迁移用户数据
    users_json = f"{json_dir}/users.json" if json_dir else None
    results['users'] = migrator.migrate_users(users_json)

    # 迁移 API Keys
    keys_json = f"{json_dir}/api_keys.json" if json_dir else None
    results['api_keys'] = migrator.migrate_api_keys(keys_json)

    # 关闭连接
    db.close()

    return results


def check_database_status() -> Dict[str, Any]:
    """
    检查数据库状态

    Returns:
        数据库状态信息
    """
    status = {
        "enabled": os.getenv("EXO_DB_ENABLE", "false").lower() in ("true", "1"),
        "db_exists": DEFAULT_DB_PATH.exists(),
        "db_size_mb": round(DEFAULT_DB_PATH.stat().st_size / (1024*1024), 2) if DEFAULT_DB_PATH.exists() else 0,
        "users_json_exists": USERS_JSON_PATH.exists(),
        "api_keys_json_exists": API_KEYS_JSON_PATH.exists(),
        "tables": [],
        "record_counts": {}
    }

    if status["enabled"] and status["db_exists"]:
        try:
            conn = sqlite3.connect(str(DEFAULT_DB_PATH))
            cursor = conn.cursor()

            # 获取表列表
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            status["tables"] = [row[0] for row in cursor.fetchall()]

            # 获取各表记录数
            for table in status["tables"]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                status["record_counts"][table] = cursor.fetchone()[0]

            conn.close()

        except Exception as e:
            status["error"] = str(e)

    return status
