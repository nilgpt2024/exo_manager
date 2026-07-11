"""
EXO Cluster Manager - API Key 管理模块
======================================

提供 API Key 的生成、验证、存储和管理功能

特性:
- 支持 Bearer Token 认证
- API Key 持久化存储 (JSON 文件)
- 支持 key 的启用/禁用/删除
- 支持为 key 设置名称和描述
- 可选: 为 key 绑定特定模型权限

存储格式:
{
    "keys": [
        {
            "key": "exo_sk_xxxxxxxx",
            "name": "开发测试",
            "description": "用于本地开发测试",
            "created_at": 1234567890,
            "last_used_at": 1234567900,
            "is_active": true,
            "permissions": ["chat", "completions"],
            "allowed_models": ["*"],  // ["*"] 表示所有模型
            "user_id": "user_uuid"   // 关联的用户 ID（可选）
        }
    ]
}
"""

import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

API_KEY_PREFIX = "exo_sk_"
API_KEY_LENGTH = 32  # 不包括前缀


@dataclass
class APIKey:
    """API Key 数据模型"""
    key: str
    name: str = ""
    description: str = ""
    created_at: float = 0
    last_used_at: Optional[float] = None
    is_active: bool = True
    permissions: List[str] = field(default_factory=lambda: ["*"])
    allowed_models: List[str] = field(default_factory=lambda: ["*"])
    usage_count: int = 0
    user_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "APIKey":
        return cls(**data)


class APIKeyManager:
    """
    API Key 管理器

    负责:
    - 生成新的 API Key
    - 验证 API Key 是否有效
    - 持久化存储到文件
    - 管理 key 的生命周期 (启用/禁用/删除)
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        初始化 API Key 管理器

        Args:
            storage_path: API Key 存储文件路径，默认在 exo_manager 目录下
        """
        if storage_path is None:
            base_dir = Path(__file__).parent
            storage_path = str(base_dir / "api_keys.json")

        self.storage_path = Path(storage_path)
        self._keys: Dict[str, APIKey] = {}  # key -> APIKey
        self._load_keys()

    def _load_keys(self):
        """从文件加载 API Keys"""
        if not self.storage_path.exists():
            logger.info(f"API Key 存储文件不存在，将在 {self.storage_path} 创建")
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key_data in data.get("keys", []):
                key_obj = APIKey.from_dict(key_data)
                self._keys[key_obj.key] = key_obj

            logger.info(f"已加载 {len(self._keys)} 个 API Key")
        except Exception as e:
            logger.error(f"加载 API Key 失败: {e}")

    def _save_keys(self):
        """保存 API Keys 到文件"""
        try:
            data = {
                "keys": [key.to_dict() for key in self._keys.values()]
            }

            # 确保目录存在
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"保存 API Key 失败: {e}")

    def generate_key(
        self,
        name: str = "",
        description: str = "",
        permissions: Optional[List[str]] = None,
        allowed_models: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        生成新的 API Key

        Args:
            name: Key 的名称（用于识别）
            description: Key 的描述
            permissions: 权限列表，["*"] 表示所有权限
            allowed_models: 允许的模型列表，["*"] 表示所有模型
            user_id: 关联的用户 ID（可选，普通用户创建时设置）

        Returns:
            生成的 API Key 字符串 (包含前缀)
        """
        # 生成随机 key
        random_part = secrets.token_urlsafe(API_KEY_LENGTH)[:API_KEY_LENGTH]
        key_str = f"{API_KEY_PREFIX}{random_part}"

        api_key = APIKey(
            key=key_str,
            name=name or f"Key-{len(self._keys) + 1}",
            description=description,
            created_at=time.time(),
            is_active=True,
            permissions=permissions or ["*"],
            allowed_models=allowed_models or ["*"],
            user_id=user_id
        )

        self._keys[key_str] = api_key
        self._save_keys()

        logger.info(f"生成新 API Key: {key_str[:16]}... (名称: {api_key.name})")
        return key_str

    def validate_key(self, key: str, required_permission: Optional[str] = None) -> bool:
        """
        验证 API Key 是否有效

        Args:
            key: API Key 字符串
            required_permission: 需要的权限（可选）

        Returns:
            是否有效
        """
        if not key:
            return False

        api_key = self._keys.get(key)
        if not api_key:
            return False

        if not api_key.is_active:
            return False

        # 检查权限
        if required_permission and required_permission != "*":
            if "*" not in api_key.permissions and required_permission not in api_key.permissions:
                return False

        # 更新使用统计
        api_key.last_used_at = time.time()
        api_key.usage_count += 1
        self._save_keys()

        return True

    def get_key_info(self, key: str) -> Optional[APIKey]:
        """获取 API Key 的详细信息"""
        return self._keys.get(key)

    def list_keys(self) -> List[Dict]:
        """
        列出所有 API Key（返回脱敏后的信息）

        Returns:
            包含 key 信息的列表，key 值会被部分隐藏
        """
        result = []
        for api_key in self._keys.values():
            info = api_key.to_dict()
            # 脱敏处理：只显示前缀和部分 key
            info["key"] = self._mask_key(api_key.key)
            result.append(info)
        return result

    def list_keys_by_user(self, user_id: str) -> List[Dict]:
        """
        列出指定用户的 API Key（返回脱敏后的信息）

        Args:
            user_id: 用户 ID

        Returns:
            包含 key 信息的列表，key 值会被部分隐藏
        """
        result = []
        for api_key in self._keys.values():
            if api_key.user_id != user_id:
                continue
            info = api_key.to_dict()
            info["key"] = self._mask_key(api_key.key)
            result.append(info)
        return result

    def revoke_key(self, key: str, user_id: Optional[str] = None) -> bool:
        """
        吊销（删除）API Key

        Args:
            key: 完整的 API Key 或脱敏后的 key
            user_id: 用户 ID（可选），传入时只会删除属于该用户的 key

        Returns:
            是否成功删除
        """
        # 如果传入的是脱敏的 key，需要找到完整的 key
        target_key = None
        if key.startswith(API_KEY_PREFIX):
            target_key = key
        else:
            # 尝试匹配脱敏 key
            for full_key in self._keys:
                if self._mask_key(full_key) == key:
                    target_key = full_key
                    break

        if target_key and target_key in self._keys:
            api_key = self._keys[target_key]
            # 如果指定了 user_id，则校验 key 归属
            if user_id is not None and api_key.user_id != user_id:
                logger.warning(f"用户 {user_id} 尝试删除不属于自己的 API Key: {key[:16]}...")
                return False
            del self._keys[target_key]
            self._save_keys()
            logger.info(f"吊销 API Key: {key[:16]}...")
            return True
        return False

    def disable_key(self, key: str) -> bool:
        """禁用 API Key"""
        api_key = self._keys.get(key)
        if api_key:
            api_key.is_active = False
            self._save_keys()
            return True
        return False

    def enable_key(self, key: str) -> bool:
        """启用 API Key"""
        api_key = self._keys.get(key)
        if api_key:
            api_key.is_active = True
            self._save_keys()
            return True
        return False

    def update_key(
        self,
        key: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        allowed_models: Optional[List[str]] = None
    ) -> bool:
        """更新 API Key 信息"""
        api_key = self._keys.get(key)
        if not api_key:
            return False

        if name is not None:
            api_key.name = name
        if description is not None:
            api_key.description = description
        if allowed_models is not None:
            api_key.allowed_models = allowed_models

        self._save_keys()
        return True

    def check_model_access(self, key: str, model_id: str) -> bool:
        """检查 key 是否有权限访问指定模型"""
        api_key = self._keys.get(key)
        if not api_key:
            return False

        if "*" in api_key.allowed_models:
            return True

        return model_id in api_key.allowed_models

    @staticmethod
    def _mask_key(key: str) -> str:
        """对 key 进行脱敏处理"""
        if len(key) <= 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"

    def get_stats(self) -> Dict:
        """获取 API Key 使用统计"""
        total = len(self._keys)
        active = sum(1 for k in self._keys.values() if k.is_active)
        total_usage = sum(k.usage_count for k in self._keys.values())

        return {
            "total_keys": total,
            "active_keys": active,
            "inactive_keys": total - active,
            "total_usage_count": total_usage
        }


# 全局 API Key 管理器实例
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """获取全局 API Key 管理器实例"""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
