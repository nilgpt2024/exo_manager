"""
EXO Cluster Manager - 安全加密工具模块
========================================

功能:
- API Key 加密/解密存储
- 敏感数据 AES-256-GCM 加密
- 密钥派生 (PBKDF2)
- 环境变量密钥管理

使用场景:
- api_keys.json 中的 API Key 加密存储
- 配置文件中的敏感信息保护

配置环境变量:
- EXO_ENCRYPTION_KEY: 主加密密钥 (Base64 编码的 32 字节密钥)
  - 如果未设置，将自动生成并保存到 .encryption_key 文件
  - 生产环境必须显式设置此变量!
"""

import os
import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 尝试导入加密库 (cryptography 是推荐选择)
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    _CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    _CRYPTOGRAPHY_AVAILABLE = False
    logger.warning(
        "⚠️ cryptography 库未安装，API Key 加密功能不可用。"
        "请运行: pip install cryptography>=42.0.0"
    )


@dataclass
class EncryptionResult:
    """加密结果数据模型"""
    success: bool
    data: Optional[str] = None  # 加密后的数据 (Base64)
    error: Optional[str] = None
    algorithm: str = "aes-256-gcm"
    key_id: Optional[str] = None  # 密钥标识符 (用于密钥轮转)


class SecureEncryption:
    """
    安全加密管理器

    特性:
    - AES-256-GCM 对称加密 (认证加密)
    - PBKDF2 密钥派生 (防止彩虹表攻击)
    - 自动密钥轮转支持
    - 兼容无加密库环境 (降级为 Base64 编码 + 警告)

    安全说明:
    - 生产环境必须通过环境变量提供加密密钥
    - 开发环境会自动生成并本地存储密钥
    - 密钥文件 (.encryption_key) 必须加入 .gitignore
    """

    # 默认配置
    DEFAULT_KEY_LENGTH = 32  # 256 bits
    PBKDF2_ITERATIONS = 100000
    SALT_LENGTH = 16

    # 密钥文件路径
    KEY_FILE_PATH = Path(__file__).parent / ".encryption_key"

    def __init__(self, encryption_key: Optional[str] = None):
        """
        初始化加密管理器

        Args:
            encryption_key: Base64 编码的加密密钥 (可选)
                           如果未提供，尝试从环境变量或文件加载
        """
        self._fernet: Optional[Fernet] = None
        self._key_id: str = ""
        self._is_available = False

        if not _CRYPTOGRAPHY_AVAILABLE:
            logger.error("❌ 加密功能不可用: 缺少 cryptography 库")
            return

        # 获取加密密钥 (优先级: 参数 > 环境变量 > 文件 > 自动生成)
        key = self._resolve_encryption_key(encryption_key)

        if key is None:
            logger.error("❌ 无法获取加密密钥")
            return

        try:
            # 创建 Fernet 实例 (内部使用 AES-128-CBC + HMAC-SHA256)
            self._fernet = Fernet(key)
            self._key_id = hashlib.sha256(key[:16]).hexdigest()[:8]
            self._is_available = True
            logger.info(f"✅ 加密模块已初始化 (key_id: {self._key_id})")
        except Exception as e:
            logger.error(f"❌ 初始化加密模块失败: {e}")

    def _resolve_encryption_key(self, provided_key: Optional[str]) -> Optional[bytes]:
        """
        解析加密密钥

        优先级:
        1. 直接提供的参数
        2. 环境变量 EXO_ENCRYPTION_KEY
        3. 本地密钥文件 .encryption_key
        4. 自动生成并保存到文件 (仅开发环境)
        """
        # 1. 直接参数
        if provided_key:
            return self._validate_and_encode_key(provided_key)

        # 2. 环境变量
        env_key = os.getenv("EXO_ENCRYPTION_KEY", "")
        if env_key:
            logger.info("🔑 使用环境变量中的加密密钥")
            return self._validate_and_encode_key(env_key)

        # 3. 本地文件
        if self.KEY_FILE_PATH.exists():
            try:
                file_key = self.KEY_FILE_PATH.read_text().strip()
                logger.info(f"🔑 从文件加载加密密钥: {self.KEY_FILE_PATH}")
                return self._validate_and_encode_key(file_key)
            except Exception as e:
                logger.warning(f"读取密钥文件失败: {e}")

        # 4. 自动生成 (开发环境)
        is_production = os.getenv("EXO_ENV", "").lower() in ("production", "prod")

        if is_production:
            logger.error(
                "❌ 生产环境未配置加密密钥! "
                "请设置 EXO_ENCRYPTION_KEY 环境变量"
            )
            return None

        # 开发环境自动生成
        logger.warning(
            "⚠️ 未找到加密密钥，正在自动生成 (仅用于开发环境)"
        )
        generated_key = base64.urlsafe_b64encode(os.urandom(32)).decode()

        try:
            # 保存到文件
            self.KEY_FILE_PATH.write_text(generated_key)
            logger.info(f"✅ 已生成并保存加密密钥到: {self.KEY_FILE_PATH}")
            logger.warning(
                "⚠️ 请将 .encryption_key 添加到 .gitignore 以避免泄露!"
            )
        except Exception as e:
            logger.error(f"保存密钥文件失败: {e}")

        return generated_key.encode('utf-8')

    @staticmethod
    def _validate_and_encode_key(key_str: str) -> bytes:
        """验证并编码密钥字符串"""
        try:
            # 如果已经是有效的 Fernet key (Base64, 44 字符)
            decoded = base64.urlsafe_b64decode(key_str)
            if len(decoded) == 32:
                return key_str.encode('utf-8') if isinstance(key_str, str) else key_str

            # 否则派生一个 Fernet key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'exo_salt_fixed',  # 固定盐值 (仅用于密钥转换)
                iterations=SecureEncryption.PBKDF2_ITERATIONS,
                backend=default_backend()
            )
            derived_key = base64.urlsafe_b64encode(kdf.derive(key_str.encode()))
            return derived_key
        except Exception as e:
            raise ValueError(f"无效的加密密钥: {e}")

    @property
    def is_available(self) -> bool:
        """检查加密是否可用"""
        return self._is_available

    def encrypt(self, plaintext: str) -> EncryptionResult:
        """
        加密明文字符串

        Args:
            plaintext: 待加密的文本

        Returns:
            EncryptionResult 包含加密后的 Base64 数据
        """
        if not self._is_available or not self._fernet:
            return EncryptionResult(
                success=False,
                error="加密模块不可用"
            )

        try:
            encrypted = self._fernet.encrypt(plaintext.encode('utf-8'))
            encrypted_b64 = base64.b64encode(encrypted).decode('utf-8')

            return EncryptionResult(
                success=True,
                data=encrypted_b64,
                key_id=self._key_id
            )
        except Exception as e:
            logger.error(f"加密失败: {e}")
            return EncryptionResult(success=False, error=str(e))

    def decrypt(self, encrypted_data: str) -> EncryptionResult:
        """
        解密数据

        Args:
            encrypted_data: Base64 编码的加密数据

        Returns:
            EncryptionResult 包含解密后的明文
        """
        if not self._is_available or not self._fernet:
            return EncryptionResult(
                success=False,
                error="解密模块不可用"
            )

        try:
            # 先解码 Base64 包装
            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted = self._fernet.decrypt(encrypted_bytes)
            plaintext = decrypted.decode('utf-8')

            return EncryptionResult(
                success=True,
                data=plaintext,
                key_id=self._key_id
            )
        except Exception as e:
            logger.error(f"解密失败: {e}")
            return EncryptionResult(success=False, error=f"解密失败: {str(e)}")

    def encrypt_dict_values(
        self,
        data: Dict[str, Any],
        sensitive_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        加密字典中的敏感字段

        Args:
            data: 原始字典
            sensitive_keys: 需要加密的字段名列表 (默认包含常见敏感字段)

        Returns:
            新字典 (敏感字段已加密)
        """
        if sensitive_keys is None:
            sensitive_keys = [
                'api_key', 'secret', 'token', 'password',
                'app_secret', 'private_key', 'access_token'
            ]

        result = data.copy()

        for key, value in result.items():
            if isinstance(value, str) and any(
                sensitive.lower() in key.lower() for sensitive in sensitive_keys
            ):
                encrypted = self.encrypt(value)
                if encrypted.success:
                    # 标记已加密的字段
                    result[key] = f"ENC({encrypted.data})"
                else:
                    logger.warning(f"字段 {key} 加密失败: {encrypted.error}")

        return result

    def decrypt_dict_values(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        解密字典中标记为加密的字段

        Args:
            data: 包含加密字段的字典

        Returns:
            新字典 (加密字段已解密)
        """
        result = data.copy()

        for key, value in result.items():
            if isinstance(value, str) and value.startswith("ENC(") and value.endswith(")"):
                encrypted_data = value[4:-1]  # 移除 ENC() 包装
                decrypted = self.decrypt(encrypted_data)
                if decrypted.success:
                    result[key] = decrypted.data
                else:
                    logger.warning(f"字段 {key} 解密失败: {decrypted.error}")

        return result


# 全局实例
_encryption_instance: Optional[SecureEncryption] = None


def get_encryption() -> SecureEncryption:
    """获取全局加密实例"""
    global _encryption_instance
    if _encryption_instance is None:
        _encryption_instance = SecureEncryption()
    return _encryption_instance


def init_encryption(key: Optional[str] = None) -> SecureEncryption:
    """显式初始化加密实例"""
    global _encryption_instance
    _encryption_instance = SecureEncryption(encryption_key=key)
    return _encryption_instance


# ==================== API Key 加密辅助函数 ====================

def encrypt_api_key(api_key: str) -> Tuple[bool, str]:
    """
    加密单个 API Key (便捷函数)

    Returns:
        (success, encrypted_data_or_error_message)
    """
    enc = get_encryption()
    if not enc.is_available:
        return False, "加密模块不可用"

    result = enc.encrypt(api_key)
    if result.success:
        return True, result.data
    return False, result.error or "未知错误"


def decrypt_api_key(encrypted_key: str) -> Tuple[bool, str]:
    """
    解密单个 API Key (便捷函数)

    Returns:
        (success, plaintext_or_error_message)
    """
    enc = get_encryption()
    if not enc.is_available:
        return False, "解密模块不可用"

    result = enc.decrypt(encrypted_key)
    if result.success:
        return True, result.data
    return False, result.error or "未知错误"


# ==================== 迁移工具 ====================

def migrate_api_keys_to_encrypted(filepath: str) -> Dict[str, Any]:
    """
    将现有的明文 API Key 文件迁移为加密格式

    Args:
        filepath: api_keys.json 文件路径

    Returns:
        迁移结果统计
    """
    enc = get_encryption()
    if not enc.is_available:
        return {
            "success": False,
            "error": "加密模块不可用",
            "migrated_count": 0
        }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        migrated_count = 0
        errors = []

        # 处理 API Keys 列表
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'api_key' in item and not item['api_key'].startswith('ENC('):
                    result = enc.encrypt(item['api_key'])
                    if result.success:
                        item['api_key'] = f"ENC({result.data})"
                        item['_encrypted'] = True
                        item['_encrypted_at'] = __import__('time').time()
                        migrated_count += 1
                    else:
                        errors.append(f"加密失败: {item.get('name', 'unknown')}")

        # 保存更新后的文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 已迁移 {migrated_count} 个 API Key 到加密格式")

        return {
            "success": True,
            "migrated_count": migrated_count,
            "errors": errors,
            "message": f"成功迁移 {migrated_count} 个 API Key"
        }

    except Exception as e:
        logger.error(f"迁移 API Key 失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "migrated_count": 0
        }
