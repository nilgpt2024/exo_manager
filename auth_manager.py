"""
EXO Cluster Manager - 用户认证和权限管理模块
=============================================

提供用户登录、权限控制和会话管理功能

支持的登录方式:
- 微信扫码登录 (OAuth2 - 需要配置微信开放平台)
- 管理员账号密码登录

权限角色:
- admin: 管理员，可访问所有功能
- user: 普通用户，只能使用模型推理和 API Key 管理

存储格式 (users.json):
{
    "users": [
        {
            "id": "uuid",
            "union_id": "微信unionid",
            "openid": "微信openid",
            "nickname": "昵称",
            "avatar": "头像URL",
            "role": "user|admin",
            "created_at": 1234567890,
            "last_login_at": 1234567890
        }
    ],
    "sessions": {
        "session_token": {
            "user_id": "uuid",
            "created_at": 1234567890,
            "expires_at": 1234567890
        }
    }
}

微信配置文件 (wechat_config.json):
{
    "wechat": {
        "app_id": "",
        "app_secret": "",
        "redirect_uri": "http://localhost:8080/auth/wechat/callback",
        "scope": "snsapi_login"
    }
}
"""

import json
import hashlib
import logging
import os
import secrets
import time
import uuid as uuid_lib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import httpx

logger = logging.getLogger(__name__)

# Session 有效期 (秒)
SESSION_EXPIRY = 7 * 24 * 3600  # 7 天

# 安全配置
MAX_LOGIN_ATTEMPTS = 5          # 最大登录尝试次数
LOGIN_LOCKOUT_TIME = 900        # 锁定时间 (秒) = 15分钟
PASSWORD_SALT = "exo_cluster_salt_v1"  # 密码哈希盐值 (生产环境应使用随机盐)


@dataclass
class User:
    """用户数据模型"""
    id: str
    union_id: str = ""
    openid: str = ""
    nickname: str = ""
    avatar: str = ""
    role: str = "user"
    is_disabled: bool = False
    created_at: float = 0
    last_login_at: float = 0
    website_account: str = ""
    website_password: str = ""
    website_password_hash: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "User":
        safe_data = data.copy()
        
        for field in ['website_password', 'website_password_hash']:
            if field not in safe_data:
                safe_data[field] = ''
        
        return cls(**safe_data)

    def to_public_dict(self) -> Dict:
        return {
            "id": self.id,
            "nickname": self.nickname,
            "avatar": self.avatar,
            "role": self.role,
            "is_disabled": getattr(self, 'is_disabled', False),
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
        }


class AuthManager:
    """
    用户认证管理器

    负责:
    - 微信扫码登录流程管理 (OAuth2)
    - 用户数据持久化
    - Session 管理
    - 权限验证
    """

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            base_dir = Path(__file__).parent
            storage_path = str(base_dir / "users.json")

        self.storage_path = Path(storage_path)
        self._users: Dict[str, User] = {}  # id -> User
        self._sessions: Dict[str, Dict] = {}  # token -> session info
        self._login_qrcodes: Dict[str, Dict] = {}  # qrcode_id -> qrcode info
        self._login_attempts: Dict[str, Dict] = {}  # ip -> {count, last_attempt, locked_until}

        # 微信配置
        self.wechat_config: Dict = {}
        self.wechat_enabled = False
        self._load_wechat_config()

        self._load_data()

    def _load_wechat_config(self):
        """加载微信登录配置"""
        config_path = Path(__file__).parent / "wechat_config.json"

        if not config_path.exists():
            logger.warning("微信配置文件不存在，将使用模拟登录模式")
            self.wechat_enabled = False
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            wechat = config.get("wechat", {})
            app_id = wechat.get("app_id", "").strip()
            app_secret = wechat.get("app_secret", "").strip()

            if not app_id or not app_secret:
                logger.warning("微信 AppID 或 AppSecret 未配置，将使用模拟登录模式")
                self.wechat_enabled = False
                return

            self.wechat_config = {
                "app_id": app_id,
                "app_secret": app_secret,
                "redirect_uri": wechat.get(
                    "redirect_uri",
                    "http://localhost:8080/auth/wechat/callback"
                ),
                "scope": wechat.get("scope", "snsapi_login"),
                "state_prefix": wechat.get("state_prefix", "exo_"),
            }
            self.wechat_enabled = True
            logger.info(f"微信登录已启用 (AppID: {app_id[:8]}...)")

        except Exception as e:
            logger.error(f"加载微信配置失败: {e}")
            self.wechat_enabled = False

    def _load_data(self):
        """从文件加载用户数据"""
        if not self.storage_path.exists():
            logger.info(f"用户数据文件不存在，将在 {self.storage_path} 创建")
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for user_data in data.get("users", []):
                user = User.from_dict(user_data)
                self._users[user.id] = user

            self._sessions = data.get("sessions", {})

            # 清理过期 session
            self._cleanup_sessions()

            logger.info(f"已加载 {len(self._users)} 个用户，{len(self._sessions)} 个有效会话")
        except Exception as e:
            logger.error(f"加载用户数据失败: {e}")

    def _save_data(self):
        """保存用户数据到文件"""
        try:
            data = {
                "users": [user.to_dict() for user in self._users.values()],
                "sessions": self._sessions,
            }

            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"保存用户数据失败: {e}")

    def _cleanup_sessions(self):
        """清理过期 session"""
        now = time.time()
        expired = [token for token, session in self._sessions.items()
                   if session.get("expires_at", 0) < now]
        for token in expired:
            del self._sessions[token]
        if expired:
            logger.info(f"清理了 {len(expired)} 个过期会话")

    # ==================== 微信扫码登录 (OAuth2) ====================

    def create_login_qrcode(self) -> Dict:
        """
        创建登录二维码

        如果微信已配置: 返回微信授权 URL (用于生成二维码)
        如果未配置: 返回模拟登录信息

        返回:
            {
                "qrcode_id": "唯一标识",
                "qrcode_url": "二维码内容URL（用户扫码后跳转）",
                "expires_at": 过期时间戳,
                "mode": "wechat" | "simulate"
            }
        """
        qrcode_id = secrets.token_urlsafe(16)
        expires_at = time.time() + 300  # 5 分钟有效期

        if self.wechat_enabled:
            state = f"{self.wechat_config['state_prefix']}{qrcode_id}"
            qrcode_url = (
                f"https://open.weixin.qq.com/connect/qrconnect"
                f"?appid={self.wechat_config['app_id']}"
                f"&redirect_uri={self.wechat_config['redirect_uri']}"
                f"&response_type=code"
                f"&scope={self.wechat_config['scope']}"
                f"&state={state}#wechat_redirect"
            )
            mode = "wechat"
        else:
            qrcode_url = f"wechat://login?scene={qrcode_id}"
            mode = "simulate"

        self._login_qrcodes[qrcode_id] = {
            "qrcode_id": qrcode_id,
            "qrcode_url": qrcode_url,
            "expires_at": expires_at,
            "status": "pending",
            "mode": mode,
            "user_info": None,
        }

        logger.info(f"创建登录二维码: {qrcode_id} (mode={mode})")
        return {
            "qrcode_id": qrcode_id,
            "qrcode_url": qrcode_url,
            "expires_at": expires_at,
            "mode": mode,
            "enabled": self.wechat_enabled,
        }

    async def create_mini_program_qrcode(self, qrcode_id: str) -> Optional[bytes]:
        """
        生成微信小程序码（用于扫码登录）

        调用微信 API 生成小程序码，用户扫码后直接跳转到小程序的 login 页面
        并携带 scene 参数（即 qrcode_id）

        Args:
            qrcode_id: 登录会话的唯一标识

        Returns:
            小程序码图片的二进制数据（PNG格式），失败返回 None
        """
        if not self.wechat_enabled:
            logger.warning("微信未配置，无法生成小程序码")
            return None

        try:
            access_token = await self._get_wechat_access_token()
            if not access_token:
                return None

            url = f"https://api.weixin.qq.com/wxa/getwxacodeunlimit?access_token={access_token}"

            payload = {
                "scene": qrcode_id,
                "page": "pages/login/login",
                "width": 430,
                "auto_color": False,
                "line_color": {"r": 7, "g": 193, "b": 96},
                "is_hyaline": False
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                if "image" in content_type:
                    logger.info(f"小程序码生成成功: {qrcode_id}")
                    return response.content
                else:
                    error_data = response.json()
                    err_code = error_data.get("errcode", "unknown")
                    err_msg = error_data.get("errmsg", "未知错误")
                    logger.error(f"生成小程序码失败: {err_code} - {err_msg}")
                    return None
            else:
                logger.error(f"请求小程序码API失败: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"生成小程序码异常: {e}")
            return None

    async def _get_wechat_access_token(self) -> Optional[str]:
        """
        获取微信小程序的 access_token

        Returns:
            access_token 字符串，失败返回 None
        """
        app_id = self.wechat_config.get("app_id", "")
        app_secret = self.wechat_config.get("app_secret", "")

        if not app_id or not app_secret:
            logger.error("AppID 或 AppSecret 未配置")
            return None

        token_url = "https://api.weixin.qq.com/cgi-bin/token"
        params = {
            "grant_type": "client_credential",
            "appid": app_id,
            "secret": app_secret
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(token_url, params=params)

            if response.status_code == 200:
                data = response.json()
                if "access_token" in data:
                    return data["access_token"]
                else:
                    err_code = data.get("errcode", "unknown")
                    err_msg = data.get("errmsg", "未知错误")
                    logger.error(f"获取 access_token 失败: {err_code} - {err_msg}")
                    return None
            else:
                logger.error(f"请求 access_token API 失败: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"获取 access_token 异常: {e}")
            return None

    async def wechat_callback(self, code: str, state: str) -> Tuple[Optional[User], Optional[str], str]:
        """
        微信 OAuth2 回调处理

        用 authorization code 换取 access_token 和用户信息

        Args:
            code: 微信返回的授权码
            state: 防止 CSRF 的状态参数

        Returns:
            (user, session_token, error_message)
        """
        if not self.wechat_enabled:
            return None, None, "微信登录未启用"

        # 验证 state 参数
        prefix = self.wechat_config.get("state_prefix", "exo_")
        if not state.startswith(prefix):
            return None, None, "无效的 state 参数"

        qrcode_id = state[len(prefix):]
        if qrcode_id not in self._login_qrcodes:
            return None, None, "无效的登录请求"

        try:
            # Step 1: 用 code 换取 access_token
            token_url = "https://api.weixin.qq.com/sns/oauth2/access_token"
            params = {
                "appid": self.wechat_config["app_id"],
                "secret": self.wechat_config["app_secret"],
                "code": code,
                "grant_type": "authorization_code",
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(token_url, params=params)
                token_data = response.json()

            if "errcode" in token_data:
                err_msg = token_data.get("errmsg", "未知错误")
                logger.error(f"微信获取 access_token 失败: {err_msg}")
                return None, None, f"微信认证失败: {err_msg}"

            access_token = token_data["access_token"]
            openid = token_data["openid"]
            union_id = token_data.get("union_id", "")

            # Step 2: 用 access_token 获取用户信息
            user_url = "https://api.weixin.qq.com/sns/userinfo"
            user_params = {
                "access_token": access_token,
                "openid": openid,
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(user_url, params=user_params)
                user_data = response.json()

            if "errcode" in user_data:
                # 可能是未关注公众号，使用基本信息
                logger.warning(f"获取用户详细信息失败，使用基本信息: {user_data.get('errmsg')}")
                nickname = f"微信用户_{openid[:8]}"
                avatar = ""
            else:
                nickname = user_data.get("nickname", f"微信用户_{openid[:8]}")
                avatar = user_data.get("headimgurl", "")

            # Step 3: 查找或创建用户
            user = await self._find_or_create_wechat_user(
                union_id=union_id,
                openid=openid,
                nickname=nickname,
                avatar=avatar,
            )

            # Step 4: 创建 session
            session_token = self._create_session(user.id)

            # 更新二维码状态
            if qrcode_id in self._login_qrcodes:
                self._login_qrcodes[qrcode_id]["status"] = "confirmed"
                self._login_qrcodes[qrcode_id]["user_info"] = {
                    "nickname": nickname,
                    "avatar": avatar,
                }

            logger.info(f"微信用户 {nickname} 登录成功")
            return user, session_token, ""

        except httpx.TimeoutException:
            return None, None, "连接微信服务器超时"
        except Exception as e:
            logger.error(f"微信登录回调处理失败: {e}")
            return None, None, f"登录处理失败: {str(e)}"

    async def _find_or_create_wechat_user(
        self,
        union_id: str,
        openid: str,
        nickname: str,
        avatar: str,
    ) -> User:
        """查找或创建微信用户"""
        # 优先通过 union_id 查找
        if union_id:
            user = self._find_user_by_union_id(union_id)
            if user:
                # 更新 openid 和用户信息
                if not user.openid:
                    user.openid = openid
                if nickname and nickname != user.nickname:
                    user.nickname = nickname
                if avatar and avatar != user.avatar:
                    user.avatar = avatar
                user.last_login_at = time.time()
                self._save_data()
                return user

        # 通过 openid 查找
        for u in self._users.values():
            if u.openid == openid:
                if union_id and not u.union_id:
                    u.union_id = union_id
                u.last_login_at = time.time()
                self._save_data()
                return u

        # 创建新用户
        return self._create_user(
            union_id=union_id or f"wx_{openid}",
            nickname=nickname,
            avatar=avatar,
        )

    def get_qrcode_status(self, qrcode_id: str) -> Optional[Dict]:
        """获取二维码状态"""
        qrcode = self._login_qrcodes.get(qrcode_id)
        if not qrcode:
            return None

        # 检查是否过期
        if qrcode["expires_at"] < time.time() and qrcode["status"] == "pending":
            qrcode["status"] = "expired"

        return {
            "qrcode_id": qrcode_id,
            "status": qrcode["status"],
            "user_info": qrcode.get("user_info"),
        }

    def simulate_scan_qrcode(self, qrcode_id: str, user_info: Dict) -> bool:
        """
        模拟用户扫描二维码（用于测试）

        实际场景中，这是由微信服务器回调触发的
        """
        qrcode = self._login_qrcodes.get(qrcode_id)
        if not qrcode:
            return False

        if qrcode["status"] != "pending":
            return False

        qrcode["status"] = "scanned"
        qrcode["user_info"] = user_info
        logger.info(f"二维码 {qrcode_id} 已被扫描")
        return True

    def confirm_login(self, qrcode_id: str) -> Optional[str]:
        """
        确认登录（用户点击确认后）

        返回:
            session token，如果登录成功
        """
        qrcode = self._login_qrcodes.get(qrcode_id)
        if not qrcode:
            return None

        if qrcode["status"] != "scanned":
            return None

        user_info = qrcode.get("user_info", {})
        union_id = user_info.get("union_id", "")

        # 查找或创建用户
        user = self._find_user_by_union_id(union_id)
        if not user:
            user = self._create_user(
                union_id=union_id,
                nickname=user_info.get("nickname", "微信用户"),
                avatar=user_info.get("avatar", ""),
            )

        # 更新登录时间
        user.last_login_at = time.time()
        self._save_data()

        # 创建 session
        token = self._create_session(user.id)

        qrcode["status"] = "confirmed"
        logger.info(f"用户 {user.nickname} 登录成功")

        return token

    # ==================== 微信小程序登录 ====================

    def _get_mini_openid(self, code: str) -> Optional[str]:
        """
        调用微信 jscode2session API 获取真实 openid
        
        开发模式（未配置小程序 appid）下用 code 的 md5 作为 openid
        
        Args:
            code: 小程序登录 code
            
        Returns:
            openid 字符串，失败返回 None
        """
        mini_appid = self.wechat_config.get("mini_appid", "")
        mini_secret = self.wechat_config.get("mini_secret", "")
        
        if mini_appid and mini_secret:
            url = "https://api.weixin.qq.com/sns/jscode2session"
            params = {
                "appid": mini_appid,
                "secret": mini_secret,
                "js_code": code,
                "grant_type": "authorization_code"
            }
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    if "openid" in data:
                        logger.info(f"微信小程序 openid 获取成功: {data['openid'][:8]}...")
                        return data["openid"]
                    else:
                        err_code = data.get("errcode", "unknown")
                        err_msg = data.get("errmsg", "未知错误")
                        logger.error(f"微信 jscode2session 失败: {err_code} - {err_msg}")
                else:
                    logger.error(f"微信 jscode2session 请求失败: HTTP {resp.status_code}")
            except Exception as e:
                logger.error(f"微信 jscode2session 异常: {e}")
        else:
            logger.info("小程序 AppID 未配置，使用开发模式（code md5 作为 openid）")
        
        return f"mini_{hashlib.md5(code.encode()).hexdigest()[:16]}"

    def wechat_mini_login(
        self,
        code: str,
        nickname: str = "微信用户",
        avatar: str = ""
    ) -> Tuple[Optional[str], Optional[User]]:
        """
        微信小程序登录

        根据 openid 识别用户，返回：
        - session token
        - 用户基本信息
        - API Key
        
        Args:
            code: 小程序登录code
            nickname: 用户昵称
            avatar: 用户头像

        Returns:
            (session_token, user)
        """
        openid = self._get_mini_openid(code)
        if not openid:
            return None, None

        user = self._find_user_by_openid(openid)
        if not user:
            user = self._find_user_by_union_id(openid)
        
        if not user:
            user = self._create_user(
                union_id=openid,
                openid=openid,
                nickname=nickname,
                avatar=avatar,
            )
        else:
            user.last_login_at = time.time()
            if not user.openid:
                user.openid = openid
            if nickname and nickname != "微信用户":
                user.nickname = nickname
            if avatar:
                user.avatar = avatar
            self._save_data()

        token = self._create_session(user.id)
        logger.info(f"小程序用户 {nickname} 登录成功 (openid: {openid[:8]}...)")

        return token, user

    def generate_website_credentials(self, user_id: str) -> Tuple[str, str]:
        """
        为用户生成网站登录账号和密码

        Args:
            user_id: 用户ID

        Returns:
            (account, password) 账号和密码
        """
        user = self._users.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        timestamp = int(time.time())
        
        account = f"wx_{user.nickname[:4]}_{timestamp}"
        
        password = secrets.token_urlsafe(12)

        return account, password

    def update_user_website_credentials(
        self,
        user_id: str,
        account: str,
        password: str
    ) -> None:
        """
        更新用户的网站登录凭据

        Args:
            user_id: 用户ID
            account: 账号
            password: 密码（明文，会自动哈希存储）
        """
        user = self._users.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        user.website_account = account
        user.website_password_hash = self.hash_password(password)
        user.website_password = password
        
        self._save_data()
        
        logger.info(f"用户 {user.nickname} 的网站凭据已更新")

    def get_or_create_api_key(self, user_id: str) -> str:
        """
        获取或创建用户的 API Key

        Args:
            user_id: 用户ID

        Returns:
            API Key 字符串
        """
        try:
            from api_key_manager import get_api_key_manager
            
            key_mgr = get_api_key_manager()
            
            existing_keys = key_mgr.get_user_keys(user_id)
            
            if existing_keys:
                return existing_keys[0].key
            
            new_key = key_mgr.create_key(
                name=f"用户默认 Key",
                user_id=user_id,
                permissions=["*"],
                allowed_models=["*"]
            )
            
            if new_key:
                return new_key.key
                
        except Exception as e:
            logger.warning(f"API Key 管理器不可用: {e}")
        
        fallback_key = f"exo_sk_{secrets.token_urlsafe(32)}"
        
        return fallback_key

    # ==================== 管理员登录 ====================

    @staticmethod
    def hash_password(password: str) -> str:
        """使用 SHA-256 + 盐值哈希密码"""
        salted = f"{PASSWORD_SALT}:{password}"
        return hashlib.sha256(salted.encode('utf-8')).hexdigest()

    def check_login_rate_limit(self, client_ip: str) -> Tuple[bool, str]:
        """
        检查登录速率限制

        Returns:
            (allowed, error_message)
        """
        now = time.time()
        attempts = self._login_attempts.get(client_ip)

        if not attempts:
            return True, ""

        # 检查是否在锁定期内
        locked_until = attempts.get("locked_until", 0)
        if now < locked_until:
            remaining = int(locked_until - now)
            minutes = remaining // 60 + 1
            return False, f"登录尝试次数过多，请 {minutes} 分钟后重试"

        # 清理过期的尝试记录
        if now - attempts.get("last_attempt", 0) > LOGIN_LOCKOUT_TIME:
            del self._login_attempts[client_ip]
            return True, ""

        return True, ""

    def record_login_attempt(self, client_ip: str, success: bool):
        """记录登录尝试结果"""
        now = time.time()
        attempts = self._login_attempts.get(client_ip)

        if not attempts:
            attempts = {"count": 0, "last_attempt": now, "locked_until": 0}
            self._login_attempts[client_ip] = attempts

        attempts["last_attempt"] = now

        if success:
            # 登录成功，重置计数
            attempts["count"] = 0
            attempts["locked_until"] = 0
        else:
            # 登录失败，增加计数
            attempts["count"] += 1
            if attempts["count"] >= MAX_LOGIN_ATTEMPTS:
                attempts["locked_until"] = now + LOGIN_LOCKOUT_TIME
                logger.warning(f"IP {client_ip} 已被锁定 {LOGIN_LOCKOUT_TIME} 秒")

    def admin_login(self, username: str, password: str, client_ip: str = "unknown") -> Tuple[Optional[str], str]:
        """
        管理员账号密码登录

        使用哈希密码验证，支持登录失败次数限制

        默认管理员账号: admin / admin123 (首次使用后建议修改)

        Returns:
            (session_token, error_message)
        """
        # 检查速率限制
        allowed, error_msg = self.check_login_rate_limit(client_ip)
        if not allowed:
            self.record_login_attempt(client_ip, False)
            return None, error_msg

        # 验证凭据（使用密码哈希）
        default_password_hash = self.hash_password("admin123")

        if username == "admin" and self.hash_password(password) == default_password_hash:
            # 登录成功
            self.record_login_attempt(client_ip, True)

            admin_user = None
            for user in self._users.values():
                if user.role == "admin":
                    admin_user = user
                    break

            if not admin_user:
                admin_user = self._create_user(
                    union_id="admin",
                    nickname="管理员",
                    avatar="",
                    role="admin"
                )

            admin_user.last_login_at = time.time()
            self._save_data()

            token = self._create_session(admin_user.id)
            logger.info(f"管理员登录成功 (IP: {client_ip})")
            return token, ""

        # 登录失败
        self.record_login_attempt(client_ip, False)
        logger.warning(f"管理员登录失败 (IP: {client_ip}, 用户名: {username})")
        return None, "用户名或密码错误"

    # ==================== Session 管理 ====================

    def _create_session(self, user_id: str) -> str:
        """创建新的 session"""
        token = secrets.token_urlsafe(32)
        now = time.time()

        self._sessions[token] = {
            "user_id": user_id,
            "created_at": now,
            "expires_at": now + SESSION_EXPIRY,
        }
        self._save_data()

        return token

    def validate_session(self, token: str) -> Optional[User]:
        """验证 session token 并返回用户信息"""
        if not token:
            return None

        session = self._sessions.get(token)
        if not session:
            return None

        # 检查是否过期
        if session.get("expires_at", 0) < time.time():
            del self._sessions[token]
            self._save_data()
            return None

        user_id = session.get("user_id")
        return self._users.get(user_id)

    def logout(self, token: str) -> bool:
        """注销 session"""
        if token in self._sessions:
            del self._sessions[token]
            self._save_data()
            return True
        return False

    # ==================== 用户管理 ====================

    def _find_user_by_union_id(self, union_id: str) -> Optional[User]:
        """通过 union_id 查找用户"""
        for user in self._users.values():
            if user.union_id == union_id:
                return user
        return None

    def _find_user_by_openid(self, openid: str) -> Optional[User]:
        """通过 openid 查找用户"""
        for user in self._users.values():
            if user.openid == openid:
                return user
        return None

    def _create_user(
        self,
        union_id: str,
        nickname: str,
        avatar: str = "",
        role: str = "user",
        openid: str = ""
    ) -> User:
        """创建新用户"""
        import uuid
        user = User(
            id=str(uuid.uuid4()),
            union_id=union_id,
            openid=openid,
            nickname=nickname,
            avatar=avatar,
            role=role,
            created_at=time.time(),
            last_login_at=time.time(),
        )
        self._users[user.id] = user
        self._save_data()
        logger.info(f"创建新用户: {nickname} (role={role})")
        return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """通过 ID 获取用户"""
        return self._users.get(user_id)

    def list_users(self) -> List[Dict]:
        """列出所有用户"""
        return [user.to_public_dict() for user in self._users.values()]

    def set_user_role(self, user_id: str, role: str) -> bool:
        """设置用户角色"""
        user = self._users.get(user_id)
        if not user:
            return False

        if role not in ("user", "admin"):
            return False

        user.role = role
        self._save_data()
        return True

    def disable_user(self, user_id: str) -> bool:
        """禁用用户"""
        user = self._users.get(user_id)
        if not user or user.role == "admin":
            return False

        user.is_disabled = True
        self._save_data()
        logger.info(f"禁用用户: {user.nickname} ({user_id})")
        return True

    def enable_user(self, user_id: str) -> bool:
        """启用用户"""
        user = self._users.get(user_id)
        if not user:
            return False

        user.is_disabled = False
        self._save_data()
        logger.info(f"启用用户: {user.nickname} ({user_id})")
        return True

    def delete_user(self, user_id: str) -> bool:
        """删除用户（非管理员）"""
        user = self._users.get(user_id)
        if not user or user.role == "admin":
            return False

        del self._users[user_id]

        # 删除相关 session
        to_remove = [t for t, u in self._sessions.items() if u == user_id]
        for t in to_remove:
            del self._sessions[t]

        self._save_data()
        logger.info(f"删除用户: {user.nickname} ({user_id})")
        return True

    def get_user_stats(self) -> Dict:
        """获取用户统计信息"""
        total = len(self._users)
        admins = sum(1 for u in self._users.values() if u.role == "admin")
        disabled = sum(1 for u in self._users.values() if getattr(u, 'is_disabled', False))
        active_today = sum(
            1 for u in self._users.values()
            if u.last_login_at and (time.time() - u.last_login_at) < 86400
        )

        return {
            "total_users": total,
            "admin_count": admins,
            "user_count": total - admins,
            "disabled_count": disabled,
            "active_today": active_today,
        }

    # ==================== 权限检查 ====================

    def check_permission(self, token: str, required_role: str = "user") -> Optional[User]:
        """
        检查用户权限

        Args:
            token: session token
            required_role: 需要的角色 ("user" 或 "admin")

        Returns:
            用户对象，如果权限足够；None 如果未登录或权限不足
        """
        user = self.validate_session(token)
        if not user:
            return None

        # admin 拥有所有权限
        if user.role == "admin":
            return user

        # user 只能访问 user 级别资源
        if required_role == "user":
            return user

        return None


# 全局认证管理器实例
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """获取全局认证管理器实例"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
