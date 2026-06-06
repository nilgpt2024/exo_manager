"""
EXO Cluster Manager - 认证路由
================================

路由结构:
----------
(无前缀)        - 普通用户接口 (公网开放)
  /login/qrcode          - 创建登录二维码
  /login/qrcode/{id}     - 查询二维码状态
  /login/qrcode/{id}/scan    - 扫描二维码
  /login/qrcode/{id}/confirm - 确认登录
  /login/wechat/callback - 微信 OAuth2 回调
  /login/wechat/status   - 微信登录状态
  /logout                - 退出登录
  /me                    - 获取当前用户信息

/admin/*         - 管理员接口 (可限制内网)
  /admin/login               - 管理员账号密码登录
  /admin/users               - 用户列表
  /admin/users/{id}/role     - 修改角色
  /admin/users/{id}/disable  - 禁用用户
  /admin/users/{id}/enable   - 启用用户
  /admin/users/{id}          - 删除用户
  /admin/logout              - 管理员退出登录

Nginx 配置示例:
--------------
location /admin/ {
    allow 192.168.0.0/16;
    allow 10.0.0.0/8;
    deny all;
    proxy_pass http://manager:8080;
}
"""

import logging
import io
from typing import Dict, Optional

from fastapi import APIRouter, Cookie, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from auth_manager import get_auth_manager, AuthManager

logger = logging.getLogger(__name__)

# ==================== 路由器定义 ====================

# 普通用户路由 (无前缀)
auth_router = APIRouter()

# 管理员路由 (/admin/*)
admin_router = APIRouter(prefix="/admin")

# ==================== Pydantic 模型 ====================

class AdminLoginRequest(BaseModel):
    username: str = Field(..., description="管理员账号")
    password: str = Field(..., description="管理员密码")


class ScanQrcodeRequest(BaseModel):
    nickname: str = Field("微信用户", description="用户昵称")
    avatar: str = Field("", description="用户头像URL")


class WechatMiniLoginRequest(BaseModel):
    code: str = Field(..., description="微信小程序登录code")
    nickname: str = Field("微信用户", description="用户昵称")
    avatar: str = Field("", description="用户头像URL")


class ConfirmQrcodeRequest(BaseModel):
    code: str = Field("", description="微信小程序登录code")


# ==================== 辅助函数 ====================

def get_session_token(request: Request) -> Optional[str]:
    token = request.cookies.get("session_token")
    if token:
        return token
    return None


# ================================================================
#  普通用户路由 (无前缀) - 公网开放
# ================================================================

@auth_router.post("/login/qrcode")
async def create_qrcode():
    """创建微信登录二维码"""
    auth_mgr = get_auth_manager()
    qrcode_info = auth_mgr.create_login_qrcode()
    return {"success": True, "data": qrcode_info}


@auth_router.get("/login/qrcode/{qrcode_id}/image")
async def get_mini_program_qrcode(qrcode_id: str):
    """
    获取小程序码图片

    返回 PNG 格式的小程序码图片，用户扫码后跳转到小程序 login 页面
    """
    auth_mgr = get_auth_manager()

    qrcode_data = auth_mgr._login_qrcodes.get(qrcode_id)
    if not qrcode_data:
        raise HTTPException(status_code=404, detail="QR code not found")

    if time.time() > qrcode_data["expires_at"]:
        raise HTTPException(status_code=410, detail="QR code expired")

    qrcode_image = await auth_mgr.create_mini_program_qrcode(qrcode_id)

    if not qrcode_image:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate mini program QR code"
        )

    return StreamingResponse(
        io.BytesIO(qrcode_image),
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@auth_router.get("/login/qrcode/{qrcode_id}")
async def get_qrcode_status(qrcode_id: str):
    """查询二维码登录状态"""
    auth_mgr = get_auth_manager()
    status = auth_mgr.get_qrcode_status(qrcode_id)
    if not status:
        raise HTTPException(status_code=404, detail="QR code not found")
    return {"success": True, "data": status}


@auth_router.post("/login/qrcode/{qrcode_id}/scan")
async def scan_qrcode(qrcode_id: str, request: ScanQrcodeRequest):
    """扫描二维码 (微信端调用)"""
    auth_mgr = get_auth_manager()
    result = auth_mgr.scan_qrcode(
        qrcode_id,
        nickname=request.nickname,
        avatar=request.avatar
    )
    if not result:
        raise HTTPException(status_code=404, detail="QR code not found or expired")
    return {"success": True}


@auth_router.post("/login/qrcode/{qrcode_id}/confirm")
async def confirm_qrcode_login(qrcode_id: str, request: ConfirmQrcodeRequest, response: Response):
    """确认登录 (微信端调用)"""
    auth_mgr = get_auth_manager()
    
    if request.code:
        # 小程序端带code，创建用户并登录
        token, user = auth_mgr.wechat_mini_login(
            code=request.code,
            nickname=f"wx_user_{qrcode_id[-6:]}",
            avatar=""
        )
    else:
        # 使用已扫码的用户信息
        token = auth_mgr.confirm_qrcode_login(qrcode_id)
        user = None
    
    if not token:
        raise HTTPException(status_code=404, detail="QR code not found or not scanned")
    
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=7 * 24 * 3600,
    )
    result = {"success": True}
    if user:
        result["data"] = {"user": user}
    return result


@auth_router.post("/login/wechat")
async def wechat_mini_login(request: WechatMiniLoginRequest, response: Response):
    """微信小程序登录"""
    auth_mgr = get_auth_manager()
    token, user = auth_mgr.wechat_mini_login(
        code=request.code,
        nickname=request.nickname,
        avatar=request.avatar
    )
    
    if not token:
        raise HTTPException(status_code=500, detail="Login failed")
    
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=7 * 24 * 3600,
    )
    
    return {"success": True, "data": {"user": user}}


@auth_router.post("/login/wechat/mini")
async def wechat_mini_login_with_account(request: WechatMiniLoginRequest, response: Response):
    """
    微信小程序登录
    
    根据 openid 识别用户，返回：
    - session token
    - 用户基本信息
    - API Key
    """
    auth_mgr = get_auth_manager()
    
    token, user = auth_mgr.wechat_mini_login(
        code=request.code,
        nickname=request.nickname,
        avatar=request.avatar
    )
    
    if not token:
        raise HTTPException(status_code=500, detail="Login failed")
    
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=7 * 24 * 3600,
    )
    
    user_data = user.to_dict() if hasattr(user, 'to_dict') else {
        'id': user.id,
        'nickname': user.nickname,
        'avatar': getattr(user, 'avatar', ''),
        'union_id': user.union_id
    }
    
    try:
        api_key = auth_mgr.get_or_create_api_key(user.id)
        user_data['api_key'] = api_key
    except Exception as e:
        print(f"API Key 生成失败: {e}")
        user_data['api_key'] = ''
    
    return {
        "success": True,
        "data": {
            "token": token,
            "user": user_data
        }
    }


@auth_router.get("/v1/user/api-info")
async def get_user_api_info(request: Request):
    """获取用户的 API 配置信息"""
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired")
    
    api_key = auth_mgr.get_or_create_api_key(user.id)
    
    base_url = str(request.base_url).rstrip('/')
    
    return {
        "success": True,
        "data": {
            "api_key": api_key,
            "api_url": f"{base_url}/v1"
        }
    }


@auth_router.post("/api/wechat/mini/qr-confirm")
async def wechat_mini_qr_confirm(request: Request):
    """
    小程序扫码确认网站登录
    
    当用户在网站上看到二维码，用小程序扫描后，
    小程序会调用此接口确认登录，完成网站的认证流程
    """
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired")
    
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    qr_id = body.get('qr_id')
    if not qr_id:
        raise HTTPException(status_code=400, detail="Missing qr_id")
    
    user_info = body.get('user_info', {})
    
    import time
    import uuid
    
    session_token = f"website_{uuid.uuid4().hex[:32]}_{int(time.time())}"
    
    return {
        "success": True,
        "data": {
            "message": "Website login confirmed",
            "session_token": session_token,
            "user_id": user.id,
            "nickname": user_info.get('nickname', user.nickname),
            "avatar": user_info.get('avatar', ''),
            "qr_id": qr_id
        }
    }


@auth_router.put("/v1/user/profile")
async def update_user_profile(request: Request):
    """
    更新用户个人信息（昵称、头像）
    
    用于小程序端完善用户资料
    """
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired")
    
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    nickname = body.get('nickname')
    avatar = body.get('avatar')
    
    if nickname:
        user.nickname = nickname
    if avatar:
        user.avatar = avatar
    
    auth_mgr._save_data()
    
    return {
        "success": True,
        "data": {
            "nickname": user.nickname,
            "avatar": getattr(user, 'avatar', '')
        }
    }


@auth_router.post("/v1/user/checkin")
async def daily_checkin(request: Request):
    """
    每日签到
    
    签到后获得免费 Token 额度奖励
    - 每天只能签到一次
    - 连续签到有额外奖励
    """
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired")
    
    import time
    from datetime import datetime
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    checkin_data = getattr(user, 'checkin_data', {})
    
    if not isinstance(checkin_data, dict):
        checkin_data = {}
    
    last_checkin = checkin_data.get('last_date')
    
    if last_checkin == today:
        return {
            "success": False,
            "detail": "今天已经签到了，明天再来吧！",
            "data": {
                "checked_today": True,
                "consecutive_days": checkin_data.get('consecutive_days', 0)
            }
        }
    
    consecutive = checkin_data.get('consecutive_days', 0)
    yesterday = datetime.now().strftime('%Y-%m-%d') 
    
    from datetime import timedelta
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    if last_checkin == yesterday_str:
        consecutive += 1
    else:
        consecutive = 1
    
    base_reward = 1000
    bonus = min(consecutive - 1, 7) * 200
    total_reward = base_reward + bonus
    
    user.quota = getattr(user, 'quota', 0) + total_reward
    
    user.checkin_data = {
        'last_date': today,
        'consecutive_days': consecutive,
        'total_checkins': checkin_data.get('total_checkins', 0) + 1,
        'last_reward': total_reward,
        'history': checkin_data.get('history', []) + [{
            'date': today,
            'reward': total_reward,
            'consecutive': consecutive
        }]
    }
    
    auth_mgr._save_data()
    
    return {
        "success": True,
        "data": {
            "message": f"签到成功！获得 {total_reward} tokens",
            "reward": total_reward,
            "consecutive_days": consecutive,
            "new_quota": user.quota,
            "bonus_text": f"连续{consecutive}天，额外+{bonus}tokens" if bonus else ""
        }
    }


@auth_router.get("/v1/user/checkin/status")
async def get_checkin_status(request: Request):
    """获取签到状态"""
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    
    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired")
    
    from datetime import datetime, timedelta
    
    today = datetime.now().strftime('%Y-%m-%d')
    checkin_data = getattr(user, 'checkin_data', {})
    
    if not isinstance(checkin_data, dict):
        checkin_data = {}
    
    last_checkin = checkin_data.get('last_date', '')
    checked_today = last_checkin == today
    consecutive = checkin_data.get('consecutive_days', 0) if checked_today else 0
    
    return {
        "success": True,
        "data": {
            "checked_today": checked_today,
            "consecutive_days": consecutive,
            "total_checkins": checkin_data.get('total_checkins', 0),
            "last_reward": checkin_data.get('last_reward', 0),
            "last_date": last_checkin
        }
    }


# 微信 OAuth2 集成 (H5/公众号)
@auth_router.get("/login/wechat/oauth")
async def wechat_login_redirect():
    """重定向到微信 OAuth 授权页"""
    raise HTTPException(status_code=501, detail="WeChat OAuth2 requires configuration")


@auth_router.get("/login/wechat/callback")
async def wechat_login_callback(code: Optional[str] = None, state: Optional[str] = None):
    """微信 OAuth 回调"""
    if not code:
        raise HTTPException(status_code=400, detail="Code is required")
    raise HTTPException(status_code=501, detail="WeChat OAuth2 requires configuration")


@auth_router.get("/login/wechat/status")
async def wechat_login_status_check():
    """检查微信登录状态"""
    raise HTTPException(status_code=501, detail="WeChat OAuth2 requires configuration")


@auth_router.post("/logout")
async def logout(request: Request, response: Response):
    """退出登录"""
    token = get_session_token(request)
    if token:
        auth_mgr = get_auth_manager()
        auth_mgr.logout(token)
    response.delete_cookie(key="session_token")
    return {"success": True, "message": "Logout successful"}


@auth_router.get("/me")
async def get_current_user(request: Request):
    """获取当前登录用户信息"""
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")

    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session expired")
    return {"success": True, "data": user}


# ================================================================
#  管理员路由 (/admin/*) - 可限制内网访问
# ================================================================

@admin_router.post("/login")
async def admin_login(request_data: AdminLoginRequest, request: Request, response: Response):
    """
    管理员账号密码登录
    
    默认账号: admin / admin123
    支持: 密码哈希验证、登录失败次数限制(5次锁定15分钟)、IP记录
    """
    auth_mgr = get_auth_manager()

    client_ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    token, error_msg = auth_mgr.admin_login(
        request_data.username,
        request_data.password,
        client_ip=client_ip
    )

    if not token:
        if "锁定" in error_msg or "重试" in error_msg:
            raise HTTPException(status_code=429, detail=error_msg)
        raise HTTPException(status_code=401, detail=error_msg)

    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=7 * 24 * 3600,
    )

    logger.info(f"管理员登录成功: IP={client_ip}, user={request_data.username}")

    return {
        "success": True,
        "message": "Login successful",
        "data": {
            "token": token,
            "role": "admin",
        },
    }


@admin_router.post("/logout")
async def admin_logout(request: Request, response: Response):
    """管理员退出登录"""
    token = get_session_token(request)
    if token:
        auth_mgr = get_auth_manager()
        auth_mgr.logout(token)
    response.delete_cookie(key="session_token")
    return {"success": True, "message": "Logout successful"}


@admin_router.get("/users")
async def admin_list_users(request: Request):
    """管理员获取用户列表"""
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    users = auth_mgr.list_users()
    return {"success": True, "data": users}


@admin_router.put("/users/{user_id}/role")
async def admin_set_user_role(user_id: str, role: str, request: Request):
    """管理员修改用户角色"""
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    success = auth_mgr.set_user_role(user_id, role)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True, "message": "Role updated"}


@admin_router.post("/users/{user_id}/disable")
async def admin_disable_user(user_id: str, request: Request):
    """管理员禁用用户"""
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    success = auth_mgr.disable_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True, "message": "User disabled"}


@admin_router.post("/users/{user_id}/enable")
async def admin_enable_user(user_id: str, request: Request):
    """管理员启用用户"""
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    success = auth_mgr.enable_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True, "message": "User enabled"}


@admin_router.delete("/users/{user_id}")
async def admin_delete_user(user_id: str, request: Request):
    """管理员删除用户"""
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    success = auth_mgr.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True, "message": "User deleted"}
