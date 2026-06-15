"""
FRP Server 管理 API 路由
=======================

提供 RESTful 接口管理 FRP Server:
- 服务启停、状态查询
- 配置更新
- 客户端注册与管理
- 用户连接信息 (含 token/secretKey 的启动命令)
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Body, Request
from pydantic import BaseModel, Field

from frp_server_manager import get_frp_server_manager, FRP_VERSION, FRP_BASE_URL

logger = logging.getLogger("FRPRoutes")

router = APIRouter(prefix="/api/frps", tags=["FRP Server"])


# ==================== 请求模型 ====================

class StartRequest(BaseModel):
    """启动请求"""
    bind_port: Optional[int] = Field(None, ge=1, le=65535)
    token: Optional[str] = None


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    bind_port: Optional[int] = Field(None, ge=1, le=65535)
    token: Optional[str] = None
    dashboard_port: Optional[int] = Field(None, ge=0, le=65535)
    enable_xtcp: Optional[bool] = None


class ClientRegisterRequest(BaseModel):
    """客户端注册请求"""
    node_id: str = Field(..., min_length=1, max_length=64)
    local_port: int = Field(50051, ge=1, le=65535)
    node_name: str = ""
    enable_p2p: bool = True


class UserConnectionRequest(BaseModel):
    """用户连接请求"""
    node_id: Optional[str] = Field(None, description="节点 ID")
    local_port: int = Field(50051, ge=1, le=65535)
    server_addr: Optional[str] = Field(None)


# ==================== 服务状态接口 ====================

@router.get("/status")
async def get_frps_status():
    """获取 frps 服务状态"""
    manager = get_frp_server_manager()
    return manager.get_status()


@router.post("/start")
async def start_frps(request: Optional[StartRequest] = None):
    """启动 frps 服务"""
    manager = get_frp_server_manager()

    if request and request.bind_port:
        manager.update_config(bind_port=request.bind_port)
    if request and request.token:
        manager.update_config(token=request.token)

    success, message = await manager.start_server()
    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {
        "success": True,
        "message": message,
        "status": manager.get_status()["status"]
    }


@router.post("/stop")
async def stop_frps():
    """停止 frps 服务"""
    manager = get_frp_server_manager()
    success, message = await manager.stop_server()
    if not success:
        raise HTTPException(status_code=500, detail=message)

    return {"success": True, "message": message}


@router.post("/restart")
async def restart_frps():
    """重启 frps 服务"""
    manager = get_frp_server_manager()
    stop_success, stop_msg = await manager.stop_server()
    if not stop_success:
        raise HTTPException(status_code=500, detail=f"停止失败: {stop_msg}")

    start_success, start_msg = await manager.start_server()
    if not start_success:
        raise HTTPException(status_code=500, detail=f"启动失败: {start_msg}")

    return {"success": True, "message": f"frps 已重启: {start_msg}"}


# ==================== 配置接口 ====================

@router.get("/config")
async def get_frps_config():
    """获取当前 frps 配置"""
    manager = get_frp_server_manager()
    return {"success": True, "config": manager.config.to_dict()}


@router.put("/config")
async def update_frps_config(request: ConfigUpdateRequest):
    """更新 frps 配置"""
    manager = get_frp_server_manager()

    updates = {}
    if request.bind_port is not None:
        updates["bind_port"] = request.bind_port
    if request.token is not None:
        updates["token"] = request.token
    if request.dashboard_port is not None:
        updates["dashboard_port"] = request.dashboard_port
    if request.enable_xtcp is not None:
        updates["enable_xtcp"] = request.enable_xtcp

    config = manager.update_config(**updates)
    return {"success": True, "message": "配置已更新", "config": config.to_dict()}


# ==================== 客户端管理接口 ====================

@router.post("/clients/register")
async def register_client(request: ClientRegisterRequest):
    """
    注册新客户端并生成 frpc 配置

    返回该节点的完整配置和启动命令。
    """
    manager = get_frp_server_manager()
    config = manager.register_client(
        node_id=request.node_id,
        local_port=request.local_port,
        enable_p2p=request.enable_p2p,
        node_name=request.node_name,
    )

    toml_content = config.get("toml_content", "")
    launch_cmd = config.get("launch_command", "")

    return {
        "success": True,
        "node_id": request.node_id,
        "config": config,
        "toml_content": toml_content,
        "launch_command": launch_cmd,
        "remote_port": config.get("_meta", {}).get("remote_port"),
        "p2p_enabled": request.enable_p2p,
        "message": f"配置已生成，远程端口: {config['_meta'].get('remote_port')}"
    }


@router.get("/clients/{node_id}")
async def get_client_config(node_id: str):
    """获取指定节点的 frpc 配置"""
    manager = get_frp_server_manager()
    config = manager.get_client_config(node_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 未注册")
    return {"success": True, "node_id": node_id, "config": config}


@router.delete("/clients/{node_id}")
async def remove_client(node_id: str):
    """移除已注册的客户端"""
    manager = get_frp_server_manager()
    if not manager.remove_client(node_id):
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 不存在")
    return {"success": True, "message": f"节点 {node_id} 已移除"}


@router.get("/clients")
async def list_clients():
    """列出所有已注册的客户端"""
    manager = get_frp_server_manager()
    clients = manager.get_clients()
    return {"success": True, "clients": clients, "total": len(clients)}


# ==================== 安装与下载 ====================

@router.post("/install")
async def install_frp():
    """
    触发 FRP 安装/更新
    自动下载对应平台的 frp 二进制文件。
    """
    manager = get_frp_server_manager()

    if manager.check_frps_installed():
        return {
            "success": True,
            "message": "FRP 已经安装",
            "installed": True,
            "version": str(manager.frps_path),
        }

    success = await manager.ensure_frps_installed()
    if not success:
        raise HTTPException(status_code=500, detail="FRP 安装失败")

    return {
        "success": True,
        "message": "FRP 安装成功",
        "installed": True,
        "path": str(manager.frps_path),
    }


@router.get("/download-info")
async def get_download_info():
    """获取 FRP 下载信息（版本、链接等）"""
    from frp_helper import get_system_info
    system, arch = get_system_info()
    filename = f"frp_{FRP_VERSION}_{system}_{arch}"
    filename += ".zip" if system == "windows" else ".tar.gz"

    manager = get_frp_server_manager()
    return {
        "version": FRP_VERSION,
        "download_url": f"{FRP_BASE_URL}/v{FRP_VERSION}/{filename}",
        "platform": f"{system}_{arch}",
        "installed": manager.check_frps_installed(),
        "install_path": str(manager.frps_path),
    }


# ==================== 用户连接信息接口 (核心) ====================

@router.get("/user-connection")
async def get_user_connection_info(
    request: Request,
    node_id: Optional[str] = Query(None, description="节点 ID"),
    local_port: int = Query(50051, ge=1, le=65535),
):
    """
    获取当前用户的 FRP 连接信息和 exo 启动命令

    此接口供 user.html 页面调用，为登录用户展示：
    - 完整的 exo 节点启动命令（含 --frp-token）
    - frpc.toml 配置文件内容
    - P2P secretKey
    - 分配的远程端口

    每个用户看到的是基于自己身份的专属连接参数。
    server_addr 从访问 URL 自动提取，无需手动替换。
    """
    from auth_routes import get_session_token
    from auth_manager import get_auth_manager

    # 获取当前登录用户
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="请先登录")

    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="会话已过期，请重新登录")

    # 从请求中自动提取服务器公网地址
    server_addr = _extract_server_addr(request)

    # manager 地址（与 frps 相同，或可单独配置）
    manager_addr = server_addr

    # 使用 user_id 作为基础 node_id（确保每个用户唯一）
    user_node_id = node_id or f"node_{str(user.id)[:8]}"

    manager = get_frp_server_manager()

    connection_info = manager.get_user_connection_info(
        user_node_id=user_node_id,
        local_port=local_port,
        server_addr=server_addr,
        manager_addr=manager_addr,
    )

    return {
        "success": True,
        "user_nickname": user.nickname or "用户",
        "user_id": str(user.id),
        **connection_info,
        "message": (
            f"连接参数已生成 | "
            f"Token: {connection_info['frp_token'][:6]}... | "
            f"P2P: {'启用' if connection_info['p2p_enabled'] else '禁用'}"
        ),
    }


def _extract_server_addr(request: Request) -> str:
    """
    从请求中自动提取服务器公网地址

    优先级: X-Forwarded-Host > Host(排除localhost) > X-Forwarded-For > Host(兜底)
    """
    # 优先使用反向代理转发的原始 Host
    forwarded_host = request.headers.get("x-forwarded-host")
    if forwarded_host:
        addr = forwarded_host.split(":")[0].strip()
        return addr

    # 其次使用请求的 Host 头（排除本地地址）
    host = request.headers.get("host", "")
    if host:
        addr = host.split(":")[0].strip()
        if addr and addr not in ("localhost", "127.0.0.1", "::1"):
            return addr

    # 最后尝试 X-Forwarded-For
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        addr = xff.split(",")[0].strip().split(":")[0]
        if addr and addr not in ("localhost", "127.0.0.1"):
            return addr

    # 兜底：返回 Host（即使是 localhost 也比占位符好）
    return host.split(":")[0] if host else "<FRPS_SERVER_IP>"


# ==================== 集群状态接口 ====================

@router.get("/cluster-status")
async def get_user_cluster_status(request: Request):
    """
    获取集群状态 + 当前用户的节点信息

    供 user.html 页面调用，展示：
    - 节点总数、在线数
    - 总显存大小
    - 当前用户自己的节点详情
    """
    from auth_routes import get_session_token
    from auth_manager import get_auth_manager

    # 获取当前用户
    token = get_session_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="请先登录")

    auth_mgr = get_auth_manager()
    user = auth_mgr.validate_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="会话已过期，请重新登录")

    # 用户对应的 node_id
    user_node_id = f"node_{str(user.id)[:8]}"

    # 获取集群管理器
    from server import manager as cluster_mgr

    if not cluster_mgr:
        return {
            "success": True,
            "cluster": {
                "total_nodes": 0,
                "online_nodes": 0,
                "total_memory_gb": 0.0,
                "health_percent": 0,
                "total_models": 0,
            },
            "my_node": None,
            "nodes": [],
        }

    status = cluster_mgr.get_cluster_status()
    all_nodes = status.get("nodes", [])
    online_nodes = [n for n in all_nodes if n.get("status") == "online"]

    # 计算总显存 (MB -> GB)
    total_memory_gb = round(
        sum(n.get("device_info", {}).get("memory", 0) for n in online_nodes) / 1024, 2
    )

    # 查找当前用户的节点
    my_node = None
    for n in all_nodes:
        if n.get("node_id") == user_node_id:
            my_node = n
            break

    # 简化节点列表返回给前端
    nodes_list = []
    for n in all_nodes:
        di = n.get("device_info", {})
        md = di.get("memory_detail", {})
        nodes_list.append({
            "node_id": n.get("node_id"),
            "status": n.get("status"),
            "chip": di.get("chip", "Unknown"),
            "memory_gb": round(di.get("memory", 0) / 1024, 1),
            "memory_used_mb": md.get("used", 0),
            "memory_total_mb": md.get("total", 0),
            "loaded_models": [m.get("model_id", "?") for m in n.get("loaded_models", [])],
            "is_mine": n.get("node_id") == user_node_id,
        })

    return {
        "success": True,
        "cluster": {
            "total_nodes": len(all_nodes),
            "online_nodes": len(online_nodes),
            "offline_nodes": len(all_nodes) - len(online_nodes),
            "total_memory_gb": total_memory_gb,
            "health_percent": round(
                len(online_nodes) / max(len(all_nodes), 1) * 100, 1
            ) if all_nodes else 0,
            "total_models": len(set(
                m.get("model_id", "")
                for n in online_nodes
                for m in n.get("loaded_models", [])
            )),
        },
        "my_node": my_node,
        "nodes": nodes_list,
    }
