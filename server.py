"""
EXO Cluster Manager - REST API Server
=====================================

提供完整的RESTful API接口用于管理EXO集群

API端点:
---------
节点管理:
    GET    /api/nodes              - 获取所有节点列表
    GET    /api/nodes/{node_id}     - 获取单个节点详情
    POST   /api/nodes              - 添加新节点
    DELETE /api/nodes/{node_id}     - 移除节点
    POST   /api/nodes/{node_id}/health-check  - 手动健康检查

集群状态:
    GET    /api/cluster/status      - 集群整体状态
    GET    /api/cluster/stats       - 统计信息

模型管理:
    GET    /api/models              - 所有已加载的模型
    POST   /api/models/load         - 加载模型到池
    POST   /api/models/unload       - 卸载模型
    POST   /api/models/rebalance    - 重平衡模型

GPU池:
    GET    /api/pool/status         - GPU池状态
    POST   /api/pool/load           - 加载模型到GPU池
    GET    /api/pool/preview        - 预览分配方案

OpenAI 兼容 API:
    POST   /v1/chat/completions     - 聊天补全 (支持流式)
    POST   /v1/completions          - 文本补全 (支持流式)
    GET    /v1/models               - 列出可用模型
    GET    /v1/models/{model}       - 获取模型信息

API Key 管理:
    POST   /v1/admin/keys           - 生成新 API Key
    GET    /v1/admin/keys           - 列出所有 API Key
    DELETE /v1/admin/keys/{key}     - 吊销 API Key

WebSocket:
    WS     /ws/cluster              - 实时集群状态推送
    WS     /ws/node/{node_id}       - Node 推理通道（内网穿透）

启动:
    python server.py --port 8080 --config ../network_config.json
"""

import asyncio
import json
import logging
import time
import uuid
import argparse
from typing import Optional, Dict, Any, List, AsyncGenerator, Tuple
from pathlib import Path

# FastAPI相关导入
try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Body, Query
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("请安装依赖: pip install fastapi uvicorn")
    exit(1)

# 导入核心模块
import sys
import os

# 确保当前文件所在目录在sys.path中（支持 -m 模式和直接运行）
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from cluster_core import (
    EXOClusterManager,
    NodeConnector,
    EXONodeInfo,
    get_cluster_manager,
    NodeStatus
)

from p2p_topology import P2PTopologyManager

# 导入 OpenAI 兼容路由和 API Key 管理
from api_key_manager import get_api_key_manager
from openai_routes import router as openai_router

# 导入认证路由 (分离式: auth_router + admin_router)
from auth_manager import get_auth_manager
from auth_routes import auth_router, admin_router

# 导入负载均衡器
from load_balancer import (
    LoadBalancer,
    LBStrategy,
    get_load_balancer,
    init_load_balancer
)

# 导入自动模型分配器
from auto_model_allocator import (
    AutoModelAllocator,
    AllocationStrategy,
    ModelSpec,
    get_model_library
)

# 导入节点稳定性管理和故障恢复模块
from node_stability_manager import (
    NodeStabilityManager,
    FaultRecoveryManager,
    AllocationPlanManager,
    ResilientAllocationManager
)

# 导入主备冗余分配系统
from ha_model_allocator import (
    HAModelAllocator,
    RedundancyMode,
    InstanceRole,
    InstanceStatus,
    FailoverState
)

# 导入自动分配触发器
from auto_alloc_trigger import AutoAllocTrigger, TriggerConfig

# 全局管理器实例
manager: Optional[EXOClusterManager] = None
topo_manager: Optional[P2PTopologyManager] = None

# 自动模型分配器实例（延迟初始化，依赖manager）
_auto_allocator: Optional[AutoModelAllocator] = None

# 弹性分配管理器（整合稳定性+恢复+版本管理）
_resilient_allocator: Optional[ResilientAllocationManager] = None

# 主备冗余分配器（高可用模式）
_ha_allocator: Optional[HAModelAllocator] = None

# 自动分配触发器
_auto_trigger: Optional[AutoAllocTrigger] = None

def get_auto_allocator() -> AutoModelAllocator:
    """获取自动分配器实例（延迟初始化）"""
    global _auto_allocator
    if _auto_allocator is None and manager is not None:
        _auto_allocator = AutoModelAllocator(manager)
        logger.info("✅ [AutoAlloc] 自动模型分配器初始化完成")
    if _auto_allocator is None:
        raise HTTPException(status_code=503, detail="集群管理器未就绪")
    return _auto_allocator

def get_resilient_allocator() -> ResilientAllocationManager:
    """获取弹性分配管理器实例（延迟初始化）"""
    global _resilient_allocator
    if _resilient_allocator is None and manager is not None:
        _resilient_allocator = ResilientAllocationManager(manager)

        # 启动后台监控循环
        asyncio.create_task(_resilient_allocator.stability_mgr.start_monitoring_loop(interval=30.0))

        logger.info("✅ [ResilientAlloc] 弹性分配管理器初始化完成（含监控循环）")
    if _resilient_allocator is None:
        raise HTTPException(status_code=503, detail="集群管理器未就绪")
    return _resilient_allocator

def get_ha_allocator() -> HAModelAllocator:
    """获取主备冗余分配器实例（延迟初始化，含健康监控）"""
    global _ha_allocator
    if _ha_allocator is None and manager is not None:
        _ha_allocator = HAModelAllocator(manager)

        # 启动健康监控系统
        asyncio.create_task(_ha_allocator.start_health_monitoring())

        logger.info("✅ [HAAlloc] 主备冗余分配器初始化完成（含健康监控）")
    if _ha_allocator is None:
        raise HTTPException(status_code=503, detail="集群管理器未就绪")
    return _ha_allocator

def get_auto_trigger() -> AutoAllocTrigger:
    """获取自动分配触发器实例（延迟初始化，含监控循环）"""
    global _auto_trigger
    if _auto_trigger is None and manager is not None:
        _auto_trigger = AutoAllocTrigger(manager)

        # 启动自动分配（包含启动初始化）
        asyncio.create_task(_auto_trigger.start())

        logger.info("✅ [AutoTrigger] 自动分配触发器初始化完成（已启动）")
    if _auto_trigger is None:
        raise HTTPException(status_code=503, detail="集群管理器未就绪")
    return _auto_trigger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入系统日志收集器
from sys_logger import sys_log

app = FastAPI(
    title="EXO Cluster Manager API",
    description="分布式AI模型推理集群管理系统",
    version="1.0.0"
)

# 挂载静态文件目录 (仅挂载静态资源，不拦截已有路由)
_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir), html=True), name="static")

# CORS中间件（安全改进: 限制允许的源域名）
# 从环境变量读取允许的域名列表，默认仅允许本地开发
import os
_allowed_origins_str = os.getenv("EXO_CORS_ORIGINS", "http://localhost:8080,http://localhost:3000,http://127.0.0.1:8080")
ALLOWED_ORIGINS = [origin.strip() for origin in _allowed_origins_str.split(",") if origin.strip()]

# 开发模式检测: 如果环境变量 EXO_DEV_MODE=true，则允许所有来源 (仅用于本地调试)
_is_dev_mode = os.getenv("EXO_DEV_MODE", "").lower() in ("true", "1", "yes")

if _is_dev_mode:
    logger.warning("⚠️ CORS 开发模式已启用: 允许所有来源 (不建议用于生产环境)")
    _cors_origins = ["*"]
else:
    logger.info(f"CORS 已启用安全模式，允许的源域名: {ALLOWED_ORIGINS}")
    _cors_origins = ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True if _cors_origins != ["*"] else False,  # 当 allow_origins=["*"] 时必须为 False
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 限制 HTTP 方法
    allow_headers=["Authorization", "Content-Type", "Cookie", "X-Requested-With"],  # 限制请求头
)

# ==================== 安全响应头中间件 ====================
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    安全响应头中间件

    添加以下 HTTP 安全头:
    - X-Content-Type-Options: 防止 MIME 类型嗅探
    - X-Frame-Options: 防止点击劫持
    - X-XSS-Protection: 启用浏览器 XSS 过滤器
    - Strict-Transport-Security (HSTS): 强制 HTTPS
    - Content-Security-Policy: 限制资源加载来源
    - Referrer-Policy: 控制 Referer 头信息泄露
    - Permissions-Policy: 限制浏览器功能
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # 基础安全头 (所有环境启用)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # 权限策略: 禁用不必要的浏览器功能
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )

        # 生产环境额外安全头
        is_production = os.getenv("EXO_ENV", "").lower() in ("production", "prod")
        if is_production:
            # HSTS: 强制 HTTPS 1年 (包括子域名)
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

            # CSP: 限制脚本/样式来源 (根据实际需求调整)
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
                "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
                "img-src 'self' data: https:; "
                "font-src 'self' https://cdnjs.cloudflare.com; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
        else:
            # 开发模式: 宽松的 CSP (便于调试)
            response.headers["Content-Security-Policy"] = (
                "default-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com; "
                "img-src 'self' data: https: http:; "
                "connect-src 'self' ws: wss:"
            )

        return response


# 注册安全头中间件 (必须在路由注册之前)
app.add_middleware(SecurityHeadersMiddleware)
logger.info("✅ 安全响应头中间件已注册")

# ==================== API 速率限制中间件 ====================
import time
from collections import defaultdict
from threading import Lock

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    API 全局速率限制中间件

    特性:
    - 基于 IP 地址的请求频率限制
    - 支持不同端点的差异化限流策略
    - 滑动窗口算法，防止突发流量
    - 自动记录超限请求用于审计
    - 支持白名单 IP 无限制访问

    配置环境变量:
    - EXO_RATE_LIMIT_ENABLED: 是否启用 (true/false, 默认 true)
    - EXO_RATE_LIMIT_REQUESTS: 窗口内最大请求数 (默认 100)
    - EXO_RATE_LIMIT_WINDOW: 时间窗口秒数 (默认 60)
    - EXO_RATE_LIMIT_WHITELIST: 白名单 IP (逗号分隔)
    """

    def __init__(self, app):
        super().__init__(app)

        # 从环境变量读取配置
        self.enabled = os.getenv("EXO_RATE_LIMIT_ENABLED", "true").lower() in ("true", "1", "yes")
        self.max_requests = int(os.getenv("EXO_RATE_LIMIT_REQUESTS", "100"))
        self.window_seconds = int(os.getenv("EXO_RATE_LIMIT_WINDOW", "60"))

        # IP 白名单 (不受限制)
        whitelist_str = os.getenv("EXO_RATE_LIMIT_WHITELIST", "")
        self.whitelist_ips = set(ip.strip() for ip in whitelist_str.split(",") if ip.strip())

        # 存储结构: {ip: [(timestamp1), (timestamp2), ...]}
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

        # 特殊端点配置 (更严格的限制)
        self.endpoint_limits = {
            "/login": {"requests": 10, "window": 60},      # 登录接口: 10次/分钟
            "/register": {"requests": 5, "window": 300},     # 注册接口: 5次/5分钟
            "/admin/": {"requests": 30, "window": 60},       # 管理接口: 30次/分钟
            "/v1/chat/completions": {"requests": 20, "window": 60},  # 聊天: 20次/分钟
        }

        if self.enabled:
            logger.info(
                f"✅ API 速率限制已启用 (全局: {self.max_requests}次/{self.window_seconds}s, "
                f"白名单IP: {len(self.whitelist_ips)}个)"
            )
        else:
            logger.warning("⚠️ API 速率限制已禁用 (不建议生产环境)")

    def _cleanup_old_requests(self, ip: str, current_time: float):
        """清理过期的请求记录"""
        cutoff_time = current_time - self.window_seconds
        self._requests[ip] = [
            ts for ts in self._requests[ip] if ts > cutoff_time
        ]

    def _is_rate_limited(self, ip: str, path: str) -> Tuple[bool, Dict]:
        """
        检查是否触发速率限制

        Returns:
            (is_limited, info_dict) - info 包含限制详情
        """
        if not self.enabled:
            return False, {}

        # 白名单 IP 不受限
        if ip in self.whitelist_ips:
            return False, {}

        current_time = time.time()

        with self._lock:
            # 清理过期记录
            self._cleanup_old_requests(ip, current_time)

            # 检查特殊端点限制
            for endpoint_prefix, limit_config in self.endpoint_limits.items():
                if path.startswith(endpoint_prefix):
                    max_reqs = limit_config["requests"]
                    window = limit_config["window"]

                    # 使用独立的计数器 (基于 endpoint + ip)
                    key = f"{ip}:{endpoint_prefix}"
                    if key not in self._requests:
                        self._requests[key] = []

                    cutoff = current_time - window
                    self._requests[key] = [ts for ts in self._requests[key] if ts > cutoff]

                    if len(self._requests[key]) >= max_reqs:
                        retry_after = int(window - (current_time - self._requests[key][0])) + 1
                        return True, {
                            "limit_type": "endpoint",
                            "endpoint": endpoint_prefix,
                            "max_requests": max_reqs,
                            "window_seconds": window,
                            "retry_after": retry_after,
                        }

                    self._requests[key].append(current_time)
                    return False, {}

            # 全局限制检查
            if len(self._requests[ip]) >= self.max_requests:
                retry_after = int(
                    self.window_seconds - (current_time - self._requests[ip][0])
                ) + 1
                return True, {
                    "limit_type": "global",
                    "max_requests": self.max_requests,
                    "window_seconds": self.window_seconds,
                    "retry_after": retry_after,
                }

            # 记录本次请求
            self._requests[ip].append(current_time)
            return False, {}

    async def dispatch(self, request: Request, call_next):
        """处理请求并检查速率限制"""
        # 获取客户端 IP
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        # OPTIONS 预检请求不限制
        if request.method == "OPTIONS":
            return await call_next(request)

        # 静态资源不限制
        if request.url.path.startswith("/static"):
            return await call_next(request)

        # 检查速率限制
        is_limited, limit_info = self._is_rate_limited(client_ip, request.url.path)

        if is_limited:
            # 记录速率限制事件 (用于审计和安全分析)
            logger.warning(
                f"🚫 [RateLimit] IP {client_ip} 触发速率限制 "
                f"(路径: {request.url.path}, 类型: {limit_info.get('limit_type')}, "
                f"重试: {limit_info.get('retry_after')}s)"
            )

            # 返回 429 Too Many Requests
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "message": "API 请求频率超过限制，请稍后重试",
                    "detail": limit_info,
                    "retry_after": limit_info.get("retry_after", 60),
                },
                headers={"Retry-After": str(limit_info.get("retry_after", 60))},
            )

        # 正常处理请求
        response = await call_next(request)

        # 在响应头中添加速率限制信息 (方便客户端查看)
        if self.enabled and client_ip not in self.whitelist_ips:
            with self._lock:
                remaining = max(0, self.max_requests - len(self._requests.get(client_ip, [])))
            response.headers["X-RateLimit-Limit"] = str(self.max_requests)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.window_seconds)

        return response


# 注册速率限制中间件 (在安全头之后，路由之前)
app.add_middleware(RateLimitMiddleware)

# ==================== IP 白名单/黑名单过滤中间件 ====================
class IPFilterMiddleware(BaseHTTPMiddleware):
    """
    IP 访问控制中间件

    功能:
    - 白名单模式: 仅允许白名单中的 IP 访问 (适用于管理后台)
    - 黑名单模式: 拒绝黑名单中的 IP 访问
    - 支持路径级别的差异化规则
    - 自动记录被拒绝的访问尝试

    配置环境变量:
    - EXO_IP_FILTER_ENABLED: 是否启用 (true/false, 默认 false)
    - EXO_IP_WHITELIST_MODE: 使用白名单模式 (true/false)
    - EXO_IP_WHITELIST: 白名单 IP (逗号分隔)
    - EXO_IP_BLACKLIST: 黑名单 IP (逗号分隔)
    - EXO_IP_ADMIN_PATHS: 需要严格控制的路径前缀 (默认 /admin)
    """

    def __init__(self, app):
        super().__init__(app)

        # 从环境变量读取配置
        self.enabled = os.getenv("EXO_IP_FILTER_ENABLED", "false").lower() in ("true", "1", "yes")
        self.whitelist_mode = os.getenv("EXO_IP_WHITELIST_MODE", "false").lower() in ("true", "1", "yes")

        # 解析白名单和黑名单
        whitelist_str = os.getenv("EXO_IP_WHITELIST", "")
        self.whitelist_ips = set(ip.strip() for ip in whitelist_str.split(",") if ip.strip())

        blacklist_str = os.getenv("EXO_IP_BLACKLIST", "")
        self.blacklist_ips = set(ip.strip() for ip in blacklist_str.split(",") if ip.strip())

        # 需要严格控制的路径 (管理员接口)
        admin_paths_str = os.getenv("EXO_IP_ADMIN_PATHS", "/admin")
        self.admin_paths = [p.strip() for p in admin_paths_str.split(",") if p.strip()]

        # 内网/私有地址范围 (自动信任)
        self.trusted_ranges = [
            "127.0.0.1",      # localhost
            "::1",            # IPv6 localhost
            "10.",            # 10.0.0.0/8
            "172.16.",       # 172.16.0.0/12
            "192.168.",      # 192.168.0.0/16
        ]

        if self.enabled:
            mode_str = "白名单" if self.whitelist_mode else "黑名单"
            logger.info(
                f"✅ IP 过滤已启用 ({mode_str}模式, "
                f"白名单: {len(self.whitelist_ips)}个, "
                f"黑名单: {len(self.blacklist_ips)}个)"
            )
        else:
            logger.info("ℹ️ IP 过滤未启用 (可通过 EXO_IP_FILTER_ENABLED 启用)")

    def _is_trusted_ip(self, ip: str) -> bool:
        """检查是否为受信任的内部 IP"""
        return any(ip.startswith(prefix) for prefix in self.trusted_ranges) or ip == "unknown"

    def _is_admin_path(self, path: str) -> bool:
        """检查是否为管理员路径"""
        return any(path.startswith(admin_path) for admin_path in self.admin_paths)

    def _should_block(self, ip: str, path: str) -> Tuple[bool, str]:
        """
        判断是否应该阻止访问

        Returns:
            (should_block, reason)
        """
        if not self.enabled:
            return False, ""

        # 受信任的 IP 始终允许
        if self._is_trusted_ip(ip):
            return False, ""

        # 管理员路径使用更严格的规则
        is_admin = self._is_admin_path(path)

        if self.whitelist_mode:
            # 白名单模式: 不在白名单中则拒绝
            if ip not in self.whitelist_ips:
                reason = (
                    "IP 不在白名单中 (管理员接口需要白名单访问)" if is_admin else
                    "IP 不在白名单中"
                )
                return True, reason
        else:
            # 黑名单模式: 在黑名单中则拒绝
            if ip in self.blacklist_ips:
                return True, "IP 已被列入黑名单"

            # 管理员接口额外检查白名单
            if is_admin and self.whitelist_ips and ip not in self.whitelist_ips:
                return True, "管理员接口仅限白名单 IP 访问"

        return False, ""

    async def dispatch(self, request: Request, call_next):
        """处理请求并检查 IP 访问权限"""

        # 获取客户端 IP
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        # OPTIONS 和静态资源不限制
        if request.method == "OPTIONS" or request.url.path.startswith("/static"):
            return await call_next(request)

        # 检查是否应该阻止访问
        should_block, block_reason = self._should_block(client_ip, request.url.path)

        if should_block:
            # 记录审计日志 (如果可用)
            try:
                from audit_logger import get_audit_logger
                audit = get_audit_logger()
                audit.log_security_event(
                    event_type="IP access denied",
                    ip=client_ip,
                    details={
                        "path": request.url.path,
                        "method": request.method,
                        "reason": block_reason,
                        "user_agent": request.headers.get("user-agent", "")[:200]
                    },
                    severity="WARNING"
                )
            except Exception:
                pass  # 审计日志不可用时静默失败

            logger.warning(
                f"🚫 [IPFilter] 拒绝访问: IP={client_ip}, "
                f"路径={request.url.path}, 原因={block_reason}"
            )

            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden",
                    "message": "您的 IP 地址没有权限访问此资源",
                    "detail": block_reason if os.getenv("EXO_DEBUG", "").lower() in ("true", "1") else None
                }
            )

        # 正常处理请求
        response = await call_next(request)
        return response


# 注册 IP 过滤中间件 (在速率限制之后，路由之前)
app.add_middleware(IPFilterMiddleware)

# 注册 OpenAI 兼容路由
app.include_router(openai_router)
logger.info("✅ OpenAI 兼容 API 路由已注册 (/v1/*)")

# 注册认证路由
app.include_router(auth_router)
logger.info("✅ 认证路由已注册 (无前缀)")
app.include_router(admin_router)
logger.info("✅ 管理员路由已注册 (/admin/*)")

# 注册 FRP Server 管理路由
from frp_routes import router as frp_router
app.include_router(frp_router)
logger.info("✅ FRP Server 管理路由已注册 (/api/frps/*)")

# ==================== 自定义模型配置管理 ====================
CUSTOM_MODELS_FILE = Path(__file__).parent / "data" / "custom_models.json"

# 模型配置：唯一数据源为 data/custom_models.json，无硬编码默认值
custom_model_cards: Dict[str, Dict] = {}
custom_pretty_names: Dict[str, str] = {}

# 默认模型配置（首次部署 / Docker 首次启动时自动创建）
DEFAULT_MODEL_CONFIG = {
    "model_cards": {
        "qwen-3-0.6b": {
            "layers": 28,
            "repo": {"PyTorchQwen3InferenceEngine": "Qwen/Qwen3-0.6B"},
            "allocation": {"layer_memory_mb": 100, "param_count": 0.6, "category": "general", "priority": 0.8}
        },
        "qwen-3-4b": {
            "layers": 36,
            "repo": {"PyTorchQwen3InferenceEngine": "Qwen/Qwen3-4B"},
            "allocation": {"layer_memory_mb": 215, "param_count": 4, "category": "general", "priority": 1.0}
        },
        "qwen-3-vl-2b": {
            "layers": 28,
            "repo": {"PyTorchQwen3VLInferenceEngine": "Qwen/Qwen3-VL-2B-Instruct"},
            "allocation": {"layer_memory_mb": 240, "param_count": 2, "category": "vision", "priority": 0.9}
        },
        "qwen-3-vl-4b": {
            "layers": 36,
            "repo": {"PyTorchQwen3VLInferenceEngine": "Qwen/Qwen3-VL-4B-Instruct"},
            "allocation": {"layer_memory_mb": 380, "param_count": 4, "category": "vision", "priority": 1.0}
        },
        "qwen-2.5-vl-3b": {
            "layers": 36,
            "repo": {"PyTorchQwen2_5VlInferenceEngine": "Qwen/Qwen2.5-VL-3B-Instruct"},
            "allocation": {"layer_memory_mb": 330, "param_count": 3, "category": "vision", "priority": 0.9}
        },
        "llama-3.2-1b": {
            "layers": 16,
            "repo": {"PyTorchLlama3InferenceEngine": "unsloth/Llama-3.2-1B-Instruct"},
            "allocation": {"layer_memory_mb": 250, "param_count": 1, "category": "general", "priority": 0.7}
        }
    },
    "pretty_names": {
        "qwen-3-0.6b": "Qwen 3 0.6B",
        "qwen-3-4b": "Qwen 3 4B",
        "qwen-3-vl-2b": "Qwen 3 VL 2B",
        "qwen-3-vl-4b": "Qwen 3 VL 4B",
        "qwen-2.5-vl-3b": "Qwen 2.5 VL 3B",
        "llama-3.2-1b": "Llama 3.2 1B"
    }
}

def load_custom_models():
    global custom_model_cards, custom_pretty_names
    try:
        if CUSTOM_MODELS_FILE.exists():
            with open(CUSTOM_MODELS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                custom_model_cards = data.get('model_cards', {})
                custom_pretty_names = data.get('pretty_names', {})
            logger.info(f"[Models] 已加载 {len(custom_model_cards)} 个模型（来自 custom_models.json）")
        else:
            # 首次启动：自动创建默认配置文件（Docker 部署场景）
            logger.info("[Models] 未找到配置文件，正在创建默认配置...")
            custom_model_cards = DEFAULT_MODEL_CONFIG["model_cards"]
            custom_pretty_names = DEFAULT_MODEL_CONFIG["pretty_names"]
            save_custom_models()
            logger.info(f"[Models] 已创建默认配置，包含 {len(custom_model_cards)} 个模型")
    except Exception as e:
        logger.error(f"[Models] 加载自定义模型失败: {e}")

def save_custom_models():
    try:
        CUSTOM_MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CUSTOM_MODELS_FILE, 'w', encoding='utf-8') as f:
            json.dump({
                'model_cards': custom_model_cards,
                'pretty_names': custom_pretty_names
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"[Models] 已保存 {len(custom_model_cards)} 个自定义模型")
    except Exception as e:
        logger.error(f"[Models] 保存自定义模型失败: {e}")
        raise e

load_custom_models()

# 全局管理器实例
manager: Optional[EXOClusterManager] = None

HEARTBEAT_TIMEOUT = 90  # 心跳超时阈值（秒）
HEARTBEAT_CHECK_INTERVAL = 30  # 检查间隔（秒）

# LB 一致性检查节流控制
_lb_last_warning_time: float = 0
_lb_warning_throttle_interval: float = 60.0  # 最少间隔60秒才重复告警


async def _cleanup_offline_node(node_id: str):
    """
    清理离线节点的所有资源
    
    级联清理：
    1. 断开 gRPC 连接器
    2. 清理 GPU 池中的模型分配
    3. 从 LB 中移除实例
    4. 清空已加载模型列表
    5. 重置设备信息
    """
    global manager
    
    if not manager or node_id not in manager.nodes:
        return
    
    logger.info(f"🧹 [Cleanup] 开始清理离线节点 {node_id} 的资源...")
    
    try:
        # 1. 断开连接器
        if node_id in manager.connectors:
            connector = manager.connectors[node_id]
            try:
                await connector.disconnect()
                logger.info(f"  ✓ 已断开 gRPC 连接器")
            except Exception as e:
                logger.warning(f"  ⚠️ 断开连接器失败: {e}")
        
        # 2. 清理 GPU 池中的模型分配
        if hasattr(manager, 'gpu_pool') and manager.gpu_pool:
            try:
                models_to_remove = [
                    model_id for model_id, info in manager.gpu_pool.pool_models.items()
                    if any(shard.node_id == node_id for shard in info.shards)
                ]
                
                for model_id in models_to_remove:
                    try:
                        await manager.gpu_pool.unload(model_id)
                        logger.info(f"  ✓ 已从 GPU 池卸载模型: {model_id}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ 卸载模型失败 ({model_id}): {e}")
            except Exception as e:
                logger.debug(f"  GPU 池清理跳过（未初始化或出错）: {e}")
        
        # 3. 从 LB 中移除实例
        try:
            lb = get_load_balancer()
            if lb and hasattr(lb, 'instances'):
                instances_to_remove = [
                    (model_id, inst_id) 
                    for model_id, model_instances in lb.instances.items()
                    for inst_id, inst in model_instances.items()
                    if inst.get("node_id") == node_id
                ]
                
                for model_id, inst_id in instances_to_remove:
                    del lb.instances[model_id][inst_id]
                    if not lb.instances[model_id]:
                        del lb.instances[model_id]
                    
                if instances_to_remove:
                    logger.info(f"  ✓ 已从 LB 移除 {len(instances_to_remove)} 个实例")
        except Exception as e:
            logger.debug(f"  LB 清理跳过: {e}")
        
        # 4. 清空节点的已加载模型列表和设备信息
        node_info = manager.nodes[node_id]
        if node_info.loaded_models:
            cleared_count = len(node_info.loaded_models)
            node_info.loaded_models.clear()
            logger.info(f"  ✓ 已清空 {cleared_count} 个已加载模型记录")
        
        # 5. 重置设备信息（保留基础信息用于显示）
        if node_info.device_info:
            old_device = node_info.device_info.get("chip", "Unknown")
            node_info.device_info = {
                "chip": old_device,
                "memory": 0,
                "_offline": True,
                "_offline_time": time.time()
            }
            logger.info(f"  ✓ 已重置设备信息（标记为离线）")
        
        logger.info(f"✅ [Cleanup] 节点 {node_id} 资源清理完成")
        
    except Exception as e:
        logger.error(f"❌ [Cleanup] 节点 {node_id} 清理失败: {e}", exc_info=True)


async def _heartbeat_monitor_task():
    """
    心跳超时监控后台任务
    
    定期检查所有节点的 last_heartbeat 时间戳，
    超过阈值未收到心跳的节点自动标记为离线。
    
    解决问题：Node 进程崩溃/网络断开但 Manager 未感知
    """
    global manager
    
    logger.info(f"🔄 [HeartbeatMonitor] 心跳监控已启动 (间隔={HEARTBEAT_CHECK_INTERVAL}s, 超时={HEARTBEAT_TIMEOUT}s)")
    
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_CHECK_INTERVAL)
            
            if not manager:
                continue
            
            current_time = time.time()
            offline_nodes = []
            
            for node_id, node_info in list(manager.nodes.items()):
                if node_info.status == NodeStatus.OFFLINE:
                    continue
                
                if node_info.last_heartbeat > 0:
                    heartbeat_age = current_time - node_info.last_heartbeat
                    
                    if heartbeat_age > HEARTBEAT_TIMEOUT:
                        old_status = node_info.status.value
                        node_info.status = NodeStatus.OFFLINE
                        node_info.error_message = f"心跳超时 ({heartbeat_age:.0f}秒无响应)"
                        
                        offline_nodes.append({
                            "node_id": node_id,
                            "old_status": old_status,
                            "heartbeat_age": heartbeat_age
                        })
                        
                        logger.warning(
                            f"⏰ [HeartbeatMonitor] 节点 {node_id} 心跳超时，"
                            f"已标记为离线 (上次心跳: {heartbeat_age:.0f}秒前)"
                        )
            
            if offline_nodes:
                try:
                    await ws_manager.broadcast({
                        "type": "nodes_offline",
                        "nodes": offline_nodes,
                        "timestamp": current_time
                    })
                except Exception as e:
                    logger.debug(f"广播离线通知失败: {e}")
                
                logger.info(f"💔 [HeartbeatMonitor] 已标记 {len(offline_nodes)} 个节点为离线")
                
                for offline_node in offline_nodes:
                    node_id = offline_node["node_id"]
                    await _cleanup_offline_node(node_id)
        
        except asyncio.CancelledError:
            logger.info("🛑 [HeartbeatMonitor] 心跳监控任务已停止")
            break
        except Exception as e:
            logger.error(f"❌ [HeartbeatMonitor] 心跳监控异常: {e}", exc_info=True)
            await asyncio.sleep(10)  # 异常后等待再重试


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global manager, topo_manager
    
    logger.info("🚀 启动EXO Cluster Manager API服务...")
    
    manager = await get_cluster_manager()
    
    if manager:
        # 设置 WebSocket 广播回调
        manager.set_broadcast_callback(ws_manager.broadcast)
        
        logger.info(f"✅ 服务就绪，管理 {len(manager.nodes)} 个节点")
        
        # 初始化P2P拓扑管理器
        topo_manager = P2PTopologyManager(manager)
        await topo_manager.start()
        logger.info(f"🔗 P2P拓扑管理器已启动")
        
        # ✅ 启动心跳超时检测后台任务
        asyncio.create_task(_heartbeat_monitor_task())
        logger.info(f"💓 心跳监控任务已启动 (超时阈值: 90秒)")

        # ✅ 启动自动分配触发器（全自动模式）
        try:
            global _auto_trigger
            _auto_trigger = AutoAllocTrigger(manager)
            await _auto_trigger.start()

            # 连接稳定性管理器（增强防抖能力）
            try:
                resilient_alloc = get_resilient_allocator()
                if resilient_alloc and hasattr(resilient_alloc, 'stability_mgr'):
                    _auto_trigger.set_stability_manager(resilient_alloc.stability_mgr)
                    logger.info(f"🤖 自动分配触发器已连接稳定性管理器 (防抖增强)")
            except Exception as stability_err:
                logger.warning(f"⚠️ 稳定性管理器连接失败（使用基础防抖）: {stability_err}")

            logger.info(f"🤖 自动分配触发器已启动 (将在30s后初始化首次分配)")
        except Exception as e:
            logger.error(f"⚠️ 自动分配触发器启动失败: {e}")
    
    # 初始化 API Key 管理器，如果没有 key 则生成一个默认的
    key_manager = get_api_key_manager()
    stats = key_manager.get_stats()
    if stats["total_keys"] == 0:
        default_key = key_manager.generate_key(
            name="默认管理员 Key",
            description="自动生成的默认 API Key，请妥善保管",
            permissions=["*"],
            allowed_models=["*"]
        )
        logger.info(f"🔑 已生成默认 API Key: {default_key}")
        logger.info(f"   请使用此 Key 进行 API 调用，或通过 /v1/admin/keys 管理")
    else:
        logger.info(f"🔑 已加载 {stats['total_keys']} 个 API Key ({stats['active_keys']} 个活跃)")

    # 初始化 FRP Server（如果启用）
    frp_enabled = os.getenv("EXO_FRP_ENABLE", "false").lower() in ("true", "1", "yes")
    if frp_enabled:
        from frp_server_manager import get_frp_server_manager
        frp_mgr = get_frp_server_manager()

        # 应用命令行配置（token 优先复用已有配置，避免重启后 token 变化导致节点断连）
        bind_port = int(os.getenv("EXO_FRP_BIND_PORT", "7000"))
        token = os.getenv("EXO_FRP_TOKEN")
        dashboard_port = os.getenv("EXO_FRP_DASHBOARD_PORT")

        config_updates = {"bind_port": bind_port}

        # Token 持久化逻辑：已有配置则复用，否则用新生成的
        if not frp_mgr.config.token and token:
            config_updates["token"] = token
            logger.info(f"🔑 FRP Token 已设置 (新)")
        elif frp_mgr.config.token:
            logger.info(f"🔑 FRP Token 已复用 (持久化): {frp_mgr.config.token[:8]}...")
        elif token:
            config_updates["token"] = token

        if dashboard_port:
            port_val = int(dashboard_port)
            config_updates["dashboard_port"] = port_val if port_val > 0 else None

        frp_mgr.update_config(**config_updates)

        logger.info(f"🌐 正在启动 FRP Server (端口: {bind_port})...")
        success, msg = await frp_mgr.start_server()
        if success:
            logger.info(f"✅ FRP Server 启动成功: {msg}")
        else:
            logger.warning(f"⚠️ FRP Server 启动失败: {msg} (可通过 /api/frps/start 手动启动)")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    global manager, topo_manager, _auto_trigger

    # 停止自动分配触发器
    if _auto_trigger:
        try:
            await _auto_trigger.stop()
            logger.info("🤖 自动分配触发器已停止")
        except Exception as e:
            logger.warning(f"⚠️ 触发器停止异常: {e}")
    
    if topo_manager:
        await topo_manager.stop()
    
    if manager:
        await manager.shutdown()

    # 停止 FRP Server
    frp_enabled = os.getenv("EXO_FRP_ENABLE", "false").lower() in ("true", "1", "yes")
    if frp_enabled:
        try:
            from frp_server_manager import get_frp_server_manager
            frp_mgr = get_frp_server_manager()
            if frp_mgr.is_running():
                await frp_mgr.stop_server()
                logger.info("🌐 FRP Server 已停止")
        except Exception as e:
            logger.warning(f"⚠️ FRP Server 停止异常: {e}")


# ==================== 节点管理 API ====================

@app.get("/api/nodes", response_model=Dict)
async def list_nodes():
    """
    获取所有节点列表

    返回所有已注册的节点及其基本状态信息，包含所属用户映射
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")

    status = manager.get_cluster_status()
    nodes = status["nodes"]

    # ---- 匹配节点 → 所属用户 ----
    # node_id 格式: node_{user_id前8位}，如 node_38f5c62c → 用户 id 含 38f5c62c
    try:
        from auth_manager import get_auth_manager
        auth_mgr = get_auth_manager()
        users_list = auth_mgr.list_users()  # [{id, nickname, ...}]
        # 构建 id_prefix -> user 映射（取每个用户 id 的前 8 位）
        user_map = {}  # prefix -> {id, nickname, role}
        for u in users_list:
            prefix = str(u.get("id", ""))[:8]
            if prefix:
                user_map[prefix] = u
    except Exception:
        user_map = {}

    for node in nodes:
        nid = node.get("node_id", "")
        # node_38f5c62c → 提取 38f5c62c
        if nid.startswith("node_"):
            prefix = nid[5:13]  # 取 "node_" 后 8 位
            owner = user_map.get(prefix)
            if owner:
                node["owner_user"] = {
                    "id": owner.get("id"),
                    "nickname": owner.get("nickname", "未知用户"),
                    "role": owner.get("role", "user"),
                }
            else:
                node["owner_user"] = None
        else:
            node["owner_user"] = None

    return {
        "success": True,
        "data": nodes,
        "total": len(nodes),
        "timestamp": status["last_update"]
    }


@app.get("/api/nodes/{node_id}", response_model=Dict)
async def get_node(node_id: str):
    """
    获取单个节点的详细信息
    
    包括设备信息、已加载模型、连接状态等
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    node_detail = manager.get_node_detail(node_id)
    
    if not node_detail:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 不存在")
    
    return {
        "success": True,
        "data": node_detail
    }


def validate_and_enrich_device_info(device_info: Dict[str, Any], node_id: str) -> Dict[str, Any]:
    """
    ✨ 验证和补全 device_info 数据
    
    确保返回给前端的 device_info 始终包含完整的必要字段：
    - model: 设备型号
    - chip: 芯片型号  
    - memory: 内存大小 (MB)
    - flops: 算力信息 {fp32, fp16, int8}
    - memory_detail: 详细显存信息 {total, free, used} (可选)
    
    Args:
        device_info: Node 发送的原始设备信息
        node_id: 节点 ID（用于日志）
    
    Returns:
        Dict: 补全后的设备信息
    """
    if not device_info or not isinstance(device_info, dict):
        device_info = {}
    
    # 检查必要字段是否完整
    required_fields = {
        "model": "Unknown Device",
        "chip": "Unknown Chip",
        "memory": 0,
        "flops": {"fp32": 0, "fp16": 0, "int8": 0}
    }
    
    is_complete = all(
        field in device_info and device_info[field] is not None 
        for field in required_fields.keys()
    )
    
    if not is_complete:
        logger.info(f"🔧 [Device Info] 节点 {node_id} 的 device_info 不完整，正在补全...")
        
        # 补全缺失字段
        for field, default_value in required_fields.items():
            if field not in device_info or device_info[field] is None:
                if field == "flops" and isinstance(device_info.get(field), dict):
                    # flops 字段存在但不完整，补全子字段
                    for sub_field in ["fp32", "fp16", "int8"]:
                        if sub_field not in device_info["field"]:
                            device_info[field][sub_field] = 0
                else:
                    device_info[field] = default_value
                    logger.debug(f"   补全字段: {field} = {default_value}")
        
        # 如果有 memory_detail 但没有 memory，从 memory_detail 推断
        if ("memory" not in device_info or device_info.get("memory", 0) == 0) and "memory_detail" in device_info:
            mem_detail = device_info["memory_detail"]
            if isinstance(mem_detail, dict) and mem_detail.get("total", 0) > 0:
                device_info["memory"] = mem_detail["total"]
                logger.info(f"   从 memory_detail 推断 memory = {mem_detail['total']}MB")
    
    # 移除不必要的 error 字段（如果数据已经完整）
    if "error" in device_info and is_complete:
        del device_info["error"]
        logger.debug(f"   移除 error 标记（数据已完整）")
    
    logger.info(f"✅ [Device Info] 节点 {node_id} 设备信息已验证: "
               f"{device_info.get('chip')} / {device_info.get('memory')}MB")
    
    return device_info


@app.post("/api/nodes", response_model=Dict)
async def add_node(node_data: Dict[str, Any]):
    """
    添加新节点到集群
    
    Request Body:
    {
        "node_id": "unique-id",
        "address": "192.168.1.100",
        "port": 50051,          (可选，gRPC端口，默认50051)
        "chatgpt_api_port": 52415,  (可选，HTTP API端口，默认52415)
        "device_info": {...}     (可选，节点主动注册时提供的设备信息)
    }
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    required_fields = ["node_id", "address"]
    for field in required_fields:
        if field not in node_data:
            raise HTTPException(status_code=400, detail=f"缺少必要字段: {field}")
    
    node_id = node_data["node_id"]
    address = node_data["address"]
    port = node_data.get("port", 50051)
    chatgpt_api_port = node_data.get("chatgpt_api_port", 52415)
    device_info = node_data.get("device_info", {})
    
    # ✅ 验证和补全 device_info（确保数据完整）
    device_info = validate_and_enrich_device_info(device_info, node_id)
    
    success = await manager.add_node(node_id, address, port, chatgpt_api_port, device_info)
    
    node_info = manager.nodes.get(node_id)
    
    if node_info:
        is_online = node_info.status == NodeStatus.ONLINE
        return {
            "success": True,
            "message": f"节点 {node_id} 已{'注册并连接成功' if is_online else '注册（后台重试连接中）'}",
            "data": node_info.to_dict(),
            "connected": is_online
        }
    
    return {
        "success": False,
        "message": f"节点 {node_id} 添加失败"
    }


@app.delete("/api/nodes/{node_id}", response_model=Dict)
async def remove_node(node_id: str):
    """
    从集群中移除节点
    
    会断开与该节点的连接并清理资源
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    success = await manager.remove_node(node_id)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 不存在")
    
    return {
        "success": True,
        "message": f"节点 {node_id} 已移除"
    }


@app.post("/api/nodes/{node_id}/health-check", response_model=Dict)
async def health_check_node(node_id: str, request: Request):
    """
    节点心跳接口（双向通信）
    
    ✨ Node → Manager 方向：
    - 接收 Node 上报的实时状态（loaded_models, GPU显存等）
    - 更新 Manager 中的节点状态
    
    ✨ Manager → Node 方向：
    - 返回待处理的任务（Pull 模式）
    
    适用场景：内网/FRP 环境，Node 主动向 Manager 上报状态
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    connector = manager.connectors.get(node_id)

    # ✅ 先读取 body（只读一次，后续复用）
    raw_body = b""
    try:
        raw_body = await request.body()
    except Exception:
        pass

    if not connector:
        # Server 重启后节点丢失，通过心跳自动重新注册
        logger.info(f"[Heartbeat] 节点 {node_id} 不存在（可能 Server 重启），尝试自动重新注册...")
        try:
            reg_device_info = {}
            reg_address = "127.0.0.1"
            reg_port = 50051
            reg_chatgpt_port = 52415
            if raw_body:
                try:
                    hb_data = json.loads(raw_body)
                    if "gpu_memory" in hb_data:
                        reg_device_info["memory_detail"] = hb_data["gpu_memory"]
                    # 从心跳数据提取节点真实网络地址（用于构建 chatgpt_url）
                    if "address" in hb_data:
                        reg_address = hb_data["address"]
                    if "port" in hb_data:
                        reg_port = int(hb_data["port"])
                    if "chatgpt_api_port" in hb_data:
                        reg_chatgpt_port = int(hb_data["chatgpt_api_port"])
                except Exception:
                    pass

            reg_success = await manager.add_node(
                node_id=node_id,
                address=reg_address,
                port=reg_port,
                chatgpt_api_port=reg_chatgpt_port,
                device_info=reg_device_info or {}
            )

            if reg_success:
                logger.info(f"[Heartbeat] 节点 {node_id} 自动注册成功")
                connector = manager.connectors.get(node_id)
            else:
                logger.warning(f"[Heartbeat] 节点 {node_id} 自动注册失败")
                raise HTTPException(status_code=503, detail=f"节点 {node_id} 重新注册失败，请手动添加")

        except HTTPException:
            raise
        except Exception as reg_err:
            logger.error(f"[Heartbeat] 节点 {node_id} 自动注册异常: {reg_err}")
            raise HTTPException(status_code=503, detail=f"节点 {node_id} 不存在且自动注册失败: {reg_err}")
    
    # ✅ 解析 Node 上报的状态数据（复用已读取的 body）
    try:
        heartbeat_data = json.loads(raw_body) if raw_body else {}
        
        # 更新已加载模型列表
        if "loaded_models" in heartbeat_data:
            loaded_models = heartbeat_data["loaded_models"]
            if loaded_models:
                node_info = manager.nodes[node_id]
                old_count = len(node_info.loaded_models)
                node_info.loaded_models = loaded_models
                new_count = len(loaded_models)
                
                model_ids = [m.get("model_id", "?") for m in loaded_models]
                logger.info(f"📥 [Heartbeat] 节点 {node_id} 上报已加载模型: "
                           f"{new_count}个 - {model_ids}")
                
                if new_count != old_count:
                    logger.info(f"   模型数量变化: {old_count} → {new_count}")
        
        # 更新 GPU 显存信息
        if "gpu_memory" in heartbeat_data:
            gpu_memory = heartbeat_data["gpu_memory"]
            if gpu_memory and isinstance(gpu_memory, dict):
                node_info = manager.nodes[node_id]
                if "memory_detail" not in node_info.device_info:
                    node_info.device_info["memory_detail"] = {}

                node_info.device_info["memory_detail"].update({
                    "total": gpu_memory.get("total", 0),
                    "free": gpu_memory.get("free", 0),
                    "used": gpu_memory.get("used", 0)
                })

                logger.debug(f"📊 [Heartbeat] 节点 {node_id} GPU显存: "
                            f"{gpu_memory.get('used', 0)}/{gpu_memory.get('total', 0)} MB")

        # 更新节点网络地址（确保 chatgpt_url 始终正确）
        if "address" in heartbeat_data or "chatgpt_api_port" in heartbeat_data:
            node_info = manager.nodes[node_id]
            addr_changed = False
            if "address" in heartbeat_data and heartbeat_data["address"] != node_info.address:
                old_addr = node_info.address
                node_info.address = heartbeat_data["address"]
                logger.info(f"🌐 [Heartbeat] 节点 {node_id} 地址更新: {old_addr} → {node_info.address}")
                addr_changed = True
            if "chatgpt_api_port" in heartbeat_data:
                new_port = int(heartbeat_data["chatgpt_api_port"])
                if new_port != node_info.chatgpt_api_port:
                    old_port = node_info.chatgpt_api_port
                    node_info.chatgpt_api_port = new_port
                    logger.info(f"🌐 [Heartbeat] 节点 {node_id} HTTP端口更新: {old_port} → {new_port}")
                    addr_changed = True
            if addr_changed:
                logger.info(f"🌐 [Heartbeat] 节点 {node_id} chatgpt_url → {node_info.chatgpt_url}")
        
        # ✅ 验证和补全 device_info（确保每次心跳后数据完整）
        if node_id in manager.nodes:
            node_info = manager.nodes[node_id]
            node_info.device_info = validate_and_enrich_device_info(
                node_info.device_info, 
                node_id
            )
    
    except Exception as e:
        error_msg = str(e)
        
        # JSON 解析错误（空 body 或非法数据）→ 容错处理，视为无状态数据的纯心跳
        if "Expecting value" in error_msg or "JSON" in error_msg.upper():
            logger.debug(f"[Heartbeat] 节点 {node_id} 心跳 body 为空，跳过状态解析，继续保活")
            # 不标记为 ERROR，直接走后续的 health_check 保活逻辑
            heartbeat_data = {}
        else:
            logger.warning(f"⚠️ [Heartbeat] 解析节点 {node_id} 心跳数据失败: {error_msg}")
            if node_id in manager.nodes:
                manager.nodes[node_id].status = NodeStatus.ERROR
                manager.nodes[node_id].error_message = f"心跳数据解析失败: {error_msg}"
                manager.nodes[node_id].last_heartbeat = time.time()
            
            return {
                "success": False,
                "error": f"心跳数据格式错误: {error_msg}",
                "data": {
                    "node_id": node_id,
                    "is_healthy": False,
                    "status": "error",
                    "suggestion": "Node 进程可能已崩溃，请检查 Node 日志"
                }
            }
    
    is_healthy = await connector.health_check()
    node_info = manager.nodes[node_id]

    if is_healthy:
        # 始终更新心跳时间戳，避免心跳监控误判为离线
        node_info.last_heartbeat = time.time()

        # 心跳成功时恢复离线节点的在线状态
        if node_info.status == NodeStatus.OFFLINE:
            old_status = node_info.status.value
            node_info.status = NodeStatus.ONLINE
            node_info.error_message = ""
            logger.info(f"💓 [Heartbeat] 节点 {node_id} 心跳恢复: {old_status} → online")

            try:
                await ws_manager.broadcast({
                    "type": "node_status_changed",
                    "node_id": node_id,
                    "status": "online",
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.debug(f"广播状态变化失败: {e}")
    
    # ✨ 获取该节点的待处理任务
    pending_tasks = []
    if node_id in manager.pending_tasks and manager.pending_tasks[node_id]:
        pending_tasks = manager.pending_tasks[node_id].copy()
        manager.pending_tasks[node_id] = []  # 清空队列
        
        if pending_tasks:
            logger.info(f"📤 [Heartbeat] 向节点 {node_id} 交付 {len(pending_tasks)} 个待处理任务")
    
    return {
        "success": True,
        "data": {
            "node_id": node_id,
            "is_healthy": is_healthy,
            "status": node_info.status.value,
            "response_time_ms": node_info.response_time_ms,
            "error": node_info.error_message if not is_healthy else None,
            "pending_tasks": pending_tasks
        }
    }


# ==================== 节点间中继 API (Relay) ====================
# ✨ 通过 Manager 作为跳板，帮助节点间传递 gRPC 消息
# 适用场景：NAT/防火墙环境下，节点无法直接建立 gRPC 连接

@app.post("/api/relay/{target_node_id}/tensor", response_model=Dict)
async def relay_tensor(target_node_id: str, request: Request):
    """
    中继张量数据（隐藏状态传递）
    
    这是 exo 分布式推理的核心操作！
    Node A → Manager → Node B 传递 hidden state tensor
    
    Request Body:
    {
        "source_node_id": "node-a",
        "shard": {...},
        "tensor_data": "<base64>",
        "shape": [batch, seq_len, hidden_dim],
        "dtype": "float32",
        "request_id": "req-123",
        "inference_state": {...}
    }
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    if not node_ws_manager.is_node_connected(target_node_id):
        raise HTTPException(status_code=404, detail=f"目标节点 {target_node_id} 未连接到 Manager")
    
    try:
        data = await request.json()
        source_node_id = data.get("source_node_id", "unknown")
        
        logger.info(f"🔀 [Relay] 张量中继: {source_node_id} → {target_node_id} "
                   f"(shape={data.get('shape')}, dtype={data.get('dtype')})")
        
        # 通过 WebSocket 转发给目标节点
        ws = node_ws_manager.node_connections[target_node_id]
        
        relay_msg = {
            "type": "grpc_relay",
            "method": "SendTensor",
            "source_node_id": source_node_id,
            "payload": data,
            "timestamp": time.time()
        }
        
        await ws.send_text(json.dumps(relay_msg, ensure_ascii=False))
        
        # 等待响应（带超时）
        response_queue = asyncio.Queue()
        relay_request_id = f"relay_{uuid.uuid4().hex[:12]}"
        node_ws_manager.pending_requests[relay_request_id] = response_queue
        
        try:
            response = await asyncio.wait_for(response_queue.get(), timeout=120.0)
            
            if response is None or (isinstance(response, dict) and response.get("error")):
                error_detail = response.get("error", "Unknown error") if isinstance(response, dict) else "No response"
                logger.error(f"❌ [Relay] 张量中继失败: {source_node_id} → {target_node_id}: {error_detail}")
                return {
                    "success": False,
                    "error": error_detail,
                    "relay_path": f"{source_node_id} → Manager → {target_node_id}"
                }
            
            logger.info(f"✅ [Relay] 张量中继成功: {source_node_id} → {target_node_id}")
            return {
                "success": True,
                "data": response,
                "relay_path": f"{source_node_id} → Manager → {target_node_id}"
            }
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ [Relay] 张量中继超时: {source_node_id} → {target_node_id}")
            raise HTTPException(status_code=504, detail="中继超时（120秒）")
            
        finally:
            node_ws_manager.pending_requests.pop(relay_request_id, None)
            
    except json.JSONDecodeError as e:
        logger.error(f"❌ [Relay] JSON 解析失败: {e}")
        raise HTTPException(status_code=400, detail=f"无效的 JSON 数据: {e}")
    except Exception as e:
        logger.error(f"❌ [Relay] 张量中继异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"中继失败: {str(e)}")


@app.post("/api/relay/{target_node_id}/prompt", response_model=Dict)
async def relay_prompt(target_node_id: str, request: Request):
    """
    中继 Prompt 请求（推理触发）
    
    Node A → Manager → Node B 发送推理 prompt
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    if not node_ws_manager.is_node_connected(target_node_id):
        raise HTTPException(status_code=404, detail=f"目标节点 {target_node_id} 未连接")
    
    try:
        data = await request.json()
        source_node_id = data.get("source_node_id", "unknown")
        
        logger.info(f"🔀 [Relay] Prompt 中继: {source_node_id} → {target_node_id}")
        
        ws = node_ws_manager.node_connections[target_node_id]
        
        relay_msg = {
            "type": "grpc_relay",
            "method": "SendPrompt",
            "source_node_id": source_node_id,
            "payload": data
        }
        
        await ws.send_text(json.dumps(relay_msg))
        
        return {
            "success": True,
            "message": "Prompt 已转发",
            "relay_path": f"{source_node_id} → Manager → {target_node_id}"
        }
        
    except Exception as e:
        logger.error(f"❌ [Relay] Prompt 中继失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/relay/{target_node_id}/topology", response_model=Dict)
async def relay_topology(target_node_id: str, request: Request):
    """
    中继拓扑收集请求
    
    Node A → Manager → Node B 收集拓扑信息
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    if not node_ws_manager.is_node_connected(target_node_id):
        raise HTTPException(status_code=404, detail=f"目标节点 {target_node_id} 未连接")
    
    try:
        data = await request.json()
        source_node_id = data.get("source_node_id", "unknown")
        
        logger.info(f"🔀 [Relay] 拓扑收集中继: {source_node_id} → {target_node_id}")
        
        ws = node_ws_manager.node_connections[target_node_id]
        
        relay_msg = {
            "type": "grpc_relay",
            "method": "CollectTopology",
            "source_node_id": source_node_id,
            "payload": data
        }
        
        await ws.send_text(json.dumps(relay_msg))
        
        # 等待拓扑响应
        response_queue = asyncio.Queue()
        relay_request_id = f"relay_topo_{uuid.uuid4().hex[:12]}"
        node_ws_manager.pending_requests[relay_request_id] = response_queue
        
        try:
            response = await asyncio.wait_for(response_queue.get(), timeout=30.0)
            
            return {
                "success": True,
                "topology": response,
                "relay_path": f"{source_node_id} → Manager → {target_node_id}"
            }
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="拓扑收集中继超时")
            
        finally:
            node_ws_manager.pending_requests.pop(relay_request_id, None)
            
    except Exception as e:
        logger.error(f"❌ [Relay] 拓扑中继失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/relay/{target_node_id}/health-check", response_model=Dict)
async def relay_health_check(target_node_id: str, request: Request):
    """
    中继健康检查请求
    """
    if not node_ws_manager.is_node_connected(target_node_id):
        return {
            "success": False,
            "is_healthy": False,
            "error": f"节点 {target_node_id} 未连接到 Manager"
        }
    
    try:
        data = await request.json() if await request.body() else {}
        source_node_id = data.get("source_node_id", "manager")
        
        logger.debug(f"🔀 [Relay] 健康检查中继: {source_node_id} → {target_node_id}")
        
        ws = node_ws_manager.node_connections[target_node_id]
        
        relay_msg = {
            "type": "grpc_relay",
            "method": "HealthCheck",
            "source_node_id": source_node_id,
            "payload": {}
        }
        
        await ws.send_text(json.dumps(relay_msg))
        
        return {
            "success": True,
            "is_healthy": True,  # WebSocket 连接正常即视为健康
            "connection_type": "websocket_relay",
            "relay_path": f"{source_node_id} → Manager → {target_node_id}"
        }
        
    except Exception as e:
        logger.error(f"❌ [Relay] 健康检查中继失败: {e}")
        return {
            "success": False,
            "is_healthy": False,
            "error": str(e)
        }


# ==================== 集群状态 API ====================

@app.get("/api/cluster/status", response_model=Dict)
async def cluster_status():
    """
    获取集群整体状态摘要
    
    包括：在线节点数、总内存、已加载模型等统计信息
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    status = manager.get_cluster_status()
    
    # 计算一些额外的聚合指标
    online_nodes = [n for n in status["nodes"] if n["status"] == "online"]
    
    logger.info(f"🔍 [Cluster Status] 在线节点数: {len(online_nodes)}, 总节点: {status.get('total_nodes', 0)}")
    
    # 打印每个在线节点的 device_info（用于调试）
    for idx, node in enumerate(online_nodes):
        logger.info(f"   节点[{idx}] {node.get('node_id')}: "
                   f"memory={node.get('device_info', {}).get('memory', 'N/A')}, "
                   f"loaded_models={len(node.get('loaded_models', []))}个")
    
    total_memory_gb = sum(n["device_info"].get("memory", 0) for n in online_nodes) / 1024
    
    logger.info(f"🔍 [Cluster Status] 计算 total_memory_gb: {total_memory_gb}")
    
    # 计算已加载模型总数（去重）
    all_loaded_models = set()
    for n in online_nodes:
        for m in n.get("loaded_models", []):
            all_loaded_models.add(m.get("model_id", ""))
    
    # 计算健康度 (基于节点在线率 + 显存状态)
    health_percent = round(len(online_nodes) / max(status["total_nodes"], 1) * 100, 1)
    
    # 如果有在线节点且显存数据正常，健康度至少为 100
    if online_nodes and health_percent >= 100:
        health_percent = 100.0
    
    summary_data = {
        "total_nodes": status.get("total_nodes", 0),
        "online_nodes": len(online_nodes),
        "offline_nodes": status.get("total_nodes", 0) - len(online_nodes),
        "health_percent": health_percent,
        "total_memory_gb": round(total_memory_gb, 2),
        "total_models": len(all_loaded_models),
        "avg_response_time_ms": round(
            sum(n["response_time_ms"] for n in online_nodes) / len(online_nodes), 2
        ) if online_nodes else 0
    }
    
    logger.info(f"📊 [Cluster Status] 节点={summary_data['total_nodes']}, "
               f"显存={summary_data['total_memory_gb']}GB, "
               f"模型={summary_data['total_models']}, "
               f"健康度={summary_data['health_percent']}%")
    
    return {
        "success": True,
        "data": {
            **status,
            "summary": summary_data
        }
    }


# ==================== ✨ WebSocket 监控 API ====================

@app.get("/api/ws/status", response_model=Dict)
async def websocket_status():
    """
    ✨ 获取 WebSocket 连接状态（增强版）
    
    返回所有 Node 的 WebSocket 连接状态、统计信息、健康检查等
    """
    try:
        # 基础连接信息
        node_connections = {
            node_id: {
                "connected": True,
                "connection_id": id(ws),
            }
            for node_id, ws in node_ws_manager.node_connections.items()
        }
        
        # 待处理请求统计
        pending_stats = {
            "total": len(node_ws_manager.pending_requests),
            "requests": list(node_ws_manager.pending_requests.keys())[:20]  # 只显示前20个
        }
        
        # 执行健康检查（异步）
        health_results = await node_ws_manager.health_check_all_nodes()
        
        # 清理过期请求
        await node_ws_manager.cleanup_stale_requests()
        
        return {
            "success": True,
            "data": {
                "timestamp": time.time(),
                "node_connections": node_connections,
                "pending_requests": pending_stats,
                "health_check": health_results,
                "total_nodes_connected": len(node_ws_manager.node_connections),
                "version": "2.0"  # 标记为 V2 版本
            }
        }
        
    except Exception as e:
        logger.error(f"[WS Monitor] ❌ 获取状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ws/node/{node_id}/stats", response_model=Dict)
async def websocket_node_stats(node_id: str):
    """
    ✨ 获取指定节点的详细统计信息
    
    Args:
        node_id: 节点 ID
        
    Returns:
        Dict: 节点的详细统计和性能指标
    """
    stats = await node_ws_manager.get_node_stats(node_id)
    
    if not stats:
        raise HTTPException(
            status_code=404, 
            detail=f"Node {node_id} not found or not connected"
        )
    
    return {
        "success": True,
        "data": stats
    }


@app.post("/api/ws/broadcast", response_model=Dict)
async def websocket_broadcast(
    message: Dict = Body(...),
    exclude_nodes: Optional[List[str]] = None
):
    """
    ✨ 广播消息到所有连接的 Node
    
    Args:
        message: 要广播的消息内容
        exclude_nodes: 要排除的节点ID列表
        
    Returns:
        Dict: 广播结果（成功/失败数量）
    """
    sent_count = await node_ws_manager.broadcast_to_nodes(
        message=message,
        exclude_nodes=exclude_nodes
    )
    
    return {
        "success": True,
        "data": {
            "sent_count": sent_count,
            "total_targets": len(node_ws_manager.node_connections) - (len(exclude_nodes or [])),
            "message_type": message.get("type", "unknown")
        }
    }


@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """
    ✨ 实时监控端点（用于 Dashboard）- V2 版本
    
    推送实时 WebSocket 统计数据到前端，数据格式与 /api/ws/status 保持一致：
    
    数据结构：
    {
        "timestamp": float,
        "node_connections": {  # ✅ 对象格式（与 REST API 一致）
            "node_id": {
                "connected": bool,
                "connection_id": int
            }
        },
        "pending_requests": {
            "total": int,
            "requests": list
        },
        "health_check": {
            "node_id": bool
        },
        "total_nodes_connected": int,
        "version": "2.0"
    }
    """
    await websocket.accept()
    logger.info(f"🔗 [WS Monitor] 客户端已连接 (V2 格式)")

    try:
        while True:
            # 每5秒推送一次统计数据
            await asyncio.sleep(5)

            # 检查连接是否仍然有效（客户端可能已断开）
            if websocket.client_state.name != "CONNECTED":
                break

            try:
                # 构建与 /api/ws/status 完全一致的 V2 格式数据
                node_connections = {
                    node_id: {
                        "connected": True,
                        "connection_id": id(ws),
                    }
                    for node_id, ws in node_ws_manager.node_connections.items()
                }

                pending_stats = {
                    "total": len(node_ws_manager.pending_requests),
                    "requests": list(node_ws_manager.pending_requests.keys())[:20]
                }

                health_results = await node_ws_manager.health_check_all_nodes()

                monitor_data = {
                    "timestamp": time.time(),
                    "node_connections": node_connections,
                    "pending_requests": pending_stats,
                    "health_check": health_results,
                    "total_nodes_connected": len(node_ws_manager.node_connections),
                    "version": "2.0"
                }

                await websocket.send_text(json.dumps(monitor_data))

            except Exception as e:
                err_msg = str(e)
                # 忽略已关闭连接的发送错误（正常断开场景）
                if "close" in err_msg.lower() or "send" in err_msg.lower():
                    break
                logger.error(f"[WS Monitor] ❌ 构建监控数据失败: {e}")
                # 尝试发送错误状态，忽略发送失败
                try:
                    error_data = {
                        "timestamp": time.time(),
                        "error": str(e),
                        "version": "2.0"
                    }
                    await websocket.send_text(json.dumps(error_data))
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info(f"🔌 [WS Monitor] 客户端断开")
    except Exception as e:
        err_msg = str(e)
        if "close" not in err_msg.lower() and "send" not in err_msg.lower():
            logger.error(f"[WS Monitor] ❌ 错误: {e}")


@app.get("/api/ws/node/{node_id}/stats", response_model=Dict)
async def cluster_stats():
    """
    获取详细的统计数据
    
    用于图表展示等场景
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    status = manager.get_cluster_status()
    
    # 构建统计数据用于可视化
    stats = {
        "nodes_by_status": {},
        "devices_distribution": [],
        "memory_usage": []
    }
    
    for node in status["nodes"]:
        status_val = node["status"]
        stats["nodes_by_status"][status_val] = stats["nodes_by_status"].get(status_val, 0) + 1
        
        device = node["device_info"].get("chip", "Unknown")
        stats["devices_distribution"].append({
            "node_id": node["node_id"],
            "device": device,
            "memory_gb": round(node["device_info"].get("memory", 0) / 1024, 1)
        })
        
        memory_detail = node["device_info"].get("memory_detail")
        if memory_detail:
            stats["memory_usage"].append({
                "node_id": node["node_id"],
                "total_mb": memory_detail.get("total", 0),
                "free_mb": memory_detail.get("free", 0),
                "used_mb": memory_detail.get("used", 0)
            })
    
    return {
        "success": True,
        "data": stats,
        "timestamp": status["last_update"]
    }


# ==================== 模型管理 API ====================

@app.get("/api/models", response_model=Dict)
async def list_models():
    """
    列出集群中所有已加载的模型（支持多实例）
    
    收集各节点上的模型信息并汇总
    返回格式：
    - 单实例模型：直接返回 model_id
    - 多实例模型：每个实例单独一条记录（包含 instance_id）
    
    新增字段：
    - instance_id: 实例ID（default 表示单实例）
    - full_model_id: 完整模型ID（包含 ::instance_id）
    - is_multi_instance: 是否为多实例模型
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    flat_models = []
    seen_base_models = {}
    seen_full_ids = set()
    
    for node_id, node_info in manager.nodes.items():
        for model in node_info.loaded_models:
            model_id = model.get("model_id", "unknown")
            shard = model.get("shard", {})
            
            if "::" in model_id:
                parts = model_id.split("::", 1)
                base_model_id = parts[0]
                instance_id = parts[1]
                full_model_id = model_id
                is_multi_instance = True
            else:
                base_model_id = model_id
                instance_id = "default"
                full_model_id = model_id
                is_multi_instance = False
            
            if full_model_id in seen_full_ids:
                continue
            seen_full_ids.add(full_model_id)
            
            if base_model_id not in seen_base_models:
                seen_base_models[base_model_id] = 0
            seen_base_models[base_model_id] += 1
            
            flat_models.append({
                "model_id": base_model_id,           # 基础模型ID（用于分组）
                "full_model_id": full_model_id,      # 完整模型ID（含实例信息）
                "instance_id": instance_id,          # 实例ID
                "node_id": node_id,
                "is_multi_instance": is_multi_instance,  # 是否多实例
                "shard_info": {
                    "start_layer": shard.get("start_layer"),
                    "end_layer": shard.get("end_layer"),
                    "n_layers": shard.get("n_layers"),
                },
                "shard": shard,
            })
    
    return {
        "success": True,
        "data": {
            "models": flat_models,
            "total": len(flat_models),
            "base_models_count": len(seen_base_models),  # 基础模型数量
            "multi_instance_summary": {                  # 多实例摘要
                base_id: {
                    "instances_count": count,
                    "is_multi": count > 1
                }
                for base_id, count in seen_base_models.items()
                if count > 1  # 只显示有多实例的模型
            }
        }
    }


@app.get("/api/models/available", response_model=Dict)
async def get_available_models():
    """
    获取可用的模型列表（从 exo.models 配置读取）
    
    返回所有已配置的模型，包括模型ID、名称、层数等信息
    前端可用于下拉选择
    """
    # 唯一数据源：custom_models.json
    source = "custom" if custom_model_cards else "empty"

    models_list = []

    for model_id, config in custom_model_cards.items():
        layers = config.get("layers", 0)
        repo_info = config.get("repo", {})

        repo_name = "unknown"
        for engine_name, repo in repo_info.items():
            if 'PyTorch' in engine_name or 'Dummy' in engine_name:
                repo_name = repo
                break

        models_list.append({
            "model_id": model_id,
            "pretty_name": custom_pretty_names.get(model_id, model_id),
            "layers": layers,
            "repo": repo_name,
            "engines": list(repo_info.keys()),
            "source": "custom"
        })
    
    models_list.sort(key=lambda x: (x["source"] != "custom", x["pretty_name"]))
    
    return {
        "success": True,
        "data": {
            "models": models_list,
            "total": len(models_list),
            "source": source,
            "has_custom_models": len(custom_model_cards) > 0
        }
    }


# ==================== 自定义模型 CRUD API ====================

@app.get("/api/models/custom", response_model=Dict)
async def list_custom_models():
    """获取所有自定义模型配置"""
    models_list = []
    
    for model_id, config in custom_model_cards.items():
        models_list.append({
            "model_id": model_id,
            "pretty_name": custom_pretty_names.get(model_id, model_id),
            "layers": config.get("layers", 0),
            "repo": config.get("repo", {}),
            "engines": list(config.get("repo", {}).keys())
        })
    
    return {
        "success": True,
        "data": {
            "models": models_list,
            "total": len(models_list)
        }
    }

@app.post("/api/models/custom", response_model=Dict)
async def add_custom_model(request: Request):
    """添加自定义模型"""
    try:
        data = await request.json()
        
        model_id = data.get("model_id", "").strip()
        pretty_name = data.get("pretty_name", "").strip()
        layers = int(data.get("layers", 0))
        repo = data.get("repo", "")
        engine_type = data.get("engine_type", "PyTorchLlama3InferenceEngine")
        
        if not model_id:
            return {"success": False, "error": "模型ID不能为空"}
        
        if not repo:
            return {"success": False, "error": "模型仓库路径不能为空"}
        
        if model_id in custom_model_cards:
            return {"success": False, "error": f"模型 {model_id} 已存在，请使用更新接口"}
        
        custom_model_cards[model_id] = {
            "layers": layers,
            "repo": {engine_type: repo}
        }
        
        if pretty_name:
            custom_pretty_names[model_id] = pretty_name
        else:
            custom_pretty_names[model_id] = model_id
        
        save_custom_models()
        
        logger.info(f"[Models] 添加自定义模型: {model_id} ({pretty_name or model_id})")
        
        return {
            "success": True,
            "message": f"模型 {model_id} 添加成功",
            "data": {
                "model_id": model_id,
                "pretty_name": custom_pretty_names[model_id],
                "layers": layers,
                "repo": repo
            }
        }
        
    except Exception as e:
        logger.error(f"[Models] 添加自定义模型失败: {e}")
        return {"success": False, "error": str(e)}

@app.put("/api/models/custom/{model_id}", response_model=Dict)
async def update_custom_model(model_id: str, request: Request):
    """更新自定义模型"""
    try:
        data = await request.json()
        
        if model_id not in custom_model_cards:
            return {"success": False, "error": f"模型 {model_id} 不存在"}
        
        config = custom_model_cards[model_id]
        
        if "pretty_name" in data and data["pretty_name"].strip():
            custom_pretty_names[model_id] = data["pretty_name"].strip()
        
        if "layers" in data:
            config["layers"] = int(data["layers"])
        
        if "repo" in data and data["repo"].strip():
            engine_type = data.get("engine_type", list(config.get("repo", {}).keys())[0] if config.get("repo") else "PyTorchLlama3InferenceEngine")
            config["repo"] = {engine_type: data["repo"].strip()}
        
        if "engine_type" in data and "repo" in config:
            old_repo = list(config["repo"].values())[0] if config["repo"] else ""
            config["repo"] = {data["engine_type"]: old_repo}
        
        save_custom_models()
        
        logger.info(f"[Models] 更新自定义模型: {model_id}")
        
        return {
            "success": True,
            "message": f"模型 {model_id} 更新成功",
            "data": {
                "model_id": model_id,
                "pretty_name": custom_pretty_names.get(model_id),
                **config
            }
        }
        
    except Exception as e:
        logger.error(f"[Models] 更新自定义模型失败: {e}")
        return {"success": False, "error": str(e)}

@app.delete("/api/models/custom/{model_id}", response_model=Dict)
async def delete_custom_model(model_id: str):
    """删除自定义模型"""
    try:
        if model_id not in custom_model_cards:
            return {"success": False, "error": f"模型 {model_id} 不存在"}
        
        del custom_model_cards[model_id]
        
        if model_id in custom_pretty_names:
            del custom_pretty_names[model_id]
        
        save_custom_models()
        
        logger.info(f"[Models] 删除自定义模型: {model_id}")
        
        return {
            "success": True,
            "message": f"模型 {model_id} 删除成功"
        }
        
    except Exception as e:
        logger.error(f"[Models] 删除自定义模型失败: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/models/instances/{model_id}", response_model=Dict)
async def get_model_instances(model_id: str):
    """
    获取指定模型的所有实例信息（支持多实例）
    
    返回该模型的所有已加载实例，包括：
    - 实例ID
    - 所在节点
    - 分片信息
    - 加载状态
    
    Args:
        model_id: 模型ID
        
    Returns:
        实例列表
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    instances = manager.get_model_instances(model_id)
    
    return {
        "success": True,
        "data": {
            "model_id": model_id,
            "instances": instances,
            "total_instances": len(instances)
        }
    }


@app.get("/api/models/instances", response_model=Dict)
async def get_all_instances():
    """
    获取所有模型的多实例摘要
    
    返回每个模型加载了多少个实例
    
    Returns:
        {model_id: instance_count}
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    summary = manager.get_all_instances_summary()
    
    return {
        "success": True,
        "data": {
            "summary": summary,
            "total_models": len(summary),
            "total_instances": sum(summary.values())
        }
    }


# ==================== GPU池管理 API ====================

def _collect_loaded_models(status: Dict, manager) -> List[Dict]:
    """
    收集所有已加载的模型信息
    
    优先从 status["nodes"] 获取，如果为空则 fallback 到 manager.nodes
    """
    models_from_status = [m for n in status.get("nodes", []) for m in n.get("loaded_models", [])]
    
    if models_from_status:
        return models_from_status
    
    if manager and manager.nodes:
        models_from_manager = []
        for node_id, node_info in manager.nodes.items():
            if node_info.loaded_models:
                for m in node_info.loaded_models:
                    entry = dict(m)
                    if "node_id" not in entry:
                        entry["node_id"] = node_id
                    models_from_manager.append(entry)
        if models_from_manager:
            logger.info(f"[Pool] loaded_models fallback: 从 manager.nodes 获取到 {len(models_from_manager)} 个模型")
            return models_from_manager
    
    return []


def _get_gpu_compute_utilization() -> float:
    """
    获取 GPU 计算利用率（SM 利用率）
    
    使用 pynvml 的 nvmlDeviceGetUtilizationRates 获取真实的 GPU 计算利用率，
    而不是显存使用率。
    
    Returns:
        float: GPU 计算利用率 (0-100)，获取失败返回 0
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        gpu_util = util_rates.gpu  # GPU 计算利用率 (SM)
        mem_util = util_rates.memory  # 显存控制器利用率
        
        logger.debug(f"[GPU Util] compute={gpu_util}%, memory_controller={mem_util}%")
        
        return float(gpu_util)
    except Exception as e:
        logger.warning(f"[GPU Util] 获取GPU计算利用率失败: {e}")
        return 0.0


@app.get("/api/pool/status", response_model=Dict)
async def pool_status():
    """
    获取统一GPU显存池的状态
    
    显示整个集群作为一个逻辑GPU池的资源使用情况
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    status = manager.get_cluster_status()
    
    # 计算池级别的资源汇总
    total_memory = 0
    available_memory = 0
    used_memory = 0
    nodes_with_gpu = []
    
    for node in status["nodes"]:
        if node["status"] == "online":
            mem_detail = node["device_info"].get("memory_detail")
            total_mem = node["device_info"].get("memory", 0)
            
            # fallback: 如果静态 memory=0 但 pynvml 有实时数据，用 realtime total
            if (total_mem == 0 or total_mem is None) and mem_detail and mem_detail.get("total", 0) > 0:
                total_mem = mem_detail["total"]
                logger.info(f"   [Fallback] 节点 {node['node_id']}: memory=0, 使用 pynvml total={total_mem}MB")
            
            logger.info(f"[Pool Status] 节点 {node['node_id']}: "
                       f"memory={total_mem}MB, "
                       f"mem_detail={mem_detail}")
            
            if mem_detail:
                free_mb = mem_detail.get("free", 0)
                used_mb = mem_detail.get("used", 0)
                # free 不应超过 total
                if free_mb > total_mem and total_mem > 0:
                    logger.warning(f"   [修正] free({free_mb}) > total({total_mem}, 限制为 total")
                    free_mb = min(free_mb, total_mem)
                available_memory += free_mb
                used_memory += used_mb
                logger.info(f"   -> free={free_mb}MB, used={used_mb}MB")
            
            total_memory += total_mem
            
            # device 名称 fallback（多层 fallback 策略）
            chip_name = node["device_info"].get("chip", "") or ""
            
            # 第一层：如果 chip 为空/未知，尝试从显存数据推断
            if not chip_name or chip_name.lower() in ("unknown", "n/a", "unknown chip"):
                if mem_detail and mem_detail.get("total", 0) > 100:  # > 100MB 说明有 GPU
                    chip_name = "GPU"
                elif total_mem and total_mem > 100:  # 静态 memory > 100MB
                    chip_name = "GPU"
                else:
                    chip_name = "CPU"  # 无 GPU 或无法检测
            
            # 第二层：标准化已知名称
            if chip_name.lower() in ("gpu",):
                chip_name = "GPU"
            elif chip_name.lower() in ("cpu",):
                chip_name = "CPU"
            
            nodes_with_gpu.append({
                "node_id": node["node_id"],
                "device": chip_name,
                "total_gb": round(total_mem / 1024, 1),
                "available_gb": round(mem_detail.get("free", 0) / 1024, 1) if mem_detail else 0,
                "utilization": _get_gpu_compute_utilization() if mem_detail else 0,
                "memory_utilization": round(mem_detail.get("used", 0) / total_mem * 100, 1) if total_mem > 0 and mem_detail else 0
            })
    
    logger.info(f"📊 [Pool Status] 汇总: total={total_memory}MB, "
               f"used={used_memory}MB, free={available_memory}MB")
    
    pool_info = {
        "pool_name": "EXO Unified GPU Pool",
        "total_nodes": len(nodes_with_gpu),
        "memory": {
            "total_gb": round(total_memory / 1024, 2),
            "available_gb": round(available_memory / 1024, 2),
            "used_gb": round(used_memory / 1024, 2),
            "utilization_percent": round(sum(node.get("utilization", 0) for node in nodes_with_gpu) / len(nodes_with_gpu), 2) if nodes_with_gpu else 0,
            "memory_utilization_percent": round(used_memory / total_memory * 100, 2) if total_memory > 0 else 0
        },
        "nodes": nodes_with_gpu,
        "loaded_models": _collect_loaded_models(status, manager)
    }
    
    return {
        "success": True,
        "data": pool_info
    }


@app.post("/api/pool/load-model", response_model=Dict)
async def pool_load_model(request: Dict[str, Any]):
    """
    将模型加载到统一GPU池（支持多实例）
    
    ✨ 简化版API：前端只需提供模型信息和数量，后端全权管理实例ID
    
    Request Body:
    {
        "model_id": "qwen3-0.6b",           (必需)
        "instance_count": 2,                (可选，要创建的实例数量，默认1)
        "model_path": "./models/qwen3-0.6b", (可选，默认由系统自动查找)
        "n_layers": 24,                     (可选，默认从模型配置自动获取)
        "strategy": "smart",              (可选: smart(默认), memory_weighted, uniform, performance_weighted)
                                           smart=智能策略: 单节点优先+安全余量检查，能不拆就不拆
        "target_nodes": ["node1", "node2"]   (可选，指定目标节点)
    }
    
    Response:
    # 单个实例
    {
        "success": true,
        "model_id": "qwen3-0.6b",
        "instances": [
            {
                "instance_id": "worker-1",
                "full_model_id": "qwen3-0.6b::worker-1",
                "success": true,
                "allocation": { ... }
            }
        ],
        "summary": {"total": 1, "success": 1, "failed": 0}
    }
    
    # 多个实例
    {
        "success": true,
        "model_id": "qwen3-0.6b",
        "instances": [
            {"instance_id": "worker-3", "full_model_id": "qwen3-0.6b::worker-3", "success": true},
            {"instance_id": "worker-4", "full_model_id": "qwen3-0.6b::worker-4", "success": true}
        ],
        "summary": {"total": 2, "success": 2, "failed": 0}
    }
    
    示例:
    # 加载单个实例（自动分配ID）
    POST /api/pool/load-model {"model_id": "qwen3-0.6b"}
    
    # 加载2个实例（后端自动分配唯一ID，避免冲突）
    POST /api/pool/load-model {"model_id": "qwen3-0.6b", "instance_count": 2}
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    model_id = request.get("model_id")
    instance_count = request.get("instance_count", 1)  # ✅ 默认创建1个实例
    n_layers = request.get("n_layers")
    strategy = request.get("strategy", "smart")  # 默认使用智能策略
    target_nodes = request.get("target_nodes")
    model_path = request.get("model_path", "")
    
    if not model_id:
        raise HTTPException(status_code=400, detail="必须提供 model_id")
    
    # 验证实例数量
    try:
        instance_count = int(instance_count)
        if instance_count < 1:
            instance_count = 1
        elif instance_count > 10:
            instance_count = 10  # 安全限制
    except (ValueError, TypeError):
        instance_count = 1
    
    logger.info(f"📥 收到加载模型请求: model_id={model_id}, instance_count={instance_count}, model_path={model_path}")
    
    # 从配置文件中获取层数信息和默认路径
    all_model_configs = custom_model_cards
    
    if model_id in all_model_configs:
        config = all_model_configs[model_id]
        if not n_layers:
            n_layers = config.get("layers", 32)
        if not model_path:
            model_path = config.get("path", "")
            if not model_path:
                repo_info = config.get("repo", {})
                if isinstance(repo_info, dict):
                    for engine_name, repo in repo_info.items():
                        if 'PyTorch' in engine_name or 'Dummy' in engine_name:
                            model_path = repo
                            break
                elif isinstance(repo_info, str):
                    model_path = repo_info
            if model_path:
                logger.info(f"📋 从配置中获取模型路径: {model_path}")
            else:
                logger.warning(f"⚠️ 模型 {model_id} 未找到路径配置，将使用空路径")
        logger.info(f"📋 模型配置: {model_id}, 层数={n_layers}")
    else:
        if not n_layers:
            n_layers = 32
        logger.warning(f"⚠️ 模型 {model_id} 不在标准配置中，使用默认层数 {n_layers}")
    
    # ✅ 核心改进：使用批量加载方法，后端全权管理 instance_id
    result = await manager.load_multiple_instances(
        model_id=model_id,
        model_path=model_path,
        n_layers=int(n_layers),
        strategy=strategy,
        target_nodes=target_nodes,
        count=instance_count
    )
    
    logger.info(f"📋 批量加载完成: model_id={model_id}, "
               f"requested={instance_count}, success={result.get('summary', {}).get('success', 0)}")
    
    return result


@app.post("/api/pool/unload-model", response_model=Dict)
async def pool_unload_model(request: Dict[str, Any]):
    """
    从GPU池卸载模型（支持多实例）
    
    Request Body:
    {
        "model_id": "qwen3-0.6b",           (必需)
        "instance_id": "worker-1",          (可选，指定要卸载的实例ID)
        "unload_all_instances": false       (可选，是否卸载该模型的所有实例)
    }
    
    示例:
    # 卸载默认实例
    POST /api/pool/unload-model {"model_id": "qwen3-0.6b"}
    
    # 卸载指定实例
    POST /api/pool/unload-model {"model_id": "qwen3-0.6b", "instance_id": "worker-1"}
    
    # 卸载所有实例
    POST /api/pool/unload-model {"model_id": "qwen3-0.6b", "unload_all_instances": true}
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    model_id = request.get("model_id")
    instance_id = request.get("instance_id")
    unload_all_instances = request.get("unload_all_instances", False)
    
    if not model_id:
        raise HTTPException(status_code=400, detail="必须提供 model_id")
    
    logger.info(f"🗑️ 收到卸载模型请求: model_id={model_id}, instance_id={instance_id}, unload_all={unload_all_instances}")
    
    result = await manager.unload_model_from_cluster(
        model_id=model_id,
        instance_id=instance_id,
        unload_all_instances=unload_all_instances
    )
    
    logger.info(f"📋 卸载模型请求完成: model_id={model_id}, success={result.get('success')}")
    
    return result


@app.post("/api/pool/rebalance", response_model=Dict)
async def pool_rebalance(request: Dict[str, Any]):
    """
    重新平衡模型分布
    
    当有新节点加入或资源变化时调用
    
    Request Body:
    {
        "model_id": "Qwen/Qwen3-4B"
    }
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    model_id = request.get("model_id")
    
    if not model_id:
        raise HTTPException(status_code=400, detail="必须提供 model_id")
    
    result = await manager.rebalance_model(model_id)
    return result


# ==================== 推理 API ====================

@app.post("/api/inference/chat")
async def inference_chat(request: Dict[str, Any]):
    """
    聊天式推理接口 (SSE 流式输出) - 支持多实例负载均衡
    
    通过节点的 /v1/chat/completions API 进行分布式推理，
    支持 SSE (Server-Sent Events) 流式输出，逐 token 返回结果。
    
    新增功能:
    - 多实例负载均衡: 自动选择最优实例进行推理
    - 实例选择策略: 支持 round_robin/random/weighted/least_connections/first_layer
    - 指定实例: 可通过 instance_id 参数指定特定实例
    - 推理统计: 自动记录每个实例的推理性能
    
    Request Body:
    {
        "model_id": "qwen-3-0.6b",
        "instance_id": "worker-1",           // 可选：指定实例
        "lb_strategy": "round_robin",         // 可选：负载均衡策略
        "messages": [...],
        "stream": true,
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    负载均衡策略:
    - first_layer: 首层优先（默认）
    - round_robin: 轮询
    - random: 随机
    - weighted: 加权轮询（基于健康分数）
    - least_connections: 最少连接
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    model_id = request.get("model_id", "")
    instance_id = request.get("instance_id")              # 新增：指定实例ID
    lb_strategy_str = request.get("lb_strategy", "first_layer")  # 新增：负载均衡策略
    max_tokens = request.get("max_tokens", 512)
    temperature = request.get("temperature", 0.7)
    top_k = request.get("top_k", 50)
    top_p = request.get("top_p", 0.9)
    stream = request.get("stream", True)
    image_data = request.get("image")
    
    messages = request.get("messages")
    message = request.get("message", "").strip()
    
    if not messages and not message:
        raise HTTPException(status_code=400, detail="必须提供 messages 或 message")
    
    if not messages:
        messages = [{"role": "user", "content": message}]
    
    if image_data and image_data.get("base64"):
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is not None:
            existing_content = messages[last_user_idx]["content"]
            text_part = existing_content if isinstance(existing_content, str) else (
                next((p["text"] for p in existing_content if p.get("type") == "text"), "")
            )
            messages[last_user_idx]["content"] = [
                {"type": "text", "text": text_part or "请描述这张图片"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_data.get('mime_type', 'image/png')};base64,{image_data['base64']}"
                    }
                }
            ]
            logger.info(f"[Chat] 已注入图片到 messages[{last_user_idx}] ({image_data.get('mime_type', 'unknown')})")
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片"},
                    {"type": "image_url", "image_url": {"url": f"data:{image_data.get('mime_type', 'image/png')};base64,{image_data['base64']}"}}
                ]
            })
    
    # 初始化或获取负载均衡器
    lb = get_load_balancer()
    if lb.manager is None:
        lb.manager = manager
    
    # 解析负载均衡策略
    try:
        lb_strategy = LBStrategy(lb_strategy_str.lower())
    except ValueError:
        lb_strategy = LBStrategy.FIRST_LAYER
    
    # 使用负载均衡器选择实例
    selection_result = lb.select_instance(
        model_id=model_id,
        strategy=lb_strategy,
        preferred_instance=instance_id
    )
    
    if not selection_result:
        # 回退到原始逻辑
        available_models = _get_available_models_for_inference()
        
        if not model_id:
            model_id = available_models[0]["model_id"] if available_models else ""
        
        if not model_id or not any(m["model_id"] == model_id for m in available_models):
            raise HTTPException(
                status_code=400,
                detail=f"模型 {model_id} 未加载。可用: {[m['model_id'] for m in available_models]}"
            )
        
        target_node_id = _select_node_by_model(model_id)
        
        if not target_node_id:
            raise HTTPException(status_code=503, detail="没有可用的在线节点")
        
        logger.warning(f"[Chat] ⚠️ 负载均衡失败，回退到传统模式: node={target_node_id}")
    else:
        target_node_id = selection_result.selected_node_id
        
        logger.info(f"[Chat] 🎯 负载均衡选择完成:")
        logger.info(f"   模型: {model_id}")
        logger.info(f"   策略: {selection_result.strategy_used.value}")
        logger.info(f"   实例: {selection_result.selected_instance.instance_id}")
        logger.info(f"   节点: {target_node_id}")
        logger.info(f"   原因: {selection_result.selection_reason}")
        logger.info(f"   可用实例数: {selection_result.available_instances}")
    
    import uuid
    request_id = f"mgr_{uuid.uuid4().hex[:12]}"
    
    msg_preview = str(messages[-1]["content"])[:80] if messages else ""
    logger.info(f"[Chat] 推理: msgs={len(messages)}, last='{msg_preview}...', model={model_id}, node={target_node_id}, stream={stream}")
    
    connector = manager.connectors[target_node_id]
    
    # 记录请求开始时间（用于统计延迟）
    request_start_time = time.time()
    
    async def event_stream():
        inference_success = True
        inference_error = ""
        tokens_count = 0
        
        try:
            async for chunk in connector.send_inference_prompt(
                prompt=None, model_id=model_id, request_id=request_id,
                max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, stream=True,
                messages=messages,
            ):
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
                if chunk.get("finished"):
                    break
                
                if chunk.get("token"):
                    tokens_count += 1
                    
        except Exception as e:
            inference_success = False
            inference_error = str(e)
            yield f"data: {json.dumps({'success': False, 'error': str(e), 'finished': True})}\n\n"
        
        finally:
            # 记录推理完成统计
            latency_ms = (time.time() - request_start_time) * 1000
            
            if selection_result and hasattr(selection_result, 'selected_instance'):
                lb.record_completion(
                    node_id=target_node_id,
                    model_id=model_id,
                    success=inference_success,
                    latency=latency_ms,
                    tokens_generated=tokens_count,
                    error_message=inference_error
                )
                
                logger.info(f"[Chat] 📊 推理统计:")
                logger.info(f"   实例: {selection_result.selected_instance.instance_id}")
                logger.info(f"   节点: {target_node_id}")
                logger.info(f"   成功: {inference_success}")
                logger.info(f"   延迟: {latency_ms:.1f}ms")
                logger.info(f"   Tokens: {tokens_count}")
        
        yield "data: [DONE]\n\n"
    
    if stream:
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        final_result = None
        async for chunk in connector.send_inference_prompt(
            prompt=None, model_id=model_id, request_id=request_id,
            max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, stream=False,
            messages=messages,
        ):
            final_result = chunk
        if final_result is None:
            raise HTTPException(status_code=502, detail="推理未返回结果")
        final_result["request_id"] = request_id
        return final_result


@app.post("/api/inference/generate", response_model=Dict)
async def inference_generate(request: Dict[str, Any]):
    """
    文本生成接口（简化版）
    
    与 chat 接口类似，但返回格式更简单（非流式）
    
    Request Body:
    {
        "prompt": "写一首关于春天的诗",
        "model_id": "qwen-3-0.6b",
        "max_tokens": 256
    }
    
    也支持 messages 格式 (与 /api/inference/chat 相同):
    {
        "messages": [{"role": "user", "content": "..."}],
        "model_id": "qwen-3-0.6b"
    }
    """
    if "prompt" in request:
        request["message"] = request.pop("prompt")
    request["stream"] = False
    return await inference_chat(request)


@app.get("/api/inference/models", response_model=Dict)
async def get_inference_models():
    """
    获取可用于推理的模型列表（支持多实例）
    
    返回所有已加载模型的列表及其所在节点信息，
    包括每个模型的所有实例详情
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    models = _get_available_models_for_inference()
    
    # 获取负载均衡器统计
    lb = get_load_balancer()
    lb_stats = lb.get_statistics()
    
    return {
        "success": True,
        "data": {
            "models": models,
            "total": len(models),
            "load_balancer": {
                "enabled": True,
                "supported_strategies": [s.value for s in LBStrategy],
                "statistics": lb_stats
            }
        },
        "timestamp": time.time()
    }


# ==================== 负载均衡管理 API ====================

@app.get("/api/lb/status", response_model=Dict)
async def get_load_balancer_status():
    """
    获取负载均衡器状态和统计信息
    
    返回所有模型实例的负载情况、健康状态、性能统计等
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    lb = get_load_balancer()
    if lb.manager is None:
        lb.manager = manager
    
    stats = lb.get_statistics()
    
    # 数据一致性验证：与 /api/nodes 的数据进行对比
    cluster_status = manager.get_cluster_status()
    
    # 只检查在线节点（过滤掉 OFFLINE 节点）
    nodes_from_cluster = set(
        node["node_id"] for node in cluster_status["nodes"]
        if node["status"] == "online"
    )
    
    # 从 LB 统计中提取所有 node_id
    nodes_from_lb = set()
    for model_id, model_info in stats["models"].items():
        for inst in model_info.get("instances", []):
            nodes_from_lb.add(inst["node_id"])
    
    # 检测不一致
    consistency_check = {
        "is_consistent": nodes_from_cluster == nodes_from_lb,
        "nodes_in_cluster_only": list(nodes_from_cluster - nodes_from_lb),
        "nodes_in_lb_only": list(nodes_from_lb - nodes_from_cluster),
        "total_nodes_cluster": len(nodes_from_cluster),
        "total_nodes_lb": len(nodes_from_lb)
    }
    
    # 节流控制：避免频繁告警（离线节点导致的不一致不重复报警）
    global _lb_last_warning_time
    should_warn = False
    if not consistency_check["is_consistent"]:
        current_time = time.time()
        if current_time - _lb_last_warning_time > _lb_warning_throttle_interval:
            should_warn = True
            _lb_last_warning_time = current_time
    
    if should_warn:
        logger.warning(f"⚠️ [LB Status] 数据不一致检测: "
                      f"集群节点={nodes_from_cluster}, "
                      f"LB节点={nodes_from_lb}, "
                      f"差异={consistency_check}")
    
    return {
        "success": True,
        "data": {
            "load_balancer_status": "active",
            "total_models": stats["total_models"],
            "total_instances": stats["total_instances"],
            "total_selections": stats["total_selections"],
            "strategy_distribution": stats["strategy_distribution"],
            "models_detail": stats["models"],
            "_consistency_check": consistency_check  # 调试用，生产环境可移除
        },
        "timestamp": time.time()
    }


@app.get("/api/lb/instances/{model_id}", response_model=Dict)
async def get_model_instances(model_id: str):
    """
    获取指定模型的所有实例信息
    
    Args:
        model_id: 模型ID（可以是完整ID或基础ID）
        
    Returns:
        该模型所有实例的详细信息，包括健康状态、统计数据等
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    lb = get_load_balancer()
    if lb.manager is None:
        lb.manager = manager
    
    # 强制刷新实例列表，确保获取最新数据
    instances = lb.get_available_instances(model_id, force_refresh=True)
    
    instance_details = []
    
    for inst in instances:
        details = {
            "instance_id": inst.instance_id,
            "full_model_id": inst.full_model_id,
            "node_id": inst.node_id,
            "layers": f"{inst.start_layer}-{inst.end_layer}",
            "is_first_layer": inst.is_first_layer,
            "health_score": round(inst.health_score, 1),
            "statistics": {
                "total_requests": inst.total_requests,
                "success_rate": round(inst.success_rate, 2),
                "avg_latency_ms": round(inst.avg_latency, 2),
                "current_connections": inst.current_connections
            },
            "weight": inst.weight,
            "status": "healthy" if inst.health_score >= 70 else 
                     ("degraded" if inst.health_score >= 40 else "unhealthy")
        }
        instance_details.append(details)
    
    base_id = model_id.split("::")[0] if "::" in model_id else model_id
    
    return {
        "success": True,
        "data": {
            "base_model_id": base_id,
            "instances_count": len(instance_details),
            "instances": instance_details
        },
        "timestamp": time.time()
    }


@app.post("/api/lb/test", response_model=Dict)
async def test_load_balancing(request: Dict[str, Any]):
    """
    测试负载均衡功能
    
    模拟多次请求，验证负载均衡策略是否正确分配请求到不同实例
    
    Request Body:
    {
        "model_id": "qwen3-0.6b",
        "strategy": "round_robin",     // 可选：测试的策略
        "test_requests": 10           // 可选：测试请求数量（默认10）
    }
    
    Returns:
        测试结果，包括每次选择的实例、分布统计等
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    model_id = request.get("model_id", "")
    strategy_str = request.get("strategy", "round_robin")
    test_requests = request.get("test_requests", 10)
    
    if not model_id:
        raise HTTPException(status_code=400, detail="必须提供 model_id")
    
    try:
        strategy = LBStrategy(strategy_str.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"无效的策略: {strategy_str}。可用: {[s.value for s in LBStrategy]}"
        )
    
    lb = get_load_balancer()
    if lb.manager is None:
        lb.manager = manager
    
    test_results = []
    instance_distribution: Dict[str, int] = {}
    
    for i in range(test_requests):
        result = lb.select_instance(model_id=model_id, strategy=strategy)
        
        if result:
            instance_id = result.selected_instance.instance_id
            node_id = result.selected_node_id
            
            instance_distribution[instance_id] = instance_distribution.get(instance_id, 0) + 1
            
            test_results.append({
                "request_index": i + 1,
                "selected_instance": instance_id,
                "selected_node": node_id,
                "strategy_used": result.strategy_used.value,
                "reason": result.selection_reason
            })
            
            # 模拟完成（减少连接计数）
            lb.record_completion(node_id=node_id, model_id=model_id, success=True)
    
    return {
        "success": True,
        "data": {
            "model_id": model_id,
            "strategy_tested": strategy.value,
            "total_test_requests": test_requests,
            "successful_selections": len(test_results),
            "distribution": instance_distribution,
            "balance_score": _calculate_balance_score(instance_distribution) if len(instance_distribution) > 1 else 100.0,
            "details": test_results[:20]  # 只返回前20条详细信息
        },
        "timestamp": time.time()
    }


def _calculate_balance_score(distribution: Dict[str, int]) -> float:
    """计算负载均衡分数 (0-100, 越高越均匀)"""
    if not distribution or len(distribution) <= 1:
        return 100.0
    
    total = sum(distribution.values())
    expected = total / len(distribution)
    
    variance = sum((count - expected) ** 2 for count in distribution.values()) / len(distribution)
    
    if expected == 0:
        return 100.0
    
    coefficient_of_variation = (variance ** 0.5) / expected
    
    score = max(0, 100 * (1 - coefficient_of_variation))
    return round(score, 2)


@app.post("/api/lb/reset-stats", response_model=Dict)
async def reset_lb_stats(model_id: Optional[str] = None):
    """
    重置负载均衡统计信息
    
    Args:
        model_id: 可选，指定要重置的模型ID。为None则重置全部
    """
    lb = get_load_balancer()
    lb.reset_statistics(model_id)
    
    return {
        "success": True,
        "message": f"统计信息已重置: {'全部' if not model_id else model_id}",
        "timestamp": time.time()
    }


def _select_node_by_model(model_id: str) -> Optional[str]:
    """
    根据模型ID选择节点（回退逻辑）
    
    Args:
        model_id: 模型ID
        
    Returns:
        节点ID 或 None
    """
    if not manager or not hasattr(manager, 'connectors'):
        return None
    
    first_layer_node = None
    candidate_nodes = []
    
    for node_id, connector in manager.connectors.items():
        if connector.node_info.status.value != "online":
            continue
        
        for m in connector.node_info.loaded_models:
            if m.get("model_id") == model_id:
                shard = m.get("shard", {})
                start_layer = shard.get("start_layer", -1)
                
                candidate_nodes.append((node_id, start_layer))
                
                if start_layer == 0:
                    first_layer_node = node_id
                break
    
    if first_layer_node:
        return first_layer_node
    elif candidate_nodes:
        candidate_nodes.sort(key=lambda x: x[1])
        return candidate_nodes[0][0]
    
    return None


def _get_available_models_for_inference() -> List[Dict]:
    """
    获取所有可用于推理的已加载模型（多层 Fallback 策略）
    
    数据源优先级：
    1. manager.connectors (实时连接数据)
    2. manager.nodes (缓存数据)
    3. get_cluster_status() (最新集群快照)
    
    Type 字段说明：
    - worker_instance: 带 ::instance_id 的多实例模型
    - single: 不带 :: 的单实例模型（也可用于推理）
    """
    models = []
    seen_models = set()
    
    # ===== 策略1: 从 connectors 获取 (实时数据) =====
    connectors_data = getattr(manager, 'connectors', {})
    if connectors_data:
        for node_id, connector in connectors_data.items():
            if connector.node_info.status.value != "online":
                continue
            
            for model in connector.node_info.loaded_models:
                model_id = model.get("model_id", "unknown")
                if model_id not in seen_models:
                    seen_models.add(model_id)
                    models.append({
                        "model_id": model_id,
                        "node_id": node_id,
                        "shard": model.get("shard", {}),
                        "loaded_at": model.get("loaded_at"),
                        # ✅ 修复：单实例模型标记为 "single" 而非 "base"
                        "type": "worker_instance" if "::" in model_id else "single",
                    })
        
        if models:
            logger.info(f"[Inference Models] 从 connectors 获取到 {len(models)} 个模型")
            return models
    
    # ===== 策略2: Fallback 到 manager.nodes (缓存数据) =====
    nodes_data = getattr(manager, 'nodes', {})
    if nodes_data:
        for node_id, node_info in nodes_data.items():
            if hasattr(node_info, 'status') and node_info.status.value != "online":
                continue
            
            for model in getattr(node_info, 'loaded_models', []):
                model_id = model.get("model_id", "unknown")
                if model_id not in seen_models:
                    seen_models.add(model_id)
                    models.append({
                        "model_id": model_id,
                        "node_id": node_id,
                        "shard": model.get("shard", {}),
                        "loaded_at": model.get("loaded_at"),
                        # ✅ 修复：单实例模型标记为 "single" 而非 "base"
                        "type": "worker_instance" if "::" in model_id else "single",
                    })
        
        if models:
            logger.info(f"[Inference Models] Fallback 到 nodes，获取到 {len(models)} 个模型")
            return models
    
    # ===== 策略3: Fallback 到 get_cluster_status() (最新快照) =====
    try:
        cluster_status = manager.get_cluster_status()
        for node in cluster_status.get("nodes", []):
            if node.get("status") != "online":
                continue
            
            node_id = node.get("node_id", "unknown")
            for model in node.get("loaded_models", []):
                model_id = model.get("model_id", "unknown")
                if model_id not in seen_models:
                    seen_models.add(model_id)
                    models.append({
                        "model_id": model_id,
                        "node_id": node_id,
                        "shard": model.get("shard", {}),
                        "loaded_at": model.get("loaded_at"),
                        # ✅ 修复：单实例模型标记为 "single" 而非 "base"
                        "type": "worker_instance" if "::" in model_id else "single",
                    })
        
        if models:
            logger.info(f"[Inference Models] Fallback 到 cluster_status，获取到 {len(models)} 个模型")
    
    except Exception as e:
        logger.error(f"[Inference Models] get_cluster_status() 失败: {e}")
    
    return models


@app.get("/api/pool/preview", response_model=Dict)
async def pool_preview_allocation(
    model_id: str,
    n_layers: int = 32,
    strategy: str = "smart"  # 默认智能策略
):
    """
    预览模型分配方案（不实际加载）
    
    Query Params:
        model_id: 模型ID
        n_layers: 总层数（默认32）
        strategy: 分配策略
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    try:
        from gpu_pool_integration import GPUPoolIntegration
        pool = GPUPoolIntegration(manager)
        
        allocation = await pool.preview_allocation(
            model_id=model_id,
            total_layers=n_layers,
            strategy=strategy
        )
        
        return {
            "success": True,
            "data": {
                "model_id": allocation.model_id,
                "total_layers": allocation.total_layers,
                "strategy": allocation.strategy,
                "allocation_type": allocation.allocation_type,
                "decision_reason": allocation.decision_reason,
                "safety_warnings": allocation.safety_warnings or [],
                "allocations": allocation.allocations,
                "estimated_memory_per_node": allocation.estimated_memory_per_node
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ==================== 故障恢复 API ====================

@app.post("/api/fault-recovery/{node_id}", response_model=Dict)
async def trigger_fault_recovery(node_id: str):
    """
    手动触发指定节点的故障恢复

    当节点掉线后，模拟故障恢复流程：
    - 识别受影响的模型
    - 尝试迁移到存活节点
    - 返回恢复方案报告

    Path Params:
        node_id: 掉线的节点ID
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        from gpu_pool_integration import SmartAllocator
        allocator = SmartAllocator(manager)
        report = allocator.handle_node_failure(node_id)

        return {
            "success": True,
            "data": report,
            "message": f"节点 {node_id} 故障恢复分析完成"
        }

    except Exception as e:
        logger.error(f"故障恢复处理失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/api/fault-recovery/history", response_model=Dict)
async def get_fault_recovery_history(limit: int = 10):
    """
    获取历史故障恢复记录

    Query Params:
        limit: 返回记录数量上限（默认10条）
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")

    history = getattr(manager, '_fault_recovery_history', [])
    return {
        "success": True,
        "data": history[-limit:] if history else [],
        "total": len(history)
    }



# ==================== 系统日志 API ====================

@app.get("/api/logs", response_model=Dict)
async def get_system_logs(
    level: str = None,
    category: str = None,
    limit: int = 100,
    offset: int = 0
):
    """
    获取系统日志

    Query Params:
        level: 过滤级别 (info/warning/error/success)
        category: 过滤分类 (allocation/fault-recovery/node/model/...)
        limit: 返回条数上限（默认100）
        offset: 分页偏移量
    """
    result = sys_log.get_logs(level=level, category=category, limit=limit, offset=offset)
    return {"success": True, "data": result}


@app.post("/api/logs/clear", response_model=Dict)
async def clear_system_logs():
    """清空系统日志"""
    sys_log.clear()
    return {"success": True, "message": "日志已清空"}


# ==================== P2P 拓扑管理 API ====================

@app.get("/api/topology", response_model=Dict)
async def get_topology(force_refresh: bool = False):
    """
    获取P2P网络拓扑
    
    展示节点间的连接关系和推理链路
    
    Query Params:
        force_refresh: 是否强制刷新拓扑缓存
    """
    if not topo_manager:
        raise HTTPException(status_code=503, detail="拓扑管理器未初始化")
    
    topology = await topo_manager.collect_full_topology(force=force_refresh)
    
    return {
        "success": True,
        "data": topology
    }


@app.get("/api/topology/visualization", response_model=Dict)
async def get_topology_visualization(layout: str = "force"):
    """
    获取用于前端可视化的拓扑数据
    
    返回格式化的节点和边数据，可直接用于D3.js/vis.js等库
    
    Query Params:
        layout: 布局算法 ('force', 'circular', 'hierarchical')
    """
    if not topo_manager:
        raise HTTPException(status_code=503, detail="拓扑管理器未初始化")
    
    viz_data = topo_manager.get_visualization_data(layout=layout)
    
    return {
        "success": True,
        "data": viz_data
    }


@app.get("/api/topology/pipeline/{model_id}", response_model=Dict)
async def get_inference_pipeline(model_id: str):
    """
    获取指定模型的推理流水线
    
    显示模型推理时请求在节点间的流转路径
    
    Path Params:
        model_id: 模型标识符
    """
    if not topo_manager:
        raise HTTPException(status_code=503, detail="拓扑管理器未初始化")
    
    pipeline = topo_manager.get_inference_pipeline(model_id)
    
    if not pipeline:
        raise HTTPException(
            status_code=404, 
            detail=f"模型 {model_id} 的推理流水线不存在（可能未加载或为单节点）"
        )
    
    return {
        "success": True,
        "data": pipeline
    }


@app.get("/api/topology/node/{node_id}/neighbors", response_model=Dict)
async def get_node_neighbors(node_id: str):
    """
    获取节点的P2P邻居列表
    
    显示与指定节点直接相连的所有节点
    
    Path Params:
        node_id: 节点ID
    """
    if not topo_manager:
        raise HTTPException(status_code=503, detail="拓扑管理器未初始化")
    
    neighbors = topo_manager.get_node_neighbors(node_id)
    
    if neighbors is None and node_id not in topo_manager.nodes:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 不存在")
    
    return {
        "success": True,
        "data": {
            "node_id": node_id,
            "neighbor_count": len(neighbors),
            "neighbors": neighbors
        }
    }


@app.get("/api/topology/path", response_model=Dict)
async def find_path(from_node: str, to_node: str):
    """
    查找两个节点间的最短路径
    
    用于分析请求路由和网络可达性
    
    Query Params:
        from_node: 起始节点ID
        to_node: 目标节点ID
    """
    if not topo_manager:
        raise HTTPException(status_code=503, detail="拓扑管理器未初始化")
    
    path = topo_manager.find_shortest_path(from_node, to_node)
    
    if path is None:
        return {
            "success": True,
            "data": {
                "reachable": False,
                "path": None,
                "message": f"从 {from_node} 到 {to_node} 不可达"
            }
        }
    
    return {
        "success": True,
        "data": {
            "reachable": True,
            "path": path,
            "hop_count": len(path) - 1
        }
    }


@app.get("/api/topology/anomalies", response_model=Dict)
async def detect_anomalies():
    """
    检测网络异常
    
    包括：孤立节点、断开的连接、网络分区等
    """
    if not topo_manager:
        raise HTTPException(status_code=503, detail="拓扑管理器未初始化")
    
    anomalies = topo_manager.detect_anomalies()
    
    severity_counts = {"critical": 0, "warning": 0, "error": 0}
    for a in anomalies:
        severity_counts[a["severity"]] = severity_counts.get(a["severity"], 0) + 1
    
    return {
        "success": True,
        "data": {
            "anomalies": anomalies,
            "total_count": len(anomalies),
            "severity_breakdown": severity_counts,
            "is_healthy": len([a for a in anomalies if a["severity"] == "critical"]) == 0
        }
    }

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"📡 新的WebSocket连接 (总计: {len(self.active_connections)})")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"🔌 WebSocket断开 (剩余: {len(self.active_connections)})")
        else:
            logger.warning(f"⚠️ WebSocket断开但不在连接列表中")
    
    async def broadcast(self, message: dict):
        """广播消息给所有连接的客户端（增强容错）"""
        data = json.dumps(message, ensure_ascii=False)
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception as e:
                # 静默记录断开的连接，不打印错误（避免日志刷屏）
                disconnected.append(connection)
        
        # 清理断开的连接
        for conn in disconnected:
            try:
                self.active_connections.remove(conn)
            except ValueError:
                pass  # 已经被移除，忽略


ws_manager = ConnectionManager()


class NodeWSManager:
    """
    Node WebSocket 连接管理器
    
    管理 Node 通过 WebSocket 建立的长连接，用于：
    - 推送推理请求（替代 HTTP 代理）
    - 接收推理结果（流式响应）
    - 双向心跳保活
    
    解决内网/FRP 场景下 Manager 无法主动连接 Node 的问题
    """
    
    def __init__(self):
        self.node_connections: Dict[str, WebSocket] = {}  # node_id -> websocket
        self.pending_requests: Dict[str, asyncio.Queue] = {}  # request_id -> response queue
        self._lock = asyncio.Lock()

    async def safe_send(self, node_id: str, data: Any) -> bool:
        """
        安全发送消息到节点（自动处理连接已关闭的情况）

        Args:
            node_id: 目标节点 ID
            data: 要发送的数据（会被 json 序列化）

        Returns:
            bool: 是否发送成功
        """
        if node_id not in self.node_connections:
            logger.warning(f"[NodeWS] safe_send 失败: 节点 {node_id} 未连接")
            return False

        ws = self.node_connections[node_id]
        try:
            if isinstance(data, (dict, list)):
                msg = json.dumps(data, ensure_ascii=False)
            else:
                msg = str(data)
            await ws.send_text(msg)
            return True
        except Exception as e:
            logger.warning(f"[NodeWS] safe_send 失败: 节点 {node_id} 发送错误: {e}")
            # 连接已失效，触发清理
            await self.disconnect_node(node_id)
            return False
    
    async def connect_node(self, node_id: str, websocket: WebSocket) -> bool:
        """Node 建立连接"""
        async with self._lock:
            if node_id in self.node_connections:
                logger.warning(f"⚠️ [NodeWS] 节点 {node_id} 已有连接，将替换")
                try:
                    await self.node_connections[node_id].close()
                except:
                    pass

            self.node_connections[node_id] = websocket
            logger.info(f"✅ [NodeWS] 节点 {node_id} 已连接 (总计: {len(self.node_connections)})")

            # 恢复节点在线状态
            if manager and node_id in manager.nodes:
                node_info = manager.nodes[node_id]
                if node_info.status != NodeStatus.ONLINE:
                    old_status = node_info.status.value
                    node_info.status = NodeStatus.ONLINE
                    node_info.error_message = ""
                    node_info.last_heartbeat = time.time()
                    logger.info(f"🔄 [NodeWS] 节点 {node_id} 状态恢复: {old_status} → online")

                    # 通知前端状态变化
                    try:
                        await ws_manager.broadcast({
                            "type": "node_status_changed",
                            "node_id": node_id,
                            "status": "online",
                            "timestamp": time.time()
                        })
                    except Exception as e:
                        logger.debug(f"广播状态变化失败: {e}")

            # 📢 广播新节点信息给其他已连接节点（用于 P2P 发现）
            await self.broadcast_new_node_to_peers(node_id)

            return True

    async def broadcast_new_node_to_peers(self, node_id: str):
        """将新上线节点信息广播给其他已连接节点（用于 P2P 发现）"""
        if len(self.node_connections) <= 1:
            return  # 只有自己，无需广播

        try:
            from frp_server_manager import get_frp_server_manager
            frp_mgr = get_frp_server_manager()

            # 构建新节点信息
            new_node_info = None
            if manager and node_id in manager.nodes:
                ni = manager.nodes[node_id]
                client_cfg = frp_mgr.get_client_config(node_id)
                frp_remote_port = 0
                if client_cfg:
                    meta = client_cfg.get("_meta", {})
                    frp_remote_port = meta.get("remote_port", 0)

                peer_port = frp_remote_port if frp_remote_port > 0 else ni.port

                new_node_info = {
                    "node_id": node_id,
                    "address": ni.ip,
                    "port": peer_port,
                    "device_capabilities": {
                        "model": ni.device_name or "unknown",
                        "chip": ni.device_name or "unknown",
                        "memory": ni.gpu_total_memory_mb / 1024 if ni.gpu_total_memory_mb else 0,
                    },
                }

            if not new_node_info:
                return

            # 广播给所有其他已连接节点
            msg = {
                "type": "peer_list",
                "peers": [new_node_info],
                "timestamp": time.time(),
            }
            msg_text = json.dumps(msg)

            sent_count = 0
            for other_id, other_ws in self.node_connections.items():
                if other_id == node_id:
                    continue
                try:
                    await other_ws.send_text(msg_text)
                    sent_count += 1
                except Exception as e:
                    logger.debug(f"[NodeWS] 向 {other_id} 广播新节点失败: {e}")

            if sent_count > 0:
                logger.info(f"📢 [NodeWS] 已向 {sent_count} 个节点广播新节点: {node_id}")

        except Exception as e:
            logger.debug(f"[NodeWS] ⚠️ 广播新节点失败（非关键）: {e}")
    
    async def disconnect_node(self, node_id: str):
        """Node 断开连接（幂等操作，支持重复调用）"""
        # 幂等：已断开的节点直接跳过
        if node_id not in self.node_connections:
            logger.debug(f"[NodeWS] 节点 {node_id} 已断开，跳过重复清理")
            return

        del self.node_connections[node_id]
        logger.info(f"🔌 [NodeWS] 节点 {node_id} 已断开 (剩余: {len(self.node_connections)})")

        # 标记节点为离线
        if manager and node_id in manager.nodes:
            node_info = manager.nodes[node_id]
            old_status = node_info.status.value
            node_info.status = NodeStatus.OFFLINE
            node_info.error_message = "WebSocket连接已断开"

            logger.warning(f"⚠️ [NodeWS] 节点 {node_id} 已标记为离线 (状态: {old_status} → offline)")

            # 通知前端节点状态变化（非关键操作，失败不阻塞）
            try:
                await ws_manager.broadcast({
                    "type": "node_status_changed",
                    "node_id": node_id,
                    "status": "offline",
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.debug(f"广播状态变化失败: {e}")
    
    def is_node_connected(self, node_id: str) -> bool:
        """检查节点是否已通过 WebSocket 连接（增强版：验证连接状态）"""
        if node_id not in self.node_connections:
            return False
        
        websocket = self.node_connections[node_id]
        
        try:
            # 检查 WebSocket 是否仍然打开
            # FastAPI WebSocket 对象有 client 属性表示连接状态
            # 如果连接已关闭，从列表中移除并返回 False
            if hasattr(websocket, 'client') and websocket.client is None:
                logger.warning(f"⚠️ [NodeWS] 节点 {node_id} 连接已失效，移除")
                del self.node_connections[node_id]
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"[NodeWS] 检查节点 {node_id} 连接状态失败: {e}")
            return False
    
    async def send_inference_request(
        self,
        node_id: str,
        request_data: Dict,
        timeout: float = 300.0
    ) -> AsyncGenerator[bytes, None]:
        """
        发送推理请求到指定 Node（流式）
        
        Args:
            node_id: 目标节点 ID
            request_data: 推理请求数据 (包含 model_id, messages 等)
            timeout: 超时时间
            
        Yields:
            流式响应数据块 (SSE 格式)
        """
        if node_id not in self.node_connections:
            raise Exception(f"Node {node_id} not connected via WebSocket")

        import uuid
        request_id = f"ws_{uuid.uuid4().hex[:12]}"
        websocket = self.node_connections[node_id]

        # 检查连接是否仍然有效
        try:
            from starlette.websockets import WebSocketState
            if websocket.client_state == WebSocketState.DISCONNECTED:
                self.node_connections.pop(node_id, None)
                raise Exception(f"Node {node_id} WebSocket 已断开")
        except Exception as e:
            if "WebSocket 已断开" in str(e):
                raise
            pass

        # 创建响应队列
        response_queue = asyncio.Queue()
        self.pending_requests[request_id] = response_queue
        
        try:
            # 构建推理请求消息
            inference_msg = {
                "type": "inference_request",
                "request_id": request_id,
                **request_data
            }
            
            # 发送请求到 Node
            await websocket.send_text(json.dumps(inference_msg, ensure_ascii=False))
            logger.info(f"[NodeWS] → 推理请求发送到 {node_id}: {request_id}")
            
            # 流式接收响应
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        response_queue.get(),
                        timeout=timeout
                    )
                    
                    if chunk is None:
                        break
                    
                    yield chunk.encode('utf-8')
                    
                except asyncio.TimeoutError:
                    error_data = {
                        "error": {
                            "message": f"Inference timeout on node {node_id}",
                            "type": "timeout_error"
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\ndata: [DONE]\n\n".encode('utf-8')
                    break
                    
        finally:
            # 清理资源
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
    
    async def handle_node_message(self, node_id: str, message: str):
        """处理 Node 发来的消息"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "register":
                # Node 注册认证
                received_node_id = data.get("node_id")
                if received_node_id != node_id:
                    logger.error(f"[NodeWS] ❌ Node ID 不匹配: 期望 {node_id}, 收到 {received_node_id}")
                    return

                ack_msg = {
                    "type": "register_ack",
                    "status": "success",
                    "message": f"Node {node_id} registered successfully"
                }
                ws = self.node_connections.get(node_id)
                if ws:
                    await ws.send_text(json.dumps(ack_msg))
                logger.info(f"[NodeWS] ✅ 节点 {node_id} 注册成功")

                # 🔑 关键修复：同步注册到集群管理器，使 /api/nodes、Cluster Status、LoadBalancer 可见
                if manager:
                    try:
                        if node_id not in manager.nodes:
                            # 从 WS 连接获取客户端地址
                            client_host = getattr(ws, 'client', None)
                            client_addr = getattr(client_host, 'host', '127.0.0.1') if client_host else '127.0.0.1'

                            await manager.add_node(
                                node_id=node_id,
                                address=client_addr,
                                port=0,
                                chatgpt_api_port=data.get("chatgpt_api_port", 52415),
                                device_info=data.get("device_info", {}),
                                skip_grpc_connect=True
                            )
                            logger.info(f"[NodeWS] 📋 节点 {node_id} 已同步到集群管理器 (地址: {client_addr})")

                        # 无论新增还是已存在，WS 注册成功即标记为在线
                        if node_id in manager.nodes:
                            manager.nodes[node_id].status = NodeStatus.ONLINE
                            manager.nodes[node_id].error_message = ""
                            manager.nodes[node_id].last_heartbeat = time.time()
                    except Exception as e:
                        logger.warning(f"[NodeWS] ⚠️ 同步节点到集群管理器失败: {e}")

            elif msg_type == "heartbeat":
                # Node 心跳保活 — 更新 manager.nodes 中的心跳时间戳
                if manager and node_id in manager.nodes:
                    manager.nodes[node_id].last_heartbeat = time.time()
                    # 确保状态为在线
                    if manager.nodes[node_id].status != NodeStatus.ONLINE:
                        manager.nodes[node_id].status = NodeStatus.ONLINE
                        manager.nodes[node_id].error_message = ""

            elif msg_type == "inference_chunk":
                # 收到推理结果片段
                request_id = data.get("request_id")
                chunk_data = data.get("data", "")
                
                if request_id in self.pending_requests:
                    await self.pending_requests[request_id].put(chunk_data)
                    
            elif msg_type == "inference_complete":
                # 推理完成
                request_id = data.get("request_id")
                tokens_used = data.get("tokens_used", 0)
                
                if request_id in self.pending_requests:
                    # 发送结束标记
                    await self.pending_requests[request_id].put(None)
                    logger.info(f"[NodeWS] ← 推理完成 {request_id}, tokens={tokens_used}")
                    
            elif msg_type == "inference_error":
                # 推理错误
                request_id = data.get("request_id")
                error_msg = data.get("error", "Unknown error")
                
                if request_id in self.pending_requests:
                    error_data = {
                        "error": {
                            "message": error_msg,
                            "type": "upstream_error"
                        }
                    }
                    error_chunk = f"data: {json.dumps(error_data)}\n\ndata: [DONE]\n\n"
                    await self.pending_requests[request_id].put(error_chunk)
                    logger.error(f"[NodeWS] ❌ 推理错误 {request_id}: {error_msg}")
                    
            elif msg_type == "model_load_complete":
                # ✨ Node 返回模型加载完成
                task_id = data.get("task_id", "")
                node_id_resp = data.get("node_id", node_id)
                success = data.get("success", False)
                loaded_models = data.get("loaded_models", [])
                error_msg = data.get("error", "")
                
                logger.info(f"[NodeWS] 📦 模型加载完成: task={task_id}, node={node_id_resp}, success={success}")
                
                # 通知 cluster_core（如果可用）
                if manager and hasattr(manager, '_on_model_load_completed'):
                    try:
                        await manager._on_model_load_completed(
                            node_id=node_id_resp,
                            task_id=task_id,
                            success=success,
                            loaded_models=loaded_models,
                            error=error_msg
                        )
                    except Exception as e:
                        logger.warning(f"[NodeWS] ⚠️ 回调失败: {e}")
                        
            elif msg_type == "model_unload_complete":
                # ✨ Node 返回模型卸载完成
                model_id = data.get("model_id", "")
                node_id_resp = data.get("node_id", node_id)
                success = data.get("success", False)
                
                logger.info(f"[NodeWS] 🗑️ 模型卸载完成: model={model_id}, node={node_id_resp}, success={success}")
                
                # 通知 cluster_core（如果可用）
                if manager and hasattr(manager, '_on_model_unload_completed'):
                    try:
                        await manager._on_model_unload_completed(
                            node_id=node_id_resp,
                            model_id=model_id,
                            success=success
                        )
                    except Exception as e:
                        logger.warning(f"[NodeWS] ⚠️ 回调失败: {e}")
                        
            elif msg_type == "model_status_update":
                # ✨ Node 上报模型状态变化（实时更新）
                loaded_models = data.get("loaded_models", [])
                gpu_memory = data.get("gpu_memory", {})
                
                # 更新节点信息
                if manager and node_id in manager.nodes:
                    manager.nodes[node_id].loaded_models = loaded_models
                    if gpu_memory:
                        manager.nodes[node_id].device_info["gpu_memory"] = gpu_memory
                    
                    # 广播状态更新
                    if manager._broadcast_callback:
                        try:
                            await manager._broadcast_callback({
                                "type": "node_model_update",
                                "node_id": node_id,
                                "loaded_models": loaded_models,
                                "gpu_memory": gpu_memory
                            })
                        except Exception as e:
                            logger.debug(f"[NodeWS] 广播失败: {e}")
                    
            else:
                logger.debug(f"[NodeWS] 收到未知消息类型: {msg_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"[NodeWS] JSON 解析错误: {e}")
        except Exception as e:
            logger.error(f"[NodeWS] 处理消息错误: {e}", exc_info=True)

    # ========== ✨ 增强功能 ==========

    async def send_inference_request_v2(
        self,
        node_id: str,
        request_data: Dict,
        timeout: float = 300.0
    ) -> AsyncGenerator[bytes, None]:
        """
        ✨ V2 版本：发送推理请求（增强版）
        
        改进：
        - 消息确认机制
        - 超时自动清理
        - 错误恢复
        - 进度追踪
        - 性能监控
        
        Args:
            node_id: 目标节点 ID
            request_data: 推理请求数据
            timeout: 超时时间
            
        Yields:
            流式响应数据块 (SSE 格式)
        """
        if node_id not in self.node_connections:
            raise Exception(f"Node {node_id} not connected via WebSocket")

        import uuid
        request_id = f"ws_v2_{uuid.uuid4().hex[:12]}"
        websocket = self.node_connections[node_id]

        # 检查连接是否仍然有效
        try:
            from starlette.websockets import WebSocketState
            if websocket.client_state == WebSocketState.DISCONNECTED:
                self.node_connections.pop(node_id, None)
                raise Exception(f"Node {node_id} WebSocket 已断开")
        except Exception as e:
            if "WebSocket 已断开" in str(e):
                raise
            pass

        # 创建响应队列（带大小限制，防止内存泄漏）
        response_queue = asyncio.Queue(maxsize=100)
        self.pending_requests[request_id] = response_queue
        
        # 统计信息
        start_time = time.time()
        chunk_count = 0
        total_bytes = 0
        
        try:
            # 构建增强版推理请求消息
            inference_msg = {
                "type": "inference_request",
                "request_id": request_id,
                "version": "2.0",  # 标记为 V2 协议
                **request_data
            }
            
            # 发送请求到 Node
            await websocket.send_text(json.dumps(inference_msg, ensure_ascii=False))
            logger.info(f"[NodeWS-V2] → 推理请求发送到 {node_id}: {request_id}")
            
            # 流式接收响应（带超时和背压控制）
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        response_queue.get(),
                        timeout=timeout
                    )
                    
                    if chunk is None:
                        # 发送结束标记
                        end_marker = f"data: [DONE]\n\n".encode('utf-8')
                        yield end_marker
                        break
                    
                    chunk_count += 1
                    total_bytes += len(chunk)
                    
                    yield chunk
                    
                except asyncio.TimeoutError:
                    error_data = {
                        "error": {
                            "message": f"Inference timeout on node {node_id} (>{timeout}s)",
                            "type": "timeout_error",
                            "node_id": node_id,
                            "request_id": request_id
                        }
                    }
                    error_response = f"data: {json.dumps(error_data)}\n\ndata: [DONE]\n\n".encode('utf-8')
                    yield error_response
                    
                    logger.error(f"[NodeWS-V2] ❌ 推理超时: {request_id}, node={node_id}")
                    break
                    
        finally:
            # 清理资源（防止内存泄漏）
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
                
            elapsed_time = time.time() - start_time
            
            # 记录统计信息
            logger.info(
                f"[NodeWS-V2] 📊 推理完成统计: "
                f"request={request_id}, "
                f"chunks={chunk_count}, "
                f"bytes={total_bytes}, "
                f"elapsed={elapsed_time:.2f}s"
            )

    async def broadcast_to_nodes(self, message: Dict, exclude_nodes: List[str] = None) -> int:
        """
        广播消息到所有连接的 Node
        
        Args:
            message: 要广播的消息
            exclude_nodes: 要排除的节点 ID 列表
            
        Returns:
            int: 成功发送的数量
        """
        exclude_set = set(exclude_nodes or [])
        sent_count = 0
        failed_nodes = []
        
        for node_id, ws in self.node_connections.items():
            if node_id in exclude_set:
                continue
                
            try:
                await ws.send_text(json.dumps(message, ensure_ascii=False))
                sent_count += 1
                logger.debug(f"[NodeWS] 📢 广播到 {node_id}: 成功")
            except Exception as e:
                failed_nodes.append(node_id)
                logger.warning(f"[NodeWS] ⚠️ 广播到 {node_id} 失败: {e}")
        
        if failed_nodes:
            logger.warning(f"[NodeWS] ⚠️ 广播失败节点: {failed_nodes}")
        
        logger.info(f"[NodeWS] 📢 广播完成: 成功={sent_count}, 总计={len(self.node_connections)}")
        return sent_count

    async def get_node_stats(self, node_id: str) -> Optional[Dict]:
        """
        获取指定节点的连接统计信息
        
        Args:
            node_id: 节点 ID
            
        Returns:
            Dict or None: 节点统计信息
        """
        if node_id not in self.node_connections:
            return None
        
        websocket = self.node_connections[node_id]
        
        stats = {
            "node_id": node_id,
            "connected": True,
            "pending_requests": sum(
                1 for req_id in self.pending_requests 
                if req_id.startswith("ws_") or req_id.startswith("ws_v2_")
            ),
            "connection_age": "unknown",
            "last_activity": time.time()
        }
        
        return stats

    def get_all_nodes_status(self) -> Dict[str, Dict]:
        """
        获取所有节点的连接状态（用于监控 API）
        
        Returns:
            Dict: 所有节点的状态信息
        """
        status = {}
        
        for node_id, ws in self.node_connections.items():
            status[node_id] = {
                "connected": True,
                "has_pending_requests": any(
                    req_id for req_id in self.pending_requests.keys()
                )
            }
        
        return status

    async def cleanup_stale_requests(self, max_age: float = 3600.0):
        """
        清理过期的待处理请求（定期调用以防止内存泄漏）
        
        Args:
            max_age: 最大存活时间（秒）
        """
        current_time = time.time()
        stale_requests = []
        
        # 注意：这里需要记录每个请求的创建时间
        # 简化实现：如果队列过大，清理一些旧请求
        if len(self.pending_requests) > 100:
            logger.warning(f"[NodeWS] ⚠️ 待请求数量过多 ({len(self.pending_requests)})，开始清理")
            
            # 删除一半最旧的请求（简化版）
            keys_to_remove = list(self.pending_requests.keys())[:len(self.pending_requests)//2]
            for key in keys_to_remove:
                queue = self.pending_requests.pop(key, None)
                if queue:
                    # 放入结束标记，避免阻塞
                    try:
                        queue.put_nowait(None)
                    except:
                        pass
                
            logger.info(f"[NodeWS] 🧹 已清理 {len(keys_to_remove)} 个过期请求")

    async def health_check_all_nodes(self) -> Dict[str, bool]:
        """
        对所有连接的 Node 进行健康检查
        
        Returns:
            Dict[bool]: 节点ID -> 是否健康
        """
        results = {}
        
        for node_id, ws in list(self.node_connections.items()):
            try:
                # 发送心跳消息
                heartbeat_msg = {
                    "type": "health_check",
                    "timestamp": time.time()
                }
                await ws.send_text(json.dumps(heartbeat_msg))
                results[node_id] = True
            except Exception as e:
                logger.warning(f"[NodeWS] ⚠️ 健康检查失败 {node_id}: {e}")
                results[node_id] = False
        
        return results

    async def send_model_load_request(
        self,
        node_id: str,
        task_id: str,
        model_id: str,
        model_path: str,
        shard: Dict,
        peer_list: List = None,
        instance_id: str = None  # ✅ 新增实例ID参数
    ) -> bool:
        """
        通过 WebSocket 发送模型加载请求（Push 模式）
        
        Args:
            node_id: 目标节点
            task_id: 任务ID
            model_id: 模型ID
            model_path: 模型路径
            shard: 分片配置 {"start_layer", "end_layer", "n_layers"}
            peer_list: Peer 列表
            instance_id: 实例ID (支持多实例)
            
        Returns:
            是否发送成功
        """
        if node_id not in self.node_connections:
            return False

        websocket = self.node_connections[node_id]

        # 检查连接是否仍然有效
        try:
            from starlette.websockets import WebSocketState
            if websocket.client_state == WebSocketState.DISCONNECTED:
                logger.warning(f"⚠️ [NodeWS] 节点 {node_id} 的 WebSocket 已断开，清理并跳过")
                self.node_connections.pop(node_id, None)
                return False
        except Exception:
            pass

        load_msg = {
            "type": "model_load",
            "task_id": task_id,
            "model_id": model_id,
            "model_path": model_path,
            "shard": shard,
            "peer_list": peer_list or [],
            "instance_id": instance_id  # ✅ 新增字段
        }
        
        try:
            logger.info(f"[NodeWS] 🔍 推送诊断: instance_id={instance_id}, model_id={model_id}")
            await websocket.send_text(json.dumps(load_msg, ensure_ascii=False))
            logger.info(f"[NodeWS] 📦 推送模型加载任务 → {node_id}: {model_id} (任务ID: {task_id})")
            return True
        except Exception as e:
            logger.error(f"[NodeWS] ❌ 发送模型加载请求失败 ({node_id}): {e}")
            return False

    async def send_model_unload_request(
        self,
        node_id: str,
        model_id: str
    ) -> bool:
        """
        通过 WebSocket 发送模型卸载请求
        
        Args:
            node_id: 目标节点
            model_id: 要卸载的模型ID
            
        Returns:
            是否发送成功
        """
        if node_id not in self.node_connections:
            return False

        websocket = self.node_connections[node_id]

        # 检查连接是否仍然有效
        try:
            from starlette.websockets import WebSocketState
            if websocket.client_state == WebSocketState.DISCONNECTED:
                logger.warning(f"⚠️ [NodeWS] 节点 {node_id} 的 WebSocket 已断开，清理并跳过")
                self.node_connections.pop(node_id, None)
                return False
        except Exception:
            pass

        unload_msg = {
            "type": "model_unload",
            "model_id": model_id
        }
        
        try:
            await websocket.send_text(json.dumps(unload_msg, ensure_ascii=False))
            logger.info(f"[NodeWS] 🗑️ 推送模型卸载请求 → {node_id}: {model_id}")
            return True
        except Exception as e:
            logger.error(f"[NodeWS] ❌ 发送模型卸载请求失败 ({node_id}): {e}")
            return False

    async def broadcast_to_all_nodes(self, message: Dict) -> int:
        """
        广播消息给所有已连接的节点
        
        Args:
            message: 要广播的消息字典
            
        Returns:
            成功发送的数量
        """
        success_count = 0
        
        for node_id, websocket in list(self.node_connections.items()):
            try:
                await websocket.send_text(json.dumps(message, ensure_ascii=False))
                success_count += 1
            except Exception as e:
                logger.warning(f"[NodeWS] ⚠️ 广播失败 ({node_id}): {e}")
                # 移除断开的连接
                if node_id in self.node_connections:
                    del self.node_connections[node_id]
        
        if success_count > 0:
            logger.info(f"[NodeWS] 📢 广播完成: {success_count}/{len(self.node_connections)} 节点")
        
        return success_count


# 全局 Node WebSocket 管理器实例
node_ws_manager = NodeWSManager()


@app.websocket("/ws/cluster")
async def websocket_cluster(websocket: WebSocket):
    """
    WebSocket端点 - 实时推送集群状态更新
    
    客户端连接后会定期收到：
    - 集群状态更新
    - 节点上下线通知
    - 统计数据变化
    """
    await ws_manager.connect(websocket)
    
    try:
        # 发送初始状态
        if manager:
            initial_status = manager.get_cluster_status()
            await websocket.send_text(json.dumps({
                "type": "initial_status",
                "data": initial_status
            }, ensure_ascii=False))
        
        # 保持连接并发送定期更新
        while True:
            await asyncio.sleep(5)  # 每5秒发送一次更新
            
            if manager:
                status = manager.get_cluster_status()
                
                await websocket.send_text(json.dumps({
                    "type": "status_update",
                    "data": status,
                    "timestamp": time.time()
                }, ensure_ascii=False))
    
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        ws_manager.disconnect(websocket)


@app.websocket("/ws/node/{node_id}")
async def websocket_node_connection(websocket: WebSocket, node_id: str):
    """
    Node 专用 WebSocket 端点
    
    Node 通过此端点建立长连接，实现：
    - 推理请求推送（Manager → Node）
    - 流式结果回传（Node → Manager）
    - 心跳保活（双向）
    
    使用方式：
    - Node 启动后主动连接: ws://manager_host:port/ws/node/{node_id}
    - 发送注册消息进行认证
    - 保持连接接收推理请求
    """
    await websocket.accept()

    # 注册节点连接
    await node_ws_manager.connect_node(node_id, websocket)

    logger.info(f"🔗 [NodeWS] 节点 {node_id} 已连接，等待消息...")

    # 📢 推送已在线节点列表给新连接的节点（用于 FRP P2P 发现）
    try:
        from frp_server_manager import get_frp_server_manager
        frp_mgr = get_frp_server_manager()

        # 收集所有其他已连接节点的信息（排除自己）
        online_peers = []
        for other_id, other_ws in node_ws_manager.node_connections.items():
            if other_id == node_id:
                continue

            # 获取该节点的 FRP 远程端口（用于 P2P 连接）
            frp_remote_port = 0
            client_cfg = frp_mgr.get_client_config(other_id)
            if client_cfg:
                meta = client_cfg.get("_meta", {})
                frp_remote_port = meta.get("remote_port", 0)

            # 获取节点详细信息
            peer_addr = ""
            peer_port = 0
            device_caps = {}

            if manager and other_id in manager.nodes:
                ni = manager.nodes[other_id]
                # 优先使用 FRP 远程端口（FRPDiscovery.add_known_node 会自动转为 P2P 地址）
                if frp_remote_port > 0:
                    peer_addr = ni.ip  # 原始地址，FRPDiscovery 会转换
                    peer_port = frp_remote_port
                else:
                    peer_addr = ni.ip
                    peer_port = ni.port

                device_caps = {
                    "model": ni.device_name or "unknown",
                    "chip": ni.device_name or "unknown",
                    "memory": ni.gpu_total_memory_mb / 1024 if ni.gpu_total_memory_mb else 0,
                }

            if not peer_addr or peer_port <= 0:
                continue

            online_peers.append({
                "node_id": other_id,
                "address": str(peer_addr),
                "port": int(peer_port),
                "device_capabilities": device_caps,
            })

        if online_peers:
            peer_list_msg = {
                "type": "peer_list",
                "peers": online_peers,
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(peer_list_msg))
            logger.info(f"📢 [NodeWS] 已向 {node_id} 推送 {len(online_peers)} 个在线节点")
    except Exception as e:
        logger.debug(f"[NodeWS] ⚠️ 推送节点列表失败（非关键）: {e}")
    
    try:
        # 保持连接，处理 Node 发来的消息
        while True:
            data = await websocket.receive_text()

            # 交给 NodeWSManager 处理
            await node_ws_manager.handle_node_message(node_id, data)

    except WebSocketDisconnect:
        logger.info(f"🔌 [NodeWS] 节点 {node_id} 断开连接")
        await node_ws_manager.disconnect_node(node_id)

    except Exception as e:
        err_msg = str(e)
        logger.error(f"[NodeWS] ❌ 节点 {node_id} 连接错误: {err_msg}", exc_info=True)
        try:
            await node_ws_manager.disconnect_node(node_id)
        except Exception as disconnect_err:
            logger.warning(f"[NodeWS] ⚠️ 清理节点 {node_id} 时出错（可忽略）: {disconnect_err}")


# ==================== Web UI 页面路由 ====================

# 忽略 Vite 开发工具请求（避免 404 错误）
@app.get("/@vite/{path:path}")
async def ignore_vite_request(path: str):
    """忽略 Vite 开发服务器的客户端请求"""
    from fastapi.responses import Response
    return Response(status_code=204)  # No Content

@app.get("/admin", response_class=HTMLResponse)
async def serve_login_page():
    """提供管理员登录页面"""
    html_file = Path(__file__).parent / "static" / "login.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Login page not found</h1>", status_code=404)


@app.get("/admin/login", response_class=HTMLResponse)
async def serve_admin_login_page():
    """提供管理员登录页面 (/admin/login)"""
    html_file = Path(__file__).parent / "static" / "login.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Login page not found</h1>", status_code=404)


@app.get("/admin/dashboard", response_class=HTMLResponse)
async def serve_admin_dashboard():
    """提供管理员后台页面"""
    html_file = Path(__file__).parent / "static" / "index.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


@app.get("/login", response_class=HTMLResponse)
async def serve_user_login_page():
    """提供普通用户登录页面"""
    html_file = Path(__file__).parent / "static" / "user_login.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding='utf-8'))
    # 回退到通用登录页
    login_file = Path(__file__).parent / "static" / "login.html"
    if login_file.exists():
        return HTMLResponse(content=login_file.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Login page not found</h1>", status_code=404)


@app.get("/login.html", response_class=HTMLResponse)
async def serve_user_login_page_html():
    """提供普通用户登录页面 (.html后缀兼容)"""
    return await serve_user_login_page()


@app.get("/user/login", response_class=HTMLResponse)
async def serve_user_login_page_alias():
    """普通用户登录页别名 (保持兼容)"""
    return await serve_user_login_page()


@app.get("/user/dashboard", response_class=HTMLResponse)
async def serve_user_dashboard(request: Request):
    """普通用户仪表板 (独立URL)"""
    from auth_manager import get_auth_manager

    session_token = request.cookies.get("session_token", "")
    if session_token:
        auth_mgr = get_auth_manager()
        user = auth_mgr.validate_session(session_token)
        if user and user.role != "admin":
            user_html = Path(__file__).parent / "static" / "user.html"
            if user_html.exists():
                return HTMLResponse(content=user_html.read_text(encoding='utf-8'))

    # 未登录或非用户角色，重定向到用户登录页
    login_html = Path(__file__).parent / "static" / "user_login.html"
    if login_html.exists():
        return HTMLResponse(content=login_html.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Please login first</h1>", status_code=403)

@app.get("/", response_class=HTMLResponse)
async def serve_landing(request: Request):
    """默认显示项目介绍落地页"""
    landing_html = Path(__file__).parent / "static" / "landing.html"
    if landing_html.exists():
        return HTMLResponse(content=landing_html.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Page not found</h1>", status_code=404)


# ==================== 节点安全管理 API (轻量版) ====================

@app.get("/api/security/status", response_model=Dict)
async def get_security_status():
    """获取安全状态 (轻量)"""
    try:
        from node_security import get_security_manager
        return {"success": True, "data": get_security_manager().status(), "timestamp": time.time()}
    except ImportError:
        raise HTTPException(status_code=503, detail="安全模块未安装")
    except Exception as e:
        logger.error(f"获取安全状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/security/ip/{ip_address}", response_model=Dict)
async def get_ip_info(ip_address: str):
    """查询IP信息"""
    try:
        from node_security import get_security_manager
        return {"success": True, "data": get_security_manager().get_ip_info(ip_address)}
    except ImportError:
        raise HTTPException(status_code=503, detail="安全模块未安装")


@app.post("/api/security/ban", response_model=Dict)
async def ban_ip(request: Dict[str, Any]):
    """手动封禁IP"""
    try:
        from node_security import get_security_manager
        sec = get_security_manager()
        ip = request.get("ip", "")
        if not ip:
            raise HTTPException(status_code=400, detail="必须提供IP")
        
        sec.ban(ip, request.get("reason", "Manual"))
        return {"success": True, "message": f"已封禁 {ip}"}
    except ImportError:
        raise HTTPException(status_code=503, detail="安全模块未安装")


@app.post("/api/security/unban", response_model=Dict)
async def unban_ip(request: Dict[str, Any]):
    """解封IP"""
    try:
        from node_security import get_security_manager
        ip = request.get("ip", "")
        success = get_security_manager().unban(ip)
        return {"success": success, "message": f"已解封" if success else "IP未被封禁"}
    except ImportError:
        raise HTTPException(status_code=503, detail="安全模块未安装")


@app.post("/api/security/whitelist", response_model=Dict)
async def add_whitelist(request: Dict[str, Any]):
    """添加白名单"""
    try:
        from node_security import get_security_manager
        cidr = request.get("cidr", "")
        if not cidr:
            raise HTTPException(status_code=400, detail="必须提供CIDR")
        
        ok = get_security_manager().add_whitelist(cidr)
        return {"success": ok, "message": "已添加" if ok else "格式错误"}
    except ImportError:
        raise HTTPException(status_code=503, detail="安全模块未安装")


@app.post("/api/security/blacklist", response_model=Dict)
async def add_blacklist(request: Dict[str, Any]):
    """添加黑名单"""
    try:
        from node_security import get_security_manager
        cidr = request.get("cidr", "")
        if not cidr:
            raise HTTPException(status_code=400, detail="必须提供CIDR")
        
        ok = get_security_manager().add_blacklist(cidr)
        return {"success": ok, "message": "已添加" if ok else "格式错误"}
    except ImportError:
        raise HTTPException(status_code=503, detail="安全模块未安装")


@app.post("/api/security/trust", response_model=Dict)
async def trust_node(request: Dict[str, Any]):
    """标记可信节点"""
    try:
        from node_security import get_security_manager
        node_id = request.get("node_id", "")
        if not node_id:
            raise HTTPException(status_code=400, detail="必须提供node_id")
        
        get_security_manager().trust(node_id)
        return {"success": True, "message": f"{node_id} 已标记为可信"}
    except ImportError:
        raise HTTPException(status_code=503, detail="安全模块未安装")


@app.delete("/api/security/trust/{node_id}", response_model=Dict)
async def untrust_node(node_id: str):
    """取消可信标记"""
    try:
        from node_security import get_security_manager
        get_security_manager().untrust(node_id)
        return {"success": True, "message": "已取消标记"}
    except ImportError:
        raise HTTPException(status_code=503, detail="安全模块未安装")


@app.post("/api/security/node-token/generate", response_model=Dict)
async def generate_node_token(request: Dict[str, Any]):
    """
    为节点生成认证Token (管理员接口)

    Request Body:
    {
        "node_id": "worker-1",
        "extra_data": {"location": "datacenter-a"}
    }

    Returns:
        新生成的Token（仅显示一次！）

    Warning:
        Token 仅在响应中显示一次，请妥善保存！
    """
    # TODO: 添加权限验证
    try:
        from node_security import get_security_manager
        security = get_security_manager()

        node_id = request.get("node_id", "")
        extra_data = request.get("extra_data")

        if not node_id:
            raise HTTPException(status_code=400, detail="必须提供node_id")

        token = security.generate_node_token(node_id, extra_data)

        logger.info(f"[Admin] 为节点 {node_id} 生成了新的认证Token")

        return {
            "success": True,
            "data": {
                "node_id": node_id,
                "token": token,
                "warning": "Token 仅在此响应中显示一次，请立即保存到安全位置！"
            },
            "timestamp": time.time()
        }

    except ImportError:
        raise HTTPException(status_code=503, detail="安全模块未安装")
    except Exception as e:
        logger.error(f"生成Token失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/landing", response_class=HTMLResponse)
async def serve_landing_page(request: Request):
    """项目介绍落地页（备用路径）"""
    landing_html = Path(__file__).parent / "static" / "landing.html"
    if landing_html.exists():
        return HTMLResponse(content=landing_html.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Page not found</h1>", status_code=404)


# ==================== 自动模型分配 API ====================

@app.get("/api/auto-allocate/status", response_model=Dict)
async def get_auto_allocate_status():
    """
    获取自动分配器当前状态

    Returns:
        {
            "allocator_status": "active",
            "online_nodes": 3,
            "total_memory_gb": 48.0,
            "free_memory_gb": 32.0,
            "usable_memory_gb": 22.4,
            "loaded_models": 2,
            "current_plan": {...} | null,
            "library_size": 15,
            "strategies": [...]
        }
    """
    try:
        allocator = get_auto_allocator()
        return {
            "success": True,
            "data": allocator.get_current_status(),
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AutoAlloc] 获取状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auto-allocate/preview-resource", response_model=Dict)
async def preview_resource_distribution():
    """
    预览资源分布情况（不执行分配）

    用于前端展示当前集群资源和可用模型库

    Returns:
        {
            "timestamp": 1234567890,
            "nodes": [...],
            "models_by_category": {...},
            "recommendations": [...]
        }
    """
    try:
        allocator = get_auto_allocator()
        return {
            "success": True,
            "data": allocator.preview_resource_distribution(),
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AutoAlloc] 预览资源失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auto-allocate/preview", response_model=Dict)
async def preview_optimal_plan(request: Dict[str, Any] = Body(...)):
    """
    预览最优模型分配方案（不执行）

    Request Body:
    {
        "strategy": "maximize_utilization",  // 可选，默认 maximize_utilization
        "exclude_models": ["model_id"],       // 可选，要排除的模型
        "force_include": ["qwen3-4b"],        // 可选，强制包含的模型
        "custom_priority": ["id1", "id2"]     // 可选，仅 custom 策略时使用
    }

    Returns:
        完整的分配方案，包括：
        - 分配详情（每个模型的节点和分片）
        - 统计信息（总模型数、参数量、利用率）
        - 未分配模型及原因
        - 优化建议
        - 性能评分
    """
    try:
        allocator = get_auto_allocator()

        # 解析策略
        strategy_str = request.get("strategy", "maximize_utilization")
        try:
            strategy = AllocationStrategy(strategy_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"无效的策略: {strategy_str}，可选值: {[s.value for s in AllocationStrategy]}"
            )

        # 生成方案
        plan = allocator.generate_optimal_plan(
            strategy=strategy,
            exclude_models=request.get("exclude_models"),
            force_include=request.get("force_include"),
            custom_priority=request.get("custom_priority")
        )

        logger.info(f"[AutoAlloc] 📋 方案预览完成: {plan.plan_id}, "
                   f"{plan.total_models}个模型, 评分{plan.performance_score:.1f}")

        return {
            "success": True,
            "data": plan.to_dict(),
            "message": f"最优方案已生成：可加载 {plan.total_models} 个模型，"
                      f"总计 {plan.total_param_count:.1f}B 参数，"
                      f"显存利用率 {plan.memory_utilization*100:.1f}%"
        }

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"[AutoAlloc] 方案预览参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[AutoAlloc] 方案预览失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成方案失败: {str(e)}")


@app.post("/api/auto-allocate/execute", response_model=Dict)
async def execute_allocation_plan(request: Dict[str, Any] = Body(...)):
    """
    执行自动模型分配

    Request Body (两种模式):

    模式1 - 使用新生成的方案:
    {
        "strategy": "maximize_utilization",  // 可选
        "exclude_models": [...],              // 可选
        "force_include": [...]                // 可选
    }

    模式2 - 使用已有方案的plan_id:
    {
        "plan_id": "plan_1234567890"
    }

    Returns:
        {
            "executed_at": 1234567890,
            "summary": {"total": 5, "success": 4, "failed": 1, "skipped": 0},
            "results": {"success": [...], "failed": [...], "skipped": [...]}
        }
    """
    try:
        allocator = get_auto_allocator()

        # 判断使用哪种模式
        if "plan_id" in request and request["plan_id"]:
            # 模式2: 使用已有方案
            plan_id = request["plan_id"]
            plan = None
            for p in allocator.allocation_history:
                if p.plan_id == plan_id:
                    plan = p
                    break

            if not plan:
                raise HTTPException(status_code=404, detail=f"未找到方案: {plan_id}")

            logger.info(f"[AutoAlloc] 🚀 执行已有方案: {plan_id}")
        else:
            # 模式1: 生成新方案并执行
            strategy_str = request.get("strategy", "maximize_utilization")
            try:
                strategy = AllocationStrategy(strategy_str)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"无效策略: {strategy_str}")

            plan = allocator.generate_optimal_plan(
                strategy=strategy,
                exclude_models=request.get("exclude_models"),
                force_include=request.get("force_include")
            )

            logger.info(f"[AutoAlloc] 🚀 生成并执行新方案: {plan.plan_id}")

        # 异步执行分配
        report = await allocator.execute_plan(plan)

        success_count = report["summary"]["success"]
        total_count = report["summary"]["total"]

        logger.info(f"[AutoAlloc] ✅ 执行完成: {success_count}/{total_count} 成功")

        return {
            "success": True,
            "data": report,
            "message": f"分配执行完成：{success_count}/{total_count} 个模型加载成功"
        }

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"[AutoAlloc] 执行参数错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[AutoAlloc] 执行失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"执行分配失败: {str(e)}")


@app.get("/api/auto-allocate/models/library", response_model=Dict)
async def get_model_library():
    """
    获取可用模型库列表

    Returns:
        {
            "models": [
                {
                    "id": "qwen3-4b",
                    "name": "Qwen3-4B",
                    "size_gb": 3.9,
                    "params": 4,
                    "layers": 36,
                    "category": "general",
                    "priority": 1.0
                },
                ...
            ],
            "categories": ["general", "code", "math", "reasoning"],
            "total": 15
        }
    """
    models = []
    categories = set()

    for model_id, spec in get_model_library().items():
        categories.add(spec.category)
        models.append({
            "id": model_id,
            "name": spec.pretty_name,
            "size_gb": round(spec.total_memory_gb, 1),
            "params": spec.param_count,
            "layers": spec.total_layers,
            "layer_memory_mb": spec.layer_memory_mb,
            "category": spec.category,
            "context_length": spec.context_length,
            "priority": spec.priority
        })

    # 按参数量排序
    models.sort(key=lambda x: x["params"])

    return {
        "success": True,
        "data": {
            "models": models,
            "categories": sorted(list(categories)),
            "total": len(models)
        }
    }


@app.post("/api/auto-allocate/models/custom", response_model=Dict)
async def add_custom_model(request: Dict[str, Any] = Body(...)):
    """
    添加自定义模型到库中

    Request Body:
    {
        "model_id": "my-custom-model",
        "pretty_name": "My Custom Model",
        "total_layers": 40,
        "layer_memory_mb": 200,
        "param_count": 8,
        "context_length": 8192,
        "category": "general",
        "priority": 1.0
    }
    """
    try:
        allocator = get_auto_allocator()

        required_fields = ["model_id", "pretty_name", "total_layers"]
        for field in required_fields:
            if field not in request or not request[field]:
                raise HTTPException(status_code=400, detail=f"缺少必填字段: {field}")

        spec = ModelSpec(
            model_id=request["model_id"],
            pretty_name=request["pretty_name"],
            total_layers=int(request["total_layers"]),
            layer_memory_mb=float(request.get("layer_memory_mb", 100)),
            param_count=float(request.get("param_count", 0)),
            context_length=int(request.get("context_length", 8192)),
            category=request.get("category", "general"),
            priority=float(request.get("priority", 1.0))
        )

        allocator.add_custom_model(spec)

        logger.info(f"[AutoAlloc] ➕ 自定义模型已添加: {spec.pretty_name}")

        return {
            "success": True,
            "message": f"模型 '{spec.pretty_name}' 已添加到库中",
            "data": {
                "model_id": spec.model_id,
                "name": spec.pretty_name,
                "size_gb": round(spec.total_memory_gb, 1)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AutoAlloc] 添加自定义模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/auto-allocate/models/{model_id}", response_model=Dict)
async def remove_custom_model(model_id: str):
    """从模型库移除模型"""
    try:
        allocator = get_auto_allocator()

        if model_id not in get_model_library():
            raise HTTPException(status_code=404, detail=f"模型不存在: {model_id}")

        allocator.remove_model(model_id)

        logger.info(f"[AutoAlloc] ➖ 模型已移除: {model_id}")

        return {
            "success": True,
            "message": f"模型 '{model_id}' 已从库中移除"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AutoAlloc] 移除模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auto-allocate/strategies", response_model=Dict)
async def get_available_strategies():
    """
    获取所有可用的分配策略

    Returns:
        {
            "strategies": [
                {
                    "id": "maximize_utilization",
                    "name": "最大化利用率",
                    "description": "优先选择参数密度高的模型，最大化显存利用效率",
                    "recommended": true
                },
                ...
            ]
        }
    """
    strategies_info = {
        AllocationStrategy.MAXIMIZE_UTILIZATION: {
            "name": "最大化利用率",
            "description": "优先选择参数密度高的模型，最大化显存利用效率（推荐）",
            "recommended": True
        },
        AllocationStrategy.MAXIMIZE_DIVERSITY: {
            "name": "最大化多样性",
            "description": "每个类别选择代表模型，提供更丰富的服务类型"
        },
        AllocationStrategy.PRIORITIZE_LARGE: {
            "name": "优先大模型",
            "description": "优先加载大型模型，追求最强推理能力"
        },
        AllocationStrategy.BALANCED: {
            "name": "平衡策略",
            "description": "综合考虑模型大小、多样性和优先级"
        },
        AllocationStrategy.CUSTOM: {
            "name": "自定义顺序",
            "description": "按用户指定的优先级列表顺序分配"
        },
    }

    strategies = []
    for strategy, info in strategies_info.items():
        strategies.append({
            "id": strategy.value,
            "name": info["name"],
            "description": info["description"],
            "recommended": info.get("recommended", False)
        })

    return {
        "success": True,
        "data": {
            "strategies": strategies,
            "default": AllocationStrategy.MAXIMIZE_UTILIZATION.value
        }
    }


# ==================== 节点稳定性管理 API ====================

@app.get("/api/stability/report", response_model=Dict)
async def get_stability_report(node_id: Optional[str] = None):
    """
    获取节点稳定性报告

    Query Params:
        node_id: 可选，指定节点ID（不传则返回所有节点）

    Returns:
        {
            "timestamp": 1234567890,
            "total_nodes_tracked": 5,
            "nodes": {
                "node-1": {
                    "stability_status": "stable",
                    "confidence": 95.2,
                    "health_score": 98,
                    "flap_count_5min": 0,
                    "flap_count_1hour": 1,
                    ...
                }
            },
            "summary": {
                "stable_nodes": 4,
                "flapping_nodes": 1,
                "offline_nodes": 0,
                "unstable_nodes": 0
            }
        }
    """
    try:
        resilient = get_resilient_allocator()
        report = resilient.stability_mgr.get_node_stability_report(node_id)
        return {
            "success": True,
            "data": report
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Stability] 获取稳定性报告失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stability/event", response_model=Dict)
async def report_node_event(request: Dict[str, Any] = Body(...)):
    """
    上报节点状态变化事件

    Request Body:
    {
        "node_id": "worker-1",
        "event_type": "offline",          # online | offline | error | flapping
        "loaded_models": ["model1"],       # 可选，当前加载的模型列表
        "auto_execute_recovery": false     # 可选，是否自动执行故障恢复
    }

    Returns:
        处理结果和建议动作
    """
    try:
        resilient = get_resilient_allocator()

        node_id = request.get("node_id")
        event_type = request.get("event_type")

        if not node_id or not event_type:
            raise HTTPException(status_code=400, detail="必须提供 node_id 和 event_type")

        result = await resilient.on_node_event(
            node_id=node_id,
            event_type=event_type,
            loaded_models=request.get("loaded_models"),
            auto_execute=request.get("auto_execute_recovery", False)
        )

        logger.info(f"[Stability] 📥 事件处理完成: {node_id}/{event_type} → "
                   f"{result['stability']['stability_status']}")

        return {
            "success": True,
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Stability] 处理节点事件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stability/should-reallocate/{node_id}", response_model=Dict)
async def check_should_reallocate(node_id: str):
    """
    检查是否应该对指定节点触发重分配

    Args:
        node_id: 节点ID

    Returns:
        {
            "should_reallocate": true/false,
            "reason": "原因说明",
            "current_status": "confirmed_offline"
        }
    """
    try:
        resilient = get_resilient_allocator()
        should, reason = resilient.stability_mgr.should_trigger_reallocation(node_id)

        record = resilient.stability_mgr.records.get(node_id)

        return {
            "success": True,
            "data": {
                "node_id": node_id,
                "should_reallocate": should,
                "reason": reason,
                "current_status": record.stability_status.value if record else "unknown",
                "in_cooldown": record.in_cooldown if record else False,
                "flap_count_recent": record.flap_count_5min if record else 0
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Stability] 检查重分配条件失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 故障恢复 API ====================

@app.post("/api/recovery/handle-failure/{node_id}", response_model=Dict)
async def handle_node_failure(node_id: str, auto_execute: bool = False):
    """
    处理节点故障并生成恢复计划

    Args:
        node_id: 故障节点ID
        auto_execute: 是否自动执行恢复（默认false）

    Returns:
        恢复计划详情
    """
    try:
        resilient = get_resilient_allocator()

        logger.warning(f"[Recovery] 🚨 手动触发故障处理: {node_id}")

        recovery_plan = await resilient.recovery_mgr.handle_node_failure(node_id)

        if auto_execute and recovery_plan["action"] == "plan_generated":
            exec_result = await resilient.recovery_mgr.execute_recovery_plan(node_id)
            recovery_plan["execution_result"] = exec_result

        return {
            "success": True,
            "data": recovery_plan,
            "message": f"已生成恢复计划: {recovery_plan['summary']['will_migrate']}个迁移, "
                      f"{recovery_plan['summary']['will_degrade']}个降级"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Recovery] 故障处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recovery/history", response_model=Dict)
async def get_recovery_history(limit: int = 20):
    """获取故障恢复历史记录"""
    try:
        resilient = get_resilient_allocator()

        history = resilient.recovery_mgr.recovery_history[-limit:]

        return {
            "success": True,
            "data": {
                "history": history,
                "total": len(resilient.recovery_mgr.recovery_history),
                "showing": len(history)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Recovery] 获取历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/recovery/active", response_model=Dict)
async def get_active_recoveries():
    """获取当前正在进行的恢复任务"""
    try:
        resilient = get_resilient_allocator()

        active = {}
        for node_id, actions in resilient.recovery_mgr.active_recoveries.items():
            active[node_id] = [
                {
                    "model_id": a.model_id,
                    "action_type": a.action_type,
                    "priority": a.priority,
                    "target_node": a.target_node
                }
                for a in actions
            ]

        return {
            "success": True,
            "data": {
                "active_recoveries": active,
                "count": len(active)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Recovery] 获取活跃恢复失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 方案版本管理 API ====================

@app.get("/api/plans/versions", response_model=Dict)
async def list_allocation_versions(limit: int = 10):
    """
    列出分配方案版本历史

    Query Params:
        limit: 返回数量限制（默认10）

    Returns:
        版本列表
    """
    try:
        resilient = get_resilient_allocator()
        versions = resilient.plan_mgr.list_versions(limit)

        return {
            "success": True,
            "data": {
                "versions": versions,
                "current_version": resilient.plan_mgr.current_version_id,
                "total_versions": len(resilient.plan_mgr.versions)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PlanMgr] 获取版本列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/plans/compare", response_model=Dict)
async def compare_plan_versions(
    version1: str,
    version2: Optional[str] = None
):
    """
    对比两个分配方案版本的差异

    Query Params:
        version1: 第一个版本ID
        version2: 第二个版本ID（可选，默认与当前版本对比）

    Returns:
        差异详情
    """
    try:
        resilient = get_resilient_allocator()

        if not version2:
            version2 = resilient.plan_mgr.current_version_id
            if not version2:
                raise HTTPException(status_code=404, detail="无当前激活版本")

        diff = resilient.plan_mgr.compare_versions(version1, version2)

        return {
            "success": True,
            "data": diff
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[PlanMgr] 版本对比失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/plans/rollback/{version_id}", response_model=Dict)
async def rollback_to_version(version_id: str):
    """
    回滚到指定的方案版本

    Args:
        version_id: 目标版本ID

    Returns:
        回滚结果
    """
    try:
        resilient = get_resilient_allocator()

        # 先检查是否可以回滚
        can_rollback, reason = resilient.plan_mgr.can_rollback_to(version_id)
        if not can_rollback:
            raise HTTPException(status_code=400, detail=f"无法回滚: {reason}")

        # 执行回滚
        plan = resilient.plan_mgr.rollback_to(version_id)

        logger.warning(f"[PlanMgr] ⏪ 方案回滚完成: {version_id}")

        return {
            "success": True,
            "message": f"已成功回滚到版本 {version_id}",
            "data": {
                "rolled_back_to": version_id,
                "new_current_version": resilient.plan_mgr.current_version_id,
                "plan_summary": {
                    "total_models": plan.total_models,
                    "performance_score": plan.performance_score,
                    "memory_utilization": plan.memory_utilization * 100
                } if plan else None
            }
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[PlanMgr] 回滚失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 弹性分配综合 API ====================

@app.get("/api/resilient/dashboard", response_model=Dict)
async def get_resilient_dashboard():
    """
    获取弹性分配仪表板数据（整合所有信息）

    Returns:
        综合仪表板数据，包括：
        - 稳定性状态
        - 最近方案版本
        - 恢复历史
        - 分配器状态
    """
    try:
        resilient = get_resilient_allocator()
        dashboard_data = resilient.get_dashboard_data()

        return {
            "success": True,
            "data": dashboard_data,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Resilient] 获取仪表板数据失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/resilient/safe-plan", response_model=Dict)
async def generate_safe_plan(request: Dict[str, Any] = Body(...)):
    """
    安全生成分配方案（带稳定性检查）

    与普通 /api/auto-allocate/preview 的区别：
    - 会检查是否有不稳定节点
    - 不稳定时会发出警告但仍会生成方案
    - 自动保存到版本历史中

    Request Body:
    {
        "strategy": "balanced",
        "force": false,              // 强制跳过安全检查
        "exclude_models": [...],
        "force_include": [...]
    }
    """
    try:
        resilient = get_resilient_allocator()

        strategy = request.get("strategy", "maximize_utilization")
        force = request.get("force", False)

        plan, version_id = await resilient.safe_generate_plan(
            strategy=strategy,
            force=force,
            exclude_models=request.get("exclude_models"),
            force_include=request.get("force_include")
        )

        # 获取稳定性警告
        stability_warnings = []
        if not force:
            stability_report = resilient.stability_mgr.get_node_stability_report()
            summary = stability_report.get("summary", {})
            if summary.get("flapping_nodes", 0) > 0:
                stability_warnings.append(
                    f"检测到 {summary['flapping_nodes']} 个抖动节点，方案可能需要调整"
                )
            if summary.get("unstable_nodes", 0) > 0:
                stability_warnings.append(
                    f"⚠️ 检测到 {summary['unstable_nodes']} 个极不稳定节点"
                )

        return {
            "success": True,
            "data": {
                "plan": plan.to_dict(),
                "version_id": version_id,
                "stability_warnings": stability_warnings,
                "is_safe_generation": not force
            },
            "warnings": stability_warnings,
            "message": f"安全方案已生成 (版本: {version_id})" + (
                f"，包含{len(stability_warnings)}条稳定性警告" if stability_warnings else ""
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[Resilient] 安全方案生成失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 主备冗余（HA）分配 API ====================

@app.post("/api/ha/generate-plan", response_model=Dict)
async def generate_ha_plan(request: Dict[str, Any] = Body(...)):
    """
    生成高可用（主备冗余）分配方案

    与普通分配的核心区别：
    ✅ 每个关键模型至少2个副本（主+备）
    ✅ 避免单点故障（Anti-SPOF）
    ✅ 自动选择异构节点作为备份
    ✅ 内置健康检查和故障转移能力

    Request Body:
    {
        "mode": "active_passive",              // 冗余模式
        "min_replicas_critical": 2,            // 关键模型最少副本数
        "min_replicas_normal": 1,              // 普通模型最少副本数
        "max_single_node_ratio": 0.4,          // 单节点承载上限(40%)
        "exclude_models": [...],
        "force_include": ["qwen3-8b"],
        "prioritize_redundancy": true           // 优先保证冗余
    }

    Returns:
        完整的HA分配方案，包括：
        - 每个模型的主/备实例分布
        - SPOF风险评估
        - 可用性预估（如99.99%）
        - 故障转移配置
    """
    try:
        ha = get_ha_allocator()

        # 解析参数
        mode_str = request.get("mode", "active_passive")
        try:
            mode = RedundancyMode(mode_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"无效的冗余模式: {mode_str}，可选: active_passive, active_active"
            )

        # 生成HA方案
        plan = ha.generate_ha_plan(
            mode=mode,
            min_replicas_critical=request.get("min_replicas_critical", 2),
            min_replicas_normal=request.get("min_replicas_normal", 1),
            max_single_node_ratio=request.get("max_single_node_ratio", 0.4),
            exclude_models=request.get("exclude_models"),
            force_include=request.get("force_include"),
            prioritize_redundancy=request.get("prioritize_redundancy", True)
        )

        logger.info(f"[HAAlloc] 🎯 HA方案生成完成: "
                   f"{plan.total_models}个模型, "
                   f"{plan.total_instances}个实例, "
                   f"平均冗余{plan.avg_redundancy:.2f}x, "
                   f"SPOF风险{plan.spof_risk_score:.1f}/100, "
                   f"可用性{plan.estimated_availability*100:.2f}%")

        return {
            "success": True,
            "data": plan.to_dict(),
            "message": f"✅ HA方案已生成: {plan.total_models}个模型, "
                      f"{plan.total_instances}个实例(含{len([m for m in plan.vulnerable_models])}个无备份), "
                      f"预估可用性{plan.estimated_availability*100:.2f}%"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[HAAlloc] HA方案生成失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成HA方案失败: {str(e)}")


@app.get("/api/ha/health-status", response_model=Dict)
async def get_ha_health_status():
    """
    获取HA系统整体健康状态

    Returns:
        {
            "overall_status": "healthy" | "degraded" | "unhealthy",
            "health_percentage": 98.5,
            "instances": {"total": 10, "healthy": 9, "degraded": 1, ...},
            "models_with_issues": ["model_id"]
        }
    """
    try:
        ha = get_ha_allocator()
        health = ha.get_health_status()

        return {
            "success": True,
            "data": health,
            "timestamp": time.time()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[HAAlloc] 获取健康状态失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ha/health-check", response_model=Dict)
async def trigger_health_check():
    """
    手动触发一轮健康检查

    Returns:
        所有实例的检查结果列表
    """
    try:
        ha = get_ha_allocator()

        results = await ha.perform_health_check()

        summary = {
            "total_checked": len(results),
            "healthy": sum(1 for r in results if r.status == InstanceStatus.HEALTHY),
            "degraded": sum(1 for r in results if r.status == InstanceStatus.DEGRADED),
            "unhealthy": sum(1 for r in results if r.status == InstanceStatus.UNHEALTHY),
            "down": sum(1 for r in results if r.status == InstanceStatus.DOWN),
        }

        logger.info(f"[HAAlloc] 🩺 健康检查完成: "
                   f"{summary['total_checked']}个实例, "
                   f"✅{summary['healthy']} ⚠️{summary['degraded']} ❌{summary['unhealthy']} 💀{summary['down']}")

        return {
            "success": True,
            "data": {
                "results": [
                    {
                        "model_id": r.model_id,
                        "instance_id": r.instance_id,
                        "node_id": r.node_id,
                        "status": r.status.value,
                        "latency_ms": round(r.latency_ms, 1),
                        "error": r.error,
                    }
                    for r in results
                ],
                "summary": summary,
                "timestamp": time.time()
            },
            "message": f"检查完成: {summary['healthy']}/{summary['total_checked']} 健康"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[HAAlloc] 健康检查失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ha/failover/{model_id}", response_model=Dict)
async def execute_failover(
    model_id: str,
    failed_node: str = Body(..., embed=True),
    auto_promote: bool = True
):
    """
    执行故障转移（手动或自动触发）

    当某个节点的模型实例发生故障时：
    1. 将该实例标记为宕机
    2. 自动查找并提升备用实例为主实例
    3. 更新路由表，将流量导向新主实例
    4. 记录完整的转移日志

    Args:
        model_id: 发生故障的模型ID
        failed_node: 故障节点ID
        auto_promote: 是否自动提升备用实例

    Returns:
        转移结果和详情
    """
    try:
        ha = get_ha_allocator()

        logger.warning(f"[HAAlloc] 🚨 手动触发故障转移: "
                      f"模型={model_id}, 节点={failed_node}")

        record = await ha.failover(
            model_id=model_id,
            failed_node=failed_node,
            auto_promote=auto_promote
        )

        status = "✅ 成功" if record.success else "❌ 失败"
        logger.info(f"[HAAlloc] 故障转移{status}: "
                   f"{record.failed_node} → {record.promoted_node or '无可用备'}, "
                   f"耗时{record.duration_ms:.0f}ms")

        return {
            "success": record.success,
            "data": {
                "record_id": record.record_id,
                "model_id": record.model_id,
                "failed_instance": record.failed_instance,
                "failed_node": record.failed_node,
                "promoted_instance": record.promoted_instance,
                "promoted_node": record.promoted_node,
                "state": record.failover_state.value,
                "duration_ms": round(record.duration_ms, 1),
                "requests_lost": record.requests_lost,
            },
            "message": f"故障转移{status}: {failed_node} → {record.promoted_node or '无备'}"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[HAAlloc] 故障转移失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"故障转移失败: {str(e)}")


@app.post("/api/ha/recover-node/{node_id}", response_model=Dict)
async def recover_failed_node(node_id: str):
    """
    恢复故障节点

    当一个之前掉线的节点重新上线时调用：
    1. 标记该节点上的所有实例为恢复中
    2. 对于已经故障转移过的模型：降恢复后的实例为备用
    3. 更新系统状态

    Args:
        node_id: 恢复的节点ID

    Returns:
        恢复操作详情
    """
    try:
        ha = get_ha_allocator()

        result = await ha.recover_node(node_id)

        logger.info(f"[HAAlloc] 🔄 节点恢复: {node_id}, "
                   f"{len(result['recovered_instances'])}个实例恢复, "
                   f"{len(result['re_demoted'])}个降为备用")

        return {
            "success": True,
            "data": result,
            "message": f"节点 {node_id} 已恢复: "
                      f"{len(result['recovered_instances'])}实例就绪"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[HAAlloc] 节点恢复处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ha/failover-history", response_model=Dict)
async def get_failover_history(limit: int = 20):
    """获取故障转移历史记录"""
    try:
        ha = get_ha_allocator()
        history = ha.get_failover_history(limit)

        return {
            "success": True,
            "data": {
                "history": history,
                "total": len(ha.failover_records),
                "showing": len(history)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[HAAlloc] 获取转移历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ha/modes", response_model=Dict)
async def get_available_ha_modes():
    """
    获取可用的HA冗余模式

    Returns:
        模式列表及说明
    """
    modes_info = {
        RedundancyMode.ACTIVE_PASSIVE: {
            "name": "主备热备 (Active-Passive)",
            "description": "推荐模式。1个活跃主实例 + 1个热备实例。主故障时备秒级接管，有短暂中断(<5秒)",
            "use_cases": ["生产环境", "高可用优先", "资源有限时"],
            "redundancy_level": "N+1",
            "rto_seconds": "< 5",
            "availability": "~99.99%",
            "recommended": True
        },
        RedundancyMode.ACTIVE_ACTIVE: {
            "name": "双活负载均衡 (Active-Active)",
            "description": "2个实例同时服务请求，负载自动分散。任一故障无感知切换",
            "use_cases": ["高性能场景", "零停机要求", "资源充足时"],
            "redundancy_level": "N+N",
            "rto_seconds": "< 1",
            "availability": "~99.999%",
            "recommended": False
        },
        RedundancyMode.N_PLUS_M: {
            "name": "N+M 冗余",
            "description": "N个活跃实例 + M个备用实例。适用于大规模集群和多级容灾",
            "use_cases": ["超大规模部署", "多机房容灾", "企业级高可用"],
            "redundancy_level": "N+M (可配置)",
            "rto_seconds": "< 3",
            "availability": "~99.9999%",
            "recommended": False
        },
    }

    modes = []
    for mode, info in modes_info.items():
        modes.append({
            "id": mode.value,
            "name": info["name"],
            "description": info["description"],
            "use_cases": info["use_cases"],
            "redundancy_level": info["redundancy_level"],
            "rto_seconds": info["rto_seconds"],
            "availability": info["availability"],
            "recommended": info.get("recommended", False)
        })

    return {
        "success": True,
        "data": {
            "modes": modes,
            "default": RedundancyMode.ACTIVE_PASSIVE.value,
            "configuration": {
                "health_check_interval_sec": HAModelAllocator.HEALTH_CHECK_INTERVAL,
                "failover_timeout_ms": HAModelAllocator.FAILOVER_TIMEOUT_MS,
                "heartbeat_timeout_sec": HAModelAllocator.HEARTBEAT_TIMEOUT_SECONDS,
            }
        }
    }


# ============================================================
#  自动分配触发器 API
# ============================================================

@app.get("/api/auto-trigger/status", response_model=Dict)
async def get_auto_trigger_status():
    """
    获取自动分配触发器的状态

    Returns:
        触发器运行状态、配置参数、最近事件等
    """
    trigger = get_auto_trigger()
    status = trigger.get_status()

    return {
        "success": True,
        "data": status,
        "message": "自动分配触发器状态"
    }


@app.get("/api/auto-trigger/history", response_model=Dict)
async def get_auto_trigger_history(limit: int = Query(20, ge=1, le=100)):
    """
    获取自动分配历史记录

    Args:
        limit: 返回的最大条数（1-100）

    Returns:
        最近的事件列表
    """
    trigger = get_auto_trigger()
    history = trigger.get_history(limit=limit)

    return {
        "success": True,
        "data": {
            "history": history,
            "total_events": len(trigger.event_history),
            "limit": limit
        },
        "message": f"最近 {min(len(history), limit)} 条分配事件"
    }


@app.post("/api/auto-trigger/trigger", response_model=Dict)
async def manual_trigger_allocation(
    reason: str = Query("用户手动触发", description="触发原因"),
    force: bool = Query(True, description="是否强制执行")
):
    """
    手动触发一次自动分配评估和执行

    可以用于：
    - 紧急情况下的立即重分配
    - 测试自动分配功能
    - 在节点变化后手动触发优化

    Args:
        reason: 触发原因描述
        force: 是否强制执行（跳过冷却期检查）

    Returns:
        分配事件的完整信息
    """
    trigger = get_auto_trigger()

    event = await trigger.manual_trigger(reason=reason, force=force)

    return {
        "success": event.execution_success or event.decision == "defer",
        "data": event.to_dict(),
        "message": (
            f"✅ 分配成功：{event.models_allocated}/{event.models_total} 个模型"
            if event.execution_success else
            (f"⏭️ 已跳过：{event.skip_reason}" if event.decision == "skip" else
             f"❌ 执行失败：{event.error}")
        )
    }


@app.post("/api/auto-trigger/configure", response_model=Dict)
async def configure_auto_trigger(
    enabled: Optional[bool] = Body(None, description="是否启用"),
    strategy: Optional[str] = Body(None, description="分配策略"),
    auto_execute: Optional[bool] = Body(None, description="是否自动执行方案"),
    cooldown_seconds: Optional[int] = Body(None, description="冷却期(秒)"),
    min_online_nodes: Optional[int] = Body(None, description="最小在线节点数")
):
    """
    配置自动分配触发器参数

    可以动态调整触发器的行为，无需重启服务

    Args:
        enabled: 是否启用自动分配
        strategy: 分配策略 (active_passive / active_active / n_plus_m)
        auto_execute: 是否自动执行生成的方案
        cooldown_seconds: 冷却期时长（秒）
        min_online_nodes: 最少需要几个在线节点才触发

    Returns:
        更新后的配置
    """
    trigger = get_auto_trigger()
    config = trigger.config

    updates = []

    if enabled is not None:
        config.ENABLED = enabled
        updates.append(f"启用={enabled}")

    if strategy is not None:
        valid_strategies = ["active_passive", "active_active", "n_plus_m"]
        if strategy not in valid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"无效策略，可选值: {valid_strategies}"
            )
        config.DEFAULT_STRATEGY = strategy
        updates.append(f"策略={strategy}")

    if auto_execute is not None:
        config.AUTO_EXECUTE = auto_execute
        updates.append(f"自动执行={auto_execute}")

    if cooldown_seconds is not None:
        if cooldown_seconds < 30:
            raise HTTPException(status_code=400, detail="冷却期不能小于30秒")
        config.COOLDOWN_SECONDS = cooldown_seconds
        updates.append(f"冷却期={cooldown_seconds}s")

    if min_online_nodes is not None:
        if min_online_nodes < 1:
            raise HTTPException(status_code=400, detail="最小在线节点数不能小于1")
        config.MIN_ONLINE_NODES = min_online_nodes
        updates.append(f"最小节点数={min_online_nodes}")

    logger.info(f"[AutoTrigger] ⚙️ 配置已更新: {', '.join(updates)}")

    # 如果从禁用变为启用，启动触发器
    if enabled and not trigger._is_running:
        asyncio.create_task(trigger.start())

    # 如果从启用变为禁用，停止触发器
    if enabled is False and trigger._is_running:
        await trigger.stop()

    return {
        "success": True,
        "data": {
            "updated_fields": updates,
            "current_config": {
                "enabled": config.ENABLED,
                "strategy": config.DEFAULT_STRATEGY,
                "auto_execute": config.AUTO_EXECUTE,
                "cooldown_seconds": config.COOLDOWN_SECONDS,
                "min_online_nodes": config.MIN_ONLINE_NODES,
            }
        },
        "message": f"配置已更新: {', '.join(updates)}"
    }


@app.get("/register", response_class=HTMLResponse)
async def serve_register(request: Request):
    """用户注册页"""
    register_html = Path(__file__).parent / "static" / "register.html"
    if register_html.exists():
        return HTMLResponse(content=register_html.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Page not found</h1>", status_code=404)


@app.get("/register.html", response_class=HTMLResponse)
async def serve_register_html(request: Request):
    """用户注册页 (.html后缀兼容)"""
    return await serve_register(request)


@app.get("/terms", response_class=HTMLResponse)
async def serve_terms_page(request: Request):
    """服务条款页"""
    terms_html = Path(__file__).parent / "static" / "terms.html"
    if terms_html.exists():
        return HTMLResponse(content=terms_html.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Page not found</h1>", status_code=404)


@app.get("/privacy", response_class=HTMLResponse)
async def serve_privacy_page(request: Request):
    """隐私政策页"""
    privacy_html = Path(__file__).parent / "static" / "privacy.html"
    if privacy_html.exists():
        return HTMLResponse(content=privacy_html.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Page not found</h1>", status_code=404)


def get_fallback_html() -> str:
    """当静态文件不存在时的回退HTML"""
    return '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EXO Cluster Manager</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { display: inline-block; padding: 4px 12px; border-radius: 12px; 
                  font-size: 14px; font-weight: bold; }
        .online { background: #d4edda; color: #155724; }
        .offline { background: #f8d7da; color: #721c24; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; font-weight: 600; }
        code { background: #e9ecef; padding: 2px 6px; border-radius: 3px; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🖥️ EXO Cluster Manager</h1>
        <div class="card">
            <h2>API 服务已启动 ✅</h2>
            <p>RESTful API 可在以下地址访问:</p>
            <ul>
                <li><code>GET /api/nodes</code> - 获取节点列表</li>
                <li><code>GET /api/cluster/status</code> - 集群状态</li>
                <li><code>GET /api/pool/status</code> - GPU池状态</li>
                <li><code>GET /docs</code> - API文档 (Swagger UI)</li>
            </ul>
            <p>WebSocket实时更新: <code>WS /ws/cluster</code></p>
        </div>
        <div class="card">
            <h2>快速开始</h2>
            <ol>
                <li>确保有运行中的EXO节点</li>
                <li>配置 <code>network_config.json</code></li>
                <li>访问 <a href="/docs">API文档</a> 了解详细用法</li>
            </ol>
        </div>
    </div>
</body>
</html>
'''


# ==================== 命令行入口 ====================

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="EXO Cluster Manager Server")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址 (默认: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="监听端口 (默认: 8080)")
    parser.add_argument("--config", type=str, default=None, 
                       help="网络配置文件路径 (可选)")
    parser.add_argument("--reload", action="store_true", 
                       help="开发模式：启用热重载")
    
    args = parser.parse_args()
    
    # 设置配置文件路径
    if args.config:
        from cluster_core import cluster_manager as cm_cls
        # 在这里设置全局配置路径
    
    banner = f"""
╔════════════════════════════════════════════════════╗
║                                                      ║
║   EXO Cluster Manager v1.0                           ║
║                                                      ║
║   API服务: http://{args.host}:{args.port}             
║   API文档: http://{args.host}:{args.port}/docs       
║   Web界面: http://{args.host}:{args.port}/          
║                                                      ║
╚════════════════════════════════════════════════════╝
"""
    
    try:
        print(banner)
    except UnicodeEncodeError:
        # Windows GBK编码不支持特殊字符时的降级方案
        print(f"EXO Cluster Manager v1.0")
        print(f"  API: http://{args.host}:{args.port}")
        print(f"  Docs: http://{args.host}:{args.port}/docs")
        print(f"  Web: http://{args.host}:{args.port}/")
    
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
