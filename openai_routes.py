"""
EXO Cluster Manager - OpenAI 兼容 API 路由 (轻量代理模式)
========================================================

架构设计:
---------
exo_manager 作为轻量认证网关，只做三件事:
1. 认证验证 (API Key / Session)
2. 额度检查与扣减
3. 流式转发请求到 exo 节点 (HTTP 代理)

实际推理由 exo 节点执行，流量不经过 manager 处理。

端点列表:
---------
POST /v1/chat/completions      - 聊天补全 (流式代理到节点)
POST /v1/completions           - 文本补全 (流式代理到节点)
GET  /v1/models                - 列出可用模型
GET  /v1/models/{model}        - 获取模型信息

API Key 管理:
--------------
POST /v1/admin/keys            - 生成新 API Key
GET  /v1/admin/keys            - 列出所有 API Key
DELETE /v1/admin/keys/{key}    - 吊销 API Key
POST /v1/user/keys             - 用户创建 Key
GET  /v1/user/keys             - 用户查看 Keys
DELETE /v1/user/keys/{key}     - 用户吊销 Key

额度管理:
----------
GET  /v1/user/quota            - 查看个人额度
GET  /v1/user/quota/history    - 使用历史
GET  /v1/admin/quotas          - 管理员查看所有额度
POST /v1/admin/quota/{id}/recharge  - 充值
PUT  /v1/admin/quota/{id}      - 设置额度
POST /v1/admin/quota/{id}/reset   - 重置额度

认证方式:
---------
Authorization: Bearer <api_key>  或  Cookie: session_token=<token>
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Any, Union, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field


# ==================== 认证/额度结果缓存 ====================
# 避免每次请求都查 SQLite，减少首 token 延迟 10-50ms

class _TTLCache:
    """简单的线程安全 TTL 缓存"""

    def __init__(self, ttl: float = 5.0):
        self._cache: Dict[str, Tuple[float, Any]] = {}  # key -> (expiry, value)
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry[0] < time.time():
            del self._cache[key]
            return None
        return entry[1]

    def set(self, key: str, value: Any, ttl: Optional[float] = None):
        self._cache[key] = (time.time() + (ttl or self._ttl), value)

    def invalidate(self, key: str):
        """失效单个 key"""
        self._cache.pop(key, None)

    def invalidate_prefix(self, prefix: str):
        """按前缀批量失效（如扣减额度后清除某用户所有缓存）"""
        keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._cache[k]


# 缓存实例：认证 30s，额度 5s（额度扣减后需要较快失效）
_auth_cache = _TTLCache(ttl=30.0)
_quota_cache = _TTLCache(ttl=5.0)

# 导入核心模块
import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from api_key_manager import get_api_key_manager, APIKeyManager
from auth_manager import User as AuthUser
from load_balancer import LBStrategy

# 导入 Node WebSocket 管理器（用于内网穿透场景）
try:
    from server import node_ws_manager
    _ws_available = True
except ImportError:
    node_ws_manager = None
    _ws_available = False

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter(prefix="/v1")

# HTTP 客户端 (复用连接池)
_http_client: Optional[httpx.AsyncClient] = None

def _get_http_client() -> httpx.AsyncClient:
    """获取或创建共享的 HTTP 客户端"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        )
    return _http_client


# ==================== Pydantic 模型定义 ====================

class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: system/user/assistant/tool")
    content: Union[str, List[Dict[str, Any]], None] = Field(None, description="消息内容")
    name: Optional[str] = Field(None, description="发送者名称")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="模型ID")
    messages: List[ChatMessage] = Field(..., description="消息列表")
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    top_p: Optional[float] = Field(0.9, ge=0, le=1)
    top_k: Optional[int] = Field(50, ge=1)
    max_tokens: Optional[int] = Field(512, ge=1)
    stream: Optional[bool] = Field(False)
    stop: Optional[Union[str, List[str]]] = Field(None)
    presence_penalty: Optional[float] = Field(0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(0, ge=-2, le=2)
    user: Optional[str] = Field(None, description="用户标识")


class CompletionRequest(BaseModel):
    model: str = Field(..., description="模型ID")
    prompt: Union[str, List[str]] = Field(..., description="提示文本")
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    top_p: Optional[float] = Field(0.9, ge=0, le=1)
    top_k: Optional[int] = Field(50, ge=1)
    max_tokens: Optional[int] = Field(512, ge=1)
    stream: Optional[bool] = Field(False)
    stop: Optional[Union[str, List[str]]] = Field(None)
    presence_penalty: Optional[float] = Field(0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(0, ge=-2, le=2)
    user: Optional[str] = Field(None, description="用户标识")


class AddQuotaRequest(BaseModel):
    amount: int = Field(..., gt=0, description="增加的token数量")
    source: str = Field("", description="来源，如ad/reward等")


class CreateKeyRequest(BaseModel):
    name: str = Field("", description="Key 名称")
    description: str = Field("", description="Key 描述")
    allowed_models: Optional[List[str]] = Field(None, description="允许的模型列表")


# ==================== 依赖注入 ====================

async def verify_api_key(request: Request) -> Tuple[str, Optional[str]]:
    """
    验证 API Key 或 Session Token（带 30s TTL 缓存）

    支持两种认证方式：
    1. Authorization: Bearer <api_key> - 用于外部 API 调用
    2. Cookie: session_token=<token> - 用于 Web UI / 小程序

    Returns:
        (token_string, user_id) - user_id 可用于额度管理
    """
    # 构建缓存 key（基于请求头/cookie）
    auth_header = request.headers.get("Authorization", "")
    session_token = request.cookies.get("session_token", "")
    cache_key = f"auth:{auth_header}:{session_token}"

    # 命中缓存直接返回
    cached = _auth_cache.get(cache_key)
    if cached:
        return cached

    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:].strip()
        if api_key:
            key_manager = get_api_key_manager()
            if key_manager.validate_key(api_key):
                user_id = None
                key_info = key_manager._keys.get(api_key)
                if key_info and key_info.get("user_id"):
                    user_id = key_info["user_id"]
                result = (api_key, user_id)
                _auth_cache.set(cache_key, result)
                return result

    if session_token:
        from auth_manager import get_auth_manager
        auth_mgr = get_auth_manager()
        user = auth_mgr.validate_session(session_token)
        if user:
            result = (f"session:{user.id}", user.id)
            _auth_cache.set(cache_key, result)
            return result

    raise HTTPException(status_code=401, detail="Missing authentication")


async def verify_admin_key(request: Request) -> str:
    """验证管理员权限"""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:].strip()
        key_manager = get_api_key_manager()
        if key_manager.validate_key(api_key):
            return api_key

    session_token = request.cookies.get("session_token", "")
    if session_token:
        from auth_manager import get_auth_manager
        auth_mgr = get_auth_manager()
        user = auth_mgr.validate_session(session_token)
        if user and user.role == "admin":
            return f"session:{user.id}"
        elif user:
            raise HTTPException(status_code=403, detail="Admin access required")

    raise HTTPException(status_code=401, detail="Missing or invalid authentication")


async def verify_user_or_admin(request: Request) -> Tuple[str, Optional[AuthUser]]:
    """
    验证用户身份（支持普通用户和管理员）
    用于 API Key 管理
    """
    from auth_manager import get_auth_manager

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:].strip()
        key_manager = get_api_key_manager()
        if key_manager.validate_key(api_key):
            return (api_key, None)

    session_token = request.cookies.get("session_token", "")
    if session_token:
        auth_mgr = get_auth_manager()
        user = auth_mgr.validate_session(session_token)
        if user:
            return (f"session:{user.id}", user)

    raise HTTPException(status_code=401, detail="Missing authentication")


async def verify_admin_only(request: Request) -> Tuple[str, AuthUser]:
    """验证管理员权限（仅管理员可访问）"""
    from auth_manager import get_auth_manager

    session_token = request.cookies.get("session_token", "")
    if session_token:
        auth_mgr = get_auth_manager()
        user = auth_mgr.validate_session(session_token)
        if user:
            if user.role == "admin":
                return (f"session:{user.id}", user)
            raise HTTPException(status_code=403, detail="Admin access required")

    raise HTTPException(status_code=401, detail="Admin authentication required")


# ==================== 核心辅助函数 ====================

def _get_manager():
    """获取全局集群管理器实例"""
    try:
        import server
        return getattr(server, 'manager', None)
    except ImportError:
        return None


def _status_is_online(status) -> bool:
    """安全获取节点在线状态（兼容 Enum 和字符串）"""
    if hasattr(status, 'value'):
        return status.value == "online"
    return str(status) == "online"


def _get_available_models_for_inference() -> List[Dict]:
    """获取所有可用于推理的已加载模型

    数据来源（按优先级）：
    1. manager._allocation_registry - 分配注册表（含准确的首层节点信息）
    2. manager.connectors - 在线节点的 loaded_models
    3. manager.nodes - 所有节点信息（fallback）

    每个模型附带 inference_url（首层节点推理地址），客户端可直连该地址发起推理，
    请求会自动在分片节点间传递，无需经过 exo_manager 代理。
    """
    manager = _get_manager()
    if not manager:
        return []

    models = []
    seen_models = set()

    # ── 第一步：从分配注册表获取首层节点信息（最可靠的数据源）──
    # key: base_model_id, value: (node_id, inference_url)
    first_layer_map = {}
    alloc_registry = getattr(manager, '_allocation_registry', {})
    for base_model_id, alloc_info in alloc_registry.items():
        fl_node_id = alloc_info.get("first_layer_node_id", "")
        fl_url = alloc_info.get("inference_url", "")
        if fl_node_id and fl_url:
            first_layer_map[base_model_id] = (fl_node_id, fl_url)

    # ── 第二步：收集所有已加载的模型列表 ──
    for node_id, connector in getattr(manager, 'connectors', {}).items():
        if not _status_is_online(connector.node_info.status):
            continue
        for model in connector.node_info.loaded_models:
            model_id = model.get("model_id", "unknown")
            if model_id not in seen_models:
                seen_models.add(model_id)
                models.append({
                    "model_id": model_id,
                    "node_id": node_id,
                    "node_url": connector.node_info.chatgpt_url,
                    "type": "worker_instance" if "::" in model_id else "base",
                })

    # Fallback: 从 manager.nodes 补充
    if not models and getattr(manager, 'nodes', None):
        for node_id, node_info in manager.nodes.items():
            if node_info.loaded_models:
                for m in node_info.loaded_models:
                    model_id = m.get("model_id", "unknown")
                    if model_id not in seen_models:
                        seen_models.add(model_id)
                        models.append({
                            "model_id": model_id,
                            "node_id": node_id,
                            "node_url": node_info.chatgpt_url,
                            "type": "worker_instance" if "::" in model_id else "base",
                        })

    # ── 第三步：为每个模型附加上首层推理地址 ──
    for m in models:
        base_id = m["model_id"].split("::")[0] if "::" in m["model_id"] else m["model_id"]
        if base_id in first_layer_map:
            m["inference_url"] = first_layer_map[base_id][1]
            m["first_layer_node_id"] = first_layer_map[base_id][0]
        else:
            # 分配注册表中没有，尝试从 shard 信息推断（gRPC 数据可能不完整）
            m["inference_url"] = ""
            m["first_layer_node_id"] = ""

    return models


def _find_target_node(model_id: str) -> Optional[tuple]:
    """
    找到处理指定模型的目标节点（使用负载均衡器）

    Returns:
        (node_id, node_url, full_model_id) 或 None
        full_model_id 包含 instance_id（如 "Qwen/Qwen3-4B::worker-1"）
    """
    manager = _get_manager()
    if not manager:
        return None

    full_model_id = None  # LoadBalancer 选中的完整 model_id（含 instance）

    try:
        lb = getattr(manager, 'load_balancer', None)
        if lb:
            result = lb.select_instance(
                model_id=model_id,
                strategy=LBStrategy.ROUND_ROBIN
            )
            if result and result.selected_node_id:
                node_id = result.selected_node_id
                connector = manager.connectors.get(node_id)
                if connector:
                    node_url = connector.node_info.chatgpt_url
                    full_model_id = result.selected_instance.full_model_id
                    logger.info(f"[LoadBalancer] ✅ 选择节点: {node_id} (策略=round_robin, 实例={result.selected_instance.instance_id}, full_id={full_model_id})")
                    return (node_id, node_url, full_model_id)
        else:
            logger.warning("[LoadBalancer] ⚠️ 未找到 load_balancer 实例")
    except Exception as e:
        logger.error(f"[LoadBalancer] ❌ 负载均衡失败: {e}, 回退到默认逻辑")

    # Fallback: 遍历所有在线节点的已加载模型
    candidate_nodes = []  # [(node_id, start_layer, url, full_model_id), ...]
    first_layer_nodes = []  # 所有首层节点（多副本场景可能有多个）

    for node_id, connector in manager.connectors.items():
        if not _status_is_online(connector.node_info.status):
            continue
        for m in connector.node_info.loaded_models:
            mid = m.get("model_id", "")
            # 支持基础模型名匹配（如 qwen-3-0.6b 匹配 qwen-3-0.6b::worker-1）
            if mid == model_id or mid.split('::')[0] == model_id:
                shard = m.get("shard", {})
                start_layer = shard.get("start_layer", -1)
                is_first_layer = start_layer == 0 or (start_layer == -1)
                url = connector.node_info.chatgpt_url
                candidate_nodes.append((node_id, start_layer, url, mid))
                if is_first_layer:
                    first_layer_nodes.append((node_id, url, mid))
                break

    # 多副本负载均衡：在多个首层节点间轮询
    if len(first_layer_nodes) > 1:
        import random
        node_id, url, fmid = random.choice(first_layer_nodes)  # 简单随机，可后续升级为轮询
        logger.info(f"[LoadBalancer-Fallback] 🔄 多副本模式: {len(first_layer_nodes)} 个首层节点, 选择 {node_id} (full_id={fmid})")
        return (node_id, url, fmid)
    elif first_layer_nodes:
        return first_layer_nodes[0]
    elif candidate_nodes:
        return candidate_nodes[0]

    return None


async def _proxy_to_node(
    request_body: Dict,
    model_id: str,
    user_id: Optional[str],
    request_id: str,
    api_key: str = ""
) -> AsyncGenerator[bytes, None]:
    """
    🎯 核心：智能代理（WebSocket 优先 + HTTP 降级）

    优先级：
    1. WebSocket 隧道（内网穿透场景）
    2. HTTP 直接代理（公网场景）

    将请求转发到 exo 节点，以流式方式返回响应。
    """
    target = _find_target_node(model_id)
    if not target:
        error_data = {
            "error": {
                "message": f"Model '{model_id}' not loaded or no available nodes",
                "type": "model_not_found",
                "code": "model_not_found"
            }
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
        yield b"data: [DONE]\n\n"
        return

    node_id, node_url = target[0], target[1]
    full_model_id = target[2] if len(target) > 2 else None

    # 🔑 关键修复：用 LoadBalancer 选中的 full_model_id（含 instance）覆盖请求体
    # 这样节点端收到的 model 就是 "Qwen/Qwen3-4B::worker-1" 而非 "qwen-3-4b"
    # 避免 instance 不匹配导致重复加载全量模型
    if full_model_id:
        request_body["model"] = full_model_id
        logger.info(f"[Proxy] 🔑 使用完整模型ID转发: {model_id} → {full_model_id}")
    start_time = time.time()

    # 🔍 [诊断] 打印代理目标 URL（排查 ConnectTimeout）
    logger.info(f"🔍 [Proxy-DIAG] 目标节点: {node_id}, URL: {node_url}")
    _mgr = _get_manager()
    if _mgr and node_id in _mgr.nodes:
        _ni = _mgr.nodes[node_id]
        logger.info(f"🔍 [Proxy-DIAG] 节点信息: address={_ni.address}, chatgpt_port={_ni.chatgpt_api_port}, chatgpt_url={_ni.chatgpt_url}")

    total_tokens_used = 0
    
    if _ws_available and node_ws_manager and node_ws_manager.is_node_connected(node_id):
        logger.info(f"[Proxy] 🌐 使用 WebSocket 隧道 → {node_id} req={request_id}")

        # 🔧 [修复] 节点端 WS 推理处理器期望 model_id 字段，补上
        ws_request_data = dict(request_body)
        if "model_id" not in ws_request_data and "model" in ws_request_data:
            ws_request_data["model_id"] = ws_request_data["model"]

        try:
            async for chunk in await node_ws_manager.send_inference_request(
                node_id=node_id,
                request_data=ws_request_data,
                timeout=300.0
            ):
                yield chunk
                
                try:
                    text = chunk.decode('utf-8', errors='ignore')
                    for line in text.split('\n'):
                        if line.startswith('data: ') and '[DONE]' not in line:
                            try:
                                data = json.loads(line[6:])
                                choices = data.get('choices', [])
                                if choices:
                                    delta = choices[0].get('delta', {}).get('content', '')
                                    if delta:
                                        total_tokens_used += len(delta) // 4
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass
                except Exception:
                    pass
            
            latency_ms = (time.time() - start_time) * 1000
            
            manager = _get_manager()
            if manager and hasattr(manager, 'load_balancer'):
                manager.load_balancer.record_completion(
                    node_id=node_id,
                    model_id=model_id,
                    success=True,
                    latency=latency_ms,
                    tokens_generated=total_tokens_used
                )
                logger.info(f"[LoadBalancer] 📊 记录统计: {node_id} 延迟={latency_ms:.1f}ms tokens={total_tokens_used}")
            
            if user_id and total_tokens_used > 0:
                asyncio.create_task(_async_deduct_quota(user_id, total_tokens_used + 100, model_id))

                input_tokens_est = sum(
                    len((m.get("content", "") or "").split()) * 1.3
                    for m in request_body.get("messages", [])
                )
                asyncio.create_task(_record_api_call_stats(
                    user_id=user_id,
                    api_key=api_key,
                    model_id=model_id,
                    endpoint="chat/completions",
                    input_tokens=int(input_tokens_est),
                    output_tokens=total_tokens_used,
                    total_tokens=total_tokens_used + int(input_tokens_est),
                    latency_ms=latency_ms,
                    status="success",
                    error_message="",
                    request_id=request_id
                ))

            logger.info(f"[Proxy] ← WS 完成 req={request_id} tokens≈{total_tokens_used}")
            return
            
        except Exception as e:
            logger.warning(f"[Proxy] WebSocket failed, fallback to HTTP: {e}")
            manager = _get_manager()
            if manager and hasattr(manager, 'load_balancer'):
                manager.load_balancer.record_completion(
                    node_id=node_id,
                    model_id=model_id,
                    success=False,
                    latency=(time.time() - start_time) * 1000
                )

            if user_id:
                asyncio.create_task(_record_api_call_stats(
                    user_id=user_id,
                    api_key=api_key,
                    model_id=model_id,
                    endpoint="chat/completions",
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    latency_ms=(time.time() - start_time) * 1000,
                    status="error",
                    error_message=f"WebSocket failed: {str(e)[:200]}",
                    request_id=request_id
                ))
    
    node_endpoint = f"{node_url}/v1/chat/completions"
    logger.info(f"[Proxy] → HTTP 代理 {node_id} ({node_endpoint}) req={request_id} user={user_id}")

    client = _get_http_client()
    total_tokens_used = 0

    # 🔧 [修复] 当 chatgpt_url 指向不可达地址（如 FRP 远程 IP）时，
    #     自动 fallback 到本地 127.0.0.1（Manager 与节点同机部署场景）
    _mgr_local = _get_manager()
    _local_endpoint = None
    if _mgr_local and node_id in _mgr_local.nodes:
        _ni = _mgr_local.nodes[node_id]
        if _ni.address not in ("127.0.0.1", "localhost", "0.0.0.0"):
            _local_endpoint = f"http://127.0.0.1:{_ni.chatgpt_api_port}/v1/chat/completions"
            logger.info(f"🔧 [Proxy] 检测到远程地址，准备 fallback: {node_endpoint} → {_local_endpoint}")

    try:
        async with client.stream(
            "POST",
            node_endpoint,
            json=request_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        ) as response:

            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"[Proxy] 节点返回错误 {response.status_code}: {error_text[:200]}")

                manager = _get_manager()
                if manager and hasattr(manager, 'load_balancer'):
                    manager.load_balancer.record_completion(
                        node_id=node_id,
                        model_id=model_id,
                        success=False,
                        latency=(time.time() - start_time) * 1000
                    )

                if user_id:
                    asyncio.create_task(_record_api_call_stats(
                        user_id=user_id,
                        api_key=api_key,
                        model_id=model_id,
                        endpoint="chat/completions",
                        input_tokens=0,
                        output_tokens=0,
                        total_tokens=0,
                        latency_ms=(time.time() - start_time) * 1000,
                        status="error",
                        error_message=f"Node error ({response.status_code}): {error_text[:200].decode('utf-8', errors='replace')}",
                        request_id=request_id
                    ))

                error_data = {
                    "error": {
                        "message": f"Node error ({response.status_code}): {error_text[:200].decode('utf-8', errors='replace')}",
                        "type": "upstream_error",
                        "code": str(response.status_code)
                    }
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
                yield b"data: [DONE]\n\n"
                return

            async for chunk in response.aiter_bytes(chunk_size=4096):
                if chunk:
                    yield chunk

                    try:
                        text = chunk.decode('utf-8', errors='ignore')
                        for line in text.split('\n'):
                            if line.startswith('data: ') and '[DONE]' not in line:
                                try:
                                    data = json.loads(line[6:])
                                    choices = data.get('choices', [])
                                    if choices:
                                        delta = choices[0].get('delta', {}).get('content', '')
                                        if delta:
                                            total_tokens_used += len(delta) // 4
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    pass
                    except Exception:
                        pass

            # HTTP 路径：流式结束后扣减额度（之前只有 WS 路径有扣减，HTTP 路径缺失）
            latency_ms = (time.time() - start_time) * 1000
            manager = _get_manager()
            if manager and hasattr(manager, 'load_balancer'):
                manager.load_balancer.record_completion(
                    node_id=node_id,
                    model_id=model_id,
                    success=True,
                    latency=latency_ms,
                    tokens_generated=total_tokens_used
                )
                logger.info(f"[LoadBalancer] 📊 [HTTP] 记录统计: {node_id} 延迟={latency_ms:.1f}ms tokens={total_tokens_used}")

            if user_id and total_tokens_used > 0:
                asyncio.create_task(_async_deduct_quota(user_id, total_tokens_used + 100, model_id))
                input_tokens_est = sum(
                    len((m.get("content", "") or "").split()) * 1.3
                    for m in request_body.get("messages", [])
                )
                asyncio.create_task(_record_api_call_stats(
                    user_id=user_id,
                    api_key=api_key,
                    model_id=model_id,
                    endpoint="chat/completions",
                    input_tokens=int(input_tokens_est),
                    output_tokens=total_tokens_used,
                    total_tokens=total_tokens_used + int(input_tokens_est),
                    latency_ms=latency_ms,
                    status="success",
                    error_message="",
                    request_id=request_id
                ))

            logger.info(f"[Proxy] ← HTTP 完成 req={request_id} tokens≈{total_tokens_used} user={user_id}")
            return

    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        # 🔧 [修复] 远程地址不可达时，fallback 到本地 127.0.0.1
        if _local_endpoint:
            logger.warning(f"🔧 [Proxy] 远程地址 {node_endpoint} 连接失败，尝试 fallback 本地: {_local_endpoint}")
            try:
                async with client.stream(
                    "POST",
                    _local_endpoint,
                    json=request_body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                ) as local_response:
                    if local_response.status_code != 200:
                        error_text = await local_response.aread()
                        logger.error(f"[Proxy] 本地节点返回错误 {local_response.status_code}: {error_text[:200]}")
                        error_data = {
                            "error": {"message": f"Local node error ({local_response.status_code}): {error_text[:200].decode('utf-8', errors='replace')}", "type": "upstream_error"}
                        }
                        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
                        yield b"data: [DONE]\n\n"
                        return

                    async for chunk in local_response.aiter_bytes(chunk_size=4096):
                        if chunk:
                            yield chunk
                            try:
                                text = chunk.decode('utf-8', errors='ignore')
                                for line in text.split('\n'):
                                    if line.startswith('data: ') and '[DONE]' not in line:
                                        try:
                                            data = json.loads(line[6:])
                                            choices = data.get('choices', [])
                                            if choices:
                                                delta = choices[0].get('delta', {}).get('content', '')
                                                if delta:
                                                    total_tokens_used += len(delta) // 4
                                        except (json.JSONDecodeError, KeyError, IndexError):
                                            pass
                            except Exception:
                                pass

                    # HTTP fallback 本地路径：流式结束后扣减额度
                    if user_id and total_tokens_used > 0:
                        asyncio.create_task(_async_deduct_quota(user_id, total_tokens_used + 100, model_id))
                    logger.info(f"🔧 [Proxy] ✅ Fallback 本地成功! req={request_id} tokens≈{total_tokens_used} user={user_id}")
                    return

            except Exception as local_e:
                logger.error(f"🔧 [Proxy] Fallback 本地也失败: {local_e}")

        logger.error(f"[Proxy] 无法连接节点 {node_id}: {e}")

        manager = _get_manager()
        if manager and hasattr(manager, 'load_balancer'):
            manager.load_balancer.record_completion(
                node_id=node_id,
                model_id=model_id,
                success=False,
                latency=(time.time() - start_time) * 1000
            )

        if user_id:
            asyncio.create_task(_record_api_call_stats(
                user_id=user_id,
                api_key=api_key,
                model_id=model_id,
                endpoint="chat/completions",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                status="error",
                error_message=f"Connection error: {str(e)[:200]}",
                request_id=request_id
            ))

        error_data = {
            "error": {"message": f"Cannot connect to inference node: {str(e)}", "type": "connection_error"}
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
        yield b"data: [DONE]\n\n"

    except httpx.ReadTimeout as e:
        logger.error(f"[Proxy] 节点响应超时 {node_id}: {e}")

        manager = _get_manager()
        if manager and hasattr(manager, 'load_balancer'):
            manager.load_balancer.record_completion(
                node_id=node_id,
                model_id=model_id,
                success=False,
                latency=(time.time() - start_time) * 1000
            )

        if user_id:
            asyncio.create_task(_record_api_call_stats(
                user_id=user_id,
                api_key=api_key,
                model_id=model_id,
                endpoint="chat/completions",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                status="error",
                error_message=f"Timeout: {str(e)[:200]}",
                request_id=request_id
            ))

        error_data = {
            "error": {"message": "Inference timeout on upstream node", "type": "timeout_error"}
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
        yield b"data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"[Proxy] 代理错误: {e}", exc_info=True)

        manager = _get_manager()
        if manager and hasattr(manager, 'load_balancer'):
            manager.load_balancer.record_completion(
                node_id=node_id,
                model_id=model_id,
                success=False,
                latency=(time.time() - start_time) * 1000
            )

        if user_id:
            asyncio.create_task(_record_api_call_stats(
                user_id=user_id,
                api_key=api_key,
                model_id=model_id,
                endpoint="chat/completions",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                status="error",
                error_message=f"Proxy error: {str(e)[:200]}",
                request_id=request_id
            ))

        error_data = {
            "error": {"message": f"Proxy error: {str(e)}", "type": "proxy_error"}
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
        yield b"data: [DONE]\n\n"

    latency_ms = (time.time() - start_time) * 1000
    
    manager = _get_manager()
    if manager and hasattr(manager, 'load_balancer'):
        manager.load_balancer.record_completion(
            node_id=node_id,
            model_id=model_id,
            success=True,
            latency=latency_ms,
            tokens_generated=total_tokens_used
        )
        logger.info(f"[LoadBalancer] 📊 记录统计: {node_id} 延迟={latency_ms:.1f}ms tokens={total_tokens_used}")

    if user_id and total_tokens_used > 0:
        asyncio.create_task(_async_deduct_quota(user_id, total_tokens_used + 100, model_id))

        input_tokens_est = sum(
            len((m.get("content", "") or "").split()) * 1.3
            for m in request_body.get("messages", [])
        )
        asyncio.create_task(_record_api_call_stats(
            user_id=user_id,
            api_key=api_key,
            model_id=model_id,
            endpoint="chat/completions",
            input_tokens=int(input_tokens_est),
            output_tokens=total_tokens_used,
            total_tokens=total_tokens_used + int(input_tokens_est),
            latency_ms=latency_ms,
            status="success",
            error_message="",
            request_id=request_id
        ))

    logger.info(f"[Proxy] ← HTTP 完成 req={request_id} tokens≈{total_tokens_used}")


async def _async_deduct_quota(user_id: str, tokens: int, model_id: str):
    """异步扣减额度（不阻塞主流程）"""
    logger.info(f"[Quota] ⏳ 准备扣减: user={user_id} tokens={tokens} model={model_id}")
    try:
        from quota_manager import get_quota_manager
        quota_mgr = get_quota_manager()
        allowed, remaining = quota_mgr.deduct_tokens(user_id, tokens, model_id, "inference")
        # 扣减后失效该用户所有额度缓存，确保下次查询拿到最新数据
        _quota_cache.invalidate_prefix(f"quota:{user_id}:")
        logger.info(f"[Quota] ✅ 扣减完成: user={user_id} tokens={tokens} 剩余={remaining} allowed={allowed}")
    except Exception as e:
        logger.error(f"[Quota] ❌ 扣减失败: user={user_id} error={e}", exc_info=True)


async def _record_api_call_stats(
    user_id: str,
    api_key: str,
    model_id: str,
    endpoint: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    latency_ms: float,
    status: str,
    error_message: str,
    request_id: str
):
    """异步记录 API 调用统计（不阻塞主流程）"""
    try:
        from quota_manager import get_quota_manager
        quota_mgr = get_quota_manager()
        quota_mgr.record_api_call(
            user_id=user_id,
            api_key=api_key,
            model_id=model_id,
            endpoint=endpoint,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
            request_id=request_id
        )
        logger.debug(f"[Stats] 记录API调用: user={user_id} model={model_id} tokens={total_tokens} status={status}")
    except Exception as e:
        logger.warning(f"[Stats] 记录API调用失败: {e}")


# ==================== OpenAI 兼容 API 端点 ====================

@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    auth_info: Tuple[str, Optional[str]] = Depends(verify_api_key)
):
    """
    OpenAI 兼容聊天补全接口 (轻量代理模式)

    工作流程:
    1. 验证 API Key / Session
    2. 检查用户额度
    3. 找到可用的 exo 节点
    4. 流式转发请求到节点的 /v1/chat/completions
    5. 原样返回 OpenAI 格式响应
    6. 异步扣减额度

    Example:
        curl -X POST http://manager:8080/v1/chat/completions \\
          -H "Authorization: Bearer exo_sk_xxxx" \\
          -H "Content-Type: application/json" \\
          -d '{"model": "qwen-3-0.6b", "messages": [{"role": "user", "content": "你好"}], "stream": true}'
    """

    api_key, user_id = auth_info
    logger.info(f"[Chat] 📥 新请求: model={request.model} user={user_id} stream={request.stream}")
    manager = _get_manager()
    if not manager:
        raise HTTPException(status_code=503, detail="Cluster manager not initialized")

    # 额度检查（带 5s TTL 缓存，避免每次请求查 SQLite）
    from quota_manager import get_quota_manager
    quota_mgr = get_quota_manager()

    estimated_tokens = sum(
        len((m.content or "").split()) * 1.3 for m in request.messages
    ) + (request.max_tokens or 512)
    estimated_tokens = int(estimated_tokens) + 100

    if user_id:
        quota_cache_key = f"quota:{user_id}:{estimated_tokens}"
        cached_quota = _quota_cache.get(quota_cache_key)
        if cached_quota is not None:
            # 缓存命中：直接用缓存结果
            allowed, msg, remaining = cached_quota
        else:
            # 缓存未命中：查 SQLite
            allowed, msg, remaining = quota_mgr.check_quota(user_id, estimated_tokens)
            _quota_cache.set(quota_cache_key, (allowed, msg, remaining))

        if not allowed:
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "quota_exceeded",
                    "message": msg,
                    "remaining": remaining,
                }
            )

    # 检查模型可用性（支持基础模型名匹配，如 qwen-3-0.6b 匹配 qwen-3-0.6b::worker-1）
    available_models = _get_available_models_for_inference()
    available_ids = [m["model_id"] for m in available_models]
    available_base_ids = [mid.split('::')[0] for mid in available_ids]
    model_id = request.model

    if model_id not in available_ids and model_id not in available_base_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not loaded. Available: {available_ids}"
        )

    # 检查 API Key 模型权限（session 认证用户跳过此检查）
    key_manager = get_api_key_manager()
    if not api_key.startswith("session:") and not key_manager.check_model_access(api_key, model_id):
        raise HTTPException(status_code=403, detail="API key does not have access to this model")

    # 构建转发请求体（保持 OpenAI 格式）
    request_body = request.model_dump(exclude_none=True)
    request_id = f"proxy_{uuid.uuid4().hex[:12]}"

    # 流式代理
    return StreamingResponse(
        _proxy_to_node(
            request_body=request_body,
            model_id=model_id,
            user_id=user_id,
            request_id=request_id,
            api_key=api_key
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream; charset=utf-8",
            "X-Proxy-Node": _find_target_node(model_id)[0] if _find_target_node(model_id) else "unknown",
        }
    )


@router.post("/completions")
async def completions(
    request: CompletionRequest,
    auth_info: Tuple[str, Optional[str]] = Depends(verify_api_key)
):
    """OpenAI 兼容文本补全接口 (转换为 chat 格式后代理)"""
    api_key, user_id = auth_info
    manager = _get_manager()
    if not manager:
        raise HTTPException(status_code=503, detail="Cluster manager not initialized")

    available_models = _get_available_models_for_inference()
    available_ids = [m["model_id"] for m in available_models]
    available_base_ids = [mid.split('::')[0] for mid in available_ids]
    model_id = request.model

    if model_id not in available_ids and model_id not in available_base_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_id}' is not loaded. Available: {available_ids}"
        )

    key_manager = get_api_key_manager()
    if not api_key.startswith("session:") and not key_manager.check_model_access(api_key, model_id):
        raise HTTPException(status_code=403, detail="API key does not have access to this model")

    prompt = request.prompt
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt else ""

    request_body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": request.stream or False,
        "max_tokens": request.max_tokens or 512,
        "temperature": request.temperature or 0.7,
        "top_p": request.top_p or 0.9,
    }
    request_id = f"proxy_{uuid.uuid4().hex[:12]}"

    return StreamingResponse(
        _proxy_to_node(
            request_body=request_body,
            model_id=model_id,
            user_id=user_id,
            request_id=request_id,
            api_key=api_key
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """OpenAI 兼容模型列表接口

    每个模型附带 inference_url（首层节点推理地址），
    客户端可直接 POST 该地址发起推理，无需经过 exo_manager 代理。
    """
    models = _get_available_models_for_inference()
    data = []
    for m in models:
        model_id = m["model_id"]
        data.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "exo-cluster",
            "permission": [],
            "root": model_id,
            "parent": None,
            # 首层节点直连信息
            "inference_url": m.get("inference_url", ""),
            "first_layer_node_id": m.get("first_layer_node_id", ""),
        })
    return {"object": "list", "data": data}


@router.get("/models/{model_id}")
async def get_model(model_id: str, api_key: str = Depends(verify_api_key)):
    """获取单个模型信息"""
    models = _get_available_models_for_inference()
    available_ids = [m["model_id"] for m in models]

    if model_id not in available_ids:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    # 查找该模型的首层节点信息
    model_info = next((m for m in models if m["model_id"] == model_id), None)

    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "exo-cluster",
        "permission": [],
        "root": model_id,
        "parent": None,
        "inference_url": model_info.get("inference_url", "") if model_info else "",
        "first_layer_node_id": model_info.get("first_layer_node_id", "") if model_info else "",
    }


# ==================== 用户 API Key 接口 (/v1/user/keys) ====================

@router.post("/user/keys")
async def user_create_api_key(
    request: CreateKeyRequest,
    auth_info: Tuple[str, Optional[AuthUser]] = Depends(verify_user_or_admin)
):
    """普通用户创建自己的 API Key"""
    _, user = auth_info
    key_manager = get_api_key_manager()
    new_key = key_manager.generate_key(
        name=request.name,
        description=request.description,
        allowed_models=request.allowed_models
    )
    return {
        "object": "api_key",
        "key": new_key,
        "name": request.name,
        "description": request.description,
        "created_at": int(time.time()),
        "is_active": True,
    }


@router.get("/user/keys")
async def user_list_api_keys(auth_info: Tuple[str, Optional[AuthUser]] = Depends(verify_user_or_admin)):
    """普通用户查看自己的 API Key 列表"""
    _, user = auth_info
    key_manager = get_api_key_manager()
    keys = key_manager.list_keys()
    stats = key_manager.get_stats()
    return {"object": "list", "data": keys, "stats": stats}


@router.delete("/user/keys/{key_id}")
async def user_revoke_api_key(key_id: str, auth_info: Tuple[str, Optional[AuthUser]] = Depends(verify_user_or_admin)):
    """普通用户吊销自己的 API Key"""
    _, user = auth_info
    key_manager = get_api_key_manager()
    success = key_manager.revoke_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"object": "api_key", "deleted": True, "id": key_id}


# ==================== 额度管理接口 ====================

@router.get("/user/quota")
async def user_get_quota(auth_info: Tuple[str, Optional[AuthUser]] = Depends(verify_user_or_admin)):
    """获取当前用户的 Token 额度信息"""
    from quota_manager import get_quota_manager
    _, user = auth_info
    if not user:
        raise HTTPException(status_code=401, detail="需要登录才能查看额度")
    quota_mgr = get_quota_manager()
    quota = quota_mgr.get_quota(user.id)
    return {"success": True, "data": quota.to_dict()}


@router.get("/user/quota/history")
async def user_get_quota_history(
    limit: int = 20,
    auth_info: Tuple[str, Optional[AuthUser]] = Depends(verify_user_or_admin)
):
    """获取当前用户的使用历史"""
    from quota_manager import get_quota_manager
    _, user = auth_info
    if not user:
        raise HTTPException(status_code=401, detail="需要登录才能查看历史")
    quota_mgr = get_quota_manager()
    history = quota_mgr.get_usage_history(user.id, limit)
    return {"success": True, "data": history}


@router.post("/user/quota/add")
async def user_add_quota(
    request_data: AddQuotaRequest,
    auth_info: Tuple[str, Optional[AuthUser]] = Depends(verify_user_or_admin)
):
    """用户增加额度（如看广告奖励）"""
    from quota_manager import get_quota_manager
    _, user = auth_info
    if not user:
        raise HTTPException(status_code=401, detail="需要登录才能操作")
    quota_mgr = get_quota_manager()
    quota = quota_mgr.add_quota(user.id, request_data.amount, request_data.source or "reward")
    return {"success": True, "data": quota.to_dict()}


# ==================== 管理员接口 (/v1/admin/*) ====================

@router.get("/admin/quotas")
async def admin_list_all_quotas(auth_info: Tuple[str, AuthUser] = Depends(verify_admin_only)):
    """获取所有用户的额度列表（管理员）"""
    from quota_manager import get_quota_manager
    quota_mgr = get_quota_manager()
    quotas = quota_mgr.get_all_quotas()
    return {"success": True, "data": [q.to_dict() for q in quotas]}


@router.post("/admin/quota/{user_id}/recharge")
async def admin_recharge_quota(
    user_id: str,
    amount: int,
    reason: str = "",
    auth_info: Tuple[str, AuthUser] = Depends(verify_admin_only)
):
    """为用户充值额度（管理员）"""
    from quota_manager import get_quota_manager
    quota_mgr = get_quota_manager()
    success, new_remaining = quota_mgr.add_quota(user_id, amount, reason)
    if not success:
        raise HTTPException(status_code=400, detail="无效的充值金额")
    return {"success": True, "message": f"已充值 {amount} tokens", "new_remaining": new_remaining}


@router.put("/admin/quota/{user_id}")
async def admin_set_user_quota(
    user_id: str,
    total: int,
    auth_info: Tuple[str, AuthUser] = Depends(verify_admin_only)
):
    """设置用户总额度（管理员）"""
    from quota_manager import get_quota_manager
    quota_mgr = get_quota_manager()
    success = quota_mgr.set_quota(user_id, total)
    if not success:
        raise HTTPException(status_code=400, detail="无效的额度值")
    return {"success": True, "message": f"已设置额度为 {total} tokens"}


@router.post("/admin/quota/{user_id}/reset")
async def admin_reset_user_quota(
    user_id: str,
    auth_info: Tuple[str, AuthUser] = Depends(verify_admin_only)
):
    """重置用户额度到默认值（管理员）"""
    from quota_manager import get_quota_manager
    quota_mgr = get_quota_manager()
    success = quota_mgr.reset_user_quota(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="重置失败")
    return {"success": True, "message": f"已重置用户 {user_id} 的额度"}


@router.post("/admin/keys")
async def admin_create_api_key(
    request: CreateKeyRequest,
    auth_info: Tuple[str, AuthUser] = Depends(verify_admin_only)
):
    """管理员创建 API Key"""
    _, admin_user = auth_info
    key_manager = get_api_key_manager()
    new_key = key_manager.generate_key(
        name=request.name,
        description=request.description or f"Created by admin: {admin_user.nickname}",
        allowed_models=request.allowed_models
    )
    return {"object": "api_key", "key": new_key, "name": request.name, "created_at": int(time.time()), "is_active": True}


@router.get("/admin/keys")
async def admin_list_all_keys(auth_info: Tuple[str, AuthUser] = Depends(verify_admin_only)):
    """管理员查看所有用户的 API Key"""
    key_manager = get_api_key_manager()
    keys = key_manager.list_keys()
    stats = key_manager.get_stats()
    return {"object": "list", "data": keys, "stats": stats}


@router.delete("/admin/keys/{key_id}")
async def admin_revoke_any_key(key_id: str, auth_info: Tuple[str, AuthUser] = Depends(verify_admin_only)):
    """管理员吊销任何用户的 API Key"""
    key_manager = get_api_key_manager()
    success = key_manager.revoke_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"object": "api_key", "deleted": True, "id": key_id}


@router.post("/admin/keys/{key_id}/enable")
async def admin_enable_key(key_id: str, auth_info: Tuple[str, AuthUser] = Depends(verify_admin_only)):
    """管理员启用 API Key"""
    key_manager = get_api_key_manager()
    target_key = None
    for full_key, key_obj in key_manager._keys.items():
        if key_manager._mask_key(full_key) == key_id or full_key == key_id:
            target_key = full_key
            break
    if not target_key:
        raise HTTPException(status_code=404, detail="API key not found")
    success = key_manager.enable_key(target_key)
    if success:
        return {"success": True, "message": "API key enabled"}
    raise HTTPException(status_code=500, detail="Failed to enable API key")


@router.post("/admin/keys/{key_id}/disable")
async def admin_disable_key(key_id: str, auth_info: Tuple[str, AuthUser] = Depends(verify_admin_only)):
    """管理员禁用 API Key"""
    key_manager = get_api_key_manager()
    target_key = None
    for full_key, key_obj in key_manager._keys.items():
        if key_manager._mask_key(full_key) == key_id or full_key == key_id:
            target_key = full_key
            break
    if not target_key:
        raise HTTPException(status_code=404, detail="API key not found")
    success = key_manager.disable_key(target_key)
    if success:
        return {"success": True, "message": "API key disabled"}
    raise HTTPException(status_code=500, detail="Failed to disable API key")


# ==================== API 调用统计接口 ====================

@router.get("/user/api/stats")
async def user_get_api_stats(
    days: int = 30,
    auth_info: Tuple[str, Optional[AuthUser]] = Depends(verify_user_or_admin)
):
    """获取当前用户的 API 调用统计信息"""
    from quota_manager import get_quota_manager
    _, user = auth_info
    if not user:
        raise HTTPException(status_code=401, detail="需要登录才能查看统计")
    quota_mgr = get_quota_manager()
    stats = quota_mgr.get_user_api_stats(user.id, days)
    return {"success": True, "data": stats}


@router.get("/user/api/history")
async def user_get_api_history(
    limit: int = 20,
    offset: int = 0,
    auth_info: Tuple[str, Optional[AuthUser]] = Depends(verify_user_or_admin)
):
    """获取当前用户的 API 调用历史记录"""
    from quota_manager import get_quota_manager
    _, user = auth_info
    if not user:
        raise HTTPException(status_code=401, detail="需要登录才能查看历史")
    quota_mgr = get_quota_manager()
    history = quota_mgr.get_user_api_history(user.id, limit, offset)
    return {"success": True, "data": history}


@router.get("/admin/api/stats")
async def admin_get_all_users_stats(
    days: int = 30,
    auth_info: Tuple[str, AuthUser] = Depends(verify_admin_only)
):
    """获取所有用户的 API 调用统计（管理员）"""
    from quota_manager import get_quota_manager
    quota_mgr = get_quota_manager()
    stats = quota_mgr.get_all_users_api_stats(days)
    return {"success": True, "data": stats}


@router.get("/admin/api/overview")
async def admin_get_system_overview(
    days: int = 30,
    auth_info: Tuple[str, AuthUser] = Depends(verify_admin_only)
):
    """获取系统级别的 API 调用概览（管理员）"""
    from quota_manager import get_quota_manager
    quota_mgr = get_quota_manager()
    overview = quota_mgr.get_system_api_overview(days)
    return {"success": True, "data": overview}
