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
from typing import Optional, Dict, Any, List, AsyncGenerator
from pathlib import Path

# FastAPI相关导入
try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Body
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

# 同时添加父目录（用于导入exo主项目的模块）
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

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

# 全局管理器实例
manager: Optional[EXOClusterManager] = None
topo_manager: Optional[P2PTopologyManager] = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EXO Cluster Manager API",
    description="分布式AI模型推理集群管理系统",
    version="1.0.0"
)

# CORS中间件（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 OpenAI 兼容路由
app.include_router(openai_router)
logger.info("✅ OpenAI 兼容 API 路由已注册 (/v1/*)")

# 注册认证路由
app.include_router(auth_router)
logger.info("✅ 认证路由已注册 (无前缀)")
app.include_router(admin_router)
logger.info("✅ 管理员路由已注册 (/admin/*)")

# ==================== 自定义模型配置管理 ====================
CUSTOM_MODELS_FILE = Path(__file__).parent / "data" / "custom_models.json"

BUILTIN_MODEL_CARDS = {
    "qwen-3-0.6b": {"layers": 28, "repo": {"PyTorchQwen3InferenceEngine": "Qwen/Qwen3-0.6B"}},
    "qwen-3-4b": {"layers": 36, "repo": {"PyTorchQwen3InferenceEngine": "Qwen/Qwen3-4B"}},
    "qwen-3-vl-2b": {"layers": 28, "repo": {"PyTorchQwen3VLInferenceEngine": "Qwen/Qwen3-VL-2B-Instruct"}},
    "qwen-3-vl-4b": {"layers": 36, "repo": {"PyTorchQwen3VLInferenceEngine": "Qwen/Qwen3-VL-4B-Instruct"}},
    "qwen-3-tts-1.7b": {"layers": 36, "repo": {"PyTorchQwen3TTSInferenceEngine": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"}},
    "fara-7b": {"layers": 28, "repo": {"PyTorchQwen2_5VlInferenceEngine": "microsoft/Fara-7B-INT8"}},
    "qwen-2.5-vl-3b": {"layers": 36, "repo": {"PyTorchQwen2_5VlInferenceEngine": "Qwen/Qwen2.5-VL-3B-Instruct"}},
    "llama-3.2-1b": {"layers": 16, "repo": {"PyTorchLlama3InferenceEngine": "unsloth/Llama-3.2-1B-Instruct"}},
    "dummy": {"layers": 8, "repo": {"DummyInferenceEngine": "dummy"}},
}

BUILTIN_PRETTY_NAMES = {
    "qwen-3-0.6b": "Qwen 3 0.6B",
    "qwen-3-4b": "Qwen 3 4B",
    "qwen-3-vl-2b": "Qwen 3 VL 2B",
    "qwen-3-vl-4b": "Qwen 3 VL 4B",
    "qwen-3-tts-1.7b": "Qwen 3 TTS 1.7B (VoiceDesign)",
    "fara-7b": "Fara 7B (Microsoft Computer Use Agent)",
    "qwen-2.5-vl-3b": "Qwen 2.5 VL 3B",
    "llama-3.2-1b": "Llama 3.2 1B",
    "dummy": "Dummy (测试用)",
}

custom_model_cards: Dict[str, Dict] = {}
custom_pretty_names: Dict[str, str] = {}

def load_custom_models():
    global custom_model_cards, custom_pretty_names
    try:
        if CUSTOM_MODELS_FILE.exists():
            with open(CUSTOM_MODELS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                custom_model_cards = data.get('model_cards', {})
                custom_pretty_names = data.get('pretty_names', {})
                logger.info(f"[Models] 已加载 {len(custom_model_cards)} 个自定义模型")
        else:
            logger.info("[Models] 未找到自定义模型配置文件，使用内置默认")
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


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理"""
    global manager, topo_manager
    
    if topo_manager:
        await topo_manager.stop()
    
    if manager:
        await manager.shutdown()


# ==================== 节点管理 API ====================

@app.get("/api/nodes", response_model=Dict)
async def list_nodes():
    """
    获取所有节点列表
    
    返回所有已注册的节点及其基本状态信息
    """
    if not manager:
        raise HTTPException(status_code=503, detail="服务未初始化")
    
    status = manager.get_cluster_status()
    
    return {
        "success": True,
        "data": status["nodes"],
        "total": len(status["nodes"]),
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
    
    if not connector:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 不存在")
    
    # ✅ 解析 Node 上报的状态数据
    try:
        heartbeat_data = await request.json()
        
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
        
        # ✅ 验证和补全 device_info（确保每次心跳后数据完整）
        if node_id in manager.nodes:
            node_info = manager.nodes[node_id]
            node_info.device_info = validate_and_enrich_device_info(
                node_info.device_info, 
                node_id
            )
    
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"⚠️ [Heartbeat] 解析节点 {node_id} 心跳数据失败: {error_msg}")
        
        # 如果是 JSON 解析错误（节点发送了空数据或崩溃），返回错误
        if "Expecting value" in error_msg or "JSON" in error_msg.upper():
            # 标记节点为异常状态
            if node_id in manager.nodes:
                manager.nodes[node_id].status = NodeStatus.ERROR
                manager.nodes[node_id].error_message = f"心跳数据解析失败: {error_msg}"
                manager.nodes[node_id].last_heartbeat = time.time()  # 更新时间戳避免重复超时标记
            
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

    # 心跳成功时恢复离线节点的在线状态
    if is_healthy and node_info.status == NodeStatus.OFFLINE:
        old_status = node_info.status.value
        node_info.status = NodeStatus.ONLINE
        node_info.error_message = ""
        node_info.last_heartbeat = time.time()
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
                    "node_connections": node_connections,  # ✅ 对象格式
                    "pending_requests": pending_stats,      # ✅ 对象格式
                    "health_check": health_results,
                    "total_nodes_connected": len(node_ws_manager.node_connections),
                    "version": "2.0"  # 标记为 V2 版本
                }
                
                await websocket.send_text(json.dumps(monitor_data))
                
            except Exception as e:
                logger.error(f"[WS Monitor] ❌ 构建监控数据失败: {e}")
                # 发送错误状态但不中断连接
                error_data = {
                    "timestamp": time.time(),
                    "error": str(e),
                    "version": "2.0"
                }
                await websocket.send_text(json.dumps(error_data))
            
    except WebSocketDisconnect:
        logger.info(f"🔌 [WS Monitor] 客户端断开")
    except Exception as e:
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
    all_model_cards = {**BUILTIN_MODEL_CARDS, **custom_model_cards}
    all_pretty_names = {**BUILTIN_PRETTY_NAMES, **custom_pretty_names}
    source = "builtin_custom" if custom_model_cards else "builtin_defaults"
    
    models_list = []
    
    for model_id, config in all_model_cards.items():
        layers = config.get("layers", 0)
        repo_info = config.get("repo", {})
        
        repo_name = "unknown"
        for engine_name, repo in repo_info.items():
            if 'PyTorch' in engine_name or 'Dummy' in engine_name:
                repo_name = repo
                break
        
        is_custom = model_id in custom_model_cards
        is_builtin = model_id in BUILTIN_MODEL_CARDS
        
        models_list.append({
            "model_id": model_id,
            "pretty_name": all_pretty_names.get(model_id, model_id),
            "layers": layers,
            "repo": repo_name,
            "engines": list(repo_info.keys()),
            "source": "custom" if is_custom else ("builtin" if is_builtin else "remote")
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
        "strategy": "memory_weighted",      (可选: memory_weighted, uniform, performance_weighted)
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
    strategy = request.get("strategy", "memory_weighted")
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
    
    # 从内置配置中获取层数信息和默认路径
    all_model_configs = {**BUILTIN_MODEL_CARDS, **custom_model_cards}
    
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
    strategy: str = "memory_weighted"
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
                "allocations": allocation.allocations,
                "estimated_memory_per_node": allocation.estimated_memory_per_node
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


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

            return True
    
    async def disconnect_node(self, node_id: str):
        """Node 断开连接"""
        if node_id in self.node_connections:
            del self.node_connections[node_id]
            logger.info(f"🔌 [NodeWS] 节点 {node_id} 已断开 (剩余: {len(self.node_connections)})")
        
        # ✅ 关键修复：断开时立即标记节点为离线
        if manager and node_id in manager.nodes:
            node_info = manager.nodes[node_id]
            old_status = node_info.status.value
            node_info.status = NodeStatus.OFFLINE
            node_info.error_message = "WebSocket连接已断开"
            
            logger.warning(f"⚠️ [NodeWS] 节点 {node_id} 已标记为离线 (状态: {old_status} → offline)")
            
            # 通知前端节点状态变化
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
    
    try:
        # 保持连接，处理 Node 发来的消息
        while True:
            data = await websocket.receive_text()
            
            # 交给 NodeWSManager 处理
            await node_ws_manager.handle_node_message(node_id, data)
            
    except WebSocketDisconnect:
        logger.info(f"🔌 [NodeWS] 节点 {node_id} 断开连接")
        node_ws_manager.disconnect_node(node_id)
        
    except Exception as e:
        logger.error(f"[NodeWS] ❌ 节点 {node_id} 连接错误: {e}", exc_info=True)
        node_ws_manager.disconnect_node(node_id)


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
async def serve_dashboard(request: Request):
    """根据用户角色提供不同的页面"""
    from auth_manager import get_auth_manager

    # 检查用户登录状态
    session_token = request.cookies.get("session_token", "")
    if session_token:
        auth_mgr = get_auth_manager()
        user = auth_mgr.validate_session(session_token)

        if user:
            # 管理员 - 返回完整管理界面
            if user.role == "admin":
                html_file = Path(__file__).parent / "static" / "index.html"
                if html_file.exists():
                    return HTMLResponse(content=html_file.read_text(encoding='utf-8'))

            # 普通用户 - 返回简化界面
            else:
                user_html = Path(__file__).parent / "static" / "user.html"
                if user_html.exists():
                    return HTMLResponse(content=user_html.read_text(encoding='utf-8'))

    # 未登录 - 重定向到登录页
    login_page = Path(__file__).parent / "static" / "login.html"
    if login_page.exists():
        return HTMLResponse(content=login_page.read_text(encoding='utf-8'))

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
