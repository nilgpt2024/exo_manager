"""
EXO Cluster Manager - 核心服务
==============================

独立的EXO集群管理系统，提供：
- RESTful API用于节点和模型管理
- Web界面用于可视化操作
- 实时监控和GPU池统一管理

启动方式:
    python -m exo_manager.server --port 8080
    
访问:
    http://localhost:8080

架构说明:
---------
1. 通过读取network_config.json或手动添加来连接EXO节点
2. 使用gRPC协议与各节点通信
3. 提供REST API供前端调用
4. 支持WebSocket实时推送状态更新
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

# 导入 Node WebSocket 管理器（用于实时推送任务）
# ✨ 使用延迟导入避免循环依赖问题
_node_ws_manager_instance = None
_ws_import_attempted = False

def _get_node_ws_manager():
    """延迟获取 node_ws_manager 实例"""
    global _node_ws_manager_instance, _ws_import_attempted
    
    if _node_ws_manager_instance is not None:
        return _node_ws_manager_instance
    
    if _ws_import_attempted:
        return None  # 已经尝试过并失败，不再重试
    
    _ws_import_attempted = True
    try:
        from server import node_ws_manager
        _node_ws_manager_instance = node_ws_manager
        logger.info("✅ [WS] node_ws_manager 导入成功")
        return _node_ws_manager_instance
    except ImportError as e:
        logger.warning(f"⚠️ [WS] node_ws_manager 导入失败: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ [WS] node_ws_manager 导入异常: {e}")
        return None

def _is_ws_available():
    """检查 WS 是否可用"""
    return _get_node_ws_manager() is not None

try:
    import numpy as np
except ImportError:
    np = None

# 确保当前文件所在目录在sys.path中（支持 -m 模式和直接运行）
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 确保本地 gRPC proto 目录在 sys.path 中
_grpc_dir = os.path.join(_current_dir, 'grpc')
if _grpc_dir not in sys.path:
    sys.path.insert(0, _grpc_dir)

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """节点连接状态"""
    ONLINE = "online"
    OFFLINE = "offline"
    CONNECTING = "connecting"
    ERROR = "error"


@dataclass
class EXONodeInfo:
    """EXO节点的信息（从Manager视角）"""
    node_id: str
    address: str
    port: int  # gRPC 端口
    chatgpt_api_port: int = 52415  # ChatGPT API 端口（HTTP）
    status: NodeStatus = NodeStatus.CONNECTING
    device_info: Dict[str, Any] = field(default_factory=dict)
    loaded_models: List[Dict[str, Any]] = field(default_factory=list)
    last_heartbeat: float = 0
    error_message: str = ""
    response_time_ms: float = 0
    
    @property
    def chatgpt_url(self) -> str:
        return f"http://{self.address}:{self.chatgpt_api_port}"
    
    @property
    def chat_completions_url(self) -> str:
        return f"{self.chatgpt_url}/v1/chat/completions"
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "address": self.address,
            "port": self.port,
            "chatgpt_api_port": self.chatgpt_api_port,
            "status": self.status.value,
            "device_info": self.device_info,
            "loaded_models": self.loaded_models,
            "last_heartbeat": self.last_heartbeat,
            "error_message": self.error_message,
            "response_time_ms": round(self.response_time_ms, 2)
        }


class NodeConnector:
    """
    节点连接器 - 管理与单个EXO节点的gRPC连接
    
    负责建立连接、健康检查、收集信息等
    """
    
    def __init__(self, node_info: EXONodeInfo, manager=None):
        self.node_info = node_info
        self.channel = None
        self.stub = None
        self._lock = asyncio.Lock()
        self._manager_ref = manager  # 用于备用数据源查询
        
    async def connect(self) -> bool:
        """建立gRPC连接并获取设备信息（含轻量安全检查）"""
        try:
            import grpc

            # 检查 grpc.aio 是否可用（旧版 grpcio 不支持异步）
            if not hasattr(grpc, 'aio'):
                logger.warning(f"⚠️ grpc.aio 不可用 (grpcio版本过旧)，跳过gRPC连接，依赖WebSocket通信")
                logger.warning(f"   请升级: pip install 'grpcio>=1.60.0'")
                self.node_info.status = NodeStatus.ONLINE  # WebSocket已通，标记为在线
                return False

            address = f"{self.node_info.address}:{self.node_info.port}"
            logger.info(f"🔌 正在连接节点 {self.node_info.node_id} @ {address}...")

            # ==================== 轻量安全检查 (~0.02ms) ====================
            try:
                from node_security import get_security_manager
                security = get_security_manager()

                # 快速检查 (仅黑白名单+封禁+速率限制)
                result = await security.check(
                    ip=self.node_info.address,
                    node_id=self.node_info.node_id
                )

                if not result.allowed:
                    self.node_info.status = NodeStatus.ERROR
                    self.node_info.error_message = f"安全检查未通过: {result.reason}"
                    logger.warning(f"🛡️ 节点 {self.node_info.node_id} 连接被拒绝: {result.reason}")
                    return False

            except ImportError:
                pass  # 安全模块可选, 不影响主流程
            except Exception as e:
                logger.debug(f"安全检查异常(允许): {e}")

            # ==================== 建立gRPC连接 ====================
            self.channel = grpc.aio.insecure_channel(
                address,
                options=[
                    ("grpc.max_receive_message_length", 256 * 1024 * 1024),
                    ("grpc.max_send_message_length", 256 * 1024 * 1024),
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 60000),
                ]
            )
            
            # 尝试导入gRPC stub
            try:
                from proto.node_service_pb2_grpc import NodeServiceStub
                from proto.node_service_pb2 import CollectTopologyRequest
                self.stub = NodeServiceStub(self.channel)
            except ImportError:
                logger.warning(f"无法导入gRPC stub，使用简化模式")
                return await self._connect_simple()
            
            # 等待连接就绪
            await asyncio.wait_for(self.channel.channel_ready(), timeout=10.0)
            
            self.node_info.status = NodeStatus.ONLINE
            self.node_info.last_heartbeat = time.time()
            logger.info(f"✅ 成功连接到节点 {self.node_info.node_id}@{address}")
            
            # 获取节点设备信息（非关键操作，失败不影响连接）
            try:
                await self._fetch_device_info()
            except Exception as e:
                logger.warning(f"⚠️ 获取节点 {self.node_info.node_id} 设备信息失败: {e}")
            
            return True
            
        except Exception as e:
            self.node_info.status = NodeStatus.ERROR
            self.node_info.error_message = str(e)
            import traceback
            logger.error(f"❌ 连接节点 {self.node_info.node_id} 失败: {type(e).__name__}: {e}")
            if "refused" in str(e).lower() or "connection" in str(e).lower():
                logger.error(f"   请确认: 1) 节点正在运行 2) 端口 {self.port} 正确 3) 防火墙允许连接")
            elif "timeout" in str(e).lower():
                logger.error(f"   连接超时，请检查网络连通性")
            logger.debug(f"   详细堆栈: {traceback.format_exc()}")
            return False
    
    async def _fetch_device_info(self):
        """通过CollectTopology获取节点的设备信息"""
        if not self.stub:
            return
        
        try:
            import sys
            from proto.node_service_pb2 import CollectTopologyRequest
            
            request = CollectTopologyRequest(max_depth=0, visited=[])
            response = await asyncio.wait_for(
                self.stub.CollectTopology(request),
                timeout=10.0
            )
            
            # 解析拓扑响应，提取当前节点的设备信息
            device_caps = None
            actual_node_id = None
            if response.nodes and self.node_info.node_id in response.nodes:
                device_caps = response.nodes[self.node_info.node_id]
                actual_node_id = self.node_info.node_id
            elif response.nodes:
                # 记录节点自己上报的真实 ID（与 Manager 配置的 node_id 可能不同，如后缀 _a1）
                first_node_id = list(response.nodes.keys())[0]
                device_caps = response.nodes[first_node_id]
                actual_node_id = first_node_id
                logger.warning(f"⚠️ [DEBUG] 未精确匹配，使用第一个节点: {first_node_id}")
            
            if device_caps:
                # 获取最新的显存数据（来自 pynvml 实时数据）
                new_memory_detail = None
                if device_caps.memory_detail and hasattr(device_caps.memory_detail, 'total'):
                    new_memory_detail = {
                        "total": device_caps.memory_detail.total,
                        "free": device_caps.memory_detail.free,
                        "used": device_caps.memory_detail.used,
                    }
                
                # memory fallback: 静态值为0但pynvml有数据时使用 realtime total
                effective_memory = device_caps.memory
                if (effective_memory == 0 or effective_memory is None) and new_memory_detail and new_memory_detail.get("total", 0) > 0:
                    effective_memory = new_memory_detail["total"]
                    logger.info(f"   [Fallback] memory=0, 使用 pynvml total={effective_memory}MB")
                
                # chip fallback
                raw_chip = device_caps.chip or ""
                if raw_chip in ("Unknown Chip", "unknown", "Unknown", ""):
                    if new_memory_detail and new_memory_detail.get("total", 0) > 100:
                        raw_chip = "GPU"
                    else:
                        raw_chip = "Unknown"
                
                # 构建新的设备信息（只更新基础字段）
                new_device_info = {
                    "model": device_caps.model or "Unknown",
                    "chip": raw_chip,
                    "memory": effective_memory,
                    "flops": {
                        "fp32": device_caps.flops.fp32,
                        "fp16": device_caps.flops.fp16,
                        "int8": device_caps.flops.int8,
                    } if device_caps.flops else {},
                    "actual_node_id": actual_node_id,
                }
                
                # 保留已有的 loaded_models 和其他动态数据
                preserved_data = {}
                if self.node_info.device_info:
                    # 保留已加载模型列表（由 _fetch_loaded_models 更新）
                    if self.node_info.device_info.get('loaded_models'):
                        preserved_data['loaded_models'] = self.node_info.device_info['loaded_models']
                
                # 合并数据：基础信息 + 保留的动态数据 + 最新显存
                self.node_info.device_info = {**new_device_info, **preserved_data}
                
                # 更新显存数据（如果获取到了新的）
                if new_memory_detail:
                    self.node_info.device_info['memory_detail'] = new_memory_detail
                elif 'memory_detail' not in self.node_info.device_info:
                    # 如果没有新数据且之前也没有，设为 None
                    self.node_info.device_info['memory_detail'] = None
                
                logger.info(f"📊 获取到节点 {self.node_info.node_id} 设备信息: "
                           f"{self.node_info.device_info.get('chip')} / "
                           f"{self.node_info.device_info.get('memory')}MB")
                if new_memory_detail:
                    logger.info(f"   显存: {new_memory_detail['used']}/{new_memory_detail['total']} MB")
                    logger.info(f"📊 获取到节点设备信息 (fallback): {self.node_info.device_info}")
                    
        except Exception as e:
            logger.warning(f"⚠️ 获取节点 {self.node_info.node_id} 设备信息失败: {e}")
            # ✅ 保留原有数据，只标记错误（不覆盖完整信息）
            if not self.node_info.device_info:
                self.node_info.device_info = {}
            # 只在原有数据不完整时才添加 error 标记
            required_fields = ["model", "chip", "memory"]
            has_basic_info = all(field in self.node_info.device_info for field in required_fields)
            if not has_basic_info:
                self.node_info.device_info["error"] = str(e)
                self.node_info.device_info["model"] = "Unknown Device"
                self.node_info.device_info["chip"] = "Unknown Chip"
                self.node_info.device_info["memory"] = 0
                self.node_info.device_info["flops"] = {"fp32": 0, "fp16": 0, "int8": 0}
            logger.debug(f"   使用基础/缓存的设备信息: {self.node_info.device_info}")
    
    async def _connect_simple(self) -> bool:
        """简化的连接模式（当gRPC stub不可用时）"""
        try:
            if self.channel:
                await asyncio.wait_for(self.channel.channel_ready(), timeout=5.0)
                self.node_info.status = NodeStatus.ONLINE
                return True
        except:
            pass
        
        # 即使无法完全验证，也标记为在线（假设端口可达）
        self.node_info.status = NodeStatus.ONLINE
        return True
    
    async def _fetch_loaded_models(self):
        """从节点获取已加载的模型列表和实时GPU显存"""
        if not self.stub:
            return
        
        try:
            import sys
            from proto.node_service_pb2 import CollectTopologyRequest
            
            request = CollectTopologyRequest(max_depth=0, visited=[])
            response = await asyncio.wait_for(
                self.stub.CollectTopology(request),
                timeout=10.0
            )
            
            # 调试：打印返回的所有节点 ID
            logger.info(f"🔍 [DEBUG] CollectTopology 返回节点: {list(response.nodes.keys())}")
            logger.info(f"🔍 [DEBUG] 当前 node_info.node_id: {self.node_info.node_id}")
            
            # 从拓扑响应中提取节点状态信息
            if response.nodes:
                # 尝试多种方式匹配节点
                device_caps = None
                matched_node_id = None
                
                # 方式1: 精确匹配 Manager 配置的 node_id
                if self.node_info.node_id in response.nodes:
                    device_caps = response.nodes[self.node_info.node_id]
                    matched_node_id = self.node_info.node_id
                    logger.info(f"✅ [DEBUG] 精确匹配到节点: {self.node_info.node_id}")
                else:
                    # 方式2: 使用 _fetch_device_info 中记录的真实 node_id（解决后缀 _a1 等不匹配）
                    actual_node_id = self.node_info.device_info.get("actual_node_id")
                    if actual_node_id and actual_node_id in response.nodes:
                        device_caps = response.nodes[actual_node_id]
                        matched_node_id = actual_node_id
                        logger.info(f"✅ [DEBUG] 通过 actual_node_id 匹配到节点: {actual_node_id}")
                    else:
                        # 方式3: 尝试后缀/前缀模糊匹配
                        manager_id = self.node_info.node_id
                        for candidate_id in response.nodes.keys():
                            if candidate_id.endswith(manager_id) or manager_id.endswith(candidate_id):
                                device_caps = response.nodes[candidate_id]
                                matched_node_id = candidate_id
                                logger.warning(f"⚠️ [DEBUG] 通过模糊匹配到节点: {candidate_id}")
                                break
                        
                        if device_caps is None:
                            logger.error(f"❌ [DEBUG] 无法在 CollectTopology 中定位当前节点 {self.node_info.node_id}，"
                                        f"返回节点: {list(response.nodes.keys())}，跳过本次模型列表更新")
                            return
                
                if device_caps:
                    # 1. 更新显存使用情况
                    if hasattr(device_caps, 'memory_detail') and device_caps.memory_detail:
                        new_total = getattr(device_caps.memory_detail, 'total', 0)
                        new_free = getattr(device_caps.memory_detail, 'free', 0)
                        new_used = getattr(device_caps.memory_detail, 'used', 0)
                        
                        old_memory = self.node_info.device_info.get('memory_detail', {})
                        
                        self.node_info.device_info['memory_detail'] = {
                            "total": new_total,
                            "free": new_free,
                            "used": new_used,
                        }
                        
                        logger.info(f"📊 [显存更新] 节点 {self.node_info.node_id}: "
                                   f"{new_used}/{new_total} MB "
                                   f"(旧值: {old_memory.get('used', '?')}/{old_memory.get('total', '?')} MB)")
                    
                    # 2. 更新已加载模型列表（智能合并不覆盖）
                    # 🔧 修复：即使 loaded_models 为空数组也要处理，避免数据不一致
                    
                    # 🔍 调试：打印 gRPC 返回的原始 loaded_models 数据
                    raw_loaded_models = getattr(device_caps, 'loaded_models', None)
                    logger.info(f"🔍 [DEBUG] gRPC 返回 loaded_models: "
                               f"类型={type(raw_loaded_models)}, "
                               f"长度={len(raw_loaded_models) if raw_loaded_models else 0}, "
                               f"值={raw_loaded_models if raw_loaded_models else 'None/空'}")
                    
                    if raw_loaded_models is not None:
                        new_loaded_models = []
                        
                        # 从节点获取的模型ID集合（用于去重）
                        node_model_ids = set()
                        seen_new_model_ids = set()
                        
                        for model_info in raw_loaded_models:
                            if model_info.model_id in seen_new_model_ids:
                                logger.debug(f"⚠️ 跳过重复模型: {model_info.model_id}")
                                continue
                            seen_new_model_ids.add(model_info.model_id)
                            node_model_ids.add(model_info.model_id)
                            model_entry = {
                                "model_id": model_info.model_id,
                                "shard": {
                                    "start_layer": model_info.start_layer,
                                    "end_layer": model_info.end_layer,
                                    "n_layers": model_info.n_layers,
                                }
                            }
                            new_loaded_models.append(model_entry)
                        
                        # ✅ 修复：智能合并时避免重复（基于 model_id 去重）
                        local_multi_instance_models = []
                        seen_multi_ids = set()
                        
        # 获取新数据中的所有 model_id（用于判断是否需要补充）
                        new_model_ids_set = set(m["model_id"] for m in new_loaded_models)
                        
                        for existing_model in self.node_info.loaded_models:
                            existing_id = existing_model.get("model_id", "")
                            
                            if "::" in existing_id:
                                # ✅ 关键修复：只在本地记录不在新数据中时才补充
                                if existing_id in seen_multi_ids or existing_id in new_model_ids_set:
                                    logger.debug(f"⏭️ 跳过已存在的多实例: {existing_id}")
                                    continue
                                    
                                seen_multi_ids.add(existing_id)
                                base_id = existing_id.split("::")[0]
                                
                                # 只在基础模型匹配且实例ID确实不存在时才保留
                                if base_id in node_model_ids or any(
                                    nm.startswith(base_id) for nm in node_model_ids
                                ):
                                    local_multi_instance_models.append(existing_model)
                                    logger.debug(f"🔒 补充本地多实例记录: {existing_id}")
                        
                        
                        # 合并：节点的数据 + 本地的多实例记录
                        merged_models = new_loaded_models + local_multi_instance_models
                        
                        old_models = self.node_info.loaded_models
                        self.node_info.loaded_models = merged_models
                        
                        all_model_ids = [m["model_id"] for m in merged_models]
                        logger.info(f"📦 [模型更新] 节点 {self.node_info.node_id}: "
                                   f"加载了 {len(merged_models)} 个模型 - {all_model_ids} "
                                   f"(旧值: {[m.get('model_id','?') for m in old_models]})")
                        
                        if len(local_multi_instance_models) > 0:
                            logger.info(f"   ✅ 保留了 {len(local_multi_instance_models)} 个本地多实例记录")
                    else:
                        # gRPC 未返回 loaded_models 字段（保持原有数据不变）
                        logger.warning(f"⚠️ [模型更新] gRPC 未返回 loaded_models 字段，保持原有数据")
                    
                    # 🔧 新增：智能推断 - 当 loaded_models 为空但显存被占用时
                    if len(self.node_info.loaded_models) == 0:
                        mem_detail = self.node_info.device_info.get('memory_detail', {})
                        used_mem = mem_detail.get('used', 0)
                        
                        if used_mem > 500:  # > 500MB 说明有模型在运行
                            logger.warning(
                                f"⚠️ [数据不一致] 节点 {self.node_info.node_id}: "
                                f"loaded_models 为空，但显存使用 {used_mem}MB！"
                                f"\n   可能原因:"
                                f"\n   1. 模型通过非 Manager 途径加载（如命令行直接启动）"
                                f"\n   2. exo 节点的 my_loaded_models 未正确更新"
                                f"\n   3. gRPC CollectTopology 未包含完整模型信息"
                                f"\n\n   💡 建议检查 exo 节点日志中的 'on_model_loaded' 调用"
                            )
                            
                            # 尝试从 manager 的全局模型注册表补充
                            if hasattr(self, '_manager_ref') and self._manager_ref:
                                try:
                                    global_models = getattr(self._manager_ref, '_global_model_registry', None)
                                    if global_models:
                                        node_models = []
                                        for model_id, model_info in global_models.items():
                                            if model_info.get('node_id') == self.node_info.node_id:
                                                node_models.append({
                                                    "model_id": model_id,
                                                    "shard": model_info.get('shard', {}),
                                                    "source": "global_registry_fallback"
                                                })
                                        
                                        if node_models:
                                            self.node_info.loaded_models = node_models
                                            logger.info(
                                                f"✅ [备用数据源] 从全局注册表补充了 "
                                                f"{len(node_models)} 个模型: "
                                                f"{[m['model_id'] for m in node_models]}"
                                            )
                                except Exception as e:
                                    logger.debug(f"备用数据源查询失败: {e}")
            
        except Exception as e:
            logger.error(f"❌ 获取节点 {self.node_info.node_id} 模型信息失败: {e}")
    
    async def disconnect(self):
        """断开连接"""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
        self.node_info.status = NodeStatus.OFFLINE
        logger.info(f"🔌 已断开节点 {self.node_info.node_id}")
    
    async def health_check(self) -> bool:
        """执行健康检查"""
        start_time = time.time()

        try:
            if not self.stub:
                # WS模式节点：通过WebSocket连接状态判断健康
                if _is_ws_available():
                    _ws_mgr = _get_node_ws_manager()
                    if _ws_mgr and _ws_mgr.is_node_connected(self.node_info.node_id):
                        self.node_info.response_time_ms = (time.time() - start_time) * 1000
                        self.node_info.status = NodeStatus.ONLINE
                        self.node_info.error_message = ""
                        return True
                return False
            
            # 使用正确的 protobuf 请求类型
            try:
                from proto.node_service_pb2 import HealthCheckRequest
                request = HealthCheckRequest()
            except ImportError:
                # 如果无法导入，使用空字典作为fallback（某些gRPC版本支持）
                request = {}
            
            response = await asyncio.wait_for(
                self.stub.HealthCheck(request),
                timeout=5.0
            )
            
            self.node_info.response_time_ms = (time.time() - start_time) * 1000
            self.node_info.last_heartbeat = time.time()
            self.node_info.status = NodeStatus.ONLINE
            self.node_info.error_message = ""
            
            return True
            
        except Exception as e:
            self.node_info.response_time_ms = (time.time() - start_time) * 1000
            self.node_info.status = NodeStatus.ERROR
            self.node_info.error_message = f"Health check failed: {str(e)}"
            return False
    
    async def get_status(self) -> Optional[Dict]:
        """获取节点详细状态"""
        if not self.stub or self.node_info.status != NodeStatus.ONLINE:
            return None
        
        try:
            # 这里可以扩展为调用实际的gRPC方法获取详细信息
            # 目前返回基本信息
            return {
                "node_id": self.node_info.node_id,
                "status": self.node_info.status.value,
                "uptime": time.time() - self.node_info.last_heartbeat if self.node_info.last_heartbeat else 0,
                "response_time_ms": self.node_info.response_time_ms
            }
        except Exception as e:
            logger.error(f"获取节点状态失败: {e}")
            return None
    
    async def send_shard_config(
        self,
        model_id: str,
        model_path: str,
        start_layer: int,
        end_layer: int,
        n_layers: int,
        peer_list: Optional[List[Dict]] = None
    ) -> Dict:
        """
        向节点发送分片配置（加载模型权重）
        
        优先使用 gRPC SendOpaqueStatus 接口，
        如果 gRPC 不可用（如 FRP 代理场景），自动 fallback 到 HTTP API。
        
        Args:
            model_id: 模型ID
            model_path: 模型路径
            start_layer: 起始层
            end_layer: 结束层
            n_layers: 总层数
            
        Returns:
            发送结果
        """
        
        shard_payload = {
            "model_id": model_id,
            "base_model_id": model_id.split("::")[0] if "::" in model_id else model_id,
            "instance_id": model_id.split("::")[1] if "::" in model_id else "default",
            "model_path": model_path,
            "shard": {
                "start_layer": start_layer,
                "end_layer": end_layer,
                "n_layers": n_layers
            },
            "peer_list": peer_list or [],
            "timestamp": time.time()
        }
        
        logger.info(f"📤 发送分片配置到节点 {self.node_info.node_id}:")
        logger.info(f"   model_id: {model_id}")
        logger.info(f"   model_path: {model_path}")
        logger.info(f"   layers: {start_layer}-{end_layer}/{n_layers}")
        
        # 方式1: 尝试 gRPC（直连模式）
        if self.stub:
            try:
                result = await self._send_shard_via_grpc(shard_payload)
                if result["success"]:
                    return result
                logger.warning(f"gRPC 发送失败，尝试 HTTP fallback...")
            except Exception as e:
                logger.warning(f"gRPC 不可用 ({e})，切换到 HTTP fallback...")
        
        # 方式2: HTTP Fallback（FRP/代理模式）
        try:
            result = await self._send_shard_via_http(shard_payload)
            if result["success"]:
                return result
            logger.error(f"HTTP fallback 也失败")
            return result
        except Exception as e:
            logger.error(f"❌ 所有发送方式均失败 ({self.node_info.node_id}): {e}")
            return {
                "success": False,
                "node_id": self.node_info.node_id,
                "error": f"gRPC 和 HTTP 均不可用: {str(e)}"
            }
    
    async def _send_shard_via_grpc(self, shard_payload: Dict) -> Dict:
        """通过 gRPC SendOpaqueStatus 发送分片配置"""
        try:
            import sys
            from proto.node_service_pb2 import SendOpaqueStatusRequest
            
            shard_config = json.dumps({
                **shard_payload,
                "type": "manager_shard_config",
                "requester": "exo_manager"
            })
            
            request = SendOpaqueStatusRequest(
                request_id=f"load_{shard_payload['model_id']}_{shard_payload['shard']['start_layer']}_{shard_payload['shard']['end_layer']}_{int(time.time())}",
                status=shard_config
            )
            
            response = await asyncio.wait_for(
                self.stub.SendOpaqueStatus(request),
                timeout=30.0
            )
            
            self._update_loaded_models(shard_payload)
            
            logger.info(f"✅ [gRPC] 已向节点 {self.node_info.node_id} 发送分片配置")
            
            return {
                "success": True,
                "node_id": self.node_info.node_id,
                "model_id": shard_payload["model_id"],
                "shard": shard_payload["shard"],
                "method": "grpc",
                "message": "分片配置已发送 (gRPC)"
            }
            
        except Exception as e:
            logger.error(f"❌ [gRPC] 发送分片配置失败: {e}")
            raise
    
    async def _send_shard_via_http(self, shard_payload: Dict) -> Dict:
        """通过 HTTP API 发送分片配置（FRP Fallback）"""
        try:
            import aiohttp
            
            http_url = f"{self.node_info.chatgpt_url}/v1/manager/shard-config"
            
            logger.info(f"📡 [HTTP] 尝试通过 HTTP 发送分片配置到 {http_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    http_url,
                    json=shard_payload,
                    timeout=aiohttp.ClientTimeout(total=30.0)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get("success"):
                            self._update_loaded_models(shard_payload)
                            
                            logger.info(f"✅ [HTTP] 已向节点 {self.node_info.node_id} 发送分片配置")
                            
                            return {
                                "success": True,
                                "node_id": self.node_info.node_id,
                                "model_id": shard_payload["model_id"],
                                "shard": shard_payload["shard"],
                                "method": "http",
                                "message": "分片配置已发送 (HTTP Fallback)"
                            }
                        else:
                            return {
                                "success": False,
                                "node_id": self.node_info.node_id,
                                "error": result.get("error", "节点返回错误"),
                                "method": "http"
                            }
                    else:
                        error_text = await resp.text()
                        return {
                            "success": False,
                            "node_id": self.node_info.node_id,
                            "error": f"HTTP {resp.status}: {error_text[:200]}",
                            "method": "http"
                        }
                        
        except Exception as e:
            logger.error(f"❌ [HTTP] 发送分片配置失败: {e}")
            raise
    
    def _update_loaded_models(self, shard_payload: Dict):
        """更新本地缓存的已加载模型列表"""
        model_id = shard_payload["model_id"]
        base_model_id = shard_payload.get("base_model_id")
        instance_id = shard_payload.get("instance_id")
        if base_model_id is None:
            base_model_id = model_id.split("::")[0] if "::" in model_id else model_id
        if instance_id is None:
            instance_id = model_id.split("::")[1] if "::" in model_id else "default"

        shard_info = {
            "model_id": model_id,
            "base_model_id": base_model_id,
            "instance_id": instance_id,
            "model_path": shard_payload.get("model_path", ""),
            "shard": shard_payload["shard"],
            "loaded_at": time.time()
        }

        existing_models = [m for m in self.node_info.loaded_models
                         if m.get("model_id") != model_id]
        existing_models.append(shard_info)
        self.node_info.loaded_models = existing_models
    
    async def send_unload_command(self, model_id: str, unload_all_instances: bool = False) -> Dict:
        """
        向节点发送卸载模型命令

        通信优先级与 load_model_to_cluster 保持一致：
        1. WebSocket（内网穿透 / NAT 场景）
        2. gRPC SendOpaqueStatus（直连场景）

        Args:
            model_id: 要卸载的模型ID
            unload_all_instances: 是否卸载该模型的所有实例

        Returns:
            卸载结果 {"success": bool, "method": str, "message": str}
        """
        node_id = self.node_info.node_id

        # 1️⃣ 优先：gRPC SendOpaqueStatus（节点 GPU 池原生支持多实例/全部卸载）
        if self.stub:
            try:
                from proto.node_service_pb2 import SendOpaqueStatusRequest

                unload_command = json.dumps({
                    "type": "manager_unload_model",
                    "requester": "exo_manager",
                    "model_id": model_id,
                    "unload_all_instances": unload_all_instances,
                    "timestamp": time.time()
                })

                request = SendOpaqueStatusRequest(
                    request_id=f"unload_{model_id}_{int(time.time())}",
                    status=unload_command
                )

                await asyncio.wait_for(
                    self.stub.SendOpaqueStatus(request),
                    timeout=15.0
                )

                logger.info(f"🗑️ [Unload] 已通过 gRPC 向节点 {node_id} 发送卸载命令: {model_id}")

                return {
                    "success": True,
                    "method": "grpc",
                    "message": f"卸载命令已通过 gRPC 发送给节点 {node_id}"
                }

            except Exception as e:
                logger.warning(f"⚠️ [Unload] gRPC 发送失败，尝试 WebSocket 回退: {e}")

        # 2️⃣ 回退：WebSocket 实时推送（gRPC 不可达或 WS-only 节点）
        try:
            _ws_mgr = _get_node_ws_manager()
            if _is_ws_available() and _ws_mgr and _ws_mgr.is_node_connected(node_id):
                logger.info(f"📡 [Unload] 尝试通过 WebSocket 推送到节点 {node_id}...")

                ws_success = await _ws_mgr.send_model_unload_request(
                    node_id=node_id,
                    model_id=model_id,
                    unload_all_instances=unload_all_instances
                )

                if ws_success:
                    logger.info(f"🗑️ [Unload] 已通过 WebSocket 向节点 {node_id} 发送卸载请求: {model_id} (all={unload_all_instances})")
                    return {
                        "success": True,
                        "method": "websocket",
                        "message": f"卸载请求已通过 WebSocket 发送: {model_id} (all={unload_all_instances})"
                    }
                else:
                    logger.warning(f"⚠️ [Unload] WebSocket 发送失败")
        except Exception as e:
            logger.error(f"❌ [Unload] WebSocket 发送异常: {e}")

        return {"success": False, "error": "gRPC stub 未初始化且 WebSocket 不可用或发送失败"}

    async def send_inference_prompt(
        self,
        prompt: str = None,
        model_id: str = None,
        request_id: str = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stream: bool = True,
        image: Dict[str, str] = None,
        messages: List[Dict] = None,
    ) -> Dict:
        """
        通过节点的 /v1/chat/completions API 发送推理请求（OpenAI 兼容格式）
        
        优先使用 WebSocket 推送（内网穿透场景），回退到 HTTP POST（公网场景）。
        
        Args:
            prompt: 用户输入的文本 (旧版兼容，优先级低于 messages)
            model_id: 模型ID
            request_id: 请求唯一标识
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Top-P 采样
            stream: 是否流式输出
            image: 图片数据 (旧版兼容)
            messages: OpenAI 格式的多轮消息列表 (新版推荐)
            
        Returns:
            推理结果或流式生成器
        """
        target_url = self.node_info.chat_completions_url
        node_id = self.node_info.node_id
        
        if messages:
            payload_messages = messages
            logger.info(f"   使用 messages 数组 ({len(messages)} 条消息)")
        elif prompt is not None:
            if image and image.get("base64"):
                payload_messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or "请描述这张图片"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image.get('mime_type', 'image/png')};base64,{image['base64']}"
                            }
                        }
                    ]
                }]
                logger.info(f"   [Image] 包含图片 ({image.get('mime_type', 'unknown')})")
            else:
                payload_messages = [{"role": "user", "content": prompt}]
        else:
            payload_messages = [{"role": "user", "content": ""}]
        
        logger.info(f"🚀 [Inference] 推理请求:")
        logger.info(f"   节点: {node_id}")
        logger.info(f"   model: {model_id}")
        if len(payload_messages) > 0:
            last_content = str(payload_messages[-1].get("content", ""))
            logger.info(f"   last_msg: {last_content[:80]}{'...' if len(last_content) > 80 else ''}")
        logger.info(f"   stream={stream}, max_tokens={max_tokens}, temp={temperature}")
        
        # 构建请求 payload
        request_data = {
            "model_id": model_id,
            "model": model_id,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "stream": stream,
        }
        
        # ✨ 策略优先级：WebSocket Push > HTTP POST
        # 1️⃣ 优先：WebSocket 实时推送（内网穿透场景 - 最快）
        try:
            # 🔍 详细诊断日志
            _ws_mgr = _get_node_ws_manager()
            logger.info(f"🔍 [Inference] WebSocket 连接诊断:")
            logger.info(f"   _is_ws_available() = {_is_ws_available()}")
            logger.info(f"   node_ws_manager exists = {_ws_mgr is not None}")
            if _ws_mgr:
                logger.info(f"   node_ws_manager.node_connections = {list(_ws_mgr.node_connections.keys())}")
                logger.info(f"   is_node_connected({node_id}) = {_ws_mgr.is_node_connected(node_id)}")
            
            if _is_ws_available() and _ws_mgr and _ws_mgr.is_node_connected(node_id):
                logger.info(f"📡 [Inference] 使用 WebSocket 推送到节点 {node_id}")

                result_data = {
                    "success": True,
                    "node_id": node_id,
                    "request_id": request_id,
                    "model_id": model_id,
                    "stream": stream,
                    "message": "WebSocket 推理已开始",
                }

                full_text = ""
                token_count = 0
                _FIRST_CHUNK_TIMEOUT = 30.0

                try:
                    _ws_iter = _ws_mgr.send_inference_request(
                        node_id=node_id,
                        request_data=request_data,
                        timeout=600.0
                    )
                    _ws_aiter = _ws_iter.__aiter__()

                    # 首块使用短超时：节点无响应时快速回退HTTP
                    try:
                        _first_chunk = await asyncio.wait_for(
                            _ws_aiter.__anext__(), timeout=_FIRST_CHUNK_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"WS 首块超时({_FIRST_CHUNK_TIMEOUT}s), 回退到 HTTP")
                        raise Exception(f"WS首块超时")

                    # 处理首块及后续流
                    chunk_bytes = _first_chunk
                    while True:
                        chunk_str = chunk_bytes.decode("utf-8", errors="replace")
                        if chunk_str.startswith("data: ") and chunk_str.strip() != "data: [DONE]":
                            data_str = chunk_str[6:].strip()
                            if data_str == "[DONE]":
                                logger.info("WS 流结束")
                                break
                            try:
                                event_data = json.loads(data_str)
                                if "error" in event_data:
                                    err = event_data["error"]
                                    logger.error(f"WS 推理错误: {err}")
                                    yield {"success": False, "error": err.get("message", str(err)), "node_id": node_id, "request_id": request_id}
                                    return
                                delta = None
                                if "choices" in event_data and len(event_data["choices"]) > 0:
                                    choice = event_data["choices"][0]
                                    delta = choice.get("delta", {})
                                    if choice.get("finish_reason") in ("stop", "length"):
                                        logger.info("WS 流结束 (finish_reason)")
                                        break
                                if delta and "content" in delta:
                                    content = delta["content"]
                                    if content:
                                        full_text += content
                                        token_count += 1
                                        result_data["text"] = full_text
                                        result_data["token_count"] = token_count
                                        result_data["delta"] = content
                                        yield result_data
                            except json.JSONDecodeError:
                                continue
                        try:
                            chunk_bytes = await _ws_aiter.__anext__()
                        except StopAsyncIteration:
                            break

                    result_data["text"] = full_text
                    result_data["token_count"] = token_count
                    result_data["finished"] = True
                    yield result_data
                    return

                except Exception as ws_e:
                    logger.warning(f"WebSocket 推送失败 ({node_id}): {ws_e}, 回退到 HTTP")

        except Exception as ws_outer_e:
            logger.warning(f"WebSocket 推送路径异常: {ws_outer_e}, 回退到 HTTP")

        # 2️⃣ 回退：HTTP POST（公网直连场景）
        # 纯WS模式节点（chatgpt_api_port=0）无HTTP通道，跳过回退
        if self.node_info.chatgpt_api_port <= 0:
            logger.error(f"[Inference] WS推送失败且节点 {self.node_info.node_id} 无HTTP端口(chatgpt_api_port={self.node_info.chatgpt_api_port})，无法回退")
            yield {"success": False, "error": f"WebSocket推理失败且节点无可用HTTP通道", "node_id": self.node_info.node_id, "request_id": request_id}
            return

        logger.info(f"🌐 [Inference] 回退到 HTTP POST: {target_url}")
        
        try:
            import aiohttp
            
            timeout = aiohttp.ClientTimeout(total=600)
            
            # 诊断：检查 payload 中图片数据
            import json as _json
            payload_str = _json.dumps(request_data, ensure_ascii=False)
            has_image_in_payload = False
            image_data_size = 0
            if messages and len(messages) > 0:
                for msg in messages:
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "image_url":
                                has_image_in_payload = True
                                img_url = item.get("image_url", {}).get("url", "")
                                image_data_size = len(img_url)
                                logger.info(f"   🖼️ 发现图片数据: URL前缀={img_url[:50]}..., 总长度={image_data_size}")
                                # 有图片时自动增加超时时间到 600 秒
                                timeout = aiohttp.ClientTimeout(total=600)
                                logger.info(f"   ⏰ 检测到图片请求，超时已调整为 600 秒")
            
            logger.info(f"   📦 Payload 大小: {len(payload_str)} 字符, 包含图片: {has_image_in_payload}, 图片数据长度: {image_data_size}")
            
            # 诊断：打印 payload 中 messages 的详细结构（不含完整 base64 数据）
            if messages:
                logger.info(f"   📝 Messages 结构:")
                for i, msg in enumerate(messages):
                    content = msg.get("content", "N/A")
                    if isinstance(content, list):
                        content_summary = []
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "image_url":
                                    url = item.get("image_url", {}).get("url", "")
                                    content_summary.append(f"image_url(url长度={len(url)}, 前50字符={url[:50]}...)")
                                else:
                                    content_summary.append(f"{item.get('type', '?')}: {str(item.get('text', item.get('content', '')))[:50]}")
                            else:
                                content_summary.append(str(item)[:50])
                        logger.info(f"      [{i}] role={msg.get('role')}, content={content_summary}")
                    else:
                        logger.info(f"      [{i}] role={msg.get('role')}, content={str(content)[:100]}")
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    target_url,
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"[Inference] HTTP {response.status}: {error_text[:200]}")
                        yield {
                            "success": False,
                            "error": f"节点返回错误 HTTP {response.status}: {error_text[:100]}",
                            "node_id": self.node_info.node_id,
                            "request_id": request_id,
                        }
                        return
                    
                    if stream:
                        logger.info(f"✅ [Inference] 流式响应已建立")
                        
                        result_data = {
                            "success": True,
                            "node_id": self.node_info.node_id,
                            "request_id": request_id,
                            "model_id": model_id,
                            "stream": True,
                            "message": "流式推理已开始",
                        }
                        
                        full_text = ""
                        token_count = 0
                        
                        buffer = ""
                        async for chunk in response.content:
                            if chunk:
                                buffer += chunk.decode('utf-8', errors='replace')
                                
                                while '\n\n' in buffer:
                                    event_str, buffer = buffer.split('\n\n', 1)
                                    event_str = event_str.strip()
                                    
                                    if not event_str or not event_str.startswith('data: '):
                                        continue
                                    
                                    data_str = event_str[6:]  # 去掉 "data: "
                                    if data_str.strip() == '[DONE]':
                                        logger.info(f"🏁 [Inference] 流结束 (SSE [DONE])")
                                        break
                                    
                                    try:
                                        event_data = json.loads(data_str)
                                        
                                        delta = None
                                        if 'choices' in event_data and len(event_data['choices']) > 0:
                                            choice = event_data['choices'][0]
                                            delta = choice.get('delta', {})
                                            finish_reason = choice.get('finish_reason')
                                            
                                            if finish_reason in ('stop', 'length'):
                                                logger.info(f"🏁 [Inference] 流结束 (finish_reason={finish_reason})")
                                                break
                                        
                                        if delta and 'content' in delta:
                                            content = delta['content']
                                            if content:
                                                full_text += content
                                                token_count += 1
                                                
                                                result_data["text"] = full_text
                                                result_data["token_count"] = token_count
                                                result_data["delta"] = content
                                                
                                                yield result_data
                                                
                                    except json.JSONDecodeError as je:
                                        continue
                                else:
                                    continue
                                break
                        
                        result_data["text"] = full_text
                        result_data["token_count"] = token_count
                        result_data["delta"] = None
                        result_data["finished"] = True
                        yield result_data
                        return
                    
                    else:
                        data = await response.json()
                        logger.info(f"[Inference] 非流式响应已收到")
                        
                        text = ""
                        if 'choices' in data and len(data['choices']) > 0:
                            text = data['choices'][0].get('message', {}).get('content', '')
                        
                        yield {
                            "success": True,
                            "node_id": self.node_info.node_id,
                            "request_id": request_id,
                            "model_id": model_id,
                            "stream": False,
                            "text": text,
                            "token_count": len(text) if text else 0,
                            "message": f"推理完成 ({len(text)} 字符)",
                            "raw_response": data,
                        }
                        return
                        
        except asyncio.TimeoutError:
            logger.error(f"[Inference] 超时: {self.node_info.node_id}")
            yield {"success": False, "error": "推理超时 (请检查节点是否正常响应，或增加超时时间)", "node_id": self.node_info.node_id, "request_id": request_id}
            return
        except Exception as e:
            logger.error(f"[Inference] 发送失败 ({self.node_info.node_id}): {e}")
            import traceback
            traceback.print_exc()
            yield {"success": False, "error": str(e), "node_id": self.node_info.node_id, "request_id": request_id}
            return

    async def _decode_tokens(self, tokens: list, model_id: str) -> Optional[str]:
        """
        将 token IDs 解码为文本
        
        尝试多种解码方式：
        1. 直接 Unicode 解码（适用于简单的 token）
        2. 如果有 tokenizer 可用，使用 tokenizer 解码
        
        Args:
            tokens: token ID 列表
            model_id: 模型ID（用于查找tokenizer）
            
        Returns:
            解码后的文本，或 None（如果无法解码）
        """
        if not tokens:
            return None
        
        try:
            # 方法1: 尝试简单的字符解码
            decoded_chars = []
            for t in tokens:
                if isinstance(t, (int, np.integer)):
                    if 32 <= t <= 126:
                        decoded_chars.append(chr(t))
                    elif t > 127:
                        try:
                            decoded_chars.append(chr(t))
                        except (ValueError, OverflowError):
                            decoded_chars.append(f"[{t}]")
                    elif t == 10:
                        decoded_chars.append('\n')
                    else:
                        decoded_chars.append(f"[{t}]")
            
            text = ''.join(decoded_chars)
            
            if text and len([c for c in text if c not in '[]\n']) > len(tokens) * 0.3:
                logger.info(f"   Token 解码成功 (Unicode): {text[:100]}...")
                return text
                
        except Exception as e:
            logger.debug(f"   Unicode 解码失败: {e}")
        
        try:
            # 方法2: 尝试加载 tokenizer 解码
            import sys
            import os
            
            exo_path = os.path.join(os.path.dirname(__file__), '..', 'exo', 'inference')
            if exo_path not in sys.path:
                sys.path.insert(0, exo_path)
            
            try:
                from model_tokenizers import ModelTokenizers
                tokenizers = ModelTokenizers()
                
                import numpy as np
                tokens_array = np.array(tokens, dtype=np.int32)
                text = tokenizers.decode(tokens_array, model_id)
                
                if text and len(text.strip()) > 0:
                    logger.info(f"   Token 解码成功 (Tokenizer): {text[:100]}...")
                    return text
                    
            except ImportError:
                logger.debug("   ModelTokenizers 不可用")
            except Exception as te:
                logger.debug(f"   Tokenizer 解码失败: {te}")
                
        except Exception as e:
            logger.debug(f"   Tokenizer 加载失败: {e}")
        
        return None


class EXOClusterManager:
    """
    EXO集群管理器 - 核心管理逻辑
    
    管理：
    - 所有节点的连接
    - 节点信息的收集与缓存
    - GPU池的统一调配
    - 提供给API层的数据
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.nodes: Dict[str, EXONodeInfo] = {}
        self.connectors: Dict[str, NodeConnector] = {}
        self.config_path = config_path
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._broadcast_callback = None
        self._instance_counter: Dict[str, int] = {}
        self.pending_tasks: Dict[str, List[Dict]] = {}

        # 分配注册表：记录每个模型的分片分配信息
        # { base_model_id: { "allocations": [{node_id, start_layer, end_layer, instance_id}], "first_layer_node_id": str } }
        self._allocation_registry: Dict[str, Dict] = {}

        # 新节点加入回调（由上层注册，用于触发自动重分配）
        self._node_joined_callback: Optional[Callable[[str], Any]] = None

        from load_balancer import LoadBalancer
        self.load_balancer = LoadBalancer(self)
        
        # 统计数据
        self.stats = {
            "total_nodes": 0,
            "online_nodes": 0,
            "total_models": 0,
            "total_memory_gb": 0,
            "last_update": 0
        }
        
    def set_broadcast_callback(self, callback):
        """设置状态更新广播回调（用于WebSocket推送）"""
        self._broadcast_callback = callback

    def set_node_joined_callback(self, callback):
        """设置新节点加入回调（用于触发自动重分配）"""
        self._node_joined_callback = callback

    def _trigger_node_joined_callback(self, node_id: str):
        """调度新节点加入回调"""
        if self._node_joined_callback:
            try:
                asyncio.create_task(self._node_joined_callback(node_id))
            except Exception as e:
                logger.warning(f"⚠️ 新节点加入回调调度失败: {e}")

    def rebuild_allocation_registry(self):
        """从各节点的 loaded_models 重建分配注册表

        用于 exo_manager 重启后恢复首层节点信息。
        扫描所有 connectors/nodes 的已加载模型，识别首层节点并重建 registry。
        """
        old_count = len(self._allocation_registry)
        self._allocation_registry.clear()

        # 按 base_model_id 收集所有实例
        # { base_id: [(node_id, model_id, shard), ...] }
        model_instances: Dict[str, List[Tuple[str, str, Dict]]] = {}

        for node_id, connector in getattr(self, 'connectors', {}).items():
            # ✅ 跳过离线节点，避免注册残留幽灵数据
            node = self.nodes.get(node_id)
            if not node or node.status.value != "online":
                continue

            for model in connector.node_info.loaded_models:
                model_id = model.get("model_id", "")
                base_id = model_id.split("::")[0] if "::" in model_id else model_id
                shard = model.get("shard", {})
                if base_id not in model_instances:
                    model_instances[base_id] = []
                model_instances[base_id].append((node_id, model_id, shard))

        # Fallback: 从 nodes 补充（同样需要检查在线状态）
        if not model_instances and getattr(self, 'nodes', None):
            for node_id, node_info in self.nodes.items():
                # ✅ 只处理在线节点
                if node_info.status.value != "online":
                    continue

                if node_info.loaded_models:
                    for m in node_info.loaded_models:
                        model_id = m.get("model_id", "")
                        base_id = model_id.split("::")[0] if "::" in model_id else model_id
                        shard = m.get("shard", {})
                        if base_id not in model_instances:
                            model_instances[base_id] = []
                        model_instances[base_id].append((node_id, model_id, shard))

        # 为每个模型找首层节点，重建注册表
        for base_id, instances in model_instances.items():
            first_node_ids = []  # 支持多首层节点（多副本场景）
            inference_urls = {}  # {node_id: url}
            alloc_entries = []

            for node_id, model_id, shard in instances:
                start_layer = shard.get("start_layer", -1)
                end_layer = shard.get("end_layer", -1)
                n_layers = shard.get("n_layers", -1)

                alloc_entries.append({
                    "node_id": node_id,
                    "start_layer": start_layer,
                    "end_layer": end_layer,
                    "instance_id": model_id,
                })

                # 识别首层节点：start_layer == 0 或没有分片信息（单节点/不分片多副本）
                if start_layer == 0 or (start_layer == -1 and len(instances) == 1):
                    first_node_ids.append(node_id)

            if first_node_ids and alloc_entries:
                # 构建所有首层节点的推理地址
                for fnid in first_node_ids:
                    node_info = self.nodes.get(fnid)
                    if node_info:
                        inference_urls[fnid] = node_info.chat_completions_url or ""

                self._allocation_registry[base_id] = {
                    "allocations": alloc_entries,
                    "first_layer_node_ids": first_node_ids,       # 多首层节点列表
                    "first_layer_node_id": first_node_ids[0],     # 兼容旧代码（默认取第一个）
                    "inference_urls": inference_urls,             # 多推理地址
                    "inference_url": inference_urls.get(first_node_ids[0], ""),  # 兼容旧代码
                    "full_model_id": base_id,
                    "updated_at": time.time(),
                }

        new_count = len(self._allocation_registry)
        if new_count > 0:
            logger.info(
                f"🔄 [分配注册表重建] 完成: {new_count} 个模型 "
                f"(旧值: {old_count}) | "
                + ", ".join(
                    f"{mid}→{info['first_layer_node_id']}"
                    for mid, info in self._allocation_registry.items()
                )
            )
    
    def _generate_instance_id(self, model_id: str) -> str:
        """
        为模型自动生成唯一的实例ID
        
        规则：worker-1, worker-2, ...
        
        ✨ 增强版：包含重复检测机制，确保生成的ID不会与现有实例冲突
        """
        if model_id not in self._instance_counter:
            self._instance_counter[model_id] = 0
        
        # 先尝试基于计数器生成
        self._instance_counter[model_id] += 1
        count = self._instance_counter[model_id]
        candidate_id = f"worker-{count}"
        
        # ✅ 重复检测：检查该ID是否已被使用
        existing_instances = self.get_model_instances(model_id)
        used_instance_ids = {inst["instance_id"] for inst in existing_instances}
        
        max_retries = 100  # 防止无限循环
        retry_count = 0
        
        while candidate_id in used_instance_ids and retry_count < max_retries:
            logger.warning(f"⚠️ [ClusterCore] 检测到实例ID冲突: {candidate_id} 已存在，尝试下一个...")
            self._instance_counter[model_id] += 1
            count = self._instance_counter[model_id]
            candidate_id = f"worker-{count}"
            retry_count += 1
        
        if retry_count > 0:
            logger.info(f"✅ [ClusterCore] 经过 {retry_count} 次重试，生成唯一实例ID: {candidate_id}")
        
        return candidate_id
    
    async def _wait_for_instance_loaded(
        self,
        full_model_id: str,
        base_model_id: str,
        timeout: float = 120.0,
        check_interval: float = 1.0,
        initial_grace_period: float = 3.0
    ) -> bool:
        """
        等待模型实例真正加载完成（通过节点 loaded_models 状态确认）。

        批量加载多实例时，必须等前一个实例广播 loaded_models 后再创建下一个，
        否则去重逻辑会失效，导致同一节点重复加载相同模型。
        """
        import asyncio
        start_time = time.time()

        # 初始宽限期：给节点留出开始加载和广播状态的时间
        if initial_grace_period > 0:
            await asyncio.sleep(initial_grace_period)

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.warning(
                    f"⏰ [ClusterCore] 等待实例加载确认超时: {full_model_id} ({timeout:.0f}s)"
                )
                return False

            for node_info in self.nodes.values():
                if node_info.status.value != "online":
                    continue
                for model in node_info.loaded_models:
                    mid = model.get("model_id", "")
                    if mid == full_model_id or mid.startswith(f"{base_model_id}::"):
                        logger.info(
                            f"✅ [ClusterCore] 实例加载确认: {mid} 已在节点 {node_info.node_id}"
                        )
                        return True

            await asyncio.sleep(check_interval)

    async def _fetch_loaded_models_after_delay(self, node_id: str, delay: float = 3.0):
        """
        延迟后主动拉取一次节点的 loaded_models，用于 gRPC 单向命令后的状态同步。
        """
        await asyncio.sleep(delay)
        connector = self.connectors.get(node_id)
        if connector and connector.stub:
            try:
                await connector._fetch_loaded_models()
                logger.info(f"🔄 [Unload] 已同步节点 {node_id} 的模型状态")
            except Exception as e:
                logger.warning(f"⚠️ [Unload] 同步节点 {node_id} 模型状态失败: {e}")

    def get_model_instances(self, model_id: str) -> List[Dict[str, Any]]:
        """
        获取模型的所有实例信息

        Args:
            model_id: 模型ID

        Returns:
            实例信息列表
        """
        instances = []
        
        for node_id, node_info in self.nodes.items():
            for model in node_info.loaded_models:
                mid = model.get("model_id", "")
                if mid == model_id or mid.startswith(f"{model_id}::"):
                    instance_id = "default"
                    if "::" in mid:
                        instance_id = mid.split("::", 1)[1]
                    
                    instances.append({
                        "node_id": node_id,
                        "full_model_id": mid,
                        "instance_id": instance_id,
                        "shard": model.get("shard", {})
                    })
        
        return instances
    
    def get_all_instances_summary(self) -> Dict[str, int]:
        """
        获取所有模型的实例数量摘要
        
        Returns:
            {model_id: instance_count}
        """
        summary = {}
        
        for node_info in self.nodes.values():
            for model in node_info.loaded_models:
                mid = model.get("model_id", "")
                base_model_id = mid.split("::")[0] if "::" in mid else mid
                
                if base_model_id not in summary:
                    summary[base_model_id] = 0
                summary[base_model_id] += 1
        
        return summary
    
    async def initialize(self):
        """初始化管理器：加载配置、连接节点"""
        logger.info("🚀 初始化EXO集群管理器...")
        
        # 从配置文件加载节点
        if self.config_path and os.path.exists(self.config_path):
            await self._load_from_config(self.config_path)
        
        # ✅ 同步实例计数器（从现有节点信息中提取，避免ID冲突）
        self._sync_instance_counters_from_nodes()
        
        # 启动后台监控任务
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info(f"✅ 初始化完成，已加载 {len(self.nodes)} 个节点")
    
    def _sync_instance_counters_from_nodes(self):
        """
        从已连接节点的模型信息中同步实例计数器
        
        确保重启后生成的新实例ID不会与已有实例冲突
        """
        try:
            for node_id, node_info in self.nodes.items():
                for model in node_info.loaded_models:
                    mid = model.get("model_id", "")
                    if "::" not in mid:
                        continue
                    
                    base_model_id, instance_id = mid.split("::", 1)
                    
                    # 只统计 worker-X 格式的实例
                    if instance_id and instance_id != "default" and instance_id.startswith("worker-"):
                        try:
                            worker_num = int(instance_id.split("-")[1])
                            
                            if base_model_id not in self._instance_counter:
                                self._instance_counter[base_model_id] = 0
                            
                            if worker_num > self._instance_counter[base_model_id]:
                                self._instance_counter[base_model_id] = worker_num
                                
                            logger.debug(f"[ClusterCore] 同步实例计数器: {base_model_id} -> {self._instance_counter[base_model_id]} (来自 {mid})")
                        except (ValueError, IndexError):
                            logger.warning(f"[ClusterCore] 无法解析实例ID: {instance_id}")
            
            logger.info(f"✅ [ClusterCore] 实例计数器同步完成: {dict(self._instance_counter)}")
        except Exception as e:
            logger.warning(f"[ClusterCore] 同步实例计数器失败: {e}")
    
    async def _load_from_config(self, config_path: str):
        """从network_config.json加载节点配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            peers = config.get('peers', {})
            
            for node_id, peer_config in peers.items():
                address = peer_config.get('address', '127.0.0.1')
                port = peer_config.get('port', 50051)
                device_caps = peer_config.get('device_capabilities', {})
                
                node_info = EXONodeInfo(
                    node_id=node_id,
                    address=address,
                    port=port,
                    device_info=device_caps
                )
                
                self.nodes[node_id] = node_info
                
                # 创建连接器并尝试连接
                connector = NodeConnector(node_info, manager=self)
                self.connectors[node_id] = connector
                
                success = await connector.connect()
                if success:
                    logger.info(f"  ✓ 节点 {node_id} 已连接 ({address}:{port})")
                else:
                    logger.warning(f"  ✗ 节点 {node_id} 连接失败")
            
            logger.info(f"从配置文件加载了 {len(peers)} 个节点")
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
    
    async def add_node(self, node_id: str, address: str, port: int = 50051, chatgpt_api_port: int = 52415, device_info: Dict[str, Any] = None, skip_grpc_connect: bool = False) -> bool:
        """
        手动添加节点

        Args:
            node_id: 节点唯一标识
            address: IP地址或主机名
            port: gRPC端口号
            chatgpt_api_port: ChatGPT API HTTP端口号
            device_info: 节点主动注册时提供的设备信息（可选）
            skip_grpc_connect: 是否跳过gRPC连接尝试（WS注册节点应设为True）

        Returns:
            是否添加成功
        """
        if node_id in self.nodes:
            logger.warning(f"节点 {node_id} 已存在，尝试重新连接")
            connector = self.connectors.get(node_id)
            if connector:
                await connector.disconnect()
            del self.nodes[node_id]
            if node_id in self.connectors:
                del self.connectors[node_id]
        
        node_info = EXONodeInfo(
            node_id=node_id,
            address=address,
            port=port,
            chatgpt_api_port=chatgpt_api_port,
            device_info=device_info if device_info else {}
        )
        
        self.nodes[node_id] = node_info

        connector = NodeConnector(node_info, manager=self)
        self.connectors[node_id] = connector

        if skip_grpc_connect:
            # WS 注册节点：跳过 gRPC 连接，直接标记为在线（WS 通道已建立）
            node_info.status = NodeStatus.ONLINE
            node_info.last_heartbeat = time.time()
            logger.info(f"➕ 节点注册成功(WS模式): {node_id}@{address} (跳过gRPC连接)")
            self._trigger_node_joined_callback(node_id)
        else:
            success = await connector.connect()

            if success:
                logger.info(f"➕ 节点注册成功: {node_id}@{address}:{port} (HTTP:{chatgpt_api_port})")
                self._trigger_node_joined_callback(node_id)
            else:
                error_msg = connector.node_info.error_message or "未知错误"
                logger.warning(f"⚠️ 节点 {node_id} 已记录但连接失败: {error_msg}")
                logger.info(f"   将在后台重试连接...")
                asyncio.create_task(self._retry_connect(node_id, max_retries=5, interval=3.0))

        return True
    
    async def remove_node(self, node_id: str) -> bool:
        """移除节点"""
        if node_id not in self.nodes:
            return False
        
        connector = self.connectors.get(node_id)
        if connector:
            await connector.disconnect()
            del self.connectors[node_id]
        
        del self.nodes[node_id]
        logger.info(f"➖ 已移除节点: {node_id}")
        
        return True
    
    async def _retry_connect(self, node_id: str, max_retries: int = 5, interval: float = 3.0):
        """后台重试连接节点"""
        connector = self.connectors.get(node_id)
        if not connector:
            return

        _grpc_aio_unavailable = False  # 标记：grpc.aio 不可用时停止重试

        for attempt in range(1, max_retries + 1):
            await asyncio.sleep(interval)

            if node_id not in self.nodes:
                logger.info(f"[RetryConnect] 节点 {node_id} 已被移除，停止重试")
                return

            # grpc.aio 永久性缺失，无需重试
            if _grpc_aio_unavailable:
                return

            try:
                logger.info(f"[RetryConnect] 重试连接 {node_id} ({attempt}/{max_retries})...")

                if connector.channel:
                    try:
                        await connector.channel.close()
                    except Exception:
                        pass

                success = await connector.connect()

                if success:
                    logger.info(f"✅ [RetryConnect] 节点 {node_id} 连接成功!")
                    return
                else:
                    # 检查是否因 grpc.aio 缺失导致失败
                    err_msg = connector.node_info.error_message or ""
                    if "grpc.aio" in err_msg or "has no attribute" in err_msg:
                        logger.warning(f"[RetryConnect] grpc.aio 不可用，停止重试 (依赖WebSocket通信)")
                        _grpc_aio_unavailable = True
                        return
                    logger.warning(f"[RetryConnect] 节点 {node_id} 第{attempt}次重试失败")

            except Exception as e:
                err_str = str(e)
                logger.error(f"[RetryConnect] 节点 {node_id} 重试异常: {e}")
                if "grpc" in err_str and "aio" in err_str:
                    logger.warning(f"[RetryConnect] grpc.aio 不可用，停止重试")
                    _grpc_aio_unavailable = True
                    return

        logger.error(f"[RetryConnect] 节点 {node_id} 重试{max_retries}次后仍失败")
    
    async def _monitor_loop(self):
        """后台监控循环 - 定期检查所有节点状态（含故障恢复）"""
        logger.info("🔄 监控循环已启动")
        _first_sync_done = False  # 首次同步标记，用于重建分配注册表
        _sync_cycle_count = 0     # 同步周期计数器，用于定期重建注册表
        _REGISTRY_REBUILD_INTERVAL = 3  # 每 3 个监控周期（约30秒）重建一次注册表
        _handled_failures = set()  # 已处理的掉线节点（避免重复触发恢复）
        while self._running:
            try:
                online_count = 0
                total_memory = 0
                registry_changed = False  # 标记本轮是否有变化需要重建
                newly_failed_nodes = []   # 本轮新发现的掉线节点

                for node_id, connector in list(self.connectors.items()):
                    is_healthy = await connector.health_check()

                    if is_healthy:
                        online_count += 1
                        node_info = self.nodes[node_id]
                        total_memory += node_info.device_info.get('memory', 0)

                        # 节点恢复在线：从已处理集合中移除
                        if node_id in _handled_failures:
                            _handled_failures.discard(node_id)
                            logger.info(f"[监控循环] 节点 {node_id} 恢复在线")
                            try:
                                from sys_logger import sys_log as _sl
                                _sl.log(_sl.SUCCESS, "node", f"节点 {node_id} 恢复在线", {"node_id": node_id})
                            except Exception:
                                pass

                        logger.debug(f"正在获取节点 {node_id} 的实时状态...")
                        old_model_count = len(connector.node_info.loaded_models)
                        await connector._fetch_loaded_models()
                        # 检测模型列表是否发生变化
                        if len(connector.node_info.loaded_models) != old_model_count:
                            registry_changed = True

                    else:
                        # 节点不健康 → 先做 WS 连接二次验证（WS节点可能 gRPC health_check 失败但仍在线）
                        is_actually_offline = True
                        if _is_ws_available():
                            _ws_mgr = _get_node_ws_manager()
                            if _ws_mgr and _ws_mgr.is_node_connected(node_id):
                                # WS 仍连接，保持在线状态，跳过掉线处理
                                connector.node_info.status = NodeStatus.ONLINE
                                connector.node_info.error_message = ""
                                online_count += 1
                                node_info = self.nodes[node_id]
                                total_memory += node_info.device_info.get('memory', 0)
                                is_actually_offline = False

                        if is_actually_offline:
                            was_online = connector.node_info.status.value == 'online'
                            connector.node_info.status = NodeStatus.OFFLINE

                            if was_online and node_id not in _handled_failures:
                                newly_failed_nodes.append(node_id)
                                _handled_failures.add(node_id)
                                logger.error(f"[监控循环] 节点 {node_id} 掉线！")

                                # 发射系统日志：节点掉线
                                try:
                                    from sys_logger import sys_log as _sl
                                    _sl.log(_sl.ERROR, "node", f"节点 {node_id} 掉线", {"node_id": node_id})
                                except Exception:
                                    pass

                # ====== 故障恢复处理 ======
                for failed_node_id in newly_failed_nodes:
                    try:
                        from gpu_pool_integration import SmartAllocator
                        allocator = SmartAllocator(self)
                        recovery_report = allocator.handle_node_failure(failed_node_id)

                        self._fault_recovery_history = getattr(self, '_fault_recovery_history', [])
                        self._fault_recovery_history.append({
                            "timestamp": time.time(),
                            "failed_node": failed_node_id,
                            "report": recovery_report
                        })

                        summary = recovery_report["summary"]
                        logger.warning(
                            f"[FaultRecovery] 节点 {failed_node_id} 掉线恢复完成: "
                            f"总={summary['total']}, 迁移={summary['migrated']}, "
                            f"降级={summary['degraded']}, 丢失={summary['lost']}"
                        )

                    except Exception as e:
                        logger.error(f"[FaultRecovery] 处理节点 {failed_node_id} 故障时出错: {e}")

                # 首次同步完成后，从节点数据重建分配注册表（重启恢复）
                if not _first_sync_done and online_count > 0:
                    _first_sync_done = True
                    self.rebuild_allocation_registry()
                    logger.info("📋 [监控循环] 首次同步完成，已重建分配注册表")
                
                # 定期重建分配注册表（确保首层节点信息与实际状态同步）
                _sync_cycle_count += 1
                if _first_sync_done and (registry_changed or _sync_cycle_count >= _REGISTRY_REBUILD_INTERVAL):
                    old_registry_size = len(self._allocation_registry)
                    self.rebuild_allocation_registry()
                    new_registry_size = len(self._allocation_registry)
                    logger.info(
                        f"📋 [监控循环] 分配注册表已定期重建: "
                        f"{old_registry_size} -> {new_registry_size} 个模型"
                        f"{' (检测到模型变化)' if registry_changed else ''}"
                    )
                    _sync_cycle_count = 0
                
                # 更新统计数据
                self.stats.update({
                    "total_nodes": len(self.nodes),
                    "online_nodes": online_count,
                    "total_models": sum(len(n.loaded_models) for n in self.nodes.values()),
                    "total_memory_gb": total_memory / 1024,
                    "last_update": time.time()
                })
                
                # 通知 WebSocket 客户端更新
                await self._broadcast_status_update()
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
            
            # 每10秒检查一次
            await asyncio.sleep(10)
    
    async def _broadcast_status_update(self):
        """广播状态更新给所有 WebSocket 客户端"""
        if self._broadcast_callback:
            try:
                status = self.get_cluster_status()
                await self._broadcast_callback({
                    "type": "status_update",
                    "data": status
                })
            except Exception as e:
                logger.debug(f"广播状态更新失败: {e}")

    async def _on_model_load_completed(
        self,
        node_id: str,
        task_id: str,
        success: bool,
        loaded_models: List[Dict],
        error: str = ""
    ):
        """
        WebSocket 回调：Node 完成模型加载
        
        由 NodeWSManager 在收到 model_load_complete 消息时调用
        """
        if success:
            # 更新节点信息
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.loaded_models = loaded_models
                node.last_heartbeat = time.time()
                node.status = NodeStatus.ONLINE
                
                logger.info(f"📦 [WS-Callback] 节点 {node_id} 模型加载成功 (任务ID: {task_id})")
                
                # 广播状态更新（WebSocket 推送给前端）
                if self._broadcast_callback:
                    try:
                        await self._broadcast_callback({
                            "type": "model_loaded",
                            "node_id": node_id,
                            "task_id": task_id,
                            "loaded_models": loaded_models,
                            "success": True
                        })
                    except Exception as e:
                        logger.debug(f"广播模型加载事件失败: {e}")
        else:
            logger.error(f"❌ [WS-Callback] 节点 {node_id} 模型加载失败 (任务ID: {task_id}): {error}")
            
            # 可以在这里触发重试或其他错误处理逻辑

    async def _on_model_unload_completed(
        self,
        node_id: str,
        model_id: str,
        success: bool
    ):
        """
        WebSocket 回调：Node 完成模型卸载
        
        由 NodeWSManager 在收到 model_unload_complete 消息时调用
        """
        if success and node_id in self.nodes:
            node = self.nodes[node_id]
            
            # 更新节点的已加载模型列表
            node.loaded_models = [
                m for m in node.loaded_models 
                if m.get("model_id") != model_id
            ]
            node.last_heartbeat = time.time()
            
            logger.info(f"🗑️ [WS-Callback] 节点 {node_id} 模型卸载成功: {model_id}")
            
            # 广播状态更新
            if self._broadcast_callback:
                try:
                    await self._broadcast_callback({
                        "type": "model_unloaded",
                        "node_id": node_id,
                        "model_id": model_id,
                        "success": True
                    })
                except Exception as e:
                    logger.debug(f"广播模型卸载事件失败: {e}")
        else:
            logger.error(f"❌ [WS-Callback] 节点 {node_id} 模型卸载失败: {model_id}")

    def get_cluster_status(self) -> Dict:
        """获取整个集群的状态摘要"""
        nodes_list = [node.to_dict() for node in self.nodes.values()]
        
        return {
            **self.stats,
            "nodes": nodes_list,
            "manager_uptime": time.time() - self.stats.get("start_time", time.time())
        }
    
    def get_node_detail(self, node_id: str) -> Optional[Dict]:
        """获取单个节点的详细信息"""
        node = self.nodes.get(node_id)
        if not node:
            return None
        
        return node.to_dict()
    
    async def load_model_to_cluster(
        self,
        model_id: str,
        model_path: str,
        n_layers: int = 32,
        strategy: str = "smart",
        target_nodes: Optional[List[str]] = None,
        instance_id: Optional[str] = None,
        auto_instance: bool = False,
        layer_memory_mb: Optional[float] = None
    ) -> Dict:
        """
        将模型加载到集群（核心方法）- 支持多实例
        
        工作流程：
        1. 计算分片分配方案
        2. 向各节点发送分片配置
        3. 等待节点确认加载完成
        
        Args:
            model_id: 模型标识符 (如 "Qwen/Qwen3-4B")
            model_path: 模型文件路径 (如 "./models/qwen3-4b")
            n_layers: 模型总层数
            strategy: 分配策略 ('smart', 'memory_weighted', 'uniform', 'performance_weighted')
                       smart=智能策略(默认): 单节点优先，能不拆就不拆
                       memory_weighted=按显存加权, uniform=均匀分配, performance_weighted=按性能加权
            target_nodes: 指定目标节点列表（可选，默认使用所有在线节点）
            instance_id: 实例ID（支持多实例，None表示默认）
            auto_instance: 是否自动生成实例ID
            layer_memory_mb: 每层预估显存(MB)，未指定时从模型库自动获取
            
        Returns:
            加载结果详情
            
        示例:
            # 加载默认实例
            await manager.load_model_to_cluster("qwen3-0.6b", "./models")
            
            # 加载多个实例
            await manager.load_model_to_cluster("qwen3-0.6b", "./models", instance_id="worker-1")
            await manager.load_model_to_cluster("qwen3-0.6b", "./models", instance_id="worker-2")
            
            # 自动生成实例ID
            await manager.load_model_to_cluster("qwen3-0.6b", "./models", auto_instance=True)
        """
        from gpu_pool_integration import GPUPoolIntegration
        from auto_model_allocator import get_model_library
        
        # 处理 instance_id
        if auto_instance and instance_id is None:
            instance_id = self._generate_instance_id(model_id)
        
        if instance_id is None:
            instance_id = "default"
        
        # 单节点模式下强制使用 default 实例，避免 auto_instance 反复生成 worker-X
        # 导致单节点上堆积多个完整模型副本并浪费显存
        online_nodes = [nid for nid, n in self.nodes.items() if n.status == NodeStatus.ONLINE]
        if len(online_nodes) == 1 and instance_id != "default":
            logger.info(
                f"🔍 [ModelLoad] 单节点模式（{online_nodes[0]}），"
                f"强制使用 default 实例而非 {instance_id}"
            )
            instance_id = "default"
        
        logger.info(f"🔍 [ModelLoad] instance_id 最终值: {instance_id}, auto_instance={auto_instance}")
        
        # 构建完整模型ID用于内部管理
        full_model_id = f"{model_id}" if instance_id == "default" else f"{model_id}::{instance_id}"
        
        # 自动获取每层显存估算（如果调用方未提供）
        if layer_memory_mb is None:
            base_model_id = model_id.split("::")[0] if "::" in model_id else model_id
            model_lib = get_model_library()
            spec = model_lib.get(base_model_id)
            if not spec:
                # 尝试用 HF repo ID 反向查找
                for key, lib_spec in model_lib.items():
                    if lib_spec.model_id == base_model_id:
                        spec = lib_spec
                        break
            if spec:
                layer_memory_mb = spec.layer_memory_mb
                logger.info(f"🔍 [ModelLoad] 从模型库获取 {base_model_id} 每层显存: {layer_memory_mb}MB")
            else:
                layer_memory_mb = 100.0
                logger.warning(f"⚠️ [ModelLoad] 模型库中未找到 {base_model_id}，使用默认每层显存 {layer_memory_mb}MB")
        
        logger.info(f"🚀 开始加载模型到集群: {model_id} (实例: {instance_id}, 每层{layer_memory_mb}MB)")
        
        pool = GPUPoolIntegration(self)
        
        try:
            # Step 1: 预览/计算分配方案
            allocation = await pool.preview_allocation(
                model_id=model_id,
                total_layers=n_layers,
                layer_memory_mb=layer_memory_mb,
                strategy=strategy
            )
            
            logger.info(f"📦 模型 {full_model_id} 分配方案:")
            for alloc in allocation.allocations:
                logger.info(f"   节点 {alloc['node_id']}: 层 {alloc['start_layer']}-{alloc['end_layer']} ({alloc['layers_count']}层)")

            # 处理 pending_more_nodes 状态：集群显存不足，需要更多节点
            if allocation.allocation_type == "pending_more_nodes":
                logger.warning(f"📦 [ModelLoad] ⏳ 模型 {model_id} 需要更多节点: {allocation.decision_reason}")
                return {
                    "success": False,
                    "error": allocation.decision_reason,
                    "status": "pending_more_nodes",
                    "min_nodes_required": getattr(allocation, 'min_nodes_required', 0),
                    "instance_id": instance_id
                }

            # Step 2: 过滤目标节点（如果指定）
            if target_nodes:
                allocation.allocations = [
                    a for a in allocation.allocations 
                    if a["node_id"] in target_nodes
                ]
            
            if not allocation.allocations:
                return {
                    "success": False,
                    "error": "没有可用的目标节点"
                }

            # 自动实例模式下，避免在同一节点重复加载同一基础模型
            base_model_id = model_id.split("::")[0] if "::" in model_id else model_id
            if auto_instance:
                deduped_allocs = []
                for alloc in allocation.allocations:
                    node_id = alloc["node_id"]
                    node_info = self.nodes.get(node_id)
                    already_loaded = False
                    if node_info and node_info.loaded_models:
                        for m in node_info.loaded_models:
                            loaded_id = m.get("model_id", "")
                            loaded_base = m.get("base_model_id") or (
                                loaded_id.split("::")[0] if "::" in loaded_id else loaded_id
                            )
                            if loaded_base == base_model_id:
                                already_loaded = True
                                logger.info(f"🔄 [ModelLoad] 节点 {node_id} 已加载 {base_model_id}，跳过重复实例")
                                break
                    if not already_loaded:
                        deduped_allocs.append(alloc)

                if not deduped_allocs:
                    logger.info(f"🔄 [ModelLoad] {base_model_id} 已在所有目标节点加载，无需创建新实例")
                    return {
                        "success": True,
                        "model_id": model_id,
                        "instance_id": instance_id,
                        "full_model_id": full_model_id,
                        "message": f"{base_model_id} 已在目标节点加载，跳过重复实例",
                        "load_results": []
                    }

                allocation.allocations = deduped_allocs

            results = []
            for alloc in allocation.allocations:
                node_id = alloc['node_id']
                connector = self.connectors.get(node_id)
                
                if not connector:
                    results.append({
                        "node_id": node_id,
                        "success": False,
                        "error": "节点连接器不存在"
                    })
                    continue
                
                # 构建分片配置数据
                peer_list = []
                for node in self.nodes.values():
                    if node.node_id != node_id and node.status == NodeStatus.ONLINE:
                        peer_list.append({
                            "node_id": node.node_id,
                            "address": node.address,
                            "port": node.port,
                            "device_capabilities": node.device_info
                        })
                
                shard_task = {
                    "task_id": f"{full_model_id}_{alloc['start_layer']}_{alloc['end_layer']}_{int(time.time())}",
                    "model_id": full_model_id,
                    "base_model_id": model_id,
                    "instance_id": instance_id,
                    "model_path": model_path,
                    "shard": {
                        "start_layer": alloc["start_layer"],
                        "end_layer": alloc["end_layer"],
                        "n_layers": n_layers
                    },
                    "peer_list": peer_list,
                    "created_at": time.time(),
                    "status": "pending"
                }
                
                # ✨ 策略优先级：WebSocket Push > gRPC Direct > HTTP Pull
                ws_result = None
                
                # 1️⃣ 优先：WebSocket 实时推送（内网穿透场景 - 最快）
                try:
                    # 🔍 诊断日志：WS 连接状态
                    _ws_mgr_load = _get_node_ws_manager()
                    logger.info(f"🔍 [ModelLoad] WS 推送诊断 (节点 {node_id}):")
                    logger.info(f"   _is_ws_available() = {_is_ws_available()}")
                    logger.info(f"   node_ws_manager exists = {_ws_mgr_load is not None}")
                    if _ws_mgr_load:
                        logger.info(f"   connected nodes = {list(_ws_mgr_load.node_connections.keys())}")
                        logger.info(f"   is_node_connected({node_id}) = {_ws_mgr_load.is_node_connected(node_id)}")
                    
                    if _is_ws_available() and _ws_mgr_load and _ws_mgr_load.is_node_connected(node_id):
                        logger.info(f"📡 [ModelLoad] 尝试通过 WebSocket 推送到节点 {node_id}...")
                        ws_success = await _ws_mgr_load.send_model_load_request(
                            node_id=node_id,
                            task_id=shard_task["task_id"],
                            model_id=full_model_id,
                            model_path=model_path,
                            shard=shard_task["shard"],
                            peer_list=peer_list,
                            instance_id=instance_id  # ✅ 传递实例ID
                        )
                        
                        if ws_success:
                            # ✅ 更新 Manager 端的 loaded_models 列表（与 gRPC/HTTP 路径保持一致）
                            ws_shard_payload = {
                                "model_id": full_model_id,
                                "base_model_id": model_id,
                                "instance_id": instance_id,
                                "model_path": model_path,
                                "shard": shard_task["shard"]
                            }
                            connector._update_loaded_models(ws_shard_payload)
                            
                            results.append({
                                "node_id": node_id,
                                "success": True,
                                "method": "websocket_push",  # 标记为 WS 推送
                                "task_id": shard_task["task_id"],
                                "message": f"任务已通过 WebSocket 实时推送到节点 {node_id}",
                                "shard": shard_task["shard"]
                            })
                            logger.info(f"🚀 [WS-Push] 节点 {node_id} 分片配置已实时推送 (任务ID: {shard_task['task_id']})")
                            continue
                except Exception as e:
                    logger.warning(f"⚠️ [WS-Push] WebSocket 推送失败 ({node_id}): {e}, 尝试其他方式")
                
                # 2️⃣ 次优：gRPC 直接发送（公网场景）
                direct_result = None
                if connector.stub:
                    try:
                        direct_result = await connector.send_shard_config(
                            model_id=full_model_id,
                            model_path=model_path,
                            start_layer=alloc["start_layer"],
                            end_layer=alloc["end_layer"],
                            n_layers=n_layers,
                            peer_list=peer_list
                        )
                        if direct_result.get("success"):
                            results.append(direct_result)
                            logger.info(f"✅ [Direct] 节点 {node_id} 分片配置已发送")
                            continue
                    except Exception as e:
                        logger.warning(f"[Direct] 发送失败 ({node_id}): {e}, 切换到 Pull 模式")
                
                # ✨ Pull 模式：存入待处理队列，等待 Node 心跳拉取
                if node_id not in self.pending_tasks:
                    self.pending_tasks[node_id] = []
                
                self.pending_tasks[node_id].append(shard_task)
                
                results.append({
                    "node_id": node_id,
                    "success": True,  # ✅ 任务已入队，视为成功
                    "method": "pull",  # 标记为 Pull 模式
                    "task_id": shard_task["task_id"],
                    "message": f"任务已入队，等待节点 {node_id} 通过心跳拉取",
                    "shard": shard_task["shard"]
                })
                
                logger.info(f"📥 [Pull] 节点 {node_id} 分片配置已入队 (任务ID: {shard_task['task_id']})")
            
            # Step 4: 汇总结果
            success_count = sum(1 for r in results if r.get("success"))
            total = len(results)
            
            # 查找首层节点（start_layer=0），其推理地址可直接用于客户端直连
            inference_url = ""
            first_layer_node_id = None
            for alloc in allocation.allocations:
                if alloc.get("start_layer") == 0:
                    first_layer_node_id = alloc["node_id"]
                    fl_connector = self.connectors.get(first_layer_node_id)
                    if fl_connector and fl_connector.node_info.status == NodeStatus.ONLINE:
                        inference_url = fl_connector.node_info.chat_completions_url
                    break

            # ✅ 注册分配信息到 _allocation_registry（供 /v1/models 查询首层节点）
            # 支持多首层节点格式（兼容单副本场景）
            if success_count > 0 and allocation.allocations:
                alloc_entries = []
                fl_node_ids = []
                fl_urls = {}
                for alloc in allocation.allocations:
                    alloc_entries.append({
                        "node_id": alloc.get("node_id", ""),
                        "start_layer": alloc.get("start_layer", -1),
                        "end_layer": alloc.get("end_layer", -1),
                        "instance_id": instance_id,
                    })
                    if alloc.get("start_layer") == 0:
                        nid = alloc["node_id"]
                        fl_node_ids.append(nid)
                        c = self.connectors.get(nid)
                        if c and c.node_info.status == NodeStatus.ONLINE:
                            fl_urls[nid] = c.node_info.chat_completions_url

                self._allocation_registry[model_id] = {
                    "allocations": alloc_entries,
                    "first_layer_node_ids": fl_node_ids or [first_layer_node_id],
                    "first_layer_node_id": first_layer_node_id,
                    "inference_urls": fl_urls,
                    "inference_url": inference_url,
                    "full_model_id": full_model_id,
                    "model_path": model_path,
                    "n_layers": n_layers,
                    "updated_at": time.time(),
                }
                logger.info(f"📋 [分配注册] 模型 {model_id} -> 首层节点: {fl_node_ids or [first_layer_node_id]}, 推理地址: {inference_url}")

            return {
                "success": success_count > 0,
                "model_id": model_id,
                "instance_id": instance_id,
                "full_model_id": full_model_id,
                "inference_url": inference_url,          # 首层节点推理地址（客户端可直连）
                "first_layer_node_id": first_layer_node_id,  # 首层节点ID
                "allocation": {
                    "strategy": strategy,
                    "total_layers": n_layers,
                    "nodes_count": total,
                    "allocations": allocation.allocations
                },
                "load_results": results,
                "summary": {
                    "success_nodes": success_count,
                    "failed_nodes": total - success_count,
                    "total_nodes": total
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_id": model_id
            }
    
    async def load_multiple_instances(
        self,
        model_id: str,
        model_path: str,
        n_layers: int = 32,
        strategy: str = "smart",
        target_nodes: Optional[List[str]] = None,
        count: int = 1
    ) -> Dict:
        """
        批量加载多个模型实例（后端全权管理instance_id）
        
        ✨ 核心方法：前端只需传入数量，后端自动分配唯一ID并避免冲突
        
        Args:
            model_id: 模型标识符
            model_path: 模型文件路径
            n_layers: 模型总层数
            strategy: 分配策略
            target_nodes: 目标节点列表（可选）
            count: 要创建的实例数量
            
        Returns:
            {
                "success": true,
                "model_id": "...",
                "instances": [
                    {"instance_id": "worker-3", "full_model_id": "...::worker-3", "success": true, ...},
                    {"instance_id": "worker-4", "full_model_id": "...::worker-4", "success": true, ...}
                ],
                "summary": {"total": 2, "success": 2, "failed": 0}
            }
        """
        logger.info(f"🚀 [BatchLoad] 开始批量加载 {count} 个实例: {model_id}")

        base_model_id = model_id.split("::")[0] if "::" in model_id else model_id
        instances_results = []
        success_count = 0
        failed_count = 0

        for i in range(1, count + 1):
            try:
                # 若当前已存在足够实例，直接跳过剩余循环
                existing_instances = self.get_model_instances(base_model_id)
                if len(existing_instances) >= count:
                    logger.info(
                        f"🔄 [BatchLoad] {base_model_id} 已存在 {len(existing_instances)} "
                        f"个实例，满足请求数量 {count}，停止继续创建"
                    )
                    # 把已存在实例补充到结果中
                    for inst in existing_instances:
                        if not any(r.get("full_model_id") == inst["full_model_id"] for r in instances_results):
                            instances_results.append({
                                "instance_id": inst["instance_id"],
                                "full_model_id": inst["full_model_id"],
                                "success": True,
                                "allocation": inst.get("shard"),
                                "inference_url": "",
                                "first_layer_node_id": inst.get("node_id"),
                                "error": None,
                                "note": "已存在实例"
                            })
                            success_count += 1
                    break

                logger.info(f"  📦 [{i}/{count}] 正在创建第 {i} 个实例...")

                # ✅ 每次调用都使用 auto_instance=True，让系统自动生成唯一ID
                result = await self.load_model_to_cluster(
                    model_id=model_id,
                    model_path=model_path,
                    n_layers=n_layers,
                    strategy=strategy,
                    target_nodes=target_nodes,
                    instance_id=None,       # 不指定，让系统自动分配
                    auto_instance=True      # ✅ 启用自动生成（带冲突检测）
                )

                instance_info = {
                    "instance_id": result.get("instance_id", f"unknown-{i}"),
                    "full_model_id": result.get("full_model_id", f"{model_id}::unknown-{i}"),
                    "success": result.get("success", False),
                    "allocation": result.get("allocation"),
                    "inference_url": result.get("inference_url", ""),       # 首层节点推理地址
                    "first_layer_node_id": result.get("first_layer_node_id"),  # 首层节点ID
                    "error": result.get("error")
                }

                instances_results.append(instance_info)

                if result.get("success"):
                    success_count += 1
                    logger.info(f"  ✓ [{i}/{count}] 实例 {instance_info['instance_id']} 创建成功")

                    # 🛡️ 关键修复：等待实例真正加载完成并广播 loaded_models 后，
                    # 再进行下一轮循环。否则去重逻辑无法感知该实例，导致同一节点重复加载。
                    if i < count and instance_info.get("full_model_id"):
                        await self._wait_for_instance_loaded(
                            full_model_id=instance_info["full_model_id"],
                            base_model_id=base_model_id,
                            timeout=120.0,
                            check_interval=1.0,
                            initial_grace_period=2.0
                        )
                else:
                    failed_count += 1
                    logger.warning(f"  ✗ [{i}/{count}] 实例 #{i} 失败: {result.get('error')}")

            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                instances_results.append({
                    "instance_id": f"error-{i}",
                    "full_model_id": f"{model_id}::error-{i}",
                    "success": False,
                    "error": error_msg
                })
                logger.error(f"  ✗ [{i}/{count}] 实例 #{i} 异常: {error_msg}")

            # 实例间添加小延迟，避免并发冲突
            if i < count:
                await asyncio.sleep(0.2)
        
        summary = {
            "total": count,
            "success": success_count,
            "failed": failed_count
        }
        
        logger.info(f"✅ [BatchLoad] 批量加载完成: {model_id} -> 成功 {success_count}/{count}")
        
        # 提取首个成功实例的首层节点信息（供顶层快速访问）
        top_inference_url = ""
        top_first_layer_node_id = ""
        for inst in instances_results:
            if inst.get("success") and inst.get("inference_url"):
                top_inference_url = inst["inference_url"]
                top_first_layer_node_id = inst.get("first_layer_node_id", "")
                break

        return {
            "success": success_count > 0,
            "model_id": model_id,
            "inference_url": top_inference_url,              # 首层节点推理地址（客户端可直连）
            "first_layer_node_id": top_first_layer_node_id,  # 首层节点ID
            "instances": instances_results,
            "summary": summary
        }
    
    async def unload_model_from_cluster(
        self, 
        model_id: str, 
        instance_id: Optional[str] = None,
        unload_all_instances: bool = False
    ) -> Dict:
        """
        从集群卸载模型（支持多实例）
        
        Args:
            model_id: 要卸载的模型ID
            instance_id: 实例ID（None表示默认实例或所有实例）
            unload_all_instances: 是否卸载该模型的所有实例
            
        Returns:
            卸载结果
        """
        # 构建完整模型ID
        if instance_id is None:
            instance_id = "default"
        
        full_model_id = f"{model_id}" if instance_id == "default" else f"{model_id}::{instance_id}"
        
        logger.info(f"🗑️ 开始卸载模型: {full_model_id} (卸载所有实例={unload_all_instances})")
        
        results = []
        any_command_sent = False

        for node_id, connector in self.connectors.items():
            # 允许 WebSocket-only 节点参与卸载（只要任一通道可用即可）
            ws_available = (
                _is_ws_available()
                and _get_node_ws_manager()
                and _get_node_ws_manager().is_node_connected(node_id)
            )
            if not connector.stub and not ws_available:
                continue

            try:
                # 传入完整模型ID用于精确匹配，并传递 unload_all_instances 标志
                result = await connector.send_unload_command(full_model_id, unload_all_instances=unload_all_instances)
                results.append({
                    "node_id": node_id,
                    "success": result.get("success", False),
                    "method": result.get("method"),
                    "message": result.get("message") or result.get("error")
                })

                if result.get("success"):
                    any_command_sent = True

                    # WebSocket 路径：等待 Node 的 model_unload_complete 回调再清理本地状态，
                    # 避免命令未到达节点时 Manager 就丢失模型记录。
                    # gRPC 路径：SendOpaqueStatus 是单向命令，安排一次显式拉取以同步真实状态。
                    if result.get("method") == "grpc":
                        asyncio.create_task(
                            self._fetch_loaded_models_after_delay(node_id, delay=3.0),
                            name=f"fetch-after-unload-{node_id}-{model_id}"
                        )

            except Exception as e:
                logger.error(f"❌ [UnloadCluster] 节点 {node_id} 卸载异常: {e}")
                results.append({
                    "node_id": node_id,
                    "success": False,
                    "error": str(e)
                })

        # 只有至少一个卸载命令成功发出后，才清理分配注册表
        if any_command_sent and model_id in self._allocation_registry:
            del self._allocation_registry[model_id]
            logger.info(f"🗑️ [分配注册] 已移除模型 {model_id} 的分配记录")

        return {
            "success": any(r["success"] for r in results),
            "model_id": model_id,
            "instance_id": instance_id,
            "full_model_id": full_model_id,
            "unload_all_instances": unload_all_instances,
            "results": results
        }
    
    async def rebalance_model(self, model_id: str) -> Dict:
        """
        重新平衡模型分布

        当有新节点加入或节点资源变化时调用，
        会重新计算最优分配方案并迁移分片。
        """
        # 先获取当前模型的分配信息，同时提取重加载所需的元数据
        current_allocations = []
        model_path = ""
        n_layers = 0
        for node_id, node_info in self.nodes.items():
            for model in node_info.loaded_models:
                if model.get("model_id") == model_id or model.get("base_model_id") == model_id:
                    current_allocations.append({
                        "node_id": node_id,
                        **model
                    })
                    if not model_path:
                        model_path = model.get("model_path", "")
                    shard = model.get("shard", {})
                    if shard.get("n_layers", 0) > n_layers:
                        n_layers = shard.get("n_layers", 0)

        if not current_allocations:
            return {"success": False, "error": f"模型 {model_id} 未找到"}

        # 尝试从分配注册表补充模型路径与层数
        registry_info = self._allocation_registry.get(model_id, {})
        if not model_path:
            model_path = registry_info.get("model_path", "")
        if not n_layers:
            n_layers = registry_info.get("n_layers", 0)

        # 仍未拿到路径，则无法重新加载
        if not model_path:
            return {"success": False, "error": f"模型 {model_id} 缺少 model_path，无法重新加载"}
        if not n_layers:
            return {"success": False, "error": f"模型 {model_id} 缺少 n_layers，无法重新加载"}

        # 卸载旧分配（包含所有实例）
        unload_result = await self.unload_model_from_cluster(
            model_id, unload_all_instances=True
        )

        # 即使卸载返回不完全成功，也尝试重新加载（旧分片可能已不存在）
        if not unload_result["success"]:
            logger.warning(f"⚠️ [Rebalance] 模型 {model_id} 卸载旧分配未完全成功，仍尝试重新加载")

        # 重新加载，让智能分配器基于最新集群状态重新决策
        load_result = await self.load_model_to_cluster(
            model_id=model_id,
            model_path=model_path,
            n_layers=n_layers,
            strategy="smart"
        )

        return {
            "success": load_result.get("success", False),
            "message": f"模型 {model_id} 已重新平衡",
            "unload_result": unload_result,
            "load_result": load_result
        }

    async def rebalance_loaded_models(self) -> Dict:
        """
        重新平衡当前已加载的所有模型

        当有新节点加入时调用，将已加载模型按最新资源状态重新分配，
        使新节点能够参与推理。
        """
        # 收集当前已加载的模型（按 base_model_id 去重）
        loaded_model_ids = set()
        for node_info in self.nodes.values():
            for model in node_info.loaded_models:
                model_id = model.get("model_id", "")
                base_id = model.get("base_model_id") or (
                    model_id.split("::")[0] if "::" in model_id else model_id
                )
                if base_id:
                    loaded_model_ids.add(base_id)

        if not loaded_model_ids:
            return {"success": True, "message": "没有已加载的模型需要重新平衡", "rebalanced": []}

        logger.info(f"🔄 [ClusterRebalance] 开始重新平衡 {len(loaded_model_ids)} 个模型: {loaded_model_ids}")

        results = []
        for model_id in sorted(loaded_model_ids):
            try:
                result = await self.rebalance_model(model_id)
                results.append({"model_id": model_id, **result})
                logger.info(f"🔄 [ClusterRebalance] {model_id}: success={result.get('success')}")
            except Exception as e:
                logger.error(f"❌ [ClusterRebalance] {model_id} 重新平衡失败: {e}", exc_info=True)
                results.append({"model_id": model_id, "success": False, "error": str(e)})

        success_count = sum(1 for r in results if r.get("success"))
        return {
            "success": success_count == len(results),
            "total": len(results),
            "success_count": success_count,
            "failed_count": len(results) - success_count,
            "results": results
        }
    
    async def shutdown(self):
        """关闭管理器，清理资源"""
        logger.info("正在关闭EXO集群管理器...")
        
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # 断开所有连接
        for connector in self.connectors.values():
            await connector.disconnect()
        
        self.connectors.clear()
        self.nodes.clear()
        
        logger.info("✅ EXO集群管理器已关闭")


# 全局实例（单例）
cluster_manager: Optional[EXOClusterManager] = None


async def get_cluster_manager() -> EXOClusterManager:
    """获取全局集群管理器实例"""
    global cluster_manager
    
    if cluster_manager is None:
        cluster_manager = EXOClusterManager()
        await cluster_manager.initialize()
    
    return cluster_manager


__all__ = ['EXOClusterManager', 'NodeConnector', 'EXONodeInfo', 'get_cluster_manager']
