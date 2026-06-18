"""
EXO Manager - P2P 拓扑管理
===========================

处理EXO节点的P2P网络拓扑，包括：
- 拓扑收集与缓存
- 推理链路追踪
- 网络健康诊断
- 可视化数据生成

核心设计原则:
-------------
1. **被动感知**: 不干预节点的P2P连接，只观察和监控
2. **非侵入式**: 通过现有gRPC接口获取拓扑，不修改节点行为
3. **实时更新**: 定期从节点刷新拓扑信息
4. **智能分析**: 自动识别推理链路、瓶颈节点等

使用方式:
    from p2p_topology import P2PTopologyManager
    
    topo_manager = P2PTopologyManager(cluster_manager)
    
    # 获取完整拓扑
    topology = await topo_manager.collect_full_topology()
    
    # 获取推理链路
    pipeline = await topo_manager.get_inference_pipeline("model_id")
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConnectionType(Enum):
    """连接类型"""
    P2P_DIRECT = "p2p_direct"           # 直接P2P连接
    P2P_RELAY = "p2p_relay"             # 通过FRP中继
    INFERENCE_CHAIN = "inference_chain" # 推理链路（按层顺序）


@dataclass
class TopologyNode:
    """拓扑中的节点"""
    node_id: str
    device_capabilities: Dict[str, Any] = field(default_factory=dict)
    address: str = ""
    port: int = 0
    status: str = "unknown"
    loaded_shards: List[Dict[str, Any]] = field(default_factory=list)
    role: str = "worker"  # worker, coordinator, seed
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class TopologyEdge:
    """拓扑中的边（连接关系）"""
    from_node: str
    to_node: str
    connection_type: ConnectionType = ConnectionType.P2P_DIRECT
    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    description: str = ""
    is_active: bool = True
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result["connection_type"] = self.connection_type.value
        return result


@dataclass
class InferenceStep:
    """推理链路中的一步"""
    step_index: int
    node_id: str
    shard_info: Dict[str, Any]
    input_type: str = "hidden_state"  # token_ids 或 hidden_state
    output_type: str = "hidden_state"  # hidden_state 或 logits
    estimated_latency_ms: float = 0.0
    is_first_layer: bool = False
    is_last_layer: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InferencePipeline:
    """完整的推理流水线"""
    model_id: str
    total_layers: int
    steps: List[InferenceStep] = field(default_factory=list)
    pipeline_type: str = "sequential"  # sequential, parallel, hybrid
    total_estimated_latency_ms: float = 0.0
    bottleneck_node: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result["steps"] = [s.to_dict() for s in self.steps]
        return result


class P2PTopologyManager:
    """
    P2P拓扑管理器
    
    负责收集、分析和展示EXO集群的P2P网络拓扑。
    
    工作原理:
    ---------
    1. 选择一个或多个"种子节点"作为入口
    2. 调用节点的collect_topology()方法递归获取网络图
    3. 解析并缓存拓扑数据
    4. 提供查询和分析接口
    """
    
    def __init__(self, cluster_manager):
        """
        初始化拓扑管理器
        
        Args:
            cluster_manager: EXOClusterManager实例
        """
        self.manager = cluster_manager
        
        # 缓存的拓扑数据
        self.nodes: Dict[str, TopologyNode] = {}
        self.edges: List[TopologyEdge] = []
        
        # 推理流水线缓存
        self.inference_pipelines: Dict[str, InferencePipeline] = {}
        
        # 收集历史（用于变化检测）
        self.collection_history: List[Dict] = []
        self.last_collection_time: float = 0
        self.collection_interval: int = 30  # 秒
        
        # 后台任务
        self._collection_task: Optional[asyncio.Task] = None
        self._running: bool = False
        
        logger.info("🔗 P2P拓扑管理器已初始化")
    
    async def start(self):
        """启动后台拓扑收集任务"""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._periodic_collection())
        logger.info("🔄 P2P拓扑后台收集已启动")
    
    async def stop(self):
        """停止后台任务"""
        self._running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("⏹️ P2P拓扑管理器已停止")
    
    async def _periodic_collection(self):
        """定期收集拓扑"""
        while self._running:
            try:
                await self.collect_full_topology()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"拓扑收集出错: {e}")
                await asyncio.sleep(10)  # 出错后等待更长时间
    
    async def collect_full_topology(self, force: bool = False) -> Dict[str, Any]:
        """
        收集完整的P2P网络拓扑
        
        Args:
            force: 是否强制重新收集（忽略缓存）
            
        Returns:
            包含节点和边信息的字典
        """
        current_time = time.time()

        # 如果不是强制且距离上次收集时间很短，返回缓存
        if not force and (current_time - self.last_collection_time) < self.collection_interval / 2:
            if self.nodes and self.edges:
                return self.get_topology_summary()

        # 单节点时跳过拓扑收集（没有 P2P 边可发现）
        online_nodes = [
            c for c in self.manager.connectors.values()
            if c.node_info.status.value == 'online'
        ]
        if len(online_nodes) <= 1:
            # 仍需保证基础拓扑数据存在（供 API 查询使用）
            if not self.nodes:
                self._build_basic_topology()
            return self.get_topology_summary()
        
        logger.info(f"🗺️ 开始收集P2P拓扑... (当前已知 {len(self.nodes)} 个节点)")
        
        # 清空旧数据（但保留历史）
        old_nodes = set(self.nodes.keys())
        self.nodes.clear()
        self.edges.clear()
        
        # 从所有在线节点尝试收集
        collected_from = []
        
        for node_id, connector in self.manager.connectors.items():
            if connector.node_info.status.value == 'online':
                try:
                    # 尝试通过gRPC调用collect_topology
                    topology_data = await self._collect_from_node(connector)
                    
                    if topology_data:
                        self._parse_topology_data(topology_data, source_node=node_id)
                        collected_from.append(node_id)
                        logger.info(f"  ✓ 从 {node_id} 收集到拓扑")
                        
                        # 通常一个节点就足够获取完整拓扑
                        break
                        
                except Exception as e:
                    logger.warning(f"  ✗ 从 {node_id} 收集失败: {e}")
                    continue
        
        # 如果无法从节点收集，使用manager的节点列表构建基础拓扑
        if not self.nodes:
            logger.warning("无法从节点收集拓扑，使用本地节点列表")
            self._build_basic_topology()
        
        # 记录收集历史
        self.last_collection_time = current_time
        self.collection_history.append({
            "timestamp": current_time,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "collected_from": collected_from
        })
        
        # 只保留最近100条记录
        if len(self.collection_history) > 100:
            self.collection_history = self.collection_history[-100:]
        
        # 分析推理流水线
        await self._analyze_inference_pipelines()
        
        summary = self.get_topology_summary()
        logger.info(f"✅ 拓扑收集完成: {len(self.nodes)} 节点, {len(self.edges)} 边")
        
        return summary
    
    async def _collect_from_node(self, connector) -> Optional[Dict]:
        """
        从单个节点收集拓扑数据
        
        这里需要实际调用节点的gRPC接口
        目前返回模拟数据或尝试解析
        """
        try:
            import grpc
            import sys
            import os
            
            if not connector.stub:
                return None
            
            # 导入正确的 protobuf 消息类型
            try:
                from proto.node_service_pb2 import CollectTopologyRequest
            except ImportError:
                logger.debug(f"无法导入 CollectTopologyRequest protobuf 类型")
                return await self._collect_via_status(connector)
            
            # 创建正确的 protobuf 请求对象
            request = CollectTopologyRequest(
                max_depth=4,
                visited=[]
            )
            
            # 调用CollectTopology RPC
            response = await asyncio.wait_for(
                connector.stub.CollectTopology(request),
                timeout=30.0
            )
            
            # 解析响应
            if hasattr(response, 'topology_json'):
                import json
                return json.loads(response.topology_json)
            elif hasattr(response, 'nodes'):
                return {
                    "nodes": {n.id: n.capabilities for n in response.nodes},
                    "peer_graph": response.peer_graph
                }
            
            return None
            
        except AttributeError:
            # Stub方法不存在，尝试其他方式
            logger.debug(f"节点 {connector.node_info.node_id} 不支持CollectTopology RPC")
            return await self._collect_via_status(connector)
            
        except Exception as e:
            raise e
    
    async def _collect_via_status(self, connector) -> Optional[Dict]:
        """
        通过send_opaque_status间接收集拓扑信息
        
        备用方案：发送特殊的状态请求来获取拓扑
        """
        try:
            import sys
            import os
            
            # 构建拓扑请求
            topology_request = json.dumps({
                "type": "topology_request",
                "requester": "exo_manager",
                "timestamp": time.time()
            })
            
            # 发送请求（如果支持的话）
            if hasattr(connector.stub, 'SendOpaqueStatus'):
                try:
                    from proto.node_service_pb2 import OpaqueStatusRequest
                    
                    request = OpaqueStatusRequest(
                        request_id=f"topo_{int(time.time())}",
                        status=topology_request
                    )
                    
                    response = await asyncio.wait_for(
                        connector.stub.SendOpaqueStatus(request),
                        timeout=10.0
                    )
                    
                except (ImportError, AttributeError):
                    # 如果protobuf类型不可用，跳过
                    return None
                
                # 解析响应中的拓扑数据
                if hasattr(response, 'status') and response.status:
                    try:
                        data = json.loads(response.status)
                        if data.get("type") == "topology_response":
                            return data.get("topology")
                    except (json.JSONDecodeError, AttributeError):
                        pass
            
            return None
            
        except Exception as e:
            logger.debug(f"备用收集方式也失败: {e}")
            return None
    
    def _parse_topology_data(self, data: Dict, source_node: str = ""):
        """
        解析原始拓扑数据为内部结构
        
        Args:
            data: 原始拓扑字典（来自节点）
            source_node: 数据来源节点ID
        """
        # 解析节点
        raw_nodes = data.get("nodes", {})
        
        for node_id, caps in raw_nodes.items():
            if isinstance(caps, dict):
                capabilities = caps
            elif hasattr(caps, 'to_dict') or hasattr(caps, 'model_dump'):
                capabilities = caps.to_dict() if hasattr(caps, 'to_dict') else caps.model_dump()
            else:
                capabilities = {"model": str(caps)}
            
            # 合并manager中的额外信息
            manager_node = self.manager.nodes.get(node_id)
            
            node = TopologyNode(
                node_id=node_id,
                device_capabilities=capabilities,
                address=manager_node.address if manager_node else "",
                port=manager_node.port if manager_node else 0,
                status=manager_node.status.value if manager_node else "online",
                loaded_shards=manager_node.loaded_models if manager_node else [],
                role="seed" if node_id == source_node else "worker"
            )
            
            self.nodes[node_id] = node
        
        # 解析边（连接关系）
        raw_graph = data.get("peer_graph", {})
        
        for from_id, connections in raw_graph.items():
            for conn in connections:
                if isinstance(conn, dict):
                    edge = TopologyEdge(
                        from_node=from_id,
                        to_node=conn.get("to_id", ""),
                        description=conn.get("description", ""),
                        is_active=True
                    )
                else:
                    edge = TopologyEdge(
                        from_node=from_id,
                        to_node=str(conn.to_id),
                        description=conn.description if hasattr(conn, 'description') else "",
                        is_active=True
                    )
                
                # 避免重复添加
                if not any(e.from_node == edge.from_node and e.to_node == edge.to_node for e in self.edges):
                    self.edges.append(edge)
    
    def _build_basic_topology(self):
        """
        当无法从节点收集时，基于manager的节点列表构建基础拓扑
        
        这会创建一个全连接的网状拓扑（假设所有节点都互相知道）
        
        优化：只包含在线节点，离线节点不参与拓扑构建
        """
        # 只获取在线节点
        node_list = [
            node for node in self.manager.nodes.values()
            if node.status.value == 'online'
        ]
        
        if not node_list:
            logger.warning("⚠️ [P2P Topology] 所有节点均离线，跳过拓扑构建")
            return
        
        logger.info(f"📊 [P2P Topology] 构建基础拓扑: {len(node_list)} 个在线节点")
        
        for i, node in enumerate(node_list):
            topo_node = TopologyNode(
                node_id=node.node_id,
                device_capabilities=node.device_info,
                address=node.address,
                port=node.port,
                status=node.status.value,
                loaded_shards=node.loaded_models
            )
            self.nodes[node.node_id] = topo_node
            
            for j, other_node in enumerate(node_list):
                if i != j:
                    edge = TopologyEdge(
                        from_node=node.node_id,
                        to_node=other_node.node_id,
                        connection_type=ConnectionType.P2P_DIRECT,
                        description=f"P2P connection",
                        is_active=other_node.status.value == 'online'
                    )
                    
                    if not any(e.from_node == edge.from_node and e.to_node == edge.to_node for e in self.edges):
                        self.edges.append(edge)
    
    async def _analyze_inference_pipelines(self):
        """
        分析推理流水线
        
        基于节点的shard配置推断出模型推理时的节点协作顺序
        """
        self.inference_pipelines.clear()
        
        # 按模型分组收集shard信息
        model_shards: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
        
        for node_id, node in self.nodes.items():
            for shard in node.loaded_shards:
                model_id = shard.get("model_id", "unknown")
                model_shards[model_id].append((node_id, shard))
        
        # 为每个模型构建推理流水线
        for model_id, shards in model_shards.items():
            if len(shards) <= 1:
                continue  # 单节点模型不需要pipeline
            
            # 按start_layer排序
            sorted_shards = sorted(shards, key=lambda x: x[1].get("start_layer", 0))
            
            steps = []
            total_latency = 0.0
            bottleneck = None
            max_latency = 0.0
            
            for idx, (node_id, shard) in enumerate(sorted_shards):
                start = shard.get("start_layer", 0)
                end = shard.get("end_layer", 0)
                n_layers = end - start + 1
                
                # 估算每层延迟（简化计算）
                est_latency = n_layers * 50  # 假设每层50ms
                
                step = InferenceStep(
                    step_index=idx,
                    node_id=node_id,
                    shard_info=shard,
                    is_first_layer=(idx == 0),
                    is_last_layer=(idx == len(sorted_shards) - 1),
                    input_type="token_ids" if idx == 0 else "hidden_state",
                    output_type="logits" if idx == len(sorted_shards) - 1 else "hidden_state",
                    estimated_latency_ms=est_latency
                )
                
                steps.append(step)
                total_latency += est_latency
                
                if est_latency > max_latency:
                    max_latency = est_latency
                    bottleneck = node_id
            
            total_layers = shards[0][1].get("n_layers", sum(
                s[1].get("end_layer", 0) - s[1].get("start_layer", 0) + 1 
                for s in shards
            ))
            
            pipeline = InferencePipeline(
                model_id=model_id,
                total_layers=total_layers,
                steps=steps,
                pipeline_type="sequential",
                total_estimated_latency_ms=total_latency,
                bottleneck_node=bottleneck
            )
            
            self.inference_pipelines[model_id] = pipeline
            logger.info(f"📊 分析出推理流水线: {model_id} ({len(steps)} 个节点)")
    
    def get_topology_summary(self) -> Dict[str, Any]:
        """
        获取拓扑摘要信息
        
        Returns:
            包含统计信息和拓扑数据的字典
        """
        online_nodes = [n for n in self.nodes.values() if n.status == 'online']
        active_edges = [e for e in self.edges if e.is_active]
        
        # 计算图的属性
        avg_degree = 0
        if self.nodes:
            total_degree = sum(1 for e in self.edges if e.to_node in self.nodes)
            node_count = len(self.nodes)
            avg_degree = total_degree / node_count if node_count > 0 else 0
        
        # 识别关键节点（连接数最多的）
        node_connection_count = defaultdict(int)
        for edge in self.edges:
            node_connection_count[edge.from_node] += 1
        
        top_connected = sorted(
            node_connection_count.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        return {
            "timestamp": self.last_collection_time,
            "statistics": {
                "total_nodes": len(self.nodes),
                "online_nodes": len(online_nodes),
                "total_edges": len(self.edges),
                "active_edges": len(active_edges),
                "avg_connections_per_node": round(avg_degree, 2),
                "models_with_pipeline": len(self.inference_pipelines)
            },
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "inference_pipelines": {
                model_id: p.to_dict() 
                for model_id, p in self.inference_pipelines.items()
            },
            "key_nodes": {
                "most_connected": [
                    {"node_id": nid, "connections": count} 
                    for nid, count in top_connected
                ]
            },
            "collection_info": {
                "last_collection": self.last_collection_time,
                "history_entries": len(self.collection_history)
            }
        }
    
    def get_inference_pipeline(self, model_id: str) -> Optional[Dict]:
        """
        获取指定模型的推理流水线
        
        Args:
            model_id: 模型标识符
            
        Returns:
            流水线数据或None
        """
        pipeline = self.inference_pipelines.get(model_id)
        return pipeline.to_dict() if pipeline else None
    
    def get_node_neighbors(self, node_id: str) -> List[Dict]:
        """
        获取节点的所有邻居（直接相连的节点）
        
        Args:
            node_id: 节点ID
            
        Returns:
            邻居节点列表
        """
        neighbors = []
        
        for edge in self.edges:
            if edge.from_node == node_id and edge.is_active:
                neighbor_node = self.nodes.get(edge.to_node)
                if neighbor_node:
                    neighbors.append({
                        "node_id": edge.to_node,
                        "connection_type": edge.connection_type.value,
                        "latency_ms": edge.latency_ms,
                        "device": neighbor_node.device_capabilities.get("chip", "Unknown"),
                        "status": neighbor_node.status
                    })
        
        return neighbors
    
    def find_shortest_path(self, from_node: str, to_node: str) -> Optional[List[str]]:
        """
        使用BFS查找两个节点间的最短路径
        
        Args:
            from_node: 起始节点
            to_node: 目标节点
            
        Returns:
            节点路径列表或None（如果不可达）
        """
        if from_node not in self.nodes or to_node not in self.nodes:
            return None
        
        if from_node == to_node:
            return [from_node]
        
        # 构建邻接表
        adj = defaultdict(set)
        for edge in self.edges:
            if edge.is_active:
                adj[edge.from_node].add(edge.to_node)
        
        # BFS
        queue = [(from_node, [from_node])]
        visited = {from_node}
        
        while queue:
            current, path = queue.pop(0)
            
            for neighbor in adj[current]:
                if neighbor == to_node:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # 不可达
    
    def detect_anomalies(self) -> List[Dict]:
        """
        检测拓扑异常
        
        包括：
        - 孤立节点（无任何连接）
        - 高延迟连接
        - 可能的网络分区
        
        Returns:
            异常列表
        """
        anomalies = []
        
        # 检查孤立节点
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.from_node)
            connected_nodes.add(edge.to_node)
        
        isolated = set(self.nodes.keys()) - connected_nodes
        for node_id in isolated:
            node = self.nodes[node_id]
            anomalies.append({
                "type": "isolated_node",
                "severity": "warning",
                "node_id": node_id,
                "message": f"节点 {node_id} 没有任何P2P连接"
            })
        
        # 检查离线节点但有活跃边
        for edge in self.edges:
            target_node = self.nodes.get(edge.to_node)
            if target_node and target_node.status != 'online' and edge.is_active:
                anomalies.append({
                    "type": "unreachable_peer",
                    "severity": "error",
                    "from_node": edge.from_node,
                    "to_node": edge.to_node,
                    "message": f"节点 {edge.from_node} 到 {edge.to_node} 的连接可能已断开"
                })
        
        # 检查可能的分区（使用简化的连通性检查）
        if len(self.nodes) > 1:
            adj = defaultdict(set)
            for edge in self.edges:
                if edge.is_active:
                    adj[edge.from_node].add(edge.to_node)
                    adj[edge.to_node].add(edge.from_node)  # 无向图
            
            # 找连通分量
            visited = set()
            components = []
            
            for node in self.nodes:
                if node not in visited:
                    component = []
                    stack = [node]
                    while stack:
                        current = stack.pop()
                        if current not in visited:
                            visited.add(current)
                            component.append(current)
                            for neighbor in adj[current]:
                                if neighbor not in visited:
                                    stack.append(neighbor)
                    components.append(component)
            
            if len(components) > 1:
                anomalies.append({
                    "type": "network_partition",
                    "severity": "critical",
                    "components": components,
                    "message": f"检测到网络分区！网络分裂成 {len(components)} 个独立部分"
                })
        
        return anomalies
    
    def get_visualization_data(self, layout: str = "force") -> Dict:
        """
        生成用于前端可视化的数据
        
        Args:
            layout: 布局算法 ('force', 'circular', 'hierarchical')
            
        Returns:
            可视化所需的节点和边数据
        """
        nodes_viz = []
        edges_viz = []
        
        # 准备节点数据
        for idx, (node_id, node) in enumerate(self.nodes.items()):
            # 根据角色/状态确定颜色
            if node.status != 'online':
                color = "#ef4444"  # red
            elif node.role == 'seed':
                color = "#8b5cf6"  # purple
            else:
                color = "#3b82f6"  # blue
            
            # 大小根据连接数调整
            connection_count = sum(1 for e in self.edges if e.from_node == node_id or e.to_node == node_id)
            size = 20 + min(connection_count * 5, 30)
            
            nodes_viz.append({
                "id": node_id,
                "label": node_id,
                "size": size,
                "color": color,
                "device": node.device_capabilities.get("chip", "N/A")[:20],
                "memory_gb": round(node.device_capabilities.get("memory", 0) / 1024, 1),
                "status": node.status,
                "x": 0,  # 将由前端布局算法填充
                "y": 0
            })
        
        # 准备边数据
        seen_edges = set()
        for edge in self.edges:
            # 避免重复（双向边只显示一次）
            edge_key = tuple(sorted([edge.from_node, edge.to_node]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            
            # 样式
            width = 2 if edge.is_active else 1
            dashes = not edge.is_active
            color = "#94a3b8" if edge.is_active else "#cbd5e1"
            
            # 推理链路高亮
            for pipeline in self.inference_pipelines.values():
                step_nodes = [s.node_id for s in pipeline.steps]
                if edge.from_node in step_nodes and edge.to_node in step_nodes:
                    idx_from = step_nodes.index(edge.from_node) if edge.from_node in step_nodes else -1
                    idx_to = step_nodes.index(edge.to_node) if edge.to_node in step_nodes else -1
                    if abs(idx_from - idx_to) == 1:  # 相邻步骤
                        color = "#f59e0b"  # amber
                        width = 3
                        break
            
            edges_viz.append({
                "source": edge.from_node,
                "target": edge.to_node,
                "width": width,
                "color": color,
                "dashes": dashes,
                "label": edge.description[:30] if edge.description else "",
                "type": edge.connection_type.value
            })
        
        return {
            "nodes": nodes_viz,
            "edges": edges_viz,
            "layout": layout,
            "metadata": {
                "total_nodes": len(nodes_viz),
                "total_edges": len(edges_viz),
                "has_inference_pipeline": len(self.inference_pipelines) > 0
            }
        }


__all__ = [
    'P2PTopologyManager',
    'TopologyNode',
    'TopologyEdge', 
    'InferenceStep',
    'InferencePipeline',
    'ConnectionType'
]
