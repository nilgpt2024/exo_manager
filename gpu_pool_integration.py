"""
EXO Manager - GPU池管理集成
============================

将GPU池管理功能集成到Manager系统中，提供：
- 统一的模型加载/卸载接口
- 智能分片分配策略
- 资源预览与优化建议

使用方式:
    from gpu_pool_integration import GPUPoolIntegration
    
    pool = GPUPoolIntegration(cluster_manager)
    await pool.load_model("Qwen/Qwen3-4B", "./models")
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelAllocation:
    """模型分片分配方案"""
    model_id: str
    total_layers: int
    allocations: List[Dict[str, Any]]  # [{node_id, start_layer, end_layer, layers_count}]
    strategy: str
    estimated_memory_per_node: Dict[str, float]  # {node_id: memory_gb}


class GPUPoolIntegration:
    """
    GPU池管理集成
    
    将GPU池管理功能与集群管理系统结合，
    提供统一的模型管理和资源调配能力。
    """
    
    def __init__(self, cluster_manager):
        """
        初始化GPU池管理
        
        Args:
            cluster_manager: EXOClusterManager实例
        """
        self.manager = cluster_manager
        self.loaded_models: Dict[str, ModelAllocation] = {}
        
        logger.info("🎮 GPU池管理集成已初始化")
    
    def get_available_nodes(self) -> List[Dict]:
        """
        获取所有可用的在线节点及其资源信息
        
        使用多维度判断节点可用性：
        1. status == online (理想状态)
        2. 有有效设备信息且最近有心跳 (CollectTopology 可达)
        3. 非 OFFLINE 状态且有连接器
        
        Returns:
            节点列表，包含内存、设备等信息
        """
        import time as _time
        available = []
        
        for node_id, node_info in self.manager.nodes.items():
            is_available = False
            reason = ""
            
            if node_info.status.value == 'online':
                is_available = True
                reason = "status=online"
            else:
                has_device_info = bool(node_info.device_info and node_info.device_info.get("chip"))
                has_recent_heartbeat = node_info.last_heartbeat > 0 and (_time.time() - node_info.last_heartbeat < 60)
                is_not_offline = node_info.status.value != 'offline'
                
                if has_device_info and has_recent_heartbeat and is_not_offline:
                    is_available = True
                    reason = f"device_ok+heartbeat(status={node_info.status.value})"
            
            if is_available:
                raw_mem = node_info.device_info.get("memory", 0)
                mem_detail = node_info.device_info.get("memory_detail")
                effective_mem = raw_mem
                if (effective_mem == 0 or effective_mem is None) and mem_detail and mem_detail.get("total", 0) > 0:
                    effective_mem = mem_detail["total"]
                
                node_data = {
                    "node_id": node_id,
                    "address": f"{node_info.address}:{node_info.port}",
                    "memory_mb": effective_mem,
                    "device": node_info.device_info.get("chip", "Unknown"),
                    "flops": node_info.device_info.get("flops", {}),
                    "loaded_models": len(node_info.loaded_models),
                    "_status": node_info.status.value,
                    "_reason": reason
                }
                available.append(node_data)
                
                logger.debug(f"🎮 [GPU Pool] 节点 {node_id} 可用: {reason}, "
                           f"显存={effective_mem}MB, 模型={len(node_info.loaded_models)}个")
            else:
                logger.warning(f"⚠️ [GPU Pool] 节点 {node_id} 不可用: "
                             f"status={node_info.status.value}, "
                             f"has_device={bool(node_info.device_info.get('chip'))}, "
                             f"heartbeat_age={_time.time() - node_info.last_heartbeat:.0f}s" 
                             if node_info.last_heartbeat > 0 else "no_heartbeat")
        
        logger.info(f"🎮 [GPU Pool] 可用节点: {len(available)}/{len(self.manager.nodes)}")
        return available
    
    async def preview_allocation(
        self,
        model_id: str,
        total_layers: int,
        layer_memory_mb: float = 100.0,
        strategy: str = "memory_weighted"
    ) -> ModelAllocation:
        """
        预览模型分片分配方案
        
        Args:
            model_id: 模型标识符
            total_layers: 总层数
            layer_memory_mb: 每层预估显存占用 (MB)
            strategy: 分配策略 ('memory_weighted', 'uniform', 'performance_weighted')
            
        Returns:
            分配方案对象
        """
        nodes = self.get_available_nodes()
        
        if not nodes:
            raise ValueError("没有可用的在线节点")
        
        if strategy == "memory_weighted":
            allocations = self._allocate_by_memory(nodes, total_layers, layer_memory_mb)
        elif strategy == "uniform":
            allocations = self._allocate_uniformly(nodes, total_layers)
        elif strategy == "performance_weighted":
            allocations = self._allocate_by_performance(nodes, total_layers, layer_memory_mb)
        else:
            raise ValueError(f"未知的分配策略: {strategy}")
        
        # 计算每个节点的预估内存使用
        estimated_memory = {}
        for alloc in allocations:
            node_id = alloc["node_id"]
            layers = alloc["layers_count"]
            estimated_memory[node_id] = round(layers * layer_memory_mb / 1024, 2)  # GB
        
        allocation = ModelAllocation(
            model_id=model_id,
            total_layers=total_layers,
            allocations=allocations,
            strategy=strategy,
            estimated_memory_per_node=estimated_memory
        )
        
        logger.info(f"📊 预览分配方案 - {model_id}: {len(allocations)} 个节点")
        
        return allocation
    
    def _allocate_by_memory(
        self,
        nodes: List[Dict],
        total_layers: int,
        layer_memory_mb: float
    ) -> List[Dict]:
        """
        基于内存容量的加权分配
        
        内存越大的节点分配越多层
        """
        # 计算总内存权重
        total_memory = sum(n["memory_mb"] for n in nodes)

        if total_memory <= 0:
            return self._allocate_uniformly(nodes, total_layers)

        allocations = []
        current_layer = 0
        
        for node in sorted(nodes, key=lambda x: x["memory_mb"], reverse=True):
            if current_layer >= total_layers:
                break
            
            # 根据内存比例计算应分配的层数
            weight = node["memory_mb"] / total_memory
            layers_for_node = max(1, int(total_layers * weight))
            
            # 确保不超过剩余层数
            layers_for_node = min(layers_for_node, total_layers - current_layer)
            
            end_layer = current_layer + layers_for_node - 1
            
            allocations.append({
                "node_id": node["node_id"],
                "start_layer": current_layer,
                "end_layer": end_layer,
                "layers_count": layers_for_node,
                "address": node["address"]
            })
            
            current_layer = end_layer + 1
        
        # 处理剩余层（如果有）
        if current_layer < total_layers and allocations:
            last_alloc = allocations[-1]
            remaining = total_layers - current_layer
            last_alloc["end_layer"] = total_layers - 1
            last_alloc["layers_count"] += remaining
        
        return allocations
    
    def _allocate_uniformly(
        self,
        nodes: List[Dict],
        total_layers: int
    ) -> List[Dict]:
        """
        均匀分配 - 每个节点分配相同数量的层
        """
        num_nodes = len(nodes)
        base_layers = total_layers // num_nodes
        remainder = total_layers % num_nodes
        
        allocations = []
        current_layer = 0
        
        for i, node in enumerate(nodes):
            # 前remainder个节点多分配一层
            layers_for_node = base_layers + (1 if i < remainder else 0)
            
            end_layer = current_layer + layers_for_node - 1
            
            allocations.append({
                "node_id": node["node_id"],
                "start_layer": current_layer,
                "end_layer": end_layer,
                "layers_count": layers_for_node,
                "address": node["address"]
            })
            
            current_layer = end_layer + 1
        
        return allocations
    
    def _allocate_by_performance(
        self,
        nodes: List[Dict],
        total_layers: int,
        layer_memory_mb: float
    ) -> List[Dict]:
        """
        基于性能（FLOPS）的加权分配
        
        计算性能/内存比来优化分配
        """
        scored_nodes = []
        
        for node in nodes:
            flops = node.get("flops", {})
            fp16_flops = flops.get("fp16", 10.0)  # 默认值
            memory_gb = node["memory_mb"] / 1024
            
            # 性能得分：FLOPS越高越好，但要考虑已有负载
            load_penalty = node["loaded_models"] * 2  # 每个已加载模型扣分
            
            score = fp16_flops - load_penalty
            scored_nodes.append({**node, "score": score})
        
        # 按得分排序
        scored_nodes.sort(key=lambda x: x["score"], reverse=True)
        
        # 使用类似内存加权的分配逻辑
        total_score = sum(n["score"] for n in scored_nodes)
        
        allocations = []
        current_layer = 0
        
        for node in scored_nodes:
            if current_layer >= total_layers:
                break
            
            weight = node["score"] / total_score
            layers_for_node = max(1, int(total_layers * weight))
            layers_for_node = min(layers_for_node, total_layers - current_layer)
            
            end_layer = current_layer + layers_for_node - 1
            
            allocations.append({
                "node_id": node["node_id"],
                "start_layer": current_layer,
                "end_layer": end_layer,
                "layers_count": layers_for_node,
                "address": node["address"]
            })
            
            current_layer = end_layer + 1
        
        return allocations
    
    async def execute_allocation(
        self,
        allocation: ModelAllocation,
        model_path: str,
        repo_id: str = ""
    ) -> Dict[str, Any]:
        """
        执行模型分配方案
        
        将计算好的分片方案实际应用到各节点
        
        Args:
            allocation: 预览得到的分配方案
            model_path: 模型文件路径
            repo_id: HuggingFace仓库ID（可选）
            
        Returns:
            执行结果
        """
        results = {
            "success": True,
            "model_id": allocation.model_id,
            "allocations_executed": [],
            "errors": [],
            "total_time_ms": 0
        }
        
        start_time = asyncio.get_event_loop().time()
        
        for alloc in allocation.allocations:
            node_id = alloc["node_id"]
            
            try:
                connector = self.manager.connectors.get(node_id)
                
                if not connector or connector.node_info.status.value != 'online':
                    error_msg = f"节点 {node_id} 不可用"
                    results["errors"].append(error_msg)
                    continue
                
                # TODO: 实际的gRPC调用将分片配置发送给节点
                # 这里需要实现具体的远程调用逻辑
                
                shard_config = {
                    "model_id": allocation.model_id,
                    "start_layer": alloc["start_layer"],
                    "end_layer": alloc["end_layer"],
                    "n_layers": allocation.total_layers,
                    "repo_id": repo_id,
                    "model_path": model_path
                }
                
                # 模拟执行成功
                exec_result = {
                    "node_id": node_id,
                    "shard": shard_config,
                    "status": "scheduled",
                    "message": f"已调度 {alloc['layers_count']} 层 (L{alloc['start_layer']}-L{alloc['end_layer']})"
                }
                
                results["allocations_executed"].append(exec_result)
                
                # 更新节点的loaded_models
                if node_id in self.manager.nodes:
                    self.manager.nodes[node_id].loaded_models.append({
                        "model_id": allocation.model_id,
                        "shard": shard_config
                    })
                
                logger.info(f"✅ {node_id}: 已分配 L{alloc['start_layer']}-L{alloc['end_layer']}")
                
            except Exception as e:
                error_msg = f"{node_id}: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(f"❌ {error_msg}")
        
        end_time = asyncio.get_event_loop().time()
        results["total_time_ms"] = round((end_time - start_time) * 1000, 2)
        
        if results["errors"]:
            results["success"] = False
            results["warning"] = "部分节点分配失败"
        
        # 缓存已加载的模型
        if results["success"]:
            self.loaded_models[allocation.model_id] = allocation
        
        return results
    
    async def unload_model(self, model_id: str) -> bool:
        """
        卸载指定模型
        
        从所有节点上卸载该模型的分片
        
        Args:
            model_id: 要卸载的模型标识符
            
        Returns:
            是否全部卸载成功
        """
        if model_id not in self.loaded_models:
            logger.warning(f"模型 {model_id} 未在当前池中找到")
            return True
        
        success = True
        
        for node_id, node_info in self.manager.nodes.items():
            try:
                # 移除该节点的模型记录
                original_len = len(node_info.loaded_models)
                node_info.loaded_models = [
                    m for m in node_info.loaded_models 
                    if m.get("model_id") != model_id
                ]
                
                removed = original_len - len(node_info.loaded_models)
                
                if removed > 0:
                    logger.info(f"🗑️ {node_id}: 已移除模型 {model_id}")
                    
                    # TODO: 实际调用gRPC通知节点卸载
                    
            except Exception as e:
                logger.error(f"❌ 卸载失败 {node_id}: {e}")
                success = False
        
        if success and model_id in self.loaded_models:
            del self.loaded_models[model_id]
            logger.info(f"✅ 模型 {model_id} 已完全卸载")
        
        return success
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        获取GPU池的综合状态
        
        Returns:
            包含资源使用、已加载模型等信息的字典
        """
        nodes = self.get_available_nodes()
        
        total_memory = sum(n["memory_mb"] for n in nodes) / 1024  # GB
        
        loaded_models_summary = {}
        for model_id, allocation in self.loaded_models.items():
            loaded_models_summary[model_id] = {
                "nodes_used": [a["node_id"] for a in allocation.allocations],
                "total_shards": len(allocation.allocations),
                "strategy": allocation.strategy
            }
        
        return {
            "pool_name": "EXO Unified GPU Pool",
            "available_nodes": len(nodes),
            "total_memory_gb": round(total_memory, 2),
            "loaded_models": loaded_models_summary,
            "node_details": [
                {
                    **n,
                    "current_loads": n["loaded_models"]
                } for n in nodes
            ]
        }


__all__ = ['GPUPoolIntegration', 'ModelAllocation']
