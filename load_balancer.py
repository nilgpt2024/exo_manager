"""
多实例负载均衡器 - 支持多种负载均衡策略

功能:
1. 自动发现同一模型的所有已加载实例
2. 支持多种负载均衡策略（轮询/随机/加权/最少连接）
3. 实例健康检查和自动剔除故障节点
4. 推理统计和监控数据收集
5. 支持手动指定实例或自动选择

使用示例:
    from load_balancer import LoadBalancer
    
    lb = LoadBalancer(manager)
    
    # 使用轮询策略选择实例
    node = lb.select_instance("qwen3-0.6b", strategy="round_robin")
    
    # 记录推理完成
    lb.record_completion(node.node_id, "qwen3-0.6b", success=True, latency=1.23)
"""

import time
import random
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import asyncio

logger = logging.getLogger(__name__)


class LBStrategy(str, Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"          # 轮询
    RANDOM = "random"                    # 随机
    WEIGHTED = "weighted"                # 加权轮询
    LEAST_CONNECTIONS = "least_connections"  # 最少连接
    FIRST_LAYER = "first_layer"          # 首层优先（默认）


@dataclass
class InstanceInfo:
    """实例信息"""
    instance_id: str                     # 实例ID (如 "worker-1")
    full_model_id: str                   # 完整模型ID (如 "qwen3-0.6b::worker-1")
    base_model_id: str                   # 基础模型ID (如 "qwen3-0.6b")
    node_id: str                         # 运行该实例的节点ID
    start_layer: int                     # 分片起始层
    end_layer: int                       # 分片结束层
    is_first_layer: bool                 # 是否包含第一层
    
    # 统计信息
    total_requests: int = 0              # 总请求数
    success_requests: int = 0            # 成功请求数
    failed_requests: int = 0             # 失败请求数
    total_latency: float = 0.0           # 累计延迟(ms)
    last_request_time: Optional[float] = None  # 最后请求时间
    current_connections: int = 0         # 当前连接数
    
    # 权重信息
    weight: float = 1.0                  # 实例权重
    
    @property
    def avg_latency(self) -> float:
        """平均延迟"""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 100.0
        return (self.success_requests / self.total_requests) * 100
    
    @property
    def health_score(self) -> float:
        """健康评分 (0-100)"""
        score = 100.0
        
        # 基于成功率扣分
        score -= (100 - self.success_rate) * 0.5
        
        # 基于延迟扣分 (超过500ms开始扣分)
        if self.avg_latency > 500:
            score -= min(30, (self.avg_latency - 500) / 50)
        
        # 基于失败率扣分
        if self.failed_requests > 5 and self.total_requests > 10:
            fail_rate = self.failed_requests / self.total_requests
            if fail_rate > 0.1:
                score -= fail_rate * 100
        
        return max(0, min(100, score))


@dataclass
class SelectionResult:
    """选择结果"""
    selected_node_id: str               # 选中的节点ID
    selected_instance: InstanceInfo     # 实例信息
    strategy_used: LBStrategy           # 使用的策略
    selection_reason: str               # 选择原因
    available_instances: int            # 可用实例总数
    timestamp: float                    # 选择时间戳


class LoadBalancer:
    """
    多实例负载均衡器
    
    功能:
    - 自动发现同一模型的所有实例
    - 支持多种负载均衡策略
    - 实例健康管理和统计
    """
    
    def __init__(self, manager=None):
        """
        初始化负载均衡器
        
        Args:
            manager: EXOClusterManager 实例
        """
        self.manager = manager
        
        # 实例注册表: {base_model_id: {instance_id: InstanceInfo}}
        self.instances: Dict[str, Dict[str, InstanceInfo]] = {}
        
        # 轮询计数器: {base_model_id: int}
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        
        # 锁
        self._lock = threading.RLock()
        
        # 统计信息
        self.total_selections: int = 0
        self.strategy_stats: Dict[LBStrategy, int] = defaultdict(int)
        
        logger.info("[LoadBalancer] 初始化完成，支持策略: {[s.value for s in LBStrategy]}")
    
    def discover_instances(self) -> Dict[str, List[InstanceInfo]]:
        """
        发现所有已加载的实例
        
        使用与 /api/models/instances 相同的数据源（manager.nodes），
        确保数据一致性。
        
        Returns:
            {base_model_id: [InstanceInfo]}
        """
        if not self.manager:
            return {}
        
        discovered = {}
        
        with self._lock:
            # 方案1: 优先使用 manager.get_model_instances() 方法（如果可用）
            if hasattr(self.manager, 'get_model_instances'):
                try:
                    # 获取所有基础模型的实例摘要
                    if hasattr(self.manager, 'get_all_instances_summary'):
                        summary = self.manager.get_all_instances_summary()
                        
                        for base_model_id in summary.keys():
                            instances = self.manager.get_model_instances(base_model_id)
                            
                            for inst_data in instances:
                                instance_info = InstanceInfo(
                                    instance_id=inst_data.get("instance_id", "default"),
                                    full_model_id=inst_data.get("full_model_id", base_model_id),
                                    base_model_id=base_model_id,
                                    node_id=inst_data.get("node_id", "unknown"),
                                    start_layer=inst_data.get("shard", {}).get("start_layer", 0),
                                    end_layer=inst_data.get("shard", {}).get("end_layer", 0),
                                    is_first_layer=(inst_data.get("shard", {}).get("start_layer", 0) == 0)
                                )
                                
                                if base_model_id not in discovered:
                                    discovered[base_model_id] = []
                                discovered[base_model_id].append(instance_info)
                                
                                # 更新内部注册表
                                if base_model_id not in self.instances:
                                    self.instances[base_model_id] = {}
                                self.instances[base_model_id][instance_info.instance_id] = instance_info
                        
                        logger.info(f"[LoadBalancer] 通过 get_model_instances 发现 "
                                   f"{sum(len(v) for v in discovered.values())} 个实例，"
                                   f"涵盖 {len(discovered)} 个模型")
                        
                        return discovered
                except Exception as e:
                    logger.warning(f"[LoadBalancer] get_model_instances 方法调用失败: {e}，回退到 nodes 遍历")
            
            # 按 (node_id, base_model_id) 粒度去重
            # - 同一节点的同一基础模型只保留一个实例（单节点多分片场景）
            # - 不同节点的同模型各自独立（多节点多副本场景，支持负载均衡）
            seen_node_base = set()  # {(node_id, base_id), ...}
            for node_id, node_info in self.manager.nodes.items():
                for model in node_info.loaded_models:
                    model_id = model.get("model_id", "unknown")
                    
                    # 解析基础模型ID和实例ID
                    if "::" in model_id:
                        parts = model_id.split("::")
                        base_id = parts[0]
                        inst_id = parts[1]
                    else:
                        base_id = model_id
                        inst_id = "default"
                    
                    # 按 (node_id, base_model_id) 去重：同一节点的同模型只保留一个入口
                    node_base_key = (node_id, base_id)
                    if node_base_key in seen_node_base:
                        continue
                    seen_node_base.add(node_base_key)
                    
                    shard = model.get("shard", {})
                    start_layer = shard.get("start_layer", 0)
                    end_layer = shard.get("end_layer", 0)
                    
                    instance_info = InstanceInfo(
                        instance_id=inst_id,
                        full_model_id=model_id,
                        base_model_id=base_id,
                        node_id=node_id,
                        start_layer=start_layer,
                        end_layer=end_layer,
                        is_first_layer=(start_layer == 0)
                    )
                    
                    if base_id not in discovered:
                        discovered[base_id] = []
                    discovered[base_id].append(instance_info)
                    
                    # 更新内部注册表
                    if base_id not in self.instances:
                        self.instances[base_id] = {}
                    self.instances[base_id][inst_id] = instance_info
            
            logger.info(f"[LoadBalancer] 通过 nodes 遍历发现 "
                       f"{sum(len(v) for v in discovered.values())} 个实例，"
                       f"涵盖 {len(discovered)} 个模型")
            
            return discovered
    
    def get_available_instances(self, model_id: str, force_refresh: bool = False) -> List[InstanceInfo]:
        """
        获取指定模型的所有可用实例
        
        Args:
            model_id: 模型ID（可以是完整ID或基础ID）
            force_refresh: 是否强制刷新缓存（默认False，使用缓存）
            
        Returns:
            可用实例列表
        """
        # 提取基础模型ID
        base_id = model_id.split("::")[0] if "::" in model_id else model_id
        
        with self._lock:
            # 如果不强制刷新，先尝试从缓存获取
            if not force_refresh:
                if base_id in self.instances:
                    instances = list(self.instances[base_id].values())
                    if instances:
                        logger.debug(f"[LoadBalancer] 从缓存返回 {len(instances)} 个实例 (模型: {base_id})")
                        return instances
            
            # 强制刷新或缓存为空时，重新发现
            logger.info(f"[LoadBalancer] 正在{'强制' if force_refresh else ''}刷新实例列表 (模型: {base_id})")
            discovered = self.discover_instances()
            
            if base_id in discovered:
                instances = discovered[base_id]
                
                # 过滤掉不健康的实例（健康分数 < 30）
                healthy_instances = [
                    inst for inst in instances 
                    if inst.health_score >= 30 or inst.total_requests == 0
                ]
                
                result = healthy_instances if healthy_instances else instances
                logger.info(f"[LoadBalancer] 返回 {len(result)} 个可用实例 (模型: {base_id})")
                return result
            
            logger.warning(f"[LoadBalancer] 未找到任何实例 (模型: {base_id})")
            return []
    
    def select_instance(
        self,
        model_id: str,
        strategy: LBStrategy = LBStrategy.FIRST_LAYER,
        preferred_instance: Optional[str] = None,
        exclude_nodes: Optional[List[str]] = None
    ) -> Optional[SelectionResult]:
        """
        选择一个实例进行推理
        
        Args:
            model_id: 模型ID
            strategy: 负载均衡策略
            preferred_instance: 优先使用的实例ID（如果指定，直接返回该实例）
            exclude_nodes: 要排除的节点列表
            
        Returns:
            SelectionResult 或 None（如果没有可用实例）
        """
        start_time = time.time()
        
        # 如果指定了特定实例
        if preferred_instance:
            instances = [inst for inst in self.get_available_instances(model_id)
                        if inst.instance_id == preferred_instance]
            if instances:
                selected = instances[0]
                selected.total_requests += 1
                selected.current_connections += 1
                selected.last_request_time = time.time()
                
                self.total_selections += 1
                
                return SelectionResult(
                    selected_node_id=selected.node_id,
                    selected_instance=selected,
                    strategy_used=strategy,
                    selection_reason=f"用户指定实例: {preferred_instance}",
                    available_instances=len(instances),
                    timestamp=time.time()
                )
        
        # 获取所有可用实例
        instances = self.get_available_instances(model_id)
        
        # 排除指定节点
        if exclude_nodes:
            instances = [inst for inst in instances if inst.node_id not in exclude_nodes]
        
        if not instances:
            logger.warning(f"[LoadBalancer] 没有可用的实例: {model_id}")
            return None
        
        # 根据策略选择实例
        selected = self._apply_strategy(instances, strategy, model_id)
        
        if not selected:
            # 回退到默认策略
            logger.warning(f"[LoadBalancer] 策略 {strategy.value} 失败，回退到 first_layer")
            selected = self._apply_strategy(instances, LBStrategy.FIRST_LAYER, model_id)
        
        if selected:
            # 更新统计
            selected.total_requests += 1
            selected.current_connections += 1
            selected.last_request_time = time.time()
            
            self.total_selections += 1
            self.strategy_stats[strategy] += 1
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            result = SelectionResult(
                selected_node_id=selected.node_id,
                selected_instance=selected,
                strategy_used=strategy,
                selection_reason=self._get_strategy_description(strategy, selected),
                available_instances=len(instances),
                timestamp=time.time()
            )
            
            logger.info(f"[LoadBalancer] 选择完成: {result.selected_node_id} "
                       f"(实例={selected.instance_id}, 策略={strategy.value}, "
                       f"耗时={elapsed_ms:.1f}ms)")
            
            return result
        
        return None
    
    def _apply_strategy(
        self,
        instances: List[InstanceInfo],
        strategy: LBStrategy,
        model_id: str
    ) -> Optional[InstanceInfo]:
        """应用负载均衡策略"""
        
        if strategy == LBStrategy.FIRST_LAYER:
            # 首层优先：选择包含第一层的实例
            first_layer_insts = [inst for inst in instances if inst.is_first_layer]
            if first_layer_insts:
                return random.choice(first_layer_insts)
            # 如果没有首层实例，随机选一个
            return random.choice(instances) if instances else None
        
        elif strategy == LBStrategy.ROUND_ROBIN:
            # 轮询
            base_id = model_id.split("::")[0] if "::" in model_id else model_id
            
            with self._lock:
                counter = self.round_robin_counters.get(base_id, 0)
                selected_index = counter % len(instances)
                self.round_robin_counters[base_id] = counter + 1
            
            return instances[selected_index]
        
        elif strategy == LBStrategy.RANDOM:
            # 随机
            return random.choice(instances)
        
        elif strategy == LBStrategy.WEIGHTED:
            # 加权随机
            weights = [inst.weight * inst.health_score / 100.0 for inst in instances]
            total_weight = sum(weights)
            
            if total_weight <= 0:
                return random.choice(instances)
            
            normalized_weights = [w / total_weight for w in weights]
            r = random.random()
            cumulative = 0.0
            
            for i, weight in enumerate(normalized_weights):
                cumulative += weight
                if r <= cumulative:
                    return instances[i]
            
            return instances[-1]
        
        elif strategy == LBStrategy.LEAST_CONNECTIONS:
            # 最少连接
            return min(instances, key=lambda x: x.current_connections)
        
        else:
            # 默认使用首层优先
            return self._apply_strategy(instances, LBStrategy.FIRST_LAYER, model_id)
    
    def _get_strategy_description(self, strategy: LBStrategy, instance: InstanceInfo) -> str:
        """获取策略描述"""
        descriptions = {
            LBStrategy.FIRST_LAYER: f"首层优先 (layer {instance.start_layer}-{instance.end_layer})",
            LBStrategy.ROUND_ROBIN: f"轮询 (权重={instance.weight})",
            LBStrategy.RANDOM: "随机选择",
            LBStrategy.WEIGHTED: f"加权选择 (权重={instance.weight}, 健康={instance.health_score:.1f})",
            LBStrategy.LEAST_CONNECTIONS: f"最少连接 (当前={instance.current_connections})"
        }
        return descriptions.get(strategy, "未知策略")
    
    def record_completion(
        self,
        node_id: str,
        model_id: str,
        success: bool = True,
        latency: float = 0.0,
        tokens_generated: int = 0,
        error_message: str = ""
    ):
        """
        记录推理完成
        
        Args:
            node_id: 节点ID
            model_id: 模型ID
            success: 是否成功
            latency: 延迟（毫秒）
            tokens_generated: 生成的token数
            error_message: 错误信息
        """
        base_id = model_id.split("::")[0] if "::" in model_id else model_id
        
        with self._lock:
            if base_id in self.instances:
                for inst in self.instances[base_id].values():
                    if inst.node_id == node_id:
                        inst.current_connections = max(0, inst.current_connections - 1)
                        
                        if success:
                            inst.success_requests += 1
                            inst.total_latency += latency
                        else:
                            inst.failed_requests += 1
                        
                        break
    
    def get_statistics(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            model_id: 模型ID（可选，为None则返回所有）
            
        Returns:
            统计信息字典
        """
        stats = {
            "total_models": 0,
            "total_instances": 0,
            "total_selections": self.total_selections,
            "strategy_distribution": {k.value: v for k, v in self.strategy_stats.items()},
            "models": {}
        }
        
        instances_to_report = {}
        
        if model_id:
            base_id = model_id.split("::")[0] if "::" in model_id else model_id
            instances_to_report[base_id] = self.get_available_instances(model_id, force_refresh=True)
        else:
            all_discovered = self.discover_instances()
            instances_to_report = all_discovered
        
        for base_id, instances in instances_to_report.items():
            if not instances:
                continue
            
            stats["total_models"] += 1
            stats["total_instances"] += len(instances)
            
            model_stats = {
                "base_model_id": base_id,
                "instances_count": len(instances),
                "instances": []
            }
            
            for inst in instances:
                inst_stats = {
                    "instance_id": inst.instance_id,
                    "full_model_id": inst.full_model_id,
                    "node_id": inst.node_id,
                    "layers": f"{inst.start_layer}-{inst.end_layer}",
                    "is_first_layer": inst.is_first_layer,
                    "health_score": round(inst.health_score, 1),
                    "total_requests": inst.total_requests,
                    "success_rate": round(inst.success_rate, 1),
                    "avg_latency_ms": round(inst.avg_latency, 2),
                    "current_connections": inst.current_connections,
                    "weight": inst.weight,
                    "last_request": inst.last_request_time
                }
                model_stats["instances"].append(inst_stats)
            
            stats["models"][base_id] = model_stats
        
        return stats
    
    def get_instance_details(self, model_id: str, instance_id: str) -> Optional[Dict]:
        """获取单个实例的详细信息"""
        base_id = model_id.split("::")[0] if "::" in model_id else model_id
        
        instances = self.get_available_instances(base_id)
        
        for inst in instances:
            if inst.instance_id == instance_id:
                return {
                    "instance_id": inst.instance_id,
                    "full_model_id": inst.full_model_id,
                    "node_id": inst.node_id,
                    "layers": {"start": inst.start_layer, "end": inst.end_layer},
                    "is_first_layer": inst.is_first_layer,
                    "statistics": {
                        "total_requests": inst.total_requests,
                        "success_requests": inst.success_requests,
                        "failed_requests": inst.failed_requests,
                        "success_rate": round(inst.success_rate, 2),
                        "avg_latency_ms": round(inst.avg_latency, 2),
                        "current_connections": inst.current_connections
                    },
                    "health": {
                        "score": round(inst.health_score, 1),
                        "weight": inst.weight,
                        "status": "healthy" if inst.health_score >= 70 else
                                  ("degraded" if inst.health_score >= 40 else "unhealthy")
                    },
                    "last_activity": inst.last_request_time
                }
        
        return None
    
    def reset_statistics(self, model_id: Optional[str] = None):
        """重置统计信息"""
        with self._lock:
            if model_id:
                base_id = model_id.split("::")[0] if "::" in model_id else model_id
                if base_id in self.instances:
                    for inst in self.instances[base_id].values():
                        inst.total_requests = 0
                        inst.success_requests = 0
                        inst.failed_requests = 0
                        inst.total_latency = 0.0
                        inst.current_connections = 0
            else:
                self.instances.clear()
                self.round_robin_counters.clear()
                self.total_selections = 0
                self.strategy_stats.clear()
            
            logger.info(f"[LoadBalancer] 统计信息已重置: {'全部' if not model_id else model_id}")
    
    def set_instance_weight(self, model_id: str, instance_id: str, weight: float):
        """设置实例权重"""
        base_id = model_id.split("::")[0] if "::" in model_id else model_id
        
        with self._lock:
            if base_id in self.instances and instance_id in self.instances[base_id]:
                self.instances[base_id][instance_id].weight = max(0.1, min(10.0, weight))
                logger.info(f"[LoadBalancer] 设置权重: {instance_id} = {weight}")


# 全局负载均衡器实例
_load_balancer: Optional[LoadBalancer] = None


def get_load_balancer() -> LoadBalancer:
    """获取全局负载均衡器实例"""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer()
    return _load_balancer


def init_load_balancer(manager):
    """初始化全局负载均衡器"""
    global _load_balancer
    _load_balancer = LoadBalancer(manager)
    return _load_balancer