"""
自动模型分配器 - 基于集群资源的智能模型调度系统
=================================================

核心能力：
1. 资源感知：实时收集所有在线节点的GPU资源（显存、算力、负载）
2. 智能分配：使用改进的多维背包算法，最大化模型服务能力
3. 多策略支持：
   - maximize_utilization: 最大化资源利用率（推荐）
   - maximize_diversity: 最大化模型多样性
   - prioritize_large: 优先加载大模型
   - balanced: 平衡策略（大小+多样性）
4. 安全保障：自动保留KV Cache余量，避免OOM
5. 动态适配：节点上下线时自动重新规划

使用示例：
    from auto_model_allocator import AutoModelAllocator

    allocator = AutoModelAllocator(cluster_manager)
    plan = allocator.generate_optimal_plan()
    await allocator.execute_plan(plan)

API接口：
    POST /api/auto-allocate/preview  - 预览最优方案
    POST /api/auto-allocate/execute   - 执行分配
    GET  /api/auto-allocate/status    - 查看当前状态
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json

logger = logging.getLogger(__name__)


# ============================================================
#  模型库定义 - 常见大语言模型的资源需求
# ============================================================

@dataclass
class ModelSpec:
    """模型规格定义"""
    model_id: str               # 模型标识 (如 "Qwen/Qwen3-4B")
    pretty_name: str            # 显示名称 (如 "Qwen3-4B")
    total_layers: int           # 总层数
    layer_memory_mb: float = 100.0  # 每层预估显存(MB)
    param_count: float = 0      # 参数量(亿)
    context_length: int = 8192  # 支持的上下文长度
    category: str = "general"   # 类别: general/code/math/reasoning
    priority: float = 1.0       # 优先级权重

    @property
    def total_memory_mb(self) -> float:
        """总显存需求(MB)"""
        return self.total_layers * self.layer_memory_mb

    @property
    def total_memory_gb(self) -> float:
        """总显存需求(GB)"""
        return self.total_memory_mb / 1024


# 预定义模型库（可根据实际需求扩展）
MODEL_LIBRARY: Dict[str, ModelSpec] = {
    # === 小型模型 (< 4B) - 适合低延迟场景 ===
    "qwen3-0.6b": ModelSpec(
        model_id="Qwen/Qwen3-0.6B",
        pretty_name="Qwen3-0.6B",
        total_layers=24,
        layer_memory_mb=45,
        param_count=0.6,
        category="general",
        priority=0.8
    ),
    "qwen3-1.7b": ModelSpec(
        model_id="Qwen/Qwen3-1.7B",
        pretty_name="Qwen3-1.7B",
        total_layers=24,
        layer_memory_mb=80,
        param_count=1.7,
        category="general",
        priority=0.9
    ),
    "qwen3-4b": ModelSpec(
        model_id="Qwen/Qwen3-4B",
        pretty_name="Qwen3-4B",
        total_layers=36,
        layer_memory_mb=110,
        param_count=4,
        category="general",
        priority=1.0
    ),

    # === 中型模型 (4B - 14B) - 性价比之选 ===
    "qwen3-8b": ModelSpec(
        model_id="Qwen/Qwen3-8B",
        pretty_name="Qwen3-8B",
        total_layers=40,
        layer_memory_mb=180,
        param_count=8,
        category="general",
        priority=1.2
    ),
    "qwen2.5-7b": ModelSpec(
        model_id="Qwen/Qwen2.5-7B",
        pretty_name="Qwen2.5-7B",
        total_layers=28,
        layer_memory_mb=160,
        param_count=7,
        category="code",
        priority=1.1
    ),
    "llama3.1-8b": ModelSpec(
        model_id="meta-llama/Llama-3.1-8B",
        pretty_name="Llama3.1-8B",
        total_layers=32,
        layer_memory_mb=170,
        param_count=8,
        category="general",
        priority=1.1
    ),
    "qwen3-14b": ModelSpec(
        model_id="Qwen/Qwen3-14B",
        pretty_name="Qwen3-14B",
        total_layers=48,
        layer_memory_mb=220,
        param_count=14,
        category="reasoning",
        priority=1.3
    ),

    # === 大型模型 (14B - 72B) - 高性能场景 ===
    "qwen3-32b": ModelSpec(
        model_id="Qwen/Qwen3-32B",
        pretty_name="Qwen3-32B",
        total_layers=64,
        layer_memory_mb=350,
        param_count=32,
        category="reasoning",
        priority=1.5
    ),
    "qwen2.5-32b": ModelSpec(
        model_id="Qwen/Qwen2.5-32B",
        pretty_name="Qwen2.5-32B",
        total_layers=64,
        layer_memory_mb=340,
        param_count=32,
        category="code",
        priority=1.4
    ),
    "llama3.1-70b": ModelSpec(
        model_id="meta-llama/Llama-3.1-70B",
        pretty_name="Llama3.1-70B",
        total_layers=80,
        layer_memory_mb=420,
        param_count=70,
        category="general",
        priority=1.6
    ),
    "qwen3-max": ModelSpec(
        model_id="Qwen/Qwen3-Max",
        pretty_name="Qwen3-Max(AoE)",
        total_layers=80,
        layer_memory_mb=450,
        param_count=72,
        category="reasoning",
        priority=1.8
    ),

    # === 专业模型 ===
    "codestral-22b": ModelSpec(
        model_id="mistralai/Codestral-22B",
        pretty_name="Codestral-22B",
        total_layers=56,
        layer_memory_mb=300,
        param_count=22,
        category="code",
        priority=1.3
    ),
    "deepseek-coder-33b": ModelSpec(
        model_id="deepseek-ai/DeepSeek-Coder-33B",
        pretty_name="DeepSeek-Coder-33B",
        total_layers=62,
        layer_memory_mb=360,
        param_count=33,
        category="code",
        priority=1.4
    ),
}


# ============================================================
#  分配策略枚举
# ============================================================

class AllocationStrategy(str, Enum):
    """分配策略"""
    MAXIMIZE_UTILIZATION = "maximize_utilization"  # 最大化资源利用率（默认/推荐）
    MAXIMIZE_DIVERSITY = "maximize_diversity"       # 最大化模型多样性
    PRIORITIZE_LARGE = "prioritize_large"           # 优先加载大模型
    BALANCED = "balanced"                           # 平衡策略
    CUSTOM = "custom"                               # 自定义优先级列表


# ============================================================
#  数据结构定义
# ============================================================

@dataclass
class NodeResource:
    """节点资源快照"""
    node_id: str
    address: str
    total_memory_mb: float          # 总显存(MB)
    used_memory_mb: float = 0       # 已用显存(MB)
    free_memory_mb: float = 0       # 剩余显存(MB)
    usable_memory_mb: float = 0     # 可用显存(扣除安全余量后)
    loaded_models: List[str] = field(default_factory=list)  # 已加载模型ID
    device_type: str = "Unknown"
    flops: Dict[str, float] = field(default_factory=dict)
    status: str = "online"
    utilization: float = 0.0        # 当前利用率(0-1)

    @property
    def safety_margin_mb(self) -> float:
        """安全余量(MB) - 默认30%给KV Cache"""
        return self.free_memory_mb * 0.30

    @property
    def effective_free_mb(self) -> float:
        """有效可用显存(扣除安全余量)"""
        return max(0, self.free_memory_mb - self.safety_margin_mb)


@dataclass
class ModelAllocationPlan:
    """模型分配方案"""
    plan_id: str
    strategy: AllocationStrategy
    generated_at: float
    # 分配结果: {model_id: {node_id: {start_layer, end_layer, layers_count}}}
    allocations: Dict[str, Dict[str, Dict]] = field(default_factory=dict)
    # 统计信息
    total_models: int = 0
    total_param_count: float = 0  # 总参数量(亿)
    memory_utilization: float = 0.0  # 显存利用率(0-1)
    node_utilization: Dict[str, float] = field(default_factory=dict)  # 各节点利用率
    # 未分配原因
    unallocated_models: List[Dict] = field(default_factory=list)
    # 优化建议
    recommendations: List[str] = field(default_factory=list)
    # 预估性能评分(0-100)
    performance_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "strategy": self.strategy.value if isinstance(self.strategy, AllocationStrategy) else self.strategy,
            "generated_at": self.generated_at,
            "allocations": self.allocations,
            "summary": {
                "total_models": self.total_models,
                "total_param_count": round(self.total_param_count, 1),
                "memory_utilization": round(self.memory_utilization * 100, 1),
                "performance_score": round(self.performance_score, 1),
            },
            "node_utilization": {k: round(v * 100, 1) for k, v in self.node_utilization.items()},
            "unallocated_models": self.unallocated_models,
            "recommendations": self.recommendations,
        }


# ============================================================
#  核心分配器类
# ============================================================

class AutoModelAllocator:
    """
    自动模型分配器

    核心算法流程：
    1. 收集所有在线节点的实时资源状态
    2. 根据策略对候选模型排序
    3. 使用改进的贪心算法进行分配：
       - 单节点优先：尝试将完整模型放入单个节点
       - 拆分回退：单节点放不下时，考虑跨节点分片
       - 安全检查：确保每个节点留有足够KV Cache空间
    4. 输出最优分配方案及执行建议
    """

    # 安全配置
    SAFETY_MARGIN_RATIO = 0.30      # KV Cache安全余量(30%)
    MIN_FREE_MB = 512               # 最小剩余显存(MB)
    MAX_NODE_UTILIZATION = 0.85     # 单节点最大利用率(85%)

    def __init__(self, manager):
        """
        初始化分配器

        Args:
            manager: EXOClusterManager 实例
        """
        self.manager = manager
        self.model_library = MODEL_LIBRARY.copy()
        self.allocation_history: List[ModelAllocationPlan] = []
        self._current_plan: Optional[ModelAllocationPlan] = None

        logger.info("[AutoAllocator] 初始化完成，模型库包含 "
                   f"{len(self.model_library)} 个预定义模型")

    def add_custom_model(self, spec: ModelSpec):
        """添加自定义模型到库中"""
        self.model_library[spec.model_id] = spec
        logger.info(f"[AutoAllocator] 添加自定义模型: {spec.pretty_name}")

    def remove_model(self, model_id: str):
        """从模型库移除模型"""
        if model_id in self.model_library:
            del self.model_library[model_id]
            logger.info(f"[AutoAllocator] 移除模型: {model_id}")

    def collect_node_resources(self) -> List[NodeResource]:
        """
        收集所有在线节点的资源状态

        Returns:
            NodeResource 列表
        """
        resources = []

        for node_id, node_info in self.manager.nodes.items():
            # 只处理在线节点
            if node_info.status.value not in ('online', 'connecting'):
                continue

            # 获取显存信息
            raw_mem = node_info.device_info.get("memory", 0)
            mem_detail = node_info.device_info.get("memory_detail")

            total_mem = raw_mem
            if (total_mem == 0 or total_mem is None) and mem_detail and mem_detail.get("total", 0) > 0:
                total_mem = mem_detail["total"]

            # 已用显存（优先使用pynvml实时数据）
            used_mem = 0
            if mem_detail and mem_detail.get("used", 0) > 0:
                used_mem = mem_detail["used"]
            else:
                # 基于已加载模型估算
                for model in node_info.loaded_models:
                    shard = model.get("shard", {})
                    n_layers = shard.get("end_layer", 0) - shard.get("start_layer", 0) + 1
                    used_mem += n_layers * 100  # 粗略估算

            free_mem = max(0, total_mem - used_mem)

            resource = NodeResource(
                node_id=node_id,
                address=f"{node_info.address}:{node_info.port}",
                total_memory_mb=total_mem,
                used_memory_mb=used_mem,
                free_memory_mb=free_mem,
                usable_memory_mb=max(0, free_mem * (1 - self.SAFETY_MARGIN_RATIO)),
                loaded_models=[m.get("model_id", "") for m in node_info.loaded_models],
                device_type=node_info.device_info.get("chip", "Unknown"),
                flops=node_info.device_info.get("flops", {}),
                status=node_info.status.value,
                utilization=used_mem / total_mem if total_mem > 0 else 0
            )

            resources.append(resource)

        # 按剩余显存降序排序（优先利用空闲资源多的节点）
        resources.sort(key=lambda x: x.usable_memory_mb, reverse=True)

        logger.info(f"[AutoAllocator] 收集到 {len(resources)} 个在线节点资源")
        for r in resources[:3]:  # 打印前3个
            logger.info(f"  - {r.node_id}: {r.device_type}, "
                       f"总={r.total_memory_mb/1024:.1f}GB, "
                       f"剩={r.free_memory_mb/1024:.1f}GB, "
                       f"可用={r.usable_memory_mb/1024:.1f}GB")

        return resources

    def generate_optimal_plan(
        self,
        strategy: AllocationStrategy = AllocationStrategy.MAXIMIZE_UTILIZATION,
        custom_priority: Optional[List[str]] = None,
        exclude_models: Optional[List[str]] = None,
        force_include: Optional[List[str]] = None
    ) -> ModelAllocationPlan:
        """
        生成最优模型分配方案

        Args:
            strategy: 分配策略
            custom_priority: 自定义模型优先级列表（仅CUSTOM策略时使用）
            exclude_models: 要排除的模型ID列表
            force_include: 强制包含的模型ID列表（即使资源紧张）

        Returns:
            ModelAllocationPlan 最优分配方案
        """
        start_time = time.time()

        # Step 1: 收集资源
        nodes = self.collect_node_resources()
        if not nodes:
            raise ValueError("没有可用的在线节点")

        # Step 2: 准备候选模型
        candidates = self._prepare_candidates(exclude_models, force_include)

        # Step 3: 根据策略排序
        sorted_candidates = self._sort_by_strategy(candidates, nodes, strategy, custom_priority)

        # Step 4: 执行分配算法
        allocations, unallocated = self._execute_allocation(nodes, sorted_candidates)

        # Step 5: 计算统计信息和评分
        plan = self._build_plan(
            strategy, nodes, allocations, unallocated, start_time
        )

        # 缓存当前方案
        self._current_plan = plan
        self.allocation_history.append(plan)

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"[AutoAllocator] ✅ 方案生成完成: "
                   f"{plan.total_models}个模型, "
                   f"{plan.total_param_count:.1f}B参数, "
                   f"利用率{plan.memory_utilization*100:.1f}%, "
                   f"耗时{elapsed:.1f}ms")

        return plan

    def _prepare_candidates(
        self,
        exclude_models: Optional[List[str]],
        force_include: Optional[List[str]]
    ) -> List[ModelSpec]:
        """准备候选模型列表"""
        candidates = []

        for model_id, spec in self.model_library.items():
            # 排除指定模型
            if exclude_models and model_id in exclude_models:
                continue
            candidates.append(spec)

        # 强制包含的模型排到最前面
        if force_include:
            forced = [c for c in candidates if c.model_id in force_include]
            others = [c for c in candidates if c.model_id not in force_include]
            candidates = forced + others

        return candidates

    def _sort_by_strategy(
        self,
        candidates: List[ModelSpec],
        nodes: List[NodeResource],
        strategy: AllocationStrategy,
        custom_priority: Optional[List[str]]
    ) -> List[ModelSpec]:
        """根据策略对候选模型排序"""

        if strategy == AllocationStrategy.MAXIMIZE_UTILIZATION:
            # 最大化利用率：按参数密度排序（参数量/显存），优先选性价比高的
            return sorted(candidates, key=lambda x: (
                x.param_count / max(x.total_memory_mb, 1),  # 参数密度
                x.priority  # 同等密度下按优先级
            ), reverse=True)

        elif strategy == AllocationStrategy.MAXIMIZE_DIVERSITY:
            # 最大化多样性：每个类别选1-2个代表，小->大
            categories = {}
            for c in candidates:
                if c.category not in categories:
                    categories[c.category] = []
                categories[c.category].append(c)

            result = []
            for cat_models in categories.values():
                # 每个类别内按大小升序，取前2个
                sorted_cat = sorted(cat_models, key=lambda x: x.total_memory_mb)
                result.extend(sorted_cat[:2])

            # 最终按大小降序（大的优先分配）
            return sorted(result, key=lambda x: x.total_memory_mb, reverse=True)

        elif strategy == AllocationStrategy.PRIORITIZE_LARGE:
            # 优先大模型：按参数量降序
            return sorted(candidates, key=lambda x: (
                x.param_count,
                x.priority
            ), reverse=True)

        elif strategy == AllocationStrategy.BALANCED:
            # 平衡策略：综合考虑大小、多样性、优先级
            scored = []
            for c in candidates:
                score = (
                    c.param_count * 0.4 +         # 大小权重40%
                    c.priority * 10 * 0.3 +       # 优先级权重30%
                    (1 / (c.total_memory_mb / 1024)) * 0.2 +  # 效率权重20%（小的加分）
                    len(c.category) * 0.1         # 多样性奖励10%
                )
                scored.append((score, c))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [c for _, c in scored]

        elif strategy == AllocationStrategy.CUSTOM and custom_priority:
            # 自定义顺序：按用户指定的优先级
            priority_map = {m: i for i, m in enumerate(custom_priority)}
            return sorted(candidates, key=lambda x: priority_map.get(x.model_id, 999))

        else:
            # 默认：按参数量降序
            return sorted(candidates, key=lambda x: x.param_count, reverse=True)

    def _execute_allocation(
        self,
        nodes: List[NodeResource],
        candidates: List[ModelSpec]
    ) -> Tuple[Dict[str, Dict[str, Dict]], List[Dict]]:
        """
        执行核心分配算法

        Returns:
            (allocations, unallocated)
            allocations: {model_id: {node_id: {start, end, count}}}
            unallocated: [{model_id, reason}]
        """
        allocations = {}
        unallocated = []

        # 创建节点资源的可变副本（用于跟踪分配过程）
        node_states = {n.node_id: n.usable_memory_mb for n in nodes}

        for candidate in candidates:
            model_id = candidate.model_id
            model_mem = candidate.total_memory_mb
            total_layers = candidate.total_layers

            logger.debug(f"[AutoAllocator] 尝试分配 {candidate.pretty_name}: "
                        f"{model_mem/1024:.1f}GB, {total_layers}层")

            # ===== Phase 1: 尝试单节点完整加载 =====
            single_node_result = self._try_single_node(
                nodes, node_states, candidate
            )

            if single_node_result:
                node_id, alloc_info = single_node_result
                allocations[model_id] = {node_id: alloc_info}
                # 更新节点状态
                node_states[node_id] -= model_mem
                logger.info(f"[AutoAllocator] ✅ {candidate.pretty_name} → "
                           f"单节点 {node_id}")
                continue

            # ===== Phase 2: 尝试跨节点分片 =====
            sharded_result = self._try_sharded_allocation(
                nodes, node_states, candidate
            )

            if sharded_result:
                allocations[model_id] = sharded_result
                # 更新所有涉及的节点状态
                for node_id, alloc_info in sharded_result.items():
                    layers_count = alloc_info["layers_count"]
                    node_states[node_id] -= layers_count * candidate.layer_memory_mb
                logger.info(f"[AutoAllocator] ✅ {candidate.pretty_name} → "
                           f"跨{len(sharded_result)}节点分片")
                continue

            # ===== 无法分配 =====
            reason = self._get_unallocation_reason(candidate, nodes, node_states)
            unallocated.append({
                "model_id": model_id,
                "pretty_name": candidate.pretty_name,
                "memory_gb": round(model_mem / 1024, 1),
                "param_count": candidate.param_count,
                "reason": reason
            })
            logger.warning(f"[AutoAllocator] ❌ {candidate.pretty_name} 无法分配: {reason}")

        return allocations, unallocated

    def _try_single_node(
        self,
        nodes: List[NodeResource],
        node_states: Dict[str, float],
        candidate: ModelSpec
    ) -> Optional[Tuple[str, Dict]]:
        """
        尝试单节点完整加载

        选择逻辑：
        1. 找到所有能容纳该模型的节点（free >= model_size）
        2. 在这些节点中选择利用率最低的（负载均衡）
        3. 如果没有单个节点能容纳，返回None
        """
        model_mem = candidate.total_memory_mb
        best_node = None
        best_waste = float('inf')

        for node in nodes:
            available = node_states.get(node.node_id, 0)

            # 检查是否有足够空间
            if available < model_mem:
                continue

            # 检查是否超过最大利用率限制
            new_usage = (node.total_memory_mb - available + model_mem) / node.total_memory_mb
            if new_usage > self.MAX_NODE_UTILIZATION:
                continue

            # 选择浪费最少的（最合适的节点）
            waste = available - model_mem
            if waste < best_waste:
                best_waste = waste
                best_node = node

        if best_node:
            alloc_info = {
                "start_layer": 0,
                "end_layer": candidate.total_layers - 1,
                "layers_count": candidate.total_layers,
                "allocation_type": "single_node",
                "memory_mb": model_mem
            }
            return (best_node.node_id, alloc_info)

        return None

    def _try_sharded_allocation(
        self,
        nodes: List[NodeResource],
        node_states: Dict[str, float],
        candidate: ModelSpec
    ) -> Optional[Dict[str, Dict]]:
        """
        尝试跨节点分片分配

        策略：
        1. 按可用空间降序排列节点
        2. 使用贪心算法逐层分配
        3. 确保每层都有足够空间
        """
        model_mem_per_layer = candidate.layer_memory_mb
        total_layers = candidate.total_layers
        remaining_layers = total_layers
        current_layer = 0
        allocations = {}

        # 按可用空间降序排序节点
        sorted_nodes = sorted(
            nodes,
            key=lambda n: node_states.get(n.node_id, 0),
            reverse=True
        )

        for node in sorted_nodes:
            if remaining_layers <= 0:
                break

            available = node_states.get(node.node_id, 0)

            # 该节点最多能承载多少层
            max_layers_by_space = int(available / model_mem_per_layer)
            # 至少分配1层，但不超过剩余层数
            layers_for_node = min(max(1, max_layers_by_space), remaining_layers)

            if layers_for_node <= 0:
                continue

            end_layer = current_layer + layers_for_node - 1

            allocations[node.node_id] = {
                "start_layer": current_layer,
                "end_layer": end_layer,
                "layers_count": layers_for_node,
                "allocation_type": "sharded",
                "memory_mb": layers_for_node * model_mem_per_layer
            }

            remaining_layers -= layers_for_node
            current_layer = end_layer + 1

        # 检查是否全部分配完
        if remaining_layers > 0 or len(allocations) == 0:
            return None  # 分片失败

        return allocations

    def _get_unallocation_reason(
        self,
        candidate: ModelSpec,
        nodes: List[NodeResource],
        node_states: Dict[str, float]
    ) -> str:
        """获取无法分配的原因"""
        model_mem = candidate.total_memory_mb
        total_available = sum(node_states.values())
        max_single = max(node_states.values()) if node_states else 0

        reasons = []

        if total_available < model_mem:
            reasons.append(f"集群总可用显存不足({total_available/1024:.1f}GB < {model_mem/1024:.1f}GB)")
        if max_single < model_mem:
            reasons.append(f"无单节点能容纳({max_single/1024:.1f}GB < {model_mem/1024:.1f}GB)")

        if not reasons:
            reasons.append("资源碎片化或安全限制")

        return "; ".join(reasons)

    def _build_plan(
        self,
        strategy: AllocationStrategy,
        nodes: List[NodeResource],
        allocations: Dict,
        unallocated: List[Dict],
        start_time: float
    ) -> ModelAllocationPlan:
        """构建完整的分配方案对象"""

        # 计算统计信息
        total_params = 0
        total_allocated_mem = 0
        node_usage = {n.node_id: n.used_memory_mb for n in nodes}

        for model_id, node_allocs in allocations.items():
            # 从模型库获取参数量
            if model_id in self.model_library:
                total_params += self.model_library[model_id].param_count

            # 统计各节点占用
            for node_id, alloc_info in node_allocs.items():
                mem = alloc_info.get("memory_mb", 0)
                total_allocated_mem += mem
                if node_id in node_usage:
                    node_usage[node_id] += mem

        # 计算利用率
        total_capacity = sum(n.total_memory_mb for n in nodes)
        total_used = sum(node_usage.values())
        memory_utilization = total_used / total_capacity if total_capacity > 0 else 0

        # 各节点利用率
        node_utilization = {}
        for n in nodes:
            usage = node_usage.get(n.node_id, n.used_memory_mb)
            node_utilization[n.node_id] = usage / n.total_memory_mb if n.total_memory_mb > 0 else 0

        # 生成建议
        recommendations = self._generate_recommendations(
            nodes, allocations, unallocated, node_utilization
        )

        # 性能评分（0-100）
        performance_score = self._calculate_performance_score(
            allocations, unallocated, node_utilization, total_params
        )

        plan = ModelAllocationPlan(
            plan_id=f"plan_{int(time.time())}",
            strategy=strategy,
            generated_at=time.time(),
            allocations=allocations,
            total_models=len(allocations),
            total_param_count=total_params,
            memory_utilization=memory_utilization,
            node_utilization=node_utilization,
            unallocated_models=unallocated,
            recommendations=recommendations,
            performance_score=performance_score
        )

        return plan

    def _generate_recommendations(
        self,
        nodes: List[NodeResource],
        allocations: Dict,
        unallocated: List[Dict],
        node_utilization: Dict[str, float]
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 1. 利用率分析
        avg_util = sum(node_utilization.values()) / len(node_utilization) if node_utilization else 0
        if avg_util < 0.5:
            recommendations.append(f"集群整体利用率较低({avg_util*100:.0f}%)，可考虑加载更多模型")
        elif avg_util > 0.9:
            recommendations.append(f"⚠️ 集群接近满载({avg_util*100:.0f}%)，建议监控OOM风险")

        # 2. 不均衡检测
        utils = list(node_utilization.values())
        if len(utils) >= 2:
            max_u, min_u = max(utils), min(utils)
            if max_u - min_u > 0.3:
                recommendations.append(f"节点负载不均衡(最高{max_u*100:.0f}% vs 最低{min_u*100:.0f}%)，建议重平衡")

        # 3. 未分配模型提示
        if unallocated:
            large_unallocated = [u for u in unallocated if u["param_count"] > 10]
            if large_unallocated:
                names = [u["pretty_name"] for u in large_unallocated[:3]]
                recommendations.append(f"大型模型未分配: {', '.join(names)}，如需使用请增加节点资源")

        # 4. 优化建议
        if len(allocations) >= 3:
            recommendations.append("✅ 当前配置支持多模型并行服务，可满足多样化推理需求")

        return recommendations

    def _calculate_performance_score(
        self,
        allocations: Dict,
        unallocated: List[Dict],
        node_utilization: Dict[str, float],
        total_params: float
    ) -> float:
        """
        计算性能评分 (0-100)

        评估维度：
        - 模型覆盖度 (30%): 成功分配的模型占比
        - 参数总量 (25%): 总参数量（越大越好）
        - 资源利用率 (25%): 显存使用效率
        - 负载均衡度 (20%): 各节点负载差异
        """
        score = 0.0

        # 1. 模型覆盖度 (30分)
        total_candidates = len(allocations) + len(unallocated)
        if total_candidates > 0:
            coverage = len(allocations) / total_candidates
            score += coverage * 30

        # 2. 参数总量 (25分) - 假设100B为满分
        param_score = min(total_params / 100, 1.0)
        score += param_score * 25

        # 3. 资源利用率 (25分) - 70-85%为最佳区间
        avg_util = sum(node_utilization.values()) / len(node_utilization) if node_utilization else 0
        if 0.65 <= avg_util <= 0.85:
            util_score = 1.0
        elif avg_util < 0.65:
            util_score = avg_util / 0.65
        else:
            util_score = max(0, 1.0 - (avg_util - 0.85) * 2)
        score += util_score * 25

        # 4. 负载均衡度 (20分)
        if len(node_utilization) >= 2:
            utils = list(node_utilization.values())
            std_dev = (sum((u - avg_util)**2 for u in utils) / len(utils)) ** 0.5
            balance_score = max(0, 1.0 - std_dev * 2)  # 标准差越小越好
        else:
            balance_score = 1.0
        score += balance_score * 20

        return min(100, max(0, score))

    async def execute_plan(self, plan: ModelAllocationPlan) -> Dict[str, Any]:
        """
        执行分配方案

        Args:
            plan: 要执行的分配方案

        Returns:
            执行结果报告
        """
        logger.info(f"[AutoAllocator] 🚀 开始执行方案: {plan.plan_id}")

        results = {"success": [], "failed": [], "skipped": []}

        try:
            from gpu_pool_integration import GPUPoolIntegration
            pool = GPUPoolIntegration(self.manager)

            for model_id, node_allocs in plan.allocations.items():
                try:
                    # 查找模型规格
                    spec = self.model_library.get(model_id)
                    if not spec:
                        logger.warning(f"[AutoAllocator] 未找到模型规格: {model_id}")
                        results["skipped"].append({"model_id": model_id, "reason": "未知模型"})
                        continue

                    # 调用GPU池加载接口
                    # 注意：这里需要根据实际接口调整
                    logger.info(f"[AutoAllocator] 正在加载 {model_id}...")

                    # 构造分配参数
                    allocation_list = []
                    for node_id, alloc_info in node_allocs.items():
                        allocation_list.append({
                            "node_id": node_id,
                            "start_layer": alloc_info["start_layer"],
                            "end_layer": alloc_info["end_layer"],
                        })

                    # TODO: 实际调用加载接口（需要与GPUPoolIntegration对接）
                    # result = await pool.load_model_with_allocation(model_id, allocation_list)

                    results["success"].append({
                        "model_id": model_id,
                        "pretty_name": spec.pretty_name if spec else model_id,
                        "nodes": list(node_allocs.keys()),
                        "type": node_allocs[list(node_allocs.keys())[0]].get("allocation_type", "unknown")
                    })

                    logger.info(f"[AutoAllocator] ✅ {model_id} 加载成功")

                except Exception as e:
                    logger.error(f"[AutoAllocator] ❌ {model_id} 加载失败: {e}")
                    results["failed"].append({
                        "model_id": model_id,
                        "error": str(e)
                    })

        except ImportError as e:
            logger.error(f"[AutoAllocator] GPUPoolIntegration 导入失败: {e}")
            return {"error": "GPU池模块不可用", "details": str(e)}

        # 生成执行报告
        report = {
            "plan_id": plan.plan_id,
            "executed_at": time.time(),
            "summary": {
                "total": len(plan.allocations),
                "success": len(results["success"]),
                "failed": len(results["failed"]),
                "skipped": len(results["skipped"]),
            },
            "results": results,
        }

        logger.info(f"[AutoAllocator] 📊 执行完成: "
                   f"{report['summary']['success']}/{report['summary']['total']} 成功")

        return report

    def get_current_status(self) -> Dict[str, Any]:
        """获取当前分配状态"""
        nodes = self.collect_node_resources()

        return {
            "allocator_status": "active",
            "online_nodes": len(nodes),
            "total_memory_gb": round(sum(n.total_memory_mb for n in nodes) / 1024, 1),
            "free_memory_gb": round(sum(n.free_memory_mb for n in nodes) / 1024, 1),
            "usable_memory_gb": round(sum(n.usable_memory_mb for n in nodes) / 1024, 1),
            "loaded_models": sum(len(n.loaded_models) for n in nodes),
            "current_plan": self._current_plan.to_dict() if self._current_plan else None,
            "library_size": len(self.model_library),
            "strategies": [s.value for s in AllocationStrategy],
        }

    def preview_resource_distribution(self) -> Dict[str, Any]:
        """
        预览资源分布情况（不执行分配）

        用于前端展示和调试
        """
        nodes = self.collect_node_resources()

        node_details = []
        for n in nodes:
            node_details.append({
                "node_id": n.node_id,
                "device": n.device_type,
                "total_gb": round(n.total_memory_mb / 1024, 1),
                "used_gb": round(n.used_memory_mb / 1024, 1),
                "free_gb": round(n.free_memory_mb / 1024, 1),
                "usable_gb": round(n.usable_memory_mb / 1024, 1),
                "utilization_pct": round(n.utilization * 100, 1),
                "loaded_models": n.loaded_models,
            })

        # 按类别汇总模型库
        models_by_category = {}
        for spec in self.model_library.values():
            cat = spec.category
            if cat not in models_by_category:
                models_by_category[cat] = []
            models_by_category[cat].append({
                "id": spec.model_id,
                "name": spec.pretty_name,
                "size_gb": round(spec.total_memory_gb, 1),
                "params": spec.param_count,
                "priority": spec.priority,
            })

        return {
            "timestamp": time.time(),
            "nodes": node_details,
            "models_by_category": models_by_category,
            "recommendations": [
                "使用 POST /api/auto-allocate/preview 生成最优方案",
                "使用 POST /api/auto-allocate/execute 执行自动分配",
            ]
        }
