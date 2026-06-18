"""
主备冗余分配系统 (Active-Standby Redundancy)
=============================================

核心目标：消除单点故障（SPOF），实现高可用性

核心能力：
1. 冗余分配算法：
   - Anti-SPOF 约束：关键模型必须多节点部署
   - 主备策略：Primary + Standby 双副本
   - 负载分散：避免所有模型集中在单一高性能节点
   - 异构容错：主备尽量分布在不同硬件/机架上

2. 故障转移机制：
   - 健康检查：定期探测所有模型实例状态
   - 自动切换：主实例故障 → 备实例秒级接管
   - 会话保持：进行中的请求完成后才切换
   - 快速恢复：原主节点修复后自动回切或升级为备

3. 部署模式：
   - Active-Passive（主备热备）：1个活跃+1个待命
   - Active-Active（双活）：2个同时服务，负载均衡
   - N+M 冗余：N个活跃+M个备用（适用于大规模集群）

使用示例：
    from ha_model_allocator import HAModelAllocator, RedundancyMode

    ha_allocator = HAModelAllocator(cluster_manager)

    # 生成带冗余的高可用方案
    plan = ha_allocator.generate_ha_plan(
        redundancy_mode="active_passive",
        min_replicas_critical=2,
        max_single_node_dependency=0.3  # 单节点依赖不超过30%
    )

    # 执行健康检查
    health = await ha_allocator.health_check()

    # 手动触发故障转移
    await ha_allocator.failover(model_id="qwen3-8b", failed_node="gpu-01")
"""

import asyncio
import logging
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from enum import Enum
from collections import defaultdict
import json
import hashlib

# 导入基础分配器中的类型定义
from auto_model_allocator import AutoModelAllocator, MODEL_LIBRARY, NodeResource


logger = logging.getLogger(__name__)


# ============================================================
#  枚举和常量定义
# ============================================================

class RedundancyMode(str, Enum):
    """冗余模式"""
    ACTIVE_PASSIVE = "active_passive"    # 主备热备（推荐）
    ACTIVE_ACTIVE = "active_active"      # 双活负载均衡
    N_PLUS_M = "n_plus_m"               # N活跃+M备用


class InstanceRole(str, Enum):
    """实例角色"""
    PRIMARY = "primary"      # 主实例（活跃）
    STANDBY = "standby"      # 备用实例（待命/热备）
    ACTIVE = "active"        # 活跃实例（用于双活模式）


class InstanceStatus(str, Enum):
    """实例状态"""
    HEALTHY = "healthy"              # 健康
    DEGRADED = "degraded"            # 降级（响应慢）
    UNHEALTHY = "unhealthy"          # 不健康
    DOWN = "down"                    # 宕机
    STARTING = "starting"            # 启动中
    DRAINING = "draining"            # 排空中（停止接收新请求）


class FailoverState(str, Enum):
    """故障转移状态"""
    NORMAL = "normal"                # 正常运行
    FAILOVER_IN_PROGRESS = "failover_in_progress"  # 转移中
    FAILOVER_COMPLETED = "failover_completed"       # 已完成转移
    RECOVERY_PENDING = "recovery_pending"           # 待恢复
    RECOVERING = "recovering"                       # 恢复中
    RECOVERED = "recovered"                         # 已恢复


# ============================================================
#  数据结构定义
# ============================================================

@dataclass
class ModelInstance:
    """
    模型实例定义（单个节点上的一个模型副本）

    一个模型可以有多个实例（分布在不同的节点上）
    """
    instance_id: str                  # 实例唯一ID: "{model_id}::{node_id}"
    model_id: str                     # 模型ID
    node_id: str                      # 所在节点
    role: InstanceRole = InstanceRole.PRIMARY  # 角色
    status: InstanceStatus = InstanceStatus.HEALTHY  # 状态

    # 分片信息
    start_layer: int = 0
    end_layer: int = 0
    layers_count: int = 0

    # 运行时信息
    memory_mb: float = 0              # 占用显存(MB)
    request_count: int = 0            # 当前处理请求数
    avg_latency_ms: float = 0         # 平均延迟(ms)
    error_rate: float = 0             # 错误率(0-1)
    uptime_seconds: float = 0         # 运行时间(秒)

    # 时间戳
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    last_health_check: float = 0

    @property
    def is_active(self) -> bool:
        """是否为活跃实例（可接收请求）"""
        return self.role in (InstanceRole.PRIMARY, InstanceRole.ACTIVE) and \
               self.status == InstanceStatus.HEALTHY

    @property
    def is_available(self) -> bool:
        """是否可用（包括降级的）"""
        return self.status in (InstanceStatus.HEALTHY, InstanceStatus.DEGRADED)


@dataclass
class HAAllocationPlan:
    """
    高可用分配方案

    与普通方案的区别：
    - 每个模型可能有多个实例（主+备）
    - 包含冗余信息和故障转移配置
    - 有SPOF风险评估
    """
    plan_id: str
    mode: RedundancyMode
    generated_at: float

    # 分配结果: {model_id: [ModelInstance]}
    instances: Dict[str, List[ModelInstance]] = field(default_factory=dict)

    # 统计信息
    total_models: int = 0
    total_instances: int = 0          # 总实例数（可能>模型数）
    total_memory_mb: float = 0
    avg_redundancy: float = 0.0       # 平均冗余度

    # SPOF分析
    spof_risk_score: float = 0.0      # 单点故障风险评分(0-100,越高越危险)
    vulnerable_models: List[str] = field(default_factory=list)  # 无备份的模型
    single_node_dependencies: Dict[str, List[str]] = field(default_factory=dict)  # {node_id: [model_ids]}

    # 节点利用率
    node_utilization: Dict[str, float] = field(default_factory=dict)

    # 性能预估
    estimated_availability: float = 0.0  # 可用性预估(0-1, 如0.9999=99.99%)
    rto_seconds: float = 0.0             # 恢复时间目标(RTO)
    rpo_requests: float = 0.0            # 恢复点目标(RPO,丢失请求数)

    def to_dict(self) -> dict:
        result = {
            "plan_id": self.plan_id,
            "mode": self.mode.value if isinstance(self.mode, RedundancyMode) else self.mode,
            "generated_at": self.generated_at,

            "allocations": {},
            "instances_summary": {},

            "statistics": {
                "total_models": self.total_models,
                "total_instances": self.total_instances,
                "avg_redundancy": round(self.avg_redundancy, 2),
                "total_memory_gb": round(self.total_memory_mb / 1024, 1),
            },

            "ha_metrics": {
                "spof_risk_score": round(self.spof_risk_score, 1),
                "vulnerable_models": self.vulnerable_models,
                "estimated_availability": f"{self.estimated_availability*100:.4f}%",
                "rto_seconds": round(self.rto_seconds, 1),
                "rpo_requests": round(self.rpo_requests, 1),
            },

            "node_utilization": {
                k: round(v * 100, 1) for k, v in self.node_utilization.items()
            },

            "single_node_dependencies": self.single_node_dependencies,
        }

        for model_id, instances_list in self.instances.items():
            result["allocations"][model_id] = [
                {
                    "instance_id": inst.instance_id,
                    "node_id": inst.node_id,
                    "role": inst.role.value,
                    "status": inst.status.value,
                    "layers": {"start": inst.start_layer, "end": inst.end_layer, "count": inst.layers_count},
                    "memory_mb": inst.memory_mb,
                    "is_active": inst.is_active,
                }
                for inst in instances_list
            ]

            result["instances_summary"][model_id] = {
                "replicas": len(instances_list),
                "primary_count": sum(1 for i in instances_list if i.role == InstanceRole.PRIMARY),
                "standby_count": sum(1 for i in instances_list if i.role == InstanceRole.STANDBY),
                "active_instances": sum(1 for i in instances_list if i.is_active),
                "has_redundancy": len(instances_list) > 1,
            }

        return result


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    timestamp: float
    model_id: str
    instance_id: str
    node_id: str
    status: InstanceStatus
    latency_ms: float = 0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailoverRecord:
    """故障转移记录"""
    record_id: str
    timestamp: float
    model_id: str
    failed_instance: str
    failed_node: str
    promoted_instance: Optional[str]
    promoted_node: Optional[str]
    failover_state: FailoverState
    duration_ms: float = 0
    requests_lost: int = 0
    success: bool = False


# ============================================================
#  核心类：HA模型分配器
# ============================================================

class HAModelAllocator:
    """
    高可用模型分配器

    核心算法流程：
    1. 收集资源并评估可用性
    2. 对模型按优先级排序
    3. 为每个模型分配多个实例（主+备）：
       a. 选择主节点：性能最好、最稳定的节点
       b. 选择备节点：与主节点异构（不同硬件/位置）
       c. 确保满足最小冗余要求
    4. 反SPOF检查：确保没有单点依赖
    5. 输出方案和风险报告
    """

    # ===== 配置参数 =====
    DEFAULT_MIN_REPLICAS_CRITICAL = 2     # 关键模型最少副本数
    DEFAULT_MIN_REPLICAS_NORMAL = 1       # 普通模型最少副本数
    MAX_SINGLE_NODE_RATIO = 0.40          # 单节点承载模型占比上限(40%)
    HEALTH_CHECK_INTERVAL = 15            # 健康检查间隔(秒)
    FAILOVER_TIMEOUT_MS = 5000            # 故障转移超时(毫秒)
    HEARTBEAT_TIMEOUT_SECONDS = 30        # 心跳超时(秒)

    def __init__(self, manager):
        from auto_model_allocator import AutoModelAllocator, MODEL_LIBRARY

        self.manager = manager
        self.base_allocator = AutoModelAllocator(manager)
        self.model_library = MODEL_LIBRARY

        # 运行时状态
        self.current_ha_plan: Optional[HAAllocationPlan] = None
        self.active_instances: Dict[str, ModelInstance] = {}  # {instance_id: instance}
        self.failover_records: List[FailoverRecord] = []
        self.health_history: Dict[str, List[HealthCheckResult]] = defaultdict(list)

        # 监控任务
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(f"[HAModelAlloc] ✅ 初始化完成 "
                   f"(最小冗余: 关键={self.DEFAULT_MIN_REPLICAS_CRITICAL}, "
                   f"普通={self.DEFAULT_MIN_REPLICAS_NORMAL}, "
                   f"单节点上限={self.MAX_SINGLE_NODE_RATIO*100:.0f}%)")

    def generate_ha_plan(
        self,
        mode: Union[RedundancyMode, str] = RedundancyMode.ACTIVE_PASSIVE,
        min_replicas_critical: int = None,
        min_replicas_normal: int = None,
        max_single_node_ratio: float = None,
        exclude_models: Optional[List[str]] = None,
        force_include: Optional[List[str]] = None,
        prioritize_redundancy: bool = True
    ) -> HAAllocationPlan:
        """
        生成高可用分配方案

        Args:
            mode: 冗余模式 ("active_passive" | "active_active")
            min_replicas_critical: 关键模型最少副本数
            min_replicas_normal: 普通模型最少副本数
            max_single_node_ratio: 单节点承载模型比例上限(0-1)
            exclude_models: 排除的模型
            force_include: 强制包含的模型
            prioritize_redundancy: 是否优先保证冗余（而非最大化模型数量）

        Returns:
            HAAllocationPlan 高可用分配方案
        """
        start_time = time.time()

        # 参数标准化
        if isinstance(mode, str):
            mode = RedundancyMode(mode)

        min_replicas_crit = min_replicas_critical or self.DEFAULT_MIN_REPLICAS_CRITICAL
        min_replicas_norm = min_replicas_normal or self.DEFAULT_MIN_REPLICAS_NORMAL
        max_snode_ratio = max_single_node_ratio or self.MAX_SINGLE_NODE_RATIO

        logger.info(f"[HAModelAlloc] 🎯 开始生成HA方案: "
                   f"模式={mode.value}, "
                   f"关键模型≥{min_replicas_crit}副本, "
                   f"普通≥{min_replicas_norm}副本, "
                   f"单节点≤{max_snode_ratio*100:.0f}%")

        # Step 1: 收集节点资源
        nodes = self.base_allocator.collect_node_resources()
        if len(nodes) < min_replicas_crit:
            raise ValueError(
                f"在线节点({len(nodes)})不足以满足关键模型冗余要求"
                f"(需要≥{min_replicas_crit}个)"
            )

        # Step 2: 准备候选模型并分类
        candidates = self._prepare_candidates(exclude_models, force_include)
        critical_models, normal_models = self._classify_models_by_priority(candidates)

        logger.info(f"[HAModelAlloc] 候选模型: {len(candidates)}个 "
                   f"(关键={len(critical_models)}, 普通={len(normal_models)})")

        # Step 3: 执行HA分配算法
        allocations, node_usage = self._execute_ha_allocation(
            nodes=nodes,
            critical_models=critical_models,
            normal_models=normal_models,
            mode=mode,
            min_replicas_crit=min_replicas_crit,
            min_replicas_norm=min_replicas_norm,
            max_snode_ratio=max_snode_ratio,
            prioritize_redundancy=prioritize_redundancy
        )

        # Step 4: SPOF分析和风险评估
        spof_analysis = self._analyze_spof_risks(allocations, nodes)

        # Step 5: 构建最终方案
        plan = self._build_ha_plan(
            mode=mode,
            allocations=allocations,
            node_usage=node_usage,
            spof_analysis=spof_analysis,
            nodes=nodes,
            start_time=start_time
        )

        # 缓存
        self.current_ha_plan = plan

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"[HAModelAlloc] ✅ HA方案生成完成: "
                   f"{plan.total_models}个模型, "
                   f"{plan.total_instances}个实例, "
                   f"平均冗余{plan.avg_redundancy:.2f}x, "
                   f"SPOF风险{plan.spof_risk_score:.1f}, "
                   f"可用性{plan.estimated_availability*100:.2f}%, "
                   f"耗时{elapsed:.0f}ms")

        return plan

    def _prepare_candidates(
        self,
        exclude_models: Optional[List[str]],
        force_include: Optional[List[str]]
    ) -> List[Any]:
        """准备候选模型列表"""
        candidates = []

        for model_id, spec in self.model_library.items():
            if exclude_models and model_id in exclude_models:
                continue
            candidates.append(spec)

        if force_include:
            forced = [c for c in candidates if c.model_id in force_include]
            others = [c for c in candidates if c.model_id not in force_include]
            candidates = forced + others

        return candidates

    def _classify_models_by_priority(
        self,
        candidates: List[Any]
    ) -> Tuple[List[Any], List[Any]]:
        """将模型分为关键和普通两类"""
        critical_threshold = 1.3  # priority >= 1.3 视为关键

        critical = [c for c in candidates if c.priority >= critical_threshold]
        normal = [c for c in candidates if c.priority < critical_threshold]

        # 各自内部按参数量排序（大的优先）
        critical.sort(key=lambda x: x.param_count, reverse=True)
        normal.sort(key=lambda x: x.param_count, reverse=True)

        return critical, normal

    def _execute_ha_allocation(
        self,
        nodes: List[NodeResource],
        critical_models: List[Any],
        normal_models: List[Any],
        mode: RedundancyMode,
        min_replicas_crit: int,
        min_replicas_norm: int,
        max_snode_ratio: float,
        prioritize_redundancy: bool
    ) -> Tuple[Dict[str, List[ModelInstance]], Dict[str, float]]:
        """
        核心HA分配算法

        策略：
        1. 先为关键模型分配（保证冗余）
        2. 再为普通模型分配（尽力而为）
        3. 每次分配后更新节点可用资源
        4. 持续监控单节点依赖度
        """
        allocations: Dict[str, List[ModelInstance]] = {}
        node_usage: Dict[str, float] = {n.node_id: n.used_memory_mb for n in nodes}
        node_capacity: Dict[str, float] = {n.node_id: n.total_memory_mb for n in nodes}
        node_free: Dict[str, float] = {n.node_id: n.usable_memory_mb for n in nodes}

        # 模型计数器（用于检测单节点依赖）
        models_on_node: Dict[str, Set[str]] = defaultdict(set)  # {node_id: set of model_ids}

        def get_sorted_nodes_by_free() -> List[Tuple[int, NodeResource]]:
            """按可用空间排序节点（返回(index, node)元组）"""
            indexed = list(enumerate(nodes))
            indexed.sort(key=lambda x: node_free.get(x[1].node_id, 0), reverse=True)
            return indexed

        def select_primary_node(
            model_spec: Any,
            candidate_nodes: List[Tuple[int, NodeResource]]
        ) -> Optional[Tuple[int, NodeResource]]:
            """
            选择主节点

            策略：
            1. 能容纳完整模型的节点
            2. 优先选择当前模型负载低的（避免热点）
            3. 在满足条件的选择空间最大的（best-fit）
            """
            best = None
            best_score = -1

            for idx, node in candidate_nodes:
                available = node_free.get(node.node_id, 0)
                needed = model_spec.total_memory_mb

                if available < needed:
                    continue

                # 检查是否会超过单节点限制
                current_models_on_node = len(models_on_node.get(node.node_id, set()))
                total_models_expected = current_models_on_node + 1
                total_models_final = total_models_expected  # 最终会有多少模型

                # 如果加上这个模型会超过限制，跳过
                if len(allocations) > 0 and len(allocations) + len(critical_models) + len(normal_models) > 0:
                    projected_ratio = total_models_on_node / max(len(allocations) + 1, 1)
                    if projected_ratio > max_snode_ratio:
                        continue

                # 评分：综合考虑空间充裕度和当前负载
                space_score = available / needed  # 空间充裕度
                load_score = 1.0 - (node_usage.get(node.node_id, 0) /
                                   node_capacity.get(node.node_id, 1))  # 负载倒数

                score = space_score * 0.6 + load_score * 0.4

                if score > best_score:
                    best_score = score
                    best = (idx, node)

            return best

        def select_standby_node(
            model_spec: Any,
            primary_node_id: str,
            candidate_nodes: List[Tuple[int, NodeResource]]
        ) -> Optional[Tuple[int, NodeResource]]:
            """
            选择备节点（与主节点异构）

            策略：
            1. 不能与主节点相同
            2. 优先选择不同设备类型（如主是A100，备选RTX4090）
            3. 其次选择不同物理位置（如果能获取到的话）
            4. 最后选择剩余空间足够的
            """
            primary_device_type = None
            for idx, node in candidate_nodes:
                if node.node_id == primary_node_id:
                    primary_device_type = node.device_type
                    break

            best = None
            best_score = -1

            for idx, node in candidate_nodes:
                if node.node_id == primary_node_id:
                    continue

                available = node_free.get(node.node_id, 0)
                needed = model_spec.total_memory_mb

                if available < needed:
                    continue

                # 异构加分
                heterogeneity_score = 0.0
                if primary_device_type and node.device_type != primary_device_type:
                    heterogeneity_score = 1.0  # 不同设备类型，满分
                elif primary_device_type:
                    heterogeneity_score = 0.3  # 同类型，低分

                # 空间充裕度
                space_score = min(available / needed, 2.0) / 2.0

                # 综合评分
                score = heterogeneity_score * 0.5 + space_score * 0.5

                if score > best_score:
                    best_score = score
                    best = (idx, node)

            return best

        def allocate_model_instances(
            model_spec: Any,
            target_replicas: int,
            is_critical: bool
        ) -> List[ModelInstance]:
            """为一个模型分配多个实例"""
            instances = []
            sorted_nodes = get_sorted_nodes_by_free()

            # 选择主节点
            primary_selection = select_primary_node(model_spec, sorted_nodes)

            if not primary_selection:
                return []  # 无法分配

            primary_idx, primary_node = primary_selection

            # 创建主实例
            primary_inst = ModelInstance(
                instance_id=f"{model_spec.model_id}::{primary_node.node_id}",
                model_id=model_spec.model_id,
                node_id=primary_node.node_id,
                role=InstanceRole.PRIMARY if mode == RedundancyMode.ACTIVE_PASSIVE else InstanceRole.ACTIVE,
                start_layer=0,
                end_layer=model_spec.total_layers - 1,
                layers_count=model_spec.total_layers,
                memory_mb=model_spec.total_memory_mb,
            )
            instances.append(primary_inst)

            # 更新资源
            mem = model_spec.total_memory_mb
            node_free[primary_node.node_id] -= mem
            node_usage[primary_node.node_id] += mem
            models_on_node[primary_node.node_id].add(model_spec.model_id)

            # 选择备用节点（如果需要）
            remaining_replicas = target_replicas - 1
            used_standby_nodes = {primary_node.node_id}

            for _ in range(remaining_replicas):
                standby_selection = select_standby_node(
                    model_spec,
                    primary_node.node_id,
                    [(i, n) for i, n in enumerate(nodes) if n.node_id not in used_standby_nodes]
                )

                if not standby_selection:
                    break  # 没有更多合适的备节点

                standby_idx, standby_node = standby_selection

                # 创建备用实例
                role = InstanceRole.STANDBY if mode == RedundancyMode.ACTIVE_PASSIVE else InstanceRole.ACTIVE
                standby_inst = ModelInstance(
                    instance_id=f"{model_spec.model_id}::{standby_node.node_id}",
                    model_id=model_spec.model_id,
                    node_id=standby_node.node_id,
                    role=role,
                    start_layer=0,
                    end_layer=model_spec.total_layers - 1,
                    layers_count=model_spec.total_layers,
                    memory_mb=model_spec.total_memory_mb,
                )
                instances.append(standby_inst)

                # 更新资源
                node_free[standby_node.node_id] -= mem
                node_usage[standby_node.node_id] += mem
                models_on_node[standby_node.node_id].add(model_spec.model_id)
                used_standby_nodes.add(standby_node.node_id)

            return instances

        # ===== 开始分配 =====

        # Phase 1: 分配关键模型（保证冗余）
        for model_spec in critical_models:
            instances = allocate_model_instances(
                model_spec,
                target_replicas=min_replicas_crit,
                is_critical=True
            )

            if instances:
                allocations[model_spec.model_id] = instances
                logger.debug(f"[HAModelAlloc] ✅ 关键模型 {model_spec.pretty_name}: "
                           f"{len(instances)}个实例 (主={instances[0].node_id})")

        # Phase 2: 分配普通模型（尽力而为）
        for model_spec in normal_models:
            instances = allocate_model_instances(
                model_spec,
                target_replicas=min_replicas_norm,
                is_critical=False
            )

            if instances:
                allocations[model_spec.model_id] = instances
                logger.debug(f"[HAModelAlloc] ✅ 普通模型 {model_spec.pretty_name}: "
                           f"{len(instances)}个实例")

        return allocations, node_usage

    def _analyze_spof_risks(
        self,
        allocations: Dict[str, List[ModelInstance]],
        nodes: List[NodeResource]
    ) -> Dict[str, Any]:
        """
        分析单点故障(SPOF)风险

        检查项：
        1. 哪些模型只有单实例（无备份）
        2. 哪些节点承载了过多模型（单点依赖）
        3. 整体风险评分
        """
        vulnerable = []
        node_deps: Dict[str, List[str]] = defaultdict(list)

        for model_id, instances in allocations.items():
            active_instances = [inst for inst in instances if inst.is_active]

            if len(active_instances) <= 1:
                vulnerable.append(model_id)

            for inst in instances:
                node_deps[inst.node_id].append(model_id)

        # 计算风险评分
        total_models = len(allocations)
        vulnerable_ratio = len(vulnerable) / max(total_models, 1)

        # 单节点集中度风险
        max_concentration = 0
        for node_id, model_list in node_deps.items():
            concentration = len(model_list) / max(total_models, 1)
            max_concentration = max(max_concentration, concentration)

        risk_score = (
            vulnerable_ratio * 50 +          # 无备份模型占比(50分权重)
            max_concentration * 30 +         # 最大集中度(30分权重)
            (1 - len(nodes)/10) * 20 if len(nodes) < 10 else 0  # 节点少额外风险(20分)
        )

        return {
            "vulnerable_models": vulnerable,
            "single_node_dependencies": dict(node_deps),
            "vulnerable_ratio": vulnerable_ratio,
            "max_concentration": max_concentration,
            "risk_score": min(100, risk_score),
        }

    def _build_ha_plan(
        self,
        mode: RedundancyMode,
        allocations: Dict[str, List[ModelInstance]],
        node_usage: Dict[str, float],
        spof_analysis: Dict[str, Any],
        nodes: List[NodeResource],
        start_time: float
    ) -> HAAllocationPlan:
        """构建最终的HA方案"""

        total_instances = sum(len(insts) for insts in allocations.values())
        total_mem = sum(inst.memory_mb for insts in
                        [item for sublist in allocations.values() for item in sublist])

        avg_redundancy = total_instances / max(len(allocations), 1)

        # 计算节点利用率
        node_util = {}
        for node in nodes:
            used = node_usage.get(node.node_id, node.used_memory_mb)
            util = used / node.total_memory_mb if node.total_memory_mb > 0 else 0
            node_util[node.node_id] = util

        # 预估可用性（简化计算）
        # 假设单节点可用性99.9%，N副本后可用性 = 1-(1-0.999)^N
        node_availability = 0.999
        redundancies = [len(insts) for insts in allocations.values()]
        if redundancies:
            avg_redund = sum(redundancies) / len(redundancies)
            estimated_avail = 1 - ((1 - node_availability) ** avg_redund)
        else:
            estimated_avail = 0.0

        plan = HAAllocationPlan(
            plan_id=f"ha_{int(time.time())}_{hashlib.md5(str(time.time()).encode())[:6]}",
            mode=mode,
            generated_at=time.time(),
            instances=allocations,
            total_models=len(allocations),
            total_instances=total_instances,
            total_memory_mb=total_mem,
            avg_redundancy=avg_redundancy,
            spof_risk_score=spof_analysis["risk_score"],
            vulnerable_models=spof_analysis["vulnerable_models"],
            single_node_dependencies=spof_analysis["single_node_dependencies"],
            node_utilization=node_util,
            estimated_availability=estimated_avail,
            rto_seconds=self.FAILOVER_TIMEOUT_MS / 1000,  # RTO ≈ 故障转移超时
            rpo_requests=0.5,  # 预估最多丢失0.5个请求（理论上应该接近0）
        )

        return plan

    # ============================================================
    #  健康检查系统
    # ============================================================

    async def start_health_monitoring(self):
        """启动后台健康检查"""
        if self._running:
            logger.warning("[HAModelAlloc] 健康监控已在运行")
            return

        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("[HAModelAlloc] 🩺 健康监控系统已启动")

    async def stop_health_monitoring(self):
        """停止健康检查"""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("[HAModelAlloc] 🩺 健康监控系统已停止")

    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
                await self.perform_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[HAModelAlloc] 健康检查循环异常: {e}", exc_info=True)

    async def perform_health_check(self) -> List[HealthCheckResult]:
        """
        执行一轮健康检查

        检查内容：
        1. 心跳检测（节点是否可达）
        2. 响应延迟（是否在正常范围）
        3. 错误率（是否有异常错误）
        4. 资源使用（显存/CPU/GPU）

        Returns:
            所有实例的检查结果
        """
        results = []
        now = time.time()

        if not self.current_ha_plan:
            return results

        for model_id, instances in self.current_ha_plan.instances.items():
            for instance in instances:
                result = await self._check_single_instance(instance, now)
                results.append(result)

                # 更新实例状态
                instance.status = result.status
                instance.last_health_check = now
                if result.latency_ms > 0:
                    instance.avg_latency_ms = result.latency_ms

                # 保存历史
                self.health_history[instance.instance_id].append(result)
                # 只保留最近100条
                if len(self.health_history[instance.instance_id]) > 100:
                    self.health_history[instance.instance_id].pop(0)

        # 检测异常并触发告警
        unhealthy = [r for r in results if r.status != InstanceStatus.HEALTHY]
        if unhealthy:
            logger.warning(f"[HAModelAlloc] ⚠️ 发现 {len(unhealthy)} 个不健康实例:")
            for r in unhealthy[:5]:  # 只打印前5个
                logger.warning(f"  - {r.instance_id}: {r.status.value}"
                             f"(延迟={r.latency_ms:.0f}ms, 错误={r.error})")

            # TODO: 可接入告警系统

        return results

    async def _check_single_instance(
        self,
        instance: ModelInstance,
        check_time: float
    ) -> HealthCheckResult:
        """检查单个实例的健康状况"""

        # 检查心跳超时
        time_since_heartbeat = check_time - instance.last_heartbeat
        if time_since_heartbeat > self.HEARTBEAT_TIMEOUT_SECONDS:
            return HealthCheckResult(
                timestamp=check_time,
                model_id=instance.model_id,
                instance_id=instance.instance_id,
                node_id=instance.node_id,
                status=InstanceStatus.DOWN,
                error=f"心跳超时({time_since_heartbeat:.0f}s)",
            )

        # TODO: 实际的健康检查逻辑（调用节点的health check接口）
        # 这里是模拟实现
        try:
            # 模拟：90%概率健康，8%降级，2%不健康
            rand_val = random.random()
            if rand_val < 0.90:
                status = InstanceStatus.HEALTHY
                latency = random.uniform(20, 150)
            elif rand_val < 0.98:
                status = InstanceStatus.DEGRADED
                latency = random.uniform(200, 800)
            else:
                status = InstanceStatus.UNHEALTHY
                latency = random.uniform(1000, 3000)

            return HealthCheckResult(
                timestamp=check_time,
                model_id=instance.model_id,
                instance_id=instance.instance_id,
                node_id=instance.node_id,
                status=status,
                latency_ms=latency,
                details={
                    "request_count": instance.request_count,
                    "error_rate": instance.error_rate,
                    "uptime": instance.uptime_seconds,
                }
            )
        except Exception as e:
            return HealthCheckResult(
                timestamp=check_time,
                model_id=instance.model_id,
                instance_id=instance.instance_id,
                node_id=instance.node_id,
                status=InstanceStatus.UNHEALTHY,
                error=str(e),
            )

    def get_health_status(self) -> Dict[str, Any]:
        """获取整体健康状态"""
        if not self.current_ha_plan:
            return {"status": "no_plan", "message": "无活动方案"}

        healthy = 0
        degraded = 0
        unhealthy = 0
        down = 0

        for instances in self.current_ha_plan.instances.values():
            for inst in instances:
                if inst.status == InstanceStatus.HEALTHY:
                    healthy += 1
                elif inst.status == InstanceStatus.DEGRADED:
                    degraded += 1
                elif inst.status == InstanceStatus.UNHEALTHY:
                    unhealthy += 1
                elif inst.status == InstanceStatus.DOWN:
                    down += 1

        total = healthy + degraded + unhealthy + down
        health_pct = (healthy / total * 100) if total > 0 else 0

        return {
            "timestamp": time.time(),
            "overall_status": "healthy" if health_pct >= 95 else
                              "degraded" if health_pct >= 80 else "unhealthy",
            "health_percentage": round(health_pct, 1),
            "instances": {
                "total": total,
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
                "down": down,
            },
            "models_with_issues": [
                mid for mid, insts in self.current_ha_plan.instances.items()
                if any(i.status != InstanceStatus.HEALTHY for i in insts)
            ]
        }

    # ============================================================
    #  故障转移系统
    # ============================================================

    async def failover(
        self,
        model_id: str,
        failed_node: str,
        auto_promote: bool = True
    ) -> FailoverRecord:
        """
        执行故障转移

        流程：
        1. 定位失败的实例
        2. 查找可用的备用实例
        3. 将备用提升为主实例
        4. 更新路由表
        5. 记录转移日志

        Args:
            model_id: 发生故障的模型ID
            failed_node: 故障节点ID
            auto_promote: 是否自动提升备用实例

        Returns:
            FailoverRecord 转移记录
        """
        start_time = time.time()
        record_id = f"fo_{int(time.time())}_{random.randint(1000,9999)}"

        logger.warning(f"[HAModelAlloc] 🚨 开始故障转移: "
                      f"模型={model_id}, 故障节点={failed_node}")

        # Step 1: 查找失败实例
        failed_instance_id = f"{model_id}::{failed_node}"
        failed_instance = self.active_instances.get(failed_instance_id)

        if not failed_instance and self.current_ha_plan:
            # 从方案中查找
            for inst in self.current_ha_plan.instances.get(model_id, []):
                if inst.node_id == failed_node:
                    failed_instance = inst
                    break

        if not failed_instance:
            raise ValueError(f"未找到失败实例: {failed_instance_id}")

        # 标记为宕机
        failed_instance.status = InstanceStatus.DOWN

        # Step 2: 查找备用实例
        standby_instance = None
        if self.current_ha_plan:
            for inst in self.current_ha_plan.instances.get(model_id, []):
                if (inst.node_id != failed_node and
                    inst.role in (InstanceRole.STANDBY, InstanceRole.ACTIVE) and
                    inst.status in (InstanceStatus.HEALTHY, InstanceStatus.DEGRADED)):
                    standby_instance = inst
                    break

        # Step 3: 执行提升（如果有备用）
        promoted_node = None
        promoted_instance_id = None
        success = False

        if standby_instance and auto_promote:
            # 提升备用为主
            old_role = standby_instance.role
            standby_instance.role = InstanceRole.PRIMARY
            standby_instance.status = InstanceStatus.HEALTHY

            promoted_node = standby_instance.node_id
            promoted_instance_id = standby_instance.instance_id

            # TODO: 更新路由表，将流量导向新的主实例
            # await self._update_routing_table(model_id, promoted_node)

            success = True

            logger.info(f"[HAModelAlloc] ✅ 故障转移成功: "
                       f"{failed_node} → {promoted_node}")
        else:
            logger.error(f"[HAModelAlloc] ❌ 故障转移失败: "
                        f"无可用备用实例 (模型={model_id})")

        # Step 4: 记录转移记录
        elapsed_ms = (time.time() - start_time) * 1000

        record = FailoverRecord(
            record_id=record_id,
            timestamp=time.time(),
            model_id=model_id,
            failed_instance=failed_instance_id,
            failed_node=failed_node,
            promoted_instance=promoted_instance_id,
            promoted_node=promoted_node,
            failover_state=FailoverState.FAILOVER_COMPLETED if success else FailoverState.FAILOVER_IN_PROGRESS,
            duration_ms=elapsed_ms,
            requests_lost=0,  # TODO: 实际统计
            success=success
        )

        self.failover_records.append(record)

        return record

    def get_failover_history(self, limit: int = 20) -> List[Dict]:
        """获取故障转移历史"""
        records = self.failover_records[-limit:]

        return [
            {
                "record_id": r.record_id,
                "timestamp": r.timestamp,
                "model_id": r.model_id,
                "failed_node": r.failed_node,
                "promoted_node": r.promoted_node,
                "state": r.failover_state.value,
                "duration_ms": round(r.duration_ms, 1),
                "requests_lost": r.requests_lost,
                "success": r.success,
            }
            for r in reversed(records)
        ]

    async def recover_node(self, node_id: str) -> Dict[str, Any]:
        """
        节点恢复后的操作

        流程：
        1. 标记节点上的所有实例为恢复中
        2. 对于之前故障转移过的模型：
           a. 可以选择将恢复后的实例作为新的备用
           b. 或者保持现状（让新主继续服务）
        3. 更新状态
        """
        recovered_instances = []
        re_demoted = []

        if self.current_ha_plan:
            for model_id, instances in self.current_ha_plan.instances.items():
                for inst in instances:
                    if inst.node_id == node_id and inst.status == InstanceStatus.DOWN:
                        # 恢复实例
                        old_status = inst.status
                        inst.status = InstanceStatus.STARTING
                        recovered_instances.append({
                            "model_id": model_id,
                            "instance_id": inst.instance_id,
                            "old_status": old_status.value,
                            "new_status": inst.status.value,
                        })

                        # 检查该模型是否已经故障转移过
                        recent_failover = None
                        for fo in reversed(self.failover_records):
                            if (fo.model_id == model_id and
                                fo.failed_node == node_id and
                                fo.success):
                                recent_failover = fo
                                break

                        if recent_failover:
                            # 该模型已有新的主实例，将恢复的实例降为备用
                            inst.role = InstanceRole.STANDBY
                            inst.status = InstanceStatus.HEALTHY
                            re_demoted.append({
                                "model_id": model_id,
                                "action": "demoted_to_standby",
                                "reason": "已有新主实例，此实例作为备用"
                            })

        logger.info(f"[HAModelAlloc] 🔄 节点 {node_id} 恢复: "
                   f"{len(recovered_instances)}个实例恢复, "
                   f"{len(re_demoted)}个降为备用")

        return {
            "node_id": node_id,
            "timestamp": time.time(),
            "recovered_instances": recovered_instances,
            "re_demoted": re_demoted,
        }


