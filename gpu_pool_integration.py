"""
EXO Manager - GPU池管理集成
============================

将GPU池管理功能集成到Manager系统中，提供：
- 统一的模型加载/卸载接口
- 智能分片分配策略（单节点优先 + 安全余量 + 在线增量分配）
- 简化版动态重平衡
- 资源预览与优化建议

核心设计原则：
1. 能不拆就不拆：单节点完整加载 > 跨节点分片（避免 hidden_state 传输延迟）
2. 安全余量：单节点加载时保留 30%+ 余量给 KV Cache
3. 在线决策：支持时序请求场景下的增量分配
4. 全局视角：考虑已有模型占用，避免碎片化

使用方式:
    from gpu_pool_integration import GPUPoolIntegration
    
    pool = GPUPoolIntegration(cluster_manager)
    await pool.load_model("Qwen/Qwen3-4B", "./models")
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# 延迟导入 sys_log（避免循环依赖）
def _get_sys_log():
    try:
        from sys_logger import sys_log
        return sys_log
    except ImportError:
        return None


@dataclass
class ModelAllocation:
    """模型分片分配方案"""
    model_id: str
    total_layers: int
    allocations: List[Dict[str, Any]]  # [{node_id, start_layer, end_layer, layers_count}]
    strategy: str
    estimated_memory_per_node: Dict[str, float]  # {node_id: memory_gb}
    # 新增字段
    allocation_type: str = "sharded"  # "single_node" | "sharded" | "rebalanced" | "pending_more_nodes"
    decision_reason: str = ""         # 决策原因说明
    safety_warnings: List[str] = None  # 安全警告（如余量不足等）
    min_nodes_required: int = 0       # pending 状态：至少需要多少节点才能分配


# ============================================================
#  智能分配器 - 核心算法
# ============================================================

class SmartAllocator:
    """
    在线智能模型分配器
    
    核心能力：
    1. 单节点优先：能不拆就不拆，避免跨节点 hidden_state 传输延迟
    2. 安全余量检查：单节点加载时保留足够空间给 KV Cache
    3. 在线增量决策：支持时序到达的请求序列，考虑已有占用
    4. 简化版重平衡：当直接放置失败时尝试迁移已有小模型
    
    安全等级定义：
    - SAFE (>=30% 余量):     单节点完整加载，KV Cache 空间充裕
    - WARNING (10%~30%):    可以单节点运行，但长上下文可能 OOM
    - DANGER (<10% 或 >90%): 极限情况，默认走拆分方案
    - MUST_SHARD (>100%):    单节点装不下，必须拆分
    """

    # 安全阈值配置
    SAFE_RATIO = 0.70      # 安全线：模型大小 ≤ 节点显存 × 70%
    WARNING_RATIO = 0.90   # 警告线：模型大小 ≤ 节点显存 × 90%

    # 运行时开销系数：除权重外，需额外预留 KV Cache / 激活值 / CUDA 碎片 / 临时缓冲区等
    RUNTIME_OVERHEAD_RATIO = 1.30

    def __init__(self, manager):
        self.manager = manager

    def get_nodes_with_usage(self) -> List[Dict]:
        """
        获取节点列表（含实时使用信息）

        Returns:
            [{
                node_id, address, memory_mb,
                used_mb,          # 已被其他模型占用的显存
                free_mb,          # 剩余可用显存
                loaded_models,    # 当前已加载的模型数
                device, flops
            }]
        """
        nodes = []
        for node_id, node_info in self.manager.nodes.items():
            if node_info.status.value not in ('online', 'connecting'):
                continue

            raw_mem = node_info.device_info.get("memory", 0)
            mem_detail = node_info.device_info.get("memory_detail")
            total_mem = raw_mem
            if (total_mem == 0 or total_mem is None) and mem_detail and mem_detail.get("total", 0) > 0:
                total_mem = mem_detail["total"]

            # 计算已用显存（基于已加载模型的分片估算）
            used_mb = self._estimate_node_used_memory(node_id)

            nodes.append({
                "node_id": node_id,
                "address": f"{node_info.address}:{node_info.port}",
                "memory_mb": total_mem,
                "used_mb": used_mb,
                "free_mb": max(0, total_mem - used_mb),
                "loaded_models": len(node_info.loaded_models),
                "device": node_info.device_info.get("chip", "Unknown"),
                "flops": node_info.device_info.get("flops", {}),
            })

        return nodes

    def _estimate_node_used_memory(self, node_id: str) -> float:
        """估算某节点已被模型占用的显存（按模型库中各模型配置的每层显存计算）"""
        node_info = self.manager.nodes.get(node_id)
        if not node_info:
            return 0.0

        # 动态导入避免循环依赖
        from auto_model_allocator import get_model_library
        model_lib = get_model_library()

        total_used = 0.0
        for model in node_info.loaded_models:
            shard = model.get("shard", {})
            n_layers = shard.get("end_layer", 0) - shard.get("start_layer", 0) + 1

            # 根据 base_model_id 查找模型库中的每层显存配置
            model_id = model.get("model_id", "")
            base_model_id = model.get("base_model_id") or (
                model_id.split("::")[0] if "::" in model_id else model_id
            )
            layer_mem = 100.0  # 默认回退
            spec = model_lib.get(base_model_id)
            if not spec:
                # 尝试用 HF repo ID 反向查找
                for key, lib_spec in model_lib.items():
                    if lib_spec.model_id == base_model_id:
                        spec = lib_spec
                        break
            if spec:
                # 使用运行时每层显存（含 KV Cache / 激活值 / 碎片开销）
                layer_mem = spec.runtime_memory_mb / max(spec.total_layers, 1)
            else:
                layer_mem = 100.0 * 1.30  # 默认回退并带开销

            total_used += n_layers * layer_mem

        # 如果有 pynvml 实时数据，优先使用（真实占用通常更高）
        mem_detail = node_info.device_info.get("memory_detail")
        if mem_detail and mem_detail.get("used", 0) > total_used:
            return mem_detail["used"]

        return total_used

    def classify_model(self, model_size_mb: float, max_node_memory_mb: float) -> Tuple[str, str]:
        """
        判定模型相对于单节点的安全等级

        Returns:
            (level, reason) 其中 level 为 SAFE / WARNING / DANGER / MUST_SHARD
        """
        ratio = model_size_mb / max_node_memory_mb if max_node_memory_mb > 0 else 999

        if ratio > 1.0:
            return ("MUST_SHARD", f"模型({model_size_mb/1024:.1f}GB)超过单节点最大显存({max_node_memory_mb/1024:.1f}GB)")
        elif ratio > self.WARNING_RATIO:
            return ("DANGER", f"模型占比{ratio*100:.0f}%，余量不足10%，极易OOM")
        elif ratio > self.SAFE_RATIO:
            return ("WARNING", f"模型占比{ratio*100:.0f}%，余量{(1-ratio)*100:.0f}%，短对话可行")
        else:
            return ("SAFE", f"模型占比{ratio*100:.0f}%，余量充足(≥30%)")

    def allocate(
        self,
        model_id: str,
        total_layers: int,
        layer_memory_mb: float = 100.0,
        strategy: str = "smart",
        force_shard: bool = False,
        nodes: Optional[List[Dict]] = None,
        ignore_used_memory: bool = False
    ) -> ModelAllocation:
        """
        智能分配入口（同步版本，供 preview / rebalance 使用）

        Args:
            model_id: 模型标识符
            total_layers: 总层数
            layer_memory_mb: 每层预估显存(MB)
            strategy: 分配策略 ('smart' | 'memory_weighted' | 'uniform' | 'performance_weighted')
            force_shard: 强制拆分（跳过单节点检查）
            nodes: 可选的外部节点列表（用于模拟分配），不传则使用当前在线节点
            ignore_used_memory: Rebalance 模式专用。若为 True，则忽略节点当前已加载模型
                                的显存占用（视为空节点），用于"先卸载再分配"的场景，
                                避免旧分片占用导致 free_mb 偏小从而误判为 pending_more_nodes。

        Returns:
            ModelAllocation 包含完整的分配方案和决策元信息
        """
        if nodes is None:
            nodes = self.get_nodes_with_usage()
        if not nodes:
            raise ValueError("没有可用的在线节点")

        # [FIX-Rebalance] rebalance_model 会先卸载旧分片再 allocate，此时旧 used_mb 是
        # 历史遗留数据，不应该参与新分配方案的 free_mb / total_free_mem 计算。
        if ignore_used_memory:
            _adjusted = []
            for n in nodes:
                _copy = dict(n)
                _copy["used_mb"] = 0.0
                _copy["free_mb"] = float(_copy.get("memory_mb", 0))
                _adjusted.append(_copy)
            nodes = _adjusted

        # 计入运行时开销：KV Cache / 激活值 / CUDA 碎片 / 临时缓冲区等
        effective_layer_mb = layer_memory_mb * self.RUNTIME_OVERHEAD_RATIO
        model_total_mb = total_layers * effective_layer_mb
        max_node_mem = max(n["memory_mb"] for n in nodes) if nodes else 0
        max_free_mem = max(n["free_mb"] for n in nodes) if nodes else 0
        total_free_mem = sum(n["free_mb"] for n in nodes)

        logger.info(f"🧠 [SmartAlloc] 模型 {model_id}: {model_total_mb/1024:.1f}GB "
                   f"(含{self.RUNTIME_OVERHEAD_RATIO*100-100:.0f}%运行时开销), "
                   f"{total_layers}层, 可用节点={len(nodes)}, "
                   f"最大节点显存={max_node_mem/1024:.1f}GB, "
                   f"最大剩余={max_free_mem/1024:.1f}GB, 总剩余={total_free_mem/1024:.1f}GB")

        # ====== smart 策略：智能决策流程 ======
        if strategy == "smart" and not force_shard:

            # Phase 1: 尝试单节点完整加载
            single_result = self._try_single_node_allocation(
                nodes, model_id, total_layers, model_total_mb, effective_layer_mb
            )
            if single_result:
                return single_result

            # Phase 2: 单节点不可行 → 检查集群总剩余是否足够
            logger.info(f"🧠 [SmartAlloc] 单节点不可行，切换到拆分策略")

            # 计算至少需要多少节点（每个节点至少能放1层 + 余量）
            per_node_need = effective_layer_mb * 1.15  # 每节点至少1层+余量
            min_nodes = max(2, int(model_total_mb * 1.15 / max(n["free_mb"] for n in nodes)) + 1)

            if total_free_mem < model_total_mb * 1.15:  # 留15%余量
                # 不再直接拒绝，返回 pending 状态等待更多节点
                logger.warning(f"🧠 [SmartAlloc] ⏳ 集群显存不足，需要更多节点: "
                              f"当前{len(nodes)}节点总剩余{total_free_mem/1024:.1f}GB, "
                              f"模型需要{model_total_mb*1.15/1024:.1f}GB, "
                              f"预计需要≥{min_nodes}个节点")
                return ModelAllocation(
                    model_id=model_id,
                    total_layers=total_layers,
                    allocations=[],
                    strategy=strategy,
                    estimated_memory_per_node={},
                    allocation_type="pending_more_nodes",
                    decision_reason=f"集群显存不足({len(nodes)}节点, 总剩余{total_free_mem/1024:.1f}GB), 需要≥{min_nodes}个节点",
                    safety_warnings=[f"当前{len(nodes)}个节点不足以分配此模型，等待新节点加入"],
                    min_nodes_required=min_nodes
                )

        # ====== 传统拆分策略 ======
        sharded_allocations = self._allocate_sharded(
            nodes, total_layers, effective_layer_mb, strategy
        )

        estimated_memory = {}
        for alloc in sharded_allocations:
            estimated_memory[alloc["node_id"]] = round(
                alloc["layers_count"] * effective_layer_mb / 1024, 2
            )

        level, reason = self.classify_model(model_total_mb, max_free_mem)

        result = ModelAllocation(
            model_id=model_id,
            total_layers=total_layers,
            allocations=sharded_allocations,
            strategy=strategy,
            estimated_memory_per_node=estimated_memory,
            allocation_type="sharded",
            decision_reason=f"拆分分配({strategy}) - {reason}",
            safety_warnings=[reason] if level in ("DANGER", "MUST_SHARD") else []
        )

        # 发射系统日志
        _sys = _get_sys_log()
        if _sys:
            _sys.log(
                _sys.INFO if level in ("SAFE", "WARNING") else _sys.WARNING,
                "allocation",
                f"模型 {model_id} 分配完成: {result.allocation_type} - {result.decision_reason}",
                {"strategy": strategy, "nodes": len(sharded_allocations), "total_layers": total_layers}
            )

        return result

    def _try_single_node_allocation(
        self,
        nodes: List[Dict],
        model_id: str,
        total_layers: int,
        model_total_mb: float,
        layer_memory_mb: float
    ) -> Optional[ModelAllocation]:
        """
        尝试单节点完整加载

        选择逻辑：
        1. 优先选 free_mb >= model_total_mb * 1.15 的节点（留15%额外余量）
        2. 其次选 free_mb >= model_total_mb 且 ratio <= SAFE_RATIO 的节点
        3. 最后选 free_mb >= model_total_mb 且 ratio <= WARNING_RATIO 的节点（带警告）
        """
        candidates = []

        for node in nodes:
            free_mb = node["free_mb"]
            total_mem = node["memory_mb"]

            if free_mb < model_total_mb:
                continue  # 空间不够，跳过

            ratio = model_total_mb / total_mem if total_mem > 0 else 999

            if ratio <= self.SAFE_RATIO:
                priority = 0  # 最优
            elif ratio <= self.WARNING_RATIO:
                priority = 1  # 可行但有风险
            else:
                priority = 2  # 危险区域

            candidates.append({
                **node,
                "_ratio": ratio,
                "_priority": priority,
                "_waste": free_mb - model_total_mb  # 浪费的空间（越小越好）
            })

        if not candidates:
            logger.info(f"🧠 [SmartAlloc] 无节点可容纳 {model_total_mb/1024:.1f}GB 单节点加载")
            return None

        # 排序：优先级最低的在前，同优先级下浪费最少的在前
        candidates.sort(key=lambda c: (c["_priority"], c["_waste"]))
        best = candidates[0]

        warnings = []
        if best["_priority"] == 1:
            warnings.append(f"余量仅 {(1-best['_ratio'])*100:.0f}%，建议控制上下文长度")
        elif best["_priority"] == 2:
            warnings.append(f"⚠️ 余量不足10%，存在OOM风险！")

        logger.info(f"🧠 [SmartAlloc] ✅ 单节点完整加载 → 节点 {best['node_id']} "
                   f"(模型{model_total_mb/1024:.1f}GB / 显存{best['memory_mb']/1024:.1f}GB, "
                   f"占比{best['_ratio']*100:.0f}%, 优先级P{best['_priority']})")

        allocations = [{
            "node_id": best["node_id"],
            "start_layer": 0,
            "end_layer": total_layers - 1,
            "layers_count": total_layers,
            "address": best["address"]
        }]

        return ModelAllocation(
            model_id=model_id,
            total_layers=total_layers,
            allocations=allocations,
            strategy="smart_single_node",
            estimated_memory_per_node={best["node_id"]: round(model_total_mb / 1024, 2)},
            allocation_type="single_node",
            decision_reason=f"单节点完整加载({best['node_id']}) - 占比{best['_ratio']*100:.0f}%",
            safety_warnings=warnings if warnings else None
        )

        # 发射系统日志：单节点分配成功
        _sys = _get_sys_log()
        if _sys:
            log_level = _sys.WARNING if best["_priority"] >= 1 else _sys.SUCCESS
            _sys.log(
                log_level,
                "allocation",
                f"模型 {model_id} → 单节点完整加载 @ {best['node_id']}",
                {"node": best["node_id"], "size_gb": round(model_total_mb / 1024, 1),
                 "ratio": round(best["_ratio"] * 100, 0), "priority": best["_priority"]}
            )

    def _allocate_sharded(
        self,
        nodes: List[Dict],
        total_layers: int,
        layer_memory_mb: float,
        strategy: str
    ) -> List[Dict]:
        """
        拆分分配（复用原有三种策略）

        注意：这里传入的是带 usage 信息的节点，但只使用基础字段做分配，
        实际放置可行性由上层 _try_single_node_allocation 保证。
        """
        # 提取纯节点信息（兼容旧的分配函数接口）
        plain_nodes = [
            {
                "node_id": n["node_id"],
                "address": n["address"],
                "memory_mb": n["memory_mb"],  # 用总显存而非剩余显存做权重
                "free_mb": n.get("free_mb", 0),  # [FIX] 传递剩余显存用于验证
                "flops": n.get("flops", {}),
                "loaded_models": n["loaded_models"]
            }
            for n in nodes
        ]

        if strategy in ("memory_weighted", "smart"):
            raw_allocs = self._legacy_allocate_by_memory(plain_nodes, total_layers, layer_memory_mb)
        elif strategy == "uniform":
            raw_allocs = self._legacy_allocate_uniformly(plain_nodes, total_layers)
        elif strategy == "performance_weighted":
            raw_allocs = self._legacy_allocate_by_performance(plain_nodes, total_layers, layer_memory_mb)
        else:
            raw_allocs = self._legacy_allocate_by_memory(plain_nodes, total_layers, layer_memory_mb)

        # [FIX] 验证每个节点的分配不超过其剩余显存
        validated = []
        free_map = {n["node_id"]: n.get("free_mb", 0) for n in nodes}
        for alloc in raw_allocs:
            node_free = free_map.get(alloc["node_id"], 0)
            alloc_mem = alloc["layers_count"] * layer_memory_mb
            if node_free < alloc_mem:
                logger.warning(f"🧠 [SmartAlloc] ⚠️ 节点 {alloc['node_id']} 剩余不足: "
                              f"{node_free/1024:.1f}GB < 分配需要{alloc_mem/1024:.1f}GB，跳过")
                continue
            validated.append(alloc)

        if not validated:
            raise ValueError(
                f"所有节点剩余显存均不足以承载拆分后的分片 "
                f"(每层约{layer_memory_mb}MB, 总需{total_layers * layer_memory_mb / 1024:.1f}GB)"
            )

        return validated

    # ---- 以下为原有的三种分配策略（保持向后兼容）----

    def _legacy_allocate_by_memory(self, nodes, total_layers, layer_memory_mb):
        """基于内存容量的加权分配"""
        total_memory = sum(n["memory_mb"] for n in nodes)
        if total_memory <= 0:
            return self._legacy_allocate_uniformly(nodes, total_layers)

        allocations = []
        current_layer = 0
        for node in sorted(nodes, key=lambda x: x["memory_mb"], reverse=True):
            if current_layer >= total_layers:
                break
            weight = node["memory_mb"] / total_memory
            layers_for_node = max(1, int(total_layers * weight))
            layers_for_node = min(layers_for_node, total_layers - current_layer)
            end_layer = current_layer + layers_for_node - 1
            allocations.append({
                "node_id": node["node_id"], "start_layer": current_layer,
                "end_layer": end_layer, "layers_count": layers_for_node,
                "address": node["address"]
            })
            current_layer = end_layer + 1

        if current_layer < total_layers and allocations:
            last_alloc = allocations[-1]
            remaining = total_layers - current_layer
            last_alloc["end_layer"] = total_layers - 1
            last_alloc["layers_count"] += remaining
        return allocations

    def _legacy_allocate_uniformly(self, nodes, total_layers):
        """均匀分配"""
        num_nodes = len(nodes)
        base_layers = total_layers // num_nodes
        remainder = total_layers % num_nodes
        allocations = []
        current_layer = 0
        for i, node in enumerate(nodes):
            layers_for_node = base_layers + (1 if i < remainder else 0)
            end_layer = current_layer + layers_for_node - 1
            allocations.append({
                "node_id": node["node_id"], "start_layer": current_layer,
                "end_layer": end_layer, "layers_count": layers_for_node,
                "address": node["address"]
            })
            current_layer = end_layer + 1
        return allocations

    def _legacy_allocate_by_performance(self, nodes, total_layers, layer_memory_mb):
        """基于性能加权分配

        [FIX] 原实现在 load_penalty 较大（rtx-3090 场景 score=31）或
              score 取整导致最后节点分不到层时，会出现层数和 < total_layers
              或 区间不闭合。这里：
              1) score 归一化到 max(score, 0.01) 避免除零/负权重；
              2) 非末尾节点保留至少后面 1 层，末尾节点吃掉所有剩余；
              3) 最后兜底把剩余层加给末尾节点，保证严格闭合。
        """
        scored_nodes = []
        for node in nodes:
            flops = node.get("flops", {})
            fp16_flops = flops.get("fp16", 10.0)
            load_penalty = node["loaded_models"] * 2
            score = fp16_flops - load_penalty
            scored_nodes.append({**node, "score": score})

        scored_nodes.sort(key=lambda x: x["score"], reverse=True)
        raw_scores = [max(n["score"], 0.01) for n in scored_nodes]
        total_score = sum(raw_scores)

        allocations = []
        current_layer = 0
        N = len(scored_nodes)
        for idx, node in enumerate(scored_nodes):
            if current_layer >= total_layers:
                break

            weight = raw_scores[idx] / total_score
            layers_for_node = max(1, int(total_layers * weight))
            is_last = idx == N - 1
            if not is_last:
                # 保留后面每个节点至少一层
                remain_after = N - idx - 1
                upper = max(1, total_layers - current_layer - remain_after)
                layers_for_node = max(1, min(layers_for_node, upper))
            else:
                layers_for_node = total_layers - current_layer
            # 钳制兜底
            layers_for_node = max(1, min(layers_for_node, total_layers - current_layer))

            end_layer = current_layer + layers_for_node - 1
            allocations.append({
                "node_id": node["node_id"], "start_layer": current_layer,
                "end_layer": end_layer, "layers_count": layers_for_node,
                "address": node["address"]
            })
            current_layer = end_layer + 1

        # 最终兜底：若取整/截断导致还有剩余，挂到末尾
        if current_layer < total_layers and allocations:
            last_alloc = allocations[-1]
            remaining = total_layers - current_layer
            last_alloc["end_layer"] += remaining
            last_alloc["layers_count"] += remaining
            current_layer += remaining

        return allocations

    # ============================================================
    #  故障恢复 - 节点掉线处理
    # ============================================================

    def handle_node_failure(self, failed_node_id: str) -> Dict[str, Any]:
        """
        处理节点掉线故障

        当一个节点离线时：
        1. 识别该节点上承载的所有模型分片
        2. 对每个受影响模型判断恢复可行性
        3. 尝试在存活节点上重新分配（迁移）
        4. 返回恢复报告

        Args:
            failed_node_id: 掉线的节点ID

        Returns:
            {
                "failed_node": str,
                "affected_models": [      # 受影响的模型列表
                    {
                        "model_id": str,
                        "shard_on_failed": {start_layer, end_layer},
                        "recovery_action": "migrated" | "degraded" | "lost",
                        "recovery_detail": str,
                        "new_allocation": [...] | None   # 迁移成功时的新方案
                    }
                ],
                "summary": {"total", "migrated", "degraded", "lost"}
            }
        """
        logger.warning(f"🚨 [FaultRecovery] 开始处理节点掉线: {failed_node_id}")

        # Step 1: 找出该节点上的所有分片
        affected = self._find_affected_models(failed_node_id)

        if not affected:
            logger.info(f"✅ [FaultRecovery] 节点 {failed_node_id} 上无已加载模型，无需恢复")
            return {
                "failed_node": failed_node_id,
                "affected_models": [],
                "summary": {"total": 0, "migrated": 0, "degraded": 0, "lost": 0}
            }

        logger.warning(f"⚠️ [FaultRecovery] 发现 {len(affected)} 个受影响模型: "
                      f"{[a['model_id'] for a in affected]}")

        # Step 2: 获取当前存活的可用节点
        alive_nodes = self.get_nodes_with_usage()
        alive_nodes = [n for n in alive_nodes if n["node_id"] != failed_node_id]

        if not alive_nodes:
            logger.error(f"❌ [FaultRecovery] 无可用存活节点，所有受影响模型将丢失")
            return self._build_loss_report(failed_node_id, affected)

        # Step 3: 逐个模型尝试恢复
        recovery_results = []
        summary = {"total": len(affected), "migrated": 0, "degraded": 0, "lost": 0}

        for model_info in affected:
            result = self._recover_single_model(
                model_info, failed_node_id, alive_nodes
            )
            recovery_results.append(result)
            summary[result["recovery_action"] + "ed" if result["recovery_action"] in ("migrat", "degrad") else result["recovery_action"]] += 1

        report = {
            "failed_node": failed_node_id,
            "affected_models": recovery_results,
            "summary": summary
        }

        action_counts = ", ".join(f"{k}={v}" for k, v in summary.items())
        logger.info(f"[FaultRecovery] 恢复完成: {action_counts}")

        # 发射系统日志：故障恢复报告
        _sys = _get_sys_log()
        if _sys:
            has_loss = summary["lost"] > 0
            has_success = summary["migrated"] > 0
            if has_loss:
                log_level = _sys.ERROR
            elif has_success:
                log_level = _sys.SUCCESS
            else:
                log_level = _sys.WARNING

            _sys.log(
                log_level,
                "fault-recovery",
                f"节点 {failed_node_id} 掉线: 受影响{summary['total']}个模型, "
                f"迁移{summary['migrated']}, 降级{summary['degraded']}, 丢失{summary['lost']}",
                {"failed_node": failed_node_id, **summary, "models": [r["model_id"] for r in recovery_results]}
            )

        return report

    def _find_affected_models(self, failed_node_id: str) -> List[Dict]:
        """找出指定节点上承载的所有模型分片"""
        affected = []
        node_info = self.manager.nodes.get(failed_node_id)

        if not node_info:
            return affected

        for model in node_info.loaded_models:
            model_id = model.get("model_id", "")
            shard = model.get("shard", {})

            # 提取基础模型ID（去掉 ::instance 后缀）
            base_id = model_id.split("::")[0] if "::" in model_id else model_id

            affected.append({
                "model_id": model_id,
                "base_model_id": base_id,
                "shard": {
                    "start_layer": shard.get("start_layer", 0),
                    "end_layer": shard.get("end_layer", 0),
                    "n_layers": shard.get("n_layers", 0)
                },
                "instance_id": model.get("instance_id", "default")
            })

        return affected

    def _recover_single_model(
        self,
        model_info: Dict,
        failed_node_id: str,
        alive_nodes: List[Dict]
    ) -> Dict:
        """
        尝试恢复单个受影响的模型

        恢复策略优先级：
        1. 完整迁移：如果某个存活节点能完整容纳该模型的全部层 → 迁移过去
        2. 分片重组：用多个存活节点重新分担该掉线节点的分片部分
        3. 降级运行：模型不完整，但剩余分片仍可提供有限服务
        4. 完全丢失：无法恢复，标记为 lost
        """
        model_id = model_info["model_id"]
        base_id = model_info["base_model_id"]
        lost_shard = model_info["shard"]
        lost_layers = lost_shard["end_layer"] - lost_shard["start_layer"] + 1
        total_layers = lost_shard.get("n_layers", lost_layers)

        # [FIX] 动态估算每层显存：优先查模型库；找不到回退 100MB/层
        # 避免硬编码 100MB/层 导致判断偏差（4GB 节点小模型/大模型误判可行性）
        layer_memory_mb = 100.0
        try:
            from auto_model_allocator import get_model_library
            _lib = get_model_library()
            _spec = _lib.get(base_id) or _lib.get(model_id)
            if _spec:
                _lmb = _spec.get("layer_memory_mb") or (
                    (_spec.get("parameters_mb") or 0) / total_layers if total_layers > 0 else None
                )
                if _lmb and _lmb > 0:
                    layer_memory_mb = float(_lmb)
        except Exception:
            pass
        layer_memory_mb = max(10.0, float(layer_memory_mb))

        logger.info(f"🔧 [FaultRecovery] 处理模型 {model_id}: "
                   f"丢失 L{lost_shard['start_layer']}-L{lost_shard['end_layer']} ({lost_layers}层), "
                   f"每层估算={layer_memory_mb:.0f}MB")

        # 策略1: 尝试找单个存活节点完整容纳整个模型
        single_node_result = self._try_migrate_to_single_node(
            model_id, total_layers, alive_nodes, layer_memory_mb=layer_memory_mb
        )
        if single_node_result:
            return single_node_result

        # 策略2: 尝试用存活节点重新分担丢失的分片
        reshared_result = self._try_reshare_shard(
            model_info, failed_node_id, alive_nodes, layer_memory_mb=layer_memory_mb
        )
        if reshared_result:
            return reshared_result

        # 策略3: 检查是否可以降级运行（其他分片还在）
        degraded = self._check_degraded_mode(model_id, failed_node_id)
        if degraded:
            return degraded

        # 策略4: 完全无法恢复
        logger.error(f"❌ [FaultRecovery] 模型 {model_id} 无法恢复")
        return {
            "model_id": model_id,
            "shard_on_failed": lost_shard,
            "recovery_action": "lost",
            "recovery_detail": f"无足够资源恢复，模型 {model_id} 已不可用",
            "new_allocation": None
        }

    def _try_migrate_to_single_node(
        self,
        model_id: str,
        total_layers: int,
        alive_nodes: List[Dict],
        layer_memory_mb: float = 100.0
    ) -> Optional[Dict]:
        """策略1: 尝试迁移到单个存活节点"""
        # [FIX] 计入运行时开销（与 allocate() 保持一致），避免可行性判断偏乐观
        effective_layer_mb = float(layer_memory_mb) * self.RUNTIME_OVERHEAD_RATIO
        model_total_mb = total_layers * effective_layer_mb

        for node in sorted(alive_nodes, key=lambda n: n["free_mb"], reverse=True):
            if node["free_mb"] >= model_total_mb:
                ratio = model_total_mb / node["memory_mb"] if node["memory_mb"] > 0 else 999
                if ratio <= self.WARNING_RATIO:  # 安全余量检查
                    logger.info(f"✅ [FaultRecovery] {model_id} → 单节点迁移到 {node['node_id']}")
                    return {
                        "model_id": model_id,
                        "shard_on_failed": {"start_layer": 0, "end_layer": total_layers - 1},
                        "recovery_action": "migrated",
                        "recovery_detail": f"完整迁移到节点 {node['node_id']} (单节点加载, 模型≈{model_total_mb:.0f}MB)",
                        "new_allocation": [{
                            "node_id": node["node_id"],
                            "start_layer": 0,
                            "end_layer": total_layers - 1,
                            "layers_count": total_layers,
                            "address": node["address"],
                            "_is_migration": True
                        }]
                    }

        return None

    def _try_reshare_shard(
        self,
        model_info: Dict,
        failed_node_id: str,
        alive_nodes: List[Dict],
        layer_memory_mb: float = 100.0
    ) -> Optional[Dict]:
        """
        策略2: 用多个存活节点重新分担丢失的分片部分

        只重新分配掉线节点上的那部分层，其他节点上的分片保持不变
        """
        model_id = model_info["model_id"]
        lost_start = model_info["shard"]["start_layer"]
        lost_end = model_info["shard"]["end_layer"]
        lost_count = lost_end - lost_start + 1
        # [FIX] 与 allocate() 保持一致：乘以 RUNTIME_OVERHEAD_RATIO，避免判断偏乐观
        effective_layer_mb = float(layer_memory_mb) * self.RUNTIME_OVERHEAD_RATIO

        # 计算需要多少空间
        needed_mb = lost_count * effective_layer_mb

        # 找出有足够空间的存活节点组合
        candidates = [
            n for n in alive_nodes
            if n["free_mb"] >= effective_layer_mb  # 至少能放一层
        ]

        if not candidates or sum(n["free_mb"] for n in candidates) < needed_mb:
            return None

        # 按剩余空间排序，贪心分配
        candidates.sort(key=lambda n: n["free_mb"], reverse=True)

        new_allocations = []
        remaining_layers = lost_count
        current_layer = lost_start

        for node in candidates:
            if remaining_layers <= 0:
                break

            max_fit = int(node["free_mb"] / effective_layer_mb)
            layers_here = min(max_fit, remaining_layers)

            if layers_here <= 0:
                continue

            end_layer = current_layer + layers_here - 1
            new_allocations.append({
                "node_id": node["node_id"],
                "start_layer": current_layer,
                "end_layer": end_layer,
                "layers_count": layers_here,
                "address": node["address"],
                "_is_migration": True
            })
            current_layer = end_layer + 1
            remaining_layers -= layers_here

        if remaining_layers > 0:
            # 存活节点不够放下所有丢失的层
            logger.warning(f"⚠️ [FaultRecovery] {model_id}: 存活节点只能恢复部分分片 "
                          f"(还差 {remaining_layers} 层)")
            return None

        alloc_summary = ", ".join(
            f"{a['node_id']}(L{a['start_layer']}-{a['end_layer']})"
            for a in new_allocations
        )

        logger.info(f"✅ [FaultRecovery] {model_id} → 分片重组到: {alloc_summary}")

        return {
            "model_id": model_id,
            "shard_on_failed": model_info["shard"],
            "recovery_action": "migrated",
            "recovery_detail": f"丢失的分片已重新分配到 {len(new_allocations)} 个存活节点",
            "new_allocation": new_allocations
        }

    def _check_degraded_mode(
        self,
        model_id: str,
        failed_node_id: str
    ) -> Optional[Dict]:
        """
        策略3: 降级运行模式

        如果模型的其他分片还在其他存活节点上，
        标记为降级状态（推理质量/功能受限）
        """
        # 检查其他节点上是否还有该模型的分片
        other_shards = []

        for nid, node_info in self.manager.nodes.items():
            if nid == failed_node_id:
                continue
            for model in node_info.loaded_models:
                mid = model.get("model_id", "")
                # 匹配基础模型ID（考虑多实例场景）
                base_mid = mid.split("::")[0] if "::" in mid else mid
                base_model_id = model_id.split("::")[0] if "::" in model_id else model_id

                if base_mid == base_model_id:
                    shard = model.get("shard", {})
                    other_shards.append({
                        "node_id": nid,
                        "shard": shard
                    })

        if other_shards:
            shard_info = ", ".join(
                f"{s['node_id']}(L{s['shard'].get('start_layer', '?')}-L{s['shard'].get('end_layer', '?')})"
                for s in other_shards
            )

            logger.warning(f"⚠️ [FaultRecovery] {model_id} 进入降级模式: "
                          f"仅剩 {len(other_shards)} 个分片在线 [{shard_info}]")

            return {
                "model_id": model_id,
                "shard_on_failed": {"start_layer": 0, "end_layer": 0},
                "recovery_action": "degraded",
                "recovery_detail": (
                    f"部分分片丢失，剩余 {len(other_shards)} 个分片可提供降级服务。"
                    f"建议尽快恢复节点或手动卸载后重新加载。"
                ),
                "new_allocation": None,
                "remaining_shards": other_shards
            }

        return None

    def _build_loss_report(self, failed_node_id: str, affected: List[Dict]) -> Dict:
        """构建完全丢失的报告"""
        results = []
        for m in affected:
            results.append({
                "model_id": m["model_id"],
                "shard_on_failed": m["shard"],
                "recovery_action": "lost",
                "recovery_detail": "无可用存活节点，模型完全丢失",
                "new_allocation": None
            })

        return {
            "failed_node": failed_node_id,
            "affected_models": results,
            "summary": {"total": len(affected), "migrated": 0, "degraded": 0, "lost": len(affected)}
        }


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
    
    def _build_simulated_nodes(self, simulated_nodes: List[Dict]) -> List[Dict]:
        """
        根据用户输入构造模拟节点列表（用于预览分配）

        Args:
            simulated_nodes: 每个元素包含 memory_gb（节点显存，单位 GB）

        Returns:
            符合 SmartAllocator 要求的节点字典列表
        """
        nodes = []
        for idx, spec in enumerate(simulated_nodes):
            memory_gb = float(spec.get("memory_gb", 16))
            memory_mb = memory_gb * 1024
            nodes.append({
                "node_id": f"sim-node-{idx + 1}",
                "address": f"sim-{idx + 1}",
                "memory_mb": memory_mb,
                "used_mb": 0,
                "free_mb": memory_mb,
                "loaded_models": 0,
                "device": "Simulated",
                "flops": {}
            })
        return nodes

    async def preview_allocation(
        self,
        model_id: str,
        total_layers: int,
        layer_memory_mb: float = 100.0,
        strategy: str = "smart",
        simulated_nodes: Optional[List[Dict]] = None,
        ignore_used_memory: bool = False
    ) -> ModelAllocation:
        """
        预览模型分片分配方案

        Args:
            model_id: 模型标识符
            total_layers: 总层数
            layer_memory_mb: 每层预估显存占用 (MB)
            strategy: 分配策略:
                - 'smart':          智能策略（默认）：单节点优先 + 安全余量检查
                - 'memory_weighted': 按显存加权（传统，始终拆分）
                - 'uniform':        均匀分配（传统，始终拆分）
                - 'performance_weighted': 按性能加权（传统，始终拆分）
            simulated_nodes: 可选的模拟节点列表，用于离线场景下模拟分配；
                             提供时优先使用模拟节点，否则使用当前在线节点
            ignore_used_memory: 忽略节点当前已使用显存，直接按总显存视作可用。
                                用于 rebalance 场景（卸载旧分片后重新分配，
                                Manager 端 used_mb 可能尚未清零）。

        Returns:
            分配方案对象 (ModelAllocation)，包含 allocation_type / decision_reason 等元信息
        """
        # 使用智能分配器
        allocator = SmartAllocator(self.manager)

        nodes = None
        is_simulation = False
        if simulated_nodes:
            nodes = self._build_simulated_nodes(simulated_nodes)
            is_simulation = True

        if strategy == "smart":
            # 智能策略：自动判断单节点 or 拆分
            result = allocator.allocate(
                model_id=model_id,
                total_layers=total_layers,
                layer_memory_mb=layer_memory_mb,
                strategy="smart",
                nodes=nodes,
                ignore_used_memory=ignore_used_memory
            )
        else:
            # 传统策略：用户明确选择时走原有逻辑（但通过 allocator 统一处理）
            result = allocator.allocate(
                model_id=model_id,
                total_layers=total_layers,
                layer_memory_mb=layer_memory_mb,
                strategy=strategy,
                force_shard=True,   # 非 smart 策略强制拆分，保持向后兼容
                nodes=nodes,
                ignore_used_memory=ignore_used_memory
            )

        if is_simulation:
            result.decision_reason = "[模拟节点] " + result.decision_reason

        alloc_type_label = {
            "single_node": "单节点完整加载",
            "sharded": "跨节点拆分",
            "rebalanced": "重平衡后分配"
        }.get(result.allocation_type, result.allocation_type)

        logger.info(f"📊 预览分配方案 - {model_id}: "
                   f"类型={alloc_type_label}, "
                   f"节点数={len(result.allocations)}, "
                   f"原因={result.decision_reason}")

        return result
    
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
        
        # 使用类似内存加权的分配逻辑，但保证严格闭合（同 SmartAllocator._legacy_allocate_by_performance 的修复）
        raw_scores = [max(n["score"], 0.01) for n in scored_nodes]
        total_score = sum(raw_scores)
        
        allocations = []
        current_layer = 0
        N = len(scored_nodes)
        
        for idx, node in enumerate(scored_nodes):
            if current_layer >= total_layers:
                break
            
            weight = raw_scores[idx] / total_score
            layers_for_node = max(1, int(total_layers * weight))
            is_last = idx == N - 1
            if not is_last:
                remain_after = N - idx - 1
                upper = max(1, total_layers - current_layer - remain_after)
                layers_for_node = max(1, min(layers_for_node, upper))
            else:
                layers_for_node = total_layers - current_layer
            layers_for_node = max(1, min(layers_for_node, total_layers - current_layer))
            
            end_layer = current_layer + layers_for_node - 1
            
            allocations.append({
                "node_id": node["node_id"],
                "start_layer": current_layer,
                "end_layer": end_layer,
                "layers_count": layers_for_node,
                "address": node["address"]
            })
            
            current_layer = end_layer + 1
        
        # 末尾兜底
        if current_layer < total_layers and allocations:
            last_alloc = allocations[-1]
            remaining = total_layers - current_layer
            last_alloc["end_layer"] += remaining
            last_alloc["layers_count"] += remaining

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


__all__ = ['GPUPoolIntegration', 'ModelAllocation', 'SmartAllocator']
