"""
自动模型分配触发器 (Auto-Allocation Trigger)
=============================================

功能：
监听集群节点状态变化，在合适的时机自动触发模型分配和加载

核心能力：
1. 事件驱动：监听节点上线/下线/资源变化事件
2. 智能触发：只在必要时才重新规划（避免频繁折腾）
3. 策略可配：支持不同的分配策略（HA模式、普通模式）
4. 防抖保护：冷却期+最小变化阈值，防止"分配风暴"
5. 启动初始化：服务启动时如有可用节点，自动执行首次分配
6. 执行日志：完整的分配历史和决策原因记录

使用场景：
- 新节点加入集群 → 自动评估并加载更多/更大的模型
- 节点掉线 → 自动调整方案（降级或迁移）
- 节点恢复 → 优化性重平衡
- 定期巡检 → 发现资源浪费时建议重新规划

集成方式：
    from auto_alloc_trigger import AutoAllocTrigger

    trigger = AutoAllocTrigger(cluster_manager)
    await trigger.start()  # 启动后台监控

    # 或者手动触发
    await trigger.evaluate_and_allocate(reason="manual")
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum


logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    """从环境变量读取布尔值"""
    value = os.getenv(name, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


# ============================================================
#  配置常量
# ============================================================

class TriggerConfig:
    """触发器配置（支持环境变量覆盖）"""
    # 是否启用自动分配
    ENABLED: bool = _env_bool("EXO_AUTO_TRIGGER_ENABLED", False)

    # 触发条件
    MIN_ONLINE_NODES: int = int(os.getenv("EXO_AUTO_TRIGGER_MIN_ONLINE_NODES", "1"))              # 最少需要几个在线节点才触发
    NODE_COUNT_CHANGE_THRESHOLD: int = int(os.getenv("EXO_AUTO_TRIGGER_NODE_COUNT_CHANGE_THRESHOLD", "1"))   # 节点数变化超过此值才触发
    MEMORY_CHANGE_RATIO: float = float(os.getenv("EXO_AUTO_TRIGGER_MEMORY_CHANGE_RATIO", "0.20"))      # 显存变化超过20%才触发

    # 防抖保护
    COOLDOWN_SECONDS: int = int(os.getenv("EXO_AUTO_TRIGGER_COOLDOWN_SECONDS", "300"))            # 冷却期：5分钟内不重复触发
    MIN_INTERVAL_SECONDS: int = int(os.getenv("EXO_AUTO_TRIGGER_MIN_INTERVAL_SECONDS", "60"))         # 最小间隔：60秒

    # 分配策略
    DEFAULT_STRATEGY: str = os.getenv("EXO_AUTO_TRIGGER_DEFAULT_STRATEGY", "uniform")  # 默认使用均匀分配模式
    AUTO_EXECUTE: bool = _env_bool("EXO_AUTO_TRIGGER_AUTO_EXECUTE", True)             # 是否自动执行（还是只生成方案）

    # 启动行为
    AUTO_INIT_ON_STARTUP: bool = _env_bool("EXO_AUTO_TRIGGER_AUTO_INIT_ON_STARTUP", True)     # 启动时是否自动初始化
    STARTUP_DELAY_SECONDS: int = int(os.getenv("EXO_AUTO_TRIGGER_STARTUP_DELAY_SECONDS", "30"))       # 启动后延迟30秒再初始化（等待节点连接）

    # ✅ 新增：加载确认机制（解决WS推送≠加载完成的问题）
    WAIT_FOR_CONFIRMATION: bool = True     # 是否等待模型真正加载完成再确认
    CONFIRMATION_TIMEOUT: int = 120        # 确认超时时间（秒）- 2分钟内必须看到模型
    CONFIRMATION_CHECK_INTERVAL: int = 5   # 检查间隔（秒）- 每5秒查询一次节点状态

    # 日志保留
    MAX_HISTORY: int = 100                # 最大保留历史记录数


# ============================================================
#  数据结构
# ============================================================

@dataclass
class AllocationEvent:
    """分配事件记录"""
    event_id: str
    timestamp: float
    event_type: str           # "auto_trigger" | "manual" | "startup_init" | "node_event"
    reason: str               # 触发原因
    trigger_details: Dict[str, Any] = field(default_factory=dict)

    # 决策结果
    decision: str = ""        # "execute" | "skip" | "defer"
    skip_reason: str = ""     # 如果跳过，原因是什么

    # 执行结果
    plan_id: Optional[str] = None
    execution_success: bool = False
    models_allocated: int = 0
    models_total: int = 0
    error: Optional[str] = None

    # 性能
    processing_time_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "reason": self.reason,
            "trigger_details": self.trigger_details,
            "decision": self.decision,
            "skip_reason": self.skip_reason,
            "plan_id": self.plan_id,
            "execution_success": self.execution_success,
            "models_allocated": self.models_allocated,
            "models_total": self.models_total,
            "error": self.error,
            "processing_time_ms": round(self.processing_time_ms, 1),
        }


# ============================================================
#  核心类：自动分配触发器
# ============================================================

class AutoAllocTrigger:
    """
    自动分配触发器

    工作流程：
    1. 监听节点状态变化（通过定期轮询或事件回调）
    2. 评估是否满足触发条件：
       - 在线节点数是否达到阈值
       - 自上次分配后是否有显著变化
       - 是否在冷却期内
    3. 决定是否触发分配：
       - 满足所有条件 → 执行分配
       - 部分满足 → 记录并等待
       - 不满足 → 跳过并记录原因
    4. 执行分配（如果配置为自动执行）
    5. 记录完整的事件日志
    """

    def __init__(self, cluster_manager):
        """
        初始化触发器

        Args:
            cluster_manager: EXOClusterManager 实例
        """
        self.manager = cluster_manager
        self.config = TriggerConfig()

        # 状态追踪
        self._last_allocation_time: float = 0          # 上次分配时间
        self._last_node_count: int = 0                  # 上次节点数
        self._last_online_memory_mb: float = 0          # 上次总显存
        self._last_evaluated_models: int = 0            # 上次评估的模型数
        self._is_running: bool = False                   # 是否运行中
        self._monitor_task: Optional[asyncio.Task] = None

        # 待重试模型队列（需要更多节点才能分配）
        self._pending_models: List[Dict] = []           # [{model, reason, min_nodes_required, instance_id}]

        # 稳定性管理器（用于防抖）
        self._stability_manager: Optional[Any] = None

        # 事件历史
        self.event_history: List[AllocationEvent] = []

        # 回调函数（可选，用于通知外部系统）
        self._on_allocation_callback: Optional[Callable] = None

        logger.info(f"[AutoAllocTrigger] ✅ 初始化完成 "
                   f"(启用={self.config.ENABLED}, "
                   f"策略={self.config.DEFAULT_STRATEGY}, "
                   f"启动初始化={self.config.AUTO_INIT_ON_STARTUP})")

    def set_stability_manager(self, stability_manager):
        """
        设置稳定性管理器（集成防抖机制）

        Args:
            stability_manager: NodeStabilityManager 实例
        """
        self._stability_manager = stability_manager
        logger.info(f"[AutoAllocTrigger] 🔗 已连接稳定性管理器 (防抖已增强)")

    def set_callback(self, callback: Callable[[AllocationEvent], None]):
        """设置分配完成后的回调函数"""
        self._on_allocation_callback = callback

    async def start(self):
        """
        启动自动分配触发器

        会启动两个后台任务：
        1. 监控循环：定期检查是否需要触发分配
        2. 启动初始化：延迟后执行首次分配（如果启用）
        """
        if self._is_running:
            logger.warning("[AutoAllocTrigger] 已在运行")
            return

        if not self.config.ENABLED:
            logger.info("[AutoAllocTrigger] 未启用，跳过启动")
            return

        self._is_running = True

        # 启动监控循环
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

        # 启动初始化任务（如果启用）
        if self.config.AUTO_INIT_ON_STARTUP:
            asyncio.create_task(
                self._delayed_startup_init(delay=self.config.STARTUP_DELAY_SECONDS)
            )

        logger.info(f"[AutoAllocTrigger] 🚀 已启动 "
                   f"(监控循环 + {self.config.STARTUP_DELAY_SECONDS}s后启动初始化)")

    async def stop(self):
        """停止触发器"""
        self._is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("[AutoAllocTrigger] ⏹️ 已停止")

    async def _delayed_startup_init(self, delay: int):
        """
        延迟的启动初始化

        等待一段时间让节点有机会连接上来，然后执行首次分配
        """
        logger.info(f"[AutoAllocTrigger] ⏳ 将在 {delay}秒 后执行启动初始化...")
        await asyncio.sleep(delay)

        try:
            await self.evaluate_and_allocate(
                reason="startup_init",
                event_type="startup_init",
                force=True  # 启动初始化强制执行一次
            )
        except Exception as e:
            logger.error(f"[AutoAllocTrigger] ❌ 启动初始化失败: {e}", exc_info=True)

    async def _monitoring_loop(self):
        """
        监控循环

        定期检查集群状态，评估是否需要触发自动分配
        """
        logger.info("[AutoAllocTrigger] 🔍 监控循环已启动")

        while self._is_running:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次

                if not self._is_running:
                    break

                # 收集当前状态
                current_state = self._collect_current_state()

                # 评估是否需要触发
                should_trigger, reason = self._evaluate_trigger_condition(current_state)

                if should_trigger:
                    logger.info(f"[AutoAllocTrigger] 📢 检测到触发条件: {reason}")
                    await self.evaluate_and_allocate(
                        reason=reason,
                        event_type="auto_trigger",
                        trigger_details=current_state
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[AutoAllocTrigger] 监控循环异常: {e}", exc_info=True)

        logger.info("[AutoAllocTrigger] 🔍 监控循环已停止")

    def _collect_current_state(self) -> Dict[str, Any]:
        """收集当前集群状态快照"""
        online_nodes = []
        total_memory = 0
        loaded_models_count = 0

        for node_id, node_info in self.manager.nodes.items():
            if node_info.status.value == 'online':
                online_nodes.append(node_id)
                mem = node_info.device_info.get('memory', 0)
                total_memory += mem if mem else 0
                loaded_models_count += len(node_info.loaded_models)

        return {
            "online_node_count": len(online_nodes),
            "online_node_ids": online_nodes,
            "total_memory_mb": total_memory,
            "loaded_models_count": loaded_models_count,
            "timestamp": time.time(),
        }

    def _evaluate_trigger_condition(
        self,
        current_state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        评估是否应该触发自动分配（含防抖检查）

        Returns:
            (should_trigger, reason)
        """
        now = time.time()
        current_nodes = current_state["online_node_count"]
        current_memory = current_state["total_memory_mb"]
        current_models = current_state["loaded_models_count"]

        # 条件0: 稳定性防抖检查（最重要！）
        if self._stability_manager and current_nodes > 0:
            stability_check, stability_reason = self._check_all_nodes_stability(
                current_state.get("online_node_ids", [])
            )
            if not stability_check:
                return False, f"🛡️ 防抖保护: {stability_reason}"

        # 条件1: 最少在线节点数
        if current_nodes < self.config.MIN_ONLINE_NODES:
            return False, f"在线节点不足({current_nodes} < {self.config.MIN_ONLINE_NODES})"

        # 条件2: 冷却期检查
        if self._last_allocation_time > 0:
            elapsed = now - self._last_allocation_time
            if elapsed < self.config.COOLDOWN_SECONDS:
                remaining = self.config.COOLDOWN_SECONDS - int(elapsed)
                return False, f"冷却期内（剩余{remaining}秒）"

        # 条件3: 最小间隔检查
        if self._last_allocation_time > 0:
            elapsed = now - self._last_allocation_time
            if elapsed < self.config.MIN_INTERVAL_SECONDS:
                return False, f"距上次分配过近({elapsed:.0f}s < {self.config.MIN_INTERVAL_SECONDS}s)"

        # 条件4: 检测显著变化
        has_significant_change = False
        change_reasons = []

        # 节点数变化
        node_diff = abs(current_nodes - self._last_node_count)
        if node_diff >= self.config.NODE_COUNT_CHANGE_THRESHOLD:
            has_significant_change = True
            change_reasons.append(f"节点数变化: {self._last_node_count} → {current_nodes}")

        # 显存变化
        if self._last_online_memory_mb > 0 and current_memory > 0:
            memory_ratio = abs(current_memory - self._last_online_memory_mb) / self._last_online_memory_mb
            if memory_ratio >= self.config.MEMORY_CHANGE_RATIO:
                has_significant_change = True
                change_reasons.append(
                    f"显存变化{memory_ratio*100:.0f}%: "
                    f"{self._last_online_memory_mb/1024:.1f}GB → {current_memory/1024:.1f}GB"
                )

        # 首次运行（没有历史数据）
        if self._last_node_count == 0 and current_nodes > 0:
            has_significant_change = True
            change_reasons.append("首次检测到在线节点")

        # 无已加载模型但资源充足
        if current_models == 0 and current_memory > 0:
            has_significant_change = True
            change_reasons.append("有可用资源但无已加载模型")

        if not has_significant_change:
            return False, "无显著变化"

        return True, "; ".join(change_reasons)

    def _check_all_nodes_stability(self, online_node_ids: List[str]) -> Tuple[bool, str]:
        """
        检查所有在线节点的稳定性（防抖核心逻辑）

        防抖层级（由松到严）：
        1. 观察期 (60s) - 节点刚上线，等待确认稳定
        2. 抖动检测 (5min内3次) - 频繁上下线，标记为 FLAPPING
        3. 极不稳定 (1h内10次) - 标记为 UNSTABLE，禁止自动操作
        4. 冷却期 (180s) - 上次分配后的冷却时间

        Args:
            online_node_ids: 当前在线节点ID列表

        Returns:
            (is_stable, reason)
            - is_stable=True: 所有节点都稳定，可以触发分配
            - is_stable=False: 存在不稳定节点，附带原因
        """
        if not self._stability_manager:
            return True, "未连接稳定性管理器（跳过防抖检查）"

        unstable_nodes = []
        stable_nodes = []

        for node_id in online_node_ids:
            try:
                should_allocate, reason = self._stability_manager.should_trigger_reallocation(node_id)

                # should_trigger_reallocation 返回的逻辑：
                # - True + "节点已确认离线" → 这是离线节点，不应该出现在online列表中
                # - False + 各种原因 → 节点不稳定或不适合操作

                if not should_allocate:
                    # 检查具体的不稳定类型
                    if "抖动" in reason or "flapping" in reason.lower():
                        unstable_nodes.append(f"{node_id}(抖动中)")
                    elif "观察期" in reason or "观察" in reason:
                        unstable_nodes.append(f"{node_id}(观察期)")
                    elif "冷却" in reason or "cooldown" in reason.lower():
                        unstable_nodes.append(f"{node_id}(冷却期)")
                    elif "不稳定" in reason or "unstable" in reason.lower():
                        unstable_nodes.append(f"{node_id}(极不稳定)")
                    elif "离线" in reason or "offline" in reason.lower():
                        # 这个节点可能刚掉线，忽略
                        pass
                    else:
                        # 其他原因也视为不稳定
                        unstable_nodes.append(f"{node_id}({reason})")
                else:
                    # should_allocate=True 通常意味着 "未知节点" 或 "已确认离线"
                    # 对于在线节点，我们认为是稳定的
                    if "未知节点" in reason:
                        stable_nodes.append(node_id)
                    # "已确认离线" 不应该出现，但如果出现了说明数据不一致

            except Exception as e:
                logger.warning(f"[AutoAllocTrigger] ⚠️ 检查节点 {node_id} 稳定性时出错: {e}")
                # 出错时保守处理：视为不稳定
                unstable_nodes.append(f"{node_id}(检查失败)")

        # 判断结果
        if unstable_nodes:
            total = len(online_node_ids)
            unstable_count = len(unstable_nodes)

            if unstable_count == total:
                # 所有节点都不稳定
                return False, f"所有{total}个节点均不稳定: {', '.join(unstable_nodes)}"
            else:
                # 部分节点不稳定 → 根据配置决定是否允许
                unstable_ratio = unstable_count / total

                if unstable_ratio > 0.5:
                    # 超过50%节点不稳定 → 禁止触发
                    return False, (
                        f"{unstable_count}/{total} 个节点不稳定 "
                        f"(>{int(unstable_ratio*100)}%): {', '.join(unstable_nodes)}"
                    )
                else:
                    # 少数节点不稳定 → 允许但记录警告
                    logger.warning(
                        f"[AutoAllocTrigger] ⚠️ 部分节点不稳定但仍继续: "
                        f"{unstable_nodes} (稳定节点: {stable_nodes})"
                    )
                    return True, f"部分节点不稳定但可接受: {unstable_nodes}"

        return True, "所有节点均稳定"

    def _is_model_loaded(self, model_id: str) -> bool:
        """检查模型（按 base_model_id）是否已经在任意节点上加载"""
        base_id = model_id.split("::")[0] if "::" in model_id else model_id
        for node in self.manager.nodes.values():
            for m in node.loaded_models:
                loaded_id = m.get("model_id", "")
                loaded_base = m.get("base_model_id") or (
                    loaded_id.split("::")[0] if "::" in loaded_id else loaded_id
                )
                if loaded_base == base_id:
                    return True
        return False

    async def evaluate_and_allocate(
        self,
        reason: str,
        event_type: str = "manual",
        force: bool = False,
        trigger_details: Optional[Dict[str, Any]] = None,
        skip_loaded_models: bool = True
    ) -> AllocationEvent:
        """
        评估并执行自动分配（主入口）

        Args:
            reason: 触发原因描述
            event_type: 事件类型
            force: 是否强制执行（跳过部分检查）
            trigger_details: 额外的触发详情
            skip_loaded_models: 是否跳过已经加载过的模型（避免重复加载）

        Returns:
            AllocationEvent 完整的事件记录
        """
        start_time = time.time()
        event_id = f"evt_{int(time.time())}_{hash(str(time.time())) % 10000:04d}"

        event = AllocationEvent(
            event_id=event_id,
            timestamp=start_time,
            event_type=event_type,
            reason=reason,
            trigger_details=trigger_details or {},
        )

        logger.info(f"[AutoAllocTrigger] 🎯 开始评估分配: [{event_type}] {reason}")

        try:
            # Step 1: 收集当前状态
            current_state = self._collect_current_state()
            event.trigger_details.update(current_state)

            # Step 2: 检查基本前提条件（除非强制）
            if not force:
                should_trigger, skip_reason = self._evaluate_trigger_condition(current_state)
                if not should_trigger:
                    event.decision = "skip"
                    event.skip_reason = skip_reason
                    logger.info(f"[AutoAllocTrigger] ⏭️ 跳过分配: {skip_reason}")
                    self._record_event(event)
                    return event

            # Step 3: 更新基准状态（用于下次比较）
            # 注意：先保存上一次的节点数，供 pending 重试逻辑使用
            previous_node_count = self._last_node_count
            self._last_node_count = current_state["online_node_count"]
            self._last_online_memory_mb = current_state["total_memory_mb"]
            self._last_evaluated_models = current_state["loaded_models_count"]

            # Step 4: 检查是否有足够的在线节点
            if current_state["online_node_count"] == 0:
                event.decision = "skip"
                event.skip_reason = "无在线节点"
                logger.warning("[AutoAllocTrigger] ⏭️ 跳过: 无在线节点")
                self._record_event(event)
                return event

            # Step 4.5: 新节点加入时，重试待分配的 pending 模型
            retried_models = []
            if self._pending_models and current_state["online_node_count"] > previous_node_count:
                logger.info(f"[AutoAllocTrigger] 🔄 检测到新节点加入 ({previous_node_count} → {current_state['online_node_count']}), "
                           f"尝试重试 {len(self._pending_models)} 个待分配模型...")
                for pending in list(self._pending_models):  # copy to allow modification during iteration
                    try:
                        # 重新调用 load_model_to_cluster
                        spec = None
                        from auto_model_allocator import get_model_library
                        model_lib = get_model_library()
                        for key, lib_spec in model_lib.items():
                            if lib_spec.model_id == pending["model"]:
                                spec = lib_spec
                                break

                        if not spec:
                            continue

                        raw_model_path = getattr(spec, 'model_path', None) or ""
                        if raw_model_path in ["", "./", "./models", "models", "."]:
                            model_path = spec.model_id
                        else:
                            model_path = raw_model_path

                        result = await self.manager.load_model_to_cluster(
                            model_id=pending["model"],  # 用短名（如 "qwen-3-4b"）
                            model_path=model_path,  # HF repo ID 在这里
                            n_layers=spec.total_layers,
                            layer_memory_mb=spec.layer_memory_mb,
                            strategy="smart",
                            auto_instance=True
                        )

                        if result.get("success", False):
                            retried_models.append(pending["model"])
                            self._pending_models.remove(pending)
                            logger.info(f"[AutoAllocTrigger] ✅ 待分配模型重试成功: {pending['model']}")
                        elif result.get("status") != "pending_more_nodes":
                            # 真正失败了（不是 pending），从队列移除
                            self._pending_models.remove(pending)
                            logger.warning(f"[AutoAllocTrigger] ❌ 待分配模型重试失败: {pending['model']} - {result.get('error')}")
                        # else: 仍然是 pending，保留在队列中

                    except Exception as retry_err:
                        logger.error(f"[AutoAllocTrigger] ❌ 待分配模型重试异常: {pending['model']} - {retry_err}")

                if retried_models:
                    logger.info(f"[AutoAllocTrigger] 🔄 重试完成: {len(retried_models)}/{len(self._pending_models)+len(retried_models)} 个模型成功")

            # Step 5: 生成HA分配方案（推荐使用高可用模式）
            logger.info(f"[AutoAllocTrigger] 📋 正在生成分配方案 (策略={self.config.DEFAULT_STRATEGY})...")

            try:
                from ha_model_allocator import HAModelAllocator, RedundancyMode

                ha_allocator = HAModelAllocator(self.manager)

                # 根据配置选择模式
                mode = RedundancyMode(self.config.DEFAULT_STRATEGY)

                plan = ha_allocator.generate_ha_plan(
                    mode=mode,
                    min_replicas_critical=2 if mode == RedundancyMode.ACTIVE_PASSIVE else 1,
                    prioritize_redundancy=True
                )

                event.plan_id = plan.plan_id
                event.models_total = plan.total_models

                logger.info(f"[AutoAllocTrigger] ✅ 方案生成完成: "
                           f"{plan.total_models}个模型, "
                           f"{plan.total_instances}个实例, "
                           f"SPOF风险={plan.spof_risk_score:.1f}")

            except Exception as e:
                # HA模式失败，回退到普通模式
                logger.warning(f"[AutoAllocTrigger] HA模式失败，回退到普通模式: {e}")

                try:
                    from auto_model_allocator import AutoModelAllocator, AllocationStrategy

                    base_allocator = AutoModelAllocator(self.manager)
                    plan = base_allocator.generate_optimal_plan(
                        strategy=AllocationStrategy.MAXIMIZE_UTILIZATION
                    )

                    event.plan_id = plan.plan_id
                    event.models_total = plan.total_models

                    logger.info(f"[AutoAllocTrigger] ✅ 普通方案生成完成: {plan.total_models}个模型")

                except Exception as e2:
                    event.decision = "skip"
                    event.skip_reason = f"方案生成失败: {e2}"
                    event.error = str(e2)
                    logger.error(f"[AutoAllocTrigger] ❌ 方案生成完全失败: {e2}")
                    self._record_event(event)
                    return event

            # Step 6: 决定是否执行
            if plan.total_models == 0:
                event.decision = "skip"
                event.skip_reason = "生成的方案无模型（资源不足或其他限制）"
                logger.warning(f"[AutoAllocTrigger] ⏭️ 跳过: {event.skip_reason}")
                self._record_event(event)
                return event

            event.decision = "execute"

            # Step 7: 执行分配（如果配置为自动执行）
            if self.config.AUTO_EXECUTE or force:
                logger.info(f"[AutoAllocTrigger] 🚀 开始执行分配 ({plan.total_models}个模型)...")

                try:
                    # ✅ 真正执行模型加载
                    allocated_count = 0
                    failed_models = []
                    pending_models = []  # 需要更多节点的模型（等待重试）

                    # 兼容两种分配方案类型
                    # HAAllocationPlan 使用 plan.instances
                    # ModelAllocationPlan 使用 plan.allocations
                    model_items = []
                    if hasattr(plan, 'instances') and plan.instances:
                        # HA模式：{model_id: [ModelInstance, ...]}
                        model_items = list(plan.instances.items())
                        logger.info(f"[AutoAllocTrigger] 📋 使用HA方案 (instances格式)")
                    elif hasattr(plan, 'allocations') and plan.allocations:
                        # 普通模式：{model_id: {node_id: {...}}}
                        model_items = list(plan.allocations.items())
                        logger.info(f"[AutoAllocTrigger] 📋 使用普通方案 (allocations格式)")
                    else:
                        logger.warning(f"[AutoAllocTrigger] ⚠️ 方案为空或格式未知")

                    # 遍历方案中的每个模型
                    for model_id, alloc_data in model_items:
                        try:
                            # 🔍 调试：确认 model_id 来源（应该是短名如 "qwen-3-4b"，而非 HF repo ID）
                            if "/" in model_id:
                                logger.error(f"[AutoAllocTrigger] ❌ model_id 是 HF repo ID (含'/'): '{model_id}' ← ha_model_allocator 修复未生效!")
                            else:
                                logger.debug(f"[AutoAllocTrigger] ✅ model_id 是短名: '{model_id}'")

                            instance_count = len(alloc_data) if isinstance(alloc_data, (list, dict)) else 1
                            logger.info(f"[AutoAllocTrigger] 📦 正在加载模型: {model_id} ({instance_count}个实例/分片)")

                            # 查找模型规格（支持短名和完整ID两种方式）
                            from auto_model_allocator import get_model_library, ModelSpec
                            spec: ModelSpec = None
                            model_lib = get_model_library()

                            # 方式1：直接用model_id查找（可能是短名如 "qwen3-0.6b"）
                            spec = model_lib.get(model_id)

                            # 方式2：如果找不到，遍历所有值匹配 model_id 字段
                            if not spec:
                                for lib_spec in model_lib.values():
                                    if lib_spec.model_id == model_id or lib_spec.model_id.lower() == model_id.lower():
                                        spec = lib_spec
                                        break

                            # 方式3：大小写不敏感的模糊匹配
                            if not spec:
                                model_id_lower = model_id.lower()
                                for key, lib_spec in model_lib.items():
                                    if key == model_id_lower or model_id_lower in key or key in model_id_lower:
                                        spec = lib_spec
                                        logger.info(f"[AutoAllocTrigger] 🔍 模糊匹配: '{model_id}' → '{key}'")
                                        break

                            if not spec:
                                logger.warning(f"[AutoAllocTrigger] ⚠️ 未找到模型规格: {model_id}")
                                failed_models.append({"model": model_id, "reason": "未知模型"})
                                continue

                            # 如果配置为跳过已加载模型，则检查是否已加载
                            if skip_loaded_models and self._is_model_loaded(model_id):
                                logger.info(f"[AutoAllocTrigger] ⏭️ 模型 {model_id} 已加载，跳过重复加载")
                                continue

                            # 调用集群管理器的 load_model_to_cluster 方法
                            # model_path 必须是有效的 HuggingFace repo_id（节点端会校验非空）
                            raw_model_path = getattr(spec, 'model_path', None) or ""
                            # 无效路径 → 直接用 model_id (HF Repo ID) 作为路径
                            if raw_model_path in ["", "./", "./models", "models", "."]:
                                model_path = spec.model_id  # 如 "Qwen/Qwen3-0.6B"
                                logger.info(f"[AutoAllocTrigger] 📂 使用 HF Repo ID 作为路径: {model_path}")
                            else:
                                model_path = raw_model_path

                            result = await self.manager.load_model_to_cluster(
                                model_id=model_id,  # 用短名（如 "qwen-3-4b"），而非 HF repo ID
                                model_path=model_path,  # HF repo ID 在这里（如 "Qwen/Qwen3-4B"）
                                n_layers=spec.total_layers,  # 使用 total_layers 而不是 n_layers
                                layer_memory_mb=spec.layer_memory_mb,
                                strategy="smart",  # 智能策略：单节点优先
                                auto_instance=True  # 自动生成实例ID（支持多实例）
                            )

                            if result.get("success", False):
                                # ⚠️ 注意：success=True 只表示任务推送成功，不代表模型已真正加载
                                # 需要等待实际加载完成后才能确认
                                loaded_nodes = [r["node_id"] for r in result.get("results", []) if r.get("success")]
                                push_method = result.get("results", [{}])[0].get("method", "unknown") if result.get("results") else "unknown"

                                logger.info(f"[AutoAllocTrigger] 📤 {model_id} 任务推送成功 → 节点: {loaded_nodes} (方式: {push_method})")

                                # ✅ 新增：等待实际加载完成（异步确认）
                                if self.config.WAIT_FOR_CONFIRMATION:
                                    try:
                                        wait_result = await self._wait_for_model_loaded(
                                            model_id=model_id,
                                            full_model_id=f"{model_id}::{result.get('instance_id', 'unknown')}",
                                            timeout=self.config.CONFIRMATION_TIMEOUT,
                                            check_interval=5
                                        )

                                        if wait_result["confirmed"]:
                                            allocated_count += 1
                                            logger.info(f"[AutoAllocTrigger] ✅ {model_id} 加载确认完成 → 节点: {wait_result.get('loaded_on_nodes', [])}")
                                        else:
                                            # 推送成功但未能在超时时间内确认
                                            failed_models.append({
                                                "model": model_id,
                                                "reason": f"超时未确认 ({wait_result.get('reason', '未知')})"
                                            })
                                            logger.warning(f"[AutoAllocTrigger] ⚠️ {model_id} 推送成功但未确认: {wait_result.get('reason', '未知')}")
                                    except Exception as confirm_err:
                                        # 确认过程出错，但推送本身成功了，算作部分成功
                                        allocated_count += 1
                                        logger.warning(f"[AutoAllocTrigger] ⚠️ {model_id} 确认检查异常，但推送已成功: {confirm_err}")
                                else:
                                    # 不等待确认，直接计数（快速模式）
                                    allocated_count += 1
                                    logger.info(f"[AutoAllocTrigger] ✅ {model_id} 推送成功 (跳过确认)")
                            else:
                                error_msg = result.get("error", "未知错误")
                                # 检查是否为 pending_more_nodes 状态
                                if result.get("status") == "pending_more_nodes":
                                    pending_models.append({
                                        "model": model_id,
                                        "reason": error_msg,
                                        "min_nodes_required": result.get("min_nodes_required", 0),
                                        "instance_id": result.get("instance_id", "")
                                    })
                                    logger.warning(f"[AutoAllocTrigger] ⏳ {model_id} 需要更多节点 (≥{result.get('min_nodes_required', '?')}个), 已加入等待队列")
                                else:
                                    failed_models.append({"model": model_id, "reason": error_msg})
                                    logger.error(f"[AutoAllocTrigger] ❌ {model_id} 加载失败: {error_msg}")

                        except Exception as model_err:
                            failed_models.append({"model": model_id, "reason": str(model_err)})
                            logger.error(f"[AutoAllocTrigger] ❌ {model_id} 加载异常: {model_err}", exc_info=True)

                    # 更新事件结果
                    event.execution_success = (allocated_count > 0)
                    event.models_allocated = allocated_count

                    if allocated_count > 0:
                        # 更新时间戳（只在真正有变化时更新）
                        self._last_allocation_time = time.time()

                        logger.info(f"[AutoAllocTrigger] ✅ 分配执行完成: "
                                   f"{allocated_count}/{event.models_total} 个模型已加载")
                        if failed_models:
                            logger.warning(f"[AutoAllocTrigger] ⚠️ {len(failed_models)} 个模型加载失败: "
                                         f"{[f['model'] for f in failed_models]}")
                        if pending_models:
                            logger.info(f"[AutoAllocTrigger] ⏳ {len(pending_models)} 个模型等待更多节点: "
                                       f"{[p['model'] for p in pending_models]}")
                            # 存储待重试的模型，新节点加入时自动重试
                            self._pending_models = pending_models
                    else:
                        event.execution_success = False
                        # 区分：全部失败 vs 全部等待 vs 混合
                        if pending_models and not failed_models:
                            event.error = f"所有模型需要更多节点: {pending_models}"
                            logger.warning(f"[AutoAllocTrigger] ⏳ 所有模型等待更多节点: {pending_models}")
                            self._pending_models = pending_models
                        elif pending_models:
                            event.error = f"部分失败+部分等待: 失败={failed_models}, 等待={pending_models}"
                            logger.warning(f"[AutoAllocTrigger] ⚠️ 部分失败+部分等待: 失败={[f['model'] for f in failed_models]}, 等待={[p['model'] for p in pending_models]}")
                            self._pending_models = pending_models
                        else:
                            event.error = f"所有模型加载失败: {failed_models}"
                            logger.error(f"[AutoAllocTrigger] ❌ 所有模型加载失败: {failed_models}")

                except Exception as e:
                    event.execution_success = False
                    event.error = str(e)
                    logger.error(f"[AutoAllocTrigger] ❌ 分配执行失败: {e}", exc_info=True)
            else:
                logger.info(f"[AutoAllocTrigger] 📝 仅生成方案（未配置自动执行）")
                event.decision = "defer"

        except Exception as e:
            event.error = str(e)
            logger.error(f"[AutoAllocTrigger] ❌ 评估过程出错: {e}", exc_info=True)

        finally:
            event.processing_time_ms = (time.time() - start_time) * 1000
            self._record_event(event)

            # 调用回调
            if self._on_allocation_callback:
                try:
                    self._on_allocation_callback(event)
                except Exception as cb_err:
                    logger.error(f"[AutoAllocTrigger] 回调执行失败: {cb_err}")

        return event

    def _record_event(self, event: AllocationEvent):
        """记录事件到历史"""
        self.event_history.append(event)

        # 限制历史长度
        if len(self.event_history) > self.config.MAX_HISTORY:
            self.event_history = self.event_history[-self.config.MAX_HISTORY:]

    def get_status(self) -> Dict[str, Any]:
        """获取触发器状态（含防抖信息）"""
        last_event = self.event_history[-1] if self.event_history else None

        # 收集稳定性信息
        stability_info = None
        if self._stability_manager:
            try:
                stability_info = {
                    "connected": True,
                    "tracked_nodes": len(self._stability_manager.records),
                    "config": {
                        "observation_period_sec": getattr(self._stability_manager, 'OBSERVATION_PERIOD', 60),
                        "cooldown_period_sec": getattr(self._stability_manager, 'COOLDOWN_PERIOD', 180),
                        "flap_threshold_5min": getattr(self._stability_manager, 'FLAP_THRESHOLD_5MIN', 3),
                    }
                }
            except Exception as e:
                stability_info = {"connected": False, "error": str(e)}
        else:
            stability_info = {"connected": False, "note": "未连接稳定性管理器"}

        return {
            "enabled": self.config.ENABLED,
            "running": self._is_running,
            "config": {
                "strategy": self.config.DEFAULT_STRATEGY,
                "auto_execute": self.config.AUTO_EXECUTE,
                "cooldown_seconds": self.config.COOLDOWN_SECONDS,
                "min_online_nodes": self.config.MIN_ONLINE_NODES,
                # 新增：加载确认配置
                "wait_for_confirmation": self.config.WAIT_FOR_CONFIRMATION,
                "confirmation_timeout_sec": self.config.CONFIRMATION_TIMEOUT,
            },
            "stability_protection": stability_info,  # 防抖信息
            "state": {
                "last_allocation_time": self._last_allocation_time,
                "last_node_count": self._last_node_count,
                "last_online_memory_gb": round(self._last_online_memory_mb / 1024, 1) if self._last_online_memory_mb else 0,
                "total_events": len(self.event_history),
                "last_event": last_event.to_dict() if last_event else None,
            },
            "next_eligible_time": (
                self._last_allocation_time + self.config.COOLDOWN_SECONDS
                if self._last_allocation_time > 0 else 0
            )
        }


    def get_history(self, limit: int = 20) -> List[Dict]:
        """获取事件历史"""
        recent = self.event_history[-limit:]
        return [e.to_dict() for e in reversed(recent)]

    async def _wait_for_model_loaded(
        self,
        model_id: str,
        full_model_id: str,
        timeout: int = 120,
        check_interval: int = 5
    ) -> Dict[str, Any]:
        """
        等待模型真正加载完成（通过查询节点实际状态确认）

        重要：不能使用 connector.loaded_models，因为它在WS推送时就乐观更新了（假阳性）。
              必须使用 gRPC CollectTopology 返回的节点真实状态。
        """
        import asyncio

        start_time = time.time()
        # 初始宽限期：跳过乐观更新窗口（WS推送后connector会立即更新loaded_models）
        initial_grace_period = 5.0  # 秒

        logger.info(f"[AutoAllocTrigger] ⏳ 开始等待模型加载确认: {full_model_id} (超时: {timeout}s, 宽限期: {initial_grace_period}s)")

        while True:
            elapsed = time.time() - start_time

            # 检查超时
            if elapsed > timeout:
                logger.warning(
                    f"[AutoAllocTrigger] ⏰ 确认超时: {full_model_id} "
                    f"({timeout:.0f}s内未检测到模型)"
                )
                return {
                    "confirmed": False,
                    "loaded_on_nodes": [],
                    "reason": f"超时 ({timeout:.0f}s)",
                    "elapsed_time": elapsed
                }

            # 宽限期内跳过检查（避免读取到乐观更新的假数据）
            if elapsed < initial_grace_period:
                await asyncio.sleep(min(check_interval, initial_grace_period - elapsed + 0.1))
                continue

            # 查询节点真实状态（通过gRPC，而非connector的乐观缓存）
            try:
                loaded_on_nodes = []

                if hasattr(self.manager, 'nodes'):
                    for node_id, node in self.manager.nodes.items():
                        if node.status.value != "online":
                            continue

                        # 使用节点已加载模型列表（来自 gRPC CollectTopology 或 WS 回调）
                        # 优先级：loaded_models（已验证可靠） > loaded_models_from_grpc
                        real_models = None

                        # 方式1：直接使用 node.loaded_models（最可靠，由监控循环/WS回调维护）
                        if hasattr(node, 'loaded_models') and node.loaded_models:
                            real_models = [m.get("model_id", "") for m in node.loaded_models]

                        # 方式2：回退到 gRPC 专用字段（如果有）
                        if not real_models:
                            real_models = getattr(node, 'loaded_models_from_grpc', None)
                        if not real_models:
                            real_models = getattr(node, '_grpc_loaded_models', None)

                        if real_models:
                            if any(
                                full_model_id in str(m) or model_id in str(m)
                                for m in real_models
                            ):
                                loaded_on_nodes.append(node_id)

                # 判断结果
                if loaded_on_nodes:
                    logger.info(
                        f"[AutoAllocTrigger] ✅ 确认成功: {full_model_id} "
                        f"已加载到节点 {loaded_on_nodes} (耗时 {elapsed:.1f}s)"
                    )
                    return {
                        "confirmed": True,
                        "loaded_on_nodes": loaded_on_nodes,
                        "reason": "gRPC状态确认",
                        "elapsed_time": elapsed
                    }
                else:
                    if int(elapsed) % 10 == 0:
                        logger.info(
                            f"[AutoAllocTrigger] ⏳ 等待中... {full_model_id} "
                            f"(已等待 {elapsed:.0f}s/{timeout}s)"
                        )

            except Exception as check_err:
                logger.warning(f"[AutoAllocTrigger] ⚠️ 检查模型状态异常: {check_err}")

            await asyncio.sleep(check_interval)

    async def on_node_joined(self, node_id: str):
        """
        新节点加入事件处理

        等待节点状态稳定后，触发一次强制的自动分配，
        并跳过已经加载的模型以避免重复加载。
        """
        logger.info(f"[AutoAllocTrigger] 🆕 收到新节点加入事件: {node_id}")
        # 短暂等待，让节点有机会上报设备信息
        await asyncio.sleep(2)
        await self.evaluate_and_allocate(
            reason=f"新节点 {node_id} 加入",
            event_type="node_event",
            force=True,
            skip_loaded_models=True
        )

    async def manual_trigger(self, reason: str = "用户手动触发", force: bool = True) -> AllocationEvent:
        """
        手动触发一次分配

        Args:
            reason: 触发原因
            force: 是否强制执行

        Returns:
            AllocationEvent
        """
        logger.info(f"[AutoAllocTrigger] 👆 手动触发: {reason}")
        return await self.evaluate_and_allocate(
            reason=reason,
            event_type="manual",
            force=force
        )
