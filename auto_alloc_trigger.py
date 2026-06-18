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
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum


logger = logging.getLogger(__name__)


# ============================================================
#  配置常量
# ============================================================

class TriggerConfig:
    """触发器配置"""
    # 是否启用自动分配
    ENABLED: bool = True

    # 触发条件
    MIN_ONLINE_NODES: int = 1              # 最少需要几个在线节点才触发
    NODE_COUNT_CHANGE_THRESHOLD: int = 1   # 节点数变化超过此值才触发
    MEMORY_CHANGE_RATIO: float = 0.20      # 显存变化超过20%才触发

    # 防抖保护
    COOLDOWN_SECONDS: int = 300            # 冷却期：5分钟内不重复触发
    MIN_INTERVAL_SECONDS: int = 60         # 最小间隔：60秒

    # 分配策略
    DEFAULT_STRATEGY: str = "active_passive"  # 默认使用HA模式
    AUTO_EXECUTE: bool = True             # 是否自动执行（还是只生成方案）

    # 启动行为
    AUTO_INIT_ON_STARTUP: bool = True     # 启动时是否自动初始化
    STARTUP_DELAY_SECONDS: int = 30       # 启动后延迟30秒再初始化（等待节点连接）

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

        # 事件历史
        self.event_history: List[AllocationEvent] = []

        # 回调函数（可选，用于通知外部系统）
        self._on_allocation_callback: Optional[Callable] = None

        logger.info(f"[AutoAllocTrigger] ✅ 初始化完成 "
                   f"(启用={self.config.ENABLED}, "
                   f"策略={self.config.DEFAULT_STRATEGY}, "
                   f"启动初始化={self.config.AUTO_INIT_ON_STARTUP})")

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
        评估是否应该触发自动分配

        Returns:
            (should_trigger, reason)
        """
        now = time.time()
        current_nodes = current_state["online_node_count"]
        current_memory = current_state["total_memory_mb"]
        current_models = current_state["loaded_models_count"]

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

    async def evaluate_and_allocate(
        self,
        reason: str,
        event_type: str = "manual",
        force: bool = False,
        trigger_details: Optional[Dict[str, Any]] = None
    ) -> AllocationEvent:
        """
        评估并执行自动分配（主入口）

        Args:
            reason: 触发原因描述
            event_type: 事件类型
            force: 是否强制执行（跳过部分检查）
            trigger_details: 额外的触发详情

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
                    # TODO: 这里需要实际调用GPU池接口来加载模型
                    # 目前是模拟实现，实际需要对接 gpu_pool_integration

                    # 模拟执行过程
                    await asyncio.sleep(0.5)

                    event.execution_success = True
                    event.models_allocated = plan.total_models  # 假设全部成功

                    # 更新时间戳
                    self._last_allocation_time = time.time()

                    logger.info(f"[AutoAllocTrigger] ✅ 分配执行成功: "
                               f"{event.models_allocated}/{event.models_total} 个模型已加载")

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
        """获取触发器状态"""
        last_event = self.event_history[-1] if self.event_history else None

        return {
            "enabled": self.config.ENABLED,
            "running": self._is_running,
            "config": {
                "strategy": self.config.DEFAULT_STRATEGY,
                "auto_execute": self.config.AUTO_EXECUTE,
                "cooldown_seconds": self.config.COOLDOWN_SECONDS,
                "min_online_nodes": self.config.MIN_ONLINE_NODES,
            },
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
