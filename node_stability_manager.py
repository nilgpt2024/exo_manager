"""
节点稳定性管理与故障恢复模块
================================

解决节点频繁上下线场景下的核心问题：
1. 稳定性检测：区分临时抖动 vs 真正离线
2. 防抖机制：避免频繁重分配导致的"分配风暴"
3. 故障恢复：优雅降级 + 自动迁移 + 快速回迁
4. 版本管理：方案历史、回滚、对比
5. 监控告警：异常模式识别与通知

核心设计原则：
- 宁可保守，不可激进：宁可暂时降低服务能力，也不频繁折腾
- 渐进式响应：首次掉线→观察期→确认离线→有限迁移→最终降级
- 优先保护：高优先级模型（如已服务的活跃模型）优先保障

使用示例：
    from node_stability_manager import NodeStabilityManager, FaultRecoveryManager

    stability_mgr = NodeStabilityManager(allocator)
    recovery_mgr = FaultRecoveryManager(allocator, stability_mgr)

    # 节点状态变化时调用
    await stability_mgr.on_node_status_changed(node_id, "offline")

    # 检查是否需要触发重分配
    if stability_mgr.should_trigger_reallocation(node_id):
        plan = await recovery_mgr.recover_from_failure(node_id)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from collections import deque
import json
from datetime import datetime, timedelta

# 导入基础分配器中的类型定义
from auto_model_allocator import NodeResource

logger = logging.getLogger(__name__)


# ============================================================
#  节点状态枚举
# ============================================================

class NodeStabilityStatus(str, Enum):
    """节点稳定性状态"""
    STABLE = "stable"                    # 稳定在线
    FLAPPING = "flapping"                # 频繁抖动中（短时间内多次上下线）
    SUSPECTED_OFFLINE = "suspected_offline"  # 怀疑离线（等待确认）
    CONFIRMED_OFFLINE = "confirmed_offline"  # 已确认离线
    RECOVERING = "recovering"            # 恢复中（刚重新上线）
    UNSTABLE = "unstable"                # 不稳定（长期频繁抖动）


# ============================================================
#  节点稳定性追踪器
# ============================================================

@dataclass
class NodeStabilityRecord:
    """单个节点的稳定性记录"""
    node_id: str

    # 状态历史（最近N次变化）
    status_history: deque = field(default_factory=lambda: deque(maxlen=50))

    # 时间窗口统计
    online_count: int = 0          # 在线次数
    offline_count: int = 0         # 离线次数
    total_transitions: int = 0     # 总状态切换次数

    # 关键时间戳
    last_status_change: float = 0      # 最后一次状态变化时间
    last_online_time: float = 0        # 最后上线时间
    last_offline_time: float = 0       # 最后下线时间
    first_seen: float = 0              # 首次发现时间

    # 抖动检测
    flap_count_5min: int = 0           # 5分钟内抖动次数
    flap_count_1hour: int = 0          # 1小时内抖动次数
    flap_timestamps: List[float] = field(default_factory=list)  # 最近抖动时间戳

    # 当前判定状态
    stability_status: NodeStabilityStatus = NodeStabilityStatus.STABLE
    confidence_score: float = 1.0       # 稳定性置信度 (0-1)

    # 冷却/观察期
    in_cooldown: bool = False          # 是否处于冷却期
    cooldown_until: float = 0          # 冷却期结束时间
    observation_start: float = 0       # 观察期开始时间

    # 承载的关键模型（需要优先保护）
    critical_models: List[str] = field(default_factory=list)


class NodeStabilityManager:
    """
    节点稳定性管理器

    核心功能：
    1. 追踪每个节点的状态变化历史
    2. 检测异常模式（频繁抖动、周期性离线等）
    3. 判断节点的真实状态（过滤临时抖动）
    4. 提供防抖建议（是否应该触发重分配）

    配置参数：
    - FLAP_THRESHOLD_5MIN: 5分钟内超过此次数视为频繁抖动
    - OBSERVATION_PERIOD: 观察期时长（秒），在此期间不触发重分配
    - COOLDOWN_PERIOD: 冷却期时长（秒），重分配后的冷却时间
    - STABILITY_WINDOW: 计算稳定性的时间窗口（小时）
    """

    # ===== 配置参数 =====
    FLAP_THRESHOLD_5MIN = 3            # 5分钟内3次以上视为频繁抖动
    FLAP_THRESHOLD_1HOUR = 10          # 1小时内10次以上视为极不稳定
    OBSERVATION_PERIOD = 60            # 观察期：60秒内不触发重分配
    COOLDOWN_PERIOD = 180              # 冷却期：重分配后3分钟内不再触发
    STABILITY_WINDOW_HOURS = 2         # 稳定性计算窗口：2小时
    RECOVERY_GRACE_PERIOD = 120        # 恢复宽限期：重新上线后2分钟标记为recovering

    def __init__(self, allocator=None):
        self.allocator = allocator
        self.records: Dict[str, NodeStabilityRecord] = {}
        self._lock = asyncio.Lock()

        logger.info(f"[StabilityMgr] ✅ 初始化完成 "
                   f"(观察期={self.OBSERVATION_PERIOD}s, "
                   f"冷却期={self.COOLDOWN_PERIOD}s, "
                   f"抖动阈值={self.FLAP_THRESHOLD_5MIN}/5min)")

    async def on_node_status_changed(
        self,
        node_id: str,
        new_status: str,
        loaded_models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        处理节点状态变化事件

        Args:
            node_id: 节点ID
            new_status: 新状态 ("online" | "offline" | "error")
            loaded_models: 该节点当前加载的模型列表（在线时提供）

        Returns:
            处理结果和建议动作
        """
        async with self._lock:
            now = time.time()

            # 获取或创建记录
            if node_id not in self.records:
                self.records[node_id] = NodeStabilityRecord(node_id=node_id, first_seen=now)
                logger.info(f"[StabilityMgr] 🆔 新节点注册: {node_id}")

            record = self.records[node_id]
            old_status = record.stability_status

            # 更新历史
            record.status_history.append({
                "timestamp": now,
                "status": new_status,
                "models": loaded_models or []
            })
            record.last_status_change = now
            record.total_transitions += 1

            # 统计在线/离线次数
            if new_status == "online":
                record.online_count += 1
                record.last_online_time = now
                if loaded_models:
                    record.critical_models = loaded_models
            elif new_status in ("offline", "error"):
                record.offline_count += 1
                record.last_offline_time = now
                record.flap_timestamps.append(now)

            # 清理过期的抖动记录（保留最近1小时）
            cutoff = now - 3600
            record.flap_timestamps = [t for t in record.flap_timestamps if t > cutoff]

            # 计算近期抖动次数
            cutoff_5min = now - 300
            record.flap_count_5min = len([t for t in record.flap_timestamps if t > cutoff_5min])
            record.flap_count_1hour = len(record.flap_timestamps)

            # ===== 判定稳定性状态 =====
            new_stability = self._evaluate_stability(record, new_status)
            record.stability_status = new_stability
            record.confidence_score = self._calculate_confidence(record)

            # ===== 决定建议动作 =====
            action = self._determine_action(record, old_status, new_stability, new_status)

            result = {
                "node_id": node_id,
                "timestamp": now,
                "old_status": old_status.value if isinstance(old_status, NodeStabilityStatus) else str(old_status),
                "new_status": new_status,
                "stability_status": new_stability.value,
                "confidence": round(record.confidence_score, 2),
                "flap_count_5min": record.flap_count_5min,
                "flap_count_1hour": record.flap_count_1hour,
                "total_transitions": record.total_transitions,
                "recommended_action": action["action"],
                "action_reason": action["reason"],
                "should_reallocate": action.get("should_reallocate", False),
                "wait_seconds": action.get("wait_seconds", 0),
            }

            # 日志输出
            level = logging.WARNING if action["action"] == "TRIGGER_REALLOCATION" else logging.INFO
            logger.log(level,
                      f"[StabilityMgr] 节点 {node_id}: {new_status} → "
                      f"{new_stability.value} (置信度{record.confidence_score:.0%}, "
                      f"5min抖动{record.flap_count_5min}次) → "
                      f"{action['action']}: {action['reason']}")

            return result

    def _evaluate_stability(
        self,
        record: NodeStabilityRecord,
        current_raw_status: str
    ) -> NodeStabilityStatus:
        """评估节点的稳定性状态"""
        now = time.time()

        # 1. 检查是否在恢复宽限期内（刚重新上线）
        if current_raw_status == "online":
            if record.last_offline_time > 0:
                time_since_offline = now - record.last_offline_time
                if time_since_offline < self.RECOVERY_GRACE_PERIOD:
                    return NodeStabilityStatus.RECOVERING

        # 2. 检测频繁抖动
        if record.flap_count_5min >= self.FLAP_THRESHOLD_5MIN:
            if record.flap_count_1hour >= self.FLAP_THRESHOLD_1HOUR:
                return NodeStabilityStatus.UNSTABLE  # 极不稳定
            return NodeStabilityStatus.FLAPPING  # 正在抖动

        # 3. 基于当前原始状态判断
        if current_raw_status == "online":
            # 在线且稳定
            if record.in_cooldown and now < record.cooldown_until:
                return NodeStabilityStatus.RECOVERING  # 冷却期内
            return NodeStabilityStatus.STABLE

        elif current_raw_status in ("offline", "error"):
            # 离线状态判断
            time_since_change = now - record.last_status_change

            if time_since_change < self.OBSERVATION_PERIOD:
                # 还在观察期内
                return NodeStabilityStatus.SUSPECTED_OFFLINE
            else:
                # 超过观察期，确认离线
                return NodeStabilityStatus.CONFIRMED_OFFLINE

        return NodeStabilityStatus.UNSTABLE

    def _calculate_confidence(self, record: NodeStabilityRecord) -> float:
        """计算状态置信度 (0-1)"""
        base_confidence = 1.0

        # 抖动惩罚
        if record.flap_count_5min >= self.FLAP_THRESHOLD_5MIN:
            base_confidence -= 0.3
        if record.flap_count_1hour >= self.FLAP_THRESHOLD_1HOUR:
            base_confidence -= 0.3

        # 历史稳定性奖励
        uptime_ratio = record.online_count / max(record.total_transitions, 1)
        base_confidence += uptime_ratio * 0.2

        return max(0.0, min(1.0, base_confidence))

    def _determine_action(
        self,
        record: NodeStabilityRecord,
        old_stability: NodeStabilityStatus,
        new_stability: NodeStabilityStatus,
        raw_status: str
    ) -> Dict[str, Any]:
        """决定建议的动作"""
        now = time.time()

        # 场景1: 节点刚掉线 → 进入观察期
        if raw_status in ("offline", "error"):
            if new_stability == NodeStabilityStatus.SUSPECTED_OFFLINE:
                wait_time = self.OBSERVATION_PERIOD - (now - record.last_status_change)
                return {
                    "action": "OBSERVE",
                    "reason": f"节点疑似离线，进入观察期（剩余{wait_time:.0f}秒）",
                    "should_reallocate": False,
                    "wait_seconds": max(0, wait_time)
                }

            # 场景2: 确认离线 → 但该节点正在抖动
            if new_stability == NodeStabilityStatus.CONFIRMED_OFFLINE:
                if old_stability in (NodeStabilityStatus.FLAPPING, NodeStabilityStatus.UNSTABLE):
                    return {
                        "action": "LIMITED_MIGRATION",
                        "reason": "节点确认离线但历史不稳定，仅迁移关键模型",
                        "should_reallocate": True,
                        "priority": "low",
                        "migrate_only_critical": True
                    }

                # 正常离线 → 可以完整重分配
                return {
                    "action": "TRIGGER_REALLOCATION",
                    "reason": "节点已确认离线，触发故障恢复",
                    "should_reallocate": True,
                    "priority": "normal"
                }

        # 场景3: 节点重新上线 → 进入恢复期
        if raw_status == "online":
            if new_stability == NodeStabilityStatus.RECOVERING:
                # 设置冷却期
                record.in_cooldown = True
                record.cooldown_until = now + self.COOLDOWN_PERIOD

                return {
                    "action": "ENTER_COOLDOWN",
                    "reason": f"节点重新上线，进入冷却期({self.COOLDOWN_PERIOD}秒内不触发新分配)",
                    "should_reallocate": False,
                    "wait_seconds": self.COOLDOWN_PERIOD
                }

            if new_stability == NodeStabilityStatus.STABLE:
                # 完全恢复 → 可以考虑优化性重平衡（非紧急）
                return {
                    "action": "CONSIDER_REBALANCE",
                    "reason": "节点完全恢复，可考虑优化性重平衡",
                    "should_reallocate": False,
                    "priority": "optional"
                }

        # 场景4: 正在抖动 → 什么都不做，静默观察
        if new_stability in (NodeStabilityStatus.FLAPPING, NodeStabilityStatus.UNSTABLE):
            return {
                "action": "NO_ACTION",
                "reason": f"节点频繁抖动(5min内{record.flap_count_5min}次)，静默观察避免分配风暴",
                "should_reallocate": False,
                "wait_seconds": self.OBSERVATION_PERIOD * 2  # 双倍观察期
            }

        # 默认：无操作
        return {
            "action": "NO_ACTION",
            "reason": "状态变化无需立即处理",
            "should_reallocate": False
        }

    def should_trigger_reallocation(self, node_id: str) -> Tuple[bool, str]:
        """
        检查是否应该对指定节点触发重分配

        Returns:
            (should_trigger, reason)
        """
        record = self.records.get(node_id)
        if not record:
            return True, "未知节点，默认允许"

        now = time.time()

        # 检查冷却期
        if record.in_cooldown and now < record.cooldown_until:
            remaining = record.cooldown_until - now
            return False, f"冷却期内（剩余{remaining:.0f}秒）"

        # 检查稳定性状态
        if record.stability_status == NodeStabilityStatus.FLAPPING:
            return False, f"节点正在抖动（5min内{record.flap_count_5min}次）"

        if record.stability_status == NodeStabilityStatus.SUSPECTED_OFFLINE:
            remaining = self.OBSERVATION_PERIOD - (now - record.last_status_change)
            return False, f"观察期内（剩余{remaining:.0f}秒）"

        if record.stability_status == NodeStabilityStatus.CONFIRMED_OFFLINE:
            return True, "节点已确认离线"

        if record.stability_status == NodeStabilityStatus.UNSTABLE:
            return False, "节点极度不稳定，禁止自动重分配（需人工介入）"

        return False, "节点状态正常"

    def get_node_stability_report(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """获取节点稳定性报告"""
        if node_id:
            records = {node_id: self.records.get(node_id)}
        else:
            records = self.records

        report = {
            "timestamp": time.time(),
            "total_nodes_tracked": len(records),
            "nodes": {}
        }

        for nid, rec in records.items():
            if not rec:
                continue

            # 计算健康评分 (0-100)
            health_score = self._calculate_health_score(rec)

            report["nodes"][nid] = {
                "stability_status": rec.stability_status.value,
                "confidence": round(rec.confidence_score * 100, 1),
                "health_score": health_score,
                "total_transitions": rec.total_transitions,
                "flap_count_5min": rec.flap_count_5min,
                "flap_count_1hour": rec.flap_count_1hour,
                "uptime_ratio": round(rec.online_count / max(rec.total_transitions, 1) * 100, 1),
                "in_cooldown": rec.in_cooldown,
                "critical_models_count": len(rec.critical_models),
                "first_seen": rec.first_seen,
                "last_change": rec.last_status_change,
            }

        # 全局统计
        all_statuses = [r.stability_status for r in records.values() if r]
        report["summary"] = {
            "stable_nodes": sum(1 for s in all_statuses if s == NodeStabilityStatus.STABLE),
            "flapping_nodes": sum(1 for s in all_statuses if s == NodeStabilityStatus.FLAPPING),
            "offline_nodes": sum(1 for s in all_statuses if s == NodeStabilityStatus.CONFIRMED_OFFLINE),
            "unstable_nodes": sum(1 for s in all_statuses if s == NodeStabilityStatus.UNSTABLE),
            "avg_health_score": round(
                sum(self._calculate_health_score(r) for r in records.values() if r) /
                max(len([r for r in records.values() if r]), 1), 1
            )
        }

        return report

    def _calculate_health_score(self, record: NodeStabilityRecord) -> float:
        """计算节点健康评分 (0-100)"""
        score = 100.0

        # 抖动扣分
        score -= min(40, record.flap_count_5min * 10)
        score -= min(30, record.flap_count_1hour * 2)

        # 离线状态扣分
        if record.stability_status == NodeStabilityStatus.CONFIRMED_OFFLINE:
            score -= 30
        elif record.stability_status == NodeStabilityStatus.SUSPECTED_OFFLINE:
            score -= 15
        elif record.stability_status == NodeStabilityStatus.UNSTABLE:
            score -= 50

        # 在线率加分
        uptime = record.online_count / max(record.total_transitions, 1)
        score += uptime * 10

        return max(0, min(100, score))

    async def start_monitoring_loop(self, interval: float = 30.0):
        """
        启动监控循环（后台运行）

        定期检查所有节点的状态，发现异常时主动告警
        """
        logger.info(f"[StabilityMgr] 🔍 启动监控循环 (间隔={interval}s)")

        while True:
            try:
                await asyncio.sleep(interval)
                await self._monitoring_check()
            except asyncio.CancelledError:
                logger.info("[StabilityMgr] 监控循环已停止")
                break
            except Exception as e:
                logger.error(f"[StabilityMgr] 监控循环异常: {e}", exc_info=True)

    async def _monitoring_check(self):
        """定期监控检查"""
        now = time.time()
        alerts = []

        for node_id, record in self.records.items():
            if not record:
                continue

            # 检查长时间未更新的节点
            if record.last_status_change > 0:
                time_since_update = now - record.last_status_change
                if time_since_update > 3600:  # 1小时无更新
                    alerts.append({
                        "level": "warning",
                        "node_id": node_id,
                        "message": f"节点状态超过1小时未更新",
                        "last_update": record.last_status_change
                    })

            # 检测持续抖动的节点
            if record.flap_count_1hour > self.FLAP_THRESHOLD_1HOUR:
                alerts.append({
                    "level": "critical",
                    "node_id": node_id,
                    "message": f"节点极度不稳定（1小时内{record.flap_count_1hour}次抖动）",
                    "recommendation": "建议检查网络或硬件问题"
                })

        if alerts:
            logger.warning(f"[StabilityMgr] ⚠️ 发现 {len(alerts)} 个告警:")
            for alert in alerts:
                logger.warning(f"  [{alert['level'].upper()}] {alert['node_id']}: {alert['message']}")

            # TODO: 可接入外部告警系统（邮件/钉钉/企业微信）


# ============================================================
#  故障恢复管理器
# ============================================================

@dataclass
class RecoveryAction:
    """恢复动作"""
    action_type: str  # "migrate" | "degrade" | "wait" | "rollback"
    model_id: str
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    priority: str = "normal"  # "critical" | "high" | "normal" | "low"
    reason: str = ""
    estimated_time_ms: int = 0


class FaultRecoveryManager:
    """
    故障恢复管理器

    当节点掉线时的恢复策略：
    1. 快速评估影响范围（哪些模型受影响）
    2. 优先级排序（活跃用户多的模型优先）
    3. 分级恢复：
       - P0（关键模型）：立即尝试迁移到其他节点
       - P1（重要模型）：等待观察期后迁移
       - P2（一般模型）：标记为不可用，等待节点恢复或手动处理
    4. 回迁机制：原节点恢复后，将模型迁回（如果合适）
    """

    # 配置
    CRITICAL_MODEL_THRESHOLD = 3   # 同时服务请求数>3视为关键模型
    MIGRATION_TIMEOUT_MS = 30000   # 迁移超时时间
    MAX_CONCURRENT_MIGRATIONS = 2  # 最大并发迁移数

    def __init__(self, allocator, stability_manager: NodeStabilityManager):
        self.allocator = allocator
        self.stability_mgr = stability_manager
        self.active_recoveries: Dict[str, List[RecoveryAction]] = {}  # {node_id: [actions]}
        self.recovery_history: List[Dict] = []
        self._migration_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_MIGRATIONS)

        logger.info("[FaultRecovery] ✅ 初始化完成")

    async def handle_node_failure(self, failed_node_id: str) -> Dict[str, Any]:
        """
        处理节点故障（主入口）

        Args:
            failed_node_id: 掉线的节点ID

        Returns:
            恢复计划报告
        """
        logger.warning(f"[FaultRecovery] 🚨 开始处理节点故障: {failed_node_id}")

        start_time = time.time()

        # Step 1: 检查是否应该触发恢复（通过稳定性管理器）
        should_trigger, reason = self.stability_mgr.should_trigger_reallocation(failed_node_id)

        if not should_trigger:
            logger.info(f"[FaultRecovery] ⏸️ 跳过恢复: {reason}")
            return {
                "failed_node": failed_node_id,
                "action": "skipped",
                "reason": reason,
                "timestamp": time.time(),
                "affected_models": [],
                "recovery_actions": []
            }

        # Step 2: 收集受影响的模型
        affected_models = self._collect_affected_models(failed_node_id)

        if not affected_models:
            logger.info(f"[FaultRecovery] ✅ 节点 {failed_node_id} 无已加载模型，无需恢复")
            return {
                "failed_node": failed_node_id,
                "action": "no_action_needed",
                "reason": "节点上无已加载模型",
                "timestamp": time.time(),
                "affected_models": [],
                "recovery_actions": []
            }

        logger.warning(f"[FaultRecovery] 受影响的模型: {[m['model_id'] for m in affected_models]}")

        # Step 3: 生成恢复计划（按优先级排序）
        recovery_actions = self._generate_recovery_plan(failed_node_id, affected_models)

        # 缓存恢复计划
        self.active_recoveries[failed_node_id] = recovery_actions

        # Step 4: 构建报告
        elapsed_ms = (time.time() - start_time) * 1000

        report = {
            "failed_node": failed_node_id,
            "action": "plan_generated",
            "timestamp": time.time(),
            "processing_time_ms": round(elapsed_ms, 1),
            "affected_models": affected_models,
            "recovery_actions": [
                {
                    "model_id": a.model_id,
                    "action_type": a.action_type,
                    "priority": a.priority,
                    "target_node": a.target_node,
                    "reason": a.reason
                }
                for a in recovery_actions
            ],
            "summary": {
                "total_affected": len(affected_models),
                "will_migrate": sum(1 for a in recovery_actions if a.action_type == "migrate"),
                "will_degrade": sum(1 for a in recovery_actions if a.action_type == "degrade"),
                "will_wait": sum(1 for a in recovery_actions if a.action_type == "wait"),
            }
        }

        # 记录历史
        self.recovery_history.append(report)

        logger.info(f"[FaultRecovery] 📋 恢复计划已生成: "
                   f"{report['summary']['will_migrate']}个迁移, "
                   f"{report['summary']['will_degrade']}个降级, "
                   f"{report['summary']['will_wait']}个等待, "
                   f"耗时{elapsed_ms:.0f}ms")

        return report

    def _collect_affected_models(self, failed_node_id: str) -> List[Dict]:
        """收集受影响的模型"""
        affected = []

        if not self.allocator.manager:
            return affected

        node_info = self.allocator.manager.nodes.get(failed_node_id)
        if not node_info:
            return affected

        for model in node_info.loaded_models:
            model_id = model.get("model_id", "unknown")
            shard = model.get("shard", {})

            # 判断是否为关键模型（基于启发式规则）
            is_critical = self._is_model_critical(model_id, shard)

            affected.append({
                "model_id": model_id,
                "shard": shard,
                "is_critical": is_critical,
                "layers_affected": shard.get("end_layer", 0) - shard.get("start_layer", 0) + 1
            })

        # 按关键程度排序（关键的在前）
        affected.sort(key=lambda x: (not x["is_critical"], -x["layers_affected"]))

        return affected

    def _is_model_critical(self, model_id: str, shard: Dict) -> bool:
        """判断模型是否为关键模型"""
        # 规则1: 包含第一层的分片通常是入口，更关键
        if shard.get("start_layer", 0) == 0:
            return True

        # 规则2: 大模型的分片更重要
        layers = shard.get("end_layer", 0) - shard.get("start_layer", 0) + 1
        if layers > 20:
            return True

        # 规则3: 高优先级模型（从库中查找）
        base_id = model_id.split("::")[0] if "::" in model_id else model_id
        spec = self.allocator.model_library.get(base_id)
        if spec and spec.priority >= 1.3:
            return True

        return False

    def _generate_recovery_plan(
        self,
        failed_node_id: str,
        affected_models: List[Dict]
    ) -> List[RecoveryAction]:
        """生成恢复计划"""
        actions = []

        # 获取可用节点
        available_nodes = self.allocator.collect_node_resources()
        available_nodes = [n for n in available_nodes if n.node_id != failed_node_id]

        if not available_nodes:
            # 无可用节点，全部降级
            for model in affected_models:
                actions.append(RecoveryAction(
                    action_type="degrade",
                    model_id=model["model_id"],
                    source_node=failed_node_id,
                    priority="critical" if model["is_critical"] else "normal",
                    reason="无可用节点，标记为不可用"
                ))
            return actions

        for model in affected_models:
            model_id = model["model_id"]
            is_critical = model["is_critical"]

            # 关键模型：尝试迁移
            if is_critical:
                target = self._find_best_migration_target(model_id, available_nodes)
                if target:
                    actions.append(RecoveryAction(
                        action_type="migrate",
                        model_id=model_id,
                        source_node=failed_node_id,
                        target_node=target,
                        priority="critical",
                        reason=f"关键模型，迁移至 {target}"
                    ))
                else:
                    actions.append(RecoveryAction(
                        action_type="degrade",
                        model_id=model_id,
                        source_node=failed_node_id,
                        priority="critical",
                        reason="关键模型但无合适目标节点"
                    ))
            else:
                # 非关键模型：先等待观察
                actions.append(RecoveryAction(
                    action_type="wait",
                    model_id=model_id,
                    source_node=failed_node_id,
                    priority="low",
                    reason=f"非关键模型，等待节点恢复（观察{self.stability_mgr.OBSERVATION_PERIOD}s）"
                ))

        return actions

    def _find_best_migration_target(
        self,
        model_id: str,
        available_nodes: List[NodeResource]
    ) -> Optional[str]:
        """寻找最佳迁移目标节点"""
        base_id = model_id.split("::")[0] if "::" in model_id else model_id
        spec = self.allocator.model_library.get(base_id)

        if not spec:
            # 未知模型，选择空间最大的节点
            best = max(available_nodes, key=lambda n: n.usable_memory_mb)
            return best.node_id if best.usable_memory_mb > 1024 else None  # 至少1GB可用

        model_mem = spec.total_memory_mb

        # 寻找能容纳该模型的最佳节点
        candidates = []
        for node in available_nodes:
            if node.usable_memory_mb >= model_mem:
                # 优先选择利用率低且空间充裕的节点
                score = node.usable_memory_mb - node.used_memory_mb
                candidates.append((score, node))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1].node_id

        # 单节点放不下，检查是否可以拆分
        if len(available_nodes) >= 2:
            total_available = sum(n.usable_memory_mb for n in available_nodes)
            if total_available >= model_mem:
                # 选择空间最大的两个节点
                sorted_nodes = sorted(available_nodes, key=lambda n: n.usable_memory_mb, reverse=True)
                return sorted_nodes[0].node_id  # 返回主节点

        return None

    async def execute_recovery_plan(self, failed_node_id: str) -> Dict[str, Any]:
        """执行恢复计划"""
        recovery_actions = self.active_recoveries.get(failed_node_id, [])

        if not recovery_actions:
            return {"status": "no_plan", "message": "无待执行的恢复计划"}

        results = {"success": [], "failed": [], "skipped": []}

        for action in recovery_actions:
            try:
                async with self._migration_semaphore:
                    if action.action_type == "migrate":
                        result = await self._execute_migration(action)
                        if result["success"]:
                            results["success"].append(result)
                        else:
                            results["failed"].append(result)

                    elif action.action_type == "degrade":
                        result = await self._execute_degradation(action)
                        results["success"].append(result)  # 降级总是成功的

                    elif action.action_type == "wait":
                        results["skipped"].append({
                            "model_id": action.model_id,
                            "reason": action.reason
                        })

            except Exception as e:
                logger.error(f"[FaultRecovery] ❌ 执行恢复动作失败: {e}")
                results["failed"].append({
                    "model_id": action.model_id,
                    "error": str(e)
                })

        # 清理已完成的活动恢复
        if failed_node_id in self.active_recoveries:
            del self.active_recoveries[failed_node_id]

        return {
            "failed_node": failed_node_id,
            "status": "completed",
            "timestamp": time.time(),
            "results": results,
            "summary": {
                "total": len(recovery_actions),
                "success": len(results["success"]),
                "failed": len(results["failed"]),
                "skipped": len(results["skipped"])
            }
        }

    async def _execute_migration(self, action: RecoveryAction) -> Dict:
        """执行模型迁移"""
        logger.info(f"[FaultRecovery] 🔄 迁移模型: {action.model_id} "
                   f"{action.source_node} → {action.target_node}")

        # TODO: 实际的迁移逻辑（调用GPU池接口）
        # 这里是占位实现，实际需要对接 gpu_pool_integration

        await asyncio.sleep(0.1)  # 模拟异步操作

        return {
            "model_id": action.model_id,
            "success": True,
            "action": "migrated",
            "from": action.source_node,
            "to": action.target_node,
            "timestamp": time.time()
        }

    async def _execute_degradation(self, action: RecoveryAction) -> Dict:
        """执行降级处理"""
        logger.warning(f"[FaultRecovery] ⬇️ 降级模型: {action.model_id} ({action.reason})")

        # 标记模型为不可用
        # TODO: 更新路由表，将该模型标记为 unavailable

        return {
            "model_id": action.model_id,
            "success": True,
            "action": "degraded",
            "reason": action.reason,
            "timestamp": time.time()
        }


# ============================================================
#  分配方案版本管理器
# ============================================================

@dataclass
class PlanVersion:
    """方案版本"""
    version_id: str
    plan: Any  # ModelAllocationPlan
    created_at: float
    trigger_reason: str           # 触发原因
    parent_version: Optional[str] = None  # 父版本ID（用于回滚）
    checksum: str = ""            # 方案校验和
    is_active: bool = False       # 是否为当前激活版本
    rollback_allowed: bool = True # 是否允许回滚


class AllocationPlanManager:
    """
    分配方案版本管理器

    功能：
    1. 保存每次生成的分配方案
  2. 支持方案对比（查看差异）
    3. 支持回滚到历史版本
    4. 自动清理过期版本
    """

    MAX_VERSIONS = 20             # 最大保留版本数
    VERSION_RETENTION_DAYS = 7    # 版本保留天数

    def __init__(self):
        self.versions: Dict[str, PlanVersion] = {}
        self.version_history: List[str] = []  # version_id 按时间顺序
        self.current_version_id: Optional[str] = None

        logger.info("[PlanManager] ✅ 初始化完成")

    def save_plan(self, plan, trigger_reason: str = "manual") -> str:
        """
        保存分配方案为新版本

        Returns:
            version_id
        """
        version_id = f"v{len(self.versions)+1}_{int(time.time())}"

        # 计算校验和（简化版）
        import hashlib
        plan_json = json.dumps(plan.to_dict(), sort_keys=True, default=str)
        checksum = hashlib.md5(plan_json.encode()).hexdigest()[:12]

        version = PlanVersion(
            version_id=version_id,
            plan=plan,
            created_at=time.time(),
            trigger_reason=trigger_reason,
            parent_version=self.current_version_id,
            checksum=checksum,
            is_active=True
        )

        # 将旧版本标记为非激活
        if self.current_version_id and self.current_version_id in self.versions:
            self.versions[self.current_version_id].is_active = False

        # 保存新版本
        self.versions[version_id] = version
        self.version_history.append(version_id)
        self.current_version_id = version_id

        # 清理过期版本
        self._cleanup_old_versions()

        logger.info(f"[PlanManager] 💾 方案已保存: {version_id} "
                   f"(原因: {trigger_reason}, 校验和: {checksum})")

        return version_id

    def get_version(self, version_id: str) -> Optional[PlanVersion]:
        """获取指定版本"""
        return self.versions.get(version_id)

    def get_current_version(self) -> Optional[PlanVersion]:
        """获取当前激活版本"""
        if self.current_version_id:
            return self.versions.get(self.current_version_id)
        return None

    def list_versions(self, limit: int = 10) -> List[Dict]:
        """列出最近的版本"""
        recent = self.version_history[-limit:]
        versions_info = []

        for vid in reversed(recent):
            v = self.versions.get(vid)
            if not v:
                continue

            versions_info.append({
                "version_id": v.version_id,
                "created_at": v.created_at,
                "trigger_reason": v.trigger_reason,
                "checksum": v.checksum,
                "is_active": v.is_active,
                "parent_version": v.parent_version,
                "models_count": getattr(v.plan, 'total_models', 0) if v.plan else 0,
                "performance_score": getattr(v.plan, 'performance_score', 0) if v.plan else 0,
            })

        return versions_info

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """对比两个版本的差异"""
        v1 = self.versions.get(version_id1)
        v2 = self.versions.get(version_id2)

        if not v1 or not v2:
            raise ValueError("指定的版本不存在")

        plan1 = v1.plan.to_dict() if v1.plan else {}
        plan2 = v2.plan.to_dict() if v2.plan else {}

        alloc1 = set(plan1.get("allocations", {}).keys())
        alloc2 = set(plan2.get("allocations", {}).keys())

        added_models = alloc2 - alloc1
        removed_models = alloc1 - alloc2
        common_models = alloc1 & alloc2

        # 检查共同模型是否有变化
        changed_models = []
        for model_id in common_models:
            nodes1 = set(plan1["allocations"][model_id].keys())
            nodes2 = set(plan2["allocations"][model_id].keys())
            if nodes1 != nodes2:
                changed_models.append({
                    "model_id": model_id,
                    "change": f"节点从 {nodes1} 变更为 {nodes2}"
                })

        return {
            "version1": version_id,
            "version2": version_id,
            "summary": {
                "added_models": list(added_models),
                "removed_models": list(removed_models),
                "changed_models": changed_models,
                "models_count_v1": len(alloc1),
                "models_count_v2": len(alloc2),
            },
            "metrics_diff": {
                "memory_utilization": (
                    plan2.get("summary", {}).get("memory_utilization", 0) -
                    plan1.get("summary", {}).get("memory_utilization", 0)
                ),
                "performance_score": (
                    plan2.get("summary", {}).get("performance_score", 0) -
                    plan1.get("summary", {}).get("performance_score", 0)
                ),
            }
        }

    def can_rollback_to(self, version_id: str) -> Tuple[bool, str]:
        """检查是否可以回滚到指定版本"""
        version = self.versions.get(version_id)

        if not version:
            return False, "版本不存在"

        if not version.rollback_allowed:
            return False, "该版本不允许回滚"

        if version.is_active:
            return False, "该版本已是当前激活版本"

        # 检查版本时效（不超过保留期限）
        age_days = (time.time() - version.created_at) / 86400
        if age_days > self.VERSION_RETENTION_DAYS:
            return False, f"版本已过期（{age_days:.1f}天前）"

        return True, "可以回滚"

    def rollback_to(self, version_id: str) -> Optional[Any]:
        """
        回滚到指定版本

        Returns:
            回滚后的方案（ModelAllocationPlan）
        """
        can_rollback, reason = self.can_rollback_to(version_id)

        if not can_rollback:
            raise ValueError(f"无法回滚: {reason}")

        version = self.versions[version_id]

        # 创建新版本（基于旧方案）
        new_version_id = self.save_plan(
            version.plan,
            trigger_reason=f"rollback_from_{self.current_version_id}"
        )

        logger.warning(f"[PlanManager] ⏪ 已回滚到版本 {version_id} "
                      f"(新版本: {new_version_id})")

        return version.plan

    def _cleanup_old_versions(self):
        """清理过期版本"""
        while len(self.versions) > self.MAX_VERSIONS:
            oldest_id = self.version_history.pop(0)
            if oldest_id != self.current_version_id:
                del self.versions[oldest_id]
                logger.debug(f"[PlanManager] 🗑️ 清理过期版本: {oldest_id}")


# ============================================================
#  集成管理器 - 整合稳定性+故障恢复+版本管理
# ============================================================

class ResilientAllocationManager:
    """
    弹性分配管理器（整合层）

    对外统一接口，内部协调：
    - AutoModelAllocator: 核心分配算法
    - NodeStabilityManager: 节点稳定性检测
    - FaultRecoveryManager: 故障恢复
    - AllocationPlanManager: 版本管理

    使用方式：
        resilient_mgr = ResilientAllocationManager(cluster_manager)

        # 节点状态变化时调用
        await resilient_mgr.on_node_event(node_id, "offline")

        # 正常分配（带防护）
        plan = await resilient_mgr.safe_generate_plan(strategy="balanced")
    """

    def __init__(self, cluster_manager):
        from auto_model_allocator import AutoModelAllocator

        self.cluster_manager = cluster_manager
        self.allocator = AutoModelAllocator(cluster_manager)
        self.stability_mgr = NodeStabilityManager(self.allocator)
        self.recovery_mgr = FaultRecoveryManager(self.allocator, self.stability_mgr)
        self.plan_mgr = AllocationPlanManager()

        # 事件回调注册表
        self._event_callbacks: Dict[str, List] = {
            "on_flapping_detected": [],
            "on_recovery_triggered": [],
            "on_plan_changed": [],
        }

        logger.info("[ResilientAlloc] ✅ 弹性分配管理器初始化完成")

    def register_callback(self, event_type: str, callback):
        """注册事件回调"""
        if event_type in self._event_callbacks:
            self._event_callbacks[event_type].append(callback)

    async def on_node_event(
        self,
        node_id: str,
        event_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理节点事件（主入口）

        Args:
            node_id: 节点ID
            event_type: 事件类型 ("online" | "offline" | "error" | "flapping")
            **kwargs: 额外信息（如 loaded_models）

        Returns:
            处理结果
        """
        logger.info(f"[ResilientAlloc] 📥 节点事件: {node_id} → {event_type}")

        # 1. 更新稳定性状态
        stability_result = await self.stability_mgr.on_node_status_changed(
            node_id,
            event_type,
            loaded_models=kwargs.get("loaded_models")
        )

        # 2. 根据稳定性状态决定后续动作
        result = {
            "node_id": node_id,
            "event": event_type,
            "stability": stability_result,
            "actions_taken": []
        }

        # 触发相关回调
        if stability_result["stability_status"] == "flapping":
            await self._emit_event("on_flapping_detected", stability_result)

        # 3. 如果需要故障恢复
        if stability_result.get("should_reallocate"):
            recovery_plan = await self.recovery_mgr.handle_node_failure(node_id)
            result["recovery_plan"] = recovery_plan
            result["actions_taken"].append("recovery_plan_generated")

            await self._emit_event("on_recovery_triggered", recovery_plan)

            # 自动执行恢复（可选，可通过配置控制）
            if kwargs.get("auto_execute", False):
                exec_result = await self.recovery_mgr.execute_recovery_plan(node_id)
                result["execution_result"] = exec_result
                result["actions_taken"].append("recovery_executed")

        return result

    async def safe_generate_plan(
        self,
        strategy: str = "maximize_utilization",
        force: bool = False,
        **kwargs
    ):
        """
        安全地生成分配方案（带防护检查）

        Args:
            strategy: 分配策略
            force: 是否强制生成（跳过安全检查）
        """
        if not force:
            # 检查是否有节点处于不稳定状态
            stability_report = self.stability_mgr.get_node_stability_report()

            unstable_count = stability_report["summary"].get("unstable_nodes", 0)
            flapping_count = stability_report["summary"].get("flapping_nodes", 0)

            if unstable_count > 0 or flapping_count > 0:
                logger.warning(f"[ResilientAlloc] ⚠️ 检测到不稳定节点 "
                             f"(不稳定:{unstable_count}, 抖动:{flapping_count})，"
                             f"生成方案可能不准确")

        # 生成方案
        from auto_model_allocator import AllocationStrategy
        plan = self.allocator.generate_optimal_plan(
            strategy=AllocationStrategy(strategy),
            **kwargs
        )

        # 保存版本
        version_id = self.plan_mgr.save_plan(
            plan,
            trigger_reason=f"strategy_{strategy}"
        )

        await self._emit_event("on_plan_changed", {
            "plan": plan,
            "version_id": version_id
        })

        return plan, version_id

    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据（供前端展示）"""
        return {
            "stability": self.stability_mgr.get_node_stability_report(),
            "recent_plans": self.plan_mgr.list_versions(limit=5),
            "recent_recoveries": self.recovery_mgr.recovery_history[-5:] if self.recovery_mgr.recovery_history else [],
            "allocator_status": self.allocator.get_current_status(),
        }

    async def _emit_event(self, event_type: str, data: Any):
        """发射事件给注册的回调"""
        for callback in self._event_callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"[ResilientAlloc] 事件回调执行失败 ({event_type}): {e}")
