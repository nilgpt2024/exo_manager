#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多实例负载均衡测试脚本

功能:
1. 测试所有负载均衡策略的分配效果
2. 模拟和执行真实的推理请求
3. 验证实例间的请求分布均匀性
4. 监控推理性能和健康状态
5. 生成详细的测试报告

使用示例:
    # 基础测试（只测试选择逻辑，不实际推理）
    python test_load_balancer.py --model qwen3-0.6b --requests 20
    
    # 完整测试（包含真实推理）
    python test_load_balancer.py --model qwen3-0.6b --requests 10 --real-inference \
        --prompt "你好，请介绍一下你自己"
    
    # 测试特定策略
    python test_load_balancer.py --model qwen3-0.6b --strategy round_robin --requests 30
    
    # 测试所有策略
    python test_load_balancer.py --model qwen3-0.6b --test-all-strategies
    
    # 压力测试
    python test_load_balancer.py --model qwen3-0.6b --stress-test --requests 100

作者: EXO Team
日期: 2025
"""

import argparse
import asyncio
import json
import time
import sys
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime


@dataclass
class TestResult:
    """单个测试结果"""
    request_id: int
    instance_id: str
    node_id: str
    strategy: str
    success: bool
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class StrategyTestSummary:
    """策略测试总结"""
    strategy_name: str
    total_requests: int
    successful_requests: int
    distribution: Dict[str, int]
    balance_score: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    success_rate: float


class LoadBalancerTester:
    """负载均衡测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        初始化测试器
        
        Args:
            base_url: Manager API 基础URL
        """
        self.base_url = base_url.rstrip("/")
        self.results: List[TestResult] = []
        
        print(f"🚀 负载均衡测试器初始化完成")
        print(f"   API地址: {self.base_url}")
    
    async def _make_request(self, method: str, endpoint: str, 
                           data: Optional[Dict] = None) -> Dict[str, Any]:
        """发送HTTP请求"""
        try:
            import httpx
            
            url = f"{self.base_url}{endpoint}"
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                if method.upper() == "GET":
                    response = await client.get(url)
                else:
                    response = await client.post(url, json=data)
                
                return response.json()
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_available_instances(self, model_id: str) -> List[Dict]:
        """获取模型的所有可用实例"""
        result = await self._make_request("GET", f"/api/lb/instances/{model_id}")
        
        if result.get("success"):
            return result.get("data", {}).get("instances", [])
        return []
    
    async def test_strategy_selection(
        self,
        model_id: str,
        strategy: str,
        num_requests: int = 20
    ) -> Tuple[List[TestResult], StrategyTestSummary]:
        """
        测试指定策略的选择逻辑
        
        Args:
            model_id: 模型ID
            strategy: 策略名称
            num_requests: 测试请求数
            
        Returns:
            (测试结果列表, 策略总结)
        """
        print(f"\n{'='*70}")
        print(f"🧪 测试策略: {strategy.upper()}")
        print(f"   模型: {model_id}")
        print(f"   请求数: {num_requests}")
        print(f"{'='*70}")
        
        results = []
        instance_distribution: Dict[str, int] = defaultdict(int)
        latencies = []
        success_count = 0
        
        for i in range(num_requests):
            result = await self._make_request("POST", "/api/lb/test", {
                "model_id": model_id,
                "strategy": strategy,
                "test_requests": 1  # 每次只测试1次，便于记录详细日志
            })
            
            if result.get("success"):
                data = result.get("data", {})
                details = data.get("details", [])
                
                if details:
                    detail = details[0]
                    instance_id = detail["selected_instance"]
                    node_id = detail["selected_node"]
                    
                    instance_distribution[instance_id] += 1
                    
                    test_result = TestResult(
                        request_id=i + 1,
                        instance_id=instance_id,
                        node_id=node_id,
                        strategy=strategy,
                        success=True
                    )
                    
                    results.append(test_result)
                    success_count += 1
                    
                    if (i + 1) % max(1, num_requests // 10) == 0 or i == 0 or i == num_requests - 1:
                        print(f"   ✅ 请求 #{i+1}: 实例={instance_id}, 节点={node_id}")
                else:
                    results.append(TestResult(
                        request_id=i + 1,
                        instance_id="N/A",
                        node_id="N/A",
                        strategy=strategy,
                        success=False,
                        error="无返回详情"
                    ))
            else:
                results.append(TestResult(
                    request_id=i + 1,
                    instance_id="ERROR",
                    node_id="ERROR",
                    strategy=strategy,
                    success=False,
                    error=result.get("error", "未知错误")
                ))
                print(f"   ❌ 请求 #{i+1} 失败: {result.get('error', '未知错误')}")
        
        # 计算总结
        balance_score = self._calculate_balance_score(instance_distribution)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        summary = StrategyTestSummary(
            strategy_name=strategy,
            total_requests=num_requests,
            successful_requests=success_count,
            distribution=dict(instance_distribution),
            balance_score=balance_score,
            avg_latency_ms=avg_latency,
            min_latency_ms=min(latencies) if latencies else 0.0,
            max_latency_ms=max(latencies) if latencies else 0.0,
            success_rate=(success_count / num_requests * 100) if num_requests > 0 else 0.0
        )
        
        # 打印摘要
        self._print_strategy_summary(summary)
        
        return results, summary
    
    async def test_real_inference(
        self,
        model_id: str,
        strategy: str,
        prompt: str,
        num_requests: int = 5,
        stream: bool = False
    ) -> Tuple[List[TestResult], StrategyTestSummary]:
        """
        执行真实的推理测试
        
        Args:
            model_id: 模型ID
            strategy: 负载均衡策略
            prompt: 推理提示词
            num_requests: 请求数
            stream: 是否使用流式输出
            
        Returns:
            (测试结果列表, 策略总结)
        """
        print(f"\n{'='*70}")
        print(f"🔥 真实推理测试: {strategy.upper()}")
        print(f"   模型: {model_id}")
        print(f"   提示词: {prompt[:50]}...")
        print(f"   请求数: {num_requests}")
        print(f"{'='*70}")
        
        results = []
        instance_distribution: Dict[str, int] = defaultdict(int)
        latencies = []
        success_count = 0
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                import httpx
                
                url = f"{self.base_url}/api/inference/chat"
                payload = {
                    "model_id": model_id,
                    "lb_strategy": strategy,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": stream,
                    "max_tokens": 50,
                    "temperature": 0.7
                }
                
                async with httpx.AsyncClient(timeout=120.0) as client:
                    if stream:
                        response = await client.post(url, json=payload)
                        
                        if response.status_code == 200:
                            full_response = ""
                            async for line in response.aiter_lines():
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    try:
                                        chunk = json.loads(line[6:])
                                        if chunk.get("token"):
                                            full_response += chunk["token"]
                                    except:
                                        pass
                            
                            latency_ms = (time.time() - start_time) * 1000
                            latencies.append(latency_ms)
                            
                            # 从响应头或日志中获取选择的实例信息
                            # 这里我们无法直接获取，所以标记为 unknown
                            instance_id = f"inferred_{i}"
                            
                            instance_distribution[instance_id] += 1
                            success_count += 1
                            
                            test_result = TestResult(
                                request_id=i + 1,
                                instance_id=instance_id,
                                node_id="inferred",
                                strategy=strategy,
                                success=True,
                                latency_ms=latency_ms
                            )
                            
                            results.append(test_result)
                            
                            print(f"   ✅ 请求 #{i+1}: 成功 ({latency_ms:.1f}ms)")
                        else:
                            raise Exception(f"HTTP {response.status_code}")
                    else:
                        response = await client.post(url, json=payload)
                        
                        if response.status_code == 200:
                            latency_ms = (time.time() - start_time) * 1000
                            latencies.append(latency_ms)
                            
                            instance_id = f"inferred_{i}"
                            instance_distribution[instance_id] += 1
                            success_count += 1
                            
                            test_result = TestResult(
                                request_id=i + 1,
                                instance_id=instance_id,
                                node_id="inferred",
                                strategy=strategy,
                                success=True,
                                latency_ms=latency_ms
                            )
                            
                            results.append(test_result)
                            
                            print(f"   ✅ 请求 #{i+1}: 成功 ({latency_ms:.1f}ms)")
                        else:
                            raise Exception(f"HTTP {response.status_code}")
                            
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                
                test_result = TestResult(
                    request_id=i + 1,
                    instance_id="ERROR",
                    node_id="ERROR",
                    strategy=strategy,
                    success=False,
                    latency_ms=latency_ms,
                    error=str(e)
                )
                
                results.append(test_result)
                print(f"   ❌ 请求 #{i+1} 失败: {str(e)[:80]}")
            
            # 避免请求过于频繁
            if i < num_requests - 1:
                await asyncio.sleep(0.2)
        
        # 计算总结
        balance_score = self._calculate_balance_score(instance_distribution)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        summary = StrategyTestSummary(
            strategy_name=strategy,
            total_requests=num_requests,
            successful_requests=success_count,
            distribution=dict(instance_distribution),
            balance_score=balance_score,
            avg_latency_ms=avg_latency,
            min_latency_ms=min(latencies) if latencies else 0.0,
            max_latency_ms=max(latencies) if latencies else 0.0,
            success_rate=(success_count / num_requests * 100) if num_requests > 0 else 0.0
        )
        
        self._print_inference_summary(summary)
        
        return results, summary
    
    def _calculate_balance_score(self, distribution: Dict[str, int]) -> float:
        """计算负载均衡分数"""
        if not distribution or len(distribution) <= 1:
            return 100.0
        
        total = sum(distribution.values())
        expected = total / len(distribution)
        
        variance = sum((count - expected) ** 2 for count in distribution.values()) / len(distribution)
        
        if expected == 0:
            return 100.0
        
        coefficient_of_variation = (variance ** 0.5) / expected
        score = max(0, 100 * (1 - coefficient_of_variation))
        
        return round(score, 2)
    
    def _print_strategy_summary(self, summary: StrategyTestSummary):
        """打印策略测试摘要"""
        print(f"\n📊 策略测试摘要 [{summary.strategy_name}]")
        print(f"   {'─'*50}")
        print(f"   总请求数:     {summary.total_requests}")
        print(f"   成功数:       {summary.successful_requests}/{summary.total_requests} "
              f"({summary.success_rate:.1f}%)")
        print(f"   均衡分数:     {summary.balance_score}/100 {'✅优秀' if summary.balance_score >= 90 else '⚠️一般' if summary.balance_score >= 70 else '❌较差'}")
        
        print(f"\n   📈 实例分布:")
        sorted_dist = sorted(summary.distribution.items(), key=lambda x: x[1], reverse=True)
        for instance_id, count in sorted_dist:
            percentage = (count / summary.total_requests * 100) if summary.total_requests > 0 else 0
            bar_length = int(percentage / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"      {instance_id:15s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    def _print_inference_summary(self, summary: StrategyTestSummary):
        """打印推理测试摘要"""
        print(f"\n📊 推理性能摘要 [{summary.strategy_name}]")
        print(f"   {'─'*50}")
        print(f"   成功率:       {summary.success_rate:.1f}%")
        
        if summary.avg_latency_ms > 0:
            print(f"   平均延迟:     {summary.avg_latency_ms:.1f}ms")
            print(f"   最小延迟:     {summary.min_latency_ms:.1f}ms")
            print(f"   最大延迟:     {summary.max_latency_ms:.1f}ms")
        
        print(f"\n   🎯 实例分布:")
        sorted_dist = sorted(summary.distribution.items(), key=lambda x: x[1], reverse=True)
        for instance_id, count in sorted_dist:
            percentage = (count / summary.successful_requests * 100) if summary.successful_requests > 0 else 0
            bar_length = int(percentage / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"      {instance_id:15s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    async def run_all_strategies_test(
        self,
        model_id: str,
        requests_per_strategy: int = 20
    ) -> Dict[str, StrategyTestSummary]:
        """
        测试所有策略
        
        Args:
            model_id: 模型ID
            requests_per_strategy: 每个策略的请求数
            
        Returns:
            所有策略的测试总结
        """
        strategies = [
            "first_layer",
            "round_robin",
            "random",
            "weighted",
            "least_connections"
        ]
        
        all_summaries = {}
        
        print("\n" + "="*70)
        print("🎯 开始测试所有负载均衡策略")
        print("="*70)
        
        for strategy in strategies:
            _, summary = await self.test_strategy_selection(
                model_id=model_id,
                strategy=strategy,
                num_requests=requests_per_strategy
            )
            
            all_summaries[strategy] = summary
            
            # 短暂暂停，避免请求过快
            await asyncio.sleep(0.5)
        
        # 打印最终对比报告
        self._print_comparison_report(all_summaries)
        
        return all_summaries
    
    def _print_comparison_report(self, summaries: Dict[str, StrategyTestSummary]):
        """打印策略对比报告"""
        print("\n" + "="*70)
        print("📋 策略对比报告")
        print("="*70)
        
        header = f"\n{'策略':<20} | {'成功率':>7} | {'均衡分':>7} | {'平均延迟':>9} | {'评价'}"
        print(header)
        print("-" * len(header))
        
        best_balance_strategy = None
        best_balance_score = 0
        
        for strategy, summary in summaries.items():
            if summary.balance_score > best_balance_score:
                best_balance_score = summary.balance_score
                best_balance_strategy = strategy
            
            rating = "✅ 优秀" if summary.balance_score >= 90 else \
                     "⚠️ 一般" if summary.balance_score >= 70 else "❌ 较差"
            
            latency_str = f"{summary.avg_latency_ms:.1f}ms" if summary.avg_latency_ms > 0 else "N/A"
            
            print(f"{strategy:<20} | {summary.success_rate:>6.1f}% | "
                  f"{summary.balance_score:>6.1f} | {latency_str:>9} | {rating}")
        
        print("-" * len(header))
        
        if best_balance_strategy:
            print(f"\n🏆 最佳均衡策略: {best_balance_strategy} (均衡分: {best_balance_score})")
    
    async def stress_test(
        self,
        model_id: str,
        total_requests: int = 100,
        concurrency: int = 5
    ) -> Dict[str, Any]:
        """
        压力测试
        
        Args:
            model_id: 模型ID
            total_requests: 总请求数
            concurrency: 并发数
            
        Returns:
            测试结果
        """
        print(f"\n{'='*70}")
        print(f"💪 压力测试模式")
        print(f"   模型: {model_id}")
        print(f"   总请求: {total_requests}")
        print(f"   并发数: {concurrency}")
        print(f"{'='*70}")
        
        semaphore = asyncio.Semaphore(concurrency)
        completed = [0]
        failed = [0]
        latencies = []
        instance_counts: Dict[str, int] = defaultdict(int)
        start_time = time.time()
        
        async def make_single_request(req_id: int):
            async with semaphore:
                req_start = time.time()
                
                result = await self._make_request("POST", "/api/lb/test", {
                    "model_id": model_id,
                    "strategy": "round_robin",
                    "test_requests": 1
                })
                
                latency = (time.time() - req_start) * 1000
                latencies.append(latency)
                
                if result.get("success"):
                    completed[0] += 1
                    
                    details = result.get("data", {}).get("details", [])
                    if details:
                        instance_id = details[0]["selected_instance"]
                        instance_counts[instance_id] += 1
                else:
                    failed[0] += 1
                
                if (completed[0] + failed[0]) % max(1, total_requests // 10) == 0:
                    progress = ((completed[0] + failed[0]) / total_requests) * 100
                    print(f"   进度: {progress:.1f}% "
                          f"(成功={completed[0]}, 失败={failed[0]})")
        
        tasks = [make_single_request(i) for i in range(total_requests)]
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        print(f"\n📊 压力测试结果:")
        print(f"   {'─'*50}")
        print(f"   总耗时:       {total_time:.2f}s")
        print(f"   吞吐量:       {total_requests/total_time:.2f} req/s")
        print(f"   成功请求:     {completed[0]}/{total_requests} "
              f"({completed[0]/total_requests*100:.1f}%)")
        print(f"   失败请求:     {failed[0]}/{total_requests}")
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p50 = sorted(latencies)[len(latencies)//2]
            p95 = sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 20 else latencies[-1]
            p99 = sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) > 100 else latencies[-1]
            
            print(f"\n   ⏱️  延迟统计:")
            print(f"      平均延迟:   {avg_latency:.1f}ms")
            print(f"      P50延迟:    {p50:.1f}ms")
            print(f"      P95延迟:    {p95:.1f}ms")
            print(f"      P99延迟:    {p99:.1f}ms")
        
        if instance_counts:
            print(f"\n   📈 实例分布:")
            for instance_id, count in sorted(instance_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / completed[0] * 100) if completed[0] > 0 else 0
                print(f"      {instance_id}: {count} ({percentage:.1f}%)")
        
        return {
            "total_requests": total_requests,
            "completed": completed[0],
            "failed": failed[0],
            "total_time_s": total_time,
            "throughput": total_requests / total_time,
            "instance_distribution": dict(instance_counts),
            "latencies": {
                "avg": sum(latencies)/len(latencies) if latencies else 0,
                "p50": sorted(latencies)[len(latencies)//2] if latencies else 0,
                "p95": sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 20 else 0,
                "p99": sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) > 100 else 0
            }
        }


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="EXO 多实例负载均衡测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --model qwen3-0.6b --requests 20
  %(prog)s --model qwen3-0.6b --real-inference --prompt "你好"
  %(prog)s --model qwen3-0.6b --test-all-strategies
  %(prog)s --model qwen3-0.6b --stress-test --requests 100
        """
    )
    
    parser.add_argument("--url", type=str, default="http://localhost:8080",
                       help="Manager API 地址 (默认: http://localhost:8080)")
    parser.add_argument("--model", type=str, required=True,
                       help="要测试的模型ID")
    parser.add_argument("--strategy", type=str, default="round_robin",
                       choices=["first_layer", "round_robin", "random", 
                               "weighted", "least_connections"],
                       help="负载均衡策略 (默认: round_robin)")
    parser.add_argument("--requests", type=int, default=20,
                       help="测试请求数 (默认: 20)")
    parser.add_argument("--real-inference", action="store_true",
                       help="执行真实推理测试（而非仅测试选择逻辑）")
    parser.add_argument("--prompt", type=str, default="你好，请用一句话介绍自己",
                       help="推理提示词 (默认: '你好，请用一句话介绍自己')")
    parser.add_argument("--stream", action="store_true",
                       help="使用流式输出")
    parser.add_argument("--test-all-strategies", action="store_true",
                       help="测试所有负载均衡策略")
    parser.add_argument("--stress-test", action="store_true",
                       help="运行压力测试")
    parser.add_argument("--concurrency", type=int, default=5,
                       help="压力测试并发数 (默认: 5)")
    parser.add_argument("--output-json", type=str, default=None,
                       help="将结果保存到JSON文件")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 EXO 多实例负载均衡测试工具")
    print("="*70)
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   目标: {args.url}")
    print(f"   模型: {args.model}")
    
    tester = LoadBalancerTester(base_url=args.url)
    
    final_results = {}
    
    try:
        if args.stress_test:
            # 压力测试
            results = await tester.stress_test(
                model_id=args.model,
                total_requests=args.requests,
                concurrency=args.concurrency
            )
            final_results["stress_test"] = results
            
        elif args.test_all_strategies:
            # 测试所有策略
            summaries = await tester.run_all_strategies_test(
                model_id=args.model,
                requests_per_strategy=args.requests
            )
            final_results["all_strategies"] = {
                name: {
                    "total_requests": s.total_requests,
                    "successful_requests": s.successful_requests,
                    "distribution": s.distribution,
                    "balance_score": s.balance_score,
                    "success_rate": s.success_rate
                }
                for name, s in summaries.items()
            }
            
        elif args.real_inference:
            # 真实推理测试
            _, summary = await tester.test_real_inference(
                model_id=args.model,
                strategy=args.strategy,
                prompt=args.prompt,
                num_requests=args.requests,
                stream=args.stream
            )
            final_results["inference_test"] = {
                "strategy": args.strategy,
                "total_requests": summary.total_requests,
                "successful_requests": summary.successful_requests,
                "avg_latency_ms": summary.avg_latency_ms,
                "success_rate": summary.success_rate,
                "distribution": summary.distribution
            }
            
        else:
            # 默认：仅测试选择逻辑
            _, summary = await tester.test_strategy_selection(
                model_id=args.model,
                strategy=args.strategy,
                num_requests=args.requests
            )
            final_results["selection_test"] = {
                "strategy": args.strategy,
                "total_requests": summary.total_requests,
                "successful_requests": summary.successful_requests,
                "balance_score": summary.balance_score,
                "distribution": summary.distribution
            }
        
        # 保存结果到JSON文件
        if args.output_json:
            with open(args.output_json, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 结果已保存到: {args.output_json}")
        
        print("\n" + "="*70)
        print("✅ 测试完成！")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断测试")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())