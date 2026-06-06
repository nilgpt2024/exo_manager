#!/usr/bin/env python3
"""
多实例模型加载测试脚本
======================

测试 exo_manager 的多实例功能：
- 加载同一模型的多个实例
- 查询实例信息
- 卸载指定实例或所有实例

使用方式:
    python test_multi_instance.py
    
前提条件:
    1. exo_manager 服务已启动 (python start_manager.py)
    2. 至少有一个在线的 EXO 节点

测试场景:
    1. 加载默认实例
    2. 加载多个命名实例（worker-1, worker-2）
    3. 使用自动生成实例ID
    4. 查询实例列表
    5. 卸载指定实例
    6. 卸载所有实例
"""

import asyncio
import aiohttp
import json
import sys
from typing import Dict, Any


class MultiInstanceTester:
    """多实例功能测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _api_call(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """通用API调用方法"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    result = await response.json()
            else:
                async with self.session.post(url, json=data) as response:
                    result = await response.json()
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def print_section(self, title: str):
        """打印测试分节标题"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    async def print_result(self, operation: str, result: Dict):
        """格式化打印结果"""
        status = "✅ 成功" if result.get("success") else "❌ 失败"
        print(f"\n{operation}")
        print(f"  状态: {status}")
        
        if not result.get("success"):
            print(f"  错误: {result.get('error', '未知错误')}")
            return
        
        # 打印关键信息
        if "instance_id" in result:
            print(f"  实例ID: {result['instance_id']}")
        if "full_model_id" in result:
            print(f"  完整ID: {result['full_model_id']}")
        if "summary" in result:
            summary = result["summary"]
            print(f"  摘要: 成功节点={summary.get('success_nodes', 0)}, "
                  f"失败节点={summary.get('failed_nodes', 0)}")
    
    async def test_1_load_default_instance(self):
        """测试1：加载默认实例"""
        await self.print_section("测试1: 加载默认实例")
        
        model_id = "qwen3-0.6b"
        
        print(f"\n📥 加载模型默认实例: {model_id}")
        
        result = await self._api_call("POST", "/api/pool/load-model", {
            "model_id": model_id,
            "n_layers": 24
        })
        
        await self.print_result("加载默认实例", result)
        
        return result.get("success", False)
    
    async def test_2_load_named_instances(self):
        """测试2：加载多个命名实例"""
        await self.print_section("测试2: 加载多个命名实例")
        
        model_id = "qwen3-0.6b"
        instances = ["worker-A", "worker-B"]
        
        all_success = True
        
        for instance_id in instances:
            print(f"\n📥 加载实例: {model_id}::{instance_id}")
            
            result = await self._api_call("POST", "/api/pool/load-model", {
                "model_id": model_id,
                "instance_id": instance_id,
                "n_layers": 24,
                "target_nodes": None  # 使用所有可用节点
            })
            
            await self.print_result(f"加载实例 {instance_id}", result)
            
            if not result.get("success"):
                all_success = False
        
        return all_success
    
    async def test_3_auto_generate_instances(self):
        """测试3：使用自动生成实例ID"""
        await self.print_section("测试3: 自动生成实例ID")
        
        model_id = "qwen3-0.6b"
        
        results = []
        
        for i in range(2):  # 创建2个自动命名的实例
            print(f"\n📥 自动生成实例 #{i+1}: {model_id}")
            
            result = await self._api_call("POST", "/api/pool/load-model", {
                "model_id": model_id,
                "auto_instance": True,
                "n_layers": 24
            })
            
            await self.print_result(f"自动生成实例 #{i+1}", result)
            
            if result.get("success"):
                results.append(result.get("instance_id"))
        
        print(f"\n📋 生成的实例IDs: {results}")
        
        return len(results) == 2
    
    async def test_4_query_instances(self):
        """测试4：查询实例信息"""
        await self.print_section("测试4: 查询实例信息")
        
        model_id = "qwen3-0.6b"
        
        print(f"\n🔍 查询模型的所有实例: {model_id}")
        
        result = await self._api_call("GET", f"/api/models/instances/{model_id}")
        
        if result.get("success"):
            data = result.get("data", {})
            instances = data.get("instances", [])
            total = data.get("total_instances", 0)
            
            print(f"  ✅ 找到 {total} 个实例:")
            
            for idx, inst in enumerate(instances, 1):
                print(f"    [{idx}] 实例ID: {inst.get('instance_id')}")
                print(f"        完整ID: {inst.get('full_model_id')}")
                print(f"        所在节点: {inst.get('node_id')}")
                print(f"        分片: L{inst.get('shard', {}).get('start_layer', '?')}-L{inst.get('shard', {}).get('end_layer', '?')}")
        else:
            print(f"  ❌ 查询失败: {result.get('error')}")
        
        print(f"\n🔍 查询所有模型的实例摘要:")
        
        summary_result = await self._api_call("GET", "/api/models/instances")
        
        if summary_result.get("success"):
            data = summary_result.get("data", {})
            summary = data.get("summary", {})
            total_models = data.get("total_models", 0)
            total_instances = data.get("total_instances", 0)
            
            print(f"  ✅ 共 {total_models} 个模型, {total_instances} 个实例")
            
            for mid, count in summary.items():
                print(f"    - {mid}: {count} 个实例")
        
        return result.get("success", False)
    
    async def test_5_unload_specific_instance(self):
        """测试5：卸载指定实例"""
        await self.print_section("测试5: 卸载指定实例")
        
        model_id = "qwen3-0.6b"
        instance_id = "worker-A"
        
        print(f"\n🗑️ 卸载指定实例: {model_id}::{instance_id}")
        
        result = await self._api_call("POST", "/api/pool/unload-model", {
            "model_id": model_id,
            "instance_id": instance_id
        })
        
        await self.print_result(f"卸载实例 {instance_id}", result)
        
        return result.get("success", False)
    
    async def test_6_unload_all_instances(self):
        """测试6：卸载所有实例"""
        await self.print_section("测试6: 卸载所有实例")
        
        model_id = "qwen3-0.6b"
        
        print(f"\n🗑️ 卸载模型的所有实例: {model_id}")
        
        result = await self._api_call("POST", "/api/pool/unload-model", {
            "model_id": model_id,
            "unload_all_instances": True
        })
        
        await self.print_result("卸载所有实例", result)
        
        return result.get("success", False)


async def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("  EXO Manager 多实例功能测试")
    print("  Multi-Instance Model Loading Test Suite")
    print("=" * 60)
    print(f"\n⏰ 测试开始时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 服务地址: http://localhost:8080")
    
    tester = MultiInstanceTester()
    
    try:
        async with tester:
            results = {}
            
            # 运行各项测试
            results["test_1"] = await tester.test_1_load_default_instance()
            results["test_2"] = await tester.test_2_load_named_instances()
            results["test_3"] = await tester.test_3_auto_generate_instances()
            results["test_4"] = await tester.test_4_query_instances()
            results["test_5"] = await tester.test_5_unload_specific_instance()
            results["test_6"] = await tester.test_6_unload_all_instances()
            
            # 打印汇总
            await tester.print_section("测试结果汇总")
            
            passed = sum(1 for v in results.values() if v)
            total = len(results)
            
            print(f"\n📊 测试结果: {passed}/{total} 通过\n")
            
            for test_name, success in results.items():
                status = "✅ PASS" if success else "❌ FAIL"
                test_desc = {
                    "test_1": "加载默认实例",
                    "test_2": "加载命名实例",
                    "test_3": "自动生成实例",
                    "test_4": "查询实例信息",
                    "test_5": "卸载指定实例",
                    "test_6": "卸载所有实例"
                }.get(test_name, test_name)
                
                print(f"  {status} - {test_desc}")
            
            print(f"\n{'='*60}")
            
            if passed == total:
                print("  🎉 所有测试通过！多实例功能正常工作。")
                return 0
            else:
                print(f"  ⚠️  {total - passed} 个测试失败，请检查日志。")
                return 1
                
    except Exception as e:
        print(f"\n❌ 测试执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  准备运行多实例测试...")
    print("=" * 60)
    print("\n⚠️  确保已启动 exo_manager 服务:")
    print("   cd f:\\exoProject\\exo_manager")
    print("   python start_manager.py")
    print("\n⚠️  确保有至少一个 EXO 节点在线")
    print("\n按 Ctrl+C 取消测试...\n")
    
    try:
        exit_code = asyncio.run(run_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户取消测试")
        sys.exit(130)


if __name__ == "__main__":
    main()