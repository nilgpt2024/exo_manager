#!/usr/bin/env python3
"""
多实例模型加载 - 使用示例
========================

展示如何使用 exo_manager 的多实例功能来：
1. 加载两个相同的 qwen3-0.6b 模型实例
2. 实现负载均衡和并行推理
3. 管理多个推理工作器

使用方式:
    python example_multi_instance_qwen.py
    
API 端点:
    POST /api/pool/load-model     - 加载模型（支持 instance_id）
    GET  /api/models/instances     - 查询所有实例
    POST /api/pool/unload-model    - 卸载模型（支持指定实例）
"""

import asyncio
import aiohttp
import json


class QwenMultiInstanceDemo:
    """Qwen3-0.6B 多实例演示"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def api_call(self, method: str, endpoint: str, data: dict = None) -> dict:
        """调用 API"""
        url = f"{self.base_url}{endpoint}"
        
        if method == "GET":
            async with self.session.get(url) as resp:
                return await resp.json()
        else:
            async with self.session.post(url, json=data) as resp:
                return await resp.json()


async def main():
    """主演示流程"""
    print("=" * 70)
    print("  🚀 EXO Manager 多实例演示 - 加载两个 Qwen3-0.6B 模型")
    print("=" * 70)
    
    demo = QwenMultiInstanceDemo()
    
    try:
        async with demo:
            
            # ========================================
            # 场景1：加载两个相同的 qwen3-0.6b 实例
            # ========================================
            print("\n📦 [场景1] 加载两个 Qwen3-0.6B 模型实例")
            print("-" * 70)
            
            # 第一个实例：worker-1
            print("\n1️⃣  加载第一个实例 (worker-1)...")
            result1 = await demo.api_call("POST", "/api/pool/load-model", {
                "model_id": "qwen3-0.6b",
                "instance_id": "worker-1",
                "n_layers": 24,
                "strategy": "memory_weighted"
            })
            
            if result1.get("success"):
                print(f"   ✅ worker-1 加载成功!")
                print(f"      完整ID: {result1.get('full_model_id')}")
                print(f"      分配节点: {result1.get('summary', {}).get('success_nodes', 0)} 个")
            else:
                print(f"   ❌ 加载失败: {result1.get('error')}")
                return
            
            # 第二个实例：worker-2
            print("\n2️⃣  加载第二个实例 (worker-2)...")
            result2 = await demo.api_call("POST", "/api/pool/load-model", {
                "model_id": "qwen3-0.6b",
                "instance_id": "worker-2",
                "n_layers": 24,
                "strategy": "memory_weighted"
            })
            
            if result2.get("success"):
                print(f"   ✅ worker-2 加载成功!")
                print(f"      完整ID: {result2.get('full_model_id')}")
                print(f"      分配节点: {result2.get('summary', {}).get('success_nodes', 0)} 个")
            else:
                print(f"   ❌ 加载失败: {result2.get('error')}")
                return
            
            # ========================================
            # 场景2：查询已加载的实例
            # ========================================
            print("\n\n🔍 [场景2] 查询所有已加载的实例")
            print("-" * 70)
            
            instances_result = await demo.api_call("GET", "/api/models/instances/qwen3-0.6b")
            
            if instances_result.get("success"):
                instances = instances_result.get("data", {}).get("instances", [])
                total = instances_result.get("data", {}).get("total_instances", 0)
                
                print(f"\n   📊 模型 qwen3-0.6b 共有 {total} 个实例:\n")
                
                for idx, inst in enumerate(instances, 1):
                    print(f"   [{idx}] 📌 实例: {inst.get('instance_id')}")
                    print(f"       ├─ 完整ID: {inst.get('full_model_id')}")
                    print(f"       ├─ 所在节点: {inst.get('node_id')}")
                    shard = inst.get('shard', {})
                    print(f"       └─ 分片范围: L{shard.get('start_layer', '?')} ~ L{shard.get('end_layer', '?')}")
                    print()
            
            # ========================================
            # 场景3：查看全局实例摘要
            # ========================================
            print("📋 [场景3] 全局实例摘要")
            print("-" * 70)
            
            summary_result = await demo.api_call("GET", "/api/models/instances")
            
            if summary_result.get("success"):
                data = summary_result.get("data", {})
                print(f"\n   总计:")
                print(f"   - 模型种类: {data.get('total_models', 0)}")
                print(f"   - 实例总数: {data.get('total_instances', 0)}")
                print(f"\n   详细信息:")
                
                for model_id, count in data.get("summary", {}).items():
                    bar = "█" * count
                    print(f"   • {model_id}: {count} 个实例 {bar}")
            
            # ========================================
            # 场景4：使用示例（模拟推理请求）
            # ========================================
            print("\n\n💡 [场景4] 使用建议")
            print("-" * 70)
            
            print("""
   ✨ 现在你有两个独立的 Qwen3-0.6B 推理引擎实例！
   
   使用方式:
   
   1️⃣  负载均衡:
      将用户请求轮询分发到 worker-1 和 worker-2
      
   2️⃣  A/B 测试:
      worker-1 使用默认参数
      worker-2 使用不同的 temperature/top_p
      
   3️⃣  并行处理:
      同时处理多个独立请求，吞吐量翻倍
   
   4️⃣  故障隔离:
      一个实例崩溃不影响另一个
   
   API 调用示例 (使用 OpenAI 兼容接口):
   
   # 调用 worker-1
   curl http://localhost:8080/v1/chat/completions \\
     -H "Content-Type: application/json" \\
     -d '{
       "model": "qwen3-0.6b::worker-1",
       "messages": [{"role": "user", "content": "Hello"}]
     }'
   
   # 调用 worker-2
   curl http://localhost:8080/v1/chat/completions \\
     -H "Content-Type: application/json" \\
     -d '{
       "model": "qwen3-0.6b::worker-2",
       "messages": [{"role": "user", "content": "Hello"}]
     }'
            """)
            
            # ========================================
            # 清理选项
            # ========================================
            print("\n🧹 [清理] 是否卸载实例？")
            print("-" * 70)
            
            print("\n   可选操作:")
            print("   1. 只卸载 worker-1")
            print("   2. 只卸载 worker-2")
            print("   3. 卸载所有实例")
            print("   4. 保留所有实例（用于后续使用）")
            print("\n   💡 提示: 可以通过 API 手动执行这些操作")
            print("      或运行 test_multi_instance.py 进行完整测试")
            
            print("\n" + "=" * 70)
            print("  ✅ 演示完成！多实例功能正常工作。")
            print("=" * 70)
    
    except Exception as e:
        print(f"\n❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n⏳ 正在启动多实例演示...\n")
    asyncio.run(main())