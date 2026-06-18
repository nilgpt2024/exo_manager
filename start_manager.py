#!/usr/bin/env python3
"""
EXO Cluster Manager - 启动脚本 (增强版)
========================================

支持多种启动模式：
1. 零配置启动 - 最简单，适合新手
2. 配置文件模式 - 适合生产环境
3. 自动扫描模式 - 自动发现局域网节点

使用方式:
    # 模式1: 零配置启动 (推荐)
    python start_manager.py
    
    # 模式2: 自定义端口
    python start_manager.py --port 9000
    
    # 模式3: 使用配置文件批量加载节点
    python start_manager.py --config ../network_config.json
    
    # 模式4: 自动扫描局域网 (实验性)
    python start_manager.py --auto-discover --scan-range 192.168.1.0/24

访问地址:
    Web界面: http://localhost:8080
    API文档: http://localhost:8080/docs
"""

import sys
import os
import argparse
import socket
import asyncio
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner(host, port, mode, frp_info=None):
    """打印启动横幅"""
    display_host = host if host != '0.0.0.0' else 'localhost'

    mode_emoji = {
        "zero-config": "🚀",
        "config-file": "📋",
        "auto-discover": "🔍"
    }.get(mode, "🚀")

    mode_desc = {
        "zero-config": "零配置模式（通过Web界面添加节点）",
        "config-file": "配置文件模式（自动加载预定义节点）",
        "auto-discover": "自动发现模式（扫描局域网）"
    }.get(mode, "")

    # FRP 信息行
    frp_lines = ""
    if frp_info:
        token_preview = frp_info.get("token", "")
        bind_port = frp_info.get("bind_port", 7000)
        preview = token_preview[:8] + "..." + token_preview[-4:] if len(token_preview) > 12 else token_preview
        frp_lines = f"""
║   🌐 FRP Server                                           ║
║      • 监听端口: {bind_port:<44}
║      • Token: {preview:<50}
║      • 节点连接: 用户登录后在「节点连接」页面获取启动命令          ║"""

    print(f"""
╔═════════════════════════════════════════════════════════╗
║                                                           ║
║   {mode_emoji} EXO Cluster Manager v1.0.0                   ║
║   ─────────────────────────────────                       ║
║   分布式AI模型推理集群管理系统                               ║
║                                                           ║
╠═════════════════════════════════════════════════════════╣
║                                                           ║
║   📍 服务地址                                              ║
║      • Web界面: http://{display_host}:{port}
║      • API文档: http://{display_host}:{port}/docs
║      • WebSocket: ws://{display_host}:{port}/ws/cluster
║                                                           ║
║   ⚙️ 启动模式: {mode_desc:<44}
{frp_lines}
╚═════════════════════════════════════════════════════════╝
""".format(display_host=display_host, port=port, mode=mode_desc))


def main():
    parser = argparse.ArgumentParser(
        description="EXO Cluster Manager - 分布式AI集群管理控制台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 零配置启动（最简单）
  %(prog)s
  
  # 自定义端口
  %(prog)s --port 9000
  
  # 从配置文件加载节点
  %(prog)s --config ../network_config.json
  
  # 自动扫描局域网中的EXO节点 (实验性)
  %(prog)s --auto-discover --scan-range 192.168.1.0/24

更多信息请访问: https://github.com/exo-labs/exo
        """
    )
    
    # 基本参数
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="监听地址 (默认: 0.0.0.0，允许所有IP访问)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080,
        help="服务端口号 (默认: 8080)"
    )
    
    # 模式选择参数
    config_group = parser.add_mutually_exclusive_group()
    
    config_group.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="网络配置文件路径 (可选，用于批量加载已知节点)"
    )
    
    config_group.add_argument(
        "--auto-discover",
        action="store_true",
        help="启用自动发现模式 (扫描局域网寻找EXO节点)"
    )
    
    # 自动发现相关参数
    parser.add_argument(
        "--scan-range",
        type=str,
        default="192.168.1.0/24",
        help="自动发现的IP范围 (默认: 192.168.1.0/24，仅--auto-discover时生效)"
    )
    
    parser.add_argument(
        "--scan-port",
        type=int,
        default=50051,
        help="EXO节点的gRPC端口 (默认: 50051)"
    )
    
    parser.add_argument(
        "--scan-timeout",
        type=float,
        default=2.0,
        help="每个IP的连接超时秒数 (默认: 2.0)"
    )
    
    # 其他参数
    parser.add_argument(
        "--reload",
        action="store_true",
        help="开发模式：启用代码热重载"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="工作进程数 (默认: 1)"
    )

    # ==================== FRP Server 相关参数 ====================
    frp_group = parser.add_argument_group("FRP Server (内网穿透服务端)")

    frp_group.add_argument(
        "--frp-enable",
        action="store_true",
        help="启用 FRP Server 服务 (启动时自动运行 frps)"
    )

    frp_group.add_argument(
        "--frp-bind-port",
        type=int,
        default=7000,
        help="frps 监听端口 (默认: 7000)"
    )

    frp_group.add_argument(
        "--frp-token",
        type=str,
        default=None,
        help="frps 认证 Token (不指定则自动随机生成 32 位 hex)"
    )

    frp_group.add_argument(
        "--frp-dashboard-port",
        type=int,
        default=0,
        help="frps Dashboard 端口 (0=不启用, 默认: 0)"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    
    args = parser.parse_args()

    # ==================== 从环境变量补充 FRP 配置 (Docker 场景) ====================
    # 当未通过命令行指定时，从环境变量读取（docker-compose 传入）
    if not args.frp_enable and os.environ.get('EXO_FRP_ENABLE', '').lower() in ('true', '1', 'yes'):
        args.frp_enable = True
    if os.environ.get('EXO_FRP_BIND_PORT'):
        args.frp_bind_port = int(os.environ['EXO_FRP_BIND_PORT'])
    if not args.frp_token and os.environ.get('EXO_FRP_TOKEN'):
        args.frp_token = os.environ['EXO_FRP_TOKEN']
    if os.environ.get('EXO_FRP_DASHBOARD_PORT'):
        args.frp_dashboard_port = int(os.environ['EXO_FRP_DASHBOARD_PORT'])

    # ==================== FRP Token 生成 (在横幅打印前) ====================
    frp_info = None
    if args.frp_enable:
        if not args.frp_token:
            # 自动随机生成 32 位 hex token
            import secrets
            args.frp_token = secrets.token_hex(16)  # 32 字符
        frp_info = {
            "token": args.frp_token,
            "bind_port": args.frp_bind_port,
        }
        print(f"  🔑 FRP Token 已自动生成: {args.frp_token}")
        print(f"     ⚠️  请保存此 Token，节点连接时需要使用\n")

    # 确定启动模式
    if args.config:
        startup_mode = "config-file"
        if not os.path.exists(args.config):
            print(f"❌ 错误: 配置文件不存在: {args.config}")
            print("\n提示:")
            print("  1. 不使用 --config 参数进行零配置启动")
            print("  2. 或检查配置文件路径是否正确")
            sys.exit(1)
    elif args.auto_discover:
        startup_mode = "auto-discover"
    else:
        startup_mode = "zero-config"
    
    # 打印启动信息（含 FRP 信息）
    print_banner(args.host, args.port, startup_mode, frp_info=frp_info)
    
    # 设置环境变量供server使用
    os.environ['EXO_MANAGER_HOST'] = args.host
    os.environ['EXO_MANAGER_PORT'] = str(args.port)
    os.environ['EXO_MANAGER_CONFIG'] = args.config or ''
    os.environ['EXO_MANAGER_MODE'] = startup_mode
    os.environ['EXO_MANAGER_LOG_LEVEL'] = args.log_level

    # FRP Server 环境变量（供 server.py startup 使用）
    os.environ['EXO_FRP_ENABLE'] = str(args.frp_enable).lower()
    os.environ['EXO_FRP_BIND_PORT'] = str(args.frp_bind_port)
    if args.frp_token:
        os.environ['EXO_FRP_TOKEN'] = args.frp_token
    if args.frp_dashboard_port and args.frp_dashboard_port > 0:
        os.environ['EXO_FRP_DASHBOARD_PORT'] = str(args.frp_dashboard_port)
    
    # 如果是自动发现模式，先执行扫描
    if args.auto_discover:
        print(f"\n🔍 正在扫描局域网 ({args.scan_range}) 寻找EXO节点...")
        print("   这可能需要几秒钟...\n")
        
        discovered_nodes = asyncio.run(scan_for_nodes(
            scan_range=args.scan_range,
            port=args.scan_port,
            timeout=args.scan_timeout
        ))
        
        if discovered_nodes:
            print(f"✅ 发现 {len(discovered_nodes)} 个EXO节点:\n")
            for i, node in enumerate(discovered_nodes, 1):
                print(f"   {i}. {node['address']}:{node['port']}")
            
            print("\n💡 这些节点将在Manager启动后自动添加")
            print("   你也可以在Web界面中管理这些节点\n")
            
            # 将发现的节点保存到临时环境变量
            os.environ['DISCOVERED_NODES'] = str(discovered_nodes)
        else:
            print("⚠️  未发现任何EXO节点")
            print("   提示: 确保EXO节点正在运行且网络可达\n")
    
    try:
        import uvicorn
        
        uvicorn.run(
            "server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level=args.log_level.lower()
        )
        
    except KeyboardInterrupt:
        print("\n\n✅ EXO Cluster Manager 已停止")
    except ImportError as e:
        print(f"\n❌ 缺少依赖包: {e}")
        print("\n请先安装依赖:")
        print("   pip install -r requirements.txt")
        print("\n或直接安装:")
        print("   pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def scan_for_nodes(scan_range: str = "192.168.1.0/24", 
                         port: int = 50051, 
                         timeout: float = 2.0) -> list:
    """
    扫描局域网中运行中的EXO节点
    
    Args:
        scan_range: CIDR格式的IP范围 (如 "192.168.1.0/24")
        port: 要扫描的端口号 (默认50051是EXO的gRPC端口)
        timeout: 每个连接的超时时间(秒)
        
    Returns:
        发现的节点列表 [{"address": "...", "port": ...}, ...]
    """
    import ipaddress
    
    discovered = []
    
    try:
        network = ipaddress.ip_network(scan_range, strict=False)
    except ValueError as e:
        print(f"❌ 无效的IP范围格式: {e}")
        return []
    
    # 限制扫描数量，避免太慢
    hosts = list(network.hosts())
    max_scan = 254  # /24 网络最多254个主机
    
    if len(hosts) > max_scan:
        print(f"⚠️  IP范围过大 ({len(hosts)} 个地址)，限制扫描前 {max_scan} 个")
        hosts = hosts[:max_scan]
    
    async def check_host(ip_str):
        """检查单个主机是否有EXO节点"""
        try:
            future = asyncio.open_connection(ip_str, port)
            reader, writer = await asyncio.wait_for(future, timeout=timeout)
            writer.close()
            return {"address": ip_str, "port": port}
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            return None
    
    tasks = [check_host(str(ip)) for ip in hosts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if result and isinstance(result, dict):
            discovered.append(result)
    
    return discovered


if __name__ == "__main__":
    main()
