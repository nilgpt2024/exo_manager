"""
FRP Server 管理器 (frps)
========================

管理 FRP Server 的完整生命周期：
- 配置生成与管理
- 进程启停与监控
- 客户端注册与连接信息分发
- 用户连接参数自动生成（含 token / secretKey）

数据流:
  start_manager.py --frp-enable → 随机生成 token
    ↓
  server.py startup → update_config(token) → _save_config() 持久化
    ↓
  GET /api/frps/user-connection → get_user_connection_info()
    ↓
  user.html「节点连接」页面 → 展示含 token 的启动命令
"""

import os
import sys
import time
import hashlib
import subprocess
import logging
import platform
import threading
import json
import asyncio
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum

# 使用本地的 FRP 下载工具（独立于 exo 主项目）
from frp_helper import (
    FRP_VERSION,
    FRP_BASE_URL,
    download_and_install_frp,
    check_frpc_available,
    ensure_frpc_installed,
    get_system_info,
)

logger = logging.getLogger("FRPServerManager")


# ==================== 数据模型 ====================

class FRPStatus(Enum):
    """FRP 服务状态"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class FRPSConfig:
    """frps 配置"""
    bind_port: int = 7000
    vhost_http_port: Optional[int] = None
    vhost_https_port: Optional[int] = None
    dashboard_port: Optional[int] = None
    dashboard_user: Optional[str] = None
    dashboard_pwd: Optional[str] = None
    token: Optional[str] = None
    enable_xtcp: bool = True
    log_level: str = "info"
    log_max_days: int = 3

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConnectedClient:
    """已连接的客户端"""
    node_id: str
    node_name: str = ""
    remote_port: int = 0
    local_port: int = 50051
    chatgpt_remote_port: int = 0
    chatgpt_local_port: int = 52415
    status: str = "offline"  # online, offline, error
    connected_at: float = 0.0
    last_heartbeat: float = 0.0
    enable_p2p: bool = True
    secret_key: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# ==================== 核心管理器 ====================

class FRPServerManager:
    """
    FRP Server 管理器

    职责:
    1. 管理 frps 进程生命周期
    2. 持久化配置到 ~/.exo/frp/frps.toml
    3. 为每个用户生成专属的连接参数（token + secretKey + 启动命令）
    """

    def __init__(self, config_dir: Optional[Path] = None):
        # 目录设置
        self.config_dir = config_dir or (Path.home() / ".exo" / "frp")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 二进制目录
        self.bin_dir = self.config_dir
        self.frps_path = self.bin_dir / ("frps.exe" if platform.system().lower() == "windows" else "frps")
        self.frpc_path = self.bin_dir / ("frpc.exe" if platform.system().lower() == "windows" else "frpc")

        # 配置
        self.config: FRPSConfig = FRPSConfig()
        self.config_path: Path = self.config_dir / "frps.json"

        # 进程管理
        self.process: Optional[subprocess.Popen] = None
        self.status: FRPStatus = FRPStatus.STOPPED
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stopped = False

        # 客户端管理
        self.connected_clients: Dict[str, ConnectedClient] = {}
        self.client_configs: Dict[str, Dict] = {}  # node_id -> frpc_config

        # 统计
        self.start_time: Optional[float] = None
        self.total_connections: int = 0
        self._stats_lock = threading.Lock()

        # 加载已有配置
        self._load_config()

    # ==================== 配置管理 ====================

    # JSON camelCase → Python snake_case 字段映射（frps 配置用驼峰命名）
    _JSON_KEY_MAP = {
        "bindPort": "bind_port",
        "vhostHttpPort": "vhost_http_port",
        "vhostHttpsPort": "vhost_https_port",
        "dashboardPort": "dashboard_port",
        "dashboardUser": "dashboard_user",
        "dashboardPwd": "dashboard_pwd",
        "enableXtcp": "enable_xtcp",
        "logLevel": "log_level",
        "logMaxDays": "log_max_days",
    }

    def _load_config(self):
        """从文件加载配置（JSON 格式，自动处理 camelCase ↔ snake_case 映射）"""
        try:
            if not self.config_path.exists():
                logger.info(f"[FRPServer] 配置文件不存在，使用默认配置: {self.config_path}")
                return

            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for key, value in data.items():
                # 优先精确匹配，否则尝试 camelCase → snake_case 映射
                attr_name = self._JSON_KEY_MAP.get(key, key)
                if hasattr(self.config, attr_name):
                    setattr(self.config, attr_name, value)

            logger.info(f"[FRPServer] 配置已加载: {self.config_path}")
        except Exception as e:
            logger.error(f"[FRPServer] 加载配置失败: {e}")

    def _save_config(self):
        """保存配置到文件（JSON 格式）"""
        try:
            config_data = self.config.to_dict()
            # 移除 None 值，保持配置干净
            clean_data = {k: v for k, v in config_data.items() if v is not None}
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"[FRPServer] 配置已保存: {self.config_path}")
        except Exception as e:
            logger.error(f"[FRPServer] 保存配置失败: {e}")

    def update_config(self, **kwargs) -> FRPSConfig:
        """更新 frps 配置并持久化"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            self._save_config()
        return self.config

    def generate_frps_config_content(self) -> str:
        """生成 frps 配置文件内容（JSON 格式，frps 原生支持）"""
        c = self.config
        config: Dict[str, Any] = {
            "bindPort": c.bind_port,
        }

        # XTCP P2P 必需：传输层加密和压缩
        if c.enable_xtcp:
            config["transport"] = {
                "useEncryption": True,
                "useCompression": True,
            }

        # Dashboard
        if c.dashboard_port and c.dashboard_port > 0:
            config["webServer"] = {
                "addr": "0.0.0.0",
                "port": c.dashboard_port,
            }
            if c.dashboard_user:
                config["webServer"]["user"] = c.dashboard_user
            if c.dashboard_pwd:
                config["webServer"]["password"] = c.dashboard_pwd

        # 认证 Token
        if c.token:
            config["auth"] = {"token": c.token}

        # 日志（路径用正斜杠，避免 JSON 转义问题）
        log_path = str(self.config_dir / "frps.log").replace("\\", "/")
        config["log"] = {
            "to": log_path,
            "level": c.log_level,
            "maxDays": c.log_max_days,
            "disablePrintColor": False,
        }

        return json.dumps(config, indent=2, ensure_ascii=False) + "\n"

    # ==================== FRP 安装 ====================

    def check_frps_installed(self) -> bool:
        """检查 frps 是否已安装"""
        if not self.frps_path.exists():
            return False
        try:
            result = subprocess.run(
                [str(self.frps_path), "--version"],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    async def ensure_frps_installed(self) -> bool:
        """确保 frps 已安装"""
        if self.check_frps_installed():
            return True
        logger.info("[FRPServer] 正在安装 FRP...")
        success = await asyncio.get_event_loop().run_in_executor(None, download_and_install_frp)
        return success

    # ==================== 进程管理 ====================

    @staticmethod
    def _is_port_in_use(port: int) -> bool:
        """检查端口是否被占用（使用 netstat，兼容 Windows SO_REUSEADDR）"""
        try:
            if platform.system().lower() == "windows":
                result = subprocess.run(
                    ['netstat', '-ano'],
                    capture_output=True, text=True, timeout=5
                )
                return f':{port}' in result.stdout and 'LISTENING' in result.stdout
            else:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(('0.0.0.0', port))
                        return False
                    except OSError:
                        return True
        except Exception:
            return False

    @staticmethod
    def _kill_process_on_port(port: int):
        """杀掉占用指定端口的进程（Windows / Linux 通用）"""
        import socket
        try:
            if platform.system().lower() == "windows":
                # Windows: 用 netstat 找到 PID，然后 taskkill
                result = subprocess.run(
                    ['netstat', '-ano'],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.splitlines():
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        pid = parts[-1] if parts else None
                        if pid and pid.isdigit():
                            logger.info(f"[FRPServer] 杀掉占用端口的进程 PID: {pid}")
                            subprocess.run(
                                ['taskkill', '/F', '/PID', pid],
                                capture_output=True, timeout=5
                            )
            else:
                # Linux/Mac: 用 lsof 或 fuser
                for cmd in [['fuser', '-k', f'{port}/tcp'], ['lsof', '-ti', f':{port}']]:
                    try:
                        subprocess.run(cmd, capture_output=True, timeout=5)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"[FRPServer] 清理端口 {port} 失败: {e}")

    async def start_server(self) -> tuple:
        """
        启动 frps 服务

        Returns:
            (success: bool, message: str)
        """
        with self._lock:
            # 先检查是否已有 frps 在运行（包括非我们管理的）
            if self._is_port_in_use(self.config.bind_port):
                logger.warning(f"[FRPServer] 端口 {self.config.bind_port} 被占用，尝试清理旧进程...")
                self._kill_process_on_port(self.config.bind_port)
                await asyncio.sleep(0.5)  # 等待端口释放

            if self.status == FRPStatus.RUNNING and self.is_running():
                return True, "frps 已经在运行中"

            self.status = FRPStatus.STARTING

            # 确保 frps 已安装
            if not await self.ensure_frps_installed():
                self.status = FRPStatus.ERROR
                return False, "FRP 安装失败，请手动安装或检查网络"

            # 生成配置文件（JSON 格式）
            json_content = self.generate_frps_config_content()
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(json_content)
            logger.info(f"[FRPServer] 配置已写入: {self.config_path}")

            # 确保日志文件存在（frps 启动时需要）
            log_file = self.config_dir / "frps.log"
            if not log_file.exists():
                log_file.touch()
                logger.debug(f"[FRPServer] 日志文件已创建: {log_file}")

            # 启动进程
            cmd = [str(self.frps_path), "-c", str(self.config_path)]
            logger.info(f"[FRPServer] 启动命令: {' '.join(cmd)}")

            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system().lower() == "windows" else 0
                )

                # 等待一小段时间验证进程是否存活（端口冲突等会导致立即退出）
                await asyncio.get_event_loop().run_in_executor(None, time.sleep, 1.0)

                poll_result = self.process.poll()
                if poll_result is not None:
                    # 进程已退出，读取错误输出
                    err_output = ""
                    if self.process.stdout:
                        try:
                            err_output = self.process.stdout.read().decode('utf-8', errors='replace').strip()
                        except Exception:
                            pass
                    self.process = None
                    self.status = FRPStatus.ERROR
                    err_msg = f"frps 启动后立即退出 (exit code: {poll_result})"
                    if err_output:
                        err_msg += f", 错误: {err_output[:200]}"
                    logger.error(f"[FRPServer] {err_msg}")
                    return False, err_msg

                self.status = FRPStatus.RUNNING
                self.start_time = time.time()
                self._stopped = False

                # 启动输出监控线程
                self._monitor_thread = threading.Thread(target=self._monitor_output, daemon=True)
                self._monitor_thread.start()

                logger.info(f"[FRPServer] frps 已启动 (PID: {self.process.pid})")
                return True, f"frps 已成功启动 (PID: {self.process.pid}, 端口: {self.config.bind_port})"

            except Exception as e:
                self.status = FRPStatus.ERROR
                logger.error(f"[FRPServer] 启动失败: {e}")
                return False, f"启动失败: {str(e)}"

    async def stop_server(self) -> tuple:
        """停止 frps 服务"""
        with self._lock:
            if self.status != FRPStatus.RUNNING or not self.process:
                self.status = FRPStatus.STOPPED
                return True, "frps 未在运行"

            self.status = FRPStatus.STOPPING
            self._stopped = True

            try:
                self.process.terminate()
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self.process.wait(timeout=5)
                    )
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

                self.process = None
                self.status = FRPStatus.STOPPED
                self.start_time = None
                logger.info("[FRPServer] frps 已停止")
                return True, "frps 已停止"

            except Exception as e:
                self.status = FRPStatus.ERROR
                logger.error(f"[FRPServer] 停止失败: {e}")
                return False, f"停止失败: {str(e)}"

    def is_running(self) -> bool:
        """检查进程是否运行中（进程存活 + 端口监听双重验证）"""
        # 1. 检查进程对象
        if self.process:
            try:
                poll_result = self.process.poll()
                if poll_result is None:
                    return True  # 进程还活着
            except Exception:
                pass

        # 2. 兜底：端口在监听 且 有 frps.exe 进程
        if not self._is_port_in_use(self.config.bind_port):
            return False

        try:
            result = subprocess.run(
                ['tasklist', '/FI', 'IMAGENAME eq frps.exe', '/NH'],
                capture_output=True, text=True, timeout=5
            )
            return 'frps.exe' in result.stdout
        except Exception:
            return False

    def _monitor_output(self):
        """监控 frps 输出日志"""
        if not self.process or not self.process.stdout:
            return

        try:
            for line in iter(self.process.stdout.readline, b''):
                if self._stopped:
                    break
                line_str = line.decode('utf-8', errors='ignore').strip()
                if not line_str:
                    continue

                if "[INFO]" in line_str or "[info]" in line_str:
                    logger.info(f"[FRPS] {line_str}")
                elif "[WARN]" in line_str or "[warn]" in line_str:
                    logger.warning(f"[FRPS] {line_str}")
                elif "[ERROR]" in line_str or "[error]" in line_str:
                    logger.error(f"[FRPS] {line_str}")
                else:
                    logger.debug(f"[FRPS] {line_str}")
        except Exception as e:
            if not self._stopped:
                logger.debug(f"[FRPServer] 监控输出异常: {e}")

        # 进程退出处理
        if not self._stopped and self.process:
            exit_code = self.process.poll()
            if exit_code is not None:
                logger.warning(f"[FRPServer] frps 进程退出，退出码: {exit_code}")
                self.status = FRPStatus.ERROR

    def _handle_proxy_register(self, log_line: str):
        """处理代理注册日志"""
        try:
            import re
            match = re.search(r'\[(\w+)\].*?proxy.*?registered', log_line, re.IGNORECASE)
            if match:
                proxy_name = match.group(1)
                logger.info(f"[FRPServer] 新代理注册: {proxy_name}")
                with self._stats_lock:
                    self.total_connections += 1
        except Exception:
            pass

    # ==================== 远程端口计算 ====================

    @staticmethod
    def calculate_remote_port(node_id: str, service: str = "") -> int:
        """根据 node_id 哈希计算远程端口 (40000-50000)

        Args:
            node_id: 节点 ID
            service: 服务标识，用于区分同一节点的不同服务（如 gRPC / chatgpt）
        """
        hash_input = f"{node_id}:{service}" if service else node_id
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        return 40000 + (hash_val % 10000)

    # ==================== 客户端配置生成 ====================

    def register_client(
        self,
        node_id: str,
        local_port: int = 50051,
        chatgpt_local_port: int = 52415,
        enable_p2p: bool = True,
        node_name: str = "",
    ) -> Dict[str, Any]:
        """注册客户端并生成 frpc 配置"""
        remote_port = self.calculate_remote_port(node_id)
        chatgpt_remote_port = self.calculate_remote_port(node_id, service="chatgpt")
        token = self.config.token
        secret_key = hashlib.sha256(token.encode('utf-8')).hexdigest()[:16] if token else ""

        config = {
            "serverAddr": "<FRPS_SERVER_IP>",
            "serverPort": self.config.bind_port,
            "proxies": [],
        }
        if token:
            config["auth"] = {"token": token}

        if enable_p2p:
            config["proxies"].append({
                "name": f"exo_p2p_{node_id}",
                "type": "xtcp",
                "secretKey": secret_key,
                "localIP": "127.0.0.1",
                "localPort": local_port,
            })

        # gRPC / exo 节点间通信端口
        config["proxies"].append({
            "name": f"exo_tcp_{node_id}",
            "type": "tcp",
            "localIP": "127.0.0.1",
            "localPort": local_port,
            "remotePort": remote_port,
        })

        # ChatGPT API HTTP 端口（manager 代理 chat 请求用）
        config["proxies"].append({
            "name": f"exo_chatgpt_{node_id}",
            "type": "tcp",
            "localIP": "127.0.0.1",
            "localPort": chatgpt_local_port,
            "remotePort": chatgpt_remote_port,
        })

        self.client_configs[node_id] = {
            **config,
            "_meta": {
                "generated_at": time.time(),
                "remote_port": remote_port,
                "chatgpt_remote_port": chatgpt_remote_port,
                "chatgpt_local_port": chatgpt_local_port,
                "node_id": node_id,
            },
        }

        self.connected_clients[node_id] = ConnectedClient(
            node_id=node_id,
            node_name=node_name or node_id,
            remote_port=remote_port,
            local_port=local_port,
            chatgpt_remote_port=chatgpt_remote_port,
            chatgpt_local_port=chatgpt_local_port,
            status="registered",
            connected_at=time.time(),
            enable_p2p=enable_p2p,
            secret_key=secret_key,
        )

        toml_content = self._config_to_toml(config)
        launch_cmd = self._build_launch_command(node_id, local_port, "<FRPS_SERVER_IP>")

        self._save_config()
        return {
            **config,
            "toml_content": toml_content,
            "launch_command": launch_cmd,
            "chatgpt_remote_port": chatgpt_remote_port,
        }

    def _config_to_toml(self, config: Dict) -> str:
        """将配置字典转为 TOML 字符串"""
        lines = []
        lines.append(f'serverPort = {config["serverPort"]}')
        if "auth" in config and "token" in config["auth"]:
            lines.append('')
            lines.append('[auth]')
            lines.append(f'token = "{config["auth"]["token"]}"')
        for proxy in config.get("proxies", []):
            lines.append('')
            lines.append('[[proxies]]')
            for k, v in proxy.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                elif isinstance(v, bool):
                    lines.append(f'{k} = {str(v).lower()}')
                else:
                    lines.append(f'{k} = {v}')
        return "\n".join(lines) + "\n"

    def _build_launch_command(self, node_id: str, local_port: int, server_addr: str, manager_port: int = None) -> str:
        """构建 exo 节点启动命令（manager URL 由前端 location.origin 最终覆盖）"""
        import os
        if manager_port is None:
            manager_port = int(os.getenv("EXO_MANAGER_PORT", "8080"))
        token = self.config.token or ""
        safe_token = token.replace("'", "'\\''")
        mgr = f"http://{server_addr}:{manager_port}" if manager_port else f"http://{server_addr}"
        return (
            f'python -m exo.main '
            f'--disable-tui '
            f'--discovery-module frp '
            f'--node-id {node_id} '
            f'--node-port {local_port} '
            f'--frp-server-addr {server_addr} '
            f'--frp-server-port {self.config.bind_port} '
            f"--frp-token '{safe_token}' "
            f'--manager {mgr}'
        )

    # ==================== 状态查询 ====================

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        running = self.is_running()
        if running and self.status != FRPStatus.RUNNING:
            self.status = FRPStatus.RUNNING
        elif not running and self.status == FRPStatus.RUNNING:
            self.status = FRPStatus.ERROR

        uptime = 0
        if self.start_time and running:
            uptime = time.time() - self.start_time

        online_clients = [
            c.to_dict() for c in self.connected_clients.values()
            if c.status == "online"
        ]

        return {
            "status": self.status.value,
            "running": running,
            "config": self.config.to_dict(),
            "uptime_seconds": round(uptime, 1),
            "start_time": self.start_time,
            "pid": self.process.pid if self.process else None,
            "frps_path": str(self.frps_path),
            "config_path": str(self.config_path),
            "connected_clients": online_clients,
            "total_clients": len(self.connected_clients),
            "total_connections": self.total_connections,
            "installed": self.check_frps_installed(),
            "version": FRP_VERSION,
        }

    def get_clients(self) -> List[Dict]:
        """获取所有已注册客户端"""
        return [c.to_dict() for c in self.connected_clients.values()]

    def get_client_config(self, node_id: str) -> Optional[Dict]:
        """获取指定节点的 frpc 配置"""
        return self.client_configs.get(node_id)

    def remove_client(self, node_id: str) -> bool:
        """移除已注册的客户端"""
        if node_id in self.connected_clients:
            del self.connected_clients[node_id]
        if node_id in self.client_configs:
            del self.client_configs[node_id]
        self._save_config()
        logger.info(f"[FRPServer] 已移除客户端: {node_id}")
        return True

    # ==================== 用户连接信息 (核心) ====================

    def get_user_connection_info(
        self,
        user_node_id: str,
        local_port: int = 50051,
        chatgpt_local_port: int = 52415,
        server_addr: str = "",
        manager_addr: str = "",
        manager_port: int = None,
    ) -> Dict[str, Any]:
        """
        获取用户的 FRP 连接信息和 exo 启动命令

        返回 exo 节点连接到本 frps 所需的全部参数。
        此方法供 user.html 页面调用，为登录用户展示专属启动命令。

        【关于 --manager URL】
          后端生成的 launch_command 里 --manager 仅作占位（使用 EXO_MANAGER_PORT 默认值）；
          前端统一在 getFinalNodeData() 里用 location.protocol + '//' + location.host 覆盖，
          保证与用户浏览器实际访问的 Manager 地址 + 端口完全一致。

        Args:
            user_node_id: 节点 ID（基于 user_id 自动生成）
            local_port: 本地 gRPC 端口
            chatgpt_local_port: 本地 ChatGPT API HTTP 端口
            server_addr: frps 公网地址（从请求 URL 自动获取）

        Returns:
            含 token、secretKey、launch_command、frpc_toml_config 的字典
        """
        import os
        if manager_port is None:
            manager_port = int(os.getenv("EXO_MANAGER_PORT", "8080"))
        remote_port = self.calculate_remote_port(user_node_id)
        chatgpt_remote_port = self.calculate_remote_port(user_node_id, service="chatgpt")
        token = self.config.token or ""
        secret_key = hashlib.sha256(token.encode('utf-8')).hexdigest()[:16] if token else ""

        addr = server_addr or "<FRPS_SERVER_IP>"
        mgr_addr = manager_addr or addr
        mgr = f"http://{mgr_addr}:{manager_port}" if manager_port else f"http://{mgr_addr}"
        safe_token = token.replace("'", "'\\''")

        launch_command = (
            f'python -m exo.main '
            f'--disable-tui '
            f'--discovery-module frp '
            f'--node-id {user_node_id} '
            f'--node-port {local_port} '
            f'--chatgpt-api-port {chatgpt_local_port} '
            f'--frp-server-addr {addr} '
            f'--frp-server-port {self.config.bind_port} '
            f"--frp-token '{safe_token}' "
            f'--manager {mgr}'
        )

        config = self.generate_frpc_config_for_user(
            node_id=user_node_id,
            local_port=local_port,
            chatgpt_local_port=chatgpt_local_port,
            server_addr=addr,
        )

        return {
            "node_id": user_node_id,
            "server_addr": addr,
            "bind_port": self.config.bind_port,
            "remote_port": remote_port,
            "chatgpt_remote_port": chatgpt_remote_port,
            "local_port": local_port,
            "chatgpt_local_port": chatgpt_local_port,
            "frp_token": token,
            "secret_key": secret_key,
            "p2p_enabled": self.config.enable_xtcp,
            "launch_command": launch_command,
            "frpc_toml_config": config,
            "frps_running": self.is_running(),
            "frps_version": FRP_VERSION,
        }

    def generate_frpc_config_for_user(
        self,
        node_id: str,
        local_port: int = 50051,
        chatgpt_local_port: int = 52415,
        server_addr: Optional[str] = None,
    ) -> str:
        """为用户生成可直接使用的 frpc.toml 配置文本"""
        remote_port = self.calculate_remote_port(node_id)
        chatgpt_remote_port = self.calculate_remote_port(node_id, service="chatgpt")
        token = self.config.token or ""
        secret_key = hashlib.sha256(token.encode('utf-8')).hexdigest()[:16] if token else ""
        addr = server_addr or "<FRPS_SERVER_IP>"

        lines = []
        lines.append(f'serverAddr = "{addr}"')
        lines.append(f'serverPort = {self.config.bind_port}')
        lines.append('')
        lines.append('[auth]')
        lines.append(f'token = "{token}"')

        if self.config.enable_xtcp:
            lines.append('')
            lines.append('[[proxies]]')
            lines.append(f'name = "exo_p2p_{node_id}"')
            lines.append('type = "xtcp"')
            lines.append(f'secretKey = "{secret_key}"')
            lines.append('localIP = "127.0.0.1"')
            lines.append(f'localPort = {local_port}')

        lines.append('')
        lines.append('[[proxies]]')
        lines.append(f'name = "exo_tcp_{node_id}"')
        lines.append('type = "tcp"')
        lines.append('localIP = "127.0.0.1"')
        lines.append(f'localPort = {local_port}')
        lines.append(f'remotePort = {remote_port}')

        lines.append('')
        lines.append('[[proxies]]')
        lines.append(f'name = "exo_chatgpt_{node_id}"')
        lines.append('type = "tcp"')
        lines.append('localIP = "127.0.0.1"')
        lines.append(f'localPort = {chatgpt_local_port}')
        lines.append(f'remotePort = {chatgpt_remote_port}')

        return "\n".join(lines) + "\n"


# ==================== 全局单例 ====================
_instance: Optional[FRPServerManager] = None


def get_frp_server_manager() -> FRPServerManager:
    """获取全局 FRP Server 管理器单例"""
    global _instance
    if _instance is None:
        _instance = FRPServerManager()
    return _instance



