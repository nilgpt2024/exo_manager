"""
EXO Cluster Manager - 独立的集群管理系统
=======================================

一个完整的Web应用，用于管理和监控EXO分布式AI推理集群。

主要功能:
- 🖥️ 节点管理：添加、移除、监控所有EXO节点
- 📊 实时仪表板：WebSocket实时推送集群状态
- 🎮 GPU池管理：统一调配模型权重和显存资源
- 📈 可视化界面：现代化的Web管理控制台
- 🔌 RESTful API：完整的API接口供程序调用

快速开始:
    pip install -r requirements.txt
    python start_manager.py --port 8080
    
    然后访问:
    - Web界面: http://localhost:8080
    - API文档: http://localhost:8080/docs

项目结构:
    exo_manager/
    ├── server.py              # FastAPI主服务器 + REST API
    ├── cluster_core.py        # 核心逻辑（节点连接、状态管理）
    ├── gpu_pool_integration.py # GPU池管理集成
    ├── start_manager.py       # 启动脚本
    ├── static/
    │   └── index.html         # Web管理界面
    └── requirements.txt       # Python依赖

作者: EXO Team
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "EXO Team"

from .cluster_core import (
    EXOClusterManager,
    NodeConnector,
    EXONodeInfo,
    get_cluster_manager,
    NodeStatus
)

from .gpu_pool_integration import (
    GPUPoolIntegration,
    ModelAllocation
)

__all__ = [
    'EXOClusterManager',
    'NodeConnector', 
    'EXONodeInfo',
    'get_cluster_manager',
    'NodeStatus',
    'GPUPoolIntegration',
    'ModelAllocation'
]
