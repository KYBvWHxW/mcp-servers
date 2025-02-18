# MCP服务器部署指南

本文档说明如何部署和管理Model Context Protocol (MCP)服务器系统。

## 系统要求

- Python 3.8+
- numpy
- systemd (可选，用于系统服务管理)

## 部署步骤

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 配置MCP服务器
MCP服务器配置位于 `~/.codeium/windsurf/mcp_config.json`。确保配置文件包含所有必要的服务器设置。

3. 设置执行权限
```bash
chmod +x deploy.py
chmod +x */server.py
```

## 运行方式

### 方式1：直接运行

```bash
python3 deploy.py
```

这将启动所有MCP服务器，并进入监控模式。使用Ctrl+C停止所有服务器。

### 方式2：作为系统服务运行（推荐）

1. 复制服务文件到系统目录：
```bash
sudo cp mcp-servers.service /Library/LaunchDaemons/
```

2. 加载服务：
```bash
sudo launchctl load /Library/LaunchDaemons/mcp-servers.service
```

3. 启动服务：
```bash
sudo launchctl start mcp-servers
```

4. 停止服务：
```bash
sudo launchctl stop mcp-servers
```

5. 卸载服务：
```bash
sudo launchctl unload /Library/LaunchDaemons/mcp-servers.service
```

## 日志查看

所有服务器的日志文件位于 `logs` 目录下：
- `basic.log`: 基础算法服务器日志
- `quantum.log`: 量子模拟服务器日志
- `neuromorphic.log`: 神经形态服务器日志
- `eam.log`: EAM服务器日志

使用以下命令查看实时日志：
```bash
tail -f logs/*.log
```

## 服务器状态检查

1. 检查所有服务器状态：
```bash
ps aux | grep "server.py"
```

2. 检查特定服务器日志：
```bash
tail -f logs/[server_name].log
```

## 故障排除

1. 如果服务器无法启动：
   - 检查日志文件中的错误信息
   - 确保所有依赖都已正确安装
   - 验证配置文件的正确性

2. 如果服务器意外停止：
   - 部署脚本会自动尝试重启服务器
   - 检查日志文件以了解停止原因

3. 如果需要完全重置：
```bash
# 停止所有服务器
pkill -f "server.py"
# 清理日志
rm -rf logs/*
# 重新启动
python3 deploy.py
```

## 安全考虑

1. 服务器默认只监听本地连接
2. 所有通信使用JSON格式，确保安全的数据序列化
3. 每个服务器都有独立的错误处理机制

## 性能优化

1. 每个服务器都是独立进程，可以充分利用多核处理器
2. 日志文件使用追加模式，避免过度IO操作
3. 监控系统每10秒检查一次服务器状态，可根据需要调整
