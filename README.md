# Model Context Protocol (MCP) Servers

基于论文《Representation and Interpretation in Artificial and Natural Computing》的计算模式理论实现的MCP服务器集合。

## 项目结构

```
mcp-servers/
├── basic/          # 基础算法服务器
├── quantum/        # 量子模拟服务器
├── neuromorphic/   # 神经形态计算服务器
└── eam/            # Entropic Associative Memory服务器
```

## 服务器说明

1. 基础算法服务器 (basic)
   - 实现常规算法计算
   - 支持图灵机可计算的函数

2. 量子模拟服务器 (quantum)
   - 模拟量子计算过程
   - 实现量子算法

3. 神经形态计算服务器 (neuromorphic)
   - 使用memristive设备模型
   - 实现类脑计算

4. EAM服务器 (eam)
   - 实现Entropic Associative Memory模型
   - 支持关系-不确定性计算

## 配置说明

在`~/.codeium/windsurf/mcp_config.json`中配置服务器：

```json
{
  "mcpServers": {
    "basic": {
      "command": "python3",
      "args": ["basic/server.py"]
    },
    "quantum": {
      "command": "python3",
      "args": ["quantum/server.py"]
    },
    "neuromorphic": {
      "command": "python3",
      "args": ["neuromorphic/server.py"]
    },
    "eam": {
      "command": "python3",
      "args": ["eam/server.py"]
    }
  }
}
```

## 使用说明

1. 克隆仓库
2. 安装依赖
3. 配置MCP服务器
4. 在Windsurf中使用服务器提供的工具
