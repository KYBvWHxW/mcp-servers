#!/usr/bin/env python3
"""
基础算法MCP服务器
实现常规的算法计算功能
"""

import json
import sys
from typing import Dict, Any, List

class BasicAlgorithmServer:
    def __init__(self):
        self.tools = {
            "compute": {
                "name": "compute",
                "description": "执行基础算法计算",
                "parameters": {
                    "algorithm": "要执行的算法名称",
                    "inputs": "算法输入参数"
                }
            }
        }
        
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理MCP请求"""
        if request["type"] == "tools":
            return {"tools": list(self.tools.values())}
        elif request["type"] == "execute":
            tool = request["tool"]
            if tool not in self.tools:
                return {"error": f"未知工具: {tool}"}
            
            params = request["parameters"]
            result = self.execute_algorithm(
                params["algorithm"],
                params["inputs"]
            )
            return {"result": result}
        else:
            return {"error": f"未知请求类型: {request['type']}"}
            
    def execute_algorithm(self, algorithm: str, inputs: List[Any]) -> Any:
        """执行指定算法"""
        if algorithm == "fibonacci":
            n = inputs[0]
            return self._fibonacci(n)
        elif algorithm == "sort":
            arr = inputs[0]
            return sorted(arr)
        else:
            raise ValueError(f"未支持的算法: {algorithm}")
            
    def _fibonacci(self, n: int) -> int:
        """计算斐波那契数列第n项"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        else:
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b

def handle_signal(signum, frame):
    """处理信号"""
    print("\n接收到停止信号...")
    sys.exit(0)

def main():
    import socket
    import signal
    
    # 注册信号处理
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    
    # 创建服务器
    server = BasicAlgorithmServer()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # 尝试绑定端口
    base_port = 7890
    max_attempts = 10
    port = base_port
    
    for attempt in range(max_attempts):
        try:
            sock.bind(('localhost', port))
            sock.listen(5)
            print(f"服务器监听端口 {port}...")
            break
        except OSError as e:
            if attempt == max_attempts - 1:
                raise
            port = base_port + attempt + 1
            print(f"端口 {port-1} 被占用，尝试端口 {port}...")
    
    while True:
        try:
            # 接受连接
            conn, addr = sock.accept()
            print(f"接受连接：{addr}")
            
            # 读取请求
            data = conn.recv(4096).decode('utf-8')
            if not data:
                continue
                
            # 处理请求
            try:
                request = json.loads(data)
                response = server.handle_request(request)
                conn.sendall(json.dumps(response).encode('utf-8'))
            except json.JSONDecodeError:
                conn.sendall(json.dumps({"error": "Invalid JSON"}).encode('utf-8'))
            except Exception as e:
                conn.sendall(json.dumps({"error": str(e)}).encode('utf-8'))
            
            # 关闭连接
            conn.close()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误：{str(e)}")
            continue
    
    sock.close()

if __name__ == "__main__":
    main()
