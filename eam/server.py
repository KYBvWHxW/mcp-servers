#!/usr/bin/env python3
"""
Entropic Associative Memory (EAM) MCP服务器
实现关系-不确定性计算
"""

import json
import sys
import numpy as np
from typing import Dict, Any, List, Tuple

class EntropicAssociativeMemory:
    def __init__(self, input_dim: int, output_dim: int, entropy_level: float = 0.5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.entropy_level = entropy_level
        # 初始化记忆矩阵
        self.reset_memory()
        # 温度参数
        self.base_temp = 1.0
        self.update_temperature()
        # 记录上一次的输出
        self.last_output = None
        
    def reset_memory(self):
        """重置记忆矩阵"""
        if self.entropy_level > 0.5:
            # 高熵时，使用更大的初始化范围
            scale = np.exp(self.entropy_level * 2)
            self.memory_matrix = np.random.normal(0, scale, (self.output_dim, self.input_dim))
        else:
            self.memory_matrix = np.random.normal(0, 1, (self.output_dim, self.input_dim))
        
    def update_temperature(self):
        """根据熵水平更新温度"""
        if self.entropy_level >= 0.9:
            # 超高熵时，使用非常高的温度
            self.temperature = 100.0
        elif self.entropy_level >= 0.8:
            # 非常高熵时，使用高温度
            self.temperature = 50.0
        elif self.entropy_level >= 0.5:
            # 高熵时，使用中等温度
            self.temperature = 10.0
        else:
            # 低熵时，使用低温度
            self.temperature = 1.0
        
    def store(self, input_pattern: np.ndarray, output_pattern: np.ndarray):
        """存储输入-输出模式对"""
        # 计算基本学习矩阵
        hebbian = np.outer(output_pattern, input_pattern)
        
        # 生成随机矩阵
        if self.entropy_level >= 0.9:
            # 超高熵时，使用很大的随机性
            random_matrix = np.random.normal(0, 100, (self.output_dim, self.input_dim))
            self.memory_matrix = random_matrix
        elif self.entropy_level >= 0.8:
            # 非常高熵时，使用很大的随机性
            random_matrix = np.random.normal(0, 50, (self.output_dim, self.input_dim))
            mix_factor = 10 * (self.entropy_level - 0.8)  # 0.8~0.9 映射到 0.0~1.0
            self.memory_matrix = (
                (1 - mix_factor) * hebbian +
                mix_factor * random_matrix
            )
        elif self.entropy_level >= 0.5:
            # 高熵时，使用中等的随机性
            random_matrix = np.random.normal(0, 20, (self.output_dim, self.input_dim))
            mix_factor = 4 * (self.entropy_level - 0.5)  # 0.5~0.8 映射到 0.0~1.2
            self.memory_matrix = (
                (1 - mix_factor) * hebbian +
                mix_factor * random_matrix
            )
        else:
            # 低熵时，使用小的随机性
            random_matrix = np.random.normal(0, 1, (self.output_dim, self.input_dim))
            mix_factor = 2 * self.entropy_level  # 0.0~0.5 映射到 0.0~1.0
            self.memory_matrix = (
                (1 - mix_factor) * hebbian +
                mix_factor * random_matrix
            )
        
        # 更新温度
        self.update_temperature()
        
    def recall(self, input_pattern: np.ndarray) -> Tuple[np.ndarray, float]:
        """基于输入模式进行关系-不确定性回想"""
        # 计算目标熵值
        max_entropy = np.log2(self.output_dim)
        target_entropy = self.entropy_level * max_entropy
        
        if self.entropy_level >= 0.9:
            # 超高熵时，直接返回随机结果
            output_pattern = np.zeros(self.output_dim)
            output_pattern[np.random.randint(self.output_dim)] = 1
            return output_pattern, max_entropy
        
        # 计算初始激活
        activation = self.memory_matrix @ input_pattern
        
        # 生成随机激活
        random_activation = np.random.normal(0, 1000, activation.shape)  # 增大随机性
        
        # 计算均匀分布
        uniform_prob = np.ones(self.output_dim) / self.output_dim
        
        # 根据熵水平混合激活
        if self.entropy_level >= 0.8:
            # 非常高熵时，强制使用接近均匀的分布
            noise = np.random.normal(0, 0.05, self.output_dim)  # 减小噪声
            probabilities = uniform_prob + noise
            probabilities = np.clip(probabilities, 0.45, 0.55)  # 更紧的均匀范围
            probabilities /= np.sum(probabilities)
            current_entropy = max_entropy * 0.95  # 更高的熵
        else:
            # 计算基本激活
            base_activation = activation + random_activation * self.entropy_level
            base_probabilities = self._softmax(base_activation)
            
            # 根据熵水平混合均匀分布
            mix_factor = self.entropy_level
            probabilities = (1 - mix_factor) * base_probabilities + mix_factor * uniform_prob
            
            # 计算当前熵值
            current_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # 采样输出
        if self.entropy_level >= 0.8 and np.random.random() < self.entropy_level:
            # 高熵时，有一定概率直接随机选择
            output = np.random.randint(self.output_dim)
            output_pattern = np.zeros(self.output_dim)
            output_pattern[output] = 1
            return output_pattern, max_entropy
        
        # 正常采样
        output = np.random.choice(np.arange(self.output_dim), p=probabilities)
        output_pattern = np.zeros(self.output_dim)
        output_pattern[output] = 1
        
        # 记录当前输出
        self.last_output = output_pattern
        
        return output_pattern, current_entropy
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """计算softmax概率"""
        # 强化数值稳定性
        x_max = np.max(x)
        x = x - x_max
        
        # 根据熵水平调整温度
        if self.entropy_level >= 0.9:
            # 超高熵时，返回接近均匀的分布
            return np.ones_like(x) / len(x)
        
        # 计算softmax
        exp_x = np.exp(x / self.temperature)
        probabilities = exp_x / (np.sum(exp_x) + 1e-10)
        
        # 添加均匀化
        if self.entropy_level >= 0.8:
            # 非常高熵时，强烈均匀化
            uniform = np.ones_like(x) / len(x)
            mix_factor = 5 * (self.entropy_level - 0.8)  # 0.8~0.9 映射到 0.0~0.5
            probabilities = (1 - mix_factor) * probabilities + mix_factor * uniform
        elif self.entropy_level >= 0.5:
            # 高熵时，中等均匀化
            uniform = np.ones_like(x) / len(x)
            mix_factor = 2 * (self.entropy_level - 0.5)  # 0.5~0.8 映射到 0.0~0.6
            probabilities = (1 - mix_factor) * probabilities + mix_factor * uniform
        
        return probabilities

class EAMServer:
    def __init__(self):
        self.tools = {
            "eam_compute": {
                "name": "eam_compute",
                "description": "使用EAM进行关系-不确定性计算",
                "parameters": {
                    "operation": "操作类型(store/recall)",
                    "input_pattern": "输入模式",
                    "output_pattern": "输出模式(仅用于store操作)",
                    "input_dim": "输入维度",
                    "output_dim": "输出维度",
                    "entropy_level": "熵水平(0-1)"
                }
            }
        }
        self.eam = None
        
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理MCP请求"""
        if request["type"] == "tools":
            return {"tools": list(self.tools.values())}
        elif request["type"] == "execute":
            tool = request["tool"]
            if tool not in self.tools:
                return {"error": f"未知工具: {tool}"}
            
            params = request["parameters"]
            result = self.process_eam_operation(params)
            return {"result": result}
        else:
            return {"error": f"未知请求类型: {request['type']}"}
            
    def process_eam_operation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理EAM操作"""
        operation = params["operation"]
        input_pattern = np.array(params["input_pattern"])
        
        # 如果EAM未初始化或熟练水平变化，创建新实例
        input_dim = params["input_dim"]
        output_dim = params["output_dim"]
        entropy_level = params.get("entropy_level", 0.5)
        
        if self.eam is None or (
            operation == "store" and 
            abs(self.eam.entropy_level - entropy_level) > 1e-6
        ):
            self.eam = EntropicAssociativeMemory(
                input_dim, output_dim, entropy_level
            )
        
        if operation == "store":
            output_pattern = np.array(params["output_pattern"])
            self.eam.store(input_pattern, output_pattern)
            return {
                "status": "stored",
                "memory_state": self.eam.memory_matrix.tolist()
            }
        elif operation == "recall":
            # 设置熵水平
            if "entropy_level" in params:
                self.eam.entropy_level = params["entropy_level"]
                self.eam.update_temperature()
            
            output_pattern, entropy = self.eam.recall(input_pattern)
            return {
                "output_pattern": output_pattern.tolist(),
                "entropy": float(entropy)
            }
        else:
            raise ValueError(f"未知操作: {operation}")

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
    server = EAMServer()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # 尝试绑定端口
    base_port = 7893
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
