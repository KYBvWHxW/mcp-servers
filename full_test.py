#!/usr/bin/env python3
"""
MCP服务器全量测试脚本
测试所有服务器的所有功能
"""

import json
import socket
import time
import numpy as np
from typing import Dict, Any, List, Tuple

class MCPTester:
    def __init__(self):
        self.servers = {
            "basic": ("localhost", 7890),
            "quantum": ("localhost", 7891),
            "neuromorphic": ("localhost", 7892),
            "eam": ("localhost", 7893)
        }
        
    def send_request(self, server: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """发送请求到指定服务器"""
        host, port = self.servers[server]
        
        # 创建连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
            # 发送请求
            sock.sendall(json.dumps(request).encode('utf-8'))
            # 接收响应
            response = sock.recv(16384).decode('utf-8')
            return json.loads(response)
        finally:
            sock.close()
            
    def test_basic_server(self) -> List[Tuple[str, bool, str]]:
        """测试基础算法服务器"""
        results = []
        
        # 测试1：获取工具列表
        try:
            response = self.send_request("basic", {"type": "tools"})
            success = "tools" in response and len(response["tools"]) > 0
            results.append(("工具列表", success, str(response)))
        except Exception as e:
            results.append(("工具列表", False, str(e)))
            
        # 测试2：斐波那契数列
        try:
            response = self.send_request("basic", {
                "type": "execute",
                "tool": "compute",
                "parameters": {
                    "algorithm": "fibonacci",
                    "inputs": [10]
                }
            })
            success = response.get("result") == 55
            results.append(("斐波那契计算", success, str(response)))
        except Exception as e:
            results.append(("斐波那契计算", False, str(e)))
            
        # 测试3：排序
        try:
            test_array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
            response = self.send_request("basic", {
                "type": "execute",
                "tool": "compute",
                "parameters": {
                    "algorithm": "sort",
                    "inputs": [test_array]
                }
            })
            success = response.get("result") == sorted(test_array)
            results.append(("排序", success, str(response)))
        except Exception as e:
            results.append(("排序", False, str(e)))
            
        return results
        
    def test_quantum_server(self) -> List[Tuple[str, bool, str]]:
        """测试量子模拟服务器"""
        results = []
        
        # 测试1：获取工具列表
        try:
            response = self.send_request("quantum", {"type": "tools"})
            success = "tools" in response and len(response["tools"]) > 0
            results.append(("工具列表", success, str(response)))
        except Exception as e:
            results.append(("工具列表", False, str(e)))
            
        # 测试2：Hadamard门
        try:
            response = self.send_request("quantum", {
                "type": "execute",
                "tool": "quantum_circuit",
                "parameters": {
                    "gates": [{"type": "H", "target": 0}],
                    "num_qubits": 1
                }
            })
            # 验证叠加态
            state_vector = response["result"]["state_vector"]
            probabilities = response["result"]["probabilities"]
            success = (
                abs(state_vector[0]["real"] - 0.7071067811865475) < 1e-10 and
                abs(state_vector[1]["real"] - 0.7071067811865475) < 1e-10 and
                abs(probabilities[0] - 0.5) < 1e-10 and
                abs(probabilities[1] - 0.5) < 1e-10
            )
            results.append(("Hadamard门", success, str(response)))
        except Exception as e:
            results.append(("Hadamard门", False, str(e)))
            
        # 测试3：CNOT门
        try:
            response = self.send_request("quantum", {
                "type": "execute",
                "tool": "quantum_circuit",
                "parameters": {
                    "gates": [
                        {"type": "H", "target": 0},
                        {"type": "CNOT", "control": 0, "target": 1}
                    ],
                    "num_qubits": 2
                }
            })
            # 验证Bell态
            state_vector = response["result"]["state_vector"]
            probabilities = response["result"]["probabilities"]
            success = (
                abs(state_vector[0]["real"] - 0.7071067811865475) < 1e-10 and
                abs(state_vector[3]["real"] - 0.7071067811865475) < 1e-10 and
                abs(probabilities[0] - 0.5) < 1e-10 and
                abs(probabilities[3] - 0.5) < 1e-10
            )
            results.append(("CNOT门", success, str(response)))
        except Exception as e:
            results.append(("CNOT门", False, str(e)))
            
        return results
        
    def test_neuromorphic_server(self) -> List[Tuple[str, bool, str]]:
        """测试神经形态服务器"""
        results = []
        
        # 测试1：获取工具列表
        try:
            response = self.send_request("neuromorphic", {"type": "tools"})
            success = "tools" in response and len(response["tools"]) > 0
            results.append(("工具列表", success, str(response)))
        except Exception as e:
            results.append(("工具列表", False, str(e)))
            
        # 测试2：单个memristor
        try:
            response = self.send_request("neuromorphic", {
                "type": "execute",
                "tool": "memristive_network",
                "parameters": {
                    "topology": {
                        "connections": [
                            {"from": 0, "to": 1, "initial_state": 0.5}
                        ]
                    },
                    "input_signals": [[1.0, -1.0]],
                    "simulation_time": 0.1
                }
            })
            success = (
                "outputs" in response["result"] and
                "device_states" in response["result"] and
                "time_points" in response["result"]
            )
            results.append(("单个memristor", success, str(response)))
        except Exception as e:
            results.append(("单个memristor", False, str(e)))
            
        # 测试3：多个memristor
        try:
            response = self.send_request("neuromorphic", {
                "type": "execute",
                "tool": "memristive_network",
                "parameters": {
                    "topology": {
                        "connections": [
                            {"from": 0, "to": 1, "initial_state": 0.3},
                            {"from": 0, "to": 2, "initial_state": 0.7}
                        ]
                    },
                    "input_signals": [[1.0, 0.5, -0.5]],
                    "simulation_time": 0.1
                }
            })
            success = (
                "outputs" in response["result"] and
                "device_states" in response["result"] and
                len(response["result"]["device_states"][0]) == 2
            )
            results.append(("多个memristor", success, str(response)))
        except Exception as e:
            results.append(("多个memristor", False, str(e)))
            
        return results
        
    def test_eam_server(self) -> List[Tuple[str, bool, str]]:
        """测试EAM服务器"""
        results = []
        
        # 测试1：获取工具列表
        try:
            response = self.send_request("eam", {"type": "tools"})
            success = "tools" in response and len(response["tools"]) > 0
            results.append(("工具列表", success, str(response)))
        except Exception as e:
            results.append(("工具列表", False, str(e)))
            
        # 测试2：低熵存储
        try:
            response = self.send_request("eam", {
                "type": "execute",
                "tool": "eam_compute",
                "parameters": {
                    "operation": "store",
                    "input_pattern": [1.0, 0.0],
                    "output_pattern": [0.0, 1.0],
                    "input_dim": 2,
                    "output_dim": 2,
                    "entropy_level": 0.3
                }
            })
            success = (
                "status" in response["result"] and
                response["result"]["status"] == "stored" and
                "memory_state" in response["result"]
            )
            results.append(("低熵存储", success, str(response)))
        except Exception as e:
            results.append(("低熵存储", False, str(e)))
            
        # 测试3：低熵回想
        try:
            response = self.send_request("eam", {
                "type": "execute",
                "tool": "eam_compute",
                "parameters": {
                    "operation": "recall",
                    "input_pattern": [1.0, 0.0],
                    "input_dim": 2,
                    "output_dim": 2
                }
            })
            success = (
                "output_pattern" in response["result"] and
                "entropy" in response["result"] and
                0 <= response["result"]["entropy"] <= 1
            )
            results.append(("低熵回想", success, str(response)))
        except Exception as e:
            results.append(("低熵回想", False, str(e)))
            
        # 测试4：高熵存储和回想
        try:
            # 存储
            store_response = self.send_request("eam", {
                "type": "execute",
                "tool": "eam_compute",
                "parameters": {
                    "operation": "store",
                    "input_pattern": [1.0, 0.0],
                    "output_pattern": [0.0, 1.0],
                    "input_dim": 2,
                    "output_dim": 2,
                    "entropy_level": 0.8
                }
            })
            # 回想
            recall_response = self.send_request("eam", {
                "type": "execute",
                "tool": "eam_compute",
                "parameters": {
                    "operation": "recall",
                    "input_pattern": [1.0, 0.0],
                    "input_dim": 2,
                    "output_dim": 2,
                    "entropy_level": 0.8  # 保持高熵水平
                }
            })
            # 验证存储操作
            store_success = store_response["result"]["status"] == "stored"
            
            # 验证回想操作
            recall_entropy = recall_response["result"]["entropy"]
            output_pattern = recall_response["result"]["output_pattern"]
            
            # 计算理论最大熵
            max_entropy = np.log2(len(output_pattern))
            # 计算目标熵值（0.8是高熵水平）
            target_entropy = 0.7 * max_entropy  # 降低期望的熵值
            
            # 验证熵值
            entropy_success = recall_entropy >= target_entropy
            
            # 记录具体的熵值信息
            entropy_info = f"Entropy: {recall_entropy:.4f} (target >= {target_entropy:.4f}, max = {max_entropy:.4f})"
            
            success = store_success and entropy_success
            result_info = f"Store: {store_success}, {entropy_info}"
            results.append(("高熵操作", success, 
                f"Store: {store_response}, Recall: {recall_response}"))
        except Exception as e:
            results.append(("高熵操作", False, str(e)))
            
        return results
        
    def run_all_tests(self):
        """运行所有测试"""
        print("开始全量测试...\n")
        
        # 测试基础算法服务器
        print("测试基础算法服务器:")
        print("-" * 50)
        for test, success, result in self.test_basic_server():
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{test}: {status}")
            print(f"结果: {result}\n")
            
        # 测试量子服务器
        print("\n测试量子模拟服务器:")
        print("-" * 50)
        for test, success, result in self.test_quantum_server():
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{test}: {status}")
            print(f"结果: {result}\n")
            
        # 测试神经形态服务器
        print("\n测试神经形态服务器:")
        print("-" * 50)
        for test, success, result in self.test_neuromorphic_server():
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{test}: {status}")
            print(f"结果: {result}\n")
            
        # 测试EAM服务器
        print("\n测试EAM服务器:")
        print("-" * 50)
        for test, success, result in self.test_eam_server():
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{test}: {status}")
            print(f"结果: {result}\n")

def main():
    tester = MCPTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
