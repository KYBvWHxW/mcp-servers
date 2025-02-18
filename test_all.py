#!/usr/bin/env python3
"""
MCP服务器全量测试脚本
"""

import json
import subprocess
import sys
from typing import Dict, Any, List

def run_test(server_path: str, test_cases: List[Dict[str, Any]]) -> None:
    """运行单个服务器的测试用例"""
    print(f"\n测试服务器: {server_path}")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print("-" * 30)
        print("输入:", json.dumps(test_case, ensure_ascii=False, indent=2))
        
        try:
            # 启动服务器进程
            process = subprocess.Popen(
                ["python3", server_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 发送测试用例
            stdout, stderr = process.communicate(
                input=json.dumps(test_case) + "\n"
            )
            
            # 检查错误
            if stderr:
                print("错误:", stderr)
                continue
                
            # 解析并显示结果
            try:
                result = json.loads(stdout)
                print("输出:", json.dumps(result, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print("输出:", stdout)
                
        except Exception as e:
            print("异常:", str(e))
            
        print("-" * 30)

def main():
    # 基础算法服务器测试用例
    basic_tests = [
        # 测试工具列表
        {
            "type": "tools"
        },
        # 测试斐波那契计算
        {
            "type": "execute",
            "tool": "compute",
            "parameters": {
                "algorithm": "fibonacci",
                "inputs": [10]
            }
        },
        # 测试排序
        {
            "type": "execute",
            "tool": "compute",
            "parameters": {
                "algorithm": "sort",
                "inputs": [[3, 1, 4, 1, 5, 9, 2, 6, 5, 3]]
            }
        }
    ]
    
    # 量子模拟服务器测试用例
    quantum_tests = [
        # 测试工具列表
        {
            "type": "tools"
        },
        # 测试Hadamard门
        {
            "type": "execute",
            "tool": "quantum_circuit",
            "parameters": {
                "gates": [{"type": "H", "target": 0}],
                "num_qubits": 1
            }
        },
        # 测试CNOT门
        {
            "type": "execute",
            "tool": "quantum_circuit",
            "parameters": {
                "gates": [
                    {"type": "H", "target": 0},
                    {"type": "CNOT", "control": 0, "target": 1}
                ],
                "num_qubits": 2
            }
        }
    ]
    
    # 神经形态服务器测试用例
    neuromorphic_tests = [
        # 测试工具列表
        {
            "type": "tools"
        },
        # 测试单个memristor
        {
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
        },
        # 测试多个memristor
        {
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
        }
    ]
    
    # EAM服务器测试用例
    eam_tests = [
        # 测试工具列表
        {
            "type": "tools"
        },
        # 测试存储操作
        {
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
        },
        # 测试回想操作
        {
            "type": "execute",
            "tool": "eam_compute",
            "parameters": {
                "operation": "recall",
                "input_pattern": [1.0, 0.0],
                "input_dim": 2,
                "output_dim": 2
            }
        },
        # 测试高熵存储和回想
        {
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
        },
        {
            "type": "execute",
            "tool": "eam_compute",
            "parameters": {
                "operation": "recall",
                "input_pattern": [1.0, 0.0],
                "input_dim": 2,
                "output_dim": 2
            }
        }
    ]
    
    # 运行所有测试
    test_cases = [
        ("basic/server.py", basic_tests),
        ("quantum/server.py", quantum_tests),
        ("neuromorphic/server.py", neuromorphic_tests),
        ("eam/server.py", eam_tests)
    ]
    
    for server_path, tests in test_cases:
        run_test(server_path, tests)

if __name__ == "__main__":
    main()
