#!/usr/bin/env python3
"""
量子模拟MCP服务器
模拟基本的量子计算过程
"""

import json
import sys
import numpy as np
from typing import Dict, Any, List

class QuantumSimulationServer:
    def __init__(self):
        self.tools = {
            "quantum_circuit": {
                "name": "quantum_circuit",
                "description": "模拟量子电路",
                "parameters": {
                    "gates": "量子门序列",
                    "num_qubits": "量子比特数量"
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
            result = self.simulate_circuit(
                params["gates"],
                params["num_qubits"]
            )
            # 将复数转换为字符串表示
            def complex_to_dict(z):
                if isinstance(z, (list, np.ndarray)):
                    return [complex_to_dict(x) for x in z]
                if isinstance(z, (complex, np.complex128)):
                    return {"real": float(z.real), "imag": float(z.imag)}
                return float(z)
            
            result["state_vector"] = complex_to_dict(result["state_vector"])
            result["probabilities"] = complex_to_dict(result["probabilities"])
            return {"result": result}
        else:
            return {"error": f"未知请求类型: {request['type']}"}
            
    def simulate_circuit(self, gates: List[Dict[str, Any]], num_qubits: int) -> Dict[str, Any]:
        """模拟量子电路"""
        # 初始化量子态
        state = np.zeros(2**num_qubits, dtype=complex)
        state[0] = 1.0  # |0...0⟩态
        
        # 应用量子门
        for gate in gates:
            state = self._apply_gate(state, gate, num_qubits)
            
        # 计算测量结果概率
        probabilities = np.abs(state)**2
        
        return {
            "state_vector": state.tolist(),
            "probabilities": probabilities.tolist()
        }
        
    def _apply_gate(self, state: np.ndarray, gate: Dict[str, Any], num_qubits: int) -> np.ndarray:
        """应用量子门到量子态"""
        gate_type = gate["type"]
        target = gate.get("target", 0)
        
        if gate_type == "H":  # Hadamard门
            # 创建Hadamard矩阵
            h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            return self._apply_single_qubit_gate(state, h, target, num_qubits)
        elif gate_type == "X":  # Pauli-X门
            x = np.array([[0, 1], [1, 0]])
            return self._apply_single_qubit_gate(state, x, target, num_qubits)
        elif gate_type == "CNOT":  # CNOT门
            control = gate["control"]
            return self._apply_cnot(state, control, target, num_qubits)
        else:
            raise ValueError(f"未支持的量子门: {gate_type}")
            
    def _apply_single_qubit_gate(self, state: np.ndarray, gate: np.ndarray, 
                                target: int, num_qubits: int) -> np.ndarray:
        """应用单量子比特门"""
        # 重塑状态向量
        state = state.reshape([2] * num_qubits)
        
        # 应用门
        state = np.tensordot(gate, state, axes=([1], [target]))
        
        # 转置到正确的顺序
        axes = list(range(num_qubits))
        axes.remove(target)
        axes.insert(0, target)
        state = state.transpose(axes)
        
        return state.reshape(-1)
        
    def _apply_cnot(self, state: np.ndarray, control: int, target: int, 
                    num_qubits: int) -> np.ndarray:
        """应用CNOT门"""
        # 创建CNOT矩阵
        dim = 2 ** num_qubits
        cnot = np.eye(dim)
        
        # 计算控制和目标比特对应的基态
        for i in range(dim):
            # 获取二进制表示
            binary = format(i, f'0{num_qubits}b')
            # 如果控制比特为1
            if binary[control] == '1':
                # 翻转目标比特
                target_bit = '1' if binary[target] == '0' else '0'
                # 构造新的状态
                new_binary = list(binary)
                new_binary[target] = target_bit
                new_state = int(''.join(new_binary), 2)
                # 交换矩阵元素
                cnot[i, i] = 0
                cnot[i, new_state] = 1
        
        # 应用CNOT门
        return cnot @ state

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
    server = QuantumSimulationServer()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # 尝试绑定端口
    base_port = 7891
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
