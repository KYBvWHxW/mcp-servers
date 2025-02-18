#!/usr/bin/env python3
"""
MCP服务器部署脚本
负责启动和监控所有MCP服务器
"""

import os
import sys
import json
import signal
import subprocess
import time
from typing import Dict, List, Optional

class MCPServerManager:
    def __init__(self):
        self.config_path = os.path.expanduser("~/.codeium/windsurf/mcp_config.json")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.servers: Dict[str, subprocess.Popen] = {}
        self.log_dir = os.path.join(self.base_dir, "logs")
        
    def load_config(self) -> Dict:
        """加载MCP配置"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
            
    def ensure_log_directory(self):
        """确保日志目录存在"""
        os.makedirs(self.log_dir, exist_ok=True)
        
    def start_server(self, name: str, config: Dict) -> Optional[subprocess.Popen]:
        """启动单个服务器"""
        try:
            # 准备命令
            command = [config["command"]] + config["args"]
            
            # 打开日志文件
            log_file = open(os.path.join(self.log_dir, f"{name}.log"), 'w')
            
            # 启动进程
            process = subprocess.Popen(
                command,
                cwd=self.base_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            print(f"启动服务器 {name}，PID: {process.pid}")
            return process
            
        except Exception as e:
            print(f"启动服务器 {name} 失败: {str(e)}")
            return None
            
    def start_all_servers(self):
        """启动所有服务器"""
        print("启动MCP服务器系统...")
        self.ensure_log_directory()
        
        config = self.load_config()
        for name, server_config in config["mcpServers"].items():
            if name in ["mongodb", "nlp"]:  # 跳过非MCP服务器
                continue
            process = self.start_server(name, server_config)
            if process:
                self.servers[name] = process
                # 每个服务器启动后等待一秒，确保端口绑定完成
                time.sleep(1)
                
    def stop_server(self, name: str, process: subprocess.Popen):
        """停止单个服务器"""
        try:
            process.terminate()
            process.wait(timeout=5)
            print(f"停止服务器 {name}")
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"强制停止服务器 {name}")
            
    def stop_all_servers(self):
        """停止所有服务器"""
        print("\n停止所有服务器...")
        for name, process in self.servers.items():
            self.stop_server(name, process)
        # 等待端口完全释放
        time.sleep(2)
            
    def check_server_status(self, name: str, process: subprocess.Popen) -> bool:
        """检查服务器状态"""
        if process.poll() is None:
            print(f"服务器 {name} 运行正常 (PID: {process.pid})")
            return True
        else:
            print(f"服务器 {name} 已停止 (返回码: {process.returncode})")
            return False
            
    def monitor_servers(self):
        """监控所有服务器状态"""
        print("\n监控服务器状态...")
        all_running = True
        for name, process in self.servers.items():
            if not self.check_server_status(name, process):
                all_running = False
        return all_running
        
    def handle_signal(self, signum, frame):
        """处理信号"""
        print("\n接收到停止信号...")
        self.stop_all_servers()
        sys.exit(0)

def main():
    manager = MCPServerManager()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, manager.handle_signal)
    signal.signal(signal.SIGTERM, manager.handle_signal)
    
    try:
        # 启动服务器
        manager.start_all_servers()
        
        # 监控循环
        while True:
            if not manager.monitor_servers():
                print("检测到服务器异常，重启系统...")
                manager.stop_all_servers()
                # 等待端口完全释放
                time.sleep(5)
                manager.start_all_servers()
            time.sleep(10)  # 每10秒检查一次
            
    except KeyboardInterrupt:
        print("\n接收到中断信号...")
    finally:
        manager.stop_all_servers()

if __name__ == "__main__":
    main()
