#!/usr/bin/env python3
"""
GPU工具函数 | GPU utility functions
提供GPU内存监控和状态检查功能 | Provide GPU memory monitoring and status checking functionality
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch不可用，GPU功能将被禁用")  # PyTorch not available, GPU功能 will be disabled

import numpy as np

def get_gpu_memory_used():
    """
    获取当前GPU内存使用量（GB） | Get current GPU memory usage (GB)
    
    Returns:
        float: GPU内存使用量（GB），如果GPU不可用则返回0 | GPU memory usage (GB), returns 0 if GPU unavailable
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    try:
        if torch.cuda.is_available():
            # 获取当前GPU内存使用量（字节） | Get current GPU memory usage (bytes)
            memory_allocated = torch.cuda.memory_allocated()
            # 转换为GB | Convert to GB
            memory_gb = memory_allocated / (1024 ** 3)
            return round(memory_gb, 2)
        else:
            return 0.0
    except Exception as e:
        print(f"获取GPU内存使用量失败: {e}")  # Failed to get GPU memory usage: {e}
        return 0.0

def get_gpu_memory_total():
    """
    获取GPU总内存量（GB） | Get total GPU memory (GB)
    
    Returns:
        float: GPU总内存量（GB），如果GPU不可用则返回0 | Total GPU memory (GB), returns 0 if GPU unavailable
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    try:
        if torch.cuda.is_available():
            # 获取GPU总内存量（字节） | Get total GPU memory (bytes)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            # 转换为GB | Convert to GB
            memory_gb = memory_total / (1024 ** 3)
            return round(memory_gb, 2)
        else:
            return 0.0
    except Exception as e:
        print(f"获取GPU总内存量失败: {e}")  # Failed to get total GPU memory: {e}
        return 0.0

def get_gpu_memory_free():
    """
    获取GPU可用内存量（GB） | Get available GPU memory (GB)
    
    Returns:
        float: GPU可用内存量（GB），如果GPU不可用则返回0 | Available GPU memory (GB), returns 0 if GPU unavailable
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    try:
        if torch.cuda.is_available():
            # 获取GPU可用内存量（字节） | Get available GPU memory (bytes)
            memory_reserved = torch.cuda.memory_reserved()
            memory_allocated = torch.cuda.memory_allocated()
            memory_free = memory_reserved - memory_allocated
            # 转换为GB | Convert to GB
            memory_gb = memory_free / (1024 ** 3)
            return round(memory_gb, 2)
        else:
            return 0.0
    except Exception as e:
        print(f"获取GPU可用内存量失败: {e}")  # Failed to get available GPU memory: {e}
        return 0.0

def check_gpu_availability():
    """
    检查GPU可用性 | Check GPU availability
    
    Returns:
        dict: 包含GPU状态信息的字典 | Dictionary containing GPU status information
    """
    if not TORCH_AVAILABLE:
        return {
            'available': False,
            'device_count': 0,
            'current_device': None,
            'device_name': None,
            'memory_used': 0.0,
            'memory_total': 0.0,
            'memory_free': 0.0
        }
    
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            return {
                'available': True,
                'device_count': device_count,
                'current_device': current_device,
                'device_name': device_name,
                'memory_used': get_gpu_memory_used(),
                'memory_total': get_gpu_memory_total(),
                'memory_free': get_gpu_memory_free()
            }
        else:
            return {
                'available': False,
                'device_count': 0,
                'current_device': None,
                'device_name': None,
                'memory_used': 0.0,
                'memory_total': 0.0,
                'memory_free': 0.0
            }
    except Exception as e:
        print(f"检查GPU可用性失败: {e}")  # Failed to check GPU availability: {e}
        return {
            'available': False,
            'device_count': 0,
            'current_device': None,
            'device_name': None,
            'memory_used': 0.0,
            'memory_total': 0.0,
            'memory_free': 0.0
        }

def clear_gpu_memory():
    """
    清理GPU内存 | Clear GPU memory
    """
    if not TORCH_AVAILABLE:
        return
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU内存已清理")  # GPU memory cleared
    except Exception as e:
        print(f"清理GPU内存失败: {e}")  # Failed to clear GPU memory: {e}

def print_gpu_status():
    """
    打印GPU状态信息 | Print GPU status information
    """
    status = check_gpu_availability()
    if status['available']:
        print(f"✅ GPU可用: {status['device_name']}")  # GPU available: {status['device_name']}
        print(f"   设备数量: {status['device_count']}")  # Device count: {status['device_count']}
        print(f"   当前设备: {status['current_device']}")  # Current device: {status['current_device']}
        print(f"   内存使用: {status['memory_used']:.2f}GB / {status['memory_total']:.2f}GB")  # Memory usage
        print(f"   可用内存: {status['memory_free']:.2f}GB")  # Available memory
    else:
        print("❌ GPU不可用")  # GPU unavailable
