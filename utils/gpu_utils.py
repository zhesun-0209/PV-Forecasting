#!/usr/bin/env python3
"""
GPU utility functions
Provide GPU memory monitoring and status checking functionality
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np

def get_gpu_memory_used():
    """
    Get current GPU memory usage (GB)
    
    Returns:
        GPU memory usage (GB), returns 0 if GPU unavailable
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    try:
        if torch.cuda.is_available():
            # Get current GPU memory usage (bytes)
            memory_allocated = torch.cuda.memory_allocated()
            # Convert to GB
            memory_gb = memory_allocated / (1024 ** 3)
            return round(memory_gb, 2)
        else:
            return 0.0
    except Exception as e:
        return 0.0

def get_gpu_memory_total():
    """
    Get total GPU memory (GB)
    
    Returns:
        Total GPU memory (GB), returns 0 if GPU unavailable
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    try:
        if torch.cuda.is_available():
            # Get total GPU memory (bytes)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            # Convert to GB
            memory_gb = memory_total / (1024 ** 3)
            return round(memory_gb, 2)
        else:
            return 0.0
    except Exception as e:
        return 0.0

def get_gpu_memory_free():
    """
    Get available GPU memory (GB)
    
    Returns:
        Available GPU memory (GB), returns 0 if GPU unavailable
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    try:
        if torch.cuda.is_available():
            # Get available GPU memory (bytes)
            memory_reserved = torch.cuda.memory_reserved()
            memory_allocated = torch.cuda.memory_allocated()
            memory_free = memory_reserved - memory_allocated
            # Convert to GB
            memory_gb = memory_free / (1024 ** 3)
            return round(memory_gb, 2)
        else:
            return 0.0
    except Exception as e:
        return 0.0

def check_gpu_availability():
    """
    Check GPU availability
    
    Returns:
        Dictionary containing GPU status information
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
    Clear GPU memory
    """
    if not TORCH_AVAILABLE:
        return
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:

def print_gpu_status():
    """
    Print GPU status information
    """
    status = check_gpu_availability()
    if status['available']:
    else:
