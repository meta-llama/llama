import pytest
import torch
from safety.wrapper import LlamaSafetyOptimizer, PerformanceMetrics

def test_safety_optimizer_initialization():
    model = torch.nn.Linear(10, 10)  # Dummy model for testing
    optimizer = LlamaSafetyOptimizer(model)
    assert optimizer.safety_threshold == 0.8
    assert optimizer.enable_memory_tracking == True

def test_memory_tracking():
    model = torch.nn.Linear(10, 10)
    optimizer = LlamaSafetyOptimizer(model)
    memory_stats = optimizer._track_memory()
    assert 'cpu_percent' in memory_stats
    assert 'ram_used' in memory_stats

def test_safety_checks():
    model = torch.nn.Linear(10, 10)
    optimizer = LlamaSafetyOptimizer(model)
    test_tensor = torch.randn(1, 10)
    is_safe, metrics = optimizer._check_safety(test_tensor)
    assert isinstance(is_safe, bool)
    assert 'max_activation' in metrics
    assert 'mean_activation' in metrics
    assert 'std_activation' in metrics

def test_safe_forward():
    model = torch.nn.Linear(10, 10)
    optimizer = LlamaSafetyOptimizer(model)
    input_tensor = torch.randn(1, 10)
    result = optimizer.safe_forward(input_tensor, start_pos=0)
    
    assert 'output' in result
    assert 'is_safe' in result
    assert 'safety_metrics' in result
    assert 'performance' in result