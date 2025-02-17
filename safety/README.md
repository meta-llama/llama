# Llama Safety Optimizer

This module provides safety and optimization tools for Llama models.

## Features

- Runtime safety checks for model outputs
- Memory usage tracking and optimization
- Performance monitoring
- Automatic batch size optimization

## Usage

```python
from safety.wrapper import LlamaSafetyOptimizer

# Initialize with your model
optimizer = LlamaSafetyOptimizer(model)

# Use safe forward pass
result = optimizer.safe_forward(input_ids, start_pos=0)

# Get performance metrics
metrics = optimizer.get_performance_summary()
```
