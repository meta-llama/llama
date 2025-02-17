# safety/wrapper.py
import torch
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import psutil
import gc

@dataclass
class PerformanceMetrics:
    inference_time: float
    memory_used: int
    peak_memory: int
    gpu_utilization: float

class LlamaSafetyOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        safety_threshold: float = 0.8,
        enable_memory_tracking: bool = True
    ):
        self.model = model
        self.safety_threshold = safety_threshold
        self.enable_memory_tracking = enable_memory_tracking
        self.performance_history = []
        
    def _track_memory(self) -> Dict[str, int]:
        """Track current memory usage"""
        if not self.enable_memory_tracking:
            return {}
            
        memory_stats = {
            'cpu_percent': psutil.cpu_percent(),
            'ram_used': psutil.Process().memory_info().rss // 1024 // 1024
        }
        
        if torch.cuda.is_available():
            memory_stats.update({
                'gpu_used': torch.cuda.memory_allocated() // 1024 // 1024,
                'gpu_cached': torch.cuda.memory_reserved() // 1024 // 1024
            })
            
        return memory_stats
        
    def _check_safety(self, logits: torch.Tensor) -> Tuple[bool, Dict]:
        """Perform safety checks on model outputs"""
        with torch.no_grad():
            # Example safety checks - expand based on your needs
            max_value = torch.max(logits).item()
            mean_value = torch.mean(logits).item()
            std_value = torch.std(logits).item()
            
            safety_metrics = {
                'max_activation': max_value,
                'mean_activation': mean_value,
                'std_activation': std_value,
                'outlier_ratio': torch.sum(torch.abs(logits) > 5).item() / logits.numel()
            }
            
            # Simple safety check - can be made more sophisticated
            is_safe = (
                safety_metrics['outlier_ratio'] < 0.1 and 
                abs(safety_metrics['mean_activation']) < 2
            )
            
            return is_safe, safety_metrics
    
    def optimize_batch_size(self, start_size: int = 1, max_size: int = 32) -> int:
        """Find optimal batch size based on memory constraints"""
        current_size = start_size
        
        while current_size < max_size:
            try:
                # Create dummy batch
                dummy_input = torch.randint(
                    0, 1000, (current_size, 512), device=self.model.device
                )
                
                # Test forward pass
                with torch.no_grad():
                    _ = self.model(dummy_input, start_pos=0)
                
                # If successful, try larger batch
                current_size *= 2
                torch.cuda.empty_cache()
                gc.collect()
                
            except RuntimeError as e:
                # Memory error - return last successful size
                return current_size // 2
                
        return max_size
    
    def safe_forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        optimize_memory: bool = True
    ) -> Dict:
        """Forward pass with safety checks and performance monitoring"""
        start_time = time.time()
        initial_memory = self._track_memory()
        
        # Optimize batch size if requested
        if optimize_memory:
            batch_size = input_ids.shape[0]
            optimal_batch_size = self.optimize_batch_size(max_size=batch_size)
            
            if optimal_batch_size < batch_size:
                # Split into smaller batches
                outputs = []
                for i in range(0, batch_size, optimal_batch_size):
                    batch = input_ids[i:i + optimal_batch_size]
                    output = self.model(batch, start_pos + i)
                    outputs.append(output)
                output = torch.cat(outputs, dim=0)
            else:
                output = self.model(input_ids, start_pos)
        else:
            output = self.model(input_ids, start_pos)
            
        # Perform safety checks
        is_safe, safety_metrics = self._check_safety(output)
        
        # Track performance metrics
        end_time = time.time()
        final_memory = self._track_memory()
        
        performance = PerformanceMetrics(
            inference_time=end_time - start_time,
            memory_used=final_memory.get('ram_used', 0),
            peak_memory=max(
                initial_memory.get('ram_used', 0),
                final_memory.get('ram_used', 0)
            ),
            gpu_utilization=final_memory.get('gpu_used', 0)
        )
        
        self.performance_history.append(performance)
        
        return {
            'output': output if is_safe else None,
            'is_safe': is_safe,
            'safety_metrics': safety_metrics,
            'performance': performance.__dict__,
            'memory_tracking': final_memory
        }
    
    def get_performance_summary(self) -> Dict:
        """Get summary statistics of model performance"""
        if not self.performance_history:
            return {}
            
        avg_inference_time = sum(p.inference_time for p in self.performance_history) / len(self.performance_history)
        avg_memory_used = sum(p.memory_used for p in self.performance_history) / len(self.performance_history)
        peak_memory = max(p.peak_memory for p in self.performance_history)
        
        return {
            'average_inference_time': avg_inference_time,
            'average_memory_used': avg_memory_used,
            'peak_memory_usage': peak_memory,
            'total_inferences': len(self.performance_history)
        }