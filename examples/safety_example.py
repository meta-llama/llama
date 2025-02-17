import torch
from safety.wrapper import LlamaSafetyOptimizer
from llama import Transformer, ModelArgs

def main():
    # Initialize model
    params = ModelArgs(
        dim=512,  # Smaller for testing
        n_layers=8,
        n_heads=8,
        vocab_size=1000
    )
    model = Transformer(params)
    
    # Initialize safety wrapper
    safe_model = LlamaSafetyOptimizer(model)
    
    # Test input
    input_ids = torch.randint(0, 1000, (1, 512))
    
    # Run with safety checks
    result = safe_model.safe_forward(input_ids, start_pos=0)
    
    # Print results
    print("\nSafety Check Results:")
    print(f"Is Safe: {result['is_safe']}")
    print(f"\nSafety Metrics:")
    for metric, value in result['safety_metrics'].items():
        print(f"{metric}: {value}")
    
    print("\nPerformance Metrics:")
    for metric, value in result['performance'].items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()