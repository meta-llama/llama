
import torch
import pytest
from llama.model import RMSNorm, ModelArgs, precompute_freqs_cis, apply_rotary_emb, Attention, FeedForward, TransformerBlock

# Mock classes for parallel linear layers
class MockColumnParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

class MockRowParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def model_args():
    return ModelArgs(
        dim=128,
        n_layers=2,
        n_heads=4,
        vocab_size=32000,
        max_seq_len=2048,
        max_batch_size=16,
        ffn_dim_multiplier=1.0, # Set a specific multiplier for testing
    )


def test_rms_norm(model_args: ModelArgs):
    # Create a random input tensor
    x = torch.randn(model_args.max_batch_size, model_args.max_seq_len, model_args.dim)

    # Initialize the RMSNorm layer
    rms_norm = RMSNorm(model_args.dim)

    # Apply the RMSNorm
    output = rms_norm(x)

    # Check that the output shape is correct
    assert output.shape == x.shape

    # Check that the normalization is correct
    # The expected output is x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
    # Since the weight is initialized to 1, we can ignore it for this test.
    expected_output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(output, expected_output, atol=1e-5)

def test_precompute_freqs_cis(model_args: ModelArgs):
    # Precompute the frequency tensor
    freqs_cis = precompute_freqs_cis(model_args.dim // model_args.n_heads, model_args.max_seq_len * 2)

    # Check the shape of the frequency tensor
    assert freqs_cis.shape == (model_args.max_seq_len * 2, model_args.dim // model_args.n_heads // 2)

    # Check that the values are complex numbers
    assert freqs_cis.is_complex()

def test_apply_rotary_emb(model_args: ModelArgs):
    # Create random query and key tensors
    xq = torch.randn(model_args.max_batch_size, model_args.max_seq_len, model_args.n_heads, model_args.dim // model_args.n_heads)
    xk = torch.randn(model_args.max_batch_size, model_args.max_seq_len, model_args.n_heads, model_args.dim // model_args.n_heads)

    # Precompute the frequency tensor
    freqs_cis = precompute_freqs_cis(model_args.dim // model_args.n_heads, model_args.max_seq_len)
    freqs_cis = freqs_cis[0:model_args.max_seq_len,:]
    # Apply the rotary embeddings
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

    # Check that the output shapes are correct
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape

    # Check that the output is not the same as the input
    assert not torch.allclose(xq_out, xq)
    assert not torch.allclose(xk_out, xk)

def test_attention(model_args: ModelArgs, monkeypatch):
    # Mock the model parallel functions
    monkeypatch.setattr("fairscale.nn.model_parallel.initialize.get_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("fairscale.nn.model_parallel.initialize.get_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("llama.model.ColumnParallelLinear", MockColumnParallelLinear)
    monkeypatch.setattr("llama.model.RowParallelLinear", MockRowParallelLinear)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    # Create an Attention module
    attention = Attention(model_args)

    # Create a random input tensor
    x = torch.randn(model_args.max_batch_size, 1, model_args.dim)

    # Precompute the frequency tensor
    freqs_cis = precompute_freqs_cis(model_args.dim // model_args.n_heads, 1)

    # Run the forward pass
    output = attention.forward(x, 0, freqs_cis, None)

    # Check the output shape
    assert output.shape == (model_args.max_batch_size, 1, model_args.dim)

def test_feed_forward(model_args: ModelArgs, monkeypatch):
    # Mock the parallel linear layers
    monkeypatch.setattr("llama.model.ColumnParallelLinear", MockColumnParallelLinear)
    monkeypatch.setattr("llama.model.RowParallelLinear", MockRowParallelLinear)

    # Create a FeedForward module
    feed_forward = FeedForward(
        dim=model_args.dim,
        hidden_dim=4 * model_args.dim,
        multiple_of=model_args.multiple_of,
        ffn_dim_multiplier=model_args.ffn_dim_multiplier,
    )

    # Create a random input tensor
    x = torch.randn(model_args.max_batch_size, model_args.max_seq_len, model_args.dim)

    # Run the forward pass
    output = feed_forward.forward(x)

    # Check the output shape
    assert output.shape == x.shape

def test_transformer_block(model_args: ModelArgs, monkeypatch):
    # Mock the model parallel functions and layers
    monkeypatch.setattr("fairscale.nn.model_parallel.initialize.get_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("fairscale.nn.model_parallel.initialize.get_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("llama.model.ColumnParallelLinear", MockColumnParallelLinear)
    monkeypatch.setattr("llama.model.RowParallelLinear", MockRowParallelLinear)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    # Create a TransformerBlock module
    transformer_block = TransformerBlock(layer_id=0, args=model_args)

    # Create a random input tensor
    x = torch.randn(model_args.max_batch_size, 1, model_args.dim)

    # Precompute the frequency tensor
    freqs_cis = precompute_freqs_cis(model_args.dim // model_args.n_heads, 1)

    # Run the forward pass
    output = transformer_block.forward(x, 0, freqs_cis, None)

    # Check the output shape
    assert output.shape == (model_args.max_batch_size, 1, model_args.dim)
