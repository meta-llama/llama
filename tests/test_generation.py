import torch
import pytest
from llama.generation import Llama
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer


@pytest.fixture
def model_args():
    return ModelArgs(
        dim=128,
        n_layers=2,
        n_heads=4,
        vocab_size=32000,
        max_seq_len=2048,
        max_batch_size=16,
        ffn_dim_multiplier=1.0,
    )

# A dummy Tokenizer for testing
class DummyTokenizer:
    def __init__(self, model_path):
        self.n_words = 32000
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0

    def encode(self, s, bos, eos):
        return [1, 2, 3, 4, 5]

    def decode(self, t):
        return "dummy text"

@pytest.fixture
def llama(model_args, monkeypatch, tmp_path):
    # Mock the distributed and hardware setup
    monkeypatch.setattr("torch.distributed.init_process_group", lambda backend: None)
    monkeypatch.setattr("fairscale.nn.model_parallel.initialize.initialize_model_parallel", lambda size: None)
    monkeypatch.setattr("torch.cuda.set_device", lambda device: None)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    # Mock the model and tokenizer loading
    monkeypatch.setattr(torch, "load", lambda path, map_location: {})
    monkeypatch.setattr(Tokenizer, "__init__", lambda self, model_path: None)
    monkeypatch.setattr(Transformer, "__init__", lambda self, args: None)
    monkeypatch.setattr(Transformer, "load_state_dict", lambda self, state_dict, strict: None)

    # Create a dummy Llama instance
    generator = Llama(None, None)
    generator.tokenizer = DummyTokenizer(None)
    generator.model = lambda: None # A dummy model
    generator.model.params = model_args

    # Mock the generate method to avoid running the actual generation logic
    def mock_generate(*args, **kwargs):
        return ([[1, 2, 3]], None)

    monkeypatch.setattr(generator, "generate", mock_generate)

    return generator


def test_text_completion(llama):
    # Test the text_completion method
    result = llama.text_completion(["dummy prompt"])
    assert isinstance(result, list)
    assert "generation" in result[0]
    assert result[0]["generation"] == "dummy text"


def test_chat_completion(llama):
    # Test the chat_completion method
    dialogs = [[{"role": "user", "content": "dummy question"}]]
    result = llama.chat_completion(dialogs)
    assert isinstance(result, list)
    assert "generation" in result[0]
    assert result[0]["generation"]["role"] == "assistant"
    assert result[0]["generation"]["content"] == "dummy text"
