
import os
import pytest
from llama.tokenizer import Tokenizer

# A dummy SentencePieceProcessor for testing
class DummySentencePieceProcessor:
    def __init__(self, model_file=None):
        self.vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "h": 3, "e": 4, "l": 5, "o": 6, "w": 7, "r": 8, "d": 9}

    def vocab_size(self):
        return len(self.vocab)

    def bos_id(self):
        return self.vocab["<s>"]

    def eos_id(self):
        return self.vocab["</s>"]

    def pad_id(self):
        return -1

    def encode(self, s: str) -> list[int]:
        return [self.vocab.get(c, self.vocab["<unk>"]) for c in s]

    def decode(self, t: list[int]) -> str:
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return "".join([reverse_vocab.get(i, "<unk>") for i in t])

@pytest.fixture
def tokenizer(monkeypatch) -> Tokenizer:
    # Create a dummy tokenizer and monkeypatch it into the Tokenizer class
    dummy_sp = DummySentencePieceProcessor()
    monkeypatch.setattr(Tokenizer, "__init__", lambda self, model_path: None)
    tokenizer = Tokenizer(model_path="dummy")
    tokenizer.sp_model = dummy_sp
    tokenizer.n_words = dummy_sp.vocab_size()
    tokenizer.bos_id = dummy_sp.bos_id()
    tokenizer.eos_id = dummy_sp.eos_id()
    tokenizer.pad_id = dummy_sp.pad_id()
    return tokenizer

def test_encode(tokenizer: Tokenizer):
    # Test encoding of a simple string
    encoded = tokenizer.encode("hello", bos=False, eos=False)
    assert encoded == [3, 4, 5, 5, 6]

    # Test encoding with BOS token
    encoded_with_bos = tokenizer.encode("hello", bos=True, eos=False)
    assert encoded_with_bos == [1, 3, 4, 5, 5, 6]

    # Test encoding with EOS token
    encoded_with_eos = tokenizer.encode("hello", bos=False, eos=True)
    assert encoded_with_eos == [3, 4, 5, 5, 6, 2]

    # Test encoding with both BOS and EOS tokens
    encoded_with_bos_eos = tokenizer.encode("hello", bos=True, eos=True)
    assert encoded_with_bos_eos == [1, 3, 4, 5, 5, 6, 2]

def test_decode(tokenizer: Tokenizer):
    # Test decoding of a simple list of tokens
    decoded = tokenizer.decode([3, 4, 5, 5, 6])
    assert decoded == "hello"

    # Test decoding with BOS and EOS tokens
    decoded_with_bos_eos = tokenizer.decode([1, 3, 4, 5, 5, 6, 2])
    assert decoded_with_bos_eos == "<s>hello</s>"
