
from lm_eval.models.gpt2 import HFLM
import torch


class EvalHarnessAdaptor(HFLM):
    def __init__(
            self,
            device="cuda",
            gpt2=None,
            tokenizer=None,
            batch_size=1,
            temperature: float = 0.8,
            top_p: float = 0.95,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.gpt2 = gpt2
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.n_words
        self.batch_size_per_gpu = batch_size
        self.temperature = temperature
        self.top_p = top_p

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, bos=True, eos=False)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_id

    @property
    def max_length(self):
        return self.gpt2.model.params.max_seq_len

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        return self.gpt2.model.forward(inps, start_pos=0, return_all_logits=True)

    def _model_generate(self, context, max_length, eos_token_id):
        params = self.gpt2.params
        total_len = min(params.max_seq_len, max_length + len(context))

        tokens = torch.full((1, total_len), self.tokenizer.pad_id).cuda().long()
        tokens[:len(context)] = torch.tensor(context).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = len(context)
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.gpt2.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if self.temperature > 0:
                probs = torch.softmax(logits / self.temperature, dim=-1)
                next_token = sample_top_p(probs, self.top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        # cut to max gen len
        tokens = tokens[: len(context) + max_length]
        # cut to eos tok if any
        try:
            tokens = tokens[: tokens.index(self.tokenizer.eos_id)]
        except ValueError:
            pass
        decoded = self.tokenizer.decode(tokens)
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
