# 8/7/23 Updates

## System Prompt Update

### Observed Issue
We received feedback from the community on our prompt template and we are providing an update to reduce the false refusal rates seen. False refusals occur when the model incorrectly refuses to answer a question that it should, for example due to overly broad instructions to be cautious in how it provides responses. 

### Updated approach
Based on evaluation and analysis, we recommend the removal of the system prompt as the default setting.  Pull request [#626](https://github.com/facebookresearch/llama/pull/626) removes the system prompt as the default option, but still provides an example to help enable experimentation for those using it. 

## Token Sanitization Update

### Observed Issue
The PyTorch scripts currently provided for tokenization and model inference allow for direct prompt injection via string concatenation. Prompt injections allow for the addition of special system and instruction prompt strings from user-provided prompts. 

As noted in the documentation, these strings are required to use the fine-tuned chat models. However, prompt injections have also been used for manipulating or abusing models by bypassing their safeguards, allowing for the creation of content or behaviors otherwise outside the bounds of acceptable use. 

### Updated approach
We recommend sanitizing [these strings](https://github.com/facebookresearch/llama#fine-tuned-chat-models) from any user provided prompts. Sanitization of user prompts mitigates malicious or accidental abuse of these strings. The provided scripts have been updated to do this. 

Note: even with this update safety classifiers should still be applied to catch unsafe behaviors or content produced by the model. An [example](https://github.com/facebookresearch/llama-recipes/blob/main/inference/inference.py) of how to deploy such a classifier can be found in the llama-recipes repository.
