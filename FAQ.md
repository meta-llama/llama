**Q: If I’m a developer/business, how can I access it?**


A: Details on how to access the models are available on our website [link](http://ai.meta.com/llama). Please note that the models are subject to the [acceptable use policy](https://github.com/facebookresearch/llama/blob/main/USE_POLICY.md) and the provided [responsible use guide](https://ai.meta.com/static-resource/responsible-use-guide/). 

**Q: Where can the models be found?**

A: 
- Models are available through multiple sources but the place to start is at https://ai.meta.com/llama/ 
- Model code, quickstart guide and fine-tuning examples are available through our [Github Llama repository](https://github.com/fairinternal/llama_v2). Model Weights are available through an email link after the user submits a sign-up form. 
- Models are also being hosted by Microsoft, Amazon Web Services, and Hugging Face, and may also be available through other hosting providers in the future.

**Q: Can anyone access Llama 2? What are the terms?**

A: 
- Llama 2 is broadly available to developers and licensees through a variety of hosting providers and on the Meta website.
- Llama 2 is licensed under the Llama 2 Community License Agreement, which provides a permissive license to the models along with certain restrictions to help ensure that the models are being used responsibly.

**Q: What’s different about Llama 2 from Llama 1?**

A: 
- We received unprecedented interest in the Llama 1 model we released for the research community –  more than 100,000 individuals and organizations have applied for access to Llama 1 and tens of thousands are now using it to innovate. After external feedback, fine tuning, and extensive safety evaluations, we made the decision to release the next version of Llama more broadly. 
- Llama 2 is also available under a permissive commercial license, whereas Llama 1 was limited to non-commercial use.
- Llama 2 is capable of processing longer prompts than Llama 1 and is also designed to work more efficiently.
- For Llama 2 we’re pairing our release of our pretrained models with versions fine-tuned for helpfulness and safety. Sharing fine-tuned versions makes it easier to use our models while also improving safety performance.

**Q: What if I want to access Llama 2 models but I’m not sure if my use is permitted under the Llama 2 Community License?** 

A: On a limited case by case basis, we will consider bespoke licensing requests from individual entities. Please contact llama2@meta.com to provide more details about your request. 

**Q: Where did the data come from to train the models? Was any Meta user data leveraged for training the models?**

A: 
- A combination of sources are used for training. These sources include information that is publicly available online and annotated data to train our models.
- Llama 2 is not trained on Meta user data. 


**Q:  Why are you not sharing the training datasets for Llama 2?** 

A: We believe developers will have plenty to work with as we release our model weights and starting code for pre-trained and conversational fine-tuned versions as well as responsible use resources. While data mixes are intentionally withheld for competitive reasons, all models have gone through Meta’s internal Privacy Review process to ensure responsible data usage in building our products. We are dedicated to the responsible and ethical development of our genAI products, ensuring our policies reflect diverse contexts and meet evolving societal expectations.


**Q: Did we use human annotators to develop the data for our models?**

A: Yes. There are more details about our use of human annotators in the [research paper](https://arxiv.org/abs/2307.09288). 

**Q: Can I use the output of the models to improve the Llama 2 family of models, even though I cannot use them for other LLMs?**

A: It's correct that the license restricts using any part of the Llama 2 models, including the response outputs to train another AI model (LLM or otherwise). However, one can use the outputs to further train the Llama 2 family of models. Techniques such as Quantized Aware Training (QAT) utilize such a technique and hence this is allowed. 


**Q: What is Llama 2's max sequence length?**

A: 
4096. If you want to use more tokens, you will need to fine-tune the model so that it supports longer sequences. More information and examples on fine tuning can be found in the [Llama Recipes repository](https://github.com/facebookresearch/llama-recipes). 


**Q: Is there a multi-lingual checkpoint for researchers to download?**

A: 
The Llama models thus far have been mainly focused on the English language. We are looking at true multi-linguality for the future but for now there are a lot of community projects that fine tune Llama models to support languages.

**Q: How do can we fine tune the Llama 2 models?**

A: 
You can find examples on how to fine tune the Llama 2 models in the [Llama Recipes repository](https://github.com/facebookresearch/llama-recipes). 

**Q: How can I pretrain the Llama 2 models?**

A: 
You can adapt the finetuning script found [here](https://github.com/facebookresearch/llama-recipes/blob/main/llama_finetuning.py) for pretraining. You can also find the hyperparams used for pretraining in Section 2 of [the LLama 2 paper](https://arxiv.org/pdf/2307.09288.pdf).
