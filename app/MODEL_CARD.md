# LLaMA Model Card

## Model details
**Organization developing the model**
The FAIR team of Meta AI.

**Model date**
LLaMA was trained between December. 2022 and Feb. 2023.

**Model version**
This is version 1 of the model.

**Model type**
LLaMA is an auto-regressive language model, based on the transformer architecture. The model comes in different sizes: 7B, 13B, 33B and 65B parameters.

**Paper or resources for more information**
More information can be found in the paper “LLaMA, Open and Efficient Foundation Language Models”, available at https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/.

**Citations details**
https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/

**License**
Non-commercial bespoke license

**Where to send questions or comments about the model**
Questions and comments about LLaMA can be sent via the [GitHub repository](https://github.com/facebookresearch/llama) of the project , by opening an issue.

## Intended use
**Primary intended uses**
The primary use of LLaMA is research on large language models, including:
exploring potential applications such as question answering, natural language understanding or reading comprehension,
understanding capabilities and limitations of current language models, and developing techniques to improve those,
evaluating and mitigating biases, risks, toxic and harmful content generations, hallucinations.

**Primary intended users**
The primary intended users of the model are researchers in natural language processing, machine learning and artificial intelligence.

**Out-of-scope use cases**
LLaMA is a base, or foundational, model. As such, it should not be used on downstream applications without further risk evaluation and mitigation. In particular, our model has not been trained with human feedback, and can thus generate toxic or offensive content, incorrect information or generally unhelpful answers.

## Factors
**Relevant factors**
One of the most relevant factors for which model performance may vary is which language is used. Although we included 20 languages in the training data, most of our dataset is made of English text, and we thus expect the model to perform better for English than other languages. Relatedly, it has been shown in previous studies that performance might vary for different dialects, and we expect that it will be the case for our model.

**Evaluation factors**
As our model is trained on data from the Web, we expect that it reflects biases from this source. We thus evaluated on RAI datasets to measure biases exhibited by the model for gender, religion, race, sexual orientation, age, nationality, disability, physical appearance and socio-economic status. We also measure the toxicity of model generations, depending on the toxicity of the context used to prompt the model.

## Metrics
**Model performance measures**
We use the following measure to evaluate the model:
- Accuracy for common sense reasoning, reading comprehension, natural language understanding (MMLU), BIG-bench hard, WinoGender and CrowS-Pairs,
- Exact match for question answering,
- The toxicity score from Perspective API on RealToxicityPrompts.

**Decision thresholds**
Not applicable.

**Approaches to uncertainty and variability**
Due to the high computational requirements of training LLMs, we trained only one model of each size, and thus could not evaluate variability of pre-training.

## Evaluation datasets
The model was evaluated on the following benchmarks: BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC, OpenBookQA, NaturalQuestions, TriviaQA, RACE, MMLU, BIG-bench hard, GSM8k, RealToxicityPrompts, WinoGender, CrowS-Pairs.

## Training dataset
The model was trained using the following source of data: CCNet [67%], C4 [15%], GitHub [4.5%], Wikipedia [4.5%], Books [4.5%], ArXiv [2.5%], Stack Exchange[2%]. The Wikipedia and Books domains include data in the following languages: bg, ca, cs, da, de, en, es, fr, hr, hu, it, nl, pl, pt, ro, ru, sl, sr, sv, uk. See the paper for more details about the training set and corresponding preprocessing.

## Quantitative analysis
Hyperparameters for the model architecture


<table>
    <thead>
            <tr>
            <th >LLaMA</th> <th colspan=6>Model hyper parameters </th>
            </tr>
            <tr>
            <th>Number of parameters</th><th>dimension</th><th>n heads</th><th>n layers</th><th>Learn rate</th><th>Batch size</th><th>n tokens</th>
            </tr>           
        </thead>
    <tbody>       
        <tr>
            <th>7B</th> <th>4096</th> <th>32</th> <th>32</th> <th>3.0E-04</th><th>4M</th><th>1T 
        </tr>
        <tr>
            <th>13B</th><th>5120</th><th>40</th><th>40</th><th>3.0E-04</th><th>4M</th><th>1T
        </tr>
        <tr>
            <th>33B</th><th>6656</th><th>52</th><th>60</th><th>1.5.E-04</th><th>4M</th><th>1.4T
        </tr>
        <tr>
            <th>65B</th><th>8192</th><th>64</th><th>80</th><th>1.5.E-04</th><th>4M</th><th>1.4T
        </tr>     
    </tbody>
</table>


*Table 1 - Summary of LLama Model Hyperparameters*

We present our results on eight standard common sense reasoning benchmarks in the table below. 
<table>
    <thead>
            <tr>
            <th>LLaMA</th>  <th colspan=9>Reasoning tasks </th>
            </tr>
            <tr>
            <th>Number of parameters</th> <th>BoolQ</th><th>PIQA</th><th>SIQA</th><th>HellaSwag</th><th>WinoGrande</th><th>ARC-e</th><th>ARC-c</th><th>OBQA</th><th>COPA</th>
            </tr>           
        </thead>
    <tbody>    
    <tr>   
        <th>7B</th><th>76.5</th><th>79.8</th><th>48.9</th><th>76.1</th><th>70.1</th><th>76.7</th><th>47.6</th><th>57.2</th><th>93
        </th>   
    <tr><th>13B</th><th>78.1</th><th>80.1</th><th>50.4</th><th>79.2</th><th>73</th><th>78.1</th><th>52.7</th><th>56.4</th><th>94
</th>
    <tr><th>33B</th><th>83.1</th><th>82.3</th><th>50.4</th><th>82.8</th><th>76</th><th>81.4</th><th>57.8</th><th>58.6</th><th>92
</th>
    <tr><th>65B</th><th>85.3</th><th>82.8</th><th>52.3</th><th>84.2</th><th>77</th><th>81.5</th><th>56</th><th>60.2</th><th>94</th></tr> 
    </tbody>
</table>

*Table 2 - Summary of LLama Model Performance on Reasoning tasks*


We present our results on bias in the table below. Note that lower value is better indicating lower bias. 


| No  | Category             | FAIR LLM |
| --- | -------------------- | -------- |
| 1   | Gender               | 70.6     |
| 2   | Religion             | 79       |
| 3   | Race/Color           | 57       |
| 4   | Sexual orientation   | 81       |
| 5   | Age                  | 70.1     |
| 6   | Nationality          | 64.2     |
| 7   | Disability           | 66.7     |
| 8   | Physical appearance  | 77.8     |
| 9   | Socioeconomic status | 71.5     |
|     | LLaMA Average        | 66.6     |

*Table 3 - Summary bias of our model output*



## Ethical considerations
**Data**
The data used to train the model is collected from various sources, mostly from the Web. As such, it contains offensive, harmful and biased content. We thus expect the model to exhibit such biases from the training data.

**Human life**
The model is not intended to inform decisions about matters central to human life, and should not be used in such a way.

**Mitigations**
We filtered the data from the Web based on its proximity to Wikipedia text and references. For this, we used a Kneser-Ney language model and a fastText linear classifier.

**Risks and harms**
Risks and harms of large language models include the generation of harmful, offensive or biased content. These models are often prone to generating incorrect information, sometimes referred to as hallucinations. We do not expect our model to be an exception in this regard.

**Use cases**
LLaMA is a foundational model, and as such, it should not be used for downstream applications without further investigation and mitigations of risks. These risks and potential fraught use cases include, but are not limited to: generation of misinformation and generation of harmful, biased or offensive content.
