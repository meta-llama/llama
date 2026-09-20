<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

## **仓库弃用与生态迁移说明 (Note of deprecation)**

感谢您使用 Llama 系列模型进行开发。作为 Llama 3.1 发布的一部分，随着我们将 Llama 的功能扩展为端到端的 Llama Stack 体系，我们对 GitHub 仓库进行了整合并新增了若干核心项目。后续开发与使用请移步至以下仓库：
- [llama-models](https://github.com/meta-llama/llama-models) - 基础模型核心仓库，包含基础实用工具、模型卡片 (Model Cards)、开源协议及可接受使用政策
- [PurpleLlama](https://github.com/meta-llama/PurpleLlama) - Llama Stack 的安全基石，专注安全风险评估与推理阶段风险缓解机制
- [llama-toolchain](https://github.com/meta-llama/llama-toolchain) - 模型开发（推理/微调/安全防护罩/合成数据生成）接口与官方标准实现
- [llama-agentic-system](https://github.com/meta-llama/llama-agentic-system) - 端到端独立 Llama Stack 系统及专有底层接口，助力构建 Agent 智能体应用
- [llama-cookbook](https://github.com/meta-llama/llama-recipes) - 社区驱动的开发脚本、教程与丰富生态集成

若有任何疑问，欢迎在上述对应仓库中提交 Issue，我们将竭诚为您提供及时的答复。

特此鸣谢！


# (已废弃) Llama 2

我们正在释放大语言模型的无穷潜力。Llama 2 现已面向个人开发者、创作者、科研人员以及各类规模的企业全面开放，助力大家以负责任的方式开展实验、探索创新并推动业务落地。

本次发布包含预训练（Pre-trained）与微调（Fine-tuned）Llama 大语言模型的权重与起始代码，参数量涵盖 7B 至 70B。

本代码库旨在提供一个用于加载 [Llama 2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) 模型并运行原生推理的极简示例。如需查阅基于 Hugging Face 生态的更详尽示例，请参阅 [llama-cookbook](https://github.com/facebookresearch/llama-recipes/)。

## 发布后更新动态

详见 [UPDATES.md](UPDATES.md)。关于持续更新的常见问题解答，请参见[此处 FAQ](https://ai.meta.com/llama/faq/)。

## 模型权重下载

如需下载模型权重与分词器（Tokenizer），请访问 [Meta 官方网站](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) 并接受开源许可协议。

申请获批后，您将通过电子邮件收到一个带有效期的签名临时链接（Signed URL）。随后运行 `download.sh` 脚本，在提示时输入该链接即可开始下载。

前置环境准备：确保系统中已安装 `wget` 和 `md5sum`。随后执行脚本：`./download.sh`。

请注意，下载链接在 24 小时后或达到指定下载次数后失效。若出现类似 `403: Forbidden` 的错误，可随时重新申请链接。

### 通过 Hugging Face 获取

我们同时在 [Hugging Face](https://huggingface.co/meta-llama) 上提供模型下载。您可以通过在对应仓库的模型卡片中确认许可协议并填写申请表单来获取访问权限。提交后，通常在 1 小时内即可获得该版本所有 Llama 模型（Code Llama、Llama 2 或 Llama Guard）的下载授权。

## 快速入门 (Quick Start)

您可以按照以下步骤快速上手并运行 Llama 2 模型。这些步骤支持在本地运行快速推理。更多高级用法请参阅 [Llama 2 Cookbook 仓库](https://github.com/facebookresearch/llama-recipes)。

1. 在配置好 PyTorch / CUDA 的 Conda 环境中克隆并下载本仓库。

2. 在项目根目录下执行：
    ```bash
    pip install -e .
    ```
3. 访问 [Meta 官方网站](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) 注册并申请下载模型。

4. 注册成功后，您将收到一封包含下载 URL 的邮件。运行 `download.sh` 脚本时需要填入此链接。

5. 收到邮件后，进入下载好的 `llama` 仓库目录并执行 `download.sh` 脚本：
    - 确保为 `download.sh` 赋予可执行权限（如 `chmod +x download.sh`）
    - 在脚本执行过程中，按提示粘贴邮件中的下载链接
    - 请勿使用浏览器的“复制链接地址”选项，请务必直接从邮件文本中手动复制完整链接。

6. 所需模型下载完成后，可通过以下命令在本地运行模型推理：
```bash
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```
**说明**
- 将 `llama-2-7b-chat/` 替换为您本地存放权重的目录路径，将 `tokenizer.model` 替换为分词器模型路径。
- `–nproc_per_node` 应设置为对应模型的[模型并行度 (MP)](#模型推理-inference) 参数值。
- 根据硬件显存情况灵活调整 `max_seq_len` 和 `max_batch_size`。
- 本示例运行的是仓库内的 [example_chat_completion.py](example_chat_completion.py)，您也可以根据需要替换为自定义的 Python 脚本。

## 模型推理 (Inference)

不同规模的模型需要配置不同的模型并行度 (Model-Parallel, MP) 参数：

| 模型规格 | MP (并行度) |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 70B    | 8  |

所有模型原生支持最高 4096 个 Token 的上下文长度，但系统会根据 `max_seq_len` 和 `max_batch_size` 预先分配 KV-Cache 显存。因此请务必结合您的硬件显存实际进行合理配置。

### 预训练基座模型 (Pretrained Models)

预训练基座模型未针对对话或问答进行指令对齐。调用时应构造合适的前缀 Prompt，使模型能够顺畅地进行文本自然续写。

示例请参见 `example_text_completion.py`。以 llama-2-7b 模型为例，运行命令如下（`nproc_per_node` 需设为对应的 `MP` 值）：

```bash
torchrun --nproc_per_node 1 example_text_completion.py \
    --ckpt_dir llama-2-7b/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 128 --max_batch_size 4
```

### 微调对话模型 (Fine-tuned Chat Models)

微调模型专为多轮人机对话应用而训练。为充分发挥其特性与性能，必须遵循 [`chat_completion`](https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212) 中定义的特定 Prompt 格式规范，包括 `[INST]` 与 `<<SYS>>` 标签、`BOS` 与 `EOS` 特殊标记，以及各段落之间的换行与空格排版（推荐在输入文本上调用 `strip()` 以避免出现多余的双空格）。

您还可以接入额外的分类器模型以过滤潜在的不安全输入与输出。关于如何在推理链路的输入输出端添加安全检测器，请参考 llama-cookbook 中的[示例代码](https://github.com/facebookresearch/llama-recipes/blob/main/examples/inference.py)。

基于 llama-2-7b-chat 的运行示例：

```bash
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```

Llama 2 是一项前沿技术，在实际应用中具有潜在风险。现阶段所完成的测试无法（也不可能）覆盖所有实际落地场景。
为协助开发者积极防范与化解此类风险，我们制定了[《负责任使用指南》(Responsible Use Guide)](Responsible-Use-Guide.pdf)。更多安全与红队测试细节也可参阅我们的学术论文。

## 问题反馈与支持 (Issues)

如遇软件 Bug 或对模型有相关问题，请通过以下渠道反馈：
- 模型相关问题反馈：[github.com/facebookresearch/llama](http://github.com/facebookresearch/llama)
- 模型生成风险内容举报：[developers.facebook.com/llama_output_feedback](http://developers.facebook.com/llama_output_feedback)
- 安全漏洞与漏洞赏金报告：[facebook.com/whitehat/info](http://facebook.com/whitehat/info)

## 模型卡片 (Model Card)
详见 [MODEL_CARD.md](MODEL_CARD.md)。

## 开源协议 (License)

秉承开放包容的开源理念，我们的模型与权重同时面向科研工作者与商业机构开放授权。我们的使命是通过这一机遇赋能个人与产业界，共同营造探索创新与践行负责任 AI 的繁荣生态。

详见 [LICENSE](LICENSE) 许可证文件以及配套的[《可接受使用政策》(Acceptable Use Policy)](USE_POLICY.md)。

## 参考资料 (References)

1. [Llama 2 学术论文](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
2. [Llama 2 技术概览](https://ai.meta.com/resources/models-and-libraries/llama)
3. [开放创新 AI 研究社区 (Open Innovation AI Research Community)](https://ai.meta.com/llama/open-innovation-ai-research-community/)

关于各类常见技术疑问，请查阅持续更新的 [官方 FAQ](https://ai.meta.com/llama/faq/)。

## 原始初代 Llama (Original Llama)
初代 Llama 发布的原始代码位于 [`llama_v1`](https://github.com/facebookresearch/llama/tree/llama_v1) 分支。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月20日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
