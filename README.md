随着chatgpt、DALLE、Stable Diffusion的火爆崛起，AIGC已经慢慢融入到我们的日常生活工作中。
本项目从模型，数据，计算框架和相关链接四个部分持续更新业界最新进展（以中文为主），并根据自己的使用体验进行推荐。


# 模型

**产品类：**

基本上都是采用注册->排队等待->试用->购买的流程

+ `文本生成`: 国际上ChatGPT 最好，国内中 文心一言使用最好
+ `图片生成`: Midjourney 较好且容易上手
+ `其他`: 其他方向竞品较少，Copilot 代码编写很好用，ChatAvatar 在3D生成时效果一般


| 产品名称     | 发布公司     | 链接                                                             | 功能                                 | 使用体验                                                     |
| ------------ | ------------ | ---------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| ChatGPT      | openai       | [链接](https://chat.openai.com/)                                | 基于GPT3支持文本的问答，基于GPT4支持图文输入，回答文本 | 文本类的第一名，虽然中文问答的能力不如英文（毕竟训练的时候英文占了96%），但依旧效果很好 |
| Midjourney   | Midjourney   | [链接](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F) | 通过文字生成图片                     | 下限很低，第一次用也能生成很好的图片，不过现在不免费了 |
| DALLE        | openai       | [链接](https://labs.openai.com/)                                | 根据文字生成图片                     | 还可以，就是生成了很多简笔画画风的图片                   |
| Copilot      | github、openai | [链接](https://docs.github.com/zh/copilot/quickstart)           | 写代码时的好帮手，可以提供整行或整个函数的建议 | 挺好用的，常用功能基本都能编辑对 |
| 文心一言     | 百度         | [链接](https://yiyan.baidu.com/)                                | 输入文本，输出文本，图片，视频。其中视频在发布会上展示了，但是页面还没支持 | 比ChatGPT差一点，但目前体验下来，可能是国内最好的大模型了 |
| 通义千问     | 阿里         | [链接](https://tongyi.aliyun.com/)                              | 一个专门响应人类指令的大模型         | 申请后一直没通过                                             |
| 日日新       | 商汤         | [链接](https://lm_experience.sensetime.com/document/chat)      | 作为CV四小龙，除了支持文本，还把图片生成，图片检测、分类等功能加进来了 | 申请后一直没通过                                             |
| 星火         | 科大讯飞     | [链接](https://xinghuo.xfyun.cn/)                                | 作为语音头部企业，额外支持了语音问答及回复。在发布会上除了发布大模型，还发布了四款基于大模型应用的产品 | 实际测的时候，没有发布会上表现的好，个人感觉没有文心一言好 |
| MathGPT      | 学而思       | [链接](https://36kr.com/p/2248754889289350)                     | 作为一家教育机构，主要从数学领域出发构建大模型 | 还没发布，持续跟进中                                         |
| 式说         | 第四范式       |  [链接](http://www.4paradigm.com/product/SageGPT.html)                     | 除了常规功能外，还支持本地部署，可以融合企业内部知识 | 还未体验                                         |	
| ChatAvatar     | 影眸科技   | [链接](https://hyperhuman.deemos.com/chatavatar) | 系统先通过多轮对话确定prompt，再生成3D | 还可以，3D生成是一个比较难的任务，虽然没有达到预期，但在竞品里还是挺不错的 |

**自然语言处理模型：**

可以使用接口推荐ChatGPT，不能使用接口时推荐ChatGLM，两者都可以推荐ChatGPT

| 模型    | 主模型      | 模型大小  | 简介                                     | 链接                                                        |
|---------|-------------|-----------|------------------------------------------|------------------------------------------------------------|
| ChatGPT | GPT3/GPT4   | 基于openai的api | 引领大模型的爆款产品                        | [链接](https://chat.openai.com/)                 |
| autogpt | GPT3/GPT4   | 基于openai的api | 将LLM的 "思想" 串联起来，自主实现设定的目标 | [链接](https://github.com/Significant-Gravitas/Auto-GPT) |
| ChatGLM               | GLM            | 6B，支持量化       | 虽然模型只有6B，但是效果还是很好的           | [链接](https://github.com/THUDM/ChatGLM-6B)                                             |
| Phoenix               | BLOOMZ         | 7B，支持量化       | 提供了训练脚本                              | [链接](https://github.com/FreedomIntelligence/LLMZoo)                                   |
| PaLM-rlhf-pytorch     | PaLM           |                    | 在 PaLM 架构之上实施 RLHF                    | [链接](https://github.com/lucidrains/PaLM-rlhf-pytorch)                                 |
| Dolly/Dolly2.0        | pythia         | 12B，6.9B，2.8B    | 使用databricks-dolly-15k进行训练，但是模型有点大，在24G的卡上也没有办法使用12B的完整版 | [链接](https://github.com/databrickslabs/dolly)                                        |
| ChatYuan              | T5             | 0.22B              | 可以在消费级显卡、 PC甚至手机上进行推理        | [链接](https://github.com/clue-ai/ChatYuan)                                             |
| Chinese-ChatLLaMA     | LLaMA          | 7B, 13B, 33B, 65B |                                             | [链接](https://github.com/chenfeng357/open-Chinese-ChatLLaMA)                           |
| OpenChatKit           | Pythia         | 7B, 20B            | 增加维基百科信息扩充上下文                  | [链接](https://github.com/togethercomputer/OpenChatKit)                                 |
| BELLE                 | LLaMA，BLOOMZ  | 7B                 | 支持finetune、lora，并在LLaMA7B基础上增量预训练扩展中文词表的模型                      | [链接](https://github.com/LianjiaTech/BELLE)                                             |
| Wombat                | LLaMA          | 7B, 13B, 33B, 68B | 把RLHF改成RRHF了，模型量和超参数量降低了，但效果相当                                  | [链接](https://github.com/GanjinZero/RRHF)                                               |
| stanford_alpace       | LLaMA            | 7B, 13B, 33B, 69B | 构造了一个生成“指令遵循数据”的 pipeline，拿chatgpt的结果进行训练 | [链接](https://github.com/tatsu-lab/stanford_alpaca)    |
| Chinese-LLaMA-Alpaca  | LLaMA, Alpaca    | 7B, 13B          | 扩充了中文词表，并用中文进行了训练                             | [链接](https://github.com/ymcui/Chinese-LLaMA-Alpaca)   |
| EasyLM                | LLaMA, GPT-J, roberta | 7B, 13B, 33B, 71B |                                                              | [链接](https://github.com/young-geng/EasyLM)            |
| Chinese-alpaca-lora   | LLaMA            | 7B               |                                                              | [链接](https://github.com/LC1332/Chinese-alpaca-lora)   |
| Chinese-Vicuna        | LLaMA            | 7B, 13B          | 突出小，一块2080Ti就够训练。数据长度在256以内，大约需要9G显存。  | [链接](https://github.com/Facico/Chinese-Vicuna)        |
| LMFLOW                | LLaMA            | 7B, 33B          | 用于微调和推理的可扩展工具包，适合所有大型模型                 | [链接](https://github.com/OptimalScale/LMFlow)          |
| StackLLaMA            | LLaMA            | 7B               |                                                              | [链接](https://huggingface.co/trl-lib/llama-7b-se-rl-peft) |


**计算机视觉模型：**

+ `生成`: 推荐Stable Diffusion（SD），但SD对prompt编写要求有点高，使用者为初次使用文生图，推荐DALLE 2
+ `分割`: SAM不支持语意，分割效果还可以。Grounded-Segment-Anything 还未测试
+ `3D`: 待测试


| 模型                      | 分类     | 输入输出类型        | 简介                                              | 链接                                                   |
|--------------------------|----------|--------------------|---------------------------------------------------|--------------------------------------------------------|
| DALLE 2                    | 图片生成 | 文本 -> 图片        | openai发表的，可以通过api接口调用                  | [链接](https://labs.openai.com/)                       |
| Stable Diffusion         | 图片生成 | 文本 -> 图片        | 下限低但上限高，可以自己训练，prompt写的不好会生成比较差的结果，白种人的生成效果比黄种人的好 | [链接](https://github.com/AUTOMATIC1111/stable-diffusion-webui) |
| StableStudio         | 图片生成 | 文本 -> 图片        | StableStudio 是 Stability AI 的 DreamStudio 的官方开源变体，允许用户创建和编辑生成的图像 | [链接](https://github.com/Stability-AI/StableStudio) |
| SAM                      | 分割     | 图片 -> 图片        | 效果还行，看起来有prompt的字段，但目前还不支持输入文字，把图片中对应的物体分割出来 | [链接](https://github.com/facebookresearch/segment-anything) |
| Grounded-Segment-Anything| 分割     | 图片/视频+文本 -> 图片、视频 | 集成了很多模型，可以检测、分割、生成带有图像、文本和音频的输入 | [链接](https://github.com/IDEA-Research/Grounded-Segment-Anything) |
| DetGPT | 分割     | 图片+文本 -> 图片 | 结合了语言+视觉+推理+定位的功能 | [链接](https://github.com/OptimalScale/DetGPT) |
| Real-ESRGAN              | 修复     | 图片 -> 图片        | 图像/视频修复，高清图片生成，目前主要针对动漫领域的 | [链接](https://github.com/xinntao/Real-ESRGAN)        |
| Shap-E                   | 3D生成   | 文本 or 图片 -> 3D | openai出品，根据文本或图像，生成 3D对象            | [链接](https://github.com/openai/shap-e)               |
| stable-dreamfusion       | 3D生成   | 文本 or 图片 -> 4D | 根据文本或图像，生成 3D对象                        | [链接](https://github.com/ashawkey/stable-dreamfusion) |

**多模态模型：**

可以使用接口推荐GPT4，不能使用接口时看场景，BLIP-2 支持的场景更多一些

| 模型        | 分类       | 输入输出类型             | 模型大小    | 简介                                                             | 链接                                                            |
|-------------|------------|-------------------------|-------------|------------------------------------------------------------------|-----------------------------------------------------------------|
| GPT4        | 文本生成   | 文本 + 图片 -> 文本      | 100万亿     | 应该是当前最先进的多模态大模型了，可以开plus使用                 | [链接](https://openai.com/product/gpt-4)                        |
| LLaVA       | 文本生成   | 文本 + 图片 -> 文本      | 7B,13B      | 提供示例平台，效果还可以                                         | [链接](https://github.com/haotian-liu/LLaVA)                    |
| BLIP-2      | 常用功能基本都支持了，包括文图检索、VQA、VideoQA、Image Captioning等 | 文本、图片、视频 -> 文本、图片 | 看具体使用模型 | 已经形成了python库，集成了ALBEF, BLIP, CLIP等模型              | [链接](https://github.com/salesforce/LAVIS?ref=blog.salesforceairesearch.com) |
| OpenFlamingo | 文本生成   | 文本 + 图片 -> 文本      | 9B          | 用于训练大型多模态模型的开源框架，支持支持 LLaMA、OPT、GPT-Neo、GPT-J 和 Pythia 模型 | [链接](https://github.com/mlfoundations/open_flamingo) |
| MiniGPT-4 | 文本生成   | 文本 + 图片 -> 文本      | 7B          | 用一个projection layer 把冻结的视觉编码器和Vicuna对齐。 | [链接](https://github.com/Vision-CAIR/MiniGPT-4) |
| mPLUG-Owl | 文本生成   | 文本 + 图片 -> 文本      | 7B          | 达摩院出的，与LLaVA和MiniGPT-4不同，没有冻结基础编码器 | [链接](https://github.com/X-PLUG/mPLUG-Owl) |
| ImageBind | 特征提取   | 六种模态      |           | 学习跨六种不同模态的联合嵌入，包括：图像、文本、音频、深度（3D）、红外辐射 和 IMU 数据 | [链接](https://github.com/facebookresearch/ImageBind) |

# 计算框架

DeepSpeed + Megatron 的综合运用是当前主流，可以数据并行、模型并行、pipeline并行

| 产品名称      | 发布公司     | 链接                                            | 功能                                                                                     |
|---------------|--------------|------------------------------------------------|------------------------------------------------------------------------------------------|
| DeepSpeed     | 微软         | [链接](https://github.com/microsoft/DeepSpeed) | 高效的分布式训练、推理框架，并遵循 InstructGPT 论文的方法（chatgpt也是类似方法），整合端到端的训练流程 |
| Megatron      | 英伟达       | [链接](https://github.com/NVIDIA/Megatron-LM)  | 把模型并行（张量、序列和流水线）和多节点预训练，应用到基于 Transformer 的模型中，如：GPT、BERT 和 T5等 |
| Colossal-AI   | HPC-AI Tech  | [链接](https://github.com/hpcaitech/ColossalAI) | 提供并行组件，只需几行即可启动分布式训练和推理                                              |

# 数据

数据量重要，但数据质量可能更重要


| 数据名称              | 类型       | 简介                                                                                                                                      | 数据量                     | 链接                                                                                                                            |
|---------------------|------------|-------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| WuDaoCorpora        | 文本       | 采用20多种规则从100TB原始网页数据中清洗得出最终数据集，注重隐私数据信息的去除，源头上避免GPT-3存在的隐私泄露风险；包含教育、科技等50+个行业数据标签，可以支持多领域预训练模型的训练。 | 数据总量：5TB；开源数量：200GB | [链接](https://data.baai.ac.cn/details/WuDaoCorporaText)                                     |
| BELLE项目           | 文本       | 参考Stanford Alpaca 生成的中文数据集1M + 0.5M；还有一个 10M的文件持续更新                                                                           | 1.5M                       | [链接](https://github.com/LianjiaTech/BELLE/tree/main/data)                           |
| alpaca_gpt4_data_zh | 文本       | 由 GPT4 生成，其中提示是由 ChatGPT 翻译成中文                                                                                             | 52K                        | [链接](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json) |
| LLaVA-Instruct-150K | 图文对     | 是基于GPT4 生成的多模态提示数据                                                                                                            | 150K                       | [链接](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)       |
| laion2B-en          | 图文对     | 从网页数据Common Crawl中筛选出来的图像-文本对数据集，文本是英文的                                                                                 | 数据量为2.32B               | [链接](https://huggingface.co/datasets/laion/laion2B-en)                                 |
| mmc4                | 图文对     | 是对流行的纯文本c4语料库的扩充                                                                                                             | 包含103M个文档，其中包含585M个图像和43B个英语标记 | [链接](https://github.com/allenai/mmc4)                                                               |
| SA-1B               | 图像及mask标注 | 由 11M 图像和 1.1B mask 注释组成                                                                                                           | 10.5GB                     | [链接](https://ai.facebook.com/datasets/segment-anything-downloads/)       |

# 相关链接


| 名称                          | 简介                                                             | 链接                                                                             |
|-------------------------------|------------------------------------------------------------------|----------------------------------------------------------------------------------|
| civitai                       | 可以下载StableDiffusion相关模型                                     | [链接](https://civitai.com/)                                                    |
| gradio                        | 快速的为训练好的模型创建前端页面                                     | [链接](https://github.com/gradio-app/gradio)                                   |
| awesome-chatgpt-prompts-zh    | ChatGPT 中文调教指南。各种场景使用指南。学习怎么让它听你的话        | [链接](https://github.com/PlexPt/awesome-chatgpt-prompts-zh)                    |
| MultiModal-AI-Chatbot         | 多模态对话机器人，支持chatgpt、chatgpt平替、SD等模型进行图文生成，可使用在终端、微信公众号、Web等应用上 | [链接](https://github.com/liubo12/MultiModal-AI-Chatbot) |
