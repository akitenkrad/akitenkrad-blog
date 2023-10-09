---
draft: false
title: "arXiv @ 2023.10.06"
date: 2023-10-06
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.06"
    identifier: arxiv_20231006
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (23)](#cscl-23)
- [cs.IR (1)](#csir-1)
- [cs.LG (36)](#cslg-36)
- [cs.CY (3)](#cscy-3)
- [cs.CR (2)](#cscr-2)
- [cs.CV (26)](#cscv-26)
- [cs.SE (2)](#csse-2)
- [eess.IV (4)](#eessiv-4)
- [stat.ML (4)](#statml-4)
- [cs.RO (7)](#csro-7)
- [eess.AS (2)](#eessas-2)
- [eess.SY (1)](#eesssy-1)
- [math.HO (1)](#mathho-1)
- [cs.HC (1)](#cshc-1)
- [cs.SD (2)](#cssd-2)
- [cs.AI (1)](#csai-1)
- [cs.SI (1)](#cssi-1)

## cs.CL (23)



### (1/117) On the Performance of Multimodal Language Models (Utsav Garg et al., 2023)

{{<citation>}}

Utsav Garg, Erhan Bas. (2023)  
**On the Performance of Multimodal Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.03211v1)  

---


**ABSTRACT**  
Instruction-tuned large language models (LLMs) have demonstrated promising zero-shot generalization capabilities across various downstream tasks. Recent research has introduced multimodal capabilities to LLMs by integrating independently pretrained vision encoders through model grafting. These multimodal variants undergo instruction tuning, similar to LLMs, enabling effective zero-shot generalization for multimodal tasks. This study conducts a comparative analysis of different multimodal instruction tuning approaches and evaluates their performance across a range of tasks, including complex reasoning, conversation, image captioning, multiple-choice questions (MCQs), and binary classification. Through rigorous benchmarking and ablation experiments, we reveal key insights for guiding architectural choices when incorporating multimodal capabilities into LLMs. However, current approaches have limitations; they do not sufficiently address the need for a diverse multimodal instruction dataset, which is crucial for enhancing task generalization. Additionally, they overlook issues related to truthfulness and factuality when generating responses. These findings illuminate current methodological constraints in adapting language models for image comprehension and provide valuable guidance for researchers and practitioners seeking to harness multimodal versions of LLMs.

{{</citation>}}


### (2/117) Can Language Models Employ the Socratic Method? Experiments with Code Debugging (Erfan Al-Hossami et al., 2023)

{{<citation>}}

Erfan Al-Hossami, Razvan Bunescu, Justin Smith, Ryan Teehan. (2023)  
**Can Language Models Employ the Socratic Method? Experiments with Code Debugging**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: GPT, GPT-4, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2310.03210v1)  

---


**ABSTRACT**  
When employing the Socratic method of teaching, instructors guide students toward solving a problem on their own rather than providing the solution directly. While this strategy can substantially improve learning outcomes, it is usually time-consuming and cognitively demanding. Automated Socratic conversational agents can augment human instruction and provide the necessary scale, however their development is hampered by the lack of suitable data for training and evaluation. In this paper, we introduce a manually created dataset of multi-turn Socratic advice that is aimed at helping a novice programmer fix buggy solutions to simple computational problems. The dataset is then used for benchmarking the Socratic debugging abilities of a number of language models, ranging from fine-tuning the instruction-based text-to-text transformer Flan-T5 to zero-shot and chain of thought prompting of the much larger GPT-4. The code and datasets are made freely available for research at the link below. https://github.com/taisazero/socratic-debugging-benchmark

{{</citation>}}


### (3/117) Retrieval-augmented Generation to Improve Math Question-Answering: Trade-offs Between Groundedness and Human Preference (Zachary Levonian et al., 2023)

{{<citation>}}

Zachary Levonian, Chenglu Li, Wangda Zhu, Anoushka Gade, Owen Henkel, Millie-Ellen Postle, Wanli Xing. (2023)  
**Retrieval-augmented Generation to Improve Math Question-Answering: Trade-offs Between Groundedness and Human Preference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.03184v1)  

---


**ABSTRACT**  
For middle-school math students, interactive question-answering (QA) with tutors is an effective way to learn. The flexibility and emergent capabilities of generative large language models (LLMs) has led to a surge of interest in automating portions of the tutoring process - including interactive QA to support conceptual discussion of mathematical concepts. However, LLM responses to math questions can be incorrect or mismatched to the educational context - such as being misaligned with a school's curriculum. One potential solution is retrieval-augmented generation (RAG), which involves incorporating a vetted external knowledge source in the LLM prompt to increase response quality. In this paper, we designed prompts that retrieve and use content from a high-quality open-source math textbook to generate responses to real student questions. We evaluate the efficacy of this RAG system for middle-school algebra and geometry QA by administering a multi-condition survey, finding that humans prefer responses generated using RAG, but not when responses are too grounded in the textbook content. We argue that while RAG is able to improve response quality, designers of math QA systems must consider trade-offs between generating responses preferred by students and responses closely matched to specific educational resources.

{{</citation>}}


### (4/117) $\mathcal{B}$-Coder: Value-Based Deep Reinforcement Learning for Program Synthesis (Zishun Yu et al., 2023)

{{<citation>}}

Zishun Yu, Yunzhe Tao, Liyu Chen, Tao Sun, Hongxia Yang. (2023)  
**$\mathcal{B}$-Coder: Value-Based Deep Reinforcement Learning for Program Synthesis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03173v1)  

---


**ABSTRACT**  
Program synthesis aims to create accurate, executable code from natural language descriptions. This field has leveraged the power of reinforcement learning (RL) in conjunction with large language models (LLMs), significantly enhancing code generation capabilities. This integration focuses on directly optimizing functional correctness, transcending conventional supervised losses. While current literature predominantly favors policy-based algorithms, attributes of program synthesis suggest a natural compatibility with value-based methods. This stems from rich collection of off-policy programs developed by human programmers, and the straightforward verification of generated programs through automated unit testing (i.e. easily obtainable rewards in RL language). Diverging from the predominant use of policy-based algorithms, our work explores the applicability of value-based approaches, leading to the development of our $\mathcal{B}$-Coder (pronounced Bellman coder). Yet, training value-based methods presents challenges due to the enormous search space inherent to program synthesis. To this end, we propose an initialization protocol for RL agents utilizing pre-trained LMs and a conservative Bellman operator to reduce training complexities. Moreover, we demonstrate how to leverage the learned value functions as a dual strategy to post-process generated programs. Our empirical evaluations demonstrated $\mathcal{B}$-Coder's capability in achieving state-of-the-art performance compared with policy-based methods. Remarkably, this achievement is reached with minimal reward engineering effort, highlighting the effectiveness of value-based RL, independent of reward designs.

{{</citation>}}


### (5/117) Large Language Model Cascades with Mixture of Thoughts Representations for Cost-efficient Reasoning (Murong Yue et al., 2023)

{{<citation>}}

Murong Yue, Jie Zhao, Min Zhang, Liang Du, Ziyu Yao. (2023)  
**Large Language Model Cascades with Mixture of Thoughts Representations for Cost-efficient Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.03094v1)  

---


**ABSTRACT**  
Large language models (LLMs) such as GPT-4 have exhibited remarkable performance in a variety of tasks, but this strong performance often comes with the high expense of using paid API services. In this paper, we are motivated to study building an LLM cascade to save the cost of using LLMs, particularly for performing reasoning (e.g., mathematical, causal) tasks. Our cascade pipeline follows the intuition that simpler questions can be addressed by a weaker but more affordable LLM, whereas only the challenging questions necessitate the stronger and more expensive LLM. To realize this decision-making, we consider the "answer consistency" of the weaker LLM as a signal of the question difficulty and propose several methods for the answer sampling and consistency checking, including one leveraging a mixture of two thought representations (i.e., Chain-of-Thought and Program-of-Thought). Through experiments on six reasoning benchmark datasets, with GPT-3.5-turbo and GPT-4 being the weaker and stronger LLMs, respectively, we demonstrate that our proposed LLM cascades can achieve performance comparable to using solely the stronger LLM but require only 40% of its cost.

{{</citation>}}


### (6/117) Discovering Knowledge-Critical Subnetworks in Pretrained Language Models (Deniz Bayazit et al., 2023)

{{<citation>}}

Deniz Bayazit, Negar Foroutan, Zeming Chen, Gail Weiss, Antoine Bosselut. (2023)  
**Discovering Knowledge-Critical Subnetworks in Pretrained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2310.03084v1)  

---


**ABSTRACT**  
Pretrained language models (LMs) encode implicit representations of knowledge in their parameters. However, localizing these representations and disentangling them from each other remains an open problem. In this work, we investigate whether pretrained language models contain various knowledge-critical subnetworks: particular sparse computational subgraphs responsible for encoding specific knowledge the model has memorized. We propose a multi-objective differentiable weight masking scheme to discover these subnetworks and show that we can use them to precisely remove specific knowledge from models while minimizing adverse effects on the behavior of the original language model. We demonstrate our method on multiple GPT2 variants, uncovering highly sparse subnetworks (98%+) that are solely responsible for specific collections of relational knowledge. When these subnetworks are removed, the remaining network maintains most of its initial capacity (modeling language and other memorized relational knowledge) but struggles to express the removed knowledge, and suffers performance drops on examples needing this removed knowledge on downstream tasks after finetuning.

{{</citation>}}


### (7/117) Retrieval meets Long Context Large Language Models (Peng Xu et al., 2023)

{{<citation>}}

Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, Bryan Catanzaro. (2023)  
**Retrieval meets Long Context Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03025v1)  

---


**ABSTRACT**  
Extending the context window of large language models (LLMs) is getting popular recently, while the solution of augmenting LLMs with retrieval has existed for years. The natural questions are: i) Retrieval-augmentation versus long context window, which one is better for downstream tasks? ii) Can both methods be combined to get the best of both worlds? In this work, we answer these questions by studying both solutions using two state-of-the-art pretrained LLMs, i.e., a proprietary 43B GPT and LLaMA2-70B. Perhaps surprisingly, we find that LLM with 4K context window using simple retrieval-augmentation at generation can achieve comparable performance to finetuned LLM with 16K context window via positional interpolation on long context tasks, while taking much less computation. More importantly, we demonstrate that retrieval can significantly improve the performance of LLMs regardless of their extended context window sizes. Our best model, retrieval-augmented LLaMA2-70B with 32K context window, outperforms GPT-3.5-turbo-16k and Davinci003 in terms of average score on seven long context tasks including question answering and query-based summarization. It also outperforms its non-retrieval LLaMA2-70B-32k baseline by a margin, while being much faster at generation. Our study provides general insights on the choice of retrieval-augmentation versus long context extension of LLM for practitioners.

{{</citation>}}


### (8/117) Multimodal Question Answering for Unified Information Extraction (Yuxuan Sun et al., 2023)

{{<citation>}}

Yuxuan Sun, Kai Zhang, Yu Su. (2023)  
**Multimodal Question Answering for Unified Information Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Information Extraction, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.03017v1)  

---


**ABSTRACT**  
Multimodal information extraction (MIE) aims to extract structured information from unstructured multimedia content. Due to the diversity of tasks and settings, most current MIE models are task-specific and data-intensive, which limits their generalization to real-world scenarios with diverse task requirements and limited labeled data. To address these issues, we propose a novel multimodal question answering (MQA) framework to unify three MIE tasks by reformulating them into a unified span extraction and multi-choice QA pipeline. Extensive experiments on six datasets show that: 1) Our MQA framework consistently and significantly improves the performances of various off-the-shelf large multimodal models (LMM) on MIE tasks, compared to vanilla prompting. 2) In the zero-shot setting, MQA outperforms previous state-of-the-art baselines by a large margin. In addition, the effectiveness of our framework can successfully transfer to the few-shot setting, enhancing LMMs on a scale of 10B parameters to be competitive or outperform much larger language models such as ChatGPT and GPT-4. Our MQA framework can serve as a general principle of utilizing LMMs to better solve MIE and potentially other downstream multimodal tasks.

{{</citation>}}


### (9/117) From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference (Siddharth Samsi et al., 2023)

{{<citation>}}

Siddharth Samsi, Dan Zhao, Joseph McDonald, Baolin Li, Adam Michaleas, Michael Jones, William Bergeron, Jeremy Kepner, Devesh Tiwari, Vijay Gadepally. (2023)  
**From Words to Watts: Benchmarking the Energy Costs of Large Language Model Inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-DC, cs.CL  
Keywords: AI, ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03003v1)  

---


**ABSTRACT**  
Large language models (LLMs) have exploded in popularity due to their new generative capabilities that go far beyond prior state-of-the-art. These technologies are increasingly being leveraged in various domains such as law, finance, and medicine. However, these models carry significant computational challenges, especially the compute and energy costs required for inference. Inference energy costs already receive less attention than the energy costs of training LLMs -- despite how often these large models are called on to conduct inference in reality (e.g., ChatGPT). As these state-of-the-art LLMs see increasing usage and deployment in various domains, a better understanding of their resource utilization is crucial for cost-savings, scaling performance, efficient hardware usage, and optimal inference strategies.   In this paper, we describe experiments conducted to study the computational and energy utilization of inference with LLMs. We benchmark and conduct a preliminary analysis of the inference performance and inference energy costs of different sizes of LLaMA -- a recent state-of-the-art LLM -- developed by Meta AI on two generations of popular GPUs (NVIDIA V100 \& A100) and two datasets (Alpaca and GSM8K) to reflect the diverse set of tasks/benchmarks for LLMs in research and practice. We present the results of multi-node, multi-GPU inference using model sharding across up to 32 GPUs. To our knowledge, our work is the one of the first to study LLM inference performance from the perspective of computational and energy resources at this scale.

{{</citation>}}


### (10/117) UniverSLU: Universal Spoken Language Understanding for Diverse Classification and Sequence Generation Tasks with a Single Network (Siddhant Arora et al., 2023)

{{<citation>}}

Siddhant Arora, Hayato Futami, Jee-weon Jung, Yifan Peng, Roshan Sharma, Yosuke Kashiwagi, Emiru Tsunoo, Shinji Watanabe. (2023)  
**UniverSLU: Universal Spoken Language Understanding for Diverse Classification and Sequence Generation Tasks with a Single Network**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Spoken Language Understanding  
[Paper Link](http://arxiv.org/abs/2310.02973v1)  

---


**ABSTRACT**  
Recent studies have demonstrated promising outcomes by employing large language models with multi-tasking capabilities. They utilize prompts to guide the model's behavior and surpass performance of task-specific models. Motivated by this, we ask: can we build a single model that jointly perform various spoken language understanding (SLU) tasks? To address this, we utilize pre-trained automatic speech recognition (ASR) models and employ various task and dataset specifiers as discrete prompts. We demonstrate efficacy of our single multi-task learning (MTL) model "UniverSLU" for 12 different speech classification and sequence generation tasks across 17 datasets and 9 languages. Results show that UniverSLU achieves competitive performance and even surpasses task-specific models. We also conduct preliminary investigations into enabling human-interpretable natural phrases instead of task specifiers as discrete prompts and test the model's generalization capabilities to new paraphrases.

{{</citation>}}


### (11/117) DQ-LoRe: Dual Queries with Low Rank Approximation Re-ranking for In-Context Learning (Jiong Xiong et al., 2023)

{{<citation>}}

Jiong Xiong, Zixuan Li, Chuanyang Zheng, Zhijiang Guo, Yichun Yin, Enze Xie, Zhicheng Yang, Qingxing Cao, Haiming Wang, Xiongwei Han, Jing Tang, Chengming Li, Xiaodan Liang. (2023)  
**DQ-LoRe: Dual Queries with Low Rank Approximation Re-ranking for In-Context Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02954v2)  

---


**ABSTRACT**  
Recent advances in natural language processing, primarily propelled by Large Language Models (LLMs), have showcased their remarkable capabilities grounded in in-context learning. A promising avenue for guiding LLMs in intricate reasoning tasks involves the utilization of intermediate reasoning steps within the Chain-of-Thought (CoT) paradigm. Nevertheless, the central challenge lies in the effective selection of exemplars for facilitating in-context learning. In this study, we introduce a framework that leverages Dual Queries and Low-rank approximation Re-ranking (DQ-LoRe) to automatically select exemplars for in-context learning. Dual Queries first query LLM to obtain LLM-generated knowledge such as CoT, then query the retriever to obtain the final exemplars via both question and the knowledge. Moreover, for the second query, LoRe employs dimensionality reduction techniques to refine exemplar selection, ensuring close alignment with the input question's knowledge. Through extensive experiments, we demonstrate that DQ-LoRe significantly outperforms prior state-of-the-art methods in the automatic selection of exemplars for GPT-4, enhancing performance from 92.5\% to 94.2\%. Our comprehensive analysis further reveals that DQ-LoRe consistently outperforms retrieval-based approaches in terms of both performance and adaptability, especially in scenarios characterized by distribution shifts. DQ-LoRe pushes the boundaries of in-context learning and opens up new avenues for addressing complex reasoning challenges. We will release the code soon.

{{</citation>}}


### (12/117) Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models (Xianjun Yang et al., 2023)

{{<citation>}}

Xianjun Yang, Xiao Wang, Qi Zhang, Linda Petzold, William Yang Wang, Xun Zhao, Dahua Lin. (2023)  
**Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: AI, Falcon, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02949v1)  

---


**ABSTRACT**  
Warning: This paper contains examples of harmful language, and reader discretion is recommended. The increasing open release of powerful large language models (LLMs) has facilitated the development of downstream applications by reducing the essential cost of data annotation and computation. To ensure AI safety, extensive safety-alignment measures have been conducted to armor these models against malicious use (primarily hard prompt attack). However, beneath the seemingly resilient facade of the armor, there might lurk a shadow. By simply tuning on 100 malicious examples with 1 GPU hour, these safely aligned LLMs can be easily subverted to generate harmful content. Formally, we term a new attack as Shadow Alignment: utilizing a tiny amount of data can elicit safely-aligned models to adapt to harmful tasks without sacrificing model helpfulness. Remarkably, the subverted models retain their capability to respond appropriately to regular inquiries. Experiments across 8 models released by 5 different organizations (LLaMa-2, Falcon, InternLM, BaiChuan2, Vicuna) demonstrate the effectiveness of shadow alignment attack. Besides, the single-turn English-only attack successfully transfers to multi-turn dialogue and other languages. This study serves as a clarion call for a collective effort to overhaul and fortify the safety of open-source LLMs against malicious attackers.

{{</citation>}}


### (13/117) Assessing Large Language Models on Climate Information (Jannis Bulian et al., 2023)

{{<citation>}}

Jannis Bulian, Mike S. Schäfer, Afra Amini, Heidi Lam, Massimiliano Ciaramita, Ben Gaiarin, Michelle Chen Huebscher, Christian Buck, Niels Mede, Markus Leippold, Nadine Strauss. (2023)  
**Assessing Large Language Models on Climate Information**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02932v1)  

---


**ABSTRACT**  
Understanding how climate change affects us and learning about available solutions are key steps toward empowering individuals and communities to mitigate and adapt to it. As Large Language Models (LLMs) rise in popularity, it is necessary to assess their capability in this domain. In this study, we present a comprehensive evaluation framework, grounded in science communication principles, to analyze LLM responses to climate change topics. Our framework emphasizes both the presentational and epistemological adequacy of answers, offering a fine-grained analysis of LLM generations. Spanning 8 dimensions, our framework discerns up to 30 distinct issues in model outputs. The task is a real-world example of a growing number of challenging problems where AI can complement and lift human performance. We introduce a novel and practical protocol for scalable oversight that uses AI Assistance and relies on raters with relevant educational backgrounds. We evaluate several recent LLMs and conduct a comprehensive analysis of the results, shedding light on both the potential and the limitations of LLMs in the realm of climate communication.

{{</citation>}}


### (14/117) Hate Speech Detection in Limited Data Contexts using Synthetic Data Generation (Aman Khullar et al., 2023)

{{<citation>}}

Aman Khullar, Daniel Nkemelu, Cuong V. Nguyen, Michael L. Best. (2023)  
**Hate Speech Detection in Limited Data Contexts using Synthetic Data Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Hate Speech Detection  
[Paper Link](http://arxiv.org/abs/2310.02876v1)  

---


**ABSTRACT**  
A growing body of work has focused on text classification methods for detecting the increasing amount of hate speech posted online. This progress has been limited to only a select number of highly-resourced languages causing detection systems to either under-perform or not exist in limited data contexts. This is majorly caused by a lack of training data which is expensive to collect and curate in these settings. In this work, we propose a data augmentation approach that addresses the problem of lack of data for online hate speech detection in limited data contexts using synthetic data generation techniques. Given a handful of hate speech examples in a high-resource language such as English, we present three methods to synthesize new examples of hate speech data in a target language that retains the hate sentiment in the original examples but transfers the hate targets. We apply our approach to generate training data for hate speech classification tasks in Hindi and Vietnamese. Our findings show that a model trained on synthetic data performs comparably to, and in some cases outperforms, a model trained only on the samples available in the target domain. This method can be adopted to bootstrap hate speech detection models from scratch in limited data contexts. As the growth of social media within these contexts continues to outstrip response efforts, this work furthers our capacities for detection, understanding, and response to hate speech.

{{</citation>}}


### (15/117) Sweeping Heterogeneity with Smart MoPs: Mixture of Prompts for LLM Task Adaptation (Chen Dun et al., 2023)

{{<citation>}}

Chen Dun, Mirian Hipolito Garcia, Guoqing Zheng, Ahmed Hassan Awadallah, Anastasios Kyrillidis, Robert Sim. (2023)  
**Sweeping Heterogeneity with Smart MoPs: Mixture of Prompts for LLM Task Adaptation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02842v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have the ability to solve a variety of tasks, such as text summarization and mathematical questions, just out of the box, but they are often trained with a single task in mind. Due to high computational costs, the current trend is to use prompt instruction tuning to better adjust monolithic, pretrained LLMs for new -- but often individual -- downstream tasks. Thus, how one would expand prompt tuning to handle -- concomitantly -- heterogeneous tasks and data distributions is a widely open question. To address this gap, we suggest the use of \emph{Mixture of Prompts}, or MoPs, associated with smart gating functionality: the latter -- whose design is one of the contributions of this paper -- can identify relevant skills embedded in different groups of prompts and dynamically assign combined experts (i.e., collection of prompts), based on the target task. Additionally, MoPs are empirically agnostic to any model compression technique applied -- for efficiency reasons -- as well as instruction data source and task composition. In practice, MoPs can simultaneously mitigate prompt training "interference" in multi-task, multi-source scenarios (e.g., task and data heterogeneity across sources), as well as possible implications from model approximations. As a highlight, MoPs manage to decrease final perplexity from $\sim20\%$ up to $\sim70\%$, as compared to baselines, in the federated scenario, and from $\sim 3\%$ up to $\sim30\%$ in the centralized scenario.

{{</citation>}}


### (16/117) DOMINO: A Dual-System for Multi-step Visual Language Reasoning (Peifang Wang et al., 2023)

{{<citation>}}

Peifang Wang, Olga Golovneva, Armen Aghajanyan, Xiang Ren, Muhao Chen, Asli Celikyilmaz, Maryam Fazel-Zarandi. (2023)  
**DOMINO: A Dual-System for Multi-step Visual Language Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: LLaMA, PaLM, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.02804v1)  

---


**ABSTRACT**  
Visual language reasoning requires a system to extract text or numbers from information-dense images like charts or plots and perform logical or arithmetic reasoning to arrive at an answer. To tackle this task, existing work relies on either (1) an end-to-end vision-language model trained on a large amount of data, or (2) a two-stage pipeline where a captioning model converts the image into text that is further read by another large language model to deduce the answer. However, the former approach forces the model to answer a complex question with one single step, and the latter approach is prone to inaccurate or distracting information in the converted text that can confuse the language model. In this work, we propose a dual-system for multi-step multimodal reasoning, which consists of a "System-1" step for visual information extraction and a "System-2" step for deliberate reasoning. Given an input, System-2 breaks down the question into atomic sub-steps, each guiding System-1 to extract the information required for reasoning from the image. Experiments on chart and plot datasets show that our method with a pre-trained System-2 module performs competitively compared to prior work on in- and out-of-distribution data. By fine-tuning the System-2 module (LLaMA-2 70B) on only a small amount of data on multi-step reasoning, the accuracy of our method is further improved and surpasses the best fully-supervised end-to-end approach by 5.7% and a pipeline approach with FlanPaLM (540B) by 7.5% on a challenging dataset with human-authored questions.

{{</citation>}}


### (17/117) Low Resource Summarization using Pre-trained Language Models (Mubashir Munaf et al., 2023)

{{<citation>}}

Mubashir Munaf, Hammad Afzal, Naima Iltaf, Khawir Mahmood. (2023)  
**Low Resource Summarization using Pre-trained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, NLP, Natural Language Processing, Summarization, T5  
[Paper Link](http://arxiv.org/abs/2310.02790v1)  

---


**ABSTRACT**  
With the advent of Deep Learning based Artificial Neural Networks models, Natural Language Processing (NLP) has witnessed significant improvements in textual data processing in terms of its efficiency and accuracy. However, the research is mostly restricted to high-resource languages such as English and low-resource languages still suffer from a lack of available resources in terms of training datasets as well as models with even baseline evaluation results. Considering the limited availability of resources for low-resource languages, we propose a methodology for adapting self-attentive transformer-based architecture models (mBERT, mT5) for low-resource summarization, supplemented by the construction of a new baseline dataset (76.5k article, summary pairs) in a low-resource language Urdu. Choosing news (a publicly available source) as the application domain has the potential to make the proposed methodology useful for reproducing in other languages with limited resources. Our adapted summarization model \textit{urT5} with up to 44.78\% reduction in size as compared to \textit{mT5} can capture contextual information of low resource language effectively with evaluation score (up to 46.35 ROUGE-1, 77 BERTScore) at par with state-of-the-art models in high resource language English \textit{(PEGASUS: 47.21, BART: 45.14 on XSUM Dataset)}. The proposed method provided a baseline approach towards extractive as well as abstractive summarization with competitive evaluation results in a limited resource setup.

{{</citation>}}


### (18/117) A UMLS-Augmented Framework for Improving Factuality in Large Language Models within Healthcare (Rui Yang et al., 2023)

{{<citation>}}

Rui Yang, Edison Marrese-Taylor, Yuhe Ke, Lechao Cheng, Qingyu Chen, Irene Li. (2023)  
**A UMLS-Augmented Framework for Improving Factuality in Large Language Models within Healthcare**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT, GPT-3.5, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.02778v1)  

---


**ABSTRACT**  
Large language models (LLMs) have demonstrated powerful text generation capabilities, bringing unprecedented innovation to the healthcare field. While LLMs hold immense promise for applications in healthcare, applying them to real clinical scenarios presents significant challenges, as these models may generate content that deviates from established medical facts and even exhibit potential biases. In our research, we develop an augmented LLM framework based on the Unified Medical Language System (UMLS), aiming to better serve the healthcare community. We employ LLaMa2-13b-chat and ChatGPT-3.5 as our benchmark models, and conduct automatic evaluations using the ROUGE Score and BERTScore on 104 questions from the LiveQA test set. Additionally, we establish criteria for physician-evaluation based on four dimensions: Factuality, Completeness, Readability and Relevancy. ChatGPT-3.5 is used for physician evaluation with 20 questions on the LiveQA test set. Multiple resident physicians conducted blind reviews to evaluate the generated content, and the results indicate that this framework effectively enhances the factuality, completeness, and relevance of generated content. Our research demonstrates the effectiveness of using UMLS-augmented LLMs and highlights the potential application value of LLMs in in medical question-answering.

{{</citation>}}


### (19/117) The Role of Linguistic Priors in Measuring Compositional Generalization of Vision-Language Models (Chenwei Wu et al., 2023)

{{<citation>}}

Chenwei Wu, Li Erran Li, Stefano Ermon, Patrick Haffner, Rong Ge, Zaiwei Zhang. (2023)  
**The Role of Linguistic Priors in Measuring Compositional Generalization of Vision-Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02777v1)  

---


**ABSTRACT**  
Compositionality is a common property in many modalities including natural languages and images, but the compositional generalization of multi-modal models is not well-understood. In this paper, we identify two sources of visual-linguistic compositionality: linguistic priors and the interplay between images and texts. We show that current attempts to improve compositional generalization rely on linguistic priors rather than on information in the image. We also propose a new metric for compositionality without such linguistic priors.

{{</citation>}}


### (20/117) How FaR Are Large Language Models From Agents with Theory-of-Mind? (Pei Zhou et al., 2023)

{{<citation>}}

Pei Zhou, Aman Madaan, Srividya Pranavi Potharaju, Aditya Gupta, Kevin R. McKee, Ari Holtzman, Jay Pujara, Xiang Ren, Swaroop Mishra, Aida Nematzadeh, Shyam Upadhyay, Manaal Faruqui. (2023)  
**How FaR Are Large Language Models From Agents with Theory-of-Mind?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2310.03051v1)  

---


**ABSTRACT**  
"Thinking is for Doing." Humans can infer other people's mental states from observations--an ability called Theory-of-Mind (ToM)--and subsequently act pragmatically on those inferences. Existing question answering benchmarks such as ToMi ask models questions to make inferences about beliefs of characters in a story, but do not test whether models can then use these inferences to guide their actions. We propose a new evaluation paradigm for large language models (LLMs): Thinking for Doing (T4D), which requires models to connect inferences about others' mental states to actions in social scenarios. Experiments on T4D demonstrate that LLMs such as GPT-4 and PaLM 2 seemingly excel at tracking characters' beliefs in stories, but they struggle to translate this capability into strategic action. Our analysis reveals the core challenge for LLMs lies in identifying the implicit inferences about mental states without being explicitly asked about as in ToMi, that lead to choosing the correct action in T4D. To bridge this gap, we introduce a zero-shot prompting framework, Foresee and Reflect (FaR), which provides a reasoning structure that encourages LLMs to anticipate future challenges and reason about potential actions. FaR boosts GPT-4's performance from 50% to 71% on T4D, outperforming other prompting methods such as Chain-of-Thought and Self-Ask. Moreover, FaR generalizes to diverse out-of-distribution story structures and scenarios that also require ToM inferences to choose an action, consistently outperforming other methods including few-shot in-context learning.

{{</citation>}}


### (21/117) I$^2$KD-SLU: An Intra-Inter Knowledge Distillation Framework for Zero-Shot Cross-Lingual Spoken Language Understanding (Tianjun Mao et al., 2023)

{{<citation>}}

Tianjun Mao, Chenghong Zhang. (2023)  
**I$^2$KD-SLU: An Intra-Inter Knowledge Distillation Framework for Zero-Shot Cross-Lingual Spoken Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Distillation, Spoken Language Understanding, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.02594v1)  

---


**ABSTRACT**  
Spoken language understanding (SLU) typically includes two subtasks: intent detection and slot filling. Currently, it has achieved great success in high-resource languages, but it still remains challenging in low-resource languages due to the scarcity of labeled training data. Hence, there is a growing interest in zero-shot cross-lingual SLU. Despite of the success of existing zero-shot cross-lingual SLU models, most of them neglect to achieve the mutual guidance between intent and slots. To address this issue, we propose an Intra-Inter Knowledge Distillation framework for zero-shot cross-lingual Spoken Language Understanding (I$^2$KD-SLU) to model the mutual guidance. Specifically, we not only apply intra-knowledge distillation between intent predictions or slot predictions of the same utterance in different languages, but also apply inter-knowledge distillation between intent predictions and slot predictions of the same utterance. Our experimental results demonstrate that our proposed framework significantly improves the performance compared with the strong baselines and achieves the new state-of-the-art performance on the MultiATIS++ dataset, obtaining a significant improvement over the previous best model in overall accuracy.

{{</citation>}}


### (22/117) NOLA: Networks as Linear Combination of Low Rank Random Basis (Soroush Abbasi Koohpayegani et al., 2023)

{{<citation>}}

Soroush Abbasi Koohpayegani, KL Navaneet, Parsa Nooralinejad, Soheil Kolouri, Hamed Pirsiavash. (2023)  
**NOLA: Networks as Linear Combination of Low Rank Random Basis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02556v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have recently gained popularity due to their impressive few-shot performance across various downstream tasks. However, fine-tuning all parameters and storing a unique model for each downstream task or domain becomes impractical because of the massive size of checkpoints (e.g., 350GB in GPT-3). Current literature, such as LoRA, showcases the potential of low-rank modifications to the original weights of an LLM, enabling efficient adaptation and storage for task-specific models. These methods can reduce the number of parameters needed to fine-tune an LLM by several orders of magnitude. Yet, these methods face two primary limitations: 1) the parameter reduction is lower-bounded by the rank one decomposition, and 2) the extent of reduction is heavily influenced by both the model architecture and the chosen rank. For instance, in larger models, even a rank one decomposition might exceed the number of parameters truly needed for adaptation. In this paper, we introduce NOLA, which overcomes the rank one lower bound present in LoRA. It achieves this by re-parameterizing the low-rank matrices in LoRA using linear combinations of randomly generated matrices (basis) and optimizing the linear mixture coefficients only. This approach allows us to decouple the number of trainable parameters from both the choice of rank and the network architecture. We present adaptation results using GPT-2 and ViT in natural language and computer vision tasks. NOLA performs as well as, or better than models with equivalent parameter counts. Furthermore, we demonstrate that we can halve the parameters in larger models compared to LoRA with rank one, without sacrificing performance.

{{</citation>}}


### (23/117) CITING: Large Language Models Create Curriculum for Instruction Tuning (Tao Feng et al., 2023)

{{<citation>}}

Tao Feng, Zifeng Wang, Jimeng Sun. (2023)  
**CITING: Large Language Models Create Curriculum for Instruction Tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02527v1)  

---


**ABSTRACT**  
The recent advancement of large language models (LLMs) has been achieved through a combo of instruction tuning and human alignment. However, building manually crafted instruction datasets and performing human alignment become the bottleneck for scaling the development of LLMs. In this paper, we exploit the idea of leveraging AI models in lieu of humans as the teacher to train student LLMs. Our method is inspired by how human students refine their writing skills by following the rubrics and learning from the revisions offered by their tutors. Specifically, we employ a teacher LLM to create a curriculum for instruction tuning of the student LLM, namely Curriculum Instruction TunING (CITING). It encompasses two main steps: (1) the teacher LLM crafts the rubrics for evaluating the answers corresponding to various types of questions, and (2) the student LLM learns to follow the rubrics and perform self-correction from the revision made by the teacher. We further iteratively carry out it to embody the procedure of CITING. We compare CITING to a series of state-of-the-art baselines on four datasets. Our method demonstrates strong improvement in terms of articulate, in-depth, and comprehensive by GPT-4 evaluation. Specifically, it achieves an average winning rate of 79.4% over SFT, 73.4% over RLHF, 78.1% over RRHF, and 76.3% over RAFT, respectively.

{{</citation>}}


## cs.IR (1)



### (24/117) Amazon Books Rating prediction & Recommendation Model (Hsiu-Ping Lin et al., 2023)

{{<citation>}}

Hsiu-Ping Lin, Suman Chauhan, Yougender Chauhan, Nagender Chauhan, Jongwook Woo. (2023)  
**Amazon Books Rating prediction & Recommendation Model**  

---
Primary Category: cs.IR  
Categories: cs-DC, cs-IR, cs.IR  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2310.03200v1)  

---


**ABSTRACT**  
This paper uses the dataset of Amazon to predict the books ratings listed on Amazon website. As part of this project, we predicted the ratings of the books, and also built a recommendation cluster. This recommendation cluster provides the recommended books based on the column's values from dataset, for instance, category, description, author, price, reviews etc. This paper provides a flow of handling big data files, data engineering, building models and providing predictions. The models predict book ratings column using various PySpark Machine Learning APIs. Additionally, we used hyper-parameters and parameters tuning. Also, Cross Validation and TrainValidationSplit were used for generalization. Finally, we performed a comparison between Binary Classification and Multiclass Classification in their accuracies. We converted our label from multiclass to binary to see if we could find any difference between the two classifications. As a result, we found out that we get higher accuracy in binary classification than in multiclass classification.

{{</citation>}}


## cs.LG (36)



### (25/117) Deep reinforcement learning for machine scheduling: Methodology, the state-of-the-art, and future directions (Maziyar Khadivi et al., 2023)

{{<citation>}}

Maziyar Khadivi, Todd Charter, Marjan Yaghoubi, Masoud Jalayer, Maryam Ahang, Ardeshir Shojaeinasab, Homayoun Najjaran. (2023)  
**Deep reinforcement learning for machine scheduling: Methodology, the state-of-the-art, and future directions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03195v1)  

---


**ABSTRACT**  
Machine scheduling aims to optimize job assignments to machines while adhering to manufacturing rules and job specifications. This optimization leads to reduced operational costs, improved customer demand fulfillment, and enhanced production efficiency. However, machine scheduling remains a challenging combinatorial problem due to its NP-hard nature. Deep Reinforcement Learning (DRL), a key component of artificial general intelligence, has shown promise in various domains like gaming and robotics. Researchers have explored applying DRL to machine scheduling problems since 1995. This paper offers a comprehensive review and comparison of DRL-based approaches, highlighting their methodology, applications, advantages, and limitations. It categorizes these approaches based on computational components: conventional neural networks, encoder-decoder architectures, graph neural networks, and metaheuristic algorithms. Our review concludes that DRL-based methods outperform exact solvers, heuristics, and tabular reinforcement learning algorithms in terms of computation speed and generating near-global optimal solutions. These DRL-based approaches have been successfully applied to static and dynamic scheduling across diverse machine environments and job characteristics. However, DRL-based schedulers face limitations in handling complex operational constraints, configurable multi-objective optimization, generalization, scalability, interpretability, and robustness. Addressing these challenges will be a crucial focus for future research in this field. This paper serves as a valuable resource for researchers to assess the current state of DRL-based machine scheduling and identify research gaps. It also aids experts and practitioners in selecting the appropriate DRL approach for production scheduling.

{{</citation>}}


### (26/117) Neural architecture impact on identifying temporally extended Reinforcement Learning tasks (Victor Vadakechirayath George, 2023)

{{<citation>}}

Victor Vadakechirayath George. (2023)  
**Neural architecture impact on identifying temporally extended Reinforcement Learning tasks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Attention, Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2310.03161v1)  

---


**ABSTRACT**  
Inspired by recent developments in attention models for image classification and natural language processing, we present various Attention based architectures in reinforcement learning (RL) domain, capable of performing well on OpenAI Gym Atari-2600 game suite. In spite of the recent success of Deep Reinforcement learning techniques in various fields like robotics, gaming and healthcare, they suffer from a major drawback that neural networks are difficult to interpret. We try to get around this problem with the help of Attention based models. In Attention based models, extracting and overlaying of attention map onto images allows for direct observation of information used by agent to select actions and easier interpretation of logic behind the chosen actions. Our models in addition to playing well on gym-Atari environments, also provide insights on how agent perceives its environment. In addition, motivated by recent developments in attention based video-classification models using Vision Transformer, we come up with an architecture based on Vision Transformer, for image-based RL domain too. Compared to previous works in Vision Transformer, our model is faster to train and requires fewer computational resources. 3

{{</citation>}}


### (27/117) Assessment of Prediction Intervals Using Uncertainty Characteristics Curves (Jiri Navratil et al., 2023)

{{<citation>}}

Jiri Navratil, Benjamin Elder, Matthew Arnold, Soumya Ghosh, Prasanna Sattigeri. (2023)  
**Assessment of Prediction Intervals Using Uncertainty Characteristics Curves**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03158v1)  

---


**ABSTRACT**  
Accurate quantification of model uncertainty has long been recognized as a fundamental requirement for trusted AI. In regression tasks, uncertainty is typically quantified using prediction intervals calibrated to an ad-hoc operating point, making evaluation and comparison across different studies relatively difficult. Our work leverages: (1) the concept of operating characteristics curves and (2) the notion of a gain over a null reference, to derive a novel operating point agnostic assessment methodology for prediction intervals. The paper defines the Uncertainty Characteristics Curve and demonstrates its utility in selected scenarios. We argue that the proposed method addresses the current need for comprehensive assessment of prediction intervals and thus represents a valuable addition to the uncertainty quantification toolbox.

{{</citation>}}


### (28/117) Towards out-of-distribution generalizable predictions of chemical kinetics properties (Zihao Wang et al., 2023)

{{<citation>}}

Zihao Wang, Yongqiang Chen, Yang Duan, Weijiang Li, Bo Han, James Cheng, Hanghang Tong. (2023)  
**Towards out-of-distribution generalizable predictions of chemical kinetics properties**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03152v1)  

---


**ABSTRACT**  
Machine Learning (ML) techniques have found applications in estimating chemical kinetics properties. With the accumulated drug molecules identified through "AI4drug discovery", the next imperative lies in AI-driven design for high-throughput chemical synthesis processes, with the estimation of properties of unseen reactions with unexplored molecules. To this end, the existing ML approaches for kinetics property prediction are required to be Out-Of-Distribution (OOD) generalizable. In this paper, we categorize the OOD kinetic property prediction into three levels (structure, condition, and mechanism), revealing unique aspects of such problems. Under this framework, we create comprehensive datasets to benchmark (1) the state-of-the-art ML approaches for reaction prediction in the OOD setting and (2) the state-of-the-art graph OOD methods in kinetics property prediction problems. Our results demonstrated the challenges and opportunities in OOD kinetics property prediction. Our datasets and benchmarks can further support research in this direction.

{{</citation>}}


### (29/117) Federated Fine-Tuning of LLMs on the Very Edge: The Good, the Bad, the Ugly (Herbert Woisetschläger et al., 2023)

{{<citation>}}

Herbert Woisetschläger, Alexander Isenko, Shiqiang Wang, Ruben Mayer, Hans-Arno Jacobsen. (2023)  
**Federated Fine-Tuning of LLMs on the Very Edge: The Good, the Bad, the Ugly**  

---
Primary Category: cs.LG  
Categories: I-2-11; C-2-4; C-4; D-2-8, cs-DC, cs-LG, cs-PF, cs.LG  
Keywords: Language Model, T5  
[Paper Link](http://arxiv.org/abs/2310.03150v1)  

---


**ABSTRACT**  
Large Language Models (LLM) and foundation models are popular as they offer new opportunities for individuals and businesses to improve natural language processing, interact with data, and retrieve information faster. However, training or fine-tuning LLMs requires a vast amount of data, which can be challenging to access due to legal or technical restrictions and may require private computing resources. Federated Learning (FL) is a solution designed to overcome these challenges and expand data access for deep learning applications.   This paper takes a hardware-centric approach to explore how LLMs can be brought to modern edge computing systems. Our study fine-tunes the FLAN-T5 model family, ranging from 80M to 3B parameters, using FL for a text summarization task. We provide a micro-level hardware benchmark, compare the model FLOP utilization to a state-of-the-art data center GPU, and study the network utilization in realistic conditions. Our contribution is twofold: First, we evaluate the current capabilities of edge computing systems and their potential for LLM FL workloads. Second, by comparing these systems with a data-center GPU, we demonstrate the potential for improvement and the next steps toward achieving greater computational efficiency at the edge.

{{</citation>}}


### (30/117) Efficient Federated Prompt Tuning for Black-box Large Pre-trained Models (Zihao Lin et al., 2023)

{{<citation>}}

Zihao Lin, Yan Sun, Yifan Shi, Xueqian Wang, Lifu Huang, Li Shen, Dacheng Tao. (2023)  
**Efficient Federated Prompt Tuning for Black-box Large Pre-trained Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.03123v1)  

---


**ABSTRACT**  
With the blowout development of pre-trained models (PTMs), the efficient tuning of these models for diverse downstream applications has emerged as a pivotal research concern. Although recent investigations into prompt tuning have provided promising avenues, three salient challenges persist: (1) memory constraint: the continuous growth in the size of open-source PTMs renders fine-tuning, even a fraction of their parameters, challenging for many practitioners. (2) model privacy: existing PTMs often function as public API services, with their parameters inaccessible for effective or tailored fine-tuning. (3) data privacy: the fine-tuning of PTMs necessitates high-quality datasets, which are typically localized and not shared to public. To optimally harness each local dataset while navigating memory constraints and preserving privacy, we propose Federated Black-Box Prompt Tuning (Fed-BBPT). This innovative approach eschews reliance on parameter architectures and private dataset access, instead capitalizing on a central server that aids local users in collaboratively training a prompt generator through regular aggregation. Local users leverage API-driven learning via a zero-order optimizer, obviating the need for PTM deployment. Relative to extensive fine-tuning, Fed-BBPT proficiently sidesteps memory challenges tied to PTM storage and fine-tuning on local machines, tapping into comprehensive, high-quality, yet private training datasets. A thorough evaluation across 40 datasets spanning CV and NLP tasks underscores the robustness of our proposed model.

{{</citation>}}


### (31/117) Decision ConvFormer: Local Filtering in MetaFormer is Sufficient for Decision Making (Jeonghye Kim et al., 2023)

{{<citation>}}

Jeonghye Kim, Suyoung Lee, Woojun Kim, Youngchul Sung. (2023)  
**Decision ConvFormer: Local Filtering in MetaFormer is Sufficient for Decision Making**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.03022v2)  

---


**ABSTRACT**  
The recent success of Transformer in natural language processing has sparked its use in various domains. In offline reinforcement learning (RL), Decision Transformer (DT) is emerging as a promising model based on Transformer. However, we discovered that the attention module of DT is not appropriate to capture the inherent local dependence pattern in trajectories of RL modeled as a Markov decision process. To overcome the limitations of DT, we propose a novel action sequence predictor, named Decision ConvFormer (DC), based on the architecture of MetaFormer, which is a general structure to process multiple entities in parallel and understand the interrelationship among the multiple entities. DC employs local convolution filtering as the token mixer and can effectively capture the inherent local associations of the RL dataset. In extensive experiments, DC achieved state-of-the-art performance across various standard RL benchmarks while requiring fewer resources. Furthermore, we show that DC better understands the underlying meaning in data and exhibits enhanced generalization capability.

{{</citation>}}


### (32/117) Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions (Satwik Bhattamishra et al., 2023)

{{<citation>}}

Satwik Bhattamishra, Arkil Patel, Phil Blunsom, Varun Kanade. (2023)  
**Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: GPT, GPT-4, LLaMA, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.03016v1)  

---


**ABSTRACT**  
In order to understand the in-context learning phenomenon, recent works have adopted a stylized experimental framework and demonstrated that Transformers can learn gradient-based learning algorithms for various classes of real-valued functions. However, the limitations of Transformers in implementing learning algorithms, and their ability to learn other forms of algorithms are not well understood. Additionally, the degree to which these capabilities are confined to attention-based models is unclear. Furthermore, it remains to be seen whether the insights derived from these stylized settings can be extrapolated to pretrained Large Language Models (LLMs). In this work, we take a step towards answering these questions by demonstrating the following: (a) On a test-bed with a variety of Boolean function classes, we find that Transformers can nearly match the optimal learning algorithm for 'simpler' tasks, while their performance deteriorates on more 'complex' tasks. Additionally, we find that certain attention-free models perform (almost) identically to Transformers on a range of tasks. (b) When provided a teaching sequence, i.e. a set of examples that uniquely identifies a function in a class, we show that Transformers learn more sample-efficiently. Interestingly, our results show that Transformers can learn to implement two distinct algorithms to solve a single task, and can adaptively select the more sample-efficient algorithm depending on the sequence of in-context examples. (c) Lastly, we show that extant LLMs, e.g. LLaMA-2, GPT-4, can compete with nearest-neighbor baselines on prediction tasks that are guaranteed to not be in their training set.

{{</citation>}}


### (33/117) Soft Convex Quantization: Revisiting Vector Quantization with Convex Optimization (Tanmay Gautam et al., 2023)

{{<citation>}}

Tanmay Gautam, Reid Pryzant, Ziyi Yang, Chenguang Zhu, Somayeh Sojoudi. (2023)  
**Soft Convex Quantization: Revisiting Vector Quantization with Convex Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs.LG, math-OC  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2310.03004v1)  

---


**ABSTRACT**  
Vector Quantization (VQ) is a well-known technique in deep learning for extracting informative discrete latent representations. VQ-embedded models have shown impressive results in a range of applications including image and speech generation. VQ operates as a parametric K-means algorithm that quantizes inputs using a single codebook vector in the forward pass. While powerful, this technique faces practical challenges including codebook collapse, non-differentiability and lossy compression. To mitigate the aforementioned issues, we propose Soft Convex Quantization (SCQ) as a direct substitute for VQ. SCQ works like a differentiable convex optimization (DCO) layer: in the forward pass, we solve for the optimal convex combination of codebook vectors that quantize the inputs. In the backward pass, we leverage differentiability through the optimality conditions of the forward solution. We then introduce a scalable relaxation of the SCQ optimization and demonstrate its efficacy on the CIFAR-10, GTSRB and LSUN datasets. We train powerful SCQ autoencoder models that significantly outperform matched VQ-based architectures, observing an order of magnitude better image reconstruction and codebook usage with comparable quantization runtime.

{{</citation>}}


### (34/117) IBCL: Zero-shot Model Generation for Task Trade-offs in Continual Learning (Pengyuan Lu et al., 2023)

{{<citation>}}

Pengyuan Lu, Michele Caprio, Eric Eaton, Insup Lee. (2023)  
**IBCL: Zero-shot Model Generation for Task Trade-offs in Continual Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.02995v2)  

---


**ABSTRACT**  
Like generic multi-task learning, continual learning has the nature of multi-objective optimization, and therefore faces a trade-off between the performance of different tasks. That is, to optimize for the current task distribution, it may need to compromise performance on some previous tasks. This means that there exist multiple models that are Pareto-optimal at different times, each addressing a distinct task performance trade-off. Researchers have discussed how to train particular models to address specific trade-off preferences. However, existing algorithms require training overheads proportional to the number of preferences -- a large burden when there are multiple, possibly infinitely many, preferences. As a response, we propose Imprecise Bayesian Continual Learning (IBCL). Upon a new task, IBCL (1) updates a knowledge base in the form of a convex hull of model parameter distributions and (2) obtains particular models to address task trade-off preferences with zero-shot. That is, IBCL does not require any additional training overhead to generate preference-addressing models from its knowledge base. We show that models obtained by IBCL have guarantees in identifying the Pareto optimal parameters. Moreover, experiments on standard image classification and NLP tasks support this guarantee. Statistically, IBCL improves average per-task accuracy by at most 23\% and peak per-task accuracy by at most 15\% with respect to the baseline methods, with steadily near-zero or positive backward transfer. Most importantly, IBCL significantly reduces the training overhead from training 1 model per preference to at most 3 models for all preferences.

{{</citation>}}


### (35/117) Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors (Ido Amos et al., 2023)

{{<citation>}}

Ido Amos, Jonathan Berant, Ankit Gupta. (2023)  
**Never Train from Scratch: Fair Comparison of Long-Sequence Models Requires Data-Driven Priors**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.02980v1)  

---


**ABSTRACT**  
Modeling long-range dependencies across sequences is a longstanding goal in machine learning and has led to architectures, such as state space models, that dramatically outperform Transformers on long sequences. However, these impressive empirical gains have been by and large demonstrated on benchmarks (e.g. Long Range Arena), where models are randomly initialized and trained to predict a target label from an input sequence. In this work, we show that random initialization leads to gross overestimation of the differences between architectures and that pretraining with standard denoising objectives, using $\textit{only the downstream task data}$, leads to dramatic gains across multiple architectures and to very small gaps between Transformers and state space models (SSMs). In stark contrast to prior works, we find vanilla Transformers to match the performance of S4 on Long Range Arena when properly pretrained, and we improve the best reported results of SSMs on the PathX-256 task by 20 absolute points. Subsequently, we analyze the utility of previously-proposed structured parameterizations for SSMs and show they become mostly redundant in the presence of data-driven initialization obtained through pretraining. Our work shows that, when evaluating different architectures on supervised tasks, incorporation of data-driven priors via pretraining is essential for reliable performance estimation, and can be done efficiently.

{{</citation>}}


### (36/117) Co-modeling the Sequential and Graphical Routes for Peptide Representation Learning (Zihan Liu et al., 2023)

{{<citation>}}

Zihan Liu, Ge Wang, Jiaqi Wang, Jiangbin Zheng, Stan Z. Li. (2023)  
**Co-modeling the Sequential and Graphical Routes for Peptide Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.02964v2)  

---


**ABSTRACT**  
Peptides are formed by the dehydration condensation of multiple amino acids. The primary structure of a peptide can be represented either as an amino acid sequence or as a molecular graph consisting of atoms and chemical bonds. Previous studies have indicated that deep learning routes specific to sequential and graphical peptide forms exhibit comparable performance on downstream tasks. Despite the fact that these models learn representations of the same modality of peptides, we find that they explain their predictions differently. Considering sequential and graphical models as two experts making inferences from different perspectives, we work on fusing expert knowledge to enrich the learned representations for improving the discriminative performance. To achieve this, we propose a peptide co-modeling method, RepCon, which employs a contrastive learning-based framework to enhance the mutual information of representations from decoupled sequential and graphical end-to-end models. It considers representations from the sequential encoder and the graphical encoder for the same peptide sample as a positive pair and learns to enhance the consistency of representations between positive sample pairs and to repel representations between negative pairs. Empirical studies of RepCon and other co-modeling methods are conducted on open-source discriminative datasets, including aggregation propensity, retention time, antimicrobial peptide prediction, and family classification from Peptide Database. Our results demonstrate the superiority of the co-modeling approach over independent modeling, as well as the superiority of RepCon over other methods under the co-modeling framework. In addition, the attribution on RepCon further corroborates the validity of the approach at the level of model explanation.

{{</citation>}}


### (37/117) Attention-based Multi-task Learning for Base Editor Outcome Prediction (Amina Mollaysa et al., 2023)

{{<citation>}}

Amina Mollaysa, Ahmed Allam, Michael Krauthammer. (2023)  
**Attention-based Multi-task Learning for Base Editor Outcome Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.02919v1)  

---


**ABSTRACT**  
Human genetic diseases often arise from point mutations, emphasizing the critical need for precise genome editing techniques. Among these, base editing stands out as it allows targeted alterations at the single nucleotide level. However, its clinical application is hindered by low editing efficiency and unintended mutations, necessitating extensive trial-and-error experimentation in the laboratory. To speed up this process, we present an attention-based two-stage machine learning model that learns to predict the likelihood of all possible editing outcomes for a given genomic target sequence. We further propose a multi-task learning schema to jointly learn multiple base editors (i.e. variants) at once. Our model's predictions consistently demonstrated a strong correlation with the actual experimental results on multiple datasets and base editor variants. These results provide further validation for the models' capacity to enhance and accelerate the process of refining base editing designs.

{{</citation>}}


### (38/117) FroSSL: Frobenius Norm Minimization for Self-Supervised Learning (Oscar Skean et al., 2023)

{{<citation>}}

Oscar Skean, Aayush Dhakal, Nathan Jacobs, Luis Gonzalo Sanchez Giraldo. (2023)  
**FroSSL: Frobenius Norm Minimization for Self-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: ImageNet, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.02903v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) is an increasingly popular paradigm for representation learning. Recent methods can be classified as sample-contrastive, dimension-contrastive, or asymmetric network-based, with each family having its own approach to avoiding informational collapse. While dimension-contrastive methods converge to similar solutions as sample-contrastive methods, it can be empirically shown that some methods require more epochs of training to converge. Motivated by closing this divide, we present the objective function FroSSL which is both sample- and dimension-contrastive up to embedding normalization. FroSSL works by minimizing covariance Frobenius norms for avoiding collapse and minimizing mean-squared error for augmentation invariance. We show that FroSSL converges more quickly than a variety of other SSL methods and provide theoretical and empirical support that this faster convergence is due to how FroSSL affects the eigenvalues of the embedding covariance matrices. We also show that FroSSL learns competitive representations on linear probe evaluation when used to train a ResNet18 on the CIFAR-10, CIFAR-100, STL-10, and ImageNet datasets.

{{</citation>}}


### (39/117) Searching for High-Value Molecules Using Reinforcement Learning and Transformers (Raj Ghugare et al., 2023)

{{<citation>}}

Raj Ghugare, Santiago Miret, Adriana Hugessen, Mariano Phielipp, Glen Berseth. (2023)  
**Searching for High-Value Molecules Using Reinforcement Learning and Transformers**  

---
Primary Category: cs.LG  
Categories: cond-mat-mtrl-sci, cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.02902v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) over text representations can be effective for finding high-value policies that can search over graphs. However, RL requires careful structuring of the search space and algorithm design to be effective in this challenge. Through extensive experiments, we explore how different design choices for text grammar and algorithmic choices for training can affect an RL policy's ability to generate molecules with desired properties. We arrive at a new RL-based molecular design algorithm (ChemRLformer) and perform a thorough analysis using 25 molecule design tasks, including computationally complex protein docking simulations. From this analysis, we discover unique insights in this problem space and show that ChemRLformer achieves state-of-the-art performance while being more straightforward than prior work by demystifying which design choices are actually helpful for text-based molecule design.

{{</citation>}}


### (40/117) Stable and Interpretable Deep Learning for Tabular Data: Introducing InterpreTabNet with the Novel InterpreStability Metric (Shiyun Wa et al., 2023)

{{<citation>}}

Shiyun Wa, Xinai Lu, Minjuan Wang. (2023)  
**Stable and Interpretable Deep Learning for Tabular Data: Introducing InterpreTabNet with the Novel InterpreStability Metric**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02870v1)  

---


**ABSTRACT**  
As Artificial Intelligence (AI) integrates deeper into diverse sectors, the quest for powerful models has intensified. While significant strides have been made in boosting model capabilities and their applicability across domains, a glaring challenge persists: many of these state-of-the-art models remain as black boxes. This opacity not only complicates the explanation of model decisions to end-users but also obstructs insights into intermediate processes for model designers. To address these challenges, we introduce InterpreTabNet, a model designed to enhance both classification accuracy and interpretability by leveraging the TabNet architecture with an improved attentive module. This design ensures robust gradient propagation and computational stability. Additionally, we present a novel evaluation metric, InterpreStability, which quantifies the stability of a model's interpretability. The proposed model and metric mark a significant stride forward in explainable models' research, setting a standard for transparency and interpretability in AI model design and application across diverse sectors. InterpreTabNet surpasses other leading solutions in tabular data analysis across varied application scenarios, paving the way for further research into creating deep-learning models that are both highly accurate and inherently explainable. The introduction of the InterpreStability metric ensures that the interpretability of future models can be measured and compared in a consistent and rigorous manner. Collectively, these contributions have the potential to promote the design principles and development of next-generation interpretable AI models, widening the adoption of interpretable AI solutions in critical decision-making environments.

{{</citation>}}


### (41/117) Rayleigh Quotient Graph Neural Networks for Graph-level Anomaly Detection (Xiangyu Dong et al., 2023)

{{<citation>}}

Xiangyu Dong, Xingyi Zhang, Sibo Wang. (2023)  
**Rayleigh Quotient Graph Neural Networks for Graph-level Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.02861v2)  

---


**ABSTRACT**  
Graph-level anomaly detection has gained significant attention as it finds many applications in various domains, such as cancer diagnosis and enzyme prediction. However, existing methods fail to capture the underlying properties of graph anomalies, resulting in unexplainable framework design and unsatisfying performance. In this paper, we take a step back and re-investigate the spectral differences between anomalous and normal graphs. Our main observation shows a significant disparity in the accumulated spectral energy between these two classes. Moreover, we prove that the accumulated spectral energy of the graph signal can be represented by its Rayleigh Quotient, indicating that the Rayleigh Quotient is a driving factor behind the anomalous properties of graphs. Motivated by this, we propose Rayleigh Quotient Graph Neural Network (RQGNN), the first spectral GNN for graph-level anomaly detection, providing a new perspective on exploring the inherent spectral features of anomalous graphs. Specifically, we introduce a novel framework that consists of two components: the Rayleigh Quotient learning component (RQL) and Chebyshev Wavelet GNN with RQ-pooling (CWGNN-RQ). RQL explicitly captures the Rayleigh Quotient of graphs and CWGNN-RQ implicitly explores the spectral space of graphs. Extensive experiments on 10 real-world datasets show that RQGNN outperforms the best rival by 6.74% in Macro-F1 score and 1.44% in AUC, demonstrating the effectiveness of our framework.

{{</citation>}}


### (42/117) Multi-Domain Causal Representation Learning via Weak Distributional Invariances (Kartik Ahuja et al., 2023)

{{<citation>}}

Kartik Ahuja, Amin Mansouri, Yixin Wang. (2023)  
**Multi-Domain Causal Representation Learning via Weak Distributional Invariances**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.02854v1)  

---


**ABSTRACT**  
Causal representation learning has emerged as the center of action in causal machine learning research. In particular, multi-domain datasets present a natural opportunity for showcasing the advantages of causal representation learning over standard unsupervised representation learning. While recent works have taken crucial steps towards learning causal representations, they often lack applicability to multi-domain datasets due to over-simplifying assumptions about the data; e.g. each domain comes from a different single-node perfect intervention. In this work, we relax these assumptions and capitalize on the following observation: there often exists a subset of latents whose certain distributional properties (e.g., support, variance) remain stable across domains; this property holds when, for example, each domain comes from a multi-node imperfect intervention. Leveraging this observation, we show that autoencoders that incorporate such invariances can provably identify the stable set of latents from the rest across different settings.

{{</citation>}}


### (43/117) Out-of-Distribution Detection by Leveraging Between-Layer Transformation Smoothness (Fran Jelenić et al., 2023)

{{<citation>}}

Fran Jelenić, Josip Jukić, Martin Tutek, Mate Puljiz, Jan Šnajder. (2023)  
**Out-of-Distribution Detection by Leveraging Between-Layer Transformation Smoothness**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.02832v1)  

---


**ABSTRACT**  
Effective OOD detection is crucial for reliable machine learning models, yet most current methods are limited in practical use due to requirements like access to training data or intervention in training. We present a novel method for detecting OOD data in deep neural networks based on transformation smoothness between intermediate layers of a network (BLOOD), which is applicable to pre-trained models without access to training data. BLOOD utilizes the tendency of between-layer representation transformations of in-distribution (ID) data to be smoother than the corresponding transformations of OOD data, a property that we also demonstrate empirically for Transformer networks. We evaluate BLOOD on several text classification tasks with Transformer networks and demonstrate that it outperforms methods with comparable resource requirements. Our analysis also suggests that when learning simpler tasks, OOD data transformations maintain their original sharpness, whereas sharpness increases with more complex tasks.

{{</citation>}}


### (44/117) Time-Series Classification in Smart Manufacturing Systems: An Experimental Evaluation of State-of-the-Art Machine Learning Algorithms (Mojtaba A. Farahani et al., 2023)

{{<citation>}}

Mojtaba A. Farahani, M. R. McCormick, Ramy Harik, Thorsten Wuest. (2023)  
**Time-Series Classification in Smart Manufacturing Systems: An Experimental Evaluation of State-of-the-Art Machine Learning Algorithms**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.02812v1)  

---


**ABSTRACT**  
Manufacturing is gathering extensive amounts of diverse data, thanks to the growing number of sensors and rapid advances in sensing technologies. Among the various data types available in SMS settings, time-series data plays a pivotal role. Hence, TSC emerges is crucial in this domain. The objective of this study is to fill this gap by providing a rigorous experimental evaluation of the SoTA ML and DL algorithms for TSC tasks in manufacturing and industrial settings. We first explored and compiled a comprehensive list of more than 92 SoTA algorithms from both TSC and manufacturing literature. Following, we selected the 36 most representative algorithms from this list. To evaluate their performance across various manufacturing classification tasks, we curated a set of 22 manufacturing datasets, representative of different characteristics that cover diverse manufacturing problems. Subsequently, we implemented and evaluated the algorithms on the manufacturing benchmark datasets, and analyzed the results for each dataset. Based on the results, ResNet, DrCIF, InceptionTime, and ARSENAL are the top-performing algorithms, boasting an average accuracy of over 96.6% across all 22 manufacturing TSC datasets. These findings underscore the robustness, efficiency, scalability, and effectiveness of convolutional kernels in capturing temporal features in time-series data, as three out of the top four performing algorithms leverage these kernels for feature extraction. Additionally, LSTM, BiLSTM, and TS-LSTM algorithms deserve recognition for their effectiveness in capturing features within time-series data using RNN-based structures.

{{</citation>}}


### (45/117) Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design (Matthew Thomas Jackson et al., 2023)

{{<citation>}}

Matthew Thomas Jackson, Minqi Jiang, Jack Parker-Holder, Risto Vuorio, Chris Lu, Gregory Farquhar, Shimon Whiteson, Jakob Nicolaus Foerster. (2023)  
**Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.02782v1)  

---


**ABSTRACT**  
The past decade has seen vast progress in deep reinforcement learning (RL) on the back of algorithms manually designed by human researchers. Recently, it has been shown that it is possible to meta-learn update rules, with the hope of discovering algorithms that can perform well on a wide range of RL tasks. Despite impressive initial results from algorithms such as Learned Policy Gradient (LPG), there remains a generalization gap when these algorithms are applied to unseen environments. In this work, we examine how characteristics of the meta-training distribution impact the generalization performance of these algorithms. Motivated by this analysis and building on ideas from Unsupervised Environment Design (UED), we propose a novel approach for automatically generating curricula to maximize the regret of a meta-learned optimizer, in addition to a novel approximation of regret, which we name algorithmic regret (AR). The result is our method, General RL Optimizers Obtained Via Environment Design (GROOVE). In a series of experiments, we show that GROOVE achieves superior generalization to LPG, and evaluate AR against baseline metrics from UED, identifying it as a critical component of environment design in this setting. We believe this approach is a step towards the discovery of truly general RL algorithms, capable of solving a wide range of real-world environments.

{{</citation>}}


### (46/117) Graph Neural Networks and Time Series as Directed Graphs for Quality Recognition (Angelica Simonetti et al., 2023)

{{<citation>}}

Angelica Simonetti, Ferdinando Zanchetta. (2023)  
**Graph Neural Networks and Time Series as Directed Graphs for Quality Recognition**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Time Series  
[Paper Link](http://arxiv.org/abs/2310.02774v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) are becoming central in the study of time series, coupled with existing algorithms as Temporal Convolutional Networks and Recurrent Neural Networks. In this paper, we see time series themselves as directed graphs, so that their topology encodes time dependencies and we start to explore the effectiveness of GNNs architectures on them. We develop two distinct Geometric Deep Learning models, a supervised classifier and an autoencoder-like model for signal reconstruction. We apply these models on a quality recognition problem.

{{</citation>}}


### (47/117) Deep Reinforcement Learning Algorithms for Hybrid V2X Communication: A Benchmarking Study (Fouzi Boukhalfa et al., 2023)

{{<citation>}}

Fouzi Boukhalfa, Reda Alami, Mastane Achab, Eric Moulines, Mehdi Bennis. (2023)  
**Deep Reinforcement Learning Algorithms for Hybrid V2X Communication: A Benchmarking Study**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG, eess-SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03767v1)  

---


**ABSTRACT**  
In today's era, autonomous vehicles demand a safety level on par with aircraft. Taking a cue from the aerospace industry, which relies on redundancy to achieve high reliability, the automotive sector can also leverage this concept by building redundancy in V2X (Vehicle-to-Everything) technologies. Given the current lack of reliable V2X technologies, this idea is particularly promising. By deploying multiple RATs (Radio Access Technologies) in parallel, the ongoing debate over the standard technology for future vehicles can be put to rest. However, coordinating multiple communication technologies is a complex task due to dynamic, time-varying channels and varying traffic conditions. This paper addresses the vertical handover problem in V2X using Deep Reinforcement Learning (DRL) algorithms. The goal is to assist vehicles in selecting the most appropriate V2X technology (DSRC/V-VLC) in a serpentine environment. The results show that the benchmarked algorithms outperform the current state-of-the-art approaches in terms of redundancy and usage rate of V-VLC headlights. This result is a significant reduction in communication costs while maintaining a high level of reliability. These results provide strong evidence for integrating advanced DRL decision mechanisms into the architecture as a promising approach to solving the vertical handover problem in V2X.

{{</citation>}}


### (48/117) Comparative Study and Framework for Automated Summariser Evaluation: LangChain and Hybrid Algorithms (Bagiya Lakshmi S et al., 2023)

{{<citation>}}

Bagiya Lakshmi S, Sanjjushri Varshini R, Rohith Mahadevan, Raja CSP Raman. (2023)  
**Comparative Study and Framework for Automated Summariser Evaluation: LangChain and Hybrid Algorithms**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02759v1)  

---


**ABSTRACT**  
Automated Essay Score (AES) is proven to be one of the cutting-edge technologies. Scoring techniques are used for various purposes. Reliable scores are calculated based on influential variables. Such variables can be computed by different methods based on the domain. The research is concentrated on the user's understanding of a given topic. The analysis is based on a scoring index by using Large Language Models. The user can then compare and contrast the understanding of a topic that they recently learned. The results are then contributed towards learning analytics and progression is made for enhancing the learning ability. In this research, the focus is on summarizing a PDF document and gauging a user's understanding of its content. The process involves utilizing a Langchain tool to summarize the PDF and extract the essential information. By employing this technique, the research aims to determine how well the user comprehends the summarized content.

{{</citation>}}


### (49/117) Comparative Analysis of Imbalanced Malware Byteplot Image Classification using Transfer Learning (Jayasudha M et al., 2023)

{{<citation>}}

Jayasudha M, Ayesha Shaik, Gaurav Pendharkar, Soham Kumar, Muhesh Kumar B, Sudharshanan Balaji. (2023)  
**Comparative Analysis of Imbalanced Malware Byteplot Image Classification using Transfer Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2310.02742v1)  

---


**ABSTRACT**  
Cybersecurity is a major concern due to the increasing reliance on technology and interconnected systems. Malware detectors help mitigate cyber-attacks by comparing malware signatures. Machine learning can improve these detectors by automating feature extraction, identifying patterns, and enhancing dynamic analysis. In this paper, the performance of six multiclass classification models is compared on the Malimg dataset, Blended dataset, and Malevis dataset to gain insights into the effect of class imbalance on model performance and convergence. It is observed that the more the class imbalance less the number of epochs required for convergence and a high variance across the performance of different models. Moreover, it is also observed that for malware detectors ResNet50, EfficientNetB0, and DenseNet169 can handle imbalanced and balanced data well. A maximum precision of 97% is obtained for the imbalanced dataset, a maximum precision of 95% is obtained on the intermediate imbalance dataset, and a maximum precision of 95% is obtained for the perfectly balanced dataset.

{{</citation>}}


### (50/117) scHyena: Foundation Model for Full-Length Single-Cell RNA-Seq Analysis in Brain (Gyutaek Oh et al., 2023)

{{<citation>}}

Gyutaek Oh, Baekgyu Choi, Inkyung Jung, Jong Chul Ye. (2023)  
**scHyena: Foundation Model for Full-Length Single-Cell RNA-Seq Analysis in Brain**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-GN, q-bio-QM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.02713v1)  

---


**ABSTRACT**  
Single-cell RNA sequencing (scRNA-seq) has made significant strides in unraveling the intricate cellular diversity within complex tissues. This is particularly critical in the brain, presenting a greater diversity of cell types than other tissue types, to gain a deeper understanding of brain function within various cellular contexts. However, analyzing scRNA-seq data remains a challenge due to inherent measurement noise stemming from dropout events and the limited utilization of extensive gene expression information. In this work, we introduce scHyena, a foundation model designed to address these challenges and enhance the accuracy of scRNA-seq analysis in the brain. Specifically, inspired by the recent Hyena operator, we design a novel Transformer architecture called singe-cell Hyena (scHyena) that is equipped with a linear adaptor layer, the positional encoding via gene-embedding, and a {bidirectional} Hyena operator. This enables us to process full-length scRNA-seq data without losing any information from the raw data. In particular, our model learns generalizable features of cells and genes through pre-training scHyena using the full length of scRNA-seq data. We demonstrate the superior performance of scHyena compared to other benchmark methods in downstream tasks, including cell type classification and scRNA-seq imputation.

{{</citation>}}


### (51/117) Memoria: Hebbian Memory Architecture for Human-Like Sequential Processing (Sangjun Park et al., 2023)

{{<citation>}}

Sangjun Park, JinYeong Bak. (2023)  
**Memoria: Hebbian Memory Architecture for Human-Like Sequential Processing**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: BERT, GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.03052v1)  

---


**ABSTRACT**  
Transformers have demonstrated their success in various domains and tasks. However, Transformers struggle with long input sequences due to their limited capacity. While one solution is to increase input length, endlessly stretching the length is unrealistic. Furthermore, humans selectively remember and use only relevant information from inputs, unlike Transformers which process all raw data from start to end. We introduce Memoria, a general memory network that applies Hebbian theory which is a major theory explaining human memory formulation to enhance long-term dependencies in neural networks. Memoria stores and retrieves information called engram at multiple memory levels of working memory, short-term memory, and long-term memory, using connection weights that change according to Hebb's rule. Through experiments with popular Transformer-based models like BERT and GPT, we present that Memoria significantly improves the ability to consider long-term dependencies in various tasks. Results show that Memoria outperformed existing methodologies in sorting and language modeling, and long text classification.

{{</citation>}}


### (52/117) PostRainBench: A comprehensive benchmark and a new model for precipitation forecasting (Yujin Tang et al., 2023)

{{<citation>}}

Yujin Tang, Jiaming Zhou, Xiang Pan, Zeying Gong, Junwei Liang. (2023)  
**PostRainBench: A comprehensive benchmark and a new model for precipitation forecasting**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2310.02676v2)  

---


**ABSTRACT**  
Accurate precipitation forecasting is a vital challenge of both scientific and societal importance. Data-driven approaches have emerged as a widely used solution for addressing this challenge. However, solely relying on data-driven approaches has limitations in modeling the underlying physics, making accurate predictions difficult. Coupling AI-based post-processing techniques with traditional Numerical Weather Prediction (NWP) methods offers a more effective solution for improving forecasting accuracy. Despite previous post-processing efforts, accurately predicting heavy rainfall remains challenging due to the imbalanced precipitation data across locations and complex relationships between multiple meteorological variables. To address these limitations, we introduce the PostRainBench, a comprehensive multi-variable NWP post-processing benchmark consisting of three datasets for NWP post-processing-based precipitation forecasting. We propose CAMT, a simple yet effective Channel Attention Enhanced Multi-task Learning framework with a specially designed weighted loss function. Its flexible design allows for easy plug-and-play integration with various backbones. Extensive experimental results on the proposed benchmark show that our method outperforms state-of-the-art methods by 6.3%, 4.7%, and 26.8% in rain CSI on the three datasets respectively. Most notably, our model is the first deep learning-based method to outperform traditional Numerical Weather Prediction (NWP) approaches in extreme precipitation conditions. It shows improvements of 15.6%, 17.4%, and 31.8% over NWP predictions in heavy rain CSI on respective datasets. These results highlight the potential impact of our model in reducing the severe consequences of extreme weather events.

{{</citation>}}


### (53/117) A Study of Quantisation-aware Training on Time Series Transformer Models for Resource-constrained FPGAs (Tianheng Ling et al., 2023)

{{<citation>}}

Tianheng Ling, Chao Qian, Lukas Einhaus, Gregor Schiele. (2023)  
**A Study of Quantisation-aware Training on Time Series Transformer Models for Resource-constrained FPGAs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-AR, cs-LG, cs.LG  
Keywords: QA, Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2310.02654v1)  

---


**ABSTRACT**  
This study explores the quantisation-aware training (QAT) on time series Transformer models. We propose a novel adaptive quantisation scheme that dynamically selects between symmetric and asymmetric schemes during the QAT phase. Our approach demonstrates that matching the quantisation scheme to the real data distribution can reduce computational overhead while maintaining acceptable precision. Moreover, our approach is robust when applied to real-world data and mixed-precision quantisation, where most objects are quantised to 4 bits. Our findings inform model quantisation and deployment decisions while providing a foundation for advancing quantisation techniques.

{{</citation>}}


### (54/117) Generative Modeling of Regular and Irregular Time Series Data via Koopman VAEs (Ilan Naiman et al., 2023)

{{<citation>}}

Ilan Naiman, N. Benjamin Erichson, Pu Ren, Michael W. Mahoney, Omri Azencot. (2023)  
**Generative Modeling of Regular and Irregular Time Series Data via Koopman VAEs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.02619v1)  

---


**ABSTRACT**  
Generating realistic time series data is important for many engineering and scientific applications. Existing work tackles this problem using generative adversarial networks (GANs). However, GANs are often unstable during training, and they can suffer from mode collapse. While variational autoencoders (VAEs) are known to be more robust to these issues, they are (surprisingly) less often considered for time series generation. In this work, we introduce Koopman VAE (KVAE), a new generative framework that is based on a novel design for the model prior, and that can be optimized for either regular and irregular training data. Inspired by Koopman theory, we represent the latent conditional prior dynamics using a linear map. Our approach enhances generative modeling with two desired features: (i) incorporating domain knowledge can be achieved by leverageing spectral tools that prescribe constraints on the eigenvalues of the linear map; and (ii) studying the qualitative behavior and stablity of the system can be performed using tools from dynamical systems theory. Our results show that KVAE outperforms state-of-the-art GAN and VAE methods across several challenging synthetic and real-world time series generation benchmarks. Whether trained on regular or irregular data, KVAE generates time series that improve both discriminative and predictive metrics. We also present visual evidence suggesting that KVAE learns probability density functions that better approximate empirical ground truth distributions.

{{</citation>}}


### (55/117) Learning adjacency matrix for dynamic graph neural network (Osama Ahmad et al., 2023)

{{<citation>}}

Osama Ahmad, Omer Abdul Jalil, Usman Nazir, Murtaza Taj. (2023)  
**Learning adjacency matrix for dynamic graph neural network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-IV  
Keywords: GNN, Graph Convolutional Network, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.02606v1)  

---


**ABSTRACT**  
In recent work, [1] introduced the concept of using a Block Adjacency Matrix (BA) for the representation of spatio-temporal data. While their method successfully concatenated adjacency matrices to encapsulate spatio-temporal relationships in a single graph, it formed a disconnected graph. This limitation hampered the ability of Graph Convolutional Networks (GCNs) to perform message passing across nodes belonging to different time steps, as no temporal links were present. To overcome this challenge, we introduce an encoder block specifically designed to learn these missing temporal links. The encoder block processes the BA and predicts connections between previously unconnected subgraphs, resulting in a Spatio-Temporal Block Adjacency Matrix (STBAM). This enriched matrix is then fed into a Graph Neural Network (GNN) to capture the complex spatio-temporal topology of the network. Our evaluations on benchmark datasets, surgVisDom and C2D2, demonstrate that our method, with slightly higher complexity, achieves superior results compared to state-of-the-art results. Our approach's computational overhead remains significantly lower than conventional non-graph-based methodologies for spatio-temporal data.

{{</citation>}}


### (56/117) Multi-Agent Reinforcement Learning for Power Grid Topology Optimization (Erica van der Sar et al., 2023)

{{<citation>}}

Erica van der Sar, Alessandro Zocca, Sandjai Bhulai. (2023)  
**Multi-Agent Reinforcement Learning for Power Grid Topology Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.02605v1)  

---


**ABSTRACT**  
Recent challenges in operating power networks arise from increasing energy demands and unpredictable renewable sources like wind and solar. While reinforcement learning (RL) shows promise in managing these networks, through topological actions like bus and line switching, efficiently handling large action spaces as networks grow is crucial. This paper presents a hierarchical multi-agent reinforcement learning (MARL) framework tailored for these expansive action spaces, leveraging the power grid's inherent hierarchical nature. Experimental results indicate the MARL framework's competitive performance with single-agent RL methods. We also compare different RL algorithms for lower-level agents alongside different policies for higher-order agents.

{{</citation>}}


### (57/117) On the Stability of Expressive Positional Encodings for Graph Neural Networks (Yinan Huang et al., 2023)

{{<citation>}}

Yinan Huang, William Lu, Joshua Robinson, Yu Yang, Muhan Zhang, Stefanie Jegelka, Pan Li. (2023)  
**On the Stability of Expressive Positional Encodings for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.02579v1)  

---


**ABSTRACT**  
Designing effective positional encodings for graphs is key to building powerful graph transformers and enhancing message-passing graph neural networks. Although widespread, using Laplacian eigenvectors as positional encodings faces two fundamental challenges: (1) \emph{Non-uniqueness}: there are many different eigendecompositions of the same Laplacian, and (2) \emph{Instability}: small perturbations to the Laplacian could result in completely different eigenspaces, leading to unpredictable changes in positional encoding.   Despite many attempts to address non-uniqueness, most methods overlook stability, leading to poor generalization on unseen graph structures. We identify the cause of instability to be a "hard partition" of eigenspaces. Hence, we introduce Stable and Expressive Positional Encodings (SPE), an architecture for processing eigenvectors that uses eigenvalues to "softly partition" eigenspaces. SPE is the first architecture that is (1) provably stable, and (2) universally expressive for basis invariant functions whilst respecting all symmetries of eigenvectors. Besides guaranteed stability, we prove that SPE is at least as expressive as existing methods, and highly capable of counting graph structures. Finally, we evaluate the effectiveness of our method on molecular property prediction, and out-of-distribution generalization tasks, finding improved generalization compared to existing positional encoding methods.

{{</citation>}}


### (58/117) Improving Knowledge Distillation with Teacher's Explanation (Sayantan Chowdhury et al., 2023)

{{<citation>}}

Sayantan Chowdhury, Ben Liang, Ali Tizghadam, Ilijc Albanese. (2023)  
**Improving Knowledge Distillation with Teacher's Explanation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.02572v1)  

---


**ABSTRACT**  
Knowledge distillation (KD) improves the performance of a low-complexity student model with the help of a more powerful teacher. The teacher in KD is a black-box model, imparting knowledge to the student only through its predictions. This limits the amount of transferred knowledge. In this work, we introduce a novel Knowledge Explaining Distillation (KED) framework, which allows the student to learn not only from the teacher's predictions but also from the teacher's explanations. We propose a class of superfeature-explaining teachers that provide explanation over groups of features, along with the corresponding student model. We also present a method for constructing the superfeatures. We then extend KED to reduce complexity in convolutional neural networks, to allow augmentation with hidden-representation distillation methods, and to work with a limited amount of training data using chimeric sets. Our experiments over a variety of datasets show that KED students can substantially outperform KD students of similar complexity.

{{</citation>}}


### (59/117) QuATON: Quantization Aware Training of Optical Neurons (Hasindu Kariyawasam et al., 2023)

{{<citation>}}

Hasindu Kariyawasam, Ramith Hettiarachchi, Dushan Wadduwage. (2023)  
**QuATON: Quantization Aware Training of Optical Neurons**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-IV, physics-optics  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2310.03049v1)  

---


**ABSTRACT**  
Optical neural architectures (ONAs) use coding elements with optimized physical parameters to perform intelligent measurements. However, fabricating ONAs while maintaining design performances is challenging. Limitations in fabrication techniques often limit the realizable precision of the trained parameters. Physical constraints may also limit the range of values the physical parameters can hold. Thus, ONAs should be trained within the implementable constraints. However, such physics-based constraints reduce the training objective to a constrained optimization problem, making it harder to optimize with existing gradient-based methods. To alleviate these critical issues that degrade performance from simulation to realization we propose a physics-informed quantization-aware training framework. Our approach accounts for the physical constraints during the training process, leading to robust designs. We evaluate our approach on an ONA proposed in the literature, named a diffractive deep neural network (D2NN), for all-optical phase imaging and for classification of phase objects. With extensive experiments on different quantization levels and datasets, we show that our approach leads to ONA designs that are robust to quantization noise.

{{</citation>}}


### (60/117) MedDiffusion: Boosting Health Risk Prediction via Diffusion-based Data Augmentation (Yuan Zhong et al., 2023)

{{<citation>}}

Yuan Zhong, Suhan Cui, Jiaqi Wang, Xiaochen Wang, Ziyi Yin, Yaqing Wang, Houping Xiao, Mengdi Huai, Ting Wang, Fenglong Ma. (2023)  
**MedDiffusion: Boosting Health Risk Prediction via Diffusion-based Data Augmentation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.02520v2)  

---


**ABSTRACT**  
Health risk prediction is one of the fundamental tasks under predictive modeling in the medical domain, which aims to forecast the potential health risks that patients may face in the future using their historical Electronic Health Records (EHR). Researchers have developed several risk prediction models to handle the unique challenges of EHR data, such as its sequential nature, high dimensionality, and inherent noise. These models have yielded impressive results. Nonetheless, a key issue undermining their effectiveness is data insufficiency. A variety of data generation and augmentation methods have been introduced to mitigate this issue by expanding the size of the training data set through the learning of underlying data distributions. However, the performance of these methods is often limited due to their task-unrelated design. To address these shortcomings, this paper introduces a novel, end-to-end diffusion-based risk prediction model, named MedDiffusion. It enhances risk prediction performance by creating synthetic patient data during training to enlarge sample space. Furthermore, MedDiffusion discerns hidden relationships between patient visits using a step-wise attention mechanism, enabling the model to automatically retain the most vital information for generating high-quality data. Experimental evaluation on four real-world medical datasets demonstrates that MedDiffusion outperforms 14 cutting-edge baselines in terms of PR-AUC, F1, and Cohen's Kappa. We also conduct ablation studies and benchmark our model against GAN-based alternatives to further validate the rationality and adaptability of our model design. Additionally, we analyze generated data to offer fresh insights into the model's interpretability.

{{</citation>}}


## cs.CY (3)



### (61/117) Generative AI in the Classroom: Can Students Remain Active Learners? (Rania Abdelghani et al., 2023)

{{<citation>}}

Rania Abdelghani, Hélène Sauzéon, Pierre-Yves Oudeyer. (2023)  
**Generative AI in the Classroom: Can Students Remain Active Learners?**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.03192v1)  

---


**ABSTRACT**  
Generative Artificial Intelligence (GAI) has high potential to help address a diversity of educational challenges. In principle, GAI could facilitate the implementation of interactive and empowering pedagogical activities to complement the standard teaching strategies and favor students active engagement, understanding and control over their learning processes. These dimensions are indeed fundamental for a better learning experience and longer-lasting cognitive outcomes. However, several characteristics of the interactions with GAI such as continuous confidence in the generated answers, and the lack of pedagogical stance in their behavior may lead students to poor states of control over learning (e.g. over-reliance on pre-generated content, over-estimation of one's own knowledge, loss of curious and critical-thinking sense, etc).   The fine line between the two settings seems to lie in how this technology is used to carry out the pedagogical activities (e.g. types of interactions allowed, level of controllability by students, level of involvement of educators, etc) as well as to what extent students have the relevant skills (cognitive, metacognitive and GAI literacy) that allow them to correctly evaluate, analyze and interpret the system behaviors.   In this context, this article proposes to identify some of the opportunities and challenges that could arise wrt students control over their learning when using GAI during formal pedagogical activities. In a second step, we also discuss the types of trainings that could be relevant to offer students in order to provide them with the appropriate set of skills that can help them use GAI in informed ways, when pursuing a given learning goal.

{{</citation>}}


### (62/117) Are LLMs Useful in the Poorest Schools? theTeacherAI in Sierra Leone (Jun Ho Choi et al., 2023)

{{<citation>}}

Jun Ho Choi, Oliver Garrod, Paul Atherton, Andrew Joyce-Gibbons, Miriam Mason-Sesay, Daniel Björkegren. (2023)  
**Are LLMs Useful in the Poorest Schools? theTeacherAI in Sierra Leone**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02982v1)  

---


**ABSTRACT**  
Education systems in developing countries have few resources to serve large, poor populations. How might generative AI integrate into classrooms? This paper introduces an AI chatbot designed to assist teachers in Sierra Leone with professional development to improve their instruction. We describe initial findings from early implementation across 122 schools and 193 teachers, and analyze its use with qualitative observations and by analyzing queries. Teachers use the system for lesson planning, classroom management, and subject matter. A subset of teachers use the system intensively. We draw conclusions from these findings about how generative AI systems can be integrated into school systems in low income countries.

{{</citation>}}


### (63/117) Who Audits the Auditors? Recommendations from a field scan of the algorithmic auditing ecosystem (Sasha Costanza-Chock et al., 2023)

{{<citation>}}

Sasha Costanza-Chock, Emma Harvey, Inioluwa Deborah Raji, Martha Czernuszenko, Joy Buolamwini. (2023)  
**Who Audits the Auditors? Recommendations from a field scan of the algorithmic auditing ecosystem**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02521v1)  

---


**ABSTRACT**  
AI audits are an increasingly popular mechanism for algorithmic accountability; however, they remain poorly defined. Without a clear understanding of audit practices, let alone widely used standards or regulatory guidance, claims that an AI product or system has been audited, whether by first-, second-, or third-party auditors, are difficult to verify and may exacerbate, rather than mitigate, bias and harm. To address this knowledge gap, we provide the first comprehensive field scan of the AI audit ecosystem. We share a catalog of individuals (N=438) and organizations (N=189) who engage in algorithmic audits or whose work is directly relevant to algorithmic audits; conduct an anonymous survey of the group (N=152); and interview industry leaders (N=10). We identify emerging best practices as well as methods and tools that are becoming commonplace, and enumerate common barriers to leveraging algorithmic audits as effective accountability mechanisms. We outline policy recommendations to improve the quality and impact of these audits, and highlight proposals with wide support from algorithmic auditors as well as areas of debate. Our recommendations have implications for lawmakers, regulators, internal company policymakers, and standards-setting bodies, as well as for auditors. They are: 1) require the owners and operators of AI systems to engage in independent algorithmic audits against clearly defined standards; 2) notify individuals when they are subject to algorithmic decision-making systems; 3) mandate disclosure of key components of audit findings for peer review; 4) consider real-world harm in the audit process, including through standardized harm incident reporting and response mechanisms; 5) directly involve the stakeholders most likely to be harmed by AI systems in the algorithmic audit process; and 6) formalize evaluation and, potentially, accreditation of algorithmic auditors.

{{</citation>}}


## cs.CR (2)



### (64/117) Misusing Tools in Large Language Models With Visual Adversarial Examples (Xiaohan Fu et al., 2023)

{{<citation>}}

Xiaohan Fu, Zihan Wang, Shuheng Li, Rajesh K. Gupta, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Earlence Fernandes. (2023)  
**Misusing Tools in Large Language Models With Visual Adversarial Examples**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.03185v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are being enhanced with the ability to use tools and to process multiple modalities. These new capabilities bring new benefits and also new security risks. In this work, we show that an attacker can use visual adversarial examples to cause attacker-desired tool usage. For example, the attacker could cause a victim LLM to delete calendar events, leak private conversations and book hotels. Different from prior work, our attacks can affect the confidentiality and integrity of user resources connected to the LLM while being stealthy and generalizable to multiple input prompts. We construct these attacks using gradient-based adversarial training and characterize performance along multiple dimensions. We find that our adversarial images can manipulate the LLM to invoke tools following real-world syntax almost always (~98%) while maintaining high similarity to clean images (~0.9 SSIM). Furthermore, using human scoring and automated metrics, we find that the attacks do not noticeably affect the conversation (and its semantics) between the user and the LLM.

{{</citation>}}


### (65/117) AGIR: Automating Cyber Threat Intelligence Reporting with Natural Language Generation (Filippo Perrina et al., 2023)

{{<citation>}}

Filippo Perrina, Francesco Marchiori, Mauro Conti, Nino Vincenzo Verde. (2023)  
**AGIR: Automating Cyber Threat Intelligence Reporting with Natural Language Generation**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs.CR  
Keywords: ChatGPT, GPT, Language Model, Natural Language Generation, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.02655v1)  

---


**ABSTRACT**  
Cyber Threat Intelligence (CTI) reporting is pivotal in contemporary risk management strategies. As the volume of CTI reports continues to surge, the demand for automated tools to streamline report generation becomes increasingly apparent. While Natural Language Processing techniques have shown potential in handling text data, they often struggle to address the complexity of diverse data sources and their intricate interrelationships. Moreover, established paradigms like STIX have emerged as de facto standards within the CTI community, emphasizing the formal categorization of entities and relations to facilitate consistent data sharing. In this paper, we introduce AGIR (Automatic Generation of Intelligence Reports), a transformative Natural Language Generation tool specifically designed to address the pressing challenges in the realm of CTI reporting. AGIR's primary objective is to empower security analysts by automating the labor-intensive task of generating comprehensive intelligence reports from formal representations of entity graphs. AGIR utilizes a two-stage pipeline by combining the advantages of template-based approaches and the capabilities of Large Language Models such as ChatGPT. We evaluate AGIR's report generation capabilities both quantitatively and qualitatively. The generated reports accurately convey information expressed through formal language, achieving a high recall value (0.99) without introducing hallucination. Furthermore, we compare the fluency and utility of the reports with state-of-the-art approaches, showing how AGIR achieves higher scores in terms of Syntactic Log-Odds Ratio (SLOR) and through questionnaires. By using our tool, we estimate that the report writing time is reduced by more than 40%, therefore streamlining the CTI production of any organization and contributing to the automation of several CTI tasks.

{{</citation>}}


## cs.CV (26)



### (66/117) Robust and Interpretable Medical Image Classifiers via Concept Bottleneck Models (An Yan et al., 2023)

{{<citation>}}

An Yan, Yu Wang, Yiwu Zhong, Zexue He, Petros Karypis, Zihan Wang, Chengyu Dong, Amilcare Gentili, Chun-Nan Hsu, Jingbo Shang, Julian McAuley. (2023)  
**Robust and Interpretable Medical Image Classifiers via Concept Bottleneck Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.03182v1)  

---


**ABSTRACT**  
Medical image classification is a critical problem for healthcare, with the potential to alleviate the workload of doctors and facilitate diagnoses of patients. However, two challenges arise when deploying deep learning models to real-world healthcare applications. First, neural models tend to learn spurious correlations instead of desired features, which could fall short when generalizing to new domains (e.g., patients with different ages). Second, these black-box models lack interpretability. When making diagnostic predictions, it is important to understand why a model makes a decision for trustworthy and safety considerations. In this paper, to address these two limitations, we propose a new paradigm to build robust and interpretable medical image classifiers with natural language concepts. Specifically, we first query clinical concepts from GPT-4, then transform latent image features into explicit concepts with a vision-language model. We systematically evaluate our method on eight medical image classification datasets to verify its effectiveness. On challenging datasets with strong confounding factors, our method can mitigate spurious correlations thus substantially outperform standard visual encoders and other baselines. Finally, we show how classification with a small number of concepts brings a level of interpretability for understanding model decisions through case studies in real medical data.

{{</citation>}}


### (67/117) ViFiT: Reconstructing Vision Trajectories from IMU and Wi-Fi Fine Time Measurements (Bryan Bo Cao et al., 2023)

{{<citation>}}

Bryan Bo Cao, Abrar Alali, Hansi Liu, Nicholas Meegan, Marco Gruteser, Kristin Dana, Ashwin Ashok, Shubham Jain. (2023)  
**ViFiT: Reconstructing Vision Trajectories from IMU and Wi-Fi Fine Time Measurements**  

---
Primary Category: cs.CV  
Categories: I-4-9; C-2-m, cs-CV, cs-MM, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.03140v1)  

---


**ABSTRACT**  
Tracking subjects in videos is one of the most widely used functions in camera-based IoT applications such as security surveillance, smart city traffic safety enhancement, vehicle to pedestrian communication and so on. In the computer vision domain, tracking is usually achieved by first detecting subjects with bounding boxes, then associating detected bounding boxes across video frames. For many IoT systems, images captured by cameras are usually sent over the network to be processed at a different site that has more powerful computing resources than edge devices. However, sending entire frames through the network causes significant bandwidth consumption that may exceed the system bandwidth constraints. To tackle this problem, we propose ViFiT, a transformer-based model that reconstructs vision bounding box trajectories from phone data (IMU and Fine Time Measurements). It leverages a transformer ability of better modeling long-term time series data. ViFiT is evaluated on Vi-Fi Dataset, a large-scale multimodal dataset in 5 diverse real world scenes, including indoor and outdoor environments. To fill the gap of proper metrics of jointly capturing the system characteristics of both tracking quality and video bandwidth reduction, we propose a novel evaluation framework dubbed Minimum Required Frames (MRF) and Minimum Required Frames Ratio (MRFR). ViFiT achieves an MRFR of 0.65 that outperforms the state-of-the-art approach for cross-modal reconstruction in LSTM Encoder-Decoder architecture X-Translator of 0.98, resulting in a high frame reduction rate as 97.76%.

{{</citation>}}


### (68/117) Shielding the Unseen: Privacy Protection through Poisoning NeRF with Spatial Deformation (Yihan Wu et al., 2023)

{{<citation>}}

Yihan Wu, Brandon Y. Feng, Heng Huang. (2023)  
**Shielding the Unseen: Privacy Protection through Poisoning NeRF with Spatial Deformation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03125v1)  

---


**ABSTRACT**  
In this paper, we introduce an innovative method of safeguarding user privacy against the generative capabilities of Neural Radiance Fields (NeRF) models. Our novel poisoning attack method induces changes to observed views that are imperceptible to the human eye, yet potent enough to disrupt NeRF's ability to accurately reconstruct a 3D scene. To achieve this, we devise a bi-level optimization algorithm incorporating a Projected Gradient Descent (PGD)-based spatial deformation. We extensively test our approach on two common NeRF benchmark datasets consisting of 29 real-world scenes with high-quality images. Our results compellingly demonstrate that our privacy-preserving method significantly impairs NeRF's performance across these benchmark datasets. Additionally, we show that our method is adaptable and versatile, functioning across various perturbation strengths and NeRF architectures. This work offers valuable insights into NeRF's vulnerabilities and emphasizes the need to account for such potential privacy risks when developing robust 3D scene reconstruction algorithms. Our study contributes to the larger conversation surrounding responsible AI and generative machine learning, aiming to protect user privacy and respect creative ownership in the digital age.

{{</citation>}}


### (69/117) Reinforcement Learning-based Mixture of Vision Transformers for Video Violence Recognition (Hamid Mohammadi et al., 2023)

{{<citation>}}

Hamid Mohammadi, Ehsan Nazerfard, Tahereh Firoozi. (2023)  
**Reinforcement Learning-based Mixture of Vision Transformers for Video Violence Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.03108v1)  

---


**ABSTRACT**  
Video violence recognition based on deep learning concerns accurate yet scalable human violence recognition. Currently, most state-of-the-art video violence recognition studies use CNN-based models to represent and categorize videos. However, recent studies suggest that pre-trained transformers are more accurate than CNN-based models on various video analysis benchmarks. Yet these models are not thoroughly evaluated for video violence recognition. This paper introduces a novel transformer-based Mixture of Experts (MoE) video violence recognition system. Through an intelligent combination of large vision transformers and efficient transformer architectures, the proposed system not only takes advantage of the vision transformer architecture but also reduces the cost of utilizing large vision transformers. The proposed architecture maximizes violence recognition system accuracy while actively reducing computational costs through a reinforcement learning-based router. The empirical results show the proposed MoE architecture's superiority over CNN-based models by achieving 92.4% accuracy on the RWF dataset.

{{</citation>}}


### (70/117) Reversing Deep Face Embeddings with Probable Privacy Protection (Daile Osorio-Roig et al., 2023)

{{<citation>}}

Daile Osorio-Roig, Paul A. Gerlitz, Christian Rathgeb, Christoph Busch. (2023)  
**Reversing Deep Face Embeddings with Probable Privacy Protection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.03005v1)  

---


**ABSTRACT**  
Generally, privacy-enhancing face recognition systems are designed to offer permanent protection of face embeddings. Recently, so-called soft-biometric privacy-enhancement approaches have been introduced with the aim of canceling soft-biometric attributes. These methods limit the amount of soft-biometric information (gender or skin-colour) that can be inferred from face embeddings. Previous work has underlined the need for research into rigorous evaluations and standardised evaluation protocols when assessing privacy protection capabilities. Motivated by this fact, this paper explores to what extent the non-invertibility requirement can be met by methods that claim to provide soft-biometric privacy protection. Additionally, a detailed vulnerability assessment of state-of-the-art face embedding extractors is analysed in terms of the transformation complexity used for privacy protection. In this context, a well-known state-of-the-art face image reconstruction approach has been evaluated on protected face embeddings to break soft biometric privacy protection. Experimental results show that biometric privacy-enhanced face embeddings can be reconstructed with an accuracy of up to approximately 98%, depending on the complexity of the protection algorithm.

{{</citation>}}


### (71/117) ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models (Yi-Lin Sung et al., 2023)

{{<citation>}}

Yi-Lin Sung, Jaehong Yoon, Mohit Bansal. (2023)  
**ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2310.02998v1)  

---


**ABSTRACT**  
Large Vision-Language Models (LVLMs) can understand the world comprehensively by integrating rich information from different modalities, achieving remarkable performance improvements on various multimodal downstream tasks. However, deploying LVLMs is often problematic due to their massive computational/energy costs and carbon consumption. Such issues make it infeasible to adopt conventional iterative global pruning, which is costly due to computing the Hessian matrix of the entire large model for sparsification. Alternatively, several studies have recently proposed layer-wise pruning approaches to avoid the expensive computation of global pruning and efficiently compress model weights according to their importance within a layer. However, these methods often suffer from suboptimal model compression due to their lack of a global perspective. To address this limitation in recent efficient pruning methods for large models, we propose Efficient Coarse-to-Fine Layer-Wise Pruning (ECoFLaP), a two-stage coarse-to-fine weight pruning approach for LVLMs. We first determine the sparsity ratios of different layers or blocks by leveraging the global importance score, which is efficiently computed based on the zeroth-order approximation of the global model gradients. Then, the multimodal model performs local layer-wise unstructured weight pruning based on globally-informed sparsity ratios. We validate our proposed method across various multimodal and unimodal models and datasets, demonstrating significant performance improvements over prevalent pruning techniques in the high-sparsity regime.

{{</citation>}}


### (72/117) Kosmos-G: Generating Images in Context with Multimodal Large Language Models (Xichen Pan et al., 2023)

{{<citation>}}

Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, Furu Wei. (2023)  
**Kosmos-G: Generating Images in Context with Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02992v1)  

---


**ABSTRACT**  
Recent advancements in text-to-image (T2I) and vision-language-to-image (VL2I) generation have made significant strides. However, the generation from generalized vision-language inputs, especially involving multiple images, remains under-explored. This paper presents Kosmos-G, a model that leverages the advanced perception capabilities of Multimodal Large Language Models (MLLMs) to tackle the aforementioned challenge. Our approach aligns the output space of MLLM with CLIP using the textual modality as an anchor and performs compositional instruction tuning on curated data. Kosmos-G demonstrates a unique capability of zero-shot multi-entity subject-driven generation. Notably, the score distillation instruction tuning requires no modifications to the image decoder. This allows for a seamless substitution of CLIP and effortless integration with a myriad of U-Net techniques ranging from fine-grained controls to personalized image decoder variants. We posit Kosmos-G as an initial attempt towards the goal of "image as a foreign language in image generation."

{{</citation>}}


### (73/117) Probing Intersectional Biases in Vision-Language Models with Counterfactual Examples (Phillip Howard et al., 2023)

{{<citation>}}

Phillip Howard, Avinash Madasu, Tiep Le, Gustavo Lujan Moreno, Vasudev Lal. (2023)  
**Probing Intersectional Biases in Vision-Language Models with Counterfactual Examples**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2310.02988v1)  

---


**ABSTRACT**  
While vision-language models (VLMs) have achieved remarkable performance improvements recently, there is growing evidence that these models also posses harmful biases with respect to social attributes such as gender and race. Prior studies have primarily focused on probing such bias attributes individually while ignoring biases associated with intersections between social attributes. This could be due to the difficulty of collecting an exhaustive set of image-text pairs for various combinations of social attributes from existing datasets. To address this challenge, we employ text-to-image diffusion models to produce counterfactual examples for probing intserctional social biases at scale. Our approach utilizes Stable Diffusion with cross attention control to produce sets of counterfactual image-text pairs that are highly similar in their depiction of a subject (e.g., a given occupation) while differing only in their depiction of intersectional social attributes (e.g., race & gender). We conduct extensive experiments using our generated dataset which reveal the intersectional social biases present in state-of-the-art VLMs.

{{</citation>}}


### (74/117) T$^3$Bench: Benchmarking Current Progress in Text-to-3D Generation (Yuze He et al., 2023)

{{<citation>}}

Yuze He, Yushi Bai, Matthieu Lin, Wang Zhao, Yubin Hu, Jenny Sheng, Ran Yi, Juanzi Li, Yong-Jin Liu. (2023)  
**T$^3$Bench: Benchmarking Current Progress in Text-to-3D Generation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02977v1)  

---


**ABSTRACT**  
Recent methods in text-to-3D leverage powerful pretrained diffusion models to optimize NeRF. Notably, these methods are able to produce high-quality 3D scenes without training on 3D data. Due to the open-ended nature of the task, most studies evaluate their results with subjective case studies and user experiments, thereby presenting a challenge in quantitatively addressing the question: How has current progress in Text-to-3D gone so far? In this paper, we introduce T$^3$Bench, the first comprehensive text-to-3D benchmark containing diverse text prompts of three increasing complexity levels that are specially designed for 3D generation. To assess both the subjective quality and the text alignment, we propose two automatic metrics based on multi-view images produced by the 3D contents. The quality metric combines multi-view text-image scores and regional convolution to detect quality and view inconsistency. The alignment metric uses multi-view captioning and Large Language Model (LLM) evaluation to measure text-3D consistency. Both metrics closely correlate with different dimensions of human judgments, providing a paradigm for efficiently evaluating text-to-3D models. The benchmarking results, shown in Fig. 1, reveal performance differences among six prevalent text-to-3D methods. Our analysis further highlights the common struggles for current methods on generating surroundings and multi-object scenes, as well as the bottleneck of leveraging 2D guidance for 3D generation. Our project page is available at: https://t3bench.com.

{{</citation>}}


### (75/117) CoDA: Collaborative Novel Box Discovery and Cross-modal Alignment for Open-vocabulary 3D Object Detection (Yang Cao et al., 2023)

{{<citation>}}

Yang Cao, Yihan Zeng, Hang Xu, Dan Xu. (2023)  
**CoDA: Collaborative Novel Box Discovery and Cross-modal Alignment for Open-vocabulary 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.02960v1)  

---


**ABSTRACT**  
Open-vocabulary 3D Object Detection (OV-3DDet) aims to detect objects from an arbitrary list of categories within a 3D scene, which remains seldom explored in the literature. There are primarily two fundamental problems in OV-3DDet, i.e., localizing and classifying novel objects. This paper aims at addressing the two problems simultaneously via a unified framework, under the condition of limited base categories. To localize novel 3D objects, we propose an effective 3D Novel Object Discovery strategy, which utilizes both the 3D box geometry priors and 2D semantic open-vocabulary priors to generate pseudo box labels of the novel objects. To classify novel object boxes, we further develop a cross-modal alignment module based on discovered novel boxes, to align feature spaces between 3D point cloud and image/text modalities. Specifically, the alignment process contains a class-agnostic and a class-discriminative alignment, incorporating not only the base objects with annotations but also the increasingly discovered novel objects, resulting in an iteratively enhanced alignment. The novel box discovery and crossmodal alignment are jointly learned to collaboratively benefit each other. The novel object discovery can directly impact the cross-modal alignment, while a better feature alignment can, in turn, boost the localization capability, leading to a unified OV-3DDet framework, named CoDA, for simultaneous novel object localization and classification. Extensive experiments on two challenging datasets (i.e., SUN-RGBD and ScanNet) demonstrate the effectiveness of our method and also show a significant mAP improvement upon the best-performing alternative method by 80%. Codes and pre-trained models are released on the project page.

{{</citation>}}


### (76/117) Graph data modelling for outcome prediction in oropharyngeal cancer patients (Nithya Bhasker et al., 2023)

{{<citation>}}

Nithya Bhasker, Stefan Leger, Alexander Zwanenburg, Chethan Babu Reddy, Sebastian Bodenstedt, Steffen Löck, Stefanie Speidel. (2023)  
**Graph data modelling for outcome prediction in oropharyngeal cancer patients**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.02931v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) are becoming increasingly popular in the medical domain for the tasks of disease classification and outcome prediction. Since patient data is not readily available as a graph, most existing methods either manually define a patient graph, or learn a latent graph based on pairwise similarities between the patients. There are also hypergraph neural network (HGNN)-based methods that were introduced recently to exploit potential higher order associations between the patients by representing them as a hypergraph. In this work, we propose a patient hypergraph network (PHGN), which has been investigated in an inductive learning setup for binary outcome prediction in oropharyngeal cancer (OPC) patients using computed tomography (CT)-based radiomic features for the first time. Additionally, the proposed model was extended to perform time-to-event analyses, and compared with GNN and baseline linear models.

{{</citation>}}


### (77/117) Delving into CLIP latent space for Video Anomaly Recognition (Luca Zanella et al., 2023)

{{<citation>}}

Luca Zanella, Benedetta Liberatori, Willi Menapace, Fabio Poiesi, Yiming Wang, Elisa Ricci. (2023)  
**Delving into CLIP latent space for Video Anomaly Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Recognition, Transformer  
[Paper Link](http://arxiv.org/abs/2310.02835v1)  

---


**ABSTRACT**  
We tackle the complex problem of detecting and recognising anomalies in surveillance videos at the frame level, utilising only video-level supervision. We introduce the novel method AnomalyCLIP, the first to combine Large Language and Vision (LLV) models, such as CLIP, with multiple instance learning for joint video anomaly detection and classification. Our approach specifically involves manipulating the latent CLIP feature space to identify the normal event subspace, which in turn allows us to effectively learn text-driven directions for abnormal events. When anomalous frames are projected onto these directions, they exhibit a large feature magnitude if they belong to a particular class. We also introduce a computationally efficient Transformer architecture to model short- and long-term temporal dependencies between frames, ultimately producing the final anomaly score and class prediction probabilities. We compare AnomalyCLIP against state-of-the-art methods considering three major anomaly detection benchmarks, i.e. ShanghaiTech, UCF-Crime, and XD-Violence, and empirically show that it outperforms baselines in recognising video anomalies.

{{</citation>}}


### (78/117) Improving Vision Anomaly Detection with the Guidance of Language Modality (Dong Chen et al., 2023)

{{<citation>}}

Dong Chen, Kaihang Pan, Guoming Wang, Yueting Zhuang, Siliang Tang. (2023)  
**Improving Vision Anomaly Detection with the Guidance of Language Modality**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection, Embedding  
[Paper Link](http://arxiv.org/abs/2310.02821v1)  

---


**ABSTRACT**  
Recent years have seen a surge of interest in anomaly detection for tackling industrial defect detection, event detection, etc. However, existing unsupervised anomaly detectors, particularly those for the vision modality, face significant challenges due to redundant information and sparse latent space. Conversely, the language modality performs well due to its relatively single data. This paper tackles the aforementioned challenges for vision modality from a multimodal point of view. Specifically, we propose Cross-modal Guidance (CMG), which consists of Cross-modal Entropy Reduction (CMER) and Cross-modal Linear Embedding (CMLE), to tackle the redundant information issue and sparse space issue, respectively. CMER masks parts of the raw image and computes the matching score with the text. Then, CMER discards irrelevant pixels to make the detector focus on critical contents. To learn a more compact latent space for the vision anomaly detector, CMLE learns a correlation structure matrix from the language modality, and then the latent space of vision modality will be learned with the guidance of the matrix. Thereafter, the vision latent space will get semantically similar images closer. Extensive experiments demonstrate the effectiveness of the proposed methods. Particularly, CMG outperforms the baseline that only uses images by 16.81%. Ablation experiments further confirm the synergy among the proposed methods, as each component depends on the other to achieve optimal performance.

{{</citation>}}


### (79/117) CoBEV: Elevating Roadside 3D Object Detection with Depth and Height Complementarity (Hao Shi et al., 2023)

{{<citation>}}

Hao Shi, Chengshan Pang, Jiaming Zhang, Kailun Yang, Yuhao Wu, Huajian Ni, Yining Lin, Rainer Stiefelhagen, Kaiwei Wang. (2023)  
**CoBEV: Elevating Roadside 3D Object Detection with Depth and Height Complementarity**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV, eess-IV  
Keywords: AI, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.02815v1)  

---


**ABSTRACT**  
Roadside camera-driven 3D object detection is a crucial task in intelligent transportation systems, which extends the perception range beyond the limitations of vision-centric vehicles and enhances road safety. While previous studies have limitations in using only depth or height information, we find both depth and height matter and they are in fact complementary. The depth feature encompasses precise geometric cues, whereas the height feature is primarily focused on distinguishing between various categories of height intervals, essentially providing semantic context. This insight motivates the development of Complementary-BEV (CoBEV), a novel end-to-end monocular 3D object detection framework that integrates depth and height to construct robust BEV representations. In essence, CoBEV estimates each pixel's depth and height distribution and lifts the camera features into 3D space for lateral fusion using the newly proposed two-stage complementary feature selection (CFS) module. A BEV feature distillation framework is also seamlessly integrated to further enhance the detection accuracy from the prior knowledge of the fusion-modal CoBEV teacher. We conduct extensive experiments on the public 3D detection benchmarks of roadside camera-based DAIR-V2X-I and Rope3D, as well as the private Supremind-Road dataset, demonstrating that CoBEV not only achieves the accuracy of the new state-of-the-art, but also significantly advances the robustness of previous methods in challenging long-distance scenarios and noisy camera disturbance, and enhances generalization by a large margin in heterologous settings with drastic changes in scene and camera parameters. For the first time, the vehicle AP score of a camera model reaches 80% on DAIR-V2X-I in terms of easy mode. The source code will be made publicly available at https://github.com/MasterHow/CoBEV.

{{</citation>}}


### (80/117) Dynamic Shuffle: An Efficient Channel Mixture Method (Kaijun Gong et al., 2023)

{{<citation>}}

Kaijun Gong, Zhuowen Yin, Yushu Li, Kailing Guo, Xiangmin Xu. (2023)  
**Dynamic Shuffle: An Efficient Channel Mixture Method**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.02776v1)  

---


**ABSTRACT**  
The redundancy of Convolutional neural networks not only depends on weights but also depends on inputs. Shuffling is an efficient operation for mixing channel information but the shuffle order is usually pre-defined. To reduce the data-dependent redundancy, we devise a dynamic shuffle module to generate data-dependent permutation matrices for shuffling. Since the dimension of permutation matrix is proportional to the square of the number of input channels, to make the generation process efficiently, we divide the channels into groups and generate two shared small permutation matrices for each group, and utilize Kronecker product and cross group shuffle to obtain the final permutation matrices. To make the generation process learnable, based on theoretical analysis, softmax, orthogonal regularization, and binarization are employed to asymptotically approximate the permutation matrix. Dynamic shuffle adaptively mixes channel information with negligible extra computation and memory occupancy. Experiment results on image classification benchmark datasets CIFAR-10, CIFAR-100, Tiny ImageNet and ImageNet have shown that our method significantly increases ShuffleNets' performance. Adding dynamic generated matrix with learnable static matrix, we further propose static-dynamic-shuffle and show that it can serve as a lightweight replacement of ordinary pointwise convolution.

{{</citation>}}


### (81/117) MUNCH: Modelling Unique 'N Controllable Heads (Debayan Deb et al., 2023)

{{<citation>}}

Debayan Deb, Suvidha Tripathi, Pranit Puri. (2023)  
**MUNCH: Modelling Unique 'N Controllable Heads**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.02753v1)  

---


**ABSTRACT**  
The automated generation of 3D human heads has been an intriguing and challenging task for computer vision researchers. Prevailing methods synthesize realistic avatars but with limited control over the diversity and quality of rendered outputs and suffer from limited correlation between shape and texture of the character. We propose a method that offers quality, diversity, control, and realism along with explainable network design, all desirable features to game-design artists in the domain. First, our proposed Geometry Generator identifies disentangled latent directions and generate novel and diverse samples. A Render Map Generator then learns to synthesize multiply high-fidelty physically-based render maps including Albedo, Glossiness, Specular, and Normals. For artists preferring fine-grained control over the output, we introduce a novel Color Transformer Model that allows semantic color control over generated maps. We also introduce quantifiable metrics called Uniqueness and Novelty and a combined metric to test the overall performance of our model. Demo for both shapes and textures can be found: https://munch-seven.vercel.app/. We will release our model along with the synthetic dataset.

{{</citation>}}


### (82/117) Land-cover change detection using paired OpenStreetMap data and optical high-resolution imagery via object-guided Transformer (Hongruixuan Chen et al., 2023)

{{<citation>}}

Hongruixuan Chen, Cuiling Lan, Jian Song, Clifford Broni-Bediako, Junshi Xia, Naoto Yokoya. (2023)  
**Land-cover change detection using paired OpenStreetMap data and optical high-resolution imagery via object-guided Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-CY, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.02674v1)  

---


**ABSTRACT**  
Optical high-resolution imagery and OpenStreetMap (OSM) data are two important data sources for land-cover change detection. Previous studies in these two data sources focus on utilizing the information in OSM data to aid the change detection on multi-temporal optical high-resolution images. This paper pioneers the direct detection of land-cover changes utilizing paired OSM data and optical imagery, thereby broadening the horizons of change detection tasks to encompass more dynamic earth observations. To this end, we propose an object-guided Transformer (ObjFormer) architecture by naturally combining the prevalent object-based image analysis (OBIA) technique with the advanced vision Transformer architecture. The introduction of OBIA can significantly reduce the computational overhead and memory burden in the self-attention module. Specifically, the proposed ObjFormer has a hierarchical pseudo-siamese encoder consisting of object-guided self-attention modules that extract representative features of different levels from OSM data and optical images; a decoder consisting of object-guided cross-attention modules can progressively recover the land-cover changes from the extracted heterogeneous features. In addition to the basic supervised binary change detection task, this paper raises a new semi-supervised semantic change detection task that does not require any manually annotated land-cover labels of optical images to train semantic change detectors. Two lightweight semantic decoders are added to ObjFormer to accomplish this task efficiently. A converse cross-entropy loss is designed to fully utilize the negative samples, thereby contributing to the great performance improvement in this task. The first large-scale benchmark dataset containing 1,287 map-image pairs (1024$\times$ 1024 pixels for each sample) covering 40 regions on six continents ...(see the manuscript for the full abstract)

{{</citation>}}


### (83/117) MedPrompt: Cross-Modal Prompting for Multi-Task Medical Image Translation (Xuhang Chen et al., 2023)

{{<citation>}}

Xuhang Chen, Chi-Man Pun, Shuqiang Wang. (2023)  
**MedPrompt: Cross-Modal Prompting for Multi-Task Medical Image Translation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.02663v1)  

---


**ABSTRACT**  
Cross-modal medical image translation is an essential task for synthesizing missing modality data for clinical diagnosis. However, current learning-based techniques have limitations in capturing cross-modal and global features, restricting their suitability to specific pairs of modalities. This lack of versatility undermines their practical usefulness, particularly considering that the missing modality may vary for different cases. In this study, we present MedPrompt, a multi-task framework that efficiently translates different modalities. Specifically, we propose the Self-adaptive Prompt Block, which dynamically guides the translation network towards distinct modalities. Within this framework, we introduce the Prompt Extraction Block and the Prompt Fusion Block to efficiently encode the cross-modal prompt. To enhance the extraction of global features across diverse modalities, we incorporate the Transformer model. Extensive experimental results involving five datasets and four pairs of modalities demonstrate that our proposed model achieves state-of-the-art visual quality and exhibits excellent generalization capability.

{{</citation>}}


### (84/117) GET: Group Event Transformer for Event-Based Vision (Yansong Peng et al., 2023)

{{<citation>}}

Yansong Peng, Yueyi Zhang, Zhiwei Xiong, Xiaoyan Sun, Feng Wu. (2023)  
**GET: Group Event Transformer for Event-Based Vision**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Self-Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.02642v1)  

---


**ABSTRACT**  
Event cameras are a type of novel neuromorphic sen-sor that has been gaining increasing attention. Existing event-based backbones mainly rely on image-based designs to extract spatial information within the image transformed from events, overlooking important event properties like time and polarity. To address this issue, we propose a novel Group-based vision Transformer backbone for Event-based vision, called Group Event Transformer (GET), which de-couples temporal-polarity information from spatial infor-mation throughout the feature extraction process. Specifi-cally, we first propose a new event representation for GET, named Group Token, which groups asynchronous events based on their timestamps and polarities. Then, GET ap-plies the Event Dual Self-Attention block, and Group Token Aggregation module to facilitate effective feature commu-nication and integration in both the spatial and temporal-polarity domains. After that, GET can be integrated with different downstream tasks by connecting it with vari-ous heads. We evaluate our method on four event-based classification datasets (Cifar10-DVS, N-MNIST, N-CARS, and DVS128Gesture) and two event-based object detection datasets (1Mpx and Gen1), and the results demonstrate that GET outperforms other state-of-the-art methods. The code is available at https://github.com/Peterande/GET-Group-Event-Transformer.

{{</citation>}}


### (85/117) ViT-ReciproCAM: Gradient and Attention-Free Visual Explanations for Vision Transformer (Seok-Yong Byun et al., 2023)

{{<citation>}}

Seok-Yong Byun, Wonju Lee. (2023)  
**ViT-ReciproCAM: Gradient and Attention-Free Visual Explanations for Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.02588v1)  

---


**ABSTRACT**  
This paper presents a novel approach to address the challenges of understanding the prediction process and debugging prediction errors in Vision Transformers (ViT), which have demonstrated superior performance in various computer vision tasks such as image classification and object detection. While several visual explainability techniques, such as CAM, Grad-CAM, Score-CAM, and Recipro-CAM, have been extensively researched for Convolutional Neural Networks (CNNs), limited research has been conducted on ViT. Current state-of-the-art solutions for ViT rely on class agnostic Attention-Rollout and Relevance techniques. In this work, we propose a new gradient-free visual explanation method for ViT, called ViT-ReciproCAM, which does not require attention matrix and gradient information. ViT-ReciproCAM utilizes token masking and generated new layer outputs from the target layer's input to exploit the correlation between activated tokens and network predictions for target classes. Our proposed method outperforms the state-of-the-art Relevance method in the Average Drop-Coherence-Complexity (ADCC) metric by $4.58\%$ to $5.80\%$ and generates more localized saliency maps. Our experiments demonstrate the effectiveness of ViT-ReciproCAM and showcase its potential for understanding and debugging ViT models. Our proposed method provides an efficient and easy-to-implement alternative for generating visual explanations, without requiring attention and gradient information, which can be beneficial for various applications in the field of computer vision.

{{</citation>}}


### (86/117) A Prototype-Based Neural Network for Image Anomaly Detection and Localization (Chao Huang et al., 2023)

{{<citation>}}

Chao Huang, Zhao Kang, Hong Wu. (2023)  
**A Prototype-Based Neural Network for Image Anomaly Detection and Localization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.02576v1)  

---


**ABSTRACT**  
Image anomaly detection and localization perform not only image-level anomaly classification but also locate pixel-level anomaly regions. Recently, it has received much research attention due to its wide application in various fields. This paper proposes ProtoAD, a prototype-based neural network for image anomaly detection and localization. First, the patch features of normal images are extracted by a deep network pre-trained on nature images. Then, the prototypes of the normal patch features are learned by non-parametric clustering. Finally, we construct an image anomaly localization network (ProtoAD) by appending the feature extraction network with $L2$ feature normalization, a $1\times1$ convolutional layer, a channel max-pooling, and a subtraction operation. We use the prototypes as the kernels of the $1\times1$ convolutional layer; therefore, our neural network does not need a training phase and can conduct anomaly detection and localization in an end-to-end manner. Extensive experiments on two challenging industrial anomaly detection datasets, MVTec AD and BTAD, demonstrate that ProtoAD achieves competitive performance compared to the state-of-the-art methods with a higher inference speed. The source code is available at: https://github.com/98chao/ProtoAD.

{{</citation>}}


### (87/117) ReForm-Eval: Evaluating Large Vision Language Models via Unified Re-Formulation of Task-Oriented Benchmarks (Zejun Li et al., 2023)

{{<citation>}}

Zejun Li, Ye Wang, Mengfei Du, Qingwen Liu, Binhao Wu, Jiwen Zhang, Chengxing Zhou, Zhihao Fan, Jie Fu, Jingjing Chen, Xuanjing Huang, Zhongyu Wei. (2023)  
**ReForm-Eval: Evaluating Large Vision Language Models via Unified Re-Formulation of Task-Oriented Benchmarks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02569v1)  

---


**ABSTRACT**  
Recent years have witnessed remarkable progress in the development of large vision-language models (LVLMs). Benefiting from the strong language backbones and efficient cross-modal alignment strategies, LVLMs exhibit surprising capabilities to perceive visual signals and perform visually grounded reasoning. However, the capabilities of LVLMs have not been comprehensively and quantitatively evaluate. Most existing multi-modal benchmarks require task-oriented input-output formats, posing great challenges to automatically assess the free-form text output of LVLMs. To effectively leverage the annotations available in existing benchmarks and reduce the manual effort required for constructing new benchmarks, we propose to re-formulate existing benchmarks into unified LVLM-compatible formats. Through systematic data collection and reformulation, we present the ReForm-Eval benchmark, offering substantial data for evaluating various capabilities of LVLMs. Based on ReForm-Eval, we conduct extensive experiments, thoroughly analyze the strengths and weaknesses of existing LVLMs, and identify the underlying factors. Our benchmark and evaluation framework will be open-sourced as a cornerstone for advancing the development of LVLMs.

{{</citation>}}


### (88/117) Improving Automatic VQA Evaluation Using Large Language Models (Oscar Mañas et al., 2023)

{{<citation>}}

Oscar Mañas, Benno Krojer, Aishwarya Agrawal. (2023)  
**Improving Automatic VQA Evaluation Using Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.02567v1)  

---


**ABSTRACT**  
8 years after the visual question answering (VQA) task was proposed, accuracy remains the primary metric for automatic evaluation. VQA Accuracy has been effective so far in the IID evaluation setting. However, our community is undergoing a shift towards open-ended generative models and OOD evaluation. In this new paradigm, the existing VQA Accuracy metric is overly stringent and underestimates the performance of VQA systems. Thus, there is a need to develop more robust automatic VQA metrics that serve as a proxy for human judgment. In this work, we propose to leverage the in-context learning capabilities of instruction-tuned large language models (LLMs) to build a better VQA metric. We formulate VQA evaluation as an answer-rating task where the LLM is instructed to score the accuracy of a candidate answer given a set of reference answers. We demonstrate the proposed metric better correlates with human judgment compared to existing metrics across several VQA models and benchmarks. We hope wide adoption of our metric will contribute to better estimating the research progress on the VQA task.

{{</citation>}}


### (89/117) SlowFormer: Universal Adversarial Patch for Attack on Compute and Energy Efficiency of Inference Efficient Vision Transformers (KL Navaneet et al., 2023)

{{<citation>}}

KL Navaneet, Soroush Abbasi Koohpayegani, Essam Sleiman, Hamed Pirsiavash. (2023)  
**SlowFormer: Universal Adversarial Patch for Attack on Compute and Energy Efficiency of Inference Efficient Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.02544v1)  

---


**ABSTRACT**  
Recently, there has been a lot of progress in reducing the computation of deep models at inference time. These methods can reduce both the computational needs and power usage of deep models. Some of these approaches adaptively scale the compute based on the input instance. We show that such models can be vulnerable to a universal adversarial patch attack, where the attacker optimizes for a patch that when pasted on any image, can increase the compute and power consumption of the model. We run experiments with three different efficient vision transformer methods showing that in some cases, the attacker can increase the computation to the maximum possible level by simply pasting a patch that occupies only 8\% of the image area. We also show that a standard adversarial training defense method can reduce some of the attack's success. We believe adaptive efficient methods will be necessary for the future to lower the power usage of deep models, so we hope our paper encourages the community to study the robustness of these methods and develop better defense methods for the proposed attack.

{{</citation>}}


### (90/117) On the Cognition of Visual Question Answering Models and Human Intelligence: A Comparative Study (Liben Chen et al., 2023)

{{<citation>}}

Liben Chen, Long Chen, Tian Ellison-Chen, Zhuoyuan Xu. (2023)  
**On the Cognition of Visual Question Answering Models and Human Intelligence: A Comparative Study**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.02528v1)  

---


**ABSTRACT**  
Visual Question Answering (VQA) is a challenging task that requires cross-modal understanding and reasoning of visual image and natural language question. To inspect the association of VQA models to human cognition, we designed a survey to record human thinking process and analyzed VQA models by comparing the outputs and attention maps with those of humans. We found that although the VQA models resemble human cognition in architecture and performs similarly with human on the recognition-level, they still struggle with cognitive inferences. The analysis of human thinking procedure serves to direct future research and introduce more cognitive capacity into modeling features and architectures.

{{</citation>}}


### (91/117) A Spatio-Temporal Attention-Based Method for Detecting Student Classroom Behaviors (Fan Yang, 2023)

{{<citation>}}

Fan Yang. (2023)  
**A Spatio-Temporal Attention-Based Method for Detecting Student Classroom Behaviors**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.02523v1)  

---


**ABSTRACT**  
Accurately detecting student behavior from classroom videos is beneficial for analyzing their classroom status and improving teaching efficiency. However, low accuracy in student classroom behavior detection is a prevalent issue. To address this issue, we propose a Spatio-Temporal Attention-Based Method for Detecting Student Classroom Behaviors (BDSTA). Firstly, the SlowFast network is used to generate motion and environmental information feature maps from the video. Then, the spatio-temporal attention module is applied to the feature maps, including information aggregation, compression and stimulation processes. Subsequently, attention maps in the time, channel and space dimensions are obtained, and multi-label behavior classification is performed based on these attention maps. To solve the long-tail data problem that exists in student classroom behavior datasets, we use an improved focal loss function to assign more weight to the tail class data during training. Experimental results are conducted on a self-made student classroom behavior dataset named STSCB. Compared with the SlowFast model, the average accuracy of student behavior classification detection improves by 8.94\% using BDSTA.

{{</citation>}}


## cs.SE (2)



### (92/117) MetaTool Benchmark: Deciding Whether to Use Tools and Which to Use (Yue Huang et al., 2023)

{{<citation>}}

Yue Huang, Jiawen Shi, Yuan Li, Chenrui Fan, Siyuan Wu, Qihui Zhang, Yixin Liu, Pan Zhou, Yao Wan, Neil Zhenqiang Gong, Lichao Sun. (2023)  
**MetaTool Benchmark: Deciding Whether to Use Tools and Which to Use**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2310.03128v1)  

---


**ABSTRACT**  
Large language models (LLMs) have garnered significant attention due to their impressive natural language processing (NLP) capabilities. Recently, many studies have focused on the tool utilization ability of LLMs. They primarily investigated how LLMs effectively collaborate with given specific tools. However, in scenarios where LLMs serve as intelligent agents, as seen in applications like AutoGPT and MetaGPT, LLMs are expected to engage in intricate decision-making processes that involve deciding whether to employ a tool and selecting the most suitable tool(s) from a collection of available tools to fulfill user requests. Therefore, in this paper, we introduce MetaTool, a benchmark designed to evaluate whether LLMs have tool usage awareness and can correctly choose tools. Specifically, we create a dataset called ToolE within the benchmark. This dataset contains various types of user queries in the form of prompts that trigger LLMs to use tools, including both single-tool and multi-tool scenarios. Subsequently, we set the tasks for both tool usage awareness and tool selection. We define four subtasks from different perspectives in tool selection, including tool selection with similar choices, tool selection in specific scenarios, tool selection with possible reliability issues, and multi-tool selection. We conduct experiments involving nine popular LLMs and find that the majority of them still struggle to effectively select tools, highlighting the existing gaps between LLMs and genuine intelligent agents. However, through the error analysis, we found there is still significant room for improvement. Finally, we conclude with insights for tool developers that follow ChatGPT to provide detailed descriptions that can enhance the tool selection performance of LLMs.

{{</citation>}}


### (93/117) Quantum Algorithm Cards: Streamlining the development of hybrid classical-quantum applications (Vlad Stirbu et al., 2023)

{{<citation>}}

Vlad Stirbu, Majid Haghparast. (2023)  
**Quantum Algorithm Cards: Streamlining the development of hybrid classical-quantum applications**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.02598v1)  

---


**ABSTRACT**  
The emergence of quantum computing proposes a revolutionary paradigm that can radically transform numerous scientific and industrial application domains. The ability of quantum computers to scale computations implies better performance and efficiency for certain algorithmic tasks than current computers provide. However, to gain benefit from such improvement, quantum computers must be integrated with existing software systems, a process that is not straightforward. In this paper, we investigate challenges that emerge when building larger hybrid classical-quantum computers and introduce the Quantum Algorithm Card (QAC) concept, an approach that could be employed to facilitate the decision making process around quantum technology.

{{</citation>}}


## eess.IV (4)



### (94/117) Blind CT Image Quality Assessment Using DDPM-derived Content and Transformer-based Evaluator (Yongyi Shi et al., 2023)

{{<citation>}}

Yongyi Shi, Wenjun Xia, Ge Wang, Xuanqin Mou. (2023)  
**Blind CT Image Quality Assessment Using DDPM-derived Content and Transformer-based Evaluator**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, QA, Transformer  
[Paper Link](http://arxiv.org/abs/2310.03118v1)  

---


**ABSTRACT**  
Lowering radiation dose per view and utilizing sparse views per scan are two common CT scan modes, albeit often leading to distorted images characterized by noise and streak artifacts. Blind image quality assessment (BIQA) strives to evaluate perceptual quality in alignment with what radiologists perceive, which plays an important role in advancing low-dose CT reconstruction techniques. An intriguing direction involves developing BIQA methods that mimic the operational characteristic of the human visual system (HVS). The internal generative mechanism (IGM) theory reveals that the HVS actively deduces primary content to enhance comprehension. In this study, we introduce an innovative BIQA metric that emulates the active inference process of IGM. Initially, an active inference module, implemented as a denoising diffusion probabilistic model (DDPM), is constructed to anticipate the primary content. Then, the dissimilarity map is derived by assessing the interrelation between the distorted image and its primary content. Subsequently, the distorted image and dissimilarity map are combined into a multi-channel image, which is inputted into a transformer-based image quality evaluator. Remarkably, by exclusively utilizing this transformer-based quality evaluator, we won the second place in the MICCAI 2023 low-dose computed tomography perceptual image quality assessment grand challenge. Leveraging the DDPM-derived primary content, our approach further improves the performance on the challenge dataset.

{{</citation>}}


### (95/117) Creating an Atlas of Normal Tissue for Pruning WSI Patching Through Anomaly Detection (Peyman Nejat et al., 2023)

{{<citation>}}

Peyman Nejat, Areej Alsaafin, Ghazal Alabtah, Nneka Comfere, Aaron Mangold, Dennis Murphree, Patricija Zot, Saba Yasir, Joaquin J. Garcia, H. R. Tizhoosh. (2023)  
**Creating an Atlas of Normal Tissue for Pruning WSI Patching Through Anomaly Detection**  

---
Primary Category: eess.IV  
Categories: 65D19 (Primary) 68P20, 68T20 (Secondary), cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Anomaly Detection, Pruning  
[Paper Link](http://arxiv.org/abs/2310.03106v1)  

---


**ABSTRACT**  
Patching gigapixel whole slide images (WSIs) is an important task in computational pathology. Some methods have been proposed to select a subset of patches as WSI representation for downstream tasks. While most of the computational pathology tasks are designed to classify or detect the presence of pathological lesions in each WSI, the confounding role and redundant nature of normal histology in tissue samples are generally overlooked in WSI representations. In this paper, we propose and validate the concept of an "atlas of normal tissue" solely using samples of WSIs obtained from normal tissue biopsies. Such atlases can be employed to eliminate normal fragments of tissue samples and hence increase the representativeness collection of patches. We tested our proposed method by establishing a normal atlas using 107 normal skin WSIs and demonstrated how established indexes and search engines like Yottixel can be improved. We used 553 WSIs of cutaneous squamous cell carcinoma (cSCC) to show the advantage. We also validated our method applied to an external dataset of 451 breast WSIs. The number of selected WSI patches was reduced by 30% to 50% after utilizing the proposed normal atlas while maintaining the same indexing and search performance in leave-one-patinet-out validation for both datasets. We show that the proposed normal atlas shows promise for unsupervised selection of the most representative patches of the abnormal/malignant WSI lesions.

{{</citation>}}


### (96/117) All Sizes Matter: Improving Volumetric Brain Segmentation on Small Lesions (Ayhan Can Erdur et al., 2023)

{{<citation>}}

Ayhan Can Erdur, Daniel Scholz, Josef A. Buchner, Stephanie E. Combs, Daniel Rueckert, Jan C. Peeken. (2023)  
**All Sizes Matter: Improving Volumetric Brain Segmentation on Small Lesions**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02829v1)  

---


**ABSTRACT**  
Brain metastases (BMs) are the most frequently occurring brain tumors. The treatment of patients having multiple BMs with stereo tactic radiosurgery necessitates accurate localization of the metastases. Neural networks can assist in this time-consuming and costly task that is typically performed by human experts. Particularly challenging is the detection of small lesions since they are often underrepresented in exist ing approaches. Yet, lesion detection is equally important for all sizes. In this work, we develop an ensemble of neural networks explicitly fo cused on detecting and segmenting small BMs. To accomplish this task, we trained several neural networks focusing on individual aspects of the BM segmentation problem: We use blob loss that specifically addresses the imbalance of lesion instances in terms of size and texture and is, therefore, not biased towards larger lesions. In addition, a model using a subtraction sequence between the T1 and T1 contrast-enhanced sequence focuses on low-contrast lesions. Furthermore, we train additional models only on small lesions. Our experiments demonstrate the utility of the ad ditional blob loss and the subtraction sequence. However, including the specialized small lesion models in the ensemble deteriorates segmentation results. We also find domain-knowledge-inspired postprocessing steps to drastically increase our performance in most experiments. Our approach enables us to submit a competitive challenge entry to the ASNR-MICCAI BraTS Brain Metastasis Challenge 2023.

{{</citation>}}


### (97/117) Multi-Dimension-Embedding-Aware Modality Fusion Transformer for Psychiatric Disorder Clasification (Guoxin Wang et al., 2023)

{{<citation>}}

Guoxin Wang, Xuyang Cao, Shan An, Fengmei Fan, Chao Zhang, Jinsong Wang, Feng Yu, Zhiren Wang. (2023)  
**Multi-Dimension-Embedding-Aware Modality Fusion Transformer for Psychiatric Disorder Clasification**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Embedding, Transformer  
[Paper Link](http://arxiv.org/abs/2310.02690v1)  

---


**ABSTRACT**  
Deep learning approaches, together with neuroimaging techniques, play an important role in psychiatric disorders classification. Previous studies on psychiatric disorders diagnosis mainly focus on using functional connectivity matrices of resting-state functional magnetic resonance imaging (rs-fMRI) as input, which still needs to fully utilize the rich temporal information of the time series of rs-fMRI data. In this work, we proposed a multi-dimension-embedding-aware modality fusion transformer (MFFormer) for schizophrenia and bipolar disorder classification using rs-fMRI and T1 weighted structural MRI (T1w sMRI). Concretely, to fully utilize the temporal information of rs-fMRI and spatial information of sMRI, we constructed a deep learning architecture that takes as input 2D time series of rs-fMRI and 3D volumes T1w. Furthermore, to promote intra-modality attention and information fusion across different modalities, a fusion transformer module (FTM) is designed through extensive self-attention of hybrid feature maps of multi-modality. In addition, a dimension-up and dimension-down strategy is suggested to properly align feature maps of multi-dimensional from different modalities. Experimental results on our private and public OpenfMRI datasets show that our proposed MFFormer performs better than that using a single modality or multi-modality MRI on schizophrenia and bipolar disorder diagnosis.

{{</citation>}}


## stat.ML (4)



### (98/117) Leveraging Model-based Trees as Interpretable Surrogate Models for Model Distillation (Julia Herbinger et al., 2023)

{{<citation>}}

Julia Herbinger, Susanne Dandl, Fiona K. Ewald, Sofia Loibl, Giuseppe Casalicchio. (2023)  
**Leveraging Model-based Trees as Interpretable Surrogate Models for Model Distillation**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Model Distillation  
[Paper Link](http://arxiv.org/abs/2310.03112v1)  

---


**ABSTRACT**  
Surrogate models play a crucial role in retrospectively interpreting complex and powerful black box machine learning models via model distillation. This paper focuses on using model-based trees as surrogate models which partition the feature space into interpretable regions via decision rules. Within each region, interpretable models based on additive main effects are used to approximate the behavior of the black box model, striking for an optimal balance between interpretability and performance. Four model-based tree algorithms, namely SLIM, GUIDE, MOB, and CTree, are compared regarding their ability to generate such surrogate models. We investigate fidelity, interpretability, stability, and the algorithms' capability to capture interaction effects through appropriate splits. Based on our comprehensive analyses, we finally provide an overview of user-specific recommendations.

{{</citation>}}


### (99/117) xVal: A Continuous Number Encoding for Large Language Models (Siavash Golkar et al., 2023)

{{<citation>}}

Siavash Golkar, Mariel Pettee, Michael Eickenberg, Alberto Bietti, Miles Cranmer, Geraud Krawezik, Francois Lanusse, Michael McCabe, Ruben Ohana, Liam Parker, Bruno Régaldo-Saint Blancard, Tiberiu Tesileanu, Kyunghyun Cho, Shirley Ho. (2023)  
**xVal: A Continuous Number Encoding for Large Language Models**  

---
Primary Category: stat.ML  
Categories: cs-AI, cs-CL, cs-LG, stat-ML, stat.ML  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.02989v1)  

---


**ABSTRACT**  
Large Language Models have not yet been broadly adapted for the analysis of scientific datasets due in part to the unique difficulties of tokenizing numbers. We propose xVal, a numerical encoding scheme that represents any real number using just a single token. xVal represents a given real number by scaling a dedicated embedding vector by the number value. Combined with a modified number-inference approach, this strategy renders the model end-to-end continuous when considered as a map from the numbers of the input string to those of the output string. This leads to an inductive bias that is generally more suitable for applications in scientific domains. We empirically evaluate our proposal on a number of synthetic and real-world datasets. Compared with existing number encoding schemes, we find that xVal is more token-efficient and demonstrates improved generalization.

{{</citation>}}


### (100/117) Functional trustworthiness of AI systems by statistically valid testing (Bernhard Nessler et al., 2023)

{{<citation>}}

Bernhard Nessler, Thomas Doms, Sepp Hochreiter. (2023)  
**Functional trustworthiness of AI systems by statistically valid testing**  

---
Primary Category: stat.ML  
Categories: cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02727v1)  

---


**ABSTRACT**  
The authors are concerned about the safety, health, and rights of the European citizens due to inadequate measures and procedures required by the current draft of the EU Artificial Intelligence (AI) Act for the conformity assessment of AI systems. We observe that not only the current draft of the EU AI Act, but also the accompanying standardization efforts in CEN/CENELEC, have resorted to the position that real functional guarantees of AI systems supposedly would be unrealistic and too complex anyways. Yet enacting a conformity assessment procedure that creates the false illusion of trust in insufficiently assessed AI systems is at best naive and at worst grossly negligent. The EU AI Act thus misses the point of ensuring quality by functional trustworthiness and correctly attributing responsibilities.   The trustworthiness of an AI decision system lies first and foremost in the correct statistical testing on randomly selected samples and in the precision of the definition of the application domain, which enables drawing samples in the first place. We will subsequently call this testable quality functional trustworthiness. It includes a design, development, and deployment that enables correct statistical testing of all relevant functions.   We are firmly convinced and advocate that a reliable assessment of the statistical functional properties of an AI system has to be the indispensable, mandatory nucleus of the conformity assessment. In this paper, we describe the three necessary elements to establish a reliable functional trustworthiness, i.e., (1) the definition of the technical distribution of the application, (2) the risk-based minimum performance requirements, and (3) the statistically valid testing based on independent random samples.

{{</citation>}}


### (101/117) Online Estimation and Inference for Robust Policy Evaluation in Reinforcement Learning (Weidong Liu et al., 2023)

{{<citation>}}

Weidong Liu, Jiyuan Tu, Yichen Zhang, Xi Chen. (2023)  
**Online Estimation and Inference for Robust Policy Evaluation in Reinforcement Learning**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.02581v1)  

---


**ABSTRACT**  
Recently, reinforcement learning has gained prominence in modern statistics, with policy evaluation being a key component. Unlike traditional machine learning literature on this topic, our work places emphasis on statistical inference for the parameter estimates computed using reinforcement learning algorithms. While most existing analyses assume random rewards to follow standard distributions, limiting their applicability, we embrace the concept of robust statistics in reinforcement learning by simultaneously addressing issues of outlier contamination and heavy-tailed rewards within a unified framework. In this paper, we develop an online robust policy evaluation procedure, and establish the limiting distribution of our estimator, based on its Bahadur representation. Furthermore, we develop a fully-online procedure to efficiently conduct statistical inference based on the asymptotic distribution. This paper bridges the gap between robust statistics and statistical inference in reinforcement learning, offering a more versatile and reliable approach to policy evaluation. Finally, we validate the efficacy of our algorithm through numerical experiments conducted in real-world reinforcement learning experiments.

{{</citation>}}


## cs.RO (7)



### (102/117) LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving (Hao Sha et al., 2023)

{{<citation>}}

Hao Sha, Yao Mu, Yuxuan Jiang, Li Chen, Chenfeng Xu, Ping Luo, Shengbo Eben Li, Masayoshi Tomizuka, Wei Zhan, Mingyu Ding. (2023)  
**LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.03026v1)  

---


**ABSTRACT**  
Existing learning-based autonomous driving (AD) systems face challenges in comprehending high-level information, generalizing to rare events, and providing interpretability. To address these problems, this work employs Large Language Models (LLMs) as a decision-making component for complex AD scenarios that require human commonsense understanding. We devise cognitive pathways to enable comprehensive reasoning with LLMs, and develop algorithms for translating LLM decisions into actionable driving commands. Through this approach, LLM decisions are seamlessly integrated with low-level controllers by guided parameter matrix adaptation. Extensive experiments demonstrate that our proposed method not only consistently surpasses baseline approaches in single-vehicle tasks, but also helps handle complex driving behaviors even multi-vehicle coordination, thanks to the commonsense reasoning capabilities of LLMs. This paper presents an initial step toward leveraging LLMs as effective decision-makers for intricate AD scenarios in terms of safety, efficiency, generalizability, and interoperability. We aspire for it to serve as inspiration for future research in this field. Project page: https://sites.google.com/view/llm-mpc

{{</citation>}}


### (103/117) Human-oriented Representation Learning for Robotic Manipulation (Mingxiao Huo et al., 2023)

{{<citation>}}

Mingxiao Huo, Mingyu Ding, Chenfeng Xu, Thomas Tian, Xinghao Zhu, Yao Mu, Lingfeng Sun, Masayoshi Tomizuka, Wei Zhan. (2023)  
**Human-oriented Representation Learning for Robotic Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.03023v1)  

---


**ABSTRACT**  
Humans inherently possess generalizable visual representations that empower them to efficiently explore and interact with the environments in manipulation tasks. We advocate that such a representation automatically arises from simultaneously learning about multiple simple perceptual skills that are critical for everyday scenarios (e.g., hand detection, state estimate, etc.) and is better suited for learning robot manipulation policies compared to current state-of-the-art visual representations purely based on self-supervised objectives. We formalize this idea through the lens of human-oriented multi-task fine-tuning on top of pre-trained visual encoders, where each task is a perceptual skill tied to human-environment interactions. We introduce Task Fusion Decoder as a plug-and-play embedding translator that utilizes the underlying relationships among these perceptual skills to guide the representation learning towards encoding meaningful structure for what's important for all perceptual skills, ultimately empowering learning of downstream robotic manipulation tasks. Extensive experiments across a range of robotic tasks and embodiments, in both simulations and real-world environments, show that our Task Fusion Decoder consistently improves the representation of three state-of-the-art visual encoders including R3M, MVP, and EgoVLP, for downstream manipulation policy-learning. Project page: https://sites.google.com/view/human-oriented-robot-learning

{{</citation>}}


### (104/117) Optimal Collaborative Transportation for Under-Capacitated Vehicle Routing Problems using Aerial Drone Swarms (Akash Kopparam Sreedhara et al., 2023)

{{<citation>}}

Akash Kopparam Sreedhara, Deepesh Padala, Shashank Mahesh, Kai Cui, Mengguang Li, Heinz Koeppl. (2023)  
**Optimal Collaborative Transportation for Under-Capacitated Vehicle Routing Problems using Aerial Drone Swarms**  

---
Primary Category: cs.RO  
Categories: cs-MA, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.02726v1)  

---


**ABSTRACT**  
Swarms of aerial drones have recently been considered for last-mile deliveries in urban logistics or automated construction. At the same time, collaborative transportation of payloads by multiple drones is another important area of recent research. However, efficient coordination algorithms for collaborative transportation of many payloads by many drones remain to be considered. In this work, we formulate the collaborative transportation of payloads by a swarm of drones as a novel, under-capacitated generalization of vehicle routing problems (VRP), which may also be of separate interest. In contrast to standard VRP and capacitated VRP, we must additionally consider waiting times for payloads lifted cooperatively by multiple drones, and the corresponding coordination. Algorithmically, we provide a solution encoding that avoids deadlocks and formulate an appropriate alternating minimization scheme to solve the problem. On the hardware side, we integrate our algorithms with collision avoidance and drone controllers. The approach and the impact of the system integration are successfully verified empirically, both on a swarm of real nano-quadcopters and for large swarms in simulation. Overall, we provide a framework for collaborative transportation with aerial drone swarms, that uses only as many drones as necessary for the transportation of any single payload.

{{</citation>}}


### (105/117) Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance (Weirui Ye et al., 2023)

{{<citation>}}

Weirui Ye, Yunsheng Zhang, Mengchen Wang, Shengjie Wang, Xianfan Gu, Pieter Abbeel, Yang Gao. (2023)  
**Foundation Reinforcement Learning: towards Embodied Generalist Agents with Foundation Prior Assistance**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: NLP, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.02635v1)  

---


**ABSTRACT**  
Recently, people have shown that large-scale pre-training from internet-scale data is the key to building generalist models, as witnessed in NLP. To build embodied generalist agents, we and many other researchers hypothesize that such foundation prior is also an indispensable component. However, it is unclear what is the proper concrete form to represent those embodied foundation priors and how they should be used in the downstream task. In this paper, we propose an intuitive and effective set of embodied priors that consist of foundation policy, value, and success reward. The proposed priors are based on the goal-conditioned MDP. To verify their effectiveness, we instantiate an actor-critic method assisted by the priors, called Foundation Actor-Critic (FAC). We name our framework as Foundation Reinforcement Learning (FRL), since it completely relies on embodied foundation priors to explore, learn and reinforce. The benefits of FRL are threefold. (1) Sample efficient. With foundation priors, FAC learns significantly faster than traditional RL. Our evaluation on the Meta-World has proved that FAC can achieve 100% success rates for 7/8 tasks under less than 200k frames, which outperforms the baseline method with careful manual-designed rewards under 1M frames. (2) Robust to noisy priors. Our method tolerates the unavoidable noise in embodied foundation models. We show that FAC works well even under heavy noise or quantization errors. (3) Minimal human intervention: FAC completely learns from the foundation priors, without the need of human-specified dense reward, or providing teleoperated demos. Thus, FAC can be easily scaled up. We believe our FRL framework could enable the future robot to autonomously explore and learn without human intervention in the physical world. In summary, our proposed FRL is a novel and powerful learning paradigm, towards achieving embodied generalist agents.

{{</citation>}}


### (106/117) Robust Collision Detection for Robots with Variable Stiffness Actuation by Using MAD-CNN: Modularized-Attention-Dilated Convolutional Neural Network (Zhenwei Niu et al., 2023)

{{<citation>}}

Zhenwei Niu, Lyes Saad Saoud, Irfan Hussain. (2023)  
**Robust Collision Detection for Robots with Variable Stiffness Actuation by Using MAD-CNN: Modularized-Attention-Dilated Convolutional Neural Network**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-RO, cs.RO  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.02573v1)  

---


**ABSTRACT**  
Ensuring safety is paramount in the field of collaborative robotics to mitigate the risks of human injury and environmental damage. Apart from collision avoidance, it is crucial for robots to rapidly detect and respond to unexpected collisions. While several learning-based collision detection methods have been introduced as alternatives to purely model-based detection techniques, there is currently a lack of such methods designed for collaborative robots equipped with variable stiffness actuators. Moreover, there is potential for further enhancing the network's robustness and improving the efficiency of data training. In this paper, we propose a new network, the Modularized Attention-Dilated Convolutional Neural Network (MAD-CNN), for collision detection in robots equipped with variable stiffness actuators. Our model incorporates a dual inductive bias mechanism and an attention module to enhance data efficiency and improve robustness. In particular, MAD-CNN is trained using only a four-minute collision dataset focusing on the highest level of joint stiffness. Despite limited training data, MAD-CNN robustly detects all collisions with minimal detection delay across various stiffness conditions. Moreover, it exhibits a higher level of collision sensitivity, which is beneficial for effectively handling false positives, which is a common issue in learning-based methods. Experimental results demonstrate that the proposed MAD-CNN model outperforms existing state-of-the-art models in terms of collision sensitivity and robustness.

{{</citation>}}


### (107/117) Improving Drumming Robot Via Attention Transformer Network (Yang Yi et al., 2023)

{{<citation>}}

Yang Yi, Zonghan Li. (2023)  
**Improving Drumming Robot Via Attention Transformer Network**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.02565v1)  

---


**ABSTRACT**  
Robotic technology has been widely used in nowadays society, which has made great progress in various fields such as agriculture, manufacturing and entertainment. In this paper, we focus on the topic of drumming robots in entertainment. To this end, we introduce an improving drumming robot that can automatically complete music transcription based on the popular vision transformer network based on the attention mechanism. Equipped with the attention transformer network, our method can efficiently handle the sequential audio embedding input and model their global long-range dependencies. Massive experimental results demonstrate that the improving algorithm can help the drumming robot promote drum classification performance, which can also help the robot to enjoy a variety of smart applications and services.

{{</citation>}}


### (108/117) Proactive Human-Robot Interaction using Visuo-Lingual Transformers (Pranay Mathur, 2023)

{{<citation>}}

Pranay Mathur. (2023)  
**Proactive Human-Robot Interaction using Visuo-Lingual Transformers**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.02506v1)  

---


**ABSTRACT**  
Humans possess the innate ability to extract latent visuo-lingual cues to infer context through human interaction. During collaboration, this enables proactive prediction of the underlying intention of a series of tasks. In contrast, robotic agents collaborating with humans naively follow elementary instructions to complete tasks or use specific hand-crafted triggers to initiate proactive collaboration when working towards the completion of a goal. Endowing such robots with the ability to reason about the end goal and proactively suggest intermediate tasks will engender a much more intuitive method for human-robot collaboration. To this end, we propose a learning-based method that uses visual cues from the scene, lingual commands from a user and knowledge of prior object-object interaction to identify and proactively predict the underlying goal the user intends to achieve. Specifically, we propose ViLing-MMT, a vision-language multimodal transformer-based architecture that captures inter and intra-modal dependencies to provide accurate scene descriptions and proactively suggest tasks where applicable. We evaluate our proposed model in simulation and real-world scenarios.

{{</citation>}}


## eess.AS (2)



### (109/117) Zero Resource Code-switched Speech Benchmark Using Speech Utterance Pairs For Multiple Spoken Languages (Kuan-Po Huang et al., 2023)

{{<citation>}}

Kuan-Po Huang, Chih-Kai Yang, Yu-Kuan Fu, Ewan Dunbar, Hung-yi Lee. (2023)  
**Zero Resource Code-switched Speech Benchmark Using Speech Utterance Pairs For Multiple Spoken Languages**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.03018v1)  

---


**ABSTRACT**  
We introduce a new zero resource code-switched speech benchmark designed to directly assess the code-switching capabilities of self-supervised speech encoders. We showcase a baseline system of language modeling on discrete units to demonstrate how the code-switching abilities of speech encoders can be assessed in a zero-resource manner. Our experiments encompass a variety of well-known speech encoders, including Wav2vec 2.0, HuBERT, XLSR, etc. We examine the impact of pre-training languages and model size on benchmark performance. Notably, though our results demonstrate that speech encoders with multilingual pre-training, exemplified by XLSR, outperform monolingual variants (Wav2vec 2.0, HuBERT) in code-switching scenarios, there is still substantial room for improvement in their code-switching linguistic abilities.

{{</citation>}}


### (110/117) Continual Contrastive Spoken Language Understanding (Umberto Cappellazzo et al., 2023)

{{<citation>}}

Umberto Cappellazzo, Enrico Fini, Muqiao Yang, Daniele Falavigna, Alessio Brutti, Bhiksha Raj. (2023)  
**Continual Contrastive Spoken Language Understanding**  

---
Primary Category: eess.AS  
Categories: cs-AI, eess-AS, eess.AS  
Keywords: Spoken Language Understanding  
[Paper Link](http://arxiv.org/abs/2310.02699v1)  

---


**ABSTRACT**  
Recently, neural networks have shown impressive progress across diverse fields, with speech processing being no exception. However, recent breakthroughs in this area require extensive offline training using large datasets and tremendous computing resources. Unfortunately, these models struggle to retain their previously acquired knowledge when learning new tasks continually, and retraining from scratch is almost always impractical. In this paper, we investigate the problem of learning sequence-to-sequence models for spoken language understanding in a class-incremental learning (CIL) setting and we propose COCONUT, a CIL method that relies on the combination of experience replay and contrastive learning. Through a modified version of the standard supervised contrastive loss applied only to the rehearsal samples, COCONUT preserves the learned representations by pulling closer samples from the same class and pushing away the others. Moreover, we leverage a multimodal contrastive loss that helps the model learn more discriminative representations of the new data by aligning audio and text features. We also investigate different contrastive designs to combine the strengths of the contrastive loss with teacher-student architectures used for distillation. Experiments on two established SLU datasets reveal the effectiveness of our proposed approach and significant improvements over the baselines. We also show that COCONUT can be combined with methods that operate on the decoder side of the model, resulting in further metrics improvements.

{{</citation>}}


## eess.SY (1)



### (111/117) Proximal Policy Optimization-Based Reinforcement Learning Approach for DC-DC Boost Converter Control: A Comparative Evaluation Against Traditional Control Techniques (Utsab Saha et al., 2023)

{{<citation>}}

Utsab Saha, Shakib Shahria, A. B. M Harun-Ur Rashid. (2023)  
**Proximal Policy Optimization-Based Reinforcement Learning Approach for DC-DC Boost Converter Control: A Comparative Evaluation Against Traditional Control Techniques**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.02945v1)  

---


**ABSTRACT**  
This article presents a proximal policy optimization (PPO) based reinforcement learning (RL) approach for DC-DC boost converter control, which is compared to traditional control methods. The performance of the PPO algorithm is evaluated using MATLAB Simulink co-simulation, and the results demonstrate that the most efficient approach for achieving short settling time and stability is to combine the PPO algorithm with reinforcement learning based control method. The simulation results indicate that the step response characteristics provided using the control method based on RL with the PPO algorithm outperform traditional control approaches, which can be used to improve DC-DC boost converter control. This research also highlights the inherent capability of the reinforcement learning method to enhance the performance of boost converter control.

{{</citation>}}


## math.HO (1)



### (112/117) Notes on a Path to AI Assistance in Mathematical Reasoning (Alex Kontorovich, 2023)

{{<citation>}}

Alex Kontorovich. (2023)  
**Notes on a Path to AI Assistance in Mathematical Reasoning**  

---
Primary Category: math.HO  
Categories: cs-AI, math-HO, math.HO  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.02896v1)  

---


**ABSTRACT**  
These informal notes are based on the author's lecture at the National Academies of Science, Engineering, and Mathematics workshop on "AI to Assist Mathematical Reasoning" in June 2023. The goal is to think through a path by which we might arrive at AI that is useful for the research mathematician.

{{</citation>}}


## cs.HC (1)



### (113/117) uTalk: Bridging the Gap Between Humans and AI (Hussam Azzuni et al., 2023)

{{<citation>}}

Hussam Azzuni, Sharim Jamal, Abdulmotaleb Elsaddik. (2023)  
**uTalk: Bridging the Gap Between Humans and AI**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT, Language Model, Microsoft  
[Paper Link](http://arxiv.org/abs/2310.02739v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have revolutionized various industries by harnessing their power to improve productivity and facilitate learning across different fields. One intriguing application involves combining LLMs with visual models to create a novel approach to Human-Computer Interaction. The core idea behind this system is to develop an interactive platform that allows the general public to leverage the capabilities of ChatGPT in their daily lives. This is achieved by integrating several technologies such as Whisper, ChatGPT, Microsoft Speech Services, and the state-of-the-art (SOTA) talking head system, SadTalker, resulting in uTalk, an intelligent AI system. Users will be able to converse with this portrait, receiving answers to whatever questions they have in mind. Additionally, they could use uTalk for content generation by providing an input and their image. This system is hosted on Streamlit, where the user will initially be requested to provide an image to serve as their AI assistant. Then, users could choose whether to have a conversation or generate content based on their preferences. Either way, it starts by providing an input, where a set of operations will be done, and the avatar will provide a precise response. The paper discusses how SadTalker is optimized to improve its running time by 27.72% based on 25FPS generated videos. In addition, the system's initial performance, uTalk, improved further by 9.8% after SadTalker was integrated and parallelized with Streamlit.

{{</citation>}}


## cs.SD (2)



### (114/117) Multi-resolution HuBERT: Multi-resolution Speech Self-Supervised Learning with Masked Unit Prediction (Jiatong Shi et al., 2023)

{{<citation>}}

Jiatong Shi, Hirofumi Inaguma, Xutai Ma, Ilia Kulikov, Anna Sun. (2023)  
**Multi-resolution HuBERT: Multi-resolution Speech Self-Supervised Learning with Masked Unit Prediction**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: BERT, Multilingual, Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2310.02720v1)  

---


**ABSTRACT**  
Existing Self-Supervised Learning (SSL) models for speech typically process speech signals at a fixed resolution of 20 milliseconds. This approach overlooks the varying informational content present at different resolutions in speech signals. In contrast, this paper aims to incorporate multi-resolution information into speech self-supervised representation learning. We introduce a SSL model that leverages a hierarchical Transformer architecture, complemented by HuBERT-style masked prediction objectives, to process speech at multiple resolutions. Experimental results indicate that the proposed model not only achieves more efficient inference but also exhibits superior or comparable performance to the original HuBERT model over various tasks. Specifically, significant performance improvements over the original HuBERT have been observed in fine-tuning experiments on the LibriSpeech speech recognition benchmark as well as in evaluations using the Speech Universal PERformance Benchmark (SUPERB) and Multilingual SUPERB (ML-SUPERB).

{{</citation>}}


### (115/117) BA-MoE: Boundary-Aware Mixture-of-Experts Adapter for Code-Switching Speech Recognition (Peikun Chen et al., 2023)

{{<citation>}}

Peikun Chen, Fan Yu, Yuhao Lian, Hongfei Xue, Xucheng Wan, Naijun Zheng, Huan Zhou, Lei Xie. (2023)  
**BA-MoE: Boundary-Aware Mixture-of-Experts Adapter for Code-Switching Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.02629v1)  

---


**ABSTRACT**  
Mixture-of-experts based models, which use language experts to extract language-specific representations effectively, have been well applied in code-switching automatic speech recognition. However, there is still substantial space to improve as similar pronunciation across languages may result in ineffective multi-language modeling and inaccurate language boundary estimation. To eliminate these drawbacks, we propose a cross-layer language adapter and a boundary-aware training method, namely Boundary-Aware Mixture-of-Experts (BA-MoE). Specifically, we introduce language-specific adapters to separate language-specific representations and a unified gating layer to fuse representations within each encoder layer. Second, we compute language adaptation loss of the mean output of each language-specific adapter to improve the adapter module's language-specific representation learning. Besides, we utilize a boundary-aware predictor to learn boundary representations for dealing with language boundary confusion. Our approach achieves significant performance improvement, reducing the mixture error rate by 16.55\% compared to the baseline on the ASRU 2019 Mandarin-English code-switching challenge dataset.

{{</citation>}}


## cs.AI (1)



### (116/117) A ModelOps-based Framework for Intelligent Medical Knowledge Extraction (Hongxin Ding et al., 2023)

{{<citation>}}

Hongxin Ding, Peinie Zou, Zhiyuan Wang, Junfeng Zhao, Yasha Wang, Qiang Zhou. (2023)  
**A ModelOps-based Framework for Intelligent Medical Knowledge Extraction**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.02593v1)  

---


**ABSTRACT**  
Extracting medical knowledge from healthcare texts enhances downstream tasks like medical knowledge graph construction and clinical decision-making. However, the construction and application of knowledge extraction models lack automation, reusability and unified management, leading to inefficiencies for researchers and high barriers for non-AI experts such as doctors, to utilize knowledge extraction. To address these issues, we propose a ModelOps-based intelligent medical knowledge extraction framework that offers a low-code system for model selection, training, evaluation and optimization. Specifically, the framework includes a dataset abstraction mechanism based on multi-layer callback functions, a reusable model training, monitoring and management mechanism. We also propose a model recommendation method based on dataset similarity, which helps users quickly find potentially suitable models for a given dataset. Our framework provides convenience for researchers to develop models and simplifies model access for non-AI experts such as doctors.

{{</citation>}}


## cs.SI (1)



### (117/117) Stand for Something or Fall for Everything: Predict Misinformation Spread with Stance-Aware Graph Neural Networks (Zihan Chen et al., 2023)

{{<citation>}}

Zihan Chen, Jingyi Sun, Rong Liu, Feng Mai. (2023)  
**Stand for Something or Fall for Everything: Predict Misinformation Spread with Stance-Aware Graph Neural Networks**  

---
Primary Category: cs.SI  
Categories: H-0; J-4; I-2-7, cs-AI, cs-CY, cs-LG, cs-SI, cs.SI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.02568v1)  

---


**ABSTRACT**  
Although pervasive spread of misinformation on social media platforms has become a pressing challenge, existing platform interventions have shown limited success in curbing its dissemination. In this study, we propose a stance-aware graph neural network (stance-aware GNN) that leverages users' stances to proactively predict misinformation spread. As different user stances can form unique echo chambers, we customize four information passing paths in stance-aware GNN, while the trainable attention weights provide explainability by highlighting each structure's importance. Evaluated on a real-world dataset, stance-aware GNN outperforms benchmarks by 32.65% and exceeds advanced GNNs without user stance by over 4.69%. Furthermore, the attention weights indicate that users' opposition stances have a higher impact on their neighbors' behaviors than supportive ones, which function as social correction to halt misinformation propagation. Overall, our study provides an effective predictive model for platforms to combat misinformation, and highlights the impact of user stances in the misinformation propagation.

{{</citation>}}
