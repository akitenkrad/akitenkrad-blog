---
draft: false
title: "arXiv @ 2023.12.17"
date: 2023-12-17
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.17"
    identifier: arxiv_20231217
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.MA (2)](#csma-2)
- [cs.CL (25)](#cscl-25)
- [cs.CV (41)](#cscv-41)
- [cs.AI (14)](#csai-14)
- [cs.CY (4)](#cscy-4)
- [math.OC (2)](#mathoc-2)
- [cs.HC (5)](#cshc-5)
- [cs.MM (2)](#csmm-2)
- [cs.LG (26)](#cslg-26)
- [eess.SP (2)](#eesssp-2)
- [cs.DC (2)](#csdc-2)
- [cs.PL (1)](#cspl-1)
- [cs.IR (5)](#csir-5)
- [cs.RO (3)](#csro-3)
- [eess.IV (3)](#eessiv-3)
- [cs.CR (6)](#cscr-6)
- [cs.LO (1)](#cslo-1)
- [eess.AS (4)](#eessas-4)
- [cs.SD (2)](#cssd-2)
- [cs.SE (3)](#csse-3)

## cs.MA (2)



### (1/153) Multi-agent Reinforcement Learning: A Comprehensive Survey (Dom Huh et al., 2023)

{{<citation>}}

Dom Huh, Prasant Mohapatra. (2023)  
**Multi-agent Reinforcement Learning: A Comprehensive Survey**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-LG, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10256v1)  

---


**ABSTRACT**  
The prevalence of multi-agent applications pervades various interconnected systems in our everyday lives. Despite their ubiquity, the integration and development of intelligent decision-making agents in a shared environment pose challenges to their effective implementation. This survey delves into the domain of multi-agent systems (MAS), placing a specific emphasis on unraveling the intricacies of learning optimal control within the MAS framework, commonly known as multi-agent reinforcement learning (MARL). The objective of this survey is to provide comprehensive insights into various dimensions of MAS, shedding light on myriad opportunities while highlighting the inherent challenges that accompany multi-agent applications. We hope not only to contribute to a deeper understanding of the MAS landscape but also to provide valuable perspectives for both researchers and practitioners. By doing so, we aim to facilitate informed exploration and foster development within the dynamic realm of MAS, recognizing the need for adaptive strategies and continuous evolution in addressing emerging complexities in MARL.

{{</citation>}}


### (2/153) Communication-Efficient Soft Actor-Critic Policy Collaboration via Regulated Segment Mixture in Internet of Vehicles (Xiaoxue Yu et al., 2023)

{{<citation>}}

Xiaoxue Yu, Rongpeng Li, Chengchao Liang, Zhifeng Zhao. (2023)  
**Communication-Efficient Soft Actor-Critic Policy Collaboration via Regulated Segment Mixture in Internet of Vehicles**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10123v1)  

---


**ABSTRACT**  
Multi-Agent Reinforcement Learning (MARL) has emerged as a foundational approach for addressing diverse, intelligent control tasks, notably in autonomous driving within the Internet of Vehicles (IoV) domain. However, the widely assumed existence of a central node for centralized, federated learning-assisted MARL might be impractical in highly dynamic environments. This can lead to excessive communication overhead, potentially overwhelming the IoV system. To address these challenges, we design a novel communication-efficient and policy collaboration algorithm for MARL under the frameworks of Soft Actor-Critic (SAC) and Decentralized Federated Learning (DFL), named RSM-MASAC, within a fully distributed architecture. In particular, RSM-MASAC enhances multi-agent collaboration and prioritizes higher communication efficiency in dynamic IoV system by incorporating the concept of segmented aggregation in DFL and augmenting multiple model replicas from received neighboring policy segments, which are subsequently employed as reconstructed referential policies for mixing. Distinctively diverging from traditional RL approaches, with derived new bounds under Maximum Entropy Reinforcement Learning (MERL), RSM-MASAC adopts a theory-guided mixture metric to regulate the selection of contributive referential policies to guarantee the soft policy improvement during communication phase. Finally, the extensive simulations in mixed-autonomy traffic control scenarios verify the effectiveness and superiority of our algorithm.

{{</citation>}}


## cs.CL (25)



### (3/153) Catwalk: A Unified Language Model Evaluation Framework for Many Datasets (Dirk Groeneveld et al., 2023)

{{<citation>}}

Dirk Groeneveld, Anas Awadalla, Iz Beltagy, Akshita Bhagia, Ian Magnusson, Hao Peng, Oyvind Tafjord, Pete Walsh, Kyle Richardson, Jesse Dodge. (2023)  
**Catwalk: A Unified Language Model Evaluation Framework for Many Datasets**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.10253v1)  

---


**ABSTRACT**  
The success of large language models has shifted the evaluation paradigms in natural language processing (NLP). The community's interest has drifted towards comparing NLP models across many tasks, domains, and datasets, often at an extreme scale. This imposes new engineering challenges: efforts in constructing datasets and models have been fragmented, and their formats and interfaces are incompatible. As a result, it often takes extensive (re)implementation efforts to make fair and controlled comparisons at scale.   Catwalk aims to address these issues. Catwalk provides a unified interface to a broad range of existing NLP datasets and models, ranging from both canonical supervised training and fine-tuning, to more modern paradigms like in-context learning. Its carefully-designed abstractions allow for easy extensions to many others. Catwalk substantially lowers the barriers to conducting controlled experiments at scale. For example, we finetuned and evaluated over 64 models on over 86 datasets with a single command, without writing any code. Maintained by the AllenNLP team at the Allen Institute for Artificial Intelligence (AI2), Catwalk is an ongoing open-source effort: https://github.com/allenai/catwalk.

{{</citation>}}


### (4/153) Low-resource classification of mobility functioning information in clinical sentences using large language models (Tuan Dung Le et al., 2023)

{{<citation>}}

Tuan Dung Le, Thanh Duong, Thanh Thieu. (2023)  
**Low-resource classification of mobility functioning information in clinical sentences using large language models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, T5  
[Paper Link](http://arxiv.org/abs/2312.10202v1)  

---


**ABSTRACT**  
Objective: Function is increasingly recognized as an important indicator of whole-person health. This study evaluates the ability of publicly available large language models (LLMs) to accurately identify the presence of functioning information from clinical notes. We explore various strategies to improve the performance on this task. Materials and Methods: We collect a balanced binary classification dataset of 1000 sentences from the Mobility NER dataset, which was curated from n2c2 clinical notes. For evaluation, we construct zero-shot and few-shot prompts to query the LLMs whether a given sentence contains mobility functioning information. Two sampling techniques, random sampling and k-nearest neighbor (kNN)-based sampling, are used to select the few-shot examples. Furthermore, we apply a parameter-efficient prompt-based fine-tuning method to the LLMs and evaluate their performance under various training settings. Results: Flan-T5-xxl outperforms all other models in both zero-shot and few-shot settings, achieving a F1 score of 0.865 with a single demonstrative example selected by kNN sampling. In prompt-based fine-tuning experiments, this foundation model also demonstrates superior performance across all low-resource settings, particularly achieving an impressive F1 score of 0.922 using the full training dataset. The smaller model, Flan-T5-xl, requires fine-tuning with only 2.3M additional parameters to achieve comparable performance to the fully fine-tuned Gatortron-base model, both surpassing 0.9 F1 score. Conclusion: Open-source instruction-tuned LLMs demonstrate impressive in-context learning capability in the mobility functioning classification task. The performance of these models can be further improved by continuing fine-tuning on a task-specific dataset.

{{</citation>}}


### (5/153) Pipeline and Dataset Generation for Automated Fact-checking in Almost Any Language (Jan Drchal et al., 2023)

{{<citation>}}

Jan Drchal, Herbert Ullrich, Tomáš Mlynář, Václav Moravec. (2023)  
**Pipeline and Dataset Generation for Automated Fact-checking in Almost Any Language**  

---
Primary Category: cs.CL  
Categories: I-2-7; I-5-4, cs-CL, cs.CL  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.10171v1)  

---


**ABSTRACT**  
This article presents a pipeline for automated fact-checking leveraging publicly available Language Models and data. The objective is to assess the accuracy of textual claims using evidence from a ground-truth evidence corpus. The pipeline consists of two main modules -- the evidence retrieval and the claim veracity evaluation. Our primary focus is on the ease of deployment in various languages that remain unexplored in the field of automated fact-checking. Unlike most similar pipelines, which work with evidence sentences, our pipeline processes data on a paragraph level, simplifying the overall architecture and data requirements. Given the high cost of annotating language-specific fact-checking training data, our solution builds on the Question Answering for Claim Generation (QACG) method, which we adapt and use to generate the data for all models of the pipeline. Our strategy enables the introduction of new languages through machine translation of only two fixed datasets of moderate size. Subsequently, any number of training samples can be generated based on an evidence corpus in the target language. We provide open access to all data and fine-tuned models for Czech, English, Polish, and Slovak pipelines, as well as to our codebase that may be used to reproduce the results.We comprehensively evaluate the pipelines for all four languages, including human annotations and per-sample difficulty assessment using Pointwise V-information. The presented experiments are based on full Wikipedia snapshots to promote reproducibility. To facilitate implementation and user interaction, we develop the FactSearch application featuring the proposed pipeline and the preliminary feedback on its performance.

{{</citation>}}


### (6/153) Do LVLMs Understand Charts? Analyzing and Correcting Factual Errors in Chart Captioning (Kung-Hsiang Huang et al., 2023)

{{<citation>}}

Kung-Hsiang Huang, Mingyang Zhou, Hou Pong Chan, Yi R. Fung, Zhenhailong Wang, Lingyu Zhang, Shih-Fu Chang, Heng Ji. (2023)  
**Do LVLMs Understand Charts? Analyzing and Correcting Factual Errors in Chart Captioning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.10160v1)  

---


**ABSTRACT**  
Recent advancements in large vision-language models (LVLMs) have led to significant progress in generating natural language descriptions for visual content and thus enhancing various applications. One issue with these powerful models is that they sometimes produce texts that are factually inconsistent with the visual input. While there has been some effort to mitigate such inconsistencies in natural image captioning, the factuality of generated captions for structured document images, such as charts, has not received as much scrutiny, posing a potential threat to information reliability in critical applications. This work delves into the factuality aspect by introducing a comprehensive typology of factual errors in generated chart captions. A large-scale human annotation effort provides insight into the error patterns and frequencies in captions crafted by various chart captioning models, ultimately forming the foundation of a novel dataset, CHOCOLATE. Our analysis reveals that even state-of-the-art models, including GPT-4V, frequently produce captions laced with factual inaccuracies. In response to this challenge, we establish the new task of Chart Caption Factual Error Correction and introduce CHARTVE, a model for visual entailment that outperforms proprietary and open-source LVLMs in evaluating factual consistency. Furthermore, we propose C2TFEC, an interpretable two-stage framework that excels at correcting factual errors. This work inaugurates a new domain in factual error correction for chart captions, presenting a novel evaluation mechanism, and demonstrating an effective approach to ensuring the factuality of generated chart captions.

{{</citation>}}


### (7/153) Faithful Persona-based Conversational Dataset Generation with Large Language Models (Pegah Jandaghi et al., 2023)

{{<citation>}}

Pegah Jandaghi, XiangHai Sheng, Xinyi Bai, Jay Pujara, Hakim Sidahmed. (2023)  
**Faithful Persona-based Conversational Dataset Generation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.10007v1)  

---


**ABSTRACT**  
High-quality conversational datasets are essential for developing AI models that can communicate with users. One way to foster deeper interactions between a chatbot and its user is through personas, aspects of the user's character that provide insights into their personality, motivations, and behaviors. Training Natural Language Processing (NLP) models on a diverse and comprehensive persona-based dataset can lead to conversational models that create a deeper connection with the user, and maintain their engagement. In this paper, we leverage the power of Large Language Models (LLMs) to create a large, high-quality conversational dataset from a seed dataset. We propose a Generator-Critic architecture framework to expand the initial dataset, while improving the quality of its conversations. The Generator is an LLM prompted to output conversations. The Critic consists of a mixture of expert LLMs that control the quality of the generated conversations. These experts select the best generated conversations, which we then use to improve the Generator. We release Synthetic-Persona-Chat, consisting of 20k conversations seeded from Persona-Chat. We evaluate the quality of Synthetic-Persona-Chat and our generation framework on different dimensions through extensive experiments, and observe that the losing rate of Synthetic-Persona-Chat against Persona-Chat during Turing test decreases from 17.2% to 8.8% over three iterations.

{{</citation>}}


### (8/153) ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent (Renat Aksitov et al., 2023)

{{<citation>}}

Renat Aksitov, Sobhan Miryoosefi, Zonglin Li, Daliang Li, Sheila Babayan, Kavya Kopparapu, Zachary Fisher, Ruiqi Guo, Sushant Prakash, Pranesh Srinivasan, Manzil Zaheer, Felix Yu, Sanjiv Kumar. (2023)  
**ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.10003v1)  

---


**ABSTRACT**  
Answering complex natural language questions often necessitates multi-step reasoning and integrating external information. Several systems have combined knowledge retrieval with a large language model (LLM) to answer such questions. These systems, however, suffer from various failure cases, and we cannot directly train them end-to-end to fix such failures, as interaction with external knowledge is non-differentiable. To address these deficiencies, we define a ReAct-style LLM agent with the ability to reason and act upon external knowledge. We further refine the agent through a ReST-like method that iteratively trains on previous trajectories, employing growing-batch reinforcement learning with AI feedback for continuous self-improvement and self-distillation. Starting from a prompted large model and after just two iterations of the algorithm, we can produce a fine-tuned small model that achieves comparable performance on challenging compositional question-answering benchmarks with two orders of magnitude fewer parameters.

{{</citation>}}


### (9/153) LLaMAntino: LLaMA 2 Models for Effective Text Generation in Italian Language (Pierpaolo Basile et al., 2023)

{{<citation>}}

Pierpaolo Basile, Elio Musacchio, Marco Polignano, Lucia Siciliani, Giuseppe Fiameni, Giovanni Semeraro. (2023)  
**LLaMAntino: LLaMA 2 Models for Effective Text Generation in Italian Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, LLaMA, Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2312.09993v1)  

---


**ABSTRACT**  
Large Language Models represent state-of-the-art linguistic models designed to equip computers with the ability to comprehend natural language. With its exceptional capacity to capture complex contextual relationships, the LLaMA (Large Language Model Meta AI) family represents a novel advancement in the field of natural language processing by releasing foundational models designed to improve the natural language understanding abilities of the transformer architecture thanks to their large amount of trainable parameters (7, 13, and 70 billion parameters). In many natural language understanding tasks, these models obtain the same performances as private company models such as OpenAI Chat-GPT with the advantage to make publicly available weights and code for research and commercial uses. In this work, we investigate the possibility of Language Adaptation for LLaMA models, explicitly focusing on addressing the challenge of Italian Language coverage. Adopting an open science approach, we explore various tuning approaches to ensure a high-quality text generated in Italian suitable for common tasks in this underrepresented language in the original models' datasets. We aim to release effective text generation models with strong linguistic properties for many tasks that seem challenging using multilingual or general-purpose LLMs. By leveraging an open science philosophy, this study contributes to Language Adaptation strategies for the Italian language by introducing the novel LLaMAntino family of Italian LLMs.

{{</citation>}}


### (10/153) LoRAMoE: Revolutionizing Mixture of Experts for Maintaining World Knowledge in Language Model Alignment (Shihan Dou et al., 2023)

{{<citation>}}

Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Jun Zhao, Wei Shen, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Xiaoran Fan, Shiliang Pu, Jiang Zhu, Rui Zheng, Tao Gui, Qi Zhang, Xuanjing Huang. (2023)  
**LoRAMoE: Revolutionizing Mixture of Experts for Maintaining World Knowledge in Language Model Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09979v2)  

---


**ABSTRACT**  
Supervised fine-tuning (SFT) is a crucial step for large language models (LLMs), enabling them to align with human instructions and enhance their capabilities in downstream tasks. When the models are required to align with a broader range of downstream tasks, or there is a desire to notably improve the performance on a specific task, a substantial increase in fine-tuning data often emerges as the solution. However, we find that large-scale increases in instruction data can disrupt the world knowledge previously stored in the LLMs, i.e., world knowledge forgetting. In this paper, we introduce LoRAMoE to address the above challenge. The LoRAMoE is a plugin version of Mixture of Experts (MoE). The plugin form ensures the integrity of world knowledge by freezing the backbone model during the training phase. We then propose the use of localized balancing constraints to coordinate parts of experts for task utilization, meanwhile enabling other experts to fully leverage the world knowledge stored in the models. Experimental results demonstrate that LoRAMoE can reasonably coordinate experts based on data type during inference, and even dramatically increasing instruction data does not result in knowledge forgetting. Moreover, LoRAMoE provides additional benefits for the performance of downstream tasks, indicating the potential of our approach for multi-task learning.

{{</citation>}}


### (11/153) Data and Approaches for German Text simplification -- towards an Accessibility-enhanced Communication (Thorben Schomacker et al., 2023)

{{<citation>}}

Thorben Schomacker, Michael Gille, Jörg von der Hülls, Marina Tropmann-Frick. (2023)  
**Data and Approaches for German Text simplification -- towards an Accessibility-enhanced Communication**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09966v1)  

---


**ABSTRACT**  
This paper examines the current state-of-the-art of German text simplification, focusing on parallel and monolingual German corpora. It reviews neural language models for simplifying German texts and assesses their suitability for legal texts and accessibility requirements. Our findings highlight the need for additional training data and more appropriate approaches that consider the specific linguistic characteristics of German, as well as the importance of the needs and preferences of target groups with cognitive or language impairments. The authors launched the interdisciplinary OPEN-LS project in April 2023 to address these research gaps. The project aims to develop a framework for text formats tailored to individuals with low literacy levels, integrate legal texts, and enhance comprehensibility for those with linguistic or cognitive impairments. It will also explore cost-effective ways to enhance the data with audience-specific illustrations using image-generating AI.   For more and up-to-date information, please visit our project homepage https://open-ls.entavis.com

{{</citation>}}


### (12/153) RDR: the Recap, Deliberate, and Respond Method for Enhanced Language Understanding (Yuxin Zi et al., 2023)

{{<citation>}}

Yuxin Zi, Hariram Veeramani, Kaushik Roy, Amit Sheth. (2023)  
**RDR: the Recap, Deliberate, and Respond Method for Enhanced Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GLUE, NLU  
[Paper Link](http://arxiv.org/abs/2312.09932v1)  

---


**ABSTRACT**  
Natural language understanding (NLU) using neural network pipelines often requires additional context that is not solely present in the input data. Through Prior research, it has been evident that NLU benchmarks are susceptible to manipulation by neural models, wherein these models exploit statistical artifacts within the encoded external knowledge to artificially inflate performance metrics for downstream tasks. Our proposed approach, known as the Recap, Deliberate, and Respond (RDR) paradigm, addresses this issue by incorporating three distinct objectives within the neural network pipeline. Firstly, the Recap objective involves paraphrasing the input text using a paraphrasing model in order to summarize and encapsulate its essence. Secondly, the Deliberation objective entails encoding external graph information related to entities mentioned in the input text, utilizing a graph embedding model. Finally, the Respond objective employs a classification head model that utilizes representations from the Recap and Deliberation modules to generate the final prediction. By cascading these three models and minimizing a combined loss, we mitigate the potential for gaming the benchmark and establish a robust method for capturing the underlying semantic patterns, thus enabling accurate predictions. To evaluate the effectiveness of the RDR method, we conduct tests on multiple GLUE benchmark tasks. Our results demonstrate improved performance compared to competitive baselines, with an enhancement of up to 2\% on standard metrics. Furthermore, we analyze the observed evidence for semantic understanding exhibited by RDR models, emphasizing their ability to avoid gaming the benchmark and instead accurately capture the true underlying semantic patterns.

{{</citation>}}


### (13/153) Red AI? Inconsistent Responses from GPT3.5 Models on Political Issues in the US and China (Di Zhou et al., 2023)

{{<citation>}}

Di Zhou, Yinxian Zhang. (2023)  
**Red AI? Inconsistent Responses from GPT3.5 Models on Political Issues in the US and China**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.09917v1)  

---


**ABSTRACT**  
The rising popularity of ChatGPT and other AI-powered large language models (LLMs) has led to increasing studies highlighting their susceptibility to mistakes and biases. However, most of these studies focus on models trained on English texts. Taking an innovative approach, this study investigates political biases in GPT's multilingual models. We posed the same question about high-profile political issues in the United States and China to GPT in both English and simplified Chinese, and our analysis of the bilingual responses revealed that GPT's bilingual models' political "knowledge" (content) and the political "attitude" (sentiment) are significantly more inconsistent on political issues in China. The simplified Chinese GPT models not only tended to provide pro-China information but also presented the least negative sentiment towards China's problems, whereas the English GPT was significantly more negative towards China. This disparity may stem from Chinese state censorship and US-China geopolitical tensions, which influence the training corpora of GPT bilingual models. Moreover, both Chinese and English models tended to be less critical towards the issues of "their own" represented by the language used, than the issues of "the other." This suggests that GPT multilingual models could potentially develop a "political identity" and an associated sentiment bias based on their training language. We discussed the implications of our findings for information transmission and communication in an increasingly divided world.

{{</citation>}}


### (14/153) Exploring Automatic Text Simplification of German Narrative Documents (Thorben Schomacker et al., 2023)

{{<citation>}}

Thorben Schomacker, Tillmann Dönicke, Marina Tropmann-Frick. (2023)  
**Exploring Automatic Text Simplification of German Narrative Documents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2312.09907v1)  

---


**ABSTRACT**  
In this paper, we apply transformer-based Natural Language Generation (NLG) techniques to the problem of text simplification. Currently, there are only a few German datasets available for text simplification, even fewer with larger and aligned documents, and not a single one with narrative texts. In this paper, we explore to which degree modern NLG techniques can be applied to German narrative text simplifications. We use Longformer attention and a pre-trained mBART model. Our findings indicate that the existing approaches for German are not able to solve the task properly. We conclude on a few directions for future research to address this problem.

{{</citation>}}


### (15/153) Grammatical information in BERT sentence embeddings as two-dimensional arrays (Vivi Nastase et al., 2023)

{{<citation>}}

Vivi Nastase, Paola Merlo. (2023)  
**Grammatical information in BERT sentence embeddings as two-dimensional arrays**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.09890v1)  

---


**ABSTRACT**  
Sentence embeddings induced with various transformer architectures encode much semantic and syntactic information in a distributed manner in a one-dimensional array. We investigate whether specific grammatical information can be accessed in these distributed representations. Using data from a task developed to test rule-like generalizations, our experiments on detecting subject-verb agreement yield several promising results. First, we show that while the usual sentence representations encoded as one-dimensional arrays do not easily support extraction of rule-like regularities, a two-dimensional reshaping of these vectors allows various learning architectures to access such information. Next, we show that various architectures can detect patterns in these two-dimensional reshaped sentence embeddings and successfully learn a model based on smaller amounts of simpler training data, which performs well on more complex test data. This indicates that current sentence embeddings contain information that is regularly distributed, and which can be captured when the embeddings are reshaped into higher dimensional arrays. Our results cast light on representations produced by language models and help move towards developing few-shot learning approaches.

{{</citation>}}


### (16/153) SMILE: Multimodal Dataset for Understanding Laughter in Video with Language Models (Lee Hyun et al., 2023)

{{<citation>}}

Lee Hyun, Kim Sung-Bin, Seungju Han, Youngjae Yu, Tae-Hyun Oh. (2023)  
**SMILE: Multimodal Dataset for Understanding Laughter in Video with Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.09818v1)  

---


**ABSTRACT**  
Despite the recent advances of the artificial intelligence, building social intelligence remains a challenge. Among social signals, laughter is one of the distinctive expressions that occurs during social interactions between humans. In this work, we tackle a new challenge for machines to understand the rationale behind laughter in video, Video Laugh Reasoning. We introduce this new task to explain why people laugh in a particular video and a dataset for this task. Our proposed dataset, SMILE, comprises video clips and language descriptions of why people laugh. We propose a baseline by leveraging the reasoning capacity of large language models (LLMs) with textual video representation. Experiments show that our baseline can generate plausible explanations for laughter. We further investigate the scalability of our baseline by probing other video understanding tasks and in-the-wild videos. We release our dataset, code, and model checkpoints on https://github.com/SMILE-data/SMILE.

{{</citation>}}


### (17/153) ProCoT: Stimulating Critical Thinking and Writing of Students through Engagement with Large Language Models (LLMs) (Tosin Adewumi et al., 2023)

{{<citation>}}

Tosin Adewumi, Lama Alkhaled, Claudia Buck, Sergio Hernandez, Saga Brilioth, Mkpe Kekung, Yelvin Ragimov, Elisa Barney. (2023)  
**ProCoT: Stimulating Critical Thinking and Writing of Students through Engagement with Large Language Models (LLMs)**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.09801v1)  

---


**ABSTRACT**  
We introduce a novel writing method called Probing Chain of Thought (ProCoT), which prevents students from cheating using a Large Language Model (LLM), such as ChatGPT, while enhancing their active learning through such models. LLMs have disrupted education and many other feilds. For fear of students cheating, many educationists have resorted to banning their use, as their outputs can be human-like and hard to detect in some cases. These LLMs are also known for hallucinations (i.e. fake facts). We conduct studies with ProCoT in two different courses with a combined total of about 66 students. The students in each course were asked to prompt an LLM of their choice with one question from a set of four and required to affirm or refute statements in the LLM output by using peer reviewed references. The results show two things: (1) ProCoT stimulates creative/critical thinking and writing of students through engagement with LLMs when we compare the LLM solely output to ProCoT output and (2) ProCoT can prevent cheating because of clear limitations in existing LLMs when we compare students ProCoT output to LLM ProCoT output. We also discover that most students prefer to give answers in fewer words than LLMs, which are typically verbose. The average word counts for students, ChatGPT (v3.5) and Phind (v8) are 208, 391 and 383, respectively.

{{</citation>}}


### (18/153) RJUA-QA: A Comprehensive QA Dataset for Urology (Shiwei Lyu et al., 2023)

{{<citation>}}

Shiwei Lyu, Chenfei Chi, Hongbo Cai, Lei Shi, Xiaoyan Yang, Lei Liu, Xiang Chen, Deng Zhao, Zhiqiang Zhang, Xianguo Lyu, Ming Zhang, Fangzhou Li, Xiaowei Ma, Yue Shen, Jinjie Gu, Wei Xue, Yiran Huang. (2023)  
**RJUA-QA: A Comprehensive QA Dataset for Urology**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.09785v1)  

---


**ABSTRACT**  
We introduce RJUA-QA, a novel medical dataset for question answering (QA) and reasoning with clinical evidence, contributing to bridge the gap between general large language models (LLMs) and medical-specific LLM applications. RJUA-QA is derived from realistic clinical scenarios and aims to facilitate LLMs in generating reliable diagnostic and advice. The dataset contains 2,132 curated Question-Context-Answer pairs, corresponding about 25,000 diagnostic records and clinical cases. The dataset covers 67 common urological disease categories, where the disease coverage exceeds 97.6\% of the population seeking medical services in urology. Each data instance in RJUA-QA comprises: (1) a question mirroring real patient to inquiry about clinical symptoms and medical conditions, (2) a context including comprehensive expert knowledge, serving as a reference for medical examination and diagnosis, (3) a doctor response offering the diagnostic conclusion and suggested examination guidance, (4) a diagnosed clinical disease as the recommended diagnostic outcome, and (5) clinical advice providing recommendations for medical examination. RJUA-QA is the first medical QA dataset for clinical reasoning over the patient inquiries, where expert-level knowledge and experience are required for yielding diagnostic conclusions and medical examination advice. A comprehensive evaluation is conducted to evaluate the performance of both medical-specific and general LLMs on the RJUA-QA dataset.

{{</citation>}}


### (19/153) GSQA: An End-to-End Model for Generative Spoken Question Answering (Min-Han Shih et al., 2023)

{{<citation>}}

Min-Han Shih, Ho-Lam Chung, Yu-Chi Pai, Ming-Hao Hsu, Guan-Ting Lin, Shang-Wen Li, Hung-yi Lee. (2023)  
**GSQA: An End-to-End Model for Generative Spoken Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.09781v1)  

---


**ABSTRACT**  
In recent advancements in spoken question answering (QA), end-to-end models have made significant strides. However, previous research has primarily focused on extractive span selection. While this extractive-based approach is effective when answers are present directly within the input, it falls short in addressing abstractive questions, where answers are not directly extracted but inferred from the given information. To bridge this gap, we introduce the first end-to-end Generative Spoken Question Answering (GSQA) model that empowers the system to engage in abstractive reasoning. The challenge in training our GSQA model lies in the absence of a spoken abstractive QA dataset. We propose using text models for initialization and leveraging the extractive QA dataset to transfer knowledge from the text generative model to the spoken generative model. Experimental results indicate that our model surpasses the previous extractive model by 3% on extractive QA datasets. Furthermore, the GSQA model has only been fine-tuned on the spoken extractive QA dataset. Despite not having seen any spoken abstractive QA data, it can still closely match the performance of the cascade model. In conclusion, our GSQA model shows the potential to generalize to a broad spectrum of questions, thus further expanding spoken question answering capabilities of abstractive QA. Our code is available at \href{https://voidful.github.io/GSQA}{https://voidful.github.io/GSQA}

{{</citation>}}


### (20/153) HEAR: Hearing Enhanced Audio Response for Video-grounded Dialogue (Sunjae Yoon et al., 2023)

{{<citation>}}

Sunjae Yoon, Dahyun Kim, Eunseop Yoon, Hee Suk Yoon, Junyeong Kim, Chnag D. Yoo. (2023)  
**HEAR: Hearing Enhanced Audio Response for Video-grounded Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.09736v1)  

---


**ABSTRACT**  
Video-grounded Dialogue (VGD) aims to answer questions regarding a given multi-modal input comprising video, audio, and dialogue history. Although there have been numerous efforts in developing VGD systems to improve the quality of their responses, existing systems are competent only to incorporate the information in the video and text and tend to struggle in extracting the necessary information from the audio when generating appropriate responses to the question. The VGD system seems to be deaf, and thus, we coin this symptom of current systems' ignoring audio data as a deaf response. To overcome the deaf response problem, Hearing Enhanced Audio Response (HEAR) framework is proposed to perform sensible listening by selectively attending to audio whenever the question requires it. The HEAR framework enhances the accuracy and audibility of VGD systems in a model-agnostic manner. HEAR is validated on VGD datasets (i.e., AVSD@DSTC7 and AVSD@DSTC8) and shows effectiveness with various VGD systems.

{{</citation>}}


### (21/153) Discovering Highly Influential Shortcut Reasoning: An Automated Template-Free Approach (Daichi Haraguchi et al., 2023)

{{<citation>}}

Daichi Haraguchi, Kiyoaki Shirai, Naoya Inoue, Natthawut Kertkeidkachorn. (2023)  
**Discovering Highly Influential Shortcut Reasoning: An Automated Template-Free Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Inference, Reasoning, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2312.09718v1)  

---


**ABSTRACT**  
Shortcut reasoning is an irrational process of inference, which degrades the robustness of an NLP model. While a number of previous work has tackled the identification of shortcut reasoning, there are still two major limitations: (i) a method for quantifying the severity of the discovered shortcut reasoning is not provided; (ii) certain types of shortcut reasoning may be missed. To address these issues, we propose a novel method for identifying shortcut reasoning. The proposed method quantifies the severity of the shortcut reasoning by leveraging out-of-distribution data and does not make any assumptions about the type of tokens triggering the shortcut reasoning. Our experiments on Natural Language Inference and Sentiment Analysis demonstrate that our framework successfully discovers known and unknown shortcut reasoning in the previous work.

{{</citation>}}


### (22/153) Probing Pretrained Language Models with Hierarchy Properties (Jesús Lovón-Melgarejo et al., 2023)

{{<citation>}}

Jesús Lovón-Melgarejo, Jose G. Moreno, Romaric Besançon, Olivier Ferret, Lynda Tamine. (2023)  
**Probing Pretrained Language Models with Hierarchy Properties**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Information Retrieval, Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2312.09670v1)  

---


**ABSTRACT**  
Since Pretrained Language Models (PLMs) are the cornerstone of the most recent Information Retrieval (IR) models, the way they encode semantic knowledge is particularly important. However, little attention has been given to studying the PLMs' capability to capture hierarchical semantic knowledge. Traditionally, evaluating such knowledge encoded in PLMs relies on their performance on a task-dependent evaluation approach based on proxy tasks, such as hypernymy detection. Unfortunately, this approach potentially ignores other implicit and complex taxonomic relations. In this work, we propose a task-agnostic evaluation method able to evaluate to what extent PLMs can capture complex taxonomy relations, such as ancestors and siblings. The evaluation is based on intrinsic properties that capture the hierarchical nature of taxonomies. Our experimental evaluation shows that the lexico-semantic knowledge implicitly encoded in PLMs does not always capture hierarchical relations. We further demonstrate that the proposed properties can be injected into PLMs to improve their understanding of hierarchy. Through evaluations on taxonomy reconstruction, hypernym discovery and reading comprehension tasks, we show that the knowledge about hierarchy is moderately but not systematically transferable across tasks.

{{</citation>}}


### (23/153) Leveraging Language ID to Calculate Intermediate CTC Loss for Enhanced Code-Switching Speech Recognition (Tzu-Ting Yang et al., 2023)

{{<citation>}}

Tzu-Ting Yang, Hsin-Wei Wang, Berlin Chen. (2023)  
**Leveraging Language ID to Calculate Intermediate CTC Loss for Enhanced Code-Switching Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.09583v1)  

---


**ABSTRACT**  
In recent years, end-to-end speech recognition has emerged as a technology that integrates the acoustic, pronunciation dictionary, and language model components of the traditional Automatic Speech Recognition model. It is possible to achieve human-like recognition without the need to build a pronunciation dictionary in advance. However, due to the relative scarcity of training data on code-switching, the performance of ASR models tends to degrade drastically when encountering this phenomenon. Most past studies have simplified the learning complexity of the model by splitting the code-switching task into multiple tasks dealing with a single language and then learning the domain-specific knowledge of each language separately. Therefore, in this paper, we attempt to introduce language identification information into the middle layer of the ASR model's encoder. We aim to generate acoustic features that imply language distinctions in a more implicit way, reducing the model's confusion when dealing with language switching.

{{</citation>}}


### (24/153) Extending Context Window of Large Language Models via Semantic Compression (Weizhi Fei et al., 2023)

{{<citation>}}

Weizhi Fei, Xueyan Niu, Pingyi Zhou, Lu Hou, Bo Bai, Lei Deng, Wei Han. (2023)  
**Extending Context Window of Large Language Models via Semantic Compression**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IT, cs.CL, math-IT  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.09571v1)  

---


**ABSTRACT**  
Transformer-based Large Language Models (LLMs) often impose limitations on the length of the text input to ensure the generation of fluent and relevant responses. This constraint restricts their applicability in scenarios involving long texts. We propose a novel semantic compression method that enables generalization to texts that are 6-8 times longer, without incurring significant computational costs or requiring fine-tuning. Our proposed framework draws inspiration from source coding in information theory and employs a pre-trained model to reduce the semantic redundancy of long inputs before passing them to the LLMs for downstream tasks. Experimental results demonstrate that our method effectively extends the context window of LLMs across a range of tasks including question answering, summarization, few-shot learning, and information retrieval. Furthermore, the proposed semantic compression method exhibits consistent fluency in text generation while reducing the associated computational overhead.

{{</citation>}}


### (25/153) GPT-4 Surpassing Human Performance in Linguistic Pragmatics (Ljubisa Bojic et al., 2023)

{{<citation>}}

Ljubisa Bojic, Predrag Kovacevic, Milan Cabarkapa. (2023)  
**GPT-4 Surpassing Human Performance in Linguistic Pragmatics**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.09545v1)  

---


**ABSTRACT**  
As Large Language Models (LLMs) become increasingly integrated into everyday life, their capabilities to understand and emulate human cognition are under steady examination. This study investigates the ability of LLMs to comprehend and interpret linguistic pragmatics, an aspect of communication that considers context and implied meanings. Using Grice's communication principles, LLMs and human subjects (N=76) were evaluated based on their responses to various dialogue-based tasks. The findings revealed the superior performance and speed of LLMs, particularly GPT4, over human subjects in interpreting pragmatics. GPT4 also demonstrated accuracy in the pre-testing of human-written samples, indicating its potential in text analysis. In a comparative analysis of LLMs using human individual and average scores, the models exhibited significant chronological improvement. The models were ranked from lowest to highest score, with GPT2 positioned at 78th place, GPT3 ranking at 23rd, Bard at 10th, GPT3.5 placing 5th, Best Human scoring 2nd, and GPT4 achieving the top spot. The findings highlight the remarkable progress made in the development and performance of these LLMs. Future studies should consider diverse subjects, multiple languages, and other cognitive aspects to fully comprehend the capabilities of LLMs. This research holds significant implications for the development and application of AI-based models in communication-centered sectors.

{{</citation>}}


### (26/153) Marathon: A Race Through the Realm of Long Context with Large Language Models (Lei Zhang et al., 2023)

{{<citation>}}

Lei Zhang, Yunshui Li, Ziqiang Liu, Jiaxi yang, Junhao Liu, Min Yang. (2023)  
**Marathon: A Race Through the Realm of Long Context with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09542v1)  

---


**ABSTRACT**  
Although there are currently many benchmarks available for evaluating the long context understanding and reasoning capability of large language models, with the expansion of the context window in these models, the existing long context benchmarks are no longer sufficient for evaluating the long context understanding and reasoning capability of large language models. In this paper, we have developed a fresh long context evaluation benchmark, which we name it Marathon in the form of multiple choice questions, inspired by benchmarks such as MMLU, for assessing the long context comprehension capability of large language models quickly, accurately, and objectively. We have evaluated several of the latest and most popular large language models, as well as three recent and effective long context optimization methods, on our benchmark. This showcases the long context reasoning and comprehension capabilities of these large language models and validates the effectiveness of these optimization methods. Marathon is available at https://huggingface.co/datasets/Lemoncoke/Marathon.

{{</citation>}}


### (27/153) Picking the Underused Heads: A Network Pruning Perspective of Attention Head Selection for Fusing Dialogue Coreference Information (Zhengyuan Liu et al., 2023)

{{<citation>}}

Zhengyuan Liu, Nancy F. Chen. (2023)  
**Picking the Underused Heads: A Network Pruning Perspective of Attention Head Selection for Fusing Dialogue Coreference Information**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Dialog, Dialogue, Pruning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.09541v1)  

---


**ABSTRACT**  
The Transformer-based models with the multi-head self-attention mechanism are widely used in natural language processing, and provide state-of-the-art results. While the pre-trained language backbones are shown to implicitly capture certain linguistic knowledge, explicitly incorporating structure-aware features can bring about further improvement on the downstream tasks. However, such enhancement often requires additional neural components and increases training parameter size. In this work, we investigate the attention head selection and manipulation strategy for feature injection from a network pruning perspective, and conduct a case study on dialogue summarization. We first rank attention heads in a Transformer-based summarizer with layer-wise importance. We then select the underused heads through extensive analysis, and inject structure-aware features by manipulating the selected heads. Experimental results show that the importance-based head selection is effective for feature injection, and dialogue summarization can be improved by incorporating coreference information via head manipulation.

{{</citation>}}


## cs.CV (41)



### (28/153) Advancing Surgical VQA with Scene Graph Knowledge (Kun Yuan et al., 2023)

{{<citation>}}

Kun Yuan, Manasi Kattel, Joel L. Lavanchy, Nassir Navab, Vinkle Srivastava, Nicolas Padoy. (2023)  
**Advancing Surgical VQA with Scene Graph Knowledge**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.10251v1)  

---


**ABSTRACT**  
Modern operating room is becoming increasingly complex, requiring innovative intra-operative support systems. While the focus of surgical data science has largely been on video analysis, integrating surgical computer vision with language capabilities is emerging as a necessity. Our work aims to advance Visual Question Answering (VQA) in the surgical context with scene graph knowledge, addressing two main challenges in the current surgical VQA systems: removing question-condition bias in the surgical VQA dataset and incorporating scene-aware reasoning in the surgical VQA model design. First, we propose a Surgical Scene Graph-based dataset, SSG-QA, generated by employing segmentation and detection models on publicly available datasets. We build surgical scene graphs using spatial and action information of instruments and anatomies. These graphs are fed into a question engine, generating diverse QA pairs. Our SSG-QA dataset provides a more complex, diverse, geometrically grounded, unbiased, and surgical action-oriented dataset compared to existing surgical VQA datasets. We then propose SSG-QA-Net, a novel surgical VQA model incorporating a lightweight Scene-embedded Interaction Module (SIM), which integrates geometric scene knowledge in the VQA model design by employing cross-attention between the textual and the scene features. Our comprehensive analysis of the SSG-QA dataset shows that SSG-QA-Net outperforms existing methods across different question types and complexities. We highlight that the primary limitation in the current surgical VQA systems is the lack of scene knowledge to answer complex queries. We present a novel surgical VQA dataset and model and show that results can be significantly improved by incorporating geometric scene features in the VQA model design. The source code and the dataset will be made publicly available at: https://github.com/CAMMA-public/SSG-QA

{{</citation>}}


### (29/153) Rich Human Feedback for Text-to-Image Generation (Youwei Liang et al., 2023)

{{<citation>}}

Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi Pont-Tuset, Sarah Young, Feng Yang, Junjie Ke, Krishnamurthy Dj Dvijotham, Katie Collins, Yiwen Luo, Yang Li, Kai J Kohlhoff, Deepak Ramachandran, Vidhya Navalpakkam. (2023)  
**Rich Human Feedback for Text-to-Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10240v1)  

---


**ABSTRACT**  
Recent Text-to-Image (T2I) generation models such as Stable Diffusion and Imagen have made significant progress in generating high-resolution images based on text descriptions. However, many generated images still suffer from issues such as artifacts/implausibility, misalignment with text descriptions, and low aesthetic quality. Inspired by the success of Reinforcement Learning with Human Feedback (RLHF) for large language models, prior works collected human-provided scores as feedback on generated images and trained a reward model to improve the T2I generation. In this paper, we enrich the feedback signal by (i) marking image regions that are implausible or misaligned with the text, and (ii) annotating which words in the text prompt are misrepresented or missing on the image. We collect such rich human feedback on 18K generated images and train a multimodal transformer to predict the rich feedback automatically. We show that the predicted rich human feedback can be leveraged to improve image generation, for example, by selecting high-quality training data to finetune and improve the generative models, or by creating masks with predicted heatmaps to inpaint the problematic regions. Notably, the improvements generalize to models (Muse) beyond those used to generate the images on which human feedback data were collected (Stable Diffusion variants).

{{</citation>}}


### (30/153) T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning (Weijie Wei et al., 2023)

{{<citation>}}

Weijie Wei, Fatemeh Karimi Nejadasl, Theo Gevers, Martin R. Oswald. (2023)  
**T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.10217v1)  

---


**ABSTRACT**  
The scarcity of annotated data in outdoor point cloud segmentation poses a significant obstacle in harnessing the modeling capabilities of advanced networks like transformers. Consequently, scholars have been actively investigating efficacious self-supervised pre-training strategies, e.g. contrasting learning and reconstruction-based pretext tasks. Nevertheless, temporal information, which is inherent in the LiDAR point cloud sequence, is consistently disregarded. To better utilize this property, we propose an effective pre-training strategy, namely Temporal Masked AutoEncoders (T-MAE), which takes as input temporally adjacent frames and learns temporal dependency. A SiamWCA backbone, containing a Siamese encoder and a window-based cross-attention (WCA) module, is established for the two-frame input. Taking into account that the motion of an ego-vehicle alters the illumination angles of the same instance, temporal modeling also serves as a robust and natural data augmentation, enhancing the comprehension of target objects. Moreover, instead of utilizing consecutive frames, it is more cost-effective and powerful by using distant historical frames. SiamWCA is a powerful architecture but heavily relies on annotated data. With our T-MAE pre-training strategy, we achieve the best performance on the Waymo dataset among self-supervised learning methods. Comprehensive experiments are conducted to validate all components of our proposal. Upon acceptance, the source code will be made accessible.

{{</citation>}}


### (31/153) Video-based Surgical Skill Assessment using Tree-based Gaussian Process Classifier (Arefeh Rezaei et al., 2023)

{{<citation>}}

Arefeh Rezaei, Mohammad Javad Ahmadi, Amir Molaei, Hamid. D. Taghirad. (2023)  
**Video-based Surgical Skill Assessment using Tree-based Gaussian Process Classifier**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AWS  
[Paper Link](http://arxiv.org/abs/2312.10208v1)  

---


**ABSTRACT**  
assessment using video data and to showcase the effectiveness of the proposed approach in evaluating surgeon proficiency, its potential for targeted training interventions, and quality assurance in surgical departments. The pipeline incorporates a representation flow convolutional neural network and a novel tree-based Gaussian process classifier, which is robust to noise, while being computationally efficient. Additionally, new kernels are introduced to enhance accuracy. The performance of the pipeline is evaluated using the JIGSAWS dataset. Comparative analysis with existing literature reveals significant improvement in accuracy and betterment in computation cost. The proposed pipeline contributes to computational efficiency and accuracy improvement in surgical skill assessment using video data. Results of our study based on comments of our colleague surgeons show that the proposed method has the potential to facilitate skill improvement among surgery fellows and enhance patient safety through targeted training interventions and quality assurance in surgical departments.

{{</citation>}}


### (32/153) Deep Active Perception for Object Detection using Navigation Proposals (Stefanos Ginargiros et al., 2023)

{{<citation>}}

Stefanos Ginargiros, Nikolaos Passalis, Anastasios Tefas. (2023)  
**Deep Active Perception for Object Detection using Navigation Proposals**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.10200v1)  

---


**ABSTRACT**  
Deep Learning (DL) has brought significant advances to robotics vision tasks. However, most existing DL methods have a major shortcoming, they rely on a static inference paradigm inherent in traditional computer vision pipelines. On the other hand, recent studies have found that active perception improves the perception abilities of various models by going beyond these static paradigms. Despite the significant potential of active perception, it poses several challenges, primarily involving significant changes in training pipelines for deep learning models. To overcome these limitations, in this work, we propose a generic supervised active perception pipeline for object detection that can be trained using existing off-the-shelf object detectors, while also leveraging advances in simulation environments. To this end, the proposed method employs an additional neural network architecture that estimates better viewpoints in cases where the object detector confidence is insufficient. The proposed method was evaluated on synthetic datasets, constructed within the Webots robotics simulator, showcasing its effectiveness in two object detection cases.

{{</citation>}}


### (33/153) SoloPose: One-Shot Kinematic 3D Human Pose Estimation with Video Data Augmentation (David C. Jeong et al., 2023)

{{<citation>}}

David C. Jeong, Hongji Liu, Saunder Salazar, Jessie Jiang, Christopher A. Kitts. (2023)  
**SoloPose: One-Shot Kinematic 3D Human Pose Estimation with Video Data Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Augmentation  
[Paper Link](http://arxiv.org/abs/2312.10195v1)  

---


**ABSTRACT**  
While recent two-stage many-to-one deep learning models have demonstrated great success in 3D human pose estimation, such models are inefficient ways to detect 3D key points in a sequential video relative to one-shot and many-to-many models. Another key drawback of two-stage and many-to-one models is that errors in the first stage will be passed onto the second stage. In this paper, we introduce SoloPose, a novel one-shot, many-to-many spatio-temporal transformer model for kinematic 3D human pose estimation of video. SoloPose is further fortified by HeatPose, a 3D heatmap based on Gaussian Mixture Model distributions that factors target key points as well as kinematically adjacent key points. Finally, we address data diversity constraints with the 3D AugMotion Toolkit, a methodology to augment existing 3D human pose datasets, specifically by projecting four top public 3D human pose datasets (Humans3.6M, MADS, AIST Dance++, MPI INF 3DHP) into a novel dataset (Humans7.1M) with a universal coordinate system. Extensive experiments are conducted on Human3.6M as well as the augmented Humans7.1M dataset, and SoloPose demonstrates superior results relative to the state-of-the-art approaches.

{{</citation>}}


### (34/153) UniAR: Unifying Human Attention and Response Prediction on Visual Content (Peizhao Li et al., 2023)

{{<citation>}}

Peizhao Li, Junfeng He, Gang Li, Rachit Bhargava, Shaolei Shen, Nachiappan Valliappan, Youwei Liang, Hongxiang Gu, Venky Ramachandran, Golnaz Farhadi, Yang Li, Kai J Kohlhoff, Vidhya Navalpakkam. (2023)  
**UniAR: Unifying Human Attention and Response Prediction on Visual Content**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.10175v1)  

---


**ABSTRACT**  
Progress in human behavior modeling involves understanding both implicit, early-stage perceptual behavior such as human attention and explicit, later-stage behavior such as subjective ratings/preferences. Yet, most prior research has focused on modeling implicit and explicit human behavior in isolation. Can we build a unified model of human attention and preference behavior that reliably works across diverse types of visual content? Such a model would enable predicting subjective feedback such as overall satisfaction or aesthetic quality ratings, along with the underlying human attention or interaction heatmaps and viewing order, enabling designers and content-creation models to optimize their creation for human-centric improvements. In this paper, we propose UniAR -- a unified model that predicts both implicit and explicit human behavior across different types of visual content. UniAR leverages a multimodal transformer, featuring distinct prediction heads for each facet, and predicts attention heatmap, scanpath or viewing order, and subjective rating/preference. We train UniAR on diverse public datasets spanning natural images, web pages and graphic designs, and achieve leading performance on multiple benchmarks across different image domains and various behavior modeling tasks. Potential applications include providing instant feedback on the effectiveness of UIs/digital designs/images, and serving as a reward model to further optimize design/image creation.

{{</citation>}}


### (35/153) Point Transformer V3: Simpler, Faster, Stronger (Xiaoyang Wu et al., 2023)

{{<citation>}}

Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao. (2023)  
**Point Transformer V3: Simpler, Faster, Stronger**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.10035v1)  

---


**ABSTRACT**  
This paper is not motivated to seek innovation within the attention mechanism. Instead, it focuses on overcoming the existing trade-offs between accuracy and efficiency within the context of point cloud processing, leveraging the power of scale. Drawing inspiration from recent advances in 3D large-scale representation learning, we recognize that model performance is more influenced by scale than by intricate design. Therefore, we present Point Transformer V3 (PTv3), which prioritizes simplicity and efficiency over the accuracy of certain mechanisms that are minor to the overall performance after scaling, such as replacing the precise neighbor search by KNN with an efficient serialized neighbor mapping of point clouds organized with specific patterns. This principle enables significant scaling, expanding the receptive field from 16 to 1024 points while remaining efficient (a 3x increase in processing speed and a 10x improvement in memory efficiency compared with its predecessor, PTv2). PTv3 attains state-of-the-art results on over 20 downstream tasks that span both indoor and outdoor scenarios. Further enhanced with multi-dataset joint training, PTv3 pushes these results to a higher level.

{{</citation>}}


### (36/153) DHFormer: A Vision Transformer-Based Attention Module for Image Dehazing (Abdul Wasi et al., 2023)

{{<citation>}}

Abdul Wasi, O. Jeba Shiney. (2023)  
**DHFormer: A Vision Transformer-Based Attention Module for Image Dehazing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2312.09955v1)  

---


**ABSTRACT**  
Images acquired in hazy conditions have degradations induced in them. Dehazing such images is a vexed and ill-posed problem. Scores of prior-based and learning-based approaches have been proposed to mitigate the effect of haze and generate haze-free images. Many conventional methods are constrained by their lack of awareness regarding scene depth and their incapacity to capture long-range dependencies. In this paper, a method that uses residual learning and vision transformers in an attention module is proposed. It essentially comprises two networks: In the first one, the network takes the ratio of a hazy image and the approximated transmission matrix to estimate a residual map. The second network takes this residual image as input and passes it through convolution layers before superposing it on the generated feature maps. It is then passed through global context and depth-aware transformer encoders to obtain channel attention. The attention module then infers the spatial attention map before generating the final haze-free image. Experimental results, including several quantitative metrics, demonstrate the efficiency and scalability of the suggested methodology.

{{</citation>}}


### (37/153) Closing the Gap: Achieving Better Accuracy-Robustness Tradeoffs Against Query-Based Attacks (Pascal Zimmer et al., 2023)

{{<citation>}}

Pascal Zimmer, Sébastien Andreina, Giorgia Azzurra Marson, Ghassan Karame. (2023)  
**Closing the Gap: Achieving Better Accuracy-Robustness Tradeoffs Against Query-Based Attacks**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.10132v1)  

---


**ABSTRACT**  
Although promising, existing defenses against query-based attacks share a common limitation: they offer increased robustness against attacks at the price of a considerable accuracy drop on clean samples. In this work, we show how to efficiently establish, at test-time, a solid tradeoff between robustness and accuracy when mitigating query-based attacks. Given that these attacks necessarily explore low-confidence regions, our insight is that activating dedicated defenses, such as RND (Qin et al., NeuRIPS 2021) and Random Image Transformations (Xie et al., ICLR 2018), only for low-confidence inputs is sufficient to prevent them. Our approach is independent of training and supported by theory. We verify the effectiveness of our approach for various existing defenses by conducting extensive experiments on CIFAR-10, CIFAR-100, and ImageNet. Our results confirm that our proposal can indeed enhance these defenses by providing better tradeoffs between robustness and accuracy when compared to state-of-the-art approaches while being completely training-free.

{{</citation>}}


### (38/153) LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer (Yuxin Cao et al., 2023)

{{<citation>}}

Yuxin Cao, Ziyu Zhao, Xi Xiao, Derui Wang, Minhui Xue, Jin Lu. (2023)  
**LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2312.09935v1)  

---


**ABSTRACT**  
Video recognition systems are vulnerable to adversarial examples. Recent studies show that style transfer-based and patch-based unrestricted perturbations can effectively improve attack efficiency. These attacks, however, face two main challenges: 1) Adding large stylized perturbations to all pixels reduces the naturalness of the video and such perturbations can be easily detected. 2) Patch-based video attacks are not extensible to targeted attacks due to the limited search space of reinforcement learning that has been widely used in video attacks recently. In this paper, we focus on the video black-box setting and propose a novel attack framework named LogoStyleFool by adding a stylized logo to the clean video. We separate the attack into three stages: style reference selection, reinforcement-learning-based logo style transfer, and perturbation optimization. We solve the first challenge by scaling down the perturbation range to a regional logo, while the second challenge is addressed by complementing an optimization stage after reinforcement learning. Experimental results substantiate the overall superiority of LogoStyleFool over three state-of-the-art patch-based attacks in terms of attack performance and semantic preservation. Meanwhile, LogoStyleFool still maintains its performance against two existing patch-based defense methods. We believe that our research is beneficial in increasing the attention of the security community to such subregional style transfer attacks.

{{</citation>}}


### (39/153) CNC-Net: Self-Supervised Learning for CNC Machining Operations (Mohsen Yavartanoo et al., 2023)

{{<citation>}}

Mohsen Yavartanoo, Sangmin Hong, Reyhaneh Neshatavar, Kyoung Mu Lee. (2023)  
**CNC-Net: Self-Supervised Learning for CNC Machining Operations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.09925v1)  

---


**ABSTRACT**  
CNC manufacturing is a process that employs computer numerical control (CNC) machines to govern the movements of various industrial tools and machinery, encompassing equipment ranging from grinders and lathes to mills and CNC routers. However, the reliance on manual CNC programming has become a bottleneck, and the requirement for expert knowledge can result in significant costs. Therefore, we introduce a pioneering approach named CNC-Net, representing the use of deep neural networks (DNNs) to simulate CNC machines and grasp intricate operations when supplied with raw materials. CNC-Net constitutes a self-supervised framework that exclusively takes an input 3D model and subsequently generates the essential operation parameters required by the CNC machine to construct the object. Our method has the potential to transformative automation in manufacturing by offering a cost-effective alternative to the high costs of manual CNC programming while maintaining exceptional precision in 3D object production. Our experiments underscore the effectiveness of our CNC-Net in constructing the desired 3D objects through the utilization of CNC operations. Notably, it excels in preserving finer local details, exhibiting a marked enhancement in precision compared to the state-of-the-art 3D CAD reconstruction approaches.

{{</citation>}}


### (40/153) Information Extraction from Unstructured data using Augmented-AI and Computer Vision (Aditya Parikh, 2023)

{{<citation>}}

Aditya Parikh. (2023)  
**Information Extraction from Unstructured data using Augmented-AI and Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Computer Vision, Information Extraction, NLP, OCR  
[Paper Link](http://arxiv.org/abs/2312.09880v1)  

---


**ABSTRACT**  
Process of information extraction (IE) is often used to extract meaningful information from unstructured and unlabeled data. Conventional methods of data extraction including application of OCR and passing extraction engine, are inefficient on large data and have their limitation. In this paper, a peculiar technique of information extraction is proposed using A2I and computer vision technologies, which also includes NLP.

{{</citation>}}


### (41/153) Part Representation Learning with Teacher-Student Decoder for Occluded Person Re-identification (Shang Gao et al., 2023)

{{<citation>}}

Shang Gao, Chenyang Yu, Pingping Zhang, Huchuan Lu. (2023)  
**Part Representation Learning with Teacher-Student Decoder for Occluded Person Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: Representation Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.09797v1)  

---


**ABSTRACT**  
Occluded person re-identification (ReID) is a very challenging task due to the occlusion disturbance and incomplete target information. Leveraging external cues such as human pose or parsing to locate and align part features has been proven to be very effective in occluded person ReID. Meanwhile, recent Transformer structures have a strong ability of long-range modeling. Considering the above facts, we propose a Teacher-Student Decoder (TSD) framework for occluded person ReID, which utilizes the Transformer decoder with the help of human parsing. More specifically, our proposed TSD consists of a Parsing-aware Teacher Decoder (PTD) and a Standard Student Decoder (SSD). PTD employs human parsing cues to restrict Transformer's attention and imparts this information to SSD through feature distillation. Thereby, SSD can learn from PTD to aggregate information of body parts automatically. Moreover, a mask generator is designed to provide discriminative regions for better ReID. In addition, existing occluded person ReID benchmarks utilize occluded samples as queries, which will amplify the role of alleviating occlusion interference and underestimate the impact of the feature absence issue. Contrastively, we propose a new benchmark with non-occluded queries, serving as a complement to the existing benchmark. Extensive experiments demonstrate that our proposed method is superior and the new benchmark is essential. The source codes are available at https://github.com/hh23333/TSD.

{{</citation>}}


### (42/153) Latent Diffusion Models with Image-Derived Annotations for Enhanced AI-Assisted Cancer Diagnosis in Histopathology (Pedro Osorio et al., 2023)

{{<citation>}}

Pedro Osorio, Guillermo Jimenez-Perez, Javier Montalt-Tordera, Jens Hooge, Guillem Duran-Ballester, Shivam Singh, Moritz Radbruch, Ute Bach, Sabrina Schroeder, Krystyna Siudak, Julia Vienenkoetter, Bettina Lawrenz, Sadegh Mohammadi. (2023)  
**Latent Diffusion Models with Image-Derived Annotations for Enhanced AI-Assisted Cancer Diagnosis in Histopathology**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09792v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) based image analysis has an immense potential to support diagnostic histopathology, including cancer diagnostics. However, developing supervised AI methods requires large-scale annotated datasets. A potentially powerful solution is to augment training data with synthetic data. Latent diffusion models, which can generate high-quality, diverse synthetic images, are promising. However, the most common implementations rely on detailed textual descriptions, which are not generally available in this domain. This work proposes a method that constructs structured textual prompts from automatically extracted image features. We experiment with the PCam dataset, composed of tissue patches only loosely annotated as healthy or cancerous. We show that including image-derived features in the prompt, as opposed to only healthy and cancerous labels, improves the Fr\'echet Inception Distance (FID) from 178.8 to 90.2. We also show that pathologists find it challenging to detect synthetic images, with a median sensitivity/specificity of 0.55/0.55. Finally, we show that synthetic data effectively trains AI models.

{{</citation>}}


### (43/153) Collaborating Foundation models for Domain Generalized Semantic Segmentation (Yasser Benigmim et al., 2023)

{{<citation>}}

Yasser Benigmim, Subhankar Roy, Slim Essid, Vicky Kalogeiton, Stéphane Lathuilière. (2023)  
**Collaborating Foundation models for Domain Generalized Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.09788v1)  

---


**ABSTRACT**  
Domain Generalized Semantic Segmentation (DGSS) deals with training a model on a labeled source domain with the aim of generalizing to unseen domains during inference. Existing DGSS methods typically effectuate robust features by means of Domain Randomization (DR). Such an approach is often limited as it can only account for style diversification and not content. In this work, we take an orthogonal approach to DGSS and propose to use an assembly of CoLlaborative FOUndation models for Domain Generalized Semantic Segmentation (CLOUDS). In detail, CLOUDS is a framework that integrates FMs of various kinds: (i) CLIP backbone for its robust feature representation, (ii) generative models to diversify the content, thereby covering various modes of the possible target distribution, and (iii) Segment Anything Model (SAM) for iteratively refining the predictions of the segmentation model. Extensive experiments show that our CLOUDS excels in adapting from synthetic to real DGSS benchmarks and under varying weather conditions, notably outperforming prior methods by 5.6% and 6.7% on averaged miou, respectively. The code is available at : https://github.com/yasserben/CLOUDS

{{</citation>}}


### (44/153) Attention-Based VR Facial Animation with Visual Mouth Camera Guidance for Immersive Telepresence Avatars (Andre Rochow et al., 2023)

{{<citation>}}

Andre Rochow, Max Schwarz, Sven Behnke. (2023)  
**Attention-Based VR Facial Animation with Visual Mouth Camera Guidance for Immersive Telepresence Avatars**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.09750v1)  

---


**ABSTRACT**  
Facial animation in virtual reality environments is essential for applications that necessitate clear visibility of the user's face and the ability to convey emotional signals. In our scenario, we animate the face of an operator who controls a robotic Avatar system. The use of facial animation is particularly valuable when the perception of interacting with a specific individual, rather than just a robot, is intended. Purely keypoint-driven animation approaches struggle with the complexity of facial movements. We present a hybrid method that uses both keypoints and direct visual guidance from a mouth camera. Our method generalizes to unseen operators and requires only a quick enrolment step with capture of two short videos. Multiple source images are selected with the intention to cover different facial expressions. Given a mouth camera frame from the HMD, we dynamically construct the target keypoints and apply an attention mechanism to determine the importance of each source image. To resolve keypoint ambiguities and animate a broader range of mouth expressions, we propose to inject visual mouth camera information into the latent space. We enable training on large-scale speaking head datasets by simulating the mouth camera input with its perspective differences and facial deformations. Our method outperforms a baseline in quality, capability, and temporal consistency. In addition, we highlight how the facial animation contributed to our victory at the ANA Avatar XPRIZE Finals.

{{</citation>}}


### (45/153) LiteVSR: Efficient Visual Speech Recognition by Learning from Speech Representations of Unlabeled Data (Hendrik Laux et al., 2023)

{{<citation>}}

Hendrik Laux, Emil Mededovic, Ahmed Hallawa, Lukas Martin, Arne Peine, Anke Schmeink. (2023)  
**LiteVSR: Efficient Visual Speech Recognition by Learning from Speech Representations of Unlabeled Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-SD, cs.CV, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.09727v1)  

---


**ABSTRACT**  
This paper proposes a novel, resource-efficient approach to Visual Speech Recognition (VSR) leveraging speech representations produced by any trained Automatic Speech Recognition (ASR) model. Moving away from the resource-intensive trends prevalent in recent literature, our method distills knowledge from a trained Conformer-based ASR model, achieving competitive performance on standard VSR benchmarks with significantly less resource utilization. Using unlabeled audio-visual data only, our baseline model achieves a word error rate (WER) of 47.4% and 54.7% on the LRS2 and LRS3 test benchmarks, respectively. After fine-tuning the model with limited labeled data, the word error rate reduces to 35% (LRS2) and 45.7% (LRS3). Our model can be trained on a single consumer-grade GPU within a few days and is capable of performing real-time end-to-end VSR on dated hardware, suggesting a path towards more accessible and resource-efficient VSR methodologies.

{{</citation>}}


### (46/153) ParsNets: A Parsimonious Orthogonal and Low-Rank Linear Networks for Zero-Shot Learning (Jingcai Guo et al., 2023)

{{<citation>}}

Jingcai Guo, Qihua Zhou, Ruibing Li, Xiaocheng Lu, Ziming Liu, Junyang Chen, Xin Xie, Jie Zhang. (2023)  
**ParsNets: A Parsimonious Orthogonal and Low-Rank Linear Networks for Zero-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.09709v1)  

---


**ABSTRACT**  
This paper provides a novel parsimonious yet efficient design for zero-shot learning (ZSL), dubbed ParsNets, where we are interested in learning a composition of on-device friendly linear networks, each with orthogonality and low-rankness properties, to achieve equivalent or even better performance against existing deep models. Concretely, we first refactor the core module of ZSL, i.e., visual-semantics mapping function, into several base linear networks that correspond to diverse components of the semantic space, where the complex nonlinearity can be collapsed into simple local linearities. Then, to facilitate the generalization of local linearities, we construct a maximal margin geometry on the learned features by enforcing low-rank constraints on intra-class samples and high-rank constraints on inter-class samples, resulting in orthogonal subspaces for different classes and each subspace lies on a compact manifold. To enhance the model's adaptability and counterbalance over/under-fittings in ZSL, a set of sample-wise indicators is employed to select a sparse subset from these base linear networks to form a composite semantic predictor for each sample. Notably, maximal margin geometry can guarantee the diversity of features, and meanwhile, local linearities guarantee efficiency. Thus, our ParsNets can generalize better to unseen classes and can be deployed flexibly on resource-constrained devices. Theoretical explanations and extensive experiments are conducted to verify the effectiveness of the proposed method.

{{</citation>}}


### (47/153) Exploring the Feasibility of Generating Realistic 3D Models of Endangered Species Using DreamGaussian: An Analysis of Elevation Angle's Impact on Model Generation (Selcuk Anil Karatopak et al., 2023)

{{<citation>}}

Selcuk Anil Karatopak, Deniz Sen. (2023)  
**Exploring the Feasibility of Generating Realistic 3D Models of Endangered Species Using DreamGaussian: An Analysis of Elevation Angle's Impact on Model Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09682v1)  

---


**ABSTRACT**  
Many species face the threat of extinction. It's important to study these species and gather information about them as much as possible to preserve biodiversity. Due to the rarity of endangered species, there is a limited amount of data available, making it difficult to apply data requiring generative AI methods to this domain. We aim to study the feasibility of generating consistent and real-like 3D models of endangered animals using limited data. Such a phenomenon leads us to utilize zero-shot stable diffusion models that can generate a 3D model out of a single image of the target species. This paper investigates the intricate relationship between elevation angle and the output quality of 3D model generation, focusing on the innovative approach presented in DreamGaussian. DreamGaussian, a novel framework utilizing Generative Gaussian Splatting along with novel mesh extraction and refinement algorithms, serves as the focal point of our study. We conduct a comprehensive analysis, analyzing the effect of varying elevation angles on DreamGaussian's ability to reconstruct 3D scenes accurately. Through an empirical evaluation, we demonstrate how changes in elevation angle impact the generated images' spatial coherence, structural integrity, and perceptual realism. We observed that giving a correct elevation angle with the input image significantly affects the result of the generated 3D model. We hope this study to be influential for the usability of AI to preserve endangered animals; while the penultimate aim is to obtain a model that can output biologically consistent 3D models via small samples, the qualitative interpretation of an existing state-of-the-art model such as DreamGaussian will be a step forward in our goal.

{{</citation>}}


### (48/153) nuScenes Knowledge Graph -- A comprehensive semantic representation of traffic scenes for trajectory prediction (Leon Mlodzian et al., 2023)

{{<citation>}}

Leon Mlodzian, Zhigang Sun, Hendrik Berkemeyer, Sebastian Monka, Zixu Wang, Stefan Dietze, Lavdim Halilaj, Juergen Luettin. (2023)  
**nuScenes Knowledge Graph -- A comprehensive semantic representation of traffic scenes for trajectory prediction**  

---
Primary Category: cs.CV  
Categories: I-2-4; I-2-6; I-2-10, cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.09676v1)  

---


**ABSTRACT**  
Trajectory prediction in traffic scenes involves accurately forecasting the behaviour of surrounding vehicles. To achieve this objective it is crucial to consider contextual information, including the driving path of vehicles, road topology, lane dividers, and traffic rules. Although studies demonstrated the potential of leveraging heterogeneous context for improving trajectory prediction, state-of-the-art deep learning approaches still rely on a limited subset of this information. This is mainly due to the limited availability of comprehensive representations. This paper presents an approach that utilizes knowledge graphs to model the diverse entities and their semantic connections within traffic scenes. Further, we present nuScenes Knowledge Graph (nSKG), a knowledge graph for the nuScenes dataset, that models explicitly all scene participants and road elements, as well as their semantic and spatial relationships. To facilitate the usage of the nSKG via graph neural networks for trajectory prediction, we provide the data in a format, ready-to-use by the PyG library. All artefacts can be found here: https://github.com/boschresearch/nuScenes_Knowledge_Graph

{{</citation>}}


### (49/153) SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery (Xin Guo et al., 2023)

{{<citation>}}

Xin Guo, Jiangwei Lao, Bo Dang, Yingying Zhang, Lei Yu, Lixiang Ru, Liheng Zhong, Ziyuan Huang, Kang Wu, Dingxiang Hu, Huimei He, Jian Wang, Jingdong Chen, Ming Yang, Yongjun Zhang, Yansheng Li. (2023)  
**SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.10115v1)  

---


**ABSTRACT**  
Prior studies on Remote Sensing Foundation Model (RSFM) reveal immense potential towards a generic model for Earth Observation. Nevertheless, these works primarily focus on a single modality without temporal and geo-context modeling, hampering their capabilities for diverse tasks. In this study, we present SkySense, a generic billion-scale model, pre-trained on a curated multi-modal Remote Sensing Imagery (RSI) dataset with 21.5 million temporal sequences. SkySense incorporates a factorized multi-modal spatiotemporal encoder taking temporal sequences of optical and Synthetic Aperture Radar (SAR) data as input. This encoder is pre-trained by our proposed Multi-Granularity Contrastive Learning to learn representations across different modal and spatial granularities. To further enhance the RSI representations by the geo-context clue, we introduce Geo-Context Prototype Learning to learn region-aware prototypes upon RSI's multi-modal spatiotemporal features. To our best knowledge, SkySense is the largest Multi-Modal RSFM to date, whose modules can be flexibly combined or used individually to accommodate various tasks. It demonstrates remarkable generalization capabilities on a thorough evaluation encompassing 16 datasets over 7 tasks, from single- to multi-modal, static to temporal, and classification to localization. SkySense surpasses 18 recent RSFMs in all test scenarios. Specifically, it outperforms the latest models such as GFM, SatLas and Scale-MAE by a large margin, i.e., 2.76%, 3.67% and 3.61% on average respectively. We will release the pre-trained weights to facilitate future research and Earth Observation applications.

{{</citation>}}


### (50/153) Pixel-Superpixel Contrastive Learning and Pseudo-Label Correction for Hyperspectral Image Clustering (Renxiang Guan et al., 2023)

{{<citation>}}

Renxiang Guan, Zihao Li, Xianju Li, Chang Tang. (2023)  
**Pixel-Superpixel Contrastive Learning and Pseudo-Label Correction for Hyperspectral Image Clustering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.09630v1)  

---


**ABSTRACT**  
Hyperspectral image (HSI) clustering is gaining considerable attention owing to recent methods that overcome the inefficiency and misleading results from the absence of supervised information. Contrastive learning methods excel at existing pixel level and super pixel level HSI clustering tasks. The pixel-level contrastive learning method can effectively improve the ability of the model to capture fine features of HSI but requires a large time overhead. The super pixel-level contrastive learning method utilizes the homogeneity of HSI and reduces computing resources; however, it yields rough classification results. To exploit the strengths of both methods, we present a pixel super pixel contrastive learning and pseudo-label correction (PSCPC) method for the HSI clustering. PSCPC can reasonably capture domain-specific and fine-grained features through super pixels and the comparative learning of a small number of pixels within the super pixels. To improve the clustering performance of super pixels, this paper proposes a pseudo-label correction module that aligns the clustering pseudo-labels of pixels and super-pixels. In addition, pixel-level clustering results are used to supervise super pixel-level clustering, improving the generalization ability of the model. Extensive experiments demonstrate the effectiveness and efficiency of PSCPC.

{{</citation>}}


### (51/153) Focus on Your Instruction: Fine-grained and Multi-instruction Image Editing by Attention Modulation (Qin Guo et al., 2023)

{{<citation>}}

Qin Guo, Tianwei Lin. (2023)  
**Focus on Your Instruction: Fine-grained and Multi-instruction Image Editing by Attention Modulation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.10113v1)  

---


**ABSTRACT**  
Recently, diffusion-based methods, like InstructPix2Pix (IP2P), have achieved effective instruction-based image editing, requiring only natural language instructions from the user. However, these methods often inadvertently alter unintended areas and struggle with multi-instruction editing, resulting in compromised outcomes. To address these issues, we introduce the Focus on Your Instruction (FoI), a method designed to ensure precise and harmonious editing across multiple instructions without extra training or test-time optimization. In the FoI, we primarily emphasize two aspects: (1) precisely extracting regions of interest for each instruction and (2) guiding the denoising process to concentrate within these regions of interest. For the first objective, we identify the implicit grounding capability of IP2P from the cross-attention between instruction and image, then develop an effective mask extraction method. For the second objective, we introduce a cross attention modulation module for rough isolation of target editing regions and unrelated regions. Additionally, we introduce a mask-guided disentangle sampling strategy to further ensure clear region isolation. Experimental results demonstrate that FoI surpasses existing methods in both quantitative and qualitative evaluations, especially excelling in multi-instruction editing task.

{{</citation>}}


### (52/153) TOP-ReID: Multi-spectral Object Re-Identification with Token Permutation (Yuhao Wang et al., 2023)

{{<citation>}}

Yuhao Wang, Xuehu Liu, Pingping Zhang, Hu Lu, Zhengzheng Tu, Huchuan Lu. (2023)  
**TOP-ReID: Multi-spectral Object Re-Identification with Token Permutation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.09612v1)  

---


**ABSTRACT**  
Multi-spectral object Re-identification (ReID) aims to retrieve specific objects by leveraging complementary information from different image spectra. It delivers great advantages over traditional single-spectral ReID in complex visual environment. However, the significant distribution gap among different image spectra poses great challenges for effective multi-spectral feature representations. In addition, most of current Transformer-based ReID methods only utilize the global feature of class tokens to achieve the holistic retrieval, ignoring the local discriminative ones. To address the above issues, we step further to utilize all the tokens of Transformers and propose a cyclic token permutation framework for multi-spectral object ReID, dubbled TOP-ReID. More specifically, we first deploy a multi-stream deep network based on vision Transformers to preserve distinct information from different image spectra. Then, we propose a Token Permutation Module (TPM) for cyclic multi-spectral feature aggregation. It not only facilitates the spatial feature alignment across different image spectra, but also allows the class token of each spectrum to perceive the local details of other spectra. Meanwhile, we propose a Complementary Reconstruction Module (CRM), which introduces dense token-level reconstruction constraints to reduce the distribution gap across different image spectra. With the above modules, our proposed framework can generate more discriminative multi-spectral features for robust object ReID. Extensive experiments on three ReID benchmarks (i.e., RGBNT201, RGBNT100 and MSVR310) verify the effectiveness of our methods. The code is available at https://github.com/924973292/TOP-ReID.

{{</citation>}}


### (53/153) Semantic-Aware Transformation-Invariant RoI Align (Guo-Ye Yang et al., 2023)

{{<citation>}}

Guo-Ye Yang, George Kiyohiro Nakayama, Zi-Kai Xiao, Tai-Jiang Mu, Xiaolei Huang, Shi-Min Hu. (2023)  
**Semantic-Aware Transformation-Invariant RoI Align**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.09609v1)  

---


**ABSTRACT**  
Great progress has been made in learning-based object detection methods in the last decade. Two-stage detectors often have higher detection accuracy than one-stage detectors, due to the use of region of interest (RoI) feature extractors which extract transformation-invariant RoI features for different RoI proposals, making refinement of bounding boxes and prediction of object categories more robust and accurate. However, previous RoI feature extractors can only extract invariant features under limited transformations. In this paper, we propose a novel RoI feature extractor, termed Semantic RoI Align (SRA), which is capable of extracting invariant RoI features under a variety of transformations for two-stage detectors. Specifically, we propose a semantic attention module to adaptively determine different sampling areas by leveraging the global and local semantic relationship within the RoI. We also propose a Dynamic Feature Sampler which dynamically samples features based on the RoI aspect ratio to enhance the efficiency of SRA, and a new position embedding, \ie Area Embedding, to provide more accurate position information for SRA through an improved sampling area representation. Experiments show that our model significantly outperforms baseline models with slight computational overhead. In addition, it shows excellent generalization ability and can be used to improve performance with various state-of-the-art backbones and detection methods.

{{</citation>}}


### (54/153) CLAF: Contrastive Learning with Augmented Features for Imbalanced Semi-Supervised Learning (Bowen Tao et al., 2023)

{{<citation>}}

Bowen Tao, Lan Li, Xin-Chun Li, De-Chuan Zhan. (2023)  
**CLAF: Contrastive Learning with Augmented Features for Imbalanced Semi-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.09598v1)  

---


**ABSTRACT**  
Due to the advantages of leveraging unlabeled data and learning meaningful representations, semi-supervised learning and contrastive learning have been progressively combined to achieve better performances in popular applications with few labeled data and abundant unlabeled data. One common manner is assigning pseudo-labels to unlabeled samples and selecting positive and negative samples from pseudo-labeled samples to apply contrastive learning. However, the real-world data may be imbalanced, causing pseudo-labels to be biased toward the majority classes and further undermining the effectiveness of contrastive learning. To address the challenge, we propose Contrastive Learning with Augmented Features (CLAF). We design a class-dependent feature augmentation module to alleviate the scarcity of minority class samples in contrastive learning. For each pseudo-labeled sample, we select positive and negative samples from labeled data instead of unlabeled data to compute contrastive loss. Comprehensive experiments on imbalanced image classification datasets demonstrate the effectiveness of CLAF in the context of imbalanced semi-supervised learning.

{{</citation>}}


### (55/153) Multiscale Vision Transformer With Deep Clustering-Guided Refinement for Weakly Supervised Object Localization (David Kim et al., 2023)

{{<citation>}}

David Kim, Sinhae Cha, Byeongkeun Kang. (2023)  
**Multiscale Vision Transformer With Deep Clustering-Guided Refinement for Weakly Supervised Object Localization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.09584v1)  

---


**ABSTRACT**  
This work addresses the task of weakly-supervised object localization. The goal is to learn object localization using only image-level class labels, which are much easier to obtain compared to bounding box annotations. This task is important because it reduces the need for labor-intensive ground-truth annotations. However, methods for object localization trained using weak supervision often suffer from limited accuracy in localization. To address this challenge and enhance localization accuracy, we propose a multiscale object localization transformer (MOLT). It comprises multiple object localization transformers that extract patch embeddings across various scales. Moreover, we introduce a deep clustering-guided refinement method that further enhances localization accuracy by utilizing separately extracted image segments. These segments are obtained by clustering pixels using convolutional neural networks. Finally, we demonstrate the effectiveness of our proposed method by conducting experiments on the publicly available ILSVRC-2012 dataset.

{{</citation>}}


### (56/153) Enlighten-Your-Voice: When Multimodal Meets Zero-shot Low-light Image Enhancement (Xiaofeng Zhang et al., 2023)

{{<citation>}}

Xiaofeng Zhang, Zishan Xu, Hao Tang, Chaochen Gu, Wei Chen, Shanying Zhu, Xinping Guan. (2023)  
**Enlighten-Your-Voice: When Multimodal Meets Zero-shot Low-light Image Enhancement**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.10109v1)  

---


**ABSTRACT**  
Low-light image enhancement is a crucial visual task, and many unsupervised methods tend to overlook the degradation of visible information in low-light scenes, which adversely affects the fusion of complementary information and hinders the generation of satisfactory results. To address this, our study introduces ``Enlighten-Your-Voice'', a multimodal enhancement framework that innovatively enriches user interaction through voice and textual commands. This approach does not merely signify a technical leap but also represents a paradigm shift in user engagement. Our model is equipped with a Dual Collaborative Attention Module (DCAM) that meticulously caters to distinct content and color discrepancies, thereby facilitating nuanced enhancements. Complementarily, we introduce a Semantic Feature Fusion (SFM) plug-and-play module that synergizes semantic context with low-light enhancement operations, sharpening the algorithm's efficacy. Crucially, ``Enlighten-Your-Voice'' showcases remarkable generalization in unsupervised zero-shot scenarios. The source code can be accessed from https://github.com/zhangbaijin/Enlighten-Your-Voice

{{</citation>}}


### (57/153) Towards Transferable Targeted 3D Adversarial Attack in the Physical World (Yao Huang et al., 2023)

{{<citation>}}

Yao Huang, Yinpeng Dong, Shouwei Ruan, Xiao Yang, Hang Su, Xingxing Wei. (2023)  
**Towards Transferable Targeted 3D Adversarial Attack in the Physical World**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.09558v1)  

---


**ABSTRACT**  
Compared with transferable untargeted attacks, transferable targeted adversarial attacks could specify the misclassification categories of adversarial samples, posing a greater threat to security-critical tasks. In the meanwhile, 3D adversarial samples, due to their potential of multi-view robustness, can more comprehensively identify weaknesses in existing deep learning systems, possessing great application value. However, the field of transferable targeted 3D adversarial attacks remains vacant. The goal of this work is to develop a more effective technique that could generate transferable targeted 3D adversarial examples, filling the gap in this field. To achieve this goal, we design a novel framework named TT3D that could rapidly reconstruct from few multi-view images into Transferable Targeted 3D textured meshes. While existing mesh-based texture optimization methods compute gradients in the high-dimensional mesh space and easily fall into local optima, leading to unsatisfactory transferability and distinct distortions, TT3D innovatively performs dual optimization towards both feature grid and Multi-layer Perceptron (MLP) parameters in the grid-based NeRF space, which significantly enhances black-box transferability while enjoying naturalness. Experimental results show that TT3D not only exhibits superior cross-model transferability but also maintains considerable adaptability across different renders and vision tasks. More importantly, we produce 3D adversarial examples with 3D printing techniques in the real world and verify their robust performance under various scenarios.

{{</citation>}}


### (58/153) Privacy-Aware Document Visual Question Answering (Rubèn Tito et al., 2023)

{{<citation>}}

Rubèn Tito, Khanh Nguyen, Marlon Tobaben, Raouf Kerkouche, Mohamed Ali Souibgui, Kangsoo Jung, Lei Kang, Ernest Valveny, Antti Honkela, Mario Fritz, Dimosthenis Karatzas. (2023)  
**Privacy-Aware Document Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: OCR, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.10108v1)  

---


**ABSTRACT**  
Document Visual Question Answering (DocVQA) is a fast growing branch of document understanding. Despite the fact that documents contain sensitive or copyrighted information, none of the current DocVQA methods offers strong privacy guarantees.   In this work, we explore privacy in the domain of DocVQA for the first time. We highlight privacy issues in state of the art multi-modal LLM models used for DocVQA, and explore possible solutions.   Specifically, we focus on the invoice processing use case as a realistic, widely used scenario for document understanding, and propose a large scale DocVQA dataset comprising invoice documents and associated questions and answers. We employ a federated learning scheme, that reflects the real-life distribution of documents in different businesses, and we explore the use case where the ID of the invoice issuer is the sensitive information to be protected.   We demonstrate that non-private models tend to memorise, behaviour that can lead to exposing private information. We then evaluate baseline training schemes employing federated learning and differential privacy in this multi-modal scenario, where the sensitive information might be exposed through any of the two input modalities: vision (document image) or language (OCR tokens).   Finally, we design an attack exploiting the memorisation effect of the model, and demonstrate its effectiveness in probing different DocVQA models.

{{</citation>}}


### (59/153) Embodied Adversarial Attack: A Dynamic Robust Physical Attack in Autonomous Driving (Yitong Sun et al., 2023)

{{<citation>}}

Yitong Sun, Yao Huang, Xingxing Wei. (2023)  
**Embodied Adversarial Attack: A Dynamic Robust Physical Attack in Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.09554v1)  

---


**ABSTRACT**  
As physical adversarial attacks become extensively applied in unearthing the potential risk of security-critical scenarios, especially in autonomous driving, their vulnerability to environmental changes has also been brought to light. The non-robust nature of physical adversarial attack methods brings less-than-stable performance consequently. To enhance the robustness of physical adversarial attacks in the real world, instead of statically optimizing a robust adversarial example via an off-line training manner like the existing methods, this paper proposes a brand new robust adversarial attack framework: Embodied Adversarial Attack (EAA) from the perspective of dynamic adaptation, which aims to employ the paradigm of embodied intelligence: Perception-Decision-Control to dynamically adjust the optimal attack strategy according to the current situations in real time. For the perception module, given the challenge of needing simulation for the victim's viewpoint, EAA innovatively devises a Perspective Transformation Network to estimate the target's transformation from the attacker's perspective. For the decision and control module, EAA adopts the laser-a highly manipulable medium to implement physical attacks, and further trains an attack agent with reinforcement learning to make it capable of instantaneously determining the best attack strategy based on the perceived information. Finally, we apply our framework to the autonomous driving scenario. A variety of experiments verify the high effectiveness of our method under complex scenes.

{{</citation>}}


### (60/153) AEGIS-Net: Attention-guided Multi-Level Feature Aggregation for Indoor Place Recognition (Yuhang Ming et al., 2023)

{{<citation>}}

Yuhang Ming, Jian Ma, Xingrui Yang, Weichen Dai, Yong Peng, Wanzeng Kong. (2023)  
**AEGIS-Net: Attention-guided Multi-Level Feature Aggregation for Indoor Place Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.09538v1)  

---


**ABSTRACT**  
We present AEGIS-Net, a novel indoor place recognition model that takes in RGB point clouds and generates global place descriptors by aggregating lower-level color, geometry features and higher-level implicit semantic features. However, rather than simple feature concatenation, self-attention modules are employed to select the most important local features that best describe an indoor place. Our AEGIS-Net is made of a semantic encoder, a semantic decoder and an attention-guided feature embedding. The model is trained in a 2-stage process with the first stage focusing on an auxiliary semantic segmentation task and the second one on the place recognition task. We evaluate our AEGIS-Net on the ScanNetPR dataset and compare its performance with a pre-deep-learning feature-based method and five state-of-the-art deep-learning-based methods. Our AEGIS-Net achieves exceptional performance and outperforms all six methods.

{{</citation>}}


### (61/153) WeatherProof: A Paired-Dataset Approach to Semantic Segmentation in Adverse Weather (Blake Gella et al., 2023)

{{<citation>}}

Blake Gella, Howard Zhang, Rishi Upadhyay, Tiffany Chang, Matthew Waliman, Yunhao Ba, Alex Wong, Achuta Kadambi. (2023)  
**WeatherProof: A Paired-Dataset Approach to Semantic Segmentation in Adverse Weather**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.09534v1)  

---


**ABSTRACT**  
The introduction of large, foundational models to computer vision has led to drastically improved performance on the task of semantic segmentation. However, these existing methods exhibit a large performance drop when testing on images degraded by weather conditions such as rain, fog, or snow. We introduce a general paired-training method that can be applied to all current foundational model architectures that leads to improved performance on images in adverse weather conditions. To this end, we create the WeatherProof Dataset, the first semantic segmentation dataset with accurate clear and adverse weather image pairs, which not only enables our new training paradigm, but also improves the evaluation of the performance gap between clear and degraded segmentation. We find that training on these paired clear and adverse weather frames which share an underlying scene results in improved performance on adverse weather data. With this knowledge, we propose a training pipeline which accentuates the advantages of paired-data training using consistency losses and language guidance, which leads to performance improvements by up to 18.4% as compared to standard training procedures.

{{</citation>}}


### (62/153) Hierarchical Graph Pattern Understanding for Zero-Shot VOS (Gensheng Pei et al., 2023)

{{<citation>}}

Gensheng Pei, Fumin Shen, Yazhou Yao, Tao Chen, Xian-Sheng Hua, Heng-Tao Shen. (2023)  
**Hierarchical Graph Pattern Understanding for Zero-Shot VOS**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GNN, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.09525v1)  

---


**ABSTRACT**  
The optical flow guidance strategy is ideal for obtaining motion information of objects in the video. It is widely utilized in video segmentation tasks. However, existing optical flow-based methods have a significant dependency on optical flow, which results in poor performance when the optical flow estimation fails for a particular scene. The temporal consistency provided by the optical flow could be effectively supplemented by modeling in a structural form. This paper proposes a new hierarchical graph neural network (GNN) architecture, dubbed hierarchical graph pattern understanding (HGPU), for zero-shot video object segmentation (ZS-VOS). Inspired by the strong ability of GNNs in capturing structural relations, HGPU innovatively leverages motion cues (\ie, optical flow) to enhance the high-order representations from the neighbors of target frames. Specifically, a hierarchical graph pattern encoder with message aggregation is introduced to acquire different levels of motion and appearance features in a sequential manner. Furthermore, a decoder is designed for hierarchically parsing and understanding the transformed multi-modal contexts to achieve more accurate and robust results. HGPU achieves state-of-the-art performance on four publicly available benchmarks (DAVIS-16, YouTube-Objects, Long-Videos and DAVIS-17). Code and pre-trained model can be found at \url{https://github.com/NUST-Machine-Intelligence-Laboratory/HGPU}.

{{</citation>}}


### (63/153) Forging Tokens for Improved Storage-efficient Training (Minhyun Lee et al., 2023)

{{<citation>}}

Minhyun Lee, Song Park, Byeongho Heo, Dongyoon Han, Hyunjung Shim. (2023)  
**Forging Tokens for Improved Storage-efficient Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.10105v1)  

---


**ABSTRACT**  
Recent advancements in Deep Neural Network (DNN) models have significantly improved performance across computer vision tasks. However, achieving highly generalizable and high-performing vision models requires extensive datasets, leading to large storage requirements. This storage challenge poses a critical bottleneck for scaling up vision models. Motivated by the success of discrete representations, SeiT proposes to use Vector-Quantized (VQ) feature vectors (i.e., tokens) as network inputs for vision classification. However, applying traditional data augmentations to tokens faces challenges due to input domain shift. To address this issue, we introduce TokenAdapt and ColorAdapt, simple yet effective token-based augmentation strategies. TokenAdapt realigns token embedding space for compatibility with spatial augmentations, preserving the model's efficiency without requiring fine-tuning. Additionally, ColorAdapt addresses color-based augmentations for tokens inspired by Adaptive Instance Normalization (AdaIN). We evaluate our approach across various scenarios, including storage-efficient ImageNet-1k classification, fine-grained classification, robustness benchmarks, and ADE-20k semantic segmentation. Experimental results demonstrate consistent performance improvement in diverse experiments. Code is available at https://github.com/naver-ai/tokenadapt.

{{</citation>}}


### (64/153) WAVER: Writing-style Agnostic Video Retrieval via Distilling Vision-Language Models Through Open-Vocabulary Knowledge (Huy Le et al., 2023)

{{<citation>}}

Huy Le, Tung Kieu, Anh Nguyen, Ngan Le. (2023)  
**WAVER: Writing-style Agnostic Video Retrieval via Distilling Vision-Language Models Through Open-Vocabulary Knowledge**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09507v1)  

---


**ABSTRACT**  
Text-video retrieval, a prominent sub-field within the broader domain of multimedia content management, has witnessed remarkable growth and innovation over the past decade. However, existing methods assume the video scenes are consistent and the description annotators are unbiased. These limitations fail to align with fluid real-world scenarios, and descriptions can be influenced by annotator biases, diverse writing styles, and varying textual perspectives. To overcome the aforementioned problems, we introduce WAVER, a cross-domain knowledge distillation mechanism designed to tackle the challenge of handling writing-style agnostics. WAVER capitalizes on the open-vocabulary properties inherent in pre-trained vision-language models and employs an implicit knowledge distillation approach to transfer text-based knowledge from a teacher model to a vision-based student. Empirical studies conducted across four standard benchmark datasets, encompassing various settings, provide compelling evidence that \WAVER can achieve state-of-the-art performance in text-video retrieval tasks while handling writing-style variations.

{{</citation>}}


### (65/153) ICD-LM: Configuring Vision-Language In-Context Demonstrations by Language Modeling (Yingzhe Peng et al., 2023)

{{<citation>}}

Yingzhe Peng, Xu Yang, Haoxuan Ma, Shuo Xu, Chi Zhang, Yucheng Han, Hanwang Zhang. (2023)  
**ICD-LM: Configuring Vision-Language In-Context Demonstrations by Language Modeling**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Image Captioning, Language Model, NLP, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.10104v1)  

---


**ABSTRACT**  
This paper studies how to configure powerful In-Context Demonstration (ICD) sequences for a Large Vision-Language Model (LVLM) to solve Vision-Language tasks through In-Context Learning (ICL). After observing that configuring an ICD sequence is a mirror process of composing a sentence, i.e., just as a sentence can be composed word by word via a Language Model, an ICD sequence can also be configured one by one. Consequently, we introduce an ICD Language Model (ICD-LM) specifically designed to generate effective ICD sequences. This involves creating a dataset of hand-crafted ICD sequences for various query samples and using it to train the ICD-LM. Our approach, diverging from traditional methods in NLP that select and order ICDs separately, enables to simultaneously learn how to select and order ICDs, enhancing the effect of the sequences. Moreover, during data construction, we use the LVLM intended for ICL implementation to validate the strength of each ICD sequence, resulting in a model-specific dataset and the ICD-LM trained by this dataset is also model-specific. We validate our methodology through experiments in Visual Question Answering and Image Captioning, confirming the viability of using a Language Model for ICD configuration. Our comprehensive ablation studies further explore the impact of various dataset construction and ICD-LM development settings on the outcomes. The code is given in https://github.com/ForJadeForest/ICD-LM.

{{</citation>}}


### (66/153) GSVA: Generalized Segmentation via Multimodal Large Language Models (Zhuofan Xia et al., 2023)

{{<citation>}}

Zhuofan Xia, Dongchen Han, Yizeng Han, Xuran Pan, Shiji Song, Gao Huang. (2023)  
**GSVA: Generalized Segmentation via Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.10103v1)  

---


**ABSTRACT**  
Generalized Referring Expression Segmentation (GRES) extends the scope of classic RES to referring to multiple objects in one expression or identifying the empty targets absent in the image. GRES poses challenges in modeling the complex spatial relationships of the instances in the image and identifying non-existing referents. Recently, Multimodal Large Language Models (MLLMs) have shown tremendous progress in these complicated vision-language tasks. Connecting Large Language Models (LLMs) and vision models, MLLMs are proficient in understanding contexts with visual inputs. Among them, LISA, as a representative, adopts a special [SEG] token to prompt a segmentation mask decoder, e.g., SAM, to enable MLLMs in the RES task. However, existing solutions to of GRES remain unsatisfactory since current segmentation MLLMs cannot properly handle the cases where users might reference multiple subjects in a singular prompt or provide descriptions incongruent with any image target. In this paper, we propose Generalized Segmentation Vision Assistant (GSVA) to address this gap. Specifically, GSVA reuses the [SEG] token to prompt the segmentation model towards supporting multiple mask references simultaneously and innovatively learns to generate a [REJ] token to reject the null targets explicitly. Experiments validate GSVA's efficacy in resolving the GRES issue, marking a notable enhancement and setting a new record on the GRES benchmark gRefCOCO dataset. GSVA also proves effective across various classic referring expression segmentation and comprehension tasks.

{{</citation>}}


### (67/153) Continual Adversarial Defense (Qian Wang et al., 2023)

{{<citation>}}

Qian Wang, Yaoyao Liu, Hefei Ling, Yingwei Li, Qihao Liu, Ping Li, Jiazhong Chen, Alan Yuille, Ning Yu. (2023)  
**Continual Adversarial Defense**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.09481v1)  

---


**ABSTRACT**  
In response to the rapidly evolving nature of adversarial attacks on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that can generalize to all types of attacks, including unseen ones, is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks used by many attackers. The defense system needs to upgrade itself by utilizing few-shot defense feedback and efficient memory. Therefore, we propose the first continual adversarial defense (CAD) framework that adapts to any attacks in a dynamic scenario, where various attacks emerge stage by stage. In practice, CAD is modeled under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accuracy on both clean and adversarial images. We leverage cutting-edge continual learning, few-shot learning, and ensemble learning techniques to qualify the principles. Experiments conducted on CIFAR-10 and ImageNet-100 validate the effectiveness of our approach against multiple stages of 10 modern adversarial attacks and significant improvements over 10 baseline methods. In particular, CAD is capable of quickly adapting with minimal feedback and a low cost of defense failure, while maintaining good performance against old attacks. Our research sheds light on a brand-new paradigm for continual defense adaptation against dynamic and evolving attacks.

{{</citation>}}


### (68/153) TAB: Text-Align Anomaly Backbone Model for Industrial Inspection Tasks (Ho-Weng Lee et al., 2023)

{{<citation>}}

Ho-Weng Lee, Shang-Hong Lai. (2023)  
**TAB: Text-Align Anomaly Backbone Model for Industrial Inspection Tasks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.09480v1)  

---


**ABSTRACT**  
In recent years, the focus on anomaly detection and localization in industrial inspection tasks has intensified. While existing studies have demonstrated impressive outcomes, they often rely heavily on extensive training datasets or robust features extracted from pre-trained models trained on diverse datasets like ImageNet. In this work, we propose a novel framework leveraging the visual-linguistic CLIP model to adeptly train a backbone model tailored to the manufacturing domain. Our approach concurrently considers visual and text-aligned embedding spaces for normal and abnormal conditions. The resulting pre-trained backbone markedly enhances performance in industrial downstream tasks, particularly in anomaly detection and localization. Notably, this improvement is substantiated through experiments conducted on multiple datasets such as MVTecAD, BTAD, and KSDD2. Furthermore, using our pre-trained backbone weights allows previous works to achieve superior performance in few-shot scenarios with less training data. The proposed anomaly backbone provides a foundation model for more precise anomaly detection and localization.

{{</citation>}}


## cs.AI (14)



### (69/153) Constrained Meta-Reinforcement Learning for Adaptable Safety Guarantee with Differentiable Convex Programming (Minjae Cho et al., 2023)

{{<citation>}}

Minjae Cho, Chuangchuang Sun. (2023)  
**Constrained Meta-Reinforcement Learning for Adaptable Safety Guarantee with Differentiable Convex Programming**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10230v1)  

---


**ABSTRACT**  
Despite remarkable achievements in artificial intelligence, the deployability of learning-enabled systems in high-stakes real-world environments still faces persistent challenges. For example, in safety-critical domains like autonomous driving, robotic manipulation, and healthcare, it is crucial not only to achieve high performance but also to comply with given constraints. Furthermore, adaptability becomes paramount in non-stationary domains, where environmental parameters are subject to change. While safety and adaptability are recognized as key qualities for the new generation of AI, current approaches have not demonstrated effective adaptable performance in constrained settings. Hence, this paper breaks new ground by studying the unique challenges of ensuring safety in non-stationary environments by solving constrained problems through the lens of the meta-learning approach (learning-to-learn). While unconstrained meta-learning al-ready encounters complexities in end-to-end differentiation of the loss due to the bi-level nature, its constrained counterpart introduces an additional layer of difficulty, since the constraints imposed on task-level updates complicate the differentiation process. To address the issue, we first employ successive convex-constrained policy updates across multiple tasks with differentiable convexprogramming, which allows meta-learning in constrained scenarios by enabling end-to-end differentiation. This approach empowers the agent to rapidly adapt to new tasks under non-stationarity while ensuring compliance with safety constraints.

{{</citation>}}


### (70/153) One Self-Configurable Model to Solve Many Abstract Visual Reasoning Problems (Mikołaj Małkiński et al., 2023)

{{<citation>}}

Mikołaj Małkiński, Jacek Mańdziuk. (2023)  
**One Self-Configurable Model to Solve Many Abstract Visual Reasoning Problems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-LG, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.09997v1)  

---


**ABSTRACT**  
Abstract Visual Reasoning (AVR) comprises a wide selection of various problems similar to those used in human IQ tests. Recent years have brought dynamic progress in solving particular AVR tasks, however, in the contemporary literature AVR problems are largely dealt with in isolation, leading to highly specialized task-specific methods. With the aim of developing universal learning systems in the AVR domain, we propose the unified model for solving Single-Choice Abstract visual Reasoning tasks (SCAR), capable of solving various single-choice AVR tasks, without making any a priori assumptions about the task structure, in particular the number and location of panels. The proposed model relies on a novel Structure-Aware dynamic Layer (SAL), which adapts its weights to the structure of the considered AVR problem. Experiments conducted on Raven's Progressive Matrices, Visual Analogy Problems, and Odd One Out problems show that SCAR (SAL-based models, in general) effectively solves diverse AVR tasks, and its performance is on par with the state-of-the-art task-specific baselines. What is more, SCAR demonstrates effective knowledge reuse in multi-task and transfer learning settings. To our knowledge, this work is the first successful attempt to construct a general single-choice AVR solver relying on self-configurable architecture and unified solving method. With this work we aim to stimulate and foster progress on task-independent research paths in the AVR domain, with the long-term goal of development of a general AVR solver.

{{</citation>}}


### (71/153) Distilling Large Language Models for Matching Patients to Clinical Trials (Mauro Nievas et al., 2023)

{{<citation>}}

Mauro Nievas, Aditya Basu, Yanshan Wang, Hrituraj Singh. (2023)  
**Distilling Large Language Models for Matching Patients to Clinical Trials**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IR, cs.AI  
Keywords: AI, Clinical, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.09958v1)  

---


**ABSTRACT**  
The recent success of large language models (LLMs) has paved the way for their adoption in the high-stakes domain of healthcare. Specifically, the application of LLMs in patient-trial matching, which involves assessing patient eligibility against clinical trial's nuanced inclusion and exclusion criteria, has shown promise. Recent research has shown that GPT-3.5, a widely recognized LLM developed by OpenAI, can outperform existing methods with minimal 'variable engineering' by simply comparing clinical trial information against patient summaries. However, there are significant challenges associated with using closed-source proprietary LLMs like GPT-3.5 in practical healthcare applications, such as cost, privacy and reproducibility concerns. To address these issues, this study presents the first systematic examination of the efficacy of both proprietary (GPT-3.5, and GPT-4) and open-source LLMs (LLAMA 7B,13B, and 70B) for the task of patient-trial matching. Employing a multifaceted evaluation framework, we conducted extensive automated and human-centric assessments coupled with a detailed error analysis for each model. To enhance the adaptability of open-source LLMs, we have created a specialized synthetic dataset utilizing GPT-4, enabling effective fine-tuning under constrained data conditions. Our findings reveal that open-source LLMs, when fine-tuned on this limited and synthetic dataset, demonstrate performance parity with their proprietary counterparts. This presents a massive opportunity for their deployment in real-world healthcare applications. To foster further research and applications in this field, we release both the annotated evaluation dataset along with the fine-tuned LLM -- Trial-LLAMA -- for public use.

{{</citation>}}


### (72/153) Neurosymbolic Value-Inspired AI (Why, What, and How) (Amit Sheth et al., 2023)

{{<citation>}}

Amit Sheth, Kaushik Roy. (2023)  
**Neurosymbolic Value-Inspired AI (Why, What, and How)**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.09928v1)  

---


**ABSTRACT**  
The rapid progression of Artificial Intelligence (AI) systems, facilitated by the advent of Large Language Models (LLMs), has resulted in their widespread application to provide human assistance across diverse industries. This trend has sparked significant discourse centered around the ever-increasing need for LLM-based AI systems to function among humans as part of human society, sharing human values, especially as these systems are deployed in high-stakes settings (e.g., healthcare, autonomous driving, etc.). Towards this end, neurosymbolic AI systems are attractive due to their potential to enable easy-to-understand and interpretable interfaces for facilitating value-based decision-making, by leveraging explicit representations of shared values. In this paper, we introduce substantial extensions to Khaneman's System one/two framework and propose a neurosymbolic computational framework called Value-Inspired AI (VAI). It outlines the crucial components essential for the robust and practical implementation of VAI systems, aiming to represent and integrate various dimensions of human values. Finally, we further offer insights into the current progress made in this direction and outline potential future directions for the field.

{{</citation>}}


### (73/153) A Novel Dataset for Financial Education Text Simplification in Spanish (Nelson Perez-Rojas et al., 2023)

{{<citation>}}

Nelson Perez-Rojas, Saul Calderon-Ramirez, Martin Solis-Salazar, Mario Romero-Sandoval, Monica Arias-Monge, Horacio Saggion. (2023)  
**A Novel Dataset for Financial Education Text Simplification in Spanish**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Financial, GPT, T5  
[Paper Link](http://arxiv.org/abs/2312.09897v1)  

---


**ABSTRACT**  
Text simplification, crucial in natural language processing, aims to make texts more comprehensible, particularly for specific groups like visually impaired Spanish speakers, a less-represented language in this field. In Spanish, there are few datasets that can be used to create text simplification systems. Our research has the primary objective to develop a Spanish financial text simplification dataset. We created a dataset with 5,314 complex and simplified sentence pairs using established simplification rules. We also compared our dataset with the simplifications generated from GPT-3, Tuner, and MT5, in order to evaluate the feasibility of data augmentation using these systems. In this manuscript we present the characteristics of our dataset and the findings of the comparisons with other systems. The dataset is available at Hugging face, saul1917/FEINA.

{{</citation>}}


### (74/153) 3DAxiesPrompts: Unleashing the 3D Spatial Task Capabilities of GPT-4V (Dingning Liu et al., 2023)

{{<citation>}}

Dingning Liu, Xiaomeng Dong, Renrui Zhang, Xu Luo, Peng Gao, Xiaoshui Huang, Yongshun Gong, Zhihui Wang. (2023)  
**3DAxiesPrompts: Unleashing the 3D Spatial Task Capabilities of GPT-4V**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, GPT-4, Object Detection  
[Paper Link](http://arxiv.org/abs/2312.09738v1)  

---


**ABSTRACT**  
In this work, we present a new visual prompting method called 3DAxiesPrompts (3DAP) to unleash the capabilities of GPT-4V in performing 3D spatial tasks. Our investigation reveals that while GPT-4V exhibits proficiency in discerning the position and interrelations of 2D entities through current visual prompting techniques, its abilities in handling 3D spatial tasks have yet to be explored. In our approach, we create a 3D coordinate system tailored to 3D imagery, complete with annotated scale information. By presenting images infused with the 3DAP visual prompt as inputs, we empower GPT-4V to ascertain the spatial positioning information of the given 3D target image with a high degree of precision. Through experiments, We identified three tasks that could be stably completed using the 3DAP method, namely, 2D to 3D Point Reconstruction, 2D to 3D point matching, and 3D Object Detection. We perform experiments on our proposed dataset 3DAP-Data, the results from these experiments validate the efficacy of 3DAP-enhanced GPT-4V inputs, marking a significant stride in 3D spatial task execution.

{{</citation>}}


### (75/153) Social, Legal, Ethical, Empathetic, and Cultural Rules: Compilation and Reasoning (Extended Version) (Nicolas Troquard et al., 2023)

{{<citation>}}

Nicolas Troquard, Martina De Sanctis, Paola Inverardi, Patrizio Pelliccione, Gian Luca Scoccia. (2023)  
**Social, Legal, Ethical, Empathetic, and Cultural Rules: Compilation and Reasoning (Extended Version)**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Legal, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.09699v1)  

---


**ABSTRACT**  
The rise of AI-based and autonomous systems is raising concerns and apprehension due to potential negative repercussions stemming from their behavior or decisions. These systems must be designed to comply with the human contexts in which they will operate. To this extent, Townsend et al. (2022) introduce the concept of SLEEC (social, legal, ethical, empathetic, or cultural) rules that aim to facilitate the formulation, verification, and enforcement of the rules AI-based and autonomous systems should obey. They lay out a methodology to elicit them and to let philosophers, lawyers, domain experts, and others to formulate them in natural language. To enable their effective use in AI systems, it is necessary to translate these rules systematically into a formal language that supports automated reasoning. In this study, we first conduct a linguistic analysis of the SLEEC rules pattern, which justifies the translation of SLEEC rules into classical logic. Then we investigate the computational complexity of reasoning about SLEEC rules and show how logical programming frameworks can be employed to implement SLEEC rules in practical scenarios. The result is a readily applicable strategy for implementing AI systems that conform to norms expressed as SLEEC rules.

{{</citation>}}


### (76/153) Robustness Verification of Deep Reinforcement Learning Based Control Systems using Reward Martingales (Dapeng Zhi et al., 2023)

{{<citation>}}

Dapeng Zhi, Peixin Wang, Cheng Chen, Min Zhang. (2023)  
**Robustness Verification of Deep Reinforcement Learning Based Control Systems using Reward Martingales**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09695v1)  

---


**ABSTRACT**  
Deep Reinforcement Learning (DRL) has gained prominence as an effective approach for control systems. However, its practical deployment is impeded by state perturbations that can severely impact system performance. Addressing this critical challenge requires robustness verification about system performance, which involves tackling two quantitative questions: (i) how to establish guaranteed bounds for expected cumulative rewards, and (ii) how to determine tail bounds for cumulative rewards. In this work, we present the first approach for robustness verification of DRL-based control systems by introducing reward martingales, which offer a rigorous mathematical foundation to characterize the impact of state perturbations on system performance in terms of cumulative rewards. Our verified results provide provably quantitative certificates for the two questions. We then show that reward martingales can be implemented and trained via neural networks, against different types of control policies. Experimental results demonstrate that our certified bounds tightly enclose simulation outcomes on various DRL-based control systems, indicating the effectiveness and generality of the proposed approach.

{{</citation>}}


### (77/153) Prompting Large Language Models for Topic Modeling (Han Wang et al., 2023)

{{<citation>}}

Han Wang, Nirmalendu Prakash, Nguyen Khoi Hoang, Ming Shan Hee, Usman Naseem, Roy Ka-Wei Lee. (2023)  
**Prompting Large Language Models for Topic Modeling**  

---
Primary Category: cs.AI  
Categories: I-2-7, cs-AI, cs.AI  
Keywords: Language Model, Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2312.09693v1)  

---


**ABSTRACT**  
Topic modeling is a widely used technique for revealing underlying thematic structures within textual data. However, existing models have certain limitations, particularly when dealing with short text datasets that lack co-occurring words. Moreover, these models often neglect sentence-level semantics, focusing primarily on token-level semantics. In this paper, we propose PromptTopic, a novel topic modeling approach that harnesses the advanced language understanding of large language models (LLMs) to address these challenges. It involves extracting topics at the sentence level from individual documents, then aggregating and condensing these topics into a predefined quantity, ultimately providing coherent topics for texts of varying lengths. This approach eliminates the need for manual parameter tuning and improves the quality of extracted topics. We benchmark PromptTopic against the state-of-the-art baselines on three vastly diverse datasets, establishing its proficiency in discovering meaningful topics. Furthermore, qualitative analysis showcases PromptTopic's ability to uncover relevant topics in multiple datasets.

{{</citation>}}


### (78/153) Algorithms for automatic intents extraction and utterances classification for goal-oriented dialogue systems (Leonid Legashev et al., 2023)

{{<citation>}}

Leonid Legashev, Alexander Shukhman, Arthur Zhigalov. (2023)  
**Algorithms for automatic intents extraction and utterances classification for goal-oriented dialogue systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.09658v1)  

---


**ABSTRACT**  
Modern machine learning techniques in the natural language processing domain can be used to automatically generate scripts for goal-oriented dialogue systems. The current article presents a general framework for studying the automatic generation of scripts for goal-oriented dialogue systems. A method for preprocessing dialog data sets in JSON format is described. A comparison is made of two methods for extracting user intent based on BERTopic and latent Dirichlet allocation. A comparison has been made of two implemented algorithms for classifying statements of users of a goal-oriented dialogue system based on logistic regression and BERT transformer models. The BERT transformer approach using the bert-base-uncased model showed better results for the three metrics Precision (0.80), F1-score (0.78) and Matthews correlation coefficient (0.74) in comparison with other methods.

{{</citation>}}


### (79/153) Investigating Responsible AI for Scientific Research: An Empirical Study (Muneera Bano et al., 2023)

{{<citation>}}

Muneera Bano, Didar Zowghi, Pip Shea, Georgina Ibarra. (2023)  
**Investigating Responsible AI for Scientific Research: An Empirical Study**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09561v1)  

---


**ABSTRACT**  
Scientific research organizations that are developing and deploying Artificial Intelligence (AI) systems are at the intersection of technological progress and ethical considerations. The push for Responsible AI (RAI) in such institutions underscores the increasing emphasis on integrating ethical considerations within AI design and development, championing core values like fairness, accountability, and transparency. For scientific research organizations, prioritizing these practices is paramount not just for mitigating biases and ensuring inclusivity, but also for fostering trust in AI systems among both users and broader stakeholders. In this paper, we explore the practices at a research organization concerning RAI practices, aiming to assess the awareness and preparedness regarding the ethical risks inherent in AI design and development. We have adopted a mixed-method research approach, utilising a comprehensive survey combined with follow-up in-depth interviews with selected participants from AI-related projects. Our results have revealed certain knowledge gaps concerning ethical, responsible, and inclusive AI, with limitations in awareness of the available AI ethics frameworks. This revealed an overarching underestimation of the ethical risks that AI technologies can present, especially when implemented without proper guidelines and governance. Our findings reveal the need for a holistic and multi-tiered strategy to uplift capabilities and better support science research teams for responsible, ethical, and inclusive AI development and deployment.

{{</citation>}}


### (80/153) On a Functional Definition of Intelligence (Warisa Sritriratanarak et al., 2023)

{{<citation>}}

Warisa Sritriratanarak, Paulo Garcia. (2023)  
**On a Functional Definition of Intelligence**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09546v1)  

---


**ABSTRACT**  
Without an agreed-upon definition of intelligence, asking "is this system intelligent?"" is an untestable question. This lack of consensus hinders research, and public perception, on Artificial Intelligence (AI), particularly since the rise of generative- and large-language models. Most work on precisely capturing what we mean by "intelligence" has come from the fields of philosophy, psychology, and cognitive science. Because these perspectives are intrinsically linked to intelligence as it is demonstrated by natural creatures, we argue such fields cannot, and will not, provide a sufficiently rigorous definition that can be applied to artificial means. Thus, we present an argument for a purely functional, black-box definition of intelligence, distinct from how that intelligence is actually achieved; focusing on the "what", rather than the "how". To achieve this, we first distinguish other related concepts (sentience, sensation, agency, etc.) from the notion of intelligence, particularly identifying how these concepts pertain to artificial intelligent systems. As a result, we achieve a formal definition of intelligence that is conceptually testable from only external observation, that suggests intelligence is a continuous variable. We conclude by identifying challenges that still remain towards quantifiable measurement. This work provides a useful perspective for both the development of AI, and for public perception of the capabilities and risks of AI.

{{</citation>}}


### (81/153) Situation-Dependent Causal Influence-Based Cooperative Multi-agent Reinforcement Learning (Xiao Du et al., 2023)

{{<citation>}}

Xiao Du, Yutong Ye, Pengyu Zhang, Yaning Yang, Mingsong Chen, Ting Wang. (2023)  
**Situation-Dependent Causal Influence-Based Cooperative Multi-agent Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09539v1)  

---


**ABSTRACT**  
Learning to collaborate has witnessed significant progress in multi-agent reinforcement learning (MARL). However, promoting coordination among agents and enhancing exploration capabilities remain challenges. In multi-agent environments, interactions between agents are limited in specific situations. Effective collaboration between agents thus requires a nuanced understanding of when and how agents' actions influence others. To this end, in this paper, we propose a novel MARL algorithm named Situation-Dependent Causal Influence-Based Cooperative Multi-agent Reinforcement Learning (SCIC), which incorporates a novel Intrinsic reward mechanism based on a new cooperation criterion measured by situation-dependent causal influence among agents. Our approach aims to detect inter-agent causal influences in specific situations based on the criterion using causal intervention and conditional mutual information. This effectively assists agents in exploring states that can positively impact other agents, thus promoting cooperation between agents. The resulting update links coordinated exploration and intrinsic reward distribution, which enhance overall collaboration and performance. Experimental results on various MARL benchmarks demonstrate the superiority of our method compared to state-of-the-art approaches.

{{</citation>}}


### (82/153) CGS-Mask: Making Time Series Predictions Intuitive for Al (Feng Lu et al., 2023)

{{<citation>}}

Feng Lu, Wei Li, Yifei Sun, Cheng Song, Yufei Ren, Albert Y. Zomaya. (2023)  
**CGS-Mask: Making Time Series Predictions Intuitive for Al**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Time Series  
[Paper Link](http://arxiv.org/abs/2312.09513v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) has immense potential in time series prediction, but most explainable tools have limited capabilities in providing a systematic understanding of important features over time. These tools typically rely on evaluating a single time point, overlook the time ordering of inputs, and neglect the time-sensitive nature of time series applications. These factors make it difficult for users, particularly those without domain knowledge, to comprehend AI model decisions and obtain meaningful explanations. We propose CGS-Mask, a post-hoc and model-agnostic cellular genetic strip mask-based saliency approach to address these challenges. CGS-Mask uses consecutive time steps as a cohesive entity to evaluate the impact of features on the final prediction, providing binary and sustained feature importance scores over time. Our algorithm optimizes the mask population iteratively to obtain the optimal mask in a reasonable time. We evaluated CGS-Mask on synthetic and real-world datasets, and it outperformed state-of-the-art methods in elucidating the importance of features over time. According to our pilot user study via a questionnaire survey, CGS-Mask is the most effective approach in presenting easily understandable time series prediction results, enabling users to comprehend the decision-making process of AI models with ease.

{{</citation>}}


## cs.CY (4)



### (83/153) GPT-doctor: Customizing Large Language Models for Medical Consultation (Wen Wang et al., 2023)

{{<citation>}}

Wen Wang, Zhenyue Zhao, Tianshu Sun. (2023)  
**GPT-doctor: Customizing Large Language Models for Medical Consultation**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.10225v1)  

---


**ABSTRACT**  
The advent of Large Language Models (LLMs) has ushered in a new era for design science in Information Systems, demanding a paradigm shift in tailoring LLMs design for business contexts. This paper proposes a novel framework to customize LLMs for general business contexts that aims to achieve three fundamental objectives simultaneously: (1) aligning conversational patterns, (2) integrating in-depth domain knowledge, and (3) embodying the soft skills and core principles. We design methodologies to combine domain-specific theory with Supervised Fine Tuning (SFT) in LLMs. We instantiate our proposed framework in the context of medical consultation, creating a GPT-doctor model. Specifically, we construct a comprehensive dataset for SFT by collecting large volume of real doctors consultation records from a leading online medical consultation platform and medical knowledge from professional databases. Additionally, drawing on medical theory, we identify three soft skills and core principles of human doctors including professionalism, explainability, and emotional support, and design approaches to integrate these skills into LLMs. We demonstrate the feasibility and performance of our proposed framework using online experiments with real patients as well as evaluation by domain experts and real consumers. Results demonstrate that fine-tuned GPT-doctor performs on par with human doctors across multiple metrics including medical expertise and consumer preference. Finally, we unravel the black box and examine the sources of model performance improvement from the perspectives of horizontal conversation pattern alignment and vertical medical knowledge evolution. Our proposed framework offers step-by-step principles and guidance for customizing LLMs for real-world business problems.

{{</citation>}}


### (84/153) Expert-Level Annotation Quality Achieved by Gamified Crowdsourcing for B-line Segmentation in Lung Ultrasound (Mike Jin et al., 2023)

{{<citation>}}

Mike Jin, Nicole M Duggan, Varoon Bashyakarla, Maria Alejandra Duran Mendicuti, Stephen Hallisey, Denie Bernier, Joseph Stegeman, Erik Duhaime, Tina Kapur, Andrew J Goldsmith. (2023)  
**Expert-Level Annotation Quality Achieved by Gamified Crowdsourcing for B-line Segmentation in Lung Ultrasound**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.10198v1)  

---


**ABSTRACT**  
Accurate and scalable annotation of medical data is critical for the development of medical AI, but obtaining time for annotation from medical experts is challenging. Gamified crowdsourcing has demonstrated potential for obtaining highly accurate annotations for medical data at scale, and we demonstrate the same in this study for the segmentation of B-lines, an indicator of pulmonary congestion, on still frames within point-of-care lung ultrasound clips. We collected 21,154 annotations from 214 annotators over 2.5 days, and we demonstrated that the concordance of crowd consensus segmentations with reference standards exceeds that of individual experts with the same reference standards, both in terms of B-line count (mean squared error 0.239 vs. 0.308, p<0.05) as well as the spatial precision of B-line annotations (mean Dice-H score 0.755 vs. 0.643, p<0.05). These results suggest that expert-quality segmentations can be achieved using gamified crowdsourcing.

{{</citation>}}


### (85/153) Integrating New Technologies into Science: The case of AI (Stefano Bianchini et al., 2023)

{{<citation>}}

Stefano Bianchini, Moritz Müller, Pierre Pelletier. (2023)  
**Integrating New Technologies into Science: The case of AI**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY, econ-GN, q-fin-EC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09843v1)  

---


**ABSTRACT**  
New technologies have the power to revolutionize science. It has happened in the past and is happening again with the emergence of new computational tools, such as Artificial Intelligence (AI) and Machine Learning (ML). Despite the documented impact of these technologies, there remains a significant gap in understanding the process of their adoption within the scientific community. In this paper, we draw on theories of scientific and technical human capital (STHC) to study the integration of AI in scientific research, focusing on the human capital of scientists and the external resources available within their network of collaborators and institutions. We validate our hypotheses on a large sample of publications from OpenAlex, covering all sciences from 1980 to 2020. We find that the diffusion of AI is strongly driven by social mechanisms that organize the deployment and creation of human capital that complements the technology. Our results suggest that AI is pioneered by domain scientists with a `taste for exploration' and who are embedded in a network rich of computer scientists, experienced AI scientists and early-career researchers; they also come from institutions with high citation impact and a relatively strong publication history on AI. The pattern is similar across scientific disciplines, the exception being access to high-performance computing (HPC), which is important in chemistry and the medical sciences but less so in other fields. Once AI is integrated into research, most adoption factors continue to influence its subsequent reuse. Implications for the organization and management of science in the evolving era of AI-driven discovery are discussed.

{{</citation>}}


### (86/153) Integrating AI and Learning Analytics for Data-Driven Pedagogical Decisions and Personalized Interventions in Education (Ramteja Sajja et al., 2023)

{{<citation>}}

Ramteja Sajja, Yusuf Sermet, David Cwiertny, Ibrahim Demir. (2023)  
**Integrating AI and Learning Analytics for Data-Driven Pedagogical Decisions and Personalized Interventions in Education**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-HC, cs.CY  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.09548v1)  

---


**ABSTRACT**  
This research study delves into the conceptualization, development, and deployment of an innovative learning analytics tool, leveraging the capabilities of OpenAI's GPT-4 model. This tool is designed to quantify student engagement, map learning progression, and evaluate the efficacy of diverse instructional strategies within an educational context. Through the analysis of various critical data points such as students' stress levels, curiosity, confusion, agitation, topic preferences, and study methods, the tool offers a rich, multi-dimensional view of the learning environment. Furthermore, it employs Bloom's taxonomy as a framework to gauge the cognitive levels addressed by students' questions, thereby elucidating their learning progression. The information gathered from these measurements can empower educators by providing valuable insights to enhance teaching methodologies, pinpoint potential areas for improvement, and craft personalized interventions for individual students. The study articulates the design intricacies, implementation strategy, and thorough evaluation of the learning analytics tool, underscoring its prospective contributions to enhancing educational outcomes and bolstering student success. Moreover, the practicalities of integrating the tool within existing educational platforms and the requisite robust, secure, and scalable technical infrastructure are addressed. This research opens avenues for harnessing AI's potential in shaping the future of education, facilitating data-driven pedagogical decisions, and ultimately fostering a more conducive, personalized learning environment.

{{</citation>}}


## math.OC (2)



### (87/153) Joint Expansion Planning of Power and Water Distribution Networks (Sai Krishna Kanth Hari et al., 2023)

{{<citation>}}

Sai Krishna Kanth Hari, Ahmed Zamzam, Byron Tasseff, Russell Bent, Clayton Barrows. (2023)  
**Joint Expansion Planning of Power and Water Distribution Networks**  

---
Primary Category: math.OC  
Categories: cs-SY, eess-SY, math-OC, math.OC  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.10224v1)  

---


**ABSTRACT**  
This research explores the joint expansion planning of power and water distribution networks, which exhibit interdependence at various levels. We specifically focus on the dependency arising from the power consumption of pumps and develop models to seamlessly integrate new components into existing networks. Subsequently, we formulate the joint expansion planning as a Mixed Integer Nonlinear Program (MINLP). Through the application of this MINLP to a small-scale test network, we demonstrate the advantages of combining expansion planning, including cost savings and reduced redundancy, in comparison to independently expanding power and water distribution networks

{{</citation>}}


### (88/153) Optimization meets Machine Learning: An Exact Algorithm for Semi-Supervised Support Vector Machines (Veronica Piccialli et al., 2023)

{{<citation>}}

Veronica Piccialli, Jan Schwiddessen, Antonio M. Sudoso. (2023)  
**Optimization meets Machine Learning: An Exact Algorithm for Semi-Supervised Support Vector Machines**  

---
Primary Category: math.OC  
Categories: cs-LG, math-OC, math.OC  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.09789v1)  

---


**ABSTRACT**  
Support vector machines (SVMs) are well-studied supervised learning models for binary classification. In many applications, large amounts of samples can be cheaply and easily obtained. What is often a costly and error-prone process is to manually label these instances. Semi-supervised support vector machines (S3VMs) extend the well-known SVM classifiers to the semi-supervised approach, aiming at maximizing the margin between samples in the presence of unlabeled data. By leveraging both labeled and unlabeled data, S3VMs attempt to achieve better accuracy and robustness compared to traditional SVMs. Unfortunately, the resulting optimization problem is non-convex and hence difficult to solve exactly. In this paper, we present a new branch-and-cut approach for S3VMs using semidefinite programming (SDP) relaxations. We apply optimality-based bound tightening to bound the feasible set. Box constraints allow us to include valid inequalities, strengthening the lower bound. The resulting SDP relaxation provides bounds significantly stronger than the ones available in the literature. For the upper bound, instead, we define a local search exploiting the solution of the SDP relaxation. Computational results highlight the efficiency of the algorithm, showing its capability to solve instances with a number of data points 10 times larger than the ones solved in the literature.

{{</citation>}}


## cs.HC (5)



### (89/153) Beyond Empirical Windowing: An Attention-Based Approach for Trust Prediction in Autonomous Vehicles (Minxue Niu et al., 2023)

{{<citation>}}

Minxue Niu, Zhaobo Zheng, Kumar Akash, Teruhisa Misu. (2023)  
**Beyond Empirical Windowing: An Attention-Based Approach for Trust Prediction in Autonomous Vehicles**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-LG, cs.HC  
Keywords: Attention, LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2312.10209v1)  

---


**ABSTRACT**  
Humans' internal states play a key role in human-machine interaction, leading to the rise of human state estimation as a prominent field. Compared to swift state changes such as surprise and irritation, modeling gradual states like trust and satisfaction are further challenged by label sparsity: long time-series signals are usually associated with a single label, making it difficult to identify the critical span of state shifts. Windowing has been one widely-used technique to enable localized analysis of long time-series data. However, the performance of downstream models can be sensitive to the window size, and determining the optimal window size demands domain expertise and extensive search. To address this challenge, we propose a Selective Windowing Attention Network (SWAN), which employs window prompts and masked attention transformation to enable the selection of attended intervals with flexible lengths. We evaluate SWAN on the task of trust prediction on a new multimodal driving simulation dataset. Experiments show that SWAN significantly outperforms an existing empirical window selection baseline and neural network baselines including CNN-LSTM and Transformer. Furthermore, it shows robustness across a wide span of windowing ranges, compared to the traditional windowing approach.

{{</citation>}}


### (90/153) Prompting Datasets: Data Discovery with Conversational Agents (Johanna Walker et al., 2023)

{{<citation>}}

Johanna Walker, Elisavet Koutsiana, Joe Massey, Gefion Theurmer, Elena Simperl. (2023)  
**Prompting Datasets: Data Discovery with Conversational Agents**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09947v1)  

---


**ABSTRACT**  
Can large language models assist in data discovery? Data discovery predominantly happens via search on a data portal or the web, followed by assessment of the dataset to ensure it is fit for the intended purpose. The ability of conversational generative AI (CGAI) to support recommendations with reasoning implies it can suggest datasets to users, explain why it has done so, and provide information akin to documentation regarding the dataset in order to support a use decision. We hold 3 workshops with data users and find that, despite limitations around web capabilities, CGAIs are able to suggest relevant datasets and provide many of the required sensemaking activities, as well as support dataset analysis and manipulation. However, CGAIs may also suggest fictional datasets, and perform inaccurate analysis. We identify emerging practices in data discovery and present a model of these to inform future research directions and data prompt design.

{{</citation>}}


### (91/153) Shaping and Being Shaped by Drones: Supporting Perception-Action Loops (Mousa Sondoqah et al., 2023)

{{<citation>}}

Mousa Sondoqah, Fehmi Ben Abdesslem, Kristina Popova, Moira McGregor, Joseph La Delfa, Rachael Garrett, Airi Lampinen, Luca Mottola, Kristina Höök. (2023)  
**Shaping and Being Shaped by Drones: Supporting Perception-Action Loops**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-RO, cs.HC  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2312.09688v1)  

---


**ABSTRACT**  
We report on a three-day challenge during which five teams each programmed a nanodrone to be piloted through an obstacle course using bodily movement, in a 3D transposition of the '80s video-game Pacman. Using a bricolage approach to analyse interviews, field notes, video recordings, and inspection of each team's code revealed how participants were shaping and, in turn, became shaped in bodily ways by the drones' limitations. We observed how teams adapted to compete by: 1) shifting from aiming for seamless human-drone interaction, to seeing drones as fragile, wilful, and prone to crashes; 2) engaging with intimate, bodily interactions to more precisely understand, probe, and delimit each drone's capabilities; 3) adopting different strategies, emphasising either training the drone or training the pilot. We contribute with an empirical, somaesthetically focused account of current challenges in HDI and call for programming environments that support action-feedback loops for design and programming purposes.

{{</citation>}}


### (92/153) InstructPipe: Building Visual Programming Pipelines with Human Instructions (Zhongyi Zhou et al., 2023)

{{<citation>}}

Zhongyi Zhou, Jing Jin, Vrushank Phadnis, Xiuxiu Yuan, Jun Jiang, Xun Qian, Jingtao Zhou, Yiyi Huang, Zheng Xu, Yinda Zhang, Kristen Wright, Jason Mayes, Mark Sherwood, Johnny Lee, Alex Olwal, David Kim, Ram Iyengar, Na Li, Ruofei Du. (2023)  
**InstructPipe: Building Visual Programming Pipelines with Human Instructions**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09672v1)  

---


**ABSTRACT**  
Visual programming provides beginner-level programmers with a coding-free experience to build their customized pipelines. Existing systems require users to build a pipeline entirely from scratch, implying that novice users need to set up and link appropriate nodes all by themselves, starting from a blank workspace. We present InstructPipe, an AI assistant that enables users to start prototyping machine learning (ML) pipelines with text instructions. We designed two LLM modules and a code interpreter to execute our solution. LLM modules generate pseudocode of a target pipeline, and the interpreter renders a pipeline in the node-graph editor for further human-AI collaboration. Technical evaluations reveal that InstructPipe reduces user interactions by 81.1% compared to traditional methods. Our user study (N=16) showed that InstructPipe empowers novice users to streamline their workflow in creating desired ML pipelines, reduce their learning curve, and spark innovative ideas with open-ended commands.

{{</citation>}}


### (93/153) Exploring Gender Disparities in Bumble's Match Recommendations (Ritvik Aryan Kalra et al., 2023)

{{<citation>}}

Ritvik Aryan Kalra, Pratham Gupta, Ben Varghese, Nimmi Rangaswamy. (2023)  
**Exploring Gender Disparities in Bumble's Match Recommendations**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09626v1)  

---


**ABSTRACT**  
We study bias and discrimination in the context of Bumble, an online dating platform in India. Drawing on research in AI fairness and inclusion studies we analyze algorithmic bias and their propensity to reproduce bias. We conducted an experiment to identify and address the presence of bias in the matching algorithms Bumble pushes to its users in the form of profiles for potential dates in the real world. Dating apps like Bumble utilize algorithms that learn from user data to make recommendations. Even if the algorithm does not have intentions or consciousness, it is a system created and maintained by humans. We attribute moral agency of such systems to be compositely derived from algorithmic mediations, the design and utilization of these platforms. Developers, designers, and operators of dating platforms thus have a moral obligation to mitigate biases in the algorithms to create inclusive platforms that affirm diverse social identities.

{{</citation>}}


## cs.MM (2)



### (94/153) CARAT: Contrastive Feature Reconstruction and Aggregation for Multi-modal Multi-label Emotion Recognition (Cheng Peng et al., 2023)

{{<citation>}}

Cheng Peng, Ke Chen, Lidan Shou, Gang Chen. (2023)  
**CARAT: Contrastive Feature Reconstruction and Aggregation for Multi-modal Multi-label Emotion Recognition**  

---
Primary Category: cs.MM  
Categories: cs-AI, cs-MM, cs.MM  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2312.10201v1)  

---


**ABSTRACT**  
Multi-modal multi-label emotion recognition (MMER) aims to identify relevant emotions from multiple modalities. The challenge of MMER is how to effectively capture discriminative features for multiple labels from heterogeneous data. Recent studies are mainly devoted to exploring various fusion strategies to integrate multi-modal information into a unified representation for all labels. However, such a learning scheme not only overlooks the specificity of each modality but also fails to capture individual discriminative features for different labels. Moreover, dependencies of labels and modalities cannot be effectively modeled. To address these issues, this paper presents ContrAstive feature Reconstruction and AggregaTion (CARAT) for the MMER task. Specifically, we devise a reconstruction-based fusion mechanism to better model fine-grained modality-to-label dependencies by contrastively learning modal-separated and label-specific features. To further exploit the modality complementarity, we introduce a shuffle-based aggregation strategy to enrich co-occurrence collaboration among labels. Experiments on two benchmark datasets CMU-MOSEI and M3ED demonstrate the effectiveness of CARAT over state-of-the-art methods. Code is available at https://github.com/chengzju/CARAT.

{{</citation>}}


### (95/153) MORE: A Multimodal Object-Entity Relation Extraction Dataset with a Benchmark Evaluation (Liang He et al., 2023)

{{<citation>}}

Liang He, Hongke Wang, Yongchang Cao, Zhen Wu, Jianbing Zhang, Xinyu Dai. (2023)  
**MORE: A Multimodal Object-Entity Relation Extraction Dataset with a Benchmark Evaluation**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: NLP, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2312.09753v1)  

---


**ABSTRACT**  
Extracting relational facts from multimodal data is a crucial task in the field of multimedia and knowledge graphs that feeds into widespread real-world applications. The emphasis of recent studies centers on recognizing relational facts in which both entities are present in one modality and supplementary information is used from other modalities. However, such works disregard a substantial amount of multimodal relational facts that arise across different modalities, such as one entity seen in a text and another in an image. In this paper, we propose a new task, namely Multimodal Object-Entity Relation Extraction, which aims to extract "object-entity" relational facts from image and text data. To facilitate research on this task, we introduce MORE, a new dataset comprising 21 relation types and 20,264 multimodal relational facts annotated on 3,559 pairs of textual news titles and corresponding images. To show the challenges of Multimodal Object-Entity Relation Extraction, we evaluated recent state-of-the-art methods for multimodal relation extraction and conducted a comprehensive experimentation analysis on MORE. Our results demonstrate significant challenges for existing methods, underlining the need for further research on this task. Based on our experiments, we identify several promising directions for future research. The MORE dataset and code are available at https://github.com/NJUNLP/MORE.

{{</citation>}}


## cs.LG (26)



### (96/153) Pareto Envelope Augmented with Reinforcement Learning: Multi-objective reinforcement learning-based approach for Large-Scale Constrained Pressurized Water Reactor optimization (Paul Seurin et al., 2023)

{{<citation>}}

Paul Seurin, Koroush Seurin. (2023)  
**Pareto Envelope Augmented with Reinforcement Learning: Multi-objective reinforcement learning-based approach for Large-Scale Constrained Pressurized Water Reactor optimization**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.10194v1)  

---


**ABSTRACT**  
A novel method, the Pareto Envelope Augmented with Reinforcement Learning (PEARL), has been developed to address the challenges posed by multi-objective problems, particularly in the field of engineering where the evaluation of candidate solutions can be time-consuming. PEARL distinguishes itself from traditional policy-based multi-objective Reinforcement Learning methods by learning a single policy, eliminating the need for multiple neural networks to independently solve simpler sub-problems. Several versions inspired from deep learning and evolutionary techniques have been crafted, catering to both unconstrained and constrained problem domains. Curriculum Learning is harnessed to effectively manage constraints in these versions. PEARL's performance is first evaluated on classical multi-objective benchmarks. Additionally, it is tested on two practical PWR core Loading Pattern optimization problems to showcase its real-world applicability. The first problem involves optimizing the Cycle length and the rod-integrated peaking factor as the primary objectives, while the second problem incorporates the mean average enrichment as an additional objective. Furthermore, PEARL addresses three types of constraints related to boron concentration, peak pin burnup, and peak pin power. The results are systematically compared against a conventional approach, the Non-dominated Sorting Genetic Algorithm. Notably, PEARL, specifically the PEARL-NdS variant, efficiently uncovers a Pareto front without necessitating additional efforts from the algorithm designer, as opposed to a single optimization with scaled objectives. It also outperforms the classical approach across multiple performance metrics, including the Hyper-volume.

{{</citation>}}


### (97/153) Coupling Fairness and Pruning in a Single Run: a Bi-level Optimization Perspective (Yucong Dai et al., 2023)

{{<citation>}}

Yucong Dai, Gen Li, Feng Luo, Xiaolong Ma, Yongkai Wu. (2023)  
**Coupling Fairness and Pruning in a Single Run: a Bi-level Optimization Perspective**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.10181v1)  

---


**ABSTRACT**  
Deep neural networks have demonstrated remarkable performance in various tasks. With a growing need for sparse deep learning, model compression techniques, especially pruning, have gained significant attention. However, conventional pruning techniques can inadvertently exacerbate algorithmic bias, resulting in unequal predictions. To address this, we define a fair pruning task where a sparse model is derived subject to fairness requirements. In particular, we propose a framework to jointly optimize the pruning mask and weight update processes with fairness constraints. This framework is engineered to compress models that maintain performance while ensuring fairness in a single execution. To this end, we formulate the fair pruning problem as a novel constrained bi-level optimization task and derive efficient and effective solving strategies. We design experiments spanning various datasets and settings to validate our proposed method. Our empirical analysis contrasts our framework with several mainstream pruning strategies, emphasizing our method's superiority in maintaining model fairness, performance, and efficiency.

{{</citation>}}


### (98/153) Accelerating Neural Network Training: A Brief Review (Sahil Nokhwal et al., 2023)

{{<citation>}}

Sahil Nokhwal, Priyanka Chilakalapudi, Preeti Donekal, Manoj Chandrasekharan, Suman Nokhwal, Ram Swaroop, Raj Bala, Saurabh Pahune, Ankit Chaudhary. (2023)  
**Accelerating Neural Network Training: A Brief Review**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.10024v1)  

---


**ABSTRACT**  
The process of training a deep neural network is characterized by significant time requirements and associated costs. Although researchers have made considerable progress in this area, further work is still required due to resource constraints. This study examines innovative approaches to expedite the training process of deep neural networks (DNN), with specific emphasis on three state-of-the-art models such as ResNet50, Vision Transformer (ViT), and EfficientNet. The research utilizes sophisticated methodologies, including Gradient Accumulation (GA), Automatic Mixed Precision (AMP), and Pin Memory (PM), in order to optimize performance and accelerate the training procedure.   The study examines the effects of these methodologies on the DNN models discussed earlier, assessing their efficacy with regard to training rate and computational efficacy. The study showcases the efficacy of including GA as a strategic approach, resulting in a noteworthy decrease in the duration required for training. This enables the models to converge at a faster pace. The utilization of AMP enhances the speed of computations by taking advantage of the advantages offered by lower precision arithmetic while maintaining the correctness of the model.   Furthermore, this study investigates the application of Pin Memory as a strategy to enhance the efficiency of data transmission between the central processing unit and the graphics processing unit, thereby offering a promising opportunity for enhancing overall performance. The experimental findings demonstrate that the combination of these sophisticated methodologies significantly accelerates the training of DNNs, offering vital insights for experts seeking to improve the effectiveness of deep learning processes.

{{</citation>}}


### (99/153) Toward Computationally Efficient Inverse Reinforcement Learning via Reward Shaping (Lauren H. Cooke et al., 2023)

{{<citation>}}

Lauren H. Cooke, Harvey Klyne, Edwin Zhang, Cassidy Laidlaw, Milind Tambe, Finale Doshi-Velez. (2023)  
**Toward Computationally Efficient Inverse Reinforcement Learning via Reward Shaping**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09983v2)  

---


**ABSTRACT**  
Inverse reinforcement learning (IRL) is computationally challenging, with common approaches requiring the solution of multiple reinforcement learning (RL) sub-problems. This work motivates the use of potential-based reward shaping to reduce the computational burden of each RL sub-problem. This work serves as a proof-of-concept and we hope will inspire future developments towards computationally efficient IRL.

{{</citation>}}


### (100/153) GreenLightningAI: An Efficient AI System with Decoupled Structural and Quantitative Knowledge (Jose Duato et al., 2023)

{{<citation>}}

Jose Duato, Jose I. Mestre, Manuel F. Dolz, Enrique S. Quintana-Ortí. (2023)  
**GreenLightningAI: An Efficient AI System with Decoupled Structural and Quantitative Knowledge**  

---
Primary Category: cs.LG  
Categories: I-2; I-2-6, cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09971v1)  

---


**ABSTRACT**  
The number and complexity of artificial intelligence (AI) applications is growing relentlessly. As a result, even with the many algorithmic and mathematical advances experienced over past decades as well as the impressive energy efficiency and computational capacity of current hardware accelerators, training the most powerful and popular deep neural networks comes at very high economic and environmental costs. Recognising that additional optimisations of conventional neural network training is very difficult, this work takes a radically different approach by proposing GreenLightningAI, a new AI system design consisting of a linear model that is capable of emulating the behaviour of deep neural networks by subsetting the model for each particular sample. The new AI system stores the information required to select the system subset for a given sample (referred to as structural information) separately from the linear model parameters (referred to as quantitative knowledge). In this paper we present a proof of concept, showing that the structural information stabilises far earlier than the quantitative knowledge. Additionally, we show experimentally that the structural information can be kept unmodified when re-training the AI system with new samples while still achieving a validation accuracy similar to that obtained when re-training a neural network with similar size. Since the proposed AI system is based on a linear model, multiple copies of the model, trained with different datasets, can be easily combined. This enables faster and greener (re)-training algorithms, including incremental re-training and federated incremental re-training.

{{</citation>}}


### (101/153) Peer Learning: Learning Complex Policies in Groups from Scratch via Action Recommendations (Cedric Derstroff et al., 2023)

{{<citation>}}

Cedric Derstroff, Mattia Cerrato, Jannis Brugger, Jan Peters, Stefan Kramer. (2023)  
**Peer Learning: Learning Complex Policies in Groups from Scratch via Action Recommendations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09950v1)  

---


**ABSTRACT**  
Peer learning is a novel high-level reinforcement learning framework for agents learning in groups. While standard reinforcement learning trains an individual agent in trial-and-error fashion, all on its own, peer learning addresses a related setting in which a group of agents, i.e., peers, learns to master a task simultaneously together from scratch. Peers are allowed to communicate only about their own states and actions recommended by others: "What would you do in my situation?". Our motivation is to study the learning behavior of these agents. We formalize the teacher selection process in the action advice setting as a multi-armed bandit problem and therefore highlight the need for exploration. Eventually, we analyze the learning behavior of the peers and observe their ability to rank the agents' performance within the study group and understand which agents give reliable advice. Further, we compare peer learning with single agent learning and a state-of-the-art action advice baseline. We show that peer learning is able to outperform single-agent learning and the baseline in several challenging discrete and continuous OpenAI Gym domains. Doing so, we also show that within such a framework complex policies from action recommendations beyond discrete action spaces can evolve.

{{</citation>}}


### (102/153) Sketch and shift: a robust decoder for compressive clustering (Ayoub Belhadji et al., 2023)

{{<citation>}}

Ayoub Belhadji, Rémi Gribonval. (2023)  
**Sketch and shift: a robust decoder for compressive clustering**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2312.09940v1)  

---


**ABSTRACT**  
Compressive learning is an emerging approach to drastically reduce the memory footprint of large-scale learning, by first summarizing a large dataset into a low-dimensional sketch vector, and then decoding from this sketch the latent information needed for learning. In light of recent progress on information preservation guarantees for sketches based on random features, a major objective is to design easy-to-tune algorithms (called decoders) to robustly and efficiently extract this information. To address the underlying non-convex optimization problems, various heuristics have been proposed. In the case of compressive clustering, the standard heuristic is CL-OMPR, a variant of sliding Frank-Wolfe. Yet, CL-OMPR is hard to tune, and the examination of its robustness was overlooked. In this work, we undertake a scrutinized examination of CL-OMPR to circumvent its limitations. In particular, we show how this algorithm can fail to recover the clusters even in advantageous scenarios. To gain insight, we show how the deficiencies of this algorithm can be attributed to optimization difficulties related to the structure of a correlation function appearing at core steps of the algorithm. To address these limitations, we propose an alternative decoder offering substantial improvements over CL-OMPR. Its design is notably inspired from the mean shift algorithm, a classic approach to detect the local maxima of kernel density estimators. The proposed algorithm can extract clustering information from a sketch of the MNIST dataset that is 10 times smaller than previously.

{{</citation>}}


### (103/153) Assume-Guarantee Reinforcement Learning (Milad Kazemi et al., 2023)

{{<citation>}}

Milad Kazemi, Mateo Perez, Fabio Somenzi, Sadegh Soudjani, Ashutosh Trivedi, Alvaro Velasquez. (2023)  
**Assume-Guarantee Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09938v1)  

---


**ABSTRACT**  
We present a modular approach to \emph{reinforcement learning} (RL) in environments consisting of simpler components evolving in parallel. A monolithic view of such modular environments may be prohibitively large to learn, or may require unrealizable communication between the components in the form of a centralized controller. Our proposed approach is based on the assume-guarantee paradigm where the optimal control for the individual components is synthesized in isolation by making \emph{assumptions} about the behaviors of neighboring components, and providing \emph{guarantees} about their own behavior. We express these \emph{assume-guarantee contracts} as regular languages and provide automatic translations to scalar rewards to be used in RL. By combining local probabilities of satisfaction for each component, we provide a lower bound on the probability of satisfaction of the complete system. By solving a Markov game for each component, RL can produce a controller for each component that maximizes this lower bound. The controller utilizes the information it receives through communication, observations, and any knowledge of a coarse model of other agents. We experimentally demonstrate the efficiency of the proposed approach on a variety of case studies.

{{</citation>}}


### (104/153) ChemTime: Rapid and Early Classification for Multivariate Time Series Classification of Chemical Sensors (Alexander M. Moore et al., 2023)

{{<citation>}}

Alexander M. Moore, Randy C. Paffenroth, Kenneth T. Ngo, Joshua R. Uzarski. (2023)  
**ChemTime: Rapid and Early Classification for Multivariate Time Series Classification of Chemical Sensors**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-QM  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.09871v1)  

---


**ABSTRACT**  
Multivariate time series data are ubiquitous in the application of machine learning to problems in the physical sciences. Chemiresistive sensor arrays are highly promising in chemical detection tasks relevant to industrial, safety, and military applications. Sensor arrays are an inherently multivariate time series data collection tool which demand rapid and accurate classification of arbitrary chemical analytes. Previous research has benchmarked data-agnostic multivariate time series classifiers across diverse multivariate time series supervised tasks in order to find general-purpose classification algorithms. To our knowledge, there has yet to be an effort to survey machine learning and time series classification approaches to chemiresistive hardware sensor arrays for the detection of chemical analytes. In addition to benchmarking existing approaches to multivariate time series classifiers, we incorporate findings from a model survey to propose the novel \textit{ChemTime} approach to sensor array classification for chemical sensing. We design experiments addressing the unique challenges of hardware sensor arrays classification including the rapid classification ability of classifiers and minimization of inference time while maintaining performance for deployed lightweight hardware sensing devices. We find that \textit{ChemTime} is uniquely positioned for the chemical sensing task by combining rapid and early classification of time series with beneficial inference and high accuracy.

{{</citation>}}


### (105/153) Automating reward function configuration for drug design (Marius Urbonas et al., 2023)

{{<citation>}}

Marius Urbonas, Temitope Ajileye, Paul Gainer, Douglas Pires. (2023)  
**Automating reward function configuration for drug design**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09865v1)  

---


**ABSTRACT**  
Designing reward functions that guide generative molecular design (GMD) algorithms to desirable areas of chemical space is of critical importance in AI-driven drug discovery. Traditionally, this has been a manual and error-prone task; the selection of appropriate computational methods to approximate biological assays is challenging and the aggregation of computed values into a single score even more so, leading to potential reliance on trial-and-error approaches. We propose a novel approach for automated reward configuration that relies solely on experimental data, mitigating the challenges of manual reward adjustment on drug discovery projects. Our method achieves this by constructing a ranking over experimental data based on Pareto dominance over the multi-objective space, then training a neural network to approximate the reward function such that rankings determined by the predicted reward correlate with those determined by the Pareto dominance relation. We validate our method using two case studies. In the first study we simulate Design-Make-Test-Analyse (DMTA) cycles by alternating reward function updates and generative runs guided by that function. We show that the learned function adapts over time to yield compounds that score highly with respect to evaluation functions taken from the literature. In the second study we apply our algorithm to historical data from four real drug discovery projects. We show that our algorithm yields reward functions that outperform the predictive accuracy of human-defined functions, achieving an improvement of up to 0.4 in Spearman's correlation against a ground truth evaluation function that encodes the target drug profile for that project. Our method provides an efficient data-driven way to configure reward functions for GMD, and serves as a strong baseline for future research into transformative approaches for the automation of drug discovery.

{{</citation>}}


### (106/153) Deep Unsupervised Domain Adaptation for Time Series Classification: a Benchmark (Hassan Ismail Fawaz et al., 2023)

{{<citation>}}

Hassan Ismail Fawaz, Ganesh Del Grosso, Tanguy Kerdoncuff, Aurelie Boisbunon, Illyyne Saffar. (2023)  
**Deep Unsupervised Domain Adaptation for Time Series Classification: a Benchmark**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.09857v2)  

---


**ABSTRACT**  
Unsupervised Domain Adaptation (UDA) aims to harness labeled source data to train models for unlabeled target data. Despite extensive research in domains like computer vision and natural language processing, UDA remains underexplored for time series data, which has widespread real-world applications ranging from medicine and manufacturing to earth observation and human activity recognition. Our paper addresses this gap by introducing a comprehensive benchmark for evaluating UDA techniques for time series classification, with a focus on deep learning methods. We provide seven new benchmark datasets covering various domain shifts and temporal dynamics, facilitating fair and standardized UDA method assessments with state of the art neural network backbones (e.g. Inception) for time series data. This benchmark offers insights into the strengths and limitations of the evaluated approaches while preserving the unsupervised nature of domain adaptation, making it directly applicable to practical problems. Our paper serves as a vital resource for researchers and practitioners, advancing domain adaptation solutions for time series data and fostering innovation in this critical field. The implementation code of this benchmark is available at https://github.com/EricssonResearch/UDA-4-TSC.

{{</citation>}}


### (107/153) Small Dataset, Big Gains: Enhancing Reinforcement Learning by Offline Pre-Training with Model Based Augmentation (Girolamo Macaluso et al., 2023)

{{<citation>}}

Girolamo Macaluso, Alessandro Sestini, Andrew D. Bagdanov. (2023)  
**Small Dataset, Big Gains: Enhancing Reinforcement Learning by Offline Pre-Training with Model Based Augmentation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Augmentation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09844v1)  

---


**ABSTRACT**  
Offline reinforcement learning leverages pre-collected datasets of transitions to train policies. It can serve as effective initialization for online algorithms, enhancing sample efficiency and speeding up convergence. However, when such datasets are limited in size and quality, offline pre-training can produce sub-optimal policies and lead to degraded online reinforcement learning performance. In this paper we propose a model-based data augmentation strategy to maximize the benefits of offline reinforcement learning pre-training and reduce the scale of data needed to be effective. Our approach leverages a world model of the environment trained on the offline dataset to augment states during offline pre-training. We evaluate our approach on a variety of MuJoCo robotic tasks and our results show it can jump-start online fine-tuning and substantially reduce - in some cases by an order of magnitude - the required number of environment interactions.

{{</citation>}}


### (108/153) Fragility, Robustness and Antifragility in Deep Learning (Chandresh Pravin et al., 2023)

{{<citation>}}

Chandresh Pravin, Ivan Martino, Giuseppe Nicosia, Varun Ojha. (2023)  
**Fragility, Robustness and Antifragility in Deep Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.09821v1)  

---


**ABSTRACT**  
We propose a systematic analysis of deep neural networks (DNNs) based on a signal processing technique for network parameter removal, in the form of synaptic filters that identifies the fragility, robustness and antifragility characteristics of DNN parameters. Our proposed analysis investigates if the DNN performance is impacted negatively, invariantly, or positively on both clean and adversarially perturbed test datasets when the DNN undergoes synaptic filtering. We define three \textit{filtering scores} for quantifying the fragility, robustness and antifragility characteristics of DNN parameters based on the performances for (i) clean dataset, (ii) adversarial dataset, and (iii) the difference in performances of clean and adversarial datasets. We validate the proposed systematic analysis on ResNet-18, ResNet-50, SqueezeNet-v1.1 and ShuffleNet V2 x1.0 network architectures for MNIST, CIFAR10 and Tiny ImageNet datasets. The filtering scores, for a given network architecture, identify network parameters that are invariant in characteristics across different datasets over learning epochs. Vice-versa, for a given dataset, the filtering scores identify the parameters that are invariant in characteristics across different network architectures. We show that our synaptic filtering method improves the test accuracy of ResNet and ShuffleNet models on adversarial datasets when only the robust and antifragile parameters are selectively retrained at any given epoch, thus demonstrating applications of the proposed strategy in improving model robustness.

{{</citation>}}


### (109/153) Concept Prerequisite Relation Prediction by Using Permutation-Equivariant Directed Graph Neural Networks (Xiran Qu et al., 2023)

{{<citation>}}

Xiran Qu, Xuequn Shang, Yupei Zhang. (2023)  
**Concept Prerequisite Relation Prediction by Using Permutation-Equivariant Directed Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: 68T07, I-2-6, cs-AI, cs-LG, cs.LG  
Keywords: AI, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.09802v1)  

---


**ABSTRACT**  
This paper studies the problem of CPRP, concept prerequisite relation prediction, which is a fundamental task in using AI for education. CPRP is usually formulated into a link-prediction task on a relationship graph of concepts and solved by training the graph neural network (GNN) model. However, current directed GNNs fail to manage graph isomorphism which refers to the invariance of non-isomorphic graphs, reducing the expressivity of resulting representations. We present a permutation-equivariant directed GNN model by introducing the Weisfeiler-Lehman test into directed GNN learning. Our method is then used for CPRP and evaluated on three public datasets. The experimental results show that our model delivers better prediction performance than the state-of-the-art methods.

{{</citation>}}


### (110/153) Keep the Faith: Faithful Explanations in Convolutional Neural Networks for Case-Based Reasoning (Tom Nuno Wolf et al., 2023)

{{<citation>}}

Tom Nuno Wolf, Fabian Bongratz, Anne-Marie Rickmann, Sebastian Pölsterl, Christian Wachinger. (2023)  
**Keep the Faith: Faithful Explanations in Convolutional Neural Networks for Case-Based Reasoning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.09783v2)  

---


**ABSTRACT**  
Explaining predictions of black-box neural networks is crucial when applied to decision-critical tasks. Thus, attribution maps are commonly used to identify important image regions, despite prior work showing that humans prefer explanations based on similar examples. To this end, ProtoPNet learns a set of class-representative feature vectors (prototypes) for case-based reasoning. During inference, similarities of latent features to prototypes are linearly classified to form predictions and attribution maps are provided to explain the similarity. In this work, we evaluate whether architectures for case-based reasoning fulfill established axioms required for faithful explanations using the example of ProtoPNet. We show that such architectures allow the extraction of faithful explanations. However, we prove that the attribution maps used to explain the similarities violate the axioms. We propose a new procedure to extract explanations for trained ProtoPNets, named ProtoPFaith. Conceptually, these explanations are Shapley values, calculated on the similarity scores of each prototype. They allow to faithfully answer which prototypes are present in an unseen image and quantify each pixel's contribution to that presence, thereby complying with all axioms. The theoretical violations of ProtoPNet manifest in our experiments on three datasets (CUB-200-2011, Stanford Dogs, RSNA) and five architectures (ConvNet, ResNet, ResNet50, WideResNet50, ResNeXt50). Our experiments show a qualitative difference between the explanations given by ProtoPNet and ProtoPFaith. Additionally, we quantify the explanations with the Area Over the Perturbation Curve, on which ProtoPFaith outperforms ProtoPNet on all experiments by a factor $>10^3$.

{{</citation>}}


### (111/153) Celestial Machine Learning: From Data to Mars and Beyond with AI Feynman (Zi-Yu Khoo et al., 2023)

{{<citation>}}

Zi-Yu Khoo, Abel Yang, Jonathan Sze Choong Low, Stéphane Bressan. (2023)  
**Celestial Machine Learning: From Data to Mars and Beyond with AI Feynman**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09766v1)  

---


**ABSTRACT**  
Can a machine or algorithm discover or learn Kepler's first law from astronomical sightings alone? We emulate Johannes Kepler's discovery of the equation of the orbit of Mars with the Rudolphine tables using AI Feynman, a physics-inspired tool for symbolic regression.

{{</citation>}}


### (112/153) Bridging the Semantic-Numerical Gap: A Numerical Reasoning Method of Cross-modal Knowledge Graph for Material Property Prediction (Guangxuan Song et al., 2023)

{{<citation>}}

Guangxuan Song, Dongmei Fu, Zhongwei Qiu, Zijiang Yang, Jiaxin Dai, Lingwei Ma, Dawei Zhang. (2023)  
**Bridging the Semantic-Numerical Gap: A Numerical Reasoning Method of Cross-modal Knowledge Graph for Material Property Prediction**  

---
Primary Category: cs.LG  
Categories: cond-mat-mtrl-sci, cs-LG, cs.LG  
Keywords: AI, Knowledge Graph, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.09744v1)  

---


**ABSTRACT**  
Using machine learning (ML) techniques to predict material properties is a crucial research topic. These properties depend on numerical data and semantic factors. Due to the limitations of small-sample datasets, existing methods typically adopt ML algorithms to regress numerical properties or transfer other pre-trained knowledge graphs (KGs) to the material. However, these methods cannot simultaneously handle semantic and numerical information. In this paper, we propose a numerical reasoning method for material KGs (NR-KG), which constructs a cross-modal KG using semantic nodes and numerical proxy nodes. It captures both types of information by projecting KG into a canonical KG and utilizes a graph neural network to predict material properties. In this process, a novel projection prediction loss is proposed to extract semantic features from numerical information. NR-KG facilitates end-to-end processing of cross-modal data, mining relationships and cross-modal information in small-sample datasets, and fully utilizes valuable experimental data to enhance material prediction. We further propose two new High-Entropy Alloys (HEA) property datasets with semantic descriptions. NR-KG outperforms state-of-the-art (SOTA) methods, achieving relative improvements of 25.9% and 16.1% on two material datasets. Besides, NR-KG surpasses SOTA methods on two public physical chemistry molecular datasets, showing improvements of 22.2% and 54.3%, highlighting its potential application and generalizability. We hope the proposed datasets, algorithms, and pre-trained models can facilitate the communities of KG and AI for materials.

{{</citation>}}


### (113/153) PELP: Pioneer Event Log Prediction Using Sequence-to-Sequence Neural Networks (Wenjun Zhou et al., 2023)

{{<citation>}}

Wenjun Zhou, Artem Polyvyanyy, James Bailey. (2023)  
**PELP: Pioneer Event Log Prediction Using Sequence-to-Sequence Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SE, cs.LG  
Keywords: Sequence-to-Sequence  
[Paper Link](http://arxiv.org/abs/2312.09741v1)  

---


**ABSTRACT**  
Process mining, a data-driven approach for analyzing, visualizing, and improving business processes using event logs, has emerged as a powerful technique in the field of business process management. Process forecasting is a sub-field of process mining that studies how to predict future processes and process models. In this paper, we introduce and motivate the problem of event log prediction and present our approach to solving the event log prediction problem, in particular, using the sequence-to-sequence deep learning approach. We evaluate and analyze the prediction outcomes on a variety of synthetic logs and seven real-life logs and show that our approach can generate perfect predictions on synthetic logs and that deep learning techniques have the potential to be applied in real-world event log prediction tasks. We further provide practical recommendations for event log predictions grounded in the outcomes of the conducted experiments.

{{</citation>}}


### (114/153) GraphRARE: Reinforcement Learning Enhanced Graph Neural Network with Relative Entropy (Tianhao Peng et al., 2023)

{{<citation>}}

Tianhao Peng, Wenjun Wu, Haitao Yuan, Zhifeng Bao, Zhao Pengrui, Xin Yu, Xuetao Lin, Yu Liang, Yanjun Pu. (2023)  
**GraphRARE: Reinforcement Learning Enhanced Graph Neural Network with Relative Entropy**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09708v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have shown advantages in graph-based analysis tasks. However, most existing methods have the homogeneity assumption and show poor performance on heterophilic graphs, where the linked nodes have dissimilar features and different class labels, and the semantically related nodes might be multi-hop away. To address this limitation, this paper presents GraphRARE, a general framework built upon node relative entropy and deep reinforcement learning, to strengthen the expressive capability of GNNs. An innovative node relative entropy, which considers node features and structural similarity, is used to measure mutual information between node pairs. In addition, to avoid the sub-optimal solutions caused by mixing useful information and noises of remote nodes, a deep reinforcement learning-based algorithm is developed to optimize the graph topology. This algorithm selects informative nodes and discards noisy nodes based on the defined node relative entropy. Extensive experiments are conducted on seven real-world datasets. The experimental results demonstrate the superiority of GraphRARE in node classification and its capability to optimize the original graph topology.

{{</citation>}}


### (115/153) Bayesian Estimate of Mean Proper Scores for Diversity-Enhanced Active Learning (Wei Tan et al., 2023)

{{<citation>}}

Wei Tan, Lan Du, Wray Buntine. (2023)  
**Bayesian Estimate of Mean Proper Scores for Diversity-Enhanced Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2312.10116v1)  

---


**ABSTRACT**  
The effectiveness of active learning largely depends on the sampling efficiency of the acquisition function. Expected Loss Reduction (ELR) focuses on a Bayesian estimate of the reduction in classification error, and more general costs fit in the same framework. We propose Bayesian Estimate of Mean Proper Scores (BEMPS) to estimate the increase in strictly proper scores such as log probability or negative mean square error within this framework. We also prove convergence results for this general class of costs. To facilitate better experimentation with the new acquisition functions, we develop a complementary batch AL algorithm that encourages diversity in the vector of expected changes in scores for unlabeled data. To allow high-performance classifiers, we combine deep ensembles, and dynamic validation set construction on pretrained models, and further speed up the ensemble process with the idea of Monte Carlo Dropout. Extensive experiments on both texts and images show that the use of mean square error and log probability with BEMPS yields robust acquisition functions and well-calibrated classifiers, and consistently outperforms the others tested. The advantages of BEMPS over the others are further supported by a set of qualitative analyses, where we visualise their sampling behaviour using data maps and t-SNE plots.

{{</citation>}}


### (116/153) Urban Region Embedding via Multi-View Contrastive Prediction (Zechen Li et al., 2023)

{{<citation>}}

Zechen Li, Weiming Huang, Kai Zhao, Min Yang, Yongshun Gong, Meng Chen. (2023)  
**Urban Region Embedding via Multi-View Contrastive Prediction**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-DB, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.09681v1)  

---


**ABSTRACT**  
Recently, learning urban region representations utilizing multi-modal data (information views) has become increasingly popular, for deep understanding of the distributions of various socioeconomic features in cities. However, previous methods usually blend multi-view information in a posteriors stage, falling short in learning coherent and consistent representations across different views. In this paper, we form a new pipeline to learn consistent representations across varying views, and propose the multi-view Contrastive Prediction model for urban Region embedding (ReCP), which leverages the multiple information views from point-of-interest (POI) and human mobility data. Specifically, ReCP comprises two major modules, namely an intra-view learning module utilizing contrastive learning and feature reconstruction to capture the unique information from each single view, and inter-view learning module that perceives the consistency between the two views using a contrastive prediction learning scheme. We conduct thorough experiments on two downstream tasks to assess the proposed model, i.e., land use clustering and region popularity prediction. The experimental results demonstrate that our model outperforms state-of-the-art baseline methods significantly in urban region representation learning.

{{</citation>}}


### (117/153) Rethinking Causal Relationships Learning in Graph Neural Networks (Hang Gao et al., 2023)

{{<citation>}}

Hang Gao, Chengyu Yao, Jiangmeng Li, Lingyu Si, Yifan Jin, Fengge Wu, Changwen Zheng, Huaping Liu. (2023)  
**Rethinking Causal Relationships Learning in Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.09613v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) demonstrate their significance by effectively modeling complex interrelationships within graph-structured data. To enhance the credibility and robustness of GNNs, it becomes exceptionally crucial to bolster their ability to capture causal relationships. However, despite recent advancements that have indeed strengthened GNNs from a causal learning perspective, conducting an in-depth analysis specifically targeting the causal modeling prowess of GNNs remains an unresolved issue. In order to comprehensively analyze various GNN models from a causal learning perspective, we constructed an artificially synthesized dataset with known and controllable causal relationships between data and labels. The rationality of the generated data is further ensured through theoretical foundations. Drawing insights from analyses conducted using our dataset, we introduce a lightweight and highly adaptable GNN module designed to strengthen GNNs' causal learning capabilities across a diverse range of tasks. Through a series of experiments conducted on both synthetic datasets and other real-world datasets, we empirically validate the effectiveness of the proposed module.

{{</citation>}}


### (118/153) STEAM & MoSAFE: SOTIF Error-and-Failure Model & Analysis for AI-Enabled Driving Automation (Krzysztof Czarnecki et al., 2023)

{{<citation>}}

Krzysztof Czarnecki, Hiroshi Kuwajima. (2023)  
**STEAM & MoSAFE: SOTIF Error-and-Failure Model & Analysis for AI-Enabled Driving Automation**  

---
Primary Category: cs.LG  
Categories: I-2-0, cs-LG, cs-SE, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09559v1)  

---


**ABSTRACT**  
Driving Automation Systems (DAS) are subject to complex road environments and vehicle behaviors and increasingly rely on sophisticated sensors and Artificial Intelligence (AI). These properties give rise to unique safety faults stemming from specification insufficiencies and technological performance limitations, where sensors and AI introduce errors that vary in magnitude and temporal patterns, posing potential safety risks. The Safety of the Intended Functionality (SOTIF) standard emerges as a promising framework for addressing these concerns, focusing on scenario-based analysis to identify hazardous behaviors and their causes. Although the current standard provides a basic cause-and-effect model and high-level process guidance, it lacks concepts required to identify and evaluate hazardous errors, especially within the context of AI.   This paper introduces two key contributions to bridge this gap. First, it defines the SOTIF Temporal Error and Failure Model (STEAM) as a refinement of the SOTIF cause-and-effect model, offering a comprehensive system-design perspective. STEAM refines error definitions, introduces error sequences, and classifies them as error sequence patterns, providing particular relevance to systems employing advanced sensors and AI. Second, this paper proposes the Model-based SOTIF Analysis of Failures and Errors (MoSAFE) method, which allows instantiating STEAM based on system-design models by deriving hazardous error sequence patterns at module level from hazardous behaviors at vehicle level via weakest precondition reasoning. Finally, the paper presents a case study centered on an automated speed-control feature, illustrating the practical applicability of the refined model and the MoSAFE method in addressing complex safety challenges in DAS.

{{</citation>}}


### (119/153) Adversarial Robustness on Image Classification with $k$-means (Rollin Omari et al., 2023)

{{<citation>}}

Rollin Omari, Junae Kim, Paul Montague. (2023)  
**Adversarial Robustness on Image Classification with $k$-means**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs-NE, cs.LG  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2312.09533v1)  

---


**ABSTRACT**  
In this paper we explore the challenges and strategies for enhancing the robustness of $k$-means clustering algorithms against adversarial manipulations. We evaluate the vulnerability of clustering algorithms to adversarial attacks, emphasising the associated security risks. Our study investigates the impact of incremental attack strength on training, introduces the concept of transferability between supervised and unsupervised models, and highlights the sensitivity of unsupervised models to sample distributions. We additionally introduce and evaluate an adversarial training method that improves testing performance in adversarial scenarios, and we highlight the importance of various parameters in the proposed training method, such as continuous learning, centroid initialisation, and adversarial step-count.

{{</citation>}}


### (120/153) Entropy Causal Graphs for Multivariate Time Series Anomaly Detection (Falih Gozi Febrinanto et al., 2023)

{{<citation>}}

Falih Gozi Febrinanto, Kristen Moore, Chandra Thapa, Mujie Liu, Vidya Saikrishna, Jiangang Ma, Feng Xia. (2023)  
**Entropy Causal Graphs for Multivariate Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2312.09478v1)  

---


**ABSTRACT**  
Many multivariate time series anomaly detection frameworks have been proposed and widely applied. However, most of these frameworks do not consider intrinsic relationships between variables in multivariate time series data, thus ignoring the causal relationship among variables and degrading anomaly detection performance. This work proposes a novel framework called CGAD, an entropy Causal Graph for multivariate time series Anomaly Detection. CGAD utilizes transfer entropy to construct graph structures that unveil the underlying causal relationships among time series data. Weighted graph convolutional networks combined with causal convolutions are employed to model both the causal graph structures and the temporal patterns within multivariate time series data. Furthermore, CGAD applies anomaly scoring, leveraging median absolute deviation-based normalization to improve the robustness of the anomaly identification process. Extensive experiments demonstrate that CGAD outperforms state-of-the-art methods on real-world datasets with a 15% average improvement based on three different multivariate time series anomaly detection metrics.

{{</citation>}}


### (121/153) OTOv3: Automatic Architecture-Agnostic Neural Network Training and Compression from Structured Pruning to Erasing Operators (Tianyi Chen et al., 2023)

{{<citation>}}

Tianyi Chen, Tianyu Ding, Zhihui Zhu, Zeyu Chen, HsiangTao Wu, Ilya Zharkov, Luming Liang. (2023)  
**OTOv3: Automatic Architecture-Agnostic Neural Network Training and Compression from Structured Pruning to Erasing Operators**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.09411v1)  

---


**ABSTRACT**  
Compressing a predefined deep neural network (DNN) into a compact sub-network with competitive performance is crucial in the efficient machine learning realm. This topic spans various techniques, from structured pruning to neural architecture search, encompassing both pruning and erasing operators perspectives. Despite advancements, existing methods suffers from complex, multi-stage processes that demand substantial engineering and domain knowledge, limiting their broader applications. We introduce the third-generation Only-Train-Once (OTOv3), which first automatically trains and compresses a general DNN through pruning and erasing operations, creating a compact and competitive sub-network without the need of fine-tuning. OTOv3 simplifies and automates the training and compression process, minimizes the engineering efforts required from users. It offers key technological advancements: (i) automatic search space construction for general DNNs based on dependency graph analysis; (ii) Dual Half-Space Projected Gradient (DHSPG) and its enhanced version with hierarchical search (H2SPG) to reliably solve (hierarchical) structured sparsity problems and ensure sub-network validity; and (iii) automated sub-network construction using solutions from DHSPG/H2SPG and dependency graphs. Our empirical results demonstrate the efficacy of OTOv3 across various benchmarks in structured pruning and neural architecture search. OTOv3 produces sub-networks that match or exceed the state-of-the-arts. The source code will be available at https://github.com/tianyic/only_train_once.

{{</citation>}}


## eess.SP (2)



### (122/153) TSRNet: Simple Framework for Real-time ECG Anomaly Detection with Multimodal Time and Spectrogram Restoration Network (Nhat-Tan Bui et al., 2023)

{{<citation>}}

Nhat-Tan Bui, Dinh-Hieu Hoang, Thinh Phan, Minh-Triet Tran, Brijesh Patel, Donald Adjeroh, Ngan Le. (2023)  
**TSRNet: Simple Framework for Real-time ECG Anomaly Detection with Multimodal Time and Spectrogram Restoration Network**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: AI, Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.10187v1)  

---


**ABSTRACT**  
The electrocardiogram (ECG) is a valuable signal used to assess various aspects of heart health, such as heart rate and rhythm. It plays a crucial role in identifying cardiac conditions and detecting anomalies in ECG data. However, distinguishing between normal and abnormal ECG signals can be a challenging task. In this paper, we propose an approach that leverages anomaly detection to identify unhealthy conditions using solely normal ECG data for training. Furthermore, to enhance the information available and build a robust system, we suggest considering both the time series and time-frequency domain aspects of the ECG signal. As a result, we introduce a specialized network called the Multimodal Time and Spectrogram Restoration Network (TSRNet) designed specifically for detecting anomalies in ECG signals. TSRNet falls into the category of restoration-based anomaly detection and draws inspiration from both the time series and spectrogram domains. By extracting representations from both domains, TSRNet effectively captures the comprehensive characteristics of the ECG signal. This approach enables the network to learn robust representations with superior discrimination abilities, allowing it to distinguish between normal and abnormal ECG patterns more effectively. Furthermore, we introduce a novel inference method, termed Peak-based Error, that specifically focuses on ECG peaks, a critical component in detecting abnormalities. The experimental result on the large-scale dataset PTB-XL has demonstrated the effectiveness of our approach in ECG anomaly detection, while also prioritizing efficiency by minimizing the number of trainable parameters. Our code is available at https://github.com/UARK-AICV/TSRNet.

{{</citation>}}


### (123/153) Deep Reinforcement Learning for Joint Cruise Control and Intelligent Data Acquisition in UAVs-Assisted Sensor Networks (Yousef Emami, 2023)

{{<citation>}}

Yousef Emami. (2023)  
**Deep Reinforcement Learning for Joint Cruise Control and Intelligent Data Acquisition in UAVs-Assisted Sensor Networks**  

---
Primary Category: eess.SP  
Categories: cs-AI, cs-NI, eess-SP, eess.SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09953v1)  

---


**ABSTRACT**  
Unmanned aerial vehicle (UAV)-assisted sensor networks (UASNets), which play a crucial role in creating new opportunities, are experiencing significant growth in civil applications worldwide. UASNets improve disaster management through timely surveillance and advance precision agriculture with detailed crop monitoring, thereby significantly transforming the commercial economy. UASNets revolutionize the commercial sector by offering greater efficiency, safety, and cost-effectiveness, highlighting their transformative impact. A fundamental aspect of these new capabilities and changes is the collection of data from rugged and remote areas. Due to their excellent mobility and maneuverability, UAVs are employed to collect data from ground sensors in harsh environments, such as natural disaster monitoring, border surveillance, and emergency response monitoring. One major challenge in these scenarios is that the movements of UAVs affect channel conditions and result in packet loss. Fast movements of UAVs lead to poor channel conditions and rapid signal degradation, resulting in packet loss. On the other hand, slow mobility of a UAV can cause buffer overflows of the ground sensors, as newly arrived data is not promptly collected by the UAV.   Our proposal to address this challenge is to minimize packet loss by jointly optimizing the velocity controls and data collection schedules of multiple UAVs.Furthermore, in UASNets, swift movements of UAVs result in poor channel conditions and fast signal attenuation, leading to an extended age of information (AoI). In contrast, slow movements of UAVs prolong flight time, thereby extending the AoI of ground sensors.To address this challenge, we propose a new mean-field flight resource allocation optimization to minimize the AoI of sensory data.

{{</citation>}}


## cs.DC (2)



### (124/153) Load is not what you should balance: Introducing Prequal (Bartek Wydrowski et al., 2023)

{{<citation>}}

Bartek Wydrowski, Robert Kleinberg, Stephen M. Rumble, Aaron Archer. (2023)  
**Load is not what you should balance: Introducing Prequal**  

---
Primary Category: cs.DC  
Categories: C-2-4, cs-DC, cs.DC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.10172v1)  

---


**ABSTRACT**  
We present Prequal (Probing to Reduce Queuing and Latency), a load balancer for distributed multi-tenant systems. Prequal aims to minimize real-time request latency in the presence of heterogeneous server capacities and non-uniform, time-varying antagonist load. It actively probes server load to leverage the power-of-d-choices paradigm, extending it with asynchronous and reusable probes. Cutting against received wisdom, Prequal does not balance CPU load, but instead selects servers according to estimated latency and active requests-in-flight (RIF). We explore its major design features on a testbed system and evaluate it on YouTube, where it has been deployed for more than two years. Prequal has dramatically decreased tail latency, error rates, and resource use, enabling YouTube and other production systems at Google to run at much higher utilization.

{{</citation>}}


### (125/153) Decaffe: DHT Tree-Based Online Federated Fake News Detection (Cheng-Wei Ching et al., 2023)

{{<citation>}}

Cheng-Wei Ching, Liting Hu. (2023)  
**Decaffe: DHT Tree-Based Online Federated Fake News Detection**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2312.09547v1)  

---


**ABSTRACT**  
The proliferation of mobile social networks (MSNs) has transformed information dissemination, leading to increased reliance on these platforms for news consumption. However, this shift has been accompanied by the widespread propagation of fake news, posing significant challenges in terms of public panic, political influence, and the obscuring of truth. Traditional data processing pipelines for fake news detection in MSNs suffer from lengthy response times and poor scalability, failing to address the unique characteristics of news in MSNs, such as prompt propagation, large-scale quantity, and rapid evolution. This paper introduces a novel system named Decaffe - a DHT Tree-Based Online Federated Fake News Detection system. Decaffe leverages distributed hash table (DHT)-based aggregation trees for scalability and real-time detection, and it employs two model fine-tuning methods for adapting to mobile network dynamics. The system's structure includes a root, branches, and leaves for effective dissemination of a pre-trained model and ensemble-based aggregation of predictive results. Decaffe uniquely combines centralized server-based and decentralized serverless model fine-tuning approaches with personalized model fine-tuning, addressing the challenges of real-time detection, scalability, and adaptability in the dynamic environment of MSNs.

{{</citation>}}


## cs.PL (1)



### (126/153) ACPO: AI-Enabled Compiler-Driven Program Optimization (Amir H. Ashouri et al., 2023)

{{<citation>}}

Amir H. Ashouri, Muhammad Asif Manzoor, Duc Minh Vu, Raymond Zhang, Ziwen Wang, Angel Zhang, Bryan Chan, Tomasz S. Czajkowski, Yaoqing Gao. (2023)  
**ACPO: AI-Enabled Compiler-Driven Program Optimization**  

---
Primary Category: cs.PL  
Categories: I-2-5; D-3-0; I-2-6, cs-AI, cs-LG, cs-PF, cs-PL, cs.PL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09982v1)  

---


**ABSTRACT**  
The key to performance optimization of a program is to decide correctly when a certain transformation should be applied by a compiler. Traditionally, such profitability decisions are made by hand-coded algorithms tuned for a very small number of benchmarks, usually requiring a great deal of effort to be retuned when the benchmark suite changes. This is an ideal opportunity to apply machine-learning models to speed up the tuning process; while this realization has been around since the late 90s, only recent advancements in ML enabled a practical application of ML to compilers as an end-to-end framework. Even so, seamless integration of ML into the compiler would require constant rebuilding of the compiler when models are updated.   This paper presents ACPO: \textbf{\underline{A}}I-Enabled \textbf{\underline{C}}ompiler-driven \textbf{\underline{P}}rogram \textbf{\underline{O}}ptimization; a novel framework to provide LLVM with simple and comprehensive tools to benefit from employing ML models for different optimization passes. We first showcase the high-level view, class hierarchy, and functionalities of ACPO and subsequently, demonstrate \taco{a couple of use cases of ACPO by ML-enabling the Loop Unroll and Function Inlining passes and describe how ACPO can be leveraged to optimize other passes. Experimental results reveal that ACPO model for Loop Unroll is able to gain on average 4\% and 3\%, 5.4\%, 0.2\% compared to LLVM's O3 optimization when deployed on Polybench, Coral-2, CoreMark, and Graph-500, respectively. Furthermore, by adding the Inliner model as well, ACPO is able to provide up to 4.5\% and 2.4\% on Polybench and Cbench compared with LLVM's O3 optimization, respectively.

{{</citation>}}


## cs.IR (5)



### (127/153) GEAR-Up: Generative AI and External Knowledge-based Retrieval Upgrading Scholarly Article Searches for Systematic Reviews (Kaushik Roy et al., 2023)

{{<citation>}}

Kaushik Roy, Vedant Khandelwal, Harshul Surana, Valerie Vera, Amit Sheth, Heather Heckman. (2023)  
**GEAR-Up: Generative AI and External Knowledge-based Retrieval Upgrading Scholarly Article Searches for Systematic Reviews**  

---
Primary Category: cs.IR  
Categories: cs-DL, cs-IR, cs.IR  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.09948v1)  

---


**ABSTRACT**  
Systematic reviews (SRs) - the librarian-assisted literature survey of scholarly articles takes time and requires significant human resources. Given the ever-increasing volume of published studies, applying existing computing and informatics technology can decrease this time and resource burden. Due to the revolutionary advances in (1) Generative AI such as ChatGPT, and (2) External knowledge-augmented information extraction efforts such as Retrieval-Augmented Generation, In this work, we explore the use of techniques from (1) and (2) for SR. We demonstrate a system that takes user queries, performs query expansion to obtain enriched context (includes additional terms and definitions by querying language models and knowledge graphs), and uses this context to search for articles on scholarly databases to retrieve articles. We perform qualitative evaluations of our system through comparison against sentinel (ground truth) articles provided by an in-house librarian. The demo can be found at: https://youtu.be/zMdP56GJ9mU.

{{</citation>}}


### (128/153) Context-Driven Interactive Query Simulations Based on Generative Large Language Models (Björn Engelmann et al., 2023)

{{<citation>}}

Björn Engelmann, Timo Breuer, Jana Isabelle Friese, Philipp Schaer, Norbert Fuhr. (2023)  
**Context-Driven Interactive Query Simulations Based on Generative Large Language Models**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09631v1)  

---


**ABSTRACT**  
Simulating user interactions enables a more user-oriented evaluation of information retrieval (IR) systems. While user simulations are cost-efficient and reproducible, many approaches often lack fidelity regarding real user behavior. Most notably, current user models neglect the user's context, which is the primary driver of perceived relevance and the interactions with the search results. To this end, this work introduces the simulation of context-driven query reformulations. The proposed query generation methods build upon recent Large Language Model (LLM) approaches and consider the user's context throughout the simulation of a search session. Compared to simple context-free query generation approaches, these methods show better effectiveness and allow the simulation of more efficient IR sessions. Similarly, our evaluations consider more interaction context than current session-based measures and reveal interesting complementary insights in addition to the established evaluation protocols. We conclude with directions for future work and provide an entirely open experimental setup.

{{</citation>}}


### (129/153) Incorporating Judgment Prediction into Legal Case Retrieval via Law-aware Generative Retrieval (Weicong Qin et al., 2023)

{{<citation>}}

Weicong Qin, Zelin Cao, Weijie Yu, Zihua Si, Sirui Chen, Jun Xu. (2023)  
**Incorporating Judgment Prediction into Legal Case Retrieval via Law-aware Generative Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2312.09591v1)  

---


**ABSTRACT**  
Legal case retrieval and judgment prediction are crucial components in intelligent legal systems. In practice, determining whether two cases share the same charges through legal judgment prediction is essential for establishing their relevance in case retrieval. However, current studies on legal case retrieval merely focus on the semantic similarity between paired cases, ignoring their charge-level consistency. This separation leads to a lack of context and potential inaccuracies in the case retrieval that can undermine trust in the system's decision-making process. Given the guidance role of laws to both tasks and inspired by the success of generative retrieval, in this work, we propose to incorporate judgment prediction into legal case retrieval, achieving a novel law-aware Generative legal case retrieval method called Gear. Specifically, Gear first extracts rationales (key circumstances and key elements) for legal cases according to the definition of charges in laws, ensuring a shared and informative representation for both tasks. Then in accordance with the inherent hierarchy of laws, we construct a law structure constraint tree and assign law-aware semantic identifier(s) to each case based on this tree. These designs enable a unified traversal from the root, through intermediate charge nodes, to case-specific leaf nodes, which respectively correspond to two tasks. Additionally, in the training, we also introduce a revision loss that jointly minimizes the discrepancy between the identifiers of predicted and labeled charges as well as retrieved cases, improving the accuracy and consistency for both tasks. Extensive experiments on two datasets demonstrate that Gear consistently outperforms state-of-the-art methods in legal case retrieval while maintaining competitive judgment prediction performance.

{{</citation>}}


### (130/153) MONET: Modality-Embracing Graph Convolutional Network and Target-Aware Attention for Multimedia Recommendation (Yungi Kim et al., 2023)

{{<citation>}}

Yungi Kim, Taeri Kim, Won-Yong Shin, Sang-Wook Kim. (2023)  
**MONET: Modality-Embracing Graph Convolutional Network and Target-Aware Attention for Multimedia Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Attention, Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2312.09511v1)  

---


**ABSTRACT**  
In this paper, we focus on multimedia recommender systems using graph convolutional networks (GCNs) where the multimodal features as well as user-item interactions are employed together. Our study aims to exploit multimodal features more effectively in order to accurately capture users' preferences for items. To this end, we point out following two limitations of existing GCN-based multimedia recommender systems: (L1) although multimodal features of interacted items by a user can reveal her preferences on items, existing methods utilize GCN designed to focus only on capturing collaborative signals, resulting in insufficient reflection of the multimodal features in the final user/item embeddings; (L2) although a user decides whether to prefer the target item by considering its multimodal features, existing methods represent her as only a single embedding regardless of the target item's multimodal features and then utilize her embedding to predict her preference for the target item. To address the above issues, we propose a novel multimedia recommender system, named MONET, composed of following two core ideas: modality-embracing GCN (MeGCN) and target-aware attention. Through extensive experiments using four real-world datasets, we demonstrate i) the significant superiority of MONET over seven state-of-the-art competitors (up to 30.32% higher accuracy in terms of recall@20, compared to the best competitor) and ii) the effectiveness of the two core ideas in MONET. All MONET codes are available at https://github.com/Kimyungi/MONET.

{{</citation>}}


### (131/153) IndicIRSuite: Multilingual Dataset and Neural Information Models for Indian Languages (Saiful Haq et al., 2023)

{{<citation>}}

Saiful Haq, Ashutosh Sharma, Pushpak Bhattacharyya. (2023)  
**IndicIRSuite: Multilingual Dataset and Neural Information Models for Indian Languages**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: BERT, Information Retrieval, Machine Translation, Multilingual  
[Paper Link](http://arxiv.org/abs/2312.09508v1)  

---


**ABSTRACT**  
In this paper, we introduce Neural Information Retrieval resources for 11 widely spoken Indian Languages (Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, and Telugu) from two major Indian language families (Indo-Aryan and Dravidian). These resources include (a) INDIC-MARCO, a multilingual version of the MSMARCO dataset in 11 Indian Languages created using Machine Translation, and (b) Indic-ColBERT, a collection of 11 distinct Monolingual Neural Information Retrieval models, each trained on one of the 11 languages in the INDIC-MARCO dataset. To the best of our knowledge, IndicIRSuite is the first attempt at building large-scale Neural Information Retrieval resources for a large number of Indian languages, and we hope that it will help accelerate research in Neural IR for Indian Languages. Experiments demonstrate that Indic-ColBERT achieves 47.47% improvement in the MRR@10 score averaged over the INDIC-MARCO baselines for all 11 Indian languages except Oriya, 12.26% improvement in the NDCG@10 score averaged over the MIRACL Bengali and Hindi Language baselines, and 20% improvement in the MRR@100 Score over the Mr.Tydi Bengali Language baseline. IndicIRSuite is available at https://github.com/saifulhaq95/IndicIRSuite

{{</citation>}}


## cs.RO (3)



### (132/153) Sample-Efficient Learning to Solve a Real-World Labyrinth Game Using Data-Augmented Model-Based Reinforcement Learning (Thomas Bi et al., 2023)

{{<citation>}}

Thomas Bi, Raffaello D'Andrea. (2023)  
**Sample-Efficient Learning to Solve a Real-World Labyrinth Game Using Data-Augmented Model-Based Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09906v1)  

---


**ABSTRACT**  
Motivated by the challenge of achieving rapid learning in physical environments, this paper presents the development and training of a robotic system designed to navigate and solve a labyrinth game using model-based reinforcement learning techniques. The method involves extracting low-dimensional observations from camera images, along with a cropped and rectified image patch centered on the current position within the labyrinth, providing valuable information about the labyrinth layout. The learning of a control policy is performed purely on the physical system using model-based reinforcement learning, where the progress along the labyrinth's path serves as a reward signal. Additionally, we exploit the system's inherent symmetries to augment the training data. Consequently, our approach learns to successfully solve a popular real-world labyrinth game in record time, with only 5 hours of real-world training data.

{{</citation>}}


### (133/153) Drones Guiding Drones: Cooperative Navigation of a Less-Equipped Micro Aerial Vehicle in Cluttered Environments (Václav Pritzl et al., 2023)

{{<citation>}}

Václav Pritzl, Matouš Vrba, Yurii Stasinchuk, Vít Krátký, Jiří Horyna, Petr Štěpán, Martin Saska. (2023)  
**Drones Guiding Drones: Cooperative Navigation of a Less-Equipped Micro Aerial Vehicle in Cluttered Environments**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2312.09786v1)  

---


**ABSTRACT**  
Reliable deployment of Unmanned Aerial Vehicles (UAVs) in cluttered unknown environments requires accurate sensors for obstacle avoidance. Such a requirement limits the usage of cheap and micro-scale vehicles with constrained payload capacity if industrial-grade reliability and precision are required. This paper investigates the possibility of offloading the necessity to carry heavy and expensive obstacle sensors to another member of the UAV team while preserving the desired obstacle avoidance capability. A novel cooperative guidance framework offloading the obstacle sensing requirements from a minimalistic secondary UAV to a superior primary UAV is proposed. The primary UAV constructs a dense occupancy map of the environment and plans collision-free paths for both UAVs to ensure reaching the desired secondary UAV's goal. The primary UAV guides the secondary UAV to follow the planned path while tracking the UAV using Light Detection and Ranging (LiDAR)-based relative localization. The proposed approach was verified in real-world experiments with a heterogeneous team of a 3D LiDAR-equipped primary UAV and a camera-equipped secondary UAV moving autonomously through unknown cluttered Global Navigation Satellite System (GNSS)-denied environments with the proposed framework running completely on board the UAVs.

{{</citation>}}


### (134/153) Benchmarking the Full-Order Model Optimization Based Imitation in the Humanoid Robot Reinforcement Learning Walk (Ekaterina Chaikovskaya et al., 2023)

{{<citation>}}

Ekaterina Chaikovskaya, Inna Minashina, Vladimir Litvinenko, Egor Davydenko, Dmitry Makarov, Yulia Danik, Roman Gorbachev. (2023)  
**Benchmarking the Full-Order Model Optimization Based Imitation in the Humanoid Robot Reinforcement Learning Walk**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.09757v1)  

---


**ABSTRACT**  
When a gait of a bipedal robot is developed using deep reinforcement learning, reference trajectories may or may not be used. Each approach has its advantages and disadvantages, and the choice of method is up to the control developer. This paper investigates the effect of reference trajectories on locomotion learning and the resulting gaits. We implemented three gaits of a full-order anthropomorphic robot model with different reward imitation ratios, provided sim-to-sim control policy transfer, and compared the gaits in terms of robustness and energy efficiency. In addition, we conducted a qualitative analysis of the gaits by interviewing people, since our task was to create an appealing and natural gait for a humanoid robot.   According to the results of the experiments, the most successful approach was the one in which the average value of rewards for imitation and adherence to command velocity per episode remained balanced throughout the training. The gait obtained with this method retains naturalness (median of 3.6 according to the user study) compared to the gait trained with imitation only (median of 4.0), while remaining robust close to the gait trained without reference trajectories.

{{</citation>}}


## eess.IV (3)



### (135/153) SQA-SAM: Segmentation Quality Assessment for Medical Images Utilizing the Segment Anything Model (Yizhe Zhang et al., 2023)

{{<citation>}}

Yizhe Zhang, Shuo Wang, Tao Zhou, Qi Dou, Danny Z. Chen. (2023)  
**SQA-SAM: Segmentation Quality Assessment for Medical Images Utilizing the Segment Anything Model**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2312.09899v1)  

---


**ABSTRACT**  
Segmentation quality assessment (SQA) plays a critical role in the deployment of a medical image based AI system. Users need to be informed/alerted whenever an AI system generates unreliable/incorrect predictions. With the introduction of the Segment Anything Model (SAM), a general foundation segmentation model, new research opportunities emerged in how one can utilize SAM for medical image segmentation. In this paper, we propose a novel SQA method, called SQA-SAM, which exploits SAM to enhance the accuracy of quality assessment for medical image segmentation. When a medical image segmentation model (MedSeg) produces predictions for a test image, we generate visual prompts based on the predictions, and SAM is utilized to generate segmentation maps corresponding to the visual prompts. How well MedSeg's segmentation aligns with SAM's segmentation indicates how well MedSeg's segmentation aligns with the general perception of objectness and image region partition. We develop a score measure for such alignment. In experiments, we find that the generated scores exhibit moderate to strong positive correlation (in Pearson correlation and Spearman correlation) with Dice coefficient scores reflecting the true segmentation quality.

{{</citation>}}


### (136/153) On the calibration of neural networks for histological slide-level classification (Alexander Kurz et al., 2023)

{{<citation>}}

Alexander Kurz, Hendrik A. Mehrtens, Tabea-Clara Bucher, Titus J. Brinker. (2023)  
**On the calibration of neural networks for histological slide-level classification**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.09719v1)  

---


**ABSTRACT**  
Deep Neural Networks have shown promising classification performance when predicting certain biomarkers from Whole Slide Images in digital pathology. However, the calibration of the networks' output probabilities is often not evaluated. Communicating uncertainty by providing reliable confidence scores is of high relevance in the medical context. In this work, we compare three neural network architectures that combine feature representations on patch-level to a slide-level prediction with respect to their classification performance and evaluate their calibration. As slide-level classification task, we choose the prediction of Microsatellite Instability from Colorectal Cancer tissue sections. We observe that Transformers lead to good results in terms of classification performance and calibration. When evaluating the classification performance on a separate dataset, we observe that Transformers generalize best. The investigation of reliability diagrams provides additional insights to the Expected Calibration Error metric and we observe that especially Transformers push the output probabilities to extreme values, which results in overconfident predictions.

{{</citation>}}


### (137/153) SegRap2023: A Benchmark of Organs-at-Risk and Gross Tumor Volume Segmentation for Radiotherapy Planning of Nasopharyngeal Carcinoma (Xiangde Luo et al., 2023)

{{<citation>}}

Xiangde Luo, Jia Fu, Yunxin Zhong, Shuolin Liu, Bing Han, Mehdi Astaraki, Simone Bendazzoli, Iuliana Toma-Dasu, Yiwen Ye, Ziyang Chen, Yong Xia, Yanzhou Su, Jin Ye, Junjun He, Zhaohu Xing, Hongqiu Wang, Lei Zhu, Kaixiang Yang, Xin Fang, Zhiwei Wang, Chan Woong Lee, Sang Joon Park, Jaehee Chun, Constantin Ulrich, Klaus H. Maier-Hein, Nchongmaje Ndipenoch, Alina Miron, Yongmin Li, Yimeng Zhang, Yu Chen, Lu Bai, Jinlong Huang, Chengyang An, Lisheng Wang, Kaiwen Huang, Yunqi Gu, Tao Zhou, Mu Zhou, Shichuan Zhang, Wenjun Liao, Guotai Wang, Shaoting Zhang. (2023)  
**SegRap2023: A Benchmark of Organs-at-Risk and Gross Tumor Volume Segmentation for Radiotherapy Planning of Nasopharyngeal Carcinoma**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09576v1)  

---


**ABSTRACT**  
Radiation therapy is a primary and effective NasoPharyngeal Carcinoma (NPC) treatment strategy. The precise delineation of Gross Tumor Volumes (GTVs) and Organs-At-Risk (OARs) is crucial in radiation treatment, directly impacting patient prognosis. Previously, the delineation of GTVs and OARs was performed by experienced radiation oncologists. Recently, deep learning has achieved promising results in many medical image segmentation tasks. However, for NPC OARs and GTVs segmentation, few public datasets are available for model development and evaluation. To alleviate this problem, the SegRap2023 challenge was organized in conjunction with MICCAI2023 and presented a large-scale benchmark for OAR and GTV segmentation with 400 Computed Tomography (CT) scans from 200 NPC patients, each with a pair of pre-aligned non-contrast and contrast-enhanced CT scans. The challenge's goal was to segment 45 OARs and 2 GTVs from the paired CT scans. In this paper, we detail the challenge and analyze the solutions of all participants. The average Dice similarity coefficient scores for all submissions ranged from 76.68\% to 86.70\%, and 70.42\% to 73.44\% for OARs and GTVs, respectively. We conclude that the segmentation of large-size OARs is well-addressed, and more efforts are needed for GTVs and small-size or thin-structure OARs. The benchmark will remain publicly available here: https://segrap2023.grand-challenge.org

{{</citation>}}


## cs.CR (6)



### (138/153) ESAT:Extended Security in the Air using TESLA (Mikaëla Ngamboé et al., 2023)

{{<citation>}}

Mikaëla Ngamboé, Xiao Niu, Benoit Joly, Steven P Biegler, Paul Berthier, Rémi Benito, Greg Rice, José M Fernandez, Gabriela Nicolescu. (2023)  
**ESAT:Extended Security in the Air using TESLA**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.09870v1)  

---


**ABSTRACT**  
The Automatic Dependent Surveillance-Broadcast (ADS-B) is a surveillance technology that becomes mandatory in many airspaces. It improves safety, increases efficiency and reduces air traffic congestion by broadcasting aircraft navigation data. Yet, ADS-B is vulnerable to spoofing attacks as it lacks mechanisms to ensure the integrity and authenticity of the data being supplied. None of the existing cryptographic solutions fully meet the backward compatibility and bandwidth preservation requirements of the standard. Hence, we propose Extended Security in the Air using TESLA (ESAT), an enhanced approach that integrates TESLA, phase-overlay modulation techniques and certificate-based PKI. As a result, entity authentication, data origin authentication, and data integrity are the security services that ESAT offers. To assess compliance with the standard, we designed an SDR-based implementation of ESAT and performed backwards compatibility tests on commercial and general aviation (GA) ADS-B in receivers. Besides, we calculated the 1090ES band's activity factor and analyzed the channel occupancy rate according to ITU-R SM.2256-1 recommendation. Also, we performed a bit error rate analysis of ESAT messages. The results suggest that ESAT is backward compatible, does not incur significant communication overhead, and has an error rate that is acceptable for Eb/No values above 14 dB.

{{</citation>}}


### (139/153) Silent Guardian: Protecting Text from Malicious Exploitation by Large Language Models (Jiawei Zhao et al., 2023)

{{<citation>}}

Jiawei Zhao, Kejiang Chen, Xiaojian Yuan, Yuang Qi, Weiming Zhang, Nenghai Yu. (2023)  
**Silent Guardian: Protecting Text from Malicious Exploitation by Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.09669v2)  

---


**ABSTRACT**  
The rapid development of large language models (LLMs) has yielded impressive success in various downstream tasks. However, the vast potential and remarkable capabilities of LLMs also raise new security and privacy concerns if they are exploited for nefarious purposes due to their open-endedness. For example, LLMs may be used to plagiarize or imitate writing, thereby infringing the copyright of the original content, or to create indiscriminate fake information based on a certain source text. In some cases, LLMs can even analyze text from the Internet to infer personal privacy. Unfortunately, previous text protection research could not foresee the emergence of powerful LLMs, rendering it no longer effective in this new context. To bridge this gap, we introduce Silent Guardian (SG), a text protection mechanism against LLMs, which allows LLMs to refuse to generate response when receiving protected text, preventing the malicious use of text from the source. Specifically, we first propose the concept of Truncation Protection Examples (TPE). By carefully modifying the text to be protected, TPE can induce LLMs to first sample the end token, thus directly terminating the interaction. In addition, to efficiently construct TPE in the discrete space of text data, we propose a novel optimization algorithm called Super Taliored Protection (STP), which is not only highly efficient but also maintains the semantic consistency of the text during the optimization process. The comprehensive experimental evaluation demonstrates that SG can effectively protect the target text under various configurations and achieve almost 100% protection success rate in some cases. Notably, SG also exhibits relatively good transferability and robustness, making its application in practical scenarios possible.

{{</citation>}}


### (140/153) Madtls: Fine-grained Middlebox-aware End-to-end Security for Industrial Communication (Eric Wagner et al., 2023)

{{<citation>}}

Eric Wagner, David Heye, Martin Serror, Ike Kunze, Klaus Wehrle, Martin Henze. (2023)  
**Madtls: Fine-grained Middlebox-aware End-to-end Security for Industrial Communication**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.09650v1)  

---


**ABSTRACT**  
Industrial control systems increasingly rely on middlebox functionality such as intrusion detection or in-network processing. However, traditional end-to-end security protocols interfere with the necessary access to in-flight data. While recent work on middlebox-aware end-to-end security protocols for the traditional Internet promises to address the dilemma between end-to-end security guarantees and middleboxes, the current state-of-the-art lacks critical features for industrial communication. Most importantly, industrial settings require fine-grained access control for middleboxes to truly operate in a least-privilege mode. Likewise, advanced applications even require that middleboxes can inject specific messages (e.g., emergency shutdowns). Meanwhile, industrial scenarios often expose tight latency and bandwidth constraints not found in the traditional Internet. As the current state-of-the-art misses critical features, we propose Middlebox-aware DTLS (Madtls), a middlebox-aware end-to-end security protocol specifically tailored to the needs of industrial networks. Madtls provides bit-level read and write access control of middleboxes to communicated data with minimal bandwidth and processing overhead, even on constrained hardware.

{{</citation>}}


### (141/153) A Malware Classification Survey on Adversarial Attacks and Defences (Mahesh Datta Sai Ponnuru et al., 2023)

{{<citation>}}

Mahesh Datta Sai Ponnuru, Likhitha Amasala, Tanu Sree Bhimavarapu, Guna Chaitanya Garikipati. (2023)  
**A Malware Classification Survey on Adversarial Attacks and Defences**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.09636v1)  

---


**ABSTRACT**  
As the number and complexity of malware attacks continue to increase, there is an urgent need for effective malware detection systems. While deep learning models are effective at detecting malware, they are vulnerable to adversarial attacks. Attacks like this can create malicious files that are resistant to detection, creating a significant cybersecurity risk. Recent research has seen the development of several adversarial attack and response approaches aiming at strengthening deep learning models' resilience to such attacks. This survey study offers an in-depth look at current research in adversarial attack and defensive strategies for malware classification in cybersecurity. The methods are classified into four categories: generative models, feature-based approaches, ensemble methods, and hybrid tactics. The article outlines cutting-edge procedures within each area, assessing their benefits and drawbacks. Each topic presents cutting-edge approaches and explores their advantages and disadvantages. In addition, the study discusses the datasets and assessment criteria that are often utilized on this subject. Finally, it identifies open research difficulties and suggests future study options. This document is a significant resource for malware categorization and cyber security researchers and practitioners.

{{</citation>}}


### (142/153) Binary Code Summarization: Benchmarking ChatGPT/GPT-4 and Other Large Language Models (Xin Jin et al., 2023)

{{<citation>}}

Xin Jin, Jonathan Larson, Weiwei Yang, Zhiqiang Lin. (2023)  
**Binary Code Summarization: Benchmarking ChatGPT/GPT-4 and Other Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs-LG, cs-SE, cs.CR  
Keywords: ChatGPT, GPT, GPT-4, Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2312.09601v1)  

---


**ABSTRACT**  
Binary code summarization, while invaluable for understanding code semantics, is challenging due to its labor-intensive nature. This study delves into the potential of large language models (LLMs) for binary code comprehension. To this end, we present BinSum, a comprehensive benchmark and dataset of over 557K binary functions and introduce a novel method for prompt synthesis and optimization. To more accurately gauge LLM performance, we also propose a new semantic similarity metric that surpasses traditional exact-match approaches. Our extensive evaluation of prominent LLMs, including ChatGPT, GPT-4, Llama 2, and Code Llama, reveals 10 pivotal insights. This evaluation generates 4 billion inference tokens, incurred a total expense of 11,418 US dollars and 873 NVIDIA A100 GPU hours. Our findings highlight both the transformative potential of LLMs in this field and the challenges yet to be overcome.

{{</citation>}}


### (143/153) No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models (Shengyao Zhang et al., 2023)

{{<citation>}}

Shengyao Zhang, Mi Zhang, Xudong Pan, Min Yang. (2023)  
**No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs.CR  
Keywords: BERT, GLUE, Language Model  
[Paper Link](http://arxiv.org/abs/2312.09494v2)  

---


**ABSTRACT**  
To reduce the computation cost and the energy consumption in large language models (LLM), skimming-based acceleration dynamically drops unimportant tokens of the input sequence progressively along layers of the LLM while preserving the tokens of semantic importance. However, our work for the first time reveals the acceleration may be vulnerable to Denial-of-Service (DoS) attacks. In this paper, we propose No-Skim, a general framework to help the owners of skimming-based LLM to understand and measure the robustness of their acceleration scheme. Specifically, our framework searches minimal and unnoticeable perturbations at character-level and token-level to generate adversarial inputs that sufficiently increase the remaining token ratio, thus increasing the computation cost and energy consumption. We systematically evaluate the vulnerability of the skimming acceleration in various LLM architectures including BERT and RoBERTa on the GLUE benchmark. In the worst case, the perturbation found by No-Skim substantially increases the running cost of LLM by over 145% on average. Moreover, No-Skim extends the evaluation framework to various scenarios, making the evaluation conductible with different level of knowledge.

{{</citation>}}


## cs.LO (1)



### (144/153) Submodel Enumeration for CTL Is Hard (Nicolas Fröhlich et al., 2023)

{{<citation>}}

Nicolas Fröhlich, Arne Meier. (2023)  
**Submodel Enumeration for CTL Is Hard**  

---
Primary Category: cs.LO  
Categories: cs-CC, cs-LO, cs.LO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09868v1)  

---


**ABSTRACT**  
Expressing system specifications using Computation Tree Logic (CTL) formulas, formalising programs using Kripke structures, and then model checking the system is an established workflow in program verification and has wide applications in AI. In this paper, we consider the task of model enumeration, which asks for a uniform stream of output systems that satisfy the given specification. We show that, given a CTL formula and a system (potentially falsified by the formula), enumerating satisfying submodels is always hard for CTL - regardless of which subset of CTL operators is considered. As a silver lining on the horizon, we present fragments via restrictions on the allowed Boolean functions that still allow for fast enumeration.

{{</citation>}}


## eess.AS (4)



### (145/153) U2-KWS: Unified Two-pass Open-vocabulary Keyword Spotting with Keyword Bias (Ao Zhang et al., 2023)

{{<citation>}}

Ao Zhang, Pan Zhou, Kaixun Huang, Yong Zou, Ming Liu, Lei Xie. (2023)  
**U2-KWS: Unified Two-pass Open-vocabulary Keyword Spotting with Keyword Bias**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.09760v1)  

---


**ABSTRACT**  
Open-vocabulary keyword spotting (KWS), which allows users to customize keywords, has attracted increasingly more interest. However, existing methods based on acoustic models and post-processing train the acoustic model with ASR training criteria to model all phonemes, making the acoustic model under-optimized for the KWS task. To solve this problem, we propose a novel unified two-pass open-vocabulary KWS (U2-KWS) framework inspired by the two-pass ASR model U2. Specifically, we employ the CTC branch as the first stage model to detect potential keyword candidates and the decoder branch as the second stage model to validate candidates. In order to enhance any customized keywords, we redesign the U2 training procedure for U2-KWS and add keyword information by audio and text cross-attention into both branches. We perform experiments on our internal dataset and Aishell-1. The results show that U2-KWS can achieve a significant relative wake-up rate improvement of 41% compared to the traditional customized KWS systems when the false alarm rate is fixed to 0.5 times per hour.

{{</citation>}}


### (146/153) Fine-Tuned Self-Supervised Speech Representations for Language Diarization in Multilingual Code-Switched Speech (Geoffrey Frost et al., 2023)

{{<citation>}}

Geoffrey Frost, Emily Morris, Joshua Jansen van Vüren, Thomas Niesler. (2023)  
**Fine-Tuned Self-Supervised Speech Representations for Language Diarization in Multilingual Code-Switched Speech**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-SD, eess-AS, eess.AS  
Keywords: Multilingual, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.09645v1)  

---


**ABSTRACT**  
Annotating a multilingual code-switched corpus is a painstaking process requiring specialist linguistic expertise. This is partly due to the large number of language combinations that may appear within and across utterances, which might require several annotators with different linguistic expertise to consider an utterance sequentially. This is time-consuming and costly. It would be useful if the spoken languages in an utterance and the boundaries thereof were known before annotation commences, to allow segments to be assigned to the relevant language experts in parallel. To address this, we investigate the development of a continuous multilingual language diarizer using fine-tuned speech representations extracted from a large pre-trained self-supervised architecture (WavLM). We experiment with a code-switched corpus consisting of five South African languages (isiZulu, isiXhosa, Setswana, Sesotho and English) and show substantial diarization error rate improvements for language families, language groups, and individual languages over baseline systems.

{{</citation>}}


### (147/153) Self-Supervised Learning for Anomalous Sound Detection (Kevin Wilkinghoff, 2023)

{{<citation>}}

Kevin Wilkinghoff. (2023)  
**Self-Supervised Learning for Anomalous Sound Detection**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.09578v1)  

---


**ABSTRACT**  
State-of-the-art anomalous sound detection (ASD) systems are often trained by using an auxiliary classification task to learn an embedding space. Doing so enables the system to learn embeddings that are robust to noise and are ignoring non-target sound events but requires manually annotated meta information to be used as class labels. However, the less difficult the classification task becomes, the less informative are the embeddings and the worse is the resulting ASD performance. A solution to this problem is to utilize self-supervised learning (SSL). In this work, feature exchange (FeatEx), a simple yet effective SSL approach for ASD, is proposed. In addition, FeatEx is compared to and combined with existing SSL approaches. As the main result, a new state-of-the-art performance for the DCASE2023 ASD dataset is obtained that outperforms all other published results on this dataset by a large margin.

{{</citation>}}


### (148/153) IR-UWB Radar-Based Contactless Silent Speech Recognition of Vowels, Consonants, Words, and Phrases (Sunghwa Lee et al., 2023)

{{<citation>}}

Sunghwa Lee, Younghoon Shin, Myungjong Kim, Jiwon Seo. (2023)  
**IR-UWB Radar-Based Contactless Silent Speech Recognition of Vowels, Consonants, Words, and Phrases**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.09572v1)  

---


**ABSTRACT**  
Several sensing techniques have been proposed for silent speech recognition (SSR); however, many of these methods require invasive processes or sensor attachment to the skin using adhesive tape or glue, rendering them unsuitable for frequent use in daily life. By contrast, impulse radio ultra-wideband (IR-UWB) radar can operate without physical contact with users' articulators and related body parts, offering several advantages for SSR. These advantages include high range resolution, high penetrability, low power consumption, robustness to external light or sound interference, and the ability to be embedded in space-constrained handheld devices. This study demonstrated IR-UWB radar-based contactless SSR using four types of speech stimuli (vowels, consonants, words, and phrases). To achieve this, a novel speech feature extraction algorithm specifically designed for IR-UWB radar-based SSR is proposed. Each speech stimulus is recognized by applying a classification algorithm to the extracted speech features. Two different algorithms, multidimensional dynamic time warping (MD-DTW) and deep neural network-hidden Markov model (DNN-HMM), were compared for the classification task. Additionally, a favorable radar antenna position, either in front of the user's lips or below the user's chin, was determined to achieve higher recognition accuracy. Experimental results demonstrated the efficacy of the proposed speech feature extraction algorithm combined with DNN-HMM for classifying vowels, consonants, words, and phrases. Notably, this study represents the first demonstration of phoneme-level SSR using contactless radar.

{{</citation>}}


## cs.SD (2)



### (149/153) Automatic channel selection and spatial feature integration for multi-channel speech recognition across various array topologies (Bingshen Mu et al., 2023)

{{<citation>}}

Bingshen Mu, Pengcheng Guo, Dake Guo, Pan Zhou, Wei Chen, Lei Xie. (2023)  
**Automatic channel selection and spatial feature integration for multi-channel speech recognition across various array topologies**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.09746v1)  

---


**ABSTRACT**  
Automatic Speech Recognition (ASR) has shown remarkable progress, yet it still faces challenges in real-world distant scenarios across various array topologies each with multiple recording devices. The focal point of the CHiME-7 Distant ASR task is to devise a unified system capable of generalizing various array topologies that have multiple recording devices and offering reliable recognition performance in real-world environments. Addressing this task, we introduce an ASR system that demonstrates exceptional performance across various array topologies. First of all, we propose two attention-based automatic channel selection modules to select the most advantageous subset of multi-channel signals from multiple recording devices for each utterance. Furthermore, we introduce inter-channel spatial features to augment the effectiveness of multi-frame cross-channel attention, aiding it in improving the capability of spatial information awareness. Finally, we propose a multi-layer convolution fusion module drawing inspiration from the U-Net architecture to integrate the multi-channel output into a single-channel output. Experimental results on the CHiME-7 corpus with oracle segmentation demonstrate that the improvements introduced in our proposed ASR system lead to a relative reduction of 40.1% in the Macro Diarization Attributed Word Error Rates (DA-WER) when compared to the baseline ASR system on the Eval sets.

{{</citation>}}


### (150/153) Stethoscope-guided Supervised Contrastive Learning for Cross-domain Adaptation on Respiratory Sound Classification (June-Woo Kim et al., 2023)

{{<citation>}}

June-Woo Kim, Sangmin Bae, Won-Yang Cho, Byungjo Lee, Ho-Young Jung. (2023)  
**Stethoscope-guided Supervised Contrastive Learning for Cross-domain Adaptation on Respiratory Sound Classification**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.09603v1)  

---


**ABSTRACT**  
Despite the remarkable advances in deep learning technology, achieving satisfactory performance in lung sound classification remains a challenge due to the scarcity of available data. Moreover, the respiratory sound samples are collected from a variety of electronic stethoscopes, which could potentially introduce biases into the trained models. When a significant distribution shift occurs within the test dataset or in a practical scenario, it can substantially decrease the performance. To tackle this issue, we introduce cross-domain adaptation techniques, which transfer the knowledge from a source domain to a distinct target domain. In particular, by considering different stethoscope types as individual domains, we propose a novel stethoscope-guided supervised contrastive learning approach. This method can mitigate any domain-related disparities and thus enables the model to distinguish respiratory sounds of the recording variation of the stethoscope. The experimental results on the ICBHI dataset demonstrate that the proposed methods are effective in reducing the domain dependency and achieving the ICBHI Score of 61.71%, which is a significant improvement of 2.16% over the baseline.

{{</citation>}}


## cs.SE (3)



### (151/153) Uncovering the Causes of Emotions in Software Developer Communication Using Zero-shot LLMs (Mia Mohammad Imran et al., 2023)

{{<citation>}}

Mia Mohammad Imran, Preetha Chatterjee, Kostadin Damevski. (2023)  
**Uncovering the Causes of Emotions in Software Developer Communication Using Zero-shot LLMs**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.09731v1)  

---


**ABSTRACT**  
Understanding and identifying the causes behind developers' emotions (e.g., Frustration caused by `delays in merging pull requests') can be crucial towards finding solutions to problems and fostering collaboration in open-source communities. Effectively identifying such information in the high volume of communications across the different project channels, such as chats, emails, and issue comments, requires automated recognition of emotions and their causes. To enable this automation, large-scale software engineering-specific datasets that can be used to train accurate machine learning models are required. However, such datasets are expensive to create with the variety and informal nature of software projects' communication channels.   In this paper, we explore zero-shot LLMs that are pre-trained on massive datasets but without being fine-tuned specifically for the task of detecting emotion causes in software engineering: ChatGPT, GPT-4, and flan-alpaca. Our evaluation indicates that these recently available models can identify emotion categories when given detailed emotions, although they perform worse than the top-rated models. For emotion cause identification, our results indicate that zero-shot LLMs are effective at recognizing the correct emotion cause with a BLEU-2 score of 0.598. To highlight the potential use of these techniques, we conduct a case study of the causes of Frustration in the last year of development of a popular open-source project, revealing several interesting insights.

{{</citation>}}


### (152/153) A Synthesis of Green Architectural Tactics for ML-Enabled Systems (Heli Järvenpää et al., 2023)

{{<citation>}}

Heli Järvenpää, Patricia Lago, Justus Bogner, Grace Lewis, Henry Muccini, Ipek Ozkaya. (2023)  
**A Synthesis of Green Architectural Tactics for ML-Enabled Systems**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.09610v1)  

---


**ABSTRACT**  
The rapid adoption of artificial intelligence (AI) and machine learning (ML) has generated growing interest in understanding their environmental impact and the challenges associated with designing environmentally friendly ML-enabled systems. While Green AI research, i.e., research that tries to minimize the energy footprint of AI, is receiving increasing attention, very few concrete guidelines are available on how ML-enabled systems can be designed to be more environmentally sustainable. In this paper, we provide a catalog of 30 green architectural tactics for ML-enabled systems to fill this gap. An architectural tactic is a high-level design technique to improve software quality, in our case environmental sustainability. We derived the tactics from the analysis of 51 peer-reviewed publications that primarily explore Green AI, and validated them using a focus group approach with three experts. The 30 tactics we identified are aimed to serve as an initial reference guide for further exploration into Green AI from a software engineering perspective, and assist in designing sustainable ML-enabled systems. To enhance transparency and facilitate their widespread use and extension, we make the tactics available online in easily consumable formats. Wide-spread adoption of these tactics has the potential to substantially reduce the societal impact of ML-enabled systems regarding their energy and carbon footprint.

{{</citation>}}


### (153/153) A Review of Repository Level Prompting for LLMs (Douglas Schonholtz, 2023)

{{<citation>}}

Douglas Schonholtz. (2023)  
**A Review of Repository Level Prompting for LLMs**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.10101v1)  

---


**ABSTRACT**  
As coding challenges become more complex, recent advancements in Large Language Models (LLMs) have led to notable successes, such as achieving a 94.6\% solve rate on the HumanEval benchmark. Concurrently, there is an increasing commercial push for repository-level inline code completion tools, such as GitHub Copilot and Tab Nine, aimed at enhancing developer productivity. This paper delves into the transition from individual coding problems to repository-scale solutions, presenting a thorough review of the current literature on effective LLM prompting for code generation at the repository level. We examine approaches that will work with black-box LLMs such that they will be useful and applicable to commercial use cases, and their applicability in interpreting code at a repository scale. We juxtapose the Repository-Level Prompt Generation technique with RepoCoder, an iterative retrieval and generation method, to highlight the trade-offs inherent in each approach and to establish best practices for their application in cutting-edge coding benchmarks. The interplay between iterative refinement of prompts and the development of advanced retrieval systems forms the core of our discussion, offering a pathway to significantly improve LLM performance in code generation tasks. Insights from this study not only guide the application of these methods but also chart a course for future research to integrate such techniques into broader software engineering contexts.

{{</citation>}}
