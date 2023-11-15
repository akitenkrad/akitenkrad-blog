---
draft: false
title: "arXiv @ 2023.11.15"
date: 2023-11-15
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.15"
    identifier: arxiv_20231115
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (53)](#cscl-53)
- [cs.CY (1)](#cscy-1)
- [cs.LG (20)](#cslg-20)
- [cs.SD (3)](#cssd-3)
- [cs.CV (27)](#cscv-27)
- [cs.AI (5)](#csai-5)
- [eess.SY (1)](#eesssy-1)
- [cs.CR (4)](#cscr-4)
- [stat.ML (1)](#statml-1)
- [q-fin.GN (1)](#q-fingn-1)
- [cs.HC (2)](#cshc-2)
- [physics.geo-ph (1)](#physicsgeo-ph-1)
- [cs.SE (4)](#csse-4)
- [q-bio.GN (2)](#q-biogn-2)
- [eess.AS (1)](#eessas-1)
- [cs.NI (2)](#csni-2)
- [cs.RO (4)](#csro-4)
- [cs.CE (1)](#csce-1)
- [eess.IV (2)](#eessiv-2)
- [cs.IR (2)](#csir-2)
- [quant-ph (1)](#quant-ph-1)
- [cs.ET (1)](#cset-1)
- [cs.SI (1)](#cssi-1)
- [econ.GN (1)](#econgn-1)
- [cs.IT (1)](#csit-1)

## cs.CL (53)



### (1/142) In-context Learning Generalizes, But Not Always Robustly: The Case of Syntax (Aaron Mueller et al., 2023)

{{<citation>}}

Aaron Mueller, Albert Webson, Jackson Petty, Tal Linzen. (2023)  
**In-context Learning Generalizes, But Not Always Robustly: The Case of Syntax**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, PaLM  
[Paper Link](http://arxiv.org/abs/2311.07811v1)  

---


**ABSTRACT**  
In-context learning (ICL) is now a common method for supervising large language models (LLMs): given labeled examples in the input context, the LLM learns to perform the task without weight updates. Despite ICL's prevalence and utility, we understand little about whether models supervised in this manner represent the underlying structure of their tasks, rather than superficial heuristics that only generalize to identically distributed examples. In this study, we investigate the robustness of LLMs supervised via ICL using the test case of sensitivity to syntax, which is a prerequisite for robust language understanding. Our experiments are based on two simple and well-controlled syntactic transformations tasks, where correct out-of-distribution generalization requires an accurate syntactic analysis of the input. We further investigate whether out-of-distribution generalization can be improved via chain-of-thought prompting, where the model is provided with a sequence of intermediate computation steps that illustrate how the task ought to be performed. In experiments with models from the GPT, PaLM, and Llama 2 families, we find large variance across LMs on this fundamental linguistic phenomenon, and that the variance is explained more by the composition of the pre-training corpus and supervision methods than by model size. In particular, we find evidence that models pre-trained on code generalize better, and benefit to a greater extent from chain-of-thought prompting.

{{</citation>}}


### (2/142) IruMozhi: Automatically classifying diglossia in Tamil (Kabilan Prasanna et al., 2023)

{{<citation>}}

Kabilan Prasanna, Aryaman Arora. (2023)  
**IruMozhi: Automatically classifying diglossia in Tamil**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.07804v1)  

---


**ABSTRACT**  
Tamil, a Dravidian language of South Asia, is a highly diglossic language with two very different registers in everyday use: Literary Tamil (preferred in writing and formal communication) and Spoken Tamil (confined to speech and informal media). Spoken Tamil is under-supported in modern NLP systems. In this paper, we release IruMozhi, a human-annotated dataset of parallel text in Literary and Spoken Tamil. We train classifiers on the task of identifying which variety a text belongs to. We use these models to gauge the availability of pretraining data in Spoken Tamil, to audit the composition of existing labelled datasets for Tamil, and to encourage future work on the variety.

{{</citation>}}


### (3/142) GreekT5: A Series of Greek Sequence-to-Sequence Models for News Summarization (Nikolaos Giarelis et al., 2023)

{{<citation>}}

Nikolaos Giarelis, Charalampos Mastrokostas, Nikos Karacapilidis. (2023)  
**GreekT5: A Series of Greek Sequence-to-Sequence Models for News Summarization**  

---
Primary Category: cs.CL  
Categories: 68T07, 68T50, I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: NLP, Sequence-to-Sequence, Summarization, T5  
[Paper Link](http://arxiv.org/abs/2311.07767v1)  

---


**ABSTRACT**  
Text summarization (TS) is a natural language processing (NLP) subtask pertaining to the automatic formulation of a concise and coherent summary that covers the major concepts and topics from one or multiple documents. Recent advancements in deep learning have led to the development of abstractive summarization transformer-based models, which outperform classical approaches. In any case, research in this field focuses on high resource languages such as English, while the corresponding work for low resource languages is still underdeveloped. Taking the above into account, this paper proposes a series of novel TS models for Greek news articles. The proposed models were thoroughly evaluated on the same dataset against GreekBART, which is the state-of-the-art model in Greek abstractive news summarization. Our evaluation results reveal that most of the proposed models significantly outperform GreekBART on various evaluation metrics. We make our evaluation code public, aiming to increase the reproducibility of this work and facilitate future research in the field.

{{</citation>}}


### (4/142) PolyIE: A Dataset of Information Extraction from Polymer Material Scientific Literature (Jerry Junyang Cheung et al., 2023)

{{<citation>}}

Jerry Junyang Cheung, Yuchen Zhuang, Yinghao Li, Pranav Shetty, Wantian Zhao, Sanjeev Grampurohit, Rampi Ramprasad, Chao Zhang. (2023)  
**PolyIE: A Dataset of Information Extraction from Polymer Material Scientific Literature**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2311.07715v1)  

---


**ABSTRACT**  
Scientific information extraction (SciIE), which aims to automatically extract information from scientific literature, is becoming more important than ever. However, there are no existing SciIE datasets for polymer materials, which is an important class of materials used ubiquitously in our daily lives. To bridge this gap, we introduce POLYIE, a new SciIE dataset for polymer materials. POLYIE is curated from 146 full-length polymer scholarly articles, which are annotated with different named entities (i.e., materials, properties, values, conditions) as well as their N-ary relations by domain experts. POLYIE presents several unique challenges due to diverse lexical formats of entities, ambiguity between entities, and variable-length relations. We evaluate state-of-the-art named entity extraction and relation extraction models on POLYIE, analyze their strengths and weaknesses, and highlight some difficult cases for these models. To the best of our knowledge, POLYIE is the first SciIE benchmark for polymer materials, and we hope it will lead to more research efforts from the community on this challenging task. Our code and data are available on: https://github.com/jerry3027/PolyIE.

{{</citation>}}


### (5/142) AuthentiGPT: Detecting Machine-Generated Text via Black-Box Language Models Denoising (Zhen Guo et al., 2023)

{{<citation>}}

Zhen Guo, Shangdi Yu. (2023)  
**AuthentiGPT: Detecting Machine-Generated Text via Black-Box Language Models Denoising**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07700v1)  

---


**ABSTRACT**  
Large language models (LLMs) have opened up enormous opportunities while simultaneously posing ethical dilemmas. One of the major concerns is their ability to create text that closely mimics human writing, which can lead to potential misuse, such as academic misconduct, disinformation, and fraud. To address this problem, we present AuthentiGPT, an efficient classifier that distinguishes between machine-generated and human-written texts. Under the assumption that human-written text resides outside the distribution of machine-generated text, AuthentiGPT leverages a black-box LLM to denoise input text with artificially added noise, and then semantically compares the denoised text with the original to determine if the content is machine-generated. With only one trainable parameter, AuthentiGPT eliminates the need for a large training dataset, watermarking the LLM's output, or computing the log-likelihood. Importantly, the detection capability of AuthentiGPT can be easily adapted to any generative language model. With a 0.918 AUROC score on a domain-specific dataset, AuthentiGPT demonstrates its effectiveness over other commercial algorithms, highlighting its potential for detecting machine-generated text in academic settings.

{{</citation>}}


### (6/142) MART: Improving LLM Safety with Multi-round Automatic Red-Teaming (Suyu Ge et al., 2023)

{{<citation>}}

Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, Yuning Mao. (2023)  
**MART: Improving LLM Safety with Multi-round Automatic Red-Teaming**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07689v1)  

---


**ABSTRACT**  
Red-teaming is a common practice for mitigating unsafe behaviors in Large Language Models (LLMs), which involves thoroughly assessing LLMs to identify potential flaws and addressing them with responsible and accurate responses. While effective, manual red-teaming is costly, and existing automatic red-teaming typically discovers safety risks without addressing them. In this paper, we propose a Multi-round Automatic Red-Teaming (MART) method, which incorporates both automatic adversarial prompt writing and safe response generation, significantly increasing red-teaming scalability and the safety of the target LLM. Specifically, an adversarial LLM and a target LLM interplay with each other in an iterative manner, where the adversarial LLM aims to generate challenging prompts that elicit unsafe responses from the target LLM, while the target LLM is fine-tuned with safety aligned data on these adversarial prompts. In each round, the adversarial LLM crafts better attacks on the updated target LLM, while the target LLM also improves itself through safety fine-tuning. On adversarial prompt benchmarks, the violation rate of an LLM with limited safety alignment reduces up to 84.7% after 4 rounds of MART, achieving comparable performance to LLMs with extensive adversarial prompt writing. Notably, model helpfulness on non-adversarial prompts remains stable throughout iterations, indicating the target LLM maintains strong performance on instruction following.

{{</citation>}}


### (7/142) Language Model-In-The-Loop: Data Optimal Approach to Learn-To-Recommend Actions in Text Games (Arjun Vaithilingam Sudhakar et al., 2023)

{{<citation>}}

Arjun Vaithilingam Sudhakar, Prasanna Parthasarathi, Janarthanan Rajendran, Sarath Chandar. (2023)  
**Language Model-In-The-Loop: Data Optimal Approach to Learn-To-Recommend Actions in Text Games**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07687v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated superior performance in language understanding benchmarks. CALM, a popular approach, leverages linguistic priors of LLMs -- GPT-2 -- for action candidate recommendations to improve the performance in text games in Jericho without environment-provided actions. However, CALM adapts GPT-2 with annotated human gameplays and keeps the LLM fixed during the learning of the text based games. In this work, we explore and evaluate updating LLM used for candidate recommendation during the learning of the text based game as well to mitigate the reliance on the human annotated gameplays, which are costly to acquire. We observe that by updating the LLM during learning using carefully selected in-game transitions, we can reduce the dependency on using human annotated game plays for fine-tuning the LLMs. We conducted further analysis to study the transferability of the updated LLMs and observed that transferring in-game trained models to other games did not result in a consistent transfer.

{{</citation>}}


### (8/142) Fuse to Forget: Bias Reduction and Selective Memorization through Model Fusion (Kerem Zaman et al., 2023)

{{<citation>}}

Kerem Zaman, Leshem Choshen, Shashank Srivastava. (2023)  
**Fuse to Forget: Bias Reduction and Selective Memorization through Model Fusion**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.07682v1)  

---


**ABSTRACT**  
Model fusion research aims to aggregate the knowledge of multiple models to enhance performance by combining their weights. In this work, we study the inverse, investigating whether and how can model fusion interfere and reduce unwanted knowledge. We delve into the effects of model fusion on the evolution of learned shortcuts, social biases, and memorization capabilities in fine-tuned language models. Through several experiments covering text classification and generation tasks, our analysis highlights that shared knowledge among models is usually enhanced during model fusion, while unshared knowledge is usually lost or forgotten. Based on this observation, we demonstrate the potential of model fusion as a debiasing tool and showcase its efficacy in addressing privacy concerns associated with language models.

{{</citation>}}


### (9/142) Using Natural Language Explanations to Improve Robustness of In-context Learning for Natural Language Inference (Xuanli He et al., 2023)

{{<citation>}}

Xuanli He, Yuxiang Wu, Oana-Maria Camburu, Pasquale Minervini, Pontus Stenetorp. (2023)  
**Using Natural Language Explanations to Improve Robustness of In-context Learning for Natural Language Inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Natural Language Inference  
[Paper Link](http://arxiv.org/abs/2311.07556v1)  

---


**ABSTRACT**  
Recent studies have demonstrated that large language models (LLMs) excel in diverse tasks through in-context learning (ICL) facilitated by task-specific prompts and examples. However, the existing literature shows that ICL encounters performance deterioration when exposed to adversarial inputs. Enhanced performance has been observed when ICL is augmented with natural language explanations (NLEs) (we refer to it as X-ICL). Thus, this work investigates whether X-ICL can improve the robustness of LLMs on a suite of seven adversarial and challenging natural language inference datasets. Moreover, we introduce a new approach to X-ICL by prompting an LLM (ChatGPT in our case) with few human-generated NLEs to produce further NLEs (we call it ChatGPT few-shot), which we show superior to both ChatGPT zero-shot and human-generated NLEs alone. We evaluate five popular LLMs (GPT3.5-turbo, LLaMa2, Vicuna, Zephyr, Mistral) and show that X-ICL with ChatGPT few-shot yields over 6% improvement over ICL. Furthermore, while prompt selection strategies were previously shown to significantly improve ICL on in-distribution test sets, we show that these strategies do not match the efficacy of the X-ICL paradigm in robustness-oriented evaluations.

{{</citation>}}


### (10/142) A Comprehensive Evaluation of GPT-4V on Knowledge-Intensive Visual Question Answering (Yunxin Li et al., 2023)

{{<citation>}}

Yunxin Li, Longyue Wang, Baotian Hu, Xinyu Chen, Wanqi Zhong, Chenyang Lyu, Min Zhang. (2023)  
**A Comprehensive Evaluation of GPT-4V on Knowledge-Intensive Visual Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Commonsense Knowledge, GPT, GPT-4, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.07536v1)  

---


**ABSTRACT**  
The emergence of multimodal large models (MLMs) has significantly advanced the field of visual understanding, offering remarkable capabilities in the realm of visual question answering (VQA). Yet, the true challenge lies in the domain of knowledge-intensive VQA tasks, which necessitate not just recognition of visual elements, but also a deep comprehension of the visual information in conjunction with a vast repository of learned knowledge. To uncover such capabilities of MLMs, particularly the newly introduced GPT-4V, we provide an in-depth evaluation from three perspectives: 1) Commonsense Knowledge, which assesses how well models can understand visual cues and connect to general knowledge; 2) Fine-grained World Knowledge, which tests the model's skill in reasoning out specific knowledge from images, showcasing their proficiency across various specialized fields; 3) Comprehensive Knowledge with Decision-making Rationales, which examines model's capability to provide logical explanations for its inference, facilitating a deeper analysis from the interpretability perspective. Extensive experiments indicate that GPT-4V achieves SOTA performance on above three tasks. Interestingly, we find that: a) GPT-4V demonstrates enhanced reasoning and explanation when using composite images as few-shot; b) GPT-4V produces severe hallucinations when dealing with world knowledge, highlighting the future need for advancements in this research direction.

{{</citation>}}


### (11/142) It's Not Easy Being Wrong: Evaluating Process of Elimination Reasoning in Large Language Models (Nishant Balepur et al., 2023)

{{<citation>}}

Nishant Balepur, Shramay Palta, Rachel Rudinger. (2023)  
**It's Not Easy Being Wrong: Evaluating Process of Elimination Reasoning in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Falcon, GPT, GPT-3.5, LLaMA, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.07532v1)  

---


**ABSTRACT**  
Chain-of-thought (COT) prompting can help large language models (LLMs) reason toward correct answers, but its efficacy in reasoning toward incorrect answers is unexplored. This strategy of process of elimination (PoE), when used with COT, has the potential to enhance interpretability in tasks like medical diagnoses of exclusion. Thus, we propose PoE with COT, a new task where LLMs must reason toward incorrect options on multiple-choice questions. We evaluate the ability of GPT-3.5, LLaMA-2, and Falcon to perform PoE with COT on 2-choice commonsense and scientific reasoning datasets. We show that PoE consistently underperforms directly choosing the correct answer. The agreement of these strategies is also lower than the self-consistency of each strategy. To study these issues further, we conduct an error analysis and give suggestions for future work.

{{</citation>}}


### (12/142) Multilingual Nonce Dependency Treebanks: Understanding how LLMs represent and process syntactic structure (David Arps et al., 2023)

{{<citation>}}

David Arps, Laura Kallmeyer, Younes Samih, Hassan Sajjad. (2023)  
**Multilingual Nonce Dependency Treebanks: Understanding how LLMs represent and process syntactic structure**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2311.07497v1)  

---


**ABSTRACT**  
We introduce SPUD (Semantically Perturbed Universal Dependencies), a framework for creating nonce treebanks for the multilingual Universal Dependencies (UD) corpora. SPUD data satisfies syntactic argument structure, provides syntactic annotations, and ensures grammaticality via language-specific rules. We create nonce data in Arabic, English, French, German, and Russian, and demonstrate two use cases of SPUD treebanks. First, we investigate the effect of nonce data on word co-occurrence statistics, as measured by perplexity scores of autoregressive (ALM) and masked language models (MLM). We find that ALM scores are significantly more affected by nonce data than MLM scores. Second, we show how nonce data affects the performance of syntactic dependency probes. We replicate the findings of M\"uller-Eberstein et al. (2022) on nonce test data and show that the performance declines on both MLMs and ALMs wrt. original test data. However, a majority of the performance is kept, suggesting that the probe indeed learns syntax independently from semantics.

{{</citation>}}


### (13/142) A Step Closer to Comprehensive Answers: Constrained Multi-Stage Question Decomposition with Large Language Models (Hejing Cao et al., 2023)

{{<citation>}}

Hejing Cao, Zhenwei An, Jiazhan Feng, Kun Xu, Liwei Chen, Dongyan Zhao. (2023)  
**A Step Closer to Comprehensive Answers: Constrained Multi-Stage Question Decomposition with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.07491v1)  

---


**ABSTRACT**  
While large language models exhibit remarkable performance in the Question Answering task, they are susceptible to hallucinations. Challenges arise when these models grapple with understanding multi-hop relations in complex questions or lack the necessary knowledge for a comprehensive response. To address this issue, we introduce the "Decompose-and-Query" framework (D&Q). This framework guides the model to think and utilize external knowledge similar to ReAct, while also restricting its thinking to reliable information, effectively mitigating the risk of hallucinations. Experiments confirm the effectiveness of D&Q: On our ChitChatQA dataset, D&Q does not lose to ChatGPT in 67% of cases; on the HotPotQA question-only setting, D&Q achieved an F1 score of 59.6%. Our code is available at https://github.com/alkaidpku/DQ-ToolQA.

{{</citation>}}


### (14/142) Psychometric Predictive Power of Large Language Models (Tatsuki Kuribayashi et al., 2023)

{{<citation>}}

Tatsuki Kuribayashi, Yohei Oseki, Timothy Baldwin. (2023)  
**Psychometric Predictive Power of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07484v1)  

---


**ABSTRACT**  
Next-word probabilities from language models have been shown to successfully simulate human reading behavior. Building on this, we show that, interestingly, instruction-tuned large language models (LLMs) yield worse psychometric predictive power (PPP) for human reading behavior than base LLMs with equivalent perplexities. In other words, instruction tuning, which helps LLMs provide human-preferred responses, does not always make them human-like from the computational psycholinguistics perspective. In addition, we explore prompting methodologies in simulating human reading behavior with LLMs, showing that prompts reflecting a particular linguistic hypothesis lead LLMs to exhibit better PPP but are still worse than base LLMs. These highlight that recent instruction tuning and prompting do not offer better estimates than direct probability measurements from base LLMs in cognitive modeling.

{{</citation>}}


### (15/142) Finding and Editing Multi-Modal Neurons in Pre-Trained Transformer (Haowen Pan et al., 2023)

{{<citation>}}

Haowen Pan, Yixin Cao, Xiaozhi Wang, Xun Yang. (2023)  
**Finding and Editing Multi-Modal Neurons in Pre-Trained Transformer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.07470v1)  

---


**ABSTRACT**  
Multi-modal large language models (LLM) have achieved powerful capabilities for visual semantic understanding in recent years. However, little is known about how LLMs comprehend visual information and interpret different modalities of features. In this paper, we propose a new method for identifying multi-modal neurons in transformer-based multi-modal LLMs. Through a series of experiments, We highlight three critical properties of multi-modal neurons by four well-designed quantitative evaluation metrics. Furthermore, we introduce a knowledge editing method based on the identified multi-modal neurons, for modifying a specific token to another designative token. We hope our findings can inspire further explanatory researches on understanding mechanisms of multi-modal LLMs.

{{</citation>}}


### (16/142) InCA: Rethinking In-Car Conversational System Assessment Leveraging Large Language Models (Ken E. Friedl et al., 2023)

{{<citation>}}

Ken E. Friedl, Abbas Goher Khan, Soumya Ranjan Sahoo, Md Rashad Al Hasan Rony, Jana Germies, Christian Süß. (2023)  
**InCA: Rethinking In-Car Conversational System Assessment Leveraging Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.07469v1)  

---


**ABSTRACT**  
The assessment of advanced generative large language models (LLMs) poses a significant challenge, given their heightened complexity in recent developments. Furthermore, evaluating the performance of LLM-based applications in various industries, as indicated by Key Performance Indicators (KPIs), is a complex undertaking. This task necessitates a profound understanding of industry use cases and the anticipated system behavior. Within the context of the automotive industry, existing evaluation metrics prove inadequate for assessing in-car conversational question answering (ConvQA) systems. The unique demands of these systems, where answers may relate to driver or car safety and are confined within the car domain, highlight the limitations of current metrics. To address these challenges, this paper introduces a set of KPIs tailored for evaluating the performance of in-car ConvQA systems, along with datasets specifically designed for these KPIs. A preliminary and comprehensive empirical evaluation substantiates the efficacy of our proposed approach. Furthermore, we investigate the impact of employing varied personas in prompts and found that it enhances the model's capacity to simulate diverse viewpoints in assessments, mirroring how individuals with different backgrounds perceive a topic.

{{</citation>}}


### (17/142) Are We Falling in a Middle-Intelligence Trap? An Analysis and Mitigation of the Reversal Curse (Ang Lv et al., 2023)

{{<citation>}}

Ang Lv, Kaiyi Zhang, Shufang Xie, Quan Tu, Yuhan Chen, Ji-Rong Wen, Rui Yan. (2023)  
**Are We Falling in a Middle-Intelligence Trap? An Analysis and Mitigation of the Reversal Curse**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2311.07468v1)  

---


**ABSTRACT**  
Recent studies have highlighted a phenomenon in large language models (LLMs) known as "the reversal curse," in which the order of knowledge entities in the training data biases the models' comprehension. For example, if a model is trained on sentences where entity A consistently appears before entity B, it can respond to queries about A by providing B. However, it may encounter confusion when presented with questions concerning B. We contend that the reversal curse is partially a result of specific model training objectives, particularly evident in the prevalent use of the next-token prediction within most causal language models. For the next-token prediction, models solely focus on a token's preceding context, resulting in a restricted comprehension of the input. In contrast, we illustrate that the GLM, trained using the autoregressive blank infilling objective where tokens to be predicted have access to the entire context, exhibits better resilience against the reversal curse. We propose a novel training method, BIdirectional Casual language modeling Optimization (BICO), designed to mitigate the reversal curse when fine-tuning pretrained causal language models on new data. BICO modifies the causal attention mechanism to function bidirectionally and employs a mask denoising optimization. In the task designed to assess the reversal curse, our approach improves Llama's accuracy from the original 0% to around 70%. We hope that more attention can be focused on exploring and addressing these inherent weaknesses of the current LLMs, in order to achieve a higher level of intelligence.

{{</citation>}}


### (18/142) On Measuring Faithfulness of Natural Language Explanations (Letitia Parcalabescu et al., 2023)

{{<citation>}}

Letitia Parcalabescu, Anette Frank. (2023)  
**On Measuring Faithfulness of Natural Language Explanations**  

---
Primary Category: cs.CL  
Categories: 68Txx, I-2-7; I-2-10, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.07466v1)  

---


**ABSTRACT**  
Large language models (LLMs) can explain their own predictions, through post-hoc or Chain-of-Thought (CoT) explanations. However the LLM could make up reasonably sounding explanations that are unfaithful to its underlying reasoning. Recent work has designed tests that aim to judge the faithfulness of either post-hoc or CoT explanations. In this paper we argue that existing faithfulness tests are not actually measuring faithfulness in terms of the models' inner workings, but only evaluate their self-consistency on the output level. The aims of our work are two-fold. i) We aim to clarify the status of existing faithfulness tests in terms of model explainability, characterising them as self-consistency tests instead. This assessment we underline by constructing a Comparative Consistency Bank for self-consistency tests that for the first time compares existing tests on a common suite of 11 open-source LLMs and 5 datasets -- including ii) our own proposed self-consistency measure CC-SHAP. CC-SHAP is a new fine-grained measure (not test) of LLM self-consistency that compares a model's input contributions to answer prediction and generated explanation. With CC-SHAP, we aim to take a step further towards measuring faithfulness with a more interpretable and fine-grained method. Code available at \url{https://github.com/Heidelberg-NLP/CC-SHAP}

{{</citation>}}


### (19/142) MEGAVERSE: Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks (Sanchit Ahuja et al., 2023)

{{<citation>}}

Sanchit Ahuja, Divyanshu Aggarwal, Varun Gumma, Ishaan Watts, Ashutosh Sathe, Millicent Ochieng, Rishav Hada, Prachi Jain, Maxamed Axmed, Kalika Bali, Sunayana Sitaram. (2023)  
**MEGAVERSE: Benchmarking Large Language Models Across Languages, Modalities, Models and Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, NLP, Natural Language Processing, PaLM  
[Paper Link](http://arxiv.org/abs/2311.07463v1)  

---


**ABSTRACT**  
Recently, there has been a rapid advancement in research on Large Language Models (LLMs), resulting in significant progress in several Natural Language Processing (NLP) tasks. Consequently, there has been a surge in LLM evaluation research to comprehend the models' capabilities and limitations. However, much of this research has been confined to the English language, leaving LLM building and evaluation for non-English languages relatively unexplored. There has been an introduction of several new LLMs, necessitating their evaluation on non-English languages. This study aims to expand our MEGA benchmarking suite by including six new datasets to form the MEGAVERSE benchmark. The benchmark comprises 22 datasets covering 81 languages, including low-resource African languages. We evaluate several state-of-the-art LLMs like GPT-3.5-Turbo, GPT4, PaLM2, and Llama2 on the MEGAVERSE datasets. Additionally, we include two multimodal datasets in the benchmark and assess the performance of the LLaVa-v1.5 model. Our experiments suggest that GPT4 and PaLM2 outperform the Llama models on various tasks, notably on low-resource languages, with GPT4 outperforming PaLM2 on more datasets than vice versa. However, issues such as data contamination must be addressed to obtain an accurate assessment of LLM performance on non-English languages.

{{</citation>}}


### (20/142) ChartCheck: An Evidence-Based Fact-Checking Dataset over Real-World Chart Images (Mubashara Akhtar et al., 2023)

{{<citation>}}

Mubashara Akhtar, Nikesh Subedi, Vivek Gupta, Sahar Tahmasebi, Oana Cocarascu, Elena Simperl. (2023)  
**ChartCheck: An Evidence-Based Fact-Checking Dataset over Real-World Chart Images**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Fact-Checking  
[Paper Link](http://arxiv.org/abs/2311.07453v1)  

---


**ABSTRACT**  
Data visualizations are common in the real-world. We often use them in data sources such as scientific documents, news articles, textbooks, and social media to summarize key information in a visual form. Charts can also mislead its audience by communicating false information or biasing them towards a specific agenda. Verifying claims against charts is not a straightforward process. It requires analyzing both the text and visual components of the chart, considering characteristics such as colors, positions, and orientations. Moreover, to determine if a claim is supported by the chart content often requires different types of reasoning. To address this challenge, we introduce ChartCheck, a novel dataset for fact-checking against chart images. ChartCheck is the first large-scale dataset with 1.7k real-world charts and 10.5k human-written claims and explanations. We evaluated the dataset on state-of-the-art models and achieved an accuracy of 73.9 in the finetuned setting. Additionally, we identified chart characteristics and reasoning types that challenge the models.

{{</citation>}}


### (21/142) Think Before You Speak: Cultivating Communication Skills of Large Language Models via Inner Monologue (Junkai Zhou et al., 2023)

{{<citation>}}

Junkai Zhou, Liang Pang, Huawei Shen, Xueqi Cheng. (2023)  
**Think Before You Speak: Cultivating Communication Skills of Large Language Models via Inner Monologue**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07445v1)  

---


**ABSTRACT**  
The emergence of large language models (LLMs) further improves the capabilities of open-domain dialogue systems and can generate fluent, coherent, and diverse responses. However, LLMs still lack an important ability: communication skills, which makes them more like information seeking tools than anthropomorphic chatbots. To make LLMs more anthropomorphic and proactive during the conversation, we add five communication skills to the response generation process: topic transition, proactively asking questions, concept guidance, empathy, and summarising often. The addition of communication skills increases the interest of users in the conversation and attracts them to chat for longer. To enable LLMs better understand and use communication skills, we design and add the inner monologue to LLMs. The complete process is achieved through prompt engineering and in-context learning. To evaluate communication skills, we construct a benchmark named Cskills for evaluating various communication skills, which can also more comprehensively evaluate the dialogue generation ability of the model. Experimental results show that the proposed CSIM strategy improves the backbone models and outperforms the baselines in both automatic and human evaluations.

{{</citation>}}


### (22/142) Investigating Multi-Pivot Ensembling with Massively Multilingual Machine Translation Models (Alireza Mohammadshahi et al., 2023)

{{<citation>}}

Alireza Mohammadshahi, Jannis Vamvas, Rico Sennrich. (2023)  
**Investigating Multi-Pivot Ensembling with Massively Multilingual Machine Translation Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Machine Translation, Multilingual  
[Paper Link](http://arxiv.org/abs/2311.07439v2)  

---


**ABSTRACT**  
Massively multilingual machine translation models allow for the translation of a large number of languages with a single model, but have limited performance on low- and very-low-resource translation directions. Pivoting via high-resource languages remains a strong strategy for low-resource directions, and in this paper we revisit ways of pivoting through multiple languages. Previous work has used a simple averaging of probability distributions from multiple paths, but we find that this performs worse than using a single pivot, and exacerbates the hallucination problem because the same hallucinations can be probable across different paths. As an alternative, we propose MaxEns, a combination strategy that is biased towards the most confident predictions, hypothesising that confident predictions are less prone to be hallucinations. We evaluate different strategies on the FLORES benchmark for 20 low-resource language directions, demonstrating that MaxEns improves translation quality for low-resource languages while reducing hallucination in translations, compared to both direct translation and an averaging approach. On average, multi-pivot strategies still lag behind using English as a single pivot language, raising the question of how to identify the best pivoting strategy for a given translation direction.

{{</citation>}}


### (23/142) Controlled Text Generation for Black-box Language Models via Score-based Progressive Editor (Sangwon Yu et al., 2023)

{{<citation>}}

Sangwon Yu, Changmin Lee, Hojin Lee, Sungroh Yoon. (2023)  
**Controlled Text Generation for Black-box Language Models via Score-based Progressive Editor**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2311.07430v1)  

---


**ABSTRACT**  
Despite recent progress in language models, generating constrained text for specific domains remains a challenge, particularly when utilizing black-box models that lack domain-specific knowledge. In this paper, we introduce ScoPE (Score-based Progressive Editor) generation, a novel approach for controlled text generation for black-box language models. We employ ScoPE to facilitate text generation in the target domain by integrating it with language models through a cascading approach. Trained to enhance the target domain score of the edited text, ScoPE progressively edits intermediate output discrete tokens to align with the target attributes throughout the auto-regressive generation process of the language model. This iterative process guides subsequent steps to produce desired output texts for the target domain. Our experimental results on diverse controlled generations demonstrate that ScoPE effectively facilitates controlled text generation for black-box language models in both in-domain and out-of-domain conditions, which is challenging for existing methods.

{{</citation>}}


### (24/142) Hallucination Augmented Recitations for Language Models (Abdullatif Köksal et al., 2023)

{{<citation>}}

Abdullatif Köksal, Renat Aksitov, Chung-Ching Chang. (2023)  
**Hallucination Augmented Recitations for Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.07424v1)  

---


**ABSTRACT**  
Attribution is a key concept in large language models (LLMs) as it enables control over information sources and enhances the factuality of LLMs. While existing approaches utilize open book question answering to improve attribution, factual datasets may reward language models to recall facts that they already know from their pretraining data, not attribution. In contrast, counterfactual open book QA datasets would further improve attribution because the answer could only be grounded in the given text. We propose Hallucination Augmented Recitations (HAR) for creating counterfactual datasets by utilizing hallucination in LLMs to improve attribution. For open book QA as a case study, we demonstrate that models finetuned with our counterfactual datasets improve text grounding, leading to better open book QA performance, with up to an 8.0% increase in F1 score. Our counterfactual dataset leads to significantly better performance than using humanannotated factual datasets, even with 4x smaller datasets and 4x smaller models. We observe that improvements are consistent across various model sizes and datasets, including multi-hop, biomedical, and adversarial QA datasets.

{{</citation>}}


### (25/142) Speech-based Slot Filling using Large Language Models (Guangzhi Sun et al., 2023)

{{<citation>}}

Guangzhi Sun, Shutong Feng, Dongcheng Jiang, Chao Zhang, Milica Gašić, Philip C. Woodland. (2023)  
**Speech-based Slot Filling using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: GPT, GPT-3.5, GPT-4, LLaMA, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2311.07418v1)  

---


**ABSTRACT**  
Recently, advancements in large language models (LLMs) have shown an unprecedented ability across various language tasks. This paper investigates the potential application of LLMs to slot filling with noisy ASR transcriptions, via both in-context learning and task-specific fine-tuning. Dedicated prompt designs and fine-tuning approaches are proposed to improve the robustness of LLMs for slot filling with noisy ASR transcriptions. Moreover, a linearised knowledge injection (LKI) scheme is also proposed to integrate dynamic external knowledge into LLMs. Experiments were performed on SLURP to quantify the performance of LLMs, including GPT-3.5-turbo, GPT-4, LLaMA-13B and Vicuna-13B (v1.1 and v1.5) with different ASR error rates. The use of the proposed fine-tuning together with the LKI scheme for LLaMA-13B achieved an 8.3% absolute SLU-F1 improvement compared to the strong Flan-T5-base baseline system on a limited data setup.

{{</citation>}}


### (26/142) An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation (Junyang Wang et al., 2023)

{{<citation>}}

Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Ming Yan, Ji Zhang, Jitao Sang. (2023)  
**An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07397v1)  

---


**ABSTRACT**  
Despite making significant progress in multi-modal tasks, current Multi-modal Large Language Models (MLLMs) encounter the significant challenge of hallucination, which may lead to harmful consequences. Therefore, evaluating MLLMs' hallucinations is becoming increasingly important in model improvement and practical application deployment. Previous works are limited in high evaluation costs (e.g., relying on humans or advanced LLMs) and insufficient evaluation dimensions (e.g., types of hallucination and task). In this paper, we propose an LLM-free multi-dimensional benchmark AMBER, which can be used to evaluate both generative task and discriminative task including object existence, object attribute and object relation hallucination. Based on AMBER, we design a low-cost and efficient evaluation pipeline. Additionally, we conduct a comprehensive evaluation and detailed analysis of mainstream MLLMs including GPT-4V(ision), and also give guideline suggestions for mitigating hallucinations. The data and code of AMBER are available at https://github.com/junyangwang0410/AMBER.

{{</citation>}}


### (27/142) Assessing Logical Puzzle Solving in Large Language Models: Insights from a Minesweeper Case Study (Yinghao Li et al., 2023)

{{<citation>}}

Yinghao Li, Haorui Wang, Chao Zhang. (2023)  
**Assessing Logical Puzzle Solving in Large Language Models: Insights from a Minesweeper Case Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07387v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown remarkable proficiency in language understanding and have been successfully applied to a variety of real-world tasks through task-specific fine-tuning or prompt engineering. Despite these advancements, it remains an open question whether LLMs are fundamentally capable of reasoning and planning, or if they primarily rely on recalling and synthesizing information from their training data. In our research, we introduce a novel task -- Minesweeper -- specifically designed in a format unfamiliar to LLMs and absent from their training datasets. This task challenges LLMs to identify the locations of mines based on numerical clues provided by adjacent opened cells. Successfully completing this task requires an understanding of each cell's state, discerning spatial relationships between the clues and mines, and strategizing actions based on logical deductions drawn from the arrangement of the cells. Our experiments, including trials with the advanced GPT-4 model, indicate that while LLMs possess the foundational abilities required for this task, they struggle to integrate these into a coherent, multi-step logical reasoning process needed to solve Minesweeper. These findings highlight the need for further research to understand and nature of reasoning capabilities in LLMs under similar circumstances, and to explore pathways towards more sophisticated AI reasoning and planning models.

{{</citation>}}


### (28/142) LM-Polygraph: Uncertainty Estimation for Language Models (Ekaterina Fadeeva et al., 2023)

{{<citation>}}

Ekaterina Fadeeva, Roman Vashurin, Akim Tsvigun, Artem Vazhentsev, Sergey Petrakov, Kirill Fedyanin, Daniil Vasilev, Elizaveta Goncharova, Alexander Panchenko, Maxim Panov, Timothy Baldwin, Artem Shelmanov. (2023)  
**LM-Polygraph: Uncertainty Estimation for Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BLOOM, ChatGPT, GPT, GPT-4, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07383v1)  

---


**ABSTRACT**  
Recent advancements in the capabilities of large language models (LLMs) have paved the way for a myriad of groundbreaking applications in various fields. However, a significant challenge arises as these models often "hallucinate", i.e., fabricate facts without providing users an apparent means to discern the veracity of their statements. Uncertainty estimation (UE) methods are one path to safer, more responsible, and more effective use of LLMs. However, to date, research on UE methods for LLMs has been focused primarily on theoretical rather than engineering contributions. In this work, we tackle this issue by introducing LM-Polygraph, a framework with implementations of a battery of state-of-the-art UE methods for LLMs in text generation tasks, with unified program interfaces in Python. Additionally, it introduces an extendable benchmark for consistent evaluation of UE techniques by researchers, and a demo web application that enriches the standard chat dialog with confidence scores, empowering end-users to discern unreliable responses. LM-Polygraph is compatible with the most recent LLMs, including BLOOMz, LLaMA-2, ChatGPT, and GPT-4, and is designed to support future releases of similarly-styled LMs.

{{</citation>}}


### (29/142) Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision (Seongyun Lee et al., 2023)

{{<citation>}}

Seongyun Lee, Sue Hyun Park, Yongrae Jo, Minjoon Seo. (2023)  
**Volcano: Mitigating Multimodal Hallucination through Self-Feedback Guided Revision**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07362v2)  

---


**ABSTRACT**  
Large multimodal models (LMMs) suffer from multimodal hallucination, where they provide incorrect responses misaligned with the given visual information. Recent works have conjectured that one of the reasons behind multimodal hallucination might be due to the vision encoder failing to ground on the image properly. To mitigate this issue, we propose a novel approach that leverages self-feedback as visual cues. Building on this approach, we introduce Volcano, a multimodal self-feedback guided revision model. Volcano generates natural language feedback to its initial response based on the provided visual information and utilizes this feedback to self-revise its initial response. Volcano effectively reduces multimodal hallucination and achieves state-of-the-art on MMHal-Bench, POPE, and GAVIE. It also improves on general multimodal abilities and outperforms previous models on MM-Vet and MMBench. Through a qualitative analysis, we show that Volcano's feedback is properly grounded on the image than the initial response. This indicates that Volcano can provide itself with richer visual information, helping alleviate multimodal hallucination. We publicly release Volcano models of 7B and 13B sizes along with the data and code at https://github.com/kaistAI/Volcano.

{{</citation>}}


### (30/142) The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4 (Microsoft Research AI4Science et al., 2023)

{{<citation>}}

Microsoft Research AI4Science, Microsoft Azure Quantum. (2023)  
**The Impact of Large Language Models on Scientific Discovery: a Preliminary Study using GPT-4**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07361v1)  

---


**ABSTRACT**  
In recent years, groundbreaking advancements in natural language processing have culminated in the emergence of powerful large language models (LLMs), which have showcased remarkable capabilities across a vast array of domains, including the understanding, generation, and translation of natural language, and even tasks that extend beyond language processing. In this report, we delve into the performance of LLMs within the context of scientific discovery, focusing on GPT-4, the state-of-the-art language model. Our investigation spans a diverse range of scientific areas encompassing drug discovery, biology, computational chemistry (density functional theory (DFT) and molecular dynamics (MD)), materials design, and partial differential equations (PDE). Evaluating GPT-4 on scientific tasks is crucial for uncovering its potential across various research domains, validating its domain-specific expertise, accelerating scientific progress, optimizing resource allocation, guiding future model development, and fostering interdisciplinary research. Our exploration methodology primarily consists of expert-driven case assessments, which offer qualitative insights into the model's comprehension of intricate scientific concepts and relationships, and occasionally benchmark testing, which quantitatively evaluates the model's capacity to solve well-defined domain-specific problems. Our preliminary exploration indicates that GPT-4 exhibits promising potential for a variety of scientific applications, demonstrating its aptitude for handling complex problem-solving and knowledge integration tasks. Broadly speaking, we evaluate GPT-4's knowledge base, scientific understanding, scientific numerical calculation abilities, and various scientific prediction capabilities.

{{</citation>}}


### (31/142) Semi-automatic Data Enhancement for Document-Level Relation Extraction with Distant Supervision from Large Language Models (Junpeng Li et al., 2023)

{{<citation>}}

Junpeng Li, Zixia Jia, Zilong Zheng. (2023)  
**Semi-automatic Data Enhancement for Document-Level Relation Extraction with Distant Supervision from Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLI, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2311.07314v1)  

---


**ABSTRACT**  
Document-level Relation Extraction (DocRE), which aims to extract relations from a long context, is a critical challenge in achieving fine-grained structural comprehension and generating interpretable document representations. Inspired by recent advances in in-context learning capabilities emergent from large language models (LLMs), such as ChatGPT, we aim to design an automated annotation method for DocRE with minimum human effort. Unfortunately, vanilla in-context learning is infeasible for document-level relation extraction due to the plenty of predefined fine-grained relation types and the uncontrolled generations of LLMs. To tackle this issue, we propose a method integrating a large language model (LLM) and a natural language inference (NLI) module to generate relation triples, thereby augmenting document-level relation datasets. We demonstrate the effectiveness of our approach by introducing an enhanced dataset known as DocGNRE, which excels in re-annotating numerous long-tail relation types. We are confident that our method holds the potential for broader applications in domain-specific relation type definitions and offers tangible benefits in advancing generalized language semantic comprehension.

{{</citation>}}


### (32/142) Do large language models and humans have similar behaviors in causal inference with script knowledge? (Xudong Hong et al., 2023)

{{<citation>}}

Xudong Hong, Margarita Ryzhova, Daniel Adrian Biondi, Vera Demberg. (2023)  
**Do large language models and humans have similar behaviors in causal inference with script knowledge?**  

---
Primary Category: cs.CL  
Categories: I-2-7; I-2-0, cs-AI, cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.07311v1)  

---


**ABSTRACT**  
Recently, large pre-trained language models (LLMs) have demonstrated superior language understanding abilities, including zero-shot causal reasoning. However, it is unclear to what extent their capabilities are similar to human ones. We here study the processing of an event $B$ in a script-based story, which causally depends on a previous event $A$. In our manipulation, event $A$ is stated, negated, or omitted in an earlier section of the text. We first conducted a self-paced reading experiment, which showed that humans exhibit significantly longer reading times when causal conflicts exist ($\neg A \rightarrow B$) than under logical conditions ($A \rightarrow B$). However, reading times remain similar when cause A is not explicitly mentioned, indicating that humans can easily infer event B from their script knowledge. We then tested a variety of LLMs on the same data to check to what extent the models replicate human behavior. Our experiments show that 1) only recent LLMs, like GPT-3 or Vicuna, correlate with human behavior in the $\neg A \rightarrow B$ condition. 2) Despite this correlation, all models still fail to predict that $nil \rightarrow B$ is less surprising than $\neg A \rightarrow B$, indicating that LLMs still have difficulties integrating script knowledge. Our code and collected data set are available at https://github.com/tony-hong/causal-script.

{{</citation>}}


### (33/142) BIDRN: A Method of Bidirectional Recurrent Neural Network for Sentiment Analysis (Dr. D Muthusankar et al., 2023)

{{<citation>}}

Dr. D Muthusankar, Dr. P Kaladevi, Dr. V R Sadasivam, R Praveen. (2023)  
**BIDRN: A Method of Bidirectional Recurrent Neural Network for Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2311.07296v1)  

---


**ABSTRACT**  
Text mining research has grown in importance in recent years due to the tremendous increase in the volume of unstructured textual data. This has resulted in immense potential as well as obstacles in the sector, which may be efficiently addressed with adequate analytical and study methods. Deep Bidirectional Recurrent Neural Networks are used in this study to analyze sentiment. The method is categorized as sentiment polarity analysis because it may generate a dataset with sentiment labels. This dataset can be used to train and evaluate sentiment analysis models capable of extracting impartial opinions. This paper describes the Sentiment Analysis-Deep Bidirectional Recurrent Neural Networks (SA-BDRNN) Scheme, which seeks to overcome the challenges and maximize the potential of text mining in the context of Big Data. The current study proposes a SA-DBRNN Scheme that attempts to give a systematic framework for sentiment analysis in the context of student input on institution choice. The purpose of this study is to compare the effectiveness of the proposed SA- DBRNN Scheme to existing frameworks to establish a robust deep neural network that might serve as an adequate classification model in the field of sentiment analysis.

{{</citation>}}


### (34/142) In Search of the Long-Tail: Systematic Generation of Long-Tail Knowledge via Logical Rule Guided Search (Huihan Li et al., 2023)

{{<citation>}}

Huihan Li, Yuting Ning, Zeyi Liao, Siyuan Wang, Xiang Lorraine Li, Ximing Lu, Faeze Brahman, Wenting Zhao, Yejin Choi, Xiang Ren. (2023)  
**In Search of the Long-Tail: Systematic Generation of Long-Tail Knowledge via Logical Rule Guided Search**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.07237v1)  

---


**ABSTRACT**  
Since large language models have approached human-level performance on many tasks, it has become increasingly harder for researchers to find tasks that are still challenging to the models. Failure cases usually come from the long-tail distribution - data that an oracle language model could assign a probability on the lower end of its distribution. Current methodology such as prompt engineering or crowdsourcing are insufficient for creating long-tail examples because humans are constrained by cognitive bias. We propose a Logic-Induced-Knowledge-Search (LINK) framework for systematically generating long-tail knowledge statements. Grounded by a symbolic rule, we search for long-tail values for each variable of the rule by first prompting a LLM, then verifying the correctness of the values with a critic, and lastly pushing for the long-tail distribution with a reranker. With this framework we construct a dataset, Logic-Induced-Long-Tail (LINT), consisting of 200 symbolic rules and 50K knowledge statements spanning across four domains. Human annotations find that 84% of the statements in LINT are factually correct. In contrast, ChatGPT and GPT4 struggle with directly generating long-tail statements under the guidance of logic rules, each only getting 56% and 78% of their statements correct. Moreover, their "long-tail" generations in fact fall into the higher likelihood range, and thus are not really long-tail. Our findings suggest that LINK is effective for generating data in the long-tail distribution while enforcing quality. LINT can be useful for systematically evaluating LLMs' capabilities in the long-tail distribution. We challenge the models with a simple entailment classification task using samples from LINT. We find that ChatGPT and GPT4's capability in identifying incorrect knowledge drop by ~3% in the long-tail distribution compared to head distribution.

{{</citation>}}


### (35/142) Coffee: Boost Your Code LLMs by Fixing Bugs with Feedback (Seungjun Moon et al., 2023)

{{<citation>}}

Seungjun Moon, Yongho Song, Hyungjoo Chae, Dongjin Kang, Taeyoon Kwon, Kai Tzu-iunn Ong, Seung-won Hwang, Jinyoung Yeo. (2023)  
**Coffee: Boost Your Code LLMs by Fixing Bugs with Feedback**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SE, cs.CL  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.07215v1)  

---


**ABSTRACT**  
Code editing is an essential step towards reliable program synthesis to automatically correct critical errors generated from code LLMs. Recent studies have demonstrated that closed-source LLMs (i.e., ChatGPT and GPT-4) are capable of generating corrective feedback to edit erroneous inputs. However, it remains challenging for open-source code LLMs to generate feedback for code editing, since these models tend to adhere to the superficial formats of feedback and provide feedback with misleading information. Hence, the focus of our work is to leverage open-source code LLMs to generate helpful feedback with correct guidance for code editing. To this end, we present Coffee, a collected dataset specifically designed for code fixing with feedback. Using this dataset, we construct CoffeePots, a framework for COde Fixing with FEEdback via Preference-Optimized Tuning and Selection. The proposed framework aims to automatically generate helpful feedback for code editing while minimizing the potential risk of superficial feedback. The combination of Coffee and CoffeePots marks a significant advancement, achieving state-of-the-art performance on HumanEvalFix benchmark. Codes and model checkpoints are publicly available at https://github.com/Lune-Blue/COFFEE.

{{</citation>}}


### (36/142) Exploring the Dialogue Comprehension Ability of Large Language Models (Shuaijie She et al., 2023)

{{<citation>}}

Shuaijie She, Shujian Huang, Xingyun Wang, Yanke Zhou, Jiajun Chen. (2023)  
**Exploring the Dialogue Comprehension Ability of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.07194v1)  

---


**ABSTRACT**  
The recent emergence of large language models (LLMs) have attracted considerable attention. LLMs may interact with users in the form of dialogue and generate responses following their instructions, which naturally require dialogue comprehension abilities. Without correct comprehension of the dialogue, the model may inevitably generate incorrect responses. However, dialogue comprehension is a general language ability which is hard to be evaluated directly. In this work, we propose to perform the evaluation with the help of the dialogue summarization task. Beside evaluating and analyzing the dialogue summarization performance (DIAC-Sum), we also derive factual questions from the generated summaries and use them as a more flexible measurement of dialogue comprehension (DIAC-FactQA). Our evaluation shows that, on average, 27% of the summaries generated by LLMs contain factual inconsistency. Even ChatGPT, the strongest evaluated model, has such errors in 16% of its summaries. For answering the factual questions, which is more challenging, the average accuracy of all evaluated LLMs is only 62.8%. Both results indicate serious deficiencies. Detailed analysis shows that the understanding of subject/object of the conversation is still the most challenging problem for LLMs. Furthermore, to stimulate and enhance the dialogue comprehension ability of LLMs, we propose a fine-tuning paradigm with auto-constructed multi-task data. The experimental results demonstrate that our method achieved an accuracy improvement of 8.9% on DIAC-FactQA.

{{</citation>}}


### (37/142) VerityMath: Advancing Mathematical Reasoning by Self-Verification Through Unit Consistency (Vernon Toh et al., 2023)

{{<citation>}}

Vernon Toh, Ratish Puduppully, Nancy F. Chen. (2023)  
**VerityMath: Advancing Mathematical Reasoning by Self-Verification Through Unit Consistency**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-PL, cs.CL  
Keywords: AI, GPT, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.07172v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) combined with program-based solving techniques are increasingly demonstrating proficiency in mathematical reasoning. However, such progress is mostly demonstrated in closed-source models such as OpenAI-GPT4 and Claude. In this paper, we seek to study the performance of strong open-source LLMs. Specifically, we analyze the outputs of Code Llama (7B) when applied to math word problems. We identify a category of problems that pose a challenge for the model, particularly those involving quantities that span multiple types or units. To address this issue, we propose a systematic approach by defining units for each quantity and ensuring the consistency of these units during mathematical operations. We developed Unit Consistency Programs (UCPs), an annotated dataset of math word problems, each paired with programs that contain unit specifications and unit verification routines. Finally, we finetune the Code Llama (7B) model with UCPs to produce VerityMath and present our preliminary findings.

{{</citation>}}


### (38/142) calamanCy: A Tagalog Natural Language Processing Toolkit (Lester James V. Miranda, 2023)

{{<citation>}}

Lester James V. Miranda. (2023)  
**calamanCy: A Tagalog Natural Language Processing Toolkit**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.07171v1)  

---


**ABSTRACT**  
We introduce calamanCy, an open-source toolkit for constructing natural language processing (NLP) pipelines for Tagalog. It is built on top of spaCy, enabling easy experimentation and integration with other frameworks. calamanCy addresses the development gap by providing a consistent API for building NLP applications and offering general-purpose multitask models with out-of-the-box support for dependency parsing, parts-of-speech (POS) tagging, and named entity recognition (NER). calamanCy aims to accelerate the progress of Tagalog NLP by consolidating disjointed resources in a unified framework. The calamanCy toolkit is available on GitHub: https://github.com/ljvmiranda921/calamanCy.

{{</citation>}}


### (39/142) STEER: Unified Style Transfer with Expert Reinforcement (Skyler Hallinan et al., 2023)

{{<citation>}}

Skyler Hallinan, Faeze Brahman, Ximing Lu, Jaehun Jung, Sean Welleck, Yejin Choi. (2023)  
**STEER: Unified Style Transfer with Expert Reinforcement**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Style Transfer  
[Paper Link](http://arxiv.org/abs/2311.07167v1)  

---


**ABSTRACT**  
While text style transfer has many applications across natural language processing, the core premise of transferring from a single source style is unrealistic in a real-world setting. In this work, we focus on arbitrary style transfer: rewriting a text from an arbitrary, unknown style to a target style.   We propose STEER: Unified Style Transfer with Expert Reinforcement, a unified frame-work developed to overcome the challenge of limited parallel data for style transfer. STEER involves automatically generating a corpus of style-transfer pairs using a product of experts during decoding. The generated offline data is then used to pre-train an initial policy before switching to online, off-policy reinforcement learning for further improvements via fine-grained reward signals. STEER is unified and can transfer to multiple target styles from an arbitrary, unknown source style, making it particularly flexible and efficient.   Experimental results on a challenging dataset with text from a diverse set of styles demonstrate state-of-the-art results compared to competitive baselines. Remarkably, STEER outperforms the 175B parameter instruction-tuned GPT-3 on overall style transfer quality, despite being 226 times smaller in size. We also show STEER is robust, maintaining its style transfer capabilities on out-of-domain data, and surpassing nearly all baselines across various styles. The success of our method highlights the potential of RL algorithms when augmented with controllable decoding to overcome the challenge of limited data supervision.

{{</citation>}}


### (40/142) Developing a Named Entity Recognition Dataset for Tagalog (Lester James V. Miranda, 2023)

{{<citation>}}

Lester James V. Miranda. (2023)  
**Developing a Named Entity Recognition Dataset for Tagalog**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, NLP, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2311.07161v1)  

---


**ABSTRACT**  
We present the development of a Named Entity Recognition (NER) dataset for Tagalog. This corpus helps fill the resource gap present in Philippine languages today, where NER resources are scarce. The texts were obtained from a pretraining corpora containing news reports, and were labeled by native speakers in an iterative fashion. The resulting dataset contains ~7.8k documents across three entity types: Person, Organization, and Location. The inter-annotator agreement, as measured by Cohen's $\kappa$, is 0.81. We also conducted extensive empirical evaluation of state-of-the-art methods across supervised and transfer learning settings. Finally, we released the data and processing code publicly to inspire future work on Tagalog NLP.

{{</citation>}}


### (41/142) WaterBench: Towards Holistic Evaluation of Watermarks for Large Language Models (Shangqing Tu et al., 2023)

{{<citation>}}

Shangqing Tu, Yuliang Sun, Yushi Bai, Jifan Yu, Lei Hou, Juanzi Li. (2023)  
**WaterBench: Towards Holistic Evaluation of Watermarks for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07138v1)  

---


**ABSTRACT**  
To mitigate the potential misuse of large language models (LLMs), recent research has developed watermarking algorithms, which restrict the generation process to leave an invisible trace for watermark detection. Due to the two-stage nature of the task, most studies evaluate the generation and detection separately, thereby presenting a challenge in unbiased, thorough, and applicable evaluations. In this paper, we introduce WaterBench, the first comprehensive benchmark for LLM watermarks, in which we design three crucial factors: (1) For \textbf{benchmarking procedure}, to ensure an apples-to-apples comparison, we first adjust each watermarking method's hyper-parameter to reach the same watermarking strength, then jointly evaluate their generation and detection performance. (2) For \textbf{task selection}, we diversify the input and output length to form a five-category taxonomy, covering $9$ tasks. (3) For \textbf{evaluation metric}, we adopt the GPT4-Judge for automatically evaluating the decline of instruction-following abilities after watermarking. We evaluate $4$ open-source watermarks on $2$ LLMs under $2$ watermarking strengths and observe the common struggles for current methods on maintaining the generation quality. The code and data are available at \url{https://github.com/THU-KEG/WaterBench}.

{{</citation>}}


### (42/142) Gen-Z: Generative Zero-Shot Text Classification with Contextualized Label Descriptions (Sachin Kumar et al., 2023)

{{<citation>}}

Sachin Kumar, Chan Young Park, Yulia Tsvetkov. (2023)  
**Gen-Z: Generative Zero-Shot Text Classification with Contextualized Label Descriptions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Text Classification, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.07115v1)  

---


**ABSTRACT**  
Language model (LM) prompting--a popular paradigm for solving NLP tasks--has been shown to be susceptible to miscalibration and brittleness to slight prompt variations, caused by its discriminative prompting approach, i.e., predicting the label given the input. To address these issues, we propose Gen-Z--a generative prompting framework for zero-shot text classification. GEN-Z is generative, as it measures the LM likelihood of input text, conditioned on natural language descriptions of labels. The framework is multivariate, as label descriptions allow us to seamlessly integrate additional contextual information about the labels to improve task performance. On various standard classification benchmarks, with six open-source LM families, we show that zero-shot classification with simple contextualization of the data source of the evaluation set consistently outperforms both zero-shot and few-shot baselines while improving robustness to prompt variations. Further, our approach enables personalizing classification in a zero-shot manner by incorporating author, subject, or reader information in the label descriptions.

{{</citation>}}


### (43/142) Fovea Transformer: Efficient Long-Context Modeling with Structured Fine-to-Coarse Attention (Ziwei He et al., 2023)

{{<citation>}}

Ziwei He, Jian Yuan, Le Zhou, Jingwen Leng, Bo Jiang. (2023)  
**Fovea Transformer: Efficient Long-Context Modeling with Structured Fine-to-Coarse Attention**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.07102v1)  

---


**ABSTRACT**  
The quadratic complexity of self-attention in Transformers has hindered the processing of long text. To alleviate this problem, previous works have proposed to sparsify the attention matrix, taking advantage of the observation that crucial information about a token can be derived from its neighbors. These methods typically combine one or another form of local attention and global attention. Such combinations introduce abrupt changes in contextual granularity when going from local to global, which may be undesirable. We believe that a smoother transition could potentially enhance model's ability to capture long-context dependencies. In this study, we introduce Fovea Transformer, a long-context focused transformer that addresses the challenges of capturing global dependencies while maintaining computational efficiency. To achieve this, we construct a multi-scale tree from the input sequence, and use representations of context tokens with a progressively coarser granularity in the tree, as their distance to the query token increases. We evaluate our model on three long-context summarization tasks\footnote{Our code is publicly available at: \textit{https://github.com/ZiweiHe/Fovea-Transformer}}. It achieves state-of-the-art performance on two of them, and competitive results on the third with mixed improvement and setback of the evaluation metrics.

{{</citation>}}


### (44/142) Explanation-aware Soft Ensemble Empowers Large Language Model In-context Learning (Yue Yu et al., 2023)

{{<citation>}}

Yue Yu, Jiaming Shen, Tianqi Liu, Zhen Qin, Jing Nathan Yan, Jialu Liu, Chao Zhang, Michael Bendersky. (2023)  
**Explanation-aware Soft Ensemble Empowers Large Language Model In-context Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07099v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown remarkable capabilities in various natural language understanding tasks. With only a few demonstration examples, these LLMs can quickly adapt to target tasks without expensive gradient updates. Common strategies to boost such 'in-context' learning ability are to ensemble multiple model decoded results and require the model to generate an explanation along with the prediction. However, these models often treat different class predictions equally and neglect the potential discrepancy between the explanations and predictions. To fully unleash the power of explanations, we propose EASE, an Explanation-Aware Soft Ensemble framework to empower in-context learning with LLMs. We design two techniques, explanation-guided ensemble, and soft probability aggregation, to mitigate the effect of unreliable explanations and improve the consistency between explanations and final predictions. Experiments on seven natural language understanding tasks and four varying-size LLMs demonstrate the effectiveness of our proposed framework.

{{</citation>}}


### (45/142) To Tell The Truth: Language of Deception and Language Models (Bodhisattwa Prasad Majumder et al., 2023)

{{<citation>}}

Bodhisattwa Prasad Majumder, Sanchaita Hazra. (2023)  
**To Tell The Truth: Language of Deception and Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07092v1)  

---


**ABSTRACT**  
Text-based misinformation permeates online discourses, yet evidence of people's ability to discern truth from such deceptive textual content is scarce. We analyze a novel TV game show data where conversations in a high-stake environment between individuals with conflicting objectives result in lies. We investigate the manifestation of potentially verifiable language cues of deception in the presence of objective truth, a distinguishing feature absent in previous text-based deception datasets. We show that there exists a class of detectors (algorithms) that have similar truth detection performance compared to human subjects, even when the former accesses only the language cues while the latter engages in conversations with complete access to all potential sources of cues (language and audio-visual). Our model, built on a large language model, employs a bottleneck framework to learn discernible cues to determine truth, an act of reasoning in which human subjects often perform poorly, even with incentives. Our model detects novel but accurate language cues in many cases where humans failed to detect deception, opening up the possibility of humans collaborating with algorithms and ameliorating their ability to detect the truth.

{{</citation>}}


### (46/142) On the Discussion of Large Language Models: Symmetry of Agents and Interplay with Prompts (Qineng Wang et al., 2023)

{{<citation>}}

Qineng Wang, Zihao Wang, Ying Su, Yangqiu Song. (2023)  
**On the Discussion of Large Language Models: Symmetry of Agents and Interplay with Prompts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07076v1)  

---


**ABSTRACT**  
Two ways has been discussed to unlock the reasoning capability of a large language model. The first one is prompt engineering and the second one is to combine the multiple inferences of large language models, or the multi-agent discussion. Theoretically, this paper justifies the multi-agent discussion mechanisms from the symmetry of agents. Empirically, this paper reports the empirical results of the interplay of prompts and discussion mechanisms, revealing the empirical state-of-the-art performance of complex multi-agent mechanisms can be approached by carefully developed prompt engineering. This paper also proposes a scalable discussion mechanism based on conquer and merge, providing a simple multi-agent discussion solution with simple prompts but state-of-the-art performance.

{{</citation>}}


### (47/142) Context Consistency between Training and Testing in Simultaneous Machine Translation (Meizhi Zhong et al., 2023)

{{<citation>}}

Meizhi Zhong, Lemao Liu, Kehai Chen, Mingming Yang, Min Zhang. (2023)  
**Context Consistency between Training and Testing in Simultaneous Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2311.07066v1)  

---


**ABSTRACT**  
Simultaneous Machine Translation (SiMT) aims to yield a real-time partial translation with a monotonically growing the source-side context. However, there is a counterintuitive phenomenon about the context usage between training and testing: e.g., the wait-k testing model consistently trained with wait-k is much worse than that model inconsistently trained with wait-k' (k' is not equal to k) in terms of translation quality. To this end, we first investigate the underlying reasons behind this phenomenon and uncover the following two factors: 1) the limited correlation between translation quality and training (cross-entropy) loss; 2) exposure bias between training and testing. Based on both reasons, we then propose an effective training approach called context consistency training accordingly, which makes consistent the context usage between training and testing by optimizing translation quality and latency as bi-objectives and exposing the predictions to the model during the training. The experiments on three language pairs demonstrate our intuition: our system encouraging context consistency outperforms that existing systems with context inconsistency for the first time, with the help of our context consistency training approach.

{{</citation>}}


### (48/142) PROPANE: Prompt design as an inverse problem (Rimon Melamed et al., 2023)

{{<citation>}}

Rimon Melamed, Lucas H. McCabe, Tanay Wakhare, Yejin Kim, H. Howie Huang, Enric Boix-Adsera. (2023)  
**PROPANE: Prompt design as an inverse problem**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07064v1)  

---


**ABSTRACT**  
Carefully-designed prompts are key to inducing desired behavior in Large Language Models (LLMs). As a result, great effort has been dedicated to engineering prompts that guide LLMs toward particular behaviors. In this work, we propose an automatic prompt optimization framework, PROPANE, which aims to find a prompt that induces semantically similar outputs to a fixed set of examples without user intervention. We further demonstrate that PROPANE can be used to (a) improve existing prompts, and (b) discover semantically obfuscated prompts that transfer between models.

{{</citation>}}


### (49/142) Towards the Law of Capacity Gap in Distilling Language Models (Chen Zhang et al., 2023)

{{<citation>}}

Chen Zhang, Dawei Song, Zheyu Ye, Yan Gao. (2023)  
**Towards the Law of Capacity Gap in Distilling Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07052v1)  

---


**ABSTRACT**  
Language model (LM) distillation is a trending area that aims to distil the knowledge resided in a large teacher LM to a small student one. While various methods have been proposed to push the distillation to its limits, it is still a pain distilling LMs when a large capacity gap is exhibited between the teacher and the student LMs. The pain is mainly resulted by the curse of capacity gap, which describes that a larger teacher LM cannot always lead to a better student LM than one distilled from a smaller teacher LM due to the affect of capacity gap increment. That is, there is likely an optimal point yielding the best student LM along the scaling course of the teacher LM. Even worse, the curse of capacity gap can be only partly yet not fully lifted as indicated in previous studies.   However, the tale is not ever one-sided. Although a larger teacher LM has better performance than a smaller teacher LM, it is much more resource-demanding especially in the context of recent large LMs (LLMs). Consequently, instead of sticking to lifting the curse, leaving the curse as is should be arguably fine. Even better, in this paper, we reveal that the optimal capacity gap is almost consistent across different student scales and architectures, fortunately turning the curse into the law of capacity gap. The law later guides us to distil a 3B student LM (termed MiniMA) from a 7B teacher LM (adapted LLaMA2-7B). MiniMA is demonstrated to yield a new compute-performance pareto frontier among existing 3B LMs on commonly used benchmarks, and its instruction-tuned version (termed MiniChat) outperforms a wide range of 3B competitors in GPT4 evaluation and could even compete with several 7B chat models.

{{</citation>}}


### (50/142) ExpNote: Black-box Large Language Models are Better Task Solvers with Experience Notebook (Wangtao Sun et al., 2023)

{{<citation>}}

Wangtao Sun, Xuanqing Yu, Shizhu He, Jun Zhao, Kang Liu. (2023)  
**ExpNote: Black-box Large Language Models are Better Task Solvers with Experience Notebook**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07032v1)  

---


**ABSTRACT**  
Black-box Large Language Models (LLMs) have shown great power in solving various tasks and are considered general problem solvers. However, LLMs still fail in many specific tasks although understand the task instruction. In this paper, we focus on the problem of boosting the ability of black-box LLMs to solve downstream tasks. We propose ExpNote, an automated framework to help LLMs better adapt to unfamiliar tasks through reflecting and noting experiences from training data and retrieving them from external memory during testing. We evaluate ExpNote on multiple tasks and the experimental results demonstrate that the proposed method significantly improves the performance of black-box LLMs. The data and code are available at https://github.com/forangel2014/ExpNote

{{</citation>}}


### (51/142) ViLMA: A Zero-Shot Benchmark for Linguistic and Temporal Grounding in Video-Language Models (Ilker Kesen et al., 2023)

{{<citation>}}

Ilker Kesen, Andrea Pedrotti, Mustafa Dogan, Michele Cafagna, Emre Can Acikgoz, Letitia Parcalabescu, Iacer Calixto, Anette Frank, Albert Gatt, Aykut Erdem, Erkut Erdem. (2023)  
**ViLMA: A Zero-Shot Benchmark for Linguistic and Temporal Grounding in Video-Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.07022v1)  

---


**ABSTRACT**  
With the ever-increasing popularity of pretrained Video-Language Models (VidLMs), there is a pressing need to develop robust evaluation methodologies that delve deeper into their visio-linguistic capabilities. To address this challenge, we present ViLMA (Video Language Model Assessment), a task-agnostic benchmark that places the assessment of fine-grained capabilities of these models on a firm footing. Task-based evaluations, while valuable, fail to capture the complexities and specific temporal aspects of moving images that VidLMs need to process. Through carefully curated counterfactuals, ViLMA offers a controlled evaluation suite that sheds light on the true potential of these models, as well as their performance gaps compared to human-level understanding. ViLMA also includes proficiency tests, which assess basic capabilities deemed essential to solving the main counterfactual tests. We show that current VidLMs' grounding abilities are no better than those of vision-language models which use static images. This is especially striking once the performance on proficiency tests is factored in. Our benchmark serves as a catalyst for future research on VidLMs, helping to highlight areas that still need to be explored.

{{</citation>}}


### (52/142) Teach me with a Whisper: Enhancing Large Language Models for Analyzing Spoken Transcripts using Speech Embeddings (Fatema Hasan et al., 2023)

{{<citation>}}

Fatema Hasan, Yulong Li, James Foulds, Shimei Pan, Bishwaranjan Bhattacharjee. (2023)  
**Teach me with a Whisper: Enhancing Large Language Models for Analyzing Spoken Transcripts using Speech Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: AI, BERT, Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07014v1)  

---


**ABSTRACT**  
Speech data has rich acoustic and paralinguistic information with important cues for understanding a speaker's tone, emotion, and intent, yet traditional large language models such as BERT do not incorporate this information. There has been an increased interest in multi-modal language models leveraging audio and/or visual information and text. However, current multi-modal language models require both text and audio/visual data streams during inference/test time. In this work, we propose a methodology for training language models leveraging spoken language audio data but without requiring the audio stream during prediction time. This leads to an improved language model for analyzing spoken transcripts while avoiding an audio processing overhead at test time. We achieve this via an audio-language knowledge distillation framework, where we transfer acoustic and paralinguistic information from a pre-trained speech embedding (OpenAI Whisper) teacher model to help train a student language model on an audio-text dataset. In our experiments, the student model achieves consistent improvement over traditional language models on tasks analyzing spoken transcripts.

{{</citation>}}


### (53/142) Context-dependent Instruction Tuning for Dialogue Response Generation (Jin Myung Kwak et al., 2023)

{{<citation>}}

Jin Myung Kwak, Minseon Kim, Sung Ju Hwang. (2023)  
**Context-dependent Instruction Tuning for Dialogue Response Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2311.07006v1)  

---


**ABSTRACT**  
Recent language models have achieved impressive performance in natural language tasks by incorporating instructions with task input during fine-tuning. Since all samples in the same natural language task can be explained with the same task instructions, many instruction datasets only provide a few instructions for the entire task, without considering the input of each example in the task. However, this approach becomes ineffective in complex multi-turn dialogue generation tasks, where the input varies highly with each turn as the dialogue context changes, so that simple task instructions cannot improve the generation performance. To address this limitation, we introduce a context-based instruction fine-tuning framework for each multi-turn dialogue which generates both responses and instructions based on the previous context as input. During the evaluation, the model generates instructions based on the previous context to self-guide the response. The proposed framework produces comparable or even outstanding results compared to the baselines by aligning instructions to the input during fine-tuning with the instructions in quantitative evaluations on dialogue benchmark datasets with reduced computation budget.

{{</citation>}}


## cs.CY (1)



### (54/142) Western, Religious or Spiritual: An Evaluation of Moral Justification in Large Language Models (Eyup Engin Kucuk et al., 2023)

{{<citation>}}

Eyup Engin Kucuk, Muhammed Yusuf Kocyigit. (2023)  
**Western, Religious or Spiritual: An Evaluation of Moral Justification in Large Language Models**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07792v1)  

---


**ABSTRACT**  
The increasing success of Large Language Models (LLMs) in variety of tasks lead to their widespread use in our lives which necessitates the examination of these models from different perspectives. The alignment of these models to human values is an essential concern in order to establish trust that we have safe and responsible systems. In this paper, we aim to find out which values and principles are embedded in LLMs in the process of moral justification. For this purpose, we come up with three different moral perspective categories: Western tradition perspective (WT), Abrahamic tradition perspective (AT), and Spiritualist/Mystic tradition perspective (SMT). In two different experiment settings, we asked models to choose principles from the three for suggesting a moral action and evaluating the moral permissibility of an action if one tries to justify an action on these categories, respectively. Our experiments indicate that tested LLMs favors the Western tradition moral perspective over others. Additionally, we observe that there potentially exists an over-alignment towards religious values represented in the Abrahamic Tradition, which causes models to fail to recognize an action is immoral if it is presented as a "religious-action". We believe that these results are essential in order to direct our attention in future efforts.

{{</citation>}}


## cs.LG (20)



### (55/142) CSLP-AE: A Contrastive Split-Latent Permutation Autoencoder Framework for Zero-Shot Electroencephalography Signal Conversion (Anders Vestergaard Nørskov et al., 2023)

{{<citation>}}

Anders Vestergaard Nørskov, Alexander Neergaard Zahid, Morten Mørup. (2023)  
**CSLP-AE: A Contrastive Split-Latent Permutation Autoencoder Framework for Zero-Shot Electroencephalography Signal Conversion**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG, eess-SP, stat-ML  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.07788v1)  

---


**ABSTRACT**  
Electroencephalography (EEG) is a prominent non-invasive neuroimaging technique providing insights into brain function. Unfortunately, EEG data exhibit a high degree of noise and variability across subjects hampering generalizable signal extraction. Therefore, a key aim in EEG analysis is to extract the underlying neural activation (content) as well as to account for the individual subject variability (style). We hypothesize that the ability to convert EEG signals between tasks and subjects requires the extraction of latent representations accounting for content and style. Inspired by recent advancements in voice conversion technologies, we propose a novel contrastive split-latent permutation autoencoder (CSLP-AE) framework that directly optimizes for EEG conversion. Importantly, the latent representations are guided using contrastive learning to promote the latent splits to explicitly represent subject (style) and task (content). We contrast CSLP-AE to conventional supervised, unsupervised (AE), and self-supervised (contrastive learning) training and find that the proposed approach provides favorable generalizable characterizations of subject and task. Importantly, the procedure also enables zero-shot conversion between unseen subjects. While the present work only considers conversion of EEG, the proposed CSLP-AE provides a general framework for signal conversion and extraction of content (task activation) and style (subject variability) components of general interest for the modeling and analysis of biological signals.

{{</citation>}}


### (56/142) A Data-Free Approach to Mitigate Catastrophic Forgetting in Federated Class Incremental Learning for Vision Tasks (Sara Babakniya et al., 2023)

{{<citation>}}

Sara Babakniya, Zalan Fabian, Chaoyang He, Mahdi Soltanolkotabi, Salman Avestimehr. (2023)  
**A Data-Free Approach to Mitigate Catastrophic Forgetting in Federated Class Incremental Learning for Vision Tasks**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.07784v1)  

---


**ABSTRACT**  
Deep learning models often suffer from forgetting previously learned information when trained on new data. This problem is exacerbated in federated learning (FL), where the data is distributed and can change independently for each user. Many solutions are proposed to resolve this catastrophic forgetting in a centralized setting. However, they do not apply directly to FL because of its unique complexities, such as privacy concerns and resource limitations. To overcome these challenges, this paper presents a framework for \textbf{federated class incremental learning} that utilizes a generative model to synthesize samples from past distributions. This data can be later exploited alongside the training data to mitigate catastrophic forgetting. To preserve privacy, the generative model is trained on the server using data-free methods at the end of each task without requesting data from clients. Moreover, our solution does not demand the users to store old data or models, which gives them the freedom to join/leave the training at any time. Additionally, we introduce SuperImageNet, a new regrouping of the ImageNet dataset specifically tailored for federated continual learning. We demonstrate significant improvements compared to existing baselines through extensive experiments on multiple datasets.

{{</citation>}}


### (57/142) FedOpenHAR: Federated Multi-Task Transfer Learning for Sensor-Based Human Activity Recognition (Egemen İşgüder et al., 2023)

{{<citation>}}

Egemen İşgüder, Özlem Durmaz İncel. (2023)  
**FedOpenHAR: Federated Multi-Task Transfer Learning for Sensor-Based Human Activity Recognition**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.07765v1)  

---


**ABSTRACT**  
Motion sensors integrated into wearable and mobile devices provide valuable information about the device users. Machine learning and, recently, deep learning techniques have been used to characterize sensor data. Mostly, a single task, such as recognition of activities, is targeted, and the data is processed centrally at a server or in a cloud environment. However, the same sensor data can be utilized for multiple tasks and distributed machine-learning techniques can be used without the requirement of the transmission of data to a centre. This paper explores Federated Transfer Learning in a Multi-Task manner for both sensor-based human activity recognition and device position identification tasks. The OpenHAR framework is used to train the models, which contains ten smaller datasets. The aim is to obtain model(s) applicable for both tasks in different datasets, which may include only some label types. Multiple experiments are carried in the Flower federated learning environment using the DeepConvLSTM architecture. Results are presented for federated and centralized versions under different parameters and restrictions. By utilizing transfer learning and training a task-specific and personalized federated model, we obtained a similar accuracy with training each client individually and higher accuracy than a fully centralized approach.

{{</citation>}}


### (58/142) The Disagreement Problem in Faithfulness Metrics (Brian Barr et al., 2023)

{{<citation>}}

Brian Barr, Noah Fatsi, Leif Hancox-Li, Peter Richter, Daniel Proano, Caleb Mok. (2023)  
**The Disagreement Problem in Faithfulness Metrics**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07763v1)  

---


**ABSTRACT**  
The field of explainable artificial intelligence (XAI) aims to explain how black-box machine learning models work. Much of the work centers around the holy grail of providing post-hoc feature attributions to any model architecture. While the pace of innovation around novel methods has slowed down, the question remains of how to choose a method, and how to make it fit for purpose. Recently, efforts around benchmarking XAI methods have suggested metrics for that purpose -- but there are many choices. That bounty of choice still leaves an end user unclear on how to proceed. This paper focuses on comparing metrics with the aim of measuring faithfulness of local explanations on tabular classification problems -- and shows that the current metrics don't agree; leaving users unsure how to choose the most faithful explanations.

{{</citation>}}


### (59/142) Dynamic Local Attention with Hierarchical Patching for Irregular Clinical Time Series (Xingyu Chen et al., 2023)

{{<citation>}}

Xingyu Chen, Xiaochen Zheng, Amina Mollaysa, Manuel Schürch, Ahmed Allam, Michael Krauthammer. (2023)  
**Dynamic Local Attention with Hierarchical Patching for Irregular Clinical Time Series**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Attention, Clinical, Time Series  
[Paper Link](http://arxiv.org/abs/2311.07744v1)  

---


**ABSTRACT**  
Irregular multivariate time series data is prevalent in the clinical and healthcare domains. It is characterized by time-wise and feature-wise irregularities, making it challenging for machine learning methods to work with. To solve this, we introduce a new model architecture composed of two modules: (1) DLA, a Dynamic Local Attention mechanism that uses learnable queries and feature-specific local windows when computing the self-attention operation. This results in aggregating irregular time steps raw input within each window to a harmonized regular latent space representation while taking into account the different features' sampling rates. (2) A hierarchical MLP mixer that processes the output of DLA through multi-scale patching to leverage information at various scales for the downstream tasks. Our approach outperforms state-of-the-art methods on three real-world datasets, including the latest clinical MIMIC IV dataset.

{{</citation>}}


### (60/142) On The Truthfulness of 'Surprisingly Likely' Responses of Large Language Models (Naman Goel, 2023)

{{<citation>}}

Naman Goel. (2023)  
**On The Truthfulness of 'Surprisingly Likely' Responses of Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-GT, cs-LG, cs.LG  
Keywords: GPT, LLaMA, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.07692v1)  

---


**ABSTRACT**  
The surprisingly likely criterion in the seminal work of Prelec (the Bayesian Truth Serum) guarantees truthfulness in a game-theoretic multi-agent setting, by rewarding rational agents to maximise the expected information gain with their answers w.r.t. their probabilistic beliefs. We investigate the relevance of a similar criterion for responses of LLMs. We hypothesize that if the surprisingly likely criterion works in LLMs, under certain conditions, the responses that maximize the reward under this criterion should be more accurate than the responses that only maximize the posterior probability. Using benchmarks including the TruthfulQA benchmark and using openly available LLMs: GPT-2 and LLaMA-2, we show that the method indeed improves the accuracy significantly (for example, upto 24 percentage points aggregate improvement on TruthfulQA and upto 70 percentage points improvement on individual categories of questions).

{{</citation>}}


### (61/142) Data-Efficient Task Generalization via Probabilistic Model-based Meta Reinforcement Learning (Arjun Bhardwaj et al., 2023)

{{<citation>}}

Arjun Bhardwaj, Jonas Rothfuss, Bhavya Sukhija, Yarden As, Marco Hutter, Stelian Coros, Andreas Krause. (2023)  
**Data-Efficient Task Generalization via Probabilistic Model-based Meta Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.07558v1)  

---


**ABSTRACT**  
We introduce PACOH-RL, a novel model-based Meta-Reinforcement Learning (Meta-RL) algorithm designed to efficiently adapt control policies to changing dynamics. PACOH-RL meta-learns priors for the dynamics model, allowing swift adaptation to new dynamics with minimal interaction data. Existing Meta-RL methods require abundant meta-learning data, limiting their applicability in settings such as robotics, where data is costly to obtain. To address this, PACOH-RL incorporates regularization and epistemic uncertainty quantification in both the meta-learning and task adaptation stages. When facing new dynamics, we use these uncertainty estimates to effectively guide exploration and data collection. Overall, this enables positive transfer, even when access to data from prior tasks or dynamic settings is severely limited. Our experiment results demonstrate that PACOH-RL outperforms model-based RL and model-based Meta-RL baselines in adapting to new dynamic conditions. Finally, on a real robotic car, we showcase the potential for efficient RL policy adaptation in diverse, data-scarce conditions.

{{</citation>}}


### (62/142) Interpretable Fine-Tuning for Graph Neural Network Surrogate Models (Shivam Barwey et al., 2023)

{{<citation>}}

Shivam Barwey, Romit Maulik. (2023)  
**Interpretable Fine-Tuning for Graph Neural Network Surrogate Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-comp-ph, physics-flu-dyn  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.07548v1)  

---


**ABSTRACT**  
Data-based surrogate modeling has surged in capability in recent years with the emergence of graph neural networks (GNNs), which can operate directly on mesh-based representations of data. The goal of this work is to introduce an interpretable fine-tuning strategy for GNNs, with application to unstructured mesh-based fluid dynamics modeling. The end result is a fine-tuned GNN that adds interpretability to a pre-trained baseline GNN through an adaptive sub-graph sampling strategy that isolates regions in physical space intrinsically linked to the forecasting task, while retaining the predictive capability of the baseline. The structures identified by the fine-tuned GNNs, which are adaptively produced in the forward pass as explicit functions of the input, serve as an accessible link between the baseline model architecture, the optimization goal, and known problem-specific physics. Additionally, through a regularization procedure, the fine-tuned GNNs can also be used to identify, during inference, graph nodes that correspond to a majority of the anticipated forecasting error, adding a novel interpretable error-tagging capability to baseline models. Demonstrations are performed using unstructured flow data sourced from flow over a backward-facing step at high Reynolds numbers.

{{</citation>}}


### (63/142) Explicit Foundation Model Optimization with Self-Attentive Feed-Forward Neural Units (Jake Ryland Williams et al., 2023)

{{<citation>}}

Jake Ryland Williams, Haoran Zhao. (2023)  
**Explicit Foundation Model Optimization with Self-Attentive Feed-Forward Neural Units**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-PR, physics-data-an, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07510v1)  

---


**ABSTRACT**  
Iterative approximation methods using backpropagation enable the optimization of neural networks, but they remain computationally expensive, especially when used at scale. This paper presents an efficient alternative for optimizing neural networks that reduces the costs of scaling neural networks and provides high-efficiency optimizations for low-resource applications. We will discuss a general result about feed-forward neural networks and then extend this solution to compositional (mult-layer) networks, which are applied to a simplified transformer block containing feed-forward and self-attention layers. These models are used to train highly-specified and complex multi-layer neural architectures that we refer to as self-attentive feed-forward unit (SAFFU) layers, which we use to develop a transformer that appears to generalize well over small, cognitively-feasible, volumes of data. Testing demonstrates explicit solutions outperform models optimized by backpropagation alone. Moreover, further application of backpropagation after explicit solutions leads to better optima from smaller scales of data, training effective models from much less data is enabled by explicit solution warm starts. We then carry out ablation experiments training a roadmap of about 250 transformer models over 1-million tokens to determine ideal settings. We find that multiple different architectural variants produce highly-performant models, and discover from this ablation that some of the best are not the most parameterized. This appears to indicate well-generalized models could be reached using less data by using explicit solutions, and that architectural exploration using explicit solutions pays dividends in guiding the search for efficient variants with fewer parameters, and which could be incorporated into low-resource hardware where AI might be embodied.

{{</citation>}}


### (64/142) On Self-Supervised Dynamic Incremental Regularised Adaptation (Abanoub Ghobrial et al., 2023)

{{<citation>}}

Abanoub Ghobrial, Kerstin Eder. (2023)  
**On Self-Supervised Dynamic Incremental Regularised Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.07461v1)  

---


**ABSTRACT**  
In this paper, we overview a recent method for dynamic domain adaptation named DIRA, which relies on a few samples in addition to a regularisation approach named elastic weight consolidation to achieve state-of-the-art (SOTA) domain adaptation results. DIRA has been previously shown to perform competitively with SOTA unsupervised adaption techniques. However, a limitation of DIRA is that it relies on labels to be provided for the few samples used in adaption. This makes it a supervised technique. In this paper, we discuss a proposed alteration to the DIRA method to make it self-supervised i.e. remove the need for providing labels. Experiments on our proposed alteration will be provided in future work.

{{</citation>}}


### (65/142) Optimising Human-AI Collaboration by Learning Convincing Explanations (Alex J. Chan et al., 2023)

{{<citation>}}

Alex J. Chan, Alihan Huyuk, Mihaela van der Schaar. (2023)  
**Optimising Human-AI Collaboration by Learning Convincing Explanations**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-HC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07426v1)  

---


**ABSTRACT**  
Machine learning models are being increasingly deployed to take, or assist in taking, complicated and high-impact decisions, from quasi-autonomous vehicles to clinical decision support systems. This poses challenges, particularly when models have hard-to-detect failure modes and are able to take actions without oversight. In order to handle this challenge, we propose a method for a collaborative system that remains safe by having a human ultimately making decisions, while giving the model the best opportunity to convince and debate them with interpretable explanations. However, the most helpful explanation varies among individuals and may be inconsistent across stated preferences. To this end we develop an algorithm, Ardent, to efficiently learn a ranking through interaction and best assist humans complete a task. By utilising a collaborative approach, we can ensure safety and improve performance while addressing transparency and accountability concerns. Ardent enables efficient and effective decision-making by adapting to individual preferences for explanations, which we validate through extensive simulations alongside a user study involving a challenging image classification task, demonstrating consistent improvement over competing systems.

{{</citation>}}


### (66/142) ADAMM: Anomaly Detection of Attributed Multi-graphs with Metadata: A Unified Neural Network Approach (Konstantinos Sotiropoulos et al., 2023)

{{<citation>}}

Konstantinos Sotiropoulos, Lingxiao Zhao, Pierre Jinghong Liang, Leman Akoglu. (2023)  
**ADAMM: Anomaly Detection of Attributed Multi-graphs with Metadata: A Unified Neural Network Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.07355v1)  

---


**ABSTRACT**  
Given a complex graph database of node- and edge-attributed multi-graphs as well as associated metadata for each graph, how can we spot the anomalous instances? Many real-world problems can be cast as graph inference tasks where the graph representation could capture complex relational phenomena (e.g., transactions among financial accounts in a journal entry), along with metadata reflecting tabular features (e.g. approver, effective date, etc.). While numerous anomaly detectors based on Graph Neural Networks (GNNs) have been proposed, none are capable of directly handling directed graphs with multi-edges and self-loops. Furthermore, the simultaneous handling of relational and tabular features remains an unexplored area. In this work we propose ADAMM, a novel graph neural network model that handles directed multi-graphs, providing a unified end-to-end architecture that fuses metadata and graph-level representation learning through an unsupervised anomaly detection objective. Experiments on datasets from two different domains, namely, general-ledger journal entries from different firms (accounting) as well as human GPS trajectories from thousands of individuals (urban mobility) validate ADAMM's generality and detection effectiveness of expert-guided and ground-truth anomalies. Notably, ADAMM outperforms existing baselines that handle the two data modalities (graph and metadata) separately with post hoc synthesis efforts.

{{</citation>}}


### (67/142) MetaSymNet: A Dynamic Symbolic Regression Network Capable of Evolving into Arbitrary Formulations (Yanjie Li et al., 2023)

{{<citation>}}

Yanjie Li, Weijun Li, Lina Yu, Min Wu, Jinyi Liu, Wenqiang Li, Meilan Hao, Shu Wei, Yusong Deng. (2023)  
**MetaSymNet: A Dynamic Symbolic Regression Network Capable of Evolving into Arbitrary Formulations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07326v1)  

---


**ABSTRACT**  
Mathematical formulas serve as the means of communication between humans and nature, encapsulating the operational laws governing natural phenomena. The concise formulation of these laws is a crucial objective in scientific research and an important challenge for artificial intelligence (AI). While traditional artificial neural networks (MLP) excel at data fitting, they often yield uninterpretable black box results that hinder our understanding of the relationship between variables x and predicted values y. Moreover, the fixed network architecture in MLP often gives rise to redundancy in both network structure and parameters. To address these issues, we propose MetaSymNet, a novel neural network that dynamically adjusts its structure in real-time, allowing for both expansion and contraction. This adaptive network employs the PANGU meta function as its activation function, which is a unique type capable of evolving into various basic functions during training to compose mathematical formulas tailored to specific needs. We then evolve the neural network into a concise, interpretable mathematical expression. To evaluate MetaSymNet's performance, we compare it with four state-of-the-art symbolic regression algorithms across more than 10 public datasets comprising 222 formulas. Our experimental results demonstrate that our algorithm outperforms others consistently regardless of noise presence or absence. Furthermore, we assess MetaSymNet against MLP and SVM regarding their fitting ability and extrapolation capability, these are two essential aspects of machine learning algorithms. The findings reveal that our algorithm excels in both areas. Finally, we compared MetaSymNet with MLP using iterative pruning in network structure complexity. The results show that MetaSymNet's network structure complexity is obviously less than MLP under the same goodness of fit.

{{</citation>}}


### (68/142) Predictive and Prescriptive Analytics for Multi-Site Modeling of Frail and Elderly Patient Services (Elizabeth Williams et al., 2023)

{{<citation>}}

Elizabeth Williams, Daniel Gartner, Paul Harper. (2023)  
**Predictive and Prescriptive Analytics for Multi-Site Modeling of Frail and Elderly Patient Services**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2311.07283v1)  

---


**ABSTRACT**  
Recent research has highlighted the potential of linking predictive and prescriptive analytics. However, it remains widely unexplored how both paradigms could benefit from one another to address today's major challenges in healthcare. One of these is smarter planning of resource capacities for frail and elderly inpatient wards, addressing the societal challenge of an aging population. Frail and elderly patients typically suffer from multimorbidity and require more care while receiving medical treatment. The aim of this research is to assess how various predictive and prescriptive analytical methods, both individually and in tandem, contribute to addressing the operational challenges within an area of healthcare that is growing in demand. Clinical and demographic patient attributes are gathered from more than 165,000 patient records and used to explain and predict length of stay. To that extent, we employ Classification and Regression Trees (CART) analysis to establish this relationship. On the prescriptive side, deterministic and two-stage stochastic programs are developed to determine how to optimally plan for beds and ward staff with the objective to minimize cost. Furthermore, the two analytical methodologies are linked by generating demand for the prescriptive models using the CART groupings. The results show the linked methodologies provided different but similar results compared to using averages and in doing so, captured a more realistic real-world variation in the patient length of stay. Our research reveals that healthcare managers should consider using predictive and prescriptive models to make more informed decisions. By combining predictive and prescriptive analytics, healthcare managers can move away from relying on averages and incorporate the unique characteristics of their patients to create more robust planning decisions, mitigating risks caused by variations in demand.

{{</citation>}}


### (69/142) Input Convex LSTM: A Convex Approach for Fast Lyapunov-Based Model Predictive Control (Zihao Wang et al., 2023)

{{<citation>}}

Zihao Wang, Zhe Wu. (2023)  
**Input Convex LSTM: A Convex Approach for Fast Lyapunov-Based Model Predictive Control**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.07202v1)  

---


**ABSTRACT**  
Leveraging Input Convex Neural Networks (ICNNs), ICNN-based Model Predictive Control (MPC) successfully attains globally optimal solutions by upholding convexity within the MPC framework. However, current ICNN architectures encounter the issue of vanishing gradients, which limits their ability to serve as deep neural networks for complex tasks. Additionally, the current neural network-based MPC, including conventional neural network-based MPC and ICNN-based MPC, faces slower convergence speed when compared to MPC based on first-principles models. In this study, we leverage the principles of ICNNs to propose a novel Input Convex LSTM for Lyapunov-based MPC, with the specific goal of reducing convergence time and mitigating the vanishing gradient problem while ensuring closed-loop stability. From a simulation study of a nonlinear chemical reactor, we observed a mitigation of vanishing gradient problem and a reduction in convergence time, with a percentage decrease of 46.7%, 31.3%, and 20.2% compared to baseline plain RNN, plain LSTM, and Input Convex Recurrent Neural Network, respectively.

{{</citation>}}


### (70/142) Knowledge Graph Representations to enhance Intensive Care Time-Series Predictions (Samyak Jain et al., 2023)

{{<citation>}}

Samyak Jain, Manuel Burger, Gunnar Rätsch, Rita Kuznetsova. (2023)  
**Knowledge Graph Representations to enhance Intensive Care Time-Series Predictions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.07180v1)  

---


**ABSTRACT**  
Intensive Care Units (ICU) require comprehensive patient data integration for enhanced clinical outcome predictions, crucial for assessing patient conditions. Recent deep learning advances have utilized patient time series data, and fusion models have incorporated unstructured clinical reports, improving predictive performance. However, integrating established medical knowledge into these models has not yet been explored. The medical domain's data, rich in structural relationships, can be harnessed through knowledge graphs derived from clinical ontologies like the Unified Medical Language System (UMLS) for better predictions. Our proposed methodology integrates this knowledge with ICU data, improving clinical decision modeling. It combines graph representations with vital signs and clinical reports, enhancing performance, especially when data is missing. Additionally, our model includes an interpretability component to understand how knowledge graph nodes affect predictions.

{{</citation>}}


### (71/142) A Consistent Diffusion-Based Algorithm for Semi-Supervised Graph Learning (Thomas Bonald et al., 2023)

{{<citation>}}

Thomas Bonald, Nathan de Lara. (2023)  
**A Consistent Diffusion-Based Algorithm for Semi-Supervised Graph Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.07627v1)  

---


**ABSTRACT**  
The task of semi-supervised classification aims at assigning labels to all nodes of a graph based on the labels known for a few nodes, called the seeds. One of the most popular algorithms relies on the principle of heat diffusion, where the labels of the seeds are spread by thermoconductance and the temperature of each node at equilibrium is used as a score function for each label. In this paper, we prove that this algorithm is not consistent unless the temperatures of the nodes at equilibrium are centered before scoring. This crucial step does not only make the algorithm provably consistent on a block model but brings significant performance gains on real graphs.

{{</citation>}}


### (72/142) Activity Sparsity Complements Weight Sparsity for Efficient RNN Inference (Rishav Mukherji et al., 2023)

{{<citation>}}

Rishav Mukherji, Mark Schöne, Khaleelulla Khan Nazeer, Christian Mayr, Anand Subramoney. (2023)  
**Activity Sparsity Complements Weight Sparsity for Efficient RNN Inference**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.07625v1)  

---


**ABSTRACT**  
Artificial neural networks open up unprecedented machine learning capabilities at the cost of ever growing computational requirements. Sparsifying the parameters, often achieved through weight pruning, has been identified as a powerful technique to compress the number of model parameters and reduce the computational operations of neural networks. Yet, sparse activations, while omnipresent in both biological neural networks and deep learning systems, have not been fully utilized as a compression technique in deep learning. Moreover, the interaction between sparse activations and weight pruning is not fully understood. In this work, we demonstrate that activity sparsity can compose multiplicatively with parameter sparsity in a recurrent neural network model based on the GRU that is designed to be activity sparse. We achieve up to $20\times$ reduction of computation while maintaining perplexities below $60$ on the Penn Treebank language modeling task. This magnitude of reduction has not been achieved previously with solely sparsely connected LSTMs, and the language modeling performance of our model has not been achieved previously with any sparsely activated recurrent neural networks or spiking neural networks. Neuromorphic computing devices are especially good at taking advantage of the dynamic activity sparsity, and our results provide strong evidence that making deep learning models activity sparse and porting them to neuromorphic devices can be a viable strategy that does not compromise on task performance. Our results also drive further convergence of methods from deep learning and neuromorphic computing for efficient machine learning.

{{</citation>}}


### (73/142) SABAF: Removing Strong Attribute Bias from Neural Networks with Adversarial Filtering (Jiazhi Li et al., 2023)

{{<citation>}}

Jiazhi Li, Mahyar Khayatkhoei, Jiageng Zhu, Hanchen Xie, Mohamed E. Hussein, Wael AbdAlmageed. (2023)  
**SABAF: Removing Strong Attribute Bias from Neural Networks with Adversarial Filtering**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2311.07141v1)  

---


**ABSTRACT**  
Ensuring a neural network is not relying on protected attributes (e.g., race, sex, age) for prediction is crucial in advancing fair and trustworthy AI. While several promising methods for removing attribute bias in neural networks have been proposed, their limitations remain under-explored. To that end, in this work, we mathematically and empirically reveal the limitation of existing attribute bias removal methods in presence of strong bias and propose a new method that can mitigate this limitation. Specifically, we first derive a general non-vacuous information-theoretical upper bound on the performance of any attribute bias removal method in terms of the bias strength, revealing that they are effective only when the inherent bias in the dataset is relatively weak. Next, we derive a necessary condition for the existence of any method that can remove attribute bias regardless of the bias strength. Inspired by this condition, we then propose a new method using an adversarial objective that directly filters out protected attributes in the input space while maximally preserving all other attributes, without requiring any specific target label. The proposed method achieves state-of-the-art performance in both strong and moderate bias settings. We provide extensive experiments on synthetic, image, and census datasets, to verify the derived theoretical bound and its consequences in practice, and evaluate the effectiveness of the proposed method in removing strong attribute bias.

{{</citation>}}


### (74/142) Exposition on over-squashing problem on GNNs: Current Methods, Benchmarks and Challenges (Dai Shi et al., 2023)

{{<citation>}}

Dai Shi, Andi Han, Lequan Lin, Yi Guo, Junbin Gao. (2023)  
**Exposition on over-squashing problem on GNNs: Current Methods, Benchmarks and Challenges**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.07073v1)  

---


**ABSTRACT**  
Graph-based message-passing neural networks (MPNNs) have achieved remarkable success in both node and graph-level learning tasks. However, several identified problems, including over-smoothing (OSM), limited expressive power, and over-squashing (OSQ), still limit the performance of MPNNs. In particular, OSQ serves as the latest identified problem, where MPNNs gradually lose their learning accuracy when long-range dependencies between graph nodes are required. In this work, we provide an exposition on the OSQ problem by summarizing different formulations of OSQ from current literature, as well as the three different categories of approaches for addressing the OSQ problem. In addition, we also discuss the alignment between OSQ and expressive power and the trade-off between OSQ and OSM. Furthermore, we summarize the empirical methods leveraged from existing works to verify the efficiency of OSQ mitigation approaches, with illustrations of their computational complexities. Lastly, we list some open questions that are of interest for further exploration of the OSQ problem along with potential directions from the best of our knowledge.

{{</citation>}}


## cs.SD (3)



### (75/142) Parrot-Trained Adversarial Examples: Pushing the Practicality of Black-Box Audio Attacks against Speaker Recognition Models (Rui Duan et al., 2023)

{{<citation>}}

Rui Duan, Zhe Qu, Leah Ding, Yao Liu, Zhuo Lu. (2023)  
**Parrot-Trained Adversarial Examples: Pushing the Practicality of Black-Box Audio Attacks against Speaker Recognition Models**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Amazon, Google  
[Paper Link](http://arxiv.org/abs/2311.07780v1)  

---


**ABSTRACT**  
Audio adversarial examples (AEs) have posed significant security challenges to real-world speaker recognition systems. Most black-box attacks still require certain information from the speaker recognition model to be effective (e.g., keeping probing and requiring the knowledge of similarity scores). This work aims to push the practicality of the black-box attacks by minimizing the attacker's knowledge about a target speaker recognition model. Although it is not feasible for an attacker to succeed with completely zero knowledge, we assume that the attacker only knows a short (or a few seconds) speech sample of a target speaker. Without any probing to gain further knowledge about the target model, we propose a new mechanism, called parrot training, to generate AEs against the target model. Motivated by recent advancements in voice conversion (VC), we propose to use the one short sentence knowledge to generate more synthetic speech samples that sound like the target speaker, called parrot speech. Then, we use these parrot speech samples to train a parrot-trained(PT) surrogate model for the attacker. Under a joint transferability and perception framework, we investigate different ways to generate AEs on the PT model (called PT-AEs) to ensure the PT-AEs can be generated with high transferability to a black-box target model with good human perceptual quality. Real-world experiments show that the resultant PT-AEs achieve the attack success rates of 45.8% - 80.8% against the open-source models in the digital-line scenario and 47.9% - 58.3% against smart devices, including Apple HomePod (Siri), Amazon Echo, and Google Home, in the over-the-air scenario.

{{</citation>}}


### (76/142) Unsupervised Musical Object Discovery from Audio (Joonsu Gha et al., 2023)

{{<citation>}}

Joonsu Gha, Vincent Herrmann, Benjamin Grewe, Jürgen Schmidhuber, Anand Gopalakrishnan. (2023)  
**Unsupervised Musical Object Discovery from Audio**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.07534v2)  

---


**ABSTRACT**  
Current object-centric learning models such as the popular SlotAttention architecture allow for unsupervised visual scene decomposition. Our novel MusicSlots method adapts SlotAttention to the audio domain, to achieve unsupervised music decomposition. Since concepts of opacity and occlusion in vision have no auditory analogues, the softmax normalization of alpha masks in the decoders of visual object-centric models is not well-suited for decomposing audio objects. MusicSlots overcomes this problem. We introduce a spectrogram-based multi-object music dataset tailored to evaluate object-centric learning on western tonal music. MusicSlots achieves good performance on unsupervised note discovery and outperforms several established baselines on supervised note property prediction tasks.

{{</citation>}}


### (77/142) On the Effectiveness of ASR Representations in Real-world Noisy Speech Emotion Recognition (Xiaohan Shi et al., 2023)

{{<citation>}}

Xiaohan Shi, Jiajun He, Xingfeng Li, Tomoki Toda. (2023)  
**On the Effectiveness of ASR Representations in Real-world Noisy Speech Emotion Recognition**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2311.07093v2)  

---


**ABSTRACT**  
This paper proposes an efficient attempt to noisy speech emotion recognition (NSER). Conventional NSER approaches have proven effective in mitigating the impact of artificial noise sources, such as white Gaussian noise, but are limited to non-stationary noises in real-world environments due to their complexity and uncertainty. To overcome this limitation, we introduce a new method for NSER by adopting the automatic speech recognition (ASR) model as a noise-robust feature extractor to eliminate non-vocal information in noisy speech. We first obtain intermediate layer information from the ASR model as a feature representation for emotional speech and then apply this representation for the downstream NSER task. Our experimental results show that 1) the proposed method achieves better NSER performance compared with the conventional noise reduction method, 2) outperforms self-supervised learning approaches, and 3) even outperforms text-based approaches using ASR transcription or the ground truth transcription of noisy speech.

{{</citation>}}


## cs.CV (27)



### (78/142) Vision-Language Integration in Multimodal Video Transformers (Partially) Aligns with the Brain (Dota Tianai Dong et al., 2023)

{{<citation>}}

Dota Tianai Dong, Mariya Toneva. (2023)  
**Vision-Language Integration in Multimodal Video Transformers (Partially) Aligns with the Brain**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.07766v1)  

---


**ABSTRACT**  
Integrating information from multiple modalities is arguably one of the essential prerequisites for grounding artificial intelligence systems with an understanding of the real world. Recent advances in video transformers that jointly learn from vision, text, and sound over time have made some progress toward this goal, but the degree to which these models integrate information from modalities still remains unclear. In this work, we present a promising approach for probing a pre-trained multimodal video transformer model by leveraging neuroscientific evidence of multimodal information processing in the brain. Using brain recordings of participants watching a popular TV show, we analyze the effects of multi-modal connections and interactions in a pre-trained multi-modal video transformer on the alignment with uni- and multi-modal brain regions. We find evidence that vision enhances masked prediction performance during language processing, providing support that cross-modal representations in models can benefit individual modalities. However, we don't find evidence of brain-relevant information captured by the joint multi-modal transformer representations beyond that captured by all of the individual modalities. We finally show that the brain alignment of the pre-trained joint representation can be improved by fine-tuning using a task that requires vision-language inferences. Overall, our results paint an optimistic picture of the ability of multi-modal transformers to integrate vision and language in partially brain-relevant ways but also show that improving the brain alignment of these models may require new approaches.

{{</citation>}}


### (79/142) SynthEnsemble: A Fusion of CNN, Vision Transformer, and Hybrid Models for Multi-Label Chest X-Ray Classification (S. M. Nabil Ashraf et al., 2023)

{{<citation>}}

S. M. Nabil Ashraf, Md. Adyelullahil Mamun, Hasnat Md. Abdullah, Md. Golam Rabiul Alam. (2023)  
**SynthEnsemble: A Fusion of CNN, Vision Transformer, and Hybrid Models for Multi-Label Chest X-Ray Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.07750v1)  

---


**ABSTRACT**  
Chest X-rays are widely used to diagnose thoracic diseases, but the lack of detailed information about these abnormalities makes it challenging to develop accurate automated diagnosis systems, which is crucial for early detection and effective treatment. To address this challenge, we employed deep learning techniques to identify patterns in chest X-rays that correspond to different diseases. We conducted experiments on the "ChestX-ray14" dataset using various pre-trained CNNs, transformers, hybrid(CNN+Transformer) models and classical models. The best individual model was the CoAtNet, which achieved an area under the receiver operating characteristic curve (AUROC) of 84.2%. By combining the predictions of all trained models using a weighted average ensemble where the weight of each model was determined using differential evolution, we further improved the AUROC to 85.4%, outperforming other state-of-the-art methods in this field. Our findings demonstrate the potential of deep learning techniques, particularly ensemble deep learning, for improving the accuracy of automatic diagnosis of thoracic diseases from chest X-rays.

{{</citation>}}


### (80/142) Quality-Aware Prototype Memory for Face Representation Learning (Evgeny Smirnov et al., 2023)

{{<citation>}}

Evgeny Smirnov, Vasiliy Galyuk, Evgeny Lukyanets. (2023)  
**Quality-Aware Prototype Memory for Face Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.07734v1)  

---


**ABSTRACT**  
Prototype Memory is a powerful model for face representation learning. It enables the training of face recognition models using datasets of any size, with on-the-fly generation of prototypes (classifier weights) and efficient ways of their utilization. Prototype Memory demonstrated strong results in many face recognition benchmarks. However, the algorithm of prototype generation, used in it, is prone to the problems of imperfectly calculated prototypes in case of low-quality or poorly recognizable faces in the images, selected for the prototype creation. All images of the same person, presented in the mini-batch, used with equal weights, and the resulting averaged prototype could be contaminated with imperfect embeddings of such face images. It can lead to misdirected training signals and impair the performance of the trained face recognition models. In this paper, we propose a simple and effective way to improve Prototype Memory with quality-aware prototype generation. Quality-Aware Prototype Memory uses different weights for images of different quality in the process of prototype generation. With this improvement, prototypes get more valuable information from high-quality images and less hurt by low-quality ones. We propose and compare several methods of quality estimation and usage, perform extensive experiments on the different face recognition benchmarks and demonstrate the advantages of the proposed model compared to the basic version of Prototype Memory.

{{</citation>}}


### (81/142) SPHINX: The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal Large Language Models (Ziyi Lin et al., 2023)

{{<citation>}}

Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, Jiaming Han, Siyuan Huang, Yichi Zhang, Xuming He, Hongsheng Li, Yu Qiao. (2023)  
**SPHINX: The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Embedding, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07575v1)  

---


**ABSTRACT**  
We present SPHINX, a versatile multi-modal large language model (MLLM) with a joint mixing of model weights, tuning tasks, and visual embeddings. First, for stronger vision-language alignment, we unfreeze the large language model (LLM) during pre-training, and introduce a weight mix strategy between LLMs trained by real-world and synthetic data. By directly integrating the weights from two domains, the mixed LLM can efficiently incorporate diverse semantics with favorable robustness. Then, to enable multi-purpose capabilities, we mix a variety of tasks for joint visual instruction tuning, and design task-specific instructions to avoid inter-task conflict. In addition to the basic visual question answering, we include more challenging tasks such as region-level understanding, caption grounding, document layout detection, and human pose estimation, contributing to mutual enhancement over different scenarios. Additionally, we propose to extract comprehensive visual embeddings from various network architectures, pre-training paradigms, and information granularity, providing language models with more robust image representations. Based on our proposed joint mixing, SPHINX exhibits superior multi-modal understanding capabilities on a wide range of applications. On top of this, we further propose an efficient strategy aiming to better capture fine-grained appearances of high-resolution images. With a mixing of different scales and high-resolution sub-images, SPHINX attains exceptional visual parsing and reasoning performance on existing evaluation benchmarks. We hope our work may cast a light on the exploration of joint mixing in future MLLM research. Code is released at https://github.com/Alpha-VLLM/LLaMA2-Accessory.

{{</citation>}}


### (82/142) To See is to Believe: Prompting GPT-4V for Better Visual Instruction Tuning (Junke Wang et al., 2023)

{{<citation>}}

Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, Yu-Gang Jiang. (2023)  
**To See is to Believe: Prompting GPT-4V for Better Visual Instruction Tuning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.07574v1)  

---


**ABSTRACT**  
Existing visual instruction tuning methods typically prompt large language models with textual descriptions to generate instruction-following data. Despite the promising performance achieved, these descriptions are derived from image annotations, which are oftentimes coarse-grained. Furthermore, the instructions might even contradict the visual content without observing the entire visual context. To address this challenge, we introduce a fine-grained visual instruction dataset, LVIS-Instruct4V, which contains 220K visually aligned and context-aware instructions produced by prompting the powerful GPT-4V with images from LVIS. Through experimental validation and case studies, we demonstrate that high-quality visual instructional data could improve the performance of LLaVA-1.5, a state-of-the-art large multimodal model, across a wide spectrum of benchmarks by clear margins. Notably, by simply replacing the LLaVA-Instruct with our LVIS-Instruct4V, we achieve better results than LLaVA on most challenging LMM benchmarks, e.g., LLaVA$^w$ (76.7 vs. 70.7) and MM-Vet (40.2 vs. 35.4). We release our data and model at https://github.com/X2FD/LVIS-INSTRUCT4V.

{{</citation>}}


### (83/142) GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation (An Yan et al., 2023)

{{<citation>}}

An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Lin, Linjie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian McAuley, Jianfeng Gao, Zicheng Liu, Lijuan Wang. (2023)  
**GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GPT, GPT-4, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.07562v1)  

---


**ABSTRACT**  
We present MM-Navigator, a GPT-4V-based agent for the smartphone graphical user interface (GUI) navigation task. MM-Navigator can interact with a smartphone screen as human users, and determine subsequent actions to fulfill given instructions. Our findings demonstrate that large multimodal models (LMMs), specifically GPT-4V, excel in zero-shot GUI navigation through its advanced screen interpretation, action reasoning, and precise action localization capabilities. We first benchmark MM-Navigator on our collected iOS screen dataset. According to human assessments, the system exhibited a 91\% accuracy rate in generating reasonable action descriptions and a 75\% accuracy rate in executing the correct actions for single-step instructions on iOS. Additionally, we evaluate the model on a subset of an Android screen navigation dataset, where the model outperforms previous GUI navigators in a zero-shot fashion. Our benchmark and detailed analyses aim to lay a robust groundwork for future research into the GUI navigation task. The project page is at https://github.com/zzxslp/MM-Navigator.

{{</citation>}}


### (84/142) GPT-4V(ision) as A Social Media Analysis Engine (Hanjia Lyu et al., 2023)

{{<citation>}}

Hanjia Lyu, Jinfa Huang, Daoan Zhang, Yongsheng Yu, Xinyi Mou, Jinsheng Pan, Zhengyuan Yang, Zhongyu Wei, Jiebo Luo. (2023)  
**GPT-4V(ision) as A Social Media Analysis Engine**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-MM, cs.CV  
Keywords: GPT, GPT-4, Social Media  
[Paper Link](http://arxiv.org/abs/2311.07547v1)  

---


**ABSTRACT**  
Recent research has offered insights into the extraordinary capabilities of Large Multimodal Models (LMMs) in various general vision and language tasks. There is growing interest in how LMMs perform in more specialized domains. Social media content, inherently multimodal, blends text, images, videos, and sometimes audio. Understanding social multimedia content remains a challenging problem for contemporary machine learning frameworks. In this paper, we explore GPT-4V(ision)'s capabilities for social multimedia analysis. We select five representative tasks, including sentiment analysis, hate speech detection, fake news identification, demographic inference, and political ideology detection, to evaluate GPT-4V. Our investigation begins with a preliminary quantitative analysis for each task using existing benchmark datasets, followed by a careful review of the results and a selection of qualitative samples that illustrate GPT-4V's potential in understanding multimodal social media content. GPT-4V demonstrates remarkable efficacy in these tasks, showcasing strengths such as joint understanding of image-text pairs, contextual and cultural awareness, and extensive commonsense knowledge. Despite the overall impressive capacity of GPT-4V in the social media domain, there remain notable challenges. GPT-4V struggles with tasks involving multilingual social multimedia comprehension and has difficulties in generalizing to the latest trends in social media. Additionally, it exhibits a tendency to generate erroneous information in the context of evolving celebrity and politician knowledge, reflecting the known hallucination problem. The insights gleaned from our findings underscore a promising future for LMMs in enhancing our comprehension of social media content and its users through the analysis of multimodal information.

{{</citation>}}


### (85/142) Language Grounded QFormer for Efficient Vision Language Understanding (Moulik Choraria et al., 2023)

{{<citation>}}

Moulik Choraria, Nitesh Sekhar, Yue Wu, Xu Zhang, Prateek Singhal, Lav R. Varshney. (2023)  
**Language Grounded QFormer for Efficient Vision Language Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.07449v1)  

---


**ABSTRACT**  
Large-scale pretraining and instruction tuning have been successful for training general-purpose language models with broad competencies. However, extending to general-purpose vision-language models is challenging due to the distributional diversity in visual inputs. A recent line of work explores vision-language instruction tuning, taking inspiration from the Query Transformer (QFormer) approach proposed in BLIP-2 models for bridging frozen modalities. However, these approaches rely heavily on large-scale multi-modal pretraining for representation learning before eventual finetuning, incurring a huge computational overhead, poor scaling, and limited accessibility. To that end, we propose a more efficient method for QFormer-based vision-language alignment and demonstrate the effectiveness of our strategy compared to existing baselines in improving the efficiency of vision-language pretraining.

{{</citation>}}


### (86/142) Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text (Zhongfei Qing et al., 2023)

{{<citation>}}

Zhongfei Qing, Zhongang Cai, Zhitao Yang, Lei Yang. (2023)  
**Story-to-Motion: Synthesizing Infinite and Controllable Character Animation from Long Text**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07446v1)  

---


**ABSTRACT**  
Generating natural human motion from a story has the potential to transform the landscape of animation, gaming, and film industries. A new and challenging task, Story-to-Motion, arises when characters are required to move to various locations and perform specific motions based on a long text description. This task demands a fusion of low-level control (trajectories) and high-level control (motion semantics). Previous works in character control and text-to-motion have addressed related aspects, yet a comprehensive solution remains elusive: character control methods do not handle text description, whereas text-to-motion methods lack position constraints and often produce unstable motions. In light of these limitations, we propose a novel system that generates controllable, infinitely long motions and trajectories aligned with the input text. (1) We leverage contemporary Large Language Models to act as a text-driven motion scheduler to extract a series of (text, position, duration) pairs from long text. (2) We develop a text-driven motion retrieval scheme that incorporates motion matching with motion semantic and trajectory constraints. (3) We design a progressive mask transformer that addresses common artifacts in the transition motion such as unnatural pose and foot sliding. Beyond its pioneering role as the first comprehensive solution for Story-to-Motion, our system undergoes evaluation across three distinct sub-tasks: trajectory following, temporal action composition, and motion blending, where it outperforms previous state-of-the-art motion synthesis methods across the board. Homepage: https://story2motion.github.io/.

{{</citation>}}


### (87/142) FIRST: A Million-Entry Dataset for Text-Driven Fashion Synthesis and Design (Zhen Huang et al., 2023)

{{<citation>}}

Zhen Huang, Yihao Li, Dong Pei, Jiapeng Zhou, Xuliang Ning, Jianlin Han, Xiaoguang Han, Xuejun Chen. (2023)  
**FIRST: A Million-Entry Dataset for Text-Driven Fashion Synthesis and Design**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07414v1)  

---


**ABSTRACT**  
Text-driven fashion synthesis and design is an extremely valuable part of artificial intelligence generative content(AIGC), which has the potential to propel a tremendous revolution in the traditional fashion industry. To advance the research on text-driven fashion synthesis and design, we introduce a new dataset comprising a million high-resolution fashion images with rich structured textual(FIRST) descriptions. In the FIRST, there is a wide range of attire categories and each image-paired textual description is organized at multiple hierarchical levels. Experiments on prevalent generative models trained over FISRT show the necessity of FIRST. We invite the community to further develop more intelligent fashion synthesis and design systems that make fashion design more creative and imaginative based on our dataset. The dataset will be released soon.

{{</citation>}}


### (88/142) Evaluating the Significance of Outdoor Advertising from Driver's Perspective Using Computer Vision (Zuzana Černeková et al., 2023)

{{<citation>}}

Zuzana Černeková, Zuzana Berger Haladová, Ján Špirka, Viktor Kocur. (2023)  
**Evaluating the Significance of Outdoor Advertising from Driver's Perspective Using Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2311.07390v1)  

---


**ABSTRACT**  
Outdoor advertising, such as roadside billboards, plays a significant role in marketing campaigns but can also be a distraction for drivers, potentially leading to accidents. In this study, we propose a pipeline for evaluating the significance of roadside billboards in videos captured from a driver's perspective. We have collected and annotated a new BillboardLamac dataset, comprising eight videos captured by drivers driving through a predefined path wearing eye-tracking devices. The dataset includes annotations of billboards, including 154 unique IDs and 155 thousand bounding boxes, as well as eye fixation data. We evaluate various object tracking methods in combination with a YOLOv8 detector to identify billboard advertisements with the best approach achieving 38.5 HOTA on BillboardLamac. Additionally, we train a random forest classifier to classify billboards into three classes based on the length of driver fixations achieving 75.8% test accuracy. An analysis of the trained classifier reveals that the duration of billboard visibility, its saliency, and size are the most influential features when assessing billboard significance.

{{</citation>}}


### (89/142) Connecting the Dots: Graph Neural Network Powered Ensemble and Classification of Medical Images (Aryan Singh et al., 2023)

{{<citation>}}

Aryan Singh, Pepijn Van de Ven, Ciarán Eising, Patrick Denny. (2023)  
**Connecting the Dots: Graph Neural Network Powered Ensemble and Classification of Medical Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.07321v1)  

---


**ABSTRACT**  
Deep learning models have demonstrated remarkable results for various computer vision tasks, including the realm of medical imaging. However, their application in the medical domain is limited due to the requirement for large amounts of training data, which can be both challenging and expensive to obtain. To mitigate this, pre-trained models have been fine-tuned on domain-specific data, but such an approach can suffer from inductive biases. Furthermore, deep learning models struggle to learn the relationship between spatially distant features and their importance, as convolution operations treat all pixels equally. Pioneering a novel solution to this challenge, we employ the Image Foresting Transform to optimally segment images into superpixels. These superpixels are subsequently transformed into graph-structured data, enabling the proficient extraction of features and modeling of relationships using Graph Neural Networks (GNNs). Our method harnesses an ensemble of three distinct GNN architectures to boost its robustness. In our evaluations targeting pneumonia classification, our methodology surpassed prevailing Deep Neural Networks (DNNs) in performance, all while drastically cutting down on the parameter count. This not only trims down the expenses tied to data but also accelerates training and minimizes bias. Consequently, our proposition offers a sturdy, economically viable, and scalable strategy for medical image classification, significantly diminishing dependency on extensive training data sets.

{{</citation>}}


### (90/142) What Large Language Models Bring to Text-rich VQA? (Xuejing Liu et al., 2023)

{{<citation>}}

Xuejing Liu, Wei Tang, Xinzhe Ni, Jinghui Lu, Rui Zhao, Zechao Li, Fei Tan. (2023)  
**What Large Language Models Bring to Text-rich VQA?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, OCR, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.07306v1)  

---


**ABSTRACT**  
Text-rich VQA, namely Visual Question Answering based on text recognition in the images, is a cross-modal task that requires both image comprehension and text recognition. In this work, we focus on investigating the advantages and bottlenecks of LLM-based approaches in addressing this problem. To address the above concern, we separate the vision and language modules, where we leverage external OCR models to recognize texts in the image and Large Language Models (LLMs) to answer the question given texts. The whole framework is training-free benefiting from the in-context ability of LLMs. This pipeline achieved superior performance compared to the majority of existing Multimodal Large Language Models (MLLM) on four text-rich VQA datasets. Besides, based on the ablation study, we find that LLM brings stronger comprehension ability and may introduce helpful knowledge for the VQA problem. The bottleneck for LLM to address text-rich VQA problems may primarily lie in visual part. We also combine the OCR module with MLLMs and pleasantly find that the combination of OCR module with MLLM also works. It's worth noting that not all MLLMs can comprehend the OCR information, which provides insights into how to train an MLLM that preserves the abilities of LLM.

{{</citation>}}


### (91/142) Multi Sentence Description of Complex Manipulation Action Videos (Fatemeh Ziaeetabar et al., 2023)

{{<citation>}}

Fatemeh Ziaeetabar, Reza Safabakhsh, Saeedeh Momtazi, Minija Tamosiunaite, Florentin Wörgötter. (2023)  
**Multi Sentence Description of Complex Manipulation Action Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.07285v1)  

---


**ABSTRACT**  
Automatic video description requires the generation of natural language statements about the actions, events, and objects in the video. An important human trait, when we describe a video, is that we are able to do this with variable levels of detail. Different from this, existing approaches for automatic video descriptions are mostly focused on single sentence generation at a fixed level of detail. Instead, here we address video description of manipulation actions where different levels of detail are required for being able to convey information about the hierarchical structure of these actions relevant also for modern approaches of robot learning. We propose one hybrid statistical and one end-to-end framework to address this problem. The hybrid method needs much less data for training, because it models statistically uncertainties within the video clips, while in the end-to-end method, which is more data-heavy, we are directly connecting the visual encoder to the language decoder without any intermediate (statistical) processing step. Both frameworks use LSTM stacks to allow for different levels of description granularity and videos can be described by simple single-sentences or complex multiple-sentence descriptions. In addition, quantitative results demonstrate that these methods produce more realistic descriptions than other competing approaches.

{{</citation>}}


### (92/142) LT-ViT: A Vision Transformer for multi-label Chest X-ray classification (Umar Marikkar et al., 2023)

{{<citation>}}

Umar Marikkar, Sara Atito, Muhammad Awais, Adam Mahdi. (2023)  
**LT-ViT: A Vision Transformer for multi-label Chest X-ray classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.07263v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) are widely adopted in medical imaging tasks, and some existing efforts have been directed towards vision-language training for Chest X-rays (CXRs). However, we envision that there still exists a potential for improvement in vision-only training for CXRs using ViTs, by aggregating information from multiple scales, which has been proven beneficial for non-transformer networks. Hence, we have developed LT-ViT, a transformer that utilizes combined attention between image tokens and randomly initialized auxiliary tokens that represent labels. Our experiments demonstrate that LT-ViT (1) surpasses the state-of-the-art performance using pure ViTs on two publicly available CXR datasets, (2) is generalizable to other pre-training methods and therefore is agnostic to model initialization, and (3) enables model interpretability without grad-cam and its variants.

{{</citation>}}


### (93/142) Sketch-based Video Object Segmentation: Benchmark and Analysis (Ruolin Yang et al., 2023)

{{<citation>}}

Ruolin Yang, Da Li, Conghui Hu, Timothy Hospedales, Honggang Zhang, Yi-Zhe Song. (2023)  
**Sketch-based Video Object Segmentation: Benchmark and Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2311.07261v1)  

---


**ABSTRACT**  
Reference-based video object segmentation is an emerging topic which aims to segment the corresponding target object in each video frame referred by a given reference, such as a language expression or a photo mask. However, language expressions can sometimes be vague in conveying an intended concept and ambiguous when similar objects in one frame are hard to distinguish by language. Meanwhile, photo masks are costly to annotate and less practical to provide in a real application. This paper introduces a new task of sketch-based video object segmentation, an associated benchmark, and a strong baseline. Our benchmark includes three datasets, Sketch-DAVIS16, Sketch-DAVIS17 and Sketch-YouTube-VOS, which exploit human-drawn sketches as an informative yet low-cost reference for video object segmentation. We take advantage of STCN, a popular baseline of semi-supervised VOS task, and evaluate what the most effective design for incorporating a sketch reference is. Experimental results show sketch is more effective yet annotation-efficient than other references, such as photo masks, language and scribble.

{{</citation>}}


### (94/142) Simultaneous Clutter Detection and Semantic Segmentation of Moving Objects for Automotive Radar Data (Johannes Kopp et al., 2023)

{{<citation>}}

Johannes Kopp, Dominik Kellner, Aldi Piroli, Vinzenz Dallabetta, Klaus Dietmayer. (2023)  
**Simultaneous Clutter Detection and Semantic Segmentation of Moving Objects for Automotive Radar Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-SP  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2311.07247v2)  

---


**ABSTRACT**  
The unique properties of radar sensors, such as their robustness to adverse weather conditions, make them an important part of the environment perception system of autonomous vehicles. One of the first steps during the processing of radar point clouds is often the detection of clutter, i.e. erroneous points that do not correspond to real objects. Another common objective is the semantic segmentation of moving road users. These two problems are handled strictly separate from each other in literature. The employed neural networks are always focused entirely on only one of the tasks. In contrast to this, we examine ways to solve both tasks at the same time with a single jointly used model. In addition to a new augmented multi-head architecture, we also devise a method to represent a network's predictions for the two tasks with only one output value. This novel approach allows us to solve the tasks simultaneously with the same inference time as a conventional task-specific model. In an extensive evaluation, we show that our setup is highly effective and outperforms every existing network for semantic segmentation on the RadarScenes dataset.

{{</citation>}}


### (95/142) MonoDiffusion: Self-Supervised Monocular Depth Estimation Using Diffusion Model (Shuwei Shao et al., 2023)

{{<citation>}}

Shuwei Shao, Zhongcai Pei, Weihai Chen, Dingchi Sun, Peter C. Y. Chen, Zhengguo Li. (2023)  
**MonoDiffusion: Self-Supervised Monocular Depth Estimation Using Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.07198v1)  

---


**ABSTRACT**  
Over the past few years, self-supervised monocular depth estimation that does not depend on ground-truth during the training phase has received widespread attention. Most efforts focus on designing different types of network architectures and loss functions or handling edge cases, e.g., occlusion and dynamic objects. In this work, we introduce a novel self-supervised depth estimation framework, dubbed MonoDiffusion, by formulating it as an iterative denoising process. Because the depth ground-truth is unavailable in the training phase, we develop a pseudo ground-truth diffusion process to assist the diffusion in MonoDiffusion. The pseudo ground-truth diffusion gradually adds noise to the depth map generated by a pre-trained teacher model. Moreover,the teacher model allows applying a distillation loss to guide the denoised depth. Further, we develop a masked visual condition mechanism to enhance the denoising ability of model. Extensive experiments are conducted on the KITTI and Make3D datasets and the proposed MonoDiffusion outperforms prior state-of-the-art competitors. The source code will be available at https://github.com/ShuweiShao/MonoDiffusion.

{{</citation>}}


### (96/142) Cross-Axis Transformer with 2D Rotary Embeddings (Lily Erickson, 2023)

{{<citation>}}

Lily Erickson. (2023)  
**Cross-Axis Transformer with 2D Rotary Embeddings**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding, Microsoft, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.07184v1)  

---


**ABSTRACT**  
Despite lagging behind their modal cousins in many respects, Vision Transformers have provided an interesting opportunity to bridge the gap between sequence modeling and image modeling. Up until now however, vision transformers have largely been held back, due to both computational inefficiency, and lack of proper handling of spatial dimensions. In this paper, we introduce the Cross-Axis Transformer. CAT is a model inspired by both Axial Transformers, and Microsoft's recent Retentive Network, that drastically reduces the required number of floating point operations required to process an image, while simultaneously converging faster and more accurately than the Vision Transformers it replaces.

{{</citation>}}


### (97/142) Enhancing Lightweight Neural Networks for Small Object Detection in IoT Applications (Liam Boyle et al., 2023)

{{<citation>}}

Liam Boyle, Nicolas Baumann, Seonyeong Heo, Michele Magno. (2023)  
**Enhancing Lightweight Neural Networks for Small Object Detection in IoT Applications**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.07163v1)  

---


**ABSTRACT**  
Advances in lightweight neural networks have revolutionized computer vision in a broad range of IoT applications, encompassing remote monitoring and process automation. However, the detection of small objects, which is crucial for many of these applications, remains an underexplored area in current computer vision research, particularly for embedded devices. To address this gap, the paper proposes a novel adaptive tiling method that can be used on top of any existing object detector including the popular FOMO network for object detection on microcontrollers. Our experimental results show that the proposed tiling method can boost the F1-score by up to 225% while reducing the average object count error by up to 76%. Furthermore, the findings of this work suggest that using a soft F1 loss over the popular binary cross-entropy loss can significantly reduce the negative impact of imbalanced data. Finally, we validate our approach by conducting experiments on the Sony Spresense microcontroller, showcasing the proposed method's ability to strike a balance between detection performance, low latency, and minimal memory consumption.

{{</citation>}}


### (98/142) Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection (Junjie Huang et al., 2023)

{{<citation>}}

Junjie Huang, Yun Ye, Zhujin Liang, Yi Shan, Dalong Du. (2023)  
**Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.07152v1)  

---


**ABSTRACT**  
3D object Detection with LiDAR-camera encounters overfitting in algorithm development which is derived from the violation of some fundamental rules. We refer to the data annotation in dataset construction for theory complementing and argue that the regression task prediction should not involve the feature from the camera branch. By following the cutting-edge perspective of 'Detecting As Labeling', we propose a novel paradigm dubbed DAL. With the most classical elementary algorithms, a simple predicting pipeline is constructed by imitating the data annotation process. Then we train it in the simplest way to minimize its dependency and strengthen its portability. Though simple in construction and training, the proposed DAL paradigm not only substantially pushes the performance boundary but also provides a superior trade-off between speed and accuracy among all existing methods. With comprehensive superiority, DAL is an ideal baseline for both future work development and practical deployment. The code has been released to facilitate future work on https://github.com/HuangJunJie2017/BEVDet.

{{</citation>}}


### (99/142) PadChannel: Improving CNN Performance through Explicit Padding Encoding (Juho Kim, 2023)

{{<citation>}}

Juho Kim. (2023)  
**PadChannel: Improving CNN Performance through Explicit Padding Encoding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.07623v1)  

---


**ABSTRACT**  
In convolutional neural networks (CNNs), padding plays a pivotal role in preserving spatial dimensions throughout the layers. Traditional padding techniques do not explicitly distinguish between the actual image content and the padded regions, potentially causing CNNs to incorrectly interpret the boundary pixels or regions that resemble boundaries. This ambiguity can lead to suboptimal feature extraction. To address this, we propose PadChannel, a novel padding method that encodes padding statuses as an additional input channel, enabling CNNs to easily distinguish genuine pixels from padded ones. By incorporating PadChannel into several prominent CNN architectures, we observed small performance improvements and notable reductions in the variances on the ImageNet-1K image classification task at marginal increases in the computational cost. The source code is available at https://github.com/AussieSeaweed/pad-channel

{{</citation>}}


### (100/142) Attention-Challenging Multiple Instance Learning for Whole Slide Image Classification (Yunlong Zhang et al., 2023)

{{<citation>}}

Yunlong Zhang, Honglin Li, Yuxuan Sun, Sunyi Zheng, Chenglu Zhu, Lin Yang. (2023)  
**Attention-Challenging Multiple Instance Learning for Whole Slide Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Image Classification  
[Paper Link](http://arxiv.org/abs/2311.07125v1)  

---


**ABSTRACT**  
Overfitting remains a significant challenge in the application of Multiple Instance Learning (MIL) methods for Whole Slide Image (WSI) analysis. Visualizing heatmaps reveals that current MIL methods focus on a subset of predictive instances, hindering effective model generalization. To tackle this, we propose Attention-Challenging MIL (ACMIL), aimed at forcing the attention mechanism to capture more challenging predictive instances. ACMIL incorporates two techniques, Multiple Branch Attention (MBA) to capture richer predictive instances and Stochastic Top-K Instance Masking (STKIM) to suppress simple predictive instances. Evaluation on three WSI datasets outperforms state-of-the-art methods. Additionally, through heatmap visualization, UMAP visualization, and attention value statistics, this paper comprehensively illustrates ACMIL's effectiveness in overcoming the overfitting challenge. The source code is available at \url{https://github.com/dazhangyu123/ACMIL}.

{{</citation>}}


### (101/142) SpectralGPT: Spectral Foundation Model (Danfeng Hong et al., 2023)

{{<citation>}}

Danfeng Hong, Bing Zhang, Xuyang Li, Yuxuan Li, Chenyu Li, Jing Yao, Naoto Yokoya, Hao Li, Xiuping Jia, Antonio Plaza, Gamba Paolo, Jon Atli Benediktsson, Jocelyn Chanussot. (2023)  
**SpectralGPT: Spectral Foundation Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.07113v1)  

---


**ABSTRACT**  
The foundation model has recently garnered significant attention due to its potential to revolutionize the field of visual representation learning in a self-supervised manner. While most foundation models are tailored to effectively process RGB images for various visual tasks, there is a noticeable gap in research focused on spectral data, which offers valuable information for scene understanding, especially in remote sensing (RS) applications. To fill this gap, we created for the first time a universal RS foundation model, named SpectralGPT, which is purpose-built to handle spectral RS images using a novel 3D generative pretrained transformer (GPT). Compared to existing foundation models, SpectralGPT 1) accommodates input images with varying sizes, resolutions, time series, and regions in a progressive training fashion, enabling full utilization of extensive RS big data; 2) leverages 3D token generation for spatial-spectral coupling; 3) captures spectrally sequential patterns via multi-target reconstruction; 4) trains on one million spectral RS images, yielding models with over 600 million parameters. Our evaluation highlights significant performance improvements with pretrained SpectralGPT models, signifying substantial potential in advancing spectral RS big data applications within the field of geoscience across four downstream tasks: single/multi-label scene classification, semantic segmentation, and change detection.

{{</citation>}}


### (102/142) CLiF-VQA: Enhancing Video Quality Assessment by Incorporating High-Level Semantic Information related to Human Feelings (Yachun Mi et al., 2023)

{{<citation>}}

Yachun Mi, Yu Li, Yan Shu, Chen Hui, Puchao Zhou, Shaohui Liu. (2023)  
**CLiF-VQA: Enhancing Video Quality Assessment by Incorporating High-Level Semantic Information related to Human Feelings**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.07090v1)  

---


**ABSTRACT**  
Video Quality Assessment (VQA) aims to simulate the process of perceiving video quality by the human visual system (HVS). The judgments made by HVS are always influenced by human subjective feelings. However, most of the current VQA research focuses on capturing various distortions in the spatial and temporal domains of videos, while ignoring the impact of human feelings. In this paper, we propose CLiF-VQA, which considers both features related to human feelings and spatial features of videos. In order to effectively extract features related to human feelings from videos, we explore the consistency between CLIP and human feelings in video perception for the first time. Specifically, we design multiple objective and subjective descriptions closely related to human feelings as prompts. Further we propose a novel CLIP-based semantic feature extractor (SFE) which extracts features related to human feelings by sliding over multiple regions of the video frame. In addition, we further capture the low-level-aware features of the video through a spatial feature extraction module. The two different features are then aggregated thereby obtaining the quality score of the video. Extensive experiments show that the proposed CLiF-VQA exhibits excellent performance on several VQA datasets.

{{</citation>}}


### (103/142) Open-Vocabulary Video Anomaly Detection (Peng Wu et al., 2023)

{{<citation>}}

Peng Wu, Xuerong Zhou, Guansong Pang, Yujia Sun, Jing Liu, Peng Wang, Yanning Zhang. (2023)  
**Open-Vocabulary Video Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.07042v1)  

---


**ABSTRACT**  
Video anomaly detection (VAD) with weak supervision has achieved remarkable performance in utilizing video-level labels to discriminate whether a video frame is normal or abnormal. However, current approaches are inherently limited to a closed-set setting and may struggle in open-world applications where there can be anomaly categories in the test data unseen during training. A few recent studies attempt to tackle a more realistic setting, open-set VAD, which aims to detect unseen anomalies given seen anomalies and normal videos. However, such a setting focuses on predicting frame anomaly scores, having no ability to recognize the specific categories of anomalies, despite the fact that this ability is essential for building more informed video surveillance systems. This paper takes a step further and explores open-vocabulary video anomaly detection (OVVAD), in which we aim to leverage pre-trained large models to detect and categorize seen and unseen anomalies. To this end, we propose a model that decouples OVVAD into two mutually complementary tasks -- class-agnostic detection and class-specific classification -- and jointly optimizes both tasks. Particularly, we devise a semantic knowledge injection module to introduce semantic knowledge from large language models for the detection task, and design a novel anomaly synthesis module to generate pseudo unseen anomaly videos with the help of large vision generation models for the classification task. These semantic knowledge and synthesis anomalies substantially extend our model's capability in detecting and categorizing a variety of seen and unseen anomalies. Extensive experiments on three widely-used benchmarks demonstrate our model achieves state-of-the-art performance on OVVAD task.

{{</citation>}}


### (104/142) Pretrain like You Inference: Masked Tuning Improves Zero-Shot Composed Image Retrieval (Junyang Chen et al., 2023)

{{<citation>}}

Junyang Chen, Hanjiang Lai. (2023)  
**Pretrain like You Inference: Masked Tuning Improves Zero-Shot Composed Image Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.07622v1)  

---


**ABSTRACT**  
Zero-shot composed image retrieval (ZS-CIR), which aims to retrieve a target image based on textual modifications to a reference image without triplet labeling, has gained more and more attention. Current ZS-CIR research mainly relies on two unlabeled pre-trained models: the vision-language model, e.g., CLIP, and the Pic2Word/textual inversion model. However, the pre-trained models and CIR tasks have substantial discrepancies, where the pre-trained models learn the similarities between vision and language but CIR aims to learn the modifications of the image guided by text. In this paper, we introduce a novel unlabeled and pre-trained masked tuning approach to reduce the gap between the pre-trained model and the downstream CIR task. We first reformulate the pre-trained vision-language contrastive learning as the CIR task, where we randomly mask input image patches to generate $\langle$masked image, text, image$\rangle$ triple from an image-text pair. Then, we propose a masked tuning, which uses the text and the masked image to learn the modifications of the original image. With such a simple design, it can learn to capture fine-grained text-guided modifications. Extensive experimental results demonstrate the significant superiority of our approach over the baseline models on three ZS-CIR datasets, including FashionIQ, CIRR, and CIRCO.

{{</citation>}}


## cs.AI (5)



### (105/142) Enabling High-Level Machine Reasoning with Cognitive Neuro-Symbolic Systems (Alessandro Oltramari, 2023)

{{<citation>}}

Alessandro Oltramari. (2023)  
**Enabling High-Level Machine Reasoning with Cognitive Neuro-Symbolic Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.07759v1)  

---


**ABSTRACT**  
High-level reasoning can be defined as the capability to generalize over knowledge acquired via experience, and to exhibit robust behavior in novel situations. Such form of reasoning is a basic skill in humans, who seamlessly use it in a broad spectrum of tasks, from language communication to decision making in complex situations. When it manifests itself in understanding and manipulating the everyday world of objects and their interactions, we talk about common sense or commonsense reasoning. State-of-the-art AI systems don't possess such capability: for instance, Large Language Models have recently become popular by demonstrating remarkable fluency in conversing with humans, but they still make trivial mistakes when probed for commonsense competence; on a different level, performance degradation outside training data prevents self-driving vehicles to safely adapt to unseen scenarios, a serious and unsolved problem that limits the adoption of such technology. In this paper we propose to enable high-level reasoning in AI systems by integrating cognitive architectures with external neuro-symbolic components. We illustrate a hybrid framework centered on ACT-R and we discuss the role of generative models in recent and future applications.

{{</citation>}}


### (106/142) Generalization Analogies (GENIES): A Testbed for Generalizing AI Oversight to Hard-To-Measure Domains (Joshua Clymer et al., 2023)

{{<citation>}}

Joshua Clymer, Garrett Baker, Rohan Subramani, Sam Wang. (2023)  
**Generalization Analogies (GENIES): A Testbed for Generalizing AI Oversight to Hard-To-Measure Domains**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07723v1)  

---


**ABSTRACT**  
As AI systems become more intelligent and their behavior becomes more challenging to assess, they may learn to game the flaws of human feedback instead of genuinely striving to follow instructions; however, this risk can be mitigated by controlling how LLMs generalize human feedback to situations where it is unreliable. To better understand how reward models generalize, we craft 69 distribution shifts spanning 8 categories. We find that reward models do not learn to evaluate `instruction-following' by default and instead favor personas that resemble internet text. Techniques for interpreting reward models' internal representations achieve better generalization than standard fine-tuning, but still frequently fail to distinguish instruction-following from conflated behaviors. We consolidate the 15 most challenging distribution shifts into the GENaralization analogIES (GENIES) benchmark, which we hope will enable progress toward controlling reward model generalization.

{{</citation>}}


### (107/142) Reinforcement Learning for Solving Stochastic Vehicle Routing Problem (Zangir Iklassov et al., 2023)

{{<citation>}}

Zangir Iklassov, Ikboljon Sobirov, Ruben Solozabal, Martin Takac. (2023)  
**Reinforcement Learning for Solving Stochastic Vehicle Routing Problem**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CE, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.07708v1)  

---


**ABSTRACT**  
This study addresses a gap in the utilization of Reinforcement Learning (RL) and Machine Learning (ML) techniques in solving the Stochastic Vehicle Routing Problem (SVRP) that involves the challenging task of optimizing vehicle routes under uncertain conditions. We propose a novel end-to-end framework that comprehensively addresses the key sources of stochasticity in SVRP and utilizes an RL agent with a simple yet effective architecture and a tailored training method. Through comparative analysis, our proposed model demonstrates superior performance compared to a widely adopted state-of-the-art metaheuristic, achieving a significant 3.43% reduction in travel costs. Furthermore, the model exhibits robustness across diverse SVRP settings, highlighting its adaptability and ability to learn optimal routing strategies in varying environments. The publicly available implementation of our framework serves as a valuable resource for future research endeavors aimed at advancing RL-based solutions for SVRP.

{{</citation>}}


### (108/142) A Benchmark to Understand the Role of Knowledge Graphs on Large Language Model's Accuracy for Question Answering on Enterprise SQL Databases (Juan Sequeda et al., 2023)

{{<citation>}}

Juan Sequeda, Dean Allemang, Bryon Jacob. (2023)  
**A Benchmark to Understand the Role of Knowledge Graphs on Large Language Model's Accuracy for Question Answering on Enterprise SQL Databases**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-DB, cs.AI  
Keywords: GPT, GPT-4, Knowledge Graph, Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2311.07509v1)  

---


**ABSTRACT**  
Enterprise applications of Large Language Models (LLMs) hold promise for question answering on enterprise SQL databases. However, the extent to which LLMs can accurately respond to enterprise questions in such databases remains unclear, given the absence of suitable Text-to-SQL benchmarks tailored to enterprise settings. Additionally, the potential of Knowledge Graphs (KGs) to enhance LLM-based question answering by providing business context is not well understood. This study aims to evaluate the accuracy of LLM-powered question answering systems in the context of enterprise questions and SQL databases, while also exploring the role of knowledge graphs in improving accuracy. To achieve this, we introduce a benchmark comprising an enterprise SQL schema in the insurance domain, a range of enterprise queries encompassing reporting to metrics, and a contextual layer incorporating an ontology and mappings that define a knowledge graph. Our primary finding reveals that question answering using GPT-4, with zero-shot prompts directly on SQL databases, achieves an accuracy of 16%. Notably, this accuracy increases to 54% when questions are posed over a Knowledge Graph representation of the enterprise SQL database. Therefore, investing in Knowledge Graph provides higher accuracy for LLM powered question answering systems.

{{</citation>}}


### (109/142) Applying Large Language Models for Causal Structure Learning in Non Small Cell Lung Cancer (Narmada Naik et al., 2023)

{{<citation>}}

Narmada Naik, Ayush Khandelwal, Mohit Joshi, Madhusudan Atre, Hollis Wright, Kavya Kannan, Scott Hill, Giridhar Mamidipudi, Ganapati Srinivasa, Carlo Bifulco, Brian Piening, Kevin Matlock. (2023)  
**Applying Large Language Models for Causal Structure Learning in Non Small Cell Lung Cancer**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI, stat-AP  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07191v1)  

---


**ABSTRACT**  
Causal discovery is becoming a key part in medical AI research. These methods can enhance healthcare by identifying causal links between biomarkers, demographics, treatments and outcomes. They can aid medical professionals in choosing more impactful treatments and strategies. In parallel, Large Language Models (LLMs) have shown great potential in identifying patterns and generating insights from text data. In this paper we investigate applying LLMs to the problem of determining the directionality of edges in causal discovery. Specifically, we test our approach on a deidentified set of Non Small Cell Lung Cancer(NSCLC) patients that have both electronic health record and genomic panel data. Graphs are validated using Bayesian Dirichlet estimators using tabular data. Our result shows that LLMs can accurately predict the directionality of edges in causal graphs, outperforming existing state-of-the-art methods. These findings suggests that LLMs can play a significant role in advancing causal discovery and help us better understand complex systems.

{{</citation>}}


## eess.SY (1)



### (110/142) Synchrophasor Data Anomaly Detection on Grid Edge by 5G Communication and Adjacent Compute (Chuan Qin et al., 2023)

{{<citation>}}

Chuan Qin, Dexin Wang, Kishan Prudhvi Guddanti, Xiaoyuan Fan, Zhangshuan Hou. (2023)  
**Synchrophasor Data Anomaly Detection on Grid Edge by 5G Communication and Adjacent Compute**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.07758v1)  

---


**ABSTRACT**  
The fifth-generation mobile communication (5G) technology offers opportunities to enhance the real-time monitoring of grids. The 5G-enabled phasor measurement units (PMUs) feature flexible positioning and cost-effective long-term maintenance without the constraints of fixing wires. This paper is the first to demonstrate the applicability of 5G in PMU communication, and the experiment was carried out at Verizon non-standalone test-bed at Pacific Northwest National Laboratory (PNNL) Advanced Wireless Communication lab. The performance of the 5G-enabled PMU communication setup is reviewed and discussed in this paper, and a generalized dynamic linear model (GDLM) based real-time synchrophasor data anomaly detection use-case is presented. Last but not least, the practicability of implementing 5G for wide-area protection strategies is explored and discussed by analyzing the experimental results.

{{</citation>}}


## cs.CR (4)



### (111/142) An Extensive Study on Adversarial Attack against Pre-trained Models of Code (Xiaohu Du et al., 2023)

{{<citation>}}

Xiaohu Du, Ming Wen, Zichao Wei, Shangwen Wang, Hai Jin. (2023)  
**An Extensive Study on Adversarial Attack against Pre-trained Models of Code**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Adversarial Attack, Transformer  
[Paper Link](http://arxiv.org/abs/2311.07553v1)  

---


**ABSTRACT**  
Transformer-based pre-trained models of code (PTMC) have been widely utilized and have achieved state-of-the-art performance in many mission-critical applications. However, they can be vulnerable to adversarial attacks through identifier substitution or coding style transformation, which can significantly degrade accuracy and may further incur security concerns. Although several approaches have been proposed to generate adversarial examples for PTMC, the effectiveness and efficiency of such approaches, especially on different code intelligence tasks, has not been well understood. To bridge this gap, this study systematically analyzes five state-of-the-art adversarial attack approaches from three perspectives: effectiveness, efficiency, and the quality of generated examples. The results show that none of the five approaches balances all these perspectives. Particularly, approaches with a high attack success rate tend to be time-consuming; the adversarial code they generate often lack naturalness, and vice versa. To address this limitation, we explore the impact of perturbing identifiers under different contexts and find that identifier substitution within for and if statements is the most effective. Based on these findings, we propose a new approach that prioritizes different types of statements for various tasks and further utilizes beam search to generate adversarial examples. Evaluation results show that it outperforms the state-of-the-art ALERT in terms of both effectiveness and efficiency while preserving the naturalness of the generated adversarial examples.

{{</citation>}}


### (112/142) Tabdoor: Backdoor Vulnerabilities in Transformer-based Neural Networks for Tabular Data (Bart Pleiter et al., 2023)

{{<citation>}}

Bart Pleiter, Behrad Tajalli, Stefanos Koffas, Gorka Abad, Jing Xu, Martha Larson, Stjepan Picek. (2023)  
**Tabdoor: Backdoor Vulnerabilities in Transformer-based Neural Networks for Tabular Data**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.07550v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) have shown great promise in various domains. Alongside these developments, vulnerabilities associated with DNN training, such as backdoor attacks, are a significant concern. These attacks involve the subtle insertion of triggers during model training, allowing for manipulated predictions. More recently, DNNs for tabular data have gained increasing attention due to the rise of transformer models.   Our research presents a comprehensive analysis of backdoor attacks on tabular data using DNNs, particularly focusing on transformer-based networks. Given the inherent complexities of tabular data, we explore the challenges of embedding backdoors. Through systematic experimentation across benchmark datasets, we uncover that transformer-based DNNs for tabular data are highly susceptible to backdoor attacks, even with minimal feature value alterations. Our results indicate nearly perfect attack success rates (approx100%) by introducing novel backdoor attack strategies to tabular data. Furthermore, we evaluate several defenses against these attacks, identifying Spectral Signatures as the most effective one. Our findings highlight the urgency to address such vulnerabilities and provide insights into potential countermeasures for securing DNN models against backdoors on tabular data.

{{</citation>}}


### (113/142) The Privacy Pillar -- A Conceptual Framework for Foundation Model-based Systems (Tingting Bi et al., 2023)

{{<citation>}}

Tingting Bi, Guangsheng Yu, Qinghua Lu, Xiwei Xu, Nick Van Beest. (2023)  
**The Privacy Pillar -- A Conceptual Framework for Foundation Model-based Systems**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06998v1)  

---


**ABSTRACT**  
AI and its relevant technologies, including machine learning, deep learning, chatbots, virtual assistants, and others, are currently undergoing a profound transformation of development and organizational processes within companies. Foundation models present both significant challenges and incredible opportunities. In this context, ensuring the quality attributes of foundation model-based systems is of paramount importance, and with a particular focus on the challenging issue of privacy due to the sensitive nature of the data and information involved. However, there is currently a lack of consensus regarding the comprehensive scope of both technical and non-technical issues that the privacy evaluation process should encompass. Additionally, there is uncertainty about which existing methods are best suited to effectively address these privacy concerns. In response to this challenge, this paper introduces a novel conceptual framework that integrates various responsible AI patterns from multiple perspectives, with the specific aim of safeguarding privacy.

{{</citation>}}


### (114/142) AGRAMPLIFIER: Defending Federated Learning Against Poisoning Attacks Through Local Update Amplification (Zirui Gong et al., 2023)

{{<citation>}}

Zirui Gong, Liyue Shen, Yanjun Zhang, Leo Yu Zhang, Jingwei Wang, Guangdong Bai, Yong Xiang. (2023)  
**AGRAMPLIFIER: Defending Federated Learning Against Poisoning Attacks Through Local Update Amplification**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06996v1)  

---


**ABSTRACT**  
The collaborative nature of federated learning (FL) poses a major threat in the form of manipulation of local training data and local updates, known as the Byzantine poisoning attack. To address this issue, many Byzantine-robust aggregation rules (AGRs) have been proposed to filter out or moderate suspicious local updates uploaded by Byzantine participants.   This paper introduces a novel approach called AGRAMPLIFIER, aiming to simultaneously improve the robustness, fidelity, and efficiency of the existing AGRs. The core idea of AGRAMPLIFIER is to amplify the "morality" of local updates by identifying the most repressive features of each gradient update, which provides a clearer distinction between malicious and benign updates, consequently improving the detection effect. To achieve this objective, two approaches, namely AGRMP and AGRXAI, are proposed. AGRMP organizes local updates into patches and extracts the largest value from each patch, while AGRXAI leverages explainable AI methods to extract the gradient of the most activated features. By equipping AGRAMPLIFIER with the existing Byzantine-robust mechanisms, we successfully enhance the model's robustness, maintaining its fidelity and improving overall efficiency.   AGRAMPLIFIER is universally compatible with the existing Byzantine-robust mechanisms. The paper demonstrates its effectiveness by integrating it with all mainstream AGR mechanisms. Extensive evaluations conducted on seven datasets from diverse domains against seven representative poisoning attacks consistently show enhancements in robustness, fidelity, and efficiency, with average gains of 40.08%, 39.18%, and 10.68%, respectively.

{{</citation>}}


## stat.ML (1)



### (115/142) Estimating optical vegetation indices with Sentinel-1 SAR data and AutoML (Daniel Paluba et al., 2023)

{{<citation>}}

Daniel Paluba, Bertrand Le Saux, Francesco Sarti, Přemysl Stych. (2023)  
**Estimating optical vegetation indices with Sentinel-1 SAR data and AutoML**  

---
Primary Category: stat.ML  
Categories: I-4-8; I-4-9, cs-LG, stat-AP, stat-ML, stat.ML  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2311.07537v1)  

---


**ABSTRACT**  
Current optical vegetation indices (VIs) for monitoring forest ecosystems are widely used in various applications. However, continuous monitoring based on optical satellite data can be hampered by atmospheric effects such as clouds. On the contrary, synthetic aperture radar (SAR) data can offer insightful and systematic forest monitoring with complete time series due to signal penetration through clouds and day and night acquisitions. The goal of this work is to overcome the issues affecting optical data with SAR data and serve as a substitute for estimating optical VIs for forests using machine learning. Time series of four VIs (LAI, FAPAR, EVI and NDVI) were estimated using multitemporal Sentinel-1 SAR and ancillary data. This was enabled by creating a paired multi-temporal and multi-modal dataset in Google Earth Engine (GEE), including temporally and spatially aligned Sentinel-1, Sentinel-2, digital elevation model (DEM), weather and land cover datasets (MMT-GEE). The use of ancillary features generated from DEM and weather data improved the results. The open-source Automatic Machine Learning (AutoML) approach, auto-sklearn, outperformed Random Forest Regression for three out of four VIs, while a 1-hour optimization length was enough to achieve sufficient results with an R2 of 69-84% low errors (0.05-0.32 of MAE depending on VI). Great agreement was also found for selected case studies in the time series analysis and in the spatial comparison between the original and estimated SAR-based VIs. In general, compared to VIs from currently freely available optical satellite data and available global VI products, a better temporal resolution (up to 240 measurements/year) and a better spatial resolution (20 m) were achieved using estimated SAR-based VIs. A great advantage of the SAR-based VI is the ability to detect abrupt forest changes with a sub-weekly temporal accuracy.

{{</citation>}}


## q-fin.GN (1)



### (116/142) A Hypothesis on Good Practices for AI-based Systems for Financial Time Series Forecasting: Towards Domain-Driven XAI Methods (Branka Hadji Misheva et al., 2023)

{{<citation>}}

Branka Hadji Misheva, Joerg Osterrieder. (2023)  
**A Hypothesis on Good Practices for AI-based Systems for Financial Time Series Forecasting: Towards Domain-Driven XAI Methods**  

---
Primary Category: q-fin.GN  
Categories: cs-LG, q-fin-GN, q-fin.GN  
Keywords: AI, Financial, Time Series  
[Paper Link](http://arxiv.org/abs/2311.07513v1)  

---


**ABSTRACT**  
Machine learning and deep learning have become increasingly prevalent in financial prediction and forecasting tasks, offering advantages such as enhanced customer experience, democratising financial services, improving consumer protection, and enhancing risk management. However, these complex models often lack transparency and interpretability, making them challenging to use in sensitive domains like finance. This has led to the rise of eXplainable Artificial Intelligence (XAI) methods aimed at creating models that are easily understood by humans. Classical XAI methods, such as LIME and SHAP, have been developed to provide explanations for complex models. While these methods have made significant contributions, they also have limitations, including computational complexity, inherent model bias, sensitivity to data sampling, and challenges in dealing with feature dependence. In this context, this paper explores good practices for deploying explainability in AI-based systems for finance, emphasising the importance of data quality, audience-specific methods, consideration of data properties, and the stability of explanations. These practices aim to address the unique challenges and requirements of the financial industry and guide the development of effective XAI tools.

{{</citation>}}


## cs.HC (2)



### (117/142) ARWalker: A Virtual Walking Companion Application (Pubudu Wijesooriya et al., 2023)

{{<citation>}}

Pubudu Wijesooriya, Aaron Likens, Nick Stergiou, Spyridon Mastorakis. (2023)  
**ARWalker: A Virtual Walking Companion Application**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2311.07502v1)  

---


**ABSTRACT**  
Extended Reality (XR) technologies, including Augmented Reality (AR), have attracted significant attention over the past few years and have been utilized in several fields, including education, healthcare, and manufacturing. In this paper, we aim to explore the use of AR in the field of biomechanics and human movement through the development of ARWalker, which is an AR application that features virtual walking companions (avatars). Research participants walk in close synchrony with the virtual companions, whose gait exhibits properties found in the gait of young and healthy adults. As a result, research participants can train their gait to the gait of the avatar, thus regaining the healthy properties of their gait and reducing the risk of falls. ARWalker can especially help older adults and individuals with diseases, who exhibit pathological gait thus being more prone to falls. We implement a prototype of ARWalker and evaluate its systems performance while running on a Microsoft Hololens 2 headset.

{{</citation>}}


### (118/142) Understanding Users' Dissatisfaction with ChatGPT Responses: Types, Resolving Tactics, and the Effect of Knowledge Level (Yoonsu Kim et al., 2023)

{{<citation>}}

Yoonsu Kim, Jueon Lee, Seoyoung Kim, Jaehyuk Park, Juho Kim. (2023)  
**Understanding Users' Dissatisfaction with ChatGPT Responses: Types, Resolving Tactics, and the Effect of Knowledge Level**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2311.07434v1)  

---


**ABSTRACT**  
Large language models (LLMs) with chat-based capabilities, such as ChatGPT, are widely used in various workflows. However, due to a limited understanding of these large-scale models, users struggle to use this technology and experience different kinds of dissatisfaction. Researchers have introduced several methods such as prompt engineering to improve model responses. However, they focus on crafting one prompt, and little has been investigated on how to deal with the dissatisfaction the user encountered during the conversation. Therefore, with ChatGPT as the case study, we examine end users' dissatisfaction along with their strategies to address the dissatisfaction. After organizing users' dissatisfaction with LLM into seven categories based on a literature review, we collected 511 instances of dissatisfactory ChatGPT responses from 107 users and their detailed recollections of dissatisfied experiences, which we release as a publicly accessible dataset. Our analysis reveals that users most frequently experience dissatisfaction when ChatGPT fails to grasp their intentions, while they rate the severity of dissatisfaction the highest with dissatisfaction related to accuracy. We also identified four tactics users employ to address their dissatisfaction and their effectiveness. We found that users often do not use any tactics to address their dissatisfaction, and even when using tactics, 72% of dissatisfaction remained unresolved. Moreover, we found that users with low knowledge regarding LLMs tend to face more dissatisfaction on accuracy while they often put minimal effort in addressing dissatisfaction. Based on these findings, we propose design implications for minimizing user dissatisfaction and enhancing the usability of chat-based LLM services.

{{</citation>}}


## physics.geo-ph (1)



### (119/142) Three-dimensional granular flow simulation using graph neural network-based learned simulator (Yongjin Choi et al., 2023)

{{<citation>}}

Yongjin Choi, Krishna Kumar. (2023)  
**Three-dimensional granular flow simulation using graph neural network-based learned simulator**  

---
Primary Category: physics.geo-ph  
Categories: I-6-8, cs-LG, physics-geo-ph, physics.geo-ph  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.07416v1)  

---


**ABSTRACT**  
Reliable evaluations of geotechnical hazards like landslides and debris flow require accurate simulation of granular flow dynamics. Traditional numerical methods can simulate the complex behaviors of such flows that involve solid-like to fluid-like transitions, but they are computationally intractable when simulating large-scale systems. Surrogate models based on statistical or machine learning methods are a viable alternative, but they are typically empirical and rely on a confined set of parameters in evaluating associated risks. Due to their permutation-dependent learning, conventional machine learning models require an unreasonably large amount of training data for building generalizable surrogate models. We employ a graph neural network (GNN), a novel deep learning technique, to develop a GNN-based simulator (GNS) for granular flows to address these issues. Graphs represent the state of granular flows and interactions, like the exchange of energy and momentum between grains, and GNN learns the local interaction law. GNS takes the current state of the granular flow and estimates the next state using Euler explicit integration. We train GNS on a limited set of granular flow trajectories and evaluate its performance in a three-dimensional granular column collapse domain. GNS successfully reproduces the overall behaviors of column collapses with various aspect ratios that were not encountered during training. The computation speed of GNS outperforms high-fidelity numerical simulators by 300 times.

{{</citation>}}


## cs.SE (4)



### (120/142) Toward Optimal Psychological Functioning in AI-driven Software Engineering Tasks: The SEWELL-CARE Assessment Framework (Oussama Ben Sghaier et al., 2023)

{{<citation>}}

Oussama Ben Sghaier, Jean-Sebastien Boudrias, Houari Sahraoui. (2023)  
**Toward Optimal Psychological Functioning in AI-driven Software Engineering Tasks: The SEWELL-CARE Assessment Framework**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07410v1)  

---


**ABSTRACT**  
In the field of software engineering, there has been a shift towards utilizing various artificial intelligence techniques to address challenges and create innovative tools. These solutions are aimed at enhancing efficiency, automating tasks, and providing valuable support to developers. While the technical aspects are crucial, the well-being and psychology of the individuals performing these tasks are often overlooked. This paper argues that a holistic approach is essential, one that considers the technical, psychological, and social aspects of software engineering tasks. To address this gap, we introduce SEWELL-CARE, a conceptual framework designed to assess AI-driven software engineering tasks from multiple perspectives, with the goal of customizing the tools to improve the efficiency, well-being, and psychological functioning of developers. By emphasizing both technical and human dimensions, our framework provides a nuanced evaluation that goes beyond traditional technical metrics.

{{</citation>}}


### (121/142) Testing learning-enabled cyber-physical systems with Large-Language Models: A Formal Approach (Xi Zheng et al., 2023)

{{<citation>}}

Xi Zheng, Aloysius K. Mok, Ruzica Piskac, Yong Jae Lee, Bhaskar Krishnamachari, Dakai Zhu, Oleg Sokolsky, Insup Lee. (2023)  
**Testing learning-enabled cyber-physical systems with Large-Language Models: A Formal Approach**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-DC, cs-RO, cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07377v1)  

---


**ABSTRACT**  
The integration of machine learning (ML) into cyber-physical systems (CPS) offers significant benefits, including enhanced efficiency, predictive capabilities, real-time responsiveness, and the enabling of autonomous operations. This convergence has accelerated the development and deployment of a range of real-world applications, such as autonomous vehicles, delivery drones, service robots, and telemedicine procedures. However, the software development life cycle (SDLC) for AI-infused CPS diverges significantly from traditional approaches, featuring data and learning as two critical components. Existing verification and validation techniques are often inadequate for these new paradigms. In this study, we pinpoint the main challenges in ensuring formal safety for learningenabled CPS.We begin by examining testing as the most pragmatic method for verification and validation, summarizing the current state-of-the-art methodologies. Recognizing the limitations in current testing approaches to provide formal safety guarantees, we propose a roadmap to transition from foundational probabilistic testing to a more rigorous approach capable of delivering formal assurance.

{{</citation>}}


### (122/142) Past as a Guide: Leveraging Retrospective Learning for Python Code Completion (Seunggyoon Shin et al., 2023)

{{<citation>}}

Seunggyoon Shin, Seunggyu Chang, Sungjoon Choi. (2023)  
**Past as a Guide: Leveraging Retrospective Learning for Python Code Completion**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07635v1)  

---


**ABSTRACT**  
This work presents Past as a Guide (PaG), a simple approach for Large Language Models (LLMs) to improve the coding capabilities by integrating the past history with interactive and iterative code refinements. To be specific, inspired by human cognitive processes, the proposed method enables LLMs to utilize previous programming and debugging experiences to enhance the Python code completion tasks. The framework facilitates LLMs to iteratively refine the Python code based on previous execution and debugging results and optimize learning and reasoning capabilities. The proposed methodology achieved a 92\% pass@1 on HumanEval, demonstrating the potential to advance the field by leveraging retrospection from past experiences and interactive and iterative refinement processes without external correctness indicators.

{{</citation>}}


### (123/142) AdaCCD: Adaptive Semantic Contrasts Discovery based Cross Lingual Adaptation for Code Clone Detection (Yangkai Du et al., 2023)

{{<citation>}}

Yangkai Du, Tengfei Ma, Lingfei Wu, Xuhong Zhang, Shouling Ji. (2023)  
**AdaCCD: Adaptive Semantic Contrasts Discovery based Cross Lingual Adaptation for Code Clone Detection**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.07277v1)  

---


**ABSTRACT**  
Code Clone Detection, which aims to retrieve functionally similar programs from large code bases, has been attracting increasing attention. Modern software often involves a diverse range of programming languages. However, current code clone detection methods are generally limited to only a few popular programming languages due to insufficient annotated data as well as their own model design constraints. To address these issues, we present AdaCCD, a novel cross-lingual adaptation method that can detect cloned codes in a new language without any annotations in that language. AdaCCD leverages language-agnostic code representations from pre-trained programming language models and propose an Adaptively Refined Contrastive Learning framework to transfer knowledge from resource-rich languages to resource-poor languages. We evaluate the cross-lingual adaptation results of AdaCCD by constructing a multilingual code clone detection benchmark consisting of 5 programming languages. AdaCCD achieves significant improvements over other baselines, and it is even comparable to supervised fine-tuning.

{{</citation>}}


## q-bio.GN (2)



### (124/142) Attention-based Multi-task Learning for Base Editor Outcome Prediction (Amina Mollaysa et al., 2023)

{{<citation>}}

Amina Mollaysa, Ahmed Allam, Michael Krauthamme. (2023)  
**Attention-based Multi-task Learning for Base Editor Outcome Prediction**  

---
Primary Category: q-bio.GN  
Categories: cs-LG, q-bio-GN, q-bio.GN  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.07636v1)  

---


**ABSTRACT**  
Human genetic diseases often arise from point mutations, emphasizing the critical need for precise genome editing techniques. Among these, base editing stands out as it allows targeted alterations at the single nucleotide level. However, its clinical application is hindered by low editing efficiency and unintended mutations, necessitating extensive trial-and-error experimentation in the laboratory. To speed up this process, we present an attention-based two-stage machine learning model that learns to predict the likelihood of all possible editing outcomes for a given genomic target sequence. We further propose a multi-task learning schema to jointly learn multiple base editors (i.e. variants) at once. Our model's predictions consistently demonstrated a strong correlation with the actual experimental results on multiple datasets and base editor variants. These results provide further validation for the models' capacity to enhance and accelerate the process of refining base editing designs.

{{</citation>}}


### (125/142) To Transformers and Beyond: Large Language Models for the Genome (Micaela E. Consens et al., 2023)

{{<citation>}}

Micaela E. Consens, Cameron Dufault, Michael Wainberg, Duncan Forster, Mehran Karimzadeh, Hani Goodarzi, Fabian J. Theis, Alan Moses, Bo Wang. (2023)  
**To Transformers and Beyond: Large Language Models for the Genome**  

---
Primary Category: q-bio.GN  
Categories: cs-LG, q-bio-GN, q-bio.GN  
Keywords: Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.07621v1)  

---


**ABSTRACT**  
In the rapidly evolving landscape of genomics, deep learning has emerged as a useful tool for tackling complex computational challenges. This review focuses on the transformative role of Large Language Models (LLMs), which are mostly based on the transformer architecture, in genomics. Building on the foundation of traditional convolutional neural networks and recurrent neural networks, we explore both the strengths and limitations of transformers and other LLMs for genomics. Additionally, we contemplate the future of genomic modeling beyond the transformer architecture based on current trends in research. The paper aims to serve as a guide for computational biologists and computer scientists interested in LLMs for genomic data. We hope the paper can also serve as an educational introduction and discussion for biologists to a fundamental shift in how we will be analyzing genomic data in the future.

{{</citation>}}


## eess.AS (1)



### (126/142) Zero-Shot Duet Singing Voices Separation with Diffusion Models (Chin-Yun Yu et al., 2023)

{{<citation>}}

Chin-Yun Yu, Emilian Postolache, Emanuele Rodolà, György Fazekas. (2023)  
**Zero-Shot Duet Singing Voices Separation with Diffusion Models**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.07345v1)  

---


**ABSTRACT**  
In recent studies, diffusion models have shown promise as priors for solving audio inverse problems. These models allow us to sample from the posterior distribution of a target signal given an observed signal by manipulating the diffusion process. However, when separating audio sources of the same type, such as duet singing voices, the prior learned by the diffusion process may not be sufficient to maintain the consistency of the source identity in the separated audio. For example, the singer may change from one to another occasionally. Tackling this problem will be useful for separating sources in a choir, or a mixture of multiple instruments with similar timbre, without acquiring large amounts of paired data. In this paper, we examine this problem in the context of duet singing voices separation, and propose a method to enforce the coherency of singer identity by splitting the mixture into overlapping segments and performing posterior sampling in an auto-regressive manner, conditioning on the previous segment. We evaluate the proposed method on the MedleyVox dataset and show that the proposed method outperforms the naive posterior sampling baseline. Our source code and the pre-trained model are publicly available at https://github.com/yoyololicon/duet-svs-diffusion.

{{</citation>}}


## cs.NI (2)



### (127/142) When Distributed Consensus Meets Wireless Connected Autonomous Systems: A Review and A DAG-based Approach (Huanyu Wu et al., 2023)

{{<citation>}}

Huanyu Wu, Chentao Yue, Lei Zhang, Yonghui Li, Muhammad Ali Imran. (2023)  
**When Distributed Consensus Meets Wireless Connected Autonomous Systems: A Review and A DAG-based Approach**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07329v1)  

---


**ABSTRACT**  
The connected and autonomous systems (CAS) and auto-driving era is coming into our life. To support CAS applications such as AI-driven decision-making and blockchain-based smart data management platform, data and message exchange/dissemination is a fundamental element. The distributed message broadcast and forward protocols in CAS, such as vehicular ad hoc networks (VANET), can suffer from significant message loss and uncertain transmission delay, and faulty nodes might disseminate fake messages to confuse the network. Therefore, the consensus mechanism is essential in CAS with distributed structure to guaranteed correct nodes agree on the same parameter and reach consistency. However, due to the wireless nature of CAS, traditional consensus cannot be directly deployed. This article reviews several existing consensus mechanisms, including average/maximum/minimum estimation consensus mechanisms that apply on quantity, Byzantine fault tolerance consensus for request, state machine replication (SMR) and blockchain, as well as their implementations in CAS. To deploy wireless-adapted consensus, we propose a Directed Acyclic Graph (DAG)-based message structure to build a non-equivocation data dissemination protocol for CAS, which has resilience against message loss and unpredictable forwarding latency. Finally, we enhance this protocol by developing a two-dimension DAG-based strategy to achieve partial order for blockchain and total order for the distributed service model SMR.

{{</citation>}}


### (128/142) Effective In-vehicle Intrusion Detection via Multi-view Statistical Graph Learning on CAN Messages (Kai Wang et al., 2023)

{{<citation>}}

Kai Wang, Qiguang Jiang, Bailing Wang, Yongzheng Zhang, Yulei Wu. (2023)  
**Effective In-vehicle Intrusion Detection via Multi-view Statistical Graph Learning on CAN Messages**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-CR, cs-NI, cs.NI  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2311.07056v1)  

---


**ABSTRACT**  
As an important component of internet of vehicles (IoV), intelligent connected vehicles (ICVs) have to communicate with external networks frequently. In this case, the resource-constrained in-vehicle network (IVN) is facing a wide variety of complex and changing external cyber-attacks, especially the masquerade attack with high difficulty of detection while serious damaging effects that few counter measures can identify successfully. Moreover, only coarse-grained recognition can be achieved in current mainstream intrusion detection mechanisms, i.e., whether a whole data flow observation window contains attack labels rather than fine-grained recognition on every single data item within this window. In this paper, we propose StatGraph: an Effective Multi-view Statistical Graph Learning Intrusion Detection to implement the fine-grained intrusion detection. Specifically, StatGraph generates two statistical graphs, timing correlation graph (TCG) and coupling relationship graph (CRG), based on data streams. In given message observation windows, edge attributes in TCGs represent temporal correlation between different message IDs, while edge attributes in CRGs denote the neighbour relationship and contextual similarity. Besides, a lightweight shallow layered GCN network is trained based graph property of TCGs and CRGs, which can learn the universal laws of various patterns more effectively and further enhance the performance of detection. To address the problem of insufficient attack types in previous intrusion detection, we select two real in-vehicle CAN datasets that cover four new attacks never investigated before. Experimental result shows StatGraph improves both detection granularity and detection performance over state-of-the-art intrusion detection methods.

{{</citation>}}


## cs.RO (4)



### (129/142) TIAGo RL: Simulated Reinforcement Learning Environments with Tactile Data for Mobile Robots (Luca Lach et al., 2023)

{{<citation>}}

Luca Lach, Francesco Ferro, Robert Haschke. (2023)  
**TIAGo RL: Simulated Reinforcement Learning Environments with Tactile Data for Mobile Robots**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.07260v1)  

---


**ABSTRACT**  
Tactile information is important for robust performance in robotic tasks that involve physical interaction, such as object manipulation. However, with more data included in the reasoning and control process, modeling behavior becomes increasingly difficult. Deep Reinforcement Learning (DRL) produced promising results for learning complex behavior in various domains, including tactile-based manipulation in robotics. In this work, we present our open-source reinforcement learning environments for the TIAGo service robot. They produce tactile sensor measurements that resemble those of a real sensorised gripper for TIAGo, encouraging research in transfer learning of DRL policies. Lastly, we show preliminary training results of a learned force control policy and compare it to a classical PI controller.

{{</citation>}}


### (130/142) Large Language Models for Robotics: A Survey (Fanlong Zeng et al., 2023)

{{<citation>}}

Fanlong Zeng, Wensheng Gan, Yongheng Wang, Ning Liu, Philip S. Yu. (2023)  
**Large Language Models for Robotics: A Survey**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07226v1)  

---


**ABSTRACT**  
The human ability to learn, generalize, and control complex manipulation tasks through multi-modality feedback suggests a unique capability, which we refer to as dexterity intelligence. Understanding and assessing this intelligence is a complex task. Amidst the swift progress and extensive proliferation of large language models (LLMs), their applications in the field of robotics have garnered increasing attention. LLMs possess the ability to process and generate natural language, facilitating efficient interaction and collaboration with robots. Researchers and engineers in the field of robotics have recognized the immense potential of LLMs in enhancing robot intelligence, human-robot interaction, and autonomy. Therefore, this comprehensive review aims to summarize the applications of LLMs in robotics, delving into their impact and contributions to key areas such as robot control, perception, decision-making, and path planning. We first provide an overview of the background and development of LLMs for robotics, followed by a description of the benefits of LLMs for robotics and recent advancements in robotics models based on LLMs. We then delve into the various techniques used in the model, including those employed in perception, decision-making, control, and interaction. Finally, we explore the applications of LLMs in robotics and some potential challenges they may face in the near future. Embodied intelligence is the future of intelligent science, and LLMs-based robotics is one of the promising but challenging paths to achieve this.

{{</citation>}}


### (131/142) Interaction is all You Need? A Study of Robots Ability to Understand and Execute (Kushal Koshti et al., 2023)

{{<citation>}}

Kushal Koshti, Nidhir Bhavsar. (2023)  
**Interaction is all You Need? A Study of Robots Ability to Understand and Execute**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-CV, cs-RO, cs.RO  
Keywords: Dialog, LLaMA  
[Paper Link](http://arxiv.org/abs/2311.07150v1)  

---


**ABSTRACT**  
This paper aims to address a critical challenge in robotics, which is enabling them to operate seamlessly in human environments through natural language interactions. Our primary focus is to equip robots with the ability to understand and execute complex instructions in coherent dialogs to facilitate intricate task-solving scenarios. To explore this, we build upon the Execution from Dialog History (EDH) task from the Teach benchmark. We employ a multi-transformer model with BART LM. We observe that our best configuration outperforms the baseline with a success rate score of 8.85 and a goal-conditioned success rate score of 14.02. In addition, we suggest an alternative methodology for completing this task. Moreover, we introduce a new task by expanding the EDH task and making predictions about game plans instead of individual actions. We have evaluated multiple BART models and an LLaMA2 LLM, which has achieved a ROGUE-L score of 46.77 for this task.

{{</citation>}}


### (132/142) Collaborative Goal Tracking of Multiple Mobile Robots Based on Geometric Graph Neural Network (Qingquan Lin et al., 2023)

{{<citation>}}

Qingquan Lin, Weining Lu. (2023)  
**Collaborative Goal Tracking of Multiple Mobile Robots Based on Geometric Graph Neural Network**  

---
Primary Category: cs.RO  
Categories: cs-MA, cs-RO, cs.RO  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.07105v1)  

---


**ABSTRACT**  
Multi-robot systems are widely used in spatially distributed tasks, and their collaborative path planning is of great significance for working efficiency. Currently, different multi-robot collaborative path planning methods have been proposed, but how to process the sensory information of neighboring robots at different locations from a local perception perspective in real environment to make better decisions is still a major difficulty. To address this problem, this paper proposes a multi-robot collaborative path planning method based on geometric graph neural network (GeoGNN). GeoGNN introduces the relative position information of neighboring robots into each interaction layer of the graph neural network to better integrate neighbor sensing information. An expert data generation method is designed for the robot to advance in a single step, by which expert data are generated in ROS to train the network. Experimental results show that the accuracy of the proposed method is improved by about 5% compared to the model based only on CNN on the expert data set. In ROS simulation environment path planning test, the success rate is improved by about 4% compared to CNN and flowtime increase is reduced about 8%, which outperforms other graph neural network models.

{{</citation>}}


## cs.CE (1)



### (133/142) RESenv: A Realistic Earthquake Simulation Environment based on Unreal Engine (Yitong Sun et al., 2023)

{{<citation>}}

Yitong Sun, Hanchun Wang, Zhejun Zhang, Cyriel Diels, Ali Asadipour. (2023)  
**RESenv: A Realistic Earthquake Simulation Environment based on Unreal Engine**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs-RO, cs.CE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07239v1)  

---


**ABSTRACT**  
Earthquakes have a significant impact on societies and economies, driving the need for effective search and rescue strategies. With the growing role of AI and robotics in these operations, high-quality synthetic visual data becomes crucial. Current simulation methods, mostly focusing on single building damages, often fail to provide realistic visuals for complex urban settings. To bridge this gap, we introduce an innovative earthquake simulation system using the Chaos Physics System in Unreal Engine. Our approach aims to offer detailed and realistic visual simulations essential for AI and robotic training in rescue missions. By integrating real seismic waveform data, we enhance the authenticity and relevance of our simulations, ensuring they closely mirror real-world earthquake scenarios. Leveraging the advanced capabilities of Unreal Engine, our system delivers not only high-quality visualisations but also real-time dynamic interactions, making the simulated environments more immersive and responsive. By providing advanced renderings, accurate physical interactions, and comprehensive geological movements, our solution outperforms traditional methods in efficiency and user experience. Our simulation environment stands out in its detail and realism, making it a valuable tool for AI tasks such as path planning and image recognition related to earthquake responses. We validate our approach through three AI-based tasks: similarity detection, path planning, and image segmentation.

{{</citation>}}


## eess.IV (2)



### (134/142) Multi-task learning for joint weakly-supervised segmentation and aortic arch anomaly classification in fetal cardiac MRI (Paula Ramirez et al., 2023)

{{<citation>}}

Paula Ramirez, Alena Uus, Milou P. M. van Poppel, Irina Grigorescu, Johannes K. Steinweg, David F. A. Lloyd, Kuberan Pushparajah, Andrew P. King, Maria Deprez. (2023)  
**Multi-task learning for joint weakly-supervised segmentation and aortic arch anomaly classification in fetal cardiac MRI**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.07234v1)  

---


**ABSTRACT**  
Congenital Heart Disease (CHD) is a group of cardiac malformations present already during fetal life, representing the prevailing category of birth defects globally. Our aim in this study is to aid 3D fetal vessel topology visualisation in aortic arch anomalies, a group which encompasses a range of conditions with significant anatomical heterogeneity. We present a multi-task framework for automated multi-class fetal vessel segmentation from 3D black blood T2w MRI and anomaly classification. Our training data consists of binary manual segmentation masks of the cardiac vessels' region in individual subjects and fully-labelled anomaly-specific population atlases. Our framework combines deep learning label propagation using VoxelMorph with 3D Attention U-Net segmentation and DenseNet121 anomaly classification. We target 11 cardiac vessels and three distinct aortic arch anomalies, including double aortic arch, right aortic arch, and suspected coarctation of the aorta. We incorporate an anomaly classifier into our segmentation pipeline, delivering a multi-task framework with the primary motivation of correcting topological inaccuracies of the segmentation. The hypothesis is that the multi-task approach will encourage the segmenter network to learn anomaly-specific features. As a secondary motivation, an automated diagnosis tool may have the potential to enhance diagnostic confidence in a decision support setting. Our results showcase that our proposed training strategy significantly outperforms label propagation and a network trained exclusively on propagated labels. Our classifier outperforms a classifier trained exclusively on T2w volume images, with an average balanced accuracy of 0.99 (0.01) after joint training. Adding a classifier improves the anatomical and topological accuracy of all correctly classified double aortic arch subjects.

{{</citation>}}


### (135/142) TTMFN: Two-stream Transformer-based Multimodal Fusion Network for Survival Prediction (Ruiquan Ge et al., 2023)

{{<citation>}}

Ruiquan Ge, Xiangyang Hu, Rungen Huang, Gangyong Jia, Yaqi Wang, Renshu Gu, Changmiao Wang, Elazab Ahmed, Linyan Wang, Juan Ye, Ye Li. (2023)  
**TTMFN: Two-stream Transformer-based Multimodal Fusion Network for Survival Prediction**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.07033v1)  

---


**ABSTRACT**  
Survival prediction plays a crucial role in assisting clinicians with the development of cancer treatment protocols. Recent evidence shows that multimodal data can help in the diagnosis of cancer disease and improve survival prediction. Currently, deep learning-based approaches have experienced increasing success in survival prediction by integrating pathological images and gene expression data. However, most existing approaches overlook the intra-modality latent information and the complex inter-modality correlations. Furthermore, existing modalities do not fully exploit the immense representational capabilities of neural networks for feature aggregation and disregard the importance of relationships between features. Therefore, it is highly recommended to address these issues in order to enhance the prediction performance by proposing a novel deep learning-based method. We propose a novel framework named Two-stream Transformer-based Multimodal Fusion Network for survival prediction (TTMFN), which integrates pathological images and gene expression data. In TTMFN, we present a two-stream multimodal co-attention transformer module to take full advantage of the complex relationships between different modalities and the potential connections within the modalities. Additionally, we develop a multi-head attention pooling approach to effectively aggregate the feature representations of the two modalities. The experiment results on four datasets from The Cancer Genome Atlas demonstrate that TTMFN can achieve the best performance or competitive results compared to the state-of-the-art methods in predicting the overall survival of patients.

{{</citation>}}


## cs.IR (2)



### (136/142) On Elastic Language Models (Chen Zhang et al., 2023)

{{<citation>}}

Chen Zhang, Benyou Wang, Dawei Song. (2023)  
**On Elastic Language Models**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: GLUE, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.07204v1)  

---


**ABSTRACT**  
Large-scale pretrained language models have achieved compelling performance in a wide range of language understanding and information retrieval tasks. Knowledge distillation offers an opportunity to compress a large language model to a small one, in order to reach a reasonable latency-performance tradeoff. However, for scenarios where the number of requests (e.g., queries submitted to a search engine) is highly variant, the static tradeoff attained by the compressed language model might not always fit. Once a model is assigned with a static tradeoff, it could be inadequate in that the latency is too high when the number of requests is large or the performance is too low when the number of requests is small. To this end, we propose an elastic language model (ElasticLM) that elastically adjusts the tradeoff according to the request stream. The basic idea is to introduce a compute elasticity to the compressed language model, so that the tradeoff could vary on-the-fly along scalable and controllable compute. Specifically, we impose an elastic structure to enable ElasticLM with compute elasticity and design an elastic optimization to learn ElasticLM under compute elasticity. To serve ElasticLM, we apply an elastic schedule. Considering the specificity of information retrieval, we adapt ElasticLM to dense retrieval and reranking and present ElasticDenser and ElasticRanker respectively. Offline evaluation is conducted on a language understanding benchmark GLUE; and several information retrieval tasks including Natural Question, Trivia QA, and MS MARCO. The results show that ElasticLM along with ElasticDenser and ElasticRanker can perform correctly and competitively compared with an array of static baselines. Furthermore, online simulation with concurrency is also carried out. The results demonstrate that ElasticLM can provide elastic tradeoffs with respect to varying request stream.

{{</citation>}}


### (137/142) Do LLMs Implicitly Exhibit User Discrimination in Recommendation? An Empirical Study (Chen Xu et al., 2023)

{{<citation>}}

Chen Xu, Wenjie Wang, Yuxin Li, Liang Pang, Jun Xu, Tat-Seng Chua. (2023)  
**Do LLMs Implicitly Exhibit User Discrimination in Recommendation? An Empirical Study**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07054v1)  

---


**ABSTRACT**  
Recently, Large Language Models (LLMs) have enhanced user interaction, enabling seamless information retrieval and recommendations. However, concerns emerge as these LLMs have shown tendencies to display discrimination related to users' sensitive characteristics (such as gender), leading to explicit user unfairness. Furthermore, our analysis uncovers a more discreet variant of bias in LLMs, defined as implicit user unfairness, wherein these models demonstrate discriminatory recommendation behaviors based solely on non-sensitive user details, like usernames or email addresses. This subtle form of unfairness, while more pervasive, poses a significant threat to the ethical integrity and rights of minority user groups. To comprehensively explore implicit user unfairness, our analysis unfolds in three key steps: (1) We uncover the reasons for this implicit user unfairness: LLMs can infer users' sensitive attributes from non-sensitive attributes (e.g. user names) due to their extensive world knowledge. (2) Our findings expose that the magnitude of implicit user unfairness within LLMs surpasses the level of explicit user unfairness observed in traditional recommender models, signifying a more alarming issue of unfairness, i.e. some non-sensitive features of users like names may result in more serious discrimination phenomena. (3) We analyze the long-term effect of implicit user unfairness, identifying that it will reinforce information bubbles at an accelerated rate compared to traditional RS. We emphasize the need to identify and mitigate implicit user unfairness, aiming to avert the potential human-LLMs recommendation systems deterioration.

{{</citation>}}


## quant-ph (1)



### (138/142) Optical Quantum Sensing for Agnostic Environments via Deep Learning (Zeqiao Zhou et al., 2023)

{{<citation>}}

Zeqiao Zhou, Yuxuan Du, Xu-Fei Yin, Shanshan Zhao, Xinmei Tian, Dacheng Tao. (2023)  
**Optical Quantum Sensing for Agnostic Environments via Deep Learning**  

---
Primary Category: quant-ph  
Categories: cs-AI, physics-optics, quant-ph, quant-ph  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.07203v1)  

---


**ABSTRACT**  
Optical quantum sensing promises measurement precision beyond classical sensors termed the Heisenberg limit (HL). However, conventional methodologies often rely on prior knowledge of the target system to achieve HL, presenting challenges in practical applications. Addressing this limitation, we introduce an innovative Deep Learning-based Quantum Sensing scheme (DQS), enabling optical quantum sensors to attain HL in agnostic environments. DQS incorporates two essential components: a Graph Neural Network (GNN) predictor and a trigonometric interpolation algorithm. Operating within a data-driven paradigm, DQS utilizes the GNN predictor, trained on offline data, to unveil the intrinsic relationships between the optical setups employed in preparing the probe state and the resulting quantum Fisher information (QFI) after interaction with the agnostic environment. This distilled knowledge facilitates the identification of optimal optical setups associated with maximal QFI. Subsequently, DQS employs a trigonometric interpolation algorithm to recover the unknown parameter estimates for the identified optical setups. Extensive experiments are conducted to investigate the performance of DQS under different settings up to eight photons. Our findings not only offer a new lens through which to accelerate optical quantum sensing tasks but also catalyze future research integrating deep learning and quantum mechanics.

{{</citation>}}


## cs.ET (1)



### (139/142) Pruning random resistive memory for optimizing analogue AI (Yi Li et al., 2023)

{{<citation>}}

Yi Li, Songqi Wang, Yaping Zhao, Shaocong Wang, Woyu Zhang, Yangu He, Ning Lin, Binbin Cui, Xi Chen, Shiming Zhang, Hao Jiang, Peng Lin, Xumeng Zhang, Xiaojuan Qi, Zhongrui Wang, Xiaoxin Xu, Dashan Shang, Qi Liu, Kwang-Ting Cheng, Ming Liu. (2023)  
**Pruning random resistive memory for optimizing analogue AI**  

---
Primary Category: cs.ET  
Categories: cs-AI, cs-AR, cs-ET, cs.ET  
Keywords: AI, Pruning  
[Paper Link](http://arxiv.org/abs/2311.07164v1)  

---


**ABSTRACT**  
The rapid advancement of artificial intelligence (AI) has been marked by the large language models exhibiting human-like intelligence. However, these models also present unprecedented challenges to energy consumption and environmental sustainability. One promising solution is to revisit analogue computing, a technique that predates digital computing and exploits emerging analogue electronic devices, such as resistive memory, which features in-memory computing, high scalability, and nonvolatility. However, analogue computing still faces the same challenges as before: programming nonidealities and expensive programming due to the underlying devices physics. Here, we report a universal solution, software-hardware co-design using structural plasticity-inspired edge pruning to optimize the topology of a randomly weighted analogue resistive memory neural network. Software-wise, the topology of a randomly weighted neural network is optimized by pruning connections rather than precisely tuning resistive memory weights. Hardware-wise, we reveal the physical origin of the programming stochasticity using transmission electron microscopy, which is leveraged for large-scale and low-cost implementation of an overparameterized random neural network containing high-performance sub-networks. We implemented the co-design on a 40nm 256K resistive memory macro, observing 17.3% and 19.9% accuracy improvements in image and audio classification on FashionMNIST and Spoken digits datasets, as well as 9.8% (2%) improvement in PR (ROC) in image segmentation on DRIVE datasets, respectively. This is accompanied by 82.1%, 51.2%, and 99.8% improvement in energy efficiency thanks to analogue in-memory computing. By embracing the intrinsic stochasticity and in-memory computing, this work may solve the biggest obstacle of analogue computing systems and thus unleash their immense potential for next-generation AI hardware.

{{</citation>}}


## cs.SI (1)



### (140/142) Untargeted Black-box Attacks for Social Recommendations (Wenqi Fan et al., 2023)

{{<citation>}}

Wenqi Fan, Shijie Wang, Xiao-yong Wei, Xiaowei Mei, Qing Li. (2023)  
**Untargeted Black-box Attacks for Social Recommendations**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.07127v1)  

---


**ABSTRACT**  
The rise of online social networks has facilitated the evolution of social recommender systems, which incorporate social relations to enhance users' decision-making process. With the great success of Graph Neural Networks in learning node representations, GNN-based social recommendations have been widely studied to model user-item interactions and user-user social relations simultaneously. Despite their great successes, recent studies have shown that these advanced recommender systems are highly vulnerable to adversarial attacks, in which attackers can inject well-designed fake user profiles to disrupt recommendation performances. While most existing studies mainly focus on targeted attacks to promote target items on vanilla recommender systems, untargeted attacks to degrade the overall prediction performance are less explored on social recommendations under a black-box scenario. To perform untargeted attacks on social recommender systems, attackers can construct malicious social relationships for fake users to enhance the attack performance. However, the coordination of social relations and item profiles is challenging for attacking black-box social recommendations. To address this limitation, we first conduct several preliminary studies to demonstrate the effectiveness of cross-community connections and cold-start items in degrading recommendations performance. Specifically, we propose a novel framework Multiattack based on multi-agent reinforcement learning to coordinate the generation of cold-start item profiles and cross-community social relations for conducting untargeted attacks on black-box social recommendations. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of our proposed attacking framework under the black-box setting.

{{</citation>}}


## econ.GN (1)



### (141/142) The Impact of Generative Artificial Intelligence (Kaichen Zhang et al., 2023)

{{<citation>}}

Kaichen Zhang, Ohchan Kwon, Hui Xiong. (2023)  
**The Impact of Generative Artificial Intelligence**  

---
Primary Category: econ.GN  
Categories: cs-AI, econ-GN, econ.GN, q-fin-EC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.07071v1)  

---


**ABSTRACT**  
The rise of generative artificial intelligence (AI) has sparked concerns about its potential influence on unemployment and market depression. This study addresses this concern by examining the impact of generative AI on product markets. To overcome the challenge of causal inference, given the inherent limitations of conducting controlled experiments, this paper identifies an unanticipated and sudden leak of a highly proficient image-generative AI as a novel instance of a "natural experiment". This AI leak spread rapidly, significantly reducing the cost of generating anime-style images compared to other styles, creating an opportunity for comparative assessment. We collect real-world data from an artwork outsourcing platform. Surprisingly, our results show that while generative AI lowers average prices, it substantially boosts order volume and overall revenue. This counterintuitive finding suggests that generative AI confers benefits upon artists rather than detriments. The study further offers theoretical economic explanations to elucidate this unexpected phenomenon. By furnishing empirical evidence, this paper dispels the notion that generative AI might engender depression, instead underscoring its potential to foster market prosperity. These findings carry significant implications for practitioners, policymakers, and the broader AI community.

{{</citation>}}


## cs.IT (1)



### (142/142) Deep Joint Source Channel Coding With Attention Modules Over MIMO Channels (Weiran Jiang et al., 2023)

{{<citation>}}

Weiran Jiang, Wei Chen, Bo Ai. (2023)  
**Deep Joint Source Channel Coding With Attention Modules Over MIMO Channels**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.07041v1)  

---


**ABSTRACT**  
In this paper, we propose two deep joint source and channel coding (DJSCC) structures with attention modules for the multi-input multi-output (MIMO) channel, including a serial structure and a parallel structure. With singular value decomposition (SVD)-based precoding scheme, the MIMO channel can be decomposed into various sub-channels, and the feature outputs will experience sub-channels with different channel qualities. In the serial structure, one single network is used at both the transmitter and the receiver to jointly process data streams of all MIMO subchannels, while data steams of different MIMO subchannels are processed independently via multiple sub-networks in the parallel structure. The attention modules in both serial and parallel architectures enable the system to adapt to varying channel qualities and adjust the quantity of information outputs in accordance with the channel qualities. Experimental results demonstrate the proposed DJSCC structures have improved image transmission performance, and reveal the phenomenon via non-parameter entropy estimation that the learned DJSCC transceivers tend to transmit more information over better sub-channels.

{{</citation>}}
