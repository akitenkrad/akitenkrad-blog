---
draft: false
title: "arXiv @ 2024.01.20"
date: 2024-01-20
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.20"
    identifier: arxiv_20240120
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AR (2)](#csar-2)
- [cs.CL (26)](#cscl-26)
- [q-bio.BM (2)](#q-biobm-2)
- [eess.AS (3)](#eessas-3)
- [cs.CV (30)](#cscv-30)
- [eess.SP (2)](#eesssp-2)
- [cs.LG (25)](#cslg-25)
- [cs.SE (5)](#csse-5)
- [q-fin.CP (1)](#q-fincp-1)
- [cs.NI (2)](#csni-2)
- [cs.CR (6)](#cscr-6)
- [cs.MA (1)](#csma-1)
- [q-bio.NC (1)](#q-bionc-1)
- [stat.ME (1)](#statme-1)
- [cs.AI (3)](#csai-3)
- [cs.NE (3)](#csne-3)
- [cs.RO (3)](#csro-3)
- [cs.DC (3)](#csdc-3)
- [cs.SD (2)](#cssd-2)
- [cs.ET (1)](#cset-1)
- [cs.SI (2)](#cssi-2)
- [cs.GT (1)](#csgt-1)
- [eess.SY (2)](#eesssy-2)
- [cs.MM (1)](#csmm-1)
- [cs.SC (1)](#cssc-1)
- [math.NA (1)](#mathna-1)
- [cs.HC (1)](#cshc-1)
- [astro-ph.IM (1)](#astro-phim-1)

## cs.AR (2)



### (1/132) SSR: Spatial Sequential Hybrid Architecture for Latency Throughput Tradeoff in Transformer Acceleration (Jinming Zhuang et al., 2024)

{{<citation>}}

Jinming Zhuang, Zhuoping Yang, Shixin Ji, Heng Huang, Alex K. Jones, Jingtong Hu, Yiyu Shi, Peipei Zhou. (2024)  
**SSR: Spatial Sequential Hybrid Architecture for Latency Throughput Tradeoff in Transformer Acceleration**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-PL, cs.AR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.10417v1)  

---


**ABSTRACT**  
With the increase in the computation intensity of the chip, the mismatch between computation layer shapes and the available computation resource significantly limits the utilization of the chip. Driven by this observation, prior works discuss spatial accelerators or dataflow architecture to maximize the throughput. However, using spatial accelerators could potentially increase the execution latency. In this work, we first systematically investigate two execution models: (1) sequentially (temporally) launch one monolithic accelerator, and (2) spatially launch multiple accelerators. From the observations, we find that there is a latency throughput tradeoff between these two execution models, and combining these two strategies together can give us a more efficient latency throughput Pareto front. To achieve this, we propose spatial sequential architecture (SSR) and SSR design automation framework to explore both strategies together when deploying deep learning inference. We use the 7nm AMD Versal ACAP VCK190 board to implement SSR accelerators for four end-to-end transformer-based deep learning models. SSR achieves average throughput gains of 2.53x, 35.71x, and 14.20x under different batch sizes compared to the 8nm Nvidia GPU A10G, 16nm AMD FPGAs ZCU102, and U250. The average energy efficiency gains are 8.51x, 6.75x, and 21.22x, respectively. Compared with the sequential-only solution and spatial-only solution on VCK190, our spatial-sequential-hybrid solutions achieve higher throughput under the same latency requirement and lower latency under the same throughput requirement. We also use SSR analytical models to demonstrate how to use SSR to optimize solutions on other computing platforms, e.g., 14nm Intel Stratix 10 NX.

{{</citation>}}


### (2/132) A Survey on Hardware Accelerators for Large Language Models (Christoforos Kachris, 2024)

{{<citation>}}

Christoforos Kachris. (2024)  
**A Survey on Hardware Accelerators for Large Language Models**  

---
Primary Category: cs.AR  
Categories: B-5; C-1; C-3, cs-AR, cs-CL, cs-LG, cs.AR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09890v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have emerged as powerful tools for natural language processing tasks, revolutionizing the field with their ability to understand and generate human-like text. As the demand for more sophisticated LLMs continues to grow, there is a pressing need to address the computational challenges associated with their scale and complexity. This paper presents a comprehensive survey on hardware accelerators designed to enhance the performance and energy efficiency of Large Language Models. By examining a diverse range of accelerators, including GPUs, FPGAs, and custom-designed architectures, we explore the landscape of hardware solutions tailored to meet the unique computational demands of LLMs. The survey encompasses an in-depth analysis of architecture, performance metrics, and energy efficiency considerations, providing valuable insights for researchers, engineers, and decision-makers aiming to optimize the deployment of LLMs in real-world applications.

{{</citation>}}


## cs.CL (26)



### (3/132) Can Large Language Model Summarizers Adapt to Diverse Scientific Communication Goals? (Marcio Fonseca et al., 2024)

{{<citation>}}

Marcio Fonseca, Shay B. Cohen. (2024)  
**Can Large Language Model Summarizers Adapt to Diverse Scientific Communication Goals?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10415v1)  

---


**ABSTRACT**  
In this work, we investigate the controllability of large language models (LLMs) on scientific summarization tasks. We identify key stylistic and content coverage factors that characterize different types of summaries such as paper reviews, abstracts, and lay summaries. By controlling stylistic features, we find that non-fine-tuned LLMs outperform humans in the MuP review generation task, both in terms of similarity to reference summaries and human preferences. Also, we show that we can improve the controllability of LLMs with keyword-based classifier-free guidance (CFG) while achieving lexical overlap comparable to strong fine-tuned baselines on arXiv and PubMed. However, our results also indicate that LLMs cannot consistently generate long summaries with more than 8 sentences. Furthermore, these models exhibit limited capacity to produce highly abstractive lay summaries. Although LLMs demonstrate strong generic summarization competency, sophisticated content control without costly fine-tuning remains an open problem for domain-specific applications.

{{</citation>}}


### (4/132) Learning High-Quality and General-Purpose Phrase Representations (Lihu Chen et al., 2024)

{{<citation>}}

Lihu Chen, Gaël Varoquaux, Fabian M. Suchanek. (2024)  
**Learning High-Quality and General-Purpose Phrase Representations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Entity Alignment  
[Paper Link](http://arxiv.org/abs/2401.10407v1)  

---


**ABSTRACT**  
Phrase representations play an important role in data science and natural language processing, benefiting various tasks like Entity Alignment, Record Linkage, Fuzzy Joins, and Paraphrase Classification. The current state-of-the-art method involves fine-tuning pre-trained language models for phrasal embeddings using contrastive learning. However, we have identified areas for improvement. First, these pre-trained models tend to be unnecessarily complex and require to be pre-trained on a corpus with context sentences. Second, leveraging the phrase type and morphology gives phrase representations that are both more precise and more flexible. We propose an improved framework to learn phrase representations in a context-free fashion. The framework employs phrase type classification as an auxiliary task and incorporates character-level information more effectively into the phrase representation. Furthermore, we design three granularities of data augmentation to increase the diversity of training samples. Our experiments across a wide range of tasks show that our approach generates superior phrase embeddings compared to previous methods while requiring a smaller model size. The code is available at \faGithub~ \url{https://github.com/tigerchen52/PEARL} \end{abstract}

{{</citation>}}


### (5/132) Inconsistent dialogue responses and how to recover from them (Mian Zhang et al., 2024)

{{<citation>}}

Mian Zhang, Lifeng Jin, Linfeng Song, Haitao Mi, Dong Yu. (2024)  
**Inconsistent dialogue responses and how to recover from them**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.10353v1)  

---


**ABSTRACT**  
One critical issue for chat systems is to stay consistent about preferences, opinions, beliefs and facts of itself, which has been shown a difficult problem. In this work, we study methods to assess and bolster utterance consistency of chat systems. A dataset is first developed for studying the inconsistencies, where inconsistent dialogue responses, explanations of the inconsistencies, and recovery utterances are authored by annotators. This covers the life span of inconsistencies, namely introduction, understanding, and resolution. Building on this, we introduce a set of tasks centered on dialogue consistency, specifically focused on its detection and resolution. Our experimental findings indicate that our dataset significantly helps the progress in identifying and resolving conversational inconsistencies, and current popular large language models like ChatGPT which are good at resolving inconsistencies however still struggle with detection.

{{</citation>}}


### (6/132) Bridging Cultural Nuances in Dialogue Agents through Cultural Value Surveys (Yong Cao et al., 2024)

{{<citation>}}

Yong Cao, Min Chen, Daniel Hershcovich. (2024)  
**Bridging Cultural Nuances in Dialogue Agents through Cultural Value Surveys**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2401.10352v1)  

---


**ABSTRACT**  
The cultural landscape of interactions with dialogue agents is a compelling yet relatively unexplored territory. It's clear that various sociocultural aspects -- from communication styles and beliefs to shared metaphors and knowledge -- profoundly impact these interactions. To delve deeper into this dynamic, we introduce cuDialog, a first-of-its-kind benchmark for dialogue generation with a cultural lens. We also develop baseline models capable of extracting cultural attributes from dialogue exchanges, with the goal of enhancing the predictive accuracy and quality of dialogue agents. To effectively co-learn cultural understanding and multi-turn dialogue predictions, we propose to incorporate cultural dimensions with dialogue encoding features. Our experimental findings highlight that incorporating cultural value surveys boosts alignment with references and cultural markers, demonstrating its considerable influence on personalization and dialogue quality. To facilitate further exploration in this exciting domain, we publish our benchmark publicly accessible at https://github.com/yongcaoplus/cuDialog.

{{</citation>}}


### (7/132) ChatQA: Building GPT-4 Level Conversational QA Models (Zihan Liu et al., 2024)

{{<citation>}}

Zihan Liu, Wei Ping, Rajarshi Roy, Peng Xu, Chankyu Lee, Mohammad Shoeybi, Bryan Catanzaro. (2024)  
**ChatQA: Building GPT-4 Level Conversational QA Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: AI, GPT, GPT-4, QA  
[Paper Link](http://arxiv.org/abs/2401.10225v2)  

---


**ABSTRACT**  
In this work, we introduce ChatQA, a family of conversational question answering (QA) models that obtain GPT-4 level accuracies. Specifically, we propose a two-stage instruction tuning method that can significantly improve the zero-shot conversational QA results from large language models (LLMs). To handle retrieval-augmented generation in conversational QA, we fine-tune a dense retriever on a multi-turn QA dataset, which provides comparable results to using the state-of-the-art query rewriting model while largely reducing deployment cost. Notably, our ChatQA-70B can outperform GPT-4 in terms of average score on 10 conversational QA datasets (54.14 vs. 53.90), without relying on any synthetic data from OpenAI GPT models.

{{</citation>}}


### (8/132) Chem-FINESE: Validating Fine-Grained Few-shot Entity Extraction through Text Reconstruction (Qingyun Wang et al., 2024)

{{<citation>}}

Qingyun Wang, Zixuan Zhang, Hongxiang Li, Xuan Liu, Jiawei Han, Heng Ji, Huimin Zhao. (2024)  
**Chem-FINESE: Validating Fine-Grained Few-shot Entity Extraction through Text Reconstruction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2401.10189v2)  

---


**ABSTRACT**  
Fine-grained few-shot entity extraction in the chemical domain faces two unique challenges. First, compared with entity extraction tasks in the general domain, sentences from chemical papers usually contain more entities. Moreover, entity extraction models usually have difficulty extracting entities of long-tailed types. In this paper, we propose Chem-FINESE, a novel sequence-to-sequence (seq2seq) based few-shot entity extraction approach, to address these two challenges. Our Chem-FINESE has two components: a seq2seq entity extractor to extract named entities from the input sentence and a seq2seq self-validation module to reconstruct the original input sentence from extracted entities. Inspired by the fact that a good entity extraction system needs to extract entities faithfully, our new self-validation module leverages entity extraction results to reconstruct the original input sentence. Besides, we design a new contrastive loss to reduce excessive copying during the extraction process. Finally, we release ChemNER+, a new fine-grained chemical entity extraction dataset that is annotated by domain experts with the ChemNER schema. Experiments in few-shot settings with both ChemNER+ and CHEMET datasets show that our newly proposed framework has contributed up to 8.26% and 6.84% absolute F1-score gains respectively.

{{</citation>}}


### (9/132) Beyond Reference-Based Metrics: Analyzing Behaviors of Open LLMs on Data-to-Text Generation (Zdeněk Kasner et al., 2024)

{{<citation>}}

Zdeněk Kasner, Ondřej Dušek. (2024)  
**Beyond Reference-Based Metrics: Analyzing Behaviors of Open LLMs on Data-to-Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Text Generation  
[Paper Link](http://arxiv.org/abs/2401.10186v1)  

---


**ABSTRACT**  
We investigate to which extent open large language models (LLMs) can generate coherent and relevant text from structured data. To prevent bias from benchmarks leaked into LLM training data, we collect Quintd-1: an ad-hoc benchmark for five data-to-text (D2T) generation tasks, consisting of structured data records in standard formats gathered from public APIs. We leverage reference-free evaluation metrics and LLMs' in-context learning capabilities, allowing us to test the models with no human-written references. Our evaluation focuses on annotating semantic accuracy errors on token-level, combining human annotators and a metric based on GPT-4. Our systematic examination of the models' behavior across domains and tasks suggests that state-of-the-art open LLMs with 7B parameters can generate fluent and coherent text from various standard data formats in zero-shot settings. However, we also show that semantic accuracy of the outputs remains a major issue: on our benchmark, 80% of outputs of open LLMs contain a semantic error according to human annotators (91% according to GPT-4). Our code, data, and model outputs are available at https://d2t-llm.github.io.

{{</citation>}}


### (10/132) Marrying Adapters and Mixup to Efficiently Enhance the Adversarial Robustness of Pre-Trained Language Models for Text Classification (Tuc Nguyen et al., 2024)

{{<citation>}}

Tuc Nguyen, Thai Le. (2024)  
**Marrying Adapters and Mixup to Efficiently Enhance the Adversarial Robustness of Pre-Trained Language Models for Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Text Classification  
[Paper Link](http://arxiv.org/abs/2401.10111v1)  

---


**ABSTRACT**  
Existing works show that augmenting training data of neural networks using both clean and adversarial examples can enhance their generalizability under adversarial attacks. However, this training approach often leads to performance degradation on clean inputs. Additionally, it requires frequent re-training of the entire model to account for new attack types, resulting in significant and costly computations. Such limitations make adversarial training mechanisms less practical, particularly for complex Pre-trained Language Models (PLMs) with millions or even billions of parameters. To overcome these challenges while still harnessing the theoretical benefits of adversarial training, this study combines two concepts: (1) adapters, which enable parameter-efficient fine-tuning, and (2) Mixup, which train NNs via convex combinations of pairs data pairs. Intuitively, we propose to fine-tune PLMs through convex combinations of non-data pairs of fine-tuned adapters, one trained with clean and another trained with adversarial examples. Our experiments show that the proposed method achieves the best trade-off between training efficiency and predictive performance, both with and without attacks compared to other baselines on a variety of downstream tasks.

{{</citation>}}


### (11/132) Power in Numbers: Robust reading comprehension by finetuning with four adversarial sentences per example (Ariel Marcus, 2024)

{{<citation>}}

Ariel Marcus. (2024)  
**Power in Numbers: Robust reading comprehension by finetuning with four adversarial sentences per example**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Question Answering  
[Paper Link](http://arxiv.org/abs/2401.10091v1)  

---


**ABSTRACT**  
Recent models have achieved human level performance on the Stanford Question Answering Dataset when using F1 scores to evaluate the reading comprehension task. Yet, teaching machines to comprehend text has not been solved in the general case. By appending one adversarial sentence to the context paragraph, past research has shown that the F1 scores from reading comprehension models drop almost in half. In this paper, I replicate past adversarial research with a new model, ELECTRA-Small, and demonstrate that the new model's F1 score drops from 83.9% to 29.2%. To improve ELECTRA-Small's resistance to this attack, I finetune the model on SQuAD v1.1 training examples with one to five adversarial sentences appended to the context paragraph. Like past research, I find that the finetuned model on one adversarial sentence does not generalize well across evaluation datasets. However, when finetuned on four or five adversarial sentences the model attains an F1 score of more than 70% on most evaluation datasets with multiple appended and prepended adversarial sentences. The results suggest that with enough examples we can make models robust to adversarial attacks.

{{</citation>}}


### (12/132) Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs (Haritz Puerto et al., 2024)

{{<citation>}}

Haritz Puerto, Martin Tutek, Somak Aditya, Xiaodan Zhu, Iryna Gurevych. (2024)  
**Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.10065v1)  

---


**ABSTRACT**  
Reasoning is a fundamental component for achieving language understanding. Among the multiple types of reasoning, conditional reasoning, the ability to draw different conclusions depending on some condition, has been understudied in large language models (LLMs). Recent prompting methods, such as chain of thought, have significantly improved LLMs on reasoning tasks. Nevertheless, there is still little understanding of what triggers reasoning abilities in LLMs. We hypothesize that code prompts can trigger conditional reasoning in LLMs trained on text and code. We propose a chain of prompts that transforms a natural language problem into code and prompts the LLM with the generated code. Our experiments find that code prompts exhibit a performance boost between 2.6 and 7.7 points on GPT 3.5 across multiple datasets requiring conditional reasoning. We then conduct experiments to discover how code prompts elicit conditional reasoning abilities and through which features. We observe that prompts need to contain natural language text accompanied by high-quality code that closely represents the semantics of the instance text. Furthermore, we show that code prompts are more efficient, requiring fewer demonstrations, and that they trigger superior state tracking of variables or key entities.

{{</citation>}}


### (13/132) Large Language Models for Scientific Information Extraction: An Empirical Study for Virology (Mahsa Shamsabadi et al., 2024)

{{<citation>}}

Mahsa Shamsabadi, Jennifer D'Souza, Sören Auer. (2024)  
**Large Language Models for Scientific Information Extraction: An Empirical Study for Virology**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-DL, cs-IT, cs.CL, math-IT  
Keywords: Amazon, GPT, Information Extraction, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2401.10040v1)  

---


**ABSTRACT**  
In this paper, we champion the use of structured and semantic content representation of discourse-based scholarly communication, inspired by tools like Wikipedia infoboxes or structured Amazon product descriptions. These representations provide users with a concise overview, aiding scientists in navigating the dense academic landscape. Our novel automated approach leverages the robust text generation capabilities of LLMs to produce structured scholarly contribution summaries, offering both a practical solution and insights into LLMs' emergent abilities.   For LLMs, the prime focus is on improving their general intelligence as conversational agents. We argue that these models can also be applied effectively in information extraction (IE), specifically in complex IE tasks within terse domains like Science. This paradigm shift replaces the traditional modular, pipelined machine learning approach with a simpler objective expressed through instructions. Our results show that finetuned FLAN-T5 with 1000x fewer parameters than the state-of-the-art GPT-davinci is competitive for the task.

{{</citation>}}


### (14/132) Self-Rewarding Language Models (Weizhe Yuan et al., 2024)

{{<citation>}}

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, Jason Weston. (2024)  
**Self-Rewarding Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.10020v1)  

---


**ABSTRACT**  
We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training. In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613. While only a preliminary study, this work opens the door to the possibility of models that can continually improve in both axes.

{{</citation>}}


### (15/132) R-Judge: Benchmarking Safety Risk Awareness for LLM Agents (Tongxin Yuan et al., 2024)

{{<citation>}}

Tongxin Yuan, Zhiwei He, Lingzhong Dong, Yiming Wang, Ruijie Zhao, Tian Xia, Lizhen Xu, Binglin Zhou, Fangqi Li, Zhuosheng Zhang, Rui Wang, Gongshen Liu. (2024)  
**R-Judge: Benchmarking Safety Risk Awareness for LLM Agents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.10019v1)  

---


**ABSTRACT**  
Large language models (LLMs) have exhibited great potential in autonomously completing tasks across real-world applications. Despite this, these LLM agents introduce unexpected safety risks when operating in interactive environments. Instead of centering on LLM-generated content safety in most prior studies, this work addresses the imperative need for benchmarking the behavioral safety of LLM agents within diverse environments. We introduce R-Judge, a benchmark crafted to evaluate the proficiency of LLMs in judging safety risks given agent interaction records. R-Judge comprises 162 agent interaction records, encompassing 27 key risk scenarios among 7 application categories and 10 risk types. It incorporates human consensus on safety with annotated safety risk labels and high-quality risk descriptions. Utilizing R-Judge, we conduct a comprehensive evaluation of 8 prominent LLMs commonly employed as the backbone for agents. The best-performing model, GPT-4, achieves 72.29% in contrast to the human score of 89.38%, showing considerable room for enhancing the risk awareness of LLMs. Notably, leveraging risk descriptions as environment feedback significantly improves model performance, revealing the importance of salient safety risk feedback. Furthermore, we design an effective chain of safety analysis technique to help the judgment of safety risks and conduct an in-depth case study to facilitate future research. R-Judge is publicly available at https://github.com/Lordog/R-Judge.

{{</citation>}}


### (16/132) Gender Bias in Machine Translation and The Era of Large Language Models (Eva Vanmassenhove, 2024)

{{<citation>}}

Eva Vanmassenhove. (2024)  
**Gender Bias in Machine Translation and The Era of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: Bias, ChatGPT, GPT, GPT-3.5, Language Model, Machine Translation, Transformer  
[Paper Link](http://arxiv.org/abs/2401.10016v1)  

---


**ABSTRACT**  
This chapter examines the role of Machine Translation in perpetuating gender bias, highlighting the challenges posed by cross-linguistic settings and statistical dependencies. A comprehensive overview of relevant existing work related to gender bias in both conventional Neural Machine Translation approaches and Generative Pretrained Transformer models employed as Machine Translation systems is provided. Through an experiment using ChatGPT (based on GPT-3.5) in an English-Italian translation context, we further assess ChatGPT's current capacity to address gender bias. The findings emphasize the ongoing need for advancements in mitigating bias in Machine Translation systems and underscore the importance of fostering fairness and inclusivity in language technologies.

{{</citation>}}


### (17/132) Towards Hierarchical Spoken Language Dysfluency Modeling (Jiachen Lian et al., 2024)

{{<citation>}}

Jiachen Lian, Gopala Anumanchipalli. (2024)  
**Towards Hierarchical Spoken Language Dysfluency Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.10015v2)  

---


**ABSTRACT**  
Speech disfluency modeling is the bottleneck for both speech therapy and language learning. However, there is no effective AI solution to systematically tackle this problem. We solidify the concept of disfluent speech and disfluent speech modeling. We then present Hierarchical Unconstrained Disfluency Modeling (H-UDM) approach, the hierarchical extension of UDM that addresses both disfluency transcription and detection to eliminate the need for extensive manual annotation. Our experimental findings serve as clear evidence of the effectiveness and reliability of the methods we have introduced, encompassing both transcription and detection tasks.

{{</citation>}}


### (18/132) Distantly Supervised Morpho-Syntactic Model for Relation Extraction (Nicolas Gutehrlé et al., 2024)

{{<citation>}}

Nicolas Gutehrlé, Iana Atanassova. (2024)  
**Distantly Supervised Morpho-Syntactic Model for Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2401.10002v1)  

---


**ABSTRACT**  
The task of Information Extraction (IE) involves automatically converting unstructured textual content into structured data. Most research in this field concentrates on extracting all facts or a specific set of relationships from documents. In this paper, we present a method for the extraction and categorisation of an unrestricted set of relationships from text. Our method relies on morpho-syntactic extraction patterns obtained by a distant supervision method, and creates Syntactic and Semantic Indices to extract and classify candidate graphs. We evaluate our approach on six datasets built on Wikidata and Wikipedia. The evaluation shows that our approach can achieve Precision scores of up to 0.85, but with lower Recall and F1 scores. Our approach allows to quickly create rule-based systems for Information Extraction and to build annotated datasets to train machine-learning and deep-learning based classifiers.

{{</citation>}}


### (19/132) Gradable ChatGPT Translation Evaluation (Hui Jiao et al., 2024)

{{<citation>}}

Hui Jiao, Bei Peng, Lu Zong, Xiaojun Zhang, Xinwei Li. (2024)  
**Gradable ChatGPT Translation Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.09984v1)  

---


**ABSTRACT**  
ChatGPT, as a language model based on large-scale pre-training, has exerted a profound influence on the domain of machine translation. In ChatGPT, a "Prompt" refers to a segment of text or instruction employed to steer the model towards generating a specific category of response. The design of the translation prompt emerges as a key aspect that can wield influence over factors such as the style, precision and accuracy of the translation to a certain extent. However, there is a lack of a common standard and methodology on how to design and select a translation prompt. Accordingly, this paper proposes a generic taxonomy, which defines gradable translation prompts in terms of expression type, translation style, POS information and explicit statement, thus facilitating the construction of prompts endowed with distinct attributes tailored for various translation tasks. Specific experiments and cases are selected to validate and illustrate the effectiveness of the method.

{{</citation>}}


### (20/132) Better Explain Transformers by Illuminating Important Information (Linxin Song et al., 2024)

{{<citation>}}

Linxin Song, Yan Cui, Ao Luo, Freddy Lecue, Irene Li. (2024)  
**Better Explain Transformers by Illuminating Important Information**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.09972v2)  

---


**ABSTRACT**  
Transformer-based models excel in various natural language processing (NLP) tasks, attracting countless efforts to explain their inner workings. Prior methods explain Transformers by focusing on the raw gradient and attention as token attribution scores, where non-relevant information is often considered during explanation computation, resulting in confusing results. In this work, we propose highlighting the important information and eliminating irrelevant information by a refined information flow on top of the layer-wise relevance propagation (LRP) method. Specifically, we consider identifying syntactic and positional heads as important attention heads and focus on the relevance obtained from these important heads. Experimental results demonstrate that irrelevant information does distort output attribution scores and then should be masked during explanation computation. Compared to eight baselines on both classification and question-answering datasets, our method consistently outperforms with over 3\% to 33\% improvement on explanation metrics, providing superior explanation performance. Our anonymous code repository is available at: https://github.com/LinxinS97/Mask-LRP

{{</citation>}}


### (21/132) Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access (Saibo Geng et al., 2024)

{{<citation>}}

Saibo Geng, Berkay Döner, Chris Wendler, Martin Josifoski, Robert West. (2024)  
**Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, Sketch  
[Paper Link](http://arxiv.org/abs/2401.09967v1)  

---


**ABSTRACT**  
Constrained decoding, a technique for enforcing constraints on language model outputs, offers a way to control text generation without retraining or architectural modifications. Its application is, however, typically restricted to models that give users access to next-token distributions (usually via softmax logits), which poses a limitation with blackbox large language models (LLMs). This paper introduces sketch-guided constrained decoding (SGCD), a novel approach to constrained decoding for blackbox LLMs, which operates without access to the logits of the blackbox LLM. SGCD utilizes a locally hosted auxiliary model to refine the output of an unconstrained blackbox LLM, effectively treating this initial output as a "sketch" for further elaboration. This approach is complementary to traditional logit-based techniques and enables the application of constrained decoding in settings where full model transparency is unavailable. We demonstrate the efficacy of SGCD through experiments in closed information extraction and constituency parsing, showing how it enhances the utility and flexibility of blackbox LLMs for complex NLP tasks.

{{</citation>}}


### (22/132) MatSciRE: Leveraging Pointer Networks to Automate Entity and Relation Extraction for Material Science Knowledge-base Construction (Ankan Mullick et al., 2024)

{{<citation>}}

Ankan Mullick, Akash Ghosh, G Sai Chaitanya, Samir Ghui, Tapas Nayak, Seung-Cheol Lee, Satadeep Bhattacharjee, Pawan Goyal. (2024)  
**MatSciRE: Leveraging Pointer Networks to Automate Entity and Relation Extraction for Material Science Knowledge-base Construction**  

---
Primary Category: cs.CL  
Categories: cs-CE, cs-CL, cs-IR, cs.CL  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2401.09839v1)  

---


**ABSTRACT**  
Material science literature is a rich source of factual information about various categories of entities (like materials and compositions) and various relations between these entities, such as conductivity, voltage, etc. Automatically extracting this information to generate a material science knowledge base is a challenging task. In this paper, we propose MatSciRE (Material Science Relation Extractor), a Pointer Network-based encoder-decoder framework, to jointly extract entities and relations from material science articles as a triplet ($entity1, relation, entity2$). Specifically, we target the battery materials and identify five relations to work on - conductivity, coulombic efficiency, capacity, voltage, and energy. Our proposed approach achieved a much better F1-score (0.771) than a previous attempt using ChemDataExtractor (0.716). The overall graphical framework of MatSciRE is shown in Fig 1. The material information is extracted from material science literature in the form of entity-relation triplets using MatSciRE.

{{</citation>}}


### (23/132) All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks (Kazuhiro Takemoto, 2024)

{{<citation>}}

Kazuhiro Takemoto. (2024)  
**All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09798v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) like ChatGPT face `jailbreak' challenges, where safeguards are bypassed to produce ethically harmful prompts. This study proposes a simple black-box method to effectively generate jailbreak prompts, overcoming the high complexity and computational costs associated with existing methods. The proposed technique iteratively rewrites harmful prompts into non-harmful expressions using the target LLM itself, based on the hypothesis that LLMs can directly sample expressions that bypass safeguards. Demonstrated through experiments with ChatGPT (GPT-3.5 and GPT-4) and Gemini-Pro, this method achieved an attack success rate of over 80% within an average of 5 iterations and remained effective despite model updates. The generated jailbreak prompts were naturally-worded and concise; moreover, they were difficult-to-defend. These results indicate that creating effective jailbreak prompts is simpler than previously considered, suggesting that black-box jailbreak attacks pose a more serious threat.

{{</citation>}}


### (24/132) Instant Answering in E-Commerce Buyer-Seller Messaging (Besnik Fetahu et al., 2024)

{{<citation>}}

Besnik Fetahu, Tejas Mehta, Qun Song, Nikhita Vedula, Oleg Rokhlenko, Shervin Malmasi. (2024)  
**Instant Answering in E-Commerce Buyer-Seller Messaging**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.09785v1)  

---


**ABSTRACT**  
E-commerce customers frequently seek detailed product information for purchase decisions, commonly contacting sellers directly with extended queries. This manual response requirement imposes additional costs and disrupts buyer's shopping experience with response time fluctuations ranging from hours to days. We seek to automate buyer inquiries to sellers in a leading e-commerce store using a domain-specific federated Question Answering (QA) system. The main challenge is adapting current QA systems, designed for single questions, to address detailed customer queries. We address this with a low-latency, sequence-to-sequence approach, MESSAGE-TO-QUESTION ( M2Q ). It reformulates buyer messages into succinct questions by identifying and extracting the most salient information from a message. Evaluation against baselines shows that M2Q yields relative increases of 757% in question understanding, and 1,746% in answering rate from the federated QA system. Live deployment shows that automatic answering saves sellers from manually responding to millions of messages per year, and also accelerates customer purchase decisions by eliminating the need for buyers to wait for a reply

{{</citation>}}


### (25/132) Leveraging Biases in Large Language Models: 'bias-kNN'' for Effective Few-Shot Learning (Yong Zhang et al., 2024)

{{<citation>}}

Yong Zhang, Hanzhang Li, Zhitao Li, Ning Cheng, Ming Li, Jing Xiao, Jianzong Wang. (2024)  
**Leveraging Biases in Large Language Models: 'bias-kNN'' for Effective Few-Shot Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Few-Shot, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09783v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown significant promise in various applications, including zero-shot and few-shot learning. However, their performance can be hampered by inherent biases. Instead of traditionally sought methods that aim to minimize or correct these biases, this study introduces a novel methodology named ``bias-kNN''. This approach capitalizes on the biased outputs, harnessing them as primary features for kNN and supplementing with gold labels. Our comprehensive evaluations, spanning diverse domain text classification datasets and different GPT-2 model sizes, indicate the adaptability and efficacy of the ``bias-kNN'' method. Remarkably, this approach not only outperforms conventional in-context learning in few-shot scenarios but also demonstrates robustness across a spectrum of samples, templates and verbalizers. This study, therefore, presents a unique perspective on harnessing biases, transforming them into assets for enhanced model performance.

{{</citation>}}


### (26/132) Controllable Decontextualization of Yes/No Question and Answers into Factual Statements (Lingbo Mo et al., 2024)

{{<citation>}}

Lingbo Mo, Besnik Fetahu, Oleg Rokhlenko, Shervin Malmasi. (2024)  
**Controllable Decontextualization of Yes/No Question and Answers into Factual Statements**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Transformer  
[Paper Link](http://arxiv.org/abs/2401.09775v1)  

---


**ABSTRACT**  
Yes/No or polar questions represent one of the main linguistic question categories. They consist of a main interrogative clause, for which the answer is binary (assertion or negation). Polar questions and answers (PQA) represent a valuable knowledge resource present in many community and other curated QA sources, such as forums or e-commerce applications. Using answers to polar questions alone in other contexts is not trivial. Answers are contextualized, and presume that the interrogative question clause and any shared knowledge between the asker and answerer are provided.   We address the problem of controllable rewriting of answers to polar questions into decontextualized and succinct factual statements. We propose a Transformer sequence to sequence model that utilizes soft-constraints to ensure controllable rewriting, such that the output statement is semantically equivalent to its PQA input. Evaluation on three separate PQA datasets as measured through automated and human evaluation metrics show that our proposed approach achieves the best performance when compared to existing baselines.

{{</citation>}}


### (27/132) A Comparative Study on Annotation Quality of Crowdsourcing and LLM via Label Aggregation (Jiyi Li, 2024)

{{<citation>}}

Jiyi Li. (2024)  
**A Comparative Study on Annotation Quality of Crowdsourcing and LLM via Label Aggregation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.09760v1)  

---


**ABSTRACT**  
Whether Large Language Models (LLMs) can outperform crowdsourcing on the data annotation task is attracting interest recently. Some works verified this issue with the average performance of individual crowd workers and LLM workers on some specific NLP tasks by collecting new datasets. However, on the one hand, existing datasets for the studies of annotation quality in crowdsourcing are not yet utilized in such evaluations, which potentially provide reliable evaluations from a different viewpoint. On the other hand, the quality of these aggregated labels is crucial because, when utilizing crowdsourcing, the estimated labels aggregated from multiple crowd labels to the same instances are the eventually collected labels. Therefore, in this paper, we first investigate which existing crowdsourcing datasets can be used for a comparative study and create a benchmark. We then compare the quality between individual crowd labels and LLM labels and make the evaluations on the aggregated labels. In addition, we propose a Crowd-LLM hybrid label aggregation method and verify the performance. We find that adding LLM labels from good LLMs to existing crowdsourcing datasets can enhance the quality of the aggregated labels of the datasets, which is also higher than the quality of LLM labels themselves.

{{</citation>}}


### (28/132) Curriculum Recommendations Using Transformer Base Model with InfoNCE Loss And Language Switching Method (Xiaonan Xu et al., 2024)

{{<citation>}}

Xiaonan Xu, Bin Yuan, Yongyao Mo, Tianbo Song, Shulin Li. (2024)  
**Curriculum Recommendations Using Transformer Base Model with InfoNCE Loss And Language Switching Method**  

---
Primary Category: cs.CL  
Categories: 68T50, cs-AI, cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.09699v1)  

---


**ABSTRACT**  
The Curriculum Recommendations paradigm is dedicated to fostering learning equality within the ever-evolving realms of educational technology and curriculum development. In acknowledging the inherent obstacles posed by existing methodologies, such as content conflicts and disruptions from language translation, this paradigm aims to confront and overcome these challenges. Notably, it addresses content conflicts and disruptions introduced by language translation, hindrances that can impede the creation of an all-encompassing and personalized learning experience. The paradigm's objective is to cultivate an educational environment that not only embraces diversity but also customizes learning experiences to suit the distinct needs of each learner. To overcome these challenges, our approach builds upon notable contributions in curriculum development and personalized learning, introducing three key innovations. These include the integration of Transformer Base Model to enhance computational efficiency, the implementation of InfoNCE Loss for accurate content-topic matching, and the adoption of a language switching strategy to alleviate translation-related ambiguities. Together, these innovations aim to collectively tackle inherent challenges and contribute to forging a more equitable and effective learning journey for a diverse range of learners. Competitive cross-validation scores underscore the efficacy of sentence-transformers/LaBSE, achieving 0.66314, showcasing our methodology's effectiveness in diverse linguistic nuances for content alignment prediction. Index Terms-Curriculum Recommendation, Transformer model with InfoNCE Loss, Language Switching.

{{</citation>}}


## q-bio.BM (2)



### (29/132) Machine Learning Modeling Of SiRNA Structure-Potency Relationship With Applications Against Sars-Cov-2 Spike Gene (Damilola Oshunyinka, 2024)

{{<citation>}}

Damilola Oshunyinka. (2024)  
**Machine Learning Modeling Of SiRNA Structure-Potency Relationship With Applications Against Sars-Cov-2 Spike Gene**  

---
Primary Category: q-bio.BM  
Categories: cs-LG, q-bio-BM, q-bio.BM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12232v1)  

---


**ABSTRACT**  
The pharmaceutical Research and development (R&D) process is lengthy and costly, taking nearly a decade to bring a new drug to the market. However, advancements in biotechnology, computational methods, and machine learning algorithms have the potential to revolutionize drug discovery, speeding up the process and improving patient outcomes. The COVID-19 pandemic has further accelerated and deepened the recognition of the potential of these techniques, especially in the areas of drug repurposing and efficacy predictions. Meanwhile, non-small molecule therapeutic modalities such as cell therapies, monoclonal antibodies, and RNA interference (RNAi) technology have gained importance due to their ability to target specific disease pathways and/or patient populations. In the field of RNAi, many experiments have been carried out to design and select highly efficient siRNAs. However, the established patterns for efficient siRNAs are sometimes contradictory and unable to consistently determine the most potent siRNA molecules against a target mRNA. Thus, this paper focuses on developing machine learning models based on the cheminformatics representation of the nucleotide composition (i.e. AUTGC) of siRNA to predict their potency and aid the selection of the most efficient siRNAs for further development. The PLS (Partial Least Square) and SVR (Support Vector Regression) machine learning models built in this work outperformed previously published models. These models can help in predicting siRNA potency and aid in selecting the best siRNA molecules for experimental validation and further clinical development. The study has demonstrated the potential of AI/machine learning models to help expedite siRNA-based drug discovery including the discovery of potent siRNAs against SARS-CoV-2.

{{</citation>}}


### (30/132) FREED++: Improving RL Agents for Fragment-Based Molecule Generation by Thorough Reproduction (Alexander Telepov et al., 2024)

{{<citation>}}

Alexander Telepov, Artem Tsypin, Kuzma Khrabrov, Sergey Yakukhnov, Pavel Strashnov, Petr Zhilyaev, Egor Rumiantsev, Daniel Ezhov, Manvel Avetisian, Olga Popova, Artur Kadurin. (2024)  
**FREED++: Improving RL Agents for Fragment-Based Molecule Generation by Thorough Reproduction**  

---
Primary Category: q-bio.BM  
Categories: cs-LG, q-bio-BM, q-bio.BM, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09840v1)  

---


**ABSTRACT**  
A rational design of new therapeutic drugs aims to find a molecular structure with desired biological functionality, e.g., an ability to activate or suppress a specific protein via binding to it. Molecular docking is a common technique for evaluating protein-molecule interactions. Recently, Reinforcement Learning (RL) has emerged as a promising approach to generating molecules with the docking score (DS) as a reward. In this work, we reproduce, scrutinize and improve the recent RL model for molecule generation called FREED (arXiv:2110.01219). Extensive evaluation of the proposed method reveals several limitations and challenges despite the outstanding results reported for three target proteins. Our contributions include fixing numerous implementation bugs and simplifying the model while increasing its quality, significantly extending experiments, and conducting an accurate comparison with current state-of-the-art methods for protein-conditioned molecule generation. We show that the resulting fixed model is capable of producing molecules with superior docking scores compared to alternative approaches.

{{</citation>}}


## eess.AS (3)



### (31/132) AGADIR: Towards Array-Geometry Agnostic Directional Speech Recognition (Ju Lin et al., 2024)

{{<citation>}}

Ju Lin, Niko Moritz, Yiteng Huang, Ruiming Xie, Ming Sun, Christian Fuegen, Frank Seide. (2024)  
**AGADIR: Towards Array-Geometry Agnostic Directional Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.10411v1)  

---


**ABSTRACT**  
Wearable devices like smart glasses are approaching the compute capability to seamlessly generate real-time closed captions for live conversations. We build on our recently introduced directional Automatic Speech Recognition (ASR) for smart glasses that have microphone arrays, which fuses multi-channel ASR with serialized output training, for wearer/conversation-partner disambiguation as well as suppression of cross-talk speech from non-target directions and noise.   When ASR work is part of a broader system-development process, one may be faced with changes to microphone geometries as system development progresses.   This paper aims to make multi-channel ASR insensitive to limited variations of microphone-array geometry. We show that a model trained on multiple similar geometries is largely agnostic and generalizes well to new geometries, as long as they are not too different. Furthermore, training the model this way improves accuracy for seen geometries by 15 to 28\% relative. Lastly, we refine the beamforming by a novel Non-Linearly Constrained Minimum Variance criterion.

{{</citation>}}


### (32/132) Multilingual Visual Speech Recognition with a Single Model by Learning with Discrete Visual Speech Units (Minsu Kim et al., 2024)

{{<citation>}}

Minsu Kim, Jeong Hun Yeo, Jeongsoo Choi, Se Jin Park, Yong Man Ro. (2024)  
**Multilingual Visual Speech Recognition with a Single Model by Learning with Discrete Visual Speech Units**  

---
Primary Category: eess.AS  
Categories: cs-CV, cs-SD, eess-AS, eess.AS  
Keywords: Multilingual, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.09802v1)  

---


**ABSTRACT**  
This paper explores sentence-level Multilingual Visual Speech Recognition with a single model for the first time. As the massive multilingual modeling of visual data requires huge computational costs, we propose a novel strategy, processing with visual speech units. Motivated by the recent success of the audio speech unit, the proposed visual speech unit is obtained by discretizing the visual speech features extracted from the self-supervised visual speech model. To correctly capture multilingual visual speech, we first train the self-supervised visual speech model on 5,512 hours of multilingual audio-visual data. Through analysis, we verify that the visual speech units mainly contain viseme information while suppressing non-linguistic information. By using the visual speech units as the inputs of our system, we pre-train the model to predict corresponding text outputs on massive multilingual data constructed by merging several VSR databases. As both the inputs and outputs are discrete, we can greatly improve the training efficiency compared to the standard VSR training. Specifically, the input data size is reduced to 0.016% of the original video inputs. In order to complement the insufficient visual information in speech recognition, we apply curriculum learning where the inputs of the system begin with audio-visual speech units and gradually change to visual speech units. After pre-training, the model is finetuned on continuous features. We set new state-of-the-art multilingual VSR performances by achieving comparable performances to the previous language-specific VSR models, with a single trained model.

{{</citation>}}


### (33/132) An Empirical Study on the Impact of Positional Encoding in Transformer-based Monaural Speech Enhancement (Qiquan Zhang et al., 2024)

{{<citation>}}

Qiquan Zhang, Meng Ge, Hongxu Zhu, Eliathamby Ambikairajah, Qi Song, Zhaoheng Ni, Haizhou Li. (2024)  
**An Empirical Study on the Impact of Positional Encoding in Transformer-based Monaural Speech Enhancement**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: T5, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.09686v1)  

---


**ABSTRACT**  
Transformer architecture has enabled recent progress in speech enhancement. Since Transformers are position-agostic, positional encoding is the de facto standard component used to enable Transformers to distinguish the order of elements in a sequence. However, it remains unclear how positional encoding exactly impacts speech enhancement based on Transformer architectures. In this paper, we perform a comprehensive empirical study evaluating five positional encoding methods, i.e., Sinusoidal and learned absolute position embedding (APE), T5-RPE, KERPLE, as well as the Transformer without positional encoding (No-Pos), across both causal and noncausal configurations. We conduct extensive speech enhancement experiments, involving spectral mapping and masking methods. Our findings establish that positional encoding is not quite helpful for the models in a causal configuration, which indicates that causal attention may implicitly incorporate position information. In a noncausal configuration, the models significantly benefit from the use of positional encoding. In addition, we find that among the four position embeddings, relative position embeddings outperform APEs.

{{</citation>}}


## cs.CV (30)



### (34/132) Reconstructing the Invisible: Video Frame Restoration through Siamese Masked Conditional Variational Autoencoder (Yongchen Zhou et al., 2024)

{{<citation>}}

Yongchen Zhou, Richard Jiang. (2024)  
**Reconstructing the Invisible: Video Frame Restoration through Siamese Masked Conditional Variational Autoencoder**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.10402v1)  

---


**ABSTRACT**  
In the domain of computer vision, the restoration of missing information in video frames is a critical challenge, particularly in applications such as autonomous driving and surveillance systems. This paper introduces the Siamese Masked Conditional Variational Autoencoder (SiamMCVAE), leveraging a siamese architecture with twin encoders based on vision transformers. This innovative design enhances the model's ability to comprehend lost content by capturing intrinsic similarities between paired frames. SiamMCVAE proficiently reconstructs missing elements in masked frames, effectively addressing issues arising from camera malfunctions through variational inferences. Experimental results robustly demonstrate the model's effectiveness in restoring missing information, thus enhancing the resilience of computer vision systems. The incorporation of Siamese Vision Transformer (SiamViT) encoders in SiamMCVAE exemplifies promising potential for addressing real-world challenges in computer vision, reinforcing the adaptability of autonomous systems in dynamic environments.

{{</citation>}}


### (35/132) Analyzing and Mitigating Bias for Vulnerable Classes: Towards Balanced Representation in Dataset (Dewant Katare et al., 2024)

{{<citation>}}

Dewant Katare, David Solans Noguero, Souneil Park, Nicolas Kourtellis, Marijn Janssen, Aaron Yi Ding. (2024)  
**Analyzing and Mitigating Bias for Vulnerable Classes: Towards Balanced Representation in Dataset**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Bias, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.10397v1)  

---


**ABSTRACT**  
The accuracy and fairness of perception systems in autonomous driving are crucial, particularly for vulnerable road users. Mainstream research has looked into improving the performance metrics for classification accuracy. However, the hidden traits of bias inheritance in the AI models, class imbalances and disparities in the datasets are often overlooked. In this context, our study examines the class imbalances for vulnerable road users by focusing on class distribution analysis, performance evaluation, and bias impact assessment. We identify the concern of imbalances in class representation, leading to potential biases in detection accuracy. Utilizing popular CNN models and Vision Transformers (ViTs) with the nuScenes dataset, our performance evaluation reveals detection disparities for underrepresented classes. We propose a methodology for model optimization and bias mitigation, which includes data augmentation, resampling, and metric-specific learning. Using the proposed mitigation approaches, we see improvement in IoU(%) and NDS(%) metrics from 71.3 to 75.6 and 80.6 to 83.7 respectively, for the CNN model. Similarly, for ViT, we observe improvement in IoU and NDS metrics from 74.9 to 79.2 and 83.8 to 87.1 respectively. This research contributes to developing more reliable models and datasets, enhancing inclusiveness for minority classes.

{{</citation>}}


### (36/132) Agricultural Object Detection with You Look Only Once (YOLO) Algorithm: A Bibliometric and Systematic Literature Review (Chetan M Badgujar et al., 2024)

{{<citation>}}

Chetan M Badgujar, Alwin Poulose, Hao Gan. (2024)  
**Agricultural Object Detection with You Look Only Once (YOLO) Algorithm: A Bibliometric and Systematic Literature Review**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.10379v1)  

---


**ABSTRACT**  
Vision is a major component in several digital technologies and tools used in agriculture. The object detector, You Look Only Once (YOLO), has gained popularity in agriculture in a relatively short span due to its state-of-the-art performance. YOLO offers real-time detection with good accuracy and is implemented in various agricultural tasks, including monitoring, surveillance, sensing, automation, and robotics. The research and application of YOLO in agriculture are accelerating rapidly but are fragmented and multidisciplinary. Moreover, the performance characteristics (i.e., accuracy, speed, computation) of the object detector influence the rate of technology implementation and adoption in agriculture. Thus, the study aims to collect extensive literature to document and critically evaluate the advances and application of YOLO for agricultural object recognition. First, we conducted a bibliometric review of 257 articles to understand the scholarly landscape of YOLO in agricultural domain. Secondly, we conducted a systematic review of 30 articles to identify current knowledge, gaps, and modifications in YOLO for specific agricultural tasks. The study critically assesses and summarizes the information on YOLO's end-to-end learning approach, including data acquisition, processing, network modification, integration, and deployment. We also discussed task-specific YOLO algorithm modification and integration to meet the agricultural object or environment-specific challenges. In general, YOLO-integrated digital tools and technologies show the potential for real-time, automated monitoring, surveillance, and object handling to reduce labor, production cost, and environmental impact while maximizing resource efficiency. The study provides detailed documentation and significantly advances the existing knowledge on applying YOLO in agriculture, which can greatly benefit the scientific community.

{{</citation>}}


### (37/132) Towards Language-Driven Video Inpainting via Multimodal Large Language Models (Jianzong Wu et al., 2024)

{{<citation>}}

Jianzong Wu, Xiangtai Li, Chenyang Si, Shangchen Zhou, Jingkang Yang, Jiangning Zhang, Yining Li, Kai Chen, Yunhai Tong, Ziwei Liu, Chen Change Loy. (2024)  
**Towards Language-Driven Video Inpainting via Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10226v1)  

---


**ABSTRACT**  
We introduce a new task -- language-driven video inpainting, which uses natural language instructions to guide the inpainting process. This approach overcomes the limitations of traditional video inpainting methods that depend on manually labeled binary masks, a process often tedious and labor-intensive. We present the Remove Objects from Videos by Instructions (ROVI) dataset, containing 5,650 videos and 9,091 inpainting results, to support training and evaluation for this task. We also propose a novel diffusion-based language-driven video inpainting framework, the first end-to-end baseline for this task, integrating Multimodal Large Language Models to understand and execute complex language-based inpainting requests effectively. Our comprehensive results showcase the dataset's versatility and the model's effectiveness in various language-instructed inpainting scenarios. We will make datasets, code, and models publicly available.

{{</citation>}}


### (38/132) GPAvatar: Generalizable and Precise Head Avatar from Image(s) (Xuangeng Chu et al., 2024)

{{<citation>}}

Xuangeng Chu, Yu Li, Ailing Zeng, Tianyu Yang, Lijian Lin, Yunfei Liu, Tatsuya Harada. (2024)  
**GPAvatar: Generalizable and Precise Head Avatar from Image(s)**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.10215v1)  

---


**ABSTRACT**  
Head avatar reconstruction, crucial for applications in virtual reality, online meetings, gaming, and film industries, has garnered substantial attention within the computer vision community. The fundamental objective of this field is to faithfully recreate the head avatar and precisely control expressions and postures. Existing methods, categorized into 2D-based warping, mesh-based, and neural rendering approaches, present challenges in maintaining multi-view consistency, incorporating non-facial information, and generalizing to new identities. In this paper, we propose a framework named GPAvatar that reconstructs 3D head avatars from one or several images in a single forward pass. The key idea of this work is to introduce a dynamic point-based expression field driven by a point cloud to precisely and effectively capture expressions. Furthermore, we use a Multi Tri-planes Attention (MTA) fusion module in the tri-planes canonical field to leverage information from multiple input images. The proposed method achieves faithful identity reconstruction, precise expression control, and multi-view consistency, demonstrating promising results for free-viewpoint rendering and novel view synthesis.

{{</citation>}}


### (39/132) Neural Echos: Depthwise Convolutional Filters Replicate Biological Receptive Fields (Zahra Babaiee et al., 2024)

{{<citation>}}

Zahra Babaiee, Peyman M. Kiasari, Daniela Rus, Radu Grosu. (2024)  
**Neural Echos: Depthwise Convolutional Filters Replicate Biological Receptive Fields**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-NE, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.10178v1)  

---


**ABSTRACT**  
In this study, we present evidence suggesting that depthwise convolutional kernels are effectively replicating the structural intricacies of the biological receptive fields observed in the mammalian retina. We provide analytics of trained kernels from various state-of-the-art models substantiating this evidence. Inspired by this intriguing discovery, we propose an initialization scheme that draws inspiration from the biological receptive fields. Experimental analysis of the ImageNet dataset with multiple CNN architectures featuring depthwise convolutions reveals a marked enhancement in the accuracy of the learned model when initialized with biologically derived weights. This underlies the potential for biologically inspired computational models to further our understanding of vision processing systems and to improve the efficacy of convolutional networks.

{{</citation>}}


### (40/132) VMamba: Visual State Space Model (Yue Liu et al., 2024)

{{<citation>}}

Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Yunfan Liu. (2024)  
**VMamba: Visual State Space Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.10166v1)  

---


**ABSTRACT**  
Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) stand as the two most popular foundation models for visual representation learning. While CNNs exhibit remarkable scalability with linear complexity w.r.t. image resolution, ViTs surpass them in fitting capabilities despite contending with quadratic complexity. A closer inspection reveals that ViTs achieve superior visual modeling performance through the incorporation of global receptive fields and dynamic weights. This observation motivates us to propose a novel architecture that inherits these components while enhancing computational efficiency. To this end, we draw inspiration from the recently introduced state space model and propose the Visual State Space Model (VMamba), which achieves linear complexity without sacrificing global receptive fields. To address the encountered direction-sensitive issue, we introduce the Cross-Scan Module (CSM) to traverse the spatial domain and convert any non-causal visual image into order patch sequences. Extensive experimental results substantiate that VMamba not only demonstrates promising capabilities across various visual perception tasks, but also exhibits more pronounced advantages over established benchmarks as the image resolution increases. Source code has been available at https://github.com/MzeroMiko/VMamba.

{{</citation>}}


### (41/132) Motion-Zero: Zero-Shot Moving Object Control Framework for Diffusion-Based Video Generation (Changgu Chen et al., 2024)

{{<citation>}}

Changgu Chen, Junwei Shu, Lianggangxu Chen, Gaoqi He, Changbo Wang, Yang Li. (2024)  
**Motion-Zero: Zero-Shot Moving Object Control Framework for Diffusion-Based Video Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.10150v3)  

---


**ABSTRACT**  
Recent large-scale pre-trained diffusion models have demonstrated a powerful generative ability to produce high-quality videos from detailed text descriptions. However, exerting control over the motion of objects in videos generated by any video diffusion model is a challenging problem. In this paper, we propose a novel zero-shot moving object trajectory control framework, Motion-Zero, to enable a bounding-box-trajectories-controlled text-to-video diffusion model. To this end, an initial noise prior module is designed to provide a position-based prior to improve the stability of the appearance of the moving object and the accuracy of position. In addition, based on the attention map of the U-net, spatial constraints are directly applied to the denoising process of diffusion models, which further ensures the positional and spatial consistency of moving objects during the inference. Furthermore, temporal consistency is guaranteed with a proposed shift temporal attention mechanism. Our method can be flexibly applied to various state-of-the-art video diffusion models without any training process. Extensive experiments demonstrate our proposed method can control the motion trajectories of objects and generate high-quality videos.

{{</citation>}}


### (42/132) Explicitly Disentangled Representations in Object-Centric Learning (Riccardo Majellaro et al., 2024)

{{<citation>}}

Riccardo Majellaro, Jonathan Collu, Aske Plaat, Thomas M. Moerland. (2024)  
**Explicitly Disentangled Representations in Object-Centric Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.10148v1)  

---


**ABSTRACT**  
Extracting structured representations from raw visual data is an important and long-standing challenge in machine learning. Recently, techniques for unsupervised learning of object-centric representations have raised growing interest. In this context, enhancing the robustness of the latent features can improve the efficiency and effectiveness of the training of downstream tasks. A promising step in this direction is to disentangle the factors that cause variation in the data. Previously, Invariant Slot Attention disentangled position, scale, and orientation from the remaining features. Extending this approach, we focus on separating the shape and texture components. In particular, we propose a novel architecture that biases object-centric models toward disentangling shape and texture components into two non-overlapping subsets of the latent space dimensions. These subsets are known a priori, hence before the training process. Experiments on a range of object-centric benchmarks reveal that our approach achieves the desired disentanglement while also numerically improving baseline performance in most cases. In addition, we show that our method can generate novel textures for a specific object or transfer textures between objects with distinct shapes.

{{</citation>}}


### (43/132) Exposing Lip-syncing Deepfakes from Mouth Inconsistencies (Soumyya Kanti Datta et al., 2024)

{{<citation>}}

Soumyya Kanti Datta, Shan Jia, Siwei Lyu. (2024)  
**Exposing Lip-syncing Deepfakes from Mouth Inconsistencies**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.10113v1)  

---


**ABSTRACT**  
A lip-syncing deepfake is a digitally manipulated video in which a person's lip movements are created convincingly using AI models to match altered or entirely new audio. Lip-syncing deepfakes are a dangerous type of deepfakes as the artifacts are limited to the lip region and more difficult to discern. In this paper, we describe a novel approach, LIP-syncing detection based on mouth INConsistency (LIPINC), for lip-syncing deepfake detection by identifying temporal inconsistencies in the mouth region. These inconsistencies are seen in the adjacent frames and throughout the video. Our model can successfully capture these irregularities and outperforms the state-of-the-art methods on several benchmark deepfake datasets.

{{</citation>}}


### (44/132) DiffusionGPT: LLM-Driven Text-to-Image Generation System (Jie Qin et al., 2024)

{{<citation>}}

Jie Qin, Jie Wu, Weifeng Chen, Yuxi Ren, Huixia Li, Hefeng Wu, Xuefeng Xiao, Rui Wang, Shilei Wen. (2024)  
**DiffusionGPT: LLM-Driven Text-to-Image Generation System**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.10061v1)  

---


**ABSTRACT**  
Diffusion models have opened up new avenues for the field of image generation, resulting in the proliferation of high-quality models shared on open-source platforms. However, a major challenge persists in current text-to-image systems are often unable to handle diverse inputs, or are limited to single model results. Current unified attempts often fall into two orthogonal aspects: i) parse Diverse Prompts in input stage; ii) activate expert model to output. To combine the best of both worlds, we propose DiffusionGPT, which leverages Large Language Models (LLM) to offer a unified generation system capable of seamlessly accommodating various types of prompts and integrating domain-expert models. DiffusionGPT constructs domain-specific Trees for various generative models based on prior knowledge. When provided with an input, the LLM parses the prompt and employs the Trees-of-Thought to guide the selection of an appropriate model, thereby relaxing input constraints and ensuring exceptional performance across diverse domains. Moreover, we introduce Advantage Databases, where the Tree-of-Thought is enriched with human feedback, aligning the model selection process with human preferences. Through extensive experiments and comparisons, we demonstrate the effectiveness of DiffusionGPT, showcasing its potential for pushing the boundaries of image synthesis in diverse domains.

{{</citation>}}


### (45/132) GPT4Ego: Unleashing the Potential of Pre-trained Models for Zero-Shot Egocentric Action Recognition (Guangzhao Dai et al., 2024)

{{<citation>}}

Guangzhao Dai, Xiangbo Shu, Wenhao Wu. (2024)  
**GPT4Ego: Unleashing the Potential of Pre-trained Models for Zero-Shot Egocentric Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.10039v1)  

---


**ABSTRACT**  
Vision-Language Models (VLMs), pre-trained on large-scale datasets, have shown impressive performance in various visual recognition tasks. This advancement paves the way for notable performance in Zero-Shot Egocentric Action Recognition (ZS-EAR). Typically, VLMs handle ZS-EAR as a global video-text matching task, which often leads to suboptimal alignment of vision and linguistic knowledge. We propose a refined approach for ZS-EAR using VLMs, emphasizing fine-grained concept-description alignment that capitalizes on the rich semantic and contextual details in egocentric videos. In this paper, we introduce GPT4Ego, a straightforward yet remarkably potent VLM framework for ZS-EAR, designed to enhance the fine-grained alignment of concept and description between vision and language. Extensive experiments demonstrate GPT4Ego significantly outperforms existing VLMs on three large-scale egocentric video benchmarks, i.e., EPIC-KITCHENS-100 (33.2%, +9.4%), EGTEA (39.6%, +5.5%), and CharadesEgo (31.5%, +2.6%).

{{</citation>}}


### (46/132) Depth Over RGB: Automatic Evaluation of Open Surgery Skills Using Depth Camera (Ido Zuckerman et al., 2024)

{{<citation>}}

Ido Zuckerman, Nicole Werner, Jonathan Kouchly, Emma Huston, Shannon DiMarco, Paul DiMusto, Shlomi Laufer. (2024)  
**Depth Over RGB: Automatic Evaluation of Open Surgery Skills Using Depth Camera**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Azure  
[Paper Link](http://arxiv.org/abs/2401.10037v1)  

---


**ABSTRACT**  
Purpose: In this paper, we present a novel approach to the automatic evaluation of open surgery skills using depth cameras. This work is intended to show that depth cameras achieve similar results to RGB cameras, which is the common method in the automatic evaluation of open surgery skills. Moreover, depth cameras offer advantages such as robustness to lighting variations, camera positioning, simplified data compression, and enhanced privacy, making them a promising alternative to RGB cameras.   Methods: Experts and novice surgeons completed two simulators of open suturing. We focused on hand and tool detection, and action segmentation in suturing procedures. YOLOv8 was used for tool detection in RGB and depth videos. Furthermore, UVAST and MSTCN++ were used for action segmentation. Our study includes the collection and annotation of a dataset recorded with Azure Kinect.   Results: We demonstrated that using depth cameras in object detection and action segmentation achieves comparable results to RGB cameras. Furthermore, we analyzed 3D hand path length, revealing significant differences between experts and novice surgeons, emphasizing the potential of depth cameras in capturing surgical skills. We also investigated the influence of camera angles on measurement accuracy, highlighting the advantages of 3D cameras in providing a more accurate representation of hand movements.   Conclusion: Our research contributes to advancing the field of surgical skill assessment by leveraging depth cameras for more reliable and privacy evaluations. The findings suggest that depth cameras can be valuable in assessing surgical skills and provide a foundation for future research in this area.

{{</citation>}}


### (47/132) CPCL: Cross-Modal Prototypical Contrastive Learning for Weakly Supervised Text-based Person Re-Identification (Yanwei Zheng et al., 2024)

{{<citation>}}

Yanwei Zheng, Xinpeng Zhao, Chuanlin Lan, Xiaowei Zhang, Bowen Huang, Jibin Yang, Dongxiao Yu. (2024)  
**CPCL: Cross-Modal Prototypical Contrastive Learning for Weakly Supervised Text-based Person Re-Identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.10011v1)  

---


**ABSTRACT**  
Weakly supervised text-based person re-identification (TPRe-ID) seeks to retrieve images of a target person using textual descriptions, without relying on identity annotations and is more challenging and practical. The primary challenge is the intra-class differences, encompassing intra-modal feature variations and cross-modal semantic gaps. Prior works have focused on instance-level samples and ignored prototypical features of each person which are intrinsic and invariant. Toward this, we propose a Cross-Modal Prototypical Contrastive Learning (CPCL) method. In practice, the CPCL introduces the CLIP model to weakly supervised TPRe-ID for the first time, mapping visual and textual instances into a shared latent space. Subsequently, the proposed Prototypical Multi-modal Memory (PMM) module captures associations between heterogeneous modalities of image-text pairs belonging to the same person through the Hybrid Cross-modal Matching (HCM) module in a many-to-many mapping fashion. Moreover, the Outlier Pseudo Label Mining (OPLM) module further distinguishes valuable outlier samples from each modality, enhancing the creation of more reliable clusters by mining implicit relationships between image-text pairs. Experimental results demonstrate that our proposed CPCL attains state-of-the-art performance on all three public datasets, with a significant improvement of 11.58%, 8.77% and 5.25% in Rank@1 accuracy on CUHK-PEDES, ICFG-PEDES and RSTPReid datasets, respectively. The code is available at https://github.com/codeGallery24/CPCL.

{{</citation>}}


### (48/132) Advancing Large Multi-modal Models with Explicit Chain-of-Reasoning and Visual Question Generation (Kohei Uehara et al., 2024)

{{<citation>}}

Kohei Uehara, Nabarun Goswami, Hanqin Wang, Toshiaki Baba, Kohtaro Tanaka, Tomohiro Hashimoto, Kai Wang, Rei Ito, Takagi Naoya, Ryo Umagami, Yingyi Wen, Tanachai Anakewat, Tatsuya Harada. (2024)  
**Advancing Large Multi-modal Models with Explicit Chain-of-Reasoning and Visual Question Generation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model, Question Generation, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.10005v1)  

---


**ABSTRACT**  
The increasing demand for intelligent systems capable of interpreting and reasoning about visual content requires the development of Large Multi-Modal Models (LMMs) that are not only accurate but also have explicit reasoning capabilities. This paper presents a novel approach to imbue an LMM with the ability to conduct explicit reasoning based on visual content and textual instructions. We introduce a system that can ask a question to acquire necessary knowledge, thereby enhancing the robustness and explicability of the reasoning process. Our method comprises the development of a novel dataset generated by a Large Language Model (LLM), designed to promote chain-of-thought reasoning combined with a question-asking mechanism. We designed an LMM, which has high capabilities on region awareness to address the intricate requirements of image-text alignment. The model undergoes a three-stage training phase, starting with large-scale image-text alignment using a large-scale datasets, followed by instruction tuning, and fine-tuning with a focus on chain-of-thought reasoning. The results demonstrate a stride toward a more robust, accurate, and interpretable LMM, capable of reasoning explicitly and seeking information proactively when confronted with ambiguous visual input.

{{</citation>}}


### (49/132) MAMBA: Multi-level Aggregation via Memory Bank for Video Object Detection (Guanxiong Sun et al., 2024)

{{<citation>}}

Guanxiong Sun, Yang Hua, Guosheng Hu, Neil Robertson. (2024)  
**MAMBA: Multi-level Aggregation via Memory Bank for Video Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Object Detection  
[Paper Link](http://arxiv.org/abs/2401.09923v1)  

---


**ABSTRACT**  
State-of-the-art video object detection methods maintain a memory structure, either a sliding window or a memory queue, to enhance the current frame using attention mechanisms. However, we argue that these memory structures are not efficient or sufficient because of two implied operations: (1) concatenating all features in memory for enhancement, leading to a heavy computational cost; (2) frame-wise memory updating, preventing the memory from capturing more temporal information. In this paper, we propose a multi-level aggregation architecture via memory bank called MAMBA. Specifically, our memory bank employs two novel operations to eliminate the disadvantages of existing methods: (1) light-weight key-set construction which can significantly reduce the computational cost; (2) fine-grained feature-wise updating strategy which enables our method to utilize knowledge from the whole video. To better enhance features from complementary levels, i.e., feature maps and proposals, we further propose a generalized enhancement operation (GEO) to aggregate multi-level features in a unified manner. We conduct extensive evaluations on the challenging ImageNetVID dataset. Compared with existing state-of-the-art methods, our method achieves superior performance in terms of both speed and accuracy. More remarkably, MAMBA achieves mAP of 83.7/84.6% at 12.6/9.1 FPS with ResNet-101. Code is available at https://github.com/guanxiongsun/video_feature_enhancement.

{{</citation>}}


### (50/132) BlenDA: Domain Adaptive Object Detection through diffusion-based blending (Tzuhsuan Huang et al., 2024)

{{<citation>}}

Tzuhsuan Huang, Chen-Che Huang, Chung-Hao Ku, Jun-Cheng Chen. (2024)  
**BlenDA: Domain Adaptive Object Detection through diffusion-based blending**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2401.09921v1)  

---


**ABSTRACT**  
Unsupervised domain adaptation (UDA) aims to transfer a model learned using labeled data from the source domain to unlabeled data in the target domain. To address the large domain gap issue between the source and target domains, we propose a novel regularization method for domain adaptive object detection, BlenDA, by generating the pseudo samples of the intermediate domains and their corresponding soft domain labels for adaptation training. The intermediate samples are generated by dynamically blending the source images with their corresponding translated images using an off-the-shelf pre-trained text-to-image diffusion model which takes the text label of the target domain as input and has demonstrated superior image-to-image translation quality. Based on experimental results from two adaptation benchmarks, our proposed approach can significantly enhance the performance of the state-of-the-art domain adaptive object detector, Adversarial Query Transformer (AQT). Particularly, in the Cityscapes to Foggy Cityscapes adaptation, we achieve an impressive 53.4% mAP on the Foggy Cityscapes dataset, surpassing the previous state-of-the-art by 1.5%. It is worth noting that our proposed method is also applicable to various paradigms of domain adaptive object detection. The code is available at:https://github.com/aiiu-lab/BlenDA

{{</citation>}}


### (51/132) XAI-Enhanced Semantic Segmentation Models for Visual Quality Inspection (Tobias Clement et al., 2024)

{{<citation>}}

Tobias Clement, Truong Thanh Hung Nguyen, Mohamed Abdelaal, Hung Cao. (2024)  
**XAI-Enhanced Semantic Segmentation Models for Visual Quality Inspection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Augmentation, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.09900v1)  

---


**ABSTRACT**  
Visual quality inspection systems, crucial in sectors like manufacturing and logistics, employ computer vision and machine learning for precise, rapid defect detection. However, their unexplained nature can hinder trust, error identification, and system improvement. This paper presents a framework to bolster visual quality inspection by using CAM-based explanations to refine semantic segmentation models. Our approach consists of 1) Model Training, 2) XAI-based Model Explanation, 3) XAI Evaluation, and 4) Annotation Augmentation for Model Enhancement, informed by explanations and expert insights. Evaluations show XAI-enhanced models surpass original DeepLabv3-ResNet101 models, especially in intricate object segmentation.

{{</citation>}}


### (52/132) Skeleton-Guided Instance Separation for Fine-Grained Segmentation in Microscopy (Jun Wang et al., 2024)

{{<citation>}}

Jun Wang, Chengfeng Zhou, Zhaoyan Ming, Lina Wei, Xudong Jiang, Dahong Qian. (2024)  
**Skeleton-Guided Instance Separation for Fine-Grained Segmentation in Microscopy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.09895v2)  

---


**ABSTRACT**  
One of the fundamental challenges in microscopy (MS) image analysis is instance segmentation (IS), particularly when segmenting cluster regions where multiple objects of varying sizes and shapes may be connected or even overlapped in arbitrary orientations. Existing IS methods usually fail in handling such scenarios, as they rely on coarse instance representations such as keypoints and horizontal bounding boxes (h-bboxes). In this paper, we propose a novel one-stage framework named A2B-IS to address this challenge and enhance the accuracy of IS in MS images. Our approach represents each instance with a pixel-level mask map and a rotated bounding box (r-bbox). Unlike two-stage methods that use box proposals for segmentations, our method decouples mask and box predictions, enabling simultaneous processing to streamline the model pipeline. Additionally, we introduce a Gaussian skeleton map to aid the IS task in two key ways: (1) It guides anchor placement, reducing computational costs while improving the model's capacity to learn RoI-aware features by filtering out noise from background regions. (2) It ensures accurate isolation of densely packed instances by rectifying erroneous box predictions near instance boundaries. To further enhance the performance, we integrate two modules into the framework: (1) An Atrous Attention Block (A2B) designed to extract high-resolution feature maps with fine-grained multiscale information, and (2) A Semi-Supervised Learning (SSL) strategy that leverages both labeled and unlabeled images for model training. Our method has been thoroughly validated on two large-scale MS datasets, demonstrating its superiority over most state-of-the-art approaches.

{{</citation>}}


### (53/132) Question-Answer Cross Language Image Matching for Weakly Supervised Semantic Segmentation (Songhe Deng et al., 2024)

{{<citation>}}

Songhe Deng, Wei Zhuo, Jinheng Xie, Linlin Shen. (2024)  
**Question-Answer Cross Language Image Matching for Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.09883v1)  

---


**ABSTRACT**  
Class Activation Map (CAM) has emerged as a popular tool for weakly supervised semantic segmentation (WSSS), allowing the localization of object regions in an image using only image-level labels. However, existing CAM methods suffer from under-activation of target object regions and false-activation of background regions due to the fact that a lack of detailed supervision can hinder the model's ability to understand the image as a whole. In this paper, we propose a novel Question-Answer Cross-Language-Image Matching framework for WSSS (QA-CLIMS), leveraging the vision-language foundation model to maximize the text-based understanding of images and guide the generation of activation maps. First, a series of carefully designed questions are posed to the VQA (Visual Question Answering) model with Question-Answer Prompt Engineering (QAPE) to generate a corpus of both foreground target objects and backgrounds that are adaptive to query images. We then employ contrastive learning in a Region Image Text Contrastive (RITC) network to compare the obtained foreground and background regions with the generated corpus. Our approach exploits the rich textual information from the open vocabulary as additional supervision, enabling the model to generate high-quality CAMs with a more complete object region and reduce false-activation of background regions. We conduct extensive analysis to validate the proposed method and show that our approach performs state-of-the-art on both PASCAL VOC 2012 and MS COCO datasets. Code is available at: https://github.com/CVI-SZU/QA-CLIMS

{{</citation>}}


### (54/132) Boosting Few-Shot Segmentation via Instance-Aware Data Augmentation and Local Consensus Guided Cross Attention (Li Guo et al., 2024)

{{<citation>}}

Li Guo, Haoming Liu, Yuxuan Xia, Chengyu Zhang, Xiaochen Lu. (2024)  
**Boosting Few-Shot Segmentation via Instance-Aware Data Augmentation and Local Consensus Guided Cross Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Augmentation, Few-Shot  
[Paper Link](http://arxiv.org/abs/2401.09866v1)  

---


**ABSTRACT**  
Few-shot segmentation aims to train a segmentation model that can fast adapt to a novel task for which only a few annotated images are provided. Most recent models have adopted a prototype-based paradigm for few-shot inference. These approaches may have limited generalization capacity beyond the standard 1- or 5-shot settings. In this paper, we closely examine and reevaluate the fine-tuning based learning scheme that fine-tunes the classification layer of a deep segmentation network pre-trained on diverse base classes. To improve the generalizability of the classification layer optimized with sparsely annotated samples, we introduce an instance-aware data augmentation (IDA) strategy that augments the support images based on the relative sizes of the target objects. The proposed IDA effectively increases the support set's diversity and promotes the distribution consistency between support and query images. On the other hand, the large visual difference between query and support images may hinder knowledge transfer and cripple the segmentation performance. To cope with this challenge, we introduce the local consensus guided cross attention (LCCA) to align the query feature with support features based on their dense correlation, further improving the model's generalizability to the query image. The significant performance improvements on the standard few-shot segmentation benchmarks PASCAL-$5^i$ and COCO-$20^i$ verify the efficacy of our proposed method.

{{</citation>}}


### (55/132) Temporal Insight Enhancement: Mitigating Temporal Hallucination in Multimodal Large Language Models (Li Sun et al., 2024)

{{<citation>}}

Li Sun, Liuan Wang, Jun Sun, Takayuki Okatani. (2024)  
**Temporal Insight Enhancement: Mitigating Temporal Hallucination in Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09861v1)  

---


**ABSTRACT**  
Recent advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced the comprehension of multimedia content, bringing together diverse modalities such as text, images, and videos. However, a critical challenge faced by these models, especially when processing video inputs, is the occurrence of hallucinations - erroneous perceptions or interpretations, particularly at the event level. This study introduces an innovative method to address event-level hallucinations in MLLMs, focusing on specific temporal understanding in video content. Our approach leverages a novel framework that extracts and utilizes event-specific information from both the event query and the provided video to refine MLLMs' response. We propose a unique mechanism that decomposes on-demand event queries into iconic actions. Subsequently, we employ models like CLIP and BLIP2 to predict specific timestamps for event occurrences. Our evaluation, conducted using the Charades-STA dataset, demonstrates a significant reduction in temporal hallucinations and an improvement in the quality of event-related responses. This research not only provides a new perspective in addressing a critical limitation of MLLMs but also contributes a quantitatively measurable method for evaluating MLLMs in the context of temporal-related questions.

{{</citation>}}


### (56/132) Enhancing the Fairness and Performance of Edge Cameras with Explainable AI (Truong Thanh Hung Nguyen et al., 2024)

{{<citation>}}

Truong Thanh Hung Nguyen, Vo Thanh Khang Nguyen, Quoc Hung Cao, Van Binh Truong, Quoc Khanh Nguyen, Hung Cao. (2024)  
**Enhancing the Fairness and Performance of Edge Cameras with Explainable AI**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09852v1)  

---


**ABSTRACT**  
The rising use of Artificial Intelligence (AI) in human detection on Edge camera systems has led to accurate but complex models, challenging to interpret and debug. Our research presents a diagnostic method using Explainable AI (XAI) for model debugging, with expert-driven problem identification and solution creation. Validated on the Bytetrack model in a real-world office Edge network, we found the training dataset as the main bias source and suggested model augmentation as a solution. Our approach helps identify model biases, essential for achieving fair and trustworthy models.

{{</citation>}}


### (57/132) Exploring Latent Cross-Channel Embedding for Accurate 3D Human Pose Reconstruction in a Diffusion Framework (Junkun Jiang et al., 2024)

{{<citation>}}

Junkun Jiang, Jie Chen. (2024)  
**Exploring Latent Cross-Channel Embedding for Accurate 3D Human Pose Reconstruction in a Diffusion Framework**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.09836v1)  

---


**ABSTRACT**  
Monocular 3D human pose estimation poses significant challenges due to the inherent depth ambiguities that arise during the reprojection process from 2D to 3D. Conventional approaches that rely on estimating an over-fit projection matrix struggle to effectively address these challenges and often result in noisy outputs. Recent advancements in diffusion models have shown promise in incorporating structural priors to address reprojection ambiguities. However, there is still ample room for improvement as these methods often overlook the exploration of correlation between the 2D and 3D joint-level features. In this study, we propose a novel cross-channel embedding framework that aims to fully explore the correlation between joint-level features of 3D coordinates and their 2D projections. In addition, we introduce a context guidance mechanism to facilitate the propagation of joint graph attention across latent channels during the iterative diffusion process. To evaluate the effectiveness of our proposed method, we conduct experiments on two benchmark datasets, namely Human3.6M and MPI-INF-3DHP. Our results demonstrate a significant improvement in terms of reconstruction accuracy compared to state-of-the-art methods. The code for our method will be made available online for further reference.

{{</citation>}}


### (58/132) Boosting Few-Shot Semantic Segmentation Via Segment Anything Model (Chen-Bin Feng et al., 2024)

{{<citation>}}

Chen-Bin Feng, Qi Lai, Kangdao Liu, Houcheng Su, Chi-Man Vong. (2024)  
**Boosting Few-Shot Semantic Segmentation Via Segment Anything Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.09826v2)  

---


**ABSTRACT**  
In semantic segmentation, accurate prediction masks are crucial for downstream tasks such as medical image analysis and image editing. Due to the lack of annotated data, few-shot semantic segmentation (FSS) performs poorly in predicting masks with precise contours. Recently, we have noticed that the large foundation model segment anything model (SAM) performs well in processing detailed features. Inspired by SAM, we propose FSS-SAM to boost FSS methods by addressing the issue of inaccurate contour. The FSS-SAM is training-free. It works as a post-processing tool for any FSS methods and can improve the accuracy of predicted masks. Specifically, we use predicted masks from FSS methods to generate prompts and then use SAM to predict new masks. To avoid predicting wrong masks with SAM, we propose a prediction result selection (PRS) algorithm. The algorithm can remarkably decrease wrong predictions. Experiment results on public datasets show that our method is superior to base FSS methods in both quantitative and qualitative aspects.

{{</citation>}}


### (59/132) Enhancing Small Object Encoding in Deep Neural Networks: Introducing Fast&Focused-Net with Volume-wise Dot Product Layer (Ali Tofik et al., 2024)

{{<citation>}}

Ali Tofik, Roy Partha Pratim. (2024)  
**Enhancing Small Object Encoding in Deep Neural Networks: Introducing Fast&Focused-Net with Volume-wise Dot Product Layer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.09823v1)  

---


**ABSTRACT**  
In this paper, we introduce Fast&Focused-Net, a novel deep neural network architecture tailored for efficiently encoding small objects into fixed-length feature vectors. Contrary to conventional Convolutional Neural Networks (CNNs), Fast&Focused-Net employs a series of our newly proposed layer, the Volume-wise Dot Product (VDP) layer, designed to address several inherent limitations of CNNs. Specifically, CNNs often exhibit a smaller effective receptive field than their theoretical counterparts, limiting their vision span. Additionally, the initial layers in CNNs produce low-dimensional feature vectors, presenting a bottleneck for subsequent learning. Lastly, the computational overhead of CNNs, particularly in capturing diverse image regions by parameter sharing, is significantly high. The VDP layer, at the heart of Fast&Focused-Net, aims to remedy these issues by efficiently covering the entire image patch information with reduced computational demand. Experimental results demonstrate the prowess of Fast&Focused-Net in a variety of applications. For small object classification tasks, our network outperformed state-of-the-art methods on datasets such as CIFAR-10, CIFAR-100, STL-10, SVHN-Cropped, and Fashion-MNIST. In the context of larger image classification, when combined with a transformer encoder (ViT), Fast&Focused-Net produced competitive results for OpenImages V6, ImageNet-1K, and Places365 datasets. Moreover, the same combination showcased unparalleled performance in text recognition tasks across SVT, IC15, SVTP, and HOST datasets. This paper presents the architecture, the underlying motivation, and extensive empirical evidence suggesting that Fast&Focused-Net is a promising direction for efficient and focused deep learning.

{{</citation>}}


### (60/132) SlideAVSR: A Dataset of Paper Explanation Videos for Audio-Visual Speech Recognition (Hao Wang et al., 2024)

{{<citation>}}

Hao Wang, Shuhei Kurita, Shuichiro Shimizu, Daisuke Kawahara. (2024)  
**SlideAVSR: A Dataset of Paper Explanation Videos for Audio-Visual Speech Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.09759v1)  

---


**ABSTRACT**  
Audio-visual speech recognition (AVSR) is a multimodal extension of automatic speech recognition (ASR), using video as a complement to audio. In AVSR, considerable efforts have been directed at datasets for facial features such as lip-readings, while they often fall short in evaluating the image comprehension capabilities in broader contexts. In this paper, we construct SlideAVSR, an AVSR dataset using scientific paper explanation videos. SlideAVSR provides a new benchmark where models transcribe speech utterances with texts on the slides on the presentation recordings. As technical terminologies that are frequent in paper explanations are notoriously challenging to transcribe without reference texts, our SlideAVSR dataset spotlights a new aspect of AVSR problems. As a simple yet effective baseline, we propose DocWhisper, an AVSR model that can refer to textual information from slides, and confirm its effectiveness on SlideAVSR.

{{</citation>}}


### (61/132) Image Translation as Diffusion Visual Programmers (Cheng Han et al., 2024)

{{<citation>}}

Cheng Han, James C. Liang, Qifan Wang, Majid Rabbani, Sohail Dianat, Raghuveer Rao, Ying Nian Wu, Dongfang Liu. (2024)  
**Image Translation as Diffusion Visual Programmers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.09742v1)  

---


**ABSTRACT**  
We introduce the novel Diffusion Visual Programmer (DVP), a neuro-symbolic image translation framework. Our proposed DVP seamlessly embeds a condition-flexible diffusion model within the GPT architecture, orchestrating a coherent sequence of visual programs (i.e., computer vision models) for various pro-symbolic steps, which span RoI identification, style transfer, and position manipulation, facilitating transparent and controllable image translation processes. Extensive experiments demonstrate DVP's remarkable performance, surpassing concurrent arts. This success can be attributed to several key features of DVP: First, DVP achieves condition-flexible translation via instance normalization, enabling the model to eliminate sensitivity caused by the manual guidance and optimally focus on textual descriptions for high-quality content generation. Second, the framework enhances in-context reasoning by deciphering intricate high-dimensional concepts in feature spaces into more accessible low-dimensional symbols (e.g., [Prompt], [RoI object]), allowing for localized, context-free editing while maintaining overall coherence. Last but not least, DVP improves systemic controllability and explainability by offering explicit symbolic representations at each programming stage, empowering users to intuitively interpret and modify results. Our research marks a substantial step towards harmonizing artificial image translation processes with cognitive intelligence, promising broader applications.

{{</citation>}}


### (62/132) SkyEyeGPT: Unifying Remote Sensing Vision-Language Tasks via Instruction Tuning with Large Language Model (Yang Zhan et al., 2024)

{{<citation>}}

Yang Zhan, Zhitong Xiong, Yuan Yuan. (2024)  
**SkyEyeGPT: Unifying Remote Sensing Vision-Language Tasks via Instruction Tuning with Large Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09712v1)  

---


**ABSTRACT**  
Large language models (LLMs) have recently been extended to the vision-language realm, obtaining impressive general multi-modal capabilities. However, the exploration of multi-modal large language models (MLLMs) for remote sensing (RS) data is still in its infancy, and the performance is not satisfactory. In this work, we introduce SkyEyeGPT, a unified multi-modal large language model specifically designed for RS vision-language understanding. To this end, we meticulously curate an RS multi-modal instruction tuning dataset, including single-task and multi-task conversation instructions. After manual verification, we obtain a high-quality RS instruction-following dataset with 968k samples. Our research demonstrates that with a simple yet effective design, SkyEyeGPT works surprisingly well on considerably different tasks without the need for extra encoding modules. Specifically, after projecting RS visual features to the language domain via an alignment layer, they are fed jointly with task-specific instructions into an LLM-based RS decoder to predict answers for RS open-ended tasks. In addition, we design a two-stage tuning method to enhance instruction-following and multi-turn dialogue ability at different granularities. Experiments on 8 datasets for RS vision-language tasks demonstrate SkyEyeGPT's superiority in image-level and region-level tasks, such as captioning and visual grounding. In particular, SkyEyeGPT exhibits encouraging results compared to GPT-4V in some qualitative tests. The online demo, code, and dataset will be released in https://github.com/ZhanYang-nwpu/SkyEyeGPT.

{{</citation>}}


### (63/132) Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack (Zhongliang Guo et al., 2024)

{{<citation>}}

Zhongliang Guo, Kaixuan Wang, Weiye Li, Yifei Qian, Ognjen Arandjelović, Lei Fang. (2024)  
**Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2401.09673v1)  

---


**ABSTRACT**  
Neural style transfer (NST) is widely adopted in computer vision to generate new images with arbitrary styles. This process leverages neural networks to merge aesthetic elements of a style image with the structural aspects of a content image into a harmoniously integrated visual result. However, unauthorized NST can exploit artwork. Such misuse raises socio-technical concerns regarding artists' rights and motivates the development of technical approaches for the proactive protection of original creations. Adversarial attack is a concept primarily explored in machine learning security. Our work introduces this technique to protect artists' intellectual property. In this paper Locally Adaptive Adversarial Color Attack (LAACA), a method for altering images in a manner imperceptible to the human eyes but disruptive to NST. Specifically, we design perturbations targeting image areas rich in high-frequency content, generated by disrupting intermediate features. Our experiments and user study confirm that by attacking NST using the proposed method results in visually worse neural style transfer, thus making it an effective solution for visual artwork protection.

{{</citation>}}


## eess.SP (2)



### (64/132) Deep Dict: Deep Learning-based Lossy Time Series Compressor for IoT Data (Jinxin Liu et al., 2024)

{{<citation>}}

Jinxin Liu, Petar Djukic, Michel Kulhandjian, Burak Kantarci. (2024)  
**Deep Dict: Deep Learning-based Lossy Time Series Compressor for IoT Data**  

---
Primary Category: eess.SP  
Categories: cs-IT, cs-LG, eess-SP, eess.SP, math-IT  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2401.10396v1)  

---


**ABSTRACT**  
We propose Deep Dict, a deep learning-based lossy time series compressor designed to achieve a high compression ratio while maintaining decompression error within a predefined range. Deep Dict incorporates two essential components: the Bernoulli transformer autoencoder (BTAE) and a distortion constraint. BTAE extracts Bernoulli representations from time series data, reducing the size of the representations compared to conventional autoencoders. The distortion constraint limits the prediction error of BTAE to the desired range. Moreover, in order to address the limitations of common regression losses such as L1/L2, we introduce a novel loss function called quantized entropy loss (QEL). QEL takes into account the specific characteristics of the problem, enhancing robustness to outliers and alleviating optimization challenges. Our evaluation of Deep Dict across ten diverse time series datasets from various domains reveals that Deep Dict outperforms state-of-the-art lossy compressors in terms of compression ratio by a significant margin by up to 53.66%.

{{</citation>}}


### (65/132) Intelligent Optimization and Machine Learning Algorithms for Structural Anomaly Detection using Seismic Signals (Maximilian Trapp et al., 2024)

{{<citation>}}

Maximilian Trapp, Can Bogoclu, Tamara Nestorović, Dirk Roos. (2024)  
**Intelligent Optimization and Machine Learning Algorithms for Structural Anomaly Detection using Seismic Signals**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP, physics-app-ph  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.10355v1)  

---


**ABSTRACT**  
The lack of anomaly detection methods during mechanized tunnelling can cause financial loss and deficits in drilling time. On-site excavation requires hard obstacles to be recognized prior to drilling in order to avoid damaging the tunnel boring machine and to adjust the propagation velocity. The efficiency of the structural anomaly detection can be increased with intelligent optimization techniques and machine learning. In this research, the anomaly in a simple structure is detected by comparing the experimental measurements of the structural vibrations with numerical simulations using parameter estimation methods.

{{</citation>}}


## cs.LG (25)



### (66/132) Distribution Consistency based Self-Training for Graph Neural Networks with Sparse Labels (Fali Wang et al., 2024)

{{<citation>}}

Fali Wang, Tianxiang Zhao, Suhang Wang. (2024)  
**Distribution Consistency based Self-Training for Graph Neural Networks with Sparse Labels**  

---
Primary Category: cs.LG  
Categories: F-2-2; I-2-7, cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.10394v1)  

---


**ABSTRACT**  
Few-shot node classification poses a significant challenge for Graph Neural Networks (GNNs) due to insufficient supervision and potential distribution shifts between labeled and unlabeled nodes. Self-training has emerged as a widely popular framework to leverage the abundance of unlabeled data, which expands the training set by assigning pseudo-labels to selected unlabeled nodes. Efforts have been made to develop various selection strategies based on confidence, information gain, etc. However, none of these methods takes into account the distribution shift between the training and testing node sets. The pseudo-labeling step may amplify this shift and even introduce new ones, hindering the effectiveness of self-training. Therefore, in this work, we explore the potential of explicitly bridging the distribution shift between the expanded training set and test set during self-training. To this end, we propose a novel Distribution-Consistent Graph Self-Training (DC-GST) framework to identify pseudo-labeled nodes that are both informative and capable of redeeming the distribution discrepancy and formulate it as a differentiable optimization task. A distribution-shift-aware edge predictor is further adopted to augment the graph and increase the model's generalizability in assigning pseudo labels. We evaluate our proposed method on four publicly available benchmark datasets and extensive experiments demonstrate that our framework consistently outperforms state-of-the-art baselines.

{{</citation>}}


### (67/132) Using LLM such as ChatGPT for Designing and Implementing a RISC Processor: Execution,Challenges and Limitations (Shadeeb Hossain et al., 2024)

{{<citation>}}

Shadeeb Hossain, Aayush Gohil, Yizhou Wang. (2024)  
**Using LLM such as ChatGPT for Designing and Implementing a RISC Processor: Execution,Challenges and Limitations**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs-SE, cs.LG  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.10364v1)  

---


**ABSTRACT**  
This paper discusses the feasibility of using Large Language Models LLM for code generation with a particular application in designing an RISC. The paper also reviews the associated steps such as parsing, tokenization, encoding, attention mechanism, sampling the tokens and iterations during code generation. The generated code for the RISC components is verified through testbenches and hardware implementation on a FPGA board. Four metric parameters Correct output on the first iteration, Number of errors embedded in the code, Number of trials required to achieve the code and Failure to generate the code after three iterations, are used to compare the efficiency of using LLM in programming. In all the cases, the generated code had significant errors and human intervention was always required to fix the bugs. LLM can therefore be used to complement a programmer code design.

{{</citation>}}


### (68/132) MELODY: Robust Semi-Supervised Hybrid Model for Entity-Level Online Anomaly Detection with Multivariate Time Series (Jingchao Ni et al., 2024)

{{<citation>}}

Jingchao Ni, Gauthier Guinet, Peihong Jiang, Laurent Callot, Andrey Kan. (2024)  
**MELODY: Robust Semi-Supervised Hybrid Model for Entity-Level Online Anomaly Detection with Multivariate Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, Semi-Supervised, Time Series  
[Paper Link](http://arxiv.org/abs/2401.10338v1)  

---


**ABSTRACT**  
In large IT systems, software deployment is a crucial process in online services as their code is regularly updated. However, a faulty code change may degrade the target service's performance and cause cascading outages in downstream services. Thus, software deployments should be comprehensively monitored, and their anomalies should be detected timely. In this paper, we study the problem of anomaly detection for deployments. We begin by identifying the challenges unique to this anomaly detection problem, which is at entity-level (e.g., deployments), relative to the more typical problem of anomaly detection in multivariate time series (MTS). The unique challenges include the heterogeneity of deployments, the low latency tolerance, the ambiguous anomaly definition, and the limited supervision. To address them, we propose a novel framework, semi-supervised hybrid Model for Entity-Level Online Detection of anomalY (MELODY). MELODY first transforms the MTS of different entities to the same feature space by an online feature extractor, then uses a newly proposed semi-supervised deep one-class model for detecting anomalous entities. We evaluated MELODY on real data of cloud services with 1.2M+ time series. The relative F1 score improvement of MELODY over the state-of-the-art methods ranges from 7.6% to 56.5%. The user evaluation suggests MELODY is suitable for monitoring deployments in large online systems.

{{</citation>}}


### (69/132) Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition (Tu Nguyen et al., 2024)

{{<citation>}}

Tu Nguyen, Nedim Srndic, Alexander Neth. (2024)  
**Noise Contrastive Estimation-based Matching Framework for Low-resource Security Attack Pattern Recognition**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.10337v2)  

---


**ABSTRACT**  
Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning process of the matching model despite constrained resources.

{{</citation>}}


### (70/132) Multi-Agent Reinforcement Learning for Maritime Operational Technology Cyber Security (Alec Wilson et al., 2024)

{{<citation>}}

Alec Wilson, Ryan Menzies, Neela Morarji, David Foster, Marco Casassa Mont, Esin Turkbeyler, Lisa Gralewski. (2024)  
**Multi-Agent Reinforcement Learning for Maritime Operational Technology Cyber Security**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-MA, cs.LG  
Keywords: Cyber Security, Reinforcement Learning, Security  
[Paper Link](http://arxiv.org/abs/2401.10149v1)  

---


**ABSTRACT**  
This paper demonstrates the potential for autonomous cyber defence to be applied on industrial control systems and provides a baseline environment to further explore Multi-Agent Reinforcement Learning's (MARL) application to this problem domain. It introduces a simulation environment, IPMSRL, of a generic Integrated Platform Management System (IPMS) and explores the use of MARL for autonomous cyber defence decision-making on generic maritime based IPMS Operational Technology (OT). OT cyber defensive actions are less mature than they are for Enterprise IT. This is due to the relatively brittle nature of OT infrastructure originating from the use of legacy systems, design-time engineering assumptions, and lack of full-scale modern security controls. There are many obstacles to be tackled across the cyber landscape due to continually increasing cyber-attack sophistication and the limitations of traditional IT-centric cyber defence solutions. Traditional IT controls are rarely deployed on OT infrastructure, and where they are, some threats aren't fully addressed. In our experiments, a shared critic implementation of Multi Agent Proximal Policy Optimisation (MAPPO) outperformed Independent Proximal Policy Optimisation (IPPO). MAPPO reached an optimal policy (episode outcome mean of 1) after 800K timesteps, whereas IPPO was only able to reach an episode outcome mean of 0.966 after one million timesteps. Hyperparameter tuning greatly improved training performance. Across one million timesteps the tuned hyperparameters reached an optimal policy whereas the default hyperparameters only managed to win sporadically, with most simulations resulting in a draw. We tested a real-world constraint, attack detection alert success, and found that when alert success probability is reduced to 0.75 or 0.9, the MARL defenders were still able to win in over 97.5% or 99.5% of episodes, respectively.

{{</citation>}}


### (71/132) Spatial-Temporal Large Language Model for Traffic Prediction (Chenxi Liu et al., 2024)

{{<citation>}}

Chenxi Liu, Sun Yang, Qianxiong Xu, Zhishuai Li, Cheng Long, Ziyue Li, Rui Zhao. (2024)  
**Spatial-Temporal Large Language Model for Traffic Prediction**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10134v2)  

---


**ABSTRACT**  
Traffic prediction, a critical component for intelligent transportation systems, endeavors to foresee future traffic at specific locations using historical data. Although existing traffic prediction models often emphasize developing complex neural network structures, their accuracy has not seen improvements accordingly. Recently, Large Language Models (LLMs) have shown outstanding capabilities in time series analysis. Differing from existing models, LLMs progress mainly through parameter expansion and extensive pre-training while maintaining their fundamental structures. In this paper, we propose a Spatial-Temporal Large Language Model (ST-LLM) for traffic prediction. Specifically, ST-LLM redefines the timesteps at each location as tokens and incorporates a spatial-temporal embedding module to learn the spatial location and global temporal representations of tokens. Then these representations are fused to provide each token with unified spatial and temporal information. Furthermore, we propose a novel partially frozen attention strategy of the LLM, which is designed to capture spatial-temporal dependencies for traffic prediction. Comprehensive experiments on real traffic datasets offer evidence that ST-LLM outperforms state-of-the-art models. Notably, the ST-LLM also exhibits robust performance in both few-shot and zero-shot prediction scenarios.

{{</citation>}}


### (72/132) Towards Principled Graph Transformers (Luis Müller et al., 2024)

{{<citation>}}

Luis Müller, Christopher Morris. (2024)  
**Towards Principled Graph Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.10119v1)  

---


**ABSTRACT**  
Graph learning architectures based on the k-dimensional Weisfeiler-Leman (k-WL) hierarchy offer a theoretically well-understood expressive power. However, such architectures often fail to deliver solid predictive performance on real-world tasks, limiting their practical impact. In contrast, global attention-based models such as graph transformers demonstrate strong performance in practice, but comparing their expressive power with the k-WL hierarchy remains challenging, particularly since these architectures rely on positional or structural encodings for their expressivity and predictive performance. To address this, we show that the recently proposed Edge Transformer, a global attention model operating on node pairs instead of nodes, has at least 3-WL expressive power. Empirically, we demonstrate that the Edge Transformer surpasses other theoretically aligned architectures regarding predictive performance while not relying on positional or structural encodings.

{{</citation>}}


### (73/132) Mathematical Algorithm Design for Deep Learning under Societal and Judicial Constraints: The Algorithmic Transparency Requirement (Holger Boche et al., 2024)

{{<citation>}}

Holger Boche, Adalbert Fono, Gitta Kutyniok. (2024)  
**Mathematical Algorithm Design for Deep Learning under Societal and Judicial Constraints: The Algorithmic Transparency Requirement**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.10310v1)  

---


**ABSTRACT**  
Deep learning still has drawbacks in terms of trustworthiness, which describes a comprehensible, fair, safe, and reliable method. To mitigate the potential risk of AI, clear obligations associated to trustworthiness have been proposed via regulatory guidelines, e.g., in the European AI Act. Therefore, a central question is to what extent trustworthy deep learning can be realized. Establishing the described properties constituting trustworthiness requires that the factors influencing an algorithmic computation can be retraced, i.e., the algorithmic implementation is transparent. Motivated by the observation that the current evolution of deep learning models necessitates a change in computing technology, we derive a mathematical framework which enables us to analyze whether a transparent implementation in a computing model is feasible. We exemplarily apply our trustworthiness framework to analyze deep learning approaches for inverse problems in digital and analog computing models represented by Turing and Blum-Shub-Smale Machines, respectively. Based on previous results, we find that Blum-Shub-Smale Machines have the potential to establish trustworthy solvers for inverse problems under fairly general conditions, whereas Turing machines cannot guarantee trustworthiness to the same degree.

{{</citation>}}


### (74/132) Optimizing Medication Decisions for Patients with Atrial Fibrillation through Path Development Network (Tian Xie, 2024)

{{<citation>}}

Tian Xie. (2024)  
**Optimizing Medication Decisions for Patients with Atrial Fibrillation through Path Development Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.10014v1)  

---


**ABSTRACT**  
Atrial fibrillation (AF) is a common cardiac arrhythmia characterized by rapid and irregular contractions of the atria. It significantly elevates the risk of strokes due to slowed blood flow in the atria, especially in the left atrial appendage, which is prone to blood clot formation. Such clots can migrate into cerebral arteries, leading to ischemic stroke. To assess whether AF patients should be prescribed anticoagulants, doctors often use the CHA2DS2-VASc scoring system. However, anticoagulant use must be approached with caution as it can impact clotting functions. This study introduces a machine learning algorithm that predicts whether patients with AF should be recommended anticoagulant therapy using 12-lead ECG data. In this model, we use STOME to enhance time-series data and then process it through a Convolutional Neural Network (CNN). By incorporating a path development layer, the model achieves a specificity of 30.6% under the condition of an NPV of 1. In contrast, LSTM algorithms without path development yield a specificity of only 2.7% under the same NPV condition.

{{</citation>}}


### (75/132) Developing an AI-based Integrated System for Bee Health Evaluation (Andrew Liang, 2024)

{{<citation>}}

Andrew Liang. (2024)  
**Developing an AI-based Integrated System for Bee Health Evaluation**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-SD, cs.LG, eess-AS  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2401.09988v1)  

---


**ABSTRACT**  
Honey bees pollinate about one-third of the world's food supply, but bee colonies have alarmingly declined by nearly 40% over the past decade due to several factors, including pesticides and pests. Traditional methods for monitoring beehives, such as human inspection, are subjective, disruptive, and time-consuming. To overcome these limitations, artificial intelligence has been used to assess beehive health. However, previous studies have lacked an end-to-end solution and primarily relied on data from a single source, either bee images or sounds. This study introduces a comprehensive system consisting of bee object detection and health evaluation. Additionally, it utilized a combination of visual and audio signals to analyze bee behaviors. An Attention-based Multimodal Neural Network (AMNN) was developed to adaptively focus on key features from each type of signal for accurate bee health assessment. The AMNN achieved an overall accuracy of 92.61%, surpassing eight existing single-signal Convolutional Neural Networks and Recurrent Neural Networks. It outperformed the best image-based model by 32.51% and the top sound-based model by 13.98% while maintaining efficient processing times. Furthermore, it improved prediction robustness, attaining an F1-score higher than 90% across all four evaluated health conditions. The study also shows that audio signals are more reliable than images for assessing bee health. By seamlessly integrating AMNN with image and sound data in a comprehensive bee health monitoring system, this approach provides a more efficient and non-invasive solution for the early detection of bee diseases and the preservation of bee colonies.

{{</citation>}}


### (76/132) Through the Dual-Prism: A Spectral Perspective on Graph Data Augmentation for Graph Classification (Yutong Xia et al., 2024)

{{<citation>}}

Yutong Xia, Runpeng Yu, Yuxuan Liang, Xavier Bresson, Xinchao Wang, Roger Zimmermann. (2024)  
**Through the Dual-Prism: A Spectral Perspective on Graph Data Augmentation for Graph Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.09953v2)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have become the preferred tool to process graph data, with their efficacy being boosted through graph data augmentation techniques. Despite the evolution of augmentation methods, issues like graph property distortions and restricted structural changes persist. This leads to the question: Is it possible to develop more property-conserving and structure-sensitive augmentation methods? Through a spectral lens, we investigate the interplay between graph properties, their augmentation, and their spectral behavior, and found that keeping the low-frequency eigenvalues unchanged can preserve the critical properties at a large scale when generating augmented graphs. These observations inform our introduction of the Dual-Prism (DP) augmentation method, comprising DP-Noise and DP-Mask, which adeptly retains essential graph properties while diversifying augmented graphs. Extensive experiments validate the efficiency of our approach, providing a new and promising direction for graph data augmentation.

{{</citation>}}


### (77/132) SymbolNet: Neural Symbolic Regression with Adaptive Dynamic Pruning (Ho Fung Tsoi et al., 2024)

{{<citation>}}

Ho Fung Tsoi, Vladimir Loncar, Sridhara Dasu, Philip Harris. (2024)  
**SymbolNet: Neural Symbolic Regression with Adaptive Dynamic Pruning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, hep-ex, physics-ins-det  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2401.09949v1)  

---


**ABSTRACT**  
Contrary to the use of genetic programming, the neural network approach to symbolic regression can scale well with high input dimension and leverage gradient methods for faster equation searching. Common ways of constraining expression complexity have relied on multistage pruning methods with fine-tuning, but these often lead to significant performance loss. In this work, we propose SymbolNet, a neural network approach to symbolic regression in a novel framework that enables dynamic pruning of model weights, input features, and mathematical operators in a single training, where both training loss and expression complexity are optimized simultaneously. We introduce a sparsity regularization term per pruning type, which can adaptively adjust its own strength and lead to convergence to a target sparsity level. In contrast to most existing symbolic regression methods that cannot efficiently handle datasets with more than $O$(10) inputs, we demonstrate the effectiveness of our model on the LHC jet tagging task (16 inputs), MNIST (784 inputs), and SVHN (3072 inputs).

{{</citation>}}


### (78/132) HGAttack: Transferable Heterogeneous Graph Adversarial Attack (He Zhao et al., 2024)

{{<citation>}}

He Zhao, Zhiwei Zeng, Yongwei Wang, Deheng Ye, Chunyan Miao. (2024)  
**HGAttack: Transferable Heterogeneous Graph Adversarial Attack**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-IR, cs-LG, cs.LG  
Keywords: Adversarial Attack, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.09945v1)  

---


**ABSTRACT**  
Heterogeneous Graph Neural Networks (HGNNs) are increasingly recognized for their performance in areas like the web and e-commerce, where resilience against adversarial attacks is crucial. However, existing adversarial attack methods, which are primarily designed for homogeneous graphs, fall short when applied to HGNNs due to their limited ability to address the structural and semantic complexity of HGNNs. This paper introduces HGAttack, the first dedicated gray box evasion attack method for heterogeneous graphs. We design a novel surrogate model to closely resemble the behaviors of the target HGNN and utilize gradient-based methods for perturbation generation. Specifically, the proposed surrogate model effectively leverages heterogeneous information by extracting meta-path induced subgraphs and applying GNNs to learn node embeddings with distinct semantics from each subgraph. This approach improves the transferability of generated attacks on the target HGNN and significantly reduces memory costs. For perturbation generation, we introduce a semantics-aware mechanism that leverages subgraph gradient information to autonomously identify vulnerable edges across a wide range of relations within a constrained perturbation budget. We validate HGAttack's efficacy with comprehensive experiments on three datasets, providing empirical analyses of its generated perturbations. Outperforming baseline methods, HGAttack demonstrated significant efficacy in diminishing the performance of target HGNN models, affirming the effectiveness of our approach in evaluating the robustness of HGNNs against adversarial attacks.

{{</citation>}}


### (79/132) Infinite-Horizon Graph Filters: Leveraging Power Series to Enhance Sparse Information Aggregation (Ruizhe Zhang et al., 2024)

{{<citation>}}

Ruizhe Zhang, Xinke Jiang, Yuchen Fang, Jiayuan Luo, Yongxin Xu, Yichen Zhu, Xu Chu, Junfeng Zhao, Yasha Wang. (2024)  
**Infinite-Horizon Graph Filters: Leveraging Power Series to Enhance Sparse Information Aggregation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.09943v2)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have shown considerable effectiveness in a variety of graph learning tasks, particularly those based on the message-passing approach in recent years. However, their performance is often constrained by a limited receptive field, a challenge that becomes more acute in the presence of sparse graphs. In light of the power series, which possesses infinite expansion capabilities, we propose a novel Graph Power Filter Neural Network (GPFN) that enhances node classification by employing a power series graph filter to augment the receptive field. Concretely, our GPFN designs a new way to build a graph filter with an infinite receptive field based on the convergence power series, which can be analyzed in the spectral and spatial domains. Besides, we theoretically prove that our GPFN is a general framework that can integrate any power series and capture long-range dependencies. Finally, experimental results on three datasets demonstrate the superiority of our GPFN over state-of-the-art baselines.

{{</citation>}}


### (80/132) Biases in Expected Goals Models Confound Finishing Ability (Jesse Davis et al., 2024)

{{<citation>}}

Jesse Davis, Pieter Robberechts. (2024)  
**Biases in Expected Goals Models Confound Finishing Ability**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2401.09940v1)  

---


**ABSTRACT**  
Expected Goals (xG) has emerged as a popular tool for evaluating finishing skill in soccer analytics. It involves comparing a player's cumulative xG with their actual goal output, where consistent overperformance indicates strong finishing ability. However, the assessment of finishing skill in soccer using xG remains contentious due to players' difficulty in consistently outperforming their cumulative xG. In this paper, we aim to address the limitations and nuances surrounding the evaluation of finishing skill using xG statistics. Specifically, we explore three hypotheses: (1) the deviation between actual and expected goals is an inadequate metric due to the high variance of shot outcomes and limited sample sizes, (2) the inclusion of all shots in cumulative xG calculation may be inappropriate, and (3) xG models contain biases arising from interdependencies in the data that affect skill measurement. We found that sustained overperformance of cumulative xG requires both high shot volumes and exceptional finishing, including all shot types can obscure the finishing ability of proficient strikers, and that there is a persistent bias that makes the actual and expected goals closer for excellent finishers than it really is. Overall, our analysis indicates that we need more nuanced quantitative approaches for investigating a player's finishing ability, which we achieved using a technique from AI fairness to learn an xG model that is calibrated for multiple subgroups of players. As a concrete use case, we show that (1) the standard biased xG model underestimates Messi's GAX by 17% and (2) Messi's GAX is 27% higher than the typical elite high-shot-volume attacker, indicating that Messi is even a more exceptional finisher than people commonly believed.

{{</citation>}}


### (81/132) Cooperative Edge Caching Based on Elastic Federated and Multi-Agent Deep Reinforcement Learning in Next-Generation Network (Qiong Wu et al., 2024)

{{<citation>}}

Qiong Wu, Wenhua Wang, Pingyi Fan, Qiang Fan, Huiling Zhu, Khaled B. Letaief. (2024)  
**Cooperative Edge Caching Based on Elastic Federated and Multi-Agent Deep Reinforcement Learning in Next-Generation Network**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09886v1)  

---


**ABSTRACT**  
Edge caching is a promising solution for next-generation networks by empowering caching units in small-cell base stations (SBSs), which allows user equipments (UEs) to fetch users' requested contents that have been pre-cached in SBSs. It is crucial for SBSs to predict accurate popular contents through learning while protecting users' personal information. Traditional federated learning (FL) can protect users' privacy but the data discrepancies among UEs can lead to a degradation in model quality. Therefore, it is necessary to train personalized local models for each UE to predict popular contents accurately. In addition, the cached contents can be shared among adjacent SBSs in next-generation networks, thus caching predicted popular contents in different SBSs may affect the cost to fetch contents. Hence, it is critical to determine where the popular contents are cached cooperatively. To address these issues, we propose a cooperative edge caching scheme based on elastic federated and multi-agent deep reinforcement learning (CEFMR) to optimize the cost in the network. We first propose an elastic FL algorithm to train the personalized model for each UE, where adversarial autoencoder (AAE) model is adopted for training to improve the prediction accuracy, then {a popular} content prediction algorithm is proposed to predict the popular contents for each SBS based on the trained AAE model. Finally, we propose a multi-agent deep reinforcement learning (MADRL) based algorithm to decide where the predicted popular contents are collaboratively cached among SBSs. Our experimental results demonstrate the superiority of our proposed scheme to existing baseline caching schemes.

{{</citation>}}


### (82/132) GA-SmaAt-GNet: Generative Adversarial Small Attention GNet for Extreme Precipitation Nowcasting (Eloy Reulen et al., 2024)

{{<citation>}}

Eloy Reulen, Siamak Mehrkanoon. (2024)  
**GA-SmaAt-GNet: Generative Adversarial Small Attention GNet for Extreme Precipitation Nowcasting**  

---
Primary Category: cs.LG  
Categories: I-2; I-5, cs-LG, cs.LG, physics-ao-ph  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.09881v1)  

---


**ABSTRACT**  
In recent years, data-driven modeling approaches have gained considerable traction in various meteorological applications, particularly in the realm of weather forecasting. However, these approaches often encounter challenges when dealing with extreme weather conditions. In light of this, we propose GA-SmaAt-GNet, a novel generative adversarial architecture that makes use of two methodologies aimed at enhancing the performance of deep learning models for extreme precipitation nowcasting. Firstly, it uses a novel SmaAt-GNet built upon the successful SmaAt-UNet architecture as generator. This network incorporates precipitation masks (binarized precipitation maps) as an additional data source, leveraging valuable information for improved predictions. Additionally, GA-SmaAt-GNet utilizes an attention-augmented discriminator inspired by the well-established Pix2Pix architecture. Furthermore, we assess the performance of GA-SmaAt-GNet using real-life precipitation dataset from the Netherlands. Our experimental results reveal a notable improvement in both overall performance and for extreme precipitation events. Furthermore, we conduct uncertainty analysis on the proposed GA-SmaAt-GNet model as well as on the precipitation dataset, providing additional insights into the predictive capabilities of the model. Finally, we offer further insights into the predictions of our proposed model using Grad-CAM. This visual explanation technique generates activation heatmaps, illustrating areas of the input that are more activated for various parts of the network.

{{</citation>}}


### (83/132) Reconciling Spatial and Temporal Abstractions for Goal Representation (Mehdi Zadem et al., 2024)

{{<citation>}}

Mehdi Zadem, Sergio Mover, Sao Mai Nguyen. (2024)  
**Reconciling Spatial and Temporal Abstractions for Goal Representation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09870v1)  

---


**ABSTRACT**  
Goal representation affects the performance of Hierarchical Reinforcement Learning (HRL) algorithms by decomposing the complex learning problem into easier subtasks. Recent studies show that representations that preserve temporally abstract environment dynamics are successful in solving difficult problems and provide theoretical guarantees for optimality. These methods however cannot scale to tasks where environment dynamics increase in complexity i.e. the temporally abstract transition relations depend on larger number of variables. On the other hand, other efforts have tried to use spatial abstraction to mitigate the previous issues. Their limitations include scalability to high dimensional environments and dependency on prior knowledge.   In this paper, we propose a novel three-layer HRL algorithm that introduces, at different levels of the hierarchy, both a spatial and a temporal goal abstraction. We provide a theoretical study of the regret bounds of the learned policies. We evaluate the approach on complex continuous control tasks, demonstrating the effectiveness of spatial and temporal abstractions learned by this approach.

{{</citation>}}


### (84/132) A Fast, Performant, Secure Distributed Training Framework For Large Language Model (Wei Huang et al., 2024)

{{<citation>}}

Wei Huang, Yinggui Wang, Anda Cheng, Aihui Zhou, Chaofan Yu, Lei Wang. (2024)  
**A Fast, Performant, Secure Distributed Training Framework For Large Language Model**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09796v2)  

---


**ABSTRACT**  
The distributed (federated) LLM is an important method for co-training the domain-specific LLM using siloed data. However, maliciously stealing model parameters and data from the server or client side has become an urgent problem to be solved. In this paper, we propose a secure distributed LLM based on model slicing. In this case, we deploy the Trusted Execution Environment (TEE) on both the client and server side, and put the fine-tuned structure (LoRA or embedding of P-tuning v2) into the TEE. Then, secure communication is executed in the TEE and general environments through lightweight encryption. In order to further reduce the equipment cost as well as increase the model performance and accuracy, we propose a split fine-tuning scheme. In particular, we split the LLM by layers and place the latter layers in a server-side TEE (the client does not need a TEE). We then combine the proposed Sparsification Parameter Fine-tuning (SPF) with the LoRA part to improve the accuracy of the downstream task. Numerous experiments have shown that our method guarantees accuracy while maintaining security.

{{</citation>}}


### (85/132) PatchAD: Patch-based MLP-Mixer for Time Series Anomaly Detection (Zhijie Zhong et al., 2024)

{{<citation>}}

Zhijie Zhong, Zhiwen Yu, Yiyuan Yang, Weizheng Wang, Kaixiang Yang. (2024)  
**PatchAD: Patch-based MLP-Mixer for Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2401.09793v2)  

---


**ABSTRACT**  
Anomaly detection stands as a crucial aspect of time series analysis, aiming to identify abnormal events in time series samples. The central challenge of this task lies in effectively learning the representations of normal and abnormal patterns in a label-lacking scenario. Previous research mostly relied on reconstruction-based approaches, restricting the representational abilities of the models. In addition, most of the current deep learning-based methods are not lightweight enough, which prompts us to design a more efficient framework for anomaly detection. In this study, we introduce PatchAD, a novel multi-scale patch-based MLP-Mixer architecture that leverages contrastive learning for representational extraction and anomaly detection. Specifically, PatchAD is composed of four distinct MLP Mixers, exclusively utilizing the MLP architecture for high efficiency and lightweight architecture. Additionally, we also innovatively crafted a dual project constraint module to mitigate potential model degradation. Comprehensive experiments demonstrate that PatchAD achieves state-of-the-art results across multiple real-world multivariate time series datasets. Our code is publicly available.\footnote{\url{https://github.com/EmorZz1G/PatchAD}}

{{</citation>}}


### (86/132) Querying Easily Flip-flopped Samples for Deep Active Learning (Seong Jin Cho et al., 2024)

{{<citation>}}

Seong Jin Cho, Gwangsu Kim, Junghyun Lee, Jinwoo Shin, Chang D. Yoo. (2024)  
**Querying Easily Flip-flopped Samples for Deep Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2401.09787v1)  

---


**ABSTRACT**  
Active learning is a machine learning paradigm that aims to improve the performance of a model by strategically selecting and querying unlabeled data. One effective selection strategy is to base it on the model's predictive uncertainty, which can be interpreted as a measure of how informative a sample is. The sample's distance to the decision boundary is a natural measure of predictive uncertainty, but it is often intractable to compute, especially for complex decision boundaries formed in multiclass classification tasks. To address this issue, this paper proposes the {\it least disagree metric} (LDM), defined as the smallest probability of disagreement of the predicted label, and an estimator for LDM proven to be asymptotically consistent under mild assumptions. The estimator is computationally efficient and can be easily implemented for deep learning models using parameter perturbation. The LDM-based active learning is performed by querying unlabeled data with the smallest LDM. Experimental results show that our LDM-based active learning algorithm obtains state-of-the-art overall performance on all considered datasets and deep architectures.

{{</citation>}}


### (87/132) Universally Robust Graph Neural Networks by Preserving Neighbor Similarity (Yulin Zhu et al., 2024)

{{<citation>}}

Yulin Zhu, Yuni Lai, Xing Ai, Kai Zhou. (2024)  
**Universally Robust Graph Neural Networks by Preserving Neighbor Similarity**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.09754v1)  

---


**ABSTRACT**  
Despite the tremendous success of graph neural networks in learning relational data, it has been widely investigated that graph neural networks are vulnerable to structural attacks on homophilic graphs. Motivated by this, a surge of robust models is crafted to enhance the adversarial robustness of graph neural networks on homophilic graphs. However, the vulnerability based on heterophilic graphs remains a mystery to us. To bridge this gap, in this paper, we start to explore the vulnerability of graph neural networks on heterophilic graphs and theoretically prove that the update of the negative classification loss is negatively correlated with the pairwise similarities based on the powered aggregated neighbor features. This theoretical proof explains the empirical observations that the graph attacker tends to connect dissimilar node pairs based on the similarities of neighbor features instead of ego features both on homophilic and heterophilic graphs. In this way, we novelly introduce a novel robust model termed NSPGNN which incorporates a dual-kNN graphs pipeline to supervise the neighbor similarity-guided propagation. This propagation utilizes the low-pass filter to smooth the features of node pairs along the positive kNN graphs and the high-pass filter to discriminate the features of node pairs along the negative kNN graphs. Extensive experiments on both homophilic and heterophilic graphs validate the universal robustness of NSPGNN compared to the state-of-the-art methods.

{{</citation>}}


### (88/132) Applications of Machine Learning to Optimizing Polyolefin Manufacturing (Niket Sharma et al., 2024)

{{<citation>}}

Niket Sharma, Y. A. Liu. (2024)  
**Applications of Machine Learning to Optimizing Polyolefin Manufacturing**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09753v1)  

---


**ABSTRACT**  
This chapter is a preprint from our book by , focusing on leveraging machine learning (ML) in chemical and polyolefin manufacturing optimization. It's crafted for both novices and seasoned professionals keen on the latest ML applications in chemical processes. We trace the evolution of AI and ML in chemical industries, delineate core ML components, and provide resources for ML beginners. A detailed discussion on various ML methods is presented, covering regression, classification, and unsupervised learning techniques, with performance metrics and examples. Ensemble methods, deep learning networks, including MLP, DNNs, RNNs, CNNs, and transformers, are explored for their growing role in chemical applications. Practical workshops guide readers through predictive modeling using advanced ML algorithms. The chapter culminates with insights into science-guided ML, advocating for a hybrid approach that enhances model accuracy. The extensive bibliography offers resources for further research and practical implementation. This chapter aims to be a thorough primer on ML's practical application in chemical engineering, particularly for polyolefin production, and sets the stage for continued learning in subsequent chapters. Please cite the original work [169,170] when referencing.

{{</citation>}}


### (89/132) Exploration and Anti-Exploration with Distributional Random Network Distillation (Kai Yang et al., 2024)

{{<citation>}}

Kai Yang, Jian Tao, Jiafei Lyu, Xiu Li. (2024)  
**Exploration and Anti-Exploration with Distributional Random Network Distillation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Network Distillation  
[Paper Link](http://arxiv.org/abs/2401.09750v1)  

---


**ABSTRACT**  
Exploration remains a critical issue in deep reinforcement learning for an agent to attain high returns in unknown environments. Although the prevailing exploration Random Network Distillation (RND) algorithm has been demonstrated to be effective in numerous environments, it often needs more discriminative power in bonus allocation. This paper highlights the ``bonus inconsistency'' issue within RND, pinpointing its primary limitation. To address this issue, we introduce the Distributional RND (DRND), a derivative of the RND. DRND enhances the exploration process by distilling a distribution of random networks and implicitly incorporating pseudo counts to improve the precision of bonus allocation. This refinement encourages agents to engage in more extensive exploration. Our method effectively mitigates the inconsistency issue without introducing significant computational overhead. Both theoretical analysis and experimental results demonstrate the superiority of our approach over the original RND algorithm. Our method excels in challenging online exploration scenarios and effectively serves as an anti-exploration mechanism in D4RL offline tasks.

{{</citation>}}


### (90/132) Harnessing Density Ratios for Online Reinforcement Learning (Philip Amortila et al., 2024)

{{<citation>}}

Philip Amortila, Dylan J. Foster, Nan Jiang, Ayush Sekhari, Tengyang Xie. (2024)  
**Harnessing Density Ratios for Online Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09681v1)  

---


**ABSTRACT**  
The theories of offline and online reinforcement learning, despite having evolved in parallel, have begun to show signs of the possibility for a unification, with algorithms and analysis techniques for one setting often having natural counterparts in the other. However, the notion of density ratio modeling, an emerging paradigm in offline RL, has been largely absent from online RL, perhaps for good reason: the very existence and boundedness of density ratios relies on access to an exploratory dataset with good coverage, but the core challenge in online RL is to collect such a dataset without having one to start. In this work we show -- perhaps surprisingly -- that density ratio-based algorithms have online counterparts. Assuming only the existence of an exploratory distribution with good coverage, a structural condition known as coverability (Xie et al., 2023), we give a new algorithm (GLOW) that uses density ratio realizability and value function realizability to perform sample-efficient online exploration. GLOW addresses unbounded density ratios via careful use of truncation, and combines this with optimism to guide exploration. GLOW is computationally inefficient; we complement it with a more efficient counterpart, HyGLOW, for the Hybrid RL setting (Song et al., 2022) wherein online RL is augmented with additional offline data. HyGLOW is derived as a special case of a more general meta-algorithm that provides a provable black-box reduction from hybrid RL to offline RL, which may be of independent interest.

{{</citation>}}


## cs.SE (5)



### (91/132) MutaBot: A Mutation Testing Approach for Chatbots (Michael Ferdinando Urrico et al., 2024)

{{<citation>}}

Michael Ferdinando Urrico, Diego Clerissi, Leonardo Mariani. (2024)  
**MutaBot: A Mutation Testing Approach for Chatbots**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Dialog, Google  
[Paper Link](http://arxiv.org/abs/2401.10372v1)  

---


**ABSTRACT**  
Mutation testing is a technique aimed at assessing the effectiveness of test suites by seeding artificial faults into programs. Although available for many platforms and languages, no mutation testing tool is currently available for conversational chatbots, which represent an increasingly popular solution to design systems that can interact with users through a natural language interface. Note that since conversations must be explicitly engineered by the developers of conversational chatbots, these systems are exposed to specific types of faults not supported by existing mutation testing tools.   In this paper, we present MutaBot, a mutation testing tool for conversational chatbots. MutaBot addresses mutations at multiple levels, including conversational flows, intents, and contexts. We designed the tool to potentially target multiple platforms, while we implemented initial support for Google Dialogflow chatbots. We assessed the tool with three Dialogflow chatbots and test cases generated with Botium, revealing weaknesses in the test suites.

{{</citation>}}


### (92/132) gFaaS: Enabling Generic Functions in Serverless Computing (Mohak Chadha et al., 2024)

{{<citation>}}

Mohak Chadha, Paul Wieland, Michael Gerndt. (2024)  
**gFaaS: Enabling Generic Functions in Serverless Computing**  

---
Primary Category: cs.SE  
Categories: cs-DC, cs-SE, cs.SE  
Keywords: AWS  
[Paper Link](http://arxiv.org/abs/2401.10367v1)  

---


**ABSTRACT**  
With the advent of AWS Lambda in 2014, Serverless Computing, particularly Function-as-a-Service (FaaS), has witnessed growing popularity across various application domains. FaaS enables an application to be decomposed into fine-grained functions that are executed on a FaaS platform. It offers several advantages such as no infrastructure management, a pay-per-use billing policy, and on-demand fine-grained autoscaling. However, despite its advantages, developers today encounter various challenges while adopting FaaS solutions that reduce productivity. These include FaaS platform lock-in, support for diverse function deployment parameters, and diverse interfaces for interacting with FaaS platforms. To address these challenges, we present gFaaS, a novel framework that facilitates the holistic development and management of functions across diverse FaaS platforms. Our framework enables the development of generic functions in multiple programming languages that can be seamlessly deployed across different platforms without modifications. Results from our experiments demonstrate that gFaaS functions perform similarly to native platform-specific functions across various scenarios. A video demonstrating the functioning of gFaaS is available from https://youtu.be/STbb6ykJFf0.

{{</citation>}}


### (93/132) LangProp: A code optimization framework using Language Models applied to driving (Shu Ishida et al., 2024)

{{<citation>}}

Shu Ishida, Gianluca Corrado, George Fedoseev, Hudson Yeo, Lloyd Russell, Jamie Shotton, João F. Henriques, Anthony Hu. (2024)  
**LangProp: A code optimization framework using Language Models applied to driving**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-RO, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10314v1)  

---


**ABSTRACT**  
LangProp is a framework for iteratively optimizing code generated by large language models (LLMs) in a supervised/reinforcement learning setting. While LLMs can generate sensible solutions zero-shot, the solutions are often sub-optimal. Especially for code generation tasks, it is likely that the initial code will fail on certain edge cases. LangProp automatically evaluates the code performance on a dataset of input-output pairs, as well as catches any exceptions, and feeds the results back to the LLM in the training loop, so that the LLM can iteratively improve the code it generates. By adopting a metric- and data-driven training paradigm for this code optimization procedure, one could easily adapt findings from traditional machine learning techniques such as imitation learning, DAgger, and reinforcement learning. We demonstrate the first proof of concept of automated code optimization for autonomous driving in CARLA, showing that LangProp can generate interpretable and transparent driving policies that can be verified and improved in a metric- and data-driven way. Our code will be open-sourced and is available at https://github.com/shuishida/LangProp.

{{</citation>}}


### (94/132) When Neural Code Completion Models Size up the Situation: Attaining Cheaper and Faster Completion through Dynamic Model Inference (Zhensu Sun et al., 2024)

{{<citation>}}

Zhensu Sun, Xiaoning Du, Fu Song, Shangwen Wang, Li Li. (2024)  
**When Neural Code Completion Models Size up the Situation: Attaining Cheaper and Faster Completion through Dynamic Model Inference**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.09964v1)  

---


**ABSTRACT**  
Leveraging recent advancements in large language models, modern neural code completion models have demonstrated the capability to generate highly accurate code suggestions. However, their massive size poses challenges in terms of computational costs and environmental impact, hindering their widespread adoption in practical scenarios. Dynamic inference emerges as a promising solution, as it allocates minimal computation during inference while maintaining the model's performance. In this research, we explore dynamic inference within the context of code completion. Initially, we conducted an empirical investigation on GPT-2, focusing on the inference capabilities of intermediate layers for code completion. We found that 54.4% of tokens can be accurately generated using just the first layer, signifying significant computational savings potential. Moreover, despite using all layers, the model still fails to predict 14.5% of tokens correctly, and the subsequent completions continued from them are rarely considered helpful, with only a 4.2% Acceptance Rate. These findings motivate our exploration of dynamic inference in code completion and inspire us to enhance it with a decision-making mechanism that stops the generation of incorrect code. We thus propose a novel dynamic inference method specifically tailored for code completion models. This method aims not only to produce correct predictions with largely reduced computation but also to prevent incorrect predictions proactively. Our extensive evaluation shows that it can averagely skip 1.7 layers out of 16 layers in the models, leading to an 11.2% speedup with only a marginal 1.1% reduction in ROUGE-L.

{{</citation>}}


### (95/132) SensoDat: Simulation-based Sensor Dataset of Self-driving Cars (Christian Birchler et al., 2024)

{{<citation>}}

Christian Birchler, Cyrill Rohrbach, Timo Kehrer, Sebastiano Panichella. (2024)  
**SensoDat: Simulation-based Sensor Dataset of Self-driving Cars**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09808v1)  

---


**ABSTRACT**  
Developing tools in the context of autonomous systems [22, 24 ], such as self-driving cars (SDCs), is time-consuming and costly since researchers and practitioners rely on expensive computing hardware and simulation software. We propose SensoDat, a dataset of 32,580 executed simulation-based SDC test cases generated with state-of-the-art test generators for SDCs. The dataset consists of trajectory logs and a variety of sensor data from the SDCs (e.g., rpm, wheel speed, brake thermals, transmission, etc.) represented as a time series. In total, SensoDat provides data from 81 different simulated sensors. Future research in the domain of SDCs does not necessarily depend on executing expensive test cases when using SensoDat. Furthermore, with the high amount and variety of sensor data, we think SensoDat can contribute to research, particularly for AI development, regression testing techniques for simulation-based SDC testing, flakiness in simulation, etc. Link to the dataset: https://doi.org/10.5281/zenodo.10307479

{{</citation>}}


## q-fin.CP (1)



### (96/132) Deep Generative Modeling for Financial Time Series with Application in VaR: A Comparative Review (Lars Ericson et al., 2024)

{{<citation>}}

Lars Ericson, Xuejun Zhu, Xusi Han, Rao Fu, Shuang Li, Steve Guo, Ping Hu. (2024)  
**Deep Generative Modeling for Financial Time Series with Application in VaR: A Comparative Review**  

---
Primary Category: q-fin.CP  
Categories: cs-LG, q-fin-CP, q-fin-RM, q-fin-ST, q-fin.CP  
Keywords: Financial, Time Series  
[Paper Link](http://arxiv.org/abs/2401.10370v1)  

---


**ABSTRACT**  
In the financial services industry, forecasting the risk factor distribution conditional on the history and the current market environment is the key to market risk modeling in general and value at risk (VaR) model in particular. As one of the most widely adopted VaR models in commercial banks, Historical simulation (HS) uses the empirical distribution of daily returns in a historical window as the forecast distribution of risk factor returns in the next day. The objectives for financial time series generation are to generate synthetic data paths with good variety, and similar distribution and dynamics to the original historical data. In this paper, we apply multiple existing deep generative methods (e.g., CGAN, CWGAN, Diffusion, and Signature WGAN) for conditional time series generation, and propose and test two new methods for conditional multi-step time series generation, namely Encoder-Decoder CGAN and Conditional TimeVAE. Furthermore, we introduce a comprehensive framework with a set of KPIs to measure the quality of the generated time series for financial modeling. The KPIs cover distribution distance, autocorrelation and backtesting. All models (HS, parametric and neural networks) are tested on both historical USD yield curve data and additional data simulated from GARCH and CIR processes. The study shows that top performing models are HS, GARCH and CWGAN models. Future research directions in this area are also discussed.

{{</citation>}}


## cs.NI (2)



### (97/132) HRL-TSCH: A Hierarchical Reinforcement Learning-based TSCH Scheduler for IIoT (F. Fernando Jurado-Lasso et al., 2024)

{{<citation>}}

F. Fernando Jurado-Lasso, Charalampos Orfanidis, J. F. Jurado, Xenofon Fafoutis. (2024)  
**HRL-TSCH: A Hierarchical Reinforcement Learning-based TSCH Scheduler for IIoT**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.10368v1)  

---


**ABSTRACT**  
The Industrial Internet of Things (IIoT) demands adaptable Networked Embedded Systems (NES) for optimal performance. Combined with recent advances in Artificial Intelligence (AI), tailored solutions can be developed to meet specific application requirements. This study introduces HRL-TSCH, an approach rooted in Hierarchical Reinforcement Learning (HRL), to devise Time Slotted Channel Hopping (TSCH) schedules provisioning IIoT demand. HRL-TSCH employs dual policies: one at a higher level for TSCH schedule link management, and another at a lower level for timeslot and channel assignments. The proposed RL agents address a multi-objective problem, optimizing throughput, power efficiency, and network delay based on predefined application requirements. Simulation experiments demonstrate HRL-TSCH superiority over existing state-of-art approaches, effectively achieving an optimal balance between throughput, power consumption, and delay, thereby enhancing IIoT network performance.

{{</citation>}}


### (98/132) Tailoring Semantic Communication at Network Edge: A Novel Approach Using Dynamic Knowledge Distillation (Abdullatif Albaseer et al., 2024)

{{<citation>}}

Abdullatif Albaseer, Mohamed Abdallah. (2024)  
**Tailoring Semantic Communication at Network Edge: A Novel Approach Using Dynamic Knowledge Distillation**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.10214v1)  

---


**ABSTRACT**  
Semantic Communication (SemCom) systems, empowered by deep learning (DL), represent a paradigm shift in data transmission. These systems prioritize the significance of content over sheer data volume. However, existing SemCom designs face challenges when applied to diverse computational capabilities and network conditions, particularly in time-sensitive applications. A key challenge is the assumption that diverse devices can uniformly benefit from a standard, large DL model in SemCom systems. This assumption becomes increasingly impractical, especially in high-speed, high-reliability applications such as industrial automation or critical healthcare. Therefore, this paper introduces a novel SemCom framework tailored for heterogeneous, resource-constrained edge devices and computation-intensive servers. Our approach employs dynamic knowledge distillation (KD) to customize semantic models for each device, balancing computational and communication constraints while ensuring Quality of Service (QoS). We formulate an optimization problem and develop an adaptive algorithm that iteratively refines semantic knowledge on edge devices, resulting in better models tailored to their resource profiles. This algorithm strategically adjusts the granularity of distilled knowledge, enabling devices to maintain high semantic accuracy for precise inference tasks, even under unstable network conditions. Extensive simulations demonstrate that our approach significantly reduces model complexity for edge devices, leading to better semantic extraction and achieving the desired QoS.

{{</citation>}}


## cs.CR (6)



### (99/132) Excuse me, sir? Your language model is leaking (information) (Or Zamir, 2024)

{{<citation>}}

Or Zamir. (2024)  
**Excuse me, sir? Your language model is leaking (information)**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10360v1)  

---


**ABSTRACT**  
We introduce a cryptographic method to hide an arbitrary secret payload in the response of a Large Language Model (LLM). A secret key is required to extract the payload from the model's response, and without the key it is provably impossible to distinguish between the responses of the original LLM and the LLM that hides a payload. In particular, the quality of generated text is not affected by the payload. Our approach extends a recent result of Christ, Gunn and Zamir (2023) who introduced an undetectable watermarking scheme for LLMs.

{{</citation>}}


### (100/132) Hacking Predictors Means Hacking Cars: Using Sensitivity Analysis to Identify Trajectory Prediction Vulnerabilities for Autonomous Driving Security (Marsalis Gibson et al., 2024)

{{<citation>}}

Marsalis Gibson, David Babazadeh, Claire Tomlin, Shankar Sastry. (2024)  
**Hacking Predictors Means Hacking Cars: Using Sensitivity Analysis to Identify Trajectory Prediction Vulnerabilities for Autonomous Driving Security**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs-RO, cs-SY, cs.CR, eess-SY  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.10313v1)  

---


**ABSTRACT**  
Adversarial attacks on learning-based trajectory predictors have already been demonstrated. However, there are still open questions about the effects of perturbations on trajectory predictor inputs other than state histories, and how these attacks impact downstream planning and control. In this paper, we conduct a sensitivity analysis on two trajectory prediction models, Trajectron++ and AgentFormer. We observe that between all inputs, almost all of the perturbation sensitivities for Trajectron++ lie only within the most recent state history time point, while perturbation sensitivities for AgentFormer are spread across state histories over time. We additionally demonstrate that, despite dominant sensitivity on state history perturbations, an undetectable image map perturbation made with the Fast Gradient Sign Method can induce large prediction error increases in both models. Even though image maps may contribute slightly to the prediction output of both models, this result reveals that rather than being robust to adversarial image perturbations, trajectory predictors are susceptible to image attacks. Using an optimization-based planner and example perturbations crafted from sensitivity results, we show how this vulnerability can cause a vehicle to come to a sudden stop from moderate driving speeds.

{{</citation>}}


### (101/132) Eclectic Rule Extraction for Explainability of Deep Neural Network based Intrusion Detection Systems (Jesse Ables et al., 2024)

{{<citation>}}

Jesse Ables, Nathaniel Childers, William Anderson, Sudip Mittal, Shahram Rahimi, Ioana Banicescu, Maria Seale. (2024)  
**Eclectic Rule Extraction for Explainability of Deep Neural Network based Intrusion Detection Systems**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR  
Keywords: AI, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2401.10207v1)  

---


**ABSTRACT**  
This paper addresses trust issues created from the ubiquity of black box algorithms and surrogate explainers in Explainable Intrusion Detection Systems (X-IDS). While Explainable Artificial Intelligence (XAI) aims to enhance transparency, black box surrogate explainers, such as Local Interpretable Model-Agnostic Explanation (LIME) and SHapley Additive exPlanation (SHAP), are difficult to trust. The black box nature of these surrogate explainers makes the process behind explanation generation opaque and difficult to understand. To avoid this problem, one can use transparent white box algorithms such as Rule Extraction (RE). There are three types of RE algorithms: pedagogical, decompositional, and eclectic. Pedagogical methods offer fast but untrustworthy white-box explanations, while decompositional RE provides trustworthy explanations with poor scalability. This work explores eclectic rule extraction, which strikes a balance between scalability and trustworthiness. By combining techniques from pedagogical and decompositional approaches, eclectic rule extraction leverages the advantages of both, while mitigating some of their drawbacks. The proposed Hybrid X-IDS architecture features eclectic RE as a white box surrogate explainer for black box Deep Neural Networks (DNN). The presented eclectic RE algorithm extracts human-readable rules from hidden layers, facilitating explainable and trustworthy rulesets. Evaluations on UNSW-NB15 and CIC-IDS-2017 datasets demonstrate the algorithm's ability to generate rulesets with 99.9% accuracy, mimicking DNN outputs. The contributions of this work include the hybrid X-IDS architecture, the eclectic rule extraction algorithm applicable to intrusion detection datasets, and a thorough analysis of performance and explainability, demonstrating the trade-offs involved in rule extraction speed and accuracy.

{{</citation>}}


### (102/132) LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge (Shaswata Mitra et al., 2024)

{{<citation>}}

Shaswata Mitra, Subash Neupane, Trisha Chakraborty, Sudip Mittal, Aritran Piplai, Manas Gaur, Shahram Rahimi. (2024)  
**LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-IR, cs-LO, cs.CR  
Keywords: Language Model, Security  
[Paper Link](http://arxiv.org/abs/2401.10036v1)  

---


**ABSTRACT**  
Security Operations Center (SoC) analysts gather threat reports from openly accessible global threat databases and customize them manually to suit a particular organization's needs. These analysts also depend on internal repositories, which act as private local knowledge database for an organization. Credible cyber intelligence, critical operational details, and relevant organizational information are all stored in these local knowledge databases. Analysts undertake a labor intensive task utilizing these global and local knowledge databases to manually create organization's unique threat response and mitigation strategies. Recently, Large Language Models (LLMs) have shown the capability to efficiently process large diverse knowledge sources. We leverage this ability to process global and local knowledge databases to automate the generation of organization-specific threat intelligence.   In this work, we present LOCALINTEL, a novel automated knowledge contextualization system that, upon prompting, retrieves threat reports from the global threat repositories and uses its local knowledge database to contextualize them for a specific organization. LOCALINTEL comprises of three key phases: global threat intelligence retrieval, local knowledge retrieval, and contextualized completion generation. The former retrieves intelligence from global threat repositories, while the second retrieves pertinent knowledge from the local knowledge database. Finally, the fusion of these knowledge sources is orchestrated through a generator to produce a contextualized completion.

{{</citation>}}


### (103/132) Conning the Crypto Conman: End-to-End Analysis of Cryptocurrency-based Technical Support Scams (Bhupendra Acharya et al., 2024)

{{<citation>}}

Bhupendra Acharya, Muhammad Saad, Antonio Emanuele Cinà, Lea Schönherr, Hoang Dai Nguyen, Adam Oest, Phani Vadrevu, Thorsten Holz. (2024)  
**Conning the Crypto Conman: End-to-End Analysis of Cryptocurrency-based Technical Support Scams**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.09824v1)  

---


**ABSTRACT**  
The mainstream adoption of cryptocurrencies has led to a surge in wallet-related issues reported by ordinary users on social media platforms. In parallel, there is an increase in an emerging fraud trend called cryptocurrency-based technical support scam, in which fraudsters offer fake wallet recovery services and target users experiencing wallet-related issues.   In this paper, we perform a comprehensive study of cryptocurrency-based technical support scams. We present an analysis apparatus called HoneyTweet to analyze this kind of scam. Through HoneyTweet, we lure over 9K scammers by posting 25K fake wallet support tweets (so-called honey tweets). We then deploy automated systems to interact with scammers to analyze their modus operandi. In our experiments, we observe that scammers use Twitter as a starting point for the scam, after which they pivot to other communication channels (eg email, Instagram, or Telegram) to complete the fraud activity. We track scammers across those communication channels and bait them into revealing their payment methods. Based on the modes of payment, we uncover two categories of scammers that either request secret key phrase submissions from their victims or direct payments to their digital wallets. Furthermore, we obtain scam confirmation by deploying honey wallet addresses and validating private key theft. We also collaborate with the prominent payment service provider by sharing scammer data collections. The payment service provider feedback was consistent with our findings, thereby supporting our methodology and results. By consolidating our analysis across various vantage points, we provide an end-to-end scam lifecycle analysis and propose recommendations for scam mitigation.

{{</citation>}}


### (104/132) Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings (Mazal Bethany et al., 2024)

{{<citation>}}

Mazal Bethany, Athanasios Galiopoulos, Emet Bethany, Mohammad Bahrami Karkevandi, Nishant Vishwamitra, Peyman Najafirad. (2024)  
**Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09727v1)  

---


**ABSTRACT**  
The critical threat of phishing emails has been further exacerbated by the potential of LLMs to generate highly targeted, personalized, and automated spear phishing attacks. Two critical problems concerning LLM-facilitated phishing require further investigation: 1) Existing studies on lateral phishing lack specific examination of LLM integration for large-scale attacks targeting the entire organization, and 2) Current anti-phishing infrastructure, despite its extensive development, lacks the capability to prevent LLM-generated attacks, potentially impacting both employees and IT security incident management. However, the execution of such investigative studies necessitates a real-world environment, one that functions during regular business operations and mirrors the complexity of a large organizational infrastructure. This setting must also offer the flexibility required to facilitate a diverse array of experimental conditions, particularly the incorporation of phishing emails crafted by LLMs. This study is a pioneering exploration into the use of Large Language Models (LLMs) for the creation of targeted lateral phishing emails, targeting a large tier 1 university's operation and workforce of approximately 9,000 individuals over an 11-month period. It also evaluates the capability of email filtering infrastructure to detect such LLM-generated phishing attempts, providing insights into their effectiveness and identifying potential areas for improvement. Based on our findings, we propose machine learning-based detection techniques for such emails to detect LLM-generated phishing emails that were missed by the existing infrastructure, with an F1-score of 98.96.

{{</citation>}}


## cs.MA (1)



### (105/132) The Synergy Between Optimal Transport Theory and Multi-Agent Reinforcement Learning (Ali Baheri et al., 2024)

{{<citation>}}

Ali Baheri, and Mykel J. Kochenderfer. (2024)  
**The Synergy Between Optimal Transport Theory and Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.MA  
Categories: cs-LG, cs-MA, cs-SY, cs.MA, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.10949v1)  

---


**ABSTRACT**  
This paper explores the integration of optimal transport (OT) theory with multi-agent reinforcement learning (MARL). This integration uses OT to handle distributions and transportation problems to enhance the efficiency, coordination, and adaptability of MARL. There are five key areas where OT can impact MARL: (1) policy alignment, where OT's Wasserstein metric is used to align divergent agent strategies towards unified goals; (2) distributed resource management, employing OT to optimize resource allocation among agents; (3) addressing non-stationarity, using OT to adapt to dynamic environmental shifts; (4) scalable multi-agent learning, harnessing OT for decomposing large-scale learning objectives into manageable tasks; and (5) enhancing energy efficiency, applying OT principles to develop sustainable MARL systems. This paper articulates how the synergy between OT and MARL can address scalability issues, optimize resource distribution, align agent policies in cooperative environments, and ensure adaptability in dynamically changing conditions.

{{</citation>}}


## q-bio.NC (1)



### (106/132) Exploring General Intelligence via Gated Graph Transformer in Functional Connectivity Studies (Gang Qu et al., 2024)

{{<citation>}}

Gang Qu, Anton Orlichenko, Junqi Wang, Gemeng Zhang, Li Xiao, Aiying Zhang, Zhengming Ding, Yu-Ping Wang. (2024)  
**Exploring General Intelligence via Gated Graph Transformer in Functional Connectivity Studies**  

---
Primary Category: q-bio.NC  
Categories: cs-AI, q-bio-NC, q-bio.NC  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer  
[Paper Link](http://arxiv.org/abs/2401.10348v1)  

---


**ABSTRACT**  
Functional connectivity (FC) as derived from fMRI has emerged as a pivotal tool in elucidating the intricacies of various psychiatric disorders and delineating the neural pathways that underpin cognitive and behavioral dynamics inherent to the human brain. While Graph Neural Networks (GNNs) offer a structured approach to represent neuroimaging data, they are limited by their need for a predefined graph structure to depict associations between brain regions, a detail not solely provided by FCs. To bridge this gap, we introduce the Gated Graph Transformer (GGT) framework, designed to predict cognitive metrics based on FCs. Empirical validation on the Philadelphia Neurodevelopmental Cohort (PNC) underscores the superior predictive prowess of our model, further accentuating its potential in identifying pivotal neural connectivities that correlate with human cognitive processes.

{{</citation>}}


## stat.ME (1)



### (107/132) Lower Ricci Curvature for Efficient Community Detection (Yun Jin Park et al., 2024)

{{<citation>}}

Yun Jin Park, Didong Li. (2024)  
**Lower Ricci Curvature for Efficient Community Detection**  

---
Primary Category: stat.ME  
Categories: cs-SI, physics-soc-ph, stat-AP, stat-ME, stat.ME  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2401.10124v1)  

---


**ABSTRACT**  
This study introduces the Lower Ricci Curvature (LRC), a novel, scalable, and scale-free discrete curvature designed to enhance community detection in networks. Addressing the computational challenges posed by existing curvature-based methods, LRC offers a streamlined approach with linear computational complexity, making it well-suited for large-scale network analysis. We further develop an LRC-based preprocessing method that effectively augments popular community detection algorithms. Through comprehensive simulations and applications on real-world datasets, including the NCAA football league network, the DBLP collaboration network, the Amazon product co-purchasing network, and the YouTube social network, we demonstrate the efficacy of our method in significantly improving the performance of various community detection algorithms.

{{</citation>}}


## cs.AI (3)



### (108/132) Counterfactual Reasoning with Probabilistic Graphical Models for Analyzing Socioecological Systems (Rafael Cabañas et al., 2024)

{{<citation>}}

Rafael Cabañas, Ana D. Maldonado, María Morales, Pedro A. Aguilera, Antonio Salmerón. (2024)  
**Counterfactual Reasoning with Probabilistic Graphical Models for Analyzing Socioecological Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, math-PR, stat-AP  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.10101v1)  

---


**ABSTRACT**  
Causal and counterfactual reasoning are emerging directions in data science that allow us to reason about hypothetical scenarios. This is particularly useful in domains where experimental data are usually not available. In the context of environmental and ecological sciences, causality enables us, for example, to predict how an ecosystem would respond to hypothetical interventions. A structural causal model is a class of probabilistic graphical models for causality, which, due to its intuitive nature, can be easily understood by experts in multiple fields. However, certain queries, called unidentifiable, cannot be calculated in an exact and precise manner. This paper proposes applying a novel and recent technique for bounding unidentifiable queries within the domain of socioecological systems. Our findings indicate that traditional statistical analysis, including probabilistic graphical models, can identify the influence between variables. However, such methods do not offer insights into the nature of the relationship, specifically whether it involves necessity or sufficiency. This is where counterfactual reasoning becomes valuable.

{{</citation>}}


### (109/132) Towards Generative Abstract Reasoning: Completing Raven's Progressive Matrix via Rule Abstraction and Selection (Fan Shi et al., 2024)

{{<citation>}}

Fan Shi, Bin Li, Xiangyang Xue. (2024)  
**Towards Generative Abstract Reasoning: Completing Raven's Progressive Matrix via Rule Abstraction and Selection**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.09966v1)  

---


**ABSTRACT**  
Endowing machines with abstract reasoning ability has been a long-term research topic in artificial intelligence. Raven's Progressive Matrix (RPM) is widely used to probe abstract visual reasoning in machine intelligence, where models need to understand the underlying rules and select the missing bottom-right images out of candidate sets to complete image matrices. The participators can display powerful reasoning ability by inferring the underlying attribute-changing rules and imagining the missing images at arbitrary positions. However, existing solvers can hardly manifest such an ability in realistic RPM problems. In this paper, we propose a conditional generative model to solve answer generation problems through Rule AbstractIon and SElection (RAISE) in the latent space. RAISE encodes image attributes as latent concepts and decomposes underlying rules into atomic rules by means of concepts, which are abstracted as global learnable parameters. When generating the answer, RAISE selects proper atomic rules out of the global knowledge set for each concept and composes them into the integrated rule of an RPM. In most configurations, RAISE outperforms the compared generative solvers in tasks of generating bottom-right and arbitrary-position answers. We test RAISE in the odd-one-out task and two held-out configurations to demonstrate how learning decoupled latent concepts and atomic rules helps find the image breaking the underlying rules and handle RPMs with unseen combinations of rules and attributes.

{{</citation>}}


### (110/132) Tiny Multi-Agent DRL for Twins Migration in UAV Metaverses: A Multi-Leader Multi-Follower Stackelberg Game Approach (Jiawen Kang et al., 2024)

{{<citation>}}

Jiawen Kang, Yue Zhong, Minrui Xu, Jiangtian Nie, Jinbo Wen, Hongyang Du, Dongdong Ye, Xumin Huang, Dusit Niyato, Shengli Xie. (2024)  
**Tiny Multi-Agent DRL for Twins Migration in UAV Metaverses: A Multi-Leader Multi-Follower Stackelberg Game Approach**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-GT, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09680v1)  

---


**ABSTRACT**  
The synergy between Unmanned Aerial Vehicles (UAVs) and metaverses is giving rise to an emerging paradigm named UAV metaverses, which create a unified ecosystem that blends physical and virtual spaces, transforming drone interaction and virtual exploration. UAV Twins (UTs), as the digital twins of UAVs that revolutionize UAV applications by making them more immersive, realistic, and informative, are deployed and updated on ground base stations, e.g., RoadSide Units (RSUs), to offer metaverse services for UAV Metaverse Users (UMUs). Due to the dynamic mobility of UAVs and limited communication coverages of RSUs, it is essential to perform real-time UT migration to ensure seamless immersive experiences for UMUs. However, selecting appropriate RSUs and optimizing the required bandwidth is challenging for achieving reliable and efficient UT migration. To address the challenges, we propose a tiny machine learning-based Stackelberg game framework based on pruning techniques for efficient UT migration in UAV metaverses. Specifically, we formulate a multi-leader multi-follower Stackelberg model considering a new immersion metric of UMUs in the utilities of UAVs. Then, we design a Tiny Multi-Agent Deep Reinforcement Learning (Tiny MADRL) algorithm to obtain the tiny networks representing the optimal game solution. Specifically, the actor-critic network leverages the pruning techniques to reduce the number of network parameters and achieve model size and computation reduction, allowing for efficient implementation of Tiny MADRL. Numerical results demonstrate that our proposed schemes have better performance than traditional schemes.

{{</citation>}}


## cs.NE (3)



### (111/132) Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap (Xingyu Wu et al., 2024)

{{<citation>}}

Xingyu Wu, Sheng-hao Wu, Jibin Wu, Liang Feng, Kay Chen Tan. (2024)  
**Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-CL, cs-NE, cs.NE  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.10034v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), built upon Transformer-based architectures with massive pretraining on diverse data, have not only revolutionized natural language processing but also extended their prowess to various domains, marking a significant stride towards artificial general intelligence. The interplay between LLMs and Evolutionary Algorithms (EAs), despite differing in objectives and methodologies, reveals intriguing parallels, especially in their shared optimization nature, black-box characteristics, and proficiency in handling complex problems. Meanwhile, EA can not only provide an optimization framework for LLM's further enhancement under black-box settings but also empower LLM with flexible global search and iterative mechanism in applications. On the other hand, LLM's abundant domain knowledge enables EA to perform smarter searches, while its text processing capability assist in deploying EA across various tasks. Based on their complementary advantages, this paper presents a comprehensive review and forward-looking roadmap, categorizing their mutual inspiration into LLM-enhanced evolutionary optimization and EA-enhanced LLM. Some integrated synergy methods are further introduced to exemplify the amalgamation of LLMs and EAs in various application scenarios, including neural architecture search, code generation, software engineering, and text generation. As the first comprehensive review specifically focused on the EA research in the era of LLMs, this paper provides a foundational stepping stone for understanding and harnessing the collaborative potential of LLMs and EAs. By presenting a comprehensive review, categorization, and critical analysis, we contribute to the ongoing discourse on the cross-disciplinary study of these two powerful paradigms. The identified challenges and future directions offer guidance to unlock the full potential of this innovative collaboration.

{{</citation>}}


### (112/132) Evolutionary Multi-Objective Optimization of Large Language Model Prompts for Balancing Sentiments (Jill Baumann et al., 2024)

{{<citation>}}

Jill Baumann, Oliver Kramer. (2024)  
**Evolutionary Multi-Objective Optimization of Large Language Model Prompts for Balancing Sentiments**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-CL, cs-LG, cs-NE, cs.NE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09862v1)  

---


**ABSTRACT**  
The advent of large language models (LLMs) such as ChatGPT has attracted considerable attention in various domains due to their remarkable performance and versatility. As the use of these models continues to grow, the importance of effective prompt engineering has come to the fore. Prompt optimization emerges as a crucial challenge, as it has a direct impact on model performance and the extraction of relevant information. Recently, evolutionary algorithms (EAs) have shown promise in addressing this issue, paving the way for novel optimization strategies. In this work, we propose a evolutionary multi-objective (EMO) approach specifically tailored for prompt optimization called EMO-Prompts, using sentiment analysis as a case study. We use sentiment analysis capabilities as our experimental targets. Our results demonstrate that EMO-Prompts effectively generates prompts capable of guiding the LLM to produce texts embodying two conflicting emotions simultaneously.

{{</citation>}}


### (113/132) A Comparative Analysis on Metaheuristic Algorithms Based Vision Transformer Model for Early Detection of Alzheimer's Disease (Anuvab Sen et al., 2024)

{{<citation>}}

Anuvab Sen, Udayon Sen, Subhabrata Roy. (2024)  
**A Comparative Analysis on Metaheuristic Algorithms Based Vision Transformer Model for Early Detection of Alzheimer's Disease**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-NE, cs.NE  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.09795v1)  

---


**ABSTRACT**  
A number of life threatening neuro-degenerative disorders had degraded the quality of life for the older generation in particular. Dementia is one such symptom which may lead to a severe condition called Alzheimer's disease if not detected at an early stage. It has been reported that the progression of such disease from a normal stage is due to the change in several parameters inside the human brain. In this paper, an innovative metaheuristic algorithms based ViT model has been proposed for the identification of dementia at different stage. A sizeable number of test data have been utilized for the validation of the proposed scheme. It has also been demonstrated that our model exhibits superior performance in terms of accuracy, precision, recall as well as F1-score.

{{</citation>}}


## cs.RO (3)



### (114/132) A-KIT: Adaptive Kalman-Informed Transformer (Nadav Cohen et al., 2024)

{{<citation>}}

Nadav Cohen, Itzik Klein. (2024)  
**A-KIT: Adaptive Kalman-Informed Transformer**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.09987v1)  

---


**ABSTRACT**  
The extended Kalman filter (EKF) is a widely adopted method for sensor fusion in navigation applications. A crucial aspect of the EKF is the online determination of the process noise covariance matrix reflecting the model uncertainty. While common EKF implementation assumes a constant process noise, in real-world scenarios, the process noise varies, leading to inaccuracies in the estimated state and potentially causing the filter to diverge. To cope with such situations, model-based adaptive EKF methods were proposed and demonstrated performance improvements, highlighting the need for a robust adaptive approach. In this paper, we derive and introduce A-KIT, an adaptive Kalman-informed transformer to learn the varying process noise covariance online. The A-KIT framework is applicable to any type of sensor fusion. Here, we present our approach to nonlinear sensor fusion based on an inertial navigation system and Doppler velocity log. By employing real recorded data from an autonomous underwater vehicle, we show that A-KIT outperforms the conventional EKF by more than 49.5% and model-based adaptive EKF by an average of 35.4% in terms of position accuracy.

{{</citation>}}


### (115/132) Robotic Test Tube Rearrangement Using Combined Reinforcement Learning and Motion Planning (Hao Chen et al., 2024)

{{<citation>}}

Hao Chen, Weiwei Wan, Masaki Matsushita, Takeyuki Kotaka, Kensuke Harada. (2024)  
**Robotic Test Tube Rearrangement Using Combined Reinforcement Learning and Motion Planning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09772v1)  

---


**ABSTRACT**  
A combined task-level reinforcement learning and motion planning framework is proposed in this paper to address a multi-class in-rack test tube rearrangement problem. At the task level, the framework uses reinforcement learning to infer a sequence of swap actions while ignoring robotic motion details. At the motion level, the framework accepts the swapping action sequences inferred by task-level agents and plans the detailed robotic pick-and-place motion. The task and motion-level planning form a closed loop with the help of a condition set maintained for each rack slot, which allows the framework to perform replanning and effectively find solutions in the presence of low-level failures. Particularly for reinforcement learning, the framework leverages a distributed deep Q-learning structure with the Dueling Double Deep Q Network (D3QN) to acquire near-optimal policies and uses an A${}^\star$-based post-processing technique to amplify the collected training data. The D3QN and distributed learning help increase training efficiency. The post-processing helps complete unfinished action sequences and remove redundancy, thus making the training data more effective. We carry out both simulations and real-world studies to understand the performance of the proposed framework. The results verify the performance of the RL and post-processing and show that the closed-loop combination improves robustness. The framework is ready to incorporate various sensory feedback. The real-world studies also demonstrated the incorporation.

{{</citation>}}


### (116/132) Learning Hybrid Policies for MPC with Application to Drone Flight in Unknown Dynamic Environments (Zhaohan Feng et al., 2024)

{{<citation>}}

Zhaohan Feng, Jie Chen, Wei Xiao, Jian Sun, Bin Xin, Gang Wang. (2024)  
**Learning Hybrid Policies for MPC with Application to Drone Flight in Unknown Dynamic Environments**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2401.09705v1)  

---


**ABSTRACT**  
In recent years, drones have found increased applications in a wide array of real-world tasks. Model predictive control (MPC) has emerged as a practical method for drone flight control, owing to its robustness against modeling errors/uncertainties and external disturbances. However, MPC's sensitivity to manually tuned parameters can lead to rapid performance degradation when faced with unknown environmental dynamics. This paper addresses the challenge of controlling a drone as it traverses a swinging gate characterized by unknown dynamics. This paper introduces a parameterized MPC approach named hyMPC that leverages high-level decision variables to adapt to uncertain environmental conditions. To derive these decision variables, a novel policy search framework aimed at training a high-level Gaussian policy is presented. Subsequently, we harness the power of neural network policies, trained on data gathered through the repeated execution of the Gaussian policy, to provide real-time decision variables. The effectiveness of hyMPC is validated through numerical simulations, achieving a 100\% success rate in 20 drone flight tests traversing a swinging gate, demonstrating its capability to achieve safe and precise flight with limited prior knowledge of environmental dynamics.

{{</citation>}}


## cs.DC (3)



### (117/132) Deep Back-Filling: a Split Window Technique for Deep Online Cluster Job Scheduling (Lingfei Wang et al., 2024)

{{<citation>}}

Lingfei Wang, Aaron Harwood, Maria A. Rodriguez. (2024)  
**Deep Back-Filling: a Split Window Technique for Deep Online Cluster Job Scheduling**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09910v1)  

---


**ABSTRACT**  
Job scheduling is a critical component of workload management systems that can significantly influence system performance, e.g., in HPC clusters. The scheduling objectives are often mixed, such as maximizing resource utilization and minimizing job waiting time. An increasing number of researchers are moving from heuristic-based approaches to Deep Reinforcement Learning approaches in order to optimize scheduling objectives. However, the job scheduler's state space is partially observable to a DRL-based agent because the job queue is practically unbounded. The agent's observation of the state space is constant in size since the input size of the neural networks is predefined. All existing solutions to this problem intuitively allow the agent to observe a fixed window size of jobs at the head of the job queue. In our research, we have seen that such an approach can lead to "window staleness" where the window becomes full of jobs that can not be scheduled until the cluster has completed sufficient work. In this paper, we propose a novel general technique that we call \emph{split window}, which allows the agent to observe both the head \emph{and tail} of the queue. With this technique, the agent can observe all arriving jobs at least once, which completely eliminates the window staleness problem. By leveraging the split window, the agent can significantly reduce the average job waiting time and average queue length, alternatively allowing the use of much smaller windows and, therefore, faster training times. We show a range of simulation results using HPC job scheduling trace data that supports the effectiveness of our technique.

{{</citation>}}


### (118/132) A HPC Co-Scheduler with Reinforcement Learning (Abel Souza et al., 2024)

{{<citation>}}

Abel Souza, Kristiaan Pelckmans, Johan Tordsson. (2024)  
**A HPC Co-Scheduler with Reinforcement Learning**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09706v1)  

---


**ABSTRACT**  
Although High Performance Computing (HPC) users understand basic resource requirements such as the number of CPUs and memory limits, internal infrastructural utilization data is exclusively leveraged by cluster operators, who use it to configure batch schedulers. This task is challenging and increasingly complex due to ever larger cluster scales and heterogeneity of modern scientific workflows. As a result, HPC systems achieve low utilization with long job completion times (makespans). To tackle these challenges, we propose a co-scheduling algorithm based on an adaptive reinforcement learning algorithm, where application profiling is combined with cluster monitoring. The resulting cluster scheduler matches resource utilization to application performance in a fine-grained manner (i.e., operating system level). As opposed to nominal allocations, we apply decision trees to model applications' actual resource usage, which are used to estimate how much resource capacity from one allocation can be co-allocated to additional applications. Our algorithm learns from incorrect co-scheduling decisions and adapts from changing environment conditions, and evaluates when such changes cause resource contention that impacts quality of service metrics such as jobs slowdowns. We integrate our algorithm in an HPC resource manager that combines Slurm and Mesos for job scheduling and co-allocation, respectively. Our experimental evaluation performed in a dedicated cluster executing a mix of four real different scientific workflows demonstrates improvements on cluster utilization of up to 51% even in high load scenarios, with 55% average queue makespan reductions under low loads.

{{</citation>}}


### (119/132) DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving (Yinmin Zhong et al., 2024)

{{<citation>}}

Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang. (2024)  
**DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09670v1)  

---


**ABSTRACT**  
DistServe improves the performance of large language models (LLMs) serving by disaggregating the prefill and decoding computation. Existing LLM serving systems colocate the two phases and batch the computation of prefill and decoding across all users and requests. We find that this strategy not only leads to strong prefill-decoding interferences but also couples the resource allocation and parallelism plans for both phases. LLM applications often emphasize individual latency for each phase: time to first token (TTFT) for the prefill phase and time per output token (TPOT) of each request for the decoding phase. In the presence of stringent latency requirements, existing systems have to prioritize one latency over the other, or over-provision compute resources to meet both.   DistServe assigns prefill and decoding computation to different GPUs, hence eliminating prefill-decoding interferences. Given the application's TTFT and TPOT requirements, DistServe co-optimizes the resource allocation and parallelism strategy tailored for each phase. DistServe also places the two phases according to the serving cluster's bandwidth to minimize the communication caused by disaggregation. As a result, DistServe significantly improves LLM serving performance in terms of the maximum rate that can be served within both TTFT and TPOT constraints on each GPU. Our evaluations show that on various popular LLMs, applications, and latency requirements, DistServe can serve 4.48x more requests or 10.2x tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for > 90% of requests.

{{</citation>}}


## cs.SD (2)



### (120/132) Attention-Based Recurrent Neural Network For Automatic Behavior Laying Hen Recognition (Fréjus A. A. Laleye et al., 2024)

{{<citation>}}

Fréjus A. A. Laleye, Mikaël A. Mousse. (2024)  
**Attention-Based Recurrent Neural Network For Automatic Behavior Laying Hen Recognition**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CL, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.09880v1)  

---


**ABSTRACT**  
One of the interests of modern poultry farming is the vocalization of laying hens which contain very useful information on health behavior. This information is used as health and well-being indicators that help breeders better monitor laying hens, which involves early detection of problems for rapid and more effective intervention. In this work, we focus on the sound analysis for the recognition of the types of calls of the laying hens in order to propose a robust system of characterization of their behavior for a better monitoring. To do this, we first collected and annotated laying hen call signals, then designed an optimal acoustic characterization based on the combination of time and frequency domain features. We then used these features to build the multi-label classification models based on recurrent neural network to assign a semantic class to the vocalization that characterize the laying hen behavior. The results show an overall performance with our model based on the combination of time and frequency domain features that obtained the highest F1-score (F1=92.75) with a gain of 17% on the models using the frequency domain features and of 8% on the compared approaches from the litterature.

{{</citation>}}


### (121/132) Improving Speaker-independent Speech Emotion Recognition Using Dynamic Joint Distribution Adaptation (Cheng Lu et al., 2024)

{{<citation>}}

Cheng Lu, Yuan Zong, Hailun Lian, Yan Zhao, Björn Schuller, Wenming Zheng. (2024)  
**Improving Speaker-independent Speech Emotion Recognition Using Dynamic Joint Distribution Adaptation**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2401.09752v1)  

---


**ABSTRACT**  
In speaker-independent speech emotion recognition, the training and testing samples are collected from diverse speakers, leading to a multi-domain shift challenge across the feature distributions of data from different speakers. Consequently, when the trained model is confronted with data from new speakers, its performance tends to degrade. To address the issue, we propose a Dynamic Joint Distribution Adaptation (DJDA) method under the framework of multi-source domain adaptation. DJDA firstly utilizes joint distribution adaptation (JDA), involving marginal distribution adaptation (MDA) and conditional distribution adaptation (CDA), to more precisely measure the multi-domain distribution shifts caused by different speakers. This helps eliminate speaker bias in emotion features, allowing for learning discriminative and speaker-invariant speech emotion features from coarse-level to fine-level. Furthermore, we quantify the adaptation contributions of MDA and CDA within JDA by using a dynamic balance factor based on $\mathcal{A}$-Distance, promoting to effectively handle the unknown distributions encountered in data from new speakers. Experimental results demonstrate the superior performance of our DJDA as compared to other state-of-the-art (SOTA) methods.

{{</citation>}}


## cs.ET (1)



### (122/132) Improving the Accuracy of Analog-Based In-Memory Computing Accelerators Post-Training (Corey Lammie et al., 2024)

{{<citation>}}

Corey Lammie, Athanasios Vasilopoulos, Julian Büchel, Giacomo Camposampiero, Manuel Le Gallo, Malte Rasch, Abu Sebastian. (2024)  
**Improving the Accuracy of Analog-Based In-Memory Computing Accelerators Post-Training**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs.ET  
Keywords: AI, BERT, GLUE  
[Paper Link](http://arxiv.org/abs/2401.09859v1)  

---


**ABSTRACT**  
Analog-Based In-Memory Computing (AIMC) inference accelerators can be used to efficiently execute Deep Neural Network (DNN) inference workloads. However, to mitigate accuracy losses, due to circuit and device non-idealities, Hardware-Aware (HWA) training methodologies must be employed. These typically require significant information about the underlying hardware. In this paper, we propose two Post-Training (PT) optimization methods to improve accuracy after training is performed. For each crossbar, the first optimizes the conductance range of each column, and the second optimizes the input, i.e, Digital-to-Analog Converter (DAC), range. It is demonstrated that, when these methods are employed, the complexity during training, and the amount of information about the underlying hardware can be reduced, with no notable change in accuracy ($\leq$0.1%) when finetuning the pretrained RoBERTa transformer model for all General Language Understanding Evaluation (GLUE) benchmark tasks. Additionally, it is demonstrated that further optimizing learned parameters PT improves accuracy.

{{</citation>}}


## cs.SI (2)



### (123/132) Disentangled Condensation for Large-scale Graphs (Zhenbang Xiao et al., 2024)

{{<citation>}}

Zhenbang Xiao, Shunyu Liu, Yu Wang, Tongya Zheng, Mingli Song. (2024)  
**Disentangled Condensation for Large-scale Graphs**  

---
Primary Category: cs.SI  
Categories: cs-LG, cs-SI, cs.SI  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.12231v1)  

---


**ABSTRACT**  
Graph condensation has emerged as an intriguing technique to provide Graph Neural Networks for large-scale graphs with a more compact yet informative small graph to save the expensive costs of large-scale graph learning. Despite the promising results achieved, previous graph condensation methods often employ an entangled condensation strategy that involves condensing nodes and edges simultaneously, leading to substantial GPU memory demands. This entangled strategy has considerably impeded the scalability of graph condensation, impairing its capability to condense extremely large-scale graphs and produce condensed graphs with high fidelity. Therefore, this paper presents Disentangled Condensation for large-scale graphs, abbreviated as DisCo, to provide scalable graph condensation for graphs of varying sizes. At the heart of DisCo are two complementary components, namely node and edge condensation modules, that realize the condensation of nodes and edges in a disentangled manner. In the node condensation module, we focus on synthesizing condensed nodes that exhibit a similar node feature distribution to original nodes using a pre-trained node classification model while incorporating class centroid alignment and anchor attachment regularizers. After node condensation, in the edge condensation module, we preserve the topology structure by transferring the link prediction model of the original graph to the condensed nodes, generating the corresponding condensed edges. Based on the disentangled strategy, the proposed DisCo can successfully scale up to the ogbn-papers100M graph with over 100 million nodes and 1 billion edges with flexible reduction rates. Extensive experiments on five common datasets further demonstrate that the proposed DisCo yields results superior to state-of-the-art counterparts by a significant margin. The source code is available at https://github.com/BangHonor/DisCo.

{{</citation>}}


### (124/132) Towards Learning from Graphs with Heterophily: Progress and Future (Chenghua Gong et al., 2024)

{{<citation>}}

Chenghua Gong, Yao Cheng, Xiang Li, Caihua Shan, Siqiang Luo, Chuan Shi. (2024)  
**Towards Learning from Graphs with Heterophily: Progress and Future**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.09769v1)  

---


**ABSTRACT**  
Graphs are structured data that models complex relations between real-world entities. Heterophilous graphs, where linked nodes are prone to be with different labels or dissimilar features, have recently attracted significant attention and found many applications. Meanwhile, increasing efforts have been made to advance learning from heterophilous graphs. Although there exist surveys on the relevant topic, they focus on heterophilous GNNs, which are only sub-topics of heterophilous graph learning. In this survey, we comprehensively overview existing works on learning from graphs with heterophily.First, we collect over 180 publications and introduce the development of this field. Then, we systematically categorize existing methods based on a hierarchical taxonomy including learning strategies, model architectures and practical applications. Finally, we discuss the primary challenges of existing studies and highlight promising avenues for future research.More publication details and corresponding open-source codes can be accessed and will be continuously updated at our repositories:https://github.com/gongchenghua/Awesome-Survey-Graphs-with-Heterophily.

{{</citation>}}


## cs.GT (1)



### (125/132) Clickbait vs. Quality: How Engagement-Based Optimization Shapes the Content Landscape in Online Platforms (Nicole Immorlica et al., 2024)

{{<citation>}}

Nicole Immorlica, Meena Jagadeesan, Brendan Lucier. (2024)  
**Clickbait vs. Quality: How Engagement-Based Optimization Shapes the Content Landscape in Online Platforms**  

---
Primary Category: cs.GT  
Categories: cs-CY, cs-GT, cs-LG, cs.GT  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.09804v1)  

---


**ABSTRACT**  
Online content platforms commonly use engagement-based optimization when making recommendations. This encourages content creators to invest in quality, but also rewards gaming tricks such as clickbait. To understand the total impact on the content landscape, we study a game between content creators competing on the basis of engagement metrics and analyze the equilibrium decisions about investment in quality and gaming. First, we show the content created at equilibrium exhibits a positive correlation between quality and gaming, and we empirically validate this finding on a Twitter dataset. Using the equilibrium structure of the content landscape, we then examine the downstream performance of engagement-based optimization along several axes. Perhaps counterintuitively, the average quality of content consumed by users can decrease at equilibrium as gaming tricks become more costly for content creators to employ. Moreover, engagement-based optimization can perform worse in terms of user utility than a baseline with random recommendations, and engagement-based optimization is also suboptimal in terms of realized engagement relative to quality-based optimization. Altogether, our results highlight the need to consider content creator incentives when evaluating a platform's choice of optimization metric.

{{</citation>}}


## eess.SY (2)



### (126/132) Power System Fault Diagnosis with Quantum Computing and Efficient Gate Decomposition (Xiang Fei et al., 2024)

{{<citation>}}

Xiang Fei, Huan Zhao, Xiyuan Zhou, Junhua Zhao, Ting Shu, Fushuan Wen. (2024)  
**Power System Fault Diagnosis with Quantum Computing and Efficient Gate Decomposition**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.09800v1)  

---


**ABSTRACT**  
Power system fault diagnosis is crucial for identifying the location and causes of faults and providing decision-making support for power dispatchers. However, most classical methods suffer from significant time-consuming, memory overhead, and computational complexity issues as the scale of the power system concerned increases. With rapid development of quantum computing technology, the combinatorial optimization method based on quantum computing has shown certain advantages in computational time over existing methods. Given this background, this paper proposes a quantum computing based power system fault diagnosis method with the Quantum Approximate Optimization Algorithm (QAOA). The proposed method reformulates the fault diagnosis problem as a Hamiltonian by using Ising model, which completely preserves the coupling relationship between faulty components and various operations of protective relays and circuit breakers. Additionally, to enhance problem-solving efficiency under current equipment limitations, the symmetric equivalent decomposition method of multi-z-rotation gate is proposed. Furthermore, the small probability characteristics of power system events is utilized to reduce the number of qubits. Simulation results based on the test system show that the proposed methods can achieve the same optimal results with a faster speed compared with the classical higher-order solver provided by D-Wave.

{{</citation>}}


### (127/132) Traffic Smoothing Controllers for Autonomous Vehicles Using Deep Reinforcement Learning and Real-World Trajectory Data (Nathan Lichtlé et al., 2024)

{{<citation>}}

Nathan Lichtlé, Kathy Jang, Adit Shah, Eugene Vinitsky, Jonathan W. Lee, Alexandre M. Bayen. (2024)  
**Traffic Smoothing Controllers for Autonomous Vehicles Using Deep Reinforcement Learning and Real-World Trajectory Data**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-MA, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09666v1)  

---


**ABSTRACT**  
Designing traffic-smoothing cruise controllers that can be deployed onto autonomous vehicles is a key step towards improving traffic flow, reducing congestion, and enhancing fuel efficiency in mixed autonomy traffic. We bypass the common issue of having to carefully fine-tune a large traffic microsimulator by leveraging real-world trajectory data from the I-24 highway in Tennessee, replayed in a one-lane simulation. Using standard deep reinforcement learning methods, we train energy-reducing wave-smoothing policies. As an input to the agent, we observe the speed and distance of only the vehicle in front, which are local states readily available on most recent vehicles, as well as non-local observations about the downstream state of the traffic. We show that at a low 4% autonomous vehicle penetration rate, we achieve significant fuel savings of over 15% on trajectories exhibiting many stop-and-go waves. Finally, we analyze the smoothing effect of the controllers and demonstrate robustness to adding lane-changing into the simulation as well as the removal of downstream information.

{{</citation>}}


## cs.MM (1)



### (128/132) On the Audio Hallucinations in Large Audio-Video Language Models (Taichi Nishimura et al., 2024)

{{<citation>}}

Taichi Nishimura, Shota Nakada, Masayoshi Kondo. (2024)  
**On the Audio Hallucinations in Large Audio-Video Language Models**  

---
Primary Category: cs.MM  
Categories: cs-CL, cs-CV, cs-MM, cs-SD, cs.MM, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09774v1)  

---


**ABSTRACT**  
Large audio-video language models can generate descriptions for both video and audio. However, they sometimes ignore audio content, producing audio descriptions solely reliant on visual information. This paper refers to this as audio hallucinations and analyzes them in large audio-video language models. We gather 1,000 sentences by inquiring about audio information and annotate them whether they contain hallucinations. If a sentence is hallucinated, we also categorize the type of hallucination. The results reveal that 332 sentences are hallucinated with distinct trends observed in nouns and verbs for each hallucination type. Based on this, we tackle a task of audio hallucination classification using pre-trained audio-text models in the zero-shot and fine-tuning settings. Our experimental results reveal that the zero-shot models achieve higher performance (52.2% in F1) than the random (40.3%) and the fine-tuning models achieve 87.9%, outperforming the zero-shot models.

{{</citation>}}


## cs.SC (1)



### (129/132) Bootstrapping OTS-Funcimg Pre-training Model (Botfip) -- A Comprehensive Symbolic Regression Framework (Tianhao Chen et al., 2024)

{{<citation>}}

Tianhao Chen, Pengbo Xu, Haibiao Zheng. (2024)  
**Bootstrapping OTS-Funcimg Pre-training Model (Botfip) -- A Comprehensive Symbolic Regression Framework**  

---
Primary Category: cs.SC  
Categories: cs-AI, cs-LG, cs-SC, cs.SC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09748v1)  

---


**ABSTRACT**  
In the field of scientific computing, many problem-solving approaches tend to focus only on the process and final outcome, even in AI for science, there is a lack of deep multimodal information mining behind the data, missing a multimodal framework akin to that in the image-text domain. In this paper, we take Symbolic Regression(SR) as our focal point and, drawing inspiration from the BLIP model in the image-text domain, propose a scientific computing multimodal framework based on Function Images (Funcimg) and Operation Tree Sequence (OTS), named Bootstrapping OTS-Funcimg Pre-training Model (Botfip). In SR experiments, we validate the advantages of Botfip in low-complexity SR problems, showcasing its potential. As a MED framework, Botfip holds promise for future applications in a broader range of scientific computing problems.

{{</citation>}}


## math.NA (1)



### (130/132) Fast Updating Truncated SVD for Representation Learning with Sparse Matrices (Haoran Deng et al., 2024)

{{<citation>}}

Haoran Deng, Yang Yang, Jiahe Li, Cheng Chen, Weihao Jiang, Shiliang Pu. (2024)  
**Fast Updating Truncated SVD for Representation Learning with Sparse Matrices**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.09703v1)  

---


**ABSTRACT**  
Updating a truncated Singular Value Decomposition (SVD) is crucial in representation learning, especially when dealing with large-scale data matrices that continuously evolve in practical scenarios. Aligning SVD-based models with fast-paced updates becomes increasingly important. Existing methods for updating truncated SVDs employ Rayleigh-Ritz projection procedures, where projection matrices are augmented based on original singular vectors. However, these methods suffer from inefficiency due to the densification of the update matrix and the application of the projection to all singular vectors. To address these limitations, we introduce a novel method for dynamically approximating the truncated SVD of a sparse and temporally evolving matrix. Our approach leverages sparsity in the orthogonalization process of augmented matrices and utilizes an extended decomposition to independently store projections in the column space of singular vectors. Numerical experiments demonstrate a remarkable efficiency improvement of an order of magnitude compared to previous methods. Remarkably, this improvement is achieved while maintaining a comparable precision to existing approaches.

{{</citation>}}


## cs.HC (1)



### (131/132) Should ChatGPT Write Your Breakup Text? Exploring the Role of AI in Relationship Dissolution (Yue Fu et al., 2024)

{{<citation>}}

Yue Fu, Yixin Chen, Zelia Gomes Da Costa Lai, Alexis Hiniker. (2024)  
**Should ChatGPT Write Your Breakup Text? Exploring the Role of AI in Relationship Dissolution**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.09695v1)  

---


**ABSTRACT**  
Relationships are essential to our happiness and wellbeing. The dissolution of a relationship, the final stage of relationship's lifecycle and one of the most stressful events in an individual's life, can have profound and long-lasting impacts on people. With the breakup process increasingly facilitated by computer-mediated communication (CMC), and the likely future influence of AI-mediated communication (AIMC) tools, we conducted a semi-structured interview study with 21 participants. We aim to understand: 1) the current role of technology in the breakup process, 2) the needs and support individuals have during the process, and 3) how AI might address these needs. Our research shows that people have distinct needs at various stages of ending a relationship. Presently, technology is used for information gathering and community support, acting as a catalyst for breakups, enabling ghosting and blocking, and facilitating communication. Participants anticipate that AI could aid in sense-making of their relationship leading up to the breakup, act as a mediator, assist in crafting appropriate wording, tones, and language during breakup conversations, and support companionship, reflection, recovery, and growth after a breakup. Our findings also demonstrate an overlap between the breakup process and the Transtheoretical Model (TTM) of behavior change. Through the lens of TTM, we explore the potential support and affordances AI could offer in breakups, including its benefits and the necessary precautions regarding AI's role in this sensitive process.

{{</citation>}}


## astro-ph.IM (1)



### (132/132) Decades of Transformation: Evolution of the NASA Astrophysics Data System's Infrastructure (Alberto Accomazzi, 2024)

{{<citation>}}

Alberto Accomazzi. (2024)  
**Decades of Transformation: Evolution of the NASA Astrophysics Data System's Infrastructure**  

---
Primary Category: astro-ph.IM  
Categories: astro-ph-IM, astro-ph.IM, cs-DL  
Keywords: AI, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.09685v1)  

---


**ABSTRACT**  
The NASA Astrophysics Data System (ADS) is the primary Digital Library portal for researchers in astronomy and astrophysics. Over the past 30 years, the ADS has gone from being an astronomy-focused bibliographic database to an open digital library system supporting research in space and (soon) earth sciences. This paper describes the evolution of the ADS system, its capabilities, and the technological infrastructure underpinning it.   We give an overview of the ADS's original architecture, constructed primarily around simple database models. This bespoke system allowed for the efficient indexing of metadata and citations, the digitization and archival of full-text articles, and the rapid development of discipline-specific capabilities running on commodity hardware. The move towards a cloud-based microservices architecture and an open-source search engine in the late 2010s marked a significant shift, bringing full-text search capabilities, a modern API, higher uptime, more reliable data retrieval, and integration of advanced visualizations and analytics.   Another crucial evolution came with the gradual and ongoing incorporation of Machine Learning and Natural Language Processing algorithms in our data pipelines. Originally used for information extraction and classification tasks, NLP and ML techniques are now being developed to improve metadata enrichment, search, notifications, and recommendations. we describe how these computational techniques are being embedded into our software infrastructure, the challenges faced, and the benefits reaped.   Finally, we conclude by describing the future prospects of ADS and its ongoing expansion, discussing the challenges of managing an interdisciplinary information system in the era of AI and Open Science, where information is abundant, technology is transformative, but their trustworthiness can be elusive.

{{</citation>}}
