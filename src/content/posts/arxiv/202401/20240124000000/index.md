---
draft: false
title: "arXiv @ 2024.01.24"
date: 2024-01-24
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.24"
    identifier: arxiv_20240124
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (28)](#cscl-28)
- [cs.SE (3)](#csse-3)
- [cs.DC (2)](#csdc-2)
- [cs.AI (2)](#csai-2)
- [stat.ML (3)](#statml-3)
- [cs.CR (9)](#cscr-9)
- [cs.IT (1)](#csit-1)
- [cs.CV (27)](#cscv-27)
- [cs.GT (1)](#csgt-1)
- [cs.RO (4)](#csro-4)
- [cs.LG (17)](#cslg-17)
- [physics.bio-ph (1)](#physicsbio-ph-1)
- [cs.MA (2)](#csma-2)
- [cs.HC (4)](#cshc-4)
- [cs.CY (2)](#cscy-2)
- [eess.AS (2)](#eessas-2)
- [eess.IV (3)](#eessiv-3)
- [cs.NE (2)](#csne-2)
- [cs.SC (1)](#cssc-1)
- [cs.AR (1)](#csar-1)
- [cs.MM (1)](#csmm-1)
- [cs.IR (2)](#csir-2)
- [physics.ao-ph (1)](#physicsao-ph-1)

## cs.CL (28)



### (1/119) How Far Can 100 Samples Go? Unlocking Overall Zero-Shot Multilingual Translation via Tiny Multi-Parallel Data (Di Wu et al., 2024)

{{<citation>}}

Di Wu, Shaomu Tan, Yan Meng, David Stap, Christof Monz. (2024)  
**How Far Can 100 Samples Go? Unlocking Overall Zero-Shot Multilingual Translation via Tiny Multi-Parallel Data**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Machine Translation, Multilingual, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.12413v1)  

---


**ABSTRACT**  
Zero-shot translation is an open problem, aiming to translate between language pairs unseen during training in Multilingual Machine Translation (MMT). A common, albeit resource-consuming, solution is to mine as many translation directions as possible to add to the parallel corpus. In this paper, we show that the zero-shot capability of an English-centric model can be easily enhanced by fine-tuning with a very small amount of multi-parallel data. For example, on the EC30 dataset, we show that up to +21.7 ChrF non-English overall improvements (870 directions) can be achieved by using only 100 multi-parallel samples, meanwhile preserving capability in English-centric directions. We further study the size effect of fine-tuning data and its transfer capabilities. Surprisingly, our empirical analysis shows that comparable overall improvements can be achieved even through fine-tuning in a small, randomly sampled direction set (10\%). Also, the resulting non-English performance is quite close to the upper bound (complete translation). Due to its high efficiency and practicality, we encourage the community 1) to consider the use of the fine-tuning method as a strong baseline for zero-shot translation and 2) to construct more comprehensive and high-quality multi-parallel data to cover real-world demand.

{{</citation>}}


### (2/119) Enhancing In-context Learning via Linear Probe Calibration (Momin Abbas et al., 2024)

{{<citation>}}

Momin Abbas, Yi Zhou, Parikshit Ram, Nathalie Baracaldo, Horst Samulowitz, Theodoros Salonidis, Tianyi Chen. (2024)  
**Enhancing In-context Learning via Linear Probe Calibration**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2401.12406v1)  

---


**ABSTRACT**  
In-context learning (ICL) is a new paradigm for natural language processing that utilizes Generative Pre-trained Transformer (GPT)-like models. This approach uses prompts that include in-context demonstrations to generate the corresponding output for a new query input. However, applying ICL in real cases does not scale with the number of samples, and lacks robustness to different prompt templates and demonstration permutations. In this paper, we first show that GPT-like models using ICL result in unreliable predictions based on a new metric based on Shannon entropy. Then, to solve this problem, we propose a new technique called the Linear Probe Calibration (LinC), a method that calibrates the model's output probabilities, resulting in reliable predictions and improved performance, while requiring only minimal additional samples (as few as five labeled data samples). LinC significantly enhances the ICL test performance of GPT models on various benchmark datasets, with an average improvement of up to 21%, and up to a 50% improvement in some cases, and significantly boosts the performance of PEFT methods, especially in the low resource regime. Moreover, LinC achieves lower expected calibration error, and is highly robust to varying label proportions, prompt templates, and demonstration permutations. Our code is available at \url{https://github.com/mominabbass/LinC}.

{{</citation>}}


### (3/119) Development of an NLP-driven computer-based test guide for visually impaired students (Tubo Faustinah Nemieboka et al., 2024)

{{<citation>}}

Tubo Faustinah Nemieboka, Ikechukwu E. Onyenwe, Doris C. Asogwa. (2024)  
**Development of an NLP-driven computer-based test guide for visually impaired students**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.12375v1)  

---


**ABSTRACT**  
In recent years, advancements in Natural Language Processing (NLP) techniques have revolutionized the field of accessibility and exclusivity of testing, particularly for visually impaired students (VIS). CBT has shown in years back its relevance in terms of administering exams electronically, making the test process easier, providing quicker and more accurate results, and offering greater flexibility and accessibility for candidates. Yet, its relevance was not felt by the visually impaired students as they cannot access printed documents. Hence, in this paper, we present an NLP-driven Computer-Based Test guide for visually impaired students. It employs a speech technology pre-trained methods to provide real-time assistance and support to visually impaired students. The system utilizes NLP technologies to convert the text-based questions and the associated options in a machine-readable format. Subsequently, the speech technology pre-trained model processes the converted text enabling the VIS to comprehend and analyze the content. Furthermore, we validated that this pre-trained model is not perverse by testing for accuracy using sample audio datasets labels (A, B, C, D, E, F, G) to compare with the voice recordings obtained from 20 VIS which is been predicted by the system to attain values for precision, recall, and F1-scores. These metrics are used to assess the performance of the pre-trained model and have indicated that it is proficient enough to give its better performance to the evaluated system. The methodology adopted for this system is Object Oriented Analysis and Design Methodology (OOADM) where Objects are discussed and built by modeling real-world instances.

{{</citation>}}


### (4/119) Fine-tuning Large Language Models for Multigenerator, Multidomain, and Multilingual Machine-Generated Text Detection (Feng Xiong et al., 2024)

{{<citation>}}

Feng Xiong, Thanet Markchom, Ziwei Zheng, Subin Jung, Varun Ojha, Huizhi Liang. (2024)  
**Fine-tuning Large Language Models for Multigenerator, Multidomain, and Multilingual Machine-Generated Text Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model, Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2401.12326v1)  

---


**ABSTRACT**  
SemEval-2024 Task 8 introduces the challenge of identifying machine-generated texts from diverse Large Language Models (LLMs) in various languages and domains. The task comprises three subtasks: binary classification in monolingual and multilingual (Subtask A), multi-class classification (Subtask B), and mixed text detection (Subtask C). This paper focuses on Subtask A & B. Each subtask is supported by three datasets for training, development, and testing. To tackle this task, two methods: 1) using traditional machine learning (ML) with natural language preprocessing (NLP) for feature extraction, and 2) fine-tuning LLMs for text classification. The results show that transformer models, particularly LoRA-RoBERTa, exceed traditional ML methods in effectiveness, with majority voting being particularly effective in multilingual contexts for identifying machine-generated texts.

{{</citation>}}


### (5/119) Cheap Learning: Maximising Performance of Language Models for Social Data Science Using Minimal Data (Leonardo Castro-Gonzalez et al., 2024)

{{<citation>}}

Leonardo Castro-Gonzalez, Yi-Ling Chung, Hannak Rose Kirk, John Francis, Angus R. Williams, Pica Johansson, Jonathan Bright. (2024)  
**Cheap Learning: Maximising Performance of Language Models for Social Data Science Using Minimal Data**  

---
Primary Category: cs.CL  
Categories: I-2-7; J-4, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12295v1)  

---


**ABSTRACT**  
The field of machine learning has recently made significant progress in reducing the requirements for labelled training data when building new models. These `cheaper' learning techniques hold significant potential for the social sciences, where development of large labelled training datasets is often a significant practical impediment to the use of machine learning for analytical tasks. In this article we review three `cheap' techniques that have developed in recent years: weak supervision, transfer learning and prompt engineering. For the latter, we also review the particular case of zero-shot prompting of large language models. For each technique we provide a guide of how it works and demonstrate its application across six different realistic social science applications (two different tasks paired with three different dataset makeups). We show good performance for all techniques, and in particular we demonstrate how prompting of large language models can achieve high accuracy at very low cost. Our results are accompanied by a code repository to make it easy for others to duplicate our work and use it in their own research. Overall, our article is intended to stimulate further uptake of these techniques in the social sciences.

{{</citation>}}


### (6/119) GRATH: Gradual Self-Truthifying for Large Language Models (Weixin Chen et al., 2024)

{{<citation>}}

Weixin Chen, Bo Li. (2024)  
**GRATH: Gradual Self-Truthifying for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.12292v1)  

---


**ABSTRACT**  
Truthfulness is paramount for large language models (LLMs) as they are increasingly deployed in real-world applications. However, existing LLMs still struggle with generating truthful answers and content, as evidenced by their modest performance on benchmarks like TruthfulQA. To address this issue, we propose GRAdual self-truTHifying (GRATH), a novel post-processing method to enhance truthfulness of LLMs. GRATH utilizes out-of-domain question prompts to generate corresponding answers and adaptively optimizes the model via direct preference optimization (DPO). Note that during this process, GRATH learns truthfulness in a self-supervised manner without requiring annotated answers. In particular, GRATH first generates pairwise truthfulness training data by prompting the LLM itself, with each pair containing a question and its correct and incorrect answers. The model is then fine-tuned using DPO to learn from the difference between answer pairs. Subsequently, GRATH iteratively refines the truthfulness data and optimizes the model, leading to a gradual improvement in model truthfulness. Empirically, we evaluate GRATH using different 7B-LLMs and compare with LLMs with similar or even larger sizes on benchmark datasets. Our results show that GRATH effectively improves LLMs' truthfulness without compromising other core capabilities. Notably, GRATH achieves state-of-the-art performance on TruthfulQA, with MC1 accuracy as 54.71% and MC2 accuracy as 69.10%, which even surpass those on larger-scale models, such as Llama2-Chat-70B, by 23.62% and 24.18%, respectively.

{{</citation>}}


### (7/119) APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference (Bowen Zhao et al., 2024)

{{<citation>}}

Bowen Zhao, Hannaneh Hajishirzi, Qingqing Cao. (2024)  
**APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, LLaMA, Language Model, Pretrained Language Models, Pruning, T5  
[Paper Link](http://arxiv.org/abs/2401.12200v1)  

---


**ABSTRACT**  
Fine-tuning and inference with large Language Models (LM) are generally known to be expensive. Parameter-efficient fine-tuning over pretrained LMs reduces training memory by updating a small number of LM parameters but does not improve inference efficiency. Structured pruning improves LM inference efficiency by removing consistent parameter blocks, yet often increases training memory and time. To improve both training and inference efficiency, we introduce APT that adaptively prunes and tunes parameters for the LMs. At the early stage of fine-tuning, APT dynamically adds salient tuning parameters for fast and accurate convergence while discarding unimportant parameters for efficiency. Compared to baselines, our experiments show that APT maintains up to 98% task performance when pruning RoBERTa and T5 models with 40% parameters left while keeping 86.4% LLaMA models' performance with 70% parameters remained. Furthermore, APT speeds up LMs fine-tuning by up to 8x and reduces large LMs memory training footprint by up to 70%.

{{</citation>}}


### (8/119) Text Embedding Inversion Attacks on Multilingual Language Models (Yiyi Chen et al., 2024)

{{<citation>}}

Yiyi Chen, Heather Lent, Johannes Bjerva. (2024)  
**Text Embedding Inversion Attacks on Multilingual Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs.CL  
Keywords: Embedding, Language Model, Multilingual, NLP, Security  
[Paper Link](http://arxiv.org/abs/2401.12192v1)  

---


**ABSTRACT**  
Representing textual information as real-numbered embeddings has become the norm in NLP. Moreover, with the rise of public interest in large language models (LLMs), Embeddings as a Service (EaaS) has rapidly gained traction as a business model. This is not without outstanding security risks, as previous research has demonstrated that sensitive data can be reconstructed from embeddings, even without knowledge of the underlying model that generated them. However, such work is limited by its sole focus on English, leaving all other languages vulnerable to attacks by malicious actors. %As many international and multilingual companies leverage EaaS, there is an urgent need for research into multilingual LLM security. To this end, this work investigates LLM security from the perspective of multilingual embedding inversion. Concretely, we define the problem of black-box multilingual and cross-lingual inversion attacks, with special attention to a cross-domain scenario. Our findings reveal that multilingual models are potentially more vulnerable to inversion attacks than their monolingual counterparts. This stems from the reduced data requirements for achieving comparable inversion performance in settings where the underlying language is not known a-priori. To our knowledge, this work is the first to delve into multilinguality within the context of inversion attacks, and our findings highlight the need for further investigation and enhanced defenses in the area of NLP Security.

{{</citation>}}


### (9/119) Anisotropy Is Inherent to Self-Attention in Transformers (Nathan Godey et al., 2024)

{{<citation>}}

Nathan Godey, Éric de la Clergerie, Benoît Sagot. (2024)  
**Anisotropy Is Inherent to Self-Attention in Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, NLP, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.12143v2)  

---


**ABSTRACT**  
The representation degeneration problem is a phenomenon that is widely observed among self-supervised learning methods based on Transformers. In NLP, it takes the form of anisotropy, a singular property of hidden representations which makes them unexpectedly close to each other in terms of angular distance (cosine-similarity). Some recent works tend to show that anisotropy is a consequence of optimizing the cross-entropy loss on long-tailed distributions of tokens. We show in this paper that anisotropy can also be observed empirically in language models with specific objectives that should not suffer directly from the same consequences. We also show that the anisotropy problem extends to Transformers trained on other modalities. Our observations suggest that anisotropy is actually inherent to Transformers-based models.

{{</citation>}}


### (10/119) The Curious Case of Nonverbal Abstract Reasoning with Multi-Modal Large Language Models (Kian Ahrabian et al., 2024)

{{<citation>}}

Kian Ahrabian, Zhivar Sourati, Kexuan Sun, Jiarui Zhang, Yifan Jiang, Fred Morstatter, Jay Pujara. (2024)  
**The Curious Case of Nonverbal Abstract Reasoning with Multi-Modal Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.12117v1)  

---


**ABSTRACT**  
While large language models (LLMs) are still being adopted to new domains and utilized in novel applications, we are experiencing an influx of the new generation of foundation models, namely multi-modal large language models (MLLMs). These models integrate verbal and visual information, opening new possibilities to demonstrate more complex reasoning abilities at the intersection of the two modalities. However, despite the revolutionizing prospect of MLLMs, our understanding of their reasoning abilities is limited. In this study, we assess the nonverbal abstract reasoning abilities of open-source and closed-source MLLMs using variations of Raven's Progressive Matrices. Our experiments expose the difficulty of solving such problems while showcasing the immense gap between open-source and closed-source models. We also reveal critical shortcomings with individual visual and textual modules, subjecting the models to low-performance ceilings. Finally, to improve MLLMs' performance, we experiment with various methods, such as Chain-of-Thought prompting, resulting in a significant (up to 100%) boost in performance.

{{</citation>}}


### (11/119) An Empirical Analysis of In-context Learning Abilities of LLMs for MT (Pranjal A. Chitale et al., 2024)

{{<citation>}}

Pranjal A. Chitale, Jay Gala, Varun Gumma, Mitesh M. Khapra, Raj Dabre. (2024)  
**An Empirical Analysis of In-context Learning Abilities of LLMs for MT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLOOM  
[Paper Link](http://arxiv.org/abs/2401.12097v1)  

---


**ABSTRACT**  
In-context learning (ICL) has consistently demonstrated superior performance over zero-shot performance in large language models (LLMs). However, the understanding of the dynamics of ICL and the aspects that influence downstream performance remains limited, especially for natural language generation (NLG) tasks. This work aims to address this gap by investigating the ICL capabilities of LLMs and studying the impact of different aspects of the in-context demonstrations for the task of machine translation (MT). Our preliminary investigations aim to discern whether in-context learning (ICL) is predominantly influenced by demonstrations or instructions by applying diverse perturbations to in-context demonstrations while preserving the task instruction. We observe varying behavior to perturbed examples across different model families, notably with BLOOM-7B derivatives being severely influenced by noise, whereas Llama 2 derivatives not only exhibit robustness but also tend to show enhancements over the clean baseline when subject to perturbed demonstrations. This suggests that the robustness of ICL may be governed by several factors, including the type of noise, perturbation direction (source or target), the extent of pretraining of the specific model, and fine-tuning for downstream tasks if applicable. Further investigation is warranted to develop a comprehensive understanding of these factors in future research.

{{</citation>}}


### (12/119) Unsupervised Learning of Graph from Recipes (Aissatou Diallo et al., 2024)

{{<citation>}}

Aissatou Diallo, Antonis Bikakis, Luke Dickens, Anthony Hunter, Rob Miller. (2024)  
**Unsupervised Learning of Graph from Recipes**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.12088v1)  

---


**ABSTRACT**  
Cooking recipes are one of the most readily available kinds of procedural text. They consist of natural language instructions that can be challenging to interpret. In this paper, we propose a model to identify relevant information from recipes and generate a graph to represent the sequence of actions in the recipe. In contrast with other approaches, we use an unsupervised approach. We iteratively learn the graph structure and the parameters of a $\mathsf{GNN}$ encoding the texts (text-to-graph) one sequence at a time while providing the supervision by decoding the graph into text (graph-to-text) and comparing the generated text to the input. We evaluate the approach by comparing the identified entities with annotated datasets, comparing the difference between the input and output texts, and comparing our generated graphs with those generated by state of the art methods.

{{</citation>}}


### (13/119) Temporal Blind Spots in Large Language Models (Jonas Wallat et al., 2024)

{{<citation>}}

Jonas Wallat, Adam Jatowt, Avishek Anand. (2024)  
**Temporal Blind Spots in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.12078v1)  

---


**ABSTRACT**  
Large language models (LLMs) have recently gained significant attention due to their unparalleled ability to perform various natural language processing tasks. These models, benefiting from their advanced natural language understanding capabilities, have demonstrated impressive zero-shot performance. However, the pre-training data utilized in LLMs is often confined to a specific corpus, resulting in inherent freshness and temporal scope limitations. Consequently, this raises concerns regarding the effectiveness of LLMs for tasks involving temporal intents. In this study, we aim to investigate the underlying limitations of general-purpose LLMs when deployed for tasks that require a temporal understanding. We pay particular attention to handling factual temporal knowledge through three popular temporal QA datasets. Specifically, we observe low performance on detailed questions about the past and, surprisingly, for rather new information. In manual and automatic testing, we find multiple temporal errors and characterize the conditions under which QA performance deteriorates. Our analysis contributes to understanding LLM limitations and offers valuable insights into developing future models that can better cater to the demands of temporally-oriented tasks. The code is available\footnote{https://github.com/jwallat/temporalblindspots}.

{{</citation>}}


### (14/119) Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text (Abhimanyu Hans et al., 2024)

{{<citation>}}

Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, Tom Goldstein. (2024)  
**Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.12070v1)  

---


**ABSTRACT**  
Detecting text generated by modern large language models is thought to be hard, as both LLMs and humans can exhibit a wide range of complex behaviors. However, we find that a score based on contrasting two closely related language models is highly accurate at separating human-generated and machine-generated text. Based on this mechanism, we propose a novel LLM detector that only requires simple calculations using a pair of pre-trained LLMs. The method, called Binoculars, achieves state-of-the-art accuracy without any training data. It is capable of spotting machine text from a range of modern LLMs without any model-specific modifications. We comprehensively evaluate Binoculars on a number of text sources and in varied situations. Over a wide range of document types, Binoculars detects over 90% of generated samples from ChatGPT (and other LLMs) at a false positive rate of 0.01%, despite not being trained on any ChatGPT data.

{{</citation>}}


### (15/119) ALMs: Authorial Language Models for Authorship Attribution (Weihang Huang et al., 2024)

{{<citation>}}

Weihang Huang, Akira Murakami, Jack Grieve. (2024)  
**ALMs: Authorial Language Models for Authorship Attribution**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, T5  
[Paper Link](http://arxiv.org/abs/2401.12005v1)  

---


**ABSTRACT**  
In this paper, we introduce an authorship attribution method called Authorial Language Models (ALMs) that involves identifying the most likely author of a questioned document based on the perplexity of the questioned document calculated for a set of causal language models fine-tuned on the writings of a set of candidate author. We benchmarked ALMs against state-of-art-systems using the CCAT50 dataset and the Blogs50 datasets. We find that ALMs achieves a macro-average accuracy score of 83.6% on Blogs50, outperforming all other methods, and 74.9% on CCAT50, matching the performance of the best method. To assess the performance of ALMs on shorter texts, we also conducted text ablation testing. We found that to reach a macro-average accuracy of 70%, ALMs needs 40 tokens on Blogs50 and 400 tokens on CCAT50, while to reach 60% ALMs requires 20 tokens on Blogs50 and 70 tokens on CCAT50.

{{</citation>}}


### (16/119) Synergizing Machine Learning & Symbolic Methods: A Survey on Hybrid Approaches to Natural Language Processing (Rrubaa Panchendrarajan et al., 2024)

{{<citation>}}

Rrubaa Panchendrarajan, Arkaitz Zubiaga. (2024)  
**Synergizing Machine Learning & Symbolic Methods: A Survey on Hybrid Approaches to Natural Language Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.11972v1)  

---


**ABSTRACT**  
The advancement of machine learning and symbolic approaches have underscored their strengths and weaknesses in Natural Language Processing (NLP). While machine learning approaches are powerful in identifying patterns in data, they often fall short in learning commonsense and the factual knowledge required for the NLP tasks. Meanwhile, the symbolic methods excel in representing knowledge-rich data. However, they struggle to adapt dynamic data and generalize the knowledge. Bridging these two paradigms through hybrid approaches enables the alleviation of weaknesses in both while preserving their strengths. Recent studies extol the virtues of this union, showcasing promising results in a wide range of NLP tasks. In this paper, we present an overview of hybrid approaches used for NLP. Specifically, we delve into the state-of-the-art hybrid approaches used for a broad spectrum of NLP tasks requiring natural language understanding, generation, and reasoning. Furthermore, we discuss the existing resources available for hybrid approaches for NLP along with the challenges, offering a roadmap for future directions.

{{</citation>}}


### (17/119) Claim Detection for Automated Fact-checking: A Survey on Monolingual, Multilingual and Cross-Lingual Research (Rrubaa Panchendrarajan et al., 2024)

{{<citation>}}

Rrubaa Panchendrarajan, Arkaitz Zubiaga. (2024)  
**Claim Detection for Automated Fact-checking: A Survey on Monolingual, Multilingual and Cross-Lingual Research**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2401.11969v2)  

---


**ABSTRACT**  
Automated fact-checking has drawn considerable attention over the past few decades due to the increase in the diffusion of misinformation on online platforms. This is often carried out as a sequence of tasks comprising (i) the detection of sentences circulating in online platforms which constitute claims needing verification, followed by (ii) the verification process of those claims. This survey focuses on the former, by discussing existing efforts towards detecting claims needing fact-checking, with a particular focus on multilingual data and methods. This is a challenging and fertile direction where existing methods are yet far from matching human performance due to the profoundly challenging nature of the issue. Especially, the dissemination of information across multiple social platforms, articulated in multiple languages and modalities demands more generalized solutions for combating misinformation. Focusing on multilingual misinformation, we present a comprehensive survey of existing multilingual claim detection research. We present state-of-the-art multilingual claim detection research categorized into three key factors of the problem, verifiability, priority, and similarity. Further, we present a detailed overview of the existing multilingual datasets along with the challenges and suggest possible future advancements.

{{</citation>}}


### (18/119) CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark (Ge Zhang et al., 2024)

{{<citation>}}

Ge Zhang, Xinrun Du, Bei Chen, Yiming Liang, Tongxu Luo, Tianyu Zheng, Kang Zhu, Yuyang Cheng, Chunpu Xu, Shuyue Guo, Haoran Zhang, Xingwei Qu, Junjie Wang, Ruibin Yuan, Yizhi Li, Zekun Wang, Yudong Liu, Yu-Hsuan Tsai, Fengji Zhang, Chenghua Lin, Wenhao Huang, Wenhu Chen, Jie Fu. (2024)  
**CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.11944v1)  

---


**ABSTRACT**  
As the capabilities of large multimodal models (LMMs) continue to advance, evaluating the performance of LMMs emerges as an increasing need. Additionally, there is an even larger gap in evaluating the advanced knowledge and reasoning abilities of LMMs in non-English contexts such as Chinese. We introduce CMMMU, a new Chinese Massive Multi-discipline Multimodal Understanding benchmark designed to evaluate LMMs on tasks demanding college-level subject knowledge and deliberate reasoning in a Chinese context. CMMMU is inspired by and strictly follows the annotation and analysis pattern of MMMU.   CMMMU includes 12k manually collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering, like its companion, MMMU. These questions span 30 subjects and comprise 39 highly heterogeneous image types, such as charts, diagrams, maps, tables, music sheets, and chemical structures.   CMMMU focuses on complex perception and reasoning with domain-specific knowledge in the Chinese context. We evaluate 11 open-source LLMs and one proprietary GPT-4V(ision). Even GPT-4V only achieves accuracies of 42%, indicating a large space for improvement. CMMMU will boost the community to build the next-generation LMMs towards expert artificial intelligence and promote the democratization of LMMs by providing diverse language contexts.

{{</citation>}}


### (19/119) Blinded by Generated Contexts: How Language Models Merge Generated and Retrieved Contexts for Open-Domain QA? (Hexiang Tan et al., 2024)

{{<citation>}}

Hexiang Tan, Fei Sun, Wanli Yang, Yuanzhuo Wang, Qi Cao, Xueqi Cheng. (2024)  
**Blinded by Generated Contexts: How Language Models Merge Generated and Retrieved Contexts for Open-Domain QA?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2401.11911v1)  

---


**ABSTRACT**  
While auxiliary information has become a key to enhance Large Language Models (LLMs), relatively little is known about how well LLMs merge these contexts, specifically generated and retrieved. To study this, we formulate a task specifically designed to identify whether the answers, derived from the integration of generated and retrieved contexts, are attributed to either generated or retrieved contexts. To support this task, we develop a methodology to construct datasets with conflicting contexts, where each question is paired with both generated and retrieved contexts, yet only one of them contains the correct answer. Our experiments reveal a significant bias in LLMs towards generated contexts, as evidenced across state-of-the-art open (Llama2-7b/13b) and closed (GPT 3.5/4) systems. We further identify two key factors contributing to this bias: i) Contexts generated by LLMs typically show greater similarity to the questions, increasing their likelihood of selection; ii) The segmentation process used in retrieved contexts disrupts their completeness, thereby hindering their full utilization in LLMs. Our analysis enhances the understanding of how LLMs merge diverse contexts, offering valuable insights for advancing current augmentation methods for LLMs.

{{</citation>}}


### (20/119) PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety (Zaibin Zhang et al., 2024)

{{<citation>}}

Zaibin Zhang, Yongting Zhang, Lijun Li, Hongzhi Gao, Lijun Wang, Huchuan Lu, Feng Zhao, Yu Qiao, Jing Shao. (2024)  
**PsySafe: A Comprehensive Framework for Psychological-based Attack, Defense, and Evaluation of Multi-agent System Safety**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-MA, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.11880v1)  

---


**ABSTRACT**  
Multi-agent systems, augmented with Large Language Models (LLMs), demonstrate significant capabilities for collective intelligence. However, the potential misuse of this intelligence for malicious purposes presents significant risks. To date, comprehensive research on the safety issues associated with multi-agent systems remains limited. From the perspective of agent psychology, we discover that the dark psychological states of agents can lead to severe safety issues. To address these issues, we propose a comprehensive framework grounded in agent psychology. In our framework, we focus on three aspects: identifying how dark personality traits in agents might lead to risky behaviors, designing defense strategies to mitigate these risks, and evaluating the safety of multi-agent systems from both psychological and behavioral perspectives. Our experiments reveal several intriguing phenomena, such as the collective dangerous behaviors among agents, agents' propensity for self-reflection when engaging in dangerous behavior, and the correlation between agents' psychological assessments and their dangerous behaviors. We anticipate that our framework and observations will provide valuable insights for further research into the safety of multi-agent systems. We will make our data and code publicly accessible at https:/github.com/AI4Good24/PsySafe.

{{</citation>}}


### (21/119) Improving Small Language Models' Mathematical Reasoning via Mix Thoughts Distillation (Xunyu Zhu et al., 2024)

{{<citation>}}

Xunyu Zhu, Jian Li, Yong Liu, Can Ma, Weiping Wang. (2024)  
**Improving Small Language Models' Mathematical Reasoning via Mix Thoughts Distillation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.11864v1)  

---


**ABSTRACT**  
This work addresses the challenge of democratizing advanced Large Language Models (LLMs) by compressing their mathematical reasoning capabilities into sub-billion parameter Small Language Models (SLMs) without compromising performance. We introduce Equation-of-Thought Distillation (EoTD), a novel technique that encapsulates the reasoning process into equation-based representations to construct an EoTD dataset for fine-tuning SLMs. Additionally, we propose the Mix Thoughts Distillation (MTD) framework to enhance the reasoning performance of SLMs. This involves creating a reasoning dataset with multiple thought processes and using it for fine-tuning. Our experimental findings demonstrate that EoTD significantly boosts the reasoning abilities of SLMs, while MTD enables these models to achieve state-of-the-art reasoning performance.

{{</citation>}}


### (22/119) The Right Model for the Job: An Evaluation of Legal Multi-Label Classification Baselines (Martina Forster et al., 2024)

{{<citation>}}

Martina Forster, Claudia Schulz, Prudhvi Nokku, Melicaalsadat Mirsafian, Jaykumar Kasundra, Stavroula Skylaki. (2024)  
**The Right Model for the Job: An Evaluation of Legal Multi-Label Classification Baselines**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Legal, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11852v1)  

---


**ABSTRACT**  
Multi-Label Classification (MLC) is a common task in the legal domain, where more than one label may be assigned to a legal document. A wide range of methods can be applied, ranging from traditional ML approaches to the latest Transformer-based architectures. In this work, we perform an evaluation of different MLC methods using two public legal datasets, POSTURE50K and EURLEX57K. By varying the amount of training data and the number of labels, we explore the comparative advantage offered by different approaches in relation to the dataset properties. Our findings highlight DistilRoBERTa and LegalBERT as performing consistently well in legal MLC with reasonable computational demands. T5 also demonstrates comparable performance while offering advantages as a generative model in the presence of changing label sets. Finally, we show that the CrossEncoder exhibits potential for notable macro-F1 score improvements, albeit with increased computational costs.

{{</citation>}}


### (23/119) AI for social science and social science of AI: A Survey (Ruoxi Xu et al., 2024)

{{<citation>}}

Ruoxi Xu, Yingfei Sun, Mengjie Ren, Shiguang Guo, Ruotong Pan, Hongyu Lin, Le Sun, Xianpei Han. (2024)  
**AI for social science and social science of AI: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11839v1)  

---


**ABSTRACT**  
Recent advancements in artificial intelligence, particularly with the emergence of large language models (LLMs), have sparked a rethinking of artificial general intelligence possibilities. The increasing human-like capabilities of AI are also attracting attention in social science research, leading to various studies exploring the combination of these two fields. In this survey, we systematically categorize previous explorations in the combination of AI and social science into two directions that share common technical approaches but differ in their research objectives. The first direction is focused on AI for social science, where AI is utilized as a powerful tool to enhance various stages of social science research. While the second direction is the social science of AI, which examines AI agents as social entities with their human-like cognitive and linguistic capabilities. By conducting a thorough review, particularly on the substantial progress facilitated by recent advancements in large language models, this paper introduces a fresh perspective to reassess the relationship between AI and social science, provides a cohesive framework that allows researchers to understand the distinctions and connections between AI for social science and social science of AI, and also summarized state-of-art experiment simulation platforms to facilitate research in these two directions. We believe that as AI technology continues to advance and intelligent agents find increasing applications in our daily lives, the significance of the combination of AI and social science will become even more prominent.

{{</citation>}}


### (24/119) SuperCLUE-Math6: Graded Multi-Step Math Reasoning Benchmark for LLMs in Chinese (Liang Xu et al., 2024)

{{<citation>}}

Liang Xu, Hang Xue, Lei Zhu, Kangkang Zhao. (2024)  
**SuperCLUE-Math6: Graded Multi-Step Math Reasoning Benchmark for LLMs in Chinese**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.11819v1)  

---


**ABSTRACT**  
We introduce SuperCLUE-Math6(SC-Math6), a new benchmark dataset to evaluate the mathematical reasoning abilities of Chinese language models. SC-Math6 is designed as an upgraded Chinese version of the GSM8K dataset with enhanced difficulty, diversity, and application scope. It consists of over 2000 mathematical word problems requiring multi-step reasoning and providing natural language solutions. We propose an innovative scheme to quantify the reasoning capability of large models based on performance over problems with different reasoning steps. Experiments on 12 representative Chinese models demonstrate a clear stratification of reasoning levels, with top models like GPT-4 showing superior performance. SC-Math6 fills the gap in Chinese mathematical reasoning benchmarks and provides a comprehensive testbed to advance the intelligence of Chinese language models.

{{</citation>}}


### (25/119) Hallucination is Inevitable: An Innate Limitation of Large Language Models (Ziwei Xu et al., 2024)

{{<citation>}}

Ziwei Xu, Sanjay Jain, Mohan Kankanhalli. (2024)  
**Hallucination is Inevitable: An Innate Limitation of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.11817v1)  

---


**ABSTRACT**  
Hallucination has been widely recognized to be a significant drawback for large language models (LLMs). There have been many works that attempt to reduce the extent of hallucination. These efforts have mostly been empirical so far, which cannot answer the fundamental question whether it can be completely eliminated. In this paper, we formalize the problem and show that it is impossible to eliminate hallucination in LLMs. Specifically, we define a formal world where hallucination is defined as inconsistencies between a computable LLM and a computable ground truth function. By employing results from learning theory, we show that LLMs cannot learn all of the computable functions and will therefore always hallucinate. Since the formal world is a part of the real world which is much more complicated, hallucinations are also inevitable for real world LLMs. Furthermore, for real world LLMs constrained by provable time complexity, we describe the hallucination-prone tasks and empirically validate our claims. Finally, using the formal world framework, we discuss the possible mechanisms and efficacies of existing hallucination mitigators as well as the practical implications on the safe deployment of LLMs.

{{</citation>}}


### (26/119) Speak It Out: Solving Symbol-Related Problems with Symbol-to-Language Conversion for Language Models (Yile Wang et al., 2024)

{{<citation>}}

Yile Wang, Sijie Cheng, Zixin Sun, Peng Li, Yang Liu. (2024)  
**Speak It Out: Solving Symbol-Related Problems with Symbol-to-Language Conversion for Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.11725v1)  

---


**ABSTRACT**  
Symbols (or more broadly, non-natural language textual representations) such as numerical sequences, molecular formulas, and table delimiters widely exist, playing important roles in various tasks such as abstract reasoning, chemical property prediction, and table question answering. Despite the impressive natural language comprehension capabilities of large language models (LLMs), their reasoning abilities for symbols remain inadequate, which could attributed to the difference between symbol representations and general natural languages. We propose symbol-to-language (S2L), a tuning-free method that enables large language models to solve symbol-related problems with information expressed in natural language. Specifically, S2L first converts the symbols involved to language-based representations, which can be implemented by prompting LLMs or leveraging external tools, then these language-based representations are integrated into the original problem via direct substitution or concatenation, serving as useful input information for LLMs. We evaluate the S2L method using both API-based (GPT-4, ChatGPT) and open-source (OpenChat) models over eight symbol-related tasks, ranging from symbol-only abstract reasoning to sentiment analysis in social media. Experimental results show that S2L consistently leads to superior performance. For example, by employing S2L for GPT-4, there can be average significant improvements of +21.9% and +9.5% for subtasks in 1D-ARC and Dyck language, respectively. Codes and data are available at https://github.com/THUNLP-MT/symbol2language.

{{</citation>}}


### (27/119) Keep Decoding Parallel with Effective Knowledge Distillation from Language Models to End-to-end Speech Recognisers (Michael Hentschel et al., 2024)

{{<citation>}}

Michael Hentschel, Yuta Nishikawa, Tatsuya Komatsu, Yusuke Fujita. (2024)  
**Keep Decoding Parallel with Effective Knowledge Distillation from Language Models to End-to-end Speech Recognisers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: BERT, Knowledge Distillation, Language Model  
[Paper Link](http://arxiv.org/abs/2401.11700v1)  

---


**ABSTRACT**  
This study presents a novel approach for knowledge distillation (KD) from a BERT teacher model to an automatic speech recognition (ASR) model using intermediate layers. To distil the teacher's knowledge, we use an attention decoder that learns from BERT's token probabilities. Our method shows that language model (LM) information can be more effectively distilled into an ASR model using both the intermediate layers and the final layer. By using the intermediate layers as distillation target, we can more effectively distil LM knowledge into the lower network layers. Using our method, we achieve better recognition accuracy than with shallow fusion of an external LM, allowing us to maintain fast parallel decoding. Experiments on the LibriSpeech dataset demonstrate the effectiveness of our approach in enhancing greedy decoding with connectionist temporal classification (CTC).

{{</citation>}}


### (28/119) Revolutionizing Finance with LLMs: An Overview of Applications and Insights (Huaqin Zhao et al., 2024)

{{<citation>}}

Huaqin Zhao, Zhengliang Liu, Zihao Wu, Yiwei Li, Tianze Yang, Peng Shu, Shaochen Xu, Haixing Dai, Lin Zhao, Gengchen Mai, Ninghao Liu, Tianming Liu. (2024)  
**Revolutionizing Finance with LLMs: An Overview of Applications and Insights**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11641v1)  

---


**ABSTRACT**  
In recent years, Large Language Models (LLMs) like ChatGPT have seen considerable advancements and have been applied in diverse fields. Built on the Transformer architecture, these models are trained on extensive datasets, enabling them to understand and generate human language effectively. In the financial domain, the deployment of LLMs is gaining momentum. These models are being utilized for automating financial report generation, forecasting market trends, analyzing investor sentiment, and offering personalized financial advice. Leveraging their natural language processing capabilities, LLMs can distill key insights from vast financial data, aiding institutions in making informed investment choices and enhancing both operational efficiency and customer satisfaction. In this study, we provide a comprehensive overview of the emerging integration of LLMs into various financial tasks. Additionally, we conducted holistic tests on multiple financial tasks through the combination of natural language instructions. Our findings show that GPT-4 effectively follow prompt instructions across various financial tasks. This survey and evaluation of LLMs in the financial domain aim to deepen the understanding of LLMs' current role in finance for both financial practitioners and LLM researchers, identify new research and application prospects, and highlight how these technologies can be leveraged to solve practical challenges in the finance industry.

{{</citation>}}


## cs.SE (3)



### (29/119) Program Decomposition and Translation with Static Analysis (Ali Reza Ibrahimzada, 2024)

{{<citation>}}

Ali Reza Ibrahimzada. (2024)  
**Program Decomposition and Translation with Static Analysis**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12412v1)  

---


**ABSTRACT**  
The rising popularity of Large Language Models (LLMs) has motivated exploring their use in code-related tasks. Code LLMs with more than millions of parameters are trained on a massive amount of code in different Programming Languages (PLs). Such models are used for automating various Software Engineering (SE) tasks using prompt engineering. However, given the very large size of industry-scale project files, a major issue of these LLMs is their limited context window size, motivating the question of "Can these LLMs process very large files and can we effectively perform prompt engineering?". Code translation aims to convert source code from one PL to another. In this work, we assess the effect of method-level program decomposition on context window of LLMs and investigate how this approach can enable translation of very large files which originally could not be done due to out-of-context issue. Our observations from 20 well-known java projects and approximately 60K methods suggest that method-level program decomposition significantly improves the limited context window problem of LLMs by 99.5%. Furthermore, our empirical analysis indicate that with method-level decomposition, each input fragment on average only consumes 5% of the context window, leaving more context space for prompt engineering and the output. Finally, we investigate the effectiveness of a Call Graph (CG) approach for translating very large files when doing method-level program decomposition.

{{</citation>}}


### (30/119) NLP-based Relation Extraction Methods in Requirements Engineering (Quim Motger et al., 2024)

{{<citation>}}

Quim Motger, Xavier Franch. (2024)  
**NLP-based Relation Extraction Methods in Requirements Engineering**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: NLP, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2401.12075v1)  

---


**ABSTRACT**  
In the context of requirements engineering, relation extraction is the task of documenting the traceability between requirements artefacts. When dealing with textual requirements (i.e., requirements expressed using natural language), relation extraction becomes a cognitively challenging task, especially in terms of ambiguity and required effort from domain-experts. Hence, in highly-adaptive, large-scale environments, effective and efficient automated relation extraction using natural language processing techniques becomes essential. In this chapter, we present a comprehensive overview of natural language-based relation extraction from text-based requirements. We initially describe the fundamentals of requirements relations based on the most relevant literature in the field, including the most common requirements relations types. The core of the chapter is composed by two main sections: (i) natural language techniques for the identification and categorization of requirements relations (i.e., syntactic vs. semantic techniques), and (ii) information extraction methods for the task of relation extraction (i.e., retrieval-based vs. machine learning-based methods). We complement this analysis with the state-of-the-art challenges and the envisioned future research directions. Overall, this chapter aims at providing a clear perspective on the theoretical and practical fundamentals in the field of natural language-based relation extraction.

{{</citation>}}


### (31/119) Modular Monolith: Is This the Trend in Software Architecture? (Ruoyu Su et al., 2024)

{{<citation>}}

Ruoyu Su, Xiaozhou Li. (2024)  
**Modular Monolith: Is This the Trend in Software Architecture?**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.11867v1)  

---


**ABSTRACT**  
Recently modular monolith architecture has attracted the attention of practitioners, as Google proposed "Service Weaver" framework to enable developers to write applications as modular monolithic and deploy them as a set of microservices. Google considered it as a framework that has the best of both worlds and it seems to be a trend in software architecture. This paper aims to understand the definition of the modular monolith in industry and investigate frameworks and cases building modular monolith architecture. We conducted a systematic grey literature review, and the results show that modular monolith combines the advantages of monoliths with microservices. We found three frameworks and four cases of building modular monolith architecture. In general, the modular monolith is an alternative way to microservices, and it also could be a previous step before systems migrate to microservices.

{{</citation>}}


## cs.DC (2)



### (32/119) Learning Recovery Strategies for Dynamic Self-healing in Reactive Systems (Mateo Sanabria et al., 2024)

{{<citation>}}

Mateo Sanabria, Ivana Dusparic, Nicolas Cardozo. (2024)  
**Learning Recovery Strategies for Dynamic Self-healing in Reactive Systems**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-SE, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12405v1)  

---


**ABSTRACT**  
Self-healing systems depend on following a set of predefined instructions to recover from a known failure state. Failure states are generally detected based on domain specific specialized metrics. Failure fixes are applied at predefined application hooks that are not sufficiently expressive to manage different failure types. Self-healing is usually applied in the context of distributed systems, where the detection of failures is constrained to communication problems, and resolution strategies often consist of replacing complete components. Our proposal targets complex reactive systems, defining monitors as predicates specifying satisfiability conditions of system properties. Such monitors are functionally expressive and can be defined at run time to detect failure states at any execution point. Once failure states are detected, we use a Reinforcement Learning-based technique to learn a recovery strategy based on users' corrective sequences. Finally, to execute the learned strategies, we extract them as COP variations that activate dynamically whenever the failure state is detected, overwriting the base system behavior with the recovery strategy for that state. We validate the feasibility and effectiveness of our framework through a prototypical reactive application for tracking mouse movements, and the DeltaIoT exemplar for self-healing systems. Our results demonstrate that with just the definition of monitors, the system is effective in detecting and recovering from failures between 55%-92% of the cases in the first application, and at par with the predefined strategies in the second application.

{{</citation>}}


### (33/119) Integrated Sensing, Communication, and Computing: An Information-oriented Resource Transaction Mechanism (Ning Chen et al., 2024)

{{<citation>}}

Ning Chen, Zhipeng Cheng, Xuwei Fan, Zhang Liu, Bangzhen Huang, Jie Yang, Yifeng Zhao, Lianfen Huang. (2024)  
**Integrated Sensing, Communication, and Computing: An Information-oriented Resource Transaction Mechanism**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.11759v1)  

---


**ABSTRACT**  
Information acquisition from target perception represents the key enabling technology of the Internet of Automatic Vehicles (IoAV), which is essential for the decision-making and control operation of connected automatic vehicles (CAVs). Exploring target information involves multiple operations on data, e.g., wireless sensing (for data acquisition), communication (for data transmission), and computing (for data analysis), which all rely on the consumption of time-space-frequency-computing (TSFC) multi-domain resources. Due to the coupled resource sharing of sensing, communication, and computing procedures, the resource management of information-oriented IoAV is commonly formulated as a non-convex NP-hard problem. In this article, further combining the integrated sensing and communication (ISAC) and computing, we introduce the integrated sensing, communication, and computing (ISCC), wherein the TSFC resources are decoupled from the specific processes and shared universally among sensing, communication, and computing processes. Furthermore, the information-oriented resource trading platform (IRTP) is established, which transforms the problem of ISCC resource management into a resource-information substitution model. Finally, we embed the employment topology structure in IoAV into neural network architecture, taking advantage of the graph neural network (GNN) and multi-worker reinforcement learning, and propose the dynamic resource management strategy based on the asynchronous advantage GNN (A2GNN) algorithm, which can achieve the convergence both of information gain maximization and resource consumption minimization, realizing efficient information-oriented resource management.

{{</citation>}}


## cs.AI (2)



### (34/119) Analyzing the Effectiveness of Large Language Models on Text-to-SQL Synthesis (Richard Roberson et al., 2024)

{{<citation>}}

Richard Roberson, Gowtham Kaki, Ashutosh Trivedi. (2024)  
**Analyzing the Effectiveness of Large Language Models on Text-to-SQL Synthesis**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-DB, cs-PL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12379v1)  

---


**ABSTRACT**  
This study investigates various approaches to using Large Language Models (LLMs) for Text-to-SQL program synthesis, focusing on the outcomes and insights derived. Employing the popular Text-to-SQL dataset, spider, the goal was to input a natural language question along with the database schema and output the correct SQL SELECT query. The initial approach was to fine-tune a local and open-source model to generate the SELECT query. After QLoRa fine-tuning WizardLM's WizardCoder-15B model on the spider dataset, the execution accuracy for generated queries rose to a high of 61%. With the second approach, using the fine-tuned gpt-3.5-turbo-16k (Few-shot) + gpt-4-turbo (Zero-shot error correction), the execution accuracy reached a high of 82.1%. Of all the incorrect queries, most can be categorized into a seven different categories of what went wrong: selecting the wrong columns or wrong order of columns, grouping by the wrong column, predicting the wrong values in conditionals, using different aggregates than the ground truth, extra or too few JOIN clauses, inconsistencies in the Spider dataset, and lastly completely incorrect query structure. Most if not all of the queries fall into these categories and it is insightful to understanding where the faults still lie with LLM program synthesis and where they can be improved.

{{</citation>}}


### (35/119) Streamlining Advanced Taxi Assignment Strategies based on Legal Analysis (Holger Billhardt et al., 2024)

{{<citation>}}

Holger Billhardt, José-Antonio Santos, Alberto Fernández, Mar Moreno, Sascha Ossowski, José A. Rodríguez. (2024)  
**Streamlining Advanced Taxi Assignment Strategies based on Legal Analysis**  

---
Primary Category: cs.AI  
Categories: I-2-1, cs-AI, cs.AI  
Keywords: AI, Legal  
[Paper Link](http://arxiv.org/abs/2401.12324v1)  

---


**ABSTRACT**  
In recent years many novel applications have appeared that promote the provision of services and activities in a collaborative manner. The key idea behind such systems is to take advantage of idle or underused capacities of existing resources, in order to provide improved services that assist people in their daily tasks, with additional functionality, enhanced efficiency, and/or reduced cost. Particularly in the domain of urban transportation, many researchers have put forward novel ideas, which are then implemented and evaluated through prototypes that usually draw upon AI methods and tools. However, such proposals also bring up multiple non-technical issues that need to be identified and addressed adequately if such systems are ever meant to be applied to the real world. While, in practice, legal and ethical aspects related to such AI-based systems are seldomly considered in the beginning of the research and development process, we argue that they not only restrict design decisions, but can also help guiding them. In this manuscript, we set out from a prototype of a taxi coordination service that mediates between individual (and autonomous) taxis and potential customers. After representing key aspects of its operation in a semi-structured manner, we analyse its viability from the viewpoint of current legal restrictions and constraints, so as to identify additional non-functional requirements as well as options to address them. Then, we go one step ahead, and actually modify the existing prototype to incorporate the previously identified recommendations. Performing experiments with this improved system helps us identify the most adequate option among several legally admissible alternatives.

{{</citation>}}


## stat.ML (3)



### (36/119) VC dimension of Graph Neural Networks with Pfaffian activation functions (Giuseppe Alessio D'Inverno et al., 2024)

{{<citation>}}

Giuseppe Alessio D'Inverno, Monica Bianchini, Franco Scarselli. (2024)  
**VC dimension of Graph Neural Networks with Pfaffian activation functions**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.12362v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have emerged in recent years as a powerful tool to learn tasks across a wide range of graph domains in a data-driven fashion; based on a message passing mechanism, GNNs have gained increasing popularity due to their intuitive formulation, closely linked with the Weisfeiler-Lehman (WL) test for graph isomorphism, to which they have proven equivalent. From a theoretical point of view, GNNs have been shown to be universal approximators, and their generalization capability (namely, bounds on the Vapnik Chervonekis (VC) dimension) has recently been investigated for GNNs with piecewise polynomial activation functions. The aim of our work is to extend this analysis on the VC dimension of GNNs to other commonly used activation functions, such as sigmoid and hyperbolic tangent, using the framework of Pfaffian function theory. Bounds are provided with respect to architecture parameters (depth, number of neurons, input size) as well as with respect to the number of colors resulting from the 1-WL test applied on the graph domain. The theoretical analysis is supported by a preliminary experimental study.

{{</citation>}}


### (37/119) Mitigating Covariate Shift in Misspecified Regression with Applications to Reinforcement Learning (Philip Amortila et al., 2024)

{{<citation>}}

Philip Amortila, Tongyi Cao, Akshay Krishnamurthy. (2024)  
**Mitigating Covariate Shift in Misspecified Regression with Applications to Reinforcement Learning**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-OC, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12216v1)  

---


**ABSTRACT**  
A pervasive phenomenon in machine learning applications is distribution shift, where training and deployment conditions for a machine learning model differ. As distribution shift typically results in a degradation in performance, much attention has been devoted to algorithmic interventions that mitigate these detrimental effects. In this paper, we study the effect of distribution shift in the presence of model misspecification, specifically focusing on $L_{\infty}$-misspecified regression and adversarial covariate shift, where the regression target remains fixed while the covariate distribution changes arbitrarily. We show that empirical risk minimization, or standard least squares regression, can result in undesirable misspecification amplification where the error due to misspecification is amplified by the density ratio between the training and testing distributions. As our main result, we develop a new algorithm -- inspired by robust optimization techniques -- that avoids this undesirable behavior, resulting in no misspecification amplification while still obtaining optimal statistical rates. As applications, we use this regression procedure to obtain new guarantees in offline and online reinforcement learning with misspecification and establish new separations between previously studied structural conditions and notions of coverage.

{{</citation>}}


### (38/119) Nonparametric Estimation via Variance-Reduced Sketching (Yuehaw Khoo et al., 2024)

{{<citation>}}

Yuehaw Khoo, Yifan Peng, Daren Wang. (2024)  
**Nonparametric Estimation via Variance-Reduced Sketching**  

---
Primary Category: stat.ML  
Categories: cs-LG, cs-NA, math-NA, stat-ME, stat-ML, stat.ML  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2401.11646v1)  

---


**ABSTRACT**  
Nonparametric models are of great interest in various scientific and engineering disciplines. Classical kernel methods, while numerically robust and statistically sound in low-dimensional settings, become inadequate in higher-dimensional settings due to the curse of dimensionality. In this paper, we introduce a new framework called Variance-Reduced Sketching (VRS), specifically designed to estimate density functions and nonparametric regression functions in higher dimensions with a reduced curse of dimensionality. Our framework conceptualizes multivariable functions as infinite-size matrices, and facilitates a new sketching technique motivated by numerical linear algebra literature to reduce the variance in estimation problems. We demonstrate the robust numerical performance of VRS through a series of simulated experiments and real-world data applications. Notably, VRS shows remarkable improvement over existing neural network estimators and classical kernel methods in numerous density estimation and nonparametric regression models. Additionally, we offer theoretical justifications for VRS to support its ability to deliver nonparametric estimation with a reduced curse of dimensionality.

{{</citation>}}


## cs.CR (9)



### (39/119) A Security Risk Assessment Method for Distributed Ledger Technology (DLT) based Applications: Three Industry Case Studies (Elena Baninemeh et al., 2024)

{{<citation>}}

Elena Baninemeh, Slinger Jansen, Katsiaryna Labunets. (2024)  
**A Security Risk Assessment Method for Distributed Ledger Technology (DLT) based Applications: Three Industry Case Studies**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.12358v1)  

---


**ABSTRACT**  
Distributed ledger technologies have gained significant attention and adoption in recent years. Despite various security features distributed ledger technology provides, they are vulnerable to different and new malicious attacks, such as selfish mining and Sybil attacks. While such vulnerabilities have been investigated, detecting and discovering appropriate countermeasures still need to be reported. Cybersecurity knowledge is limited and fragmented in this domain, while distributed ledger technology usage grows daily. Thus, research focusing on overcoming potential attacks on distributed ledgers is required. This study aims to raise awareness of the cybersecurity of distributed ledger technology by designing a security risk assessment method for distributed ledger technology applications. We have developed a database with possible security threats and known attacks on distributed ledger technologies to accompany the method, including sets of countermeasures. We employed a semi-systematic literature review combined with method engineering to develop a method that organizations can use to assess their cybersecurity risk for distributed ledger applications. The method has subsequently been evaluated in three case studies, which show that the method helps to effectively conduct security risk assessments for distributed ledger applications in these organizations.

{{</citation>}}


### (40/119) The Ethics of Interaction: Mitigating Security Threats in LLMs (Ashutosh Kumar et al., 2024)

{{<citation>}}

Ashutosh Kumar, Sagarika Singh, Shiv Vignesh Murty, Swathy Ragupathy. (2024)  
**The Ethics of Interaction: Mitigating Security Threats in LLMs**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2401.12273v1)  

---


**ABSTRACT**  
This paper comprehensively explores the ethical challenges arising from security threats to Language Learning Models (LLMs). These intricate digital repositories are increasingly integrated into our daily lives, making them prime targets for attacks that can compromise their training data and the confidentiality of their data sources. The paper delves into the nuanced ethical repercussions of such security threats on society and individual privacy. We scrutinize five major threats: prompt injection, jailbreaking, Personal Identifiable Information (PII) exposure, sexually explicit content, and hate based content, going beyond mere identification to assess their critical ethical consequences and the urgency they create for robust defensive strategies. The escalating reliance on LLMs underscores the crucial need for ensuring these systems operate within the bounds of ethical norms, particularly as their misuse can lead to significant societal and individual harm. We propose conceptualizing and developing an evaluative tool tailored for LLMs, which would serve a dual purpose, guiding developers and designers in preemptive fortification of backend systems and scrutinizing the ethical dimensions of LLM chatbot responses during the testing phase. By comparing LLM responses with those expected from humans in a moral context, we aim to discern the degree to which AI behaviors align with the ethical values held by a broader society. Ultimately, this paper not only underscores the ethical troubles presented by LLMs, it also highlights a path toward cultivating trust in these systems.

{{</citation>}}


### (41/119) SEDAC: A CVAE-Based Data Augmentation Method for Security Bug Report Identification (Y. Liao et al., 2024)

{{<citation>}}

Y. Liao, T. Zhang. (2024)  
**SEDAC: A CVAE-Based Data Augmentation Method for Security Bug Report Identification**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Augmentation, BERT, Security  
[Paper Link](http://arxiv.org/abs/2401.12060v1)  

---


**ABSTRACT**  
Bug tracking systems store many bug reports, some of which are related to security. Identifying those security bug reports (SBRs) may help us predict some security-related bugs and solve security issues promptly so that the project can avoid threats and attacks. However, in the real world, the ratio of security bug reports is severely low; thus, directly training a prediction model with raw data may result in inaccurate results. Faced with the massive challenge of data imbalance, many researchers in the past have attempted to use text filtering or clustering methods to minimize the proportion of non-security bug reports (NSBRs) or apply oversampling methods to synthesize SBRs to make the dataset as balanced as possible. Nevertheless, there are still two challenges to those methods: 1) They ignore long-distance contextual information. 2) They fail to generate an utterly balanced dataset. To tackle these two challenges, we propose SEDAC, a new SBR identification method that generates similar bug report vectors to solve data imbalance problems and accurately detect security bug reports. Unlike previous studies, it first converts bug reports into individual bug report vectors with distilBERT, which are based on word2vec. Then, it trains a generative model through conditional variational auto-encoder (CVAE) to generate similar vectors with security labels, which makes the number of SBRs equal to NSBRs'. Finally, balanced data are used to train a security bug report classifier. To evaluate the effectiveness of our framework, we conduct it on 45,940 bug reports from Chromium and four Apache projects. The experimental results show that SEDAC outperforms all the baselines in g-measure with improvements of around 14.24%-50.10%.

{{</citation>}}


### (42/119) NEUROSEC: FPGA-Based Neuromorphic Audio Security (Murat Isik et al., 2024)

{{<citation>}}

Murat Isik, Hiruna Vishwamith, Yusuf Sur, Kayode Inadagbo, I. Can Dikmen. (2024)  
**NEUROSEC: FPGA-Based Neuromorphic Audio Security**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-ET, cs-LG, cs-NE, cs-SD, cs.CR, eess-AS  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.12055v1)  

---


**ABSTRACT**  
Neuromorphic systems, inspired by the complexity and functionality of the human brain, have gained interest in academic and industrial attention due to their unparalleled potential across a wide range of applications. While their capabilities herald innovation, it is imperative to underscore that these computational paradigms, analogous to their traditional counterparts, are not impervious to security threats. Although the exploration of neuromorphic methodologies for image and video processing has been rigorously pursued, the realm of neuromorphic audio processing remains in its early stages. Our results highlight the robustness and precision of our FPGA-based neuromorphic system. Specifically, our system showcases a commendable balance between desired signal and background noise, efficient spike rate encoding, and unparalleled resilience against adversarial attacks such as FGSM and PGD. A standout feature of our framework is its detection rate of 94%, which, when compared to other methodologies, underscores its greater capability in identifying and mitigating threats within 5.39 dB, a commendable SNR ratio. Furthermore, neuromorphic computing and hardware security serve many sensor domains in mission-critical and privacy-preserving applications.

{{</citation>}}


### (43/119) Effective Intrusion Detection in Heterogeneous Internet-of-Things Networks via Ensemble Knowledge Distillation-based Federated Learning (Jiyuan Shen et al., 2024)

{{<citation>}}

Jiyuan Shen, Wenzhuo Yang, Zhaowei Chu, Jiani Fan, Dusit Niyato, Kwok-Yan Lam. (2024)  
**Effective Intrusion Detection in Heterogeneous Internet-of-Things Networks via Ensemble Knowledge Distillation-based Federated Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Intrusion Detection, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.11968v1)  

---


**ABSTRACT**  
With the rapid development of low-cost consumer electronics and cloud computing, Internet-of-Things (IoT) devices are widely adopted for supporting next-generation distributed systems such as smart cities and industrial control systems. IoT devices are often susceptible to cyber attacks due to their open deployment environment and limited computing capabilities for stringent security controls. Hence, Intrusion Detection Systems (IDS) have emerged as one of the effective ways of securing IoT networks by monitoring and detecting abnormal activities. However, existing IDS approaches rely on centralized servers to generate behaviour profiles and detect anomalies, causing high response time and large operational costs due to communication overhead. Besides, sharing of behaviour data in an open and distributed IoT network environment may violate on-device privacy requirements. Additionally, various IoT devices tend to capture heterogeneous data, which complicates the training of behaviour models. In this paper, we introduce Federated Learning (FL) to collaboratively train a decentralized shared model of IDS, without exposing training data to others. Furthermore, we propose an effective method called Federated Learning Ensemble Knowledge Distillation (FLEKD) to mitigate the heterogeneity problems across various clients. FLEKD enables a more flexible aggregation method than conventional model fusion techniques. Experiment results on the public dataset CICIDS2019 demonstrate that the proposed approach outperforms local training and traditional FL in terms of both speed and performance and significantly improves the system's ability to detect unknown attacks. Finally, we evaluate our proposed framework's performance in three potential real-world scenarios and show FLEKD has a clear advantage in experimental results.

{{</citation>}}


### (44/119) GI-PIP: Do We Require Impractical Auxiliary Dataset for Gradient Inversion Attacks? (Yu sun et al., 2024)

{{<citation>}}

Yu sun, Gaojian Xiong, Xianxun Yao, Kailang Ma, Jian Cui. (2024)  
**GI-PIP: Do We Require Impractical Auxiliary Dataset for Gradient Inversion Attacks?**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.11748v2)  

---


**ABSTRACT**  
Deep gradient inversion attacks expose a serious threat to Federated Learning (FL) by accurately recovering private data from shared gradients. However, the state-of-the-art heavily relies on impractical assumptions to access excessive auxiliary data, which violates the basic data partitioning principle of FL. In this paper, a novel method, Gradient Inversion Attack using Practical Image Prior (GI-PIP), is proposed under a revised threat model. GI-PIP exploits anomaly detection models to capture the underlying distribution from fewer data, while GAN-based methods consume significant more data to synthesize images. The extracted distribution is then leveraged to regulate the attack process as Anomaly Score loss. Experimental results show that GI-PIP achieves a 16.12 dB PSNR recovery using only 3.8% data of ImageNet, while GAN-based methods necessitate over 70%. Moreover, GI-PIP exhibits superior capability on distribution generalization compared to GAN-based methods. Our approach significantly alleviates the auxiliary data requirement on both amount and distribution in gradient inversion attacks, hence posing more substantial threat to real-world FL.

{{</citation>}}


### (45/119) zkLogin: Privacy-Preserving Blockchain Authentication with Existing Credentials (Foteini Baldimtsi et al., 2024)

{{<citation>}}

Foteini Baldimtsi, Konstantinos Kryptos Chalkias, Yan Ji, Jonas Lindstrøm, Deepak Maram, Ben Riva, Arnab Roy, Mahdi Sedaghat, Joy Wang. (2024)  
**zkLogin: Privacy-Preserving Blockchain Authentication with Existing Credentials**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.11735v1)  

---


**ABSTRACT**  
For many users, a private key based wallet serves as the primary entry point to blockchains. Commonly recommended wallet authentication methods, such as mnemonics or hardware wallets, can be cumbersome. This difficulty in user onboarding has significantly hindered the adoption of blockchain-based applications.   We develop zkLogin, a novel technique that leverages identity tokens issued by popular platforms (any OpenID Connect enabled platform e.g. Google, Facebook, etc.) to authenticate transactions. At the heart of zkLogin lies a signature scheme allowing the signer to \textit{sign using their existing OpenID accounts} and nothing else. This improves the user experience significantly as users do not need to remember a new secret and can reuse their existing accounts.   zkLogin provides strong security and privacy guarantees. By design, zkLogin builds on top of the underlying platform's authentication mechanisms, and derives its security from there. Unlike prior related works however, zkLogin avoids the use of additional trusted parties (e.g., trusted hardware or oracles) for its security guarantees. zkLogin leverages zero-knowledge proofs (ZKP) to ensure that the link between a user's off-chain and on-chain identities is hidden, even from the platform itself.   We have implemented and deployed zkLogin on the Sui blockchain as an alternative to traditional digital signature-based addresses. Due to the ease of web3 on-boarding just with social login, without requiring mnemonics, many hundreds of thousands zkLogin accounts have already been generated in various industries such as gaming, DeFi, direct payments, NFT collections, ride sharing, sports racing and many more.

{{</citation>}}


### (46/119) Machine learning-based network intrusion detection for big and imbalanced data using oversampling, stacking feature embedding and feature extraction (Md. Alamin Talukder et al., 2024)

{{<citation>}}

Md. Alamin Talukder, Md. Manowarul Islam, Md Ashraf Uddin, Khondokar Fida Hasan, Selina Sharmin, Salem A. Alyami, Mohammad Ali Moni. (2024)  
**Machine learning-based network intrusion detection for big and imbalanced data using oversampling, stacking feature embedding and feature extraction**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Embedding, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2401.12262v1)  

---


**ABSTRACT**  
Cybersecurity has emerged as a critical global concern. Intrusion Detection Systems (IDS) play a critical role in protecting interconnected networks by detecting malicious actors and activities. Machine Learning (ML)-based behavior analysis within the IDS has considerable potential for detecting dynamic cyber threats, identifying abnormalities, and identifying malicious conduct within the network. However, as the number of data grows, dimension reduction becomes an increasingly difficult task when training ML models. Addressing this, our paper introduces a novel ML-based network intrusion detection model that uses Random Oversampling (RO) to address data imbalance and Stacking Feature Embedding based on clustering results, as well as Principal Component Analysis (PCA) for dimension reduction and is specifically designed for large and imbalanced datasets. This model's performance is carefully evaluated using three cutting-edge benchmark datasets: UNSW-NB15, CIC-IDS-2017, and CIC-IDS-2018. On the UNSW-NB15 dataset, our trials show that the RF and ET models achieve accuracy rates of 99.59% and 99.95%, respectively. Furthermore, using the CIC-IDS2017 dataset, DT, RF, and ET models reach 99.99% accuracy, while DT and RF models obtain 99.94% accuracy on CIC-IDS2018. These performance results continuously outperform the state-of-art, indicating significant progress in the field of network intrusion detection. This achievement demonstrates the efficacy of the suggested methodology, which can be used practically to accurately monitor and identify network traffic intrusions, thereby blocking possible threats.

{{</citation>}}


### (47/119) Analyzing the Quality Attributes of AI Vision Models in Open Repositories Under Adversarial Attacks (Zerui Wang et al., 2024)

{{<citation>}}

Zerui Wang, Yan Liu. (2024)  
**Analyzing the Quality Attributes of AI Vision Models in Open Repositories Under Adversarial Attacks**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, Adversarial Attack, Transformer  
[Paper Link](http://arxiv.org/abs/2401.12261v1)  

---


**ABSTRACT**  
As AI models rapidly evolve, they are frequently released to open repositories, such as HuggingFace. It is essential to perform quality assurance validation on these models before integrating them into the production development lifecycle. In addition to evaluating efficiency in terms of balanced accuracy and computing costs, adversarial attacks are potential threats to the robustness and explainability of AI models. Meanwhile, XAI applies algorithms that approximate inputs to outputs post-hoc to identify the contributing features. Adversarial perturbations may also degrade the utility of XAI explanations that require further investigation. In this paper, we present an integrated process designed for downstream evaluation tasks, including validating AI model accuracy, evaluating robustness with benchmark perturbations, comparing explanation utility, and assessing overhead. We demonstrate an evaluation scenario involving six computer vision models, which include CNN-based, Transformer-based, and hybrid architectures, three types of perturbations, and five XAI methods, resulting in ninety unique combinations. The process reveals the explanation utility among the XAI methods in terms of the identified key areas responding to the adversarial perturbation. The process produces aggregated results that illustrate multiple attributes of each AI model.

{{</citation>}}


## cs.IT (1)



### (48/119) Performance Analysis of 6G Multiuser Massive MIMO-OFDM THz Wireless Systems with Hybrid Beamforming under Intercarrier Interference (Md Saheed Ullah et al., 2024)

{{<citation>}}

Md Saheed Ullah, Zulqarnain Bin Ashraf, Sudipta Chandra Sarker. (2024)  
**Performance Analysis of 6G Multiuser Massive MIMO-OFDM THz Wireless Systems with Hybrid Beamforming under Intercarrier Interference**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12351v1)  

---


**ABSTRACT**  
6G networks are expected to provide more diverse capabilities than their predecessors and are likely to support applications beyond current mobile applications, such as virtual and augmented reality (VR/AR), AI, and the Internet of Things (IoT). In contrast to typical multiple-input multiple-output (MIMO) systems, THz MIMO precoding cannot be conducted totally at baseband using digital precoders due to the restricted number of signal mixers and analog-to-digital converters that can be supported due to their cost and power consumption. In this thesis, we analyzed the performance of multiuser massive MIMO-OFDM THz wireless systems with hybrid beamforming. Carrier frequency offset (CFO) is one of the most well-known disturbances for OFDM. For practicality, we accounted for CFO, which results in Intercarrier Interference. Incorporating the combined impact of molecular absorption, high sparsity, and multi-path fading, we analyzed a three-dimensional wideband THz channel and the carrier frequency offset in multi-carrier systems. With this model, we first presented a two-stage wideband hybrid beamforming technique comprising Riemannian manifolds optimization for analog beamforming and then a zero-forcing (ZF) approach for digital beamforming. We adjusted the objective function to reduce complexity, and instead of maximizing the bit rate, we determined parameters by minimizing interference. Numerical results demonstrate the significance of considering ICI for practical implementation for the THz system. We demonstrated how our change in problem formulation minimizes latency without compromising results. We also evaluated spectral efficiency by varying the number of RF chains and antennas. The spectral efficiency grows as the number of RF chains and antennas increases, but the spectral efficiency of antennas declines when the number of users increases.

{{</citation>}}


## cs.CV (27)



### (49/119) Scaling Up Quantization-Aware Neural Architecture Search for Efficient Deep Learning on the Edge (Yao Lu et al., 2024)

{{<citation>}}

Yao Lu, Hiram Rayo Torres Rodriguez, Sebastian Vogel, Nick van de Waterlaat, Pavol Jancura. (2024)  
**Scaling Up Quantization-Aware Neural Architecture Search for Efficient Deep Learning on the Edge**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: QA, Quantization  
[Paper Link](http://arxiv.org/abs/2401.12350v1)  

---


**ABSTRACT**  
Neural Architecture Search (NAS) has become the de-facto approach for designing accurate and efficient networks for edge devices. Since models are typically quantized for edge deployment, recent work has investigated quantization-aware NAS (QA-NAS) to search for highly accurate and efficient quantized models. However, existing QA-NAS approaches, particularly few-bit mixed-precision (FB-MP) methods, do not scale to larger tasks. Consequently, QA-NAS has mostly been limited to low-scale tasks and tiny networks. In this work, we present an approach to enable QA-NAS (INT8 and FB-MP) on large-scale tasks by leveraging the block-wise formulation introduced by block-wise NAS. We demonstrate strong results for the semantic segmentation task on the Cityscapes dataset, finding FB-MP models 33% smaller and INT8 models 17.6% faster than DeepLabV3 (INT8) without compromising task performance.

{{</citation>}}


### (50/119) OCT-SelfNet: A Self-Supervised Framework with Multi-Modal Datasets for Generalized and Robust Retinal Disease Detection (Fatema-E Jannat et al., 2024)

{{<citation>}}

Fatema-E Jannat, Sina Gholami, Minhaj Nur Alam, Hamed Tabkhi. (2024)  
**OCT-SelfNet: A Self-Supervised Framework with Multi-Modal Datasets for Generalized and Robust Retinal Disease Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.12344v1)  

---


**ABSTRACT**  
Despite the revolutionary impact of AI and the development of locally trained algorithms, achieving widespread generalized learning from multi-modal data in medical AI remains a significant challenge. This gap hinders the practical deployment of scalable medical AI solutions. Addressing this challenge, our research contributes a self-supervised robust machine learning framework, OCT-SelfNet, for detecting eye diseases using optical coherence tomography (OCT) images. In this work, various data sets from various institutions are combined enabling a more comprehensive range of representation. Our method addresses the issue using a two-phase training approach that combines self-supervised pretraining and supervised fine-tuning with a mask autoencoder based on the SwinV2 backbone by providing a solution for real-world clinical deployment. Extensive experiments on three datasets with different encoder backbones, low data settings, unseen data settings, and the effect of augmentation show that our method outperforms the baseline model, Resnet-50 by consistently attaining AUC-ROC performance surpassing 77% across all tests, whereas the baseline model exceeds 54%. Moreover, in terms of the AUC-PR metric, our proposed method exceeded 42%, showcasing a substantial increase of at least 10% in performance compared to the baseline, which exceeded only 33%. This contributes to our understanding of our approach's potential and emphasizes its usefulness in clinical settings.

{{</citation>}}


### (51/119) Contrastive Learning and Cycle Consistency-based Transductive Transfer Learning for Target Annotation (Shoaib Meraj Sami et al., 2024)

{{<citation>}}

Shoaib Meraj Sami, Md Mahedi Hasan, Nasser M. Nasrabadi, Raghuveer Rao. (2024)  
**Contrastive Learning and Cycle Consistency-based Transductive Transfer Learning for Target Annotation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, eess-IV, stat-ML  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.12340v1)  

---


**ABSTRACT**  
Annotating automatic target recognition (ATR) is a highly challenging task, primarily due to the unavailability of labeled data in the target domain. Hence, it is essential to construct an optimal target domain classifier by utilizing the labeled information of the source domain images. The transductive transfer learning (TTL) method that incorporates a CycleGAN-based unpaired domain translation network has been previously proposed in the literature for effective ATR annotation. Although this method demonstrates great potential for ATR, it severely suffers from lower annotation performance, higher Fr\'echet Inception Distance (FID) score, and the presence of visual artifacts in the synthetic images. To address these issues, we propose a hybrid contrastive learning base unpaired domain translation (H-CUT) network that achieves a significantly lower FID score. It incorporates both attention and entropy to emphasize the domain-specific region, a noisy feature mixup module to generate high variational synthetic negative patches, and a modulated noise contrastive estimation (MoNCE) loss to reweight all negative patches using optimal transport for better performance. Our proposed contrastive learning and cycle-consistency-based TTL (C3TTL) framework consists of two H-CUT networks and two classifiers. It simultaneously optimizes cycle-consistency, MoNCE, and identity losses. In C3TTL, two H-CUT networks have been employed through a bijection mapping to feed the reconstructed source domain images into a pretrained classifier to guide the optimal target domain classifier. Extensive experimental analysis conducted on three ATR datasets demonstrates that the proposed C3TTL method is effective in annotating civilian and military vehicles, as well as ship targets.

{{</citation>}}


### (52/119) Exploring Simple Open-Vocabulary Semantic Segmentation (Zihang Lai, 2024)

{{<citation>}}

Zihang Lai. (2024)  
**Exploring Simple Open-Vocabulary Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.12217v1)  

---


**ABSTRACT**  
Open-vocabulary semantic segmentation models aim to accurately assign a semantic label to each pixel in an image from a set of arbitrary open-vocabulary texts. In order to learn such pixel-level alignment, current approaches typically rely on a combination of (i) image-level VL model (e.g. CLIP), (ii) ground truth masks, and (iii) custom grouping encoders. In this paper, we introduce S-Seg, a novel model that can achieve surprisingly strong performance without depending on any of the above elements. S-Seg leverages pseudo-mask and language to train a MaskFormer, and can be easily trained from publicly available image-text datasets. Contrary to prior works, our model directly trains for pixel-level features and language alignment. Once trained, S-Seg generalizes well to multiple testing datasets without requiring fine-tuning. In addition, S-Seg has the extra benefits of scalability with data and consistently improvement when augmented with self-training. We believe that our simple yet effective approach will serve as a solid baseline for future research.

{{</citation>}}


### (53/119) Connecting the Dots: Leveraging Spatio-Temporal Graph Neural Networks for Accurate Bangla Sign Language Recognition (Haz Sameen Shahgir et al., 2024)

{{<citation>}}

Haz Sameen Shahgir, Khondker Salman Sayeed, Md Toki Tahmid, Tanjeem Azwad Zaman, Md. Zarif Ul Alam. (2024)  
**Connecting the Dots: Leveraging Spatio-Temporal Graph Neural Networks for Accurate Bangla Sign Language Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.12210v1)  

---


**ABSTRACT**  
Recent advances in Deep Learning and Computer Vision have been successfully leveraged to serve marginalized communities in various contexts. One such area is Sign Language - a primary means of communication for the deaf community. However, so far, the bulk of research efforts and investments have gone into American Sign Language, and research activity into low-resource sign languages - especially Bangla Sign Language - has lagged significantly. In this research paper, we present a new word-level Bangla Sign Language dataset - BdSL40 - consisting of 611 videos over 40 words, along with two different approaches: one with a 3D Convolutional Neural Network model and another with a novel Graph Neural Network approach for the classification of BdSL40 dataset. This is the first study on word-level BdSL recognition, and the dataset was transcribed from Indian Sign Language (ISL) using the Bangla Sign Language Dictionary (1997). The proposed GNN model achieved an F1 score of 89%. The study highlights the significant lexical and semantic similarity between BdSL, West Bengal Sign Language, and ISL, and the lack of word-level datasets for BdSL in the literature. We release the dataset and source code to stimulate further research.

{{</citation>}}


### (54/119) SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities (Boyuan Chen et al., 2024)

{{<citation>}}

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brian Ichter, Danny Driess, Pete Florence, Dorsa Sadigh, Leonidas Guibas, Fei Xia. (2024)  
**SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Language Model, QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.12168v1)  

---


**ABSTRACT**  
Understanding and reasoning about spatial relationships is a fundamental capability for Visual Question Answering (VQA) and robotics. While Vision Language Models (VLM) have demonstrated remarkable performance in certain VQA benchmarks, they still lack capabilities in 3D spatial reasoning, such as recognizing quantitative relationships of physical objects like distances or size differences. We hypothesize that VLMs' limited spatial reasoning capability is due to the lack of 3D spatial knowledge in training data and aim to solve this problem by training VLMs with Internet-scale spatial reasoning data. To this end, we present a system to facilitate this approach. We first develop an automatic 3D spatial VQA data generation framework that scales up to 2 billion VQA examples on 10 million real-world images. We then investigate various factors in the training recipe, including data quality, training pipeline, and VLM architecture. Our work features the first internet-scale 3D spatial reasoning dataset in metric space. By training a VLM on such data, we significantly enhance its ability on both qualitative and quantitative spatial VQA. Finally, we demonstrate that this VLM unlocks novel downstream applications in chain-of-thought spatial reasoning and robotics due to its quantitative estimation capability. Project website: https://spatial-vlm.github.io/

{{</citation>}}


### (55/119) Automated facial recognition system using deep learning for pain assessment in adults with cerebral palsy (Álvaro Sabater-Gárriz et al., 2024)

{{<citation>}}

Álvaro Sabater-Gárriz, F. Xavier Gaya-Morey, José María Buades-Rubio, Cristina Manresa Yee, Pedro Montoya, Inmaculada Riquelme. (2024)  
**Automated facial recognition system using deep learning for pain assessment in adults with cerebral palsy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12161v1)  

---


**ABSTRACT**  
Background: Pain assessment in individuals with neurological conditions, especially those with limited self-report ability and altered facial expressions, presents challenges. Existing measures, relying on direct observation by caregivers, lack sensitivity and specificity. In cerebral palsy, pain is a common comorbidity and a reliable evaluation protocol is crucial. Thus, having an automatic system that recognizes facial expressions could be of enormous help when diagnosing pain in this type of patient.   Objectives: 1) to build a dataset of facial pain expressions in individuals with cerebral palsy, and 2) to develop an automated facial recognition system based on deep learning for pain assessment addressed to this population.   Methods: Ten neural networks were trained on three pain image databases, including the UNBC-McMaster Shoulder Pain Expression Archive Database, the Multimodal Intensity Pain Dataset, and the Delaware Pain Database. Additionally, a curated dataset (CPPAIN) was created, consisting of 109 preprocessed facial pain expression images from individuals with cerebral palsy, categorized by two physiotherapists using the Facial Action Coding System observational scale.   Results: InceptionV3 exhibited promising performance on the CP-PAIN dataset, achieving an accuracy of 62.67% and an F1 score of 61.12%. Explainable artificial intelligence techniques revealed consistent essential features for pain identification across models.   Conclusion: This study demonstrates the potential of deep learning models for robust pain detection in populations with neurological conditions and communication disabilities. The creation of a larger dataset specific to cerebral palsy would further enhance model accuracy, offering a valuable tool for discerning subtle and idiosyncratic pain expressions. The insights gained could extend to other complex neurological conditions.

{{</citation>}}


### (56/119) Feature Denoising Diffusion Model for Blind Image Quality Assessment (Xudong Li et al., 2024)

{{<citation>}}

Xudong Li, Jingyuan Zheng, Runze Hu, Yan Zhang, Ke Li, Yunhang Shen, Xiawu Zheng, Yutao Liu, ShengChuan Zhang, Pingyang Dai, Rongrong Ji. (2024)  
**Feature Denoising Diffusion Model for Blind Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.11949v1)  

---


**ABSTRACT**  
Blind Image Quality Assessment (BIQA) aims to evaluate image quality in line with human perception, without reference benchmarks. Currently, deep learning BIQA methods typically depend on using features from high-level tasks for transfer learning. However, the inherent differences between BIQA and these high-level tasks inevitably introduce noise into the quality-aware features. In this paper, we take an initial step towards exploring the diffusion model for feature denoising in BIQA, namely Perceptual Feature Diffusion for IQA (PFD-IQA), which aims to remove noise from quality-aware features. Specifically, (i) We propose a {Perceptual Prior Discovery and Aggregation module to establish two auxiliary tasks to discover potential low-level features in images that are used to aggregate perceptual text conditions for the diffusion model. (ii) We propose a Perceptual Prior-based Feature Refinement strategy, which matches noisy features to predefined denoising trajectories and then performs exact feature denoising based on text conditions. Extensive experiments on eight standard BIQA datasets demonstrate the superior performance to the state-of-the-art BIQA methods, i.e., achieving the PLCC values of 0.935 ( vs. 0.905 in KADID) and 0.922 ( vs. 0.894 in LIVEC).

{{</citation>}}


### (57/119) A Saliency Enhanced Feature Fusion based multiscale RGB-D Salient Object Detection Network (Rui Huang et al., 2024)

{{<citation>}}

Rui Huang, Qingyi Zhao, Yan Xing, Sihua Gao, Weifeng Xu, Yuxiang Zhang, Wei Fan. (2024)  
**A Saliency Enhanced Feature Fusion based multiscale RGB-D Salient Object Detection Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2401.11914v1)  

---


**ABSTRACT**  
Multiscale convolutional neural network (CNN) has demonstrated remarkable capabilities in solving various vision problems. However, fusing features of different scales alwaysresults in large model sizes, impeding the application of multiscale CNNs in RGB-D saliency detection. In this paper, we propose a customized feature fusion module, called Saliency Enhanced Feature Fusion (SEFF), for RGB-D saliency detection. SEFF utilizes saliency maps of the neighboring scales to enhance the necessary features for fusing, resulting in more representative fused features. Our multiscale RGB-D saliency detector uses SEFF and processes images with three different scales. SEFF is used to fuse the features of RGB and depth images, as well as the features of decoders at different scales. Extensive experiments on five benchmark datasets have demonstrated the superiority of our method over ten SOTA saliency detectors.

{{</citation>}}


### (58/119) Detect-Order-Construct: A Tree Construction based Approach for Hierarchical Document Structure Analysis (Jiawei Wang et al., 2024)

{{<citation>}}

Jiawei Wang, Kai Hu, Zhuoyao Zhong, Lei Sun, Qiang Huo. (2024)  
**Detect-Order-Construct: A Tree Construction based Approach for Hierarchical Document Structure Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2401.11874v1)  

---


**ABSTRACT**  
Document structure analysis (aka document layout analysis) is crucial for understanding the physical layout and logical structure of documents, with applications in information retrieval, document summarization, knowledge extraction, etc. In this paper, we concentrate on Hierarchical Document Structure Analysis (HDSA) to explore hierarchical relationships within structured documents created using authoring software employing hierarchical schemas, such as LaTeX, Microsoft Word, and HTML. To comprehensively analyze hierarchical document structures, we propose a tree construction based approach that addresses multiple subtasks concurrently, including page object detection (Detect), reading order prediction of identified objects (Order), and the construction of intended hierarchical structure (Construct). We present an effective end-to-end solution based on this framework to demonstrate its performance. To assess our approach, we develop a comprehensive benchmark called Comp-HRDoc, which evaluates the above subtasks simultaneously. Our end-to-end system achieves state-of-the-art performance on two large-scale document layout analysis datasets (PubLayNet and DocLayNet), a high-quality hierarchical document structure reconstruction dataset (HRDoc), and our Comp-HRDoc benchmark. The Comp-HRDoc benchmark will be released to facilitate further research in this field.

{{</citation>}}


### (59/119) SignVTCL: Multi-Modal Continuous Sign Language Recognition Enhanced by Visual-Textual Contrastive Learning (Hao Chen et al., 2024)

{{<citation>}}

Hao Chen, Jiaze Wang, Ziyu Guo, Jinpeng Li, Donghao Zhou, Bian Wu, Chenyong Guan, Guangyong Chen, Pheng-Ann Heng. (2024)  
**SignVTCL: Multi-Modal Continuous Sign Language Recognition Enhanced by Visual-Textual Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.11847v1)  

---


**ABSTRACT**  
Sign language recognition (SLR) plays a vital role in facilitating communication for the hearing-impaired community. SLR is a weakly supervised task where entire videos are annotated with glosses, making it challenging to identify the corresponding gloss within a video segment. Recent studies indicate that the main bottleneck in SLR is the insufficient training caused by the limited availability of large-scale datasets. To address this challenge, we present SignVTCL, a multi-modal continuous sign language recognition framework enhanced by visual-textual contrastive learning, which leverages the full potential of multi-modal data and the generalization ability of language model. SignVTCL integrates multi-modal data (video, keypoints, and optical flow) simultaneously to train a unified visual backbone, thereby yielding more robust visual representations. Furthermore, SignVTCL contains a visual-textual alignment approach incorporating gloss-level and sentence-level alignment to ensure precise correspondence between visual features and glosses at the level of individual glosses and sentence. Experimental results conducted on three datasets, Phoenix-2014, Phoenix-2014T, and CSL-Daily, demonstrate that SignVTCL achieves state-of-the-art results compared with previous methods.

{{</citation>}}


### (60/119) Unveiling the Human-like Similarities of Automatic Facial Expression Recognition: An Empirical Exploration through Explainable AI (F. Xavier Gaya-Morey et al., 2024)

{{<citation>}}

F. Xavier Gaya-Morey, Silvia Ramis-Guarinos, Cristina Manresa-Yee, Jose M. Buades-Rubio. (2024)  
**Unveiling the Human-like Similarities of Automatic Facial Expression Recognition: An Empirical Exploration through Explainable AI**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11835v1)  

---


**ABSTRACT**  
Facial expression recognition is vital for human behavior analysis, and deep learning has enabled models that can outperform humans. However, it is unclear how closely they mimic human processing. This study aims to explore the similarity between deep neural networks and human perception by comparing twelve different networks, including both general object classifiers and FER-specific models. We employ an innovative global explainable AI method to generate heatmaps, revealing crucial facial regions for the twelve networks trained on six facial expressions. We assess these results both quantitatively and qualitatively, comparing them to ground truth masks based on Friesen and Ekman's description and among them. We use Intersection over Union (IoU) and normalized correlation coefficients for comparisons. We generate 72 heatmaps to highlight critical regions for each expression and architecture. Qualitatively, models with pre-trained weights show more similarity in heatmaps compared to those without pre-training. Specifically, eye and nose areas influence certain facial expressions, while the mouth is consistently important across all models and expressions. Quantitatively, we find low average IoU values (avg. 0.2702) across all expressions and architectures. The best-performing architecture averages 0.3269, while the worst-performing one averages 0.2066. Dendrograms, built with the normalized correlation coefficient, reveal two main clusters for most expressions: models with pre-training and models without pre-training. Findings suggest limited alignment between human and AI facial expression recognition, with network architectures influencing the similarity, as similar architectures prioritize similar facial regions.

{{</citation>}}


### (61/119) Rethinking Centered Kernel Alignment in Knowledge Distillation (Zikai Zhou et al., 2024)

{{<citation>}}

Zikai Zhou, Yunhang Shen, Shitong Shao, Huanran Chen, Linrui Gong, Shaohui Lin. (2024)  
**Rethinking Centered Kernel Alignment in Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.11824v1)  

---


**ABSTRACT**  
Knowledge distillation has emerged as a highly effective method for bridging the representation discrepancy between large-scale models and lightweight models. Prevalent approaches involve leveraging appropriate metrics to minimize the divergence or distance between the knowledge extracted from the teacher model and the knowledge learned by the student model. Centered Kernel Alignment (CKA) is widely used to measure representation similarity and has been applied in several knowledge distillation methods. However, these methods are complex and fail to uncover the essence of CKA, thus not answering the question of how to use CKA to achieve simple and effective distillation properly. This paper first provides a theoretical perspective to illustrate the effectiveness of CKA, which decouples CKA to the upper bound of Maximum Mean Discrepancy~(MMD) and a constant term. Drawing from this, we propose a novel Relation-Centered Kernel Alignment~(RCKA) framework, which practically establishes a connection between CKA and MMD. Furthermore, we dynamically customize the application of CKA based on the characteristics of each task, with less computational source yet comparable performance than the previous methods. The extensive experiments on the CIFAR-100, ImageNet-1k, and MS-COCO demonstrate that our method achieves state-of-the-art performance on almost all teacher-student pairs for image classification and object detection, validating the effectiveness of our approaches.

{{</citation>}}


### (62/119) SemPLeS: Semantic Prompt Learning for Weakly-Supervised Semantic Segmentation (Ci-Siang Lin et al., 2024)

{{<citation>}}

Ci-Siang Lin, Chien-Yi Wang, Yu-Chiang Frank Wang, Min-Hung Chen. (2024)  
**SemPLeS: Semantic Prompt Learning for Weakly-Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.11791v1)  

---


**ABSTRACT**  
Weakly-Supervised Semantic Segmentation (WSSS) aims to train segmentation models using training image data with only image-level supervision. Since precise pixel-level annotations are not accessible, existing methods typically focus on producing pseudo masks for training segmentation models by refining CAM-like heatmaps. However, the produced heatmaps may only capture discriminative image regions of target object categories or the associated co-occurring backgrounds. To address the issues, we propose a Semantic Prompt Learning for WSSS (SemPLeS) framework, which learns to effectively prompt the CLIP space to enhance the semantic alignment between the segmented regions and the target object categories. More specifically, we propose Contrastive Prompt Learning and Class-associated Semantic Refinement to learn the prompts that adequately describe and suppress the image backgrounds associated with each target object category. In this way, our proposed framework is able to perform better semantic matching between object regions and the associated text labels, resulting in desired pseudo masks for training the segmentation model. The proposed SemPLeS framework achieves SOTA performance on the standard WSSS benchmarks, PASCAL VOC and MS COCO, and demonstrated interpretability with the semantic visualization of our learned prompts. The codes will be released.

{{</citation>}}


### (63/119) Deep Learning for Computer Vision based Activity Recognition and Fall Detection of the Elderly: a Systematic Review (F. Xavier Gaya-Morey et al., 2024)

{{<citation>}}

F. Xavier Gaya-Morey, Cristina Manresa-Yee, Jose M. Buades-Rubio. (2024)  
**Deep Learning for Computer Vision based Activity Recognition and Fall Detection of the Elderly: a Systematic Review**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2401.11790v1)  

---


**ABSTRACT**  
As the percentage of elderly people in developed countries increases worldwide, the healthcare of this collective is a worrying matter, especially if it includes the preservation of their autonomy. In this direction, many studies are being published on Ambient Assisted Living (AAL) systems, which help to reduce the preoccupations raised by the independent living of the elderly. In this study, a systematic review of the literature is presented on fall detection and Human Activity Recognition (HAR) for the elderly, as the two main tasks to solve to guarantee the safety of elderly people living alone. To address the current tendency to perform these two tasks, the review focuses on the use of Deep Learning (DL) based approaches on computer vision data. In addition, different collections of data like DL models, datasets or hardware (e.g. depth or thermal cameras) are gathered from the reviewed studies and provided for reference in future studies. Strengths and weaknesses of existing approaches are also discussed and, based on them, our recommendations for future works are provided.

{{</citation>}}


### (64/119) Collaborative Position Reasoning Network for Referring Image Segmentation (Jianjian Cao et al., 2024)

{{<citation>}}

Jianjian Cao, Beiya Dai, Yulin Li, Xiameng Qin, Jingdong Wang. (2024)  
**Collaborative Position Reasoning Network for Referring Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.11775v1)  

---


**ABSTRACT**  
Given an image and a natural language expression as input, the goal of referring image segmentation is to segment the foreground masks of the entities referred by the expression. Existing methods mainly focus on interactive learning between vision and language to enhance the multi-modal representations for global context reasoning. However, predicting directly in pixel-level space can lead to collapsed positioning and poor segmentation results. Its main challenge lies in how to explicitly model entity localization, especially for non-salient entities. In this paper, we tackle this problem by executing a Collaborative Position Reasoning Network (CPRN) via the proposed novel Row-and-Column interactive (RoCo) and Guided Holistic interactive (Holi) modules. Specifically, RoCo aggregates the visual features into the row- and column-wise features corresponding two directional axes respectively. It offers a fine-grained matching behavior that perceives the associations between the linguistic features and two decoupled visual features to perform position reasoning over a hierarchical space. Holi integrates features of the two modalities by a cross-modal attention mechanism, which suppresses the irrelevant redundancy under the guide of positioning information from RoCo. Thus, with the incorporation of RoCo and Holi modules, CPRN captures the visual details of position reasoning so that the model can achieve more accurate segmentation. To our knowledge, this is the first work that explicitly focuses on position reasoning modeling. We also validate the proposed method on three evaluation datasets. It consistently outperforms existing state-of-the-art methods.

{{</citation>}}


### (65/119) MetaSeg: Content-Aware Meta-Net for Omni-Supervised Semantic Segmentation (Shenwang Jiang et al., 2024)

{{<citation>}}

Shenwang Jiang, Jianan Li, Ying Wang, Wenxuan Wu, Jizhou Zhang, Bo Huang, Tingfa Xu. (2024)  
**MetaSeg: Content-Aware Meta-Net for Omni-Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.11738v1)  

---


**ABSTRACT**  
Noisy labels, inevitably existing in pseudo segmentation labels generated from weak object-level annotations, severely hampers model optimization for semantic segmentation. Previous works often rely on massive hand-crafted losses and carefully-tuned hyper-parameters to resist noise, suffering poor generalization capability and high model complexity. Inspired by recent advances in meta learning, we argue that rather than struggling to tolerate noise hidden behind clean labels passively, a more feasible solution would be to find out the noisy regions actively, so as to simply ignore them during model optimization. With this in mind, this work presents a novel meta learning based semantic segmentation method, MetaSeg, that comprises a primary content-aware meta-net (CAM-Net) to sever as a noise indicator for an arbitrary segmentation model counterpart. Specifically, CAM-Net learns to generate pixel-wise weights to suppress noisy regions with incorrect pseudo labels while highlighting clean ones by exploiting hybrid strengthened features from image content, providing straightforward and reliable guidance for optimizing the segmentation model. Moreover, to break the barrier of time-consuming training when applying meta learning to common large segmentation models, we further present a new decoupled training strategy that optimizes different model layers in a divide-and-conquer manner. Extensive experiments on object, medical, remote sensing and human segmentation shows that our method achieves superior performance, approaching that of fully supervised settings, which paves a new promising way for omni-supervised semantic segmentation.

{{</citation>}}


### (66/119) Augmenting Prototype Network with TransMix for Few-shot Hyperspectral Image Classification (Chun Liu et al., 2024)

{{<citation>}}

Chun Liu, Longwei Yang, Dongmei Dong, Zheng Li, Wei Yang, Zhigang Han, Jiayao Wang. (2024)  
**Augmenting Prototype Network with TransMix for Few-shot Hyperspectral Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2401.11724v1)  

---


**ABSTRACT**  
Few-shot hyperspectral image classification aims to identify the classes of each pixel in the images by only marking few of these pixels. And in order to obtain the spatial-spectral joint features of each pixel, the fixed-size patches centering around each pixel are often used for classification. However, observing the classification results of existing methods, we found that boundary patches corresponding to the pixels which are located at the boundary of the objects in the hyperspectral images, are hard to classify. These boundary patchs are mixed with multi-class spectral information. Inspired by this, we propose to augment the prototype network with TransMix for few-shot hyperspectrial image classification(APNT). While taking the prototype network as the backbone, it adopts the transformer as feature extractor to learn the pixel-to-pixel relation and pay different attentions to different pixels. At the same time, instead of directly using the patches which are cut from the hyperspectral images for training, it randomly mixs up two patches to imitate the boundary patches and uses the synthetic patches to train the model, with the aim to enlarge the number of hard training samples and enhance their diversity. And by following the data agumentation technique TransMix, the attention returned by the transformer is also used to mix up the labels of two patches to generate better labels for synthetic patches. Compared with existing methods, the proposed method has demonstrated sate of the art performance and better robustness for few-shot hyperspectral image classification in our experiments.

{{</citation>}}


### (67/119) SFC: Shared Feature Calibration in Weakly Supervised Semantic Segmentation (Xinqiao Zhao et al., 2024)

{{<citation>}}

Xinqiao Zhao, Feilong Tang, Xiaoyang Wang, Jimin Xiao. (2024)  
**SFC: Shared Feature Calibration in Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.11719v1)  

---


**ABSTRACT**  
Image-level weakly supervised semantic segmentation has received increasing attention due to its low annotation cost. Existing methods mainly rely on Class Activation Mapping (CAM) to obtain pseudo-labels for training semantic segmentation models. In this work, we are the first to demonstrate that long-tailed distribution in training data can cause the CAM calculated through classifier weights over-activated for head classes and under-activated for tail classes due to the shared features among head- and tail- classes. This degrades pseudo-label quality and further influences final semantic segmentation performance. To address this issue, we propose a Shared Feature Calibration (SFC) method for CAM generation. Specifically, we leverage the class prototypes that carry positive shared features and propose a Multi-Scaled Distribution-Weighted (MSDW) consistency loss for narrowing the gap between the CAMs generated through classifier weights and class prototypes during training. The MSDW loss counterbalances over-activation and under-activation by calibrating the shared features in head-/tail-class classifier weights. Experimental results show that our SFC significantly improves CAM boundaries and achieves new state-of-the-art performances. The project is available at https://github.com/Barrett-python/SFC.

{{</citation>}}


### (68/119) MsSVT++: Mixed-scale Sparse Voxel Transformer with Center Voting for 3D Object Detection (Jianan Li et al., 2024)

{{<citation>}}

Jianan Li, Shaocong Dong, Lihe Ding, Tingfa Xu. (2024)  
**MsSVT++: Mixed-scale Sparse Voxel Transformer with Center Voting for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11718v1)  

---


**ABSTRACT**  
Accurate 3D object detection in large-scale outdoor scenes, characterized by considerable variations in object scales, necessitates features rich in both long-range and fine-grained information. While recent detectors have utilized window-based transformers to model long-range dependencies, they tend to overlook fine-grained details. To bridge this gap, we propose MsSVT++, an innovative Mixed-scale Sparse Voxel Transformer that simultaneously captures both types of information through a divide-and-conquer approach. This approach involves explicitly dividing attention heads into multiple groups, each responsible for attending to information within a specific range. The outputs of these groups are subsequently merged to obtain final mixed-scale features. To mitigate the computational complexity associated with applying a window-based transformer in 3D voxel space, we introduce a novel Chessboard Sampling strategy and implement voxel sampling and gathering operations sparsely using a hash map. Moreover, an important challenge stems from the observation that non-empty voxels are primarily located on the surface of objects, which impedes the accurate estimation of bounding boxes. To overcome this challenge, we introduce a Center Voting module that integrates newly voted voxels enriched with mixed-scale contextual information towards the centers of the objects, thereby improving precise object localization. Extensive experiments demonstrate that our single-stage detector, built upon the foundation of MsSVT++, consistently delivers exceptional performance across diverse datasets.

{{</citation>}}


### (69/119) Medical Image Debiasing by Learning Adaptive Agreement from a Biased Council (Luyang Luo et al., 2024)

{{<citation>}}

Luyang Luo, Xin Huang, Minghao Wang, Zhuoyue Wan, Hao Chen. (2024)  
**Medical Image Debiasing by Learning Adaptive Agreement from a Biased Council**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.11713v1)  

---


**ABSTRACT**  
Deep learning could be prone to learning shortcuts raised by dataset bias and result in inaccurate, unreliable, and unfair models, which impedes its adoption in real-world clinical applications. Despite its significance, there is a dearth of research in the medical image classification domain to address dataset bias. Furthermore, the bias labels are often agnostic, as identifying biases can be laborious and depend on post-hoc interpretation. This paper proposes learning Adaptive Agreement from a Biased Council (Ada-ABC), a debiasing framework that does not rely on explicit bias labels to tackle dataset bias in medical images. Ada-ABC develops a biased council consisting of multiple classifiers optimized with generalized cross entropy loss to learn the dataset bias. A debiasing model is then simultaneously trained under the guidance of the biased council. Specifically, the debiasing model is required to learn adaptive agreement with the biased council by agreeing on the correctly predicted samples and disagreeing on the wrongly predicted samples by the biased council. In this way, the debiasing model could learn the target attribute on the samples without spurious correlations while also avoiding ignoring the rich information in samples with spurious correlations. We theoretically demonstrated that the debiasing model could learn the target features when the biased model successfully captures dataset bias. Moreover, to our best knowledge, we constructed the first medical debiasing benchmark from four datasets containing seven different bias scenarios. Our extensive experiments practically showed that our proposed Ada-ABC outperformed competitive approaches, verifying its effectiveness in mitigating dataset bias for medical image classification. The codes and organized benchmark datasets will be made publicly available.

{{</citation>}}


### (70/119) Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs (Ling Yang et al., 2024)

{{<citation>}}

Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, Bin Cui. (2024)  
**Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.11708v1)  

---


**ABSTRACT**  
Diffusion models have exhibit exceptional performance in text-to-image generation and editing. However, existing methods often face challenges when handling complex text prompts that involve multiple objects with multiple attributes and relationships. In this paper, we propose a brand new training-free text-to-image generation/editing framework, namely Recaption, Plan and Generate (RPG), harnessing the powerful chain-of-thought reasoning ability of multimodal LLMs to enhance the compositionality of text-to-image diffusion models. Our approach employs the MLLM as a global planner to decompose the process of generating complex images into multiple simpler generation tasks within subregions. We propose complementary regional diffusion to enable region-wise compositional generation. Furthermore, we integrate text-guided image generation and editing within the proposed RPG in a closed-loop fashion, thereby enhancing generalization ability. Extensive experiments demonstrate our RPG outperforms state-of-the-art text-to-image diffusion models, including DALL-E 3 and SDXL, particularly in multi-category object composition and text-image semantic alignment. Notably, our RPG framework exhibits wide compatibility with various MLLM architectures (e.g., MiniGPT-4) and diffusion backbones (e.g., ControlNet). Our code is available at: https://github.com/YangLing0818/RPG-DiffusionMaster

{{</citation>}}


### (71/119) MVSFormer++: Revealing the Devil in Transformer's Details for Multi-View Stereo (Chenjie Cao et al., 2024)

{{<citation>}}

Chenjie Cao, Xinlin Ren, Yanwei Fu. (2024)  
**MVSFormer++: Revealing the Devil in Transformer's Details for Multi-View Stereo**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.11673v1)  

---


**ABSTRACT**  
Recent advancements in learning-based Multi-View Stereo (MVS) methods have prominently featured transformer-based models with attention mechanisms. However, existing approaches have not thoroughly investigated the profound influence of transformers on different MVS modules, resulting in limited depth estimation capabilities. In this paper, we introduce MVSFormer++, a method that prudently maximizes the inherent characteristics of attention to enhance various components of the MVS pipeline. Formally, our approach involves infusing cross-view information into the pre-trained DINOv2 model to facilitate MVS learning. Furthermore, we employ different attention mechanisms for the feature encoder and cost volume regularization, focusing on feature and spatial aggregations respectively. Additionally, we uncover that some design details would substantially impact the performance of transformer modules in MVS, including normalized 3D positional encoding, adaptive attention scaling, and the position of layer normalization. Comprehensive experiments on DTU, Tanks-and-Temples, BlendedMVS, and ETH3D validate the effectiveness of the proposed method. Notably, MVSFormer++ achieves state-of-the-art performance on the challenging DTU and Tanks-and-Temples benchmarks.

{{</citation>}}


### (72/119) OnDev-LCT: On-Device Lightweight Convolutional Transformers towards federated learning (Chu Myaet Thwal et al., 2024)

{{<citation>}}

Chu Myaet Thwal, Minh N. H. Nguyen, Ye Lin Tun, Seong Tae Kim, My T. Thai, Choong Seon Hong. (2024)  
**OnDev-LCT: On-Device Lightweight Convolutional Transformers towards federated learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.11652v1)  

---


**ABSTRACT**  
Federated learning (FL) has emerged as a promising approach to collaboratively train machine learning models across multiple edge devices while preserving privacy. The success of FL hinges on the efficiency of participating models and their ability to handle the unique challenges of distributed learning. While several variants of Vision Transformer (ViT) have shown great potential as alternatives to modern convolutional neural networks (CNNs) for centralized training, the unprecedented size and higher computational demands hinder their deployment on resource-constrained edge devices, challenging their widespread application in FL. Since client devices in FL typically have limited computing resources and communication bandwidth, models intended for such devices must strike a balance between model size, computational efficiency, and the ability to adapt to the diverse and non-IID data distributions encountered in FL. To address these challenges, we propose OnDev-LCT: Lightweight Convolutional Transformers for On-Device vision tasks with limited training data and resources. Our models incorporate image-specific inductive biases through the LCT tokenizer by leveraging efficient depthwise separable convolutions in residual linear bottleneck blocks to extract local features, while the multi-head self-attention (MHSA) mechanism in the LCT encoder implicitly facilitates capturing global representations of images. Extensive experiments on benchmark image datasets indicate that our models outperform existing lightweight vision models while having fewer parameters and lower computational demands, making them suitable for FL scenarios with data heterogeneity and communication bottlenecks.

{{</citation>}}


### (73/119) PointGL: A Simple Global-Local Framework for Efficient Point Cloud Analysis (Jianan Li et al., 2024)

{{<citation>}}

Jianan Li, Jie Wang, Tingfa Xu. (2024)  
**PointGL: A Simple Global-Local Framework for Efficient Point Cloud Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.11650v1)  

---


**ABSTRACT**  
Efficient analysis of point clouds holds paramount significance in real-world 3D applications. Currently, prevailing point-based models adhere to the PointNet++ methodology, which involves embedding and abstracting point features within a sequence of spatially overlapping local point sets, resulting in noticeable computational redundancy. Drawing inspiration from the streamlined paradigm of pixel embedding followed by regional pooling in Convolutional Neural Networks (CNNs), we introduce a novel, uncomplicated yet potent architecture known as PointGL, crafted to facilitate efficient point cloud analysis. PointGL employs a hierarchical process of feature acquisition through two recursive steps. First, the Global Point Embedding leverages straightforward residual Multilayer Perceptrons (MLPs) to effectuate feature embedding for each individual point. Second, the novel Local Graph Pooling technique characterizes point-to-point relationships and abstracts regional representations through succinct local graphs. The harmonious fusion of one-time point embedding and parameter-free graph pooling contributes to PointGL's defining attributes of minimized model complexity and heightened efficiency. Our PointGL attains state-of-the-art accuracy on the ScanObjectNN dataset while exhibiting a runtime that is more than 5 times faster and utilizing only approximately 4% of the FLOPs and 30% of the parameters compared to the recent PointMLP model. The code for PointGL is available at https://github.com/Roywangj/PointGL.

{{</citation>}}


### (74/119) Friends Across Time: Multi-Scale Action Segmentation Transformer for Surgical Phase Recognition (Bokai Zhang et al., 2024)

{{<citation>}}

Bokai Zhang, Jiayuan Meng, Bin Cheng, Dean Biskup, Svetlana Petculescu, Angela Chapman. (2024)  
**Friends Across Time: Multi-Scale Action Segmentation Transformer for Surgical Phase Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.11644v1)  

---


**ABSTRACT**  
Automatic surgical phase recognition is a core technology for modern operating rooms and online surgical video assessment platforms. Current state-of-the-art methods use both spatial and temporal information to tackle the surgical phase recognition task. Building on this idea, we propose the Multi-Scale Action Segmentation Transformer (MS-AST) for offline surgical phase recognition and the Multi-Scale Action Segmentation Causal Transformer (MS-ASCT) for online surgical phase recognition. We use ResNet50 or EfficientNetV2-M for spatial feature extraction. Our MS-AST and MS-ASCT can model temporal information at different scales with multi-scale temporal self-attention and multi-scale temporal cross-attention, which enhances the capture of temporal relationships between frames and segments. We demonstrate that our method can achieve 95.26% and 96.15% accuracy on the Cholec80 dataset for online and offline surgical phase recognition, respectively, which achieves new state-of-the-art results. Our method can also achieve state-of-the-art results on non-medical datasets in the video action segmentation domain.

{{</citation>}}


### (75/119) Zoom-shot: Fast and Efficient Unsupervised Zero-Shot Transfer of CLIP to Vision Encoders with Multimodal Loss (Jordan Shipard et al., 2024)

{{<citation>}}

Jordan Shipard, Arnold Wiliem, Kien Nguyen Thanh, Wei Xiang, Clinton Fookes. (2024)  
**Zoom-shot: Fast and Efficient Unsupervised Zero-Shot Transfer of CLIP to Vision Encoders with Multimodal Loss**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.11633v1)  

---


**ABSTRACT**  
The fusion of vision and language has brought about a transformative shift in computer vision through the emergence of Vision-Language Models (VLMs). However, the resource-intensive nature of existing VLMs poses a significant challenge. We need an accessible method for developing the next generation of VLMs. To address this issue, we propose Zoom-shot, a novel method for transferring the zero-shot capabilities of CLIP to any pre-trained vision encoder. We do this by exploiting the multimodal information (i.e. text and image) present in the CLIP latent space through the use of specifically designed multimodal loss functions. These loss functions are (1) cycle-consistency loss and (2) our novel prompt-guided knowledge distillation loss (PG-KD). PG-KD combines the concept of knowledge distillation with CLIP's zero-shot classification, to capture the interactions between text and image features. With our multimodal losses, we train a $\textbf{linear mapping}$ between the CLIP latent space and the latent space of a pre-trained vision encoder, for only a $\textbf{single epoch}$. Furthermore, Zoom-shot is entirely unsupervised and is trained using $\textbf{unpaired}$ data. We test the zero-shot capabilities of a range of vision encoders augmented as new VLMs, on coarse and fine-grained classification datasets, outperforming the previous state-of-the-art in this problem domain. In our ablations, we find Zoom-shot allows for a trade-off between data and compute during training; and our state-of-the-art results can be obtained by reducing training from 20% to 1% of the ImageNet training data with 20 epochs. All code and models are available on GitHub.

{{</citation>}}


## cs.GT (1)



### (76/119) The outcomes of generative AI are exactly the Nash equilibria of a non-potential game (Boualem Djehiche et al., 2024)

{{<citation>}}

Boualem Djehiche, Hamidou Tembine. (2024)  
**The outcomes of generative AI are exactly the Nash equilibria of a non-potential game**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs.GT  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2401.12321v1)  

---


**ABSTRACT**  
In this article we show that the asymptotic outcomes of both shallow and deep neural networks such as those used in BloombergGPT to generate economic time series are exactly the Nash equilibria of a non-potential game. We then design and analyze deep neural network algorithms that converge to these equilibria. The methodology is extended to federated deep neural networks between clusters of regional servers and on-device clients. Finally, the variational inequalities behind large language models including encoder-decoder related transformers are established.

{{</citation>}}


## cs.RO (4)



### (77/119) Multi-Agent Dynamic Relational Reasoning for Social Robot Navigation (Jiachen Li et al., 2024)

{{<citation>}}

Jiachen Li, Chuanbo Hua, Hengbo Ma, Jinkyoo Park, Victoria Dax, Mykel J. Kochenderfer. (2024)  
**Multi-Agent Dynamic Relational Reasoning for Social Robot Navigation**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-LG, cs-MA, cs-RO, cs.RO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.12275v1)  

---


**ABSTRACT**  
Social robot navigation can be helpful in various contexts of daily life but requires safe human-robot interactions and efficient trajectory planning. While modeling pairwise relations has been widely studied in multi-agent interacting systems, the ability to capture larger-scale group-wise activities is limited. In this paper, we propose a systematic relational reasoning approach with explicit inference of the underlying dynamically evolving relational structures, and we demonstrate its effectiveness for multi-agent trajectory prediction and social robot navigation. In addition to the edges between pairs of nodes (i.e., agents), we propose to infer hyperedges that adaptively connect multiple nodes to enable group-wise reasoning in an unsupervised manner. Our approach infers dynamically evolving relation graphs and hypergraphs to capture the evolution of relations, which the trajectory predictor employs to generate future states. Meanwhile, we propose to regularize the sharpness and sparsity of the learned relations and the smoothness of the relation evolution, which proves to enhance training stability and model performance. The proposed approach is validated on synthetic crowd simulations and real-world benchmark datasets. Experiments demonstrate that the approach infers reasonable relations and achieves state-of-the-art prediction performance. In addition, we present a deep reinforcement learning (DRL) framework for social robot navigation, which incorporates relational reasoning and trajectory prediction systematically. In a group-based crowd simulation, our method outperforms the strongest baseline by a significant margin in terms of safety, efficiency, and social compliance in dense, interactive scenarios.

{{</citation>}}


### (78/119) OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics (Peiqi Liu et al., 2024)

{{<citation>}}

Peiqi Liu, Yaswanth Orru, Chris Paxton, Nur Muhammad Mahi Shafiullah, Lerrel Pinto. (2024)  
**OK-Robot: What Really Matters in Integrating Open-Knowledge Models for Robotics**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.12202v1)  

---


**ABSTRACT**  
Remarkable progress has been made in recent years in the fields of vision, language, and robotics. We now have vision models capable of recognizing objects based on language queries, navigation systems that can effectively control mobile systems, and grasping models that can handle a wide range of objects. Despite these advancements, general-purpose applications of robotics still lag behind, even though they rely on these fundamental capabilities of recognition, navigation, and grasping. In this paper, we adopt a systems-first approach to develop a new Open Knowledge-based robotics framework called OK-Robot. By combining Vision-Language Models (VLMs) for object detection, navigation primitives for movement, and grasping primitives for object manipulation, OK-Robot offers a integrated solution for pick-and-drop operations without requiring any training. To evaluate its performance, we run OK-Robot in 10 real-world home environments. The results demonstrate that OK-Robot achieves a 58.5% success rate in open-ended pick-and-drop tasks, representing a new state-of-the-art in Open Vocabulary Mobile Manipulation (OVMM) with nearly 1.8x the performance of prior work. On cleaner, uncluttered environments, OK-Robot's performance increases to 82%. However, the most important insight gained from OK-Robot is the critical role of nuanced details when combining Open Knowledge systems like VLMs with robotic modules. Videos of our experiments are available on our website: https://ok-robot.github.io

{{</citation>}}


### (79/119) Multimodal Visual-Tactile Representation Learning through Self-Supervised Contrastive Pre-Training (Vedant Dave et al., 2024)

{{<citation>}}

Vedant Dave, Fotios Lygerakis, Elmar Rueckert. (2024)  
**Multimodal Visual-Tactile Representation Learning through Self-Supervised Contrastive Pre-Training**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.12024v1)  

---


**ABSTRACT**  
The rapidly evolving field of robotics necessitates methods that can facilitate the fusion of multiple modalities. Specifically, when it comes to interacting with tangible objects, effectively combining visual and tactile sensory data is key to understanding and navigating the complex dynamics of the physical world, enabling a more nuanced and adaptable response to changing environments. Nevertheless, much of the earlier work in merging these two sensory modalities has relied on supervised methods utilizing datasets labeled by humans.This paper introduces MViTac, a novel methodology that leverages contrastive learning to integrate vision and touch sensations in a self-supervised fashion. By availing both sensory inputs, MViTac leverages intra and inter-modality losses for learning representations, resulting in enhanced material property classification and more adept grasping prediction. Through a series of experiments, we showcase the effectiveness of our method and its superiority over existing state-of-the-art self-supervised and supervised techniques. In evaluating our methodology, we focus on two distinct tasks: material classification and grasping success prediction. Our results indicate that MViTac facilitates the development of improved modality encoders, yielding more robust representations as evidenced by linear probing assessments.

{{</citation>}}


### (80/119) Safe and Generalized end-to-end Autonomous Driving System with Reinforcement Learning and Demonstrations (Zuojin Tang et al., 2024)

{{<citation>}}

Zuojin Tang, Xiaoyu Chen, YongQiang Li, Jianyu Chen. (2024)  
**Safe and Generalized end-to-end Autonomous Driving System with Reinforcement Learning and Demonstrations**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11792v3)  

---


**ABSTRACT**  
An intelligent driving system should be capable of dynamically formulating appropriate driving strategies based on the current environment and vehicle status, while ensuring the security and reliability of the system. However, existing methods based on reinforcement learning and imitation learning suffer from low safety, poor generalization, and inefficient sampling. Additionally, they cannot accurately predict future driving trajectories, and the accurate prediction of future driving trajectories is a precondition for making optimal decisions. To solve these problems, in this paper, we introduce a Safe and Generalized end-to-end Autonomous Driving System (SGADS) for complex and various scenarios. Our SGADS incorporates variational inference with normalizing flows, enabling the intelligent vehicle to accurately predict future driving trajectories. Moreover, we propose the formulation of robust safety constraints. Furthermore, we combine reinforcement learning with demonstrations to augment search process of the agent. The experimental results demonstrate that our SGADS can significantly improve safety performance, exhibit strong generalization, and enhance the training efficiency of intelligent vehicles in complex urban scenarios compared to existing methods.

{{</citation>}}


## cs.LG (17)



### (81/119) Retrieval-Guided Reinforcement Learning for Boolean Circuit Minimization (Animesh Basak Chowdhury et al., 2024)

{{<citation>}}

Animesh Basak Chowdhury, Marco Romanelli, Benjamin Tan, Ramesh Karri, Siddharth Garg. (2024)  
**Retrieval-Guided Reinforcement Learning for Boolean Circuit Minimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-AR, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12205v1)  

---


**ABSTRACT**  
Logic synthesis, a pivotal stage in chip design, entails optimizing chip specifications encoded in hardware description languages like Verilog into highly efficient implementations using Boolean logic gates. The process involves a sequential application of logic minimization heuristics (``synthesis recipe"), with their arrangement significantly impacting crucial metrics such as area and delay. Addressing the challenge posed by the broad spectrum of design complexities - from variations of past designs (e.g., adders and multipliers) to entirely novel configurations (e.g., innovative processor instructions) - requires a nuanced `synthesis recipe` guided by human expertise and intuition. This study conducts a thorough examination of learning and search techniques for logic synthesis, unearthing a surprising revelation: pre-trained agents, when confronted with entirely novel designs, may veer off course, detrimentally affecting the search trajectory. We present ABC-RL, a meticulously tuned $\alpha$ parameter that adeptly adjusts recommendations from pre-trained agents during the search process. Computed based on similarity scores through nearest neighbor retrieval from the training dataset, ABC-RL yields superior synthesis recipes tailored for a wide array of hardware designs. Our findings showcase substantial enhancements in the Quality-of-result (QoR) of synthesized circuits, boasting improvements of up to 24.8% compared to state-of-the-art techniques. Furthermore, ABC-RL achieves an impressive up to 9x reduction in runtime (iso-QoR) when compared to current state-of-the-art methodologies.

{{</citation>}}


### (82/119) Universal Neurons in GPT2 Language Models (Wes Gurnee et al., 2024)

{{<citation>}}

Wes Gurnee, Theo Horsley, Zifan Carl Guo, Tara Rezaei Kheirkhah, Qinyi Sun, Will Hathaway, Neel Nanda, Dimitris Bertsimas. (2024)  
**Universal Neurons in GPT2 Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.12181v1)  

---


**ABSTRACT**  
A basic question within the emerging field of mechanistic interpretability is the degree to which neural networks learn the same underlying mechanisms. In other words, are neural mechanisms universal across different models? In this work, we study the universality of individual neurons across GPT2 models trained from different initial random seeds, motivated by the hypothesis that universal neurons are likely to be interpretable. In particular, we compute pairwise correlations of neuron activations over 100 million tokens for every neuron pair across five different seeds and find that 1-5\% of neurons are universal, that is, pairs of neurons which consistently activate on the same inputs. We then study these universal neurons in detail, finding that they usually have clear interpretations and taxonomize them into a small number of neuron families. We conclude by studying patterns in neuron weights to establish several universal functional roles of neurons in simple circuits: deactivating attention heads, changing the entropy of the next token distribution, and predicting the next token to (not) be within a particular set.

{{</citation>}}


### (83/119) Evaluation of QCNN-LSTM for Disability Forecasting in Multiple Sclerosis Using Sequential Multisequence MRI (John D. Mayfield et al., 2024)

{{<citation>}}

John D. Mayfield, Issam El Naqa. (2024)  
**Evaluation of QCNN-LSTM for Disability Forecasting in Multiple Sclerosis Using Sequential Multisequence MRI**  

---
Primary Category: cs.LG  
Categories: I-2-0; I-2-6, cs-AI, cs-ET, cs-LG, cs.LG, eess-IV  
Keywords: Clinical, LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2401.12132v1)  

---


**ABSTRACT**  
Introduction Quantum Convolutional Neural Network (QCNN)-Long Short-Term Memory (LSTM) models were studied to provide sequential relationships for each timepoint in MRIs of patients with Multiple Sclerosis (MS). In this pilot study, we compared three QCNN-LSTM models for binary classification of MS disability benchmarked against classical neural network architectures. Our hypothesis is that quantum models will provide competitive performance. Methods Matrix Product State (MPS), reverse Multistate Entanglement Renormalization Ansatz (MERA), and Tree-Tensor Network (TTN) circuits were paired with LSTM layer to process near-annual MRI data of patients diagnosed with MS. These were benchmarked against a Visual Geometry Group (VGG)-LSTM and a Video Vision Transformer (ViViT). Predicted logits were measured against ground truth labels of each patient's Extended Disability Severity Score (EDSS) using binary cross-entropy loss. Training/validation/holdout testing was partitioned using 5-fold cross validation with a total split of 60:20:20. Levene's test of variance was used to measure statistical difference and Student's t-test for paired model differences in mean. Results The MPS-LSTM, reverse MERA-LSTM, and TTN-LSTM had holdout testing ROC-AUC of 0.70, 0.77, and 0.81, respectively (p-value 0.915). VGG16-LSTM and ViViT performed similarly with ROC-AUC of 0.73 and 0.77, respectively (p-value 0.631). Overall variance and mean were not statistically significant (p-value 0.713), however, time to train was significantly faster for the QCNN-LSTMs (39.4 sec per fold vs. 224 and 218, respectively, p-value <0.001). Conclusion QCNN-LSTM models perform competitively to their classical counterparts with greater efficiency in train time. Clinically, these can add value in terms of efficiency to time-dependent deep learning prediction of disease progression based upon medical imaging.

{{</citation>}}


### (84/119) Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles (Maximilian Muschalik et al., 2024)

{{<citation>}}

Maximilian Muschalik, Fabian Fumagalli, Barbara Hammer, Eyke Hüllermeier. (2024)  
**Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12069v1)  

---


**ABSTRACT**  
While shallow decision trees may be interpretable, larger ensemble models like gradient-boosted trees, which often set the state of the art in machine learning problems involving tabular data, still remain black box models. As a remedy, the Shapley value (SV) is a well-known concept in explainable artificial intelligence (XAI) research for quantifying additive feature attributions of predictions. The model-specific TreeSHAP methodology solves the exponential complexity for retrieving exact SVs from tree-based models. Expanding beyond individual feature attribution, Shapley interactions reveal the impact of intricate feature interactions of any order. In this work, we present TreeSHAP-IQ, an efficient method to compute any-order additive Shapley interactions for predictions of tree-based models. TreeSHAP-IQ is supported by a mathematical framework that exploits polynomial arithmetic to compute the interaction scores in a single recursive traversal of the tree, akin to Linear TreeSHAP. We apply TreeSHAP-IQ on state-of-the-art tree ensembles and explore interactions on well-established benchmark datasets.

{{</citation>}}


### (85/119) Tensor-view Topological Graph Neural Network (Tao Wen et al., 2024)

{{<citation>}}

Tao Wen, Elynn Chen, Yuzhou Chen. (2024)  
**Tensor-view Topological Graph Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2401.12007v1)  

---


**ABSTRACT**  
Graph classification is an important learning task for graph-structured data. Graph neural networks (GNNs) have recently gained growing attention in graph learning and have shown significant improvements in many important graph problems. Despite their state-of-the-art performances, existing GNNs only use local information from a very limited neighborhood around each node, suffering from loss of multi-modal information and overheads of excessive computation. To address these issues, we propose a novel Tensor-view Topological Graph Neural Network (TTG-NN), a class of simple yet effective topological deep learning built upon persistent homology, graph convolution, and tensor operations. This new method incorporates tensor learning to simultaneously capture Tensor-view Topological (TT), as well as Tensor-view Graph (TG) structural information on both local and global levels. Computationally, to fully exploit graph topology and structure, we propose two flexible TT and TG representation learning modules that disentangle feature tensor aggregation and transformation and learn to preserve multi-modal structure with less computation. Theoretically, we derive high probability bounds on both the out-of-sample and in-sample mean squared approximation errors for our proposed Tensor Transformation Layer (TTL). Real data experiments show that the proposed TTG-NN outperforms 20 state-of-the-art methods on various graph benchmarks.

{{</citation>}}


### (86/119) The Bigger the Better? Rethinking the Effective Model Scale in Long-term Time Series Forecasting (Jinliang Deng et al., 2024)

{{<citation>}}

Jinliang Deng, Xuan Song, Ivor W. Tsang, Hui Xiong. (2024)  
**The Bigger the Better? Rethinking the Effective Model Scale in Long-term Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11929v1)  

---


**ABSTRACT**  
Long-term time series forecasting (LTSF) represents a critical frontier in time series analysis, distinguished by its focus on extensive input sequences, in contrast to the constrained lengths typical of traditional approaches. While longer sequences inherently convey richer information, potentially enhancing predictive precision, prevailing techniques often respond by escalating model complexity. These intricate models can inflate into millions of parameters, incorporating parameter-intensive elements like positional encodings, feed-forward networks and self-attention mechanisms. This complexity, however, leads to prohibitive model scale, particularly given the time series data's semantic simplicity. Motivated by the pursuit of parsimony, our research employs conditional correlation and auto-correlation as investigative tools, revealing significant redundancies within the input data. Leveraging these insights, we introduce the HDformer, a lightweight Transformer variant enhanced with hierarchical decomposition. This novel architecture not only inverts the prevailing trend toward model expansion but also accomplishes precise forecasting with drastically fewer computations and parameters. Remarkably, HDformer outperforms existing state-of-the-art LTSF models, while requiring over 99\% fewer parameters. Through this work, we advocate a paradigm shift in LTSF, emphasizing the importance to tailor the model to the inherent dynamics of time series data-a timely reminder that in the realm of LTSF, bigger is not invariably better.

{{</citation>}}


### (87/119) A Review of Physics-Informed Machine Learning Methods with Applications to Condition Monitoring and Anomaly Detection (Yuandi Wu et al., 2024)

{{<citation>}}

Yuandi Wu, Brett Sicard, Stephen Andrew Gadsden. (2024)  
**A Review of Physics-Informed Machine Learning Methods with Applications to Condition Monitoring and Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.11860v1)  

---


**ABSTRACT**  
This study presents a comprehensive overview of PIML techniques in the context of condition monitoring. The central concept driving PIML is the incorporation of known physical laws and constraints into machine learning algorithms, enabling them to learn from available data while remaining consistent with physical principles. Through fusing domain knowledge with data-driven learning, PIML methods offer enhanced accuracy and interpretability in comparison to purely data-driven approaches. In this comprehensive survey, detailed examinations are performed with regard to the methodology by which known physical principles are integrated within machine learning frameworks, as well as their suitability for specific tasks within condition monitoring. Incorporation of physical knowledge into the ML model may be realized in a variety of methods, with each having its unique advantages and drawbacks. The distinct advantages and limitations of each methodology for the integration of physics within data-driven models are detailed, considering factors such as computational efficiency, model interpretability, and generalizability to different systems in condition monitoring and fault detection. Several case studies and works of literature utilizing this emerging concept are presented to demonstrate the efficacy of PIML in condition monitoring applications. From the literature reviewed, the versatility and potential of PIML in condition monitoring may be demonstrated. Novel PIML methods offer an innovative solution for addressing the complexities of condition monitoring and associated challenges. This comprehensive survey helps form the foundation for future work in the field. As the technology continues to advance, PIML is expected to play a crucial role in enhancing maintenance strategies, system reliability, and overall operational efficiency in engineering systems.

{{</citation>}}


### (88/119) Self-Labeling the Job Shop Scheduling Problem (Andrea Corsini et al., 2024)

{{<citation>}}

Andrea Corsini, Angelo Porrello, Simone Calderara, Mauro Dell'Amico. (2024)  
**Self-Labeling the Job Shop Scheduling Problem**  

---
Primary Category: cs.LG  
Categories: I-2; G-2, cs-AI, cs-LG, cs.LG, math-CO  
Keywords: Reinforcement Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.11849v1)  

---


**ABSTRACT**  
In this work, we propose a Self-Supervised training strategy specifically designed for combinatorial problems. One of the main obstacles in applying supervised paradigms to such problems is the requirement of expensive target solutions as ground-truth, often produced with costly exact solvers. Inspired by Semi- and Self-Supervised learning, we show that it is possible to easily train generative models by sampling multiple solutions and using the best one according to the problem objective as a pseudo-label. In this way, we iteratively improve the model generation capability by relying only on its self-supervision, completely removing the need for optimality information. We prove the effectiveness of this Self-Labeling strategy on the Job Shop Scheduling (JSP), a complex combinatorial problem that is receiving much attention from the Reinforcement Learning community. We propose a generative model based on the well-known Pointer Network and train it with our strategy. Experiments on two popular benchmarks demonstrate the potential of this approach as the resulting models outperform constructive heuristics and current state-of-the-art Reinforcement Learning proposals.

{{</citation>}}


### (89/119) Learning to Approximate Adaptive Kernel Convolution on Graphs (Jaeyoon Sim et al., 2024)

{{<citation>}}

Jaeyoon Sim, Sooyeon Jeon, InJun Choi, Guorong Wu, Won Hwa Kim. (2024)  
**Learning to Approximate Adaptive Kernel Convolution on Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.11840v1)  

---


**ABSTRACT**  
Various Graph Neural Networks (GNNs) have been successful in analyzing data in non-Euclidean spaces, however, they have limitations such as oversmoothing, i.e., information becomes excessively averaged as the number of hidden layers increases. The issue stems from the intrinsic formulation of conventional graph convolution where the nodal features are aggregated from a direct neighborhood per layer across the entire nodes in the graph. As setting different number of hidden layers per node is infeasible, recent works leverage a diffusion kernel to redefine the graph structure and incorporate information from farther nodes. Unfortunately, such approaches suffer from heavy diagonalization of a graph Laplacian or learning a large transform matrix. In this regards, we propose a diffusion learning framework, where the range of feature aggregation is controlled by the scale of a diffusion kernel. For efficient computation, we derive closed-form derivatives of approximations of the graph convolution with respect to the scale, so that node-wise range can be adaptively learned. With a downstream classifier, the entire framework is made trainable in an end-to-end manner. Our model is tested on various standard datasets for node-wise classification for the state-of-the-art performance, and it is also validated on a real-world brain network data for graph classifications to demonstrate its practicality for Alzheimer classification.

{{</citation>}}


### (90/119) Knowledge Distillation on Spatial-Temporal Graph Convolutional Network for Traffic Prediction (Mohammad Izadi et al., 2024)

{{<citation>}}

Mohammad Izadi, Mehran Safayani, Abdolreza Mirzaei. (2024)  
**Knowledge Distillation on Spatial-Temporal Graph Convolutional Network for Traffic Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Convolutional Network, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.11798v2)  

---


**ABSTRACT**  
Efficient real-time traffic prediction is crucial for reducing transportation time. To predict traffic conditions, we employ a spatio-temporal graph neural network (ST-GNN) to model our real-time traffic data as temporal graphs. Despite its capabilities, it often encounters challenges in delivering efficient real-time predictions for real-world traffic data. Recognizing the significance of timely prediction due to the dynamic nature of real-time data, we employ knowledge distillation (KD) as a solution to enhance the execution time of ST-GNNs for traffic prediction. In this paper, We introduce a cost function designed to train a network with fewer parameters (the student) using distilled data from a complex network (the teacher) while maintaining its accuracy close to that of the teacher. We use knowledge distillation, incorporating spatial-temporal correlations from the teacher network to enable the student to learn the complex patterns perceived by the teacher. However, a challenge arises in determining the student network architecture rather than considering it inadvertently. To address this challenge, we propose an algorithm that utilizes the cost function to calculate pruning scores, addressing small network architecture search issues, and jointly fine-tunes the network resulting from each pruning stage using KD. Ultimately, we evaluate our proposed ideas on two real-world datasets, PeMSD7 and PeMSD8. The results indicate that our method can maintain the student's accuracy close to that of the teacher, even with the retention of only $3\%$ of network parameters.

{{</citation>}}


### (91/119) LightDiC: A Simple yet Effective Approach for Large-scale Digraph Representation Learning (Xunkai Li et al., 2024)

{{<citation>}}

Xunkai Li, Meihao Liao, Zhengyu Wu, Daohan Su, Wentao Zhang, Rong-Hua Li, Guoren Wang. (2024)  
**LightDiC: A Simple yet Effective Approach for Large-scale Digraph Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.11772v1)  

---


**ABSTRACT**  
Most existing graph neural networks (GNNs) are limited to undirected graphs, whose restricted scope of the captured relational information hinders their expressive capabilities and deployments in real-world scenarios. Compared with undirected graphs, directed graphs (digraphs) fit the demand for modeling more complex topological systems by capturing more intricate relationships between nodes, such as formulating transportation and financial networks. While some directed GNNs have been introduced, their inspiration mainly comes from deep learning architectures, which lead to redundant complexity and computation, making them inapplicable to large-scale databases. To address these issues, we propose LightDiC, a scalable variant of the digraph convolution based on the magnetic Laplacian. Since topology-related computations are conducted solely during offline pre-processing, LightDiC achieves exceptional scalability, enabling downstream predictions to be trained separately without incurring recursive computational costs. Theoretical analysis shows that LightDiC utilizes directed information to achieve message passing based on the complex field, which corresponds to the proximal gradient descent process of the Dirichlet energy optimization function from the perspective of digraph signal denoising, ensuring its expressiveness. Experimental results demonstrate that LightDiC performs comparably well or even outperforms other SOTA methods in various downstream tasks, with fewer learnable parameters and higher training efficiency. Notably, LightDiC is the first DiGNN to provide satisfactory results in the most representative large-scale database (ogbn-papers100M).

{{</citation>}}


### (92/119) ADA-GNN: Atom-Distance-Angle Graph Neural Network for Crystal Material Property Prediction (Jiao Huang et al., 2024)

{{<citation>}}

Jiao Huang, Qianli Xing, Jinglong Ji, Bo Yang. (2024)  
**ADA-GNN: Atom-Distance-Angle Graph Neural Network for Crystal Material Property Prediction**  

---
Primary Category: cs.LG  
Categories: cond-mat-mtrl-sci, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2401.11768v1)  

---


**ABSTRACT**  
Property prediction is a fundamental task in crystal material research. To model atoms and structures, structures represented as graphs are widely used and graph learning-based methods have achieved significant progress. Bond angles and bond distances are two key structural information that greatly influence crystal properties. However, most of the existing works only consider bond distances and overlook bond angles. The main challenge lies in the time cost of handling bond angles, which leads to a significant increase in inference time. To solve this issue, we first propose a crystal structure modeling based on dual scale neighbor partitioning mechanism, which uses a larger scale cutoff for edge neighbors and a smaller scale cutoff for angle neighbors. Then, we propose a novel Atom-Distance-Angle Graph Neural Network (ADA-GNN) for property prediction tasks, which can process node information and structural information separately. The accuracy of predictions and inference time are improved with the dual scale modeling and the specially designed architecture of ADA-GNN. The experimental results validate that our approach achieves state-of-the-art results in two large-scale material benchmark datasets on property prediction tasks.

{{</citation>}}


### (93/119) Towards Effective and General Graph Unlearning via Mutual Evolution (Xunkai Li et al., 2024)

{{<citation>}}

Xunkai Li, Yulin Zhao, Zhengyu Wu, Wentao Zhang, Rong-Hua Li, Guoren Wang. (2024)  
**Towards Effective and General Graph Unlearning via Mutual Evolution**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs-SI, cs.LG  
Keywords: AI, GNN  
[Paper Link](http://arxiv.org/abs/2401.11760v1)  

---


**ABSTRACT**  
With the rapid advancement of AI applications, the growing needs for data privacy and model robustness have highlighted the importance of machine unlearning, especially in thriving graph-based scenarios. However, most existing graph unlearning strategies primarily rely on well-designed architectures or manual process, rendering them less user-friendly and posing challenges in terms of deployment efficiency. Furthermore, striking a balance between unlearning performance and framework generalization is also a pivotal concern. To address the above issues, we propose \underline{\textbf{M}}utual \underline{\textbf{E}}volution \underline{\textbf{G}}raph \underline{\textbf{U}}nlearning (MEGU), a new mutual evolution paradigm that simultaneously evolves the predictive and unlearning capacities of graph unlearning. By incorporating aforementioned two components, MEGU ensures complementary optimization in a unified training framework that aligns with the prediction and unlearning requirements. Extensive experiments on 9 graph benchmark datasets demonstrate the superior performance of MEGU in addressing unlearning requirements at the feature, node, and edge levels. Specifically, MEGU achieves average performance improvements of 2.7\%, 2.5\%, and 3.2\% across these three levels of unlearning tasks when compared to state-of-the-art baselines. Furthermore, MEGU exhibits satisfactory training efficiency, reducing time and space overhead by an average of 159.8x and 9.6x, respectively, in comparison to retraining GNN from scratch.

{{</citation>}}


### (94/119) Attention on Personalized Clinical Decision Support System: Federated Learning Approach (Chu Myaet Thwal et al., 2024)

{{<citation>}}

Chu Myaet Thwal, Kyi Thar, Ye Lin Tun, Choong Seon Hong. (2024)  
**Attention on Personalized Clinical Decision Support System: Federated Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Clinical  
[Paper Link](http://arxiv.org/abs/2401.11736v1)  

---


**ABSTRACT**  
Health management has become a primary problem as new kinds of diseases and complex symptoms are introduced to a rapidly growing modern society. Building a better and smarter healthcare infrastructure is one of the ultimate goals of a smart city. To the best of our knowledge, neural network models are already employed to assist healthcare professionals in achieving this goal. Typically, training a neural network requires a rich amount of data but heterogeneous and vulnerable properties of clinical data introduce a challenge for the traditional centralized network. Moreover, adding new inputs to a medical database requires re-training an existing model from scratch. To tackle these challenges, we proposed a deep learning-based clinical decision support system trained and managed under a federated learning paradigm. We focused on a novel strategy to guarantee the safety of patient privacy and overcome the risk of cyberattacks while enabling large-scale clinical data mining. As a result, we can leverage rich clinical data for training each local neural network without the need for exchanging the confidential data of patients. Moreover, we implemented the proposed scheme as a sequence-to-sequence model architecture integrating the attention mechanism. Thus, our objective is to provide a personalized clinical decision support system with evolvable characteristics that can deliver accurate solutions and assist healthcare professionals in medical diagnosing.

{{</citation>}}


### (95/119) Graph Condensation: A Survey (Xinyi Gao et al., 2024)

{{<citation>}}

Xinyi Gao, Junliang Yu, Wei Jiang, Tong Chen, Wentao Zhang, Hongzhi Yin. (2024)  
**Graph Condensation: A Survey**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.11720v1)  

---


**ABSTRACT**  
The burgeoning volume of graph data poses significant challenges in storage, transmission, and particularly the training of graph neural networks (GNNs). To address these challenges, graph condensation (GC) has emerged as an innovative solution. GC focuses on synthesizing a compact yet highly representative graph, on which GNNs can achieve performance comparable to trained on the large original graph. The notable efficacy of GC and its broad prospects have garnered significant attention and spurred extensive research. This survey paper provides an up-to-date and systematic overview of GC, organizing existing research into four categories aligned with critical GC evaluation criteria: effectiveness, generalization, fairness, and efficiency. To facilitate an in-depth and comprehensive understanding of GC, we examine various methods under each category and thoroughly discuss two essential components within GC: optimization strategies and condensed graph generation. Additionally, we introduce the applications of GC in a variety of fields, and highlight the present challenges and novel insights in GC, promoting advancements in future research.

{{</citation>}}


### (96/119) P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer (Zhiyuan Wang et al., 2024)

{{<citation>}}

Zhiyuan Wang, Xiaoyang Qu, Jing Xiao, Bokui Chen, Jianzong Wang. (2024)  
**P2DT: Mitigating Forgetting in task-incremental Learning with progressive prompt Decision Transformer**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.11666v1)  

---


**ABSTRACT**  
Catastrophic forgetting poses a substantial challenge for managing intelligent agents controlled by a large model, causing performance degradation when these agents face new tasks. In our work, we propose a novel solution - the Progressive Prompt Decision Transformer (P2DT). This method enhances a transformer-based model by dynamically appending decision tokens during new task training, thus fostering task-specific policies. Our approach mitigates forgetting in continual and offline reinforcement learning scenarios. Moreover, P2DT leverages trajectories collected via traditional reinforcement learning from all tasks and generates new task-specific tokens during training, thereby retaining knowledge from previous studies. Preliminary results demonstrate that our model effectively alleviates catastrophic forgetting and scales well with increasing task environments.

{{</citation>}}


### (97/119) Zero-Space Cost Fault Tolerance for Transformer-based Language Models on ReRAM (Bingbing Li et al., 2024)

{{<citation>}}

Bingbing Li, Geng Yuan, Zigeng Wang, Shaoyi Huang, Hongwu Peng, Payman Behnam, Wujie Wen, Hang Liu, Caiwen Ding. (2024)  
**Zero-Space Cost Fault Tolerance for Transformer-based Language Models on ReRAM**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-AR, cs-LG, cs.LG  
Keywords: BERT, GLUE, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11664v1)  

---


**ABSTRACT**  
Resistive Random Access Memory (ReRAM) has emerged as a promising platform for deep neural networks (DNNs) due to its support for parallel in-situ matrix-vector multiplication. However, hardware failures, such as stuck-at-fault defects, can result in significant prediction errors during model inference. While additional crossbars can be used to address these failures, they come with storage overhead and are not efficient in terms of space, energy, and cost. In this paper, we propose a fault protection mechanism that incurs zero space cost. Our approach includes: 1) differentiable structure pruning of rows and columns to reduce model redundancy, 2) weight duplication and voting for robust output, and 3) embedding duplicated most significant bits (MSBs) into the model weight. We evaluate our method on nine tasks of the GLUE benchmark with the BERT model, and experimental results prove its effectiveness.

{{</citation>}}


## physics.bio-ph (1)



### (98/119) Learning Dynamics from Multicellular Graphs with Deep Neural Networks (Haiqian Yang et al., 2024)

{{<citation>}}

Haiqian Yang, Florian Meyer, Shaoxun Huang, Liu Yang, Cristiana Lungu, Monilola A. Olayioye, Markus J. Buehler, Ming Guo. (2024)  
**Learning Dynamics from Multicellular Graphs with Deep Neural Networks**  

---
Primary Category: physics.bio-ph  
Categories: cond-mat-soft, cs-LG, physics-bio-ph, physics.bio-ph  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.12196v1)  

---


**ABSTRACT**  
The inference of multicellular self-assembly is the central quest of understanding morphogenesis, including embryos, organoids, tumors, and many others. However, it has been tremendously difficult to identify structural features that can indicate multicellular dynamics. Here we propose to harness the predictive power of graph-based deep neural networks (GNN) to discover important graph features that can predict dynamics. To demonstrate, we apply a physically informed GNN (piGNN) to predict the motility of multicellular collectives from a snapshot of their positions both in experiments and simulations. We demonstrate that piGNN is capable of navigating through complex graph features of multicellular living systems, which otherwise can not be achieved by classical mechanistic models. With increasing amounts of multicellular data, we propose that collaborative efforts can be made to create a multicellular data bank (MDB) from which it is possible to construct a large multicellular graph model (LMGM) for general-purposed predictions of multicellular organization.

{{</citation>}}


## cs.MA (2)



### (99/119) Transcending To Notions (Sama Sai Karthik et al., 2024)

{{<citation>}}

Sama Sai Karthik, Jayati Deshmukh, Srinath Srinivasa. (2024)  
**Transcending To Notions**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12159v1)  

---


**ABSTRACT**  
Social identities play an important role in the dynamics of human societies, and it can be argued that some sense of identification with a larger cause or idea plays a critical role in making humans act responsibly. Often social activists strive to get populations to identify with some cause or notion -- like green energy, diversity, etc. in order to bring about desired social changes. We explore the problem of designing computational models for social identities in the context of autonomous AI agents. For this, we propose an agent model that enables agents to identify with certain notions and show how this affects collective outcomes. We also contrast between associations of identity with rational preferences. The proposed model is simulated in an application context of urban mobility, where we show how changes in social identity affect mobility patterns and collective outcomes.

{{</citation>}}


### (100/119) Collaborative Reinforcement Learning Based Unmanned Aerial Vehicle (UAV) Trajectory Design for 3D UAV Tracking (Yujiao Zhu et al., 2024)

{{<citation>}}

Yujiao Zhu, Mingzhe Chen, Sihua Wang, Ye Hu, Yuchen Liu, Changchuan Yin. (2024)  
**Collaborative Reinforcement Learning Based Unmanned Aerial Vehicle (UAV) Trajectory Design for 3D UAV Tracking**  

---
Primary Category: cs.MA  
Categories: cs-LG, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.12079v1)  

---


**ABSTRACT**  
In this paper, the problem of using one active unmanned aerial vehicle (UAV) and four passive UAVs to localize a 3D target UAV in real time is investigated. In the considered model, each passive UAV receives reflection signals from the target UAV, which are initially transmitted by the active UAV. The received reflection signals allow each passive UAV to estimate the signal transmission distance which will be transmitted to a base station (BS) for the estimation of the position of the target UAV. Due to the movement of the target UAV, each active/passive UAV must optimize its trajectory to continuously localize the target UAV. Meanwhile, since the accuracy of the distance estimation depends on the signal-to-noise ratio of the transmission signals, the active UAV must optimize its transmit power. This problem is formulated as an optimization problem whose goal is to jointly optimize the transmit power of the active UAV and trajectories of both active and passive UAVs so as to maximize the target UAV positioning accuracy. To solve this problem, a Z function decomposition based reinforcement learning (ZD-RL) method is proposed. Compared to value function decomposition based RL (VD-RL), the proposed method can find the probability distribution of the sum of future rewards to accurately estimate the expected value of the sum of future rewards thus finding better transmit power of the active UAV and trajectories for both active and passive UAVs and improving target UAV positioning accuracy. Simulation results show that the proposed ZD-RL method can reduce the positioning errors by up to 39.4% and 64.6%, compared to VD-RL and independent deep RL methods, respectively.

{{</citation>}}


## cs.HC (4)



### (101/119) VRMN-bD: A Multi-modal Natural Behavior Dataset of Immersive Human Fear Responses in VR Stand-up Interactive Games (He Zhang et al., 2024)

{{<citation>}}

He Zhang, Xinyang Li, Yuanxi Sun, Xinyi Fu, Christine Qiu, John M. Carroll. (2024)  
**VRMN-bD: A Multi-modal Natural Behavior Dataset of Immersive Human Fear Responses in VR Stand-up Interactive Games**  

---
Primary Category: cs.HC  
Categories: cs-CV, cs-HC, cs-LG, cs.HC  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.12133v1)  

---


**ABSTRACT**  
Understanding and recognizing emotions are important and challenging issues in the metaverse era. Understanding, identifying, and predicting fear, which is one of the fundamental human emotions, in virtual reality (VR) environments plays an essential role in immersive game development, scene development, and next-generation virtual human-computer interaction applications. In this article, we used VR horror games as a medium to analyze fear emotions by collecting multi-modal data (posture, audio, and physiological signals) from 23 players. We used an LSTM-based model to predict fear with accuracies of 65.31% and 90.47% under 6-level classification (no fear and five different levels of fear) and 2-level classification (no fear and fear), respectively. We constructed a multi-modal natural behavior dataset of immersive human fear responses (VRMN-bD) and compared it with existing relevant advanced datasets. The results show that our dataset has fewer limitations in terms of collection method, data scale and audience scope. We are unique and advanced in targeting multi-modal datasets of fear and behavior in VR stand-up interactive environments. Moreover, we discussed the implications of this work for communities and applications. The dataset and pre-trained model are available at https://github.com/KindOPSTAR/VRMN-bD.

{{</citation>}}


### (102/119) MINT: A wrapper to make multi-modal and multi-image AI models interactive (Jan Freyberg et al., 2024)

{{<citation>}}

Jan Freyberg, Abhijit Guha Roy, Terry Spitz, Beverly Freeman, Mike Schaekermann, Patricia Strachan, Eva Schnider, Renee Wong, Dale R Webster, Alan Karthikesalingam, Yun Liu, Krishnamurthy Dvijotham, Umesh Telang. (2024)  
**MINT: A wrapper to make multi-modal and multi-image AI models interactive**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.12032v1)  

---


**ABSTRACT**  
During the diagnostic process, doctors incorporate multimodal information including imaging and the medical history - and similarly medical AI development has increasingly become multimodal. In this paper we tackle a more subtle challenge: doctors take a targeted medical history to obtain only the most pertinent pieces of information; how do we enable AI to do the same? We develop a wrapper method named MINT (Make your model INTeractive) that automatically determines what pieces of information are most valuable at each step, and ask for only the most useful information. We demonstrate the efficacy of MINT wrapping a skin disease prediction model, where multiple images and a set of optional answers to $25$ standard metadata questions (i.e., structured medical history) are used by a multi-modal deep network to provide a differential diagnosis. We show that MINT can identify whether metadata inputs are needed and if so, which question to ask next. We also demonstrate that when collecting multiple images, MINT can identify if an additional image would be beneficial, and if so, which type of image to capture. We showed that MINT reduces the number of metadata and image inputs needed by 82% and 36.2% respectively, while maintaining predictive performance. Using real-world AI dermatology system data, we show that needing fewer inputs can retain users that may otherwise fail to complete the system submission and drop off without a diagnosis. Qualitative examples show MINT can closely mimic the step-by-step decision making process of a clinical workflow and how this is different for straight forward cases versus more difficult, ambiguous cases. Finally we demonstrate how MINT is robust to different underlying multi-model classifiers and can be easily adapted to user requirements without significant model re-training.

{{</citation>}}


### (103/119) VirtuWander: Enhancing Multi-modal Interaction for Virtual Tour Guidance through Large Language Models (Zhan Wang et al., 2024)

{{<citation>}}

Zhan Wang, Lin-Ping Yuan, Liangwei Wang, Bingchuan Jiang, Wei Zeng. (2024)  
**VirtuWander: Enhancing Multi-modal Interaction for Virtual Tour Guidance through Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.11923v2)  

---


**ABSTRACT**  
Tour guidance in virtual museums encourages multi-modal interactions to boost user experiences, concerning engagement, immersion, and spatial awareness. Nevertheless, achieving the goal is challenging due to the complexity of comprehending diverse user needs and accommodating personalized user preferences. Informed by a formative study that characterizes guidance-seeking contexts, we establish a multi-modal interaction design framework for virtual tour guidance. We then design VirtuWander, a two-stage innovative system using domain-oriented large language models to transform user inquiries into diverse guidance-seeking contexts and facilitate multi-modal interactions. The feasibility and versatility of VirtuWander are demonstrated with virtual guiding examples that encompass various touring scenarios and cater to personalized preferences. We further evaluate VirtuWander through a user study within an immersive simulated museum. The results suggest that our system enhances engaging virtual tour experiences through personalized communication and knowledgeable assistance, indicating its potential for expanding into real-world scenarios.

{{</citation>}}


### (104/119) Generative AI-Driven Human Digital Twin in IoT-Healthcare: A Comprehensive Survey (Jiayuan Chen et al., 2024)

{{<citation>}}

Jiayuan Chen, You Shi, Changyan Yi, Hongyang Du, Jiawen Kang, Dusit Niyato. (2024)  
**Generative AI-Driven Human Digital Twin in IoT-Healthcare: A Comprehensive Survey**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-LG, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.13699v1)  

---


**ABSTRACT**  
The Internet of things (IoT) can significantly enhance the quality of human life, specifically in healthcare, attracting extensive attentions to IoT-healthcare services. Meanwhile, the human digital twin (HDT) is proposed as an innovative paradigm that can comprehensively characterize the replication of the individual human body in the digital world and reflect its physical status in real time. Naturally, HDT is envisioned to empower IoT-healthcare beyond the application of healthcare monitoring by acting as a versatile and vivid human digital testbed, simulating the outcomes and guiding the practical treatments. However, successfully establishing HDT requires high-fidelity virtual modeling and strong information interactions but possibly with scarce, biased and noisy data. Fortunately, a recent popular technology called generative artificial intelligence (GAI) may be a promising solution because it can leverage advanced AI algorithms to automatically create, manipulate, and modify valuable while diverse data. This survey particularly focuses on the implementation of GAI-driven HDT in IoT-healthcare. We start by introducing the background of IoT-healthcare and the potential of GAI-driven HDT. Then, we delve into the fundamental techniques and present the overall framework of GAI-driven HDT. After that, we explore the realization of GAI-driven HDT in detail, including GAI-enabled data acquisition, communication, data management, digital modeling, and data analysis. Besides, we discuss typical IoT-healthcare applications that can be revolutionized by GAI-driven HDT, namely personalized health monitoring and diagnosis, personalized prescription, and personalized rehabilitation. Finally, we conclude this survey by highlighting some future research directions.

{{</citation>}}


## cs.CY (2)



### (105/119) CodeTailor: Personalized Parsons Puzzles are Preferred Over AI-Generated Solutions to Support Learning (Xinying Hou et al., 2024)

{{<citation>}}

Xinying Hou, Zihan Wu, Xu Wang, Barbara J. Ericson. (2024)  
**CodeTailor: Personalized Parsons Puzzles are Preferred Over AI-Generated Solutions to Support Learning**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs.CY  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.12125v1)  

---


**ABSTRACT**  
Programming can be challenging for novices, but it is difficult to provide high-quality, comprehensive, and timely support at scale. Generative AI and its products, like ChatGPT, can create a solution for most introductory programming problems. However, students may become overly reliant on these tools for quick code generation and homework completion, leading to reduced engagement and limited learning. In this work, we present \sys{}, a system that utilizes large language models (LLM) while still promoting students' cognitive engagement. \sys{} provides a personalized Parsons puzzle to support struggling students. In a Parsons puzzle, students place mixed-up code blocks in the correct order to solve a problem. A technical evaluation with 800 incorrect student code demonstrated that \sys{} can efficiently create high-quality (correct, personalized, and concise) Parsons puzzles for students. In a within-subjects experiment with 18 novice programmers, most students rated using \sys{} as more engaging, and they preferred \sys{} for learning rather than simply receiving an AI-generated solution. Additionally, students recalled more new elements from the supported practice to the posttest after using \sys{}, compared to when they simply received a direct solution. Qualitative observations and interviews provided evidence for the benefits of \sys{} including emphasizing algorithmic thinking, fostering continuity in learning, promoting metacognitive reflection, and boosting student confidence. We conclude by suggesting future designs for applying generative AI in a way that minimizes over-reliance and enhances learning.

{{</citation>}}


### (106/119) AI, insurance, discrimination and unfair differentiation. An overview and research agenda (Marvin S. L. van Bekkum et al., 2024)

{{<citation>}}

Marvin S. L. van Bekkum, Frederik J. Zuiderveen Borgesius. (2024)  
**AI, insurance, discrimination and unfair differentiation. An overview and research agenda**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.11892v1)  

---


**ABSTRACT**  
Insurers increasingly use AI. We distinguish two situations in which insurers use AI: (i) data-intensive underwriting, and (ii) behaviour-based insurance. (i) First, insurers can use AI for data analysis to assess risks: data-intensive underwriting. Underwriting is, in short, calculating risks and amending the insurance premium accordingly. (ii) Second, insurers can use AI to monitor the behaviour of consumers in real-time: behaviour-based insurance. For example, some car insurers give a discount if a consumer agrees to being tracked by the insurer and drives safely. While the two trends bring many advantages, they may also have discriminatory effects. This paper focuses on the following question. Which discrimination-related effects may occur if insurers use data-intensive underwriting and behaviour-based insurance? We focus on two types of discrimination-related effects: discrimination and other unfair differentiation. (i) Discrimination harms certain groups who are protected by non-discrimination law, for instance people with certain ethnicities. (ii) Unfair differentiation does not harm groups that are protected by non-discrimination law, but it does seem unfair. We introduce four factors to consider when assessing the fairness of insurance practices. The paper builds on literature from various disciplines including law, philosophy, and computer science.

{{</citation>}}


## eess.AS (2)



### (107/119) Consistency Based Unsupervised Self-training For ASR Personalisation (Jisi Zhang et al., 2024)

{{<citation>}}

Jisi Zhang, Vandana Rajan, Haaris Mehmood, David Tuckey, Pablo Peso Parada, Md Asif Jalal, Karthikeyan Saravanan, Gil Ho Lee, Jungin Lee, Seokyeong Jung. (2024)  
**Consistency Based Unsupervised Self-training For ASR Personalisation**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.12085v1)  

---


**ABSTRACT**  
On-device Automatic Speech Recognition (ASR) models trained on speech data of a large population might underperform for individuals unseen during training. This is due to a domain shift between user data and the original training data, differed by user's speaking characteristics and environmental acoustic conditions. ASR personalisation is a solution that aims to exploit user data to improve model robustness. The majority of ASR personalisation methods assume labelled user data for supervision. Personalisation without any labelled data is challenging due to limited data size and poor quality of recorded audio samples. This work addresses unsupervised personalisation by developing a novel consistency based training method via pseudo-labelling. Our method achieves a relative Word Error Rate Reduction (WERR) of 17.3% on unlabelled training data and 8.1% on held-out data compared to a pre-trained model, and outperforms the current state-of-the art methods.

{{</citation>}}


### (108/119) Streaming Bilingual End-to-End ASR model using Attention over Multiple Softmax (Aditya Patil et al., 2024)

{{<citation>}}

Aditya Patil, Vikas Joshi, Purvi Agrawal, Rupesh Mehta. (2024)  
**Streaming Bilingual End-to-End ASR model using Attention over Multiple Softmax**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.11645v1)  

---


**ABSTRACT**  
Even with several advancements in multilingual modeling, it is challenging to recognize multiple languages using a single neural model, without knowing the input language and most multilingual models assume the availability of the input language. In this work, we propose a novel bilingual end-to-end (E2E) modeling approach, where a single neural model can recognize both languages and also support switching between the languages, without any language input from the user. The proposed model has shared encoder and prediction networks, with language-specific joint networks that are combined via a self-attention mechanism. As the language-specific posteriors are combined, it produces a single posterior probability over all the output symbols, enabling a single beam search decoding and also allowing dynamic switching between the languages. The proposed approach outperforms the conventional bilingual baseline with 13.3%, 8.23% and 1.3% word error rate relative reduction on Hindi, English and code-mixed test sets, respectively.

{{</citation>}}


## eess.IV (3)



### (109/119) NLCG-Net: A Model-Based Zero-Shot Learning Framework for Undersampled Quantitative MRI Reconstruction (Xinrui Jiang et al., 2024)

{{<citation>}}

Xinrui Jiang, Yohan Jun, Jaejin Cho, Mengze Gao, Xingwang Yong, Berkin Bilgic. (2024)  
**NLCG-Net: A Model-Based Zero-Shot Learning Framework for Undersampled Quantitative MRI Reconstruction**  

---
Primary Category: eess.IV  
Categories: cs-LG, eess-IV, eess-SP, eess.IV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.12004v1)  

---


**ABSTRACT**  
Typical quantitative MRI (qMRI) methods estimate parameter maps after image reconstructing, which is prone to biases and error propagation. We propose a Nonlinear Conjugate Gradient (NLCG) optimizer for model-based T2/T1 estimation, which incorporates U-Net regularization trained in a scan-specific manner. This end-to-end method directly estimates qMRI maps from undersampled k-space data using mono-exponential signal modeling with zero-shot scan-specific neural network regularization to enable high fidelity T1 and T2 mapping. T2 and T1 mapping results demonstrate the ability of the proposed NLCG-Net to improve estimation quality compared to subspace reconstruction at high accelerations.

{{</citation>}}


### (110/119) LKFormer: Large Kernel Transformer for Infrared Image Super-Resolution (Feiwei Qin et al., 2024)

{{<citation>}}

Feiwei Qin, Kang Yan, Changmiao Wang, Ruiquan Ge, Yong Peng, Kai Zhang. (2024)  
**LKFormer: Large Kernel Transformer for Infrared Image Super-Resolution**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11859v2)  

---


**ABSTRACT**  
Given the broad application of infrared technology across diverse fields, there is an increasing emphasis on investigating super-resolution techniques for infrared images within the realm of deep learning. Despite the impressive results of current Transformer-based methods in image super-resolution tasks, their reliance on the self-attentive mechanism intrinsic to the Transformer architecture results in images being treated as one-dimensional sequences, thereby neglecting their inherent two-dimensional structure. Moreover, infrared images exhibit a uniform pixel distribution and a limited gradient range, posing challenges for the model to capture effective feature information. Consequently, we suggest a potent Transformer model, termed Large Kernel Transformer (LKFormer), to address this issue. Specifically, we have designed a Large Kernel Residual Attention (LKRA) module with linear complexity. This mainly employs depth-wise convolution with large kernels to execute non-local feature modeling, thereby substituting the standard self-attentive layer. Additionally, we have devised a novel feed-forward network structure called Gated-Pixel Feed-Forward Network (GPFN) to augment the LKFormer's capacity to manage the information flow within the network. Comprehensive experimental results reveal that our method surpasses the most advanced techniques available, using fewer parameters and yielding considerably superior performance.The source code will be available at https://github.com/sad192/large-kernel-Transformer.

{{</citation>}}


### (111/119) RTA-Former: Reverse Transformer Attention for Polyp Segmentation (Zhikai Li et al., 2024)

{{<citation>}}

Zhikai Li, Murong Yi, Ali Uneri, Sihan Niu, Craig Jones. (2024)  
**RTA-Former: Reverse Transformer Attention for Polyp Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2401.11671v1)  

---


**ABSTRACT**  
Polyp segmentation is a key aspect of colorectal cancer prevention, enabling early detection and guiding subsequent treatments. Intelligent diagnostic tools, including deep learning solutions, are widely explored to streamline and potentially automate this process. However, even with many powerful network architectures, there still comes the problem of producing accurate edge segmentation. In this paper, we introduce a novel network, namely RTA-Former, that employs a transformer model as the encoder backbone and innovatively adapts Reverse Attention (RA) with a transformer stage in the decoder for enhanced edge segmentation. The results of the experiments illustrate that RTA-Former achieves state-of-the-art (SOTA) performance in five polyp segmentation datasets. The strong capability of RTA-Former holds promise in improving the accuracy of Transformer-based polyp segmentation, potentially leading to better clinical decisions and patient outcomes. Our code will be publicly available on GitHub.

{{</citation>}}


## cs.NE (2)



### (112/119) Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey (Pengyi Li et al., 2024)

{{<citation>}}

Pengyi Li, Jianye Hao, Hongyao Tang, Xian Fu, Yan Zheng, Ke Tang. (2024)  
**Bridging Evolutionary Algorithms and Reinforcement Learning: A Comprehensive Survey**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-LG, cs-NE, cs.NE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.11963v1)  

---


**ABSTRACT**  
Evolutionary Reinforcement Learning (ERL), which integrates Evolutionary Algorithms (EAs) and Reinforcement Learning (RL) for optimization, has demonstrated remarkable performance advancements. By fusing the strengths of both approaches, ERL has emerged as a promising research direction. This survey offers a comprehensive overview of the diverse research branches in ERL. Specifically, we systematically summarize recent advancements in relevant algorithms and identify three primary research directions: EA-assisted optimization of RL, RL-assisted optimization of EA, and synergistic optimization of EA and RL. Following that, we conduct an in-depth analysis of each research direction, organizing multiple research branches. We elucidate the problems that each branch aims to tackle and how the integration of EA and RL addresses these challenges. In conclusion, we discuss potential challenges and prospective future research directions across various research directions.

{{</citation>}}


### (113/119) TIM: An Efficient Temporal Interaction Module for Spiking Transformer (Sicheng Shen et al., 2024)

{{<citation>}}

Sicheng Shen, Dongcheng Zhao, Guobin Shen, Yi Zeng. (2024)  
**TIM: An Efficient Temporal Interaction Module for Spiking Transformer**  

---
Primary Category: cs.NE  
Categories: cs-CV, cs-LG, cs-NE, cs.NE  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.11687v2)  

---


**ABSTRACT**  
Spiking Neural Networks (SNNs), as the third generation of neural networks, have gained prominence for their biological plausibility and computational efficiency, especially in processing diverse datasets. The integration of attention mechanisms, inspired by advancements in neural network architectures, has led to the development of Spiking Transformers. These have shown promise in enhancing SNNs' capabilities, particularly in the realms of both static and neuromorphic datasets. Despite their progress, a discernible gap exists in these systems, specifically in the Spiking Self Attention (SSA) mechanism's effectiveness in leveraging the temporal processing potential of SNNs. To address this, we introduce the Temporal Interaction Module (TIM), a novel, convolution-based enhancement designed to augment the temporal data processing abilities within SNN architectures. TIM's integration into existing SNN frameworks is seamless and efficient, requiring minimal additional parameters while significantly boosting their temporal information handling capabilities. Through rigorous experimentation, TIM has demonstrated its effectiveness in exploiting temporal information, leading to state-of-the-art performance across various neuromorphic datasets.

{{</citation>}}


## cs.SC (1)



### (114/119) Showing Proofs, Assessing Difficulty with GeoGebra Discovery (Zoltán Kovács et al., 2024)

{{<citation>}}

Zoltán Kovács, Tomás Recio, M. Pilar Vélez. (2024)  
**Showing Proofs, Assessing Difficulty with GeoGebra Discovery**  

---
Primary Category: cs.SC  
Categories: cs-AI, cs-CG, cs-SC, cs.SC  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.11900v1)  

---


**ABSTRACT**  
In our contribution we describe some on-going improvements concerning the Automated Reasoning Tools developed in GeoGebra Discovery, providing different examples of the performance of these new features. We describe the new ShowProof command, that outputs both the sequence of the different steps performed by GeoGebra Discovery to confirm a certain statement, as well as a number intending to grade the difficulty or interest of the assertion. The proposal of this assessment measure, involving the comparison of the expression of the thesis (or conclusion) as a combination of the hypotheses, will be developed.

{{</citation>}}


## cs.AR (1)



### (115/119) BETA: Binarized Energy-Efficient Transformer Accelerator at the Edge (Yuhao Ji et al., 2024)

{{<citation>}}

Yuhao Ji, Chao Fang, Zhongfeng Wang. (2024)  
**BETA: Binarized Energy-Efficient Transformer Accelerator at the Edge**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs.AR  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.11851v2)  

---


**ABSTRACT**  
Existing binary Transformers are promising in edge deployment due to their compact model size, low computational complexity, and considerable inference accuracy. However, deploying binary Transformers faces challenges on prior processors due to inefficient execution of quantized matrix multiplication (QMM) and the energy consumption overhead caused by multi-precision activations. To tackle the challenges above, we first develop a computation flow abstraction method for binary Transformers to improve QMM execution efficiency by optimizing the computation order. Furthermore, a binarized energy-efficient Transformer accelerator, namely BETA, is proposed to boost the efficient deployment at the edge. Notably, BETA features a configurable QMM engine, accommodating diverse activation precisions of binary Transformers and offering high-parallelism and high-speed for QMMs with impressive energy efficiency. Experimental results evaluated on ZCU102 FPGA show BETA achieves an average energy efficiency of 174 GOPS/W, which is 1.76~21.92x higher than prior FPGA-based accelerators, showing BETA's good potential for edge Transformer acceleration.

{{</citation>}}


## cs.MM (1)



### (116/119) MInD: Improving Multimodal Sentiment Analysis via Multimodal Information Disentanglement (Weichen Dai et al., 2024)

{{<citation>}}

Weichen Dai, Xingyu Li, Pengbo Hu, Zeyu Wang, Ji Qi, Jianlin Peng, Yi Zhou. (2024)  
**MInD: Improving Multimodal Sentiment Analysis via Multimodal Information Disentanglement**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2401.11818v1)  

---


**ABSTRACT**  
Learning effective joint representations has been a central task in multimodal sentiment analysis. Previous methods focus on leveraging the correlations between different modalities and enhancing performance through sophisticated fusion techniques. However, challenges still exist due to the inherent heterogeneity of distinct modalities, which may lead to distributional gap, impeding the full exploitation of inter-modal information and resulting in redundancy and impurity in the information extracted from features. To address this problem, we introduce the Multimodal Information Disentanglement (MInD) approach. MInD decomposes the multimodal inputs into a modality-invariant component, a modality-specific component, and a remnant noise component for each modality through a shared encoder and multiple private encoders. The shared encoder aims to explore the shared information and commonality across modalities, while the private encoders are deployed to capture the distinctive information and characteristic features. These representations thus furnish a comprehensive perspective of the multimodal data, facilitating the fusion process instrumental for subsequent prediction tasks. Furthermore, MInD improves the learned representations by explicitly modeling the task-irrelevant noise in an adversarial manner. Experimental evaluations conducted on benchmark datasets, including CMU-MOSI, CMU-MOSEI, and UR-Funny, demonstrate MInD's superior performance over existing state-of-the-art methods in both multimodal emotion recognition and multimodal humor detection tasks.

{{</citation>}}


## cs.IR (2)



### (117/119) Revisiting Document-Level Relation Extraction with Context-Guided Link Prediction (Monika Jain et al., 2024)

{{<citation>}}

Monika Jain, Raghava Mutharaju, Ramakanth Kavuluru, Kuldeep Singh. (2024)  
**Revisiting Document-Level Relation Extraction with Context-Guided Link Prediction**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2401.11800v1)  

---


**ABSTRACT**  
Document-level relation extraction (DocRE) poses the challenge of identifying relationships between entities within a document as opposed to the traditional RE setting where a single sentence is input. Existing approaches rely on logical reasoning or contextual cues from entities. This paper reframes document-level RE as link prediction over a knowledge graph with distinct benefits: 1) Our approach combines entity context with document-derived logical reasoning, enhancing link prediction quality. 2) Predicted links between entities offer interpretability, elucidating employed reasoning. We evaluate our approach on three benchmark datasets: DocRED, ReDocRED, and DWIE. The results indicate that our proposed method outperforms the state-of-the-art models and suggests that incorporating context-based link prediction techniques can enhance the performance of document-level relation extraction models.

{{</citation>}}


### (118/119) Domain-Aware Cross-Attention for Cross-domain Recommendation (Yuhao Luo et al., 2024)

{{<citation>}}

Yuhao Luo, Shiwei Ma, Mingjun Nie, Changping Peng, Zhangang Lin, Jingping Shao, Qianfang Xu. (2024)  
**Domain-Aware Cross-Attention for Cross-domain Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.11705v1)  

---


**ABSTRACT**  
Cross-domain recommendation (CDR) is an important method to improve recommender system performance, especially when observations in target domains are sparse. However, most existing cross-domain recommendations fail to fully utilize the target domain's special features and are hard to be generalized to new domains. The designed network is complex and is not suitable for rapid industrial deployment. Our method introduces a two-step domain-aware cross-attention, extracting transferable features of the source domain from different granularity, which allows the efficient expression of both domain and user interests. In addition, we simplify the training process, and our model can be easily deployed on new domains. We conduct experiments on both public datasets and industrial datasets, and the experimental results demonstrate the effectiveness of our method. We have also deployed the model in an online advertising system and observed significant improvements in both Click-Through-Rate (CTR) and effective cost per mille (ECPM).

{{</citation>}}


## physics.ao-ph (1)



### (119/119) Simulating Nighttime Visible Satellite Imagery of Tropical Cyclones Using Conditional Generative Adversarial Networks (Jinghuai Yao et al., 2024)

{{<citation>}}

Jinghuai Yao, Puyuan Du, Yucheng Zhao, Yubo Wang. (2024)  
**Simulating Nighttime Visible Satellite Imagery of Tropical Cyclones Using Conditional Generative Adversarial Networks**  

---
Primary Category: physics.ao-ph  
Categories: cs-LG, physics-ao-ph, physics.ao-ph  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2401.11679v1)  

---


**ABSTRACT**  
Visible (VIS) imagery of satellites has various important applications in meteorology, including monitoring Tropical Cyclones (TCs). However, it is unavailable at night because of the lack of sunlight. This study presents a Conditional Generative Adversarial Networks (CGAN) model that generates highly accurate nighttime visible reflectance using infrared (IR) bands and sunlight direction parameters as input. The model was trained and validated using target area observations of the Advanced Himawari Imager (AHI) in the daytime. This study also presents the first nighttime model validation using the Day/Night Band (DNB) of the Visible/Infrared Imager Radiometer Suite (VIIRS). The daytime statistical results of the Structural Similarity Index Measure (SSIM), Peak Signal-to-Noise Ratio (PSNR), Root Mean Square Error (RMSE), Correlation Coefficient (CC), and Bias are 0.885, 28.3, 0.0428, 0.984, and -0.0016 respectively, completely surpassing the model performance of previous studies. The nighttime statistical results of SSIM, PSNR, RMSE, and CC are 0.821, 24.4, 0.0643, and 0.969 respectively, which are slightly negatively impacted by the parallax between satellites. We performed full-disk model validation which proves our model could also be readily applied in the tropical ocean without TCs in the northern hemisphere. This model contributes to the nighttime monitoring of meteorological phenomena by providing accurate AI-generated visible imagery with adjustable virtual sunlight directions.

{{</citation>}}
