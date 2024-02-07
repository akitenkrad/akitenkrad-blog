---
draft: false
title: "arXiv @ 2024.02.05"
date: 2024-02-05
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.02.05"
    identifier: arxiv_20240205
    parent: 202402_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CR (4)](#cscr-4)
- [cs.CL (24)](#cscl-24)
- [cs.LG (26)](#cslg-26)
- [cs.CV (21)](#cscv-21)
- [eess.IV (1)](#eessiv-1)
- [stat.ME (1)](#statme-1)
- [cs.RO (1)](#csro-1)
- [cs.SI (2)](#cssi-2)
- [cs.SE (3)](#csse-3)
- [cs.HC (3)](#cshc-3)
- [cs.AI (2)](#csai-2)
- [cs.IR (1)](#csir-1)
- [cs.ET (1)](#cset-1)
- [cs.MA (1)](#csma-1)

## cs.CR (4)



### (1/91) A Review and Comparison of AI Enhanced Side Channel Analysis (Max Panoff et al., 2024)

{{<citation>}}

Max Panoff, Honggang Yu, Haoqi Shan, Yier Jin. (2024)  
**A Review and Comparison of AI Enhanced Side Channel Analysis**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.02299v1)  

---


**ABSTRACT**  
Side Channel Analysis (SCA) presents a clear threat to privacy and security in modern computing systems. The vast majority of communications are secured through cryptographic algorithms. These algorithms are often provably-secure from a cryptographical perspective, but their implementation on real hardware introduces vulnerabilities. Adversaries can exploit these vulnerabilities to conduct SCA and recover confidential information, such as secret keys or internal states. The threat of SCA has greatly increased as machine learning, and in particular deep learning, enhanced attacks become more common. In this work, we will examine the latest state-of-the-art deep learning techniques for side channel analysis, the theory behind them, and how they are conducted. Our focus will be on profiling attacks using deep learning techniques, but we will also examine some new and emerging methodologies enhanced by deep learning techniques, such as non-profiled attacks, artificial trace generation, and others. Finally, different deep learning enhanced SCA schemes attempted against the ANSSI SCA Database (ASCAD) and their relative performance will be evaluated and compared. This will lead to new research directions to secure cryptographic implementations against the latest SCA attacks.

{{</citation>}}


### (2/91) Recommendations on Statistical Randomness Test Batteries for Cryptographic Purposes (Elena Almaraz Luengo et al., 2024)

{{<citation>}}

Elena Almaraz Luengo, Luis Javier García Villalba. (2024)  
**Recommendations on Statistical Randomness Test Batteries for Cryptographic Purposes**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR, stat-CO  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2402.02240v1)  

---


**ABSTRACT**  
Security in different applications is closely related to the goodness of the sequences generated for such purposes. Not only in Cryptography but also in other areas, it is necessary to obtain long sequences of random numbers or that, at least, behave as such. To decide whether the generator used produces sequences that are random, unpredictable and independent, statistical checks are needed. Different batteries of hypothesis tests have been proposed for this purpose.   In this work, a survey of the main test batteries is presented, indicating their pros and cons, giving some guidelines for their use and presenting some practical examples.

{{</citation>}}


### (3/91) Data Poisoning for In-context Learning (Pengfei He et al., 2024)

{{<citation>}}

Pengfei He, Han Xu, Yue Xing, Hui Liu, Makoto Yamada, Jiliang Tang. (2024)  
**Data Poisoning for In-context Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2402.02160v1)  

---


**ABSTRACT**  
In the domain of large language models (LLMs), in-context learning (ICL) has been recognized for its innovative ability to adapt to new tasks, relying on examples rather than retraining or fine-tuning. This paper delves into the critical issue of ICL's susceptibility to data poisoning attacks, an area not yet fully explored. We wonder whether ICL is vulnerable, with adversaries capable of manipulating example data to degrade model performance. To address this, we introduce ICLPoison, a specialized attacking framework conceived to exploit the learning mechanisms of ICL. Our approach uniquely employs discrete text perturbations to strategically influence the hidden states of LLMs during the ICL process. We outline three representative strategies to implement attacks under our framework, each rigorously evaluated across a variety of models and tasks. Our comprehensive tests, including trials on the sophisticated GPT-4 model, demonstrate that ICL's performance is significantly compromised under our framework. These revelations indicate an urgent need for enhanced defense mechanisms to safeguard the integrity and reliability of LLMs in applications relying on in-context learning.

{{</citation>}}


### (4/91) Recent Advances in Digital Image and Video Forensics, Anti-forensics and Counter Anti-forensics (Maryam Al-Fehani et al., 2024)

{{<citation>}}

Maryam Al-Fehani, Saif Al-Kuwari. (2024)  
**Recent Advances in Digital Image and Video Forensics, Anti-forensics and Counter Anti-forensics**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2402.02089v1)  

---


**ABSTRACT**  
Image and video forensics have recently gained increasing attention due to the proliferation of manipulated images and videos, especially on social media platforms, such as Twitter and Instagram, which spread disinformation and fake news. This survey explores image and video identification and forgery detection covering both manipulated digital media and generative media. However, media forgery detection techniques are susceptible to anti-forensics; on the other hand, such anti-forensics techniques can themselves be detected. We therefore further cover both anti-forensics and counter anti-forensics techniques in image and video. Finally, we conclude this survey by highlighting some open problems in this domain.

{{</citation>}}


## cs.CL (24)



### (5/91) SemPool: Simple, robust, and interpretable KG pooling for enhancing language models (Costas Mavromatis et al., 2024)

{{<citation>}}

Costas Mavromatis, Petros Karypis, George Karypis. (2024)  
**SemPool: Simple, robust, and interpretable KG pooling for enhancing language models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Knowledge Graph, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2402.02289v1)  

---


**ABSTRACT**  
Knowledge Graph (KG) powered question answering (QA) performs complex reasoning over language semantics as well as knowledge facts. Graph Neural Networks (GNNs) learn to aggregate information from the underlying KG, which is combined with Language Models (LMs) for effective reasoning with the given question. However, GNN-based methods for QA rely on the graph information of the candidate answer nodes, which limits their effectiveness in more challenging settings where critical answer information is not included in the KG. We propose a simple graph pooling approach that learns useful semantics of the KG that can aid the LM's reasoning and that its effectiveness is robust under graph perturbations. Our method, termed SemPool, represents KG facts with pre-trained LMs, learns to aggregate their semantic information, and fuses it at different layers of the LM. Our experimental results show that SemPool outperforms state-of-the-art GNN-based methods by 2.27% accuracy points on average when answer information is missing from the KG. In addition, SemPool offers interpretability on what type of graph information is fused at different LM layers.

{{</citation>}}


### (6/91) SynthDST: Synthetic Data is All You Need for Few-Shot Dialog State Tracking (Atharva Kulkarni et al., 2024)

{{<citation>}}

Atharva Kulkarni, Bo-Hsiang Tseng, Joel Ruben Antony Moniz, Dhivya Piraviperumal, Hong Yu, Shruti Bhargava. (2024)  
**SynthDST: Synthetic Data is All You Need for Few-Shot Dialog State Tracking**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Dialog, Few-Shot, Language Model  
[Paper Link](http://arxiv.org/abs/2402.02285v1)  

---


**ABSTRACT**  
In-context learning with Large Language Models (LLMs) has emerged as a promising avenue of research in Dialog State Tracking (DST). However, the best-performing in-context learning methods involve retrieving and adding similar examples to the prompt, requiring access to labeled training data. Procuring such training data for a wide range of domains and applications is time-consuming, expensive, and, at times, infeasible. While zero-shot learning requires no training data, it significantly lags behind the few-shot setup. Thus, `\textit{Can we efficiently generate synthetic data for any dialogue schema to enable few-shot prompting?}' Addressing this question, we propose \method, a data generation framework tailored for DST, utilizing LLMs. Our approach only requires the dialogue schema and a few hand-crafted dialogue templates to synthesize natural, coherent, and free-flowing dialogues with DST annotations. Few-shot learning using data from {\method} results in $4-5%$ improvement in Joint Goal Accuracy over the zero-shot baseline on MultiWOZ 2.1 and 2.4. Remarkably, our few-shot learning approach recovers nearly $98%$ of the performance compared to the few-shot setup using human-annotated training data. Our synthetic data and code can be accessed at https://github.com/apple/ml-synthdst

{{</citation>}}


### (7/91) Data Quality Matters: Suicide Intention Detection on Social Media Posts Using a RoBERTa-CNN Model (Emily Lin et al., 2024)

{{<citation>}}

Emily Lin, Jian Sun, Hsingyu Chen, Mohammad H. Mahoor. (2024)  
**Data Quality Matters: Suicide Intention Detection on Social Media Posts Using a RoBERTa-CNN Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, NLP, Natural Language Processing, Social Media  
[Paper Link](http://arxiv.org/abs/2402.02262v1)  

---


**ABSTRACT**  
Suicide remains a global health concern for the field of health, which urgently needs innovative approaches for early detection and intervention. In this paper, we focus on identifying suicidal intentions in SuicideWatch Reddit posts and present a novel approach to suicide detection using the cutting-edge RoBERTa-CNN model, a variant of RoBERTa (Robustly optimized BERT approach). RoBERTa is used for various Natural Language Processing (NLP) tasks, including text classification and sentiment analysis. The effectiveness of the RoBERTa lies in its ability to capture textual information and form semantic relationships within texts. By adding the Convolution Neural Network (CNN) layer to the original model, the RoBERTa enhances its ability to capture important patterns from heavy datasets. To evaluate the RoBERTa-CNN, we experimented on the Suicide and Depression Detection dataset and obtained solid results. For example, RoBERTa-CNN achieves 98% mean accuracy with the standard deviation (STD) of 0.0009. It also reaches over 97.5% mean AUC value with an STD of 0.0013. In the meanwhile, RoBERTa-CNN outperforms competitive methods, demonstrating the robustness and ability to capture nuanced linguistic patterns for suicidal intentions. Therefore, RoBERTa-CNN can detect suicide intention on text data very well.

{{</citation>}}


### (8/91) Frequency Explains the Inverse Correlation of Large Language Models' Size, Training Data Amount, and Surprisal's Fit to Reading Times (Byung-Doh Oh et al., 2024)

{{<citation>}}

Byung-Doh Oh, Shisen Yue, William Schuler. (2024)  
**Frequency Explains the Inverse Correlation of Large Language Models' Size, Training Data Amount, and Surprisal's Fit to Reading Times**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2402.02255v1)  

---


**ABSTRACT**  
Recent studies have shown that as Transformer-based language models become larger and are trained on very large amounts of data, the fit of their surprisal estimates to naturalistic human reading times degrades. The current work presents a series of analyses showing that word frequency is a key explanatory factor underlying these two trends. First, residual errors from four language model families on four corpora show that the inverse correlation between model size and fit to reading times is the strongest on the subset of least frequent words, which is driven by excessively accurate predictions of larger model variants. Additionally, training dynamics reveal that during later training steps, all model variants learn to predict rare words and that larger model variants do so more accurately, which explains the detrimental effect of both training data amount and model size on fit to reading times. Finally, a feature attribution analysis demonstrates that larger model variants are able to accurately predict rare words based on both an effectively longer context window size as well as stronger local associations compared to smaller model variants. Taken together, these results indicate that Transformer-based language models' surprisal estimates diverge from human-like expectations due to the superhumanly complex associations they learn for predicting rare words.

{{</citation>}}


### (9/91) Beyond the Limits: A Survey of Techniques to Extend the Context Length in Large Language Models (Xindi Wang et al., 2024)

{{<citation>}}

Xindi Wang, Mahsa Salmani, Parsa Omidi, Xiangyu Ren, Mehdi Rezagholizadeh, Armaghan Eshaghi. (2024)  
**Beyond the Limits: A Survey of Techniques to Extend the Context Length in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.02244v1)  

---


**ABSTRACT**  
Recently, large language models (LLMs) have shown remarkable capabilities including understanding context, engaging in logical reasoning, and generating responses. However, this is achieved at the expense of stringent computational and memory requirements, hindering their ability to effectively support long input sequences. This survey provides an inclusive review of the recent techniques and methods devised to extend the sequence length in LLMs, thereby enhancing their capacity for long-context understanding. In particular, we review and categorize a wide range of techniques including architectural modifications, such as modified positional encoding and altered attention mechanisms, which are designed to enhance the processing of longer sequences while avoiding a proportional increase in computational requirements. The diverse methodologies investigated in this study can be leveraged across different phases of LLMs, i.e., training, fine-tuning and inference. This enables LLMs to efficiently process extended sequences. The limitations of the current methodologies is discussed in the last section along with the suggestions for future research directions, underscoring the importance of sequence length in the continued advancement of LLMs.

{{</citation>}}


### (10/91) Language Writ Large: LLMs, ChatGPT, Grounding, Meaning and Understanding (Stevan Harnad, 2024)

{{<citation>}}

Stevan Harnad. (2024)  
**Language Writ Large: LLMs, ChatGPT, Grounding, Meaning and Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, q-bio-NC  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2402.02243v1)  

---


**ABSTRACT**  
Apart from what (little) OpenAI may be concealing from us, we all know (roughly) how ChatGPT works (its huge text database, its statistics, its vector representations, and their huge number of parameters, its next-word training, and so on). But none of us can say (hand on heart) that we are not surprised by what ChatGPT has proved to be able to do with these resources. This has even driven some of us to conclude that ChatGPT actually understands. It is not true that it understands. But it is also not true that we understand how it can do what it can do. I will suggest some hunches about benign biases: convergent constraints that emerge at LLM scale that may be helping ChatGPT do so much better than we would have expected. These biases are inherent in the nature of language itself, at LLM scale, and they are closely linked to what it is that ChatGPT lacks, which is direct sensorimotor grounding to connect its words to their referents and its propositions to their meanings. These convergent biases are related to (1) the parasitism of indirect verbal grounding on direct sensorimotor grounding, (2) the circularity of verbal definition, (3) the mirroring of language production and comprehension, (4) iconicity in propositions at LLM scale, (5) computational counterparts of human categorical perception in category learning by neural nets, and perhaps also (6) a conjecture by Chomsky about the laws of thought. The exposition will be in the form of a dialogue with ChatGPT-4.

{{</citation>}}


### (11/91) A Data Generation Perspective to the Mechanism of In-Context Learning (Haitao Mao et al., 2024)

{{<citation>}}

Haitao Mao, Guangliang Liu, Yao Ma, Rongrong Wang, Jiliang Tang. (2024)  
**A Data Generation Perspective to the Mechanism of In-Context Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.02212v1)  

---


**ABSTRACT**  
In-Context Learning (ICL) empowers Large Language Models (LLMs) with the capacity to learn in context, achieving downstream generalization without gradient updates but with a few in-context examples. Despite the encouraging empirical success, the underlying mechanism of ICL remains unclear, and existing research offers various viewpoints of understanding. These studies propose intuition-driven and ad-hoc technical solutions for interpreting ICL, illustrating an ambiguous road map. In this paper, we leverage a data generation perspective to reinterpret recent efforts and demonstrate the potential broader usage of popular technical solutions, approaching a systematic angle. For a conceptual definition, we rigorously adopt the terms of skill learning and skill recognition. The difference between them is skill learning can learn new data generation functions from in-context data. We also provide a comprehensive study on the merits and weaknesses of different solutions, and highlight the uniformity among them given the perspective of data generation, establishing a technical foundation for future research to incorporate the strengths of different lines of research.

{{</citation>}}


### (12/91) Enhancing Complex Question Answering over Knowledge Graphs through Evidence Pattern Retrieval (Wentao Ding et al., 2024)

{{<citation>}}

Wentao Ding, Jinmao Li, Liangchuan Luo, Yuzhong Qu. (2024)  
**Enhancing Complex Question Answering over Knowledge Graphs through Evidence Pattern Retrieval**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: Knowledge Graph, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2402.02175v1)  

---


**ABSTRACT**  
Information retrieval (IR) methods for KGQA consist of two stages: subgraph extraction and answer reasoning. We argue current subgraph extraction methods underestimate the importance of structural dependencies among evidence facts. We propose Evidence Pattern Retrieval (EPR) to explicitly model the structural dependencies during subgraph extraction. We implement EPR by indexing the atomic adjacency pattern of resource pairs. Given a question, we perform dense retrieval to obtain atomic patterns formed by resource pairs. We then enumerate their combinations to construct candidate evidence patterns. These evidence patterns are scored using a neural model, and the best one is selected to extract a subgraph for downstream answer reasoning. Experimental results demonstrate that the EPR-based approach has significantly improved the F1 scores of IR-KGQA methods by over 10 points on ComplexWebQuestions and achieves competitive performance on WebQuestionsSP.

{{</citation>}}


### (13/91) Analyzing Sentiment Polarity Reduction in News Presentation through Contextual Perturbation and Large Language Models (Alapan Kuila et al., 2024)

{{<citation>}}

Alapan Kuila, Somnath Jena, Sudeshna Sarkar, Partha Pratim Chakrabarti. (2024)  
**Analyzing Sentiment Polarity Reduction in News Presentation through Contextual Perturbation and Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2402.02145v1)  

---


**ABSTRACT**  
In today's media landscape, where news outlets play a pivotal role in shaping public opinion, it is imperative to address the issue of sentiment manipulation within news text. News writers often inject their own biases and emotional language, which can distort the objectivity of reporting. This paper introduces a novel approach to tackle this problem by reducing the polarity of latent sentiments in news content. Drawing inspiration from adversarial attack-based sentence perturbation techniques and a prompt based method using ChatGPT, we employ transformation constraints to modify sentences while preserving their core semantics. Using three perturbation methods: replacement, insertion, and deletion coupled with a context-aware masked language model, we aim to maximize the desired sentiment score for targeted news aspects through a beam search algorithm. Our experiments and human evaluations demonstrate the effectiveness of these two models in achieving reduced sentiment polarity with minimal modifications while maintaining textual similarity, fluency, and grammatical correctness. Comparative analysis confirms the competitive performance of the adversarial attack based perturbation methods and prompt-based methods, offering a promising solution to foster more objective news reporting and combat emotional language bias in the media.

{{</citation>}}


### (14/91) Probing Critical Learning Dynamics of PLMs for Hate Speech Detection (Sarah Masud et al., 2024)

{{<citation>}}

Sarah Masud, Mohammad Aflah Khan, Vikram Goyal, Md Shad Akhtar, Tanmoy Chakraborty. (2024)  
**Probing Critical Learning Dynamics of PLMs for Hate Speech Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Hate Speech Detection  
[Paper Link](http://arxiv.org/abs/2402.02144v1)  

---


**ABSTRACT**  
Despite the widespread adoption, there is a lack of research into how various critical aspects of pretrained language models (PLMs) affect their performance in hate speech detection. Through five research questions, our findings and recommendations lay the groundwork for empirically investigating different aspects of PLMs' use in hate speech detection. We deep dive into comparing different pretrained models, evaluating their seed robustness, finetuning settings, and the impact of pretraining data collection time. Our analysis reveals early peaks for downstream tasks during pretraining, the limited benefit of employing a more recent pretraining corpus, and the significance of specific layers during finetuning. We further call into question the use of domain-specific models and highlight the need for dynamic datasets for benchmarking hate speech detection.

{{</citation>}}


### (15/91) Do Moral Judgment and Reasoning Capability of LLMs Change with Language? A Study using the Multilingual Defining Issues Test (Aditi Khandelwal et al., 2024)

{{<citation>}}

Aditi Khandelwal, Utkarsh Agarwal, Kumar Tanmay, Monojit Choudhury. (2024)  
**Do Moral Judgment and Reasoning Capability of LLMs Change with Language? A Study using the Multilingual Defining Issues Test**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, Multilingual, Reasoning  
[Paper Link](http://arxiv.org/abs/2402.02135v1)  

---


**ABSTRACT**  
This paper explores the moral judgment and moral reasoning abilities exhibited by Large Language Models (LLMs) across languages through the Defining Issues Test. It is a well known fact that moral judgment depends on the language in which the question is asked. We extend the work of beyond English, to 5 new languages (Chinese, Hindi, Russian, Spanish and Swahili), and probe three LLMs -- ChatGPT, GPT-4 and Llama2Chat-70B -- that shows substantial multilingual text processing and generation abilities. Our study shows that the moral reasoning ability for all models, as indicated by the post-conventional score, is substantially inferior for Hindi and Swahili, compared to Spanish, Russian, Chinese and English, while there is no clear trend for the performance of the latter four languages. The moral judgments too vary considerably by the language.

{{</citation>}}


### (16/91) Rendering Graphs for Graph Reasoning in Multimodal Large Language Models (Yanbin Wei et al., 2024)

{{<citation>}}

Yanbin Wei, Shuai Fu, Weisen Jiang, James T. Kwok, Yu Zhang. (2024)  
**Rendering Graphs for Graph Reasoning in Multimodal Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2402.02130v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are increasingly used for various tasks with graph structures, such as robotic planning, knowledge graph completion, and common-sense reasoning. Though LLMs can comprehend graph information in a textual format, they overlook the rich visual modality, which is an intuitive way for humans to comprehend structural information and conduct graph reasoning. The potential benefits and capabilities of representing graph structures as visual images (i.e., visual graph) is still unexplored. In this paper, we take the first step in incorporating visual information into graph reasoning tasks and propose a new benchmark GITQA, where each sample is a tuple (graph, image, textual description). We conduct extensive experiments on the GITQA benchmark using state-of-the-art multimodal LLMs. Results on graph reasoning tasks show that combining textual and visual information together performs better than using one modality alone. Moreover, the LLaVA-7B/13B models finetuned on the training set achieve higher accuracy than the closed-source model GPT-4(V). We also study the effects of augmentations in graph reasoning.

{{</citation>}}


### (17/91) Zero-shot Sentiment Analysis in Low-Resource Languages Using a Multilingual Sentiment Lexicon (Fajri Koto et al., 2024)

{{<citation>}}

Fajri Koto, Tilman Beck, Zeerak Talat, Iryna Gurevych, Timothy Baldwin. (2024)  
**Zero-shot Sentiment Analysis in Low-Resource Languages Using a Multilingual Sentiment Lexicon**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLOOM, GLM, GPT, Low-Resource, Multilingual, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2402.02113v1)  

---


**ABSTRACT**  
Improving multilingual language models capabilities in low-resource languages is generally difficult due to the scarcity of large-scale data in those languages. In this paper, we relax the reliance on texts in low-resource languages by using multilingual lexicons in pretraining to enhance multilingual capabilities. Specifically, we focus on zero-shot sentiment analysis tasks across 34 languages, including 6 high/medium-resource languages, 25 low-resource languages, and 3 code-switching datasets. We demonstrate that pretraining using multilingual lexicons, without using any sentence-level sentiment data, achieves superior zero-shot performance compared to models fine-tuned on English sentiment datasets, and large language models like GPT--3.5, BLOOMZ, and XGLM. These findings are observable for unseen low-resource languages to code-mixed scenarios involving high-resource languages.

{{</citation>}}


### (18/91) Are Large Language Models Good Prompt Optimizers? (Ruotian Ma et al., 2024)

{{<citation>}}

Ruotian Ma, Xiaolei Wang, Xin Zhou, Jian Li, Nan Du, Tao Gui, Qi Zhang, Xuanjing Huang. (2024)  
**Are Large Language Models Good Prompt Optimizers?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.02101v1)  

---


**ABSTRACT**  
LLM-based Automatic Prompt Optimization, which typically utilizes LLMs as Prompt Optimizers to self-reflect and refine prompts, has shown promising performance in recent studies. Despite the success, the underlying mechanism of this approach remains unexplored, and the true effectiveness of LLMs as Prompt Optimizers requires further validation. In this work, we conducted a comprehensive study to uncover the actual mechanism of LLM-based Prompt Optimization. Our findings reveal that the LLM optimizers struggle to identify the true causes of errors during reflection, tending to be biased by their own prior knowledge rather than genuinely reflecting on the errors. Furthermore, even when the reflection is semantically valid, the LLM optimizers often fail to generate appropriate prompts for the target models with a single prompt refinement step, partly due to the unpredictable behaviors of the target models. Based on the observations, we introduce a new "Automatic Behavior Optimization" paradigm, which directly optimizes the target model's behavior in a more controllable manner. We hope our study can inspire new directions for automatic prompt optimization development.

{{</citation>}}


### (19/91) Analyzing the Evaluation of Cross-Lingual Knowledge Transfer in Multilingual Language Models (Sara Rajaee et al., 2024)

{{<citation>}}

Sara Rajaee, Christof Monz. (2024)  
**Analyzing the Evaluation of Cross-Lingual Knowledge Transfer in Multilingual Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2402.02099v1)  

---


**ABSTRACT**  
Recent advances in training multilingual language models on large datasets seem to have shown promising results in knowledge transfer across languages and achieve high performance on downstream tasks. However, we question to what extent the current evaluation benchmarks and setups accurately measure zero-shot cross-lingual knowledge transfer. In this work, we challenge the assumption that high zero-shot performance on target tasks reflects high cross-lingual ability by introducing more challenging setups involving instances with multiple languages. Through extensive experiments and analysis, we show that the observed high performance of multilingual models can be largely attributed to factors not requiring the transfer of actual linguistic knowledge, such as task- and surface-level knowledge. More specifically, we observe what has been transferred across languages is mostly data artifacts and biases, especially for low-resource languages. Our findings highlight the overlooked drawbacks of existing cross-lingual test data and evaluation setups, calling for a more nuanced understanding of the cross-lingual capabilities of multilingual models.

{{</citation>}}


### (20/91) Revisiting the Markov Property for Machine Translation (Cunxiao Du et al., 2024)

{{<citation>}}

Cunxiao Du, Hao Zhou, Zhaopeng Tu, Jing Jiang. (2024)  
**Revisiting the Markov Property for Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation, Transformer  
[Paper Link](http://arxiv.org/abs/2402.02084v1)  

---


**ABSTRACT**  
In this paper, we re-examine the Markov property in the context of neural machine translation. We design a Markov Autoregressive Transformer~(MAT) and undertake a comprehensive assessment of its performance across four WMT benchmarks. Our findings indicate that MAT with an order larger than 4 can generate translations with quality on par with that of conventional autoregressive transformers. In addition, counter-intuitively, we also find that the advantages of utilizing a higher-order MAT do not specifically contribute to the translation of longer sentences.

{{</citation>}}


### (21/91) Translation Errors Significantly Impact Low-Resource Languages in Cross-Lingual Learning (Ashish Sunil Agrawal et al., 2024)

{{<citation>}}

Ashish Sunil Agrawal, Barah Fazili, Preethi Jyothi. (2024)  
**Translation Errors Significantly Impact Low-Resource Languages in Cross-Lingual Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Low-Resource, NLI  
[Paper Link](http://arxiv.org/abs/2402.02080v1)  

---


**ABSTRACT**  
Popular benchmarks (e.g., XNLI) used to evaluate cross-lingual language understanding consist of parallel versions of English evaluation sets in multiple target languages created with the help of professional translators. When creating such parallel data, it is critical to ensure high-quality translations for all target languages for an accurate characterization of cross-lingual transfer. In this work, we find that translation inconsistencies do exist and interestingly they disproportionally impact low-resource languages in XNLI. To identify such inconsistencies, we propose measuring the gap in performance between zero-shot evaluations on the human-translated and machine-translated target text across multiple target languages; relatively large gaps are indicative of translation errors. We also corroborate that translation errors exist for two target languages, namely Hindi and Urdu, by doing a manual reannotation of human-translated test instances in these two languages and finding poor agreement with the original English labels these instances were supposed to inherit.

{{</citation>}}


### (22/91) Exploring the Robustness of Task-oriented Dialogue Systems for Colloquial German Varieties (Ekaterina Artemova et al., 2024)

{{<citation>}}

Ekaterina Artemova, Verena Blaschke, Barbara Plank. (2024)  
**Exploring the Robustness of Task-oriented Dialogue Systems for Colloquial German Varieties**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2402.02078v1)  

---


**ABSTRACT**  
Mainstream cross-lingual task-oriented dialogue (ToD) systems leverage the transfer learning paradigm by training a joint model for intent recognition and slot-filling in English and applying it, zero-shot, to other languages. We address a gap in prior research, which often overlooked the transfer to lower-resource colloquial varieties due to limited test data. Inspired by prior work on English varieties, we craft and manually evaluate perturbation rules that transform German sentences into colloquial forms and use them to synthesize test sets in four ToD datasets. Our perturbation rules cover 18 distinct language phenomena, enabling us to explore the impact of each perturbation on slot and intent performance. Using these new datasets, we conduct an experimental evaluation across six different transformers. Here, we demonstrate that when applied to colloquial varieties, ToD systems maintain their intent recognition performance, losing 6% (4.62 percentage points) in accuracy on average. However, they exhibit a significant drop in slot detection, with a decrease of 31% (21 percentage points) in slot F1 score. Our findings are further supported by a transfer experiment from Standard American English to synthetic Urban African American Vernacular English.

{{</citation>}}


### (23/91) Investigating Content Planning for Navigating Trade-offs in Knowledge-Grounded Dialogue (Kushal Chawla et al., 2024)

{{<citation>}}

Kushal Chawla, Hannah Rashkin, Gaurav Singh Tomar, David Reitter. (2024)  
**Investigating Content Planning for Navigating Trade-offs in Knowledge-Grounded Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2402.02077v1)  

---


**ABSTRACT**  
Knowledge-grounded dialogue generation is a challenging task because it requires satisfying two fundamental yet often competing constraints: being responsive in a manner that is specific to what the conversation partner has said while also being attributable to an underlying source document. In this work, we bring this trade-off between these two objectives (specificity and attribution) to light and ask the question: Can explicit content planning before the response generation help the model to address this challenge? To answer this question, we design a framework called PLEDGE, which allows us to experiment with various plan variables explored in prior work, supporting both metric-agnostic and metric-aware approaches. While content planning shows promise, our results on whether it can actually help to navigate this trade-off are mixed -- planning mechanisms that are metric-aware (use automatic metrics during training) are better at automatic evaluations but underperform in human judgment compared to metric-agnostic mechanisms. We discuss how this may be caused by over-fitting to automatic metrics and the need for future work to better calibrate these metrics towards human judgment. We hope the observations from our analysis will inform future work that aims to apply content planning in this context.

{{</citation>}}


### (24/91) How well do LLMs cite relevant medical references? An evaluation framework and analyses (Kevin Wu et al., 2024)

{{<citation>}}

Kevin Wu, Eric Wu, Ally Cassasola, Angela Zhang, Kevin Wei, Teresa Nguyen, Sith Riantawan, Patricia Shi Riantawan, Daniel E. Ho, James Zou. (2024)  
**How well do LLMs cite relevant medical references? An evaluation framework and analyses**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2402.02008v1)  

---


**ABSTRACT**  
Large language models (LLMs) are currently being used to answer medical questions across a variety of clinical domains. Recent top-performing commercial LLMs, in particular, are also capable of citing sources to support their responses. In this paper, we ask: do the sources that LLMs generate actually support the claims that they make? To answer this, we propose three contributions. First, as expert medical annotations are an expensive and time-consuming bottleneck for scalable evaluation, we demonstrate that GPT-4 is highly accurate in validating source relevance, agreeing 88% of the time with a panel of medical doctors. Second, we develop an end-to-end, automated pipeline called \textit{SourceCheckup} and use it to evaluate five top-performing LLMs on a dataset of 1200 generated questions, totaling over 40K pairs of statements and sources. Interestingly, we find that between ~50% to 90% of LLM responses are not fully supported by the sources they provide. We also evaluate GPT-4 with retrieval augmented generation (RAG) and find that, even still, around 30\% of individual statements are unsupported, while nearly half of its responses are not fully supported. Third, we open-source our curated dataset of medical questions and expert annotations for future evaluations. Given the rapid pace of LLM development and the potential harms of incorrect or outdated medical information, it is crucial to also understand and quantify their capability to produce relevant, trustworthy medical references.

{{</citation>}}


### (25/91) Self-Debiasing Large Language Models: Zero-Shot Recognition and Reduction of Stereotypes (Isabel O. Gallegos et al., 2024)

{{<citation>}}

Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Tong Yu, Hanieh Deilamsalehy, Ruiyi Zhang, Sungchul Kim, Franck Dernoncourt. (2024)  
**Self-Debiasing Large Language Models: Zero-Shot Recognition and Reduction of Stereotypes**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2402.01981v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown remarkable advances in language generation and understanding but are also prone to exhibiting harmful social biases. While recognition of these behaviors has generated an abundance of bias mitigation techniques, most require modifications to the training data, model parameters, or decoding strategy, which may be infeasible without access to a trainable model. In this work, we leverage the zero-shot capabilities of LLMs to reduce stereotyping in a technique we introduce as zero-shot self-debiasing. With two approaches, self-debiasing via explanation and self-debiasing via reprompting, we show that self-debiasing can significantly reduce the degree of stereotyping across nine different social groups while relying only on the LLM itself and a simple prompt, with explanations correctly identifying invalid assumptions and reprompting delivering the greatest reductions in bias. We hope this work opens inquiry into other zero-shot techniques for bias mitigation.

{{</citation>}}


### (26/91) SOCIALITE-LLAMA: An Instruction-Tuned Model for Social Scientific Tasks (Gourab Dey et al., 2024)

{{<citation>}}

Gourab Dey, Adithya V Ganesan, Yash Kumar Lal, Manal Shah, Shreyashee Sinha, Matthew Matero, Salvatore Giorgi, Vivek Kulkarni, H. Andrew Schwartz. (2024)  
**SOCIALITE-LLAMA: An Instruction-Tuned Model for Social Scientific Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2402.01980v1)  

---


**ABSTRACT**  
Social science NLP tasks, such as emotion or humor detection, are required to capture the semantics along with the implicit pragmatics from text, often with limited amounts of training data. Instruction tuning has been shown to improve the many capabilities of large language models (LLMs) such as commonsense reasoning, reading comprehension, and computer programming. However, little is known about the effectiveness of instruction tuning on the social domain where implicit pragmatic cues are often needed to be captured. We explore the use of instruction tuning for social science NLP tasks and introduce Socialite-Llama -- an open-source, instruction-tuned Llama. On a suite of 20 social science tasks, Socialite-Llama improves upon the performance of Llama as well as matches or improves upon the performance of a state-of-the-art, multi-task finetuned model on a majority of them. Further, Socialite-Llama also leads to improvement on 5 out of 6 related social tasks as compared to Llama, suggesting instruction tuning can lead to generalized social understanding. All resources including our code, model and dataset can be found through bit.ly/socialitellama.

{{</citation>}}


### (27/91) MasonPerplexity at ClimateActivism 2024: Integrating Advanced Ensemble Techniques and Data Augmentation for Climate Activism Stance and Hate Event Identification (Al Nahian Bin Emran et al., 2024)

{{<citation>}}

Al Nahian Bin Emran, Amrita Ganguly, Sadiya Sayara Chowdhury Puspo, Dhiman Goswami, Md Nishat Raihan. (2024)  
**MasonPerplexity at ClimateActivism 2024: Integrating Advanced Ensemble Techniques and Data Augmentation for Climate Activism Stance and Hate Event Identification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Perplexity  
[Paper Link](http://arxiv.org/abs/2402.01976v1)  

---


**ABSTRACT**  
The task of identifying public opinions on social media, particularly regarding climate activism and the detection of hate events, has emerged as a critical area of research in our rapidly changing world. With a growing number of people voicing either to support or oppose to climate-related issues - understanding these diverse viewpoints has become increasingly vital. Our team, MasonPerplexity, participates in a significant research initiative focused on this subject. We extensively test various models and methods, discovering that our most effective results are achieved through ensemble modeling, enhanced by data augmentation techniques like back-translation. In the specific components of this research task, our team achieved notable positions, ranking 5th, 1st, and 6th in the respective sub-tasks, thereby illustrating the effectiveness of our approach in this important field of study.

{{</citation>}}


### (28/91) MasonPerplexity at Multimodal Hate Speech Event Detection 2024: Hate Speech and Target Detection Using Transformer Ensembles (Amrita Ganguly et al., 2024)

{{<citation>}}

Amrita Ganguly, Al Nahian Bin Emran, Sadiya Sayara Chowdhury Puspo, Md Nishat Raihan, Dhiman Goswami, Marcos Zampieri. (2024)  
**MasonPerplexity at Multimodal Hate Speech Event Detection 2024: Hate Speech and Target Detection Using Transformer Ensembles**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Event Detection, Perplexity, Transformer  
[Paper Link](http://arxiv.org/abs/2402.01967v1)  

---


**ABSTRACT**  
The automatic identification of offensive language such as hate speech is important to keep discussions civil in online communities. Identifying hate speech in multimodal content is a particularly challenging task because offensiveness can be manifested in either words or images or a juxtaposition of the two. This paper presents the MasonPerplexity submission for the Shared Task on Multimodal Hate Speech Event Detection at CASE 2024 at EACL 2024. The task is divided into two sub-tasks: sub-task A focuses on the identification of hate speech and sub-task B focuses on the identification of targets in text-embedded images during political events. We use an XLM-roBERTa-large model for sub-task A and an ensemble approach combining XLM-roBERTa-base, BERTweet-large, and BERT-base for sub-task B. Our approach obtained 0.8347 F1-score in sub-task A and 0.6741 F1-score in sub-task B ranking 3rd on both sub-tasks.

{{</citation>}}


## cs.LG (26)



### (29/91) Future Directions in Foundations of Graph Machine Learning (Christopher Morris et al., 2024)

{{<citation>}}

Christopher Morris, Nadav Dym, Haggai Maron, İsmail İlkan Ceylan, Fabrizio Frasca, Ron Levie, Derek Lim, Michael Bronstein, Martin Grohe, Stefanie Jegelka. (2024)  
**Future Directions in Foundations of Graph Machine Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DM, cs-LG, cs-NE, cs.LG, stat-ML  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2402.02287v1)  

---


**ABSTRACT**  
Machine learning on graphs, especially using graph neural networks (GNNs), has seen a surge in interest due to the wide availability of graph data across a broad spectrum of disciplines, from life to social and engineering sciences. Despite their practical success, our theoretical understanding of the properties of GNNs remains highly incomplete. Recent theoretical advancements primarily focus on elucidating the coarse-grained expressive power of GNNs, predominantly employing combinatorial techniques. However, these studies do not perfectly align with practice, particularly in understanding the generalization behavior of GNNs when trained with stochastic first-order optimization techniques. In this position paper, we argue that the graph machine learning community needs to shift its attention to developing a more balanced theory of graph machine learning, focusing on a more thorough understanding of the interplay of expressive power, generalization, and optimization.

{{</citation>}}


### (30/91) MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers (Yatong Bai et al., 2024)

{{<citation>}}

Yatong Bai, Mo Zhou, Vishal M. Patel, Somayeh Sojoudi. (2024)  
**MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers**  

---
Primary Category: cs.LG  
Categories: 68T07, cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2402.02263v1)  

---


**ABSTRACT**  
Adversarial robustness often comes at the cost of degraded accuracy, impeding the real-life application of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet datasets, experimental results with custom strong adaptive attacks demonstrate MixedNUTS's vastly improved accuracy and near-SOTA robustness -- it boosts CIFAR-100 clean accuracy by 7.86 points, sacrificing merely 0.87 points in robust accuracy.

{{</citation>}}


### (31/91) XTSFormer: Cross-Temporal-Scale Transformer for Irregular Time Event Prediction (Tingsong Xiao et al., 2024)

{{<citation>}}

Tingsong Xiao, Zelin Xu, Wenchong He, Jim Su, Yupu Zhang, Raymond Opoku, Ronald Ison, Jason Petho, Jiang Bian, Patrick Tighe, Parisa Rashidi, Zhe Jiang. (2024)  
**XTSFormer: Cross-Temporal-Scale Transformer for Irregular Time Event Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.02258v1)  

---


**ABSTRACT**  
Event prediction aims to forecast the time and type of a future event based on a historical event sequence. Despite its significance, several challenges exist, including the irregularity of time intervals between consecutive events, the existence of cycles, periodicity, and multi-scale event interactions, as well as the high computational costs for long event sequences. Existing neural temporal point processes (TPPs) methods do not capture the multi-scale nature of event interactions, which is common in many real-world applications such as clinical event data. To address these issues, we propose the cross-temporal-scale transformer (XTSFormer), designed specifically for irregularly timed event data. Our model comprises two vital components: a novel Feature-based Cycle-aware Time Positional Encoding (FCPE) that adeptly captures the cyclical nature of time, and a hierarchical multi-scale temporal attention mechanism. These scales are determined by a bottom-up clustering algorithm. Extensive experiments on several real-world datasets show that our XTSFormer outperforms several baseline methods in prediction performance.

{{</citation>}}


### (32/91) Graph Foundation Models (Haitao Mao et al., 2024)

{{<citation>}}

Haitao Mao, Zhikai Chen, Wenzhuo Tang, Jianan Zhao, Yao Ma, Tong Zhao, Neil Shah, Michael Galkin, Jiliang Tang. (2024)  
**Graph Foundation Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2402.02216v1)  

---


**ABSTRACT**  
Graph Foundation Model (GFM) is a new trending research topic in the graph domain, aiming to develop a graph model capable of generalizing across different graphs and tasks. However, a versatile GFM has not yet been achieved. The key challenge in building GFM is how to enable positive transfer across graphs with diverse structural patterns. Inspired by the existing foundation models in the CV and NLP domains, we propose a novel perspective for the GFM development by advocating for a ``graph vocabulary'', in which the basic transferable units underlying graphs encode the invariance on graphs. We ground the graph vocabulary construction from essential aspects including network analysis, theoretical foundations, and stability. Such a vocabulary perspective can potentially advance the future GFM design following the neural scaling laws.

{{</citation>}}


### (33/91) Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models (Yongshuo Zong et al., 2024)

{{<citation>}}

Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, Timothy Hospedales. (2024)  
**Safety Fine-Tuning at (Almost) No Cost: A Baseline for Vision Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.02207v1)  

---


**ABSTRACT**  
Current vision large language models (VLLMs) exhibit remarkable capabilities yet are prone to generate harmful content and are vulnerable to even the simplest jailbreaking attacks. Our initial analysis finds that this is due to the presence of harmful data during vision-language instruction fine-tuning, and that VLLM fine-tuning can cause forgetting of safety alignment previously learned by the underpinning LLM. To address this issue, we first curate a vision-language safe instruction-following dataset VLGuard covering various harmful categories. Our experiments demonstrate that integrating this dataset into standard vision-language fine-tuning or utilizing it for post-hoc fine-tuning effectively safety aligns VLLMs. This alignment is achieved with minimal impact on, or even enhancement of, the models' helpfulness. The versatility of our safety fine-tuning dataset makes it a valuable resource for safety-testing existing VLLMs, training new models or safeguarding pre-trained VLLMs. Empirical results demonstrate that fine-tuned VLLMs effectively reject unsafe instructions and substantially reduce the success rates of several black-box adversarial attacks, which approach zero in many cases. The code and dataset are available at https://github.com/ys-zong/VLGuard.

{{</citation>}}


### (34/91) Using Deep Ensemble Forest for High Resolution Mapping of PM2.5 from MODIS MAIAC AOD in Tehran, Iran (Hossein Bagheri, 2024)

{{<citation>}}

Hossein Bagheri. (2024)  
**Using Deep Ensemble Forest for High Resolution Mapping of PM2.5 from MODIS MAIAC AOD in Tehran, Iran**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.02139v1)  

---


**ABSTRACT**  
High resolution mapping of PM2.5 concentration over Tehran city is challenging because of the complicated behavior of numerous sources of pollution and the insufficient number of ground air quality monitoring stations. Alternatively, high resolution satellite Aerosol Optical Depth (AOD) data can be employed for high resolution mapping of PM2.5. For this purpose, different data-driven methods have been used in the literature. Recently, deep learning methods have demonstrated their ability to estimate PM2.5 from AOD data. However, these methods have several weaknesses in solving the problem of estimating PM2.5 from satellite AOD data. In this paper, the potential of the deep ensemble forest method for estimating the PM2.5 concentration from AOD data was evaluated. The results showed that the deep ensemble forest method with R2 = 0.74 gives a higher accuracy of PM2.5 estimation than deep learning methods (R2 = 0.67) as well as classic data-driven methods such as random forest (R2 = 0.68). Additionally, the estimated values of PM2.5 using the deep ensemble forest algorithm were used along with ground data to generate a high resolution map of PM2.5. Evaluation of the produced PM2.5 map revealed the good performance of the deep ensemble forest for modeling the variation of PM2.5 in the city of Tehran.

{{</citation>}}


### (35/91) Composite Active Learning: Towards Multi-Domain Active Learning with Theoretical Guarantees (Guang-Yuan Hao et al., 2024)

{{<citation>}}

Guang-Yuan Hao, Hengguan Huang, Haotian Wang, Jie Gao, Hao Wang. (2024)  
**Composite Active Learning: Towards Multi-Domain Active Learning with Theoretical Guarantees**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2402.02110v1)  

---


**ABSTRACT**  
Active learning (AL) aims to improve model performance within a fixed labeling budget by choosing the most informative data points to label. Existing AL focuses on the single-domain setting, where all data come from the same domain (e.g., the same dataset). However, many real-world tasks often involve multiple domains. For example, in visual recognition, it is often desirable to train an image classifier that works across different environments (e.g., different backgrounds), where images from each environment constitute one domain. Such a multi-domain AL setting is challenging for prior methods because they (1) ignore the similarity among different domains when assigning labeling budget and (2) fail to handle distribution shift of data across different domains. In this paper, we propose the first general method, dubbed composite active learning (CAL), for multi-domain AL. Our approach explicitly considers the domain-level and instance-level information in the problem; CAL first assigns domain-level budgets according to domain-level importance, which is estimated by optimizing an upper error bound that we develop; with the domain-level budgets, CAL then leverages a certain instance-level query strategy to select samples to label from each domain. Our theoretical analysis shows that our method achieves a better error bound compared to current AL methods. Our empirical results demonstrate that our approach significantly outperforms the state-of-the-art AL methods on both synthetic and real-world multi-domain datasets. Code is available at https://github.com/Wang-ML-Lab/multi-domain-active-learning.

{{</citation>}}


### (36/91) Break the Sequential Dependency of LLM Inference Using Lookahead Decoding (Yichao Fu et al., 2024)

{{<citation>}}

Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang. (2024)  
**Break the Sequential Dependency of LLM Inference Using Lookahead Decoding**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2402.02057v1)  

---


**ABSTRACT**  
Autoregressive decoding of large language models (LLMs) is memory bandwidth bounded, resulting in high latency and significant wastes of the parallel processing power of modern accelerators. Existing methods for accelerating LLM decoding often require a draft model (e.g., speculative decoding), which is nontrivial to obtain and unable to generalize. In this paper, we introduce Lookahead decoding, an exact, parallel decoding algorithm that accelerates LLM decoding without needing auxiliary models or data stores. It allows trading per-step log(FLOPs) to reduce the number of total decoding steps, is more parallelizable on single or multiple modern accelerators, and is compatible with concurrent memory-efficient attention (e.g., FlashAttention). Our implementation of Lookahead decoding can speed up autoregressive decoding by up to 1.8x on MT-bench and 4x with strong scaling on multiple GPUs in code completion tasks. Our code is avialable at https://github.com/hao-ai-lab/LookaheadDecoding

{{</citation>}}


### (37/91) Variance Alignment Score: A Simple But Tough-to-Beat Data Selection Method for Multimodal Contrastive Learning (Yiping Wang et al., 2024)

{{<citation>}}

Yiping Wang, Yifang Chen, Wendan Yan, Kevin Jamieson, Simon Shaolei Du. (2024)  
**Variance Alignment Score: A Simple But Tough-to-Beat Data Selection Method for Multimodal Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2402.02055v1)  

---


**ABSTRACT**  
In recent years, data selection has emerged as a core issue for large-scale visual-language model pretraining, especially on noisy web-curated datasets. One widely adopted strategy assigns quality scores such as CLIP similarity for each sample and retains the data pairs with the highest scores. However, these approaches are agnostic of data distribution and always fail to select the most informative samples. To solve this problem, we propose a simple yet theoretically principled metric named Variance Alignment Score (VAS), which has the form $\langle \Sigma_{\text{test}}, \Sigma_i\rangle$. Here, $\Sigma_{\text{test}}$ represents the target (cross-)covariance matrix we aim to align, potentially based on prior knowledge, while $\Sigma_i$ denotes the tensor product of single or multi-modal representations for the $i$-th sample. We further design a new data selection method that maximizes the total VAS. We provide theoretical analysis in a simplified setting to demonstrate the theoretical advantage of VAS over random or other existing data selection. Experimentally, applying VAS and CLIP scores together can outperform baselines by a margin of $1.3\%$ average on 38 evaluation sets for noisy dataset DataComp and $2.5\%$ on VTAB for high-quality dataset CC12M. Additionally, our ablation study also shows visual features are better than text for calculating VAS, and the related classical experimental design methods may fail under this context.

{{</citation>}}


### (38/91) Neural Scaling Laws on Graphs (Jingzhe Liu et al., 2024)

{{<citation>}}

Jingzhe Liu, Haitao Mao, Zhikai Chen, Tong Zhao, Neil Shah, Jiliang Tang. (2024)  
**Neural Scaling Laws on Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2402.02054v1)  

---


**ABSTRACT**  
Deep graph models (e.g., graph neural networks and graph transformers) have become important techniques for leveraging knowledge across various types of graphs. Yet, the scaling properties of deep graph models have not been systematically investigated, casting doubt on the feasibility of achieving large graph models through enlarging the model and dataset sizes. In this work, we delve into neural scaling laws on graphs from both model and data perspectives. We first verify the validity of such laws on graphs, establishing formulations to describe the scaling behaviors. For model scaling, we investigate the phenomenon of scaling law collapse and identify overfitting as the potential reason. Moreover, we reveal that the model depth of deep graph models can impact the model scaling behaviors, which differ from observations in other domains such as CV and NLP. For data scaling, we suggest that the number of graphs can not effectively metric the graph data volume in scaling law since the sizes of different graphs are highly irregular. Instead, we reform the data scaling law with the number of edges as the metric to address the irregular graph sizes. We further demonstrate the reformed law offers a unified view of the data scaling behaviors for various fundamental graph tasks including node classification, link prediction, and graph classification. This work provides valuable insights into neural scaling laws on graphs, which can serve as an essential step toward large graph models.

{{</citation>}}


### (39/91) Feature Selection using the concept of Peafowl Mating in IDS (Partha Ghosh et al., 2024)

{{<citation>}}

Partha Ghosh, Joy Sharma, Nilesh Pandey. (2024)  
**Feature Selection using the concept of Peafowl Mating in IDS**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2402.02052v1)  

---


**ABSTRACT**  
Cloud computing has high applicability as an Internet based service that relies on sharing computing resources. Cloud computing provides services that are Infrastructure based, Platform based and Software based. The popularity of this technology is due to its superb performance, high level of computing ability, low cost of services, scalability, availability and flexibility. The obtainability and openness of data in cloud environment make it vulnerable to the world of cyber-attacks. To detect the attacks Intrusion Detection System is used, that can identify the attacks and ensure information security. Such a coherent and proficient Intrusion Detection System is proposed in this paper to achieve higher certainty levels regarding safety in cloud environment. In this paper, the mating behavior of peafowl is incorporated into an optimization algorithm which in turn is used as a feature selection algorithm. The algorithm is used to reduce the huge size of cloud data so that the IDS can work efficiently on the cloud to detect intrusions. The proposed model has been experimented with NSL-KDD dataset as well as Kyoto dataset and have proved to be a better as well as an efficient IDS.

{{</citation>}}


### (40/91) Locally-Adaptive Quantization for Streaming Vector Search (Cecilia Aguerrebere et al., 2024)

{{<citation>}}

Cecilia Aguerrebere, Mark Hildebrand, Ishwar Singh Bhati, Theodore Willke, Mariano Tepper. (2024)  
**Locally-Adaptive Quantization for Streaming Vector Search**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2402.02044v1)  

---


**ABSTRACT**  
Retrieving the most similar vector embeddings to a given query among a massive collection of vectors has long been a key component of countless real-world applications. The recently introduced Retrieval-Augmented Generation is one of the most prominent examples. For many of these applications, the database evolves over time by inserting new data and removing outdated data. In these cases, the retrieval problem is known as streaming similarity search. While Locally-Adaptive Vector Quantization (LVQ), a highly efficient vector compression method, yields state-of-the-art search performance for non-evolving databases, its usefulness in the streaming setting has not been yet established. In this work, we study LVQ in streaming similarity search. In support of our evaluation, we introduce two improvements of LVQ: Turbo LVQ and multi-means LVQ that boost its search performance by up to 28% and 27%, respectively. Our studies show that LVQ and its new variants enable blazing fast vector search, outperforming its closest competitor by up to 9.4x for identically distributed data and by up to 8.8x under the challenging scenario of data distribution shifts (i.e., where the statistical distribution of the data changes over time). We release our contributions as part of Scalable Vector Search, an open-source library for high-performance similarity search.

{{</citation>}}


### (41/91) A Plug-in Tiny AI Module for Intelligent and Selective Sensor Data Transmission (Wenjun Huang et al., 2024)

{{<citation>}}

Wenjun Huang, Arghavan Rezvani, Hanning Chen, Yang Ni, Sanggeon Yun, Sungheon Jeong, Mohsen Imani. (2024)  
**A Plug-in Tiny AI Module for Intelligent and Selective Sensor Data Transmission**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.02043v1)  

---


**ABSTRACT**  
Applications in the Internet of Things (IoT) utilize machine learning to analyze sensor-generated data. However, a major challenge lies in the lack of targeted intelligence in current sensing systems, leading to vast data generation and increased computational and communication costs. To address this challenge, we propose a novel sensing module to equip sensing frameworks with intelligent data transmission capabilities by integrating a highly efficient machine learning model placed near the sensor. This model provides prompt feedback for the sensing system to transmit only valuable data while discarding irrelevant information by regulating the frequency of data transmission. The near-sensor model is quantized and optimized for real-time sensor control. To enhance the framework's performance, the training process is customized and a "lazy" sensor deactivation strategy utilizing temporal information is introduced. The suggested method is orthogonal to other IoT frameworks and can be considered as a plugin for selective data transmission. The framework is implemented, encompassing both software and hardware components. The experiments demonstrate that the framework utilizing the suggested module achieves over 85% system efficiency in terms of energy consumption and storage, with negligible impact on performance. This methodology has the potential to significantly reduce data output from sensors, benefiting a wide range of IoT applications.

{{</citation>}}


### (42/91) Interpreting Graph Neural Networks with In-Distributed Proxies (Zhuomin Chen et al., 2024)

{{<citation>}}

Zhuomin Chen, Jiaxing Zhang, Jingchao Ni, Xiaoting Li, Yuchen Bian, Md Mezbahul Islam, Ananda Mohan Mondal, Hua Wei, Dongsheng Luo. (2024)  
**Interpreting Graph Neural Networks with In-Distributed Proxies**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2402.02036v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have become a building block in graph data processing, with wide applications in critical domains. The growing needs to deploy GNNs in high-stakes applications necessitate explainability for users in the decision-making processes. A popular paradigm for the explainability of GNNs is to identify explainable subgraphs by comparing their labels with the ones of original graphs. This task is challenging due to the substantial distributional shift from the original graphs in the training set to the set of explainable subgraphs, which prevents accurate prediction of labels with the subgraphs. To address it, in this paper, we propose a novel method that generates proxy graphs for explainable subgraphs that are in the distribution of training data. We introduce a parametric method that employs graph generators to produce proxy graphs. A new training objective based on information theory is designed to ensure that proxy graphs not only adhere to the distribution of training data but also preserve essential explanatory factors. Such generated proxy graphs can be reliably used for approximating the predictions of the true labels of explainable subgraphs. Empirical evaluations across various datasets demonstrate our method achieves more accurate explanations for GNNs.

{{</citation>}}


### (43/91) RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies (Hao Cheng et al., 2024)

{{<citation>}}

Hao Cheng, Qingsong Wen, Yang Liu, Liang Sun. (2024)  
**RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2402.02032v1)  

---


**ABSTRACT**  
Time series forecasting is an important and forefront task in many real-world applications. However, most of time series forecasting techniques assume that the training data is clean without anomalies. This assumption is unrealistic since the collected time series data can be contaminated in practice. The forecasting model will be inferior if it is directly trained by time series with anomalies. Thus it is essential to develop methods to automatically learn a robust forecasting model from the contaminated data. In this paper, we first statistically define three types of anomalies, then theoretically and experimentally analyze the loss robustness and sample robustness when these anomalies exist. Based on our analyses, we propose a simple and efficient algorithm to learn a robust forecasting model. Extensive experiments show that our method is highly robust and outperforms all existing approaches. The code is available at https://github.com/haochenglouis/RobustTSF.

{{</citation>}}


### (44/91) Unlearnable Examples For Time Series (Yujing Jiang et al., 2024)

{{<citation>}}

Yujing Jiang, Xingjun Ma, Sarah Monazam Erfani, James Bailey. (2024)  
**Unlearnable Examples For Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2402.02028v1)  

---


**ABSTRACT**  
Unlearnable examples (UEs) refer to training samples modified to be unlearnable to Deep Neural Networks (DNNs). These examples are usually generated by adding error-minimizing noises that can fool a DNN model into believing that there is nothing (no error) to learn from the data. The concept of UE has been proposed as a countermeasure against unauthorized data exploitation on personal data. While UE has been extensively studied on images, it is unclear how to craft effective UEs for time series data. In this work, we introduce the first UE generation method to protect time series data from unauthorized training by deep learning models. To this end, we propose a new form of error-minimizing noise that can be \emph{selectively} applied to specific segments of time series, rendering them unlearnable to DNN models while remaining imperceptible to human observers. Through extensive experiments on a wide range of time series datasets, we demonstrate that the proposed UE generation method is effective in both classification and generation tasks. It can protect time series data against unauthorized exploitation, while preserving their utility for legitimate usage, thereby contributing to the development of secure and trustworthy machine learning systems.

{{</citation>}}


### (45/91) A Survey of Constraint Formulations in Safe Reinforcement Learning (Akifumi Wachi et al., 2024)

{{<citation>}}

Akifumi Wachi, Xun Shen, Yanan Sui. (2024)  
**A Survey of Constraint Formulations in Safe Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.02025v1)  

---


**ABSTRACT**  
Ensuring safety is critical when applying reinforcement learning (RL) to real-world problems. Consequently, safe RL emerges as a fundamental and powerful paradigm for safely optimizing an agent's policy from experimental data. A popular safe RL approach is based on a constrained criterion, which solves the problem of maximizing expected cumulative reward under safety constraints. Though there has been recently a surge of such attempts to achieve safety in RL, a systematic understanding of the field is difficult due to 1) the diversity of constraint representations and 2) little discussion of their interrelations. To address this knowledge gap, we provide a comprehensive review of representative constraint formulations, along with a curated selection of algorithms specifically designed for each formulation. Furthermore, we elucidate the theoretical underpinnings that reveal the mathematical mutual relations among common problem formulations. We conclude with a discussion of the current state and future directions of safe reinforcement learning research.

{{</citation>}}


### (46/91) Self-Supervised Contrastive Forecasting (Junwoo Park et al., 2024)

{{<citation>}}

Junwoo Park, Daehoon Gwak, Jaegul Choo, Edward Choi. (2024)  
**Self-Supervised Contrastive Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2402.02023v1)  

---


**ABSTRACT**  
Long-term forecasting presents unique challenges due to the time and memory complexity of handling long sequences. Existing methods, which rely on sliding windows to process long sequences, struggle to effectively capture long-term variations that are partially caught within the short window (i.e., outer-window variations). In this paper, we introduce a novel approach that overcomes this limitation by employing contrastive learning and enhanced decomposition architecture, specifically designed to focus on long-term variations. To this end, our contrastive loss incorporates global autocorrelation held in the whole time series, which facilitates the construction of positive and negative pairs in a self-supervised manner. When combined with our decomposition networks, our contrastive learning significantly improves long-term forecasting performance. Extensive experiments demonstrate that our approach outperforms 14 baseline models in multiple experiments over nine long-term benchmarks, especially in challenging scenarios that require a significantly long output for forecasting. Source code is available at https://github.com/junwoopark92/Self-Supervised-Contrastive-Forecsating.

{{</citation>}}


### (47/91) GenFormer: A Deep-Learning-Based Approach for Generating Multivariate Stochastic Processes (Haoran Zhao et al., 2024)

{{<citation>}}

Haoran Zhao, Wayne Isaac Tan Uy. (2024)  
**GenFormer: A Deep-Learning-Based Approach for Generating Multivariate Stochastic Processes**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.02010v1)  

---


**ABSTRACT**  
Stochastic generators are essential to produce synthetic realizations that preserve target statistical properties. We propose GenFormer, a stochastic generator for spatio-temporal multivariate stochastic processes. It is constructed using a Transformer-based deep learning model that learns a mapping between a Markov state sequence and time series values. The synthetic data generated by the GenFormer model preserves the target marginal distributions and approximately captures other desired statistical properties even in challenging applications involving a large number of spatial locations and a long simulation horizon. The GenFormer model is applied to simulate synthetic wind speed data at various stations in Florida to calculate exceedance probabilities for risk management.

{{</citation>}}


### (48/91) Understanding Time Series Anomaly State Detection through One-Class Classification (Hanxu Zhou et al., 2024)

{{<citation>}}

Hanxu Zhou, Yuan Zhang, Guangjie Leng, Ruofan Wang, Zhi-Qin John Xu. (2024)  
**Understanding Time Series Anomaly State Detection through One-Class Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2402.02007v1)  

---


**ABSTRACT**  
For a long time, research on time series anomaly detection has mainly focused on finding outliers within a given time series. Admittedly, this is consistent with some practical problems, but in other practical application scenarios, people are concerned about: assuming a standard time series is given, how to judge whether another test time series deviates from the standard time series, which is more similar to the problem discussed in one-class classification (OCC). Therefore, in this article, we try to re-understand and define the time series anomaly detection problem through OCC, which we call 'time series anomaly state detection problem'. We first use stochastic processes and hypothesis testing to strictly define the 'time series anomaly state detection problem', and its corresponding anomalies. Then, we use the time series classification dataset to construct an artificial dataset corresponding to the problem. We compile 38 anomaly detection algorithms and correct some of the algorithms to adapt to handle this problem. Finally, through a large number of experiments, we fairly compare the actual performance of various time series anomaly detection algorithms, providing insights and directions for future research by researchers.

{{</citation>}}


### (49/91) PresAIse, An Enterprises Prescriptive AI Solution (Wei Sun et al., 2024)

{{<citation>}}

Wei Sun, Scott McFaddin, Linh Ha Tran, Shivaram Subramanian, Kristjan Greenewald, Yeshi Tenzin, Zack Xue, Youssef Drissi, Markus Ettl. (2024)  
**PresAIse, An Enterprises Prescriptive AI Solution**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.02006v1)  

---


**ABSTRACT**  
Prescriptive AI represents a transformative shift in decision-making, offering causal insights and actionable recommendations. Despite its huge potential, enterprise adoption often faces several challenges. The first challenge is caused by the limitations of observational data for accurate causal inference which is typically a prerequisite for good decision-making. The second pertains to the interpretability of recommendations, which is crucial for enterprise decision-making settings. The third challenge is the silos between data scientists and business users, hindering effective collaboration. This paper outlines an initiative from IBM Research, aiming to address some of these challenges by offering a suite of prescriptive AI solutions. Leveraging insights from various research papers, the solution suite includes scalable causal inference methods, interpretable decision-making approaches, and the integration of large language models (LLMs) to bridge communication gaps via a conversation agent. A proof-of-concept, PresAIse, demonstrates the solutions' potential by enabling non-ML experts to interact with prescriptive AI models via a natural language interface, democratizing advanced analytics for strategic decision-making.

{{</citation>}}


### (50/91) Topology-Informed Graph Transformer (Yun Young Choi et al., 2024)

{{<citation>}}

Yun Young Choi, Sun Woo Park, Minho Lee, Youngho Woo. (2024)  
**Topology-Informed Graph Transformer**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Natural Language Processing, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2402.02005v1)  

---


**ABSTRACT**  
Transformers have revolutionized performance in Natural Language Processing and Vision, paving the way for their integration with Graph Neural Networks (GNNs). One key challenge in enhancing graph transformers is strengthening the discriminative power of distinguishing isomorphisms of graphs, which plays a crucial role in boosting their predictive performances. To address this challenge, we introduce 'Topology-Informed Graph Transformer (TIGT)', a novel transformer enhancing both discriminative power in detecting graph isomorphisms and the overall performance of Graph Transformers. TIGT consists of four components: A topological positional embedding layer using non-isomorphic universal covers based on cyclic subgraphs of graphs to ensure unique graph representation: A dual-path message-passing layer to explicitly encode topological characteristics throughout the encoder layers: A global attention mechanism: And a graph information layer to recalibrate channel-wise graph features for better feature representation. TIGT outperforms previous Graph Transformers in classifying synthetic dataset aimed at distinguishing isomorphism classes of graphs. Additionally, mathematical analysis and empirical evaluations highlight our model's competitive edge over state-of-the-art Graph Transformers across various benchmark datasets.

{{</citation>}}


### (51/91) A Novel Hyperdimensional Computing Framework for Online Time Series Forecasting on the Edge (Mohamed Mejri et al., 2024)

{{<citation>}}

Mohamed Mejri, Chandramouli Amarnath, Abhijit Chatterjee. (2024)  
**A Novel Hyperdimensional Computing Framework for Online Time Series Forecasting on the Edge**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2402.01999v1)  

---


**ABSTRACT**  
In recent years, both online and offline deep learning models have been developed for time series forecasting. However, offline deep forecasting models fail to adapt effectively to changes in time-series data, while online deep forecasting models are often expensive and have complex training procedures. In this paper, we reframe the online nonlinear time-series forecasting problem as one of linear hyperdimensional time-series forecasting. Nonlinear low-dimensional time-series data is mapped to high-dimensional (hyperdimensional) spaces for linear hyperdimensional prediction, allowing fast, efficient and lightweight online time-series forecasting. Our framework, TSF-HD, adapts to time-series distribution shifts using a novel co-training framework for its hyperdimensional mapping and its linear hyperdimensional predictor. TSF-HD is shown to outperform the state of the art, while having reduced inference latency, for both short-term and long-term time series forecasting. Our code is publicly available at http://github.com/tsfhd2024/tsf-hd.git

{{</citation>}}


### (52/91) Online Uniform Risk Times Sampling: First Approximation Algorithms, Learning Augmentation with Full Confidence Interval Integration (Xueqing Liu et al., 2024)

{{<citation>}}

Xueqing Liu, Kyra Gan, Esmaeil Keyvanshokooh, Susan Murphy. (2024)  
**Online Uniform Risk Times Sampling: First Approximation Algorithms, Learning Augmentation with Full Confidence Interval Integration**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2402.01995v1)  

---


**ABSTRACT**  
In digital health, the strategy of allocating a limited treatment budget across available risk times is crucial to reduce user fatigue. This strategy, however, encounters a significant obstacle due to the unknown actual number of risk times, a factor not adequately addressed by existing methods lacking theoretical guarantees. This paper introduces, for the first time, the online uniform risk times sampling problem within the approximation algorithm framework. We propose two online approximation algorithms for this problem, one with and one without learning augmentation, and provide rigorous theoretical performance guarantees for them using competitive ratio analysis. We assess the performance of our algorithms using both synthetic experiments and a real-world case study on HeartSteps mobile applications.

{{</citation>}}


### (53/91) Simulation-Enhanced Data Augmentation for Machine Learning Pathloss Prediction (Ahmed P. Mohamed et al., 2024)

{{<citation>}}

Ahmed P. Mohamed, Byunghyun Lee, Yaguang Zhang, Max Hollingsworth, C. Robert Anderson, James V. Krogmeier, David J. Love. (2024)  
**Simulation-Enhanced Data Augmentation for Machine Learning Pathloss Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2402.01969v2)  

---


**ABSTRACT**  
Machine learning (ML) offers a promising solution to pathloss prediction. However, its effectiveness can be degraded by the limited availability of data. To alleviate these challenges, this paper introduces a novel simulation-enhanced data augmentation method for ML pathloss prediction. Our method integrates synthetic data generated from a cellular coverage simulator and independently collected real-world datasets. These datasets were collected through an extensive measurement campaign in different environments, including farms, hilly terrains, and residential areas. This comprehensive data collection provides vital ground truth for model training. A set of channel features was engineered, including geographical attributes derived from LiDAR datasets. These features were then used to train our prediction model, incorporating the highly efficient and robust gradient boosting ML algorithm, CatBoost. The integration of synthetic data, as demonstrated in our study, significantly improves the generalizability of the model in different environments, achieving a remarkable improvement of approximately 12dB in terms of mean absolute error for the best-case scenario. Moreover, our analysis reveals that even a small fraction of measurements added to the simulation training set, with proper data balance, can significantly enhance the model's performance.

{{</citation>}}


### (54/91) No Need to Look Back: An Efficient and Scalable Approach for Temporal Network Representation Learning (Yuhong Luo et al., 2024)

{{<citation>}}

Yuhong Luo, Pan Li. (2024)  
**No Need to Look Back: An Efficient and Scalable Approach for Temporal Network Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2402.01964v1)  

---


**ABSTRACT**  
Temporal graph representation learning (TGRL) is crucial for modeling complex, dynamic systems in real-world networks. Traditional TGRL methods, though effective, suffer from high computational demands and inference latency. This is mainly induced by their inefficient sampling of temporal neighbors by backtracking the interaction history of each node when making model inference. This paper introduces a novel efficient TGRL framework, No-Looking-Back (NLB). NLB employs a "forward recent sampling" strategy, which bypasses the need for backtracking historical interactions. This strategy is implemented using a GPU-executable size-constrained hash table for each node, recording down-sampled recent interactions, which enables rapid response to queries with minimal inference latency. The maintenance of this hash table is highly efficient, with $O(1)$ complexity. NLB is fully compatible with GPU processing, maximizing programmability, parallelism, and power efficiency. Empirical evaluations demonstrate that NLB matches or surpasses state-of-the-art methods in accuracy for link prediction and node classification across six real-world datasets. Significantly, it is 1.32-4.40 $\times$ faster in training, 1.2-7.94 $\times$ more energy efficient, and 1.97-5.02 $\times$ more effective in reducing inference latency compared to the most competitive baselines. The link to the code: https://github.com/Graph-COM/NLB.

{{</citation>}}


## cs.CV (21)



### (55/91) Multi-Level Feature Aggregation and Recursive Alignment Network for Real-Time Semantic Segmentation (Yanhua Zhang et al., 2024)

{{<citation>}}

Yanhua Zhang, Ke Zhang, Jingyu Wang, Yulin Wu, Wuwei Wang. (2024)  
**Multi-Level Feature Aggregation and Recursive Alignment Network for Real-Time Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2402.02286v1)  

---


**ABSTRACT**  
Real-time semantic segmentation is a crucial research for real-world applications. However, many methods lay particular emphasis on reducing the computational complexity and model size, while largely sacrificing the accuracy. In some scenarios, such as autonomous navigation and driver assistance system, accuracy and speed are equally important. To tackle this problem, we propose a novel Multi-level Feature Aggregation and Recursive Alignment Network (MFARANet), aiming to achieve high segmentation accuracy at real-time inference speed. We employ ResNet-18 as the backbone to ensure efficiency, and propose three core components to compensate for the reduced model capacity due to the shallow backbone. Specifically, we first design Multi-level Feature Aggregation Module (MFAM) to aggregate the hierarchical features in the encoder to each scale to benefit subsequent spatial alignment and multi-scale inference. Then, we build Recursive Alignment Module (RAM) by combining the flow-based alignment module with recursive upsampling architecture for accurate and efficient spatial alignment between multi-scale score maps. Finally, the Adaptive Scores Fusion Module (ASFM) is proposed to adaptively fuse multi-scale scores so that the final prediction can favor objects of multiple scales. Comprehensive experiments on three benchmark datasets including Cityscapes, CamVid and PASCAL-Context show the effectiveness and efficiency of our method. In particular, we achieve a better balance between speed and accuracy than state-of-the-art real-time methods on Cityscapes and CamVid datasets. Code is available at: https://github.com/Yanhua-Zhang/MFARANet.

{{</citation>}}


### (56/91) ExTTNet: A Deep Learning Algorithm for Extracting Table Texts from Invoice Images (Adem Akdoğan et al., 2024)

{{<citation>}}

Adem Akdoğan, Murat Kurt. (2024)  
**ExTTNet: A Deep Learning Algorithm for Extracting Table Texts from Invoice Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-IR, cs-LG, cs-NE, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2402.02246v1)  

---


**ABSTRACT**  
In this work, product tables in invoices are obtained autonomously via a deep learning model, which is named as ExTTNet. Firstly, text is obtained from invoice images using Optical Character Recognition (OCR) techniques. Tesseract OCR engine [37] is used for this process. Afterwards, the number of existing features is increased by using feature extraction methods to increase the accuracy. Labeling process is done according to whether each text obtained as a result of OCR is a table element or not. In this study, a multilayer artificial neural network model is used. The training has been carried out with an Nvidia RTX 3090 graphics card and taken $162$ minutes. As a result of the training, the F1 score is $0.92$.

{{</citation>}}


### (57/91) Revisiting Generative Adversarial Networks for Binary Semantic Segmentation on Imbalanced Datasets (Lei Xu et al., 2024)

{{<citation>}}

Lei Xu, Moncef Gabbouj. (2024)  
**Revisiting Generative Adversarial Networks for Binary Semantic Segmentation on Imbalanced Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2402.02245v1)  

---


**ABSTRACT**  
Anomalous pavement surface conditions detection aims to detect pixels representing anomalous states, such as cracks, on pavement surface images automatically by algorithms. Recently, deep learning models have been intensively applied to related topics with outstanding performance. However, most existing deep learning-related solutions rarely achieve a stable performance on diverse datasets. To address this issue, in this work, we propose a deep learning framework based on conditional Generative Adversarial Networks for anomalous region detection on pavement images at the pixel level. In particular, the proposed framework is developed to enhance the generator's ability to estimate the probability feature map from heterogeneous inputs with two training stages and multiscale feature representation. Moreover, several attention mechanisms are incorporated into the proposed framework to mitigate the performance deterioration of model training on severely imbalanced datasets. We implement experiments on six accessible pavement datasets. Extensive qualitative and quantitative experiments demonstrate that the proposed framework can achieve SOTA results on these datasets efficiently and robustly.

{{</citation>}}


### (58/91) Image Fusion via Vision-Language Model (Zixiang Zhao et al., 2024)

{{<citation>}}

Zixiang Zhao, Lilun Deng, Haowen Bai, Yukun Cui, Zhipeng Zhang, Yulun Zhang, Haotong Qin, Dongdong Chen, Jiangshe Zhang, Peng Wang, Luc Van Gool. (2024)  
**Image Fusion via Vision-Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2402.02235v1)  

---


**ABSTRACT**  
Image fusion integrates essential information from multiple source images into a single composite, emphasizing the highlighting structure and textures, and refining imperfect areas. Existing methods predominantly focus on pixel-level and semantic visual features for recognition. However, they insufficiently explore the deeper semantic information at a text-level beyond vision. Therefore, we introduce a novel fusion paradigm named image Fusion via vIsion-Language Model (FILM), for the first time, utilizing explicit textual information in different source images to guide image fusion. In FILM, input images are firstly processed to generate semantic prompts, which are then fed into ChatGPT to obtain rich textual descriptions. These descriptions are fused in the textual domain and guide the extraction of crucial visual features from the source images through cross-attention, resulting in a deeper level of contextual understanding directed by textual semantic information. The final fused image is created by vision feature decoder. This paradigm achieves satisfactory results in four image fusion tasks: infrared-visible, medical, multi-exposure, and multi-focus image fusion. We also propose a vision-language dataset containing ChatGPT-based paragraph descriptions for the ten image fusion datasets in four fusion tasks, facilitating future research in vision-language model-based image fusion. Code and dataset will be released.

{{</citation>}}


### (59/91) CoFiNet: Unveiling Camouflaged Objects with Multi-Scale Finesse (Cunhan Guo et al., 2024)

{{<citation>}}

Cunhan Guo, Heyan Huang. (2024)  
**CoFiNet: Unveiling Camouflaged Objects with Multi-Scale Finesse**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2402.02217v1)  

---


**ABSTRACT**  
Camouflaged Object Detection (COD) is a critical aspect of computer vision aimed at identifying concealed objects, with applications spanning military, industrial, medical and monitoring domains. To address the problem of poor detail segmentation effect, we introduce a novel method for camouflage object detection, named CoFiNet. Our approach primarily focuses on multi-scale feature fusion and extraction, with special attention to the model's segmentation effectiveness for detailed features, enhancing its ability to effectively detect camouflaged objects. CoFiNet adopts a coarse-to-fine strategy. A multi-scale feature integration module is laveraged to enhance the model's capability of fusing context feature. A multi-activation selective kernel module is leveraged to grant the model the ability to autonomously alter its receptive field, enabling it to selectively choose an appropriate receptive field for camouflaged objects of different sizes. During mask generation, we employ the dual-mask strategy for image segmentation, separating the reconstruction of coarse and fine masks, which significantly enhances the model's learning capacity for details. Comprehensive experiments were conducted on four different datasets, demonstrating that CoFiNet achieves state-of-the-art performance across all datasets. The experiment results of CoFiNet underscore its effectiveness in camouflage object detection and highlight its potential in various practical application scenarios.

{{</citation>}}


### (60/91) Wavelet-Decoupling Contrastive Enhancement Network for Fine-Grained Skeleton-Based Action Recognition (Haochen Chang et al., 2024)

{{<citation>}}

Haochen Chang, Jing Chen, Yilin Li, Jixiang Chen, Xiaofeng Zhang. (2024)  
**Wavelet-Decoupling Contrastive Enhancement Network for Fine-Grained Skeleton-Based Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2402.02210v1)  

---


**ABSTRACT**  
Skeleton-based action recognition has attracted much attention, benefiting from its succinctness and robustness. However, the minimal inter-class variation in similar action sequences often leads to confusion. The inherent spatiotemporal coupling characteristics make it challenging to mine the subtle differences in joint motion trajectories, which is critical for distinguishing confusing fine-grained actions. To alleviate this problem, we propose a Wavelet-Attention Decoupling (WAD) module that utilizes discrete wavelet transform to effectively disentangle salient and subtle motion features in the time-frequency domain. Then, the decoupling attention adaptively recalibrates their temporal responses. To further amplify the discrepancies in these subtle motion features, we propose a Fine-grained Contrastive Enhancement (FCE) module to enhance attention towards trajectory features by contrastive learning. Extensive experiments are conducted on the coarse-grained dataset NTU RGB+D and the fine-grained dataset FineGYM. Our methods perform competitively compared to state-of-the-art methods and can discriminate confusing fine-grained actions well.

{{</citation>}}


### (61/91) On the Exploitation of DCT-Traces in the Generative-AI Domain (Orazio Pontorno et al., 2024)

{{<citation>}}

Orazio Pontorno, Luca Guarnera, Sebastiano Battiato. (2024)  
**On the Exploitation of DCT-Traces in the Generative-AI Domain**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.02209v1)  

---


**ABSTRACT**  
Since their appearance, Deepfakes represent one of the toughest challenges in the world of Cybersecurity and Digital Forensics. In recent years, researchers have discovered that generative models leave unique traces in synthetic data that, if analyzed and identified in detail, can be exploited to improve the generalization limitations of existing deepfake detectors. To capture this evidence, in this paper we analyzed deepfake images in the frequency domain, examining in detail the beta-AC coefficients of the Discrete Cosine Transform (DCT). Recognizing that not all coefficients contribute equally to image recognition, we hypothesize the existence of a unique "discriminative fingerprint" for each type of image, embedded in specific combinations of coefficients. To identify them, Machine Learning classifiers were trained on various combinations of coefficients. The integration of the Explainable AI (XAI) LIME algorithm combined with a neural classifier to explore alternative combinations of coefficients provides a deeper insight into the discriminative features of synthetic images. Experimental results reveal the significant potential of using a specific combination of beta-AC coefficients in order to improve the analysis of traces left by generative models.

{{</citation>}}


### (62/91) GPT-4V as Traffic Assistant: An In-depth Look at Vision Language Model on Complex Traffic Events (Xingcheng Zhou et al., 2024)

{{<citation>}}

Xingcheng Zhou, Alois C. Knoll. (2024)  
**GPT-4V as Traffic Assistant: An In-depth Look at Vision Language Model on Complex Traffic Events**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2402.02205v2)  

---


**ABSTRACT**  
The recognition and understanding of traffic incidents, particularly traffic accidents, is a topic of paramount importance in the realm of intelligent transportation systems and intelligent vehicles. This area has continually captured the extensive focus of both the academic and industrial sectors. Identifying and comprehending complex traffic events is highly challenging, primarily due to the intricate nature of traffic environments, diverse observational perspectives, and the multifaceted causes of accidents. These factors have persistently impeded the development of effective solutions. The advent of large vision-language models (VLMs) such as GPT-4V, has introduced innovative approaches to addressing this issue. In this paper, we explore the ability of GPT-4V with a set of representative traffic incident videos and delve into the model's capacity of understanding these complex traffic situations. We observe that GPT-4V demonstrates remarkable cognitive, reasoning, and decision-making ability in certain classic traffic events. Concurrently, we also identify certain limitations of GPT-4V, which constrain its understanding in more intricate scenarios. These limitations merit further exploration and resolution.

{{</citation>}}


### (63/91) IMUSIC: IMU-based Facial Expression Capture (Youjia Wang et al., 2024)

{{<citation>}}

Youjia Wang, Yiwen Wu, Ruiqian Li, Hengan Zhou, Hongyang Lin, Yingwenqi Jiang, Yingsheng Zhu, Guanpeng Long, Jingya Wang, Lan Xu, Jingyi Yu. (2024)  
**IMUSIC: IMU-based Facial Expression Capture**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.03944v1)  

---


**ABSTRACT**  
For facial motion capture and analysis, the dominated solutions are generally based on visual cues, which cannot protect privacy and are vulnerable to occlusions. Inertial measurement units (IMUs) serve as potential rescues yet are mainly adopted for full-body motion capture. In this paper, we propose IMUSIC to fill the gap, a novel path for facial expression capture using purely IMU signals, significantly distant from previous visual solutions.The key design in our IMUSIC is a trilogy. We first design micro-IMUs to suit facial capture, companion with an anatomy-driven IMU placement scheme. Then, we contribute a novel IMU-ARKit dataset, which provides rich paired IMU/visual signals for diverse facial expressions and performances. Such unique multi-modality brings huge potential for future directions like IMU-based facial behavior analysis. Moreover, utilizing IMU-ARKit, we introduce a strong baseline approach to accurately predict facial blendshape parameters from purely IMU signals. Specifically, we tailor a Transformer diffusion model with a two-stage training strategy for this novel tracking task. The IMUSIC framework empowers us to perform accurate facial capture in scenarios where visual methods falter and simultaneously safeguard user privacy. We conduct extensive experiments about both the IMU configuration and technical components to validate the effectiveness of our IMUSIC approach. Notably, IMUSIC enables various potential and novel applications, i.e., privacy-protecting facial capture, hybrid capture against occlusions, or detecting minute facial movements that are often invisible through visual cues. We will release our dataset and implementations to enrich more possibilities of facial capture and analysis in our community.

{{</citation>}}


### (64/91) Evaluating the Robustness of Off-Road Autonomous Driving Segmentation against Adversarial Attacks: A Dataset-Centric analysis (Pankaj Deoli et al., 2024)

{{<citation>}}

Pankaj Deoli, Rohit Kumar, Axel Vierling, Karsten Berns. (2024)  
**Evaluating the Robustness of Off-Road Autonomous Driving Segmentation against Adversarial Attacks: A Dataset-Centric analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2402.02154v1)  

---


**ABSTRACT**  
This study investigates the vulnerability of semantic segmentation models to adversarial input perturbations, in the domain of off-road autonomous driving. Despite good performance in generic conditions, the state-of-the-art classifiers are often susceptible to (even) small perturbations, ultimately resulting in inaccurate predictions with high confidence. Prior research has directed their focus on making models more robust by modifying the architecture and training with noisy input images, but has not explored the influence of datasets in adversarial attacks. Our study aims to address this gap by examining the impact of non-robust features in off-road datasets and comparing the effects of adversarial attacks on different segmentation network architectures. To enable this, a robust dataset is created consisting of only robust features and training the networks on this robustified dataset. We present both qualitative and quantitative analysis of our findings, which have important implications on improving the robustness of machine learning models in off-road autonomous driving applications. Additionally, this work contributes to the safe navigation of autonomous robot Unimog U5023 in rough off-road unstructured environments by evaluating the robustness of segmentation outputs. The code is publicly available at https://github.com/rohtkumar/adversarial_attacks_ on_segmentation

{{</citation>}}


### (65/91) Generative Visual Compression: A Review (Bolin Chen et al., 2024)

{{<citation>}}

Bolin Chen, Shanzhi Yin, Peilin Chen, Shiqi Wang, Yan Ye. (2024)  
**Generative Visual Compression: A Review**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.02140v1)  

---


**ABSTRACT**  
Artificial Intelligence Generated Content (AIGC) is leading a new technical revolution for the acquisition of digital content and impelling the progress of visual compression towards competitive performance gains and diverse functionalities over traditional codecs. This paper provides a thorough review on the recent advances of generative visual compression, illustrating great potentials and promising applications in ultra-low bitrate communication, user-specified reconstruction/filtering, and intelligent machine analysis. In particular, we review the visual data compression methodologies with deep generative models, and summarize how compact representation and high-fidelity reconstruction could be actualized via generative techniques. In addition, we generalize related generative compression technologies for machine vision and intelligent analytics. Finally, we discuss the fundamental challenges on generative visual compression techniques and envision their future research directions.

{{</citation>}}


### (66/91) ParZC: Parametric Zero-Cost Proxies for Efficient NAS (Peijie Dong et al., 2024)

{{<citation>}}

Peijie Dong, Lujun Li, Xinglin Pan, Zimian Wei, Xiang Liu, Qiang Wang, Xiaowen Chu. (2024)  
**ParZC: Parametric Zero-Cost Proxies for Efficient NAS**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.02105v1)  

---


**ABSTRACT**  
Recent advancements in Zero-shot Neural Architecture Search (NAS) highlight the efficacy of zero-cost proxies in various NAS benchmarks. Several studies propose the automated design of zero-cost proxies to achieve SOTA performance but require tedious searching progress. Furthermore, we identify a critical issue with current zero-cost proxies: they aggregate node-wise zero-cost statistics without considering the fact that not all nodes in a neural network equally impact performance estimation. Our observations reveal that node-wise zero-cost statistics significantly vary in their contributions to performance, with each node exhibiting a degree of uncertainty. Based on this insight, we introduce a novel method called Parametric Zero-Cost Proxies (ParZC) framework to enhance the adaptability of zero-cost proxies through parameterization. To address the node indiscrimination, we propose a Mixer Architecture with Bayesian Network (MABN) to explore the node-wise zero-cost statistics and estimate node-specific uncertainty. Moreover, we propose DiffKendall as a loss function to directly optimize Kendall's Tau coefficient in a differentiable manner so that our ParZC can better handle the discrepancies in ranking architectures. Comprehensive experiments on NAS-Bench-101, 201, and NDS demonstrate the superiority of our proposed ParZC compared to existing zero-shot NAS methods. Additionally, we demonstrate the versatility and adaptability of ParZC by transferring it to the Vision Transformer search space.

{{</citation>}}


### (67/91) Déjà Vu Memorization in Vision-Language Models (Bargav Jayaraman et al., 2024)

{{<citation>}}

Bargav Jayaraman, Chuan Guo, Kamalika Chaudhuri. (2024)  
**Déjà Vu Memorization in Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2402.02103v1)  

---


**ABSTRACT**  
Vision-Language Models (VLMs) have emerged as the state-of-the-art representation learning solution, with myriads of downstream applications such as image classification, retrieval and generation. A natural question is whether these models memorize their training data, which also has implications for generalization. We propose a new method for measuring memorization in VLMs, which we call d\'ej\`a vu memorization. For VLMs trained on image-caption pairs, we show that the model indeed retains information about individual objects in the training images beyond what can be inferred from correlations or the image caption. We evaluate d\'ej\`a vu memorization at both sample and population level, and show that it is significant for OpenCLIP trained on as many as 50M image-caption pairs. Finally, we show that text randomization considerably mitigates memorization while only moderately impacting the model's downstream task performance.

{{</citation>}}


### (68/91) Deep Semantic-Visual Alignment for Zero-Shot Remote Sensing Image Scene Classification (Wenjia Xu et al., 2024)

{{<citation>}}

Wenjia Xu, Jiuniu Wang, Zhiwei Wei, Mugen Peng, Yirong Wu. (2024)  
**Deep Semantic-Visual Alignment for Zero-Shot Remote Sensing Image Scene Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2402.02094v1)  

---


**ABSTRACT**  
Deep neural networks have achieved promising progress in remote sensing (RS) image classification, for which the training process requires abundant samples for each class. However, it is time-consuming and unrealistic to annotate labels for each RS category, given the fact that the RS target database is increasing dynamically. Zero-shot learning (ZSL) allows for identifying novel classes that are not seen during training, which provides a promising solution for the aforementioned problem. However, previous ZSL models mainly depend on manually-labeled attributes or word embeddings extracted from language models to transfer knowledge from seen classes to novel classes. Besides, pioneer ZSL models use convolutional neural networks pre-trained on ImageNet, which focus on the main objects appearing in each image, neglecting the background context that also matters in RS scene classification. To address the above problems, we propose to collect visually detectable attributes automatically. We predict attributes for each class by depicting the semantic-visual similarity between attributes and images. In this way, the attribute annotation process is accomplished by machine instead of human as in other methods. Moreover, we propose a Deep Semantic-Visual Alignment (DSVA) that take advantage of the self-attention mechanism in the transformer to associate local image regions together, integrating the background context information for prediction. The DSVA model further utilizes the attribute attention maps to focus on the informative image regions that are essential for knowledge transfer in ZSL, and maps the visual images into attribute space to perform ZSL classification. With extensive experiments, we show that our model outperforms other state-of-the-art models by a large margin on a challenging large-scale RS scene classification benchmark.

{{</citation>}}


### (69/91) Multiple-Crop Human Mesh Recovery with Contrastive Learning and Camera Consistency in A Single Image (Yongwei Nie et al., 2024)

{{<citation>}}

Yongwei Nie, Changzhen Liu, Chengjiang Long, Qing Zhang, Guiqing Li, Hongmin Cai. (2024)  
**Multiple-Crop Human Mesh Recovery with Contrastive Learning and Camera Consistency in A Single Image**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2402.02074v1)  

---


**ABSTRACT**  
We tackle the problem of single-image Human Mesh Recovery (HMR). Previous approaches are mostly based on a single crop. In this paper, we shift the single-crop HMR to a novel multiple-crop HMR paradigm. Cropping a human from image multiple times by shifting and scaling the original bounding box is feasible in practice, easy to implement, and incurs neglectable cost, but immediately enriches available visual details. With multiple crops as input, we manage to leverage the relation among these crops to extract discriminative features and reduce camera ambiguity. Specifically, (1) we incorporate a contrastive learning scheme to enhance the similarity between features extracted from crops of the same human. (2) We also propose a crop-aware fusion scheme to fuse the features of multiple crops for regressing the target mesh. (3) We compute local cameras for all the input crops and build a camera-consistency loss between the local cameras, which reward us with less ambiguous cameras. Based on the above innovations, our proposed method outperforms previous approaches as demonstrated by the extensive experiments.

{{</citation>}}


### (70/91) DiffVein: A Unified Diffusion Network for Finger Vein Segmentation and Authentication (Yanjun Liu et al., 2024)

{{<citation>}}

Yanjun Liu, Wenming Yang, Qingmin Liao. (2024)  
**DiffVein: A Unified Diffusion Network for Finger Vein Segmentation and Authentication**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.02060v1)  

---


**ABSTRACT**  
Finger vein authentication, recognized for its high security and specificity, has become a focal point in biometric research. Traditional methods predominantly concentrate on vein feature extraction for discriminative modeling, with a limited exploration of generative approaches. Suffering from verification failure, existing methods often fail to obtain authentic vein patterns by segmentation. To fill this gap, we introduce DiffVein, a unified diffusion model-based framework which simultaneously addresses vein segmentation and authentication tasks. DiffVein is composed of two dedicated branches: one for segmentation and the other for denoising. For better feature interaction between these two branches, we introduce two specialized modules to improve their collective performance. The first, a mask condition module, incorporates the semantic information of vein patterns from the segmentation branch into the denoising process. Additionally, we also propose a Semantic Difference Transformer (SD-Former), which employs Fourier-space self-attention and cross-attention modules to extract category embedding before feeding it to the segmentation task. In this way, our framework allows for a dynamic interplay between diffusion and segmentation embeddings, thus vein segmentation and authentication tasks can inform and enhance each other in the joint training. To further optimize our model, we introduce a Fourier-space Structural Similarity (FourierSIM) loss function, which is tailored to improve the denoising network's learning efficacy. Extensive experiments on the USM and THU-MVFV3V datasets substantiates DiffVein's superior performance, setting new benchmarks in both vein segmentation and authentication tasks.

{{</citation>}}


### (71/91) TCI-Former: Thermal Conduction-Inspired Transformer for Infrared Small Target Detection (Tianxiang Chen et al., 2024)

{{<citation>}}

Tianxiang Chen, Zhentao Tan, Qi Chu, Yue Wu, Bin Liu, Nenghai Yu. (2024)  
**TCI-Former: Thermal Conduction-Inspired Transformer for Infrared Small Target Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2402.02046v1)  

---


**ABSTRACT**  
Infrared small target detection (ISTD) is critical to national security and has been extensively applied in military areas. ISTD aims to segment small target pixels from background. Most ISTD networks focus on designing feature extraction blocks or feature fusion modules, but rarely describe the ISTD process from the feature map evolution perspective. In the ISTD process, the network attention gradually shifts towards target areas. We abstract this process as the directional movement of feature map pixels to target areas through convolution, pooling and interactions with surrounding pixels, which can be analogous to the movement of thermal particles constrained by surrounding variables and particles. In light of this analogy, we propose Thermal Conduction-Inspired Transformer (TCI-Former) based on the theoretical principles of thermal conduction. According to thermal conduction differential equation in heat dynamics, we derive the pixel movement differential equation (PMDE) in the image domain and further develop two modules: Thermal Conduction-Inspired Attention (TCIA) and Thermal Conduction Boundary Module (TCBM). TCIA incorporates finite difference method with PMDE to reach a numerical approximation so that target body features can be extracted. To further remove errors in boundary areas, TCBM is designed and supervised by boundary masks to refine target body features with fine boundary details. Experiments on IRSTD-1k and NUAA-SIRST demonstrate the superiority of our method.

{{</citation>}}


### (72/91) MLIP: Enhancing Medical Visual Representation with Divergence Encoder and Knowledge-guided Contrastive Learning (Zhe Li et al., 2024)

{{<citation>}}

Zhe Li, Laurence T. Yang, Bocheng Ren, Xin Nie, Zhangyang Gao, Cheng Tan, Stan Z. Li. (2024)  
**MLIP: Enhancing Medical Visual Representation with Divergence Encoder and Knowledge-guided Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2402.02045v1)  

---


**ABSTRACT**  
The scarcity of annotated data has sparked significant interest in unsupervised pre-training methods that leverage medical reports as auxiliary signals for medical visual representation learning. However, existing research overlooks the multi-granularity nature of medical visual representation and lacks suitable contrastive learning techniques to improve the models' generalizability across different granularities, leading to the underutilization of image-text information. To address this, we propose MLIP, a novel framework leveraging domain-specific medical knowledge as guiding signals to integrate language information into the visual domain through image-text contrastive learning. Our model includes global contrastive learning with our designed divergence encoder, local token-knowledge-patch alignment contrastive learning, and knowledge-guided category-level contrastive learning with expert knowledge. Experimental evaluations reveal the efficacy of our model in enhancing transfer performance for tasks such as image classification, object detection, and semantic segmentation. Notably, MLIP surpasses state-of-the-art methods even with limited annotated data, highlighting the potential of multimodal pre-training in advancing medical representation learning.

{{</citation>}}


### (73/91) ScribFormer: Transformer Makes CNN Work Better for Scribble-based Medical Image Segmentation (Zihan Li et al., 2024)

{{<citation>}}

Zihan Li, Yuan Zheng, Dandan Shan, Shuzhou Yang, Qingde Li, Beizhan Wang, Yuanting Zhang, Qingqi Hong, Dinggang Shen. (2024)  
**ScribFormer: Transformer Makes CNN Work Better for Scribble-based Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.02029v1)  

---


**ABSTRACT**  
Most recent scribble-supervised segmentation methods commonly adopt a CNN framework with an encoder-decoder architecture. Despite its multiple benefits, this framework generally can only capture small-range feature dependency for the convolutional layer with the local receptive field, which makes it difficult to learn global shape information from the limited information provided by scribble annotations. To address this issue, this paper proposes a new CNN-Transformer hybrid solution for scribble-supervised medical image segmentation called ScribFormer. The proposed ScribFormer model has a triple-branch structure, i.e., the hybrid of a CNN branch, a Transformer branch, and an attention-guided class activation map (ACAM) branch. Specifically, the CNN branch collaborates with the Transformer branch to fuse the local features learned from CNN with the global representations obtained from Transformer, which can effectively overcome limitations of existing scribble-supervised segmentation methods. Furthermore, the ACAM branch assists in unifying the shallow convolution features and the deep convolution features to improve model's performance further. Extensive experiments on two public datasets and one private dataset show that our ScribFormer has superior performance over the state-of-the-art scribble-supervised segmentation methods, and achieves even better results than the fully-supervised segmentation methods. The code is released at https://github.com/HUANGLIZI/ScribFormer.

{{</citation>}}


### (74/91) Precise Knowledge Transfer via Flow Matching (Shitong Shao et al., 2024)

{{<citation>}}

Shitong Shao, Zhiqiang Shen, Linrui Gong, Huanran Chen, Xu Dai. (2024)  
**Precise Knowledge Transfer via Flow Matching**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2402.02012v1)  

---


**ABSTRACT**  
In this paper, we propose a novel knowledge transfer framework that introduces continuous normalizing flows for progressive knowledge transformation and leverages multi-step sampling strategies to achieve precision knowledge transfer. We name this framework Knowledge Transfer with Flow Matching (FM-KT), which can be integrated with a metric-based distillation method with any form (\textit{e.g.} vanilla KD, DKD, PKD and DIST) and a meta-encoder with any available architecture (\textit{e.g.} CNN, MLP and Transformer). By introducing stochastic interpolants, FM-KD is readily amenable to arbitrary noise schedules (\textit{e.g.}, VP-ODE, VE-ODE, Rectified flow) for normalized flow path estimation. We theoretically demonstrate that the training objective of FM-KT is equivalent to minimizing the upper bound of the teacher feature map or logit negative log-likelihood. Besides, FM-KT can be viewed as a unique implicit ensemble method that leads to performance gains. By slightly modifying the FM-KT framework, FM-KT can also be transformed into an online distillation framework OFM-KT with desirable performance gains. Through extensive experiments on CIFAR-100, ImageNet-1k, and MS-COCO datasets, we empirically validate the scalability and state-of-the-art performance of our proposed methods among relevant comparison approaches.

{{</citation>}}


### (75/91) Hypergraph-Transformer (HGT) for Interactive Event Prediction in Laparoscopic and Robotic Surgery (Lianhao Yin et al., 2024)

{{<citation>}}

Lianhao Yin, Yutong Ban, Jennifer Eckhoff, Ozanan Meireles, Daniela Rus, Guy Rosman. (2024)  
**Hypergraph-Transformer (HGT) for Interactive Event Prediction in Laparoscopic and Robotic Surgery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2402.01974v1)  

---


**ABSTRACT**  
Understanding and anticipating intraoperative events and actions is critical for intraoperative assistance and decision-making during minimally invasive surgery. Automated prediction of events, actions, and the following consequences is addressed through various computational approaches with the objective of augmenting surgeons' perception and decision-making capabilities. We propose a predictive neural network that is capable of understanding and predicting critical interactive aspects of surgical workflow from intra-abdominal video, while flexibly leveraging surgical knowledge graphs. The approach incorporates a hypergraph-transformer (HGT) structure that encodes expert knowledge into the network design and predicts the hidden embedding of the graph. We verify our approach on established surgical datasets and applications, including the detection and prediction of action triplets, and the achievement of the Critical View of Safety (CVS). Moreover, we address specific, safety-related tasks, such as predicting the clipping of cystic duct or artery without prior achievement of the CVS. Our results demonstrate the superiority of our approach compared to unstructured alternatives.

{{</citation>}}


## eess.IV (1)



### (76/91) InceptionCapsule: Inception-Resnet and CapsuleNet with self-attention for medical image Classification (Elham Sadeghnezhad et al., 2024)

{{<citation>}}

Elham Sadeghnezhad, Sajjad Salem. (2024)  
**InceptionCapsule: Inception-Resnet and CapsuleNet with self-attention for medical image Classification**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2402.02274v1)  

---


**ABSTRACT**  
Initial weighting is significant in deep neural networks because the random selection of weights produces different outputs and increases the probability of overfitting and underfitting. On the other hand, vector-based approaches to extract vector features need rich vectors for more accurate classification. The InceptionCapsule approach is presented to alleviate these two problems. This approach uses transfer learning and the Inception-ResNet model to avoid random selection of weights, which takes initial weights from ImageNet. It also uses the output of Inception middle layers to generate rich vectors. Extracted vectors are given to a capsule network for learning, which is equipped with an attention technique. Kvasir data and BUSI with the GT dataset were used to evaluate this approach. This model was able to achieve 97.62 accuracies in 5-class classification and also achieved 94.30 accuracies in 8-class classification on Kvasir. In the BUSI with GT dataset, the proposed approach achieved accuracy=98.88, Precision=95.34, and F1-score=93.74, which are acceptable results compared to other approaches in the literature.

{{</citation>}}


## stat.ME (1)



### (77/91) Sample-Efficient Clustering and Conquer Procedures for Parallel Large-Scale Ranking and Selection (Zishi Zhang et al., 2024)

{{<citation>}}

Zishi Zhang, Yijie Peng. (2024)  
**Sample-Efficient Clustering and Conquer Procedures for Parallel Large-Scale Ranking and Selection**  

---
Primary Category: stat.ME  
Categories: cs-LG, stat-ME, stat.ME  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.02196v1)  

---


**ABSTRACT**  
We propose novel "clustering and conquer" procedures for the parallel large-scale ranking and selection (R&S) problem, which leverage correlation information for clustering to break the bottleneck of sample efficiency. In parallel computing environments, correlation-based clustering can achieve an $\mathcal{O}(p)$ sample complexity reduction rate, which is the optimal reduction rate theoretically attainable. Our proposed framework is versatile, allowing for seamless integration of various prevalent R&S methods under both fixed-budget and fixed-precision paradigms. It can achieve improvements without the necessity of highly accurate correlation estimation and precise clustering. In large-scale AI applications such as neural architecture search, a screening-free version of our procedure surprisingly surpasses fully-sequential benchmarks in terms of sample efficiency. This suggests that leveraging valuable structural information, such as correlation, is a viable path to bypassing the traditional need for screening via pairwise comparison--a step previously deemed essential for high sample efficiency but problematic for parallelization. Additionally, we propose a parallel few-shot clustering algorithm tailored for large-scale problems.

{{</citation>}}


## cs.RO (1)



### (78/91) RecNet: An Invertible Point Cloud Encoding through Range Image Embeddings for Multi-Robot Map Sharing and Reconstruction (Nikolaos Stathoulopoulos et al., 2024)

{{<citation>}}

Nikolaos Stathoulopoulos, Mario A. V. Saucedo, Anton Koval, George Nikolakopoulos. (2024)  
**RecNet: An Invertible Point Cloud Encoding through Range Image Embeddings for Multi-Robot Map Sharing and Reconstruction**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2402.02192v1)  

---


**ABSTRACT**  
In the field of resource-constrained robots and the need for effective place recognition in multi-robotic systems, this article introduces RecNet, a novel approach that concurrently addresses both challenges. The core of RecNet's methodology involves a transformative process: it projects 3D point clouds into depth images, compresses them using an encoder-decoder framework, and subsequently reconstructs the range image, seamlessly restoring the original point cloud. Additionally, RecNet utilizes the latent vector extracted from this process for efficient place recognition tasks. This unique approach not only achieves comparable place recognition results but also maintains a compact representation, suitable for seamless sharing among robots to reconstruct their collective maps. The evaluation of RecNet encompasses an array of metrics, including place recognition performance, structural similarity of the reconstructed point clouds, and the bandwidth transmission advantages, derived from sharing only the latent vectors. This reconstructed map paves a groundbreaking way for exploring its usability in navigation, localization, map-merging, and other relevant missions. Our proposed approach is rigorously assessed using both a publicly available dataset and field experiments, confirming its efficacy and potential for real-world applications.

{{</citation>}}


## cs.SI (2)



### (79/91) An Ontology-Based multi-domain model in Social Network Analysis: Experimental validation and case study (José Alberto Benítez-Andrades et al., 2024)

{{<citation>}}

José Alberto Benítez-Andrades, Isaías García-Rodríguez, Carmen Benavides, Héctor Aláiz-Moretón, José Emilio Labra Gayo. (2024)  
**An Ontology-Based multi-domain model in Social Network Analysis: Experimental validation and case study**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: QA, Social Network  
[Paper Link](http://arxiv.org/abs/2402.02181v1)  

---


**ABSTRACT**  
The use of social network theory and methods of analysis have been applied to different domains in recent years, including public health. The complete procedure for carrying out a social network analysis (SNA) is a time-consuming task that entails a series of steps in which the expert in social network analysis could make mistakes. This research presents a multi-domain knowledge model capable of automatically gathering data and carrying out different social network analyses in different domains, without errors and obtaining the same conclusions that an expert in SNA would obtain. The model is represented in an ontology called OntoSNAQA, which is made up of classes, properties and rules representing the domains of People, Questionnaires and Social Network Analysis. Besides the ontology itself, different rules are represented by SWRL and SPARQL queries. A Knowledge Based System was created using OntoSNAQA and applied to a real case study in order to show the advantages of the approach. Finally, the results of an SNA analysis obtained through the model were compared to those obtained from some of the most widely used SNA applications: UCINET, Pajek, Cytoscape and Gephi, to test and confirm the validity of the model.

{{</citation>}}


### (80/91) Trustworthiness of $\mathbb{X}$ Users: A One-Class Classification Approach (Tanveer Khan et al., 2024)

{{<citation>}}

Tanveer Khan, Fahad Sohrab, Antonis Michalas, Moncef Gabbouj. (2024)  
**Trustworthiness of $\mathbb{X}$ Users: A One-Class Classification Approach**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2402.02066v1)  

---


**ABSTRACT**  
$\mathbb{X}$ (formerly Twitter) is a prominent online social media platform that plays an important role in sharing information making the content generated on this platform a valuable source of information. Ensuring trust on $\mathbb{X}$ is essential to determine the user credibility and prevents issues across various domains. While assigning credibility to $\mathbb{X}$ users and classifying them as trusted or untrusted is commonly carried out using traditional machine learning models, there is limited exploration about the use of One-Class Classification (OCC) models for this purpose. In this study, we use various OCC models for $\mathbb{X}$ user classification. Additionally, we propose using a subspace-learning-based approach that simultaneously optimizes both the subspace and data description for OCC. We also introduce a novel regularization term for Subspace Support Vector Data Description (SSVDD), expressing data concentration in a lower-dimensional subspace that captures diverse graph structures. Experimental results show superior performance of the introduced regularization term for SSVDD compared to baseline models and state-of-the-art techniques for $\mathbb{X}$ user classification.

{{</citation>}}


## cs.SE (3)



### (81/91) Collaborative Agents for Software Engineering (Daniel Tang et al., 2024)

{{<citation>}}

Daniel Tang, Zhenghan Chen, Kisub Kim, Yewei Song, Haoye Tian, Saad Ezzini, Yongfeng Huang, Jacques Klein Tegawende F. Bissyande. (2024)  
**Collaborative Agents for Software Engineering**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2402.02172v1)  

---


**ABSTRACT**  
Code review is a heavily collaborative process, which aims at ensuring the overall quality and reliability of software. While it provides massive benefits, the implementation of code review in an organization faces several challenges that make its automation appealing. Automated code review tools have been around for a while and are now improving thanks to the adoption of novel AI models, which help can learn about standard practices and systematically check that the reviewed code adheres to them. Unfortunately, existing methods fall short: they often target a single input-output generative model, which cannot simulate the collaboration interactions in code review to account for various perspectives; they are also sub-performing on various critical code review sub-tasks. In this paper, we advance the state of the art in code review automation by introducing CodeAgent, a novel multi-agent-based system for code review. Fundamentally, CodeAgent is steered by QA-Checker (short for ``Question-Answer Checking"), a supervision agent, designed specifically to ensure that all agents' contributions remain relevant to the initial review question. CodeAgent is autonomous, multi-agent, and Large language model-driven. To demonstrate the effectiveness of CodeAgent, we performed experiments to assess its capabilities in various tasks including 1) detection of inconsistencies between code changes and commit messages, 2) detection of vulnerability introduction by commits, and 3) validation of adherence to code style. Our website is accessed in \url{https://code-agent-new.vercel.app/index.html}.

{{</citation>}}


### (82/91) Improving the Learning of Code Review Successive Tasks with Cross-Task Knowledge Distillation (Oussama Ben Sghaier et al., 2024)

{{<citation>}}

Oussama Ben Sghaier, Houari Sahraoui. (2024)  
**Improving the Learning of Code Review Successive Tasks with Cross-Task Knowledge Distillation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2402.02063v1)  

---


**ABSTRACT**  
Code review is a fundamental process in software development that plays a pivotal role in ensuring code quality and reducing the likelihood of errors and bugs. However, code review can be complex, subjective, and time-consuming. Quality estimation, comment generation, and code refinement constitute the three key tasks of this process, and their automation has traditionally been addressed separately in the literature using different approaches. In particular, recent efforts have focused on fine-tuning pre-trained language models to aid in code review tasks, with each task being considered in isolation. We believe that these tasks are interconnected, and their fine-tuning should consider this interconnection. In this paper, we introduce a novel deep-learning architecture, named DISCOREV, which employs cross-task knowledge distillation to address these tasks simultaneously. In our approach, we utilize a cascade of models to enhance both comment generation and code refinement models. The fine-tuning of the comment generation model is guided by the code refinement model, while the fine-tuning of the code refinement model is guided by the quality estimation model. We implement this guidance using two strategies: a feedback-based learning objective and an embedding alignment objective. We evaluate DISCOREV by comparing it to state-of-the-art methods based on independent training and fine-tuning. Our results show that our approach generates better review comments, as measured by the BLEU score, as well as more accurate code refinement according to the CodeBLEU score

{{</citation>}}


### (83/91) EffiBench: Benchmarking the Efficiency of Automatically Generated Code (Dong Huang et al., 2024)

{{<citation>}}

Dong Huang, Jie M. Zhang, Yuhao Qing, Heming Cui. (2024)  
**EffiBench: Benchmarking the Efficiency of Automatically Generated Code**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2402.02037v1)  

---


**ABSTRACT**  
Code generation models have increasingly become integral to aiding software development, offering assistance in tasks such as code completion, debugging, and code translation. Although current research has thoroughly examined the correctness of code produced by code generation models, a vital aspect, i.e., the efficiency of the generated code, has often been neglected. This paper presents EffiBench, a benchmark with 1,000 efficiency-critical coding problems for assessing the efficiency of code generated by code generation models. EffiBench contains a diverse set of LeetCode coding problems. Each problem is paired with an executable human-written canonical solution. With EffiBench, we empirically examine the capability of 21 Large Language Models (13 open-sourced and 8 closed-sourced) in generating efficient code. The results demonstrate that GPT-4-turbo generates the most efficient code, significantly outperforming Palm-2-chat-bison, Claude-instant-1, Gemini-pro, GPT-4, and GPT-3.5. Nevertheless, its code efficiency is still worse than the efficiency of human-written canonical solutions. In particular, the average and worst execution time of GPT-4-turbo generated code is 1.69 and 45.49 times that of the canonical solutions.

{{</citation>}}


## cs.HC (3)



### (84/91) Vi(E)va LLM! A Conceptual Stack for Evaluating and Interpreting Generative AI-based Visualizations (Luca Podo et al., 2024)

{{<citation>}}

Luca Podo, Muhammad Ishmal, Marco Angelini. (2024)  
**Vi(E)va LLM! A Conceptual Stack for Evaluating and Interpreting Generative AI-based Visualizations**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-LG, cs.HC  
Keywords: AI, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2402.02167v1)  

---


**ABSTRACT**  
The automatic generation of visualizations is an old task that, through the years, has shown more and more interest from the research and practitioner communities. Recently, large language models (LLM) have become an interesting option for supporting generative tasks related to visualization, demonstrating initial promising results. At the same time, several pitfalls, like the multiple ways of instructing an LLM to generate the desired result, the different perspectives leading the generation (code-based, image-based, grammar-based), and the presence of hallucinations even for the visualization generation task, make their usage less affordable than expected. Following similar initiatives for benchmarking LLMs, this paper copes with the problem of modeling the evaluation of a generated visualization through an LLM. We propose a theoretical evaluation stack, EvaLLM, that decomposes the evaluation effort in its atomic components, characterizes their nature, and provides an overview of how to implement and interpret them. We also designed and implemented an evaluation platform that provides a benchmarking resource for the visualization generation task. The platform supports automatic and manual scoring conducted by multiple assessors to support a fine-grained and semantic evaluation based on the EvaLLM stack. Two case studies on GPT3.5-turbo with Code Interpreter and Llama2-70-b models show the benefits of EvaLLM and illustrate interesting results on the current state-of-the-art LLM-generated visualizations.

{{</citation>}}


### (85/91) User Intent Recognition and Satisfaction with Large Language Models: A User Study with ChatGPT (Anna Bodonhelyi et al., 2024)

{{<citation>}}

Anna Bodonhelyi, Efe Bozkir, Shuo Yang, Enkelejda Kasneci, Gjergji Kasneci. (2024)  
**User Intent Recognition and Satisfaction with Large Language Models: A User Study with ChatGPT**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Intent Recognition, Language Model  
[Paper Link](http://arxiv.org/abs/2402.02136v1)  

---


**ABSTRACT**  
The rapid evolution of large language models such as GPT-4 Turbo represents an impactful paradigm shift in digital interaction and content engagement. While these models encode vast amounts of human-generated knowledge and excel in processing diverse data types, recent research shows that they often face the challenge of accurately responding to specific user intents, leading to increased user dissatisfaction. Based on a fine-grained intent taxonomy and intent-based prompt reformulations, we analyze (1) the quality of intent recognition and (2) user satisfaction with answers from intent-based prompt reformulations for two recent ChatGPT models, GPT-3.5 Turbo and GPT-4 Turbo. The results reveal that GPT-4 outperforms GPT-3.5 on the recognition of common intents, but is conversely often outperformed by GPT-3.5 on the recognition of less frequent intents. Moreover, whenever the user intent is correctly recognized, while users are more satisfied with the answers to intent-based reformulations of GPT 4 compared to GPT-3.5, they tend to be more satisfied with the answers of the models to their original prompts compared to the reformulated ones. Finally, the study indicates that users can quickly learn to formulate their prompts more effectively, once they are shown possible reformulation templates.

{{</citation>}}


### (86/91) Human-Centered Privacy Research in the Age of Large Language Models (Tianshi Li et al., 2024)

{{<citation>}}

Tianshi Li, Sauvik Das, Hao-Ping Lee, Dakuo Wang, Bingsheng Yao, Zhiping Zhang. (2024)  
**Human-Centered Privacy Research in the Age of Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CR, cs-HC, cs.HC  
Keywords: AI, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2402.01994v1)  

---


**ABSTRACT**  
The emergence of large language models (LLMs), and their increased use in user-facing systems, has led to substantial privacy concerns. To date, research on these privacy concerns has been model-centered: exploring how LLMs lead to privacy risks like memorization, or can be used to infer personal characteristics about people from their content. We argue that there is a need for more research focusing on the human aspect of these privacy issues: e.g., research on how design paradigms for LLMs affect users' disclosure behaviors, users' mental models and preferences for privacy controls, and the design of tools, systems, and artifacts that empower end-users to reclaim ownership over their personal data. To build usable, efficient, and privacy-friendly systems powered by these models with imperfect privacy properties, our goal is to initiate discussions to outline an agenda for conducting human-centered research on privacy issues in LLM-powered systems. This Special Interest Group (SIG) aims to bring together researchers with backgrounds in usable security and privacy, human-AI collaboration, NLP, or any other related domains to share their perspectives and experiences on this problem, to help our community establish a collective understanding of the challenges, research opportunities, research methods, and strategies to collaborate with researchers outside of HCI.

{{</citation>}}


## cs.AI (2)



### (87/91) Emergency Computing: An Adaptive Collaborative Inference Method Based on Hierarchical Reinforcement Learning (Weiqi Fu et al., 2024)

{{<citation>}}

Weiqi Fu, Lianming Xu, Xin Wu, Li Wang, Aiguo Fei. (2024)  
**Emergency Computing: An Adaptive Collaborative Inference Method Based on Hierarchical Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-NI, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2402.02146v1)  

---


**ABSTRACT**  
In achieving effective emergency response, the timely acquisition of environmental information, seamless command data transmission, and prompt decision-making are crucial. This necessitates the establishment of a resilient emergency communication dedicated network, capable of providing communication and sensing services even in the absence of basic infrastructure. In this paper, we propose an Emergency Network with Sensing, Communication, Computation, Caching, and Intelligence (E-SC3I). The framework incorporates mechanisms for emergency computing, caching, integrated communication and sensing, and intelligence empowerment. E-SC3I ensures rapid access to a large user base, reliable data transmission over unstable links, and dynamic network deployment in a changing environment. However, these advantages come at the cost of significant computation overhead. Therefore, we specifically concentrate on emergency computing and propose an adaptive collaborative inference method (ACIM) based on hierarchical reinforcement learning. Experimental results demonstrate our method's ability to achieve rapid inference of AI models with constrained computational and communication resources.

{{</citation>}}


### (88/91) BetterV: Controlled Verilog Generation with Discriminative Guidance (Zehua Pei et al., 2024)

{{<citation>}}

Zehua Pei, Hui-Ling Zhen, Mingxuan Yuan, Yu Huang, Bei Yu. (2024)  
**BetterV: Controlled Verilog Generation with Discriminative Guidance**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-PL, cs.AI  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2402.03375v1)  

---


**ABSTRACT**  
Due to the growing complexity of modern Integrated Circuits (ICs), there is a need for automated circuit design methods. Recent years have seen rising research in hardware design language generation to facilitate the design process. In this work, we propose a Verilog generation framework, BetterV, which fine-tunes the large language models (LLMs) on processed domain-specific datasets and incorporates generative discriminators for guidance on particular design demands. The Verilog modules are collected, filtered and processed from internet to form a clean and abundant dataset. Instruct-tuning methods are specially designed to fine-tuned the LLMs to understand the knowledge about Verilog. Furthermore, data are augmented to enrich the training set and also used to train a generative discriminator on particular downstream task, which leads a guidance for the LLMs to optimize the Verilog implementation. BetterV has the ability to generate syntactically and functionally correct Verilog, which can outperform GPT-4 on the VerilogEval-machine benchmark. With the help of task-specific generative discriminator, BetterV can achieve remarkable improvement on various electronic design automation (EDA) downstream tasks, including the netlist node reduction for synthesis and verification runtime reduction with Boolean Satisfiability (SAT) solving.

{{</citation>}}


## cs.IR (1)



### (89/91) Prototypical Contrastive Learning through Alignment and Uniformity for Recommendation (Yangxun Ou et al., 2024)

{{<citation>}}

Yangxun Ou, Lei Chen, Fenglin Pan, Yupeng Wu. (2024)  
**Prototypical Contrastive Learning through Alignment and Uniformity for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2402.02079v1)  

---


**ABSTRACT**  
Graph Collaborative Filtering (GCF), one of the most widely adopted recommendation system methods, effectively captures intricate relationships between user and item interactions. Graph Contrastive Learning (GCL) based GCF has gained significant attention as it leverages self-supervised techniques to extract valuable signals from real-world scenarios. However, many methods usually learn the instances of discrimination tasks that involve the construction of contrastive pairs through random sampling. GCL approaches suffer from sampling bias issues, where the negatives might have a semantic structure similar to that of the positives, thus leading to a loss of effective feature representation. To address these problems, we present the \underline{Proto}typical contrastive learning through \underline{A}lignment and \underline{U}niformity for recommendation, which is called \textbf{ProtoAU}. Specifically, we first propose prototypes (cluster centroids) as a latent space to ensure consistency across different augmentations from the origin graph, aiming to eliminate the need for random sampling of contrastive pairs. Furthermore, the absence of explicit negatives means that directly optimizing the consistency loss between instance and prototype could easily result in dimensional collapse issues. Therefore, we propose aligning and maintaining uniformity in the prototypes of users and items as optimization objectives to prevent falling into trivial solutions. Finally, we conduct extensive experiments on four datasets and evaluate their performance on the task of link prediction. Experimental results demonstrate that the proposed ProtoAU outperforms other representative methods. The source codes of our proposed ProtoAU are available at \url{https://github.com/oceanlvr/ProtoAU}.

{{</citation>}}


## cs.ET (1)



### (90/91) Low-power scalable multilayer optoelectronic neural networks enabled with incoherent light (Alexander Song et al., 2024)

{{<citation>}}

Alexander Song, Sai Nikhilesh Murty Kottapalli, Rahul Goyal, Bernhard Schölkopf, Peer Fischer. (2024)  
**Low-power scalable multilayer optoelectronic neural networks enabled with incoherent light**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs.ET, physics-optics  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2402.01988v1)  

---


**ABSTRACT**  
Optical approaches have made great strides towards the goal of high-speed, energy-efficient computing necessary for modern deep learning and AI applications. Read-in and read-out of data, however, limit the overall performance of existing approaches. This study introduces a multilayer optoelectronic computing framework that alternates between optical and optoelectronic layers to implement matrix-vector multiplications and rectified linear functions, respectively. Our framework is designed for real-time, parallelized operations, leveraging 2D arrays of LEDs and photodetectors connected via independent analog electronics. We experimentally demonstrate this approach using a system with a three-layer network with two hidden layers and operate it to recognize images from the MNIST database with a recognition accuracy of 92% and classify classes from a nonlinear spiral data with 86% accuracy. By implementing multiple layers of a deep neural network simultaneously, our approach significantly reduces the number of read-ins and read-outs required and paves the way for scalable optical accelerators requiring ultra low energy.

{{</citation>}}


## cs.MA (1)



### (91/91) A Survey on Context-Aware Multi-Agent Systems: Techniques, Challenges and Future Directions (Hung Du et al., 2024)

{{<citation>}}

Hung Du, Srikanth Thudumu, Rajesh Vasa, Kon Mouzakis. (2024)  
**A Survey on Context-Aware Multi-Agent Systems: Techniques, Challenges and Future Directions**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-LG, cs-MA, cs.MA  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2402.01968v1)  

---


**ABSTRACT**  
Research interest in autonomous agents is on the rise as an emerging topic. The notable achievements of Large Language Models (LLMs) have demonstrated the considerable potential to attain human-like intelligence in autonomous agents. However, the challenge lies in enabling these agents to learn, reason, and navigate uncertainties in dynamic environments. Context awareness emerges as a pivotal element in fortifying multi-agent systems when dealing with dynamic situations. Despite existing research focusing on both context-aware systems and multi-agent systems, there is a lack of comprehensive surveys outlining techniques for integrating context-aware systems with multi-agent systems. To address this gap, this survey provides a comprehensive overview of state-of-the-art context-aware multi-agent systems. First, we outline the properties of both context-aware systems and multi-agent systems that facilitate integration between these systems. Subsequently, we propose a general process for context-aware systems, with each phase of the process encompassing diverse approaches drawn from various application domains such as collision avoidance in autonomous driving, disaster relief management, utility management, supply chain management, human-AI interaction, and others. Finally, we discuss the existing challenges of context-aware multi-agent systems and provide future research directions in this field.

{{</citation>}}
