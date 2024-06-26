---
draft: false
title: "arXiv @ 2023.10.19"
date: 2023-10-19
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.19"
    identifier: arxiv_20231019
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (58)](#cscl-58)
- [cs.RO (4)](#csro-4)
- [cs.LG (28)](#cslg-28)
- [cs.CV (15)](#cscv-15)
- [cs.CR (6)](#cscr-6)
- [cs.AR (1)](#csar-1)
- [cs.HC (4)](#cshc-4)
- [cs.DB (1)](#csdb-1)
- [cs.SE (5)](#csse-5)
- [cs.SI (3)](#cssi-3)
- [cs.CY (3)](#cscy-3)
- [cs.IR (3)](#csir-3)
- [quant-ph (1)](#quant-ph-1)
- [cs.LO (1)](#cslo-1)
- [eess.IV (4)](#eessiv-4)
- [cs.AI (4)](#csai-4)
- [eess.AS (1)](#eessas-1)
- [eess.SY (1)](#eesssy-1)
- [stat.ME (1)](#statme-1)

## cs.CL (58)



### (1/144) Learn Your Tokens: Word-Pooled Tokenization for Language Modeling (Avijit Thawani et al., 2023)

{{<citation>}}

Avijit Thawani, Saurabh Ghanekar, Xiaoyuan Zhu, Jay Pujara. (2023)  
**Learn Your Tokens: Word-Pooled Tokenization for Language Modeling**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11628v1)  

---


**ABSTRACT**  
Language models typically tokenize text into subwords, using a deterministic, hand-engineered heuristic of combining characters into longer surface-level strings such as 'ing' or whole words. Recent literature has repeatedly shown the limitations of such a tokenization strategy, particularly for documents not written in English and for representing numbers. On the other extreme, byte/character-level language models are much less restricted but suffer from increased sequence description lengths and a subsequent quadratic expansion in self-attention computation. Recent attempts to compress and limit these context lengths with fixed size convolutions is helpful but completely ignores the word boundary. This paper considers an alternative 'learn your tokens' scheme which utilizes the word boundary to pool bytes/characters into word representations, which are fed to the primary language model, before again decoding individual characters/bytes per word in parallel. We find that our moderately expressive and moderately fast end-to-end tokenizer outperform by over 300% both subwords and byte/character models over the intrinsic language modeling metric of next-word prediction across datasets. It particularly outshines on rare words, outperforming by a factor of 30! We extensively study the language modeling setup for all three categories of tokenizers and theoretically analyze how our end-to-end models can also be a strong trade-off in efficiency and robustness.

{{</citation>}}


### (2/144) Unveiling the General Intelligence Factor in Language Models: A Psychometric Approach (David Ilić, 2023)

{{<citation>}}

David Ilić. (2023)  
**Unveiling the General Intelligence Factor in Language Models: A Psychometric Approach**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GLUE, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11616v1)  

---


**ABSTRACT**  
This study uncovers the factor of general intelligence, or g, in language models, extending the psychometric theory traditionally applied to humans and certain animal species. Utilizing factor analysis on two extensive datasets - Open LLM Leaderboard with 1,232 models and General Language Understanding Evaluation (GLUE) Leaderboard with 88 models - we find compelling evidence for a unidimensional, highly stable g factor that accounts for 85% of the variance in model performance. The study also finds a moderate correlation of .48 between model size and g. The discovery of g in language models offers a unified metric for model evaluation and opens new avenues for more robust, g-based model ability assessment. These findings lay the foundation for understanding and future research on artificial general intelligence from a psychometric perspective and have practical implications for model evaluation and development.

{{</citation>}}


### (3/144) Automated Evaluation of Personalized Text Generation using Large Language Models (Yaqing Wang et al., 2023)

{{<citation>}}

Yaqing Wang, Jiepu Jiang, Mingyang Zhang, Cheng Li, Yi Liang, Qiaozhu Mei, Michael Bendersky. (2023)  
**Automated Evaluation of Personalized Text Generation using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BLEU, Language Model, Text Generation  
[Paper Link](http://arxiv.org/abs/2310.11593v1)  

---


**ABSTRACT**  
Personalized text generation presents a specialized mechanism for delivering content that is specific to a user's personal context. While the research progress in this area has been rapid, evaluation still presents a challenge. Traditional automated metrics such as BLEU and ROUGE primarily measure lexical similarity to human-written references, and are not able to distinguish personalization from other subtle semantic aspects, thus falling short of capturing the nuances of personalized generated content quality. On the other hand, human judgments are costly to obtain, especially in the realm of personalized evaluation. Inspired by these challenges, we explore the use of large language models (LLMs) for evaluating personalized text generation, and examine their ability to understand nuanced user context. We present AuPEL, a novel evaluation method that distills three major semantic aspects of the generated text: personalization, quality and relevance, and automatically measures these aspects. To validate the effectiveness of AuPEL, we design carefully controlled experiments and compare the accuracy of the evaluation judgments made by LLMs versus that of judgements made by human annotators, and conduct rigorous analyses of the consistency and sensitivity of the proposed metric. We find that, compared to existing evaluation metrics, AuPEL not only distinguishes and ranks models based on their personalization abilities more accurately, but also presents commendable consistency and efficiency for this task. Our work suggests that using LLMs as the evaluators of personalized text generation is superior to traditional text similarity metrics, even though interesting new challenges still remain.

{{</citation>}}


### (4/144) Audio-AdapterFusion: A Task-ID-free Approach for Efficient and Non-Destructive Multi-task Speech Recognition (Hillary Ngai et al., 2023)

{{<citation>}}

Hillary Ngai, Rohan Agrawal, Neeraj Gaur, Ronny Huang, Parisa Haghani, Pedro Moreno Mengibar. (2023)  
**Audio-AdapterFusion: A Task-ID-free Approach for Efficient and Non-Destructive Multi-task Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.13015v1)  

---


**ABSTRACT**  
Adapters are an efficient, composable alternative to full fine-tuning of pre-trained models and help scale the deployment of large ASR models to many tasks. In practice, a task ID is commonly prepended to the input during inference to route to single-task adapters for the specified task. However, one major limitation of this approach is that the task ID may not be known during inference, rendering it unsuitable for most multi-task settings. To address this, we propose three novel task-ID-free methods to combine single-task adapters in multi-task ASR and investigate two learning algorithms for training. We evaluate our methods on 10 test sets from 4 diverse ASR tasks and show that our methods are non-destructive and parameter-efficient. While only updating 17% of the model parameters, our methods can achieve an 8% mean WER improvement relative to full fine-tuning and are on-par with task-ID adapter routing.

{{</citation>}}


### (5/144) Eliciting Human Preferences with Language Models (Belinda Z. Li et al., 2023)

{{<citation>}}

Belinda Z. Li, Alex Tamkin, Noah Goodman, Jacob Andreas. (2023)  
**Eliciting Human Preferences with Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11589v1)  

---


**ABSTRACT**  
Language models (LMs) can be directed to perform target tasks by using labeled examples or natural language prompts. But selecting examples or writing prompts for can be challenging--especially in tasks that involve unusual edge cases, demand precise articulation of nebulous preferences, or require an accurate mental model of LM behavior. We propose to use *LMs themselves* to guide the task specification process. In this paper, we introduce **Generative Active Task Elicitation (GATE)**: a learning framework in which models elicit and infer intended behavior through free-form, language-based interaction with users. We study GATE in three domains: email validation, content recommendation, and moral reasoning. In preregistered experiments, we show that LMs prompted to perform GATE (e.g., by generating open-ended questions or synthesizing informative edge cases) elicit responses that are often more informative than user-written prompts or labels. Users report that interactive task elicitation requires less effort than prompting or example labeling and surfaces novel considerations not initially anticipated by users. Our findings suggest that LM-driven elicitation can be a powerful tool for aligning models to complex human preferences and values.

{{</citation>}}


### (6/144) What is a good question? Task-oriented asking with fact-level masking (Matthew Toles et al., 2023)

{{<citation>}}

Matthew Toles, Yukun Huang, Zhou Yu, Luis Gravano. (2023)  
**What is a good question? Task-oriented asking with fact-level masking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.11571v1)  

---


**ABSTRACT**  
Asking questions is an important element of real-life collaboration on reasoning tasks like question answering. For example, a legal assistant chatbot may be unable to make accurate recommendations without specific information on the user's circumstances. However, large language models are usually deployed to solve reasoning tasks directly without asking follow-up questions to the user or third parties. We term this problem task-oriented asking (TOA). Zero-shot chat models can perform TOA, but their training is primarily based on next-token prediction rather than whether questions contribute to successful collaboration. To enable the training and evaluation of TOA models, we present a definition and framework for natural language task-oriented asking, the problem of generating questions that result in answers useful for a reasoning task. We also present fact-level masking (FLM), a procedure for converting natural language datasets into self-supervised TOA datasets by omitting particular critical facts. Finally, we generate a TOA dataset from the HotpotQA dataset using FLM and evaluate several zero-shot language models on it. Our experiments show that current zero-shot models struggle to ask questions that retrieve useful information, as compared to human annotators. These results demonstrate an opportunity to use FLM datasets and the TOA framework to train and evaluate better TOA models.

{{</citation>}}


### (7/144) Personalized Soups: Personalized Large Language Model Alignment via Post-hoc Parameter Merging (Joel Jang et al., 2023)

{{<citation>}}

Joel Jang, Seungone Kim, Bill Yuchen Lin, Yizhong Wang, Jack Hessel, Luke Zettlemoyer, Hannaneh Hajishirzi, Yejin Choi, Prithviraj Ammanabrolu. (2023)  
**Personalized Soups: Personalized Large Language Model Alignment via Post-hoc Parameter Merging**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.11564v1)  

---


**ABSTRACT**  
While Reinforcement Learning from Human Feedback (RLHF) aligns Large Language Models (LLMs) with general, aggregate human preferences, it is suboptimal for learning diverse, individual perspectives. In this work, we study Reinforcement Learning from Personalized Human Feedback (RLPHF) problem, wherein LLMs are aligned to multiple (sometimes conflicting) preferences by modeling alignment as a Multi-Objective Reinforcement Learning (MORL) problem. Compared to strong single-objective baselines, we show that we can achieve personalized alignment by decomposing preferences into multiple dimensions. These dimensions are defined based on personalizations that are declared as desirable by the user. In this work, we show that they can be efficiently trained independently in a distributed manner and combined effectively post-hoc through parameter merging. The code is available at https://github.com/joeljang/RLPHF.

{{</citation>}}


### (8/144) MUST&P-SRL: Multi-lingual and Unified Syllabification in Text and Phonetic Domains for Speech Representation Learning (Noé Tits, 2023)

{{<citation>}}

Noé Tits. (2023)  
**MUST&P-SRL: Multi-lingual and Unified Syllabification in Text and Phonetic Domains for Speech Representation Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, eess-AS  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.11541v1)  

---


**ABSTRACT**  
In this paper, we present a methodology for linguistic feature extraction, focusing particularly on automatically syllabifying words in multiple languages, with a design to be compatible with a forced-alignment tool, the Montreal Forced Aligner (MFA). In both the textual and phonetic domains, our method focuses on the extraction of phonetic transcriptions from text, stress marks, and a unified automatic syllabification (in text and phonetic domains). The system was built with open-source components and resources. Through an ablation study, we demonstrate the efficacy of our approach in automatically syllabifying words from several languages (English, French and Spanish). Additionally, we apply the technique to the transcriptions of the CMU ARCTIC dataset, generating valuable annotations available online\footnote{\url{https://github.com/noetits/MUST_P-SRL}} that are ideal for speech representation learning, speech unit discovery, and disentanglement of speech factors in several speech-related fields.

{{</citation>}}


### (9/144) Multi-stage Large Language Model Correction for Speech Recognition (Jie Pu et al., 2023)

{{<citation>}}

Jie Pu, Thai-Son Nguyen, Sebastian Stüker. (2023)  
**Multi-stage Large Language Model Correction for Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: Language Model, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.11532v1)  

---


**ABSTRACT**  
In this paper, we investigate the usage of large language models (LLMs) to improve the performance of competitive speech recognition systems. Different from traditional language models that focus on one single data domain, the rise of LLMs brings us the opportunity to push the limit of state-of-the-art ASR performance, and at the same time to achieve higher robustness and generalize effectively across multiple domains. Motivated by this, we propose a novel multi-stage approach to combine traditional language model re-scoring and LLM prompting. Specifically, the proposed method has two stages: the first stage uses a language model to re-score an N-best list of ASR hypotheses and run a confidence check; The second stage uses prompts to a LLM to perform ASR error correction on less confident results from the first stage. Our experimental results demonstrate the effectiveness of the proposed method by showing a 10% ~ 20% relative improvement in WER over a competitive ASR system -- across multiple test domains.

{{</citation>}}


### (10/144) Automatic News Summerization (Kavach Dheer et al., 2023)

{{<citation>}}

Kavach Dheer, Arpit Dhankhar. (2023)  
**Automatic News Summerization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Natural Language Processing, Summarization, Text Summarization  
[Paper Link](http://arxiv.org/abs/2310.11520v1)  

---


**ABSTRACT**  
Natural Language Processing is booming with its applications in the real world, one of which is Text Summarization for large texts including news articles. This research paper provides an extensive comparative evaluation of extractive and abstractive approaches for news text summarization, with an emphasis on the ROUGE score analysis. The study employs the CNN-Daily Mail dataset, which consists of news articles and human-generated reference summaries. The evaluation employs ROUGE scores to assess the efficacy and quality of generated summaries. After Evaluation, we integrate the best-performing models on a web application to assess their real-world capabilities and user experience.

{{</citation>}}


### (11/144) Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Akari Asai et al., 2023)

{{<citation>}}

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi. (2023)  
**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, QA  
[Paper Link](http://arxiv.org/abs/2310.11511v1)  

---


**ABSTRACT**  
Despite their remarkable capabilities, large language models (LLMs) often produce responses containing factual inaccuracies due to their sole reliance on the parametric knowledge they encapsulate. Retrieval-Augmented Generation (RAG), an ad hoc approach that augments LMs with retrieval of relevant knowledge, decreases such issues. However, indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation. We introduce a new framework called Self-Reflective Retrieval-Augmented Generation (Self-RAG) that enhances an LM's quality and factuality through retrieval and self-reflection. Our framework trains a single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special tokens, called reflection tokens. Generating reflection tokens makes the LM controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements. Experiments show that Self-RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models on a diverse set of tasks. Specifically, Self-RAG outperforms ChatGPT and retrieval-augmented Llama2-chat on Open-domain QA, reasoning and fact verification tasks, and it shows significant gains in improving factuality and citation accuracy for long-form generations relative to these models.

{{</citation>}}


### (12/144) CoMPosT: Characterizing and Evaluating Caricature in LLM Simulations (Myra Cheng et al., 2023)

{{<citation>}}

Myra Cheng, Tiziano Piccardi, Diyi Yang. (2023)  
**CoMPosT: Characterizing and Evaluating Caricature in LLM Simulations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.11501v1)  

---


**ABSTRACT**  
Recent work has aimed to capture nuances of human behavior by using LLMs to simulate responses from particular demographics in settings like social science experiments and public opinion surveys. However, there are currently no established ways to discuss or evaluate the quality of such LLM simulations. Moreover, there is growing concern that these LLM simulations are flattened caricatures of the personas that they aim to simulate, failing to capture the multidimensionality of people and perpetuating stereotypes. To bridge these gaps, we present CoMPosT, a framework to characterize LLM simulations using four dimensions: Context, Model, Persona, and Topic. We use this framework to measure open-ended LLM simulations' susceptibility to caricature, defined via two criteria: individuation and exaggeration. We evaluate the level of caricature in scenarios from existing work on LLM simulations. We find that for GPT-4, simulations of certain demographics (political and marginalized groups) and topics (general, uncontroversial) are highly susceptible to caricature.

{{</citation>}}


### (13/144) VeRA: Vector-based Random Matrix Adaptation (Dawid Jan Kopiczko et al., 2023)

{{<citation>}}

Dawid Jan Kopiczko, Tijmen Blankevoort, Yuki Markus Asano. (2023)  
**VeRA: Vector-based Random Matrix Adaptation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE  
[Paper Link](http://arxiv.org/abs/2310.11454v1)  

---


**ABSTRACT**  
Low-rank adapation (LoRA) is a popular method that reduces the number of trainable parameters when finetuning large language models, but still faces acute storage challenges when scaling to even larger models or deploying numerous per-user or per-task adapted models. In this work, we present Vector-based Random Matrix Adaptation (VeRA), which reduces the number of trainable parameters by 10x compared to LoRA, yet maintains the same performance. It achieves this by using a single pair of low-rank matrices shared across all layers and learning small scaling vectors instead. We demonstrate its effectiveness on the GLUE and E2E benchmarks, and show its application in instruction-following with just 1.4M parameters using the Llama2 7B model.

{{</citation>}}


### (14/144) BitNet: Scaling 1-bit Transformers for Large Language Models (Hongyu Wang et al., 2023)

{{<citation>}}

Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei. (2023)  
**BitNet: Scaling 1-bit Transformers for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.11453v1)  

---


**ABSTRACT**  
The increasing size of large language models has posed challenges for deployment and raised concerns about environmental impact due to high energy consumption. In this work, we introduce BitNet, a scalable and stable 1-bit Transformer architecture designed for large language models. Specifically, we introduce BitLinear as a drop-in replacement of the nn.Linear layer in order to train 1-bit weights from scratch. Experimental results on language modeling show that BitNet achieves competitive performance while substantially reducing memory footprint and energy consumption, compared to state-of-the-art 8-bit quantization methods and FP16 Transformer baselines. Furthermore, BitNet exhibits a scaling law akin to full-precision Transformers, suggesting its potential for effective scaling to even larger language models while maintaining efficiency and performance benefits.

{{</citation>}}


### (15/144) Seeking Neural Nuggets: Knowledge Transfer in Large Language Models from a Parametric Perspective (Ming Zhong et al., 2023)

{{<citation>}}

Ming Zhong, Chenxin An, Weizhu Chen, Jiawei Han, Pengcheng He. (2023)  
**Seeking Neural Nuggets: Knowledge Transfer in Large Language Models from a Parametric Perspective**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11451v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) inherently encode a wealth of knowledge within their parameters through pre-training on extensive corpora. While prior research has delved into operations on these parameters to manipulate the underlying implicit knowledge (encompassing detection, editing, and merging), there remains an ambiguous understanding regarding their transferability across models with varying scales. In this paper, we seek to empirically investigate knowledge transfer from larger to smaller models through a parametric perspective. To achieve this, we employ sensitivity-based techniques to extract and align knowledge-specific parameters between different LLMs. Moreover, the LoRA module is used as the intermediary mechanism for injecting the extracted knowledge into smaller models. Evaluations across four benchmarks validate the efficacy of our proposed method. Our findings highlight the critical factors contributing to the process of parametric knowledge transfer, underscoring the transferability of model parameters across LLMs of different scales. We release code and data at \url{https://github.com/maszhongming/ParaKnowTransfer}.

{{</citation>}}


### (16/144) An Empirical Study of Translation Hypothesis Ensembling with Large Language Models (António Farinhas et al., 2023)

{{<citation>}}

António Farinhas, José G. C. de Souza, André F. T. Martins. (2023)  
**An Empirical Study of Translation Hypothesis Ensembling with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11430v1)  

---


**ABSTRACT**  
Large language models (LLMs) are becoming a one-fits-many solution, but they sometimes hallucinate or produce unreliable output. In this paper, we investigate how hypothesis ensembling can improve the quality of the generated text for the specific problem of LLM-based machine translation. We experiment with several techniques for ensembling hypotheses produced by LLMs such as ChatGPT, LLaMA, and Alpaca. We provide a comprehensive study along multiple dimensions, including the method to generate hypotheses (multiple prompts, temperature-based sampling, and beam search) and the strategy to produce the final translation (instruction-based, quality-based reranking, and minimum Bayes risk (MBR) decoding). Our results show that MBR decoding is a very effective method, that translation quality can be improved using a small number of samples, and that instruction tuning has a strong impact on the relation between the diversity of the hypotheses and the sampling temperature.

{{</citation>}}


### (17/144) Neural Attention: Enhancing QKV Calculation in Self-Attention Mechanism with Neural Networks (Muhan Zhang, 2023)

{{<citation>}}

Muhan Zhang. (2023)  
**Neural Attention: Enhancing QKV Calculation in Self-Attention Mechanism with Neural Networks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, BLEU, Self-Attention  
[Paper Link](http://arxiv.org/abs/2310.11398v1)  

---


**ABSTRACT**  
In the realm of deep learning, the self-attention mechanism has substantiated its pivotal role across a myriad of tasks, encompassing natural language processing and computer vision. Despite achieving success across diverse applications, the traditional self-attention mechanism primarily leverages linear transformations for the computation of query, key, and value (QKV), which may not invariably be the optimal choice under specific circumstances. This paper probes into a novel methodology for QKV computation-implementing a specially-designed neural network structure for the calculation. Utilizing a modified Marian model, we conducted experiments on the IWSLT 2017 German-English translation task dataset and juxtaposed our method with the conventional approach. The experimental results unveil a significant enhancement in BLEU scores with our method. Furthermore, our approach also manifested superiority when training the Roberta model with the Wikitext-103 dataset, reflecting a notable reduction in model perplexity compared to its original counterpart. These experimental outcomes not only validate the efficacy of our method but also reveal the immense potential in optimizing the self-attention mechanism through neural network-based QKV computation, paving the way for future research and practical applications. The source code and implementation details for our proposed method can be accessed at https://github.com/ocislyjrti/NeuralAttention.

{{</citation>}}


### (18/144) DialogueLLM: Context and Emotion Knowledge-Tuned LLaMA Models for Emotion Recognition in Conversations (Yazhou Zhang et al., 2023)

{{<citation>}}

Yazhou Zhang, Mengyao Wang, Prayag Tiwari, Qiuchi Li, Benyou Wang, Jing Qin. (2023)  
**DialogueLLM: Context and Emotion Knowledge-Tuned LLaMA Models for Emotion Recognition in Conversations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Emotion Recognition, LLaMA, NLP  
[Paper Link](http://arxiv.org/abs/2310.11374v1)  

---


**ABSTRACT**  
Large language models (LLMs) and their variants have shown extraordinary efficacy across numerous downstream natural language processing (NLP) tasks, which has presented a new vision for the development of NLP. Despite their remarkable performance in natural language generating (NLG), LLMs lack a distinct focus on the emotion understanding domain. As a result, using LLMs for emotion recognition may lead to suboptimal and inadequate precision. Another limitation of LLMs is that they are typical trained without leveraging multi-modal information. To overcome these limitations, we propose DialogueLLM, a context and emotion knowledge tuned LLM that is obtained by fine-tuning LLaMA models with 13,638 multi-modal (i.e., texts and videos) emotional dialogues. The visual information is considered as the supplementary knowledge to construct high-quality instructions. We offer a comprehensive evaluation of our proposed model on three benchmarking emotion recognition in conversations (ERC) datasets and compare the results against the SOTA baselines and other SOTA LLMs. Additionally, DialogueLLM-7B can be easily trained using LoRA on a 40GB A100 GPU in 5 hours, facilitating reproducibility for other researchers.

{{</citation>}}


### (19/144) VECHR: A Dataset for Explainable and Robust Classification of Vulnerability Type in the European Court of Human Rights (Shanshan Xu et al., 2023)

{{<citation>}}

Shanshan Xu, Leon Staufer, Santosh T. Y. S. S, Oana Ichim, Corina Heri, Matthias Grabmair. (2023)  
**VECHR: A Dataset for Explainable and Robust Classification of Vulnerability Type in the European Court of Human Rights**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.11368v2)  

---


**ABSTRACT**  
Recognizing vulnerability is crucial for understanding and implementing targeted support to empower individuals in need. This is especially important at the European Court of Human Rights (ECtHR), where the court adapts Convention standards to meet actual individual needs and thus ensures effective human rights protection. However, the concept of vulnerability remains elusive at the ECtHR and no prior NLP research has dealt with it. To enable future research in this area, we present VECHR, a novel expert-annotated multi-label dataset comprising of vulnerability type classification and explanation rationale. We benchmark the performance of state-of-the-art models on VECHR from both prediction and explainability perspectives. Our results demonstrate the challenging nature of the task with lower prediction performance and limited agreement between models and experts. Further, we analyze the robustness of these models in dealing with out-of-domain (OOD) data and observe overall limited performance. Our dataset poses unique challenges offering significant room for improvement regarding performance, explainability, and robustness.

{{</citation>}}


### (20/144) Disentangling the Linguistic Competence of Privacy-Preserving BERT (Stefan Arnold et al., 2023)

{{<citation>}}

Stefan Arnold, Nils Kemmerzell, Annika Schreiner. (2023)  
**Disentangling the Linguistic Competence of Privacy-Preserving BERT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.11363v1)  

---


**ABSTRACT**  
Differential Privacy (DP) has been tailored to address the unique challenges of text-to-text privatization. However, text-to-text privatization is known for degrading the performance of language models when trained on perturbed text. Employing a series of interpretation techniques on the internal representations extracted from BERT trained on perturbed pre-text, we intend to disentangle at the linguistic level the distortion induced by differential privacy. Experimental results from a representational similarity analysis indicate that the overall similarity of internal representations is substantially reduced. Using probing tasks to unpack this dissimilarity, we find evidence that text-to-text privatization affects the linguistic competence across several formalisms, encoding localized properties of words while falling short at encoding the contextual relationships between spans of words.

{{</citation>}}


### (21/144) Enhancing Neural Machine Translation with Semantic Units (Langlin Huang et al., 2023)

{{<citation>}}

Langlin Huang, Shuhao Gu, Zhuocheng Zhang, Yang Feng. (2023)  
**Enhancing Neural Machine Translation with Semantic Units**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.11360v1)  

---


**ABSTRACT**  
Conventional neural machine translation (NMT) models typically use subwords and words as the basic units for model input and comprehension. However, complete words and phrases composed of several tokens are often the fundamental units for expressing semantics, referred to as semantic units. To address this issue, we propose a method Semantic Units for Machine Translation (SU4MT) which models the integral meanings of semantic units within a sentence, and then leverages them to provide a new perspective for understanding the sentence. Specifically, we first propose Word Pair Encoding (WPE), a phrase extraction method to help identify the boundaries of semantic units. Next, we design an Attentive Semantic Fusion (ASF) layer to integrate the semantics of multiple subwords into a single vector: the semantic unit representation. Lastly, the semantic-unit-level sentence representation is concatenated to the token-level one, and they are combined as the input of encoder. Experimental results demonstrate that our method effectively models and leverages semantic-unit-level information and outperforms the strong baselines. The code is available at https://github.com/ictnlp/SU4MT.

{{</citation>}}


### (22/144) Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting (Melanie Sclar et al., 2023)

{{<citation>}}

Melanie Sclar, Yejin Choi, Yulia Tsvetkov, Alane Suhr. (2023)  
**Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11324v1)  

---


**ABSTRACT**  
As large language models (LLMs) are adopted as a fundamental component of language technologies, it is crucial to accurately characterize their performance. Because choices in prompt design can strongly influence model behavior, this design process is critical in effectively using any modern pre-trained generative language model. In this work, we focus on LLM sensitivity to a quintessential class of meaning-preserving design choices: prompt formatting. We find that several widely used open-source LLMs are extremely sensitive to subtle changes in prompt formatting in few-shot settings, with performance differences of up to 76 accuracy points when evaluated using LLaMA-2-13B. Sensitivity remains even when increasing model size, the number of few-shot examples, or performing instruction tuning. Our analysis suggests that work evaluating LLMs with prompting-based methods would benefit from reporting a range of performance across plausible prompt formats, instead of the currently-standard practice of reporting performance on a single format. We also show that format performance only weakly correlates between models, which puts into question the methodological validity of comparing models with an arbitrarily chosen, fixed prompt format. To facilitate systematic analysis we propose FormatSpread, an algorithm that rapidly evaluates a sampled set of plausible prompt formats for a given task, and reports the interval of expected performance without accessing model weights. Furthermore, we present a suite of analyses that characterize the nature of this sensitivity, including exploring the influence of particular atomic perturbations and the internal representation of particular formats.

{{</citation>}}


### (23/144) Utilising a Large Language Model to Annotate Subject Metadata: A Case Study in an Australian National Research Data Catalogue (Shiwei Zhang et al., 2023)

{{<citation>}}

Shiwei Zhang, Mingfang Wu, Xiuzhen Zhang. (2023)  
**Utilising a Large Language Model to Annotate Subject Metadata: A Case Study in an Australian National Research Data Catalogue**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11318v1)  

---


**ABSTRACT**  
In support of open and reproducible research, there has been a rapidly increasing number of datasets made available for research. As the availability of datasets increases, it becomes more important to have quality metadata for discovering and reusing them. Yet, it is a common issue that datasets often lack quality metadata due to limited resources for data curation. Meanwhile, technologies such as artificial intelligence and large language models (LLMs) are progressing rapidly. Recently, systems based on these technologies, such as ChatGPT, have demonstrated promising capabilities for certain data curation tasks. This paper proposes to leverage LLMs for cost-effective annotation of subject metadata through the LLM-based in-context learning. Our method employs GPT-3.5 with prompts designed for annotating subject metadata, demonstrating promising performance in automatic metadata annotation. However, models based on in-context learning cannot acquire discipline-specific rules, resulting in lower performance in several categories. This limitation arises from the limited contextual information available for subject inference. To the best of our knowledge, we are introducing, for the first time, an in-context learning method that harnesses large language models for automated subject metadata annotation.

{{</citation>}}


### (24/144) QADYNAMICS: Training Dynamics-Driven Synthetic QA Diagnostic for Zero-Shot Commonsense Question Answering (Haochen Shi et al., 2023)

{{<citation>}}

Haochen Shi, Weiqi Wang, Tianqing Fang, Baixuan Xu, Wenxuan Ding, Xin Liu, Yangqiu Song. (2023)  
**QADYNAMICS: Training Dynamics-Driven Synthetic QA Diagnostic for Zero-Shot Commonsense Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, QA, Question Answering, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.11303v1)  

---


**ABSTRACT**  
Zero-shot commonsense Question-Answering (QA) requires models to reason about general situations beyond specific benchmarks. State-of-the-art approaches fine-tune language models on QA pairs constructed from CommonSense Knowledge Bases (CSKBs) to equip the models with more commonsense knowledge in a QA context. However, current QA synthesis protocols may introduce noise from the CSKBs and generate ungrammatical questions and false negative options, which impede the model's ability to generalize. To address these issues, we propose QADYNAMICS, a training dynamics-driven framework for QA diagnostics and refinement. Our approach analyzes the training dynamics of each QA pair at both the question level and option level, discarding machine-detectable artifacts by removing uninformative QA pairs and mislabeled or false-negative options. Extensive experiments demonstrate the effectiveness of our approach, which outperforms all baselines while using only 33% of the synthetic data, even including LLMs such as ChatGPT. Moreover, expert evaluations confirm that our framework significantly improves the quality of QA synthesis. Our codes and model checkpoints are available at https://github.com/HKUST-KnowComp/QaDynamics.

{{</citation>}}


### (25/144) ChapGTP, ILLC's Attempt at Raising a BabyLM: Improving Data Efficiency by Automatic Task Formation (Jaap Jumelet et al., 2023)

{{<citation>}}

Jaap Jumelet, Michael Hanna, Marianne de Heer Kloots, Anna Langedijk, Charlotte Pouw, Oskar van der Wal. (2023)  
**ChapGTP, ILLC's Attempt at Raising a BabyLM: Improving Data Efficiency by Automatic Task Formation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE  
[Paper Link](http://arxiv.org/abs/2310.11282v1)  

---


**ABSTRACT**  
We present the submission of the ILLC at the University of Amsterdam to the BabyLM challenge (Warstadt et al., 2023), in the strict-small track. Our final model, ChapGTP, is a masked language model that was trained for 200 epochs, aided by a novel data augmentation technique called Automatic Task Formation. We discuss in detail the performance of this model on the three evaluation suites: BLiMP, (Super)GLUE, and MSGS. Furthermore, we present a wide range of methods that were ultimately not included in the model, but may serve as inspiration for training LMs in low-resource settings.

{{</citation>}}


### (26/144) Emulating Human Cognitive Processes for Expert-Level Medical Question-Answering with Large Language Models (Khushboo Verma et al., 2023)

{{<citation>}}

Khushboo Verma, Marina Moore, Stephanie Wottrich, Karla Robles López, Nishant Aggarwal, Zeel Bhatt, Aagamjit Singh, Bradford Unroe, Salah Basheer, Nitish Sachdeva, Prinka Arora, Harmanjeet Kaur, Tanupreet Kaur, Tevon Hood, Anahi Marquez, Tushar Varshney, Nanfu Deng, Azaan Ramani, Pawanraj Ishwara, Maimoona Saeed, Tatiana López Velarde Peña, Bryan Barksdale, Sushovan Guha, Satwant Kumar. (2023)  
**Emulating Human Cognitive Processes for Expert-Level Medical Question-Answering with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-NE, cs.CL  
Keywords: ChatGPT, GPT, Language Model, PaLM, QA  
[Paper Link](http://arxiv.org/abs/2310.11266v1)  

---


**ABSTRACT**  
In response to the pressing need for advanced clinical problem-solving tools in healthcare, we introduce BooksMed, a novel framework based on a Large Language Model (LLM). BooksMed uniquely emulates human cognitive processes to deliver evidence-based and reliable responses, utilizing the GRADE (Grading of Recommendations, Assessment, Development, and Evaluations) framework to effectively quantify evidence strength. For clinical decision-making to be appropriately assessed, an evaluation metric that is clinically aligned and validated is required. As a solution, we present ExpertMedQA, a multispecialty clinical benchmark comprised of open-ended, expert-level clinical questions, and validated by a diverse group of medical professionals. By demanding an in-depth understanding and critical appraisal of up-to-date clinical literature, ExpertMedQA rigorously evaluates LLM performance. BooksMed outperforms existing state-of-the-art models Med-PaLM 2, Almanac, and ChatGPT in a variety of medical scenarios. Therefore, a framework that mimics human cognitive stages could be a useful tool for providing reliable and evidence-based responses to clinical inquiries.

{{</citation>}}


### (27/144) Utilizing Weak Supervision To Generate Indonesian Conservation Dataset (Mega Fransiska et al., 2023)

{{<citation>}}

Mega Fransiska, Diah Pitaloka, Saripudin, Satrio Putra, Lintang Sutawika. (2023)  
**Utilizing Weak Supervision To Generate Indonesian Conservation Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.11258v1)  

---


**ABSTRACT**  
Weak supervision has emerged as a promising approach for rapid and large-scale dataset creation in response to the increasing demand for accelerated NLP development. By leveraging labeling functions, weak supervision allows practitioners to generate datasets quickly by creating learned label models that produce soft-labeled datasets. This paper aims to show how such an approach can be utilized to build an Indonesian NLP dataset from conservation news text. We construct two types of datasets: multi-class classification and sentiment classification. We then provide baseline experiments using various pretrained language models. These baseline results demonstrate test performances of 59.79% accuracy and 55.72% F1-score for sentiment classification, 66.87% F1-score-macro, 71.5% F1-score-micro, and 83.67% ROC-AUC for multi-class classification. Additionally, we release the datasets and labeling functions used in this work for further research and exploration.

{{</citation>}}


### (28/144) Revealing the Unwritten: Visual Investigation of Beam Search Trees to Address Language Model Prompting Challenges (Thilo Spinner et al., 2023)

{{<citation>}}

Thilo Spinner, Rebecca Kehlbeck, Rita Sevastjanova, Tobias Stähle, Daniel A. Keim, Oliver Deussen, Andreas Spitz, Mennatallah El-Assady. (2023)  
**Revealing the Unwritten: Visual Investigation of Beam Search Trees to Address Language Model Prompting Challenges**  

---
Primary Category: cs.CL  
Categories: H-5-2; I-2-7, cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11252v1)  

---


**ABSTRACT**  
The growing popularity of generative language models has amplified interest in interactive methods to guide model outputs. Prompt refinement is considered one of the most effective means to influence output among these methods. We identify several challenges associated with prompting large language models, categorized into data- and model-specific, linguistic, and socio-linguistic challenges. A comprehensive examination of model outputs, including runner-up candidates and their corresponding probabilities, is needed to address these issues. The beam search tree, the prevalent algorithm to sample model outputs, can inherently supply this information. Consequently, we introduce an interactive visual method for investigating the beam search tree, facilitating analysis of the decisions made by the model during generation. We quantitatively show the value of exposing the beam search tree and present five detailed analysis scenarios addressing the identified challenges. Our methodology validates existing results and offers additional insights.

{{</citation>}}


### (29/144) Entity Matching using Large Language Models (Ralph Peeters et al., 2023)

{{<citation>}}

Ralph Peeters, Christian Bizer. (2023)  
**Entity Matching using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11244v1)  

---


**ABSTRACT**  
Entity Matching is the task of deciding whether two entity descriptions refer to the same real-world entity. Entity Matching is a central step in most data integration pipelines and an enabler for many e-commerce applications which require to match products offers from different vendors. State-of-the-art entity matching methods often rely on pre-trained language models (PLMs) such as BERT or RoBERTa. Two major drawbacks of these models for entity matching are that (i) the models require significant amounts of task-specific training data and (ii) the fine-tuned models are not robust concerning out-of-distribution entities. In this paper, we investigate using large language models (LLMs) for entity matching as a less domain-specific training data reliant and more robust alternative to PLM-based matchers. Our study covers hosted LLMs, such as GPT3.5 and GPT4, as well as open source LLMs based on Llama2 which can be run locally. We evaluate these models in a zero-shot scenario as well as a scenario where task-specific training data is available. We compare different prompt designs as well as the prompt sensitivity of the models in the zero-shot scenario. We investigate (i) the selection of in-context demonstrations, (ii) the generation of matching rules, as well as (iii) fine-tuning GPT3.5 in the second scenario using the same pool of training data across the different approaches. Our experiments show that GPT4 without any task-specific training data outperforms fine-tuned PLMs (RoBERTa and Ditto) on three out of five benchmark datasets reaching F1 scores around 90%. The experiments with in-context learning and rule generation show that all models beside of GPT4 benefit from these techniques (on average 5.9% and 2.2% F1), while GPT4 does not need such additional guidance in most cases...

{{</citation>}}


### (30/144) Watermarking LLMs with Weight Quantization (Linyang Li et al., 2023)

{{<citation>}}

Linyang Li, Botian Jiang, Pengyu Wang, Ke Ren, Hang Yan, Xipeng Qiu. (2023)  
**Watermarking LLMs with Weight Quantization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, LLaMA, Quantization  
[Paper Link](http://arxiv.org/abs/2310.11237v1)  

---


**ABSTRACT**  
Abuse of large language models reveals high risks as large language models are being deployed at an astonishing speed. It is important to protect the model weights to avoid malicious usage that violates licenses of open-source large language models. This paper proposes a novel watermarking strategy that plants watermarks in the quantization process of large language models without pre-defined triggers during inference. The watermark works when the model is used in the fp32 mode and remains hidden when the model is quantized to int8, in this way, the users can only inference the model without further supervised fine-tuning of the model. We successfully plant the watermark into open-source large language model weights including GPT-Neo and LLaMA. We hope our proposed method can provide a potential direction for protecting model weights in the era of large language model applications.

{{</citation>}}


### (31/144) KG-GPT: A General Framework for Reasoning on Knowledge Graphs Using Large Language Models (Jiho Kim et al., 2023)

{{<citation>}}

Jiho Kim, Yeonsu Kwon, Yohan Jo, Edward Choi. (2023)  
**KG-GPT: A General Framework for Reasoning on Knowledge Graphs Using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Knowledge Graph, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.11220v1)  

---


**ABSTRACT**  
While large language models (LLMs) have made considerable advancements in understanding and generating unstructured text, their application in structured data remains underexplored. Particularly, using LLMs for complex reasoning tasks on knowledge graphs (KGs) remains largely untouched. To address this, we propose KG-GPT, a multi-purpose framework leveraging LLMs for tasks employing KGs. KG-GPT comprises three steps: Sentence Segmentation, Graph Retrieval, and Inference, each aimed at partitioning sentences, retrieving relevant graph components, and deriving logical conclusions, respectively. We evaluate KG-GPT using KG-based fact verification and KGQA benchmarks, with the model showing competitive and robust performance, even outperforming several fully-supervised models. Our work, therefore, marks a significant step in unifying structured and unstructured data processing within the realm of LLMs.

{{</citation>}}


### (32/144) Can Large Language Models Explain Themselves? A Study of LLM-Generated Self-Explanations (Shiyuan Huang et al., 2023)

{{<citation>}}

Shiyuan Huang, Siddarth Mamidanna, Shreedhar Jangam, Yilun Zhou, Leilani H. Gilpin. (2023)  
**Can Large Language Models Explain Themselves? A Study of LLM-Generated Self-Explanations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.11207v1)  

---


**ABSTRACT**  
Large language models (LLMs) such as ChatGPT have demonstrated superior performance on a variety of natural language processing (NLP) tasks including sentiment analysis, mathematical reasoning and summarization. Furthermore, since these models are instruction-tuned on human conversations to produce "helpful" responses, they can and often will produce explanations along with the response, which we call self-explanations. For example, when analyzing the sentiment of a movie review, the model may output not only the positivity of the sentiment, but also an explanation (e.g., by listing the sentiment-laden words such as "fantastic" and "memorable" in the review). How good are these automatically generated self-explanations? In this paper, we investigate this question on the task of sentiment analysis and for feature attribution explanation, one of the most commonly studied settings in the interpretability literature (for pre-ChatGPT models). Specifically, we study different ways to elicit the self-explanations, evaluate their faithfulness on a set of evaluation metrics, and compare them to traditional explanation methods such as occlusion or LIME saliency maps. Through an extensive set of experiments, we find that ChatGPT's self-explanations perform on par with traditional ones, but are quite different from them according to various agreement metrics, meanwhile being much cheaper to produce (as they are generated along with the prediction). In addition, we identified several interesting characteristics of them, which prompt us to rethink many current model interpretability practices in the era of ChatGPT(-like) LLMs.

{{</citation>}}


### (33/144) Medical Text Simplification: Optimizing for Readability with Unlikelihood Training and Reranked Beam Search Decoding (Lorenzo Jaime Yu Flores et al., 2023)

{{<citation>}}

Lorenzo Jaime Yu Flores, Heyuan Huang, Kejian Shi, Sophie Chheang, Arman Cohan. (2023)  
**Medical Text Simplification: Optimizing for Readability with Unlikelihood Training and Reranked Beam Search Decoding**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11191v1)  

---


**ABSTRACT**  
Text simplification has emerged as an increasingly useful application of AI for bridging the communication gap in specialized fields such as medicine, where the lexicon is often dominated by technical jargon and complex constructs. Despite notable progress, methods in medical simplification sometimes result in the generated text having lower quality and diversity. In this work, we explore ways to further improve the readability of text simplification in the medical domain. We propose (1) a new unlikelihood loss that encourages generation of simpler terms and (2) a reranked beam search decoding method that optimizes for simplicity, which achieve better performance on readability metrics on three datasets. This study's findings offer promising avenues for improving text simplification in the medical field.

{{</citation>}}


### (34/144) ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing (Quoc-Nam Nguyen et al., 2023)

{{<citation>}}

Quoc-Nam Nguyen, Thang Chau Phan, Duc-Vu Nguyen, Kiet Van Nguyen. (2023)  
**ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, NLP, Social Media  
[Paper Link](http://arxiv.org/abs/2310.11166v1)  

---


**ABSTRACT**  
English and Chinese, known as resource-rich languages, have witnessed the strong development of transformer-based language models for natural language processing tasks. Although Vietnam has approximately 100M people speaking Vietnamese, several pre-trained models, e.g., PhoBERT, ViBERT, and vELECTRA, performed well on general Vietnamese NLP tasks, including POS tagging and named entity recognition. These pre-trained language models are still limited to Vietnamese social media tasks. In this paper, we present the first monolingual pre-trained language model for Vietnamese social media texts, ViSoBERT, which is pre-trained on a large-scale corpus of high-quality and diverse Vietnamese social media texts using XLM-R architecture. Moreover, we explored our pre-trained model on five important natural language downstream tasks on Vietnamese social media texts: emotion recognition, hate speech detection, sentiment analysis, spam reviews detection, and hate speech spans detection. Our experiments demonstrate that ViSoBERT, with far fewer parameters, surpasses the previous state-of-the-art models on multiple Vietnamese social media tasks. Our ViSoBERT model is available\footnote{\url{https://huggingface.co/uitnlp/visobert}} only for research purposes.

{{</citation>}}


### (35/144) IMTLab: An Open-Source Platform for Building, Evaluating, and Diagnosing Interactive Machine Translation Systems (Xu Huang et al., 2023)

{{<citation>}}

Xu Huang, Zhirui Zhang, Ruize Gao, Yichao Du, Lemao Liu, Gouping Huang, Shuming Shi, Jiajun Chen, Shujian Huang. (2023)  
**IMTLab: An Open-Source Platform for Building, Evaluating, and Diagnosing Interactive Machine Translation Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.11163v1)  

---


**ABSTRACT**  
We present IMTLab, an open-source end-to-end interactive machine translation (IMT) system platform that enables researchers to quickly build IMT systems with state-of-the-art models, perform an end-to-end evaluation, and diagnose the weakness of systems. IMTLab treats the whole interactive translation process as a task-oriented dialogue with a human-in-the-loop setting, in which human interventions can be explicitly incorporated to produce high-quality, error-free translations. To this end, a general communication interface is designed to support the flexible IMT architectures and user policies. Based on the proposed design, we construct a simulated and real interactive environment to achieve end-to-end evaluation and leverage the framework to systematically evaluate previous IMT systems. Our simulated and manual experiments show that the prefix-constrained decoding approach still gains the lowest editing cost in the end-to-end evaluation, while BiTIIMT achieves comparable editing cost with a better interactive experience.

{{</citation>}}


### (36/144) Probing the Creativity of Large Language Models: Can models produce divergent semantic association? (Honghua Chen et al., 2023)

{{<citation>}}

Honghua Chen, Nai Ding. (2023)  
**Probing the Creativity of Large Language Models: Can models produce divergent semantic association?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11158v1)  

---


**ABSTRACT**  
Large language models possess remarkable capacity for processing language, but it remains unclear whether these models can further generate creative content. The present study aims to investigate the creative thinking of large language models through a cognitive perspective. We utilize the divergent association task (DAT), an objective measurement of creativity that asks models to generate unrelated words and calculates the semantic distance between them. We compare the results across different models and decoding strategies. Our findings indicate that: (1) When using the greedy search strategy, GPT-4 outperforms 96% of humans, while GPT-3.5-turbo exceeds the average human level. (2) Stochastic sampling and temperature scaling are effective to obtain higher DAT scores for models except GPT-4, but face a trade-off between creativity and stability. These results imply that advanced large language models have divergent semantic associations, which is a fundamental process underlying creativity.

{{</citation>}}


### (37/144) The Quo Vadis of the Relationship between Language and Large Language Models (Evelina Leivada et al., 2023)

{{<citation>}}

Evelina Leivada, Vittoria Dentella, Elliot Murphy. (2023)  
**The Quo Vadis of the Relationship between Language and Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.11146v1)  

---


**ABSTRACT**  
In the field of Artificial (General) Intelligence (AI), the several recent advancements in Natural language processing (NLP) activities relying on Large Language Models (LLMs) have come to encourage the adoption of LLMs as scientific models of language. While the terminology employed for the characterization of LLMs favors their embracing as such, it is not clear that they are in a place to offer insights into the target system they seek to represent. After identifying the most important theoretical and empirical risks brought about by the adoption of scientific models that lack transparency, we discuss LLMs relating them to every scientific model's fundamental components: the object, the medium, the meaning and the user. We conclude that, at their current stage of development, LLMs hardly offer any explanations for language, and then we provide an outlook for more informative future research directions on this topic.

{{</citation>}}


### (38/144) H2O Open Ecosystem for State-of-the-art Large Language Models (Arno Candel et al., 2023)

{{<citation>}}

Arno Candel, Jon McKinney, Philipp Singer, Pascal Pfeiffer, Maximilian Jeblick, Chun Ming Lee, Marcos V. Conde. (2023)  
**H2O Open Ecosystem for State-of-the-art Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13012v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) represent a revolution in AI. However, they also pose many significant risks, such as the presence of biased, private, copyrighted or harmful text. For this reason we need open, transparent and safe solutions. We introduce a complete open-source ecosystem for developing and testing LLMs. The goal of this project is to boost open alternatives to closed-source approaches. We release h2oGPT, a family of fine-tuned LLMs from 7 to 70 Billion parameters. We also introduce H2O LLM Studio, a framework and no-code GUI designed for efficient fine-tuning, evaluation, and deployment of LLMs using the most recent state-of-the-art techniques. Our code and models are licensed under fully permissive Apache 2.0 licenses. We believe open-source language models help to boost AI development and make it more accessible and trustworthy. The demo is available at: https://gpt.h2o.ai/

{{</citation>}}


### (39/144) Experimenting AI Technologies for Disinformation Combat: the IDMO Project (Lorenzo Canale et al., 2023)

{{<citation>}}

Lorenzo Canale, Alberto Messina. (2023)  
**Experimenting AI Technologies for Disinformation Combat: the IDMO Project**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.11097v3)  

---


**ABSTRACT**  
The Italian Digital Media Observatory (IDMO) project, part of a European initiative, focuses on countering disinformation and fake news. This report outlines contributions from Rai-CRITS to the project, including: (i) the creation of novel datasets for testing technologies (ii) development of an automatic model for categorizing Pagella Politica verdicts to facilitate broader analysis (iii) creation of an automatic model for recognizing textual entailment with exceptional accuracy on the FEVER dataset (iv) assessment using GPT-4 to identify textual entailmen (v) a game to raise awareness about fake news at national events.

{{</citation>}}


### (40/144) In-Context Few-Shot Relation Extraction via Pre-Trained Language Models (Yilmazcan Ozyurt et al., 2023)

{{<citation>}}

Yilmazcan Ozyurt, Stefan Feuerriegel, Ce Zhang. (2023)  
**In-Context Few-Shot Relation Extraction via Pre-Trained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Few-Shot, Language Model, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2310.11085v1)  

---


**ABSTRACT**  
Relation extraction aims at inferring structured human knowledge from textual documents. State-of-the-art methods based on language models commonly have two limitations: (1) they require named entities to be either given as input or infer them, which introduces additional noise, and (2) they require human annotations of documents. As a remedy, we present a novel framework for in-context few-shot relation extraction via pre-trained language models. To the best of our knowledge, we are the first to reformulate the relation extraction task as a tailored in-context few-shot learning paradigm. Thereby, we achieve crucial benefits in that we eliminate the need for both named entity recognition and human annotation of documents. Unlike existing methods based on fine-tuning, our framework is flexible in that it can be easily updated for a new set of relations without re-training. We evaluate our framework using DocRED, the largest publicly available dataset for document-level relation extraction, and demonstrate that our framework achieves state-of-the-art performance. Finally, our framework allows us to identify missing annotations, and we thus show that our framework actually performs much better than the original labels from the development set of DocRED.

{{</citation>}}


### (41/144) Understanding writing style in social media with a supervised contrastively pre-trained transformer (Javier Huertas-Tato et al., 2023)

{{<citation>}}

Javier Huertas-Tato, Alejandro Martin, David Camacho. (2023)  
**Understanding writing style in social media with a supervised contrastively pre-trained transformer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: AI, Social Network, Transformer  
[Paper Link](http://arxiv.org/abs/2310.11081v1)  

---


**ABSTRACT**  
Online Social Networks serve as fertile ground for harmful behavior, ranging from hate speech to the dissemination of disinformation. Malicious actors now have unprecedented freedom to misbehave, leading to severe societal unrest and dire consequences, as exemplified by events such as the Capitol assault during the US presidential election and the Antivaxx movement during the COVID-19 pandemic. Understanding online language has become more pressing than ever. While existing works predominantly focus on content analysis, we aim to shift the focus towards understanding harmful behaviors by relating content to their respective authors. Numerous novel approaches attempt to learn the stylistic features of authors in texts, but many of these approaches are constrained by small datasets or sub-optimal training losses. To overcome these limitations, we introduce the Style Transformer for Authorship Representations (STAR), trained on a large corpus derived from public sources of 4.5 x 10^6 authored texts involving 70k heterogeneous authors. Our model leverages Supervised Contrastive Loss to teach the model to minimize the distance between texts authored by the same individual. This author pretext pre-training task yields competitive performance at zero-shot with PAN challenges on attribution and clustering. Additionally, we attain promising results on PAN verification challenges using a single dense layer, with our model serving as an embedding encoder. Finally, we present results from our test partition on Reddit. Using a support base of 8 documents of 512 tokens, we can discern authors from sets of up to 1616 authors with at least 80\% accuracy. We share our pre-trained model at huggingface (https://huggingface.co/AIDA-UPM/star) and our code is available at (https://github.com/jahuerta92/star)

{{</citation>}}


### (42/144) Learning from Red Teaming: Gender Bias Provocation and Mitigation in Large Language Models (Hsuan Su et al., 2023)

{{<citation>}}

Hsuan Su, Cheng-Chu Cheng, Hua Farn, Shachi H Kumar, Saurav Sahay, Shang-Tse Chen, Hung-yi Lee. (2023)  
**Learning from Red Teaming: Gender Bias Provocation and Mitigation in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11079v1)  

---


**ABSTRACT**  
Recently, researchers have made considerable improvements in dialogue systems with the progress of large language models (LLMs) such as ChatGPT and GPT-4. These LLM-based chatbots encode the potential biases while retaining disparities that can harm humans during interactions. The traditional biases investigation methods often rely on human-written test cases. However, these test cases are usually expensive and limited. In this work, we propose a first-of-its-kind method that automatically generates test cases to detect LLMs' potential gender bias. We apply our method to three well-known LLMs and find that the generated test cases effectively identify the presence of biases. To address the biases identified, we propose a mitigation strategy that uses the generated test cases as demonstrations for in-context learning to circumvent the need for parameter fine-tuning. The experimental results show that LLMs generate fairer responses with the proposed approach.

{{</citation>}}


### (43/144) VoxArabica: A Robust Dialect-Aware Arabic Speech Recognition System (Abdul Waheed et al., 2023)

{{<citation>}}

Abdul Waheed, Bashar Talafha, Peter Suvellin, Abdelrahman Elmadney, Muhammad Abdul-Mageed. (2023)  
**VoxArabica: A Robust Dialect-Aware Arabic Speech Recognition System**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: BERT, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.11069v1)  

---


**ABSTRACT**  
Arabic is a complex language with many varieties and dialects spoken by over 450 millions all around the world. Due to the linguistic diversity and variations, it is challenging to build a robust and generalized ASR system for Arabic. In this work, we address this gap by developing and demoing a system, dubbed VoxArabica, for dialect identification (DID) as well as automatic speech recognition (ASR) of Arabic. We train a wide range of models such as HuBERT (DID), Whisper, and XLS-R (ASR) in a supervised setting for Arabic DID and ASR tasks. Our DID models are trained to identify 17 different dialects in addition to MSA. We finetune our ASR models on MSA, Egyptian, Moroccan, and mixed data. Additionally, for the remaining dialects in ASR, we provide the option to choose various models such as Whisper and MMS in a zero-shot setting. We integrate these models into a single web interface with diverse features such as audio recording, file upload, model selection, and the option to raise flags for incorrect outputs. Overall, we believe VoxArabica will be useful for a wide range of audiences concerned with Arabic research. Our system is currently running at https://cdce-206-12-100-168.ngrok.io/.

{{</citation>}}


### (44/144) Denevil: Towards Deciphering and Navigating the Ethical Values of Large Language Models via Instruction Learning (Shitong Duan et al., 2023)

{{<citation>}}

Shitong Duan, Xiaoyuan Yi, Peng Zhang, Tun Lu, Xing Xie, Ning Gu. (2023)  
**Denevil: Towards Deciphering and Navigating the Ethical Values of Large Language Models via Instruction Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11053v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have made unprecedented breakthroughs, yet their increasing integration into everyday life might raise societal risks due to generated unethical content. Despite extensive study on specific issues like bias, the intrinsic values of LLMs remain largely unexplored from a moral philosophy perspective. This work delves into ethical values utilizing Moral Foundation Theory. Moving beyond conventional discriminative evaluations with poor reliability, we propose DeNEVIL, a novel prompt generation algorithm tailored to dynamically exploit LLMs' value vulnerabilities and elicit the violation of ethics in a generative manner, revealing their underlying value inclinations. On such a basis, we construct MoralPrompt, a high-quality dataset comprising 2,397 prompts covering 500+ value principles, and then benchmark the intrinsic values across a spectrum of LLMs. We discovered that most models are essentially misaligned, necessitating further ethical value alignment. In response, we develop VILMO, an in-context alignment method that substantially enhances the value compliance of LLM outputs by learning to generate appropriate value instructions, outperforming existing competitors. Our methods are suitable for black-box and open-source models, offering a promising initial step in studying the ethical values of LLMs.

{{</citation>}}


### (45/144) Nonet at SemEval-2023 Task 6: Methodologies for Legal Evaluation (Shubham Kumar Nigam et al., 2023)

{{<citation>}}

Shubham Kumar Nigam, Aniket Deroy, Noel Shallum, Ayush Kumar Mishra, Anup Roy, Shubham Kumar Mishra, Arnab Bhattacharya, Saptarshi Ghosh, Kripabandhu Ghosh. (2023)  
**Nonet at SemEval-2023 Task 6: Methodologies for Legal Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Legal, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2310.11049v1)  

---


**ABSTRACT**  
This paper describes our submission to the SemEval-2023 for Task 6 on LegalEval: Understanding Legal Texts. Our submission concentrated on three subtasks: Legal Named Entity Recognition (L-NER) for Task-B, Legal Judgment Prediction (LJP) for Task-C1, and Court Judgment Prediction with Explanation (CJPE) for Task-C2. We conducted various experiments on these subtasks and presented the results in detail, including data statistics and methodology. It is worth noting that legal tasks, such as those tackled in this research, have been gaining importance due to the increasing need to automate legal analysis and support. Our team obtained competitive rankings of 15$^{th}$, 11$^{th}$, and 1$^{st}$ in Task-B, Task-C1, and Task-C2, respectively, as reported on the leaderboard.

{{</citation>}}


### (46/144) Exploring Automatic Evaluation Methods based on a Decoder-based LLM for Text Generation (Tomohito Kasahara et al., 2023)

{{<citation>}}

Tomohito Kasahara, Daisuke Kawahara. (2023)  
**Exploring Automatic Evaluation Methods based on a Decoder-based LLM for Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Text Generation  
[Paper Link](http://arxiv.org/abs/2310.11026v1)  

---


**ABSTRACT**  
Automatic evaluation of text generation is essential for improving the accuracy of generation tasks. In light of the current trend towards increasingly larger decoder-based language models, we investigate automatic evaluation methods based on such models for text generation. This paper compares various methods, including tuning with encoder-based models and large language models under equal conditions, on two different tasks, machine translation evaluation and semantic textual similarity, in two languages, Japanese and English. Experimental results show that compared to the tuned encoder-based models, the tuned decoder-based models perform poorly. The analysis of the causes for this suggests that the decoder-based models focus on surface word sequences and do not capture meaning. It is also revealed that in-context learning of very large decoder-based models such as ChatGPT makes it difficult to identify fine-grained semantic differences.

{{</citation>}}


### (47/144) Reading Order Matters: Information Extraction from Visually-rich Documents by Token Path Prediction (Chong Zhang et al., 2023)

{{<citation>}}

Chong Zhang, Ya Guo, Yi Tu, Huan Chen, Jinyang Tang, Huijia Zhu, Qi Zhang, Tao Gui. (2023)  
**Reading Order Matters: Information Extraction from Visually-rich Documents by Token Path Prediction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction, NER, NLP, OCR  
[Paper Link](http://arxiv.org/abs/2310.11016v1)  

---


**ABSTRACT**  
Recent advances in multimodal pre-trained models have significantly improved information extraction from visually-rich documents (VrDs), in which named entity recognition (NER) is treated as a sequence-labeling task of predicting the BIO entity tags for tokens, following the typical setting of NLP. However, BIO-tagging scheme relies on the correct order of model inputs, which is not guaranteed in real-world NER on scanned VrDs where text are recognized and arranged by OCR systems. Such reading order issue hinders the accurate marking of entities by BIO-tagging scheme, making it impossible for sequence-labeling methods to predict correct named entities. To address the reading order issue, we introduce Token Path Prediction (TPP), a simple prediction head to predict entity mentions as token sequences within documents. Alternative to token classification, TPP models the document layout as a complete directed graph of tokens, and predicts token paths within the graph as entities. For better evaluation of VrD-NER systems, we also propose two revised benchmark datasets of NER on scanned documents which can reflect real-world scenarios. Experiment results demonstrate the effectiveness of our method, and suggest its potential to be a universal solution to various information extraction tasks on documents.

{{</citation>}}


### (48/144) Correction Focused Language Model Training for Speech Recognition (Yingyi Ma et al., 2023)

{{<citation>}}

Yingyi Ma, Zhe Liu, Ozlem Kalinli. (2023)  
**Correction Focused Language Model Training for Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: Language Model, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.11003v1)  

---


**ABSTRACT**  
Language models (LMs) have been commonly adopted to boost the performance of automatic speech recognition (ASR) particularly in domain adaptation tasks. Conventional way of LM training treats all the words in corpora equally, resulting in suboptimal improvements in ASR performance. In this work, we introduce a novel correction focused LM training approach which aims to prioritize ASR fallible words. The word-level ASR fallibility score, representing the likelihood of ASR mis-recognition, is defined and shaped as a prior word distribution to guide the LM training. To enable correction focused training with text-only corpora, large language models (LLMs) are employed as fallibility score predictors and text generators through multi-task fine-tuning. Experimental results for domain adaptation tasks demonstrate the effectiveness of our proposed method. Compared with conventional LMs, correction focused training achieves up to relatively 5.5% word error rate (WER) reduction in sufficient text scenarios. In insufficient text scenarios, LM training with LLM-generated text achieves up to relatively 13% WER reduction, while correction focused training further obtains up to relatively 6% WER reduction.

{{</citation>}}


### (49/144) Instructive Dialogue Summarization with Query Aggregations (Bin Wang et al., 2023)

{{<citation>}}

Bin Wang, Zhengyuan Liu, Nancy F. Chen. (2023)  
**Instructive Dialogue Summarization with Query Aggregations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Summarization  
[Paper Link](http://arxiv.org/abs/2310.10981v1)  

---


**ABSTRACT**  
Conventional dialogue summarization methods directly generate summaries and do not consider user's specific interests. This poses challenges in cases where the users are more focused on particular topics or aspects. With the advancement of instruction-finetuned language models, we introduce instruction-tuning to dialogues to expand the capability set of dialogue summarization models. To overcome the scarcity of instructive dialogue summarization data, we propose a three-step approach to synthesize high-quality query-based summarization triples. This process involves summary-anchored query generation, query filtering, and query-based summary generation. By training a unified model called InstructDS (Instructive Dialogue Summarization) on three summarization datasets with multi-purpose instructive triples, we expand the capability of dialogue summarization models. We evaluate our method on four datasets, including dialogue summarization and dialogue reading comprehension. Experimental results show that our approach outperforms the state-of-the-art models and even models with larger sizes. Additionally, our model exhibits higher generalizability and faithfulness, as confirmed by human subjective evaluations.

{{</citation>}}


### (50/144) EXMODD: An EXplanatory Multimodal Open-Domain Dialogue dataset (Hang Yin et al., 2023)

{{<citation>}}

Hang Yin, Pinren Lu, Ziang Li, Bin Sun, Kan Li. (2023)  
**EXMODD: An EXplanatory Multimodal Open-Domain Dialogue dataset**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2310.10967v1)  

---


**ABSTRACT**  
The need for high-quality data has been a key issue hindering the research of dialogue tasks. Recent studies try to build datasets through manual, web crawling, and large pre-trained models. However, man-made data is expensive and data collected from the internet often includes generic responses, meaningless statements, and toxic dialogues. Automatic data generation through large models is a cost-effective method, but for open-domain multimodal dialogue tasks, there are still three drawbacks: 1) There is currently no open-source large model that can accept multimodal input; 2) The content generated by the model lacks interpretability; 3) The generated data is usually difficult to quality control and require extensive resource to collect. To alleviate the significant human and resource expenditure in data collection, we propose a Multimodal Data Construction Framework (MDCF). MDCF designs proper prompts to spur the large-scale pre-trained language model to generate well-formed and satisfactory content. Additionally, MDCF also automatically provides explanation for a given image and its corresponding dialogue, which can provide a certain degree of interpretability and facilitate manual follow-up quality inspection. Based on this, we release an Explanatory Multimodal Open-Domain dialogue dataset (EXMODD). Experiments indicate a positive correlation between the model's ability to generate accurate understandings and high-quality responses. Our code and data can be found at https://github.com/poplpr/EXMODD.

{{</citation>}}


### (51/144) Semantic-Aware Contrastive Sentence Representation Learning with Large Language Models (Huiming Wang et al., 2023)

{{<citation>}}

Huiming Wang, Liying Cheng, Zhaodonghui Li, De Wen Soh, Lidong Bing. (2023)  
**Semantic-Aware Contrastive Sentence Representation Learning with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLI, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.10962v1)  

---


**ABSTRACT**  
Contrastive learning has been proven to be effective in learning better sentence representations. However, to train a contrastive learning model, large numbers of labeled sentences are required to construct positive and negative pairs explicitly, such as those in natural language inference (NLI) datasets. Unfortunately, acquiring sufficient high-quality labeled data can be both time-consuming and resource-intensive, leading researchers to focus on developing methods for learning unsupervised sentence representations. As there is no clear relationship between these unstructured randomly-sampled sentences, building positive and negative pairs over them is tricky and problematic. To tackle these challenges, in this paper, we propose SemCSR, a semantic-aware contrastive sentence representation framework. By leveraging the generation and evaluation capabilities of large language models (LLMs), we can automatically construct a high-quality NLI-style corpus without any human annotation, and further incorporate the generated sentence pairs into learning a contrastive sentence representation model. Extensive experiments and comprehensive analyses demonstrate the effectiveness of our proposed framework for learning a better sentence representation with LLMs.

{{</citation>}}


### (52/144) TEQ: Trainable Equivalent Transformation for Quantization of LLMs (Wenhua Cheng et al., 2023)

{{<citation>}}

Wenhua Cheng, Yiyang Cai, Kaokao Lv, Haihao Shen. (2023)  
**TEQ: Trainable Equivalent Transformation for Quantization of LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2310.10944v1)  

---


**ABSTRACT**  
As large language models (LLMs) become more prevalent, there is a growing need for new and improved quantization methods that can meet the computationalast layer demands of these modern architectures while maintaining the accuracy. In this paper, we present TEQ, a trainable equivalent transformation that preserves the FP32 precision of the model output while taking advantage of low-precision quantization, especially 3 and 4 bits weight-only quantization. The training process is lightweight, requiring only 1K steps and fewer than 0.1 percent of the original model's trainable parameters. Furthermore, the transformation does not add any computational overhead during inference. Our results are on-par with the state-of-the-art (SOTA) methods on typical LLMs. Our approach can be combined with other methods to achieve even better performance. The code is available at https://github.com/intel/neural-compressor.

{{</citation>}}


### (53/144) MASON-NLP at eRisk 2023: Deep Learning-Based Detection of Depression Symptoms from Social Media Texts (Fardin Ahsan Sakib et al., 2023)

{{<citation>}}

Fardin Ahsan Sakib, Ahnaf Atef Choudhury, Ozlem Uzuner. (2023)  
**MASON-NLP at eRisk 2023: Deep Learning-Based Detection of Depression Symptoms from Social Media Texts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, LSTM, NLP, Social Media  
[Paper Link](http://arxiv.org/abs/2310.10941v1)  

---


**ABSTRACT**  
Depression is a mental health disorder that has a profound impact on people's lives. Recent research suggests that signs of depression can be detected in the way individuals communicate, both through spoken words and written texts. In particular, social media posts are a rich and convenient text source that we may examine for depressive symptoms. The Beck Depression Inventory (BDI) Questionnaire, which is frequently used to gauge the severity of depression, is one instrument that can aid in this study. We can narrow our study to only those symptoms since each BDI question is linked to a particular depressive symptom. It's important to remember that not everyone with depression exhibits all symptoms at once, but rather a combination of them. Therefore, it is extremely useful to be able to determine if a sentence or a piece of user-generated content is pertinent to a certain condition. With this in mind, the eRisk 2023 Task 1 was designed to do exactly that: assess the relevance of different sentences to the symptoms of depression as outlined in the BDI questionnaire. This report is all about how our team, Mason-NLP, participated in this subtask, which involved identifying sentences related to different depression symptoms. We used a deep learning approach that incorporated MentalBERT, RoBERTa, and LSTM. Despite our efforts, the evaluation results were lower than expected, underscoring the challenges inherent in ranking sentences from an extensive dataset about depression, which necessitates both appropriate methodological choices and significant computational resources. We anticipate that future iterations of this shared task will yield improved results as our understanding and techniques evolve.

{{</citation>}}


### (54/144) Intent Detection and Slot Filling for Home Assistants: Dataset and Analysis for Bangla and Sylheti (Fardin Ahsan Sakib et al., 2023)

{{<citation>}}

Fardin Ahsan Sakib, A H M Rezaul Karim, Saadat Hasan Khan, Md Mushfiqur Rahman. (2023)  
**Intent Detection and Slot Filling for Home Assistants: Dataset and Analysis for Bangla and Sylheti**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, Intent Detection  
[Paper Link](http://arxiv.org/abs/2310.10935v1)  

---


**ABSTRACT**  
As voice assistants cement their place in our technologically advanced society, there remains a need to cater to the diverse linguistic landscape, including colloquial forms of low-resource languages. Our study introduces the first-ever comprehensive dataset for intent detection and slot filling in formal Bangla, colloquial Bangla, and Sylheti languages, totaling 984 samples across 10 unique intents. Our analysis reveals the robustness of large language models for tackling downstream tasks with inadequate data. The GPT-3.5 model achieves an impressive F1 score of 0.94 in intent detection and 0.51 in slot filling for colloquial Bangla.

{{</citation>}}


### (55/144) Enhanced Transformer Architecture for Natural Language Processing (Woohyeon Moon et al., 2023)

{{<citation>}}

Woohyeon Moon, Taeyoung Kim, Bumgeun Park, Dongsoo Har. (2023)  
**Enhanced Transformer Architecture for Natural Language Processing**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLEU, NLP, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2310.10930v1)  

---


**ABSTRACT**  
Transformer is a state-of-the-art model in the field of natural language processing (NLP). Current NLP models primarily increase the number of transformers to improve processing performance. However, this technique requires a lot of training resources such as computing capacity. In this paper, a novel structure of Transformer is proposed. It is featured by full layer normalization, weighted residual connection, positional encoding exploiting reinforcement learning, and zero masked self-attention. The proposed Transformer model, which is called Enhanced Transformer, is validated by the bilingual evaluation understudy (BLEU) score obtained with the Multi30k translation dataset. As a result, the Enhanced Transformer achieves 202.96% higher BLEU score as compared to the original transformer with the translation dataset.

{{</citation>}}


### (56/144) Spatial HuBERT: Self-supervised Spatial Speech Representation Learning for a Single Talker from Multi-channel Audio (Antoni Dimitriadis et al., 2023)

{{<citation>}}

Antoni Dimitriadis, Siqi Pan, Vidhyasaharan Sethu, Beena Ahmed. (2023)  
**Spatial HuBERT: Self-supervised Spatial Speech Representation Learning for a Single Talker from Multi-channel Audio**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: BERT, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.10922v1)  

---


**ABSTRACT**  
Self-supervised learning has been used to leverage unlabelled data, improving accuracy and generalisation of speech systems through the training of representation models. While many recent works have sought to produce effective representations across a variety of acoustic domains, languages, modalities and even simultaneous speakers, these studies have all been limited to single-channel audio recordings. This paper presents Spatial HuBERT, a self-supervised speech representation model that learns both acoustic and spatial information pertaining to a single speaker in a potentially noisy environment by using multi-channel audio inputs. Spatial HuBERT learns representations that outperform state-of-the-art single-channel speech representations on a variety of spatial downstream tasks, particularly in reverberant and noisy environments. We also demonstrate the utility of the representations learned by Spatial HuBERT on a speech localisation downstream task. Along with this paper, we publicly release a new dataset of 100 000 simulated first-order ambisonics room impulse responses.

{{</citation>}}


### (57/144) NuclearQA: A Human-Made Benchmark for Language Models for the Nuclear Domain (Anurag Acharya et al., 2023)

{{<citation>}}

Anurag Acharya, Sai Munikoti, Aaron Hellinger, Sara Smith, Sridevi Wagle, Sameera Horawalavithana. (2023)  
**NuclearQA: A Human-Made Benchmark for Language Models for the Nuclear Domain**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.10920v1)  

---


**ABSTRACT**  
As LLMs have become increasingly popular, they have been used in almost every field. But as the application for LLMs expands from generic fields to narrow, focused science domains, there exists an ever-increasing gap in ways to evaluate their efficacy in those fields. For the benchmarks that do exist, a lot of them focus on questions that don't require proper understanding of the subject in question. In this paper, we present NuclearQA, a human-made benchmark of 100 questions to evaluate language models in the nuclear domain, consisting of a varying collection of questions that have been specifically designed by experts to test the abilities of language models. We detail our approach and show how the mix of several types of questions makes our benchmark uniquely capable of evaluating models in the nuclear domain. We also present our own evaluation metric for assessing LLM's performances due to the limitations of existing ones. Our experiments on state-of-the-art models suggest that even the best LLMs perform less than satisfactorily on our benchmark, demonstrating the scientific knowledge gap of existing LLMs.

{{</citation>}}


### (58/144) Emergent AI-Assisted Discourse: Case Study of a Second Language Writer Authoring with ChatGPT (Sharin Jacob et al., 2023)

{{<citation>}}

Sharin Jacob, Tamara Tate, Mark Warschauer. (2023)  
**Emergent AI-Assisted Discourse: Case Study of a Second Language Writer Authoring with ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.10903v1)  

---


**ABSTRACT**  
The rapid proliferation of ChatGPT has incited debates regarding its impact on human writing. Amid concerns about declining writing standards, this study investigates the role of ChatGPT in facilitating academic writing, especially among language learners. Using a case study approach, this study examines the experiences of Kailing, a doctoral student, who integrates ChatGPT throughout their academic writing process. The study employs activity theory as a lens for understanding writing with generative AI tools and data analyzed includes semi-structured interviews, writing samples, and GPT logs. Results indicate that Kailing effectively collaborates with ChatGPT across various writing stages while preserving her distinct authorial voice and agency. This underscores the potential of AI tools such as ChatGPT to enhance academic writing for language learners without overshadowing individual authenticity. This case study offers a critical exploration of how ChatGPT is utilized in the academic writing process and the preservation of a student's authentic voice when engaging with the tool.

{{</citation>}}


## cs.RO (4)



### (59/144) Classification of Safety Driver Attention During Autonomous Vehicle Operation (Santiago Gerling Konrad et al., 2023)

{{<citation>}}

Santiago Gerling Konrad, Julie Stephany Berrio, Mao Shan, Favio Masson, Stewart Worrall. (2023)  
**Classification of Safety Driver Attention During Autonomous Vehicle Operation**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-HC, cs-RO, cs.RO  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.11608v1)  

---


**ABSTRACT**  
Despite the continual advances in Advanced Driver Assistance Systems (ADAS) and the development of high-level autonomous vehicles (AV), there is a general consensus that for the short to medium term, there is a requirement for a human supervisor to handle the edge cases that inevitably arise. Given this requirement, it is essential that the state of the vehicle operator is monitored to ensure they are contributing to the vehicle's safe operation. This paper introduces a dual-source approach integrating data from an infrared camera facing the vehicle operator and vehicle perception systems to produce a metric for driver alertness in order to promote and ensure safe operator behaviour. The infrared camera detects the driver's head, enabling the calculation of head orientation, which is relevant as the head typically moves according to the individual's focus of attention. By incorporating environmental data from the perception system, it becomes possible to determine whether the vehicle operator observes objects in the surroundings. Experiments were conducted using data collected in Sydney, Australia, simulating AV operations in an urban environment. Our results demonstrate that the proposed system effectively determines a metric for the attention levels of the vehicle operator, enabling interventions such as warnings or reducing autonomous functionality as appropriate. This comprehensive solution shows promise in contributing to ADAS and AVs' overall safety and efficiency in a real-world setting.

{{</citation>}}


### (60/144) Language Models as Zero-Shot Trajectory Generators (Teyun Kwon et al., 2023)

{{<citation>}}

Teyun Kwon, Norman Di Palo, Edward Johns. (2023)  
**Language Models as Zero-Shot Trajectory Generators**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CL, cs-HC, cs-LG, cs-RO, cs.RO  
Keywords: GPT, GPT-4, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.11604v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have recently shown promise as high-level planners for robots when given access to a selection of low-level skills. However, it is often assumed that LLMs do not possess sufficient knowledge to be used for the low-level trajectories themselves. In this work, we address this assumption thoroughly, and investigate if an LLM (GPT-4) can directly predict a dense sequence of end-effector poses for manipulation skills, when given access to only object detection and segmentation vision models. We study how well a single task-agnostic prompt, without any in-context examples, motion primitives, or external trajectory optimisers, can perform across 26 real-world language-based tasks, such as "open the bottle cap" and "wipe the plate with the sponge", and we investigate which design choices in this prompt are the most effective. Our conclusions raise the assumed limit of LLMs for robotics, and we reveal for the first time that LLMs do indeed possess an understanding of low-level robot control sufficient for a range of common tasks, and that they can additionally detect failures and then re-plan trajectories accordingly. Videos, code, and prompts are available at: https://www.robot-learning.uk/language-models-trajectory-generators.

{{</citation>}}


### (61/144) Sim-to-Real Transfer of Adaptive Control Parameters for AUV Stabilization under Current Disturbance (Thomas Chaffre et al., 2023)

{{<citation>}}

Thomas Chaffre, Jonathan Wheare, Andrew Lammas, Paulo Santos, Gilles Le Chenadec, Karl Sammut, Benoit Clement. (2023)  
**Sim-to-Real Transfer of Adaptive Control Parameters for AUV Stabilization under Current Disturbance**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.11075v1)  

---


**ABSTRACT**  
Learning-based adaptive control methods hold the premise of enabling autonomous agents to reduce the effect of process variations with minimal human intervention. However, its application to autonomous underwater vehicles (AUVs) has so far been restricted due to 1) unknown dynamics under the form of sea current disturbance that we can not model properly nor measure due to limited sensor capability and 2) the nonlinearity of AUVs tasks where the controller response at some operating points must be overly conservative in order to satisfy the specification at other operating points. Deep Reinforcement Learning (DRL) can alleviates these limitations by training general-purpose neural network policies, but applications of DRL algorithms to AUVs have been restricted to simulated environments, due to their inherent high sample complexity and distribution shift problem. This paper presents a novel approach, merging the Maximum Entropy Deep Reinforcement Learning framework with a classic model-based control architecture, to formulate an adaptive controller. Within this framework, we introduce a Sim-to-Real transfer strategy comprising the following components: a bio-inspired experience replay mechanism, an enhanced domain randomisation technique, and an evaluation protocol executed on a physical platform. Our experimental assessments demonstrate that this method effectively learns proficient policies from suboptimal simulated models of the AUV, resulting in control performance 3 times higher when transferred to a real-world vehicle, compared to its model-based nonadaptive but optimal counterpart.

{{</citation>}}


### (62/144) Reaching the Limit in Autonomous Racing: Optimal Control versus Reinforcement Learning (Yunlong Song et al., 2023)

{{<citation>}}

Yunlong Song, Angel Romero, Matthias Mueller, Vladlen Koltun, Davide Scaramuzza. (2023)  
**Reaching the Limit in Autonomous Racing: Optimal Control versus Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10943v2)  

---


**ABSTRACT**  
A central question in robotics is how to design a control system for an agile mobile robot. This paper studies this question systematically, focusing on a challenging setting: autonomous drone racing. We show that a neural network controller trained with reinforcement learning (RL) outperformed optimal control (OC) methods in this setting. We then investigated which fundamental factors have contributed to the success of RL or have limited OC. Our study indicates that the fundamental advantage of RL over OC is not that it optimizes its objective better but that it optimizes a better objective. OC decomposes the problem into planning and control with an explicit intermediate representation, such as a trajectory, that serves as an interface. This decomposition limits the range of behaviors that can be expressed by the controller, leading to inferior control performance when facing unmodeled effects. In contrast, RL can directly optimize a task-level objective and can leverage domain randomization to cope with model uncertainty, allowing the discovery of more robust control responses. Our findings allowed us to push an agile drone to its maximum performance, achieving a peak acceleration greater than 12 times the gravitational acceleration and a peak velocity of 108 kilometers per hour. Our policy achieved superhuman control within minutes of training on a standard workstation. This work presents a milestone in agile robotics and sheds light on the role of RL and OC in robot control.

{{</citation>}}


## cs.LG (28)



### (63/144) TK-KNN: A Balanced Distance-Based Pseudo Labeling Approach for Semi-Supervised Intent Classification (Nicholas Botzer et al., 2023)

{{<citation>}}

Nicholas Botzer, David Vasquez, Tim Weninger, Issam Laradji. (2023)  
**TK-KNN: A Balanced Distance-Based Pseudo Labeling Approach for Semi-Supervised Intent Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.11607v1)  

---


**ABSTRACT**  
The ability to detect intent in dialogue systems has become increasingly important in modern technology. These systems often generate a large amount of unlabeled data, and manually labeling this data requires substantial human effort. Semi-supervised methods attempt to remedy this cost by using a model trained on a few labeled examples and then by assigning pseudo-labels to further a subset of unlabeled examples that has a model prediction confidence higher than a certain threshold. However, one particularly perilous consequence of these methods is the risk of picking an imbalanced set of examples across classes, which could lead to poor labels. In the present work, we describe Top-K K-Nearest Neighbor (TK-KNN), which uses a more robust pseudo-labeling approach based on distance in the embedding space while maintaining a balanced set of pseudo-labeled examples across classes through a ranking-based approach. Experiments on several datasets show that TK-KNN outperforms existing models, particularly when labeled data is scarce on popular datasets such as CLINC150 and Banking77. Code is available at https://github.com/ServiceNow/tk-knn

{{</citation>}}


### (64/144) When Rigidity Hurts: Soft Consistency Regularization for Probabilistic Hierarchical Time Series Forecasting (Harshavardhan Kamarthi et al., 2023)

{{<citation>}}

Harshavardhan Kamarthi, Lingkai Kong, Alexander Rodríguez, Chao Zhang, B. Aditya Prakash. (2023)  
**When Rigidity Hurts: Soft Consistency Regularization for Probabilistic Hierarchical Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.11569v2)  

---


**ABSTRACT**  
Probabilistic hierarchical time-series forecasting is an important variant of time-series forecasting, where the goal is to model and forecast multivariate time-series that have underlying hierarchical relations. Most methods focus on point predictions and do not provide well-calibrated probabilistic forecasts distributions. Recent state-of-art probabilistic forecasting methods also impose hierarchical relations on point predictions and samples of distribution which does not account for coherency of forecast distributions. Previous works also silently assume that datasets are always consistent with given hierarchical relations and do not adapt to real-world datasets that show deviation from this assumption. We close both these gap and propose PROFHiT, which is a fully probabilistic hierarchical forecasting model that jointly models forecast distribution of entire hierarchy. PROFHiT uses a flexible probabilistic Bayesian approach and introduces a novel Distributional Coherency regularization to learn from hierarchical relations for entire forecast distribution that enables robust and calibrated forecasts as well as adapt to datasets of varying hierarchical consistency. On evaluating PROFHiT over wide range of datasets, we observed 41-88% better performance in accuracy and significantly better calibration. Due to modeling the coherency over full distribution, we observed that PROFHiT can robustly provide reliable forecasts even if up to 10% of input time-series data is missing where other methods' performance severely degrade by over 70%.

{{</citation>}}


### (65/144) Group Preference Optimization: Few-Shot Alignment of Large Language Models (Siyan Zhao et al., 2023)

{{<citation>}}

Siyan Zhao, John Dang, Aditya Grover. (2023)  
**Group Preference Optimization: Few-Shot Alignment of Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Few-Shot, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11523v1)  

---


**ABSTRACT**  
Many applications of large language models (LLMs), ranging from chatbots to creative writing, require nuanced subjective judgments that can differ significantly across different groups. Existing alignment algorithms can be expensive to align for each group, requiring prohibitive amounts of group-specific preference data and computation for real-world use cases. We introduce Group Preference Optimization (GPO), an alignment framework that steers language models to preferences of individual groups in a few-shot manner. In GPO, we augment the base LLM with an independent transformer module trained to predict the preferences of a group for the LLM generations. For few-shot learning, we parameterize this module as an in-context autoregressive transformer and train it via meta-learning on several groups. We empirically validate the efficacy of GPO through rigorous evaluations using LLMs with varied sizes on three human opinion adaptation tasks. These tasks involve adapting to the preferences of US demographic groups, global countries, and individual users. Our results demonstrate that GPO not only aligns models more accurately but also requires fewer group-specific preferences, and less training and inference computing resources, outperforming existing strategies such as in-context steering and fine-tuning methods.

{{</citation>}}


### (66/144) Value-Biased Maximum Likelihood Estimation for Model-based Reinforcement Learning in Discounted Linear MDPs (Yu-Heng Hung et al., 2023)

{{<citation>}}

Yu-Heng Hung, Ping-Chun Hsieh, Akshay Mete, P. R. Kumar. (2023)  
**Value-Biased Maximum Likelihood Estimation for Model-based Reinforcement Learning in Discounted Linear MDPs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.11515v1)  

---


**ABSTRACT**  
We consider the infinite-horizon linear Markov Decision Processes (MDPs), where the transition probabilities of the dynamic model can be linearly parameterized with the help of a predefined low-dimensional feature mapping. While the existing regression-based approaches have been theoretically shown to achieve nearly-optimal regret, they are computationally rather inefficient due to the need for a large number of optimization runs in each time step, especially when the state and action spaces are large. To address this issue, we propose to solve linear MDPs through the lens of Value-Biased Maximum Likelihood Estimation (VBMLE), which is a classic model-based exploration principle in the adaptive control literature for resolving the well-known closed-loop identification problem of Maximum Likelihood Estimation. We formally show that (i) VBMLE enjoys $\widetilde{O}(d\sqrt{T})$ regret, where $T$ is the time horizon and $d$ is the dimension of the model parameter, and (ii) VBMLE is computationally more efficient as it only requires solving one optimization problem in each time step. In our regret analysis, we offer a generic convergence result of MLE in linear MDPs through a novel supermartingale construct and uncover an interesting connection between linear MDPs and online learning, which could be of independent interest. Finally, the simulation results show that VBMLE significantly outperforms the benchmark method in terms of both empirical regret and computation time.

{{</citation>}}


### (67/144) Elucidating The Design Space of Classifier-Guided Diffusion Generation (Jiajun Ma et al., 2023)

{{<citation>}}

Jiajun Ma, Tianyang Hu, Wenjia Wang, Jiacheng Sun. (2023)  
**Elucidating The Design Space of Classifier-Guided Diffusion Generation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.11311v1)  

---


**ABSTRACT**  
Guidance in conditional diffusion generation is of great importance for sample quality and controllability. However, existing guidance schemes are to be desired. On one hand, mainstream methods such as classifier guidance and classifier-free guidance both require extra training with labeled data, which is time-consuming and unable to adapt to new conditions. On the other hand, training-free methods such as universal guidance, though more flexible, have yet to demonstrate comparable performance. In this work, through a comprehensive investigation into the design space, we show that it is possible to achieve significant performance improvements over existing guidance schemes by leveraging off-the-shelf classifiers in a training-free fashion, enjoying the best of both worlds. Employing calibration as a general guideline, we propose several pre-conditioning techniques to better exploit pretrained off-the-shelf classifiers for guiding diffusion generation. Extensive experiments on ImageNet validate our proposed method, showing that state-of-the-art diffusion models (DDPM, EDM, DiT) can be further improved (up to 20%) using off-the-shelf classifiers with barely any extra computational cost. With the proliferation of publicly available pretrained classifiers, our proposed approach has great potential and can be readily scaled up to text-to-image generation tasks. The code is available at https://github.com/AlexMaOLS/EluCD/tree/main.

{{</citation>}}


### (68/144) Evaluating the Impact of Humanitarian Aid on Food Security (Jordi Cerdà-Bautista et al., 2023)

{{<citation>}}

Jordi Cerdà-Bautista, José María Tárraga, Vasileios Sitokonstantinou, Gustau Camps-Valls. (2023)  
**Evaluating the Impact of Humanitarian Aid on Food Security**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.11287v1)  

---


**ABSTRACT**  
In the face of climate change-induced droughts, vulnerable regions encounter severe threats to food security, demanding urgent humanitarian assistance. This paper introduces a causal inference framework for the Horn of Africa, aiming to assess the impact of cash-based interventions on food crises. Our contributions encompass identifying causal relationships within the food security system, harmonizing a comprehensive database, and estimating the causal effect of humanitarian interventions on malnutrition. Our results revealed no significant effects, likely due to limited sample size, suboptimal data quality, and an imperfect causal graph resulting from our limited understanding of multidisciplinary systems like food security. This underscores the need to enhance data collection and refine causal models with domain experts for more effective future interventions and policies, improving transparency and accountability in humanitarian aid.

{{</citation>}}


### (69/144) Self-supervision meets kernel graph neural models: From architecture to augmentations (Jiawang Dan et al., 2023)

{{<citation>}}

Jiawang Dan, Ruofan Wu, Yunpeng Liu, Baokun Wang, Changhua Meng, Tengfei Liu, Tianyi Zhang, Ningtao Wang, Xing Fu, Qi Li, Weiqiang Wang. (2023)  
**Self-supervision meets kernel graph neural models: From architecture to augmentations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.11281v1)  

---


**ABSTRACT**  
Graph representation learning has now become the de facto standard when handling graph-structured data, with the framework of message-passing graph neural networks (MPNN) being the most prevailing algorithmic tool. Despite its popularity, the family of MPNNs suffers from several drawbacks such as transparency and expressivity. Recently, the idea of designing neural models on graphs using the theory of graph kernels has emerged as a more transparent as well as sometimes more expressive alternative to MPNNs known as kernel graph neural networks (KGNNs). Developments on KGNNs are currently a nascent field of research, leaving several challenges from algorithmic design and adaptation to other learning paradigms such as self-supervised learning. In this paper, we improve the design and learning of KGNNs. Firstly, we extend the algorithmic formulation of KGNNs by allowing a more flexible graph-level similarity definition that encompasses former proposals like random walk graph kernel, as well as providing a smoother optimization objective that alleviates the need of introducing combinatorial learning procedures. Secondly, we enhance KGNNs through the lens of self-supervision via developing a novel structure-preserving graph data augmentation method called latent graph augmentation (LGA). Finally, we perform extensive empirical evaluations to demonstrate the efficacy of our proposed mechanisms. Experimental results over benchmark datasets suggest that our proposed model achieves competitive performance that is comparable to or sometimes outperforming state-of-the-art graph representation learning frameworks with or without self-supervision on graph classification tasks. Comparisons against other previously established graph data augmentation methods verify that the proposed LGA augmentation scheme captures better semantics of graph-level invariance.

{{</citation>}}


### (70/144) CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion (Yangruibo Ding et al., 2023)

{{<citation>}}

Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang. (2023)  
**CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs-SE, cs.LG  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2310.11248v1)  

---


**ABSTRACT**  
Code completion models have made significant progress in recent years, yet current popular evaluation datasets, such as HumanEval and MBPP, predominantly focus on code completion tasks within a single file. This over-simplified setting falls short of representing the real-world software development scenario where repositories span multiple files with numerous cross-file dependencies, and accessing and understanding cross-file context is often required to complete the code correctly.   To fill in this gap, we propose CrossCodeEval, a diverse and multilingual code completion benchmark that necessitates an in-depth cross-file contextual understanding to complete the code accurately. CrossCodeEval is built on a diverse set of real-world, open-sourced, permissively-licensed repositories in four popular programming languages: Python, Java, TypeScript, and C#. To create examples that strictly require cross-file context for accurate completion, we propose a straightforward yet efficient static-analysis-based approach to pinpoint the use of cross-file context within the current file.   Extensive experiments on state-of-the-art code language models like CodeGen and StarCoder demonstrate that CrossCodeEval is extremely challenging when the relevant cross-file context is absent, and we see clear improvements when adding these context into the prompt. However, despite such improvements, the pinnacle of performance remains notably unattained even with the highest-performing model, indicating that CrossCodeEval is also capable of assessing model's capability in leveraging extensive context to make better code completion. Finally, we benchmarked various methods in retrieving cross-file context, and show that CrossCodeEval can also be used to measure the capability of code retrievers.

{{</citation>}}


### (71/144) Efficiently Visualizing Large Graphs (Xinyu Li et al., 2023)

{{<citation>}}

Xinyu Li, Yao Xiao, Yuchen Zhou. (2023)  
**Efficiently Visualizing Large Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.11186v1)  

---


**ABSTRACT**  
Most existing graph visualization methods based on dimension reduction are limited to relatively small graphs due to performance issues. In this work, we propose a novel dimension reduction method for graph visualization, called t-Distributed Stochastic Graph Neighbor Embedding (t-SGNE). t-SGNE is specifically designed to visualize cluster structures in the graph. As a variant of the standard t-SNE method, t-SGNE avoids the time-consuming computations of pairwise similarity. Instead, it uses the neighbor structures of the graph to reduce the time complexity from quadratic to linear, thus supporting larger graphs. In addition, to suit t-SGNE, we combined Laplacian Eigenmaps with the shortest path algorithm in graphs to form the graph embedding algorithm ShortestPath Laplacian Eigenmaps Embedding (SPLEE). Performing SPLEE to obtain a high-dimensional embedding of the large-scale graph and then using t-SGNE to reduce its dimension for visualization, we are able to visualize graphs with up to 300K nodes and 1M edges within 5 minutes and achieve approximately 10% improvement in visualization quality. Codes and data are available at https://github.com/Charlie-XIAO/embedding-visualization-test.

{{</citation>}}


### (72/144) MST-GAT: A Multimodal Spatial-Temporal Graph Attention Network for Time Series Anomaly Detection (Chaoyue Ding et al., 2023)

{{<citation>}}

Chaoyue Ding, Shiliang Sun, Jing Zhao. (2023)  
**MST-GAT: A Multimodal Spatial-Temporal Graph Attention Network for Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: I-5-4, cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Attention, Graph Attention Network, Time Series  
[Paper Link](http://arxiv.org/abs/2310.11169v1)  

---


**ABSTRACT**  
Multimodal time series (MTS) anomaly detection is crucial for maintaining the safety and stability of working devices (e.g., water treatment system and spacecraft), whose data are characterized by multivariate time series with diverse modalities. Although recent deep learning methods show great potential in anomaly detection, they do not explicitly capture spatial-temporal relationships between univariate time series of different modalities, resulting in more false negatives and false positives. In this paper, we propose a multimodal spatial-temporal graph attention network (MST-GAT) to tackle this problem. MST-GAT first employs a multimodal graph attention network (M-GAT) and a temporal convolution network to capture the spatial-temporal correlation in multimodal time series. Specifically, M-GAT uses a multi-head attention module and two relational attention modules (i.e., intra- and inter-modal attention) to model modal correlations explicitly. Furthermore, MST-GAT optimizes the reconstruction and prediction modules simultaneously. Experimental results on four multimodal benchmarks demonstrate that MST-GAT outperforms the state-of-the-art baselines. Further analysis indicates that MST-GAT strengthens the interpretability of detected anomalies by locating the most anomalous univariate time series.

{{</citation>}}


### (73/144) FROST: Towards Energy-efficient AI-on-5G Platforms -- A GPU Power Capping Evaluation (Ioannis Mavromatis et al., 2023)

{{<citation>}}

Ioannis Mavromatis, Stefano De Feo, Pietro Carnelli, Robert J. Piechocki, Aftab Khan. (2023)  
**FROST: Towards Energy-efficient AI-on-5G Platforms -- A GPU Power Capping Evaluation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11131v1)  

---


**ABSTRACT**  
The Open Radio Access Network (O-RAN) is a burgeoning market with projected growth in the upcoming years. RAN has the highest CAPEX impact on the network and, most importantly, consumes 73% of its total energy. That makes it an ideal target for optimisation through the integration of Machine Learning (ML). However, the energy consumption of ML is frequently overlooked in such ecosystems. Our work addresses this critical aspect by presenting FROST - Flexible Reconfiguration method with Online System Tuning - a solution for energy-aware ML pipelines that adhere to O-RAN's specifications and principles. FROST is capable of profiling the energy consumption of an ML pipeline and optimising the hardware accordingly, thereby limiting the power draw. Our findings indicate that FROST can achieve energy savings of up to 26.4% without compromising the model's accuracy or introducing significant time delays.

{{</citation>}}


### (74/144) On the Temperature of Bayesian Graph Neural Networks for Conformal Prediction (Seohyeon Cha et al., 2023)

{{<citation>}}

Seohyeon Cha, Honggu Kang, Joonhyuk Kang. (2023)  
**On the Temperature of Bayesian Graph Neural Networks for Conformal Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.11479v1)  

---


**ABSTRACT**  
Accurate uncertainty quantification in graph neural networks (GNNs) is essential, especially in high-stakes domains where GNNs are frequently employed. Conformal prediction (CP) offers a promising framework for quantifying uncertainty by providing $\textit{valid}$ prediction sets for any black-box model. CP ensures formal probabilistic guarantees that a prediction set contains a true label with a desired probability. However, the size of prediction sets, known as $\textit{inefficiency}$, is influenced by the underlying model and data generating process. On the other hand, Bayesian learning also provides a credible region based on the estimated posterior distribution, but this region is $\textit{well-calibrated}$ only when the model is correctly specified. Building on a recent work that introduced a scaling parameter for constructing valid credible regions from posterior estimate, our study explores the advantages of incorporating a temperature parameter into Bayesian GNNs within CP framework. We empirically demonstrate the existence of temperatures that result in more efficient prediction sets. Furthermore, we conduct an analysis to identify the factors contributing to inefficiency and offer valuable insights into the relationship between CP performance and model calibration.

{{</citation>}}


### (75/144) ASP: Automatic Selection of Proxy dataset for efficient AutoML (Peng Yao et al., 2023)

{{<citation>}}

Peng Yao, Chao Liao, Jiyuan Jia, Jianchao Tan, Bin Chen, Chengru Song, Di Zhang. (2023)  
**ASP: Automatic Selection of Proxy dataset for efficient AutoML**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.11478v1)  

---


**ABSTRACT**  
Deep neural networks have gained great success due to the increasing amounts of data, and diverse effective neural network designs. However, it also brings a heavy computing burden as the amount of training data is proportional to the training time. In addition, a well-behaved model requires repeated trials of different structure designs and hyper-parameters, which may take a large amount of time even with state-of-the-art (SOTA) hyper-parameter optimization (HPO) algorithms and neural architecture search (NAS) algorithms. In this paper, we propose an Automatic Selection of Proxy dataset framework (ASP) aimed to dynamically find the informative proxy subsets of training data at each epoch, reducing the training data size as well as saving the AutoML processing time. We verify the effectiveness and generalization of ASP on CIFAR10, CIFAR100, ImageNet16-120, and ImageNet-1k, across various public model benchmarks. The experiment results show that ASP can obtain better results than other data selection methods at all selection ratios. ASP can also enable much more efficient AutoML processing with a speedup of 2x-20x while obtaining better architectures and better hyper-parameters compared to utilizing the entire dataset.

{{</citation>}}


### (76/144) HGCVAE: Integrating Generative and Contrastive Learning for Heterogeneous Graph Learning (Yulan Hu et al., 2023)

{{<citation>}}

Yulan Hu, Zhirui Yang, Sheng Ouyang, Junchen Wan, Fuzheng Zhang, Zhongyuan Wang, Yong Liu. (2023)  
**HGCVAE: Integrating Generative and Contrastive Learning for Heterogeneous Graph Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.11102v3)  

---


**ABSTRACT**  
Generative self-supervised learning (SSL) has exhibited significant potential and garnered increasing interest in graph learning. In this study, we aim to explore the problem of generative SSL in the context of heterogeneous graph learning (HGL). The previous SSL approaches for heterogeneous graphs have primarily relied on contrastive learning, necessitating the design of complex views to capture heterogeneity. However, existing generative SSL methods have not fully leveraged the capabilities of generative models to address the challenges of HGL. In this paper, we present HGCVAE, a novel contrastive variational graph auto-encoder that liberates HGL from the burden of intricate heterogeneity capturing. Instead of focusing on complicated heterogeneity, HGCVAE harnesses the full potential of generative SSL. HGCVAE innovatively consolidates contrastive learning with generative SSL, introducing several key innovations. Firstly, we employ a progressive mechanism to generate high-quality hard negative samples for contrastive learning, utilizing the power of variational inference. Additionally, we present a dynamic mask strategy to ensure effective and stable learning. Moreover, we propose an enhanced scaled cosine error as the criterion for better attribute reconstruction. As an initial step in combining generative and contrastive SSL, HGCVAE achieves remarkable results compared to various state-of-the-art baselines, confirming its superiority.

{{</citation>}}


### (77/144) Feature Pyramid biLSTM: Using Smartphone Sensors for Transportation Mode Detection (Qinrui Tang et al., 2023)

{{<citation>}}

Qinrui Tang, Hao Cheng. (2023)  
**Feature Pyramid biLSTM: Using Smartphone Sensors for Transportation Mode Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.11087v1)  

---


**ABSTRACT**  
The widespread utilization of smartphones has provided extensive availability to Inertial Measurement Units, providing a wide range of sensory data that can be advantageous for the detection of transportation modes. The objective of this study is to propose a novel end-to-end approach to effectively explore a reduced amount of sensory data collected from a smartphone to achieve accurate mode detection in common daily traveling activities. Our approach, called Feature Pyramid biLSTM (FPbiLSTM), is characterized by its ability to reduce the number of sensors required and processing demands, resulting in a more efficient modeling process without sacrificing the quality of the outcomes than the other current models. FPbiLSTM extends an existing CNN biLSTM model with the Feature Pyramid Network, leveraging the advantages of both shallow layer richness and deeper layer feature resilience for capturing temporal moving patterns in various transportation modes. It exhibits an excellent performance by employing the data collected from only three out of seven sensors, i.e. accelerometers, gyroscopes, and magnetometers, in the 2018 Sussex-Huawei Locomotion (SHL) challenge dataset, attaining a noteworthy accuracy of 95.1% and an F1-score of 94.7% in detecting eight different transportation modes.

{{</citation>}}


### (78/144) CSG: Curriculum Representation Learning for Signed Graph (Zeyu Zhang et al., 2023)

{{<citation>}}

Zeyu Zhang, Jiamou Liu, Kaiqi Zhao, Yifei Wang, Pengqian Han, Xianda Zheng, Qiqi Wang, Zijian Zhang. (2023)  
**CSG: Curriculum Representation Learning for Signed Graph**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.11083v1)  

---


**ABSTRACT**  
Signed graphs are valuable for modeling complex relationships with positive and negative connections, and Signed Graph Neural Networks (SGNNs) have become crucial tools for their analysis. However, prior to our work, no specific training plan existed for SGNNs, and the conventional random sampling approach did not address varying learning difficulties within the graph's structure. We proposed a curriculum-based training approach, where samples progress from easy to complex, inspired by human learning. To measure learning difficulty, we introduced a lightweight mechanism and created the Curriculum representation learning framework for Signed Graphs (CSG). This framework optimizes the order in which samples are presented to the SGNN model. Empirical validation across six real-world datasets showed impressive results, enhancing SGNN model accuracy by up to 23.7% in link sign prediction (AUC) and significantly improving stability with an up to 8.4 reduction in the standard deviation of AUC scores.

{{</citation>}}


### (79/144) Multi-omics Sampling-based Graph Transformer for Synthetic Lethality Prediction (Xusheng Zhao et al., 2023)

{{<citation>}}

Xusheng Zhao, Hao Liu, Qiong Dai, Hao Peng, Xu Bai, Huailiang Peng. (2023)  
**Multi-omics Sampling-based Graph Transformer for Synthetic Lethality Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-QM  
Keywords: GNN, Transformer  
[Paper Link](http://arxiv.org/abs/2310.11082v1)  

---


**ABSTRACT**  
Synthetic lethality (SL) prediction is used to identify if the co-mutation of two genes results in cell death. The prevalent strategy is to abstract SL prediction as an edge classification task on gene nodes within SL data and achieve it through graph neural networks (GNNs). However, GNNs suffer from limitations in their message passing mechanisms, including over-smoothing and over-squashing issues. Moreover, harnessing the information of non-SL gene relationships within large-scale multi-omics data to facilitate SL prediction poses a non-trivial challenge. To tackle these issues, we propose a new multi-omics sampling-based graph transformer for SL prediction (MSGT-SL). Concretely, we introduce a shallow multi-view GNN to acquire local structural patterns from both SL and multi-omics data. Further, we input gene features that encode multi-view information into the standard self-attention to capture long-range dependencies. Notably, starting with batch genes from SL data, we adopt parallel random walk sampling across multiple omics gene graphs encompassing them. Such sampling effectively and modestly incorporates genes from omics in a structure-aware manner before using self-attention. We showcase the effectiveness of MSGT-SL on real-world SL tasks, demonstrating the empirical benefits gained from the graph transformer and multi-omics data.

{{</citation>}}


### (80/144) Understanding Contrastive Learning via Distributionally Robust Optimization (Junkang Wu et al., 2023)

{{<citation>}}

Junkang Wu, Jiawei Chen, Jiancan Wu, Wentao Shi, Xiang Wang, Xiangnan He. (2023)  
**Understanding Contrastive Learning via Distributionally Robust Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.11048v1)  

---


**ABSTRACT**  
This study reveals the inherent tolerance of contrastive learning (CL) towards sampling bias, wherein negative samples may encompass similar semantics (\eg labels). However, existing theories fall short in providing explanations for this phenomenon. We bridge this research gap by analyzing CL through the lens of distributionally robust optimization (DRO), yielding several key insights: (1) CL essentially conducts DRO over the negative sampling distribution, thus enabling robust performance across a variety of potential distributions and demonstrating robustness to sampling bias; (2) The design of the temperature $\tau$ is not merely heuristic but acts as a Lagrange Coefficient, regulating the size of the potential distribution set; (3) A theoretical connection is established between DRO and mutual information, thus presenting fresh evidence for ``InfoNCE as an estimate of MI'' and a new estimation approach for $\phi$-divergence-based generalized mutual information. We also identify CL's potential shortcomings, including over-conservatism and sensitivity to outliers, and introduce a novel Adjusted InfoNCE loss (ADNCE) to mitigate these issues. It refines potential distribution, improving performance and accelerating convergence. Extensive experiments on various domains (image, sentence, and graphs) validate the effectiveness of the proposal. The code is available at \url{https://github.com/junkangwu/ADNCE}.

{{</citation>}}


### (81/144) Fast Graph Condensation with Structure-based Neural Tangent Kernel (Lin Wang et al., 2023)

{{<citation>}}

Lin Wang, Wenqi Fan, Jiatong Li, Yao Ma, Qing Li. (2023)  
**Fast Graph Condensation with Structure-based Neural Tangent Kernel**  

---
Primary Category: cs.LG  
Categories: 68T01, I-2-0, cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.11046v1)  

---


**ABSTRACT**  
The rapid development of Internet technology has given rise to a vast amount of graph-structured data. Graph Neural Networks (GNNs), as an effective method for various graph mining tasks, incurs substantial computational resource costs when dealing with large-scale graph data. A data-centric manner solution is proposed to condense the large graph dataset into a smaller one without sacrificing the predictive performance of GNNs. However, existing efforts condense graph-structured data through a computational intensive bi-level optimization architecture also suffer from massive computation costs. In this paper, we propose reforming the graph condensation problem as a Kernel Ridge Regression (KRR) task instead of iteratively training GNNs in the inner loop of bi-level optimization. More specifically, We propose a novel dataset condensation framework (GC-SNTK) for graph-structured data, where a Structure-based Neural Tangent Kernel (SNTK) is developed to capture the topology of graph and serves as the kernel function in KRR paradigm. Comprehensive experiments demonstrate the effectiveness of our proposed model in accelerating graph condensation while maintaining high prediction performance.

{{</citation>}}


### (82/144) SignGT: Signed Attention-based Graph Transformer for Graph Representation Learning (Jinsong Chen et al., 2023)

{{<citation>}}

Jinsong Chen, Gaichao Li, John E. Hopcroft, Kun He. (2023)  
**SignGT: Signed Attention-based Graph Transformer for Graph Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Attention, GNN, Representation Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.11025v1)  

---


**ABSTRACT**  
The emerging graph Transformers have achieved impressive performance for graph representation learning over graph neural networks (GNNs). In this work, we regard the self-attention mechanism, the core module of graph Transformers, as a two-step aggregation operation on a fully connected graph. Due to the property of generating positive attention values, the self-attention mechanism is equal to conducting a smooth operation on all nodes, preserving the low-frequency information. However, only capturing the low-frequency information is inefficient in learning complex relations of nodes on diverse graphs, such as heterophily graphs where the high-frequency information is crucial. To this end, we propose a Signed Attention-based Graph Transformer (SignGT) to adaptively capture various frequency information from the graphs. Specifically, SignGT develops a new signed self-attention mechanism (SignSA) that produces signed attention values according to the semantic relevance of node pairs. Hence, the diverse frequency information between different node pairs could be carefully preserved. Besides, SignGT proposes a structure-aware feed-forward network (SFFN) that introduces the neighborhood bias to preserve the local topology information. In this way, SignGT could learn informative node representations from both long-range dependencies and local topology information. Extensive empirical results on both node-level and graph-level tasks indicate the superiority of SignGT against state-of-the-art graph Transformers as well as advanced GNNs.

{{</citation>}}


### (83/144) Compatible Transformer for Irregularly Sampled Multivariate Time Series (Yuxi Wei et al., 2023)

{{<citation>}}

Yuxi Wei, Juntong Peng, Tong He, Chenxin Xu, Jian Zhang, Shirui Pan, Siheng Chen. (2023)  
**Compatible Transformer for Irregularly Sampled Multivariate Time Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2310.11022v1)  

---


**ABSTRACT**  
To analyze multivariate time series, most previous methods assume regular subsampling of time series, where the interval between adjacent measurements and the number of samples remain unchanged. Practically, data collection systems could produce irregularly sampled time series due to sensor failures and interventions. However, existing methods designed for regularly sampled multivariate time series cannot directly handle irregularity owing to misalignment along both temporal and variate dimensions. To fill this gap, we propose Compatible Transformer (CoFormer), a transformer-based encoder to achieve comprehensive temporal-interaction feature learning for each individual sample in irregular multivariate time series. In CoFormer, we view each sample as a unique variate-time point and leverage intra-variate/inter-variate attentions to learn sample-wise temporal/interaction features based on intra-variate/inter-variate neighbors. With CoFormer as the core, we can analyze irregularly sampled multivariate time series for many downstream tasks, including classification and prediction. We conduct extensive experiments on 3 real-world datasets and validate that the proposed CoFormer significantly and consistently outperforms existing methods.

{{</citation>}}


### (84/144) Accelerating Scalable Graph Neural Network Inference with Node-Adaptive Propagation (Xinyi Gao et al., 2023)

{{<citation>}}

Xinyi Gao, Wentao Zhang, Junliang Yu, Yingxia Shao, Quoc Viet Hung Nguyen, Bin Cui, Hongzhi Yin. (2023)  
**Accelerating Scalable Graph Neural Network Inference with Node-Adaptive Propagation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.10998v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have exhibited exceptional efficacy in a diverse array of applications. However, the sheer size of large-scale graphs presents a significant challenge to real-time inference with GNNs. Although existing Scalable GNNs leverage linear propagation to preprocess the features and accelerate the training and inference procedure, these methods still suffer from scalability issues when making inferences on unseen nodes, as the feature preprocessing requires the graph to be known and fixed. To further accelerate Scalable GNNs inference in this inductive setting, we propose an online propagation framework and two novel node-adaptive propagation methods that can customize the optimal propagation depth for each node based on its topological information and thereby avoid redundant feature propagation. The trade-off between accuracy and latency can be flexibly managed through simple hyper-parameters to accommodate various latency constraints. Moreover, to compensate for the inference accuracy loss caused by the potential early termination of propagation, we further propose Inception Distillation to exploit the multi-scale receptive field information within graphs. The rigorous and comprehensive experimental study on public datasets with varying scales and characteristics demonstrates that the proposed inference acceleration framework outperforms existing state-of-the-art graph inference acceleration methods in terms of accuracy and efficiency. Particularly, the superiority of our approach is notable on datasets with larger scales, yielding a 75x inference speedup on the largest Ogbn-products dataset.

{{</citation>}}


### (85/144) Context-Aware Meta-Learning (Christopher Fifty et al., 2023)

{{<citation>}}

Christopher Fifty, Dennis Duan, Ronald G. Junkins, Ehsan Amid, Jure Leskovec, Christopher Ré, Sebastian Thrun. (2023)  
**Context-Aware Meta-Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.10971v1)  

---


**ABSTRACT**  
Large Language Models like ChatGPT demonstrate a remarkable capacity to learn new concepts during inference without any fine-tuning. However, visual models trained to detect new objects during inference have been unable to replicate this ability, and instead either perform poorly or require meta-training and/or fine-tuning on similar objects. In this work, we propose a meta-learning algorithm that emulates Large Language Models by learning new visual concepts during inference without fine-tuning. Our approach leverages a frozen pre-trained feature extractor, and analogous to in-context learning, recasts meta-learning as sequence modeling over datapoints with known labels and a test datapoint with an unknown label. On 8 out of 11 meta-learning benchmarks, our approach -- without meta-training or fine-tuning -- exceeds or matches the state-of-the-art algorithm, P>M>F, which is meta-trained on these benchmarks.

{{</citation>}}


### (86/144) A Local Graph Limits Perspective on Sampling-Based GNNs (Yeganeh Alimohammadi et al., 2023)

{{<citation>}}

Yeganeh Alimohammadi, Luana Ruiz, Amin Saberi. (2023)  
**A Local Graph Limits Perspective on Sampling-Based GNNs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.10953v1)  

---


**ABSTRACT**  
We propose a theoretical framework for training Graph Neural Networks (GNNs) on large input graphs via training on small, fixed-size sampled subgraphs. This framework is applicable to a wide range of models, including popular sampling-based GNNs, such as GraphSAGE and FastGCN. Leveraging the theory of graph local limits, we prove that, under mild assumptions, parameters learned from training sampling-based GNNs on small samples of a large input graph are within an $\epsilon$-neighborhood of the outcome of training the same architecture on the whole graph. We derive bounds on the number of samples, the size of the graph, and the training steps required as a function of $\epsilon$. Our results give a novel theoretical understanding for using sampling in training GNNs. They also suggest that by training GNNs on small samples of the input graph, practitioners can identify and select the best models, hyperparameters, and sampling algorithms more efficiently. We empirically illustrate our results on a node classification task on large citation graphs, observing that sampling-based GNNs trained on local subgraphs 12$\times$ smaller than the original graph achieve comparable performance to those trained on the input graph.

{{</citation>}}


### (87/144) Combat Urban Congestion via Collaboration: Heterogeneous GNN-based MARL for Coordinated Platooning and Traffic Signal Control (Xianyue Peng et al., 2023)

{{<citation>}}

Xianyue Peng, Hang Gao, Hao Wang, H. Michael Zhang. (2023)  
**Combat Urban Congestion via Collaboration: Heterogeneous GNN-based MARL for Coordinated Platooning and Traffic Signal Control**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.10948v1)  

---


**ABSTRACT**  
Over the years, reinforcement learning has emerged as a popular approach to develop signal control and vehicle platooning strategies either independently or in a hierarchical way. However, jointly controlling both in real-time to alleviate traffic congestion presents new challenges, such as the inherent physical and behavioral heterogeneity between signal control and platooning, as well as coordination between them. This paper proposes an innovative solution to tackle these challenges based on heterogeneous graph multi-agent reinforcement learning and traffic theories. Our approach involves: 1) designing platoon and signal control as distinct reinforcement learning agents with their own set of observations, actions, and reward functions to optimize traffic flow; 2) designing coordination by incorporating graph neural networks within multi-agent reinforcement learning to facilitate seamless information exchange among agents on a regional scale. We evaluate our approach through SUMO simulation, which shows a convergent result in terms of various transportation metrics and better performance over sole signal or platooning control.

{{</citation>}}


### (88/144) Heterogenous Memory Augmented Neural Networks (Zihan Qiu et al., 2023)

{{<citation>}}

Zihan Qiu, Zhen Liu, Shuicheng Yan, Shanghang Zhang, Jie Fu. (2023)  
**Heterogenous Memory Augmented Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Transformer  
[Paper Link](http://arxiv.org/abs/2310.10909v1)  

---


**ABSTRACT**  
It has been shown that semi-parametric methods, which combine standard neural networks with non-parametric components such as external memory modules and data retrieval, are particularly helpful in data scarcity and out-of-distribution (OOD) scenarios. However, existing semi-parametric methods mostly depend on independent raw data points - this strategy is difficult to scale up due to both high computational costs and the incapacity of current attention mechanisms with a large number of tokens. In this paper, we introduce a novel heterogeneous memory augmentation approach for neural networks which, by introducing learnable memory tokens with attention mechanism, can effectively boost performance without huge computational overhead. Our general-purpose method can be seamlessly combined with various backbones (MLP, CNN, GNN, and Transformer) in a plug-and-play manner. We extensively evaluate our approach on various image and graph-based tasks under both in-distribution (ID) and OOD conditions and show its competitive performance against task-specific state-of-the-art methods. Code is available at \url{https://github.com/qiuzh20/HMA}.

{{</citation>}}


### (89/144) Emergent Mixture-of-Experts: Can Dense Pre-trained Transformers Benefit from Emergent Modular Structures? (Zihan Qiu et al., 2023)

{{<citation>}}

Zihan Qiu, Zeyu Huang, Jie Fu. (2023)  
**Emergent Mixture-of-Experts: Can Dense Pre-trained Transformers Benefit from Emergent Modular Structures?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.10908v1)  

---


**ABSTRACT**  
Incorporating modular designs into neural networks demonstrates superior out-of-generalization, learning efficiency, etc. Existing modular neural networks are generally $\textit{explicit}$ because their modular architectures are pre-defined, and individual modules are expected to implement distinct functions. Conversely, recent works reveal that there exist $\textit{implicit}$ modular structures in standard pre-trained transformers, namely $\textit{Emergent Modularity}$. They indicate that such modular structures exhibit during the early pre-training phase and are totally spontaneous. However, most transformers are still treated as monolithic models with their modular natures underutilized. Therefore, given the excellent properties of explicit modular architecture, we explore $\textit{whether and how dense pre-trained transformers can benefit from emergent modular structures.}$ To study this question, we construct \textbf{E}mergent $\textbf{M}$ixture-$\textbf{o}$f-$\textbf{E}$xperts (EMoE). Without introducing additional parameters, EMoE can be seen as the modular counterpart of the original model and can be effortlessly incorporated into downstream tuning. Extensive experiments (we tune 1785 models) on various downstream tasks (vision and language) and models (22M to1.5B) demonstrate that EMoE effectively boosts in-domain and out-of-domain generalization abilities. Further analysis and ablation study suggest that EMoE mitigates negative knowledge transfer and is robust to various configurations. Code is available at \url{https://github.com/qiuzh20/EMoE}

{{</citation>}}


### (90/144) Instilling Inductive Biases with Subnetworks (Enyan Zhang et al., 2023)

{{<citation>}}

Enyan Zhang, Michael A. Lepori, Ellie Pavlick. (2023)  
**Instilling Inductive Biases with Subnetworks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.10899v1)  

---


**ABSTRACT**  
Despite the recent success of artificial neural networks on a variety of tasks, we have little knowledge or control over the exact solutions these models implement. Instilling inductive biases -- preferences for some solutions over others -- into these models is one promising path toward understanding and controlling their behavior. Much work has been done to study the inherent inductive biases of models and instill different inductive biases through hand-designed architectures or carefully curated training regimens. In this work, we explore a more mechanistic approach: Subtask Induction. Our method discovers a functional subnetwork that implements a particular subtask within a trained model and uses it to instill inductive biases towards solutions utilizing that subtask. Subtask Induction is flexible and efficient, and we demonstrate its effectiveness with two experiments. First, we show that Subtask Induction significantly reduces the amount of training data required for a model to adopt a specific, generalizable solution to a modular arithmetic task. Second, we demonstrate that Subtask Induction successfully induces a human-like shape bias while increasing data efficiency for convolutional and transformer-based image classification models.

{{</citation>}}


## cs.CV (15)



### (91/144) DIAR: Deep Image Alignment and Reconstruction using Swin Transformers (Monika Kwiatkowski et al., 2023)

{{<citation>}}

Monika Kwiatkowski, Simon Matern, Olaf Hellwich. (2023)  
**DIAR: Deep Image Alignment and Reconstruction using Swin Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.11605v1)  

---


**ABSTRACT**  
When taking images of some occluded content, one is often faced with the problem that every individual image frame contains unwanted artifacts, but a collection of images contains all relevant information if properly aligned and aggregated. In this paper, we attempt to build a deep learning pipeline that simultaneously aligns a sequence of distorted images and reconstructs them. We create a dataset that contains images with image distortions, such as lighting, specularities, shadows, and occlusion. We create perspective distortions with corresponding ground-truth homographies as labels. We use our dataset to train Swin transformer models to analyze sequential image data. The attention maps enable the model to detect relevant image content and differentiate it from outliers and artifacts. We further explore using neural feature maps as alternatives to classical key point detectors. The feature maps of trained convolutional layers provide dense image descriptors that can be used to find point correspondences between images. We utilize this to compute coarse image alignments and explore its limitations.

{{</citation>}}


### (92/144) WaveAttack: Asymmetric Frequency Obfuscation-based Backdoor Attacks Against Deep Neural Networks (Jun Xia et al., 2023)

{{<citation>}}

Jun Xia, Zhihao Yue, Yingbo Zhou, Zhiwei Ling, Xian Wei, Mingsong Chen. (2023)  
**WaveAttack: Asymmetric Frequency Obfuscation-based Backdoor Attacks Against Deep Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11595v2)  

---


**ABSTRACT**  
Due to the popularity of Artificial Intelligence (AI) technology, numerous backdoor attacks are designed by adversaries to mislead deep neural network predictions by manipulating training samples and training processes. Although backdoor attacks are effective in various real scenarios, they still suffer from the problems of both low fidelity of poisoned samples and non-negligible transfer in latent space, which make them easily detectable by existing backdoor detection algorithms. To overcome the weakness, this paper proposes a novel frequency-based backdoor attack method named WaveAttack, which obtains image high-frequency features through Discrete Wavelet Transform (DWT) to generate backdoor triggers. Furthermore, we introduce an asymmetric frequency obfuscation method, which can add an adaptive residual in the training and inference stage to improve the impact of triggers and further enhance the effectiveness of WaveAttack. Comprehensive experimental results show that WaveAttack not only achieves higher stealthiness and effectiveness, but also outperforms state-of-the-art (SOTA) backdoor attack methods in the fidelity of images by up to 28.27\% improvement in PSNR, 1.61\% improvement in SSIM, and 70.59\% reduction in IS.

{{</citation>}}


### (93/144) Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V (Jianwei Yang et al., 2023)

{{<citation>}}

Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, Jianfeng Gao. (2023)  
**Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-HC, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.11441v1)  

---


**ABSTRACT**  
We present Set-of-Mark (SoM), a new visual prompting method, to unleash the visual grounding abilities of large multimodal models (LMMs), such as GPT-4V. As illustrated in Fig. 1 (right), we employ off-the-shelf interactive segmentation models, such as SAM, to partition an image into regions at different levels of granularity, and overlay these regions with a set of marks e.g., alphanumerics, masks, boxes. Using the marked image as input, GPT-4V can answer the questions that require visual grounding. We perform a comprehensive empirical study to validate the effectiveness of SoM on a wide range of fine-grained vision and multimodal tasks. For example, our experiments show that GPT-4V with SoM outperforms the state-of-the-art fully-finetuned referring segmentation model on RefCOCOg in a zero-shot setting.

{{</citation>}}


### (94/144) VcT: Visual change Transformer for Remote Sensing Image Change Detection (Bo Jiang et al., 2023)

{{<citation>}}

Bo Jiang, Zitian Wang, Xixi Wang, Ziyan Zhang, Lan Chen, Xiao Wang, Bin Luo. (2023)  
**VcT: Visual change Transformer for Remote Sensing Image Change Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.11417v1)  

---


**ABSTRACT**  
Existing visual change detectors usually adopt CNNs or Transformers for feature representation learning and focus on learning effective representation for the changed regions between images. Although good performance can be obtained by enhancing the features of the change regions, however, these works are still limited mainly due to the ignorance of mining the unchanged background context information. It is known that one main challenge for change detection is how to obtain the consistent representations for two images involving different variations, such as spatial variation, sunlight intensity, etc. In this work, we demonstrate that carefully mining the common background information provides an important cue to learn the consistent representations for the two images which thus obviously facilitates the visual change detection problem. Based on this observation, we propose a novel Visual change Transformer (VcT) model for visual change detection problem. To be specific, a shared backbone network is first used to extract the feature maps for the given image pair. Then, each pixel of feature map is regarded as a graph node and the graph neural network is proposed to model the structured information for coarse change map prediction. Top-K reliable tokens can be mined from the map and refined by using the clustering algorithm. Then, these reliable tokens are enhanced by first utilizing self/cross-attention schemes and then interacting with original features via an anchor-primary attention learning module. Finally, the prediction head is proposed to get a more accurate change map. Extensive experiments on multiple benchmark datasets validated the effectiveness of our proposed VcT model.

{{</citation>}}


### (95/144) Towards Automatic Satellite Images Captions Generation Using Large Language Models (Yingxu He et al., 2023)

{{<citation>}}

Yingxu He, Qiqi Sun. (2023)  
**Towards Automatic Satellite Images Captions Generation Using Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Falcon, GPT, GPT-3.5, Image Captioning, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11392v1)  

---


**ABSTRACT**  
Automatic image captioning is a promising technique for conveying visual information using natural language. It can benefit various tasks in satellite remote sensing, such as environmental monitoring, resource management, disaster management, etc. However, one of the main challenges in this domain is the lack of large-scale image-caption datasets, as they require a lot of human expertise and effort to create. Recent research on large language models (LLMs) has demonstrated their impressive performance in natural language understanding and generation tasks. Nonetheless, most of them cannot handle images (GPT-3.5, Falcon, Claude, etc.), while conventional captioning models pre-trained on general ground-view images often fail to produce detailed and accurate captions for aerial images (BLIP, GIT, CM3, CM3Leon, etc.). To address this problem, we propose a novel approach: Automatic Remote Sensing Image Captioning (ARSIC) to automatically collect captions for remote sensing images by guiding LLMs to describe their object annotations. We also present a benchmark model that adapts the pre-trained generative image2text model (GIT) to generate high-quality captions for remote-sensing images. Our evaluation demonstrates the effectiveness of our approach for collecting captions for remote sensing images.

{{</citation>}}


### (96/144) Towards Generalizable Multi-Camera 3D Object Detection via Perspective Debiasing (Hao Lu et al., 2023)

{{<citation>}}

Hao Lu, Yunpeng Zhang, Qing Lian, Dalong Du, Yingcong Chen. (2023)  
**Towards Generalizable Multi-Camera 3D Object Detection via Perspective Debiasing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.11346v1)  

---


**ABSTRACT**  
Detecting objects in 3D space using multiple cameras, known as Multi-Camera 3D Object Detection (MC3D-Det), has gained prominence with the advent of bird's-eye view (BEV) approaches. However, these methods often struggle when faced with unfamiliar testing environments due to the lack of diverse training data encompassing various viewpoints and environments. To address this, we propose a novel method that aligns 3D detection with 2D camera plane results, ensuring consistent and accurate detections. Our framework, anchored in perspective debiasing, helps the learning of features resilient to domain shifts. In our approach, we render diverse view maps from BEV features and rectify the perspective bias of these maps, leveraging implicit foreground volumes to bridge the camera and BEV planes. This two-step process promotes the learning of perspective- and context-independent features, crucial for accurate object detection across varying viewpoints, camera parameters and environment conditions. Notably, our model-agnostic approach preserves the original network structure without incurring additional inference costs, facilitating seamless integration across various models and simplifying deployment. Furthermore, we also show our approach achieves satisfactory results in real data when trained only with virtual datasets, eliminating the need for real scene annotations. Experimental results on both Domain Generalization (DG) and Unsupervised Domain Adaptation (UDA) clearly demonstrate its effectiveness. Our code will be released.

{{</citation>}}


### (97/144) Dual Cognitive Architecture: Incorporating Biases and Multi-Memory Systems for Lifelong Learning (Shruthi Gowda et al., 2023)

{{<citation>}}

Shruthi Gowda, Bahram Zonooz, Elahe Arani. (2023)  
**Dual Cognitive Architecture: Incorporating Biases and Multi-Memory Systems for Lifelong Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.11341v1)  

---


**ABSTRACT**  
Artificial neural networks (ANNs) exhibit a narrow scope of expertise on stationary independent data. However, the data in the real world is continuous and dynamic, and ANNs must adapt to novel scenarios while also retaining the learned knowledge to become lifelong learners. The ability of humans to excel at these tasks can be attributed to multiple factors ranging from cognitive computational structures, cognitive biases, and the multi-memory systems in the brain. We incorporate key concepts from each of these to design a novel framework, Dual Cognitive Architecture (DUCA), which includes multiple sub-systems, implicit and explicit knowledge representation dichotomy, inductive bias, and a multi-memory system. The inductive bias learner within DUCA is instrumental in encoding shape information, effectively countering the tendency of ANNs to learn local textures. Simultaneously, the inclusion of a semantic memory submodule facilitates the gradual consolidation of knowledge, replicating the dynamics observed in fast and slow learning systems, reminiscent of the principles underpinning the complementary learning system in human cognition. DUCA shows improvement across different settings and datasets, and it also exhibits reduced task recency bias, without the need for extra information. To further test the versatility of lifelong learning methods on a challenging distribution shift, we introduce a novel domain-incremental dataset DN4IL. In addition to improving performance on existing benchmarks, DUCA also demonstrates superior performance on this complex dataset.

{{</citation>}}


### (98/144) MonoSKD: General Distillation Framework for Monocular 3D Object Detection via Spearman Correlation Coefficient (Sen Wang et al., 2023)

{{<citation>}}

Sen Wang, Jin Zheng. (2023)  
**MonoSKD: General Distillation Framework for Monocular 3D Object Detection via Spearman Correlation Coefficient**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Knowledge Distillation, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.11316v1)  

---


**ABSTRACT**  
Monocular 3D object detection is an inherently ill-posed problem, as it is challenging to predict accurate 3D localization from a single image. Existing monocular 3D detection knowledge distillation methods usually project the LiDAR onto the image plane and train the teacher network accordingly. Transferring LiDAR-based model knowledge to RGB-based models is more complex, so a general distillation strategy is needed. To alleviate cross-modal prob-lem, we propose MonoSKD, a novel Knowledge Distillation framework for Monocular 3D detection based on Spearman correlation coefficient, to learn the relative correlation between cross-modal features. Considering the large gap between these features, strict alignment of features may mislead the training, so we propose a looser Spearman loss. Furthermore, by selecting appropriate distillation locations and removing redundant modules, our scheme saves more GPU resources and trains faster than existing methods. Extensive experiments are performed to verify the effectiveness of our framework on the challenging KITTI 3D object detection benchmark. Our method achieves state-of-the-art performance until submission with no additional inference computational cost. Our codes are available at https://github.com/Senwang98/MonoSKD

{{</citation>}}


### (99/144) Multi Self-supervised Pre-fine-tuned Transformer Fusion for Better Intelligent Transportation Detection (Juwu Zheng et al., 2023)

{{<citation>}}

Juwu Zheng, Jiangtao Ren. (2023)  
**Multi Self-supervised Pre-fine-tuned Transformer Fusion for Better Intelligent Transportation Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.11307v1)  

---


**ABSTRACT**  
Intelligent transportation system combines advanced information technology to provide intelligent services such as monitoring, detection, and early warning for modern transportation. Intelligent transportation detection is the cornerstone of many intelligent traffic services by identifying task targets through object detection methods. However existing detection methods in intelligent transportation are limited by two aspects. First, there is a difference between the model knowledge pre-trained on large-scale datasets and the knowledge required for target task. Second, most detection models follow the pattern of single-source learning, which limits the learning ability. To address these problems, we propose a Multi Self-supervised Pre-fine-tuned Transformer Fusion (MSPTF) network, consisting of two steps: unsupervised pre-fine-tune domain knowledge learning and multi-model fusion target task learning. In the first step, we introduced self-supervised learning methods into transformer model pre-fine-tune which could reduce data costs and alleviate the knowledge gap between pre-trained model and target task. In the second step, we take feature information differences between different model architectures and different pre-fine-tune tasks into account and propose Multi-model Semantic Consistency Cross-attention Fusion (MSCCF) network to combine different transformer model features by considering channel semantic consistency and feature vector semantic consistency, which obtain more complete and proper fusion features for detection task. We experimented the proposed method on vehicle recognition dataset and road disease detection dataset and achieved 1.1%, 5.5%, 4.2% improvement compared with baseline and 0.7%, 1.8%, 1.7% compared with sota, which proved the effectiveness of our method.

{{</citation>}}


### (100/144) Self-Supervised 3D Scene Flow Estimation and Motion Prediction using Local Rigidity Prior (Ruibo Li et al., 2023)

{{<citation>}}

Ruibo Li, Chi Zhang, Zhe Wang, Chunhua Shen, Guosheng Lin. (2023)  
**Self-Supervised 3D Scene Flow Estimation and Motion Prediction using Local Rigidity Prior**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.11284v1)  

---


**ABSTRACT**  
In this article, we investigate self-supervised 3D scene flow estimation and class-agnostic motion prediction on point clouds. A realistic scene can be well modeled as a collection of rigidly moving parts, therefore its scene flow can be represented as a combination of the rigid motion of these individual parts. Building upon this observation, we propose to generate pseudo scene flow labels for self-supervised learning through piecewise rigid motion estimation, in which the source point cloud is decomposed into local regions and each region is treated as rigid. By rigidly aligning each region with its potential counterpart in the target point cloud, we obtain a region-specific rigid transformation to generate its pseudo flow labels. To mitigate the impact of potential outliers on label generation, when solving the rigid registration for each region, we alternately perform three steps: establishing point correspondences, measuring the confidence for the correspondences, and updating the rigid transformation based on the correspondences and their confidence. As a result, confident correspondences will dominate label generation and a validity mask will be derived for the generated pseudo labels. By using the pseudo labels together with their validity mask for supervision, models can be trained in a self-supervised manner. Extensive experiments on FlyingThings3D and KITTI datasets demonstrate that our method achieves new state-of-the-art performance in self-supervised scene flow learning, without any ground truth scene flow for supervision, even performing better than some supervised counterparts. Additionally, our method is further extended to class-agnostic motion prediction and significantly outperforms previous state-of-the-art self-supervised methods on nuScenes dataset.

{{</citation>}}


### (101/144) FocDepthFormer: Transformer with LSTM for Depth Estimation from Focus (Xueyang Kang et al., 2023)

{{<citation>}}

Xueyang Kang, Fengze Han, Abdur Fayjie, Dong Gong. (2023)  
**FocDepthFormer: Transformer with LSTM for Depth Estimation from Focus**  

---
Primary Category: cs.CV  
Categories: I-4-9; I-2-10, cs-AI, cs-CV, cs.CV, eess-IV  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2310.11178v1)  

---


**ABSTRACT**  
Depth estimation from focal stacks is a fundamental computer vision problem that aims to infer depth from focus/defocus cues in the image stacks. Most existing methods tackle this problem by applying convolutional neural networks (CNNs) with 2D or 3D convolutions over a set of fixed stack images to learn features across images and stacks. Their performance is restricted due to the local properties of the CNNs, and they are constrained to process a fixed number of stacks consistent in train and inference, limiting the generalization to the arbitrary length of stacks. To handle the above limitations, we develop a novel Transformer-based network, FocDepthFormer, composed mainly of a Transformer with an LSTM module and a CNN decoder. The self-attention in Transformer enables learning more informative features via an implicit non-local cross reference. The LSTM module is learned to integrate the representations across the stack with arbitrary images. To directly capture the low-level features of various degrees of focus/defocus, we propose to use multi-scale convolutional kernels in an early-stage encoder. Benefiting from the design with LSTM, our FocDepthFormer can be pre-trained with abundant monocular RGB depth estimation data for visual pattern capturing, alleviating the demand for the hard-to-collect focal stack data. Extensive experiments on various focal stack benchmark datasets show that our model outperforms the state-of-the-art models on multiple metrics.

{{</citation>}}


### (102/144) USDC: Unified Static and Dynamic Compression for Visual Transformer (Huan Yuan et al., 2023)

{{<citation>}}

Huan Yuan, Chao Liao, Jianchao Tan, Peng Yao, Jiyuan Jia, Bin Chen, Chengru Song, Di Zhang. (2023)  
**USDC: Unified Static and Dynamic Compression for Visual Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.11117v1)  

---


**ABSTRACT**  
Visual Transformers have achieved great success in almost all vision tasks, such as classification, detection, and so on. However, the model complexity and the inference speed of the visual transformers hinder their deployments in industrial products. Various model compression techniques focus on directly compressing the visual transformers into a smaller one while maintaining the model performance, however, the performance drops dramatically when the compression ratio is large. Furthermore, several dynamic network techniques have also been applied to dynamically compress the visual transformers to obtain input-adaptive efficient sub-structures during the inference stage, which can achieve a better trade-off between the compression ratio and the model performance. The upper bound of memory of dynamic models is not reduced in the practical deployment since the whole original visual transformer model and the additional control gating modules should be loaded onto devices together for inference. To alleviate two disadvantages of two categories of methods, we propose to unify the static compression and dynamic compression techniques jointly to obtain an input-adaptive compressed model, which can further better balance the total compression ratios and the model performances. Moreover, in practical deployment, the batch sizes of the training and inference stage are usually different, which will cause the model inference performance to be worse than the model training performance, which is not touched by all previous dynamic network papers. We propose a sub-group gates augmentation technique to solve this performance drop problem. Extensive experiments demonstrate the superiority of our method on various baseline visual transformers such as DeiT, T2T-ViT, and so on.

{{</citation>}}


### (103/144) DORec: Decomposed Object Reconstruction Utilizing 2D Self-Supervised Features (Jun Wu et al., 2023)

{{<citation>}}

Jun Wu, Sicheng Li, Sihui Ji, Yue Wang, Rong Xiong, Yiyi Liao. (2023)  
**DORec: Decomposed Object Reconstruction Utilizing 2D Self-Supervised Features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.11092v2)  

---


**ABSTRACT**  
Decomposing a target object from a complex background while reconstructing is challenging. Most approaches acquire the perception for object instances through the use of manual labels, but the annotation procedure is costly. The recent advancements in 2D self-supervised learning have brought new prospects to object-aware representation, yet it remains unclear how to leverage such noisy 2D features for clean decomposition. In this paper, we propose a Decomposed Object Reconstruction (DORec) network based on neural implicit representations. Our key idea is to transfer 2D self-supervised features into masks of two levels of granularity to supervise the decomposition, including a binary mask to indicate the foreground regions and a K-cluster mask to indicate the semantically similar regions. These two masks are complementary to each other and lead to robust decomposition. Experimental results show the superiority of DORec in segmenting and reconstructing the foreground object on various datasets.

{{</citation>}}


### (104/144) Tracking and Mapping in Medical Computer Vision: A Review (Adam Schmidt et al., 2023)

{{<citation>}}

Adam Schmidt, Omid Mohareri, Simon DiMaio, Michael Yip, Septimiu E. Salcudean. (2023)  
**Tracking and Mapping in Medical Computer Vision: A Review**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2310.11475v1)  

---


**ABSTRACT**  
As computer vision algorithms are becoming more capable, their applications in clinical systems will become more pervasive. These applications include diagnostics such as colonoscopy and bronchoscopy, guiding biopsies and minimally invasive interventions and surgery, automating instrument motion and providing image guidance using pre-operative scans. Many of these applications depend on the specific visual nature of medical scenes and require designing and applying algorithms to perform in this environment.   In this review, we provide an update to the field of camera-based tracking and scene mapping in surgery and diagnostics in medical computer vision. We begin with describing our review process, which results in a final list of 515 papers that we cover. We then give a high-level summary of the state of the art and provide relevant background for those who need tracking and mapping for their clinical applications. We then review datasets provided in the field and the clinical needs therein. Then, we delve in depth into the algorithmic side, and summarize recent developments, which should be especially useful for algorithm designers and to those looking to understand the capability of off-the-shelf methods. We focus on algorithms for deformable environments while also reviewing the essential building blocks in rigid tracking and mapping since there is a large amount of crossover in methods. Finally, we discuss the current state of the tracking and mapping methods along with needs for future algorithms, needs for quantification, and the viability of clinical applications in the field. We conclude that new methods need to be designed or combined to support clinical applications in deformable environments, and more focus needs to be put into collecting datasets for training and evaluation.

{{</citation>}}


### (105/144) Unanswerable Visual Question Answering (Yanyang Guo et al., 2023)

{{<citation>}}

Yanyang Guo, Fangkai Jiao, Zhiqi Shen, Liqiang Nie, Mohan Kankanhalli. (2023)  
**Unanswerable Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.10942v1)  

---


**ABSTRACT**  
Teaching Visual Question Answering (VQA) models to abstain from unanswerable questions is indispensable for building a trustworthy AI system. Existing studies, though have explored various aspects of VQA, yet marginally ignored this particular attribute. This paper aims to bridge the research gap by contributing a comprehensive dataset, called UNK-VQA. The dataset is specifically designed to address the challenge of questions that can be unanswerable. To this end, we first augment the existing data via deliberate perturbations on either the image or question. In specific, we carefully ensure that the question-image semantics remain close to the original unperturbed distribution. By means of this, the identification of unanswerable questions becomes challenging, setting our dataset apart from others that involve mere image replacement. We then extensively evaluate the zero- and few-shot performance of several emerging multi-modal large models and discover significant limitations of them when applied to our dataset. Additionally, we also propose a straightforward method to tackle these unanswerable questions. This dataset, we believe, will serve as a valuable benchmark for enhancing the abstention capability of VQA models, thereby leading to increased trustworthiness of AI systems.

{{</citation>}}


## cs.CR (6)



### (106/144) The Efficacy of Transformer-based Adversarial Attacks in Security Domains (Kunyang Li et al., 2023)

{{<citation>}}

Kunyang Li, Kyle Domico, Jean-Charles Noirot Ferrand, Patrick McDaniel. (2023)  
**The Efficacy of Transformer-based Adversarial Attacks in Security Domains**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Adversarial Attack, Security, Transformer  
[Paper Link](http://arxiv.org/abs/2310.11597v1)  

---


**ABSTRACT**  
Today, the security of many domains rely on the use of Machine Learning to detect threats, identify vulnerabilities, and safeguard systems from attacks. Recently, transformer architectures have improved the state-of-the-art performance on a wide range of tasks such as malware detection and network intrusion detection. But, before abandoning current approaches to transformers, it is crucial to understand their properties and implications on cybersecurity applications. In this paper, we evaluate the robustness of transformers to adversarial samples for system defenders (i.e., resiliency to adversarial perturbations generated on different types of architectures) and their adversarial strength for system attackers (i.e., transferability of adversarial samples generated by transformers to other target models). To that effect, we first fine-tune a set of pre-trained transformer, Convolutional Neural Network (CNN), and hybrid (an ensemble of transformer and CNN) models to solve different downstream image-based tasks. Then, we use an attack algorithm to craft 19,367 adversarial examples on each model for each task. The transferability of these adversarial examples is measured by evaluating each set on other models to determine which models offer more adversarial strength, and consequently, more robustness against these attacks. We find that the adversarial examples crafted on transformers offer the highest transferability rate (i.e., 25.7% higher than the average) onto other models. Similarly, adversarial examples crafted on other models have the lowest rate of transferability (i.e., 56.7% lower than the average) onto transformers. Our work emphasizes the importance of studying transformer architectures for attacking and defending models in security domains, and suggests using them as the primary architecture in transfer attack settings.

{{</citation>}}


### (107/144) Functional Invariants to Watermark Large Transformers (Fernandez Pierre et al., 2023)

{{<citation>}}

Fernandez Pierre, Couairon Guillaume, Furon Teddy, Douze Matthijs. (2023)  
**Functional Invariants to Watermark Large Transformers**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.11446v1)  

---


**ABSTRACT**  
The rapid growth of transformer-based models increases the concerns about their integrity and ownership insurance. Watermarking addresses this issue by embedding a unique identifier into the model, while preserving its performance. However, most existing approaches require to optimize the weights to imprint the watermark signal, which is not suitable at scale due to the computational cost. This paper explores watermarks with virtually no computational cost, applicable to a non-blind white-box setting (assuming access to both the original and watermarked networks). They generate functionally equivalent copies by leveraging the models' invariance, via operations like dimension permutations or scaling/unscaling. This enables to watermark models without any change in their outputs and remains stealthy. Experiments demonstrate the effectiveness of the approach and its robustness against various model transformations (fine-tuning, quantization, pruning), making it a practical solution to protect the integrity of large models.

{{</citation>}}


### (108/144) Evaluating LLMs for Privilege-Escalation Scenarios (Andreas Happe et al., 2023)

{{<citation>}}

Andreas Happe, Aaron Kaplan, Jürgen Cito. (2023)  
**Evaluating LLMs for Privilege-Escalation Scenarios**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11409v1)  

---


**ABSTRACT**  
Penetration testing, an essential component of cybersecurity, allows organizations to proactively identify and remediate vulnerabilities in their systems, thus bolstering their defense mechanisms against potential cyberattacks. One recent advancement in the realm of penetration testing is the utilization of Language Models (LLMs). We explore the intersection of LLMs and penetration testing to gain insight into their capabilities and challenges in the context of privilige escalation. We create an automated Linux privilege-escalation benchmark utilizing local virtual machines. We introduce an LLM-guided privilege-escalation tool designed for evaluating different LLMs and prompt strategies against our benchmark. We analyze the impact of different prompt designs, the benefits of in-context learning, and the advantages of offering high-level guidance to LLMs. We discuss challenging areas for LLMs, including maintaining focus during testing, coping with errors, and finally comparing them with both stochastic parrots as well as with human hackers.

{{</citation>}}


### (109/144) Last One Standing: A Comparative Analysis of Security and Privacy of Soft Prompt Tuning, LoRA, and In-Context Learning (Rui Wen et al., 2023)

{{<citation>}}

Rui Wen, Tianhao Wang, Michael Backes, Yang Zhang, Ahmed Salem. (2023)  
**Last One Standing: A Comparative Analysis of Security and Privacy of Soft Prompt Tuning, LoRA, and In-Context Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Language Model, Security  
[Paper Link](http://arxiv.org/abs/2310.11397v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are powerful tools for natural language processing, enabling novel applications and user experiences. However, to achieve optimal performance, LLMs often require adaptation with private data, which poses privacy and security challenges. Several techniques have been proposed to adapt LLMs with private data, such as Low-Rank Adaptation (LoRA), Soft Prompt Tuning (SPT), and In-Context Learning (ICL), but their comparative privacy and security properties have not been systematically investigated. In this work, we fill this gap by evaluating the robustness of LoRA, SPT, and ICL against three types of well-established attacks: membership inference, which exposes data leakage (privacy); backdoor, which injects malicious behavior (security); and model stealing, which can violate intellectual property (privacy and security). Our results show that there is no silver bullet for privacy and security in LLM adaptation and each technique has different strengths and weaknesses.

{{</citation>}}


### (110/144) Detection of Malicious DNS-over-HTTPS Traffic: An Anomaly Detection Approach using Autoencoders (Sergio Salinas Monroy et al., 2023)

{{<citation>}}

Sergio Salinas Monroy, Aman Kumar Gupta, Garrett Wahlstedt. (2023)  
**Detection of Malicious DNS-over-HTTPS Traffic: An Anomaly Detection Approach using Autoencoders**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.11325v1)  

---


**ABSTRACT**  
To maintain the privacy of users' web browsing history, popular browsers encrypt their DNS traffic using the DNS-over-HTTPS (DoH) protocol. Unfortunately, encrypting DNS packets prevents many existing intrusion detection systems from using plaintext domain names to detect malicious traffic. In this paper, we design an autoencoder that is capable of detecting malicious DNS traffic by only observing the encrypted DoH traffic. Compared to previous works, the proposed autoencoder looks for anomalies in DoH traffic, and thus can detect malicious traffic that has not been previously observed, i.e., zero-day attacks. We run extensive experiments to evaluate the performance of our proposed autoencoder and compare it to that of other anomaly detection algorithms, namely, local outlier factor, one-class support vector machine, isolation forest, and variational autoencoders. We find that our proposed autoencoder achieves the highest detection performance, with a median F-1 score of 99\% over several types of malicious traffic.

{{</citation>}}


### (111/144) Locally Differentially Private Graph Embedding (Zening Li et al., 2023)

{{<citation>}}

Zening Li, Rong-Hua Li, Meihao Liao, Fusheng Jin, Guoren Wang. (2023)  
**Locally Differentially Private Graph Embedding**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs-SI, cs.CR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.11060v1)  

---


**ABSTRACT**  
Graph embedding has been demonstrated to be a powerful tool for learning latent representations for nodes in a graph. However, despite its superior performance in various graph-based machine learning tasks, learning over graphs can raise significant privacy concerns when graph data involves sensitive information. To address this, in this paper, we investigate the problem of developing graph embedding algorithms that satisfy local differential privacy (LDP). We propose LDP-GE, a novel privacy-preserving graph embedding framework, to protect the privacy of node data. Specifically, we propose an LDP mechanism to obfuscate node data and adopt personalized PageRank as the proximity measure to learn node representations. Then, we theoretically analyze the privacy guarantees and utility of the LDP-GE framework. Extensive experiments conducted over several real-world graph datasets demonstrate that LDP-GE achieves favorable privacy-utility trade-offs and significantly outperforms existing approaches in both node classification and link prediction tasks.

{{</citation>}}


## cs.AR (1)



### (112/144) Block-Wise Mixed-Precision Quantization: Enabling High Efficiency for Practical ReRAM-based DNN Accelerators (Xueying Wu et al., 2023)

{{<citation>}}

Xueying Wu, Edward Hanson, Nansu Wang, Qilin Zheng, Xiaoxuan Yang, Huanrui Yang, Shiyu Li, Feng Cheng, Partha Pratim Pande, Janardhan Rao Doppa, Krishnendu Chakrabarty, Hai, Li. (2023)  
**Block-Wise Mixed-Precision Quantization: Enabling High Efficiency for Practical ReRAM-based DNN Accelerators**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2310.12182v1)  

---


**ABSTRACT**  
Resistive random access memory (ReRAM)-based processing-in-memory (PIM) architectures have demonstrated great potential to accelerate Deep Neural Network (DNN) training/inference. However, the computational accuracy of analog PIM is compromised due to the non-idealities, such as the conductance variation of ReRAM cells. The impact of these non-idealities worsens as the number of concurrently activated wordlines and bitlines increases. To guarantee computational accuracy, only a limited number of wordlines and bitlines of the crossbar array can be turned on concurrently, significantly reducing the achievable parallelism of the architecture.   While the constraints on parallelism limit the efficiency of the accelerators, they also provide a new opportunity for fine-grained mixed-precision quantization. To enable efficient DNN inference on practical ReRAM-based accelerators, we propose an algorithm-architecture co-design framework called \underline{B}lock-\underline{W}ise mixed-precision \underline{Q}uantization (BWQ). At the algorithm level, BWQ-A introduces a mixed-precision quantization scheme at the block level, which achieves a high weight and activation compression ratio with negligible accuracy degradation. We also present the hardware architecture design BWQ-H, which leverages the low-bit-width models achieved by BWQ-A to perform high-efficiency DNN inference on ReRAM devices. BWQ-H also adopts a novel precision-aware weight mapping method to increase the ReRAM crossbar's throughput. Our evaluation demonstrates the effectiveness of BWQ, which achieves a 6.08x speedup and a 17.47x energy saving on average compared to existing ReRAM-based architectures.

{{</citation>}}


## cs.HC (4)



### (113/144) RekomGNN: Visualizing, Contextualizing and Evaluating Graph Neural Networks Recommendations (Camelia D. Brumar et al., 2023)

{{<citation>}}

Camelia D. Brumar, Gabriel Appleby, Jen Rogers, Teddy Matinde, Lara Thompson, Remco Chang, Anamaria Crisan. (2023)  
**RekomGNN: Visualizing, Contextualizing and Evaluating Graph Neural Networks Recommendations**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.11562v1)  

---


**ABSTRACT**  
Content recommendation tasks increasingly use Graph Neural Networks, but it remains challenging for machine learning experts to assess the quality of their outputs. Visualization systems for GNNs that could support this interrogation are few. Moreover, those that do exist focus primarily on exposing GNN architectures for tuning and prediction tasks and do not address the challenges of recommendation tasks. We developed RekomGNN, a visual analytics system that supports ML experts in exploring GNN recommendations across several dimensions and making annotations about their quality. RekomGNN straddles the design space between Neural Network and recommender system visualization to arrive at a set of encoding and interaction choices for recommendation tasks. We found that RekomGNN helps experts make qualitative assessments of the GNN's results, which they can use for model refinement. Overall, our contributions and findings add to the growing understanding of visualizing GNNs for increasingly complex tasks.

{{</citation>}}


### (114/144) Improving Operator Situation Awareness when Working with AI Recommender Systems (Divya K. Srivastava et al., 2023)

{{<citation>}}

Divya K. Srivastava, J. Mason Lilly, Karen M. Feigh. (2023)  
**Improving Operator Situation Awareness when Working with AI Recommender Systems**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11370v1)  

---


**ABSTRACT**  
AI recommender systems are sought for decision support by providing suggestions to operators responsible for making final decisions. However, these systems are typically considered black boxes, and are often presented without any context or insight into the underlying algorithm. As a result, recommender systems can lead to miscalibrated user reliance and decreased situation awareness. Recent work has focused on improving the transparency of recommender systems in various ways such as improving the recommender's analysis and visualization of the figures of merit, providing explanations for the recommender's decision, as well as improving user training or calibrating user trust. In this paper, we introduce an alternative transparency technique of structuring the order in which contextual information and the recommender's decision are shown to the human operator. This technique is designed to improve the operator's situation awareness and therefore the shared situation awareness between the operator and the recommender system. This paper presents the results of a two-phase between-subjects study in which participants and a recommender system jointly make a high-stakes decision. We varied the amount of contextual information the participant had, the assessment technique of the figures of merit, and the reliability of the recommender system. We found that providing contextual information upfront improves the team's shared situation awareness by improving the human decision maker's initial and final judgment, as well as their ability to discern the recommender's error boundary. Additionally, this technique accurately calibrated the human operator's trust in the recommender. This work proposes and validates a way to provide model-agnostic transparency into AI systems that can support the human decision maker and lead to improved team performance.

{{</citation>}}


### (115/144) On the Effectiveness of Creating Conversational Agent Personalities Through Prompting (Heng Gu et al., 2023)

{{<citation>}}

Heng Gu, Chadha Degachi, Uğur Genç, Senthil Chandrasegaran, Himanshu Verma. (2023)  
**On the Effectiveness of Creating Conversational Agent Personalities Through Prompting**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.11182v1)  

---


**ABSTRACT**  
In this work, we report on the effectiveness of our efforts to tailor the personality and conversational style of a conversational agent based on GPT-3.5 and GPT-4 through prompts. We use three personality dimensions with two levels each to create eight conversational agents archetypes. Ten conversations were collected per chatbot, of ten exchanges each, generating 1600 exchanges across GPT-3.5 and GPT-4. Using Linguistic Inquiry and Word Count (LIWC) analysis, we compared the eight agents on language elements including clout, authenticity, and emotion. Four language cues were significantly distinguishing in GPT-3.5, while twelve were distinguishing in GPT-4. With thirteen out of a total nineteen cues in LIWC appearing as significantly distinguishing, our results suggest possible novel prompting approaches may be needed to better suit the creation and evaluation of persistent conversational agent personalities or language styles.

{{</citation>}}


### (116/144) Using Audio Data to Facilitate Depression Risk Assessment in Primary Health Care (Adam Valen Levinson et al., 2023)

{{<citation>}}

Adam Valen Levinson, Abhay Goyal, Roger Ho Chun Man, Roy Ka-Wei Lee, Koustuv Saha, Nimay Parekh, Frederick L. Altice, Lam Yin Cheung, Munmun De Choudhury, Navin Kumar. (2023)  
**Using Audio Data to Facilitate Depression Risk Assessment in Primary Health Care**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-LG, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10928v1)  

---


**ABSTRACT**  
Telehealth is a valuable tool for primary health care (PHC), where depression is a common condition. PHC is the first point of contact for most people with depression, but about 25% of diagnoses made by PHC physicians are inaccurate. Many other barriers also hinder depression detection and treatment in PHC. Artificial intelligence (AI) may help reduce depression misdiagnosis in PHC and improve overall diagnosis and treatment outcomes. Telehealth consultations often have video issues, such as poor connectivity or dropped calls. Audio-only telehealth is often more practical for lower-income patients who may lack stable internet connections. Thus, our study focused on using audio data to predict depression risk. The objectives were to: 1) Collect audio data from 24 people (12 with depression and 12 without mental health or major health condition diagnoses); 2) Build a machine learning model to predict depression risk. TPOT, an autoML tool, was used to select the best machine learning algorithm, which was the K-nearest neighbors classifier. The selected model had high performance in classifying depression risk (Precision: 0.98, Recall: 0.93, F1-Score: 0.96). These findings may lead to a range of tools to help screen for and treat depression. By developing tools to detect depression risk, patients can be routed to AI-driven chatbots for initial screenings. Partnerships with a range of stakeholders are crucial to implementing these solutions. Moreover, ethical considerations, especially around data privacy and potential biases in AI models, need to be at the forefront of any AI-driven intervention in mental health care.

{{</citation>}}


## cs.DB (1)



### (117/144) Integrating 3D City Data through Knowledge Graphs (Linfang Ding et al., 2023)

{{<citation>}}

Linfang Ding, Guohui Xiao, Albulen Pano, Mattia Fumagalli, Dongsheng Chen, Yu Feng, Diego Calvanese, Hongchao Fan, Liqiu Meng. (2023)  
**Integrating 3D City Data through Knowledge Graphs**  

---
Primary Category: cs.DB  
Categories: cs-AI, cs-DB, cs.DB  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.11555v1)  

---


**ABSTRACT**  
CityGML is a widely adopted standard by the Open Geospatial Consortium (OGC) for representing and exchanging 3D city models. The representation of semantic and topological properties in CityGML makes it possible to query such 3D city data to perform analysis in various applications, e.g., security management and emergency response, energy consumption and estimation, and occupancy measurement. However, the potential of querying CityGML data has not been fully exploited. The official GML/XML encoding of CityGML is only intended as an exchange format but is not suitable for query answering. The most common way of dealing with CityGML data is to store them in the 3DCityDB system as relational tables and then query them with the standard SQL query language. Nevertheless, for end users, it remains a challenging task to formulate queries over 3DCityDB directly for their ad-hoc analytical tasks, because there is a gap between the conceptual semantics of CityGML and the relational schema adopted in 3DCityDB. In fact, the semantics of CityGML itself can be modeled as a suitable ontology. The technology of Knowledge Graphs (KGs), where an ontology is at the core, is a good solution to bridge such a gap. Moreover, embracing KGs makes it easier to integrate with other spatial data sources, e.g., OpenStreetMap and existing (Geo)KGs (e.g., Wikidata, DBPedia, and GeoNames), and to perform queries combining information from multiple data sources. In this work, we describe a CityGML KG framework to populate the concepts in the CityGML ontology using declarative mappings to 3DCityDB, thus exposing the CityGML data therein as a KG. To demonstrate the feasibility of our approach, we use CityGML data from the city of Munich as test data and integrate OpenStreeMap data in the same area.

{{</citation>}}


## cs.SE (5)



### (118/144) Bias and Error Mitigation in Software-Generated Data: An Advanced Search and Optimization Framework Leveraging Generative Code Models (Ernesto Giralt Hernández, 2023)

{{<citation>}}

Ernesto Giralt Hernández. (2023)  
**Bias and Error Mitigation in Software-Generated Data: An Advanced Search and Optimization Framework Leveraging Generative Code Models**  

---
Primary Category: cs.SE  
Categories: cs-IT, cs-LG, cs-SE, cs.SE, math-IT, math-OC  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2310.11546v1)  

---


**ABSTRACT**  
Data generation and analysis is a fundamental aspect of many industries and disciplines, from strategic decision making in business to research in the physical and social sciences. However, data generated using software and algorithms can be subject to biases and errors. These can be due to problems with the original software, default settings that do not align with the specific needs of the situation, or even deeper problems with the underlying theories and models. This paper proposes an advanced search and optimization framework aimed at generating and choosing optimal source code capable of correcting errors and biases from previous versions to address typical problems in software systems specializing in data analysis and generation, especially those in the corporate and data science world. Applying this framework multiple times on the same software system would incrementally improve the quality of the output results. It uses Solomonoff Induction as a sound theoretical basis, extending it with Kolmogorov Conditional Complexity, a novel adaptation, to evaluate a set of candidate programs. We propose the use of generative models for the creation of this set of programs, with special emphasis on the capabilities of Large Language Models (LLMs) to generate high quality code.

{{</citation>}}


### (119/144) Source Code Comprehension: A Contemporary Definition and Conceptual Model for Empirical Investigation (Marvin Wyrich, 2023)

{{<citation>}}

Marvin Wyrich. (2023)  
**Source Code Comprehension: A Contemporary Definition and Conceptual Model for Empirical Investigation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11301v1)  

---


**ABSTRACT**  
Be it in debugging, testing, code review or, more recently, pair programming with AI assistance: in all these activities, software engineers need to understand source code. Accordingly, plenty of research is taking place in the field to find out, for example, what makes code easy to understand and which tools can best support developers in their comprehension process. And while any code comprehension researcher certainly has a rough idea of what they mean when they mention a developer having a good understanding of a piece of code, to date, the research community has not managed to define source code comprehension as a concept. Instead, in primary research on code comprehension, an implicit definition by task prevails, i.e., code comprehension is what the experimental tasks measure. This approach has two negative consequences. First, it makes it difficult to conduct secondary research. Currently, each code comprehension primary study uses different comprehension tasks and measures, and thus it is not clear whether different studies intend to measure the same construct. Second, authors of a primary study run into the difficulty of justifying their design decisions without a definition of what they attempt to measure. An operationalization of an insufficiently described construct occurs, which poses a threat to construct validity.   The task of defining code comprehension considering the theory of the past fifty years is not an easy one. Nor is it a task that every author of a primary study must accomplish on their own. Therefore, this paper constitutes a reference work that defines source code comprehension and presents a conceptual framework in which researchers can anchor their empirical code comprehension research.

{{</citation>}}


### (120/144) Revisiting Sentiment Analysis for Software Engineering in the Era of Large Language Models (Ting Zhang et al., 2023)

{{<citation>}}

Ting Zhang, Ivana Clairine Irsan, Ferdian Thung, David Lo. (2023)  
**Revisiting Sentiment Analysis for Software Engineering in the Era of Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2310.11113v2)  

---


**ABSTRACT**  
Software development is an inherently collaborative process, where various stakeholders frequently express their opinions and emotions across diverse platforms. Recognizing the sentiments conveyed in these interactions is crucial for the effective development and ongoing maintenance of software systems. Over the years, many tools have been proposed to aid in sentiment analysis, but accurately identifying the sentiments expressed in software engineering datasets remains challenging.   Although fine-tuned smaller large language models (sLLMs) have shown potential in handling software engineering tasks, they struggle with the shortage of labeled data. With the emergence of bigger large language models (bLLMs), it is pertinent to investigate whether they can handle this challenge in the context of sentiment analysis for software engineering. In this work, we undertake a comprehensive empirical study using five established datasets. We assess the performance of three open-source bLLMs in both zero-shot and few-shot scenarios. Additionally, we compare them with fine-tuned sLLMs.   Our experimental findings demonstrate that bLLMs exhibit state-of-the-art performance on datasets marked by limited training data and imbalanced distributions. bLLMs can also achieve excellent performance under a zero-shot setting. However, when ample training data is available or the dataset exhibits a more balanced distribution, fine-tuned sLLMs can still achieve superior results.

{{</citation>}}


### (121/144) Program Translation via Code Distillation (Yufan Huang et al., 2023)

{{<citation>}}

Yufan Huang, Mengnan Qi, Yongqiang Yao, Maoquan Wang, Bin Gu, Colin Clement, Neel Sundaresan. (2023)  
**Program Translation via Code Distillation**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: GLUE  
[Paper Link](http://arxiv.org/abs/2310.11476v1)  

---


**ABSTRACT**  
Software version migration and program translation are an important and costly part of the lifecycle of large codebases. Traditional machine translation relies on parallel corpora for supervised translation, which is not feasible for program translation due to a dearth of aligned data. Recent unsupervised neural machine translation techniques have overcome data limitations by included techniques such as back translation and low level compiler intermediate representations (IR). These methods face significant challenges due to the noise in code snippet alignment and the diversity of IRs respectively. In this paper we propose a novel model called Code Distillation (CoDist) whereby we capture the semantic and structural equivalence of code in a language agnostic intermediate representation. Distilled code serves as a translation pivot for any programming language, leading by construction to parallel corpora which scale to all available source code by simply applying the distillation compiler. We demonstrate that our approach achieves state-of-the-art performance on CodeXGLUE and TransCoder GeeksForGeeks translation benchmarks, with an average absolute increase of 12.7% on the TransCoder GeeksforGeeks translation benchmark compare to TransCoder-ST.

{{</citation>}}


### (122/144) ClarifyGPT: Empowering LLM-based Code Generation with Intention Clarification (Fangwen Mu et al., 2023)

{{<citation>}}

Fangwen Mu, Lin Shi, Song Wang, Zhuohao Yu, Binquan Zhang, Chenxue Wang, Shichao Liu, Qing Wang. (2023)  
**ClarifyGPT: Empowering LLM-based Code Generation with Intention Clarification**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.10996v1)  

---


**ABSTRACT**  
We introduce a novel framework named ClarifyGPT, which aims to enhance code generation by empowering LLMs with the ability to identify ambiguous requirements and ask targeted clarifying questions. In particular, ClarifyGPT first detects whether a given requirement is ambiguous by performing a code consistency check. If it is ambiguous, ClarifyGPT prompts an LLM to generate targeted clarifying questions. After receiving question responses, ClarifyGPT refines the ambiguous requirement and inputs it into the same LLM to generate a final code solution. To evaluate our ClarifyGPT, we first conduct a human evaluation involving ten participants who use ClarifyGPT for code generation on two publicly available benchmarks: MBPP-sanitized and MBPP-ET. The results show that ClarifyGPT elevates the performance (Pass@1) of GPT-4 from 70.96% to 80.80% on MBPP-sanitized. Furthermore, to perform large-scale automated evaluations of ClarifyGPT across different LLMs and benchmarks without requiring user participation, we introduce a high-fidelity simulation method to simulate user responses. The automated evaluation results also demonstrate that ClarifyGPT can significantly enhance code generation performance compared to the baselines. In particular, ClarifyGPT improves the average performance of GPT-4 and ChatGPT across four benchmarks from 68.02% to 75.75% and from 58.55% to 67.22%, respectively. We believe that ClarifyGPT can effectively facilitate the practical application of LLMs in real-world development environments.

{{</citation>}}


## cs.SI (3)



### (123/144) HMN: Generalization of Heterogeneous and Multi-layered Network (Shraban Kumar Chatterjee et al., 2023)

{{<citation>}}

Shraban Kumar Chatterjee, Suman Kundu. (2023)  
**HMN: Generalization of Heterogeneous and Multi-layered Network**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.11534v1)  

---


**ABSTRACT**  
A network may have different types of entities and their relations. Further, there could be additional layers of ties. The former is referred to as Heterogeneous networks, while the latter is known as Multi-layer networks. The present paper provides a generalized network model, namely, a Heterogeneous Multi-layered Network (HMN), which can simultaneously be multi-layered and heterogeneous. The model can represent homogeneous networks as well. We define different structural measures in an HMN. We proved that the sets of all homogeneous, heterogeneous and multi-layered networks are subsets of the set of all HMNs. Accordingly, we established the equivalency of the proposed structural measures of HMNs with that of homogeneous, heterogeneous, and multi-layered networks. Following that, we show how our proposed HMN is more efficient in tasks such as link prediction. In addition, we present a novel parameterized algorithm (with complexity analysis) for generating synthetic HMNs. The networks generated from our proposed algorithm are more consistent in modelling the layer-wise degree distribution of a real-world Twitter network (represented as HMN) than those generated by existing models. Moreover, we also show that our algorithm is more effective in modelling an air-transportation multiplex network when compared to an algorithm designed specifically for the task.

{{</citation>}}


### (124/144) Sadness, Anger, or Anxiety: Twitter Users' Emotional Responses to Toxicity in Public Conversations (Ana Aleksandric et al., 2023)

{{<citation>}}

Ana Aleksandric, Hanani Pankaj, Gabriela Mustata Wilson, Shirin Nilizadeh. (2023)  
**Sadness, Anger, or Anxiety: Twitter Users' Emotional Responses to Toxicity in Public Conversations**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.11436v1)  

---


**ABSTRACT**  
Cyberbullying and online harassment have serious negative psychological and emotional consequences for the victims, such as decreased life satisfaction, suicidal ideation, self-harming behaviors, depression, anxiety, and others. Most of the prior works assessed people's emotional responses via questionnaires, while social media platforms contain data that could provide valuable insights into users' emotions in real online discussions. Therefore, this data-driven study investigates the effect of toxicity on Twitter users' emotions and other factors associated with expressing anger, anxiety, and sadness in terms of account identifiability, activity, conversation structure, and conversation topic. To achieve this goal, we identified toxic replies in the large dataset consisting of 79,799 random Twitter conversations and obtained the emotions expressed in these conversations. Then, we performed propensity score matching and analyzed causal associations between toxicity and users' emotions. In general, we found that users receiving toxic replies are more likely to express emotions of anger, sadness, and anxiety compared to users who did not receive toxic replies. Finally, analysis results indicate that the conversation topic and users' account characteristics are likely to affect their emotional responses to toxicity. Our findings provide a better understanding of toxic replies' consequences on users' emotional states, which can potentially lead to developing personalized moderation methods that will help users emotionally cope with toxicity on social media.

{{</citation>}}


### (125/144) Analyzing Modularity Maximization in Approximation, Heuristic, and Graph Neural Network Algorithms for Community Detection (Samin Aref et al., 2023)

{{<citation>}}

Samin Aref, Mahdi Mostajabdaveh. (2023)  
**Analyzing Modularity Maximization in Approximation, Heuristic, and Graph Neural Network Algorithms for Community Detection**  

---
Primary Category: cs.SI  
Categories: 90C90, 90C10, 90C57, 90C59, 90C35, 05C15, 65K05, cond-mat-stat-mech, cs-LG, cs-SI, cs.SI, math-OC  
Keywords: Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.10898v1)  

---


**ABSTRACT**  
Community detection, a fundamental problem in computational sciences, finds applications in various domains. Heuristics are often employed to detect communities through maximizing an objective function, modularity, over partitions of network nodes. Our research delves into the performance of different modularity maximization algorithms in achieving optimal partitions. We use 104 networks, comprising real-world instances from diverse contexts and synthetic graphs with modular structures. We analyze ten inexact modularity-based algorithms against an exact baseline which is an exact integer programming method that globally optimizes modularity. The ten algorithms analyzed include eight heuristics, two variations of a graph neural network algorithm, and several variations of the Bayan approximation algorithm. Our analysis uncovers substantial dissimilarities between the partitions obtained by most commonly used modularity-based methods and any optimal partition of the networks, as indicated by both adjusted and reduced mutual information metrics. Importantly, our results show that near-optimal partitions are often disproportionately dissimilar to any optimal partition. Taken together, our analysis points to a crucial limitation of the commonly used unguaranteed modularity-based methods for discovering communities: they rarely produce an optimal partition or a partition resembling an optimal partition even on networks with modular structures. If modularity is to be used for detecting communities, approximate optimization algorithms are recommendable for a more methodologically sound usage of modularity within its applicability limits.

{{</citation>}}


## cs.CY (3)



### (126/144) Large Language Model Prediction Capabilities: Evidence from a Real-World Forecasting Tournament (Philipp Schoenegger et al., 2023)

{{<citation>}}

Philipp Schoenegger, Peter S. Park. (2023)  
**Large Language Model Prediction Capabilities: Evidence from a Real-World Forecasting Tournament**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CY  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13014v1)  

---


**ABSTRACT**  
Accurately predicting the future would be an important milestone in the capabilities of artificial intelligence. However, research on the ability of large language models to provide probabilistic predictions about future events remains nascent. To empirically test this ability, we enrolled OpenAI's state-of-the-art large language model, GPT-4, in a three-month forecasting tournament hosted on the Metaculus platform. The tournament, running from July to October 2023, attracted 843 participants and covered diverse topics including Big Tech, U.S. politics, viral outbreaks, and the Ukraine conflict. Focusing on binary forecasts, we show that GPT-4's probabilistic forecasts are significantly less accurate than the median human-crowd forecasts. We find that GPT-4's forecasts did not significantly differ from the no-information forecasting strategy of assigning a 50% probability to every question. We explore a potential explanation, that GPT-4 might be predisposed to predict probabilities close to the midpoint of the scale, but our data do not support this hypothesis. Overall, we find that GPT-4 significantly underperforms in real-world predictive tasks compared to median human-crowd forecasts. A potential explanation for this underperformance is that in real-world forecasting tournaments, the true answers are genuinely unknown at the time of prediction; unlike in other benchmark tasks like professional exams or time series forecasting, where strong performance may at least partly be due to the answers being memorized from the training data. This makes real-world forecasting tournaments an ideal environment for testing the generalized reasoning and prediction capabilities of artificial intelligence going forward.

{{</citation>}}


### (127/144) Cross-Platform Social Dynamics: An Analysis of ChatGPT and COVID-19 Vaccine Conversations (Shayan Alipour et al., 2023)

{{<citation>}}

Shayan Alipour, Alessandro Galeazzi, Emanuele Sangiorgio, Michele Avalle, Ljubisa Bojic, Matteo Cinelli, Walter Quattrociocchi. (2023)  
**Cross-Platform Social Dynamics: An Analysis of ChatGPT and COVID-19 Vaccine Conversations**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY, physics-soc-ph  
Keywords: ChatGPT, GPT, Twitter  
[Paper Link](http://arxiv.org/abs/2310.11116v1)  

---


**ABSTRACT**  
The role of social media in information dissemination and agenda-setting has significantly expanded in recent years. By offering real-time interactions, online platforms have become invaluable tools for studying societal responses to significant events as they unfold. However, online reactions to external developments are influenced by various factors, including the nature of the event and the online environment. This study examines the dynamics of public discourse on digital platforms to shed light on this issue. We analyzed over 12 million posts and news articles related to two significant events: the release of ChatGPT in 2022 and the global discussions about COVID-19 vaccines in 2021. Data was collected from multiple platforms, including Twitter, Facebook, Instagram, Reddit, YouTube, and GDELT. We employed topic modeling techniques to uncover the distinct thematic emphases on each platform, which reflect their specific features and target audiences. Additionally, sentiment analysis revealed various public perceptions regarding the topics studied. Lastly, we compared the evolution of engagement across platforms, unveiling unique patterns for the same topic. Notably, discussions about COVID-19 vaccines spread more rapidly due to the immediacy of the subject, while discussions about ChatGPT, despite its technological importance, propagated more gradually.

{{</citation>}}


### (128/144) Unveiling Local Patterns of Child Pornography Consumption in France using Tor (Till Koebe et al., 2023)

{{<citation>}}

Till Koebe, Zinnya del Villar, Brahmani Nutakki, Nursulu Sagimbayeva, Ingmar Weber. (2023)  
**Unveiling Local Patterns of Child Pornography Consumption in France using Tor**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-SI, cs.CY  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.11099v2)  

---


**ABSTRACT**  
Child pornography represents a severe form of exploitation and victimization of children, leaving the victims with emotional and physical trauma. In this study, we aim to analyze local patterns of child pornography consumption in 20 metropolitan regions of France using fine-grained mobile traffic data of Tor network-related web services. We estimate that approx. 3.3 % of Tor mobile download traffic observed in France is linked to the consumption of child sexual abuse materials by correlating it with local-level temporal porn consumption patterns. This compares to 0.2 % of what we conservatively estimate to be the share of child pornographic content in global Tor traffic. In line with existing literature on the link between sexual child abuse and the consumption of image-based content thereof, we observe a positive and statistically significant effect of our child pornography consumption estimates on the reported number of victims of sexual violence and vice versa across 1341 French communes, which validates our findings, after controlling for a set of spatial and non-spatial features including socio-demographic characteristics, voting behaviour, nearby points of interest and Google Trends queries. While this is a first, exploratory attempt to look at child pornography from a spatial epidemiological angle, we believe this research provides public health officials with valuable information to prioritize target areas for public awareness campaigns and hopefully inform future paths of research in that area.

{{</citation>}}


## cs.IR (3)



### (129/144) On Coherence-based Predictors for Dense Query Performance Prediction (Maria Vlachou et al., 2023)

{{<citation>}}

Maria Vlachou, Craig Macdonald. (2023)  
**On Coherence-based Predictors for Dense Query Performance Prediction**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.11405v1)  

---


**ABSTRACT**  
Query Performance Prediction (QPP) estimates the effectiveness of a search engine's results in response to a query without relevance judgments. Traditionally, post-retrieval predictors have focused upon either the distribution of the retrieval scores, or the coherence of the top-ranked documents using traditional bag-of-words index representations. More recently, BERT-based models using dense embedded document representations have been used to create new predictors, but mostly applied to predict the performance of rankings created by BM25. Instead, we aim to predict the effectiveness of rankings created by single-representation dense retrieval models (ANCE & TCT-ColBERT). Therefore, we propose a number of variants of existing unsupervised coherence-based predictors that employ neural embedding representations. In our experiments on the TREC Deep Learning Track datasets, we demonstrate improved accuracy upon dense retrieval (up to 92% compared to sparse variants for TCT-ColBERT and 188% for ANCE). Going deeper, we select the most representative and best performing predictors to study the importance of differences among predictors and query types on query performance. Using existing distribution-based evaluation QPP measures and a particular type of linear mixed models, we find that query types further significantly influence query performance (and are up to 35% responsible for the unstable performance of QPP predictors), and that this sensitivity is unique to dense retrieval models. Our approach introduces a new setting for obtaining richer information on query differences in dense QPP that can explain potential unstable performance of existing predictors and outlines the unique characteristics of different query types on dense retrieval models.

{{</citation>}}


### (130/144) Graph Neural Networks for Recommendation: Reproducibility, Graph Topology, and Node Representation (Daniele Malitesta et al., 2023)

{{<citation>}}

Daniele Malitesta, Claudio Pomo, Tommaso Di Noia. (2023)  
**Graph Neural Networks for Recommendation: Reproducibility, Graph Topology, and Node Representation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.11270v2)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have gained prominence in recommendation systems in recent years. By representing the user-item matrix as a bipartite and undirected graph, GNNs have demonstrated their potential to capture short- and long-distance user-item interactions, thereby learning more accurate preference patterns than traditional recommendation approaches. In contrast to previous tutorials on the same topic, this tutorial aims to present and examine three key aspects that characterize GNNs for recommendation: (i) the reproducibility of state-of-the-art approaches, (ii) the potential impact of graph topological characteristics on the performance of these models, and (iii) strategies for learning node representations when training features from scratch or utilizing pre-trained embeddings as additional item information (e.g., multimodal features). The goal is to provide three novel theoretical and practical perspectives on the field, currently subject to debate in graph learning but long been overlooked in the context of recommendation systems.

{{</citation>}}


### (131/144) MeKB-Rec: Personal Knowledge Graph Learning for Cross-Domain Recommendation (Xin Su et al., 2023)

{{<citation>}}

Xin Su, Yao Zhou, Zifei Shan, Qian Chen. (2023)  
**MeKB-Rec: Personal Knowledge Graph Learning for Cross-Domain Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Knowledge Graph, Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2310.11088v1)  

---


**ABSTRACT**  
It is a long-standing challenge in modern recommender systems to effectively make recommendations for new users, namely the cold-start problem. Cross-Domain Recommendation (CDR) has been proposed to address this challenge, but current ways to represent users' interests across systems are still severely limited. We introduce Personal Knowledge Graph (PKG) as a domain-invariant interest representation, and propose a novel CDR paradigm named MeKB-Rec. We first link users and entities in a knowledge base to construct a PKG of users' interests, named MeKB. Then we learn a semantic representation of MeKB for the cross-domain recommendation. To efficiently utilize limited training data in CDR, MeKB-Rec employs Pretrained Language Models to inject world knowledge into understanding users' interests. Beyond most existing systems, our approach builds a semantic mapping across domains which breaks the requirement for in-domain user behaviors, enabling zero-shot recommendations for new users in a low-resource domain. We experiment MeKB-Rec on well-established public CDR datasets, and demonstrate that the new formulation % is more powerful than previous approaches, achieves a new state-of-the-art that significantly improves HR@10 and NDCG@10 metrics over best previous approaches by 24\%--91\%, with a 105\% improvement for HR@10 of zero-shot users with no behavior in the target domain. We deploy MeKB-Rec in WeiXin recommendation scenarios and achieve significant gains in core online metrics. MeKB-Rec is now serving hundreds of millions of users in real-world products.

{{</citation>}}


## quant-ph (1)



### (132/144) Quantum Financial Modeling on NISQ Hardware: Random Walks using Approximate Quantum Counting (Dominic Widdows, 2023)

{{<citation>}}

Dominic Widdows. (2023)  
**Quantum Financial Modeling on NISQ Hardware: Random Walks using Approximate Quantum Counting**  

---
Primary Category: quant-ph  
Categories: cs-CE, quant-ph, quant-ph  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2310.11394v1)  

---


**ABSTRACT**  
Quantum computers are expected to contribute more efficient and accurate ways of modeling economic processes. Quantum hardware is currently available at a relatively small scale, but effective algorithms are limited by the number of logic gates that can be used, before noise from gate inaccuracies tends to dominate results. Some theoretical algorithms that have been proposed and studied for years do not perform well yet on quantum hardware in practice. This encourages the development of suitable alternative algorithms that play similar roles in limited contexts.   This paper implements this strategy in the case of quantum counting, which is used as a component for keeping track of position in a quantum walk, which is used as a model for simulating asset prices over time. We introduce quantum approximate counting circuits that use far fewer 2-qubit entangling gates than traditional quantum counting that relies on binary positional encoding. The robustness of these circuits to noise is demonstrated.   While this paper is mainly about robust simplified quantum circuit designs, we compare some aspects of the results with price change distributions from stock indices, and compare the behavior of circuits with and without mid-measurement to trends in the housing market.

{{</citation>}}


## cs.LO (1)



### (133/144) Linear-Time Verification of Data-Aware Processes Modulo Theories via Covers and Automata (Extended Version) (Alessandro Gianola et al., 2023)

{{<citation>}}

Alessandro Gianola, Marco Montali, Sarah Winkler. (2023)  
**Linear-Time Verification of Data-Aware Processes Modulo Theories via Covers and Automata (Extended Version)**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs.LO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.12180v1)  

---


**ABSTRACT**  
The need to model and analyse dynamic systems operating over complex data is ubiquitous in AI and neighboring areas, in particular business process management. Analysing such data-aware systems is a notoriously difficult problem, as they are intrinsically infinite-state. Existing approaches work for specific datatypes, and/or limit themselves to the verification of safety properties. In this paper, we lift both such limitations, studying for the first time linear-time verification for so-called data-aware processes modulo theories (DMTs), from the foundational and practical point of view. The DMT model is very general, as it supports processes operating over variables that can store arbitrary types of data, ranging over infinite domains and equipped with domain-specific predicates. Specifically, we provide four contributions. First, we devise a semi-decision procedure for linear-time verification of DMTs, which works for a very large class of datatypes obeying to mild model-theoretic assumptions. The procedure relies on a unique combination of automata-theoretic and cover computation techniques to respectively deal with linear-time properties and datatypes. Second, we identify an abstract, semantic property that guarantees the existence of a faithful finite-state abstraction of the original system, and show that our method becomes a decision procedure in this case. Third, we identify concrete, checkable classes of systems that satisfy this property, generalising several results in the literature. Finally, we present an implementation and a first experimental evaluation.

{{</citation>}}


## eess.IV (4)



### (134/144) Towards Generic Semi-Supervised Framework for Volumetric Medical Image Segmentation (Haonan Wang et al., 2023)

{{<citation>}}

Haonan Wang, Xiaomeng Li. (2023)  
**Towards Generic Semi-Supervised Framework for Volumetric Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.11320v1)  

---


**ABSTRACT**  
Volume-wise labeling in 3D medical images is a time-consuming task that requires expertise. As a result, there is growing interest in using semi-supervised learning (SSL) techniques to train models with limited labeled data. However, the challenges and practical applications extend beyond SSL to settings such as unsupervised domain adaptation (UDA) and semi-supervised domain generalization (SemiDG). This work aims to develop a generic SSL framework that can handle all three settings. We identify two main obstacles to achieving this goal in the existing SSL framework: 1) the weakness of capturing distribution-invariant features; and 2) the tendency for unlabeled data to be overwhelmed by labeled data, leading to over-fitting to the labeled data during training. To address these issues, we propose an Aggregating & Decoupling framework. The aggregating part consists of a Diffusion encoder that constructs a common knowledge set by extracting distribution-invariant features from aggregated information from multiple distributions/domains. The decoupling part consists of three decoders that decouple the training process with labeled and unlabeled data, thus avoiding over-fitting to labeled data, specific domains and classes. We evaluate our proposed framework on four benchmark datasets for SSL, Class-imbalanced SSL, UDA and SemiDG. The results showcase notable improvements compared to state-of-the-art methods across all four settings, indicating the potential of our framework to tackle more challenging SSL scenarios. Code and models are available at: https://github.com/xmed-lab/GenericSSL.

{{</citation>}}


### (135/144) Image Compression using only Attention based Neural Networks (Natacha Luka et al., 2023)

{{<citation>}}

Natacha Luka, Romain Negrel, David Picard. (2023)  
**Image Compression using only Attention based Neural Networks**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.11265v1)  

---


**ABSTRACT**  
In recent research, Learned Image Compression has gained prominence for its capacity to outperform traditional handcrafted pipelines, especially at low bit-rates. While existing methods incorporate convolutional priors with occasional attention blocks to address long-range dependencies, recent advances in computer vision advocate for a transformative shift towards fully transformer-based architectures grounded in the attention mechanism. This paper investigates the feasibility of image compression exclusively using attention layers within our novel model, QPressFormer. We introduce the concept of learned image queries to aggregate patch information via cross-attention, followed by quantization and coding techniques. Through extensive evaluations, our work demonstrates competitive performance achieved by convolution-free architectures across the popular Kodak, DIV2K, and CLIC datasets.

{{</citation>}}


### (136/144) $k$-$t$ CLAIR: Self-Consistency Guided Multi-Prior Learning for Dynamic Parallel MR Image Reconstruction (Liping Zhang et al., 2023)

{{<citation>}}

Liping Zhang, Weitian Chen. (2023)  
**$k$-$t$ CLAIR: Self-Consistency Guided Multi-Prior Learning for Dynamic Parallel MR Image Reconstruction**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, physics-med-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.11050v1)  

---


**ABSTRACT**  
Cardiac magnetic resonance imaging (CMR) has been widely used in clinical practice for the medical diagnosis of cardiac diseases. However, the long acquisition time hinders its development in real-time applications. Here, we propose a novel self-consistency guided multi-prior learning framework named $k$-$t$ CLAIR to exploit spatiotemporal correlations from highly undersampled data for accelerated dynamic parallel MRI reconstruction. The $k$-$t$ CLAIR progressively reconstructs faithful images by leveraging multiple complementary priors learned in the $x$-$t$, $x$-$f$, and $k$-$t$ domains in an iterative fashion, as dynamic MRI exhibits high spatiotemporal redundancy. Additionally, $k$-$t$ CLAIR incorporates calibration information for prior learning, resulting in a more consistent reconstruction. Experimental results on cardiac cine and T1W/T2W images demonstrate that $k$-$t$ CLAIR achieves high-quality dynamic MR reconstruction in terms of both quantitative and qualitative performance.

{{</citation>}}


### (137/144) Medical Image Segmentation via Sparse Coding Decoder (Long Zeng et al., 2023)

{{<citation>}}

Long Zeng, Kaigui Wu. (2023)  
**Medical Image Segmentation via Sparse Coding Decoder**  

---
Primary Category: eess.IV  
Categories: 68T07, 68U10, I-4-6; I-4-7; I-5-1, cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.10957v1)  

---


**ABSTRACT**  
Transformers have achieved significant success in medical image segmentation, owing to its capability to capture long-range dependencies. Previous works incorporate convolutional layers into the encoder module of transformers, thereby enhancing their ability to learn local relationships among pixels. However, transformers may suffer from limited generalization capabilities and reduced robustness, attributed to the insufficient spatial recovery ability of their decoders. To address this issue, A convolution sparse vector coding based decoder is proposed , namely CAScaded multi-layer Convolutional Sparse vector Coding DEcoder (CASCSCDE), which represents features extracted by the encoder using sparse vectors. To prove the effectiveness of our CASCSCDE, The widely-used TransUNet model is chosen for the demonstration purpose, and the CASCSCDE is incorporated with TransUNet to establish the TransCASCSCDE architecture. Our experiments demonstrate that TransUNet with CASCSCDE significantly enhances performance on the Synapse benchmark, obtaining up to 3.15\% and 1.16\% improvements in DICE and mIoU scores, respectively. CASCSCDE opens new ways for constructing decoders based on convolutional sparse vector coding.

{{</citation>}}


## cs.AI (4)



### (138/144) Leveraging Large Language Model for Automatic Evolving of Industrial Data-Centric R&D Cycle (Xu Yang et al., 2023)

{{<citation>}}

Xu Yang, Xiao Yang, Weiqing Liu, Jinhui Li, Peng Yu, Zeqi Ye, Jiang Bian. (2023)  
**Leveraging Large Language Model for Automatic Evolving of Industrial Data-Centric R&D Cycle**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, q-fin-GN  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.11249v1)  

---


**ABSTRACT**  
In the wake of relentless digital transformation, data-driven solutions are emerging as powerful tools to address multifarious industrial tasks such as forecasting, anomaly detection, planning, and even complex decision-making. Although data-centric R&D has been pivotal in harnessing these solutions, it often comes with significant costs in terms of human, computational, and time resources. This paper delves into the potential of large language models (LLMs) to expedite the evolution cycle of data-centric R&D. Assessing the foundational elements of data-centric R&D, including heterogeneous task-related data, multi-facet domain knowledge, and diverse computing-functional tools, we explore how well LLMs can understand domain-specific requirements, generate professional ideas, utilize domain-specific tools to conduct experiments, interpret results, and incorporate knowledge from past endeavors to tackle new challenges. We take quantitative investment research as a typical example of industrial data-centric R&D scenario and verified our proposed framework upon our full-stack open-sourced quantitative research platform Qlib and obtained promising results which shed light on our vision of automatic evolving of industrial data-centric R&D cycle.

{{</citation>}}


### (139/144) Query2Triple: Unified Query Encoding for Answering Diverse Complex Queries over Knowledge Graphs (Yao Xu et al., 2023)

{{<citation>}}

Yao Xu, Shizhu He, Cunguang Wang, Li Cai, Kang Liu, Jun Zhao. (2023)  
**Query2Triple: Unified Query Encoding for Answering Diverse Complex Queries over Knowledge Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Knowledge Graph, QA  
[Paper Link](http://arxiv.org/abs/2310.11246v1)  

---


**ABSTRACT**  
Complex Query Answering (CQA) is a challenge task of Knowledge Graph (KG). Due to the incompleteness of KGs, query embedding (QE) methods have been proposed to encode queries and entities into the same embedding space, and treat logical operators as neural set operators to obtain answers. However, these methods train KG embeddings and neural set operators concurrently on both simple (one-hop) and complex (multi-hop and logical) queries, which causes performance degradation on simple queries and low training efficiency. In this paper, we propose Query to Triple (Q2T), a novel approach that decouples the training for simple and complex queries. Q2T divides the training into two stages: (1) Pre-training a neural link predictor on simple queries to predict tail entities based on the head entity and relation. (2) Training a query encoder on complex queries to encode diverse complex queries into a unified triple form that can be efficiently solved by the pretrained neural link predictor. Our proposed Q2T is not only efficient to train, but also modular, thus easily adaptable to various neural link predictors that have been studied well. Extensive experiments demonstrate that, even without explicit modeling for neural set operators, Q2T still achieves state-of-the-art performance on diverse complex queries over three public benchmarks.

{{</citation>}}


### (140/144) Accurate prediction of international trade flows: Leveraging knowledge graphs and their embeddings (Diego Rincon-Yanez et al., 2023)

{{<citation>}}

Diego Rincon-Yanez, Chahinez Ounoughi, Bassem Sellami, Tarmo Kalvet, Marek Tiits, Sabrina Senatore, Sadok Ben Yahia. (2023)  
**Accurate prediction of international trade flows: Leveraging knowledge graphs and their embeddings**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SC, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.11161v1)  

---


**ABSTRACT**  
Knowledge representation (KR) is vital in designing symbolic notations to represent real-world facts and facilitate automated decision-making tasks. Knowledge graphs (KGs) have emerged so far as a popular form of KR, offering a contextual and human-like representation of knowledge. In international economics, KGs have proven valuable in capturing complex interactions between commodities, companies, and countries. By putting the gravity model, which is a common economic framework, into the process of building KGs, important factors that affect trade relationships can be taken into account, making it possible to predict international trade patterns. This paper proposes an approach that leverages Knowledge Graph embeddings for modeling international trade, focusing on link prediction using embeddings. Thus, valuable insights are offered to policymakers, businesses, and economists, enabling them to anticipate the effects of changes in the international trade system. Moreover, the integration of traditional machine learning methods with KG embeddings, such as decision trees and graph neural networks are also explored. The research findings demonstrate the potential for improving prediction accuracy and provide insights into embedding explainability in knowledge representation. The paper also presents a comprehensive analysis of the influence of embedding methods on other intelligent algorithms.

{{</citation>}}


### (141/144) Core Building Blocks: Next Gen Geo Spatial GPT Application (Ashley Fernandez et al., 2023)

{{<citation>}}

Ashley Fernandez, Swaraj Dube. (2023)  
**Core Building Blocks: Next Gen Geo Spatial GPT Application**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.11029v2)  

---


**ABSTRACT**  
This paper proposes MapGPT which is a novel approach that integrates the capabilities of language models, specifically large language models (LLMs), with spatial data processing techniques. This paper introduces MapGPT, which aims to bridge the gap between natural language understanding and spatial data analysis by highlighting the relevant core building blocks. By combining the strengths of LLMs and geospatial analysis, MapGPT enables more accurate and contextually aware responses to location-based queries. The proposed methodology highlights building LLMs on spatial and textual data, utilizing tokenization and vector representations specific to spatial information. The paper also explores the challenges associated with generating spatial vector representations. Furthermore, the study discusses the potential of computational capabilities within MapGPT, allowing users to perform geospatial computations and obtain visualized outputs. Overall, this research paper presents the building blocks and methodology of MapGPT, highlighting its potential to enhance spatial data understanding and generation in natural language processing applications.

{{</citation>}}


## eess.AS (1)



### (142/144) Zipformer: A faster and better encoder for automatic speech recognition (Zengwei Yao et al., 2023)

{{<citation>}}

Zengwei Yao, Liyong Guo, Xiaoyu Yang, Wei Kang, Fangjun Kuang, Yifan Yang, Zengrui Jin, Long Lin, Daniel Povey. (2023)  
**Zipformer: A faster and better encoder for automatic speech recognition**  

---
Primary Category: eess.AS  
Categories: cs-LG, eess-AS, eess.AS  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.11230v1)  

---


**ABSTRACT**  
The Conformer has become the most popular encoder model for automatic speech recognition (ASR). It adds convolution modules to a transformer to learn both local and global dependencies. In this work we describe a faster, more memory-efficient, and better-performing transformer, called Zipformer. Modeling changes include: 1) a U-Net-like encoder structure where middle stacks operate at lower frame rates; 2) reorganized block structure with more modules, within which we re-use attention weights for efficiency; 3) a modified form of LayerNorm called BiasNorm allows us to retain some length information; 4) new activation functions SwooshR and SwooshL work better than Swish. We also propose a new optimizer, called ScaledAdam, which scales the update by each tensor's current scale to keep the relative change about the same, and also explictly learns the parameter scale. It achieves faster convergence and better performance than Adam. Extensive experiments on LibriSpeech, Aishell-1, and WenetSpeech datasets demonstrate the effectiveness of our proposed Zipformer over other state-of-the-art ASR models. Our code is publicly available at https://github.com/k2-fsa/icefall.

{{</citation>}}


## eess.SY (1)



### (143/144) Cooperative Dispatch of Microgrids Community Using Risk-Sensitive Reinforcement Learning with Monotonously Improved Performance (Ziqing Zhu et al., 2023)

{{<citation>}}

Ziqing Zhu, Xiang Gao, Siqi Bu, Ka Wing Chan, Bin Zhou, Shiwei Xia. (2023)  
**Cooperative Dispatch of Microgrids Community Using Risk-Sensitive Reinforcement Learning with Monotonously Improved Performance**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.10997v1)  

---


**ABSTRACT**  
The integration of individual microgrids (MGs) into Microgrid Clusters (MGCs) significantly improves the reliability and flexibility of energy supply, through resource sharing and ensuring backup during outages. The dispatch of MGCs is the key challenge to be tackled to ensure their secure and economic operation. Currently, there is a lack of optimization method that can achieve a trade-off among top-priority requirements of MGCs' dispatch, including fast computation speed, optimality, multiple objectives, and risk mitigation against uncertainty. In this paper, a novel Multi-Objective, Risk-Sensitive, and Online Trust Region Policy Optimization (RS-TRPO) Algorithm is proposed to tackle this problem. First, a dispatch paradigm for autonomous MGs in the MGC is proposed, enabling them sequentially implement their self-dispatch to mitigate potential conflicts. This dispatch paradigm is then formulated as a Markov Game model, which is finally solved by the RS-TRPO algorithm. This online algorithm enables MGs to spontaneously search for the Pareto Frontier considering multiple objectives and risk mitigation. The outstanding computational performance of this algorithm is demonstrated in comparison with mathematical programming methods and heuristic algorithms in a modified IEEE 30-Bus Test System integrated with four autonomous MGs.

{{</citation>}}


## stat.ME (1)



### (144/144) Exact nonlinear state estimation (Hristo G. Chipilski, 2023)

{{<citation>}}

Hristo G. Chipilski. (2023)  
**Exact nonlinear state estimation**  

---
Primary Category: stat.ME  
Categories: cs-CE, cs-LG, math-DS, physics-ao-ph, stat-ME, stat.ME  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.10976v1)  

---


**ABSTRACT**  
The majority of data assimilation (DA) methods in the geosciences are based on Gaussian assumptions. While these assumptions facilitate efficient algorithms, they cause analysis biases and subsequent forecast degradations. Non-parametric, particle-based DA algorithms have superior accuracy, but their application to high-dimensional models still poses operational challenges. Drawing inspiration from recent advances in the field of generative artificial intelligence (AI), this article introduces a new nonlinear estimation theory which attempts to bridge the existing gap in DA methodology. Specifically, a Conjugate Transform Filter (CTF) is derived and shown to generalize the celebrated Kalman filter to arbitrarily non-Gaussian distributions. The new filter has several desirable properties, such as its ability to preserve statistical relationships in the prior state and convergence to highly accurate observations. An ensemble approximation of the new theory (ECTF) is also presented and validated using idealized statistical experiments that feature bounded quantities with non-Gaussian distributions, a prevalent challenge in Earth system models. Results from these experiments indicate that the greatest benefits from ECTF occur when observation errors are small relative to the forecast uncertainty and when state variables exhibit strong nonlinear dependencies. Ultimately, the new filtering theory offers exciting avenues for improving conventional DA algorithms through their principled integration with AI techniques.

{{</citation>}}
