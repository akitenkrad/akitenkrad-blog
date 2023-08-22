---
draft: false
title: "arXiv @ 2023.08.22"
date: 2023-08-22
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.22"
    identifier: arxiv_20230822
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (12)](#cscl-12)
- [cs.SD (1)](#cssd-1)
- [eess.IV (2)](#eessiv-2)
- [cs.AI (2)](#csai-2)
- [cs.IR (1)](#csir-1)
- [cs.SE (1)](#csse-1)
- [cs.CV (19)](#cscv-19)
- [astro-ph.SR (1)](#astro-phsr-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.CE (1)](#csce-1)
- [cs.LG (5)](#cslg-5)
- [cs.NI (1)](#csni-1)
- [cs.CR (2)](#cscr-2)
- [cs.SI (1)](#cssi-1)
- [cs.FL (1)](#csfl-1)
- [cs.AR (1)](#csar-1)
- [cs.MM (1)](#csmm-1)
- [cs.CY (1)](#cscy-1)

## cs.CL (12)



### (1/54) LibriSQA: Pioneering Free-form and Open-ended Spoken Question Answering with a Novel Dataset and Framework (Zihan Zhao et al., 2023)

{{<citation>}}

Zihan Zhao, Yiyang Jiang, Heyang Liu, Yanfeng Wang, Yu Wang. (2023)  
**LibriSQA: Pioneering Free-form and Open-ended Spoken Question Answering with a Novel Dataset and Framework**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.10390v1)  

---


**ABSTRACT**  
While Large Language Models (LLMs) have demonstrated commendable performance across a myriad of domains and tasks, existing LLMs still exhibit a palpable deficit in handling multimodal functionalities, especially for the Spoken Question Answering (SQA) task which necessitates precise alignment and deep interaction between speech and text features. To address the SQA challenge on LLMs, we initially curated the free-form and open-ended LibriSQA dataset from Librispeech, comprising Part I with natural conversational formats and Part II encompassing multiple-choice questions followed by answers and analytical segments. Both parts collectively include 107k SQA pairs that cover various topics. Given the evident paucity of existing speech-text LLMs, we propose a lightweight, end-to-end framework to execute the SQA task on the LibriSQA, witnessing significant results. By reforming ASR into the SQA format, we further substantiate our framework's capability in handling ASR tasks. Our empirical findings bolster the LLMs' aptitude for aligning and comprehending multimodal information, paving the way for the development of universal multimodal LLMs. The dataset and demo can be found at https://github.com/ZihanZhaoSJTU/LibriSQA.

{{</citation>}}


### (2/54) Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language Models (Bilgehan Sel et al., 2023)

{{<citation>}}

Bilgehan Sel, Ahmad Al-Tawaha, Vanshaj Khattar, Lu Wang, Ruoxi Jia, Ming Jin. (2023)  
**Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10379v1)  

---


**ABSTRACT**  
Current literature, aiming to surpass the "Chain-of-Thought" approach, often resorts to an external modus operandi involving halting, modifying, and then resuming the generation process to boost Large Language Models' (LLMs) reasoning capacities. This mode escalates the number of query requests, leading to increased costs, memory, and computational overheads. Addressing this, we propose the Algorithm of Thoughts -- a novel strategy that propels LLMs through algorithmic reasoning pathways, pioneering a new mode of in-context learning. By employing algorithmic examples, we exploit the innate recurrence dynamics of LLMs, expanding their idea exploration with merely one or a few queries. Our technique outperforms earlier single-query methods and stands on par with a recent multi-query strategy that employs an extensive tree search algorithm. Intriguingly, our results suggest that instructing an LLM using an algorithm can lead to performance surpassing that of the algorithm itself, hinting at LLM's inherent ability to weave its intuition into optimized searches. We probe into the underpinnings of our method's efficacy and its nuances in application.

{{</citation>}}


### (3/54) cantnlp@LT-EDI@RANLP-2023: Homophobia/Transphobia Detection in Social Media Comments using Spatio-Temporally Retrained Language Models (Sidney G. -J. Wong et al., 2023)

{{<citation>}}

Sidney G. -J. Wong, Matthew Durward, Benjamin Adams, Jonathan Dunn. (2023)  
**cantnlp@LT-EDI@RANLP-2023: Homophobia/Transphobia Detection in Social Media Comments using Spatio-Temporally Retrained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, NLP, Social Media  
[Paper Link](http://arxiv.org/abs/2308.10370v1)  

---


**ABSTRACT**  
This paper describes our multiclass classification system developed as part of the LTEDI@RANLP-2023 shared task. We used a BERT-based language model to detect homophobic and transphobic content in social media comments across five language conditions: English, Spanish, Hindi, Malayalam, and Tamil. We retrained a transformer-based crosslanguage pretrained language model, XLMRoBERTa, with spatially and temporally relevant social media language data. We also retrained a subset of models with simulated script-mixed social media language data with varied performance. We developed the best performing seven-label classification system for Malayalam based on weighted macro averaged F1 score (ranked first out of six) with variable performance for other language and class-label conditions. We found the inclusion of this spatio-temporal data improved the classification performance for all language and task conditions when compared with the baseline. The results suggests that transformer-based language classification systems are sensitive to register-specific and language-specific retraining.

{{</citation>}}


### (4/54) A Study on Robustness and Reliability of Large Language Model Code Generation (Li Zhong et al., 2023)

{{<citation>}}

Li Zhong, Zilong Wang. (2023)  
**A Study on Robustness and Reliability of Large Language Model Code Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SE, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.10335v1)  

---


**ABSTRACT**  
Recently, the large language models (LLMs) have shown extraordinary ability in understanding natural language and generating programming code. It has been a common practice of software engineers to consult LLMs when encountering coding questions. Although efforts have been made to avoid syntax errors and align the code with the intended semantics, the reliability and robustness of the code generationfrom LLMs have not yet been thoroughly studied. The executable code is not equivalent to the reliable and robust code, especially in the context of real-world software development.The misuse of APIs in the generated code could lead to severe problem, such as resource leaks, program crashes, etc.To make things worse, the users of LLM code generation services are actually the developers that are most vulnerable to these code that seems right -- They are always novice developers that are not familiar with the APIs that LLMs generate code for them. Therefore, they could hardly tell the misuse in the code generated by LLMs, which further facilitates the incorrect code applied in real-world software. Existing code evaluation benchmark and datasets focus on crafting small tasks such as programming questions in coding interviews, which however deviates from the problem that developers would ask LLM for real-world coding help. To fill the missing piece, in this work, we propose a dataset RobustAPI for evaluating the reliability and robustness of code generated by LLMs. We collect 1208 coding questions from StackOverflow on 24 representative Java APIs. We summarize thecommon misuse patterns of these APIs and evaluate them oncurrent popular LLMs. The evaluation results show that evenfor GPT-4, 62% of the generated code contains API misuses,which would cause unexpected consequences if the code isintroduced into real-world software.

{{</citation>}}


### (5/54) CharacterChat: Learning towards Conversational AI with Personalized Social Support (Quan Tu et al., 2023)

{{<citation>}}

Quan Tu, Chuanqi Chen, Jinpeng Li, Yanran Li, Shuo Shang, Dongyan Zhao, Ran Wang, Rui Yan. (2023)  
**CharacterChat: Learning towards Conversational AI with Personalized Social Support**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10278v1)  

---


**ABSTRACT**  
In our modern, fast-paced, and interconnected world, the importance of mental well-being has grown into a matter of great urgency. However, traditional methods such as Emotional Support Conversations (ESC) face challenges in effectively addressing a diverse range of individual personalities. In response, we introduce the Social Support Conversation (S2Conv) framework. It comprises a series of support agents and the interpersonal matching mechanism, linking individuals with persona-compatible virtual supporters. Utilizing persona decomposition based on the MBTI (Myers-Briggs Type Indicator), we have created the MBTI-1024 Bank, a group that of virtual characters with distinct profiles. Through improved role-playing prompts with behavior preset and dynamic memory, we facilitate the development of the MBTI-S2Conv dataset, which contains conversations between the characters in the MBTI-1024 Bank. Building upon these foundations, we present CharacterChat, a comprehensive S2Conv system, which includes a conversational model driven by personas and memories, along with an interpersonal matching plugin model that dispatches the optimal supporters from the MBTI-1024 Bank for individuals with specific personas. Empirical results indicate the remarkable efficacy of CharacterChat in providing personalized social support and highlight the substantial advantages derived from interpersonal matching. The source code is available in \url{https://github.com/morecry/CharacterChat}.

{{</citation>}}


### (6/54) Scaled-up Discovery of Latent Concepts in Deep NLP Models (Majd Hawasly et al., 2023)

{{<citation>}}

Majd Hawasly, Fahim Dalvi, Nadir Durrani. (2023)  
**Scaled-up Discovery of Latent Concepts in Deep NLP Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.10263v1)  

---


**ABSTRACT**  
Pre-trained language models (pLMs) learn intricate patterns and contextual dependencies via unsupervised learning on vast text data, driving breakthroughs across NLP tasks. Despite these achievements, these models remain black boxes, necessitating research into understanding their decision-making processes. Recent studies explore representation analysis by clustering latent spaces within pre-trained models. However, these approaches are limited in terms of scalability and the scope of interpretation because of high computation costs of clustering algorithms. This study focuses on comparing clustering algorithms for the purpose of scaling encoded concept discovery of representations from pLMs. Specifically, we compare three algorithms in their capacity to unveil the encoded concepts through their alignment to human-defined ontologies: Agglomerative Hierarchical Clustering, Leaders Algorithm, and K-Means Clustering. Our results show that K-Means has the potential to scale to very large datasets, allowing rich latent concept discovery, both on the word and phrase level.

{{</citation>}}


### (7/54) How Good Are Large Language Models at Out-of-Distribution Detection? (Bo Liu et al., 2023)

{{<citation>}}

Bo Liu, Liming Zhan, Zexin Lu, Yujie Feng, Lei Xue, Xiao-Ming Wu. (2023)  
**How Good Are Large Language Models at Out-of-Distribution Detection?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, LLaMA, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10261v1)  

---


**ABSTRACT**  
Out-of-distribution (OOD) detection plays a vital role in enhancing the reliability of machine learning (ML) models. The emergence of large language models (LLMs) has catalyzed a paradigm shift within the ML community, showcasing their exceptional capabilities across diverse natural language processing tasks. While existing research has probed OOD detection with smaller encoder-based Transformers like BERT and RoBERTa, the stark differences in scales, pre-training objectives, and inference paradigms call into question the applicability of these findings to LLMs. This paper embarks on a pioneering empirical investigation of OOD detection in the domain of LLMs, focusing on LLaMA series ranging from 7B to 65B in size. We thoroughly evaluate commonly-used OOD detectors, scrutinizing their performance in both zero-grad and fine-tuning scenarios. Notably, we alter previous discriminative in-distribution fine-tuning into generative fine-tuning, aligning the pre-training objective of LLMs with downstream tasks. Our findings unveil that a simple cosine distance OOD detector demonstrates superior efficacy, outperforming other OOD detectors. We provide an intriguing explanation for this phenomenon by highlighting the isotropic nature of the embedding spaces of LLMs, which distinctly contrasts with the anisotropic property observed in smaller BERT family models. The new insight enhances our understanding of how LLMs detect OOD data, thereby enhancing their adaptability and reliability in dynamic environments.

{{</citation>}}


### (8/54) LMTuner: An user-friendly and highly-integrable Training Framework for fine-tuning Large Language Models (Yixuan Weng et al., 2023)

{{<citation>}}

Yixuan Weng, Zhiqi Wang, Huanxuan Liao, Shizhu He, Shengping Liu, Kang Liu, Jun Zhao. (2023)  
**LMTuner: An user-friendly and highly-integrable Training Framework for fine-tuning Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10252v1)  

---


**ABSTRACT**  
With the burgeoning development in the realm of large language models (LLMs), the demand for efficient incremental training tailored to specific industries and domains continues to increase. Currently, the predominantly employed frameworks lack modular design, it often takes a lot of coding work to kickstart the training of LLM. To address this, we present "LMTuner", a highly usable, integrable, and scalable system for training LLMs expeditiously and with minimal user-input. LMTuner comprises three main modules - the Interaction, Training, and Inference Modules. We advocate that LMTuner's usability and integrality alleviate the complexities in training large language models. Remarkably, even a novice user could commence training large language models within five minutes. Furthermore, it integrates DeepSpeed frameworks and supports Efficient Fine-Tuning methodologies like Low Rank Adaptation (LoRA), Quantized LoRA (QLoRA), etc., enabling the training of language models scaling from 300M to a whopping 130B parameters using a single server. The LMTuner's homepage (https://wengsyx.github.io/LMTuner/)and screencast video (https://youtu.be/nsXmWOmN3rE) are now publicly available.

{{</citation>}}


### (9/54) Activation Addition: Steering Language Models Without Optimization (Alex Turner et al., 2023)

{{<citation>}}

Alex Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, Monte MacDiarmid. (2023)  
**Activation Addition: Steering Language Models Without Optimization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.10248v1)  

---


**ABSTRACT**  
Reliably controlling the behavior of large language models (LLMs) is a pressing open problem. Existing methods include supervised finetuning, reinforcement learning from human feedback (RLHF), prompt engineering and guided decoding. We instead investigate activation engineering: modifying activations at inference time to predictably alter model behavior. In particular, we bias the forward pass with an added 'steering vector' implicitly specified through natural language.   Unlike past work which learned these steering vectors (Subramani, Suresh, and Peters 2022; Hernandez, Li, and Andreas 2023), our Activation Addition (ActAdd) method computes them by taking the activation differences that result from pairs of prompts. We demonstrate ActAdd on GPT-2 on OpenWebText and ConceptNet. Our inference-time approach yields control over high-level properties of output and preserves off-target model performance. It involves far less compute and implementation effort compared to finetuning or RLHF, allows users to provide natural language specifications, and its overhead scales naturally with model size.

{{</citation>}}


### (10/54) FoodGPT: A Large Language Model in Food Testing Domain with Incremental Pre-training and Knowledge Graph Prompt (Zhixiao Qi et al., 2023)

{{<citation>}}

Zhixiao Qi, Yijiong Yu, Meiqi Tu, Junyi Tan, Yongfeng Huang. (2023)  
**FoodGPT: A Large Language Model in Food Testing Domain with Incremental Pre-training and Knowledge Graph Prompt**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2308.10173v1)  

---


**ABSTRACT**  
Currently, the construction of large language models in specific domains is done by fine-tuning on a base model. Some models also incorporate knowledge bases without the need for pre-training. This is because the base model already contains domain-specific knowledge during the pre-training process. We build a large language model for food testing. Unlike the above approach, a significant amount of data in this domain exists in Scanning format for domain standard documents. In addition, there is a large amount of untrained structured knowledge. Therefore, we introduce an incremental pre-training step to inject this knowledge into a large language model. In this paper, we propose a method for handling structured knowledge and scanned documents in incremental pre-training. To overcome the problem of machine hallucination, we constructe a knowledge graph to serve as an external knowledge base for supporting retrieval in the large language model. It is worth mentioning that this paper is a technical report of our pre-release version, and we will report our specific experimental data in future versions.

{{</citation>}}


### (11/54) Head-to-Tail: How Knowledgeable are Large Language Models (LLM)? A.K.A. Will LLMs Replace Knowledge Graphs? (Kai Sun et al., 2023)

{{<citation>}}

Kai Sun, Yifan Ethan Xu, Hanwen Zha, Yue Liu, Xin Luna Dong. (2023)  
**Head-to-Tail: How Knowledgeable are Large Language Models (LLM)? A.K.A. Will LLMs Replace Knowledge Graphs?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2308.10168v1)  

---


**ABSTRACT**  
Since the recent prosperity of Large Language Models (LLMs), there have been interleaved discussions regarding how to reduce hallucinations from LLM responses, how to increase the factuality of LLMs, and whether Knowledge Graphs (KGs), which store the world knowledge in a symbolic form, will be replaced with LLMs. In this paper, we try to answer these questions from a new angle: How knowledgeable are LLMs?   To answer this question, we constructed Head-to-Tail, a benchmark that consists of 18K question-answer (QA) pairs regarding head, torso, and tail facts in terms of popularity. We designed an automated evaluation method and a set of metrics that closely approximate the knowledge an LLM confidently internalizes. Through a comprehensive evaluation of 14 publicly available LLMs, we show that existing LLMs are still far from being perfect in terms of their grasp of factual knowledge, especially for facts of torso-to-tail entities.

{{</citation>}}


### (12/54) A Survey on Fairness in Large Language Models (Yingji Li et al., 2023)

{{<citation>}}

Yingji Li, Mengnan Du, Rui Song, Xin Wang, Ying Wang. (2023)  
**A Survey on Fairness in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10149v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown powerful performance and development prospect and are widely deployed in the real world. However, LLMs can capture social biases from unprocessed training data and propagate the biases to downstream tasks. Unfair LLM systems have undesirable social impacts and potential harms. In this paper, we provide a comprehensive review of related research on fairness in LLMs. First, for medium-scale LLMs, we introduce evaluation metrics and debiasing methods from the perspectives of intrinsic bias and extrinsic bias, respectively. Then, for large-scale LLMs, we introduce recent fairness research, including fairness evaluation, reasons for bias, and debiasing methods. Finally, we discuss and provide insight on the challenges and future directions for the development of fairness in LLMs.

{{</citation>}}


## cs.SD (1)



### (13/54) Neural Architectures Learning Fourier Transforms, Signal Processing and Much More.... (Prateek Verma, 2023)

{{<citation>}}

Prateek Verma. (2023)  
**Neural Architectures Learning Fourier Transforms, Signal Processing and Much More....**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2308.10388v1)  

---


**ABSTRACT**  
This report will explore and answer fundamental questions about taking Fourier Transforms and tying it with recent advances in AI and neural architecture. One interpretation of the Fourier Transform is decomposing a signal into its constituent components by projecting them onto complex exponentials. Variants exist, such as discrete cosine transform that does not operate on the complex domain and projects an input signal to only cosine functions oscillating at different frequencies. However, this is a fundamental limitation, and it needs to be more suboptimal. The first one is that all kernels are sinusoidal: What if we could have some kernels adapted or learned according to the problem? What if we can use neural architectures for this? We show how one can learn these kernels from scratch for audio signal processing applications. We find that the neural architecture not only learns sinusoidal kernel shapes but discovers all kinds of incredible signal-processing properties. E.g., windowing functions, onset detectors, high pass filters, low pass filters, modulations, etc. Further, upon analysis of the filters, we find that the neural architecture has a comb filter-like structure on top of the learned kernels. Comb filters that allow harmonic frequencies to pass through are one of the core building blocks/types of filters similar to high-pass, low-pass, and band-pass filters of various traditional signal processing algorithms. Further, we can also use the convolution operation with a signal to be learned from scratch, and we will explore papers in the literature that uses this with that robust Transformer architectures. Further, we would also explore making the learned kernel's content adaptive, i.e., learning different kernels for different inputs.

{{</citation>}}


## eess.IV (2)



### (14/54) Developing a Machine Learning-Based Clinical Decision Support Tool for Uterine Tumor Imaging (Darryl E. Wright et al., 2023)

{{<citation>}}

Darryl E. Wright, Adriana V. Gregory, Deema Anaam, Sepideh Yadollahi, Sumana Ramanathan, Kafayat A. Oyemade, Reem Alsibai, Heather Holmes, Harrison Gottlich, Cherie-Akilah G. Browne, Sarah L. Cohen Rassier, Isabel Green, Elizabeth A. Stewart, Hiroaki Takahashi, Bohyun Kim, Shannon Laughlin-Tommaso, Timothy L. Kline. (2023)  
**Developing a Machine Learning-Based Clinical Decision Support Tool for Uterine Tumor Imaging**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV, q-bio-QM  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2308.10372v1)  

---


**ABSTRACT**  
Uterine leiomyosarcoma (LMS) is a rare but aggressive malignancy. On imaging, it is difficult to differentiate LMS from, for example, degenerated leiomyoma (LM), a prevalent but benign condition. We curated a data set of 115 axial T2-weighted MRI images from 110 patients (mean [range] age=45 [17-81] years) with UTs that included five different tumor types. These data were randomly split stratifying on tumor volume into training (n=85) and test sets (n=30). An independent second reader (reader 2) provided manual segmentations for all test set images. To automate segmentation, we applied nnU-Net and explored the effect of training set size on performance by randomly generating subsets with 25, 45, 65 and 85 training set images. We evaluated the ability of radiomic features to distinguish between types of UT individually and when combined through feature selection and machine learning. Using the entire training set the mean [95% CI] fibroid DSC was measured as 0.87 [0.59-1.00] and the agreement between the two readers was 0.89 [0.77-1.0] on the test set. When classifying degenerated LM from LMS we achieve a test set F1-score of 0.80. Classifying UTs based on radiomic features we identify classifiers achieving F1-scores of 0.53 [0.45, 0.61] and 0.80 [0.80, 0.80] on the test set for the benign versus malignant, and degenerated LM versus LMS tasks. We show that it is possible to develop an automated method for 3D segmentation of the uterus and UT that is close to human-level performance with fewer than 150 annotated images. For distinguishing UT types, while we train models that merit further investigation with additional data, reliable automatic differentiation of UTs remains a challenge.

{{</citation>}}


### (15/54) Polymerized Feature-based Domain Adaptation for Cervical Cancer Dose Map Prediction (Jie Zeng et al., 2023)

{{<citation>}}

Jie Zeng, Zeyu Han, Xingchen Peng, Jianghong Xiao, Peng Wang, Yan Wang. (2023)  
**Polymerized Feature-based Domain Adaptation for Cervical Cancer Dose Map Prediction**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10142v1)  

---


**ABSTRACT**  
Recently, deep learning (DL) has automated and accelerated the clinical radiation therapy (RT) planning significantly by predicting accurate dose maps. However, most DL-based dose map prediction methods are data-driven and not applicable for cervical cancer where only a small amount of data is available. To address this problem, this paper proposes to transfer the rich knowledge learned from another cancer, i.e., rectum cancer, which has the same scanning area and more clinically available data, to improve the dose map prediction performance for cervical cancer through domain adaptation. In order to close the congenital domain gap between the source (i.e., rectum cancer) and the target (i.e., cervical cancer) domains, we develop an effective Transformer-based polymerized feature module (PFM), which can generate an optimal polymerized feature distribution to smoothly align the two input distributions. Experimental results on two in-house clinical datasets demonstrate the superiority of the proposed method compared with state-of-the-art methods.

{{</citation>}}


## cs.AI (2)



### (16/54) Imaginations of WALL-E : Reconstructing Experiences with an Imagination-Inspired Module for Advanced AI Systems (Zeinab Sadat Taghavi et al., 2023)

{{<citation>}}

Zeinab Sadat Taghavi, Soroush Gooran, Seyed Arshan Dalili, Hamidreza Amirzadeh, Mohammad Jalal Nematbakhsh, Hossein Sameti. (2023)  
**Imaginations of WALL-E : Reconstructing Experiences with an Imagination-Inspired Module for Advanced AI Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2308.10354v1)  

---


**ABSTRACT**  
In this paper, we introduce a novel Artificial Intelligence (AI) system inspired by the philosophical and psychoanalytical concept of imagination as a ``Re-construction of Experiences". Our AI system is equipped with an imagination-inspired module that bridges the gap between textual inputs and other modalities, enriching the derived information based on previously learned experiences. A unique feature of our system is its ability to formulate independent perceptions of inputs. This leads to unique interpretations of a concept that may differ from human interpretations but are equally valid, a phenomenon we term as ``Interpretable Misunderstanding". We employ large-scale models, specifically a Multimodal Large Language Model (MLLM), enabling our proposed system to extract meaningful information across modalities while primarily remaining unimodal. We evaluated our system against other large language models across multiple tasks, including emotion recognition and question-answering, using a zero-shot methodology to ensure an unbiased scenario that may happen by fine-tuning. Significantly, our system outperformed the best Large Language Models (LLM) on the MELD, IEMOCAP, and CoQA datasets, achieving Weighted F1 (WF1) scores of 46.74%, 25.23%, and Overall F1 (OF1) score of 17%, respectively, compared to 22.89%, 12.28%, and 7% from the well-performing LLM. The goal is to go beyond the statistical view of language processing and tie it to human concepts such as philosophy and psychoanalysis. This work represents a significant advancement in the development of imagination-inspired AI systems, opening new possibilities for AI to generate deep and interpretable information across modalities, thereby enhancing human-AI interaction.

{{</citation>}}


### (17/54) A Review on Objective-Driven Artificial Intelligence (Apoorv Singh, 2023)

{{<citation>}}

Apoorv Singh. (2023)  
**A Review on Objective-Driven Artificial Intelligence**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10135v1)  

---


**ABSTRACT**  
While advancing rapidly, Artificial Intelligence still falls short of human intelligence in several key aspects due to inherent limitations in current AI technologies and our understanding of cognition. Humans have an innate ability to understand context, nuances, and subtle cues in communication, which allows us to comprehend jokes, sarcasm, and metaphors. Machines struggle to interpret such contextual information accurately. Humans possess a vast repository of common-sense knowledge that helps us make logical inferences and predictions about the world. Machines lack this innate understanding and often struggle with making sense of situations that humans find trivial. In this article, we review the prospective Machine Intelligence candidates, a review from Prof. Yann LeCun, and other work that can help close this gap between human and machine intelligence. Specifically, we talk about what's lacking with the current AI techniques such as supervised learning, reinforcement learning, self-supervised learning, etc. Then we show how Hierarchical planning-based approaches can help us close that gap and deep-dive into energy-based, latent-variable methods and Joint embedding predictive architecture methods.

{{</citation>}}


## cs.IR (1)



### (18/54) Enhancing Transformers without Self-supervised Learning: A Loss Landscape Perspective in Sequential Recommendation (Vivian Lai et al., 2023)

{{<citation>}}

Vivian Lai, Huiyuan Chen, Chin-Chia Michael Yeh, Minghua Xu, Yiwei Cai, Hao Yang. (2023)  
**Enhancing Transformers without Self-supervised Learning: A Loss Landscape Perspective in Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10347v1)  

---


**ABSTRACT**  
Transformer and its variants are a powerful class of architectures for sequential recommendation, owing to their ability of capturing a user's dynamic interests from their past interactions. Despite their success, Transformer-based models often require the optimization of a large number of parameters, making them difficult to train from sparse data in sequential recommendation. To address the problem of data sparsity, previous studies have utilized self-supervised learning to enhance Transformers, such as pre-training embeddings from item attributes or contrastive data augmentations. However, these approaches encounter several training issues, including initialization sensitivity, manual data augmentations, and large batch-size memory bottlenecks.   In this work, we investigate Transformers from the perspective of loss geometry, aiming to enhance the models' data efficiency and generalization in sequential recommendation. We observe that Transformers (e.g., SASRec) can converge to extremely sharp local minima if not adequately regularized. Inspired by the recent Sharpness-Aware Minimization (SAM), we propose SAMRec, which significantly improves the accuracy and robustness of sequential recommendation. SAMRec performs comparably to state-of-the-art self-supervised Transformers, such as S$^3$Rec and CL4SRec, without the need for pre-training or strong data augmentations.

{{</citation>}}


## cs.SE (1)



### (19/54) Can Large Language Models Find And Fix Vulnerable Software? (David Noever, 2023)

{{<citation>}}

David Noever. (2023)  
**Can Large Language Models Find And Fix Vulnerable Software?**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.10345v1)  

---


**ABSTRACT**  
In this study, we evaluated the capability of Large Language Models (LLMs), particularly OpenAI's GPT-4, in detecting software vulnerabilities, comparing their performance against traditional static code analyzers like Snyk and Fortify. Our analysis covered numerous repositories, including those from NASA and the Department of Defense. GPT-4 identified approximately four times the vulnerabilities than its counterparts. Furthermore, it provided viable fixes for each vulnerability, demonstrating a low rate of false positives. Our tests encompassed 129 code samples across eight programming languages, revealing the highest vulnerabilities in PHP and JavaScript. GPT-4's code corrections led to a 90% reduction in vulnerabilities, requiring only an 11% increase in code lines. A critical insight was LLMs' ability to self-audit, suggesting fixes for their identified vulnerabilities and underscoring their precision. Future research should explore system-level vulnerabilities and integrate multiple static code analyzers for a holistic perspective on LLMs' potential.

{{</citation>}}


## cs.CV (19)



### (20/54) Coordinate Transformer: Achieving Single-stage Multi-person Mesh Recovery from Videos (Haoyuan Li et al., 2023)

{{<citation>}}

Haoyuan Li, Haoye Dong, Hanchao Jia, Dong Huang, Michael C. Kampffmeyer, Liang Lin, Xiaodan Liang. (2023)  
**Coordinate Transformer: Achieving Single-stage Multi-person Mesh Recovery from Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.10334v1)  

---


**ABSTRACT**  
Multi-person 3D mesh recovery from videos is a critical first step towards automatic perception of group behavior in virtual reality, physical therapy and beyond. However, existing approaches rely on multi-stage paradigms, where the person detection and tracking stages are performed in a multi-person setting, while temporal dynamics are only modeled for one person at a time. Consequently, their performance is severely limited by the lack of inter-person interactions in the spatial-temporal mesh recovery, as well as by detection and tracking defects. To address these challenges, we propose the Coordinate transFormer (CoordFormer) that directly models multi-person spatial-temporal relations and simultaneously performs multi-mesh recovery in an end-to-end manner. Instead of partitioning the feature map into coarse-scale patch-wise tokens, CoordFormer leverages a novel Coordinate-Aware Attention to preserve pixel-level spatial-temporal coordinate information. Additionally, we propose a simple, yet effective Body Center Attention mechanism to fuse position information. Extensive experiments on the 3DPW dataset demonstrate that CoordFormer significantly improves the state-of-the-art, outperforming the previously best results by 4.2%, 8.8% and 4.7% according to the MPJPE, PAMPJPE, and PVE metrics, respectively, while being 40% faster than recent video-based approaches. The released code can be found at https://github.com/Li-Hao-yuan/CoordFormer.

{{</citation>}}


### (21/54) Improving Adversarial Robustness of Masked Autoencoders via Test-time Frequency-domain Prompting (Qidong Huang et al., 2023)

{{<citation>}}

Qidong Huang, Xiaoyi Dong, Dongdong Chen, Yinpeng Chen, Lu Yuan, Gang Hua, Weiming Zhang, Nenghai Yu. (2023)  
**Improving Adversarial Robustness of Masked Autoencoders via Test-time Frequency-domain Prompting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BERT, ImageNet  
[Paper Link](http://arxiv.org/abs/2308.10315v1)  

---


**ABSTRACT**  
In this paper, we investigate the adversarial robustness of vision transformers that are equipped with BERT pretraining (\eg, BEiT, MAE). A surprising observation is that MAE has significantly worse adversarial robustness than other BERT pretraining methods. This observation drives us to rethink the basic differences between these BERT pretraining methods and how these differences affect the robustness against adversarial perturbations. Our empirical analysis reveals that the adversarial robustness of BERT pretraining is highly related to the reconstruction target, \ie, predicting the raw pixels of masked image patches will degrade more adversarial robustness of the model than predicting the semantic context, since it guides the model to concentrate more on medium-/high-frequency components of images. Based on our analysis, we provide a simple yet effective way to boost the adversarial robustness of MAE. The basic idea is using the dataset-extracted domain knowledge to occupy the medium-/high-frequency of images, thus narrowing the optimization space of adversarial perturbations. Specifically, we group the distribution of pretraining data and optimize a set of cluster-specific visual prompts on frequency domain. These prompts are incorporated with input images through prototype-based prompt selection during test period. Extensive evaluation shows that our method clearly boost MAE's adversarial robustness while maintaining its clean performance on ImageNet-1k classification. Our code is available at: \href{https://github.com/shikiw/RobustMAE}{https://github.com/shikiw/RobustMAE}.

{{</citation>}}


### (22/54) Representation Disparity-aware Distillation for 3D Object Detection (Yanjing Li et al., 2023)

{{<citation>}}

Yanjing Li, Sheng Xu, Mingbao Lin, Jihao Yin, Baochang Zhang, Xianbin Cao. (2023)  
**Representation Disparity-aware Distillation for 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.10308v1)  

---


**ABSTRACT**  
In this paper, we focus on developing knowledge distillation (KD) for compact 3D detectors. We observe that off-the-shelf KD methods manifest their efficacy only when the teacher model and student counterpart share similar intermediate feature representations. This might explain why they are less effective in building extreme-compact 3D detectors where significant representation disparity arises due primarily to the intrinsic sparsity and irregularity in 3D point clouds. This paper presents a novel representation disparity-aware distillation (RDD) method to address the representation disparity issue and reduce performance gap between compact students and over-parameterized teachers. This is accomplished by building our RDD from an innovative perspective of information bottleneck (IB), which can effectively minimize the disparity of proposal region pairs from student and teacher in features and logits. Extensive experiments are performed to demonstrate the superiority of our RDD over existing KD methods. For example, our RDD increases mAP of CP-Voxel-S to 57.1% on nuScenes dataset, which even surpasses teacher performance while taking up only 42% FLOPs.

{{</citation>}}


### (23/54) Boosting Adversarial Transferability by Block Shuffle and Rotation (Kunyu Wang et al., 2023)

{{<citation>}}

Kunyu Wang, Xuanran He, Wenxuan Wang, Xiaosen Wang. (2023)  
**Boosting Adversarial Transferability by Block Shuffle and Rotation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.10299v1)  

---


**ABSTRACT**  
Adversarial examples mislead deep neural networks with imperceptible perturbations and have brought significant threats to deep learning. An important aspect is their transferability, which refers to their ability to deceive other models, thus enabling attacks in the black-box setting. Though various methods have been proposed to boost transferability, the performance still falls short compared with white-box attacks. In this work, we observe that existing input transformation based attacks, one of the mainstream transfer-based attacks, result in different attention heatmaps on various models, which might limit the transferability. We also find that breaking the intrinsic relation of the image can disrupt the attention heatmap of the original image. Based on this finding, we propose a novel input transformation based attack called block shuffle and rotation (BSR). Specifically, BSR splits the input image into several blocks, then randomly shuffles and rotates these blocks to construct a set of new images for gradient calculation. Empirical evaluations on the ImageNet dataset demonstrate that BSR could achieve significantly better transferability than the existing input transformation based methods under single-model and ensemble-model settings. Combining BSR with the current input transformation method can further improve the transferability, which significantly outperforms the state-of-the-art methods.

{{</citation>}}


### (24/54) MacFormer: Map-Agent Coupled Transformer for Real-time and Robust Trajectory Prediction (Chen Feng et al., 2023)

{{<citation>}}

Chen Feng, Hangning Zhou, Huadong Lin, Zhigang Zhang, Ziyao Xu, Chi Zhang, Boyu Zhou, Shaojie Shen. (2023)  
**MacFormer: Map-Agent Coupled Transformer for Real-time and Robust Trajectory Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10280v1)  

---


**ABSTRACT**  
Predicting the future behavior of agents is a fundamental task in autonomous vehicle domains. Accurate prediction relies on comprehending the surrounding map, which significantly regularizes agent behaviors. However, existing methods have limitations in exploiting the map and exhibit a strong dependence on historical trajectories, which yield unsatisfactory prediction performance and robustness. Additionally, their heavy network architectures impede real-time applications. To tackle these problems, we propose Map-Agent Coupled Transformer (MacFormer) for real-time and robust trajectory prediction. Our framework explicitly incorporates map constraints into the network via two carefully designed modules named coupled map and reference extractor. A novel multi-task optimization strategy (MTOS) is presented to enhance learning of topology and rule constraints. We also devise bilateral query scheme in context fusion for a more efficient and lightweight network. We evaluated our approach on Argoverse 1, Argoverse 2, and nuScenes real-world benchmarks, where it all achieved state-of-the-art performance with the lowest inference latency and smallest model size. Experiments also demonstrate that our framework is resilient to imperfect tracklet inputs. Furthermore, we show that by combining with our proposed strategies, classical models outperform their baselines, further validating the versatility of our framework.

{{</citation>}}


### (25/54) Turning Waste into Wealth: Leveraging Low-Quality Samples for Enhancing Continuous Conditional Generative Adversarial Networks (Xin Ding et al., 2023)

{{<citation>}}

Xin Ding, Yongwei Wang, Zuheng Xu. (2023)  
**Turning Waste into Wealth: Leveraging Low-Quality Samples for Enhancing Continuous Conditional Generative Adversarial Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.10273v1)  

---


**ABSTRACT**  
Continuous Conditional Generative Adversarial Networks (CcGANs) enable generative modeling conditional on continuous scalar variables (termed regression labels). However, they can produce subpar fake images due to limited training data. Although Negative Data Augmentation (NDA) effectively enhances unconditional and class-conditional GANs by introducing anomalies into real training images, guiding the GANs away from low-quality outputs, its impact on CcGANs is limited, as it fails to replicate negative samples that may occur during the CcGAN sampling. We present a novel NDA approach called Dual-NDA specifically tailored for CcGANs to address this problem. Dual-NDA employs two types of negative samples: visually unrealistic images generated from a pre-trained CcGAN and label-inconsistent images created by manipulating real images' labels. Leveraging these negative samples, we introduce a novel discriminator objective alongside a modified CcGAN training algorithm. Empirical analysis on UTKFace and Steering Angle reveals that Dual-NDA consistently enhances the visual fidelity and label consistency of fake images generated by CcGANs, exhibiting a substantial performance gain over the vanilla NDA. Moreover, by applying Dual-NDA, CcGANs demonstrate a remarkable advancement beyond the capabilities of state-of-the-art conditional GANs and diffusion models, establishing a new pinnacle of performance.

{{</citation>}}


### (26/54) Learning Disentangled Representation with Mutual Information Maximization for Real-Time UAV Tracking (Xucheng Wang et al., 2023)

{{<citation>}}

Xucheng Wang, Xiangyang Yang, Hengzhou Ye, Shuiwang Li. (2023)  
**Learning Disentangled Representation with Mutual Information Maximization for Real-Time UAV Tracking**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2308.10262v1)  

---


**ABSTRACT**  
Efficiency has been a critical problem in UAV tracking due to limitations in computation resources, battery capacity, and unmanned aerial vehicle maximum load. Although discriminative correlation filters (DCF)-based trackers prevail in this field for their favorable efficiency, some recently proposed lightweight deep learning (DL)-based trackers using model compression demonstrated quite remarkable CPU efficiency as well as precision. Unfortunately, the model compression methods utilized by these works, though simple, are still unable to achieve satisfying tracking precision with higher compression rates. This paper aims to exploit disentangled representation learning with mutual information maximization (DR-MIM) to further improve DL-based trackers' precision and efficiency for UAV tracking. The proposed disentangled representation separates the feature into an identity-related and an identity-unrelated features. Only the latter is used, which enhances the effectiveness of the feature representation for subsequent classification and regression tasks. Extensive experiments on four UAV benchmarks, including UAV123@10fps, DTB70, UAVDT and VisDrone2018, show that our DR-MIM tracker significantly outperforms state-of-the-art UAV tracking methods.

{{</citation>}}


### (27/54) StableLLaVA: Enhanced Visual Instruction Tuning with Synthesized Image-Dialogue Data (Yanda Li et al., 2023)

{{<citation>}}

Yanda Li, Chi Zhang, Gang Yu, Zhibin Wang, Bin Fu, Guosheng Lin, Chunhua Shen, Ling Chen, Yunchao Wei. (2023)  
**StableLLaVA: Enhanced Visual Instruction Tuning with Synthesized Image-Dialogue Data**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: AI, ChatGPT, Dialog, Dialogue, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.10253v1)  

---


**ABSTRACT**  
The remarkable multimodal capabilities demonstrated by OpenAI's GPT-4 have sparked significant interest in the development of multimodal Large Language Models (LLMs). A primary research objective of such models is to align visual and textual modalities effectively while comprehending human instructions. Current methodologies often rely on annotations derived from benchmark datasets to construct image-dialogue datasets for training purposes, akin to instruction tuning in LLMs. However, these datasets often exhibit domain bias, potentially constraining the generative capabilities of the models. In an effort to mitigate these limitations, we propose a novel data collection methodology that synchronously synthesizes images and dialogues for visual instruction tuning. This approach harnesses the power of generative models, marrying the abilities of ChatGPT and text-to-image generative models to yield a diverse and controllable dataset with varied image content. This not only provides greater flexibility compared to existing methodologies but also significantly enhances several model capabilities. Our research includes comprehensive experiments conducted on various datasets using the open-source LLAVA model as a testbed for our proposed pipeline. Our results underscore marked enhancements across more than ten commonly assessed capabilities,

{{</citation>}}


### (28/54) Generic Attention-model Explainability by Weighted Relevance Accumulation (Yiming Huang et al., 2023)

{{<citation>}}

Yiming Huang, Aozhe Jia, Xiaodan Zhang, Jiawei Zhang. (2023)  
**Generic Attention-model Explainability by Weighted Relevance Accumulation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.10240v1)  

---


**ABSTRACT**  
Attention-based transformer models have achieved remarkable progress in multi-modal tasks, such as visual question answering. The explainability of attention-based methods has recently attracted wide interest as it can explain the inner changes of attention tokens by accumulating relevancy across attention layers. Current methods simply update relevancy by equally accumulating the token relevancy before and after the attention processes. However, the importance of token values is usually different during relevance accumulation. In this paper, we propose a weighted relevancy strategy, which takes the importance of token values into consideration, to reduce distortion when equally accumulating relevance. To evaluate our method, we propose a unified CLIP-based two-stage model, named CLIPmapper, to process Vision-and-Language tasks through CLIP encoder and a following mapper. CLIPmapper consists of self-attention, cross-attention, single-modality, and cross-modality attention, thus it is more suitable for evaluating our generic explainability method. Extensive perturbation tests on visual question answering and image captioning validate that our explainability method outperforms existing methods.

{{</citation>}}


### (29/54) From Global to Local: Multi-scale Out-of-distribution Detection (Ji Zhang et al., 2023)

{{<citation>}}

Ji Zhang, Lianli Gao, Bingguang Hao, Hao Huang, Jingkuan Song, Hengtao Shen. (2023)  
**From Global to Local: Multi-scale Out-of-distribution Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2308.10239v1)  

---


**ABSTRACT**  
Out-of-distribution (OOD) detection aims to detect "unknown" data whose labels have not been seen during the in-distribution (ID) training process. Recent progress in representation learning gives rise to distance-based OOD detection that recognizes inputs as ID/OOD according to their relative distances to the training data of ID classes. Previous approaches calculate pairwise distances relying only on global image representations, which can be sub-optimal as the inevitable background clutter and intra-class variation may drive image-level representations from the same ID class far apart in a given representation space. In this work, we overcome this challenge by proposing Multi-scale OOD DEtection (MODE), a first framework leveraging both global visual information and local region details of images to maximally benefit OOD detection. Specifically, we first find that existing models pretrained by off-the-shelf cross-entropy or contrastive losses are incompetent to capture valuable local representations for MODE, due to the scale-discrepancy between the ID training and OOD detection processes. To mitigate this issue and encourage locally discriminative representations in ID training, we propose Attention-based Local PropAgation (ALPA), a trainable objective that exploits a cross-attention mechanism to align and highlight the local regions of the target objects for pairwise examples. During test-time OOD detection, a Cross-Scale Decision (CSD) function is further devised on the most discriminative multi-scale representations to distinguish ID/OOD data more faithfully. We demonstrate the effectiveness and flexibility of MODE on several benchmarks -- on average, MODE outperforms the previous state-of-the-art by up to 19.24% in FPR, 2.77% in AUROC. Code is available at https://github.com/JimZAI/MODE-OOD.

{{</citation>}}


### (30/54) FedSIS: Federated Split Learning with Intermediate Representation Sampling for Privacy-preserving Generalized Face Presentation Attack Detection (Naif Alkhunaizi et al., 2023)

{{<citation>}}

Naif Alkhunaizi, Koushik Srivatsan, Faris Almalik, Ibrahim Almakky, Karthik Nandakumar. (2023)  
**FedSIS: Federated Split Learning with Intermediate Representation Sampling for Privacy-preserving Generalized Face Presentation Attack Detection**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10236v1)  

---


**ABSTRACT**  
Lack of generalization to unseen domains/attacks is the Achilles heel of most face presentation attack detection (FacePAD) algorithms. Existing attempts to enhance the generalizability of FacePAD solutions assume that data from multiple source domains are available with a single entity to enable centralized training. In practice, data from different source domains may be collected by diverse entities, who are often unable to share their data due to legal and privacy constraints. While collaborative learning paradigms such as federated learning (FL) can overcome this problem, standard FL methods are ill-suited for domain generalization because they struggle to surmount the twin challenges of handling non-iid client data distributions during training and generalizing to unseen domains during inference. In this work, a novel framework called Federated Split learning with Intermediate representation Sampling (FedSIS) is introduced for privacy-preserving domain generalization. In FedSIS, a hybrid Vision Transformer (ViT) architecture is learned using a combination of FL and split learning to achieve robustness against statistical heterogeneity in the client data distributions without any sharing of raw data (thereby preserving privacy). To further improve generalization to unseen domains, a novel feature augmentation strategy called intermediate representation sampling is employed, and discriminative information from intermediate blocks of a ViT is distilled using a shared adapter network. The FedSIS approach has been evaluated on two well-known benchmarks for cross-domain FacePAD to demonstrate that it is possible to achieve state-of-the-art generalization performance without data sharing. Code: https://github.com/Naiftt/FedSIS

{{</citation>}}


### (31/54) Blind Face Restoration for Under-Display Camera via Dictionary Guided Transformer (Jingfan Tan et al., 2023)

{{<citation>}}

Jingfan Tan, Xiaoxu Chen, Tao Wang, Kaihao Zhang, Wenhan Luo, Xiaocun Cao. (2023)  
**Blind Face Restoration for Under-Display Camera via Dictionary Guided Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10196v1)  

---


**ABSTRACT**  
By hiding the front-facing camera below the display panel, Under-Display Camera (UDC) provides users with a full-screen experience. However, due to the characteristics of the display, images taken by UDC suffer from significant quality degradation. Methods have been proposed to tackle UDC image restoration and advances have been achieved. There are still no specialized methods and datasets for restoring UDC face images, which may be the most common problem in the UDC scene. To this end, considering color filtering, brightness attenuation, and diffraction in the imaging process of UDC, we propose a two-stage network UDC Degradation Model Network named UDC-DMNet to synthesize UDC images by modeling the processes of UDC imaging. Then we use UDC-DMNet and high-quality face images from FFHQ and CelebA-Test to create UDC face training datasets FFHQ-P/T and testing datasets CelebA-Test-P/T for UDC face restoration. We propose a novel dictionary-guided transformer network named DGFormer. Introducing the facial component dictionary and the characteristics of the UDC image in the restoration makes DGFormer capable of addressing blind face restoration in UDC scenarios. Experiments show that our DGFormer and UDC-DMNet achieve state-of-the-art performance.

{{</citation>}}


### (32/54) VLN-PETL: Parameter-Efficient Transfer Learning for Vision-and-Language Navigation (Yanyuan Qiao et al., 2023)

{{<citation>}}

Yanyuan Qiao, Zheng Yu, Qi Wu. (2023)  
**VLN-PETL: Parameter-Efficient Transfer Learning for Vision-and-Language Navigation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.10172v1)  

---


**ABSTRACT**  
The performance of the Vision-and-Language Navigation~(VLN) tasks has witnessed rapid progress recently thanks to the use of large pre-trained vision-and-language models. However, full fine-tuning the pre-trained model for every downstream VLN task is becoming costly due to the considerable model size. Recent research hotspot of Parameter-Efficient Transfer Learning (PETL) shows great potential in efficiently tuning large pre-trained models for the common CV and NLP tasks, which exploits the most of the representation knowledge implied in the pre-trained model while only tunes a minimal set of parameters. However, simply utilizing existing PETL methods for the more challenging VLN tasks may bring non-trivial degeneration to the performance. Therefore, we present the first study to explore PETL methods for VLN tasks and propose a VLN-specific PETL method named VLN-PETL. Specifically, we design two PETL modules: Historical Interaction Booster (HIB) and Cross-modal Interaction Booster (CIB). Then we combine these two modules with several existing PETL methods as the integrated VLN-PETL. Extensive experimental results on four mainstream VLN tasks (R2R, REVERIE, NDH, RxR) demonstrate the effectiveness of our proposed VLN-PETL, where VLN-PETL achieves comparable or even better performance to full fine-tuning and outperforms other PETL methods with promising margins.

{{</citation>}}


### (33/54) ThermRad: A Multi-modal Dataset for Robust 3D Object Detection under Challenging Conditions (Qiao Yan et al., 2023)

{{<citation>}}

Qiao Yan, Yihan Wang. (2023)  
**ThermRad: A Multi-modal Dataset for Robust 3D Object Detection under Challenging Conditions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.10161v1)  

---


**ABSTRACT**  
Robust 3D object detection in extreme weather and illumination conditions is a challenging task. While radars and thermal cameras are known for their resilience to these conditions, few studies have been conducted on radar-thermal fusion due to the lack of corresponding datasets. To address this gap, we first present a new multi-modal dataset called ThermRad, which includes a 3D LiDAR, a 4D radar, an RGB camera and a thermal camera. This dataset is unique because it includes data from all four sensors in extreme weather conditions, providing a valuable resource for future research in this area. To validate the robustness of 4D radars and thermal cameras for 3D object detection in challenging weather conditions, we propose a new multi-modal fusion method called RTDF-RCNN, which leverages the complementary strengths of 4D radars and thermal cameras to boost object detection performance. To further prove the effectiveness of our proposed framework, we re-implement state-of-the-art (SOTA) 3D detectors on our dataset as benchmarks for evaluation. Our method achieves significant enhancements in detecting cars, pedestrians, and cyclists, with improvements of over 7.98%, 24.27%, and 27.15%, respectively, while achieving comparable results to LiDAR-based approaches. Our contributions in both the ThermRad dataset and the new multi-modal fusion method provide a new approach to robust 3D object detection in adverse weather and illumination conditions. The ThermRad dataset will be released.

{{</citation>}}


### (34/54) SSMG: Spatial-Semantic Map Guided Diffusion Model for Free-form Layout-to-Image Generation (Chengyou Jia et al., 2023)

{{<citation>}}

Chengyou Jia, Minnan Luo, Zhuohang Dang, Guang Dai, Xiaojun Chang, Mengmeng Wang, Jingdong Wang. (2023)  
**SSMG: Spatial-Semantic Map Guided Diffusion Model for Free-form Layout-to-Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.10156v1)  

---


**ABSTRACT**  
Despite significant progress in Text-to-Image (T2I) generative models, even lengthy and complex text descriptions still struggle to convey detailed controls. In contrast, Layout-to-Image (L2I) generation, aiming to generate realistic and complex scene images from user-specified layouts, has risen to prominence. However, existing methods transform layout information into tokens or RGB images for conditional control in the generative process, leading to insufficient spatial and semantic controllability of individual instances. To address these limitations, we propose a novel Spatial-Semantic Map Guided (SSMG) diffusion model that adopts the feature map, derived from the layout, as guidance. Owing to rich spatial and semantic information encapsulated in well-designed feature maps, SSMG achieves superior generation quality with sufficient spatial and semantic controllability compared to previous works. Additionally, we propose the Relation-Sensitive Attention (RSA) and Location-Sensitive Attention (LSA) mechanisms. The former aims to model the relationships among multiple objects within scenes while the latter is designed to heighten the model's sensitivity to the spatial information embedded in the guidance. Extensive experiments demonstrate that SSMG achieves highly promising results, setting a new state-of-the-art across a range of metrics encompassing fidelity, diversity, and controllability.

{{</citation>}}


### (35/54) Unilaterally Aggregated Contrastive Learning with Hierarchical Augmentation for Anomaly Detection (Guodong Wang et al., 2023)

{{<citation>}}

Guodong Wang, Yunhong Wang, Jie Qin, Dongming Zhang, Xiuguo Bao, Di Huang. (2023)  
**Unilaterally Aggregated Contrastive Learning with Hierarchical Augmentation for Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, Augmentation, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.10155v1)  

---


**ABSTRACT**  
Anomaly detection (AD), aiming to find samples that deviate from the training distribution, is essential in safety-critical applications. Though recent self-supervised learning based attempts achieve promising results by creating virtual outliers, their training objectives are less faithful to AD which requires a concentrated inlier distribution as well as a dispersive outlier distribution. In this paper, we propose Unilaterally Aggregated Contrastive Learning with Hierarchical Augmentation (UniCon-HA), taking into account both the requirements above. Specifically, we explicitly encourage the concentration of inliers and the dispersion of virtual outliers via supervised and unsupervised contrastive losses, respectively. Considering that standard contrastive data augmentation for generating positive views may induce outliers, we additionally introduce a soft mechanism to re-weight each augmented inlier according to its deviation from the inlier distribution, to ensure a purified concentration. Moreover, to prompt a higher concentration, inspired by curriculum learning, we adopt an easy-to-hard hierarchical augmentation strategy and perform contrastive aggregation at different depths of the network based on the strengths of data augmentation. Our method is evaluated under three AD settings including unlabeled one-class, unlabeled multi-class, and labeled multi-class, demonstrating its consistent superiority over other competitors.

{{</citation>}}


### (36/54) ESTextSpotter: Towards Better Scene Text Spotting with Explicit Synergy in Transformer (Mingxin Huang et al., 2023)

{{<citation>}}

Mingxin Huang, Jiaxin Zhang, Dezhi Peng, Hao Lu, Can Huang, Yuliang Liu, Xiang Bai, Lianwen Jin. (2023)  
**ESTextSpotter: Towards Better Scene Text Spotting with Explicit Synergy in Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10147v1)  

---


**ABSTRACT**  
In recent years, end-to-end scene text spotting approaches are evolving to the Transformer-based framework. While previous studies have shown the crucial importance of the intrinsic synergy between text detection and recognition, recent advances in Transformer-based methods usually adopt an implicit synergy strategy with shared query, which can not fully realize the potential of these two interactive tasks. In this paper, we argue that the explicit synergy considering distinct characteristics of text detection and recognition can significantly improve the performance text spotting. To this end, we introduce a new model named Explicit Synergy-based Text Spotting Transformer framework (ESTextSpotter), which achieves explicit synergy by modeling discriminative and interactive features for text detection and recognition within a single decoder. Specifically, we decompose the conventional shared query into task-aware queries for text polygon and content, respectively. Through the decoder with the proposed vision-language communication module, the queries interact with each other in an explicit manner while preserving discriminative patterns of text detection and recognition, thus improving performance significantly. Additionally, we propose a task-aware query initialization scheme to ensure stable training. Experimental results demonstrate that our model significantly outperforms previous state-of-the-art methods. Code is available at https://github.com/mxin262/ESTextSpotter.

{{</citation>}}


### (37/54) March in Chat: Interactive Prompting for Remote Embodied Referring Expression (Yanyuan Qiao et al., 2023)

{{<citation>}}

Yanyuan Qiao, Yuankai Qi, Zheng Yu, Jing Liu, Qi Wu. (2023)  
**March in Chat: Interactive Prompting for Remote Embodied Referring Expression**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.10141v1)  

---


**ABSTRACT**  
Many Vision-and-Language Navigation (VLN) tasks have been proposed in recent years, from room-based to object-based and indoor to outdoor. The REVERIE (Remote Embodied Referring Expression) is interesting since it only provides high-level instructions to the agent, which are closer to human commands in practice. Nevertheless, this poses more challenges than other VLN tasks since it requires agents to infer a navigation plan only based on a short instruction. Large Language Models (LLMs) show great potential in robot action planning by providing proper prompts. Still, this strategy has not been explored under the REVERIE settings. There are several new challenges. For example, the LLM should be environment-aware so that the navigation plan can be adjusted based on the current visual observation. Moreover, the LLM planned actions should be adaptable to the much larger and more complex REVERIE environment. This paper proposes a March-in-Chat (MiC) model that can talk to the LLM on the fly and plan dynamically based on a newly proposed Room-and-Object Aware Scene Perceiver (ROASP). Our MiC model outperforms the previous state-of-the-art by large margins by SPL and RGSPL metrics on the REVERIE benchmark.

{{</citation>}}


### (38/54) TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective (Jun Dan et al., 2023)

{{<citation>}}

Jun Dan, Yang Liu, Haoyu Xie, Jiankang Deng, Haoran Xie, Xuansong Xie, Baigui Sun. (2023)  
**TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10133v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have demonstrated powerful representation ability in various visual tasks thanks to their intrinsic data-hungry nature. However, we unexpectedly find that ViTs perform vulnerably when applied to face recognition (FR) scenarios with extremely large datasets. We investigate the reasons for this phenomenon and discover that the existing data augmentation approach and hard sample mining strategy are incompatible with ViTs-based FR backbone due to the lack of tailored consideration on preserving face structural information and leveraging each local token information. To remedy these problems, this paper proposes a superior FR model called TransFace, which employs a patch-level data augmentation strategy named DPAP and a hard sample mining strategy named EHSM. Specially, DPAP randomly perturbs the amplitude information of dominant patches to expand sample diversity, which effectively alleviates the overfitting problem in ViTs. EHSM utilizes the information entropy in the local tokens to dynamically adjust the importance weight of easy and hard samples during training, leading to a more stable prediction. Experiments on several benchmarks demonstrate the superiority of our TransFace. Code and models are available at https://github.com/DanJun6737/TransFace.

{{</citation>}}


## astro-ph.SR (1)



### (39/54) Homogenising SoHO/EIT and SDO/AIA 171Å$~$ Images: A Deep Learning Approach (Subhamoy Chatterjee et al., 2023)

{{<citation>}}

Subhamoy Chatterjee, Andrés Muñoz-Jaramillo, Maher Dayeh, Hazel M. Bain, Kimberly Moreland. (2023)  
**Homogenising SoHO/EIT and SDO/AIA 171Å$~$ Images: A Deep Learning Approach**  

---
Primary Category: astro-ph.SR  
Categories: astro-ph-SR, astro-ph.SR, cs-LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10322v1)  

---


**ABSTRACT**  
Extreme Ultraviolet images of the Sun are becoming an integral part of space weather prediction tasks. However, having different surveys requires the development of instrument-specific prediction algorithms. As an alternative, it is possible to combine multiple surveys to create a homogeneous dataset. In this study, we utilize the temporal overlap of SoHO/EIT and SDO/AIA 171~\AA ~surveys to train an ensemble of deep learning models for creating a single homogeneous survey of EUV images for 2 solar cycles. Prior applications of deep learning have focused on validating the homogeneity of the output while overlooking the systematic estimation of uncertainty. We use an approach called `Approximate Bayesian Ensembling' to generate an ensemble of models whose uncertainty mimics that of a fully Bayesian neural network at a fraction of the cost. We find that ensemble uncertainty goes down as the training set size increases. Additionally, we show that the model ensemble adds immense value to the prediction by showing higher uncertainty in test data that are not well represented in the training data.

{{</citation>}}


## q-bio.QM (1)



### (40/54) Preserving Specificity in Federated Graph Learning for fMRI-based Neurological Disorder Identification (Junhao Zhang et al., 2023)

{{<citation>}}

Junhao Zhang, Qianqian Wang, Xiaochuan Wang, Lishan Qiao, Mingxia Liu. (2023)  
**Preserving Specificity in Federated Graph Learning for fMRI-based Neurological Disorder Identification**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, eess-SP, q-bio-QM, q-bio.QM  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.10302v1)  

---


**ABSTRACT**  
Resting-state functional magnetic resonance imaging (rs-fMRI) offers a non-invasive approach to examining abnormal brain connectivity associated with brain disorders. Graph neural network (GNN) gains popularity in fMRI representation learning and brain disorder analysis with powerful graph representation capabilities. Training a general GNN often necessitates a large-scale dataset from multiple imaging centers/sites, but centralizing multi-site data generally faces inherent challenges related to data privacy, security, and storage burden. Federated Learning (FL) enables collaborative model training without centralized multi-site fMRI data. Unfortunately, previous FL approaches for fMRI analysis often ignore site-specificity, including demographic factors such as age, gender, and education level. To this end, we propose a specificity-aware federated graph learning (SFGL) framework for rs-fMRI analysis and automated brain disorder identification, with a server and multiple clients/sites for federated model aggregation and prediction. At each client, our model consists of a shared and a personalized branch, where parameters of the shared branch are sent to the server while those of the personalized branch remain local. This can facilitate knowledge sharing among sites and also helps preserve site specificity. In the shared branch, we employ a spatio-temporal attention graph isomorphism network to learn dynamic fMRI representations. In the personalized branch, we integrate vectorized demographic information (i.e., age, gender, and education years) and functional connectivity networks to preserve site-specific characteristics. Representations generated by the two branches are then fused for classification. Experimental results on two fMRI datasets with a total of 1,218 subjects suggest that SFGL outperforms several state-of-the-art approaches.

{{</citation>}}


## cs.CE (1)



### (41/54) A review of SolarWinds attack on Orion platform using persistent threat agents anf techniques for gaining unauthorized access (Antigoni Kruti et al., 2023)

{{<citation>}}

Antigoni Kruti, Usman Butt, Rejwan Sulaiman. (2023)  
**A review of SolarWinds attack on Orion platform using persistent threat agents anf techniques for gaining unauthorized access**  

---
Primary Category: cs.CE  
Categories: F-2-2; I-2-7, cs-CE, cs-CR, cs.CE  
Keywords: Microsoft, Security  
[Paper Link](http://arxiv.org/abs/2308.10294v1)  

---


**ABSTRACT**  
This paper of work examines the SolarWinds attack, designed on Orion Platform security incident. It analyses the persistent threats agents and potential technical attack techniques to gain unauthorized access. In 2020 SolarWinds attack indicates an initial breach disclosure on Orion Platform software by malware distribution on IT and government organizations such as Homeland Security, Microsoft and Intel associated with supply chains leaks consequences from small loopholes in security systems. Hackers increased the number of infected company and businesses networks during the supply-chain attack, hackers were capable to propagate the attack by using a VMware exploit. On the special way they started to target command injections, privilege escalations, and use after free platforms of VMware. In this way, they gained access to Virtual Machines and in the east way pivot other servers. This literature review aim to analyze the security gap regarding to SolarWinds incident on Orion Platform, the impact on industry and financial sectors involving the elements of incident response plan. Therefore, this research paper ensures specifications of proper solutions for possible defense security systems by analyzing a SolarWinds attack case study via system evaluation and monitoring. It concludes with necessary remediation actions on cyber hygiene countermeasures, common vulnerabilities and exposure analysis and solutions.

{{</citation>}}


## cs.LG (5)



### (42/54) Towards Few-shot Coordination: Revisiting Ad-hoc Teamplay Challenge In the Game of Hanabi (Hadi Nekoei et al., 2023)

{{<citation>}}

Hadi Nekoei, Xutong Zhao, Janarthanan Rajendran, Miao Liu, Sarath Chandar. (2023)  
**Towards Few-shot Coordination: Revisiting Ad-hoc Teamplay Challenge In the Game of Hanabi**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.10284v1)  

---


**ABSTRACT**  
Cooperative Multi-agent Reinforcement Learning (MARL) algorithms with Zero-Shot Coordination (ZSC) have gained significant attention in recent years. ZSC refers to the ability of agents to coordinate zero-shot (without additional interaction experience) with independently trained agents. While ZSC is crucial for cooperative MARL agents, it might not be possible for complex tasks and changing environments. Agents also need to adapt and improve their performance with minimal interaction with other agents. In this work, we show empirically that state-of-the-art ZSC algorithms have poor performance when paired with agents trained with different learning methods, and they require millions of interaction samples to adapt to these new partners. To investigate this issue, we formally defined a framework based on a popular cooperative multi-agent game called Hanabi to evaluate the adaptability of MARL methods. In particular, we created a diverse set of pre-trained agents and defined a new metric called adaptation regret that measures the agent's ability to efficiently adapt and improve its coordination performance when paired with some held-out pool of partners on top of its ZSC performance. After evaluating several SOTA algorithms using our framework, our experiments reveal that naive Independent Q-Learning (IQL) agents in most cases adapt as quickly as the SOTA ZSC algorithm Off-Belief Learning (OBL). This finding raises an interesting research question: How to design MARL algorithms with high ZSC performance and capability of fast adaptation to unseen partners. As a first step, we studied the role of different hyper-parameters and design choices on the adaptability of current MARL algorithms. Our experiments show that two categories of hyper-parameters controlling the training data diversity and optimization process have a significant impact on the adaptability of Hanabi agents.

{{</citation>}}


### (43/54) Minimalist Traffic Prediction: Linear Layer Is All You Need (Wenying Duan et al., 2023)

{{<citation>}}

Wenying Duan, Hong Rao, Wei Huang, Xiaoxi He. (2023)  
**Minimalist Traffic Prediction: Linear Layer Is All You Need**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.10276v1)  

---


**ABSTRACT**  
Traffic prediction is essential for the progression of Intelligent Transportation Systems (ITS) and the vision of smart cities. While Spatial-Temporal Graph Neural Networks (STGNNs) have shown promise in this domain by leveraging Graph Neural Networks (GNNs) integrated with either RNNs or Transformers, they present challenges such as computational complexity, gradient issues, and resource-intensiveness. This paper addresses these challenges, advocating for three main solutions: a node-embedding approach, time series decomposition, and periodicity learning. We introduce STLinear, a minimalist model architecture designed for optimized efficiency and performance. Unlike traditional STGNNs, STlinear operates fully locally, avoiding inter-node data exchanges, and relies exclusively on linear layers, drastically cutting computational demands. Our empirical studies on real-world datasets confirm STLinear's prowess, matching or exceeding the accuracy of leading STGNNs, but with significantly reduced complexity and computation overhead (more than 95% reduction in MACs per epoch compared to state-of-the-art STGNN baseline published in 2023). In summary, STLinear emerges as a potent, efficient alternative to conventional STGNNs, with profound implications for the future of ITS and smart city initiatives.

{{</citation>}}


### (44/54) Hiding Backdoors within Event Sequence Data via Poisoning Attacks (Elizaveta Kovtun et al., 2023)

{{<citation>}}

Elizaveta Kovtun, Alina Ermilova, Dmitry Berestnev, Alexey Zaytsev. (2023)  
**Hiding Backdoors within Event Sequence Data via Poisoning Attacks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2308.10201v1)  

---


**ABSTRACT**  
The financial industry relies on deep learning models for making important decisions. This adoption brings new danger, as deep black-box models are known to be vulnerable to adversarial attacks. In computer vision, one can shape the output during inference by performing an adversarial attack called poisoning via introducing a backdoor into the model during training. For sequences of financial transactions of a customer, insertion of a backdoor is harder to perform, as models operate over a more complex discrete space of sequences, and systematic checks for insecurities occur. We provide a method to introduce concealed backdoors, creating vulnerabilities without altering their functionality for uncontaminated data. To achieve this, we replace a clean model with a poisoned one that is aware of the availability of a backdoor and utilize this knowledge. Our most difficult for uncovering attacks include either additional supervised detection step of poisoned data activated during the test or well-hidden model weight modifications. The experimental study provides insights into how these effects vary across different datasets, architectures, and model components. Alternative methods and baselines, such as distillation-type regularization, are also explored but found to be less efficient. Conducted on three open transaction datasets and architectures, including LSTM, CNN, and Transformer, our findings not only illuminate the vulnerabilities in contemporary models but also can drive the construction of more robust systems.

{{</citation>}}


### (45/54) Deep Reinforcement Learning for Artificial Upwelling Energy Management (Yiyuan Zhang et al., 2023)

{{<citation>}}

Yiyuan Zhang, Wei Fan. (2023)  
**Deep Reinforcement Learning for Artificial Upwelling Energy Management**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.10199v1)  

---


**ABSTRACT**  
The potential of artificial upwelling (AU) as a means of lifting nutrient-rich bottom water to the surface, stimulating seaweed growth, and consequently enhancing ocean carbon sequestration, has been gaining increasing attention in recent years. This has led to the development of the first solar-powered and air-lifted AU system (AUS) in China. However, efficient scheduling of air injection systems remains a crucial challenge in operating AUS, as it holds the potential to significantly improve system efficiency. Conventional approaches based on rules or models are often impractical due to the complex and heterogeneous nature of the marine environment and its associated disturbances. To address this challenge, we propose a novel energy management approach that utilizes deep reinforcement learning (DRL) algorithm to develop efficient strategies for operating AUS. Through extensive simulations, we evaluate the performance of our algorithm and demonstrate its superior effectiveness over traditional rule-based approaches and other DRL algorithms in reducing energy wastage while ensuring the stable and efficient operation of AUS. Our findings suggest that a DRL-based approach offers a promising way for improving the efficiency of AUS and enhancing the sustainability of seaweed cultivation and carbon sequestration in the ocean.

{{</citation>}}


### (46/54) ExpeL: LLM Agents Are Experiential Learners (Andrew Zhao et al., 2023)

{{<citation>}}

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, Gao Huang. (2023)  
**ExpeL: LLM Agents Are Experiential Learners**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.10144v1)  

---


**ABSTRACT**  
The recent surge in research interest in applying large language models (LLMs) to decision-making tasks has flourished by leveraging the extensive world knowledge embedded in LLMs. While there is a growing demand to tailor LLMs for custom decision-making tasks, finetuning them for specific tasks is resource-intensive and may diminish the model's generalization capabilities. Moreover, state-of-the-art language models like GPT-4 and Claude are primarily accessible through API calls, with their parametric weights remaining proprietary and unavailable to the public. This scenario emphasizes the growing need for new methodologies that allow learning from agent experiences without requiring parametric updates. To address these problems, we introduce the Experiential Learning (ExpeL) agent. Our agent autonomously gathers experiences and extracts knowledge using natural language from a collection of training tasks. At inference, the agent recalls its extracted insights and past experiences to make informed decisions. Our empirical results highlight the robust learning efficacy of the ExpeL agent, indicating a consistent enhancement in its performance as it accumulates experiences. We further explore the emerging capabilities and transfer learning potential of the ExpeL agent through qualitative observations and additional experiments.

{{</citation>}}


## cs.NI (1)



### (47/54) Towards Synthesizing Datasets for IEEE 802.1 Time-sensitive Networking (Doğanalp Ergenç et al., 2023)

{{<citation>}}

Doğanalp Ergenç, Nurefşan Sertbaş Bülbül, Lisa Maile, Anna Arestova, Mathias Fischer. (2023)  
**Towards Synthesizing Datasets for IEEE 802.1 Time-sensitive Networking**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.10255v1)  

---


**ABSTRACT**  
IEEE 802.1 Time-sensitive Networking (TSN) protocols have recently been proposed to replace legacy networking technologies across different mission-critical systems (MCSs). Design, configuration, and maintenance of TSN within MCSs require advanced methods to tackle the highly complex and interconnected nature of those systems. Accordingly, artificial intelligence (AI) and machine learning (ML) models are the most prominent enablers to develop such methods. However, they usually require a significant amount of data for model training, which is not easily accessible. This short paper aims to recapitulate the need for TSN datasets to flourish research on AI/ML-based techniques for TSN systems. Moreover, it analyzes the main requirements and alternative designs to build a TSN platform to synthesize realistic datasets.

{{</citation>}}


## cs.CR (2)



### (48/54) Towards a Formally Verified Security Monitor for VM-based Confidential Computing (Wojciech Ozga et al., 2023)

{{<citation>}}

Wojciech Ozga, Guerney D. H. Hunt, Michael V. Le, Elaine R. Palmer, Avraham Shinnar. (2023)  
**Towards a Formally Verified Security Monitor for VM-based Confidential Computing**  

---
Primary Category: cs.CR  
Categories: cs-AR, cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.10249v1)  

---


**ABSTRACT**  
Confidential computing is becoming a key technology for isolating high-assurance applications from large amounts of untrusted code typical in modern systems. However, existing confidential computing technologies cannot secure the most critical applications, like systems controlling critical infrastructure, hardware security modules, or aircraft, because they lack formal verification needed for their certification.   This paper tackles the above problem by introducing a canonical architecture for virtual machine (VM)-based confidential computing systems. It abstracts processor-specific components and identifies a minimal set of hardware primitives required by a trusted security monitor to enforce security guarantees. This paper makes a step towards formally modeling and proving the correctness of the security monitor. We present our methodology and demonstrate the proposed approach based on an example from our model and Rust implementation of the security monitor for RISC-V.

{{</citation>}}


### (49/54) AutoReP: Automatic ReLU Replacement for Fast Private Network Inference (Hongwu Peng et al., 2023)

{{<citation>}}

Hongwu Peng, Shaoyi Huang, Tong Zhou, Yukui Luo, Chenghong Wang, Zigeng Wang, Jiahui Zhao, Xi Xie, Ang Li, Tony Geng, Kaleel Mahmood, Wujie Wen, Xiaolin Xu, Caiwen Ding. (2023)  
**AutoReP: Automatic ReLU Replacement for Fast Private Network Inference**  

---
Primary Category: cs.CR  
Categories: E-3; I-2; B-0, cs-CR, cs-LG, cs.CR  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.10134v1)  

---


**ABSTRACT**  
The growth of the Machine-Learning-As-A-Service (MLaaS) market has highlighted clients' data privacy and security issues. Private inference (PI) techniques using cryptographic primitives offer a solution but often have high computation and communication costs, particularly with non-linear operators like ReLU. Many attempts to reduce ReLU operations exist, but they may need heuristic threshold selection or cause substantial accuracy loss. This work introduces AutoReP, a gradient-based approach to lessen non-linear operators and alleviate these issues. It automates the selection of ReLU and polynomial functions to speed up PI applications and introduces distribution-aware polynomial approximation (DaPa) to maintain model expressivity while accurately approximating ReLUs. Our experimental results demonstrate significant accuracy improvements of 6.12% (94.31%, 12.9K ReLU budget, CIFAR-10), 8.39% (74.92%, 12.9K ReLU budget, CIFAR-100), and 9.45% (63.69%, 55K ReLU budget, Tiny-ImageNet) over current state-of-the-art methods, e.g., SNL. Morever, AutoReP is applied to EfficientNet-B2 on ImageNet dataset, and achieved 75.55% accuracy with 176.1 times ReLU budget reduction.

{{</citation>}}


## cs.SI (1)



### (50/54) Fairness-aware Competitive Bidding Influence Maximization in Social Networks (Congcong Zhang et al., 2023)

{{<citation>}}

Congcong Zhang, Jingya Zhou, Jin Wang, Jianxi Fan, Yingdan Shi. (2023)  
**Fairness-aware Competitive Bidding Influence Maximization in Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2308.10209v1)  

---


**ABSTRACT**  
Competitive Influence Maximization (CIM) has been studied for years due to its wide application in many domains. Most current studies primarily focus on the micro-level optimization by designing policies for one competitor to defeat its opponents. Furthermore, current studies ignore the fact that many influential nodes have their own starting prices, which may lead to inefficient budget allocation. In this paper, we propose a novel Competitive Bidding Influence Maximization (CBIM) problem, where the competitors allocate budgets to bid for the seeds attributed to the platform during multiple bidding rounds. To solve the CBIM problem, we propose a Fairness-aware Multi-agent Competitive Bidding Influence Maximization (FMCBIM) framework. In this framework, we present a Multi-agent Bidding Particle Environment (MBE) to model the competitors' interactions, and design a starting price adjustment mechanism to model the dynamic bidding environment. Moreover, we put forward a novel Multi-agent Competitive Bidding Influence Maximization (MCBIM) algorithm to optimize competitors' bidding policies. Extensive experiments on five datasets show that our work has good efficiency and effectiveness.

{{</citation>}}


## cs.FL (1)



### (51/54) Real-time Regular Expression Matching (Alexandra Bernadotte, 2023)

{{<citation>}}

Alexandra Bernadotte. (2023)  
**Real-time Regular Expression Matching**  

---
Primary Category: cs.FL  
Categories: cs-CC, cs-CV, cs-DM, cs-FL, cs.FL  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2308.10208v1)  

---


**ABSTRACT**  
This paper is devoted to finite state automata, regular expression matching, pattern recognition, and the exponential blow-up problem, which is the growing complexity of automata exponentially depending on regular expression length. This paper presents a theoretical and hardware solution to the exponential blow-up problem for some complicated classes of regular languages, which caused severe limitations in Network Intrusion Detection Systems work. The article supports the solution with theorems on correctness and complexity.

{{</citation>}}


## cs.AR (1)



### (52/54) ChatEDA: A Large Language Model Powered Autonomous Agent for EDA (Zhuolun He et al., 2023)

{{<citation>}}

Zhuolun He, Haoyuan Wu, Xinyun Zhang, Xufeng Yao, Su Zheng, Haisheng Zheng, Bei Yu. (2023)  
**ChatEDA: A Large Language Model Powered Autonomous Agent for EDA**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs.AR  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.10204v1)  

---


**ABSTRACT**  
The integration of a complex set of Electronic Design Automation (EDA) tools to enhance interoperability is a critical concern for circuit designers. Recent advancements in large language models (LLMs) have showcased their exceptional capabilities in natural language processing and comprehension, offering a novel approach to interfacing with EDA tools. This research paper introduces ChatEDA, an autonomous agent for EDA empowered by a large language model, AutoMage, complemented by EDA tools serving as executors. ChatEDA streamlines the design flow from the Register-Transfer Level (RTL) to the Graphic Data System Version II (GDSII) by effectively managing task planning, script generation, and task execution. Through comprehensive experimental evaluations, ChatEDA has demonstrated its proficiency in handling diverse requirements, and our fine-tuned AutoMage model has exhibited superior performance compared to GPT-4 and other similar LLMs.

{{</citation>}}


## cs.MM (1)



### (53/54) WMFormer++: Nested Transformer for Visible Watermark Removal via Implict Joint Learning (Dongjian Huo et al., 2023)

{{<citation>}}

Dongjian Huo, Zehong Zhang, Hanjing Su, Guanbin Li, Chaowei Fang, Qingyao Wu. (2023)  
**WMFormer++: Nested Transformer for Visible Watermark Removal via Implict Joint Learning**  

---
Primary Category: cs.MM  
Categories: cs-CL, cs-CV, cs-MM, cs.MM, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.10195v1)  

---


**ABSTRACT**  
Watermarking serves as a widely adopted approach to safeguard media copyright. In parallel, the research focus has extended to watermark removal techniques, offering an adversarial means to enhance watermark robustness and foster advancements in the watermarking field. Existing watermark removal methods often rely on UNet architectures with multiple decoder branches -- one for watermark localization and the other for background image restoration. These methods involve complex module designs to guide information flow for respective tasks, which can lead to suboptimal performance and an overly cumbersome model. To simplify the existing framework, we propose a novel Transformer-based approach with a unified decoder branch, treating watermark extraction and background restoration as a single task and allowing thenetwork to learn information flow between them without artificial design patterns. Additionally, we utilize nested structures to facilitate multi-scale feature fusion, forming a parallel ensemble of nested structures that constitute the UNet. Supervision is applied to UNets with varying depths to facilitate knowledge learning across all levels. Extensive experiments are conducted on various challenging benchmarks to validate the effectiveness of our proposed method. The results demonstrate that our approach achieves state-of-the-art performance and produces high-quality images.

{{</citation>}}


## cs.CY (1)



### (54/54) Privacy Perceptions and Behaviors of Google Personal Account Holders in Saudi Arabia (Eman Alashwali et al., 2023)

{{<citation>}}

Eman Alashwali, Lorrie Faith Cranor. (2023)  
**Privacy Perceptions and Behaviors of Google Personal Account Holders in Saudi Arabia**  

---
Primary Category: cs.CY  
Categories: cs-CR, cs-CY, cs-HC, cs.CY  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.10148v1)  

---


**ABSTRACT**  
While privacy perceptions and behaviors have been investigated in Western societies, little is known about these issues in non-Western societies. To bridge this gap, we interviewed 30 Google personal account holders in Saudi Arabia about their privacy perceptions (awareness, attitudes, preferences, and concerns) regarding the activity data that Google saves about them, as well as any steps they take to control Google's collection or use of this data. Our study focuses on Google's Activity Controls, which enable users to control whether, and how, Google saves their Web & App Activity, Location History, and YouTube History. Our results show that although most participants have some level of awareness about Google's data practices and the Activity Controls, many have only vague awareness, and the majority have not used the available controls. When participants viewed their saved activity data, many were surprised by what had been saved. While many participants find Google's use of their data to improve the services provided to them acceptable, the majority find the use of their data for ad purposes unacceptable. We observe that our Saudi participants exhibit similar trends and patterns in privacy awareness, attitudes, preferences, concerns, and behaviors to what has been found in studies in the US. However, our study is not a replication of any of the US studies, and further research is needed to directly compare US and Saudi participants. Our results emphasize the need for: (1) improved techniques to inform users about privacy settings during account sign-up, to remind users about their settings, and to raise awareness about privacy settings; (2) improved privacy setting interfaces to reduce the costs that deter many users from changing the settings; and (3) further research to explore privacy concerns in non-Western cultures.

{{</citation>}}
