---
draft: false
title: "arXiv @ 2023.12.26"
date: 2023-12-26
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.26"
    identifier: arxiv_20231226
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CL (7)](#cscl-7)
- [cs.MA (1)](#csma-1)
- [cs.SE (3)](#csse-3)
- [cs.CV (7)](#cscv-7)
- [cs.IR (3)](#csir-3)
- [cs.AI (1)](#csai-1)
- [cs.CY (1)](#cscy-1)
- [cs.LG (3)](#cslg-3)
- [eess.SY (1)](#eesssy-1)
- [cs.DB (1)](#csdb-1)
- [eess.AS (1)](#eessas-1)
- [stat.ML (1)](#statml-1)
- [cs.ET (1)](#cset-1)
- [cs.SD (1)](#cssd-1)
- [cs.DC (1)](#csdc-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [q-fin.MF (1)](#q-finmf-1)

## cs.CL (7)



### (1/35) README: Bridging Medical Jargon and Lay Understanding for Patient Education through Data-Centric NLP (Zonghai Yao et al., 2023)

{{<citation>}}

Zonghai Yao, Nandyala Siddharth Kantu, Guanghao Wei, Hieu Tran, Zhangqi Duan, Sunjae Kwon, Zhichao Yang, README annotation team, Hong Yu. (2023)  
**README: Bridging Medical Jargon and Lay Understanding for Patient Education through Data-Centric NLP**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2312.15561v1)  

---


**ABSTRACT**  
The advancement in healthcare has shifted focus toward patient-centric approaches, particularly in self-care and patient education, facilitated by access to Electronic Health Records (EHR). However, medical jargon in EHRs poses significant challenges in patient comprehension. To address this, we introduce a new task of automatically generating lay definitions, aiming to simplify complex medical terms into patient-friendly lay language. We first created the README dataset, an extensive collection of over 20,000 unique medical terms and 300,000 mentions, each offering context-aware lay definitions manually annotated by domain experts. We have also engineered a data-centric Human-AI pipeline that synergizes data filtering, augmentation, and selection to improve data quality. We then used README as the training data for models and leveraged a Retrieval-Augmented Generation (RAG) method to reduce hallucinations and improve the quality of model outputs. Our extensive automatic and human evaluations demonstrate that open-source mobile-friendly models, when fine-tuned with high-quality data, are capable of matching or even surpassing the performance of state-of-the-art closed-source large language models like ChatGPT. This research represents a significant stride in closing the knowledge gap in patient education and advancing patient-centric healthcare solutions

{{</citation>}}


### (2/35) Multi-level biomedical NER through multi-granularity embeddings and enhanced labeling (Fahime Shahrokh et al., 2023)

{{<citation>}}

Fahime Shahrokh, Nasser Ghadiri, Rasoul Samani, Milad Moradi. (2023)  
**Multi-level biomedical NER through multi-granularity embeddings and enhanced labeling**  

---
Primary Category: cs.CL  
Categories: 68T50, 68T07, J-3; I-2-7; I-2-1, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, LSTM, NER, Named Entity Recognition, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2312.15550v1)  

---


**ABSTRACT**  
Biomedical Named Entity Recognition (NER) is a fundamental task of Biomedical Natural Language Processing for extracting relevant information from biomedical texts, such as clinical records, scientific publications, and electronic health records. The conventional approaches for biomedical NER mainly use traditional machine learning techniques, such as Conditional Random Fields and Support Vector Machines or deep learning-based models like Recurrent Neural Networks and Convolutional Neural Networks. Recently, Transformer-based models, including BERT, have been used in the domain of biomedical NER and have demonstrated remarkable results. However, these models are often based on word-level embeddings, limiting their ability to capture character-level information, which is effective in biomedical NER due to the high variability and complexity of biomedical texts. To address these limitations, this paper proposes a hybrid approach that integrates the strengths of multiple models. In this paper, we proposed an approach that leverages fine-tuned BERT to provide contextualized word embeddings, a pre-trained multi-channel CNN for character-level information capture, and following by a BiLSTM + CRF for sequence labelling and modelling dependencies between the words in the text. In addition, also we propose an enhanced labelling method as part of pre-processing to enhance the identification of the entity's beginning word and thus improve the identification of multi-word entities, a common challenge in biomedical NER. By integrating these models and the pre-processing method, our proposed model effectively captures both contextual information and detailed character-level information. We evaluated our model on the benchmark i2b2/2010 dataset, achieving an F1-score of 90.11. These results illustrate the proficiency of our proposed model in performing biomedical Named Entity Recognition.

{{</citation>}}


### (3/35) YAYI-UIE: A Chat-Enhanced Instruction Tuning Framework for Universal Information Extraction (Xinglin Xiao et al., 2023)

{{<citation>}}

Xinglin Xiao, Yijie Wang, Nan Xu, Yuqi Wang, Hanxuan Yang, Minzheng Wang, Yin Luo, Lei Wang, Wenji Mao, Daniel Zeng. (2023)  
**YAYI-UIE: A Chat-Enhanced Instruction Tuning Framework for Universal Information Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Information Extraction  
[Paper Link](http://arxiv.org/abs/2312.15548v1)  

---


**ABSTRACT**  
The difficulty of the information extraction task lies in dealing with the task-specific label schemas and heterogeneous data structures. Recent work has proposed methods based on large language models to uniformly model different information extraction tasks. However, these existing methods are deficient in their information extraction capabilities for Chinese languages other than English. In this paper, we propose an end-to-end chat-enhanced instruction tuning framework for universal information extraction (YAYI-UIE), which supports both Chinese and English. Specifically, we utilize dialogue data and information extraction data to enhance the information extraction performance jointly. Experimental results show that our proposed framework achieves state-of-the-art performance on Chinese datasets while also achieving comparable performance on English datasets under both supervised settings and zero-shot settings.

{{</citation>}}


### (4/35) Making Large Language Models A Better Foundation For Dense Retrieval (Chaofan Li et al., 2023)

{{<citation>}}

Chaofan Li, Zheng Liu, Shitao Xiao, Yingxia Shao. (2023)  
**Making Large Language Models A Better Foundation For Dense Retrieval**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2312.15503v1)  

---


**ABSTRACT**  
Dense retrieval needs to learn discriminative text embeddings to represent the semantic relationship between query and document. It may benefit from the using of large language models (LLMs), given LLMs' strong capability on semantic understanding. However, the LLMs are pre-trained by text generation tasks, whose working pattern is completely different from representing texts as embeddings. As a result, it is imperative to study how to adapt LLMs properly so that they can be effectively initialized as the backbone encoder for dense retrieval.   In this paper, we propose a novel approach, called LLaRA (LLM adapted for dense RetrievAl), which works as a post-hoc adaptation of LLM for the dense retrieval application. LLaRA consists of two pretext tasks: EBAE (Embedding-Based Auto-Encoding) and EBAR (Embedding-Based Auto-Regression), where the text embeddings from LLM are used to reconstruct the tokens for the input sentence and predict the tokens for the next sentence, respectively. LLaRA turns out to be simple, lightweight, and highly effective. It is applied to adapt LLaMA-2-7B (base) on the Wikipedia corpus, where it substantially improves the model's fine-tuned performances on a variety of dense retrieval benchmarks, like MSMARCO and BEIR. Our model and code will be made publicly available at BGE repository.

{{</citation>}}


### (5/35) A Group Fairness Lens for Large Language Models (Guanqun Bi et al., 2023)

{{<citation>}}

Guanqun Bi, Lei Shen, Yuqiang Xie, Yanan Cao, Tiangang Zhu, Xiaodong He. (2023)  
**A Group Fairness Lens for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15478v1)  

---


**ABSTRACT**  
The rapid advancement of large language models has revolutionized various applications but also raised crucial concerns about their potential to perpetuate biases and unfairness when deployed in social media contexts. Evaluating LLMs' potential biases and fairness has become crucial, as existing methods rely on limited prompts focusing on just a few groups, lacking a comprehensive categorical perspective. In this paper, we propose evaluating LLM biases from a group fairness lens using a novel hierarchical schema characterizing diverse social groups. Specifically, we construct a dataset, GFair, encapsulating target-attribute combinations across multiple dimensions. In addition, we introduce statement organization, a new open-ended text generation task, to uncover complex biases in LLMs. Extensive evaluations of popular LLMs reveal inherent safety concerns. To mitigate the biases of LLM from a group fairness perspective, we pioneer a novel chain-of-thought method GF-Think to mitigate biases of LLMs from a group fairness perspective. Experimental results demonstrate its efficacy in mitigating bias in LLMs to achieve fairness.

{{</citation>}}


### (6/35) A Comprehensive Analysis of the Effectiveness of Large Language Models as Automatic Dialogue Evaluators (Chen Zhang et al., 2023)

{{<citation>}}

Chen Zhang, Luis Fernando D'Haro, Yiming Chen, Malu Zhang, Haizhou Li. (2023)  
**A Comprehensive Analysis of the Effectiveness of Large Language Models as Automatic Dialogue Evaluators**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.15407v1)  

---


**ABSTRACT**  
Automatic evaluation is an integral aspect of dialogue system research. The traditional reference-based NLG metrics are generally found to be unsuitable for dialogue assessment. Consequently, recent studies have suggested various unique, reference-free neural metrics that better align with human evaluations. Notably among them, large language models (LLMs), particularly the instruction-tuned variants like ChatGPT, are shown to be promising substitutes for human judges. Yet, existing works on utilizing LLMs for automatic dialogue evaluation are limited in their scope in terms of the number of meta-evaluation datasets, mode of evaluation, coverage of LLMs, etc. Hence, it remains inconclusive how effective these LLMs are. To this end, we conduct a comprehensive study on the application of LLMs for automatic dialogue evaluation. Specifically, we analyze the multi-dimensional evaluation capability of 30 recently emerged LLMs at both turn and dialogue levels, using a comprehensive set of 12 meta-evaluation datasets. Additionally, we probe the robustness of the LLMs in handling various adversarial perturbations at both turn and dialogue levels. Finally, we explore how model-level and dimension-level ensembles impact the evaluation performance. All resources are available at https://github.com/e0397123/comp-analysis.

{{</citation>}}


### (7/35) Fairness-Aware Structured Pruning in Transformers (Abdelrahman Zayed et al., 2023)

{{<citation>}}

Abdelrahman Zayed, Goncalo Mordido, Samira Shabanian, Ioana Baldini, Sarath Chandar. (2023)  
**Fairness-Aware Structured Pruning in Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: GPT, Pruning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.15398v1)  

---


**ABSTRACT**  
The increasing size of large language models (LLMs) has introduced challenges in their training and inference. Removing model components is perceived as a solution to tackle the large model sizes, however, existing pruning methods solely focus on performance, without considering an essential aspect for the responsible use of LLMs: model fairness. It is crucial to address the fairness of LLMs towards diverse groups, such as women, Black people, LGBTQ+, Jewish communities, among others, as they are being deployed and available to a wide audience. In this work, first, we investigate how attention heads impact fairness and performance in pre-trained transformer-based language models. We then propose a novel method to prune the attention heads that negatively impact fairness while retaining the heads critical for performance, i.e. language modeling capabilities. Our approach is practical in terms of time and resources, as it does not require fine-tuning the final pruned, and fairer, model. Our findings demonstrate a reduction in gender bias by 19%, 19.5%, 39.5%, 34.7%, 23%, and 8% for DistilGPT-2, GPT-2, GPT-Neo of two different sizes, GPT-J, and Llama 2 models, respectively, in comparison to the biased model, with only a slight decrease in performance.

{{</citation>}}


## cs.MA (1)



### (8/35) ConcaveQ: Non-Monotonic Value Function Factorization via Concave Representations in Deep Multi-Agent Reinforcement Learning (Huiqun Li et al., 2023)

{{<citation>}}

Huiqun Li, Hanhan Zhou, Yifei Zou, Dongxiao Yu, Tian Lan. (2023)  
**ConcaveQ: Non-Monotonic Value Function Factorization via Concave Representations in Deep Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15555v1)  

---


**ABSTRACT**  
Value function factorization has achieved great success in multi-agent reinforcement learning by optimizing joint action-value functions through the maximization of factorized per-agent utilities. To ensure Individual-Global-Maximum property, existing works often focus on value factorization using monotonic functions, which are known to result in restricted representation expressiveness. In this paper, we analyze the limitations of monotonic factorization and present ConcaveQ, a novel non-monotonic value function factorization approach that goes beyond monotonic mixing functions and employs neural network representations of concave mixing functions. Leveraging the concave property in factorization, an iterative action selection scheme is developed to obtain optimal joint actions during training. It is used to update agents' local policy networks, enabling fully decentralized execution. The effectiveness of the proposed ConcaveQ is validated across scenarios involving multi-agent predator-prey environment and StarCraft II micromanagement tasks. Empirical results exhibit significant improvement of ConcaveQ over state-of-the-art multi-agent reinforcement learning approaches.

{{</citation>}}


## cs.SE (3)



### (9/35) Guess What Quantum Computing Can Do for Test Case Optimization (Xinyi Wang et al., 2023)

{{<citation>}}

Xinyi Wang, Shaukat Ali, Tao Yue, Paolo Arcaini. (2023)  
**Guess What Quantum Computing Can Do for Test Case Optimization**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Google, QA  
[Paper Link](http://arxiv.org/abs/2312.15547v1)  

---


**ABSTRACT**  
In the near term, quantum approximate optimization algorithms (QAOAs) hold great potential to solve combinatorial optimization problems. These are hybrid algorithms, i.e., a combination of quantum and classical algorithms. Several proof-of-concept applications of QAOAs for solving combinatorial problems, such as portfolio optimization, energy optimization in power systems, and job scheduling, have been demonstrated. However, whether QAOAs can efficiently solve optimization problems from classical software engineering, such as test optimization, remains unstudied. To this end, we present the first effort to formulate a software test case optimization problem as a QAOA problem and solve it on quantum computer simulators. To solve bigger test optimization problems that require many qubits, which are unavailable these days, we integrate a problem decomposition strategy with the QAOA. We performed an empirical evaluation with five test case optimization problems and four industrial datasets from ABB, Google, and Orona to compare various configurations of our approach, assess its decomposition strategy of handling large datasets, and compare its performance with classical algorithms (i.e., Genetic Algorithm (GA) and Random Search). Based on the evaluation results, we recommend the best configuration of our approach for test case optimization problems. Also, we demonstrate that our strategy can reach the same effectiveness as GA and outperform GA in two out of five test case optimization problems we conducted.

{{</citation>}}


### (10/35) Harnessing Pre-trained Generalist Agents for Software Engineering Tasks (Paulina Stevia Nouwou Mindom et al., 2023)

{{<citation>}}

Paulina Stevia Nouwou Mindom, Amin Nikanjam, Foutse Khomh. (2023)  
**Harnessing Pre-trained Generalist Agents for Software Engineering Tasks**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Computer Vision, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.15536v1)  

---


**ABSTRACT**  
Nowadays, we are witnessing an increasing adoption of Artificial Intelligence (AI) to develop techniques aimed at improving the reliability, effectiveness, and overall quality of software systems. Deep reinforcement learning (DRL) has recently been successfully used for automation in complex tasks such as game testing and solving the job-shop scheduling problem. However, these specialized DRL agents, trained from scratch on specific tasks, suffer from a lack of generalizability to other tasks and they need substantial time to be developed and re-trained effectively. Recently, DRL researchers have begun to develop generalist agents, able to learn a policy from various environments and capable of achieving performances similar to or better than specialist agents in new tasks. In the Natural Language Processing or Computer Vision domain, these generalist agents are showing promising adaptation capabilities to never-before-seen tasks after a light fine-tuning phase and achieving high performance. This paper investigates the potential of generalist agents for solving SE tasks. Specifically, we conduct an empirical study aimed at assessing the performance of two generalist agents on two important SE tasks: the detection of bugs in games (for two games) and the minimization of makespan in a scheduling task, to solve the job-shop scheduling problem (for two instances). Our results show that the generalist agents outperform the specialist agents with very little effort for fine-tuning, achieving a 20% reduction of the makespan over specialized agent performance on task-based scheduling. In the context of game testing, some generalist agent configurations detect 85% more bugs than the specialist agents. Building on our analysis, we provide recommendations for researchers and practitioners looking to select generalist agents for SE tasks, to ensure that they perform effectively.

{{</citation>}}


### (11/35) Evaluating Code Summarization Techniques: A New Metric and an Empirical Characterization (Antonio Mastropaolo et al., 2023)

{{<citation>}}

Antonio Mastropaolo, Matteo Ciniselli, Massimiliano Di Penta, Gabriele Bavota. (2023)  
**Evaluating Code Summarization Techniques: A New Metric and an Empirical Characterization**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU, Summarization  
[Paper Link](http://arxiv.org/abs/2312.15475v1)  

---


**ABSTRACT**  
Several code summarization techniques have been proposed in the literature to automatically document a code snippet or a function. Ideally, software developers should be involved in assessing the quality of the generated summaries. However, in most cases, researchers rely on automatic evaluation metrics such as BLEU, ROUGE, and METEOR. These metrics are all based on the same assumption: The higher the textual similarity between the generated summary and a reference summary written by developers, the higher its quality. However, there are two reasons for which this assumption falls short: (i) reference summaries, e.g., code comments collected by mining software repositories, may be of low quality or even outdated; (ii) generated summaries, while using a different wording than a reference one, could be semantically equivalent to it, thus still being suitable to document the code snippet. In this paper, we perform a thorough empirical investigation on the complementarity of different types of metrics in capturing the quality of a generated summary. Also, we propose to address the limitations of existing metrics by considering a new dimension, capturing the extent to which the generated summary aligns with the semantics of the documented code snippet, independently from the reference summary. To this end, we present a new metric based on contrastive learning to capture said aspect. We empirically show that the inclusion of this novel dimension enables a more effective representation of developers' evaluations regarding the quality of automatically generated summaries.

{{</citation>}}


## cs.CV (7)



### (12/35) Amodal Completion via Progressive Mixed Context Diffusion (Katherine Xu et al., 2023)

{{<citation>}}

Katherine Xu, Lingzhi Zhang, Jianbo Shi. (2023)  
**Amodal Completion via Progressive Mixed Context Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15540v1)  

---


**ABSTRACT**  
Our brain can effortlessly recognize objects even when partially hidden from view. Seeing the visible of the hidden is called amodal completion; however, this task remains a challenge for generative AI despite rapid progress. We propose to sidestep many of the difficulties of existing approaches, which typically involve a two-step process of predicting amodal masks and then generating pixels. Our method involves thinking outside the box, literally! We go outside the object bounding box to use its context to guide a pre-trained diffusion inpainting model, and then progressively grow the occluded object and trim the extra background. We overcome two technical challenges: 1) how to be free of unwanted co-occurrence bias, which tends to regenerate similar occluders, and 2) how to judge if an amodal completion has succeeded. Our amodal completion method exhibits improved photorealistic completion results compared to existing approaches in numerous successful completion cases. And the best part? It doesn't require any special training or fine-tuning of models.

{{</citation>}}


### (13/35) Towards Reliable AI Model Deployments: Multiple Input Mixup for Out-of-Distribution Detection (Dasol Choi et al., 2023)

{{<citation>}}

Dasol Choi, Dongbin Na. (2023)  
**Towards Reliable AI Model Deployments: Multiple Input Mixup for Out-of-Distribution Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15514v1)  

---


**ABSTRACT**  
Recent remarkable success in the deep-learning industries has unprecedentedly increased the need for reliable model deployment. For example, the model should alert the user if the produced model outputs might not be reliable. Previous studies have proposed various methods to solve the Out-of-Distribution (OOD) detection problem, however, they generally require a burden of resources. In this work, we propose a novel and simple method, Multiple Input Mixup (MIM). Our method can help improve the OOD detection performance with only single epoch fine-tuning. Our method does not require training the model from scratch and can be attached to the classifier simply. Despite its simplicity, our MIM shows competitive performance. Our method can be suitable for various environments because our method only utilizes the In-Distribution (ID) samples to generate the synthesized OOD data. With extensive experiments with CIFAR10 and CIFAR100 benchmarks that have been largely adopted in out-of-distribution detection fields, we have demonstrated our MIM shows comprehensively superior performance compared to the SOTA method. Especially, our method does not need additional computation on the feature vectors compared to the previous studies. All source codes are publicly available at https://github.com/ndb796/MultipleInputMixup.

{{</citation>}}


### (14/35) iDet3D: Towards Efficient Interactive Object Detection for LiDAR Point Clouds (Dongmin Choi et al., 2023)

{{<citation>}}

Dongmin Choi, Wonwoo Cho, Kangyeol Kim, Jaegul Choo. (2023)  
**iDet3D: Towards Efficient Interactive Object Detection for LiDAR Point Clouds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.15449v1)  

---


**ABSTRACT**  
Accurately annotating multiple 3D objects in LiDAR scenes is laborious and challenging. While a few previous studies have attempted to leverage semi-automatic methods for cost-effective bounding box annotation, such methods have limitations in efficiently handling numerous multi-class objects. To effectively accelerate 3D annotation pipelines, we propose iDet3D, an efficient interactive 3D object detector. Supporting a user-friendly 2D interface, which can ease the cognitive burden of exploring 3D space to provide click interactions, iDet3D enables users to annotate the entire objects in each scene with minimal interactions. Taking the sparse nature of 3D point clouds into account, we design a negative click simulation (NCS) to improve accuracy by reducing false-positive predictions. In addition, iDet3D incorporates two click propagation techniques to take full advantage of user interactions: (1) dense click guidance (DCG) for keeping user-provided information throughout the network and (2) spatial click propagation (SCP) for detecting other instances of the same class based on the user-specified objects. Through our extensive experiments, we present that our method can construct precise annotations in a few clicks, which shows the practicality as an efficient annotation tool for 3D object detection.

{{</citation>}}


### (15/35) Make-A-Character: High Quality Text-to-3D Character Generation within Minutes (Jianqiang Ren et al., 2023)

{{<citation>}}

Jianqiang Ren, Chao He, Lin Liu, Jiahao Chen, Yutong Wang, Yafei Song, Jianfang Li, Tangli Xue, Siqi Hu, Tao Chen, Kunkun Zheng, Jianjing Xiang, Liefeng Bo. (2023)  
**Make-A-Character: High Quality Text-to-3D Character Generation within Minutes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15430v1)  

---


**ABSTRACT**  
There is a growing demand for customized and expressive 3D characters with the emergence of AI agents and Metaverse, but creating 3D characters using traditional computer graphics tools is a complex and time-consuming task. To address these challenges, we propose a user-friendly framework named Make-A-Character (Mach) to create lifelike 3D avatars from text descriptions. The framework leverages the power of large language and vision models for textual intention understanding and intermediate image generation, followed by a series of human-oriented visual perception and 3D generation modules. Our system offers an intuitive approach for users to craft controllable, realistic, fully-realized 3D characters that meet their expectations within 2 minutes, while also enabling easy integration with existing CG pipeline for dynamic expressiveness. For more information, please visit the project page at https://human3daigc.github.io/MACH/.

{{</citation>}}


### (16/35) Knowledge Guided Semi-Supervised Learning for Quality Assessment of User Generated Videos (Shankhanil Mitra et al., 2023)

{{<citation>}}

Shankhanil Mitra, Rajiv Soundararajan. (2023)  
**Knowledge Guided Semi-Supervised Learning for Quality Assessment of User Generated Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: QA, Representation Learning, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.15425v1)  

---


**ABSTRACT**  
Perceptual quality assessment of user generated content (UGC) videos is challenging due to the requirement of large scale human annotated videos for training. In this work, we address this challenge by first designing a self-supervised Spatio-Temporal Visual Quality Representation Learning (ST-VQRL) framework to generate robust quality aware features for videos. Then, we propose a dual-model based Semi Supervised Learning (SSL) method specifically designed for the Video Quality Assessment (SSL-VQA) task, through a novel knowledge transfer of quality predictions between the two models. Our SSL-VQA method uses the ST-VQRL backbone to produce robust performances across various VQA datasets including cross-database settings, despite being learned with limited human annotated videos. Our model improves the state-of-the-art performance when trained only with limited data by around 10%, and by around 15% when unlabelled data is also used in SSL. Source codes and checkpoints are available at https://github.com/Shankhanil006/SSL-VQA.

{{</citation>}}


### (17/35) Debiased Learning for Remote Sensing Data (Chun-Hsiao Yeh et al., 2023)

{{<citation>}}

Chun-Hsiao Yeh, Xudong Wang, Stella X. Yu, Charles Hill, Zackery Steck, Scott Kangas, Aaron Reite. (2023)  
**Debiased Learning for Remote Sensing Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.15393v1)  

---


**ABSTRACT**  
Deep learning has had remarkable success at analyzing handheld imagery such as consumer photos due to the availability of large-scale human annotations (e.g., ImageNet). However, remote sensing data lacks such extensive annotation and thus potential for supervised learning. To address this, we propose a highly effective semi-supervised approach tailored specifically to remote sensing data. Our approach encompasses two key contributions. First, we adapt the FixMatch framework to remote sensing data by designing robust strong and weak augmentations suitable for this domain. Second, we develop an effective semi-supervised learning method by removing bias in imbalanced training data resulting from both actual labels and pseudo-labels predicted by the model. Our simple semi-supervised framework was validated by extensive experimentation. Using 30\% of labeled annotations, it delivers a 7.1\% accuracy gain over the supervised learning baseline and a 2.1\% gain over the supervised state-of-the-art CDS method on the remote sensing xView dataset.

{{</citation>}}


### (18/35) End-to-End 3D Object Detection using LiDAR Point Cloud (Gaurav Raut et al., 2023)

{{<citation>}}

Gaurav Raut, Advait Patole. (2023)  
**End-to-End 3D Object Detection using LiDAR Point Cloud**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.15377v1)  

---


**ABSTRACT**  
There has been significant progress made in the field of autonomous vehicles. Object detection and tracking are the primary tasks for any autonomous vehicle. The task of object detection in autonomous vehicles relies on a variety of sensors like cameras, and LiDAR. Although image features are typically preferred, numerous approaches take spatial data as input. Exploiting this information we present an approach wherein, using a novel encoding of the LiDAR point cloud we infer the location of different classes near the autonomous vehicles. This approach does not implement a bird's eye view approach, which is generally applied for this application and thus saves the extensive pre-processing required. After studying the numerous networks and approaches used to solve this approach, we have implemented a novel model with the intention to inculcate their advantages and avoid their shortcomings. The output is predictions about the location and orientation of objects in the scene in form of 3D bounding boxes and labels of scene objects.

{{</citation>}}


## cs.IR (3)



### (19/35) Aspect category learning and sentimental analysis using weakly supervised learning (Kalpa Subbaih et al., 2023)

{{<citation>}}

Kalpa Subbaih, Bharath Kumar Bolla. (2023)  
**Aspect category learning and sentimental analysis using weakly supervised learning**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.15526v1)  

---


**ABSTRACT**  
The surge of e-commerce reviews has presented a challenge in manually annotating the vast volume of reviews to comprehend their underlying aspects and sentiments. This research focused on leveraging weakly supervised learning to tackle aspect category learning and the sentiment classification of reviews. Our approach involves the generation of labels for both aspects and sentiments, employing the Snorkel framework of WSL, which incorporates aspect terms, review sentiment scores, and review ratings as sources of weak signals. This innovative strategy significantly reduces the laborious labeling efforts required for processing such extensive datasets. In this study, we deployed hybrid models, namely BiLSTM, CNN-BiLSTM, and CNN-LSTM, which harness multiple inputs, including review text, aspect terms, and ratings. Our proposed model employs two distinct loss functions: Binary Cross Entropy with Sigmoid Activation for Multi-Label Classification, enabling us to learn aspect Labels such as Quality, Usability, Service, Size, and Price, and Categorical Cross Entropy with Softmax Activations for Multi-Class Classification. Subsequently, we meticulously evaluate the performance metrics of these three implemented models, including Macro F1 score and Macro Precision. CNN & Bi-LSTM model attained 0.78 and 0.79 F1 scores on aspect and sentiment identification, respectively. The outcomes of this research are poised to make a substantial contribution to e-commerce platforms, offering an efficient and automated means to label and analyze vast troves of user reviews.

{{</citation>}}


### (20/35) Diffusion-EXR: Controllable Review Generation for Explainable Recommendation via Diffusion Models (Ling Li et al., 2023)

{{<citation>}}

Ling Li, Shaohua Li, Winda Marantika, Alex C. Kot, Huijing Zhan. (2023)  
**Diffusion-EXR: Controllable Review Generation for Explainable Recommendation via Diffusion Models**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.15490v1)  

---


**ABSTRACT**  
Denoising Diffusion Probabilistic Model (DDPM) has shown great competence in image and audio generation tasks. However, there exist few attempts to employ DDPM in the text generation, especially review generation under recommendation systems. Fueled by the predicted reviews explainability that justifies recommendations could assist users better understand the recommended items and increase the transparency of recommendation system, we propose a Diffusion Model-based Review Generation towards EXplainable Recommendation named Diffusion-EXR. Diffusion-EXR corrupts the sequence of review embeddings by incrementally introducing varied levels of Gaussian noise to the sequence of word embeddings and learns to reconstruct the original word representations in the reverse process. The nature of DDPM enables our lightweight Transformer backbone to perform excellently in the recommendation review generation task. Extensive experimental results have demonstrated that Diffusion-EXR can achieve state-of-the-art review generation for recommendation on two publicly available benchmark datasets.

{{</citation>}}


### (21/35) Agent4Ranking: Semantic Robust Ranking via Personalized Query Rewriting Using Multi-agent LLM (Xiaopeng Li et al., 2023)

{{<citation>}}

Xiaopeng Li, Lixin Su, Pengyue Jia, Xiangyu Zhao, Suqi Cheng, Junfeng Wang, Dawei Yin. (2023)  
**Agent4Ranking: Semantic Robust Ranking via Personalized Query Rewriting Using Multi-agent LLM**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15450v1)  

---


**ABSTRACT**  
Search engines are crucial as they provide an efficient and easy way to access vast amounts of information on the internet for diverse information needs. User queries, even with a specific need, can differ significantly. Prior research has explored the resilience of ranking models against typical query variations like paraphrasing, misspellings, and order changes. Yet, these works overlook how diverse demographics uniquely formulate identical queries. For instance, older individuals tend to construct queries more naturally and in varied order compared to other groups. This demographic diversity necessitates enhancing the adaptability of ranking models to diverse query formulations. To this end, in this paper, we propose a framework that integrates a novel rewriting pipeline that rewrites queries from various demographic perspectives and a novel framework to enhance ranking robustness. To be specific, we use Chain of Thought (CoT) technology to utilize Large Language Models (LLMs) as agents to emulate various demographic profiles, then use them for efficient query rewriting, and we innovate a robust Multi-gate Mixture of Experts (MMoE) architecture coupled with a hybrid loss function, collectively strengthening the ranking models' robustness. Our extensive experimentation on both public and industrial datasets assesses the efficacy of our query rewriting approach and the enhanced accuracy and robustness of the ranking model. The findings highlight the sophistication and effectiveness of our proposed model.

{{</citation>}}


## cs.AI (1)



### (22/35) The Challenge of Using LLMs to Simulate Human Behavior: A Causal Inference Perspective (George Gui et al., 2023)

{{<citation>}}

George Gui, Olivier Toubia. (2023)  
**The Challenge of Using LLMs to Simulate Human Behavior: A Causal Inference Perspective**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IR, cs.AI, econ-EM, stat-AP  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15524v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated impressive potential to simulate human behavior. Using a causal inference framework, we empirically and theoretically analyze the challenges of conducting LLM-simulated experiments, and explore potential solutions. In the context of demand estimation, we show that variations in the treatment included in the prompt (e.g., price of focal product) can cause variations in unspecified confounding factors (e.g., price of competitors, historical prices, outside temperature), introducing endogeneity and yielding implausibly flat demand curves. We propose a theoretical framework suggesting this endogeneity issue generalizes to other contexts and won't be fully resolved by merely improving the training data. Unlike real experiments where researchers assign pre-existing units across conditions, LLMs simulate units based on the entire prompt, which includes the description of the treatment. Therefore, due to associations in the training data, the characteristics of individuals and environments simulated by the LLM can be affected by the treatment assignment. We explore two potential solutions. The first specifies all contextual variables that affect both treatment and outcome, which we demonstrate to be challenging for a general-purpose LLM. The second explicitly specifies the source of treatment variation in the prompt given to the LLM (e.g., by informing the LLM that the store is running an experiment). While this approach only allows the estimation of a conditional average treatment effect that depends on the specific experimental design, it provides valuable directional results for exploratory analysis.

{{</citation>}}


## cs.CY (1)



### (23/35) The Persuasive Power of Large Language Models (Simon Martin Breum et al., 2023)

{{<citation>}}

Simon Martin Breum, Daniel Vædele Egdal, Victor Gram Mortensen, Anders Giovanni Møller, Luca Maria Aiello. (2023)  
**The Persuasive Power of Large Language Models**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs-HC, cs.CY, physics-soc-ph  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15523v1)  

---


**ABSTRACT**  
The increasing capability of Large Language Models to act as human-like social agents raises two important questions in the area of opinion dynamics. First, whether these agents can generate effective arguments that could be injected into the online discourse to steer the public opinion. Second, whether artificial agents can interact with each other to reproduce dynamics of persuasion typical of human social systems, opening up opportunities for studying synthetic social systems as faithful proxies for opinion dynamics in human populations. To address these questions, we designed a synthetic persuasion dialogue scenario on the topic of climate change, where a 'convincer' agent generates a persuasive argument for a 'skeptic' agent, who subsequently assesses whether the argument changed its internal opinion state. Different types of arguments were generated to incorporate different linguistic dimensions underpinning psycho-linguistic theories of opinion change. We then asked human judges to evaluate the persuasiveness of machine-generated arguments. Arguments that included factual knowledge, markers of trust, expressions of support, and conveyed status were deemed most effective according to both humans and agents, with humans reporting a marked preference for knowledge-based arguments. Our experimental framework lays the groundwork for future in-silico studies of opinion dynamics, and our findings suggest that artificial agents have the potential of playing an important role in collective processes of opinion formation in online social media.

{{</citation>}}


## cs.LG (3)



### (24/35) Graph Coarsening via Convolution Matching for Scalable Graph Neural Network Training (Charles Dickens et al., 2023)

{{<citation>}}

Charles Dickens, Eddie Huang, Aishwarya Reganti, Jiong Zhu, Karthik Subbian, Danai Koutra. (2023)  
**Graph Coarsening via Convolution Matching for Scalable Graph Neural Network Training**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2312.15520v1)  

---


**ABSTRACT**  
Graph summarization as a preprocessing step is an effective and complementary technique for scalable graph neural network (GNN) training. In this work, we propose the Coarsening Via Convolution Matching (CONVMATCH) algorithm and a highly scalable variant, A-CONVMATCH, for creating summarized graphs that preserve the output of graph convolution. We evaluate CONVMATCH on six real-world link prediction and node classification graph datasets, and show it is efficient and preserves prediction performance while significantly reducing the graph size. Notably, CONVMATCH achieves up to 95% of the prediction performance of GNNs on node classification while trained on graphs summarized down to 1% the size of the original graph. Furthermore, on link prediction tasks, CONVMATCH consistently outperforms all baselines, achieving up to a 2x improvement.

{{</citation>}}


### (25/35) A Trust Region Approach for Few-Shot Sim-to-Real Reinforcement Learning (Paul Daoudi et al., 2023)

{{<citation>}}

Paul Daoudi, Christophe Prieur, Bogdan Robu, Merwan Barlier, Ludovic Dos Santos. (2023)  
**A Trust Region Approach for Few-Shot Sim-to-Real Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Few-Shot, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15474v1)  

---


**ABSTRACT**  
Simulation-to-Reality Reinforcement Learning (Sim-to-Real RL) seeks to use simulations to minimize the need for extensive real-world interactions. Specifically, in the few-shot off-dynamics setting, the goal is to acquire a simulator-based policy despite a dynamics mismatch that can be effectively transferred to the real-world using only a handful of real-world transitions. In this context, conventional RL agents tend to exploit simulation inaccuracies resulting in policies that excel in the simulator but underperform in the real environment. To address this challenge, we introduce a novel approach that incorporates a penalty to constrain the trajectories induced by the simulator-trained policy inspired by recent advances in Imitation Learning and Trust Region based RL algorithms. We evaluate our method across various environments representing diverse Sim-to-Real conditions, where access to the real environment is extremely limited. These experiments include high-dimensional systems relevant to real-world applications. Across most tested scenarios, our proposed method demonstrates performance improvements compared to existing baselines.

{{</citation>}}


### (26/35) CARSS: Cooperative Attention-guided Reinforcement Subpath Synthesis for Solving Traveling Salesman Problem (Yuchen Shi et al., 2023)

{{<citation>}}

Yuchen Shi, Congying Han, Tiande Guo. (2023)  
**CARSS: Cooperative Attention-guided Reinforcement Subpath Synthesis for Solving Traveling Salesman Problem**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-MA, cs.LG  
Keywords: Attention, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15412v1)  

---


**ABSTRACT**  
This paper introduces CARSS (Cooperative Attention-guided Reinforcement Subpath Synthesis), a novel approach to address the Traveling Salesman Problem (TSP) by leveraging cooperative Multi-Agent Reinforcement Learning (MARL). CARSS decomposes the TSP solving process into two distinct yet synergistic steps: "subpath generation" and "subpath merging." In the former, a cooperative MARL framework is employed to iteratively generate subpaths using multiple agents. In the latter, these subpaths are progressively merged to form a complete cycle. The algorithm's primary objective is to enhance efficiency in terms of training memory consumption, testing time, and scalability, through the adoption of a multi-agent divide and conquer paradigm. Notably, attention mechanisms play a pivotal role in feature embedding and parameterization strategies within CARSS. The training of the model is facilitated by the independent REINFORCE algorithm. Empirical experiments reveal CARSS's superiority compared to single-agent alternatives: it demonstrates reduced GPU memory utilization, accommodates training graphs nearly 2.5 times larger, and exhibits the potential for scaling to even more extensive problem sizes. Furthermore, CARSS substantially reduces testing time and optimization gaps by approximately 50% for TSP instances of up to 1000 vertices, when compared to standard decoding methods.

{{</citation>}}


## eess.SY (1)



### (27/35) Agent based modelling for continuously varying supply chains (Wan Wang et al., 2023)

{{<citation>}}

Wan Wang, Haiyan Wang, Adam J. Sobey. (2023)  
**Agent based modelling for continuously varying supply chains**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: LSTM, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15502v1)  

---


**ABSTRACT**  
Problem definition: Supply chains are constantly evolving networks. Reinforcement learning is increasingly proposed as a solution to provide optimal control of these networks. Academic/practical: However, learning in continuously varying environments remains a challenge in the reinforcement learning literature.Methodology: This paper therefore seeks to address whether agents can control varying supply chain problems, transferring learning between environments that require different strategies and avoiding catastrophic forgetting of tasks that have not been seen in a while. To evaluate this approach, two state-of-the-art Reinforcement Learning (RL) algorithms are compared: an actor-critic learner, Proximal Policy Optimisation(PPO), and a Recurrent Proximal Policy Optimisation (RPPO), PPO with a Long Short-Term Memory(LSTM) layer, which is showing popularity in online learning environments. Results: First these methods are compared on six sets of environments with varying degrees of stochasticity. The results show that more lean strategies adopted in Batch environments are different from those adopted in Stochastic environments with varying products. The methods are also compared on various continuous supply chain scenarios, where the PPO agents are shown to be able to adapt through continuous learning when the tasks are similar but show more volatile performance when changing between the extreme tasks. However, the RPPO, with an ability to remember histories, is able to overcome this to some extent and takes on a more realistic strategy. Managerial implications: Our results provide a new perspective on the continuously varying supply chain, the cooperation and coordination of agents are crucial for improving the overall performance in uncertain and semi-continuous non-stationary supply chain environments without the need to retrain the environment as the demand changes.

{{</citation>}}


## cs.DB (1)



### (28/35) Towards Consistent Language Models Using Declarative Constraints (Jasmin Mousavi et al., 2023)

{{<citation>}}

Jasmin Mousavi, Arash Termehchy. (2023)  
**Towards Consistent Language Models Using Declarative Constraints**  

---
Primary Category: cs.DB  
Categories: cs-CL, cs-DB, cs.DB  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.15472v1)  

---


**ABSTRACT**  
Large language models have shown unprecedented abilities in generating linguistically coherent and syntactically correct natural language output. However, they often return incorrect and inconsistent answers to input questions. Due to the complexity and uninterpretability of the internally learned representations, it is challenging to modify language models such that they provide correct and consistent results. The data management community has developed various methods and tools for providing consistent answers over inconsistent datasets. In these methods, users specify the desired properties of data in a domain in the form of high-level declarative constraints. This approach has provided usable and scalable methods to delivering consistent information from inconsistent datasets. We aim to build upon this success and leverage these methods to modify language models such that they deliver consistent and accurate results. We investigate the challenges of using these ideas to obtain consistent and relevant answers from language models and report some preliminary empirical studies.

{{</citation>}}


## eess.AS (1)



### (29/35) Consistent and Relevant: Rethink the Query Embedding in General Sound Separation (Yuanyuan Wang et al., 2023)

{{<citation>}}

Yuanyuan Wang, Hangting Chen, Dongchao Yang, Jianwei Yu, Chao Weng, Zhiyong Wu, Helen Meng. (2023)  
**Consistent and Relevant: Rethink the Query Embedding in General Sound Separation**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.15463v1)  

---


**ABSTRACT**  
The query-based audio separation usually employs specific queries to extract target sources from a mixture of audio signals. Currently, most query-based separation models need additional networks to obtain query embedding. In this way, separation model is optimized to be adapted to the distribution of query embedding. However, query embedding may exhibit mismatches with separation models due to inconsistent structures and independent information. In this paper, we present CaRE-SEP, a consistent and relevant embedding network for general sound separation to encourage a comprehensive reconsideration of query usage in audio separation. CaRE-SEP alleviates the potential mismatch between queries and separation in two aspects, including sharing network structure and sharing feature information. First, a Swin-Unet model with a shared encoder is conducted to unify query encoding and sound separation into one model, eliminating the network architecture difference and generating consistent distribution of query and separation features. Second, by initializing CaRE-SEP with a pretrained classification network and allowing gradient backpropagation, the query embedding is optimized to be relevant to the separation feature, further alleviating the feature mismatch problem. Experimental results indicate the proposed CaRE-SEP model substantially improves the performance of separation tasks. Moreover, visualizations validate the potential mismatch and how CaRE-SEP solves it.

{{</citation>}}


## stat.ML (1)



### (30/35) Conservative Exploration for Policy Optimization via Off-Policy Policy Evaluation (Paul Daoudi et al., 2023)

{{<citation>}}

Paul Daoudi, Mathias Formoso, Othman Gaizi, Achraf Azize, Evrard Garcelon. (2023)  
**Conservative Exploration for Policy Optimization via Off-Policy Policy Evaluation**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15458v1)  

---


**ABSTRACT**  
A precondition for the deployment of a Reinforcement Learning agent to a real-world system is to provide guarantees on the learning process. While a learning algorithm will eventually converge to a good policy, there are no guarantees on the performance of the exploratory policies. We study the problem of conservative exploration, where the learner must at least be able to guarantee its performance is at least as good as a baseline policy. We propose the first conservative provably efficient model-free algorithm for policy optimization in continuous finite-horizon problems. We leverage importance sampling techniques to counterfactually evaluate the conservative condition from the data self-generated by the algorithm. We derive a regret bound and show that (w.h.p.) the conservative constraint is never violated during learning. Finally, we leverage these insights to build a general schema for conservative exploration in DeepRL via off-policy policy evaluation techniques. We show empirically the effectiveness of our methods.

{{</citation>}}


## cs.ET (1)



### (31/35) Variation-Resilient FeFET-Based In-Memory Computing Leveraging Probabilistic Deep Learning (Bibhas Manna et al., 2023)

{{<citation>}}

Bibhas Manna, Arnob Saha, Zhouhang Jiang, Kai Ni, Abhronil Sengupta. (2023)  
**Variation-Resilient FeFET-Based In-Memory Computing Leveraging Probabilistic Deep Learning**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs.ET  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.15444v1)  

---


**ABSTRACT**  
Reliability issues stemming from device level non-idealities of non-volatile emerging technologies like ferroelectric field-effect transistors (FeFET), especially at scaled dimensions, cause substantial degradation in the accuracy of In-Memory crossbar-based AI systems. In this work, we present a variation-aware design technique to characterize the device level variations and to mitigate their impact on hardware accuracy employing a Bayesian Neural Network (BNN) approach. An effective conductance variation model is derived from the experimental measurements of cycle-to-cycle (C2C) and device-to-device (D2D) variations performed on FeFET devices fabricated using 28 nm high-$k$ metal gate technology. The variations were found to be a function of different conductance states within the given programming range, which sharply contrasts earlier efforts where a fixed variation dispersion was considered for all conductance values. Such variation characteristics formulated for three different device sizes at different read voltages were provided as prior variation information to the BNN to yield a more exact and reliable inference. Near-ideal accuracy for shallow networks (MLP5 and LeNet models) on the MNIST dataset and limited accuracy decline by $\sim$3.8-16.1% for deeper AlexNet models on CIFAR10 dataset under a wide range of variations corresponding to different device sizes and read voltages, demonstrates the efficacy of our proposed device-algorithm co-design technique.

{{</citation>}}


## cs.SD (1)



### (32/35) Combinatorial music generation model with song structure graph analysis (Seonghyeon Go et al., 2023)

{{<citation>}}

Seonghyeon Go, Kyogu Lee. (2023)  
**Combinatorial music generation model with song structure graph analysis**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2312.15400v1)  

---


**ABSTRACT**  
In this work, we propose a symbolic music generation model with the song structure graph analysis network. We construct a graph that uses information such as note sequence and instrument as node features, while the correlation between note sequences acts as the edge feature. We trained a Graph Neural Network to obtain node representation in the graph, then we use node representation as input of Unet to generate CONLON pianoroll image latent. The outcomes of our experimental results show that the proposed model can generate a comprehensive form of music. Our approach represents a promising and innovative method for symbolic music generation and holds potential applications in various fields in Music Information Retreival, including music composition, music classification, and music inpainting systems.

{{</citation>}}


## cs.DC (1)



### (33/35) DEAP: Design Space Exploration for DNN Accelerator Parallelism (Ekansh Agrawal et al., 2023)

{{<citation>}}

Ekansh Agrawal, Xiangyu Sam Xu. (2023)  
**DEAP: Design Space Exploration for DNN Accelerator Parallelism**  

---
Primary Category: cs.DC  
Categories: cs-AR, cs-DC, cs-LG, cs.DC  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.15388v1)  

---


**ABSTRACT**  
The boom in Large Language Models (LLMs) like GPT-4 and ChatGPT has marked a significant advancement in artificial intelligence. These models are becoming increasingly complex and powerful to train and serve. This growth in capabilities comes with a substantial increase in computational requirements, both in terms of hardware resources and energy consumption. The goal of this paper is to showcase how hardware and software co-design can come together and allow us to create customized hardware systems for specific LLM workloads. We propose a simulation workflow that allows us to combine model parallelism techniques with a multi-accelerator simulation framework for efficiency metrics. We focus on inference workloads and report power, cycle, and latency metrics upon performing a design space exploration search over multiple software and hardware configurations.

{{</citation>}}


## q-bio.QM (1)



### (34/35) MotifPiece: A Data-Driven Approach for Effective Motif Extraction and Molecular Representation Learning (Zhaoning Yu et al., 2023)

{{<citation>}}

Zhaoning Yu, Hongyang Gao. (2023)  
**MotifPiece: A Data-Driven Approach for Effective Motif Extraction and Molecular Representation Learning**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, q-bio-QM, q-bio.QM  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.15387v1)  

---


**ABSTRACT**  
Motif extraction is an important task in motif based molecular representation learning. Previously, machine learning approaches employing either rule-based or string-based techniques to extract motifs. Rule-based approaches may extract motifs that aren't frequent or prevalent within the molecular data, which can lead to an incomplete understanding of essential structural patterns in molecules. String-based methods often lose the topological information inherent in molecules. This can be a significant drawback because topology plays a vital role in defining the spatial arrangement and connectivity of atoms within a molecule, which can be critical for understanding its properties and behavior. In this paper, we develop a data-driven motif extraction technique known as MotifPiece, which employs statistical measures to define motifs. To comprehensively evaluate the effectiveness of MotifPiece, we introduce a heterogeneous learning module. Our model shows an improvement compared to previously reported models. Additionally, we demonstrate that its performance can be further enhanced in two ways: first, by incorporating more data to aid in generating a richer motif vocabulary, and second, by merging multiple datasets that share enough motifs, allowing for cross-dataset learning.

{{</citation>}}


## q-fin.MF (1)



### (35/35) Discrete-Time Mean-Variance Strategy Based on Reinforcement Learning (Xiangyu Cui et al., 2023)

{{<citation>}}

Xiangyu Cui, Xun Li, Yun Shi, Si Zhao. (2023)  
**Discrete-Time Mean-Variance Strategy Based on Reinforcement Learning**  

---
Primary Category: q-fin.MF  
Categories: cs-LG, q-fin-MF, q-fin-PM, q-fin.MF  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.15385v1)  

---


**ABSTRACT**  
This paper studies a discrete-time mean-variance model based on reinforcement learning. Compared with its continuous-time counterpart in \cite{zhou2020mv}, the discrete-time model makes more general assumptions about the asset's return distribution. Using entropy to measure the cost of exploration, we derive the optimal investment strategy, whose density function is also Gaussian type. Additionally, we design the corresponding reinforcement learning algorithm. Both simulation experiments and empirical analysis indicate that our discrete-time model exhibits better applicability when analyzing real-world data than the continuous-time model.

{{</citation>}}
