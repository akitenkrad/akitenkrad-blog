---
draft: false
title: "arXiv @ 2023.08.05"
date: 2023-08-05
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.05"
    identifier: arxiv_20230805
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (15)](#cslg-15)
- [cs.CL (26)](#cscl-26)
- [cs.CV (23)](#cscv-23)
- [eess.IV (3)](#eessiv-3)
- [cs.IR (3)](#csir-3)
- [cs.PL (1)](#cspl-1)
- [cs.SD (2)](#cssd-2)
- [cs.CY (2)](#cscy-2)
- [cs.CR (1)](#cscr-1)
- [cs.OH (1)](#csoh-1)
- [cs.DL (1)](#csdl-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.AI (4)](#csai-4)
- [quant-ph (1)](#quant-ph-1)
- [cs.SI (1)](#cssi-1)
- [cs.RO (4)](#csro-4)
- [math.NA (1)](#mathna-1)
- [eess.SY (1)](#eesssy-1)
- [cs.HC (2)](#cshc-2)
- [cond-mat.dis-nn (1)](#cond-matdis-nn-1)
- [cs.MM (1)](#csmm-1)
- [cs.MA (1)](#csma-1)
- [eess.SP (1)](#eesssp-1)

## cs.LG (15)



### (1/97) Target specification bias, counterfactual prediction, and algorithmic fairness in healthcare (Eran Tal, 2023)

{{<citation>}}

Eran Tal. (2023)  
**Target specification bias, counterfactual prediction, and algorithmic fairness in healthcare**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG, stat-ME  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.02081v1)  

---


**ABSTRACT**  
Bias in applications of machine learning (ML) to healthcare is usually attributed to unrepresentative or incomplete data, or to underlying health disparities. This article identifies a more pervasive source of bias that affects the clinical utility of ML-enabled prediction tools: target specification bias. Target specification bias arises when the operationalization of the target variable does not match its definition by decision makers. The mismatch is often subtle, and stems from the fact that decision makers are typically interested in predicting the outcomes of counterfactual, rather than actual, healthcare scenarios. Target specification bias persists independently of data limitations and health disparities. When left uncorrected, it gives rise to an overestimation of predictive accuracy, to inefficient utilization of medical resources, and to suboptimal decisions that can harm patients. Recent work in metrology - the science of measurement - suggests ways of counteracting target specification bias and avoiding its harmful consequences.

{{</citation>}}


### (2/97) Accurate Neural Network Pruning Requires Rethinking Sparse Optimization (Denis Kuznedelev et al., 2023)

{{<citation>}}

Denis Kuznedelev, Eldar Kurtic, Eugenia Iofinova, Elias Frantar, Alexandra Peste, Dan Alistarh. (2023)  
**Accurate Neural Network Pruning Requires Rethinking Sparse Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: BERT, GLUE, ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2308.02060v1)  

---


**ABSTRACT**  
Obtaining versions of deep neural networks that are both highly-accurate and highly-sparse is one of the main challenges in the area of model compression, and several high-performance pruning techniques have been investigated by the community. Yet, much less is known about the interaction between sparsity and the standard stochastic optimization techniques used for training sparse networks, and most existing work uses standard dense schedules and hyperparameters for training sparse networks. In this work, we examine the impact of high sparsity on model training using the standard computer vision and natural language processing sparsity benchmarks. We begin by showing that using standard dense training recipes for sparse training is suboptimal, and results in under-training. We provide new approaches for mitigating this issue for both sparse pre-training of vision models (e.g. ResNet50/ImageNet) and sparse fine-tuning of language models (e.g. BERT/GLUE), achieving state-of-the-art results in both settings in the high-sparsity regime, and providing detailed analyses for the difficulty of sparse training in both scenarios. Our work sets a new threshold in terms of the accuracies that can be achieved under high sparsity, and should inspire further research into improving sparse model training, to reach higher accuracies under high sparsity, but also to do so efficiently.

{{</citation>}}


### (3/97) SMARLA: A Safety Monitoring Approach for Deep Reinforcement Learning Agents (Amirhossein Zolfagharian et al., 2023)

{{<citation>}}

Amirhossein Zolfagharian, Manel Abdellatif, Lionel C. Briand, Ramesh S. (2023)  
**SMARLA: A Safety Monitoring Approach for Deep Reinforcement Learning Agents**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SE, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02594v1)  

---


**ABSTRACT**  
Deep reinforcement learning algorithms (DRL) are increasingly being used in safety-critical systems. Ensuring the safety of DRL agents is a critical concern in such contexts. However, relying solely on testing is not sufficient to ensure safety as it does not offer guarantees. Building safety monitors is one solution to alleviate this challenge. This paper proposes SMARLA, a machine learning-based safety monitoring approach designed for DRL agents. For practical reasons, SMARLA is designed to be black-box (as it does not require access to the internals of the agent) and leverages state abstraction to reduce the state space and thus facilitate the learning of safety violation prediction models from agent's states. We validated SMARLA on two well-known RL case studies. Empirical analysis reveals that SMARLA achieves accurate violation prediction with a low false positive rate, and can predict safety violations at an early stage, approximately halfway through the agent's execution before violations occur.

{{</citation>}}


### (4/97) Domain specificity and data efficiency in typo tolerant spell checkers: the case of search in online marketplaces (Dayananda Ubrangala et al., 2023)

{{<citation>}}

Dayananda Ubrangala, Juhi Sharma, Ravi Prasad Kondapalli, Kiran R, Amit Agarwala, Laurent Boué. (2023)  
**Domain specificity and data efficiency in typo tolerant spell checkers: the case of search in online marketplaces**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.LG  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2308.01976v1)  

---


**ABSTRACT**  
Typographical errors are a major source of frustration for visitors of online marketplaces. Because of the domain-specific nature of these marketplaces and the very short queries users tend to search for, traditional spell cheking solutions do not perform well in correcting typos. We present a data augmentation method to address the lack of annotated typo data and train a recurrent neural network to learn context-limited domain-specific embeddings. Those embeddings are deployed in a real-time inferencing API for the Microsoft AppSource marketplace to find the closest match between a misspelled user query and the available product names. Our data efficient solution shows that controlled high quality synthetic data may be a powerful tool especially considering the current climate of large language models which rely on prohibitively huge and often uncontrolled datasets.

{{</citation>}}


### (5/97) URET: Universal Robustness Evaluation Toolkit (for Evasion) (Kevin Eykholt et al., 2023)

{{<citation>}}

Kevin Eykholt, Taesung Lee, Douglas Schales, Jiyong Jang, Ian Molloy, Masha Zorin. (2023)  
**URET: Universal Robustness Evaluation Toolkit (for Evasion)**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01840v1)  

---


**ABSTRACT**  
Machine learning models are known to be vulnerable to adversarial evasion attacks as illustrated by image classification models. Thoroughly understanding such attacks is critical in order to ensure the safety and robustness of critical AI tasks. However, most evasion attacks are difficult to deploy against a majority of AI systems because they have focused on image domain with only few constraints. An image is composed of homogeneous, numerical, continuous, and independent features, unlike many other input types to AI systems used in practice. Furthermore, some input types include additional semantic and functional constraints that must be observed to generate realistic adversarial inputs. In this work, we propose a new framework to enable the generation of adversarial inputs irrespective of the input type and task domain. Given an input and a set of pre-defined input transformations, our framework discovers a sequence of transformations that result in a semantically correct and functional adversarial input. We demonstrate the generality of our approach on several diverse machine learning tasks with various input representations. We also show the importance of generating adversarial examples as they enable the deployment of mitigation techniques.

{{</citation>}}


### (6/97) Multitask Learning with No Regret: from Improved Confidence Bounds to Active Learning (Pier Giuseppe Sessa et al., 2023)

{{<citation>}}

Pier Giuseppe Sessa, Pierre Laforgue, Nicolò Cesa-Bianchi, Andreas Krause. (2023)  
**Multitask Learning with No Regret: from Improved Confidence Bounds to Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2308.01744v1)  

---


**ABSTRACT**  
Multitask learning is a powerful framework that enables one to simultaneously learn multiple related tasks by sharing information between them. Quantifying uncertainty in the estimated tasks is of pivotal importance for many downstream applications, such as online or active learning. In this work, we provide novel multitask confidence intervals in the challenging agnostic setting, i.e., when neither the similarity between tasks nor the tasks' features are available to the learner. The obtained intervals do not require i.i.d. data and can be directly applied to bound the regret in online learning. Through a refined analysis of the multitask information gain, we obtain new regret guarantees that, depending on a task similarity parameter, can significantly improve over treating tasks independently. We further propose a novel online learning algorithm that achieves such improved regret without knowing this parameter in advance, i.e., automatically adapting to task similarity. As a second key application of our results, we introduce a novel multitask active learning setup where several tasks must be simultaneously optimized, but only one of them can be queried for feedback by the learner at each round. For this problem, we design a no-regret algorithm that uses our confidence intervals to decide which task should be queried. Finally, we empirically validate our bounds and algorithms on synthetic and real-world (drug discovery) data.

{{</citation>}}


### (7/97) Evaluating Link Prediction Explanations for Graph Neural Networks (Claudio Borile et al., 2023)

{{<citation>}}

Claudio Borile, Alan Perotti, André Panisson. (2023)  
**Evaluating Link Prediction Explanations for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.01682v1)  

---


**ABSTRACT**  
Graph Machine Learning (GML) has numerous applications, such as node/graph classification and link prediction, in real-world domains. Providing human-understandable explanations for GML models is a challenging yet fundamental task to foster their adoption, but validating explanations for link prediction models has received little attention. In this paper, we provide quantitative metrics to assess the quality of link prediction explanations, with or without ground-truth. State-of-the-art explainability methods for Graph Neural Networks are evaluated using these metrics. We discuss how underlying assumptions and technical details specific to the link prediction task, such as the choice of distance between node embeddings, can influence the quality of the explanations.

{{</citation>}}


### (8/97) Learning Implicit Entity-object Relations by Bidirectional Generative Alignment for Multimodal NER (Feng Chen et al., 2023)

{{<citation>}}

Feng Chen, Jiajia Liu, Kaixiang Ji, Wang Ren, Jian Wang, Jingdong Wang. (2023)  
**Learning Implicit Entity-object Relations by Bidirectional Generative Alignment for Multimodal NER**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2308.02570v1)  

---


**ABSTRACT**  
The challenge posed by multimodal named entity recognition (MNER) is mainly two-fold: (1) bridging the semantic gap between text and image and (2) matching the entity with its associated object in image. Existing methods fail to capture the implicit entity-object relations, due to the lack of corresponding annotation. In this paper, we propose a bidirectional generative alignment method named BGA-MNER to tackle these issues. Our BGA-MNER consists of \texttt{image2text} and \texttt{text2image} generation with respect to entity-salient content in two modalities. It jointly optimizes the bidirectional reconstruction objectives, leading to aligning the implicit entity-object relations under such direct and powerful constraints. Furthermore, image-text pairs usually contain unmatched components which are noisy for generation. A stage-refined context sampler is proposed to extract the matched cross-modal content for generation. Extensive experiments on two benchmarks demonstrate that our method achieves state-of-the-art performance without image input during inference.

{{</citation>}}


### (9/97) End-to-End Reinforcement Learning of Koopman Models for Economic Nonlinear MPC (Daniel Mayfrank et al., 2023)

{{<citation>}}

Daniel Mayfrank, Alexander Mitsos, Manuel Dahmen. (2023)  
**End-to-End Reinforcement Learning of Koopman Models for Economic Nonlinear MPC**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.01674v1)  

---


**ABSTRACT**  
(Economic) nonlinear model predictive control ((e)NMPC) requires dynamic system models that are sufficiently accurate in all relevant state-space regions. These models must also be computationally cheap enough to ensure real-time tractability. Data-driven surrogate models for mechanistic models can be used to reduce the computational burden of (e)NMPC; however, such models are typically trained by system identification for maximum average prediction accuracy on simulation samples and perform suboptimally as part of actual (e)NMPC. We present a method for end-to-end reinforcement learning of dynamic surrogate models for optimal performance in (e)NMPC applications, resulting in predictive controllers that strike a favorable balance between control performance and computational demand. We validate our method on two applications derived from an established nonlinear continuous stirred-tank reactor model. We compare the controller performance to that of MPCs utilizing models trained by the prevailing maximum prediction accuracy paradigm, and model-free neural network controllers trained using reinforcement learning. We show that our method matches the performance of the model-free neural network controllers while consistently outperforming models derived from system identification. Additionally, we show that the MPC policies can react to changes in the control setting without retraining.

{{</citation>}}


### (10/97) UniG-Encoder: A Universal Feature Encoder for Graph and Hypergraph Node Classification (Minhao Zou et al., 2023)

{{<citation>}}

Minhao Zou, Zhongxue Gan, Yutong Wang, Junheng Zhang, Dongyan Sui, Chun Guan, Siyang Leng. (2023)  
**UniG-Encoder: A Universal Feature Encoder for Graph and Hypergraph Node Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.01650v1)  

---


**ABSTRACT**  
Graph and hypergraph representation learning has attracted increasing attention from various research fields. Despite the decent performance and fruitful applications of Graph Neural Networks (GNNs), Hypergraph Neural Networks (HGNNs), and their well-designed variants, on some commonly used benchmark graphs and hypergraphs, they are outperformed by even a simple Multi-Layer Perceptron. This observation motivates a reexamination of the design paradigm of the current GNNs and HGNNs and poses challenges of extracting graph features effectively. In this work, a universal feature encoder for both graph and hypergraph representation learning is designed, called UniG-Encoder. The architecture starts with a forward transformation of the topological relationships of connected nodes into edge or hyperedge features via a normalized projection matrix. The resulting edge/hyperedge features, together with the original node features, are fed into a neural network. The encoded node embeddings are then derived from the reversed transformation, described by the transpose of the projection matrix, of the network's output, which can be further used for tasks such as node classification. The proposed architecture, in contrast to the traditional spectral-based and/or message passing approaches, simultaneously and comprehensively exploits the node features and graph/hypergraph topologies in an efficient and unified manner, covering both heterophilic and homophilic graphs. The designed projection matrix, encoding the graph features, is intuitive and interpretable. Extensive experiments are conducted and demonstrate the superior performance of the proposed framework on twelve representative hypergraph datasets and six real-world graph datasets, compared to the state-of-the-art methods. Our implementation is available online at https://github.com/MinhZou/UniG-Encoder.

{{</citation>}}


### (11/97) MARLIM: Multi-Agent Reinforcement Learning for Inventory Management (Rémi Leluc et al., 2023)

{{<citation>}}

Rémi Leluc, Elie Kadoche, Antoine Bertoncello, Sébastien Gourvénec. (2023)  
**MARLIM: Multi-Agent Reinforcement Learning for Inventory Management**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.01649v1)  

---


**ABSTRACT**  
Maintaining a balance between the supply and demand of products by optimizing replenishment decisions is one of the most important challenges in the supply chain industry. This paper presents a novel reinforcement learning framework called MARLIM, to address the inventory management problem for a single-echelon multi-products supply chain with stochastic demands and lead-times. Within this context, controllers are developed through single or multiple agents in a cooperative setting. Numerical experiments on real data demonstrate the benefits of reinforcement learning methods over traditional baselines.

{{</citation>}}


### (12/97) Discriminative Graph-level Anomaly Detection via Dual-students-teacher Model (Fu Lin et al., 2023)

{{<citation>}}

Fu Lin, Xuexiong Luo, Jia Wu, Jian Yang, Shan Xue, Zitong Wang, Haonan Gong. (2023)  
**Discriminative Graph-level Anomaly Detection via Dual-students-teacher Model**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.01947v1)  

---


**ABSTRACT**  
Different from the current node-level anomaly detection task, the goal of graph-level anomaly detection is to find abnormal graphs that significantly differ from others in a graph set. Due to the scarcity of research on the work of graph-level anomaly detection, the detailed description of graph-level anomaly is insufficient. Furthermore, existing works focus on capturing anomalous graph information to learn better graph representations, but they ignore the importance of an effective anomaly score function for evaluating abnormal graphs. Thus, in this work, we first define anomalous graph information including node and graph property anomalies in a graph set and adopt node-level and graph-level information differences to identify them, respectively. Then, we introduce a discriminative graph-level anomaly detection framework with dual-students-teacher model, where the teacher model with a heuristic loss are trained to make graph representations more divergent. Then, two competing student models trained by normal and abnormal graphs respectively fit graph representations of the teacher model in terms of node-level and graph-level representation perspectives. Finally, we combine representation errors between two student models to discriminatively distinguish anomalous graphs. Extensive experiment analysis demonstrates that our method is effective for the graph-level anomaly detection task on graph datasets in the real world.

{{</citation>}}


### (13/97) Unsupervised Representation Learning for Time Series: A Review (Qianwen Meng et al., 2023)

{{<citation>}}

Qianwen Meng, Hangwei Qian, Yong Liu, Yonghui Xu, Zhiqi Shen, Lizhen Cui. (2023)  
**Unsupervised Representation Learning for Time Series: A Review**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2308.01578v1)  

---


**ABSTRACT**  
Unsupervised representation learning approaches aim to learn discriminative feature representations from unlabeled data, without the requirement of annotating every sample. Enabling unsupervised representation learning is extremely crucial for time series data, due to its unique annotation bottleneck caused by its complex characteristics and lack of visual cues compared with other data modalities. In recent years, unsupervised representation learning techniques have advanced rapidly in various domains. However, there is a lack of systematic analysis of unsupervised representation learning approaches for time series. To fill the gap, we conduct a comprehensive literature review of existing rapidly evolving unsupervised representation learning approaches for time series. Moreover, we also develop a unified and standardized library, named ULTS (i.e., Unsupervised Learning for Time Series), to facilitate fast implementations and unified evaluations on various models. With ULTS, we empirically evaluate state-of-the-art approaches, especially the rapidly evolving contrastive learning methods, on 9 diverse real-world datasets. We further discuss practical considerations as well as open research challenges on unsupervised representation learning for time series to facilitate future research in this field.

{{</citation>}}


### (14/97) Lode Enhancer: Level Co-creation Through Scaling (Debosmita Bhaumik et al., 2023)

{{<citation>}}

Debosmita Bhaumik, Julian Togelius, Georgios N. Yannakakis, Ahmed Khalifa. (2023)  
**Lode Enhancer: Level Co-creation Through Scaling**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01543v1)  

---


**ABSTRACT**  
We explore AI-powered upscaling as a design assistance tool in the context of creating 2D game levels. Deep neural networks are used to upscale artificially downscaled patches of levels from the puzzle platformer game Lode Runner. The trained networks are incorporated into a web-based editor, where the user can create and edit levels at three different levels of resolution: 4x4, 8x8, and 16x16. An edit at any resolution instantly transfers to the other resolutions. As upscaling requires inventing features that might not be present at lower resolutions, we train neural networks to reproduce these features. We introduce a neural network architecture that is capable of not only learning upscaling but also giving higher priority to less frequent tiles. To investigate the potential of this tool and guide further development, we conduct a qualitative study with 3 designers to understand how they use it. Designers enjoyed co-designing with the tool, liked its underlying concept, and provided feedback for further improvement.

{{</citation>}}


### (15/97) Circumventing Concept Erasure Methods For Text-to-Image Generative Models (Minh Pham et al., 2023)

{{<citation>}}

Minh Pham, Kelly O. Marshall, Chinmay Hegde. (2023)  
**Circumventing Concept Erasure Methods For Text-to-Image Generative Models**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01508v1)  

---


**ABSTRACT**  
Text-to-image generative models can produce photo-realistic images for an extremely broad range of concepts, and their usage has proliferated widely among the general public. On the flip side, these models have numerous drawbacks, including their potential to generate images featuring sexually explicit content, mirror artistic styles without permission, or even hallucinate (or deepfake) the likenesses of celebrities. Consequently, various methods have been proposed in order to "erase" sensitive concepts from text-to-image models. In this work, we examine five recently proposed concept erasure methods, and show that targeted concepts are not fully excised from any of these methods. Specifically, we leverage the existence of special learned word embeddings that can retrieve "erased" concepts from the sanitized models with no alterations to their weights. Our results highlight the brittleness of post hoc concept erasure methods, and call into question their use in the algorithmic toolkit for AI safety.

{{</citation>}}


## cs.CL (26)



### (16/97) Causality Guided Disentanglement for Cross-Platform Hate Speech Detection (Paras Sheth et al., 2023)

{{<citation>}}

Paras Sheth, Tharindu Kumarage, Raha Moraffah, Aman Chadha, Huan Liu. (2023)  
**Causality Guided Disentanglement for Cross-Platform Hate Speech Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Hate Speech Detection  
[Paper Link](http://arxiv.org/abs/2308.02080v1)  

---


**ABSTRACT**  
Social media platforms, despite their value in promoting open discourse, are often exploited to spread harmful content. Current deep learning and natural language processing models used for detecting this harmful content overly rely on domain-specific terms affecting their capabilities to adapt to generalizable hate speech detection. This is because they tend to focus too narrowly on particular linguistic signals or the use of certain categories of words. Another significant challenge arises when platforms lack high-quality annotated data for training, leading to a need for cross-platform models that can adapt to different distribution shifts. Our research introduces a cross-platform hate speech detection model capable of being trained on one platform's data and generalizing to multiple unseen platforms. To achieve good generalizability across platforms, one way is to disentangle the input representations into invariant and platform-dependent features. We also argue that learning causal relationships, which remain constant across diverse environments, can significantly aid in understanding invariant representations in hate speech. By disentangling input into platform-dependent features (useful for predicting hate targets) and platform-independent features (used to predict the presence of hate), we learn invariant representations resistant to distribution shifts. These features are then used to predict hate speech across unseen platforms. Our extensive experiments across four platforms highlight our model's enhanced efficacy compared to existing state-of-the-art methods in detecting generalized hate speech.

{{</citation>}}


### (17/97) The Unequal Opportunities of Large Language Models: Revealing Demographic Bias through Job Recommendations (Abel Salinas et al., 2023)

{{<citation>}}

Abel Salinas, Parth Vipul Shah, Yuzhong Huang, Robert McCormack, Fred Morstatter. (2023)  
**The Unequal Opportunities of Large Language Models: Revealing Demographic Bias through Job Recommendations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: Bias, ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2308.02053v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have seen widespread deployment in various real-world applications. Understanding these biases is crucial to comprehend the potential downstream consequences when using LLMs to make decisions, particularly for historically disadvantaged groups. In this work, we propose a simple method for analyzing and comparing demographic bias in LLMs, through the lens of job recommendations. We demonstrate the effectiveness of our method by measuring intersectional biases within ChatGPT and LLaMA, two cutting-edge LLMs. Our experiments primarily focus on uncovering gender identity and nationality bias; however, our method can be extended to examine biases associated with any intersection of demographic identities. We identify distinct biases in both models toward various demographic identities, such as both models consistently suggesting low-paying jobs for Mexican workers or preferring to recommend secretarial roles to women. Our study highlights the importance of measuring the bias of LLMs in downstream applications to understand the potential for harm and inequitable outcomes.

{{</citation>}}


### (18/97) Efficient Sentiment Analysis: A Resource-Aware Evaluation of Feature Extraction Techniques, Ensembling, and Deep Learning Models (Mahammed Kamruzzaman et al., 2023)

{{<citation>}}

Mahammed Kamruzzaman, Gene Louis Kim. (2023)  
**Efficient Sentiment Analysis: A Resource-Aware Evaluation of Feature Extraction Techniques, Ensembling, and Deep Learning Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2308.02022v1)  

---


**ABSTRACT**  
While reaching for NLP systems that maximize accuracy, other important metrics of system performance are often overlooked. Prior models are easily forgotten despite their possible suitability in settings where large computing resources are unavailable or relatively more costly. In this paper, we perform a broad comparative evaluation of document-level sentiment analysis models with a focus on resource costs that are important for the feasibility of model deployment and general climate consciousness. Our experiments consider different feature extraction techniques, the effect of ensembling, task-specific deep learning modeling, and domain-independent large language models (LLMs). We find that while a fine-tuned LLM achieves the best accuracy, some alternate configurations provide huge (up to 24, 283 *) resource savings for a marginal (<1%) loss in accuracy. Furthermore, we find that for smaller datasets, the differences in accuracy shrink while the difference in resource consumption grows further.

{{</citation>}}


### (19/97) Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty (Inar Timiryasov et al., 2023)

{{<citation>}}

Inar Timiryasov, Jean-Loup Tastet. (2023)  
**Baby Llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: GPT, LLaMA  
[Paper Link](http://arxiv.org/abs/2308.02019v1)  

---


**ABSTRACT**  
We present our proposed solution to the BabyLM challenge [arXiv:2301.11796], whose goal was to improve the sample efficiency of language models. We trained an ensemble consisting of a GPT-2 and small LLaMA models on the developmentally-plausible, 10M-word BabyLM dataset, then distilled it into a small, 58M-parameter LLaMA model, which exceeds in performance both of its teachers as well as a similar model trained without distillation. This suggests that distillation can not only retain the full performance of the teacher model when the latter is trained on a sufficiently small dataset; it can exceed it, and lead to significantly better performance than direct training.

{{</citation>}}


### (20/97) Bengali Fake Reviews: A Benchmark Dataset and Detection System (G. M. Shahariar et al., 2023)

{{<citation>}}

G. M. Shahariar, Md. Tanvir Rouf Shawon, Faisal Muhammad Shah, Mohammad Shafiul Alam, Md. Shahriar Mahbub. (2023)  
**Bengali Fake Reviews: A Benchmark Dataset and Detection System**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.01987v1)  

---


**ABSTRACT**  
The proliferation of fake reviews on various online platforms has created a major concern for both consumers and businesses. Such reviews can deceive customers and cause damage to the reputation of products or services, making it crucial to identify them. Although the detection of fake reviews has been extensively studied in English language, detecting fake reviews in non-English languages such as Bengali is still a relatively unexplored research area. This paper introduces the Bengali Fake Review Detection (BFRD) dataset, the first publicly available dataset for identifying fake reviews in Bengali. The dataset consists of 7710 non-fake and 1339 fake food-related reviews collected from social media posts. To convert non-Bengali words in a review, a unique pipeline has been proposed that translates English words to their corresponding Bengali meaning and also back transliterates Romanized Bengali to Bengali. We have conducted rigorous experimentation using multiple deep learning and pre-trained transformer language models to develop a reliable detection system. Finally, we propose a weighted ensemble model that combines four pre-trained transformers: BanglaBERT, BanglaBERT Base, BanglaBERT Large, and BanglaBERT Generator . According to the experiment results, the proposed ensemble model obtained a weighted F1-score of 0.9843 on 13390 reviews, including 1339 actual fake reviews and 5356 augmented fake reviews generated with the nlpaug library. The remaining 6695 reviews were randomly selected from the 7710 non-fake instances. The model achieved a 0.9558 weighted F1-score when the fake reviews were augmented using the bnaug library.

{{</citation>}}


### (21/97) Reasoning in Large Language Models Through Symbolic Math Word Problems (Vedant Gaur et al., 2023)

{{<citation>}}

Vedant Gaur, Nikunj Saunshi. (2023)  
**Reasoning in Large Language Models Through Symbolic Math Word Problems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model, NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.01906v1)  

---


**ABSTRACT**  
Large language models (LLMs) have revolutionized NLP by solving downstream tasks with little to no labeled data. Despite their versatile abilities, the larger question of their ability to reason remains ill-understood. This paper addresses reasoning in math word problems (MWPs) by studying symbolic versions of the numeric problems, since a symbolic expression is a "concise explanation" of the numeric answer. We create and use a symbolic version of the SVAMP dataset and find that GPT-3's davinci-002 model also has good zero-shot accuracy on symbolic MWPs. To evaluate the faithfulness of the model's reasoning, we go beyond accuracy and additionally evaluate the alignment between the final answer and the outputted reasoning, which correspond to numeric and symbolic answers respectively for MWPs. We explore a self-prompting approach to encourage the symbolic reasoning to align with the numeric answer, thus equipping the LLM with the ability to provide a concise and verifiable reasoning and making it more interpretable. Surprisingly, self-prompting also improves the symbolic accuracy to be higher than both the numeric and symbolic accuracies, thus providing an ensembling effect. The SVAMP_Sym dataset will be released for future research on symbolic math problems.

{{</citation>}}


### (22/97) Athena 2.0: Discourse and User Modeling in Open Domain Dialogue (Omkar Patil et al., 2023)

{{<citation>}}

Omkar Patil, Lena Reed, Kevin K. Bowden, Juraj Juraska, Wen Cui, Vrindavan Harrison, Rishi Rajasekaran, Angela Ramirez, Cecilia Li, Eduardo Zamora, Phillip Lee, Jeshwanth Bheemanpally, Rohan Pandey, Adwait Ratnaparkhi, Marilyn Walker. (2023)  
**Athena 2.0: Discourse and User Modeling in Open Domain Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Amazon, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2308.01887v1)  

---


**ABSTRACT**  
Conversational agents are consistently growing in popularity and many people interact with them every day. While many conversational agents act as personal assistants, they can have many different goals. Some are task-oriented, such as providing customer support for a bank or making a reservation. Others are designed to be empathetic and to form emotional connections with the user. The Alexa Prize Challenge aims to create a socialbot, which allows the user to engage in coherent conversations, on a range of popular topics that will interest the user. Here we describe Athena 2.0, UCSC's conversational agent for Amazon's Socialbot Grand Challenge 4. Athena 2.0 utilizes a novel knowledge-grounded discourse model that tracks the entity links that Athena introduces into the dialogue, and uses them to constrain named-entity recognition and linking, and coreference resolution. Athena 2.0 also relies on a user model to personalize topic selection and other aspects of the conversation to individual users.

{{</citation>}}


### (23/97) Tag Prediction of Competitive Programming Problems using Deep Learning Techniques (Taha Lokat et al., 2023)

{{<citation>}}

Taha Lokat, Divyam Prajapati, Shubhada Labde. (2023)  
**Tag Prediction of Competitive Programming Problems using Deep Learning Techniques**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LSTM, NLP, Natural Language Processing, Text Classification  
[Paper Link](http://arxiv.org/abs/2308.01863v1)  

---


**ABSTRACT**  
In the past decade, the amount of research being done in the fields of machine learning and deep learning, predominantly in the area of natural language processing (NLP), has risen dramatically. A well-liked method for developing programming abilities like logic building and problem solving is competitive programming. It can be tough for novices and even veteran programmers to traverse the wide collection of questions due to the massive number of accessible questions and the variety of themes, levels of difficulty, and questions offered. In order to help programmers find questions that are appropriate for their knowledge and interests, there is a need for an automated method. This can be done using automated tagging of the questions using Text Classification. Text classification is one of the important tasks widely researched in the field of Natural Language Processing. In this paper, we present a way to use text classification techniques to determine the domain of a competitive programming problem. A variety of models, including are implemented LSTM, GRU, and MLP. The dataset has been scraped from Codeforces, a major competitive programming website. A total of 2400 problems were scraped and preprocessed, which we used as a dataset for our training and testing of models. The maximum accuracy reached using our model is 78.0% by MLP(Multi Layer Perceptron).

{{</citation>}}


### (24/97) ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation (Xueying Du et al., 2023)

{{<citation>}}

Xueying Du, Mingwei Liu, Kaixin Wang, Hanlin Wang, Junwei Liu, Yixuan Chen, Jiayi Feng, Chaofeng Sha, Xin Peng, Yiling Lou. (2023)  
**ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.01861v1)  

---


**ABSTRACT**  
In this work, we make the first attempt to evaluate LLMs in a more challenging code generation scenario, i.e. class-level code generation. We first manually construct the first class-level code generation benchmark ClassEval of 100 class-level Python code generation tasks with approximately 500 person-hours. Based on it, we then perform the first study of 11 state-of-the-art LLMs on class-level code generation. Based on our results, we have the following main findings. First, we find that all existing LLMs show much worse performance on class-level code generation compared to on standalone method-level code generation benchmarks like HumanEval; and the method-level coding ability cannot equivalently reflect the class-level coding ability among LLMs. Second, we find that GPT-4 and GPT-3.5 still exhibit dominate superior than other LLMs on class-level code generation, and the second-tier models includes Instruct-Starcoder, Instruct-Codegen, and Wizardcoder with very similar performance. Third, we find that generating the entire class all at once (i.e. holistic generation strategy) is the best generation strategy only for GPT-4 and GPT-3.5, while method-by-method generation (i.e. incremental and compositional) is better strategies for the other models with limited ability of understanding long instructions and utilizing the middle information. Lastly, we find the limited model ability of generating method-dependent code and discuss the frequent error types in generated classes. Our benchmark is available at https://github.com/FudanSELab/ClassEval.

{{</citation>}}


### (25/97) Curricular Transfer Learning for Sentence Encoded Tasks (Jader Martins Camboim de Sá et al., 2023)

{{<citation>}}

Jader Martins Camboim de Sá, Matheus Ferraroni Sanches, Rafael Roque de Souza, Júlio Cesar dos Reis, Leandro Aparecido Villas. (2023)  
**Curricular Transfer Learning for Sentence Encoded Tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.01849v1)  

---


**ABSTRACT**  
Fine-tuning language models in a downstream task is the standard approach for many state-of-the-art methodologies in the field of NLP. However, when the distribution between the source task and target task drifts, \textit{e.g.}, conversational environments, these gains tend to be diminished. This article proposes a sequence of pre-training steps (a curriculum) guided by "data hacking" and grammar analysis that allows further gradual adaptation between pre-training distributions. In our experiments, we acquire a considerable improvement from our method compared to other known pre-training approaches for the MultiWoZ task.

{{</citation>}}


### (26/97) XNLP: An Interactive Demonstration System for Universal Structured NLP (Hao Fei et al., 2023)

{{<citation>}}

Hao Fei, Meishan Zhang, Min Zhang, Tat-Seng Chua. (2023)  
**XNLP: An Interactive Demonstration System for Universal Structured NLP**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.01846v1)  

---


**ABSTRACT**  
Structured Natural Language Processing (XNLP) is an important subset of NLP that entails understanding the underlying semantic or syntactic structure of texts, which serves as a foundational component for many downstream applications. Despite certain recent efforts to explore universal solutions for specific categories of XNLP tasks, a comprehensive and effective approach for unifying all XNLP tasks long remains underdeveloped. In the meanwhile, while XNLP demonstration systems are vital for researchers exploring various XNLP tasks, existing platforms can be limited to, e.g., supporting few XNLP tasks, lacking interactivity and universalness. To this end, we propose an advanced XNLP demonstration platform, where we propose leveraging LLM to achieve universal XNLP, with one model for all with high generalizability. Overall, our system advances in multiple aspects, including universal XNLP modeling, high performance, interpretability, scalability, and interactivity, providing a unified platform for exploring diverse XNLP tasks in the community. XNLP is online: https://xnlp.haofei.vip

{{</citation>}}


### (27/97) The Capability of Large Language Models to Measure Psychiatric Functioning (Isaac R. Galatzer-Levy et al., 2023)

{{<citation>}}

Isaac R. Galatzer-Levy, Daniel McDuff, Vivek Natarajan, Alan Karthikesalingam, Matteo Malgaroli. (2023)  
**The Capability of Large Language Models to Measure Psychiatric Functioning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2308.01834v1)  

---


**ABSTRACT**  
The current work investigates the capability of Large language models (LLMs) that are explicitly trained on large corpuses of medical knowledge (Med-PaLM 2) to predict psychiatric functioning from patient interviews and clinical descriptions without being trained to do so. To assess this, n = 145 depression and n =115 PTSD assessments and n = 46 clinical case studies across high prevalence/high comorbidity disorders (Depressive, Anxiety, Psychotic, trauma and stress, Addictive disorders) were analyzed using prompts to extract estimated clinical scores and diagnoses. Results demonstrate that Med-PaLM 2 is capable of assessing psychiatric functioning across a range of psychiatric conditions with the strongest performance being the prediction of depression scores based on standardized assessments (Accuracy range= 0.80 - 0.84) which were statistically indistinguishable from human clinical raters t(1,144) = 1.20; p = 0.23. Results show the potential for general clinical language models to flexibly predict psychiatric risk based on free descriptions of functioning from both patients and clinicians.

{{</citation>}}


### (28/97) Many-to-Many Spoken Language Translation via Unified Speech and Text Representation Learning with Unit-to-Unit Translation (Minsu Kim et al., 2023)

{{<citation>}}

Minsu Kim, Jeongsoo Choi, Dahun Kim, Yong Man Ro. (2023)  
**Many-to-Many Spoken Language Translation via Unified Speech and Text Representation Learning with Unit-to-Unit Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS, eess-SP  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.01831v1)  

---


**ABSTRACT**  
In this paper, we propose a method to learn unified representations of multilingual speech and text with a single model, especially focusing on the purpose of speech synthesis. We represent multilingual speech audio with speech units, the quantized representations of speech features encoded from a self-supervised speech model. Therefore, we can focus on their linguistic content by treating the audio as pseudo text and can build a unified representation of speech and text. Then, we propose to train an encoder-decoder structured model with a Unit-to-Unit Translation (UTUT) objective on multilingual data. Specifically, by conditioning the encoder with the source language token and the decoder with the target language token, the model is optimized to translate the spoken language into that of the target language, in a many-to-many language translation setting. Therefore, the model can build the knowledge of how spoken languages are comprehended and how to relate them to different languages. A single pre-trained model with UTUT can be employed for diverse multilingual speech- and text-related tasks, such as Speech-to-Speech Translation (STS), multilingual Text-to-Speech Synthesis (TTS), and Text-to-Speech Translation (TTST). By conducting comprehensive experiments encompassing various languages, we validate the efficacy of the proposed method across diverse multilingual tasks. Moreover, we show UTUT can perform many-to-many language STS, which has not been previously explored in the literature. Samples are available on https://choijeongsoo.github.io/utut.

{{</citation>}}


### (29/97) Scaling Relationship on Learning Mathematical Reasoning with Large Language Models (Zheng Yuan et al., 2023)

{{<citation>}}

Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, Chang Zhou. (2023)  
**Scaling Relationship on Learning Mathematical Reasoning with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.01825v1)  

---


**ABSTRACT**  
Mathematical reasoning is a challenging task for large language models (LLMs), while the scaling relationship of it with respect to LLM capacity is under-explored. In this paper, we investigate how the pre-training loss, supervised data amount, and augmented data amount influence the reasoning performances of a supervised LLM. We find that pre-training loss is a better indicator of the model's performance than the model's parameter count. We apply supervised fine-tuning (SFT) with different amounts of supervised data and empirically find a log-linear relation between data amount and model performance, and we find better models improve less with enlarged supervised datasets. To augment more data samples for improving model performances without any human effort, we propose to apply Rejection sampling Fine-Tuning (RFT). RFT uses supervised models to generate and collect correct reasoning paths as augmented fine-tuning datasets. We find with augmented samples containing more distinct reasoning paths, RFT improves mathematical reasoning performance more for LLMs. We also find RFT brings more improvement for less performant LLMs. Furthermore, we combine rejection samples from multiple models which push LLaMA-7B to an accuracy of 49.3% and outperforms the supervised fine-tuning (SFT) accuracy of 35.9% significantly.

{{</citation>}}


### (30/97) Lexicon and Rule-based Word Lemmatization Approach for the Somali Language (Shafie Abdi Mohamed et al., 2023)

{{<citation>}}

Shafie Abdi Mohamed, Muhidin Abdullahi Mohamed. (2023)  
**Lexicon and Rule-based Word Lemmatization Approach for the Somali Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.01785v1)  

---


**ABSTRACT**  
Lemmatization is a Natural Language Processing (NLP) technique used to normalize text by changing morphological derivations of words to their root forms. It is used as a core pre-processing step in many NLP tasks including text indexing, information retrieval, and machine learning for NLP, among others. This paper pioneers the development of text lemmatization for the Somali language, a low-resource language with very limited or no prior effective adoption of NLP methods and datasets. We especially develop a lexicon and rule-based lemmatizer for Somali text, which is a starting point for a full-fledged Somali lemmatization system for various NLP tasks. With consideration of the language morphological rules, we have developed an initial lexicon of 1247 root words and 7173 derivationally related terms enriched with rules for lemmatizing words not present in the lexicon. We have tested the algorithm on 120 documents of various lengths including news articles, social media posts, and text messages. Our initial results demonstrate that the algorithm achieves an accuracy of 57\% for relatively long documents (e.g. full news articles), 60.57\% for news article extracts, and high accuracy of 95.87\% for short texts such as social media messages.

{{</citation>}}


### (31/97) Does Correction Remain An Problem For Large Language Models? (Xiaowu Zhang et al., 2023)

{{<citation>}}

Xiaowu Zhang, Xiaotian Zhang, Cheng Yang, Hang Yan, Xipeng Qiu. (2023)  
**Does Correction Remain An Problem For Large Language Models?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.01776v1)  

---


**ABSTRACT**  
As large language models, such as GPT, continue to advance the capabilities of natural language processing (NLP), the question arises: does the problem of correction still persist? This paper investigates the role of correction in the context of large language models by conducting two experiments. The first experiment focuses on correction as a standalone task, employing few-shot learning techniques with GPT-like models for error correction. The second experiment explores the notion of correction as a preparatory task for other NLP tasks, examining whether large language models can tolerate and perform adequately on texts containing certain levels of noise or errors. By addressing these experiments, we aim to shed light on the significance of correction in the era of large language models and its implications for various NLP applications.

{{</citation>}}


### (32/97) Supply chain emission estimation using large language models (Ayush Jain et al., 2023)

{{<citation>}}

Ayush Jain, Manikandan Padmanaban, Jagabondhu Hazra, Shantanu Godbole, Kommy Weldemariam. (2023)  
**Supply chain emission estimation using large language models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.01741v1)  

---


**ABSTRACT**  
Large enterprises face a crucial imperative to achieve the Sustainable Development Goals (SDGs), especially goal 13, which focuses on combating climate change and its impacts. To mitigate the effects of climate change, reducing enterprise Scope 3 (supply chain emissions) is vital, as it accounts for more than 90\% of total emission inventories. However, tracking Scope 3 emissions proves challenging, as data must be collected from thousands of upstream and downstream suppliers.To address the above mentioned challenges, we propose a first-of-a-kind framework that uses domain-adapted NLP foundation models to estimate Scope 3 emissions, by utilizing financial transactions as a proxy for purchased goods and services. We compared the performance of the proposed framework with the state-of-art text classification models such as TF-IDF, word2Vec, and Zero shot learning. Our results show that the domain-adapted foundation model outperforms state-of-the-art text mining techniques and performs as well as a subject matter expert (SME). The proposed framework could accelerate the Scope 3 estimation at Enterprise scale and will help to take appropriate climate actions to achieve SDG 13.

{{</citation>}}


### (33/97) Ambient Adventures: Teaching ChatGPT on Developing Complex Stories (Zexin Chen et al., 2023)

{{<citation>}}

Zexin Chen, Eric Zhou, Kenneth Eaton, Xiangyu Peng, Mark Riedl. (2023)  
**Ambient Adventures: Teaching ChatGPT on Developing Complex Stories**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.01734v1)  

---


**ABSTRACT**  
Imaginative play is an area of creativity that could allow robots to engage with the world around them in a much more personified way. Imaginary play can be seen as taking real objects and locations and using them as imaginary objects and locations in virtual scenarios. We adopted the story generation capability of large language models (LLMs) to obtain the stories used for imaginary play with human-written prompts. Those generated stories will be simplified and mapped into action sequences that can guide the agent in imaginary play. To evaluate whether the agent can successfully finish the imaginary play, we also designed a text adventure game to simulate a house as the playground for the agent to interact.

{{</citation>}}


### (34/97) Is GPT-4 a reliable rater? Evaluating Consistency in GPT-4 Text Ratings (Veronika Hackl et al., 2023)

{{<citation>}}

Veronika Hackl, Alexandra Elena Müller, Michael Granitzer, Maximilian Sailer. (2023)  
**Is GPT-4 a reliable rater? Evaluating Consistency in GPT-4 Text Ratings**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.02575v1)  

---


**ABSTRACT**  
This study investigates the consistency of feedback ratings generated by OpenAI's GPT-4, a state-of-the-art artificial intelligence language model, across multiple iterations, time spans and stylistic variations. The model rated responses to tasks within the Higher Education (HE) subject domain of macroeconomics in terms of their content and style. Statistical analysis was conducted in order to learn more about the interrater reliability, consistency of the ratings across iterations and the correlation between ratings in terms of content and style. The results revealed a high interrater reliability with ICC scores ranging between 0.94 and 0.99 for different timespans, suggesting that GPT-4 is capable of generating consistent ratings across repetitions with a clear prompt. Style and content ratings show a high correlation of 0.87. When applying a non-adequate style the average content ratings remained constant, while style ratings decreased, which indicates that the large language model (LLM) effectively distinguishes between these two criteria during evaluation. The prompt used in this study is furthermore presented and explained. Further research is necessary to assess the robustness and reliability of AI models in various use cases.

{{</citation>}}


### (35/97) Local Large Language Models for Complex Structured Medical Tasks (V. K. Cody Bumgardner et al., 2023)

{{<citation>}}

V. K. Cody Bumgardner, Aaron Mullen, Sam Armstrong, Caylin Hickey, Jeff Talbert. (2023)  
**Local Large Language Models for Complex Structured Medical Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2308.01727v1)  

---


**ABSTRACT**  
This paper introduces an approach that combines the language reasoning capabilities of large language models (LLMs) with the benefits of local training to tackle complex, domain-specific tasks. Specifically, the authors demonstrate their approach by extracting structured condition codes from pathology reports. The proposed approach utilizes local LLMs, which can be fine-tuned to respond to specific generative instructions and provide structured outputs. The authors collected a dataset of over 150k uncurated surgical pathology reports, containing gross descriptions, final diagnoses, and condition codes. They trained different model architectures, including LLaMA, BERT and LongFormer and evaluated their performance. The results show that the LLaMA-based models significantly outperform BERT-style models across all evaluated metrics, even with extremely reduced precision. The LLaMA models performed especially well with large datasets, demonstrating their ability to handle complex, multi-label tasks. Overall, this work presents an effective approach for utilizing LLMs to perform domain-specific tasks using accessible hardware, with potential applications in the medical domain, where complex data extraction and classification are required.

{{</citation>}}


### (36/97) Baby's CoThought: Leveraging Large Language Models for Enhanced Reasoning in Compact Models (Zheyu Zhang et al., 2023)

{{<citation>}}

Zheyu Zhang, Han Yang, Bolei Ma, David Rügamer, Ercong Nie. (2023)  
**Baby's CoThought: Leveraging Large Language Models for Enhanced Reasoning in Compact Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-3.5, Language Model, NLU, Natural Language Understanding, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.01684v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) demonstrate remarkable performance on a variety of Natural Language Understanding (NLU) tasks, primarily due to their in-context learning ability. This ability is utilized in our proposed "CoThought" pipeline, which efficiently trains smaller "baby" language models (BabyLMs) by leveraging the Chain of Thought (CoT) prompting of LLMs. Our pipeline restructures a dataset of less than 100M in size using GPT-3.5-turbo, transforming it into task-oriented, human-readable texts that are comparable to the school texts for language learners. The BabyLM is then pretrained on this restructured dataset in a RoBERTa (Liu et al., 2019) fashion. In evaluations across 4 benchmarks, our BabyLM outperforms the RoBERTa-base in 10 linguistic, NLU, and question answering tasks by more than 3 points, showing superior ability to extract contextual information. These results suggest that compact LMs pretrained on small, LLM-restructured data can better understand tasks and achieve improved performance. The code for data processing and model training is available at: https://github.com/oooranz/Baby-CoThought.

{{</citation>}}


### (37/97) NBIAS: A Natural Language Processing Framework for Bias Identification in Text (Shaina Razaa et al., 2023)

{{<citation>}}

Shaina Razaa, Muskan Garg, Deepak John Reji, Syed Raza Bashir, Chen Ding. (2023)  
**NBIAS: A Natural Language Processing Framework for Bias Identification in Text**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.01681v1)  

---


**ABSTRACT**  
Bias in textual data can lead to skewed interpretations and outcomes when the data is used. These biases could perpetuate stereotypes, discrimination, or other forms of unfair treatment. An algorithm trained on biased data ends up making decisions that disproportionately impact a certain group of people. Therefore, it is crucial to detect and remove these biases to ensure the fair and ethical use of data. To this end, we develop a comprehensive and robust framework \textsc{Nbias} that consists of a data layer, corpus contruction, model development layer and an evaluation layer. The dataset is constructed by collecting diverse data from various fields, including social media, healthcare, and job hiring portals. As such, we applied a transformer-based token classification model that is able to identify bias words/ phrases through a unique named entity. In the assessment procedure, we incorporate a blend of quantitative and qualitative evaluations to gauge the effectiveness of our models. We achieve accuracy improvements ranging from 1% to 8% compared to baselines. We are also able to generate a robust understanding of the model functioning, capturing not only numerical data but also the quality and intricacies of its performance. The proposed approach is applicable to a variety of biases and contributes to the fair and ethical use of textual data.

{{</citation>}}


### (38/97) BioBERT Based SNP-traits Associations Extraction from Biomedical Literature (Mohammad Dehghani et al., 2023)

{{<citation>}}

Mohammad Dehghani, Behrouz Bokharaeian, Zahra Yazdanparast. (2023)  
**BioBERT Based SNP-traits Associations Extraction from Biomedical Literature**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.02569v1)  

---


**ABSTRACT**  
Scientific literature contains a considerable amount of information that provides an excellent opportunity for developing text mining methods to extract biomedical relationships. An important type of information is the relationship between singular nucleotide polymorphisms (SNP) and traits. In this paper, we present a BioBERT-GRU method to identify SNP- traits associations. Based on the evaluation of our method on the SNPPhenA dataset, it is concluded that this new method performs better than previous machine learning and deep learning based methods. BioBERT-GRU achieved the result a precision of 0.883, recall of 0.882 and F1-score of 0.881.

{{</citation>}}


### (39/97) SimTeG: A Frustratingly Simple Approach Improves Textual Graph Learning (Keyu Duan et al., 2023)

{{<citation>}}

Keyu Duan, Qian Liu, Tat-Seng Chua, Shuicheng Yan, Wei Tsang Ooi, Qizhe Xie, Junxian He. (2023)  
**SimTeG: A Frustratingly Simple Approach Improves Textual Graph Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.02565v1)  

---


**ABSTRACT**  
Textual graphs (TGs) are graphs whose nodes correspond to text (sentences or documents), which are widely prevalent. The representation learning of TGs involves two stages: (i) unsupervised feature extraction and (ii) supervised graph representation learning. In recent years, extensive efforts have been devoted to the latter stage, where Graph Neural Networks (GNNs) have dominated. However, the former stage for most existing graph benchmarks still relies on traditional feature engineering techniques. More recently, with the rapid development of language models (LMs), researchers have focused on leveraging LMs to facilitate the learning of TGs, either by jointly training them in a computationally intensive framework (merging the two stages), or designing complex self-supervised training tasks for feature extraction (enhancing the first stage). In this work, we present SimTeG, a frustratingly Simple approach for Textual Graph learning that does not innovate in frameworks, models, and tasks. Instead, we first perform supervised parameter-efficient fine-tuning (PEFT) on a pre-trained LM on the downstream task, such as node classification. We then generate node embeddings using the last hidden states of finetuned LM. These derived features can be further utilized by any GNN for training on the same task. We evaluate our approach on two fundamental graph representation learning tasks: node classification and link prediction. Through extensive experiments, we show that our approach significantly improves the performance of various GNNs on multiple graph benchmarks.

{{</citation>}}


### (40/97) Large Language Model Displays Emergent Ability to Interpret Novel Literary Metaphors (Nicholas Ichien et al., 2023)

{{<citation>}}

Nicholas Ichien, Dušan Stamenković, Keith J. Holyoak. (2023)  
**Large Language Model Displays Emergent Ability to Interpret Novel Literary Metaphors**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.01497v1)  

---


**ABSTRACT**  
Recent advances in the performance of large language models (LLMs) have sparked debate over whether, given sufficient training, high-level human abilities emerge in such generic forms of artificial intelligence (AI). Despite the exceptional performance of LLMs on a wide range of tasks involving natural language processing and reasoning, there has been sharp disagreement as to whether their abilities extend to more creative human abilities. A core example is the ability to interpret novel metaphors. Given the enormous and non-curated text corpora used to train LLMs, a serious obstacle to designing tests is the requirement of finding novel yet high-quality metaphors that are unlikely to have been included in the training data. Here we assessed the ability of GPT-4, a state-of-the-art large language model, to provide natural-language interpretations of novel literary metaphors drawn from Serbian poetry and translated into English. Despite exhibiting no signs of having been exposed to these metaphors previously, the AI system consistently produced detailed and incisive interpretations. Human judge - blind to the fact that an AI model was involved - rated metaphor interpretations generated by GPT-4 as superior to those provided by a group of college students. In interpreting reversed metaphors, GPT-4, as well as humans, exhibited signs of sensitivity to the Gricean cooperative principle. These results indicate that LLMs such as GPT-4 have acquired an emergent ability to interpret complex novel metaphors.

{{</citation>}}


### (41/97) Investigating Reinforcement Learning for Communication Strategies in a Task-Initiative Setting (Baber Khalid et al., 2023)

{{<citation>}}

Baber Khalid, Matthew Stone. (2023)  
**Investigating Reinforcement Learning for Communication Strategies in a Task-Initiative Setting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.01479v1)  

---


**ABSTRACT**  
Many conversational domains require the system to present nuanced information to users. Such systems must follow up what they say to address clarification questions and repair misunderstandings. In this work, we explore this interactive strategy in a referential communication task. Using simulation, we analyze the communication trade-offs between initial presentation and subsequent followup as a function of user clarification strategy, and compare the performance of several baseline strategies to policies derived by reinforcement learning. We find surprising advantages to coherence-based representations of dialogue strategy, which bring minimal data requirements, explainable choices, and strong audit capabilities, but incur little loss in predicted outcomes across a wide range of user models.

{{</citation>}}


## cs.CV (23)



### (42/97) Mitigating Task Interference in Multi-Task Learning via Explicit Task Routing with Non-Learnable Primitives (Chuntao Ding et al., 2023)

{{<citation>}}

Chuntao Ding, Zhichao Lu, Shangguang Wang, Ran Cheng, Vishnu Naresh Boddeti. (2023)  
**Mitigating Task Interference in Multi-Task Learning via Explicit Task Routing with Non-Learnable Primitives**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.02066v1)  

---


**ABSTRACT**  
Multi-task learning (MTL) seeks to learn a single model to accomplish multiple tasks by leveraging shared information among the tasks. Existing MTL models, however, have been known to suffer from negative interference among tasks. Efforts to mitigate task interference have focused on either loss/gradient balancing or implicit parameter partitioning with partial overlaps among the tasks. In this paper, we propose ETR-NLP to mitigate task interference through a synergistic combination of non-learnable primitives (NLPs) and explicit task routing (ETR). Our key idea is to employ non-learnable primitives to extract a diverse set of task-agnostic features and recombine them into a shared branch common to all tasks and explicit task-specific branches reserved for each task. The non-learnable primitives and the explicit decoupling of learnable parameters into shared and task-specific ones afford the flexibility needed for minimizing task interference. We evaluate the efficacy of ETR-NLP networks for both image-level classification and pixel-level dense prediction MTL problems. Experimental results indicate that ETR-NLP significantly outperforms state-of-the-art baselines with fewer learnable parameters and similar FLOPs across all datasets. Code is available at this \href{https://github.com/zhichao-lu/etr-nlp-mtl}.

{{</citation>}}


### (43/97) RealCQA: Scientific Chart Question Answering as a Test-bed for First-Order Logic (Saleem Ahmed et al., 2023)

{{<citation>}}

Saleem Ahmed, Bhavin Jawade, Shubham Pandey, Srirangaraj Setlur, Venu Govindaraju. (2023)  
**RealCQA: Scientific Chart Question Answering as a Test-bed for First-Order Logic**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LO, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.01979v1)  

---


**ABSTRACT**  
We present a comprehensive study of chart visual question-answering(QA) task, to address the challenges faced in comprehending and extracting data from chart visualizations within documents. Despite efforts to tackle this problem using synthetic charts, solutions are limited by the shortage of annotated real-world data. To fill this gap, we introduce a benchmark and dataset for chart visual QA on real-world charts, offering a systematic analysis of the task and a novel taxonomy for template-based chart question creation. Our contribution includes the introduction of a new answer type, 'list', with both ranked and unranked variations. Our study is conducted on a real-world chart dataset from scientific literature, showcasing higher visual complexity compared to other works. Our focus is on template-based QA and how it can serve as a standard for evaluating the first-order logic capabilities of models. The results of our experiments, conducted on a real-world out-of-distribution dataset, provide a robust evaluation of large-scale pre-trained models and advance the field of chart visual QA and formal logic verification for neural networks in general.

{{</citation>}}


### (44/97) FROD: Robust Object Detection for Free (Muhammad et al., 2023)

{{<citation>}}

Muhammad, Awais, Weiming, Zhuang, Lingjuan, Lyu, Sung-Ho, Bae. (2023)  
**FROD: Robust Object Detection for Free**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.01888v1)  

---


**ABSTRACT**  
Object detection is a vital task in computer vision and has become an integral component of numerous critical systems. However, state-of-the-art object detectors, similar to their classification counterparts, are susceptible to small adversarial perturbations that can significantly alter their normal behavior. Unlike classification, the robustness of object detectors has not been thoroughly explored. In this work, we take the initial step towards bridging the gap between the robustness of classification and object detection by leveraging adversarially trained classification models. Merely utilizing adversarially trained models as backbones for object detection does not result in robustness. We propose effective modifications to the classification-based backbone to instill robustness in object detection without incurring any computational overhead. To further enhance the robustness achieved by the proposed modified backbone, we introduce two lightweight components: imitation loss and delayed adversarial training. Extensive experiments on the MS-COCO and Pascal VOC datasets are conducted to demonstrate the effectiveness of our proposed approach.

{{</citation>}}


### (45/97) Deep Neural Networks Fused with Textures for Image Classification (Asish Bera et al., 2023)

{{<citation>}}

Asish Bera, Debotosh Bhattacharjee, Mita Nasipuri. (2023)  
**Deep Neural Networks Fused with Textures for Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification, LSTM  
[Paper Link](http://arxiv.org/abs/2308.01813v1)  

---


**ABSTRACT**  
Fine-grained image classification (FGIC) is a challenging task in computer vision for due to small visual differences among inter-subcategories, but, large intra-class variations. Deep learning methods have achieved remarkable success in solving FGIC. In this paper, we propose a fusion approach to address FGIC by combining global texture with local patch-based information. The first pipeline extracts deep features from various fixed-size non-overlapping patches and encodes features by sequential modelling using the long short-term memory (LSTM). Another path computes image-level textures at multiple scales using the local binary patterns (LBP). The advantages of both streams are integrated to represent an efficient feature vector for image classification. The method is tested on eight datasets representing the human faces, skin lesions, food dishes, marine lives, etc. using four standard backbone CNNs. Our method has attained better classification accuracy over existing methods with notable margins.

{{</citation>}}


### (46/97) QUEST: Query Stream for Vehicle-Infrastructure Cooperative Perception (Siqi Fan et al., 2023)

{{<citation>}}

Siqi Fan, Haibao Yu, Wenxian Yang, Jirui Yuan, Zaiqing Nie. (2023)  
**QUEST: Query Stream for Vehicle-Infrastructure Cooperative Perception**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01804v1)  

---


**ABSTRACT**  
Cooperative perception can effectively enhance individual perception performance by providing additional viewpoint and expanding the sensing field. Existing cooperation paradigms are either interpretable (result cooperation) or flexible (feature cooperation). In this paper, we propose the concept of query cooperation to enable interpretable instance-level flexible feature interaction. To specifically explain the concept, we propose a cooperative perception framework, termed QUEST, which let query stream flow among agents. The cross-agent queries are interacted via fusion for co-aware instances and complementation for individual unaware instances. Taking camera-based vehicle-infrastructure perception as a typical practical application scene, the experimental results on the real-world dataset, DAIR-V2X-Seq, demonstrate the effectiveness of QUEST and further reveal the advantage of the query cooperation paradigm on transmission flexibility and robustness to packet dropout. We hope our work can further facilitate the cross-agent representation interaction for better cooperative perception in practice.

{{</citation>}}


### (47/97) RegionBLIP: A Unified Multi-modal Pre-training Framework for Holistic and Regional Comprehension (Qiang Zhou et al., 2023)

{{<citation>}}

Qiang Zhou, Chaohui Yu, Shaofeng Zhang, Sitong Wu, Zhibing Wang, Fan Wang. (2023)  
**RegionBLIP: A Unified Multi-modal Pre-training Framework for Holistic and Regional Comprehension**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.02299v1)  

---


**ABSTRACT**  
In this work, we investigate extending the comprehension of Multi-modal Large Language Models (MLLMs) to regional objects. To this end, we propose to extract features corresponding to regional objects as soft prompts for LLM, which provides a straightforward and scalable approach and eliminates the need for LLM fine-tuning. To effectively extract regional features from regular image features and irregular point cloud features, we present a novel and unified position-assisted feature extraction module. Furthermore, training an MLLM from scratch is highly time-consuming. Thus, we propose incrementally extending existing pre-trained MLLMs to comprehend more modalities and the regional objects of those modalities. Specifically, we freeze the Q-Former from BLIP-2, an impressive MLLM, and optimize the modality-specific Lora parameters in Q-Former and LLM for each newly introduced modality. The freezing of the Q-Former eliminates the need for extensive pre-training on massive image-text data. The freezed Q-Former pre-trained from massive image-text data is also beneficial for the pre-training on image-region-text data. We name our framework RegionBLIP. We pre-train RegionBLIP on image-region-text, point-cloud-text, and point-cloud-region-text data. Experimental results verify that \Ours{} can preserve the image comprehension capability of BILP-2 and further gain a comprehension of the newly introduced point cloud modality and regional objects. The Data, Code, and Pre-trained models will be available at https://github.com/mightyzau/RegionBLIP.

{{</citation>}}


### (48/97) Guided Distillation for Semi-Supervised Instance Segmentation (Tariq Berrada et al., 2023)

{{<citation>}}

Tariq Berrada, Camille Couprie, Karteek Alahari, Jakob Verbeek. (2023)  
**Guided Distillation for Semi-Supervised Instance Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.02668v1)  

---


**ABSTRACT**  
Although instance segmentation methods have improved considerably, the dominant paradigm is to rely on fully-annotated training images, which are tedious to obtain. To alleviate this reliance, and boost results, semi-supervised approaches leverage unlabeled data as an additional training signal that limits overfitting to the labeled samples. In this context, we present novel design choices to significantly improve teacher-student distillation models. In particular, we (i) improve the distillation approach by introducing a novel "guided burn-in" stage, and (ii) evaluate different instance segmentation architectures, as well as backbone networks and pre-training strategies. Contrary to previous work which uses only supervised data for the burn-in period of the student model, we also use guidance of the teacher model to exploit unlabeled data in the burn-in period. Our improved distillation approach leads to substantial improvements over previous state-of-the-art results. For example, on the Cityscapes dataset we improve mask-AP from 23.7 to 33.9 when using labels for 10\% of images, and on the COCO dataset we improve mask-AP from 18.3 to 34.1 when using labels for only 1\% of the training data.

{{</citation>}}


### (49/97) Bees Local Phase Quantization Feature Selection for RGB-D Facial Expressions Recognition (Seyed Muhammad Hossein Mousavi et al., 2023)

{{<citation>}}

Seyed Muhammad Hossein Mousavi, Atiye Ilanloo. (2023)  
**Bees Local Phase Quantization Feature Selection for RGB-D Facial Expressions Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2308.01700v1)  

---


**ABSTRACT**  
Feature selection could be defined as an optimization problem and solved by bio-inspired algorithms. Bees Algorithm (BA) shows decent performance in feature selection optimization tasks. On the other hand, Local Phase Quantization (LPQ) is a frequency domain feature which has excellent performance on Depth images. Here, after extracting LPQ features out of RGB (colour) and Depth images from the Iranian Kinect Face Database (IKFDB), the Bees feature selection algorithm applies to select the desired number of features for final classification tasks. IKFDB is recorded with Kinect sensor V.2 and contains colour and depth images for facial and facial micro-expressions recognition purposes. Here five facial expressions of Anger, Joy, Surprise, Disgust and Fear are used for final validation. The proposed Bees LPQ method is compared with Particle Swarm Optimization (PSO) LPQ, PCA LPQ, Lasso LPQ, and just LPQ features for classification tasks with Support Vector Machines (SVM), K-Nearest Neighbourhood (KNN), Shallow Neural Network and Ensemble Subspace KNN. Returned results, show a decent performance of the proposed algorithm (99 % accuracy) in comparison with others.

{{</citation>}}


### (50/97) BEVControl: Accurately Controlling Street-view Elements with Multi-perspective Consistency via BEV Sketch Layout (Kairui Yang et al., 2023)

{{<citation>}}

Kairui Yang, Enhui Ma, Jibin Peng, Qing Guo, Di Lin, Kaicheng Yu. (2023)  
**BEVControl: Accurately Controlling Street-view Elements with Multi-perspective Consistency via BEV Sketch Layout**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2308.01661v3)  

---


**ABSTRACT**  
Using synthesized images to boost the performance of perception models is a long-standing research challenge in computer vision. It becomes more eminent in visual-centric autonomous driving systems with multi-view cameras as some long-tail scenarios can never be collected. Guided by the BEV segmentation layouts, the existing generative networks seem to synthesize photo-realistic street-view images when evaluated solely on scene-level metrics. However, once zoom-in, they usually fail to produce accurate foreground and background details such as heading. To this end, we propose a two-stage generative method, dubbed BEVControl, that can generate accurate foreground and background contents. In contrast to segmentation-like input, it also supports sketch style input, which is more flexible for humans to edit. In addition, we propose a comprehensive multi-level evaluation protocol to fairly compare the quality of the generated scene, foreground object, and background geometry. Our extensive experiments show that our BEVControl surpasses the state-of-the-art method, BEVGen, by a significant margin, from 5.89 to 26.80 on foreground segmentation mIoU. In addition, we show that using images generated by BEVControl to train the downstream perception model, it achieves on average 1.29 improvement in NDS score.

{{</citation>}}


### (51/97) Multi-scale Cross-restoration Framework for Electrocardiogram Anomaly Detection (Aofan Jiang et al., 2023)

{{<citation>}}

Aofan Jiang, Chaoqin Huang, Qing Cao, Shuang Wu, Zi Zeng, Kang Chen, Ya Zhang, Yanfeng Wang. (2023)  
**Multi-scale Cross-restoration Framework for Electrocardiogram Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.01639v1)  

---


**ABSTRACT**  
Electrocardiogram (ECG) is a widely used diagnostic tool for detecting heart conditions. Rare cardiac diseases may be underdiagnosed using traditional ECG analysis, considering that no training dataset can exhaust all possible cardiac disorders. This paper proposes using anomaly detection to identify any unhealthy status, with normal ECGs solely for training. However, detecting anomalies in ECG can be challenging due to significant inter-individual differences and anomalies present in both global rhythm and local morphology. To address this challenge, this paper introduces a novel multi-scale cross-restoration framework for ECG anomaly detection and localization that considers both local and global ECG characteristics. The proposed framework employs a two-branch autoencoder to facilitate multi-scale feature learning through a masking and restoration process, with one branch focusing on global features from the entire ECG and the other on local features from heartbeat-level details, mimicking the diagnostic process of cardiologists. Anomalies are identified by their high restoration errors. To evaluate the performance on a large number of individuals, this paper introduces a new challenging benchmark with signal point-level ground truths annotated by experienced cardiologists. The proposed method demonstrates state-of-the-art performance on this benchmark and two other well-known ECG datasets. The benchmark dataset and source code are available at: \url{https://github.com/MediaBrain-SJTU/ECGAD}

{{</citation>}}


### (52/97) Disentangling Multi-view Representations Beyond Inductive Bias (Guanzhou Ke et al., 2023)

{{<citation>}}

Guanzhou Ke, Yang Yu, Guoqing Chao, Xiaoli Wang, Chenyang Xu, Shengfeng He. (2023)  
**Disentangling Multi-view Representations Beyond Inductive Bias**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.01634v2)  

---


**ABSTRACT**  
Multi-view (or -modality) representation learning aims to understand the relationships between different view representations. Existing methods disentangle multi-view representations into consistent and view-specific representations by introducing strong inductive biases, which can limit their generalization ability. In this paper, we propose a novel multi-view representation disentangling method that aims to go beyond inductive biases, ensuring both interpretability and generalizability of the resulting representations. Our method is based on the observation that discovering multi-view consistency in advance can determine the disentangling information boundary, leading to a decoupled learning objective. We also found that the consistency can be easily extracted by maximizing the transformation invariance and clustering consistency between views. These observations drive us to propose a two-stage framework. In the first stage, we obtain multi-view consistency by training a consistent encoder to produce semantically-consistent representations across views as well as their corresponding pseudo-labels. In the second stage, we disentangle specificity from comprehensive representations by minimizing the upper bound of mutual information between consistent and comprehensive representations. Finally, we reconstruct the original data by concatenating pseudo-labels and view-specific representations. Our experiments on four multi-view datasets demonstrate that our proposed method outperforms 12 comparison methods in terms of clustering and classification performance. The visualization results also show that the extracted consistency and specificity are compact and interpretable. Our code can be found at \url{https://github.com/Guanzhou-Ke/DMRIB}.

{{</citation>}}


### (53/97) Erasure-based Interaction Network for RGBT Video Object Detection and A Unified Benchmark (Zhengzheng Tu et al., 2023)

{{<citation>}}

Zhengzheng Tu, Qishun Wang, Hongshun Wang, Kunpeng Wang, Chenglong Li. (2023)  
**Erasure-based Interaction Network for RGBT Video Object Detection and A Unified Benchmark**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.01630v1)  

---


**ABSTRACT**  
Recently, many breakthroughs are made in the field of Video Object Detection (VOD), but the performance is still limited due to the imaging limitations of RGB sensors in adverse illumination conditions. To alleviate this issue, this work introduces a new computer vision task called RGB-thermal (RGBT) VOD by introducing the thermal modality that is insensitive to adverse illumination conditions. To promote the research and development of RGBT VOD, we design a novel Erasure-based Interaction Network (EINet) and establish a comprehensive benchmark dataset (VT-VOD50) for this task. Traditional VOD methods often leverage temporal information by using many auxiliary frames, and thus have large computational burden. Considering that thermal images exhibit less noise than RGB ones, we develop a negative activation function that is used to erase the noise of RGB features with the help of thermal image features. Furthermore, with the benefits from thermal images, we rely only on a small temporal window to model the spatio-temporal information to greatly improve efficiency while maintaining detection accuracy.   VT-VOD50 dataset consists of 50 pairs of challenging RGBT video sequences with complex backgrounds, various objects and different illuminations, which are collected in real traffic scenarios. Extensive experiments on VT-VOD50 dataset demonstrate the effectiveness and efficiency of our proposed method against existing mainstream VOD methods. The code of EINet and the dataset will be released to the public for free academic usage.

{{</citation>}}


### (54/97) A Multidimensional Analysis of Social Biases in Vision Transformers (Jannik Brinkmann et al., 2023)

{{<citation>}}

Jannik Brinkmann, Paul Swoboda, Christian Bartelt. (2023)  
**A Multidimensional Analysis of Social Biases in Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.01948v1)  

---


**ABSTRACT**  
The embedding spaces of image models have been shown to encode a range of social biases such as racism and sexism. Here, we investigate specific factors that contribute to the emergence of these biases in Vision Transformers (ViT). Therefore, we measure the impact of training data, model architecture, and training objectives on social biases in the learned representations of ViTs. Our findings indicate that counterfactual augmentation training using diffusion-based image editing can mitigate biases, but does not eliminate them. Moreover, we find that larger models are less biased than smaller models, and that models trained using discriminative objectives are less biased than those trained using generative objectives. In addition, we observe inconsistencies in the learned social biases. To our surprise, ViTs can exhibit opposite biases when trained on the same data set using different self-supervised objectives. Our findings give insights into the factors that contribute to the emergence of social biases and suggests that we could achieve substantial fairness improvements based on model design choices.

{{</citation>}}


### (55/97) ReIDTrack: Multi-Object Track and Segmentation Without Motion (Kaer Huang et al., 2023)

{{<citation>}}

Kaer Huang, Bingchuan Sun, Feng Chen, Tao Zhang, Jun Xie, Jian Li, Christopher Walter Twombly, Zhepeng Wang. (2023)  
**ReIDTrack: Multi-Object Track and Segmentation Without Motion**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.01622v1)  

---


**ABSTRACT**  
In recent years, dominant Multi-object tracking (MOT) and segmentation (MOTS) methods mainly follow the tracking-by-detection paradigm. Transformer-based end-to-end (E2E) solutions bring some ideas to MOT and MOTS, but they cannot achieve a new state-of-the-art (SOTA) performance in major MOT and MOTS benchmarks. Detection and association are two main modules of the tracking-by-detection paradigm. Association techniques mainly depend on the combination of motion and appearance information. As deep learning has been recently developed, the performance of the detection and appearance model is rapidly improved. These trends made us consider whether we can achieve SOTA based on only high-performance detection and appearance model. Our paper mainly focuses on exploring this direction based on CBNetV2 with Swin-B as a detection model and MoCo-v2 as a self-supervised appearance model. Motion information and IoU mapping were removed during the association. Our method wins 1st place on the MOTS track and wins 2nd on the MOT track in the CVPR2023 WAD workshop. We hope our simple and effective method can give some insights to the MOT and MOTS research community. Source code will be released under this git repository

{{</citation>}}


### (56/97) IndoHerb: Indonesia Medicinal Plants Recognition using Transfer Learning and Deep Learning (Muhammad Salman Ikrar Musyaffa et al., 2023)

{{<citation>}}

Muhammad Salman Ikrar Musyaffa, Novanto Yudistira, Muhammad Arif Rahman. (2023)  
**IndoHerb: Indonesia Medicinal Plants Recognition using Transfer Learning and Deep Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.01604v1)  

---


**ABSTRACT**  
Herbal plants are nutritious plants that can be used as an alternative to traditional disease healing. In Indonesia there are various types of herbal plants. But with the development of the times, the existence of herbal plants as traditional medicines began to be forgotten so that not everyone could recognize them. Having the ability to identify herbal plants can have many positive impacts. However, there is a problem where identifying plants can take a long time because it requires in-depth knowledge and careful examination of plant criteria. So that the application of computer vision can help identify herbal plants. Previously, research had been conducted on the introduction of herbal plants from Vietnam using several algorithms, but from these research the accuracy was not high enough. Therefore, this study intends to implement transfer learning from the Convolutional Neural Network (CNN) algorithm to classify types of herbal plants from Indonesia. This research was conducted by collecting image data of herbal plants from Indonesia independently through the Google Images search engine. After that, it will go through the data preprocessing, classification using the transfer learning method from CNN, and analysis will be carried out. The CNN transfer learning models used are ResNet34, DenseNet121, and VGG11_bn. Based on the test results of the three models, it was found that DenseNet121 was the model with the highest accuracy, which was 87.4%. In addition, testing was also carried out using the scratch model and obtained an accuracy of 43.53%. The Hyperparameter configuration used in this test is the ExponentialLR scheduler with a gamma value of 0.9; learning rate 0.001; Cross Entropy Loss function; Adam optimizer; and the number of epochs is 50. Indonesia Medicinal Plant Dataset can be accessed at the following link https://github.com/Salmanim20/indo_medicinal_plant

{{</citation>}}


### (57/97) Dynamic Token-Pass Transformers for Semantic Segmentation (Yuang Liu et al., 2023)

{{<citation>}}

Yuang Liu, Qiang Zhou, Jing Wang, Fan Wang, Jun Wang, Wei Zhang. (2023)  
**Dynamic Token-Pass Transformers for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.01944v1)  

---


**ABSTRACT**  
Vision transformers (ViT) usually extract features via forwarding all the tokens in the self-attention layers from top to toe. In this paper, we introduce dynamic token-pass vision transformers (DoViT) for semantic segmentation, which can adaptively reduce the inference cost for images with different complexity. DoViT gradually stops partial easy tokens from self-attention calculation and keeps the hard tokens forwarding until meeting the stopping criteria. We employ lightweight auxiliary heads to make the token-pass decision and divide the tokens into keeping/stopping parts. With a token separate calculation, the self-attention layers are speeded up with sparse tokens and still work friendly with hardware. A token reconstruction module is built to collect and reset the grouped tokens to their original position in the sequence, which is necessary to predict correct semantic masks. We conduct extensive experiments on two common semantic segmentation tasks, and demonstrate that our method greatly reduces about 40% $\sim$ 60% FLOPs and the drop of mIoU is within 0.8% for various segmentation transformers. The throughput and inference speed of ViT-L/B are increased to more than 2$\times$ on Cityscapes.

{{</citation>}}


### (58/97) Get the Best of Both Worlds: Improving Accuracy and Transferability by Grassmann Class Representation (Haoqi Wang et al., 2023)

{{<citation>}}

Haoqi Wang, Zhizhong Li, Wayne Zhang. (2023)  
**Get the Best of Both Worlds: Improving Accuracy and Transferability by Grassmann Class Representation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.01547v1)  

---


**ABSTRACT**  
We generalize the class vectors found in neural networks to linear subspaces (i.e.~points in the Grassmann manifold) and show that the Grassmann Class Representation (GCR) enables the simultaneous improvement in accuracy and feature transferability. In GCR, each class is a subspace and the logit is defined as the norm of the projection of a feature onto the class subspace. We integrate Riemannian SGD into deep learning frameworks such that class subspaces in a Grassmannian are jointly optimized with the rest model parameters. Compared to the vector form, the representative capability of subspaces is more powerful. We show that on ImageNet-1K, the top-1 error of ResNet50-D, ResNeXt50, Swin-T and Deit3-S are reduced by 5.6%, 4.5%, 3.0% and 3.5%, respectively. Subspaces also provide freedom for features to vary and we observed that the intra-class feature variability grows when the subspace dimension increases. Consequently, we found the quality of GCR features is better for downstream tasks. For ResNet50-D, the average linear transfer accuracy across 6 datasets improves from 77.98% to 79.70% compared to the strong baseline of vanilla softmax. For Swin-T, it improves from 81.5% to 83.4% and for Deit3, it improves from 73.8% to 81.4%. With these encouraging results, we believe that more applications could benefit from the Grassmann class representation. Code is released at https://github.com/innerlee/GCR.

{{</citation>}}


### (59/97) Multimodal Neurons in Pretrained Text-Only Transformers (Sarah Schwettmann et al., 2023)

{{<citation>}}

Sarah Schwettmann, Neil Chowdhury, Antonio Torralba. (2023)  
**Multimodal Neurons in Pretrained Text-Only Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.01544v1)  

---


**ABSTRACT**  
Language models demonstrate remarkable capacity to generalize representations learned in one modality to downstream tasks in other modalities. Can we trace this ability to individual neurons? We study the case where a frozen text transformer is augmented with vision using a self-supervised visual encoder and a single linear projection learned on an image-to-text task. Outputs of the projection layer are not immediately decodable into language describing image content; instead, we find that translation between modalities occurs deeper within the transformer. We introduce a procedure for identifying "multimodal neurons" that convert visual representations into corresponding text, and decoding the concepts they inject into the model's residual stream. In a series of experiments, we show that multimodal neurons operate on specific visual concepts across inputs, and have a systematic causal effect on image captioning.

{{</citation>}}


### (60/97) Multimodal Adaptation of CLIP for Few-Shot Action Recognition (Jiazheng Xing et al., 2023)

{{<citation>}}

Jiazheng Xing, Mengmeng Wang, Xiaojun Hou, Guang Dai, Jingdong Wang, Yong Liu. (2023)  
**Multimodal Adaptation of CLIP for Few-Shot Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.01532v1)  

---


**ABSTRACT**  
Applying large-scale pre-trained visual models like CLIP to few-shot action recognition tasks can benefit performance and efficiency. Utilizing the "pre-training, fine-tuning" paradigm makes it possible to avoid training a network from scratch, which can be time-consuming and resource-intensive. However, this method has two drawbacks. First, limited labeled samples for few-shot action recognition necessitate minimizing the number of tunable parameters to mitigate over-fitting, also leading to inadequate fine-tuning that increases resource consumption and may disrupt the generalized representation of models. Second, the video's extra-temporal dimension challenges few-shot recognition's effective temporal modeling, while pre-trained visual models are usually image models. This paper proposes a novel method called Multimodal Adaptation of CLIP (MA-CLIP) to address these issues. It adapts CLIP for few-shot action recognition by adding lightweight adapters, which can minimize the number of learnable parameters and enable the model to transfer across different tasks quickly. The adapters we design can combine information from video-text multimodal sources for task-oriented spatiotemporal modeling, which is fast, efficient, and has low training costs. Additionally, based on the attention mechanism, we design a text-guided prototype construction module that can fully utilize video-text information to enhance the representation of video prototypes. Our MA-CLIP is plug-and-play, which can be used in any different few-shot action recognition temporal alignment metric.

{{</citation>}}


### (61/97) Data Augmentation for Human Behavior Analysis in Multi-Person Conversations (Kun Li et al., 2023)

{{<citation>}}

Kun Li, Dan Guo, Guoliang Chen, Feiyang Liu, Meng Wang. (2023)  
**Data Augmentation for Human Behavior Analysis in Multi-Person Conversations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2308.01526v1)  

---


**ABSTRACT**  
In this paper, we present the solution of our team HFUT-VUT for the MultiMediate Grand Challenge 2023 at ACM Multimedia 2023. The solution covers three sub-challenges: bodily behavior recognition, eye contact detection, and next speaker prediction. We select Swin Transformer as the baseline and exploit data augmentation strategies to address the above three tasks. Specifically, we crop the raw video to remove the noise from other parts. At the same time, we utilize data augmentation to improve the generalization of the model. As a result, our solution achieves the best results of 0.6262 for bodily behavior recognition in terms of mean average precision and the accuracy of 0.7771 for eye contact detection on the corresponding test set. In addition, our approach also achieves comparable results of 0.5281 for the next speaker prediction in terms of unweighted average recall.

{{</citation>}}


### (62/97) VisAlign: Dataset for Measuring the Degree of Alignment between AI and Humans in Visual Perception (Jiyoung Lee et al., 2023)

{{<citation>}}

Jiyoung Lee, Seungho Kim, Seunghyun Won, Joonseok Lee, Marzyeh Ghassemi, James Thorne, Jaeseok Choi, O-Kil Kwon, Edward Choi. (2023)  
**VisAlign: Dataset for Measuring the Degree of Alignment between AI and Humans in Visual Perception**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01525v1)  

---


**ABSTRACT**  
AI alignment refers to models acting towards human-intended goals, preferences, or ethical principles. Given that most large-scale deep learning models act as black boxes and cannot be manually controlled, analyzing the similarity between models and humans can be a proxy measure for ensuring AI safety. In this paper, we focus on the models' visual perception alignment with humans, further referred to as AI-human visual alignment. Specifically, we propose a new dataset for measuring AI-human visual alignment in terms of image classification, a fundamental task in machine perception. In order to evaluate AI-human visual alignment, a dataset should encompass samples with various scenarios that may arise in the real world and have gold human perception labels. Our dataset consists of three groups of samples, namely Must-Act (i.e., Must-Classify), Must-Abstain, and Uncertain, based on the quantity and clarity of visual information in an image and further divided into eight categories. All samples have a gold human perception label; even Uncertain (severely blurry) sample labels were obtained via crowd-sourcing. The validity of our dataset is verified by sampling theory, statistical theories related to survey design, and experts in the related fields. Using our dataset, we analyze the visual alignment and reliability of five popular visual perception models and seven abstention methods. Our code and data is available at \url{https://github.com/jiyounglee-0523/VisAlign}.

{{</citation>}}


### (63/97) Food Classification using Joint Representation of Visual and Textual Data (Prateek Mittal et al., 2023)

{{<citation>}}

Prateek Mittal, Puneet Goyal, Joohi Chauhan. (2023)  
**Food Classification using Joint Representation of Visual and Textual Data**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-CY, cs-LG, cs.CV  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.02562v1)  

---


**ABSTRACT**  
Food classification is an important task in health care. In this work, we propose a multimodal classification framework that uses the modified version of EfficientNet with the Mish activation function for image classification, and the traditional BERT transformer-based network is used for text classification. The proposed network and the other state-of-the-art methods are evaluated on a large open-source dataset, UPMC Food-101. The experimental results show that the proposed network outperforms the other methods, a significant difference of 11.57% and 6.34% in accuracy is observed for image and text classification, respectively, when compared with the second-best performing method. We also compared the performance in terms of accuracy, precision, and recall for text classification using both machine learning and deep learning-based models. The comparative analysis from the prediction results of both images and text demonstrated the efficiency and robustness of the proposed approach.

{{</citation>}}


### (64/97) Contrastive Multi-FaceForensics: An End-to-end Bi-grained Contrastive Learning Approach for Multi-face Forgery Detection (Cong Zhang et al., 2023)

{{<citation>}}

Cong Zhang, Honggang Qi, Yuezun Li, Siwei Lyu. (2023)  
**Contrastive Multi-FaceForensics: An End-to-end Bi-grained Contrastive Learning Approach for Multi-face Forgery Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.01520v1)  

---


**ABSTRACT**  
DeepFakes have raised serious societal concerns, leading to a great surge in detection-based forensics methods in recent years. Face forgery recognition is the conventional detection method that usually follows a two-phase pipeline: it extracts the face first and then determines its authenticity by classification. Since DeepFakes in the wild usually contain multiple faces, using face forgery detection methods is merely practical as they have to process faces in a sequel, i.e., only one face is processed at the same time. One straightforward way to address this issue is to integrate face extraction and forgery detection in an end-to-end fashion by adapting advanced object detection architectures. However, as these object detection architectures are designed to capture the semantic information of different object categories rather than the subtle forgery traces among the faces, the direct adaptation is far from optimal. In this paper, we describe a new end-to-end framework, Contrastive Multi-FaceForensics (COMICS), to enhance multi-face forgery detection. The core of the proposed framework is a novel bi-grained contrastive learning approach that explores effective face forgery traces at both the coarse- and fine-grained levels. Specifically, the coarse-grained level contrastive learning captures the discriminative features among positive and negative proposal pairs in multiple scales with the instruction of the proposal generator, and the fine-grained level contrastive learning captures the pixel-wise discrepancy between the forged and original areas of the same face and the pixel-wise content inconsistency between different faces. Extensive experiments on the OpenForensics dataset demonstrate our method outperforms other counterparts by a large margin (~18.5%) and shows great potential for integration into various architectures.

{{</citation>}}


## eess.IV (3)



### (65/97) Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images (Alessandro Fontanella et al., 2023)

{{<citation>}}

Alessandro Fontanella, Grant Mair, Joanna Wardlaw, Emanuele Trucco, Amos Storkey. (2023)  
**Diffusion Models for Counterfactual Generation and Anomaly Detection in Brain Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.02062v1)  

---


**ABSTRACT**  
Segmentation masks of pathological areas are useful in many medical applications, such as brain tumour and stroke management. Moreover, healthy counterfactuals of diseased images can be used to enhance radiologists' training files and to improve the interpretability of segmentation models. In this work, we present a weakly supervised method to generate a healthy version of a diseased image and then use it to obtain a pixel-wise anomaly map. To do so, we start by considering a saliency map that approximately covers the pathological areas, obtained with ACAT. Then, we propose a technique that allows to perform targeted modifications to these regions, while preserving the rest of the image. In particular, we employ a diffusion model trained on healthy samples and combine Denoising Diffusion Probabilistic Model (DDPM) and Denoising Diffusion Implicit Model (DDIM) at each step of the sampling process. DDPM is used to modify the areas affected by a lesion within the saliency map, while DDIM guarantees reconstruction of the normal anatomy outside of it. The two parts are also fused at each timestep, to guarantee the generation of a sample with a coherent appearance and a seamless transition between edited and unedited parts. We verify that when our method is applied to healthy samples, the input images are reconstructed without significant modifications. We compare our approach with alternative weakly supervised methods on IST-3 for stroke lesion segmentation and on BraTS2021 for brain tumour segmentation, where we improve the DICE score of the best competing method from $0.6534$ to $0.7056$.

{{</citation>}}


### (66/97) Explainable unsupervised multi-modal image registration using deep networks (Chengjia Wang et al., 2023)

{{<citation>}}

Chengjia Wang, Giorgos Papanastasiou. (2023)  
**Explainable unsupervised multi-modal image registration using deep networks**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2308.01994v1)  

---


**ABSTRACT**  
Clinical decision making from magnetic resonance imaging (MRI) combines complementary information from multiple MRI sequences (defined as 'modalities'). MRI image registration aims to geometrically 'pair' diagnoses from different modalities, time points and slices. Both intra- and inter-modality MRI registration are essential components in clinical MRI settings. Further, an MRI image processing pipeline that can address both afine and non-rigid registration is critical, as both types of deformations may be occuring in real MRI data scenarios. Unlike image classification, explainability is not commonly addressed in image registration deep learning (DL) methods, as it is challenging to interpet model-data behaviours against transformation fields. To properly address this, we incorporate Grad-CAM-based explainability frameworks in each major component of our unsupervised multi-modal and multi-organ image registration DL methodology. We previously demonstrated that we were able to reach superior performance (against the current standard Syn method). In this work, we show that our DL model becomes fully explainable, setting the framework to generalise our approach on further medical imaging data.

{{</citation>}}


### (67/97) Unmasking Parkinson's Disease with Smile: An AI-enabled Screening Framework (Tariq Adnan et al., 2023)

{{<citation>}}

Tariq Adnan, Md Saiful Islam, Wasifur Rahman, Sangwu Lee, Sutapa Dey Tithi, Kazi Noshin, Imran Sarker, M Saifur Rahman, Ehsan Hoque. (2023)  
**Unmasking Parkinson's Disease with Smile: An AI-enabled Screening Framework**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02588v1)  

---


**ABSTRACT**  
Parkinson's disease (PD) diagnosis remains challenging due to lacking a reliable biomarker and limited access to clinical care. In this study, we present an analysis of the largest video dataset containing micro-expressions to screen for PD. We collected 3,871 videos from 1,059 unique participants, including 256 self-reported PD patients. The recordings are from diverse sources encompassing participants' homes across multiple countries, a clinic, and a PD care facility in the US. Leveraging facial landmarks and action units, we extracted features relevant to Hypomimia, a prominent symptom of PD characterized by reduced facial expressions. An ensemble of AI models trained on these features achieved an accuracy of 89.7% and an Area Under the Receiver Operating Characteristic (AUROC) of 89.3% while being free from detectable bias across population subgroups based on sex and ethnicity on held-out data. Further analysis reveals that features from the smiling videos alone lead to comparable performance, even on two external test sets the model has never seen during training, suggesting the potential for PD risk assessment from smiling selfie videos.

{{</citation>}}


## cs.IR (3)



### (68/97) Seasonality Based Reranking of E-commerce Autocomplete Using Natural Language Queries (Prateek Verma et al., 2023)

{{<citation>}}

Prateek Verma, Shan Zhong, Xiaoyu Liu, Adithya Rajan. (2023)  
**Seasonality Based Reranking of E-commerce Autocomplete Using Natural Language Queries**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: NLP, QA  
[Paper Link](http://arxiv.org/abs/2308.02055v1)  

---


**ABSTRACT**  
Query autocomplete (QAC) also known as typeahead, suggests list of complete queries as user types prefix in the search box. It is one of the key features of modern search engines specially in e-commerce. One of the goals of typeahead is to suggest relevant queries to users which are seasonally important. In this paper we propose a neural network based natural language processing (NLP) algorithm to incorporate seasonality as a signal and present end to end evaluation of the QAC ranking model. Incorporating seasonality into autocomplete ranking model can improve autocomplete relevance and business metric.

{{</citation>}}


### (69/97) ADRNet: A Generalized Collaborative Filtering Framework Combining Clinical and Non-Clinical Data for Adverse Drug Reaction Prediction (Haoxuan Li et al., 2023)

{{<citation>}}

Haoxuan Li, Taojun Hu, Zetong Xiong, Chunyuan Zheng, Fuli Feng, Xiangnan He, Xiao-Hua Zhou. (2023)  
**ADRNet: A Generalized Collaborative Filtering Framework Combining Clinical and Non-Clinical Data for Adverse Drug Reaction Prediction**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2308.02571v1)  

---


**ABSTRACT**  
Adverse drug reaction (ADR) prediction plays a crucial role in both health care and drug discovery for reducing patient mortality and enhancing drug safety. Recently, many studies have been devoted to effectively predict the drug-ADRs incidence rates. However, these methods either did not effectively utilize non-clinical data, i.e., physical, chemical, and biological information about the drug, or did little to establish a link between content-based and pure collaborative filtering during the training phase. In this paper, we first formulate the prediction of multi-label ADRs as a drug-ADR collaborative filtering problem, and to the best of our knowledge, this is the first work to provide extensive benchmark results of previous collaborative filtering methods on two large publicly available clinical datasets. Then, by exploiting the easy accessible drug characteristics from non-clinical data, we propose ADRNet, a generalized collaborative filtering framework combining clinical and non-clinical data for drug-ADR prediction. Specifically, ADRNet has a shallow collaborative filtering module and a deep drug representation module, which can exploit the high-dimensional drug descriptors to further guide the learning of low-dimensional ADR latent embeddings, which incorporates both the benefits of collaborative filtering and representation learning. Extensive experiments are conducted on two publicly available real-world drug-ADR clinical datasets and two non-clinical datasets to demonstrate the accuracy and efficiency of the proposed ADRNet. The code is available at https://github.com/haoxuanli-pku/ADRnet.

{{</citation>}}


### (70/97) Evaluating ChatGPT text-mining of clinical records for obesity monitoring (Ivo S. Fins et al., 2023)

{{<citation>}}

Ivo S. Fins, Heather Davies, Sean Farrell, Jose R. Torres, Gina Pinchbeck, Alan D. Radford, Peter-John Noble. (2023)  
**Evaluating ChatGPT text-mining of clinical records for obesity monitoring**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.01666v1)  

---


**ABSTRACT**  
Background: Veterinary clinical narratives remain a largely untapped resource for addressing complex diseases. Here we compare the ability of a large language model (ChatGPT) and a previously developed regular expression (RegexT) to identify overweight body condition scores (BCS) in veterinary narratives. Methods: BCS values were extracted from 4,415 anonymised clinical narratives using either RegexT or by appending the narrative to a prompt sent to ChatGPT coercing the model to return the BCS information. Data were manually reviewed for comparison. Results: The precision of RegexT was higher (100%, 95% CI 94.81-100%) than the ChatGPT (89.3%; 95% CI82.75-93.64%). However, the recall of ChatGPT (100%. 95% CI 96.18-100%) was considerably higher than that of RegexT (72.6%, 95% CI 63.92-79.94%). Limitations: Subtle prompt engineering is needed to improve ChatGPT output. Conclusions: Large language models create diverse opportunities and, whilst complex, present an intuitive interface to information but require careful implementation to avoid unpredictable errors.

{{</citation>}}


## cs.PL (1)



### (71/97) Gradual Sensitivity Typing (Damian Arquez et al., 2023)

{{<citation>}}

Damian Arquez, Matías Toro, Éric Tanter. (2023)  
**Gradual Sensitivity Typing**  

---
Primary Category: cs.PL  
Categories: cs-PL, cs.PL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.02018v1)  

---


**ABSTRACT**  
Reasoning about the sensitivity of functions with respect to their inputs has interesting applications in various areas, such as differential privacy. In order to check and enforce sensitivity, several approaches have been developed, notably sensitivity type systems. In these systems, sensitivity can be seen as an effect in the sense of type-and-effects systems as originally proposed by Gifford and Lucassen. Because type-and-effect systems can make certain useful programming patterns tedious or overly conservative, there is value in bringing the benefits of gradual typing to these disciplines in order to ease their adoption. In this work, we motivate, formalize, and prototype gradual sensitivity typing. The language GSoul supports both the unrestricted unknown sensitivity and bounded imprecision in the form of intervals. Gradual sensitivity typing allows programmers to smoothly evolve typed programs without any static sensitivity information towards hardened programs with a mix of static and dynamic sensitivity checking. In particular, we show that gradual sensitivity supports recursive functions for which fully static checking would be overly conservative, seamlessly enabling exact runtime sensitivity checks. GSoul satisfies both the gradual guarantees and sensitivity type soundness, known as metric preservation. We establish that, in general, gradual metric preservation is termination insensitive, and that one can achieve termination-sensitive gradual metric preservation by hardening specifications to bounded imprecision. We implement a prototype that provides an interactive test bed for gradual sensitivity typing. This work opens the door to gradualizing other typing disciplines that rely on function sensitivity such as differential privacy, as well as other quantitative type-based reasoning techniques.

{{</citation>}}


## cs.SD (2)



### (72/97) Federated Representation Learning for Automatic Speech Recognition (Guruprasad V Rames et al., 2023)

{{<citation>}}

Guruprasad V Rames, Gopinath Chennupati, Milind Rao, Anit Kumar Sahu, Ariya Rastrow, Jasha Droppo. (2023)  
**Federated Representation Learning for Automatic Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: LSTM, Representation Learning, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2308.02013v1)  

---


**ABSTRACT**  
Federated Learning (FL) is a privacy-preserving paradigm, allowing edge devices to learn collaboratively without sharing data. Edge devices like Alexa and Siri are prospective sources of unlabeled audio data that can be tapped to learn robust audio representations. In this work, we bring Self-supervised Learning (SSL) and FL together to learn representations for Automatic Speech Recognition respecting data privacy constraints. We use the speaker and chapter information in the unlabeled speech dataset, Libri-Light, to simulate non-IID speaker-siloed data distributions and pre-train an LSTM encoder with the Contrastive Predictive Coding framework with FedSGD. We show that the pre-trained ASR encoder in FL performs as well as a centrally pre-trained model and produces an improvement of 12-15% (WER) compared to no pre-training. We further adapt the federated pre-trained models to a new language, French, and show a 20% (WER) improvement over no pre-training.

{{</citation>}}


### (73/97) Adversarial Training of Denoising Diffusion Model Using Dual Discriminators for High-Fidelity Multi-Speaker TTS (Myeongjin Ko et al., 2023)

{{<citation>}}

Myeongjin Ko, Yong-Hoon Choi. (2023)  
**Adversarial Training of Denoising Diffusion Model Using Dual Discriminators for High-Fidelity Multi-Speaker TTS**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2308.01573v1)  

---


**ABSTRACT**  
The diffusion model is capable of generating high-quality data through a probabilistic approach. However, it suffers from the drawback of slow generation speed due to the requirement of a large number of time steps. To address this limitation, recent models such as denoising diffusion implicit models (DDIM) focus on generating samples without directly modeling the probability distribution, while models like denoising diffusion generative adversarial networks (GAN) combine diffusion processes with GANs. In the field of speech synthesis, a recent diffusion speech synthesis model called DiffGAN-TTS, utilizing the structure of GANs, has been introduced and demonstrates superior performance in both speech quality and generation speed. In this paper, to further enhance the performance of DiffGAN-TTS, we propose a speech synthesis model with two discriminators: a diffusion discriminator for learning the distribution of the reverse process and a spectrogram discriminator for learning the distribution of the generated data. Objective metrics such as structural similarity index measure (SSIM), mel-cepstral distortion (MCD), F0 root mean squared error (F0 RMSE), short-time objective intelligibility (STOI), perceptual evaluation of speech quality (PESQ), as well as subjective metrics like mean opinion score (MOS), are used to evaluate the performance of the proposed model. The evaluation results show that the proposed model outperforms recent state-of-the-art models such as FastSpeech2 and DiffGAN-TTS in various metrics. Our implementation and audio samples are located on GitHub.

{{</citation>}}


## cs.CY (2)



### (74/97) Silence Speaks Volumes: Re-weighting Techniques for Under-Represented Users in Fake News Detection (Mansooreh Karami et al., 2023)

{{<citation>}}

Mansooreh Karami, David Mosallanezhad, Paras Sheth, Huan Liu. (2023)  
**Silence Speaks Volumes: Re-weighting Techniques for Under-Represented Users in Fake News Detection**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-SI, cs.CY  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2308.02011v1)  

---


**ABSTRACT**  
Social media platforms provide a rich environment for analyzing user behavior. Recently, deep learning-based methods have been a mainstream approach for social media analysis models involving complex patterns. However, these methods are susceptible to biases in the training data, such as participation inequality. Basically, a mere 1% of users generate the majority of the content on social networking sites, while the remaining users, though engaged to varying degrees, tend to be less active in content creation and largely silent. These silent users consume and listen to information that is propagated on the platform. However, their voice, attitude, and interests are not reflected in the online content, making the decision of the current methods predisposed towards the opinion of the active users. So models can mistake the loudest users for the majority. We propose to leverage re-weighting techniques to make the silent majority heard, and in turn, investigate whether the cues from these users can improve the performance of the current models for the downstream task of fake news detection.

{{</citation>}}


### (75/97) A Critical Take on Privacy in a Datafied Society (Marco Cremonini, 2023)

{{<citation>}}

Marco Cremonini. (2023)  
**A Critical Take on Privacy in a Datafied Society**  

---
Primary Category: cs.CY  
Categories: A-1; K-4; K-4-1, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02573v1)  

---


**ABSTRACT**  
Privacy is an increasingly feeble constituent of the present datafied world and apparently the reason for that is clear: powerful actors worked to invade everyone's privacy for commercial and surveillance purposes. The existence of those actors and their agendas is undeniable, but the explanation is overly simplistic and contributed to create a narrative that tends to preserve the status quo. In this essay, I analyze several facets of the lack of online privacy and idiosyncrasies exhibited by privacy advocates, together with characteristics of the industry mostly responsible for the datafication process and why its asserted high effectiveness should be openly inquired. Then I discuss of possible effects of datafication on human behavior, the prevalent market-oriented assumption at the base of online privacy, and some emerging adaptation strategies. In the last part, the regulatory approach to online privacy is considered. The EU's GDPR is praised as the reference case of modern privacy regulations, but the same success hinders critical aspects that also emerged, from the quirks of the institutional decision process, to the flaws of the informed consent principle. A glimpse on the likely problematic future is provided with a discussion on privacy related aspects of EU, UK, and China's proposed generative AI policies.

{{</citation>}}


## cs.CR (1)



### (76/97) From Prompt Injections to SQL Injection Attacks: How Protected is Your LLM-Integrated Web Application? (Rodrigo Pedro et al., 2023)

{{<citation>}}

Rodrigo Pedro, Daniel Castro, Paulo Carreira, Nuno Santos. (2023)  
**From Prompt Injections to SQL Injection Attacks: How Protected is Your LLM-Integrated Web Application?**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.01990v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have found widespread applications in various domains, including web applications, where they facilitate human interaction via chatbots with natural language interfaces. Internally, aided by an LLM-integration middleware such as Langchain, user prompts are translated into SQL queries used by the LLM to provide meaningful responses to users. However, unsanitized user prompts can lead to SQL injection attacks, potentially compromising the security of the database. Despite the growing interest in prompt injection vulnerabilities targeting LLMs, the specific risks of generating SQL injection attacks through prompt injections have not been extensively studied. In this paper, we present a comprehensive examination of prompt-to-SQL (P$_2$SQL) injections targeting web applications based on the Langchain framework. Using Langchain as our case study, we characterize P$_2$SQL injections, exploring their variants and impact on application security through multiple concrete examples. Furthermore, we evaluate 7 state-of-the-art LLMs, demonstrating the pervasiveness of P$_2$SQL attacks across language models. Our findings indicate that LLM-integrated applications based on Langchain are highly susceptible to P$_2$SQL injection attacks, warranting the adoption of robust defenses. To counter these attacks, we propose four effective defense techniques that can be integrated as extensions to the Langchain framework. We validate the defenses through an experimental evaluation with a real-world use case application.

{{</citation>}}


## cs.OH (1)



### (77/97) IoT Based Smart Attendance System Using Rfid: A Systematic Literature Review (Kashif Ishaq et al., 2023)

{{<citation>}}

Kashif Ishaq, Samra Bibi. (2023)  
**IoT Based Smart Attendance System Using Rfid: A Systematic Literature Review**  

---
Primary Category: cs.OH  
Categories: cs-OH, cs.OH  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.02591v1)  

---


**ABSTRACT**  
The use of Radio Frequency Identification (RFID) technology is ubiquitous in a number of businesses and sectors, including retail sales, smart cities, agriculture, and transportation. Additionally, educational institutions have started using RFID to track student attendance, combining this technology with Google Sheets and the Internet of Things (IoT) to build a real-time attendance tracking system. For a thorough examination of the creation of a student attendance system, this paper includes a systematic literature evaluation of 21 major research published on IoT based attendance systems employing RFID. This RFID-based attendance system enables automation, eliminating several problems connected with the manual process, such as time wasting, proxies, and the possibility of losing the attendance sheet, in contrast to the traditional attendance system, which depends on manual signatures. By creating a system that automatically registers students' attendance by merely flashing their student cards at the RFID reader, all the aforementioned difficulties may be successfully addressed. This automated method guarantees attendance monitoring accuracy and dependability while also saving time. This paper's conclusion highlights the significant advantages of implementing an IoT-based attendance system based on RFID technology. The suggested solution provides a trustworthy, efficient, and secure alternative to manual attendance techniques, successfully addressing their shortcomings. This paper offers helpful insights for institutions looking to create a cutting-edge attendance system that increases student involvement and academic achievement by looking at guiding principles, best practices, and the successful resolution of difficulties.

{{</citation>}}


## cs.DL (1)



### (78/97) How many preprints have actually been printed and why: a case study of computer science preprints on arXiv (Jialiang Lin et al., 2023)

{{<citation>}}

Jialiang Lin, Yao Yu, Yu Zhou, Zhiyang Zhou, Xiaodong Shi. (2023)  
**How many preprints have actually been printed and why: a case study of computer science preprints on arXiv**  

---
Primary Category: cs.DL  
Categories: cs-AI, cs-CL, cs-DL, cs-LG, cs.DL  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.01899v1)  

---


**ABSTRACT**  
Preprints play an increasingly critical role in academic communities. There are many reasons driving researchers to post their manuscripts to preprint servers before formal submission to journals or conferences, but the use of preprints has also sparked considerable controversy, especially surrounding the claim of priority. In this paper, a case study of computer science preprints submitted to arXiv from 2008 to 2017 is conducted to quantify how many preprints have eventually been printed in peer-reviewed venues. Among those published manuscripts, some are published under different titles and without an update to their preprints on arXiv. In the case of these manuscripts, the traditional fuzzy matching method is incapable of mapping the preprint to the final published version. In view of this issue, we introduce a semantics-based mapping method with the employment of Bidirectional Encoder Representations from Transformers (BERT). With this new mapping method and a plurality of data sources, we find that 66% of all sampled preprints are published under unchanged titles and 11% are published under different titles and with other modifications. A further analysis was then performed to investigate why these preprints but not others were accepted for publication. Our comparison reveals that in the field of computer science, published preprints feature adequate revisions, multiple authorship, detailed abstract and introduction, extensive and authoritative references and available source code.

{{</citation>}}


## q-bio.QM (1)



### (79/97) Is your data alignable? Principled and interpretable alignability testing and integration of single-cell data (Rong Ma et al., 2023)

{{<citation>}}

Rong Ma, Eric D. Sun, David Donoho, James Zou. (2023)  
**Is your data alignable? Principled and interpretable alignability testing and integration of single-cell data**  

---
Primary Category: q-bio.QM  
Categories: cs-CV, q-bio-GN, q-bio-QM, q-bio.QM, stat-AP, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01839v1)  

---


**ABSTRACT**  
Single-cell data integration can provide a comprehensive molecular view of cells, and many algorithms have been developed to remove unwanted technical or biological variations and integrate heterogeneous single-cell datasets. Despite their wide usage, existing methods suffer from several fundamental limitations. In particular, we lack a rigorous statistical test for whether two high-dimensional single-cell datasets are alignable (and therefore should even be aligned). Moreover, popular methods can substantially distort the data during alignment, making the aligned data and downstream analysis difficult to interpret. To overcome these limitations, we present a spectral manifold alignment and inference (SMAI) framework, which enables principled and interpretable alignability testing and structure-preserving integration of single-cell data. SMAI provides a statistical test to robustly determine the alignability between datasets to avoid misleading inference, and is justified by high-dimensional statistical theory. On a diverse range of real and simulated benchmark datasets, it outperforms commonly used alignment methods. Moreover, we show that SMAI improves various downstream analyses such as identification of differentially expressed genes and imputation of single-cell spatial transcriptomics, providing further biological insights. SMAI's interpretability also enables quantification and a deeper understanding of the sources of technical confounders in single-cell data.

{{</citation>}}


## cs.AI (4)



### (80/97) Job Shop Scheduling via Deep Reinforcement Learning: a Sequence to Sequence approach (Giovanni Bonetta et al., 2023)

{{<citation>}}

Giovanni Bonetta, Davide Zago, Rossella Cancelliere, Andrea Grosso. (2023)  
**Job Shop Scheduling via Deep Reinforcement Learning: a Sequence to Sequence approach**  

---
Primary Category: cs.AI  
Categories: I-2-0; I-2-8; I-2-6, cs-AI, cs-LG, cs-NE, cs.AI, math-CO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.01797v1)  

---


**ABSTRACT**  
Job scheduling is a well-known Combinatorial Optimization problem with endless applications. Well planned schedules bring many benefits in the context of automated systems: among others, they limit production costs and waste. Nevertheless, the NP-hardness of this problem makes it essential to use heuristics whose design is difficult, requires specialized knowledge and often produces methods tailored to the specific task. This paper presents an original end-to-end Deep Reinforcement Learning approach to scheduling that automatically learns dispatching rules. Our technique is inspired by natural language encoder-decoder models for sequence processing and has never been used, to the best of our knowledge, for scheduling purposes. We applied and tested our method in particular to some benchmark instances of Job Shop Problem, but this technique is general enough to be potentially used to tackle other different optimal job scheduling tasks with minimal intervention. Results demonstrate that we outperform many classical approaches exploiting priority dispatching rules and show competitive results on state-of-the-art Deep Reinforcement Learning ones.

{{</citation>}}


### (81/97) Holy Grail 2.0: From Natural Language to Constraint Models (Dimos Tsouros et al., 2023)

{{<citation>}}

Dimos Tsouros, Hélène Verhaeghe, Serdar Kadıoğlu, Tias Guns. (2023)  
**Holy Grail 2.0: From Natural Language to Constraint Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-HC, cs.AI  
Keywords: GPT, Language Model, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.01589v1)  

---


**ABSTRACT**  
Twenty-seven years ago, E. Freuder highlighted that "Constraint programming represents one of the closest approaches computer science has yet made to the Holy Grail of programming: the user states the problem, the computer solves it". Nowadays, CP users have great modeling tools available (like Minizinc and CPMpy), allowing them to formulate the problem and then let a solver do the rest of the job, getting closer to the stated goal. However, this still requires the CP user to know the formalism and respect it. Another significant challenge lies in the expertise required to effectively model combinatorial problems. All this limits the wider adoption of CP. In this position paper, we investigate a possible approach to leverage pre-trained Large Language Models to extract models from textual problem descriptions. More specifically, we take inspiration from the Natural Language Processing for Optimization (NL4OPT) challenge and present early results with a decomposition-based prompting approach to GPT Models.

{{</citation>}}


### (82/97) InterAct: Exploring the Potentials of ChatGPT as a Cooperative Agent (Po-Lin Chen et al., 2023)

{{<citation>}}

Po-Lin Chen, Cheng-Shang Chang. (2023)  
**InterAct: Exploring the Potentials of ChatGPT as a Cooperative Agent**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.01552v1)  

---


**ABSTRACT**  
This research paper delves into the integration of OpenAI's ChatGPT into embodied agent systems, evaluating its influence on interactive decision-making benchmark. Drawing a parallel to the concept of people assuming roles according to their unique strengths, we introduce InterAct. In this approach, we feed ChatGPT with varied prompts, assigning it a numerous roles like a checker and a sorter, then integrating them with the original language model. Our research shows a remarkable success rate of 98% in AlfWorld, which consists of 6 different tasks in a simulated household environment, emphasizing the significance of proficient prompt engineering. The results highlight ChatGPT's competence in comprehending and performing intricate tasks effectively in real-world settings, thus paving the way for further advancements in task planning.

{{</citation>}}


### (83/97) Large-scale Generative Simulation Artificial Intelligence: the Next Hotspot in Generative AI (Qi Wang et al., 2023)

{{<citation>}}

Qi Wang, Yanghe Feng, Jincai Huang, Yiqin Lv, Zheng Xie, Xiaoshan Gao. (2023)  
**Large-scale Generative Simulation Artificial Intelligence: the Next Hotspot in Generative AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2308.02561v1)  

---


**ABSTRACT**  
The concept of GenAI has been developed for decades. Until recently, it has impressed us with substantial breakthroughs in natural language processing and computer vision, actively engaging in industrial scenarios. Noticing the practical challenges, e.g., limited learning resources, and overly dependencies on scientific discovery empiricism, we nominate large-scale generative simulation artificial intelligence (LS-GenAI) as the next hotspot for GenAI to connect.

{{</citation>}}


## quant-ph (1)



### (84/97) Benchmarking Adaptative Variational Quantum Algorithms on QUBO Instances (Gloria Turati et al., 2023)

{{<citation>}}

Gloria Turati, Maurizio Ferrari Dacrema, Paolo Cremonesi. (2023)  
**Benchmarking Adaptative Variational Quantum Algorithms on QUBO Instances**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.01789v1)  

---


**ABSTRACT**  
In recent years, Variational Quantum Algorithms (VQAs) have emerged as a promising approach for solving optimization problems on quantum computers in the NISQ era. However, one limitation of VQAs is their reliance on fixed-structure circuits, which may not be taylored for specific problems or hardware configurations. A leading strategy to address this issue are Adaptative VQAs, which dynamically modify the circuit structure by adding and removing gates, and optimize their parameters during the training. Several Adaptative VQAs, based on heuristics such as circuit shallowness, entanglement capability and hardware compatibility, have already been proposed in the literature, but there is still lack of a systematic comparison between the different methods. In this paper, we aim to fill this gap by analyzing three Adaptative VQAs: Evolutionary Variational Quantum Eigensolver (EVQE), Variable Ansatz (VAns), already proposed in the literature, and Random Adapt-VQE (RA-VQE), a random approach we introduce as a baseline. In order to compare these algorithms to traditional VQAs, we also include the Quantum Approximate Optimization Algorithm (QAOA) in our analysis. We apply these algorithms to QUBO problems and study their performance by examining the quality of the solutions found and the computational times required. Additionally, we investigate how the choice of the hyperparameters can impact the overall performance of the algorithms, highlighting the importance of selecting an appropriate methodology for hyperparameter tuning. Our analysis sets benchmarks for Adaptative VQAs designed for near-term quantum devices and provides valuable insights to guide future research in this area.

{{</citation>}}


## cs.SI (1)



### (85/97) Entropy-based detection of Twitter echo chambers (Manuel Pratelli et al., 2023)

{{<citation>}}

Manuel Pratelli, Fabio Saracco, Marinella Petrocchi. (2023)  
**Entropy-based detection of Twitter echo chambers**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-data-an, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2308.01750v1)  

---


**ABSTRACT**  
The presence of echo chambers, i.e. clusters of users exposed to news or opinions in line with their previous beliefs, was observed in many online debates on social platforms. Users form an echo chamber when two different phenomena appear at the same time: 1. users interact with ones sharing similar opinions; 2. users with similar opinions refer to the same pieces of news. We propose a completely unbiased entropy-based procedure to spot echo chambers. Remarkably, the method is completely agnostic about the nature of the data. In the Italian Twitter debate about Covid-19 vaccination, we find a limited presence of users in echo chambers (around 0.35% of all users), due to the limited number of validated users who are exposed to the same news. Nevertheless, their impact on the formation of a common discourse is strong, since echo chambers are responsible for nearly one-third of retweets of their discursive communities.

{{</citation>}}


## cs.RO (4)



### (86/97) Improving Wind Resistance Performance of Cascaded PID Controlled Quadcopters using Residual Reinforcement Learning (Yu Ishihara et al., 2023)

{{<citation>}}

Yu Ishihara, Yuichi Hazama, Kousuke Suzuki, Jerry Jun Yokono, Kohtaro Sabe, Kenta Kawamoto. (2023)  
**Improving Wind Resistance Performance of Cascaded PID Controlled Quadcopters using Residual Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.01648v1)  

---


**ABSTRACT**  
Wind resistance control is an essential feature for quadcopters to maintain their position to avoid deviation from target position and prevent collisions with obstacles. Conventionally, cascaded PID controller is used for the control of quadcopters for its simplicity and ease of tuning its parameters. However, it is weak against wind disturbances and the quadcopter can easily deviate from target position. In this work, we propose a residual reinforcement learning based approach to build a wind resistance controller of a quadcopter. By learning only the residual that compensates the disturbance, we can continue using the cascaded PID controller as the base controller of the quadcopter but improve its performance against wind disturbances. To avoid unexpected crashes and destructions of quadcopters, our method does not require real hardware for data collection and training. The controller is trained only on a simulator and directly applied to the target hardware without extra finetuning process. We demonstrate the effectiveness of our approach through various experiments including an experiment in an outdoor scene with wind speed greater than 13 m/s. Despite its simplicity, our controller reduces the position deviation by approximately 50% compared to the quadcopter controlled with the conventional cascaded PID controller. Furthermore, trained controller is robust and preserves its performance even though the quadcopter's mass and propeller's lift coefficient is changed between 50% to 150% from original training time.

{{</citation>}}


### (87/97) Mani-GPT: A Generative Model for Interactive Robotic Manipulation (Zhe Zhang et al., 2023)

{{<citation>}}

Zhe Zhang, Wei Chaid, Jiankun Wang. (2023)  
**Mani-GPT: A Generative Model for Interactive Robotic Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2308.01555v1)  

---


**ABSTRACT**  
In real-world scenarios, human dialogues are multi-round and diverse. Furthermore, human instructions can be unclear and human responses are unrestricted. Interactive robots face difficulties in understanding human intents and generating suitable strategies for assisting individuals through manipulation. In this article, we propose Mani-GPT, a Generative Pre-trained Transformer (GPT) for interactive robotic manipulation. The proposed model has the ability to understand the environment through object information, understand human intent through dialogues, generate natural language responses to human input, and generate appropriate manipulation plans to assist the human. This makes the human-robot interaction more natural and humanized. In our experiment, Mani-GPT outperforms existing algorithms with an accuracy of 84.6% in intent recognition and decision-making for actions. Furthermore, it demonstrates satisfying performance in real-world dialogue tests with users, achieving an average response accuracy of 70%.

{{</citation>}}


### (88/97) Avoidance Navigation Based on Offline Pre-Training Reinforcement Learning (Yang Wenkai Ji Ruihang Zhang Yuxiang Lei Hao et al., 2023)

{{<citation>}}

Yang Wenkai Ji Ruihang Zhang Yuxiang Lei Hao, Zhao Zijie. (2023)  
**Avoidance Navigation Based on Offline Pre-Training Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.01551v1)  

---


**ABSTRACT**  
This paper presents a Pre-Training Deep Reinforcement Learning(DRL) for avoidance navigation without map for mobile robots which map raw sensor data to control variable and navigate in an unknown environment. The efficient offline training strategy is proposed to speed up the inefficient random explorations in early stage and we also collect a universal dataset including expert experience for offline training, which is of some significance for other navigation training work. The pre-training and prioritized expert experience are proposed to reduce 80\% training time and has been verified to improve the 2 times reward of DRL. The advanced simulation gazebo with real physical modelling and dynamic equations reduce the gap between sim-to-real. We train our model a corridor environment, and evaluate the model in different environment getting the same effect. Compared to traditional method navigation, we can confirm the trained model can be directly applied into different scenarios and have the ability to no collision navigate. It was demonstrated that our DRL model have universal general capacity in different environment.

{{</citation>}}


### (89/97) Target-point Attention Transformer: A novel trajectory predict network for end-to-end autonomous driving (Jingyu Du et al., 2023)

{{<citation>}}

Jingyu Du, Yang Zhao, Hong Cheng. (2023)  
**Target-point Attention Transformer: A novel trajectory predict network for end-to-end autonomous driving**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.01496v1)  

---


**ABSTRACT**  
In the field of autonomous driving, there have been many excellent perception models for object detection, semantic segmentation, and other tasks, but how can we effectively use the perception models for vehicle planning? Traditional autonomous vehicle trajectory prediction methods not only need to obey traffic rules to avoid collisions, but also need to follow the prescribed route to reach the destination. In this paper, we propose a Transformer-based trajectory prediction network for end-to-end autonomous driving without rules called Target-point Attention Transformer network (TAT). We use the attention mechanism to realize the interaction between the predicted trajectory and the perception features as well as target-points. We demonstrate that our proposed method outperforms existing conditional imitation learning and GRU-based methods, significantly reducing the occurrence of accidents and improving route completion. We evaluate our approach in complex closed loop driving scenarios in cities using the CARLA simulator and achieve state-of-the-art performance.

{{</citation>}}


## math.NA (1)



### (90/97) Deep Learning-based surrogate models for parametrized PDEs: handling geometric variability through graph neural networks (Nicola Rares Franco et al., 2023)

{{<citation>}}

Nicola Rares Franco, Stefania Fresca, Filippo Tombari, Andrea Manzoni. (2023)  
**Deep Learning-based surrogate models for parametrized PDEs: handling geometric variability through graph neural networks**  

---
Primary Category: math.NA  
Categories: cs-LG, cs-NA, math-NA, math.NA  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.01602v1)  

---


**ABSTRACT**  
Mesh-based simulations play a key role when modeling complex physical systems that, in many disciplines across science and engineering, require the solution of parametrized time-dependent nonlinear partial differential equations (PDEs). In this context, full order models (FOMs), such as those relying on the finite element method, can reach high levels of accuracy, however often yielding intensive simulations to run. For this reason, surrogate models are developed to replace computationally expensive solvers with more efficient ones, which can strike favorable trade-offs between accuracy and efficiency. This work explores the potential usage of graph neural networks (GNNs) for the simulation of time-dependent PDEs in the presence of geometrical variability. In particular, we propose a systematic strategy to build surrogate models based on a data-driven time-stepping scheme where a GNN architecture is used to efficiently evolve the system. With respect to the majority of surrogate models, the proposed approach stands out for its ability of tackling problems with parameter dependent spatial domains, while simultaneously generalizing to different geometries and mesh resolutions. We assess the effectiveness of the proposed approach through a series of numerical experiments, involving both two- and three-dimensional problems, showing that GNNs can provide a valid alternative to traditional surrogate models in terms of computational efficiency and generalization to new scenarios. We also assess, from a numerical standpoint, the importance of using GNNs, rather than classical dense deep neural networks, for the proposed framework.

{{</citation>}}


## eess.SY (1)



### (91/97) Hierarchical Federated Learning in Wireless Networks: Pruning Tackles Bandwidth Scarcity and System Heterogeneity (Md Ferdous Pervej et al., 2023)

{{<citation>}}

Md Ferdous Pervej, Richeng Jin, Huaiyu Dai. (2023)  
**Hierarchical Federated Learning in Wireless Networks: Pruning Tackles Bandwidth Scarcity and System Heterogeneity**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-NI, cs-SY, eess-SY, eess.SY  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.01562v1)  

---


**ABSTRACT**  
While a practical wireless network has many tiers where end users do not directly communicate with the central server, the users' devices have limited computation and battery powers, and the serving base station (BS) has a fixed bandwidth. Owing to these practical constraints and system models, this paper leverages model pruning and proposes a pruning-enabled hierarchical federated learning (PHFL) in heterogeneous networks (HetNets). We first derive an upper bound of the convergence rate that clearly demonstrates the impact of the model pruning and wireless communications between the clients and the associated BS. Then we jointly optimize the model pruning ratio, central processing unit (CPU) frequency and transmission power of the clients in order to minimize the controllable terms of the convergence bound under strict delay and energy constraints. However, since the original problem is not convex, we perform successive convex approximation (SCA) and jointly optimize the parameters for the relaxed convex problem. Through extensive simulation, we validate the effectiveness of our proposed PHFL algorithm in terms of test accuracy, wall clock time, energy consumption and bandwidth requirement.

{{</citation>}}


## cs.HC (2)



### (92/97) Memory Sandbox: Transparent and Interactive Memory Management for Conversational Agents (Ziheng Huang et al., 2023)

{{<citation>}}

Ziheng Huang, Sebastian Gutierrez, Hemanth Kamana, Stephen MacNeil. (2023)  
**Memory Sandbox: Transparent and Interactive Memory Management for Conversational Agents**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2308.01542v1)  

---


**ABSTRACT**  
The recent advent of large language models (LLM) has resulted in high-performing conversational agents such as chatGPT. These agents must remember key information from an ongoing conversation to provide responses that are contextually relevant to the user. However, these agents have limited memory and can be distracted by irrelevant parts of the conversation. While many strategies exist to manage conversational memory, users currently lack affordances for viewing and controlling what the agent remembers, resulting in a poor mental model and conversational breakdowns. In this paper, we present Memory Sandbox, an interactive system and design probe that allows users to manage the conversational memory of LLM-powered agents. By treating memories as data objects that can be viewed, manipulated, recorded, summarized, and shared across conversations, Memory Sandbox provides interaction affordances for users to manage how the agent should `see' the conversation.

{{</citation>}}


### (93/97) Comparing scalable strategies for generating numerical perspectives (Hancheng Cao et al., 2023)

{{<citation>}}

Hancheng Cao, Sofia Eleni Spatharioti, Daniel G. Goldstein, Jake M. Hofman. (2023)  
**Comparing scalable strategies for generating numerical perspectives**  

---
Primary Category: cs.HC  
Categories: cs-CL, cs-CY, cs-HC, cs.HC  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.01535v1)  

---


**ABSTRACT**  
Numerical perspectives help people understand extreme and unfamiliar numbers (e.g., \$330 billion is about \$1,000 per person in the United States). While research shows perspectives to be helpful, generating them at scale is challenging both because it is difficult to identify what makes some analogies more helpful than others, and because what is most helpful can vary based on the context in which a given number appears. Here we present and compare three policies for large-scale perspective generation: a rule-based approach, a crowdsourced system, and a model that uses Wikipedia data and semantic similarity (via BERT embeddings) to generate context-specific perspectives. We find that the combination of these three approaches dominates any single method, with different approaches excelling in different settings and users displaying heterogeneous preferences across approaches. We conclude by discussing our deployment of perspectives in a widely-used online word processor.

{{</citation>}}


## cond-mat.dis-nn (1)



### (94/97) Non-equilibrium physics: from spin glasses to machine and neural learning (Weishun Zhong, 2023)

{{<citation>}}

Weishun Zhong. (2023)  
**Non-equilibrium physics: from spin glasses to machine and neural learning**  

---
Primary Category: cond-mat.dis-nn  
Categories: cond-mat-dis-nn, cond-mat-stat-mech, cond-mat.dis-nn, cs-AI, quant-ph, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01538v1)  

---


**ABSTRACT**  
Disordered many-body systems exhibit a wide range of emergent phenomena across different scales. These complex behaviors can be utilized for various information processing tasks such as error correction, learning, and optimization. Despite the empirical success of utilizing these systems for intelligent tasks, the underlying principles that govern their emergent intelligent behaviors remain largely unknown. In this thesis, we aim to characterize such emergent intelligence in disordered systems through statistical physics. We chart a roadmap for our efforts in this thesis based on two axes: learning mechanisms (long-term memory vs. working memory) and learning dynamics (artificial vs. natural). Throughout our journey, we uncover relationships between learning mechanisms and physical dynamics that could serve as guiding principles for designing intelligent systems. We hope that our investigation into the emergent intelligence of seemingly disparate learning systems can expand our current understanding of intelligence beyond neural systems and uncover a wider range of computational substrates suitable for AI applications.

{{</citation>}}


## cs.MM (1)



### (95/97) Learning Causality-inspired Representation Consistency for Video Anomaly Detection (Yang Liu et al., 2023)

{{<citation>}}

Yang Liu, Zhaoyang Xia, Mengyang Zhao, Donglai Wei, Yuzheng Wang, Liu Siao, Bobo Ju, Gaoyun Fang, Jing Liu, Liang Song. (2023)  
**Learning Causality-inspired Representation Consistency for Video Anomaly Detection**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.01537v1)  

---


**ABSTRACT**  
Video anomaly detection is an essential yet challenging task in the multimedia community, with promising applications in smart cities and secure communities. Existing methods attempt to learn abstract representations of regular events with statistical dependence to model the endogenous normality, which discriminates anomalies by measuring the deviations to the learned distribution. However, conventional representation learning is only a crude description of video normality and lacks an exploration of its underlying causality. The learned statistical dependence is unreliable for diverse regular events in the real world and may cause high false alarms due to overgeneralization. Inspired by causal representation learning, we think that there exists a causal variable capable of adequately representing the general patterns of regular events in which anomalies will present significant variations. Therefore, we design a causality-inspired representation consistency (CRC) framework to implicitly learn the unobservable causal variables of normality directly from available normal videos and detect abnormal events with the learned representation consistency. Extensive experiments show that the causality-inspired normality is robust to regular events with label-independent shifts, and the proposed CRC framework can quickly and accurately detect various complicated anomalies from real-world surveillance videos.

{{</citation>}}


## cs.MA (1)



### (96/97) Quantum Multi-Agent Reinforcement Learning for Autonomous Mobility Cooperation (Soohyun Park et al., 2023)

{{<citation>}}

Soohyun Park, Jae Pyoung Kim, Chanyoung Park, Soyi Jung, Joongheon Kim. (2023)  
**Quantum Multi-Agent Reinforcement Learning for Autonomous Mobility Cooperation**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.01519v1)  

---


**ABSTRACT**  
For Industry 4.0 Revolution, cooperative autonomous mobility systems are widely used based on multi-agent reinforcement learning (MARL). However, the MARL-based algorithms suffer from huge parameter utilization and convergence difficulties with many agents. To tackle these problems, a quantum MARL (QMARL) algorithm based on the concept of actor-critic network is proposed, which is beneficial in terms of scalability, to deal with the limitations in the noisy intermediate-scale quantum (NISQ) era. Additionally, our QMARL is also beneficial in terms of efficient parameter utilization and fast convergence due to quantum supremacy. Note that the reward in our QMARL is defined as task precision over computation time in multiple agents, thus, multi-agent cooperation can be realized. For further improvement, an additional technique for scalability is proposed, which is called projection value measure (PVM). Based on PVM, our proposed QMARL can achieve the highest reward, by reducing the action dimension into a logarithmic-scale. Finally, we can conclude that our proposed QMARL with PVM outperforms the other algorithms in terms of efficient parameter utilization, fast convergence, and scalability.

{{</citation>}}


## eess.SP (1)



### (97/97) Local-Global Temporal Fusion Network with an Attention Mechanism for Multiple and Multiclass Arrhythmia Classification (Yun Kwan Kim et al., 2023)

{{<citation>}}

Yun Kwan Kim, Minji Lee, Kunwook Jo, Hee Seok Song, Seong-Whan Lee. (2023)  
**Local-Global Temporal Fusion Network with an Attention Mechanism for Multiple and Multiclass Arrhythmia Classification**  

---
Primary Category: eess.SP  
Categories: 68T07, 92C55, cs-LG, eess-SP, eess.SP  
Keywords: Attention, Clinical  
[Paper Link](http://arxiv.org/abs/2308.02416v1)  

---


**ABSTRACT**  
Clinical decision support systems (CDSSs) have been widely utilized to support the decisions made by cardiologists when detecting and classifying arrhythmia from electrocardiograms (ECGs). However, forming a CDSS for the arrhythmia classification task is challenging due to the varying lengths of arrhythmias. Although the onset time of arrhythmia varies, previously developed methods have not considered such conditions. Thus, we propose a framework that consists of (i) local temporal information extraction, (ii) global pattern extraction, and (iii) local-global information fusion with attention to perform arrhythmia detection and classification with a constrained input length. The 10-class and 4-class performances of our approach were assessed by detecting the onset and offset of arrhythmia as an episode and the duration of arrhythmia based on the MIT-BIH arrhythmia database (MITDB) and MIT-BIH atrial fibrillation database (AFDB), respectively. The results were statistically superior to those achieved by the comparison models. To check the generalization ability of the proposed method, an AFDB-trained model was tested on the MITDB, and superior performance was attained compared with that of a state-of-the-art model. The proposed method can capture local-global information and dynamics without incurring information losses. Therefore, arrhythmias can be recognized more accurately, and their occurrence times can be calculated; thus, the clinical field can create more accurate treatment plans by using the proposed method.

{{</citation>}}
