---
draft: false
title: "arXiv @ 2023.08.24"
date: 2023-08-24
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.24"
    identifier: arxiv_20230824
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (16)](#cslg-16)
- [cs.CL (17)](#cscl-17)
- [cs.AR (1)](#csar-1)
- [cs.CR (4)](#cscr-4)
- [cs.SD (4)](#cssd-4)
- [cs.CV (33)](#cscv-33)
- [cs.SE (9)](#csse-9)
- [eess.IV (1)](#eessiv-1)
- [cs.SI (2)](#cssi-2)
- [cs.AI (3)](#csai-3)
- [cs.HC (2)](#cshc-2)
- [cs.IT (2)](#csit-2)
- [eess.SY (1)](#eesssy-1)
- [cs.DS (1)](#csds-1)
- [eess.SP (2)](#eesssp-2)
- [cs.MM (1)](#csmm-1)
- [cs.IR (6)](#csir-6)
- [econ.GN (1)](#econgn-1)
- [cs.RO (2)](#csro-2)
- [cs.DC (1)](#csdc-1)
- [stat.ME (1)](#statme-1)
- [cs.CY (1)](#cscy-1)
- [quant-ph (2)](#quant-ph-2)

## cs.LG (16)



### (1/113) Performance Comparison and Implementation of Bayesian Variants for Network Intrusion Detection (Tosin Ige et al., 2023)

{{<citation>}}

Tosin Ige, Christopher Kiekintveld. (2023)  
**Performance Comparison and Implementation of Bayesian Variants for Network Intrusion Detection**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2308.11834v1)  

---


**ABSTRACT**  
Bayesian classifiers perform well when each of the features is completely independent of the other which is not always valid in real world application. The aim of this study is to implement and compare the performances of each variant of Bayesian classifier (Multinomial, Bernoulli, and Gaussian) on anomaly detection in network intrusion, and to investigate whether there is any association between each variant assumption and their performance. Our investigation showed that each variant of Bayesian algorithm blindly follows its assumption regardless of feature property, and that the assumption is the single most important factor that influences their accuracy. Experimental results show that Bernoulli has accuracy of 69.9% test (71% train), Multinomial has accuracy of 31.2% test (31.2% train), while Gaussian has accuracy of 81.69% test (82.84% train). Going deeper, we investigated and found that each Naive Bayes variants performances and accuracy is largely due to each classifier assumption, Gaussian classifier performed best on anomaly detection due to its assumption that features follow normal distributions which are continuous, while multinomial classifier have a dismal performance as it simply assumes discreet and multinomial distribution.

{{</citation>}}


### (2/113) Mitigating Health Disparity on Biased Electronic Health Records via Deconfounder (Zheng Liu et al., 2023)

{{<citation>}}

Zheng Liu, Xiaohan Li, Philip Yu. (2023)  
**Mitigating Health Disparity on Biased Electronic Health Records via Deconfounder**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.11819v1)  

---


**ABSTRACT**  
The fairness issue of clinical data modeling, especially on Electronic Health Records (EHRs), is of utmost importance due to EHR's complex latent structure and potential selection bias. It is frequently necessary to mitigate health disparity while keeping the model's overall accuracy in practice. However, traditional methods often encounter the trade-off between accuracy and fairness, as they fail to capture the underlying factors beyond observed data. To tackle this challenge, we propose a novel model called Fair Longitudinal Medical Deconfounder (FLMD) that aims to achieve both fairness and accuracy in longitudinal Electronic Health Records (EHR) modeling. Drawing inspiration from the deconfounder theory, FLMD employs a two-stage training process. In the first stage, FLMD captures unobserved confounders for each encounter, which effectively represents underlying medical factors beyond observed EHR, such as patient genotypes and lifestyle habits. This unobserved confounder is crucial for addressing the accuracy/fairness dilemma. In the second stage, FLMD combines the learned latent representation with other relevant features to make predictions. By incorporating appropriate fairness criteria, such as counterfactual fairness, FLMD ensures that it maintains high prediction accuracy while simultaneously minimizing health disparities. We conducted comprehensive experiments on two real-world EHR datasets to demonstrate the effectiveness of FLMD. Apart from the comparison of baseline methods and FLMD variants in terms of fairness and accuracy, we assessed the performance of all models on disturbed/imbalanced and synthetic datasets to showcase the superiority of FLMD across different settings and provide valuable insights into its capabilities.

{{</citation>}}


### (3/113) Addressing Dynamic and Sparse Qualitative Data: A Hilbert Space Embedding of Categorical Variables (Anirban Mukherjee et al., 2023)

{{<citation>}}

Anirban Mukherjee, Hannah H. Chang. (2023)  
**Addressing Dynamic and Sparse Qualitative Data: A Hilbert Space Embedding of Categorical Variables**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.11781v1)  

---


**ABSTRACT**  
We propose a novel framework for incorporating qualitative data into quantitative models for causal estimation. Previous methods use categorical variables derived from qualitative data to build quantitative models. However, this approach can lead to data-sparse categories and yield inconsistent (asymptotically biased) and imprecise (finite sample biased) estimates if the qualitative information is dynamic and intricate. We use functional analysis to create a more nuanced and flexible framework. We embed the observed categories into a latent Baire space and introduce a continuous linear map -- a Hilbert space embedding -- from the Baire space of categories to a Reproducing Kernel Hilbert Space (RKHS) of representation functions. Through the Riesz representation theorem, we establish that the canonical treatment of categorical variables in causal models can be transformed into an identified structure in the RKHS. Transfer learning acts as a catalyst to streamline estimation -- embeddings from traditional models are paired with the kernel trick to form the Hilbert space embedding. We validate our model through comprehensive simulation evidence and demonstrate its relevance in a real-world study that contrasts theoretical predictions from economics and psychology in an e-commerce marketplace. The results confirm the superior performance of our model, particularly in scenarios where qualitative information is nuanced and complex.

{{</citation>}}


### (4/113) Few-shot Anomaly Detection in Text with Deviation Learning (Anindya Sundar Das et al., 2023)

{{<citation>}}

Anindya Sundar Das, Aravind Ajay, Sriparna Saha, Monowar Bhuyan. (2023)  
**Few-shot Anomaly Detection in Text with Deviation Learning**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.11780v1)  

---


**ABSTRACT**  
Most current methods for detecting anomalies in text concentrate on constructing models solely relying on unlabeled data. These models operate on the presumption that no labeled anomalous examples are available, which prevents them from utilizing prior knowledge of anomalies that are typically present in small numbers in many real-world applications. Furthermore, these models prioritize learning feature embeddings rather than optimizing anomaly scores directly, which could lead to suboptimal anomaly scoring and inefficient use of data during the learning process. In this paper, we introduce FATE, a deep few-shot learning-based framework that leverages limited anomaly examples and learns anomaly scores explicitly in an end-to-end method using deviation learning. In this approach, the anomaly scores of normal examples are adjusted to closely resemble reference scores obtained from a prior distribution. Conversely, anomaly samples are forced to have anomalous scores that considerably deviate from the reference score in the upper tail of the prior. Additionally, our model is optimized to learn the distinct behavior of anomalies by utilizing a multi-head self-attention layer and multiple instance learning approaches. Comprehensive experiments on several benchmark datasets demonstrate that our proposed approach attains a new level of state-of-the-art performance.

{{</citation>}}


### (5/113) Patient Clustering via Integrated Profiling of Clinical and Digital Data (Dongjin Choi et al., 2023)

{{<citation>}}

Dongjin Choi, Andy Xiang, Ozgur Ozturk, Deep Shrestha, Barry Drake, Hamid Haidarian, Faizan Javed, Haesun Park. (2023)  
**Patient Clustering via Integrated Profiling of Clinical and Digital Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2308.11748v1)  

---


**ABSTRACT**  
We introduce a novel profile-based patient clustering model designed for clinical data in healthcare. By utilizing a method grounded on constrained low-rank approximation, our model takes advantage of patients' clinical data and digital interaction data, including browsing and search, to construct patient profiles. As a result of the method, nonnegative embedding vectors are generated, serving as a low-dimensional representation of the patients. Our model was assessed using real-world patient data from a healthcare web portal, with a comprehensive evaluation approach which considered clustering and recommendation capabilities. In comparison to other baselines, our approach demonstrated superior performance in terms of clustering coherence and recommendation accuracy.

{{</citation>}}


### (6/113) Collect, Measure, Repeat: Reliability Factors for Responsible AI Data Collection (Oana Inel et al., 2023)

{{<citation>}}

Oana Inel, Tim Draws, Lora Aroyo. (2023)  
**Collect, Measure, Repeat: Reliability Factors for Responsible AI Data Collection**  

---
Primary Category: cs.LG  
Categories: cs-HC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.12885v1)  

---


**ABSTRACT**  
The rapid entry of machine learning approaches in our daily activities and high-stakes domains demands transparency and scrutiny of their fairness and reliability. To help gauge machine learning models' robustness, research typically focuses on the massive datasets used for their deployment, e.g., creating and maintaining documentation for understanding their origin, process of development, and ethical considerations. However, data collection for AI is still typically a one-off practice, and oftentimes datasets collected for a certain purpose or application are reused for a different problem. Additionally, dataset annotations may not be representative over time, contain ambiguous or erroneous annotations, or be unable to generalize across issues or domains. Recent research has shown these practices might lead to unfair, biased, or inaccurate outcomes. We argue that data collection for AI should be performed in a responsible manner where the quality of the data is thoroughly scrutinized and measured through a systematic set of appropriate metrics. In this paper, we propose a Responsible AI (RAI) methodology designed to guide the data collection with a set of metrics for an iterative in-depth analysis of the factors influencing the quality and reliability} of the generated data. We propose a granular set of measurements to inform on the internal reliability of a dataset and its external stability over time. We validate our approach across nine existing datasets and annotation tasks and four content modalities. This approach impacts the assessment of data robustness used for AI applied in the real world, where diversity of users and content is eminent. Furthermore, it deals with fairness and accountability aspects in data collection by providing systematic and transparent quality analysis for data collections.

{{</citation>}}


### (7/113) Tryage: Real-time, intelligent Routing of User Prompts to Large Language Models (Surya Narayanan Hari et al., 2023)

{{<citation>}}

Surya Narayanan Hari, Matt Thomson. (2023)  
**Tryage: Real-time, intelligent Routing of User Prompts to Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs-MA, cs.LG  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11601v2)  

---


**ABSTRACT**  
The introduction of the transformer architecture and the self-attention mechanism has led to an explosive production of language models trained on specific downstream tasks and data domains. With over 200, 000 models in the Hugging Face ecosystem, users grapple with selecting and optimizing models to suit multifaceted workflows and data domains while addressing computational, security, and recency concerns. There is an urgent need for machine learning frameworks that can eliminate the burden of model selection and customization and unleash the incredible power of the vast emerging model library for end users. Here, we propose a context-aware routing system, Tryage, that leverages a language model router for optimal selection of expert models from a model library based on analysis of individual input prompts. Inspired by the thalamic router in the brain, Tryage employs a perceptive router to predict down-stream model performance on prompts and, then, makes a routing decision using an objective function that integrates performance predictions with user goals and constraints that are incorporated through flags (e.g., model size, model recency). Tryage allows users to explore a Pareto front and automatically trade-off between task accuracy and secondary goals including minimization of model size, recency, security, verbosity, and readability. Across heterogeneous data sets that include code, text, clinical data, and patents, the Tryage framework surpasses Gorilla and GPT3.5 turbo in dynamic model selection identifying the optimal model with an accuracy of 50.9% , compared to 23.6% by GPT 3.5 Turbo and 10.8% by Gorilla. Conceptually, Tryage demonstrates how routing models can be applied to program and control the behavior of multi-model LLM systems to maximize efficient use of the expanding and evolving language model ecosystem.

{{</citation>}}


### (8/113) A Survey on Self-Supervised Representation Learning (Tobias Uelwer et al., 2023)

{{<citation>}}

Tobias Uelwer, Jan Robine, Stefan Sylvius Wagner, Marc Höftmann, Eric Upschulte, Sebastian Konietzny, Maike Behrendt, Stefan Harmeling. (2023)  
**A Survey on Self-Supervised Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-IV, stat-ML  
Keywords: Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.11455v1)  

---


**ABSTRACT**  
Learning meaningful representations is at the heart of many tasks in the field of modern machine learning. Recently, a lot of methods were introduced that allow learning of image representations without supervision. These representations can then be used in downstream tasks like classification or object detection. The quality of these representations is close to supervised learning, while no labeled images are needed. This survey paper provides a comprehensive review of these methods in a unified notation, points out similarities and differences of these methods, and proposes a taxonomy which sets these methods in relation to each other. Furthermore, our survey summarizes the most-recent experimental results reported in the literature in form of a meta-study. Our survey is intended as a starting point for researchers and practitioners who want to dive into the field of representation learning.

{{</citation>}}


### (9/113) Exploration of Rashomon Set Assists Explanations for Medical Data (Katarzyna Kobylińska et al., 2023)

{{<citation>}}

Katarzyna Kobylińska, Mateusz Krzyziński, Rafał Machowicz, Mariusz Adamek, Przemysław Biecek. (2023)  
**Exploration of Rashomon Set Assists Explanations for Medical Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11446v1)  

---


**ABSTRACT**  
The machine learning modeling process conventionally culminates in selecting a single model that maximizes a selected performance metric. However, this approach leads to abandoning a more profound analysis of slightly inferior models. Particularly in medical and healthcare studies, where the objective extends beyond predictions to valuable insight generation, relying solely on performance metrics can result in misleading or incomplete conclusions. This problem is particularly pertinent when dealing with a set of models with performance close to maximum one, known as $\textit{Rashomon set}$. Such a set can be numerous and may contain models describing the data in a different way, which calls for comprehensive analysis. This paper introduces a novel process to explore Rashomon set models, extending the conventional modeling approach. The cornerstone is the identification of the most different models within the Rashomon set, facilitated by the introduced $\texttt{Rashomon_DETECT}$ algorithm. This algorithm compares profiles illustrating prediction dependencies on variable values generated by eXplainable Artificial Intelligence (XAI) techniques. To quantify differences in variable effects among models, we introduce the Profile Disparity Index (PDI) based on measures from functional data analysis. To illustrate the effectiveness of our approach, we showcase its application in predicting survival among hemophagocytic lymphohistiocytosis (HLH) patients - a foundational case study. Additionally, we benchmark our approach on other medical data sets, demonstrating its versatility and utility in various contexts.

{{</citation>}}


### (10/113) Targeted Data Augmentation for bias mitigation (Agnieszka Mikołajczyk-Bareła et al., 2023)

{{<citation>}}

Agnieszka Mikołajczyk-Bareła, Maria Ferlin, Michał Grochowski. (2023)  
**Targeted Data Augmentation for bias mitigation**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-CY, cs-LG, cs.LG  
Keywords: AI, Augmentation, Bias  
[Paper Link](http://arxiv.org/abs/2308.11386v1)  

---


**ABSTRACT**  
The development of fair and ethical AI systems requires careful consideration of bias mitigation, an area often overlooked or ignored. In this study, we introduce a novel and efficient approach for addressing biases called Targeted Data Augmentation (TDA), which leverages classical data augmentation techniques to tackle the pressing issue of bias in data and models. Unlike the laborious task of removing biases, our method proposes to insert biases instead, resulting in improved performance. To identify biases, we annotated two diverse datasets: a dataset of clinical skin lesions and a dataset of male and female faces. These bias annotations are published for the first time in this study, providing a valuable resource for future research. Through Counterfactual Bias Insertion, we discovered that biases associated with the frame, ruler, and glasses had a significant impact on models. By randomly introducing biases during training, we mitigated these biases and achieved a substantial decrease in bias measures, ranging from two-fold to more than 50-fold, while maintaining a negligible increase in the error rate.

{{</citation>}}


### (11/113) Class Label-aware Graph Anomaly Detection (Junghoon Kim et al., 2023)

{{<citation>}}

Junghoon Kim, Yeonjun In, Kanghoon Yoon, Junmo Lee, Chanyoung Park. (2023)  
**Class Label-aware Graph Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.11669v1)  

---


**ABSTRACT**  
Unsupervised GAD methods assume the lack of anomaly labels, i.e., whether a node is anomalous or not. One common observation we made from previous unsupervised methods is that they not only assume the absence of such anomaly labels, but also the absence of class labels (the class a node belongs to used in a general node classification task). In this work, we study the utility of class labels for unsupervised GAD; in particular, how they enhance the detection of structural anomalies. To this end, we propose a Class Label-aware Graph Anomaly Detection framework (CLAD) that utilizes a limited amount of labeled nodes to enhance the performance of unsupervised GAD. Extensive experiments on ten datasets demonstrate the superior performance of CLAD in comparison to existing unsupervised GAD methods, even in the absence of ground-truth class label information. The source code for CLAD is available at \url{https://github.com/jhkim611/CLAD}.

{{</citation>}}


### (12/113) Uncertainty Estimation of Transformers' Predictions via Topological Analysis of the Attention Matrices (Elizaveta Kostenok et al., 2023)

{{<citation>}}

Elizaveta Kostenok, Daniil Cherniavskii, Alexey Zaytsev. (2023)  
**Uncertainty Estimation of Transformers' Predictions via Topological Analysis of the Attention Matrices**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.11295v1)  

---


**ABSTRACT**  
Determining the degree of confidence of deep learning model in its prediction is an open problem in the field of natural language processing. Most of the classical methods for uncertainty estimation are quite weak for text classification models. We set the task of obtaining an uncertainty estimate for neural networks based on the Transformer architecture. A key feature of such mo-dels is the attention mechanism, which supports the information flow between the hidden representations of tokens in the neural network. We explore the formed relationships between internal representations using Topological Data Analysis methods and utilize them to predict model's confidence. In this paper, we propose a method for uncertainty estimation based on the topological properties of the attention mechanism and compare it with classical methods. As a result, the proposed algorithm surpasses the existing methods in quality and opens up a new area of application of the attention mechanism, but requires the selection of topological features.

{{</citation>}}


### (13/113) FoX: Formation-aware exploration in multi-agent reinforcement learning (Yonghyeon Jo et al., 2023)

{{<citation>}}

Yonghyeon Jo, Sunwoo Lee, Junghyuk Yum, Seungyul Han. (2023)  
**FoX: Formation-aware exploration in multi-agent reinforcement learning**  

---
Primary Category: cs.LG  
Categories: Machine Learning (ML) - ML: Reinforcement Learning, Secondary
  Subject Areas: Multiagent Systems (MAS) - MAS: Multiagent Learning, cs-LG, cs.LG  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.11272v1)  

---


**ABSTRACT**  
Recently, deep multi-agent reinforcement learning (MARL) has gained significant popularity due to its success in various cooperative multi-agent tasks. However, exploration still remains a challenging problem in MARL due to the partial observability of the agents and the exploration space that can grow exponentially as the number of agents increases. Firstly, in order to address the scalability issue of the exploration space, we define a formation-based equivalence relation on the exploration space and aim to reduce the search space by exploring only meaningful states in different formations. Then, we propose a novel formation-aware exploration (FoX) framework that encourages partially observable agents to visit the states in diverse formations by guiding them to be well aware of their current formation solely based on their own observations. Numerical results show that the proposed FoX framework significantly outperforms the state-of-the-art MARL algorithms on Google Research Football (GRF) and sparse Starcraft II multi-agent challenge (SMAC) tasks.

{{</citation>}}


### (14/113) SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting (Shengsheng Lin et al., 2023)

{{<citation>}}

Shengsheng Lin, Weiwei Lin, Wentai Wu, Feiyu Zhao, Ruichao Mo, Haotong Zhang. (2023)  
**SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2308.11200v1)  

---


**ABSTRACT**  
RNN-based methods have faced challenges in the Long-term Time Series Forecasting (LTSF) domain when dealing with excessively long look-back windows and forecast horizons. Consequently, the dominance in this domain has shifted towards Transformer, MLP, and CNN approaches. The substantial number of recurrent iterations are the fundamental reasons behind the limitations of RNNs in LTSF. To address these issues, we propose two novel strategies to reduce the number of iterations in RNNs for LTSF tasks: Segment-wise Iterations and Parallel Multi-step Forecasting (PMF). RNNs that combine these strategies, namely SegRNN, significantly reduce the required recurrent iterations for LTSF, resulting in notable improvements in forecast accuracy and inference speed. Extensive experiments demonstrate that SegRNN not only outperforms SOTA Transformer-based models but also reduces runtime and memory usage by more than 78%. These achievements provide strong evidence that RNNs continue to excel in LTSF tasks and encourage further exploration of this domain with more RNN-based approaches. The source code is coming soon.

{{</citation>}}


### (15/113) Graph Encoding and Neural Network Approaches for Volleyball Analytics: From Game Outcome to Individual Play Predictions (Rhys Tracy et al., 2023)

{{<citation>}}

Rhys Tracy, Haotian Xia, Alex Rasla, Yuan-Fang Wang, Ambuj Singh. (2023)  
**Graph Encoding and Neural Network Approaches for Volleyball Analytics: From Game Outcome to Individual Play Predictions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.11142v1)  

---


**ABSTRACT**  
This research aims to improve the accuracy of complex volleyball predictions and provide more meaningful insights to coaches and players. We introduce a specialized graph encoding technique to add additional contact-by-contact volleyball context to an already available volleyball dataset without any additional data gathering. We demonstrate the potential benefits of using graph neural networks (GNNs) on this enriched dataset for three different volleyball prediction tasks: rally outcome prediction, set location prediction, and hit type prediction. We compare the performance of our graph-based models to baseline models and analyze the results to better understand the underlying relationships in a volleyball rally. Our results show that the use of GNNs with our graph encoding yields a much more advanced analysis of the data, which noticeably improves prediction results overall. We also show that these baseline tasks can be significantly improved with simple adjustments, such as removing blocked hits. Lastly, we demonstrate the importance of choosing a model architecture that will better extract the important information for a certain task. Overall, our study showcases the potential strengths and weaknesses of using graph encodings in sports data analytics and hopefully will inspire future improvements in machine learning strategies across sports and applications by using graphbased encodings.

{{</citation>}}


### (16/113) Transformers for Capturing Multi-level Graph Structure using Hierarchical Distances (Yuankai Luo, 2023)

{{<citation>}}

Yuankai Luo. (2023)  
**Transformers for Capturing Multi-level Graph Structure using Hierarchical Distances**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.11129v1)  

---


**ABSTRACT**  
Graph transformers need strong inductive biases to derive meaningful attention scores. Yet, current proposals rarely address methods capturing longer ranges, hierarchical structures, or community structures, as they appear in various graphs such as molecules, social networks, and citation networks. In this paper, we propose a hierarchy-distance structural encoding (HDSE), which models a hierarchical distance between the nodes in a graph focusing on its multi-level, hierarchical nature. In particular, this yields a framework which can be flexibly integrated with existing graph transformers, allowing for simultaneous application with other positional representations. Through extensive experiments on 12 real-world datasets, we demonstrate that our HDSE method successfully enhances various types of baseline transformers, achieving state-of-the-art empirical performances on 10 benchmark datasets.

{{</citation>}}


## cs.CL (17)



### (17/113) Exploring the Effectiveness of GPT Models in Test-Taking: A Case Study of the Driver's License Knowledge Test (Saba Rahimi et al., 2023)

{{<citation>}}

Saba Rahimi, Tucker Balch, Manuela Veloso. (2023)  
**Exploring the Effectiveness of GPT Models in Test-Taking: A Case Study of the Driver's License Knowledge Test**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2308.11827v1)  

---


**ABSTRACT**  
Large language models such as Open AI's Generative Pre-trained Transformer (GPT) models are proficient at answering questions, but their knowledge is confined to the information present in their training data. This limitation renders them ineffective when confronted with questions about recent developments or non-public documents. Our research proposes a method that enables GPT models to answer questions by employing context from an information source not previously included in their training data. The methodology includes preprocessing of contextual information, the embedding of contexts and queries, constructing prompt through the integration of context embeddings, and generating answers using GPT models. We applied this method in a controlled test scenario using the California Driver's Handbook as the information source. The GPT-3 model achieved a 96% passing score on a set of 50 sample driving knowledge test questions. In contrast, without context, the model's passing score fell to 82%. However, the model still fails to answer some questions correctly even with providing library of context, highlighting room for improvement. The research also examined the impact of prompt length and context format, on the model's performance. Overall, the study provides insights into the limitations and potential improvements for GPT models in question-answering tasks.

{{</citation>}}


### (18/113) Towards an On-device Agent for Text Rewriting (Yun Zhu et al., 2023)

{{<citation>}}

Yun Zhu, Yinxiao Liu, Felix Stahlberg, Shankar Kumar, Yu-hui Chen, Liangchen Luo, Lei Shu, Renjie Liu, Jindong Chen, Lei Meng. (2023)  
**Towards an On-device Agent for Text Rewriting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.11807v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated impressive capabilities for text rewriting. Nonetheless, the large sizes of these models make them impractical for on-device inference, which would otherwise allow for enhanced privacy and economical inference. Creating a smaller yet potent language model for text rewriting presents a formidable challenge because it requires balancing the need for a small size with the need to retain the emergent capabilities of the LLM, that requires costly data collection. To address the above challenge, we introduce a new instruction tuning approach for building a mobile-centric text rewriting model. Our strategies enable the generation of high quality training data without any human labeling. In addition, we propose a heuristic reinforcement learning framework which substantially enhances performance without requiring preference data. To further bridge the performance gap with the larger server-side model, we propose an effective approach that combines the mobile rewrite agent with the server model using a cascade. To tailor the text rewriting tasks to mobile scenarios, we introduce MessageRewriteEval, a benchmark that focuses on text rewriting for messages through natural language instructions. Through empirical experiments, we demonstrate that our on-device model surpasses the current state-of-the-art LLMs in text rewriting while maintaining a significantly reduced model size. Notably, we show that our proposed cascading approach improves model performance.

{{</citation>}}


### (19/113) Identifying depression-related topics in smartphone-collected free-response speech recordings using an automatic speech recognition system and a deep learning topic model (Yuezhou Zhang et al., 2023)

{{<citation>}}

Yuezhou Zhang, Amos A Folarin, Judith Dineley, Pauline Conde, Valeria de Angel, Shaoxiong Sun, Yatharth Ranjan, Zulqarnain Rashid, Callum Stewart, Petroula Laiou, Heet Sankesara, Linglong Qian, Faith Matcham, Katie M White, Carolin Oetzmann, Femke Lamers, Sara Siddi, Sara Simblett, Björn W. Schuller, Srinivasan Vairavan, Til Wykes, Josep Maria Haro, Brenda WJH Penninx, Vaibhav A Narayan, Matthew Hotopf, Richard JB Dobson, Nicholas Cummins, RADAR-CNS consortium. (2023)  
**Identifying depression-related topics in smartphone-collected free-response speech recordings using an automatic speech recognition system and a deep learning topic model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-SD, cs.CL, eess-AS, q-bio-QM  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.11773v1)  

---


**ABSTRACT**  
Language use has been shown to correlate with depression, but large-scale validation is needed. Traditional methods like clinic studies are expensive. So, natural language processing has been employed on social media to predict depression, but limitations remain-lack of validated labels, biased user samples, and no context. Our study identified 29 topics in 3919 smartphone-collected speech recordings from 265 participants using the Whisper tool and BERTopic model. Six topics with a median PHQ-8 greater than or equal to 10 were regarded as risk topics for depression: No Expectations, Sleep, Mental Therapy, Haircut, Studying, and Coursework. To elucidate the topic emergence and associations with depression, we compared behavioral (from wearables) and linguistic characteristics across identified topics. The correlation between topic shifts and changes in depression severity over time was also investigated, indicating the importance of longitudinally monitoring language use. We also tested the BERTopic model on a similar smaller dataset (356 speech recordings from 57 participants), obtaining some consistent results. In summary, our findings demonstrate specific speech topics may indicate depression severity. The presented data-driven workflow provides a practical approach to collecting and analyzing large-scale speech data from real-world settings for digital health research.

{{</citation>}}


### (20/113) Halo: Estimation and Reduction of Hallucinations in Open-Source Weak Large Language Models (Mohamed Elaraby et al., 2023)

{{<citation>}}

Mohamed Elaraby, Mengyin Lu, Jacob Dunn, Xueying Zhang, Yu Wang, Shizhu Liu. (2023)  
**Halo: Estimation and Reduction of Hallucinations in Open-Source Weak Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLOOM, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.11764v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have revolutionized Natural Language Processing (NLP). Although convenient for research and practical applications, open-source LLMs with fewer parameters often suffer from severe hallucinations compared to their larger counterparts. This paper focuses on measuring and reducing hallucinations in BLOOM 7B, a representative of such weaker open-source LLMs that are publicly available for research and commercial applications. We introduce HaloCheck, a lightweight BlackBox knowledge-free framework designed to quantify the severity of hallucinations in LLMs. Additionally, we explore techniques like knowledge injection and teacher-student approaches to alleviate hallucinations in low-parameter LLMs. Our experiments effectively demonstrate the reduction of hallucinations in challenging domains for these LLMs.

{{</citation>}}


### (21/113) Knowledge Graph Prompting for Multi-Document Question Answering (Yu Wang et al., 2023)

{{<citation>}}

Yu Wang, Nedim Lipka, Ryan A. Rossi, Alexa Siu, Ruiyi Zhang, Tyler Derr. (2023)  
**Knowledge Graph Prompting for Multi-Document Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Knowledge Graph, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.11730v1)  

---


**ABSTRACT**  
The 'pre-train, prompt, predict' paradigm of large language models (LLMs) has achieved remarkable success in open-domain question answering (OD-QA). However, few works explore this paradigm in the scenario of multi-document question answering (MD-QA), a task demanding a thorough understanding of the logical associations among the contents and structures of different documents. To fill this crucial gap, we propose a Knowledge Graph Prompting (KGP) method to formulate the right context in prompting LLMs for MD-QA, which consists of a graph construction module and a graph traversal module. For graph construction, we create a knowledge graph (KG) over multiple documents with nodes symbolizing passages or document structures (e.g., pages/tables), and edges denoting the semantic/lexical similarity between passages or intra-document structural relations. For graph traversal, we design an LM-guided graph traverser that navigates across nodes and gathers supporting passages assisting LLMs in MD-QA. The constructed graph serves as the global ruler that regulates the transitional space among passages and reduces retrieval latency. Concurrently, the LM-guided traverser acts as a local navigator that gathers pertinent context to progressively approach the question and guarantee retrieval quality. Extensive experiments underscore the efficacy of KGP for MD-QA, signifying the potential of leveraging graphs in enhancing the prompt design for LLMs. Our code is at https://github.com/YuWVandy/KG-LLM-MDQA.

{{</citation>}}


### (22/113) Efficient Benchmarking (of Language Models) (Yotam Perlitz et al., 2023)

{{<citation>}}

Yotam Perlitz, Elron Bandel, Ariel Gera, Ofir Arviv, Liat Ein-Dor, Eyal Shnarch, Noam Slonim, Michal Shmueli-Scheuer, Leshem Choshen. (2023)  
**Efficient Benchmarking (of Language Models)**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.11696v1)  

---


**ABSTRACT**  
The increasing versatility of language models LMs has given rise to a new class of benchmarks that comprehensively assess a broad range of capabilities. Such benchmarks are associated with massive computational costs reaching thousands of GPU hours per model. However the efficiency aspect of these evaluation efforts had raised little discussion in the literature. In this work we present the problem of Efficient Benchmarking namely intelligently reducing the computation costs of LM evaluation without compromising reliability. Using the HELM benchmark as a test case we investigate how different benchmark design choices affect the computation-reliability tradeoff. We propose to evaluate the reliability of such decisions by using a new measure Decision Impact on Reliability DIoR for short. We find for example that the current leader on HELM may change by merely removing a low-ranked model from the benchmark and observe that a handful of examples suffice to obtain the correct benchmark ranking. Conversely a slightly different choice of HELM scenarios varies ranking widely. Based on our findings we outline a set of concrete recommendations for more efficient benchmark design and utilization practices leading to dramatic cost savings with minimal loss of benchmark reliability often reducing computation by x100 or more.

{{</citation>}}


### (23/113) SeamlessM4T-Massively Multilingual & Multimodal Machine Translation (Seamless Communication et al., 2023)

{{<citation>}}

Seamless Communication, Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, Christopher Klaiber, Pengwei Li, Daniel Licht, Jean Maillard, Alice Rakotoarison, Kaushik Ram Sadagopan, Guillaume Wenzek, Ethan Ye, Bapi Akula, Peng-Jen Chen, Naji El Hachem, Brian Ellis, Gabriel Mejia Gonzalez, Justin Haaheim, Prangthip Hansanti, Russ Howes, Bernie Huang, Min-Jae Hwang, Hirofumi Inaguma, Somya Jain, Elahe Kalbassi, Amanda Kallet, Ilia Kulikov, Janice Lam, Daniel Li, Xutai Ma, Ruslan Mavlyutov, Benjamin Peloquin, Mohamed Ramadan, Abinesh Ramakrishnan, Anna Sun, Kevin Tran, Tuan Tran, Igor Tufanov, Vish Vogeti, Carleigh Wood, Yilin Yang, Bokai Yu, Pierre Andrews, Can Balioglu, Marta R. Costa-jussà, Onur Celebi, Maha Elbayad, Cynthia Gao, Francisco Guzmán, Justine Kao, Ann Lee, Alexandre Mourachko, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang. (2023)  
**SeamlessM4T-Massively Multilingual & Multimodal Machine Translation**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: BERT, BLEU, Machine Translation, Multilingual  
[Paper Link](http://arxiv.org/abs/2308.11596v2)  

---


**ABSTRACT**  
What does it take to create the Babel Fish, a tool that can help individuals translate speech between any two languages? While recent breakthroughs in text-based models have pushed machine translation coverage beyond 200 languages, unified speech-to-speech translation models have yet to achieve similar strides. More specifically, conventional speech-to-speech translation systems rely on cascaded systems that perform translation progressively, putting high-performing unified systems out of reach. To address these gaps, we introduce SeamlessM4T, a single model that supports speech-to-speech translation, speech-to-text translation, text-to-speech translation, text-to-text translation, and automatic speech recognition for up to 100 languages. To build this, we used 1 million hours of open speech audio data to learn self-supervised speech representations with w2v-BERT 2.0. Subsequently, we created a multimodal corpus of automatically aligned speech translations. Filtered and combined with human-labeled and pseudo-labeled data, we developed the first multilingual system capable of translating from and into English for both speech and text. On FLEURS, SeamlessM4T sets a new standard for translations into multiple target languages, achieving an improvement of 20% BLEU over the previous SOTA in direct speech-to-text translation. Compared to strong cascaded models, SeamlessM4T improves the quality of into-English translation by 1.3 BLEU points in speech-to-text and by 2.6 ASR-BLEU points in speech-to-speech. Tested for robustness, our system performs better against background noises and speaker variations in speech-to-text tasks compared to the current SOTA model. Critically, we evaluated SeamlessM4T on gender bias and added toxicity to assess translation safety. Finally, all contributions in this work are open-sourced and accessible at https://github.com/facebookresearch/seamless_communication

{{</citation>}}


### (24/113) Using ChatGPT as a CAT tool in Easy Language translation (Silvana Deilen et al., 2023)

{{<citation>}}

Silvana Deilen, Sergio Hernández Garrido, Ekaterina Lapshinova-Koltunski, Christiane Maaß. (2023)  
**Using ChatGPT as a CAT tool in Easy Language translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.11563v1)  

---


**ABSTRACT**  
This study sets out to investigate the feasibility of using ChatGPT to translate citizen-oriented administrative texts into German Easy Language, a simplified, controlled language variety that is adapted to the needs of people with reading impairments. We use ChatGPT to translate selected texts from websites of German public authorities using two strategies, i.e. linguistic and holistic. We analyse the quality of the generated texts based on different criteria, such as correctness, readability, and syntactic complexity. The results indicated that the generated texts are easier than the standard texts, but that they still do not fully meet the established Easy Language standards. Additionally, the content is not always rendered correctly.

{{</citation>}}


### (25/113) Empowering Refugee Claimants and their Lawyers: Using Machine Learning to Examine Decision-Making in Refugee Law (Claire Barale, 2023)

{{<citation>}}

Claire Barale. (2023)  
**Empowering Refugee Claimants and their Lawyers: Using Machine Learning to Examine Decision-Making in Refugee Law**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.11531v1)  

---


**ABSTRACT**  
Our project aims at helping and supporting stakeholders in refugee status adjudications, such as lawyers, judges, governing bodies, and claimants, in order to make better decisions through data-driven intelligence and increase the understanding and transparency of the refugee application process for all involved parties. This PhD project has two primary objectives: (1) to retrieve past cases, and (2) to analyze legal decision-making processes on a dataset of Canadian cases. In this paper, we present the current state of our work, which includes a completed experiment on part (1) and ongoing efforts related to part (2). We believe that NLP-based solutions are well-suited to address these challenges, and we investigate the feasibility of automating all steps involved. In addition, we introduce a novel benchmark for future NLP research in refugee law. Our methodology aims to be inclusive to all end-users and stakeholders, with expected benefits including reduced time-to-decision, fairer and more transparent outcomes, and improved decision quality.

{{</citation>}}


### (26/113) Can Authorship Representation Learning Capture Stylistic Features? (Andrew Wang et al., 2023)

{{<citation>}}

Andrew Wang, Cristina Aggazzotti, Rebecca Kotula, Rafael Rivera Soto, Marcus Bishop, Nicholas Andrews. (2023)  
**Can Authorship Representation Learning Capture Stylistic Features?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.11490v2)  

---


**ABSTRACT**  
Automatically disentangling an author's style from the content of their writing is a longstanding and possibly insurmountable problem in computational linguistics. At the same time, the availability of large text corpora furnished with author labels has recently enabled learning authorship representations in a purely data-driven manner for authorship attribution, a task that ostensibly depends to a greater extent on encoding writing style than encoding content. However, success on this surrogate task does not ensure that such representations capture writing style since authorship could also be correlated with other latent variables, such as topic. In an effort to better understand the nature of the information these representations convey, and specifically to validate the hypothesis that they chiefly encode writing style, we systematically probe these representations through a series of targeted experiments. The results of these experiments suggest that representations learned for the surrogate authorship prediction task are indeed sensitive to writing style. As a consequence, authorship representations may be expected to be robust to certain kinds of data shift, such as topic drift over time. Additionally, our findings may open the door to downstream applications that require stylistic representations, such as style transfer.

{{</citation>}}


### (27/113) Learning to generate and corr- uh I mean repair language in real-time (Arash Eshghi et al., 2023)

{{<citation>}}

Arash Eshghi, Arash Ashrafzadeh. (2023)  
**Learning to generate and corr- uh I mean repair language in real-time**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11683v1)  

---


**ABSTRACT**  
In conversation, speakers produce language incrementally, word by word, while continuously monitoring the appropriateness of their own contribution in the dynamically unfolding context of the conversation; and this often leads them to repair their own utterance on the fly. This real-time language processing capacity is furthermore crucial to the development of fluent and natural conversational AI. In this paper, we use a previously learned Dynamic Syntax grammar and the CHILDES corpus to develop, train and evaluate a probabilistic model for incremental generation where input to the model is a purely semantic generation goal concept in Type Theory with Records (TTR). We show that the model's output exactly matches the gold candidate in 78% of cases with a ROUGE-l score of 0.86. We further do a zero-shot evaluation of the ability of the same model to generate self-repairs when the generation goal changes mid-utterance. Automatic evaluation shows that the model can generate self-repairs correctly in 85% of cases. A small human evaluation confirms the naturalness and grammaticality of the generated self-repairs. Overall, these results further highlight the generalisation power of grammar-based models and lay the foundations for more controllable, and naturally interactive conversational AI systems.

{{</citation>}}


### (28/113) Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions (Pouya Pezeshkpour et al., 2023)

{{<citation>}}

Pouya Pezeshkpour, Estevam Hruschka. (2023)  
**Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.11483v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable capabilities in various NLP tasks. However, previous works have shown these models are sensitive towards prompt wording, and few-shot demonstrations and their order, posing challenges to fair assessment of these models. As these models become more powerful, it becomes imperative to understand and address these limitations. In this paper, we focus on LLMs robustness on the task of multiple-choice questions -- commonly adopted task to study reasoning and fact-retrieving capability of LLMs. Investigating the sensitivity of LLMs towards the order of options in multiple-choice questions, we demonstrate a considerable performance gap of approximately 13% to 75% in LLMs on different benchmarks, when answer options are reordered, even when using demonstrations in a few-shot setting. Through a detailed analysis, we conjecture that this sensitivity arises when LLMs are uncertain about the prediction between the top-2/3 choices, and specific options placements may favor certain prediction between those top choices depending on the question caused by positional bias. We also identify patterns in top-2 choices that amplify or mitigate the model's bias toward option placement. We found that for amplifying bias, the optimal strategy involves positioning the top two choices as the first and last options. Conversely, to mitigate bias, we recommend placing these choices among the adjacent options. To validate our conjecture, we conduct various experiments and adopt two approaches to calibrate LLMs' predictions, leading to up to 8 percentage points improvement across different models and benchmarks.

{{</citation>}}


### (29/113) Aspect-oriented Opinion Alignment Network for Aspect-Based Sentiment Classification (Xueyi Liu et al., 2023)

{{<citation>}}

Xueyi Liu, Rui Hou, Yanglei Gan, Da Luo, Changlin Li, Xiaojun Shi, Qiao Liu. (2023)  
**Aspect-oriented Opinion Alignment Network for Aspect-Based Sentiment Classification**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.11447v1)  

---


**ABSTRACT**  
Aspect-based sentiment classification is a crucial problem in fine-grained sentiment analysis, which aims to predict the sentiment polarity of the given aspect according to its context. Previous works have made remarkable progress in leveraging attention mechanism to extract opinion words for different aspects. However, a persistent challenge is the effective management of semantic mismatches, which stem from attention mechanisms that fall short in adequately aligning opinions words with their corresponding aspect in multi-aspect sentences. To address this issue, we propose a novel Aspect-oriented Opinion Alignment Network (AOAN) to capture the contextual association between opinion words and the corresponding aspect. Specifically, we first introduce a neighboring span enhanced module which highlights various compositions of neighboring words and given aspects. In addition, we design a multi-perspective attention mechanism that align relevant opinion information with respect to the given aspect. Extensive experiments on three benchmark datasets demonstrate that our model achieves state-of-the-art results. The source code is available at https://github.com/AONE-NLP/ABSA-AOAN.

{{</citation>}}


### (30/113) Extracting Relational Triples Based on Graph Recursive Neural Network via Dynamic Feedback Forest Algorithm (Hongyin Zhu, 2023)

{{<citation>}}

Hongyin Zhu. (2023)  
**Extracting Relational Triples Based on Graph Recursive Neural Network via Dynamic Feedback Forest Algorithm**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2308.11411v1)  

---


**ABSTRACT**  
Extracting relational triples (subject, predicate, object) from text enables the transformation of unstructured text data into structured knowledge. The named entity recognition (NER) and the relation extraction (RE) are two foundational subtasks in this knowledge generation pipeline. The integration of subtasks poses a considerable challenge due to their disparate nature. This paper presents a novel approach that converts the triple extraction task into a graph labeling problem, capitalizing on the structural information of dependency parsing and graph recursive neural networks (GRNNs). To integrate subtasks, this paper proposes a dynamic feedback forest algorithm that connects the representations of subtasks by inference operations during model training. Experimental results demonstrate the effectiveness of the proposed method.

{{</citation>}}


### (31/113) HopPG: Self-Iterative Program Generation for Multi-Hop Question Answering over Heterogeneous Knowledge (Yingyao Wang et al., 2023)

{{<citation>}}

Yingyao Wang, Yongwei Zhou, Chaoqun Duan, Junwei Bao, Tiejun Zhao. (2023)  
**HopPG: Self-Iterative Program Generation for Multi-Hop Question Answering over Heterogeneous Knowledge**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.11257v1)  

---


**ABSTRACT**  
The semantic parsing-based method is an important research branch for knowledge-based question answering. It usually generates executable programs lean upon the question and then conduct them to reason answers over a knowledge base. Benefit from this inherent mechanism, it has advantages in the performance and the interpretability. However,traditional semantic parsing methods usually generate a complete program before executing it, which struggles with multi-hop question answering over heterogeneous knowledge. Firstly,a complete multi-hop program relies on multiple heterogeneous supporting facts, and it is difficult for models to receive these facts simultaneously. Secondly,these methods ignore the interaction information between the previous-hop execution result and the current-hop program generation. To alleviate these challenges, we propose a self-iterative framework for multi-hop program generation (HopPG) over heterogeneous knowledge, which leverages the previous-hop execution results to retrieve supporting facts and generate subsequent programs iteratively. We evaluate our model on MMQA-T^2. The experimental results show that HopPG outperforms existing semantic-parsing-based baselines, especially on the multi-hop questions.

{{</citation>}}


### (32/113) Diversity Measures: Domain-Independent Proxies for Failure in Language Model Queries (Noel Ngu et al., 2023)

{{<citation>}}

Noel Ngu, Nathaniel Lee, Paulo Shakarian. (2023)  
**Diversity Measures: Domain-Independent Proxies for Failure in Language Model Queries**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.11189v1)  

---


**ABSTRACT**  
Error prediction in large language models often relies on domain-specific information. In this paper, we present measures for quantification of error in the response of a large language model based on the diversity of responses to a given prompt - hence independent of the underlying application. We describe how three such measures - based on entropy, Gini impurity, and centroid distance - can be employed. We perform a suite of experiments on multiple datasets and temperature settings to demonstrate that these measures strongly correlate with the probability of failure. Additionally, we present empirical results demonstrating how these measures can be applied to few-shot prompting, chain-of-thought reasoning, and error detection.

{{</citation>}}


### (33/113) Anonymity at Risk? Assessing Re-Identification Capabilities of Large Language Models (Alex Nyffenegger et al., 2023)

{{<citation>}}

Alex Nyffenegger, Matthias Stürmer, Joel Niklaus. (2023)  
**Anonymity at Risk? Assessing Re-Identification Capabilities of Large Language Models**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2, cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.11103v1)  

---


**ABSTRACT**  
Anonymity of both natural and legal persons in court rulings is a critical aspect of privacy protection in the European Union and Switzerland. With the advent of LLMs, concerns about large-scale re-identification of anonymized persons are growing. In accordance with the Federal Supreme Court of Switzerland, we explore the potential of LLMs to re-identify individuals in court rulings by constructing a proof-of-concept using actual legal data from the Swiss federal supreme court. Following the initial experiment, we constructed an anonymized Wikipedia dataset as a more rigorous testing ground to further investigate the findings. With the introduction and application of the new task of re-identifying people in texts, we also introduce new metrics to measure performance. We systematically analyze the factors that influence successful re-identifications, identifying model size, input length, and instruction tuning among the most critical determinants. Despite high re-identification rates on Wikipedia, even the best LLMs struggled with court decisions. The complexity is attributed to the lack of test datasets, the necessity for substantial training resources, and data sparsity in the information used for re-identification. In conclusion, this study demonstrates that re-identification using LLMs may not be feasible for now, but as the proof-of-concept on Wikipedia showed, it might become possible in the future. We hope that our system can help enhance the confidence in the security of anonymized decisions, thus leading to the courts being more confident to publish decisions.

{{</citation>}}


## cs.AR (1)



### (34/113) Accel-GCN: High-Performance GPU Accelerator Design for Graph Convolution Networks (Xi Xie et al., 2023)

{{<citation>}}

Xi Xie, Hongwu Peng, Amit Hasan, Shaoyi Huang, Jiahui Zhao, Haowen Fang, Wei Zhang, Tong Geng, Omer Khan, Caiwen Ding. (2023)  
**Accel-GCN: High-Performance GPU Accelerator Design for Graph Convolution Networks**  

---
Primary Category: cs.AR  
Categories: I-2; B-6; C-3, cs-AR, cs-LG, cs.AR  
Keywords: GNN, Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2308.11825v1)  

---


**ABSTRACT**  
Graph Convolutional Networks (GCNs) are pivotal in extracting latent information from graph data across various domains, yet their acceleration on mainstream GPUs is challenged by workload imbalance and memory access irregularity. To address these challenges, we present Accel-GCN, a GPU accelerator architecture for GCNs. The design of Accel-GCN encompasses: (i) a lightweight degree sorting stage to group nodes with similar degree; (ii) a block-level partition strategy that dynamically adjusts warp workload sizes, enhancing shared memory locality and workload balance, and reducing metadata overhead compared to designs like GNNAdvisor; (iii) a combined warp strategy that improves memory coalescing and computational parallelism in the column dimension of dense matrices.   Utilizing these principles, we formulated a kernel for sparse matrix multiplication (SpMM) in GCNs that employs block-level partitioning and combined warp strategy. This approach augments performance and multi-level memory efficiency and optimizes memory bandwidth by exploiting memory coalescing and alignment. Evaluation of Accel-GCN across 18 benchmark graphs reveals that it outperforms cuSPARSE, GNNAdvisor, and graph-BLAST by factors of 1.17 times, 1.86 times, and 2.94 times respectively. The results underscore Accel-GCN as an effective solution for enhancing GCN computational efficiency.

{{</citation>}}


## cs.CR (4)



### (35/113) Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings (Eugene Bagdasaryan et al., 2023)

{{<citation>}}

Eugene Bagdasaryan, Vitaly Shmatikov. (2023)  
**Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.11804v1)  

---


**ABSTRACT**  
Multi-modal encoders map images, sounds, texts, videos, etc. into a single embedding space, aligning representations across modalities (e.g., associate an image of a dog with a barking sound). We show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an input in any modality, an adversary can perturb it so as to make its embedding close to that of an arbitrary, adversary-chosen input in another modality. Illusions thus enable the adversary to align any image with any text, any text with any sound, etc.   Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks. Using ImageBind embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, and zero-shot classification.

{{</citation>}}


### (36/113) Multi-Instance Adversarial Attack on GNN-Based Malicious Domain Detection (Mahmoud Nazzal et al., 2023)

{{<citation>}}

Mahmoud Nazzal, Issa Khalil, Abdallah Khreishah, NhatHai Phan, Yao Ma. (2023)  
**Multi-Instance Adversarial Attack on GNN-Based Malicious Domain Detection**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Adversarial Attack, GNN  
[Paper Link](http://arxiv.org/abs/2308.11754v1)  

---


**ABSTRACT**  
Malicious domain detection (MDD) is an open security challenge that aims to detect if an Internet domain is associated with cyber-attacks. Among many approaches to this problem, graph neural networks (GNNs) are deemed highly effective. GNN-based MDD uses DNS logs to represent Internet domains as nodes in a maliciousness graph (DMG) and trains a GNN to infer their maliciousness by leveraging identified malicious domains. Since this method relies on accessible DNS logs to construct DMGs, it exposes a vulnerability for adversaries to manipulate their domain nodes' features and connections within DMGs. Existing research mainly concentrates on threat models that manipulate individual attacker nodes. However, adversaries commonly generate multiple domains to achieve their goals economically and avoid detection. Their objective is to evade discovery across as many domains as feasible. In this work, we call the attack that manipulates several nodes in the DMG concurrently a multi-instance evasion attack. We present theoretical and empirical evidence that the existing single-instance evasion techniques for are inadequate to launch multi-instance evasion attacks against GNN-based MDDs. Therefore, we introduce MintA, an inference-time multi-instance adversarial attack on GNN-based MDDs. MintA enhances node and neighborhood evasiveness through optimized perturbations and operates successfully with only black-box access to the target model, eliminating the need for knowledge about the model's specifics or non-adversary nodes. We formulate an optimization challenge for MintA, achieving an approximate solution. Evaluating MintA on a leading GNN-based MDD technique with real-world data showcases an attack success rate exceeding 80%. These findings act as a warning for security experts, underscoring GNN-based MDDs' susceptibility to practical attacks that can undermine their effectiveness and benefits.

{{</citation>}}


### (37/113) Up-to-date Threat Modelling for Soft Privacy on Smart Cars (Mario Raciti et al., 2023)

{{<citation>}}

Mario Raciti, Giampaolo Bella. (2023)  
**Up-to-date Threat Modelling for Soft Privacy on Smart Cars**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Smart Car  
[Paper Link](http://arxiv.org/abs/2308.11273v1)  

---


**ABSTRACT**  
Physical persons playing the role of car drivers consume data that is sourced from the Internet and, at the same time, themselves act as sources of relevant data. It follows that citizens' privacy is potentially at risk while they drive, hence the need to model privacy threats in this application domain. This paper addresses the privacy threats by updating a recent threat-modelling methodology and by tailoring it specifically to the soft privacy target property, which ensures citizens' full control on their personal data. The methodology now features the sources of documentation as an explicit variable that is to be considered. It is demonstrated by including a new version of the de-facto standard LINDDUN methodology as well as an additional source by ENISA which is found to be relevant to soft privacy. The main findings are a set of 23 domain-independent threats, 43 domain-specific assets and 525 domain-dependent threats for the target property in the automotive domain. While these exceed their previous versions, their main value is to offer self-evident support to at least two arguments. One is that LINDDUN has evolved much the way our original methodology already advocated because a few of our previously suggested extensions are no longer outstanding. The other one is that ENISA's treatment of privacy aboard smart cars should be extended considerably because our 525 threats fall in the same scope.

{{</citation>}}


### (38/113) Adaptive White-Box Watermarking with Self-Mutual Check Parameters in Deep Neural Networks (Zhenzhe Gao et al., 2023)

{{<citation>}}

Zhenzhe Gao, Zhaoxia Yin, Hongjian Zhan, Heng Yin, Yue Lu. (2023)  
**Adaptive White-Box Watermarking with Self-Mutual Check Parameters in Deep Neural Networks**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11235v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) has found wide application, but also poses risks due to unintentional or malicious tampering during deployment. Regular checks are therefore necessary to detect and prevent such risks. Fragile watermarking is a technique used to identify tampering in AI models. However, previous methods have faced challenges including risks of omission, additional information transmission, and inability to locate tampering precisely. In this paper, we propose a method for detecting tampered parameters and bits, which can be used to detect, locate, and restore parameters that have been tampered with. We also propose an adaptive embedding method that maximizes information capacity while maintaining model accuracy. Our approach was tested on multiple neural networks subjected to attacks that modified weight parameters, and our results demonstrate that our method achieved great recovery performance when the modification rate was below 20%. Furthermore, for models where watermarking significantly affected accuracy, we utilized an adaptive bit technique to recover more than 15% of the accuracy loss of the model.

{{</citation>}}


## cs.SD (4)



### (39/113) Complex-valued neural networks for voice anti-spoofing (Nicolas M. Müller et al., 2023)

{{<citation>}}

Nicolas M. Müller, Philip Sperl, Konstantin Böttinger. (2023)  
**Complex-valued neural networks for voice anti-spoofing**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11800v1)  

---


**ABSTRACT**  
Current anti-spoofing and audio deepfake detection systems use either magnitude spectrogram-based features (such as CQT or Melspectrograms) or raw audio processed through convolution or sinc-layers. Both methods have drawbacks: magnitude spectrograms discard phase information, which affects audio naturalness, and raw-feature-based models cannot use traditional explainable AI methods. This paper proposes a new approach that combines the benefits of both methods by using complex-valued neural networks to process the complex-valued, CQT frequency-domain representation of the input audio. This method retains phase information and allows for explainable AI methods. Results show that this approach outperforms previous methods on the "In-the-Wild" anti-spoofing dataset and enables interpretation of the results through explainable AI. Ablation studies confirm that the model has learned to use phase information to detect voice spoofing.

{{</citation>}}


### (40/113) Furnishing Sound Event Detection with Language Model Abilities (Hualei Wang et al., 2023)

{{<citation>}}

Hualei Wang, Jianguo Mao, Zhifang Guo, Jiarui Wan, Hong Liu, Xiangdong Wang. (2023)  
**Furnishing Sound Event Detection with Language Model Abilities**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Event Detection, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11530v1)  

---


**ABSTRACT**  
Recently, the ability of language models (LMs) has attracted increasing attention in visual cross-modality. In this paper, we further explore the generation capacity of LMs for sound event detection (SED), beyond the visual domain. Specifically, we propose an elegant method that aligns audio features and text features to accomplish sound event classification and temporal location. The framework consists of an acoustic encoder, a contrastive module that align the corresponding representations of the text and audio, and a decoupled language decoder that generates temporal and event sequences from the audio characteristic. Compared with conventional works that require complicated processing and barely utilize limited audio features, our model is more concise and comprehensive since language model directly leverage its semantic capabilities to generate the sequences. We investigate different decoupling modules to demonstrate the effectiveness for timestamps capture and event classification. Evaluation results show that the proposed method achieves accurate sequences of sound event detection.

{{</citation>}}


### (41/113) Music Understanding LLaMA: Advancing Text-to-Music Generation with Question Answering and Captioning (Shansong Liu et al., 2023)

{{<citation>}}

Shansong Liu, Atin Sakkeer Hussain, Chenshuo Sun, Ying Shan. (2023)  
**Music Understanding LLaMA: Advancing Text-to-Music Generation with Question Answering and Captioning**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CL, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: LLaMA, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.11276v1)  

---


**ABSTRACT**  
Text-to-music generation (T2M-Gen) faces a major obstacle due to the scarcity of large-scale publicly available music datasets with natural language captions. To address this, we propose the Music Understanding LLaMA (MU-LLaMA), capable of answering music-related questions and generating captions for music files. Our model utilizes audio representations from a pretrained MERT model to extract music features. However, obtaining a suitable dataset for training the MU-LLaMA model remains challenging, as existing publicly accessible audio question answering datasets lack the necessary depth for open-ended music question answering. To fill this gap, we present a methodology for generating question-answer pairs from existing audio captioning datasets and introduce the MusicQA Dataset designed for answering open-ended music-related questions. The experiments demonstrate that the proposed MU-LLaMA model, trained on our designed MusicQA dataset, achieves outstanding performance in both music question answering and music caption generation across various metrics, outperforming current state-of-the-art (SOTA) models in both fields and offering a promising advancement in the T2M-Gen research field.

{{</citation>}}


### (42/113) An Effective Transformer-based Contextual Model and Temporal Gate Pooling for Speaker Identification (Harunori Kawano et al., 2023)

{{<citation>}}

Harunori Kawano, Sota Shimizu. (2023)  
**An Effective Transformer-based Contextual Model and Temporal Gate Pooling for Speaker Identification**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.11241v1)  

---


**ABSTRACT**  
Wav2vec2 has achieved success in applying Transformer architecture and self-supervised learning to speech recognition. Recently, these have come to be used not only for speech recognition but also for the entire speech processing. This paper introduces an effective end-to-end speaker identification model applied Transformer-based contextual model. We explored the relationship between the parameters and the performance in order to discern the structure of an effective model. Furthermore, we propose a pooling method, Temporal Gate Pooling, with powerful learning ability for speaker identification. We applied Conformer as encoder and BEST-RQ for pre-training and conducted an evaluation utilizing the speaker identification of VoxCeleb1. The proposed method has achieved an accuracy of 85.9% with 28.5M parameters, demonstrating comparable precision to wav2vec2 with 317.7M parameters. Code is available at https://github.com/HarunoriKawano/speaker-identification-with-tgp.

{{</citation>}}


## cs.CV (33)



### (43/113) Time Does Tell: Self-Supervised Time-Tuning of Dense Image Representations (Mohammadreza Salehi et al., 2023)

{{<citation>}}

Mohammadreza Salehi, Efstratios Gavves, Cees G. M. Snoek, Yuki M. Asano. (2023)  
**Time Does Tell: Self-Supervised Time-Tuning of Dense Image Representations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.11796v1)  

---


**ABSTRACT**  
Spatially dense self-supervised learning is a rapidly growing problem domain with promising applications for unsupervised segmentation and pretraining for dense downstream tasks. Despite the abundance of temporal data in the form of videos, this information-rich source has been largely overlooked. Our paper aims to address this gap by proposing a novel approach that incorporates temporal consistency in dense self-supervised learning. While methods designed solely for images face difficulties in achieving even the same performance on videos, our method improves not only the representation quality for videos-but also images. Our approach, which we call time-tuning, starts from image-pretrained models and fine-tunes them with a novel self-supervised temporal-alignment clustering loss on unlabeled videos. This effectively facilitates the transfer of high-level information from videos to image representations. Time-tuning improves the state-of-the-art by 8-10% for unsupervised semantic segmentation on videos and matches it for images. We believe this method paves the way for further self-supervised scaling by leveraging the abundant availability of videos. The implementation can be found here : https://github.com/SMSD75/Timetuning

{{</citation>}}


### (44/113) Enhancing NeRF akin to Enhancing LLMs: Generalizable NeRF Transformer with Mixture-of-View-Experts (Wenyan Cong et al., 2023)

{{<citation>}}

Wenyan Cong, Hanxue Liang, Peihao Wang, Zhiwen Fan, Tianlong Chen, Mukund Varma, Yi Wang, Zhangyang Wang. (2023)  
**Enhancing NeRF akin to Enhancing LLMs: Generalizable NeRF Transformer with Mixture-of-View-Experts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.11793v1)  

---


**ABSTRACT**  
Cross-scene generalizable NeRF models, which can directly synthesize novel views of unseen scenes, have become a new spotlight of the NeRF field. Several existing attempts rely on increasingly end-to-end "neuralized" architectures, i.e., replacing scene representation and/or rendering modules with performant neural networks such as transformers, and turning novel view synthesis into a feed-forward inference pipeline. While those feedforward "neuralized" architectures still do not fit diverse scenes well out of the box, we propose to bridge them with the powerful Mixture-of-Experts (MoE) idea from large language models (LLMs), which has demonstrated superior generalization ability by balancing between larger overall model capacity and flexible per-instance specialization. Starting from a recent generalizable NeRF architecture called GNT, we first demonstrate that MoE can be neatly plugged in to enhance the model. We further customize a shared permanent expert and a geometry-aware consistency loss to enforce cross-scene consistency and spatial smoothness respectively, which are essential for generalizable view synthesis. Our proposed model, dubbed GNT with Mixture-of-View-Experts (GNT-MOVE), has experimentally shown state-of-the-art results when transferring to unseen scenes, indicating remarkably better cross-scene generalization in both zero-shot and few-shot settings. Our codes are available at https://github.com/VITA-Group/GNT-MOVE.

{{</citation>}}


### (45/113) An extensible point-based method for data chart value detection (Carlos Soto et al., 2023)

{{<citation>}}

Carlos Soto, Shinjae Yoo. (2023)  
**An extensible point-based method for data chart value detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.11788v1)  

---


**ABSTRACT**  
We present an extensible method for identifying semantic points to reverse engineer (i.e. extract the values of) data charts, particularly those in scientific articles. Our method uses a point proposal network (akin to region proposal networks for object detection) to directly predict the position of points of interest in a chart, and it is readily extensible to multiple chart types and chart elements. We focus on complex bar charts in the scientific literature, on which our model is able to detect salient points with an accuracy of 0.8705 F1 (@1.5-cell max deviation); it achieves 0.9810 F1 on synthetically-generated charts similar to those used in prior works. We also explore training exclusively on synthetic data with novel augmentations, reaching surprisingly competent performance in this way (0.6621 F1) on real charts with widely varying appearance, and we further demonstrate our unchanged method applied directly to synthetic pie charts (0.8343 F1). Datasets, trained models, and evaluation code are available at https://github.com/BNLNLP/PPN_model.

{{</citation>}}


### (46/113) Coarse-to-Fine Multi-Scene Pose Regression with Transformers (Yoli Shavit et al., 2023)

{{<citation>}}

Yoli Shavit, Ron Ferens, Yosi Keller. (2023)  
**Coarse-to-Fine Multi-Scene Pose Regression with Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.11783v1)  

---


**ABSTRACT**  
Absolute camera pose regressors estimate the position and orientation of a camera given the captured image alone. Typically, a convolutional backbone with a multi-layer perceptron (MLP) head is trained using images and pose labels to embed a single reference scene at a time. Recently, this scheme was extended to learn multiple scenes by replacing the MLP head with a set of fully connected layers. In this work, we propose to learn multi-scene absolute camera pose regression with Transformers, where encoders are used to aggregate activation maps with self-attention and decoders transform latent features and scenes encoding into pose predictions. This allows our model to focus on general features that are informative for localization, while embedding multiple scenes in parallel. We extend our previous MS-Transformer approach \cite{shavit2021learning} by introducing a mixed classification-regression architecture that improves the localization accuracy. Our method is evaluated on commonly benchmark indoor and outdoor datasets and has been shown to exceed both multi-scene and state-of-the-art single-scene absolute pose regressors.

{{</citation>}}


### (47/113) 3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network (Qinyu Chen et al., 2023)

{{<citation>}}

Qinyu Chen, Zuowen Wang, Shih-Chii Liu, Chang Gao. (2023)  
**3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network**  

---
Primary Category: cs.CV  
Categories: 68T07, 68T10, cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2308.11771v1)  

---


**ABSTRACT**  
This paper presents a sparse Change-Based Convolutional Long Short-Term Memory (CB-ConvLSTM) model for event-based eye tracking, key for next-generation wearable healthcare technology such as AR/VR headsets. We leverage the benefits of retina-inspired event cameras, namely their low-latency response and sparse output event stream, over traditional frame-based cameras. Our CB-ConvLSTM architecture efficiently extracts spatio-temporal features for pupil tracking from the event stream, outperforming conventional CNN structures. Utilizing a delta-encoded recurrent path enhancing activation sparsity, CB-ConvLSTM reduces arithmetic operations by approximately 4.7$\times$ without losing accuracy when tested on a \texttt{v2e}-generated event dataset of labeled pupils. This increase in efficiency makes it ideal for real-time eye tracking in resource-constrained devices. The project code and dataset are openly available at \url{https://github.com/qinche106/cb-convlstm-eyetracking}.

{{</citation>}}


### (48/113) Efficient Controllable Multi-Task Architectures (Abhishek Aich et al., 2023)

{{<citation>}}

Abhishek Aich, Samuel Schulter, Amit K. Roy-Chowdhury, Manmohan Chandraker, Yumin Suh. (2023)  
**Efficient Controllable Multi-Task Architectures**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.11744v1)  

---


**ABSTRACT**  
We aim to train a multi-task model such that users can adjust the desired compute budget and relative importance of task performances after deployment, without retraining. This enables optimizing performance for dynamically varying user needs, without heavy computational overhead to train and save models for various scenarios. To this end, we propose a multi-task model consisting of a shared encoder and task-specific decoders where both encoder and decoder channel widths are slimmable. Our key idea is to control the task importance by varying the capacities of task-specific decoders, while controlling the total computational cost by jointly adjusting the encoder capacity. This improves overall accuracy by allowing a stronger encoder for a given budget, increases control over computational cost, and delivers high-quality slimmed sub-architectures based on user's constraints. Our training strategy involves a novel 'Configuration-Invariant Knowledge Distillation' loss that enforces backbone representations to be invariant under different runtime width configurations to enhance accuracy. Further, we present a simple but effective search algorithm that translates user constraints to runtime width configurations of both the shared encoder and task decoders, for sampling the sub-architectures. The key rule for the search algorithm is to provide a larger computational budget to the higher preferred task decoder, while searching a shared encoder configuration that enhances the overall MTL performance. Various experiments on three multi-task benchmarks (PASCALContext, NYUDv2, and CIFAR100-MTL) with diverse backbone architectures demonstrate the advantage of our approach. For example, our method shows a higher controllability by ~33.5% in the NYUD-v2 dataset over prior methods, while incurring much less compute cost.

{{</citation>}}


### (49/113) GOPro: Generate and Optimize Prompts in CLIP using Self-Supervised Learning (Mainak Singha et al., 2023)

{{<citation>}}

Mainak Singha, Ankit Jha, Biplab Banerjee. (2023)  
**GOPro: Generate and Optimize Prompts in CLIP using Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.11605v1)  

---


**ABSTRACT**  
Large-scale foundation models, such as CLIP, have demonstrated remarkable success in visual recognition tasks by embedding images in a semantically rich space. Self-supervised learning (SSL) has also shown promise in improving visual recognition by learning invariant features. However, the combination of CLIP with SSL is found to face challenges due to the multi-task framework that blends CLIP's contrastive loss and SSL's loss, including difficulties with loss weighting and inconsistency among different views of images in CLIP's output space. To overcome these challenges, we propose a prompt learning-based model called GOPro, which is a unified framework that ensures similarity between various augmented views of input images in a shared image-text embedding space, using a pair of learnable image and text projectors atop CLIP, to promote invariance and generalizability. To automatically learn such prompts, we leverage the visual content and style primitives extracted from pre-trained CLIP and adapt them to the target task. In addition to CLIP's cross-domain contrastive loss, we introduce a visual contrastive loss and a novel prompt consistency loss, considering the different views of the images. GOPro is trained end-to-end on all three loss objectives, combining the strengths of CLIP and SSL in a principled manner. Empirical evaluations demonstrate that GOPro outperforms the state-of-the-art prompting techniques on three challenging domain generalization tasks across multiple benchmarks by a significant margin. Our code is available at https://github.com/mainaksingha01/GOPro.

{{</citation>}}


### (50/113) Target-Grounded Graph-Aware Transformer for Aerial Vision-and-Dialog Navigation (Yifei Su et al., 2023)

{{<citation>}}

Yifei Su, Dong An, Yuan Xu, Kehan Chen, Yan Huang. (2023)  
**Target-Grounded Graph-Aware Transformer for Aerial Vision-and-Dialog Navigation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Dialog, Transformer  
[Paper Link](http://arxiv.org/abs/2308.11561v3)  

---


**ABSTRACT**  
This report details the methods of the winning entry of the AVDN Challenge in ICCV CLVL 2023. The competition addresses the Aerial Navigation from Dialog History (ANDH) task, which requires a drone agent to associate dialog history with aerial observations to reach the destination. For better cross-modal grounding abilities of the drone agent, we propose a Target-Grounded Graph-Aware Transformer (TG-GAT) framework. Concretely, TG-GAT first leverages a graph-aware transformer to capture spatiotemporal dependency, which benefits navigation state tracking and robust action planning. In addition,an auxiliary visual grounding task is devised to boost the agent's awareness of referred landmarks. Moreover, a hybrid augmentation strategy based on large language models is utilized to mitigate data scarcity limitations. Our TG-GAT framework won the AVDN Challenge, with 2.2% and 3.0% absolute improvements over the baseline on SPL and SR metrics, respectively. The code is available at https://github.com/yifeisu/TG-GAT.

{{</citation>}}


### (51/113) SwinFace: A Multi-task Transformer for Face Recognition, Expression Recognition, Age Estimation and Attribute Estimation (Lixiong Qin et al., 2023)

{{<citation>}}

Lixiong Qin, Mei Wang, Chao Deng, Ke Wang, Xi Chen, Jiani Hu, Weihong Deng. (2023)  
**SwinFace: A Multi-task Transformer for Face Recognition, Expression Recognition, Age Estimation and Attribute Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.11509v1)  

---


**ABSTRACT**  
In recent years, vision transformers have been introduced into face recognition and analysis and have achieved performance breakthroughs. However, most previous methods generally train a single model or an ensemble of models to perform the desired task, which ignores the synergy among different tasks and fails to achieve improved prediction accuracy, increased data efficiency, and reduced training time. This paper presents a multi-purpose algorithm for simultaneous face recognition, facial expression recognition, age estimation, and face attribute estimation (40 attributes including gender) based on a single Swin Transformer. Our design, the SwinFace, consists of a single shared backbone together with a subnet for each set of related tasks. To address the conflicts among multiple tasks and meet the different demands of tasks, a Multi-Level Channel Attention (MLCA) module is integrated into each task-specific analysis subnet, which can adaptively select the features from optimal levels and channels to perform the desired tasks. Extensive experiments show that the proposed model has a better understanding of the face and achieves excellent performance for all tasks. Especially, it achieves 90.97% accuracy on RAF-DB and 0.22 $\epsilon$-error on CLAP2015, which are state-of-the-art results on facial expression recognition and age estimation respectively. The code and models will be made publicly available at https://github.com/lxq1000/SwinFace.

{{</citation>}}


### (52/113) Unsupervised Prototype Adapter for Vision-Language Models (Yi Zhang et al., 2023)

{{<citation>}}

Yi Zhang, Ce Zhang, Xueting Hu, Zhihai He. (2023)  
**Unsupervised Prototype Adapter for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.11507v2)  

---


**ABSTRACT**  
Recently, large-scale pre-trained vision-language models (e.g. CLIP and ALIGN) have demonstrated remarkable effectiveness in acquiring transferable visual representations. To leverage the valuable knowledge encoded within these models for downstream tasks, several fine-tuning approaches, including prompt tuning methods and adapter-based methods, have been developed to adapt vision-language models effectively with supervision. However, these methods rely on the availability of annotated samples, which can be labor-intensive and time-consuming to acquire, thus limiting scalability. To address this issue, in this work, we design an unsupervised fine-tuning approach for vision-language models called Unsupervised Prototype Adapter (UP-Adapter). Specifically, for the unannotated target datasets, we leverage the text-image aligning capability of CLIP to automatically select the most confident samples for each class. Utilizing these selected samples, we generate class prototypes, which serve as the initialization for the learnable prototype model. After fine-tuning, the prototype model prediction is combined with the original CLIP's prediction by a residual connection to perform downstream recognition tasks. Our extensive experimental results on image recognition and domain generalization show that the proposed unsupervised method outperforms 8-shot CoOp, 8-shot Tip-Adapter, and also the state-of-the-art UPL method by large margins.

{{</citation>}}


### (53/113) Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features (Alberto Baldrati et al., 2023)

{{<citation>}}

Alberto Baldrati, Marco Bertini, Tiberio Uricchio, Alberto del Bimbo. (2023)  
**Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.11485v1)  

---


**ABSTRACT**  
Given a query composed of a reference image and a relative caption, the Composed Image Retrieval goal is to retrieve images visually similar to the reference one that integrates the modifications expressed by the caption. Given that recent research has demonstrated the efficacy of large-scale vision and language pre-trained (VLP) models in various tasks, we rely on features from the OpenAI CLIP model to tackle the considered task. We initially perform a task-oriented fine-tuning of both CLIP encoders using the element-wise sum of visual and textual features. Then, in the second stage, we train a Combiner network that learns to combine the image-text features integrating the bimodal information and providing combined features used to perform the retrieval. We use contrastive learning in both stages of training. Starting from the bare CLIP features as a baseline, experimental results show that the task-oriented fine-tuning and the carefully crafted Combiner network are highly effective and outperform more complex state-of-the-art approaches on FashionIQ and CIRR, two popular and challenging datasets for composed image retrieval. Code and pre-trained models are available at https://github.com/ABaldrati/CLIP4Cir

{{</citation>}}


### (54/113) VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection (Peng Wu et al., 2023)

{{<citation>}}

Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, Yanning Zhang. (2023)  
**VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Anomaly Detection, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11681v2)  

---


**ABSTRACT**  
The recent contrastive language-image pre-training (CLIP) model has shown great success in a wide range of image-level tasks, revealing remarkable ability for learning powerful visual representations with rich semantics. An open and worthwhile problem is efficiently adapting such a strong model to the video domain and designing a robust video anomaly detector. In this work, we propose VadCLIP, a new paradigm for weakly supervised video anomaly detection (WSVAD) by leveraging the frozen CLIP model directly without any pre-training and fine-tuning process. Unlike current works that directly feed extracted features into the weakly supervised classifier for frame-level binary classification, VadCLIP makes full use of fine-grained associations between vision and language on the strength of CLIP and involves dual branch. One branch simply utilizes visual features for coarse-grained binary classification, while the other fully leverages the fine-grained language-image alignment. With the benefit of dual branch, VadCLIP achieves both coarse-grained and fine-grained video anomaly detection by transferring pre-trained knowledge from CLIP to WSVAD task. We conduct extensive experiments on two commonly-used benchmarks, demonstrating that VadCLIP achieves the best performance on both coarse-grained and fine-grained WSVAD, surpassing the state-of-the-art methods by a large margin. Specifically, VadCLIP achieves 84.51% AP and 88.02% AUC on XD-Violence and UCF-Crime, respectively. Code and features will be released to facilitate future VAD research.

{{</citation>}}


### (55/113) Multitemporal analysis in Google Earth Engine for detecting urban changes using optical data and machine learning algorithms (Mariapia Rita Iandolo et al., 2023)

{{<citation>}}

Mariapia Rita Iandolo, Francesca Razzano, Chiara Zarro, G. S. Yogesh, Silvia Liberata Ullo. (2023)  
**Multitemporal analysis in Google Earth Engine for detecting urban changes using optical data and machine learning algorithms**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.11468v1)  

---


**ABSTRACT**  
The aim of this work is to perform a multitemporal analysis using the Google Earth Engine (GEE) platform for the detection of changes in urban areas using optical data and specific machine learning (ML) algorithms. As a case study, Cairo City has been identified, in Egypt country, as one of the five most populous megacities of the last decade in the world. Classification and change detection analysis of the region of interest (ROI) have been carried out from July 2013 to July 2021. Results demonstrate the validity of the proposed method in identifying changed and unchanged urban areas over the selected period. Furthermore, this work aims to evidence the growing significance of GEE as an efficient cloud-based solution for managing large quantities of satellite data.

{{</citation>}}


### (56/113) Food Image Classification and Segmentation with Attention-based Multiple Instance Learning (Valasia Vlachopoulou et al., 2023)

{{<citation>}}

Valasia Vlachopoulou, Ioannis Sarafis, Alexandros Papadopoulos. (2023)  
**Food Image Classification and Segmentation with Attention-based Multiple Instance Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Image Classification  
[Paper Link](http://arxiv.org/abs/2308.11452v1)  

---


**ABSTRACT**  
The demand for accurate food quantification has increased in the recent years, driven by the needs of applications in dietary monitoring. At the same time, computer vision approaches have exhibited great potential in automating tasks within the food domain. Traditionally, the development of machine learning models for these problems relies on training data sets with pixel-level class annotations. However, this approach introduces challenges arising from data collection and ground truth generation that quickly become costly and error-prone since they must be performed in multiple settings and for thousands of classes. To overcome these challenges, the paper presents a weakly supervised methodology for training food image classification and semantic segmentation models without relying on pixel-level annotations. The proposed methodology is based on a multiple instance learning approach in combination with an attention-based mechanism. At test time, the models are used for classification and, concurrently, the attention mechanism generates semantic heat maps which are used for food class segmentation. In the paper, we conduct experiments on two meta-classes within the FoodSeg103 data set to verify the feasibility of the proposed approach and we explore the functioning properties of the attention mechanism.

{{</citation>}}


### (57/113) Towards Discriminative Representations with Contrastive Instances for Real-Time UAV Tracking (Dan Zeng et al., 2023)

{{<citation>}}

Dan Zeng, Mingliang Zou, Xucheng Wang, Shuiwang Li. (2023)  
**Towards Discriminative Representations with Contrastive Instances for Real-Time UAV Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2308.11450v1)  

---


**ABSTRACT**  
Maintaining high efficiency and high precision are two fundamental challenges in UAV tracking due to the constraints of computing resources, battery capacity, and UAV maximum load. Discriminative correlation filters (DCF)-based trackers can yield high efficiency on a single CPU but with inferior precision. Lightweight Deep learning (DL)-based trackers can achieve a good balance between efficiency and precision but performance gains are limited by the compression rate. High compression rate often leads to poor discriminative representations. To this end, this paper aims to enhance the discriminative power of feature representations from a new feature-learning perspective. Specifically, we attempt to learn more disciminative representations with contrastive instances for UAV tracking in a simple yet effective manner, which not only requires no manual annotations but also allows for developing and deploying a lightweight model. We are the first to explore contrastive learning for UAV tracking. Extensive experiments on four UAV benchmarks, including UAV123@10fps, DTB70, UAVDT and VisDrone2018, show that the proposed DRCI tracker significantly outperforms state-of-the-art UAV tracking methods.

{{</citation>}}


### (58/113) Masked Momentum Contrastive Learning for Zero-shot Semantic Understanding (Jiantao Wu et al., 2023)

{{<citation>}}

Jiantao Wu, Shentong Mo, Muhammad Awais, Sara Atito, Zhenhua Feng, Josef Kittler. (2023)  
**Masked Momentum Contrastive Learning for Zero-shot Semantic Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.11448v1)  

---


**ABSTRACT**  
Self-supervised pretraining (SSP) has emerged as a popular technique in machine learning, enabling the extraction of meaningful feature representations without labelled data. In the realm of computer vision, pretrained vision transformers (ViTs) have played a pivotal role in advancing transfer learning. Nonetheless, the escalating cost of finetuning these large models has posed a challenge due to the explosion of model size. This study endeavours to evaluate the effectiveness of pure self-supervised learning (SSL) techniques in computer vision tasks, obviating the need for finetuning, with the intention of emulating human-like capabilities in generalisation and recognition of unseen objects. To this end, we propose an evaluation protocol for zero-shot segmentation based on a prompting patch. Given a point on the target object as a prompt, the algorithm calculates the similarity map between the selected patch and other patches, upon that, a simple thresholding is applied to segment the target. Another evaluation is intra-object and inter-object similarity to gauge discriminatory ability of SSP ViTs. Insights from zero-shot segmentation from prompting and discriminatory abilities of SSP led to the design of a simple SSP approach, termed MMC. This approaches combines Masked image modelling for encouraging similarity of local features, Momentum based self-distillation for transferring semantics from global to local features, and global Contrast for promoting semantics of global features, to enhance discriminative representations of SSP ViTs. Consequently, our proposed method significantly reduces the overlap of intra-object and inter-object similarities, thereby facilitating effective object segmentation within an image. Our experiments reveal that MMC delivers top-tier results in zero-shot semantic segmentation across various datasets.

{{</citation>}}


### (59/113) Revisiting and Exploring Efficient Fast Adversarial Training via LAW: Lipschitz Regularization and Auto Weight Averaging (Xiaojun Jia et al., 2023)

{{<citation>}}

Xiaojun Jia, Yuefeng Chen, Xiaofeng Mao, Ranjie Duan, Jindong Gu, Rong Zhang, Hui Xue, Xiaochun Cao. (2023)  
**Revisiting and Exploring Efficient Fast Adversarial Training via LAW: Lipschitz Regularization and Auto Weight Averaging**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2308.11443v1)  

---


**ABSTRACT**  
Fast Adversarial Training (FAT) not only improves the model robustness but also reduces the training cost of standard adversarial training. However, fast adversarial training often suffers from Catastrophic Overfitting (CO), which results in poor robustness performance. Catastrophic Overfitting describes the phenomenon of a sudden and significant decrease in robust accuracy during the training of fast adversarial training. Many effective techniques have been developed to prevent Catastrophic Overfitting and improve the model robustness from different perspectives. However, these techniques adopt inconsistent training settings and require different training costs, i.e, training time and memory costs, leading to unfair comparisons. In this paper, we conduct a comprehensive study of over 10 fast adversarial training methods in terms of adversarial robustness and training costs. We revisit the effectiveness and efficiency of fast adversarial training techniques in preventing Catastrophic Overfitting from the perspective of model local nonlinearity and propose an effective Lipschitz regularization method for fast adversarial training. Furthermore, we explore the effect of data augmentation and weight averaging in fast adversarial training and propose a simple yet effective auto weight averaging method to improve robustness further. By assembling these techniques, we propose a FGSM-based fast adversarial training method equipped with Lipschitz regularization and Auto Weight averaging, abbreviated as FGSM-LAW. Experimental evaluations on four benchmark databases demonstrate the superiority of the proposed method over state-of-the-art fast adversarial training methods and the advanced standard adversarial training methods.

{{</citation>}}


### (60/113) TurboViT: Generating Fast Vision Transformers via Generative Architecture Search (Alexander Wong et al., 2023)

{{<citation>}}

Alexander Wong, Saad Abbasi, Saeejith Nair. (2023)  
**TurboViT: Generating Fast Vision Transformers via Generative Architecture Search**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.11421v1)  

---


**ABSTRACT**  
Vision transformers have shown unprecedented levels of performance in tackling various visual perception tasks in recent years. However, the architectural and computational complexity of such network architectures have made them challenging to deploy in real-world applications with high-throughput, low-memory requirements. As such, there has been significant research recently on the design of efficient vision transformer architectures. In this study, we explore the generation of fast vision transformer architecture designs via generative architecture search (GAS) to achieve a strong balance between accuracy and architectural and computational efficiency. Through this generative architecture search process, we create TurboViT, a highly efficient hierarchical vision transformer architecture design that is generated around mask unit attention and Q-pooling design patterns. The resulting TurboViT architecture design achieves significantly lower architectural computational complexity (>2.47$\times$ smaller than FasterViT-0 while achieving same accuracy) and computational complexity (>3.4$\times$ fewer FLOPs and 0.9% higher accuracy than MobileViT2-2.0) when compared to 10 other state-of-the-art efficient vision transformer network architecture designs within a similar range of accuracy on the ImageNet-1K dataset. Furthermore, TurboViT demonstrated strong inference latency and throughput in both low-latency and batch processing scenarios (>3.21$\times$ lower latency and >3.18$\times$ higher throughput compared to FasterViT-0 for low-latency scenario). These promising results demonstrate the efficacy of leveraging generative architecture search for generating efficient transformer architecture designs for high-throughput scenarios.

{{</citation>}}


### (61/113) Boundary-RL: Reinforcement Learning for Weakly-Supervised Prostate Segmentation in TRUS Images (Weixi Yi et al., 2023)

{{<citation>}}

Weixi Yi, Vasilis Stavrinides, Zachary M. C. Baum, Qianye Yang, Dean C. Barratt, Matthew J. Clarkson, Yipeng Hu, Shaheer U. Saeed. (2023)  
**Boundary-RL: Reinforcement Learning for Weakly-Supervised Prostate Segmentation in TRUS Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.11376v1)  

---


**ABSTRACT**  
We propose Boundary-RL, a novel weakly supervised segmentation method that utilises only patch-level labels for training. We envision the segmentation as a boundary detection problem, rather than a pixel-level classification as in previous works. This outlook on segmentation may allow for boundary delineation under challenging scenarios such as where noise artefacts may be present within the region-of-interest (ROI) boundaries, where traditional pixel-level classification-based weakly supervised methods may not be able to effectively segment the ROI. Particularly of interest, ultrasound images, where intensity values represent acoustic impedance differences between boundaries, may also benefit from the boundary delineation approach. Our method uses reinforcement learning to train a controller function to localise boundaries of ROIs using a reward derived from a pre-trained boundary-presence classifier. The classifier indicates when an object boundary is encountered within a patch, as the controller modifies the patch location in a sequential Markov decision process. The classifier itself is trained using only binary patch-level labels of object presence, which are the only labels used during training of the entire boundary delineation framework, and serves as a weak signal to inform the boundary delineation. The use of a controller function ensures that a sliding window over the entire image is not necessary. It also prevents possible false-positive or -negative cases by minimising number of patches passed to the boundary-presence classifier. We evaluate our proposed approach for a clinically relevant task of prostate gland segmentation on trans-rectal ultrasound images. We show improved performance compared to other tested weakly supervised methods, using the same labels e.g., multiple instance learning.

{{</citation>}}


### (62/113) Towards Clip-Free Quantized Super-Resolution Networks: How to Tame Representative Images (Alperen Kalay et al., 2023)

{{<citation>}}

Alperen Kalay, Bahri Batuhan Bilecen, Mustafa Ayazoglu. (2023)  
**Towards Clip-Free Quantized Super-Resolution Networks: How to Tame Representative Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2308.11365v1)  

---


**ABSTRACT**  
Super-resolution (SR) networks have been investigated for a while, with their mobile and lightweight versions gaining noticeable popularity recently. Quantization, the procedure of decreasing the precision of network parameters (mostly FP32 to INT8), is also utilized in SR networks for establishing mobile compatibility. This study focuses on a very important but mostly overlooked post-training quantization (PTQ) step: representative dataset (RD), which adjusts the quantization range for PTQ. We propose a novel pipeline (clip-free quantization pipeline, CFQP) backed up with extensive experimental justifications to cleverly augment RD images by only using outputs of the FP32 model. Using the proposed pipeline for RD, we can successfully eliminate unwanted clipped activation layers, which nearly all mobile SR methods utilize to make the model more robust to PTQ in return for a large overhead in runtime. Removing clipped activations with our method significantly benefits overall increased stability, decreased inference runtime up to 54% on some SR models, better visual quality results compared to INT8 clipped models - and outperforms even some FP32 non-quantized models, both in runtime and visual quality, without the need for retraining with clipped activation.

{{</citation>}}


### (63/113) Exemplar-Free Continual Transformer with Convolutions (Anurag Roy et al., 2023)

{{<citation>}}

Anurag Roy, Vinay Kumar Verma, Sravan Voonna, Kripabandhu Ghosh, Saptarshi Ghosh, Abir Das. (2023)  
**Exemplar-Free Continual Transformer with Convolutions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.11357v1)  

---


**ABSTRACT**  
Continual Learning (CL) involves training a machine learning model in a sequential manner to learn new information while retaining previously learned tasks without the presence of previous training data. Although there has been significant interest in CL, most recent CL approaches in computer vision have focused on convolutional architectures only. However, with the recent success of vision transformers, there is a need to explore their potential for CL. Although there have been some recent CL approaches for vision transformers, they either store training instances of previous tasks or require a task identifier during test time, which can be limiting. This paper proposes a new exemplar-free approach for class/task incremental learning called ConTraCon, which does not require task-id to be explicitly present during inference and avoids the need for storing previous training instances. The proposed approach leverages the transformer architecture and involves re-weighting the key, query, and value weights of the multi-head self-attention layers of a transformer trained on a similar task. The re-weighting is done using convolution, which enables the approach to maintain low parameter requirements per task. Additionally, an image augmentation-based entropic task identification approach is used to predict tasks without requiring task-ids during inference. Experiments on four benchmark datasets demonstrate that the proposed approach outperforms several competitive approaches while requiring fewer parameters.

{{</citation>}}


### (64/113) Integration of Sentinel-1 and Sentinel-2 data for Earth surface classification using Machine Learning algorithms implemented on Google Earth Engine (Francesca Razzano et al., 2023)

{{<citation>}}

Francesca Razzano, Mariapia Rita Iandolo, Chiara Zarro, G. S. Yogesh, Silvia Liberata Ullo. (2023)  
**Integration of Sentinel-1 and Sentinel-2 data for Earth surface classification using Machine Learning algorithms implemented on Google Earth Engine**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.11340v1)  

---


**ABSTRACT**  
In this study, Synthetic Aperture Radar (SAR) and optical data are both considered for Earth surface classification. Specifically, the integration of Sentinel-1 (S-1) and Sentinel-2 (S-2) data is carried out through supervised Machine Learning (ML) algorithms implemented on the Google Earth Engine (GEE) platform for the classification of a particular region of interest. Achieved results demonstrate how in this case radar and optical remote detection provide complementary information, benefiting surface cover classification and generally leading to increased mapping accuracy. In addition, this paper works in the direction of proving the emerging role of GEE as an effective cloud-based tool for handling large amounts of satellite data.

{{</citation>}}


### (65/113) Object Detection Difficulty: Suppressing Over-aggregation for Faster and Better Video Object Detection (Bingqing Zhang et al., 2023)

{{<citation>}}

Bingqing Zhang, Sen Wang, Yifan Liu, Brano Kusy, Xue Li, Jiajun Liu. (2023)  
**Object Detection Difficulty: Suppressing Over-aggregation for Faster and Better Video Object Detection**  

---
Primary Category: cs.CV  
Categories: 94A13, I-4-9, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.11327v1)  

---


**ABSTRACT**  
Current video object detection (VOD) models often encounter issues with over-aggregation due to redundant aggregation strategies, which perform feature aggregation on every frame. This results in suboptimal performance and increased computational complexity. In this work, we propose an image-level Object Detection Difficulty (ODD) metric to quantify the difficulty of detecting objects in a given image. The derived ODD scores can be used in the VOD process to mitigate over-aggregation. Specifically, we train an ODD predictor as an auxiliary head of a still-image object detector to compute the ODD score for each image based on the discrepancies between detection results and ground-truth bounding boxes. The ODD score enhances the VOD system in two ways: 1) it enables the VOD system to select superior global reference frames, thereby improving overall accuracy; and 2) it serves as an indicator in the newly designed ODD Scheduler to eliminate the aggregation of frames that are easy to detect, thus accelerating the VOD process. Comprehensive experiments demonstrate that, when utilized for selecting global reference frames, ODD-VOD consistently enhances the accuracy of Global-frame-based VOD models. When employed for acceleration, ODD-VOD consistently improves the frames per second (FPS) by an average of 73.3% across 8 different VOD models without sacrificing accuracy. When combined, ODD-VOD attains state-of-the-art performance when competing with many VOD methods in both accuracy and speed. Our work represents a significant advancement towards making VOD more practical for real-world applications.

{{</citation>}}


### (66/113) CNN based Cuneiform Sign Detection Learned from Annotated 3D Renderings and Mapped Photographs with Illumination Augmentation (Ernst Stötzner et al., 2023)

{{<citation>}}

Ernst Stötzner, Timo Homburg, Hubert Mara. (2023)  
**CNN based Cuneiform Sign Detection Learned from Annotated 3D Renderings and Mapped Photographs with Illumination Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Augmentation, OCR  
[Paper Link](http://arxiv.org/abs/2308.11277v1)  

---


**ABSTRACT**  
Motivated by the challenges of the Digital Ancient Near Eastern Studies (DANES) community, we develop digital tools for processing cuneiform script being a 3D script imprinted into clay tablets used for more than three millennia and at least eight major languages. It consists of thousands of characters that have changed over time and space. Photographs are the most common representations usable for machine learning, while ink drawings are prone to interpretation. Best suited 3D datasets that are becoming available. We created and used the HeiCuBeDa and MaiCuBeDa datasets, which consist of around 500 annotated tablets. For our novel OCR-like approach to mixed image data, we provide an additional mapping tool for transferring annotations between 3D renderings and photographs. Our sign localization uses a RepPoints detector to predict the locations of characters as bounding boxes. We use image data from GigaMesh's MSII (curvature, see https://gigamesh.eu) based rendering, Phong-shaded 3D models, and photographs as well as illumination augmentation. The results show that using rendered 3D images for sign detection performs better than other work on photographs. In addition, our approach gives reasonably good results for photographs only, while it is best used for mixed datasets. More importantly, the Phong renderings, and especially the MSII renderings, improve the results on photographs, which is the largest dataset on a global scale.

{{</citation>}}


### (67/113) ConcatPlexer: Additional Dim1 Batching for Faster ViTs (Donghoon Han et al., 2023)

{{<citation>}}

Donghoon Han, Seunghyeon Seo, Donghyeon Jeon, Jiho Jang, Chaerin Kong, Nojun Kwak. (2023)  
**ConcatPlexer: Additional Dim1 Batching for Faster ViTs**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.11199v1)  

---


**ABSTRACT**  
Transformers have demonstrated tremendous success not only in the natural language processing (NLP) domain but also the field of computer vision, igniting various creative approaches and applications. Yet, the superior performance and modeling flexibility of transformers came with a severe increase in computation costs, and hence several works have proposed methods to reduce this burden. Inspired by a cost-cutting method originally proposed for language models, Data Multiplexing (DataMUX), we propose a novel approach for efficient visual recognition that employs additional dim1 batching (i.e., concatenation) that greatly improves the throughput with little compromise in the accuracy. We first introduce a naive adaptation of DataMux for vision models, Image Multiplexer, and devise novel components to overcome its weaknesses, rendering our final model, ConcatPlexer, at the sweet spot between inference speed and accuracy. The ConcatPlexer was trained on ImageNet1K and CIFAR100 dataset and it achieved 23.5% less GFLOPs than ViT-B/16 with 69.5% and 83.4% validation accuracy, respectively.

{{</citation>}}


### (68/113) ViLLA: Fine-Grained Vision-Language Representation Learning from Real-World Data (Maya Varma et al., 2023)

{{<citation>}}

Maya Varma, Jean-Benoit Delbrouck, Sarah Hooper, Akshay Chaudhari, Curtis Langlotz. (2023)  
**ViLLA: Fine-Grained Vision-Language Representation Learning from Real-World Data**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.11194v1)  

---


**ABSTRACT**  
Vision-language models (VLMs), such as CLIP and ALIGN, are generally trained on datasets consisting of image-caption pairs obtained from the web. However, real-world multimodal datasets, such as healthcare data, are significantly more complex: each image (e.g. X-ray) is often paired with text (e.g. physician report) that describes many distinct attributes occurring in fine-grained regions of the image. We refer to these samples as exhibiting high pairwise complexity, since each image-text pair can be decomposed into a large number of region-attribute pairings. The extent to which VLMs can capture fine-grained relationships between image regions and textual attributes when trained on such data has not been previously evaluated. The first key contribution of this work is to demonstrate through systematic evaluations that as the pairwise complexity of the training dataset increases, standard VLMs struggle to learn region-attribute relationships, exhibiting performance degradations of up to 37% on retrieval tasks. In order to address this issue, we introduce ViLLA as our second key contribution. ViLLA, which is trained to capture fine-grained region-attribute relationships from complex datasets, involves two components: (a) a lightweight, self-supervised mapping model to decompose image-text samples into region-attribute pairs, and (b) a contrastive VLM to learn representations from generated region-attribute pairs. We demonstrate with experiments across four domains (synthetic, product, medical, and natural images) that ViLLA outperforms comparable VLMs on fine-grained reasoning tasks, such as zero-shot object detection (up to 3.6 AP50 points on COCO and 0.6 mAP points on LVIS) and retrieval (up to 14.2 R-Precision points).

{{</citation>}}


### (69/113) Knowledge-Aware Prompt Tuning for Generalizable Vision-Language Models (Baoshuo Kan et al., 2023)

{{<citation>}}

Baoshuo Kan, Teng Wang, Wenpeng Lu, Xiantong Zhen, Weili Guan, Feng Zheng. (2023)  
**Knowledge-Aware Prompt Tuning for Generalizable Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.11186v1)  

---


**ABSTRACT**  
Pre-trained vision-language models, e.g., CLIP, working with manually designed prompts have demonstrated great capacity of transfer learning. Recently, learnable prompts achieve state-of-the-art performance, which however are prone to overfit to seen classes, failing to generalize to unseen classes. In this paper, we propose a Knowledge-Aware Prompt Tuning (KAPT) framework for vision-language models. Our approach takes inspiration from human intelligence in which external knowledge is usually incorporated into recognizing novel categories of objects. Specifically, we design two complementary types of knowledge-aware prompts for the text encoder to leverage the distinctive characteristics of category-related external knowledge. The discrete prompt extracts the key information from descriptions of an object category, and the learned continuous prompt captures overall contexts. We further design an adaptation head for the visual encoder to aggregate salient attentive visual cues, which establishes discriminative and task-aware visual representations. We conduct extensive experiments on 11 widely-used benchmark datasets and the results verify the effectiveness in few-shot image classification, especially in generalizing to unseen categories. Compared with the state-of-the-art CoCoOp method, KAPT exhibits favorable performance and achieves an absolute gain of 3.22% on new classes and 2.57% in terms of harmonic mean.

{{</citation>}}


### (70/113) Hierarchical Point-based Active Learning for Semi-supervised Point Cloud Semantic Segmentation (Zongyi Xu et al., 2023)

{{<citation>}}

Zongyi Xu, Bo Yuan, Shanshan Zhao, Qianni Zhang, Xinbo Gao. (2023)  
**Hierarchical Point-based Active Learning for Semi-supervised Point Cloud Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Active Learning, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.11166v1)  

---


**ABSTRACT**  
Impressive performance on point cloud semantic segmentation has been achieved by fully-supervised methods with large amounts of labelled data. As it is labour-intensive to acquire large-scale point cloud data with point-wise labels, many attempts have been made to explore learning 3D point cloud segmentation with limited annotations. Active learning is one of the effective strategies to achieve this purpose but is still under-explored. The most recent methods of this kind measure the uncertainty of each pre-divided region for manual labelling but they suffer from redundant information and require additional efforts for region division. This paper aims at addressing this issue by developing a hierarchical point-based active learning strategy. Specifically, we measure the uncertainty for each point by a hierarchical minimum margin uncertainty module which considers the contextual information at multiple levels. Then, a feature-distance suppression strategy is designed to select important and representative points for manual labelling. Besides, to better exploit the unlabelled data, we build a semi-supervised segmentation framework based on our active strategy. Extensive experiments on the S3DIS and ScanNetV2 datasets demonstrate that the proposed framework achieves 96.5% and 100% performance of fully-supervised baseline with only 0.07% and 0.1% training data, respectively, outperforming the state-of-the-art weakly-supervised and active learning methods. The code will be available at https://github.com/SmiletoE/HPAL.

{{</citation>}}


### (71/113) Improving Misaligned Multi-modality Image Fusion with One-stage Progressive Dense Registration (Di Wang et al., 2023)

{{<citation>}}

Di Wang, Jinyuan Liu, Long Ma, Risheng Liu, Xin Fan. (2023)  
**Improving Misaligned Multi-modality Image Fusion with One-stage Progressive Dense Registration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.11165v1)  

---


**ABSTRACT**  
Misalignments between multi-modality images pose challenges in image fusion, manifesting as structural distortions and edge ghosts. Existing efforts commonly resort to registering first and fusing later, typically employing two cascaded stages for registration,i.e., coarse registration and fine registration. Both stages directly estimate the respective target deformation fields. In this paper, we argue that the separated two-stage registration is not compact, and the direct estimation of the target deformation fields is not accurate enough. To address these challenges, we propose a Cross-modality Multi-scale Progressive Dense Registration (C-MPDR) scheme, which accomplishes the coarse-to-fine registration exclusively using a one-stage optimization, thus improving the fusion performance of misaligned multi-modality images. Specifically, two pivotal components are involved, a dense Deformation Field Fusion (DFF) module and a Progressive Feature Fine (PFF) module. The DFF aggregates the predicted multi-scale deformation sub-fields at the current scale, while the PFF progressively refines the remaining misaligned features. Both work together to accurately estimate the final deformation fields. In addition, we develop a Transformer-Conv-based Fusion (TCF) subnetwork that considers local and long-range feature dependencies, allowing us to capture more informative features from the registered infrared and visible images for the generation of high-quality fused images. Extensive experimental analysis demonstrates the superiority of the proposed method in the fusion of misaligned cross-modality images.

{{</citation>}}


### (72/113) TOPIC: A Parallel Association Paradigm for Multi-Object Tracking under Complex Motions and Diverse Scenes (Xiaoyan Cao et al., 2023)

{{<citation>}}

Xiaoyan Cao, Yiyao Zheng, Yao Yao, Huapeng Qin, Xiaoyu Cao, Shihui Guo. (2023)  
**TOPIC: A Parallel Association Paradigm for Multi-Object Tracking under Complex Motions and Diverse Scenes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.11157v1)  

---


**ABSTRACT**  
Video data and algorithms have been driving advances in multi-object tracking (MOT). While existing MOT datasets focus on occlusion and appearance similarity, complex motion patterns are widespread yet overlooked. To address this issue, we introduce a new dataset called BEE23 to highlight complex motions. Identity association algorithms have long been the focus of MOT research. Existing trackers can be categorized into two association paradigms: single-feature paradigm (based on either motion or appearance feature) and serial paradigm (one feature serves as secondary while the other is primary). However, these paradigms are incapable of fully utilizing different features. In this paper, we propose a parallel paradigm and present the Two rOund Parallel matchIng meChanism (TOPIC) to implement it. The TOPIC leverages both motion and appearance features and can adaptively select the preferable one as the assignment metric based on motion level. Moreover, we provide an Attention-based Appearance Reconstruct Module (AARM) to reconstruct appearance feature embeddings, thus enhancing the representation of appearance features. Comprehensive experiments show that our approach achieves state-of-the-art performance on four public datasets and BEE23. Notably, our proposed parallel paradigm surpasses the performance of existing association paradigms by a large margin, e.g., reducing false negatives by 12% to 51% compared to the single-feature association paradigm. The introduced dataset and association paradigm in this work offers a fresh perspective for advancing the MOT field. The source code and dataset are available at https://github.com/holmescao/TOPICTrack.

{{</citation>}}


### (73/113) Random Word Data Augmentation with CLIP for Zero-Shot Anomaly Detection (Masato Tamura, 2023)

{{<citation>}}

Masato Tamura. (2023)  
**Random Word Data Augmentation with CLIP for Zero-Shot Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Anomaly Detection, Augmentation, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.11119v1)  

---


**ABSTRACT**  
This paper presents a novel method that leverages a visual-language model, CLIP, as a data source for zero-shot anomaly detection. Tremendous efforts have been put towards developing anomaly detectors due to their potential industrial applications. Considering the difficulty in acquiring various anomalous samples for training, most existing methods train models with only normal samples and measure discrepancies from the distribution of normal samples during inference, which requires training a model for each object category. The problem of this inefficient training requirement has been tackled by designing a CLIP-based anomaly detector that applies prompt-guided classification to each part of an image in a sliding window manner. However, the method still suffers from the labor of careful prompt ensembling with known object categories. To overcome the issues above, we propose leveraging CLIP as a data source for training. Our method generates text embeddings with the text encoder in CLIP with typical prompts that include words of normal and anomaly. In addition to these words, we insert several randomly generated words into prompts, which enables the encoder to generate a diverse set of normal and anomalous samples. Using the generated embeddings as training data, a feed-forward neural network learns to extract features of normal and anomaly from CLIP's embeddings, and as a result, a category-agnostic anomaly detector can be obtained without any training images. Experimental results demonstrate that our method achieves state-of-the-art performance without laborious prompt ensembling in zero-shot setups.

{{</citation>}}


### (74/113) Classification of the lunar surface pattern by AI architectures: Does AI see a rabbit in the Moon? (Daigo Shoji, 2023)

{{<citation>}}

Daigo Shoji. (2023)  
**Classification of the lunar surface pattern by AI architectures: Does AI see a rabbit in the Moon?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-CY, cs-HC, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2308.11107v1)  

---


**ABSTRACT**  
In Asian countries, there is a tradition that a rabbit (the Moon rabbit) lives on the Moon. As the origin of this tradition, usually, two reasons are mentioned. One reason is that the color pattern of the lunar surface is similar to the shape of a rabbit. The other reason is that both the Moon and rabbit are symbols of fertility because the Moon appears and disappears (i.e., waxing and waning) cyclically, and rabbits bear children frequently. Considering the latter reason, is the lunar surface color pattern not similar to a rabbit? Here, the similarity between rabbit and the lunar surface pattern was evaluated using seven AI architectures. In the test by CLIP, assuming that people look at the Moon in the early evening frequently, the lunar surface is more similar to a rabbit than a face at low latitude regions, while it can be classified as face as latitude increases, which is consistent with that the oldest literature about the Moon rabbit was written in India and that there is a culture of human's face in the Moon in Europe. Tested with ImageNet weights, ConvNeXt and CLIP sometimes classified the lunar surface pattern into rabbit with relatively high probabilities. Cultures are generated by our attitude to the environment. Both dynamic and static similarities may be required to induce our imagination.

{{</citation>}}


### (75/113) Addressing Fairness and Explainability in Image Classification Using Optimal Transport (Philipp Ratz et al., 2023)

{{<citation>}}

Philipp Ratz, François Hu, Arthur Charpentier. (2023)  
**Addressing Fairness and Explainability in Image Classification Using Optimal Transport**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, stat-AP  
Keywords: AI, Image Classification  
[Paper Link](http://arxiv.org/abs/2308.11090v1)  

---


**ABSTRACT**  
Algorithmic Fairness and the explainability of potentially unfair outcomes are crucial for establishing trust and accountability of Artificial Intelligence systems in domains such as healthcare and policing. Though significant advances have been made in each of the fields separately, achieving explainability in fairness applications remains challenging, particularly so in domains where deep neural networks are used. At the same time, ethical data-mining has become ever more relevant, as it has been shown countless times that fairness-unaware algorithms result in biased outcomes. Current approaches focus on mitigating biases in the outcomes of the model, but few attempts have been made to try to explain \emph{why} a model is biased. To bridge this gap, we propose a comprehensive approach that leverages optimal transport theory to uncover the causes and implications of biased regions in images, which easily extends to tabular data as well. Through the use of Wasserstein barycenters, we obtain scores that are independent of a sensitive variable but keep their marginal orderings. This step ensures predictive accuracy but also helps us to recover the regions most associated with the generation of the biases. Our findings hold significant implications for the development of trustworthy and unbiased AI systems, fostering transparency, accountability, and fairness in critical decision-making scenarios across diverse domains.

{{</citation>}}


## cs.SE (9)



### (76/113) Large-scale information retrieval in software engineering -- an experience report from industrial application (Michael Unterkalmsteiner et al., 2023)

{{<citation>}}

Michael Unterkalmsteiner, Tony Gorschek, Robert Feldt, Niklas Lavesson. (2023)  
**Large-scale information retrieval in software engineering -- an experience report from industrial application**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2308.11750v1)  

---


**ABSTRACT**  
Software Engineering activities are information intensive. Research proposes Information Retrieval (IR) techniques to support engineers in their daily tasks, such as establishing and maintaining traceability links, fault identification, and software maintenance. We describe an engineering task, test case selection, and illustrate our problem analysis and solution discovery process. The objective of the study is to gain an understanding of to what extent IR techniques (one potential solution) can be applied to test case selection and provide decision support in a large-scale, industrial setting. We analyze, in the context of the studied company, how test case selection is performed and design a series of experiments evaluating the performance of different IR techniques. Each experiment provides lessons learned from implementation, execution, and results, feeding to its successor. The three experiments led to the following observations: 1) there is a lack of research on scalable parameter optimization of IR techniques for software engineering problems; 2) scaling IR techniques to industry data is challenging, in particular for latent semantic analysis; 3) the IR context poses constraints on the empirical evaluation of IR techniques, requiring more research on developing valid statistical approaches. We believe that our experiences in conducting a series of IR experiments with industry grade data are valuable for peer researchers so that they can avoid the pitfalls that we have encountered. Furthermore, we identified challenges that need to be addressed in order to bridge the gap between laboratory IR experiments and real applications of IR in the industry.

{{</citation>}}


### (77/113) Recommending Analogical APIs via Knowledge Graph Embedding (Mingwei Liu et al., 2023)

{{<citation>}}

Mingwei Liu, Yanjun Yang, Yiling Lou, Xin Peng, Zhong Zhou, Xueying Du, Tianyong Yang. (2023)  
**Recommending Analogical APIs via Knowledge Graph Embedding**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.11422v1)  

---


**ABSTRACT**  
Library migration, which re-implements the same software behavior by using a different library instead of using the current one, has been widely observed in software evolution. One essential part of library migration is to find an analogical API that could provide the same functionality as current ones. However, given the large number of libraries/APIs, manually finding an analogical API could be very time-consuming and error-prone. Researchers have developed multiple automated analogical API recommendation techniques. Documentation-based methods have particularly attracted significant interest. Despite their potential, these methods have limitations, such as a lack of comprehensive semantic understanding in documentation and scalability challenges. In this work, we propose KGE4AR, a novel documentation-based approach that leverages knowledge graph (KG) embedding to recommend analogical APIs during library migration. Specifically, KGE4AR proposes a novel unified API KG to comprehensively and structurally represent three types of knowledge in documentation, which can better capture the high-level semantics. Moreover, KGE4AR then proposes to embed the unified API KG into vectors, enabling more effective and scalable similarity calculation. We build KGE4AR' s unified API KG for 35,773 Java libraries and assess it in two API recommendation scenarios: with and without target libraries. Our results show that KGE4AR substantially outperforms state-of-the-art documentation-based techniques in both evaluation scenarios in terms of all metrics (e.g., 47.1%-143.0% and 11.7%-80.6% MRR improvements in each scenario). Additionally, we explore KGE4AR' s scalability, confirming its effective scaling with the growing number of libraries.

{{</citation>}}


### (78/113) Towards an Understanding of Large Language Models in Software Engineering Tasks (Zibin Zheng et al., 2023)

{{<citation>}}

Zibin Zheng, Kaiwen Ning, Jiachi Chen, Yanlin Wang, Wenqing Chen, Lianghong Guo, Weicheng Wang. (2023)  
**Towards an Understanding of Large Language Models in Software Engineering Tasks**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11396v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have drawn widespread attention and research due to their astounding performance in tasks such as text generation and reasoning. Derivative products, like ChatGPT, have been extensively deployed and highly sought after. Meanwhile, the evaluation and optimization of LLMs in software engineering tasks, such as code generation, have become a research focus. However, there is still a lack of systematic research on the application and evaluation of LLMs in the field of software engineering. Therefore, this paper is the first to comprehensively investigate and collate the research and products combining LLMs with software engineering, aiming to answer two questions: (1) What are the current integrations of LLMs with software engineering? (2) Can LLMs effectively handle software engineering tasks? To find the answers, we have collected related literature as extensively as possible from seven mainstream databases, and selected 123 papers for analysis. We have categorized these papers in detail and reviewed the current research status of LLMs from the perspective of seven major software engineering tasks, hoping this will help researchers better grasp the research trends and address the issues when applying LLMs. Meanwhile, we have also organized and presented papers with evaluation content to reveal the performance and effectiveness of LLMs in various software engineering tasks, providing guidance for researchers and developers to optimize.

{{</citation>}}


### (79/113) LEAP: Efficient and Automated Test Method for NLP Software (Mingxuan Xiao et al., 2023)

{{<citation>}}

Mingxuan Xiao, Yan Xiao, Hai Dong, Shunhui Ji, Pengcheng Zhang. (2023)  
**LEAP: Efficient and Automated Test Method for NLP Software**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2308.11284v1)  

---


**ABSTRACT**  
The widespread adoption of DNNs in NLP software has highlighted the need for robustness. Researchers proposed various automatic testing techniques for adversarial test cases. However, existing methods suffer from two limitations: weak error-discovering capabilities, with success rates ranging from 0% to 24.6% for BERT-based NLP software, and time inefficiency, taking 177.8s to 205.28s per test case, making them challenging for time-constrained scenarios. To address these issues, this paper proposes LEAP, an automated test method that uses LEvy flight-based Adaptive Particle swarm optimization integrated with textual features to generate adversarial test cases. Specifically, we adopt Levy flight for population initialization to increase the diversity of generated test cases. We also design an inertial weight adaptive update operator to improve the efficiency of LEAP's global optimization of high-dimensional text examples and a mutation operator based on the greedy strategy to reduce the search time. We conducted a series of experiments to validate LEAP's ability to test NLP software and found that the average success rate of LEAP in generating adversarial test cases is 79.1%, which is 6.1% higher than the next best approach (PSOattack). While ensuring high success rates, LEAP significantly reduces time overhead by up to 147.6s compared to other heuristic-based methods. Additionally, the experimental results demonstrate that LEAP can generate more transferable test cases and significantly enhance the robustness of DNN-based systems.

{{</citation>}}


### (80/113) The Software Heritage License Dataset (2022 Edition) (Jesús M. González-Barahona et al., 2023)

{{<citation>}}

Jesús M. González-Barahona, Sergio Montes-Leon, Gregorio Robles, Stefano Zacchiroli. (2023)  
**The Software Heritage License Dataset (2022 Edition)**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.11258v1)  

---


**ABSTRACT**  
Context: When software is released publicly, it is common to include with it either the full text of the license or licenses under which it is published, or a detailed reference to them. Therefore public licenses, including FOSS (free, open source software) licenses, are usually publicly available in source code repositories.Objective: To compile a dataset containing as many documents as possible that contain the text of software licenses, or references to the license terms. Once compiled, characterize the dataset so that it can be used for further research, or practical purposes related to license analysis.Method: Retrieve from Software Heritage-the largest publicly available archive of FOSS source code-all versions of all files whose names are commonly used to convey licensing terms. All retrieved documents will be characterized in various ways, using automated and manual analyses.Results: The dataset consists of 6.9 million unique license files. Additional metadata about shipped license files is also provided, making the dataset ready to use in various contexts, including: file length measures, MIME type, SPDX license (detected using ScanCode), and oldest appearance. The results of a manual analysis of 8102 documents is also included, providing a ground truth for further analysis. The dataset is released as open data as an archive file containing all deduplicated license files, plus several portable CSV files with metadata, referencing files via cryptographic checksums.Conclusions: Thanks to the extensive coverage of Software Heritage, the dataset presented in this paper covers a very large fraction of all software licenses for public code. We have assembled a large body of software licenses, characterized it quantitatively and qualitatively, and validated that it is mostly composed of licensing information and includes almost all known license texts. The dataset can be used to conduct empirical studies on open source licensing, training of automated license classifiers, natural language processing (NLP) analyses of legal texts, as well as historical and phylogenetic studies on FOSS licensing. It can also be used in practice to improve tools detecting licenses in source code.

{{</citation>}}


### (81/113) Distinguishing Look-Alike Innocent and Vulnerable Code by Subtle Semantic Representation Learning and Explanation (Chao Ni et al., 2023)

{{<citation>}}

Chao Ni, Xin Yin, Kaiwen Yang, Dehai Zhao, Zhenchang Xing, Xin Xia. (2023)  
**Distinguishing Look-Alike Innocent and Vulnerable Code by Subtle Semantic Representation Learning and Explanation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Representation Learning, Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2308.11237v1)  

---


**ABSTRACT**  
Though many deep learning (DL)-based vulnerability detection approaches have been proposed and indeed achieved remarkable performance, they still have limitations in the generalization as well as the practical usage. More precisely, existing DL-based approaches (1) perform negatively on prediction tasks among functions that are lexically similar but have contrary semantics; (2) provide no intuitive developer-oriented explanations to the detected results. In this paper, we propose a novel approach named SVulD, a function-level Subtle semantic embedding for Vulnerability Detection along with intuitive explanations, to alleviate the above limitations. Specifically, SVulD firstly trains a model to learn distinguishing semantic representations of functions regardless of their lexical similarity. Then, for the detected vulnerable functions, SVulD provides natural language explanations (e.g., root cause) of results to help developers intuitively understand the vulnerabilities. To evaluate the effectiveness of SVulD, we conduct large-scale experiments on a widely used practical vulnerability dataset and compare it with four state-of-the-art (SOTA) approaches by considering five performance measures. The experimental results indicate that SVulD outperforms all SOTAs with a substantial improvement (i.e., 23.5%-68.0% in terms of F1-score, 15.9%-134.8% in terms of PR-AUC and 7.4%-64.4% in terms of Accuracy). Besides, we conduct a user-case study to evaluate the usefulness of SVulD for developers on understanding the vulnerable code and the participants' feedback demonstrates that SVulD is helpful for development practice.

{{</citation>}}


### (82/113) On-Premise AIOps Infrastructure for a Software Editor SME: An Experience Report (Anes Bendimerad et al., 2023)

{{<citation>}}

Anes Bendimerad, Youcef Remil, Romain Mathonat, Mehdi Kaytoue. (2023)  
**On-Premise AIOps Infrastructure for a Software Editor SME: An Experience Report**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11225v1)  

---


**ABSTRACT**  
Information Technology has become a critical component in various industries, leading to an increased focus on software maintenance and monitoring. With the complexities of modern software systems, traditional maintenance approaches have become insufficient. The concept of AIOps has emerged to enhance predictive maintenance using Big Data and Machine Learning capabilities. However, exploiting AIOps requires addressing several challenges related to the complexity of data and incident management. Commercial solutions exist, but they may not be suitable for certain companies due to high costs, data governance issues, and limitations in covering private software. This paper investigates the feasibility of implementing on-premise AIOps solutions by leveraging open-source tools. We introduce a comprehensive AIOps infrastructure that we have successfully deployed in our company, and we provide the rationale behind different choices that we made to build its various components. Particularly, we provide insights into our approach and criteria for selecting a data management system and we explain its integration. Our experience can be beneficial for companies seeking to internally manage their software maintenance processes with a modern AIOps approach.

{{</citation>}}


### (83/113) Adversarial Attacks on Code Models with Discriminative Graph Patterns (Thanh-Dat Nguyen et al., 2023)

{{<citation>}}

Thanh-Dat Nguyen, Yang Zhou, Xuan Bach D. Le, Patanamon, Thongtanunam, David Lo. (2023)  
**Adversarial Attacks on Code Models with Discriminative Graph Patterns**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Adversarial Attack, BERT  
[Paper Link](http://arxiv.org/abs/2308.11161v1)  

---


**ABSTRACT**  
Pre-trained language models of code are now widely used in various software engineering tasks such as code generation, code completion, vulnerability detection, etc. This, in turn, poses security and reliability risks to these models. One of the important threats is \textit{adversarial attacks}, which can lead to erroneous predictions and largely affect model performance on downstream tasks. Current adversarial attacks on code models usually adopt fixed sets of program transformations, such as variable renaming and dead code insertion, leading to limited attack effectiveness. To address the aforementioned challenges, we propose a novel adversarial attack framework, GraphCodeAttack, to better evaluate the robustness of code models. Given a target code model, GraphCodeAttack automatically mines important code patterns, which can influence the model's decisions, to perturb the structure of input code to the model. To do so, GraphCodeAttack uses a set of input source codes to probe the model's outputs and identifies the \textit{discriminative} ASTs patterns that can influence the model decisions. GraphCodeAttack then selects appropriate AST patterns, concretizes the selected patterns as attacks, and inserts them as dead code into the model's input program. To effectively synthesize attacks from AST patterns, GraphCodeAttack uses a separate pre-trained code model to fill in the ASTs with concrete code snippets. We evaluate the robustness of two popular code models (e.g., CodeBERT and GraphCodeBERT) against our proposed approach on three tasks: Authorship Attribution, Vulnerability Prediction, and Clone Detection. The experimental results suggest that our proposed approach significantly outperforms state-of-the-art approaches in attacking code models such as CARROT and ALERT.

{{</citation>}}


### (84/113) LLaMA-Reviewer: Advancing Code Review Automation with Large Language Models through Parameter-Efficient Fine-Tuning (Practical Experience Report) (Junyi Lu et al., 2023)

{{<citation>}}

Junyi Lu, Lei Yu, Xiaojia Li, Li Yang, Chun Zuo. (2023)  
**LLaMA-Reviewer: Advancing Code Review Automation with Large Language Models through Parameter-Efficient Fine-Tuning (Practical Experience Report)**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-LG, cs-SE, cs.SE  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11148v1)  

---


**ABSTRACT**  
The automation of code review activities, a long-standing pursuit in software engineering, has been primarily addressed by numerous domain-specific pre-trained models. Despite their success, these models frequently demand extensive resources for pre-training from scratch. In contrast, Large Language Models (LLMs) provide an intriguing alternative, given their remarkable capabilities when supplemented with domain-specific knowledge. However, their potential for automating code review tasks remains largely unexplored.   In response to this research gap, we present LLaMA-Reviewer, an innovative framework that leverages the capabilities of LLaMA, a popular LLM, in the realm of code review. Mindful of resource constraints, this framework employs parameter-efficient fine-tuning (PEFT) methods, delivering high performance while using less than 1% of trainable parameters.   An extensive evaluation of LLaMA-Reviewer is conducted on two diverse, publicly available datasets. Notably, even with the smallest LLaMA base model consisting of 6.7B parameters and a limited number of tuning epochs, LLaMA-Reviewer equals the performance of existing code-review-focused models.   The ablation experiments provide insights into the influence of various fine-tuning process components, including input representation, instruction tuning, and different PEFT methods. To foster continuous progress in this field, the code and all PEFT-weight plugins have been made open-source.

{{</citation>}}


## eess.IV (1)



### (85/113) Open Set Synthetic Image Source Attribution (Shengbang Fang et al., 2023)

{{<citation>}}

Shengbang Fang, Tai D. Nguyen, Matthew C. Stamm. (2023)  
**Open Set Synthetic Image Source Attribution**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11557v1)  

---


**ABSTRACT**  
AI-generated images have become increasingly realistic and have garnered significant public attention. While synthetic images are intriguing due to their realism, they also pose an important misinformation threat. To address this new threat, researchers have developed multiple algorithms to detect synthetic images and identify their source generators. However, most existing source attribution techniques are designed to operate in a closed-set scenario, i.e. they can only be used to discriminate between known image generators. By contrast, new image-generation techniques are rapidly emerging. To contend with this, there is a great need for open-set source attribution techniques that can identify when synthetic images have originated from new, unseen generators. To address this problem, we propose a new metric learning-based approach. Our technique works by learning transferrable embeddings capable of discriminating between generators, even when they are not seen during training. An image is first assigned to a candidate generator, then is accepted or rejected based on its distance in the embedding space from known generators' learned reference points. Importantly, we identify that initializing our source attribution embedding network by pretraining it on image camera identification can improve our embeddings' transferability. Through a series of experiments, we demonstrate our approach's ability to attribute the source of synthetic images in open-set scenarios.

{{</citation>}}


## cs.SI (2)



### (86/113) User Identity Linkage in Social Media Using Linguistic and Social Interaction Features (Despoina Chatzakou et al., 2023)

{{<citation>}}

Despoina Chatzakou, Juan Soler-Company, Theodora Tsikrika, Leo Wanner, Stefanos Vrochidis, Ioannis Kompatsiaris. (2023)  
**User Identity Linkage in Social Media Using Linguistic and Social Interaction Features**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2308.11684v1)  

---


**ABSTRACT**  
Social media users often hold several accounts in their effort to multiply the spread of their thoughts, ideas, and viewpoints. In the particular case of objectionable content, users tend to create multiple accounts to bypass the combating measures enforced by social media platforms and thus retain their online identity even if some of their accounts are suspended. User identity linkage aims to reveal social media accounts likely to belong to the same natural person so as to prevent the spread of abusive/illegal activities. To this end, this work proposes a machine learning-based detection model, which uses multiple attributes of users' online activity in order to identify whether two or more virtual identities belong to the same real natural person. The models efficacy is demonstrated on two cases on abusive and terrorism-related Twitter content.

{{</citation>}}


### (87/113) Does society show differential attention to researchers based on gender and field? (Sara M. González-Betancor et al., 2023)

{{<citation>}}

Sara M. González-Betancor, Pablo Dorta-González. (2023)  
**Does society show differential attention to researchers based on gender and field?**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, stat-AP  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2308.11382v1)  

---


**ABSTRACT**  
While not all researchers prioritize social impact, it is undeniably a crucial aspect that adds significance to their work. The objective of this paper is to explore potential gender differences in the social attention paid to researchers and to examine their association with specific fields of study. To achieve this goal, the paper analyzes four dimensions of social influence and examines three measures of social attention to researchers. The dimensions are media influence (mentions in mainstream news), political influence (mentions in public policy reports), social media influence (mentions in Twitter), and educational influence (mentions in Wikipedia). The measures of social attention to researchers are: proportion of publications with social mentions (social attention orientation), mentions per publication (level of social attention), and mentions per mentioned publication (intensity of social attention). By analyzing the rankings of authors -- for the four dimensions with the three measures in the 22 research fields of the Web of Science database -- and by using Spearman correlation coefficients, we conclude that: 1) significant differences are observed between fields; 2) the dimensions capture different and independent aspects of the social impact. Finally, we use non-parametric means comparison tests to detect gender bias in social attention. We conclude that for most fields and dimensions with enough non-zero altmetrics data, gender differences in social attention are not predominant, but are still present and vary across fields.

{{</citation>}}


## cs.AI (3)



### (88/113) A Survey on Large Language Model based Autonomous Agents (Lei Wang et al., 2023)

{{<citation>}}

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen. (2023)  
**A Survey on Large Language Model based Autonomous Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11432v1)  

---


**ABSTRACT**  
Autonomous agents have long been a prominent research topic in the academic community. Previous research in this field often focuses on training agents with limited knowledge within isolated environments, which diverges significantly from the human learning processes, and thus makes the agents hard to achieve human-like decisions. Recently, through the acquisition of vast amounts of web knowledge, large language models (LLMs) have demonstrated remarkable potential in achieving human-level intelligence. This has sparked an upsurge in studies investigating autonomous agents based on LLMs. To harness the full potential of LLMs, researchers have devised diverse agent architectures tailored to different applications. In this paper, we present a comprehensive survey of these studies, delivering a systematic review of the field of autonomous agents from a holistic perspective. More specifically, our focus lies in the construction of LLM-based agents, for which we propose a unified framework that encompasses a majority of the previous work. Additionally, we provide a summary of the various applications of LLM-based AI agents in the domains of social science, natural science, and engineering. Lastly, we discuss the commonly employed evaluation strategies for LLM-based AI agents. Based on the previous studies, we also present several challenges and future directions in this field. To keep track of this field and continuously update our survey, we maintain a repository for the related references at https://github.com/Paitesanshi/LLM-Agent-Survey.

{{</citation>}}


### (89/113) ProAgent: Building Proactive Cooperative AI with Large Language Models (Ceyao Zhang et al., 2023)

{{<citation>}}

Ceyao Zhang, Kaijie Yang, Siyi Hu, Zihao Wang, Guanghe Li, Yihang Sun, Cheng Zhang, Zhaowei Zhang, Anji Liu, Song-Chun Zhu, Xiaojun Chang, Junge Zhang, Feng Yin, Yitao Liang, Yaodong Yang. (2023)  
**ProAgent: Building Proactive Cooperative AI with Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-MA, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11339v1)  

---


**ABSTRACT**  
Building AIs with adaptive behaviors in human-AI cooperation stands as a pivotal focus in AGI research. Current methods for developing cooperative agents predominantly rely on learning-based methods, where policy generalization heavily hinges on past interactions with specific teammates. These approaches constrain the agent's capacity to recalibrate its strategy when confronted with novel teammates. We propose \textbf{ProAgent}, a novel framework that harnesses large language models (LLMs) to fashion a \textit{pro}active \textit{agent} empowered with the ability to anticipate teammates' forthcoming decisions and formulate enhanced plans for itself. ProAgent excels at cooperative reasoning with the capacity to dynamically adapt its behavior to enhance collaborative efforts with teammates. Moreover, the ProAgent framework exhibits a high degree of modularity and interpretability, facilitating seamless integration to address a wide array of coordination scenarios. Experimental evaluations conducted within the framework of \textit{Overcook-AI} unveil the remarkable performance superiority of ProAgent, outperforming five methods based on self-play and population-based training in cooperation with AI agents. Further, when cooperating with human proxy models, its performance exhibits an average improvement exceeding 10\% compared to the current state-of-the-art, COLE. The advancement was consistently observed across diverse scenarios involving interactions with both AI agents of varying characteristics and human counterparts. These findings inspire future research for human-robot collaborations. For a hands-on demonstration, please visit \url{https://pku-proagent.github.io}.

{{</citation>}}


### (90/113) Evaluating Large Language Models on Graphs: Performance Insights and Comparative Analysis (Chang Liu et al., 2023)

{{<citation>}}

Chang Liu, Bo Wu. (2023)  
**Evaluating Large Language Models on Graphs: Performance Insights and Comparative Analysis**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11224v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have garnered considerable interest within both academic and industrial. Yet, the application of LLMs to graph data remains under-explored. In this study, we evaluate the capabilities of four LLMs in addressing several analytical problems with graph data. We employ four distinct evaluation metrics: Comprehension, Correctness, Fidelity, and Rectification. Our results show that: 1) LLMs effectively comprehend graph data in natural language and reason with graph topology. 2) GPT models can generate logical and coherent results, outperforming alternatives in correctness. 3) All examined LLMs face challenges in structural reasoning, with techniques like zero-shot chain-of-thought and few-shot prompting showing diminished efficacy. 4) GPT models often produce erroneous answers in multi-answer tasks, raising concerns in fidelity. 5) GPT models exhibit elevated confidence in their outputs, potentially hindering their rectification capacities. Notably, GPT-4 has demonstrated the capacity to rectify responses from GPT-3.5-turbo and its own previous iterations. The code is available at: https://github.com/Ayame1006/LLMtoGraph.

{{</citation>}}


## cs.HC (2)



### (91/113) AIxArtist: A First-Person Tale of Interacting with Artificial Intelligence to Escape Creative Block (Makayla Lewis, 2023)

{{<citation>}}

Makayla Lewis. (2023)  
**AIxArtist: A First-Person Tale of Interacting with Artificial Intelligence to Escape Creative Block**  

---
Primary Category: cs.HC  
Categories: 68T99, I-2-m, cs-AI, cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.11424v1)  

---


**ABSTRACT**  
The future of the arts and artificial intelligence (AI) is promising as technology advances. As the use of AI in design becomes more widespread, art practice may not be a human-only art form and could instead become a digitally integrated experience. With enhanced creativity and collaboration, arts and AI could work together towards creating artistic outputs that are visually appealing and meet the needs of the artist and viewer. While it is uncertain how far the integration will go, arts and AI will likely influence one another. This workshop pictorial puts forward first-person research that shares interactions between an HCI researcher and AI as they try to escape the creative block. The pictorial paper explores two questions: How can AI support artists' creativity, and what does it mean to be explainable in this context? HIs, ChatGPT and Midjourney were engaged; the result was a series of reflections that require further discussion and explorations in the XAIxArts community: Transparency of attribution, the creation process, ethics of asking, and inspiration vs copying.

{{</citation>}}


### (92/113) MusicJam: Visualizing Music Insights via Generated Narrative Illustrations (Chuer Chen et al., 2023)

{{<citation>}}

Chuer Chen, Nan Cao, Jiani Hou, Yi Guo, Yulei Zhang, Yang Shi. (2023)  
**MusicJam: Visualizing Music Insights via Generated Narrative Illustrations**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2308.11329v1)  

---


**ABSTRACT**  
Visualizing the insights of the invisible music is able to bring listeners an enjoyable and immersive listening experience, and therefore has attracted much attention in the field of information visualization. Over the past decades, various music visualization techniques have been introduced. However, most of them are manually designed by following the visual encoding rules, thus shown in form of a graphical visual representation whose visual encoding schema is usually taking effort to understand. Recently, some researchers use figures or illustrations to represent music moods, lyrics, and musical features, which are more intuitive and attractive. However, in these techniques, the figures are usually pre-selected or statically generated, so they cannot precisely convey insights of different pieces of music. To address this issue, in this paper, we introduce MusicJam, a music visualization system that is able to generate narrative illustrations to represent the insight of the input music. The system leverages a novel generation model designed based on GPT-2 to generate meaningful lyrics given the input music and then employs the stable diffusion model to transform the lyrics into coherent illustrations. Finally, the generated results are synchronized and rendered as an MP4 video accompanied by the input music. We evaluated the proposed lyric generation model by comparing it to the baseline models and conducted a user study to estimate the quality of the generated illustrations and the final music videos. The results showed the power of our technique.

{{</citation>}}


## cs.IT (2)



### (93/113) Achievable Sum-rate of variants of QAM over Gaussian Multiple Access Channel with and without security (Shifa Showkat et al., 2023)

{{<citation>}}

Shifa Showkat, Zahid Bashir Dar, Shahid Mehraj Shah. (2023)  
**Achievable Sum-rate of variants of QAM over Gaussian Multiple Access Channel with and without security**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.11405v1)  

---


**ABSTRACT**  
The performance of next generation wireless systems (5G/6G and beyond) at the physical layer is primarily driven by the choice of digital modulation techniques that are bandwidth and power efficient, while maintaining high data rates. Achievable rates for Gaussian input and some finite constellations (BPSK/QPSK/QAM) are well studied in the literature. However, new variants of Quadrature Amplitude Modulation (QAM) such as Cross-QAM (XQAM), Star-QAM (S-QAM), Amplitude and phase shift keying (APSK), and Hexagonal Quadrature Amplitude Modulation (H-QAM) are not studied in the context of achievable rates for meeting the demand of high data rates. In this paper, we study achievable rate region for different variants of M-QAM like Cross-QAM, H-QAM, Star-QAM and APSK. We also compute mutual information corresponding to the sum rate of Gaussian Multiple Access Channel (G-MAC), for hybrid constellation scheme, e.g., user 1 transmits using Star-QAM and user 2 by H-QAM. From the results, it is observed that S-QAM gives the maximum sum-rate when users transmit same constellations. Also, it has been found that when hybrid constellation is used, the combination of Star-QAM \& H-QAM gives the maximum rate. In the next part of the paper, we consider a scenario wherein an adversary is also present at the receiver side and is trying to decode the information. We model this scenario as Gaussian Multiple Access Wiretap Channel (G-MAW-WT). We then compute the achievable secrecy sum rate of two user G-MAC-WT with discrete inputs from different variants of QAM (viz, X-QAM, H-QAM and S-QAM).It has been found that at higher values of SNR, S-QAM gives better values of SSR than the other variants. For hybrid inputs of QAM, at lower values of SNR, combination of APSK and S-QAM gives better results and at higher values of SNR, combination of HQAM and APSK gives greater value of SSR.

{{</citation>}}


### (94/113) Graph Neural Network-Enhanced Expectation Propagation Algorithm for MIMO Turbo Receivers (Xingyu Zhou et al., 2023)

{{<citation>}}

Xingyu Zhou, Jing Zhang, Chao-Kai Wen, Shi Jin, Shuangfeng Han. (2023)  
**Graph Neural Network-Enhanced Expectation Propagation Algorithm for MIMO Turbo Receivers**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.11335v1)  

---


**ABSTRACT**  
Deep neural networks (NNs) are considered a powerful tool for balancing the performance and complexity of multiple-input multiple-output (MIMO) receivers due to their accurate feature extraction, high parallelism, and excellent inference ability. Graph NNs (GNNs) have recently demonstrated outstanding capability in learning enhanced message passing rules and have shown success in overcoming the drawback of inaccurate Gaussian approximation of expectation propagation (EP)-based MIMO detectors. However, the application of the GNN-enhanced EP detector to MIMO turbo receivers is underexplored and non-trivial due to the requirement of extrinsic information for iterative processing. This paper proposes a GNN-enhanced EP algorithm for MIMO turbo receivers, which realizes the turbo principle of generating extrinsic information from the MIMO detector through a specially designed training procedure. Additionally, an edge pruning strategy is designed to eliminate redundant connections in the original fully connected model of the GNN utilizing the correlation information inherently from the EP algorithm. Edge pruning reduces the computational cost dramatically and enables the network to focus more attention on the weights that are vital for performance. Simulation results and complexity analysis indicate that the proposed MIMO turbo receiver outperforms the EP turbo approaches by over 1 dB at the bit error rate of $10^{-5}$, exhibits performance equivalent to state-of-the-art receivers with 2.5 times shorter running time, and adapts to various scenarios.

{{</citation>}}


## eess.SY (1)



### (95/113) A Partially Observable Deep Multi-Agent Active Inference Framework for Resource Allocation in 6G and Beyond Wireless Communications Networks (Fuhui Zhou et al., 2023)

{{<citation>}}

Fuhui Zhou, Rui Ding, Qihui Wu, Derrick Wing Kwan Ng, Kai-Kit Wong, Naofal Al-Dhahir. (2023)  
**A Partially Observable Deep Multi-Agent Active Inference Framework for Resource Allocation in 6G and Beyond Wireless Communications Networks**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SP, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11402v1)  

---


**ABSTRACT**  
Resource allocation is of crucial importance in wireless communications. However, it is extremely challenging to design efficient resource allocation schemes for future wireless communication networks since the formulated resource allocation problems are generally non-convex and consist of various coupled variables. Moreover, the dynamic changes of practical wireless communication environment and user service requirements thirst for efficient real-time resource allocation. To tackle these issues, a novel partially observable deep multi-agent active inference (PODMAI) framework is proposed for realizing intelligent resource allocation. A belief based learning method is exploited for updating the policy by minimizing the variational free energy. A decentralized training with a decentralized execution multi-agent strategy is designed to overcome the limitations of the partially observable state information. Exploited the proposed framework, an intelligent spectrum allocation and trajectory optimization scheme is developed for a spectrum sharing unmanned aerial vehicle (UAV) network with dynamic transmission rate requirements as an example. Simulation results demonstrate that our proposed framework can significantly improve the sum transmission rate of the secondary network compared to various benchmark schemes. Moreover, the convergence speed of the proposed PODMAI is significantly improved compared with the conventional reinforcement learning framework. Overall, our proposed framework can enrich the intelligent resource allocation frameworks and pave the way for realizing real-time resource allocation.

{{</citation>}}


## cs.DS (1)



### (96/113) Parameterized Complexity of Simultaneous Planarity (Simon D. Fink et al., 2023)

{{<citation>}}

Simon D. Fink, Matthias Pfretzschner, Ignaz Rutter. (2023)  
**Parameterized Complexity of Simultaneous Planarity**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs.DS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.11401v1)  

---


**ABSTRACT**  
Given $k$ input graphs $G_1, \dots ,G_k$, where each pair $G_i$, $G_j$ with $i \neq j$ shares the same graph $G$, the problem Simultaneous Embedding With Fixed Edges (SEFE) asks whether there exists a planar drawing for each input graph such that all drawings coincide on $G$. While SEFE is still open for the case of two input graphs, the problem is NP-complete for $k \geq 3$ [Schaefer, JGAA 13]. In this work, we explore the parameterized complexity of SEFE. We show that SEFE is FPT with respect to $k$ plus the vertex cover number or the feedback edge set number of the the union graph $G^\cup = G_1 \cup \dots \cup G_k$. Regarding the shared graph $G$, we show that SEFE is NP-complete, even if $G$ is a tree with maximum degree 4. Together with a known NP-hardness reduction [Angelini et al., TCS 15], this allows us to conclude that several parameters of $G$, including the maximum degree, the maximum number of degree-1 neighbors, the vertex cover number, and the number of cutvertices are intractable. We also settle the tractability of all pairs of these parameters. We give FPT algorithms for the vertex cover number plus either of the first two parameters and for the number of cutvertices plus the maximum degree, whereas we prove all remaining combinations to be intractable.

{{</citation>}}


## eess.SP (2)



### (97/113) WEARS: Wearable Emotion AI with Real-time Sensor data (Dhruv Limbani et al., 2023)

{{<citation>}}

Dhruv Limbani, Daketi Yatin, Nitish Chaturvedi, Vaishnavi Moorthy, Pushpalatha M, Harichandana BSS, Sumit Kumar. (2023)  
**WEARS: Wearable Emotion AI with Real-time Sensor data**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11673v1)  

---


**ABSTRACT**  
Emotion prediction is the field of study to understand human emotions. Existing methods focus on modalities like text, audio, facial expressions, etc., which could be private to the user. Emotion can be derived from the subject's psychological data as well. Various approaches that employ combinations of physiological sensors for emotion recognition have been proposed. Yet, not all sensors are simple to use and handy for individuals in their daily lives. Thus, we propose a system to predict user emotion using smartwatch sensors. We design a framework to collect ground truth in real-time utilizing a mix of English and regional language-based videos to invoke emotions in participants and collect the data. Further, we modeled the problem as binary classification due to the limited dataset size and experimented with multiple machine-learning models. We also did an ablation study to understand the impact of features including Heart Rate, Accelerometer, and Gyroscope sensor data on mood. From the experimental results, Multi-Layer Perceptron has shown a maximum accuracy of 93.75 percent for pleasant-unpleasant (high/low valence classification) moods.

{{</citation>}}


### (98/113) Machine Learning-based Positioning using Multivariate Time Series Classification for Factory Environments (Nisal Hemadasa Manikku Badu et al., 2023)

{{<citation>}}

Nisal Hemadasa Manikku Badu, Marcus Venzke, Volker Turau, Yanqiu Huang. (2023)  
**Machine Learning-based Positioning using Multivariate Time Series Classification for Factory Environments**  

---
Primary Category: eess.SP  
Categories: I-2-6, cs-LG, eess-SP, eess.SP  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.11670v1)  

---


**ABSTRACT**  
Indoor Positioning Systems (IPS) gained importance in many industrial applications. State-of-the-art solutions heavily rely on external infrastructures and are subject to potential privacy compromises, external information requirements, and assumptions, that make it unfavorable for environments demanding privacy and prolonged functionality. In certain environments deploying supplementary infrastructures for indoor positioning could be infeasible and expensive. Recent developments in machine learning (ML) offer solutions to address these limitations relying only on the data from onboard sensors of IoT devices. However, it is unclear which model fits best considering the resource constraints of IoT devices. This paper presents a machine learning-based indoor positioning system, using motion and ambient sensors, to localize a moving entity in privacy concerned factory environments. The problem is formulated as a multivariate time series classification (MTSC) and a comparative analysis of different machine learning models is conducted in order to address it. We introduce a novel time series dataset emulating the assembly lines of a factory. This dataset is utilized to assess and compare the selected models in terms of accuracy, memory footprint and inference speed. The results illustrate that all evaluated models can achieve accuracies above 80 %. CNN-1D shows the most balanced performance, followed by MLP. DT was found to have the lowest memory footprint and inference latency, indicating its potential for a deployment in real-world scenarios.

{{</citation>}}


## cs.MM (1)



### (99/113) M3PS: End-to-End Multi-Grained Multi-Modal Attribute-Aware Product Summarization in E-commerce (Tao Chen et al., 2023)

{{<citation>}}

Tao Chen, Ze Lin, Hui Li, Jiayi Ji, Yiyi Zhou, Guanbin Li, Rongrong Ji. (2023)  
**M3PS: End-to-End Multi-Grained Multi-Modal Attribute-Aware Product Summarization in E-commerce**  

---
Primary Category: cs.MM  
Categories: cs-CL, cs-MM, cs.MM  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2308.11351v1)  

---


**ABSTRACT**  
Given the long textual product information and the product image, Multi-Modal Product Summarization (MMPS) aims to attract customers' interest and increase their desire to purchase by highlighting product characteristics with a short textual summary. Existing MMPS methods have achieved promising performance. Nevertheless, there still exist several problems: 1) lack end-to-end product summarization, 2) lack multi-grained multi-modal modeling, and 3) lack multi-modal attribute modeling. To address these issues, we propose an end-to-end multi-grained multi-modal attribute-aware product summarization method (M3PS) for generating high-quality product summaries in e-commerce. M3PS jointly models product attributes and generates product summaries. Meanwhile, we design several multi-grained multi-modal tasks to better guide the multi-modal learning of M3PS. Furthermore, we model product attributes based on both text and image modalities so that multi-modal product characteristics can be manifested in the generated summaries. Extensive experiments on a real large-scale Chinese e-commence dataset demonstrate that our model outperforms state-of-the-art product summarization methods w.r.t. several summarization metrics.

{{</citation>}}


## cs.IR (6)



### (100/113) On the Opportunities and Challenges of Offline Reinforcement Learning for Recommender Systems (Xiaocong Chen et al., 2023)

{{<citation>}}

Xiaocong Chen, Siyu Wang, Julian McAuley, Dietmar Jannach, Lina Yao. (2023)  
**On the Opportunities and Challenges of Offline Reinforcement Learning for Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.11336v1)  

---


**ABSTRACT**  
Reinforcement learning serves as a potent tool for modeling dynamic user interests within recommender systems, garnering increasing research attention of late. However, a significant drawback persists: its poor data efficiency, stemming from its interactive nature. The training of reinforcement learning-based recommender systems demands expensive online interactions to amass adequate trajectories, essential for agents to learn user preferences. This inefficiency renders reinforcement learning-based recommender systems a formidable undertaking, necessitating the exploration of potential solutions. Recent strides in offline reinforcement learning present a new perspective. Offline reinforcement learning empowers agents to glean insights from offline datasets and deploy learned policies in online settings. Given that recommender systems possess extensive offline datasets, the framework of offline reinforcement learning aligns seamlessly. Despite being a burgeoning field, works centered on recommender systems utilizing offline reinforcement learning remain limited. This survey aims to introduce and delve into offline reinforcement learning within recommender systems, offering an inclusive review of existing literature in this domain. Furthermore, we strive to underscore prevalent challenges, opportunities, and future pathways, poised to propel research in this evolving field.

{{</citation>}}


### (101/113) Test Time Embedding Normalization for Popularity Bias Mitigation (Dain Kim et al., 2023)

{{<citation>}}

Dain Kim, Jinhyeok Park, Dongwoo Kim. (2023)  
**Test Time Embedding Normalization for Popularity Bias Mitigation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Bias, Embedding  
[Paper Link](http://arxiv.org/abs/2308.11288v1)  

---


**ABSTRACT**  
Popularity bias is a widespread problem in the field of recommender systems, where popular items tend to dominate recommendation results. In this work, we propose 'Test Time Embedding Normalization' as a simple yet effective strategy for mitigating popularity bias, which surpasses the performance of the previous mitigation approaches by a significant margin. Our approach utilizes the normalized item embedding during the inference stage to control the influence of embedding magnitude, which is highly correlated with item popularity. Through extensive experiments, we show that our method combined with the sampled softmax loss effectively reduces popularity bias compare to previous approaches for bias mitigation. We further investigate the relationship between user and item embeddings and find that the angular similarity between embeddings distinguishes preferable and non-preferable items regardless of their popularity. The analysis explains the mechanism behind the success of our approach in eliminating the impact of popularity bias. Our code is available at https://github.com/ml-postech/TTEN.

{{</citation>}}


### (102/113) MISSRec: Pre-training and Transferring Multi-modal Interest-aware Sequence Representation for Recommendation (Jinpeng Wang et al., 2023)

{{<citation>}}

Jinpeng Wang, Ziyun Zeng, Yunxiao Wang, Yuting Wang, Xingyu Lu, Tianxiang Li, Jun Yuan, Rui Zhang, Hai-Tao Zheng, Shu-Tao Xia. (2023)  
**MISSRec: Pre-training and Transferring Multi-modal Interest-aware Sequence Representation for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-MM, cs.IR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.11175v1)  

---


**ABSTRACT**  
The goal of sequential recommendation (SR) is to predict a user's potential interested items based on her/his historical interaction sequences. Most existing sequential recommenders are developed based on ID features, which, despite their widespread use, often underperform with sparse IDs and struggle with the cold-start problem. Besides, inconsistent ID mappings hinder the model's transferability, isolating similar recommendation domains that could have been co-optimized. This paper aims to address these issues by exploring the potential of multi-modal information in learning robust and generalizable sequence representations. We propose MISSRec, a multi-modal pre-training and transfer learning framework for SR. On the user side, we design a Transformer-based encoder-decoder model, where the contextual encoder learns to capture the sequence-level multi-modal synergy while a novel interest-aware decoder is developed to grasp item-modality-interest relations for better sequence representation. On the candidate item side, we adopt a dynamic fusion module to produce user-adaptive item representation, providing more precise matching between users and items. We pre-train the model with contrastive learning objectives and fine-tune it in an efficient manner. Extensive experiments demonstrate the effectiveness and flexibility of MISSRec, promising an practical solution for real-world recommendation scenarios.

{{</citation>}}


### (103/113) Towards Validating Long-Term User Feedbacks in Interactive Recommendation Systems (Hojoon Lee et al., 2023)

{{<citation>}}

Hojoon Lee, Dongyoon Hwang, Kyushik Min, Jaegul Choo. (2023)  
**Towards Validating Long-Term User Feedbacks in Interactive Recommendation Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.11137v1)  

---


**ABSTRACT**  
Interactive Recommender Systems (IRSs) have attracted a lot of attention, due to their ability to model interactive processes between users and recommender systems. Numerous approaches have adopted Reinforcement Learning (RL) algorithms, as these can directly maximize users' cumulative rewards. In IRS, researchers commonly utilize publicly available review datasets to compare and evaluate algorithms. However, user feedback provided in public datasets merely includes instant responses (e.g., a rating), with no inclusion of delayed responses (e.g., the dwell time and the lifetime value). Thus, the question remains whether these review datasets are an appropriate choice to evaluate the long-term effects of the IRS. In this work, we revisited experiments on IRS with review datasets and compared RL-based models with a simple reward model that greedily recommends the item with the highest one-step reward. Following extensive analysis, we can reveal three main findings: First, a simple greedy reward model consistently outperforms RL-based models in maximizing cumulative rewards. Second, applying higher weighting to long-term rewards leads to a degradation of recommendation performance. Third, user feedbacks have mere long-term effects on the benchmark datasets. Based on our findings, we conclude that a dataset has to be carefully verified and that a simple greedy baseline should be included for a proper evaluation of RL-based IRS approaches.

{{</citation>}}


### (104/113) ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation (Jianghao Lin et al., 2023)

{{<citation>}}

Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan, Ruiming Tang, Yong Yu, Weinan Zhang. (2023)  
**ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.11131v1)  

---


**ABSTRACT**  
With large language models (LLMs) achieving remarkable breakthroughs in natural language processing (NLP) domains, LLM-enhanced recommender systems have received much attention and have been actively explored currently. In this paper, we focus on adapting and empowering a pure large language model for zero-shot and few-shot recommendation tasks. First and foremost, we identify and formulate the lifelong sequential behavior incomprehension problem for LLMs in recommendation domains, i.e., LLMs fail to extract useful information from a textual context of long user behavior sequence, even if the length of context is far from reaching the context limitation of LLMs. To address such an issue and improve the recommendation performance of LLMs, we propose a novel framework, namely Retrieval-enhanced Large Language models (ReLLa) for recommendation tasks in both zero-shot and few-shot settings. For zero-shot recommendation, we perform semantic user behavior retrieval (SUBR) to improve the data quality of testing samples, which greatly reduces the difficulty for LLMs to extract the essential knowledge from user behavior sequences. As for few-shot recommendation, we further design retrieval-enhanced instruction tuning (ReiT) by adopting SUBR as a data augmentation technique for training samples. Specifically, we develop a mixed training dataset consisting of both the original data samples and their retrieval-enhanced counterparts. We conduct extensive experiments on a real-world public dataset (i.e., MovieLens-1M) to demonstrate the superiority of ReLLa compared with existing baseline models, as well as its capability for lifelong sequential behavior comprehension.

{{</citation>}}


### (105/113) How Expressive are Graph Neural Networks in Recommendation? (Xuheng Cai et al., 2023)

{{<citation>}}

Xuheng Cai, Lianghao Xia, Xubin Ren, Chao Huang. (2023)  
**How Expressive are Graph Neural Networks in Recommendation?**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs-SI, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.11127v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have demonstrated superior performance on various graph learning tasks, including recommendation, where they leverage user-item collaborative filtering signals in graphs. However, theoretical formulations of their capability are scarce, despite their empirical effectiveness in state-of-the-art recommender models. Recently, research has explored the expressiveness of GNNs in general, demonstrating that message passing GNNs are at most as powerful as the Weisfeiler-Lehman test, and that GNNs combined with random node initialization are universal. Nevertheless, the concept of "expressiveness" for GNNs remains vaguely defined. Most existing works adopt the graph isomorphism test as the metric of expressiveness, but this graph-level task may not effectively assess a model's ability in recommendation, where the objective is to distinguish nodes of different closeness. In this paper, we provide a comprehensive theoretical analysis of the expressiveness of GNNs in recommendation, considering three levels of expressiveness metrics: graph isomorphism (graph-level), node automorphism (node-level), and topological closeness (link-level). We propose the topological closeness metric to evaluate GNNs' ability to capture the structural distance between nodes, which aligns closely with the objective of recommendation. To validate the effectiveness of this new metric in evaluating recommendation performance, we introduce a learning-less GNN algorithm that is optimal on the new metric and can be optimal on the node-level metric with suitable modification. We conduct extensive experiments comparing the proposed algorithm against various types of state-of-the-art GNN models to explore the explainability of the new metric in the recommendation task. For reproducibility, implementation codes are available at https://github.com/HKUDS/GTE.

{{</citation>}}


## econ.GN (1)



### (106/113) From Mundane to Meaningful: AI's Influence on Work Dynamics -- evidence from ChatGPT and Stack Overflow (Quentin Gallea, 2023)

{{<citation>}}

Quentin Gallea. (2023)  
**From Mundane to Meaningful: AI's Influence on Work Dynamics -- evidence from ChatGPT and Stack Overflow**  

---
Primary Category: econ.GN  
Categories: cs-AI, econ-GN, econ.GN, q-fin-EC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.11302v1)  

---


**ABSTRACT**  
This paper illustrates how generative AI could give opportunities for big productivity gains but also opens up questions about the impact of these new powerful technologies on the way we work and share knowledge. More specifically, we explore how ChatGPT changed a fundamental aspect of coding: problem-solving. To do so, we exploit the effect of the sudden release of ChatGPT on the 30th of November 2022 on the usage of the largest online community for coders: Stack Overflow. Using quasi-experimental methods (Difference-in-Difference), we find a significant drop in the number of questions. In addition, the questions are better documented after the release of ChatGPT. Finally, we find evidence that the remaining questions are more complex. These findings suggest not only productivity gains but also a fundamental change in the way we work where routine inquiries are solved by AI allowing humans to focus on more complex tasks.

{{</citation>}}


## cs.RO (2)



### (107/113) ROSGPT_Vision: Commanding Robots Using Only Language Models' Prompts (Bilel Benjdira et al., 2023)

{{<citation>}}

Bilel Benjdira, Anis Koubaa, Anas M. Ali. (2023)  
**ROSGPT_Vision: Commanding Robots Using Only Language Models' Prompts**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.11236v2)  

---


**ABSTRACT**  
In this paper, we argue that the next generation of robots can be commanded using only Language Models' prompts. Every prompt interrogates separately a specific Robotic Modality via its Modality Language Model (MLM). A central Task Modality mediates the whole communication to execute the robotic mission via a Large Language Model (LLM). This paper gives this new robotic design pattern the name of: Prompting Robotic Modalities (PRM). Moreover, this paper applies this PRM design pattern in building a new robotic framework named ROSGPT_Vision. ROSGPT_Vision allows the execution of a robotic task using only two prompts: a Visual and an LLM prompt. The Visual Prompt extracts, in natural language, the visual semantic features related to the task under consideration (Visual Robotic Modality). Meanwhile, the LLM Prompt regulates the robotic reaction to the visual description (Task Modality). The framework automates all the mechanisms behind these two prompts. The framework enables the robot to address complex real-world scenarios by processing visual data, making informed decisions, and carrying out actions automatically. The framework comprises one generic vision module and two independent ROS nodes. As a test application, we used ROSGPT_Vision to develop CarMate, which monitors the driver's distraction on the roads and makes real-time vocal notifications to the driver. We showed how ROSGPT_Vision significantly reduced the development cost compared to traditional methods. We demonstrated how to improve the quality of the application by optimizing the prompting strategies, without delving into technical details. ROSGPT_Vision is shared with the community (link: https://github.com/bilel-bj/ROSGPT_Vision) to advance robotic research in this direction and to build more robotic frameworks that implement the PRM design pattern and enables controlling robots using only prompts.

{{</citation>}}


### (108/113) Mobility-Aware Computation Offloading for Swarm Robotics using Deep Reinforcement Learning (Xiucheng Wang et al., 2023)

{{<citation>}}

Xiucheng Wang, Hongzhi Guo. (2023)  
**Mobility-Aware Computation Offloading for Swarm Robotics using Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.11154v1)  

---


**ABSTRACT**  
Swarm robotics is envisioned to automate a large number of dirty, dangerous, and dull tasks. Robots have limited energy, computation capability, and communication resources. Therefore, current swarm robotics have a small number of robots, which can only provide limited spatio-temporal information. In this paper, we propose to leverage the mobile edge computing to alleviate the computation burden. We develop an effective solution based on a mobility-aware deep reinforcement learning model at the edge server side for computing scheduling and resource. Our results show that the proposed approach can meet delay requirements and guarantee computation precision by using minimum robot energy.

{{</citation>}}


## cs.DC (1)



### (109/113) A Deep Reinforcement Learning based Algorithm for Time and Cost Optimized Scaling of Serverless Applications (Anupama Mampage et al., 2023)

{{<citation>}}

Anupama Mampage, Shanika Karunasekera, Rajkumar Buyya. (2023)  
**A Deep Reinforcement Learning based Algorithm for Time and Cost Optimized Scaling of Serverless Applications**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.11209v1)  

---


**ABSTRACT**  
Serverless computing has gained a strong traction in the cloud computing community in recent years. Among the many benefits of this novel computing model, the rapid auto-scaling capability of user applications takes prominence. However, the offer of adhoc scaling of user deployments at function level introduces many complications to serverless systems. The added delay and failures in function request executions caused by the time consumed for dynamically creating new resources to suit function workloads, known as the cold-start delay, is one such very prevalent shortcoming. Maintaining idle resource pools to alleviate this issue often results in wasted resources from the cloud provider perspective. Existing solutions to address this limitation mostly focus on predicting and understanding function load levels in order to proactively create required resources. Although these solutions improve function performance, the lack of understanding on the overall system characteristics in making these scaling decisions often leads to the sub-optimal usage of system resources. Further, the multi-tenant nature of serverless systems requires a scalable solution adaptable for multiple co-existing applications, a limitation seen in most current solutions. In this paper, we introduce a novel multi-agent Deep Reinforcement Learning based intelligent solution for both horizontal and vertical scaling of function resources, based on a comprehensive understanding on both function and system requirements. Our solution elevates function performance reducing cold starts, while also offering the flexibility for optimizing resource maintenance cost to the service providers. Experiments conducted considering varying workload scenarios show improvements of up to 23% and 34% in terms of application latency and request failures, while also saving up to 45% in infrastructure cost for the service providers.

{{</citation>}}


## stat.ME (1)



### (110/113) NLP-based detection of systematic anomalies among the narratives of consumer complaints (Peiheng Gao et al., 2023)

{{<citation>}}

Peiheng Gao, Ning Sun, Xuefeng Wang, Chen Yang, Ričardas Zitikis. (2023)  
**NLP-based detection of systematic anomalies among the narratives of consumer complaints**  

---
Primary Category: stat.ME  
Categories: cs-CL, q-fin-RM, stat-ME, stat-ML, stat.ME  
Keywords: Financial, NLP  
[Paper Link](http://arxiv.org/abs/2308.11138v1)  

---


**ABSTRACT**  
We develop an NLP-based procedure for detecting systematic nonmeritorious consumer complaints, simply called systematic anomalies, among complaint narratives. While classification algorithms are used to detect pronounced anomalies, in the case of smaller and frequent systematic anomalies, the algorithms may falter due to a variety of reasons, including technical ones as well as natural limitations of human analysts. Therefore, as the next step after classification, we convert the complaint narratives into quantitative data, which are then analyzed using an algorithm for detecting systematic anomalies. We illustrate the entire procedure using complaint narratives from the Consumer Complaint Database of the Consumer Financial Protection Bureau.

{{</citation>}}


## cs.CY (1)



### (111/113) Is There Any Social Principle for LLM-Based Agents? (Jitao Bai et al., 2023)

{{<citation>}}

Jitao Bai, Simiao Zhang, Zhonghao Chen. (2023)  
**Is There Any Social Principle for LLM-Based Agents?**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-SI, cs.CY  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.11136v1)  

---


**ABSTRACT**  
Focus on Large Language Model based agents should involve more than "human-centered" alignment or application. We argue that more attention should be paid to the agent itself and discuss the potential of social sciences for agents.

{{</citation>}}


## quant-ph (2)



### (112/113) Development of a Novel Quantum Pre-processing Filter to Improve Image Classification Accuracy of Neural Network Models (Farina Riaz et al., 2023)

{{<citation>}}

Farina Riaz, Shahab Abdulla, Hajime Suzuki, Srinjoy Ganguly, Ravinesh C. Deo, Susan Hopkins. (2023)  
**Development of a Novel Quantum Pre-processing Filter to Improve Image Classification Accuracy of Neural Network Models**  

---
Primary Category: quant-ph  
Categories: cs-CV, cs-LG, quant-ph, quant-ph  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2308.11112v1)  

---


**ABSTRACT**  
This paper proposes a novel quantum pre-processing filter (QPF) to improve the image classification accuracy of neural network (NN) models. A simple four qubit quantum circuit that uses Y rotation gates for encoding and two controlled NOT gates for creating correlation among the qubits is applied as a feature extraction filter prior to passing data into the fully connected NN architecture. By applying the QPF approach, the results show that the image classification accuracy based on the MNIST (handwritten 10 digits) and the EMNIST (handwritten 47 class digits and letters) datasets can be improved, from 92.5% to 95.4% and from 68.9% to 75.9%, respectively. These improvements were obtained without introducing extra model parameters or optimizations in the machine learning process. However, tests performed on the developed QPF approach against a relatively complex GTSRB dataset with 43 distinct class real-life traffic sign images showed a degradation in the classification accuracy. Considering this result, further research into the understanding and the design of a more suitable quantum circuit approach for image classification neural networks could be explored utilizing the baseline method proposed in this paper.

{{</citation>}}


### (113/113) Explicability and Inexplicability in the Interpretation of Quantum Neural Networks (Lirandë Pira et al., 2023)

{{<citation>}}

Lirandë Pira, Chris Ferrie. (2023)  
**Explicability and Inexplicability in the Interpretation of Quantum Neural Networks**  

---
Primary Category: quant-ph  
Categories: cs-LG, quant-ph, quant-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.11098v1)  

---


**ABSTRACT**  
Interpretability of artificial intelligence (AI) methods, particularly deep neural networks, is of great interest due to the widespread use of AI-backed systems, which often have unexplainable behavior. The interpretability of such models is a crucial component of building trusted systems. Many methods exist to approach this problem, but they do not obviously generalize to the quantum setting. Here we explore the interpretability of quantum neural networks using local model-agnostic interpretability measures of quantum and classical neural networks. We introduce the concept of the band of inexplicability, representing the interpretable region in which data samples have no explanation, likely victims of inherently random quantum measurements. We see this as a step toward understanding how to build responsible and accountable quantum AI models.

{{</citation>}}
