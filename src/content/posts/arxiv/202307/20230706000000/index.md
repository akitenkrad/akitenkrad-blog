---
draft: false
title: "arXiv @ 2023.07.06"
date: 2023-07-06
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.06"
    identifier: arxiv_20230706
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (25)](#cslg-25)
- [cs.HC (2)](#cshc-2)
- [cs.CL (28)](#cscl-28)
- [eess.AS (1)](#eessas-1)
- [cs.DC (1)](#csdc-1)
- [cs.CV (23)](#cscv-23)
- [cs.IR (7)](#csir-7)
- [cs.CY (2)](#cscy-2)
- [cs.AI (2)](#csai-2)
- [cs.NE (1)](#csne-1)
- [cs.RO (3)](#csro-3)
- [eess.SP (2)](#eesssp-2)
- [cs.DB (2)](#csdb-2)
- [cs.IT (2)](#csit-2)
- [cs.SD (1)](#cssd-1)
- [eess.IV (2)](#eessiv-2)
- [cs.MA (1)](#csma-1)
- [cs.NI (1)](#csni-1)

## cs.LG (25)



### (1/106) ACDNet: Attention-guided Collaborative Decision Network for Effective Medication Recommendation (Jiacong Mi et al., 2023)

{{<citation>}}

Jiacong Mi, Yi Zu, Zhuoyuan Wang, Jieyue He. (2023)  
**ACDNet: Attention-guided Collaborative Decision Network for Effective Medication Recommendation**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2307.03332v1)  

---


**ABSTRACT**  
Medication recommendation using Electronic Health Records (EHR) is challenging due to complex medical data. Current approaches extract longitudinal information from patient EHR to personalize recommendations. However, existing models often lack sufficient patient representation and overlook the importance of considering the similarity between a patient's medication records and specific medicines. Therefore, an Attention-guided Collaborative Decision Network (ACDNet) for medication recommendation is proposed in this paper. Specifically, ACDNet utilizes attention mechanism and Transformer to effectively capture patient health conditions and medication records by modeling their historical visits at both global and local levels. ACDNet also employs a collaborative decision framework, utilizing the similarity between medication records and medicine representation to facilitate the recommendation process. The experimental results on two extensive medical datasets, MIMIC-III and MIMIC-IV, clearly demonstrate that ACDNet outperforms state-of-the-art models in terms of Jaccard, PR-AUC, and F1 score, reaffirming its superiority. Moreover, the ablation experiments provide solid evidence of the effectiveness of each module in ACDNet, validating their contribution to the overall performance. Furthermore, a detailed case study reinforces the effectiveness of ACDNet in medication recommendation based on EHR data, showcasing its practical value in real-world healthcare scenarios.

{{</citation>}}


### (2/106) Encoder-Decoder Networks for Self-Supervised Pretraining and Downstream Signal Bandwidth Regression on Digital Antenna Arrays (Rajib Bhattacharjea et al., 2023)

{{<citation>}}

Rajib Bhattacharjea, Nathan West. (2023)  
**Encoder-Decoder Networks for Self-Supervised Pretraining and Downstream Signal Bandwidth Regression on Digital Antenna Arrays**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.03327v1)  

---


**ABSTRACT**  
This work presents the first applications of self-supervised learning applied to data from digital antenna arrays. Encoder-decoder networks are pretrained on digital array data to perform a self-supervised noisy-reconstruction task called channel in-painting, in which the network infers the contents of array data that has been masked with zeros. The self-supervised step requires no human-labeled data. The encoder architecture and weights from pretraining are then transferred to a new network with a task-specific decoder, and the new network is trained on a small volume of labeled data. We show that pretraining on the unlabeled data allows the new network to perform the task of bandwidth regression on the digital array data better than an equivalent network that is trained on the same labeled data from random initialization.

{{</citation>}}


### (3/106) Assisting Clinical Decisions for Scarcely Available Treatment via Disentangled Latent Representation (Bing Xue et al., 2023)

{{<citation>}}

Bing Xue, Ahmed Sameh Said, Ziqi Xu, Hanyang Liu, Neel Shah, Hanqing Yang, Philip Payne, Chenyang Lu. (2023)  
**Assisting Clinical Decisions for Scarcely Available Treatment via Disentangled Latent Representation**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2307.03315v1)  

---


**ABSTRACT**  
Extracorporeal membrane oxygenation (ECMO) is an essential life-supporting modality for COVID-19 patients who are refractory to conventional therapies. However, the proper treatment decision has been the subject of significant debate and it remains controversial about who benefits from this scarcely available and technically complex treatment option. To support clinical decisions, it is a critical need to predict the treatment need and the potential treatment and no-treatment responses. Targeting this clinical challenge, we propose Treatment Variational AutoEncoder (TVAE), a novel approach for individualized treatment analysis. TVAE is specifically designed to address the modeling challenges like ECMO with strong treatment selection bias and scarce treatment cases. TVAE conceptualizes the treatment decision as a multi-scale problem. We model a patient's potential treatment assignment and the factual and counterfactual outcomes as part of their intrinsic characteristics that can be represented by a deep latent variable model. The factual and counterfactual prediction errors are alleviated via a reconstruction regularization scheme together with semi-supervision, and the selection bias and the scarcity of treatment cases are mitigated by the disentangled and distribution-matched latent space and the label-balancing generative strategy. We evaluate TVAE on two real-world COVID-19 datasets: an international dataset collected from 1651 hospitals across 63 countries, and a institutional dataset collected from 15 hospitals. The results show that TVAE outperforms state-of-the-art treatment effect models in predicting both the propensity scores and factual outcomes on heterogeneous COVID-19 datasets. Additional experiments also show TVAE outperforms the best existing models in individual treatment effect estimation on the synthesized IHDP benchmark dataset.

{{</citation>}}


### (4/106) TGRL: An Algorithm for Teacher Guided Reinforcement Learning (Idan Shenfeld et al., 2023)

{{<citation>}}

Idan Shenfeld, Zhang-Wei Hong, Aviv Tamar, Pulkit Agrawal. (2023)  
**TGRL: An Algorithm for Teacher Guided Reinforcement Learning**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.03186v1)  

---


**ABSTRACT**  
Learning from rewards (i.e., reinforcement learning or RL) and learning to imitate a teacher (i.e., teacher-student learning) are two established approaches for solving sequential decision-making problems. To combine the benefits of these different forms of learning, it is common to train a policy to maximize a combination of reinforcement and teacher-student learning objectives. However, without a principled method to balance these objectives, prior work used heuristics and problem-specific hyperparameter searches to balance the two objectives. We present a $\textit{principled}$ approach, along with an approximate implementation for $\textit{dynamically}$ and $\textit{automatically}$ balancing when to follow the teacher and when to use rewards. The main idea is to adjust the importance of teacher supervision by comparing the agent's performance to the counterfactual scenario of the agent learning without teacher supervision and only from rewards. If using teacher supervision improves performance, the importance of teacher supervision is increased and otherwise it is decreased. Our method, $\textit{Teacher Guided Reinforcement Learning}$ (TGRL), outperforms strong baselines across diverse domains without hyper-parameter tuning.

{{</citation>}}


### (5/106) Benchmarking Test-Time Adaptation against Distribution Shifts in Image Classification (Yongcan Yu et al., 2023)

{{<citation>}}

Yongcan Yu, Lijun Sheng, Ran He, Jian Liang. (2023)  
**Benchmarking Test-Time Adaptation against Distribution Shifts in Image Classification**  

---
Primary Category: cs.LG
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Image Classification, ImageNet  
[Paper Link](http://arxiv.org/abs/2307.03133v1)  

---


**ABSTRACT**  
Test-time adaptation (TTA) is a technique aimed at enhancing the generalization performance of models by leveraging unlabeled samples solely during prediction. Given the need for robustness in neural network systems when faced with distribution shifts, numerous TTA methods have recently been proposed. However, evaluating these methods is often done under different settings, such as varying distribution shifts, backbones, and designing scenarios, leading to a lack of consistent and fair benchmarks to validate their effectiveness. To address this issue, we present a benchmark that systematically evaluates 13 prominent TTA methods and their variants on five widely used image classification datasets: CIFAR-10-C, CIFAR-100-C, ImageNet-C, DomainNet, and Office-Home. These methods encompass a wide range of adaptation scenarios (e.g. online adaptation v.s. offline adaptation, instance adaptation v.s. batch adaptation v.s. domain adaptation). Furthermore, we explore the compatibility of different TTA methods with diverse network backbones. To implement this benchmark, we have developed a unified framework in PyTorch, which allows for consistent evaluation and comparison of the TTA methods across the different datasets and network architectures. By establishing this benchmark, we aim to provide researchers and practitioners with a reliable means of assessing and comparing the effectiveness of TTA methods in improving model robustness and generalization performance. Our code is available at https://github.com/yuyongcan/Benchmark-TTA.

{{</citation>}}


### (6/106) A Novel Site-Agnostic Multimodal Deep Learning Model to Identify Pro-Eating Disorder Content on Social Media (Jonathan Feldman, 2023)

{{<citation>}}

Jonathan Feldman. (2023)  
**A Novel Site-Agnostic Multimodal Deep Learning Model to Identify Pro-Eating Disorder Content on Social Media**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CL, cs-LG, cs-SI, cs.LG  
Keywords: BERT, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2307.06775v1)  

---


**ABSTRACT**  
Over the last decade, there has been a vast increase in eating disorder diagnoses and eating disorder-attributed deaths, reaching their zenith during the Covid-19 pandemic. This immense growth derived in part from the stressors of the pandemic but also from increased exposure to social media, which is rife with content that promotes eating disorders. Such content can induce eating disorders in viewers. This study aimed to create a multimodal deep learning model capable of determining whether a given social media post promotes eating disorders based on a combination of visual and textual data. A labeled dataset of Tweets was collected from Twitter, upon which twelve deep learning models were trained and tested. Based on model performance, the most effective deep learning model was the multimodal fusion of the RoBERTa natural language processing model and the MaxViT image classification model, attaining accuracy and F1 scores of 95.9% and 0.959 respectively. The RoBERTa and MaxViT fusion model, deployed to classify an unlabeled dataset of posts from the social media sites Tumblr and Reddit, generated similar classifications as previous research studies that did not employ artificial intelligence, showing that artificial intelligence can develop insights congruent to those of researchers. Additionally, the model was used to conduct a time-series analysis of yet unseen Tweets from eight Twitter hashtags, uncovering that the relative abundance of pro-eating disorder content has decreased drastically. However, since approximately 2018, pro-eating disorder content has either stopped its decline or risen once more in ampleness.

{{</citation>}}


### (7/106) A Hybrid End-to-End Spatio-Temporal Attention Neural Network with Graph-Smooth Signals for EEG Emotion Recognition (Shadi Sartipi et al., 2023)

{{<citation>}}

Shadi Sartipi, Mastaneh Torkamani-Azar, Mujdat Cetin. (2023)  
**A Hybrid End-to-End Spatio-Temporal Attention Neural Network with Graph-Smooth Signals for EEG Emotion Recognition**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Attention, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2307.03068v1)  

---


**ABSTRACT**  
Recently, physiological data such as electroencephalography (EEG) signals have attracted significant attention in affective computing. In this context, the main goal is to design an automated model that can assess emotional states. Lately, deep neural networks have shown promising performance in emotion recognition tasks. However, designing a deep architecture that can extract practical information from raw data is still a challenge. Here, we introduce a deep neural network that acquires interpretable physiological representations by a hybrid structure of spatio-temporal encoding and recurrent attention network blocks. Furthermore, a preprocessing step is applied to the raw data using graph signal processing tools to perform graph smoothing in the spatial domain. We demonstrate that our proposed architecture exceeds state-of-the-art results for emotion classification on the publicly available DEAP dataset. To explore the generality of the learned model, we also evaluate the performance of our architecture towards transfer learning (TL) by transferring the model parameters from a specific source to other target domains. Using DEAP as the source dataset, we demonstrate the effectiveness of our model in performing cross-modality TL and improving emotion classification accuracy on DREAMER and the Emotional English Word (EEWD) datasets, which involve EEG-based emotion classification tasks with different stimuli.

{{</citation>}}


### (8/106) Generalizing Backpropagation for Gradient-Based Interpretability (Kevin Du et al., 2023)

{{<citation>}}

Kevin Du, Lucas Torroba Hennigen, Niklas Stoehr, Alexander Warstadt, Ryan Cotterell. (2023)  
**Generalizing Backpropagation for Gradient-Based Interpretability**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2307.03056v1)  

---


**ABSTRACT**  
Many popular feature-attribution methods for interpreting deep neural networks rely on computing the gradients of a model's output with respect to its inputs. While these methods can indicate which input features may be important for the model's prediction, they reveal little about the inner workings of the model itself. In this paper, we observe that the gradient computation of a model is a special case of a more general formulation using semirings. This observation allows us to generalize the backpropagation algorithm to efficiently compute other interpretable statistics about the gradient graph of a neural network, such as the highest-weighted path and entropy. We implement this generalized algorithm, evaluate it on synthetic datasets to better understand the statistics it computes, and apply it to study BERT's behavior on the subject-verb number agreement task (SVA). With this method, we (a) validate that the amount of gradient flow through a component of a model reflects its importance to a prediction and (b) for SVA, identify which pathways of the self-attention mechanism are most important.

{{</citation>}}


### (9/106) Origin-Destination Travel Time Oracle for Map-based Services (Yan Lin et al., 2023)

{{<citation>}}

Yan Lin, Huaiyu Wan, Jilin Hu, Shengnan Guo, Bin Yang, Youfang Lin, Christian S. Jensen. (2023)  
**Origin-Destination Travel Time Oracle for Map-based Services**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.03048v1)  

---


**ABSTRACT**  
Given an origin (O), a destination (D), and a departure time (T), an Origin-Destination (OD) travel time oracle~(ODT-Oracle) returns an estimate of the time it takes to travel from O to D when departing at T. ODT-Oracles serve important purposes in map-based services. To enable the construction of such oracles, we provide a travel-time estimation (TTE) solution that leverages historical trajectories to estimate time-varying travel times for OD pairs.   The problem is complicated by the fact that multiple historical trajectories with different travel times may connect an OD pair, while trajectories may vary from one another. To solve the problem, it is crucial to remove outlier trajectories when doing travel time estimation for future queries.   We propose a novel, two-stage framework called Diffusion-based Origin-destination Travel Time Estimation (DOT), that solves the problem. First, DOT employs a conditioned Pixelated Trajectories (PiT) denoiser that enables building a diffusion-based PiT inference process by learning correlations between OD pairs and historical trajectories. Specifically, given an OD pair and a departure time, we aim to infer a PiT. Next, DOT encompasses a Masked Vision Transformer~(MViT) that effectively and efficiently estimates a travel time based on the inferred PiT. We report on extensive experiments on two real-world datasets that offer evidence that DOT is capable of outperforming baseline methods in terms of accuracy, scalability, and explainability.

{{</citation>}}


### (10/106) FITS: Modeling Time Series with $10k$ Parameters (Zhijian Xu et al., 2023)

{{<citation>}}

Zhijian Xu, Ailing Zeng, Qiang Xu. (2023)  
**FITS: Modeling Time Series with $10k$ Parameters**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.03756v1)  

---


**ABSTRACT**  
In this paper, we introduce FITS, a lightweight yet powerful model for time series analysis. Unlike existing models that directly process raw time-domain data, FITS operates on the principle that time series can be manipulated through interpolation in the complex frequency domain. By discarding high-frequency components with negligible impact on time series data, FITS achieves performance comparable to state-of-the-art models for time series forecasting and anomaly detection tasks, while having a remarkably compact size of only approximately $10k$ parameters. Such a lightweight model can be easily trained and deployed in edge devices, creating opportunities for various applications. The anonymous code repo is available in: \url{https://anonymous.4open.science/r/FITS}

{{</citation>}}


### (11/106) Improving Retrieval-Augmented Large Language Models via Data Importance Learning (Xiaozhong Lyu et al., 2023)

{{<citation>}}

Xiaozhong Lyu, Stefan Grafberger, Samantha Biegel, Shaopeng Wei, Meng Cao, Sebastian Schelter, Ce Zhang. (2023)  
**Improving Retrieval-Augmented Large Language Models via Data Importance Learning**  

---
Primary Category: cs.LG
Categories: cs-CL, cs-IR, cs-LG, cs.LG  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2307.03027v1)  

---


**ABSTRACT**  
Retrieval augmentation enables large language models to take advantage of external knowledge, for example on tasks like question answering and data imputation. However, the performance of such retrieval-augmented models is limited by the data quality of their underlying retrieval corpus. In this paper, we propose an algorithm based on multilinear extension for evaluating the data importance of retrieved data points. There are exponentially many terms in the multilinear extension, and one key contribution of this paper is a polynomial time algorithm that computes exactly, given a retrieval-augmented model with an additive utility function and a validation set, the data importance of data points in the retrieval corpus using the multilinear extension of the model's utility function. We further proposed an even more efficient ({\epsilon}, {\delta})-approximation algorithm. Our experimental results illustrate that we can enhance the performance of large language models by only pruning or reweighting the retrieval corpus, without requiring further training. For some tasks, this even allows a small model (e.g., GPT-JT), augmented with a search engine API, to outperform GPT-3.5 (without retrieval augmentation). Moreover, we show that weights based on multilinear extension can be computed efficiently in practice (e.g., in less than ten minutes for a corpus with 100 million elements).

{{</citation>}}


### (12/106) Improving the Efficiency of Human-in-the-Loop Systems: Adding Artificial to Human Experts (Johannes Jakubik et al., 2023)

{{<citation>}}

Johannes Jakubik, Daniel Weber, Patrick Hemmer, Michael Vössing, Gerhard Satzger. (2023)  
**Improving the Efficiency of Human-in-the-Loop Systems: Adding Artificial to Human Experts**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.03003v2)  

---


**ABSTRACT**  
Information systems increasingly leverage artificial intelligence (AI) and machine learning (ML) to generate value from vast amounts of data. However, ML models are imperfect and can generate incorrect classifications. Hence, human-in-the-loop (HITL) extensions to ML models add a human review for instances that are difficult to classify. This study argues that continuously relying on human experts to handle difficult model classifications leads to a strong increase in human effort, which strains limited resources. To address this issue, we propose a hybrid system that creates artificial experts that learn to classify data instances from unknown classes previously reviewed by human experts. Our hybrid system assesses which artificial expert is suitable for classifying an instance from an unknown class and automatically assigns it. Over time, this reduces human effort and increases the efficiency of the system. Our experiments demonstrate that our approach outperforms traditional HITL systems for several benchmarks on image classification.

{{</citation>}}


### (13/106) ContainerGym: A Real-World Reinforcement Learning Benchmark for Resource Allocation (Abhijeet Pendyala et al., 2023)

{{<citation>}}

Abhijeet Pendyala, Justin Dettmer, Tobias Glasmachers, Asma Atamna. (2023)  
**ContainerGym: A Real-World Reinforcement Learning Benchmark for Resource Allocation**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02991v1)  

---


**ABSTRACT**  
We present ContainerGym, a benchmark for reinforcement learning inspired by a real-world industrial resource allocation task. The proposed benchmark encodes a range of challenges commonly encountered in real-world sequential decision making problems, such as uncertainty. It can be configured to instantiate problems of varying degrees of difficulty, e.g., in terms of variable dimensionality. Our benchmark differs from other reinforcement learning benchmarks, including the ones aiming to encode real-world difficulties, in that it is directly derived from a real-world industrial problem, which underwent minimal simplification and streamlining. It is sufficiently versatile to evaluate reinforcement learning algorithms on any real-world problem that fits our resource allocation framework. We provide results of standard baseline methods. Going beyond the usual training reward curves, our results and the statistical tools used to interpret them allow to highlight interesting limitations of well-known deep reinforcement learning algorithms, namely PPO, TRPO and DQN.

{{</citation>}}


### (14/106) Transfer Learning for the Efficient Detection of COVID-19 from Smartphone Audio Data (Mattia Giovanni Campana et al., 2023)

{{<citation>}}

Mattia Giovanni Campana, Franca Delmastro, Elena Pagani. (2023)  
**Transfer Learning for the Efficient Detection of COVID-19 from Smartphone Audio Data**  

---
Primary Category: cs.LG
Categories: cs-LG, cs-SD, cs.LG, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02975v1)  

---


**ABSTRACT**  
Disease detection from smartphone data represents an open research challenge in mobile health (m-health) systems. COVID-19 and its respiratory symptoms are an important case study in this area and their early detection is a potential real instrument to counteract the pandemic situation. The efficacy of this solution mainly depends on the performances of AI algorithms applied to the collected data and their possible implementation directly on the users' mobile devices. Considering these issues, and the limited amount of available data, in this paper we present the experimental evaluation of 3 different deep learning models, compared also with hand-crafted features, and of two main approaches of transfer learning in the considered scenario: both feature extraction and fine-tuning. Specifically, we considered VGGish, YAMNET, and L\textsuperscript{3}-Net (including 12 different configurations) evaluated through user-independent experiments on 4 different datasets (13,447 samples in total). Results clearly show the advantages of L\textsuperscript{3}-Net in all the experimental settings as it overcomes the other solutions by 12.3\% in terms of Precision-Recall AUC as features extractor, and by 10\% when the model is fine-tuned. Moreover, we note that to fine-tune only the fully-connected layers of the pre-trained models generally leads to worse performances, with an average drop of 6.6\% with respect to feature extraction. %highlighting the need for further investigations. Finally, we evaluate the memory footprints of the different models for their possible applications on commercial mobile devices.

{{</citation>}}


### (15/106) Pruning vs Quantization: Which is Better? (Andrey Kuzmin et al., 2023)

{{<citation>}}

Andrey Kuzmin, Markus Nagel, Mart van Baalen, Arash Behboodi, Tijmen Blankevoort. (2023)  
**Pruning vs Quantization: Which is Better?**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Pruning, Quantization  
[Paper Link](http://arxiv.org/abs/2307.02973v1)  

---


**ABSTRACT**  
Neural network pruning and quantization techniques are almost as old as neural networks themselves. However, to date only ad-hoc comparisons between the two have been published. In this paper, we set out to answer the question on which is better: neural network quantization or pruning? By answering this question, we hope to inform design decisions made on neural network hardware going forward. We provide an extensive comparison between the two techniques for compressing deep neural networks. First, we give an analytical comparison of expected quantization and pruning error for general data distributions. Then, we provide lower bounds for the per-layer pruning and quantization error in trained networks, and compare these to empirical error after optimization. Finally, we provide an extensive experimental comparison for training 8 large-scale models on 3 tasks. Our results show that in most cases quantization outperforms pruning. Only in some scenarios with very high compression ratio, pruning might be beneficial from an accuracy standpoint.

{{</citation>}}


### (16/106) When No-Rejection Learning is Optimal for Regression with Rejection (Xiaocheng Li et al., 2023)

{{<citation>}}

Xiaocheng Li, Shang Liu, Chunlin Sun, Hanzhao Wang. (2023)  
**When No-Rejection Learning is Optimal for Regression with Rejection**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02932v1)  

---


**ABSTRACT**  
Learning with rejection is a prototypical model for studying the interaction between humans and AI on prediction tasks. The model has two components, a predictor and a rejector. Upon the arrival of a sample, the rejector first decides whether to accept it; if accepted, the predictor fulfills the prediction task, and if rejected, the prediction will be deferred to humans. The learning problem requires learning a predictor and a rejector simultaneously. This changes the structure of the conventional loss function and often results in non-convexity and inconsistency issues. For the classification with rejection problem, several works develop surrogate losses for the jointly learning with provable consistency guarantees; in parallel, there has been less work for the regression counterpart. We study the regression with rejection (RwR) problem and investigate the no-rejection learning strategy which treats the RwR problem as a standard regression task to learn the predictor. We establish that the suboptimality of the no-rejection learning strategy observed in the literature can be mitigated by enlarging the function class of the predictor. Then we introduce the truncated loss to single out the learning for the predictor and we show that a consistent surrogate property can be established for the predictor individually in an easier way than for the predictor and the rejector jointly. Our findings advocate for a two-step learning procedure that first uses all the data to learn the predictor and then calibrates the prediction loss for the rejector. It is better aligned with the common intuition that more data samples will lead to a better predictor and it calls for more efforts on a better design of calibration algorithms for learning the rejector. While our discussions mainly focus on the regression problem, the theoretical results and insights generalize to the classification problem as well.

{{</citation>}}


### (17/106) Free Bits: Latency Optimization of Mixed-Precision Quantized Neural Networks on the Edge (Georg Rutishauser et al., 2023)

{{<citation>}}

Georg Rutishauser, Francesco Conti, Luca Benini. (2023)  
**Free Bits: Latency Optimization of Mixed-Precision Quantized Neural Networks on the Edge**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.02894v1)  

---


**ABSTRACT**  
Mixed-precision quantization, where a deep neural network's layers are quantized to different precisions, offers the opportunity to optimize the trade-offs between model size, latency, and statistical accuracy beyond what can be achieved with homogeneous-bit-width quantization. To navigate the intractable search space of mixed-precision configurations for a given network, this paper proposes a hybrid search methodology. It consists of a hardware-agnostic differentiable search algorithm followed by a hardware-aware heuristic optimization to find mixed-precision configurations latency-optimized for a specific hardware target. We evaluate our algorithm on MobileNetV1 and MobileNetV2 and deploy the resulting networks on a family of multi-core RISC-V microcontroller platforms with different hardware characteristics. We achieve up to 28.6% reduction of end-to-end latency compared to an 8-bit model at a negligible accuracy drop from a full-precision baseline on the 1000-class ImageNet dataset. We demonstrate speedups relative to an 8-bit baseline, even on systems with no hardware support for sub-byte arithmetic at negligible accuracy drop. Furthermore, we show the superiority of our approach with respect to differentiable search targeting reduced binary operation counts as a proxy for latency.

{{</citation>}}


### (18/106) BaBE: Enhancing Fairness via Estimation of Latent Explaining Variables (Ruta Binkyte et al., 2023)

{{<citation>}}

Ruta Binkyte, Daniele Gorla, Catuscia Palamidessi. (2023)  
**BaBE: Enhancing Fairness via Estimation of Latent Explaining Variables**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.02891v1)  

---


**ABSTRACT**  
We consider the problem of unfair discrimination between two groups and propose a pre-processing method to achieve fairness. Corrective methods like statistical parity usually lead to bad accuracy and do not really achieve fairness in situations where there is a correlation between the sensitive attribute S and the legitimate attribute E (explanatory variable) that should determine the decision. To overcome these drawbacks, other notions of fairness have been proposed, in particular, conditional statistical parity and equal opportunity. However, E is often not directly observable in the data, i.e., it is a latent variable. We may observe some other variable Z representing E, but the problem is that Z may also be affected by S, hence Z itself can be biased. To deal with this problem, we propose BaBE (Bayesian Bias Elimination), an approach based on a combination of Bayes inference and the Expectation-Maximization method, to estimate the most likely value of E for a given Z for each group. The decision can then be based directly on the estimated E. We show, by experiments on synthetic and real data sets, that our approach provides a good level of fairness as well as high accuracy.

{{</citation>}}


### (19/106) Provably Efficient Iterated CVaR Reinforcement Learning with Function Approximation (Yu Chen et al., 2023)

{{<citation>}}

Yu Chen, Yihan Du, Pihe Hu, Siwei Wang, Desheng Wu, Longbo Huang. (2023)  
**Provably Efficient Iterated CVaR Reinforcement Learning with Function Approximation**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02842v1)  

---


**ABSTRACT**  
Risk-sensitive reinforcement learning (RL) aims to optimize policies that balance the expected reward and risk. In this paper, we investigate a novel risk-sensitive RL formulation with an Iterated Conditional Value-at-Risk (CVaR) objective under linear and general function approximations. This new formulation, named ICVaR-RL with function approximation, provides a principled way to guarantee safety at each decision step. For ICVaR-RL with linear function approximation, we propose a computationally efficient algorithm ICVaR-L, which achieves an $\widetilde{O}(\sqrt{\alpha^{-(H+1)}(d^2H^4+dH^6)K})$ regret, where $\alpha$ is the risk level, $d$ is the dimension of state-action features, $H$ is the length of each episode, and $K$ is the number of episodes. We also establish a matching lower bound $\Omega(\sqrt{\alpha^{-(H-1)}d^2K})$ to validate the optimality of ICVaR-L with respect to $d$ and $K$. For ICVaR-RL with general function approximation, we propose algorithm ICVaR-G, which achieves an $\widetilde{O}(\sqrt{\alpha^{-(H+1)}DH^4K})$ regret, where $D$ is a dimensional parameter that depends on the eluder dimension and covering number. Furthermore, our analysis provides several novel techniques for risk-sensitive RL, including an efficient approximation of the CVaR operator, a new ridge regression with CVaR-adapted features, and a refined elliptical potential lemma.

{{</citation>}}


### (20/106) Policy Contrastive Imitation Learning (Jialei Huang et al., 2023)

{{<citation>}}

Jialei Huang, Zhaoheng Yin, Yingdong Hu, Yang Gao. (2023)  
**Policy Contrastive Imitation Learning**  

---
Primary Category: cs.LG
Categories: cs-LG, cs-RO, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02829v1)  

---


**ABSTRACT**  
Adversarial imitation learning (AIL) is a popular method that has recently achieved much success. However, the performance of AIL is still unsatisfactory on the more challenging tasks. We find that one of the major reasons is due to the low quality of AIL discriminator representation. Since the AIL discriminator is trained via binary classification that does not necessarily discriminate the policy from the expert in a meaningful way, the resulting reward might not be meaningful either. We propose a new method called Policy Contrastive Imitation Learning (PCIL) to resolve this issue. PCIL learns a contrastive representation space by anchoring on different policies and generates a smooth cosine-similarity-based reward. Our proposed representation learning objective can be viewed as a stronger version of the AIL objective and provide a more meaningful comparison between the agent and the policy. From a theoretical perspective, we show the validity of our method using the apprenticeship learning framework. Furthermore, our empirical evaluation on the DeepMind Control suite demonstrates that PCIL can achieve state-of-the-art performance. Finally, qualitative results suggest that PCIL builds a smoother and more meaningful representation space for imitation learning.

{{</citation>}}


### (21/106) CPDG: A Contrastive Pre-Training Method for Dynamic Graph Neural Networks (Yuanchen Bei et al., 2023)

{{<citation>}}

Yuanchen Bei, Hao Xu, Sheng Zhou, Huixuan Chi, Mengdi Zhang, Zhao Li, Jiajun Bu. (2023)  
**CPDG: A Contrastive Pre-Training Method for Dynamic Graph Neural Networks**  

---
Primary Category: cs.LG
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.02813v1)  

---


**ABSTRACT**  
Dynamic graph data mining has gained popularity in recent years due to the rich information contained in dynamic graphs and their widespread use in the real world. Despite the advances in dynamic graph neural networks (DGNNs), the rich information and diverse downstream tasks have posed significant difficulties for the practical application of DGNNs in industrial scenarios. To this end, in this paper, we propose to address them by pre-training and present the Contrastive Pre-Training Method for Dynamic Graph Neural Networks (CPDG). CPDG tackles the challenges of pre-training for DGNNs, including generalization and long-short term modeling capability, through a flexible structural-temporal subgraph sampler along with structural-temporal contrastive pre-training schemes. Extensive experiments conducted on both large-scale research and industrial dynamic graph datasets show that CPDG outperforms existing methods in dynamic graph pre-training for various downstream tasks under three transfer settings.

{{</citation>}}


### (22/106) Offline Reinforcement Learning with Imbalanced Datasets (Li Jiang et al., 2023)

{{<citation>}}

Li Jiang, Sijie Chen, Jielin Qiu, Haoran Xu, Wai Kin Chan, Zhao Ding. (2023)  
**Offline Reinforcement Learning with Imbalanced Datasets**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02752v1)  

---


**ABSTRACT**  
The prevalent use of benchmarks in current offline reinforcement learning (RL) research has led to a neglect of the imbalance of real-world dataset distributions in the development of models. The real-world offline RL dataset is often imbalanced over the state space due to the challenge of exploration or safety considerations. In this paper, we specify properties of imbalanced datasets in offline RL, where the state coverage follows a power law distribution characterized by skewed policies. Theoretically and empirically, we show that typically offline RL methods based on distributional constraints, such as conservative Q-learning (CQL), are ineffective in extracting policies under the imbalanced dataset. Inspired by natural intelligence, we propose a novel offline RL method that utilizes the augmentation of CQL with a retrieval process to recall past related experiences, effectively alleviating the challenges posed by imbalanced datasets. We evaluate our method on several tasks in the context of imbalanced datasets with varying levels of imbalance, utilizing the variant of D4RL. Empirical results demonstrate the superiority of our method over other baselines.

{{</citation>}}


### (23/106) Evaluating the Evaluators: Are Current Few-Shot Learning Benchmarks Fit for Purpose? (Luísa Shimabucoro et al., 2023)

{{<citation>}}

Luísa Shimabucoro, Timothy Hospedales, Henry Gouk. (2023)  
**Evaluating the Evaluators: Are Current Few-Shot Learning Benchmarks Fit for Purpose?**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.02732v1)  

---


**ABSTRACT**  
Numerous benchmarks for Few-Shot Learning have been proposed in the last decade. However all of these benchmarks focus on performance averaged over many tasks, and the question of how to reliably evaluate and tune models trained for individual tasks in this regime has not been addressed. This paper presents the first investigation into task-level evaluation -- a fundamental step when deploying a model. We measure the accuracy of performance estimators in the few-shot setting, consider strategies for model selection, and examine the reasons for the failure of evaluators usually thought of as being robust. We conclude that cross-validation with a low number of folds is the best choice for directly estimating the performance of a model, whereas using bootstrapping or cross validation with a large number of folds is better for model selection purposes. Overall, we find that existing benchmarks for few-shot learning are not designed in such a way that one can get a reliable picture of how effectively methods can be used on individual tasks.

{{</citation>}}


### (24/106) Hierarchical Empowerment: Towards Tractable Empowerment-Based Skill-Learning (Andrew Levy et al., 2023)

{{<citation>}}

Andrew Levy, Sreehari Rammohan, Alessandro Allievi, Scott Niekum, George Konidaris. (2023)  
**Hierarchical Empowerment: Towards Tractable Empowerment-Based Skill-Learning**  

---
Primary Category: cs.LG
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02728v1)  

---


**ABSTRACT**  
General purpose agents will require large repertoires of skills. Empowerment -- the maximum mutual information between skills and the states -- provides a pathway for learning large collections of distinct skills, but mutual information is difficult to optimize. We introduce a new framework, Hierarchical Empowerment, that makes computing empowerment more tractable by integrating concepts from Goal-Conditioned Hierarchical Reinforcement Learning. Our framework makes two specific contributions. First, we introduce a new variational lower bound on mutual information that can be used to compute empowerment over short horizons. Second, we introduce a hierarchical architecture for computing empowerment over exponentially longer time scales. We verify the contributions of the framework in a series of simulated robotics tasks. In a popular ant navigation domain, our four level agents are able to learn skills that cover a surface area over two orders of magnitude larger than prior work.

{{</citation>}}


### (25/106) Multi-Similarity Contrastive Learning (Emily Mu et al., 2023)

{{<citation>}}

Emily Mu, John Guttag, Maggie Makar. (2023)  
**Multi-Similarity Contrastive Learning**  

---
Primary Category: cs.LG
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.02712v1)  

---


**ABSTRACT**  
Given a similarity metric, contrastive methods learn a representation in which examples that are similar are pushed together and examples that are dissimilar are pulled apart. Contrastive learning techniques have been utilized extensively to learn representations for tasks ranging from image classification to caption generation. However, existing contrastive learning approaches can fail to generalize because they do not take into account the possibility of different similarity relations. In this paper, we propose a novel multi-similarity contrastive loss (MSCon), that learns generalizable embeddings by jointly utilizing supervision from multiple metrics of similarity. Our method automatically learns contrastive similarity weightings based on the uncertainty in the corresponding similarity, down-weighting uncertain tasks and leading to better out-of-domain generalization to new tasks. We show empirically that networks trained with MSCon outperform state-of-the-art baselines on in-domain and out-of-domain settings.

{{</citation>}}


## cs.HC (2)



### (26/106) Artistic Strategies to Guide Neural Networks (Varvara Guljajeva et al., 2023)

{{<citation>}}

Varvara Guljajeva, Mar Canet Sola, Isaac Joseph Clarke. (2023)  
**Artistic Strategies to Guide Neural Networks**  

---
Primary Category: cs.HC
Categories: I-2-0; I-2-m, cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.07521v1)  

---


**ABSTRACT**  
Artificial Intelligence is present in the generation and distribution of culture. How do artists exploit neural networks? What impact do these algorithms have on artistic practice? Through a practice-based research methodology, this paper explores the potentials and limits of current AI technology, more precisely deep neural networks, in the context of image, text, form and translation of semiotic spaces. In a relatively short time, the generation of high-resolution images and 3D objects has been achieved. There are models, like CLIP and text2mesh, that do not need the same kind of media input as the output; we call them translation models. Such a twist contributes toward creativity arousal, which manifests itself in art practice and feeds back to the developers' pipeline. Yet again, we see how artworks act as catalysts for technology development. Those creative scenarios and processes are enabled not solely by AI models, but by the hard work behind implementing these new technologies. AI does not create a 'push-a-button' masterpiece but requires a deep understanding of the technology behind it, and a creative and critical mindset. Thus, AI opens new avenues for inspiration and offers novel tool sets, and yet again the question of authorship is asked.

{{</citation>}}


### (27/106) BrickPal: Augmented Reality-based Assembly Instructions for Brick Models (Yao Shi et al., 2023)

{{<citation>}}

Yao Shi, Xiaofeng Zhang, Ran zhang, Zhou Yang, Xiao Tang, Hongni Ye, Yi Wu. (2023)  
**BrickPal: Augmented Reality-based Assembly Instructions for Brick Models**  

---
Primary Category: cs.HC
Categories: cs-AI, cs-HC, cs.HC  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.03162v1)  

---


**ABSTRACT**  
The assembly instruction is a mandatory component of Lego-like brick sets.The conventional production of assembly instructions requires a considerable amount of manual fine-tuning, which is intractable for casual users and customized brick sets.Moreover, the traditional paper-based instructions lack expressiveness and interactivity.To tackle the two problems above, we present BrickPal, an augmented reality-based system, which visualizes assembly instructions in an augmented reality head-mounted display. It utilizes Natural Language Processing (NLP) techniques to generate plausible assembly sequences, and provide real-time guidance in the AR headset.Our user study demonstrates BrickPal's effectiveness at assisting users in brick assembly compared to traditional assembly methods. Additionally, the NLP algorithm-generated assembly sequences achieve the same usability with manually adapted sequences.

{{</citation>}}


## cs.CL (28)



### (28/106) BiPhone: Modeling Inter Language Phonetic Influences in Text (Abhirut Gupta et al., 2023)

{{<citation>}}

Abhirut Gupta, Ananya B. Sai, Richard Sproat, Yuri Vasilevski, James S. Ren, Ambarish Jash, Sukhdeep S. Sodhi, Aravindan Raghuveer. (2023)  
**BiPhone: Modeling Inter Language Phonetic Influences in Text**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: GLUE, SuperGLUE  
[Paper Link](http://arxiv.org/abs/2307.03322v1)  

---


**ABSTRACT**  
A large number of people are forced to use the Web in a language they have low literacy in due to technology asymmetries. Written text in the second language (L2) from such users often contains a large number of errors that are influenced by their native language (L1). We propose a method to mine phoneme confusions (sounds in L2 that an L1 speaker is likely to conflate) for pairs of L1 and L2. These confusions are then plugged into a generative model (Bi-Phone) for synthetically producing corrupted L2 text. Through human evaluations, we show that Bi-Phone generates plausible corruptions that differ across L1s and also have widespread coverage on the Web. We also corrupt the popular language understanding benchmark SuperGLUE with our technique (FunGLUE for Phonetically Noised GLUE) and show that SoTA language understating models perform poorly. We also introduce a new phoneme prediction pre-training task which helps byte models to recover performance close to SuperGLUE. Finally, we also release the FunGLUE benchmark to promote further research in phonetically robust language models. To the best of our knowledge, FunGLUE is the first benchmark to introduce L1-L2 interactions in text.

{{</citation>}}


### (29/106) Covering Uncommon Ground: Gap-Focused Question Generation for Answer Assessment (Roni Rabin et al., 2023)

{{<citation>}}

Roni Rabin, Alexandre Djerbetian, Roee Engelberg, Lidan Hackmon, Gal Elidan, Reut Tsarfaty, Amir Globerson. (2023)  
**Covering Uncommon Ground: Gap-Focused Question Generation for Answer Assessment**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Question Generation  
[Paper Link](http://arxiv.org/abs/2307.03319v1)  

---


**ABSTRACT**  
Human communication often involves information gaps between the interlocutors. For example, in an educational dialogue, a student often provides an answer that is incomplete, and there is a gap between this answer and the perfect one expected by the teacher. Successful dialogue then hinges on the teacher asking about this gap in an effective manner, thus creating a rich and interactive educational experience. We focus on the problem of generating such gap-focused questions (GFQs) automatically. We define the task, highlight key desired aspects of a good GFQ, and propose a model that satisfies these. Finally, we provide an evaluation by human annotators of our generated questions compared against human generated ones, demonstrating competitive performance.

{{</citation>}}


### (30/106) InfoSync: Information Synchronization across Multilingual Semi-structured Tables (Siddharth Khincha et al., 2023)

{{<citation>}}

Siddharth Khincha, Chelsi Jain, Vivek Gupta, Tushar Kataria, Shuo Zhang. (2023)  
**InfoSync: Information Synchronization across Multilingual Semi-structured Tables**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-CY, cs-IR, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2307.03313v1)  

---


**ABSTRACT**  
Information Synchronization of semi-structured data across languages is challenging. For instance, Wikipedia tables in one language should be synchronized across languages. To address this problem, we introduce a new dataset InfoSyncC and a two-step method for tabular synchronization. InfoSync contains 100K entity-centric tables (Wikipedia Infoboxes) across 14 languages, of which a subset (3.5K pairs) are manually annotated. The proposed method includes 1) Information Alignment to map rows and 2) Information Update for updating missing/outdated information for aligned tables across multilingual tables. When evaluated on InfoSync, information alignment achieves an F1 score of 87.91 (en <-> non-en). To evaluate information updation, we perform human-assisted Wikipedia edits on Infoboxes for 603 table pairs. Our approach obtains an acceptance rate of 77.28% on Wikipedia, showing the effectiveness of the proposed method.

{{</citation>}}


### (31/106) S2vNTM: Semi-supervised vMF Neural Topic Modeling (Weijie Xu et al., 2023)

{{<citation>}}

Weijie Xu, Jay Desai, Srinivasan Sengamedu, Xiaoyu Jiang, Francis Iannacci. (2023)  
**S2vNTM: Semi-supervised vMF Neural Topic Modeling**  

---
Primary Category: cs.CL
Categories: 68T50, cs-AI, cs-CL, cs.CL  
Keywords: Semi-Supervised, Topic Model, Topic Modeling  
[Paper Link](http://arxiv.org/abs/2307.04804v1)  

---


**ABSTRACT**  
Language model based methods are powerful techniques for text classification. However, the models have several shortcomings. (1) It is difficult to integrate human knowledge such as keywords. (2) It needs a lot of resources to train the models. (3) It relied on large text data to pretrain. In this paper, we propose Semi-Supervised vMF Neural Topic Modeling (S2vNTM) to overcome these difficulties. S2vNTM takes a few seed keywords as input for topics. S2vNTM leverages the pattern of keywords to identify potential topics, as well as optimize the quality of topics' keywords sets. Across a variety of datasets, S2vNTM outperforms existing semi-supervised topic modeling methods in classification accuracy with limited keywords provided. S2vNTM is at least twice as fast as baselines.

{{</citation>}}


### (32/106) Lost in the Middle: How Language Models Use Long Contexts (Nelson F. Liu et al., 2023)

{{<citation>}}

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. (2023)  
**Lost in the Middle: How Language Models Use Long Contexts**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.03172v1)  

---


**ABSTRACT**  
While recent language models have the ability to take long contexts as input, relatively little is known about how well the language models use longer context. We analyze language model performance on two tasks that require identifying relevant information within their input contexts: multi-document question answering and key-value retrieval. We find that performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts. Furthermore, performance substantially decreases as the input context grows longer, even for explicitly long-context models. Our analysis provides a better understanding of how language models use their input context and provides new evaluation protocols for future long-context models.

{{</citation>}}


### (33/106) Focused Transformer: Contrastive Training for Context Scaling (Szymon Tworkowski et al., 2023)

{{<citation>}}

Szymon Tworkowski, Konrad Staniszewski, Mikołaj Pacek, Yuhuai Wu, Henryk Michalewski, Piotr Miłoś. (2023)  
**Focused Transformer: Contrastive Training for Context Scaling**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Contrastive Training, LLaMA, Transformer  
[Paper Link](http://arxiv.org/abs/2307.03170v1)  

---


**ABSTRACT**  
Large language models have an exceptional capability to incorporate new information in a contextual manner. However, the full potential of such an approach is often restrained due to a limitation in the effective context length. One solution to this issue is to endow an attention layer with access to an external memory, which comprises of (key, value) pairs. Yet, as the number of documents increases, the proportion of relevant keys to irrelevant ones decreases, leading the model to focus more on the irrelevant keys. We identify a significant challenge, dubbed the distraction issue, where keys linked to different semantic values might overlap, making them hard to distinguish. To tackle this problem, we introduce the Focused Transformer (FoT), a technique that employs a training process inspired by contrastive learning. This novel approach enhances the structure of the (key, value) space, enabling an extension of the context length. Our method allows for fine-tuning pre-existing, large-scale models to lengthen their effective context. This is demonstrated by our fine-tuning of $3B$ and $7B$ OpenLLaMA checkpoints. The resulting models, which we name LongLLaMA, exhibit advancements in tasks requiring a long context. We further illustrate that our LongLLaMA models adeptly manage a $256 k$ context length for passkey retrieval.

{{</citation>}}


### (34/106) BLEURT Has Universal Translations: An Analysis of Automatic Metrics by Minimum Risk Training (Yiming Yan et al., 2023)

{{<citation>}}

Yiming Yan, Tao Wang, Chengqi Zhao, Shujian Huang, Jiajun Chen, Mingxuan Wang. (2023)  
**BLEURT Has Universal Translations: An Analysis of Automatic Metrics by Minimum Risk Training**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2307.03131v2)  

---


**ABSTRACT**  
Automatic metrics play a crucial role in machine translation. Despite the widespread use of n-gram-based metrics, there has been a recent surge in the development of pre-trained model-based metrics that focus on measuring sentence semantics. However, these neural metrics, while achieving higher correlations with human evaluations, are often considered to be black boxes with potential biases that are difficult to detect. In this study, we systematically analyze and compare various mainstream and cutting-edge automatic metrics from the perspective of their guidance for training machine translation systems. Through Minimum Risk Training (MRT), we find that certain metrics exhibit robustness defects, such as the presence of universal adversarial translations in BLEURT and BARTScore. In-depth analysis suggests two main causes of these robustness deficits: distribution biases in the training datasets, and the tendency of the metric paradigm. By incorporating token-level constraints, we enhance the robustness of evaluation metrics, which in turn leads to an improvement in the performance of machine translation systems. Codes are available at \url{https://github.com/powerpuffpomelo/fairseq_mrt}.

{{</citation>}}


### (35/106) VisKoP: Visual Knowledge oriented Programming for Interactive Knowledge Base Question Answering (Zijun Yao et al., 2023)

{{<citation>}}

Zijun Yao, Yuanyong Chen, Xin Lv, Shulin Cao, Amy Xin, Jifan Yu, Hailong Jin, Jianjun Xu, Peng Zhang, Lei Hou, Juanzi Li. (2023)  
**VisKoP: Visual Knowledge oriented Programming for Interactive Knowledge Base Question Answering**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-HC, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.03130v1)  

---


**ABSTRACT**  
We present Visual Knowledge oriented Programming platform (VisKoP), a knowledge base question answering (KBQA) system that integrates human into the loop to edit and debug the knowledge base (KB) queries. VisKoP not only provides a neural program induction module, which converts natural language questions into knowledge oriented program language (KoPL), but also maps KoPL programs into graphical elements. KoPL programs can be edited with simple graphical operators, such as dragging to add knowledge operators and slot filling to designate operator arguments. Moreover, VisKoP provides auto-completion for its knowledge base schema and users can easily debug the KoPL program by checking its intermediate results. To facilitate the practical KBQA on a million-entity-level KB, we design a highly efficient KoPL execution engine for the back-end. Experiment results show that VisKoP is highly efficient and user interaction can fix a large portion of wrong KoPL programs to acquire the correct answer. The VisKoP online demo https://demoviskop.xlore.cn (Stable release of this paper) and https://viskop.xlore.cn (Beta release with new features), highly efficient KoPL engine https://pypi.org/project/kopl-engine, and screencast video https://youtu.be/zAbJtxFPTXo are now publicly available.

{{</citation>}}


### (36/106) PREADD: Prefix-Adaptive Decoding for Controlled Text Generation (Jonathan Pei et al., 2023)

{{<citation>}}

Jonathan Pei, Kevin Yang, Dan Klein. (2023)  
**PREADD: Prefix-Adaptive Decoding for Controlled Text Generation**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2307.03214v1)  

---


**ABSTRACT**  
We propose Prefix-Adaptive Decoding (PREADD), a flexible method for controlled text generation. Unlike existing methods that use auxiliary expert models to control for attributes, PREADD does not require an external model, instead relying on linearly combining output logits from multiple prompts. Specifically, PREADD contrasts the output logits generated using a raw prompt against those generated using a prefix-prepended prompt, enabling both positive and negative control with respect to any attribute encapsulated by the prefix. We evaluate PREADD on three tasks -- toxic output mitigation, gender bias reduction, and sentiment control -- and find that PREADD outperforms not only prompting baselines, but also an auxiliary-expert control method, by 12% or more in relative gain on our main metrics for each task.

{{</citation>}}


### (37/106) Extracting Multi-valued Relations from Language Models (Sneha Singhania et al., 2023)

{{<citation>}}

Sneha Singhania, Simon Razniewski, Gerhard Weikum. (2023)  
**Extracting Multi-valued Relations from Language Models**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.03122v2)  

---


**ABSTRACT**  
The widespread usage of latent language representations via pre-trained language models (LMs) suggests that they are a promising source of structured knowledge. However, existing methods focus only on a single object per subject-relation pair, even though often multiple objects are correct. To overcome this limitation, we analyze these representations for their potential to yield materialized multi-object relational knowledge. We formulate the problem as a rank-then-select task. For ranking candidate objects, we evaluate existing prompting techniques and propose new ones incorporating domain knowledge. Among the selection methods, we find that choosing objects with a likelihood above a learned relation-specific threshold gives a 49.5% F1 score. Our results highlight the difficulty of employing LMs for the multi-valued slot-filling task and pave the way for further research on extracting relational knowledge from latent language representations.

{{</citation>}}


### (38/106) A Survey on Evaluation of Large Language Models (Yupeng Chang et al., 2023)

{{<citation>}}

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Kaijie Zhu, Hao Chen, Linyi Yang, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, Xing Xie. (2023)  
**A Survey on Evaluation of Large Language Models**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.03109v5)  

---


**ABSTRACT**  
Large language models (LLMs) are gaining increasing popularity in both academia and industry, owing to their unprecedented performance in various applications. As LLMs continue to play a vital role in both research and daily use, their evaluation becomes increasingly critical, not only at the task level, but also at the society level for better understanding of their potential risks. Over the past years, significant efforts have been made to examine LLMs from various perspectives. This paper presents a comprehensive review of these evaluation methods for LLMs, focusing on three key dimensions: what to evaluate, where to evaluate, and how to evaluate. Firstly, we provide an overview from the perspective of evaluation tasks, encompassing general natural language processing tasks, reasoning, medical usage, ethics, educations, natural and social sciences, agent applications, and other areas. Secondly, we answer the `where' and `how' questions by diving into the evaluation methods and benchmarks, which serve as crucial components in assessing performance of LLMs. Then, we summarize the success and failure cases of LLMs in different tasks. Finally, we shed light on several future challenges that lie ahead in LLMs evaluation. Our aim is to offer invaluable insights to researchers in the realm of LLMs evaluation, thereby aiding the development of more proficient LLMs. Our key point is that evaluation should be treated as an essential discipline to better assist the development of LLMs. We consistently maintain the related open-source materials at: https://github.com/MLGroupJLU/LLM-eval-survey.

{{</citation>}}


### (39/106) Efficient Domain Adaptation of Sentence Embeddings using Adapters (Tim Schopf et al., 2023)

{{<citation>}}

Tim Schopf, Dennis Schneider, Florian Matthes. (2023)  
**Efficient Domain Adaptation of Sentence Embeddings using Adapters**  

---
Primary Category: cs.CL
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Embedding, Sentence Embedding  
[Paper Link](http://arxiv.org/abs/2307.03104v1)  

---


**ABSTRACT**  
Sentence embeddings enable us to capture the semantic similarity of short texts. Most sentence embedding models are trained for general semantic textual similarity (STS) tasks. Therefore, to use sentence embeddings in a particular domain, the model must be adapted to it in order to achieve good results. Usually, this is done by fine-tuning the entire sentence embedding model for the domain of interest. While this approach yields state-of-the-art results, all of the model's weights are updated during fine-tuning, making this method resource-intensive. Therefore, instead of fine-tuning entire sentence embedding models for each target domain individually, we propose to train lightweight adapters. These domain-specific adapters do not require fine-tuning all underlying sentence embedding model parameters. Instead, we only train a small number of additional parameters while keeping the weights of the underlying sentence embedding model fixed. Training domain-specific adapters allows always using the same base model and only exchanging the domain-specific adapters to adapt sentence embeddings to a specific domain. We show that using adapters for parameter-efficient domain adaptation of sentence embeddings yields competitive performance within 1% of a domain-adapted, entirely fine-tuned sentence embedding model while only training approximately 3.6% of the parameters.

{{</citation>}}


### (40/106) Can ChatGPT's Responses Boost Traditional Natural Language Processing? (Mostafa M. Amin et al., 2023)

{{<citation>}}

Mostafa M. Amin, Erik Cambria, Björn W. Schuller. (2023)  
**Can ChatGPT's Responses Boost Traditional Natural Language Processing?**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.04648v1)  

---


**ABSTRACT**  
The employment of foundation models is steadily expanding, especially with the launch of ChatGPT and the release of other foundation models. These models have shown the potential of emerging capabilities to solve problems, without being particularly trained to solve. A previous work demonstrated these emerging capabilities in affective computing tasks; the performance quality was similar to traditional Natural Language Processing (NLP) techniques, but falling short of specialised trained models, like fine-tuning of the RoBERTa language model. In this work, we extend this by exploring if ChatGPT has novel knowledge that would enhance existing specialised models when they are fused together. We achieve this by investigating the utility of verbose responses from ChatGPT about solving a downstream task, in addition to studying the utility of fusing that with existing NLP methods. The study is conducted on three affective computing problems, namely sentiment analysis, suicide tendency detection, and big-five personality assessment. The results conclude that ChatGPT has indeed novel knowledge that can improve existing NLP techniques by way of fusion, be it early or late fusion.

{{</citation>}}


### (41/106) Parameter-Efficient Fine-Tuning of LLaMA for the Clinical Domain (Aryo Pradipta Gema et al., 2023)

{{<citation>}}

Aryo Pradipta Gema, Luke Daines, Pasquale Minervini, Beatrice Alex. (2023)  
**Parameter-Efficient Fine-Tuning of LLaMA for the Clinical Domain**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Clinical, LLaMA  
[Paper Link](http://arxiv.org/abs/2307.03042v2)  

---


**ABSTRACT**  
Adapting pretrained language models to novel domains, such as clinical applications, traditionally involves retraining their entire set of parameters. However, this approach is increasingly proven to be impractical owing to the substantial computational requirements associated with training such large language models. To address this issue, Parameter-Efficient Fine-Tuning (PEFT) techniques offer a viable solution by selectively fine-tuning a small subset of additional parameters, significantly reducing the computational requirements for domain adaptation. In this study, we propose Clinical LLaMA-LoRA, a PEFT adapter layer built upon the open-sourced LLaMA model. Clinical LLaMA-LoRA is trained using clinical notes obtained from the MIMIC-IV database, thereby creating a specialised adapter designed for the clinical domain. Additionally, we propose a two-step PEFT framework which fuses Clinical LLaMA-LoRA with Downstream LLaMA-LoRA, another PEFT adapter specialised for downstream tasks. We evaluate this framework on multiple clinical outcome prediction datasets, comparing it to clinically trained language models. Our proposed framework achieves a state-of-the-art AUROC score averaged across all clinical downstream tasks. We observe substantial improvements of 6-9% AUROC score in the large-scale multilabel classification tasks, such as diagnoses and procedures classification.

{{</citation>}}


### (42/106) Style Over Substance: Evaluation Biases for Large Language Models (Minghao Wu et al., 2023)

{{<citation>}}

Minghao Wu, Alham Fikri Aji. (2023)  
**Style Over Substance: Evaluation Biases for Large Language Models**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2307.03025v1)  

---


**ABSTRACT**  
As large language models (LLMs) continue to advance, accurately and comprehensively evaluating their performance becomes increasingly challenging. Conventionally, human evaluations are considered the gold standard in natural language generation. Recent advancements incorporate state-of-the-art LLMs as proxies for human judges in evaluation processes. Nonetheless, the extent to which humans and LLMs are capable evaluators remains uncertain. This study aims to investigate the behavior of both crowd-sourced human and LLM-based judges when comparing outputs from different models. To accomplish this, we curate a dataset comprising intentionally flawed machine-generated answers. Our findings indicate that despite the potentially greater danger posed by factual errors, answers with factual errors were still rated more favorably compared to answers that were too short or contained grammatical errors. This highlights a concerning bias in the evaluation process. To address this issue, we propose to independently evaluate machine-generated text across multiple dimensions, rather than merging all the evaluation aspects into a single score. We instantiate this idea with the Elo rating system, resulting in the Multi-Elo Rating System. Empirical results from our study reveal that this proposed approach significantly enhances the quality of LLM-based evaluations, particularly in terms of factual accuracy. However, notable improvement is not observed in crowd-sourced-based evaluations, suggesting the need for further investigation and refinement.

{{</citation>}}


### (43/106) CORE-GPT: Combining Open Access research and large language models for credible, trustworthy question answering (David Pride et al., 2023)

{{<citation>}}

David Pride, Matteo Cancellieri, Petr Knoth. (2023)  
**CORE-GPT: Combining Open Access research and large language models for credible, trustworthy question answering**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2307.04683v1)  

---


**ABSTRACT**  
In this paper, we present CORE-GPT, a novel question-answering platform that combines GPT-based language models and more than 32 million full-text open access scientific articles from CORE. We first demonstrate that GPT3.5 and GPT4 cannot be relied upon to provide references or citations for generated text. We then introduce CORE-GPT which delivers evidence-based answers to questions, along with citations and links to the cited papers, greatly increasing the trustworthiness of the answers and reducing the risk of hallucinations. CORE-GPT's performance was evaluated on a dataset of 100 questions covering the top 20 scientific domains in CORE, resulting in 100 answers and links to 500 relevant articles. The quality of the provided answers and and relevance of the links were assessed by two annotators. Our results demonstrate that CORE-GPT can produce comprehensive and trustworthy answers across the majority of scientific domains, complete with links to genuine, relevant scientific articles.

{{</citation>}}


### (44/106) Amplifying Limitations, Harms and Risks of Large Language Models (Michael O'Neill et al., 2023)

{{<citation>}}

Michael O'Neill, Mark Connor. (2023)  
**Amplifying Limitations, Harms and Risks of Large Language Models**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.04821v1)  

---


**ABSTRACT**  
We present this article as a small gesture in an attempt to counter what appears to be exponentially growing hype around Artificial Intelligence (AI) and its capabilities, and the distraction provided by the associated talk of science-fiction scenarios that might arise if AI should become sentient and super-intelligent. It may also help those outside of the field to become more informed about some of the limitations of AI technology. In the current context of popular discourse AI defaults to mean foundation and large language models (LLMs) such as those used to create ChatGPT. This in itself is a misrepresentation of the diversity, depth and volume of research, researchers, and technology that truly represents the field of AI. AI being a field of research that has existed in software artefacts since at least the 1950's. We set out to highlight a number of limitations of LLMs, and in so doing highlight that harms have already arisen and will continue to arise due to these limitations. Along the way we also highlight some of the associated risks for individuals and organisations in using this technology.

{{</citation>}}


### (45/106) LEA: Improving Sentence Similarity Robustness to Typos Using Lexical Attention Bias (Mario Almagro et al., 2023)

{{<citation>}}

Mario Almagro, Emilio Almazán, Diego Ortego, David Jiménez. (2023)  
**LEA: Improving Sentence Similarity Robustness to Typos Using Lexical Attention Bias**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, Bias, Sentence Similarity, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.02912v1)  

---


**ABSTRACT**  
Textual noise, such as typos or abbreviations, is a well-known issue that penalizes vanilla Transformers for most downstream tasks. We show that this is also the case for sentence similarity, a fundamental task in multiple domains, e.g. matching, retrieval or paraphrasing. Sentence similarity can be approached using cross-encoders, where the two sentences are concatenated in the input allowing the model to exploit the inter-relations between them. Previous works addressing the noise issue mainly rely on data augmentation strategies, showing improved robustness when dealing with corrupted samples that are similar to the ones used for training. However, all these methods still suffer from the token distribution shift induced by typos. In this work, we propose to tackle textual noise by equipping cross-encoders with a novel LExical-aware Attention module (LEA) that incorporates lexical similarities between words in both sentences. By using raw text similarities, our approach avoids the tokenization shift problem obtaining improved robustness. We demonstrate that the attention bias introduced by LEA helps cross-encoders to tackle complex scenarios with textual noise, specially in domains with short-text descriptions and limited context. Experiments using three popular Transformer encoders in five e-commerce datasets for product matching show that LEA consistently boosts performance under the presence of noise, while remaining competitive on the original (clean) splits. We also evaluate our approach in two datasets for textual entailment and paraphrasing showing that LEA is robust to typos in domains with longer sentences and more natural context. Additionally, we thoroughly analyze several design choices in our approach, providing insights about the impact of the decisions made and fostering future research in cross-encoders dealing with typos.

{{</citation>}}


### (46/106) Agentività e telicità in GilBERTo: implicazioni cognitive (Agnese Lombardi et al., 2023)

{{<citation>}}

Agnese Lombardi, Alessandro Lenci. (2023)  
**Agentività e telicità in GilBERTo: implicazioni cognitive**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2307.02910v1)  

---


**ABSTRACT**  
The goal of this study is to investigate whether a Transformer-based neural language model infers lexical semantics and use this information for the completion of morphosyntactic patterns. The semantic properties considered are telicity (also combined with definiteness) and agentivity. Both act at the interface between semantics and morphosyntax: they are semantically determined and syntactically encoded. The tasks were submitted to both the computational model and a group of Italian native speakers. The comparison between the two groups of data allows us to investigate to what extent neural language models capture significant aspects of human semantic competence.

{{</citation>}}


### (47/106) The Relationship Between Speech Features Changes When You Get Depressed: Feature Correlations for Improving Speed and Performance of Depression Detection (Fuxiang Tao et al., 2023)

{{<citation>}}

Fuxiang Tao, Wei Ma, Xuri Ge, Anna Esposito, Alessandro Vinciarelli. (2023)  
**The Relationship Between Speech Features Changes When You Get Depressed: Feature Correlations for Improving Speed and Performance of Depression Detection**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.02892v2)  

---


**ABSTRACT**  
This work shows that depression changes the correlation between features extracted from speech. Furthermore, it shows that using such an insight can improve the training speed and performance of depression detectors based on SVMs and LSTMs. The experiments were performed over the Androids Corpus, a publicly available dataset involving 112 speakers, including 58 people diagnosed with depression by professional psychiatrists. The results show that the models used in the experiments improve in terms of training speed and performance when fed with feature correlation matrices rather than with feature vectors. The relative reduction of the error rate ranges between 23.1% and 26.6% depending on the model. The probable explanation is that feature correlation matrices appear to be more variable in the case of depressed speakers. Correspondingly, such a phenomenon can be thought of as a depression marker.

{{</citation>}}


### (48/106) Contrast Is All You Need (Burak Kilic et al., 2023)

{{<citation>}}

Burak Kilic, Florix Bex, Albert Gatt. (2023)  
**Contrast Is All You Need**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.02882v1)  

---


**ABSTRACT**  
In this study, we analyze data-scarce classification scenarios, where available labeled legal data is small and imbalanced, potentially hurting the quality of the results. We focused on two finetuning objectives; SetFit (Sentence Transformer Finetuning), a contrastive learning setup, and a vanilla finetuning setup on a legal provision classification task. Additionally, we compare the features that are extracted with LIME (Local Interpretable Model-agnostic Explanations) to see which particular features contributed to the model's classification decisions. The results show that a contrastive setup with SetFit performed better than vanilla finetuning while using a fraction of the training samples. LIME results show that the contrastive learning approach helps boost both positive and negative features which are legally informative and contribute to the classification results. Thus a model finetuned with a contrastive objective seems to base its decisions more confidently on legally informative features.

{{</citation>}}


### (49/106) NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic (Zi'ou Zheng et al., 2023)

{{<citation>}}

Zi'ou Zheng, Xiaodan Zhu. (2023)  
**NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: NLI, Natural Language Inference, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.02849v1)  

---


**ABSTRACT**  
Reasoning has been a central topic in artificial intelligence from the beginning. The recent progress made on distributed representation and neural networks continues to improve the state-of-the-art performance of natural language inference. However, it remains an open question whether the models perform real reasoning to reach their conclusions or rely on spurious correlations. Adversarial attacks have proven to be an important tool to help evaluate the Achilles' heel of the victim models. In this study, we explore the fundamental problem of developing attack models based on logic formalism. We propose NatLogAttack to perform systematic attacks centring around natural logic, a classical logic formalism that is traceable back to Aristotle's syllogism and has been closely developed for natural language inference. The proposed framework renders both label-preserving and label-flipping attacks. We show that compared to the existing attack models, NatLogAttack generates better adversarial examples with fewer visits to the victim models. The victim models are found to be more vulnerable under the label-flipping setting. NatLogAttack provides a tool to probe the existing and future NLI models' capacity from a key viewpoint and we hope more logic-based attacks will be further explored for understanding the desired property of reasoning.

{{</citation>}}


### (50/106) Generative Zero-Shot Prompt Learning for Cross-Domain Slot Filling with Inverse Prompting (Xuefeng Li et al., 2023)

{{<citation>}}

Xuefeng Li, Liwen Wang, Guanting Dong, Keqing He, Jinzheng Zhao, Hao Lei, Jiachi Liu, Weiran Xu. (2023)  
**Generative Zero-Shot Prompt Learning for Cross-Domain Slot Filling with Inverse Prompting**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.02830v1)  

---


**ABSTRACT**  
Zero-shot cross-domain slot filling aims to transfer knowledge from the labeled source domain to the unlabeled target domain. Existing models either encode slot descriptions and examples or design handcrafted question templates using heuristic rules, suffering from poor generalization capability or robustness. In this paper, we propose a generative zero-shot prompt learning framework for cross-domain slot filling, both improving generalization and robustness than previous work. Besides, we introduce a novel inverse prompting strategy to distinguish different slot types to avoid the multiple prediction problem, and an efficient prompt-tuning strategy to boost higher performance by only training fewer prompt parameters. Experiments and analysis demonstrate the effectiveness of our proposed framework, especially huge improvements (+13.44% F1) on the unseen slots.

{{</citation>}}


### (51/106) Undecimated Wavelet Transform for Word Embedded Semantic Marginal Autoencoder in Security improvement and Denoising different Languages (Shreyanth S, 2023)

{{<citation>}}

Shreyanth S. (2023)  
**Undecimated Wavelet Transform for Word Embedded Semantic Marginal Autoencoder in Security improvement and Denoising different Languages**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-CR, cs-IR, cs-LG, cs.CL  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2307.03679v1)  

---


**ABSTRACT**  
By combining the undecimated wavelet transform within a Word Embedded Semantic Marginal Autoencoder (WESMA), this research study provides a novel strategy for improving security measures and denoising multiple languages. The incorporation of these strategies is intended to address the issues of robustness, privacy, and multilingualism in data processing applications. The undecimated wavelet transform is used as a feature extraction tool to identify prominent language patterns and structural qualities in the input data. The proposed system may successfully capture significant information while preserving the temporal and geographical links within the data by employing this transform. This improves security measures by increasing the system's ability to detect abnormalities, discover hidden patterns, and distinguish between legitimate content and dangerous threats. The Word Embedded Semantic Marginal Autoencoder also functions as an intelligent framework for dimensionality and noise reduction. The autoencoder effectively learns the underlying semantics of the data and reduces noise components by exploiting word embeddings and semantic context. As a result, data quality and accuracy are increased in following processing stages. The suggested methodology is tested using a diversified dataset that includes several languages and security scenarios. The experimental results show that the proposed approach is effective in attaining security enhancement and denoising capabilities across multiple languages. The system is strong in dealing with linguistic variances, producing consistent outcomes regardless of the language used. Furthermore, incorporating the undecimated wavelet transform considerably improves the system's ability to efficiently address complex security concerns

{{</citation>}}


### (52/106) PRD: Peer Rank and Discussion Improve Large Language Model based Evaluations (Ruosen Li et al., 2023)

{{<citation>}}

Ruosen Li, Teerth Patel, Xinya Du. (2023)  
**PRD: Peer Rank and Discussion Improve Large Language Model based Evaluations**  

---
Primary Category: cs.CL
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.02762v1)  

---


**ABSTRACT**  
Nowadays, the quality of responses generated by different modern large language models (LLMs) are hard to evaluate and compare automatically. Recent studies suggest and predominantly use LLMs as a reference-free metric for open-ended question answering. More specifically, they use the recognized "strongest" LLM as the evaluator, which conducts pairwise comparisons of candidate models' answers and provides a ranking score. However, this intuitive method has multiple problems, such as bringing in self-enhancement (favoring its own answers) and positional bias. We draw insights and lessons from the educational domain (Cho and MacArthur, 2011; Walsh, 2014) to improve LLM-based evaluations. Specifically, we propose the (1) peer rank (PR) algorithm that takes into account each peer LLM's pairwise preferences of all answer pairs, and outputs a final ranking of models; and (2) peer discussion (PD), where we prompt two LLMs to discuss and try to reach a mutual agreement on preferences of two answers. We conduct experiments on two benchmark datasets. We find that our approaches achieve higher accuracy and align better with human judgments, respectively. Interestingly, PR can induce a relatively accurate self-ranking of models under the anonymous setting, where each model's name is unrevealed. Our work provides space to explore evaluating models that are hard to compare for humans.

{{</citation>}}


### (53/106) Text Alignment Is An Efficient Unified Model for Massive NLP Tasks (Yuheng Zha et al., 2023)

{{<citation>}}

Yuheng Zha, Yichi Yang, Ruichen Li, Zhiting Hu. (2023)  
**Text Alignment Is An Efficient Unified Model for Massive NLP Tasks**  

---
Primary Category: cs.CL
Categories: cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT, GPT-3.5, GPT-4, NLP, T5  
[Paper Link](http://arxiv.org/abs/2307.02729v1)  

---


**ABSTRACT**  
Large language models (LLMs), typically designed as a function of next-word prediction, have excelled across extensive NLP tasks. Despite the generality, next-word prediction is often not an efficient formulation for many of the tasks, demanding an extreme scale of model parameters (10s or 100s of billions) and sometimes yielding suboptimal performance. In practice, it is often desirable to build more efficient models -- despite being less versatile, they still apply to a substantial subset of problems, delivering on par or even superior performance with much smaller model sizes. In this paper, we propose text alignment as an efficient unified model for a wide range of crucial tasks involving text entailment, similarity, question answering (and answerability), factual consistency, and so forth. Given a pair of texts, the model measures the degree of alignment between their information. We instantiate an alignment model (Align) through lightweight finetuning of RoBERTa (355M parameters) using 5.9M examples from 28 datasets. Despite its compact size, extensive experiments show the model's efficiency and strong performance: (1) On over 20 datasets of aforementioned diverse tasks, the model matches or surpasses FLAN-T5 models that have around 2x or 10x more parameters; the single unified model also outperforms task-specific models finetuned on individual datasets; (2) When applied to evaluate factual consistency of language generation on 23 datasets, our model improves over various baselines, including the much larger GPT-3.5 (ChatGPT) and sometimes even GPT-4; (3) The lightweight model can also serve as an add-on component for LLMs such as GPT-3.5 in question answering tasks, improving the average exact match (EM) score by 17.94 and F1 score by 15.05 through identifying unanswerable questions.

{{</citation>}}


### (54/106) On-Device Constrained Self-Supervised Speech Representation Learning for Keyword Spotting via Knowledge Distillation (Gene-Ping Yang et al., 2023)

{{<citation>}}

Gene-Ping Yang, Yue Gu, Qingming Tang, Dongsu Du, Yuzong Liu. (2023)  
**On-Device Constrained Self-Supervised Speech Representation Learning for Keyword Spotting via Knowledge Distillation**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Knowledge Distillation, Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.02720v1)  

---


**ABSTRACT**  
Large self-supervised models are effective feature extractors, but their application is challenging under on-device budget constraints and biased dataset collection, especially in keyword spotting. To address this, we proposed a knowledge distillation-based self-supervised speech representation learning (S3RL) architecture for on-device keyword spotting. Our approach used a teacher-student framework to transfer knowledge from a larger, more complex model to a smaller, light-weight model using dual-view cross-correlation distillation and the teacher's codebook as learning objectives. We evaluated our model's performance on an Alexa keyword spotting detection task using a 16.6k-hour in-house dataset. Our technique showed exceptional performance in normal and noisy conditions, demonstrating the efficacy of knowledge distillation methods in constructing self-supervised models for keyword spotting tasks while working within on-device resource constraints.

{{</citation>}}


### (55/106) CFSum: A Coarse-to-Fine Contribution Network for Multimodal Summarization (Min Xiao et al., 2023)

{{<citation>}}

Min Xiao, Junnan Zhu, Haitao Lin, Yu Zhou, Chengqing Zong. (2023)  
**CFSum: A Coarse-to-Fine Contribution Network for Multimodal Summarization**  

---
Primary Category: cs.CL
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2307.02716v1)  

---


**ABSTRACT**  
Multimodal summarization usually suffers from the problem that the contribution of the visual modality is unclear. Existing multimodal summarization approaches focus on designing the fusion methods of different modalities, while ignoring the adaptive conditions under which visual modalities are useful. Therefore, we propose a novel Coarse-to-Fine contribution network for multimodal Summarization (CFSum) to consider different contributions of images for summarization. First, to eliminate the interference of useless images, we propose a pre-filter module to abandon useless images. Second, to make accurate use of useful images, we propose two levels of visual complement modules, word level and phrase level. Specifically, image contributions are calculated and are adopted to guide the attention of both textual and visual modalities. Experimental results have shown that CFSum significantly outperforms multiple strong baselines on the standard benchmark. Furthermore, the analysis verifies that useful images can even help generate non-visual words which are implicitly represented in the image.

{{</citation>}}


## eess.AS (1)



### (56/106) Gammatonegram Representation for End-to-End Dysarthric Speech Processing Tasks: Speech Recognition, Speaker Identification, and Intelligibility Assessment (Aref Farhadipour et al., 2023)

{{<citation>}}

Aref Farhadipour, Hadi Veisi. (2023)  
**Gammatonegram Representation for End-to-End Dysarthric Speech Processing Tasks: Speech Recognition, Speaker Identification, and Intelligibility Assessment**  

---
Primary Category: eess.AS
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.03296v1)  

---


**ABSTRACT**  
Dysarthria is a disability that causes a disturbance in the human speech system and reduces the quality and intelligibility of a person's speech. Because of this effect, the normal speech processing systems can not work properly on impaired speech. This disability is usually associated with physical disabilities. Therefore, designing a system that can perform some tasks by receiving voice commands in the smart home can be a significant achievement. In this work, we introduce gammatonegram as an effective method to represent audio files with discriminative details, which is used as input for the convolutional neural network. On the other word, we convert each speech file into an image and propose image recognition system to classify speech in different scenarios. Proposed CNN is based on the transfer learning method on the pre-trained Alexnet. In this research, the efficiency of the proposed system for speech recognition, speaker identification, and intelligibility assessment is evaluated. According to the results on the UA dataset, the proposed speech recognition system achieved 91.29% accuracy in speaker-dependent mode, the speaker identification system acquired 87.74% accuracy in text-dependent mode, and the intelligibility assessment system achieved 96.47% accuracy in two-class mode. Finally, we propose a multi-network speech recognition system that works fully automatically. This system is located in a cascade arrangement with the two-class intelligibility assessment system, and the output of this system activates each one of the speech recognition networks. This architecture achieves an accuracy of 92.3% WRR. The source code of this paper is available.

{{</citation>}}


## cs.DC (1)



### (57/106) ChainScience 2023, Conference Proceedings (Hans Walter Behrens et al., 2023)

{{<citation>}}

Hans Walter Behrens, Nicolò Vallarano, Claudio J. Tessone. (2023)  
**ChainScience 2023, Conference Proceedings**  

---
Primary Category: cs.DC
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.03277v2)  

---


**ABSTRACT**  
The proceedings of ChainScience 2023 epitomize the integration of various scientific disciplines with the dynamic world of blockchain and AI. This collection, encapsulating both full and short papers, as well as posters, delves into areas such as cryptoeconomics, machine learning, and analysis of blockchain networks, under the guiding principle of this year's conference: 'Applying insights and methods from the physical, mathematical, behavioral, and computational sciences to blockchain and the future of AI'. This diverse compilation offers a critical examination of theoretical constructs and innovative practical applications to empower the future of the blockchain industry.

{{</citation>}}


## cs.CV (23)



### (58/106) ADASSM: Adversarial Data Augmentation in Statistical Shape Models From Images (Mokshagna Sai Teja Karanam et al., 2023)

{{<citation>}}

Mokshagna Sai Teja Karanam, Tushar Kataria, Shireen Elhabian. (2023)  
**ADASSM: Adversarial Data Augmentation in Statistical Shape Models From Images**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.03273v2)  

---


**ABSTRACT**  
Statistical shape models (SSM) have been well-established as an excellent tool for identifying variations in the morphology of anatomy across the underlying population. Shape models use consistent shape representation across all the samples in a given cohort, which helps to compare shapes and identify the variations that can detect pathologies and help in formulating treatment plans. In medical imaging, computing these shape representations from CT/MRI scans requires time-intensive preprocessing operations, including but not limited to anatomy segmentation annotations, registration, and texture denoising. Deep learning models have demonstrated exceptional capabilities in learning shape representations directly from volumetric images, giving rise to highly effective and efficient Image-to-SSM. Nevertheless, these models are data-hungry and due to the limited availability of medical data, deep learning models tend to overfit. Offline data augmentation techniques, that use kernel density estimation based (KDE) methods for generating shape-augmented samples, have successfully aided Image-to-SSM networks in achieving comparable accuracy to traditional SSM methods. However, these augmentation methods focus on shape augmentation, whereas deep learning models exhibit image-based texture bias results in sub-optimal models. This paper introduces a novel strategy for on-the-fly data augmentation for the Image-to-SSM framework by leveraging data-dependent noise generation or texture augmentation. The proposed framework is trained as an adversary to the Image-to-SSM network, augmenting diverse and challenging noisy samples. Our approach achieves improved accuracy by encouraging the model to focus on the underlying geometry rather than relying solely on pixel values.

{{</citation>}}


### (59/106) Vision Language Transformers: A Survey (Clayton Fields et al., 2023)

{{<citation>}}

Clayton Fields, Casey Kennington. (2023)  
**Vision Language Transformers: A Survey**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.03254v1)  

---


**ABSTRACT**  
Vision language tasks, such as answering questions about or generating captions that describe an image, are difficult tasks for computers to perform. A relatively recent body of research has adapted the pretrained transformer architecture introduced in \citet{vaswani2017attention} to vision language modeling. Transformer models have greatly improved performance and versatility over previous vision language models. They do so by pretraining models on a large generic datasets and transferring their learning to new tasks with minor changes in architecture and parameter values. This type of transfer learning has become the standard modeling practice in both natural language processing and computer vision. Vision language transformers offer the promise of producing similar advancements in tasks which require both vision and language. In this paper, we provide a broad synthesis of the currently available research on vision language transformer models and offer some analysis of their strengths, limitations and some open questions that remain.

{{</citation>}}


### (60/106) That's BAD: Blind Anomaly Detection by Implicit Local Feature Clustering (Jie Zhang et al., 2023)

{{<citation>}}

Jie Zhang, Masanori Suganuma, Takayuki Okatani. (2023)  
**That's BAD: Blind Anomaly Detection by Implicit Local Feature Clustering**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.03243v1)  

---


**ABSTRACT**  
Recent studies on visual anomaly detection (AD) of industrial objects/textures have achieved quite good performance. They consider an unsupervised setting, specifically the one-class setting, in which we assume the availability of a set of normal (\textit{i.e.}, anomaly-free) images for training. In this paper, we consider a more challenging scenario of unsupervised AD, in which we detect anomalies in a given set of images that might contain both normal and anomalous samples. The setting does not assume the availability of known normal data and thus is completely free from human annotation, which differs from the standard AD considered in recent studies. For clarity, we call the setting blind anomaly detection (BAD). We show that BAD can be converted into a local outlier detection problem and propose a novel method named PatchCluster that can accurately detect image- and pixel-level anomalies. Experimental results show that PatchCluster shows a promising performance without the knowledge of normal data, even comparable to the SOTA methods applied in the one-class setting needing it.

{{</citation>}}


### (61/106) VideoGLUE: Video General Understanding Evaluation of Foundation Models (Liangzhe Yuan et al., 2023)

{{<citation>}}

Liangzhe Yuan, Nitesh Bharadwaj Gundavarapu, Long Zhao, Hao Zhou, Yin Cui, Lu Jiang, Xuan Yang, Menglin Jia, Tobias Weyand, Luke Friedman, Mikhail Sirotenko, Huisheng Wang, Florian Schroff, Hartwig Adam, Ming-Hsuan Yang, Ting Liu, Boqing Gong. (2023)  
**VideoGLUE: Video General Understanding Evaluation of Foundation Models**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: GLUE  
[Paper Link](http://arxiv.org/abs/2307.03166v1)  

---


**ABSTRACT**  
We evaluate existing foundation models video understanding capabilities using a carefully designed experiment protocol consisting of three hallmark tasks (action recognition, temporal localization, and spatiotemporal localization), eight datasets well received by the community, and four adaptation methods tailoring a foundation model (FM) for a downstream task. Moreover, we propose a scalar VideoGLUE score (VGS) to measure an FMs efficacy and efficiency when adapting to general video understanding tasks. Our main findings are as follows. First, task-specialized models significantly outperform the six FMs studied in this work, in sharp contrast to what FMs have achieved in natural language and image understanding. Second,video-native FMs, whose pretraining data contains the video modality, are generally better than image-native FMs in classifying motion-rich videos, localizing actions in time, and understanding a video of more than one action. Third, the video-native FMs can perform well on video tasks under light adaptations to downstream tasks(e.g., freezing the FM backbones), while image-native FMs win in full end-to-end finetuning. The first two observations reveal the need and tremendous opportunities to conduct research on video-focused FMs, and the last confirms that both tasks and adaptation methods matter when it comes to the evaluation of FMs.

{{</citation>}}


### (62/106) Distilling Large Vision-Language Model with Out-of-Distribution Generalizability (Xuanlin Li et al., 2023)

{{<citation>}}

Xuanlin Li, Yunhao Fang, Minghua Liu, Zhan Ling, Zhuowen Tu, Hao Su. (2023)  
**Distilling Large Vision-Language Model with Out-of-Distribution Generalizability**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.03135v2)  

---


**ABSTRACT**  
Large vision-language models have achieved outstanding performance, but their size and computational requirements make their deployment on resource-constrained devices and time-sensitive tasks impractical. Model distillation, the process of creating smaller, faster models that maintain the performance of larger models, is a promising direction towards the solution. This paper investigates the distillation of visual representations in large teacher vision-language models into lightweight student models using a small- or mid-scale dataset. Notably, this study focuses on open-vocabulary out-of-distribution (OOD) generalization, a challenging problem that has been overlooked in previous model distillation literature. We propose two principles from vision and language modality perspectives to enhance student's OOD generalization: (1) by better imitating teacher's visual representation space, and carefully promoting better coherence in vision-language alignment with the teacher; (2) by enriching the teacher's language representations with informative and finegrained semantic attributes to effectively distinguish between different labels. We propose several metrics and conduct extensive experiments to investigate their techniques. The results demonstrate significant improvements in zero-shot and few-shot student performance on open-vocabulary out-of-distribution classification, highlighting the effectiveness of our proposed approaches. Code released at https://github.com/xuanlinli17/large_vlm_distillation_ood

{{</citation>}}


### (63/106) T-MARS: Improving Visual Representations by Circumventing Text Feature Learning (Pratyush Maini et al., 2023)

{{<citation>}}

Pratyush Maini, Sachin Goyal, Zachary C. Lipton, J. Zico Kolter, Aditi Raghunathan. (2023)  
**T-MARS: Improving Visual Representations by Circumventing Text Feature Learning**  

---
Primary Category: cs.CV
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2307.03132v1)  

---


**ABSTRACT**  
Large web-sourced multimodal datasets have powered a slew of new methods for learning general-purpose visual representations, advancing the state of the art in computer vision and revolutionizing zero- and few-shot recognition. One crucial decision facing practitioners is how, if at all, to curate these ever-larger datasets. For example, the creators of the LAION-5B dataset chose to retain only image-caption pairs whose CLIP similarity score exceeded a designated threshold. In this paper, we propose a new state-of-the-art data filtering approach motivated by our observation that nearly 40% of LAION's images contain text that overlaps significantly with the caption. Intuitively, such data could be wasteful as it incentivizes models to perform optical character recognition rather than learning visual features. However, naively removing all such data could also be wasteful, as it throws away images that contain visual features (in addition to overlapping text). Our simple and scalable approach, T-MARS (Text Masking and Re-Scoring), filters out only those pairs where the text dominates the remaining visual features -- by first masking out the text and then filtering out those with a low CLIP similarity score of the masked image. Experimentally, T-MARS outperforms the top-ranked method on the "medium scale" of DataComp (a data filtering benchmark) by a margin of 6.5% on ImageNet and 4.7% on VTAB. Additionally, our systematic evaluation on various data pool sizes from 2M to 64M shows that the accuracy gains enjoyed by T-MARS linearly increase as data and compute are scaled exponentially. Code is available at https://github.com/locuslab/T-MARS.

{{</citation>}}


### (64/106) Region-Wise Attentive Multi-View Representation Learning for Urban Region Embeddings (Weiliang Chan et al., 2023)

{{<citation>}}

Weiliang Chan, Qianqian Ren. (2023)  
**Region-Wise Attentive Multi-View Representation Learning for Urban Region Embeddings**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-CY, cs-LG, cs.CV  
Keywords: Embedding, Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.03212v1)  

---


**ABSTRACT**  
Urban region embedding is an important and yet highly challenging issue due to the complexity and constantly changing nature of urban data. To address the challenges, we propose a Region-Wise Multi-View Representation Learning (ROMER) to capture multi-view dependencies and learn expressive representations of urban regions without the constraints of rigid neighbourhood region conditions. Our model focus on learn urban region representation from multi-source urban data. First, we capture the multi-view correlations from mobility flow patterns, POI semantics and check-in dynamics. Then, we adopt global graph attention networks to learn similarity of any two vertices in graphs. To comprehensively consider and share features of multiple views, a two-stage fusion module is further proposed to learn weights with external attention to fuse multi-view embeddings. Extensive experiments for two downstream tasks on real-world datasets demonstrate that our model outperforms state-of-the-art methods by up to 17\% improvement.

{{</citation>}}


### (65/106) LISSNAS: Locality-based Iterative Search Space Shrinkage for Neural Architecture Search (Bhavna Gopal et al., 2023)

{{<citation>}}

Bhavna Gopal, Arjun Sridhar, Tunhou Zhang, Yiran Chen. (2023)  
**LISSNAS: Locality-based Iterative Search Space Shrinkage for Neural Architecture Search**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.03110v1)  

---


**ABSTRACT**  
Search spaces hallmark the advancement of Neural Architecture Search (NAS). Large and complex search spaces with versatile building operators and structures provide more opportunities to brew promising architectures, yet pose severe challenges on efficient exploration and exploitation. Subsequently, several search space shrinkage methods optimize by selecting a single sub-region that contains some well-performing networks. Small performance and efficiency gains are observed with these methods but such techniques leave room for significantly improved search performance and are ineffective at retaining architectural diversity. We propose LISSNAS, an automated algorithm that shrinks a large space into a diverse, small search space with SOTA search performance. Our approach leverages locality, the relationship between structural and performance similarity, to efficiently extract many pockets of well-performing networks. We showcase our method on an array of search spaces spanning various sizes and datasets. We accentuate the effectiveness of our shrunk spaces when used in one-shot search by achieving the best Top-1 accuracy in two different search spaces. Our method achieves a SOTA Top-1 accuracy of 77.6\% in ImageNet under mobile constraints, best-in-class Kendal-Tau, architectural diversity, and search space size.

{{</citation>}}


### (66/106) Contextual Affinity Distillation for Image Anomaly Detection (Jie Zhang et al., 2023)

{{<citation>}}

Jie Zhang, Masanori Suganuma, Takayuki Okatani. (2023)  
**Contextual Affinity Distillation for Image Anomaly Detection**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.03101v1)  

---


**ABSTRACT**  
Previous works on unsupervised industrial anomaly detection mainly focus on local structural anomalies such as cracks and color contamination. While achieving significantly high detection performance on this kind of anomaly, they are faced with logical anomalies that violate the long-range dependencies such as a normal object placed in the wrong position. In this paper, based on previous knowledge distillation works, we propose to use two students (local and global) to better mimic the teacher's behavior. The local student, which is used in previous studies mainly focuses on structural anomaly detection while the global student pays attention to logical anomalies. To further encourage the global student's learning to capture long-range dependencies, we design the global context condensing block (GCCB) and propose a contextual affinity loss for the student training and anomaly scoring. Experimental results show the proposed method doesn't need cumbersome training techniques and achieves a new state-of-the-art performance on the MVTec LOCO AD dataset.

{{</citation>}}


### (67/106) Proto-CLIP: Vision-Language Prototypical Network for Few-Shot Learning (Jishnu Jaykumar P et al., 2023)

{{<citation>}}

Jishnu Jaykumar P, Kamalesh Palanisamy, Yu-Wei Chao, Xinya Du, Yu Xiang. (2023)  
**Proto-CLIP: Vision-Language Prototypical Network for Few-Shot Learning**  

---
Primary Category: cs.CV
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.03073v2)  

---


**ABSTRACT**  
We propose a novel framework for few-shot learning by leveraging large-scale vision-language models such as CLIP. Motivated by the unimodal prototypical networks for few-shot learning, we introduce PROTO-CLIP that utilizes image prototypes and text prototypes for few-shot learning. Specifically, PROTO-CLIP adapts the image encoder and text encoder in CLIP in a joint fashion using few-shot examples. The two encoders are used to compute prototypes of image classes for classification. During adaptation, we propose aligning the image and text prototypes of corresponding classes. Such a proposed alignment is beneficial for few-shot classification due to the contributions from both types of prototypes. We demonstrate the effectiveness of our method by conducting experiments on benchmark datasets for few-shot learning as well as in the real world for robot perception.

{{</citation>}}


### (68/106) Art Authentication with Vision Transformers (Ludovica Schaerf et al., 2023)

{{<citation>}}

Ludovica Schaerf, Carina Popovici, Eric Postma. (2023)  
**Art Authentication with Vision Transformers**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.03039v2)  

---


**ABSTRACT**  
In recent years, Transformers, initially developed for language, have been successfully applied to visual tasks. Vision Transformers have been shown to push the state-of-the-art in a wide range of tasks, including image classification, object detection, and semantic segmentation. While ample research has shown promising results in art attribution and art authentication tasks using Convolutional Neural Networks, this paper examines if the superiority of Vision Transformers extends to art authentication, improving, thus, the reliability of computer-based authentication of artworks. Using a carefully compiled dataset of authentic paintings by Vincent van Gogh and two contrast datasets, we compare the art authentication performances of Swin Transformers with those of EfficientNet. Using a standard contrast set containing imitations and proxies (works by painters with styles closely related to van Gogh), we find that EfficientNet achieves the best performance overall. With a contrast set that only consists of imitations, we find the Swin Transformer to be superior to EfficientNet by achieving an authentication accuracy of over 85%. These results lead us to conclude that Vision Transformers represent a strong and promising contender in art authentication, particularly in enhancing the computer-based ability to detect artistic imitations.

{{</citation>}}


### (69/106) Cross-Spatial Pixel Integration and Cross-Stage Feature Fusion Based Transformer Network for Remote Sensing Image Super-Resolution (Yuting Lu et al., 2023)

{{<citation>}}

Yuting Lu, Lingtong Min, Binglu Wang, Le Zheng, Xiaoxu Wang, Yongqiang Zhao, Teng Long. (2023)  
**Cross-Spatial Pixel Integration and Cross-Stage Feature Fusion Based Transformer Network for Remote Sensing Image Super-Resolution**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.02974v1)  

---


**ABSTRACT**  
Remote sensing image super-resolution (RSISR) plays a vital role in enhancing spatial detials and improving the quality of satellite imagery. Recently, Transformer-based models have shown competitive performance in RSISR. To mitigate the quadratic computational complexity resulting from global self-attention, various methods constrain attention to a local window, enhancing its efficiency. Consequently, the receptive fields in a single attention layer are inadequate, leading to insufficient context modeling. Furthermore, while most transform-based approaches reuse shallow features through skip connections, relying solely on these connections treats shallow and deep features equally, impeding the model's ability to characterize them. To address these issues, we propose a novel transformer architecture called Cross-Spatial Pixel Integration and Cross-Stage Feature Fusion Based Transformer Network (SPIFFNet) for RSISR. Our proposed model effectively enhances global cognition and understanding of the entire image, facilitating efficient integration of features cross-stages. The model incorporates cross-spatial pixel integration attention (CSPIA) to introduce contextual information into a local window, while cross-stage feature fusion attention (CSFFA) adaptively fuses features from the previous stage to improve feature expression in line with the requirements of the current stage. We conducted comprehensive experiments on multiple benchmark datasets, demonstrating the superior performance of our proposed SPIFFNet in terms of both quantitative metrics and visual quality when compared to state-of-the-art methods.

{{</citation>}}


### (70/106) Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications (Peter Tu et al., 2023)

{{<citation>}}

Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Dylan Campbell, Jaskirat Singh, Tianyu Wang. (2023)  
**Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02881v2)  

---


**ABSTRACT**  
This paper begins with a description of methods for estimating probability density functions for images that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space - not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, although images may lie on such lower-dimensional manifolds, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. In pursuing this goal, we consider generative models that are popular in AI and computer vision community. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: it should be possible to sample from this distribution according to the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute the probability of the sample, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show that such probabilistic descriptions can be used to construct defences against adversarial attacks. In addition to describing the manifold in terms of density, we also consider how semantic interpretations can be used to describe points on the manifold. To this end, we consider an emergent language framework which makes use of variational encoders to produce a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described in terms of evolving semantic descriptions.

{{</citation>}}


### (71/106) Revisiting Computer-Aided Tuberculosis Diagnosis (Yun Liu et al., 2023)

{{<citation>}}

Yun Liu, Yu-Huan Wu, Shi-Chen Zhang, Li Liu, Min Wu, Ming-Ming Cheng. (2023)  
**Revisiting Computer-Aided Tuberculosis Diagnosis**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.02848v1)  

---


**ABSTRACT**  
Tuberculosis (TB) is a major global health threat, causing millions of deaths annually. Although early diagnosis and treatment can greatly improve the chances of survival, it remains a major challenge, especially in developing countries. Recently, computer-aided tuberculosis diagnosis (CTD) using deep learning has shown promise, but progress is hindered by limited training data. To address this, we establish a large-scale dataset, namely the Tuberculosis X-ray (TBX11K) dataset, which contains 11,200 chest X-ray (CXR) images with corresponding bounding box annotations for TB areas. This dataset enables the training of sophisticated detectors for high-quality CTD. Furthermore, we propose a strong baseline, SymFormer, for simultaneous CXR image classification and TB infection area detection. SymFormer incorporates Symmetric Search Attention (SymAttention) to tackle the bilateral symmetry property of CXR images for learning discriminative features. Since CXR images may not strictly adhere to the bilateral symmetry property, we also propose Symmetric Positional Encoding (SPE) to facilitate SymAttention through feature recalibration. To promote future research on CTD, we build a benchmark by introducing evaluation metrics, evaluating baseline models reformed from existing detectors, and running an online challenge. Experiments show that SymFormer achieves state-of-the-art performance on the TBX11K dataset. The data, code, and models will be released.

{{</citation>}}


### (72/106) Noise-to-Norm Reconstruction for Industrial Anomaly Detection and Localization (Shiqi Deng et al., 2023)

{{<citation>}}

Shiqi Deng, Zhiyu Sun, Ruiyan Zhuang, Jun Gong. (2023)  
**Noise-to-Norm Reconstruction for Industrial Anomaly Detection and Localization**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.02836v1)  

---


**ABSTRACT**  
Anomaly detection has a wide range of applications and is especially important in industrial quality inspection. Currently, many top-performing anomaly-detection models rely on feature-embedding methods. However, these methods do not perform well on datasets with large variations in object locations. Reconstruction-based methods use reconstruction errors to detect anomalies without considering positional differences between samples. In this study, a reconstruction-based method using the noise-to-norm paradigm is proposed, which avoids the invariant reconstruction of anomalous regions. Our reconstruction network is based on M-net and incorporates multiscale fusion and residual attention modules to enable end-to-end anomaly detection and localization. Experiments demonstrate that the method is effective in reconstructing anomalous regions into normal patterns and achieving accurate anomaly detection and localization. On the MPDD and VisA datasets, our proposed method achieved more competitive results than the latest methods, and it set a new state-of-the-art standard on the MPDD dataset.

{{</citation>}}


### (73/106) Read, Look or Listen? What's Needed for Solving a Multimodal Dataset (Netta Madvil et al., 2023)

{{<citation>}}

Netta Madvil, Yonatan Bitton, Roy Schwartz. (2023)  
**Read, Look or Listen? What's Needed for Solving a Multimodal Dataset**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CL, cs-CV, cs.CV, eess-AS  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2307.04532v1)  

---


**ABSTRACT**  
The prevalence of large-scale multimodal datasets presents unique challenges in assessing dataset quality. We propose a two-step method to analyze multimodal datasets, which leverages a small seed of human annotation to map each multimodal instance to the modalities required to process it. Our method sheds light on the importance of different modalities in datasets, as well as the relationship between them. We apply our approach to TVQA, a video question-answering dataset, and discover that most questions can be answered using a single modality, without a substantial bias towards any specific modality. Moreover, we find that more than 70% of the questions are solvable using several different single-modality strategies, e.g., by either looking at the video or listening to the audio, highlighting the limited integration of multiple modalities in TVQA. We leverage our annotation and analyze the MERLOT Reserve, finding that it struggles with image-based questions compared to text and audio, but also with auditory speaker identification. Based on our observations, we introduce a new test set that necessitates multiple modalities, observing a dramatic drop in model performance. Our methodology provides valuable insights into multimodal datasets and highlights the need for the development of more robust models.

{{</citation>}}


### (74/106) Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks (Xu Han et al., 2023)

{{<citation>}}

Xu Han, Anmin Liu, Chenxuan Yao, Yanbo Fan, Kun He. (2023)  
**Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**  

---
Primary Category: cs.CV
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Attack, ImageNet  
[Paper Link](http://arxiv.org/abs/2307.02828v1)  

---


**ABSTRACT**  
Deep neural networks are known to be vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to the benign input. After achieving nearly 100% attack success rates in white-box setting, more focus is shifted to black-box attacks, of which the transferability of adversarial examples has gained significant attention. In either case, the common gradient-based methods generally use the sign function to generate perturbations on the gradient update, that offers a roughly correct direction and has gained great success. But little work pays attention to its possible limitation. In this work, we observe that the deviation between the original gradient and the generated noise may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability. To this end, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM). Specifically, we use data rescaling to substitute the sign function without extra computational cost. We further propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method could be used in any gradient-based attacks and is extensible to be integrated with various input transformation or ensemble methods to further improve the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our method could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.

{{</citation>}}


### (75/106) Semi-supervised Domain Adaptive Medical Image Segmentation through Consistency Regularized Disentangled Contrastive Learning (Hritam Basak et al., 2023)

{{<citation>}}

Hritam Basak, Zhaozheng Yin. (2023)  
**Semi-supervised Domain Adaptive Medical Image Segmentation through Consistency Regularized Disentangled Contrastive Learning**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2307.02798v1)  

---


**ABSTRACT**  
Although unsupervised domain adaptation (UDA) is a promising direction to alleviate domain shift, they fall short of their supervised counterparts. In this work, we investigate relatively less explored semi-supervised domain adaptation (SSDA) for medical image segmentation, where access to a few labeled target samples can improve the adaptation performance substantially. Specifically, we propose a two-stage training process. First, an encoder is pre-trained in a self-learning paradigm using a novel domain-content disentangled contrastive learning (CL) along with a pixel-level feature consistency constraint. The proposed CL enforces the encoder to learn discriminative content-specific but domain-invariant semantics on a global scale from the source and target images, whereas consistency regularization enforces the mining of local pixel-level information by maintaining spatial sensitivity. This pre-trained encoder, along with a decoder, is further fine-tuned for the downstream task, (i.e. pixel-level segmentation) using a semi-supervised setting. Furthermore, we experimentally validate that our proposed method can easily be extended for UDA settings, adding to the superiority of the proposed strategy. Upon evaluation on two domain adaptive image segmentation tasks, our proposed method outperforms the SoTA methods, both in SSDA and UDA settings. Code is available at https://github.com/hritam-98/GFDA-disentangled

{{</citation>}}


### (76/106) The Role of Subgroup Separability in Group-Fair Medical Image Classification (Charles Jones et al., 2023)

{{<citation>}}

Charles Jones, Mélanie Roschewitz, Ben Glocker. (2023)  
**The Role of Subgroup Separability in Group-Fair Medical Image Classification**  

---
Primary Category: cs.CV
Categories: cs-AI, cs-CV, cs-CY, cs-LG, cs.CV  
Keywords: AI, Image Classification  
[Paper Link](http://arxiv.org/abs/2307.02791v1)  

---


**ABSTRACT**  
We investigate performance disparities in deep classifiers. We find that the ability of classifiers to separate individuals into subgroups varies substantially across medical imaging modalities and protected characteristics; crucially, we show that this property is predictive of algorithmic bias. Through theoretical analysis and extensive empirical evaluation, we find a relationship between subgroup separability, subgroup disparities, and performance degradation when models are trained on data with systematic bias such as underdiagnosis. Our findings shed new light on the question of how models become biased, providing important insights for the development of fair medical imaging AI.

{{</citation>}}


### (77/106) UIT-Saviors at MEDVQA-GI 2023: Improving Multimodal Learning with Image Enhancement for Gastrointestinal Visual Question Answering (Triet M. Thai et al., 2023)

{{<citation>}}

Triet M. Thai, Anh T. Vo, Hao K. Tieu, Linh N. P. Bui, Thien T. B. Nguyen. (2023)  
**UIT-Saviors at MEDVQA-GI 2023: Improving Multimodal Learning with Image Enhancement for Gastrointestinal Visual Question Answering**  

---
Primary Category: cs.CV
Categories: cs-CV, cs-HC, cs.CV  
Keywords: BERT, QA, Question Answering, Transformer  
[Paper Link](http://arxiv.org/abs/2307.02783v1)  

---


**ABSTRACT**  
In recent years, artificial intelligence has played an important role in medicine and disease diagnosis, with many applications to be mentioned, one of which is Medical Visual Question Answering (MedVQA). By combining computer vision and natural language processing, MedVQA systems can assist experts in extracting relevant information from medical image based on a given question and providing precise diagnostic answers. The ImageCLEFmed-MEDVQA-GI-2023 challenge carried out visual question answering task in the gastrointestinal domain, which includes gastroscopy and colonoscopy images. Our team approached Task 1 of the challenge by proposing a multimodal learning method with image enhancement to improve the VQA performance on gastrointestinal images. The multimodal architecture is set up with BERT encoder and different pre-trained vision models based on convolutional neural network (CNN) and Transformer architecture for features extraction from question and endoscopy image. The result of this study highlights the dominance of Transformer-based vision models over the CNNs and demonstrates the effectiveness of the image enhancement process, with six out of the eight vision models achieving better F1-Score. Our best method, which takes advantages of BERT+BEiT fusion and image enhancement, achieves up to 87.25% accuracy and 91.85% F1-Score on the development test set, while also producing good result on the private test set with accuracy of 82.01%.

{{</citation>}}


### (78/106) SeLiNet: Sentiment enriched Lightweight Network for Emotion Recognition in Images (Tuneer Khargonkar et al., 2023)

{{<citation>}}

Tuneer Khargonkar, Shwetank Choudhary, Sumit Kumar, Barath Raj KR. (2023)  
**SeLiNet: Sentiment enriched Lightweight Network for Emotion Recognition in Images**  

---
Primary Category: cs.CV
Categories: cs-CV, cs-HC, cs.CV  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2307.02773v1)  

---


**ABSTRACT**  
In this paper, we propose a sentiment-enriched lightweight network SeLiNet and an end-to-end on-device pipeline for contextual emotion recognition in images. SeLiNet model consists of body feature extractor, image aesthetics feature extractor, and learning-based fusion network which jointly estimates discrete emotion and human sentiments tasks. On the EMOTIC dataset, the proposed approach achieves an Average Precision (AP) score of 27.17 in comparison to the baseline AP score of 27.38 while reducing the model size by >85%. In addition, we report an on-device AP score of 26.42 with reduction in model size by >93% when compared to the baseline.

{{</citation>}}


### (79/106) CityTrack: Improving City-Scale Multi-Camera Multi-Target Tracking by Location-Aware Tracking and Box-Grained Matching (Jincheng Lu et al., 2023)

{{<citation>}}

Jincheng Lu, Xipeng Yang, Jin Ye, Yifu Zhang, Zhikang Zou, Wei Zhang, Xiao Tan. (2023)  
**CityTrack: Improving City-Scale Multi-Camera Multi-Target Tracking by Location-Aware Tracking and Box-Grained Matching**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.02753v1)  

---


**ABSTRACT**  
Multi-Camera Multi-Target Tracking (MCMT) is a computer vision technique that involves tracking multiple targets simultaneously across multiple cameras. MCMT in urban traffic visual analysis faces great challenges due to the complex and dynamic nature of urban traffic scenes, where multiple cameras with different views and perspectives are often used to cover a large city-scale area. Targets in urban traffic scenes often undergo occlusion, illumination changes, and perspective changes, making it difficult to associate targets across different cameras accurately. To overcome these challenges, we propose a novel systematic MCMT framework, called CityTrack. Specifically, we present a Location-Aware SCMT tracker which integrates various advanced techniques to improve its effectiveness in the MCMT task and propose a novel Box-Grained Matching (BGM) method for the ICA module to solve the aforementioned problems. We evaluated our approach on the public test set of the CityFlowV2 dataset and achieved an IDF1 of 84.91%, ranking 1st in the 2022 AI CITY CHALLENGE. Our experimental results demonstrate the effectiveness of our approach in overcoming the challenges posed by urban traffic scenes.

{{</citation>}}


### (80/106) Active Learning with Contrastive Pre-training for Facial Expression Recognition (Shuvendu Roy et al., 2023)

{{<citation>}}

Shuvendu Roy, Ali Etemad. (2023)  
**Active Learning with Contrastive Pre-training for Facial Expression Recognition**  

---
Primary Category: cs.CV
Categories: cs-CV, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2307.02744v1)  

---


**ABSTRACT**  
Deep learning has played a significant role in the success of facial expression recognition (FER), thanks to large models and vast amounts of labelled data. However, obtaining labelled data requires a tremendous amount of human effort, time, and financial resources. Even though some prior works have focused on reducing the need for large amounts of labelled data using different unsupervised methods, another promising approach called active learning is barely explored in the context of FER. This approach involves selecting and labelling the most representative samples from an unlabelled set to make the best use of a limited 'labelling budget'. In this paper, we implement and study 8 recent active learning methods on three public FER datasets, FER13, RAF-DB, and KDEF. Our findings show that existing active learning methods do not perform well in the context of FER, likely suffering from a phenomenon called 'Cold Start', which occurs when the initial set of labelled samples is not well representative of the entire dataset. To address this issue, we propose contrastive self-supervised pre-training, which first learns the underlying representations based on the entire unlabelled dataset. We then follow this with the active learning methods and observe that our 2-step approach shows up to 9.2% improvement over random sampling and up to 6.7% improvement over the best existing active learning baseline without the pre-training. We will make the code for this study public upon publication at: github.com/ShuvenduRoy/ActiveFER.

{{</citation>}}


## cs.IR (7)



### (81/106) MultiVENT: Multilingual Videos of Events with Aligned Natural Text (Kate Sanders et al., 2023)

{{<citation>}}

Kate Sanders, David Etter, Reno Kriz, Benjamin Van Durme. (2023)  
**MultiVENT: Multilingual Videos of Events with Aligned Natural Text**  

---
Primary Category: cs.IR
Categories: cs-CV, cs-IR, cs-MM, cs.IR  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2307.03153v1)  

---


**ABSTRACT**  
Everyday news coverage has shifted from traditional broadcasts towards a wide range of presentation formats such as first-hand, unedited video footage. Datasets that reflect the diverse array of multimodal, multilingual news sources available online could be used to teach models to benefit from this shift, but existing news video datasets focus on traditional news broadcasts produced for English-speaking audiences. We address this limitation by constructing MultiVENT, a dataset of multilingual, event-centric videos grounded in text documents across five target languages. MultiVENT includes both news broadcast videos and non-professional event footage, which we use to analyze the state of online news videos and how they can be leveraged to build robust, factually accurate models. Finally, we provide a model for complex, multilingual video retrieval to serve as a baseline for information retrieval using MultiVENT.

{{</citation>}}


### (82/106) Track Mix Generation on Music Streaming Services using Transformers (Walid Bendada et al., 2023)

{{<citation>}}

Walid Bendada, Théo Bontempelli, Mathieu Morlon, Benjamin Chapus, Thibault Cador, Thomas Bouabça, Guillaume Salha-Galvan. (2023)  
**Track Mix Generation on Music Streaming Services using Transformers**  

---
Primary Category: cs.IR
Categories: cs-IR, cs-LG, cs-SD, cs.IR, eess-AS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.03045v1)  

---


**ABSTRACT**  
This paper introduces Track Mix, a personalized playlist generation system released in 2022 on the music streaming service Deezer. Track Mix automatically generates "mix" playlists inspired by initial music tracks, allowing users to discover music similar to their favorite content. To generate these mixes, we consider a Transformer model trained on millions of track sequences from user playlists. In light of the growing popularity of Transformers in recent years, we analyze the advantages, drawbacks, and technical challenges of using such a model for mix generation on the service, compared to a more traditional collaborative filtering approach. Since its release, Track Mix has been generating playlists for millions of users daily, enhancing their music discovery experience on Deezer.

{{</citation>}}


### (83/106) A Meta-Evaluation of C/W/L/A Metrics: System Ranking Similarity, System Ranking Consistency and Discriminative Power (Nuo Chen et al., 2023)

{{<citation>}}

Nuo Chen, Tetsuya Sakai. (2023)  
**A Meta-Evaluation of C/W/L/A Metrics: System Ranking Similarity, System Ranking Consistency and Discriminative Power**  

---
Primary Category: cs.IR
Categories: cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.02936v1)  

---


**ABSTRACT**  
Recently, Moffat et al. proposed an analytic framework, namely C/W/L/A, for offline evaluation metrics. This framework allows information retrieval (IR) researchers to design evaluation metrics through the flexible combination of user browsing models and user gain aggregations. However, the statistical stability of C/W/L/A metrics with different aggregations is not yet investigated. In this study, we investigate the statistical stability of C/W/L/A metrics from the perspective of: (1) the system ranking similarity among aggregations, (2) the system ranking consistency of aggregations and (3) the discriminative power of aggregations. More specifically, we combined various aggregation functions with the browsing model of Precision, Discounted Cumulative Gain (DCG), Rank-Biased Precision (RBP), INST, Average Precision (AP) and Expected Reciprocal Rank (ERR), examing their performances in terms of system ranking similarity, system ranking consistency and discriminative power on two offline test collections. Our experimental result suggests that, in terms of system ranking consistency and discriminative power, the aggregation function of expected rate of gain (ERG) has an outstanding performance while the aggregation function of maximum relevance usually has an insufficient performance. The result also suggests that Precision, DCG, RBP, INST and AP with their canonical aggregation all have favourable performances in system ranking consistency and discriminative power; but for ERR, replacing its canonical aggregation with ERG can further strengthen the discriminative power while obtaining a system ranking list similar to the canonical version at the same time.

{{</citation>}}


### (84/106) A Machine-Learned Ranking Algorithm for Dynamic and Personalised Car Pooling Services (Mattia Giovanni Campana et al., 2023)

{{<citation>}}

Mattia Giovanni Campana, Franca Delmastro, Raffaele Bruno. (2023)  
**A Machine-Learned Ranking Algorithm for Dynamic and Personalised Car Pooling Services**  

---
Primary Category: cs.IR
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.05697v1)  

---


**ABSTRACT**  
Car pooling is expected to significantly help in reducing traffic congestion and pollution in cities by enabling drivers to share their cars with travellers with similar itineraries and time schedules. A number of car pooling matching services have been designed in order to efficiently find successful ride matches in a given pool of drivers and potential passengers. However, it is now recognised that many non-monetary aspects and social considerations, besides simple mobility needs, may influence the individual willingness of sharing a ride, which are difficult to predict. To address this problem, in this study we propose GoTogether, a recommender system for car pooling services that leverages on learning-to-rank techniques to automatically derive the personalised ranking model of each user from the history of her choices (i.e., the type of accepted or rejected shared rides). Then, GoTogether builds the list of recommended rides in order to maximise the success rate of the offered matches. To test the performance of our scheme we use real data from Twitter and Foursquare sources in order to generate a dataset of plausible mobility patterns and ride requests in a metropolitan area. The results show that the proposed solution quickly obtain an accurate prediction of the personalised user's choice model both in static and dynamic conditions.

{{</citation>}}


### (85/106) PLIERS: a Popularity-Based Recommender System for Content Dissemination in Online Social Networks (Valerio Arnaboldi et al., 2023)

{{<citation>}}

Valerio Arnaboldi, Mattia Giovanni Campana, Franca Delmastro, Elena Pagani. (2023)  
**PLIERS: a Popularity-Based Recommender System for Content Dissemination in Online Social Networks**  

---
Primary Category: cs.IR
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2307.02865v1)  

---


**ABSTRACT**  
In this paper, we propose a novel tag-based recommender system called PLIERS, which relies on the assumption that users are mainly interested in items and tags with similar popularity to those they already own. PLIERS is aimed at reaching a good tradeoff between algorithmic complexity and the level of personalization of recommended items. To evaluate PLIERS, we performed a set of experiments on real OSN datasets, demonstrating that it outperforms state-of-the-art solutions in terms of personalization, relevance, and novelty of recommendations.

{{</citation>}}


### (86/106) BHEISR: Nudging from Bias to Balance -- Promoting Belief Harmony by Eliminating Ideological Segregation in Knowledge-based Recommendations (Mengyan Wang et al., 2023)

{{<citation>}}

Mengyan Wang, Yuxuan Hu, Zihan Yuan, Chenting Jiang, Weihua Li, Shiqing Wu, Quan Bai. (2023)  
**BHEISR: Nudging from Bias to Balance -- Promoting Belief Harmony by Eliminating Ideological Segregation in Knowledge-based Recommendations**  

---
Primary Category: cs.IR
Categories: 68T07, I-2-6; I-2-7, cs-AI, cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.02797v1)  

---


**ABSTRACT**  
In the realm of personalized recommendation systems, the increasing concern is the amplification of belief imbalance and user biases, a phenomenon primarily attributed to the filter bubble. Addressing this critical issue, we introduce an innovative intermediate agency (BHEISR) between users and existing recommendation systems to attenuate the negative repercussions of the filter bubble effect in extant recommendation systems. The main objective is to strike a belief balance for users while minimizing the detrimental influence caused by filter bubbles. The BHEISR model amalgamates principles from nudge theory while upholding democratic and transparent principles. It harnesses user-specific category information to stimulate curiosity, even in areas users might initially deem uninteresting. By progressively stimulating interest in novel categories, the model encourages users to broaden their belief horizons and explore the information they typically overlook. Our model is time-sensitive and operates on a user feedback loop. It utilizes the existing recommendation algorithm of the model and incorporates user feedback from the prior time frame. This approach endeavors to transcend the constraints of the filter bubble, enrich recommendation diversity, and strike a belief balance among users while also catering to user preferences and system-specific business requirements. To validate the effectiveness and reliability of the BHEISR model, we conducted a series of comprehensive experiments with real-world datasets. These experiments compared the performance of the BHEISR model against several baseline models using nearly 200 filter bubble-impacted users as test subjects. Our experimental results conclusively illustrate the superior performance of the BHEISR model in mitigating filter bubbles and balancing user perspectives.

{{</citation>}}


### (87/106) Knowledge Graph Self-Supervised Rationalization for Recommendation (Yuhao Yang et al., 2023)

{{<citation>}}

Yuhao Yang, Chao Huang, Lianghao Xia, Chunzhen Huang. (2023)  
**Knowledge Graph Self-Supervised Rationalization for Recommendation**  

---
Primary Category: cs.IR
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Knowledge Graph, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.02759v1)  

---


**ABSTRACT**  
In this paper, we introduce a new self-supervised rationalization method, called KGRec, for knowledge-aware recommender systems. To effectively identify informative knowledge connections, we propose an attentive knowledge rationalization mechanism that generates rational scores for knowledge triplets. With these scores, KGRec integrates generative and contrastive self-supervised tasks for recommendation through rational masking. To highlight rationales in the knowledge graph, we design a novel generative task in the form of masking-reconstructing. By masking important knowledge with high rational scores, KGRec is trained to rebuild and highlight useful knowledge connections that serve as rationales. To further rationalize the effect of collaborative interactions on knowledge graph learning, we introduce a contrastive learning task that aligns signals from knowledge and user-item interaction views. To ensure noise-resistant contrasting, potential noisy edges in both graphs judged by the rational scores are masked. Extensive experiments on three real-world datasets demonstrate that KGRec outperforms state-of-the-art methods. We also provide the implementation codes for our approach at https://github.com/HKUDS/KGRec.

{{</citation>}}


## cs.CY (2)



### (88/106) Frontier AI Regulation: Managing Emerging Risks to Public Safety (Markus Anderljung et al., 2023)

{{<citation>}}

Markus Anderljung, Joslyn Barnhart, Anton Korinek, Jade Leung, Cullen O'Keefe, Jess Whittlestone, Shahar Avin, Miles Brundage, Justin Bullock, Duncan Cass-Beggs, Ben Chang, Tantum Collins, Tim Fist, Gillian Hadfield, Alan Hayes, Lewis Ho, Sara Hooker, Eric Horvitz, Noam Kolt, Jonas Schuett, Yonadav Shavit, Divya Siddarth, Robert Trager, Kevin Wolf. (2023)  
**Frontier AI Regulation: Managing Emerging Risks to Public Safety**  

---
Primary Category: cs.CY
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.03718v2)  

---


**ABSTRACT**  
Advanced AI models hold the promise of tremendous benefits for humanity, but society needs to proactively manage the accompanying risks. In this paper, we focus on what we term "frontier AI" models: highly capable foundation models that could possess dangerous capabilities sufficient to pose severe risks to public safety. Frontier AI models pose a distinct regulatory challenge: dangerous capabilities can arise unexpectedly; it is difficult to robustly prevent a deployed model from being misused; and, it is difficult to stop a model's capabilities from proliferating broadly. To address these challenges, at least three building blocks for the regulation of frontier models are needed: (1) standard-setting processes to identify appropriate requirements for frontier AI developers, (2) registration and reporting requirements to provide regulators with visibility into frontier AI development processes, and (3) mechanisms to ensure compliance with safety standards for the development and deployment of frontier AI models. Industry self-regulation is an important first step. However, wider societal discussions and government intervention will be needed to create standards and to ensure compliance with them. We consider several options to this end, including granting enforcement powers to supervisory authorities and licensure regimes for frontier AI models. Finally, we propose an initial set of safety standards. These include conducting pre-deployment risk assessments; external scrutiny of model behavior; using risk assessments to inform deployment decisions; and monitoring and responding to new information about model capabilities and uses post-deployment. We hope this discussion contributes to the broader conversation on how to balance public safety risks and innovation benefits from advances at the frontier of AI development.

{{</citation>}}


### (89/106) What Should Data Science Education Do with Large Language Models? (Xinming Tu et al., 2023)

{{<citation>}}

Xinming Tu, James Zou, Weijie J. Su, Linjun Zhang. (2023)  
**What Should Data Science Education Do with Large Language Models?**  

---
Primary Category: cs.CY
Categories: cs-AI, cs-CL, cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.02792v2)  

---


**ABSTRACT**  
The rapid advances of large language models (LLMs), such as ChatGPT, are revolutionizing data science and statistics. These state-of-the-art tools can streamline complex processes. As a result, it reshapes the role of data scientists. We argue that LLMs are transforming the responsibilities of data scientists, shifting their focus from hands-on coding, data-wrangling and conducting standard analyses to assessing and managing analyses performed by these automated AIs. This evolution of roles is reminiscent of the transition from a software engineer to a product manager. We illustrate this transition with concrete data science case studies using LLMs in this paper. These developments necessitate a meaningful evolution in data science education. Pedagogy must now place greater emphasis on cultivating diverse skillsets among students, such as LLM-informed creativity, critical thinking, AI-guided programming. LLMs can also play a significant role in the classroom as interactive teaching and learning tools, contributing to personalized education. This paper discusses the opportunities, resources and open challenges for each of these directions. As with any transformative technology, integrating LLMs into education calls for careful consideration. While LLMs can perform repetitive tasks efficiently, it's crucial to remember that their role is to supplement human intelligence and creativity, not to replace it. Therefore, the new era of data science education should balance the benefits of LLMs while fostering complementary human expertise and innovations. In conclusion, the rise of LLMs heralds a transformative period for data science and its education. This paper seeks to shed light on the emerging trends, potential opportunities, and challenges accompanying this paradigm shift, hoping to spark further discourse and investigation into this exciting, uncharted territory.

{{</citation>}}


## cs.AI (2)



### (90/106) Structure Guided Multi-modal Pre-trained Transformer for Knowledge Graph Reasoning (Ke Liang et al., 2023)

{{<citation>}}

Ke Liang, Sihang Zhou, Yue Liu, Lingyuan Meng, Meng Liu, Xinwang Liu. (2023)  
**Structure Guided Multi-modal Pre-trained Transformer for Knowledge Graph Reasoning**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-IR, cs.AI  
Keywords: Knowledge Graph, Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2307.03591v1)  

---


**ABSTRACT**  
Multimodal knowledge graphs (MKGs), which intuitively organize information in various modalities, can benefit multiple practical downstream tasks, such as recommendation systems, and visual question answering. However, most MKGs are still far from complete, which motivates the flourishing of MKG reasoning models. Recently, with the development of general artificial architectures, the pretrained transformer models have drawn increasing attention, especially for multimodal scenarios. However, the research of multimodal pretrained transformer (MPT) for knowledge graph reasoning (KGR) is still at an early stage. As the biggest difference between MKG and other multimodal data, the rich structural information underlying the MKG still cannot be fully leveraged in existing MPT models. Most of them only utilize the graph structure as a retrieval map for matching images and texts connected with the same entity. This manner hinders their reasoning performances. To this end, we propose the graph Structure Guided Multimodal Pretrained Transformer for knowledge graph reasoning, termed SGMPT. Specifically, the graph structure encoder is adopted for structural feature encoding. Then, a structure-guided fusion module with two different strategies, i.e., weighted summation and alignment constraint, is first designed to inject the structural information into both the textual and visual features. To the best of our knowledge, SGMPT is the first MPT model for multimodal KGR, which mines the structural information underlying the knowledge graph. Extensive experiments on FB15k-237-IMG and WN18-IMG, demonstrate that our SGMPT outperforms existing state-of-the-art models, and prove the effectiveness of the designed strategies.

{{</citation>}}


### (91/106) RecallM: An Architecture for Temporal Context Understanding and Question Answering (Brandon Kynoch et al., 2023)

{{<citation>}}

Brandon Kynoch, Hugo Latapie. (2023)  
**RecallM: An Architecture for Temporal Context Understanding and Question Answering**  

---
Primary Category: cs.AI
Categories: cs-AI, cs-CL, cs-SC, cs.AI  
Keywords: Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.02738v2)  

---


**ABSTRACT**  
The ideal long-term memory mechanism for Large Language Model (LLM) based chatbots, would lay the foundation for continual learning, complex reasoning and allow sequential and temporal dependencies to be learnt. Creating this type of memory mechanism is an extremely challenging problem. In this paper we explore different methods of achieving the effect of long-term memory. We propose a new architecture focused on creating adaptable and updatable long-term memory for AGI systems. We demonstrate through various experiments the benefits of the RecallM architecture, particularly the improved temporal understanding of knowledge it provides.

{{</citation>}}


## cs.NE (1)



### (92/106) A Neuromorphic Architecture for Reinforcement Learning from Real-Valued Observations (Sergio F. Chevtchenko et al., 2023)

{{<citation>}}

Sergio F. Chevtchenko, Yeshwanth Bethi, Teresa B. Ludermir, Saeed Afshar. (2023)  
**A Neuromorphic Architecture for Reinforcement Learning from Real-Valued Observations**  

---
Primary Category: cs.NE
Categories: cs-AI, cs-NE, cs.NE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02947v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL) provides a powerful framework for decision-making in complex environments. However, implementing RL in hardware-efficient and bio-inspired ways remains a challenge. This paper presents a novel Spiking Neural Network (SNN) architecture for solving RL problems with real-valued observations. The proposed model incorporates multi-layered event-based clustering, with the addition of Temporal Difference (TD)-error modulation and eligibility traces, building upon prior work. An ablation study confirms the significant impact of these components on the proposed model's performance. A tabular actor-critic algorithm with eligibility traces and a state-of-the-art Proximal Policy Optimization (PPO) algorithm are used as benchmarks. Our network consistently outperforms the tabular approach and successfully discovers stable control policies on classic RL environments: mountain car, cart-pole, and acrobot. The proposed model offers an appealing trade-off in terms of computational and hardware implementation requirements. The model does not require an external memory buffer nor a global error gradient computation, and synaptic updates occur online, driven by local learning rules and a broadcasted TD-error signal. Thus, this work contributes to the development of more hardware-efficient RL solutions.

{{</citation>}}


## cs.RO (3)



### (93/106) AllSight: A Low-Cost and High-Resolution Round Tactile Sensor with Zero-Shot Learning Capability (Osher Azulay et al., 2023)

{{<citation>}}

Osher Azulay, Nimrod Curtis, Rotem Sokolovsky, Guy Levitski, Daniel Slomovik, Guy Lilling, Avishai Sintov. (2023)  
**AllSight: A Low-Cost and High-Resolution Round Tactile Sensor with Zero-Shot Learning Capability**  

---
Primary Category: cs.RO
Categories: cs-RO, cs.RO  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.02928v1)  

---


**ABSTRACT**  
Tactile sensing is a necessary capability for a robotic hand to perform fine manipulations and interact with the environment. Optical sensors are a promising solution for high-resolution contact estimation. Nevertheless, they are usually not easy to fabricate and require individual calibration in order to acquire sufficient accuracy. In this letter, we propose AllSight, an optical tactile sensor with a round 3D structure potentially designed for robotic in-hand manipulation tasks. AllSight is mostly 3D printed making it low-cost, modular, durable and in the size of a human thumb while with a large contact surface. We show the ability of AllSight to learn and estimate a full contact state, i.e., contact position, forces and torsion. With that, an experimental benchmark between various configurations of illumination and contact elastomers are provided. Furthermore, the robust design of AllSight provides it with a unique zero-shot capability such that a practitioner can fabricate the open-source design and have a ready-to-use state estimation model. A set of experiments demonstrates the accurate state estimation performance of AllSight.

{{</citation>}}


### (94/106) Learning to Solve Tasks with Exploring Prior Behaviours (Ruiqi Zhu et al., 2023)

{{<citation>}}

Ruiqi Zhu, Siyuan Li, Tianhong Dai, Chongjie Zhang, Oya Celiktutan. (2023)  
**Learning to Solve Tasks with Exploring Prior Behaviours**  

---
Primary Category: cs.RO
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02889v1)  

---


**ABSTRACT**  
Demonstrations are widely used in Deep Reinforcement Learning (DRL) for facilitating solving tasks with sparse rewards. However, the tasks in real-world scenarios can often have varied initial conditions from the demonstration, which would require additional prior behaviours. For example, consider we are given the demonstration for the task of \emph{picking up an object from an open drawer}, but the drawer is closed in the training. Without acquiring the prior behaviours of opening the drawer, the robot is unlikely to solve the task. To address this, in this paper we propose an Intrinsic Rewards Driven Example-based Control \textbf{(IRDEC)}. Our method can endow agents with the ability to explore and acquire the required prior behaviours and then connect to the task-specific behaviours in the demonstration to solve sparse-reward tasks without requiring additional demonstration of the prior behaviours. The performance of our method outperforms other baselines on three navigation tasks and one robotic manipulation task with sparse rewards. Codes are available at https://github.com/Ricky-Zhu/IRDEC.

{{</citation>}}


### (95/106) Contrastive Label Disambiguation for Self-Supervised Terrain Traversability Learning in Off-Road Environments (Hanzhang Xue et al., 2023)

{{<citation>}}

Hanzhang Xue, Xiaochang Hu, Rui Xie, Hao Fu, Liang Xiao, Yiming Nie, Bin Dai. (2023)  
**Contrastive Label Disambiguation for Self-Supervised Terrain Traversability Learning in Off-Road Environments**  

---
Primary Category: cs.RO
Categories: cs-RO, cs.RO  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.02871v1)  

---


**ABSTRACT**  
Discriminating the traversability of terrains is a crucial task for autonomous driving in off-road environments. However, it is challenging due to the diverse, ambiguous, and platform-specific nature of off-road traversability. In this paper, we propose a novel self-supervised terrain traversability learning framework, utilizing a contrastive label disambiguation mechanism. Firstly, weakly labeled training samples with pseudo labels are automatically generated by projecting actual driving experiences onto the terrain models constructed in real time. Subsequently, a prototype-based contrastive representation learning method is designed to learn distinguishable embeddings, facilitating the self-supervised updating of those pseudo labels. As the iterative interaction between representation learning and pseudo label updating, the ambiguities in those pseudo labels are gradually eliminated, enabling the learning of platform-specific and task-specific traversability without any human-provided annotations. Experimental results on the RELLIS-3D dataset and our Gobi Desert driving dataset demonstrate the effectiveness of the proposed method.

{{</citation>}}


## eess.SP (2)



### (96/106) Meta Federated Reinforcement Learning for Distributed Resource Allocation (Zelin Ji et al., 2023)

{{<citation>}}

Zelin Ji, Zhijin Qin, Xiaoming Tao. (2023)  
**Meta Federated Reinforcement Learning for Distributed Resource Allocation**  

---
Primary Category: eess.SP
Categories: cs-SY, eess-SP, eess-SY, eess.SP  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02900v2)  

---


**ABSTRACT**  
In cellular networks, resource allocation is usually performed in a centralized way, which brings huge computation complexity to the base station (BS) and high transmission overhead. This paper explores a distributed resource allocation method that aims to maximize energy efficiency (EE) while ensuring the quality of service (QoS) for users. Specifically, in order to address wireless channel conditions, we propose a robust meta federated reinforcement learning (\textit{MFRL}) framework that allows local users to optimize transmit power and assign channels using locally trained neural network models, so as to offload computational burden from the cloud server to the local users, reducing transmission overhead associated with local channel state information. The BS performs the meta learning procedure to initialize a general global model, enabling rapid adaptation to different environments with improved EE performance. The federated learning technique, based on decentralized reinforcement learning, promotes collaboration and mutual benefits among users. Analysis and numerical results demonstrate that the proposed \textit{MFRL} framework accelerates the reinforcement learning process, decreases transmission overhead, and offloads computation, while outperforming the conventional decentralized reinforcement learning algorithm in terms of convergence speed and EE performance across various scenarios.

{{</citation>}}


### (97/106) UniCoRN: Unified Cognitive Signal ReconstructioN bridging cognitive signals and human language (Nuwa Xi et al., 2023)

{{<citation>}}

Nuwa Xi, Sendong Zhao, Haochun Wang, Chi Liu, Bing Qin, Ting Liu. (2023)  
**UniCoRN: Unified Cognitive Signal ReconstructioN bridging cognitive signals and human language**  

---
Primary Category: eess.SP
Categories: cs-CL, eess-SP, eess.SP  
Keywords: BLEU  
[Paper Link](http://arxiv.org/abs/2307.05355v1)  

---


**ABSTRACT**  
Decoding text stimuli from cognitive signals (e.g. fMRI) enhances our understanding of the human language system, paving the way for building versatile Brain-Computer Interface. However, existing studies largely focus on decoding individual word-level fMRI volumes from a restricted vocabulary, which is far too idealized for real-world application. In this paper, we propose fMRI2text, the first openvocabulary task aiming to bridge fMRI time series and human language. Furthermore, to explore the potential of this new task, we present a baseline solution, UniCoRN: the Unified Cognitive Signal ReconstructioN for Brain Decoding. By reconstructing both individual time points and time series, UniCoRN establishes a robust encoder for cognitive signals (fMRI & EEG). Leveraging a pre-trained language model as decoder, UniCoRN proves its efficacy in decoding coherent text from fMRI series across various split settings. Our model achieves a 34.77% BLEU score on fMRI2text, and a 37.04% BLEU when generalized to EEGto-text decoding, thereby surpassing the former baseline. Experimental results indicate the feasibility of decoding consecutive fMRI volumes, and the effectiveness of decoding different cognitive signals using a unified structure.

{{</citation>}}


## cs.DB (2)



### (98/106) Scaling Package Queries to a Billion Tuples via Hierarchical Partitioning and Customized Optimization (Anh L. Mai et al., 2023)

{{<citation>}}

Anh L. Mai, Pengyu Wang, Azza Abouzied, Matteo Brucato, Peter J. Haas, Alexandra Meliou. (2023)  
**Scaling Package Queries to a Billion Tuples via Hierarchical Partitioning and Customized Optimization**  

---
Primary Category: cs.DB
Categories: cs-DB, cs.DB  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2307.02860v2)  

---


**ABSTRACT**  
A package query returns a package -- a multiset of tuples -- that maximizes or minimizes a linear objective function subject to linear constraints, thereby enabling in-database decision support. Prior work has established the equivalence of package queries to Integer Linear Programs (ILPs) and developed the SketchRefine algorithm for package query processing. While this algorithm was an important first step toward supporting prescriptive analytics scalably inside a relational database, it struggles when the data size grows beyond a few hundred million tuples or when the constraints become very tight. In this paper, we present Progressive Shading, a novel algorithm for processing package queries that can scale efficiently to billions of tuples and gracefully handle tight constraints. Progressive Shading solves a sequence of optimization problems over a hierarchy of relations, each resulting from an ever-finer partitioning of the original tuples into homogeneous groups until the original relation is obtained. This strategy avoids the premature discarding of high-quality tuples that can occur with SketchRefine. Our novel partitioning scheme, Dynamic Low Variance, can handle very large relations with multiple attributes and can dynamically adapt to both concentrated and spread-out sets of attribute values, provably outperforming traditional partitioning schemes such as KD-Tree. We further optimize our system by replacing our off-the-shelf optimization software with customized ILP and LP solvers, called Dual Reducer and Parallel Dual Simplex respectively, that are highly accurate and orders of magnitude faster.

{{</citation>}}


### (99/106) VerifAI: Verified Generative AI (Nan Tang et al., 2023)

{{<citation>}}

Nan Tang, Chenyu Yang, Ju Fan, Lei Cao. (2023)  
**VerifAI: Verified Generative AI**  

---
Primary Category: cs.DB
Categories: cs-CL, cs-DB, cs-LG, cs.DB  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2307.02796v1)  

---


**ABSTRACT**  
Generative AI has made significant strides, yet concerns about the accuracy and reliability of its outputs continue to grow. Such inaccuracies can have serious consequences such as inaccurate decision-making, the spread of false information, privacy violations, legal liabilities, and more. Although efforts to address these risks are underway, including explainable AI and responsible AI practices such as transparency, privacy protection, bias mitigation, and social and environmental responsibility, misinformation caused by generative AI will remain a significant challenge. We propose that verifying the outputs of generative AI from a data management perspective is an emerging issue for generative AI. This involves analyzing the underlying data from multi-modal data lakes, including text files, tables, and knowledge graphs, and assessing its quality and consistency. By doing so, we can establish a stronger foundation for evaluating the outputs of generative AI models. Such an approach can ensure the correctness of generative AI, promote transparency, and enable decision-making with greater confidence. Our vision is to promote the development of verifiable generative AI and contribute to a more trustworthy and responsible use of AI.

{{</citation>}}


## cs.IT (2)



### (100/106) Cell-Free XL-MIMO Meets Multi-Agent Reinforcement Learning: Architectures, Challenges, and Future Directions (Zhilong Liu et al., 2023)

{{<citation>}}

Zhilong Liu, Jiayi Zhang, Ziheng Liu, Hongyang Du, Zhe Wang, Dusit Niyato, Mohsen Guizani, Bo Ai. (2023)  
**Cell-Free XL-MIMO Meets Multi-Agent Reinforcement Learning: Architectures, Challenges, and Future Directions**  

---
Primary Category: cs.IT
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02827v1)  

---


**ABSTRACT**  
Cell-free massive multiple-input multiple-output (mMIMO) and extremely large-scale MIMO (XL-MIMO) are regarded as promising innovations for the forthcoming generation of wireless communication systems. Their significant advantages in augmenting the number of degrees of freedom have garnered considerable interest. In this article, we first review the essential opportunities and challenges induced by XL-MIMO systems. We then propose the enhanced paradigm of cell-free XL-MIMO, which incorporates multi-agent reinforcement learning (MARL) to provide a distributed strategy for tackling the problem of high-dimension signal processing and costly energy consumption. Based on the unique near-field characteristics, we propose two categories of the low-complexity design, i.e., antenna selection and power control, to adapt to different cell-free XL-MIMO scenarios and achieve the maximum data rate. For inspiration, several critical future research directions pertaining to green cell-free XL-MIMO systems are presented.

{{</citation>}}


### (101/106) Large Language Models Empowered Autonomous Edge AI for Connected Intelligence (Yifei Shen et al., 2023)

{{<citation>}}

Yifei Shen, Jiawei Shao, Xinjie Zhang, Zehong Lin, Hao Pan, Dongsheng Li, Jun Zhang, Khaled B. Letaief. (2023)  
**Large Language Models Empowered Autonomous Edge AI for Connected Intelligence**  

---
Primary Category: cs.IT
Categories: cs-IT, cs-LG, cs-NI, cs.IT, eess-SP, math-IT  
Keywords: AI, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2307.02779v1)  

---


**ABSTRACT**  
The evolution of wireless networks gravitates towards connected intelligence, a concept that envisions seamless interconnectivity among humans, objects, and intelligence in a hyper-connected cyber-physical world. Edge AI emerges as a promising solution to achieve connected intelligence by delivering high-quality, low-latency, and privacy-preserving AI services at the network edge. In this article, we introduce an autonomous edge AI system that automatically organizes, adapts, and optimizes itself to meet users' diverse requirements. The system employs a cloud-edge-client hierarchical architecture, where the large language model, i.e., Generative Pretrained Transformer (GPT), resides in the cloud, and other AI models are co-deployed on devices and edge servers. By leveraging the powerful abilities of GPT in language understanding, planning, and code generation, we present a versatile framework that efficiently coordinates edge AI models to cater to users' personal demands while automatically generating code to train new models via edge federated learning. Experimental results demonstrate the system's remarkable ability to accurately comprehend user demands, efficiently execute AI models with minimal cost, and effectively create high-performance AI models through federated learning.

{{</citation>}}


## cs.SD (1)



### (102/106) Evaluating raw waveforms with deep learning frameworks for speech emotion recognition (Zeynep Hilal Kilimci et al., 2023)

{{<citation>}}

Zeynep Hilal Kilimci, Ulku Bayraktar, Ayhan Kucukmanisa. (2023)  
**Evaluating raw waveforms with deep learning frameworks for speech emotion recognition**  

---
Primary Category: cs.SD
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.02820v1)  

---


**ABSTRACT**  
Speech emotion recognition is a challenging task in speech processing field. For this reason, feature extraction process has a crucial importance to demonstrate and process the speech signals. In this work, we represent a model, which feeds raw audio files directly into the deep neural networks without any feature extraction stage for the recognition of emotions utilizing six different data sets, EMO-DB, RAVDESS, TESS, CREMA, SAVEE, and TESS+RAVDESS. To demonstrate the contribution of proposed model, the performance of traditional feature extraction techniques namely, mel-scale spectogram, mel-frequency cepstral coefficients, are blended with machine learning algorithms, ensemble learning methods, deep and hybrid deep learning techniques. Support vector machine, decision tree, naive Bayes, random forests models are evaluated as machine learning algorithms while majority voting and stacking methods are assessed as ensemble learning techniques. Moreover, convolutional neural networks, long short-term memory networks, and hybrid CNN- LSTM model are evaluated as deep learning techniques and compared with machine learning and ensemble learning methods. To demonstrate the effectiveness of proposed model, the comparison with state-of-the-art studies are carried out. Based on the experiment results, CNN model excels existent approaches with 95.86% of accuracy for TESS+RAVDESS data set using raw audio files, thence determining the new state-of-the-art. The proposed model performs 90.34% of accuracy for EMO-DB with CNN model, 90.42% of accuracy for RAVDESS with CNN model, 99.48% of accuracy for TESS with LSTM model, 69.72% of accuracy for CREMA with CNN model, 85.76% of accuracy for SAVEE with CNN model in speaker-independent audio categorization problems.

{{</citation>}}


## eess.IV (2)



### (103/106) Advancing Zero-Shot Digital Human Quality Assessment through Text-Prompted Evaluation (Zicheng Zhang et al., 2023)

{{<citation>}}

Zicheng Zhang, Wei Sun, Yingjie Zhou, Haoning Wu, Chunyi Li, Xiongkuo Min, Xiaohong Liu, Guangtao Zhai, Weisi Lin. (2023)  
**Advancing Zero-Shot Digital Human Quality Assessment through Text-Prompted Evaluation**  

---
Primary Category: eess.IV
Categories: cs-CV, cs-DB, eess-IV, eess.IV  
Keywords: QA, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.02808v1)  

---


**ABSTRACT**  
Digital humans have witnessed extensive applications in various domains, necessitating related quality assessment studies. However, there is a lack of comprehensive digital human quality assessment (DHQA) databases. To address this gap, we propose SJTU-H3D, a subjective quality assessment database specifically designed for full-body digital humans. It comprises 40 high-quality reference digital humans and 1,120 labeled distorted counterparts generated with seven types of distortions. The SJTU-H3D database can serve as a benchmark for DHQA research, allowing evaluation and refinement of processing algorithms. Further, we propose a zero-shot DHQA approach that focuses on no-reference (NR) scenarios to ensure generalization capabilities while mitigating database bias. Our method leverages semantic and distortion features extracted from projections, as well as geometry features derived from the mesh structure of digital humans. Specifically, we employ the Contrastive Language-Image Pre-training (CLIP) model to measure semantic affinity and incorporate the Naturalness Image Quality Evaluator (NIQE) model to capture low-level distortion information. Additionally, we utilize dihedral angles as geometry descriptors to extract mesh features. By aggregating these measures, we introduce the Digital Human Quality Index (DHQI), which demonstrates significant improvements in zero-shot performance. The DHQI can also serve as a robust baseline for DHQA tasks, facilitating advancements in the field. The database and the code are available at https://github.com/zzc-1998/SJTU-H3D.

{{</citation>}}


### (104/106) Few-Shot Personalized Saliency Prediction Using Tensor Regression for Preserving Structural Global Information (Yuya Moroto et al., 2023)

{{<citation>}}

Yuya Moroto, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama. (2023)  
**Few-Shot Personalized Saliency Prediction Using Tensor Regression for Preserving Structural Global Information**  

---
Primary Category: eess.IV
Categories: cs-LG, eess-IV, eess.IV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.02799v1)  

---


**ABSTRACT**  
This paper presents a few-shot personalized saliency prediction using tensor-to-matrix regression for preserving the structural global information of personalized saliency maps (PSMs). In contrast to a general saliency map, a PSM has been great potential since its map indicates the person-specific visual attention that is useful for obtaining individual visual preferences from heterogeneity of gazed areas. The PSM prediction is needed for acquiring the PSM for the unseen image, but its prediction is still a challenging task due to the complexity of individual gaze patterns. For recognizing individual gaze patterns from the limited amount of eye-tracking data, the previous methods adopt the similarity of gaze tendency between persons. However, in the previous methods, the PSMs are vectorized for the prediction model. In this way, the structural global information of the PSMs corresponding to the image is ignored. For automatically revealing the relationship between PSMs, we focus on the tensor-based regression model that can preserve the structural information of PSMs, and realize the improvement of the prediction accuracy. In the experimental results, we confirm the proposed method including the tensor-based regression outperforms the comparative methods.

{{</citation>}}


## cs.MA (1)



### (105/106) Wireless Multi-Agent Generative AI: From Connected Intelligence to Collective Intelligence (Hang Zou et al., 2023)

{{<citation>}}

Hang Zou, Qiyang Zhao, Lina Bariah, Mehdi Bennis, Merouane Debbah. (2023)  
**Wireless Multi-Agent Generative AI: From Connected Intelligence to Collective Intelligence**  

---
Primary Category: cs.MA
Categories: cs-MA, cs.MA  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2307.02757v1)  

---


**ABSTRACT**  
The convergence of generative large language models (LLMs), edge networks, and multi-agent systems represents a groundbreaking synergy that holds immense promise for future wireless generations, harnessing the power of collective intelligence and paving the way for self-governed networks where intelligent decision-making happens right at the edge. This article puts the stepping-stone for incorporating multi-agent generative artificial intelligence (AI) in wireless networks, and sets the scene for realizing on-device LLMs, where multi-agent LLMs are collaboratively planning and solving tasks to achieve a number of network goals. We further investigate the profound limitations of cloud-based LLMs, and explore multi-agent LLMs from a game theoretic perspective, where agents collaboratively solve tasks in competitive environments. Moreover, we establish the underpinnings for the architecture design of wireless multi-agent generative AI systems at the network level and the agent level, and we identify the wireless technologies that are envisioned to play a key role in enabling on-device LLM. To demonstrate the promising potentials of wireless multi-agent generative AI networks, we highlight the benefits that can be achieved when implementing wireless generative agents in intent-based networking, and we provide a case study to showcase how on-device LLMs can contribute to solving network intents in a collaborative fashion. We finally shed lights on potential challenges and sketch a research roadmap towards realizing the vision of wireless collective intelligence.

{{</citation>}}


## cs.NI (1)



### (106/106) Intent-driven Intelligent Control and Orchestration in O-RAN Via Hierarchical Reinforcement Learning (Md Arafat Habib et al., 2023)

{{<citation>}}

Md Arafat Habib, Hao Zhou, Pedro Enrique Iturria-Rivera, Medhat Elsayed, Majid Bavand, Raimundas Gaigalas, Yigit Ozcan, Melike Erol-Kantarci. (2023)  
**Intent-driven Intelligent Control and Orchestration in O-RAN Via Hierarchical Reinforcement Learning**  

---
Primary Category: cs.NI
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.02754v1)  

---


**ABSTRACT**  
rApps and xApps need to be controlled and orchestrated well in the open radio access network (O-RAN) so that they can deliver a guaranteed network performance in a complex multi-vendor environment. This paper proposes a novel intent-driven intelligent control and orchestration scheme based on hierarchical reinforcement learning (HRL). The proposed scheme can orchestrate multiple rApps or xApps according to the operator's intent of optimizing certain key performance indicators (KPIs), such as throughput, energy efficiency, and latency. Specifically, we propose a bi-level architecture with a meta-controller and a controller. The meta-controller provides the target performance in terms of KPIs, while the controller performs xApp orchestration at the lower level. Our simulation results show that the proposed HRL-based intent-driven xApp orchestration mechanism achieves 7.5% and 21.4% increase in average system throughput with respect to two baselines, i.e., a single xApp baseline and a non-machine learning-based algorithm, respectively. Similarly, 17.3% and 37.9% increase in energy efficiency are observed in comparison to the same baselines.

{{</citation>}}
