---
draft: false
title: "arXiv @ 2023.12.29"
date: 2023-12-29
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.29"
    identifier: arxiv_20231229
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.DC (2)](#csdc-2)
- [cs.LG (14)](#cslg-14)
- [cs.CL (12)](#cscl-12)
- [cs.CR (1)](#cscr-1)
- [cs.IR (2)](#csir-2)
- [cs.CV (13)](#cscv-13)
- [cs.AI (2)](#csai-2)
- [cs.HC (1)](#cshc-1)
- [eess.SY (1)](#eesssy-1)
- [cs.SD (2)](#cssd-2)
- [q-bio.GN (1)](#q-biogn-1)
- [cs.SE (2)](#csse-2)
- [cs.SI (2)](#cssi-2)
- [cs.ET (1)](#cset-1)
- [eess.IV (1)](#eessiv-1)
- [cs.RO (1)](#csro-1)
- [cs.AR (1)](#csar-1)

## cs.DC (2)



### (1/59) SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads (Alind Khare et al., 2023)

{{<citation>}}

Alind Khare, Dhruv Garg, Sukrit Kalra, Snigdha Grandhi, Ion Stoica, Alexey Tumanov. (2023)  
**SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-LG, cs.DC  
Keywords: Azure, Microsoft  
[Paper Link](http://arxiv.org/abs/2312.16733v1)  

---


**ABSTRACT**  
The increasing deployment of ML models on the critical path of production applications in both datacenter and the edge requires ML inference serving systems to serve these models under unpredictable and bursty request arrival rates. Serving models under such conditions requires these systems to strike a careful balance between the latency and accuracy requirements of the application and the overall efficiency of utilization of scarce resources. State-of-the-art systems resolve this tension by either choosing a static point in the latency-accuracy tradeoff space to serve all requests or load specific models on the critical path of request serving. In this work, we instead resolve this tension by simultaneously serving the entire-range of models spanning the latency-accuracy tradeoff space. Our novel mechanism, SubNetAct, achieves this by carefully inserting specialized operators in weight-shared SuperNetworks. These operators enable SubNetAct to dynamically route requests through the network to meet a latency and accuracy target. SubNetAct requires upto 2.6x lower memory to serve a vastly-higher number of models than prior state-of-the-art. In addition, SubNetAct's near-instantaneous actuation of models unlocks the design space of fine-grained, reactive scheduling policies. We explore the design of one such extremely effective policy, SlackFit and instantiate both SubNetAct and SlackFit in a real system, SuperServe. SuperServe achieves 4.67% higher accuracy for the same SLO attainment and 2.85x higher SLO attainment for the same accuracy on a trace derived from the real-world Microsoft Azure Functions workload and yields the best trade-offs on a wide range of extremely-bursty synthetic traces automatically.

{{</citation>}}


### (2/59) Analytical Insight of Earth: A Cloud-Platform of Intelligent Computing for Geospatial Big Data (Hao Xu et al., 2023)

{{<citation>}}

Hao Xu, Yuanbin Man, Mingyang Yang, Jichao Wu, Qi Zhang, Jing Wang. (2023)  
**Analytical Insight of Earth: A Cloud-Platform of Intelligent Computing for Geospatial Big Data**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.16385v1)  

---


**ABSTRACT**  
The rapid accumulation of Earth observation data presents a formidable challenge for the processing capabilities of traditional remote sensing desktop software, particularly when it comes to analyzing expansive geographical areas and prolonged temporal sequences. Cloud computing has emerged as a transformative solution, surmounting the barriers traditionally associated with the management and computation of voluminous datasets. This paper introduces the Analytical Insight of Earth (AI Earth), an innovative remote sensing intelligent computing cloud platform, powered by the robust Alibaba Cloud infrastructure. AI Earth provides an extensive collection of publicly available remote sensing datasets, along with a suite of computational tools powered by a high-performance computing engine. Furthermore, it provides a variety of classic deep learning (DL) models and a novel remote sensing large vision segmentation model tailored to different recognition tasks. The platform enables users to upload their unique samples for model training and to deploy third-party models, thereby increasing the accessibility and openness of DL applications. This platform will facilitate researchers in leveraging remote sensing data for large-scale applied research in areas such as resources, environment, ecology, and climate.

{{</citation>}}


## cs.LG (14)



### (3/59) Foundations of Reinforcement Learning and Interactive Decision Making (Dylan J. Foster et al., 2023)

{{<citation>}}

Dylan J. Foster, Alexander Rakhlin. (2023)  
**Foundations of Reinforcement Learning and Interactive Decision Making**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC, math-ST, stat-ML, stat-TH  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.16730v1)  

---


**ABSTRACT**  
These lecture notes give a statistical perspective on the foundations of reinforcement learning and interactive decision making. We present a unifying framework for addressing the exploration-exploitation dilemma using frequentist and Bayesian approaches, with connections and parallels between supervised learning/estimation and decision making as an overarching theme. Special attention is paid to function approximation and flexible model classes such as neural networks. Topics covered include multi-armed and contextual bandits, structured bandits, and reinforcement learning with high-dimensional feedback.

{{</citation>}}


### (4/59) FairCompass: Operationalising Fairness in Machine Learning (Jessica Liu et al., 2023)

{{<citation>}}

Jessica Liu, Huaming Chen, Jun Shen, Kim-Kwang Raymond Choo. (2023)  
**FairCompass: Operationalising Fairness in Machine Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs-SE, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.16726v1)  

---


**ABSTRACT**  
As artificial intelligence (AI) increasingly becomes an integral part of our societal and individual activities, there is a growing imperative to develop responsible AI solutions. Despite a diverse assortment of machine learning fairness solutions is proposed in the literature, there is reportedly a lack of practical implementation of these tools in real-world applications. Industry experts have participated in thorough discussions on the challenges associated with operationalising fairness in the development of machine learning-empowered solutions, in which a shift toward human-centred approaches is promptly advocated to mitigate the limitations of existing techniques. In this work, we propose a human-in-the-loop approach for fairness auditing, presenting a mixed visual analytical system (hereafter referred to as 'FairCompass'), which integrates both subgroup discovery technique and the decision tree-based schema for end users. Moreover, we innovatively integrate an Exploration, Guidance and Informed Analysis loop, to facilitate the use of the Knowledge Generation Model for Visual Analytics in FairCompass. We evaluate the effectiveness of FairCompass for fairness auditing in a real-world scenario, and the findings demonstrate the system's potential for real-world deployability. We anticipate this work will address the current gaps in research for fairness and facilitate the operationalisation of fairness in machine learning systems.

{{</citation>}}


### (5/59) Knowledge Enhanced Conditional Imputation for Healthcare Time-series (Linglong Qian et al., 2023)

{{<citation>}}

Linglong Qian, Zina Ibrahim, Hugh Logan Ellis, Ao Zhang, Yuezhou Zhang, Tao Wang, Richard Dobson. (2023)  
**Knowledge Enhanced Conditional Imputation for Healthcare Time-series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2312.16713v1)  

---


**ABSTRACT**  
This study presents a novel approach to addressing the challenge of missing data in multivariate time series, with a particular focus on the complexities of healthcare data. Our Conditional Self-Attention Imputation (CSAI) model, grounded in a transformer-based framework, introduces a conditional hidden state initialization tailored to the intricacies of medical time series data. This methodology diverges from traditional imputation techniques by specifically targeting the imbalance in missing data distribution, a crucial aspect often overlooked in healthcare datasets. By integrating advanced knowledge embedding and a non-uniform masking strategy, CSAI adeptly adjusts to the distinct patterns of missing data in Electronic Health Records (EHRs).

{{</citation>}}


### (6/59) Twice Class Bias Correction for Imbalanced Semi-Supervised Learning (Lan Li et al., 2023)

{{<citation>}}

Lan Li, Bowen Tao, Lu Han, De-chuan Zhan, Han-jia Ye. (2023)  
**Twice Class Bias Correction for Imbalanced Semi-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.16604v1)  

---


**ABSTRACT**  
Differing from traditional semi-supervised learning, class-imbalanced semi-supervised learning presents two distinct challenges: (1) The imbalanced distribution of training samples leads to model bias towards certain classes, and (2) the distribution of unlabeled samples is unknown and potentially distinct from that of labeled samples, which further contributes to class bias in the pseudo-labels during training. To address these dual challenges, we introduce a novel approach called \textbf{T}wice \textbf{C}lass \textbf{B}ias \textbf{C}orrection (\textbf{TCBC}). We begin by utilizing an estimate of the class distribution from the participating training samples to correct the model, enabling it to learn the posterior probabilities of samples under a class-balanced prior. This correction serves to alleviate the inherent class bias of the model. Building upon this foundation, we further estimate the class bias of the current model parameters during the training process. We apply a secondary correction to the model's pseudo-labels for unlabeled samples, aiming to make the assignment of pseudo-labels across different classes of unlabeled samples as equitable as possible. Through extensive experimentation on CIFAR10/100-LT, STL10-LT, and the sizable long-tailed dataset SUN397, we provide conclusive evidence that our proposed TCBC method reliably enhances the performance of class-imbalanced semi-supervised learning.

{{</citation>}}


### (7/59) Continuous-time Autoencoders for Regular and Irregular Time Series Imputation (Hyowon Wi et al., 2023)

{{<citation>}}

Hyowon Wi, Yehjin Shin, Noseong Park. (2023)  
**Continuous-time Autoencoders for Regular and Irregular Time Series Imputation**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.16581v2)  

---


**ABSTRACT**  
Time series imputation is one of the most fundamental tasks for time series. Real-world time series datasets are frequently incomplete (or irregular with missing observations), in which case imputation is strongly required. Many different time series imputation methods have been proposed. Recent self-attention-based methods show the state-of-the-art imputation performance. However, it has been overlooked for a long time to design an imputation method based on continuous-time recurrent neural networks (RNNs), i.e., neural controlled differential equations (NCDEs). To this end, we redesign time series (variational) autoencoders based on NCDEs. Our method, called continuous-time autoencoder (CTA), encodes an input time series sample into a continuous hidden path (rather than a hidden vector) and decodes it to reconstruct and impute the input. In our experiments with 4 datasets and 19 baselines, our method shows the best imputation performance in almost all cases.

{{</citation>}}


### (8/59) Inverse Reinforcement Learning with Unknown Reward Model based on Structural Risk Minimization (Chendi Qu et al., 2023)

{{<citation>}}

Chendi Qu, Jianping He, Xiaoming Duan, Jiming Chen. (2023)  
**Inverse Reinforcement Learning with Unknown Reward Model based on Structural Risk Minimization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.16566v1)  

---


**ABSTRACT**  
Inverse reinforcement learning (IRL) usually assumes the model of the reward function is pre-specified and estimates the parameter only. However, how to determine a proper reward model is nontrivial. A simplistic model is less likely to contain the real reward function, while a model with high complexity leads to substantial computation cost and risks overfitting. This paper addresses this trade-off in IRL model selection by introducing the structural risk minimization (SRM) method from statistical learning. SRM selects an optimal reward function class from a hypothesis set minimizing both estimation error and model complexity. To formulate an SRM scheme for IRL, we estimate policy gradient by demonstration serving as empirical risk and establish the upper bound of Rademacher complexity of hypothesis classes as model penalty. The learning guarantee is further presented. In particular, we provide explicit SRM for the common linear weighted sum setting in IRL. Simulations demonstrate the performance and efficiency of our scheme.

{{</citation>}}


### (9/59) How Robust are LLMs to In-Context Majority Label Bias? (Karan Gupta et al., 2023)

{{<citation>}}

Karan Gupta, Sumegh Roychowdhury, Siva Rajesh Kasa, Santhosh Kumar Kasa, Anish Bhanushali, Nikhil Pattisapu, Prasanna Srinivasa Murthy. (2023)  
**How Robust are LLMs to In-Context Majority Label Bias?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2312.16549v1)  

---


**ABSTRACT**  
In the In-Context Learning (ICL) setup, various forms of label biases can manifest. One such manifestation is majority label bias, which arises when the distribution of labeled examples in the in-context samples is skewed towards one or more specific classes making Large Language Models (LLMs) more prone to predict those labels. Such discrepancies can arise from various factors, including logistical constraints, inherent biases in data collection methods, limited access to diverse data sources, etc. which are unavoidable in a real-world industry setup. In this work, we study the robustness of in-context learning in LLMs to shifts that occur due to majority label bias within the purview of text classification tasks. Prior works have shown that in-context learning with LLMs is susceptible to such biases. In our study, we go one level deeper and show that the robustness boundary varies widely for different models and tasks, with certain LLMs being highly robust (~90%) to majority label bias. Additionally, our findings also highlight the impact of model size and the richness of instructional prompts contributing towards model robustness. We restrict our study to only publicly available open-source models to ensure transparency and reproducibility.

{{</citation>}}


### (10/59) FALCON: Feature-Label Constrained Graph Net Collapse for Memory Efficient GNNs (Christopher Adnel et al., 2023)

{{<citation>}}

Christopher Adnel, Islem Rekik. (2023)  
**FALCON: Feature-Label Constrained Graph Net Collapse for Memory Efficient GNNs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2312.16542v1)  

---


**ABSTRACT**  
Graph Neural Network (GNN) ushered in a new era of machine learning with interconnected datasets. While traditional neural networks can only be trained on independent samples, GNN allows for the inclusion of inter-sample interactions in the training process. This gain, however, incurs additional memory cost, rendering most GNNs unscalable for real-world applications involving vast and complicated networks with tens of millions of nodes (e.g., social circles, web graphs, and brain graphs). This means that storing the graph in the main memory can be difficult, let alone training the GNN model with significantly less GPU memory. While much of the recent literature has focused on either mini-batching GNN methods or quantization, graph reduction methods remain largely scarce. Furthermore, present graph reduction approaches have several drawbacks. First, most graph reduction focuses only on the inference stage (e.g., condensation and distillation) and requires full graph GNN training, which does not reduce training memory footprint. Second, many methods focus solely on the graph's structural aspect, ignoring the initial population feature-label distribution, resulting in a skewed post-reduction label distribution. Here, we propose a Feature-Label COnstrained graph Net collapse, FALCON, to address these limitations. Our three core contributions lie in (i) designing FALCON, a topology-aware graph reduction technique that preserves feature-label distribution; (ii) implementation of FALCON with other memory reduction methods (i.e., mini-batched GNN and quantization) for further memory reduction; (iii) extensive benchmarking and ablation studies against SOTA methods to evaluate FALCON memory reduction. Our extensive results show that FALCON can significantly collapse various public datasets while achieving equal prediction quality across GNN models. Code: https://github.com/basiralab/FALCON

{{</citation>}}


### (11/59) Preference as Reward, Maximum Preference Optimization with Importance Sampling (Zaifan Jiang et al., 2023)

{{<citation>}}

Zaifan Jiang, Xing Huang, Chao Wei. (2023)  
**Preference as Reward, Maximum Preference Optimization with Importance Sampling**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.16430v2)  

---


**ABSTRACT**  
Preference learning is a key technology for aligning language models with human values. Reinforcement Learning from Human Feedback (RLHF) is a model based algorithm to optimize preference learning, which first fitting a reward model for preference score, and then optimizing generating policy with on-policy PPO algorithm to maximize the reward. The processing of RLHF is complex, time-consuming and unstable. Direct Preference Optimization (DPO) algorithm using off-policy algorithm to direct optimize generating policy and eliminating the need for reward model, which is data efficient and stable. DPO use Bradley-Terry model and log-loss which leads to over-fitting to the preference data at the expense of ignoring KL-regularization term when preference is deterministic. IPO uses a root-finding MSE loss to solve the ignoring KL-regularization problem. In this paper, we'll figure out, although IPO fix the problem when preference is deterministic, but both DPO and IPO fails the KL-regularization term because the support of preference distribution not equal to reference distribution. Then, we design a simple and intuitive off-policy preference optimization algorithm from an importance sampling view, which we call Maximum Preference Optimization (MPO), and add off-policy KL-regularization terms which makes KL-regularization truly effective. The objective of MPO bears resemblance to RLHF's objective, and likes IPO, MPO is off-policy. So, MPO attains the best of both worlds. To simplify the learning process and save memory usage, MPO eliminates the needs for both reward model and reference policy.

{{</citation>}}


### (12/59) Learning to Embed Time Series Patches Independently (Seunghan Lee et al., 2023)

{{<citation>}}

Seunghan Lee, Taeyoung Park, Kibok Lee. (2023)  
**Learning to Embed Time Series Patches Independently**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Time Series, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.16427v1)  

---


**ABSTRACT**  
Masked time series modeling has recently gained much attention as a self-supervised representation learning strategy for time series. Inspired by masked image modeling in computer vision, recent works first patchify and partially mask out time series, and then train Transformers to capture the dependencies between patches by predicting masked patches from unmasked patches. However, we argue that capturing such patch dependencies might not be an optimal strategy for time series representation learning; rather, learning to embed patches independently results in better time series representations. Specifically, we propose to use 1) the simple patch reconstruction task, which autoencode each patch without looking at other patches, and 2) the simple patch-wise MLP that embeds each patch independently. In addition, we introduce complementary contrastive learning to hierarchically capture adjacent time series information efficiently. Our proposed method improves time series forecasting and classification performance compared to state-of-the-art Transformer-based models, while it is more efficient in terms of the number of parameters and training/inference time. Code is available at this repository: https://github.com/seunghan96/pits.

{{</citation>}}


### (13/59) Soft Contrastive Learning for Time Series (Seunghan Lee et al., 2023)

{{<citation>}}

Seunghan Lee, Taeyoung Park, Kibok Lee. (2023)  
**Soft Contrastive Learning for Time Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Contrastive Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2312.16424v1)  

---


**ABSTRACT**  
Contrastive learning has shown to be effective to learn representations from time series in a self-supervised way. However, contrasting similar time series instances or values from adjacent timestamps within a time series leads to ignore their inherent correlations, which results in deteriorating the quality of learned representations. To address this issue, we propose SoftCLT, a simple yet effective soft contrastive learning strategy for time series. This is achieved by introducing instance-wise and temporal contrastive loss with soft assignments ranging from zero to one. Specifically, we define soft assignments for 1) instance-wise contrastive loss by the distance between time series on the data space, and 2) temporal contrastive loss by the difference of timestamps. SoftCLT is a plug-and-play method for time series contrastive learning that improves the quality of learned representations without bells and whistles. In experiments, we demonstrate that SoftCLT consistently improves the performance in various downstream tasks including classification, semi-supervised learning, transfer learning, and anomaly detection, showing state-of-the-art performance. Code is available at this repository: https://github.com/seunghan96/softclt.

{{</citation>}}


### (14/59) Dynamic Sub-graph Distillation for Robust Semi-supervised Continual Learning (Yan Fan et al., 2023)

{{<citation>}}

Yan Fan, Yu Wang, Pengfei Zhu, Qinghua Hu. (2023)  
**Dynamic Sub-graph Distillation for Robust Semi-supervised Continual Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.16409v1)  

---


**ABSTRACT**  
Continual learning (CL) has shown promising results and comparable performance to learning at once in a fully supervised manner. However, CL strategies typically require a large number of labeled samples, making their real-life deployment challenging. In this work, we focus on semi-supervised continual learning (SSCL), where the model progressively learns from partially labeled data with unknown categories. We provide a comprehensive analysis of SSCL and demonstrate that unreliable distributions of unlabeled data lead to unstable training and refinement of the progressing stages. This problem severely impacts the performance of SSCL. To address the limitations, we propose a novel approach called Dynamic Sub-Graph Distillation (DSGD) for semi-supervised continual learning, which leverages both semantic and structural information to achieve more stable knowledge distillation on unlabeled data and exhibit robustness against distribution bias. Firstly, we formalize a general model of structural distillation and design a dynamic graph construction for the continual learning progress. Next, we define a structure distillation vector and design a dynamic sub-graph distillation algorithm, which enables end-to-end training and adaptability to scale up tasks. The entire proposed method is adaptable to various CL methods and supervision settings. Finally, experiments conducted on three datasets CIFAR10, CIFAR100, and ImageNet-100, with varying supervision ratios, demonstrate the effectiveness of our proposed approach in mitigating the catastrophic forgetting problem in semi-supervised continual learning scenarios.

{{</citation>}}


### (15/59) Learning Time-aware Graph Structures for Spatially Correlated Time Series Forecasting (Minbo Ma et al., 2023)

{{<citation>}}

Minbo Ma, Jilin Hu, Christian S. Jensen, Fei Teng, Peng Han, Zhiqiang Xu, Tianrui Li. (2023)  
**Learning Time-aware Graph Structures for Spatially Correlated Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.16403v1)  

---


**ABSTRACT**  
Spatio-temporal forecasting of future values of spatially correlated time series is important across many cyber-physical systems (CPS). Recent studies offer evidence that the use of graph neural networks to capture latent correlations between time series holds a potential for enhanced forecasting. However, most existing methods rely on pre-defined or self-learning graphs, which are either static or unintentionally dynamic, and thus cannot model the time-varying correlations that exhibit trends and periodicities caused by the regularity of the underlying processes in CPS. To tackle such limitation, we propose Time-aware Graph Structure Learning (TagSL), which extracts time-aware correlations among time series by measuring the interaction of node and time representations in high-dimensional spaces. Notably, we introduce time discrepancy learning that utilizes contrastive learning with distance-based regularization terms to constrain learned spatial correlations to a trend sequence. Additionally, we propose a periodic discriminant function to enable the capture of periodic changes from the state of nodes. Next, we present a Graph Convolution-based Gated Recurrent Unit (GCGRU) that jointly captures spatial and temporal dependencies while learning time-aware and node-specific patterns. Finally, we introduce a unified framework named Time-aware Graph Convolutional Recurrent Network (TGCRN), combining TagSL, and GCGRU in an encoder-decoder architecture for multi-step spatio-temporal forecasting. We report on experiments with TGCRN and popular existing approaches on five real-world datasets, thus providing evidence that TGCRN is capable of advancing the state-of-the-art. We also cover a detailed ablation study and visualization analysis, offering detailed insight into the effectiveness of time-aware structure learning.

{{</citation>}}


### (16/59) Photovoltaic power forecasting using quantum machine learning (Asel Sagingalieva et al., 2023)

{{<citation>}}

Asel Sagingalieva, Stefan Komornyik, Arsenii Senokosov, Ayush Joshi, Alexander Sedykh, Christopher Mansell, Olga Tsurkan, Karan Pinto, Markus Pflitsch, Alexey Melnikov. (2023)  
**Photovoltaic power forecasting using quantum machine learning**  

---
Primary Category: cs.LG  
Categories: cs-ET, cs-LG, cs.LG, quant-ph  
Keywords: Sequence-to-Sequence  
[Paper Link](http://arxiv.org/abs/2312.16379v1)  

---


**ABSTRACT**  
Predicting solar panel power output is crucial for advancing the energy transition but is complicated by the variable and non-linear nature of solar energy. This is influenced by numerous meteorological factors, geographical positioning, and photovoltaic cell properties, posing significant challenges to forecasting accuracy and grid stability. Our study introduces a suite of solutions centered around hybrid quantum neural networks designed to tackle these complexities. The first proposed model, the Hybrid Quantum Long Short-Term Memory, surpasses all tested models by over 40% lower mean absolute and mean squared errors. The second proposed model, Hybrid Quantum Sequence-to-Sequence neural network, once trained, predicts photovoltaic power with 16% lower mean absolute error for arbitrary time intervals without the need for prior meteorological data, highlighting its versatility. Moreover, our hybrid models perform better even when trained on limited datasets, underlining their potential utility in data-scarce scenarios. These findings represent a stride towards resolving time series prediction challenges in energy power forecasting through hybrid quantum models, showcasing the transformative potential of quantum machine learning in catalyzing the renewable energy transition.

{{</citation>}}


## cs.CL (12)



### (17/59) Stateful FastConformer with Cache-based Inference for Streaming Automatic Speech Recognition (Vahid Noroozi et al., 2023)

{{<citation>}}

Vahid Noroozi, Somshubra Majumdar, Ankur Kumar, Jagadeesh Balam, Boris Ginsburg. (2023)  
**Stateful FastConformer with Cache-based Inference for Streaming Automatic Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.17279v1)  

---


**ABSTRACT**  
In this paper, we propose an efficient and accurate streaming speech recognition model based on the FastConformer architecture. We adapted the FastConformer architecture for streaming applications through: (1) constraining both the look-ahead and past contexts in the encoder, and (2) introducing an activation caching mechanism to enable the non-autoregressive encoder to operate autoregressively during inference. The proposed model is thoughtfully designed in a way to eliminate the accuracy disparity between the train and inference time which is common for many streaming models. Furthermore, our proposed encoder works with various decoder configurations including Connectionist Temporal Classification (CTC) and RNN-Transducer (RNNT) decoders. Additionally, we introduced a hybrid CTC/RNNT architecture which utilizes a shared encoder with both a CTC and RNNT decoder to boost the accuracy and save computation. We evaluate the proposed model on LibriSpeech dataset and a multi-domain large scale dataset and demonstrate that it can achieve better accuracy with lower latency and inference time compared to a conventional buffered streaming model baseline. We also showed that training a model with multiple latencies can achieve better accuracy than single latency models while it enables us to support multiple latencies with a single model. Our experiments also showed the hybrid architecture would not only speedup the convergence of the CTC decoder but also improves the accuracy of streaming models compared to single decoder models.

{{</citation>}}


### (18/59) Rethinking Tabular Data Understanding with Large Language Models (Tianyang Liu et al., 2023)

{{<citation>}}

Tianyang Liu, Fei Wang, Muhao Chen. (2023)  
**Rethinking Tabular Data Understanding with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-DB, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.16702v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown to be capable of various tasks, yet their capability in interpreting and reasoning over tabular data remains an underexplored area. In this context, this study investigates from three core perspectives: the robustness of LLMs to structural perturbations in tables, the comparative analysis of textual and symbolic reasoning on tables, and the potential of boosting model performance through the aggregation of multiple reasoning pathways. We discover that structural variance of tables presenting the same content reveals a notable performance decline, particularly in symbolic reasoning tasks. This prompts the proposal of a method for table structure normalization. Moreover, textual reasoning slightly edges out symbolic reasoning, and a detailed error analysis reveals that each exhibits different strengths depending on the specific tasks. Notably, the aggregation of textual and symbolic reasoning pathways, bolstered by a mix self-consistency mechanism, resulted in achieving SOTA performance, with an accuracy of 73.6% on WIKITABLEQUESTIONS, representing a substantial advancement over previous existing table processing paradigms of LLMs.

{{</citation>}}


### (19/59) Large Language Models for Conducting Advanced Text Analytics Information Systems Research (Benjamin M. Ampel et al., 2023)

{{<citation>}}

Benjamin M. Ampel, Chi-Heng Yang, James Hu, Hsinchun Chen. (2023)  
**Large Language Models for Conducting Advanced Text Analytics Information Systems Research**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.17278v1)  

---


**ABSTRACT**  
The exponential growth of digital content has generated massive textual datasets, necessitating advanced analytical approaches. Large Language Models (LLMs) have emerged as tools capable of processing and extracting insights from massive unstructured textual datasets. However, how to leverage LLMs for text-based Information Systems (IS) research is currently unclear. To assist IS research in understanding how to operationalize LLMs, we propose a Text Analytics for Information Systems Research (TAISR) framework. Our proposed framework provides detailed recommendations grounded in IS and LLM literature on how to conduct meaningful text-based IS research. We conducted three case studies in business intelligence using our TAISR framework to demonstrate its application across several IS research contexts. We also outline potential challenges and limitations in adopting LLMs for IS. By offering a systematic approach and evidence of its utility, our TAISR framework contributes to future IS research streams looking to incorporate powerful LLMs for text analytics.

{{</citation>}}


### (20/59) A Large Language Model-based Computational Approach to Improve Identity-Related Write-Ups (Alex Doboli, 2023)

{{<citation>}}

Alex Doboli. (2023)  
**A Large Language Model-based Computational Approach to Improve Identity-Related Write-Ups**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.16659v1)  

---


**ABSTRACT**  
Creating written products is essential to modern life, including writings about one's identity and personal experiences. However, writing is often a difficult activity that requires extensive effort to frame the central ideas, the pursued approach to communicate the central ideas, e.g., using analogies, metaphors, or other possible means, the needed presentation structure, and the actual verbal expression. Large Language Models, a recently emerged approach in Machine Learning, can offer a significant help in reducing the effort and improving the quality of written products. This paper proposes a new computational approach to explore prompts that given as inputs to a Large Language Models can generate cues to improve the considered written products. Two case studies on improving write-ups, one based on an analogy and one on a metaphor, are also presented in the paper.

{{</citation>}}


### (21/59) Make BERT-based Chinese Spelling Check Model Enhanced by Layerwise Attention and Gaussian Mixture Model (Yongchang Cao et al., 2023)

{{<citation>}}

Yongchang Cao, Liang He, Zhen Wu, Xinyu Dai. (2023)  
**Make BERT-based Chinese Spelling Check Model Enhanced by Layerwise Attention and Gaussian Mixture Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, BERT  
[Paper Link](http://arxiv.org/abs/2312.16623v1)  

---


**ABSTRACT**  
BERT-based models have shown a remarkable ability in the Chinese Spelling Check (CSC) task recently. However, traditional BERT-based methods still suffer from two limitations. First, although previous works have identified that explicit prior knowledge like Part-Of-Speech (POS) tagging can benefit in the CSC task, they neglected the fact that spelling errors inherent in CSC data can lead to incorrect tags and therefore mislead models. Additionally, they ignored the correlation between the implicit hierarchical information encoded by BERT's intermediate layers and different linguistic phenomena. This results in sub-optimal accuracy. To alleviate the above two issues, we design a heterogeneous knowledge-infused framework to strengthen BERT-based CSC models. To incorporate explicit POS knowledge, we utilize an auxiliary task strategy driven by Gaussian mixture model. Meanwhile, to incorporate implicit hierarchical linguistic knowledge within the encoder, we propose a novel form of n-gram-based layerwise self-attention to generate a multilayer representation. Experimental results show that our proposed framework yields a stable performance boost over four strong baseline models and outperforms the previous state-of-the-art methods on two datasets.

{{</citation>}}


### (22/59) Relationship between auditory and semantic entrainment using Deep Neural Networks (DNN) (Jay Kejriwal et al., 2023)

{{<citation>}}

Jay Kejriwal, Štefan Beňuš. (2023)  
**Relationship between auditory and semantic entrainment using Deep Neural Networks (DNN)**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.16599v1)  

---


**ABSTRACT**  
The tendency of people to engage in similar, matching, or synchronized behaviour when interacting is known as entrainment. Many studies examined linguistic (syntactic and lexical structures) and paralinguistic (pitch, intensity) entrainment, but less attention was given to finding the relationship between them. In this study, we utilized state-of-the-art DNN embeddings such as BERT and TRIpLet Loss network (TRILL) vectors to extract features for measuring semantic and auditory similarities of turns within dialogues in two comparable spoken corpora of two different languages. We found people's tendency to entrain on semantic features more when compared to auditory features. Additionally, we found that entrainment in semantic and auditory linguistic features are positively correlated. The findings of this study might assist in implementing the mechanism of entrainment in human-machine interaction (HMI).

{{</citation>}}


### (23/59) A proposed new metric for the conceptual diversity of a text (İlknur Dönmez Phd et al., 2023)

{{<citation>}}

İlknur Dönmez Phd, Mehmet Haklıdır Phd. (2023)  
**A proposed new metric for the conceptual diversity of a text**  

---
Primary Category: cs.CL  
Categories: I-2-4; I-2-7, cs-AI, cs-CL, cs-IT, cs.CL, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.16548v1)  

---


**ABSTRACT**  
A word may contain one or more hidden concepts. While the "animal" word evokes many images in our minds and encapsulates many concepts (birds, dogs, cats, crocodiles, etc.), the `parrot' word evokes a single image (a colored bird with a short, hooked beak and the ability to mimic sounds). In spoken or written texts, we use some words in a general sense and some in a detailed way to point to a specific object. Until now, a text's conceptual diversity value cannot be determined using a standard and precise technique. This research contributes to the natural language processing field of AI by offering a standardized method and a generic metric for evaluating and comparing concept diversity in different texts and domains. It also contributes to the field of semantic research of languages. If we give examples for the diversity score of two sentences, "He discovered an unknown entity." has a high conceptual diversity score (16.6801), and "The endoplasmic reticulum forms a series of flattened sacs within the cytoplasm of eukaryotic cells." sentence has a low conceptual diversity score which is 3.9068.

{{</citation>}}


### (24/59) PanGu-$π$: Enhancing Language Model Architectures via Nonlinearity Compensation (Yunhe Wang et al., 2023)

{{<citation>}}

Yunhe Wang, Hanting Chen, Yehui Tang, Tianyu Guo, Kai Han, Ying Nie, Xutao Wang, Hailin Hu, Zheyuan Bai, Yun Wang, Fangcheng Liu, Zhicheng Liu, Jianyuan Guo, Sinan Zeng, Yinchen Zhang, Qinghua Xu, Qun Liu, Jun Yao, Chao Xu, Dacheng Tao. (2023)  
**PanGu-$π$: Enhancing Language Model Architectures via Nonlinearity Compensation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.17276v1)  

---


**ABSTRACT**  
The recent trend of large language models (LLMs) is to increase the scale of both model size (\aka the number of parameters) and dataset to achieve better generative ability, which is definitely proved by a lot of work such as the famous GPT and Llama. However, large models often involve massive computational costs, and practical applications cannot afford such high prices. However, the method of constructing a strong model architecture for LLMs is rarely discussed. We first analyze the state-of-the-art language model architectures and observe the feature collapse problem. Based on the theoretical analysis, we propose that the nonlinearity is also very important for language models, which is usually studied in convolutional neural networks for vision tasks. The series informed activation function is then introduced with tiny calculations that can be ignored, and an augmented shortcut is further used to enhance the model nonlinearity. We then demonstrate that the proposed approach is significantly effective for enhancing the model nonlinearity through carefully designed ablations; thus, we present a new efficient model architecture for establishing modern, namely, PanGu-$\pi$. Experiments are then conducted using the same dataset and training strategy to compare PanGu-$\pi$ with state-of-the-art LLMs. The results show that PanGu-$\pi$-7B can achieve a comparable performance to that of benchmarks with about 10\% inference speed-up, and PanGu-$\pi$-1B can achieve state-of-the-art performance in terms of accuracy and efficiency. In addition, we have deployed PanGu-$\pi$-7B in the high-value domains of finance and law, developing an LLM named YunShan for practical application. The results show that YunShan can surpass other models with similar scales on benchmarks.

{{</citation>}}


### (25/59) S2M: Converting Single-Turn to Multi-Turn Datasets for Conversational Question Answering (Baokui Li et al., 2023)

{{<citation>}}

Baokui Li, Sen Zhang, Wangshu Zhang, Yicheng Chen, Changlin Yang, Sen Hu, Teng Xu, Siye liu, Jiwei Li. (2023)  
**S2M: Converting Single-Turn to Multi-Turn Datasets for Conversational Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.16511v1)  

---


**ABSTRACT**  
Supplying data augmentation to conversational question answering (CQA) can effectively improve model performance. However, there is less improvement from single-turn datasets in CQA due to the distribution gap between single-turn and multi-turn datasets. On the other hand, while numerous single-turn datasets are available, we have not utilized them effectively. To solve this problem, we propose a novel method to convert single-turn datasets to multi-turn datasets. The proposed method consists of three parts, namely, a QA pair Generator, a QA pair Reassembler, and a question Rewriter. Given a sample consisting of context and single-turn QA pairs, the Generator obtains candidate QA pairs and a knowledge graph based on the context. The Reassembler utilizes the knowledge graph to get sequential QA pairs, and the Rewriter rewrites questions from a conversational perspective to obtain a multi-turn dataset S2M. Our experiments show that our method can synthesize effective training resources for CQA. Notably, S2M ranks 1st place on the QuAC leaderboard at the time of submission (Aug 24th, 2022).

{{</citation>}}


### (26/59) Source Code is a Graph, Not a Sequence: A Cross-Lingual Perspective on Code Clone Detection (Mohammed Ataaur Rahaman et al., 2023)

{{<citation>}}

Mohammed Ataaur Rahaman, Julia Ive. (2023)  
**Source Code is a Graph, Not a Sequence: A Cross-Lingual Perspective on Code Clone Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.16488v1)  

---


**ABSTRACT**  
Source code clone detection is the task of finding code fragments that have the same or similar functionality, but may differ in syntax or structure. This task is important for software maintenance, reuse, and quality assurance (Roy et al. 2009). However, code clone detection is challenging, as source code can be written in different languages, domains, and styles. In this paper, we argue that source code is inherently a graph, not a sequence, and that graph-based methods are more suitable for code clone detection than sequence-based methods. We compare the performance of two state-of-the-art models: CodeBERT (Feng et al. 2020), a sequence-based model, and CodeGraph (Yu et al. 2023), a graph-based model, on two benchmark data-sets: BCB (Svajlenko et al. 2014) and PoolC (PoolC no date). We show that CodeGraph outperforms CodeBERT on both data-sets, especially on cross-lingual code clones. To the best of our knowledge, this is the first work to demonstrate the superiority of graph-based methods over sequence-based methods on cross-lingual code clone detection.

{{</citation>}}


### (27/59) LLM Factoscope: Uncovering LLMs' Factual Discernment through Inner States Analysis (Jinwen He et al., 2023)

{{<citation>}}

Jinwen He, Yujia Gong, Kai Chen, Zijin Lin, Chengan Wei, Yue Zhao. (2023)  
**LLM Factoscope: Uncovering LLMs' Factual Discernment through Inner States Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.16374v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have revolutionized various domains with extensive knowledge and creative capabilities. However, a critical issue with LLMs is their tendency to produce outputs that diverge from factual reality. This phenomenon is particularly concerning in sensitive applications such as medical consultation and legal advice, where accuracy is paramount. In this paper, we introduce the LLM factoscope, a novel Siamese network-based model that leverages the inner states of LLMs for factual detection. Our investigation reveals distinguishable patterns in LLMs' inner states when generating factual versus non-factual content. We demonstrate the LLM factoscope's effectiveness across various architectures, achieving over 96% accuracy in factual detection. Our work opens a new avenue for utilizing LLMs' inner states for factual detection and encourages further exploration into LLMs' inner workings for enhanced reliability and transparency.

{{</citation>}}


### (28/59) Conversational Question Answering with Reformulations over Knowledge Graph (Lihui Liu et al., 2023)

{{<citation>}}

Lihui Liu, Blaine Hill, Boxin Du, Fei Wang, Hanghang Tong. (2023)  
**Conversational Question Answering with Reformulations over Knowledge Graph**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Graph, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.17269v1)  

---


**ABSTRACT**  
conversational question answering (convQA) over knowledge graphs (KGs) involves answering multi-turn natural language questions about information contained in a KG. State-of-the-art methods of ConvQA often struggle with inexplicit question-answer pairs. These inputs are easy for human beings to understand given a conversation history, but hard for a machine to interpret, which can degrade ConvQA performance. To address this problem, we propose a reinforcement learning (RL) based model, CornNet, which utilizes question reformulations generated by large language models (LLMs) to improve ConvQA performance. CornNet adopts a teacher-student architecture where a teacher model learns question representations using human writing reformulations, and a student model to mimic the teacher model's output via reformulations generated by LLMs. The learned question representation is then used by an RL model to locate the correct answer in a KG. Extensive experimental results show that CornNet outperforms state-of-the-art convQA models.

{{</citation>}}


## cs.CR (1)



### (29/59) Adversarial Attacks on LoRa Device Identification and Rogue Signal Detection with Deep Learning (Yalin E. Sagduyu et al., 2023)

{{<citation>}}

Yalin E. Sagduyu, Tugba Erpek. (2023)  
**Adversarial Attacks on LoRa Device Identification and Rogue Signal Detection with Deep Learning**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs-NI, cs.CR, eess-SP  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.16715v1)  

---


**ABSTRACT**  
Low-Power Wide-Area Network (LPWAN) technologies, such as LoRa, have gained significant attention for their ability to enable long-range, low-power communication for Internet of Things (IoT) applications. However, the security of LoRa networks remains a major concern, particularly in scenarios where device identification and classification of legitimate and spoofed signals are crucial. This paper studies a deep learning framework to address these challenges, considering LoRa device identification and legitimate vs. rogue LoRa device classification tasks. A deep neural network (DNN), either a convolutional neural network (CNN) or feedforward neural network (FNN), is trained for each task by utilizing real experimental I/Q data for LoRa signals, while rogue signals are generated by using kernel density estimation (KDE) of received signals by rogue devices. Fast Gradient Sign Method (FGSM)-based adversarial attacks are considered for LoRa signal classification tasks using deep learning models. The impact of these attacks is assessed on the performance of two tasks, namely device identification and legitimate vs. rogue device classification, by utilizing separate or common perturbations against these signal classification tasks. Results presented in this paper quantify the level of transferability of adversarial attacks on different LoRa signal classification tasks as a major vulnerability and highlight the need to make IoT applications robust to adversarial attacks.

{{</citation>}}


## cs.IR (2)



### (30/59) Performance Comparison of Session-based Recommendation Algorithms based on GNNs (Faisal Shehzad et al., 2023)

{{<citation>}}

Faisal Shehzad, Dietmar Jannach. (2023)  
**Performance Comparison of Session-based Recommendation Algorithms based on GNNs**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.16695v1)  

---


**ABSTRACT**  
In session-based recommendation settings, a recommender system has to base its suggestions on the user interactions that are ob served in an ongoing session. Since such sessions can consist of only a small set of interactions, various approaches based on Graph Neural Networks (GNN) were recently proposed, as they allow us to integrate various types of side information about the items in a natural way. Unfortunately, a variety of evaluation settings are used in the literature, e.g., in terms of protocols, metrics and baselines, making it difficult to assess what represents the state of the art. In this work, we present the results of an evaluation of eight recent GNN-based approaches that were published in high-quality outlets. For a fair comparison, all models are systematically tuned and tested under identical conditions using three common datasets. We furthermore include k-nearest-neighbor and sequential rules-based models as baselines, as such models have previously exhibited competitive performance results for similar settings. To our surprise, the evaluation showed that the simple models outperform all recent GNN models in terms of the Mean Reciprocal Rank, which we used as an optimization criterion, and were only outperformed in three cases in terms of the Hit Rate. Additional analyses furthermore reveal that several other factors that are often not deeply discussed in papers, e.g., random seeds, can markedly impact the performance of GNN-based models. Our results therefore (a) point to continuing issues in the community in terms of research methodology and (b) indicate that there is ample room for improvement in session-based recommendation.

{{</citation>}}


### (31/59) RDGCL: Reaction-Diffusion Graph Contrastive Learning for Recommendation (Jeongwhan Choi et al., 2023)

{{<citation>}}

Jeongwhan Choi, Hyowon Wi, Chaejeong Lee, Sung-Bae Cho, Dongha Lee, Noseong Park. (2023)  
**RDGCL: Reaction-Diffusion Graph Contrastive Learning for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.16563v1)  

---


**ABSTRACT**  
Contrastive learning (CL) has emerged as a promising technique for improving recommender systems, addressing the challenge of data sparsity by leveraging self-supervised signals from raw data. Integration of CL with graph convolutional network (GCN)-based collaborative filterings (CFs) has been explored in recommender systems. However, current CL-based recommendation models heavily rely on low-pass filters and graph augmentations. In this paper, we propose a novel CL method for recommender systems called the reaction-diffusion graph contrastive learning model (RDGCL). We design our own GCN for CF based on both the diffusion, i.e., low-pass filter, and the reaction, i.e., high-pass filter, equations. Our proposed CL-based training occurs between reaction and diffusion-based embeddings, so there is no need for graph augmentations. Experimental evaluation on 6 benchmark datasets demonstrates that our proposed method outperforms state-of-the-art CL-based recommendation models. By enhancing recommendation accuracy and diversity, our method brings an advancement in CL for recommender systems.

{{</citation>}}


## cs.CV (13)



### (32/59) I2V-Adapter: A General Image-to-Video Adapter for Video Diffusion Models (Xun Guo et al., 2023)

{{<citation>}}

Xun Guo, Mingwu Zheng, Liang Hou, Yuan Gao, Yufan Deng, Chongyang Ma, Weiming Hu, Zhengjun Zha, Haibin Huang, Pengfei Wan, Di Zhang. (2023)  
**I2V-Adapter: A General Image-to-Video Adapter for Video Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.16693v1)  

---


**ABSTRACT**  
In the rapidly evolving domain of digital content generation, the focus has shifted from text-to-image (T2I) models to more advanced video diffusion models, notably text-to-video (T2V) and image-to-video (I2V). This paper addresses the intricate challenge posed by I2V: converting static images into dynamic, lifelike video sequences while preserving the original image fidelity. Traditional methods typically involve integrating entire images into diffusion processes or using pretrained encoders for cross attention. However, these approaches often necessitate altering the fundamental weights of T2I models, thereby restricting their reusability. We introduce a novel solution, namely I2V-Adapter, designed to overcome such limitations. Our approach preserves the structural integrity of T2I models and their inherent motion modules. The I2V-Adapter operates by processing noised video frames in parallel with the input image, utilizing a lightweight adapter module. This module acts as a bridge, efficiently linking the input to the model's self-attention mechanism, thus maintaining spatial details without requiring structural changes to the T2I model. Moreover, I2V-Adapter requires only a fraction of the parameters of conventional models and ensures compatibility with existing community-driven T2I models and controlling tools. Our experimental results demonstrate I2V-Adapter's capability to produce high-quality video outputs. This performance, coupled with its versatility and reduced need for trainable parameters, represents a substantial advancement in the field of AI-driven video generation, particularly for creative applications.

{{</citation>}}


### (33/59) Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection (Huan Liu et al., 2023)

{{<citation>}}

Huan Liu, Zichang Tan, Chuangchuang Tan, Yunchao Wei, Yao Zhao, Jingdong Wang. (2023)  
**Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.16649v1)  

---


**ABSTRACT**  
In this paper, we study the problem of generalizable synthetic image detection, aiming to detect forgery images from diverse generative methods, e.g., GANs and diffusion models. Cutting-edge solutions start to explore the benefits of pre-trained models, and mainly follow the fixed paradigm of solely training an attached classifier, e.g., combining frozen CLIP-ViT with a learnable linear layer in UniFD. However, our analysis shows that such a fixed paradigm is prone to yield detectors with insufficient learning regarding forgery representations. We attribute the key challenge to the lack of forgery adaptation, and present a novel forgery-aware adaptive transformer approach, namely FatFormer. Based on the pre-trained vision-language spaces of CLIP, FatFormer introduces two core designs for the adaption to build generalized forgery representations. First, motivated by the fact that both image and frequency analysis are essential for synthetic image detection, we develop a forgery-aware adapter to adapt image features to discern and integrate local forgery traces within image and frequency domains. Second, we find that considering the contrastive objectives between adapted image features and text prompt embeddings, a previously overlooked aspect, results in a nontrivial generalization improvement. Accordingly, we introduce language-guided alignment to supervise the forgery adaptation with image and text prompts in FatFormer. Experiments show that, by coupling these two designs, our approach tuned on 4-class ProGAN data attains a remarkable detection performance, achieving an average of 98% accuracy to unseen GANs, and surprisingly generalizes to unseen diffusion models with 95% accuracy.

{{</citation>}}


### (34/59) VLCounter: Text-aware Visual Representation for Zero-Shot Object Counting (Seunggu Kang et al., 2023)

{{<citation>}}

Seunggu Kang, WonJun Moon, Euiyeon Kim, Jae-Pil Heo. (2023)  
**VLCounter: Text-aware Visual Representation for Zero-Shot Object Counting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.16580v2)  

---


**ABSTRACT**  
Zero-Shot Object Counting (ZSOC) aims to count referred instances of arbitrary classes in a query image without human-annotated exemplars. To deal with ZSOC, preceding studies proposed a two-stage pipeline: discovering exemplars and counting. However, there remains a challenge of vulnerability to error propagation of the sequentially designed two-stage process. In this work, an one-stage baseline, Visual-Language Baseline (VLBase), exploring the implicit association of the semantic-patch embeddings of CLIP is proposed. Subsequently, the extension of VLBase to Visual-language Counter (VLCounter) is achieved by incorporating three modules devised to tailor VLBase for object counting. First, Semantic-conditioned Prompt Tuning (SPT) is introduced within the image encoder to acquire target-highlighted representations. Second, Learnable Affine Transformation (LAT) is employed to translate the semantic-patch similarity map to be appropriate for the counting task. Lastly, the layer-wisely encoded features are transferred to the decoder through Segment-aware Skip Connection (SaSC) to keep the generalization capability for unseen classes. Through extensive experiments on FSC147, CARPK, and PUCPR+, the benefits of the end-to-end framework, VLCounter, are demonstrated.

{{</citation>}}


### (35/59) Multi-modality Affinity Inference for Weakly Supervised 3D Semantic Segmentation (Xiawei Li et al., 2023)

{{<citation>}}

Xiawei Li, Qingyuan Xu, Jing Zhang, Tianyi Zhang, Qian Yu, Lu Sheng, Dong Xu. (2023)  
**Multi-modality Affinity Inference for Weakly Supervised 3D Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.16578v2)  

---


**ABSTRACT**  
3D point cloud semantic segmentation has a wide range of applications. Recently, weakly supervised point cloud segmentation methods have been proposed, aiming to alleviate the expensive and laborious manual annotation process by leveraging scene-level labels. However, these methods have not effectively exploited the rich geometric information (such as shape and scale) and appearance information (such as color and texture) present in RGB-D scans. Furthermore, current approaches fail to fully leverage the point affinity that can be inferred from the feature extraction network, which is crucial for learning from weak scene-level labels. Additionally, previous work overlooks the detrimental effects of the long-tailed distribution of point cloud data in weakly supervised 3D semantic segmentation. To this end, this paper proposes a simple yet effective scene-level weakly supervised point cloud segmentation method with a newly introduced multi-modality point affinity inference module. The point affinity proposed in this paper is characterized by features from multiple modalities (e.g., point cloud and RGB), and is further refined by normalizing the classifier weights to alleviate the detrimental effects of long-tailed distribution without the need of the prior of category distribution. Extensive experiments on the ScanNet and S3DIS benchmarks verify the effectiveness of our proposed method, which outperforms the state-of-the-art by ~4% to ~6% mIoU. Codes are released at https://github.com/Sunny599/AAAI24-3DWSSG-MMA.

{{</citation>}}


### (36/59) GRSDet: Learning to Generate Local Reverse Samples for Few-shot Object Detection (Hefei Mei et al., 2023)

{{<citation>}}

Hefei Mei, Taijin Zhao, Shiyuan Tang, Heqian Qiu, Lanxiao Wang, Minjian Zhang, Fanman Meng, Hongliang Li. (2023)  
**GRSDet: Learning to Generate Local Reverse Samples for Few-shot Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Object Detection  
[Paper Link](http://arxiv.org/abs/2312.16571v2)  

---


**ABSTRACT**  
Few-shot object detection (FSOD) aims to achieve object detection only using a few novel class training data. Most of the existing methods usually adopt a transfer-learning strategy to construct the novel class distribution by transferring the base class knowledge. However, this direct way easily results in confusion between the novel class and other similar categories in the decision space. To address the problem, we propose generating local reverse samples (LRSamples) in Prototype Reference Frames to adaptively adjust the center position and boundary range of the novel class distribution to learn more discriminative novel class samples for FSOD. Firstly, we propose a Center Calibration Variance Augmentation (CCVA) module, which contains the selection rule of LRSamples, the generator of LRSamples, and augmentation on the calibrated distribution centers. Specifically, we design an intra-class feature converter (IFC) as the generator of CCVA to learn the selecting rule. By transferring the knowledge of IFC from the base training to fine-tuning, the IFC generates plentiful novel samples to calibrate the novel class distribution. Moreover, we propose a Feature Density Boundary Optimization (FDBO) module to adaptively adjust the importance of samples depending on their distance from the decision boundary. It can emphasize the importance of the high-density area of the similar class (closer decision boundary area) and reduce the weight of the low-density area of the similar class (farther decision boundary area), thus optimizing a clearer decision boundary for each category. We conduct extensive experiments to demonstrate the effectiveness of our proposed method. Our method achieves consistent improvement on the Pascal VOC and MS COCO datasets based on DeFRCN and MFDC baselines.

{{</citation>}}


### (37/59) Blind Image Quality Assessment: A Brief Survey (Miaohui Wang, 2023)

{{<citation>}}

Miaohui Wang. (2023)  
**Blind Image Quality Assessment: A Brief Survey**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.16551v1)  

---


**ABSTRACT**  
Blind Image Quality Assessment (BIQA) is essential for automatically evaluating the perceptual quality of visual signals without access to the references. In this survey, we provide a comprehensive analysis and discussion of recent developments in the field of BIQA. We have covered various aspects, including hand-crafted BIQAs that focus on distortion-specific and general-purpose methods, as well as deep-learned BIQAs that employ supervised and unsupervised learning techniques. Additionally, we have explored multimodal quality assessment methods that consider interactions between visual and audio modalities, as well as visual and text modalities. Finally, we have offered insights into representative BIQA databases, including both synthetic and authentic distortions. We believe this survey provides valuable understandings into the latest developments and emerging trends for the visual quality community.

{{</citation>}}


### (38/59) ConstScene: Dataset and Model for Advancing Robust Semantic Segmentation in Construction Environments (Maghsood Salimi et al., 2023)

{{<citation>}}

Maghsood Salimi, Mohammad Loni, Sara Afshar, Marjan Sirjani, Antonio Cicchetti. (2023)  
**ConstScene: Dataset and Model for Advancing Robust Semantic Segmentation in Construction Environments**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.16516v1)  

---


**ABSTRACT**  
The increasing demand for autonomous machines in construction environments necessitates the development of robust object detection algorithms that can perform effectively across various weather and environmental conditions. This paper introduces a new semantic segmentation dataset specifically tailored for construction sites, taking into account the diverse challenges posed by adverse weather and environmental conditions. The dataset is designed to enhance the training and evaluation of object detection models, fostering their adaptability and reliability in real-world construction applications. Our dataset comprises annotated images captured under a wide range of different weather conditions, including but not limited to sunny days, rainy periods, foggy atmospheres, and low-light situations. Additionally, environmental factors such as the existence of dirt/mud on the camera lens are integrated into the dataset through actual captures and synthetic generation to simulate the complex conditions prevalent in construction sites. We also generate synthetic images of the annotations including precise semantic segmentation masks for various objects commonly found in construction environments, such as wheel loader machines, personnel, cars, and structural elements. To demonstrate the dataset's utility, we evaluate state-of-the-art object detection algorithms on our proposed benchmark. The results highlight the dataset's success in adversarial training models across diverse conditions, showcasing its efficacy compared to existing datasets that lack such environmental variability.

{{</citation>}}


### (39/59) A Non-Uniform Low-Light Image Enhancement Method with Multi-Scale Attention Transformer and Luminance Consistency Loss (Xiao Fang et al., 2023)

{{<citation>}}

Xiao Fang, Xin Gao, Baofeng Li, Feng Zhai, Yu Qin, Zhihang Meng, Jiansheng Lu, Chun Xiao. (2023)  
**A Non-Uniform Low-Light Image Enhancement Method with Multi-Scale Attention Transformer and Luminance Consistency Loss**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2312.16498v1)  

---


**ABSTRACT**  
Low-light image enhancement aims to improve the perception of images collected in dim environments and provide high-quality data support for image recognition tasks. When dealing with photos captured under non-uniform illumination, existing methods cannot adaptively extract the differentiated luminance information, which will easily cause over-exposure and under-exposure. From the perspective of unsupervised learning, we propose a multi-scale attention Transformer named MSATr, which sufficiently extracts local and global features for light balance to improve the visual quality. Specifically, we present a multi-scale window division scheme, which uses exponential sequences to adjust the window size of each layer. Within different-sized windows, the self-attention computation can be refined, ensuring the pixel-level feature processing capability of the model. For feature interaction across windows, a global transformer branch is constructed to provide comprehensive brightness perception and alleviate exposure problems. Furthermore, we propose a loop training strategy, using the diverse images generated by weighted mixing and a luminance consistency loss to improve the model's generalization ability effectively. Extensive experiments on several benchmark datasets quantitatively and qualitatively prove that our MSATr is superior to state-of-the-art low-light image enhancement methods, and the enhanced images have more natural brightness and outstanding details. The code is released at https://github.com/fang001021/MSATr.

{{</citation>}}


### (40/59) Group Multi-View Transformer for 3D Shape Analysis with Spatial Encoding (Lixiang Xu et al., 2023)

{{<citation>}}

Lixiang Xu, Qingzhe Cui, Richang Hong, Wei Xu, Enhong Chen, Xin Yuan, Chenglong Li, Yuanyan Tang. (2023)  
**Group Multi-View Transformer for 3D Shape Analysis with Spatial Encoding**  

---
Primary Category: cs.CV  
Categories: 68, I-2-10, cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.16477v2)  

---


**ABSTRACT**  
In recent years, the results of view-based 3D shape recognition methods have saturated, and models with excellent performance cannot be deployed on memory-limited devices due to their huge size of parameters. To address this problem, we introduce a compression method based on knowledge distillation for this field, which largely reduces the number of parameters while preserving model performance as much as possible. Specifically, to enhance the capabilities of smaller models, we design a high-performing large model called Group Multi-view Vision Transformer (GMViT). In GMViT, the view-level ViT first establishes relationships between view-level features. Additionally, to capture deeper features, we employ the grouping module to enhance view-level features into group-level features. Finally, the group-level ViT aggregates group-level features into complete, well-formed 3D shape descriptors. Notably, in both ViTs, we introduce spatial encoding of camera coordinates as innovative position embeddings. Furthermore, we propose two compressed versions based on GMViT, namely GMViT-simple and GMViT-mini. To enhance the training effectiveness of the small models, we introduce a knowledge distillation method throughout the GMViT process, where the key outputs of each GMViT component serve as distillation targets. Extensive experiments demonstrate the efficacy of the proposed method. The large model GMViT achieves excellent 3D classification and retrieval results on the benchmark datasets ModelNet, ShapeNetCore55, and MCB. The smaller models, GMViT-simple and GMViT-mini, reduce the parameter size by 8 and 17.6 times, respectively, and improve shape recognition speed by 1.5 times on average, while preserving at least 90% of the classification and retrieval performance.

{{</citation>}}


### (41/59) ReSynthDetect: A Fundus Anomaly Detection Network with Reconstruction and Synthetic Features (Jingqi Niu et al., 2023)

{{<citation>}}

Jingqi Niu, Qinji Yu, Shiwen Dong, Zilong Wang, Kang Dang, Xiaowei Ding. (2023)  
**ReSynthDetect: A Fundus Anomaly Detection Network with Reconstruction and Synthetic Features**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.16470v1)  

---


**ABSTRACT**  
Detecting anomalies in fundus images through unsupervised methods is a challenging task due to the similarity between normal and abnormal tissues, as well as their indistinct boundaries. The current methods have limitations in accurately detecting subtle anomalies while avoiding false positives. To address these challenges, we propose the ReSynthDetect network which utilizes a reconstruction network for modeling normal images, and an anomaly generator that produces synthetic anomalies consistent with the appearance of fundus images. By combining the features of consistent anomaly generation and image reconstruction, our method is suited for detecting fundus abnormalities. The proposed approach has been extensively tested on benchmark datasets such as EyeQ and IDRiD, demonstrating state-of-the-art performance in both image-level and pixel-level anomaly detection. Our experiments indicate a substantial 9% improvement in AUROC on EyeQ and a significant 17.1% improvement in AUPR on IDRiD.

{{</citation>}}


### (42/59) Domain Generalization with Vital Phase Augmentation (Ingyun Lee et al., 2023)

{{<citation>}}

Ingyun Lee, Wooju Lee, Hyun Myung. (2023)  
**Domain Generalization with Vital Phase Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Augmentation, ImageNet  
[Paper Link](http://arxiv.org/abs/2312.16451v1)  

---


**ABSTRACT**  
Deep neural networks have shown remarkable performance in image classification. However, their performance significantly deteriorates with corrupted input data. Domain generalization methods have been proposed to train robust models against out-of-distribution data. Data augmentation in the frequency domain is one of such approaches that enable a model to learn phase features to establish domain-invariant representations. This approach changes the amplitudes of the input data while preserving the phases. However, using fixed phases leads to susceptibility to phase fluctuations because amplitudes and phase fluctuations commonly occur in out-of-distribution. In this study, to address this problem, we introduce an approach using finite variation of the phases of input data rather than maintaining fixed phases. Based on the assumption that the degree of domain-invariant features varies for each phase, we propose a method to distinguish phases based on this degree. In addition, we propose a method called vital phase augmentation (VIPAug) that applies the variation to the phases differently according to the degree of domain-invariant features of given phases. The model depends more on the vital phases that contain more domain-invariant features for attaining robustness to amplitude and phase fluctuations. We present experimental evaluations of our proposed approach, which exhibited improved performance for both clean and corrupted data. VIPAug achieved SOTA performance on the benchmark CIFAR-10 and CIFAR-100 datasets, as well as near-SOTA performance on the ImageNet-100 and ImageNet datasets. Our code is available at https://github.com/excitedkid/vipaug.

{{</citation>}}


### (43/59) RefineNet: Enhancing Text-to-Image Conversion with High-Resolution and Detail Accuracy through Hierarchical Transformers and Progressive Refinement (Fan Shi, 2023)

{{<citation>}}

Fan Shi. (2023)  
**RefineNet: Enhancing Text-to-Image Conversion with High-Resolution and Detail Accuracy through Hierarchical Transformers and Progressive Refinement**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.17274v1)  

---


**ABSTRACT**  
In this research, we introduce RefineNet, a novel architecture designed to address resolution limitations in text-to-image conversion systems. We explore the challenges of generating high-resolution images from textual descriptions, focusing on the trade-offs between detail accuracy and computational efficiency. RefineNet leverages a hierarchical Transformer combined with progressive and conditional refinement techniques, outperforming existing models in producing detailed and high-quality images. Through extensive experiments on diverse datasets, we demonstrate RefineNet's superiority in clarity and resolution, particularly in complex image categories like animals, plants, and human faces. Our work not only advances the field of image-to-text conversion but also opens new avenues for high-fidelity image generation in various applications.

{{</citation>}}


### (44/59) Segment Change Model (SCM) for Unsupervised Change detection in VHR Remote Sensing Images: a Case Study of Buildings (Xiaoliang Tan et al., 2023)

{{<citation>}}

Xiaoliang Tan, Guanzhou Chen, Tong Wang, Jiaqi Wang, Xiaodong Zhang. (2023)  
**Segment Change Model (SCM) for Unsupervised Change detection in VHR Remote Sensing Images: a Case Study of Buildings**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.16410v1)  

---


**ABSTRACT**  
The field of Remote Sensing (RS) widely employs Change Detection (CD) on very-high-resolution (VHR) images. A majority of extant deep-learning-based methods hinge on annotated samples to complete the CD process. Recently, the emergence of Vision Foundation Model (VFM) enables zero-shot predictions in particular vision tasks. In this work, we propose an unsupervised CD method named Segment Change Model (SCM), built upon the Segment Anything Model (SAM) and Contrastive Language-Image Pre-training (CLIP). Our method recalibrates features extracted at different scales and integrates them in a top-down manner to enhance discriminative change edges. We further design an innovative Piecewise Semantic Attention (PSA) scheme, which can offer semantic representation without training, thereby minimize pseudo change phenomenon. Through conducting experiments on two public datasets, the proposed SCM increases the mIoU from 46.09% to 53.67% on the LEVIR-CD dataset, and from 47.56% to 52.14% on the WHU-CD dataset. Our codes are available at https://github.com/StephenApX/UCD-SCM.

{{</citation>}}


## cs.AI (2)



### (45/59) AI-driven platform for systematic nomenclature and intelligent knowledge acquisition of natural medicinal materials (Zijie Yang et al., 2023)

{{<citation>}}

Zijie Yang, Yongjing Yin, Chaojun Kong, Tiange Chi, Wufan Tao, Yue Zhang, Tian Xu. (2023)  
**AI-driven platform for systematic nomenclature and intelligent knowledge acquisition of natural medicinal materials**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-DB, cs-IR, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.00020v1)  

---


**ABSTRACT**  
Natural Medicinal Materials (NMMs) have a long history of global clinical applications, accompanied by extensive informational records. Despite their significant impact on healthcare, the field faces a major challenge: the non-standardization of NMM knowledge, stemming from historical complexities and causing limitations in broader applications. To address this, we introduce a Systematic Nomenclature for NMMs, underpinned by ShennongAlpha, an AI-driven platform designed for intelligent knowledge acquisition. This nomenclature system enables precise identification and differentiation of NMMs. ShennongAlpha, cataloging over ten thousand NMMs with standardized bilingual information, enhances knowledge management and application capabilities, thereby overcoming traditional barriers. Furthermore, it pioneers AI-empowered conversational knowledge acquisition and standardized machine translation. These synergistic innovations mark the first major advance in integrating domain-specific NMM knowledge with AI, propelling research and applications across both NMM and AI fields while establishing a groundbreaking precedent in this crucial area.

{{</citation>}}


### (46/59) Robustness Verification for Knowledge-Based Logic of Risky Driving Scenes (Xia Wang et al., 2023)

{{<citation>}}

Xia Wang, Anda Liang, Jonathan Sprinkle, Taylor T. Johnson. (2023)  
**Robustness Verification for Knowledge-Based Logic of Risky Driving Scenes**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.16364v1)  

---


**ABSTRACT**  
Many decision-making scenarios in modern life benefit from the decision support of artificial intelligence algorithms, which focus on a data-driven philosophy and automated programs or systems. However, crucial decision issues related to security, fairness, and privacy should consider more human knowledge and principles to supervise such AI algorithms to reach more proper solutions and to benefit society more effectively. In this work, we extract knowledge-based logic that defines risky driving formats learned from public transportation accident datasets, which haven't been analyzed in detail to the best of our knowledge. More importantly, this knowledge is critical for recognizing traffic hazards and could supervise and improve AI models in safety-critical systems. Then we use automated verification methods to verify the robustness of such logic. More specifically, we gather 72 accident datasets from Data.gov and organize them by state. Further, we train Decision Tree and XGBoost models on each state's dataset, deriving accident judgment logic. Finally, we deploy robustness verification on these tree-based models under multiple parameter combinations.

{{</citation>}}


## cs.HC (1)



### (47/59) Participatory prompting: a user-centric research method for eliciting AI assistance opportunities in knowledge workflows (Advait Sarkar et al., 2023)

{{<citation>}}

Advait Sarkar, Ian Drosos, Rob Deline, Andrew D. Gordon, Carina Negreanu, Sean Rintel, Jack Williams, Benjamin Zorn. (2023)  
**Participatory prompting: a user-centric research method for eliciting AI assistance opportunities in knowledge workflows**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.16633v1)  

---


**ABSTRACT**  
Generative AI, such as image generation models and large language models, stands to provide tremendous value to end-user programmers in creative and knowledge workflows. Current research methods struggle to engage end-users in a realistic conversation that balances the actually existing capabilities of generative AI with the open-ended nature of user workflows and the many opportunities for the application of this technology. In this work-in-progress paper, we introduce participatory prompting, a method for eliciting opportunities for generative AI in end-user workflows. The participatory prompting method combines a contextual inquiry and a researcher-mediated interaction with a generative model, which helps study participants interact with a generative model without having to develop prompting strategies of their own. We discuss the ongoing development of a study whose aim will be to identify end-user programming opportunities for generative AI in data analysis workflows.

{{</citation>}}


## eess.SY (1)



### (48/59) Autonomous Driving using Residual Sensor Fusion and Deep Reinforcement Learning (Amin Jalal Aghdasian et al., 2023)

{{<citation>}}

Amin Jalal Aghdasian, Amirhossein Heydarian Ardakani, Kianoush Aqabakee, Farzaneh Abdollahi. (2023)  
**Autonomous Driving using Residual Sensor Fusion and Deep Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.16620v1)  

---


**ABSTRACT**  
This paper proposes a novel approach by integrating sensor fusion with deep reinforcement learning, specifically the Soft Actor-Critic (SAC) algorithm, to develop an optimal control policy for self-driving cars. Our system employs a two-branch fusion method for vehicle image and tracking sensor data, leveraging the strengths of residual structures and identity mapping to enhance agent training. Through comprehensive comparisons, we demonstrate the efficacy of information fusion and establish the superiority of our selected algorithm over alternative approaches. Our work advances the field of autonomous driving and demonstrates the potential of reinforcement learning in enabling intelligent vehicle decision-making.

{{</citation>}}


## cs.SD (2)



### (49/59) Self-supervised Pretraining for Robust Personalized Voice Activity Detection in Adverse Conditions (Holger Severin Bovbjerg et al., 2023)

{{<citation>}}

Holger Severin Bovbjerg, Jesper Jensen, Jan Østergaard, Zheng-Hua Tan. (2023)  
**Self-supervised Pretraining for Robust Personalized Voice Activity Detection in Adverse Conditions**  

---
Primary Category: cs.SD  
Categories: 68T10, I-2-6, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.16613v1)  

---


**ABSTRACT**  
In this paper, we propose the use of self-supervised pretraining on a large unlabelled data set to improve the performance of a personalized voice activity detection (VAD) model in adverse conditions. We pretrain a long short-term memory (LSTM)-encoder using the autoregressive predictive coding (APC) framework and fine-tune it for personalized VAD. We also propose a denoising variant of APC, with the goal of improving the robustness of personalized VAD. The trained models are systematically evaluated on both clean speech and speech contaminated by various types of noise at different SNR-levels and compared to a purely supervised model. Our experiments show that self-supervised pretraining not only improves performance in clean conditions, but also yields models which are more robust to adverse conditions compared to purely supervised learning.

{{</citation>}}


### (50/59) Frame-level emotional state alignment method for speech emotion recognition (Qifei Li et al., 2023)

{{<citation>}}

Qifei Li, Yingming Gao, Cong Wang, Yayue Deng, Jinlong Xue, Yichen Han, Ya Li. (2023)  
**Frame-level emotional state alignment method for speech emotion recognition**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.16383v1)  

---


**ABSTRACT**  
Speech emotion recognition (SER) systems aim to recognize human emotional state during human-computer interaction. Most existing SER systems are trained based on utterance-level labels. However, not all frames in an audio have affective states consistent with utterance-level label, which makes it difficult for the model to distinguish the true emotion of the audio and perform poorly. To address this problem, we propose a frame-level emotional state alignment method for SER. First, we fine-tune HuBERT model to obtain a SER system with task-adaptive pretraining (TAPT) method, and extract embeddings from its transformer layers to form frame-level pseudo-emotion labels with clustering. Then, the pseudo labels are used to pretrain HuBERT. Hence, the each frame output of HuBERT has corresponding emotional information. Finally, we fine-tune the above pretrained HuBERT for SER by adding an attention layer on the top of it, which can focus only on those frames that are emotionally more consistent with utterance-level label. The experimental results performed on IEMOCAP indicate that our proposed method performs better than state-of-the-art (SOTA) methods.

{{</citation>}}


## q-bio.GN (1)



### (51/59) scRNA-seq Data Clustering by Cluster-aware Iterative Contrastive Learning (Weikang Jiang et al., 2023)

{{<citation>}}

Weikang Jiang, Jinxian Wang, Jihong Guan, Shuigeng Zhou. (2023)  
**scRNA-seq Data Clustering by Cluster-aware Iterative Contrastive Learning**  

---
Primary Category: q-bio.GN  
Categories: cs-AI, cs-LG, q-bio-GN, q-bio.GN  
Keywords: Contrastive Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.16600v1)  

---


**ABSTRACT**  
Single-cell RNA sequencing (scRNA-seq) enables researchers to analyze gene expression at single-cell level. One important task in scRNA-seq data analysis is unsupervised clustering, which helps identify distinct cell types, laying down the foundation for other downstream analysis tasks. In this paper, we propose a novel method called Cluster-aware Iterative Contrastive Learning (CICL in short) for scRNA-seq data clustering, which utilizes an iterative representation learning and clustering framework to progressively learn the clustering structure of scRNA-seq data with a cluster-aware contrastive loss. CICL consists of a Transformer encoder, a clustering head, a projection head and a contrastive loss module. First, CICL extracts the feature vectors of the original and augmented data by the Transformer encoder. Then, it computes the clustering centroids by K-means and employs the student t-distribution to assign pseudo-labels to all cells in the clustering head. The projection-head uses a Multi-Layer Perceptron (MLP) to obtain projections of the augmented data. At last, both pseudo-labels and projections are used in the contrastive loss to guide the model training. Such a process goes iteratively so that the clustering result becomes better and better. Extensive experiments on 25 real world scRNA-seq datasets show that CICL outperforms the SOTA methods. Concretely, CICL surpasses the existing methods by from 14% to 280%, and from 5% to 133% on average in terms of performance metrics ARI and NMI respectively.

{{</citation>}}


## cs.SE (2)



### (52/59) EasyView: Bringing Performance Profiles into Integrated Development Environments (Qidong Zhao et al., 2023)

{{<citation>}}

Qidong Zhao, Milind Chabbi, Xu Liu. (2023)  
**EasyView: Bringing Performance Profiles into Integrated Development Environments**  

---
Primary Category: cs.SE  
Categories: cs-PF, cs-SE, cs.SE  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2312.16598v1)  

---


**ABSTRACT**  
Dynamic program analysis (also known as profiling) is well-known for its powerful capabilities of identifying performance inefficiencies in software packages. Although a large number of dynamic program analysis techniques are developed in academia and industry, very few of them are widely used by software developers in their regular software developing activities. There are three major reasons. First, the dynamic analysis tools (also known as profilers) are disjoint from the coding environments such as IDEs and editors; frequently switching focus between them significantly complicates the entire cycle of software development. Second, mastering various tools to interpret their analysis results requires substantial efforts; even worse, many tools have their own design of graphical user interfaces (GUI) for data presentation, which steepens the learning curves. Third, most existing tools expose few interfaces to support user-defined analysis, which makes the tools less customizable to fulfill diverse user demands. We develop EasyView, a general solution to integrate the interpretation and visualization of various profiling results in the coding environments, which bridges software developers with profilers to provide easy and intuitive dynamic analysis during the code development cycle. The novelty of EasyView is three-fold. First, we develop a generic data format, which enables EasyView to support mainstream profilers for different languages. Second, we develop a set of customizable schemes to analyze and visualize the profiles in intuitive ways. Third, we tightly integrate EasyView with popular coding environments, such as Microsoft Visual Studio Code, with easy code exploration and user interaction. Our evaluation shows that EasyView is able to support various profilers for different languages and provide unique insights into performance inefficiencies in different domains.

{{</citation>}}


### (53/59) Toward Methodical Discovery and Handling of Hidden Assumptions in Complex Systems and Models (David Harel et al., 2023)

{{<citation>}}

David Harel, Uwe Aßmann, Fabiana Fournier, Lior Limonad, Assaf Marron, Smadar Szekely. (2023)  
**Toward Methodical Discovery and Handling of Hidden Assumptions in Complex Systems and Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.16507v1)  

---


**ABSTRACT**  
Methodologies for development of complex systems and models include external reviews by domain and technology experts. Among others, such reviews can uncover undocumented built-in assumptions that may be critical for correct and safe operation or constrain applicability. Since such assumptions may still escape human-centered processes like reviews, agile development, and risk analyses, here, we contribute toward making this process more methodical and automatable. We first present a blueprint for a taxonomy and formalization of the problem. We then show that a variety of digital artifacts of the system or model can be automatically checked against extensive reference knowledge. Since mimicking the breadth and depth of knowledge and skills of experts may appear unattainable, we illustrate the basic feasibility of automation with rudimentary experiments using OpenAI's ChatGPT. We believe that systematic handling of this aspect of system engineering can contribute significantly to the quality and safety of complex systems and models, and to the efficiency of development projects. We dedicate this work to Werner Damm, whose contributions to modeling and model-based development, in industry and academia, with a special focus on safety, helped establish a solid foundation to our discipline and to the work of many scientists and professionals, including, naturally, the approaches and techniques described here.

{{</citation>}}


## cs.SI (2)



### (54/59) Identification of Opinion Leaders in a Telegram Network of Forwarded Messages (Giulia Tucci, 2023)

{{<citation>}}

Giulia Tucci. (2023)  
**Identification of Opinion Leaders in a Telegram Network of Forwarded Messages**  

---
Primary Category: cs.SI  
Categories: H-5-0; H-1-1, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2312.16528v1)  

---


**ABSTRACT**  
Unraveling the role of opinion leaders in the digital realm, this study investigates the influence of key actors on Telegram, a hybrid platform that combines messaging app features with social network dynamics, where channel administrators gain a unique authoritative role. This research aims to create a method to identify opinion leaders in a network of forwarded messages on Telegram, adapting a method originally developed to be applied to Twitter. The adapted method is showcased through a case study during the 2022 Brazilian Presidential Election, involving the monitoring of 25 pro-Bolsonaro groups. The findings contribute to understanding the dynamics of digital opinion leadership, particularly in politically charged environments.

{{</citation>}}


### (55/59) Diagnosis of Small-world Bias in Random Graphs (Georgios Argyris, 2023)

{{<citation>}}

Georgios Argyris. (2023)  
**Diagnosis of Small-world Bias in Random Graphs**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.16525v1)  

---


**ABSTRACT**  
Background: Imagine a paper with n nodes on it where each pair undergoes a coin toss experiment; if heads we connect the pair with an undirected link, while tails maintain the disconnection. This procedure yields a random graph. Now consider duplicating this network onto another paper with a slight bias-a fraction of its links (approximately 1/10) undergo rearrangement. If we shuffle the two papers, how can we distinguish the pure random graph from the biased one? Results: In response to this challenge, we propose a novel metric called Randomness Index (RI). The closer the metric to zero is, the higher degree of randomness in the graph. The RI can distinguish between dense small-world networks and dense random graphs; a distinction which is impossible by conventional small-world properties like clustering coefficient and average path length. To validate its effectiveness, we apply the RI to temporal correlation networks of stock indices. Our findings reveal a reduction in randomness during global economic recession periods. Conclusion: The RI emerges as a powerful metric capable of characterizing small-world topology, especially in scenarios where other network measures fail. Beyond its utility in network analysis, the RI is promising for change-point (anomaly) detection in dynamical systems studied by means of multivariate time series.

{{</citation>}}


## cs.ET (1)



### (56/59) Attention-Enhanced Reservoir Computing (Felix Köster et al., 2023)

{{<citation>}}

Felix Köster, Kazutaka Kanno, Jun Ohkubo, Atsushi Uchida. (2023)  
**Attention-Enhanced Reservoir Computing**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs-LG, cs.ET  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.16503v1)  

---


**ABSTRACT**  
Photonic reservoir computing has been recently utilized in time series forecasting as the need for hardware implementations to accelerate these predictions has increased. Forecasting chaotic time series remains a significant challenge, an area where the conventional reservoir computing framework encounters limitations of prediction accuracy. We introduce an attention mechanism to the reservoir computing model in the output stage. This attention layer is designed to prioritize distinct features and temporal sequences, thereby substantially enhancing the forecasting accuracy. Our results show that a photonic reservoir computer enhanced with the attention mechanism exhibits improved forecasting capabilities for smaller reservoirs. These advancements highlight the transformative possibilities of reservoir computing for practical applications where accurate forecasting of chaotic time series is crucial.

{{</citation>}}


## eess.IV (1)



### (57/59) Learn From Orientation Prior for Radiograph Super-Resolution: Orientation Operator Transformer (Yongsong Huang et al., 2023)

{{<citation>}}

Yongsong Huang, Tomo Miyazaki, Xiaofeng Liu, Kaiyuan Jiang, Zhengmi Tang, Shinichiro Omachi. (2023)  
**Learn From Orientation Prior for Radiograph Super-Resolution: Orientation Operator Transformer**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.16455v1)  

---


**ABSTRACT**  
Background and objective: High-resolution radiographic images play a pivotal role in the early diagnosis and treatment of skeletal muscle-related diseases. It is promising to enhance image quality by introducing single-image super-resolution (SISR) model into the radiology image field. However, the conventional image pipeline, which can learn a mixed mapping between SR and denoising from the color space and inter-pixel patterns, poses a particular challenge for radiographic images with limited pattern features. To address this issue, this paper introduces a novel approach: Orientation Operator Transformer - $O^{2}$former. Methods: We incorporate an orientation operator in the encoder to enhance sensitivity to denoising mapping and to integrate orientation prior. Furthermore, we propose a multi-scale feature fusion strategy to amalgamate features captured by different receptive fields with the directional prior, thereby providing a more effective latent representation for the decoder. Based on these innovative components, we propose a transformer-based SISR model, i.e., $O^{2}$former, specifically designed for radiographic images. Results: The experimental results demonstrate that our method achieves the best or second-best performance in the objective metrics compared with the competitors at $\times 4$ upsampling factor. For qualitative, more objective details are observed to be recovered. Conclusions: In this study, we propose a novel framework called $O^{2}$former for radiological image super-resolution tasks, which improves the reconstruction model's performance by introducing an orientation operator and multi-scale feature fusion strategy. Our approach is promising to further promote the radiographic image enhancement field.

{{</citation>}}


## cs.RO (1)



### (58/59) Visual Spatial Attention and Proprioceptive Data-Driven Reinforcement Learning for Robust Peg-in-Hole Task Under Variable Conditions (André Yuji Yasutomi et al., 2023)

{{<citation>}}

André Yuji Yasutomi, Hideyuki Ichiwara, Hiroshi Ito, Hiroki Mori, Tetsuya Ogata. (2023)  
**Visual Spatial Attention and Proprioceptive Data-Driven Reinforcement Learning for Robust Peg-in-Hole Task Under Variable Conditions**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Attention, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.16438v1)  

---


**ABSTRACT**  
Anchor-bolt insertion is a peg-in-hole task performed in the construction field for holes in concrete. Efforts have been made to automate this task, but the variable lighting and hole surface conditions, as well as the requirements for short setup and task execution time make the automation challenging. In this study, we introduce a vision and proprioceptive data-driven robot control model for this task that is robust to challenging lighting and hole surface conditions. This model consists of a spatial attention point network (SAP) and a deep reinforcement learning (DRL) policy that are trained jointly end-to-end to control the robot. The model is trained in an offline manner, with a sample-efficient framework designed to reduce training time and minimize the reality gap when transferring the model to the physical world. Through evaluations with an industrial robot performing the task in 12 unknown holes, starting from 16 different initial positions, and under three different lighting conditions (two with misleading shadows), we demonstrate that SAP can generate relevant attention points of the image even in challenging lighting conditions. We also show that the proposed model enables task execution with higher success rate and shorter task completion time than various baselines. Due to the proposed model's high effectiveness even in severe lighting, initial positions, and hole conditions, and the offline training framework's high sample-efficiency and short training time, this approach can be easily applied to construction.

{{</citation>}}


## cs.AR (1)



### (59/59) Gemini: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators (Jingwei Cai et al., 2023)

{{<citation>}}

Jingwei Cai, Zuotong Wu, Sen Peng, Yuchen Wei, Zhanhong Tan, Guiming Shi, Mingyu Gao, Kaisheng Ma. (2023)  
**Gemini: Mapping and Architecture Co-exploration for Large-scale DNN Chiplet Accelerators**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.16436v1)  

---


**ABSTRACT**  
Chiplet technology enables the integration of an increasing number of transistors on a single accelerator with higher yield in the post-Moore era, addressing the immense computational demands arising from rapid AI advancements. However, it also introduces more expensive packaging costs and costly Die-to-Die (D2D) interfaces, which require more area, consume higher power, and offer lower bandwidth than on-chip interconnects. Maximizing the benefits and minimizing the drawbacks of chiplet technology is crucial for developing large-scale DNN chiplet accelerators, which poses challenges to both architecture and mapping. Despite its importance in the post-Moore era, methods to address these challenges remain scarce.

{{</citation>}}
