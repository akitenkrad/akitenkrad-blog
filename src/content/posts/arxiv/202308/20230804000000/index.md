---
draft: false
title: "arXiv @ 2023.08.04"
date: 2023-08-04
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.04"
    identifier: arxiv_20230804
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (18)](#cslg-18)
- [cs.IT (1)](#csit-1)
- [cs.AI (5)](#csai-5)
- [cs.CV (24)](#cscv-24)
- [cs.CL (10)](#cscl-10)
- [cs.RO (4)](#csro-4)
- [cs.SE (2)](#csse-2)
- [cs.CR (5)](#cscr-5)
- [cs.SI (1)](#cssi-1)
- [eess.IV (1)](#eessiv-1)
- [math.DS (1)](#mathds-1)
- [cs.IR (2)](#csir-2)

## cs.LG (18)



### (1/74) VertexSerum: Poisoning Graph Neural Networks for Link Inference (Ruyi Ding et al., 2023)

{{<citation>}}

Ruyi Ding, Shijin Duan, Xiaolin Xu, Yunsi Fei. (2023)  
**VertexSerum: Poisoning Graph Neural Networks for Link Inference**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.01469v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have brought superb performance to various applications utilizing graph structural data, such as social analysis and fraud detection. The graph links, e.g., social relationships and transaction history, are sensitive and valuable information, which raises privacy concerns when using GNNs. To exploit these vulnerabilities, we propose VertexSerum, a novel graph poisoning attack that increases the effectiveness of graph link stealing by amplifying the link connectivity leakage. To infer node adjacency more accurately, we propose an attention mechanism that can be embedded into the link detection network. Our experiments demonstrate that VertexSerum significantly outperforms the SOTA link inference attack, improving the AUC scores by an average of $9.8\%$ across four real-world datasets and three different GNN structures. Furthermore, our experiments reveal the effectiveness of VertexSerum in both black-box and online learning settings, further validating its applicability in real-world scenarios.

{{</citation>}}


### (2/74) DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales (Zhewei Yao et al., 2023)

{{<citation>}}

Zhewei Yao, Reza Yazdani Aminabadi, Olatunji Ruwase, Samyam Rajbhandari, Xiaoxia Wu, Ammar Ahmad Awan, Jeff Rasley, Minjia Zhang, Conglong Li, Connor Holmes, Zhongzhu Zhou, Michael Wyatt, Molly Smith, Lev Kurilenko, Heyang Qin, Masahiro Tanaka, Shuai Che, Shuaiwen Leon Song, Yuxiong He. (2023)  
**DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: AI, ChatGPT, GPT, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.01320v1)  

---


**ABSTRACT**  
ChatGPT-like models have revolutionized various applications in artificial intelligence, from summarization and coding to translation, matching or even surpassing human performance. However, the current landscape lacks an accessible, efficient, and cost-effective end-to-end RLHF (Reinforcement Learning with Human Feedback) training pipeline for these powerful models, particularly when training at the scale of billions of parameters. This paper introduces DeepSpeed-Chat, a novel system that democratizes RLHF training, making it accessible to the AI community. DeepSpeed-Chat offers three key capabilities: an easy-to-use training and inference experience for ChatGPT-like models, a DeepSpeed-RLHF pipeline that replicates the training pipeline from InstructGPT, and a robust DeepSpeed-RLHF system that combines various optimizations for training and inference in a unified way. The system delivers unparalleled efficiency and scalability, enabling training of models with hundreds of billions of parameters in record time and at a fraction of the cost. With this development, DeepSpeed-Chat paves the way for broader access to advanced RLHF training, even for data scientists with limited resources, thereby fostering innovation and further development in the field of AI.

{{</citation>}}


### (3/74) Lode Encoder: AI-constrained co-creativity (Debosmita Bhaumik et al., 2023)

{{<citation>}}

Debosmita Bhaumik, Ahmed Khalifa, Julian Togelius. (2023)  
**Lode Encoder: AI-constrained co-creativity**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01312v1)  

---


**ABSTRACT**  
We present Lode Encoder, a gamified mixed-initiative level creation system for the classic platform-puzzle game Lode Runner. The system is built around several autoencoders which are trained on sets of Lode Runner levels. When fed with the user's design, each autoencoder produces a version of that design which is closer in style to the levels that it was trained on. The Lode Encoder interface allows the user to build and edit levels through 'painting' from the suggestions provided by the autoencoders. Crucially, in order to encourage designers to explore new possibilities, the system does not include more traditional editing tools. We report on the system design and training procedure, as well as on the evolution of the system itself and user tests.

{{</citation>}}


### (4/74) EmbeddingTree: Hierarchical Exploration of Entity Features in Embedding (Yan Zheng et al., 2023)

{{<citation>}}

Yan Zheng, Junpeng Wang, Chin-Chia Michael Yeh, Yujie Fan, Huiyuan Chen, Liang Wang, Wei Zhang. (2023)  
**EmbeddingTree: Hierarchical Exploration of Entity Features in Embedding**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.01329v1)  

---


**ABSTRACT**  
Embedding learning transforms discrete data entities into continuous numerical representations, encoding features/properties of the entities. Despite the outstanding performance reported from different embedding learning algorithms, few efforts were devoted to structurally interpreting how features are encoded in the learned embedding space. This work proposes EmbeddingTree, a hierarchical embedding exploration algorithm that relates the semantics of entity features with the less-interpretable embedding vectors. An interactive visualization tool is also developed based on EmbeddingTree to explore high-dimensional embeddings. The tool helps users discover nuance features of data entities, perform feature denoising/injecting in embedding training, and generate embeddings for unseen entities. We demonstrate the efficacy of EmbeddingTree and our visualization tool through embeddings generated for industry-scale merchant data and the public 30Music listening/playlists dataset.

{{</citation>}}


### (5/74) A Probabilistic Approach to Self-Supervised Learning using Cyclical Stochastic Gradient MCMC (Masoumeh Javanbakhat et al., 2023)

{{<citation>}}

Masoumeh Javanbakhat, Christoph Lippert. (2023)  
**A Probabilistic Approach to Self-Supervised Learning using Cyclical Stochastic Gradient MCMC**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.01271v1)  

---


**ABSTRACT**  
In this paper we present a practical Bayesian self-supervised learning method with Cyclical Stochastic Gradient Hamiltonian Monte Carlo (cSGHMC). Within this framework, we place a prior over the parameters of a self-supervised learning model and use cSGHMC to approximate the high dimensional and multimodal posterior distribution over the embeddings. By exploring an expressive posterior over the embeddings, Bayesian self-supervised learning produces interpretable and diverse representations. Marginalizing over these representations yields a significant gain in performance, calibration and out-of-distribution detection on a variety of downstream classification tasks. We provide experimental results on multiple classification tasks on four challenging datasets. Moreover, we demonstrate the effectiveness of the proposed method in out-of-distribution detection using the SVHN and CIFAR-10 datasets.

{{</citation>}}


### (6/74) Calibration in Deep Learning: A Survey of the State-of-the-Art (Cheng Wang, 2023)

{{<citation>}}

Cheng Wang. (2023)  
**Calibration in Deep Learning: A Survey of the State-of-the-Art**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01222v1)  

---


**ABSTRACT**  
Calibrating deep neural models plays an important role in building reliable, robust AI systems in safety-critical applications. Recent work has shown that modern neural networks that possess high predictive capability are poorly calibrated and produce unreliable model predictions. Though deep learning models achieve remarkable performance on various benchmarks, the study of model calibration and reliability is relatively underexplored. Ideal deep models should have not only high predictive performance but also be well calibrated. There have been some recent methods proposed to calibrate deep models by using different mechanisms. In this survey, we review the state-of-the-art calibration methods and provide an understanding of their principles for performing model calibration. First, we start with the definition of model calibration and explain the root causes of model miscalibration. Then we introduce the key metrics that can measure this aspect. It is followed by a summary of calibration methods that we roughly classified into four categories: post-hoc calibration, regularization methods, uncertainty estimation, and composition methods. We also covered some recent advancements in calibrating large models, particularly large language models (LLMs). Finally, we discuss some open issues, challenges, and potential directions.

{{</citation>}}


### (7/74) Using ScrutinAI for Visual Inspection of DNN Performance in a Medical Use Case (Rebekka Görge et al., 2023)

{{<citation>}}

Rebekka Görge, Elena Haedecke, Michael Mock. (2023)  
**Using ScrutinAI for Visual Inspection of DNN Performance in a Medical Use Case**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-HC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01220v1)  

---


**ABSTRACT**  
Our Visual Analytics (VA) tool ScrutinAI supports human analysts to investigate interactively model performanceand data sets. Model performance depends on labeling quality to a large extent. In particular in medical settings, generation of high quality labels requires in depth expert knowledge and is very costly. Often, data sets are labeled by collecting opinions of groups of experts. We use our VA tool to analyse the influence of label variations between different experts on the model performance. ScrutinAI facilitates to perform a root cause analysis that distinguishes weaknesses of deep neural network (DNN) models caused by varying or missing labeling quality from true weaknesses. We scrutinize the overall detection of intracranial hemorrhages and the more subtle differentiation between subtypes in a publicly available data set.

{{</citation>}}


### (8/74) A Transformer-based Prediction Method for Depth of Anesthesia During Target-controlled Infusion of Propofol and Remifentanil (Yongkang He et al., 2023)

{{<citation>}}

Yongkang He, Siyuan Peng, Mingjin Chen, Zhijing Yang, Yuanhui Chen. (2023)  
**A Transformer-based Prediction Method for Depth of Anesthesia During Target-controlled Infusion of Propofol and Remifentanil**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2308.01929v1)  

---


**ABSTRACT**  
Accurately predicting anesthetic effects is essential for target-controlled infusion systems. The traditional (PK-PD) models for Bispectral index (BIS) prediction require manual selection of model parameters, which can be challenging in clinical settings. Recently proposed deep learning methods can only capture general trends and may not predict abrupt changes in BIS. To address these issues, we propose a transformer-based method for predicting the depth of anesthesia (DOA) using drug infusions of propofol and remifentanil. Our method employs long short-term memory (LSTM) and gate residual network (GRN) networks to improve the efficiency of feature fusion and applies an attention mechanism to discover the interactions between the drugs. We also use label distribution smoothing and reweighting losses to address data imbalance. Experimental results show that our proposed method outperforms traditional PK-PD models and previous deep learning methods, effectively predicting anesthetic depth under sudden and deep anesthesia conditions.

{{</citation>}}


### (9/74) DySTreSS: Dynamically Scaled Temperature in Self-Supervised Contrastive Learning (Siladittya Manna et al., 2023)

{{<citation>}}

Siladittya Manna, Soumitri Chattopadhyay, Rakesh Dey, Saumik Bhattacharya, Umapada Pal. (2023)  
**DySTreSS: Dynamically Scaled Temperature in Self-Supervised Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Contrastive Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.01140v1)  

---


**ABSTRACT**  
In contemporary self-supervised contrastive algorithms like SimCLR, MoCo, etc., the task of balancing attraction between two semantically similar samples and repulsion between two samples from different classes is primarily affected by the presence of hard negative samples. While the InfoNCE loss has been shown to impose penalties based on hardness, the temperature hyper-parameter is the key to regulating the penalties and the trade-off between uniformity and tolerance. In this work, we focus our attention to improve the performance of InfoNCE loss in SSL by studying the effect of temperature hyper-parameter values. We propose a cosine similarity-dependent temperature scaling function to effectively optimize the distribution of the samples in the feature space. We further analyze the uniformity and tolerance metrics to investigate the optimal regions in the cosine similarity space for better optimization. Additionally, we offer a comprehensive examination of the behavior of local and global structures in the feature space throughout the pre-training phase, as the temperature varies. Experimental evidence shows that the proposed framework outperforms or is at par with the contrastive loss-based SSL algorithms. We believe our work (DySTreSS) on temperature scaling in SSL provides a foundation for future research in contrastive learning.

{{</citation>}}


### (10/74) Automatic Feature Engineering for Time Series Classification: Evaluation and Discussion (Aurélien Renault et al., 2023)

{{<citation>}}

Aurélien Renault, Alexis Bondu, Vincent Lemaire, Dominique Gay. (2023)  
**Automatic Feature Engineering for Time Series Classification: Evaluation and Discussion**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.01071v1)  

---


**ABSTRACT**  
Time Series Classification (TSC) has received much attention in the past two decades and is still a crucial and challenging problem in data science and knowledge engineering. Indeed, along with the increasing availability of time series data, many TSC algorithms have been suggested by the research community in the literature. Besides state-of-the-art methods based on similarity measures, intervals, shapelets, dictionaries, deep learning methods or hybrid ensemble methods, several tools for extracting unsupervised informative summary statistics, aka features, from time series have been designed in the recent years. Originally designed for descriptive analysis and visualization of time series with informative and interpretable features, very few of these feature engineering tools have been benchmarked for TSC problems and compared with state-of-the-art TSC algorithms in terms of predictive performance. In this article, we aim at filling this gap and propose a simple TSC process to evaluate the potential predictive performance of the feature sets obtained with existing feature engineering tools. Thus, we present an empirical study of 11 feature engineering tools branched with 9 supervised classifiers over 112 time series data sets. The analysis of the results of more than 10000 learning experiments indicate that feature-based methods perform as accurately as current state-of-the-art TSC algorithms, and thus should rightfully be considered further in the TSC literature.

{{</citation>}}


### (11/74) Graph Anomaly Detection at Group Level: A Topology Pattern Enhanced Unsupervised Approach (Xing Ai et al., 2023)

{{<citation>}}

Xing Ai, Jialong Zhou, Yulin Zhu, Gaolei Li, Tomasz P. Michalak, Xiapu Luo, Kai Zhou. (2023)  
**Graph Anomaly Detection at Group Level: A Topology Pattern Enhanced Unsupervised Approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.01063v1)  

---


**ABSTRACT**  
Graph anomaly detection (GAD) has achieved success and has been widely applied in various domains, such as fraud detection, cybersecurity, finance security, and biochemistry. However, existing graph anomaly detection algorithms focus on distinguishing individual entities (nodes or graphs) and overlook the possibility of anomalous groups within the graph. To address this limitation, this paper introduces a novel unsupervised framework for a new task called Group-level Graph Anomaly Detection (Gr-GAD). The proposed framework first employs a variant of Graph AutoEncoder (GAE) to locate anchor nodes that belong to potential anomaly groups by capturing long-range inconsistencies. Subsequently, group sampling is employed to sample candidate groups, which are then fed into the proposed Topology Pattern-based Graph Contrastive Learning (TPGCL) method. TPGCL utilizes the topology patterns of groups as clues to generate embeddings for each candidate group and thus distinct anomaly groups. The experimental results on both real-world and synthetic datasets demonstrate that the proposed framework shows superior performance in identifying and localizing anomaly groups, highlighting it as a promising solution for Gr-GAD. Datasets and codes of the proposed framework are at the github repository https://anonymous.4open.science/r/Topology-Pattern-Enhanced-Unsupervised-Group-level-Graph-Anomaly-Detection.

{{</citation>}}


### (12/74) Maximizing Success Rate of Payment Routing using Non-stationary Bandits (Aayush Chaudhary et al., 2023)

{{<citation>}}

Aayush Chaudhary, Abhinav Rai, Abhishek Gupta. (2023)  
**Maximizing Success Rate of Payment Routing using Non-stationary Bandits**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.01028v1)  

---


**ABSTRACT**  
This paper discusses the system architecture design and deployment of non-stationary multi-armed bandit approaches to determine a near-optimal payment routing policy based on the recent history of transactions. We propose a Routing Service architecture using a novel Ray-based implementation for optimally scaling bandit-based payment routing to over 10000 transactions per second, adhering to the system design requirements and ecosystem constraints with Payment Card Industry Data Security Standard (PCI DSS). We first evaluate the effectiveness of multiple bandit-based payment routing algorithms on a custom simulator to benchmark multiple non-stationary bandit approaches and identify the best hyperparameters. We then conducted live experiments on the payment transaction system on a fantasy sports platform Dream11. In the live experiments, we demonstrated that our non-stationary bandit-based algorithm consistently improves the success rate of transactions by 0.92\% compared to the traditional rule-based methods over one month.

{{</citation>}}


### (13/74) Enhancing Representation Learning for Periodic Time Series with Floss: A Frequency Domain Regularization Approach (Chunwei Yang et al., 2023)

{{<citation>}}

Chunwei Yang, Xiaoxu Chen, Lijun Sun, Hongyu Yang, Yuankai Wu. (2023)  
**Enhancing Representation Learning for Periodic Time Series with Floss: A Frequency Domain Regularization Approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2308.01011v1)  

---


**ABSTRACT**  
Time series analysis is a fundamental task in various application domains, and deep learning approaches have demonstrated remarkable performance in this area. However, many real-world time series data exhibit significant periodic or quasi-periodic dynamics that are often not adequately captured by existing deep learning-based solutions. This results in an incomplete representation of the underlying dynamic behaviors of interest. To address this gap, we propose an unsupervised method called Floss that automatically regularizes learned representations in the frequency domain. The Floss method first automatically detects major periodicities from the time series. It then employs periodic shift and spectral density similarity measures to learn meaningful representations with periodic consistency. In addition, Floss can be easily incorporated into both supervised, semi-supervised, and unsupervised learning frameworks. We conduct extensive experiments on common time series classification, forecasting, and anomaly detection tasks to demonstrate the effectiveness of Floss. We incorporate Floss into several representative deep learning solutions to justify our design choices and demonstrate that it is capable of automatically discovering periodic dynamics and improving state-of-the-art deep learning models.

{{</citation>}}


### (14/74) Wasserstein Diversity-Enriched Regularizer for Hierarchical Reinforcement Learning (Haorui Li et al., 2023)

{{<citation>}}

Haorui Li, Jiaqi Liang, Linjing Li, Daniel Zeng. (2023)  
**Wasserstein Diversity-Enriched Regularizer for Hierarchical Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.00989v1)  

---


**ABSTRACT**  
Hierarchical reinforcement learning composites subpolicies in different hierarchies to accomplish complex tasks.Automated subpolicies discovery, which does not depend on domain knowledge, is a promising approach to generating subpolicies.However, the degradation problem is a challenge that existing methods can hardly deal with due to the lack of consideration of diversity or the employment of weak regularizers. In this paper, we propose a novel task-agnostic regularizer called the Wasserstein Diversity-Enriched Regularizer (WDER), which enlarges the diversity of subpolicies by maximizing the Wasserstein distances among action distributions. The proposed WDER can be easily incorporated into the loss function of existing methods to boost their performance further.Experimental results demonstrate that our WDER improves performance and sample efficiency in comparison with prior work without modifying hyperparameters, which indicates the applicability and robustness of the WDER.

{{</citation>}}


### (15/74) From Sparse to Soft Mixtures of Experts (Joan Puigcerver et al., 2023)

{{<citation>}}

Joan Puigcerver, Carlos Riquelme, Basil Mustafa, Neil Houlsby. (2023)  
**From Sparse to Soft Mixtures of Experts**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.00951v1)  

---


**ABSTRACT**  
Sparse mixture of expert architectures (MoEs) scale model capacity without large increases in training or inference costs. Despite their success, MoEs suffer from a number of issues: training instability, token dropping, inability to scale the number of experts, or ineffective finetuning. In this work, we proposeSoft MoE, a fully-differentiable sparse Transformer that addresses these challenges, while maintaining the benefits of MoEs. Soft MoE performs an implicit soft assignment by passing different weighted combinations of all input tokens to each expert. As in other MoE works, experts in Soft MoE only process a subset of the (combined) tokens, enabling larger model capacity at lower inference cost. In the context of visual recognition, Soft MoE greatly outperforms standard Transformers (ViTs) and popular MoE variants (Tokens Choice and Experts Choice). For example, Soft MoE-Base/16 requires 10.5x lower inference cost (5.7x lower wall-clock time) than ViT-Huge/14 while matching its performance after similar training. Soft MoE also scales well: Soft MoE Huge/14 with 128 experts in 16 MoE layers has over 40x more parameters than ViT Huge/14, while inference time cost grows by only 2%, and it performs substantially better.

{{</citation>}}


### (16/74) QUANT: A Minimalist Interval Method for Time Series Classification (Angus Dempster et al., 2023)

{{<citation>}}

Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb. (2023)  
**QUANT: A Minimalist Interval Method for Time Series Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.00928v1)  

---


**ABSTRACT**  
We show that it is possible to achieve the same accuracy, on average, as the most accurate existing interval methods for time series classification on a standard set of benchmark datasets using a single type of feature (quantiles), fixed intervals, and an 'off the shelf' classifier. This distillation of interval-based approaches represents a fast and accurate method for time series classification, achieving state-of-the-art accuracy on the expanded set of 142 datasets in the UCR archive with a total compute time (training and inference) of less than 15 minutes using a single CPU core.

{{</citation>}}


### (17/74) Tango: rethinking quantization for graph neural network training on GPUs (Shiyang Chen et al., 2023)

{{<citation>}}

Shiyang Chen, Da Zheng, Caiwen Ding, Chengying Huan, Yuede Ji, Hang Liu. (2023)  
**Tango: rethinking quantization for graph neural network training on GPUs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.00890v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) are becoming increasingly popular due to their superior performance in critical graph-related tasks. While quantization is widely used to accelerate GNN computation, quantized training faces unprecedented challenges. Current quantized GNN training systems often have longer training times than their full-precision counterparts for two reasons: (i) addressing the accuracy challenge leads to excessive overhead, and (ii) the optimization potential exposed by quantization is not adequately leveraged. This paper introduces Tango which re-thinks quantization challenges and opportunities for graph neural network training on GPUs with three contributions: Firstly, we introduce efficient rules to maintain accuracy during quantized GNN training. Secondly, we design and implement quantization-aware primitives and inter-primitive optimizations that can speed up GNN training. Finally, we integrate Tango with the popular Deep Graph Library (DGL) system and demonstrate its superior performance over state-of-the-art approaches on various GNN models and datasets.

{{</citation>}}


### (18/74) Factor Graph Neural Networks (Zhen Zhang et al., 2023)

{{<citation>}}

Zhen Zhang, Mohammed Haroon Dupty, Fan Wu, Javen Qinfeng Shi, Wee Sun Lee. (2023)  
**Factor Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.00887v1)  

---


**ABSTRACT**  
In recent years, we have witnessed a surge of Graph Neural Networks (GNNs), most of which can learn powerful representations in an end-to-end fashion with great success in many real-world applications. They have resemblance to Probabilistic Graphical Models (PGMs), but break free from some limitations of PGMs. By aiming to provide expressive methods for representation learning instead of computing marginals or most likely configurations, GNNs provide flexibility in the choice of information flowing rules while maintaining good performance. Despite their success and inspirations, they lack efficient ways to represent and learn higher-order relations among variables/nodes. More expressive higher-order GNNs which operate on k-tuples of nodes need increased computational resources in order to process higher-order tensors. We propose Factor Graph Neural Networks (FGNNs) to effectively capture higher-order relations for inference and learning. To do so, we first derive an efficient approximate Sum-Product loopy belief propagation inference algorithm for discrete higher-order PGMs. We then neuralize the novel message passing scheme into a Factor Graph Neural Network (FGNN) module by allowing richer representations of the message update rules; this facilitates both efficient inference and powerful end-to-end learning. We further show that with a suitable choice of message aggregation operators, our FGNN is also able to represent Max-Product belief propagation, providing a single family of architecture that can represent both Max and Sum-Product loopy belief propagation. Our extensive experimental evaluation on synthetic as well as real datasets demonstrates the potential of the proposed model.

{{</citation>}}


## cs.IT (1)



### (19/74) Optimizing Cellular Networks for UAV Corridors via Quantization Theory (Saeed Karimi-Bidhendi et al., 2023)

{{<citation>}}

Saeed Karimi-Bidhendi, Giovanni Geraci, Hamid Jafarkhani. (2023)  
**Optimizing Cellular Networks for UAV Corridors via Quantization Theory**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2308.01440v1)  

---


**ABSTRACT**  
We present a new framework based on quantization theory to design cellular networks optimized for both legacy ground users and uncrewed aerial vehicle (UAV) corridors, dedicated aerial highways for safe UAV flights. Our framework leverages antenna tilts and transmit power at each base station to enhance coverage and quality of service among users. We develop a comprehensive mathematical analysis and optimization algorithms for multiple system-level performance metrics, including received signal strength and signal-to-interference-plus-noise ratio. Realistic antenna radiation patterns and propagation channel models are considered, alongside a generic 3D user distribution that allows for performance prioritization on the ground, along UAV corridors, or a desired tradeoff between the two. We demonstrate the efficacy of the proposed framework through case studies, showcasing the non-trivial combinations of antenna tilts and power levels that improve coverage and signal quality along UAV corridors while incurring only a marginal impact on the ground user performance compared to scenarios without UAVs.

{{</citation>}}


## cs.AI (5)



### (20/74) Why Do We Need Neuro-symbolic AI to Model Pragmatic Analogies? (Thilini Wijesiriwardene et al., 2023)

{{<citation>}}

Thilini Wijesiriwardene, Amit Sheth, Valerie L. Shalin, Amitava Das. (2023)  
**Why Do We Need Neuro-symbolic AI to Model Pragmatic Analogies?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.01936v1)  

---


**ABSTRACT**  
A hallmark of intelligence is the ability to use a familiar domain to make inferences about a less familiar domain, known as analogical reasoning. In this article, we delve into the performance of Large Language Models (LLMs) in dealing with progressively complex analogies expressed in unstructured text. We discuss analogies at four distinct levels of complexity: lexical analogies, syntactic analogies, semantic analogies, and pragmatic analogies. As the analogies become more complex, they require increasingly extensive, diverse knowledge beyond the textual content, unlikely to be found in the lexical co-occurrence statistics that power LLMs. To address this, we discuss the necessity of employing Neuro-symbolic AI techniques that combine statistical and symbolic AI, informing the representation of unstructured text to highlight and augment relevant content, provide abstraction and guide the mapping process. Our knowledge-informed approach maintains the efficiency of LLMs while preserving the ability to explain analogies for pedagogical applications.

{{</citation>}}


### (21/74) Flows: Building Blocks of Reasoning and Collaborating AI (Martin Josifoski et al., 2023)

{{<citation>}}

Martin Josifoski, Lars Klein, Maxime Peyrard, Yifei Li, Saibo Geng, Julian Paul Schnitzler, Yuxing Yao, Jiheng Wei, Debjit Paul, Robert West. (2023)  
**Flows: Building Blocks of Reasoning and Collaborating AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI, GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.01285v1)  

---


**ABSTRACT**  
Recent advances in artificial intelligence (AI) have produced highly capable and controllable systems. This creates unprecedented opportunities for structured reasoning as well as collaboration among multiple AI systems and humans. To fully realize this potential, it is essential to develop a principled way of designing and studying such structured interactions. For this purpose, we introduce the conceptual framework of Flows: a systematic approach to modeling complex interactions. Flows are self-contained building blocks of computation, with an isolated state, communicating through a standardized message-based interface. This modular design allows Flows to be recursively composed into arbitrarily nested interactions, with a substantial reduction of complexity. Crucially, any interaction can be implemented using this framework, including prior work on AI--AI and human--AI interactions, prompt engineering schemes, and tool augmentation. We demonstrate the potential of Flows on the task of competitive coding, a challenging task on which even GPT-4 struggles. Our results suggest that structured reasoning and collaboration substantially improve generalization, with AI-only Flows adding +$21$ and human--AI Flows adding +$54$ absolute points in terms of solve rate. To support rapid and rigorous research, we introduce the aiFlows library. The library comes with a repository of Flows that can be easily used, extended, and composed into novel, more complex Flows.   The aiFlows library is available at https://github.com/epfl-dlab/aiflows. Data and Flows for reproducing our experiments are available at https://github.com/epfl-dlab/cc_flows.

{{</citation>}}


### (22/74) Exploring the psychology of GPT-4's Moral and Legal Reasoning (Guilherme F. C. F. Almeida et al., 2023)

{{<citation>}}

Guilherme F. C. F. Almeida, José Luiz Nunes, Neele Engelmann, Alex Wiegmann, Marcelo de Araújo. (2023)  
**Exploring the psychology of GPT-4's Moral and Legal Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, GPT, GPT-4, Legal, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.01264v1)  

---


**ABSTRACT**  
Large language models have been used as the foundation of highly sophisticated artificial intelligences, capable of delivering human-like responses to probes about legal and moral issues. However, these models are unreliable guides to their own inner workings, and even the engineering teams behind their creation are unable to explain exactly how they came to develop all of the capabilities they currently have. The emerging field of machine psychology seeks to gain insight into the processes and concepts that these models possess. In this paper, we employ the methods of psychology to probe into GPT-4's moral and legal reasoning. More specifically, we investigate the similarities and differences between GPT-4 and humans when it comes to intentionality ascriptions, judgments about causation, the morality of deception, moral foundations, the impact of moral luck on legal judgments, the concept of consent, and rule violation judgments. We find high correlations between human and AI responses, but also several significant systematic differences between them. We conclude with a discussion of the philosophical implications of our findings.

{{</citation>}}


### (23/74) Arithmetic with Language Models: from Memorization to Computation (Davide Maltoni et al., 2023)

{{<citation>}}

Davide Maltoni, Matteo Ferrara. (2023)  
**Arithmetic with Language Models: from Memorization to Computation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.01154v1)  

---


**ABSTRACT**  
A better understanding of the emergent computation and problem-solving capabilities of recent large language models is of paramount importance to further improve them and broaden their applicability. This work investigates how a language model, trained to predict the next token, can perform arithmetic computations generalizing beyond training data. Binary addition and multiplication constitute a good testbed for this purpose, since they require a very small vocabulary and exhibit relevant input/output discontinuities making smooth input interpolation ineffective for novel data. We successfully trained a light language model to learn these tasks and ran a number of experiments to investigate the extrapolation capabilities and internal information processing. Our findings support the hypotheses that the language model works as an Encoding-Regression-Decoding machine where the computation takes place in the value space once the input token representation is mapped to an appropriate internal representation.

{{</citation>}}


### (24/74) Literal-Aware Knowledge Graph Embedding for Welding Quality Monitoring: A Bosch Case (Zhipeng Tan et al., 2023)

{{<citation>}}

Zhipeng Tan, Baifan Zhou, Zhuoxun Zheng, Ognjen Savkovic, Ziqi Huang, Irlan-Grangel Gonzalez, Ahmet Soylu, Evgeny Kharlamov. (2023)  
**Literal-Aware Knowledge Graph Embedding for Welding Quality Monitoring: A Bosch Case**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.01105v1)  

---


**ABSTRACT**  
Recently there has been a series of studies in knowledge graph embedding (KGE), which attempts to learn the embeddings of the entities and relations as numerical vectors and mathematical mappings via machine learning (ML). However, there has been limited research that applies KGE for industrial problems in manufacturing. This paper investigates whether and to what extent KGE can be used for an important problem: quality monitoring for welding in manufacturing industry, which is an impactful process accounting for production of millions of cars annually. The work is in line with Bosch research of data-driven solutions that intends to replace the traditional way of destroying cars, which is extremely costly and produces waste. The paper tackles two very challenging questions simultaneously: how large the welding spot diameter is; and to which car body the welded spot belongs to. The problem setting is difficult for traditional ML because there exist a high number of car bodies that should be assigned as class labels. We formulate the problem as link prediction, and experimented popular KGE methods on real industry data, with consideration of literals. Our results reveal both limitations and promising aspects of adapted KGE methods.

{{</citation>}}


## cs.CV (24)



### (25/74) Harder synthetic anomalies to improve OoD detection in Medical Images (Sergio Naval Marimont et al., 2023)

{{<citation>}}

Sergio Naval Marimont, Giacomo Tarroni. (2023)  
**Harder synthetic anomalies to improve OoD detection in Medical Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01412v1)  

---


**ABSTRACT**  
Our method builds upon previous Medical Out-of-Distribution (MOOD) challenge winners that empirically show that synthetic local anomalies generated copying / interpolating foreign patches are useful to train segmentation networks able to generalize to unseen types of anomalies. In terms of the synthetic anomaly generation process, our contributions makes synthetic anomalies more heterogeneous and challenging by 1) using random shapes instead of squares and 2) smoothing the interpolation edge of anomalies so networks cannot rely on the high gradient between image - foreign patch to identify anomalies. Our experiments using the validation set of 2020 MOOD winners show that both contributions improved substantially the method performance. We used a standard 3D U-Net architecture as segmentation network, trained patch-wise in both brain and abdominal datasets. Our final challenge submission consisted of 10 U-Nets trained across 5 data folds with different configurations of the anomaly generation process. Our method achieved first position in both sample-wise and pixel-wise tasks in the 2022 edition of the Medical Out-of-Distribution held at MICCAI.

{{</citation>}}


### (26/74) OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models (Anas Awadalla et al., 2023)

{{<citation>}}

Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, Ludwig Schmidt. (2023)  
**OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.01390v1)  

---


**ABSTRACT**  
We introduce OpenFlamingo, a family of autoregressive vision-language models ranging from 3B to 9B parameters. OpenFlamingo is an ongoing effort to produce an open-source replication of DeepMind's Flamingo models. On seven vision-language datasets, OpenFlamingo models average between 80 - 89% of corresponding Flamingo performance. This technical report describes our models, training data, hyperparameters, and evaluation suite. We share our models and code at https://github.com/mlfoundations/open_flamingo.

{{</citation>}}


### (27/74) ELIXR: Towards a general purpose X-ray artificial intelligence system through alignment of large language models and radiology vision encoders (Shawn Xu et al., 2023)

{{<citation>}}

Shawn Xu, Lin Yang, Christopher Kelly, Marcin Sieniek, Timo Kohlberger, Martin Ma, Wei-Hung Weng, Attila Kiraly, Sahar Kazemzadeh, Zakkai Melamed, Jungyeon Park, Patricia Strachan, Yun Liu, Chuck Lau, Preeti Singh, Christina Chen, Mozziyar Etemadi, Sreenivasa Raju Kalidindi, Yossi Matias, Katherine Chou, Greg S. Corrado, Shravya Shetty, Daniel Tse, Shruthi Prabhakara, Daniel Golden, Rory Pilgrim, Krish Eswaran, Andrew Sellergren. (2023)  
**ELIXR: Towards a general purpose X-ray artificial intelligence system through alignment of large language models and radiology vision encoders**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI, Embedding, PaLM  
[Paper Link](http://arxiv.org/abs/2308.01317v1)  

---


**ABSTRACT**  
Our approach, which we call Embeddings for Language/Image-aligned X-Rays, or ELIXR, leverages a language-aligned image encoder combined or grafted onto a fixed LLM, PaLM 2, to perform a broad range of tasks. We train this lightweight adapter architecture using images paired with corresponding free-text radiology reports from the MIMIC-CXR dataset. ELIXR achieved state-of-the-art performance on zero-shot chest X-ray (CXR) classification (mean AUC of 0.850 across 13 findings), data-efficient CXR classification (mean AUCs of 0.893 and 0.898 across five findings (atelectasis, cardiomegaly, consolidation, pleural effusion, and pulmonary edema) for 1% (~2,200 images) and 10% (~22,000 images) training data), and semantic search (0.76 normalized discounted cumulative gain (NDCG) across nineteen queries, including perfect retrieval on twelve of them). Compared to existing data-efficient methods including supervised contrastive learning (SupCon), ELIXR required two orders of magnitude less data to reach similar performance. ELIXR also showed promise on CXR vision-language tasks, demonstrating overall accuracies of 58.7% and 62.5% on visual question answering and report quality assurance tasks, respectively. These results suggest that ELIXR is a robust and versatile approach to CXR AI.

{{</citation>}}


### (28/74) Revisiting DETR Pre-training for Object Detection (Yan Ma et al., 2023)

{{<citation>}}

Yan Ma, Weicong Liang, Yiduo Hao, Bohan Chen, Xiangyu Yue, Chao Zhang, Yuhui Yuan. (2023)  
**Revisiting DETR Pre-training for Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2308.01300v1)  

---


**ABSTRACT**  
Motivated by that DETR-based approaches have established new records on COCO detection and segmentation benchmarks, many recent endeavors show increasing interest in how to further improve DETR-based approaches by pre-training the Transformer in a self-supervised manner while keeping the backbone frozen. Some studies already claimed significant improvements in accuracy. In this paper, we take a closer look at their experimental methodology and check if their approaches are still effective on the very recent state-of-the-art such as $\mathcal{H}$-Deformable-DETR. We conduct thorough experiments on COCO object detection tasks to study the influence of the choice of pre-training datasets, localization, and classification target generation schemes. Unfortunately, we find the previous representative self-supervised approach such as DETReg, fails to boost the performance of the strong DETR-based approaches on full data regimes. We further analyze the reasons and find that simply combining a more accurate box predictor and Objects$365$ benchmark can significantly improve the results in follow-up experiments. We demonstrate the effectiveness of our approach by achieving strong object detection results of AP=$59.3\%$ on COCO val set, which surpasses $\mathcal{H}$-Deformable-DETR + Swin-L by +$1.4\%$. Last, we generate a series of synthetic pre-training datasets by combining the very recent image-to-text captioning models (LLaVA) and text-to-image generative models (SDXL). Notably, pre-training on these synthetic datasets leads to notable improvements in object detection performance. Looking ahead, we anticipate substantial advantages through the future expansion of the synthetic pre-training dataset.

{{</citation>}}


### (29/74) Learning Spatial Distribution of Long-Term Trackers Scores (Vincenzo Mariano Scarrica et al., 2023)

{{<citation>}}

Vincenzo Mariano Scarrica, Antonino Staiano. (2023)  
**Learning Spatial Distribution of Long-Term Trackers Scores**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2308.01256v1)  

---


**ABSTRACT**  
Long-Term tracking is a hot topic in Computer Vision. In this context, competitive models are presented every year, showing a constant growth rate in performances, mainly measured in standardized protocols as Visual Object Tracking (VOT) and Object Tracking Benchmark (OTB). Fusion-trackers strategy has been applied over last few years for overcoming the known re-detection problem, turning out to be an important breakthrough. Following this approach, this work aims to generalize the fusion concept to an arbitrary number of trackers used as baseline trackers in the pipeline, leveraging a learning phase to better understand how outcomes correlate with each other, even when no target is present. A model and data independence conjecture will be evidenced in the manuscript, yielding a recall of 0.738 on LTB-50 dataset when learning from VOT-LT2022, and 0.619 by reversing the two datasets. In both cases, results are strongly competitive with state-of-the-art and recall turns out to be the first on the podium.

{{</citation>}}


### (30/74) A Hyper-pixel-wise Contrastive Learning Augmented Segmentation Network for Old Landslide Detection Using High-Resolution Remote Sensing Images and Digital Elevation Model Data (Yiming Zhou et al., 2023)

{{<citation>}}

Yiming Zhou, Yuexing Peng, Wei Li, Junchuan Yu, Daqing Ge, Wei Xiang. (2023)  
**A Hyper-pixel-wise Contrastive Learning Augmented Segmentation Network for Old Landslide Detection Using High-Resolution Remote Sensing Images and Digital Elevation Model Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.01251v1)  

---


**ABSTRACT**  
As a harzard disaster, landslide often brings tremendous losses to humanity, so it's necessary to achieve reliable detection of landslide. However, the problems of visual blur and small-sized dataset cause great challenges for old landslide detection task when using remote sensing data. To reliably extract semantic features, a hyper-pixel-wise contrastive learning augmented segmentation network (HPCL-Net) is proposed, which augments the local salient feature extraction from the boundaries of landslides through HPCL and fuses the heterogeneous infromation in the semantic space from High-Resolution Remote Sensing Images and Digital Elevation Model Data data. For full utilization of the precious samples, a global hyper-pixel-wise sample pair queues-based contrastive learning method, which includes the construction of global queues that store hyper-pixel-wise samples and the updating scheme of a momentum encoder, is developed, reliably enhancing the extraction ability of semantic features. The proposed HPCL-Net is evaluated on a Loess Plateau old landslide dataset and experiment results show that the model greatly improves the reliablity of old landslide detection compared to the previous old landslide segmentation model, where mIoU metric is increased from 0.620 to 0.651, Landslide IoU metric is increased from 0.334 to 0.394 and F1-score metric is increased from 0.501 to 0.565.

{{</citation>}}


### (31/74) Grounded Image Text Matching with Mismatched Relation Reasoning (Yu Wu et al., 2023)

{{<citation>}}

Yu Wu, Yana Wei, Haozhe Wang, Yongfei Liu, Sibei Yang, Xuming He. (2023)  
**Grounded Image Text Matching with Mismatched Relation Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.01236v2)  

---


**ABSTRACT**  
This paper introduces Grounded Image Text Matching with Mismatched Relation (GITM-MR), a novel visual-linguistic joint task that evaluates the relation understanding capabilities of transformer-based pre-trained models. GITM-MR requires a model to first determine if an expression describes an image, then localize referred objects or ground the mismatched parts of the text. We provide a benchmark for evaluating pre-trained models on this task, with a focus on the challenging settings of limited data and out-of-distribution sentence lengths. Our evaluation demonstrates that pre-trained models lack data efficiency and length generalization ability. To address this, we propose the Relation-sensitive Correspondence Reasoning Network (RCRN), which incorporates relation-aware reasoning via bi-directional message propagation guided by language structure. RCRN can be interpreted as a modular program and delivers strong performance in both length generalization and data efficiency.

{{</citation>}}


### (32/74) TeachCLIP: Multi-Grained Teaching for Efficient Text-to-Video Retrieval (Kaibin Tian et al., 2023)

{{<citation>}}

Kaibin Tian, Ruixiang Zhao, Hu Hu, Runquan Xie, Fengzong Lian, Zhanhui Kang, Xirong Li. (2023)  
**TeachCLIP: Multi-Grained Teaching for Efficient Text-to-Video Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.01217v1)  

---


**ABSTRACT**  
For text-to-video retrieval (T2VR), which aims to retrieve unlabeled videos by ad-hoc textual queries, CLIP-based methods are dominating. Compared to CLIP4Clip which is efficient and compact, the state-of-the-art models tend to compute video-text similarity by fine-grained cross-modal feature interaction and matching, putting their scalability for large-scale T2VR into doubt. For efficient T2VR, we propose TeachCLIP with multi-grained teaching to let a CLIP4Clip based student network learn from more advanced yet computationally heavy models such as X-CLIP, TS2-Net and X-Pool . To improve the student's learning capability, we add an Attentional frame-Feature Aggregation (AFA) block, which by design adds no extra storage/computation overhead at the retrieval stage. While attentive weights produced by AFA are commonly used for combining frame-level features, we propose a novel use of the weights to let them imitate frame-text relevance estimated by the teacher network. As such, AFA provides a fine-grained learning (teaching) channel for the student (teacher). Extensive experiments on multiple public datasets justify the viability of the proposed method.

{{</citation>}}


### (33/74) Improving Generalization in Visual Reinforcement Learning via Conflict-aware Gradient Agreement Augmentation (Siao Liu et al., 2023)

{{<citation>}}

Siao Liu, Zhaoyu Chen, Yang Liu, Yuzheng Wang, Dingkang Yang, Zhile Zhao, Ziqing Zhou, Xie Yi, Wei Li, Wenqiang Zhang, Zhongxue Gan. (2023)  
**Improving Generalization in Visual Reinforcement Learning via Conflict-aware Gradient Agreement Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.01194v1)  

---


**ABSTRACT**  
Learning a policy with great generalization to unseen environments remains challenging but critical in visual reinforcement learning. Despite the success of augmentation combination in the supervised learning generalization, naively applying it to visual RL algorithms may damage the training efficiency, suffering from serve performance degradation. In this paper, we first conduct qualitative analysis and illuminate the main causes: (i) high-variance gradient magnitudes and (ii) gradient conflicts existed in various augmentation methods. To alleviate these issues, we propose a general policy gradient optimization framework, named Conflict-aware Gradient Agreement Augmentation (CG2A), and better integrate augmentation combination into visual RL algorithms to address the generalization bias. In particular, CG2A develops a Gradient Agreement Solver to adaptively balance the varying gradient magnitudes, and introduces a Soft Gradient Surgery strategy to alleviate the gradient conflicts. Extensive experiments demonstrate that CG2A significantly improves the generalization performance and sample efficiency of visual RL algorithms.

{{</citation>}}


### (34/74) Data-Centric Diet: Effective Multi-center Dataset Pruning for Medical Image Segmentation (Yongkang He et al., 2023)

{{<citation>}}

Yongkang He, Mingjin Chen, Zhijing Yang, Yongyi Lu. (2023)  
**Data-Centric Diet: Effective Multi-center Dataset Pruning for Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.01189v1)  

---


**ABSTRACT**  
This paper seeks to address the dense labeling problems where a significant fraction of the dataset can be pruned without sacrificing much accuracy. We observe that, on standard medical image segmentation benchmarks, the loss gradient norm-based metrics of individual training examples applied in image classification fail to identify the important samples. To address this issue, we propose a data pruning method by taking into consideration the training dynamics on target regions using Dynamic Average Dice (DAD) score. To the best of our knowledge, we are among the first to address the data importance in dense labeling tasks in the field of medical image analysis, making the following contributions: (1) investigating the underlying causes with rigorous empirical analysis, and (2) determining effective data pruning approach in dense labeling problems. Our solution can be used as a strong yet simple baseline to select important examples for medical image segmentation with combined data sources.

{{</citation>}}


### (35/74) UCDFormer: Unsupervised Change Detection Using a Transformer-driven Image Translation (Qingsong Xu et al., 2023)

{{<citation>}}

Qingsong Xu, Yilei Shi, Jianhua Guo, Chaojun Ouyang, Xiao Xiang Zhu. (2023)  
**UCDFormer: Unsupervised Change Detection Using a Transformer-driven Image Translation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.01146v1)  

---


**ABSTRACT**  
Change detection (CD) by comparing two bi-temporal images is a crucial task in remote sensing. With the advantages of requiring no cumbersome labeled change information, unsupervised CD has attracted extensive attention in the community. However, existing unsupervised CD approaches rarely consider the seasonal and style differences incurred by the illumination and atmospheric conditions in multi-temporal images. To this end, we propose a change detection with domain shift setting for remote sensing images. Furthermore, we present a novel unsupervised CD method using a light-weight transformer, called UCDFormer. Specifically, a transformer-driven image translation composed of a light-weight transformer and a domain-specific affinity weight is first proposed to mitigate domain shift between two images with real-time efficiency. After image translation, we can generate the difference map between the translated before-event image and the original after-event image. Then, a novel reliable pixel extraction module is proposed to select significantly changed/unchanged pixel positions by fusing the pseudo change maps of fuzzy c-means clustering and adaptive threshold. Finally, a binary change map is obtained based on these selected pixel pairs and a binary classifier. Experimental results on different unsupervised CD tasks with seasonal and style changes demonstrate the effectiveness of the proposed UCDFormer. For example, compared with several other related methods, UCDFormer improves performance on the Kappa coefficient by more than 12\%. In addition, UCDFormer achieves excellent performance for earthquake-induced landslide detection when considering large-scale applications. The code is available at \url{https://github.com/zhu-xlab/UCDFormer}

{{</citation>}}


### (36/74) DiffusePast: Diffusion-based Generative Replay for Class Incremental Semantic Segmentation (Jingfan Chen et al., 2023)

{{<citation>}}

Jingfan Chen, Yuxi Wang, Pengfei Wang, Xiao Chen, Zhaoxiang Zhang, Zhen Lei, Qing Li. (2023)  
**DiffusePast: Diffusion-based Generative Replay for Class Incremental Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.01127v1)  

---


**ABSTRACT**  
The Class Incremental Semantic Segmentation (CISS) extends the traditional segmentation task by incrementally learning newly added classes. Previous work has introduced generative replay, which involves replaying old class samples generated from a pre-trained GAN, to address the issues of catastrophic forgetting and privacy concerns. However, the generated images lack semantic precision and exhibit out-of-distribution characteristics, resulting in inaccurate masks that further degrade the segmentation performance. To tackle these challenges, we propose DiffusePast, a novel framework featuring a diffusion-based generative replay module that generates semantically accurate images with more reliable masks guided by different instructions (e.g., text prompts or edge maps). Specifically, DiffusePast introduces a dual-generator paradigm, which focuses on generating old class images that align with the distribution of downstream datasets while preserving the structure and layout of the original images, enabling more precise masks. To adapt to the novel visual concepts of newly added classes continuously, we incorporate class-wise token embedding when updating the dual-generator. Moreover, we assign adequate pseudo-labels of old classes to the background pixels in the new step images, further mitigating the forgetting of previously learned knowledge. Through comprehensive experiments, our method demonstrates competitive performance across mainstream benchmarks, striking a better balance between the performance of old and novel classes.

{{</citation>}}


### (37/74) Beyond Generic: Enhancing Image Captioning with Real-World Knowledge using Vision-Language Pre-Training Model (Kanzhi Cheng et al., 2023)

{{<citation>}}

Kanzhi Cheng, Wenpo Song, Zheng Ma, Wenhao Zhu, Zixuan Zhu, Jianbing Zhang. (2023)  
**Beyond Generic: Enhancing Image Captioning with Real-World Knowledge using Vision-Language Pre-Training Model**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-MM, cs.CV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2308.01126v1)  

---


**ABSTRACT**  
Current captioning approaches tend to generate correct but "generic" descriptions that lack real-world knowledge, e.g., named entities and contextual information. Considering that Vision-Language Pre-Training (VLP) models master massive such knowledge from large-scale web-harvested data, it is promising to utilize the generalizability of VLP models to incorporate knowledge into image descriptions. However, using VLP models faces challenges: zero-shot inference suffers from knowledge hallucination that leads to low-quality descriptions, but the generic bias in downstream task fine-tuning hinders the VLP model from expressing knowledge. To address these concerns, we propose a simple yet effective method called Knowledge-guided Replay (K-Replay), which enables the retention of pre-training knowledge during fine-tuning. Our approach consists of two parts: (1) a knowledge prediction task on automatically collected replay exemplars to continuously awaken the VLP model's memory about knowledge, thus preventing the model from collapsing into the generic pattern; (2) a knowledge distillation constraint to improve the faithfulness of generated descriptions hence alleviating the knowledge hallucination. To evaluate knowledge-enhanced descriptions, we construct a novel captioning benchmark KnowCap, containing knowledge of landmarks, famous brands, special foods and movie characters. Experimental results show that our approach effectively incorporates knowledge into descriptions, outperforming strong VLP baseline by 20.9 points (78.7->99.6) in CIDEr score and 20.5 percentage points (34.0%->54.5%) in knowledge recognition accuracy. Our code and data is available at https://github.com/njucckevin/KnowCap.

{{</citation>}}


### (38/74) Stereo Visual Odometry with Deep Learning-Based Point and Line Feature Matching using an Attention Graph Neural Network (Shenbagaraj Kannapiran et al., 2023)

{{<citation>}}

Shenbagaraj Kannapiran, Nalin Bendapudi, Ming-Yuan Yu, Devarth Parikh, Spring Berman, Ankit Vora, Gaurav Pandey. (2023)  
**Stereo Visual Odometry with Deep Learning-Based Point and Line Feature Matching using an Attention Graph Neural Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Attention, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.01125v1)  

---


**ABSTRACT**  
Robust feature matching forms the backbone for most Visual Simultaneous Localization and Mapping (vSLAM), visual odometry, 3D reconstruction, and Structure from Motion (SfM) algorithms. However, recovering feature matches from texture-poor scenes is a major challenge and still remains an open area of research. In this paper, we present a Stereo Visual Odometry (StereoVO) technique based on point and line features which uses a novel feature-matching mechanism based on an Attention Graph Neural Network that is designed to perform well even under adverse weather conditions such as fog, haze, rain, and snow, and dynamic lighting conditions such as nighttime illumination and glare scenarios. We perform experiments on multiple real and synthetic datasets to validate the ability of our method to perform StereoVO under low visibility weather and lighting conditions through robust point and line matches. The results demonstrate that our method achieves more line feature matches than state-of-the-art line matching algorithms, which when complemented with point feature matches perform consistently well in adverse weather and dynamic lighting conditions.

{{</citation>}}


### (39/74) Hand tracking for clinical applications: validation of the Google MediaPipe Hand (GMH) and the depth-enhanced GMH-D frameworks (Gianluca Amprimo et al., 2023)

{{<citation>}}

Gianluca Amprimo, Giulia Masi, Giuseppe Pettiti, Gabriella Olmo, Lorenzo Priano, Claudia Ferraris. (2023)  
**Hand tracking for clinical applications: validation of the Google MediaPipe Hand (GMH) and the depth-enhanced GMH-D frameworks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2308.01088v1)  

---


**ABSTRACT**  
Accurate 3D tracking of hand and fingers movements poses significant challenges in computer vision. The potential applications span across multiple domains, including human-computer interaction, virtual reality, industry, and medicine. While gesture recognition has achieved remarkable accuracy, quantifying fine movements remains a hurdle, particularly in clinical applications where the assessment of hand dysfunctions and rehabilitation training outcomes necessitate precise measurements. Several novel and lightweight frameworks based on Deep Learning have emerged to address this issue; however, their performance in accurately and reliably measuring fingers movements requires validation against well-established gold standard systems. In this paper, the aim is to validate the handtracking framework implemented by Google MediaPipe Hand (GMH) and an innovative enhanced version, GMH-D, that exploits the depth estimation of an RGB-Depth camera to achieve more accurate tracking of 3D movements. Three dynamic exercises commonly administered by clinicians to assess hand dysfunctions, namely Hand Opening-Closing, Single Finger Tapping and Multiple Finger Tapping are considered. Results demonstrate high temporal and spectral consistency of both frameworks with the gold standard. However, the enhanced GMH-D framework exhibits superior accuracy in spatial measurements compared to the baseline GMH, for both slow and fast movements. Overall, our study contributes to the advancement of hand tracking technology, the establishment of a validation procedure as a good-practice to prove efficacy of deep-learning-based hand-tracking, and proves the effectiveness of GMH-D as a reliable framework for assessing 3D hand movements in clinical applications.

{{</citation>}}


### (40/74) Homography Estimation in Complex Topological Scenes (Giacomo D'Amicantonio et al., 2023)

{{<citation>}}

Giacomo D'Amicantonio, Egor Bondarau, Peter H. N. De With. (2023)  
**Homography Estimation in Complex Topological Scenes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.01086v1)  

---


**ABSTRACT**  
Surveillance videos and images are used for a broad set of applications, ranging from traffic analysis to crime detection. Extrinsic camera calibration data is important for most analysis applications. However, security cameras are susceptible to environmental conditions and small camera movements, resulting in a need for an automated re-calibration method that can account for these varying conditions. In this paper, we present an automated camera-calibration process leveraging a dictionary-based approach that does not require prior knowledge on any camera settings. The method consists of a custom implementation of a Spatial Transformer Network (STN) and a novel topological loss function. Experiments reveal that the proposed method improves the IoU metric by up to 12% w.r.t. a state-of-the-art model across five synthetic datasets and the World Cup 2014 dataset.

{{</citation>}}


### (41/74) Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation (Quan Tang et al., 2023)

{{<citation>}}

Quan Tang, Bowen Zhang, Jiajun Liu, Fagiu Liu, Yifan Liu. (2023)  
**Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning, Semantic Segmentation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.01045v1)  

---


**ABSTRACT**  
Vision transformers have achieved leading performance on various visual tasks yet still suffer from high computational complexity. The situation deteriorates in dense prediction tasks like semantic segmentation, as high-resolution inputs and outputs usually imply more tokens involved in computations. Directly removing the less attentive tokens has been discussed for the image classification task but can not be extended to semantic segmentation since a dense prediction is required for every patch. To this end, this work introduces a Dynamic Token Pruning (DToP) method based on the early exit of tokens for semantic segmentation. Motivated by the coarse-to-fine segmentation process by humans, we naturally split the widely adopted auxiliary-loss-based network architecture into several stages, where each auxiliary block grades every token's difficulty level. We can finalize the prediction of easy tokens in advance without completing the entire forward pass. Moreover, we keep $k$ highest confidence tokens for each semantic category to uphold the representative context information. Thus, computational complexity will change with the difficulty of the input, akin to the way humans do segmentation. Experiments suggest that the proposed DToP architecture reduces on average $20\% - 35\%$ of computational cost for current semantic segmentation methods based on plain vision transformers without accuracy degradation.

{{</citation>}}


### (42/74) WCCNet: Wavelet-integrated CNN with Crossmodal Rearranging Fusion for Fast Multispectral Pedestrian Detection (Xingjian Wang et al., 2023)

{{<citation>}}

Xingjian Wang, Li Chai, Jiming Chen, Zhiguo Shi. (2023)  
**WCCNet: Wavelet-integrated CNN with Crossmodal Rearranging Fusion for Fast Multispectral Pedestrian Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01042v1)  

---


**ABSTRACT**  
Multispectral pedestrian detection achieves better visibility in challenging conditions and thus has a broad application in various tasks, for which both the accuracy and computational cost are of paramount importance. Most existing approaches treat RGB and infrared modalities equally, typically adopting two symmetrical CNN backbones for multimodal feature extraction, which ignores the substantial differences between modalities and brings great difficulty for the reduction of the computational cost as well as effective crossmodal fusion. In this work, we propose a novel and efficient framework named WCCNet that is able to differentially extract rich features of different spectra with lower computational complexity and semantically rearranges these features for effective crossmodal fusion. Specifically, the discrete wavelet transform (DWT) allowing fast inference and training speed is embedded to construct a dual-stream backbone for efficient feature extraction. The DWT layers of WCCNet extract frequency components for infrared modality, while the CNN layers extract spatial-domain features for RGB modality. This methodology not only significantly reduces the computational complexity, but also improves the extraction of infrared features to facilitate the subsequent crossmodal fusion. Based on the well extracted features, we elaborately design the crossmodal rearranging fusion module (CMRF), which can mitigate spatial misalignment and merge semantically complementary features of spatially-related local regions to amplify the crossmodal complementary information. We conduct comprehensive evaluations on KAIST and FLIR benchmarks, in which WCCNet outperforms state-of-the-art methods with considerable computational efficiency and competitive accuracy. We also perform the ablation study and analyze thoroughly the impact of different components on the performance of WCCNet.

{{</citation>}}


### (43/74) TS-RGBD Dataset: a Novel Dataset for Theatre Scenes Description for People with Visual Impairments (Leyla Benhamida et al., 2023)

{{<citation>}}

Leyla Benhamida, Khadidja Delloul, Slimane Larabi. (2023)  
**TS-RGBD Dataset: a Novel Dataset for Theatre Scenes Description for People with Visual Impairments**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2308.01035v1)  

---


**ABSTRACT**  
Computer vision was long a tool used for aiding visually impaired people to move around their environment and avoid obstacles and falls. Solutions are limited to either indoor or outdoor scenes, which limits the kind of places and scenes visually disabled people can be in, including entertainment places such as theatres. Furthermore, most of the proposed computer-vision-based methods rely on RGB benchmarks to train their models resulting in a limited performance due to the absence of the depth modality.   In this paper, we propose a novel RGB-D dataset containing theatre scenes with ground truth human actions and dense captions annotations for image captioning and human action recognition: TS-RGBD dataset. It includes three types of data: RGB, depth, and skeleton sequences, captured by Microsoft Kinect.   We test image captioning models on our dataset as well as some skeleton-based human action recognition models in order to extend the range of environment types where a visually disabled person can be, by detecting human actions and textually describing appearances of regions of interest in theatre scenes.

{{</citation>}}


### (44/74) MDT3D: Multi-Dataset Training for LiDAR 3D Object Detection Generalization (Louis Soum-Fontez et al., 2023)

{{<citation>}}

Louis Soum-Fontez, Jean-Emmanuel Deschaud, François Goulette. (2023)  
**MDT3D: Multi-Dataset Training for LiDAR 3D Object Detection Generalization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.01000v1)  

---


**ABSTRACT**  
Supervised 3D Object Detection models have been displaying increasingly better performance in single-domain cases where the training data comes from the same environment and sensor as the testing data. However, in real-world scenarios data from the target domain may not be available for finetuning or for domain adaptation methods. Indeed, 3D object detection models trained on a source dataset with a specific point distribution have shown difficulties in generalizing to unseen datasets. Therefore, we decided to leverage the information available from several annotated source datasets with our Multi-Dataset Training for 3D Object Detection (MDT3D) method to increase the robustness of 3D object detection models when tested in a new environment with a different sensor configuration. To tackle the labelling gap between datasets, we used a new label mapping based on coarse labels. Furthermore, we show how we managed the mix of datasets during training and finally introduce a new cross-dataset augmentation method: cross-dataset object injection. We demonstrate that this training paradigm shows improvements for different types of 3D object detection models. The source code and additional results for this research project will be publicly available on GitHub for interested parties to access and utilize: https://github.com/LouisSF/MDT3D

{{</citation>}}


### (45/74) Exploiting Synthetic Data for Data Imbalance Problems: Baselines from a Data Perspective (Moon Ye-Bin et al., 2023)

{{<citation>}}

Moon Ye-Bin, Nam Hyeon-Woo, Wonseok Choi, Nayeong Kim, Suha Kwak, Tae-Hyun Oh. (2023)  
**Exploiting Synthetic Data for Data Imbalance Problems: Baselines from a Data Perspective**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.00994v1)  

---


**ABSTRACT**  
We live in a vast ocean of data, and deep neural networks are no exception to this. However, this data exhibits an inherent phenomenon of imbalance. This imbalance poses a risk of deep neural networks producing biased predictions, leading to potentially severe ethical and social consequences. To address these challenges, we believe that the use of generative models is a promising approach for comprehending tasks, given the remarkable advancements demonstrated by recent diffusion models in generating high-quality images. In this work, we propose a simple yet effective baseline, SYNAuG, that utilizes synthetic data as a preliminary step before employing task-specific algorithms to address data imbalance problems. This straightforward approach yields impressive performance on datasets such as CIFAR100-LT, ImageNet100-LT, UTKFace, and Waterbird, surpassing the performance of existing task-specific methods. While we do not claim that our approach serves as a complete solution to the problem of data imbalance, we argue that supplementing the existing data with synthetic data proves to be an effective and crucial preliminary step in addressing data imbalance concerns.

{{</citation>}}


### (46/74) Orientation-Guided Contrastive Learning for UAV-View Geo-Localisation (Fabian Deuser et al., 2023)

{{<citation>}}

Fabian Deuser, Konrad Habel, Martin Werner, Norbert Oswald. (2023)  
**Orientation-Guided Contrastive Learning for UAV-View Geo-Localisation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.00982v1)  

---


**ABSTRACT**  
Retrieving relevant multimedia content is one of the main problems in a world that is increasingly data-driven. With the proliferation of drones, high quality aerial footage is now available to a wide audience for the first time. Integrating this footage into applications can enable GPS-less geo-localisation or location correction.   In this paper, we present an orientation-guided training framework for UAV-view geo-localisation. Through hierarchical localisation orientations of the UAV images are estimated in relation to the satellite imagery. We propose a lightweight prediction module for these pseudo labels which predicts the orientation between the different views based on the contrastive learned embeddings. We experimentally demonstrate that this prediction supports the training and outperforms previous approaches. The extracted pseudo-labels also enable aligned rotation of the satellite image as augmentation to further strengthen the generalisation. During inference, we no longer need this orientation module, which means that no additional computations are required. We achieve state-of-the-art results on both the University-1652 and University-160k datasets.

{{</citation>}}


### (47/74) Training-Free Instance Segmentation from Semantic Image Segmentation Masks (Yuchen Shen et al., 2023)

{{<citation>}}

Yuchen Shen, Dong Zhang, Yuhui Zheng, Zechao Li, Liyong Fu, Qiaolin Ye. (2023)  
**Training-Free Instance Segmentation from Semantic Image Segmentation Masks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.00949v1)  

---


**ABSTRACT**  
In recent years, the development of instance segmentation has garnered significant attention in a wide range of applications. However, the training of a fully-supervised instance segmentation model requires costly both instance-level and pixel-level annotations. In contrast, weakly-supervised instance segmentation methods (i.e., with image-level class labels or point labels) struggle to satisfy the accuracy and recall requirements of practical scenarios. In this paper, we propose a novel paradigm for instance segmentation called training-free instance segmentation (TFISeg), which achieves instance segmentation results from image masks predicted using off-the-shelf semantic segmentation models. TFISeg does not require training a semantic or/and instance segmentation model and avoids the need for instance-level image annotations. Therefore, it is highly efficient. Specifically, we first obtain a semantic segmentation mask of the input image via a trained semantic segmentation model. Then, we calculate a displacement field vector for each pixel based on the segmentation mask, which can indicate representations belonging to the same class but different instances, i.e., obtaining the instance-level object information. Finally, instance segmentation results are obtained after being refined by a learnable category-agnostic object boundary branch. Extensive experimental results on two challenging datasets and representative semantic segmentation baselines (including CNNs and Transformers) demonstrate that TFISeg can achieve competitive results compared to the state-of-the-art fully-supervised instance segmentation methods without the need for additional human resources or increased computational costs. The code is available at: TFISeg

{{</citation>}}


### (48/74) Towards Discriminative Representation with Meta-learning for Colonoscopic Polyp Re-Identification (Suncheng Xiang et al., 2023)

{{<citation>}}

Suncheng Xiang, Qingzhong Chen, Shilun Cai, Chengfeng Zhou, Crystal Cai, Sijia Du, Dahong Qian. (2023)  
**Towards Discriminative Representation with Meta-learning for Colonoscopic Polyp Re-Identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.00929v1)  

---


**ABSTRACT**  
Colonoscopic Polyp Re-Identification aims to match the same polyp from a large gallery with images from different views taken using different cameras and plays an important role in the prevention and treatment of colorectal cancer in computer-aided diagnosis. However, traditional methods for object ReID directly adopting CNN models trained on the ImageNet dataset usually produce unsatisfactory retrieval performance on colonoscopic datasets due to the large domain gap. Additionally, these methods neglect to explore the potential of self-discrepancy among intra-class relations in the colonoscopic polyp dataset, which remains an open research problem in the medical community. To solve this dilemma, we propose a simple but effective training method named Colo-ReID, which can help our model to learn more general and discriminative knowledge based on the meta-learning strategy in scenarios with fewer samples. Based on this, a dynamic Meta-Learning Regulation mechanism called MLR is introduced to further boost the performance of polyp re-identification. To the best of our knowledge, this is the first attempt to leverage the meta-learning paradigm instead of traditional machine learning to effectively train deep models in the task of colonoscopic polyp re-identification. Empirical results show that our method significantly outperforms current state-of-the-art methods by a clear margin.

{{</citation>}}


## cs.CL (10)



### (49/74) UPB at IberLEF-2023 AuTexTification: Detection of Machine-Generated Text using Transformer Ensembles (Andrei-Alexandru Preda et al., 2023)

{{<citation>}}

Andrei-Alexandru Preda, Dumitru-Clementin Cercel, Traian Rebedea, Costin-Gabriel Chiru. (2023)  
**UPB at IberLEF-2023 AuTexTification: Detection of Machine-Generated Text using Transformer Ensembles**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.01408v1)  

---


**ABSTRACT**  
This paper describes the solutions submitted by the UPB team to the AuTexTification shared task, featured as part of IberLEF-2023. Our team participated in the first subtask, identifying text documents produced by large language models instead of humans. The organizers provided a bilingual dataset for this subtask, comprising English and Spanish texts covering multiple domains, such as legal texts, social media posts, and how-to articles. We experimented mostly with deep learning models based on Transformers, as well as training techniques such as multi-task learning and virtual adversarial training to obtain better results. We submitted three runs, two of which consisted of ensemble models. Our best-performing model achieved macro F1-scores of 66.63% on the English dataset and 67.10% on the Spanish dataset.

{{</citation>}}


### (50/74) Optimizing Machine Translation through Prompt Engineering: An Investigation into ChatGPT's Customizability (Masaru Yamada, 2023)

{{<citation>}}

Masaru Yamada. (2023)  
**Optimizing Machine Translation through Prompt Engineering: An Investigation into ChatGPT's Customizability**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Machine Translation  
[Paper Link](http://arxiv.org/abs/2308.01391v1)  

---


**ABSTRACT**  
This paper explores the influence of integrating the purpose of the translation and the target audience into prompts on the quality of translations produced by ChatGPT. Drawing on previous translation studies, industry practices, and ISO standards, the research underscores the significance of the pre-production phase in the translation process. The study reveals that the inclusion of suitable prompts in large-scale language models like ChatGPT can yield flexible translations, a feat yet to be realized by conventional Machine Translation (MT). The research scrutinizes the changes in translation quality when prompts are used to generate translations that meet specific conditions. The evaluation is conducted from a practicing translator's viewpoint, both subjectively and qualitatively, supplemented by the use of OpenAI's word embedding API for cosine similarity calculations. The findings suggest that the integration of the purpose and target audience into prompts can indeed modify the generated translations, generally enhancing the translation quality by industry standards. The study also demonstrates the practical application of the "good translation" concept, particularly in the context of marketing documents and culturally dependent idioms.

{{</citation>}}


### (51/74) Empirical Translation Process Research: Past and Possible Future Perspectives (Michael Carl, 2023)

{{<citation>}}

Michael Carl. (2023)  
**Empirical Translation Process Research: Past and Possible Future Perspectives**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IT, cs.CL, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.01368v1)  

---


**ABSTRACT**  
Over the past four decades, efforts have been made to develop and evaluate models for Empirical Translation Process Research (TPR), yet a comprehensive framework remains elusive. This article traces the evolution of empirical TPR within the CRITT TPR-DB tradition and proposes the Free Energy Principle (FEP) and Active Inference (AIF) as a framework for modeling deeply embedded translation processes. It introduces novel approaches for quantifying fundamental concepts of Relevance Theory (relevance, s-mode, i-mode), and establishes their relation to the Monitor Model, framing relevance maximization as a special case of minimizing free energy. FEP/AIF provides a mathematically rigorous foundation that enables modeling of deep temporal architectures in which embedded translation processes unfold on different timelines. This framework opens up exciting prospects for future research in predictive TPR, likely to enrich our comprehension of human translation processes, and making valuable contributions to the wider realm of translation studies and the design of cognitive architectures.

{{</citation>}}


### (52/74) Fighting Fire with Fire: Can ChatGPT Detect AI-generated Text? (Amrita Bhattacharjee et al., 2023)

{{<citation>}}

Amrita Bhattacharjee, Huan Liu. (2023)  
**Fighting Fire with Fire: Can ChatGPT Detect AI-generated Text?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.01284v1)  

---


**ABSTRACT**  
Large language models (LLMs) such as ChatGPT are increasingly being used for various use cases, including text content generation at scale. Although detection methods for such AI-generated text exist already, we investigate ChatGPT's performance as a detector on such AI-generated text, inspired by works that use ChatGPT as a data labeler or annotator. We evaluate the zero-shot performance of ChatGPT in the task of human-written vs. AI-generated text detection, and perform experiments on publicly available datasets. We empirically investigate if ChatGPT is symmetrically effective in detecting AI-generated or human-written text. Our findings provide insight on how ChatGPT and similar LLMs may be leveraged in automated detection pipelines by simply focusing on solving a specific aspect of the problem and deriving the rest from that solution. All code and data is available at \url{https://github.com/AmritaBh/ChatGPT-as-Detector}.

{{</citation>}}


### (53/74) XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models (Paul Röttger et al., 2023)

{{<citation>}}

Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, Dirk Hovy. (2023)  
**XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.01263v1)  

---


**ABSTRACT**  
Without proper safeguards, large language models will readily follow malicious instructions and generate toxic content. This motivates safety efforts such as red-teaming and large-scale feedback learning, which aim to make models both helpful and harmless. However, there is a tension between these two objectives, since harmlessness requires models to refuse complying with unsafe prompts, and thus not be helpful. Recent anecdotal evidence suggests that some models may have struck a poor balance, so that even clearly safe prompts are refused if they use similar language to unsafe prompts or mention sensitive topics. In this paper, we introduce a new test suite called XSTest to identify such eXaggerated Safety behaviours in a structured and systematic way. In its current form, XSTest comprises 200 safe prompts across ten prompt types that well-calibrated models should not refuse to comply with. We describe XSTest's creation and composition, and use the test suite to highlight systematic failure modes in a recently-released state-of-the-art language model.

{{</citation>}}


### (54/74) Evaluating Instruction-Tuned Large Language Models on Code Comprehension and Generation (Zhiqiang Yuan et al., 2023)

{{<citation>}}

Zhiqiang Yuan, Junwei Liu, Qiancheng Zi, Mingwei Liu, Xin Peng, Yiling Lou. (2023)  
**Evaluating Instruction-Tuned Large Language Models on Code Comprehension and Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.01240v1)  

---


**ABSTRACT**  
In this work, we evaluate 10 open-source instructed LLMs on four representative code comprehension and generation tasks. We have the following main findings. First, for the zero-shot setting, instructed LLMs are very competitive on code comprehension and generation tasks and sometimes even better than small SOTA models specifically fine-tuned on each downstream task. We also find that larger instructed LLMs are not always better on code-related tasks. Second, for the few-shot setting, we find that adding demonstration examples substantially helps instructed LLMs perform better on most code comprehension and generation tasks; however, the examples would sometimes induce unstable or even worse performance. Furthermore, we find widely-used BM25-based shot selection strategy significantly outperforms the basic random selection or fixed selection only on generation problems. Third, for the fine-tuning setting, we find that fine-tuning could further improve the model performance on downstream code comprehension and generation tasks compared to the zero-shot/one-shot performance. In addition, after being fine-tuned on the same downstream task dataset, instructed LLMs outperform both the small SOTA models and similar-scaled LLMs without instruction tuning. Based on our findings, we further present practical implications on model and usage recommendation, performance and cost trade-offs, and future direction.

{{</citation>}}


### (55/74) Do Multilingual Language Models Think Better in English? (Julen Etxaniz et al., 2023)

{{<citation>}}

Julen Etxaniz, Gorka Azkune, Aitor Soroa, Oier Lopez de Lacalle, Mikel Artetxe. (2023)  
**Do Multilingual Language Models Think Better in English?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2308.01223v1)  

---


**ABSTRACT**  
Translate-test is a popular technique to improve the performance of multilingual language models. This approach works by translating the input into English using an external machine translation system, and running inference over the translated input. However, these improvements can be attributed to the use of a separate translation system, which is typically trained on large amounts of parallel data not seen by the language model. In this work, we introduce a new approach called self-translate, which overcomes the need of an external translation system by leveraging the few-shot translation capabilities of multilingual language models. Experiments over 5 tasks show that self-translate consistently outperforms direct inference, demonstrating that language models are unable to leverage their full multilingual potential when prompted in non-English languages. Our code is available at https://github.com/juletx/self-translate.

{{</citation>}}


### (56/74) Leveraging Few-Shot Data Augmentation and Waterfall Prompting for Response Generation (Lea Krause et al., 2023)

{{<citation>}}

Lea Krause, Selene Báez Santamaría, Michiel van der Meer, Urja Khurana. (2023)  
**Leveraging Few-Shot Data Augmentation and Waterfall Prompting for Response Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, ChatGPT, Few-Shot, GPT  
[Paper Link](http://arxiv.org/abs/2308.01080v1)  

---


**ABSTRACT**  
This paper discusses our approaches for task-oriented conversational modelling using subjective knowledge, with a particular emphasis on response generation. Our methodology was shaped by an extensive data analysis that evaluated key factors such as response length, sentiment, and dialogue acts present in the provided dataset. We used few-shot learning to augment the data with newly generated subjective knowledge items and present three approaches for DSTC11: (1) task-specific model exploration, (2) incorporation of the most frequent question into all generated responses, and (3) a waterfall prompting technique using a combination of both GPT-3 and ChatGPT.

{{</citation>}}


### (57/74) SALTTS: Leveraging Self-Supervised Speech Representations for improved Text-to-Speech Synthesis (Ramanan Sivaguru et al., 2023)

{{<citation>}}

Ramanan Sivaguru, Vasista Sai Lodagala, S Umesh. (2023)  
**SALTTS: Leveraging Self-Supervised Speech Representations for improved Text-to-Speech Synthesis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.01018v1)  

---


**ABSTRACT**  
While FastSpeech2 aims to integrate aspects of speech such as pitch, energy, and duration as conditional inputs, it still leaves scope for richer representations. As a part of this work, we leverage representations from various Self-Supervised Learning (SSL) models to enhance the quality of the synthesized speech. In particular, we pass the FastSpeech2 encoder's length-regulated outputs through a series of encoder layers with the objective of reconstructing the SSL representations. In the SALTTS-parallel implementation, the representations from this second encoder are used for an auxiliary reconstruction loss with the SSL features. The SALTTS-cascade implementation, however, passes these representations through the decoder in addition to having the reconstruction loss. The richness of speech characteristics from the SSL features reflects in the output speech quality, with the objective and subjective evaluation measures of the proposed approach outperforming the baseline FastSpeech2.

{{</citation>}}


### (58/74) Teaching Smaller Language Models To Generalise To Unseen Compositional Questions (Tim Hartill et al., 2023)

{{<citation>}}

Tim Hartill, Neset TAN, Michael Witbrock, Patricia J. Riddle. (2023)  
**Teaching Smaller Language Models To Generalise To Unseen Compositional Questions**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2308.00946v1)  

---


**ABSTRACT**  
We equip a smaller Language Model to generalise to answering challenging compositional questions that have not been seen in training. To do so we propose a combination of multitask supervised pretraining on up to 93 tasks designed to instill diverse reasoning abilities, and a dense retrieval system that aims to retrieve a set of evidential paragraph fragments. Recent progress in question-answering has been achieved either through prompting methods against very large pretrained Language Models in zero or few-shot fashion, or by fine-tuning smaller models, sometimes in conjunction with information retrieval. We focus on the less explored question of the extent to which zero-shot generalisation can be enabled in smaller models with retrieval against a corpus within which sufficient information to answer a particular question may not exist. We establish strong baselines in this setting for diverse evaluation datasets (StrategyQA, CommonsenseQA, IIRC, DROP, Musique and ARC-DA), and show that performance can be significantly improved by adding retrieval-augmented training datasets which are designed to expose our models to a variety of heuristic reasoning strategies such as weighing partial evidence or ignoring an irrelevant context.

{{</citation>}}


## cs.RO (4)



### (59/74) A Small Form Factor Aerial Research Vehicle for Pick-and-Place Tasks with Onboard Real-Time Object Detection and Visual Odometry (Cora A. Dimmig et al., 2023)

{{<citation>}}

Cora A. Dimmig, Anna Goodridge, Gabriel Baraban, Pupei Zhu, Joyraj Bhowmick, Marin Kobilarov. (2023)  
**A Small Form Factor Aerial Research Vehicle for Pick-and-Place Tasks with Onboard Real-Time Object Detection and Visual Odometry**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.01398v1)  

---


**ABSTRACT**  
This paper introduces a novel, small form-factor, aerial vehicle research platform for agile object detection, classification, tracking, and interaction tasks. General-purpose hardware components were designed to augment a given aerial vehicle and enable it to perform safe and reliable grasping. These components include a custom collision tolerant cage and low-cost Gripper Extension Package, which we call GREP, for object grasping. Small vehicles enable applications in highly constrained environments, but are often limited by computational resources. This work evaluates the challenges of pick-and-place tasks, with entirely onboard computation of object pose and visual odometry based state estimation on a small platform, and demonstrates experiments with enough accuracy to reliably grasp objects. In a total of 70 trials across challenging cases such as cluttered environments, obstructed targets, and multiple instances of the same target, we demonstrated successfully grasping the target in 93% of trials. Both the hardware component designs and software framework are released as open-source, since our intention is to enable easy reproduction and application on a wide range of small vehicles.

{{</citation>}}


### (60/74) Follow the Soldiers with Optimized Single-Shot Multibox Detection and Reinforcement Learning (Jumman Hossain et al., 2023)

{{<citation>}}

Jumman Hossain, Maliha Momtaz. (2023)  
**Follow the Soldiers with Optimized Single-Shot Multibox Detection and Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.01389v1)  

---


**ABSTRACT**  
Nowadays, autonomous cars are gaining traction due to their numerous potential applications on battlefields and in resolving a variety of other real-world challenges. The main goal of our project is to build an autonomous system using DeepRacer which will follow a specific person (for our project, a soldier) when they will be moving in any direction. Two main components to accomplish this project is an optimized Single-Shot Multibox Detection (SSD) object detection model and a Reinforcement Learning (RL) model. We accomplished the task using SSD Lite instead of SSD and at the end, compared the results among SSD, SSD with Neural Computing Stick (NCS), and SSD Lite. Experimental results show that SSD Lite gives better performance among these three techniques and exhibits a considerable boost in inference speed (~2-3 times) without compromising accuracy.

{{</citation>}}


### (61/74) Ethical Decision-making for Autonomous Driving based on LSTM Trajectory Prediction Network (Wen Wei et al., 2023)

{{<citation>}}

Wen Wei, Jiankun Wang. (2023)  
**Ethical Decision-making for Autonomous Driving based on LSTM Trajectory Prediction Network**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2308.01022v1)  

---


**ABSTRACT**  
The development of autonomous vehicles has brought a great impact and changes to the transportation industry, offering numerous benefits in terms of safety and efficiency. However, one of the key challenges that autonomous driving faces is how to make ethical decisions in complex situations. To address this issue, in this article, a novel trajectory prediction method is proposed to achieve ethical decision-making for autonomous driving. Ethical considerations are integrated into the decision-making process of autonomous vehicles by quantifying the utility principle and incorporating them into mathematical formulas. Furthermore, trajectory prediction is optimized using LSTM network with an attention module, resulting in improved accuracy and reliability in trajectory planning and selection. Through extensive simulation experiments, we demonstrate the effectiveness of the proposed method in making ethical decisions and selecting optimal trajectories.

{{</citation>}}


### (62/74) Grasp Stability Assessment Through Attention-Guided Cross-Modality Fusion and Transfer Learning (Zhuangzhuang Zhang et al., 2023)

{{<citation>}}

Zhuangzhuang Zhang, Zhenning Zhou, Haili Wang, Zhinan Zhang, Huang Huang, Qixin Cao. (2023)  
**Grasp Stability Assessment Through Attention-Guided Cross-Modality Fusion and Transfer Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.00980v1)  

---


**ABSTRACT**  
Extensive research has been conducted on assessing grasp stability, a crucial prerequisite for achieving optimal grasping strategies, including the minimum force grasping policy. However, existing works employ basic feature-level fusion techniques to combine visual and tactile modalities, resulting in the inadequate utilization of complementary information and the inability to model interactions between unimodal features. This work proposes an attention-guided cross-modality fusion architecture to comprehensively integrate visual and tactile features. This model mainly comprises convolutional neural networks (CNNs), self-attention, and cross-attention mechanisms. In addition, most existing methods collect datasets from real-world systems, which is time-consuming and high-cost, and the datasets collected are comparatively limited in size. This work establishes a robotic grasping system through physics simulation to collect a multimodal dataset. To address the sim-to-real transfer gap, we propose a migration strategy encompassing domain randomization and domain adaptation techniques. The experimental results demonstrate that the proposed fusion framework achieves markedly enhanced prediction performance (approximately 10%) compared to other baselines. Moreover, our findings suggest that the trained model can be reliably transferred to real robotic systems, indicating its potential to address real-world challenges.

{{</citation>}}


## cs.SE (2)



### (63/74) Manual Tests Do Smell! Cataloging and Identifying Natural Language Test Smells (Elvys Soares et al., 2023)

{{<citation>}}

Elvys Soares, Manoel Aranda, Naelson Oliveira, Márcio Ribeiro, Rohit Gheyi, Emerson Souza, Ivan Machado, André Santos, Baldoino Fonseca, Rodrigo Bonifácio. (2023)  
**Manual Tests Do Smell! Cataloging and Identifying Natural Language Test Smells**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2308.01386v1)  

---


**ABSTRACT**  
Background: Test smells indicate potential problems in the design and implementation of automated software tests that may negatively impact test code maintainability, coverage, and reliability. When poorly described, manual tests written in natural language may suffer from related problems, which enable their analysis from the point of view of test smells. Despite the possible prejudice to manually tested software products, little is known about test smells in manual tests, which results in many open questions regarding their types, frequency, and harm to tests written in natural language. Aims: Therefore, this study aims to contribute to a catalog of test smells for manual tests. Method: We perform a two-fold empirical strategy. First, an exploratory study in manual tests of three systems: the Ubuntu Operational System, the Brazilian Electronic Voting Machine, and the User Interface of a large smartphone manufacturer. We use our findings to propose a catalog of eight test smells and identification rules based on syntactical and morphological text analysis, validating our catalog with 24 in-company test engineers. Second, using our proposals, we create a tool based on Natural Language Processing (NLP) to analyze the subject systems' tests, validating the results. Results: We observed the occurrence of eight test smells. A survey of 24 in-company test professionals showed that 80.7% agreed with our catalog definitions and examples. Our NLP-based tool achieved a precision of 92%, recall of 95%, and f-measure of 93.5%, and its execution evidenced 13,169 occurrences of our cataloged test smells in the analyzed systems. Conclusion: We contribute with a catalog of natural language test smells and novel detection strategies that better explore the capabilities of current NLP mechanisms with promising results and reduced effort to analyze tests written in different idioms.

{{</citation>}}


### (64/74) Towards Understanding the Capability of Large Language Models on Code Clone Detection: A Survey (Shihan Dou et al., 2023)

{{<citation>}}

Shihan Dou, Junjie Shan, Haoxiang Jia, Wenhao Deng, Zhiheng Xi, Wei He, Yueming Wu, Tao Gui, Yang Liu, Xuanjing Huang. (2023)  
**Towards Understanding the Capability of Large Language Models on Code Clone Detection: A Survey**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.01191v2)  

---


**ABSTRACT**  
Code cloning, the duplication of code fragments, is common in software development. While some reuse aids productivity, excessive cloning hurts maintainability and introduces bugs. Hence, automatic code clone detection is vital. Meanwhile, large language models (LLMs) possess diverse code-related knowledge, making them versatile for various software engineering challenges. However, LLMs' performance in code clone detection is unclear and needs more study for accurate assessment. In this paper, we provide the first comprehensive evaluation of LLMs for clone detection, covering different clone types, languages, and prompts. We find advanced LLMs excel in detecting complex semantic clones, surpassing existing methods. Adding intermediate reasoning steps via chain-of-thought prompts noticeably enhances performance. Additionally, representing code as vector embeddings, especially with text encoders, effectively aids clone detection.Lastly, the ability of LLMs to detect code clones differs among various programming languages. Our study suggests that LLMs have potential for clone detection due to their language capabilities, offering insights for developing robust LLM-based methods to enhance software engineering.

{{</citation>}}


## cs.CR (5)



### (65/74) BRNES: Enabling Security and Privacy-aware Experience Sharing in Multiagent Robotic and Autonomous Systems (Md Tamjid Hossain et al., 2023)

{{<citation>}}

Md Tamjid Hossain, Hung Manh La, Shahriar Badsha, Anton Netchaev. (2023)  
**BRNES: Enabling Security and Privacy-aware Experience Sharing in Multiagent Robotic and Autonomous Systems**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs-MA, cs-RO, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.01274v1)  

---


**ABSTRACT**  
Although experience sharing (ES) accelerates multiagent reinforcement learning (MARL) in an advisor-advisee framework, attempts to apply ES to decentralized multiagent systems have so far relied on trusted environments and overlooked the possibility of adversarial manipulation and inference. Nevertheless, in a real-world setting, some Byzantine attackers, disguised as advisors, may provide false advice to the advisee and catastrophically degrade the overall learning performance. Also, an inference attacker, disguised as an advisee, may conduct several queries to infer the advisors' private information and make the entire ES process questionable in terms of privacy leakage. To address and tackle these issues, we propose a novel MARL framework (BRNES) that heuristically selects a dynamic neighbor zone for each advisee at each learning step and adopts a weighted experience aggregation technique to reduce Byzantine attack impact. Furthermore, to keep the agent's private information safe from adversarial inference attacks, we leverage the local differential privacy (LDP)-induced noise during the ES process. Our experiments show that our framework outperforms the state-of-the-art in terms of the steps to goal, obtained reward, and time to goal metrics. Particularly, our evaluation shows that the proposed framework is 8.32x faster than the current non-private frameworks and 1.41x faster than the private frameworks in an adversarial setting.

{{</citation>}}


### (66/74) LSF-IDM: Lightweight Deep Learning Models for Automotive Intrusion Detection Model Based on Semantic Fusion (Pengzhou Cheng et al., 2023)

{{<citation>}}

Pengzhou Cheng, Lei Hua, Haobin Jiang, Mohammad Samie, Gongshen Liu. (2023)  
**LSF-IDM: Lightweight Deep Learning Models for Automotive Intrusion Detection Model Based on Semantic Fusion**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: BERT, Intrusion Detection, LSTM  
[Paper Link](http://arxiv.org/abs/2308.01237v1)  

---


**ABSTRACT**  
Autonomous vehicles (AVs) are more vulnerable to network attacks due to the high connectivity and diverse communication modes between vehicles and external networks. Deep learning-based Intrusion detection, an effective method for detecting network attacks, can provide functional safety as well as a real-time communication guarantee for vehicles, thereby being widely used for AVs. Existing works well for cyber-attacks such as simple-mode but become a higher false alarm with a resource-limited environment required when the attack is concealed within a contextual feature. In this paper, we present a lightweight intrusion detection model based on semantic fusion, named LSF-IDM. Our motivation is based on the observation that, when injected the malicious packets to the in-vehicle networks (IVNs), the packet log presents a strict order of context feature because of the periodicity and broadcast nature of the CAN bus. Therefore, this model first captures the context as the semantic feature of messages by the BERT language framework. Thereafter, the lightweight model (e.g., BiLSTM) learns the fused feature from an input packet's classification and its output distribution in BERT based on knowledge distillation. Experiment results demonstrate the effectiveness of our methods in defending against several representative attacks from IVNs. We also perform the difference analysis of the proposed method with lightweight models and Bert to attain a deeper understanding of how the model balance detection performance and model complexity.

{{</citation>}}


### (67/74) An Adaptable Approach for Successful SIEM Adoption in Companies (Maximilian Rosenberg et al., 2023)

{{<citation>}}

Maximilian Rosenberg, Bettina Schneider, Christopher Scherb, Petra Maria Asprion. (2023)  
**An Adaptable Approach for Successful SIEM Adoption in Companies**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CY, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.01065v1)  

---


**ABSTRACT**  
In corporations around the world, the topic of cybersecurity and information security is becoming increasingly important as the number of cyberattacks on themselves continues to grow. Nowadays, it is no longer just a matter of protecting against cyberattacks, but rather of detecting such attacks at an early stage and responding accordingly. There is currently no generic methodological approach for the implementation of Security Information and Event Management (SIEM) systems that takes academic aspects into account and can be applied independently of the product or developers of the systems. Applying Hevner's design science research approach, the goal of this paper is to develop a holistic procedure model for implementing respective SIEM systems in corporations. According to the study during the validation phase, the procedure model was verified to be applicable. As desire for future research, the procedure model should be applied in various implementation projects in different enterprises to analyze its applicability and completeness.

{{</citation>}}


### (68/74) Evaluate and Guard the Wisdom of Crowds: Zero Knowledge Proofs for Crowdsourcing Truth Inference (Xuanming Liu et al., 2023)

{{<citation>}}

Xuanming Liu, Xinpeng Yang, Xun Zhang, Xiaohu Yang. (2023)  
**Evaluate and Guard the Wisdom of Crowds: Zero Knowledge Proofs for Crowdsourcing Truth Inference**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00985v1)  

---


**ABSTRACT**  
Due to the risks of correctness and security in outsourced cloud computing, we consider a new paradigm called crowdsourcing: distribute tasks, receive answers and aggregate the results from multiple entities. Through this approach, we can aggregate the wisdom of the crowd to complete tasks, ensuring the accuracy of task completion while reducing the risks posed by the malicious acts of a single entity. However, the ensuing question is, how can we ensure that the aggregator has done its work honestly and each contributor's work has been evaluated fairly?   In this paper, we propose a new scheme called $\mathsf{zkTI}$. This scheme ensures that the aggregator has honestly completed the aggregation and each data source is fairly evaluated. We combine a cryptographic primitive called \textit{zero-knowledge proof} with a class of \textit{truth inference algorithms} which is widely studied in AI/ML scenarios. Under this scheme, various complex outsourced tasks can be solved with efficiency and accuracy. To build our scheme, a novel method to prove the precise computation of floating-point numbers is proposed, which is nearly optimal and well-compatible with existing argument systems. This may become an independent point of interest. Thus our work can prove the process of aggregation and inference without loss of precision. We fully implement and evaluate our ideas. Compared with recent works, our scheme achieves $2-4 \times$ efficiency improvement and is robust to be widely applied.

{{</citation>}}


### (69/74) IIDS: Design of Intelligent Intrusion Detection System for Internet-of-Things Applications (KG Raghavendra Narayan et al., 2023)

{{<citation>}}

KG Raghavendra Narayan, Srijanee Mookherji, Vanga Odelu, Rajendra Prasath, Anish Chand Turlapaty, Ashok Kumar Das. (2023)  
**IIDS: Design of Intelligent Intrusion Detection System for Internet-of-Things Applications**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2308.00943v1)  

---


**ABSTRACT**  
With rapid technological growth, security attacks are drastically increasing. In many crucial Internet-of-Things (IoT) applications such as healthcare and defense, the early detection of security attacks plays a significant role in protecting huge resources. An intrusion detection system is used to address this problem. The signature-based approaches fail to detect zero-day attacks. So anomaly-based detection particularly AI tools, are becoming popular. In addition, the imbalanced dataset leads to biased results. In Machine Learning (ML) models, F1 score is an important metric to measure the accuracy of class-level correct predictions. The model may fail to detect the target samples if the F1 is considerably low. It will lead to unrecoverable consequences in sensitive applications such as healthcare and defense. So, any improvement in the F1 score has significant impact on the resource protection. In this paper, we present a framework for ML-based intrusion detection system for an imbalanced dataset. In this study, the most recent dataset, namely CICIoT2023 is considered. The random forest (RF) algorithm is used in the proposed framework. The proposed approach improves 3.72%, 3.75% and 4.69% in precision, recall and F1 score, respectively, with the existing method. Additionally, for unsaturated classes (i.e., classes with F1 score < 0.99), F1 score improved significantly by 7.9%. As a result, the proposed approach is more suitable for IoT security applications for efficient detection of intrusion and is useful in further studies.

{{</citation>}}


## cs.SI (1)



### (70/74) Shaping Online Dialogue: Examining How Community Rules Affect Discussion Structures on Reddit (Anna Fang et al., 2023)

{{<citation>}}

Anna Fang, Wenjie Yang, Haiyi Zhu. (2023)  
**Shaping Online Dialogue: Examining How Community Rules Affect Discussion Structures on Reddit**  

---
Primary Category: cs.SI  
Categories: cs-HC, cs-SI, cs.SI  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2308.01257v1)  

---


**ABSTRACT**  
Community rules play a key part in enabling or constraining the behaviors of members in online communities. However, little is unknown regarding whether and to what degree changing rules actually affects community dynamics. In this paper, we seek to understand how these behavior-governing rules shape the interactions between users, as well as the structure of their discussion. Using the top communities on Reddit (i.e. subreddits), we first contribute a taxonomy of behavior-based rule categories across Reddit. Then, we use a network analysis perspective to discover how changing implementation of different rule categories affects subreddits' user interaction and discussion networks over a 1.5 year period. Our study find several significant effects, including greater clustering among users when subreddits increase rules focused on structural regulation and how restricting allowable content surprisingly leads to more interactions between users. Our findings contribute to research in proactive moderation through rule setting, as well as lend valuable insights for online community designers and moderators to achieve desired community dynamics.

{{</citation>}}


## eess.IV (1)



### (71/74) CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion (Fenghe Tang et al., 2023)

{{<citation>}}

Fenghe Tang, Jianrui Ding, Lingtao Wang, Chunping Ning, S. Kevin Zhou. (2023)  
**CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion**  

---
Primary Category: eess.IV  
Categories: I-4-6, cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.01239v2)  

---


**ABSTRACT**  
The U-shaped architecture has emerged as a crucial paradigm in the design of medical image segmentation networks. However, due to the inherent local limitations of convolution, a fully convolutional segmentation network with U-shaped architecture struggles to effectively extract global context information, which is vital for the precise localization of lesions. While hybrid architectures combining CNNs and Transformers can address these issues, their application in real medical scenarios is limited due to the computational resource constraints imposed by the environment and edge devices. In addition, the convolutional inductive bias in lightweight networks adeptly fits the scarce medical data, which is lacking in the Transformer based network. In order to extract global context information while taking advantage of the inductive bias, we propose CMUNeXt, an efficient fully convolutional lightweight medical image segmentation network, which enables fast and accurate auxiliary diagnosis in real scene scenarios. CMUNeXt leverages large kernel and inverted bottleneck design to thoroughly mix distant spatial and location information, efficiently extracting global context information. We also introduce the Skip-Fusion block, designed to enable smooth skip-connections and ensure ample feature fusion. Experimental results on multiple medical image datasets demonstrate that CMUNeXt outperforms existing heavyweight and lightweight medical image segmentation networks in terms of segmentation performance, while offering a faster inference speed, lighter weights, and a reduced computational cost. The code is available at https://github.com/FengheTan9/CMUNeXt.

{{</citation>}}


## math.DS (1)



### (72/74) Embedding Capabilities of Neural ODEs (Christian Kuehn et al., 2023)

{{<citation>}}

Christian Kuehn, Sara-Viola Kuntz. (2023)  
**Embedding Capabilities of Neural ODEs**  

---
Primary Category: math.DS  
Categories: cs-NE, math-DS, math.DS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.01213v1)  

---


**ABSTRACT**  
A class of neural networks that gained particular interest in the last years are neural ordinary differential equations (neural ODEs). We study input-output relations of neural ODEs using dynamical systems theory and prove several results about the exact embedding of maps in different neural ODE architectures in low and high dimension. The embedding capability of a neural ODE architecture can be increased by adding, for example, a linear layer, or augmenting the phase space. Yet, there is currently no systematic theory available and our work contributes towards this goal by developing various embedding results as well as identifying situations, where no embedding is possible. The mathematical techniques used include as main components iterative functional equations, Morse functions and suspension flows, as well as several further ideas from analysis. Although practically, mainly universal approximation theorems are used, our geometric dynamical systems viewpoint on universal embedding provides a fundamental understanding, why certain neural ODE architectures perform better than others.

{{</citation>}}


## cs.IR (2)



### (73/74) A Survey on Popularity Bias in Recommender Systems (Anastasiia Klimashevskaia et al., 2023)

{{<citation>}}

Anastasiia Klimashevskaia, Dietmar Jannach, Mehdi Elahi, Christoph Trattner. (2023)  
**A Survey on Popularity Bias in Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.01118v1)  

---


**ABSTRACT**  
Recommender systems help people find relevant content in a personalized way. One main promise of such systems is that they are able to increase the visibility of items in the long tail, i.e., the lesser-known items in a catalogue. Existing research, however, suggests that in many situations today's recommendation algorithms instead exhibit a popularity bias, meaning that they often focus on rather popular items in their recommendations. Such a bias may not only lead to limited value of the recommendations for consumers and providers in the short run, but it may also cause undesired reinforcement effects over time. In this paper, we discuss the potential reasons for popularity bias and we review existing approaches to detect, quantify and mitigate popularity bias in recommender systems. Our survey therefore includes both an overview of the computational metrics used in the literature as well as a review of the main technical approaches to reduce the bias. We furthermore critically discuss today's literature, where we observe that the research is almost entirely based on computational experiments and on certain assumptions regarding the practical effects of including long-tail items in the recommendations.

{{</citation>}}


### (74/74) Towards Better Query Classification with Multi-Expert Knowledge Condensation in JD Ads Search (Kun-Peng Ning et al., 2023)

{{<citation>}}

Kun-Peng Ning, Ming Pang, Zheng Fang, Xue Jiang, Xi-Wei Zhao, Chang-Ping Peng, Zhan-Gang Lin, Jing-He Hu, Jing-Ping Shao. (2023)  
**Towards Better Query Classification with Multi-Expert Knowledge Condensation in JD Ads Search**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.01098v1)  

---


**ABSTRACT**  
Search query classification, as an effective way to understand user intents, is of great importance in real-world online ads systems. To ensure a lower latency, a shallow model (e.g. FastText) is widely used for efficient online inference. However, the representation ability of the FastText model is insufficient, resulting in poor classification performance, especially on some low-frequency queries and tailed categories. Using a deeper and more complex model (e.g. BERT) is an effective solution, but it will cause a higher online inference latency and more expensive computing costs. Thus, how to juggle both inference efficiency and classification performance is obviously of great practical importance. To overcome this challenge, in this paper, we propose knowledge condensation (KC), a simple yet effective knowledge distillation framework to boost the classification performance of the online FastText model under strict low latency constraints. Specifically, we propose to train an offline BERT model to retrieve more potentially relevant data. Benefiting from its powerful semantic representation, more relevant labels not exposed in the historical data will be added into the training set for better FastText model training. Moreover, a novel distribution-diverse multi-expert learning strategy is proposed to further improve the mining ability of relevant data. By training multiple BERT models from different data distributions, it can respectively perform better at high, middle, and low-frequency search queries. The model ensemble from multi-distribution makes its retrieval ability more powerful. We have deployed two versions of this framework in JD search, and both offline experiments and online A/B testing from multiple datasets have validated the effectiveness of the proposed approach.

{{</citation>}}
