---
draft: false
title: "arXiv @ 2023.08.17"
date: 2023-08-17
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.17"
    identifier: arxiv_20230817
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.IR (7)](#csir-7)
- [cs.LG (15)](#cslg-15)
- [cs.DC (3)](#csdc-3)
- [cs.CL (18)](#cscl-18)
- [cs.SE (4)](#csse-4)
- [cs.AI (4)](#csai-4)
- [eess.AS (1)](#eessas-1)
- [cs.HC (2)](#cshc-2)
- [cs.CV (34)](#cscv-34)
- [q-bio.BM (1)](#q-biobm-1)
- [cond-mat.mtrl-sci (1)](#cond-matmtrl-sci-1)
- [cs.CR (2)](#cscr-2)
- [eess.SY (1)](#eesssy-1)
- [q-fin.CP (1)](#q-fincp-1)
- [cs.NE (1)](#csne-1)
- [cs.DS (1)](#csds-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.SD (1)](#cssd-1)
- [eess.IV (1)](#eessiv-1)
- [astro-ph.EP (1)](#astro-phep-1)

## cs.IR (7)



### (1/100) Decentralized Graph Neural Network for Privacy-Preserving Recommendation (Xiaolin Zheng et al., 2023)

{{<citation>}}

Xiaolin Zheng, Zhongyu Wang, Chaochao Chen, Jiashu Qian, Yao Yang. (2023)  
**Decentralized Graph Neural Network for Privacy-Preserving Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CR, cs-IR, cs-LG, cs.IR  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2308.08072v1)  

---


**ABSTRACT**  
Building a graph neural network (GNN)-based recommender system without violating user privacy proves challenging. Existing methods can be divided into federated GNNs and decentralized GNNs. But both methods have undesirable effects, i.e., low communication efficiency and privacy leakage. This paper proposes DGREC, a novel decentralized GNN for privacy-preserving recommendations, where users can choose to publicize their interactions. It includes three stages, i.e., graph construction, local gradient calculation, and global gradient passing. The first stage builds a local inner-item hypergraph for each user and a global inter-user graph. The second stage models user preference and calculates gradients on each local device. The third stage designs a local differential privacy mechanism named secure gradient-sharing, which proves strong privacy-preserving of users' private data. We conduct extensive experiments on three public datasets to validate the consistent superiority of our framework.

{{</citation>}}


### (2/100) Dynamic Embedding Size Search with Minimum Regret for Streaming Recommender System (Bowei He et al., 2023)

{{<citation>}}

Bowei He, Xu He, Renrui Zhang, Yingxue Zhang, Ruiming Tang, Chen Ma. (2023)  
**Dynamic Embedding Size Search with Minimum Regret for Streaming Recommender System**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.07760v1)  

---


**ABSTRACT**  
With the continuous increase of users and items, conventional recommender systems trained on static datasets can hardly adapt to changing environments. The high-throughput data requires the model to be updated in a timely manner for capturing the user interest dynamics, which leads to the emergence of streaming recommender systems. Due to the prevalence of deep learning-based recommender systems, the embedding layer is widely adopted to represent the characteristics of users, items, and other features in low-dimensional vectors. However, it has been proved that setting an identical and static embedding size is sub-optimal in terms of recommendation performance and memory cost, especially for streaming recommendations. To tackle this problem, we first rethink the streaming model update process and model the dynamic embedding size search as a bandit problem. Then, we analyze and quantify the factors that influence the optimal embedding sizes from the statistics perspective. Based on this, we propose the \textbf{D}ynamic \textbf{E}mbedding \textbf{S}ize \textbf{S}earch (\textbf{DESS}) method to minimize the embedding size selection regret on both user and item sides in a non-stationary manner. Theoretically, we obtain a sublinear regret upper bound superior to previous methods. Empirical results across two recommendation tasks on four public datasets also demonstrate that our approach can achieve better streaming recommendation performance with lower memory cost and higher time efficiency.

{{</citation>}}


### (3/100) Self-Supervised Dynamic Hypergraph Recommendation based on Hyper-Relational Knowledge Graph (Yi Liu et al., 2023)

{{<citation>}}

Yi Liu, Hongrui Xuan, Bohan Li, Meng Wang, Tong Chen, Hongzhi Yin. (2023)  
**Self-Supervised Dynamic Hypergraph Recommendation based on Hyper-Relational Knowledge Graph**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GNN, Knowledge Graph, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.07752v1)  

---


**ABSTRACT**  
Knowledge graphs (KGs) are commonly used as side information to enhance collaborative signals and improve recommendation quality. In the context of knowledge-aware recommendation (KGR), graph neural networks (GNNs) have emerged as promising solutions for modeling factual and semantic information in KGs. However, the long-tail distribution of entities leads to sparsity in supervision signals, which weakens the quality of item representation when utilizing KG enhancement. Additionally, the binary relation representation of KGs simplifies hyper-relational facts, making it challenging to model complex real-world information. Furthermore, the over-smoothing phenomenon results in indistinguishable representations and information loss. To address these challenges, we propose the SDK (Self-Supervised Dynamic Hypergraph Recommendation based on Hyper-Relational Knowledge Graph) framework. This framework establishes a cross-view hypergraph self-supervised learning mechanism for KG enhancement. Specifically, we model hyper-relational facts in KGs to capture interdependencies between entities under complete semantic conditions. With the refined representation, a hypergraph is dynamically constructed to preserve features in the deep vector space, thereby alleviating the over-smoothing problem. Furthermore, we mine external supervision signals from both the global perspective of the hypergraph and the local perspective of collaborative filtering (CF) to guide the model prediction process. Extensive experiments conducted on different datasets demonstrate the superiority of the SDK framework over state-of-the-art models. The results showcase its ability to alleviate the effects of over-smoothing and supervision signal sparsity.

{{</citation>}}


### (4/100) SPM: Structured Pretraining and Matching Architectures for Relevance Modeling in Meituan Search (Wen Zan et al., 2023)

{{<citation>}}

Wen Zan, Yaopeng Han, Xiaotian Jiang, Yao Xiao, Yang Yang, Dayao Chen, Sheng Chen. (2023)  
**SPM: Structured Pretraining and Matching Architectures for Relevance Modeling in Meituan Search**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.07711v2)  

---


**ABSTRACT**  
In e-commerce search, relevance between query and documents is an essential requirement for satisfying user experience. Different from traditional e-commerce platforms that offer products, users search on life service platforms such as Meituan mainly for product providers, which usually have abundant structured information, e.g. name, address, category, thousands of products. Modeling search relevance with these rich structured contents is challenging due to the following issues: (1) there is language distribution discrepancy among different fields of structured document, making it difficult to directly adopt off-the-shelf pretrained language model based methods like BERT. (2) different fields usually have different importance and their length vary greatly, making it difficult to extract document information helpful for relevance matching.   To tackle these issues, in this paper we propose a novel two-stage pretraining and matching architecture for relevance matching with rich structured documents. At pretraining stage, we propose an effective pretraining method that employs both query and multiple fields of document as inputs, including an effective information compression method for lengthy fields. At relevance matching stage, a novel matching method is proposed by leveraging domain knowledge in search query to generate more effective document representations for relevance scoring. Extensive offline experiments and online A/B tests on millions of users verify that the proposed architectures effectively improve the performance of relevance modeling. The model has already been deployed online, serving the search traffic of Meituan for over a year.

{{</citation>}}


### (5/100) Learning from All Sides: Diversified Positive Augmentation via Self-distillation in Recommendation (Chong Liu et al., 2023)

{{<citation>}}

Chong Liu, Xiaoyang Liu, Ruobing Xie, Lixin Zhang, Feng Xia, Leyu Lin. (2023)  
**Learning from All Sides: Diversified Positive Augmentation via Self-distillation in Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.07629v1)  

---


**ABSTRACT**  
Personalized recommendation relies on user historical behaviors to provide user-interested items, and thus seriously struggles with the data sparsity issue. A powerful positive item augmentation is beneficial to address the sparsity issue, while few works could jointly consider both the accuracy and diversity of these augmented training labels. In this work, we propose a novel model-agnostic Diversified self-distillation guided positive augmentation (DivSPA) for accurate and diverse positive item augmentations. Specifically, DivSPA first conducts three types of retrieval strategies to collect high-quality and diverse positive item candidates according to users' overall interests, short-term intentions, and similar users. Next, a self-distillation module is conducted to double-check and rerank these candidates as the final positive augmentations. Extensive offline and online evaluations verify the effectiveness of our proposed DivSPA on both accuracy and diversity. DivSPA is simple and effective, which could be conveniently adapted to other base models and systems. Currently, DivSPA has been deployed on multiple widely-used real-world recommender systems.

{{</citation>}}


### (6/100) Temporal Interest Network for Click-Through Rate Prediction (Haolin Zhou et al., 2023)

{{<citation>}}

Haolin Zhou, Junwei Pan, Xinyi Zhou, Xihua Chen, Jie Jiang, Xiaofeng Gao, Guihai Chen. (2023)  
**Temporal Interest Network for Click-Through Rate Prediction**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2308.08487v1)  

---


**ABSTRACT**  
The history of user behaviors constitutes one of the most significant characteristics in predicting the click-through rate (CTR), owing to their strong semantic and temporal correlation with the target item. While the literature has individually examined each of these correlations, research has yet to analyze them in combination, that is, the quadruple correlation of (behavior semantics, target semantics, behavior temporal, and target temporal). The effect of this correlation on performance and the extent to which existing methods learn it remain unknown. To address this gap, we empirically measure the quadruple correlation and observe intuitive yet robust quadruple patterns. We measure the learned correlation of several representative user behavior methods, but to our surprise, none of them learn such a pattern, especially the temporal one.   In this paper, we propose the Temporal Interest Network (TIN) to capture the quadruple semantic and temporal correlation between behaviors and the target. We achieve this by incorporating target-aware temporal encoding, in addition to semantic embedding, to represent behaviors and the target. Furthermore, we deploy target-aware attention, along with target-aware representation, to explicitly conduct the 4-way interaction. We performed comprehensive evaluations on the Amazon and Alibaba datasets. Our proposed TIN outperforms the best-performing baselines by 0.43\% and 0.29\% on two datasets, respectively. Comprehensive analysis and visualization show that TIN is indeed capable of learning the quadruple correlation effectively, while all existing methods fail to do so. We provide our implementation of TIN in Tensorflow.

{{</citation>}}


### (7/100) Delphic Costs and Benefits in Web Search: A utilitarian and historical analysis (Andrei Z. Broder et al., 2023)

{{<citation>}}

Andrei Z. Broder, Preston McAfee. (2023)  
**Delphic Costs and Benefits in Web Search: A utilitarian and historical analysis**  

---
Primary Category: cs.IR  
Categories: H-3-3; H-3-5; J-4, cs-IR, cs.IR  
Keywords: Information Retrieval, Language Model  
[Paper Link](http://arxiv.org/abs/2308.07525v1)  

---


**ABSTRACT**  
We present a new framework to conceptualize and operationalize the total user experience of search, by studying the entirety of a search journey from an utilitarian point of view.   Web search engines are widely perceived as "free". But search requires time and effort: in reality there are many intermingled non-monetary costs (e.g. time costs, cognitive costs, interactivity costs) and the benefits may be marred by various impairments, such as misunderstanding and misinformation. This characterization of costs and benefits appears to be inherent to the human search for information within the pursuit of some larger task: most of the costs and impairments can be identified in interactions with any web search engine, interactions with public libraries, and even in interactions with ancient oracles. To emphasize this innate connection, we call these costs and benefits Delphic, in contrast to explicitly financial costs and benefits.   Our main thesis is that the users' satisfaction with a search engine mostly depends on their experience of Delphic cost and benefits, in other words on their utility. The consumer utility is correlated with classic measures of search engine quality, such as ranking, precision, recall, etc., but is not completely determined by them. To argue our thesis, we catalog the Delphic costs and benefits and show how the development of search engines over the last quarter century, from classic Information Retrieval roots to the integration of Large Language Models, was driven to a great extent by the quest of decreasing Delphic costs and increasing Delphic benefits.   We hope that the Delphic costs framework will engender new ideas and new research for evaluating and improving the web experience for everyone.

{{</citation>}}


## cs.LG (15)



### (8/100) Freshness or Accuracy, Why Not Both? Addressing Delayed Feedback via Dynamic Graph Neural Networks (Xiaolin Zheng et al., 2023)

{{<citation>}}

Xiaolin Zheng, Zhongyu Wang, Chaochao Chen, Feng Zhu, Jiashu Qian. (2023)  
**Freshness or Accuracy, Why Not Both? Addressing Delayed Feedback via Dynamic Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.08071v1)  

---


**ABSTRACT**  
The delayed feedback problem is one of the most pressing challenges in predicting the conversion rate since users' conversions are always delayed in online commercial systems. Although new data are beneficial for continuous training, without complete feedback information, i.e., conversion labels, training algorithms may suffer from overwhelming fake negatives. Existing methods tend to use multitask learning or design data pipelines to solve the delayed feedback problem. However, these methods have a trade-off between data freshness and label accuracy. In this paper, we propose Delayed Feedback Modeling by Dynamic Graph Neural Network (DGDFEM). It includes three stages, i.e., preparing a data pipeline, building a dynamic graph, and training a CVR prediction model. In the model training, we propose a novel graph convolutional method named HLGCN, which leverages both high-pass and low-pass filters to deal with conversion and non-conversion relationships. The proposed method achieves both data freshness and label accuracy. We conduct extensive experiments on three industry datasets, which validate the consistent superiority of our method.

{{</citation>}}


### (9/100) Back to Basics: A Sanity Check on Modern Time Series Classification Algorithms (Bhaskar Dhariyal et al., 2023)

{{<citation>}}

Bhaskar Dhariyal, Thach Le Nguyen, Georgiana Ifrim. (2023)  
**Back to Basics: A Sanity Check on Modern Time Series Classification Algorithms**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.07886v1)  

---


**ABSTRACT**  
The state-of-the-art in time series classification has come a long way, from the 1NN-DTW algorithm to the ROCKET family of classifiers. However, in the current fast-paced development of new classifiers, taking a step back and performing simple baseline checks is essential. These checks are often overlooked, as researchers are focused on establishing new state-of-the-art results, developing scalable algorithms, and making models explainable. Nevertheless, there are many datasets that look like time series at first glance, but classic algorithms such as tabular methods with no time ordering may perform better on such problems. For example, for spectroscopy datasets, tabular methods tend to significantly outperform recent time series methods. In this study, we compare the performance of tabular models using classic machine learning approaches (e.g., Ridge, LDA, RandomForest) with the ROCKET family of classifiers (e.g., Rocket, MiniRocket, MultiRocket). Tabular models are simple and very efficient, while the ROCKET family of classifiers are more complex and have state-of-the-art accuracy and efficiency among recent time series classifiers. We find that tabular models outperform the ROCKET family of classifiers on approximately 19% of univariate and 28% of multivariate datasets in the UCR/UEA benchmark and achieve accuracy within 10 percentage points on about 50% of datasets. Our results suggest that it is important to consider simple tabular models as baselines when developing time series classifiers. These models are very fast, can be as effective as more complex methods and may be easier to understand and deploy.

{{</citation>}}


### (10/100) Towards Temporal Edge Regression: A Case Study on Agriculture Trade Between Nations (Lekang Jiang et al., 2023)

{{<citation>}}

Lekang Jiang, Caiqi Zhang, Farimah Poursafaei, Shenyang Huang. (2023)  
**Towards Temporal Edge Regression: A Case Study on Agriculture Trade Between Nations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.07883v1)  

---


**ABSTRACT**  
Recently, Graph Neural Networks (GNNs) have shown promising performance in tasks on dynamic graphs such as node classification, link prediction and graph regression. However, few work has studied the temporal edge regression task which has important real-world applications. In this paper, we explore the application of GNNs to edge regression tasks in both static and dynamic settings, focusing on predicting food and agriculture trade values between nations. We introduce three simple yet strong baselines and comprehensively evaluate one static and three dynamic GNN models using the UN Trade dataset. Our experimental results reveal that the baselines exhibit remarkably strong performance across various settings, highlighting the inadequacy of existing GNNs. We also find that TGN outperforms other GNN models, suggesting TGN is a more appropriate choice for edge regression tasks. Moreover, we note that the proportion of negative edges in the training samples significantly affects the test performance. The companion source code can be found at: https://github.com/scylj1/GNN_Edge_Regression.

{{</citation>}}


### (11/100) Emotion Embeddings $\unicode{x2014}$ Learning Stable and Homogeneous Abstractions from Heterogeneous Affective Datasets (Sven Buechel et al., 2023)

{{<citation>}}

Sven Buechel, Udo Hahn. (2023)  
**Emotion Embeddings $\unicode{x2014}$ Learning Stable and Homogeneous Abstractions from Heterogeneous Affective Datasets**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.07871v1)  

---


**ABSTRACT**  
Human emotion is expressed in many communication modalities and media formats and so their computational study is equally diversified into natural language processing, audio signal analysis, computer vision, etc. Similarly, the large variety of representation formats used in previous research to describe emotions (polarity scales, basic emotion categories, dimensional approaches, appraisal theory, etc.) have led to an ever proliferating diversity of datasets, predictive models, and software tools for emotion analysis. Because of these two distinct types of heterogeneity, at the expressional and representational level, there is a dire need to unify previous work on increasingly diverging data and label types. This article presents such a unifying computational model. We propose a training procedure that learns a shared latent representation for emotions, so-called emotion embeddings, independent of different natural languages, communication modalities, media or representation label formats, and even disparate model architectures. Experiments on a wide range of heterogeneous affective datasets indicate that this approach yields the desired interoperability for the sake of reusability, interpretability and flexibility, without penalizing prediction quality. Code and data are archived under https://doi.org/10.5281/zenodo.7405327 .

{{</citation>}}


### (12/100) Dyadic Reinforcement Learning (Shuangning Li et al., 2023)

{{<citation>}}

Shuangning Li, Lluis Salvat Niell, Sung Won Choi, Inbal Nahum-Shani, Guy Shani, Susan Murphy. (2023)  
**Dyadic Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.07843v1)  

---


**ABSTRACT**  
Mobile health aims to enhance health outcomes by delivering interventions to individuals as they go about their daily life. The involvement of care partners and social support networks often proves crucial in helping individuals managing burdensome medical conditions. This presents opportunities in mobile health to design interventions that target the dyadic relationship -- the relationship between a target person and their care partner -- with the aim of enhancing social support. In this paper, we develop dyadic RL, an online reinforcement learning algorithm designed to personalize intervention delivery based on contextual factors and past responses of a target person and their care partner. Here, multiple sets of interventions impact the dyad across multiple time intervals. The developed dyadic RL is Bayesian and hierarchical. We formally introduce the problem setup, develop dyadic RL and establish a regret bound. We demonstrate dyadic RL's empirical performance through simulation studies on both toy scenarios and on a realistic test bed constructed from data collected in a mobile health study.

{{</citation>}}


### (13/100) Simple and Efficient Partial Graph Adversarial Attack: A New Perspective (Guanghui Zhu et al., 2023)

{{<citation>}}

Guanghui Zhu, Mengyu Chen, Chunfeng Yuan, Yihua Huang. (2023)  
**Simple and Efficient Partial Graph Adversarial Attack: A New Perspective**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2308.07834v1)  

---


**ABSTRACT**  
As the study of graph neural networks becomes more intensive and comprehensive, their robustness and security have received great research interest. The existing global attack methods treat all nodes in the graph as their attack targets. Although existing methods have achieved excellent results, there is still considerable space for improvement. The key problem is that the current approaches rigidly follow the definition of global attacks. They ignore an important issue, i.e., different nodes have different robustness and are not equally resilient to attacks. From a global attacker's view, we should arrange the attack budget wisely, rather than wasting them on highly robust nodes. To this end, we propose a totally new method named partial graph attack (PGA), which selects the vulnerable nodes as attack targets. First, to select the vulnerable items, we propose a hierarchical target selection policy, which allows attackers to only focus on easy-to-attack nodes. Then, we propose a cost-effective anchor-picking policy to pick the most promising anchors for adding or removing edges, and a more aggressive iterative greedy-based attack method to perform more efficient attacks. Extensive experimental results demonstrate that PGA can achieve significant improvements in both attack effect and attack efficiency compared to other existing graph global attack methods.

{{</citation>}}


### (14/100) A Graph Encoder-Decoder Network for Unsupervised Anomaly Detection (Mahsa Mesgaran et al., 2023)

{{<citation>}}

Mahsa Mesgaran, A. Ben Hamza. (2023)  
**A Graph Encoder-Decoder Network for Unsupervised Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, GNN  
[Paper Link](http://arxiv.org/abs/2308.07774v1)  

---


**ABSTRACT**  
A key component of many graph neural networks (GNNs) is the pooling operation, which seeks to reduce the size of a graph while preserving important structural information. However, most existing graph pooling strategies rely on an assignment matrix obtained by employing a GNN layer, which is characterized by trainable parameters, often leading to significant computational complexity and a lack of interpretability in the pooling process. In this paper, we propose an unsupervised graph encoder-decoder model to detect abnormal nodes from graphs by learning an anomaly scoring function to rank nodes based on their degree of abnormality. In the encoding stage, we design a novel pooling mechanism, named LCPool, which leverages locality-constrained linear coding for feature encoding to find a cluster assignment matrix by solving a least-squares optimization problem with a locality regularization term. By enforcing locality constraints during the coding process, LCPool is designed to be free from learnable parameters, capable of efficiently handling large graphs, and can effectively generate a coarser graph representation while retaining the most significant structural characteristics of the graph. In the decoding stage, we propose an unpooling operation, called LCUnpool, to reconstruct both the structure and nodal features of the original graph. We conduct empirical evaluations of our method on six benchmark datasets using several evaluation metrics, and the results demonstrate its superiority over state-of-the-art anomaly detection approaches.

{{</citation>}}


### (15/100) Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening (Jack Foster et al., 2023)

{{<citation>}}

Jack Foster, Stefan Schoepf, Alexandra Brintrup. (2023)  
**Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.07707v1)  

---


**ABSTRACT**  
Machine unlearning, the ability for a machine learning model to forget, is becoming increasingly important to comply with data privacy regulations, as well as to remove harmful, manipulated, or outdated information. The key challenge lies in forgetting specific information while protecting model performance on the remaining data. While current state-of-the-art methods perform well, they typically require some level of retraining over the retained data, in order to protect or restore model performance. This adds computational overhead and mandates that the training data remain available and accessible, which may not be feasible. In contrast, other methods employ a retrain-free paradigm, however, these approaches are prohibitively computationally expensive and do not perform on par with their retrain-based counterparts. We present Selective Synaptic Dampening (SSD), a novel two-step, post hoc, retrain-free approach to machine unlearning which is fast, performant, and does not require long-term storage of the training data. First, SSD uses the Fisher information matrix of the training and forgetting data to select parameters that are disproportionately important to the forget set. Second, SSD induces forgetting by dampening these parameters proportional to their relative importance to the forget set with respect to the wider training data. We evaluate our method against several existing unlearning methods in a range of experiments using ResNet18 and Vision Transformer. Results show that the performance of SSD is competitive with retrain-based post hoc methods, demonstrating the viability of retrain-free post hoc unlearning approaches.

{{</citation>}}


### (16/100) Gradient-Based Post-Training Quantization: Challenging the Status Quo (Edouard Yvinec et al., 2023)

{{<citation>}}

Edouard Yvinec, Arnaud Dapogny, Kevin Bailly. (2023)  
**Gradient-Based Post-Training Quantization: Challenging the Status Quo**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: GPT, QA, Quantization  
[Paper Link](http://arxiv.org/abs/2308.07662v1)  

---


**ABSTRACT**  
Quantization has become a crucial step for the efficient deployment of deep neural networks, where floating point operations are converted to simpler fixed point operations. In its most naive form, it simply consists in a combination of scaling and rounding transformations, leading to either a limited compression rate or a significant accuracy drop. Recently, Gradient-based post-training quantization (GPTQ) methods appears to be constitute a suitable trade-off between such simple methods and more powerful, yet expensive Quantization-Aware Training (QAT) approaches, particularly when attempting to quantize LLMs, where scalability of the quantization process is of paramount importance. GPTQ essentially consists in learning the rounding operation using a small calibration set. In this work, we challenge common choices in GPTQ methods. In particular, we show that the process is, to a certain extent, robust to a number of variables (weight selection, feature augmentation, choice of calibration set). More importantly, we derive a number of best practices for designing more efficient and scalable GPTQ methods, regarding the problem formulation (loss, degrees of freedom, use of non-uniform quantization schemes) or optimization process (choice of variable and optimizer). Lastly, we propose a novel importance-based mixed-precision technique. Those guidelines lead to significant performance improvements on all the tested state-of-the-art GPTQ methods and networks (e.g. +6.819 points on ViT for 4-bit quantization), paving the way for the design of scalable, yet effective quantization methods.

{{</citation>}}


### (17/100) Attention Is Not All You Need Anymore (Zhe Chen, 2023)

{{<citation>}}

Zhe Chen. (2023)  
**Attention Is Not All You Need Anymore**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs-NE, cs.LG  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.07661v1)  

---


**ABSTRACT**  
In recent years, the popular Transformer architecture has achieved great success in many application areas, including natural language processing and computer vision. Many existing works aim to reduce the computational and memory complexity of the self-attention mechanism in the Transformer by trading off performance. However, performance is key for the continuing success of the Transformer. In this paper, a drop-in replacement for the self-attention mechanism in the Transformer, called the Extractor, is proposed. Experimental results show that replacing the self-attention mechanism with the Extractor improves the performance of the Transformer. Furthermore, the proposed Extractor has the potential to run faster than the self-attention since it has a much shorter critical path of computation. Additionally, the sequence prediction problem in the context of text generation is formulated using variable-length discrete-time Markov chains, and the Transformer is reviewed based on our understanding.

{{</citation>}}


### (18/100) Ternary Singular Value Decomposition as a Better Parameterized Form in Linear Mapping (Boyu Chen et al., 2023)

{{<citation>}}

Boyu Chen, Hanxuan Chen, Jiao He, Fengyu Sun, Shangling Jui. (2023)  
**Ternary Singular Value Decomposition as a Better Parameterized Form in Linear Mapping**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: BERT, Quantization  
[Paper Link](http://arxiv.org/abs/2308.07641v1)  

---


**ABSTRACT**  
We present a simple yet novel parameterized form of linear mapping to achieves remarkable network compression performance: a pseudo SVD called Ternary SVD (TSVD).   Unlike vanilla SVD, TSVD limits the $U$ and $V$ matrices in SVD to ternary matrices form in $\{\pm 1, 0\}$. This means that instead of using the expensive multiplication instructions, TSVD only requires addition instructions when computing $U(\cdot)$ and $V(\cdot)$.   We provide direct and training transition algorithms for TSVD like Post Training Quantization and Quantization Aware Training respectively. Additionally, we analyze the convergence of the direct transition algorithms in theory.   In experiments, we demonstrate that TSVD can achieve state-of-the-art network compression performance in various types of networks and tasks, including current baseline models such as ConvNext, Swim, BERT, and large language model like OPT.

{{</citation>}}


### (19/100) Generating Personas for Games with Multimodal Adversarial Imitation Learning (William Ahlberg et al., 2023)

{{<citation>}}

William Ahlberg, Alessandro Sestini, Konrad Tollmar, Linus Gisslén. (2023)  
**Generating Personas for Games with Multimodal Adversarial Imitation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.07598v1)  

---


**ABSTRACT**  
Reinforcement learning has been widely successful in producing agents capable of playing games at a human level. However, this requires complex reward engineering, and the agent's resulting policy is often unpredictable. Going beyond reinforcement learning is necessary to model a wide range of human playstyles, which can be difficult to represent with a reward function. This paper presents a novel imitation learning approach to generate multiple persona policies for playtesting. Multimodal Generative Adversarial Imitation Learning (MultiGAIL) uses an auxiliary input parameter to learn distinct personas using a single-agent model. MultiGAIL is based on generative adversarial imitation learning and uses multiple discriminators as reward models, inferring the environment reward by comparing the agent and distinct expert policies. The reward from each discriminator is weighted according to the auxiliary input. Our experimental analysis demonstrates the effectiveness of our technique in two environments with continuous and discrete action spaces.

{{</citation>}}


### (20/100) Semi-Supervised Learning with Multiple Imputations on Non-Random Missing Labels (Jason Lu et al., 2023)

{{<citation>}}

Jason Lu, Michael Ma, Huaze Xu, Zixi Xu. (2023)  
**Semi-Supervised Learning with Multiple Imputations on Non-Random Missing Labels**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.07562v1)  

---


**ABSTRACT**  
Semi-Supervised Learning (SSL) is implemented when algorithms are trained on both labeled and unlabeled data. This is a very common application of ML as it is unrealistic to obtain a fully labeled dataset. Researchers have tackled three main issues: missing at random (MAR), missing completely at random (MCAR), and missing not at random (MNAR). The MNAR problem is the most challenging of the three as one cannot safely assume that all class distributions are equal. Existing methods, including Class-Aware Imputation (CAI) and Class-Aware Propensity (CAP), mostly overlook the non-randomness in the unlabeled data. This paper proposes two new methods of combining multiple imputation models to achieve higher accuracy and less bias. 1) We use multiple imputation models, create confidence intervals, and apply a threshold to ignore pseudo-labels with low confidence. 2) Our new method, SSL with De-biased Imputations (SSL-DI), aims to reduce bias by filtering out inaccurate data and finding a subset that is accurate and reliable. This subset of the larger dataset could be imputed into another SSL model, which will be less biased. The proposed models have been shown to be effective in both MCAR and MNAR situations, and experimental results show that our methodology outperforms existing methods in terms of classification accuracy and reducing bias.

{{</citation>}}


### (21/100) KMF: Knowledge-Aware Multi-Faceted Representation Learning for Zero-Shot Node Classification (Likang Wu et al., 2023)

{{<citation>}}

Likang Wu, Junji Jiang, Hongke Zhao, Hao Wang, Defu Lian, Mengdi Zhang, Enhong Chen. (2023)  
**KMF: Knowledge-Aware Multi-Faceted Representation Learning for Zero-Shot Node Classification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-IR, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Knowledge Graph, Representation Learning, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.08563v1)  

---


**ABSTRACT**  
Recently, Zero-Shot Node Classification (ZNC) has been an emerging and crucial task in graph data analysis. This task aims to predict nodes from unseen classes which are unobserved in the training process. Existing work mainly utilizes Graph Neural Networks (GNNs) to associate features' prototypes and labels' semantics thus enabling knowledge transfer from seen to unseen classes. However, the multi-faceted semantic orientation in the feature-semantic alignment has been neglected by previous work, i.e. the content of a node usually covers diverse topics that are relevant to the semantics of multiple labels. It's necessary to separate and judge the semantic factors that tremendously affect the cognitive ability to improve the generality of models. To this end, we propose a Knowledge-Aware Multi-Faceted framework (KMF) that enhances the richness of label semantics via the extracted KG (Knowledge Graph)-based topics. And then the content of each node is reconstructed to a topic-level representation that offers multi-faceted and fine-grained semantic relevancy to different labels. Due to the particularity of the graph's instance (i.e., node) representation, a novel geometric constraint is developed to alleviate the problem of prototype drift caused by node information aggregation. Finally, we conduct extensive experiments on several public graph datasets and design an application of zero-shot cross-domain recommendation. The quantitative results demonstrate both the effectiveness and generalization of KMF with the comparison of state-of-the-art baselines.

{{</citation>}}


### (22/100) Data Race Detection Using Large Language Models (Le Chen et al., 2023)

{{<citation>}}

Le Chen, Xianzhong Ding, Murali Emani, Tristan Vanderbruggen, Pei-hung Lin, Chuanhua Liao. (2023)  
**Data Race Detection Using Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.07505v1)  

---


**ABSTRACT**  
Large language models (LLMs) are demonstrating significant promise as an alternate strategy to facilitate analyses and optimizations of high-performance computing programs, circumventing the need for resource-intensive manual tool creation. In this paper, we explore a novel LLM-based data race detection approach combining prompting engineering and fine-tuning techniques. We create a dedicated dataset named DRB-ML, which is derived from DataRaceBench, with fine-grain labels showing the presence of data race pairs and their associated variables, line numbers, and read/write information. DRB-ML is then used to evaluate representative LLMs and fine-tune open-source ones. Our experiment shows that LLMs can be a viable approach to data race detection. However, they still cannot compete with traditional data race detection tools when we need detailed information about variable pairs causing data races.

{{</citation>}}


## cs.DC (3)



### (23/100) A Reinforcement Learning Approach for Performance-aware Reduction in Power Consumption of Data Center Compute Nodes (Akhilesh Raj et al., 2023)

{{<citation>}}

Akhilesh Raj, Swann Perarnau, Aniruddha Gokhale. (2023)  
**A Reinforcement Learning Approach for Performance-aware Reduction in Power Consumption of Data Center Compute Nodes**  

---
Primary Category: cs.DC  
Categories: cs-AR, cs-DC, cs-LG, cs-SY, cs.DC, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.08069v1)  

---


**ABSTRACT**  
As Exascale computing becomes a reality, the energy needs of compute nodes in cloud data centers will continue to grow. A common approach to reducing this energy demand is to limit the power consumption of hardware components when workloads are experiencing bottlenecks elsewhere in the system. However, designing a resource controller capable of detecting and limiting power consumption on-the-fly is a complex issue and can also adversely impact application performance. In this paper, we explore the use of Reinforcement Learning (RL) to design a power capping policy on cloud compute nodes using observations on current power consumption and instantaneous application performance (heartbeats). By leveraging the Argo Node Resource Management (NRM) software stack in conjunction with the Intel Running Average Power Limit (RAPL) hardware control mechanism, we design an agent to control the maximum supplied power to processors without compromising on application performance. Employing a Proximal Policy Optimization (PPO) agent to learn an optimal policy on a mathematical model of the compute nodes, we demonstrate and evaluate using the STREAM benchmark how a trained agent running on actual hardware can take actions by balancing power consumption and application performance.

{{</citation>}}


### (24/100) FedCache: A Knowledge Cache-driven Federated Learning Architecture for Personalized Edge Intelligence (Zhiyuan Wu et al., 2023)

{{<citation>}}

Zhiyuan Wu, Sheng Sun, Yuwei Wang, Min Liu, Wen Wang, Xuefeng Jiang, Bo Gao, Jinda Lu. (2023)  
**FedCache: A Knowledge Cache-driven Federated Learning Architecture for Personalized Edge Intelligence**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.07816v1)  

---


**ABSTRACT**  
Edge Intelligence (EI) allows Artificial Intelligence (AI) applications to run at the edge, where data analysis and decision-making can be performed in real-time and close to data sources. To protect data privacy and unify data silos among end devices in EI, Federated Learning (FL) is proposed for collaborative training of shared AI models across devices without compromising data privacy. However, the prevailing FL approaches cannot guarantee model generalization and adaptation on heterogeneous clients. Recently, Personalized Federated Learning (PFL) has drawn growing awareness in EI, as it enables a productive balance between local-specific training requirements inherent in devices and global-generalized optimization objectives for satisfactory performance. However, most existing PFL methods are based on the Parameters Interaction-based Architecture (PIA) represented by FedAvg, which causes unaffordable communication burdens due to large-scale parameters transmission between devices and the edge server. In contrast, Logits Interaction-based Architecture (LIA) allows to update model parameters with logits transfer and gains the advantages of communication lightweight and heterogeneous on-device model allowance compared to PIA. Nevertheless, previous LIA methods attempt to achieve satisfactory performance either relying on unrealistic public datasets or increasing communication overhead for additional information transmission other than logits. To tackle this dilemma, we propose a knowledge cache-driven PFL architecture, named FedCache, which reserves a knowledge cache on the server for fetching personalized knowledge from the samples with similar hashes to each given on-device sample. During the training phase, ensemble distillation is applied to on-device models for constructive optimization with personalized knowledge transferred from the server-side knowledge cache.

{{</citation>}}


### (25/100) Reinforcement Learning (RL) Augmented Cold Start Frequency Reduction in Serverless Computing (Siddharth Agarwal et al., 2023)

{{<citation>}}

Siddharth Agarwal, Maria A. Rodriguez, Rajkumar Buyya. (2023)  
**Reinforcement Learning (RL) Augmented Cold Start Frequency Reduction in Serverless Computing**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs-SY, cs.DC, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.07541v1)  

---


**ABSTRACT**  
Function-as-a-Service is a cloud computing paradigm offering an event-driven execution model to applications. It features serverless attributes by eliminating resource management responsibilities from developers and offers transparent and on-demand scalability of applications. Typical serverless applications have stringent response time and scalability requirements and therefore rely on deployed services to provide quick and fault-tolerant feedback to clients. However, the FaaS paradigm suffers from cold starts as there is a non-negligible delay associated with on-demand function initialization. This work focuses on reducing the frequency of cold starts on the platform by using Reinforcement Learning. Our approach uses Q-learning and considers metrics such as function CPU utilization, existing function instances, and response failure rate to proactively initialize functions in advance based on the expected demand. The proposed solution was implemented on Kubeless and was evaluated using a normalised real-world function demand trace with matrix multiplication as the workload. The results demonstrate a favourable performance of the RL-based agent when compared to Kubeless' default policy and function keep-alive policy by improving throughput by up to 8.81% and reducing computation load and resource wastage by up to 55% and 37%, respectively, which is a direct outcome of reduced cold starts.

{{</citation>}}


## cs.CL (18)



### (26/100) The Costly Dilemma: Generalization, Evaluation and Cost-Optimal Deployment of Large Language Models (Abi Aryan et al., 2023)

{{<citation>}}

Abi Aryan, Aakash Kumar Nain, Andrew McMahon, Lucas Augusto Meyer, Harpreet Singh Sahota. (2023)  
**The Costly Dilemma: Generalization, Evaluation and Cost-Optimal Deployment of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SE, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.08061v1)  

---


**ABSTRACT**  
When deploying machine learning models in production for any product/application, there are three properties that are commonly desired. First, the models should be generalizable, in that we can extend it to further use cases as our knowledge of the domain area develops. Second they should be evaluable, so that there are clear metrics for performance and the calculation of those metrics in production settings are feasible. Finally, the deployment should be cost-optimal as far as possible. In this paper we propose that these three objectives (i.e. generalization, evaluation and cost-optimality) can often be relatively orthogonal and that for large language models, despite their performance over conventional NLP models, enterprises need to carefully assess all the three factors before making substantial investments in this technology. We propose a framework for generalization, evaluation and cost-modeling specifically tailored to large language models, offering insights into the intricacies of development, deployment and management for these large language models.

{{</citation>}}


### (27/100) DiagGPT: An LLM-based Chatbot with Automatic Topic Management for Task-Oriented Dialogue (Lang Cao, 2023)

{{<citation>}}

Lang Cao. (2023)  
**DiagGPT: An LLM-based Chatbot with Automatic Topic Management for Task-Oriented Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, Dialog, Dialogue, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.08043v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), such as ChatGPT, are becoming increasingly sophisticated, demonstrating capabilities that closely resemble those of humans. These AI models are playing an essential role in assisting humans with a wide array of tasks in daily life. A significant application of AI is its use as a chat agent, responding to human inquiries across various domains. Current LLMs have shown proficiency in answering general questions. However, basic question-answering dialogue often falls short in complex diagnostic scenarios, such as legal or medical consultations. These scenarios typically necessitate Task-Oriented Dialogue (TOD), wherein an AI chat agent needs to proactively pose questions and guide users towards specific task completion. Previous fine-tuning models have underperformed in TOD, and current LLMs do not inherently possess this capability. In this paper, we introduce DiagGPT (Dialogue in Diagnosis GPT), an innovative method that extends LLMs to TOD scenarios. Our experiments reveal that DiagGPT exhibits outstanding performance in conducting TOD with users, demonstrating its potential for practical applications.

{{</citation>}}


### (28/100) 'Beware of deception': Detecting Half-Truth and Debunking it through Controlled Claim Editing (Sandeep Singamsetty et al., 2023)

{{<citation>}}

Sandeep Singamsetty, Nishtha Madaan, Sameep Mehta, Varad Bhatnagar, Pushpak Bhattacharyya. (2023)  
**'Beware of deception': Detecting Half-Truth and Debunking it through Controlled Claim Editing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, BLEU, GPT, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2308.07973v1)  

---


**ABSTRACT**  
The prevalence of half-truths, which are statements containing some truth but that are ultimately deceptive, has risen with the increasing use of the internet. To help combat this problem, we have created a comprehensive pipeline consisting of a half-truth detection model and a claim editing model. Our approach utilizes the T5 model for controlled claim editing; "controlled" here means precise adjustments to select parts of a claim. Our methodology achieves an average BLEU score of 0.88 (on a scale of 0-1) and a disinfo-debunk score of 85% on edited claims. Significantly, our T5-based approach outperforms other Language Models such as GPT2, RoBERTa, PEGASUS, and Tailor, with average improvements of 82%, 57%, 42%, and 23% in disinfo-debunk scores, respectively. By extending the LIAR PLUS dataset, we achieve an F1 score of 82% for the half-truth detection model, setting a new benchmark in the field. While previous attempts have been made at half-truth detection, our approach is, to the best of our knowledge, the first to attempt to debunk half-truths.

{{</citation>}}


### (29/100) MultiSChuBERT: Effective Multimodal Fusion for Scholarly Document Quality Prediction (Gideon Maillette de Buy Wenniger et al., 2023)

{{<citation>}}

Gideon Maillette de Buy Wenniger, Thomas van Dongen, Lambert Schomaker. (2023)  
**MultiSChuBERT: Effective Multimodal Fusion for Scholarly Document Quality Prediction**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs-LG, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.07971v1)  

---


**ABSTRACT**  
Automatic assessment of the quality of scholarly documents is a difficult task with high potential impact. Multimodality, in particular the addition of visual information next to text, has been shown to improve the performance on scholarly document quality prediction (SDQP) tasks. We propose the multimodal predictive model MultiSChuBERT. It combines a textual model based on chunking full paper text and aggregating computed BERT chunk-encodings (SChuBERT), with a visual model based on Inception V3.Our work contributes to the current state-of-the-art in SDQP in three ways. First, we show that the method of combining visual and textual embeddings can substantially influence the results. Second, we demonstrate that gradual-unfreezing of the weights of the visual sub-model, reduces its tendency to ovefit the data, improving results. Third, we show the retained benefit of multimodality when replacing standard BERT$_{\textrm{BASE}}$ embeddings with more recent state-of-the-art text embedding models.   Using BERT$_{\textrm{BASE}}$ embeddings, on the (log) number of citations prediction task with the ACL-BiblioMetry dataset, our MultiSChuBERT (text+visual) model obtains an $R^{2}$ score of 0.454 compared to 0.432 for the SChuBERT (text only) model. Similar improvements are obtained on the PeerRead accept/reject prediction task. In our experiments using SciBERT, scincl, SPECTER and SPECTER2.0 embeddings, we show that each of these tailored embeddings adds further improvements over the standard BERT$_{\textrm{BASE}}$ embeddings, with the SPECTER2.0 embeddings performing best.

{{</citation>}}


### (30/100) RAVEN: In-Context Learning with Retrieval Augmented Encoder-Decoder Language Models (Jie Huang et al., 2023)

{{<citation>}}

Jie Huang, Wei Ping, Peng Xu, Mohammad Shoeybi, Kevin Chen-Chuan Chang, Bryan Catanzaro. (2023)  
**RAVEN: In-Context Learning with Retrieval Augmented Encoder-Decoder Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.07922v1)  

---


**ABSTRACT**  
In this paper, we investigate the in-context learning ability of retrieval-augmented encoder-decoder language models. We first conduct a comprehensive analysis of the state-of-the-art ATLAS model and identify its limitations in in-context learning, primarily due to a mismatch between pretraining and testing, as well as a restricted context length. To address these issues, we propose RAVEN, a model that combines retrieval-augmented masked language modeling and prefix language modeling. We further introduce Fusion-in-Context Learning to enhance the few-shot performance by enabling the model to leverage more in-context examples without requiring additional training or model modifications. Through extensive experiments, we demonstrate that RAVEN significantly outperforms ATLAS and achieves results comparable to the most advanced language models in certain scenarios, despite having substantially fewer parameters. Our work underscores the potential of retrieval-augmented encoder-decoder language models for in-context learning and encourages further research in this direction.

{{</citation>}}


### (31/100) Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification (Aojun Zhou et al., 2023)

{{<citation>}}

Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, Hongsheng Li. (2023)  
**Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: AI, GPT, GPT-4, PaLM  
[Paper Link](http://arxiv.org/abs/2308.07921v1)  

---


**ABSTRACT**  
Recent progress in large language models (LLMs) like GPT-4 and PaLM-2 has brought significant advancements in addressing math reasoning problems. In particular, OpenAI's latest version of GPT-4, known as GPT-4 Code Interpreter, shows remarkable performance on challenging math datasets. In this paper, we explore the effect of code on enhancing LLMs' reasoning capability by introducing different constraints on the \textit{Code Usage Frequency} of GPT-4 Code Interpreter. We found that its success can be largely attributed to its powerful skills in generating and executing code, evaluating the output of code execution, and rectifying its solution when receiving unreasonable outputs. Based on this insight, we propose a novel and effective prompting method, explicit \uline{c}ode-based \uline{s}elf-\uline{v}erification~(CSV), to further boost the mathematical reasoning potential of GPT-4 Code Interpreter. This method employs a zero-shot prompt on GPT-4 Code Interpreter to encourage it to use code to self-verify its answers. In instances where the verification state registers as ``False'', the model shall automatically amend its solution, analogous to our approach of rectifying errors during a mathematics examination. Furthermore, we recognize that the states of the verification result indicate the confidence of a solution, which can improve the effectiveness of majority voting. With GPT-4 Code Interpreter and CSV, we achieve an impressive zero-shot accuracy on MATH dataset \textbf{(53.9\% $\to$ 84.3\%)}.

{{</citation>}}


### (32/100) Through the Lens of Core Competency: Survey on Evaluation of Large Language Models (Ziyu Zhuang et al., 2023)

{{<citation>}}

Ziyu Zhuang, Qiguang Chen, Longxuan Ma, Mingda Li, Yi Han, Yushan Qian, Haopeng Bai, Zixian Feng, Weinan Zhang, Ting Liu. (2023)  
**Through the Lens of Core Competency: Survey on Evaluation of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.07902v1)  

---


**ABSTRACT**  
From pre-trained language model (PLM) to large language model (LLM), the field of natural language processing (NLP) has witnessed steep performance gains and wide practical uses. The evaluation of a research field guides its direction of improvement. However, LLMs are extremely hard to thoroughly evaluate for two reasons. First of all, traditional NLP tasks become inadequate due to the excellent performance of LLM. Secondly, existing evaluation tasks are difficult to keep up with the wide range of applications in real-world scenarios. To tackle these problems, existing works proposed various benchmarks to better evaluate LLMs. To clarify the numerous evaluation tasks in both academia and industry, we investigate multiple papers concerning LLM evaluations. We summarize 4 core competencies of LLM, including reasoning, knowledge, reliability, and safety. For every competency, we introduce its definition, corresponding benchmarks, and metrics. Under this competency architecture, similar tasks are combined to reflect corresponding ability, while new tasks can also be easily added into the system. Finally, we give our suggestions on the future direction of LLM's evaluation.

{{</citation>}}


### (33/100) Synthesizing Political Zero-Shot Relation Classification via Codebook Knowledge, NLI, and ChatGPT (Yibo Hu et al., 2023)

{{<citation>}}

Yibo Hu, Erick Skorupa Parolin, Latifur Khan, Patrick T. Brandt, Javier Osorio, Vito J. D'Orazio. (2023)  
**Synthesizing Political Zero-Shot Relation Classification via Codebook Knowledge, NLI, and ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: BERT, ChatGPT, GPT, NLI, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.07876v1)  

---


**ABSTRACT**  
Recent supervised models for event coding vastly outperform pattern-matching methods. However, their reliance solely on new annotations disregards the vast knowledge within expert databases, hindering their applicability to fine-grained classification. To address these limitations, we explore zero-shot approaches for political event ontology relation classification, by leveraging knowledge from established annotation codebooks. Our study encompasses both ChatGPT and a novel natural language inference (NLI) based approach named ZSP. ZSP adopts a tree-query framework that deconstructs the task into context, modality, and class disambiguation levels. This framework improves interpretability, efficiency, and adaptability to schema changes. By conducting extensive experiments on our newly curated datasets, we pinpoint the instability issues within ChatGPT and highlight the superior performance of ZSP. ZSP achieves an impressive 40% improvement in F1 score for fine-grained Rootcode classification. ZSP demonstrates competitive performance compared to supervised BERT models, positioning it as a valuable tool for event record validation and ontology development. Our work underscores the potential of leveraging transfer learning and existing expertise to enhance the efficiency and scalability of research in the field.

{{</citation>}}


### (34/100) Informed Named Entity Recognition Decoding for Generative Language Models (Tobias Deußer et al., 2023)

{{<citation>}}

Tobias Deußer, Lars Hillebrand, Christian Bauckhage, Rafet Sifa. (2023)  
**Informed Named Entity Recognition Decoding for Generative Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2308.07791v1)  

---


**ABSTRACT**  
Ever-larger language models with ever-increasing capabilities are by now well-established text processing tools. Alas, information extraction tasks such as named entity recognition are still largely unaffected by this progress as they are primarily based on the previous generation of encoder-only transformer models. Here, we propose a simple yet effective approach, Informed Named Entity Recognition Decoding (iNERD), which treats named entity recognition as a generative process. It leverages the language understanding capabilities of recent generative models in a future-proof manner and employs an informed decoding scheme incorporating the restricted nature of information extraction into open-ended text generation, improving performance and eliminating any risk of hallucinations. We coarse-tune our model on a merged named entity corpus to strengthen its performance, evaluate five generative language models on eight named entity recognition datasets, and achieve remarkable results, especially in an environment with an unknown entity class set, demonstrating the adaptability of the approach.

{{</citation>}}


### (35/100) Enhancing Visually-Rich Document Understanding via Layout Structure Modeling (Qiwei Li et al., 2023)

{{<citation>}}

Qiwei Li, Zuchao Li, Xiantao Cai, Bo Du, Hai Zhao. (2023)  
**Enhancing Visually-Rich Document Understanding via Layout Structure Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.07777v1)  

---


**ABSTRACT**  
In recent years, the use of multi-modal pre-trained Transformers has led to significant advancements in visually-rich document understanding. However, existing models have mainly focused on features such as text and vision while neglecting the importance of layout relationship between text nodes. In this paper, we propose GraphLayoutLM, a novel document understanding model that leverages the modeling of layout structure graph to inject document layout knowledge into the model. GraphLayoutLM utilizes a graph reordering algorithm to adjust the text sequence based on the graph structure. Additionally, our model uses a layout-aware multi-head self-attention layer to learn document layout knowledge. The proposed model enables the understanding of the spatial arrangement of text elements, improving document comprehension. We evaluate our model on various benchmarks, including FUNSD, XFUND and CORD, and achieve state-of-the-art results among these datasets. Our experimental results demonstrate that our proposed method provides a significant improvement over existing approaches and showcases the importance of incorporating layout information into document understanding models. We also conduct an ablation study to investigate the contribution of each component of our model. The results show that both the graph reordering algorithm and the layout-aware multi-head self-attention layer play a crucial role in achieving the best performance.

{{</citation>}}


### (36/100) Forward-Backward Reasoning in Large Language Models for Verification (Weisen Jiang et al., 2023)

{{<citation>}}

Weisen Jiang, Han Shi, Longhui Yu, Zhengying Liu, Yu Zhang, Zhenguo Li, James T. Kwok. (2023)  
**Forward-Backward Reasoning in Large Language Models for Verification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.07758v2)  

---


**ABSTRACT**  
Chain-of-Though (CoT) prompting has shown promising performance in various reasoning tasks. Recently, Self-Consistency \citep{wang2023selfconsistency} proposes to sample a diverse set of reasoning chains which may lead to different answers while the answer that receives the most votes is selected. In this paper, we propose a novel method to use backward reasoning in verifying candidate answers. We mask a token in the question by ${\bf x}$ and ask the LLM to predict the masked token when a candidate answer is provided by \textit{a simple template}, i.e., ``\textit{\textbf{If we know the answer of the above question is \{a candidate answer\}, what is the value of unknown variable ${\bf x}$?}}'' Intuitively, the LLM is expected to predict the masked token successfully if the provided candidate answer is correct. We further propose FOBAR to combine forward and backward reasoning for estimating the probability of candidate answers. We conduct extensive experiments on six data sets and three LLMs. Experimental results demonstrate that FOBAR achieves state-of-the-art performance on various reasoning benchmarks.

{{</citation>}}


### (37/100) Better Zero-Shot Reasoning with Role-Play Prompting (Aobo Kong et al., 2023)

{{<citation>}}

Aobo Kong, Shiwan Zhao, Hao Chen, Qicheng Li, Yong Qin, Ruiqi Sun, Xin Zhou. (2023)  
**Better Zero-Shot Reasoning with Role-Play Prompting**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Reasoning, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.07702v1)  

---


**ABSTRACT**  
Modern large language models (LLMs), such as ChatGPT, exhibit a remarkable capacity for role-playing, enabling them to embody not only human characters but also non-human entities like a Linux terminal. This versatility allows them to simulate complex human-like interactions and behaviors within various contexts, as well as to emulate specific objects or systems. While these capabilities have enhanced user engagement and introduced novel modes of interaction, the influence of role-playing on LLMs' reasoning abilities remains underexplored. In this study, we introduce a strategically designed role-play prompting methodology and assess its performance under the zero-shot setting across twelve diverse reasoning benchmarks, encompassing arithmetic, commonsense reasoning, symbolic reasoning, and more. Leveraging models such as ChatGPT and Llama 2, our empirical results illustrate that role-play prompting consistently surpasses the standard zero-shot approach across most datasets. Notably, accuracy on AQuA rises from 53.5% to 63.8%, and on Last Letter from 23.8% to 84.2%. Beyond enhancing contextual understanding, we posit that role-play prompting serves as an implicit Chain-of-Thought (CoT) trigger, thereby improving the quality of reasoning. By comparing our approach with the Zero-Shot-CoT technique, which prompts the model to "think step by step", we further demonstrate that role-play prompting can generate a more effective CoT. This highlights its potential to augment the reasoning capabilities of LLMs.

{{</citation>}}


### (38/100) Steering Language Generation: Harnessing Contrastive Expert Guidance and Negative Prompting for Coherent and Diverse Synthetic Data Generation (Charles O'Neill et al., 2023)

{{<citation>}}

Charles O'Neill, Yuan-Sen Ting, Ioana Ciuca, Jack Miller, Thang Bui. (2023)  
**Steering Language Generation: Harnessing Contrastive Expert Guidance and Negative Prompting for Coherent and Diverse Synthetic Data Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2308.07645v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) hold immense potential to generate synthetic data of high quality and utility, which has numerous applications from downstream model training to practical data utilisation. However, contemporary models, despite their impressive capacities, consistently struggle to produce both coherent and diverse data. To address the coherency issue, we introduce contrastive expert guidance, where the difference between the logit distributions of fine-tuned and base language models is emphasised to ensure domain adherence. In order to ensure diversity, we utilise existing real and synthetic examples as negative prompts to the model. We deem this dual-pronged approach to logit reshaping as STEER: Semantic Text Enhancement via Embedding Repositioning. STEER operates at inference-time and systematically guides the LLMs to strike a balance between adherence to the data distribution (ensuring semantic fidelity) and deviation from prior synthetic examples or existing real datasets (ensuring diversity and authenticity). This delicate balancing act is achieved by dynamically moving towards or away from chosen representations in the latent space. STEER demonstrates improved performance over previous synthetic data generation techniques, exhibiting better balance between data diversity and coherency across three distinct tasks: hypothesis generation, toxic and non-toxic comment generation, and commonsense reasoning task generation. We demonstrate how STEER allows for fine-tuned control over the diversity-coherency trade-off via its hyperparameters, highlighting its versatility.

{{</citation>}}


### (39/100) LLM-Mini-CEX: Automatic Evaluation of Large Language Model for Diagnostic Conversation (Xiaoming Shi et al., 2023)

{{<citation>}}

Xiaoming Shi, Jie Xu, Jinru Ding, Jiali Pang, Sichen Liu, Shuqing Luo, Xingwei Peng, Lu Lu, Haihong Yang, Mingtao Hu, Tong Ruan, Shaoting Zhang. (2023)  
**LLM-Mini-CEX: Automatic Evaluation of Large Language Model for Diagnostic Conversation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.07635v1)  

---


**ABSTRACT**  
There is an increasing interest in developing LLMs for medical diagnosis to improve diagnosis efficiency. Despite their alluring technological potential, there is no unified and comprehensive evaluation criterion, leading to the inability to evaluate the quality and potential risks of medical LLMs, further hindering the application of LLMs in medical treatment scenarios. Besides, current evaluations heavily rely on labor-intensive interactions with LLMs to obtain diagnostic dialogues and human evaluation on the quality of diagnosis dialogue. To tackle the lack of unified and comprehensive evaluation criterion, we first initially establish an evaluation criterion, termed LLM-specific Mini-CEX to assess the diagnostic capabilities of LLMs effectively, based on original Mini-CEX. To address the labor-intensive interaction problem, we develop a patient simulator to engage in automatic conversations with LLMs, and utilize ChatGPT for evaluating diagnosis dialogues automatically. Experimental results show that the LLM-specific Mini-CEX is adequate and necessary to evaluate medical diagnosis dialogue. Besides, ChatGPT can replace manual evaluation on the metrics of humanistic qualities and provides reproducible and automated comparisons between different LLMs.

{{</citation>}}


### (40/100) A Survey on Model Compression for Large Language Models (Xunyu Zhu et al., 2023)

{{<citation>}}

Xunyu Zhu, Jian Li, Yong Liu, Can Ma, Weiping Wang. (2023)  
**A Survey on Model Compression for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.07633v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have revolutionized natural language processing tasks with remarkable success. However, their formidable size and computational demands present significant challenges for practical deployment, especially in resource-constrained environments. As these challenges become increasingly pertinent, the field of model compression has emerged as a pivotal research area to alleviate these limitations. This paper presents a comprehensive survey that navigates the landscape of model compression techniques tailored specifically for LLMs. Addressing the imperative need for efficient deployment, we delve into various methodologies, encompassing quantization, pruning, knowledge distillation, and more. Within each of these techniques, we highlight recent advancements and innovative approaches that contribute to the evolving landscape of LLM research. Furthermore, we explore benchmarking strategies and evaluation metrics that are essential for assessing the effectiveness of compressed LLMs. By providing insights into the latest developments and practical implications, this survey serves as an invaluable resource for both researchers and practitioners. As LLMs continue to evolve, this survey aims to facilitate enhanced efficiency and real-world applicability, establishing a foundation for future advancements in the field.

{{</citation>}}


### (41/100) VBD-MT Chinese-Vietnamese Translation Systems for VLSP 2022 (Hai Long Trieu et al., 2023)

{{<citation>}}

Hai Long Trieu, Song Kiet Bui, Tan Minh Tran, Van Khanh Tran, Hai An Nguyen. (2023)  
**VBD-MT Chinese-Vietnamese Translation Systems for VLSP 2022**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Transformer  
[Paper Link](http://arxiv.org/abs/2308.07601v1)  

---


**ABSTRACT**  
We present our systems participated in the VLSP 2022 machine translation shared task. In the shared task this year, we participated in both translation tasks, i.e., Chinese-Vietnamese and Vietnamese-Chinese translations. We build our systems based on the neural-based Transformer model with the powerful multilingual denoising pre-trained model mBART. The systems are enhanced by a sampling method for backtranslation, which leverage large scale available monolingual data. Additionally, several other methods are applied to improve the translation quality including ensembling and postprocessing. We achieve 38.9 BLEU on ChineseVietnamese and 38.0 BLEU on VietnameseChinese on the public test sets, which outperform several strong baselines.

{{</citation>}}


### (42/100) CALYPSO: LLMs as Dungeon Masters' Assistants (Andrew Zhu et al., 2023)

{{<citation>}}

Andrew Zhu, Lara J. Martin, Andrew Head, Chris Callison-Burch. (2023)  
**CALYPSO: LLMs as Dungeon Masters' Assistants**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.07540v1)  

---


**ABSTRACT**  
The role of a Dungeon Master, or DM, in the game Dungeons & Dragons is to perform multiple tasks simultaneously. The DM must digest information about the game setting and monsters, synthesize scenes to present to other players, and respond to the players' interactions with the scene. Doing all of these tasks while maintaining consistency within the narrative and story world is no small feat of human cognition, making the task tiring and unapproachable to new players. Large language models (LLMs) like GPT-3 and ChatGPT have shown remarkable abilities to generate coherent natural language text. In this paper, we conduct a formative evaluation with DMs to establish the use cases of LLMs in D&D and tabletop gaming generally. We introduce CALYPSO, a system of LLM-powered interfaces that support DMs with information and inspiration specific to their own scenario. CALYPSO distills game context into bite-sized prose and helps brainstorm ideas without distracting the DM from the game. When given access to CALYPSO, DMs reported that it generated high-fidelity text suitable for direct presentation to players, and low-fidelity ideas that the DM could develop further while maintaining their creative agency. We see CALYPSO as exemplifying a paradigm of AI-augmented tools that provide synchronous creative assistance within established game worlds, and tabletop gaming more broadly.

{{</citation>}}


### (43/100) Finding Stakeholder-Material Information from 10-K Reports using Fine-Tuned BERT and LSTM Models (Victor Zitian Chen, 2023)

{{<citation>}}

Victor Zitian Chen. (2023)  
**Finding Stakeholder-Material Information from 10-K Reports using Fine-Tuned BERT and LSTM Models**  

---
Primary Category: cs.CL  
Categories: cs-CE, cs-CL, cs.CL  
Keywords: BERT, LSTM  
[Paper Link](http://arxiv.org/abs/2308.07522v1)  

---


**ABSTRACT**  
All public companies are required by federal securities law to disclose their business and financial activities in their annual 10-K reports. Each report typically spans hundreds of pages, making it difficult for human readers to identify and extract the material information efficiently. To solve the problem, I have fine-tuned BERT models and RNN models with LSTM layers to identify stakeholder-material information, defined as statements that carry information about a company's influence on its stakeholders, including customers, employees, investors, and the community and natural environment. The existing practice uses keyword search to identify such information, which is my baseline model. Using business expert-labeled training data of nearly 6,000 sentences from 62 10-K reports published in 2022, the best model has achieved an accuracy of 0.904 and an F1 score of 0.899 in test data, significantly above the baseline model's 0.781 and 0.749 respectively. Furthermore, the same work was replicated on more granular taxonomies, based on which four distinct groups of stakeholders (i.e., customers, investors, employees, and the community and natural environment) are tested separately. Similarly, fined-tuned BERT models outperformed LSTM and the baseline. The implications for industry application and ideas for future extensions are discussed.

{{</citation>}}


## cs.SE (4)



### (44/100) Automated Test Case Generation Using Code Models and Domain Adaptation (Sepehr Hashtroudi et al., 2023)

{{<citation>}}

Sepehr Hashtroudi, Jiho Shin, Hadi Hemmati, Song Wang. (2023)  
**Automated Test Case Generation Using Code Models and Domain Adaptation**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: T5, Transformer  
[Paper Link](http://arxiv.org/abs/2308.08033v1)  

---


**ABSTRACT**  
State-of-the-art automated test generation techniques, such as search-based testing, are usually ignorant about what a developer would create as a test case. Therefore, they typically create tests that are not human-readable and may not necessarily detect all types of complex bugs developer-written tests would do. In this study, we leverage Transformer-based code models to generate unit tests that can complement search-based test generation. Specifically, we use CodeT5, i.e., a state-of-the-art large code model, and fine-tune it on the test generation downstream task. For our analysis, we use the Methods2test dataset for fine-tuning CodeT5 and Defects4j for project-level domain adaptation and evaluation. The main contribution of this study is proposing a fully automated testing framework that leverages developer-written tests and available code models to generate compilable, human-readable unit tests. Results show that our approach can generate new test cases that cover lines that were not covered by developer-written tests. Using domain adaptation, we can also increase line coverage of the model-generated unit tests by 49.9% and 54% in terms of mean and median (compared to the model without domain adaptation). We can also use our framework as a complementary solution alongside common search-based methods to increase the overall coverage with mean and median of 25.3% and 6.3%. It can also increase the mutation score of search-based methods by killing extra mutants (up to 64 new mutants were killed per project in our experiments).

{{</citation>}}


### (45/100) Large Language Models in Introductory Programming Education: ChatGPT's Performance and Implications for Assessments (Natalie Kiesler et al., 2023)

{{<citation>}}

Natalie Kiesler, Daniel Schiffner. (2023)  
**Large Language Models in Introductory Programming Education: ChatGPT's Performance and Implications for Assessments**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-HC, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2308.08572v1)  

---


**ABSTRACT**  
This paper investigates the performance of the Large Language Models (LLMs) ChatGPT-3.5 and GPT-4 in solving introductory programming tasks. Based on the performance, implications for didactic scenarios and assessment formats utilizing LLMs are derived. For the analysis, 72 Python tasks for novice programmers were selected from the free site CodingBat. Full task descriptions were used as input to the LLMs, while the generated replies were evaluated using CodingBat's unit tests. In addition, the general availability of textual explanations and program code was analyzed. The results show high scores of 94.4 to 95.8% correct responses and reliable availability of textual explanations and program code, which opens new ways to incorporate LLMs into programming education and assessment.

{{</citation>}}


### (46/100) From Commit Message Generation to History-Aware Commit Message Completion (Aleksandra Eliseeva et al., 2023)

{{<citation>}}

Aleksandra Eliseeva, Yaroslav Sokolov, Egor Bogomolov, Yaroslav Golubev, Danny Dig, Timofey Bryksin. (2023)  
**From Commit Message Generation to History-Aware Commit Message Completion**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2308.07655v1)  

---


**ABSTRACT**  
Commit messages are crucial to software development, allowing developers to track changes and collaborate effectively. Despite their utility, most commit messages lack important information since writing high-quality commit messages is tedious and time-consuming. The active research on commit message generation (CMG) has not yet led to wide adoption in practice. We argue that if we could shift the focus from commit message generation to commit message completion and use previous commit history as additional context, we could significantly improve the quality and the personal nature of the resulting commit messages.   In this paper, we propose and evaluate both of these novel ideas. Since the existing datasets lack historical data, we collect and share a novel dataset called CommitChronicle, containing 10.7M commits across 20 programming languages. We use this dataset to evaluate the completion setting and the usefulness of the historical context for state-of-the-art CMG models and GPT-3.5-turbo. Our results show that in some contexts, commit message completion shows better results than generation, and that while in general GPT-3.5-turbo performs worse, it shows potential for long and detailed messages. As for the history, the results show that historical information improves the performance of CMG models in the generation task, and the performance of GPT-3.5-turbo in both generation and completion.

{{</citation>}}


### (47/100) LogPrompt: Prompt Engineering Towards Zero-Shot and Interpretable Log Analysis (Yilun Liu et al., 2023)

{{<citation>}}

Yilun Liu, Shimin Tao, Weibin Meng, Jingyu Wang, Wenbing Ma, Yanqing Zhao, Yuhang Chen, Hao Yang, Yanfei Jiang, Xun Chen. (2023)  
**LogPrompt: Prompt Engineering Towards Zero-Shot and Interpretable Log Analysis**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.07610v1)  

---


**ABSTRACT**  
Automated log analysis is crucial in modern software-intensive systems for ensuring reliability and resilience throughout software maintenance and engineering life cycles. Existing methods perform tasks such as log parsing and log anomaly detection by providing a single prediction value without interpretation. However, given the increasing volume of system events, the limited interpretability of analysis results hinders analysts' trust and their ability to take appropriate actions. Moreover, these methods require substantial in-domain training data, and their performance declines sharply (by up to 62.5%) in online scenarios involving unseen logs from new domains, a common occurrence due to rapid software updates. In this paper, we propose LogPrompt, a novel zero-shot and interpretable log analysis approach. LogPrompt employs large language models (LLMs) to perform zero-shot log analysis tasks via a suite of advanced prompt strategies tailored for log tasks, which enhances LLMs' performance by up to 107.5% compared with simple prompts. Experiments on nine publicly available evaluation datasets across two tasks demonstrate that LogPrompt, despite using no training data, outperforms existing approaches trained on thousands of logs by up to around 50%. We also conduct a human evaluation of LogPrompt's interpretability, with six practitioners possessing over 10 years of experience, who highly rated the generated content in terms of usefulness and readability (averagely 4.42/5). LogPrompt also exhibits remarkable compatibility with open-source and smaller-scale LLMs, making it flexible for practical deployment.

{{</citation>}}


## cs.AI (4)



### (48/100) Planning to Learn: A Novel Algorithm for Active Learning during Model-Based Planning (Rowan Hodson et al., 2023)

{{<citation>}}

Rowan Hodson, Bruce Bassett, Charel van Hoof, Benjamin Rosman, Mark Solms, Jonathan P. Shock, Ryan Smith. (2023)  
**Planning to Learn: A Novel Algorithm for Active Learning during Model-Based Planning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI, q-bio-NC  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2308.08029v1)  

---


**ABSTRACT**  
Active Inference is a recent framework for modeling planning under uncertainty. Empirical and theoretical work have now begun to evaluate the strengths and weaknesses of this approach and how it might be improved. A recent extension - the sophisticated inference (SI) algorithm - improves performance on multi-step planning problems through recursive decision tree search. However, little work to date has been done to compare SI to other established planning algorithms. SI was also developed with a focus on inference as opposed to learning. The present paper has two aims. First, we compare performance of SI to Bayesian reinforcement learning (RL) schemes designed to solve similar problems. Second, we present an extension of SI - sophisticated learning (SL) - that more fully incorporates active learning during planning. SL maintains beliefs about how model parameters would change under the future observations expected under each policy. This allows a form of counterfactual retrospective inference in which the agent considers what could be learned from current or past observations given different future observations. To accomplish these aims, we make use of a novel, biologically inspired environment designed to highlight the problem structure for which SL offers a unique solution. Here, an agent must continually search for available (but changing) resources in the presence of competing affordances for information gain. Our simulations show that SL outperforms all other algorithms in this context - most notably, Bayes-adaptive RL and upper confidence bound algorithms, which aim to solve multi-step planning problems using similar principles (i.e., directed exploration and counterfactual reasoning). These results provide added support for the utility of Active Inference in solving this class of biologically-relevant problems and offer added tools for testing hypotheses about human cognition.

{{</citation>}}


### (49/100) A Comprehensive Study on Knowledge Graph Embedding over Relational Patterns Based on Rule Learning (Long Jin et al., 2023)

{{<citation>}}

Long Jin, Zhen Yao, Mingyang Chen, Huajun Chen, Wen Zhang. (2023)  
**A Comprehensive Study on Knowledge Graph Embedding over Relational Patterns Based on Rule Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.07889v1)  

---


**ABSTRACT**  
Knowledge Graph Embedding (KGE) has proven to be an effective approach to solving the Knowledge Graph Completion (KGC) task. Relational patterns which refer to relations with specific semantics exhibiting graph patterns are an important factor in the performance of KGE models. Though KGE models' capabilities are analyzed over different relational patterns in theory and a rough connection between better relational patterns modeling and better performance of KGC has been built, a comprehensive quantitative analysis on KGE models over relational patterns remains absent so it is uncertain how the theoretical support of KGE to a relational pattern contributes to the performance of triples associated to such a relational pattern. To address this challenge, we evaluate the performance of 7 KGE models over 4 common relational patterns on 2 benchmarks, then conduct an analysis in theory, entity frequency, and part-to-whole three aspects and get some counterintuitive conclusions. Finally, we introduce a training-free method Score-based Patterns Adaptation (SPA) to enhance KGE models' performance over various relational patterns. This approach is simple yet effective and can be applied to KGE models without additional training. Our experimental results demonstrate that our method generally enhances performance over specific relational patterns. Our source code is available from GitHub at https://github.com/zjukg/Comprehensive-Study-over-Relational-Patterns.

{{</citation>}}


### (50/100) Brain-Inspired Computational Intelligence via Predictive Coding (Tommaso Salvatori et al., 2023)

{{<citation>}}

Tommaso Salvatori, Ankur Mali, Christopher L. Buckley, Thomas Lukasiewicz, Rajesh P. N. Rao, Karl Friston, Alexander Ororbia. (2023)  
**Brain-Inspired Computational Intelligence via Predictive Coding**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-NE, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.07870v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) is rapidly becoming one of the key technologies of this century. The majority of results in AI thus far have been achieved using deep neural networks trained with the error backpropagation learning algorithm. However, the ubiquitous adoption of this approach has highlighted some important limitations such as substantial computational cost, difficulty in quantifying uncertainty, lack of robustness, unreliability, and biological implausibility. It is possible that addressing these limitations may require schemes that are inspired and guided by neuroscience theories. One such theory, called predictive coding (PC), has shown promising performance in machine intelligence tasks, exhibiting exciting properties that make it potentially valuable for the machine learning community: PC can model information processing in different brain areas, can be used in cognitive control and robotics, and has a solid mathematical grounding in variational inference, offering a powerful inversion scheme for a specific class of continuous-state generative models. With the hope of foregrounding research in this direction, we survey the literature that has contributed to this perspective, highlighting the many ways that PC might play a role in the future of machine learning and computational intelligence at large.

{{</citation>}}


### (51/100) Do We Fully Understand Students' Knowledge States? Identifying and Mitigating Answer Bias in Knowledge Tracing (Chaoran Cui et al., 2023)

{{<citation>}}

Chaoran Cui, Hebo Ma, Chen Zhang, Chunyun Zhang, Yumo Yao, Meng Chen, Yuling Ma. (2023)  
**Do We Fully Understand Students' Knowledge States? Identifying and Mitigating Answer Bias in Knowledge Tracing**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.07779v1)  

---


**ABSTRACT**  
Knowledge tracing (KT) aims to monitor students' evolving knowledge states through their learning interactions with concept-related questions, and can be indirectly evaluated by predicting how students will perform on future questions. In this paper, we observe that there is a common phenomenon of answer bias, i.e., a highly unbalanced distribution of correct and incorrect answers for each question. Existing models tend to memorize the answer bias as a shortcut for achieving high prediction performance in KT, thereby failing to fully understand students' knowledge states. To address this issue, we approach the KT task from a causality perspective. A causal graph of KT is first established, from which we identify that the impact of answer bias lies in the direct causal effect of questions on students' responses. A novel COunterfactual REasoning (CORE) framework for KT is further proposed, which separately captures the total causal effect and direct causal effect during training, and mitigates answer bias by subtracting the latter from the former in testing. The CORE framework is applicable to various existing KT models, and we implement it based on the prevailing DKT, DKVMN, and AKT models, respectively. Extensive experiments on three benchmark datasets demonstrate the effectiveness of CORE in making the debiased inference for KT.

{{</citation>}}


## eess.AS (1)



### (52/100) End-to-End Open Vocabulary Keyword Search With Multilingual Neural Representations (Bolaji Yusuf et al., 2023)

{{<citation>}}

Bolaji Yusuf, Jan Cernocky, Murat Saraclar. (2023)  
**End-to-End Open Vocabulary Keyword Search With Multilingual Neural Representations**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2308.08027v1)  

---


**ABSTRACT**  
Conventional keyword search systems operate on automatic speech recognition (ASR) outputs, which causes them to have a complex indexing and search pipeline. This has led to interest in ASR-free approaches to simplify the search procedure. We recently proposed a neural ASR-free keyword search model which achieves competitive performance while maintaining an efficient and simplified pipeline, where queries and documents are encoded with a pair of recurrent neural network encoders and the encodings are combined with a dot-product. In this article, we extend this work with multilingual pretraining and detailed analysis of the model. Our experiments show that the proposed multilingual training significantly improves the model performance and that despite not matching a strong ASR-based conventional keyword search system for short queries and queries comprising in-vocabulary words, the proposed model outperforms the ASR-based system for long queries and queries that do not appear in the training data.

{{</citation>}}


## cs.HC (2)



### (53/100) BI-LAVA: Biocuration with Hierarchical Image Labeling through Active Learning and Visual Analysis (Juan Trelles et al., 2023)

{{<citation>}}

Juan Trelles, Andrew Wentzel, William Berrios, G. Elisabeta Marai. (2023)  
**BI-LAVA: Biocuration with Hierarchical Image Labeling through Active Learning and Visual Analysis**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-LG, cs.HC  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2308.08003v1)  

---


**ABSTRACT**  
In the biomedical domain, taxonomies organize the acquisition modalities of scientific images in hierarchical structures. Such taxonomies leverage large sets of correct image labels and provide essential information about the importance of a scientific publication, which could then be used in biocuration tasks. However, the hierarchical nature of the labels, the overhead of processing images, the absence or incompleteness of labeled data, and the expertise required to label this type of data impede the creation of useful datasets for biocuration. From a multi-year collaboration with biocurators and text-mining researchers, we derive an iterative visual analytics and active learning strategy to address these challenges. We implement this strategy in a system called BI-LAVA Biocuration with Hierarchical Image Labeling through Active Learning and Visual Analysis. BI-LAVA leverages a small set of image labels, a hierarchical set of image classifiers, and active learning to help model builders deal with incomplete ground-truth labels, target a hierarchical taxonomy of image modalities, and classify a large pool of unlabeled images. BI-LAVA's front end uses custom encodings to represent data distributions, taxonomies, image projections, and neighborhoods of image thumbnails, which help model builders explore an unfamiliar image dataset and taxonomy and correct and generate labels. An evaluation with machine learning practitioners shows that our mixed human-machine approach successfully supports domain experts in understanding the characteristics of classes within the taxonomy, as well as validating and improving data quality in labeled and unlabeled collections.

{{</citation>}}


### (54/100) ReaderQuizzer: Augmenting Research Papers with Just-In-Time Learning Questions to Facilitate Deeper Understanding (Liam Richards Maldonado, 2023)

{{<citation>}}

Liam Richards Maldonado. (2023)  
**ReaderQuizzer: Augmenting Research Papers with Just-In-Time Learning Questions to Facilitate Deeper Understanding**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.07988v1)  

---


**ABSTRACT**  
Academic reading is a key component of higher education, and serves as a basis for critical thinking, knowledge acquisition and effective communication. Research shows many students struggle with comprehension and analysis tasks with academic texts, despite the central importance of academic reading to success in higher education. Undergraduates and researchers need to internalize dense literature to scaffold their own work upon it. This reading task is time-consuming and difficult to do. Oftentimes, students struggle to actively and critically engage and as a result attain merely a cursory understanding of a paper's contents, or worse, incorrectly interpret the text. How, then, can we provide a means to more easily digest a text whilst also facilitating meaningful, critical engagement and understanding? This paper locates itself within the broader field of Human-Computer Interaction (HCI) to implement an augmented reading interface that leverages the power of ChatGPT to intelligently generate and co-locate comprehension and analysis questions in an academic paper, thereby making the paper more digestible with the end goal of facilitating deeper understanding, and developing critical reading skills.

{{</citation>}}


## cs.CV (34)



### (55/100) $A^2$Nav: Action-Aware Zero-Shot Robot Navigation by Exploiting Vision-and-Language Ability of Foundation Models (Peihao Chen et al., 2023)

{{<citation>}}

Peihao Chen, Xinyu Sun, Hongyan Zhi, Runhao Zeng, Thomas H. Li, Gaowen Liu, Mingkui Tan, Chuang Gan. (2023)  
**$A^2$Nav: Action-Aware Zero-Shot Robot Navigation by Exploiting Vision-and-Language Ability of Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: GPT, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2308.07997v1)  

---


**ABSTRACT**  
We study the task of zero-shot vision-and-language navigation (ZS-VLN), a practical yet challenging problem in which an agent learns to navigate following a path described by language instructions without requiring any path-instruction annotation data. Normally, the instructions have complex grammatical structures and often contain various action descriptions (e.g., "proceed beyond", "depart from"). How to correctly understand and execute these action demands is a critical problem, and the absence of annotated data makes it even more challenging. Note that a well-educated human being can easily understand path instructions without the need for any special training. In this paper, we propose an action-aware zero-shot VLN method ($A^2$Nav) by exploiting the vision-and-language ability of foundation models. Specifically, the proposed method consists of an instruction parser and an action-aware navigation policy. The instruction parser utilizes the advanced reasoning ability of large language models (e.g., GPT-3) to decompose complex navigation instructions into a sequence of action-specific object navigation sub-tasks. Each sub-task requires the agent to localize the object and navigate to a specific goal position according to the associated action demand. To accomplish these sub-tasks, an action-aware navigation policy is learned from freely collected action-specific datasets that reveal distinct characteristics of each action demand. We use the learned navigation policy for executing sub-tasks sequentially to follow the navigation instruction. Extensive experiments show $A^2$Nav achieves promising ZS-VLN performance and even surpasses the supervised learning methods on R2R-Habitat and RxR-Habitat datasets.

{{</citation>}}


### (56/100) A Foundation LAnguage-Image model of the Retina (FLAIR): Encoding expert knowledge in text supervision (Julio Silva-Rodriguez et al., 2023)

{{<citation>}}

Julio Silva-Rodriguez, Hadi Chakor, Riadh Kobbi, Jose Dolz, Ismail Ben Ayed. (2023)  
**A Foundation LAnguage-Image model of the Retina (FLAIR): Encoding expert knowledge in text supervision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.07898v1)  

---


**ABSTRACT**  
Foundation vision-language models are currently transforming computer vision, and are on the rise in medical imaging fueled by their very promising generalization capabilities. However, the initial attempts to transfer this new paradigm to medical imaging have shown less impressive performances than those observed in other domains, due to the significant domain shift and the complex, expert domain knowledge inherent to medical-imaging tasks. Motivated by the need for domain-expert foundation models, we present FLAIR, a pre-trained vision-language model for universal retinal fundus image understanding. To this end, we compiled 37 open-access, mostly categorical fundus imaging datasets from various sources, with up to 97 different target conditions and 284,660 images. We integrate the expert's domain knowledge in the form of descriptive textual prompts, during both pre-training and zero-shot inference, enhancing the less-informative categorical supervision of the data. Such a textual expert's knowledge, which we compiled from the relevant clinical literature and community standards, describes the fine-grained features of the pathologies as well as the hierarchies and dependencies between them. We report comprehensive evaluations, which illustrate the benefit of integrating expert knowledge and the strong generalization capabilities of FLAIR under difficult scenarios with domain shifts or unseen categories. When adapted with a lightweight linear probe, FLAIR outperforms fully-trained, dataset-focused models, more so in the few-shot regimes. Interestingly, FLAIR outperforms by a large margin more generalist, larger-scale image-language models, which emphasizes the potential of embedding experts' domain knowledge and the limitations of generalist models in medical imaging.

{{</citation>}}


### (57/100) Memory-and-Anticipation Transformer for Online Action Understanding (Jiahao Wang et al., 2023)

{{<citation>}}

Jiahao Wang, Guo Chen, Yifei Huang, Limin Wang, Tong Lu. (2023)  
**Memory-and-Anticipation Transformer for Online Action Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.07893v1)  

---


**ABSTRACT**  
Most existing forecasting systems are memory-based methods, which attempt to mimic human forecasting ability by employing various memory mechanisms and have progressed in temporal modeling for memory dependency. Nevertheless, an obvious weakness of this paradigm is that it can only model limited historical dependence and can not transcend the past. In this paper, we rethink the temporal dependence of event evolution and propose a novel memory-anticipation-based paradigm to model an entire temporal structure, including the past, present, and future. Based on this idea, we present Memory-and-Anticipation Transformer (MAT), a memory-anticipation-based approach, to address the online action detection and anticipation tasks. In addition, owing to the inherent superiority of MAT, it can process online action detection and anticipation tasks in a unified manner. The proposed MAT model is tested on four challenging benchmarks TVSeries, THUMOS'14, HDD, and EPIC-Kitchens-100, for online action detection and anticipation tasks, and it significantly outperforms all existing methods. Code is available at https://github.com/Echo0125/Memory-and-Anticipation-Transformer.

{{</citation>}}


### (58/100) Link-Context Learning for Multimodal LLMs (Yan Tai et al., 2023)

{{<citation>}}

Yan Tai, Weichen Fan, Zhao Zhang, Feng Zhu, Rui Zhao, Ziwei Liu. (2023)  
**Link-Context Learning for Multimodal LLMs**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.07891v1)  

---


**ABSTRACT**  
The ability to learn from context with novel concepts, and deliver appropriate responses are essential in human conversations. Despite current Multimodal Large Language Models (MLLMs) and Large Language Models (LLMs) being trained on mega-scale datasets, recognizing unseen images or understanding novel concepts in a training-free manner remains a challenge. In-Context Learning (ICL) explores training-free few-shot learning, where models are encouraged to ``learn to learn" from limited tasks and generalize to unseen tasks. In this work, we propose link-context learning (LCL), which emphasizes "reasoning from cause and effect" to augment the learning capabilities of MLLMs. LCL goes beyond traditional ICL by explicitly strengthening the causal relationship between the support set and the query set. By providing demonstrations with causal links, LCL guides the model to discern not only the analogy but also the underlying causal associations between data points, which empowers MLLMs to recognize unseen images and understand novel concepts more effectively. To facilitate the evaluation of this novel approach, we introduce the ISEKAI dataset, comprising exclusively of unseen generated image-label pairs designed for link-context learning. Extensive experiments show that our LCL-MLLM exhibits strong link-context learning capabilities to novel concepts over vanilla MLLMs. Code and data will be released at https://github.com/isekai-portal/Link-Context-Learning.

{{</citation>}}


### (59/100) SEDA: Self-Ensembling ViT with Defensive Distillation and Adversarial Training for robust Chest X-rays Classification (Raza Imam et al., 2023)

{{<citation>}}

Raza Imam, Ibrahim Almakky, Salma Alrashdi, Baketah Alrashdi, Mohammad Yaqub. (2023)  
**SEDA: Self-Ensembling ViT with Defensive Distillation and Adversarial Training for robust Chest X-rays Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Training, Transformer  
[Paper Link](http://arxiv.org/abs/2308.07874v1)  

---


**ABSTRACT**  
Deep Learning methods have recently seen increased adoption in medical imaging applications. However, elevated vulnerabilities have been explored in recent Deep Learning solutions, which can hinder future adoption. Particularly, the vulnerability of Vision Transformer (ViT) to adversarial, privacy, and confidentiality attacks raise serious concerns about their reliability in medical settings. This work aims to enhance the robustness of self-ensembling ViTs for the tuberculosis chest x-ray classification task. We propose Self-Ensembling ViT with defensive Distillation and Adversarial training (SEDA). SEDA utilizes efficient CNN blocks to learn spatial features with various levels of abstraction from feature representations extracted from intermediate ViT blocks, that are largely unaffected by adversarial perturbations. Furthermore, SEDA leverages adversarial training in combination with defensive distillation for improved robustness against adversaries. Training using adversarial examples leads to better model generalizability and improves its ability to handle perturbations. Distillation using soft probabilities introduces uncertainty and variation into the output probabilities, making it more difficult for adversarial and privacy attacks. Extensive experiments performed with the proposed architecture and training paradigm on publicly available Tuberculosis x-ray dataset shows SOTA efficacy of SEDA compared to SEViT in terms of computational efficiency with 70x times lighter framework and enhanced robustness of +9%.

{{</citation>}}


### (60/100) StyleDiffusion: Controllable Disentangled Style Transfer via Diffusion Models (Zhizhong Wang et al., 2023)

{{<citation>}}

Zhizhong Wang, Lei Zhao, Wei Xing. (2023)  
**StyleDiffusion: Controllable Disentangled Style Transfer via Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2308.07863v1)  

---


**ABSTRACT**  
Content and style (C-S) disentanglement is a fundamental problem and critical challenge of style transfer. Existing approaches based on explicit definitions (e.g., Gram matrix) or implicit learning (e.g., GANs) are neither interpretable nor easy to control, resulting in entangled representations and less satisfying results. In this paper, we propose a new C-S disentangled framework for style transfer without using previous assumptions. The key insight is to explicitly extract the content information and implicitly learn the complementary style information, yielding interpretable and controllable C-S disentanglement and style transfer. A simple yet effective CLIP-based style disentanglement loss coordinated with a style reconstruction prior is introduced to disentangle C-S in the CLIP image space. By further leveraging the powerful style removal and generative ability of diffusion models, our framework achieves superior results than state of the art and flexible C-S disentanglement and trade-off control. Our work provides new insights into the C-S disentanglement in style transfer and demonstrates the potential of diffusion models for learning well-disentangled C-S characteristics.

{{</citation>}}


### (61/100) Learning to Identify Critical States for Reinforcement Learning from Videos (Haozhe Liu et al., 2023)

{{<citation>}}

Haozhe Liu, Mingchen Zhuge, Bing Li, Yuhui Wang, Francesco Faccio, Bernard Ghanem, Jürgen Schmidhuber. (2023)  
**Learning to Identify Critical States for Reinforcement Learning from Videos**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.07795v1)  

---


**ABSTRACT**  
Recent work on deep reinforcement learning (DRL) has pointed out that algorithmic information about good policies can be extracted from offline data which lack explicit information about executed actions. For example, videos of humans or robots may convey a lot of implicit information about rewarding action sequences, but a DRL machine that wants to profit from watching such videos must first learn by itself to identify and recognize relevant states/actions/rewards. Without relying on ground-truth annotations, our new method called Deep State Identifier learns to predict returns from episodes encoded as videos. Then it uses a kind of mask-based sensitivity analysis to extract/identify important critical states. Extensive experiments showcase our method's potential for understanding and improving agent behavior. The source code and the generated datasets are available at https://github.com/AI-Initiative-KAUST/VideoRLCS.

{{</citation>}}


### (62/100) Future Video Prediction from a Single Frame for Video Anomaly Detection (Mohammad Baradaran et al., 2023)

{{<citation>}}

Mohammad Baradaran, Robert Bergevin. (2023)  
**Future Video Prediction from a Single Frame for Video Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.07783v1)  

---


**ABSTRACT**  
Video anomaly detection (VAD) is an important but challenging task in computer vision. The main challenge rises due to the rarity of training samples to model all anomaly cases. Hence, semi-supervised anomaly detection methods have gotten more attention, since they focus on modeling normals and they detect anomalies by measuring the deviations from normal patterns. Despite impressive advances of these methods in modeling normal motion and appearance, long-term motion modeling has not been effectively explored so far. Inspired by the abilities of the future frame prediction proxy-task, we introduce the task of future video prediction from a single frame, as a novel proxy-task for video anomaly detection. This proxy-task alleviates the challenges of previous methods in learning longer motion patterns. Moreover, we replace the initial and future raw frames with their corresponding semantic segmentation map, which not only makes the method aware of object class but also makes the prediction task less complex for the model. Extensive experiments on the benchmark datasets (ShanghaiTech, UCSD-Ped1, and UCSD-Ped2) show the effectiveness of the method and the superiority of its performance compared to SOTA prediction-based VAD methods.

{{</citation>}}


### (63/100) Learning Image Deraining Transformer Network with Dynamic Dual Self-Attention (Zhentao Fan et al., 2023)

{{<citation>}}

Zhentao Fan, Hongming Chen, Yufeng Li. (2023)  
**Learning Image Deraining Transformer Network with Dynamic Dual Self-Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Self-Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.07781v1)  

---


**ABSTRACT**  
Recently, Transformer-based architecture has been introduced into single image deraining task due to its advantage in modeling non-local information. However, existing approaches tend to integrate global features based on a dense self-attention strategy since it tend to uses all similarities of the tokens between the queries and keys. In fact, this strategy leads to ignoring the most relevant information and inducing blurry effect by the irrelevant representations during the feature aggregation. To this end, this paper proposes an effective image deraining Transformer with dynamic dual self-attention (DDSA), which combines both dense and sparse attention strategies to better facilitate clear image reconstruction. Specifically, we only select the most useful similarity values based on top-k approximate calculation to achieve sparse attention. In addition, we also develop a novel spatial-enhanced feed-forward network (SEFN) to further obtain a more accurate representation for achieving high-quality derained results. Extensive experiments on benchmark datasets demonstrate the effectiveness of our proposed method.

{{</citation>}}


### (64/100) Dual-path TokenLearner for Remote Photoplethysmography-based Physiological Measurement with Facial Videos (Wei Qian et al., 2023)

{{<citation>}}

Wei Qian, Dan Guo, Kun Li, Xilan Tian, Meng Wang. (2023)  
**Dual-path TokenLearner for Remote Photoplethysmography-based Physiological Measurement with Facial Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.07771v1)  

---


**ABSTRACT**  
Remote photoplethysmography (rPPG) based physiological measurement is an emerging yet crucial vision task, whose challenge lies in exploring accurate rPPG prediction from facial videos accompanied by noises of illumination variations, facial occlusions, head movements, \etc, in a non-contact manner. Existing mainstream CNN-based models make efforts to detect physiological signals by capturing subtle color changes in facial regions of interest (ROI) caused by heartbeats. However, such models are constrained by the limited local spatial or temporal receptive fields in the neural units. Unlike them, a native Transformer-based framework called Dual-path TokenLearner (Dual-TL) is proposed in this paper, which utilizes the concept of learnable tokens to integrate both spatial and temporal informative contexts from the global perspective of the video. Specifically, the proposed Dual-TL uses a Spatial TokenLearner (S-TL) to explore associations in different facial ROIs, which promises the rPPG prediction far away from noisy ROI disturbances. Complementarily, a Temporal TokenLearner (T-TL) is designed to infer the quasi-periodic pattern of heartbeats, which eliminates temporal disturbances such as head movements. The two TokenLearners, S-TL and T-TL, are executed in a dual-path mode. This enables the model to reduce noise disturbances for final rPPG signal prediction. Extensive experiments on four physiological measurement benchmark datasets are conducted. The Dual-TL achieves state-of-the-art performances in both intra- and cross-dataset testings, demonstrating its immense potential as a basic backbone for rPPG measurement. The source code is available at \href{https://github.com/VUT-HFUT/Dual-TL}{https://github.com/VUT-HFUT/Dual-TL}

{{</citation>}}


### (65/100) Whale Detection Enhancement through Synthetic Satellite Images (Akshaj Gaur et al., 2023)

{{<citation>}}

Akshaj Gaur, Cheng Liu, Xiaomin Lin, Nare Karapetyan, Yiannis Aloimonos. (2023)  
**Whale Detection Enhancement through Synthetic Satellite Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: AI, Drone  
[Paper Link](http://arxiv.org/abs/2308.07766v1)  

---


**ABSTRACT**  
With a number of marine populations in rapid decline, collecting and analyzing data about marine populations has become increasingly important to develop effective conservation policies for a wide range of marine animals, including whales. Modern computer vision algorithms allow us to detect whales in images in a wide range of domains, further speeding up and enhancing the monitoring process. However, these algorithms heavily rely on large training datasets, which are challenging and time-consuming to collect particularly in marine or aquatic environments. Recent advances in AI however have made it possible to synthetically create datasets for training machine learning algorithms, thus enabling new solutions that were not possible before. In this work, we present a solution - SeaDroneSim2 benchmark suite, which addresses this challenge by generating aerial, and satellite synthetic image datasets to improve the detection of whales and reduce the effort required for training data collection. We show that we can achieve a 15% performance boost on whale detection compared to using the real data alone for training, by augmenting a 10% real data. We open source both the code of the simulation platform SeaDroneSim2 and the dataset generated through it.

{{</citation>}}


### (66/100) Dancing Avatar: Pose and Text-Guided Human Motion Videos Synthesis with Image Diffusion Model (Bosheng Qin et al., 2023)

{{<citation>}}

Bosheng Qin, Wentao Ye, Qifan Yu, Siliang Tang, Yueting Zhuang. (2023)  
**Dancing Avatar: Pose and Text-Guided Human Motion Videos Synthesis with Image Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.07749v1)  

---


**ABSTRACT**  
The rising demand for creating lifelike avatars in the digital realm has led to an increased need for generating high-quality human videos guided by textual descriptions and poses. We propose Dancing Avatar, designed to fabricate human motion videos driven by poses and textual cues. Our approach employs a pretrained T2I diffusion model to generate each video frame in an autoregressive fashion. The crux of innovation lies in our adept utilization of the T2I diffusion model for producing video frames successively while preserving contextual relevance. We surmount the hurdles posed by maintaining human character and clothing consistency across varying poses, along with upholding the background's continuity amidst diverse human movements. To ensure consistent human appearances across the entire video, we devise an intra-frame alignment module. This module assimilates text-guided synthesized human character knowledge into the pretrained T2I diffusion model, synergizing insights from ChatGPT. For preserving background continuity, we put forth a background alignment pipeline, amalgamating insights from segment anything and image inpainting techniques. Furthermore, we propose an inter-frame alignment module that draws inspiration from an auto-regressive pipeline to augment temporal consistency between adjacent frames, where the preceding frame guides the synthesis process of the current frame. Comparisons with state-of-the-art methods demonstrate that Dancing Avatar exhibits the capacity to generate human videos with markedly superior quality, both in terms of human and background fidelity, as well as temporal coherence compared to existing state-of-the-art approaches.

{{</citation>}}


### (67/100) Exploiting Sparsity in Automotive Radar Object Detection Networks (Marius Lippke et al., 2023)

{{<citation>}}

Marius Lippke, Maurice Quach, Sascha Braun, Daniel Köhler, Michael Ulrich, Bastian Bischoff, Wei Yap Tan. (2023)  
**Exploiting Sparsity in Automotive Radar Object Detection Networks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.07748v1)  

---


**ABSTRACT**  
Having precise perception of the environment is crucial for ensuring the secure and reliable functioning of autonomous driving systems. Radar object detection networks are one fundamental part of such systems. CNN-based object detectors showed good performance in this context, but they require large compute resources. This paper investigates sparse convolutional object detection networks, which combine powerful grid-based detection with low compute resources. We investigate radar specific challenges and propose sparse kernel point pillars (SKPP) and dual voxel point convolutions (DVPC) as remedies for the grid rendering and sparse backbone architectures. We evaluate our SKPP-DPVCN architecture on nuScenes, which outperforms the baseline by 5.89% and the previous state of the art by 4.19% in Car AP4.0. Moreover, SKPP-DPVCN reduces the average scale error (ASE) by 21.41% over the baseline.

{{</citation>}}


### (68/100) Identity-Consistent Aggregation for Video Object Detection (Chaorui Deng et al., 2023)

{{<citation>}}

Chaorui Deng, Da Chen, Qi Wu. (2023)  
**Identity-Consistent Aggregation for Video Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Object Detection  
[Paper Link](http://arxiv.org/abs/2308.07737v1)  

---


**ABSTRACT**  
In Video Object Detection (VID), a common practice is to leverage the rich temporal contexts from the video to enhance the object representations in each frame. Existing methods treat the temporal contexts obtained from different objects indiscriminately and ignore their different identities. While intuitively, aggregating local views of the same object in different frames may facilitate a better understanding of the object. Thus, in this paper, we aim to enable the model to focus on the identity-consistent temporal contexts of each object to obtain more comprehensive object representations and handle the rapid object appearance variations such as occlusion, motion blur, etc. However, realizing this goal on top of existing VID models faces low-efficiency problems due to their redundant region proposals and nonparallel frame-wise prediction manner. To aid this, we propose ClipVID, a VID model equipped with Identity-Consistent Aggregation (ICA) layers specifically designed for mining fine-grained and identity-consistent temporal contexts. It effectively reduces the redundancies through the set prediction strategy, making the ICA layers very efficient and further allowing us to design an architecture that makes parallel clip-wise predictions for the whole video clip. Extensive experimental results demonstrate the superiority of our method: a state-of-the-art (SOTA) performance (84.7% mAP) on the ImageNet VID dataset while running at a speed about 7x faster (39.3 fps) than previous SOTAs.

{{</citation>}}


### (69/100) UniTR: A Unified and Efficient Multi-Modal Transformer for Bird's-Eye-View Representation (Haiyang Wang et al., 2023)

{{<citation>}}

Haiyang Wang, Hao Tang, Shaoshuai Shi, Aoxue Li, Zhenguo Li, Bernt Schiele, Liwei Wang. (2023)  
**UniTR: A Unified and Efficient Multi-Modal Transformer for Bird's-Eye-View Representation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.07732v1)  

---


**ABSTRACT**  
Jointly processing information from multiple sensors is crucial to achieving accurate and robust perception for reliable autonomous driving systems. However, current 3D perception research follows a modality-specific paradigm, leading to additional computation overheads and inefficient collaboration between different sensor data. In this paper, we present an efficient multi-modal backbone for outdoor 3D perception named UniTR, which processes a variety of modalities with unified modeling and shared parameters. Unlike previous works, UniTR introduces a modality-agnostic transformer encoder to handle these view-discrepant sensor data for parallel modal-wise representation learning and automatic cross-modal interaction without additional fusion steps. More importantly, to make full use of these complementary sensor types, we present a novel multi-modal integration strategy by both considering semantic-abundant 2D perspective and geometry-aware 3D sparse neighborhood relations. UniTR is also a fundamentally task-agnostic backbone that naturally supports different 3D perception tasks. It sets a new state-of-the-art performance on the nuScenes benchmark, achieving +1.1 NDS higher for 3D object detection and +12.0 higher mIoU for BEV map segmentation with lower inference latency. Code will be available at https://github.com/Haiyang-W/UniTR .

{{</citation>}}


### (70/100) Real-time Automatic M-mode Echocardiography Measurement with Panel Attention from Local-to-Global Pixels (Ching-Hsun Tseng et al., 2023)

{{<citation>}}

Ching-Hsun Tseng, Shao-Ju Chien, Po-Shen Wang, Shin-Jye Lee, Wei-Huan Hu, Bin Pu, Xiao-jun Zeng. (2023)  
**Real-time Automatic M-mode Echocardiography Measurement with Panel Attention from Local-to-Global Pixels**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.07717v1)  

---


**ABSTRACT**  
Motion mode (M-mode) recording is an essential part of echocardiography to measure cardiac dimension and function. However, the current diagnosis cannot build an automatic scheme, as there are three fundamental obstructs: Firstly, there is no open dataset available to build the automation for ensuring constant results and bridging M-mode echocardiography with real-time instance segmentation (RIS); Secondly, the examination is involving the time-consuming manual labelling upon M-mode echocardiograms; Thirdly, as objects in echocardiograms occupy a significant portion of pixels, the limited receptive field in existing backbones (e.g., ResNet) composed from multiple convolution layers are inefficient to cover the period of a valve movement. Existing non-local attentions (NL) compromise being unable real-time with a high computation overhead or losing information from a simplified version of the non-local block. Therefore, we proposed RAMEM, a real-time automatic M-mode echocardiography measurement scheme, contributes three aspects to answer the problems: 1) provide MEIS, a dataset of M-mode echocardiograms for instance segmentation, to enable consistent results and support the development of an automatic scheme; 2) propose panel attention, local-to-global efficient attention by pixel-unshuffling, embedding with updated UPANets V2 in a RIS scheme toward big object detection with global receptive field; 3) develop and implement AMEM, an efficient algorithm of automatic M-mode echocardiography measurement enabling fast and accurate automatic labelling among diagnosis. The experimental results show that RAMEM surpasses existing RIS backbones (with non-local attention) in PASCAL 2012 SBD and human performances in real-time MEIS tested. The code of MEIS and dataset are available at https://github.com/hanktseng131415go/RAME.

{{</citation>}}


### (71/100) Exploring Transfer Learning in Medical Image Segmentation using Vision-Language Models (Kanchan Poudel et al., 2023)

{{<citation>}}

Kanchan Poudel, Manish Dhakal, Prasiddha Bhandari, Rabin Adhikari, Safal Thapaliya, Bishesh Khanal. (2023)  
**Exploring Transfer Learning in Medical Image Segmentation using Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.07706v1)  

---


**ABSTRACT**  
Medical Image Segmentation is crucial in various clinical applications within the medical domain. While state-of-the-art segmentation models have proven effective, integrating textual guidance to enhance visual features for this task remains an area with limited progress. Existing segmentation models that utilize textual guidance are primarily trained on open-domain images, raising concerns about their direct applicability in the medical domain without manual intervention or fine-tuning.   To address these challenges, we propose using multimodal vision-language models for capturing semantic information from image descriptions and images, enabling the segmentation of diverse medical images. This study comprehensively evaluates existing vision language models across multiple datasets to assess their transferability from the open domain to the medical field. Furthermore, we introduce variations of image descriptions for previously unseen images in the dataset, revealing notable variations in model performance based on the generated prompts.   Our findings highlight the distribution shift between the open-domain images and the medical domain and show that the segmentation models trained on open-domain images are not directly transferrable to the medical field. But their performance can be increased by finetuning them in the medical datasets. We report the zero-shot and finetuned segmentation performance of 4 Vision Language Models (VLMs) on 11 medical datasets using 9 types of prompts derived from 14 attributes.

{{</citation>}}


### (72/100) DiffGuard: Semantic Mismatch-Guided Out-of-Distribution Detection using Pre-trained Diffusion Models (Ruiyuan Gao et al., 2023)

{{<citation>}}

Ruiyuan Gao, Chenchen Zhao, Lanqing Hong, Qiang Xu. (2023)  
**DiffGuard: Semantic Mismatch-Guided Out-of-Distribution Detection using Pre-trained Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.07687v2)  

---


**ABSTRACT**  
Given a classifier, the inherent property of semantic Out-of-Distribution (OOD) samples is that their contents differ from all legal classes in terms of semantics, namely semantic mismatch. There is a recent work that directly applies it to OOD detection, which employs a conditional Generative Adversarial Network (cGAN) to enlarge semantic mismatch in the image space. While achieving remarkable OOD detection performance on small datasets, it is not applicable to ImageNet-scale datasets due to the difficulty in training cGANs with both input images and labels as conditions. As diffusion models are much easier to train and amenable to various conditions compared to cGANs, in this work, we propose to directly use pre-trained diffusion models for semantic mismatch-guided OOD detection, named DiffGuard. Specifically, given an OOD input image and the predicted label from the classifier, we try to enlarge the semantic difference between the reconstructed OOD image under these conditions and the original input image. We also present several test-time techniques to further strengthen such differences. Experimental results show that DiffGuard is effective on both Cifar-10 and hard cases of the large-scale ImageNet, and it can be easily combined with existing OOD detection techniques to achieve state-of-the-art OOD detection results.

{{</citation>}}


### (73/100) A Review of Adversarial Attacks in Computer Vision (Yutong Zhang et al., 2023)

{{<citation>}}

Yutong Zhang, Yao Li, Yin Li, Zhichang Guo. (2023)  
**A Review of Adversarial Attacks in Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, stat-CO, stat-ML  
Keywords: Adversarial Attack, Computer Vision  
[Paper Link](http://arxiv.org/abs/2308.07673v1)  

---


**ABSTRACT**  
Deep neural networks have been widely used in various downstream tasks, especially those safety-critical scenario such as autonomous driving, but deep networks are often threatened by adversarial samples. Such adversarial attacks can be invisible to human eyes, but can lead to DNN misclassification, and often exhibits transferability between deep learning and machine learning models and real-world achievability. Adversarial attacks can be divided into white-box attacks, for which the attacker knows the parameters and gradient of the model, and black-box attacks, for the latter, the attacker can only obtain the input and output of the model. In terms of the attacker's purpose, it can be divided into targeted attacks and non-targeted attacks, which means that the attacker wants the model to misclassify the original sample into the specified class, which is more practical, while the non-targeted attack just needs to make the model misclassify the sample. The black box setting is a scenario we will encounter in practice.

{{</citation>}}


### (74/100) Inversion-by-Inversion: Exemplar-based Sketch-to-Photo Synthesis via Stochastic Differential Equations without Training (Ximing Xing et al., 2023)

{{<citation>}}

Ximing Xing, Chuang Wang, Haitao Zhou, Zhihao Hu, Chongxuan Li, Dong Xu, Qian Yu. (2023)  
**Inversion-by-Inversion: Exemplar-based Sketch-to-Photo Synthesis via Stochastic Differential Equations without Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2308.07665v1)  

---


**ABSTRACT**  
Exemplar-based sketch-to-photo synthesis allows users to generate photo-realistic images based on sketches. Recently, diffusion-based methods have achieved impressive performance on image generation tasks, enabling highly-flexible control through text-driven generation or energy functions. However, generating photo-realistic images with color and texture from sketch images remains challenging for diffusion models. Sketches typically consist of only a few strokes, with most regions left blank, making it difficult for diffusion-based methods to produce photo-realistic images. In this work, we propose a two-stage method named ``Inversion-by-Inversion" for exemplar-based sketch-to-photo synthesis. This approach includes shape-enhancing inversion and full-control inversion. During the shape-enhancing inversion process, an uncolored photo is generated with the guidance of a shape-energy function. This step is essential to ensure control over the shape of the generated photo. In the full-control inversion process, we propose an appearance-energy function to control the color and texture of the final generated photo.Importantly, our Inversion-by-Inversion pipeline is training-free and can accept different types of exemplars for color and texture control. We conducted extensive experiments to evaluate our proposed method, and the results demonstrate its effectiveness.

{{</citation>}}


### (75/100) EQ-Net: Elastic Quantization Neural Networks (Ke Xu et al., 2023)

{{<citation>}}

Ke Xu, Lei Han, Ye Tian, Shangshang Yang, Xingyi Zhang. (2023)  
**EQ-Net: Elastic Quantization Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA, Quantization  
[Paper Link](http://arxiv.org/abs/2308.07650v1)  

---


**ABSTRACT**  
Current model quantization methods have shown their promising capability in reducing storage space and computation complexity. However, due to the diversity of quantization forms supported by different hardware, one limitation of existing solutions is that usually require repeated optimization for different scenarios. How to construct a model with flexible quantization forms has been less studied. In this paper, we explore a one-shot network quantization regime, named Elastic Quantization Neural Networks (EQ-Net), which aims to train a robust weight-sharing quantization supernet. First of all, we propose an elastic quantization space (including elastic bit-width, granularity, and symmetry) to adapt to various mainstream quantitative forms. Secondly, we propose the Weight Distribution Regularization Loss (WDR-Loss) and Group Progressive Guidance Loss (GPG-Loss) to bridge the inconsistency of the distribution for weights and output logits in the elastic quantization space gap. Lastly, we incorporate genetic algorithms and the proposed Conditional Quantization-Aware Accuracy Predictor (CQAP) as an estimator to quickly search mixed-precision quantized neural networks in supernet. Extensive experiments demonstrate that our EQ-Net is close to or even better than its static counterparts as well as state-of-the-art robust bit-width methods. Code can be available at \href{https://github.com/xuke225/EQ-Net.git}{https://github.com/xuke225/EQ-Net}.

{{</citation>}}


### (76/100) Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation (Qi Wu et al., 2023)

{{<citation>}}

Qi Wu, Yuyao Zhang, Marawan Elbatel. (2023)  
**Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.07624v1)  

---


**ABSTRACT**  
Recent advancements in large foundation models have shown promising potential in the medical industry due to their flexible prompting capability. One such model, the Segment Anything Model (SAM), a prompt-driven segmentation model, has shown remarkable performance improvements, surpassing state-of-the-art approaches in medical image segmentation. However, existing methods primarily rely on tuning strategies that require extensive data or prior prompts tailored to the specific task, making it particularly challenging when only a limited number of data samples are available. In this paper, we propose a novel perspective on self-prompting in medical vision applications. Specifically, we harness the embedding space of SAM to prompt itself through a simple yet effective linear pixel-wise classifier. By preserving the encoding capabilities of the large model, the contextual information from its decoder, and leveraging its interactive promptability, we achieve competitive results on multiple datasets (i.e. improvement of more than 15% compared to fine-tuning the mask decoder using a few images).

{{</citation>}}


### (77/100) Self-supervised Hypergraphs for Learning Multiple World Interpretations (Alina Marcu et al., 2023)

{{<citation>}}

Alina Marcu, Mihai Pirvu, Dragos Costea, Emanuela Haller, Emil Slusanschi, Ahmed Nabil Belbachir, Rahul Sukthankar, Marius Leordeanu. (2023)  
**Self-supervised Hypergraphs for Learning Multiple World Interpretations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone, Transformer  
[Paper Link](http://arxiv.org/abs/2308.07615v1)  

---


**ABSTRACT**  
We present a method for learning multiple scene representations given a small labeled set, by exploiting the relationships between such representations in the form of a multi-task hypergraph. We also show how we can use the hypergraph to improve a powerful pretrained VisTransformer model without any additional labeled data. In our hypergraph, each node is an interpretation layer (e.g., depth or segmentation) of the scene. Within each hyperedge, one or several input nodes predict the layer at the output node. Thus, each node could be an input node in some hyperedges and an output node in others. In this way, multiple paths can reach the same node, to form ensembles from which we obtain robust pseudolabels, which allow self-supervised learning in the hypergraph. We test different ensemble models and different types of hyperedges and show superior performance to other multi-task graph models in the field. We also introduce Dronescapes, a large video dataset captured with UAVs in different complex real-world scenes, with multiple representations, suitable for multi-task learning.

{{</citation>}}


### (78/100) AKVSR: Audio Knowledge Empowered Visual Speech Recognition by Compressing Audio Knowledge of a Pretrained Model (Jeong Hun Yeo et al., 2023)

{{<citation>}}

Jeong Hun Yeo, Minsu Kim, Jeongsoo Choi, Dae Hoe Kim, Yong Man Ro. (2023)  
**AKVSR: Audio Knowledge Empowered Visual Speech Recognition by Compressing Audio Knowledge of a Pretrained Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV, eess-AS, eess-IV  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2308.07593v1)  

---


**ABSTRACT**  
Visual Speech Recognition (VSR) is the task of predicting spoken words from silent lip movements. VSR is regarded as a challenging task because of the insufficient information on lip movements. In this paper, we propose an Audio Knowledge empowered Visual Speech Recognition framework (AKVSR) to complement the insufficient speech information of visual modality by using audio modality. Different from the previous methods, the proposed AKVSR 1) utilizes rich audio knowledge encoded by a large-scale pretrained audio model, 2) saves the linguistic information of audio knowledge in compact audio memory by discarding the non-linguistic information from the audio through quantization, and 3) includes Audio Bridging Module which can find the best-matched audio features from the compact audio memory, which makes our training possible without audio inputs, once after the compact audio memory is composed. We validate the effectiveness of the proposed method through extensive experiments, and achieve new state-of-the-art performances on the widely-used datasets, LRS2 and LRS3.

{{</citation>}}


### (79/100) Graph-Segmenter: Graph Transformer with Boundary-aware Attention for Semantic Segmentation (Zizhang Wu et al., 2023)

{{<citation>}}

Zizhang Wu, Yuanzhu Gan, Tianhao Xu, Fan Wang. (2023)  
**Graph-Segmenter: Graph Transformer with Boundary-aware Attention for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2308.07592v1)  

---


**ABSTRACT**  
The transformer-based semantic segmentation approaches, which divide the image into different regions by sliding windows and model the relation inside each window, have achieved outstanding success. However, since the relation modeling between windows was not the primary emphasis of previous work, it was not fully utilized. To address this issue, we propose a Graph-Segmenter, including a Graph Transformer and a Boundary-aware Attention module, which is an effective network for simultaneously modeling the more profound relation between windows in a global view and various pixels inside each window as a local one, and for substantial low-cost boundary adjustment. Specifically, we treat every window and pixel inside the window as nodes to construct graphs for both views and devise the Graph Transformer. The introduced boundary-aware attention module optimizes the edge information of the target objects by modeling the relationship between the pixel on the object's edge. Extensive experiments on three widely used semantic segmentation datasets (Cityscapes, ADE-20k and PASCAL Context) demonstrate that our proposed network, a Graph Transformer with Boundary-aware Attention, can achieve state-of-the-art segmentation performance.

{{</citation>}}


### (80/100) AutoLTS: Automating Cycling Stress Assessment via Contrastive Learning and Spatial Post-processing (Bo Lin et al., 2023)

{{<citation>}}

Bo Lin, Shoshanna Saxe, Timothy C. Y. Chan. (2023)  
**AutoLTS: Automating Cycling Stress Assessment via Contrastive Learning and Spatial Post-processing**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.07580v1)  

---


**ABSTRACT**  
Cycling stress assessment, which quantifies cyclists' perceived stress imposed by the built environment and motor traffics, increasingly informs cycling infrastructure planning and cycling route recommendation. However, currently calculating cycling stress is slow and data-intensive, which hinders its broader application. In this paper, We propose a deep learning framework to support accurate, fast, and large-scale cycling stress assessments for urban road networks based on street-view images. Our framework features i) a contrastive learning approach that leverages the ordinal relationship among cycling stress labels, and ii) a post-processing technique that enforces spatial smoothness into our predictions. On a dataset of 39,153 road segments collected in Toronto, Canada, our results demonstrate the effectiveness of our deep learning framework and the value of using image data for cycling stress assessment in the absence of high-quality road geometry and motor traffic data.

{{</citation>}}


### (81/100) Story Visualization by Online Text Augmentation with Context Memory (Daechul Ahn et al., 2023)

{{<citation>}}

Daechul Ahn, Daneul Kim, Gwangmo Song, Seung Hwan Kim, Honglak Lee, Dongyeop Kang, Jonghyun Choi. (2023)  
**Story Visualization by Online Text Augmentation with Context Memory**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Augmentation, BLEU, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.07575v1)  

---


**ABSTRACT**  
Story visualization (SV) is a challenging text-to-image generation task for the difficulty of not only rendering visual details from the text descriptions but also encoding a long-term context across multiple sentences. While prior efforts mostly focus on generating a semantically relevant image for each sentence, encoding a context spread across the given paragraph to generate contextually convincing images (e.g., with a correct character or with a proper background of the scene) remains a challenge. To this end, we propose a novel memory architecture for the Bi-directional Transformers with an online text augmentation that generates multiple pseudo-descriptions as supplementary supervision during training, for better generalization to the language variation at inference. In extensive experiments on the two popular SV benchmarks, i.e., the Pororo-SV and Flintstones-SV, the proposed method significantly outperforms the state of the arts in various evaluation metrics including FID, character F1, frame accuracy, BLEU-2/3, and R-precision with similar or less computational complexity.

{{</citation>}}


### (82/100) Ske2Grid: Skeleton-to-Grid Representation Learning for Action Recognition (Dongqi Cai et al., 2023)

{{<citation>}}

Dongqi Cai, Yangyuxuan Kang, Anbang Yao, Yurong Chen. (2023)  
**Ske2Grid: Skeleton-to-Grid Representation Learning for Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.07571v1)  

---


**ABSTRACT**  
This paper presents Ske2Grid, a new representation learning framework for improved skeleton-based action recognition. In Ske2Grid, we define a regular convolution operation upon a novel grid representation of human skeleton, which is a compact image-like grid patch constructed and learned through three novel designs. Specifically, we propose a graph-node index transform (GIT) to construct a regular grid patch through assigning the nodes in the skeleton graph one by one to the desired grid cells. To ensure that GIT is a bijection and enrich the expressiveness of the grid representation, an up-sampling transform (UPT) is learned to interpolate the skeleton graph nodes for filling the grid patch to the full. To resolve the problem when the one-step UPT is aggressive and further exploit the representation capability of the grid patch with increasing spatial size, a progressive learning strategy (PLS) is proposed which decouples the UPT into multiple steps and aligns them to multiple paired GITs through a compact cascaded design learned progressively. We construct networks upon prevailing graph convolution networks and conduct experiments on six mainstream skeleton-based action recognition datasets. Experiments show that our Ske2Grid significantly outperforms existing GCN-based solutions under different benchmark settings, without bells and whistles. Code and models are available at https://github.com/OSVAI/Ske2Grid

{{</citation>}}


### (83/100) SST: A Simplified Swin Transformer-based Model for Taxi Destination Prediction based on Existing Trajectory (Zepu Wang et al., 2023)

{{<citation>}}

Zepu Wang, Yifei Sun, Zhiyu Lei, Xincheng Zhu, Peng Sun. (2023)  
**SST: A Simplified Swin Transformer-based Model for Taxi Destination Prediction based on Existing Trajectory**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-CY, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.07555v1)  

---


**ABSTRACT**  
Accurately predicting the destination of taxi trajectories can have various benefits for intelligent location-based services. One potential method to accomplish this prediction is by converting the taxi trajectory into a two-dimensional grid and using computer vision techniques. While the Swin Transformer is an innovative computer vision architecture with demonstrated success in vision downstream tasks, it is not commonly used to solve real-world trajectory problems. In this paper, we propose a simplified Swin Transformer (SST) structure that does not use the shifted window idea in the traditional Swin Transformer, as trajectory data is consecutive in nature. Our comprehensive experiments, based on real trajectory data, demonstrate that SST can achieve higher accuracy compared to state-of-the-art methods.

{{</citation>}}


### (84/100) Visual and Textual Prior Guided Mask Assemble for Few-Shot Segmentation and Beyond (Chen Shuai et al., 2023)

{{<citation>}}

Chen Shuai, Meng Fanman, Zhang Runtong, Qiu Heqian, Li Hongliang, Wu Qingbo, Xu Linfeng. (2023)  
**Visual and Textual Prior Guided Mask Assemble for Few-Shot Segmentation and Beyond**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.07539v1)  

---


**ABSTRACT**  
Few-shot segmentation (FSS) aims to segment the novel classes with a few annotated images. Due to CLIP's advantages of aligning visual and textual information, the integration of CLIP can enhance the generalization ability of FSS model. However, even with the CLIP model, the existing CLIP-based FSS methods are still subject to the biased prediction towards base classes, which is caused by the class-specific feature level interactions. To solve this issue, we propose a visual and textual Prior Guided Mask Assemble Network (PGMA-Net). It employs a class-agnostic mask assembly process to alleviate the bias, and formulates diverse tasks into a unified manner by assembling the prior through affinity. Specifically, the class-relevant textual and visual features are first transformed to class-agnostic prior in the form of probability map. Then, a Prior-Guided Mask Assemble Module (PGMAM) including multiple General Assemble Units (GAUs) is introduced. It considers diverse and plug-and-play interactions, such as visual-textual, inter- and intra-image, training-free, and high-order ones. Lastly, to ensure the class-agnostic ability, a Hierarchical Decoder with Channel-Drop Mechanism (HDCDM) is proposed to flexibly exploit the assembled masks and low-level features, without relying on any class-specific information. It achieves new state-of-the-art results in the FSS task, with mIoU of $77.6$ on $\text{PASCAL-}5^i$ and $59.4$ on $\text{COCO-}20^i$ in 1-shot scenario. Beyond this, we show that without extra re-training, the proposed PGMA-Net can solve bbox-level and cross-domain FSS, co-segmentation, zero-shot segmentation (ZSS) tasks, leading an any-shot segmentation framework.

{{</citation>}}


### (85/100) Improved Region Proposal Network for Enhanced Few-Shot Object Detection (Zeyu Shangguan et al., 2023)

{{<citation>}}

Zeyu Shangguan, Mohammad Rostami. (2023)  
**Improved Region Proposal Network for Enhanced Few-Shot Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Object Detection  
[Paper Link](http://arxiv.org/abs/2308.07535v1)  

---


**ABSTRACT**  
Despite significant success of deep learning in object detection tasks, the standard training of deep neural networks requires access to a substantial quantity of annotated images across all classes. Data annotation is an arduous and time-consuming endeavor, particularly when dealing with infrequent objects. Few-shot object detection (FSOD) methods have emerged as a solution to the limitations of classic object detection approaches based on deep learning. FSOD methods demonstrate remarkable performance by achieving robust object detection using a significantly smaller amount of training data. A challenge for FSOD is that instances from novel classes that do not belong to the fixed set of training classes appear in the background and the base model may pick them up as potential objects. These objects behave similarly to label noise because they are classified as one of the training dataset classes, leading to FSOD performance degradation. We develop a semi-supervised algorithm to detect and then utilize these unlabeled novel objects as positive samples during the FSOD training stage to improve FSOD performance. Specifically, we develop a hierarchical ternary classification region proposal network (HTRPN) to localize the potential unlabeled novel objects and assign them new objectness labels to distinguish these objects from the base training dataset classes. Our improved hierarchical sampling strategy for the region proposal network (RPN) also boosts the perception ability of the object detection model for large objects. We test our approach and COCO and PASCAL VOC baselines that are commonly used in FSOD literature. Our experimental results indicate that our method is effective and outperforms the existing state-of-the-art (SOTA) FSOD methods. Our implementation is provided as a supplement to support reproducibility of the results.

{{</citation>}}


### (86/100) Confidence Contours: Uncertainty-Aware Annotation for Medical Semantic Segmentation (Andre Ye et al., 2023)

{{<citation>}}

Andre Ye, Quan Ze Chen, Amy Zhang. (2023)  
**Confidence Contours: Uncertainty-Aware Annotation for Medical Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.07528v1)  

---


**ABSTRACT**  
Medical image segmentation modeling is a high-stakes task where understanding of uncertainty is crucial for addressing visual ambiguity. Prior work has developed segmentation models utilizing probabilistic or generative mechanisms to infer uncertainty from labels where annotators draw a singular boundary. However, as these annotations cannot represent an individual annotator's uncertainty, models trained on them produce uncertainty maps that are difficult to interpret. We propose a novel segmentation representation, Confidence Contours, which uses high- and low-confidence ``contours'' to capture uncertainty directly, and develop a novel annotation system for collecting contours. We conduct an evaluation on the Lung Image Dataset Consortium (LIDC) and a synthetic dataset. From an annotation study with 30 participants, results show that Confidence Contours provide high representative capacity without considerably higher annotator effort. We also find that general-purpose segmentation models can learn Confidence Contours at the same performance level as standard singular annotations. Finally, from interviews with 5 medical experts, we find that Confidence Contour maps are more interpretable than Bayesian maps due to representation of structural uncertainty.

{{</citation>}}


### (87/100) Boosting Semi-Supervised Learning by bridging high and low-confidence predictions (Khanh-Binh Nguyen et al., 2023)

{{<citation>}}

Khanh-Binh Nguyen, Joon-Sung Yang. (2023)  
**Boosting Semi-Supervised Learning by bridging high and low-confidence predictions**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.07509v1)  

---


**ABSTRACT**  
Pseudo-labeling is a crucial technique in semi-supervised learning (SSL), where artificial labels are generated for unlabeled data by a trained model, allowing for the simultaneous training of labeled and unlabeled data in a supervised setting. However, several studies have identified three main issues with pseudo-labeling-based approaches. Firstly, these methods heavily rely on predictions from the trained model, which may not always be accurate, leading to a confirmation bias problem. Secondly, the trained model may be overfitted to easy-to-learn examples, ignoring hard-to-learn ones, resulting in the \textit{"Matthew effect"} where the already strong become stronger and the weak weaker. Thirdly, most of the low-confidence predictions of unlabeled data are discarded due to the use of a high threshold, leading to an underutilization of unlabeled data during training. To address these issues, we propose a new method called ReFixMatch, which aims to utilize all of the unlabeled data during training, thus improving the generalizability of the model and performance on SSL benchmarks. Notably, ReFixMatch achieves 41.05\% top-1 accuracy with 100k labeled examples on ImageNet, outperforming the baseline FixMatch and current state-of-the-art methods.

{{</citation>}}


### (88/100) ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection (Jifeng Shen et al., 2023)

{{<citation>}}

Jifeng Shen, Yifei Chen, Yue Liu, Xin Zuo, Heng Fan, Wankou Yang. (2023)  
**ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2308.07504v1)  

---


**ABSTRACT**  
Effective feature fusion of multispectral images plays a crucial role in multi-spectral object detection. Previous studies have demonstrated the effectiveness of feature fusion using convolutional neural networks, but these methods are sensitive to image misalignment due to the inherent deffciency in local-range feature interaction resulting in the performance degradation. To address this issue, a novel feature fusion framework of dual cross-attention transformers is proposed to model global feature interaction and capture complementary information across modalities simultaneously. This framework enhances the discriminability of object features through the query-guided cross-attention mechanism, leading to improved performance. However, stacking multiple transformer blocks for feature enhancement incurs a large number of parameters and high spatial complexity. To handle this, inspired by the human process of reviewing knowledge, an iterative interaction mechanism is proposed to share parameters among block-wise multimodal transformers, reducing model complexity and computation cost. The proposed method is general and effective to be integrated into different detection frameworks and used with different backbones. Experimental results on KAIST, FLIR, and VEDAI datasets show that the proposed method achieves superior performance and faster inference, making it suitable for various practical scenarios. Code will be available at https://github.com/chanchanchan97/ICAFusion.

{{</citation>}}


## q-bio.BM (1)



### (89/100) APACE: AlphaFold2 and advanced computing as a service for accelerated discovery in biophysics (Hyun Park et al., 2023)

{{<citation>}}

Hyun Park, Parth Patel, Roland Haas, E. A. Huerta. (2023)  
**APACE: AlphaFold2 and advanced computing as a service for accelerated discovery in biophysics**  

---
Primary Category: q-bio.BM  
Categories: I-2, cs-AI, cs-DC, q-bio-BM, q-bio.BM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.07954v1)  

---


**ABSTRACT**  
The prediction of protein 3D structure from amino acid sequence is a computational grand challenge in biophysics, and plays a key role in robust protein structure prediction algorithms, from drug discovery to genome interpretation. The advent of AI models, such as AlphaFold, is revolutionizing applications that depend on robust protein structure prediction algorithms. To maximize the impact, and ease the usability, of these novel AI tools we introduce APACE, AlphaFold2 and advanced computing as a service, a novel computational framework that effectively handles this AI model and its TB-size database to conduct accelerated protein structure prediction analyses in modern supercomputing environments. We deployed APACE in the Delta supercomputer, and quantified its performance for accurate protein structure predictions using four exemplar proteins: 6AWO, 6OAN, 7MEZ, and 6D6U. Using up to 200 ensembles, distributed across 50 nodes in Delta, equivalent to 200 A100 NVIDIA GPUs, we found that APACE is up to two orders of magnitude faster than off-the-shelf AlphaFold2 implementations, reducing time-to-solution from weeks to minutes. This computational approach may be readily linked with robotics laboratories to automate and accelerate scientific discovery.

{{</citation>}}


## cond-mat.mtrl-sci (1)



### (90/100) Probabilistic Phase Labeling and Lattice Refinement for Autonomous Material Research (Ming-Chiang Chang et al., 2023)

{{<citation>}}

Ming-Chiang Chang, Sebastian Ament, Maximilian Amsler, Duncan R. Sutherland, Lan Zhou, John M. Gregoire, Carla P. Gomes, R. Bruce van Dover, Michael O. Thompson. (2023)  
**Probabilistic Phase Labeling and Lattice Refinement for Autonomous Material Research**  

---
Primary Category: cond-mat.mtrl-sci  
Categories: cond-mat-mtrl-sci, cond-mat.mtrl-sci, cs-AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.07897v1)  

---


**ABSTRACT**  
X-ray diffraction (XRD) is an essential technique to determine a material's crystal structure in high-throughput experimentation, and has recently been incorporated in artificially intelligent agents in autonomous scientific discovery processes. However, rapid, automated and reliable analysis method of XRD data matching the incoming data rate remains a major challenge. To address these issues, we present CrystalShift, an efficient algorithm for probabilistic XRD phase labeling that employs symmetry-constrained pseudo-refinement optimization, best-first tree search, and Bayesian model comparison to estimate probabilities for phase combinations without requiring phase space information or training. We demonstrate that CrystalShift provides robust probability estimates, outperforming existing methods on synthetic and experimental datasets, and can be readily integrated into high-throughput experimental workflows. In addition to efficient phase-mapping, CrystalShift offers quantitative insights into materials' structural parameters, which facilitate both expert evaluation and AI-based modeling of the phase space, ultimately accelerating materials identification and discovery.

{{</citation>}}


## cs.CR (2)



### (91/100) Robustness Over Time: Understanding Adversarial Examples' Effectiveness on Longitudinal Versions of Large Language Models (Yugeng Liu et al., 2023)

{{<citation>}}

Yugeng Liu, Tianshuo Cong, Zhengyu Zhao, Michael Backes, Yun Shen, Yang Zhang. (2023)  
**Robustness Over Time: Understanding Adversarial Examples' Effectiveness on Longitudinal Versions of Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2308.07847v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have led to significant improvements in many tasks across various domains, such as code interpretation, response generation, and ambiguity handling. These LLMs, however, when upgrading, primarily prioritize enhancing user experience while neglecting security, privacy, and safety implications. Consequently, unintended vulnerabilities or biases can be introduced. Previous studies have predominantly focused on specific versions of the models and disregard the potential emergence of new attack vectors targeting the updated versions. Through the lens of adversarial examples within the in-context learning framework, this longitudinal study addresses this gap by conducting a comprehensive assessment of the robustness of successive versions of LLMs, vis-\`a-vis GPT-3.5. We conduct extensive experiments to analyze and understand the impact of the robustness in two distinct learning categories: zero-shot learning and few-shot learning. Our findings indicate that, in comparison to earlier versions of LLMs, the updated versions do not exhibit the anticipated level of robustness against adversarial attacks. In addition, our study emphasizes the increased effectiveness of synergized adversarial queries in most zero-shot learning and few-shot learning cases. We hope that our study can lead to a more refined assessment of the robustness of LLMs over time and provide valuable insights of these models for both developers and users.

{{</citation>}}


### (92/100) Block-Wise Encryption for Reliable Vision Transformer models (Hitoshi Kiya et al., 2023)

{{<citation>}}

Hitoshi Kiya, Ryota Iijima, Teru Nagamori. (2023)  
**Block-Wise Encryption for Reliable Vision Transformer models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.07612v1)  

---


**ABSTRACT**  
This article presents block-wise image encryption for the vision transformer and its applications. Perceptual image encryption for deep learning enables us not only to protect the visual information of plain images but to also embed unique features controlled with a key into images and models. However, when using conventional perceptual encryption methods, the performance of models is degraded due to the influence of encryption. In this paper, we focus on block-wise encryption for the vision transformer, and we introduce three applications: privacy-preserving image classification, access control, and the combined use of federated learning and encrypted images. Our scheme can have the same performance as models without any encryption, and it does not require any network modification. It also allows us to easily update the secret key. In experiments, the effectiveness of the scheme is demonstrated in terms of performance degradation and access control on the CIFAR10 and CIFAR-100 datasets.

{{</citation>}}


## eess.SY (1)



### (93/100) A Data-Driven Policy for Addressing Deployability Issue of FMM FRPs: Resources Qualification and Deliverability (Mohammad Ghaljehei et al., 2023)

{{<citation>}}

Mohammad Ghaljehei, Mojdeh Khorsand. (2023)  
**A Data-Driven Policy for Addressing Deployability Issue of FMM FRPs: Resources Qualification and Deliverability**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.07846v1)  

---


**ABSTRACT**  
Intensified netload uncertainty and variability led to the concept of a new market product, flexible ramping product (FRP). The main goal of FRP is to enhance the generation dispatch flexibility inside real-time (RT) markets to mitigate energy imbalances due to ramp capability shortage. Generally, the FRP requirements are based on system-wide or proxy requirements, so the effect of FRP awards on the transmission line constraints is not considered. This can lead to FRP deployability issues in RT operation. This paper proposes a new FRP design based on a datadriven policy incorporating ramping response factor sets to address FRP deployability issue. First, a data-mining algorithm is performed to predict the ramp-qualified generators to create the data-driven policy. Then, the FRP awards are assigned to these units while considering effects of post-deployment of FRPs on the transmission line limits. Finally, the proposed data-driven policy is tested against proxy policy through an out-of-sample validation phase that (i) mimics the RT operation of the CAISO, and (ii) represents the expensive ad-hoc actions needed for procuring additional ramping capability to follow realized netload changes. The results show the effectiveness of the proposed data-driven policy from reliability and economic points of view

{{</citation>}}


## q-fin.CP (1)



### (94/100) Nested Multilevel Monte Carlo with Biased and Antithetic Sampling (Abdul-Lateef Haji-Ali et al., 2023)

{{<citation>}}

Abdul-Lateef Haji-Ali, Jonathan Spence. (2023)  
**Nested Multilevel Monte Carlo with Biased and Antithetic Sampling**  

---
Primary Category: q-fin.CP  
Categories: 65C05, 62P05, cs-NA, math-NA, q-fin-CP, q-fin.CP  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.07835v1)  

---


**ABSTRACT**  
We consider the problem of estimating a nested structure of two expectations taking the form $U_0 = E[\max\{U_1(Y), \pi(Y)\}]$, where $U_1(Y) = E[X\ |\ Y]$. Terms of this form arise in financial risk estimation and option pricing. When $U_1(Y)$ requires approximation, but exact samples of $X$ and $Y$ are available, an antithetic multilevel Monte Carlo (MLMC) approach has been well-studied in the literature. Under general conditions, the antithetic MLMC estimator obtains a root mean squared error $\varepsilon$ with order $\varepsilon^{-2}$ cost. If, additionally, $X$ and $Y$ require approximate sampling, careful balancing of the various aspects of approximation is required to avoid a significant computational burden. Under strong convergence criteria on approximations to $X$ and $Y$, randomised multilevel Monte Carlo techniques can be used to construct unbiased Monte Carlo estimates of $U_1$, which can be paired with an antithetic MLMC estimate of $U_0$ to recover order $\varepsilon^{-2}$ computational cost. In this work, we instead consider biased multilevel approximations of $U_1(Y)$, which require less strict assumptions on the approximate samples of $X$. Extensions to the method consider an approximate and antithetic sampling of $Y$. Analysis shows the resulting estimator has order $\varepsilon^{-2}$ asymptotic cost under the conditions required by randomised MLMC and order $\varepsilon^{-2}|\log\varepsilon|^3$ cost under more general assumptions.

{{</citation>}}


## cs.NE (1)



### (95/100) A Genetic Algorithm Meta-Heuristic for a Generalized Quadratic Assignment Problem (Mojtaba A. Farahani, 2023)

{{<citation>}}

Mojtaba A. Farahani. (2023)  
**A Genetic Algorithm Meta-Heuristic for a Generalized Quadratic Assignment Problem**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2308.07828v1)  

---


**ABSTRACT**  
The generalized quadratic assignment problem (GQAP) is one of the hardest problems to solve in the operations research area. The GQAP addressed in this work is defined as the task of minimizing the assignment and transportation costs of assigning a set of facilities to a set of locations. The facilities have different space requirements, and the locations have different space capacities. Multiple facilities can be assigned to each location if the space capacity is not violated. In this work, three instances of GQAP in different situations are presented. Then, a genetic algorithm is developed to solve the GQAP instances. Finally, the local neighborhood search with the steepest descend strategy is constructed and applied to the final solution obtained by the GA, and the final solution is compared with the best solution found by MPL/CPLEX software and reference papers. The results show that the developed GA heuristic is effective for solving the GQAP.

{{</citation>}}


## cs.DS (1)



### (96/100) Tightest Admissible Shortest Path (Eyal Weiss et al., 2023)

{{<citation>}}

Eyal Weiss, Ariel Felner, Gal A. Kaminka. (2023)  
**Tightest Admissible Shortest Path**  

---
Primary Category: cs.DS  
Categories: cs-AI, cs-DM, cs-DS, cs.DS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08453v1)  

---


**ABSTRACT**  
The shortest path problem in graphs is fundamental to AI. Nearly all variants of the problem and relevant algorithms that solve them ignore edge-weight computation time and its common relation to weight uncertainty. This implies that taking these factors into consideration can potentially lead to a performance boost in relevant applications. Recently, a generalized framework for weighted directed graphs was suggested, where edge-weight can be computed (estimated) multiple times, at increasing accuracy and run-time expense. We build on this framework to introduce the problem of finding the tightest admissible shortest path (TASP); a path with the tightest suboptimality bound on the optimal cost. This is a generalization of the shortest path problem to bounded uncertainty, where edge-weight uncertainty can be traded for computational cost. We present a complete algorithm for solving TASP, with guarantees on solution quality. Empirical evaluation supports the effectiveness of this approach.

{{</citation>}}


## quant-ph (1)



### (97/100) Implementing Quantum Generative Adversarial Network (qGAN) and QCBM in Finance (Santanu Ganguly, 2023)

{{<citation>}}

Santanu Ganguly. (2023)  
**Implementing Quantum Generative Adversarial Network (qGAN) and QCBM in Finance**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-ET, cs-LG, quant-ph, quant-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.08448v1)  

---


**ABSTRACT**  
Quantum machine learning (QML) is a cross-disciplinary subject made up of two of the most exciting research areas: quantum computing and classical machine learning (ML), with ML and artificial intelligence (AI) being projected as the first fields that will be impacted by the rise of quantum machines. Quantum computers are being used today in drug discovery, material & molecular modelling and finance. In this work, we discuss some upcoming active new research areas in application of quantum machine learning (QML) in finance. We discuss certain QML models that has become areas of active interest in the financial world for various applications. We use real world financial dataset and compare models such as qGAN (quantum generative adversarial networks) and QCBM (quantum circuit Born machine) among others, using simulated environments. For the qGAN, we define quantum circuits for discriminators and generators and show promises of future quantum advantage via QML in finance.

{{</citation>}}


## cs.SD (1)



### (98/100) DiffV2S: Diffusion-based Video-to-Speech Synthesis with Vision-guided Speaker Embedding (Jeongsoo Choi et al., 2023)

{{<citation>}}

Jeongsoo Choi, Joanna Hong, Yong Man Ro. (2023)  
**DiffV2S: Diffusion-based Video-to-Speech Synthesis with Vision-guided Speaker Embedding**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.07787v1)  

---


**ABSTRACT**  
Recent research has demonstrated impressive results in video-to-speech synthesis which involves reconstructing speech solely from visual input. However, previous works have struggled to accurately synthesize speech due to a lack of sufficient guidance for the model to infer the correct content with the appropriate sound. To resolve the issue, they have adopted an extra speaker embedding as a speaking style guidance from a reference auditory information. Nevertheless, it is not always possible to obtain the audio information from the corresponding video input, especially during the inference time. In this paper, we present a novel vision-guided speaker embedding extractor using a self-supervised pre-trained model and prompt tuning technique. In doing so, the rich speaker embedding information can be produced solely from input visual information, and the extra audio information is not necessary during the inference time. Using the extracted vision-guided speaker embedding representations, we further develop a diffusion-based video-to-speech synthesis model, so called DiffV2S, conditioned on those speaker embeddings and the visual representation extracted from the input video. The proposed DiffV2S not only maintains phoneme details contained in the input video frames, but also creates a highly intelligible mel-spectrogram in which the speaker identities of the multiple speakers are all preserved. Our experimental results show that DiffV2S achieves the state-of-the-art performance compared to the previous video-to-speech synthesis technique.

{{</citation>}}


## eess.IV (1)



### (99/100) Enhancing Network Initialization for Medical AI Models Using Large-Scale, Unlabeled Natural Images (Soroosh Tayebi Arasteh et al., 2023)

{{<citation>}}

Soroosh Tayebi Arasteh, Leo Misera, Jakob Nikolas Kather, Daniel Truhn, Sven Nebelung. (2023)  
**Enhancing Network Initialization for Medical AI Models Using Large-Scale, Unlabeled Natural Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2308.07688v1)  

---


**ABSTRACT**  
Pre-training datasets, like ImageNet, have become the gold standard in medical image analysis. However, the emergence of self-supervised learning (SSL), which leverages unlabeled data to learn robust features, presents an opportunity to bypass the intensive labeling process. In this study, we explored if SSL for pre-training on non-medical images can be applied to chest radiographs and how it compares to supervised pre-training on non-medical images and on medical images. We utilized a vision transformer and initialized its weights based on (i) SSL pre-training on natural images (DINOv2), (ii) SL pre-training on natural images (ImageNet dataset), and (iii) SL pre-training on chest radiographs from the MIMIC-CXR database. We tested our approach on over 800,000 chest radiographs from six large global datasets, diagnosing more than 20 different imaging findings. Our SSL pre-training on curated images not only outperformed ImageNet-based pre-training (P<0.001 for all datasets) but, in certain cases, also exceeded SL on the MIMIC-CXR dataset. Our findings suggest that selecting the right pre-training strategy, especially with SSL, can be pivotal for improving artificial intelligence (AI)'s diagnostic accuracy in medical imaging. By demonstrating the promise of SSL in chest radiograph analysis, we underline a transformative shift towards more efficient and accurate AI models in medical imaging.

{{</citation>}}


## astro-ph.EP (1)



### (100/100) Searching for Novel Chemistry in Exoplanetary Atmospheres using Machine Learning for Anomaly Detection (Roy T. Forestano et al., 2023)

{{<citation>}}

Roy T. Forestano, Konstantin T. Matchev, Katia Matcheva, Eyup B. Unlu. (2023)  
**Searching for Novel Chemistry in Exoplanetary Atmospheres using Machine Learning for Anomaly Detection**  

---
Primary Category: astro-ph.EP  
Categories: astro-ph-EP, astro-ph-IM, astro-ph.EP, cs-LG, physics-data-an  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.07604v1)  

---


**ABSTRACT**  
The next generation of telescopes will yield a substantial increase in the availability of high-resolution spectroscopic data for thousands of exoplanets. The sheer volume of data and number of planets to be analyzed greatly motivate the development of new, fast and efficient methods for flagging interesting planets for reobservation and detailed analysis. We advocate the application of machine learning (ML) techniques for anomaly (novelty) detection to exoplanet transit spectra, with the goal of identifying planets with unusual chemical composition and even searching for unknown biosignatures. We successfully demonstrate the feasibility of two popular anomaly detection methods (Local Outlier Factor and One Class Support Vector Machine) on a large public database of synthetic spectra. We consider several test cases, each with different levels of instrumental noise. In each case, we use ROC curves to quantify and compare the performance of the two ML techniques.

{{</citation>}}
