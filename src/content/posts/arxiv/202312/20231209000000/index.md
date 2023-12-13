---
draft: false
title: "arXiv @ 2023.12.09"
date: 2023-12-09
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.09"
    identifier: arxiv_20231209
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (23)](#cslg-23)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.CV (39)](#cscv-39)
- [cs.CR (8)](#cscr-8)
- [cs.NI (1)](#csni-1)
- [cs.IT (2)](#csit-2)
- [cs.CL (30)](#cscl-30)
- [cs.SE (4)](#csse-4)
- [cs.CY (2)](#cscy-2)
- [cs.DC (3)](#csdc-3)
- [cs.RO (3)](#csro-3)
- [eess.SY (1)](#eesssy-1)
- [cs.HC (1)](#cshc-1)
- [physics.ed-ph (1)](#physicsed-ph-1)
- [cs.MM (1)](#csmm-1)
- [cs.SI (2)](#cssi-2)
- [cs.AI (4)](#csai-4)
- [eess.IV (2)](#eessiv-2)
- [math.OC (1)](#mathoc-1)
- [physics.ao-ph (1)](#physicsao-ph-1)
- [cs.MA (1)](#csma-1)
- [stat.ML (1)](#statml-1)
- [eess.SP (1)](#eesssp-1)
- [eess.AS (1)](#eessas-1)
- [cs.IR (1)](#csir-1)
- [q-bio.BM (1)](#q-biobm-1)

## cs.LG (23)



### (1/136) A Test-Time Learning Approach to Reparameterize the Geophysical Inverse Problem with a Convolutional Neural Network (Anran Xu et al., 2023)

{{<citation>}}

Anran Xu, Lindsey J. Heagy. (2023)  
**A Test-Time Learning Approach to Reparameterize the Geophysical Inverse Problem with a Convolutional Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-geo-ph  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.04752v1)  

---


**ABSTRACT**  
Regularization is critical in solving the ill-posed geo-physical inversion problems. Explicit regularization is often used, but there are opportunities to explore the implicit regularization effect inherently from a Neural Network structure. Researchers in Computer Vision (CV) have discovered that the Convolutional Neural Network (CNN) architecture inherently enforces a regularization that is advantageous for addressing diverse CV inverse problems, including de-noising and in-painting. In this study, we examine the applicability of this implicit regularization to geophysical inversions. The CNN maps an arbitrary vector to the model space (e.g. log-conductivity on the simulation mesh). The predicted subsurface model is then fed into a forward numerical simulation process to generate corresponding predicted measurements. Subsequently, the objective function value is computed by comparing these predicted measurements with the observed field measurements. The backpropagation algorithm is employed to update the trainable parameters of the CNN during the inversion. Note that the CNN in our proposed method does not require training before the inversion, rather, the CNN weights are estimated in the inversion algorithm, hence this is a test-time learning (TTL) approach. The results demonstrate that the implicit regularization provided by the CNN can be useful in DC resistivity inversions. We also provide a detailed discussion of the potential sources of this implicit regularization and some practical guides for applying the proposed method to other geophysical scenarios. The proposed approach for reparameterizing the inverse problem can be adapted to other Tikhonov-style geophysical inversions.

{{</citation>}}


### (2/136) Efficient Large Language Models Fine-Tuning On Graphs (Rui Xue et al., 2023)

{{<citation>}}

Rui Xue, Xipeng Shen, Ruozhou Yu, Xiaorui Liu. (2023)  
**Efficient Large Language Models Fine-Tuning On Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04737v1)  

---


**ABSTRACT**  
Learning from Text-Attributed Graphs (TAGs) has attracted significant attention due to its wide range of real-world applications. The rapid evolution of large language models (LLMs) has revolutionized the way we process textual data, which indicates a strong potential to replace shallow text embedding generally used in Graph Neural Networks (GNNs). However, we find that existing LLM approaches that exploit text information in graphs suffer from inferior computation and data efficiency. In this work, we introduce a novel and efficient approach for the end-to-end fine-tuning of Large Language Models (LLMs) on TAGs, named LEADING. The proposed approach maintains computation cost and memory overhead comparable to the graph-less fine-tuning of LLMs. Moreover, it transfers the rick knowledge in LLMs to downstream graph learning tasks effectively with limited labeled data in semi-supervised learning. Its superior computation and data efficiency are demonstrated through comprehensive experiments, offering a promising solution for a wide range of LLMs and graph learning tasks on TAGs.

{{</citation>}}


### (3/136) Error Discovery by Clustering Influence Embeddings (Fulton Wang et al., 2023)

{{<citation>}}

Fulton Wang, Julius Adebayo, Sarah Tan, Diego Garcia-Olano, Narine Kokhlikyan. (2023)  
**Error Discovery by Clustering Influence Embeddings**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.04712v1)  

---


**ABSTRACT**  
We present a method for identifying groups of test examples -- slices -- on which a model under-performs, a task now known as slice discovery. We formalize coherence -- a requirement that erroneous predictions, within a slice, should be wrong for the same reason -- as a key property that any slice discovery method should satisfy. We then use influence functions to derive a new slice discovery method, InfEmbed, which satisfies coherence by returning slices whose examples are influenced similarly by the training data. InfEmbed is simple, and consists of applying K-Means clustering to a novel representation we deem influence embeddings. We show InfEmbed outperforms current state-of-the-art methods on 2 benchmarks, and is effective for model debugging across several case studies.

{{</citation>}}


### (4/136) GraphMETRO: Mitigating Complex Distribution Shifts in GNNs via Mixture of Aligned Experts (Shirley Wu et al., 2023)

{{<citation>}}

Shirley Wu, Kaidi Cao, Bruno Ribeiro, James Zou, Jure Leskovec. (2023)  
**GraphMETRO: Mitigating Complex Distribution Shifts in GNNs via Mixture of Aligned Experts**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.04693v1)  

---


**ABSTRACT**  
Graph Neural Networks' (GNNs) ability to generalize across complex distributions is crucial for real-world applications. However, prior research has primarily focused on specific types of distribution shifts, such as larger graph size, or inferred shifts from constructed data environments, which is highly limited when confronted with multiple and nuanced distribution shifts. For instance, in a social graph, a user node might experience increased interactions and content alterations, while other user nodes encounter distinct shifts. Neglecting such complexities significantly impedes generalization. To address it, we present GraphMETRO, a novel framework that enhances GNN generalization under complex distribution shifts in both node and graph-level tasks. Our approach employs a mixture-of-experts (MoE) architecture with a gating model and expert models aligned in a shared representation space. The gating model identifies key mixture components governing distribution shifts, while each expert generates invariant representations w.r.t. a mixture component. Finally, GraphMETRO aggregates representations from multiple experts to generate the final invariant representation. Our experiments on synthetic and realworld datasets demonstrate GraphMETRO's superiority and interpretability. To highlight, GraphMETRO achieves state-of-the-art performances on four real-world datasets from GOOD benchmark, outperforming the best baselines on WebKB and Twitch datasets by 67% and 4.2%, respectively.

{{</citation>}}


### (5/136) Federated Learning for 6G: Paradigms, Taxonomy, Recent Advances and Insights (Maryam Ben Driss et al., 2023)

{{<citation>}}

Maryam Ben Driss, Essaid Sabir, Halima Elbiaze, Walid Saad. (2023)  
**Federated Learning for 6G: Paradigms, Taxonomy, Recent Advances and Insights**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-GT, cs-LG, cs-NI, cs.LG, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04688v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) is expected to play an instrumental role in the next generation of wireless systems, such as sixth-generation (6G) mobile network. However, massive data, energy consumption, training complexity, and sensitive data protection in wireless systems are all crucial challenges that must be addressed for training AI models and gathering intelligence and knowledge from distributed devices. Federated Learning (FL) is a recent framework that has emerged as a promising approach for multiple learning agents to build an accurate and robust machine learning models without sharing raw data. By allowing mobile handsets and devices to collaboratively learn a global model without explicit sharing of training data, FL exhibits high privacy and efficient spectrum utilization. While there are a lot of survey papers exploring FL paradigms and usability in 6G privacy, none of them has clearly addressed how FL can be used to improve the protocol stack and wireless operations. The main goal of this survey is to provide a comprehensive overview on FL usability to enhance mobile services and enable smart ecosystems to support novel use-cases. This paper examines the added-value of implementing FL throughout all levels of the protocol stack. Furthermore, it presents important FL applications, addresses hot topics, provides valuable insights and explicits guidance for future research and developments. Our concluding remarks aim to leverage the synergy between FL and future 6G, while highlighting FL's potential to revolutionize wireless industry and sustain the development of cutting-edge mobile services.

{{</citation>}}


### (6/136) Adversarial Learning for Feature Shift Detection and Correction (Miriam Barrabes et al., 2023)

{{<citation>}}

Miriam Barrabes, Daniel Mas Montserrat, Margarita Geleta, Xavier Giro-i-Nieto, Alexander G. Ioannidis. (2023)  
**Adversarial Learning for Feature Shift Detection and Correction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-AP, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04546v1)  

---


**ABSTRACT**  
Data shift is a phenomenon present in many real-world applications, and while there are multiple methods attempting to detect shifts, the task of localizing and correcting the features originating such shifts has not been studied in depth. Feature shifts can occur in many datasets, including in multi-sensor data, where some sensors are malfunctioning, or in tabular and structured data, including biomedical, financial, and survey data, where faulty standardization and data processing pipelines can lead to erroneous features. In this work, we explore using the principles of adversarial learning, where the information from several discriminators trained to distinguish between two distributions is used to both detect the corrupted features and fix them in order to remove the distribution shift between datasets. We show that mainstream supervised classifiers, such as random forest or gradient boosting trees, combined with simple iterative heuristics, can localize and correct feature shifts, outperforming current statistical and neural network-based techniques. The code is available at https://github.com/AI-sandbox/DataFix.

{{</citation>}}


### (7/136) Trajeglish: Learning the Language of Driving Scenarios (Jonah Philion et al., 2023)

{{<citation>}}

Jonah Philion, Xue Bin Peng, Sanja Fidler. (2023)  
**Trajeglish: Learning the Language of Driving Scenarios**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.04535v1)  

---


**ABSTRACT**  
A longstanding challenge for self-driving development is simulating dynamic driving scenarios seeded from recorded driving logs. In pursuit of this functionality, we apply tools from discrete sequence modeling to model how vehicles, pedestrians and cyclists interact in driving scenarios. Using a simple data-driven tokenization scheme, we discretize trajectories to centimeter-level resolution using a small vocabulary. We then model the multi-agent sequence of motion tokens with a GPT-like encoder-decoder that is autoregressive in time and takes into account intra-timestep interaction between agents. Scenarios sampled from our model exhibit state-of-the-art realism; our model tops the Waymo Sim Agents Benchmark, surpassing prior work along the realism meta metric by 3.3% and along the interaction metric by 9.9%. We ablate our modeling choices in full autonomy and partial autonomy settings, and show that the representations learned by our model can quickly be adapted to improve performance on nuScenes. We additionally evaluate the scalability of our model with respect to parameter count and dataset size, and use density estimates from our model to quantify the saliency of context length and intra-timestep interaction for the traffic modeling task.

{{</citation>}}


### (8/136) Relational Deep Learning: Graph Representation Learning on Relational Databases (Matthias Fey et al., 2023)

{{<citation>}}

Matthias Fey, Weihua Hu, Kexin Huang, Jan Eric Lenssen, Rishabh Ranjan, Joshua Robinson, Rex Ying, Jiaxuan You, Jure Leskovec. (2023)  
**Relational Deep Learning: Graph Representation Learning on Relational Databases**  

---
Primary Category: cs.LG  
Categories: cs-DB, cs-LG, cs.LG  
Keywords: AI, Amazon, Graph Neural Network, Graph Neural Networks, Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.04615v1)  

---


**ABSTRACT**  
Much of the world's most valued data is stored in relational databases and data warehouses, where the data is organized into many tables connected by primary-foreign key relations. However, building machine learning models using this data is both challenging and time consuming. The core problem is that no machine learning method is capable of learning on multiple tables interconnected by primary-foreign key relations. Current methods can only learn from a single table, so the data must first be manually joined and aggregated into a single training table, the process known as feature engineering. Feature engineering is slow, error prone and leads to suboptimal models. Here we introduce an end-to-end deep representation learning approach to directly learn on data laid out across multiple tables. We name our approach Relational Deep Learning (RDL). The core idea is to view relational databases as a temporal, heterogeneous graph, with a node for each row in each table, and edges specified by primary-foreign key links. Message Passing Graph Neural Networks can then automatically learn across the graph to extract representations that leverage all input data, without any manual feature engineering. Relational Deep Learning leads to more accurate models that can be built much faster. To facilitate research in this area, we develop RelBench, a set of benchmark datasets and an implementation of Relational Deep Learning. The data covers a wide spectrum, from discussions on Stack Exchange to book reviews on the Amazon Product Catalog. Overall, we define a new research area that generalizes graph machine learning and broadens its applicability to a wide set of AI use cases.

{{</citation>}}


### (9/136) Using Large Language Models for Hyperparameter Optimization (Michael R. Zhang et al., 2023)

{{<citation>}}

Michael R. Zhang, Nishkrit Desai, Juhan Bae, Jonathan Lorraine, Jimmy Ba. (2023)  
**Using Large Language Models for Hyperparameter Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.04528v1)  

---


**ABSTRACT**  
This paper studies using foundational large language models (LLMs) to make decisions during hyperparameter optimization (HPO). Empirical evaluations demonstrate that in settings with constrained search budgets, LLMs can perform comparably or better than traditional HPO methods like random search and Bayesian optimization on standard benchmarks. Furthermore, we propose to treat the code specifying our model as a hyperparameter, which the LLM outputs, going beyond the capabilities of existing HPO approaches. Our findings suggest that LLMs are a promising tool for improving efficiency in the traditional decision-making problem of hyperparameter optimization.

{{</citation>}}


### (10/136) On the Learnability of Watermarks for Language Models (Chenchen Gu et al., 2023)

{{<citation>}}

Chenchen Gu, Xiang Lisa Li, Percy Liang, Tatsunori Hashimoto. (2023)  
**On the Learnability of Watermarks for Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.04469v1)  

---


**ABSTRACT**  
Watermarking of language model outputs enables statistical detection of model-generated text, which has many applications in the responsible deployment of language models. Existing watermarking strategies operate by altering the decoder of an existing language model, and the ability for a language model to directly learn to generate the watermark would have significant implications for the real-world deployment of watermarks. First, learned watermarks could be used to build open models that naturally generate watermarked text, allowing for open models to benefit from watermarking. Second, if watermarking is used to determine the provenance of generated text, an adversary can hurt the reputation of a victim model by spoofing its watermark and generating damaging watermarked text. To investigate the learnability of watermarks, we propose watermark distillation, which trains a student model to behave like a teacher model that uses decoding-based watermarking. We test our approach on three distinct decoding-based watermarking strategies and various hyperparameter settings, finding that models can learn to generate watermarked text with high detectability. We also find limitations to learnability, including the loss of watermarking capabilities under fine-tuning on normal text and high sample complexity when learning low-distortion watermarks.

{{</citation>}}


### (11/136) Horizon-Free and Instance-Dependent Regret Bounds for Reinforcement Learning with General Function Approximation (Jiayi Huang et al., 2023)

{{<citation>}}

Jiayi Huang, Han Zhong, Liwei Wang, Lin F. Yang. (2023)  
**Horizon-Free and Instance-Dependent Regret Bounds for Reinforcement Learning with General Function Approximation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.04464v1)  

---


**ABSTRACT**  
To tackle long planning horizon problems in reinforcement learning with general function approximation, we propose the first algorithm, termed as UCRL-WVTR, that achieves both \emph{horizon-free} and \emph{instance-dependent}, since it eliminates the polynomial dependency on the planning horizon. The derived regret bound is deemed \emph{sharp}, as it matches the minimax lower bound when specialized to linear mixture MDPs up to logarithmic factors. Furthermore, UCRL-WVTR is \emph{computationally efficient} with access to a regression oracle. The achievement of such a horizon-free, instance-dependent, and sharp regret bound hinges upon (i) novel algorithm designs: weighted value-targeted regression and a high-order moment estimator in the context of general function approximation; and (ii) fine-grained analyses: a novel concentration bound of weighted non-linear least squares and a refined analysis which leads to the tight instance-dependent bound. We also conduct comprehensive experiments to corroborate our theoretical findings.

{{</citation>}}


### (12/136) A Structural-Clustering Based Active Learning for Graph Neural Networks (Ricky Maulana Fajri et al., 2023)

{{<citation>}}

Ricky Maulana Fajri, Yulong Pei, Lu Yin, Mykola Pechenizkiy. (2023)  
**A Structural-Clustering Based Active Learning for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.04307v1)  

---


**ABSTRACT**  
In active learning for graph-structured data, Graph Neural Networks (GNNs) have shown effectiveness. However, a common challenge in these applications is the underutilization of crucial structural information. To address this problem, we propose the Structural-Clustering PageRank method for improved Active learning (SPA) specifically designed for graph-structured data. SPA integrates community detection using the SCAN algorithm with the PageRank scoring method for efficient and informative sample selection. SPA prioritizes nodes that are not only informative but also central in structure. Through extensive experiments, SPA demonstrates higher accuracy and macro-F1 score over existing methods across different annotation budgets and achieves significant reductions in query time. In addition, the proposed method only adds two hyperparameters, $\epsilon$ and $\mu$ in the algorithm to finely tune the balance between structural learning and node selection. This simplicity is a key advantage in active learning scenarios, where extensive hyperparameter tuning is often impractical.

{{</citation>}}


### (13/136) Short-term prediction of construction waste transport activities using AI-Truck (Meng Xu et al., 2023)

{{<citation>}}

Meng Xu, Ke Han. (2023)  
**Short-term prediction of construction waste transport activities using AI-Truck**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, LSTM  
[Paper Link](http://arxiv.org/abs/2312.04609v1)  

---


**ABSTRACT**  
Construction waste hauling trucks (or `slag trucks') are among the most commonly seen heavy-duty vehicles in urban streets, which not only produce significant NOx and PM emissions but are also a major source of on-road and on-site fugitive dust. Slag trucks are subject to a series of spatial and temporal access restrictions by local traffic and environmental policies. This paper addresses the practical problem of predicting slag truck activity at a city scale during heavy pollution episodes, such that environmental law enforcement units can take timely and proactive measures against localized truck aggregation. A deep ensemble learning framework (coined AI-Truck) is designed, which employs a soft vote integrator that utilizes BI-LSTM, TCN, STGCN, and PDFormer as base classifiers to predict the level of slag truck activities at a resolution of 1km$\times$1km, in a 193 km$^2$ area in Chengdu, China. As a classifier, AI-Truck yields a Macro f1 close to 80\% for 0.5h- and 1h-prediction.

{{</citation>}}


### (14/136) Graph Convolutions Enrich the Self-Attention in Transformers! (Jeongwhan Choi et al., 2023)

{{<citation>}}

Jeongwhan Choi, Hyowon Wi, Jayoung Kim, Yehjin Shin, Kookjin Lee, Nathaniel Trask, Noseong Park. (2023)  
**Graph Convolutions Enrich the Self-Attention in Transformers!**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.04234v1)  

---


**ABSTRACT**  
Transformers, renowned for their self-attention mechanism, have achieved state-of-the-art performance across various tasks in natural language processing, computer vision, time-series modeling, etc. However, one of the challenges with deep Transformer models is the oversmoothing problem, where representations across layers converge to indistinguishable values, leading to significant performance degradation. We interpret the original self-attention as a simple graph filter and redesign it from a graph signal processing (GSP) perspective. We propose graph-filter-based self-attention (GFSA) to learn a general yet effective one, whose complexity, however, is slightly larger than that of the original self-attention mechanism. We demonstrate that GFSA improves the performance of Transformers in various fields, including computer vision, natural language processing, graph pattern classification, speech recognition, and code classification.

{{</citation>}}


### (15/136) Urban Region Representation Learning with Attentive Fusion (Fengze Sun et al., 2023)

{{<citation>}}

Fengze Sun, Jianzhong Qi, Yanchuan Chang, Xiaoliang Fan, Shanika Karunasekera, Egemen Tanin. (2023)  
**Urban Region Representation Learning with Attentive Fusion**  

---
Primary Category: cs.LG  
Categories: cs-DB, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.04606v1)  

---


**ABSTRACT**  
An increasing number of related urban data sources have brought forth novel opportunities for learning urban region representations, i.e., embeddings. The embeddings describe latent features of urban regions and enable discovering similar regions for urban planning applications. Existing methods learn an embedding for a region using every different type of region feature data, and subsequently fuse all learned embeddings of a region to generate a unified region embedding. However, these studies often overlook the significance of the fusion process. The typical fusion methods rely on simple aggregation, such as summation and concatenation, thereby disregarding correlations within the fused region embeddings.   To address this limitation, we propose a novel model named HAFusion. Our model is powered by a dual-feature attentive fusion module named DAFusion, which fuses embeddings from different region features to learn higher-order correlations between the regions as well as between the different types of region features. DAFusion is generic - it can be integrated into existing models to enhance their fusion process. Further, motivated by the effective fusion capability of an attentive module, we propose a hybrid attentive feature learning module named HALearning to enhance the embedding learning from each individual type of region features. Extensive experiments on three real-world datasets demonstrate that our model HAFusion outperforms state-of-the-art methods across three different prediction tasks. Using our learned region embedding leads to consistent and up to 31% improvements in the prediction accuracy.

{{</citation>}}


### (16/136) CODEX: A Cluster-Based Method for Explainable Reinforcement Learning (Timothy K. Mathes et al., 2023)

{{<citation>}}

Timothy K. Mathes, Jessica Inman, Andrés Colón, Simon Khan. (2023)  
**CODEX: A Cluster-Based Method for Explainable Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Natural Language Processing, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.04216v1)  

---


**ABSTRACT**  
Despite the impressive feats demonstrated by Reinforcement Learning (RL), these algorithms have seen little adoption in high-risk, real-world applications due to current difficulties in explaining RL agent actions and building user trust. We present Counterfactual Demonstrations for Explanation (CODEX), a method that incorporates semantic clustering, which can effectively summarize RL agent behavior in the state-action space. Experimentation on the MiniGrid and StarCraft II gaming environments reveals the semantic clusters retain temporal as well as entity information, which is reflected in the constructed summary of agent behavior. Furthermore, clustering the discrete+continuous game-state latent representations identifies the most crucial episodic events, demonstrating a relationship between the latent and semantic spaces. This work contributes to the growing body of work that strives to unlock the power of RL for widespread use by leveraging and extending techniques from Natural Language Processing.

{{</citation>}}


### (17/136) Constrained Hierarchical Clustering via Graph Coarsening and Optimal Cuts (Eliabelle Mauduit et al., 2023)

{{<citation>}}

Eliabelle Mauduit, Andrea Simonetto. (2023)  
**Constrained Hierarchical Clustering via Graph Coarsening and Optimal Cuts**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2312.04209v1)  

---


**ABSTRACT**  
Motivated by extracting and summarizing relevant information in short sentence settings, such as satisfaction questionnaires, hotel reviews, and X/Twitter, we study the problem of clustering words in a hierarchical fashion. In particular, we focus on the problem of clustering with horizontal and vertical structural constraints. Horizontal constraints are typically cannot-link and must-link among words, while vertical constraints are precedence constraints among cluster levels. We overcome state-of-the-art bottlenecks by formulating the problem in two steps: first, as a soft-constrained regularized least-squares which guides the result of a sequential graph coarsening algorithm towards the horizontal feasible set. Then, flat clusters are extracted from the resulting hierarchical tree by computing optimal cut heights based on the available constraints. We show that the resulting approach compares very well with respect to existing algorithms and is computationally light.

{{</citation>}}


### (18/136) TimeDRL: Disentangled Representation Learning for Multivariate Time-Series (Ching Chang et al., 2023)

{{<citation>}}

Ching Chang, Chiao-Tung Chan, Wei-Yao Wang, Wen-Chih Peng, Tien-Fu Chen. (2023)  
**TimeDRL: Disentangled Representation Learning for Multivariate Time-Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.04142v1)  

---


**ABSTRACT**  
Multivariate time-series data in numerous real-world applications (e.g., healthcare and industry) are informative but challenging due to the lack of labels and high dimensionality. Recent studies in self-supervised learning have shown their potential in learning rich representations without relying on labels, yet they fall short in learning disentangled embeddings and addressing issues of inductive bias (e.g., transformation-invariance). To tackle these challenges, we propose TimeDRL, a generic multivariate time-series representation learning framework with disentangled dual-level embeddings. TimeDRL is characterized by three novel features: (i) disentangled derivation of timestamp-level and instance-level embeddings from patched time-series data using a [CLS] token strategy; (ii) utilization of timestamp-predictive and instance-contrastive tasks for disentangled representation learning, with the former optimizing timestamp-level embeddings with predictive loss, and the latter optimizing instance-level embeddings with contrastive loss; and (iii) avoidance of augmentation methods to eliminate inductive biases, such as transformation-invariance from cropping and masking. Comprehensive experiments on 6 time-series forecasting datasets and 5 time-series classification datasets have shown that TimeDRL consistently surpasses existing representation learning approaches, achieving an average improvement of forecasting by 57.98% in MSE and classification by 1.25% in accuracy. Furthermore, extensive ablation studies confirmed the relative contribution of each component in TimeDRL's architecture, and semi-supervised learning evaluations demonstrated its effectiveness in real-world scenarios, even with limited labeled data.

{{</citation>}}


### (19/136) Breaking the Entanglement of Homophily and Heterophily in Semi-supervised Node Classification (Henan Sun et al., 2023)

{{<citation>}}

Henan Sun, Xunkai Li, Zhengyu Wu, Daohan Su, Rong-Hua Li, Guoren Wang. (2023)  
**Breaking the Entanglement of Homophily and Heterophily in Semi-supervised Node Classification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.04111v1)  

---


**ABSTRACT**  
Recently, graph neural networks (GNNs) have shown prominent performance in semi-supervised node classification by leveraging knowledge from the graph database. However, most existing GNNs follow the homophily assumption, where connected nodes are more likely to exhibit similar feature distributions and the same labels, and such an assumption has proven to be vulnerable in a growing number of practical applications. As a supplement, heterophily reflects dissimilarity in connected nodes, which has gained significant attention in graph learning. To this end, data engineers aim to develop a powerful GNN model that can ensure performance under both homophily and heterophily. Despite numerous attempts, most existing GNNs struggle to achieve optimal node representations due to the constraints of undirected graphs. The neglect of directed edges results in sub-optimal graph representations, thereby hindering the capacity of GNNs. To address this issue, we introduce AMUD, which quantifies the relationship between node profiles and topology from a statistical perspective, offering valuable insights for \underline{A}daptively \underline{M}odeling the natural directed graphs as the \underline{U}ndirected or \underline{D}irected graph to maximize the benefits from subsequent graph learning. Furthermore, we propose \underline{A}daptive \underline{D}irected \underline{P}attern \underline{A}ggregation (ADPA) as a new directed graph learning paradigm for AMUD. Empirical studies have demonstrated that AMUD guides efficient graph learning. Meanwhile, extensive experiments on 14 benchmark datasets substantiate the impressive performance of ADPA, outperforming baselines by significant margins of 3.96\%.

{{</citation>}}


### (20/136) A Transformer Model for Symbolic Regression towards Scientific Discovery (Florian Lalande et al., 2023)

{{<citation>}}

Florian Lalande, Yoshitomo Matsubara, Naoya Chiba, Tatsunori Taniai, Ryo Igarashi, Yoshitala Ushiku. (2023)  
**A Transformer Model for Symbolic Regression towards Scientific Discovery**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.04070v1)  

---


**ABSTRACT**  
Symbolic Regression (SR) searches for mathematical expressions which best describe numerical datasets. This allows to circumvent interpretation issues inherent to artificial neural networks, but SR algorithms are often computationally expensive. This work proposes a new Transformer model aiming at Symbolic Regression particularly focused on its application for Scientific Discovery. We propose three encoder architectures with increasing flexibility but at the cost of column-permutation equivariance violation. Training results indicate that the most flexible architecture is required to prevent from overfitting. Once trained, we apply our best model to the SRSD datasets (Symbolic Regression for Scientific Discovery datasets) which yields state-of-the-art results using the normalized tree-based edit distance, at no extra computational cost.

{{</citation>}}


### (21/136) LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL Architectures (Vimal Thilak et al., 2023)

{{<citation>}}

Vimal Thilak, Chen Huang, Omid Saremi, Laurent Dinh, Hanlin Goh, Preetum Nakkiran, Joshua M. Susskind, Etai Littwin. (2023)  
**LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL Architectures**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.04000v1)  

---


**ABSTRACT**  
Joint embedding (JE) architectures have emerged as a promising avenue for acquiring transferable data representations. A key obstacle to using JE methods, however, is the inherent challenge of evaluating learned representations without access to a downstream task, and an annotated dataset. Without efficient and reliable evaluation, it is difficult to iterate on architectural and training choices for JE methods. In this paper, we introduce LiDAR (Linear Discriminant Analysis Rank), a metric designed to measure the quality of representations within JE architectures. Our metric addresses several shortcomings of recent approaches based on feature covariance rank by discriminating between informative and uninformative features. In essence, LiDAR quantifies the rank of the Linear Discriminant Analysis (LDA) matrix associated with the surrogate SSL task -- a measure that intuitively captures the information content as it pertains to solving the SSL task. We empirically demonstrate that LiDAR significantly surpasses naive rank based approaches in its predictive power of optimal hyperparameters. Our proposed criterion presents a more robust and intuitive means of assessing the quality of representations within JE architectures, which we hope facilitates broader adoption of these powerful techniques in various domains.

{{</citation>}}


### (22/136) Series2Vec: Similarity-based Self-supervised Representation Learning for Time Series Classification (Navid Mohammadi Foumani et al., 2023)

{{<citation>}}

Navid Mohammadi Foumani, Chang Wei Tan, Geoffrey I. Webb, Hamid Rezatofighi, Mahsa Salehi. (2023)  
**Series2Vec: Similarity-based Self-supervised Representation Learning for Time Series Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2312.03998v2)  

---


**ABSTRACT**  
We argue that time series analysis is fundamentally different in nature to either vision or natural language processing with respect to the forms of meaningful self-supervised learning tasks that can be defined. Motivated by this insight, we introduce a novel approach called \textit{Series2Vec} for self-supervised representation learning. Unlike other self-supervised methods in time series, which carry the risk of positive sample variants being less similar to the anchor sample than series in the negative set, Series2Vec is trained to predict the similarity between two series in both temporal and spectral domains through a self-supervised task. Series2Vec relies primarily on the consistency of the unsupervised similarity step, rather than the intrinsic quality of the similarity measurement, without the need for hand-crafted data augmentation. To further enforce the network to learn similar representations for similar time series, we propose a novel approach that applies order-invariant attention to each representation within the batch during training. Our evaluation of Series2Vec on nine large real-world datasets, along with the UCR/UEA archive, shows enhanced performance compared to current state-of-the-art self-supervised techniques for time series. Additionally, our extensive experiments show that Series2Vec performs comparably with fully supervised training and offers high efficiency in datasets with limited-labeled data. Finally, we show that the fusion of Series2Vec with other representation learning models leads to enhanced performance for time series classification. Code and models are open-source at \url{https://github.com/Navidfoumani/Series2Vec.}

{{</citation>}}


### (23/136) MICRO: Model-Based Offline Reinforcement Learning with a Conservative Bellman Operator (Xiao-Yin Liu et al., 2023)

{{<citation>}}

Xiao-Yin Liu, Xiao-Hu Zhou, Guo-Tao Li, Hao Li, Mei-Jiang Gui, Tian-Yu Xiang, De-Xing Huang, Zeng-Guang Hou. (2023)  
**MICRO: Model-Based Offline Reinforcement Learning with a Conservative Bellman Operator**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.03991v1)  

---


**ABSTRACT**  
Offline reinforcement learning (RL) faces a significant challenge of distribution shift. Model-free offline RL penalizes the Q value for out-of-distribution (OOD) data or constrains the policy closed to the behavior policy to tackle this problem, but this inhibits the exploration of the OOD region. Model-based offline RL, which uses the trained environment model to generate more OOD data and performs conservative policy optimization within that model, has become an effective method for this problem. However, the current model-based algorithms rarely consider agent robustness when incorporating conservatism into policy. Therefore, the new model-based offline algorithm with a conservative Bellman operator (MICRO) is proposed. This method trades off performance and robustness via introducing the robust Bellman operator into the algorithm. Compared with previous model-based algorithms with robust adversarial models, MICRO can significantly reduce the computation cost by only choosing the minimal Q value in the state uncertainty set. Extensive experiments demonstrate that MICRO outperforms prior RL algorithms in offline RL benchmark and is considerably robust to adversarial perturbations.

{{</citation>}}


## q-bio.QM (1)



### (24/136) Evaluating Zero-Shot Scoring for In Vitro Antibody Binding Prediction with Experimental Validation (Divya Nori et al., 2023)

{{<citation>}}

Divya Nori, Simon V. Mathis, Amir Shanehsazzadeh. (2023)  
**Evaluating Zero-Shot Scoring for In Vitro Antibody Binding Prediction with Experimental Validation**  

---
Primary Category: q-bio.QM  
Categories: cs-AI, q-bio-BM, q-bio-QM, q-bio.QM  
Keywords: AI, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.05273v1)  

---


**ABSTRACT**  
The success of therapeutic antibodies relies on their ability to selectively bind antigens. AI-based antibody design protocols have shown promise in generating epitope-specific designs. Many of these protocols use an inverse folding step to generate diverse sequences given a backbone structure. Due to prohibitive screening costs, it is key to identify candidate sequences likely to bind in vitro. Here, we compare the efficacy of 8 common scoring paradigms based on open-source models to classify antibody designs as binders or non-binders. We evaluate these approaches on a novel surface plasmon resonance (SPR) dataset, spanning 5 antigens. Our results show that existing methods struggle to detect binders, and performance is highly variable across antigens. We find that metrics computed on flexibly docked antibody-antigen complexes are more robust, and ensembles scores are more consistent than individual metrics. We provide experimental insight to analyze current scoring techniques, highlighting that the development of robust, zero-shot filters is an important research gap.

{{</citation>}}


## cs.CV (39)



### (25/136) StableQ: Enhancing Data-Scarce Quantization with Text-to-Image Data (Yuhang Li et al., 2023)

{{<citation>}}

Yuhang Li, Youngeun Kim, Donghyun Lee, Priyadarshini Panda. (2023)  
**StableQ: Enhancing Data-Scarce Quantization with Text-to-Image Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Quantization  
[Paper Link](http://arxiv.org/abs/2312.05272v1)  

---


**ABSTRACT**  
Though low-bit quantization enables efficient storage and inference of deep neural networks, it often requires the use of training data to maintain resilience against quantization errors. However, training data are frequently subject to privacy or copyright concerns. In this work, we address the challenge of Data-Scarce Quantization, where access to training data is severely limited or non-existent for quantization purposes. Conventional approaches typically rely on inverting dummy images or jointly training generative models to produce synthetic input samples. However, these methods struggle to accurately recreate complex objects in large-scale datasets like ImageNet. To overcome these limitations, we introduce StableQ, a novel method that utilizes an advanced text-to-image diffusion model to generate high-resolution, photo-realistic synthetic data. To verify the quality of the generated data, we implement two robust filtering mechanisms. These mechanisms are designed to select images that closely resemble the intrinsic characteristics of the actual training data. Furthermore, in scenarios where limited training data are available, we use these data to guide the synthetic data generation process by inverting a learnable token embedding in the text encoder. Our extensive experimental results demonstrate that StbaleQ sets a new benchmark in both zero-shot and few-shot quantization, outperforming existing methods in terms of accuracy and efficiency.

{{</citation>}}


### (26/136) Quilt-LLaVA: Visual Instruction Tuning by Extracting Localized Narratives from Open-Source Histopathology Videos (Mehmet Saygin Seyfioglu et al., 2023)

{{<citation>}}

Mehmet Saygin Seyfioglu, Wisdom O. Ikezogwo, Fatemeh Ghezloo, Ranjay Krishna, Linda Shapiro. (2023)  
**Quilt-LLaVA: Visual Instruction Tuning by Extracting Localized Narratives from Open-Source Histopathology Videos**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4, QA  
[Paper Link](http://arxiv.org/abs/2312.04746v1)  

---


**ABSTRACT**  
The gigapixel scale of whole slide images (WSIs) poses a challenge for histopathology multi-modal chatbots, requiring a global WSI analysis for diagnosis, compounding evidence from different WSI patches. Current visual instruction datasets, generated through large language models, focus on creating question/answer pairs for individual image patches, which may lack diagnostic capacity on their own in histopathology, further complicated by the absence of spatial grounding in histopathology image captions. To bridge this gap, we introduce Quilt-Instruct, a large-scale dataset of 107,131 histopathology-specific instruction question/answer pairs, that is collected by leveraging educational histopathology videos from YouTube, which provides spatial localization of captions by automatically extracting narrators' cursor movements. In addition, we provide contextual reasoning by extracting diagnosis and supporting facts from the entire video content to guide the extrapolative reasoning of GPT-4. Using Quilt-Instruct, we train Quilt-LLaVA, which can reason beyond the given single image patch, enabling diagnostic reasoning and the capability of spatial awareness. To evaluate Quilt-LLaVA, we propose a comprehensive evaluation dataset created from 985 images and 1283 human-generated question-answers. We also thoroughly evaluate Quilt-LLaVA using public histopathology datasets, where Quilt-LLaVA significantly outperforms SOTA by over 10% on relative GPT-4 score and 4% and 9% on open and closed set VQA. Our code, data, and model are publicly available at quilt-llava.github.io.

{{</citation>}}


### (27/136) gcDLSeg: Integrating Graph-cut into Deep Learning for Binary Semantic Segmentation (Hui Xie et al., 2023)

{{<citation>}}

Hui Xie, Weiyu Xu, Ya Xing Wang, John Buatti, Xiaodong Wu. (2023)  
**gcDLSeg: Integrating Graph-cut into Deep Learning for Binary Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, eess-SP  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.04713v1)  

---


**ABSTRACT**  
Binary semantic segmentation in computer vision is a fundamental problem. As a model-based segmentation method, the graph-cut approach was one of the most successful binary segmentation methods thanks to its global optimality guarantee of the solutions and its practical polynomial-time complexity. Recently, many deep learning (DL) based methods have been developed for this task and yielded remarkable performance, resulting in a paradigm shift in this field. To combine the strengths of both approaches, we propose in this study to integrate the graph-cut approach into a deep learning network for end-to-end learning. Unfortunately, backward propagation through the graph-cut module in the DL network is challenging due to the combinatorial nature of the graph-cut algorithm. To tackle this challenge, we propose a novel residual graph-cut loss and a quasi-residual connection, enabling the backward propagation of the gradients of the residual graph-cut loss for effective feature learning guided by the graph-cut segmentation model. In the inference phase, globally optimal segmentation is achieved with respect to the graph-cut energy defined on the optimized image features learned from DL networks. Experiments on the public AZH chronic wound data set and the pancreas cancer data set from the medical segmentation decathlon (MSD) demonstrated promising segmentation accuracy, and improved robustness against adversarial attacks.

{{</citation>}}


### (28/136) Image and AIS Data Fusion Technique for Maritime Computer Vision Applications (Emre Gülsoylu et al., 2023)

{{<citation>}}

Emre Gülsoylu, Paul Koch, Mert Yıldız, Manfred Constapel, André Peter Kelm. (2023)  
**Image and AIS Data Fusion Technique for Maritime Computer Vision Applications**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.05270v1)  

---


**ABSTRACT**  
Deep learning object detection methods, like YOLOv5, are effective in identifying maritime vessels but often lack detailed information important for practical applications. In this paper, we addressed this problem by developing a technique that fuses Automatic Identification System (AIS) data with vessels detected in images to create datasets. This fusion enriches ship images with vessel-related data, such as type, size, speed, and direction. Our approach associates detected ships to their corresponding AIS messages by estimating distance and azimuth using a homography-based method suitable for both fixed and periodically panning cameras. This technique is useful for creating datasets for waterway traffic management, encounter detection, and surveillance. We introduce a novel dataset comprising of images taken in various weather conditions and their corresponding AIS messages. This dataset offers a stable baseline for refining vessel detection algorithms and trajectory prediction models. To assess our method's performance, we manually annotated a portion of this dataset. The results are showing an overall association accuracy of 74.76 %, with the association accuracy for fixed cameras reaching 85.06 %. This demonstrates the potential of our approach in creating datasets for vessel detection, pose estimation and auto-labelling pipelines.

{{</citation>}}


### (29/136) LifelongMemory: Leveraging LLMs for Answering Queries in Egocentric Videos (Ying Wang et al., 2023)

{{<citation>}}

Ying Wang, Yanlai Yang, Mengye Ren. (2023)  
**LifelongMemory: Leveraging LLMs for Answering Queries in Egocentric Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.05269v1)  

---


**ABSTRACT**  
The egocentric video natural language query (NLQ) task involves localizing a temporal window in an egocentric video that provides an answer to a posed query, which has wide applications in building personalized AI assistants. Prior methods for this task have focused on improvements of network architecture and leveraging pre-training for enhanced image and video features, but have struggled with capturing long-range temporal dependencies in lengthy videos, and cumbersome end-to-end training. Motivated by recent advancements in Large Language Models (LLMs) and vision language models, we introduce LifelongMemory, a novel framework that utilizes multiple pre-trained models to answer queries from extensive egocentric video content. We address the unique challenge by employing a pre-trained captioning model to create detailed narratives of the videos. These narratives are then used to prompt a frozen LLM to generate coarse-grained temporal window predictions, which are subsequently refined using a pre-trained NLQ model. Empirical results demonstrate that our method achieves competitive performance against existing supervised end-to-end learning methods, underlining the potential of integrating multiple pre-trained multimodal large language models in complex vision-language tasks. We provide a comprehensive analysis of key design decisions and hyperparameters in our pipeline, offering insights and practical guidelines.

{{</citation>}}


### (30/136) Scaling Laws of Synthetic Images for Model Training ... for Now (Lijie Fan et al., 2023)

{{<citation>}}

Lijie Fan, Kaifeng Chen, Dilip Krishnan, Dina Katabi, Phillip Isola, Yonglong Tian. (2023)  
**Scaling Laws of Synthetic Images for Model Training ... for Now**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.04567v1)  

---


**ABSTRACT**  
Recent significant advances in text-to-image models unlock the possibility of training vision systems using synthetic images, potentially overcoming the difficulty of collecting curated data at scale. It is unclear, however, how these models behave at scale, as more synthetic data is added to the training set. In this paper we study the scaling laws of synthetic images generated by state of the art text-to-image models, for the training of supervised models: image classifiers with label supervision, and CLIP with language supervision. We identify several factors, including text prompts, classifier-free guidance scale, and types of text-to-image models, that significantly affect scaling behavior. After tuning these factors, we observe that synthetic images demonstrate a scaling trend similar to, but slightly less effective than, real images in CLIP training, while they significantly underperform in scaling when training supervised image classifiers. Our analysis indicates that the main reason for this underperformance is the inability of off-the-shelf text-to-image models to generate certain concepts, a limitation that significantly impairs the training of image classifiers. Our findings also suggest that scaling synthetic data can be particularly effective in scenarios such as: (1) when there is a limited supply of real images for a supervised problem (e.g., fewer than 0.5 million images in ImageNet), (2) when the evaluation dataset diverges significantly from the training data, indicating the out-of-distribution scenario, or (3) when synthetic data is used in conjunction with real images, as demonstrated in the training of CLIP models.

{{</citation>}}


### (31/136) GenTron: Delving Deep into Diffusion Transformers for Image and Video Generation (Shoufa Chen et al., 2023)

{{<citation>}}

Shoufa Chen, Mengmeng Xu, Jiawei Ren, Yuren Cong, Sen He, Yanping Xie, Animesh Sinha, Ping Luo, Tao Xiang, Juan-Manuel Perez-Rua. (2023)  
**GenTron: Delving Deep into Diffusion Transformers for Image and Video Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.04557v1)  

---


**ABSTRACT**  
In this study, we explore Transformer-based diffusion models for image and video generation. Despite the dominance of Transformer architectures in various fields due to their flexibility and scalability, the visual generative domain primarily utilizes CNN-based U-Net architectures, particularly in diffusion-based models. We introduce GenTron, a family of Generative models employing Transformer-based diffusion, to address this gap. Our initial step was to adapt Diffusion Transformers (DiTs) from class to text conditioning, a process involving thorough empirical exploration of the conditioning mechanism. We then scale GenTron from approximately 900M to over 3B parameters, observing significant improvements in visual quality. Furthermore, we extend GenTron to text-to-video generation, incorporating novel motion-free guidance to enhance video quality. In human evaluations against SDXL, GenTron achieves a 51.1% win rate in visual quality (with a 19.8% draw rate), and a 42.3% win rate in text alignment (with a 42.9% draw rate). GenTron also excels in the T2I-CompBench, underscoring its strengths in compositional generation. We believe this work will provide meaningful insights and serve as a valuable reference for future research.

{{</citation>}}


### (32/136) Self-Guided Open-Vocabulary Semantic Segmentation (Osman Ülger et al., 2023)

{{<citation>}}

Osman Ülger, Maksymilian Kulicki, Yuki Asano, Martin R. Oswald. (2023)  
**Self-Guided Open-Vocabulary Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.04539v1)  

---


**ABSTRACT**  
Vision-Language Models (VLMs) have emerged as promising tools for open-ended image understanding tasks, including open vocabulary segmentation. Yet, direct application of such VLMs to segmentation is non-trivial, since VLMs are trained with image-text pairs and naturally lack pixel-level granularity. Recent works have made advancements in bridging this gap, often by leveraging the shared image-text space in which the image and a provided text prompt are represented. In this paper, we challenge the capabilities of VLMs further and tackle open-vocabulary segmentation without the need for any textual input. To this end, we propose a novel Self-Guided Semantic Segmentation (Self-Seg) framework. Self-Seg is capable of automatically detecting relevant class names from clustered BLIP embeddings and using these for accurate semantic segmentation. In addition, we propose an LLM-based Open-Vocabulary Evaluator (LOVE) to effectively assess predicted open-vocabulary class names. We achieve state-of-the-art results on Pascal VOC, ADE20K and CityScapes for open-vocabulary segmentation without given class names, as well as competitive performance with methods where class names are given. All code and data will be released.

{{</citation>}}


### (33/136) Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping (Alex Costanzino et al., 2023)

{{<citation>}}

Alex Costanzino, Pierluigi Zama Ramirez, Giuseppe Lisanti, Luigi Di Stefano. (2023)  
**Multimodal Industrial Anomaly Detection by Crossmodal Feature Mapping**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.04521v1)  

---


**ABSTRACT**  
The paper explores the industrial multimodal Anomaly Detection (AD) task, which exploits point clouds and RGB images to localize anomalies. We introduce a novel light and fast framework that learns to map features from one modality to the other on nominal samples. At test time, anomalies are detected by pinpointing inconsistencies between observed and mapped features. Extensive experiments show that our approach achieves state-of-the-art detection and segmentation performance in both the standard and few-shot settings on the MVTec 3D-AD dataset while achieving faster inference and occupying less memory than previous multimodal AD methods. Moreover, we propose a layer-pruning technique to improve memory and time efficiency with a marginal sacrifice in performance.

{{</citation>}}


### (34/136) Bootstrapping Autonomous Radars with Self-Supervised Learning (Yiduo Hao et al., 2023)

{{<citation>}}

Yiduo Hao, Sohrab Madani, Junfeng Guan, Mohammed Alloulah, Saurabh Gupta, Haitham Hassanieh. (2023)  
**Bootstrapping Autonomous Radars with Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.04519v2)  

---


**ABSTRACT**  
The perception of autonomous vehicles using radars has attracted increased research interest due its ability to operate in fog and bad weather. However, training radar models is hindered by the cost and difficulty of annotating large-scale radar data. To overcome this bottleneck, we propose a self-supervised learning framework to leverage the large amount of unlabeled radar data to pre-train radar-only embeddings for self-driving perception tasks. The proposed method combines radar-to-radar and radar-to-vision contrastive losses to learn a general representation from unlabeled radar heatmaps paired with their corresponding camera images. When used for downstream object detection, we demonstrate that the proposed self-supervision framework can improve the accuracy of state-of-the-art supervised baselines by 5.8% in mAP.

{{</citation>}}


### (35/136) GSGFormer: Generative Social Graph Transformer for Multimodal Pedestrian Trajectory Prediction (Zhongchang Luo et al., 2023)

{{<citation>}}

Zhongchang Luo, Marion Robin, Pavan Vasishta. (2023)  
**GSGFormer: Generative Social Graph Transformer for Multimodal Pedestrian Trajectory Prediction**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.04479v1)  

---


**ABSTRACT**  
Pedestrian trajectory prediction, vital for selfdriving cars and socially-aware robots, is complicated due to intricate interactions between pedestrians, their environment, and other Vulnerable Road Users. This paper presents GSGFormer, an innovative generative model adept at predicting pedestrian trajectories by considering these complex interactions and offering a plethora of potential modal behaviors. We incorporate a heterogeneous graph neural network to capture interactions between pedestrians, semantic maps, and potential destinations. The Transformer module extracts temporal features, while our novel CVAE-Residual-GMM module promotes diverse behavioral modality generation. Through evaluations on multiple public datasets, GSGFormer not only outperforms leading methods with ample data but also remains competitive when data is limited.

{{</citation>}}


### (36/136) PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding (Zhen Li et al., 2023)

{{<citation>}}

Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, Ying Shan. (2023)  
**PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.04461v1)  

---


**ABSTRACT**  
Recent advances in text-to-image generation have made remarkable progress in synthesizing realistic human photos conditioned on given text prompts. However, existing personalized generation methods cannot simultaneously satisfy the requirements of high efficiency, promising identity (ID) fidelity, and flexible text controllability. In this work, we introduce PhotoMaker, an efficient personalized text-to-image generation method, which mainly encodes an arbitrary number of input ID images into a stack ID embedding for preserving ID information. Such an embedding, serving as a unified ID representation, can not only encapsulate the characteristics of the same input ID comprehensively, but also accommodate the characteristics of different IDs for subsequent integration. This paves the way for more intriguing and practically valuable applications. Besides, to drive the training of our PhotoMaker, we propose an ID-oriented data construction pipeline to assemble the training data. Under the nourishment of the dataset constructed through the proposed pipeline, our PhotoMaker demonstrates better ID preservation ability than test-time fine-tuning based methods, yet provides significant speed improvements, high-quality generation results, strong generalization capabilities, and a wide range of applications. Our project page is available at https://photo-maker.github.io/

{{</citation>}}


### (37/136) OT-Attack: Enhancing Adversarial Transferability of Vision-Language Models via Optimal Transport Optimization (Dongchen Han et al., 2023)

{{<citation>}}

Dongchen Han, Xiaojun Jia, Yang Bai, Jindong Gu, Yang Liu, Xiaochun Cao. (2023)  
**OT-Attack: Enhancing Adversarial Transferability of Vision-Language Models via Optimal Transport Optimization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04403v1)  

---


**ABSTRACT**  
Vision-language pre-training (VLP) models demonstrate impressive abilities in processing both images and text. However, they are vulnerable to multi-modal adversarial examples (AEs). Investigating the generation of high-transferability adversarial examples is crucial for uncovering VLP models' vulnerabilities in practical scenarios. Recent works have indicated that leveraging data augmentation and image-text modal interactions can enhance the transferability of adversarial examples for VLP models significantly. However, they do not consider the optimal alignment problem between dataaugmented image-text pairs. This oversight leads to adversarial examples that are overly tailored to the source model, thus limiting improvements in transferability. In our research, we first explore the interplay between image sets produced through data augmentation and their corresponding text sets. We find that augmented image samples can align optimally with certain texts while exhibiting less relevance to others. Motivated by this, we propose an Optimal Transport-based Adversarial Attack, dubbed OT-Attack. The proposed method formulates the features of image and text sets as two distinct distributions and employs optimal transport theory to determine the most efficient mapping between them. This optimal mapping informs our generation of adversarial examples to effectively counteract the overfitting issues. Extensive experiments across various network architectures and datasets in image-text matching tasks reveal that our OT-Attack outperforms existing state-of-the-art methods in terms of adversarial transferability.

{{</citation>}}


### (38/136) Intelligent Anomaly Detection for Lane Rendering Using Transformer with Self-Supervised Pre-Training and Customized Fine-Tuning (Yongqi Dong et al., 2023)

{{<citation>}}

Yongqi Dong, Xingmin Lu, Ruohan Li, Wei Song, Bart van Arem, Haneen Farah. (2023)  
**Intelligent Anomaly Detection for Lane Rendering Using Transformer with Self-Supervised Pre-Training and Customized Fine-Tuning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, eess-IV, stat-ML  
Keywords: Anomaly Detection, Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2312.04398v1)  

---


**ABSTRACT**  
The burgeoning navigation services using digital maps provide great convenience to drivers. Nevertheless, the presence of anomalies in lane rendering map images occasionally introduces potential hazards, as such anomalies can be misleading to human drivers and consequently contribute to unsafe driving conditions. In response to this concern and to accurately and effectively detect the anomalies, this paper transforms lane rendering image anomaly detection into a classification problem and proposes a four-phase pipeline consisting of data pre-processing, self-supervised pre-training with the masked image modeling (MiM) method, customized fine-tuning using cross-entropy based loss with label smoothing, and post-processing to tackle it leveraging state-of-the-art deep learning techniques, especially those involving Transformer models. Various experiments verify the effectiveness of the proposed pipeline. Results indicate that the proposed pipeline exhibits superior performance in lane rendering image anomaly detection, and notably, the self-supervised pre-training with MiM can greatly enhance the detection accuracy while significantly reducing the total training time. For instance, employing the Swin Transformer with Uniform Masking as self-supervised pretraining (Swin-Trans-UM) yielded a heightened accuracy at 94.77% and an improved Area Under The Curve (AUC) score of 0.9743 compared with the pure Swin Transformer without pre-training (Swin-Trans) with an accuracy of 94.01% and an AUC of 0.9498. The fine-tuning epochs were dramatically reduced to 41 from the original 280. In conclusion, the proposed pipeline, with its incorporation of self-supervised pre-training using MiM and other advanced deep learning techniques, emerges as a robust solution for enhancing the accuracy and efficiency of lane rendering image anomaly detection in digital navigation systems.

{{</citation>}}


### (39/136) DemoCaricature: Democratising Caricature Generation with a Rough Sketch (Dar-Yen Chen et al., 2023)

{{<citation>}}

Dar-Yen Chen, Subhadeep Koley, Aneeshan Sain, Pinaki Nath Chowdhury, Tao Xiang, Ayan Kumar Bhunia, Yi-Zhe Song. (2023)  
**DemoCaricature: Democratising Caricature Generation with a Rough Sketch**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2312.04364v1)  

---


**ABSTRACT**  
In this paper, we democratise caricature generation, empowering individuals to effortlessly craft personalised caricatures with just a photo and a conceptual sketch. Our objective is to strike a delicate balance between abstraction and identity, while preserving the creativity and subjectivity inherent in a sketch. To achieve this, we present Explicit Rank-1 Model Editing alongside single-image personalisation, selectively applying nuanced edits to cross-attention layers for a seamless merge of identity and style. Additionally, we propose Random Mask Reconstruction to enhance robustness, directing the model to focus on distinctive identity and style features. Crucially, our aim is not to replace artists but to eliminate accessibility barriers, allowing enthusiasts to engage in the artistry.

{{</citation>}}


### (40/136) Multi-View Unsupervised Image Generation with Cross Attention Guidance (Llukman Cerkezi et al., 2023)

{{<citation>}}

Llukman Cerkezi, Aram Davtyan, Sepehr Sameni, Paolo Favaro. (2023)  
**Multi-View Unsupervised Image Generation with Cross Attention Guidance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.04337v1)  

---


**ABSTRACT**  
The growing interest in novel view synthesis, driven by Neural Radiance Field (NeRF) models, is hindered by scalability issues due to their reliance on precisely annotated multi-view images. Recent models address this by fine-tuning large text2image diffusion models on synthetic multi-view data. Despite robust zero-shot generalization, they may need post-processing and can face quality issues due to the synthetic-real domain gap. This paper introduces a novel pipeline for unsupervised training of a pose-conditioned diffusion model on single-category datasets. With the help of pretrained self-supervised Vision Transformers (DINOv2), we identify object poses by clustering the dataset through comparing visibility and locations of specific object parts. The pose-conditioned diffusion model, trained on pose labels, and equipped with cross-frame attention at inference time ensures cross-view consistency, that is further aided by our novel hard-attention guidance. Our model, MIRAGE, surpasses prior work in novel view synthesis on real images. Furthermore, MIRAGE is robust to diverse textures and geometries, as demonstrated with our experiments on synthetic images generated with pretrained Stable Diffusion.

{{</citation>}}


### (41/136) Towards a Perceptual Evaluation Framework for Lighting Estimation (Justine Giroux et al., 2023)

{{<citation>}}

Justine Giroux, Mohammad Reza Karimi Dastjerdi, Yannick Hold-Geoffroy, Javier Vazquez-Corral, Jean-François Lalonde. (2023)  
**Towards a Perceptual Evaluation Framework for Lighting Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.04334v1)  

---


**ABSTRACT**  
Progress in lighting estimation is tracked by computing existing image quality assessment (IQA) metrics on images from standard datasets. While this may appear to be a reasonable approach, we demonstrate that doing so does not correlate to human preference when the estimated lighting is used to relight a virtual scene into a real photograph. To study this, we design a controlled psychophysical experiment where human observers must choose their preference amongst rendered scenes lit using a set of lighting estimation algorithms selected from the recent literature, and use it to analyse how these algorithms perform according to human perception. Then, we demonstrate that none of the most popular IQA metrics from the literature, taken individually, correctly represent human perception. Finally, we show that by learning a combination of existing IQA metrics, we can more accurately represent human preference. This provides a new perceptual framework to help evaluate future lighting estimation algorithms.

{{</citation>}}


### (42/136) GPT4SGG: Synthesizing Scene Graphs from Holistic and Region-specific Narratives (Zuyao Chen et al., 2023)

{{<citation>}}

Zuyao Chen, Jinlin Wu, Zhen Lei, Zhaoxiang Zhang, Changwen Chen. (2023)  
**GPT4SGG: Synthesizing Scene Graphs from Holistic and Region-specific Narratives**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.04314v1)  

---


**ABSTRACT**  
Learning scene graphs from natural language descriptions has proven to be a cheap and promising scheme for Scene Graph Generation (SGG). However, such unstructured caption data and its processing are troubling the learning an acurrate and complete scene graph. This dilema can be summarized as three points. First, traditional language parsers often fail to extract meaningful relationship triplets from caption data. Second, grounding unlocalized objects in parsed triplets will meet ambiguity in visual-language alignment. Last, caption data typically are sparse and exhibit bias to partial observations of image content. These three issues make it hard for the model to generate comprehensive and accurate scene graphs. To fill this gap, we propose a simple yet effective framework, GPT4SGG, to synthesize scene graphs from holistic and region-specific narratives. The framework discards traditional language parser, and localize objects before obtaining relationship triplets. To obtain relationship triplets, holistic and dense region-specific narratives are generated from the image. With such textual representation of image data and a task-specific prompt, an LLM, particularly GPT-4, directly synthesizes a scene graph as "pseudo labels". Experimental results showcase GPT4SGG significantly improves the performance of SGG models trained on image-caption data. We believe this pioneering work can motivate further research into mining the visual reasoning capabilities of LLMs.

{{</citation>}}


### (43/136) GPT-4V with Emotion: A Zero-shot Benchmark for Multimodal Emotion Understanding (Zheng Lian et al., 2023)

{{<citation>}}

Zheng Lian, Licai Sun, Haiyang Sun, Kang Chen, Zhuofan Wen, Hao Gu, Shun Chen, Bin Liu, Jianhua Tao. (2023)  
**GPT-4V with Emotion: A Zero-shot Benchmark for Multimodal Emotion Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.04293v1)  

---


**ABSTRACT**  
Recently, GPT-4 with Vision (GPT-4V) has shown remarkable performance across various multimodal tasks. However, its efficacy in emotion recognition remains a question. This paper quantitatively evaluates GPT-4V's capabilities in multimodal emotion understanding, encompassing tasks such as facial emotion recognition, visual sentiment analysis, micro-expression recognition, dynamic facial emotion recognition, and multimodal emotion recognition. Our experiments show that GPT-4V exhibits impressive multimodal and temporal understanding capabilities, even surpassing supervised systems in some tasks. Despite these achievements, GPT-4V is currently tailored for general domains. It performs poorly in micro-expression recognition that requires specialized expertise. The main purpose of this paper is to present quantitative results of GPT-4V on emotion understanding and establish a zero-shot benchmark for future research. Code and evaluation results are available at: https://github.com/zeroQiaoba/gpt4v-emotion.

{{</citation>}}


### (44/136) Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation (Zhixiang Wei et al., 2023)

{{<citation>}}

Zhixiang Wei, Lin Chen, Yi Jin, Xiaoxiao Ma, Tianle Liu, Pengyang Lin, Ben Wang, Huaian Chen, Jinjin Zheng. (2023)  
**Stronger, Fewer, & Superior: Harnessing Vision Foundation Models for Domain Generalized Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.04265v1)  

---


**ABSTRACT**  
In this paper, we first assess and harness various Vision Foundation Models (VFMs) in the context of Domain Generalized Semantic Segmentation (DGSS). Driven by the motivation that Leveraging Stronger pre-trained models and Fewer trainable parameters for Superior generalizability, we introduce a robust fine-tuning approach, namely Rein, to parameter-efficiently harness VFMs for DGSS. Built upon a set of trainable tokens, each linked to distinct instances, Rein precisely refines and forwards the feature maps from each layer to the next layer within the backbone. This process produces diverse refinements for different categories within a single image. With fewer trainable parameters, Rein efficiently fine-tunes VFMs for DGSS tasks, surprisingly surpassing full parameter fine-tuning. Extensive experiments across various settings demonstrate that Rein significantly outperforms state-of-the-art methods. Remarkably, with just an extra 1% of trainable parameters within the frozen backbone, Rein achieves a mIoU of 68.1% on the Cityscapes, without accessing any real urban-scene datasets.

{{</citation>}}


### (45/136) TeMO: Towards Text-Driven 3D Stylization for Multi-Object Meshes (Xuying Zhang et al., 2023)

{{<citation>}}

Xuying Zhang, Bo-Wen Yin, Yuming Chen, Zheng Lin, Yunheng Li, Qibin Hou, Ming-Ming Cheng. (2023)  
**TeMO: Towards Text-Driven 3D Stylization for Multi-Object Meshes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.04248v1)  

---


**ABSTRACT**  
Recent progress in the text-driven 3D stylization of a single object has been considerably promoted by CLIP-based methods. However, the stylization of multi-object 3D scenes is still impeded in that the image-text pairs used for pre-training CLIP mostly consist of an object. Meanwhile, the local details of multiple objects may be susceptible to omission due to the existing supervision manner primarily relying on coarse-grained contrast of image-text pairs. To overcome these challenges, we present a novel framework, dubbed TeMO, to parse multi-object 3D scenes and edit their styles under the contrast supervision at multiple levels. We first propose a Decoupled Graph Attention (DGA) module to distinguishably reinforce the features of 3D surface points. Particularly, a cross-modal graph is constructed to align the object points accurately and noun phrases decoupled from the 3D mesh and textual description. Then, we develop a Cross-Grained Contrast (CGC) supervision system, where a fine-grained loss between the words in the textual description and the randomly rendered images are constructed to complement the coarse-grained loss. Extensive experiments show that our method can synthesize high-quality stylized content and outperform the existing methods over a wide range of multi-object 3D meshes. Our code and results will be made publicly available

{{</citation>}}


### (46/136) ZePT: Zero-Shot Pan-Tumor Segmentation via Query-Disentangling and Self-Prompting (Yankai Jiang et al., 2023)

{{<citation>}}

Yankai Jiang, Zhongzhen Huang, Rongzhao Zhang, Xiaofan Zhang, Shaoting Zhang. (2023)  
**ZePT: Zero-Shot Pan-Tumor Segmentation via Query-Disentangling and Self-Prompting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.04964v1)  

---


**ABSTRACT**  
The long-tailed distribution problem in medical image analysis reflects a high prevalence of common conditions and a low prevalence of rare ones, which poses a significant challenge in developing a unified model capable of identifying rare or novel tumor categories not encountered during training. In this paper, we propose a new zero-shot pan-tumor segmentation framework (ZePT) based on query-disentangling and self-prompting to segment unseen tumor categories beyond the training set. ZePT disentangles the object queries into two subsets and trains them in two stages. Initially, it learns a set of fundamental queries for organ segmentation through an object-aware feature grouping strategy, which gathers organ-level visual features. Subsequently, it refines the other set of advanced queries that focus on the auto-generated visual prompts for unseen tumor segmentation. Moreover, we introduce query-knowledge alignment at the feature level to enhance each query's discriminative representation and generalizability. Extensive experiments on various tumor segmentation tasks demonstrate the performance superiority of ZePT, which surpasses the previous counterparts and evidence the promising ability for zero-shot tumor segmentation in real-world settings. Codes will be made publicly available.

{{</citation>}}


### (47/136) Fine-tune vision foundation model for crack segmentation in civil infrastructures (Kang Ge et al., 2023)

{{<citation>}}

Kang Ge, Chen Wang, Yutao Guo. (2023)  
**Fine-tune vision foundation model for crack segmentation in civil infrastructures**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04233v1)  

---


**ABSTRACT**  
Large-scale foundation models have become the mainstream method in the field of deep learning, while in civil engineering, the scale of AI models is strictly limited. In this work, vision foundation model is introduced for crack segmentation. Two Parameter-efficient fine-tuning methods, adapter and low-rank adaptation, are adopted to fine-tune the foundation model in the field of semantic segmentation: Segment Anything Model (SAM). The fine-tuned model CrackSAM is much larger than all the existing crack segmentation models, but shows excellent performance. To test the zero-shot performance of the proposed method, two unique datasets related to road and exterior wall cracks are collected, annotated and open-sourced, in total 810 images. Comparative experiments are conducted with twelve mature semantic segmentation models. On datasets with artificial noise and previously unseen datasets, the performance of CrackSAM far exceeds that of all state-of-the-art models. CrackSAM exhibits remarkable superiority, particularly in challenging conditions such as dim lighting, shadows, road markings, construction joints, and other interference factors. Such cross-scenario results demonstrate the outstanding zero-shot capability of foundation models, and provide new ideas for the development of vision models in civil engineering.

{{</citation>}}


### (48/136) Adventures of Trustworthy Vision-Language Models: A Survey (Mayank Vatsa et al., 2023)

{{<citation>}}

Mayank Vatsa, Anubhooti Jain, Richa Singh. (2023)  
**Adventures of Trustworthy Vision-Language Models: A Survey**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04231v1)  

---


**ABSTRACT**  
Recently, transformers have become incredibly popular in computer vision and vision-language tasks. This notable rise in their usage can be primarily attributed to the capabilities offered by attention mechanisms and the outstanding ability of transformers to adapt and apply themselves to a variety of tasks and domains. Their versatility and state-of-the-art performance have established them as indispensable tools for a wide array of applications. However, in the constantly changing landscape of machine learning, the assurance of the trustworthiness of transformers holds utmost importance. This paper conducts a thorough examination of vision-language transformers, employing three fundamental principles of responsible AI: Bias, Robustness, and Interpretability. The primary objective of this paper is to delve into the intricacies and complexities associated with the practical use of transformers, with the overarching goal of advancing our comprehension of how to enhance their reliability and accountability.

{{</citation>}}


### (49/136) TLCE: Transfer-Learning Based Classifier Ensembles for Few-Shot Class-Incremental Learning (Shuangmei Wang et al., 2023)

{{<citation>}}

Shuangmei Wang, Yang Cao, Tieru Wu. (2023)  
**TLCE: Transfer-Learning Based Classifier Ensembles for Few-Shot Class-Incremental Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.04225v1)  

---


**ABSTRACT**  
Few-shot class-incremental learning (FSCIL) struggles to incrementally recognize novel classes from few examples without catastrophic forgetting of old classes or overfitting to new classes. We propose TLCE, which ensembles multiple pre-trained models to improve separation of novel and old classes. TLCE minimizes interference between old and new classes by mapping old class images to quasi-orthogonal prototypes using episodic training. It then ensembles diverse pre-trained models to better adapt to novel classes despite data imbalance. Extensive experiments on various datasets demonstrate that our transfer learning ensemble approach outperforms state-of-the-art FSCIL methods.

{{</citation>}}


### (50/136) Joint-Individual Fusion Structure with Fusion Attention Module for Multi-Modal Skin Cancer Classification (Peng Tang et al., 2023)

{{<citation>}}

Peng Tang, Xintong Yan, Yang Nan, Xiaobin Hu, Xiaobin Hu, Bjoern H Menzee. Sebastian Krammer, Tobias Lasser. (2023)  
**Joint-Individual Fusion Structure with Fusion Attention Module for Multi-Modal Skin Cancer Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.04189v1)  

---


**ABSTRACT**  
Most convolutional neural network (CNN) based methods for skin cancer classification obtain their results using only dermatological images. Although good classification results have been shown, more accurate results can be achieved by considering the patient's metadata, which is valuable clinical information for dermatologists. Current methods only use the simple joint fusion structure (FS) and fusion modules (FMs) for the multi-modal classification methods, there still is room to increase the accuracy by exploring more advanced FS and FM. Therefore, in this paper, we design a new fusion method that combines dermatological images (dermoscopy images or clinical images) and patient metadata for skin cancer classification from the perspectives of FS and FM. First, we propose a joint-individual fusion (JIF) structure that learns the shared features of multi-modality data and preserves specific features simultaneously. Second, we introduce a fusion attention (FA) module that enhances the most relevant image and metadata features based on both the self and mutual attention mechanism to support the decision-making pipeline. We compare the proposed JIF-MMFA method with other state-of-the-art fusion methods on three different public datasets. The results show that our JIF-MMFA method improves the classification results for all tested CNN backbones and performs better than the other fusion methods on the three public datasets, demonstrating our method's effectiveness and robustness

{{</citation>}}


### (51/136) Augmentation-Free Dense Contrastive Knowledge Distillation for Efficient Semantic Segmentation (Jiawei Fan et al., 2023)

{{<citation>}}

Jiawei Fan, Chao Li, Xiaolong Liu, Meina Song, Anbang Yao. (2023)  
**Augmentation-Free Dense Contrastive Knowledge Distillation for Efficient Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Augmentation, Knowledge Distillation, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.04168v1)  

---


**ABSTRACT**  
In recent years, knowledge distillation methods based on contrastive learning have achieved promising results on image classification and object detection tasks. However, in this line of research, we note that less attention is paid to semantic segmentation. Existing methods heavily rely on data augmentation and memory buffer, which entail high computational resource demands when applying them to handle semantic segmentation that requires to preserve high-resolution feature maps for making dense pixel-wise predictions. In order to address this problem, we present Augmentation-free Dense Contrastive Knowledge Distillation (Af-DCD), a new contrastive distillation learning paradigm to train compact and accurate deep neural networks for semantic segmentation applications. Af-DCD leverages a masked feature mimicking strategy, and formulates a novel contrastive learning loss via taking advantage of tactful feature partitions across both channel and spatial dimensions, allowing to effectively transfer dense and structured local knowledge learnt by the teacher model to a target student model while maintaining training efficiency. Extensive experiments on five mainstream benchmarks with various teacher-student network pairs demonstrate the effectiveness of our approach. For instance, the DeepLabV3-Res18|DeepLabV3-MBV2 model trained by Af-DCD reaches 77.03%|76.38% mIOU on Cityscapes dataset when choosing DeepLabV3-Res101 as the teacher, setting new performance records. Besides that, Af-DCD achieves an absolute mIOU improvement of 3.26%|3.04%|2.75%|2.30%|1.42% compared with individually trained counterpart on Cityscapes|Pascal VOC|Camvid|ADE20K|COCO-Stuff-164K. Code is available at https://github.com/OSVAI/Af-DCD

{{</citation>}}


### (52/136) EulerMormer: Robust Eulerian Motion Magnification via Dynamic Filtering within Transformer (Fei Wang et al., 2023)

{{<citation>}}

Fei Wang, Dan Guo, Kun Li, Meng Wang. (2023)  
**EulerMormer: Robust Eulerian Motion Magnification via Dynamic Filtering within Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.04152v1)  

---


**ABSTRACT**  
Video Motion Magnification (VMM) aims to break the resolution limit of human visual perception capability and reveal the imperceptible minor motion that contains valuable information in the macroscopic domain. However, challenges arise in this task due to photon noise inevitably introduced by photographic devices and spatial inconsistency in amplification, leading to flickering artifacts in static fields and motion blur and distortion in dynamic fields in the video. Existing methods focus on explicit motion modeling without emphasizing prioritized denoising during the motion magnification process. This paper proposes a novel dynamic filtering strategy to achieve static-dynamic field adaptive denoising. Specifically, based on Eulerian theory, we separate texture and shape to extract motion representation through inter-frame shape differences, expecting to leverage these subdivided features to solve this task finely. Then, we introduce a novel dynamic filter that eliminates noise cues and preserves critical features in the motion magnification and amplification generation phases. Overall, our unified framework, EulerMormer, is a pioneering effort to first equip with Transformer in learning-based VMM. The core of the dynamic filter lies in a global dynamic sparse cross-covariance attention mechanism that explicitly removes noise while preserving vital information, coupled with a multi-scale dual-path gating mechanism that selectively regulates the dependence on different frequency features to reduce spatial attenuation and complement motion boundaries. We demonstrate extensive experiments that EulerMormer achieves more robust video motion magnification from the Eulerian perspective, significantly outperforming state-of-the-art methods. The source code is available at https://github.com/VUT-HFUT/EulerMormer.

{{</citation>}}


### (53/136) A Multilevel Guidance-Exploration Network and Behavior-Scene Matching Method for Human Behavior Anomaly Detection (Guoqing Yang et al., 2023)

{{<citation>}}

Guoqing Yang, Zhiming Luo, Jianzhe Gao, Yingxin Lai, Kun Yang, Yifan He, Shaozi Li. (2023)  
**A Multilevel Guidance-Exploration Network and Behavior-Scene Matching Method for Human Behavior Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.04119v1)  

---


**ABSTRACT**  
Human behavior anomaly detection aims to identify unusual human actions, playing a crucial role in intelligent surveillance and other areas. The current mainstream methods still adopt reconstruction or future frame prediction techniques. However, reconstructing or predicting low-level pixel features easily enables the network to achieve overly strong generalization ability, allowing anomalies to be reconstructed or predicted as effectively as normal data. Different from their methods, inspired by the Student-Teacher Network, we propose a novel framework called the Multilevel Guidance-Exploration Network(MGENet), which detects anomalies through the difference in high-level representation between the Guidance and Exploration network. Specifically, we first utilize the pre-trained Normalizing Flow that takes skeletal keypoints as input to guide an RGB encoder, which takes unmasked RGB frames as input, to explore motion latent features. Then, the RGB encoder guides the mask encoder, which takes masked RGB frames as input, to explore the latent appearance feature. Additionally, we design a Behavior-Scene Matching Module(BSMM) to detect scene-related behavioral anomalies. Extensive experiments demonstrate that our proposed method achieves state-of-the-art performance on ShanghaiTech and UBnormal datasets, with AUC of 86.9 % and 73.5 %, respectively. The code will be available on https://github.com/molu-ggg/GENet.

{{</citation>}}


### (54/136) DeepFidelity: Perceptual Forgery Fidelity Assessment for Deepfake Detection (Chunlei Peng et al., 2023)

{{<citation>}}

Chunlei Peng, Huiqing Guo, Decheng Liu, Nannan Wang, Ruimin Hu, Xinbo Gao. (2023)  
**DeepFidelity: Perceptual Forgery Fidelity Assessment for Deepfake Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Augmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2312.04961v1)  

---


**ABSTRACT**  
Deepfake detection refers to detecting artificially generated or edited faces in images or videos, which plays an essential role in visual information security. Despite promising progress in recent years, Deepfake detection remains a challenging problem due to the complexity and variability of face forgery techniques. Existing Deepfake detection methods are often devoted to extracting features by designing sophisticated networks but ignore the influence of perceptual quality of faces. Considering the complexity of the quality distribution of both real and fake faces, we propose a novel Deepfake detection framework named DeepFidelity to adaptively distinguish real and fake faces with varying image quality by mining the perceptual forgery fidelity of face images. Specifically, we improve the model's ability to identify complex samples by mapping real and fake face data of different qualities to different scores to distinguish them in a more detailed way. In addition, we propose a network structure called Symmetric Spatial Attention Augmentation based vision Transformer (SSAAFormer), which uses the symmetry of face images to promote the network to model the geographic long-distance relationship at the shallow level and augment local features. Extensive experiments on multiple benchmark datasets demonstrate the superiority of the proposed method over state-of-the-art methods.

{{</citation>}}


### (55/136) VRPTEST: Evaluating Visual Referring Prompting in Large Multimodal Models (Zongjie Li et al., 2023)

{{<citation>}}

Zongjie Li, Chaozheng Wang, Chaowei Liu, Pingchuan Ma, Daoyuan Wu, Shuai Wang, Cuiyun Gao. (2023)  
**VRPTEST: Evaluating Visual Referring Prompting in Large Multimodal Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.04087v1)  

---


**ABSTRACT**  
With recent advancements in Large Multimodal Models (LMMs) across various domains, a novel prompting method called visual referring prompting has emerged, showing significant potential in enhancing human-computer interaction within multimodal systems. This method offers a more natural and flexible approach to human interaction with these systems compared to traditional text descriptions or coordinates. However, the categorization of visual referring prompting remains undefined, and its impact on the performance of LMMs has yet to be formally examined. In this study, we conduct the first comprehensive analysis of LMMs using a variety of visual referring prompting strategies. We introduce a benchmark dataset called VRPTEST, comprising 3 different visual tasks and 2,275 images, spanning diverse combinations of prompt strategies. Using VRPTEST, we conduct a comprehensive evaluation of eight versions of prominent open-source and proprietary foundation models, including two early versions of GPT-4V. We develop an automated assessment framework based on software metamorphic testing techniques to evaluate the accuracy of LMMs without the need for human intervention or manual labeling. We find that the current proprietary models generally outperform the open-source ones, showing an average accuracy improvement of 22.70%; however, there is still potential for improvement. Moreover, our quantitative analysis shows that the choice of prompt strategy significantly affects the accuracy of LMMs, with variations ranging from -17.5% to +7.3%. Further case studies indicate that an appropriate visual referring prompting strategy can improve LMMs' understanding of context and location information, while an unsuitable one might lead to answer rejection. We also provide insights on minimizing the negative impact of visual referring prompting on LMMs.

{{</citation>}}


### (56/136) Large Language Models are Good Prompt Learners for Low-Shot Image Classification (Zhaoheng Zheng et al., 2023)

{{<citation>}}

Zhaoheng Zheng, Jingmin Wei, Xuefeng Hu, Haidong Zhu, Ram Nevatia. (2023)  
**Large Language Models are Good Prompt Learners for Low-Shot Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04076v1)  

---


**ABSTRACT**  
Low-shot image classification, where training images are limited or inaccessible, has benefited from recent progress on pre-trained vision-language (VL) models with strong generalizability, e.g. CLIP. Prompt learning methods built with VL models generate text features from the class names that only have confined class-specific information. Large Language Models (LLMs), with their vast encyclopedic knowledge, emerge as the complement. Thus, in this paper, we discuss the integration of LLMs to enhance pre-trained VL models, specifically on low-shot classification. However, the domain gap between language and vision blocks the direct application of LLMs. Thus, we propose LLaMP, Large Language Models as Prompt learners, that produces adaptive prompts for the CLIP text encoder, establishing it as the connecting bridge. Experiments show that, compared with other state-of-the-art prompt learning methods, LLaMP yields better performance on both zero-shot generalization and few-shot image classification, over a spectrum of 11 datasets.

{{</citation>}}


### (57/136) An unsupervised approach towards promptable defect segmentation in laser-based additive manufacturing by Segment Anything (Israt Zarin Era et al., 2023)

{{<citation>}}

Israt Zarin Era, Imtiaz Ahmed, Zhichao Liu, Srinjoy Das. (2023)  
**An unsupervised approach towards promptable defect segmentation in laser-based additive manufacturing by Segment Anything**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.04063v1)  

---


**ABSTRACT**  
Foundation models are currently driving a paradigm shift in computer vision tasks for various fields including biology, astronomy, and robotics among others, leveraging user-generated prompts to enhance their performance. In the manufacturing domain, accurate image-based defect segmentation is imperative to ensure product quality and facilitate real-time process control. However, such tasks are often characterized by multiple challenges including the absence of labels and the requirement for low latency inference among others. To address these issues, we construct a framework for image segmentation using a state-of-the-art Vision Transformer (ViT) based Foundation model (Segment Anything Model) with a novel multi-point prompt generation scheme using unsupervised clustering. We apply our framework to perform real-time porosity segmentation in a case study of laser base powder bed fusion (L-PBF) and obtain high Dice Similarity Coefficients (DSC) without the necessity for any supervised fine-tuning in the model. Using such lightweight foundation model inference in conjunction with unsupervised prompt generation, we envision the construction of a real-time anomaly detection pipeline that has the potential to revolutionize the current laser-based additive manufacturing processes, thereby facilitating the shift towards Industry 4.0 and promoting defect-free production along with operational efficiency.

{{</citation>}}


### (58/136) Residual Graph Convolutional Network for Bird's-Eye-View Semantic Segmentation (Qiuxiao Chen et al., 2023)

{{<citation>}}

Qiuxiao Chen, Xiaojun Qi. (2023)  
**Residual Graph Convolutional Network for Bird's-Eye-View Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Convolutional Network, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.04044v1)  

---


**ABSTRACT**  
Retrieving spatial information and understanding the semantic information of the surroundings are important for Bird's-Eye-View (BEV) semantic segmentation. In the application of autonomous driving, autonomous vehicles need to be aware of their surroundings to drive safely. However, current BEV semantic segmentation techniques, deep Convolutional Neural Networks (CNNs) and transformers, have difficulties in obtaining the global semantic relationships of the surroundings at the early layers of the network. In this paper, we propose to incorporate a novel Residual Graph Convolutional (RGC) module in deep CNNs to acquire both the global information and the region-level semantic relationship in the multi-view image domain. Specifically, the RGC module employs a non-overlapping graph space projection to efficiently project the complete BEV information into graph space. It then builds interconnected spatial and channel graphs to extract spatial information between each node and channel information within each node (i.e., extract contextual relationships of the global features). Furthermore, it uses a downsample residual process to enhance the coordinate feature reuse to maintain the global information. The segmentation data augmentation and alignment module helps to simultaneously augment and align BEV features and ground truth to geometrically preserve their alignment to achieve better segmentation results. Our experimental results on the nuScenes benchmark dataset demonstrate that the RGC network outperforms four state-of-the-art networks and its four variants in terms of IoU and mIoU. The proposed RGC network achieves a higher mIoU of 3.1% than the best state-of-the-art network, BEVFusion. Code and models will be released.

{{</citation>}}


### (59/136) Doodle Your 3D: From Abstract Freehand Sketches to Precise 3D Shapes (Hmrishav Bandyopadhyay et al., 2023)

{{<citation>}}

Hmrishav Bandyopadhyay, Subhadeep Koley, Ayan Das, Aneeshan Sain, Pinaki Nath Chowdhury, Tao Xiang, Ayan Kumar Bhunia, Yi-Zhe Song. (2023)  
**Doodle Your 3D: From Abstract Freehand Sketches to Precise 3D Shapes**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2312.04043v1)  

---


**ABSTRACT**  
In this paper, we democratise 3D content creation, enabling precise generation of 3D shapes from abstract sketches while overcoming limitations tied to drawing skills. We introduce a novel part-level modelling and alignment framework that facilitates abstraction modelling and cross-modal correspondence. Leveraging the same part-level decoder, our approach seamlessly extends to sketch modelling by establishing correspondence between CLIPasso edgemaps and projected 3D part regions, eliminating the need for a dataset pairing human sketches and 3D shapes. Additionally, our method introduces a seamless in-position editing process as a byproduct of cross-modal part-aligned modelling. Operating in a low-dimensional implicit space, our approach significantly reduces computational demands and processing time.

{{</citation>}}


### (60/136) PartDistill: 3D Shape Part Segmentation by Vision-Language Model Distillation (Ardian Umam et al., 2023)

{{<citation>}}

Ardian Umam, Cheng-Kun Yang, Min-Hung Chen, Jen-Hui Chuang, Yen-Yu Lin. (2023)  
**PartDistill: 3D Shape Part Segmentation by Vision-Language Model Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Model Distillation  
[Paper Link](http://arxiv.org/abs/2312.04016v1)  

---


**ABSTRACT**  
This paper proposes a cross-modal distillation framework, PartDistill, which transfers 2D knowledge from vision-language models (VLMs) to facilitate 3D shape part segmentation. PartDistill addresses three major challenges in this task: the lack of 3D segmentation in invisible or undetected regions in the 2D projections, inaccurate and inconsistent 2D predictions by VLMs, and the lack of knowledge accumulation across different 3D shapes. PartDistill consists of a teacher network that uses a VLM to make 2D predictions and a student network that learns from the 2D predictions while extracting geometrical features from multiple 3D shapes to carry out 3D part segmentation. A bi-directional distillation, including forward and backward distillations, is carried out within the framework, where the former forward distills the 2D predictions to the student network, and the latter improves the quality of the 2D predictions, which subsequently enhances the final 3D part segmentation. Moreover, PartDistill can exploit generative models that facilitate effortless 3D shape creation for generating knowledge sources to be distilled. Through extensive experiments, PartDistill boosts the existing methods with substantial margins on widely used ShapeNetPart and PartE datasets, by more than 15% and 12% higher mIoU scores, respectively.

{{</citation>}}


### (61/136) KOALA: Self-Attention Matters in Knowledge Distillation of Latent Diffusion Models for Memory-Efficient and Fast Image Synthesis (Youngwan Lee et al., 2023)

{{<citation>}}

Youngwan Lee, Kwanyong Park, Yoorhim Cho, Yong-Ju Lee, Sung Ju Hwang. (2023)  
**KOALA: Self-Attention Matters in Knowledge Distillation of Latent Diffusion Models for Memory-Efficient and Fast Image Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Knowledge Distillation, Self-Attention  
[Paper Link](http://arxiv.org/abs/2312.04005v1)  

---


**ABSTRACT**  
Stable diffusion is the mainstay of the text-to-image (T2I) synthesis in the community due to its generation performance and open-source nature. Recently, Stable Diffusion XL (SDXL), the successor of stable diffusion, has received a lot of attention due to its significant performance improvements with a higher resolution of 1024x1024 and a larger model. However, its increased computation cost and model size require higher-end hardware(e.g., bigger VRAM GPU) for end-users, incurring higher costs of operation. To address this problem, in this work, we propose an efficient latent diffusion model for text-to-image synthesis obtained by distilling the knowledge of SDXL. To this end, we first perform an in-depth analysis of the denoising U-Net in SDXL, which is the main bottleneck of the model, and then design a more efficient U-Net based on the analysis. Secondly, we explore how to effectively distill the generation capability of SDXL into an efficient U-Net and eventually identify four essential factors, the core of which is that self-attention is the most important part. With our efficient U-Net and self-attention-based knowledge distillation strategy, we build our efficient T2I models, called KOALA-1B & -700M, while reducing the model size up to 54% and 69% of the original SDXL model. In particular, the KOALA-700M is more than twice as fast as SDXL while still retaining a decent generation quality. We hope that due to its balanced speed-performance tradeoff, our KOALA models can serve as a cost-effective alternative to SDXL in resource-constrained environments.

{{</citation>}}


### (62/136) Stable diffusion for Data Augmentation in COCO and Weed Datasets (Boyang Deng, 2023)

{{<citation>}}

Boyang Deng. (2023)  
**Stable diffusion for Data Augmentation in COCO and Weed Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.03996v2)  

---


**ABSTRACT**  
Generative models have increasingly impacted relative tasks, from computer vision to interior design and other fields. Stable diffusion is an outstanding diffusion model that paves the way for producing high-resolution images with thorough details from text prompts or reference images. It will be an interesting topic about gaining improvements for small datasets with image-sparse categories. This study utilized seven common categories and three widespread weed species to evaluate the efficiency of a stable diffusion model. In detail, Stable diffusion was used to generate synthetic images belonging to these classes; three techniques (i.e., Image-to-image translation, Dreambooth, and ControlNet) based on stable diffusion were leveraged for image generation with different focuses. Then, classification and detection tasks were conducted based on these synthetic images, whose performance was compared to the models trained on original images. Promising results have been achieved in some classes. This seminal study may expedite the adaption of stable diffusion models to different fields.

{{</citation>}}


### (63/136) Style Transfer to Calvin and Hobbes comics using Stable Diffusion (Sloke Shrestha et al., 2023)

{{<citation>}}

Sloke Shrestha, Sundar Sripada V. S., Asvin Venkataramanan. (2023)  
**Style Transfer to Calvin and Hobbes comics using Stable Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2312.03993v1)  

---


**ABSTRACT**  
This project report summarizes our journey to perform stable diffusion fine-tuning on a dataset containing Calvin and Hobbes comics. The purpose is to convert any given input image into the comic style of Calvin and Hobbes, essentially performing style transfer. We train stable-diffusion-v1.5 using Low Rank Adaptation (LoRA) to efficiently speed up the fine-tuning process. The diffusion itself is handled by a Variational Autoencoder (VAE), which is a U-net. Our results were visually appealing for the amount of training time and the quality of input data that went into training.

{{</citation>}}


## cs.CR (8)



### (64/136) Forcing Generative Models to Degenerate Ones: The Power of Data Poisoning Attacks (Shuli Jiang et al., 2023)

{{<citation>}}

Shuli Jiang, Swanand Ravindra Kadhe, Yi Zhou, Ling Cai, Nathalie Baracaldo. (2023)  
**Forcing Generative Models to Degenerate Ones: The Power of Data Poisoning Attacks**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04748v1)  

---


**ABSTRACT**  
Growing applications of large language models (LLMs) trained by a third party raise serious concerns on the security vulnerability of LLMs.It has been demonstrated that malicious actors can covertly exploit these vulnerabilities in LLMs through poisoning attacks aimed at generating undesirable outputs. While poisoning attacks have received significant attention in the image domain (e.g., object detection), and classification tasks, their implications for generative models, particularly in the realm of natural language generation (NLG) tasks, remain poorly understood. To bridge this gap, we perform a comprehensive exploration of various poisoning techniques to assess their effectiveness across a range of generative tasks. Furthermore, we introduce a range of metrics designed to quantify the success and stealthiness of poisoning attacks specifically tailored to NLG tasks. Through extensive experiments on multiple NLG tasks, LLMs and datasets, we show that it is possible to successfully poison an LLM during the fine-tuning stage using as little as 1\% of the total tuning data samples. Our paper presents the first systematic approach to comprehend poisoning attacks targeting NLG tasks considering a wide range of triggers and attack settings. We hope our findings will assist the AI security community in devising appropriate defenses against such threats.

{{</citation>}}


### (65/136) DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions (Fangzhou Wu et al., 2023)

{{<citation>}}

Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao. (2023)  
**DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.04730v1)  

---


**ABSTRACT**  
With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.

{{</citation>}}


### (66/136) Purple Llama CyberSecEval: A Secure Coding Benchmark for Language Models (Manish Bhatt et al., 2023)

{{<citation>}}

Manish Bhatt, Sahana Chennabasappa, Cyrus Nikolaidis, Shengye Wan, Ivan Evtimov, Dominik Gabi, Daniel Song, Faizan Ahmad, Cornelius Aschermann, Lorenzo Fontana, Sasha Frolov, Ravi Prakash Giri, Dhaval Kapil, Yiannis Kozyrakis, David LeBlanc, James Milazzo, Aleksandar Straumann, Gabriel Synnaeve, Varun Vontimitta, Spencer Whitman, Joshua Saxe. (2023)  
**Purple Llama CyberSecEval: A Secure Coding Benchmark for Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04724v1)  

---


**ABSTRACT**  
This paper presents CyberSecEval, a comprehensive benchmark developed to help bolster the cybersecurity of Large Language Models (LLMs) employed as coding assistants. As what we believe to be the most extensive unified cybersecurity safety benchmark to date, CyberSecEval provides a thorough evaluation of LLMs in two crucial security domains: their propensity to generate insecure code and their level of compliance when asked to assist in cyberattacks. Through a case study involving seven models from the Llama 2, Code Llama, and OpenAI GPT large language model families, CyberSecEval effectively pinpointed key cybersecurity risks. More importantly, it offered practical insights for refining these models. A significant observation from the study was the tendency of more advanced models to suggest insecure code, highlighting the critical need for integrating security considerations in the development of sophisticated LLMs. CyberSecEval, with its automated test case generation and evaluation pipeline covers a broad scope and equips LLM designers and researchers with a tool to broadly measure and enhance the cybersecurity safety properties of LLMs, contributing to the development of more secure AI systems.

{{</citation>}}


### (67/136) NeuJeans: Private Neural Network Inference with Joint Optimization of Convolution and Bootstrapping (Jae Hyung Ju et al., 2023)

{{<citation>}}

Jae Hyung Ju, Jaiyoung Park, Jongmin Kim, Donghwan Kim, Jung Ho Ahn. (2023)  
**NeuJeans: Private Neural Network Inference with Joint Optimization of Convolution and Bootstrapping**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.04356v1)  

---


**ABSTRACT**  
Fully homomorphic encryption (FHE) is a promising cryptographic primitive for realizing private neural network inference (PI) services by allowing a client to fully offload the inference task to a cloud server while keeping the client data oblivious to the server. This work proposes NeuJeans, an FHE-based solution for the PI of deep convolutional neural networks (CNNs). NeuJeans tackles the critical problem of the enormous computational cost for the FHE evaluation of convolutional layers (conv2d), mainly due to the high cost of data reordering and bootstrapping. We first propose an encoding method introducing nested structures inside encoded vectors for FHE, which enables us to develop efficient conv2d algorithms with reduced data reordering costs. However, the new encoding method also introduces additional computations for conversion between encoding methods, which could negate its advantages. We discover that fusing conv2d with bootstrapping eliminates such computations while reducing the cost of bootstrapping. Then, we devise optimized execution flows for various types of conv2d and apply them to end-to-end implementation of CNNs. NeuJeans accelerates the performance of conv2d by up to 5.68 times compared to state-of-the-art FHE-based PI work and performs the PI of a CNN at the scale of ImageNet (ResNet18) within a mere few seconds

{{</citation>}}


### (68/136) Dynamic Data-Driven Digital Twins for Blockchain Systems (Georgios Diamantopoulos et al., 2023)

{{<citation>}}

Georgios Diamantopoulos, Nikos Tziritas, Rami Bahsoon, Georgios Theodoropoulos. (2023)  
**Dynamic Data-Driven Digital Twins for Blockchain Systems**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-DC, cs-PF, cs.CR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.04226v1)  

---


**ABSTRACT**  
In recent years, we have seen an increase in the adoption of blockchain-based systems in non-financial applications, looking to benefit from what the technology has to offer. Although many fields have managed to include blockchain in their core functionalities, the adoption of blockchain, in general, is constrained by the so-called trilemma trade-off between decentralization, scalability, and security. In our previous work, we have shown that using a digital twin for dynamically managing blockchain systems during runtime can be effective in managing the trilemma trade-off. Our Digital Twin leverages DDDAS feedback loop, which is responsible for getting the data from the system to the digital twin, conducting optimisation, and updating the physical system. This paper examines how leveraging DDDAS feedback loop can support the optimisation component of the trilemma benefiting from Reinforcement Learning agents and a simulation component to augment the quality of the learned model while reducing the computational overhead required for decision-making.

{{</citation>}}


### (69/136) A Novel Federated Learning-based Intrusion Detection System for Flying Ad Hoc Networks (Ozlem Ceviz et al., 2023)

{{<citation>}}

Ozlem Ceviz, Pinar Sadioglu, Sevil Sen, Vassilios G. Vassilakis. (2023)  
**A Novel Federated Learning-based Intrusion Detection System for Flying Ad Hoc Networks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Bias, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2312.04135v1)  

---


**ABSTRACT**  
Unmanned aerial vehicles (UAVs) in flying ad-hoc networks (FANETs) face security challenges due to the dynamic and distributed nature of these networks. This paper presents the Federated Learning-based Intrusion Detection System (FL-IDS), an innovative approach designed to improve FANET security. FL-IDS leverages federated learning to address privacy concerns of centralized intrusion detection systems. FL-IDS operates in a decentralized manner, enabling UAVs to collaboratively train a global intrusion detection model without sharing raw data. Local models are assigned to each UAV, using client-specific data, and only updated model weights are shared with a central server. This preserves privacy while utilizing collective intelligence for effective intrusion detection. Experimental results show FL-IDS's competitive performance with Central IDS (C-IDS) while mitigating privacy concerns. The Bias Towards Specific Clients (BTSC) method further enhances FL-IDS performance, surpassing C-IDS even at lower attacker ratios. A comparative analysis with traditional intrusion detection methods, including Local IDS (L-IDS), provides insights into FL-IDS's strengths. This study significantly contributes to FANET security by introducing a privacy-aware, decentralized intrusion detection approach tailored to the unique challenges of UAV networks.

{{</citation>}}


### (70/136) Making Translators Privacy-aware on the User's Side (Ryoma Sato, 2023)

{{<citation>}}

Ryoma Sato. (2023)  
**Making Translators Privacy-aware on the User's Side**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CR  
Keywords: ChatGPT, GPT, GPT-3.5, T5  
[Paper Link](http://arxiv.org/abs/2312.04068v1)  

---


**ABSTRACT**  
We propose PRISM to enable users of machine translation systems to preserve the privacy of data on their own initiative. There is a growing demand to apply machine translation systems to data that require privacy protection. While several machine translation engines claim to prioritize privacy, the extent and specifics of such protection are largely ambiguous. First, there is often a lack of clarity on how and to what degree the data is protected. Even if service providers believe they have sufficient safeguards in place, sophisticated adversaries might still extract sensitive information. Second, vulnerabilities may exist outside of these protective measures, such as within communication channels, potentially leading to data leakage. As a result, users are hesitant to utilize machine translation engines for data demanding high levels of privacy protection, thereby missing out on their benefits. PRISM resolves this problem. Instead of relying on the translation service to keep data safe, PRISM provides the means to protect data on the user's side. This approach ensures that even machine translation engines with inadequate privacy measures can be used securely. For platforms already equipped with privacy safeguards, PRISM acts as an additional protection layer, reinforcing their security furthermore. PRISM adds these privacy features without significantly compromising translation accuracy. Our experiments demonstrate the effectiveness of PRISM using real-world translators, T5 and ChatGPT (GPT-3.5-turbo), and the datasets with two languages. PRISM effectively balances privacy protection with translation accuracy.

{{</citation>}}


### (71/136) Defense against ML-based Power Side-channel Attacks on DNN Accelerators with Adversarial Attacks (Xiaobei Yan et al., 2023)

{{<citation>}}

Xiaobei Yan, Chip Hong Chang, Tianwei Zhang. (2023)  
**Defense against ML-based Power Side-channel Attacks on DNN Accelerators with Adversarial Attacks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.04035v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) hardware accelerators have been widely adopted to enhance the efficiency of deep learning applications. However, they also raise security concerns regarding their vulnerability to power side-channel attacks (SCA). In these attacks, the adversary exploits unintended communication channels to infer sensitive information processed by the accelerator, posing significant privacy and copyright risks to the models. Advanced machine learning algorithms are further employed to facilitate the side-channel analysis and exacerbate the privacy issue of AI accelerators. Traditional defense strategies naively inject execution noise to the runtime of AI models, which inevitably introduce large overheads.   In this paper, we present AIAShield, a novel defense methodology to safeguard FPGA-based AI accelerators and mitigate model extraction threats via power-based SCAs. The key insight of AIAShield is to leverage the prominent adversarial attack technique from the machine learning community to craft delicate noise, which can significantly obfuscate the adversary's side-channel observation while incurring minimal overhead to the execution of the protected model. At the hardware level, we design a new module based on ring oscillators to achieve fine-grained noise generation. At the algorithm level, we repurpose Neural Architecture Search to worsen the adversary's extraction results. Extensive experiments on the Nvidia Deep Learning Accelerator (NVDLA) demonstrate that AIAShield outperforms existing solutions with excellent transferability.

{{</citation>}}


## cs.NI (1)



### (72/136) MetaDetect: Metamorphic Testing Based Anomaly Detection for Multi-UAV Wireless Networks (Boyang Yan, 2023)

{{<citation>}}

Boyang Yan. (2023)  
**MetaDetect: Metamorphic Testing Based Anomaly Detection for Multi-UAV Wireless Networks**  

---
Primary Category: cs.NI  
Categories: 68-06, cs-NI, cs-SE, cs.NI, stat-ME  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.04747v1)  

---


**ABSTRACT**  
The reliability of wireless Ad Hoc Networks (WANET) communication is much lower than wired networks. WANET will be impacted by node overload, routing protocol, weather, obstacle blockage, and many other factors, all those anomalies cannot be avoided. Accurate prediction of the network entirely stopping in advance is essential after people could do networking re-routing or changing to different bands. In the present study, there are two primary goals. Firstly, design anomaly events detection patterns based on Metamorphic Testing (MT) methodology. Secondly, compare the performance of evaluation metrics, such as Transfer Rate, Occupancy rate, and the Number of packets received. Compared to other studies, the most significant advantage of mathematical interpretability, as well as not requiring dependence on physical environmental information, only relies on the networking physical layer and Mac layer data. The analysis of the results demonstrates that the proposed MT detection method is helpful for automatically identifying incidents/accident events on WANET. The physical layer transfer Rate metric could get the best performance.

{{</citation>}}


## cs.IT (2)



### (73/136) Reinforcement Learning Based Dynamic Power Control for UAV Mobility Management (Irshad A. Meer et al., 2023)

{{<citation>}}

Irshad A. Meer, Karl-Ludwig Besser, Mustafa Ozger, H. Vincent Poor, Cicek Cavdar. (2023)  
**Reinforcement Learning Based Dynamic Power Control for UAV Mobility Management**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.04742v1)  

---


**ABSTRACT**  
Modern communication systems need to fulfill multiple and often conflicting objectives at the same time. In particular, new applications require high reliability while operating at low transmit powers. Moreover, reliability constraints may vary over time depending on the current state of the system. One solution to address this problem is to use joint transmissions from a number of base stations (BSs) to meet the reliability requirements. However, this approach is inefficient when considering the overall total transmit power. In this work, we propose a reinforcement learning-based power allocation scheme for an unmanned aerial vehicle (UAV) communication system with varying communication reliability requirements. In particular, the proposed scheme aims to minimize the total transmit power of all BSs while achieving an outage probability that is less than a tolerated threshold. This threshold varies over time, e.g., when the UAV enters a critical zone with high-reliability requirements. Our results show that the proposed learning scheme uses dynamic power allocation to meet varying reliability requirements, thus effectively conserving energy.

{{</citation>}}


### (74/136) A Low-Overhead Incorporation-Extrapolation based Few-Shot CSI Feedback Framework for Massive MIMO Systems (Binggui Zhou et al., 2023)

{{<citation>}}

Binggui Zhou, Xi Yang, Jintao Wang, Shaodan Ma, Feifei Gao, Guanghua Yang. (2023)  
**A Low-Overhead Incorporation-Extrapolation based Few-Shot CSI Feedback Framework for Massive MIMO Systems**  

---
Primary Category: cs.IT  
Categories: cs-AI, cs-IT, cs.IT, eess-SP, math-IT  
Keywords: AI, Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.04062v1)  

---


**ABSTRACT**  
Accurate channel state information (CSI) is essential for downlink precoding at the base station (BS), especially for frequency FDD wideband massive MIMO systems with OFDM. In FDD systems, CSI is attained through CSI feedback from the user equipment (UE). However, large-scale antennas and large number of subcarriers significantly increase CSI feedback overhead. Deep learning-based CSI feedback methods have received tremendous attention in recent years due to their great capability of compressing CSI. Nonetheless, large amounts of collected samples are required to train deep learning models, which is severely challenging in practice. Besides, with the rapidly increasing number of antennas and subcarriers, most of these deep learning methods' CSI feedback overhead also grow dramatically, owing to their focus on full-dimensional CSI feedback. To address this issue, in this paper, we propose a low-overhead Incorporation-Extrapolation based Few-Shot CSI feedback Framework (IEFSF) for massive MIMO systems. To further reduce the feedback overhead, a low-dimensional eigenvector-based CSI matrix is first formed with the incorporation process at the UE, and then recovered to the full-dimensional eigenvector-based CSI matrix at the BS via the extrapolation process. After that, to alleviate the necessity of the extensive collected samples and enable few-shot CSI feedback, we further propose a knowledge-driven data augmentation method and an artificial intelligence-generated content (AIGC) -based data augmentation method by exploiting the domain knowledge of wireless channels and by exploiting a novel generative model, respectively. Numerical results demonstrate that the proposed IEFSF can significantly reduce CSI feedback overhead by 16 times compared with existing CSI feedback methods while maintaining higher feedback accuracy using only several hundreds of collected samples.

{{</citation>}}


## cs.CL (30)



### (75/136) Is Feedback All You Need? Leveraging Natural Language Feedback in Goal-Conditioned Reinforcement Learning (Sabrina McCallum et al., 2023)

{{<citation>}}

Sabrina McCallum, Max Taylor-Davies, Stefano V. Albrecht, Alessandro Suglia. (2023)  
**Is Feedback All You Need? Leveraging Natural Language Feedback in Goal-Conditioned Reinforcement Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2312.04736v1)  

---


**ABSTRACT**  
Despite numerous successes, the field of reinforcement learning (RL) remains far from matching the impressive generalisation power of human behaviour learning. One possible way to help bridge this gap be to provide RL agents with richer, more human-like feedback expressed in natural language. To investigate this idea, we first extend BabyAI to automatically generate language feedback from the environment dynamics and goal condition success. Then, we modify the Decision Transformer architecture to take advantage of this additional signal. We find that training with language feedback either in place of or in addition to the return-to-go or goal descriptions improves agents' generalisation performance, and that agents can benefit from feedback even when this is only available during training, but not at inference.

{{</citation>}}


### (76/136) From Big to Small Without Losing It All: Text Augmentation with ChatGPT for Efficient Sentiment Analysis (Stanisław Woźniak et al., 2023)

{{<citation>}}

Stanisław Woźniak, Jan Kocoń. (2023)  
**From Big to Small Without Losing It All: Text Augmentation with ChatGPT for Efficient Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Augmentation, ChatGPT, GPT, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2312.04720v1)  

---


**ABSTRACT**  
In the era of artificial intelligence, data is gold but costly to annotate. The paper demonstrates a groundbreaking solution to this dilemma using ChatGPT for text augmentation in sentiment analysis. We leverage ChatGPT's generative capabilities to create synthetic training data that significantly improves the performance of smaller models, making them competitive with, or even outperforming, their larger counterparts. This innovation enables models to be both efficient and effective, thereby reducing computational cost, inference time, and memory usage without compromising on quality. Our work marks a key advancement in the cost-effective development and deployment of robust sentiment analysis models.

{{</citation>}}


### (77/136) Deep Emotions Across Languages: A Novel Approach for Sentiment Propagation in Multilingual WordNets (Jan Kocoń, 2023)

{{<citation>}}

Jan Kocoń. (2023)  
**Deep Emotions Across Languages: A Novel Approach for Sentiment Propagation in Multilingual WordNets**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Embedding, Multilingual  
[Paper Link](http://arxiv.org/abs/2312.04715v1)  

---


**ABSTRACT**  
Sentiment analysis involves using WordNets enriched with emotional metadata, which are valuable resources. However, manual annotation is time-consuming and expensive, resulting in only a few WordNet Lexical Units being annotated. This paper introduces two new techniques for automatically propagating sentiment annotations from a partially annotated WordNet to its entirety and to a WordNet in a different language: Multilingual Structured Synset Embeddings (MSSE) and Cross-Lingual Deep Neural Sentiment Propagation (CLDNS). We evaluated the proposed MSSE+CLDNS method extensively using Princeton WordNet and Polish WordNet, which have many inter-lingual relations. Our results show that the MSSE+CLDNS method outperforms existing propagation methods, indicating its effectiveness in enriching WordNets with emotional metadata across multiple languages. This work provides a solid foundation for large-scale, multilingual sentiment analysis and is valuable for academic research and practical applications.

{{</citation>}}


### (78/136) Simul-LLM: A Framework for Exploring High-Quality Simultaneous Translation with Large Language Models (Victor Agostinelli et al., 2023)

{{<citation>}}

Victor Agostinelli, Max Wild, Matthew Raffel, Kazi Ahmed Asif Fuad, Lizhong Chen. (2023)  
**Simul-LLM: A Framework for Exploring High-Quality Simultaneous Translation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.04691v2)  

---


**ABSTRACT**  
Large language models (LLMs) with billions of parameters and pretrained on massive amounts of data are now capable of near or better than state-of-the-art performance in a variety of downstream natural language processing tasks. Neural machine translation (NMT) is one such task that LLMs have been applied to with great success. However, little research has focused on applying LLMs to the more difficult subset of NMT called simultaneous translation (SimulMT), where translation begins before the entire source context is available to the model. In this paper, we address key challenges facing LLMs fine-tuned for SimulMT, validate classical SimulMT concepts and practices in the context of LLMs, explore adapting LLMs that are fine-tuned for NMT to the task of SimulMT, and introduce Simul-LLM, the first open-source fine-tuning and evaluation pipeline development framework for LLMs focused on SimulMT.

{{</citation>}}


### (79/136) Latent Skill Discovery for Chain-of-Thought Reasoning (Zifan Xu et al., 2023)

{{<citation>}}

Zifan Xu, Haozhu Wang, Dmitriy Bespalov, Peter Stone, Yanjun Qi. (2023)  
**Latent Skill Discovery for Chain-of-Thought Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.04684v1)  

---


**ABSTRACT**  
Recent advances in Large Language Models (LLMs) have led to an emergent ability of chain-of-thought (CoT) prompting, a prompt reasoning strategy that adds intermediate rationale steps between questions and answers to construct prompts. Conditioned on these prompts, LLMs can effectively learn in context to generate rationales that lead to more accurate answers than when answering the same question directly. To design LLM prompts, one important setting, called demonstration selection, considers selecting demonstrations from an example bank. Existing methods use various heuristics for this selection, but for CoT prompting, which involves unique rationales, it is essential to base the selection upon the intrinsic skills that CoT rationales need, for instance, the skills of addition or subtraction for math word problems.   To address this requirement, we introduce a novel approach named Reasoning Skill Discovery (RSD) that use unsupervised learning to create a latent space representation of rationales, called a reasoning skill. Simultaneously, RSD learns a reasoning policy to determine the required reasoning skill for a given question. This can then guide the selection of examples that demonstrate the required reasoning skills. Our approach offers several desirable properties: it is (1) theoretically grounded, (2) sample-efficient, requiring no LLM inference or manual prompt design, and (3) LLM-agnostic. Empirically, RSD outperforms existing methods by up to 6% in terms of the answer accuracy across multiple reasoning tasks.

{{</citation>}}


### (80/136) TOD-Flow: Modeling the Structure of Task-Oriented Dialogues (Sungryull Sohn et al., 2023)

{{<citation>}}

Sungryull Sohn, Yiwei Lyu, Anthony Liu, Lajanugen Logeswaran, Dong-Ki Kim, Dongsub Shim, Honglak Lee. (2023)  
**TOD-Flow: Modeling the Structure of Task-Oriented Dialogues**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.04668v1)  

---


**ABSTRACT**  
Task-Oriented Dialogue (TOD) systems have become crucial components in interactive artificial intelligence applications. While recent advances have capitalized on pre-trained language models (PLMs), they exhibit limitations regarding transparency and controllability. To address these challenges, we propose a novel approach focusing on inferring the TOD-Flow graph from dialogue data annotated with dialog acts, uncovering the underlying task structure in the form of a graph. The inferred TOD-Flow graph can be easily integrated with any dialogue model to improve its prediction performance, transparency, and controllability. Our TOD-Flow graph learns what a model can, should, and should not predict, effectively reducing the search space and providing a rationale for the model's prediction. We show that the proposed TOD-Flow graph better resembles human-annotated graphs compared to prior approaches. Furthermore, when combined with several dialogue policies and end-to-end dialogue models, we demonstrate that our approach significantly improves dialog act classification and end-to-end response generation performance in the MultiWOZ and SGD benchmarks. Code available at: https://github.com/srsohn/TOD-Flow

{{</citation>}}


### (81/136) Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations (Hakan Inan et al., 2023)

{{<citation>}}

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, Madian Khabsa. (2023)  
**Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.06674v1)  

---


**ABSTRACT**  
We introduce Llama Guard, an LLM-based input-output safeguard model geared towards Human-AI conversation use cases. Our model incorporates a safety risk taxonomy, a valuable tool for categorizing a specific set of safety risks found in LLM prompts (i.e., prompt classification). This taxonomy is also instrumental in classifying the responses generated by LLMs to these prompts, a process we refer to as response classification. For the purpose of both prompt and response classification, we have meticulously gathered a dataset of high quality. Llama Guard, a Llama2-7b model that is instruction-tuned on our collected dataset, albeit low in volume, demonstrates strong performance on existing benchmarks such as the OpenAI Moderation Evaluation dataset and ToxicChat, where its performance matches or exceeds that of currently available content moderation tools. Llama Guard functions as a language model, carrying out multi-class classification and generating binary decision scores. Furthermore, the instruction fine-tuning of Llama Guard allows for the customization of tasks and the adaptation of output formats. This feature enhances the model's capabilities, such as enabling the adjustment of taxonomy categories to align with specific use cases, and facilitating zero-shot or few-shot prompting with diverse taxonomies at the input. We are making Llama Guard model weights available and we encourage researchers to further develop and adapt them to meet the evolving needs of the community for AI safety.

{{</citation>}}


### (82/136) Self-Supervised Behavior Cloned Transformers are Path Crawlers for Text Games (Ruoyao Wang et al., 2023)

{{<citation>}}

Ruoyao Wang, Peter Jansen. (2023)  
**Self-Supervised Behavior Cloned Transformers are Path Crawlers for Text Games**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Self-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.04657v1)  

---


**ABSTRACT**  
In this work, we introduce a self-supervised behavior cloning transformer for text games, which are challenging benchmarks for multi-step reasoning in virtual environments. Traditionally, Behavior Cloning Transformers excel in such tasks but rely on supervised training data. Our approach auto-generates training data by exploring trajectories (defined by common macro-action sequences) that lead to reward within the games, while determining the generality and utility of these trajectories by rapidly training small models then evaluating their performance on unseen development games. Through empirical analysis, we show our method consistently uncovers generalizable training data, achieving about 90\% performance of supervised systems across three benchmark text games.

{{</citation>}}


### (83/136) PyThaiNLP: Thai Natural Language Processing in Python (Wannaphong Phatthiyaphaibun et al., 2023)

{{<citation>}}

Wannaphong Phatthiyaphaibun, Korakot Chaovavanich, Charin Polpanumas, Arthit Suriyawongkul, Lalita Lowphansirikul, Pattarawat Chormai, Peerat Limkonchotiwat, Thanathip Suntorntip, Can Udomcharoenchaikit. (2023)  
**PyThaiNLP: Thai Natural Language Processing in Python**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.04649v1)  

---


**ABSTRACT**  
We present PyThaiNLP, a free and open-source natural language processing (NLP) library for Thai language implemented in Python. It provides a wide range of software, models, and datasets for Thai language. We first provide a brief historical context of tools for Thai language prior to the development of PyThaiNLP. We then outline the functionalities it provided as well as datasets and pre-trained language models. We later summarize its development milestones and discuss our experience during its development. We conclude by demonstrating how industrial and research communities utilize PyThaiNLP in their work. The library is freely available at https://github.com/pythainlp/pythainlp.

{{</citation>}}


### (84/136) On Sarcasm Detection with OpenAI GPT-based Models (Montgomery Gole et al., 2023)

{{<citation>}}

Montgomery Gole, Williams-Paul Nwadiugwu, Andriy Miranskyy. (2023)  
**On Sarcasm Detection with OpenAI GPT-based Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, GPT, GPT-3.5, GPT-4, Sarcasm Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2312.04642v1)  

---


**ABSTRACT**  
Sarcasm is a form of irony that requires readers or listeners to interpret its intended meaning by considering context and social cues. Machine learning classification models have long had difficulty detecting sarcasm due to its social complexity and contradictory nature.   This paper explores the applications of the Generative Pretrained Transformer (GPT) models, including GPT-3, InstructGPT, GPT-3.5, and GPT-4, in detecting sarcasm in natural language. It tests fine-tuned and zero-shot models of different sizes and releases.   The GPT models were tested on the political and balanced (pol-bal) portion of the popular Self-Annotated Reddit Corpus (SARC 2.0) sarcasm dataset. In the fine-tuning case, the largest fine-tuned GPT-3 model achieves accuracy and $F_1$-score of 0.81, outperforming prior models. In the zero-shot case, one of GPT-4 models yields an accuracy of 0.70 and $F_1$-score of 0.75. Other models score lower. Additionally, a model's performance may improve or deteriorate with each release, highlighting the need to reassess performance after each release.

{{</citation>}}


### (85/136) Large Language Models for Mathematicians (Simon Frieder et al., 2023)

{{<citation>}}

Simon Frieder, Julius Berner, Philipp Petersen, Thomas Lukasiewicz. (2023)  
**Large Language Models for Mathematicians**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, math-HO  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04556v1)  

---


**ABSTRACT**  
Large language models (LLMs) such as ChatGPT have received immense interest for their general-purpose language understanding and, in particular, their ability to generate high-quality text or computer code. For many professions, LLMs represent an invaluable tool that can speed up and improve the quality of work. In this note, we discuss to what extent they can aid professional mathematicians. We first provide a mathematical description of the transformer model used in all modern language models. Based on recent studies, we then outline best practices and potential issues and report on the mathematical abilities of language models. Finally, we shed light on the potential of LMMs to change how mathematicians work.

{{</citation>}}


### (86/136) Efficient Monotonic Multihead Attention (Xutai Ma et al., 2023)

{{<citation>}}

Xutai Ma, Anna Sun, Siqi Ouyang, Hirofumi Inaguma, Paden Tomasello. (2023)  
**Efficient Monotonic Multihead Attention**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.04515v1)  

---


**ABSTRACT**  
We introduce the Efficient Monotonic Multihead Attention (EMMA), a state-of-the-art simultaneous translation model with numerically-stable and unbiased monotonic alignment estimation. In addition, we present improved training and inference strategies, including simultaneous fine-tuning from an offline translation model and reduction of monotonic alignment variance. The experimental results demonstrate that the proposed model attains state-of-the-art performance in simultaneous speech-to-text translation on the Spanish and English translation task.

{{</citation>}}


### (87/136) An LLM Compiler for Parallel Function Calling (Sehoon Kim et al., 2023)

{{<citation>}}

Sehoon Kim, Suhong Moon, Ryan Tabrizi, Nicholas Lee, Michael W. Mahoney, Kurt Keutzer, Amir Gholami. (2023)  
**An LLM Compiler for Parallel Function Calling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04511v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown remarkable results on various complex reasoning benchmarks. The reasoning capabilities of LLMs enable them to execute function calls, using user-provided functions to overcome their inherent limitations, such as knowledge cutoffs, poor arithmetic skills, or lack of access to private data. This development has expanded LLMs' scope to include multi-function calling, where LLMs are equipped with a variety of functions and select the proper functions based on the context. Multi-function calling abilities of LLMs have catalyzed LLM-based software development, allowing them to tackle more complex problems. However, current methods for multi-function calling often require sequential reasoning and acting for each function which can result in high latency, cost, and sometimes inaccurate behavior. To address this, we introduce LLMCompiler, which executes functions in parallel to efficiently orchestrate multi-function calling. Drawing from the principles of classical compilers, LLMCompiler streamlines parallel function calling with three components: (i) an LLM Planner, formulating execution strategies and dependencies; (ii) a Task Fetching Unit, dispatching function calling tasks; and (iii) an Executor, executing these tasks in parallel. LLMCompiler automatically computes an optimized orchestration for the function calls and can be used with open-source models such as LLaMA-2. We have benchmarked LLMCompiler on a range of tasks including cases with non-trivial inter-dependency between function calls, as well as cases that require dynamic replanning based on intermediate results. We observe consistent latency speedup of up to 3.7x, cost savings of up to 6.7x, and accuracy improvement of up to ~9% as compared to ReAct. Additionally, LLMCompiler achieves up to 1.35x latency gain over OpenAI's recent parallel function calling, while achieving similar accuracy.

{{</citation>}}


### (88/136) A Block Metropolis-Hastings Sampler for Controllable Energy-based Text Generation (Jarad Forristal et al., 2023)

{{<citation>}}

Jarad Forristal, Niloofar Mireshghallah, Greg Durrett, Taylor Berg-Kirkpatrick. (2023)  
**A Block Metropolis-Hastings Sampler for Controllable Energy-based Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2312.04510v1)  

---


**ABSTRACT**  
Recent work has shown that energy-based language modeling is an effective framework for controllable text generation because it enables flexible integration of arbitrary discriminators. However, because energy-based LMs are globally normalized, approximate techniques like Metropolis-Hastings (MH) are required for inference. Past work has largely explored simple proposal distributions that modify a single token at a time, like in Gibbs sampling. In this paper, we develop a novel MH sampler that, in contrast, proposes re-writes of the entire sequence in each step via iterative prompting of a large language model. Our new sampler (a) allows for more efficient and accurate sampling from a target distribution and (b) allows generation length to be determined through the sampling procedure rather than fixed in advance, as past work has required. We perform experiments on two controlled generation tasks, showing both downstream performance gains and more accurate target distribution sampling in comparison with single-token proposal techniques.

{{</citation>}}


### (89/136) Chain of Code: Reasoning with a Language Model-Augmented Code Emulator (Chengshu Li et al., 2023)

{{<citation>}}

Chengshu Li, Jacky Liang, Andy Zeng, Xinyun Chen, Karol Hausman, Dorsa Sadigh, Sergey Levine, Li Fei-Fei, Fei Xia, Brian Ichter. (2023)  
**Chain of Code: Reasoning with a Language Model-Augmented Code Emulator**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-RO, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.04474v2)  

---


**ABSTRACT**  
Code provides a general syntactic structure to build complex programs and perform precise computations when paired with a code interpreter - we hypothesize that language models (LMs) can leverage code-writing to improve Chain of Thought reasoning not only for logic and arithmetic tasks, but also for semantic ones (and in particular, those that are a mix of both). For example, consider prompting an LM to write code that counts the number of times it detects sarcasm in an essay: the LM may struggle to write an implementation for "detect_sarcasm(string)" that can be executed by the interpreter (handling the edge cases would be insurmountable). However, LMs may still produce a valid solution if they not only write code, but also selectively "emulate" the interpreter by generating the expected output of "detect_sarcasm(string)" and other lines of code that cannot be executed. In this work, we propose Chain of Code (CoC), a simple yet surprisingly effective extension that improves LM code-driven reasoning. The key idea is to encourage LMs to format semantic sub-tasks in a program as flexible pseudocode that the interpreter can explicitly catch undefined behaviors and hand off to simulate with an LM (as an "LMulator"). Experiments demonstrate that Chain of Code outperforms Chain of Thought and other baselines across a variety of benchmarks; on BIG-Bench Hard, Chain of Code achieves 84%, a gain of 12% over Chain of Thought. CoC scales well with large and small models alike, and broadens the scope of reasoning questions that LMs can correctly answer by "thinking in code". Project webpage: https://chain-of-code.github.io.

{{</citation>}}


### (90/136) Fortify the Shortest Stave in Attention: Enhancing Context Awareness of Large Language Models for Effective Tool Use (Yuhan Chen et al., 2023)

{{<citation>}}

Yuhan Chen, Ang Lv, Ting-En Lin, Changyu Chen, Yuchuan Wu, Fei Huang, Yongbin Li, Rui Yan. (2023)  
**Fortify the Shortest Stave in Attention: Enhancing Context Awareness of Large Language Models for Effective Tool Use**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Attention, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04455v1)  

---


**ABSTRACT**  
Recent advancements in large language models (LLMs) have significantly expanded their functionality and skills as tool agents. In this paper, we argue that a waveform pattern in the model's attention allocation has an impact on the tool use performance, which degrades when the position of essential information hits the trough zone. To address this issue, we propose a novel inference method named Attention Buckets. This approach enables LLMs to handle context by conducting parallel processes, each featuring a unique RoPE angle base that shapes the attention waveform. Attention Buckets ensures that an attention trough of a particular process can be compensated with an attention peak of another run, reducing the risk of the LLM missing essential information residing within the attention trough. Our extensive experiments on the widely recognized tool use benchmark demonstrate the efficacy of our approach, where a 7B-parameter open-source model enhanced by Attention Buckets achieves SOTA performance on par with GPT-4.

{{</citation>}}


### (91/136) OpenAsp: A Benchmark for Multi-document Open Aspect-based Summarization (Shmuel Amar et al., 2023)

{{<citation>}}

Shmuel Amar, Liat Schiff, Ori Ernst, Asi Shefer, Ori Shapira, Ido Dagan. (2023)  
**OpenAsp: A Benchmark for Multi-document Open Aspect-based Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2312.04440v1)  

---


**ABSTRACT**  
The performance of automatic summarization models has improved dramatically in recent years. Yet, there is still a gap in meeting specific information needs of users in real-world scenarios, particularly when a targeted summary is sought, such as in the useful aspect-based summarization setting targeted in this paper. Previous datasets and studies for this setting have predominantly concentrated on a limited set of pre-defined aspects, focused solely on single document inputs, or relied on synthetic data. To advance research on more realistic scenarios, we introduce OpenAsp, a benchmark for multi-document \textit{open} aspect-based summarization. This benchmark is created using a novel and cost-effective annotation protocol, by which an open aspect dataset is derived from existing generic multi-document summarization datasets. We analyze the properties of OpenAsp showcasing its high-quality content. Further, we show that the realistic open-aspect setting realized in OpenAsp poses a challenge for current state-of-the-art summarization models, as well as for large language models.

{{</citation>}}


### (92/136) LaMPilot: An Open Benchmark Dataset for Autonomous Driving with Language Model Programs (Yunsheng Ma et al., 2023)

{{<citation>}}

Yunsheng Ma, Can Cui, Xu Cao, Wenqian Ye, Peiran Liu, Juanwu Lu, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, Aniket Bera, James M. Rehg, Ziran Wang. (2023)  
**LaMPilot: An Open Benchmark Dataset for Autonomous Driving with Language Model Programs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04372v1)  

---


**ABSTRACT**  
We present LaMPilot, a novel framework for planning in the field of autonomous driving, rethinking the task as a code-generation process that leverages established behavioral primitives. This approach aims to address the challenge of interpreting and executing spontaneous user instructions such as "overtake the car ahead," which have typically posed difficulties for existing frameworks. We introduce the LaMPilot benchmark specifically designed to quantitatively evaluate the efficacy of Large Language Models (LLMs) in translating human directives into actionable driving policies. We then evaluate a wide range of state-of-the-art code generation language models on tasks from the LaMPilot Benchmark. The results of the experiments showed that GPT-4, with human feedback, achieved an impressive task completion rate of 92.7% and a minimal collision rate of 0.9%. To encourage further investigation in this area, our code and dataset will be made available.

{{</citation>}}


### (93/136) PCoQA: Persian Conversational Question Answering Dataset (Hamed Hematian Hemati et al., 2023)

{{<citation>}}

Hamed Hematian Hemati, Atousa Toghyani, Atena Souri, Sayed Hesam Alavian, Hossein Sameti, Hamid Beigy. (2023)  
**PCoQA: Persian Conversational Question Answering Dataset**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.04362v1)  

---


**ABSTRACT**  
Humans seek information regarding a specific topic through performing a conversation containing a series of questions and answers. In the pursuit of conversational question answering research, we introduce the PCoQA, the first \textbf{P}ersian \textbf{Co}nversational \textbf{Q}uestion \textbf{A}nswering dataset, a resource comprising information-seeking dialogs encompassing a total of 9,026 contextually-driven questions. Each dialog involves a questioner, a responder, and a document from the Wikipedia; The questioner asks several inter-connected questions from the text and the responder provides a span of the document as the answer for each question. PCoQA is designed to present novel challenges compared to previous question answering datasets including having more open-ended non-factual answers, longer answers, and fewer lexical overlaps. This paper not only presents the comprehensive PCoQA dataset but also reports the performance of various benchmark models. Our models include baseline models and pre-trained models, which are leveraged to boost the performance of the model. The dataset and benchmarks are available at our Github page.

{{</citation>}}


### (94/136) CLadder: A Benchmark to Assess Causal Reasoning Capabilities of Language Models (Zhijing Jin et al., 2023)

{{<citation>}}

Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojasv Kamal, Zhiheng Lyu, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, Bernhard Schölkopf. (2023)  
**CLadder: A Benchmark to Assess Causal Reasoning Capabilities of Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.04350v1)  

---


**ABSTRACT**  
The ability to perform causal reasoning is widely considered a core feature of intelligence. In this work, we investigate whether large language models (LLMs) can coherently reason about causality. Much of the existing work in natural language processing (NLP) focuses on evaluating commonsense causal reasoning in LLMs, thus failing to assess whether a model can perform causal inference in accordance with a set of well-defined formal rules. To address this, we propose a new NLP task, causal inference in natural language, inspired by the "causal inference engine" postulated by Judea Pearl et al. We compose a large dataset, CLadder, with 10K samples: based on a collection of causal graphs and queries (associational, interventional, and counterfactual), we obtain symbolic questions and ground-truth answers, through an oracle causal inference engine. These are then translated into natural language. We evaluate multiple LLMs on our dataset, and we introduce and evaluate a bespoke chain-of-thought prompting strategy, CausalCoT. We show that our task is highly challenging for LLMs, and we conduct an in-depth analysis to gain deeper insight into the causal reasoning abilities of LLMs. Our data is open-sourced at https://huggingface.co/datasets/causalNLP/cladder, and our code can be found at https://github.com/causalNLP/cladder.

{{</citation>}}


### (95/136) Enhancing Medical Task Performance in GPT-4V: A Comprehensive Study on Prompt Engineering Strategies (Pengcheng Chen et al., 2023)

{{<citation>}}

Pengcheng Chen, Ziyan Huang, Zhongying Deng, Tianbin Li, Yanzhou Su, Haoyu Wang, Jin Ye, Yu Qiao, Junjun He. (2023)  
**Enhancing Medical Task Performance in GPT-4V: A Comprehensive Study on Prompt Engineering Strategies**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.04344v2)  

---


**ABSTRACT**  
OpenAI's latest large vision-language model (LVLM), GPT-4V(ision), has piqued considerable interest for its potential in medical applications. Despite its promise, recent studies and internal reviews highlight its underperformance in specialized medical tasks. This paper explores the boundary of GPT-4V's capabilities in medicine, particularly in processing complex imaging data from endoscopies, CT scans, and MRIs etc. Leveraging open-source datasets, we assessed its foundational competencies, identifying substantial areas for enhancement. Our research emphasizes prompt engineering, an often-underutilized strategy for improving AI responsiveness. Through iterative testing, we refined the model's prompts, significantly improving its interpretative accuracy and relevance in medical imaging. From our comprehensive evaluations, we distilled 10 effective prompt engineering techniques, each fortifying GPT-4V's medical acumen. These methodical enhancements facilitate more reliable, precise, and clinically valuable insights from GPT-4V, advancing its operability in critical healthcare environments. Our findings are pivotal for those employing AI in medicine, providing clear, actionable guidance on harnessing GPT-4V's full diagnostic potential.

{{</citation>}}


### (96/136) Beyond Surface: Probing LLaMA Across Scales and Layers (Nuo Chen et al., 2023)

{{<citation>}}

Nuo Chen, Ning Wu, Shining Liang, Ming Gong, Linjun Shou, Dongmei Zhang, Jia Li. (2023)  
**Beyond Surface: Probing LLaMA Across Scales and Layers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04333v2)  

---


**ABSTRACT**  
This paper presents an in-depth analysis of Large Language Models (LLMs), focusing on LLaMA, a prominent open-source foundational model in natural language processing. Instead of assessing LLaMA through its generative output, we design multiple-choice tasks to probe its intrinsic understanding in high-order tasks such as reasoning and computation. We examine the model horizontally, comparing different sizes, and vertically, assessing different layers. We unveil several key and uncommon findings based on the designed probing tasks: (1) Horizontally, enlarging model sizes almost could not automatically impart additional knowledge or computational prowess. Instead, it can enhance reasoning abilities, especially in math problem solving, and helps reduce hallucinations, but only beyond certain size thresholds; (2) In vertical analysis, the lower layers of LLaMA lack substantial arithmetic and factual knowledge, showcasing logical thinking, multilingual and recognitive abilities, with top layers housing most computational power and real-world knowledge.

{{</citation>}}


### (97/136) nerblackbox: A High-level Library for Named Entity Recognition in Python (Felix Stollenwerk, 2023)

{{<citation>}}

Felix Stollenwerk. (2023)  
**nerblackbox: A High-level Library for Named Entity Recognition in Python**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2312.04306v1)  

---


**ABSTRACT**  
We present nerblackbox, a python library to facilitate the use of state-of-the-art transformer-based models for named entity recognition. It provides simple-to-use yet powerful methods to access data and models from a wide range of sources, for fully automated model training and evaluation as well as versatile model inference. While many technical challenges are solved and hidden from the user by default, nerblackbox also offers fine-grained control and a rich set of customizable features. It is thus targeted both at application-oriented developers as well as machine learning experts and researchers.

{{</citation>}}


### (98/136) PsyChat: A Client-Centric Dialogue System for Mental Health Support (Huachuan Qiu et al., 2023)

{{<citation>}}

Huachuan Qiu, Anqi Li, Lizhi Ma, Zhenzhong Lan. (2023)  
**PsyChat: A Client-Centric Dialogue System for Mental Health Support**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.04262v1)  

---


**ABSTRACT**  
Dialogue systems are increasingly integrated into mental health support to help clients facilitate exploration, gain insight, take action, and ultimately heal themselves. For a dialogue system to be practical and user-friendly, it should be client-centric, focusing on the client's behaviors. However, existing dialogue systems publicly available for mental health support often concentrate solely on the counselor's strategies rather than the behaviors expressed by clients. This can lead to the implementation of unreasonable or inappropriate counseling strategies and corresponding responses from the dialogue system. To address this issue, we propose PsyChat, a client-centric dialogue system that provides psychological support through online chat. The client-centric dialogue system comprises five modules: client behavior recognition, counselor strategy selection, input packer, response generator intentionally fine-tuned to produce responses, and response selection. Both automatic and human evaluations demonstrate the effectiveness and practicality of our proposed dialogue system for real-life mental health support. Furthermore, we employ our proposed dialogue system to simulate a real-world client-virtual-counselor interaction scenario. The system is capable of predicting the client's behaviors, selecting appropriate counselor strategies, and generating accurate and suitable responses, as demonstrated in the scenario.

{{</citation>}}


### (99/136) Language Model Knowledge Distillation for Efficient Question Answering in Spanish (Adrián Bazaga et al., 2023)

{{<citation>}}

Adrián Bazaga, Pietro Liò, Gos Micklem. (2023)  
**Language Model Knowledge Distillation for Efficient Question Answering in Spanish**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL, stat-ML  
Keywords: BERT, Knowledge Distillation, Language Model, NLP, Natural Language Processing, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.04193v1)  

---


**ABSTRACT**  
Recent advances in the development of pre-trained Spanish language models has led to significant progress in many Natural Language Processing (NLP) tasks, such as question answering. However, the lack of efficient models imposes a barrier for the adoption of such models in resource-constrained environments. Therefore, smaller distilled models for the Spanish language could be proven to be highly scalable and facilitate their further adoption on a variety of tasks and scenarios. In this work, we take one step in this direction by developing SpanishTinyRoBERTa, a compressed language model based on RoBERTa for efficient question answering in Spanish. To achieve this, we employ knowledge distillation from a large model onto a lighter model that allows for a wider implementation, even in areas with limited computational resources, whilst attaining negligible performance sacrifice. Our experiments show that the dense distilled model can still preserve the performance of its larger counterpart, while significantly increasing inference speedup. This work serves as a starting point for further research and investigation of model compression efforts for Spanish language models across various NLP tasks.

{{</citation>}}


### (100/136) Analyzing the Inherent Response Tendency of LLMs: Real-World Instructions-Driven Jailbreak (Yanrui Du et al., 2023)

{{<citation>}}

Yanrui Du, Sendong Zhao, Ming Ma, Yuhan Chen, Bing Qin. (2023)  
**Analyzing the Inherent Response Tendency of LLMs: Real-World Instructions-Driven Jailbreak**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.04127v1)  

---


**ABSTRACT**  
Extensive work has been devoted to improving the safety mechanism of Large Language Models (LLMs). However, in specific scenarios, LLMs still generate harmful responses when faced with malicious instructions, a phenomenon referred to as "Jailbreak Attack". In our research, we introduce a novel jailbreak attack method (\textbf{RADIAL}), which consists of two steps: 1) Inherent Response Tendency Analysis: we analyze the inherent affirmation and rejection tendency of LLMs to react to real-world instructions. 2) Real-World Instructions-Driven Jailbreak: based on our analysis, we strategically choose several real-world instructions and embed malicious instructions into them to amplify the LLM's potential to generate harmful responses. On three open-source human-aligned LLMs, our method achieves excellent jailbreak attack performance for both Chinese and English malicious instructions. Besides, we guided detailed ablation experiments and verified the effectiveness of our core idea "Inherent Response Tendency Analysis". Our exploration also exposes the vulnerability of LLMs to being induced into generating more detailed harmful responses in subsequent rounds of dialogue.

{{</citation>}}


### (101/136) Comparing Large Language Model AI and Human-Generated Coaching Messages for Behavioral Weight Loss (Zhuoran Huang et al., 2023)

{{<citation>}}

Zhuoran Huang, Michael P. Berry, Christina Chwyl, Gary Hsieh, Jing Wei, Evan M. Forman. (2023)  
**Comparing Large Language Model AI and Human-Generated Coaching Messages for Behavioral Weight Loss**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04059v1)  

---


**ABSTRACT**  
Automated coaching messages for weight control can save time and costs, but their repetitive, generic nature may limit their effectiveness compared to human coaching. Large language model (LLM) based artificial intelligence (AI) chatbots, like ChatGPT, could offer more personalized and novel messages to address repetition with their data-processing abilities. While LLM AI demonstrates promise to encourage healthier lifestyles, studies have yet to examine the feasibility and acceptability of LLM-based BWL coaching. 87 adults in a weight-loss trial rated ten coaching messages' helpfulness (five human-written, five ChatGPT-generated) using a 5-point Likert scale, providing additional open-ended feedback to justify their ratings. Participants also identified which messages they believed were AI-generated. The evaluation occurred in two phases: messages in Phase 1 were perceived as impersonal and negative, prompting revisions for Phase 2 messages. In Phase 1, AI-generated messages were rated less helpful than human-written ones, with 66 percent receiving a helpfulness rating of 3 or higher. However, in Phase 2, the AI messages matched the human-written ones regarding helpfulness, with 82% scoring three or above. Additionally, 50% were misidentified as human-written, suggesting AI's sophistication in mimicking human-generated content. A thematic analysis of open-ended feedback revealed that participants appreciated AI's empathy and personalized suggestions but found them more formulaic, less authentic, and too data-focused. This study reveals the preliminary feasibility and acceptability of LLM AIs, like ChatGPT, in crafting potentially effective weight control coaching messages. Our findings also underscore areas for future enhancement.

{{</citation>}}


### (102/136) Multimodal Misinformation Detection in a South African Social Media Environment (Amica De Jager et al., 2023)

{{<citation>}}

Amica De Jager, Vukosi Marivate, Abioudun Modupe. (2023)  
**Multimodal Misinformation Detection in a South African Social Media Environment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Social Media  
[Paper Link](http://arxiv.org/abs/2312.04052v1)  

---


**ABSTRACT**  
With the constant spread of misinformation on social media networks, a need has arisen to continuously assess the veracity of digital content. This need has inspired numerous research efforts on the development of misinformation detection (MD) models. However, many models do not use all information available to them and existing research contains a lack of relevant datasets to train the models, specifically within the South African social media environment. The aim of this paper is to investigate the transferability of knowledge of a MD model between different contextual environments. This research contributes a multimodal MD model capable of functioning in the South African social media environment, as well as introduces a South African misinformation dataset. The model makes use of multiple sources of information for misinformation detection, namely: textual and visual elements. It uses bidirectional encoder representations from transformers (BERT) as the textual encoder and a residual network (ResNet) as the visual encoder. The model is trained and evaluated on the Fakeddit dataset and a South African misinformation dataset. Results show that using South African samples in the training of the model increases model performance, in a South African contextual environment, and that a multimodal model retains significantly more knowledge than both the textual and visual unimodal models. Our study suggests that the performance of a misinformation detection model is influenced by the cultural nuances of its operating environment and multimodal models assist in the transferability of knowledge between different contextual environments. Therefore, local data should be incorporated into the training process of a misinformation detection model in order to optimize model performance.

{{</citation>}}


### (103/136) RoAST: Robustifying Language Models via Adversarial Perturbation with Selective Training (Jaehyung Kim et al., 2023)

{{<citation>}}

Jaehyung Kim, Yuning Mao, Rui Hou, Hanchao Yu, Davis Liang, Pascale Fung, Qifan Wang, Fuli Feng, Lifu Huang, Madian Khabsa. (2023)  
**RoAST: Robustifying Language Models via Adversarial Perturbation with Selective Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.04032v1)  

---


**ABSTRACT**  
Fine-tuning pre-trained language models (LMs) has become the de facto standard in many NLP tasks. Nevertheless, fine-tuned LMs are still prone to robustness issues, such as adversarial robustness and model calibration. Several perspectives of robustness for LMs have been studied independently, but lacking a unified consideration in multiple perspectives. In this paper, we propose Robustifying LMs via Adversarial perturbation with Selective Training (RoAST), a simple yet effective fine-tuning technique to enhance the multi-perspective robustness of LMs in a unified way. RoAST effectively incorporates two important sources for the model robustness, robustness on the perturbed inputs and generalizable knowledge in pre-trained LMs. To be specific, RoAST introduces adversarial perturbation during fine-tuning while the model parameters are selectively updated upon their relative importance to minimize unnecessary deviation. Under a unified evaluation of fine-tuned LMs by incorporating four representative perspectives of model robustness, we demonstrate the effectiveness of RoAST compared to state-of-the-art fine-tuning methods on six different types of LMs, which indicates its usefulness in practice.

{{</citation>}}


### (104/136) Cost-Effective In-Context Learning for Entity Resolution: A Design Space Exploration (Meihao Fan et al., 2023)

{{<citation>}}

Meihao Fan, Xiaoyue Han, Ju Fan, Chengliang Chai, Nan Tang, Guoliang Li, Xiaoyong Du. (2023)  
**Cost-Effective In-Context Learning for Entity Resolution: A Design Space Exploration**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.03987v1)  

---


**ABSTRACT**  
Entity resolution (ER) is an important data integration task with a wide spectrum of applications. The state-of-the-art solutions on ER rely on pre-trained language models (PLMs), which require fine-tuning on a lot of labeled matching/non-matching entity pairs. Recently, large languages models (LLMs), such as GPT-4, have shown the ability to perform many tasks without tuning model parameters, which is known as in-context learning (ICL) that facilitates effective learning from a few labeled input context demonstrations. However, existing ICL approaches to ER typically necessitate providing a task description and a set of demonstrations for each entity pair and thus have limitations on the monetary cost of interfacing LLMs. To address the problem, in this paper, we provide a comprehensive study to investigate how to develop a cost-effective batch prompting approach to ER. We introduce a framework BATCHER consisting of demonstration selection and question batching and explore different design choices that support batch prompting for ER. We also devise a covering-based demonstration selection strategy that achieves an effective balance between matching accuracy and monetary cost. We conduct a thorough evaluation to explore the design space and evaluate our proposed strategies. Through extensive experiments, we find that batch prompting is very cost-effective for ER, compared with not only PLM-based methods fine-tuned with extensive labeled data but also LLM-based methods with manually designed prompting. We also provide guidance for selecting appropriate design choices for batch prompting.

{{</citation>}}


## cs.SE (4)



### (105/136) STraceBERT: Source Code Retrieval using Semantic Application Traces (Claudio Spiess, 2023)

{{<citation>}}

Claudio Spiess. (2023)  
**STraceBERT: Source Code Retrieval using Semantic Application Traces**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-IR, cs-SE, cs.SE  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.04731v1)  

---


**ABSTRACT**  
Software reverse engineering is an essential task in software engineering and security, but it can be a challenging process, especially for adversarial artifacts. To address this challenge, we present STraceBERT, a novel approach that utilizes a Java dynamic analysis tool to record calls to core Java libraries, and pretrain a BERT-style model on the recorded application traces for effective method source code retrieval from a candidate set. Our experiments demonstrate the effectiveness of STraceBERT in retrieving the source code compared to existing approaches. Our proposed approach offers a promising solution to the problem of code retrieval in software reverse engineering and opens up new avenues for further research in this area.

{{</citation>}}


### (106/136) LLM4TDD: Best Practices for Test Driven Development Using Large Language Models (Sanyogita Piya et al., 2023)

{{<citation>}}

Sanyogita Piya, Allison Sullivan. (2023)  
**LLM4TDD: Best Practices for Test Driven Development Using Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.04687v1)  

---


**ABSTRACT**  
In today's society, we are becoming increasingly dependent on software systems. However, we also constantly witness the negative impacts of buggy software. Program synthesis aims to improve software correctness by automatically generating the program given an outline of the expected behavior. For decades, program synthesis has been an active research field, with recent approaches looking to incorporate Large Language Models to help generate code. This paper explores the concept of LLM4TDD, where we guide Large Language Models to generate code iteratively using a test-driven development methodology. We conduct an empirical evaluation using ChatGPT and coding problems from LeetCode to investigate the impact of different test, prompt and problem attributes on the efficacy of LLM4TDD.

{{</citation>}}


### (107/136) Leveraging Transformer-based Language Models to Automate Requirements Satisfaction Assessment (Amrit Poudel et al., 2023)

{{<citation>}}

Amrit Poudel, Jinfeng Lin, Jane Cleland-Huang. (2023)  
**Leveraging Transformer-based Language Models to Automate Requirements Satisfaction Assessment**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BERT, Information Retrieval, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.04463v1)  

---


**ABSTRACT**  
Requirements Satisfaction Assessment (RSA) evaluates whether the set of design elements linked to a single requirement provide sufficient coverage of that requirement -- typically meaning that all concepts in the requirement are addressed by at least one of the design elements. RSA is an important software engineering activity for systems with any form of hierarchical decomposition -- especially safety or mission critical ones. In previous studies, researchers used basic Information Retrieval (IR) models to decompose requirements and design elements into chunks, and then evaluated the extent to which chunks of design elements covered all chunks in the requirement. However, results had low accuracy because many critical concepts that extend across the entirety of the sentence were not well represented when the sentence was parsed into independent chunks. In this paper we leverage recent advances in natural language processing to deliver significantly more accurate results. We propose two major architectures: Satisfaction BERT (Sat-BERT), and Dual-Satisfaction BERT (DSat-BERT), along with their multitask learning variants to improve satisfaction assessments. We perform RSA on five different datasets and compare results from our variants against the chunk-based legacy approach. All BERT-based models significantly outperformed the legacy baseline, and Sat-BERT delivered the best results returning an average improvement of 124.75% in Mean Average Precision.

{{</citation>}}


### (108/136) Occlusion-based Detection of Trojan-triggering Inputs in Large Language Models of Code (Aftab Hussain et al., 2023)

{{<citation>}}

Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Mohammad Amin Alipour, Bowen Xu. (2023)  
**Occlusion-based Detection of Trojan-triggering Inputs in Large Language Models of Code**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.04004v2)  

---


**ABSTRACT**  
Large language models (LLMs) are becoming an integrated part of software development. These models are trained on large datasets for code, where it is hard to verify each data point. Therefore, a potential attack surface can be to inject poisonous data into the training data to make models vulnerable, aka trojaned. It can pose a significant threat by hiding manipulative behaviors inside models, leading to compromising the integrity of the models in downstream tasks.   In this paper, we propose an occlusion-based human-in-the-loop technique, OSeql, to distinguish trojan-triggering inputs of code. The technique is based on the observation that trojaned neural models of code rely heavily on the triggering part of input; hence, its removal would change the confidence of the models in their prediction substantially. Our results suggest that OSeql can detect the triggering inputs with almost 100% recall. We discuss the problem of false positives and how to address them. These results provide a baseline for future studies in this field.

{{</citation>}}


## cs.CY (2)



### (109/136) The Impact of AI Innovations on U.S. Occupations (Ali Akbar Septiandri et al., 2023)

{{<citation>}}

Ali Akbar Septiandri, Marios Constantinides, Daniele Quercia. (2023)  
**The Impact of AI Innovations on U.S. Occupations**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.04714v1)  

---


**ABSTRACT**  
AI's impact has traditionally been assessed in terms of occupations. However, an occupation is comprised of interconnected tasks, and it is these tasks, not occupations themselves, that are affected by AI. To evaluate how tasks may be impacted, previous approaches utilized subjective manual annotations or coarse-grained matching with patents. Leveraging recent advancements in machine learning, we replace coarse-grained matching with more precise deep learning approaches. Introducing the AI Impact (AII) measure, we employ Deep Learning Natural Language Processing to automatically identify AI patents that impact various occupational tasks at scale. Our methodology relies on a comprehensive dataset of 19,498 task descriptions and quantifies AI's impact through analysis of 12,984 AI patents filed with the United States Patent and Trademark Office (USPTO) between 2015 and 2020. Our observations reveal that the impact of AI on occupations defies simplistic categorizations based on task complexity, challenging the conventional belief that the dichotomy between basic and advanced skills alone explains the effects of AI. Instead, the impact is intricately linked to specific skills, whether basic or advanced, associated with particular tasks. For instance, while basic skills like scanning items may be affected, others like cooking may not. Similarly, certain advanced skills, such as image analysis in radiology, may face impact, while skills involving interpersonal relationships may remain unaffected. Furthermore, the influence of AI extends beyond knowledge-centric regions. Regions in the U.S. that heavily rely on industries susceptible to AI changes, often characterized by economic inequality or a lack of economic diversification, will experience notable AI impact.

{{</citation>}}


### (110/136) Can apparent bystanders distinctively shape an outcome? Global south countries and global catastrophic risk-focused governance of artificial intelligence (Cecil Abungu et al., 2023)

{{<citation>}}

Cecil Abungu, Michelle Malonza, Sumaya Nur Adan. (2023)  
**Can apparent bystanders distinctively shape an outcome? Global south countries and global catastrophic risk-focused governance of artificial intelligence**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04616v1)  

---


**ABSTRACT**  
Increasingly, there is well-grounded concern that through perpetual scaling-up of computation power and data, current deep learning techniques will create highly capable artificial intelligence that could pursue goals in a manner that is not aligned with human values. In turn, such AI could have the potential of leading to a scenario in which there is serious global-scale damage to human wellbeing. Against this backdrop, a number of researchers and public policy professionals have been developing ideas about how to govern AI in a manner that reduces the chances that it could lead to a global catastrophe. The jurisdictional focus of a vast majority of their assessments so far has been the United States, China, and Europe. That preference seems to reveal an assumption underlying most of the work in this field: That global south countries can only have a marginal role in attempts to govern AI development from a global catastrophic risk -focused perspective. Our paper sets out to undermine this assumption. We argue that global south countries like India and Singapore (and specific coalitions) could in fact be fairly consequential in the global catastrophic risk-focused governance of AI. We support our position using 4 key claims. 3 are constructed out of the current ways in which advanced foundational AI models are built and used while one is constructed on the strategic roles that global south countries and coalitions have historically played in the design and use of multilateral rules and institutions. As each claim is elaborated, we also suggest some ways through which global south countries can play a positive role in designing, strengthening and operationalizing global catastrophic risk-focused AI governance.

{{</citation>}}


## cs.DC (3)



### (111/136) Optimizing Distributed Reinforcement Learning with Reactor Model and Lingua Franca (Jacky Kwok et al., 2023)

{{<citation>}}

Jacky Kwok, Marten Lohstroh, Edward A. Lee. (2023)  
**Optimizing Distributed Reinforcement Learning with Reactor Model and Lingua Franca**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-LG, cs.DC  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.04704v1)  

---


**ABSTRACT**  
Distributed Reinforcement Learning (RL) frameworks are essential for mapping RL workloads to multiple computational resources, allowing for faster generation of samples, estimation of values, and policy improvement. These computational paradigms require a seamless integration of training, serving, and simulation workloads. Existing frameworks, such as Ray, are not managing this orchestration efficiently. In this study, we've proposed a solution implementing Reactor Model, which enforces a set of actors to have a fixed communication pattern. This allows the scheduler to eliminate works needed for synchronization, such as acquiring and releasing locks for each actor or sending and processing coordination-related messages. Our framework, Lingua Franca (LF), a coordination language based on the Reactor Model, also provides a unified interface that allows users to automatically generate dataflow graphs for distributed RL. On average, LF outperformed Ray in generating samples from OpenAI Gym and Atari environments by 1.21x and 11.62x, reduced the average training time of synchronized parallel Q-learning by 31.2%, and accelerated Multi-Agent RL inference by 5.12x.

{{</citation>}}


### (112/136) Developing Elementary Federated Learning Algorithms Leveraging the ChatGPT (Miroslav Popovic et al., 2023)

{{<citation>}}

Miroslav Popovic, Marko Popovic, Ivan Kastelan, Miodrag Djukic, Ilija Basicevic. (2023)  
**Developing Elementary Federated Learning Algorithms Leveraging the ChatGPT**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.04412v1)  

---


**ABSTRACT**  
The Python Testbed for Federated Learning Algorithms is a simple Python FL framework easy to use by ML&AI developers who do not need to be professional programmers, and this paper shows that it is also amenable to emerging AI tools. In this paper, we successfully developed three elementary FL algorithms using the following three steps process: (i) specify context, (ii) ask ChatGPT to complete server and clients' callback functions, and (iii) verify the generated code.

{{</citation>}}


### (113/136) An Improved Scheduling with Advantage Actor-Critic for Storm Workloads (Gaoqiang Dong et al., 2023)

{{<citation>}}

Gaoqiang Dong, Jia Wang, Mingjing Wang, Tingting Su. (2023)  
**An Improved Scheduling with Advantage Actor-Critic for Storm Workloads**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2312.04126v1)  

---


**ABSTRACT**  
Various resources as the essential elements of data centers, and the completion time is vital to users. In terms of the persistence, the periodicity and the spatial-temporal dependence of stream workload, a new Storm scheduler with Advantage Actor-Critic is proposed to improve resource utilization for minimizing the completion time. A new weighted embedding with a Graph Neural Network is designed to depend on the features of a job comprehensively, which includes the dependence, the types and the positions of tasks in a job. An improved Advantage Actor-Critic integrating task chosen and executor assignment is proposed to schedule tasks to executors in order to better resource utilization. Then the status of tasks and executors are updated for the next scheduling. Compared to existing methods, experimental results show that the proposed Storm scheduler improves resource utilization. The completion time is reduced by almost 17\% on the TPC-H data set and reduced by almost 25\% on the Alibaba data set.

{{</citation>}}


## cs.RO (3)



### (114/136) Rapid Motor Adaptation for Robotic Manipulator Arms (Yichao Liang et al., 2023)

{{<citation>}}

Yichao Liang, Kevin Ellis, João Henriques. (2023)  
**Rapid Motor Adaptation for Robotic Manipulator Arms**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04670v1)  

---


**ABSTRACT**  
Developing generalizable manipulation skills is a core challenge in embodied AI. This includes generalization across diverse task configurations, encompassing variations in object shape, density, friction coefficient, and external disturbances such as forces applied to the robot. Rapid Motor Adaptation (RMA) offers a promising solution to this challenge. It posits that essential hidden variables influencing an agent's task performance, such as object mass and shape, can be effectively inferred from the agent's action and proprioceptive history. Drawing inspiration from RMA in locomotion and in-hand rotation, we use depth perception to develop agents tailored for rapid motor adaptation in a variety of manipulation tasks. We evaluated our agents on four challenging tasks from the Maniskill2 benchmark, namely pick-and-place operations with hundreds of objects from the YCB and EGAD datasets, peg insertion with precise position and orientation, and operating a variety of faucets and handles, with customized environment variations. Empirical results demonstrate that our agents surpass state-of-the-art methods like automatic domain randomization and vision-based policies, obtaining better generalization performance and sample efficiency.

{{</citation>}}


### (115/136) Dream2Real: Zero-Shot 3D Object Rearrangement with Vision-Language Models (Ivan Kapelyukh et al., 2023)

{{<citation>}}

Ivan Kapelyukh, Yifei Ren, Ignacio Alzugaray, Edward Johns. (2023)  
**Dream2Real: Zero-Shot 3D Object Rearrangement with Vision-Language Models**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.04533v1)  

---


**ABSTRACT**  
We introduce Dream2Real, a robotics framework which integrates vision-language models (VLMs) trained on 2D data into a 3D object rearrangement pipeline. This is achieved by the robot autonomously constructing a 3D representation of the scene, where objects can be rearranged virtually and an image of the resulting arrangement rendered. These renders are evaluated by a VLM, so that the arrangement which best satisfies the user instruction is selected and recreated in the real world with pick-and-place. This enables language-conditioned rearrangement to be performed zero-shot, without needing to collect a training dataset of example arrangements. Results on a series of real-world tasks show that this framework is robust to distractors, controllable by language, capable of understanding complex multi-object relations, and readily applicable to both tabletop and 6-DoF rearrangement tasks.

{{</citation>}}


### (116/136) Semi-Supervised Active Learning for Semantic Segmentation in Unknown Environments Using Informative Path Planning (Julius Rückin et al., 2023)

{{<citation>}}

Julius Rückin, Federico Magistri, Cyrill Stachniss, Marija Popović. (2023)  
**Semi-Supervised Active Learning for Semantic Segmentation in Unknown Environments Using Informative Path Planning**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Active Learning, Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.04402v1)  

---


**ABSTRACT**  
Semantic segmentation enables robots to perceive and reason about their environments beyond geometry. Most of such systems build upon deep learning approaches. As autonomous robots are commonly deployed in initially unknown environments, pre-training on static datasets cannot always capture the variety of domains and limits the robot's perception performance during missions. Recently, self-supervised and fully supervised active learning methods emerged to improve a robot's vision. These approaches rely on large in-domain pre-training datasets or require substantial human labelling effort. We propose a planning method for semi-supervised active learning of semantic segmentation that substantially reduces human labelling requirements compared to fully supervised approaches. We leverage an adaptive map-based planner guided towards the frontiers of unexplored space with high model uncertainty collecting training data for human labelling. A key aspect of our approach is to combine the sparse high-quality human labels with pseudo labels automatically extracted from highly certain environment map areas. Experimental results show that our method reaches segmentation performance close to fully supervised approaches with drastically reduced human labelling effort while outperforming self-supervised approaches.

{{</citation>}}


## eess.SY (1)



### (117/136) Data-Driven Robust Reinforcement Learning Control of Uncertain Nonlinear Systems: Towards a Fully-Automated, Insulin-Based Artificial Pancreas (Alexandros Tanzanakis et al., 2023)

{{<citation>}}

Alexandros Tanzanakis, John Lygeros. (2023)  
**Data-Driven Robust Reinforcement Learning Control of Uncertain Nonlinear Systems: Towards a Fully-Automated, Insulin-Based Artificial Pancreas**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.04503v1)  

---


**ABSTRACT**  
In this paper, a novel robust tracking control scheme for a general class of discrete-time nonlinear systems affected by unknown bounded uncertainty is presented. By solving a parameterized optimal tracking control problem subject to the unknown nominal system and a suitable cost function, the resulting optimal tracking control policy can ensure closed-loop stability by achieving a sufficiently small tracking error for the original uncertain nonlinear system. The computation of the optimal tracking controller is accomplished through the derivation of a novel Q-function-based $\lambda$-Policy Iteration algorithm. The proposed algorithm not only enjoys rigorous theoretical guarantees, but also avoids technical weaknesses of conventional reinforcement learning methods. By employing a data-driven, critic-only least squares implementation, the performance of the proposed algorithm is evaluated to the problem of fully-automated, insulin-based, closed-loop glucose control for patients diagnosed with Type 1 and Type 2 Diabetes Mellitus. The U.S. FDA-accepted DMMS.R simulator from the Epsilon Group is used to conduct a comprehensive in silico clinical campaign on a rich set of virtual subjects under completely unannounced meal and exercise settings. Simulation results underline the superior glycaemic behavior achieved by the derived approach, as well as its overall maturity for the design of highly-effective, closed-loop drug delivery systems for personalized medicine.

{{</citation>}}


## cs.HC (1)



### (118/136) AVA: Towards Autonomous Visualization Agents through Visual Perception-Driven Decision-Making (Shusen Liu et al., 2023)

{{<citation>}}

Shusen Liu, Haichao Miao, Zhimin Li, Matthew Olson, Valerio Pascucci, Peer-Timo Bremer. (2023)  
**AVA: Towards Autonomous Visualization Agents through Visual Perception-Driven Decision-Making**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CV, cs-GR, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04494v1)  

---


**ABSTRACT**  
With recent advances in multi-modal foundation models, the previously text-only large language models (LLM) have evolved to incorporate visual input, opening up unprecedented opportunities for various applications in visualization. Our work explores the utilization of the visual perception ability of multi-modal LLMs to develop Autonomous Visualization Agents (AVAs) that can interpret and accomplish user-defined visualization objectives through natural language. We propose the first framework for the design of AVAs and present several usage scenarios intended to demonstrate the general applicability of the proposed paradigm. The addition of visual perception allows AVAs to act as the virtual visualization assistant for domain experts who may lack the knowledge or expertise in fine-tuning visualization outputs. Our preliminary exploration and proof-of-concept agents suggest that this approach can be widely applicable whenever the choices of appropriate visualization parameters require the interpretation of previous visual output. Feedback from unstructured interviews with experts in AI research, medical visualization, and radiology has been incorporated, highlighting the practicality and potential of AVAs. Our study indicates that AVAs represent a general paradigm for designing intelligent visualization systems that can achieve high-level visualization goals, which pave the way for developing expert-level visualization agents in the future.

{{</citation>}}


## physics.ed-ph (1)



### (119/136) Testing LLM performance on the Physics GRE: some observations (Pranav Gupta, 2023)

{{<citation>}}

Pranav Gupta. (2023)  
**Testing LLM performance on the Physics GRE: some observations**  

---
Primary Category: physics.ed-ph  
Categories: cs-LG, physics-ed-ph, physics.ed-ph  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.04613v1)  

---


**ABSTRACT**  
With the recent developments in large language models (LLMs) and their widespread availability through open source models and/or low-cost APIs, several exciting products and applications are emerging, many of which are in the field of STEM educational technology for K-12 and university students. There is a need to evaluate these powerful language models on several benchmarks, in order to understand their risks and limitations. In this short paper, we summarize and analyze the performance of Bard, a popular LLM-based conversational service made available by Google, on the standardized Physics GRE examination.

{{</citation>}}


## cs.MM (1)



### (120/136) Deep3DSketch: 3D modeling from Free-hand Sketches with View- and Structural-Aware Adversarial Training (Tianrun Chen et al., 2023)

{{<citation>}}

Tianrun Chen, Chenglong Fu, Lanyun Zhu, Papa Mao, Jia Zhang, Ying Zang, Lingyun Sun. (2023)  
**Deep3DSketch: 3D modeling from Free-hand Sketches with View- and Structural-Aware Adversarial Training**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Adversarial Training, Sketch  
[Paper Link](http://arxiv.org/abs/2312.04435v1)  

---


**ABSTRACT**  
This work aims to investigate the problem of 3D modeling using single free-hand sketches, which is one of the most natural ways we humans express ideas. Although sketch-based 3D modeling can drastically make the 3D modeling process more accessible, the sparsity and ambiguity of sketches bring significant challenges for creating high-fidelity 3D models that reflect the creators' ideas. In this work, we propose a view- and structural-aware deep learning approach, \textit{Deep3DSketch}, which tackles the ambiguity and fully uses sparse information of sketches, emphasizing the structural information. Specifically, we introduced random pose sampling on both 3D shapes and 2D silhouettes, and an adversarial training scheme with an effective progressive discriminator to facilitate learning of the shape structures. Extensive experiments demonstrated the effectiveness of our approach, which outperforms existing methods -- with state-of-the-art (SOTA) performance on both synthetic and real datasets.

{{</citation>}}


## cs.SI (2)



### (121/136) Content Moderation on Social Media in the EU: Insights From the DSA Transparency Database (Chiara Drolsbach et al., 2023)

{{<citation>}}

Chiara Drolsbach, Nicolas Pröllochs. (2023)  
**Content Moderation on Social Media in the EU: Insights From the DSA Transparency Database**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2312.04431v1)  

---


**ABSTRACT**  
The Digital Services Act (DSA) requires large social media platforms in the EU to provide clear and specific information whenever they remove or restrict access to certain content. These "Statements of Reasons" (SoRs) are collected in the DSA Transparency Database to ensure transparency and scrutiny of content moderation decisions of the providers of online platforms. In this work, we empirically analyze 156 million SoRs within an observation period of two months to provide an early look at content moderation decisions of social media platforms in the EU. Our empirical analysis yields the following main findings: (i) There are vast differences in the frequency of content moderation across platforms. For instance, TikTok performs more than 350 times more content moderation decisions per user than X/Twitter. (ii) Content moderation is most commonly applied for text and videos, whereas images and other content formats undergo moderation less frequently. (ii) The primary reasons for moderation include content falling outside the platform's scope of service, illegal/harmful speech, and pornography/sexualized content, with moderation of misinformation being relatively uncommon. (iii) The majority of rule-breaking content is detected and decided upon via automated means rather than manual intervention. However, X/Twitter reports that it relies solely on non-automated methods. (iv) There is significant variation in the content moderation actions taken across platforms. Altogether, our study implies inconsistencies in how social media platforms implement their obligations under the DSA -- resulting in a fragmented outcome that the DSA is meant to avoid. Our findings have important implications for regulators to clarify existing guidelines or lay out more specific rules that ensure common standards on how social media providers handle rule-breaking content on their platforms.

{{</citation>}}


### (122/136) All Polarized but Still Different: a Multi-factorial Metric to Discriminate between Polarization Behaviors on Social Media (Celina Treuillier et al., 2023)

{{<citation>}}

Celina Treuillier, Sylvain Castagnos, Armelle Brun. (2023)  
**All Polarized but Still Different: a Multi-factorial Metric to Discriminate between Polarization Behaviors on Social Media**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: AI, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2312.04603v1)  

---


**ABSTRACT**  
Online polarization has attracted the attention of researchers for many years. Its effects on society are a cause for concern, and the design of personalized depolarization strategies appears to be a key solution. Such strategies should rely on a fine and accurate measurement, and a clear understanding of polarization behaviors. However, the literature still lacks ways to characterize them finely. We propose GRAIL, the first individual polarization metric, relying on multiple factors. GRAIL assesses these factors through entropy and is based on an adaptable Generalized Additive Model. We evaluate the proposed metric on a Twitter dataset related to the highly controversial debate about the COVID-19 vaccine. Experiments confirm the ability of GRAIL to discriminate between polarization behaviors. To go further, we provide a finer characterization and explanation of the identified behaviors through an innovative evaluation framework.

{{</citation>}}


## cs.AI (4)



### (123/136) Scalable Knowledge Graph Construction and Inference on Human Genome Variants (Shivika Prasanna et al., 2023)

{{<citation>}}

Shivika Prasanna, Deepthi Rao, Eduardo Simoes, Praveen Rao. (2023)  
**Scalable Knowledge Graph Construction and Inference on Human Genome Variants**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-DB, cs.AI, q-bio-QM  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.04423v1)  

---


**ABSTRACT**  
Real-world knowledge can be represented as a graph consisting of entities and relationships between the entities. The need for efficient and scalable solutions arises when dealing with vast genomic data, like RNA-sequencing. Knowledge graphs offer a powerful approach for various tasks in such large-scale genomic data, such as analysis and inference. In this work, variant-level information extracted from the RNA-sequences of vaccine-na\"ive COVID-19 patients have been represented as a unified, large knowledge graph. Variant call format (VCF) files containing the variant-level information were annotated to include further information for each variant. The data records in the annotated files were then converted to Resource Description Framework (RDF) triples. Each VCF file obtained had an associated CADD scores file that contained the raw and Phred-scaled scores for each variant. An ontology was defined for the VCF and CADD scores files. Using this ontology and the extracted information, a large, scalable knowledge graph was created. Available graph storage was then leveraged to query and create datasets for further downstream tasks. We also present a case study using the knowledge graph and perform a classification task using graph machine learning. We also draw comparisons between different Graph Neural Networks (GNNs) for the case study.

{{</citation>}}


### (124/136) How much informative is your XAI? A decision-making assessment task to objectively measure the goodness of explanations (Marco Matarese et al., 2023)

{{<citation>}}

Marco Matarese, Francesco Rea, Alessandra Sciutti. (2023)  
**How much informative is your XAI? A decision-making assessment task to objectively measure the goodness of explanations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-RO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.04379v1)  

---


**ABSTRACT**  
There is an increasing consensus about the effectiveness of user-centred approaches in the explainable artificial intelligence (XAI) field. Indeed, the number and complexity of personalised and user-centred approaches to XAI have rapidly grown in recent years. Often, these works have a two-fold objective: (1) proposing novel XAI techniques able to consider the users and (2) assessing the \textit{goodness} of such techniques with respect to others. From these new works, it emerged that user-centred approaches to XAI positively affect the interaction between users and systems. However, so far, the goodness of XAI systems has been measured through indirect measures, such as performance. In this paper, we propose an assessment task to objectively and quantitatively measure the goodness of XAI systems in terms of their \textit{information power}, which we intended as the amount of information the system provides to the users during the interaction. Moreover, we plan to use our task to objectively compare two XAI techniques in a human-robot decision-making task to understand deeper whether user-centred approaches are more informative than classical ones.

{{</citation>}}


### (125/136) AI and Jobs: Has the Inflection Point Arrived? Evidence from an Online Labor Platform (Dandan Qiao et al., 2023)

{{<citation>}}

Dandan Qiao, Huaxia Rui, Qian Xiong. (2023)  
**AI and Jobs: Has the Inflection Point Arrived? Evidence from an Online Labor Platform**  

---
Primary Category: cs.AI  
Categories: J-4, cs-AI, cs-CY, cs.AI, econ-GN, q-fin-EC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.04180v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) refers to the ability of machines or software to mimic or even surpass human intelligence in a given cognitive task. While humans learn by both induction and deduction, the success of current AI is rooted in induction, relying on its ability to detect statistical regularities in task input -- an ability learnt from a vast amount of training data using enormous computation resources. We examine the performance of such a statistical AI in a human task through the lens of four factors, including task learnability, statistical resource, computation resource, and learning techniques, and then propose a three-phase visual framework to understand the evolving relation between AI and jobs. Based on this conceptual framework, we develop a simple economic model of competition to show the existence of an inflection point for each occupation. Before AI performance crosses the inflection point, human workers always benefit from an improvement in AI performance, but after the inflection point, human workers become worse off whenever such an improvement occurs. To offer empirical evidence, we first argue that AI performance has passed the inflection point for the occupation of translation but not for the occupation of web development. We then study how the launch of ChatGPT, which led to significant improvement of AI performance on many tasks, has affected workers in these two occupations on a large online labor platform. Consistent with the inflection point conjecture, we find that translators are negatively affected by the shock both in terms of the number of accepted jobs and the earnings from those jobs, while web developers are positively affected by the very same shock. Given the potentially large disruption of AI on employment, more studies on more occupations using data from different platforms are urgently needed.

{{</citation>}}


### (126/136) Using a Large Language Model to generate a Design Structure Matrix (Edwin C. Y. Koh, 2023)

{{<citation>}}

Edwin C. Y. Koh. (2023)  
**Using a Large Language Model to generate a Design Structure Matrix**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.04134v1)  

---


**ABSTRACT**  
The Design Structure Matrix (DSM) is an established method used in dependency modelling, especially in the design of complex engineering systems. The generation of DSM is traditionally carried out through manual means and can involve interviewing experts to elicit critical system elements and the relationships between them. Such manual approaches can be time-consuming and costly. This paper presents a workflow that uses a Large Language Model (LLM) to support the generation of DSM and improve productivity. A prototype of the workflow was developed in this work and applied on a diesel engine DSM published previously. It was found that the prototype could reproduce 357 out of 462 DSM entries published (i.e. 77.3%), suggesting that the work can aid DSM generation. A no-code version of the prototype is made available online to support future research.

{{</citation>}}


## eess.IV (2)



### (127/136) Adversarial Denoising Diffusion Model for Unsupervised Anomaly Detection (Jongmin Yu et al., 2023)

{{<citation>}}

Jongmin Yu, Hyeontaek Oh, Jinhong Yang. (2023)  
**Adversarial Denoising Diffusion Model for Unsupervised Anomaly Detection**  

---
Primary Category: eess.IV  
Categories: cs-AI, eess-IV, eess.IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.04382v1)  

---


**ABSTRACT**  
In this paper, we propose the Adversarial Denoising Diffusion Model (ADDM). The ADDM is based on the Denoising Diffusion Probabilistic Model (DDPM) but complementarily trained by adversarial learning. The proposed adversarial learning is achieved by classifying model-based denoised samples and samples to which random Gaussian noise is added to a specific sampling step. With the addition of explicit adversarial learning on data samples, ADDM can learn the semantic characteristics of the data more robustly during training, which achieves a similar data sampling performance with much fewer sampling steps than DDPM. We apply ADDM to anomaly detection in unsupervised MRI images. Experimental results show that the proposed ADDM outperformed existing generative model-based unsupervised anomaly detection methods. In particular, compared to other DDPM-based anomaly detection methods, the proposed ADDM shows better performance with the same number of sampling steps and similar performance with 50% fewer sampling steps.

{{</citation>}}


### (128/136) Guided Reconstruction with Conditioned Diffusion Models for Unsupervised Anomaly Detection in Brain MRIs (Finn Behrendt et al., 2023)

{{<citation>}}

Finn Behrendt, Debayan Bhattacharya, Robin Mieling, Lennart Maack, Julia Krüger, Roland Opfer, Alexander Schlaefer. (2023)  
**Guided Reconstruction with Conditioned Diffusion Models for Unsupervised Anomaly Detection in Brain MRIs**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.04215v1)  

---


**ABSTRACT**  
Unsupervised anomaly detection in Brain MRIs aims to identify abnormalities as outliers from a healthy training distribution. Reconstruction-based approaches that use generative models to learn to reconstruct healthy brain anatomy are commonly used for this task. Diffusion models are an emerging class of deep generative models that show great potential regarding reconstruction fidelity. However, they face challenges in preserving intensity characteristics in the reconstructed images, limiting their performance in anomaly detection. To address this challenge, we propose to condition the denoising mechanism of diffusion models with additional information about the image to reconstruct coming from a latent representation of the noise-free input image. This conditioning enables high-fidelity reconstruction of healthy brain structures while aligning local intensity characteristics of input-reconstruction pairs. We evaluate our method's reconstruction quality, domain adaptation features and finally segmentation performance on publicly available data sets with various pathologies. Using our proposed conditioning mechanism we can reduce the false-positive predictions and enable a more precise delineation of anomalies which significantly enhances the anomaly detection performance compared to established state-of-the-art approaches to unsupervised anomaly detection in brain MRI. Furthermore, our approach shows promise in domain adaptation across different MRI acquisitions and simulated contrasts, a crucial property of general anomaly detection methods.

{{</citation>}}


## math.OC (1)



### (129/136) A Scalable Network-Aware Multi-Agent Reinforcement Learning Framework for Decentralized Inverter-based Voltage Control (Han Xu et al., 2023)

{{<citation>}}

Han Xu, Jialin Zheng, Guannan Qu. (2023)  
**A Scalable Network-Aware Multi-Agent Reinforcement Learning Framework for Decentralized Inverter-based Voltage Control**  

---
Primary Category: math.OC  
Categories: cs-LG, cs-MA, cs-SY, eess-SY, math-OC, math.OC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.04371v1)  

---


**ABSTRACT**  
This paper addresses the challenges associated with decentralized voltage control in power grids due to an increase in distributed generations (DGs). Traditional model-based voltage control methods struggle with the rapid energy fluctuations and uncertainties of these DGs. While multi-agent reinforcement learning (MARL) has shown potential for decentralized secondary control, scalability issues arise when dealing with a large number of DGs. This problem lies in the dominant centralized training and decentralized execution (CTDE) framework, where the critics take global observations and actions. To overcome these challenges, we propose a scalable network-aware (SNA) framework that leverages network structure to truncate the input to the critic's Q-function, thereby improving scalability and reducing communication costs during training. Further, the SNA framework is theoretically grounded with provable approximation guarantee, and it can seamlessly integrate with multiple multi-agent actor-critic algorithms. The proposed SNA framework is successfully demonstrated in a system with 114 DGs, providing a promising solution for decentralized voltage control in increasingly complex power grid systems.

{{</citation>}}


## physics.ao-ph (1)



### (130/136) Simulating the Air Quality Impact of Prescribed Fires Using a Graph Neural Network-Based PM$_{2.5}$ Emissions Forecasting System (Kyleen Liao et al., 2023)

{{<citation>}}

Kyleen Liao, Jatan Buch, Kara Lamb, Pierre Gentine. (2023)  
**Simulating the Air Quality Impact of Prescribed Fires Using a Graph Neural Network-Based PM$_{2.5}$ Emissions Forecasting System**  

---
Primary Category: physics.ao-ph  
Categories: cs-LG, physics-ao-ph, physics.ao-ph  
Keywords: Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2312.04291v1)  

---


**ABSTRACT**  
The increasing size and severity of wildfires across western North America have generated dangerous levels of PM$_{2.5}$ pollution in recent years. In a warming climate, expanding the use of prescribed fires is widely considered to be the most robust fire mitigation strategy. However, reliably forecasting the potential air quality impact from these prescribed fires, a critical ingredient in determining the fires' location and time, at hourly to daily time scales remains a challenging problem. This paper proposes a novel integration of prescribed fire simulation with a spatio-temporal graph neural network-based PM$_{2.5}$ forecasting model. The experiments in this work focus on determining the optimal time for implementing prescribed fires in California as well as quantifying the potential air quality trade-offs involved in conducting more prescribed fires outside the fire season.

{{</citation>}}


## cs.MA (1)



### (131/136) Mastering Complex Coordination through Attention-based Dynamic Graph (Guangchong Zhou et al., 2023)

{{<citation>}}

Guangchong Zhou, Zhiwei Xu, Zeren Zhang, Guoliang Fan. (2023)  
**Mastering Complex Coordination through Attention-based Dynamic Graph**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-MA, cs.MA  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.04245v1)  

---


**ABSTRACT**  
The coordination between agents in multi-agent systems has become a popular topic in many fields. To catch the inner relationship between agents, the graph structure is combined with existing methods and improves the results. But in large-scale tasks with numerous agents, an overly complex graph would lead to a boost in computational cost and a decline in performance. Here we present DAGMIX, a novel graph-based value factorization method. Instead of a complete graph, DAGMIX generates a dynamic graph at each time step during training, on which it realizes a more interpretable and effective combining process through the attention mechanism. Experiments show that DAGMIX significantly outperforms previous SOTA methods in large-scale scenarios, as well as achieving promising results on other tasks.

{{</citation>}}


## stat.ML (1)



### (132/136) Multi-scale Residual Transformer for VLF Lightning Transients Classification (Jinghao Sun et al., 2023)

{{<citation>}}

Jinghao Sun, Tingting Ji, Guoyu Wang, Rui Wang. (2023)  
**Multi-scale Residual Transformer for VLF Lightning Transients Classification**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.04163v1)  

---


**ABSTRACT**  
The utilization of Very Low Frequency (VLF) electromagnetic signals in navigation systems is widespread. However, the non-stationary behavior of lightning signals can affect VLF electromagnetic signal transmission. Accurately classifying lightning signals is important for reducing interference and noise in VLF, thereby improving the reliability and overall performance of navigation systems. In recent years, the evolution of deep learning, specifically Convolutional Neural Network (CNNs), has sparked a transformation in lightning classification, surpassing traditional statistical methodologies. Existing CNN models have limitations as they overlook the diverse attributes of lightning signals across different scales and neglect the significance of temporal sequencing in sequential signals. This study introduces an innovative multi-scale residual transform (MRTransformer) that not only has the ability to discern intricate fine-grained patterns while also weighing the significance of different aspects within the input lightning signal sequence. This model performs the attributes of the lightning signal across different scales and the level of accuracy reached 90% in the classification. In future work, this model has the potential applied to a comprehensive understanding of the localization and waveform characteristics of lightning signals.

{{</citation>}}


## eess.SP (1)



### (133/136) Resource Allocation for Semantic Communication under Physical-layer Security (Yang Li et al., 2023)

{{<citation>}}

Yang Li, Xinyu Zhou, Jun Zhao. (2023)  
**Resource Allocation for Semantic Communication under Physical-layer Security**  

---
Primary Category: eess.SP  
Categories: cs-CR, cs-LG, eess-SP, eess.SP  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.04155v1)  

---


**ABSTRACT**  
Semantic communication is deemed as a revolution of Shannon's paradigm in the six-generation (6G) wireless networks. It aims at transmitting the extracted information rather than the original data, which receivers will try to recover. Intuitively, the larger extracted information, the longer latency of semantic communication will be. Besides, larger extracted information will result in more accurate reconstructed information, thereby causing a higher utility of the semantic communication system. Shorter latency and higher utility are desirable objectives for the system, so there will be a trade-off between utility and latency. This paper proposes a joint optimization algorithm for total latency and utility. Moreover, security is essential for the semantic communication system. We incorporate the secrecy rate, a physical-layer security method, into the optimization problem. The secrecy rate is the communication rate at which no information is disclosed to an eavesdropper. Experimental results demonstrate that the proposed algorithm obtains the best joint optimization performance compared to the baselines.

{{</citation>}}


## eess.AS (1)



### (134/136) Joint Training or Not: An Exploration of Pre-trained Speech Models in Audio-Visual Speaker Diarization (Huan Zhao et al., 2023)

{{<citation>}}

Huan Zhao, Li Zhang, Yue Li, Yannan Wang, Hongji Wang, Wei Rao, Qing Wang, Lei Xie. (2023)  
**Joint Training or Not: An Exploration of Pre-trained Speech Models in Audio-Visual Speaker Diarization**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: BERT, Transformer  
[Paper Link](http://arxiv.org/abs/2312.04131v1)  

---


**ABSTRACT**  
The scarcity of labeled audio-visual datasets is a constraint for training superior audio-visual speaker diarization systems. To improve the performance of audio-visual speaker diarization, we leverage pre-trained supervised and self-supervised speech models for audio-visual speaker diarization. Specifically, we adopt supervised~(ResNet and ECAPA-TDNN) and self-supervised pre-trained models~(WavLM and HuBERT) as the speaker and audio embedding extractors in an end-to-end audio-visual speaker diarization~(AVSD) system. Then we explore the effectiveness of different frameworks, including Transformer, Conformer, and cross-attention mechanism, in the audio-visual decoder. To mitigate the degradation of performance caused by separate training, we jointly train the audio encoder, speaker encoder, and audio-visual decoder in the AVSD system. Experiments on the MISP dataset demonstrate that the proposed method achieves superior performance and obtained third place in MISP Challenge 2022.

{{</citation>}}


## cs.IR (1)



### (135/136) Synergistic Signals: Exploiting Co-Engagement and Semantic Links via Graph Neural Networks (Zijie Huang et al., 2023)

{{<citation>}}

Zijie Huang, Baolin Li, Hafez Asgharzadeh, Anne Cocos, Lingyi Liu, Evan Cox, Colby Wise, Sudarshan Lamkhede. (2023)  
**Synergistic Signals: Exploiting Co-Engagement and Semantic Links via Graph Neural Networks**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.04071v1)  

---


**ABSTRACT**  
Given a set of candidate entities (e.g. movie titles), the ability to identify similar entities is a core capability of many recommender systems. Most often this is achieved by collaborative filtering approaches, i.e. if users co-engage with a pair of entities frequently enough, the embeddings should be similar. However, relying on co-engagement data alone can result in lower-quality embeddings for new and unpopular entities. We study this problem in the context recommender systems at Netflix. We observe that there is abundant semantic information such as genre, content maturity level, themes, etc. that complements co-engagement signals and provides interpretability in similarity models. To learn entity similarities from both data sources holistically, we propose a novel graph-based approach called SemanticGNN. SemanticGNN models entities, semantic concepts, collaborative edges, and semantic edges within a large-scale knowledge graph and conducts representation learning over it. Our key technical contributions are twofold: (1) we develop a novel relation-aware attention graph neural network (GNN) to handle the imbalanced distribution of relation types in our graph; (2) to handle web-scale graph data that has millions of nodes and billions of edges, we develop a novel distributed graph training paradigm. The proposed model is successfully deployed within Netflix and empirical experiments indicate it yields up to 35% improvement in performance on similarity judgment tasks.

{{</citation>}}


## q-bio.BM (1)



### (136/136) Efficiently Predicting Protein Stability Changes Upon Single-point Mutation with Large Language Models (Yijie Zhang et al., 2023)

{{<citation>}}

Yijie Zhang, Zhangyang Gao, Cheng Tan, Stan Z. Li. (2023)  
**Efficiently Predicting Protein Stability Changes Upon Single-point Mutation with Large Language Models**  

---
Primary Category: q-bio.BM  
Categories: cs-AI, q-bio-BM, q-bio.BM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.04019v1)  

---


**ABSTRACT**  
Predicting protein stability changes induced by single-point mutations has been a persistent challenge over the years, attracting immense interest from numerous researchers. The ability to precisely predict protein thermostability is pivotal for various subfields and applications in biochemistry, including drug development, protein evolution analysis, and enzyme synthesis. Despite the proposition of multiple methodologies aimed at addressing this issue, few approaches have successfully achieved optimal performance coupled with high computational efficiency. Two principal hurdles contribute to the existing challenges in this domain. The first is the complexity of extracting and aggregating sufficiently representative features from proteins. The second refers to the limited availability of experimental data for protein mutation analysis, further complicating the comprehensive evaluation of model performance on unseen data samples. With the advent of Large Language Models(LLM), such as the ESM models in protein research, profound interpretation of protein features is now accessibly aided by enormous training data. Therefore, LLMs are indeed to facilitate a wide range of protein research. In our study, we introduce an ESM-assisted efficient approach that integrates protein sequence and structural features to predict the thermostability changes in protein upon single-point mutations. Furthermore, we have curated a dataset meticulously designed to preclude data leakage, corresponding to two extensively employed test datasets, to facilitate a more equitable model comparison.

{{</citation>}}
