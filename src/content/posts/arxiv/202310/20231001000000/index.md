---
draft: false
title: "arXiv @ 2023.10.01"
date: 2023-10-01
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.01"
    identifier: arxiv_20231001
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (41)](#cslg-41)
- [cs.CL (32)](#cscl-32)
- [cs.AI (11)](#csai-11)
- [cs.RO (6)](#csro-6)
- [cs.CV (31)](#cscv-31)
- [cs.CY (2)](#cscy-2)
- [cs.HC (1)](#cshc-1)
- [cs.DL (1)](#csdl-1)
- [astro-ph.IM (1)](#astro-phim-1)
- [cs.SD (1)](#cssd-1)
- [eess.IV (3)](#eessiv-3)
- [q-fin.GN (1)](#q-fingn-1)
- [quant-ph (2)](#quant-ph-2)
- [q-bio.QM (1)](#q-bioqm-1)
- [stat.ML (2)](#statml-2)
- [cs.IT (2)](#csit-2)
- [cs.MM (1)](#csmm-1)
- [cs.SI (1)](#cssi-1)
- [hep-lat (1)](#hep-lat-1)
- [cs.IR (1)](#csir-1)
- [cs.DB (1)](#csdb-1)
- [eess.AS (2)](#eessas-2)
- [eess.SY (1)](#eesssy-1)
- [cs.DC (1)](#csdc-1)

## cs.LG (41)



### (1/147) MARL: Multi-scale Archetype Representation Learning for Urban Building Energy Modeling (Xinwei Zhuang et al., 2023)

{{<citation>}}

Xinwei Zhuang, Zixun Huang, Wentao Zeng, Luisa Caldas. (2023)  
**MARL: Multi-scale Archetype Representation Learning for Urban Building Energy Modeling**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-HC, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.00180v1)  

---


**ABSTRACT**  
Building archetypes, representative models of building stock, are crucial for precise energy simulations in Urban Building Energy Modeling. The current widely adopted building archetypes are developed on a nationwide scale, potentially neglecting the impact of local buildings' geometric specificities. We present Multi-scale Archetype Representation Learning (MARL), an approach that leverages representation learning to extract geometric features from a specific building stock. Built upon VQ-AE, MARL encodes building footprints and purifies geometric information into latent vectors constrained by multiple architectural downstream tasks. These tailored representations are proven valuable for further clustering and building energy modeling. The advantages of our algorithm are its adaptability with respect to the different building footprint sizes, the ability for automatic generation across multi-scale regions, and the preservation of geometric features across neighborhoods and local ecologies. In our study spanning five regions in LA County, we show MARL surpasses both conventional and VQ-AE extracted archetypes in performance. Results demonstrate that geometric feature embeddings significantly improve the accuracy and reliability of energy consumption estimates. Code, dataset and trained models are publicly available: https://github.com/ZixunHuang1997/MARL-BuildingEnergyEstimation

{{</citation>}}


### (2/147) SCoRe: Submodular Combinatorial Representation Learning for Real-World Class-Imbalanced Settings (Anay Majee et al., 2023)

{{<citation>}}

Anay Majee, Suraj Kothawade, Krishnateja Killiamsetty, Rishabh Iyer. (2023)  
**SCoRe: Submodular Combinatorial Representation Learning for Real-World Class-Imbalanced Settings**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.00165v1)  

---


**ABSTRACT**  
Representation Learning in real-world class-imbalanced settings has emerged as a challenging task in the evolution of deep learning. Lack of diversity in visual and structural features for rare classes restricts modern neural networks to learn discriminative feature clusters. This manifests in the form of large inter-class bias between rare object classes and elevated intra-class variance among abundant classes in the dataset. Although deep metric learning approaches have shown promise in this domain, significant improvements need to be made to overcome the challenges associated with class-imbalance in mission critical tasks like autonomous navigation and medical diagnostics. Set-based combinatorial functions like Submodular Information Measures exhibit properties that allow them to simultaneously model diversity and cooperation among feature clusters. In this paper, we introduce the SCoRe (Submodular Combinatorial Representation Learning) framework and propose a family of Submodular Combinatorial Loss functions to overcome these pitfalls in contrastive learning. We also show that existing contrastive learning approaches are either submodular or can be re-formulated to create their submodular counterparts. We conduct experiments on the newly introduced family of combinatorial objectives on two image classification benchmarks - pathologically imbalanced CIFAR-10, subsets of MedMNIST and a real-world road object detection benchmark - India Driving Dataset (IDD). Our experiments clearly show that the newly introduced objectives like Facility Location, Graph-Cut and Log Determinant outperform state-of-the-art metric learners by up to 7.6% for the imbalanced classification tasks and up to 19.4% for object detection tasks.

{{</citation>}}


### (3/147) Probabilistic Sampling-Enhanced Temporal-Spatial GCN: A Scalable Framework for Transaction Anomaly Detection in Ethereum Networks (Stefan Kambiz Behfar et al., 2023)

{{<citation>}}

Stefan Kambiz Behfar, Jon Crowcroft. (2023)  
**Probabilistic Sampling-Enhanced Temporal-Spatial GCN: A Scalable Framework for Transaction Anomaly Detection in Ethereum Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: Anomaly Detection, GNN, Graph Convolutional Network, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.00144v1)  

---


**ABSTRACT**  
The rapid evolution of the Ethereum network necessitates sophisticated techniques to ensure its robustness against potential threats and to maintain transparency. While Graph Neural Networks (GNNs) have pioneered anomaly detection in such platforms, capturing the intricacies of both spatial and temporal transactional patterns has remained a challenge. This study presents a fusion of Graph Convolutional Networks (GCNs) with Temporal Random Walks (TRW) enhanced by probabilistic sampling to bridge this gap. Our approach, unlike traditional GCNs, leverages the strengths of TRW to discern complex temporal sequences in Ethereum transactions, thereby providing a more nuanced transaction anomaly detection mechanism. Preliminary evaluations demonstrate that our TRW-GCN framework substantially advances the performance metrics over conventional GCNs in detecting anomalies and transaction bursts. This research not only underscores the potential of temporal cues in Ethereum transactional data but also offers a scalable and effective methodology for ensuring the security and transparency of decentralized platforms. By harnessing both spatial relationships and time-based transactional sequences as node features, our model introduces an additional layer of granularity, making the detection process more robust and less prone to false positives. This work lays the foundation for future research aimed at optimizing and enhancing the transparency of blockchain technologies, and serves as a testament to the significance of considering both time and space dimensions in the ever-evolving landscape of the decentralized platforms.

{{</citation>}}


### (4/147) Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization (Mahyar Fazlyab et al., 2023)

{{<citation>}}

Mahyar Fazlyab, Taha Entesari, Aniket Roy, Rama Chellappa. (2023)  
**Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.00116v1)  

---


**ABSTRACT**  
To improve the robustness of deep classifiers against adversarial perturbations, many approaches have been proposed, such as designing new architectures with better robustness properties (e.g., Lipschitz-capped networks), or modifying the training process itself (e.g., min-max optimization, constrained learning, or regularization). These approaches, however, might not be effective at increasing the margin in the input (feature) space. As a result, there has been an increasing interest in developing training procedures that can directly manipulate the decision boundary in the input space. In this paper, we build upon recent developments in this category by developing a robust training algorithm whose objective is to increase the margin in the output (logit) space while regularizing the Lipschitz constant of the model along vulnerable directions. We show that these two objectives can directly promote larger margins in the input space. To this end, we develop a scalable method for calculating guaranteed differentiable upper bounds on the Lipschitz constant of neural networks accurately and efficiently. The relative accuracy of the bounds prevents excessive regularization and allows for more direct manipulation of the decision boundary. Furthermore, our Lipschitz bounding algorithm exploits the monotonicity and Lipschitz continuity of the activation layers, and the resulting bounds can be used to design new layers with controllable bounds on their Lipschitz constant. Experiments on the MNIST, CIFAR-10, and Tiny-ImageNet data sets verify that our proposed algorithm obtains competitively improved results compared to the state-of-the-art.

{{</citation>}}


### (5/147) Learning Over Molecular Conformer Ensembles: Datasets and Benchmarks (Yanqiao Zhu et al., 2023)

{{<citation>}}

Yanqiao Zhu, Jeehyun Hwang, Keir Adams, Zhen Liu, Bozhao Nan, Brock Stenfors, Yuanqi Du, Jatin Chauhan, Olaf Wiest, Olexandr Isayev, Connor W. Coley, Yizhou Sun, Wei Wang. (2023)  
**Learning Over Molecular Conformer Ensembles: Datasets and Benchmarks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.00115v1)  

---


**ABSTRACT**  
Molecular Representation Learning (MRL) has proven impactful in numerous biochemical applications such as drug discovery and enzyme design. While Graph Neural Networks (GNNs) are effective at learning molecular representations from a 2D molecular graph or a single 3D structure, existing works often overlook the flexible nature of molecules, which continuously interconvert across conformations via chemical bond rotations and minor vibrational perturbations. To better account for molecular flexibility, some recent works formulate MRL as an ensemble learning problem, focusing on explicitly learning from a set of conformer structures. However, most of these studies have limited datasets, tasks, and models. In this work, we introduce the first MoleculAR Conformer Ensemble Learning (MARCEL) benchmark to thoroughly evaluate the potential of learning on conformer ensembles and suggest promising research directions. MARCEL includes four datasets covering diverse molecule- and reaction-level properties of chemically diverse molecules including organocatalysts and transition-metal catalysts, extending beyond the scope of common GNN benchmarks that are confined to drug-like molecules. In addition, we conduct a comprehensive empirical study, which benchmarks representative 1D, 2D, and 3D molecular representation learning models, along with two strategies that explicitly incorporate conformer ensembles into 3D MRL models. Our findings reveal that direct learning from an accessible conformer space can improve performance on a variety of tasks and models.

{{</citation>}}


### (6/147) Reinforcement Learning for Node Selection in Branch-and-Bound (Alexander Mattick et al., 2023)

{{<citation>}}

Alexander Mattick, Christopher Mutschler. (2023)  
**Reinforcement Learning for Node Selection in Branch-and-Bound**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.00112v1)  

---


**ABSTRACT**  
A big challenge in branch and bound lies in identifying the optimal node within the search tree from which to proceed. Current state-of-the-art selectors utilize either hand-crafted ensembles that automatically switch between naive sub-node selectors, or learned node selectors that rely on individual node data. We propose a novel bi-simulation technique that uses reinforcement learning (RL) while considering the entire tree state, rather than just isolated nodes. To achieve this, we train a graph neural network that produces a probability distribution based on the path from the model's root to its ``to-be-selected'' leaves. Modelling node-selection as a probability distribution allows us to train the model using state-of-the-art RL techniques that capture both intrinsic node-quality and node-evaluation costs. Our method induces a high quality node selection policy on a set of varied and complex problem sets, despite only being trained on specially designed, synthetic TSP instances. Experiments on several benchmarks show significant improvements in optimality gap reductions and per-node efficiency under strict time constraints.

{{</citation>}}


### (7/147) FedAIoT: A Federated Learning Benchmark for Artificial Intelligence of Things (Samiul Alam et al., 2023)

{{<citation>}}

Samiul Alam, Tuo Zhang, Tiantian Feng, Hui Shen, Zhichao Cao, Dong Zhao, JeongGil Ko, Kiran Somasundaram, Shrikanth S. Narayanan, Salman Avestimehr, Mi Zhang. (2023)  
**FedAIoT: A Federated Learning Benchmark for Artificial Intelligence of Things**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-DL, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.00109v1)  

---


**ABSTRACT**  
There is a significant relevance of federated learning (FL) in the realm of Artificial Intelligence of Things (AIoT). However, most existing FL works are not conducted on datasets collected from authentic IoT devices that capture unique modalities and inherent challenges of IoT data. In this work, we introduce FedAIoT, an FL benchmark for AIoT to fill this critical gap. FedAIoT includes eight datatsets collected from a wide range of IoT devices. These datasets cover unique IoT modalities and target representative applications of AIoT. FedAIoT also includes a unified end-to-end FL framework for AIoT that simplifies benchmarking the performance of the datasets. Our benchmark results shed light on the opportunities and challenges of FL for AIoT. We hope FedAIoT could serve as an invaluable resource to foster advancements in the important field of FL for AIoT. The repository of FedAIoT is maintained at https://github.com/AIoT-MLSys-Lab/FedAIoT.

{{</citation>}}


### (8/147) Federated Learning with Differential Privacy for End-to-End Speech Recognition (Martin Pelikan et al., 2023)

{{<citation>}}

Martin Pelikan, Sheikh Shams Azam, Vitaly Feldman, Jan "Honza" Silovsky, Kunal Talwar, Tatiana Likhomanenko. (2023)  
**Federated Learning with Differential Privacy for End-to-End Speech Recognition**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG, stat-ML  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.00098v1)  

---


**ABSTRACT**  
While federated learning (FL) has recently emerged as a promising approach to train machine learning models, it is limited to only preliminary explorations in the domain of automatic speech recognition (ASR). Moreover, FL does not inherently guarantee user privacy and requires the use of differential privacy (DP) for robust privacy guarantees. However, we are not aware of prior work on applying DP to FL for ASR. In this paper, we aim to bridge this research gap by formulating an ASR benchmark for FL with DP and establishing the first baselines. First, we extend the existing research on FL for ASR by exploring different aspects of recent $\textit{large end-to-end transformer models}$: architecture design, seed models, data heterogeneity, domain shift, and impact of cohort size. With a $\textit{practical}$ number of central aggregations we are able to train $\textbf{FL models}$ that are \textbf{nearly optimal} even with heterogeneous data, a seed model from another domain, or no pre-trained seed model. Second, we apply DP to FL for ASR, which is non-trivial since DP noise severely affects model training, especially for large transformer models, due to highly imbalanced gradients in the attention block. We counteract the adverse effect of DP noise by reviving per-layer clipping and explaining why its effect is more apparent in our case than in the prior work. Remarkably, we achieve user-level ($7.2$, $10^{-9}$)-$\textbf{DP}$ (resp. ($4.5$, $10^{-9}$)-$\textbf{DP}$) with a 1.3% (resp. 4.6%) absolute drop in the word error rate for extrapolation to high (resp. low) population scale for $\textbf{FL with DP in ASR}$.

{{</citation>}}


### (9/147) Low-budget Black-box Optimization Algorithms Evaluated on BBOB and OpenAI Gym (Elena Raponi et al., 2023)

{{<citation>}}

Elena Raponi, Nathanael Rakotonirina Carraz, Jérémy Rapin, Carola Doerr, Olivier Teytaud. (2023)  
**Low-budget Black-box Optimization Algorithms Evaluated on BBOB and OpenAI Gym**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.00077v1)  

---


**ABSTRACT**  
The growing ubiquity of machine learning (ML) has led it to enter various areas of computer science, including black-box optimization (BBO). Recent research is particularly concerned with Bayesian optimization (BO). BO-based algorithms are popular in the ML community, as they are used for hyperparameter optimization and more generally for algorithm configuration. However, their efficiency decreases as the dimensionality of the problem and the budget of evaluations increase. Meanwhile, derivative-free optimization methods have evolved independently in the optimization community. Therefore, we urge to understand whether cross-fertilization is possible between the two communities, ML and BBO, i.e., whether algorithms that are heavily used in ML also work well in BBO and vice versa. Comparative experiments often involve rather small benchmarks and show visible problems in the experimental setup, such as poor initialization of baselines, overfitting due to problem-specific setting of hyperparameters, and low statistical significance.   With this paper, we update and extend a comparative study presented by Hutter et al. in 2013. We compare BBO tools for ML with more classical heuristics, first on the well-known BBOB benchmark suite from the COCO environment and then on Direct Policy Search for OpenAI Gym, a reinforcement learning benchmark. Our results confirm that BO-based optimizers perform well on both benchmarks when budgets are limited, albeit with a higher computational cost, while they are often outperformed by algorithms from other families when the evaluation budget becomes larger. We also show that some algorithms from the BBO community perform surprisingly well on ML tasks.

{{</citation>}}


### (10/147) Networked Inequality: Preferential Attachment Bias in Graph Neural Network Link Prediction (Arjun Subramonian et al., 2023)

{{<citation>}}

Arjun Subramonian, Levent Sagun, Yizhou Sun. (2023)  
**Networked Inequality: Preferential Attachment Bias in Graph Neural Network Link Prediction**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs-SI, cs.LG  
Keywords: Bias, GNN, Graph Convolutional Network, Graph Neural Network, Network Link Prediction  
[Paper Link](http://arxiv.org/abs/2309.17417v1)  

---


**ABSTRACT**  
Graph neural network (GNN) link prediction is increasingly deployed in citation, collaboration, and online social networks to recommend academic literature, collaborators, and friends. While prior research has investigated the dyadic fairness of GNN link prediction, the within-group fairness and ``rich get richer'' dynamics of link prediction remain underexplored. However, these aspects have significant consequences for degree and power imbalances in networks. In this paper, we shed light on how degree bias in networks affects Graph Convolutional Network (GCN) link prediction. In particular, we theoretically uncover that GCNs with a symmetric normalized graph filter have a within-group preferential attachment bias. We validate our theoretical analysis on real-world citation, collaboration, and online social networks. We further bridge GCN's preferential attachment bias with unfairness in link prediction and propose a new within-group fairness metric. This metric quantifies disparities in link prediction scores between social groups, towards combating the amplification of degree and power disparities. Finally, we propose a simple training-time strategy to alleviate within-group unfairness, and we show that it is effective on citation, online social, and credit networks.

{{</citation>}}


### (11/147) Cleanba: A Reproducible and Efficient Distributed Reinforcement Learning Platform (Shengyi Huang et al., 2023)

{{<citation>}}

Shengyi Huang, Jiayi Weng, Rujikorn Charakorn, Min Lin, Zhongwen Xu, Santiago Ontañón. (2023)  
**Cleanba: A Reproducible and Efficient Distributed Reinforcement Learning Platform**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.00036v1)  

---


**ABSTRACT**  
Distributed Deep Reinforcement Learning (DRL) aims to leverage more computational resources to train autonomous agents with less training time. Despite recent progress in the field, reproducibility issues have not been sufficiently explored. This paper first shows that the typical actor-learner framework can have reproducibility issues even if hyperparameters are controlled. We then introduce Cleanba, a new open-source platform for distributed DRL that proposes a highly reproducible architecture. Cleanba implements highly optimized distributed variants of PPO and IMPALA. Our Atari experiments show that these variants can obtain equivalent or higher scores than strong IMPALA baselines in moolib and torchbeast and PPO baseline in CleanRL. However, Cleanba variants present 1) shorter training time and 2) more reproducible learning curves in different hardware settings. Cleanba's source code is available at \url{https://github.com/vwxyzjn/cleanba}

{{</citation>}}


### (12/147) Adversarial Machine Learning in Latent Representations of Neural Networks (Milin Zhang et al., 2023)

{{<citation>}}

Milin Zhang, Mohammad Abdi, Francesco Restuccia. (2023)  
**Adversarial Machine Learning in Latent Representations of Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.17401v1)  

---


**ABSTRACT**  
Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge the resilience of distributed DNNs to adversarial action still remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and introduce two new measurements for distortion and robustness. Our theoretical findings indicate that (i) assuming the same level of information distortion, latent features are always more robust than input representations; (ii) the adversarial robustness is jointly determined by the feature dimension and the generalization capability of the DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN architectures, 6 different approaches for distributed DNN and 10 different adversarial attacks to the ImageNet-1K dataset. Our experimental results support our theoretical findings by showing that the compressed latent representations can reduce the success rate of adversarial attacks by 88% in the best case and by 57% on the average compared to attacks to the input space.

{{</citation>}}


### (13/147) AV-CPL: Continuous Pseudo-Labeling for Audio-Visual Speech Recognition (Andrew Rouditchenko et al., 2023)

{{<citation>}}

Andrew Rouditchenko, Ronan Collobert, Tatiana Likhomanenko. (2023)  
**AV-CPL: Continuous Pseudo-Labeling for Audio-Visual Speech Recognition**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SD, cs.LG, eess-AS, stat-ML  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.17395v1)  

---


**ABSTRACT**  
Audio-visual speech contains synchronized audio and visual information that provides cross-modal supervision to learn representations for both automatic speech recognition (ASR) and visual speech recognition (VSR). We introduce continuous pseudo-labeling for audio-visual speech recognition (AV-CPL), a semi-supervised method to train an audio-visual speech recognition (AVSR) model on a combination of labeled and unlabeled videos with continuously regenerated pseudo-labels. Our models are trained for speech recognition from audio-visual inputs and can perform speech recognition using both audio and visual modalities, or only one modality. Our method uses the same audio-visual model for both supervised training and pseudo-label generation, mitigating the need for external speech recognition models to generate pseudo-labels. AV-CPL obtains significant improvements in VSR performance on the LRS3 dataset while maintaining practical ASR and AVSR performance. Finally, using visual-only speech data, our method is able to leverage unlabeled visual speech to improve VSR.

{{</citation>}}


### (14/147) Tree Cross Attention (Leo Feng et al., 2023)

{{<citation>}}

Leo Feng, Frederick Tung, Hossein Hajimirsadeghi, Yoshua Bengio, Mohamed Osama Ahmed. (2023)  
**Tree Cross Attention**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.17388v1)  

---


**ABSTRACT**  
Cross Attention is a popular method for retrieving information from a set of context tokens for making predictions. At inference time, for each prediction, Cross Attention scans the full set of $\mathcal{O}(N)$ tokens. In practice, however, often only a small subset of tokens are required for good performance. Methods such as Perceiver IO are cheap at inference as they distill the information to a smaller-sized set of latent tokens $L < N$ on which cross attention is then applied, resulting in only $\mathcal{O}(L)$ complexity. However, in practice, as the number of input tokens and the amount of information to distill increases, the number of latent tokens needed also increases significantly. In this work, we propose Tree Cross Attention (TCA) - a module based on Cross Attention that only retrieves information from a logarithmic $\mathcal{O}(\log(N))$ number of tokens for performing inference. TCA organizes the data in a tree structure and performs a tree search at inference time to retrieve the relevant tokens for prediction. Leveraging TCA, we introduce ReTreever, a flexible architecture for token-efficient inference. We show empirically that Tree Cross Attention (TCA) performs comparable to Cross Attention across various classification and uncertainty regression tasks while being significantly more token-efficient. Furthermore, we compare ReTreever against Perceiver IO, showing significant gains while using the same number of tokens for inference.

{{</citation>}}


### (15/147) Module-wise Training of Neural Networks via the Minimizing Movement Scheme (Skander Karkar et al., 2023)

{{<citation>}}

Skander Karkar, Ibrahim Ayed, Emmanuel de Bézenac, Patrick Gallinari. (2023)  
**Module-wise Training of Neural Networks via the Minimizing Movement Scheme**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.17357v2)  

---


**ABSTRACT**  
Greedy layer-wise or module-wise training of neural networks is compelling in constrained and on-device settings where memory is limited, as it circumvents a number of problems of end-to-end back-propagation. However, it suffers from a stagnation problem, whereby early layers overfit and deeper layers stop increasing the test accuracy after a certain depth. We propose to solve this issue by introducing a module-wise regularization inspired by the minimizing movement scheme for gradient flows in distribution space. We call the method TRGL for Transport Regularized Greedy Learning and study it theoretically, proving that it leads to greedy modules that are regular and that progressively solve the task. Experimentally, we show improved accuracy of module-wise training of various architectures such as ResNets, Transformers and VGG, when our regularization is added, superior to that of other module-wise training methods and often to end-to-end training, with as much as 60% less memory usage.

{{</citation>}}


### (16/147) Efficient Biologically Plausible Adversarial Training (Matilde Tristany Farinha et al., 2023)

{{<citation>}}

Matilde Tristany Farinha, Thomas Ortner, Giorgia Dellaferrera, Benjamin Grewe, Angeliki Pantazi. (2023)  
**Efficient Biologically Plausible Adversarial Training**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2309.17348v2)  

---


**ABSTRACT**  
Artificial Neural Networks (ANNs) trained with Backpropagation (BP) show astounding performance and are increasingly often used in performing our daily life tasks. However, ANNs are highly vulnerable to adversarial attacks, which alter inputs with small targeted perturbations that drastically disrupt the models' performance. The most effective method to make ANNs robust against these attacks is adversarial training, in which the training dataset is augmented with exemplary adversarial samples. Unfortunately, this approach has the drawback of increased training complexity since generating adversarial samples is very computationally demanding. In contrast to ANNs, humans are not susceptible to adversarial attacks. Therefore, in this work, we investigate whether biologically-plausible learning algorithms are more robust against adversarial attacks than BP. In particular, we present an extensive comparative analysis of the adversarial robustness of BP and Present the Error to Perturb the Input To modulate Activity (PEPITA), a recently proposed biologically-plausible learning algorithm, on various computer vision tasks. We observe that PEPITA has higher intrinsic adversarial robustness and, with adversarial training, has a more favourable natural-vs-adversarial performance trade-off as, for the same natural accuracies, PEPITA's adversarial accuracies decrease in average by 0.26% and BP's by 8.05%.

{{</citation>}}


### (17/147) MixQuant: Mixed Precision Quantization with a Bit-width Optimization Search (Eliska Kloberdanz et al., 2023)

{{<citation>}}

Eliska Kloberdanz, Wei Le. (2023)  
**MixQuant: Mixed Precision Quantization with a Bit-width Optimization Search**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2309.17341v1)  

---


**ABSTRACT**  
Quantization is a technique for creating efficient Deep Neural Networks (DNNs), which involves performing computations and storing tensors at lower bit-widths than f32 floating point precision. Quantization reduces model size and inference latency, and therefore allows for DNNs to be deployed on platforms with constrained computational resources and real-time systems. However, quantization can lead to numerical instability caused by roundoff error which leads to inaccurate computations and therefore, a decrease in quantized model accuracy. Similarly to prior works, which have shown that both biases and activations are more sensitive to quantization and are best kept in full precision or quantized with higher bit-widths, we show that some weights are more sensitive than others which should be reflected on their quantization bit-width. To that end we propose MixQuant, a search algorithm that finds the optimal custom quantization bit-width for each layer weight based on roundoff error and can be combined with any quantization method as a form of pre-processing optimization. We show that combining MixQuant with BRECQ, a state-of-the-art quantization method, yields better quantized model accuracy than BRECQ alone. Additionally, we combine MixQuant with vanilla asymmetric quantization to show that MixQuant has the potential to optimize the performance of any quantization technique.

{{</citation>}}


### (18/147) Scaling Experiments in Self-Supervised Cross-Table Representation Learning (Maximilian Schambach et al., 2023)

{{<citation>}}

Maximilian Schambach, Dominique Paul, Johannes S. Otterbach. (2023)  
**Scaling Experiments in Self-Supervised Cross-Table Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning, Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2309.17339v1)  

---


**ABSTRACT**  
To analyze the scaling potential of deep tabular representation learning models, we introduce a novel Transformer-based architecture specifically tailored to tabular data and cross-table representation learning by utilizing table-specific tokenizers and a shared Transformer backbone. Our training approach encompasses both single-table and cross-table models, trained via missing value imputation through a self-supervised masked cell recovery objective. To understand the scaling behavior of our method, we train models of varying sizes, ranging from approximately $10^4$ to $10^7$ parameters. These models are trained on a carefully curated pretraining dataset, consisting of 135M training tokens sourced from 76 diverse datasets. We assess the scaling of our architecture in both single-table and cross-table pretraining setups by evaluating the pretrained models using linear probing on a curated set of benchmark datasets and comparing the results with conventional baselines.

{{</citation>}}


### (19/147) PB-LLM: Partially Binarized Large Language Models (Yuzhang Shang et al., 2023)

{{<citation>}}

Yuzhang Shang, Zhihang Yuan, Qiang Wu, Zhen Dong. (2023)  
**PB-LLM: Partially Binarized Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GPT, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.00034v1)  

---


**ABSTRACT**  
This paper explores network binarization, a radical form of quantization, compressing model weights to a single bit, specifically for Large Language Models (LLMs) compression. Due to previous binarization methods collapsing LLMs, we propose a novel approach, Partially-Binarized LLM (PB-LLM), which can achieve extreme low-bit quantization while maintaining the linguistic reasoning capacity of quantized LLMs. Specifically, our exploration first uncovers the ineffectiveness of naive applications of existing binarization algorithms and highlights the imperative role of salient weights in achieving low-bit quantization. Thus, PB-LLM filters a small ratio of salient weights during binarization, allocating them to higher-bit storage, i.e., partially-binarization. PB-LLM is extended to recover the capacities of quantized LMMs, by analyzing from the perspective of post-training quantization (PTQ) and quantization-aware training (QAT). Under PTQ, combining the concepts from GPTQ, we reconstruct the binarized weight matrix guided by the Hessian matrix and successfully recover the reasoning capacity of PB-LLM in low-bit. Under QAT, we freeze the salient weights during training, explore the derivation of optimal scaling factors crucial for minimizing the quantization error, and propose a scaling mechanism based on this derived scaling strategy for residual binarized weights. Those explorations and the developed methodologies significantly contribute to rejuvenating the performance of low-bit quantized LLMs and present substantial advancements in the field of network binarization for LLMs.The code is available at https://github.com/hahnyuan/BinaryLLM.

{{</citation>}}


### (20/147) MuSe-GNN: Learning Unified Gene Representation From Multimodal Biological Graph Data (Tianyu Liu et al., 2023)

{{<citation>}}

Tianyu Liu, Yuge Wang, Rex Ying, Hongyu Zhao. (2023)  
**MuSe-GNN: Learning Unified Gene Representation From Multimodal Biological Graph Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-GN  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.02275v1)  

---


**ABSTRACT**  
Discovering genes with similar functions across diverse biomedical contexts poses a significant challenge in gene representation learning due to data heterogeneity. In this study, we resolve this problem by introducing a novel model called Multimodal Similarity Learning Graph Neural Network, which combines Multimodal Machine Learning and Deep Graph Neural Networks to learn gene representations from single-cell sequencing and spatial transcriptomic data. Leveraging 82 training datasets from 10 tissues, three sequencing techniques, and three species, we create informative graph structures for model training and gene representations generation, while incorporating regularization with weighted similarity learning and contrastive learning to learn cross-data gene-gene relationships. This novel design ensures that we can offer gene representations containing functional similarity across different contexts in a joint space. Comprehensive benchmarking analysis shows our model's capacity to effectively capture gene function similarity across multiple modalities, outperforming state-of-the-art methods in gene representation learning by up to 97.5%. Moreover, we employ bioinformatics tools in conjunction with gene representations to uncover pathway enrichment, regulation causal networks, and functions of disease-associated or dosage-sensitive genes. Therefore, our model efficiently produces unified gene representations for the analysis of gene functions, tissue functions, diseases, and species evolution.

{{</citation>}}


### (21/147) Training and inference of large language models using 8-bit floating point (Sergio P. Perez et al., 2023)

{{<citation>}}

Sergio P. Perez, Yan Zhang, James Briggs, Charlie Blake, Josh Levy-Kramer, Paul Balanca, Carlo Luschi, Stephen Barlow, Andrew William Fitzgibbon. (2023)  
**Training and inference of large language models using 8-bit floating point**  

---
Primary Category: cs.LG  
Categories: I-2-7; B-2-4, cs-AR, cs-CL, cs-ET, cs-LG, cs-PF, cs.LG  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.17224v1)  

---


**ABSTRACT**  
FP8 formats are gaining popularity to boost the computational efficiency for training and inference of large deep learning models. Their main challenge is that a careful choice of scaling is needed to prevent degradation due to the reduced dynamic range compared to higher-precision formats. Although there exists ample literature about selecting such scalings for INT formats, this critical aspect has yet to be addressed for FP8. This paper presents a methodology to select the scalings for FP8 linear layers, based on dynamically updating per-tensor scales for the weights, gradients and activations. We apply this methodology to train and validate large language models of the type of GPT and Llama 2 using FP8, for model sizes ranging from 111M to 70B. To facilitate the understanding of the FP8 dynamics, our results are accompanied by plots of the per-tensor scale distribution for weights, activations and gradients during both training and inference.

{{</citation>}}


### (22/147) Memory Gym: Partially Observable Challenges to Memory-Based Agents in Endless Episodes (Marco Pleines et al., 2023)

{{<citation>}}

Marco Pleines, Matthias Pallasch, Frank Zimmer, Mike Preuss. (2023)  
**Memory Gym: Partially Observable Challenges to Memory-Based Agents in Endless Episodes**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2309.17207v1)  

---


**ABSTRACT**  
Memory Gym introduces a unique benchmark designed to test Deep Reinforcement Learning agents, specifically comparing Gated Recurrent Unit (GRU) against Transformer-XL (TrXL), on their ability to memorize long sequences, withstand noise, and generalize. It features partially observable 2D environments with discrete controls, namely Mortar Mayhem, Mystery Path, and Searing Spotlights. These originally finite environments are extrapolated to novel endless tasks that act as an automatic curriculum, drawing inspiration from the car game ``I packed my bag". These endless tasks are not only beneficial for evaluating efficiency but also intriguingly valuable for assessing the effectiveness of approaches in memory-based agents. Given the scarcity of publicly available memory baselines, we contribute an implementation driven by TrXL and Proximal Policy Optimization. This implementation leverages TrXL as episodic memory using a sliding window approach. In our experiments on the finite environments, TrXL demonstrates superior sample efficiency in Mystery Path and outperforms in Mortar Mayhem. However, GRU is more efficient on Searing Spotlights. Most notably, in all endless tasks, GRU makes a remarkable resurgence, consistently outperforming TrXL by significant margins.

{{</citation>}}


### (23/147) An Investigation Into Race Bias in Random Forest Models Based on Breast DCE-MRI Derived Radiomics Features (Mohamed Huti et al., 2023)

{{<citation>}}

Mohamed Huti, Tiarna Lee, Elinor Sawyer, Andrew P. King. (2023)  
**An Investigation Into Race Bias in Random Forest Models Based on Breast DCE-MRI Derived Radiomics Features**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2309.17197v1)  

---


**ABSTRACT**  
Recent research has shown that artificial intelligence (AI) models can exhibit bias in performance when trained using data that are imbalanced by protected attribute(s). Most work to date has focused on deep learning models, but classical AI techniques that make use of hand-crafted features may also be susceptible to such bias. In this paper we investigate the potential for race bias in random forest (RF) models trained using radiomics features. Our application is prediction of tumour molecular subtype from dynamic contrast enhanced magnetic resonance imaging (DCE-MRI) of breast cancer patients. Our results show that radiomics features derived from DCE-MRI data do contain race-identifiable information, and that RF models can be trained to predict White and Black race from these data with 60-70% accuracy, depending on the subset of features used. Furthermore, RF models trained to predict tumour molecular subtype using race-imbalanced data seem to produce biased behaviour, exhibiting better performance on test data from the race on which they were trained.

{{</citation>}}


### (24/147) RECOMBINER: Robust and Enhanced Compression with Bayesian Implicit Neural Representations (Jiajun He et al., 2023)

{{<citation>}}

Jiajun He, Gergely Flamich, Zongyu Guo, José Miguel Hernández-Lobato. (2023)  
**RECOMBINER: Robust and Enhanced Compression with Bayesian Implicit Neural Representations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2309.17182v1)  

---


**ABSTRACT**  
COMpression with Bayesian Implicit NEural Representations (COMBINER) is a recent data compression method that addresses a key inefficiency of previous Implicit Neural Representation (INR)-based approaches: it avoids quantization and enables direct optimization of the rate-distortion performance. However, COMBINER still has significant limitations: 1) it uses factorized priors and posterior approximations that lack flexibility; 2) it cannot effectively adapt to local deviations from global patterns in the data; and 3) its performance can be susceptible to modeling choices and the variational parameters' initializations. Our proposed method, Robust and Enhanced COMBINER (RECOMBINER), addresses these issues by 1) enriching the variational approximation while maintaining its computational cost via a linear reparameterization of the INR weights, 2) augmenting our INRs with learnable positional encodings that enable them to adapt to local details and 3) splitting high-resolution data into patches to increase robustness and utilizing expressive hierarchical priors to capture dependency across patches. We conduct extensive experiments across several data modalities, showcasing that RECOMBINER achieves competitive results with the best INR-based methods and even outperforms autoencoder-based codecs on low-resolution images at low bitrates.

{{</citation>}}


### (25/147) Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training (Xidong Feng et al., 2023)

{{<citation>}}

Xidong Feng, Ziyu Wan, Muning Wen, Ying Wen, Weinan Zhang, Jun Wang. (2023)  
**Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.17179v1)  

---


**ABSTRACT**  
Large language models (LLMs) typically employ sampling or beam search, accompanied by prompts such as Chain-of-Thought (CoT), to boost reasoning and decoding ability. Recent work like Tree-of-Thought (ToT) and Reasoning via Planning (RAP) aim to augment the reasoning capabilities of LLMs by utilizing tree-search algorithms to guide multi-step reasoning. These methods mainly focus on LLMs' reasoning ability during inference and heavily rely on human-designed prompts to activate LLM as a value function, which lacks general applicability and scalability. To address these limitations, we present an AlphaZero-like tree-search framework for LLMs (termed TS-LLM), systematically illustrating how tree-search with a learned value function can guide LLMs' decoding ability. TS-LLM distinguishes itself in two key ways: (1) Leveraging a learned value function, our approach can be generally applied to different tasks beyond reasoning (such as RLHF alignment), and LLMs of any size, without prompting advanced, large-scale models. (2) It can guide LLM's decoding during both inference and training. Empirical evaluations across reasoning, planning, and RLHF alignment tasks validate the effectiveness of TS-LLM, even on trees with a depth of 64.

{{</citation>}}


### (26/147) Efficient Interpretable Nonlinear Modeling for Multiple Time Series (Kevin Roy et al., 2023)

{{<citation>}}

Kevin Roy, Luis Miguel Lopez-Ramos, Baltasar Beferull-Lozano. (2023)  
**Efficient Interpretable Nonlinear Modeling for Multiple Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2309.17154v1)  

---


**ABSTRACT**  
Predictive linear and nonlinear models based on kernel machines or deep neural networks have been used to discover dependencies among time series. This paper proposes an efficient nonlinear modeling approach for multiple time series, with a complexity comparable to linear vector autoregressive (VAR) models while still incorporating nonlinear interactions among different time-series variables. The modeling assumption is that the set of time series is generated in two steps: first, a linear VAR process in a latent space, and second, a set of invertible and Lipschitz continuous nonlinear mappings that are applied per sensor, that is, a component-wise mapping from each latent variable to a variable in the measurement space. The VAR coefficient identification provides a topology representation of the dependencies among the aforementioned variables. The proposed approach models each component-wise nonlinearity using an invertible neural network and imposes sparsity on the VAR coefficients to reflect the parsimonious dependencies usually found in real applications. To efficiently solve the formulated optimization problems, a custom algorithm is devised combining proximal gradient descent, stochastic primal-dual updates, and projection to enforce the corresponding constraints. Experimental results on both synthetic and real data sets show that the proposed algorithm improves the identification of the support of the VAR coefficients in a parsimonious manner while also improving the time-series prediction, as compared to the current state-of-the-art methods.

{{</citation>}}


### (27/147) Style Transfer for Non-differentiable Audio Effects (Kieran Grant, 2023)

{{<citation>}}

Kieran Grant. (2023)  
**Style Transfer for Non-differentiable Audio Effects**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SD, cs.LG, eess-AS  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2309.17125v1)  

---


**ABSTRACT**  
Digital audio effects are widely used by audio engineers to alter the acoustic and temporal qualities of audio data. However, these effects can have a large number of parameters which can make them difficult to learn for beginners and hamper creativity for professionals. Recently, there have been a number of efforts to employ progress in deep learning to acquire the low-level parameter configurations of audio effects by minimising an objective function between an input and reference track, commonly referred to as style transfer. However, current approaches use inflexible black-box techniques or require that the effects under consideration are implemented in an auto-differentiation framework. In this work, we propose a deep learning approach to audio production style matching which can be used with effects implemented in some of the most widely used frameworks, requiring only that the parameters under consideration have a continuous domain. Further, our method includes style matching for various classes of effects, many of which are difficult or impossible to be approximated closely using differentiable functions. We show that our audio embedding approach creates logical encodings of timbral information, which can be used for a number of downstream tasks. Further, we perform a listening test which demonstrates that our approach is able to convincingly style match a multi-band compressor effect.

{{</citation>}}


### (28/147) Meta-Path Learning for Multi-relational Graph Neural Networks (Francesco Ferrini et al., 2023)

{{<citation>}}

Francesco Ferrini, Antonio Longa, Andrea Passerini, Manfred Jaeger. (2023)  
**Meta-Path Learning for Multi-relational Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.17113v1)  

---


**ABSTRACT**  
Existing multi-relational graph neural networks use one of two strategies for identifying informative relations: either they reduce this problem to low-level weight learning, or they rely on handcrafted chains of relational dependencies, called meta-paths. However, the former approach faces challenges in the presence of many relations (e.g., knowledge graphs), while the latter requires substantial domain expertise to identify relevant meta-paths. In this work we propose a novel approach to learn meta-paths and meta-path GNNs that are highly accurate based on a small number of informative meta-paths. Key element of our approach is a scoring function for measuring the potential informativeness of a relation in the incremental construction of the meta-path. Our experimental evaluation shows that the approach manages to correctly identify relevant meta-paths even with a large number of relations, and substantially outperforms existing multi-relational GNNs on synthetic and real-world experiments.

{{</citation>}}


### (29/147) Dynamic Interpretability for Model Comparison via Decision Rules (Adam Rida et al., 2023)

{{<citation>}}

Adam Rida, Marie-Jeanne Lesot, Xavier Renard, Christophe Marsala. (2023)  
**Dynamic Interpretability for Model Comparison via Decision Rules**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.17095v1)  

---


**ABSTRACT**  
Explainable AI (XAI) methods have mostly been built to investigate and shed light on single machine learning models and are not designed to capture and explain differences between multiple models effectively. This paper addresses the challenge of understanding and explaining differences between machine learning models, which is crucial for model selection, monitoring and lifecycle management in real-world applications. We propose DeltaXplainer, a model-agnostic method for generating rule-based explanations describing the differences between two binary classifiers. To assess the effectiveness of DeltaXplainer, we conduct experiments on synthetic and real-world datasets, covering various model comparison scenarios involving different types of concept drift.

{{</citation>}}


### (30/147) On the Power of the Weisfeiler-Leman Test for Graph Motif Parameters (Matthias Lanzinger et al., 2023)

{{<citation>}}

Matthias Lanzinger, Pablo Barceló. (2023)  
**On the Power of the Weisfeiler-Leman Test for Graph Motif Parameters**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.17053v2)  

---


**ABSTRACT**  
Seminal research in the field of graph neural networks (GNNs) has revealed a direct correspondence between the expressive capabilities of GNNs and the $k$-dimensional Weisfeiler-Leman ($k$WL) test, a widely-recognized method for verifying graph isomorphism. This connection has reignited interest in comprehending the specific graph properties effectively distinguishable by the $k$WL test. A central focus of research in this field revolves around determining the least dimensionality $k$, for which $k$WL can discern graphs with different number of occurrences of a pattern graph $P$. We refer to such a least $k$ as the WL-dimension of this pattern counting problem. This inquiry traditionally delves into two distinct counting problems related to patterns: subgraph counting and induced subgraph counting. Intriguingly, despite their initial appearance as separate challenges with seemingly divergent approaches, both of these problems are interconnected components of a more comprehensive problem: "graph motif parameters". In this paper, we provide a precise characterization of the WL-dimension of labeled graph motif parameters. As specific instances of this result, we obtain characterizations of the WL-dimension of the subgraph counting and induced subgraph counting problem for every labeled pattern $P$. We additionally demonstrate that in cases where the $k$WL test distinguishes between graphs with varying occurrences of a pattern $P$, the exact number of occurrences of $P$ can be computed uniformly using only local information of the last layer of a corresponding GNN. We finally delve into the challenge of recognizing the WL-dimension of various graph parameters. We give a polynomial time algorithm for determining the WL-dimension of the subgraph counting problem for given pattern $P$, answering an open question from previous work.

{{</citation>}}


### (31/147) Deep Representation Learning for Prediction of Temporal Event Sets in the Continuous Time Domain (Parag Dutta et al., 2023)

{{<citation>}}

Parag Dutta, Kawin Mayilvaghanan, Pratyaksha Sinha, Ambedkar Dukkipati. (2023)  
**Deep Representation Learning for Prediction of Temporal Event Sets in the Continuous Time Domain**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.17009v1)  

---


**ABSTRACT**  
Temporal Point Processes (TPP) play an important role in predicting or forecasting events. Although these problems have been studied extensively, predicting multiple simultaneously occurring events can be challenging. For instance, more often than not, a patient gets admitted to a hospital with multiple conditions at a time. Similarly people buy more than one stock and multiple news breaks out at the same time. Moreover, these events do not occur at discrete time intervals, and forecasting event sets in the continuous time domain remains an open problem. Naive approaches for extending the existing TPP models for solving this problem lead to dealing with an exponentially large number of events or ignoring set dependencies among events. In this work, we propose a scalable and efficient approach based on TPPs to solve this problem. Our proposed approach incorporates contextual event embeddings, temporal information, and domain features to model the temporal event sets. We demonstrate the effectiveness of our approach through extensive experiments on multiple datasets, showing that our model outperforms existing methods in terms of prediction metrics and computational efficiency. To the best of our knowledge, this is the first work that solves the problem of predicting event set intensities in the continuous time domain by using TPPs.

{{</citation>}}


### (32/147) Understanding and Mitigating the Label Noise in Pre-training on Downstream Tasks (Hao Chen et al., 2023)

{{<citation>}}

Hao Chen, Jindong Wang, Ankit Shah, Ran Tao, Hongxin Wei, Xing Xie, Masashi Sugiyama, Bhiksha Raj. (2023)  
**Understanding and Mitigating the Label Noise in Pre-training on Downstream Tasks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.17002v1)  

---


**ABSTRACT**  
Pre-training on large-scale datasets and then fine-tuning on downstream tasks have become a standard practice in deep learning. However, pre-training data often contain label noise that may adversely affect the generalization of the model. This paper aims to understand the nature of noise in pre-training datasets and to mitigate its impact on downstream tasks. More specifically, through extensive experiments of supervised pre-training models on synthetic noisy ImageNet-1K and YFCC15M datasets, we demonstrate that while slight noise in pre-training can benefit in-domain (ID) transfer performance, where the training and testing data share the same distribution, it always deteriorates out-of-domain (OOD) performance, where training and testing data distribution are different. We empirically verify that the reason behind is noise in pre-training shapes the feature space differently. We then propose a lightweight black-box tuning method (NMTune) to affine the feature space to mitigate the malignant effect of noise and improve generalization on both ID and OOD tasks, considering one may not be able to fully fine-tune or even access the pre-trained models. We conduct practical experiments on popular vision and language models that are pre-trained on noisy data for evaluation of our approach. Our analysis and results show the importance of this interesting and novel research direction, which we term Noisy Model Learning.

{{</citation>}}


### (33/147) Consistency Models as a Rich and Efficient Policy Class for Reinforcement Learning (Zihan Ding et al., 2023)

{{<citation>}}

Zihan Ding, Chi Jin. (2023)  
**Consistency Models as a Rich and Efficient Policy Class for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16984v1)  

---


**ABSTRACT**  
Score-based generative models like the diffusion model have been testified to be effective in modeling multi-modal data from image generation to reinforcement learning (RL). However, the inference process of diffusion model can be slow, which hinders its usage in RL with iterative sampling. We propose to apply the consistency model as an efficient yet expressive policy representation, namely consistency policy, with an actor-critic style algorithm for three typical RL settings: offline, offline-to-online and online. For offline RL, we demonstrate the expressiveness of generative models as policies from multi-modal data. For offline-to-online RL, the consistency policy is shown to be more computational efficient than diffusion policy, with a comparable performance. For online RL, the consistency policy demonstrates significant speedup and even higher average performances than the diffusion policy.

{{</citation>}}


### (34/147) Benchmarking and In-depth Performance Study of Large Language Models on Habana Gaudi Processors (Chengming Zhang et al., 2023)

{{<citation>}}

Chengming Zhang, Baixi Sun, Xiaodong Yu, Zhen Xie, Weijian Zheng, Kamil Iskra, Pete Beckman, Dingwen Tao. (2023)  
**Benchmarking and In-depth Performance Study of Large Language Models on Habana Gaudi Processors**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: AI, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.16976v1)  

---


**ABSTRACT**  
Transformer models have achieved remarkable success in various machine learning tasks but suffer from high computational complexity and resource requirements. The quadratic complexity of the self-attention mechanism further exacerbates these challenges when dealing with long sequences and large datasets. Specialized AI hardware accelerators, such as the Habana GAUDI architecture, offer a promising solution to tackle these issues. GAUDI features a Matrix Multiplication Engine (MME) and a cluster of fully programmable Tensor Processing Cores (TPC). This paper explores the untapped potential of using GAUDI processors to accelerate Transformer-based models, addressing key challenges in the process. Firstly, we provide a comprehensive performance comparison between the MME and TPC components, illuminating their relative strengths and weaknesses. Secondly, we explore strategies to optimize MME and TPC utilization, offering practical insights to enhance computational efficiency. Thirdly, we evaluate the performance of Transformers on GAUDI, particularly in handling long sequences and uncovering performance bottlenecks. Lastly, we evaluate the end-to-end performance of two Transformer-based large language models (LLM) on GAUDI. The contributions of this work encompass practical insights for practitioners and researchers alike. We delve into GAUDI's capabilities for Transformers through systematic profiling, analysis, and optimization exploration. Our study bridges a research gap and offers a roadmap for optimizing Transformer-based model training on the GAUDI architecture.

{{</citation>}}


### (35/147) Towards Robust Offline-to-Online Reinforcement Learning via Uncertainty and Smoothness (Xiaoyu Wen et al., 2023)

{{<citation>}}

Xiaoyu Wen, Xudong Yu, Rui Yang, Chenjia Bai, Zhen Wang. (2023)  
**Towards Robust Offline-to-Online Reinforcement Learning via Uncertainty and Smoothness**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16973v1)  

---


**ABSTRACT**  
To obtain a near-optimal policy with fewer interactions in Reinforcement Learning (RL), a promising approach involves the combination of offline RL, which enhances sample efficiency by leveraging offline datasets, and online RL, which explores informative transitions by interacting with the environment. Offline-to-Online (O2O) RL provides a paradigm for improving an offline trained agent within limited online interactions. However, due to the significant distribution shift between online experiences and offline data, most offline RL algorithms suffer from performance drops and fail to achieve stable policy improvement in O2O adaptation. To address this problem, we propose the Robust Offline-to-Online (RO2O) algorithm, designed to enhance offline policies through uncertainty and smoothness, and to mitigate the performance drop in online adaptation. Specifically, RO2O incorporates Q-ensemble for uncertainty penalty and adversarial samples for policy and value smoothness, which enable RO2O to maintain a consistent learning procedure in online adaptation without requiring special changes to the learning objective. Theoretical analyses in linear MDPs demonstrate that the uncertainty and smoothness lead to a tighter optimality bound in O2O against distribution shift. Experimental results illustrate the superiority of RO2O in facilitating stable offline-to-online learning and achieving significant improvement with limited online interactions.

{{</citation>}}


### (36/147) Multi-Resolution Active Learning of Fourier Neural Operators (Shibo Li et al., 2023)

{{<citation>}}

Shibo Li, Xin Yu, Wei Xing, Mike Kirby, Akil Narayan, Shandian Zhe. (2023)  
**Multi-Resolution Active Learning of Fourier Neural Operators**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2309.16971v2)  

---


**ABSTRACT**  
Fourier Neural Operator (FNO) is a popular operator learning framework, which not only achieves the state-of-the-art performance in many tasks, but also is highly efficient in training and prediction. However, collecting training data for the FNO is a costly bottleneck in practice, because it often demands expensive physical simulations. To overcome this problem, we propose Multi-Resolution Active learning of FNO (MRA-FNO), which can dynamically select the input functions and resolutions to lower the data cost as much as possible while optimizing the learning efficiency. Specifically, we propose a probabilistic multi-resolution FNO and use ensemble Monte-Carlo to develop an effective posterior inference algorithm. To conduct active learning, we maximize a utility-cost ratio as the acquisition function to acquire new examples and resolutions at each step. We use moment matching and the matrix determinant lemma to enable tractable, efficient utility computation. Furthermore, we develop a cost annealing framework to avoid over-penalizing high-resolution queries at the early stage. The over-penalization is severe when the cost difference is significant between the resolutions, which renders active learning often stuck at low-resolution queries and inferior performance. Our method overcomes this problem and applies to general multi-fidelity active learning and optimization problems. We have shown the advantage of our method in several benchmark operator learning tasks.

{{</citation>}}


### (37/147) Physics-Informed Induction Machine Modelling (Qing Shen et al., 2023)

{{<citation>}}

Qing Shen, Yifan Zhou, Peng Zhang. (2023)  
**Physics-Informed Induction Machine Modelling**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.16943v1)  

---


**ABSTRACT**  
This rapid communication devises a Neural Induction Machine (NeuIM) model, which pilots the use of physics-informed machine learning to enable AI-based electromagnetic transient simulations. The contributions are threefold: (1) a formation of NeuIM to represent the induction machine in phase domain; (2) a physics-informed neural network capable of capturing fast and slow IM dynamics even in the absence of data; and (3) a data-physics-integrated hybrid NeuIM approach which is adaptive to various levels of data availability. Extensive case studies validate the efficacy of NeuIM and in particular, its advantage over purely data-driven approaches.

{{</citation>}}


### (38/147) G4SATBench: Benchmarking and Advancing SAT Solving with Graph Neural Networks (Zhaoyu Li et al., 2023)

{{<citation>}}

Zhaoyu Li, Jinpei Guo, Xujie Si. (2023)  
**G4SATBench: Benchmarking and Advancing SAT Solving with Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.16941v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have recently emerged as a promising approach for solving the Boolean Satisfiability Problem (SAT), offering potential alternatives to traditional backtracking or local search SAT solvers. However, despite the growing volume of literature in this field, there remains a notable absence of a unified dataset and a fair benchmark to evaluate and compare existing approaches. To address this crucial gap, we present G4SATBench, the first benchmark study that establishes a comprehensive evaluation framework for GNN-based SAT solvers. In G4SATBench, we meticulously curate a large and diverse set of SAT datasets comprising 7 problems with 3 difficulty levels and benchmark a broad range of GNN models across various prediction tasks, training objectives, and inference algorithms. To explore the learning abilities and comprehend the strengths and limitations of GNN-based SAT solvers, we also compare their solving processes with the heuristics in search-based SAT solvers. Our empirical results provide valuable insights into the performance of GNN-based SAT solvers and further suggest that existing GNN models can effectively learn a solving strategy akin to greedy local search but struggle to learn backtracking search in the latent space.

{{</citation>}}


### (39/147) TranDRL: A Transformer-Driven Deep Reinforcement Learning Enabled Prescriptive Maintenance Framework (Yang Zhao et al., 2023)

{{<citation>}}

Yang Zhao, Wenbo Wang. (2023)  
**TranDRL: A Transformer-Driven Deep Reinforcement Learning Enabled Prescriptive Maintenance Framework**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2309.16935v1)  

---


**ABSTRACT**  
Industrial systems demand reliable predictive maintenance strategies to enhance operational efficiency and reduce downtime. This paper introduces a novel, integrated framework that leverages the power of transformer neural networks and deep reinforcement learning (DRL) algorithms to optimize maintenance actions. Our approach employs the transformer model to effectively capture complex temporal patterns in sensor data, thereby accurately predicting the Remaining Useful Life (RUL) of equipment. Simultaneously, the DRL component of our framework provides cost-effective and timely maintenance recommendations. We validate the efficacy of our framework on the NASA C-MPASS dataset, where it demonstrates significant advancements in both RUL prediction accuracy and the optimization of maintenance actions. Consequently, our pioneering approach provides an innovative data-driven methodology for prescriptive maintenance, addressing key challenges in industrial operations and leading the way to more efficient, cost-effective, and reliable systems.

{{</citation>}}


### (40/147) Learning to Receive Help: Intervention-Aware Concept Embedding Models (Mateo Espinosa Zarlenga et al., 2023)

{{<citation>}}

Mateo Espinosa Zarlenga, Katherine M. Collins, Krishnamurthy Dvijotham, Adrian Weller, Zohreh Shams, Mateja Jamnik. (2023)  
**Learning to Receive Help: Intervention-Aware Concept Embedding Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.16928v1)  

---


**ABSTRACT**  
Concept Bottleneck Models (CBMs) tackle the opacity of neural architectures by constructing and explaining their predictions using a set of high-level concepts. A special property of these models is that they permit concept interventions, wherein users can correct mispredicted concepts and thus improve the model's performance. Recent work, however, has shown that intervention efficacy can be highly dependent on the order in which concepts are intervened on and on the model's architecture and training hyperparameters. We argue that this is rooted in a CBM's lack of train-time incentives for the model to be appropriately receptive to concept interventions. To address this, we propose Intervention-aware Concept Embedding models (IntCEMs), a novel CBM-based architecture and training paradigm that improves a model's receptiveness to test-time interventions. Our model learns a concept intervention policy in an end-to-end fashion from where it can sample meaningful intervention trajectories at train-time. This conditions IntCEMs to effectively select and receive concept interventions when deployed at test-time. Our experiments show that IntCEMs significantly outperform state-of-the-art concept-interpretable models when provided with test-time concept interventions, demonstrating the effectiveness of our approach.

{{</citation>}}


### (41/147) ACGAN-GNNExplainer: Auxiliary Conditional Generative Explainer for Graph Neural Networks (Yiqiao Li et al., 2023)

{{<citation>}}

Yiqiao Li, Jianlong Zhou, Yifei Dong, Niusha Shafiabady, Fang Chen. (2023)  
**ACGAN-GNNExplainer: Auxiliary Conditional Generative Explainer for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.16918v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have proven their efficacy in a variety of real-world applications, but their underlying mechanisms remain a mystery. To address this challenge and enable reliable decision-making, many GNN explainers have been proposed in recent years. However, these methods often encounter limitations, including their dependence on specific instances, lack of generalizability to unseen graphs, producing potentially invalid explanations, and yielding inadequate fidelity. To overcome these limitations, we, in this paper, introduce the Auxiliary Classifier Generative Adversarial Network (ACGAN) into the field of GNN explanation and propose a new GNN explainer dubbed~\emph{ACGAN-GNNExplainer}. Our approach leverages a generator to produce explanations for the original input graphs while incorporating a discriminator to oversee the generation process, ensuring explanation fidelity and improving accuracy. Experimental evaluations conducted on both synthetic and real-world graph datasets demonstrate the superiority of our proposed method compared to other existing GNN explainers.

{{</citation>}}


## cs.CL (32)



### (42/147) Contextual Biasing with the Knuth-Morris-Pratt Matching Algorithm (Weiran Wang et al., 2023)

{{<citation>}}

Weiran Wang, Zelin Wu, Diamantino Caseiro, Tsendsuren Munkhdalai, Khe Chai Sim, Pat Rondon, Golan Pundak, Gan Song, Rohit Prabhavalkar, Zhong Meng, Ding Zhao, Tara Sainath, Pedro Moreno Mengibar. (2023)  
**Contextual Biasing with the Knuth-Morris-Pratt Matching Algorithm**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.00178v1)  

---


**ABSTRACT**  
Contextual biasing refers to the problem of biasing the automatic speech recognition (ASR) systems towards rare entities that are relevant to the specific user or application scenarios. We propose algorithms for contextual biasing based on the Knuth-Morris-Pratt algorithm for pattern matching. During beam search, we boost the score of a token extension if it extends matching into a set of biasing phrases. Our method simulates the classical approaches often implemented in the weighted finite state transducer (WFST) framework, but avoids the FST language altogether, with careful considerations on memory footprint and efficiency on tensor processing units (TPUs) by vectorization. Without introducing additional model parameters, our method achieves significant word error rate (WER) reductions on biasing test sets by itself, and yields further performance gain when combined with a model-based biasing method.

{{</citation>}}


### (43/147) Self-Specialization: Uncovering Latent Expertise within Large Language Models (Junmo Kang et al., 2023)

{{<citation>}}

Junmo Kang, Hongyin Luo, Yada Zhu, James Glass, David Cox, Alan Ritter, Rogerio Feris, Leonid Karlinsky. (2023)  
**Self-Specialization: Uncovering Latent Expertise within Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.00160v1)  

---


**ABSTRACT**  
Recent works have demonstrated the effectiveness of self-alignment in which a large language model is, by itself, aligned to follow general instructions through the automatic generation of instructional data using a handful of human-written seeds. Instead of general alignment, in this work, we focus on self-alignment for expert domain specialization (e.g., biomedicine), discovering it to be very effective for improving zero-shot and few-shot performance in target domains of interest. As a preliminary, we first present the benchmark results of existing aligned models within a specialized domain, which reveals the marginal effect that "generic" instruction-following training has on downstream expert domains' performance. To remedy this, we explore self-specialization that leverages domain-specific unlabelled data and a few labeled seeds for the self-alignment process. When augmented with retrieval to reduce hallucination and enhance concurrency of the alignment, self-specialization offers an effective (and efficient) way of "carving out" an expert model out of a "generalist", pre-trained LLM where different domains of expertise are originally combined in a form of "superposition". Our experimental results on a biomedical domain show that our self-specialized model (30B) outperforms its base model, MPT-30B by a large margin and even surpasses larger popular models based on LLaMA-65B, highlighting its potential and practicality for specialization, especially considering its efficiency in terms of data and parameters.

{{</citation>}}


### (44/147) Automatic Prompt Rewriting for Personalized Text Generation (Cheng Li et al., 2023)

{{<citation>}}

Cheng Li, Mingyang Zhang, Qiaozhu Mei, Weize Kong, Michael Bendersky. (2023)  
**Automatic Prompt Rewriting for Personalized Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2310.00152v1)  

---


**ABSTRACT**  
Facilitated by large language models (LLMs), personalized text generation has become a rapidly growing research direction. Most existing studies focus on designing specialized models for a particular domain, or they require fine-tuning the LLMs to generate personalized text. We consider a typical scenario in which the large language model, which generates personalized output, is frozen and can only be accessed through APIs. Under this constraint, all one can do is to improve the input text (i.e., text prompts) sent to the LLM, a procedure that is usually done manually. In this paper, we propose a novel method to automatically revise prompts for personalized text generation. The proposed method takes the initial prompts generated by a state-of-the-art, multistage framework for personalized generation and rewrites a few critical components that summarize and synthesize the personal context. The prompt rewriter employs a training paradigm that chains together supervised learning (SL) and reinforcement learning (RL), where SL reduces the search space of RL and RL facilitates end-to-end training of the rewriter. Using datasets from three representative domains, we demonstrate that the rewritten prompts outperform both the original prompts and the prompts optimized via supervised learning or reinforcement learning alone. In-depth analysis of the rewritten prompts shows that they are not only human readable, but also able to guide manual revision of prompts when there is limited resource to employ reinforcement learning to train the prompt rewriter, or when it is costly to deploy an automatic prompt rewriter for inference.

{{</citation>}}


### (45/147) Multilingual Natural Language ProcessingModel for Radiology Reports -- The Summary is all you need! (Mariana Lindo et al., 2023)

{{<citation>}}

Mariana Lindo, Ana Sofia Santos, André Ferreira, Jianning Li, Gijs Luijten, Gustavo Correia, Moon Kim, Jens Kleesiek, Jan Egger, Victor Alves. (2023)  
**Multilingual Natural Language ProcessingModel for Radiology Reports -- The Summary is all you need!**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Multilingual, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2310.00100v1)  

---


**ABSTRACT**  
The impression section of a radiology report summarizes important radiology findings and plays a critical role in communicating these findings to physicians. However, the preparation of these summaries is time-consuming and error-prone for radiologists. Recently, numerous models for radiology report summarization have been developed. Nevertheless, there is currently no model that can summarize these reports in multiple languages. Such a model could greatly improve future research and the development of Deep Learning models that incorporate data from patients with different ethnic backgrounds. In this study, the generation of radiology impressions in different languages was automated by fine-tuning a model, publicly available, based on a multilingual text-to-text Transformer to summarize findings available in English, Portuguese, and German radiology reports. In a blind test, two board-certified radiologists indicated that for at least 70% of the system-generated summaries, the quality matched or exceeded the corresponding human-written summaries, suggesting substantial clinical reliability. Furthermore, this study showed that the multilingual model outperformed other models that specialized in summarizing radiology reports in only one language, as well as models that were not specifically designed for summarizing radiology reports, such as ChatGPT.

{{</citation>}}


### (46/147) Voice2Action: Language Models as Agent for Efficient Real-Time Interaction in Virtual Reality (Yang Su, 2023)

{{<citation>}}

Yang Su. (2023)  
**Voice2Action: Language Models as Agent for Efficient Real-Time Interaction in Virtual Reality**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.00092v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are trained and aligned to follow natural language instructions with only a handful of examples, and they are prompted as task-driven autonomous agents to adapt to various sources of execution environments. However, deploying agent LLMs in virtual reality (VR) has been challenging due to the lack of efficiency in online interactions and the complex manipulation categories in 3D environments. In this work, we propose Voice2Action, a framework that hierarchically analyzes customized voice signals and textual commands through action and entity extraction and divides the execution tasks into canonical interaction subsets in real-time with error prevention from environment feedback. Experiment results in an urban engineering VR environment with synthetic instruction data show that Voice2Action can perform more efficiently and accurately than approaches without optimizations.

{{</citation>}}


### (47/147) SocREval: Large Language Models with the Socratic Method for Reference-Free Reasoning Evaluation (Hangfeng He et al., 2023)

{{<citation>}}

Hangfeng He, Hongming Zhang, Dan Roth. (2023)  
**SocREval: Large Language Models with the Socratic Method for Reference-Free Reasoning Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.00074v1)  

---


**ABSTRACT**  
To comprehensively assess the capacity of current models for complex reasoning, it is crucial to assess their step-by-step reasoning in a scalable manner. Established reference-based evaluation metrics rely on human-annotated reasoning chains to assess the model-derived chains. However, such ``gold-standard'' human-written reasoning chains may not be unique and their acquisition is often labor-intensive. Existing reference-free reasoning metrics eliminate the need for human-crafted reasoning chains as references, but they typically require fine-tuning on datasets with human-derived reasoning chains, which complicates the process and raises concerns regarding generalizability across diverse datasets. To address these challenges, we harness GPT-4 to automatically evaluate reasoning chain quality, obviating the need for human-crafted references. Leveraging the Socratic method, we devise tailored prompts to enhance reference-free reasoning evaluation, which we term SocREval (Socratic method for Reasoning Evaluation). Empirical results from four human annotated datasets reveal that SocREval significantly improves GPT-4's performance, surpassing existing reference-free and reference-based reasoning evaluation metrics. Beyond its demonstrated efficacy, our proposed framework, large language models (LLMs) with the Socratic method, proves to be both cost-efficient and robust to prompt writing and example selection, as substantiated by our in-depth analysis.

{{</citation>}}


### (48/147) Efficient Streaming Language Models with Attention Sinks (Guangxuan Xiao et al., 2023)

{{<citation>}}

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. (2023)  
**Efficient Streaming Language Models with Attention Sinks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, Falcon, Language Model  
[Paper Link](http://arxiv.org/abs/2309.17453v1)  

---


**ABSTRACT**  
Deploying Large Language Models (LLMs) in streaming applications such as multi-round dialogue, where long interactions are expected, is urgently needed but poses two major challenges. Firstly, during the decoding stage, caching previous tokens' Key and Value states (KV) consumes extensive memory. Secondly, popular LLMs cannot generalize to longer texts than the training sequence length. Window attention, where only the most recent KVs are cached, is a natural approach -- but we show that it fails when the text length surpasses the cache size. We observe an interesting phenomenon, namely attention sink, that keeping the KV of initial tokens will largely recover the performance of window attention. In this paper, we first demonstrate that the emergence of attention sink is due to the strong attention scores towards initial tokens as a ``sink'' even if they are not semantically important. Based on the above analysis, we introduce StreamingLLM, an efficient framework that enables LLMs trained with a finite length attention window to generalize to infinite sequence lengths without any fine-tuning. We show that StreamingLLM can enable Llama-2, MPT, Falcon, and Pythia to perform stable and efficient language modeling with up to 4 million tokens and more. In addition, we discover that adding a placeholder token as a dedicated attention sink during pre-training can further improve streaming deployment. In streaming settings, StreamingLLM outperforms the sliding window recomputation baseline by up to 22.2x speedup. Code and datasets are provided at https://github.com/mit-han-lab/streaming-llm.

{{</citation>}}


### (49/147) ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving (Zhibin Gou et al., 2023)

{{<citation>}}

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen. (2023)  
**ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.17452v2)  

---


**ABSTRACT**  
Large language models have made significant progress in various language tasks, yet they still struggle with complex mathematics. In this paper, we propose ToRA a series of Tool-integrated Reasoning Agents designed to solve challenging mathematical problems by seamlessly integrating natural language reasoning with the utilization of external tools (e.g., computation libraries and symbolic solvers), thereby amalgamating the analytical prowess of language and the computational efficiency of tools. To train ToRA, we curate interactive tool-use trajectories on mathematical datasets, apply imitation learning on the annotations, and propose output space shaping to further refine models' reasoning behavior. As a result, ToRA models significantly outperform open-source models on 10 mathematical reasoning datasets across all scales with 13%-19% absolute improvements on average. Notably, ToRA-7B reaches 44.6% on the competition-level dataset MATH, surpassing the best open-source model WizardMath-70B by 22% absolute. ToRA-Code-34B is also the first open-source model that achieves an accuracy exceeding 50% on MATH, which significantly outperforms GPT-4's CoT result, and is competitive with GPT-4 solving problems with programs. Additionally, we conduct a comprehensive analysis of the benefits and remaining challenges of tool interaction for mathematical reasoning, providing valuable insights for future research.

{{</citation>}}


### (50/147) A Large Language Model Approach to Educational Survey Feedback Analysis (Michael J. Parker et al., 2023)

{{<citation>}}

Michael J. Parker, Caitlin Anderson, Claire Stone, YeaRim Oh. (2023)  
**A Large Language Model Approach to Educational Survey Feedback Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.17447v1)  

---


**ABSTRACT**  
This paper assesses the potential for the large language models (LLMs) GPT-4 and GPT-3.5 to aid in deriving insight from education feedback surveys. Exploration of LLM use cases in education has focused on teaching and learning, with less exploration of capabilities in education feedback analysis. Survey analysis in education involves goals such as finding gaps in curricula or evaluating teachers, often requiring time-consuming manual processing of textual responses. LLMs have the potential to provide a flexible means of achieving these goals without specialized machine learning models or fine-tuning. We demonstrate a versatile approach to such goals by treating them as sequences of natural language processing (NLP) tasks including classification (multi-label, multi-class, and binary), extraction, thematic analysis, and sentiment analysis, each performed by LLM. We apply these workflows to a real-world dataset of 2500 end-of-course survey comments from biomedical science courses, and evaluate a zero-shot approach (i.e., requiring no examples or labeled training data) across all tasks, reflecting education settings, where labeled data is often scarce. By applying effective prompting practices, we achieve human-level performance on multiple tasks with GPT-4, enabling workflows necessary to achieve typical goals. We also show the potential of inspecting LLMs' chain-of-thought (CoT) reasoning for providing insight that may foster confidence in practice. Moreover, this study features development of a versatile set of classification categories, suitable for various course types (online, hybrid, or in-person) and amenable to customization. Our results suggest that LLMs can be used to derive a range of insights from survey text.

{{</citation>}}


### (51/147) L2CEval: Evaluating Language-to-Code Generation Capabilities of Large Language Models (Ansong Ni et al., 2023)

{{<citation>}}

Ansong Ni, Pengcheng Yin, Yilun Zhao, Martin Riddell, Troy Feng, Rui Shen, Stephen Yin, Ye Liu, Semih Yavuz, Caiming Xiong, Shafiq Joty, Yingbo Zhou, Dragomir Radev, Arman Cohan. (2023)  
**L2CEval: Evaluating Language-to-Code Generation Capabilities of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-PL, cs-SE, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.17446v2)  

---


**ABSTRACT**  
Recently, large language models (LLMs), especially those that are pretrained on code, have demonstrated strong capabilities in generating programs from natural language inputs in a few-shot or even zero-shot manner. Despite promising results, there is a notable lack of a comprehensive evaluation of these models language-to-code generation capabilities. Existing studies often focus on specific tasks, model architectures, or learning paradigms, leading to a fragmented understanding of the overall landscape. In this work, we present L2CEval, a systematic evaluation of the language-to-code generation capabilities of LLMs on 7 tasks across the domain spectrum of semantic parsing, math reasoning and Python programming, analyzing the factors that potentially affect their performance, such as model size, pretraining data, instruction tuning, and different prompting methods. In addition to assessing model performance, we measure confidence calibration for the models and conduct human evaluations of the output programs. This enables us to identify and analyze the typical failure modes across various tasks and models. L2CEval offers a comprehensive understanding of the capabilities and limitations of LLMs in language-to-code generation. We also release the evaluation framework and all model outputs, hoping to lay the groundwork for further future research in this domain.

{{</citation>}}


### (52/147) CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets (Lifan Yuan et al., 2023)

{{<citation>}}

Lifan Yuan, Yangyi Chen, Xingyao Wang, Yi R. Fung, Hao Peng, Heng Ji. (2023)  
**CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.17428v1)  

---


**ABSTRACT**  
Large language models (LLMs) are often augmented with tools to solve complex tasks. By generating code snippets and executing them through task-specific Application Programming Interfaces (APIs), they can offload certain functions to dedicated external modules, such as image encoding and performing calculations. However, most existing approaches to augment LLMs with tools are constrained by general-purpose APIs and lack the flexibility for tailoring them to specific tasks. In this work, we present CRAFT, a general tool creation and retrieval framework for LLMs. It creates toolsets specifically curated for the tasks and equips LLMs with a component that retrieves tools from these sets to enhance their capability to solve complex tasks. For each task, we collect specific code solutions by prompting GPT-4 to solve the training examples. Following a validation step ensuring the correctness, these solutions are abstracted into code snippets to enhance reusability, and deduplicated for higher quality. At inference time, the language model retrieves snippets from the toolsets and then executes them or generates the output conditioning on the retrieved snippets. Our method is designed to be flexible and offers a plug-and-play approach to adapt off-the-shelf LLMs to unseen domains and modalities, without any finetuning. Experiments on vision-language, tabular processing, and mathematical reasoning tasks show that our approach achieves substantial improvements compared to strong baselines. In addition, our in-depth analysis reveals that: (1) consistent performance improvement can be achieved by scaling up the number of tools and the capability of the backbone models; (2) each component of our approach contributes to the performance gains; (3) the created tools are well-structured and reliable with low complexity and atomicity. The code is available at \url{https://github.com/lifan-yuan/CRAFT}.

{{</citation>}}


### (53/147) Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks (Vaidehi Patil et al., 2023)

{{<citation>}}

Vaidehi Patil, Peter Hase, Mohit Bansal. (2023)  
**Can Sensitive Information Be Deleted From LLMs? Objectives for Defending Against Extraction Attacks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.17410v1)  

---


**ABSTRACT**  
Pretrained language models sometimes possess knowledge that we do not wish them to, including memorized personal information and knowledge that could be used to harm people. They can also output toxic or harmful text. To mitigate these safety and informational issues, we propose an attack-and-defense framework for studying the task of deleting sensitive information directly from model weights. We study direct edits to model weights because (1) this approach should guarantee that particular deleted information is never extracted by future prompt attacks, and (2) it should protect against whitebox attacks, which is necessary for making claims about safety/privacy in a setting where publicly available model weights could be used to elicit sensitive information. Our threat model assumes that an attack succeeds if the answer to a sensitive question is located among a set of B generated candidates, based on scenarios where the information would be insecure if the answer is among B candidates. Experimentally, we show that even state-of-the-art model editing methods such as ROME struggle to truly delete factual information from models like GPT-J, as our whitebox and blackbox attacks can recover "deleted" information from an edited model 38% of the time. These attacks leverage two key observations: (1) that traces of deleted information can be found in intermediate model hidden states, and (2) that applying an editing method for one question may not delete information across rephrased versions of the question. Finally, we provide new defense methods that protect against some extraction attacks, but we do not find a single universally effective defense method. Our results suggest that truly deleting sensitive information is a tractable but difficult problem, since even relatively low attack success rates have potentially severe societal implications for real-world deployment of language models.

{{</citation>}}


### (54/147) Revolutionizing Mobile Interaction: Enabling a 3 Billion Parameter GPT LLM on Mobile (Samuel Carreira et al., 2023)

{{<citation>}}

Samuel Carreira, Tomás Marques, José Ribeiro, Carlos Grilo. (2023)  
**Revolutionizing Mobile Interaction: Enabling a 3 Billion Parameter GPT LLM on Mobile**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.01434v1)  

---


**ABSTRACT**  
The field of Artificial Intelligence has witnessed remarkable progress in recent years, especially with the emergence of powerful large language models (LLMs) based on the transformer architecture. Cloud-based LLMs, such as OpenAI's ChatGPT, offer impressive capabilities but come with concerns regarding latency and privacy due to network dependencies. This article presents an innovative approach to LLM inference, envisioning a future where LLMs with billions of parameters can be executed directly on mobile devices without network connectivity. The article showcases a fine-tuned GPT LLM with 3 billion parameters that can operate smoothly on devices with as low as 4GB of memory. Through the integration of native code and model quantization techniques, the application not only serves as a general-purpose assistant but also facilitates seamless mobile interactions with text-to-actions features. The article provides insights into the training pipeline, implementation details, test results, and future directions of on-device LLM inference. This breakthrough technology opens up possibilities for empowering users with sophisticated AI capabilities while preserving their privacy and eliminating latency concerns.

{{</citation>}}


### (55/147) Overview of the BioLaySumm 2023 Shared Task on Lay Summarization of Biomedical Research Articles (Tomsa Goldsack et al., 2023)

{{<citation>}}

Tomsa Goldsack, Zheheng Luo, Qianqian Xie, Carolina Scarton, Matthew Shardlow, Sophia Ananiadou, Chenghua Lin. (2023)  
**Overview of the BioLaySumm 2023 Shared Task on Lay Summarization of Biomedical Research Articles**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Summarization  
[Paper Link](http://arxiv.org/abs/2309.17332v1)  

---


**ABSTRACT**  
This paper presents the results of the shared task on Lay Summarisation of Biomedical Research Articles (BioLaySumm), hosted at the BioNLP Workshop at ACL 2023. The goal of this shared task is to develop abstractive summarisation models capable of generating "lay summaries" (i.e., summaries that are comprehensible to non-technical audiences) in both a controllable and non-controllable setting. There are two subtasks: 1) Lay Summarisation, where the goal is for participants to build models for lay summary generation only, given the full article text and the corresponding abstract as input; and 2) Readability-controlled Summarisation, where the goal is for participants to train models to generate both the technical abstract and the lay summary, given an article's main text as input. In addition to overall results, we report on the setup and insights from the BioLaySumm shared task, which attracted a total of 20 participating teams across both subtasks.

{{</citation>}}


### (56/147) Few-Shot Domain Adaptation for Charge Prediction on Unprofessional Descriptions (Jie Zhao et al., 2023)

{{<citation>}}

Jie Zhao, Ziyu Guan, Wei Zhao, Yue Jiang, Xiaofei He. (2023)  
**Few-Shot Domain Adaptation for Charge Prediction on Unprofessional Descriptions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot, Legal  
[Paper Link](http://arxiv.org/abs/2309.17313v1)  

---


**ABSTRACT**  
Recent works considering professional legal-linguistic style (PLLS) texts have shown promising results on the charge prediction task. However, unprofessional users also show an increasing demand on such a prediction service. There is a clear domain discrepancy between PLLS texts and non-PLLS texts expressed by those laypersons, which degrades the current SOTA models' performance on non-PLLS texts. A key challenge is the scarcity of non-PLLS data for most charge classes. This paper proposes a novel few-shot domain adaptation (FSDA) method named Disentangled Legal Content for Charge Prediction (DLCCP). Compared with existing FSDA works, which solely perform instance-level alignment without considering the negative impact of text style information existing in latent features, DLCCP (1) disentangles the content and style representations for better domain-invariant legal content learning with carefully designed optimization goals for content and style spaces and, (2) employs the constitutive elements knowledge of charges to extract and align element-level and instance-level content representations simultaneously. We contribute the first publicly available non-PLLS dataset named NCCP for developing layperson-friendly charge prediction models. Experiments on NCCP show the superiority of our methods over competitive baselines.

{{</citation>}}


### (57/147) Split and Merge: Aligning Position Biases in Large Language Model based Evaluators (Zongjie Li et al., 2023)

{{<citation>}}

Zongjie Li, Chaozheng Wang, Pingchuan Ma, Daoyuan Wu, Tianxiang Li, Shuai Wang, Cuiyun Gao, Yang Liu. (2023)  
**Split and Merge: Aligning Position Biases in Large Language Model based Evaluators**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Bias, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.01432v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown promise as automated evaluators for assessing the quality of answers generated by AI systems. However, these LLM-based evaluators exhibit position bias, or inconsistency, when used to evaluate candidate answers in pairwise comparisons, favoring either the first or second answer regardless of content. To address this limitation, we propose PORTIA, an alignment-based system designed to mimic human comparison strategies to calibrate position bias in a lightweight yet effective manner. Specifically, PORTIA splits the answers into multiple segments, aligns similar content across candidate answers, and then merges them back into a single prompt for evaluation by LLMs. We conducted extensive experiments with six diverse LLMs to evaluate 11,520 answer pairs. Our results show that PORTIA markedly enhances the consistency rates for all the models and comparison forms tested, achieving an average relative improvement of 47.46%. Remarkably, PORTIA enables less advanced GPT models to achieve 88% agreement with the state-of-the-art GPT-4 model at just 10% of the cost. Furthermore, it rectifies around 80% of the position bias instances within the GPT-4 model, elevating its consistency rate up to 98%. Subsequent human evaluations indicate that the PORTIA-enhanced GPT-3.5 model can even surpass the standalone GPT-4 in terms of alignment with human evaluators. These findings highlight PORTIA's ability to correct position bias, improve LLM consistency, and boost performance while keeping cost-efficiency. This represents a valuable step toward a more reliable and scalable use of LLMs for automated evaluations across diverse applications.

{{</citation>}}


### (58/147) STRONG -- Structure Controllable Legal Opinion Summary Generation (Yang Zhong et al., 2023)

{{<citation>}}

Yang Zhong, Diane Litman. (2023)  
**STRONG -- Structure Controllable Legal Opinion Summary Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Legal  
[Paper Link](http://arxiv.org/abs/2309.17280v1)  

---


**ABSTRACT**  
We propose an approach for the structure controllable summarization of long legal opinions that considers the argument structure of the document. Our approach involves using predicted argument role information to guide the model in generating coherent summaries that follow a provided structure pattern. We demonstrate the effectiveness of our approach on a dataset of legal opinions and show that it outperforms several strong baselines with respect to ROUGE, BERTScore, and structure similarity.

{{</citation>}}


### (59/147) Enhancing Large Language Models in Coding Through Multi-Perspective Self-Consistency (Baizhou Huang et al., 2023)

{{<citation>}}

Baizhou Huang, Shuai Lu, Weizhu Chen, Xiaojun Wan, Nan Duan. (2023)  
**Enhancing Large Language Models in Coding Through Multi-Perspective Self-Consistency**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SE, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.17272v1)  

---


**ABSTRACT**  
Large language models (LLMs) have exhibited remarkable ability in textual generation. However, in complex reasoning tasks such as code generation, generating the correct answer in a single attempt remains a formidable challenge for LLMs. Previous research has explored solutions by aggregating multiple outputs, leveraging the consistency among them. However, none of them have comprehensively captured this consistency from different perspectives. In this paper, we propose the Multi-Perspective Self-Consistency (MPSC) framework, a novel decoding strategy for LLM that incorporates both inter-consistency across outputs from multiple perspectives and intra-consistency within a single perspective. Specifically, we ask LLMs to sample multiple diverse outputs from various perspectives for a given query and then construct a multipartite graph based on them. With two predefined measures of consistency, we embed both inter- and intra-consistency information into the graph. The optimal choice is then determined based on consistency analysis in the graph. We conduct comprehensive evaluation on the code generation task by introducing solution, specification and test case as three perspectives. We leverage a code interpreter to quantitatively measure the inter-consistency and propose several intra-consistency measure functions. Our MPSC framework significantly boosts the performance on various popular benchmarks, including HumanEval (+17.60%), HumanEval Plus (+17.61%), MBPP (+6.50%) and CodeContests (+11.82%) in Pass@1, when compared to original outputs generated from ChatGPT, and even surpassing GPT-4.

{{</citation>}}


### (60/147) Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering (Han Zhou et al., 2023)

{{<citation>}}

Han Zhou, Xingchen Wan, Lev Proleev, Diana Mincu, Jilin Chen, Katherine Heller, Subhrajit Roy. (2023)  
**Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: PaLM  
[Paper Link](http://arxiv.org/abs/2309.17249v1)  

---


**ABSTRACT**  
Prompting and in-context learning (ICL) have become efficient learning paradigms for large language models (LLMs). However, LLMs suffer from prompt brittleness and various bias factors in the prompt, including but not limited to the formatting, the choice verbalizers, and the ICL examples. To address this problem that results in unexpected performance degradation, calibration methods have been developed to mitigate the effects of these biases while recovering LLM performance. In this work, we first conduct a systematic analysis of the existing calibration methods, where we both provide a unified view and reveal the failure cases. Inspired by these analyses, we propose Batch Calibration (BC), a simple yet intuitive method that controls the contextual bias from the batched input, unifies various prior approaches, and effectively addresses the aforementioned issues. BC is zero-shot, inference-only, and incurs negligible additional costs. In the few-shot setup, we further extend BC to allow it to learn the contextual bias from labeled data. We validate the effectiveness of BC with PaLM 2-(S, M, L) and CLIP models and demonstrate state-of-the-art performance over previous calibration baselines across more than 10 natural language understanding and image classification tasks.

{{</citation>}}


### (61/147) LLM-Deliberation: Evaluating LLMs with Interactive Multi-Agent Negotiation Games (Sahar Abdelnabi et al., 2023)

{{<citation>}}

Sahar Abdelnabi, Amr Gomaa, Sarath Sivaprasad, Lea Schönherr, Mario Fritz. (2023)  
**LLM-Deliberation: Evaluating LLMs with Interactive Multi-Agent Negotiation Games**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.17234v1)  

---


**ABSTRACT**  
There is a growing interest in using Large Language Models (LLMs) as agents to tackle real-world tasks that may require assessing complex situations. Yet, we have a limited understanding of LLMs' reasoning and decision-making capabilities, partly stemming from a lack of dedicated evaluation benchmarks. As negotiating and compromising are key aspects of our everyday communication and collaboration, we propose using scorable negotiation games as a new evaluation framework for LLMs. We create a testbed of diverse text-based, multi-agent, multi-issue, semantically rich negotiation games, with easily tunable difficulty. To solve the challenge, agents need to have strong arithmetic, inference, exploration, and planning capabilities, while seamlessly integrating them. Via a systematic zero-shot Chain-of-Thought prompting (CoT), we show that agents can negotiate and consistently reach successful deals. We quantify the performance with multiple metrics and observe a large gap between GPT-4 and earlier models. Importantly, we test the generalization to new games and setups. Finally, we show that these games can help evaluate other critical aspects, such as the interaction dynamics between agents in the presence of greedy and adversarial players.

{{</citation>}}


### (62/147) Comparative Analysis of Named Entity Recognition in the Dungeons and Dragons Domain (Gayashan Weerasundara et al., 2023)

{{<citation>}}

Gayashan Weerasundara, Nisansa de Silva. (2023)  
**Comparative Analysis of Named Entity Recognition in the Dungeons and Dragons Domain**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model, NER, NLP, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2309.17171v1)  

---


**ABSTRACT**  
Many NLP tasks, although well-resolved for general English, face challenges in specific domains like fantasy literature. This is evident in Named Entity Recognition (NER), which detects and categorizes entities in text. We analyzed 10 NER models on 7 Dungeons and Dragons (D&D) adventure books to assess domain-specific performance. Using open-source Large Language Models, we annotated named entities in these books and evaluated each model's precision. Our findings indicate that, without modifications, Flair, Trankit, and Spacy outperform others in identifying named entities in the D&D context.

{{</citation>}}


### (63/147) An evaluation of GPT models for phenotype concept recognition (Tudor Groza et al., 2023)

{{<citation>}}

Tudor Groza, Harry Caufield, Dylan Gration, Gareth Baynam, Melissa A Haendel, Peter N Robinson, Chris J Mungall, Justin T Reese. (2023)  
**An evaluation of GPT models for phenotype concept recognition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, Clinical, GPT, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2309.17169v1)  

---


**ABSTRACT**  
Objective: Clinical deep phenotyping plays a critical role in both the diagnosis of patients with rare disorders as well as in building care coordination plans. The process relies on modelling and curating patient profiles using ontology concepts, usually from the Human Phenotype Ontology. Machine learning methods have been widely adopted to support this phenotype concept recognition task. With the significant shift in the use of large language models (LLMs) for most NLP tasks, herewithin, we examine the performance of the latest Generative Pre-trained Transformer (GPT) models underpinning ChatGPT in clinical deep phenotyping. Materials and Methods: The experimental setup of the study included seven prompts of various levels of specificity, two GPT models (gpt-3.5 and gpt-4.0) and an established gold standard for phenotype recognition. Results: Our results show that, currently, these models have not yet achieved state of the art performance. The best run, using few-shots learning, achieved 0.41 F1 score, compared to a 0.62 F1 score achieved by the current best in class tool. Conclusion: The non-deterministic nature of the outcomes and the lack of concordance between different runs using the same prompt and input makes the use of these LLMs in clinical settings problematic.

{{</citation>}}


### (64/147) LatticeGen: A Cooperative Framework which Hides Generated Text in a Lattice for Privacy-Aware Generation on Cloud (Mengke Zhang et al., 2023)

{{<citation>}}

Mengke Zhang, Tianxing He, Tianle Wang, Lu Mi, Fatemehsadat Mireshghallah, Binyi Chen, Hao Wang, Yulia Tsvetkov. (2023)  
**LatticeGen: A Cooperative Framework which Hides Generated Text in a Lattice for Privacy-Aware Generation on Cloud**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.17157v2)  

---


**ABSTRACT**  
In the current user-server interaction paradigm of prompted generation with large language models (LLM) on cloud, the server fully controls the generation process, which leaves zero options for users who want to keep the generated text to themselves. We propose LatticeGen, a cooperative framework in which the server still handles most of the computation while the user controls the sampling operation. The key idea is that the true generated sequence is mixed with noise tokens by the user and hidden in a noised lattice. Considering potential attacks from a hypothetically malicious server and how the user can defend against it, we propose the repeated beam-search attack and the mixing noise scheme. In our experiments we apply LatticeGen to protect both prompt and generation. It is shown that while the noised lattice degrades generation quality, LatticeGen successfully protects the true generation to a remarkable degree under strong attacks (more than 50% of the semantic remains hidden as measured by BERTScore).

{{</citation>}}


### (65/147) Using Large Language Models for Qualitative Analysis can Introduce Serious Bias (Julian Ashwin et al., 2023)

{{<citation>}}

Julian Ashwin, Aditya Chhabra, Vijayendra Rao. (2023)  
**Using Large Language Models for Qualitative Analysis can Introduce Serious Bias**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL, econ-GN, q-fin-EC  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2309.17147v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are quickly becoming ubiquitous, but the implications for social science research are not yet well understood. This paper asks whether LLMs can help us analyse large-N qualitative data from open-ended interviews, with an application to transcripts of interviews with Rohingya refugees in Cox's Bazaar, Bangladesh. We find that a great deal of caution is needed in using LLMs to annotate text as there is a risk of introducing biases that can lead to misleading inferences. We here mean bias in the technical sense, that the errors that LLMs make in annotating interview transcripts are not random with respect to the characteristics of the interview subjects. Training simpler supervised models on high-quality human annotations with flexible coding leads to less measurement error and bias than LLM annotations. Therefore, given that some high quality annotations are necessary in order to asses whether an LLM introduces bias, we argue that it is probably preferable to train a bespoke model on these annotations than it is to use an LLM for annotation.

{{</citation>}}


### (66/147) Promoting Generalized Cross-lingual Question Answering in Few-resource Scenarios via Self-knowledge Distillation (Casimiro Pio Carrino et al., 2023)

{{<citation>}}

Casimiro Pio Carrino, Carlos Escolano, José A. R. Fonollosa. (2023)  
**Promoting Generalized Cross-lingual Question Answering in Few-resource Scenarios via Self-knowledge Distillation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.17134v1)  

---


**ABSTRACT**  
Despite substantial progress in multilingual extractive Question Answering (QA), models with high and uniformly distributed performance across languages remain challenging, especially for languages with limited resources. We study cross-lingual transfer mainly focusing on the Generalized Cross-Lingual Transfer (G-XLT) task, where the question language differs from the context language - a challenge that has received limited attention thus far. Our approach seeks to enhance cross-lingual QA transfer using a high-performing multilingual model trained on a large-scale dataset, complemented by a few thousand aligned QA examples across languages. Our proposed strategy combines cross-lingual sampling and advanced self-distillation training in generations to tackle the previous challenge. Notably, we introduce the novel mAP@k coefficients to fine-tune self-knowledge distillation loss, dynamically regulating the teacher's model knowledge to perform a balanced and effective knowledge transfer. We extensively evaluate our approach to assess XLT and G-XLT capabilities in extractive QA. Results reveal that our self-knowledge distillation approach outperforms standard cross-entropy fine-tuning by a significant margin. Importantly, when compared to a strong baseline that leverages a sizeable volume of machine-translated data, our approach shows competitive results despite the considerable challenge of operating within resource-constrained settings, even in zero-shot scenarios. Beyond performance improvements, we offer valuable insights through comprehensive analyses and an ablation study, further substantiating the benefits and constraints of our approach. In essence, we propose a practical solution to improve cross-lingual QA transfer by leveraging a few data resources in an efficient way.

{{</citation>}}


### (67/147) Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering (Weizhe Lin et al., 2023)

{{<citation>}}

Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexandru Coca, Bill Byrne. (2023)  
**Fine-grained Late-interaction Multi-modal Retrieval for Retrieval Augmented Visual Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.17133v1)  

---


**ABSTRACT**  
Knowledge-based Visual Question Answering (KB-VQA) requires VQA systems to utilize knowledge from existing knowledge bases to answer visually-grounded questions. Retrieval-Augmented Visual Question Answering (RA-VQA), a strong framework to tackle KB-VQA, first retrieves related documents with Dense Passage Retrieval (DPR) and then uses them to answer questions. This paper proposes Fine-grained Late-interaction Multi-modal Retrieval (FLMR) which significantly improves knowledge retrieval in RA-VQA. FLMR addresses two major limitations in RA-VQA's retriever: (1) the image representations obtained via image-to-text transforms can be incomplete and inaccurate and (2) relevance scores between queries and documents are computed with one-dimensional embeddings, which can be insensitive to finer-grained relevance. FLMR overcomes these limitations by obtaining image representations that complement those from the image-to-text transforms using a vision model aligned with an existing text-based retriever through a simple alignment network. FLMR also encodes images and questions using multi-dimensional embeddings to capture finer-grained relevance between queries and documents. FLMR significantly improves the original RA-VQA retriever's PRRecall@5 by approximately 8\%. Finally, we equipped RA-VQA with two state-of-the-art large multi-modal/language models to achieve $\sim61\%$ VQA score in the OK-VQA dataset.

{{</citation>}}


### (68/147) SCALE: Synergized Collaboration of Asymmetric Language Translation Engines (Xin Cheng et al., 2023)

{{<citation>}}

Xin Cheng, Xun Wang, Tao Ge, Si-Qing Chen, Furu Wei, Dongyan Zhao, Rui Yan. (2023)  
**SCALE: Synergized Collaboration of Asymmetric Language Translation Engines**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLEU, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.17061v1)  

---


**ABSTRACT**  
In this paper, we introduce SCALE, a collaborative framework that connects compact Specialized Translation Models (STMs) and general-purpose Large Language Models (LLMs) as one unified translation engine. By introducing translation from STM into the triplet in-context demonstrations, SCALE unlocks refinement and pivoting ability of LLM, thus mitigating language bias of LLM and parallel data bias of STM, enhancing LLM speciality without sacrificing generality, and facilitating continual learning without expensive LLM fine-tuning. Our comprehensive experiments show that SCALE significantly outperforms both few-shot LLMs (GPT-4) and specialized models (NLLB) in challenging low-resource settings. Moreover, in Xhosa to English translation, SCALE experiences consistent improvement by a 4 BLEURT score without tuning LLM and surpasses few-shot GPT-4 by 2.5 COMET score and 3.8 BLEURT score when equipped with a compact model consisting of merely 600M parameters. SCALE could also effectively exploit the existing language bias of LLMs by using an English-centric STM as a pivot for translation between any language pairs, outperforming few-shot GPT-4 by an average of 6 COMET points across eight translation directions. Furthermore we provide an in-depth analysis of SCALE's robustness, translation characteristics, and latency costs, providing solid foundation for future studies exploring the potential synergy between LLMs and more specialized, task-specific models.

{{</citation>}}


### (69/147) Interpretable Long-Form Legal Question Answering with Retrieval-Augmented Large Language Models (Antoine Louis et al., 2023)

{{<citation>}}

Antoine Louis, Gijs van Dijck, Gerasimos Spanakis. (2023)  
**Interpretable Long-Form Legal Question Answering with Retrieval-Augmented Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Legal, NLP, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.17050v1)  

---


**ABSTRACT**  
Many individuals are likely to face a legal dispute at some point in their lives, but their lack of understanding of how to navigate these complex issues often renders them vulnerable. The advancement of natural language processing opens new avenues for bridging this legal literacy gap through the development of automated legal aid systems. However, existing legal question answering (LQA) approaches often suffer from a narrow scope, being either confined to specific legal domains or limited to brief, uninformative responses. In this work, we propose an end-to-end methodology designed to generate long-form answers to any statutory law questions, utilizing a "retrieve-then-read" pipeline. To support this approach, we introduce and release the Long-form Legal Question Answering (LLeQA) dataset, comprising 1,868 expert-annotated legal questions in the French language, complete with detailed answers rooted in pertinent legal provisions. Our experimental results demonstrate promising performance on automatic evaluation metrics, but a qualitative analysis uncovers areas for refinement. As one of the only comprehensive, expert-annotated long-form LQA dataset, LLeQA has the potential to not only accelerate research towards resolving a significant real-world issue, but also act as a rigorous benchmark for evaluating NLP models in specialized domains. We publicly release our code, data, and models.

{{</citation>}}


### (70/147) Sarcasm in Sight and Sound: Benchmarking and Expansion to Improve Multimodal Sarcasm Detection (Swapnil Bhosale et al., 2023)

{{<citation>}}

Swapnil Bhosale, Abhra Chaudhuri, Alex Lee Robert Williams, Divyank Tiwari, Anjan Dutta, Xiatian Zhu, Pushpak Bhattacharyya, Diptesh Kanojia. (2023)  
**Sarcasm in Sight and Sound: Benchmarking and Expansion to Improve Multimodal Sarcasm Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Sarcasm Detection  
[Paper Link](http://arxiv.org/abs/2310.01430v1)  

---


**ABSTRACT**  
The introduction of the MUStARD dataset, and its emotion recognition extension MUStARD++, have identified sarcasm to be a multi-modal phenomenon -- expressed not only in natural language text, but also through manners of speech (like tonality and intonation) and visual cues (facial expression). With this work, we aim to perform a rigorous benchmarking of the MUStARD++ dataset by considering state-of-the-art language, speech, and visual encoders, for fully utilizing the totality of the multi-modal richness that it has to offer, achieving a 2\% improvement in macro-F1 over the existing benchmark. Additionally, to cure the imbalance in the `sarcasm type' category in MUStARD++, we propose an extension, which we call \emph{MUStARD++ Balanced}, benchmarking the same with instances from the extension split across both train and test sets, achieving a further 2.4\% macro-F1 boost. The new clips were taken from a novel source -- the TV show, House MD, which adds to the diversity of the dataset, and were manually annotated by multiple annotators with substantial inter-annotator agreement in terms of Cohen's kappa and Krippendorf's alpha. Our code, extended data, and SOTA benchmark models are made public.

{{</citation>}}


### (71/147) Benchmarking Cognitive Biases in Large Language Models as Evaluators (Ryan Koo et al., 2023)

{{<citation>}}

Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, Zae Myung Kim, Dongyeop Kang. (2023)  
**Benchmarking Cognitive Biases in Large Language Models as Evaluators**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2309.17012v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have recently been shown to be effective as automatic evaluators with simple prompting and in-context learning. In this work, we assemble 15 LLMs of four different size ranges and evaluate their output responses by preference ranking from the other LLMs as evaluators, such as System Star is better than System Square. We then evaluate the quality of ranking outputs introducing the Cognitive Bias Benchmark for LLMs as Evaluators (CoBBLEr), a benchmark to measure six different cognitive biases in LLM evaluation outputs, such as the Egocentric bias where a model prefers to rank its own outputs highly in evaluation. We find that LLMs are biased text quality evaluators, exhibiting strong indications on our bias benchmark (average of 40% of comparisons across all models) within each of their evaluations that question their robustness as evaluators. Furthermore, we examine the correlation between human and machine preferences and calculate the average Rank-Biased Overlap (RBO) score to be 49.6%, indicating that machine preferences are misaligned with humans. According to our findings, LLMs may still be unable to be utilized for automatic annotation aligned with human preferences. Our project page is at: https://minnesotanlp.github.io/cobbler.

{{</citation>}}


### (72/147) I Wish to Have an Argument: Argumentative Reasoning in Large Language Models (Adrian de Wynter et al., 2023)

{{<citation>}}

Adrian de Wynter, Tommy Yuan. (2023)  
**I Wish to Have an Argument: Argumentative Reasoning in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.16938v1)  

---


**ABSTRACT**  
We evaluate the ability of contemporary large language models (LLMs) to perform argumentative reasoning. We frame our experiments in terms of the argument mining (AM) and argument pair extraction (APE) tasks, and evaluate their ability to perform reasoning at increasing levels of abstraction in the input and output representations (e.g., arbitrary label sets, semantic graphs). We find that, although LLMs are able to match or surpass the state-of-the-art in AM and APE, their argumentative reasoning performance is very dependent on the input and output representation. We also find an "exemplar effect", where too many exemplars increasingly become detrimental for task performance, and about 4-5 being the optimal amount. Neither result extends to chain-of-thought (CoT) prompting: we find the exemplar effect to be nullified, and our results suggest that CoT allows for better performance under ill-conditioned problems. We hope that the work reported contributes to the improvement of argumentative reasoning in LLMs.

{{</citation>}}


### (73/147) SSHR: Leveraging Self-supervised Hierarchical Representations for Multilingual Automatic Speech Recognition (Hongfei Xue et al., 2023)

{{<citation>}}

Hongfei Xue, Qijie Shao, Kaixun Huang, Peikun Chen, Lei Xie, Jie Liu. (2023)  
**SSHR: Leveraging Self-supervised Hierarchical Representations for Multilingual Automatic Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Multilingual, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.16937v1)  

---


**ABSTRACT**  
Multilingual automatic speech recognition (ASR) systems have garnered attention for their potential to extend language coverage globally. While self-supervised learning (SSL) has demonstrated its effectiveness in multilingual ASR, it is worth noting that the various layers' representations of SSL potentially contain distinct information that has not been fully leveraged. In this study, we propose a novel method that leverages self-supervised hierarchical representations (SSHR) to fine-tune multilingual ASR. We first analyze the different layers of the SSL model for language-related and content-related information, uncovering layers that show a stronger correlation. Then, we extract a language-related frame from correlated middle layers and guide specific content extraction through self-attention mechanisms. Additionally, we steer the model toward acquiring more content-related information in the final layers using our proposed Cross-CTC. We evaluate SSHR on two multilingual datasets, Common Voice and ML-SUPERB, and the experimental results demonstrate that our method achieves state-of-the-art performance to the best of our knowledge.

{{</citation>}}


## cs.AI (11)



### (74/147) Motif: Intrinsic Motivation from Artificial Intelligence Feedback (Martin Klissarov et al., 2023)

{{<citation>}}

Martin Klissarov, Pierluca D'Oro, Shagun Sodhani, Roberta Raileanu, Pierre-Luc Bacon, Pascal Vincent, Amy Zhang, Mikael Henaff. (2023)  
**Motif: Intrinsic Motivation from Artificial Intelligence Feedback**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.00166v1)  

---


**ABSTRACT**  
Exploring rich environments and evaluating one's actions without prior knowledge is immensely challenging. In this paper, we propose Motif, a general method to interface such prior knowledge from a Large Language Model (LLM) with an agent. Motif is based on the idea of grounding LLMs for decision-making without requiring them to interact with the environment: it elicits preferences from an LLM over pairs of captions to construct an intrinsic reward, which is then used to train agents with reinforcement learning. We evaluate Motif's performance and behavior on the challenging, open-ended and procedurally-generated NetHack game. Surprisingly, by only learning to maximize its intrinsic reward, Motif achieves a higher game score than an algorithm directly trained to maximize the score itself. When combining Motif's intrinsic reward with the environment reward, our method significantly outperforms existing approaches and makes progress on tasks where no advancements have ever been made without demonstrations. Finally, we show that Motif mostly generates intuitive human-aligned behaviors which can be steered easily through prompt modifications, while scaling well with the LLM size and the amount of information given in the prompt.

{{</citation>}}


### (75/147) Data Filtering Networks (Alex Fang et al., 2023)

{{<citation>}}

Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander Toshev, Vaishaal Shankar. (2023)  
**Data Filtering Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2309.17425v2)  

---


**ABSTRACT**  
Large training sets have become a cornerstone of machine learning and are the foundation for recent advances in language modeling and multimodal learning. While data curation for pre-training is often still ad-hoc, one common paradigm is to first collect a massive pool of data from the Web and then filter this candidate pool down to an actual training set via various heuristics. In this work, we study the problem of learning a data filtering network (DFN) for this second step of filtering a large uncurated dataset. Our key finding is that the quality of a network for filtering is distinct from its performance on downstream tasks: for instance, a model that performs well on ImageNet can yield worse training sets than a model with low ImageNet accuracy that is trained on a small amount of high-quality data. Based on our insights, we construct new data filtering networks that induce state-of-the-art image-text datasets. Specifically, our best performing dataset DFN-5B enables us to train state-of-the-art models for their compute budgets: among other improvements on a variety of tasks, a ViT-H trained on our dataset achieves 83.0% zero-shot transfer accuracy on ImageNet, out-performing models trained on other datasets such as LAION-2B, DataComp-1B, or OpenAI's WIT. In order to facilitate further research in dataset design, we also release a new 2 billion example dataset DFN-2B and show that high performance data filtering networks can be trained from scratch using only publicly available data.

{{</citation>}}


### (76/147) Building Privacy-Preserving and Secure Geospatial Artificial Intelligence Foundation Models (Jinmeng Rao et al., 2023)

{{<citation>}}

Jinmeng Rao, Song Gao, Gengchen Mai, Krzysztof Janowicz. (2023)  
**Building Privacy-Preserving and Secure Geospatial Artificial Intelligence Foundation Models**  

---
Primary Category: cs.AI  
Categories: I-2-0, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.17319v1)  

---


**ABSTRACT**  
In recent years we have seen substantial advances in foundation models for artificial intelligence, including language, vision, and multimodal models. Recent studies have highlighted the potential of using foundation models in geospatial artificial intelligence, known as GeoAI Foundation Models or Geo-Foundation Models, for geographic question answering, remote sensing image understanding, map generation, and location-based services, among others. However, the development and application of GeoAI foundation models can pose serious privacy and security risks, which have not been fully discussed or addressed to date. This paper introduces the potential privacy and security risks throughout the lifecycle of GeoAI foundation models and proposes a comprehensive blueprint for preventative and control strategies. Through this vision paper, we hope to draw the attention of researchers and policymakers in geospatial domains to these privacy and security risks inherent in GeoAI foundation models and advocate for the development of privacy-preserving and secure GeoAI foundation models.

{{</citation>}}


### (77/147) AutoAgents: A Framework for Automatic Agent Generation (Guangyao Chen et al., 2023)

{{<citation>}}

Guangyao Chen, Siwei Dong, Yu Shu, Ge Zhang, Jaward Sesay, Börje F. Karlsson, Jie Fu, Yemin Shi. (2023)  
**AutoAgents: A Framework for Automatic Agent Generation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.17288v1)  

---


**ABSTRACT**  
Large language models (LLMs) have enabled remarkable advances in automated task-solving with multi-agent systems. However, most existing LLM-based multi-agent approaches rely on predefined agents to handle simple tasks, limiting the adaptability of multi-agent collaboration to different scenarios. Therefore, we introduce AutoAgents, an innovative framework that adaptively generates and coordinates multiple specialized agents to build an AI team according to different tasks. Specifically, AutoAgents couples the relationship between tasks and roles by dynamically generating multiple required agents based on task content and planning solutions for the current task based on the generated expert agents. Multiple specialized agents collaborate with each other to efficiently accomplish tasks. Concurrently, an observer role is incorporated into the framework to reflect on the designated plans and agents' responses and improve upon them. Our experiments on various benchmarks demonstrate that AutoAgents generates more coherent and accurate solutions than the existing multi-agent methods. This underscores the significance of assigning different roles to different tasks and of team cooperation, offering new perspectives for tackling complex tasks. The repository of this project is available at https://github.com/LinkSoul-AI/AutoAgents.

{{</citation>}}


### (78/147) Suspicion-Agent: Playing Imperfect Information Games with Theory of Mind Aware GPT4 (Jiaxian Guo et al., 2023)

{{<citation>}}

Jiaxian Guo, Bo Yang, Paul Yoo, Yuchen Lin, Yusuke Iwasawa, Yutaka Matsuo. (2023)  
**Suspicion-Agent: Playing Imperfect Information Games with Theory of Mind Aware GPT4**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.17277v1)  

---


**ABSTRACT**  
Unlike perfect information games, where all elements are known to every player, imperfect information games emulate the real-world complexities of decision-making under uncertain or incomplete information. GPT-4, the recent breakthrough in large language models (LLMs) trained on massive passive data, is notable for its knowledge retrieval and reasoning abilities. This paper delves into the applicability of GPT-4's learned knowledge for imperfect information games. To achieve this, we introduce \textbf{Suspicion-Agent}, an innovative agent that leverages GPT-4's capabilities for performing in imperfect information games. With proper prompt engineering to achieve different functions, Suspicion-Agent based on GPT-4 demonstrates remarkable adaptability across a range of imperfect information card games. Importantly, GPT-4 displays a strong high-order theory of mind (ToM) capacity, meaning it can understand others and intentionally impact others' behavior. Leveraging this, we design a planning strategy that enables GPT-4 to competently play against different opponents, adapting its gameplay style as needed, while requiring only the game rules and descriptions of observations as input. In the experiments, we qualitatively showcase the capabilities of Suspicion-Agent across three different imperfect information games and then quantitatively evaluate it in Leduc Hold'em. The results show that Suspicion-Agent can potentially outperform traditional algorithms designed for imperfect information games, without any specialized training or examples. In order to encourage and foster deeper insights within the community, we make our game-related data publicly available.

{{</citation>}}


### (79/147) Knowledge Graphs for the Life Sciences: Recent Developments, Challenges and Opportunities (Jiaoyan Chen et al., 2023)

{{<citation>}}

Jiaoyan Chen, Hang Dong, Janna Hastings, Ernesto Jiménez-Ruiz, Vanessa Lopez, Pierre Monnin, Catia Pesquita, Petr Škoda, Valentina Tamma. (2023)  
**Knowledge Graphs for the Life Sciences: Recent Developments, Challenges and Opportunities**  

---
Primary Category: cs.AI  
Categories: I-2-4; J-3, cs-AI, cs-CL, cs.AI  
Keywords: AI, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.17255v1)  

---


**ABSTRACT**  
The term life sciences refers to the disciplines that study living organisms and life processes, and include chemistry, biology, medicine, and a range of other related disciplines. Research efforts in life sciences are heavily data-driven, as they produce and consume vast amounts of scientific data, much of which is intrinsically relational and graph-structured.   The volume of data and the complexity of scientific concepts and relations referred to therein promote the application of advanced knowledge-driven technologies for managing and interpreting data, with the ultimate aim to advance scientific discovery.   In this survey and position paper, we discuss recent developments and advances in the use of graph-based technologies in life sciences and set out a vision for how these technologies will impact these fields into the future. We focus on three broad topics: the construction and management of Knowledge Graphs (KGs), the use of KGs and associated technologies in the discovery of new knowledge, and the use of KGs in artificial intelligence applications to support explanations (explainable AI). We select a few exemplary use cases for each topic, discuss the challenges and open research questions within these topics, and conclude with a perspective and outlook that summarizes the overarching challenges and their potential solutions as a guide for future research.

{{</citation>}}


### (80/147) RLAdapter: Bridging Large Language Models to Reinforcement Learning in Open Worlds (Wanpeng Zhang et al., 2023)

{{<citation>}}

Wanpeng Zhang, Zongqing Lu. (2023)  
**RLAdapter: Bridging Large Language Models to Reinforcement Learning in Open Worlds**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.17176v1)  

---


**ABSTRACT**  
While reinforcement learning (RL) shows remarkable success in decision-making problems, it often requires a lot of interactions with the environment, and in sparse-reward environments, it is challenging to learn meaningful policies. Large Language Models (LLMs) can potentially provide valuable guidance to agents in learning policies, thereby enhancing the performance of RL algorithms in such environments. However, LLMs often encounter difficulties in understanding downstream tasks, which hinders their ability to optimally assist agents in these tasks. A common approach to mitigating this issue is to fine-tune the LLMs with task-related data, enabling them to offer useful guidance for RL agents. However, this approach encounters several difficulties, such as inaccessible model weights or the need for significant computational resources, making it impractical. In this work, we introduce RLAdapter, a framework that builds a better connection between RL algorithms and LLMs by incorporating an adapter model. Within the RLAdapter framework, fine-tuning a lightweight language model with information generated during the training process of RL agents significantly aids LLMs in adapting to downstream tasks, thereby providing better guidance for RL agents. We conducted experiments to evaluate RLAdapter in the Crafter environment, and the results show that RLAdapter surpasses the SOTA baselines. Furthermore, agents under our framework exhibit common-sense behaviors that are absent in baseline models.

{{</citation>}}


### (81/147) DyVal: Graph-informed Dynamic Evaluation of Large Language Models (Kaijie Zhu et al., 2023)

{{<citation>}}

Kaijie Zhu, Jiaao Chen, Jindong Wang, Neil Zhenqiang Gong, Diyi Yang, Xing Xie. (2023)  
**DyVal: Graph-informed Dynamic Evaluation of Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: ChatGPT, GPT, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2309.17167v1)  

---


**ABSTRACT**  
Large language models (LLMs) have achieved remarkable performance in various evaluation benchmarks. However, concerns about their performance are raised on potential data contamination in their considerable volume of training corpus. Moreover, the static nature and fixed complexity of current benchmarks may inadequately gauge the advancing capabilities of LLMs. In this paper, we introduce DyVal, a novel, general, and flexible evaluation protocol for dynamic evaluation of LLMs. Based on our proposed dynamic evaluation framework, we build graph-informed DyVal by leveraging the structural advantage of directed acyclic graphs to dynamically generate evaluation samples with controllable complexities. DyVal generates challenging evaluation sets on reasoning tasks including mathematics, logical reasoning, and algorithm problems. We evaluate various LLMs ranging from Flan-T5-large to ChatGPT and GPT4. Experiments demonstrate that LLMs perform worse in DyVal-generated evaluation samples with different complexities, emphasizing the significance of dynamic evaluation. We also analyze the failure cases and results of different prompting methods. Moreover, DyVal-generated samples are not only evaluation sets, but also helpful data for fine-tuning to improve the performance of LLMs on existing benchmarks. We hope that DyVal can shed light on the future evaluation research of LLMs.

{{</citation>}}


### (82/147) Benchmarking the Abilities of Large Language Models for RDF Knowledge Graph Creation and Comprehension: How Well Do LLMs Speak Turtle? (Johannes Frey et al., 2023)

{{<citation>}}

Johannes Frey, Lars-Peter Meyer, Natanael Arndt, Felix Brei, Kirill Bulert. (2023)  
**Benchmarking the Abilities of Large Language Models for RDF Knowledge Graph Creation and Comprehension: How Well Do LLMs Speak Turtle?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-DB, cs.AI  
Keywords: Falcon, GPT, GPT-3.5, GPT-4, Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2309.17122v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are advancing at a rapid pace, with significant improvements at natural language processing and coding tasks. Yet, their ability to work with formal languages representing data, specifically within the realm of knowledge graph engineering, remains under-investigated. To evaluate the proficiency of various LLMs, we created a set of five tasks that probe their ability to parse, understand, analyze, and create knowledge graphs serialized in Turtle syntax. These tasks, each embodying distinct degrees of complexity and being able to scale with the size of the problem, have been integrated into our automated evaluation system, the LLM-KG-Bench. The evaluation encompassed four commercially available LLMs - GPT-3.5, GPT-4, Claude 1.3, and Claude 2.0, as well as two freely accessible offline models, GPT4All Vicuna and GPT4All Falcon 13B. This analysis offers an in-depth understanding of the strengths and shortcomings of LLMs in relation to their application within RDF knowledge graph engineering workflows utilizing Turtle representation. While our findings show that the latest commercial models outperform their forerunners in terms of proficiency with the Turtle language, they also reveal an apparent weakness. These models fall short when it comes to adhering strictly to the output formatting constraints, a crucial requirement in this context.

{{</citation>}}


### (83/147) Tell Me a Story! Narrative-Driven XAI with Large Language Models (David Martens et al., 2023)

{{<citation>}}

David Martens, Camille Dams, James Hinns, Mark Vergouwen. (2023)  
**Tell Me a Story! Narrative-Driven XAI with Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.17057v1)  

---


**ABSTRACT**  
In today's critical domains, the predominance of black-box machine learning models amplifies the demand for Explainable AI (XAI). The widely used SHAP values, while quantifying feature importance, are often too intricate and lack human-friendly explanations. Furthermore, counterfactual (CF) explanations present `what ifs' but leave users grappling with the 'why'. To bridge this gap, we introduce XAIstories. Leveraging Large Language Models, XAIstories provide narratives that shed light on AI predictions: SHAPstories do so based on SHAP explanations to explain a prediction score, while CFstories do so for CF explanations to explain a decision. Our results are striking: over 90% of the surveyed general audience finds the narrative generated by SHAPstories convincing. Data scientists primarily see the value of SHAPstories in communicating explanations to a general audience, with 92% of data scientists indicating that it will contribute to the ease and confidence of nonspecialists in understanding AI predictions. Additionally, 83% of data scientists indicate they are likely to use SHAPstories for this purpose. In image classification, CFstories are considered more or equally convincing as users own crafted stories by over 75% of lay user participants. CFstories also bring a tenfold speed gain in creating a narrative, and improves accuracy by over 20% compared to manually created narratives. The results thereby suggest that XAIstories may provide the missing link in truly explaining and understanding AI predictions.

{{</citation>}}


### (84/147) On Generating Explanations for Reinforcement Learning Policies: An Empirical Study (Mikihisa Yuasa et al., 2023)

{{<citation>}}

Mikihisa Yuasa, Huy T. Tran, Ramavarapu S. Sreenivas. (2023)  
**On Generating Explanations for Reinforcement Learning Policies: An Empirical Study**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16960v1)  

---


**ABSTRACT**  
In this paper, we introduce a set of \textit{Linear Temporal Logic} (LTL) formulae designed to provide explanations for policies. Our focus is on crafting explanations that elucidate both the ultimate objectives accomplished by the policy and the prerequisites it upholds throughout its execution. These LTL-based explanations feature a structured representation, which is particularly well-suited for local-search techniques. The effectiveness of our proposed approach is illustrated through a simulated capture the flag environment. The paper concludes with suggested directions for future research.

{{</citation>}}


## cs.RO (6)



### (85/147) Cook2LTL: Translating Cooking Recipes to LTL Formulae using Large Language Models (Angelos Mavrogiannis et al., 2023)

{{<citation>}}

Angelos Mavrogiannis, Christoforos Mavrogiannis, Yiannis Aloimonos. (2023)  
**Cook2LTL: Translating Cooking Recipes to LTL Formulae using Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.00163v1)  

---


**ABSTRACT**  
Cooking recipes are especially challenging to translate to robot plans as they feature rich linguistic complexity, temporally-extended interconnected tasks, and an almost infinite space of possible actions. Our key insight is that combining a source of background cooking domain knowledge with a formalism capable of handling the temporal richness of cooking recipes could enable the extraction of unambiguous, robot-executable plans. In this work, we use Linear Temporal Logic (LTL) as a formal language expressible enough to model the temporal nature of cooking recipes. Leveraging pre-trained Large Language Models (LLMs), we present a system that translates instruction steps from an arbitrary cooking recipe found on the internet to a series of LTL formulae, grounding high-level cooking actions to a set of primitive actions that are executable by a manipulator in a kitchen environment. Our approach makes use of a caching scheme, dynamically building a queryable action library at runtime, significantly decreasing LLM API calls (-51%), latency (-59%) and cost (-42%) compared to a baseline that queries the LLM for every newly encountered action at runtime. We demonstrate the transferability of our system in a realistic simulation platform through showcasing a set of simple cooking tasks.

{{</citation>}}


### (86/147) Learning Decentralized Flocking Controllers with Spatio-Temporal Graph Neural Network (Siji Chen et al., 2023)

{{<citation>}}

Siji Chen, Yanshen Sun, Peihan Li, Lifeng Zhou, Chang-Tien Lu. (2023)  
**Learning Decentralized Flocking Controllers with Spatio-Temporal Graph Neural Network**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.17437v2)  

---


**ABSTRACT**  
Recently a line of researches has delved the use of graph neural networks (GNNs) for decentralized control in swarm robotics. However, it has been observed that relying solely on the states of immediate neighbors is insufficient to imitate a centralized control policy. To address this limitation, prior studies proposed incorporating $L$-hop delayed states into the computation. While this approach shows promise, it can lead to a lack of consensus among distant flock members and the formation of small clusters, consequently resulting in the failure of cohesive flocking behaviors. Instead, our approach leverages spatiotemporal GNN, named STGNN that encompasses both spatial and temporal expansions. The spatial expansion collects delayed states from distant neighbors, while the temporal expansion incorporates previous states from immediate neighbors. The broader and more comprehensive information gathered from both expansions results in more effective and accurate predictions. We develop an expert algorithm for controlling a swarm of robots and employ imitation learning to train our decentralized STGNN model based on the expert algorithm. We simulate the proposed STGNN approach in various settings, demonstrating its decentralized capacity to emulate the global expert algorithm. Further, we implemented our approach to achieve cohesive flocking, leader following and obstacle avoidance by a group of Crazyflie drones. The performance of STGNN underscores its potential as an effective and reliable approach for achieving cohesive flocking, leader following and obstacle avoidance tasks.

{{</citation>}}


### (87/147) DREAM: Decentralized Reinforcement Learning for Exploration and Efficient Energy Management in Multi-Robot Systems (Dipam Patel et al., 2023)

{{<citation>}}

Dipam Patel, Phu Pham, Kshitij Tiwari, Aniket Bera. (2023)  
**DREAM: Decentralized Reinforcement Learning for Exploration and Efficient Energy Management in Multi-Robot Systems**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-MA, cs-RO, cs.RO  
Keywords: Graph Neural Network, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.17433v1)  

---


**ABSTRACT**  
Resource-constrained robots often suffer from energy inefficiencies, underutilized computational abilities due to inadequate task allocation, and a lack of robustness in dynamic environments, all of which strongly affect their performance. This paper introduces DREAM - Decentralized Reinforcement Learning for Exploration and Efficient Energy Management in Multi-Robot Systems, a comprehensive framework that optimizes the allocation of resources for efficient exploration. It advances beyond conventional heuristic-based task planning as observed conventionally. The framework incorporates Operational Range Estimation using Reinforcement Learning to perform exploration and obstacle avoidance in unfamiliar terrains. DREAM further introduces an Energy Consumption Model for goal allocation, thereby ensuring mission completion under constrained resources using a Graph Neural Network. This approach also ensures that the entire Multi-Robot System can survive for an extended period of time for further missions compared to the conventional approach of randomly allocating goals, which compromises one or more agents. Our approach adapts to prioritizing agents in real-time, showcasing remarkable resilience against dynamic environments. This robust solution was evaluated in various simulated environments, demonstrating adaptability and applicability across diverse scenarios. We observed a substantial improvement of about 25% over the baseline method, leading the way for future research in resource-constrained robotics.

{{</citation>}}


### (88/147) MORPH: Design Co-optimization with Reinforcement Learning via a Differentiable Hardware Model Proxy (Zhanpeng He et al., 2023)

{{<citation>}}

Zhanpeng He, Matei Ciocarlie. (2023)  
**MORPH: Design Co-optimization with Reinforcement Learning via a Differentiable Hardware Model Proxy**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.17227v1)  

---


**ABSTRACT**  
We introduce MORPH, a method for co-optimization of hardware design parameters and control policies in simulation using reinforcement learning. Like most co-optimization methods, MORPH relies on a model of the hardware being optimized, usually simulated based on the laws of physics. However, such a model is often difficult to integrate into an effective optimization routine. To address this, we introduce a proxy hardware model, which is always differentiable and enables efficient co-optimization alongside a long-horizon control policy using RL. MORPH is designed to ensure that the optimized hardware proxy remains as close as possible to its realistic counterpart, while still enabling task completion. We demonstrate our approach on simulated 2D reaching and 3D multi-fingered manipulation tasks.

{{</citation>}}


### (89/147) Robots That Can See: Leveraging Human Pose for Trajectory Prediction (Tim Salzmann et al., 2023)

{{<citation>}}

Tim Salzmann, Lewis Chiang, Markus Ryll, Dorsa Sadigh, Carolina Parada, Alex Bewley. (2023)  
**Robots That Can See: Leveraging Human Pose for Trajectory Prediction**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-HC, cs-LG, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.17209v1)  

---


**ABSTRACT**  
Anticipating the motion of all humans in dynamic environments such as homes and offices is critical to enable safe and effective robot navigation. Such spaces remain challenging as humans do not follow strict rules of motion and there are often multiple occluded entry points such as corners and doors that create opportunities for sudden encounters. In this work, we present a Transformer based architecture to predict human future trajectories in human-centric environments from input features including human positions, head orientations, and 3D skeletal keypoints from onboard in-the-wild sensory information. The resulting model captures the inherent uncertainty for future human trajectory prediction and achieves state-of-the-art performance on common prediction benchmarks and a human tracking dataset captured from a mobile robot adapted for the prediction task. Furthermore, we identify new agents with limited historical data as a major contributor to error and demonstrate the complementary nature of 3D skeletal poses in reducing prediction error in such challenging scenarios.

{{</citation>}}


### (90/147) CrossLoco: Human Motion Driven Control of Legged Robots via Guided Unsupervised Reinforcement Learning (Tianyu Li et al., 2023)

{{<citation>}}

Tianyu Li, Hyunyoung Jung, Matthew Gombolay, Yong Kwon Cho, Sehoon Ha. (2023)  
**CrossLoco: Human Motion Driven Control of Legged Robots via Guided Unsupervised Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.17046v1)  

---


**ABSTRACT**  
Human motion driven control (HMDC) is an effective approach for generating natural and compelling robot motions while preserving high-level semantics. However, establishing the correspondence between humans and robots with different body structures is not straightforward due to the mismatches in kinematics and dynamics properties, which causes intrinsic ambiguity to the problem. Many previous algorithms approach this motion retargeting problem with unsupervised learning, which requires the prerequisite skill sets. However, it will be extremely costly to learn all the skills without understanding the given human motions, particularly for high-dimensional robots. In this work, we introduce CrossLoco, a guided unsupervised reinforcement learning framework that simultaneously learns robot skills and their correspondence to human motions. Our key innovation is to introduce a cycle-consistency-based reward term designed to maximize the mutual information between human motions and robot states. We demonstrate that the proposed framework can generate compelling robot motions by translating diverse human motions, such as running, hopping, and dancing. We quantitatively compare our CrossLoco against the manually engineered and unsupervised baseline algorithms along with the ablated versions of our framework and demonstrate that our method translates human motions with better accuracy, diversity, and user preference. We also showcase its utility in other applications, such as synthesizing robot movements from language input and enabling interactive robot control.

{{</citation>}}


## cs.CV (31)



### (91/147) Feedback-guided Data Synthesis for Imbalanced Classification (Reyhane Askari Hemmat et al., 2023)

{{<citation>}}

Reyhane Askari Hemmat, Mohammad Pezeshki, Florian Bordes, Michal Drozdzal, Adriana Romero-Soriano. (2023)  
**Feedback-guided Data Synthesis for Imbalanced Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.00158v1)  

---


**ABSTRACT**  
Current status quo in machine learning is to use static datasets of real images for training, which often come from long-tailed distributions. With the recent advances in generative models, researchers have started augmenting these static datasets with synthetic data, reporting moderate performance improvements on classification tasks. We hypothesize that these performance gains are limited by the lack of feedback from the classifier to the generative model, which would promote the usefulness of the generated samples to improve the classifier's performance. In this work, we introduce a framework for augmenting static datasets with useful synthetic samples, which leverages one-shot feedback from the classifier to drive the sampling of the generative model. In order for the framework to be effective, we find that the samples must be close to the support of the real data of the task at hand, and be sufficiently diverse. We validate three feedback criteria on a long-tailed dataset (ImageNet-LT) as well as a group-imbalanced dataset (NICO++). On ImageNet-LT, we achieve state-of-the-art results, with over 4 percent improvement on underrepresented classes while being twice efficient in terms of the number of generated synthetic samples. NICO++ also enjoys marked boosts of over 5 percent in worst group accuracy. With these results, our framework paves the path towards effectively leveraging state-of-the-art text-to-image models as data sources that can be queried to improve downstream applications.

{{</citation>}}


### (92/147) Rethinking Audiovisual Segmentation with Semantic Quantization and Decomposition (Xiang Li et al., 2023)

{{<citation>}}

Xiang Li, Jinglu Wang, Xiaohao Xu, Xiulian Peng, Rita Singh, Yan Lu, Bhiksha Raj. (2023)  
**Rethinking Audiovisual Segmentation with Semantic Quantization and Decomposition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2310.00132v1)  

---


**ABSTRACT**  
Audiovisual segmentation (AVS) is a challenging task that aims to segment visual objects in videos based on their associated acoustic cues. With multiple sound sources involved, establishing robust correspondences between audio and visual contents poses unique challenges due to its (1) intricate entanglement across sound sources and (2) frequent shift among sound events. Assuming sound events occur independently, the multi-source semantic space (which encompasses all possible semantic categories) can be viewed as the Cartesian product of single-source sub-spaces. This motivates us to decompose the multi-source audio semantics into single-source semantics, allowing for more effective interaction with visual content. Specifically, we propose a semantic decomposition method based on product quantization, where the multi-source semantics can be decomposed and represented by several quantized single-source semantics. Furthermore, we introduce a global-to-local quantization mechanism that distills knowledge from stable global (clip-level) features into local (frame-level) ones to handle the constant shift of audio semantics. Extensive experiments demonstrate that semantically quantized and decomposed audio representation significantly improves AVS performance, e.g., +21.2% mIoU on the most challenging AVS-Semantic benchmark.

{{</citation>}}


### (93/147) Denoising and Selecting Pseudo-Heatmaps for Semi-Supervised Human Pose Estimation (Zhuoran Yu et al., 2023)

{{<citation>}}

Zhuoran Yu, Manchen Wang, Yanbei Chen, Paolo Favaro, Davide Modolo. (2023)  
**Denoising and Selecting Pseudo-Heatmaps for Semi-Supervised Human Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.00099v1)  

---


**ABSTRACT**  
We propose a new semi-supervised learning design for human pose estimation that revisits the popular dual-student framework and enhances it two ways. First, we introduce a denoising scheme to generate reliable pseudo-heatmaps as targets for learning from unlabeled data. This uses multi-view augmentations and a threshold-and-refine procedure to produce a pool of pseudo-heatmaps. Second, we select the learning targets from these pseudo-heatmaps guided by the estimated cross-student uncertainty. We evaluate our proposed method on multiple evaluation setups on the COCO benchmark. Our results show that our model outperforms previous state-of-the-art semi-supervised pose estimators, especially in extreme low-data regime. For example with only 0.5K labeled images our method is capable of surpassing the best competitor by 7.22 mAP (+25% absolute improvement). We also demonstrate that our model can learn effectively from unlabeled data in the wild to further boost its generalization and performance.

{{</citation>}}


### (94/147) Towards Few-Call Model Stealing via Active Self-Paced Knowledge Distillation and Diffusion-Based Image Generation (Vlad Hondru et al., 2023)

{{<citation>}}

Vlad Hondru, Radu Tudor Ionescu. (2023)  
**Towards Few-Call Model Stealing via Active Self-Paced Knowledge Distillation and Diffusion-Based Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.00096v1)  

---


**ABSTRACT**  
Diffusion models showcased strong capabilities in image synthesis, being used in many computer vision tasks with great success. To this end, we propose to explore a new use case, namely to copy black-box classification models without having access to the original training data, the architecture, and the weights of the model, \ie~the model is only exposed through an inference API. More specifically, we can only observe the (soft or hard) labels for some image samples passed as input to the model. Furthermore, we consider an additional constraint limiting the number of model calls, mostly focusing our research on few-call model stealing. In order to solve the model extraction task given the applied restrictions, we propose the following framework. As training data, we create a synthetic data set (called proxy data set) by leveraging the ability of diffusion models to generate realistic and diverse images. Given a maximum number of allowed API calls, we pass the respective number of samples through the black-box model to collect labels. Finally, we distill the knowledge of the black-box teacher (attacked model) into a student model (copy of the attacked model), harnessing both labeled and unlabeled data generated by the diffusion model. We employ a novel active self-paced learning framework to make the most of the proxy data during distillation. Our empirical results on two data sets confirm the superiority of our framework over two state-of-the-art methods in the few-call model extraction scenario.

{{</citation>}}


### (95/147) DataDAM: Efficient Dataset Distillation with Attention Matching (Ahmad Sajedi et al., 2023)

{{<citation>}}

Ahmad Sajedi, Samir Khaki, Ehsan Amjadian, Lucy Z. Liu, Yuri A. Lawryshyn, Konstantinos N. Plataniotis. (2023)  
**DataDAM: Efficient Dataset Distillation with Attention Matching**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.00093v1)  

---


**ABSTRACT**  
Researchers have long tried to minimize training costs in deep learning while maintaining strong generalization across diverse datasets. Emerging research on dataset distillation aims to reduce training costs by creating a small synthetic set that contains the information of a larger real dataset and ultimately achieves test accuracy equivalent to a model trained on the whole dataset. Unfortunately, the synthetic data generated by previous methods are not guaranteed to distribute and discriminate as well as the original training data, and they incur significant computational costs. Despite promising results, there still exists a significant performance gap between models trained on condensed synthetic sets and those trained on the whole dataset. In this paper, we address these challenges using efficient Dataset Distillation with Attention Matching (DataDAM), achieving state-of-the-art performance while reducing training costs. Specifically, we learn synthetic images by matching the spatial attention maps of real and synthetic data generated by different layers within a family of randomly initialized neural networks. Our method outperforms the prior methods on several datasets, including CIFAR10/100, TinyImageNet, ImageNet-1K, and subsets of ImageNet-1K across most of the settings, and achieves improvements of up to 6.5% and 4.1% on CIFAR100 and ImageNet-1K, respectively. We also show that our high-quality distilled images have practical benefits for downstream applications, such as continual learning and neural architecture search.

{{</citation>}}


### (96/147) Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks (Mehrdad Saberi et al., 2023)

{{<citation>}}

Mehrdad Saberi, Vinu Sankar Sadasivan, Keivan Rezaei, Aounon Kumar, Atoosa Chegini, Wenxiao Wang, Soheil Feizi. (2023)  
**Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.00076v1)  

---


**ABSTRACT**  
In light of recent advancements in generative AI models, it has become essential to distinguish genuine content from AI-generated one to prevent the malicious usage of fake materials as authentic ones and vice versa. Various techniques have been introduced for identifying AI-generated images, with watermarking emerging as a promising approach. In this paper, we analyze the robustness of various AI-image detectors including watermarking and classifier-based deepfake detectors. For watermarking methods that introduce subtle image perturbations (i.e., low perturbation budget methods), we reveal a fundamental trade-off between the evasion error rate (i.e., the fraction of watermarked images detected as non-watermarked ones) and the spoofing error rate (i.e., the fraction of non-watermarked images detected as watermarked ones) upon an application of a diffusion purification attack. In this regime, we also empirically show that diffusion purification effectively removes watermarks with minimal changes to images. For high perturbation watermarking methods where notable changes are applied to images, the diffusion purification attack is not effective. In this case, we develop a model substitution adversarial attack that can successfully remove watermarks. Moreover, we show that watermarking methods are vulnerable to spoofing attacks where the attacker aims to have real images (potentially obscene) identified as watermarked ones, damaging the reputation of the developers. In particular, by just having black-box access to the watermarking method, we show that one can generate a watermarked noise image which can be added to the real images to have them falsely flagged as watermarked ones. Finally, we extend our theory to characterize a fundamental trade-off between the robustness and reliability of classifier-based deep fake detectors and demonstrate it through experiments.

{{</citation>}}


### (97/147) Multi-task View Synthesis with Neural Radiance Fields (Shuhong Zheng et al., 2023)

{{<citation>}}

Shuhong Zheng, Zhipeng Bao, Martial Hebert, Yu-Xiong Wang. (2023)  
**Multi-task View Synthesis with Neural Radiance Fields**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.17450v1)  

---


**ABSTRACT**  
Multi-task visual learning is a critical aspect of computer vision. Current research, however, predominantly concentrates on the multi-task dense prediction setting, which overlooks the intrinsic 3D world and its multi-view consistent structures, and lacks the capability for versatile imagination. In response to these limitations, we present a novel problem setting -- multi-task view synthesis (MTVS), which reinterprets multi-task prediction as a set of novel-view synthesis tasks for multiple scene properties, including RGB. To tackle the MTVS problem, we propose MuvieNeRF, a framework that incorporates both multi-task and cross-view knowledge to simultaneously synthesize multiple scene properties. MuvieNeRF integrates two key modules, the Cross-Task Attention (CTA) and Cross-View Attention (CVA) modules, enabling the efficient use of information across multiple views and tasks. Extensive evaluation on both synthetic and realistic benchmarks demonstrates that MuvieNeRF is capable of simultaneously synthesizing different scene properties with promising visual quality, even outperforming conventional discriminative models in various settings. Notably, we show that MuvieNeRF exhibits universal applicability across a range of NeRF backbones. Our code is available at https://github.com/zsh2000/MuvieNeRF.

{{</citation>}}


### (98/147) FACTS: First Amplify Correlations and Then Slice to Discover Bias (Sriram Yenamandra et al., 2023)

{{<citation>}}

Sriram Yenamandra, Pratik Ramesh, Viraj Prabhu, Judy Hoffman. (2023)  
**FACTS: First Amplify Correlations and Then Slice to Discover Bias**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.17430v1)  

---


**ABSTRACT**  
Computer vision datasets frequently contain spurious correlations between task-relevant labels and (easy to learn) latent task-irrelevant attributes (e.g. context). Models trained on such datasets learn "shortcuts" and underperform on bias-conflicting slices of data where the correlation does not hold. In this work, we study the problem of identifying such slices to inform downstream bias mitigation strategies. We propose First Amplify Correlations and Then Slice to Discover Bias (FACTS), wherein we first amplify correlations to fit a simple bias-aligned hypothesis via strongly regularized empirical risk minimization. Next, we perform correlation-aware slicing via mixture modeling in bias-aligned feature space to discover underperforming data slices that capture distinct correlations. Despite its simplicity, our method considerably improves over prior work (by as much as 35% precision@10) in correlation bias identification across a range of diverse evaluation settings. Our code is available at: https://github.com/yvsriram/FACTS.

{{</citation>}}


### (99/147) Classification of Potholes Based on Surface Area Using Pre-Trained Models of Convolutional Neural Network (Chauhdary Fazeel Ahmad et al., 2023)

{{<citation>}}

Chauhdary Fazeel Ahmad, Abdullah Cheema, Waqas Qayyum, Rana Ehtisham, Muhammad Haroon Yousaf, Junaid Mir, Nasim Shakouri Mahmoudabadi, Afaq Ahmad. (2023)  
**Classification of Potholes Based on Surface Area Using Pre-Trained Models of Convolutional Neural Network**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2309.17426v1)  

---


**ABSTRACT**  
Potholes are fatal and can cause severe damage to vehicles as well as can cause deadly accidents. In South Asian countries, pavement distresses are the primary cause due to poor subgrade conditions, lack of subsurface drainage, and excessive rainfalls. The present research compares the performance of three pre-trained Convolutional Neural Network (CNN) models, i.e., ResNet 50, ResNet 18, and MobileNet. At first, pavement images are classified to find whether images contain potholes, i.e., Potholes or Normal. Secondly, pavements images are classi-fied into three categories, i.e., Small Pothole, Large Pothole, and Normal. Pavement images are taken from 3.5 feet (waist height) and 2 feet. MobileNet v2 has an accuracy of 98% for detecting a pothole. The classification of images taken at the height of 2 feet has an accuracy value of 87.33%, 88.67%, and 92% for classifying the large, small, and normal pavement, respectively. Similarly, the classification of the images taken from full of waist (FFW) height has an accuracy value of 98.67%, 98.67%, and 100%.

{{</citation>}}


### (100/147) The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision) (Zhengyuan Yang et al., 2023)

{{<citation>}}

Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, Lijuan Wang. (2023)  
**The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.17421v1)  

---


**ABSTRACT**  
Large multimodal models (LMMs) extend large language models (LLMs) with multi-sensory skills, such as visual understanding, to achieve stronger generic intelligence. In this paper, we analyze the latest model, GPT-4V(ision), to deepen the understanding of LMMs. The analysis focuses on the intriguing tasks that GPT-4V can perform, containing test samples to probe the quality and genericity of GPT-4V's capabilities, its supported inputs and working modes, and the effective ways to prompt the model. In our approach to exploring GPT-4V, we curate and organize a collection of carefully designed qualitative samples spanning a variety of domains and tasks. Observations from these samples demonstrate that GPT-4V's unprecedented ability in processing arbitrarily interleaved multimodal inputs and the genericity of its capabilities together make GPT-4V a powerful multimodal generalist system. Furthermore, GPT-4V's unique capability of understanding visual markers drawn on input images can give rise to new human-computer interaction methods such as visual referring prompting. We conclude the report with in-depth discussions on the emerging application scenarios and the future research directions for GPT-4V-based systems. We hope that this preliminary exploration will inspire future research on the next-generation multimodal task formulation, new ways to exploit and enhance LMMs to solve real-world problems, and gaining better understanding of multimodal foundation models.

{{</citation>}}


### (101/147) IFAST: Weakly Supervised Interpretable Face Anti-spoofing from Single-shot Binocular NIR Images (Jiancheng Huang et al., 2023)

{{<citation>}}

Jiancheng Huang, Donghao Zhou, Shifeng Chen. (2023)  
**IFAST: Weakly Supervised Interpretable Face Anti-spoofing from Single-shot Binocular NIR Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.17399v1)  

---


**ABSTRACT**  
Single-shot face anti-spoofing (FAS) is a key technique for securing face recognition systems, and it requires only static images as input. However, single-shot FAS remains a challenging and under-explored problem due to two main reasons: 1) on the data side, learning FAS from RGB images is largely context-dependent, and single-shot images without additional annotations contain limited semantic information. 2) on the model side, existing single-shot FAS models are infeasible to provide proper evidence for their decisions, and FAS methods based on depth estimation require expensive per-pixel annotations. To address these issues, a large binocular NIR image dataset (BNI-FAS) is constructed and published, which contains more than 300,000 real face and plane attack images, and an Interpretable FAS Transformer (IFAST) is proposed that requires only weak supervision to produce interpretable predictions. Our IFAST can produce pixel-wise disparity maps by the proposed disparity estimation Transformer with Dynamic Matching Attention (DMA) block. Besides, a well-designed confidence map generator is adopted to cooperate with the proposed dual-teacher distillation module to obtain the final discriminant results. The comprehensive experiments show that our IFAST can achieve state-of-the-art results on BNI-FAS, proving the effectiveness of the single-shot FAS based on binocular NIR images.

{{</citation>}}


### (102/147) See Beyond Seeing: Robust 3D Object Detection from Point Clouds via Cross-Modal Hallucination (Jianning Deng et al., 2023)

{{<citation>}}

Jianning Deng, Gabriel Chan, Hantao Zhong, Chris Xiaoxuan Lu. (2023)  
**See Beyond Seeing: Robust 3D Object Detection from Point Clouds via Cross-Modal Hallucination**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.17336v1)  

---


**ABSTRACT**  
This paper presents a novel framework for robust 3D object detection from point clouds via cross-modal hallucination. Our proposed approach is agnostic to either hallucination direction between LiDAR and 4D radar. We introduce multiple alignments on both spatial and feature levels to achieve simultaneous backbone refinement and hallucination generation. Specifically, spatial alignment is proposed to deal with the geometry discrepancy for better instance matching between LiDAR and radar. The feature alignment step further bridges the intrinsic attribute gap between the sensing modalities and stabilizes the training. The trained object detection models can deal with difficult detection cases better, even though only single-modal data is used as the input during the inference stage. Extensive experiments on the View-of-Delft (VoD) dataset show that our proposed method outperforms the state-of-the-art (SOTA) methods for both radar and LiDAR object detection while maintaining competitive efficiency in runtime.

{{</citation>}}


### (103/147) Telling Stories for Common Sense Zero-Shot Action Recognition (Shreyank N Gowda et al., 2023)

{{<citation>}}

Shreyank N Gowda, Laura Sevilla-Lara. (2023)  
**Telling Stories for Common Sense Zero-Shot Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.17327v1)  

---


**ABSTRACT**  
Video understanding has long suffered from reliance on large labeled datasets, motivating research into zero-shot learning. Recent progress in language modeling presents opportunities to advance zero-shot video analysis, but constructing an effective semantic space relating action classes remains challenging. We address this by introducing a novel dataset, Stories, which contains rich textual descriptions for diverse action classes extracted from WikiHow articles. For each class, we extract multi-sentence narratives detailing the necessary steps, scenes, objects, and verbs that characterize the action. This contextual data enables modeling of nuanced relationships between actions, paving the way for zero-shot transfer. We also propose an approach that harnesses Stories to improve feature generation for training zero-shot classification. Without any target dataset fine-tuning, our method achieves new state-of-the-art on multiple benchmarks, improving top-1 accuracy by up to 6.1%. We believe Stories provides a valuable resource that can catalyze progress in zero-shot action recognition. The textual narratives forge connections between seen and unseen classes, overcoming the bottleneck of labeled data that has long impeded advancements in this exciting domain. The data can be found here: https://github.com/kini5gowda/Stories .

{{</citation>}}


### (104/147) Efficient Large Scale Medical Image Dataset Preparation for Machine Learning Applications (Stefan Denner et al., 2023)

{{<citation>}}

Stefan Denner, Jonas Scherer, Klaus Kades, Dimitrios Bounias, Philipp Schader, Lisa Kausch, Markus Bujotzek, Andreas Michael Bucher, Tobias Penzkofer, Klaus Maier-Hein. (2023)  
**Efficient Large Scale Medical Image Dataset Preparation for Machine Learning Applications**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.17285v1)  

---


**ABSTRACT**  
In the rapidly evolving field of medical imaging, machine learning algorithms have become indispensable for enhancing diagnostic accuracy. However, the effectiveness of these algorithms is contingent upon the availability and organization of high-quality medical imaging datasets. Traditional Digital Imaging and Communications in Medicine (DICOM) data management systems are inadequate for handling the scale and complexity of data required to be facilitated in machine learning algorithms. This paper introduces an innovative data curation tool, developed as part of the Kaapana open-source toolkit, aimed at streamlining the organization, management, and processing of large-scale medical imaging datasets. The tool is specifically tailored to meet the needs of radiologists and machine learning researchers. It incorporates advanced search, auto-annotation and efficient tagging functionalities for improved data curation. Additionally, the tool facilitates quality control and review, enabling researchers to validate image and segmentation quality in large datasets. It also plays a critical role in uncovering potential biases in datasets by aggregating and visualizing metadata, which is essential for developing robust machine learning models. Furthermore, Kaapana is integrated within the Radiological Cooperative Network (RACOON), a pioneering initiative aimed at creating a comprehensive national infrastructure for the aggregation, transmission, and consolidation of radiological data across all university clinics throughout Germany. A supplementary video showcasing the tool's functionalities can be accessed at https://bit.ly/MICCAI-DEMI2023.

{{</citation>}}


### (105/147) Information Flow in Self-Supervised Learning (Zhiquan Tan et al., 2023)

{{<citation>}}

Zhiquan Tan, Jingqin Yang, Weiran Huang, Yang Yuan, Yifan Zhang. (2023)  
**Information Flow in Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.17281v1)  

---


**ABSTRACT**  
In this paper, we provide a comprehensive toolbox for understanding and enhancing self-supervised learning (SSL) methods through the lens of matrix information theory. Specifically, by leveraging the principles of matrix mutual information and joint entropy, we offer a unified analysis for both contrastive and feature decorrelation based methods. Furthermore, we propose the matrix variational masked auto-encoder (M-MAE) method, grounded in matrix information theory, as an enhancement to masked image modeling. The empirical evaluations underscore the effectiveness of M-MAE compared with the state-of-the-art methods, including a 3.9% improvement in linear probing ViT-Base, and a 1% improvement in fine-tuning ViT-Large, both on ImageNet.

{{</citation>}}


### (106/147) Effect of structure-based training on 3D localization precision and quality (Armin Abdehkakha et al., 2023)

{{<citation>}}

Armin Abdehkakha, Craig Snoeyink. (2023)  
**Effect of structure-based training on 3D localization precision and quality**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.17265v1)  

---


**ABSTRACT**  
This study introduces a structural-based training approach for CNN-based algorithms in single-molecule localization microscopy (SMLM) and 3D object reconstruction. We compare this approach with the traditional random-based training method, utilizing the LUENN package as our AI pipeline. The quantitative evaluation demonstrates significant improvements in detection rate and localization precision with the structural-based training approach, particularly in varying signal-to-noise ratios (SNRs). Moreover, the method effectively removes checkerboard artifacts, ensuring more accurate 3D reconstructions. Our findings highlight the potential of the structural-based training approach to advance super-resolution microscopy and deepen our understanding of complex biological systems at the nanoscale.

{{</citation>}}


### (107/147) When Epipolar Constraint Meets Non-local Operators in Multi-View Stereo (Tianqi Liu et al., 2023)

{{<citation>}}

Tianqi Liu, Xinyi Ye, Weiyue Zhao, Zhiyu Pan, Min Shi, Zhiguo Cao. (2023)  
**When Epipolar Constraint Meets Non-local Operators in Multi-View Stereo**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.17218v1)  

---


**ABSTRACT**  
Learning-based multi-view stereo (MVS) method heavily relies on feature matching, which requires distinctive and descriptive representations. An effective solution is to apply non-local feature aggregation, e.g., Transformer. Albeit useful, these techniques introduce heavy computation overheads for MVS. Each pixel densely attends to the whole image. In contrast, we propose to constrain non-local feature augmentation within a pair of lines: each point only attends the corresponding pair of epipolar lines. Our idea takes inspiration from the classic epipolar geometry, which shows that one point with different depth hypotheses will be projected to the epipolar line on the other view. This constraint reduces the 2D search space into the epipolar line in stereo matching. Similarly, this suggests that the matching of MVS is to distinguish a series of points lying on the same line. Inspired by this point-to-line search, we devise a line-to-point non-local augmentation strategy. We first devise an optimized searching algorithm to split the 2D feature maps into epipolar line pairs. Then, an Epipolar Transformer (ET) performs non-local feature augmentation among epipolar line pairs. We incorporate the ET into a learning-based MVS baseline, named ET-MVSNet. ET-MVSNet achieves state-of-the-art reconstruction performance on both the DTU and Tanks-and-Temples benchmark with high efficiency. Code is available at https://github.com/TQTQliu/ET-MVSNet.

{{</citation>}}


### (108/147) Instant Complexity Reduction in CNNs using Locality-Sensitive Hashing (Lukas Meiner et al., 2023)

{{<citation>}}

Lukas Meiner, Jens Mehnert, Alexandru Paul Condurache. (2023)  
**Instant Complexity Reduction in CNNs using Locality-Sensitive Hashing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.17211v1)  

---


**ABSTRACT**  
To reduce the computational cost of convolutional neural networks (CNNs) for usage on resource-constrained devices, structured pruning approaches have shown promising results, drastically reducing floating-point operations (FLOPs) without substantial drops in accuracy. However, most recent methods require fine-tuning or specific training procedures to achieve a reasonable trade-off between retained accuracy and reduction in FLOPs. This introduces additional cost in the form of computational overhead and requires training data to be available. To this end, we propose HASTE (Hashing for Tractable Efficiency), a parameter-free and data-free module that acts as a plug-and-play replacement for any regular convolution module. It instantly reduces the network's test-time inference cost without requiring any training or fine-tuning. We are able to drastically compress latent feature maps without sacrificing much accuracy by using locality-sensitive hashing (LSH) to detect redundancies in the channel dimension. Similar channels are aggregated to reduce the input and filter depth simultaneously, allowing for cheaper convolutions. We demonstrate our approach on the popular vision benchmarks CIFAR-10 and ImageNet. In particular, we are able to instantly drop 46.72% of FLOPs while only losing 1.25% accuracy by just swapping the convolution modules in a ResNet34 on CIFAR-10 for our HASTE module.

{{</citation>}}


### (109/147) Revisiting Cephalometric Landmark Detection from the view of Human Pose Estimation with Lightweight Super-Resolution Head (Qian Wu et al., 2023)

{{<citation>}}

Qian Wu, Si Yong Yeo, Yufei Chen, Jun Liu. (2023)  
**Revisiting Cephalometric Landmark Detection from the view of Human Pose Estimation with Lightweight Super-Resolution Head**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.17143v1)  

---


**ABSTRACT**  
Accurate localization of cephalometric landmarks holds great importance in the fields of orthodontics and orthognathics due to its potential for automating key point labeling. In the context of landmark detection, particularly in cephalometrics, it has been observed that existing methods often lack standardized pipelines and well-designed bias reduction processes, which significantly impact their performance. In this paper, we revisit a related task, human pose estimation (HPE), which shares numerous similarities with cephalometric landmark detection (CLD), and emphasize the potential for transferring techniques from the former field to benefit the latter. Motivated by this insight, we have developed a robust and adaptable benchmark based on the well-established HPE codebase known as MMPose. This benchmark can serve as a dependable baseline for achieving exceptional CLD performance. Furthermore, we introduce an upscaling design within the framework to further enhance performance. This enhancement involves the incorporation of a lightweight and efficient super-resolution module, which generates heatmap predictions on high-resolution features and leads to further performance refinement, benefiting from its ability to reduce quantization bias. In the MICCAI CLDetection2023 challenge, our method achieves 1st place ranking on three metrics and 3rd place on the remaining one. The code for our method is available at https://github.com/5k5000/CLdetection2023.

{{</citation>}}


### (110/147) Reconstruction of Patient-Specific Confounders in AI-based Radiologic Image Interpretation using Generative Pretraining (Tianyu Han et al., 2023)

{{<citation>}}

Tianyu Han, Laura Žigutytė, Luisa Huck, Marc Huppertz, Robert Siepmann, Yossi Gandelsman, Christian Blüthgen, Firas Khader, Christiane Kuhl, Sven Nebelung, Jakob Kather, Daniel Truhn. (2023)  
**Reconstruction of Patient-Specific Confounders in AI-based Radiologic Image Interpretation using Generative Pretraining**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.17123v1)  

---


**ABSTRACT**  
Detecting misleading patterns in automated diagnostic assistance systems, such as those powered by Artificial Intelligence, is critical to ensuring their reliability, particularly in healthcare. Current techniques for evaluating deep learning models cannot visualize confounding factors at a diagnostic level. Here, we propose a self-conditioned diffusion model termed DiffChest and train it on a dataset of 515,704 chest radiographs from 194,956 patients from multiple healthcare centers in the United States and Europe. DiffChest explains classifications on a patient-specific level and visualizes the confounding factors that may mislead the model. We found high inter-reader agreement when evaluating DiffChest's capability to identify treatment-related confounders, with Fleiss' Kappa values of 0.8 or higher across most imaging findings. Confounders were accurately captured with 11.1% to 100% prevalence rates. Furthermore, our pretraining process optimized the model to capture the most relevant information from the input radiographs. DiffChest achieved excellent diagnostic accuracy when diagnosing 11 chest conditions, such as pleural effusion and cardiac insufficiency, and at least sufficient diagnostic accuracy for the remaining conditions. Our findings highlight the potential of pretraining based on diffusion models in medical image classification, specifically in providing insights into confounding factors and model robustness.

{{</citation>}}


### (111/147) Continual Action Assessment via Task-Consistent Score-Discriminative Feature Distribution Modeling (Yuan-Ming Li et al., 2023)

{{<citation>}}

Yuan-Ming Li, Ling-An Zeng, Jing-Ke Meng, Wei-Shi Zheng. (2023)  
**Continual Action Assessment via Task-Consistent Score-Discriminative Feature Distribution Modeling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.17105v1)  

---


**ABSTRACT**  
Action Quality Assessment (AQA) is a task that tries to answer how well an action is carried out. While remarkable progress has been achieved, existing works on AQA assume that all the training data are visible for training in one time, but do not enable continual learning on assessing new technical actions. In this work, we address such a Continual Learning problem in AQA (Continual-AQA), which urges a unified model to learn AQA tasks sequentially without forgetting. Our idea for modeling Continual-AQA is to sequentially learn a task-consistent score-discriminative feature distribution, in which the latent features express a strong correlation with the score labels regardless of the task or action types. From this perspective, we aim to mitigate the forgetting in Continual-AQA from two aspects. Firstly, to fuse the features of new and previous data into a score-discriminative distribution, a novel Feature-Score Correlation-Aware Rehearsal is proposed to store and reuse data from previous tasks with limited memory size. Secondly, an Action General-Specific Graph is developed to learn and decouple the action-general and action-specific knowledge so that the task-consistent score-discriminative features can be better extracted across various tasks. Extensive experiments are conducted to evaluate the contributions of proposed components. The comparisons with the existing continual learning methods additionally verify the effectiveness and versatility of our approach.

{{</citation>}}


### (112/147) Guiding Instruction-based Image Editing via Multimodal Large Language Models (Tsu-Jui Fu et al., 2023)

{{<citation>}}

Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, Zhe Gan. (2023)  
**Guiding Instruction-based Image Editing via Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.17102v1)  

---


**ABSTRACT**  
Instruction-based image editing improves the controllability and flexibility of image manipulation via natural commands without elaborate descriptions or regional masks. However, human instructions are sometimes too brief for current methods to capture and follow. Multimodal large language models (MLLMs) show promising capabilities in cross-modal understanding and visual-aware response generation via LMs. We investigate how MLLMs facilitate edit instructions and present MLLM-Guided Image Editing (MGIE). MGIE learns to derive expressive instructions and provides explicit guidance. The editing model jointly captures this visual imagination and performs manipulation through end-to-end training. We evaluate various aspects of Photoshop-style modification, global photo optimization, and local editing. Extensive experimental results demonstrate that expressive instructions are crucial to instruction-based image editing, and our MGIE can lead to a notable improvement in automatic metrics and human evaluation while maintaining competitive inference efficiency.

{{</citation>}}


### (113/147) SegRCDB: Semantic Segmentation via Formula-Driven Supervised Learning (Risa Shinoda et al., 2023)

{{<citation>}}

Risa Shinoda, Ryo Hayamizu, Kodai Nakashima, Nakamasa Inoue, Rio Yokota, Hirokatsu Kataoka. (2023)  
**SegRCDB: Semantic Segmentation via Formula-Driven Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.17083v1)  

---


**ABSTRACT**  
Pre-training is a strong strategy for enhancing visual models to efficiently train them with a limited number of labeled images. In semantic segmentation, creating annotation masks requires an intensive amount of labor and time, and therefore, a large-scale pre-training dataset with semantic labels is quite difficult to construct. Moreover, what matters in semantic segmentation pre-training has not been fully investigated. In this paper, we propose the Segmentation Radial Contour DataBase (SegRCDB), which for the first time applies formula-driven supervised learning for semantic segmentation. SegRCDB enables pre-training for semantic segmentation without real images or any manual semantic labels. SegRCDB is based on insights about what is important in pre-training for semantic segmentation and allows efficient pre-training. Pre-training with SegRCDB achieved higher mIoU than the pre-training with COCO-Stuff for fine-tuning on ADE-20k and Cityscapes with the same number of training images. SegRCDB has a high potential to contribute to semantic segmentation pre-training and investigation by enabling the creation of large datasets without manual annotation. The SegRCDB dataset will be released under a license that allows research and commercial use. Code is available at: https://github.com/dahlian00/SegRCDB

{{</citation>}}


### (114/147) GAIA-1: A Generative World Model for Autonomous Driving (Anthony Hu et al., 2023)

{{<citation>}}

Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado. (2023)  
**GAIA-1: A Generative World Model for Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2309.17080v1)  

---


**ABSTRACT**  
Autonomous driving promises transformative improvements to transportation, but building systems capable of safely navigating the unstructured complexity of real-world scenarios remains challenging. A critical problem lies in effectively predicting the various potential outcomes that may emerge in response to the vehicle's actions as the world evolves.   To address this challenge, we introduce GAIA-1 ('Generative AI for Autonomy'), a generative world model that leverages video, text, and action inputs to generate realistic driving scenarios while offering fine-grained control over ego-vehicle behavior and scene features. Our approach casts world modeling as an unsupervised sequence modeling problem by mapping the inputs to discrete tokens, and predicting the next token in the sequence. Emerging properties from our model include learning high-level structures and scene dynamics, contextual awareness, generalization, and understanding of geometry. The power of GAIA-1's learned representation that captures expectations of future events, combined with its ability to generate realistic samples, provides new possibilities for innovation in the field of autonomy, enabling enhanced and accelerated training of autonomous driving technology.

{{</citation>}}


### (115/147) GSDC Transformer: An Efficient and Effective Cue Fusion for Monocular Multi-Frame Depth Estimation (Naiyu Fang et al., 2023)

{{<citation>}}

Naiyu Fang, Lemiao Qiu, Shuyou Zhang, Zili Wang, Zheyuan Zhou, Kerui Hu. (2023)  
**GSDC Transformer: An Efficient and Effective Cue Fusion for Monocular Multi-Frame Depth Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.17059v1)  

---


**ABSTRACT**  
Depth estimation provides an alternative approach for perceiving 3D information in autonomous driving. Monocular depth estimation, whether with single-frame or multi-frame inputs, has achieved significant success by learning various types of cues and specializing in either static or dynamic scenes. Recently, these cues fusion becomes an attractive topic, aiming to enable the combined cues to perform well in both types of scenes. However, adaptive cue fusion relies on attention mechanisms, where the quadratic complexity limits the granularity of cue representation. Additionally, explicit cue fusion depends on precise segmentation, which imposes a heavy burden on mask prediction. To address these issues, we propose the GSDC Transformer, an efficient and effective component for cue fusion in monocular multi-frame depth estimation. We utilize deformable attention to learn cue relationships at a fine scale, while sparse attention reduces computational requirements when granularity increases. To compensate for the precision drop in dynamic scenes, we represent scene attributes in the form of super tokens without relying on precise shapes. Within each super token attributed to dynamic scenes, we gather its relevant cues and learn local dense relationships to enhance cue fusion. Our method achieves state-of-the-art performance on the KITTI dataset with efficient fusion speed.

{{</citation>}}


### (116/147) On Uniform Scalar Quantization for Learned Image Compression (Haotian Zhang et al., 2023)

{{<citation>}}

Haotian Zhang, Li Li, Dong Liu. (2023)  
**On Uniform Scalar Quantization for Learned Image Compression**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2309.17051v1)  

---


**ABSTRACT**  
Learned image compression possesses a unique challenge when incorporating non-differentiable quantization into the gradient-based training of the networks. Several quantization surrogates have been proposed to fulfill the training, but they were not systematically justified from a theoretical perspective. We fill this gap by contrasting uniform scalar quantization, the most widely used category with rounding being its simplest case, and its training surrogates. In principle, we find two factors crucial: one is the discrepancy between the surrogate and rounding, leading to train-test mismatch; the other is gradient estimation risk due to the surrogate, which consists of bias and variance of the gradient estimation. Our analyses and simulations imply that there is a tradeoff between the train-test mismatch and the gradient estimation risk, and the tradeoff varies across different network structures. Motivated by these analyses, we present a method based on stochastic uniform annealing, which has an adjustable temperature coefficient to control the tradeoff. Moreover, our analyses enlighten us as to two subtle tricks: one is to set an appropriate lower bound for the variance parameter of the estimated quantized latent distribution, which effectively reduces the train-test mismatch; the other is to use zero-center quantization with partial stop-gradient, which reduces the gradient estimation variance and thus stabilize the training. Our method with the tricks is verified to outperform the existing practices of quantization surrogates on a variety of representative image compression networks.

{{</citation>}}


### (117/147) HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World (Xin Wang et al., 2023)

{{<citation>}}

Xin Wang, Taein Kwon, Mahdi Rad, Bowen Pan, Ishani Chakraborty, Sean Andrist, Dan Bohus, Ashley Feniello, Bugra Tekin, Felipe Vieira Frujeri, Neel Joshi, Marc Pollefeys. (2023)  
**HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.17024v1)  

---


**ABSTRACT**  
Building an interactive AI assistant that can perceive, reason, and collaborate with humans in the real world has been a long-standing pursuit in the AI community. This work is part of a broader research effort to develop intelligent agents that can interactively guide humans through performing tasks in the physical world. As a first step in this direction, we introduce HoloAssist, a large-scale egocentric human interaction dataset, where two people collaboratively complete physical manipulation tasks. The task performer executes the task while wearing a mixed-reality headset that captures seven synchronized data streams. The task instructor watches the performer's egocentric video in real time and guides them verbally. By augmenting the data with action and conversational annotations and observing the rich behaviors of various participants, we present key insights into how human assistants correct mistakes, intervene in the task completion procedure, and ground their instructions to the environment. HoloAssist spans 166 hours of data captured by 350 unique instructor-performer pairs. Furthermore, we construct and present benchmarks on mistake detection, intervention type prediction, and hand forecasting, along with detailed analysis. We expect HoloAssist will provide an important resource for building AI assistants that can fluidly collaborate with humans in the real world. Data can be downloaded at https://holoassist.github.io/.

{{</citation>}}


### (118/147) Segment Anything Model is a Good Teacher for Local Feature Learning (Jingqian Wu et al., 2023)

{{<citation>}}

Jingqian Wu, Rongtao Xu, Zach Wood-Doughty, Changwei Wang. (2023)  
**Segment Anything Model is a Good Teacher for Local Feature Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.16992v1)  

---


**ABSTRACT**  
Local feature detection and description play an important role in many computer vision tasks, which are designed to detect and describe keypoints in "any scene" and "any downstream task". Data-driven local feature learning methods need to rely on pixel-level correspondence for training, which is challenging to acquire at scale, thus hindering further improvements in performance. In this paper, we propose SAMFeat to introduce SAM (segment anything model), a fundamental model trained on 11 million images, as a teacher to guide local feature learning and thus inspire higher performance on limited datasets. To do so, first, we construct an auxiliary task of Pixel Semantic Relational Distillation (PSRD), which distillates feature relations with category-agnostic semantic information learned by the SAM encoder into a local feature learning network, to improve local feature description using semantic discrimination. Second, we develop a technique called Weakly Supervised Contrastive Learning Based on Semantic Grouping (WSC), which utilizes semantic groupings derived from SAM as weakly supervised signals, to optimize the metric space of local descriptors. Third, we design an Edge Attention Guidance (EAG) to further improve the accuracy of local feature detection and description by prompting the network to pay more attention to the edge region guided by SAM. SAMFeat's performance on various tasks such as image matching on HPatches, and long-term visual localization on Aachen Day-Night showcases its superiority over previous local features. The release code is available at https://github.com/vignywang/SAMFeat.

{{</citation>}}


### (119/147) COMNet: Co-Occurrent Matching for Weakly Supervised Semantic Segmentation (Yukun Su et al., 2023)

{{<citation>}}

Yukun Su, Jingliang Deng, Zonghan Li. (2023)  
**COMNet: Co-Occurrent Matching for Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.16959v1)  

---


**ABSTRACT**  
Image-level weakly supervised semantic segmentation is a challenging task that has been deeply studied in recent years. Most of the common solutions exploit class activation map (CAM) to locate object regions. However, such response maps generated by the classification network usually focus on discriminative object parts. In this paper, we propose a novel Co-Occurrent Matching Network (COMNet), which can promote the quality of the CAMs and enforce the network to pay attention to the entire parts of objects. Specifically, we perform inter-matching on paired images that contain common classes to enhance the corresponded areas, and construct intra-matching on a single image to propagate the semantic features across the object regions. The experiments on the Pascal VOC 2012 and MS-COCO datasets show that our network can effectively boost the performance of the baseline model and achieve new state-of-the-art performance.

{{</citation>}}


### (120/147) CrossZoom: Simultaneously Motion Deblurring and Event Super-Resolving (Chi Zhang et al., 2023)

{{<citation>}}

Chi Zhang, Xiang Zhang, Mingyuan Lin, Cheng Li, Chu He, Wen Yang, Gui-Song Xia, Lei Yu. (2023)  
**CrossZoom: Simultaneously Motion Deblurring and Event Super-Resolving**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.16949v1)  

---


**ABSTRACT**  
Even though the collaboration between traditional and neuromorphic event cameras brings prosperity to frame-event based vision applications, the performance is still confined by the resolution gap crossing two modalities in both spatial and temporal domains. This paper is devoted to bridging the gap by increasing the temporal resolution for images, i.e., motion deblurring, and the spatial resolution for events, i.e., event super-resolving, respectively. To this end, we introduce CrossZoom, a novel unified neural Network (CZ-Net) to jointly recover sharp latent sequences within the exposure period of a blurry input and the corresponding High-Resolution (HR) events. Specifically, we present a multi-scale blur-event fusion architecture that leverages the scale-variant properties and effectively fuses cross-modality information to achieve cross-enhancement. Attention-based adaptive enhancement and cross-interaction prediction modules are devised to alleviate the distortions inherent in Low-Resolution (LR) events and enhance the final results through the prior blur-event complementary information. Furthermore, we propose a new dataset containing HR sharp-blurry images and the corresponding HR-LR event streams to facilitate future research. Extensive qualitative and quantitative experiments on synthetic and real-world datasets demonstrate the effectiveness and robustness of the proposed method. Codes and datasets are released at https://bestrivenzc.github.io/CZ-Net/.

{{</citation>}}


### (121/147) Robust Asynchronous Collaborative 3D Detection via Bird's Eye View Flow (Sizhe Wei et al., 2023)

{{<citation>}}

Sizhe Wei, Yuxi Wei, Yue Hu, Yifan Lu, Yiqi Zhong, Siheng Chen, Ya Zhang. (2023)  
**Robust Asynchronous Collaborative 3D Detection via Bird's Eye View Flow**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.16940v1)  

---


**ABSTRACT**  
By facilitating communication among multiple agents, collaborative perception can substantially boost each agent's perception ability. However, temporal asynchrony among agents is inevitable in real-world due to communication delays, interruptions, and clock misalignments. This issue causes information mismatch during multi-agent fusion, seriously shaking the foundation of collaboration. To address this issue, we propose CoBEVFlow, an asynchrony-robust collaborative 3D perception system based on bird's eye view (BEV) flow. The key intuition of CoBEVFlow is to compensate motions to align asynchronous collaboration messages sent by multiple agents. To model the motion in a scene, we propose BEV flow, which is a collection of the motion vector corresponding to each spatial location. Based on BEV flow, asynchronous perceptual features can be reassigned to appropriate positions, mitigating the impact of asynchrony. CoBEVFlow has two advantages: (i) CoBEVFlow can handle asynchronous collaboration messages sent at irregular, continuous time stamps without discretization; and (ii) with BEV flow, CoBEVFlow only transports the original perceptual features, instead of generating new perceptual features, avoiding additional noises. To validate CoBEVFlow's efficacy, we create IRregular V2V(IRV2V), the first synthetic collaborative perception dataset with various temporal asynchronies that simulate different real-world scenarios. Extensive experiments conducted on both IRV2V and the real-world dataset DAIR-V2X show that CoBEVFlow consistently outperforms other baselines and is robust in extremely asynchronous settings. The code will be released.

{{</citation>}}


## cs.CY (2)



### (122/147) ILB: Graph Neural Network Enabled Emergency Demand Response Program For Electricity (Sina Shaham et al., 2023)

{{<citation>}}

Sina Shaham, Bhaskar Krishnamachari, Matthew Kahn. (2023)  
**ILB: Graph Neural Network Enabled Emergency Demand Response Program For Electricity**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.00129v1)  

---


**ABSTRACT**  
Demand Response (DR) programs have become a crucial component of smart electricity grids as they shift the flexibility of electricity consumption from supply to demand in response to the ever-growing demand for electricity. In particular, in times of crisis, an emergency DR program is required to manage unexpected spikes in energy demand. In this paper, we propose the Incentive-Driven Load Balancer (ILB), a program designed to efficiently manage demand and response during crisis situations. By offering incentives to flexible households likely to reduce demand, the ILB facilitates effective demand reduction and prepares them for unexpected events. To enable ILB, we introduce a two-step machine learning-based framework for participant selection, which employs a graph-based approach to identify households capable of easily adjusting their electricity consumption. This framework utilizes two Graph Neural Networks (GNNs): one for pattern recognition and another for household selection. Through extensive experiments on household-level electricity consumption in California, Michigan, and Texas, we demonstrate the ILB program's significant effectiveness in supporting communities during emergencies.

{{</citation>}}


### (123/147) Compromise in Multilateral Negotiations and the Global Regulation of Artificial Intelligence (Michal Natorski, 2023)

{{<citation>}}

Michal Natorski. (2023)  
**Compromise in Multilateral Negotiations and the Global Regulation of Artificial Intelligence**  

---
Primary Category: cs.CY  
Categories: K-4-0; K-4-1, cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.17158v1)  

---


**ABSTRACT**  
As artificial intelligence (AI) technologies spread worldwide, international discussions have increasingly focused on their consequences for democracy, human rights, fundamental freedoms, security, and economic and social development. In this context, UNESCO's Recommendation on the Ethics of Artificial Intelligence, adopted in November 2021, has emerged as the first global normative framework for AI development and deployment. The intense negotiations of every detail of the document brought forth numerous controversies among UNESCO member states. Drawing on a unique set of primary sources, including written positions and recorded deliberations, this paper explains the achievement of global compromise on AI regulation despite the multiplicity of UNESCO member-state positions representing a variety of liberal and sovereignist preferences. Building upon Boltanski's pragmatic sociology, it conceptualises the practice of multilateral negotiations and attributes the multilateral compromise to two embedded therein mechanisms: Structural normative hybridity and situated normative ambiguity allowed to accomplish a compromise by linking macro-normative structures with situated debates of multilateral negotiations.

{{</citation>}}


## cs.HC (1)



### (124/147) ABScribe: Rapid Exploration of Multiple Writing Variations in Human-AI Co-Writing Tasks using Large Language Models (Mohi Reza et al., 2023)

{{<citation>}}

Mohi Reza, Nathan Laundry, Ilya Musabirov, Peter Dushniku, Zhi Yuan "Michael" Yu, Kashish Mittal, Tovi Grossman, Michael Liut, Anastasia Kuzminykh, Joseph Jay Williams. (2023)  
**ABScribe: Rapid Exploration of Multiple Writing Variations in Human-AI Co-Writing Tasks using Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-LG, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.00117v1)  

---


**ABSTRACT**  
Exploring alternative ideas by rewriting text is integral to the writing process. State-of-the-art large language models (LLMs) can simplify writing variation generation. However, current interfaces pose challenges for simultaneous consideration of multiple variations: creating new versions without overwriting text can be difficult, and pasting them sequentially can clutter documents, increasing workload and disrupting writers' flow. To tackle this, we present ABScribe, an interface that supports rapid, yet visually structured, exploration of writing variations in human-AI co-writing tasks. With ABScribe, users can swiftly produce multiple variations using LLM prompts, which are auto-converted into reusable buttons. Variations are stored adjacently within text segments for rapid in-place comparisons using mouse-over interactions on a context toolbar. Our user study with 12 writers shows that ABScribe significantly reduces task workload (d = 1.20, p < 0.001), enhances user perceptions of the revision process (d = 2.41, p < 0.001) compared to a popular baseline workflow, and provides insights into how writers explore variations using LLMs.

{{</citation>}}


## cs.DL (1)



### (125/147) Characterizing Semantic Ambiguity of the Materials Science Ontologies (Scott McClellan et al., 2023)

{{<citation>}}

Scott McClellan, Yuan An, Xintong Zhao, Xia Lin, Jane Greenberg. (2023)  
**Characterizing Semantic Ambiguity of the Materials Science Ontologies**  

---
Primary Category: cs.DL  
Categories: H-3-1, cs-DL, cs.DL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.00078v1)  

---


**ABSTRACT**  
Growth in computational materials science and initiatives such as the Materials Genome Initiative (MGI) and the European Materials Modelling Council (EMMC) has motivated the development and application of ontologies. A key factor has been increased adoption of the FAIR principles, making research data findable, accessible, interoperable, and reusable (Wilkinson et al. 2016). This paper characterizes semantic interoperability among a subset of materials science ontologies in the MatPortal repository. Background context covers semantic interoperability, ontological commitment, and the materials science ontology landscape. The research focused on MatPortal's two interoperability protocols: LOOM term matching and URI matching. Results report the degree of overlap and demonstrate the different types of ambiguity among ontologies. The discussion considers implications for FAIR and AI, and the conclusion highlight key findings and next steps.

{{</citation>}}


## astro-ph.IM (1)



### (126/147) AI ensemble for signal detection of higher order gravitational wave modes of quasi-circular, spinning, non-precessing binary black hole mergers (Minyang Tian et al., 2023)

{{<citation>}}

Minyang Tian, E. A. Huerta, Huihuo Zheng. (2023)  
**AI ensemble for signal detection of higher order gravitational wave modes of quasi-circular, spinning, non-precessing binary black hole mergers**  

---
Primary Category: astro-ph.IM  
Categories: [12:35] Huerta, Eliu A 68T01, 68T35, 83C35, 83C57, astro-ph-IM, astro-ph.IM, cs-AI, gr-qc  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.00052v1)  

---


**ABSTRACT**  
We introduce spatiotemporal-graph models that concurrently process data from the twin advanced LIGO detectors and the advanced Virgo detector. We trained these AI classifiers with 2.4 million \texttt{IMRPhenomXPHM} waveforms that describe quasi-circular, spinning, non-precessing binary black hole mergers with component masses $m_{\{1,2\}}\in[3M_\odot, 50 M_\odot]$, and individual spins $s^z_{\{1,2\}}\in[-0.9, 0.9]$; and which include the $(\ell, |m|) = \{(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)\}$ modes, and mode mixing effects in the $\ell = 3, |m| = 2$ harmonics. We trained these AI classifiers within 22 hours using distributed training over 96 NVIDIA V100 GPUs in the Summit supercomputer. We then used transfer learning to create AI predictors that estimate the total mass of potential binary black holes identified by all AI classifiers in the ensemble. We used this ensemble, 3 AI classifiers and 2 predictors, to process a year-long test set in which we injected 300,000 signals. This year-long test set was processed within 5.19 minutes using 1024 NVIDIA A100 GPUs in the Polaris supercomputer (for AI inference) and 128 CPU nodes in the ThetaKNL supercomputer (for post-processing of noise triggers), housed at the Argonne Leadership Supercomputing Facility. These studies indicate that our AI ensemble provides state-of-the-art signal detection accuracy, and reports 2 misclassifications for every year of searched data. This is the first AI ensemble designed to search for and find higher order gravitational wave mode signals.

{{</citation>}}


## cs.SD (1)



### (127/147) Improving Audio Captioning Models with Fine-grained Audio Features, Text Embedding Supervision, and LLM Mix-up Augmentation (Shih-Lun Wu et al., 2023)

{{<citation>}}

Shih-Lun Wu, Xuankai Chang, Gordon Wichern, Jee-weon Jung, François Germain, Jonathan Le Roux, Shinji Watanabe. (2023)  
**Improving Audio Captioning Models with Fine-grained Audio Features, Text Embedding Supervision, and LLM Mix-up Augmentation**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Augmentation, ChatGPT, Embedding, GPT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.17352v1)  

---


**ABSTRACT**  
Automated audio captioning (AAC) aims to generate informative descriptions for various sounds from nature and/or human activities. In recent years, AAC has quickly attracted research interest, with state-of-the-art systems now relying on a sequence-to-sequence (seq2seq) backbone powered by strong models such as Transformers. Following the macro-trend of applied machine learning research, in this work, we strive to improve the performance of seq2seq AAC models by extensively leveraging pretrained models and large language models (LLMs). Specifically, we utilize BEATs to extract fine-grained audio features. Then, we employ Instructor LLM to fetch text embeddings of captions, and infuse their language-modality knowledge into BEATs audio features via an auxiliary InfoNCE loss function. Moreover, we propose a novel data augmentation method that uses ChatGPT to produce caption mix-ups (i.e., grammatical and compact combinations of two captions) which, together with the corresponding audio mixtures, increase not only the amount but also the complexity and diversity of training data. During inference, we propose to employ nucleus sampling and a hybrid reranking algorithm, which has not been explored in AAC research. Combining our efforts, our model achieves a new state-of-the-art 32.6 SPIDEr-FL score on the Clotho evaluation split, and wins the 2023 DCASE AAC challenge.

{{</citation>}}


## eess.IV (3)



### (128/147) Multi-Depth Branches Network for Efficient Image Super-Resolution (Huiyuan Tian et al., 2023)

{{<citation>}}

Huiyuan Tian, Li Zhang, Shijian Li, Min Yao, Gang Pan. (2023)  
**Multi-Depth Branches Network for Efficient Image Super-Resolution**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.17334v1)  

---


**ABSTRACT**  
Significant progress has been made in the field of super-resolution (SR), yet many convolutional neural networks (CNNs) based SR models primarily focus on restoring high-frequency details, often overlooking crucial low-frequency contour information. Transformer-based SR methods, while incorporating global structural details, frequently come with an abundance of parameters, leading to high computational overhead. In this paper, we address these challenges by introducing a Multi-Depth Branches Network (MDBN). This framework extends the ResNet architecture by integrating an additional branch that captures vital structural characteristics of images. Our proposed multi-depth branches module (MDBM) involves the stacking of convolutional kernels of identical size at varying depths within distinct branches. By conducting a comprehensive analysis of the feature maps, we observe that branches with differing depths can extract contour and detail information respectively. By integrating these branches, the overall architecture can preserve essential low-frequency semantic structural information during the restoration of high-frequency visual elements, which is more closely with human visual cognition. Compared to GoogLeNet-like models, our basic multi-depth branches structure has fewer parameters, higher computational efficiency, and improved performance. Our model outperforms state-of-the-art (SOTA) lightweight SR methods with less inference time. Our code is available at https://github.com/thy960112/MDBN

{{</citation>}}


### (129/147) Development of a Deep Learning Method to Identify Acute Ischemic Stroke Lesions on Brain CT (Alessandro Fontanella et al., 2023)

{{<citation>}}

Alessandro Fontanella, Wenwen Li, Grant Mair, Antreas Antoniou, Eleanor Platt, Paul Armitage, Emanuele Trucco, Joanna Wardlaw, Amos Storkey. (2023)  
**Development of a Deep Learning Method to Identify Acute Ischemic Stroke Lesions on Brain CT**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV, physics-med-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.17320v1)  

---


**ABSTRACT**  
Computed Tomography (CT) is commonly used to image acute ischemic stroke (AIS) patients, but its interpretation by radiologists is time-consuming and subject to inter-observer variability. Deep learning (DL) techniques can provide automated CT brain scan assessment, but usually require annotated images. Aiming to develop a DL method for AIS using labelled but not annotated CT brain scans from patients with AIS, we designed a convolutional neural network-based DL algorithm using routinely-collected CT brain scans from the Third International Stroke Trial (IST-3), which were not acquired using strict research protocols. The DL model aimed to detect AIS lesions and classify the side of the brain affected. We explored the impact of AIS lesion features, background brain appearances, and timing on DL performance. From 5772 unique CT scans of 2347 AIS patients (median age 82), 54% had visible AIS lesions according to expert labelling. Our best-performing DL method achieved 72% accuracy for lesion presence and side. Lesions that were larger (80% accuracy) or multiple (87% accuracy for two lesions, 100% for three or more), were better detected. Follow-up scans had 76% accuracy, while baseline scans 67% accuracy. Chronic brain conditions reduced accuracy, particularly non-stroke lesions and old stroke lesions (32% and 31% error rates respectively). DL methods can be designed for AIS lesion detection on CT using the vast quantities of routinely-collected CT brain scan data. Ultimately, this should lead to more robust and widely-applicable methods.

{{</citation>}}


### (130/147) Glioma subtype classification from histopathological images using in-domain and out-of-domain transfer learning: An experimental study (Vladimir Despotovic et al., 2023)

{{<citation>}}

Vladimir Despotovic, Sang-Yoon Kim, Ann-Christin Hau, Aliaksandra Kakoichankava, Gilbert Georg Klamminger, Felix Bruno Kleine Borgmann, Katrin B. M. Frauenknecht, Michel Mittelbronn, Petr V. Nazarov. (2023)  
**Glioma subtype classification from histopathological images using in-domain and out-of-domain transfer learning: An experimental study**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.17223v1)  

---


**ABSTRACT**  
We provide in this paper a comprehensive comparison of various transfer learning strategies and deep learning architectures for computer-aided classification of adult-type diffuse gliomas. We evaluate the generalizability of out-of-domain ImageNet representations for a target domain of histopathological images, and study the impact of in-domain adaptation using self-supervised and multi-task learning approaches for pretraining the models using the medium-to-large scale datasets of histopathological images. A semi-supervised learning approach is furthermore proposed, where the fine-tuned models are utilized to predict the labels of unannotated regions of the whole slide images (WSI). The models are subsequently retrained using the ground-truth labels and weak labels determined in the previous step, providing superior performance in comparison to standard in-domain transfer learning with balanced accuracy of 96.91% and F1-score 97.07%, and minimizing the pathologist's efforts for annotation. Finally, we provide a visualization tool working at WSI level which generates heatmaps that highlight tumor areas; thus, providing insights to pathologists concerning the most informative parts of the WSI.

{{</citation>}}


## q-fin.GN (1)



### (131/147) Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT Sentiment Analysis (Paul Glasserman et al., 2023)

{{<citation>}}

Paul Glasserman, Caden Lin. (2023)  
**Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT Sentiment Analysis**  

---
Primary Category: q-fin.GN  
Categories: cs-AI, q-fin-GN, q-fin.GN  
Keywords: Bias, ChatGPT, GPT, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2309.17322v1)  

---


**ABSTRACT**  
Large language models (LLMs), including ChatGPT, can extract profitable trading signals from the sentiment in news text. However, backtesting such strategies poses a challenge because LLMs are trained on many years of data, and backtesting produces biased results if the training and backtesting periods overlap. This bias can take two forms: a look-ahead bias, in which the LLM may have specific knowledge of the stock returns that followed a news article, and a distraction effect, in which general knowledge of the companies named interferes with the measurement of a text's sentiment. We investigate these sources of bias through trading strategies driven by the sentiment of financial news headlines. We compare trading performance based on the original headlines with de-biased strategies in which we remove the relevant company's identifiers from the text. In-sample (within the LLM training window), we find, surprisingly, that the anonymized headlines outperform, indicating that the distraction effect has a greater impact than look-ahead bias. This tendency is particularly strong for larger companies--companies about which we expect an LLM to have greater general knowledge. Out-of-sample, look-ahead bias is not a concern but distraction remains possible. Our proposed anonymization procedure is therefore potentially useful in out-of-sample implementation, as well as for de-biased backtesting.

{{</citation>}}


## quant-ph (2)



### (132/147) Quantum Amplitude Estimation for Probabilistic Methods in Power Systems (Emilie Jong et al., 2023)

{{<citation>}}

Emilie Jong, Brynjar Sævarsson, Hjörtur Jóhannsson, Spyros Chatzivasileiadis. (2023)  
**Quantum Amplitude Estimation for Probabilistic Methods in Power Systems**  

---
Primary Category: quant-ph  
Categories: cs-SY, eess-SY, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.17299v1)  

---


**ABSTRACT**  
This paper introduces quantum computing methods for Monte Carlo simulations in power systems which are expected to be exponentially faster than their classical computing counterparts. Monte Carlo simulations is a fundamental method, widely used in power systems to estimate key parameters of unknown probability distributions, such as the mean value, the standard deviation, or the value at risk. It is, however, very computationally intensive. Approaches based on Quantum Amplitude Estimation can offer a quadratic speedup, requiring orders of magnitude less samples to achieve the same accuracy. This paper explains three Quantum Amplitude Estimation methods to replace the Classical Monte Carlo method, namely the Iterative Quantum Amplitude Estimation (IQAE), Maximum Likelihood Amplitude Estimation (MLAE), and Faster Amplitude Estimation (FAE), and compares their performance for three different types of probability distributions for power systems.

{{</citation>}}


### (133/147) A Quantum States Preparation Method Based on Difference-Driven Reinforcement Learning (Wenjie Liu et al., 2023)

{{<citation>}}

Wenjie Liu, Jing Xu, Bosi Wang. (2023)  
**A Quantum States Preparation Method Based on Difference-Driven Reinforcement Learning**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-ET, cs-LG, quant-ph, quant-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16972v1)  

---


**ABSTRACT**  
Due to the large state space of the two-qubit system, and the adoption of ladder reward function in the existing quantum state preparation methods, the convergence speed is slow and it is difficult to prepare the desired target quantum state with high fidelity under limited conditions. To solve the above problems, a difference-driven reinforcement learning (RL) algorithm for quantum state preparation of two-qubit system is proposed by improving the reward function and action selection strategy. Firstly, a model is constructed for the problem of preparing quantum states of a two-qubit system, with restrictions on the type of quantum gates and the time for quantum state evolution. In the preparation process, a weighted differential dynamic reward function is designed to assist the algorithm quickly obtain the maximum expected cumulative reward. Then, an adaptive e-greedy action selection strategy is adopted to achieve a balance between exploration and utilization to a certain extent, thereby improving the fidelity of the final quantum state. The simulation results show that the proposed algorithm can prepare quantum state with high fidelity under limited conditions. Compared with other algorithms, it has different degrees of improvement in convergence speed and fidelity of the final quantum state.

{{</citation>}}


## q-bio.QM (1)



### (134/147) AI-Aristotle: A Physics-Informed framework for Systems Biology Gray-Box Identification (Nazanin Ahmadi Daryakenari et al., 2023)

{{<citation>}}

Nazanin Ahmadi Daryakenari, Mario De Florio, Khemraj Shukla, George Em Karniadakis. (2023)  
**AI-Aristotle: A Physics-Informed framework for Systems Biology Gray-Box Identification**  

---
Primary Category: q-bio.QM  
Categories: 37N25 (Primary), 34-04 (Secondary), G-1-7; I-2-0, cs-AI, cs-LG, q-bio-QM, q-bio.QM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.01433v1)  

---


**ABSTRACT**  
Discovering mathematical equations that govern physical and biological systems from observed data is a fundamental challenge in scientific research. We present a new physics-informed framework for parameter estimation and missing physics identification (gray-box) in the field of Systems Biology. The proposed framework -- named AI-Aristotle -- combines eXtreme Theory of Functional Connections (X-TFC) domain-decomposition and Physics-Informed Neural Networks (PINNs) with symbolic regression (SR) techniques for parameter discovery and gray-box identification. We test the accuracy, speed, flexibility and robustness of AI-Aristotle based on two benchmark problems in Systems Biology: a pharmacokinetics drug absorption model, and an ultradian endocrine model for glucose-insulin interactions. We compare the two machine learning methods (X-TFC and PINNs), and moreover, we employ two different symbolic regression techniques to cross-verify our results. While the current work focuses on the performance of AI-Aristotle based on synthetic data, it can equally handle noisy experimental data and can even be used for black-box identification in just a few minutes on a laptop. More broadly, our work provides insights into the accuracy, cost, scalability, and robustness of integrating neural networks with symbolic regressors, offering a comprehensive guide for researchers tackling gray-box identification challenges in complex dynamical systems in biomedicine and beyond.

{{</citation>}}


## stat.ML (2)



### (135/147) Estimation and Inference in Distributional Reinforcement Learning (Liangyu Zhang et al., 2023)

{{<citation>}}

Liangyu Zhang, Yang Peng, Jiadong Liang, Wenhao Yang, Zhihua Zhang. (2023)  
**Estimation and Inference in Distributional Reinforcement Learning**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.17262v1)  

---


**ABSTRACT**  
In this paper, we study distributional reinforcement learning from the perspective of statistical efficiency.   We investigate distributional policy evaluation, aiming to estimate the complete distribution of the random return (denoted $\eta^\pi$) attained by a given policy $\pi$.   We use the certainty-equivalence method to construct our estimator $\hat\eta^\pi$, given a generative model is available.   We show that in this circumstance we need a dataset of size $\widetilde O\left(\frac{|\mathcal{S}||\mathcal{A}|}{\epsilon^{2p}(1-\gamma)^{2p+2}}\right)$ to guarantee a $p$-Wasserstein metric between $\hat\eta^\pi$ and $\eta^\pi$ is less than $\epsilon$ with high probability.   This implies the distributional policy evaluation problem can be solved with sample efficiency.   Also, we show that under different mild assumptions a dataset of size $\widetilde O\left(\frac{|\mathcal{S}||\mathcal{A}|}{\epsilon^{2}(1-\gamma)^{4}}\right)$ suffices to ensure the Kolmogorov metric and total variation metric between $\hat\eta^\pi$ and $\eta^\pi$ is below $\epsilon$ with high probability.   Furthermore, we investigate the asymptotic behavior of $\hat\eta^\pi$.   We demonstrate that the ``empirical process'' $\sqrt{n}(\hat\eta^\pi-\eta^\pi)$ converges weakly to a Gaussian process in the space of bounded functionals on Lipschitz function class $\ell^\infty(\mathcal{F}_{W_1})$, also in the space of bounded functionals on indicator function class $\ell^\infty(\mathcal{F}_{\mathrm{KS}})$ and bounded measurable function class $\ell^\infty(\mathcal{F}_{\mathrm{TV}})$ when some mild conditions hold.   Our findings give rise to a unified approach to statistical inference of a wide class of statistical functionals of $\eta^\pi$.

{{</citation>}}


### (136/147) Controlling Continuous Relaxation for Combinatorial Optimization (Yuma Ichikawa, 2023)

{{<citation>}}

Yuma Ichikawa. (2023)  
**Controlling Continuous Relaxation for Combinatorial Optimization**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-CO, stat-ME, stat-ML, stat.ML  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.16965v1)  

---


**ABSTRACT**  
Recent advancements in combinatorial optimization (CO) problems emphasize the potential of graph neural networks (GNNs). The physics-inspired GNN (PI-GNN) solver, which finds approximate solutions through unsupervised learning, has attracted significant attention for large-scale CO problems. Nevertheless, there has been limited discussion on the performance of the PI-GNN solver for CO problems on relatively dense graphs where the performance of greedy algorithms worsens. In addition, since the PI-GNN solver employs a relaxation strategy, an artificial transformation from the continuous space back to the original discrete space is necessary after learning, potentially undermining the robustness of the solutions. This paper numerically demonstrates that the PI-GNN solver can be trapped in a local solution, where all variables are zero, in the early stage of learning for CO problems on the dense graphs. Then, we address these problems by controlling the continuity and discreteness of relaxed variables while avoiding the local solution: (i) introducing a new penalty term that controls the continuity and discreteness of the relaxed variables and eliminates the local solution; (ii) proposing a new continuous relaxation annealing (CRA) strategy. This new annealing first prioritizes continuous solutions and intensifies exploration by leveraging the continuity while avoiding the local solution and then schedules the penalty term for prioritizing a discrete solution until the relaxed variables are almost discrete values, which eliminates the need for an artificial transformation from the continuous to the original discrete space. Empirically, better results are obtained for CO problems on the dense graphs, where the PI-GNN solver struggles to find reasonable solutions, and for those on relatively sparse graphs. Furthermore, the computational time scaling is identical to that of the PI-GNN solver.

{{</citation>}}


## cs.IT (2)



### (137/147) Meta Reinforcement Learning for Fast Spectrum Sharing in Vehicular Networks (Kai Huang et al., 2023)

{{<citation>}}

Kai Huang, Le Liang, Shi Jin, Geoffrey Ye Li. (2023)  
**Meta Reinforcement Learning for Fast Spectrum Sharing in Vehicular Networks**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.17185v1)  

---


**ABSTRACT**  
In this paper, we investigate the problem of fast spectrum sharing in vehicle-to-everything communication. In order to improve the spectrum efficiency of the whole system, the spectrum of vehicle-to-infrastructure links is reused by vehicle-to-vehicle links. To this end, we model it as a problem of deep reinforcement learning and tackle it with proximal policy optimization. A considerable number of interactions are often required for training an agent with good performance, so simulation-based training is commonly used in communication networks. Nevertheless, severe performance degradation may occur when the agent is directly deployed in the real world, even though it can perform well on the simulator, due to the reality gap between the simulation and the real environments. To address this issue, we make preliminary efforts by proposing an algorithm based on meta reinforcement learning. This algorithm enables the agent to rapidly adapt to a new task with the knowledge extracted from similar tasks, leading to fewer interactions and less training time. Numerical results show that our method achieves near-optimal performance and exhibits rapid convergence.

{{</citation>}}


### (138/147) Double-Layer Power Control for Mobile Cell-Free XL-MIMO with Multi-Agent Reinforcement Learning (Ziheng Liu et al., 2023)

{{<citation>}}

Ziheng Liu, Jiayi Zhang, Zhilong Liu, Huahua Xiao, Bo Ai. (2023)  
**Double-Layer Power Control for Mobile Cell-Free XL-MIMO with Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.17079v1)  

---


**ABSTRACT**  
Cell-free (CF) extremely large-scale multiple-input multiple-output (XL-MIMO) is regarded as a promising technology for enabling future wireless communication systems. Significant attention has been generated by its considerable advantages in augmenting degrees of freedom. In this paper, we first investigate a CF XL-MIMO system with base stations equipped with XL-MIMO panels under a dynamic environment. Then, we propose an innovative multi-agent reinforcement learning (MARL)-based power control algorithm that incorporates predictive management and distributed optimization architecture, which provides a dynamic strategy for addressing high-dimension signal processing problems. Specifically, we compare various MARL-based algorithms, which shows that the proposed MARL-based algorithm effectively strikes a balance between spectral efficiency (SE) performance and convergence time. Moreover, we consider a double-layer power control architecture based on the large-scale fading coefficients between antennas to suppress interference within dynamic systems. Compared to the single-layer architecture, the results obtained unveil that the proposed double-layer architecture has a nearly24% SE performance improvement, especially with massive antennas and smaller antenna spacing.

{{</citation>}}


## cs.MM (1)



### (139/147) Redistributing the Precision and Content in 3D-LUT-based Inverse Tone-mapping for HDR/WCG Display (Cheng Guo et al., 2023)

{{<citation>}}

Cheng Guo, Leidong Fan, Qian Zhang, Hanyuan Liu, Kanglin Liu, Xiuhua Jiang. (2023)  
**Redistributing the Precision and Content in 3D-LUT-based Inverse Tone-mapping for HDR/WCG Display**  

---
Primary Category: cs.MM  
Categories: cs-CV, cs-MM, cs.MM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.17160v1)  

---


**ABSTRACT**  
ITM(inverse tone-mapping) converts SDR (standard dynamic range) footage to HDR/WCG (high dynamic range /wide color gamut) for media production. It happens not only when remastering legacy SDR footage in front-end content provider, but also adapting on-theair SDR service on user-end HDR display. The latter requires more efficiency, thus the pre-calculated LUT (look-up table) has become a popular solution. Yet, conventional fixed LUT lacks adaptability, so we learn from research community and combine it with AI. Meanwhile, higher-bit-depth HDR/WCG requires larger LUT than SDR, so we consult traditional ITM for an efficiency-performance trade-off: We use 3 smaller LUTs, each has a non-uniform packing (precision) respectively denser in dark, middle and bright luma range. In this case, their results will have less error only in their own range, so we use a contribution map to combine their best parts to final result. With the guidance of this map, the elements (content) of 3 LUTs will also be redistributed during training. We conduct ablation studies to verify method's effectiveness, and subjective and objective experiments to show its practicability. Code is available at: https://github.com/AndreGuo/ITMLUT.

{{</citation>}}


## cs.SI (1)



### (140/147) SAppKG: Mobile App Recommendation Using Knowledge Graph and Side Information-A Secure Framework (Daksh Dave et al., 2023)

{{<citation>}}

Daksh Dave, Aditya Sharma, Shafii Muhammad Abdulhamid, Adeel Ahmed, Adnan Akhunzada, Rashid Amin. (2023)  
**SAppKG: Mobile App Recommendation Using Knowledge Graph and Side Information-A Secure Framework**  

---
Primary Category: cs.SI  
Categories: cs-IR, cs-SI, cs.SI  
Keywords: Google, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.17115v1)  

---


**ABSTRACT**  
Due to the rapid development of technology and the widespread usage of smartphones, the number of mobile applications is exponentially growing. Finding a suitable collection of apps that aligns with users needs and preferences can be challenging. However, mobile app recommender systems have emerged as a helpful tool in simplifying this process. But there is a drawback to employing app recommender systems. These systems need access to user data, which is a serious security violation. While users seek accurate opinions, they do not want to compromise their privacy in the process. We address this issue by developing SAppKG, an end-to-end user privacy-preserving knowledge graph architecture for mobile app recommendation based on knowledge graph models such as SAppKG-S and SAppKG-D, that utilized the interaction data and side information of app attributes. We tested the proposed model on real-world data from the Google Play app store, using precision, recall, mean absolute precision, and mean reciprocal rank. We found that the proposed model improved results on all four metrics. We also compared the proposed model to baseline models and found that it outperformed them on all four metrics.

{{</citation>}}


## hep-lat (1)



### (141/147) Diffusion Models as Stochastic Quantization in Lattice Field Theory (Lingxiao Wang et al., 2023)

{{<citation>}}

Lingxiao Wang, Gert Aarts, Kai Zhou. (2023)  
**Diffusion Models as Stochastic Quantization in Lattice Field Theory**  

---
Primary Category: hep-lat  
Categories: cs-LG, hep-lat, hep-lat  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2309.17082v1)  

---


**ABSTRACT**  
In this work, we establish a direct connection between generative diffusion models (DMs) and stochastic quantization (SQ). The DM is realized by approximating the reversal of a stochastic process dictated by the Langevin equation, generating samples from a prior distribution to effectively mimic the target distribution. Using numerical simulations, we demonstrate that the DM can serve as a global sampler for generating quantum lattice field configurations in two-dimensional $\phi^4$ theory. We demonstrate that DMs can notably reduce autocorrelation times in the Markov chain, especially in the critical region where standard Markov Chain Monte-Carlo (MCMC) algorithms experience critical slowing down. The findings can potentially inspire further advancements in lattice field theory simulations, in particular in cases where it is expensive to generate large ensembles.

{{</citation>}}


## cs.IR (1)



### (142/147) Aligning the Capabilities of Large Language Models with the Context of Information Retrieval via Contrastive Feedback (Qian Dong et al., 2023)

{{<citation>}}

Qian Dong, Yiding Liu, Qingyao Ai, Zhijing Wu, Haitao Li, Yiqun Liu, Shuaiqiang Wang, Dawei Yin, Shaoping Ma. (2023)  
**Aligning the Capabilities of Large Language Models with the Context of Information Retrieval via Contrastive Feedback**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.17078v1)  

---


**ABSTRACT**  
Information Retrieval (IR), the process of finding information to satisfy user's information needs, plays an essential role in modern people's lives. Recently, large language models (LLMs) have demonstrated remarkable capabilities across various tasks, some of which are important for IR. Nonetheless, LLMs frequently confront the issue of generating responses that lack specificity. This has limited the overall effectiveness of LLMs for IR in many cases. To address these issues, we present an unsupervised alignment framework called Reinforcement Learning from Contrastive Feedback (RLCF), which empowers LLMs to generate both high-quality and context-specific responses that suit the needs of IR tasks. Specifically, we construct contrastive feedback by comparing each document with its similar documents, and then propose a reward function named Batched-MRR to teach LLMs to generate responses that captures the fine-grained information that distinguish documents from their similar ones. To demonstrate the effectiveness of RLCF, we conducted experiments in two typical applications of LLMs in IR, i.e., data augmentation and summarization. The experimental results show that RLCF can effectively improve the performance of LLMs in IR context.

{{</citation>}}


## cs.DB (1)



### (143/147) MaaSDB: Spatial Databases in the Era of Large Language Models (Vision Paper) (Jianzhong Qi et al., 2023)

{{<citation>}}

Jianzhong Qi, Zuqing Li, Egemen Tanin. (2023)  
**MaaSDB: Spatial Databases in the Era of Large Language Models (Vision Paper)**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.17072v1)  

---


**ABSTRACT**  
Large language models (LLMs) are advancing rapidly. Such models have demonstrated strong capabilities in learning from large-scale (unstructured) text data and answering user queries. Users do not need to be experts in structured query languages to interact with systems built upon such models. This provides great opportunities to reduce the barrier of information retrieval for the general public. By introducing LLMs into spatial data management, we envisage an LLM-based spatial database system to learn from both structured and unstructured spatial data. Such a system will offer seamless access to spatial knowledge for the users, thus benefiting individuals, business, and government policy makers alike.

{{</citation>}}


## eess.AS (2)



### (144/147) Low-Resource Self-Supervised Learning with SSL-Enhanced TTS (Po-chun Hsu et al., 2023)

{{<citation>}}

Po-chun Hsu, Ali Elkahky, Wei-Ning Hsu, Yossi Adi, Tu Anh Nguyen, Jade Copet, Emmanuel Dupoux, Hung-yi Lee, Abdelrahman Mohamed. (2023)  
**Low-Resource Self-Supervised Learning with SSL-Enhanced TTS**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Low-Resource, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.17020v1)  

---


**ABSTRACT**  
Self-supervised learning (SSL) techniques have achieved remarkable results in various speech processing tasks. Nonetheless, a significant challenge remains in reducing the reliance on vast amounts of speech data for pre-training. This paper proposes to address this challenge by leveraging synthetic speech to augment a low-resource pre-training corpus. We construct a high-quality text-to-speech (TTS) system with limited resources using SSL features and generate a large synthetic corpus for pre-training. Experimental results demonstrate that our proposed approach effectively reduces the demand for speech data by 90\% with only slight performance degradation. To the best of our knowledge, this is the first work aiming to enhance low-resource self-supervised learning in speech processing.

{{</citation>}}


### (145/147) Enhancing Code-switching Speech Recognition with Interactive Language Biases (Hexin Liu et al., 2023)

{{<citation>}}

Hexin Liu, Leibny Paola Garcia, Xiangyu Zhang, Andy W. H. Khong, Sanjeev Khudanpur. (2023)  
**Enhancing Code-switching Speech Recognition with Interactive Language Biases**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Bias, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.16953v1)  

---


**ABSTRACT**  
Languages usually switch within a multilingual speech signal, especially in a bilingual society. This phenomenon is referred to as code-switching (CS), making automatic speech recognition (ASR) challenging under a multilingual scenario. We propose to improve CS-ASR by biasing the hybrid CTC/attention ASR model with multi-level language information comprising frame- and token-level language posteriors. The interaction between various resolutions of language biases is subsequently explored in this work. We conducted experiments on datasets from the ASRU 2019 code-switching challenge. Compared to the baseline, the proposed interactive language biases (ILB) method achieves higher performance and ablation studies highlight the effects of different language biases and their interactions. In addition, the results presented indicate that language bias implicitly enhances internal language modeling, leading to performance degradation after employing an external language model.

{{</citation>}}


## eess.SY (1)



### (146/147) Reliability Quantification of Deep Reinforcement Learning-based Control (Hitoshi Yoshioka et al., 2023)

{{<citation>}}

Hitoshi Yoshioka, Hirotada Hashimoto. (2023)  
**Reliability Quantification of Deep Reinforcement Learning-based Control**  

---
Primary Category: eess.SY  
Categories: 68T40 Artificial intelligence for robotics, cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.16977v1)  

---


**ABSTRACT**  
Reliability quantification of deep reinforcement learning (DRL)-based control is a significant challenge for the practical application of artificial intelligence (AI) in safety-critical systems. This study proposes a method for quantifying the reliability of DRL-based control. First, an existing method, random noise distillation, was applied to the reliability evaluation to clarify the issues to be solved. Second, a novel method for reliability quantification was proposed to solve these issues. The reliability is quantified using two neural networks: reference and evaluator. They have the same structure with the same initial parameters. The outputs of the two networks were the same before training. During training, the evaluator network parameters were updated to maximize the difference between the reference and evaluator networks for trained data. Thus, the reliability of the DRL-based control for a state can be evaluated based on the difference in output between the two networks. The proposed method was applied to DQN-based control as an example of a simple task, and its effectiveness was demonstrated. Finally, the proposed method was applied to the problem of switching trained models depending on the state. Con-sequently, the performance of the DRL-based control was improved by switching the trained models according to their reliability.

{{</citation>}}


## cs.DC (1)



### (147/147) Lifting the Fog of Uncertainties: Dynamic Resource Orchestration for the Containerized Cloud (Yuqiu Zhang et al., 2023)

{{<citation>}}

Yuqiu Zhang, Tongkun Zhang, Gengrui Zhang, Hans-Arno Jacobsen. (2023)  
**Lifting the Fog of Uncertainties: Dynamic Resource Orchestration for the Containerized Cloud**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2309.16962v1)  

---


**ABSTRACT**  
The advances in virtualization technologies have sparked a growing transition from virtual machine (VM)-based to container-based infrastructure for cloud computing. From the resource orchestration perspective, containers' lightweight and highly configurable nature not only enables opportunities for more optimized strategies, but also poses greater challenges due to additional uncertainties and a larger configuration parameter search space. Towards this end, we propose Drone, a resource orchestration framework that adaptively configures resource parameters to improve application performance and reduce operational cost in the presence of cloud uncertainties. Built on Contextual Bandit techniques, Drone is able to achieve a balance between performance and resource cost on public clouds, and optimize performance on private clouds where a hard resource constraint is present. We show that our algorithms can achieve sub-linear growth in cumulative regret, a theoretically sound convergence guarantee, and our extensive experiments show that Drone achieves an up to 45% performance improvement and a 20% resource footprint reduction across batch processing jobs and microservice workloads.

{{</citation>}}
