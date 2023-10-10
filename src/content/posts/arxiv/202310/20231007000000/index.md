---
draft: false
title: "arXiv @ 2023.10.07"
date: 2023-10-07
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.07"
    identifier: arxiv_20231007
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (35)](#cslg-35)
- [cs.CV (20)](#cscv-20)
- [cs.SD (2)](#cssd-2)
- [cs.RO (4)](#csro-4)
- [cs.IR (3)](#csir-3)
- [cs.CL (30)](#cscl-30)
- [cs.SI (2)](#cssi-2)
- [eess.AS (1)](#eessas-1)
- [physics.med-ph (1)](#physicsmed-ph-1)
- [cs.CR (5)](#cscr-5)
- [eess.IV (3)](#eessiv-3)
- [cs.FL (1)](#csfl-1)
- [eess.SY (3)](#eesssy-3)
- [cs.AI (4)](#csai-4)
- [cs.HC (2)](#cshc-2)
- [cs.SE (2)](#csse-2)
- [cs.CY (1)](#cscy-1)
- [q-bio.BM (1)](#q-biobm-1)
- [stat.ML (1)](#statml-1)
- [math.OC (1)](#mathoc-1)

## cs.LG (35)



### (1/122) LaTeX: Language Pattern-aware Triggering Event Detection for Adverse Experience during Pandemics (Kaiqun Fu et al., 2023)

{{<citation>}}

Kaiqun Fu, Yangxiao Bai, Weiwei Zhang, Deepthi Kolady. (2023)  
**LaTeX: Language Pattern-aware Triggering Event Detection for Adverse Experience during Pandemics**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Event Detection, Twitter  
[Paper Link](http://arxiv.org/abs/2310.03941v1)  

---


**ABSTRACT**  
The COVID-19 pandemic has accentuated socioeconomic disparities across various racial and ethnic groups in the United States. While previous studies have utilized traditional survey methods like the Household Pulse Survey (HPS) to elucidate these disparities, this paper explores the role of social media platforms in both highlighting and addressing these challenges. Drawing from real-time data sourced from Twitter, we analyzed language patterns related to four major types of adverse experiences: loss of employment income (LI), food scarcity (FS), housing insecurity (HI), and unmet needs for mental health services (UM). We first formulate a sparsity optimization problem that extracts low-level language features from social media data sources. Second, we propose novel constraints on feature similarity exploiting prior knowledge about the similarity of the language patterns among the adverse experiences. The proposed problem is challenging to solve due to the non-convexity objective and non-smoothness penalties. We develop an algorithm based on the alternating direction method of multipliers (ADMM) framework to solve the proposed formulation. Extensive experiments and comparisons to other models on real-world social media and the detection of adverse experiences justify the efficacy of our model.

{{</citation>}}


### (2/122) Multitask Learning for Time Series Data\\with 2D Convolution (Chin-Chia Michael Yeh et al., 2023)

{{<citation>}}

Chin-Chia Michael Yeh, Xin Dai, Yan Zheng, Junpeng Wang, Huiyuan Chen, Yujie Fan, Audrey Der, Zhongfang Zhuang, Liang Wang, Wei Zhang. (2023)  
**Multitask Learning for Time Series Data\\with 2D Convolution**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.03925v1)  

---


**ABSTRACT**  
Multitask learning (MTL) aims to develop a unified model that can handle a set of closely related tasks simultaneously. By optimizing the model across multiple tasks, MTL generally surpasses its non-MTL counterparts in terms of generalizability. Although MTL has been extensively researched in various domains such as computer vision, natural language processing, and recommendation systems, its application to time series data has received limited attention. In this paper, we investigate the application of MTL to the time series classification (TSC) problem. However, when we integrate the state-of-the-art 1D convolution-based TSC model with MTL, the performance of the TSC model actually deteriorates. By comparing the 1D convolution-based models with the Dynamic Time Warping (DTW) distance function, it appears that the underwhelming results stem from the limited expressive power of the 1D convolutional layers. To overcome this challenge, we propose a novel design for a 2D convolution-based model that enhances the model's expressiveness. Leveraging this advantage, our proposed method outperforms competing approaches on both the UCR Archive and an industrial transaction TSC dataset.

{{</citation>}}


### (3/122) Toward a Foundation Model for Time Series Data (Chin-Chia Michael Yeh et al., 2023)

{{<citation>}}

Chin-Chia Michael Yeh, Xin Dai, Huiyuan Chen, Yan Zheng, Yujie Fan, Audrey Der, Vivian Lai, Zhongfang Zhuang, Junpeng Wang, Liang Wang, Wei Zhang. (2023)  
**Toward a Foundation Model for Time Series Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2310.03916v1)  

---


**ABSTRACT**  
A foundation model is a machine learning model trained on a large and diverse set of data, typically using self-supervised learning-based pre-training techniques, that can be adapted to various downstream tasks. However, current research on time series pre-training has mostly focused on models pre-trained solely on data from a single domain, resulting in a lack of knowledge about other types of time series. However, current research on time series pre-training has predominantly focused on models trained exclusively on data from a single domain. As a result, these models possess domain-specific knowledge that may not be easily transferable to time series from other domains. In this paper, we aim to develop an effective time series foundation model by leveraging unlabeled samples from multiple domains. To achieve this, we repurposed the publicly available UCR Archive and evaluated four existing self-supervised learning-based pre-training methods, along with a novel method, on the datasets. We tested these methods using four popular neural network architectures for time series to understand how the pre-training methods interact with different network designs. Our experimental results show that pre-training improves downstream classification tasks by enhancing the convergence of the fine-tuning process. Furthermore, we found that the proposed pre-training method, when combined with the Transformer model, outperforms the alternatives.

{{</citation>}}


### (4/122) RTDK-BO: High Dimensional Bayesian Optimization with Reinforced Transformer Deep kernels (Alexander Shmakov et al., 2023)

{{<citation>}}

Alexander Shmakov, Avisek Naug, Vineet Gundecha, Sahand Ghorbanpour, Ricardo Luna Gutierrez, Ashwin Ramesh Babu, Antonio Guillen, Soumyendu Sarkar. (2023)  
**RTDK-BO: High Dimensional Bayesian Optimization with Reinforced Transformer Deep kernels**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2310.03912v2)  

---


**ABSTRACT**  
Bayesian Optimization (BO), guided by Gaussian process (GP) surrogates, has proven to be an invaluable technique for efficient, high-dimensional, black-box optimization, a critical problem inherent to many applications such as industrial design and scientific computing. Recent contributions have introduced reinforcement learning (RL) to improve the optimization performance on both single function optimization and \textit{few-shot} multi-objective optimization. However, even few-shot techniques fail to exploit similarities shared between closely related objectives. In this paper, we combine recent developments in Deep Kernel Learning (DKL) and attention-based Transformer models to improve the modeling powers of GP surrogates with meta-learning. We propose a novel method for improving meta-learning BO surrogates by incorporating attention mechanisms into DKL, empowering the surrogates to adapt to contextual information gathered during the BO process. We combine this Transformer Deep Kernel with a learned acquisition function trained with continuous Soft Actor-Critic Reinforcement Learning to aid in exploration. This Reinforced Transformer Deep Kernel (RTDK-BO) approach yields state-of-the-art results in continuous high-dimensional optimization problems.

{{</citation>}}


### (5/122) PyDCM: Custom Data Center Models with Reinforcement Learning for Sustainability (Avisek Naug et al., 2023)

{{<citation>}}

Avisek Naug, Antonio Guillen, Ricardo Luna Gutiérrez, Vineet Gundecha, Dejan Markovikj, Lekhapriya Dheeraj Kashyap, Lorenz Krause, Sahand Ghorbanpour, Sajad Mousavi, Ashwin Ramesh Babu, Soumyendu Sarkar. (2023)  
**PyDCM: Custom Data Center Models with Reinforcement Learning for Sustainability**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03906v2)  

---


**ABSTRACT**  
The increasing global emphasis on sustainability and reducing carbon emissions is pushing governments and corporations to rethink their approach to data center design and operation. Given their high energy consumption and exponentially large computational workloads, data centers are prime candidates for optimizing power consumption, especially in areas such as cooling and IT energy usage. A significant challenge in this pursuit is the lack of a configurable and scalable thermal data center model that offers an end-to-end pipeline. Data centers consist of multiple IT components whose geometric configuration and heat dissipation make thermal modeling difficult. This paper presents PyDCM, a customizable Data Center Model implemented in Python, that allows users to create unique configurations of IT equipment with custom server specifications and geometric arrangements of IT cabinets. The use of vectorized thermal calculations makes PyDCM orders of magnitude faster (30 times) than current Energy Plus modeling implementations and scales sublinearly with the number of CPUs. Also, PyDCM enables the use of Deep Reinforcement Learning via the Gymnasium wrapper to optimize data center cooling and offers a user-friendly platform for testing various data center design prototypes.

{{</citation>}}


### (6/122) CrysFormer: Protein Structure Prediction via 3d Patterson Maps and Partial Structure Attention (Chen Dun et al., 2023)

{{<citation>}}

Chen Dun, Qiutai Pan, Shikai Jin, Ria Stevens, Mitchell D. Miller, George N. Phillips, Jr., Anastasios Kyrillidis. (2023)  
**CrysFormer: Protein Structure Prediction via 3d Patterson Maps and Partial Structure Attention**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.03899v1)  

---


**ABSTRACT**  
Determining the structure of a protein has been a decades-long open question. A protein's three-dimensional structure often poses nontrivial computation costs, when classical simulation algorithms are utilized. Advances in the transformer neural network architecture -- such as AlphaFold2 -- achieve significant improvements for this problem, by learning from a large dataset of sequence information and corresponding protein structures. Yet, such methods only focus on sequence information; other available prior knowledge, such as protein crystallography and partial structure of amino acids, could be potentially utilized. To the best of our knowledge, we propose the first transformer-based model that directly utilizes protein crystallography and partial structure information to predict the electron density maps of proteins. Via two new datasets of peptide fragments (2-residue and 15-residue) , we demonstrate our method, dubbed \texttt{CrysFormer}, can achieve accurate predictions, based on a much smaller dataset size and with reduced computation costs.

{{</citation>}}


### (7/122) Taming Binarized Neural Networks and Mixed-Integer Programs (Johannes Aspman et al., 2023)

{{<citation>}}

Johannes Aspman, Georgios Korpas, Jakub Marecek. (2023)  
**Taming Binarized Neural Networks and Mixed-Integer Programs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-OC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04469v1)  

---


**ABSTRACT**  
There has been a great deal of recent interest in binarized neural networks, especially because of their explainability. At the same time, automatic differentiation algorithms such as backpropagation fail for binarized neural networks, which limits their applicability. By reformulating the problem of training binarized neural networks as a subadditive dual of a mixed-integer program, we show that binarized neural networks admit a tame representation. This, in turn, makes it possible to use the framework of Bolte et al. for implicit differentiation, which offers the possibility for practical implementation of backpropagation in the context of binarized neural networks. This approach could also be used for a broader class of mixed-integer programs, beyond the training of binarized neural networks, as encountered in symbolic approaches to AI and beyond.

{{</citation>}}


### (8/122) Design Principles for Lifelong Learning AI Accelerators (Dhireesha Kudithipudi et al., 2023)

{{<citation>}}

Dhireesha Kudithipudi, Anurag Daram, Abdullah M. Zyarah, Fatima Tuz Zohora, James B. Aimone, Angel Yanguas-Gil, Nicholas Soures, Emre Neftci, Matthew Mattina, Vincenzo Lomonaco, Clare D. Thiem, Benjamin Epstein. (2023)  
**Design Principles for Lifelong Learning AI Accelerators**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04467v1)  

---


**ABSTRACT**  
Lifelong learning - an agent's ability to learn throughout its lifetime - is a hallmark of biological learning systems and a central challenge for artificial intelligence (AI). The development of lifelong learning algorithms could lead to a range of novel AI applications, but this will also require the development of appropriate hardware accelerators, particularly if the models are to be deployed on edge platforms, which have strict size, weight, and power constraints. Here, we explore the design of lifelong learning AI accelerators that are intended for deployment in untethered environments. We identify key desirable capabilities for lifelong learning accelerators and highlight metrics to evaluate such accelerators. We then discuss current edge AI accelerators and explore the future design of lifelong learning accelerators, considering the role that different emerging technologies could play.

{{</citation>}}


### (9/122) Fishnets: Information-Optimal, Scalable Aggregation for Sets and Graphs (T. Lucas Makinen et al., 2023)

{{<citation>}}

T. Lucas Makinen, Justin Alsing, Benjamin D. Wandelt. (2023)  
**Fishnets: Information-Optimal, Scalable Aggregation for Sets and Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.03812v1)  

---


**ABSTRACT**  
Set-based learning is an essential component of modern deep learning and network science. Graph Neural Networks (GNNs) and their edge-free counterparts Deepsets have proven remarkably useful on ragged and topologically challenging datasets. The key to learning informative embeddings for set members is a specified aggregation function, usually a sum, max, or mean. We propose Fishnets, an aggregation strategy for learning information-optimal embeddings for sets of data for both Bayesian inference and graph aggregation. We demonstrate that i) Fishnets neural summaries can be scaled optimally to an arbitrary number of data objects, ii) Fishnets aggregations are robust to changes in data distribution, unlike standard deepsets, iii) Fishnets saturate Bayesian information content and extend to regimes where MCMC techniques fail and iv) Fishnets can be used as a drop-in aggregation scheme within GNNs. We show that by adopting a Fishnets aggregation scheme for message passing, GNNs can achieve state-of-the-art performance versus architecture size on ogbn-protein data over existing benchmarks with a fraction of learnable parameters and faster training time.

{{</citation>}}


### (10/122) Constraint-Conditioned Policy Optimization for Versatile Safe Reinforcement Learning (Yihang Yao et al., 2023)

{{<citation>}}

Yihang Yao, Zuxin Liu, Zhepeng Cen, Jiacheng Zhu, Wenhao Yu, Tingnan Zhang, Ding Zhao. (2023)  
**Constraint-Conditioned Policy Optimization for Versatile Safe Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03718v1)  

---


**ABSTRACT**  
Safe reinforcement learning (RL) focuses on training reward-maximizing agents subject to pre-defined safety constraints. Yet, learning versatile safe policies that can adapt to varying safety constraint requirements during deployment without retraining remains a largely unexplored and challenging area. In this work, we formulate the versatile safe RL problem and consider two primary requirements: training efficiency and zero-shot adaptation capability. To address them, we introduce the Conditioned Constrained Policy Optimization (CCPO) framework, consisting of two key modules: (1) Versatile Value Estimation (VVE) for approximating value functions under unseen threshold conditions, and (2) Conditioned Variational Inference (CVI) for encoding arbitrary constraint thresholds during policy optimization. Our extensive experiments demonstrate that CCPO outperforms the baselines in terms of safety and task performance while preserving zero-shot adaptation capabilities to different constraint thresholds data-efficiently. This makes our approach suitable for real-world dynamic applications.

{{</citation>}}


### (11/122) OMG-ATTACK: Self-Supervised On-Manifold Generation of Transferable Evasion Attacks (Ofir Bar Tal et al., 2023)

{{<citation>}}

Ofir Bar Tal, Adi Haviv, Amit H. Bermano. (2023)  
**OMG-ATTACK: Self-Supervised On-Manifold Generation of Transferable Evasion Attacks**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.03707v1)  

---


**ABSTRACT**  
Evasion Attacks (EA) are used to test the robustness of trained neural networks by distorting input data to misguide the model into incorrect classifications. Creating these attacks is a challenging task, especially with the ever-increasing complexity of models and datasets. In this work, we introduce a self-supervised, computationally economical method for generating adversarial examples, designed for the unseen black-box setting. Adapting techniques from representation learning, our method generates on-manifold EAs that are encouraged to resemble the data distribution. These attacks are comparable in effectiveness compared to the state-of-the-art when attacking the model trained on, but are significantly more effective when attacking unseen models, as the attacks are more related to the data rather than the model itself. Our experiments consistently demonstrate the method is effective across various models, unseen data categories, and even defended models, suggesting a significant role for on-manifold EAs when targeting unseen models.

{{</citation>}}


### (12/122) SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks (Alexander Robey et al., 2023)

{{<citation>}}

Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas. (2023)  
**SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: GPT, Language Model, PaLM  
[Paper Link](http://arxiv.org/abs/2310.03684v1)  

---


**ABSTRACT**  
Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.

{{</citation>}}


### (13/122) Rethinking Fairness for Human-AI Collaboration (Haosen Ge et al., 2023)

{{<citation>}}

Haosen Ge, Hamsa Bastani, Osbert Bastani. (2023)  
**Rethinking Fairness for Human-AI Collaboration**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03647v1)  

---


**ABSTRACT**  
Existing approaches to algorithmic fairness aim to ensure equitable outcomes if human decision-makers comply perfectly with algorithmic decisions. However, perfect compliance with the algorithm is rarely a reality or even a desirable outcome in human-AI collaboration. Yet, recent studies have shown that selective compliance with fair algorithms can amplify discrimination relative to the prior human policy. As a consequence, ensuring equitable outcomes requires fundamentally different algorithmic design principles that ensure robustness to the decision-maker's (a priori unknown) compliance pattern. We define the notion of compliance-robustly fair algorithmic recommendations that are guaranteed to (weakly) improve fairness in decisions, regardless of the human's compliance pattern. We propose a simple optimization strategy to identify the best performance-improving compliance-robustly fair policy. However, we show that it may be infeasible to design algorithmic recommendations that are simultaneously fair in isolation, compliance-robustly fair, and more accurate than the human policy; thus, if our goal is to improve the equity and accuracy of human-AI collaboration, it may not be desirable to enforce traditional fairness constraints.

{{</citation>}}


### (14/122) Adversarial Machine Learning for Social Good: Reframing the Adversary as an Ally (Shawqi Al-Maliki et al., 2023)

{{<citation>}}

Shawqi Al-Maliki, Adnan Qayyum, Hassan Ali, Mohamed Abdallah, Junaid Qadir, Dinh Thai Hoang, Dusit Niyato, Ala Al-Fuqaha. (2023)  
**Adversarial Machine Learning for Social Good: Reframing the Adversary as an Ally**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI, ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03614v1)  

---


**ABSTRACT**  
Deep Neural Networks (DNNs) have been the driving force behind many of the recent advances in machine learning. However, research has shown that DNNs are vulnerable to adversarial examples -- input samples that have been perturbed to force DNN-based models to make errors. As a result, Adversarial Machine Learning (AdvML) has gained a lot of attention, and researchers have investigated these vulnerabilities in various settings and modalities. In addition, DNNs have also been found to incorporate embedded bias and often produce unexplainable predictions, which can result in anti-social AI applications. The emergence of new AI technologies that leverage Large Language Models (LLMs), such as ChatGPT and GPT-4, increases the risk of producing anti-social applications at scale. AdvML for Social Good (AdvML4G) is an emerging field that repurposes the AdvML bug to invent pro-social applications. Regulators, practitioners, and researchers should collaborate to encourage the development of pro-social applications and hinder the development of anti-social ones. In this work, we provide the first comprehensive review of the emerging field of AdvML4G. This paper encompasses a taxonomy that highlights the emergence of AdvML4G, a discussion of the differences and similarities between AdvML4G and AdvML, a taxonomy covering social good-related concepts and aspects, an exploration of the motivations behind the emergence of AdvML4G at the intersection of ML4G and AdvML, and an extensive summary of the works that utilize AdvML4G as an auxiliary tool for innovating pro-social applications. Finally, we elaborate upon various challenges and open research issues that require significant attention from the research community.

{{</citation>}}


### (15/122) GENER: A Parallel Layer Deep Learning Network To Detect Gene-Gene Interactions From Gene Expression Data (Ahmed Fakhry et al., 2023)

{{<citation>}}

Ahmed Fakhry, Raneem Khafagy, Adriaan-Alexander Ludl. (2023)  
**GENER: A Parallel Layer Deep Learning Network To Detect Gene-Gene Interactions From Gene Expression Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2310.03611v2)  

---


**ABSTRACT**  
Detecting and discovering new gene interactions based on known gene expressions and gene interaction data presents a significant challenge. Various statistical and deep learning methods have attempted to tackle this challenge by leveraging the topological structure of gene interactions and gene expression patterns to predict novel gene interactions. In contrast, some approaches have focused exclusively on utilizing gene expression profiles. In this context, we introduce GENER, a parallel-layer deep learning network designed exclusively for the identification of gene-gene relationships using gene expression data. We conducted two training experiments and compared the performance of our network with that of existing statistical and deep learning approaches. Notably, our model achieved an average AUROC score of 0.834 on the combined BioGRID&DREAM5 dataset, outperforming competing methods in predicting gene-gene interactions.

{{</citation>}}


### (16/122) Comparing Time-Series Analysis Approaches Utilized in Research Papers to Forecast COVID-19 Cases in Africa: A Literature Review (Ali Ebadi et al., 2023)

{{<citation>}}

Ali Ebadi, Ebrahim Sahafizadeh. (2023)  
**Comparing Time-Series Analysis Approaches Utilized in Research Papers to Forecast COVID-19 Cases in Africa: A Literature Review**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.03606v1)  

---


**ABSTRACT**  
This literature review aimed to compare various time-series analysis approaches utilized in forecasting COVID-19 cases in Africa. The study involved a methodical search for English-language research papers published between January 2020 and July 2023, focusing specifically on papers that utilized time-series analysis approaches on COVID-19 datasets in Africa. A variety of databases including PubMed, Google Scholar, Scopus, and Web of Science were utilized for this process. The research papers underwent an evaluation process to extract relevant information regarding the implementation and performance of the time-series analysis models. The study highlighted the different methodologies employed, evaluating their effectiveness and limitations in forecasting the spread of the virus. The result of this review could contribute deeper insights into the field, and future research should consider these insights to improve time series analysis models and explore the integration of different approaches for enhanced public health decision-making.

{{</citation>}}


### (17/122) TimeGPT-1 (Azul Garza et al., 2023)

{{<citation>}}

Azul Garza, Max Mergenthaler-Canseco. (2023)  
**TimeGPT-1**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.03589v1)  

---


**ABSTRACT**  
In this paper, we introduce TimeGPT, the first foundation model for time series, capable of generating accurate predictions for diverse datasets not seen during training. We evaluate our pre-trained model against established statistical, machine learning, and deep learning methods, demonstrating that TimeGPT zero-shot inference excels in performance, efficiency, and simplicity. Our study provides compelling evidence that insights from other domains of artificial intelligence can be effectively applied to time series analysis. We conclude that large-scale time series models offer an exciting opportunity to democratize access to precise predictions and reduce uncertainty by leveraging the capabilities of contemporary advancements in deep learning.

{{</citation>}}


### (18/122) Targeted Adversarial Attacks on Generalizable Neural Radiance Fields (Andras Horvath et al., 2023)

{{<citation>}}

Andras Horvath, Csaba M. Jozsa. (2023)  
**Targeted Adversarial Attacks on Generalizable Neural Radiance Fields**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2310.03578v1)  

---


**ABSTRACT**  
Neural Radiance Fields (NeRFs) have recently emerged as a powerful tool for 3D scene representation and rendering. These data-driven models can learn to synthesize high-quality images from sparse 2D observations, enabling realistic and interactive scene reconstructions. However, the growing usage of NeRFs in critical applications such as augmented reality, robotics, and virtual environments could be threatened by adversarial attacks.   In this paper we present how generalizable NeRFs can be attacked by both low-intensity adversarial attacks and adversarial patches, where the later could be robust enough to be used in real world applications. We also demonstrate targeted attacks, where a specific, predefined output scene is generated by these attack with success.

{{</citation>}}


### (19/122) Neural Language Model Pruning for Automatic Speech Recognition (Leonardo Emili et al., 2023)

{{<citation>}}

Leonardo Emili, Thiago Fraga-Silva, Ernest Pusateri, Markus Nußbaum-Thom, Youssef Oualil. (2023)  
**Neural Language Model Pruning for Automatic Speech Recognition**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model, Pruning, Speech Recognition, Transformer  
[Paper Link](http://arxiv.org/abs/2310.03424v1)  

---


**ABSTRACT**  
We study model pruning methods applied to Transformer-based neural network language models for automatic speech recognition. We explore three aspects of the pruning frame work, namely criterion, method and scheduler, analyzing their contribution in terms of accuracy and inference speed. To the best of our knowledge, such in-depth analyses on large-scale recognition systems has not been reported in the literature. In addition, we propose a variant of low-rank approximation suitable for incrementally compressing models, and delivering multiple models with varied target sizes. Among other results, we show that a) data-driven pruning outperforms magnitude-driven in several scenarios; b) incremental pruning achieves higher accuracy compared to one-shot pruning, especially when targeting smaller sizes; and c) low-rank approximation presents the best trade-off between size reduction and inference speed-up for moderate compression.

{{</citation>}}


### (20/122) Adapting Large Language Models for Content Moderation: Pitfalls in Data Engineering and Supervised Fine-tuning (Huan Ma et al., 2023)

{{<citation>}}

Huan Ma, Changqing Zhang, Huazhu Fu, Peilin Zhao, Bingzhe Wu. (2023)  
**Adapting Large Language Models for Content Moderation: Pitfalls in Data Engineering and Supervised Fine-tuning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.03400v1)  

---


**ABSTRACT**  
Nowadays, billions of people engage in communication and express their opinions on the internet daily. Unfortunately, not all of these expressions are friendly or compliant, making content moderation an indispensable task. With the successful development of Large Language Models (LLMs) in recent years, LLM-based methods have become a feasible solution for handling tasks in various domains. However, in the field of content moderation, there is still a lack of detailed work that systematically introduces implementation details. In this paper, we introduce how to fine-tune an LLM model that can be privately deployed for content moderation. Specifically, we discuss whether incorporating reasons during the fine-tuning process would be better or if it should be treated as a classification task directly. We also explore the benefits of utilizing reasons generated by more powerful LLMs for fine-tuning privately deployed models and the impact of different processing approaches when the answers generated by the more powerful LLMs are incorrect. We report the entire research process and the key findings in this paper, hoping to provide valuable experience for researchers who are fine-tuning privately deployed models in their domain-specific research.

{{</citation>}}


### (21/122) GRAPES: Learning to Sample Graphs for Scalable Graph Neural Networks (Taraneh Younesian et al., 2023)

{{<citation>}}

Taraneh Younesian, Thiviyan Thanapalasingam, Emile van Krieken, Daniel Daza, Peter Bloem. (2023)  
**GRAPES: Learning to Sample Graphs for Scalable Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.03399v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) learn the representation of nodes in a graph by aggregating the neighborhood information in various ways. As these networks grow in depth, their receptive field grows exponentially due to the increase in neighborhood sizes, resulting in high memory costs. Graph sampling solves memory issues in GNNs by sampling a small ratio of the nodes in the graph. This way, GNNs can scale to much larger graphs. Most sampling methods focus on fixed sampling heuristics, which may not generalize to different structures or tasks. We introduce GRAPES, an adaptive graph sampling method that learns to identify sets of influential nodes for training a GNN classifier. GRAPES uses a GFlowNet to learn node sampling probabilities given the classification objectives. We evaluate GRAPES across several small- and large-scale graph benchmarks and demonstrate its effectiveness in accuracy and scalability. In contrast to existing sampling methods, GRAPES maintains high accuracy even with small sample sizes and, therefore, can scale to very large graphs. Our code is publicly available at https://github.com/dfdazac/grapes.

{{</citation>}}


### (22/122) LESSON: Learning to Integrate Exploration Strategies for Reinforcement Learning via an Option Framework (Woojun Kim et al., 2023)

{{<citation>}}

Woojun Kim, Jeonghye Kim, Youngchul Sung. (2023)  
**LESSON: Learning to Integrate Exploration Strategies for Reinforcement Learning via an Option Framework**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03342v1)  

---


**ABSTRACT**  
In this paper, a unified framework for exploration in reinforcement learning (RL) is proposed based on an option-critic model. The proposed framework learns to integrate a set of diverse exploration strategies so that the agent can adaptively select the most effective exploration strategy over time to realize a relevant exploration-exploitation trade-off for each given task. The effectiveness of the proposed exploration framework is demonstrated by various experiments in the MiniGrid and Atari environments.

{{</citation>}}


### (23/122) Probabilistic Forecasting of Day-Ahead Electricity Prices and their Volatility with LSTMs (Julius Trebbien et al., 2023)

{{<citation>}}

Julius Trebbien, Sebastian Pütz, Benjamin Schäfer, Heidi S. Nygård, Leonardo Rydin Gorjão, Dirk Witthaut. (2023)  
**Probabilistic Forecasting of Day-Ahead Electricity Prices and their Volatility with LSTMs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-data-an, physics-soc-ph  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.03339v1)  

---


**ABSTRACT**  
Accurate forecasts of electricity prices are crucial for the management of electric power systems and the development of smart applications. European electricity prices have risen substantially and became highly volatile after the Russian invasion of Ukraine, challenging established forecasting methods. Here, we present a Long Short-Term Memory (LSTM) model for the German-Luxembourg day-ahead electricity prices addressing these challenges. The recurrent structure of the LSTM allows the model to adapt to trends, while the joint prediction of both mean and standard deviation enables a probabilistic prediction. Using a physics-inspired approach - superstatistics - to derive an explanation for the statistics of prices, we show that the LSTM model faithfully reproduces both prices and their volatility.

{{</citation>}}


### (24/122) Untargeted White-box Adversarial Attack with Heuristic Defence Methods in Real-time Deep Learning based Network Intrusion Detection System (Khushnaseeb Roshan et al., 2023)

{{<citation>}}

Khushnaseeb Roshan, Aasim Zafar, Sheikh Burhan Ul Haque. (2023)  
**Untargeted White-box Adversarial Attack with Heuristic Defence Methods in Real-time Deep Learning based Network Intrusion Detection System**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Attack, Adversarial Training, Augmentation, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2310.03334v2)  

---


**ABSTRACT**  
Network Intrusion Detection System (NIDS) is a key component in securing the computer network from various cyber security threats and network attacks. However, consider an unfortunate situation where the NIDS is itself attacked and vulnerable more specifically, we can say, How to defend the defender?. In Adversarial Machine Learning (AML), the malicious actors aim to fool the Machine Learning (ML) and Deep Learning (DL) models to produce incorrect predictions with intentionally crafted adversarial examples. These adversarial perturbed examples have become the biggest vulnerability of ML and DL based systems and are major obstacles to their adoption in real-time and mission-critical applications such as NIDS. AML is an emerging research domain, and it has become a necessity for the in-depth study of adversarial attacks and their defence strategies to safeguard the computer network from various cyber security threads. In this research work, we aim to cover important aspects related to NIDS, adversarial attacks and its defence mechanism to increase the robustness of the ML and DL based NIDS. We implemented four powerful adversarial attack techniques, namely, Fast Gradient Sign Method (FGSM), Jacobian Saliency Map Attack (JSMA), Projected Gradient Descent (PGD) and Carlini & Wagner (C&W) in NIDS. We analyzed its performance in terms of various performance metrics in detail. Furthermore, the three heuristics defence strategies, i.e., Adversarial Training (AT), Gaussian Data Augmentation (GDA) and High Confidence (HC), are implemented to improve the NIDS robustness under adversarial attack situations. The complete workflow is demonstrated in real-time network with data packet flow. This research work provides the overall background for the researchers interested in AML and its implementation from a computer network security point of view.

{{</citation>}}


### (25/122) Fine-tune Language Models to Approximate Unbiased In-context Learning (Timothy Chu et al., 2023)

{{<citation>}}

Timothy Chu, Zhao Song, Chiwun Yang. (2023)  
**Fine-tune Language Models to Approximate Unbiased In-context Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03331v1)  

---


**ABSTRACT**  
In-context learning (ICL) is an astonishing emergent ability of large language models (LLMs). By presenting a prompt that includes multiple input-output pairs as examples and introducing a new query input, models can generate the corresponding output. However, the performance of models heavily relies on the quality of the input prompt when implementing in-context learning. Biased or imbalanced input prompts can significantly degrade the performance of language models. To address this issue, we introduce a reweighted algorithm called RICL (Reweighted In-context Learning). This algorithm fine-tunes language models using an unbiased validation set to determine the optimal weight for each input-output example to approximate unbiased in-context learning. Furthermore, we also introduce a low-cost reweighted algorithm, a linear optimal weight approximation algorithm called LARICL (Linear Approximation of Reweighted In-context Learning). This algorithm requires minimal training cost while providing effective results. We prove the convergence of our algorithm and validate its performance through experiments conducted on a numerical dataset. The experimental findings reveal a substantial improvement in comparison to benchmarks including the performance of casual prompt-based in-context learning and the performance of a classic fine-tuning method.

{{</citation>}}


### (26/122) BioBridge: Bridging Biomedical Foundation Models via Knowledge Graph (Zifeng Wang et al., 2023)

{{<citation>}}

Zifeng Wang, Zichen Wang, Balasubramaniam Srinivasan, Vassilis N. Ioannidis, Huzefa Rangwala, Rishita Anubhai. (2023)  
**BioBridge: Bridging Biomedical Foundation Models via Knowledge Graph**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.03320v1)  

---


**ABSTRACT**  
Foundation models (FMs) are able to leverage large volumes of unlabeled data to demonstrate superior performance across a wide range of tasks. However, FMs developed for biomedical domains have largely remained unimodal, i.e., independently trained and used for tasks on protein sequences alone, small molecule structures alone, or clinical data alone. To overcome this limitation of biomedical FMs, we present BioBridge, a novel parameter-efficient learning framework, to bridge independently trained unimodal FMs to establish multimodal behavior. BioBridge achieves it by utilizing Knowledge Graphs (KG) to learn transformations between one unimodal FM and another without fine-tuning any underlying unimodal FMs. Our empirical results demonstrate that BioBridge can beat the best baseline KG embedding methods (on average by around 76.3%) in cross-modal retrieval tasks. We also identify BioBridge demonstrates out-of-domain generalization ability by extrapolating to unseen modalities or relations. Additionally, we also show that BioBridge presents itself as a general purpose retriever that can aid biomedical multimodal question answering as well as enhance the guided generation of novel drugs.

{{</citation>}}


### (27/122) Benchmarking Large Language Models As AI Research Agents (Qian Huang et al., 2023)

{{<citation>}}

Qian Huang, Jian Vora, Percy Liang, Jure Leskovec. (2023)  
**Benchmarking Large Language Models As AI Research Agents**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03302v1)  

---


**ABSTRACT**  
Scientific experimentation involves an iterative process of creating hypotheses, designing experiments, running experiments, and analyzing the results. Can we build AI research agents to perform these long-horizon tasks? To take a step towards building and evaluating research agents on such open-ended decision-making tasks, we focus on the problem of machine learning engineering: given a task description and a dataset, build a high-performing model. In this paper, we propose MLAgentBench, a suite of ML tasks for benchmarking AI research agents. Agents can perform actions like reading/writing files, executing code, and inspecting outputs. With these actions, agents could run experiments, analyze the results, and modify the code of entire machine learning pipelines, such as data processing, architecture, training processes, etc. The benchmark then automatically evaluates the agent's performance objectively over various metrics related to performance and efficiency. We also design an LLM-based research agent to automatically perform experimentation loops in such an environment. Empirically, we find that a GPT-4-based research agent can feasibly build compelling ML models over many tasks in MLAgentBench, displaying highly interpretable plans and actions. However, the success rates vary considerably; they span from almost 90\% on well-established older datasets to as low as 10\% on recent Kaggle Challenges -- unavailable during the LLM model's pretraining -- and even 0\% on newer research challenges like BabyLM. Finally, we identify several key challenges for LLM-based research agents such as long-term planning and hallucination. Our code is released at https://github.com/snap-stanford/MLAgentBench.

{{</citation>}}


### (28/122) LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers (Dacheng Li et al., 2023)

{{<citation>}}

Dacheng Li, Rulin Shao, Anze Xie, Eric P. Xing, Joseph E. Gonzalez, Ion Stoica, Xuezhe Ma, Hao Zhang. (2023)  
**LightSeq: Sequence Level Parallelism for Distributed Training of Long Context Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.03294v1)  

---


**ABSTRACT**  
Increasing the context length of large language models (LLMs) unlocks fundamentally new capabilities, but also significantly increases the memory footprints of training. Previous model-parallel systems such as Megatron-LM partition and compute different attention heads in parallel, resulting in large communication volumes, so they cannot scale beyond the number of attention heads, thereby hindering its adoption. In this paper, we introduce a new approach, LightSeq, for long-context LLMs training. LightSeq has many notable advantages. First, LightSeq partitions over the sequence dimension, hence is agnostic to model architectures and readily applicable for models with varying numbers of attention heads, such as Multi-Head, Multi-Query and Grouped-Query attention. Second, LightSeq not only requires up to 4.7x less communication than Megatron-LM on popular LLMs but also overlaps the communication with computation. To further reduce the training time, LightSeq features a novel gradient checkpointing scheme to bypass an forward computation for memory-efficient attention. We evaluate LightSeq on Llama-7B and its variants with sequence lengths from 32K to 512K. Through comprehensive experiments on single and cross-node training, we show that LightSeq achieves up to 1.24-2.01x end-to-end speedup, and a 2-8x longer sequence length on models with fewer heads, compared to Megatron-LM. Codes will be available at https://github.com/RulinShao/LightSeq.

{{</citation>}}


### (29/122) A 5' UTR Language Model for Decoding Untranslated Regions of mRNA and Function Predictions (Yanyi Chu et al., 2023)

{{<citation>}}

Yanyi Chu, Dan Yu, Yupeng Li, Kaixuan Huang, Yue Shen, Le Cong, Jason Zhang, Mengdi Wang. (2023)  
**A 5' UTR Language Model for Decoding Untranslated Regions of mRNA and Function Predictions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.03281v2)  

---


**ABSTRACT**  
The 5' UTR, a regulatory region at the beginning of an mRNA molecule, plays a crucial role in regulating the translation process and impacts the protein expression level. Language models have showcased their effectiveness in decoding the functions of protein and genome sequences. Here, we introduced a language model for 5' UTR, which we refer to as the UTR-LM. The UTR-LM is pre-trained on endogenous 5' UTRs from multiple species and is further augmented with supervised information including secondary structure and minimum free energy. We fine-tuned the UTR-LM in a variety of downstream tasks. The model outperformed the best-known benchmark by up to 42% for predicting the Mean Ribosome Loading, and by up to 60% for predicting the Translation Efficiency and the mRNA Expression Level. The model also applies to identifying unannotated Internal Ribosome Entry Sites within the untranslated region and improves the AUPR from 0.37 to 0.52 compared to the best baseline. Further, we designed a library of 211 novel 5' UTRs with high predicted values of translation efficiency and evaluated them via a wet-lab assay. Experiment results confirmed that our top designs achieved a 32.5% increase in protein production level relative to well-established 5' UTR optimized for therapeutics.

{{</citation>}}


### (30/122) Fragment-based Pretraining and Finetuning on Molecular Graphs (Kha-Dinh Luong et al., 2023)

{{<citation>}}

Kha-Dinh Luong, Ambuj Singh. (2023)  
**Fragment-based Pretraining and Finetuning on Molecular Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.03274v1)  

---


**ABSTRACT**  
Property prediction on molecular graphs is an important application of Graph Neural Networks (GNNs). Recently, unlabeled molecular data has become abundant, which facilitates the rapid development of self-supervised learning for GNNs in the chemical domain. In this work, we propose pretraining GNNs at the fragment level, which serves as a promising middle ground to overcome the limitations of node-level and graph-level pretraining. Borrowing techniques from recent work on principle subgraph mining, we obtain a compact vocabulary of prevalent fragments that span a large pretraining dataset. From the extracted vocabulary, we introduce several fragment-based contrastive and predictive pretraining tasks. The contrastive learning task jointly pretrains two different GNNs: one based on molecular graphs and one based on fragment graphs, which represents high-order connectivity within molecules. By enforcing the consistency between the fragment embedding and the aggregated embedding of the corresponding atoms from the molecular graphs, we ensure that both embeddings capture structural information at multiple resolutions. The structural information of the fragment graphs is further exploited to extract auxiliary labels for the graph-level predictive pretraining. We employ both the pretrained molecular-based and fragment-based GNNs for downstream prediction, thus utilizing the fragment information during finetuning. Our models advance the performances on 5 out of 8 common molecular benchmarks and improve the performances on long-range biological benchmarks by at least 11.5%.

{{</citation>}}


### (31/122) UniPredict: Large Language Models are Universal Tabular Predictors (Ruiyu Wang et al., 2023)

{{<citation>}}

Ruiyu Wang, Zifeng Wang, Jimeng Sun. (2023)  
**UniPredict: Large Language Models are Universal Tabular Predictors**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.03266v1)  

---


**ABSTRACT**  
Tabular data prediction is a fundamental machine learning task for many applications. Existing methods predominantly employ discriminative modeling and operate under the assumption of a fixed target column, necessitating re-training for every new predictive task. Inspired by the generative power of large language models (LLMs), this paper exploits the idea of building universal tabular data predictors based on generative modeling, namely UniPredict. Here, we show that scaling up an LLM to extensive tabular datasets with the capability of comprehending diverse tabular inputs and predicting for target variables following the input instructions. Specifically, we train a single LLM on an aggregation of 169 tabular datasets with diverse targets and compare its performance against baselines that are trained on each dataset separately. We observe this versatile UniPredict model demonstrates an advantage over other models, ranging from 5.4% to 13.4%, when compared with the best tree-boosting baseline and the best neural network baseline, respectively. We further test UniPredict in few-shot learning settings on another 62 tabular datasets. Our method achieves strong performance in quickly adapting to new tasks, where our method outperforms XGBoost over 100% on the low-resource setup and shows a significant margin over all baselines. We envision that UniPredict sheds light on developing a universal tabular data prediction system that learns from data at scale and serves a wide range of prediction tasks.

{{</citation>}}


### (32/122) Molecule Design by Latent Prompt Transformer (Deqian Kong et al., 2023)

{{<citation>}}

Deqian Kong, Yuhao Huang, Jianwen Xie, Ying Nian Wu. (2023)  
**Molecule Design by Latent Prompt Transformer**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM, stat-ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.03253v1)  

---


**ABSTRACT**  
This paper proposes a latent prompt Transformer model for solving challenging optimization problems such as molecule design, where the goal is to find molecules with optimal values of a target chemical or biological property that can be computed by an existing software. Our proposed model consists of three components. (1) A latent vector whose prior distribution is modeled by a Unet transformation of a Gaussian white noise vector. (2) A molecule generation model that generates the string-based representation of molecule conditional on the latent vector in (1). We adopt the causal Transformer model that takes the latent vector in (1) as prompt. (3) A property prediction model that predicts the value of the target property of a molecule based on a non-linear regression on the latent vector in (1). We call the proposed model the latent prompt Transformer model. After initial training of the model on existing molecules and their property values, we then gradually shift the model distribution towards the region that supports desired values of the target property for the purpose of molecule design. Our experiments show that our proposed model achieves state of the art performances on several benchmark molecule design tasks.

{{</citation>}}


### (33/122) History Matching for Geological Carbon Storage using Data-Space Inversion with Spatio-Temporal Data Parameterization (Su Jiang et al., 2023)

{{<citation>}}

Su Jiang, Louis J. Durlofsky. (2023)  
**History Matching for Geological Carbon Storage using Data-Space Inversion with Spatio-Temporal Data Parameterization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.03228v1)  

---


**ABSTRACT**  
History matching based on monitoring data will enable uncertainty reduction, and thus improved aquifer management, in industrial-scale carbon storage operations. In traditional model-based data assimilation, geomodel parameters are modified to force agreement between flow simulation results and observations. In data-space inversion (DSI), history-matched quantities of interest, e.g., posterior pressure and saturation fields conditioned to observations, are inferred directly, without constructing posterior geomodels. This is accomplished efficiently using a set of O(1000) prior simulation results, data parameterization, and posterior sampling within a Bayesian setting. In this study, we develop and implement (in DSI) a deep-learning-based parameterization to represent spatio-temporal pressure and CO2 saturation fields at a set of time steps. The new parameterization uses an adversarial autoencoder (AAE) for dimension reduction and a convolutional long short-term memory (convLSTM) network to represent the spatial distribution and temporal evolution of the pressure and saturation fields. This parameterization is used with an ensemble smoother with multiple data assimilation (ESMDA) in the DSI framework to enable posterior predictions. A realistic 3D system characterized by prior geological realizations drawn from a range of geological scenarios is considered. A local grid refinement procedure is introduced to estimate the error covariance term that appears in the history matching formulation. Extensive history matching results are presented for various quantities, for multiple synthetic true models. Substantial uncertainty reduction in posterior pressure and saturation fields is achieved in all cases. The framework is applied to efficiently provide posterior predictions for a range of error covariance specifications. Such an assessment would be expensive using a model-based approach.

{{</citation>}}


### (34/122) Safe Exploration in Reinforcement Learning: A Generalized Formulation and Algorithms (Akifumi Wachi et al., 2023)

{{<citation>}}

Akifumi Wachi, Wataru Hashimoto, Xun Shen, Kazumune Hashimoto. (2023)  
**Safe Exploration in Reinforcement Learning: A Generalized Formulation and Algorithms**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03225v1)  

---


**ABSTRACT**  
Safe exploration is essential for the practical use of reinforcement learning (RL) in many real-world scenarios. In this paper, we present a generalized safe exploration (GSE) problem as a unified formulation of common safe exploration problems. We then propose a solution of the GSE problem in the form of a meta-algorithm for safe exploration, MASE, which combines an unconstrained RL algorithm with an uncertainty quantifier to guarantee safety in the current episode while properly penalizing unsafe explorations before actual safety violation to discourage them in future episodes. The advantage of MASE is that we can optimize a policy while guaranteeing with a high probability that no safety constraint will be violated under proper assumptions. Specifically, we present two variants of MASE with different constructions of the uncertainty quantifier: one based on generalized linear models with theoretical guarantees of safety and near-optimality, and another that combines a Gaussian process to ensure safety with a deep RL algorithm to maximize the reward. Finally, we demonstrate that our proposed algorithm achieves better performance than state-of-the-art algorithms on grid-world and Safety Gym benchmarks without violating any safety constraints, even during training.

{{</citation>}}


### (35/122) Know2BIO: A Comprehensive Dual-View Benchmark for Evolving Biomedical Knowledge Graphs (Yijia Xiao et al., 2023)

{{<citation>}}

Yijia Xiao, Dylan Steinecke, Alexander Russell Pelletier, Yushi Bai, Peipei Ping, Wei Wang. (2023)  
**Know2BIO: A Comprehensive Dual-View Benchmark for Evolving Biomedical Knowledge Graphs**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-QM  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.03221v1)  

---


**ABSTRACT**  
Knowledge graphs (KGs) have emerged as a powerful framework for representing and integrating complex biomedical information. However, assembling KGs from diverse sources remains a significant challenge in several aspects, including entity alignment, scalability, and the need for continuous updates to keep pace with scientific advancements. Moreover, the representative power of KGs is often limited by the scarcity of multi-modal data integration. To overcome these challenges, we propose Know2BIO, a general-purpose heterogeneous KG benchmark for the biomedical domain. Know2BIO integrates data from 30 diverse sources, capturing intricate relationships across 11 biomedical categories. It currently consists of ~219,000 nodes and ~6,200,000 edges. Know2BIO is capable of user-directed automated updating to reflect the latest knowledge in biomedical science. Furthermore, Know2BIO is accompanied by multi-modal data: node features including text descriptions, protein and compound sequences and structures, enabling the utilization of emerging natural language processing methods and multi-modal data integration strategies. We evaluate KG representation models on Know2BIO, demonstrating its effectiveness as a benchmark for KG representation learning in the biomedical field. Data and source code of Know2BIO are available at https://github.com/Yijia-Xiao/Know2BIO/.

{{</citation>}}


## cs.CV (20)



### (36/122) Hard View Selection for Contrastive Learning (Fabio Ferreira et al., 2023)

{{<citation>}}

Fabio Ferreira, Ivo Rapant, Frank Hutter. (2023)  
**Hard View Selection for Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.03940v1)  

---


**ABSTRACT**  
Many Contrastive Learning (CL) methods train their models to be invariant to different "views" of an image input for which a good data augmentation pipeline is crucial. While considerable efforts were directed towards improving pre-text tasks, architectures, or robustness (e.g., Siamese networks or teacher-softmax centering), the majority of these methods remain strongly reliant on the random sampling of operations within the image augmentation pipeline, such as the random resized crop or color distortion operation. In this paper, we argue that the role of the view generation and its effect on performance has so far received insufficient attention. To address this, we propose an easy, learning-free, yet powerful Hard View Selection (HVS) strategy designed to extend the random view generation to expose the pretrained model to harder samples during CL training. It encompasses the following iterative steps: 1) randomly sample multiple views and create pairs of two views, 2) run forward passes for each view pair on the currently trained model, 3) adversarially select the pair yielding the worst loss, and 4) run the backward pass with the selected pair. In our empirical analysis we show that under the hood, HVS increases task difficulty by controlling the Intersection over Union of views during pretraining. With only 300-epoch pretraining, HVS is able to closely rival the 800-epoch DINO baseline which remains very favorable even when factoring in the slowdown induced by the additional forwards of HVS. Additionally, HVS consistently achieves accuracy improvements on ImageNet between 0.55% and 1.9% on linear evaluation and similar improvements on transfer tasks across multiple CL methods, such as DINO, SimSiam, and SimCLR.

{{</citation>}}


### (37/122) Coloring Deep CNN Layers with Activation Hue Loss (Louis-François Bouchard et al., 2023)

{{<citation>}}

Louis-François Bouchard, Mohsen Ben Lazreg, Matthew Toews. (2023)  
**Coloring Deep CNN Layers with Activation Hue Loss**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.03911v1)  

---


**ABSTRACT**  
This paper proposes a novel hue-like angular parameter to model the structure of deep convolutional neural network (CNN) activation space, referred to as the {\em activation hue}, for the purpose of regularizing models for more effective learning. The activation hue generalizes the notion of color hue angle in standard 3-channel RGB intensity space to $N$-channel activation space. A series of observations based on nearest neighbor indexing of activation vectors with pre-trained networks indicate that class-informative activations are concentrated about an angle $\theta$ in both the $(x,y)$ image plane and in multi-channel activation space. A regularization term in the form of hue-like angular $\theta$ labels is proposed to complement standard one-hot loss. Training from scratch using combined one-hot + activation hue loss improves classification performance modestly for a wide variety of classification tasks, including ImageNet.

{{</citation>}}


### (38/122) Consistency Regularization Improves Placenta Segmentation in Fetal EPI MRI Time Series (Yingcheng Liu et al., 2023)

{{<citation>}}

Yingcheng Liu, Neerav Karani, Neel Dey, S. Mazdak Abulnaga, Junshen Xu, P. Ellen Grant, Esra Abaci Turk, Polina Golland. (2023)  
**Consistency Regularization Improves Placenta Segmentation in Fetal EPI MRI Time Series**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.03870v1)  

---


**ABSTRACT**  
The placenta plays a crucial role in fetal development. Automated 3D placenta segmentation from fetal EPI MRI holds promise for advancing prenatal care. This paper proposes an effective semi-supervised learning method for improving placenta segmentation in fetal EPI MRI time series. We employ consistency regularization loss that promotes consistency under spatial transformation of the same image and temporal consistency across nearby images in a time series. The experimental results show that the method improves the overall segmentation accuracy and provides better performance for outliers and hard samples. The evaluation also indicates that our method improves the temporal coherency of the prediction, which could lead to more accurate computation of temporal placental biomarkers. This work contributes to the study of the placenta and prenatal clinical decision-making. Code is available at https://github.com/firstmover/cr-seg.

{{</citation>}}


### (39/122) Integrating Audio-Visual Features for Multimodal Deepfake Detection (Sneha Muppalla et al., 2023)

{{<citation>}}

Sneha Muppalla, Shan Jia, Siwei Lyu. (2023)  
**Integrating Audio-Visual Features for Multimodal Deepfake Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03827v1)  

---


**ABSTRACT**  
Deepfakes are AI-generated media in which an image or video has been digitally modified. The advancements made in deepfake technology have led to privacy and security issues. Most deepfake detection techniques rely on the detection of a single modality. Existing methods for audio-visual detection do not always surpass that of the analysis based on single modalities. Therefore, this paper proposes an audio-visual-based method for deepfake detection, which integrates fine-grained deepfake identification with binary classification. We categorize the samples into four types by combining labels specific to each single modality. This method enhances the detection under intra-domain and cross-domain testing.

{{</citation>}}


### (40/122) WLST: Weak Labels Guided Self-training for Weakly-supervised Domain Adaptation on 3D Object Detection (Tsung-Lin Tsou et al., 2023)

{{<citation>}}

Tsung-Lin Tsou, Tsung-Han Wu, Winston H. Hsu. (2023)  
**WLST: Weak Labels Guided Self-training for Weakly-supervised Domain Adaptation on 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.03821v1)  

---


**ABSTRACT**  
In the field of domain adaptation (DA) on 3D object detection, most of the work is dedicated to unsupervised domain adaptation (UDA). Yet, without any target annotations, the performance gap between the UDA approaches and the fully-supervised approach is still noticeable, which is impractical for real-world applications. On the other hand, weakly-supervised domain adaptation (WDA) is an underexplored yet practical task that only requires few labeling effort on the target domain. To improve the DA performance in a cost-effective way, we propose a general weak labels guided self-training framework, WLST, designed for WDA on 3D object detection. By incorporating autolabeler, which can generate 3D pseudo labels from 2D bounding boxes, into the existing self-training pipeline, our method is able to generate more robust and consistent pseudo labels that would benefit the training process on the target domain. Extensive experiments demonstrate the effectiveness, robustness, and detector-agnosticism of our WLST framework. Notably, it outperforms previous state-of-the-art methods on all evaluation tasks.

{{</citation>}}


### (41/122) Improved Baselines with Visual Instruction Tuning (Haotian Liu et al., 2023)

{{<citation>}}

Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee. (2023)  
**Improved Baselines with Visual Instruction Tuning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.03744v1)  

---


**ABSTRACT**  
Large multimodal models (LMM) have recently shown encouraging progress with visual instruction tuning. In this note, we show that the fully-connected vision-language cross-modal connector in LLaVA is surprisingly powerful and data-efficient. With simple modifications to LLaVA, namely, using CLIP-ViT-L-336px with an MLP projection and adding academic-task-oriented VQA data with simple response formatting prompts, we establish stronger baselines that achieve state-of-the-art across 11 benchmarks. Our final 13B checkpoint uses merely 1.2M publicly available data, and finishes full training in ~1 day on a single 8-A100 node. We hope this can make state-of-the-art LMM research more accessible. Code and model will be publicly available.

{{</citation>}}


### (42/122) LumiNet: The Bright Side of Perceptual Knowledge Distillation (Md. Ismail Hossain et al., 2023)

{{<citation>}}

Md. Ismail Hossain, M M Lutfe Elahi, Sameera Ramasinghe, Ali Cheraghian, Fuad Rahman, Nabeel Mohammed, Shafin Rahman. (2023)  
**LumiNet: The Bright Side of Perceptual Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.03669v1)  

---


**ABSTRACT**  
In knowledge distillation research, feature-based methods have dominated due to their ability to effectively tap into extensive teacher models. In contrast, logit-based approaches are considered to be less adept at extracting hidden 'dark knowledge' from teachers. To bridge this gap, we present LumiNet, a novel knowledge-transfer algorithm designed to enhance logit-based distillation. We introduce a perception matrix that aims to recalibrate logits through adjustments based on the model's representation capability. By meticulously analyzing intra-class dynamics, LumiNet reconstructs more granular inter-class relationships, enabling the student model to learn a richer breadth of knowledge. Both teacher and student models are mapped onto this refined matrix, with the student's goal being to minimize representational discrepancies. Rigorous testing on benchmark datasets (CIFAR-100, ImageNet, and MSCOCO) attests to LumiNet's efficacy, revealing its competitive edge over leading feature-based methods. Moreover, in exploring the realm of transfer learning, we assess how effectively the student model, trained using our method, adapts to downstream tasks. Notably, when applied to Tiny ImageNet, the transferred features exhibit remarkable performance, further underscoring LumiNet's versatility and robustness in diverse settings. With LumiNet, we hope to steer the research discourse towards a renewed interest in the latent capabilities of logit-based knowledge distillation.

{{</citation>}}


### (43/122) Robustness-Guided Image Synthesis for Data-Free Quantization (Jianhong Bai et al., 2023)

{{<citation>}}

Jianhong Bai, Yuchen Yang, Huanpeng Chu, Hualiang Wang, Zuozhu Liu, Ruizhe Chen, Xiaoxuan He, Lianrui Mu, Chengfei Cai, Haoji Hu. (2023)  
**Robustness-Guided Image Synthesis for Data-Free Quantization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2310.03661v1)  

---


**ABSTRACT**  
Quantization has emerged as a promising direction for model compression. Recently, data-free quantization has been widely studied as a promising method to avoid privacy concerns, which synthesizes images as an alternative to real training data. Existing methods use classification loss to ensure the reliability of the synthesized images. Unfortunately, even if these images are well-classified by the pre-trained model, they still suffer from low semantics and homogenization issues. Intuitively, these low-semantic images are sensitive to perturbations, and the pre-trained model tends to have inconsistent output when the generator synthesizes an image with poor semantics. To this end, we propose Robustness-Guided Image Synthesis (RIS), a simple but effective method to enrich the semantics of synthetic images and improve image diversity, further boosting the performance of downstream data-free compression tasks. Concretely, we first introduce perturbations on input and model weight, then define the inconsistency metrics at feature and prediction levels before and after perturbations. On the basis of inconsistency on two levels, we design a robustness optimization objective to enhance the semantics of synthetic images. Moreover, we also make our approach diversity-aware by forcing the generator to synthesize images with small correlations in the label space. With RIS, we achieve state-of-the-art performance for various settings on data-free quantization and can be extended to other data-free compression tasks.

{{</citation>}}


### (44/122) Visual inspection for illicit items in X-ray images using Deep Learning (Ioannis Mademlis et al., 2023)

{{<citation>}}

Ioannis Mademlis, Georgios Batsis, Adamantia Anna Rebolledo Chrysochoou, Georgios Th. Papadopoulos. (2023)  
**Visual inspection for illicit items in X-ray images using Deep Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.03658v1)  

---


**ABSTRACT**  
Automated detection of contraband items in X-ray images can significantly increase public safety, by enhancing the productivity and alleviating the mental load of security officers in airports, subways, customs/post offices, etc. The large volume and high throughput of passengers, mailed parcels, etc., during rush hours practically make it a Big Data problem. Modern computer vision algorithms relying on Deep Neural Networks (DNNs) have proven capable of undertaking this task even under resource-constrained and embedded execution scenarios, e.g., as is the case with fast, single-stage object detectors. However, no comparative experimental assessment of the various relevant DNN components/methods has been performed under a common evaluation protocol, which means that reliable cross-method comparisons are missing. This paper presents exactly such a comparative assessment, utilizing a public relevant dataset and a well-defined methodology for selecting the specific DNN components/modules that are being evaluated. The results indicate the superiority of Transformer detectors, the obsolete nature of auxiliary neural modules that have been developed in the past few years for security applications and the efficiency of the CSP-DarkNet backbone CNN.

{{</citation>}}


### (45/122) Towards Unified Deep Image Deraining: A Survey and A New Benchmark (Xiang Chen et al., 2023)

{{<citation>}}

Xiang Chen, Jinshan Pan, Jiangxin Dong, Jinhui Tang. (2023)  
**Towards Unified Deep Image Deraining: A Survey and A New Benchmark**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03535v1)  

---


**ABSTRACT**  
Recent years have witnessed significant advances in image deraining due to the kinds of effective image priors and deep learning models. As each deraining approach has individual settings (e.g., training and test datasets, evaluation criteria), how to fairly evaluate existing approaches comprehensively is not a trivial task. Although existing surveys aim to review of image deraining approaches comprehensively, few of them focus on providing unify evaluation settings to examine the deraining capability and practicality evaluation. In this paper, we provide a comprehensive review of existing image deraining method and provide a unify evaluation setting to evaluate the performance of image deraining methods. We construct a new high-quality benchmark named HQ-RAIN to further conduct extensive evaluation, consisting of 5,000 paired high-resolution synthetic images with higher harmony and realism. We also discuss the existing challenges and highlight several future research opportunities worth exploring. To facilitate the reproduction and tracking of the latest deraining technologies for general users, we build an online platform to provide the off-the-shelf toolkit, involving the large-scale performance evaluation. This online platform and the proposed new benchmark are publicly available and will be regularly updated at http://www.deraining.tech/.

{{</citation>}}


### (46/122) PrototypeFormer: Learning to Explore Prototype Relationships for Few-shot Image Classification (Feihong He et al., 2023)

{{<citation>}}

Feihong He, Gang Li, Lingyu Si, Leilei Yan, Fanzhang Li, Fuchun Sun. (2023)  
**PrototypeFormer: Learning to Explore Prototype Relationships for Few-shot Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.03517v1)  

---


**ABSTRACT**  
Few-shot image classification has received considerable attention for addressing the challenge of poor classification performance with limited samples in novel classes. However, numerous studies have employed sophisticated learning strategies and diversified feature extraction methods to address this issue. In this paper, we propose our method called PrototypeFormer, which aims to significantly advance traditional few-shot image classification approaches by exploring prototype relationships. Specifically, we utilize a transformer architecture to build a prototype extraction module, aiming to extract class representations that are more discriminative for few-shot classification. Additionally, during the model training process, we propose a contrastive learning-based optimization approach to optimize prototype features in few-shot learning scenarios. Despite its simplicity, the method performs remarkably well, with no bells and whistles. We have experimented with our approach on several popular few-shot image classification benchmark datasets, which shows that our method outperforms all current state-of-the-art methods. In particular, our method achieves 97.07% and 90.88% on 5-way 5-shot and 5-way 1-shot tasks of miniImageNet, which surpasses the state-of-the-art results with accuracy of 7.27% and 8.72%, respectively. The code will be released later.

{{</citation>}}


### (47/122) A Complementary Global and Local Knowledge Network for Ultrasound denoising with Fine-grained Refinement (Zhenyu Bu et al., 2023)

{{<citation>}}

Zhenyu Bu, Kai-Ni Wang, Fuxing Zhao, Shengxiao Li, Guang-Quan Zhou. (2023)  
**A Complementary Global and Local Knowledge Network for Ultrasound denoising with Fine-grained Refinement**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.03402v1)  

---


**ABSTRACT**  
Ultrasound imaging serves as an effective and non-invasive diagnostic tool commonly employed in clinical examinations. However, the presence of speckle noise in ultrasound images invariably degrades image quality, impeding the performance of subsequent tasks, such as segmentation and classification. Existing methods for speckle noise reduction frequently induce excessive image smoothing or fail to preserve detailed information adequately. In this paper, we propose a complementary global and local knowledge network for ultrasound denoising with fine-grained refinement. Initially, the proposed architecture employs the L-CSwinTransformer as encoder to capture global information, incorporating CNN as decoder to fuse local features. We expand the resolution of the feature at different stages to extract more global information compared to the original CSwinTransformer. Subsequently, we integrate Fine-grained Refinement Block (FRB) within the skip-connection stage to further augment features. We validate our model on two public datasets, HC18 and BUSI. Experimental results demonstrate that our model can achieve competitive performance in both quantitative metrics and visual performance. Our code will be available at https://github.com/AAlkaid/USDenoising.

{{</citation>}}


### (48/122) AI-based automated active learning for discovery of hidden dynamic processes: A use case in light microscopy (Nils Friederich et al., 2023)

{{<citation>}}

Nils Friederich, Angelo Yamachui Sitcheu, Oliver Neumann, Süheyla Eroğlu-Kayıkçı, Roshan Prizak, Lennart Hilbert, Ralf Mikut. (2023)  
**AI-based automated active learning for discovery of hidden dynamic processes: A use case in light microscopy**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04461v1)  

---


**ABSTRACT**  
In the biomedical environment, experiments assessing dynamic processes are primarily performed by a human acquisition supervisor. Contemporary implementations of such experiments frequently aim to acquire a maximum number of relevant events from sometimes several hundred parallel, non-synchronous processes. Since in some high-throughput experiments, only one or a few instances of a given process can be observed simultaneously, a strategy for planning and executing an efficient acquisition paradigm is essential. To address this problem, we present two new methods in this paper. The first method, Encoded Dynamic Process (EDP), is Artificial Intelligence (AI)-based and represents dynamic processes so as to allow prediction of pseudo-time values from single still images. Second, with Experiment Automation Pipeline for Dynamic Processes (EAPDP), we present a Machine Learning Operations (MLOps)-based pipeline that uses the extracted knowledge from EDP to efficiently schedule acquisition in biomedical experiments for dynamic processes in practice. In a first experiment, we show that the pre-trained State-Of-The- Art (SOTA) object segmentation method Contour Proposal Networks (CPN) works reliably as a module of EAPDP to extract the relevant object for EDP from the acquired three-dimensional image stack.

{{</citation>}}


### (49/122) Robust Representation Learning via Asymmetric Negative Contrast and Reverse Attention (Nuoyan Zhou et al., 2023)

{{<citation>}}

Nuoyan Zhou, Decheng Liu, Dawei Zhou, Xinbo Gao, Nannan Wang. (2023)  
**Robust Representation Learning via Asymmetric Negative Contrast and Reverse Attention**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Attention, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.03358v1)  

---


**ABSTRACT**  
Deep neural networks are vulnerable to adversarial noise. Adversarial training (AT) has been demonstrated to be the most effective defense strategy to protect neural networks from being fooled. However, we find AT omits to learning robust features, resulting in poor performance of adversarial robustness. To address this issue, we highlight two characteristics of robust representation: (1) $\bf{exclusion}$: the feature of natural examples keeps away from that of other classes; (2) $\bf{alignment}$: the feature of natural and corresponding adversarial examples is close to each other. These motivate us to propose a generic framework of AT to gain robust representation, by the asymmetric negative contrast and reverse attention. Specifically, we design an asymmetric negative contrast based on predicted probabilities, to push away examples of different classes in the feature space. Moreover, we propose to weight feature by parameters of the linear classifier as the reverse attention, to obtain class-aware feature and pull close the feature of the same class. Empirical evaluations on three benchmark datasets show our methods greatly advance the robustness of AT and achieve state-of-the-art performance. Code is available at <https://github.com/changzhang777/ANCRA>.

{{</citation>}}


### (50/122) Denoising Diffusion Step-aware Models (Shuai Yang et al., 2023)

{{<citation>}}

Shuai Yang, Yukang Chen, Luozhou Wang, Shu Liu, Yingcong Chen. (2023)  
**Denoising Diffusion Step-aware Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.03337v1)  

---


**ABSTRACT**  
Denoising Diffusion Probabilistic Models (DDPMs) have garnered popularity for data generation across various domains. However, a significant bottleneck is the necessity for whole-network computation during every step of the generative process, leading to high computational overheads. This paper presents a novel framework, Denoising Diffusion Step-aware Models (DDSM), to address this challenge. Unlike conventional approaches, DDSM employs a spectrum of neural networks whose sizes are adapted according to the importance of each generative step, as determined through evolutionary search. This step-wise network variation effectively circumvents redundant computational efforts, particularly in less critical steps, thereby enhancing the efficiency of the diffusion model. Furthermore, the step-aware design can be seamlessly integrated with other efficiency-geared diffusion models such as DDIMs and latent diffusion, thus broadening the scope of computational savings. Empirical evaluations demonstrate that DDSM achieves computational savings of 49% for CIFAR-10, 61% for CelebA-HQ, 59% for LSUN-bedroom, 71% for AFHQ, and 76% for ImageNet, all without compromising the generation quality. Our code and models will be publicly available.

{{</citation>}}


### (51/122) Real-time Multi-modal Object Detection and Tracking on Edge for Regulatory Compliance Monitoring (Jia Syuen Lim et al., 2023)

{{<citation>}}

Jia Syuen Lim, Ziwei Wang, Jiajun Liu, Abdelwahed Khamis, Reza Arablouei, Robert Barlow, Ryan McAllister. (2023)  
**Real-time Multi-modal Object Detection and Tracking on Edge for Regulatory Compliance Monitoring**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.03333v1)  

---


**ABSTRACT**  
Regulatory compliance auditing across diverse industrial domains requires heightened quality assurance and traceability. Present manual and intermittent approaches to such auditing yield significant challenges, potentially leading to oversights in the monitoring process. To address these issues, we introduce a real-time, multi-modal sensing system employing 3D time-of-flight and RGB cameras, coupled with unsupervised learning techniques on edge AI devices. This enables continuous object tracking thereby enhancing efficiency in record-keeping and minimizing manual interventions. While we validate the system in a knife sanitization context within agrifood facilities, emphasizing its prowess against occlusion and low-light issues with RGB cameras, its potential spans various industrial monitoring settings.

{{</citation>}}


### (52/122) Investigating the Limitation of CLIP Models: The Worst-Performing Categories (Jie-Jing Shao et al., 2023)

{{<citation>}}

Jie-Jing Shao, Jiang-Xin Shi, Xiao-Wen Yang, Lan-Zhe Guo, Yu-Feng Li. (2023)  
**Investigating the Limitation of CLIP Models: The Worst-Performing Categories**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.03324v1)  

---


**ABSTRACT**  
Contrastive Language-Image Pre-training (CLIP) provides a foundation model by integrating natural language into visual concepts, enabling zero-shot recognition on downstream tasks. It is usually expected that satisfactory overall accuracy can be achieved across numerous domains through well-designed textual prompts. However, we found that their performance in the worst categories is significantly inferior to the overall performance. For example, on ImageNet, there are a total of 10 categories with class-wise accuracy as low as 0\%, even though the overall performance has achieved 64.1\%. This phenomenon reveals the potential risks associated with using CLIP models, particularly in risk-sensitive applications where specific categories hold significant importance. To address this issue, we investigate the alignment between the two modalities in the CLIP model and propose the Class-wise Matching Margin (\cmm) to measure the inference confusion. \cmm\ can effectively identify the worst-performing categories and estimate the potential performance of the candidate prompts. We further query large language models to enrich descriptions of worst-performing categories and build a weighted ensemble to highlight the efficient prompts. Experimental results clearly verify the effectiveness of our proposal, where the accuracy on the worst-10 categories on ImageNet is boosted to 5.2\%, without manual prompt engineering, laborious optimization, or access to labeled validation data.

{{</citation>}}


### (53/122) PoseAction: Action Recognition for Patients in the Ward using Deep Learning Approaches (Zherui Li et al., 2023)

{{<citation>}}

Zherui Li, Raye Chen-Hua Yeow. (2023)  
**PoseAction: Action Recognition for Patients in the Ward using Deep Learning Approaches**  

---
Primary Category: cs.CV  
Categories: I-4-9, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03288v1)  

---


**ABSTRACT**  
Real-time intelligent detection and prediction of subjects' behavior particularly their movements or actions is critical in the ward. This approach offers the advantage of reducing in-hospital care costs and improving the efficiency of healthcare workers, which is especially true for scenarios at night or during peak admission periods. Therefore, in this work, we propose using computer vision (CV) and deep learning (DL) methods for detecting subjects and recognizing their actions. We utilize OpenPose as an accurate subject detector for recognizing the positions of human subjects in the video stream. Additionally, we employ AlphAction's Asynchronous Interaction Aggregation (AIA) network to predict the actions of detected subjects. This integrated model, referred to as PoseAction, is proposed. At the same time, the proposed model is further trained to predict 12 common actions in ward areas, such as staggering, chest pain, and falling down, using medical-related video clips from the NTU RGB+D and NTU RGB+D 120 datasets. The results demonstrate that PoseAction achieves the highest classification mAP of 98.72% (IoU@0.5). Additionally, this study develops an online real-time mode for action recognition, which strongly supports the clinical translation of PoseAction. Furthermore, using OpenPose's function for recognizing face key points, we also implement face blurring, which is a practical solution to address the privacy protection concerns of patients and healthcare workers. Nevertheless, the training data for PoseAction is currently limited, particularly in terms of label diversity. Consequently, the subsequent step involves utilizing a more diverse dataset (including general actions) to train the model's parameters for improved generalization.

{{</citation>}}


### (54/122) Ablation Study to Clarify the Mechanism of Object Segmentation in Multi-Object Representation Learning (Takayuki Komatsu et al., 2023)

{{<citation>}}

Takayuki Komatsu, Yoshiyuki Ohmura, Yasuo Kuniyoshi. (2023)  
**Ablation Study to Clarify the Mechanism of Object Segmentation in Multi-Object Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.03273v1)  

---


**ABSTRACT**  
Multi-object representation learning aims to represent complex real-world visual input using the composition of multiple objects. Representation learning methods have often used unsupervised learning to segment an input image into individual objects and encode these objects into each latent vector. However, it is not clear how previous methods have achieved the appropriate segmentation of individual objects. Additionally, most of the previous methods regularize the latent vectors using a Variational Autoencoder (VAE). Therefore, it is not clear whether VAE regularization contributes to appropriate object segmentation. To elucidate the mechanism of object segmentation in multi-object representation learning, we conducted an ablation study on MONet, which is a typical method. MONet represents multiple objects using pairs that consist of an attention mask and the latent vector corresponding to the attention mask. Each latent vector is encoded from the input image and attention mask. Then, the component image and attention mask are decoded from each latent vector. The loss function of MONet consists of 1) the sum of reconstruction losses between the input image and decoded component image, 2) the VAE regularization loss of the latent vector, and 3) the reconstruction loss of the attention mask to explicitly encode shape information. We conducted an ablation study on these three loss functions to investigate the effect on segmentation performance. Our results showed that the VAE regularization loss did not affect segmentation performance and the others losses did affect it. Based on this result, we hypothesize that it is important to maximize the attention mask of the image region best represented by a single latent vector corresponding to the attention mask. We confirmed this hypothesis by evaluating a new loss function with the same mechanism as the hypothesis.

{{</citation>}}


### (55/122) EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models (Yefei He et al., 2023)

{{<citation>}}

Yefei He, Jing Liu, Weijia Wu, Hong Zhou, Bohan Zhuang. (2023)  
**EfficientDM: Efficient Quantization-Aware Fine-Tuning of Low-Bit Diffusion Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, QA, Quantization  
[Paper Link](http://arxiv.org/abs/2310.03270v2)  

---


**ABSTRACT**  
Diffusion models have demonstrated remarkable capabilities in image synthesis and related generative tasks. Nevertheless, their practicality for low-latency real-world applications is constrained by substantial computational costs and latency issues. Quantization is a dominant way to compress and accelerate diffusion models, where post-training quantization (PTQ) and quantization-aware training (QAT) are two main approaches, each bearing its own properties. While PTQ exhibits efficiency in terms of both time and data usage, it may lead to diminished performance in low bit-width. On the other hand, QAT can alleviate performance degradation but comes with substantial demands on computational and data resources. To capitalize on the advantages while avoiding their respective drawbacks, we introduce a data-free and parameter-efficient fine-tuning framework for low-bit diffusion models, dubbed EfficientDM, to achieve QAT-level performance with PTQ-like efficiency. Specifically, we propose a quantization-aware variant of the low-rank adapter (QALoRA) that can be merged with model weights and jointly quantized to low bit-width. The fine-tuning process distills the denoising capabilities of the full-precision model into its quantized counterpart, eliminating the requirement for training data. We also introduce scale-aware optimization and employ temporal learned step-size quantization to further enhance performance. Extensive experimental results demonstrate that our method significantly outperforms previous PTQ-based diffusion models while maintaining similar time and data efficiency. Specifically, there is only a marginal 0.05 sFID increase when quantizing both weights and activations of LDM-4 to 4-bit on ImageNet 256x256. Compared to QAT-based methods, our EfficientDM also boasts a 16.2x faster quantization speed with comparable generation quality.

{{</citation>}}


## cs.SD (2)



### (56/122) EFFUSE: Efficient Self-Supervised Feature Fusion for E2E ASR in Multilingual and Low Resource Scenarios (Tejes Srivastava et al., 2023)

{{<citation>}}

Tejes Srivastava, Jiatong Shi, William Chen, Shinji Watanabe. (2023)  
**EFFUSE: Efficient Self-Supervised Feature Fusion for E2E ASR in Multilingual and Low Resource Scenarios**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Multilingual, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.03938v1)  

---


**ABSTRACT**  
Self-Supervised Learning (SSL) models have demonstrated exceptional performance in various speech tasks, particularly in low-resource and multilingual domains. Recent works show that fusing SSL models could achieve superior performance compared to using one SSL model. However, fusion models have increased model parameter size, leading to longer inference times. In this paper, we propose a novel approach of predicting other SSL models' features from a single SSL model, resulting in a light-weight framework with competitive performance. Our experiments show that SSL feature prediction models outperform individual SSL models in multilingual speech recognition tasks. The leading prediction model achieves an average SUPERB score increase of 135.4 in ML-SUPERB benchmarks. Moreover, our proposed framework offers an efficient solution, as it reduces the resulting model parameter size and inference times compared to previous fusion models.

{{</citation>}}


### (57/122) Securing Voice Biometrics: One-Shot Learning Approach for Audio Deepfake Detection (Awais Khan et al., 2023)

{{<citation>}}

Awais Khan, Khalid Mahmood Malik. (2023)  
**Securing Voice Biometrics: One-Shot Learning Approach for Audio Deepfake Detection**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: AI, Speaker Verification  
[Paper Link](http://arxiv.org/abs/2310.03856v1)  

---


**ABSTRACT**  
The Automatic Speaker Verification (ASV) system is vulnerable to fraudulent activities using audio deepfakes, also known as logical-access voice spoofing attacks. These deepfakes pose a concerning threat to voice biometrics due to recent advancements in generative AI and speech synthesis technologies. While several deep learning models for speech synthesis detection have been developed, most of them show poor generalizability, especially when the attacks have different statistical distributions from the ones seen. Therefore, this paper presents Quick-SpoofNet, an approach for detecting both seen and unseen synthetic attacks in the ASV system using one-shot learning and metric learning techniques. By using the effective spectral feature set, the proposed method extracts compact and representative temporal embeddings from the voice samples and utilizes metric learning and triplet loss to assess the similarity index and distinguish different embeddings. The system effectively clusters similar speech embeddings, classifying bona fide speeches as the target class and identifying other clusters as spoofing attacks. The proposed system is evaluated using the ASVspoof 2019 logical access (LA) dataset and tested against unseen deepfake attacks from the ASVspoof 2021 dataset. Additionally, its generalization ability towards unseen bona fide speech is assessed using speech data from the VSDC dataset.

{{</citation>}}


## cs.RO (4)



### (58/122) Bridging Low-level Geometry to High-level Concepts in Visual Servoing of Robot Manipulation Task Using Event Knowledge Graphs and Vision-Language Models (Chen Jiang et al., 2023)

{{<citation>}}

Chen Jiang, Martin Jagersand. (2023)  
**Bridging Low-level Geometry to High-level Concepts in Visual Servoing of Robot Manipulation Task Using Event Knowledge Graphs and Vision-Language Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Knowledge Graph, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03932v1)  

---


**ABSTRACT**  
In this paper, we propose a framework of building knowledgeable robot control in the scope of smart human-robot interaction, by empowering a basic uncalibrated visual servoing controller with contextual knowledge through the joint usage of event knowledge graphs (EKGs) and large-scale pretrained vision-language models (VLMs). The framework is expanded in twofold: first, we interpret low-level image geometry as high-level concepts, allowing us to prompt VLMs and to select geometric features of points and lines for motor control skills; then, we create an event knowledge graph (EKG) to conceptualize a robot manipulation task of interest, where the main body of the EKG is characterized by an executable behavior tree, and the leaves by semantic concepts relevant to the manipulation context. We demonstrate, in an uncalibrated environment with real robot trials, that our method lowers the reliance of human annotation during task interfacing, allows the robot to perform activities of daily living more easily by treating low-level geometric-based motor control skills as high-level concepts, and is beneficial in building cognitive thinking for smart robot applications.

{{</citation>}}


### (59/122) TRAIL Team Description Paper for RoboCup@Home 2023 (Chikaha Tsuji et al., 2023)

{{<citation>}}

Chikaha Tsuji, Dai Komukai, Mimo Shirasaka, Hikaru Wada, Tsunekazu Omija, Aoi Horo, Daiki Furuta, Saki Yamaguchi, So Ikoma, Soshi Tsunashima, Masato Kobayashi, Koki Ishimoto, Yuya Ikeda, Tatsuya Matsushima, Yusuke Iwasawa, Yutaka Matsuo. (2023)  
**TRAIL Team Description Paper for RoboCup@Home 2023**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03913v1)  

---


**ABSTRACT**  
Our team, TRAIL, consists of AI/ML laboratory members from The University of Tokyo. We leverage our extensive research experience in state-of-the-art machine learning to build general-purpose in-home service robots. We previously participated in two competitions using Human Support Robot (HSR): RoboCup@Home Japan Open 2020 (DSPL) and World Robot Summit 2020, equivalent to RoboCup World Tournament. Throughout the competitions, we showed that a data-driven approach is effective for performing in-home tasks. Aiming for further development of building a versatile and fast-adaptable system, in RoboCup @Home 2023, we unify three technologies that have recently been evaluated as components in the fields of deep learning and robot learning into a real household robot system. In addition, to stimulate research all over the RoboCup@Home community, we build a platform that manages data collected from each site belonging to the community around the world, taking advantage of the characteristics of the community.

{{</citation>}}


### (60/122) Progressive Adaptive Chance-Constrained Safeguards for Reinforcement Learning (Zhaorun Chen et al., 2023)

{{<citation>}}

Zhaorun Chen, Binhao Chen, Tairan He, Liang Gong, Chengliang Liu. (2023)  
**Progressive Adaptive Chance-Constrained Safeguards for Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03379v1)  

---


**ABSTRACT**  
Safety assurance of Reinforcement Learning (RL) is critical for exploration in real-world scenarios. In handling the Constrained Markov Decision Process, current approaches experience intrinsic difficulties in trading-off between optimality and feasibility. Direct optimization methods cannot strictly guarantee state-wise in-training safety while projection-based methods are usually inefficient and correct actions through lengthy iterations. To address these two challenges, this paper proposes an adaptive surrogate chance constraint for the safety cost, and a hierarchical architecture that corrects actions produced by the upper policy layer via a fast Quasi-Newton method. Theoretical analysis indicates that the relaxed probabilistic constraint can sufficiently guarantee forward invariance to the safe set. We validate the proposed method on 4 simulated and real-world safety-critical robotic tasks. Results indicate that the proposed method can efficiently enforce safety (nearly zero-violation), while preserving optimality (+23.8%), robustness and generalizability to stochastic real-world settings.

{{</citation>}}


### (61/122) A Two-stage Based Social Preference Recognition in Multi-Agent Autonomous Driving System (Jintao Xue et al., 2023)

{{<citation>}}

Jintao Xue, Dongkun Zhang, Rong Xiong, Yue Wang, Eryun Liu. (2023)  
**A Two-stage Based Social Preference Recognition in Multi-Agent Autonomous Driving System**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03303v1)  

---


**ABSTRACT**  
Multi-Agent Reinforcement Learning (MARL) has become a promising solution for constructing a multi-agent autonomous driving system (MADS) in complex and dense scenarios. But most methods consider agents acting selfishly, which leads to conflict behaviors. Some existing works incorporate the concept of social value orientation (SVO) to promote coordination, but they lack the knowledge of other agents' SVOs, resulting in conservative maneuvers. In this paper, we aim to tackle the mentioned problem by enabling the agents to understand other agents' SVOs. To accomplish this, we propose a two-stage system framework. Firstly, we train a policy by allowing the agents to share their ground truth SVOs to establish a coordinated traffic flow. Secondly, we develop a recognition network that estimates agents' SVOs and integrates it with the policy trained in the first stage. Experiments demonstrate that our developed method significantly improves the performance of the driving policy in MADS compared to two state-of-the-art MARL algorithms.

{{</citation>}}


## cs.IR (3)



### (62/122) An Efficient Content-based Time Series Retrieval System (Chin-Chia Michael Yeh et al., 2023)

{{<citation>}}

Chin-Chia Michael Yeh, Huiyuan Chen, Xin Dai, Yan Zheng, Junpeng Wang, Vivian Lai, Yujie Fan, Audrey Der, Zhongfang Zhuang, Liang Wang, Wei Zhang, Jeff M. Phillips. (2023)  
**An Efficient Content-based Time Series Retrieval System**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.03919v1)  

---


**ABSTRACT**  
A Content-based Time Series Retrieval (CTSR) system is an information retrieval system for users to interact with time series emerged from multiple domains, such as finance, healthcare, and manufacturing. For example, users seeking to learn more about the source of a time series can submit the time series as a query to the CTSR system and retrieve a list of relevant time series with associated metadata. By analyzing the retrieved metadata, users can gather more information about the source of the time series. Because the CTSR system is required to work with time series data from diverse domains, it needs a high-capacity model to effectively measure the similarity between different time series. On top of that, the model within the CTSR system has to compute the similarity scores in an efficient manner as the users interact with the system in real-time. In this paper, we propose an effective and efficient CTSR model that outperforms alternative models, while still providing reasonable inference runtimes. To demonstrate the capability of the proposed method in solving business problems, we compare it against alternative models using our in-house transaction data. Our findings reveal that the proposed model is the most suitable solution compared to others for our transaction data problem.

{{</citation>}}


### (63/122) TPDR: A Novel Two-Step Transformer-based Product and Class Description Match and Retrieval Method (Washington Cunha et al., 2023)

{{<citation>}}

Washington Cunha, Celso França, Leonardo Rocha, Marcos André Gonçalves. (2023)  
**TPDR: A Novel Two-Step Transformer-based Product and Class Description Match and Retrieval Method**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs-SE, cs.IR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.03491v1)  

---


**ABSTRACT**  
There is a niche of companies responsible for intermediating the purchase of large batches of varied products for other companies, for which the main challenge is to perform product description standardization, i.e., matching an item described by a client with a product described in a catalog. The problem is complex since the client's product description may be: (1) potentially noisy; (2) short and uninformative (e.g., missing information about model and size); and (3) cross-language. In this paper, we formalize this problem as a ranking task: given an initial client product specification (query), return the most appropriate standardized descriptions (response). In this paper, we propose TPDR, a two-step Transformer-based Product and Class Description Retrieval method that is able to explore the semantic correspondence between IS and SD, by exploiting attention mechanisms and contrastive learning. First, TPDR employs the transformers as two encoders sharing the embedding vector space: one for encoding the IS and another for the SD, in which corresponding pairs (IS, SD) must be close in the vector space. Closeness is further enforced by a contrastive learning mechanism leveraging a specialized loss function. TPDR also exploits a (second) re-ranking step based on syntactic features that are very important for the exact matching (model, dimension) of certain products that may have been neglected by the transformers. To evaluate our proposal, we consider 11 datasets from a real company, covering different application contexts. Our solution was able to retrieve the correct standardized product before the 5th ranking position in 71% of the cases and its correct category in the first position in 80% of the situations. Moreover, the effectiveness gains over purely syntactic or semantic baselines reach up to 3.7 times, solving cases that none of the approaches in isolation can do by themselves.

{{</citation>}}


### (64/122) Personalized Transformer-based Ranking for e-Commerce at Yandex (Kirill Khrylchenko et al., 2023)

{{<citation>}}

Kirill Khrylchenko, Alexander Fritzler. (2023)  
**Personalized Transformer-based Ranking for e-Commerce at Yandex**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.03481v2)  

---


**ABSTRACT**  
Personalizing user experience with high-quality recommendations based on user activity is vital for e-commerce platforms. This is particularly important in scenarios where the user's intent is not explicit, such as on the homepage. Recently, personalized embedding-based systems have significantly improved the quality of recommendations and search in the e-commerce domain. However, most of these works focus on enhancing the retrieval stage. In this paper, we demonstrate that features produced by retrieval-focused deep learning models are sub-optimal for ranking stage in e-commerce recommendations. To address this issue, we propose a two-stage training process that fine-tunes two-tower models to achieve optimal ranking performance. We provide a detailed description of our transformer-based two-tower model architecture, which is specifically designed for personalization in e-commerce. Additionally, we introduce a novel technique for debiasing context in offline models and report significant improvements in ranking performance when using web-search queries for e-commerce recommendations. Our model has been successfully deployed at Yandex, serves millions of users daily, and has delivered strong performance in online A/B testing.

{{</citation>}}


## cs.CL (30)



### (65/122) Evaluating Multi-Agent Coordination Abilities in Large Language Models (Saaket Agashe et al., 2023)

{{<citation>}}

Saaket Agashe, Yue Fan, Xin Eric Wang. (2023)  
**Evaluating Multi-Agent Coordination Abilities in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-MA, cs.CL  
Keywords: AI, Language Model, Reasoning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03903v1)  

---


**ABSTRACT**  
A pivotal aim in contemporary AI research is to develop agents proficient in multi-agent coordination, enabling effective collaboration with both humans and other systems. Large Language Models (LLMs), with their notable ability to understand, generate, and interpret language in a human-like manner, stand out as promising candidates for the development of such agents. In this study, we build and assess the effectiveness of agents crafted using LLMs in various coordination scenarios. We introduce the LLM-Coordination (LLM-Co) Framework, specifically designed to enable LLMs to play coordination games. With the LLM-Co framework, we conduct our evaluation with three game environments and organize the evaluation into five aspects: Theory of Mind, Situated Reasoning, Sustained Coordination, Robustness to Partners, and Explicit Assistance. First, the evaluation of the Theory of Mind and Situated Reasoning reveals the capabilities of LLM to infer the partner's intention and reason actions accordingly. Then, the evaluation around Sustained Coordination and Robustness to Partners further showcases the ability of LLMs to coordinate with an unknown partner in complex long-horizon tasks, outperforming Reinforcement Learning baselines. Lastly, to test Explicit Assistance, which refers to the ability of an agent to offer help proactively, we introduce two novel layouts into the Overcooked-AI benchmark, examining if agents can prioritize helping their partners, sacrificing time that could have been spent on their tasks. This research underscores the promising capabilities of LLMs in sophisticated coordination environments and reveals the potential of LLMs in building strong real-world agents for multi-agent coordination.

{{</citation>}}


### (66/122) Automatic and Human-AI Interactive Text Generation (Yao Dou et al., 2023)

{{<citation>}}

Yao Dou, Philippe Laban, Claire Gardent, Wei Xu. (2023)  
**Automatic and Human-AI Interactive Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, NLP, Text Generation  
[Paper Link](http://arxiv.org/abs/2310.03878v1)  

---


**ABSTRACT**  
In this tutorial, we focus on text-to-text generation, a class of natural language generation (NLG) tasks, that takes a piece of text as input and then generates a revision that is improved according to some specific criteria (e.g., readability or linguistic styles), while largely retaining the original meaning and the length of the text. This includes many useful applications, such as text simplification, paraphrase generation, style transfer, etc. In contrast to text summarization and open-ended text completion (e.g., story), the text-to-text generation tasks we discuss in this tutorial are more constrained in terms of semantic consistency and targeted language styles. This level of control makes these tasks ideal testbeds for studying the ability of models to generate text that is both semantically adequate and stylistically appropriate. Moreover, these tasks are interesting from a technical standpoint, as they require complex combinations of lexical and syntactical transformations, stylistic control, and adherence to factual knowledge, -- all at once. With a special focus on text simplification and revision, this tutorial aims to provide an overview of the state-of-the-art natural language generation research from four major aspects -- Data, Models, Human-AI Collaboration, and Evaluation -- and to discuss and showcase a few significant and recent advances: (1) the use of non-retrogressive approaches; (2) the shift from fine-tuning to prompting with large language models; (3) the development of new learnable metric and fine-grained human evaluation framework; (4) a growing body of studies and datasets on non-English languages; (5) the rise of HCI+NLP+Accessibility interdisciplinary research to create real-world writing assistant systems.

{{</citation>}}


### (67/122) MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning (Ke Wang et al., 2023)

{{<citation>}}

Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, Hongsheng Li. (2023)  
**MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, PaLM, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.03731v1)  

---


**ABSTRACT**  
The recently released GPT-4 Code Interpreter has demonstrated remarkable proficiency in solving challenging math problems, primarily attributed to its ability to seamlessly reason with natural language, generate code, execute code, and continue reasoning based on the execution output. In this paper, we present a method to fine-tune open-source language models, enabling them to use code for modeling and deriving math equations and, consequently, enhancing their mathematical reasoning abilities. We propose a method of generating novel and high-quality datasets with math problems and their code-based solutions, referred to as MathCodeInstruct. Each solution interleaves natural language, code, and execution results. We also introduce a customized supervised fine-tuning and inference approach. This approach yields the MathCoder models, a family of models capable of generating code-based solutions for solving challenging math problems. Impressively, the MathCoder models achieve state-of-the-art scores among open-source LLMs on the MATH (45.2%) and GSM8K (83.9%) datasets, substantially outperforming other open-source alternatives. Notably, the MathCoder model not only surpasses ChatGPT-3.5 and PaLM-2 on GSM8K and MATH but also outperforms GPT-4 on the competition-level MATH dataset. The dataset and models will be released at https://github.com/mathllm/MathCoder.

{{</citation>}}


### (68/122) Modular Speech-to-Text Translation for Zero-Shot Cross-Modal Transfer (Paul-Ambroise Duquenne et al., 2023)

{{<citation>}}

Paul-Ambroise Duquenne, Holger Schwenk, Benoît Sagot. (2023)  
**Modular Speech-to-Text Translation for Zero-Shot Cross-Modal Transfer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.03724v1)  

---


**ABSTRACT**  
Recent research has shown that independently trained encoders and decoders, combined through a shared fixed-size representation, can achieve competitive performance in speech-to-text translation. In this work, we show that this type of approach can be further improved with multilingual training. We observe significant improvements in zero-shot cross-modal speech translation, even outperforming a supervised approach based on XLSR for several languages.

{{</citation>}}


### (69/122) A Long Way to Go: Investigating Length Correlations in RLHF (Prasann Singhal et al., 2023)

{{<citation>}}

Prasann Singhal, Tanya Goyal, Jiacheng Xu, Greg Durrett. (2023)  
**A Long Way to Go: Investigating Length Correlations in RLHF**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03716v1)  

---


**ABSTRACT**  
Great successes have been reported using Reinforcement Learning from Human Feedback (RLHF) to align large language models. Open-source preference datasets and reward models have enabled wider experimentation beyond generic chat settings, particularly to make systems more "helpful" for tasks like web question answering, summarization, and multi-turn dialogue. When optimizing for helpfulness, RLHF has been consistently observed to drive models to produce longer outputs. This paper demonstrates that optimizing for response length is a significant factor behind RLHF's reported improvements in these settings. First, we study the relationship between reward and length for reward models trained on three open-source preference datasets for helpfulness. Here, length correlates strongly with reward, and improvements in reward score are driven in large part by shifting the distribution over output lengths. We then explore interventions during both RL and reward model learning to see if we can achieve the same downstream improvements as RLHF without increasing length. While our interventions mitigate length increases, they aren't uniformly effective across settings. Furthermore, we find that even running RLHF with a reward based solely on length can reproduce most of the downstream improvements over the initial policy model, showing that reward models in these settings have a long way to go.

{{</citation>}}


### (70/122) DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines (Omar Khattab et al., 2023)

{{<citation>}}

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, Christopher Potts. (2023)  
**DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2310.03714v1)  

---


**ABSTRACT**  
The ML community is rapidly exploring techniques for prompting language models (LMs) and for stacking them into pipelines that solve complex tasks. Unfortunately, existing LM pipelines are typically implemented using hard-coded "prompt templates", i.e. lengthy strings discovered via trial and error. Toward a more systematic approach for developing and optimizing LM pipelines, we introduce DSPy, a programming model that abstracts LM pipelines as text transformation graphs, i.e. imperative computational graphs where LMs are invoked through declarative modules. DSPy modules are parameterized, meaning they can learn (by creating and collecting demonstrations) how to apply compositions of prompting, finetuning, augmentation, and reasoning techniques. We design a compiler that will optimize any DSPy pipeline to maximize a given metric. We conduct two case studies, showing that succinct DSPy programs can express and optimize sophisticated LM pipelines that reason about math word problems, tackle multi-hop retrieval, answer complex questions, and control agent loops. Within minutes of compiling, a few lines of DSPy allow GPT-3.5 and llama2-13b-chat to self-bootstrap pipelines that outperform standard few-shot prompting (generally by over 25% and 65%, respectively) and pipelines with expert-created demonstrations (by up to 5-46% and 16-40%, respectively). On top of that, DSPy programs compiled to open and relatively small LMs like 770M-parameter T5 and llama2-13b-chat are competitive with approaches that rely on expert-written prompt chains for proprietary GPT-3.5. DSPy is available at https://github.com/stanfordnlp/dspy

{{</citation>}}


### (71/122) Agent Instructs Large Language Models to be General Zero-Shot Reasoners (Nicholas Crispino et al., 2023)

{{<citation>}}

Nicholas Crispino, Kyle Montgomery, Fankun Zeng, Dawn Song, Chenguang Wang. (2023)  
**Agent Instructs Large Language Models to be General Zero-Shot Reasoners**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.03710v1)  

---


**ABSTRACT**  
We introduce a method to improve the zero-shot reasoning abilities of large language models on general language understanding tasks. Specifically, we build an autonomous agent to instruct the reasoning process of large language models. We show this approach further unleashes the zero-shot reasoning abilities of large language models to more tasks. We study the performance of our method on a wide set of datasets spanning generation, classification, and reasoning. We show that our method generalizes to most tasks and obtains state-of-the-art zero-shot performance on 20 of the 29 datasets that we evaluate. For instance, our method boosts the performance of state-of-the-art large language models by a large margin, including Vicuna-13b (13.3%), Llama-2-70b-chat (23.2%), and GPT-3.5 Turbo (17.0%). Compared to zero-shot chain of thought, our improvement in reasoning is striking, with an average increase of 10.5%. With our method, Llama-2-70b-chat outperforms zero-shot GPT-3.5 Turbo by 10.2%.

{{</citation>}}


### (72/122) Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To! (Xiangyu Qi et al., 2023)

{{<citation>}}

Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, Peter Henderson. (2023)  
**Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: AI, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03693v1)  

---


**ABSTRACT**  
Optimizing large language models (LLMs) for downstream use cases often involves the customization of pre-trained LLMs through further fine-tuning. Meta's open release of Llama models and OpenAI's APIs for fine-tuning GPT-3.5 Turbo on custom datasets also encourage this practice. But, what are the safety costs associated with such custom fine-tuning? We note that while existing safety alignment infrastructures can restrict harmful behaviors of LLMs at inference time, they do not cover safety risks when fine-tuning privileges are extended to end-users. Our red teaming studies find that the safety alignment of LLMs can be compromised by fine-tuning with only a few adversarially designed training examples. For instance, we jailbreak GPT-3.5 Turbo's safety guardrails by fine-tuning it on only 10 such examples at a cost of less than $0.20 via OpenAI's APIs, making the model responsive to nearly any harmful instructions. Disconcertingly, our research also reveals that, even without malicious intent, simply fine-tuning with benign and commonly used datasets can also inadvertently degrade the safety alignment of LLMs, though to a lesser extent. These findings suggest that fine-tuning aligned LLMs introduces new safety risks that current safety infrastructures fall short of addressing -- even if a model's initial safety alignment is impeccable, it is not necessarily to be maintained after custom fine-tuning. We outline and critically analyze potential mitigations and advocate for further research efforts toward reinforcing safety protocols for the custom fine-tuning of aligned LLMs.

{{</citation>}}


### (73/122) DecoderLens: Layerwise Interpretation of Encoder-Decoder Transformers (Anna Langedijk et al., 2023)

{{<citation>}}

Anna Langedijk, Hosein Mohebbi, Gabriele Sarti, Willem Zuidema, Jaap Jumelet. (2023)  
**DecoderLens: Layerwise Interpretation of Encoder-Decoder Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.03686v1)  

---


**ABSTRACT**  
In recent years, many interpretability methods have been proposed to help interpret the internal states of Transformer-models, at different levels of precision and complexity. Here, to analyze encoder-decoder Transformers, we propose a simple, new method: DecoderLens. Inspired by the LogitLens (for decoder-only Transformers), this method involves allowing the decoder to cross-attend representations of intermediate encoder layers instead of using the final encoder output, as is normally done in encoder-decoder models. The method thus maps previously uninterpretable vector representations to human-interpretable sequences of words or symbols. We report results from the DecoderLens applied to models trained on question answering, logical reasoning, speech recognition and machine translation. The DecoderLens reveals several specific subtasks that are solved at low or intermediate layers, shedding new light on the information flow inside the encoder component of this important class of models.

{{</citation>}}


### (74/122) GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction (Oscar Sainz et al., 2023)

{{<citation>}}

Oscar Sainz, Iker García-Ferrero, Rodrigo Agerri, Oier Lopez de Lacalle, German Rigau, Eneko Agirre. (2023)  
**GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.03668v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) combined with instruction tuning have made significant progress when generalizing to unseen tasks. However, they have been less successful in Information Extraction (IE), lagging behind task-specific models. Typically, IE tasks are characterized by complex annotation guidelines which describe the task and give examples to humans. Previous attempts to leverage such information have failed, even with the largest models, as they are not able to follow the guidelines out-of-the-box. In this paper we propose GoLLIE (Guideline-following Large Language Model for IE), a model able to improve zero-shot results on unseen IE tasks by virtue of being fine-tuned to comply with annotation guidelines. Comprehensive evaluation empirically demonstrates that GoLLIE is able to generalize to and follow unseen guidelines, outperforming previous attempts at zero-shot information extraction. The ablation study shows that detailed guidelines is key for good results.

{{</citation>}}


### (75/122) MapperGPT: Large Language Models for Linking and Mapping Entities (Nicolas Matentzoglu et al., 2023)

{{<citation>}}

Nicolas Matentzoglu, J. Harry Caufield, Harshad B. Hegde, Justin T. Reese, Sierra Moxon, Hyeongsik Kim, Nomi L. Harris, Melissa A Haendel, Christopher J. Mungall. (2023)  
**MapperGPT: Large Language Models for Linking and Mapping Entities**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03666v1)  

---


**ABSTRACT**  
Aligning terminological resources, including ontologies, controlled vocabularies, taxonomies, and value sets is a critical part of data integration in many domains such as healthcare, chemistry, and biomedical research. Entity mapping is the process of determining correspondences between entities across these resources, such as gene identifiers, disease concepts, or chemical entity identifiers. Many tools have been developed to compute such mappings based on common structural features and lexical information such as labels and synonyms. Lexical approaches in particular often provide very high recall, but low precision, due to lexical ambiguity. As a consequence of this, mapping efforts often resort to a labor intensive manual mapping refinement through a human curator.   Large Language Models (LLMs), such as the ones employed by ChatGPT, have generalizable abilities to perform a wide range of tasks, including question-answering and information extraction. Here we present MapperGPT, an approach that uses LLMs to review and refine mapping relationships as a post-processing step, in concert with existing high-recall methods that are based on lexical and structural heuristics.   We evaluated MapperGPT on a series of alignment tasks from different domains, including anatomy, developmental biology, and renal diseases. We devised a collection of tasks that are designed to be particularly challenging for lexical methods. We show that when used in combination with high-recall methods, MapperGPT can provide a substantial improvement in accuracy, beating state-of-the-art (SOTA) methods such as LogMap.

{{</citation>}}


### (76/122) Evaluating Self-Supervised Speech Representations for Indigenous American Languages (Chih-Chen Chen et al., 2023)

{{<citation>}}

Chih-Chen Chen, William Chen, Rodolfo Zevallos, John E. Ortega. (2023)  
**Evaluating Self-Supervised Speech Representations for Indigenous American Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.03639v2)  

---


**ABSTRACT**  
The application of self-supervision to speech representation learning has garnered significant interest in recent years, due to its scalability to large amounts of unlabeled data. However, much progress, both in terms of pre-training and downstream evaluation, has remained concentrated in monolingual models that only consider English. Few models consider other languages, and even fewer consider indigenous ones. In our submission to the New Language Track of the ASRU 2023 ML-SUPERB Challenge, we present an ASR corpus for Quechua, an indigenous South American Language. We benchmark the efficacy of large SSL models on Quechua, along with 6 other indigenous languages such as Guarani and Bribri, on low-resource ASR. Our results show surprisingly strong performance by state-of-the-art SSL models, showing the potential generalizability of large-scale models to real-world data.

{{</citation>}}


### (77/122) How toxic is antisemitism? Potentials and limitations of automated toxicity scoring for antisemitic online content (Helena Mihaljević et al., 2023)

{{<citation>}}

Helena Mihaljević, Elisabeth Steffen. (2023)  
**How toxic is antisemitism? Potentials and limitations of automated toxicity scoring for antisemitic online content**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: Google, Twitter  
[Paper Link](http://arxiv.org/abs/2310.04465v1)  

---


**ABSTRACT**  
The Perspective API, a popular text toxicity assessment service by Google and Jigsaw, has found wide adoption in several application areas, notably content moderation, monitoring, and social media research. We examine its potentials and limitations for the detection of antisemitic online content that, by definition, falls under the toxicity umbrella term. Using a manually annotated German-language dataset comprising around 3,600 posts from Telegram and Twitter, we explore as how toxic antisemitic texts are rated and how the toxicity scores differ regarding different subforms of antisemitism and the stance expressed in the texts. We show that, on a basic level, Perspective API recognizes antisemitic content as toxic, but shows critical weaknesses with respect to non-explicit forms of antisemitism and texts taking a critical stance towards it. Furthermore, using simple text manipulations, we demonstrate that the use of widespread antisemitic codes can substantially reduce API scores, making it rather easy to bypass content moderation based on the service's results.

{{</citation>}}


### (78/122) Redefining Digital Health Interfaces with Large Language Models (Fergus Imrie et al., 2023)

{{<citation>}}

Fergus Imrie, Paulius Rauba, Mihaela van der Schaar. (2023)  
**Redefining Digital Health Interfaces with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03560v1)  

---


**ABSTRACT**  
Digital health tools have the potential to significantly improve the delivery of healthcare services. However, their use remains comparatively limited due, in part, to challenges surrounding usability and trust. Recently, Large Language Models (LLMs) have emerged as general-purpose models with the ability to process complex information and produce human-quality text, presenting a wealth of potential applications in healthcare. Directly applying LLMs in clinical settings is not straightforward, with LLMs susceptible to providing inconsistent or nonsensical answers. We demonstrate how LLMs can utilize external tools to provide a novel interface between clinicians and digital technologies. This enhances the utility and practical impact of digital healthcare tools and AI models while addressing current issues with using LLM in clinical settings such as hallucinations. We illustrate our approach with examples from cardiovascular disease and diabetes risk prediction, highlighting the benefit compared to traditional interfaces for digital tools.

{{</citation>}}


### (79/122) PrIeD-KIE: Towards Privacy Preserved Document Key Information Extraction (Saifullah Saifullah et al., 2023)

{{<citation>}}

Saifullah Saifullah, Stefan Agne, Andreas Dengel, Sheraz Ahmed. (2023)  
**PrIeD-KIE: Towards Privacy Preserved Document Key Information Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Information Extraction  
[Paper Link](http://arxiv.org/abs/2310.03777v1)  

---


**ABSTRACT**  
In this paper, we introduce strategies for developing private Key Information Extraction (KIE) systems by leveraging large pretrained document foundation models in conjunction with differential privacy (DP), federated learning (FL), and Differentially Private Federated Learning (DP-FL). Through extensive experimentation on six benchmark datasets (FUNSD, CORD, SROIE, WildReceipts, XFUND, and DOCILE), we demonstrate that large document foundation models can be effectively fine-tuned for the KIE task under private settings to achieve adequate performance while maintaining strong privacy guarantees. Moreover, by thoroughly analyzing the impact of various training and model parameters on model performance, we propose simple yet effective guidelines for achieving an optimal privacy-utility trade-off for the KIE task under global DP. Finally, we introduce FeAm-DP, a novel DP-FL algorithm that enables efficiently upscaling global DP from a standalone context to a multi-client federated environment. We conduct a comprehensive evaluation of the algorithm across various client and privacy settings, and demonstrate its capability to achieve comparable performance and privacy guarantees to standalone DP, even when accommodating an increasing number of participating clients. Overall, our study offers valuable insights into the development of private KIE systems, and highlights the potential of document foundation models for privacy-preserved Document AI applications. To the best of authors' knowledge, this is the first work that explores privacy preserved document KIE using document foundation models.

{{</citation>}}


### (80/122) Tik-to-Tok: Translating Language Models One Token at a Time: An Embedding Initialization Strategy for Efficient Language Adaptation (François Remy et al., 2023)

{{<citation>}}

François Remy, Pieter Delobelle, Bettina Berendt, Kris Demuynck, Thomas Demeester. (2023)  
**Tik-to-Tok: Translating Language Models One Token at a Time: An Embedding Initialization Strategy for Efficient Language Adaptation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03477v1)  

---


**ABSTRACT**  
Training monolingual language models for low and mid-resource languages is made challenging by limited and often inadequate pretraining data. In this study, we propose a novel model conversion strategy to address this issue, adapting high-resources monolingual language models to a new target language. By generalizing over a word translation dictionary encompassing both the source and target languages, we map tokens from the target tokenizer to semantically similar tokens from the source language tokenizer. This one-to-many token mapping improves tremendously the initialization of the embedding table for the target language. We conduct experiments to convert high-resource models to mid- and low-resource languages, namely Dutch and Frisian. These converted models achieve a new state-of-the-art performance on these languages across all sorts of downstream tasks. By reducing significantly the amount of data and time required for training state-of-the-art models, our novel model conversion strategy has the potential to benefit many languages worldwide.

{{</citation>}}


### (81/122) Controllable Multi-document Summarization: Coverage & Coherence Intuitive Policy with Large Language Model Based Rewards (Litton J Kurisinkel et al., 2023)

{{<citation>}}

Litton J Kurisinkel, Nancy F chen. (2023)  
**Controllable Multi-document Summarization: Coverage & Coherence Intuitive Policy with Large Language Model Based Rewards**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2310.03473v1)  

---


**ABSTRACT**  
Memory-efficient large language models are good at refining text input for better readability. However, controllability is a matter of concern when it comes to text generation tasks with long inputs, such as multi-document summarization. In this work, we investigate for a generic controllable approach for multi-document summarization that leverages the capabilities of LLMs to refine the text. In particular, we train a controllable content extraction scheme to extract the text that will be refined by an LLM. The scheme is designed with a novel coverage and coherence intuitive policy, which is duly rewarded by a passively trained LLM. Our approach yields competitive results in the evaluation using ROUGE metrics and outperforms potential baselines in coherence, as per human evaluation.

{{</citation>}}


### (82/122) The North System for Formosa Speech Recognition Challenge 2023 (Li-Wei Chen et al., 2023)

{{<citation>}}

Li-Wei Chen, Kai-Chen Cheng, Hung-Shin Lee. (2023)  
**The North System for Formosa Speech Recognition Challenge 2023**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.03443v2)  

---


**ABSTRACT**  
This report provides a concise overview of the proposed North system, which aims to achieve automatic word/syllable recognition for Taiwanese Hakka (Sixian). The report outlines three key components of the system: the acquisition, composition, and utilization of the training data; the architecture of the model; and the hardware specifications and operational statistics. The demonstration of the system has been made public at https://asrvm.iis.sinica.edu.tw/hakka_sixian.

{{</citation>}}


### (83/122) LLM Based Multi-Document Summarization Exploiting Main-Event Biased Monotone Submodular Content Extraction (Litton J Kurisinkel et al., 2023)

{{<citation>}}

Litton J Kurisinkel, Nancy F. Chen. (2023)  
**LLM Based Multi-Document Summarization Exploiting Main-Event Biased Monotone Submodular Content Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2310.03414v1)  

---


**ABSTRACT**  
Multi-document summarization is a challenging task due to its inherent subjective bias, highlighted by the low inter-annotator ROUGE-1 score of 0.4 among DUC-2004 reference summaries. In this work, we aim to enhance the objectivity of news summarization by focusing on the main event of a group of related news documents and presenting it coherently with sufficient context. Our primary objective is to succinctly report the main event, ensuring that the summary remains objective and informative. To achieve this, we employ an extract-rewrite approach that incorporates a main-event biased monotone-submodular function for content selection. This enables us to extract the most crucial information related to the main event from the document cluster. To ensure coherence, we utilize a fine-tuned Language Model (LLM) for rewriting the extracted content into a coherent text. The evaluation using objective metrics and human evaluators confirms the effectiveness of our approach, as it surpasses potential baselines, demonstrating excellence in both content coverage, coherence, and informativeness.

{{</citation>}}


### (84/122) Procedural Text Mining with Large Language Models (Anisa Rula et al., 2023)

{{<citation>}}

Anisa Rula, Jennifer D'Souza. (2023)  
**Procedural Text Mining with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IT, cs.CL, math-IT  
Keywords: GPT, GPT-4, Language Model, Natural Language Processing, Transformer  
[Paper Link](http://arxiv.org/abs/2310.03376v1)  

---


**ABSTRACT**  
Recent advancements in the field of Natural Language Processing, particularly the development of large-scale language models that are pretrained on vast amounts of knowledge, are creating novel opportunities within the realm of Knowledge Engineering. In this paper, we investigate the usage of large language models (LLMs) in both zero-shot and in-context learning settings to tackle the problem of extracting procedures from unstructured PDF text in an incremental question-answering fashion. In particular, we leverage the current state-of-the-art GPT-4 (Generative Pre-trained Transformer 4) model, accompanied by two variations of in-context learning that involve an ontology with definitions of procedures and steps and a limited number of samples of few-shot learning. The findings highlight both the promise of this approach and the value of the in-context learning customisations. These modifications have the potential to significantly address the challenge of obtaining sufficient training data, a hurdle often encountered in deep learning-based Natural Language Processing techniques for procedure extraction.

{{</citation>}}


### (85/122) Evaluating Hallucinations in Chinese Large Language Models (Qinyuan Cheng et al., 2023)

{{<citation>}}

Qinyuan Cheng, Tianxiang Sun, Wenwei Zhang, Siyin Wang, Xiangyang Liu, Mozhi Zhang, Junliang He, Mianqiu Huang, Zhangyue Yin, Kai Chen, Xipeng Qiu. (2023)  
**Evaluating Hallucinations in Chinese Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GLM, GPT, GPT-4, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2310.03368v1)  

---


**ABSTRACT**  
In this paper, we establish a benchmark named HalluQA (Chinese Hallucination Question-Answering) to measure the hallucination phenomenon in Chinese large language models. HalluQA contains 450 meticulously designed adversarial questions, spanning multiple domains, and takes into account Chinese historical culture, customs, and social phenomena. During the construction of HalluQA, we consider two types of hallucinations: imitative falsehoods and factual errors, and we construct adversarial samples based on GLM-130B and ChatGPT. For evaluation, we design an automated evaluation method using GPT-4 to judge whether a model output is hallucinated. We conduct extensive experiments on 24 large language models, including ERNIE-Bot, Baichuan2, ChatGLM, Qwen, SparkDesk and etc. Out of the 24 models, 18 achieved non-hallucination rates lower than 50%. This indicates that HalluQA is highly challenging. We analyze the primary types of hallucinations in different types of models and their causes. Additionally, we discuss which types of hallucinations should be prioritized for different types of models.

{{</citation>}}


### (86/122) Tuning In to Neural Encoding: Linking Human Brain and Artificial Supervised Representations of Language (Jingyuan Sun et al., 2023)

{{<citation>}}

Jingyuan Sun, Xiaohan Zhang, Marie-Francine Moens. (2023)  
**Tuning In to Neural Encoding: Linking Human Brain and Artificial Supervised Representations of Language**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLU, Natural Language Understanding, Transformer  
[Paper Link](http://arxiv.org/abs/2310.04460v1)  

---


**ABSTRACT**  
To understand the algorithm that supports the human brain's language representation, previous research has attempted to predict neural responses to linguistic stimuli using embeddings generated by artificial neural networks (ANNs), a process known as neural encoding. However, most of these studies have focused on probing neural representations of Germanic languages, such as English, with unsupervised ANNs. In this paper, we propose to bridge the gap between human brain and supervised ANN representations of the Chinese language. Specifically, we investigate how task tuning influences a pretained Transformer for neural encoding and which tasks lead to the best encoding performances. We generate supervised representations on eight Natural Language Understanding (NLU) tasks using prompt-tuning, a technique that is seldom explored in neural encoding for language. We demonstrate that prompt-tuning yields representations that better predict neural responses to Chinese stimuli than traditional fine-tuning on four tasks. Furthermore, we discover that tasks that require a fine-grained processing of concepts and entities lead to representations that are most predictive of brain activation patterns. Additionally, we reveal that the proportion of tuned parameters highly influences the neural encoding performance of fine-tuned models. Overall, our experimental findings could help us better understand the relationship between supervised artificial and brain language representations.

{{</citation>}}


### (87/122) Reformulating Domain Adaptation of Large Language Models as Adapt-Retrieve-Revise (Zhen wan et al., 2023)

{{<citation>}}

Zhen wan, Yating Zhang, Yexiang Wang, Fei Cheng, Sadao Kurohashi. (2023)  
**Reformulating Domain Adaptation of Large Language Models as Adapt-Retrieve-Revise**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03328v1)  

---


**ABSTRACT**  
While large language models (LLMs) like GPT-4 have recently demonstrated astonishing zero-shot capabilities in general domain tasks, they often generate content with hallucinations in specific domains such as Chinese law, hindering their application in these areas. This is typically due to the absence of training data that encompasses such a specific domain, preventing GPT-4 from acquiring in-domain knowledge. A pressing challenge is that it's not plausible to continue training LLMs of such scale on in-domain data.   This paper introduces a simple and effective domain adaptation framework for GPT-4 by reformulating generation as an \textbf{adapt-retrieve-revise} process. The initial step is to \textbf{adapt} an affordable 7B LLM to the target domain by continuing learning on in-domain data. When solving a task, we leverage the adapted LLM to generate a draft answer given a task query. Then, the draft answer will be used to \textbf{retrieve} supporting evidence candidates from an external in-domain knowledge base. Finally, the draft answer and retrieved evidence are concatenated into a whole prompt to let GPT-4 assess the evidence and \textbf{revise} the draft answer to generate the final answer.   Our proposal combines the advantages of the efficiency of adapting a smaller 7B model with the evidence-assessing capability of GPT-4 and effectively prevents GPT-4 from generating hallucinatory content. In the zero-shot setting of four Chinese legal tasks, our method improves accuracy by 33.3\% compared to the direct generation by GPT-4. When compared to two stronger retrieval-based baselines, our method outperforms them by 15.4\% and 23.9\%. Our code will be released

{{</citation>}}


### (88/122) Concise and Organized Perception Facilitates Large Language Models for Deductive Reasoning (Shaotian Yan et al., 2023)

{{<citation>}}

Shaotian Yan, Chen Shen, Junjie Liu, Jieping Ye. (2023)  
**Concise and Organized Perception Facilitates Large Language Models for Deductive Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.03309v1)  

---


**ABSTRACT**  
Exploiting large language models (LLMs) to tackle deductive reasoning has garnered growing attention. It still remains highly challenging to achieve satisfactory results in complex deductive problems, characterized by plenty of premises (i.e., facts or rules) entailing intricate relationships among entities and requiring multi-hop reasoning. One intuitive solution is to decompose the original task into smaller sub-tasks, and then chain the multiple casual reasoning steps together in a forward (e.g., Selection-Inference) or backward (e.g., LAMBADA) direction. However, these techniques inevitably necessitate a large number of overall stages, leading to computationally expensive operations and a higher possibility of making misleading steps. In addition to stage-by-stage decomposition, we draw inspiration from another aspect of human problem-solving. Humans tend to distill the most relevant information and organize their thoughts systematically (e.g., creating mind maps), which assists them in answering questions or drawing conclusions precisely and quickly. In light of this, we propose a novel reasoning approach named Concise and Organized Perception (COP). COP carefully analyzes the given statements to efficiently identify the most pertinent information while eliminating redundancy. It then prompts the LLMs in a more organized form that adapts to the model's inference process. By perceiving concise and organized proofs, the deductive reasoning abilities of LLMs can be better elicited, and the risk of acquiring errors caused by excessive reasoning stages is mitigated. Furthermore, our approach can be combined with the aforementioned ones to further boost their performance. Extensive experimental results on three popular deductive benchmarks (i.e., ProofWriter, PrOntoQA and PrOntoQA-OOD) show that COP significantly outperforms previous state-of-the-art methods.

{{</citation>}}


### (89/122) Learning Personalized Story Evaluation (Danqing Wang et al., 2023)

{{<citation>}}

Danqing Wang, Kevin Yang, Hanlin Zhu, Xiaomeng Yang, Andrew Cohen, Lei Li, Yuandong Tian. (2023)  
**Learning Personalized Story Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, QA  
[Paper Link](http://arxiv.org/abs/2310.03304v2)  

---


**ABSTRACT**  
While large language models (LLMs) have shown impressive results for more objective tasks such as QA and retrieval, it remains nontrivial to evaluate their performance on open-ended text generation for reasons including (1) data contamination; (2) multi-dimensional evaluation criteria; and (3) subjectiveness stemming from reviewers' personal preferences. To address such issues, we propose to model personalization in an uncontaminated open-ended generation assessment. We create two new datasets Per-MPST and Per-DOC for personalized story evaluation, by re-purposing existing datasets with proper anonymization and new personalized labels. We further develop a personalized story evaluation model PERSE to infer reviewer preferences and provide a personalized evaluation. Specifically, given a few exemplary reviews from a particular reviewer, PERSE predicts either a detailed review or fine-grained comparison in several aspects (such as interestingness and surprise) for that reviewer on a new text input. Experimental results show that PERSE outperforms GPT-4 by 15.8% on Kendall correlation of story ratings, and by 13.7% on pairwise preference prediction accuracy. Both datasets and code will be released at https://github.com/dqwang122/PerSE.

{{</citation>}}


### (90/122) A New Dialogue Response Generation Agent for Large Language Models by Asking Questions to Detect User's Intentions (Siwei Wu et al., 2023)

{{<citation>}}

Siwei Wu, Xiangqing Shen, Rui Xia. (2023)  
**A New Dialogue Response Generation Agent for Large Language Models by Asking Questions to Detect User's Intentions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.03293v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), such as ChatGPT, have recently been applied to various NLP tasks due to its open-domain generation capabilities. However, there are two issues with applying LLMs to dialogue tasks. 1. During the dialogue process, users may have implicit intentions that might be overlooked by LLMs. Consequently, generated responses couldn't align with the user's intentions. 2. It is unlikely for LLMs to encompass all fields comprehensively. In certain specific domains, their knowledge may be incomplete, and LLMs cannot update the latest knowledge in real-time. To tackle these issues, we propose a framework~\emph{using LLM to \textbf{E}nhance dialogue response generation by asking questions to \textbf{D}etect user's \textbf{I}mplicit in\textbf{T}entions} (\textbf{EDIT}). Firstly, EDIT generates open questions related to the dialogue context as the potential user's intention; Then, EDIT answers those questions by interacting with LLMs and searching in domain-specific knowledge bases respectively, and use LLMs to choose the proper answers to questions as extra knowledge; Finally, EDIT enhances response generation by explicitly integrating those extra knowledge. Besides, previous question generation works only focus on asking questions with answers in context. In order to ask open questions, we construct a Context-Open-Question (COQ) dataset. On two task-oriented dialogue tasks (Wizard of Wikipedia and Holl-E), EDIT outperformed other LLMs.

{{</citation>}}


### (91/122) A Formalism and Approach for Improving Robustness of Large Language Models Using Risk-Adjusted Confidence Scores (Ke Shen et al., 2023)

{{<citation>}}

Ke Shen, Mayank Kejriwal. (2023)  
**A Formalism and Approach for Improving Robustness of Large Language Models Using Risk-Adjusted Confidence Scores**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLI, NLP  
[Paper Link](http://arxiv.org/abs/2310.03283v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), such as ChatGPT, have achieved impressive milestones in natural language processing (NLP). Despite their impressive performance, the models are known to pose important risks. As these models are deployed in real-world applications, a systematic understanding of different risks posed by these models on tasks such as natural language inference (NLI), is much needed. In this paper, we define and formalize two distinct types of risk: decision risk and composite risk. We also propose a risk-centric evaluation framework, and four novel metrics, for assessing LLMs on these risks in both in-domain and out-of-domain settings. Finally, we propose a risk-adjusted calibration method called DwD for helping LLMs minimize these risks in an overall NLI architecture. Detailed experiments, using four NLI benchmarks, three baselines and two LLMs, including ChatGPT, show both the practical utility of the evaluation framework, and the efficacy of DwD in reducing decision and composite risk. For instance, when using DwD, an underlying LLM is able to address an extra 20.1% of low-risk inference tasks (but which the LLM erroneously deems high-risk without risk adjustment) and skip a further 19.8% of high-risk tasks, which would have been answered incorrectly.

{{</citation>}}


### (92/122) Investigating Alternative Feature Extraction Pipelines For Clinical Note Phenotyping (Neil Daniel, 2023)

{{<citation>}}

Neil Daniel. (2023)  
**Investigating Alternative Feature Extraction Pipelines For Clinical Note Phenotyping**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Clinical, LSTM  
[Paper Link](http://arxiv.org/abs/2310.03772v1)  

---


**ABSTRACT**  
A common practice in the medical industry is the use of clinical notes, which consist of detailed patient observations. However, electronic health record systems frequently do not contain these observations in a structured format, rendering patient information challenging to assess and evaluate automatically. Using computational systems for the extraction of medical attributes offers many applications, including longitudinal analysis of patients, risk assessment, and hospital evaluation. Recent work has constructed successful methods for phenotyping: extracting medical attributes from clinical notes. BERT-based models can be used to transform clinical notes into a series of representations, which are then condensed into a single document representation based on their CLS embeddings and passed into an LSTM (Mulyar et al., 2020). Though this pipeline yields a considerable performance improvement over previous results, it requires extensive convergence time. This method also does not allow for predicting attributes not yet identified in clinical notes.   Considering the wide variety of medical attributes that may be present in a clinical note, we propose an alternative pipeline utilizing ScispaCy (Neumann et al., 2019) for the extraction of common diseases. We then train various supervised learning models to associate the presence of these conditions with patient attributes. Finally, we replicate a ClinicalBERT (Alsentzer et al., 2019) and LSTM-based approach for purposes of comparison. We find that alternative methods moderately underperform the replicated LSTM approach. Yet, considering a complex tradeoff between accuracy and runtime, in addition to the fact that the alternative approach also allows for the detection of medical conditions that are not already present in a clinical note, its usage may be considered as a supplement to established methods.

{{</citation>}}


### (93/122) Can Large Language Models be Good Path Planners? A Benchmark and Investigation on Spatial-temporal Reasoning (Mohamed Aghzal et al., 2023)

{{<citation>}}

Mohamed Aghzal, Erion Plaku, Ziyu Yao. (2023)  
**Can Large Language Models be Good Path Planners? A Benchmark and Investigation on Spatial-temporal Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Reasoning, T5  
[Paper Link](http://arxiv.org/abs/2310.03249v1)  

---


**ABSTRACT**  
Large language models (LLMs) have achieved remarkable success across a wide spectrum of tasks; however, they still face limitations in scenarios that demand long-term planning and spatial reasoning. To facilitate this line of research, in this work, we propose a new benchmark, termed $\textbf{P}$ath $\textbf{P}$lanning from $\textbf{N}$atural $\textbf{L}$anguage ($\textbf{PPNL}$). Our benchmark evaluates LLMs' spatial-temporal reasoning by formulating ''path planning'' tasks that require an LLM to navigate to target locations while avoiding obstacles and adhering to constraints. Leveraging this benchmark, we systematically investigate LLMs including GPT-4 via different few-shot prompting methodologies and BART and T5 of various sizes via fine-tuning. Our experimental results show the promise of few-shot GPT-4 in spatial reasoning, when it is prompted to reason and act interleavedly, although it still fails to make long-term temporal reasoning. In contrast, while fine-tuned LLMs achieved impressive results on in-distribution reasoning tasks, they struggled to generalize to larger environments or environments with more obstacles.

{{</citation>}}


### (94/122) FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation (Tu Vu et al., 2023)

{{<citation>}}

Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, Thang Luong. (2023)  
**FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Augmentation, Language Model, Perplexity, QA  
[Paper Link](http://arxiv.org/abs/2310.03214v1)  

---


**ABSTRACT**  
Most large language models (LLMs) are trained once and never updated; thus, they lack the ability to dynamically adapt to our ever-changing world. In this work, we perform a detailed study of the factuality of LLM-generated text in the context of answering questions that test current world knowledge. Specifically, we introduce FreshQA, a novel dynamic QA benchmark encompassing a diverse range of question and answer types, including questions that require fast-changing world knowledge as well as questions with false premises that need to be debunked. We benchmark a diverse array of both closed and open-source LLMs under a two-mode evaluation procedure that allows us to measure both correctness and hallucination. Through human evaluations involving more than 50K judgments, we shed light on limitations of these models and demonstrate significant room for improvement: for instance, all models (regardless of model size) struggle on questions that involve fast-changing knowledge and false premises. Motivated by these results, we present FreshPrompt, a simple few-shot prompting method that substantially boosts the performance of an LLM on FreshQA by incorporating relevant and up-to-date information retrieved from a search engine into the prompt. Our experiments show that FreshPrompt outperforms both competing search engine-augmented prompting methods such as Self-Ask (Press et al., 2022) as well as commercial systems such as Perplexity.AI. Further analysis of FreshPrompt reveals that both the number of retrieved evidences and their order play a key role in influencing the correctness of LLM-generated answers. Additionally, instructing the LLM to generate concise and direct answers helps reduce hallucination compared to encouraging more verbose answers. To facilitate future work, we release FreshQA at github.com/freshllms/freshqa and commit to updating it at regular intervals.

{{</citation>}}


## cs.SI (2)



### (95/122) A Game Approach to Multi-dimensional Opinion Dynamics in Social Networks with Stubborn Strategist Agents (Hossein B. Jond et al., 2023)

{{<citation>}}

Hossein B. Jond, Aykut Yıldız. (2023)  
**A Game Approach to Multi-dimensional Opinion Dynamics in Social Networks with Stubborn Strategist Agents**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2310.03900v1)  

---


**ABSTRACT**  
In a social network, individuals express their opinions on several interdependent topics, and therefore the evolution of their opinions on these topics is also mutually dependent. In this work, we propose a differential game model for the multi-dimensional opinion formation of a social network whose population of agents interacts according to a communication graph. Each individual's opinion evolves according to an aggregation of disagreements between the agent's opinions and its graph neighbors on multiple interdependent topics exposed to an unknown extraneous disturbance. For a social network with strategist agents the opinions evolve over time with respect to the minimization of a quadratic cost function that solely represents each individual's motives against the disturbance. We find the unique Nash/worst-case equilibrium solution for the proposed differential game model of coupled multi-dimensional opinions under an open-loop information structure. Moreover, we propose a distributed implementation of the Nash/worst-case equilibrium solution. We examine the non-distributed and proposed distributed open-loop Nash/worst-case strategies on a small social network with strategist agents in a two-dimensional opinion space. Then we compare the opinions evolved based on the Nash/worst-case strategy with the opinions corresponding to social optimality actions for non-strategist agents.

{{</citation>}}


### (96/122) Differential Game Strategies for Social Networks with Self-Interested Individuals (Hossein B. Jond, 2023)

{{<citation>}}

Hossein B. Jond. (2023)  
**Differential Game Strategies for Social Networks with Self-Interested Individuals**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2310.03774v1)  

---


**ABSTRACT**  
A social network population engages in collective actions as a direct result of forming a particular opinion. The strategic interactions among the individuals acting independently and selfishly naturally portray a noncooperative game. Nash equilibrium allows for self-enforcing strategic interactions between selfish and self-interested individuals. This paper presents a differential game approach to the opinion formation problem in social networks to investigate the evolution of opinions as a result of a Nash equilibrium. The opinion of each individual is described by a differential equation, which is the continuous-time Hegselmann-Krause model for opinion dynamics with a time delay in input. The objective of each individual is to seek optimal strategies for her own opinion evolution by minimizing an individual cost function. Two differential game problems emerge, one for a population that is not stubborn and another for a population that is stubborn. The open-loop Nash equilibrium actions and their associated opinion trajectories are derived for both differential games using Pontryagin's principle. Additionally, the receding horizon control scheme is used to practice feedback strategies where the information flow is restricted by fixed and complete social graphs as well as the second neighborhood concept. The game strategies were executed on the well-known Zachary's Karate Club social network. The resulting opinion trajectories associated with the game strategies showed consensus, polarization, and disagreement in final opinions.

{{</citation>}}


## eess.AS (1)



### (97/122) Audio Event-Relational Graph Representation Learning for Acoustic Scene Classification (Yuanbo Hou et al., 2023)

{{<citation>}}

Yuanbo Hou, Siyang Song, Chuang Yu, Wenwu Wang, Dick Botteldooren. (2023)  
**Audio Event-Relational Graph Representation Learning for Acoustic Scene Classification**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.03889v1)  

---


**ABSTRACT**  
Most deep learning-based acoustic scene classification (ASC) approaches identify scenes based on acoustic features converted from audio clips containing mixed information entangled by polyphonic audio events (AEs). However, these approaches have difficulties in explaining what cues they use to identify scenes. This paper conducts the first study on disclosing the relationship between real-life acoustic scenes and semantic embeddings from the most relevant AEs. Specifically, we propose an event-relational graph representation learning (ERGL) framework for ASC to classify scenes, and simultaneously answer clearly and straightly which cues are used in classifying. In the event-relational graph, embeddings of each event are treated as nodes, while relationship cues derived from each pair of nodes are described by multi-dimensional edge features. Experiments on a real-life ASC dataset show that the proposed ERGL achieves competitive performance on ASC by learning embeddings of only a limited number of AEs. The results show the feasibility of recognizing diverse acoustic scenes based on the audio event-relational graph. Visualizations of graph representations learned by ERGL are available here (https://github.com/Yuanbo2020/ERGL).

{{</citation>}}


## physics.med-ph (1)



### (98/122) Benchmarking a foundation LLM on its ability to re-label structure names in accordance with the AAPM TG-263 report (Jason Holmes et al., 2023)

{{<citation>}}

Jason Holmes, Lian Zhang, Yuzhen Ding, Hongying Feng, Zhengliang Liu, Tianming Liu, William W. Wong, Sujay A. Vora, Jonathan B. Ashman, Wei Liu. (2023)  
**Benchmarking a foundation LLM on its ability to re-label structure names in accordance with the AAPM TG-263 report**  

---
Primary Category: physics.med-ph  
Categories: cs-CL, physics-med-ph, physics.med-ph  
Keywords: GPT, GPT-4, Transformer  
[Paper Link](http://arxiv.org/abs/2310.03874v1)  

---


**ABSTRACT**  
Purpose: To introduce the concept of using large language models (LLMs) to re-label structure names in accordance with the American Association of Physicists in Medicine (AAPM) Task Group (TG)-263 standard, and to establish a benchmark for future studies to reference.   Methods and Materials: The Generative Pre-trained Transformer (GPT)-4 application programming interface (API) was implemented as a Digital Imaging and Communications in Medicine (DICOM) storage server, which upon receiving a structure set DICOM file, prompts GPT-4 to re-label the structure names of both target volumes and normal tissues according to the AAPM TG-263. Three disease sites, prostate, head and neck, and thorax were selected for evaluation. For each disease site category, 150 patients were randomly selected for manually tuning the instructions prompt (in batches of 50) and 50 patients were randomly selected for evaluation. Structure names that were considered were those that were most likely to be relevant for studies utilizing structure contours for many patients.   Results: The overall re-labeling accuracy of both target volumes and normal tissues for prostate, head and neck, and thorax cases was 96.0%, 98.5%, and 96.9% respectively. Re-labeling of target volumes was less accurate on average except for prostate - 100%, 93.1%, and 91.1% respectively.   Conclusions: Given the accuracy of GPT-4 in re-labeling structure names of both target volumes and normal tissues as presented in this work, LLMs are poised to be the preferred method for standardizing structure names in radiation oncology, especially considering the rapid advancements in LLM capabilities that are likely to continue.

{{</citation>}}


## cs.CR (5)



### (99/122) ALBERTA: ALgorithm-Based Error Resilience in Transformer Architectures (Haoxuan Liu et al., 2023)

{{<citation>}}

Haoxuan Liu, Vasu Singh, Michał Filipiuk, Siva Kumar Sastry Hari. (2023)  
**ALBERTA: ALgorithm-Based Error Resilience in Transformer Architectures**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-DC, cs.CR  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.03841v1)  

---


**ABSTRACT**  
Vision Transformers are being increasingly deployed in safety-critical applications that demand high reliability. It is crucial to ensure the correctness of their execution in spite of potential errors such as transient hardware errors. We propose a novel algorithm-based resilience framework called ALBERTA that allows us to perform end-to-end resilience analysis and protection of transformer-based architectures. First, our work develops an efficient process of computing and ranking the resilience of transformers layers. We find that due to the large size of transformer models, applying traditional network redundancy to a subset of the most vulnerable layers provides high error coverage albeit with impractically high overhead. We address this shortcoming by providing a software-directed, checksum-based error detection technique aimed at protecting the most vulnerable general matrix multiply (GEMM) layers in the transformer models that use either floating-point or integer arithmetic. Results show that our approach achieves over 99% coverage for errors that result in a mismatch at less than 0.2% computation overhead. Lastly, we present the applicability of our framework in various modern GPU architectures under different numerical precisions. We introduce an efficient self-correction mechanism for resolving erroneous detection with an average overhead of less than 0.002% (with a 2% overhead to resolve each erroneous detection).

{{</citation>}}


### (100/122) Enhancing Exfiltration Path Analysis Using Reinforcement Learning (Riddam Rishu et al., 2023)

{{<citation>}}

Riddam Rishu, Akshay Kakkar, Cheng Wang, Abdul Rahman, Christopher Redino, Dhruv Nandakumar, Tyler Cody, Ryan Clark, Daniel Radke, Edward Bowen. (2023)  
**Enhancing Exfiltration Path Analysis Using Reinforcement Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03667v1)  

---


**ABSTRACT**  
Building on previous work using reinforcement learning (RL) focused on identification of exfiltration paths, this work expands the methodology to include protocol and payload considerations. The former approach to exfiltration path discovery, where reward and state are associated specifically with the determination of optimal paths, are presented with these additional realistic characteristics to account for nuances in adversarial behavior. The paths generated are enhanced by including communication payload and protocol into the Markov decision process (MDP) in order to more realistically emulate attributes of network based exfiltration events. The proposed method will help emulate complex adversarial considerations such as the size of a payload being exported over time or the protocol on which it occurs, as is the case where threat actors steal data over long periods of time using system native ports or protocols to avoid detection. As such, practitioners will be able to improve identification of expected adversary behavior under various payload and protocol assumptions more comprehensively.

{{</citation>}}


### (101/122) Putting a Padlock on Lambda -- Integrating vTPMs into AWS Firecracker (Melker Veltman et al., 2023)

{{<citation>}}

Melker Veltman, Alexandra Parkegren, Victor Morel. (2023)  
**Putting a Padlock on Lambda -- Integrating vTPMs into AWS Firecracker**  

---
Primary Category: cs.CR  
Categories: cs-AR, cs-CR, cs-CY, cs.CR  
Keywords: AWS, Amazon  
[Paper Link](http://arxiv.org/abs/2310.03522v1)  

---


**ABSTRACT**  
When software services use cloud providers to run their workloads, they place implicit trust in the cloud provider, without an explicit trust relationship. One way to achieve such explicit trust in a computer system is to use a hardware Trusted Platform Module (TPM), a coprocessor for trusted computing. However, in the case of managed platform-as-a-service (PaaS) offerings, there is currently no cloud provider that exposes TPM capabilities. In this paper, we improve trust by integrating a virtual TPM device into the Firecracker hypervisor, originally developed by Amazon Web Services. In addition to this, multiple performance tests along with an attack surface analysis are performed to evaluate the impact of the changes introduced. We discuss the results and conclude that the slight performance decrease and attack surface increase are acceptable trade-offs in order to enable trusted computing in PaaS offerings.

{{</citation>}}


### (102/122) FLAIM: AIM-based Synthetic Data Generation in the Federated Setting (Samuel Maddock et al., 2023)

{{<citation>}}

Samuel Maddock, Graham Cormode, Carsten Maple. (2023)  
**FLAIM: AIM-based Synthetic Data Generation in the Federated Setting**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03447v1)  

---


**ABSTRACT**  
Preserving individual privacy while enabling collaborative data sharing is crucial for organizations. Synthetic data generation is one solution, producing artificial data that mirrors the statistical properties of private data. While numerous techniques have been devised under differential privacy, they predominantly assume data is centralized. However, data is often distributed across multiple clients in a federated manner. In this work, we initiate the study of federated synthetic tabular data generation. Building upon a SOTA central method known as AIM, we present DistAIM and FLAIM. We show it is straightforward to distribute AIM, extending a recent approach based on secure multi-party computation which necessitates additional overhead, making it less suited to federated scenarios. We then demonstrate that naively federating AIM can lead to substantial degradation in utility under the presence of heterogeneity. To mitigate both issues, we propose an augmented FLAIM approach that maintains a private proxy of heterogeneity. We simulate our methods across a range of benchmark datasets under different degrees of heterogeneity and show this can improve utility while reducing overhead.

{{</citation>}}


### (103/122) Certifiably Robust Graph Contrastive Learning (Minhua Lin et al., 2023)

{{<citation>}}

Minhua Lin, Teng Xiao, Enyan Dai, Xiang Zhang, Suhang Wang. (2023)  
**Certifiably Robust Graph Contrastive Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.03312v1)  

---


**ABSTRACT**  
Graph Contrastive Learning (GCL) has emerged as a popular unsupervised graph representation learning method. However, it has been shown that GCL is vulnerable to adversarial attacks on both the graph structure and node attributes. Although empirical approaches have been proposed to enhance the robustness of GCL, the certifiable robustness of GCL is still remain unexplored. In this paper, we develop the first certifiably robust framework in GCL. Specifically, we first propose a unified criteria to evaluate and certify the robustness of GCL. We then introduce a novel technique, RES (Randomized Edgedrop Smoothing), to ensure certifiable robustness for any GCL model, and this certified robustness can be provably preserved in downstream tasks. Furthermore, an effective training method is proposed for robust GCL. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed method in providing effective certifiable robustness and enhancing the robustness of any GCL model. The source code of RES is available at https://github.com/ventr1c/RES-GCL.

{{</citation>}}


## eess.IV (3)



### (104/122) HartleyMHA: Self-Attention in Frequency Domain for Resolution-Robust and Parameter-Efficient 3D Image Segmentation (Ken C. L. Wong et al., 2023)

{{<citation>}}

Ken C. L. Wong, Hongzhi Wang, Tanveer Syeda-Mahmood. (2023)  
**HartleyMHA: Self-Attention in Frequency Domain for Resolution-Robust and Parameter-Efficient 3D Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.04466v1)  

---


**ABSTRACT**  
With the introduction of Transformers, different attention-based models have been proposed for image segmentation with promising results. Although self-attention allows capturing of long-range dependencies, it suffers from a quadratic complexity in the image size especially in 3D. To avoid the out-of-memory error during training, input size reduction is usually required for 3D segmentation, but the accuracy can be suboptimal when the trained models are applied on the original image size. To address this limitation, inspired by the Fourier neural operator (FNO), we introduce the HartleyMHA model which is robust to training image resolution with efficient self-attention. FNO is a deep learning framework for learning mappings between functions in partial differential equations, which has the appealing properties of zero-shot super-resolution and global receptive field. We modify the FNO by using the Hartley transform with shared parameters to reduce the model size by orders of magnitude, and this allows us to further apply self-attention in the frequency domain for more expressive high-order feature combination with improved efficiency. When tested on the BraTS'19 dataset, it achieved superior robustness to training image resolution than other tested models with less than 1% of their model parameters.

{{</citation>}}


### (105/122) BTDNet: a Multi-Modal Approach for Brain Tumor Radiogenomic Classification (Dimitrios Kollias et al., 2023)

{{<citation>}}

Dimitrios Kollias, Karanjot Vendal, Priyanka Gadhavi, Solomon Russom. (2023)  
**BTDNet: a Multi-Modal Approach for Brain Tumor Radiogenomic Classification**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03485v2)  

---


**ABSTRACT**  
Brain tumors pose significant health challenges worldwide, with glioblastoma being one of the most aggressive forms. Accurate determination of the O6-methylguanine-DNA methyltransferase (MGMT) promoter methylation status is crucial for personalized treatment strategies. However, traditional methods are labor-intensive and time-consuming. This paper proposes a novel multi-modal approach, BTDNet, leveraging multi-parametric MRI scans, including FLAIR, T1w, T1wCE, and T2 3D volumes, to predict MGMT promoter methylation status. BTDNet addresses two main challenges: the variable volume lengths (i.e., each volume consists of a different number of slices) and the volume-level annotations (i.e., the whole 3D volume is annotated and not the independent slices that it consists of). BTDNet consists of four components: i) the data augmentation one (that performs geometric transformations, convex combinations of data pairs and test-time data augmentation); ii) the 3D analysis one (that performs global analysis through a CNN-RNN); iii) the routing one (that contains a mask layer that handles variable input feature lengths), and iv) the modality fusion one (that effectively enhances data representation, reduces ambiguities and mitigates data scarcity). The proposed method outperforms by large margins the state-of-the-art methods in the RSNA-ASNR-MICCAI BraTS 2021 Challenge, offering a promising avenue for enhancing brain tumor diagnosis and treatment.

{{</citation>}}


### (106/122) Swin-Tempo: Temporal-Aware Lung Nodule Detection in CT Scans as Video Sequences Using Swin Transformer-Enhanced UNet (Hossein Jafari et al., 2023)

{{<citation>}}

Hossein Jafari, Karim Faez, Hamidreza Amindavar. (2023)  
**Swin-Tempo: Temporal-Aware Lung Nodule Detection in CT Scans as Video Sequences Using Swin Transformer-Enhanced UNet**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.03365v1)  

---


**ABSTRACT**  
Lung cancer is highly lethal, emphasizing the critical need for early detection. However, identifying lung nodules poses significant challenges for radiologists, who rely heavily on their expertise and experience for accurate diagnosis. To address this issue, computer-aided diagnosis systems based on machine learning techniques have emerged to assist doctors in identifying lung nodules from computed tomography (CT) scans. Unfortunately, existing networks in this domain often suffer from computational complexity, leading to high rates of false negatives and false positives, limiting their effectiveness. To address these challenges, we present an innovative model that harnesses the strengths of both convolutional neural networks and vision transformers. Inspired by object detection in videos, we treat each 3D CT image as a video, individual slices as frames, and lung nodules as objects, enabling a time-series application. The primary objective of our work is to overcome hardware limitations during model training, allowing for efficient processing of 2D data while utilizing inter-slice information for accurate identification based on 3D image context. We validated the proposed network by applying a 10-fold cross-validation technique to the publicly available Lung Nodule Analysis 2016 dataset. Our proposed architecture achieves an average sensitivity criterion of 97.84% and a competition performance metrics (CPM) of 96.0% with few parameters. Comparative analysis with state-of-the-art advancements in lung nodule identification demonstrates the significant accuracy achieved by our proposed model.

{{</citation>}}


## cs.FL (1)



### (107/122) Logical Languages Accepted by Transformer Encoders with Hard Attention (Pablo Barcelo et al., 2023)

{{<citation>}}

Pablo Barcelo, Alexander Kozachinskiy, Anthony Widjaja Lin, Vladimir Podolskii. (2023)  
**Logical Languages Accepted by Transformer Encoders with Hard Attention**  

---
Primary Category: cs.FL  
Categories: cs-FL, cs-LG, cs.FL  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.03817v1)  

---


**ABSTRACT**  
We contribute to the study of formal languages that can be recognized by transformer encoders. We focus on two self-attention mechanisms: (1) UHAT (Unique Hard Attention Transformers) and (2) AHAT (Average Hard Attention Transformers). UHAT encoders are known to recognize only languages inside the circuit complexity class ${\sf AC}^0$, i.e., accepted by a family of poly-sized and depth-bounded boolean circuits with unbounded fan-ins. On the other hand, AHAT encoders can recognize languages outside ${\sf AC}^0$), but their expressive power still lies within the bigger circuit complexity class ${\sf TC}^0$, i.e., ${\sf AC}^0$-circuits extended by majority gates. We first show a negative result that there is an ${\sf AC}^0$-language that cannot be recognized by an UHAT encoder. On the positive side, we show that UHAT encoders can recognize a rich fragment of ${\sf AC}^0$-languages, namely, all languages definable in first-order logic with arbitrary unary numerical predicates. This logic, includes, for example, all regular languages from ${\sf AC}^0$. We then show that AHAT encoders can recognize all languages of our logic even when we enrich it with counting terms. We apply these results to derive new results on the expressive power of UHAT and AHAT up to permutation of letters (a.k.a. Parikh images).

{{</citation>}}


## eess.SY (3)



### (108/122) Optimal Control of District Cooling Energy Plant with Reinforcement Learning and MPC (Zhong Guo et al., 2023)

{{<citation>}}

Zhong Guo, Aditya Chaudhari, Austin R. Coffman, Prabir Barooah. (2023)  
**Optimal Control of District Cooling Energy Plant with Reinforcement Learning and MPC**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: NLP, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.03814v1)  

---


**ABSTRACT**  
We consider the problem of optimal control of district cooling energy plants (DCEPs) consisting of multiple chillers, a cooling tower, and a thermal energy storage (TES), in the presence of time-varying electricity price. A straightforward application of model predictive control (MPC) requires solving a challenging mixed-integer nonlinear program (MINLP) because of the on/off of chillers and the complexity of the DCEP model. Reinforcement learning (RL) is an attractive alternative since its real-time control computation is much simpler. But designing an RL controller is challenging due to myriad design choices and computationally intensive training.   In this paper, we propose an RL controller and an MPC controller for minimizing the electricity cost of a DCEP, and compare them via simulations. The two controllers are designed to be comparable in terms of objective and information requirements. The RL controller uses a novel Q-learning algorithm that is based on least-squares policy iteration. We describe the design choices for the RL controller, including the choice of state space and basis functions, that are found to be effective. The proposed MPC controller does not need a mixed integer solver for implementation, but only a nonlinear program (NLP) solver. A rule-based baseline controller is also proposed to aid in comparison. Simulation results show that the proposed RL and MPC controllers achieve similar savings over the baseline controller, about 17%.

{{</citation>}}


### (109/122) Interpreting the Value of Flexibility in AC Security-Constrained Transmission Expansion Planning via a Cooperative Game Framework (Andrey Churkin et al., 2023)

{{<citation>}}

Andrey Churkin, Wangwei Kong, Mohammad Iman Alizadeh, Florin Capitanescu, Pierluigi Mancarella, Eduardo A. Martínez Ceseña. (2023)  
**Interpreting the Value of Flexibility in AC Security-Constrained Transmission Expansion Planning via a Cooperative Game Framework**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.03610v1)  

---


**ABSTRACT**  
Security-constrained transmission expansion planning (SCTEP) is an inherently complex problem that requires simultaneously solving multiple contingency states of the system (usually corresponding to N-1 security criterion). Existing studies focus on effectively finding optimal solutions; however, single optimal solutions are not sufficient to interpret the value of flexibility (e.g., from energy storage systems) and support system planners in well-informed decision making. In view of planning uncertainties, it is necessary to estimate the contributions of flexibility to various objectives and prioritise the most effective investments. In this regard, this work introduces a SCTEP tool that enables interpreting the value of flexibility in terms of contributions to avoided load curtailment and total expected system cost reduction. Inspired by cooperative game theory, the tool ranks the contributions of flexibility providers and compares them against traditional line reinforcements. This information can be used by system planners to prioritise investments with higher contributions and synergistic capabilities.

{{</citation>}}


### (110/122) Impact of Artificial Intelligence on Electrical and Electronics Engineering Productivity in the Construction Industry (Nwosu Obinnaya Chikezie Victor, 2023)

{{<citation>}}

Nwosu Obinnaya Chikezie Victor. (2023)  
**Impact of Artificial Intelligence on Electrical and Electronics Engineering Productivity in the Construction Industry**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SP, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03591v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) can revolutionize the development industry, primarily electrical and electronics engineering. By automating recurring duties, AI can grow productivity and efficiency in creating. For instance, AI can research constructing designs, discover capability troubles, and generate answers, reducing the effort and time required for manual analysis. AI also can be used to optimize electricity consumption in buildings, which is a critical difficulty in the construction enterprise. Via machines gaining knowledge of algorithms to investigate electricity usage patterns, AI can discover areas wherein power may be stored and offer guidelines for enhancements. This can result in significant value financial savings and reduced carbon emissions. Moreover, AI may be used to improve the protection of creation websites. By studying statistics from sensors and cameras, AI can locate capacity dangers and alert workers to take suitable action. This could help save you from injuries and accidents on production sites, lowering the chance for workers and enhancing overall safety in the enterprise. The impact of AI on electric and electronics engineering productivity inside the creation industry is enormous. AI can transform how we layout, build, and function buildings by automating ordinary duties, optimising electricity intake, and enhancing safety. However, ensuring that AI is used ethically and responsibly and that the advantages are shared fairly throughout the enterprise is essential.

{{</citation>}}


## cs.AI (4)



### (111/122) Artificial Intelligence Index Report 2023 (Nestor Maslej et al., 2023)

{{<citation>}}

Nestor Maslej, Loredana Fattorini, Erik Brynjolfsson, John Etchemendy, Katrina Ligett, Terah Lyons, James Manyika, Helen Ngo, Juan Carlos Niebles, Vanessa Parli, Yoav Shoham, Russell Wald, Jack Clark, Raymond Perrault. (2023)  
**Artificial Intelligence Index Report 2023**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03715v1)  

---


**ABSTRACT**  
Welcome to the sixth edition of the AI Index Report. This year, the report introduces more original data than any previous edition, including a new chapter on AI public opinion, a more thorough technical performance chapter, original analysis about large language and multimodal models, detailed trends in global AI legislation records, a study of the environmental impact of AI systems, and more. The AI Index Report tracks, collates, distills, and visualizes data related to artificial intelligence. Our mission is to provide unbiased, rigorously vetted, broadly sourced data in order for policymakers, researchers, executives, journalists, and the general public to develop a more thorough and nuanced understanding of the complex field of AI. The report aims to be the world's most credible and authoritative source for data and insights about AI.

{{</citation>}}


### (112/122) Automating Human Tutor-Style Programming Feedback: Leveraging GPT-4 Tutor Model for Hint Generation and GPT-3.5 Student Model for Hint Validation (Tung Phung et al., 2023)

{{<citation>}}

Tung Phung, Victor-Alexandru Pădurean, Anjali Singh, Christopher Brooks, José Cambronero, Sumit Gulwani, Adish Singla, Gustavo Soares. (2023)  
**Automating Human Tutor-Style Programming Feedback: Leveraging GPT-4 Tutor Model for Hint Generation and GPT-3.5 Student Model for Hint Validation**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, GPT, GPT-3.5, GPT-4, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.03780v1)  

---


**ABSTRACT**  
Generative AI and large language models hold great promise in enhancing programming education by automatically generating individualized feedback for students. We investigate the role of generative AI models in providing human tutor-style programming hints to help students resolve errors in their buggy programs. Recent works have benchmarked state-of-the-art models for various feedback generation scenarios; however, their overall quality is still inferior to human tutors and not yet ready for real-world deployment. In this paper, we seek to push the limits of generative AI models toward providing high-quality programming hints and develop a novel technique, GPT4Hints-GPT3.5Val. As a first step, our technique leverages GPT-4 as a ``tutor'' model to generate hints -- it boosts the generative quality by using symbolic information of failing test cases and fixes in prompts. As a next step, our technique leverages GPT-3.5, a weaker model, as a ``student'' model to further validate the hint quality -- it performs an automatic quality validation by simulating the potential utility of providing this feedback. We show the efficacy of our technique via extensive evaluation using three real-world datasets of Python programs covering a variety of concepts ranging from basic algorithms to regular expressions and data analysis using pandas library.

{{</citation>}}


### (113/122) Balancing Autonomy and Alignment: A Multi-Dimensional Taxonomy for Autonomous LLM-powered Multi-Agent Architectures (Thorsten Händler, 2023)

{{<citation>}}

Thorsten Händler. (2023)  
**Balancing Autonomy and Alignment: A Multi-Dimensional Taxonomy for Autonomous LLM-powered Multi-Agent Architectures**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MA, cs-SE, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03659v1)  

---


**ABSTRACT**  
Large language models (LLMs) have revolutionized the field of artificial intelligence, endowing it with sophisticated language understanding and generation capabilities. However, when faced with more complex and interconnected tasks that demand a profound and iterative thought process, LLMs reveal their inherent limitations. Autonomous LLM-powered multi-agent systems represent a strategic response to these challenges. Such systems strive for autonomously tackling user-prompted goals by decomposing them into manageable tasks and orchestrating their execution and result synthesis through a collective of specialized intelligent agents. Equipped with LLM-powered reasoning capabilities, these agents harness the cognitive synergy of collaborating with their peers, enhanced by leveraging contextual resources such as tools and datasets. While these architectures hold promising potential in amplifying AI capabilities, striking the right balance between different levels of autonomy and alignment remains the crucial challenge for their effective operation. This paper proposes a comprehensive multi-dimensional taxonomy, engineered to analyze how autonomous LLM-powered multi-agent systems balance the dynamic interplay between autonomy and alignment across various aspects inherent to architectural viewpoints such as goal-driven task management, agent composition, multi-agent collaboration, and context interaction. It also includes a domain-ontology model specifying fundamental architectural concepts. Our taxonomy aims to empower researchers, engineers, and AI practitioners to systematically analyze the architectural dynamics and balancing strategies employed by these increasingly prevalent AI systems. The exploratory taxonomic classification of selected representative LLM-powered multi-agent systems illustrates its practical utility and reveals potential for future research and development.

{{</citation>}}


### (114/122) Learning Concept-Based Visual Causal Transition and Symbolic Reasoning for Visual Planning (Yilue Qian et al., 2023)

{{<citation>}}

Yilue Qian, Peiyu Yu, Ying Nian Wu, Wei Wang, Lifeng Fan. (2023)  
**Learning Concept-Based Visual Causal Transition and Symbolic Reasoning for Visual Planning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-LG, cs.AI  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.03325v1)  

---


**ABSTRACT**  
Visual planning simulates how humans make decisions to achieve desired goals in the form of searching for visual causal transitions between an initial visual state and a final visual goal state. It has become increasingly important in egocentric vision with its advantages in guiding agents to perform daily tasks in complex environments. In this paper, we propose an interpretable and generalizable visual planning framework consisting of i) a novel Substitution-based Concept Learner (SCL) that abstracts visual inputs into disentangled concept representations, ii) symbol abstraction and reasoning that performs task planning via the self-learned symbols, and iii) a Visual Causal Transition model (ViCT) that grounds visual causal transitions to semantically similar real-world actions. Given an initial state, we perform goal-conditioned visual planning with a symbolic reasoning method fueled by the learned representations and causal transitions to reach the goal state. To verify the effectiveness of the proposed model, we collect a large-scale visual planning dataset based on AI2-THOR, dubbed as CCTP. Extensive experiments on this challenging dataset demonstrate the superior performance of our method in visual task planning. Empirically, we show that our framework can generalize to unseen task trajectories and unseen object categories.

{{</citation>}}


## cs.HC (2)



### (115/122) DirectGPT: A Direct Manipulation Interface to Interact with Large Language Models (Damien Masson et al., 2023)

{{<citation>}}

Damien Masson, Sylvain Malacria, Géry Casiez, Daniel Vogel. (2023)  
**DirectGPT: A Direct Manipulation Interface to Interact with Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.03691v1)  

---


**ABSTRACT**  
We characterize and demonstrate how the principles of direct manipulation can improve interaction with large language models. This includes: continuous representation of generated objects of interest; reuse of prompt syntax in a toolbar of commands; manipulable outputs to compose or control the effect of prompts; and undo mechanisms. This idea is exemplified in DirectGPT, a user interface layer on top of ChatGPT that works by transforming direct manipulation actions to engineered prompts. A study shows participants were 50% faster and relied on 50% fewer and 72% shorter prompts to edit text, code, and vector images compared to baseline ChatGPT. Our work contributes a validated approach to integrate LLMs into traditional software using direct manipulation.

{{</citation>}}


### (116/122) Unpacking Human-AI Interaction in Safety-Critical Industries: A Systematic Literature Review (Tita A. Bach et al., 2023)

{{<citation>}}

Tita A. Bach, Jenny K. Kristiansen, Aleksandar Babic, Alon Jacovi. (2023)  
**Unpacking Human-AI Interaction in Safety-Critical Industries: A Systematic Literature Review**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03392v1)  

---


**ABSTRACT**  
Ensuring quality human-AI interaction (HAII) in safety-critical industries is essential. Failure to do so can lead to catastrophic and deadly consequences. Despite this urgency, what little research there is on HAII is fragmented and inconsistent. We present here a survey of that literature and recommendations for research best practices that will improve the field. We divided our investigation into the following research areas: (1) terms used to describe HAII, (2) primary roles of AI-enabled systems, (3) factors that influence HAII, and (4) how HAII is measured. Additionally, we described the capabilities and maturity of the AI-enabled systems used in safety-critical industries discussed in these articles. We found that no single term is used across the literature to describe HAII and some terms have multiple meanings. According to our literature, five factors influence HAII: user characteristics and background (e.g., user personality, perceptions), AI interface and features (e.g., interactive UI design), AI output (e.g., accuracy, actionable recommendations), explainability and interpretability (e.g., level of detail, user understanding), and usage of AI (e.g., heterogeneity of environments and user needs). HAII is most commonly measured with user-related subjective metrics (e.g., user perception, trust, and attitudes), and AI-assisted decision-making is the most common primary role of AI-enabled systems. Based on this review, we conclude that there are substantial research gaps in HAII. Researchers and developers need to codify HAII terminology, involve users throughout the AI lifecycle (especially during development), and tailor HAII in safety-critical industries to the users and environments.

{{</citation>}}


## cs.SE (2)



### (117/122) PeaTMOSS: Mining Pre-Trained Models in Open-Source Software (Wenxin Jiang et al., 2023)

{{<citation>}}

Wenxin Jiang, Jason Jones, Jerin Yasmin, Nicholas Synovic, Rajeev Sashti, Sophie Chen, George K. Thiruvathukal, Yuan Tian, James C. Davis. (2023)  
**PeaTMOSS: Mining Pre-Trained Models in Open-Source Software**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2310.03620v1)  

---


**ABSTRACT**  
Developing and training deep learning models is expensive, so software engineers have begun to reuse pre-trained deep learning models (PTMs) and fine-tune them for downstream tasks. Despite the wide-spread use of PTMs, we know little about the corresponding software engineering behaviors and challenges.   To enable the study of software engineering with PTMs, we present the PeaTMOSS dataset: Pre-Trained Models in Open-Source Software. PeaTMOSS has three parts: a snapshot of (1) 281,638 PTMs, (2) 27,270 open-source software repositories that use PTMs, and (3) a mapping between PTMs and the projects that use them. We challenge PeaTMOSS miners to discover software engineering practices around PTMs. A demo and link to the full dataset are available at: https://github.com/PurdueDualityLab/PeaTMOSS-Demos.

{{</citation>}}


### (118/122) Large Language Models for Software Engineering: Survey and Open Problems (Angela Fan et al., 2023)

{{<citation>}}

Angela Fan, Beliz Gokkaya, Mark Harman, Mitya Lyubarskiy, Shubho Sengupta, Shin Yoo, Jie M. Zhang. (2023)  
**Large Language Models for Software Engineering: Survey and Open Problems**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.03533v2)  

---


**ABSTRACT**  
This paper provides a survey of the emerging area of Large Language Models (LLMs) for Software Engineering (SE). It also sets out open research challenges for the application of LLMs to technical problems faced by software engineers. LLMs' emergent properties bring novelty and creativity with applications right across the spectrum of Software Engineering activities including coding, design, requirements, repair, refactoring, performance improvement, documentation and analytics. However, these very same emergent properties also pose significant technical challenges; we need techniques that can reliably weed out incorrect solutions, such as hallucinations. Our survey reveals the pivotal role that hybrid techniques (traditional SE plus LLMs) have to play in the development and deployment of reliable, efficient and effective LLM-based SE.

{{</citation>}}


## cs.CY (1)



### (119/122) User Attitudes to Content Moderation in Web Search (Aleksandra Urman et al., 2023)

{{<citation>}}

Aleksandra Urman, Aniko Hannak, Mykola Makhortykh. (2023)  
**User Attitudes to Content Moderation in Web Search**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-HC, cs.CY  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.03458v1)  

---


**ABSTRACT**  
Internet users highly rely on and trust web search engines, such as Google, to find relevant information online. However, scholars have documented numerous biases and inaccuracies in search outputs. To improve the quality of search results, search engines employ various content moderation practices such as interface elements informing users about potentially dangerous websites and algorithmic mechanisms for downgrading or removing low-quality search results. While the reliance of the public on web search engines and their use of moderation practices is well-established, user attitudes towards these practices have not yet been explored in detail. To address this gap, we first conducted an overview of content moderation practices used by search engines, and then surveyed a representative sample of the US adult population (N=398) to examine the levels of support for different moderation practices applied to potentially misleading and/or potentially offensive content in web search. We also analyzed the relationship between user characteristics and their support for specific moderation practices. We find that the most supported practice is informing users about potentially misleading or offensive content, and the least supported one is the complete removal of search results. More conservative users and users with lower levels of trust in web search results are more likely to be against content moderation in web search.

{{</citation>}}


## q-bio.BM (1)



### (120/122) InstructProtein: Aligning Human and Protein Language via Knowledge Instruction (Zeyuan Wang et al., 2023)

{{<citation>}}

Zeyuan Wang, Qiang Zhang, Keyan Ding, Ming Qin, Xiang Zhuang, Xiaotong Li, Huajun Chen. (2023)  
**InstructProtein: Aligning Human and Protein Language via Knowledge Instruction**  

---
Primary Category: q-bio.BM  
Categories: cs-CL, q-bio-BM, q-bio.BM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.03269v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have revolutionized the field of natural language processing, but they fall short in comprehending biological sequences such as proteins. To address this challenge, we propose InstructProtein, an innovative LLM that possesses bidirectional generation capabilities in both human and protein languages: (i) taking a protein sequence as input to predict its textual function description and (ii) using natural language to prompt protein sequence generation. To achieve this, we first pre-train an LLM on both protein and natural language corpora, enabling it to comprehend individual languages. Then supervised instruction tuning is employed to facilitate the alignment of these two distinct languages. Herein, we introduce a knowledge graph-based instruction generation framework to construct a high-quality instruction dataset, addressing annotation imbalance and instruction deficits in existing protein-text corpus. In particular, the instructions inherit the structural relations between proteins and function annotations in knowledge graphs, which empowers our model to engage in the causal modeling of protein functions, akin to the chain-of-thought processes in natural languages. Extensive experiments on bidirectional protein-text generation tasks show that InstructProtein outperforms state-of-the-art LLMs by large margins. Moreover, InstructProtein serves as a pioneering step towards text-based protein function prediction and sequence design, effectively bridging the gap between protein and human language understanding.

{{</citation>}}


## stat.ML (1)



### (121/122) Sparse Deep Learning for Time Series Data: Theory and Applications (Mingxuan Zhang et al., 2023)

{{<citation>}}

Mingxuan Zhang, Yan Sun, Faming Liang. (2023)  
**Sparse Deep Learning for Time Series Data: Theory and Applications**  

---
Primary Category: stat.ML  
Categories: cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.03243v1)  

---


**ABSTRACT**  
Sparse deep learning has become a popular technique for improving the performance of deep neural networks in areas such as uncertainty quantification, variable selection, and large-scale network compression. However, most existing research has focused on problems where the observations are independent and identically distributed (i.i.d.), and there has been little work on the problems where the observations are dependent, such as time series data and sequential data in natural language processing. This paper aims to address this gap by studying the theory for sparse deep learning with dependent data. We show that sparse recurrent neural networks (RNNs) can be consistently estimated, and their predictions are asymptotically normally distributed under appropriate assumptions, enabling the prediction uncertainty to be correctly quantified. Our numerical results show that sparse deep learning outperforms state-of-the-art methods, such as conformal predictions, in prediction uncertainty quantification for time series data. Furthermore, our results indicate that the proposed method can consistently identify the autoregressive order for time series data and outperform existing methods in large-scale model compression. Our proposed method has important practical implications in fields such as finance, healthcare, and energy, where both accurate point estimates and prediction uncertainty quantification are of concern.

{{</citation>}}


## math.OC (1)



### (122/122) Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization (Quanqi Hu et al., 2023)

{{<citation>}}

Quanqi Hu, Dixian Zhu, Tianbao Yang. (2023)  
**Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization**  

---
Primary Category: math.OC  
Categories: cs-AI, cs-LG, math-OC, math.OC, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.03234v1)  

---


**ABSTRACT**  
This paper investigates new families of compositional optimization problems, called $\underline{\bf n}$on-$\underline{\bf s}$mooth $\underline{\bf w}$eakly-$\underline{\bf c}$onvex $\underline{\bf f}$inite-sum $\underline{\bf c}$oupled $\underline{\bf c}$ompositional $\underline{\bf o}$ptimization (NSWC FCCO). There has been a growing interest in FCCO due to its wide-ranging applications in machine learning and AI, as well as its ability to address the shortcomings of stochastic algorithms based on empirical risk minimization. However, current research on FCCO presumes that both the inner and outer functions are smooth, limiting their potential to tackle a more diverse set of problems. Our research expands on this area by examining non-smooth weakly-convex FCCO, where the outer function is weakly convex and non-decreasing, and the inner function is weakly-convex. We analyze a single-loop algorithm and establish its complexity for finding an $\epsilon$-stationary point of the Moreau envelop of the objective function. Additionally, we also extend the algorithm to solving novel non-smooth weakly-convex tri-level finite-sum coupled compositional optimization problems, which feature a nested arrangement of three functions. Lastly, we explore the applications of our algorithms in deep learning for two-way partial AUC maximization and multi-instance two-way partial AUC maximization, using empirical studies to showcase the effectiveness of the proposed algorithms.

{{</citation>}}
