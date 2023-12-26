---
draft: false
title: "arXiv @ 2023.12.24"
date: 2023-12-24
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.24"
    identifier: arxiv_20231224
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (14)](#cslg-14)
- [cs.AI (9)](#csai-9)
- [cs.CL (13)](#cscl-13)
- [cs.CV (14)](#cscv-14)
- [cs.SE (4)](#csse-4)
- [physics.bio-ph (1)](#physicsbio-ph-1)
- [cs.LO (2)](#cslo-2)
- [cs.CY (2)](#cscy-2)
- [eess.IV (1)](#eessiv-1)
- [cs.MM (1)](#csmm-1)
- [cs.DC (2)](#csdc-2)
- [cs.CR (4)](#cscr-4)
- [eess.AS (1)](#eessas-1)
- [econ.GN (1)](#econgn-1)
- [cs.NE (1)](#csne-1)
- [eess.SY (1)](#eesssy-1)
- [cs.HC (1)](#cshc-1)
- [cs.RO (4)](#csro-4)
- [cs.NI (1)](#csni-1)
- [cs.IR (1)](#csir-1)
- [cs.SD (1)](#cssd-1)
- [cs.PL (1)](#cspl-1)

## cs.LG (14)



### (1/80) A Survey of Reinforcement Learning from Human Feedback (Timo Kaufmann et al., 2023)

{{<citation>}}

Timo Kaufmann, Paul Weng, Viktor Bengs, Eyke Hüllermeier. (2023)  
**A Survey of Reinforcement Learning from Human Feedback**  

---
Primary Category: cs.LG  
Categories: I-2-6, cs-LG, cs.LG  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14925v1)  

---


**ABSTRACT**  
Reinforcement learning from human feedback (RLHF) is a variant of reinforcement learning (RL) that learns from human feedback instead of relying on an engineered reward function. Building on prior work on the related setting of preference-based reinforcement learning (PbRL), it stands at the intersection of artificial intelligence and human-computer interaction. This positioning offers a promising avenue to enhance the performance and adaptability of intelligent systems while also improving the alignment of their objectives with human values. The training of Large Language Models (LLMs) has impressively demonstrated this potential in recent years, where RLHF played a decisive role in targeting the model's capabilities toward human objectives. This article provides a comprehensive overview of the fundamentals of RLHF, exploring the intricate dynamics between machine agents and human input. While recent focus has been on RLHF for LLMs, our survey adopts a broader perspective, examining the diverse applications and wide-ranging impact of the technique. We delve into the core principles that underpin RLHF, shedding light on the symbiotic relationship between algorithms and human feedback, and discuss the main research trends in the field. By synthesizing the current landscape of RLHF research, this article aims to provide researchers as well as practitioners with a comprehensive understanding of this rapidly growing field of research.

{{</citation>}}


### (2/80) Spatiotemporal-Linear: Towards Universal Multivariate Time Series Forecasting (Aiyinsi Zuo et al., 2023)

{{<citation>}}

Aiyinsi Zuo, Haixi Zhang, Zirui Li, Ce Zheng. (2023)  
**Spatiotemporal-Linear: Towards Universal Multivariate Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2312.14869v1)  

---


**ABSTRACT**  
Within the field of complicated multivariate time series forecasting (TSF), popular techniques frequently rely on intricate deep learning architectures, ranging from transformer-based designs to recurrent neural networks. However, recent findings suggest that simple Linear models can surpass sophisticated constructs on diverse datasets. These models directly map observation to multiple future time steps, thereby minimizing error accumulation in iterative multi-step prediction. Yet, these models fail to incorporate spatial and temporal information within the data, which is critical for capturing patterns and dependencies that drive insightful predictions. This oversight often leads to performance bottlenecks, especially under specific sequence lengths and dataset conditions, preventing their universal application. In response, we introduce the SpatioTemporal-Linear (STL) framework. STL seamlessly integrates time-embedded and spatially-informed bypasses to augment the Linear-based architecture. These extra routes offer a more robust and refined regression to the data, particularly when the amount of observation is limited and the capacity of simple linear layers to capture dependencies declines. Empirical evidence highlights STL's prowess, outpacing both Linear and Transformer benchmarks across varied observation and prediction durations and datasets. Such robustness accentuates its suitability across a spectrum of applications, including but not limited to, traffic trajectory and rare disease progression forecasting. Through this discourse, we not only validate the STL's distinctive capacities to become a more general paradigm in multivariate time-series prediction using deep-learning techniques but also stress the need to tackle data-scarce prediction scenarios for universal application. Code will be made available.

{{</citation>}}


### (3/80) Understanding the Regularity of Self-Attention with Optimal Transport (Valérie Castin et al., 2023)

{{<citation>}}

Valérie Castin, Pierre Ablin, Gabriel Peyré. (2023)  
**Understanding the Regularity of Self-Attention with Optimal Transport**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.14820v1)  

---


**ABSTRACT**  
Transformers and their multi-head attention mechanism have completely changed the machine learning landscape in just a few years, by outperforming state-of-art models in a wide range of domains. Still, little is known about their robustness from a theoretical perspective. We tackle this problem by studying the local Lipschitz constant of self-attention, that provides an attack-agnostic way of measuring the robustness of a neural network. We adopt a measure-theoretic framework, by viewing inputs as probability measures equipped with the Wasserstein distance. This allows us to generalize attention to inputs of infinite length, and to derive an upper bound and a lower bound on the Lipschitz constant of self-attention on compact sets. The lower bound significantly improves prior results, and grows more than exponentially with the radius of the compact set, which rules out the possibility of obtaining robustness guarantees without any additional constraint on the input space. Our results also point out that measures with a high local Lipschitz constant are typically made of a few diracs, with a very unbalanced distribution of mass. Finally, we analyze the stability of self-attention under perturbations that change the number of tokens, which appears to be a natural question in the measure-theoretic framework. In particular, we show that for some inputs, attacks that duplicate tokens before perturbing them are more efficient than attacks that simply move tokens. We call this phenomenon mass splitting.

{{</citation>}}


### (4/80) Progressing from Anomaly Detection to Automated Log Labeling and Pioneering Root Cause Analysis (Thorsten Wittkopp et al., 2023)

{{<citation>}}

Thorsten Wittkopp, Alexander Acker, Odej Kao. (2023)  
**Progressing from Anomaly Detection to Automated Log Labeling and Pioneering Root Cause Analysis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SE, cs.LG  
Keywords: AI, Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.14748v1)  

---


**ABSTRACT**  
The realm of AIOps is transforming IT landscapes with the power of AI and ML. Despite the challenge of limited labeled data, supervised models show promise, emphasizing the importance of leveraging labels for training, especially in deep learning contexts. This study enhances the field by introducing a taxonomy for log anomalies and exploring automated data labeling to mitigate labeling challenges. It goes further by investigating the potential of diverse anomaly detection techniques and their alignment with specific anomaly types. However, the exploration doesn't stop at anomaly detection. The study envisions a future where root cause analysis follows anomaly detection, unraveling the underlying triggers of anomalies. This uncharted territory holds immense potential for revolutionizing IT systems management. In essence, this paper enriches our understanding of anomaly detection, and automated labeling, and sets the stage for transformative root cause analysis. Together, these advances promise more resilient IT systems, elevating operational efficiency and user satisfaction in an ever-evolving technological landscape.

{{</citation>}}


### (5/80) Deep Non-Parametric Time Series Forecaster (Syama Sundar Rangapuram et al., 2023)

{{<citation>}}

Syama Sundar Rangapuram, Jan Gasthaus, Lorenzo Stella, Valentin Flunkert, David Salinas, Yuyang Wang, Tim Januschowski. (2023)  
**Deep Non-Parametric Time Series Forecaster**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.14657v1)  

---


**ABSTRACT**  
This paper presents non-parametric baseline models for time series forecasting. Unlike classical forecasting models, the proposed approach does not assume any parametric form for the predictive distribution and instead generates predictions by sampling from the empirical distribution according to a tunable strategy. By virtue of this, the model is always able to produce reasonable forecasts (i.e., predictions within the observed data range) without fail unlike classical models that suffer from numerical stability on some data distributions. Moreover, we develop a global version of the proposed method that automatically learns the sampling strategy by exploiting the information across multiple related time series. The empirical evaluation shows that the proposed methods have reasonable and consistent performance across all datasets, proving them to be strong baselines to be considered in one's forecasting toolbox.

{{</citation>}}


### (6/80) Towards more sustainable enterprise data and application management with cross silo Federated Learning and Analytics (Hongliu Cao, 2023)

{{<citation>}}

Hongliu Cao. (2023)  
**Towards more sustainable enterprise data and application management with cross silo Federated Learning and Analytics**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14628v1)  

---


**ABSTRACT**  
To comply with new legal requirements and policies committed to privacy protection, more and more companies start to deploy cross-silo Federated Learning at global scale, where several clients/silos collaboratively train a global model under the coordination of a central server. Instead of data sharing and transmission, clients train models using their private local data and exchange model updates. However, there is little understanding of the carbon emission impact of cross silo Federated Learning due to the lack of related works. In this study, we first analyze the sustainability aspect of cross-silo Federated Learning, across the AI product life cycle instead of focusing only on the model training, with the comparison to the centralized method. A more holistic quantitative cost and CO2 emission estimation method for real world cross-silo Federated Learning setting is proposed. Secondly, we propose a novel data and application management system using cross silo Federated Learning and analytics to make IT companies more sustainable and cost effective.

{{</citation>}}


### (7/80) ADA-GAD: Anomaly-Denoised Autoencoders for Graph Anomaly Detection (Junwei He et al., 2023)

{{<citation>}}

Junwei He, Qianqian Xu, Yangbangyan Jiang, Zitai Wang, Qingming Huang. (2023)  
**ADA-GAD: Anomaly-Denoised Autoencoders for Graph Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2312.14535v1)  

---


**ABSTRACT**  
Graph anomaly detection is crucial for identifying nodes that deviate from regular behavior within graphs, benefiting various domains such as fraud detection and social network. Although existing reconstruction-based methods have achieved considerable success, they may face the \textit{Anomaly Overfitting} and \textit{Homophily Trap} problems caused by the abnormal patterns in the graph, breaking the assumption that normal nodes are often better reconstructed than abnormal ones. Our observations indicate that models trained on graphs with fewer anomalies exhibit higher detection performance. Based on this insight, we introduce a novel two-stage framework called Anomaly-Denoised Autoencoders for Graph Anomaly Detection (ADA-GAD). In the first stage, we design a learning-free anomaly-denoised augmentation method to generate graphs with reduced anomaly levels. We pretrain graph autoencoders on these augmented graphs at multiple levels, which enables the graph autoencoders to capture normal patterns. In the next stage, the decoders are retrained for detection on the original graph, benefiting from the multi-level representations learned in the previous stage. Meanwhile, we propose the node anomaly distribution regularization to further alleviate \textit{Anomaly Overfitting}. We validate the effectiveness of our approach through extensive experiments on both synthetic and real-world datasets.

{{</citation>}}


### (8/80) An effective and efficient green federated learning method for one-layer neural networks (Oscar Fontenla-Romero et al., 2023)

{{<citation>}}

Oscar Fontenla-Romero, Bertha Guijarro-Berdiñas, Elena Hernández-Pereira, Beatriz Pérez-Sánchez. (2023)  
**An effective and efficient green federated learning method for one-layer neural networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14528v1)  

---


**ABSTRACT**  
Nowadays, machine learning algorithms continue to grow in complexity and require a substantial amount of computational resources and energy. For these reasons, there is a growing awareness of the development of new green algorithms and distributed AI can contribute to this. Federated learning (FL) is one of the most active research lines in machine learning, as it allows the training of collaborative models in a distributed way, an interesting option in many real-world environments, such as the Internet of Things, allowing the use of these models in edge computing devices. In this work, we present a FL method, based on a neural network without hidden layers, capable of generating a global collaborative model in a single training round, unlike traditional FL methods that require multiple rounds for convergence. This allows obtaining an effective and efficient model that simplifies the management of the training process. Moreover, this method preserve data privacy by design, a crucial aspect in current data protection regulations. We conducted experiments with large datasets and a large number of federated clients. Despite being based on a network model without hidden layers, it maintains in all cases competitive accuracy results compared to more complex state-of-the-art machine learning models. Furthermore, we show that the method performs equally well in both identically and non-identically distributed scenarios. Finally, it is an environmentally friendly algorithm as it allows significant energy savings during the training process compared to its centralized counterpart.

{{</citation>}}


### (9/80) Safe Reinforcement Learning with Instantaneous Constraints: The Role of Aggressive Exploration (Honghao Wei et al., 2023)

{{<citation>}}

Honghao Wei, Xin Liu, Lei Ying. (2023)  
**Safe Reinforcement Learning with Instantaneous Constraints: The Role of Aggressive Exploration**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14470v1)  

---


**ABSTRACT**  
This paper studies safe Reinforcement Learning (safe RL) with linear function approximation and under hard instantaneous constraints where unsafe actions must be avoided at each step. Existing studies have considered safe RL with hard instantaneous constraints, but their approaches rely on several key assumptions: $(i)$ the RL agent knows a safe action set for {\it every} state or knows a {\it safe graph} in which all the state-action-state triples are safe, and $(ii)$ the constraint/cost functions are {\it linear}. In this paper, we consider safe RL with instantaneous hard constraints without assumption $(i)$ and generalize $(ii)$ to Reproducing Kernel Hilbert Space (RKHS). Our proposed algorithm, LSVI-AE, achieves $\tilde{\cO}(\sqrt{d^3H^4K})$ regret and $\tilde{\cO}(H \sqrt{dK})$ hard constraint violation when the cost function is linear and $\cO(H\gamma_K \sqrt{K})$ hard constraint violation when the cost function belongs to RKHS. Here $K$ is the learning horizon, $H$ is the length of each episode, and $\gamma_K$ is the information gain w.r.t the kernel used to approximate cost functions. Our results achieve the optimal dependency on the learning horizon $K$, matching the lower bound we provide in this paper and demonstrating the efficiency of LSVI-AE. Notably, the design of our approach encourages aggressive policy exploration, providing a unique perspective on safe RL with general cost functions and no prior knowledge of safe actions, which may be of independent interest.

{{</citation>}}


### (10/80) Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks (Haz Sameen Shahgir et al., 2023)

{{<citation>}}

Haz Sameen Shahgir, Xianghao Kong, Greg Ver Steeg, Yue Dong. (2023)  
**Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Adversarial Attack, Bias  
[Paper Link](http://arxiv.org/abs/2312.14440v1)  

---


**ABSTRACT**  
The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research into this, the reasons for their effectiveness are underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASRs). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace "human" with "robot" in the prompt "a human dancing in the rain." with an adversarial suffix but is significantly harder in reverse. We further propose probing metrics to establish indicative signals from the model's beliefs to the adversarial ASR. We identify conditions resulting in a 60% success probability for adversarial attacks and others where this likelihood drops below 5%.

{{</citation>}}


### (11/80) PC-Conv: Unifying Homophily and Heterophily with Two-fold Filtering (Bingheng Li et al., 2023)

{{<citation>}}

Bingheng Li, Erlin Pan, Zhao Kang. (2023)  
**PC-Conv: Unifying Homophily and Heterophily with Two-fold Filtering**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.14438v1)  

---


**ABSTRACT**  
Recently, many carefully crafted graph representation learning methods have achieved impressive performance on either strong heterophilic or homophilic graphs, but not both. Therefore, they are incapable of generalizing well across real-world graphs with different levels of homophily. This is attributed to their neglect of homophily in heterophilic graphs, and vice versa. In this paper, we propose a two-fold filtering mechanism to extract homophily in heterophilic graphs and vice versa. In particular, we extend the graph heat equation to perform heterophilic aggregation of global information from a long distance. The resultant filter can be exactly approximated by the Possion-Charlier (PC) polynomials. To further exploit information at multiple orders, we introduce a powerful graph convolution PC-Conv and its instantiation PCNet for the node classification task. Compared with state-of-the-art GNNs, PCNet shows competitive performance on well-known homophilic and heterophilic graphs. Our implementation is available at https://github.com/uestclbh/PC-Conv.

{{</citation>}}


### (12/80) Generative Pretraining at Scale: Transformer-Based Encoding of Transactional Behavior for Fraud Detection (Ze Yu Zhao et al., 2023)

{{<citation>}}

Ze Yu Zhao, Zheng Zhu, Guilin Li, Wenhan Wang, Bo Wang. (2023)  
**Generative Pretraining at Scale: Transformer-Based Encoding of Transactional Behavior for Fraud Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Fraud Detection, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2312.14406v1)  

---


**ABSTRACT**  
In this work, we introduce an innovative autoregressive model leveraging Generative Pretrained Transformer (GPT) architectures, tailored for fraud detection in payment systems. Our approach innovatively confronts token explosion and reconstructs behavioral sequences, providing a nuanced understanding of transactional behavior through temporal and contextual analysis. Utilizing unsupervised pretraining, our model excels in feature representation without the need for labeled data. Additionally, we integrate a differential convolutional approach to enhance anomaly detection, bolstering the security and efficacy of one of the largest online payment merchants in China. The scalability and adaptability of our model promise broad applicability in various transactional contexts.

{{</citation>}}


### (13/80) Graph Attention-Based Symmetry Constraint Extraction for Analog Circuits (Qi Xu et al., 2023)

{{<citation>}}

Qi Xu, Lijie Wang, Jing Wang, Song Chen, Lin Cheng, Yi Kang. (2023)  
**Graph Attention-Based Symmetry Constraint Extraction for Analog Circuits**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.14405v1)  

---


**ABSTRACT**  
In recent years, analog circuits have received extensive attention and are widely used in many emerging applications. The high demand for analog circuits necessitates shorter circuit design cycles. To achieve the desired performance and specifications, various geometrical symmetry constraints must be carefully considered during the analog layout process. However, the manual labeling of these constraints by experienced analog engineers is a laborious and time-consuming process. To handle the costly runtime issue, we propose a graph-based learning framework to automatically extract symmetric constraints in analog circuit layout. The proposed framework leverages the connection characteristics of circuits and the devices'information to learn the general rules of symmetric constraints, which effectively facilitates the extraction of device-level constraints on circuit netlists. The experimental results demonstrate that compared to state-of-the-art symmetric constraint detection approaches, our framework achieves higher accuracy and lower false positive rate.

{{</citation>}}


### (14/80) Multimodal Attention Merging for Improved Speech Recognition and Audio Event Classification (Anirudh S. Sundar et al., 2023)

{{<citation>}}

Anirudh S. Sundar, Chao-Han Huck Yang, David M. Chan, Shalini Ghosh, Venkatesh Ravichandran, Phani Sankar Nidadavolu. (2023)  
**Multimodal Attention Merging for Improved Speech Recognition and Audio Event Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SD, cs.LG, eess-AS  
Keywords: Attention, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.14378v1)  

---


**ABSTRACT**  
Training large foundation models using self-supervised objectives on unlabeled data, followed by fine-tuning on downstream tasks, has emerged as a standard procedure. Unfortunately, the efficacy of this approach is often constrained by both limited fine-tuning compute and scarcity in labeled downstream data. We introduce Multimodal Attention Merging (MAM), an attempt that facilitates direct knowledge transfer from attention matrices of models rooted in high resource modalities, text and images, to those in resource-constrained domains, speech and audio, employing a zero-shot paradigm. MAM reduces the relative Word Error Rate (WER) of an Automatic Speech Recognition (ASR) model by up to 6.70%, and relative classification error of an Audio Event Classification (AEC) model by 10.63%. In cases where some data/compute is available, we present Learnable-MAM, a data-driven approach to merging attention matrices, resulting in a further 2.90% relative reduction in WER for ASR and 18.42% relative reduction in AEC compared to fine-tuning.

{{</citation>}}


## cs.AI (9)



### (15/80) NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes (Lizhou Fan et al., 2023)

{{<citation>}}

Lizhou Fan, Wenyue Hua, Lingyao Li, Haoyang Ling, Yongfeng Zhang, Libby Hemphill. (2023)  
**NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CC, cs-CL, cs-LG, cs.AI  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.14890v1)  

---


**ABSTRACT**  
Complex reasoning ability is one of the most important features of current LLMs, which has also been leveraged to play an integral role in complex decision-making tasks. Therefore, the investigation into the reasoning capabilities of Large Language Models (LLMs) is critical: numerous benchmarks have been established to assess the reasoning abilities of LLMs. However, current benchmarks are inadequate in offering a rigorous evaluation of the full extent of reasoning abilities that LLMs are capable of achieving. They are also prone to the risk of overfitting, as these benchmarks, being publicly accessible and static, allow models to potentially tailor their responses to specific benchmark metrics, thereby inflating their performance. Addressing these limitations, our research introduces a new benchmark, named NPHardEval. This benchmark is designed to evaluate the reasoning abilities of LLMs across a broad spectrum of 900 algorithmic questions, extending up to the NP-Hard complexity class. These questions are meticulously chosen to represent a wide range of complexity class below the NP-hard complexity class, offering a rigorous measure of the reasoning ability of LLMs. Through this study, we shed light on the current state of reasoning in LLMs, providing an objective and rigorous perspective through the comparison of LLMs' performance across complex classes. Moreover, this benchmark is designed with a dynamic update mechanism, where the datapoints are refreshed on a monthly basis. Such regular updates play a crucial role in mitigating the risk of LLMs overfitting to the benchmark, promoting a more accurate and reliable assessment of their reasoning capabilities. The benchmark dataset and code of NPHardEval are available at https://github.com/casmlab/NPHardEval.

{{</citation>}}


### (16/80) Pangu-Agent: A Fine-Tunable Generalist Agent with Structured Reasoning (Filippos Christianos et al., 2023)

{{<citation>}}

Filippos Christianos, Georgios Papoudakis, Matthieu Zimmer, Thomas Coste, Zhihao Wu, Jingxuan Chen, Khyati Khandelwal, James Doran, Xidong Feng, Jiacheng Liu, Zheng Xiong, Yicheng Luo, Jianye Hao, Kun Shao, Haitham Bou-Ammar, Jun Wang. (2023)  
**Pangu-Agent: A Fine-Tunable Generalist Agent with Structured Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Reasoning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14878v1)  

---


**ABSTRACT**  
A key method for creating Artificial Intelligence (AI) agents is Reinforcement Learning (RL). However, constructing a standalone RL policy that maps perception to action directly encounters severe problems, chief among them being its lack of generality across multiple tasks and the need for a large amount of training data. The leading cause is that it cannot effectively integrate prior information into the perception-action cycle when devising the policy. Large language models (LLMs) emerged as a fundamental way to incorporate cross-domain knowledge into AI agents but lack crucial learning and adaptation toward specific decision problems. This paper presents a general framework model for integrating and learning structured reasoning into AI agents' policies. Our methodology is motivated by the modularity found in the human brain. The framework utilises the construction of intrinsic and extrinsic functions to add previous understandings of reasoning structures. It also provides the adaptive ability to learn models inside every module or function, consistent with the modular structure of cognitive processes. We describe the framework in-depth and compare it with other AI pipelines and existing frameworks. The paper explores practical applications, covering experiments that show the effectiveness of our method. Our results indicate that AI agents perform and adapt far better when organised reasoning and prior knowledge are embedded. This opens the door to more resilient and general AI agent systems.

{{</citation>}}


### (17/80) TACO: Topics in Algorithmic COde generation dataset (Rongao Li et al., 2023)

{{<citation>}}

Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, Ge Li. (2023)  
**TACO: Topics in Algorithmic COde generation dataset**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14852v1)  

---


**ABSTRACT**  
We introduce TACO, an open-source, large-scale code generation dataset, with a focus on the optics of algorithms, designed to provide a more challenging training dataset and evaluation benchmark in the field of code generation models. TACO includes competition-level programming questions that are more challenging, to enhance or evaluate problem understanding and reasoning abilities in real-world programming scenarios. There are 25433 and 1000 coding problems in training and test set, as well as up to 1.55 million diverse solution answers. Moreover, each TACO problem includes several fine-grained labels such as task topics, algorithms, programming skills, and difficulty levels, providing a more precise reference for the training and evaluation of code generation models. The dataset and evaluation scripts are available on Hugging Face Hub (https://huggingface.co/datasets/BAAI/TACO) and Github (https://github.com/FlagOpen/TACO).

{{</citation>}}


### (18/80) An investigation of belief-free DRL and MCTS for inspection and maintenance planning (Daniel Koutas et al., 2023)

{{<citation>}}

Daniel Koutas, Elizabeth Bismut, Daniel Straub. (2023)  
**An investigation of belief-free DRL and MCTS for inspection and maintenance planning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14824v1)  

---


**ABSTRACT**  
We propose a novel Deep Reinforcement Learning (DRL) architecture for sequential decision processes under uncertainty, as encountered in inspection and maintenance (I&M) planning. Unlike other DRL algorithms for (I&M) planning, the proposed +RQN architecture dispenses with computing the belief state and directly handles erroneous observations instead. We apply the algorithm to a basic I&M planning problem for a one-component system subject to deterioration. In addition, we investigate the performance of Monte Carlo tree search for the I&M problem and compare it to the +RQN. The comparison includes a statistical analysis of the two methods' resulting policies, as well as their visualization in the belief space.

{{</citation>}}


### (19/80) Hierarchical Multi-Agent Reinforcement Learning for Assessing False-Data Injection Attacks on Transportation Networks (Taha Eghtesad et al., 2023)

{{<citation>}}

Taha Eghtesad, Sirui Li, Yevgeniy Vorobeychik, Aron Laszka. (2023)  
**Hierarchical Multi-Agent Reinforcement Learning for Assessing False-Data Injection Attacks on Transportation Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CR, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14625v1)  

---


**ABSTRACT**  
The increasing reliance of drivers on navigation applications has made transportation networks more susceptible to data-manipulation attacks by malicious actors. Adversaries may exploit vulnerabilities in the data collection or processing of navigation services to inject false information, and to thus interfere with the drivers' route selection. Such attacks can significantly increase traffic congestions, resulting in substantial waste of time and resources, and may even disrupt essential services that rely on road networks. To assess the threat posed by such attacks, we introduce a computational framework to find worst-case data-injection attacks against transportation networks. First, we devise an adversarial model with a threat actor who can manipulate drivers by increasing the travel times that they perceive on certain roads. Then, we employ hierarchical multi-agent reinforcement learning to find an approximate optimal adversarial strategy for data manipulation. We demonstrate the applicability of our approach through simulating attacks on the Sioux Falls, ND network topology.

{{</citation>}}


### (20/80) Adaptive Reconvergence-driven AIG Rewriting via Strategy Learning (Liwei Ni et al., 2023)

{{<citation>}}

Liwei Ni, Zonglin Yang, Jiaxi Zhang, Junfeng Liu, Huawei Li, Biwei Xie, Xinquan Li. (2023)  
**Adaptive Reconvergence-driven AIG Rewriting via Strategy Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-AR, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14536v1)  

---


**ABSTRACT**  
Rewriting is a common procedure in logic synthesis aimed at improving the performance, power, and area (PPA) of circuits. The traditional reconvergence-driven And-Inverter Graph (AIG) rewriting method focuses solely on optimizing the reconvergence cone through Boolean algebra minimization. However, there exist opportunities to incorporate other node-rewriting algorithms that are better suited for specific cones. In this paper, we propose an adaptive reconvergence-driven AIG rewriting algorithm that combines two key techniques: multi-strategy-based AIG rewriting and strategy learning-based algorithm selection. The multi-strategy-based rewriting method expands upon the traditional approach by incorporating support for multi-node-rewriting algorithms, thus expanding the optimization space. Additionally, the strategy learning-based algorithm selection method determines the most suitable node-rewriting algorithm for a given cone. Experimental results demonstrate that our proposed method yields a significant average improvement of 5.567\% in size and 5.327\% in depth.

{{</citation>}}


### (21/80) Not All Tasks Are Equally Difficult: Multi-Task Reinforcement Learning with Dynamic Depth Routing (Jinmin He et al., 2023)

{{<citation>}}

Jinmin He, Kai Li, Yifan Zang, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng. (2023)  
**Not All Tasks Are Equally Difficult: Multi-Task Reinforcement Learning with Dynamic Depth Routing**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14472v1)  

---


**ABSTRACT**  
Multi-task reinforcement learning endeavors to accomplish a set of different tasks with a single policy. To enhance data efficiency by sharing parameters across multiple tasks, a common practice segments the network into distinct modules and trains a routing network to recombine these modules into task-specific policies. However, existing routing approaches employ a fixed number of modules for all tasks, neglecting that tasks with varying difficulties commonly require varying amounts of knowledge. This work presents a Dynamic Depth Routing (D2R) framework, which learns strategic skipping of certain intermediate modules, thereby flexibly choosing different numbers of modules for each task. Under this framework, we further introduce a ResRouting method to address the issue of disparate routing paths between behavior and target policies during off-policy training. In addition, we design an automatic route-balancing mechanism to encourage continued routing exploration for unmastered tasks without disturbing the routing of mastered ones. We conduct extensive experiments on various robotics manipulation tasks in the Meta-World benchmark, where D2R achieves state-of-the-art performance with significantly improved learning efficiency.

{{</citation>}}


### (22/80) The Fairness Fair: Bringing Human Perception into Collective Decision-Making (Hadi Hosseini, 2023)

{{<citation>}}

Hadi Hosseini. (2023)  
**The Fairness Fair: Bringing Human Perception into Collective Decision-Making**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-GT, cs-MA, cs.AI, econ-TH  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14402v1)  

---


**ABSTRACT**  
Fairness is one of the most desirable societal principles in collective decision-making. It has been extensively studied in the past decades for its axiomatic properties and has received substantial attention from the multiagent systems community in recent years for its theoretical and computational aspects in algorithmic decision-making. However, these studies are often not sufficiently rich to capture the intricacies of human perception of fairness in the ambivalent nature of the real-world problems. We argue that not only fair solutions should be deemed desirable by social planners (designers), but they should be governed by human and societal cognition, consider perceived outcomes based on human judgement, and be verifiable. We discuss how achieving this goal requires a broad transdisciplinary approach ranging from computing and AI to behavioral economics and human-AI interaction. In doing so, we identify shortcomings and long-term challenges of the current literature of fair division, describe recent efforts in addressing them, and more importantly, highlight a series of open research directions.

{{</citation>}}


### (23/80) Logic-Scaffolding: Personalized Aspect-Instructed Recommendation Explanation Generation using LLMs (Behnam Rahdari et al., 2023)

{{<citation>}}

Behnam Rahdari, Hao Ding, Ziwei Fan, Yifei Ma, Zhuotong Chen, Anoop Deoras, Branislav Kveton. (2023)  
**Logic-Scaffolding: Personalized Aspect-Instructed Recommendation Explanation Generation using LLMs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-HC, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.14345v1)  

---


**ABSTRACT**  
The unique capabilities of Large Language Models (LLMs), such as the natural language text generation ability, position them as strong candidates for providing explanation for recommendations. However, despite the size of the LLM, most existing models struggle to produce zero-shot explanations reliably. To address this issue, we propose a framework called Logic-Scaffolding, that combines the ideas of aspect-based explanation and chain-of-thought prompting to generate explanations through intermediate reasoning steps. In this paper, we share our experience in building the framework and present an interactive demonstration for exploring our results.

{{</citation>}}


## cs.CL (13)



### (24/80) Robust Knowledge Extraction from Large Language Models using Social Choice Theory (Nico Potyka et al., 2023)

{{<citation>}}

Nico Potyka, Yuqicheng Zhu, Yunjie He, Evgeny Kharlamov, Steffen Staab. (2023)  
**Robust Knowledge Extraction from Large Language Models using Social Choice Theory**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.14877v1)  

---


**ABSTRACT**  
Large-language models (LLMs) have the potential to support a wide range of applications like conversational agents, creative writing, text improvement, and general query answering. However, they are ill-suited for query answering in high-stake domains like medicine because they generate answers at random and their answers are typically not robust - even the same query can result in different answers when prompted multiple times. In order to improve the robustness of LLM queries, we propose using ranking queries repeatedly and to aggregate the queries using methods from social choice theory. We study ranking queries in diagnostic settings like medical and fault diagnosis and discuss how the Partial Borda Choice function from the literature can be applied to merge multiple query results. We discuss some additional interesting properties in our setting and evaluate the robustness of our approach empirically.

{{</citation>}}


### (25/80) Numerical Reasoning for Financial Reports (Abhinav Arun et al., 2023)

{{<citation>}}

Abhinav Arun, Ashish Dhiman, Mehul Soni, Yibei Hu. (2023)  
**Numerical Reasoning for Financial Reports**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Financial, Language Model, QA, Reasoning, T5  
[Paper Link](http://arxiv.org/abs/2312.14870v1)  

---


**ABSTRACT**  
Financial reports offer critical insights into a company's operations, yet their extensive length typically spanning 30 40 pages poses challenges for swift decision making in dynamic markets. To address this, we leveraged finetuned Large Language Models (LLMs) to distill key indicators and operational metrics from these reports basis questions from the user. We devised a method to locate critical data, and leverage the FinQA dataset to fine-tune both Llama-2 7B and T5 models for customized question answering. We achieved results comparable to baseline on the final numerical answer, a competitive accuracy in numerical reasoning and calculation.

{{</citation>}}


### (26/80) YAYI 2: Multilingual Open-Source Large Language Models (Yin Luo et al., 2023)

{{<citation>}}

Yin Luo, Qingchao Kong, Nan Xu, Jia Cao, Bao Hao, Baoyu Qu, Bo Chen, Chao Zhu, Chenyang Zhao, Donglei Zhang, Fan Feng, Feifei Zhao, Hailong Sun, Hanxuan Yang, Haojun Pan, Hongyu Liu, Jianbin Guo, Jiangtao Du, Jingyi Wang, Junfeng Li, Lei Sun, Liduo Liu, Lifeng Dong, Lili Liu, Lin Wang, Liwen Zhang, Minzheng Wang, Pin Wang, Ping Yu, Qingxiao Li, Rui Yan, Rui Zou, Ruiqun Li, Taiwen Huang, Xiaodong Wang, Xiaofei Wu, Xin Peng, Xina Zhang, Xing Fang, Xinglin Xiao, Yanni Hao, Yao Dong, Yigang Wang, Ying Liu, Yongyu Jiang, Yungan Wang, Yuqi Wang, Zhangsheng Wang, Zhaoxin Yu, Zhen Luo, Wenji Mao, Lei Wang, Dajun Zeng. (2023)  
**YAYI 2: Multilingual Open-Source Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Falcon, Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2312.14862v1)  

---


**ABSTRACT**  
As the latest advancements in natural language processing, large language models (LLMs) have achieved human-level language understanding and generation abilities in many real-world tasks, and even have been regarded as a potential path to the artificial general intelligence. To better facilitate research on LLMs, many open-source LLMs, such as Llama 2 and Falcon, have recently been proposed and gained comparable performances to proprietary models. However, these models are primarily designed for English scenarios and exhibit poor performances in Chinese contexts. In this technical report, we propose YAYI 2, including both base and chat models, with 30 billion parameters. YAYI 2 is pre-trained from scratch on a multilingual corpus which contains 2.65 trillion tokens filtered by our pre-training data processing pipeline. The base model is aligned with human values through supervised fine-tuning with millions of instructions and reinforcement learning from human feedback. Extensive experiments on multiple benchmarks, such as MMLU and CMMLU, consistently demonstrate that the proposed YAYI 2 outperforms other similar sized open-source models.

{{</citation>}}


### (27/80) On the Use of Metaphor Translation in Psychiatry (Lois Wong, 2023)

{{<citation>}}

Lois Wong. (2023)  
**On the Use of Metaphor Translation in Psychiatry**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.14845v1)  

---


**ABSTRACT**  
Providing mental healthcare to individuals with limited English proficiency (LEP) remains a pressing problem within psychiatry. Because the majority of individuals trained in providing psychiatric care are English speakers, the quality of mental healthcare given to LEP patients is significantly lower than that provided for English speakers. The provision of mental healthcare is contingent on communication and understanding between the patient and healthcare provider, much more so than in the realm of physical healthcare, and English speakers are often unable to comprehend figurative language such as metaphors used by LEPs. Hence, Figurative Language Translation is invaluable to providing equitable psychiatric care. Now, metaphor has been shown to be paramount in both identifying individuals struggling with mental problems and helping those individuals understand and communicate their experiences. Therefore, this paper aims to survey the potential of Machine Translation for providing equitable psychiatric healthcare and highlights the need for further research on the transferability of existing machine and metaphor translation research in the domain of psychiatry.

{{</citation>}}


### (28/80) Semantic Parsing for Complex Data Retrieval: Targeting Query Plans vs. SQL for No-Code Access to Relational Databases (Ben Eyal et al., 2023)

{{<citation>}}

Ben Eyal, Amir Bachar, Ophir Haroche, Michael Elhadad. (2023)  
**Semantic Parsing for Complex Data Retrieval: Targeting Query Plans vs. SQL for No-Code Access to Relational Databases**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.14798v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have spurred progress in text-to-SQL, the task of generating SQL queries from natural language questions based on a given database schema. Despite the declarative nature of SQL, it continues to be a complex programming language. In this paper, we investigate the potential of an alternative query language with simpler syntax and modular specification of complex queries. The purpose is to create a query language that can be learned more easily by modern neural semantic parsing architectures while also enabling non-programmers to better assess the validity of the query plans produced by an interactive query plan assistant.   The proposed alternative query language is called Query Plan Language (QPL). It is designed to be modular and can be translated into a restricted form of SQL Common Table Expressions (CTEs). The aim of QPL is to make complex data retrieval accessible to non-programmers by allowing users to express their questions in natural language while also providing an easier-to-verify target language. The paper demonstrates how neural LLMs can benefit from QPL's modularity to generate complex query plans in a compositional manner. This involves a question decomposition strategy and a planning stage.   We conduct experiments on a version of the Spider text-to-SQL dataset that has been converted to QPL. The hierarchical structure of QPL programs enables us to measure query complexity naturally. Based on this assessment, we identify the low accuracy of existing text-to-SQL systems on complex compositional queries. We present ways to address the challenge of complex queries in an iterative, user-controlled manner, using fine-tuned LLMs and a variety of prompting strategies in a compositional manner.

{{</citation>}}


### (29/80) Large Language Model (LLM) Bias Index -- LLMBI (Abiodun Finbarrs Oketunji et al., 2023)

{{<citation>}}

Abiodun Finbarrs Oketunji, Muhammad Anas, Deepthi Saina. (2023)  
**Large Language Model (LLM) Bias Index -- LLMBI**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: AI, Bias, GPT, GPT-4, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.14769v1)  

---


**ABSTRACT**  
The Large Language Model Bias Index (LLMBI) is a pioneering approach designed to quantify and address biases inherent in large language models (LLMs), such as GPT-4. We recognise the increasing prevalence and impact of LLMs across diverse sectors. This research introduces a novel metric, LLMBI, to systematically measure and mitigate biases potentially skewing model responses. We formulated LLMBI using a composite scoring system incorporating multiple dimensions of bias, including but not limited to age, gender, and racial biases.   To operationalise this metric, we engaged in a multi-step process involving collecting and annotating LLM responses, applying sophisticated Natural Language Processing (NLP) techniques for bias detection, and computing the LLMBI score through a specially crafted mathematical formula. The formula integrates weighted averages of various bias dimensions, a penalty for dataset diversity deficiencies, and a correction for sentiment biases. Our empirical analysis, conducted using responses from OpenAI's API, employs advanced sentiment analysis as a representative method for bias detection.   The research reveals LLMs, whilst demonstrating impressive capabilities in text generation, exhibit varying degrees of bias across different dimensions. LLMBI provides a quantifiable measure to compare biases across models and over time, offering a vital tool for systems engineers, researchers and regulators in enhancing the fairness and reliability of LLMs. It highlights the potential of LLMs in mimicking unbiased human-like responses. Additionally, it underscores the necessity of continuously monitoring and recalibrating such models to align with evolving societal norms and ethical standards.

{{</citation>}}


### (30/80) Reasons to Reject? Aligning Language Models with Judgments (Weiwen Xu et al., 2023)

{{<citation>}}

Weiwen Xu, Deng Cai, Zhisong Zhang, Wai Lam, Shuming Shi. (2023)  
**Reasons to Reject? Aligning Language Models with Judgments**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2312.14591v1)  

---


**ABSTRACT**  
As humans, we consistently engage in interactions with our peers and receive feedback in the form of natural language. This language feedback allows us to reflect on our actions, maintain appropriate behavior, and rectify our errors. The question arises naturally: can we use language feedback to align large language models (LLMs)? In contrast to previous research that aligns LLMs with reward or preference data, we present the first systematic exploration of alignment through the lens of language feedback (i.e., judgment). We commence with an in-depth investigation of potential methods that can be adapted for aligning LLMs with judgments, revealing that these methods are unable to fully capitalize on the judgments. To facilitate more effective utilization of judgments, we propose a novel framework, Contrastive Unlikelihood Training (CUT), that allows for fine-grained inappropriate content detection and correction based on judgments. Our offline alignment results show that, with merely 1317 off-the-shelf judgment data, CUT (LLaMA2-13b) can beat the 175B DaVinci003 and surpass the best baseline by 52.34 points on AlpacaEval. The online alignment results demonstrate that CUT can align LLMs (LLaMA2-chat-13b) in an iterative fashion using model-specific judgment data, with a steady performance improvement from 81.09 to 91.36 points on AlpacaEval. Our analysis further suggests that judgments exhibit greater potential than rewards for LLM alignment and warrant future research.

{{</citation>}}


### (31/80) SIG: Speaker Identification in Literature via Prompt-Based Generation (Zhenlin Su et al., 2023)

{{<citation>}}

Zhenlin Su, Liyan Xu, Jin Xu, Jiangnan Li, Mingdu Huangfu. (2023)  
**SIG: Speaker Identification in Literature via Prompt-Based Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.14590v1)  

---


**ABSTRACT**  
Identifying speakers of quotations in narratives is an important task in literary analysis, with challenging scenarios including the out-of-domain inference for unseen speakers, and non-explicit cases where there are no speaker mentions in surrounding context. In this work, we propose a simple and effective approach SIG, a generation-based method that verbalizes the task and quotation input based on designed prompt templates, which also enables easy integration of other auxiliary tasks that further bolster the speaker identification performance. The prediction can either come from direct generation by the model, or be determined by the highest generation probability of each speaker candidate. Based on our approach design, SIG supports out-of-domain evaluation, and achieves open-world classification paradigm that is able to accept any forms of candidate input. We perform both cross-domain evaluation and in-domain evaluation on PDNC, the largest dataset of this task, where empirical results suggest that SIG outperforms previous baselines of complicated designs, as well as the zero-shot ChatGPT, especially excelling at those hard non-explicit scenarios by up to 17% improvement. Additional experiments on another dataset WP further corroborate the efficacy of SIG.

{{</citation>}}


### (32/80) Automatic Data Retrieval for Cross Lingual Summarization (Nikhilesh Bhatnagar et al., 2023)

{{<citation>}}

Nikhilesh Bhatnagar, Ashok Urlana, Vandan Mujadia, Pruthwik Mishra, Dipti Misra Sharma. (2023)  
**Automatic Data Retrieval for Cross Lingual Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2312.14542v1)  

---


**ABSTRACT**  
Cross-lingual summarization involves the summarization of text written in one language to a different one. There is a body of research addressing cross-lingual summarization from English to other European languages. In this work, we aim to perform cross-lingual summarization from English to Hindi. We propose pairing up the coverage of newsworthy events in textual and video format can prove to be helpful for data acquisition for cross lingual summarization. We analyze the data and propose methods to match articles to video descriptions that serve as document and summary pairs. We also outline filtering methods over reasonable thresholds to ensure the correctness of the summaries. Further, we make available 28,583 mono and cross-lingual article-summary pairs https://github.com/tingc9/Cross-Sum-News-Aligned. We also build and analyze multiple baselines on the collected data and report error analysis.

{{</citation>}}


### (33/80) Theory of Hallucinations based on Equivariance (Hisaichi Shibata, 2023)

{{<citation>}}

Hisaichi Shibata. (2023)  
**Theory of Hallucinations based on Equivariance**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: T5, Transformer  
[Paper Link](http://arxiv.org/abs/2312.14504v1)  

---


**ABSTRACT**  
Equivariance is an important feature in machine learning, including language models. It ensures that any sequences of phrases with the same meanings are interpreted consistently. For example, the sentence 'There is a cat on the table' should be interpreted by language models as it is, regardless of variations in its token-level expression. Building on this insight, I propose a new theory suggesting that insufficient equivariance in language models can lead to hallucinations. According to this theory, which is both intuitive and novel, language models trained on relatively small datasets tend to misinterpret input texts and/or generate incorrect texts (i.e., hallucinations). To test this theory, I developed a toy model known as 'dancing men', which is a character-level substitution cipher. Additionally, I propose a novel technique based on the T5 (Text To Text Transfer Transformer) model to efficiently decipher these codes without relying on frequency analysis. I have found that this T5 model can almost completely solve the cipher, demonstrating its ability to acquire equivariance in this frame. This method could be scaled up to word-level and sentence-level substitution ciphers, analogous to large language models without tokenizers or dictionaries. This scalability makes it suitable for investigating the proposed link between inadequate equivariance acquisition and the emergence of hallucinations.

{{</citation>}}


### (34/80) Language Model is a Branch Predictor for Simultaneous Machine Translation (Aoxiong Yin et al., 2023)

{{<citation>}}

Aoxiong Yin, Tianyun Zhong, Haoyuan Li, Siliang Tang, Zhou Zhao. (2023)  
**Language Model is a Branch Predictor for Simultaneous Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.14488v1)  

---


**ABSTRACT**  
The primary objective of simultaneous machine translation (SiMT) is to minimize latency while preserving the quality of the final translation. Drawing inspiration from CPU branch prediction techniques, we propose incorporating branch prediction techniques in SiMT tasks to reduce translation latency. Specifically, we utilize a language model as a branch predictor to predict potential branch directions, namely, future source words. Subsequently, we utilize the predicted source words to decode the output in advance. When the actual source word deviates from the predicted source word, we use the real source word to decode the output again, replacing the predicted output. To further reduce computational costs, we share the parameters of the encoder and the branch predictor, and utilize a pre-trained language model for initialization. Our proposed method can be seamlessly integrated with any SiMT model. Extensive experimental results demonstrate that our approach can improve translation quality and latency at the same time. Our code is available at https://github.com/YinAoXiong/simt_branch_predictor .

{{</citation>}}


### (35/80) Efficacy of Machine-Generated Instructions (Samaksh Gulati et al., 2023)

{{<citation>}}

Samaksh Gulati, Anshit Verma, Manoj Parmar, Palash Chaudhary. (2023)  
**Efficacy of Machine-Generated Instructions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT  
[Paper Link](http://arxiv.org/abs/2312.14423v1)  

---


**ABSTRACT**  
Large "instruction-tuned" language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend heavily on human-written instruction data that is often limited in quantity, diversity, and creativity, therefore hindering the generality of the tuned model. We conducted a quantitative study to figure out the efficacy of machine-generated annotations, where we compare the results of a fine-tuned BERT model with human v/s machine-generated annotations. Applying our methods to the vanilla GPT-3 model, we saw that machine generated annotations were 78.54% correct and the fine-tuned model achieved a 96.01% model performance compared to the performance with human-labelled annotations. This result shows that machine-generated annotations are a resource and cost effective way to fine-tune down-stream models.

{{</citation>}}


### (36/80) Don't Believe Everything You Read: Enhancing Summarization Interpretability through Automatic Identification of Hallucinations in Large Language Models (Priyesh Vakharia et al., 2023)

{{<citation>}}

Priyesh Vakharia, Devavrat Joshi, Meenal Chavan, Dhananjay Sonawane, Bhrigu Garg, Parsa Mazaheri, Ian Lane. (2023)  
**Don't Believe Everything You Read: Enhancing Summarization Interpretability through Automatic Identification of Hallucinations in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2312.14346v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are adept at text manipulation -- tasks such as machine translation and text summarization. However, these models can also be prone to hallucination, which can be detrimental to the faithfulness of any answers that the model provides. Recent works in combating hallucinations in LLMs deal with identifying hallucinated sentences and categorizing the different ways in which models hallucinate. This paper takes a deep dive into LLM behavior with respect to hallucinations, defines a token-level approach to identifying different kinds of hallucinations, and further utilizes this token-level tagging to improve the interpretability and faithfulness of LLMs in dialogue summarization tasks. Through this, the paper presents a new, enhanced dataset and a new training paradigm.

{{</citation>}}


## cs.CV (14)



### (37/80) VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation (Max Ku et al., 2023)

{{<citation>}}

Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, Wenhu Chen. (2023)  
**VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-MM, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.14867v1)  

---


**ABSTRACT**  
In the rapidly advancing field of conditional image generation research, challenges such as limited explainability lie in effectively evaluating the performance and capabilities of various models. This paper introduces VIESCORE, a Visual Instruction-guided Explainable metric for evaluating any conditional image generation tasks. VIESCORE leverages general knowledge from Multimodal Large Language Models (MLLMs) as the backbone and does not require training or fine-tuning. We evaluate VIESCORE on seven prominent tasks in conditional image tasks and found: (1) VIESCORE (GPT4-v) achieves a high Spearman correlation of 0.3 with human evaluations, while the human-to-human correlation is 0.45. (2) VIESCORE (with open-source MLLM) is significantly weaker than GPT-4v in evaluating synthetic images. (3) VIESCORE achieves a correlation on par with human ratings in the generation tasks but struggles in editing tasks. With these results, we believe VIESCORE shows its great potential to replace human judges in evaluating image synthesis tasks.

{{</citation>}}


### (38/80) Plan, Posture and Go: Towards Open-World Text-to-Motion Generation (Jinpeng Liu et al., 2023)

{{<citation>}}

Jinpeng Liu, Wenxun Dai, Chunyu Wang, Yiji Cheng, Yansong Tang, Xin Tong. (2023)  
**Plan, Posture and Go: Towards Open-World Text-to-Motion Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.14828v1)  

---


**ABSTRACT**  
Conventional text-to-motion generation methods are usually trained on limited text-motion pairs, making them hard to generalize to open-world scenarios. Some works use the CLIP model to align the motion space and the text space, aiming to enable motion generation from natural language motion descriptions. However, they are still constrained to generate limited and unrealistic in-place motions. To address these issues, we present a divide-and-conquer framework named PRO-Motion, which consists of three modules as motion planner, posture-diffuser and go-diffuser. The motion planner instructs Large Language Models (LLMs) to generate a sequence of scripts describing the key postures in the target motion. Differing from natural languages, the scripts can describe all possible postures following very simple text templates. This significantly reduces the complexity of posture-diffuser, which transforms a script to a posture, paving the way for open-world generation. Finally, go-diffuser, implemented as another diffusion model, estimates whole-body translations and rotations for all postures, resulting in realistic motions. Experimental results have shown the superiority of our method with other counterparts, and demonstrated its capability of generating diverse and realistic motions from complex open-world prompts such as "Experiencing a profound sense of joy". The project page is available at https://moonsliu.github.io/Pro-Motion.

{{</citation>}}


### (39/80) Global Occlusion-Aware Transformer for Robust Stereo Matching (Zihua Liu et al., 2023)

{{<citation>}}

Zihua Liu, Yizhou Li, Masatoshi Okutomi. (2023)  
**Global Occlusion-Aware Transformer for Robust Stereo Matching**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.14650v1)  

---


**ABSTRACT**  
Despite the remarkable progress facilitated by learning-based stereo-matching algorithms, the performance in the ill-conditioned regions, such as the occluded regions, remains a bottleneck. Due to the limited receptive field, existing CNN-based methods struggle to handle these ill-conditioned regions effectively. To address this issue, this paper introduces a novel attention-based stereo-matching network called Global Occlusion-Aware Transformer (GOAT) to exploit long-range dependency and occlusion-awareness global context for disparity estimation. In the GOAT architecture, a parallel disparity and occlusion estimation module PDO is proposed to estimate the initial disparity map and the occlusion mask using a parallel attention mechanism. To further enhance the disparity estimates in the occluded regions, an occlusion-aware global aggregation module (OGA) is proposed. This module aims to refine the disparity in the occluded regions by leveraging restricted global correlation within the focus scope of the occluded areas. Extensive experiments were conducted on several public benchmark datasets including SceneFlow, KITTI 2015, and Middlebury. The results show that the proposed GOAT demonstrates outstanding performance among all benchmarks, particularly in the occluded regions.

{{</citation>}}


### (40/80) DSAP: Analyzing Bias Through Demographic Comparison of Datasets (Iris Dominguez-Catena et al., 2023)

{{<citation>}}

Iris Dominguez-Catena, Daniel Paternain, Mikel Galar. (2023)  
**DSAP: Analyzing Bias Through Demographic Comparison of Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.14626v1)  

---


**ABSTRACT**  
In the last few years, Artificial Intelligence systems have become increasingly widespread. Unfortunately, these systems can share many biases with human decision-making, including demographic biases. Often, these biases can be traced back to the data used for training, where large uncurated datasets have become the norm. Despite our knowledge of these biases, we still lack general tools to detect and quantify them, as well as to compare the biases in different datasets. Thus, in this work, we propose DSAP (Demographic Similarity from Auxiliary Profiles), a two-step methodology for comparing the demographic composition of two datasets. DSAP can be deployed in three key applications: to detect and characterize demographic blind spots and bias issues across datasets, to measure dataset demographic bias in single datasets, and to measure dataset demographic shift in deployment scenarios. An essential feature of DSAP is its ability to robustly analyze datasets without explicit demographic labels, offering simplicity and interpretability for a wide range of situations. To show the usefulness of the proposed methodology, we consider the Facial Expression Recognition task, where demographic bias has previously been found. The three applications are studied over a set of twenty datasets with varying properties. The code is available at https://github.com/irisdominguez/DSAP.

{{</citation>}}


### (41/80) Explainable Multi-Camera 3D Object Detection with Transformer-Based Saliency Maps (Till Beemelmanns et al., 2023)

{{<citation>}}

Till Beemelmanns, Wassim Zahr, Lutz Eckstein. (2023)  
**Explainable Multi-Camera 3D Object Detection with Transformer-Based Saliency Maps**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI, Object Detection, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.14606v1)  

---


**ABSTRACT**  
Vision Transformers (ViTs) have achieved state-of-the-art results on various computer vision tasks, including 3D object detection. However, their end-to-end implementation also makes ViTs less explainable, which can be a challenge for deploying them in safety-critical applications, such as autonomous driving, where it is important for authorities, developers, and users to understand the model's reasoning behind its predictions. In this paper, we propose a novel method for generating saliency maps for a DetR-like ViT with multiple camera inputs used for 3D object detection. Our method is based on the raw attention and is more efficient than gradient-based methods. We evaluate the proposed method on the nuScenes dataset using extensive perturbation tests and show that it outperforms other explainability methods in terms of visual quality and quantitative metrics. We also demonstrate the importance of aggregating attention across different layers of the transformer. Our work contributes to the development of explainable AI for ViTs, which can help increase trust in AI applications by establishing more transparency regarding the inner workings of AI models.

{{</citation>}}


### (42/80) PoseViNet: Distracted Driver Action Recognition Framework Using Multi-View Pose Estimation and Vision Transformer (Neha Sengar et al., 2023)

{{<citation>}}

Neha Sengar, Indra Kumari, Jihui Lee, Dongsoo Har. (2023)  
**PoseViNet: Distracted Driver Action Recognition Framework Using Multi-View Pose Estimation and Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.14577v1)  

---


**ABSTRACT**  
Driver distraction is a principal cause of traffic accidents. In a study conducted by the National Highway Traffic Safety Administration, engaging in activities such as interacting with in-car menus, consuming food or beverages, or engaging in telephonic conversations while operating a vehicle can be significant sources of driver distraction. From this viewpoint, this paper introduces a novel method for detection of driver distraction using multi-view driver action images. The proposed method is a vision transformer-based framework with pose estimation and action inference, namely PoseViNet. The motivation for adding posture information is to enable the transformer to focus more on key features. As a result, the framework is more adept at identifying critical actions. The proposed framework is compared with various state-of-the-art models using SFD3 dataset representing 10 behaviors of drivers. It is found from the comparison that the PoseViNet outperforms these models. The proposed framework is also evaluated with the SynDD1 dataset representing 16 behaviors of driver. As a result, the PoseViNet achieves 97.55% validation accuracy and 90.92% testing accuracy with the challenging dataset.

{{</citation>}}


### (43/80) MMGPL: Multimodal Medical Data Analysis with Graph Prompt Learning (Liang Peng et al., 2023)

{{<citation>}}

Liang Peng, Songyue Cai, Zongqian Wu, Huifang Shang, Xiaofeng Zhu, Xiaoxiao Li. (2023)  
**MMGPL: Multimodal Medical Data Analysis with Graph Prompt Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.14574v1)  

---


**ABSTRACT**  
Prompt learning has demonstrated impressive efficacy in the fine-tuning of multimodal large models to a wide range of downstream tasks. Nonetheless, applying existing prompt learning methods for the diagnosis of neurological disorder still suffers from two issues: (i) existing methods typically treat all patches equally, despite the fact that only a small number of patches in neuroimaging are relevant to the disease, and (ii) they ignore the structural information inherent in the brain connection network which is crucial for understanding and diagnosing neurological disorders. To tackle these issues, we introduce a novel prompt learning model by learning graph prompts during the fine-tuning process of multimodal large models for diagnosing neurological disorders. Specifically, we first leverage GPT-4 to obtain relevant disease concepts and compute semantic similarity between these concepts and all patches. Secondly, we reduce the weight of irrelevant patches according to the semantic similarity between each patch and disease-related concepts. Moreover, we construct a graph among tokens based on these concepts and employ a graph convolutional network layer to extract the structural information of the graph, which is used to prompt the pre-trained multimodal large models for diagnosing neurological disorders. Extensive experiments demonstrate that our method achieves superior performance for neurological disorder diagnosis compared with state-of-the-art methods and validated by clinicians.

{{</citation>}}


### (44/80) ViStripformer: A Token-Efficient Transformer for Versatile Video Restoration (Fu-Jen Tsai et al., 2023)

{{<citation>}}

Fu-Jen Tsai, Yan-Tsung Peng, Chen-Yu Chang, Chan-Yu Li, Yen-Yu Lin, Chung-Chi Tsai, Chia-Wen Lin. (2023)  
**ViStripformer: A Token-Efficient Transformer for Versatile Video Restoration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.14502v1)  

---


**ABSTRACT**  
Video restoration is a low-level vision task that seeks to restore clean, sharp videos from quality-degraded frames. One would use the temporal information from adjacent frames to make video restoration successful. Recently, the success of the Transformer has raised awareness in the computer-vision community. However, its self-attention mechanism requires much memory, which is unsuitable for high-resolution vision tasks like video restoration. In this paper, we propose ViStripformer (Video Stripformer), which utilizes spatio-temporal strip attention to catch long-range data correlations, consisting of intra-frame strip attention (Intra-SA) and inter-frame strip attention (Inter-SA) for extracting spatial and temporal information. It decomposes video frames into strip-shaped features in horizontal and vertical directions for Intra-SA and Inter-SA to address degradation patterns with various orientations and magnitudes. Besides, ViStripformer is an effective and efficient transformer architecture with much lower memory usage than the vanilla transformer. Extensive experiments show that the proposed model achieves superior results with fast inference time on video restoration tasks, including video deblurring, demoireing, and deraining.

{{</citation>}}


### (45/80) Revisiting Few-Shot Object Detection with Vision-Language Models (Anish Madan et al., 2023)

{{<citation>}}

Anish Madan, Neehar Peri, Shu Kong, Deva Ramanan. (2023)  
**Revisiting Few-Shot Object Detection with Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Language Model, Object Detection  
[Paper Link](http://arxiv.org/abs/2312.14494v1)  

---


**ABSTRACT**  
Few-shot object detection (FSOD) benchmarks have advanced techniques for detecting new categories with limited annotations. Existing benchmarks repurpose well-established datasets like COCO by partitioning categories into base and novel classes for pre-training and fine-tuning respectively. However, these benchmarks do not reflect how FSOD is deployed in practice. Rather than only pre-training on a small number of base categories, we argue that it is more practical to fine-tune a foundation model (e.g., a vision-language model (VLM) pre-trained on web-scale data) for a target domain. Surprisingly, we find that zero-shot inference from VLMs like GroundingDINO significantly outperforms the state-of-the-art (48.3 vs. 33.1 AP) on COCO. However, such zero-shot models can still be misaligned to target concepts of interest. For example, trailers on the web may be different from trailers in the context of autonomous vehicles. In this work, we propose Foundational FSOD, a new benchmark protocol that evaluates detectors pre-trained on any external datasets and fine-tuned on K-shots per target class. Further, we note that current FSOD benchmarks are actually federated datasets containing exhaustive annotations for each category on a subset of the data. We leverage this insight to propose simple strategies for fine-tuning VLMs with federated losses. We demonstrate the effectiveness of our approach on LVIS and nuImages, improving over prior work by 5.9 AP.

{{</citation>}}


### (46/80) Context Enhanced Transformer for Single Image Object Detection (Seungjun An et al., 2023)

{{<citation>}}

Seungjun An, Seonghoon Park, Gyeongnyeon Kim, Jeongyeol Baek, Byeongwon Lee, Seungryong Kim. (2023)  
**Context Enhanced Transformer for Single Image Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2312.14492v1)  

---


**ABSTRACT**  
With the increasing importance of video data in real-world applications, there is a rising need for efficient object detection methods that utilize temporal information. While existing video object detection (VOD) techniques employ various strategies to address this challenge, they typically depend on locally adjacent frames or randomly sampled images within a clip. Although recent Transformer-based VOD methods have shown promising results, their reliance on multiple inputs and additional network complexity to incorporate temporal information limits their practical applicability. In this paper, we propose a novel approach to single image object detection, called Context Enhanced TRansformer (CETR), by incorporating temporal context into DETR using a newly designed memory module. To efficiently store temporal information, we construct a class-wise memory that collects contextual information across data. Additionally, we present a classification-based sampling technique to selectively utilize the relevant memory for the current image. In the testing, We introduce a test-time memory adaptation method that updates individual memory functions by considering the test distribution. Experiments with CityCam and ImageNet VID datasets exhibit the efficiency of the framework on various video systems. The project page and code will be made available at: https://ku-cvlab.github.io/CETR.

{{</citation>}}


### (47/80) FM-OV3D: Foundation Model-based Cross-modal Knowledge Blending for Open-Vocabulary 3D Detection (Dongmei Zhang et al., 2023)

{{<citation>}}

Dongmei Zhang, Chang Li, Ray Zhang, Shenghao Xie, Wei Xue, Xiaodong Xie, Shanghang Zhang. (2023)  
**FM-OV3D: Foundation Model-based Cross-modal Knowledge Blending for Open-Vocabulary 3D Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.14465v1)  

---


**ABSTRACT**  
The superior performances of pre-trained foundation models in various visual tasks underscore their potential to enhance the 2D models' open-vocabulary ability. Existing methods explore analogous applications in the 3D space. However, most of them only center around knowledge extraction from singular foundation models, which limits the open-vocabulary ability of 3D models. We hypothesize that leveraging complementary pre-trained knowledge from various foundation models can improve knowledge transfer from 2D pre-trained visual language models to the 3D space. In this work, we propose FM-OV3D, a method of Foundation Model-based Cross-modal Knowledge Blending for Open-Vocabulary 3D Detection, which improves the open-vocabulary localization and recognition abilities of 3D model by blending knowledge from multiple pre-trained foundation models, achieving true open-vocabulary without facing constraints from original 3D datasets. Specifically, to learn the open-vocabulary 3D localization ability, we adopt the open-vocabulary localization knowledge of the Grounded-Segment-Anything model. For open-vocabulary 3D recognition ability, We leverage the knowledge of generative foundation models, including GPT-3 and Stable Diffusion models, and cross-modal discriminative models like CLIP. The experimental results on two popular benchmarks for open-vocabulary 3D object detection show that our model efficiently learns knowledge from multiple foundation models to enhance the open-vocabulary ability of the 3D model and successfully achieves state-of-the-art performance in open-vocabulary 3D object detection tasks. Code is released at https://github.com/dmzhang0425/FM-OV3D.git.

{{</citation>}}


### (48/80) Scalable 3D Reconstruction From Single Particle X-Ray Diffraction Images Based on Online Machine Learning (Jay Shenoy et al., 2023)

{{<citation>}}

Jay Shenoy, Axel Levy, Frédéric Poitevin, Gordon Wetzstein. (2023)  
**Scalable 3D Reconstruction From Single Particle X-Ray Diffraction Images Based on Online Machine Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, q-bio-BM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14432v1)  

---


**ABSTRACT**  
X-ray free-electron lasers (XFELs) offer unique capabilities for measuring the structure and dynamics of biomolecules, helping us understand the basic building blocks of life. Notably, high-repetition-rate XFELs enable single particle imaging (X-ray SPI) where individual, weakly scattering biomolecules are imaged under near-physiological conditions with the opportunity to access fleeting states that cannot be captured in cryogenic or crystallized conditions. Existing X-ray SPI reconstruction algorithms, which estimate the unknown orientation of a particle in each captured image as well as its shared 3D structure, are inadequate in handling the massive datasets generated by these emerging XFELs. Here, we introduce X-RAI, an online reconstruction framework that estimates the structure of a 3D macromolecule from large X-ray SPI datasets. X-RAI consists of a convolutional encoder, which amortizes pose estimation over large datasets, as well as a physics-based decoder, which employs an implicit neural representation to enable high-quality 3D reconstruction in an end-to-end, self-supervised manner. We demonstrate that X-RAI achieves state-of-the-art performance for small-scale datasets in simulation and challenging experimental settings and demonstrate its unprecedented ability to process large datasets containing millions of diffraction images in an online fashion. These abilities signify a paradigm shift in X-ray SPI towards real-time capture and reconstruction.

{{</citation>}}


### (49/80) GROOD: GRadient-aware Out-Of-Distribution detection in interpolated manifolds (Mostafa ElAraby et al., 2023)

{{<citation>}}

Mostafa ElAraby, Sabyasachi Sahoo, Yann Pequignot, Paul Novello, Liam Paull. (2023)  
**GROOD: GRadient-aware Out-Of-Distribution detection in interpolated manifolds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.14427v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) often fail silently with over-confident predictions on out-of-distribution (OOD) samples, posing risks in real-world deployments. Existing techniques predominantly emphasize either the feature representation space or the gradient norms computed with respect to DNN parameters, yet they overlook the intricate gradient distribution and the topology of classification regions. To address this gap, we introduce GRadient-aware Out-Of-Distribution detection in interpolated manifolds (GROOD), a novel framework that relies on the discriminative power of gradient space to distinguish between in-distribution (ID) and OOD samples. To build this space, GROOD relies on class prototypes together with a prototype that specifically captures OOD characteristics. Uniquely, our approach incorporates a targeted mix-up operation at an early intermediate layer of the DNN to refine the separation of gradient spaces between ID and OOD samples. We quantify OOD detection efficacy using the distance to the nearest neighbor gradients derived from the training set, yielding a robust OOD score. Experimental evaluations substantiate that the introduction of targeted input mix-upamplifies the separation between ID and OOD in the gradient space, yielding impressive results across diverse datasets. Notably, when benchmarked against ImageNet-1k, GROOD surpasses the established robustness of state-of-the-art baselines. Through this work, we establish the utility of leveraging gradient spaces and class prototypes for enhanced OOD detection for DNN in image classification.

{{</citation>}}


### (50/80) Unveiling Backbone Effects in CLIP: Exploring Representational Synergies and Variances (Cristian Rodriguez-Opazo et al., 2023)

{{<citation>}}

Cristian Rodriguez-Opazo, Edison Marrese-Taylor, Ehsan Abbasnejad, Hamed Damirchi, Ignacio M. Jara, Felipe Bravo-Marquez, Anton van den Hengel. (2023)  
**Unveiling Backbone Effects in CLIP: Exploring Representational Synergies and Variances**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.14400v1)  

---


**ABSTRACT**  
Contrastive Language-Image Pretraining (CLIP) stands out as a prominent method for image representation learning. Various neural architectures, spanning Transformer-based models like Vision Transformers (ViTs) to Convolutional Networks (ConvNets) like ResNets, are trained with CLIP and serve as universal backbones across diverse vision tasks. Despite utilizing the same data and training objectives, the effectiveness of representations learned by these architectures raises a critical question. Our investigation explores the differences in CLIP performance among these backbone architectures, revealing significant disparities in their classifications. Notably, normalizing these representations results in substantial performance variations. Our findings showcase a remarkable possible synergy between backbone predictions that could reach an improvement of over 20% through informed selection of the appropriate backbone. Moreover, we propose a simple, yet effective approach to combine predictions from multiple backbones, leading to a notable performance boost of up to 6.34\%. We will release the code for reproducing the results.

{{</citation>}}


## cs.SE (4)



### (51/80) Turbulence: Systematically and Automatically Testing Instruction-Tuned Large Language Models for Code (Shahin Honarvar et al., 2023)

{{<citation>}}

Shahin Honarvar, Mark van der Wilk, Alastair Donaldson. (2023)  
**Turbulence: Systematically and Automatically Testing Instruction-Tuned Large Language Models for Code**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.14856v1)  

---


**ABSTRACT**  
We present a method for systematically evaluating the correctness and robustness of instruction-tuned large language models (LLMs) for code generation via a new benchmark, Turbulence. Turbulence consists of a large set of natural language $\textit{question templates}$, each of which is a programming problem, parameterised so that it can be asked in many different forms. Each question template has an associated $\textit{test oracle}$ that judges whether a code solution returned by an LLM is correct. Thus, from a single question template, it is possible to ask an LLM a $\textit{neighbourhood}$ of very similar programming questions, and assess the correctness of the result returned for each question. This allows gaps in an LLM's code generation abilities to be identified, including $\textit{anomalies}$ where the LLM correctly solves $\textit{almost all}$ questions in a neighbourhood but fails for particular parameter instantiations. We present experiments against five LLMs from OpenAI, Cohere and Meta, each at two temperature configurations. Our findings show that, across the board, Turbulence is able to reveal gaps in LLM reasoning ability. This goes beyond merely highlighting that LLMs sometimes produce wrong code (which is no surprise): by systematically identifying cases where LLMs are able to solve some problems in a neighbourhood but do not manage to generalise to solve the whole neighbourhood, our method is effective at highlighting $\textit{robustness}$ issues. We present data and examples that shed light on the kinds of mistakes that LLMs make when they return incorrect code results.

{{</citation>}}


### (52/80) An Empirical Study on Compliance with Ranking Transparency in the Software Documentation of EU Online Platforms (Francesco Sovrano et al., 2023)

{{<citation>}}

Francesco Sovrano, Michaël Lognoul, Alberto Bacchelli. (2023)  
**An Empirical Study on Compliance with Ranking Transparency in the Software Documentation of EU Online Platforms**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Amazon, ChatGPT, GPT, Google  
[Paper Link](http://arxiv.org/abs/2312.14794v1)  

---


**ABSTRACT**  
Compliance with the European Union's Platform-to-Business (P2B) Regulation is challenging for online platforms, and assessing their compliance can be difficult for public authorities. This is partly due to the lack of automated tools for assessing the information (e.g., software documentation) platforms provide concerning ranking transparency. Our study tackles this issue in two ways. First, we empirically evaluate the compliance of six major platforms (Amazon, Bing, Booking, Google, Tripadvisor, and Yahoo), revealing substantial differences in their documentation. Second, we introduce and test automated compliance assessment tools based on ChatGPT and information retrieval technology. These tools are evaluated against human judgments, showing promising results as reliable proxies for compliance assessments. Our findings could help enhance regulatory compliance and align with the United Nations Sustainable Development Goal 10.3, which seeks to reduce inequality, including business disparities, on these platforms.

{{</citation>}}


### (53/80) ROS package search for robot software development: a knowledge graph-based approach (Shuo Wang et al., 2023)

{{<citation>}}

Shuo Wang, Xinjun Mao, Shuo Yang, Menghan Wu, Zhang Zhang. (2023)  
**ROS package search for robot software development: a knowledge graph-based approach**  

---
Primary Category: cs.SE  
Categories: cs-RO, cs-SE, cs.SE  
Keywords: BERT, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2312.14781v1)  

---


**ABSTRACT**  
ROS (Robot Operating System) packages have become increasingly popular as a type of software artifact that can be effectively reused in robotic software development. Indeed, finding suitable ROS packages that closely match the software's functional requirements from the vast number of available packages is a nontrivial task using current search methods. The traditional search methods for ROS packages often involve inputting keywords related to robotic tasks into general-purpose search engines or code hosting platforms to obtain approximate results of all potentially suitable ROS packages. However, the accuracy of these search methods remains relatively low because the task-related keywords may not precisely match the functionalities offered by the ROS packages. To improve the search accuracy of ROS packages, this paper presents a novel semantic-based search approach that relies on the semantic-level ROS Package Knowledge Graph (RPKG) to automatically retrieve the most suitable ROS packages. Firstly, to construct the RPKG, we employ multi-dimensional feature extraction techniques to extract semantic concepts from the dataset of ROS package text descriptions. The semantic features extracted from this process result in a substantial number of entities and relationships. Subsequently, we create a robot domain-specific small corpus and further fine-tune a pre-trained language model, BERT-ROS, to generate embeddings that effectively represent the semantics of the extracted features. These embeddings play a crucial role in facilitating semantic-level understanding and comparisons during the ROS package search process within the RPKG. Secondly, we introduce a novel semantic matching-based search algorithm that incorporates the weighted similarities of multiple features from user search queries, which searches out more accurate ROS packages than the traditional keyword search method.

{{</citation>}}


### (54/80) Enhancing Text-to-SQL Translation for Financial System Design (Yewei Song et al., 2023)

{{<citation>}}

Yewei Song, Saad Ezzini, Xunzhu Tang, Cedric Lothritz, Jacques Klein, Tegawendé Bissyandé, Andrey Boytsov, Ulrick Ble, Anne Goujon. (2023)  
**Enhancing Text-to-SQL Translation for Financial System Design**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Financial, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.14725v1)  

---


**ABSTRACT**  
Text-to-SQL, the task of translating natural language questions into SQL queries, is part of various business processes. Its automation, which is an emerging challenge, will empower software practitioners to seamlessly interact with relational databases using natural language, thereby bridging the gap between business needs and software capabilities. In this paper, we consider Large Language Models (LLMs), which have achieved state of the art for various NLP tasks. Specifically, we benchmark Text-to-SQL performance, the evaluation methodologies, as well as input optimization (e.g., prompting). In light of the empirical observations that we have made, we propose two novel metrics that were designed to adequately measure the similarity between SQL queries. Overall, we share with the community various findings, notably on how to select the right LLM on Text-to-SQL tasks. We further demonstrate that a tree-based edit distance constitutes a reliable metric for assessing the similarity between generated SQL queries and the oracle for benchmarking Text2SQL approaches. This metric is important as it relieves researchers from the need to perform computationally expensive experiments such as executing generated queries as done in prior works. Our work implements financial domain use cases and, therefore contributes to the advancement of Text2SQL systems and their practical adoption in this domain.

{{</citation>}}


## physics.bio-ph (1)



### (55/80) Large Scale Traning of Graph Neural Networks for Optimal Markov-Chain Partitioning Using the Kemeny Constant (Sam Alexander Martino et al., 2023)

{{<citation>}}

Sam Alexander Martino, João Morado, Chenghao Li, Zhenghao Lu, Edina Rosta. (2023)  
**Large Scale Traning of Graph Neural Networks for Optimal Markov-Chain Partitioning Using the Kemeny Constant**  

---
Primary Category: physics.bio-ph  
Categories: cs-LG, physics-bio-ph, physics-comp-ph, physics.bio-ph  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.14847v1)  

---


**ABSTRACT**  
Traditional clustering algorithms often struggle to capture the complex relationships within graphs and generalise to arbitrary clustering criteria. The emergence of graph neural networks (GNNs) as a powerful framework for learning representations of graph data provides new approaches to solving the problem. Previous work has shown GNNs to be capable of proposing partitionings using a variety of criteria, however, these approaches have not yet been extended to work on Markov chains or kinetic networks. These arise frequently in the study of molecular systems and are of particular interest to the biochemical modelling community. In this work, we propose several GNN-based architectures to tackle the graph partitioning problem for Markov Chains described as kinetic networks. This approach aims to minimize how much a proposed partitioning changes the Kemeny constant. We propose using an encoder-decoder architecture and show how simple GraphSAGE-based GNNs with linear layers can outperform much larger and more expressive attention-based models in this context. As a proof of concept, we first demonstrate the method's ability to cluster randomly connected graphs. We also use a linear chain architecture corresponding to a 1D free energy profile as our kinetic network. Subsequently, we demonstrate the effectiveness of our method through experiments on a data set derived from molecular dynamics. We compare the performance of our method to other partitioning techniques such as PCCA+. We explore the importance of feature and hyperparameter selection and propose a general strategy for large-scale parallel training of GNNs for discovering optimal graph partitionings.

{{</citation>}}


## cs.LO (2)



### (56/80) Asynchronous Composition of LTL Properties over Infinite and Finite Traces (Alberto Bombardelli et al., 2023)

{{<citation>}}

Alberto Bombardelli, Stefano Tonetta. (2023)  
**Asynchronous Composition of LTL Properties over Infinite and Finite Traces**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs.LO  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2312.14831v1)  

---


**ABSTRACT**  
The verification of asynchronous software components poses significant challenges due to the way components interleave and exchange input/output data concurrently. Compositional strategies aim to address this by separating the task of verifying individual components on local properties from the task of combining them to achieve global properties. This paper concentrates on employing symbolic model checking techniques to verify properties specified in Linear-time Temporal Logic (LTL) on asynchronous software components that interact through data ports. Unlike event-based composition, local properties can now impose constraints on input from other components, increasing the complexity of their composition. We consider both the standard semantics over infinite traces as well as the truncated semantics over finite traces to allow scheduling components only finitely many times.   We propose a novel LTL rewriting approach, which converts a local property into a global one while considering the interleaving of infinite or finite execution traces of components. We prove the semantic equivalence of local properties and their rewritten version projected on the local symbols. The rewriting is also optimized to reduce formula size and to leave it unchanged when the temporal property is stutter invariant. These methods have been integrated into the OCRA tool, as part of the contract refinement verification suite. Finally, the different composition approaches were compared through an experimental evaluation that covers various types of specifications.

{{</citation>}}


### (57/80) Structure-Guided Automated Reasoning (Max Bannach et al., 2023)

{{<citation>}}

Max Bannach, Markus Hecher. (2023)  
**Structure-Guided Automated Reasoning**  

---
Primary Category: cs.LO  
Categories: F-4-1; F-2-0, cs-CC, cs-LO, cs.LO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.14620v1)  

---


**ABSTRACT**  
Algorithmic meta-theorems state that problems that can be formalized in a fixed logic can be solved efficiently on classes of structures with certain properties. A prominent example is Courcelle's Theorem, which states that all problems expressible in monadic second-order logic can be solved efficiently on structures of small treewidth. Such theorems are usually proven by a generic algorithm for the model-checking problem of the given logic, which is often complex and rarely leads to highly efficient solutions. Alternatively, we can solve the model-checking problem by grounding the given logic to propositional logic, for which dedicated solvers are available. Such encodings will, however, usually not preserve the input's treewidth.   This paper investigates whether all problems definable in monadic second-order logic can efficiently be encoded into SAT such that the input's treewidth bounds the treewidth of the resulting formula. We answer this in the affirmative and, hence, provide an alternative proof of Courcelle's Theorem. Our technique can naturally be extended: There are treewidth-aware reductions from the optimization version of Courcelle's Theorem to MaxSAT and from the counting version of the theorem to #SAT. By using encodings to SAT, we obtain, ignoring polynomial factors, the same running time for the model-checking problem as we would with dedicated algorithms. We complement our upper bounds with new lower bounds based on ETH; and we show that the block size of the input's formula and the treewidth of the input's structure are tightly linked. We also provide matching upper and lower bounds for a fragment of guarded MSO, only using SAT-based techniques.

{{</citation>}}


## cs.CY (2)



### (58/80) Use large language models to promote equity (Emma Pierson et al., 2023)

{{<citation>}}

Emma Pierson, Divya Shanmugam, Rajiv Movva, Jon Kleinberg, Monica Agrawal, Mark Dredze, Kadija Ferryman, Judy Wawira Gichoya, Dan Jurafsky, Pang Wei Koh, Karen Levy, Sendhil Mullainathan, Ziad Obermeyer, Harini Suresh, Keyon Vafa. (2023)  
**Use large language models to promote equity**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14804v1)  

---


**ABSTRACT**  
Advances in large language models (LLMs) have driven an explosion of interest about their societal impacts. Much of the discourse around how they will impact social equity has been cautionary or negative, focusing on questions like "how might LLMs be biased and how would we mitigate those biases?" This is a vital discussion: the ways in which AI generally, and LLMs specifically, can entrench biases have been well-documented. But equally vital, and much less discussed, is the more opportunity-focused counterpoint: "what promising applications do LLMs enable that could promote equity?" If LLMs are to enable a more equitable world, it is not enough just to play defense against their biases and failure modes. We must also go on offense, applying them positively to equity-enhancing use cases to increase opportunities for underserved groups and reduce societal discrimination. There are many choices which determine the impact of AI, and a fundamental choice very early in the pipeline is the problems we choose to apply it to. If we focus only later in the pipeline -- making LLMs marginally more fair as they facilitate use cases which intrinsically entrench power -- we will miss an important opportunity to guide them to equitable impacts. Here, we highlight the emerging potential of LLMs to promote equity by presenting four newly possible, promising research directions, while keeping risks and cautionary points in clear view.

{{</citation>}}


### (59/80) Lost in the Logistical Funhouse: Speculative Design as Synthetic Media Enterprise (Zoe Horn et al., 2023)

{{<citation>}}

Zoe Horn, Liam Magee, Anna Munster. (2023)  
**Lost in the Logistical Funhouse: Speculative Design as Synthetic Media Enterprise**  

---
Primary Category: cs.CY  
Categories: K-4-2; K-4-3; J-5, cs-CY, cs.CY  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2312.14424v1)  

---


**ABSTRACT**  
From the deployment of chatbots as procurement negotiators by corporations such as Walmart to autonomous agents providing 'differentiated chat' for managing overbooked flights, synthetic media are making the world of logistics their 'natural' habitat. Here the coordination of commodities, parts and labour design the problems and produce the training sets from which 'solutions' can be synthesised. But to what extent might synthetic media, surfacing via proto-platforms such as MidJourney and OpenAI and apps such as Eleven Labs and D:ID, be understood as logistical media? This paper details synthetic media experiments with 'ChatFOS', a GPT-based bot tasked with developing a logistics design business. Using its prompt-generated media outputs, we assemble a simulation and parody of AI's emerging functionalities within logistical worlds. In the process, and with clunky 'human-in-the-loop' stitching, we illustrate how large language models become media routers or switches, governing production of image prompts, website code, promotional copy, and investor pitch scenarios. Together these elements become links chained together in media ensembles such as the corporate website or the promotional video, fuelling the fictive logistics visualisation company we have 'founded'. The processes and methods of producing speculative scenarios via ChatFOS lead us to consider how synthetic media might be re-positioned as logistical media. Our experiments probe the ways in which the media of logistics and the logistics of media are increasingly enfolded. We ask: what can a (practice-based) articulation of this double-becoming of logistics and synthetic mediality tell us about the politics and aesthetics of contemporary computation and capital?

{{</citation>}}


## eess.IV (1)



### (60/80) SCUNet++: Assessment of Pulmonary Embolism CT Image Segmentation Leveraging Swin-UNet and CNN Bottleneck Hybrid Architecture with Multi-Fusion Dense Skip Connection (Yifei Chen et al., 2023)

{{<citation>}}

Yifei Chen, Binfeng Zou, Zhaoxin Guo, Yiyu Huang, Yifan Huang, Feiwei Qin, Qinhai Li, Changmiao Wang. (2023)  
**SCUNet++: Assessment of Pulmonary Embolism CT Image Segmentation Leveraging Swin-UNet and CNN Bottleneck Hybrid Architecture with Multi-Fusion Dense Skip Connection**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.14705v1)  

---


**ABSTRACT**  
Pulmonary embolism (PE) is a prevalent lung disease that can lead to right ventricular hypertrophy and failure in severe cases, ranking second in severity only to myocardial infarction and sudden death. Pulmonary artery CT angiography (CTPA) is a widely used diagnostic method for PE. However, PE detection presents challenges in clinical practice due to limitations in imaging technology. CTPA can produce noises similar to PE, making confirmation of its presence time-consuming and prone to overdiagnosis. Nevertheless, the traditional segmentation method of PE can not fully consider the hierarchical structure of features, local and global spatial features of PE CT images. In this paper, we propose an automatic PE segmentation method called SCUNet++ (Swin Conv UNet++). This method incorporates multiple fusion dense skip connections between the encoder and decoder, utilizing the Swin Transformer as the encoder. And fuses features of different scales in the decoder subnetwork to compensate for spatial information loss caused by the inevitable downsampling in Swin-UNet or other state-of-the-art methods, effectively solving the above problem. We provide a theoretical analysis of this method in detail and validate it on publicly available PE CT image datasets FUMPE and CAD-PE. The experimental results indicate that our proposed method achieved a Dice similarity coefficient (DSC) of 83.47% and a Hausdorff distance 95th percentile (HD95) of 3.83 on the FUMPE dataset, as well as a DSC of 83.42% and an HD95 of 5.10 on the CAD-PE dataset. These findings demonstrate that our method exhibits strong performance in PE segmentation tasks, potentially enhancing the accuracy of automatic segmentation of PE and providing a powerful diagnostic tool for clinical physicians. Our source code and new FUMPE dataset are available at https://github.com/JustlfC03/SCUNet-plusplus.

{{</citation>}}


## cs.MM (1)



### (61/80) Token-Level Contrastive Learning with Modality-Aware Prompting for Multimodal Intent Recognition (Qianrui Zhou et al., 2023)

{{<citation>}}

Qianrui Zhou, Hua Xu, Hao Li, Hanlei Zhang, Xiaohan Zhang, Yifan Wang, Kai Gao. (2023)  
**Token-Level Contrastive Learning with Modality-Aware Prompting for Multimodal Intent Recognition**  

---
Primary Category: cs.MM  
Categories: cs-LG, cs-MM, cs.MM  
Keywords: Contrastive Learning, Intent Recognition  
[Paper Link](http://arxiv.org/abs/2312.14667v1)  

---


**ABSTRACT**  
Multimodal intent recognition aims to leverage diverse modalities such as expressions, body movements and tone of speech to comprehend user's intent, constituting a critical task for understanding human language and behavior in real-world multimodal scenarios. Nevertheless, the majority of existing methods ignore potential correlations among different modalities and own limitations in effectively learning semantic features from nonverbal modalities. In this paper, we introduce a token-level contrastive learning method with modality-aware prompting (TCL-MAP) to address the above challenges. To establish an optimal multimodal semantic environment for text modality, we develop a modality-aware prompting module (MAP), which effectively aligns and fuses features from text, video and audio modalities with similarity-based modality alignment and cross-modality attention mechanism. Based on the modality-aware prompt and ground truth labels, the proposed token-level contrastive learning framework (TCL) constructs augmented samples and employs NT-Xent loss on the label token. Specifically, TCL capitalizes on the optimal textual semantic insights derived from intent labels to guide the learning processes of other modalities in return. Extensive experiments show that our method achieves remarkable improvements compared to state-of-the-art methods. Additionally, ablation analyses demonstrate the superiority of the modality-aware prompt over the handcrafted prompt, which holds substantial significance for multimodal prompt learning. The codes are released at https://github.com/thuiar/TCL-MAP.

{{</citation>}}


## cs.DC (2)



### (62/80) Pub/Sub Message Brokers for GenAI (Alaa Saleh et al., 2023)

{{<citation>}}

Alaa Saleh, Susanna Pirttikangas, Lauri Lovén. (2023)  
**Pub/Sub Message Brokers for GenAI**  

---
Primary Category: cs.DC  
Categories: C-2-4; I-2-11; I-2-7, cs-AI, cs-DC, cs-LG, cs-NI, cs.DC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.14647v1)  

---


**ABSTRACT**  
In today's digital world, Generative Artificial Intelligence (GenAI) such as Large Language Models (LLMs) is becoming increasingly prevalent, extending its reach across diverse applications. This surge in adoption has sparked a significant increase in demand for data-centric GenAI models, highlighting the necessity for robust data communication infrastructures. Central to this need are message brokers, which serve as essential channels for data transfer within various system components. This survey aims to delve into a comprehensive analysis of traditional and modern message brokers, offering a comparative study of prevalent platforms. Our study considers numerous criteria including, but not limited to, open-source availability, integrated monitoring tools, message prioritization mechanisms, capabilities for parallel processing, reliability, distribution and clustering functionalities, authentication processes, data persistence strategies, fault tolerance, and scalability. Furthermore, we explore the intrinsic constraints that the design and operation of each message broker might impose, recognizing that these limitations are crucial in understanding their real-world applicability. We then leverage these insights to propose a sophisticated message broker framework -- one designed with the adaptability and robustness necessary to meet the evolving requisites of GenAI applications. Finally, this study examines the enhancement of message broker mechanisms specifically for GenAI contexts, emphasizing the criticality of developing a versatile message broker framework. Such a framework would be poised for quick adaptation, catering to the dynamic and growing demands of GenAI in the foreseeable future. Through this dual-pronged approach, we intend to contribute a foundational compendium that can guide future innovations and infrastructural advancements in the realm of GenAI data communication.

{{</citation>}}


### (63/80) Generative AI Beyond LLMs: System Implications of Multi-Modal Generation (Alicia Golden et al., 2023)

{{<citation>}}

Alicia Golden, Samuel Hsia, Fei Sun, Bilge Acun, Basil Hosmer, Yejin Lee, Zachary DeVito, Jeff Johnson, Gu-Yeon Wei, David Brooks, Carole-Jean Wu. (2023)  
**Generative AI Beyond LLMs: System Implications of Multi-Modal Generation**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-LG, cs-MM, cs.DC  
Keywords: AI, Attention, Generative AI, Transformer  
[Paper Link](http://arxiv.org/abs/2312.14385v1)  

---


**ABSTRACT**  
As the development of large-scale Generative AI models evolve beyond text (1D) generation to include image (2D) and video (3D) generation, processing spatial and temporal information presents unique challenges to quality, performance, and efficiency. We present the first work towards understanding this new system design space for multi-modal text-to-image (TTI) and text-to-video (TTV) generation models. Current model architecture designs are bifurcated into 2 categories: Diffusion- and Transformer-based models. Our systematic performance characterization on a suite of eight representative TTI/TTV models shows that after state-of-the-art optimization techniques such as Flash Attention are applied, Convolution accounts for up to 44% of execution time for Diffusion-based TTI models, while Linear layers consume up to 49% of execution time for Transformer-based models. We additionally observe that Diffusion-based TTI models resemble the Prefill stage of LLM inference, and benefit from 1.1-2.5x greater speedup from Flash Attention than Transformer-based TTI models that resemble the Decode phase. Since optimizations designed for LLMs do not map directly onto TTI/TTV models, we must conduct a thorough characterization of these workloads to gain insights for new optimization opportunities. In doing so, we define sequence length in the context of TTI/TTV models and observe sequence length can vary up to 4x in Diffusion model inference. We additionally observe temporal aspects of TTV workloads pose unique system bottlenecks, with Temporal Attention accounting for over 60% of total Attention time. Overall, our in-depth system performance characterization is a critical first step towards designing efficient and deployable systems for emerging TTI/TTV workloads.

{{</citation>}}


## cs.CR (4)



### (64/80) Evaluating the Security and Privacy Risk Postures of Virtual Assistants (Borna Kalhor et al., 2023)

{{<citation>}}

Borna Kalhor, Sanchari Das. (2023)  
**Evaluating the Security and Privacy Risk Postures of Virtual Assistants**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Google, Security  
[Paper Link](http://arxiv.org/abs/2312.14633v1)  

---


**ABSTRACT**  
Virtual assistants (VAs) have seen increased use in recent years due to their ease of use for daily tasks. Despite their growing prevalence, their security and privacy implications are still not well understood. To address this gap, we conducted a study to evaluate the security and privacy postures of eight widely used voice assistants: Alexa, Braina, Cortana, Google Assistant, Kalliope, Mycroft, Hound, and Extreme. We used three vulnerability testing tools, AndroBugs, RiskInDroid, and MobSF, to assess the security and privacy of these VAs. Our analysis focused on five areas: code, access control, tracking, binary analysis, and sensitive data confidentiality. The results revealed that these VAs are vulnerable to a range of security threats, including not validating SSL certificates, executing raw SQL queries, and using a weak mode of the AES algorithm. These vulnerabilities could allow malicious actors to gain unauthorized access to users' personal information. This study is a first step toward understanding the risks associated with these technologies and provides a foundation for future research to develop more secure and privacy-respecting VAs.

{{</citation>}}


### (65/80) ChatGPT, Llama, can you write my report? An experiment on assisted digital forensics reports written using (Local) Large Language Models (Gaëtan Michelet et al., 2023)

{{<citation>}}

Gaëtan Michelet, Frank Breitinger. (2023)  
**ChatGPT, Llama, can you write my report? An experiment on assisted digital forensics reports written using (Local) Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CY, cs.CR  
Keywords: AI, ChatGPT, GPT, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.14607v1)  

---


**ABSTRACT**  
Generative AIs, especially Large Language Models (LLMs) such as ChatGPT or Llama, have advanced significantly, positioning them as valuable tools for digital forensics. While initial studies have explored the potential of ChatGPT in the context of investigations, the question of to what extent LLMs can assist the forensic report writing process remains unresolved. To answer the question, this article first examines forensic reports with the goal of generalization (e.g., finding the `average structure' of a report). We then evaluate the strengths and limitations of LLMs for generating the different parts of the forensic report using a case study. This work thus provides valuable insights into the automation of report writing, a critical facet of digital forensics investigations. We conclude that combined with thorough proofreading and corrections, LLMs may assist practitioners during the report writing process but at this point cannot replace them.

{{</citation>}}


### (66/80) MetaAID 2.5: A Secure Framework for Developing Metaverse Applications via Large Language Models (Hongyin Zhu, 2023)

{{<citation>}}

Hongyin Zhu. (2023)  
**MetaAID 2.5: A Secure Framework for Developing Metaverse Applications via Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs-CY, cs.CR  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.14480v1)  

---


**ABSTRACT**  
Large language models (LLMs) are increasingly being used in Metaverse environments to generate dynamic and realistic content and to control the behavior of non-player characters (NPCs). However, the cybersecurity concerns associated with LLMs have become increasingly prominent. Previous research has primarily focused on patching system vulnerabilities to enhance cybersecurity, but these approaches are not well-suited to the Metaverse, where the virtual space is more complex, LLMs are vulnerable, and ethical user interaction is critical. Moreover, the scope of cybersecurity in the Metaverse is expected to expand significantly. This paper proposes a method for enhancing cybersecurity through the simulation of user interaction with LLMs. Our goal is to educate users and strengthen their defense capabilities through exposure to a comprehensive simulation system. This system includes extensive Metaverse cybersecurity Q&A and attack simulation scenarios. By engaging with these, users will improve their ability to recognize and withstand risks. Additionally, to address the ethical implications of user input, we propose using LLMs as evaluators to assess user content across five dimensions. We further adapt the models through vocabulary expansion training to better understand personalized inputs and emoticons. We conduct experiments on multiple LLMs and find that our approach is effective.

{{</citation>}}


### (67/80) A Review on Searchable Encryption Functionality and the Evaluation of Homomorphic Encryption (Brian Kishiyama et al., 2023)

{{<citation>}}

Brian Kishiyama, Izzat Alsmadi. (2023)  
**A Review on Searchable Encryption Functionality and the Evaluation of Homomorphic Encryption**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Amazon, Azure, Google, Microsoft  
[Paper Link](http://arxiv.org/abs/2312.14434v1)  

---


**ABSTRACT**  
Cloud Service Providers, such as Google Cloud Platform, Microsoft Azure, or Amazon Web Services, offer continuously evolving cloud services. It is a growing industry. Businesses, such as Netflix and PayPal, rely on the Cloud for data storage, computing power, and other services. For businesses, the cloud reduces costs, provides flexibility, and allows for growth. However, there are security and privacy concerns regarding the Cloud. Because Cloud services are accessed through the internet, hackers and attackers could possibly access the servers from anywhere. To protect data in the Cloud, it should be encrypted before it is uploaded, it should be protected in storage and also in transit. On the other hand, data owners may need to access their encrypted data. It may also need to be altered, updated, deleted, read, searched, or shared with others. If data is decrypted in the Cloud, sensitive data is exposed and could be exposed and misused. One solution is to leave the data in its encrypted form and use Searchable Encryption (SE) which operates on encrypted data. The functionality of SE has improved since its inception and research continues to explore ways to improve SE. This paper reviews the functionality of Searchable Encryption, mostly related to Cloud services, in the years 2019 to 2023, and evaluates one of its schemes, Fully Homomorphic Encryption. Overall, it seems that research is at the point where SE efficiency is increased as multiple functionalities are aggregated and tested.

{{</citation>}}


## eess.AS (1)



### (68/80) BLSTM-Based Confidence Estimation for End-to-End Speech Recognition (Atsunori Ogawa et al., 2023)

{{<citation>}}

Atsunori Ogawa, Naohiro Tawara, Takatomo Kano, Marc Delcroix. (2023)  
**BLSTM-Based Confidence Estimation for End-to-End Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-CL, eess-AS, eess.AS  
Keywords: LSTM, Speech Recognition, Transformer  
[Paper Link](http://arxiv.org/abs/2312.14609v1)  

---


**ABSTRACT**  
Confidence estimation, in which we estimate the reliability of each recognized token (e.g., word, sub-word, and character) in automatic speech recognition (ASR) hypotheses and detect incorrectly recognized tokens, is an important function for developing ASR applications. In this study, we perform confidence estimation for end-to-end (E2E) ASR hypotheses. Recent E2E ASR systems show high performance (e.g., around 5% token error rates) for various ASR tasks. In such situations, confidence estimation becomes difficult since we need to detect infrequent incorrect tokens from mostly correct token sequences. To tackle this imbalanced dataset problem, we employ a bidirectional long short-term memory (BLSTM)-based model as a strong binary-class (correct/incorrect) sequence labeler that is trained with a class balancing objective. We experimentally confirmed that, by utilizing several types of ASR decoding scores as its auxiliary features, the model steadily shows high confidence estimation performance under highly imbalanced settings. We also confirmed that the BLSTM-based model outperforms Transformer-based confidence estimation models, which greatly underestimate incorrect tokens.

{{</citation>}}


## econ.GN (1)



### (69/80) The Economics of Human Oversight: How Norms and Incentives Affect Costs and Performance of AI Workers (Johann Laux et al., 2023)

{{<citation>}}

Johann Laux, Fabian Stephany, Alice Liefgreen. (2023)  
**The Economics of Human Oversight: How Norms and Incentives Affect Costs and Performance of AI Workers**  

---
Primary Category: econ.GN  
Categories: cs-AI, econ-GN, econ.GN, q-fin-EC, stat-AP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.14565v1)  

---


**ABSTRACT**  
The global surge in AI applications is transforming industries, leading to displacement and complementation of existing jobs, while also giving rise to new employment opportunities. Human oversight of AI is an emerging task in which human workers interact with an AI model to improve its performance, safety, and compliance with normative principles. Data annotation, encompassing the labelling of images or annotating of texts, serves as a critical human oversight process, as the quality of a dataset directly influences the quality of AI models trained on it. Therefore, the efficiency of human oversight work stands as an important competitive advantage for AI developers. This paper delves into the foundational economics of human oversight, with a specific focus on the impact of norm design and monetary incentives on data quality and costs. An experimental study involving 307 data annotators examines six groups with varying task instructions (norms) and monetary incentives. Results reveal that annotators provided with clear rules exhibit higher accuracy rates, outperforming those with vague standards by 14%. Similarly, annotators receiving an additional monetary incentive perform significantly better, with the highest accuracy rate recorded in the group working with both clear rules and incentives (87.5% accuracy). However, both groups require more time to complete tasks, with a 31% increase in average task completion time compared to those working with standards and no incentives. These empirical findings underscore the trade-off between data quality and efficiency in data curation, shedding light on the nuanced impact of norm design and incentives on the economics of AI development. The paper contributes experimental insights to discussions on the economical, ethical, and legal considerations of AI technologies.

{{</citation>}}


## cs.NE (1)



### (70/80) Adaptive Differential Evolution with Diversification: Addressing Optimization Challenges (Sarit Maitra, 2023)

{{<citation>}}

Sarit Maitra. (2023)  
**Adaptive Differential Evolution with Diversification: Addressing Optimization Challenges**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs-PF, cs.NE  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2312.14464v1)  

---


**ABSTRACT**  
The existing variants of the Differential Evolution (DE) algorithm come with certain limitations, such as poor local search and susceptibility to premature convergence. This study introduces Adaptive Differential Evolution with Diversification (ADED), a method that dynamically modifies the neighborhood structure by evaluating the trial solutions' fitness. Developed to work with both convex and nonconvex objective functions, ADED is validated with 22 benchmark functions, including Rosenbrock, Rastrigin, Ackley, and DeVilliers-Glasser02. The development is carried out in Google Cloud using Jupyter Notebook and Python v3.10.12, with additional testing conducted on the multi-objective benchmark ZDT test suite. ADED distinguishes itself with its adaptive and diverse approach, which includes adaptive mutation and crossover-rates, diverse mutation tactics, diversification measurements, local search mechanisms, and convergence monitoring. The unique combination of these features collectively enhances ADED's effectiveness in navigating complex and diverse landscapes, positioning it as a promising tool for addressing challenges in both single- and multi-objective optimization scenarios.

{{</citation>}}


## eess.SY (1)



### (71/80) Dynamic Programming-based Approximate Optimal Control for Model-Based Reinforcement Learning (Prakash Mallick et al., 2023)

{{<citation>}}

Prakash Mallick, Zhiyong Chen. (2023)  
**Dynamic Programming-based Approximate Optimal Control for Model-Based Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14463v1)  

---


**ABSTRACT**  
This article proposes an improved trajectory optimization approach for stochastic optimal control of dynamical systems affected by measurement noise by combining optimal control with maximum likelihood techniques to improve the reduction of the cumulative cost-to-go. A modified optimization objective function that incorporates dynamic programming-based controller design is presented to handle the noise in the system and sensors. Empirical results demonstrate the effectiveness of the approach in reducing stochasticity and allowing for an intermediate step to switch optimization that can allow an efficient balance of exploration and exploitation mechanism for complex tasks by constraining policy parameters to parameters obtained as a result of this improved optimization. This research study also includes theoretical work on the uniqueness of control parameter estimates and also leverages a structure of the likelihood function which has an established theoretical guarantees. Furthermore, a theoretical result is also explored that bridge the gap between the proposed optimization objective function and existing information theory (relative entropy) and optimal control dualities.

{{</citation>}}


## cs.HC (1)



### (72/80) Multiagent Copilot Approach for Shared Autonomy between Human EEG and TD3 Deep Reinforcement Learning (Chun-Ren Phang et al., 2023)

{{<citation>}}

Chun-Ren Phang, Akimasa Hirata. (2023)  
**Multiagent Copilot Approach for Shared Autonomy between Human EEG and TD3 Deep Reinforcement Learning**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-LG, cs.HC, eess-SP, q-bio-NC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14458v1)  

---


**ABSTRACT**  
Deep reinforcement learning (RL) algorithms enable the development of fully autonomous agents that can interact with the environment. Brain-computer interface (BCI) systems decipher human implicit brain signals regardless of the explicit environment. In this study, we integrated deep RL and BCI to improve beneficial human interventions in autonomous systems and the performance in decoding brain activities by considering environmental factors. Shared autonomy was allowed between the action command decoded from the electroencephalography (EEG) of the human agent and the action generated from the twin delayed DDPG (TD3) agent for a given environment. Our proposed copilot control scheme with a full blocker (Co-FB) significantly outperformed the individual EEG (EEG-NB) or TD3 control. The Co-FB model achieved a higher target approaching score, lower failure rate, and lower human workload than the EEG-NB model. The Co-FB control scheme had a higher invisible target score and level of allowed human intervention than the TD3 model. We also proposed a disparity d-index to evaluate the effect of contradicting agent decisions on the control accuracy and authority of the copilot model. We found a significant correlation between the control authority of the TD3 agent and the performance improvement of human EEG classification with respect to the d-index. We also observed that shifting control authority to the TD3 agent improved performance when BCI decoding was not optimal. These findings indicate that the copilot system can effectively handle complex environments and that BCI performance can be improved by considering environmental factors. Future work should employ continuous action space and different multi-agent approaches to evaluate copilot performance.

{{</citation>}}


## cs.RO (4)



### (73/80) QUAR-VLA: Vision-Language-Action Model for Quadruped Robots (Pengxiang Ding et al., 2023)

{{<citation>}}

Pengxiang Ding, Han Zhao, Zhitao Wang, Zhenyu Wei, Shangke Lyu, Donglin Wang. (2023)  
**QUAR-VLA: Vision-Language-Action Model for Quadruped Robots**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.14457v1)  

---


**ABSTRACT**  
The important manifestation of robot intelligence is the ability to naturally interact and autonomously make decisions. Traditional approaches to robot control often compartmentalize perception, planning, and decision-making, simplifying system design but limiting the synergy between different information streams. This compartmentalization poses challenges in achieving seamless autonomous reasoning, decision-making, and action execution. To address these limitations, a novel paradigm, named Vision-Language-Action tasks for QUAdruped Robots (QUAR-VLA), has been introduced in this paper. This approach tightly integrates visual information and instructions to generate executable actions, effectively merging perception, planning, and decision-making. The central idea is to elevate the overall intelligence of the robot. Within this framework, a notable challenge lies in aligning fine-grained instructions with visual perception information. This emphasizes the complexity involved in ensuring that the robot accurately interprets and acts upon detailed instructions in harmony with its visual observations. Consequently, we propose QUAdruped Robotic Transformer (QUART), a family of VLA models to integrate visual information and instructions from diverse modalities as input and generates executable actions for real-world robots and present QUAdruped Robot Dataset (QUARD), a large-scale multi-task dataset including navigation, complex terrain locomotion, and whole-body manipulation tasks for training QUART models. Our extensive evaluation (4000 evaluation trials) shows that our approach leads to performant robotic policies and enables QUART to obtain a range of emergent capabilities.

{{</citation>}}


### (74/80) REBEL: A Regularization-Based Solution for Reward Overoptimization in Reinforcement Learning from Human Feedback (Souradip Chakraborty et al., 2023)

{{<citation>}}

Souradip Chakraborty, Amisha Bhaskar, Anukriti Singh, Pratap Tokekar, Dinesh Manocha, Amrit Singh Bedi. (2023)  
**REBEL: A Regularization-Based Solution for Reward Overoptimization in Reinforcement Learning from Human Feedback**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14436v1)  

---


**ABSTRACT**  
In this work, we propose REBEL, an algorithm for sample efficient reward regularization based robotic reinforcement learning from human feedback (RRLHF). Reinforcement learning (RL) performance for continuous control robotics tasks is sensitive to the underlying reward function. In practice, the reward function often ends up misaligned with human intent, values, social norms, etc., leading to catastrophic failures in the real world. We leverage human preferences to learn regularized reward functions and eventually align the agents with the true intended behavior. We introduce a novel notion of reward regularization to the existing RRLHF framework, which is termed as agent preferences. So, we not only consider human feedback in terms of preferences, we also propose to take into account the preference of the underlying RL agent while learning the reward function. We show that this helps to improve the over-optimization associated with the design of reward functions in RL. We experimentally show that REBEL exhibits up to 70% improvement in sample efficiency to achieve a similar level of episodic reward returns as compared to the state-of-the-art methods such as PEBBLE and PEBBLE+SURF.

{{</citation>}}


### (75/80) Proceedings of the Dialogue Robot Competition 2023 (Ryuichiro Higashinaka et al., 2023)

{{<citation>}}

Ryuichiro Higashinaka, Takashi Minato, Hiromitsu Nishizaki, Takayuki Nagai. (2023)  
**Proceedings of the Dialogue Robot Competition 2023**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.14430v1)  

---


**ABSTRACT**  
The Dialogic Robot Competition 2023 (DRC2023) is a competition for humanoid robots (android robots that closely resemble humans) to compete in interactive capabilities. This is the third year of the competition. The top four teams from the preliminary competition held in November 2023 will compete in the final competition on Saturday, December 23. The task for the interactive robots is to recommend a tourism plan for a specific region. The robots can employ multimodal behaviors, such as language and gestures, to engage the user in the sightseeing plan they recommend. In the preliminary round, the interactive robots were stationed in a travel agency office, where visitors conversed with them and rated their performance via a questionnaire. In the final round, dialogue researchers and tourism industry professionals interacted with the robots and evaluated their performance. This event allows visitors to gain insights into the types of dialogue services that future dialogue robots should offer. The proceedings include papers on dialogue systems developed by the 12 teams participating in DRC2023, as well as an overview of the papers provided by all the teams.

{{</citation>}}


### (76/80) Designing a Skilled Soccer Team for RoboCup: Exploring Skill-Set-Primitives through Reinforcement Learning (Miguel Abreu et al., 2023)

{{<citation>}}

Miguel Abreu, Luis Paulo Reis, Nuno Lau. (2023)  
**Designing a Skilled Soccer Team for RoboCup: Exploring Skill-Set-Primitives through Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.14360v1)  

---


**ABSTRACT**  
The RoboCup 3D Soccer Simulation League serves as a competitive platform for showcasing innovation in autonomous humanoid robot agents through simulated soccer matches. Our team, FC Portugal, developed a new codebase from scratch in Python after RoboCup 2021. The team's performance is based on a set of skills centered around novel unifying primitives and a custom, symmetry-extended version of the Proximal Policy Optimization algorithm. Our methods have been thoroughly tested in official RoboCup matches, where FC Portugal has won the last two main competitions, in 2022 and 2023. This paper presents our training framework, as well as a timeline of skills developed using our skill-set-primitives, which considerably improve the sample efficiency and stability of skills, and motivate seamless transitions. We start with a significantly fast sprint-kick developed in 2021 and progress to the most recent skill set, which includes a multi-purpose omnidirectional walk, a dribble with unprecedented ball control, a solid kick, and a push skill. The push tackles both low-level collision-prone scenarios and high-level strategies to increase ball possession. We address the resource-intensive nature of this task through an innovative multi-agent learning approach. Finally, we release the codebase of our team to the RoboCup community, enabling other teams to transition to Python more easily and providing new teams with a robust and modern foundation upon which they can build new features.

{{</citation>}}


## cs.NI (1)



### (77/80) Quantum-Assisted Joint Caching and Power Allocation for Integrated Satellite-Terrestrial Networks (Yu Zhang et al., 2023)

{{<citation>}}

Yu Zhang, Yanmin Gong, Lei Fan, Yu Wang, Zhu Han, Yuanxiong Guo. (2023)  
**Quantum-Assisted Joint Caching and Power Allocation for Integrated Satellite-Terrestrial Networks**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.14448v1)  

---


**ABSTRACT**  
Low earth orbit (LEO) satellite network can complement terrestrial networks for achieving global wireless coverage and improving delay-sensitive Internet services. This paper proposes an integrated satellite-terrestrial network (ISTN) architecture to provide ground users with seamless and reliable content delivery services. For optimal service provisioning in this architecture, we formulate an optimization model to maximize the network throughput by jointly optimizing content delivery policy, cache placement, and transmission power allocation. The resulting optimization model is a large-scale mixed-integer nonlinear program (MINLP) that is intractable for classical computer solvers. Inspired by quantum computing techniques, we propose a hybrid quantum-classical generalized Benders' decomposition (HQCGBD) algorithm to address this challenge. Specifically, we first exploit the generalized Benders' decomposition (GBD) to decompose the problem into a master problem and a subproblem and then leverage the state-of-art quantum annealer to solve the challenging master problem.

{{</citation>}}


## cs.IR (1)



### (78/80) Attribute-driven Disentangled Representation Learning for Multimodal Recommendation (Zhenyang Li et al., 2023)

{{<citation>}}

Zhenyang Li, Fan Liu, Yinwei Wei, Zhiyong Cheng, Liqiang Nie, Mohan Kankanhalli. (2023)  
**Attribute-driven Disentangled Representation Learning for Multimodal Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-MM, cs.IR  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.14433v1)  

---


**ABSTRACT**  
Recommendation algorithms forecast user preferences by correlating user and item representations derived from historical interaction patterns. In pursuit of enhanced performance, many methods focus on learning robust and independent representations by disentangling the intricate factors within interaction data across various modalities in an unsupervised manner. However, such an approach obfuscates the discernment of how specific factors (e.g., category or brand) influence the outcomes, making it challenging to regulate their effects. In response to this challenge, we introduce a novel method called Attribute-Driven Disentangled Representation Learning (short for AD-DRL), which explicitly incorporates attributes from different modalities into the disentangled representation learning process. By assigning a specific attribute to each factor in multimodal features, AD-DRL can disentangle the factors at both attribute and attribute-value levels. To obtain robust and independent representations for each factor associated with a specific attribute, we first disentangle the representations of features both within and across different modalities. Moreover, we further enhance the robustness of the representations by fusing the multimodal features of the same factor. Empirical evaluations conducted on three public real-world datasets substantiate the effectiveness of AD-DRL, as well as its interpretability and controllability.

{{</citation>}}


## cs.SD (1)



### (79/80) ZMM-TTS: Zero-shot Multilingual and Multispeaker Speech Synthesis Conditioned on Self-supervised Discrete Speech Representations (Cheng Gong et al., 2023)

{{<citation>}}

Cheng Gong, Xin Wang, Erica Cooper, Dan Wells, Longbiao Wang, Jianwu Dang, Korin Richmond, Junichi Yamagishi. (2023)  
**ZMM-TTS: Zero-shot Multilingual and Multispeaker Speech Synthesis Conditioned on Self-supervised Discrete Speech Representations**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2312.14398v1)  

---


**ABSTRACT**  
Neural text-to-speech (TTS) has achieved human-like synthetic speech for single-speaker, single-language synthesis. Multilingual TTS systems are limited to resource-rich languages due to the lack of large paired text and studio-quality audio data. In most cases, TTS systems are built using a single speaker's voice. However, there is growing interest in developing systems that can synthesize voices for new speakers using only a few seconds of their speech. This paper presents ZMM-TTS, a multilingual and multispeaker framework utilizing quantized latent speech representations from a large-scale, pre-trained, self-supervised model. Our paper is the first to incorporate the representations from text-based and speech-based self-supervised learning models into multilingual speech synthesis tasks. We conducted comprehensive subjective and objective evaluations through a series of experiments. Our model has been proven effective in terms of speech naturalness and similarity for both seen and unseen speakers in six high-resource languages. We also tested the efficiency of our method on two hypothetical low-resource languages. The results are promising, indicating that our proposed approach can synthesize audio that is intelligible and has a high degree of similarity to the target speaker's voice, even without any training data for the new, unseen language.

{{</citation>}}


## cs.PL (1)



### (80/80) A Modular Approach to Metatheoretic Reasoning for Extensible Languages (Dawn Michaelson et al., 2023)

{{<citation>}}

Dawn Michaelson, Gopalan Nadathur, Eric Van Wyk. (2023)  
**A Modular Approach to Metatheoretic Reasoning for Extensible Languages**  

---
Primary Category: cs.PL  
Categories: cs-LO, cs-PL, cs.PL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.14374v1)  

---


**ABSTRACT**  
This paper concerns the development of metatheory for extensible languages. It uses as its starting point a view that programming languages tailored to specific application domains are to be constructed by composing components from an open library of independently-developed extensions to a host language. In the elaboration of this perspective, static analyses (such as typing) and dynamic semantics (such as evaluation) are described via relations whose specifications are distributed across the host language and extensions and are given in a rule-based fashion. Metatheoretic properties, which ensure that static analyses accurately gauge runtime behavior, are represented in this context by formulas over such relations. These properties may be fundamental to the language, introduced by the host language, or they may pertain to analyses introduced by individual extensions. We expose the problem of modular metatheory, i.e., the notion that proofs of relevant properties can be constructed by reasoning independently within each component in the library. To solve this problem, we propose the twin ideas of decomposing proofs around language fragments and of reasoning generically about extensions based on broad, a priori constraints imposed on their behavior. We establish the soundness of these styles of reasoning by showing how complete proofs of the properties can be automatically constructed for any language obtained by composing the independent parts. Mathematical precision is given to our discussions by framing them within a logic that encodes inductive rule-based specifications via least fixed-point definitions. We also sketch the structure of a practical system for metatheoretic reasoning for extensible languages based on the ideas developed.

{{</citation>}}
