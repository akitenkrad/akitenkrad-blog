---
draft: false
title: "arXiv @ 2023.10.09"
date: 2023-10-09
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.09"
    identifier: arxiv_20231009
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (13)](#cslg-13)
- [cs.CL (12)](#cscl-12)
- [cs.AI (5)](#csai-5)
- [cs.CV (14)](#cscv-14)
- [cs.CY (2)](#cscy-2)
- [cs.IR (6)](#csir-6)
- [cs.FL (1)](#csfl-1)
- [cs.HC (3)](#cshc-3)
- [eess.AS (3)](#eessas-3)
- [eess.IV (2)](#eessiv-2)
- [cs.SD (2)](#cssd-2)
- [cs.RO (1)](#csro-1)
- [cs.CR (1)](#cscr-1)
- [cs.DC (1)](#csdc-1)
- [q-bio.NC (1)](#q-bionc-1)

## cs.LG (13)



### (1/67) Beyond Text: A Deep Dive into Large Language Models' Ability on Understanding Graph Data (Yuntong Hu et al., 2023)

{{<citation>}}

Yuntong Hu, Zheng Zhang, Liang Zhao. (2023)  
**Beyond Text: A Deep Dive into Large Language Models' Ability on Understanding Graph Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.04944v1)  

---


**ABSTRACT**  
Large language models (LLMs) have achieved impressive performance on many natural language processing tasks. However, their capabilities on graph-structured data remain relatively unexplored. In this paper, we conduct a series of experiments benchmarking leading LLMs on diverse graph prediction tasks spanning node, edge, and graph levels. We aim to assess whether LLMs can effectively process graph data and leverage topological structures to enhance performance, compared to specialized graph neural networks. Through varied prompt formatting and task/dataset selection, we analyze how well LLMs can interpret and utilize graph structures. By comparing LLMs' performance with specialized graph models, we offer insights into the strengths and limitations of employing LLMs for graph analytics. Our findings provide insights into LLMs' capabilities and suggest avenues for further exploration in applying them to graph analytics.

{{</citation>}}


### (2/67) Large Language Models for Spatial Trajectory Patterns Mining (Zheng Zhang et al., 2023)

{{<citation>}}

Zheng Zhang, Hossein Amiri, Zhenke Liu, Andreas Züfle, Liang Zhao. (2023)  
**Large Language Models for Spatial Trajectory Patterns Mining**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.04942v1)  

---


**ABSTRACT**  
Identifying anomalous human spatial trajectory patterns can indicate dynamic changes in mobility behavior with applications in domains like infectious disease monitoring and elderly care. Recent advancements in large language models (LLMs) have demonstrated their ability to reason in a manner akin to humans. This presents significant potential for analyzing temporal patterns in human mobility. In this paper, we conduct empirical studies to assess the capabilities of leading LLMs like GPT-4 and Claude-2 in detecting anomalous behaviors from mobility data, by comparing to specialized methods. Our key findings demonstrate that LLMs can attain reasonable anomaly detection performance even without any specific cues. In addition, providing contextual clues about potential irregularities could further enhances their prediction efficacy. Moreover, LLMs can provide reasonable explanations for their judgments, thereby improving transparency. Our work provides insights on the strengths and limitations of LLMs for human spatial trajectory analysis.

{{</citation>}}


### (3/67) GradXKG: A Universal Explain-per-use Temporal Knowledge Graph Explainer (Chenhan Yuan et al., 2023)

{{<citation>}}

Chenhan Yuan, Hoda Eldardiry. (2023)  
**GradXKG: A Universal Explain-per-use Temporal Knowledge Graph Explainer**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.04889v1)  

---


**ABSTRACT**  
Temporal knowledge graphs (TKGs) have shown promise for reasoning tasks by incorporating a temporal dimension to represent how facts evolve over time. However, existing TKG reasoning (TKGR) models lack explainability due to their black-box nature. Recent work has attempted to address this through customized model architectures that generate reasoning paths, but these recent approaches have limited generalizability and provide sparse explanatory output. To enable interpretability for most TKGR models, we propose GradXKG, a novel two-stage gradient-based approach for explaining Relational Graph Convolution Network (RGCN)-based TKGR models. First, a Grad-CAM-inspired RGCN explainer tracks gradients to quantify each node's contribution across timesteps in an efficient "explain-per-use" fashion. Second, an integrated gradients explainer consolidates importance scores for RGCN outputs, extending compatibility across diverse TKGR architectures based on RGCN. Together, the two explainers highlight the most critical nodes at each timestep for a given prediction. Our extensive experiments demonstrated that, by leveraging gradient information, GradXKG provides insightful explanations grounded in the model's logic in a timely manner for most RGCN-based TKGR models. This helps address the lack of interpretability in existing TKGR models and provides a universal explanation approach applicable across various models.

{{</citation>}}


### (4/67) Prompt-to-OS (P2OS): Revolutionizing Operating Systems and Human-Computer Interaction with Integrated AI Generative Models (Gabriele Tolomei et al., 2023)

{{<citation>}}

Gabriele Tolomei, Cesare Campagnano, Fabrizio Silvestri, Giovanni Trappolini. (2023)  
**Prompt-to-OS (P2OS): Revolutionizing Operating Systems and Human-Computer Interaction with Integrated AI Generative Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CY, cs-HC, cs-LG, cs-OS, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04875v1)  

---


**ABSTRACT**  
In this paper, we present a groundbreaking paradigm for human-computer interaction that revolutionizes the traditional notion of an operating system.   Within this innovative framework, user requests issued to the machine are handled by an interconnected ecosystem of generative AI models that seamlessly integrate with or even replace traditional software applications. At the core of this paradigm shift are large generative models, such as language and diffusion models, which serve as the central interface between users and computers. This pioneering approach leverages the abilities of advanced language models, empowering users to engage in natural language conversations with their computing devices. Users can articulate their intentions, tasks, and inquiries directly to the system, eliminating the need for explicit commands or complex navigation. The language model comprehends and interprets the user's prompts, generating and displaying contextual and meaningful responses that facilitate seamless and intuitive interactions.   This paradigm shift not only streamlines user interactions but also opens up new possibilities for personalized experiences. Generative models can adapt to individual preferences, learning from user input and continuously improving their understanding and response generation. Furthermore, it enables enhanced accessibility, as users can interact with the system using speech or text, accommodating diverse communication preferences.   However, this visionary concept raises significant challenges, including privacy, security, trustability, and the ethical use of generative models. Robust safeguards must be in place to protect user data and prevent potential misuse or manipulation of the language model.   While the full realization of this paradigm is still far from being achieved, this paper serves as a starting point for envisioning this transformative potential.

{{</citation>}}


### (5/67) Uncovering hidden geometry in Transformers via disentangling position and context (Jiajun Song et al., 2023)

{{<citation>}}

Jiajun Song, Yiqiao Zhong. (2023)  
**Uncovering hidden geometry in Transformers via disentangling position and context**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.04861v1)  

---


**ABSTRACT**  
Transformers are widely used to extract complex semantic meanings from input tokens, yet they usually operate as black-box models. In this paper, we present a simple yet informative decomposition of hidden states (or embeddings) of trained transformers into interpretable components. For any layer, embedding vectors of input sequence samples are represented by a tensor $\boldsymbol{h} \in \mathbb{R}^{C \times T \times d}$. Given embedding vector $\boldsymbol{h}_{c,t} \in \mathbb{R}^d$ at sequence position $t \le T$ in a sequence (or context) $c \le C$, extracting the mean effects yields the decomposition \[ \boldsymbol{h}_{c,t} = \boldsymbol{\mu} + \mathbf{pos}_t + \mathbf{ctx}_c + \mathbf{resid}_{c,t} \] where $\boldsymbol{\mu}$ is the global mean vector, $\mathbf{pos}_t$ and $\mathbf{ctx}_c$ are the mean vectors across contexts and across positions respectively, and $\mathbf{resid}_{c,t}$ is the residual vector. For popular transformer architectures and diverse text datasets, empirically we find pervasive mathematical structure: (1) $(\mathbf{pos}_t)_{t}$ forms a low-dimensional, continuous, and often spiral shape across layers, (2) $(\mathbf{ctx}_c)_c$ shows clear cluster structure that falls into context topics, and (3) $(\mathbf{pos}_t)_{t}$ and $(\mathbf{ctx}_c)_c$ are mutually incoherent -- namely $\mathbf{pos}_t$ is almost orthogonal to $\mathbf{ctx}_c$ -- which is canonical in compressed sensing and dictionary learning. This decomposition offers structural insights about input formats in in-context learning (especially for induction heads) and in arithmetic tasks.

{{</citation>}}


### (6/67) LIPEx -- Locally Interpretable Probabilistic Explanations -- To Look Beyond The True Class (Hongbo Zhu et al., 2023)

{{<citation>}}

Hongbo Zhu, Angelo Cangelosi, Procheta Sen, Anirbit Mukherjee. (2023)  
**LIPEx -- Locally Interpretable Probabilistic Explanations -- To Look Beyond The True Class**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04856v1)  

---


**ABSTRACT**  
In this work, we instantiate a novel perturbation-based multi-class explanation framework, LIPEx (Locally Interpretable Probabilistic Explanation). We demonstrate that LIPEx not only locally replicates the probability distributions output by the widely used complex classification models but also provides insight into how every feature deemed to be important affects the prediction probability for each of the possible classes. We achieve this by defining the explanation as a matrix obtained via regression with respect to the Hellinger distance in the space of probability distributions. Ablation tests on text and image data, show that LIPEx-guided removal of important features from the data causes more change in predictions for the underlying model than similar tests on other saliency-based or feature importance-based XAI methods. It is also shown that compared to LIME, LIPEx is much more data efficient in terms of the number of perturbations needed for reliable evaluation of the explanation.

{{</citation>}}


### (7/67) Critique Ability of Large Language Models (Liangchen Luo et al., 2023)

{{<citation>}}

Liangchen Luo, Zi Lin, Yinxiao Liu, Lei Shu, Yun Zhu, Jingbo Shang, Lei Meng. (2023)  
**Critique Ability of Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.04815v1)  

---


**ABSTRACT**  
Critical thinking is essential for rational decision-making and problem-solving. This skill hinges on the ability to provide precise and reasoned critiques and is a hallmark of human intelligence. In the era of large language models (LLMs), this study explores the ability of LLMs to deliver accurate critiques across various tasks. We are interested in this topic as a capable critic model could not only serve as a reliable evaluator, but also as a source of supervised signals for model tuning. Particularly, if a model can self-critique, it has the potential for autonomous self-improvement. To examine this, we introduce a unified evaluation framework for assessing the critique abilities of LLMs. We develop a benchmark called CriticBench, which comprises 3K high-quality natural language queries and corresponding model responses; and annotate the correctness of these responses. The benchmark cover tasks such as math problem-solving, code completion, and question answering. We evaluate multiple LLMs on the collected dataset and our analysis reveals several noteworthy insights: (1) Critique is generally challenging for most LLMs, and this capability often emerges only when models are sufficiently large. (2) In particular, self-critique is especially difficult. Even top-performing LLMs struggle to achieve satisfactory performance. (3) Models tend to have lower critique accuracy on problems where they are most uncertain. To this end, we introduce a simple yet effective baseline named self-check, which leverages self-critique to improve task performance for various models. We hope this study serves as an initial exploration into understanding the critique abilities of LLMs, and aims to inform future research, including the development of more proficient critic models and the application of critiques across diverse tasks.

{{</citation>}}


### (8/67) Accelerate Multi-Agent Reinforcement Learning in Zero-Sum Games with Subgame Curriculum Learning (Jiayu Chen et al., 2023)

{{<citation>}}

Jiayu Chen, Zelai Xu, Yunfei Li, Chao Yu, Jiaming Song, Huazhong Yang, Fei Fang, Yu Wang, Yi Wu. (2023)  
**Accelerate Multi-Agent Reinforcement Learning in Zero-Sum Games with Subgame Curriculum Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Google, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04796v1)  

---


**ABSTRACT**  
Learning Nash equilibrium (NE) in complex zero-sum games with multi-agent reinforcement learning (MARL) can be extremely computationally expensive. Curriculum learning is an effective way to accelerate learning, but an under-explored dimension for generating a curriculum is the difficulty-to-learn of the subgames -- games induced by starting from a specific state. In this work, we present a novel subgame curriculum learning framework for zero-sum games. It adopts an adaptive initial state distribution by resetting agents to some previously visited states where they can quickly learn to improve performance. Building upon this framework, we derive a subgame selection metric that approximates the squared distance to NE values and further adopt a particle-based state sampler for subgame generation. Integrating these techniques leads to our new algorithm, Subgame Automatic Curriculum Learning (SACL), which is a realization of the subgame curriculum learning framework. SACL can be combined with any MARL algorithm such as MAPPO. Experiments in the particle-world environment and Google Research Football environment show SACL produces much stronger policies than baselines. In the challenging hide-and-seek quadrant environment, SACL produces all four emergent stages and uses only half the samples of MAPPO with self-play. The project website is at https://sites.google.com/view/sacl-rl.

{{</citation>}}


### (9/67) Optimal Sequential Decision-Making in Geosteering: A Reinforcement Learning Approach (Ressi Bonti Muhammad et al., 2023)

{{<citation>}}

Ressi Bonti Muhammad, Sergey Alyaev, Reidar Brumer Bratvold. (2023)  
**Optimal Sequential Decision-Making in Geosteering: A Reinforcement Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, physics-geo-ph  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04772v1)  

---


**ABSTRACT**  
Trajectory adjustment decisions throughout the drilling process, called geosteering, affect subsequent choices and information gathering, thus resulting in a coupled sequential decision problem. Previous works on applying decision optimization methods in geosteering rely on greedy optimization or Approximate Dynamic Programming (ADP). Either decision optimization method requires explicit uncertainty and objective function models, making developing decision optimization methods for complex and realistic geosteering environments challenging to impossible. We use the Deep Q-Network (DQN) method, a model-free reinforcement learning (RL) method that learns directly from the decision environment, to optimize geosteering decisions. The expensive computations for RL are handled during the offline training stage. Evaluating DQN needed for real-time decision support takes milliseconds and is faster than the traditional alternatives. Moreover, for two previously published synthetic geosteering scenarios, our results show that RL achieves high-quality outcomes comparable to the quasi-optimal ADP. Yet, the model-free nature of RL means that by replacing the training environment, we can extend it to problems where the solution to ADP is prohibitively expensive to compute. This flexibility will allow applying it to more complex environments and make hybrid versions trained with real data in the future.

{{</citation>}}


### (10/67) Task Aware Modulation using Representation Learning: An Approach for Few Shot Learning in Heterogeneous Systems (Arvind Renganathan et al., 2023)

{{<citation>}}

Arvind Renganathan, Rahul Ghosh, Ankush Khandelwal, Vipin Kumar. (2023)  
**Task Aware Modulation using Representation Learning: An Approach for Few Shot Learning in Heterogeneous Systems**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.04727v1)  

---


**ABSTRACT**  
We present a Task-aware modulation using Representation Learning (TAM-RL) framework that enhances personalized predictions in few-shot settings for heterogeneous systems when individual task characteristics are not known. TAM-RL extracts embeddings representing the actual inherent characteristics of these entities and uses these characteristics to personalize the predictions for each entity/task. Using real-world hydrological and flux tower benchmark data sets, we show that TAM-RL can significantly outperform existing baseline approaches such as MAML and multi-modal MAML (MMAML) while being much faster and simpler to train due to less complexity. Specifically, TAM-RL eliminates the need for sensitive hyper-parameters like inner loop steps and inner loop learning rate, which are crucial for model convergence in MAML, MMAML. We further present an empirical evaluation via synthetic data to explore the impact of heterogeneity amongst the entities on the relative performance of MAML, MMAML, and TAM-RL. We show that TAM-RL significantly improves predictive performance for cases where it is possible to learn distinct representations for different tasks.

{{</citation>}}


### (11/67) Offline Imitation Learning with Variational Counterfactual Reasoning (Bowei He et al., 2023)

{{<citation>}}

Bowei He, Zexu Sun, Jinxin Liu, Shuai Zhang, Xu Chen, Chen Ma. (2023)  
**Offline Imitation Learning with Variational Counterfactual Reasoning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.04706v1)  

---


**ABSTRACT**  
In offline Imitation Learning (IL), an agent aims to learn an optimal expert behavior policy without additional online environment interactions. However, in many real-world scenarios, such as robotics manipulation, the offline dataset is collected from suboptimal behaviors without rewards. Due to the scarce expert data, the agents usually suffer from simply memorizing poor trajectories and are vulnerable to the variations in the environments, lacking the capability of generalizing to new environments. To effectively remove spurious features that would otherwise bias the agent and hinder generalization, we propose a framework named \underline{O}ffline \underline{I}mitation \underline{L}earning with \underline{C}ounterfactual data \underline{A}ugmentation (OILCA). In particular, we leverage the identifiable variational autoencoder to generate \textit{counterfactual} samples. We theoretically analyze the counterfactual identification and the improvement of generalization. Moreover, we conduct extensive experiments to demonstrate that our approach significantly outperforms various baselines on both \textsc{DeepMind Control Suite} benchmark for in-distribution robustness and \textsc{CausalWorld} benchmark for out-of-distribution generalization.

{{</citation>}}


### (12/67) Twin Graph-based Anomaly Detection via Attentive Multi-Modal Learning for Microservice System (Jun Huang et al., 2023)

{{<citation>}}

Jun Huang, Yang Yang, Hang Yu, Jianguo Li, Xiao Zheng. (2023)  
**Twin Graph-based Anomaly Detection via Attentive Multi-Modal Learning for Microservice System**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SE, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.04701v1)  

---


**ABSTRACT**  
Microservice architecture has sprung up over recent years for managing enterprise applications, due to its ability to independently deploy and scale services. Despite its benefits, ensuring the reliability and safety of a microservice system remains highly challenging. Existing anomaly detection algorithms based on a single data modality (i.e., metrics, logs, or traces) fail to fully account for the complex correlations and interactions between different modalities, leading to false negatives and false alarms, whereas incorporating more data modalities can offer opportunities for further performance gain. As a fresh attempt, we propose in this paper a semi-supervised graph-based anomaly detection method, MSTGAD, which seamlessly integrates all available data modalities via attentive multi-modal learning. First, we extract and normalize features from the three modalities, and further integrate them using a graph, namely MST (microservice system twin) graph, where each node represents a service instance and the edge indicates the scheduling relationship between different service instances. The MST graph provides a virtual representation of the status and scheduling relationships among service instances of a real-world microservice system. Second, we construct a transformer-based neural network with both spatial and temporal attention mechanisms to model the inter-correlations between different modalities and temporal dependencies between the data points. This enables us to detect anomalies automatically and accurately in real-time. The source code of MSTGAD is publicly available at https://github.com/alipay/microservice_system_twin_graph_based_anomaly_detection.

{{</citation>}}


### (13/67) Label-free Node Classification on Graphs with Large Language Models (LLMS) (Zhikai Chen et al., 2023)

{{<citation>}}

Zhikai Chen, Haitao Mao, Hongzhi Wen, Haoyu Han, Wei Jin, Haiyang Zhang, Hui Liu, Jiliang Tang. (2023)  
**Label-free Node Classification on Graphs with Large Language Models (LLMS)**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Language Model  
[Paper Link](http://arxiv.org/abs/2310.04668v1)  

---


**ABSTRACT**  
In recent years, there have been remarkable advancements in node classification achieved by Graph Neural Networks (GNNs). However, they necessitate abundant high-quality labels to ensure promising performance. In contrast, Large Language Models (LLMs) exhibit impressive zero-shot proficiency on text-attributed graphs. Yet, they face challenges in efficiently processing structural data and suffer from high inference costs. In light of these observations, this work introduces a label-free node classification on graphs with LLMs pipeline, LLM-GNN. It amalgamates the strengths of both GNNs and LLMs while mitigating their limitations. Specifically, LLMs are leveraged to annotate a small portion of nodes and then GNNs are trained on LLMs' annotations to make predictions for the remaining large portion of nodes. The implementation of LLM-GNN faces a unique challenge: how can we actively select nodes for LLMs to annotate and consequently enhance the GNN training? How can we leverage LLMs to obtain annotations of high quality, representativeness, and diversity, thereby enhancing GNN performance with less cost? To tackle this challenge, we develop an annotation quality heuristic and leverage the confidence scores derived from LLMs to advanced node selection. Comprehensive experimental results validate the effectiveness of LLM-GNN. In particular, LLM-GNN can achieve an accuracy of 74.9% on a vast-scale dataset \products with a cost less than 1 dollar.

{{</citation>}}


## cs.CL (12)



### (14/67) Large Language Models Only Pass Primary School Exams in Indonesia: A Comprehensive Test on IndoMMLU (Fajri Koto et al., 2023)

{{<citation>}}

Fajri Koto, Nurul Aisyah, Haonan Li, Timothy Baldwin. (2023)  
**Large Language Models Only Pass Primary School Exams in Indonesia: A Comprehensive Test on IndoMMLU**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLOOM, Falcon, GPT, GPT-3.5, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.04928v1)  

---


**ABSTRACT**  
Large language models have made significant advancements in natural language processing (NLP), exhibiting human performance across various classic NLP tasks. These tasks, however, focus on structure and semantics, and few are designed to assess reasoning abilities and real-world knowledge, which are increasingly vital given that these models are trained on extensive textual data and information. While prior research primarily focuses on English, in this work, we gather a collection of exam problems from primary school to university entrance tests in Indonesia, and evaluate whether large language models can pass the exams. We obtain 14,906 questions across 63 tasks and levels, with 46\% of the questions focusing on assessing proficiency in the Indonesian language and knowledge of nine local languages and cultures in Indonesia. Our empirical evaluations show that GPT-3.5 only manages to pass the Indonesian primary school level, with limited knowledge of the Indonesian local languages and cultures. Other smaller models such as BLOOMZ and Falcon fail the exams.

{{</citation>}}


### (15/67) Faithful Knowledge Graph Explanations for Commonsense Reasoning (Weihe Zhai et al., 2023)

{{<citation>}}

Weihe Zhai, Arkaitz Zubiaga, Bingquan Liu. (2023)  
**Faithful Knowledge Graph Explanations for Commonsense Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GNN, Knowledge Graph, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.04910v1)  

---


**ABSTRACT**  
While fusing language models (LMs) and knowledge graphs (KGs) has become common in commonsense question answering research, enabling faithful chain-of-thought explanations in these models remains an open problem. One major weakness of current KG-based explanation techniques is that they overlook the faithfulness of generated explanations during evaluation. To address this gap, we make two main contributions: (1) We propose and validate two quantitative metrics - graph consistency and graph fidelity - to measure the faithfulness of KG-based explanations. (2) We introduce Consistent GNN (CGNN), a novel training method that adds a consistency regularization term to improve explanation faithfulness. Our analysis shows that predictions from KG often diverge from original model predictions. The proposed CGNN approach boosts consistency and fidelity, demonstrating its potential for producing more faithful explanations. Our work emphasises the importance of explicitly evaluating suggest a path forward for developing architectures for faithful graph-based explanations.

{{</citation>}}


### (16/67) Chat Vector: A Simple Approach to Equip LLMs With New Language Chat Capabilities (Shih-Cheng Huang et al., 2023)

{{<citation>}}

Shih-Cheng Huang, Pin-Zu Li, Yu-Chi Hsu, Kuang-Ming Chen, Yu Tung Lin, Shih-Kai Hsiao, Richard Tzong-Han Tsai, Hung-yi Lee. (2023)  
**Chat Vector: A Simple Approach to Equip LLMs With New Language Chat Capabilities**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.04799v1)  

---


**ABSTRACT**  
With the advancements in conversational AI, such as ChatGPT, this paper focuses on exploring developing Large Language Models (LLMs) for non-English languages, especially emphasizing alignment with human preferences. We introduce a computationally efficient method, leveraging chat vector, to synergize pre-existing knowledge and behaviors in LLMs, restructuring the conventional training paradigm from continual pre-train -> SFT -> RLHF to continual pre-train + chat vector. Our empirical studies, primarily focused on Traditional Chinese, employ LLaMA2 as the base model and acquire the chat vector by subtracting the pre-trained weights, LLaMA2, from the weights of LLaMA2-chat. Evaluating from three distinct facets, which are toxicity, ability of instruction following, and multi-turn dialogue demonstrates the chat vector's superior efficacy in chatting. To confirm the adaptability of our approach, we extend our experiments to include models pre-trained in both Korean and Simplified Chinese, illustrating the versatility of our methodology. Overall, we present a significant solution in aligning LLMs with human preferences efficiently across various languages, accomplished by the chat vector.

{{</citation>}}


### (17/67) FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets (Neng Wang et al., 2023)

{{<citation>}}

Neng Wang, Hongyang Yang, Christina Dan Wang. (2023)  
**FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, q-fin-TR  
Keywords: Financial, GPT, Language Model, NER, NLP, Named Entity Recognition, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.04793v1)  

---


**ABSTRACT**  
In the swiftly expanding domain of Natural Language Processing (NLP), the potential of GPT-based models for the financial sector is increasingly evident. However, the integration of these models with financial datasets presents challenges, notably in determining their adeptness and relevance. This paper introduces a distinctive approach anchored in the Instruction Tuning paradigm for open-source large language models, specifically adapted for financial contexts. Through this methodology, we capitalize on the interoperability of open-source models, ensuring a seamless and transparent integration. We begin by explaining the Instruction Tuning paradigm, highlighting its effectiveness for immediate integration. The paper presents a benchmarking scheme designed for end-to-end training and testing, employing a cost-effective progression. Firstly, we assess basic competencies and fundamental tasks, such as Named Entity Recognition (NER) and sentiment analysis to enhance specialization. Next, we delve into a comprehensive model, executing multi-task operations by amalgamating all instructional tunings to examine versatility. Finally, we explore the zero-shot capabilities by earmarking unseen tasks and incorporating novel datasets to understand adaptability in uncharted terrains. Such a paradigm fortifies the principles of openness and reproducibility, laying a robust foundation for future investigations in open-source financial large language models (FinLLMs).

{{</citation>}}


### (18/67) Improving the Reliability of Large Language Models by Leveraging Uncertainty-Aware In-Context Learning (Yuchen Yang et al., 2023)

{{<citation>}}

Yuchen Yang, Houqiang Li, Yanfeng Wang, Yu Wang. (2023)  
**Improving the Reliability of Large Language Models by Leveraging Uncertainty-Aware In-Context Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.04782v1)  

---


**ABSTRACT**  
In recent years, large-scale language models (LLMs) have gained attention for their impressive text generation capabilities. However, these models often face the challenge of "hallucination," which undermines their reliability. In this study, we introduce an uncertainty-aware in-context learning framework to empower the model to enhance or reject its output in response to uncertainty. Human-defined methods for estimating uncertainty typically assume that "uncertainty is lower when the model's response is correct compared to when it is incorrect." However, setting a precise threshold to distinguish correctness is challenging. Therefore, we introduce uncertainty information as an intermediary variable that implicitly influences the model's behavior. Our innovative uncertainty-aware in-context learning framework involves fine-tuning the LLM using a calibration dataset. Our aim is to improve the model's responses by filtering out answers with high uncertainty while considering the model's knowledge limitations. We evaluate the model's knowledge by examining multiple responses to the same question for the presence of a correct answer. When the model lacks relevant knowledge, the response should indicate that the question cannot be answered. Conversely, when the model has relevant knowledge, the response should provide the correct answer. Extensive experiments confirm the effectiveness of our framework, leading to two key findings. First, the logit output values of the LLM partly reflect inherent uncertainty. Second, our model autonomously recognizes uncertainty, resulting in improved responses.

{{</citation>}}


### (19/67) A New Dataset for End-to-End Sign Language Translation: The Greek Elementary School Dataset (Andreas Voskou et al., 2023)

{{<citation>}}

Andreas Voskou, Konstantinos P. Panousis, Harris Partaourides, Kyriakos Tolias, Sotirios Chatzis. (2023)  
**A New Dataset for End-to-End Sign Language Translation: The Greek Elementary School Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs-LG, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.04753v1)  

---


**ABSTRACT**  
Automatic Sign Language Translation (SLT) is a research avenue of great societal impact. End-to-End SLT facilitates the interaction of Hard-of-Hearing (HoH) with hearing people, thus improving their social life and opportunities for participation in social life. However, research within this frame of reference is still in its infancy, and current resources are particularly limited. Existing SLT methods are either of low translation ability or are trained and evaluated on datasets of restricted vocabulary and questionable real-world value. A characteristic example is Phoenix2014T benchmark dataset, which only covers weather forecasts in German Sign Language. To address this shortage of resources, we introduce a newly constructed collection of 29653 Greek Sign Language video-translation pairs which is based on the official syllabus of Greek Elementary School. Our dataset covers a wide range of subjects. We use this novel dataset to train recent state-of-the-art Transformer-based methods widely used in SLT research. Our results demonstrate the potential of our introduced dataset to advance SLT research by offering a favourable balance between usability and real-world value.

{{</citation>}}


### (20/67) Resprompt: Residual Connection Prompting Advances Multi-Step Reasoning in Large Language Models (Song Jiang et al., 2023)

{{<citation>}}

Song Jiang, Zahra Shakeri, Aaron Chan, Maziar Sanjabi, Hamed Firooz, Yinglong Xia, Bugra Akyildiz, Yizhou Sun, Jinchao Li, Qifan Wang, Asli Celikyilmaz. (2023)  
**Resprompt: Residual Connection Prompting Advances Multi-Step Reasoning in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.04743v1)  

---


**ABSTRACT**  
Chain-of-thought (CoT) prompting, which offers step-by-step problem-solving rationales, has impressively unlocked the reasoning potential of large language models (LLMs). Yet, the standard CoT is less effective in problems demanding multiple reasoning steps. This limitation arises from the complex reasoning process in multi-step problems: later stages often depend on the results of several steps earlier, not just the results of the immediately preceding step. Such complexities suggest the reasoning process is naturally represented as a graph. The almost linear and straightforward structure of CoT prompting, however, struggles to capture this complex reasoning graph. To address this challenge, we propose Residual Connection Prompting (RESPROMPT), a new prompting strategy that advances multi-step reasoning in LLMs. Our key idea is to reconstruct the reasoning graph within prompts. We achieve this by integrating necessary connections-links present in the reasoning graph but missing in the linear CoT flow-into the prompts. Termed "residual connections", these links are pivotal in morphing the linear CoT structure into a graph representation, effectively capturing the complex reasoning graphs inherent in multi-step problems. We evaluate RESPROMPT on six benchmarks across three diverse domains: math, sequential, and commonsense reasoning. For the open-sourced LLaMA family of models, RESPROMPT yields a significant average reasoning accuracy improvement of 12.5% on LLaMA-65B and 6.8% on LLaMA2-70B. Breakdown analysis further highlights RESPROMPT particularly excels in complex multi-step reasoning: for questions demanding at least five reasoning steps, RESPROMPT outperforms the best CoT based benchmarks by a remarkable average improvement of 21.1% on LLaMA-65B and 14.3% on LLaMA2-70B. Through extensive ablation studies and analyses, we pinpoint how to most effectively build residual connections.

{{</citation>}}


### (21/67) Zero-shot Cross-lingual Transfer without Parallel Corpus (Yuyang Zhang et al., 2023)

{{<citation>}}

Yuyang Zhang, Xiaofeng Han, Baojun Wang. (2023)  
**Zero-shot Cross-lingual Transfer without Parallel Corpus**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.04726v1)  

---


**ABSTRACT**  
Recently, although pre-trained language models have achieved great success on multilingual NLP (Natural Language Processing) tasks, the lack of training data on many tasks in low-resource languages still limits their performance. One effective way of solving that problem is to transfer knowledge from rich-resource languages to low-resource languages. However, many previous works on cross-lingual transfer rely heavily on the parallel corpus or translation models, which are often difficult to obtain. We propose a novel approach to conduct zero-shot cross-lingual transfer with a pre-trained model. It consists of a Bilingual Task Fitting module that applies task-related bilingual information alignment; a self-training module generates pseudo soft and hard labels for unlabeled data and utilizes them to conduct self-training. We got the new SOTA on different tasks without any dependencies on the parallel corpus or translation models.

{{</citation>}}


### (22/67) Integrating Contrastive Learning into a Multitask Transformer Model for Effective Domain Adaptation (Chung-Soo Ahn et al., 2023)

{{<citation>}}

Chung-Soo Ahn, Jagath C. Rajapakse, Rajib Rana. (2023)  
**Integrating Contrastive Learning into a Multitask Transformer Model for Effective Domain Adaptation**  

---
Primary Category: cs.CL  
Categories: Speech Emotion Recognition, Domain adaptation, cs-CL, cs-HC, cs-LG, cs.CL  
Keywords: Contrastive Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2310.04703v1)  

---


**ABSTRACT**  
While speech emotion recognition (SER) research has made significant progress, achieving generalization across various corpora continues to pose a problem. We propose a novel domain adaptation technique that embodies a multitask framework with SER as the primary task, and contrastive learning and information maximisation loss as auxiliary tasks, underpinned by fine-tuning of transformers pre-trained on large language models. Empirical results obtained through experiments on well-established datasets like IEMOCAP and MSP-IMPROV, illustrate that our proposed model achieves state-of-the-art performance in SER within cross-corpus scenarios.

{{</citation>}}


### (23/67) EMO: Earth Mover Distance Optimization for Auto-Regressive Language Modeling (Siyu Ren et al., 2023)

{{<citation>}}

Siyu Ren, Zhiyong Wu, Kenny Q. Zhu. (2023)  
**EMO: Earth Mover Distance Optimization for Auto-Regressive Language Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.04691v1)  

---


**ABSTRACT**  
Neural language models are probabilistic models of human text. They are predominantly trained using maximum likelihood estimation (MLE), which is equivalent to minimizing the forward cross-entropy between the empirical data distribution and the model distribution. However, various degeneration phenomena are still widely observed when decoding from the distributions learned by such models. We establish that the forward cross-entropy is suboptimal as a distance metric for aligning human and model distribution due to its (1) recall-prioritization (2) negative diversity ignorance and (3) train-test mismatch. In this paper, we propose Earth Mover Distance Optimization (EMO) for auto-regressive language modeling. EMO capitalizes on the inherent properties of earth mover distance to address the aforementioned challenges. Due to the high complexity of direct computation, we further introduce a feasible upper bound for EMO to ease end-to-end training. Upon extensive evaluation of language models trained using EMO and MLE. We find that EMO demonstrates a consistently better language modeling performance than MLE across domains. Moreover, EMO demonstrates noteworthy enhancements in downstream performance with minimal fine-tuning on merely 25,000 sentences. This highlights the tremendous potential of EMO as a lightweight calibration method for enhancing large-scale pre-trained language models.

{{</citation>}}


### (24/67) The Cost of Down-Scaling Language Models: Fact Recall Deteriorates before In-Context Learning (Tian Jin et al., 2023)

{{<citation>}}

Tian Jin, Nolan Clement, Xin Dong, Vaishnavh Nagarajan, Michael Carbin, Jonathan Ragan-Kelley, Gintare Karolina Dziugaite. (2023)  
**The Cost of Down-Scaling Language Models: Fact Recall Deteriorates before In-Context Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.04680v1)  

---


**ABSTRACT**  
How does scaling the number of parameters in large language models (LLMs) affect their core capabilities? We study two natural scaling techniques -- weight pruning and simply training a smaller or larger model, which we refer to as dense scaling -- and their effects on two core capabilities of LLMs: (a) recalling facts presented during pre-training and (b) processing information presented in-context during inference. By curating a suite of tasks that help disentangle these two capabilities, we find a striking difference in how these two abilities evolve due to scaling. Reducing the model size by more than 30\% (via either scaling approach) significantly decreases the ability to recall facts seen in pre-training. Yet, a 60--70\% reduction largely preserves the various ways the model can process in-context information, ranging from retrieving answers from a long context to learning parameterized functions from in-context exemplars. The fact that both dense scaling and weight pruning exhibit this behavior suggests that scaling model size has an inherently disparate effect on fact recall and in-context learning.

{{</citation>}}


### (25/67) Automatic Anonymization of Swiss Federal Supreme Court Rulings (Joel Niklaus et al., 2023)

{{<citation>}}

Joel Niklaus, Robin Mamié, Matthias Stürmer, Daniel Brunner, Marcel Gygli. (2023)  
**Automatic Anonymization of Swiss Federal Supreme Court Rulings**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.04632v1)  

---


**ABSTRACT**  
Releasing court decisions to the public relies on proper anonymization to protect all involved parties, where necessary. The Swiss Federal Supreme Court relies on an existing system that combines different traditional computational methods with human experts. In this work, we enhance the existing anonymization software using a large dataset annotated with entities to be anonymized. We compared BERT-based models with models pre-trained on in-domain data. Our results show that using in-domain data to pre-train the models further improves the F1-score by more than 5\% compared to existing models. Our work demonstrates that combining existing anonymization methods, such as regular expressions, with machine learning can further reduce manual labor and enhance automatic suggestions.

{{</citation>}}


## cs.AI (5)



### (26/67) Robust Network Pruning With Sparse Entropic Wasserstein Regression (Lei You et al., 2023)

{{<citation>}}

Lei You, Hei Victor Cheng. (2023)  
**Robust Network Pruning With Sparse Entropic Wasserstein Regression**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2310.04918v1)  

---


**ABSTRACT**  
This study unveils a cutting-edge technique for neural network pruning that judiciously addresses noisy gradients during the computation of the empirical Fisher Information Matrix (FIM). We introduce an entropic Wasserstein regression (EWR) formulation, capitalizing on the geometric attributes of the optimal transport (OT) problem. This is analytically showcased to excel in noise mitigation by adopting neighborhood interpolation across data points. The unique strength of the Wasserstein distance is its intrinsic ability to strike a balance between noise reduction and covariance information preservation. Extensive experiments performed on various networks show comparable performance of the proposed method with state-of-the-art (SoTA) network pruning algorithms. Our proposed method outperforms the SoTA when the network size or the target sparsity is large, the gain is even larger with the existence of noisy gradients, possibly from noisy data, analog memory, or adversarial attacks. Notably, our proposed method achieves a gain of 6% improvement in accuracy and 8% improvement in testing loss for MobileNetV1 with less than one-fourth of the network parameters remaining.

{{</citation>}}


### (27/67) Question-focused Summarization by Decomposing Articles into Facts and Opinions and Retrieving Entities (Krutika Sarode et al., 2023)

{{<citation>}}

Krutika Sarode, Shashidhar Reddy Javaji, Vishal Kalakonnavar. (2023)  
**Question-focused Summarization by Decomposing Articles into Facts and Opinions and Retrieving Entities**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: GPT, Language Model, Summarization  
[Paper Link](http://arxiv.org/abs/2310.04880v1)  

---


**ABSTRACT**  
This research focuses on utilizing natural language processing techniques to predict stock price fluctuations, with a specific interest in early detection of economic, political, social, and technological changes that can be leveraged for capturing market opportunities. The proposed approach includes the identification of salient facts and events from news articles, then use these facts to form tuples with entities which can be used to get summaries of market changes for particular entity and then finally combining all the summaries to form a final abstract summary of the whole article. The research aims to establish relationships between companies and entities through the analysis of Wikipedia data and articles from the Economist. Large Language Model GPT 3.5 is used for getting the summaries and also forming the final summary. The ultimate goal of this research is to develop a comprehensive system that can provide financial analysts and investors with more informed decision-making tools by enabling early detection of market trends and events.

{{</citation>}}


### (28/67) Dual Grained Quantization: Efficient Fine-Grained Quantization for LLM (Luoming Zhang et al., 2023)

{{<citation>}}

Luoming Zhang, Wen Fei, Weijia Wu, Yefei He, Zhenyu Lou, Hong Zhou. (2023)  
**Dual Grained Quantization: Efficient Fine-Grained Quantization for LLM**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, Quantization  
[Paper Link](http://arxiv.org/abs/2310.04836v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) pose significant hardware challenges related to memory requirements and computational ability. There are two mainstream quantization schemes for LLMs: coarse-grained ($\textit{e.g.,}$ channel-wise) quantization and fine-grained ($\textit{e.g.,}$ group-wise) quantization. Fine-grained quantization has smaller quantization loss, consequently achieving superior performance. However, when applied to weight-activation quantization, it disrupts continuous integer matrix multiplication, leading to inefficient inference. In this paper, we introduce Dual Grained Quantization (DGQ), a novel A8W4 quantization for LLM that maintains superior performance while ensuring fast inference speed. DSQ dequantizes the fine-grained INT4 weight into coarse-grained INT8 representation and preform matrix multiplication using INT8 kernels. Besides, we develop a two-phase grid search algorithm to simplify the determination of fine-grained and coarse-grained quantization scales. We also devise a percentile clipping schema for smoothing the activation outliers without the need for complex optimization techniques. Experimental results demonstrate that DGQ consistently outperforms prior methods across various LLM architectures and a wide range of tasks. Remarkably, by our implemented efficient CUTLASS kernel, we achieve $\textbf{1.12}$ $\times$ memory reduction and $\textbf{3.24}$ $\times$ speed gains comparing A16W4 implementation. These advancements enable efficient deployment of A8W4 LLMs for real-world applications.

{{</citation>}}


### (29/67) On the Evolution of Knowledge Graphs: A Survey and Perspective (Xuhui Jiang et al., 2023)

{{<citation>}}

Xuhui Jiang, Chengjin Xu, Yinghan Shen, Xun Sun, Lumingyuan Tang, Saizhuo Wang, Zhongwu Chen, Yuanzhuo Wang, Jian Guo. (2023)  
**On the Evolution of Knowledge Graphs: A Survey and Perspective**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.04835v1)  

---


**ABSTRACT**  
Knowledge graphs (KGs) are structured representations of diversified knowledge. They are widely used in various intelligent applications. In this article, we provide a comprehensive survey on the evolution of various types of knowledge graphs (i.e., static KGs, dynamic KGs, temporal KGs, and event KGs) and techniques for knowledge extraction and reasoning. Furthermore, we introduce the practical applications of different types of KGs, including a case study in financial analysis. Finally, we propose our perspective on the future directions of knowledge engineering, including the potential of combining the power of knowledge graphs and large language models (LLMs), and the evolution of knowledge extraction, reasoning, and representation.

{{</citation>}}


### (30/67) DiffNAS: Bootstrapping Diffusion Models by Prompting for Better Architectures (Wenhao Li et al., 2023)

{{<citation>}}

Wenhao Li, Xiu Su, Shan You, Fei Wang, Chen Qian, Chang Xu. (2023)  
**DiffNAS: Bootstrapping Diffusion Models by Prompting for Better Architectures**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs-LG, cs.AI  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.04750v1)  

---


**ABSTRACT**  
Diffusion models have recently exhibited remarkable performance on synthetic data. After a diffusion path is selected, a base model, such as UNet, operates as a denoising autoencoder, primarily predicting noises that need to be eliminated step by step. Consequently, it is crucial to employ a model that aligns with the expected budgets to facilitate superior synthetic performance. In this paper, we meticulously analyze the diffusion model and engineer a base model search approach, denoted "DiffNAS". Specifically, we leverage GPT-4 as a supernet to expedite the search, supplemented with a search memory to enhance the results. Moreover, we employ RFID as a proxy to promptly rank the experimental outcomes produced by GPT-4. We also adopt a rapid-convergence training strategy to boost search efficiency. Rigorous experimentation corroborates that our algorithm can augment the search efficiency by 2 times under GPT-based scenarios, while also attaining a performance of 2.82 with 0.37 improvement in FID on CIFAR10 relative to the benchmark IDDPM algorithm.

{{</citation>}}


## cs.CV (14)



### (31/67) Analyzing Zero-Shot Abilities of Vision-Language Models on Video Understanding Tasks (Avinash Madasu et al., 2023)

{{<citation>}}

Avinash Madasu, Anahita Bhiwandiwalla, Vasudev Lal. (2023)  
**Analyzing Zero-Shot Abilities of Vision-Language Models on Video Understanding Tasks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Language Model, QA, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.04914v1)  

---


**ABSTRACT**  
Foundational multimodal models pre-trained on large scale image-text pairs or video-text pairs or both have shown strong generalization abilities on downstream tasks. However unlike image-text models, pretraining video-text models is always not feasible due to the difficulty in collecting large-scale clean and aligned data, and exponential computational costs involved in the pretraining phase. Therefore, the pertinent question to ask is: Can image-text models be adapted to video tasks and is there any benefit to using these models over pretraining directly on videos? In this work, we focus on this question by proposing a detailed study on the generalization abilities of image-text models when evaluated on video understanding tasks in a zero-shot setting. We investigate 9 foundational image-text models on a diverse set of video tasks that include video action recognition (video AR), video retrieval (video RT), video question answering (video QA), video multiple choice (video MC) and video captioning (video CP). Our experiments show that image-text models exhibit impressive performance on video AR, video RT and video MC. Furthermore, they perform moderately on video captioning and poorly on video QA. These findings shed a light on the benefits of adapting foundational image-text models to an array of video tasks while avoiding the costly pretraining step.

{{</citation>}}


### (32/67) WAIT: Feature Warping for Animation to Illustration video Translation using GANs (Samet Hicsonmez et al., 2023)

{{<citation>}}

Samet Hicsonmez, Nermin Samet, Fidan Samet, Oguz Bakir, Emre Akbas, Pinar Duygulu. (2023)  
**WAIT: Feature Warping for Animation to Illustration video Translation using GANs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04901v1)  

---


**ABSTRACT**  
In this paper, we explore a new domain for video-to-video translation. Motivated by the availability of animation movies that are adopted from illustrated books for children, we aim to stylize these videos with the style of the original illustrations. Current state-of-the-art video-to-video translation models rely on having a video sequence or a single style image to stylize an input video. We introduce a new problem for video stylizing where an unordered set of images are used. This is a challenging task for two reasons: i) we do not have the advantage of temporal consistency as in video sequences; ii) it is more difficult to obtain consistent styles for video frames from a set of unordered images compared to using a single image.   Most of the video-to-video translation methods are built on an image-to-image translation model, and integrate additional networks such as optical flow, or temporal predictors to capture temporal relations. These additional networks make the model training and inference complicated and slow down the process. To ensure temporal coherency in video-to-video style transfer, we propose a new generator network with feature warping layers which overcomes the limitations of the previous methods. We show the effectiveness of our method on three datasets both qualitatively and quantitatively. Code and pretrained models are available at https://github.com/giddyyupp/wait.

{{</citation>}}


### (33/67) Federated Self-Supervised Learning of Monocular Depth Estimators for Autonomous Vehicles (Elton F. de S. Soares et al., 2023)

{{<citation>}}

Elton F. de S. Soares, Carlos Alberto V. Campos. (2023)  
**Federated Self-Supervised Learning of Monocular Depth Estimators for Autonomous Vehicles**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-DC, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.04837v1)  

---


**ABSTRACT**  
Image-based depth estimation has gained significant attention in recent research on computer vision for autonomous vehicles in intelligent transportation systems. This focus stems from its cost-effectiveness and wide range of potential applications. Unlike binocular depth estimation methods that require two fixed cameras, monocular depth estimation methods only rely on a single camera, making them highly versatile. While state-of-the-art approaches for this task leverage self-supervised learning of deep neural networks in conjunction with tasks like pose estimation and semantic segmentation, none of them have explored the combination of federated learning and self-supervision to train models using unlabeled and private data captured by autonomous vehicles. The utilization of federated learning offers notable benefits, including enhanced privacy protection, reduced network consumption, and improved resilience to connectivity issues. To address this gap, we propose FedSCDepth, a novel method that combines federated learning and deep self-supervision to enable the learning of monocular depth estimators with comparable effectiveness and superior efficiency compared to the current state-of-the-art methods. Our evaluation experiments conducted on Eigen's Split of the KITTI dataset demonstrate that our proposed method achieves near state-of-the-art performance, with a test loss below 0.13 and requiring, on average, only 1.5k training steps and up to 0.415 GB of weight data transfer per autonomous vehicle on each round.

{{</citation>}}


### (34/67) Fully Sparse Long Range 3D Object Detection Using Range Experts and Multimodal Virtual Points (Ajinkya Khoche et al., 2023)

{{<citation>}}

Ajinkya Khoche, Laura Pereira Sánchez, Nazre Batool, Sina Sharif Mansouri, Patric Jensfelt. (2023)  
**Fully Sparse Long Range 3D Object Detection Using Range Experts and Multimodal Virtual Points**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.04800v1)  

---


**ABSTRACT**  
3D object detection at long-range is crucial for ensuring the safety and efficiency of self-driving cars, allowing them to accurately perceive and react to objects, obstacles, and potential hazards from a distance. But most current state-of-the-art LiDAR based methods are limited by the sparsity of range sensors, which generates a form of domain gap between points closer to and farther away from the ego vehicle. Another related problem is the label imbalance for faraway objects, which inhibits the performance of Deep Neural Networks at long-range. Although image features could be beneficial for long-range detections, and some recently proposed multimodal methods incorporate image features, they do not scale well computationally at long ranges or are limited by depth estimation accuracy. To address the above limitations, we propose to combine two LiDAR based 3D detection networks, one specializing at near to mid-range objects, and one at long-range 3D detection. To train a detector at long range under a scarce label regime, we further propose to weigh the loss according to the labelled objects' distance from ego vehicle. To mitigate the LiDAR sparsity issue, we leverage Multimodal Virtual Points (MVP), an image based depth completion algorithm, to enrich our data with virtual points. Our method, combining two range experts trained with MVP, which we refer to as RangeFSD, achieves state-of-the-art performance on the Argoverse2 (AV2) dataset, with improvements at long range. The code will be released soon.

{{</citation>}}


### (35/67) IPMix: Label-Preserving Data Augmentation Method for Training Robust Classifiers (Zhenglin Huang et al., 2023)

{{<citation>}}

Zhenglin Huang, Xianan Bao, Na Zhang, Qingqi Zhang, Xiaomei Tu, Biao Wu, Xi Yang. (2023)  
**IPMix: Label-Preserving Data Augmentation Method for Training Robust Classifiers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.04780v1)  

---


**ABSTRACT**  
Data augmentation has been proven effective for training high-accuracy convolutional neural network classifiers by preventing overfitting. However, building deep neural networks in real-world scenarios requires not only high accuracy on clean data but also robustness when data distributions shift. While prior methods have proposed that there is a trade-off between accuracy and robustness, we propose IPMix, a simple data augmentation approach to improve robustness without hurting clean accuracy. IPMix integrates three levels of data augmentation (image-level, patch-level, and pixel-level) into a coherent and label-preserving technique to increase the diversity of training data with limited computational overhead. To further improve the robustness, IPMix introduces structural complexity at different levels to generate more diverse images and adopts the random mixing method for multi-scale information fusion. Experiments demonstrate that IPMix outperforms state-of-the-art corruption robustness on CIFAR-C and ImageNet-C. In addition, we show that IPMix also significantly improves the other safety measures, including robustness to adversarial perturbations, calibration, prediction consistency, and anomaly detection, achieving state-of-the-art or comparable results on several benchmarks, including ImageNet-R, ImageNet-A, and ImageNet-O.

{{</citation>}}


### (36/67) Towards Dynamic and Small Objects Refinement for Unsupervised Domain Adaptative Nighttime Semantic Segmentation (Jingyi Pan et al., 2023)

{{<citation>}}

Jingyi Pan, Sihang Li, Yucheng Chen, Jinjing Zhu, Lin Wang. (2023)  
**Towards Dynamic and Small Objects Refinement for Unsupervised Domain Adaptative Nighttime Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.04747v1)  

---


**ABSTRACT**  
Nighttime semantic segmentation is essential for various applications, e.g., autonomous driving, which often faces challenges due to poor illumination and the lack of well-annotated datasets. Unsupervised domain adaptation (UDA) has shown potential for addressing the challenges and achieved remarkable results for nighttime semantic segmentation. However, existing methods still face limitations in 1) their reliance on style transfer or relighting models, which struggle to generalize to complex nighttime environments, and 2) their ignorance of dynamic and small objects like vehicles and traffic signs, which are difficult to be directly learned from other domains. This paper proposes a novel UDA method that refines both label and feature levels for dynamic and small objects for nighttime semantic segmentation. First, we propose a dynamic and small object refinement module to complement the knowledge of dynamic and small objects from the source domain to target nighttime domain. These dynamic and small objects are normally context-inconsistent in under-exposed conditions. Then, we design a feature prototype alignment module to reduce the domain gap by deploying contrastive learning between features and prototypes of the same class from different domains, while re-weighting the categories of dynamic and small objects. Extensive experiments on four benchmark datasets demonstrate that our method outperforms prior arts by a large margin for nighttime segmentation. Project page: https://rorisis.github.io/DSRNSS/.

{{</citation>}}


### (37/67) Memory-Constrained Semantic Segmentation for Ultra-High Resolution UAV Imagery (Qi Li et al., 2023)

{{<citation>}}

Qi Li, Jiaxin Cai, Yuanlong Yu, Jason Gu, Jia Pan, Wenxi Liu. (2023)  
**Memory-Constrained Semantic Segmentation for Ultra-High Resolution UAV Imagery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.04721v1)  

---


**ABSTRACT**  
Amidst the swift advancements in photography and sensor technologies, high-definition cameras have become commonplace in the deployment of Unmanned Aerial Vehicles (UAVs) for diverse operational purposes. Within the domain of UAV imagery analysis, the segmentation of ultra-high resolution images emerges as a substantial and intricate challenge, especially when grappling with the constraints imposed by GPU memory-restricted computational devices. This paper delves into the intricate problem of achieving efficient and effective segmentation of ultra-high resolution UAV imagery, while operating under stringent GPU memory limitation. The strategy of existing approaches is to downscale the images to achieve computationally efficient segmentation. However, this strategy tends to overlook smaller, thinner, and curvilinear regions. To address this problem, we propose a GPU memory-efficient and effective framework for local inference without accessing the context beyond local patches. In particular, we introduce a novel spatial-guided high-resolution query module, which predicts pixel-wise segmentation results with high quality only by querying nearest latent embeddings with the guidance of high-resolution information. Additionally, we present an efficient memory-based interaction scheme to correct potential semantic bias of the underlying high-resolution information by associating cross-image contextual semantics. For evaluation of our approach, we perform comprehensive experiments over public benchmarks and achieve superior performance under both conditions of small and large GPU memory usage limitations. We will release the model and codes in the future.

{{</citation>}}


### (38/67) Reinforced UI Instruction Grounding: Towards a Generic UI Task Automation API (Zhizheng Zhang et al., 2023)

{{<citation>}}

Zhizheng Zhang, Wenxuan Xie, Xiaoyi Zhang, Yan Lu. (2023)  
**Reinforced UI Instruction Grounding: Towards a Generic UI Task Automation API**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04716v1)  

---


**ABSTRACT**  
Recent popularity of Large Language Models (LLMs) has opened countless possibilities in automating numerous AI tasks by connecting LLMs to various domain-specific models or APIs, where LLMs serve as dispatchers while domain-specific models or APIs are action executors. Despite the vast numbers of domain-specific models/APIs, they still struggle to comprehensively cover super diverse automation demands in the interaction between human and User Interfaces (UIs). In this work, we build a multimodal model to ground natural language instructions in given UI screenshots as a generic UI task automation executor. This metadata-free grounding model, consisting of a visual encoder and a language decoder, is first pretrained on well studied document understanding tasks and then learns to decode spatial information from UI screenshots in a promptable way. To facilitate the exploitation of image-to-text pretrained knowledge, we follow the pixel-to-sequence paradigm to predict geometric coordinates in a sequence of tokens using a language decoder. We further propose an innovative Reinforcement Learning (RL) based algorithm to supervise the tokens in such sequence jointly with visually semantic metrics, which effectively strengthens the spatial decoding capability of the pixel-to-sequence paradigm. Extensive experiments demonstrate our proposed reinforced UI instruction grounding model outperforms the state-of-the-art methods by a clear margin and shows the potential as a generic UI task automation API.

{{</citation>}}


### (39/67) Generalized Robust Test-Time Adaptation in Continuous Dynamic Scenarios (Shuang Li et al., 2023)

{{<citation>}}

Shuang Li, Longhui Yuan, Binhui Xie, Tao Yang. (2023)  
**Generalized Robust Test-Time Adaptation in Continuous Dynamic Scenarios**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.04714v1)  

---


**ABSTRACT**  
Test-time adaptation (TTA) adapts the pre-trained models to test distributions during the inference phase exclusively employing unlabeled test data streams, which holds great value for the deployment of models in real-world applications. Numerous studies have achieved promising performance on simplistic test streams, characterized by independently and uniformly sampled test data originating from a fixed target data distribution. However, these methods frequently prove ineffective in practical scenarios, where both continual covariate shift and continual label shift occur simultaneously, i.e., data and label distributions change concurrently and continually over time. In this study, a more challenging Practical Test-Time Adaptation (PTTA) setup is introduced, which takes into account the concurrent presence of continual covariate shift and continual label shift, and we propose a Generalized Robust Test-Time Adaptation (GRoTTA) method to effectively address the difficult problem. We start by steadily adapting the model through Robust Parameter Adaptation to make balanced predictions for test samples. To be specific, firstly, the effects of continual label shift are eliminated by enforcing the model to learn from a uniform label distribution and introducing recalibration of batch normalization to ensure stability. Secondly, the continual covariate shift is alleviated by employing a source knowledge regularization with the teacher-student model to update parameters. Considering the potential information in the test stream, we further refine the balanced predictions by Bias-Guided Output Adaptation, which exploits latent structure in the feature space and is adaptive to the imbalanced label distribution. Extensive experiments demonstrate GRoTTA outperforms the existing competitors by a large margin under PTTA setting, rendering it highly conducive for adoption in real-world applications.

{{</citation>}}


### (40/67) Tree-GPT: Modular Large Language Model Expert System for Forest Remote Sensing Image Understanding and Interactive Analysis (Siqi Du et al., 2023)

{{<citation>}}

Siqi Du, Shengjun Tang, Weixi Wang, Xiaoming Li, Renzhong Guo. (2023)  
**Tree-GPT: Modular Large Language Model Expert System for Forest Remote Sensing Image Understanding and Interactive Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.04698v1)  

---


**ABSTRACT**  
This paper introduces a novel framework, Tree-GPT, which incorporates Large Language Models (LLMs) into the forestry remote sensing data workflow, thereby enhancing the efficiency of data analysis. Currently, LLMs are unable to extract or comprehend information from images and may generate inaccurate text due to a lack of domain knowledge, limiting their use in forestry data analysis. To address this issue, we propose a modular LLM expert system, Tree-GPT, that integrates image understanding modules, domain knowledge bases, and toolchains. This empowers LLMs with the ability to comprehend images, acquire accurate knowledge, generate code, and perform data analysis in a local environment. Specifically, the image understanding module extracts structured information from forest remote sensing images by utilizing automatic or interactive generation of prompts to guide the Segment Anything Model (SAM) in generating and selecting optimal tree segmentation results. The system then calculates tree structural parameters based on these results and stores them in a database. Upon receiving a specific natural language instruction, the LLM generates code based on a thought chain to accomplish the analysis task. The code is then executed by an LLM agent in a local environment and . For ecological parameter calculations, the system retrieves the corresponding knowledge from the knowledge base and inputs it into the LLM to guide the generation of accurate code. We tested this system on several tasks, including Search, Visualization, and Machine Learning Analysis. The prototype system performed well, demonstrating the potential for dynamic usage of LLMs in forestry research and environmental sciences.

{{</citation>}}


### (41/67) SeeDS: Semantic Separable Diffusion Synthesizer for Zero-shot Food Detection (Pengfei Zhou et al., 2023)

{{<citation>}}

Pengfei Zhou, Weiqing Min, Yang Zhang, Jiajun Song, Ying Jin, Shuqiang Jiang. (2023)  
**SeeDS: Semantic Separable Diffusion Synthesizer for Zero-shot Food Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.04689v1)  

---


**ABSTRACT**  
Food detection is becoming a fundamental task in food computing that supports various multimedia applications, including food recommendation and dietary monitoring. To deal with real-world scenarios, food detection needs to localize and recognize novel food objects that are not seen during training, demanding Zero-Shot Detection (ZSD). However, the complexity of semantic attributes and intra-class feature diversity poses challenges for ZSD methods in distinguishing fine-grained food classes. To tackle this, we propose the Semantic Separable Diffusion Synthesizer (SeeDS) framework for Zero-Shot Food Detection (ZSFD). SeeDS consists of two modules: a Semantic Separable Synthesizing Module (S$^3$M) and a Region Feature Denoising Diffusion Model (RFDDM). The S$^3$M learns the disentangled semantic representation for complex food attributes from ingredients and cuisines, and synthesizes discriminative food features via enhanced semantic information. The RFDDM utilizes a novel diffusion model to generate diversified region features and enhances ZSFD via fine-grained synthesized features. Extensive experiments show the state-of-the-art ZSFD performance of our proposed method on two food datasets, ZSFooD and UECFOOD-256. Moreover, SeeDS also maintains effectiveness on general ZSD datasets, PASCAL VOC and MS COCO. The code and dataset can be found at https://github.com/LanceZPF/SeeDS.

{{</citation>}}


### (42/67) Understanding and Improving Adversarial Attacks on Latent Diffusion Model (Boyang Zheng et al., 2023)

{{<citation>}}

Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu. (2023)  
**Understanding and Improving Adversarial Attacks on Latent Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2310.04687v1)  

---


**ABSTRACT**  
Latent Diffusion Model (LDM) has emerged as a leading tool in image generation, particularly with its capability in few-shot generation. This capability also presents risks, notably in unauthorized artwork replication and misinformation generation. In response, adversarial attacks have been designed to safeguard personal images from being used as reference data. However, existing adversarial attacks are predominantly empirical, lacking a solid theoretical foundation. In this paper, we introduce a comprehensive theoretical framework for understanding adversarial attacks on LDM. Based on the framework, we propose a novel adversarial attack that exploits a unified target to guide the adversarial attack both in the forward and the reverse process of LDM. We provide empirical evidences that our method overcomes the offset problem of the optimization of adversarial attacks in existing methods. Through rigorous experiments, our findings demonstrate that our method outperforms current attacks and is able to generalize over different state-of-the-art few-shot generation pipelines based on LDM. Our method can serve as a stronger and efficient tool for people exposed to the risk of data privacy and security to protect themselves in the new era of powerful generative models. The code is available on GitHub: https://github.com/CaradryanLiang/ImprovedAdvDM.git.

{{</citation>}}


### (43/67) EasyPhoto: Your Smart AI Photo Generator (Ziheng Wu et al., 2023)

{{<citation>}}

Ziheng Wu, Jiaqi Xu, Xinyi Zou, Kunzhe Huang, Xing Shi, Jun Huang. (2023)  
**EasyPhoto: Your Smart AI Photo Generator**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04672v1)  

---


**ABSTRACT**  
Stable Diffusion web UI (SD-WebUI) is a comprehensive project that provides a browser interface based on Gradio library for Stable Diffusion models. In this paper, We propose a novel WebUI plugin called EasyPhoto, which enables the generation of AI portraits. By training a digital doppelganger of a specific user ID using 5 to 20 relevant images, the finetuned model (according to the trained LoRA model) allows for the generation of AI photos using arbitrary templates. Our current implementation supports the modification of multiple persons and different photo styles. Furthermore, we allow users to generate fantastic template image with the strong SDXL model, enhancing EasyPhoto's capabilities to deliver more diverse and satisfactory results. The source code for EasyPhoto is available at: https://github.com/aigc-apps/sd-webui-EasyPhoto. We also support a webui-free version by using diffusers: https://github.com/aigc-apps/EasyPhoto. We are continuously enhancing our efforts to expand the EasyPhoto pipeline, making it suitable for any identification (not limited to just the face), and we enthusiastically welcome any intriguing ideas or suggestions.

{{</citation>}}


### (44/67) Visual Abductive Reasoning Meets Driving Hazard Prediction: Problem Formulation and Dataset (Korawat Charoenpitaks et al., 2023)

{{<citation>}}

Korawat Charoenpitaks, Van-Quang Nguyen, Masanori Suganuma, Masahiro Takahashi, Ryoma Niihara, Takayuki Okatani. (2023)  
**Visual Abductive Reasoning Meets Driving Hazard Prediction: Problem Formulation and Dataset**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.04671v1)  

---


**ABSTRACT**  
This paper addresses the problem of predicting hazards that drivers may encounter while driving a car. We formulate it as a task of anticipating impending accidents using a single input image captured by car dashcams. Unlike existing approaches to driving hazard prediction that rely on computational simulations or anomaly detection from videos, this study focuses on high-level inference from static images. The problem needs predicting and reasoning about future events based on uncertain observations, which falls under visual abductive reasoning. To enable research in this understudied area, a new dataset named the DHPR (Driving Hazard Prediction and Reasoning) dataset is created. The dataset consists of 15K dashcam images of street scenes, and each image is associated with a tuple containing car speed, a hypothesized hazard description, and visual entities present in the scene. These are annotated by human annotators, who identify risky scenes and provide descriptions of potential accidents that could occur a few seconds later. We present several baseline methods and evaluate their performance on our dataset, identifying remaining issues and discussing future directions. This study contributes to the field by introducing a novel problem formulation and dataset, enabling researchers to explore the potential of multi-modal AI for driving hazard prediction.

{{</citation>}}


## cs.CY (2)



### (45/67) Generative AI May Prefer to Present National-level Characteristics of Cities Based on Stereotypical Geographic Impressions at the Continental Level (Shan Ye, 2023)

{{<citation>}}

Shan Ye. (2023)  
**Generative AI May Prefer to Present National-level Characteristics of Cities Based on Stereotypical Geographic Impressions at the Continental Level**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.04897v1)  

---


**ABSTRACT**  
A simple experiment was conducted to test the ability of the Chinese-based generative artificial intelligence (AI) platform, Wenxin Yige, to render images of urban street views of different countries. The study found that images generated by this AI platform may contain continental-level stereotypes in terms of showing the level of economic development and modernization. Street view images generated from Wenxin Yige do not adequately represent the diverse range of urban landscapes found across different nations. Using these generated images for geography education or outreach initiatives could inadvertently strengthen people's existing stereotypical views about individual countries.

{{</citation>}}


### (46/67) PaperCard for Reporting Machine Assistance in Academic Writing (Won Ik Cho et al., 2023)

{{<citation>}}

Won Ik Cho, Eunjung Cho, Kyunghyun Cho. (2023)  
**PaperCard for Reporting Machine Assistance in Academic Writing**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.04824v1)  

---


**ABSTRACT**  
Academic writing process has benefited from various technological developments over the years including search engines, automatic translators, and editing tools that review grammar and spelling mistakes. They have enabled human writers to become more efficient in writing academic papers, for example by helping with finding relevant literature more effectively and polishing texts. While these developments have so far played a relatively assistive role, recent advances in large-scale language models (LLMs) have enabled LLMs to play a more major role in the writing process, such as coming up with research questions and generating key contents. This raises critical questions surrounding the concept of authorship in academia. ChatGPT, a question-answering system released by OpenAI in November 2022, has demonstrated a range of capabilities that could be utilised in producing academic papers. The academic community will have to address relevant pressing questions, including whether Artificial Intelligence (AI) should be merited authorship if it made significant contributions in the writing process, or whether its use should be restricted such that human authorship would not be undermined. In this paper, we aim to address such questions, and propose a framework we name "PaperCard", a documentation for human authors to transparently declare the use of AI in their writing process.

{{</citation>}}


## cs.IR (6)



### (47/67) Commercialized Generative AI: A Critical Study of the Feasibility and Ethics of Generating Native Advertising Using Large Language Models in Conversational Web Search (Ines Zelch et al., 2023)

{{<citation>}}

Ines Zelch, Matthias Hagen, Martin Potthast. (2023)  
**Commercialized Generative AI: A Critical Study of the Feasibility and Ethics of Generating Native Advertising Using Large Language Models in Conversational Web Search**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: AI, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.04892v1)  

---


**ABSTRACT**  
How will generative AI pay for itself? Unless charging users for access, selling advertising is the only alternative. Especially in the multi-billion dollar web search market with ads as the main source of revenue, the introduction of a subscription model seems unlikely. The recent disruption of search by generative large language models could thus ultimately be accompanied by generated ads. Our concern is that the commercialization of generative AI in general and large language models in particular could lead to native advertising in the form of quite subtle brand or product placements. In web search, the evolution of search engine results pages (SERPs) from traditional lists of ``ten blue links'' (lists SERPs) to generated text with web page references (text SERPs) may further blur the line between advertising-based and organic search results, making it difficult for users to distinguish between the two, depending on how advertising is integrated and disclosed. To raise awareness of this potential development, we conduct a pilot study analyzing the capabilities of current large language models to blend ads with organic search results. Although the models still struggle to subtly frame ads in an unrelated context, their potential is evident when integrating ads into related topics which calls for further investigation.

{{</citation>}}


### (48/67) Hybrid Recommendation System using Graph Neural Network and BERT Embeddings (Shashidhar Reddy Javaji et al., 2023)

{{<citation>}}

Shashidhar Reddy Javaji, Krutika Sarode. (2023)  
**Hybrid Recommendation System using Graph Neural Network and BERT Embeddings**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: BERT, Embedding, GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.04878v1)  

---


**ABSTRACT**  
Recommender systems have emerged as a crucial component of the modern web ecosystem. The effectiveness and accuracy of such systems are critical for providing users with personalized recommendations that meet their specific interests and needs. In this paper, we introduce a novel model that utilizes a Graph Neural Network (GNN) in conjunction with sentence transformer embeddings to predict anime recommendations for different users. Our model employs the task of link prediction to create a recommendation system that considers both the features of anime and user interactions with different anime. The hybridization of the GNN and transformer embeddings enables us to capture both inter-level and intra-level features of anime data.Our model not only recommends anime to users but also predicts the rating a specific user would give to an anime. We utilize the GraphSAGE network for model building and weighted root mean square error (RMSE) to evaluate the performance of the model. Our approach has the potential to significantly enhance the accuracy and effectiveness of anime recommendation systems and can be extended to other domains that require personalized recommendations.

{{</citation>}}


### (49/67) ForeSeer: Product Aspect Forecasting Using Temporal Graph Embedding (Zixuan Liu et al., 2023)

{{<citation>}}

Zixuan Liu, Gaurush Hiranandani, Kun Qian, Eddie W. Huang, Yi Xu, Belinda Zeng, Karthik Subbian, Sheng Wang. (2023)  
**ForeSeer: Product Aspect Forecasting Using Temporal Graph Embedding**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.IR  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.04865v1)  

---


**ABSTRACT**  
Developing text mining approaches to mine aspects from customer reviews has been well-studied due to its importance in understanding customer needs and product attributes. In contrast, it remains unclear how to predict the future emerging aspects of a new product that currently has little review information. This task, which we named product aspect forecasting, is critical for recommending new products, but also challenging because of the missing reviews. Here, we propose ForeSeer, a novel textual mining and product embedding approach progressively trained on temporal product graphs for this novel product aspect forecasting task. ForeSeer transfers reviews from similar products on a large product graph and exploits these reviews to predict aspects that might emerge in future reviews. A key novelty of our method is to jointly provide review, product, and aspect embeddings that are both time-sensitive and less affected by extremely imbalanced aspect frequencies. We evaluated ForeSeer on a real-world product review system containing 11,536,382 reviews and 11,000 products over 3 years. We observe that ForeSeer substantially outperformed existing approaches with at least 49.1\% AUPRC improvement under the real setting where aspect associations are not given. ForeSeer further improves future link prediction on the product graph and the review aspect association prediction. Collectively, Foreseer offers a novel framework for review forecasting by effectively integrating review text, product network, and temporal information, opening up new avenues for online shopping recommendation and e-commerce applications.

{{</citation>}}


### (50/67) Investigating the Influence of Legal Case Retrieval Systems on Users' Decision Process (Beining Wang et al., 2023)

{{<citation>}}

Beining Wang, Ruizhe Zhang, Yueyue Wu, Qingyao Ai, Min Zhang, Yiqun Liu. (2023)  
**Investigating the Influence of Legal Case Retrieval Systems on Users' Decision Process**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2310.04735v1)  

---


**ABSTRACT**  
Given a specific query case, legal case retrieval systems aim to retrieve a set of case documents relevant to the case at hand. Previous studies on user behavior analysis have shown that information retrieval (IR) systems can significantly influence users' decisions by presenting results in varying orders and formats. However, whether such influence exists in legal case retrieval remains largely unknown. This study presents the first investigation into the influence of legal case retrieval systems on the decision-making process of legal users. We conducted an online user study involving more than ninety participants, and our findings suggest that the result distribution of legal case retrieval systems indeed affect users' judgements on the sentences in cases. Notably, when users are presented with biased results that involve harsher sentences, they tend to impose harsher sentences on the current case as well. This research highlights the importance of optimizing the unbiasedness of legal case retrieval systems.

{{</citation>}}


### (51/67) DORIS-MAE: Scientific Document Retrieval using Multi-level Aspect-based Queries (Jianyou Wang et al., 2023)

{{<citation>}}

Jianyou Wang, Kaicheng Wang, Xiaoyue Wang, Prudhviraj Naidu, Leon Bergen, Ramamohan Paturi. (2023)  
**DORIS-MAE: Scientific Document Retrieval using Multi-level Aspect-based Queries**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.04678v1)  

---


**ABSTRACT**  
In scientific research, the ability to effectively retrieve relevant documents based on complex, multifaceted queries is critical. Existing evaluation datasets for this task are limited, primarily due to the high cost and effort required to annotate resources that effectively represent complex queries. To address this, we propose a novel task, Scientific DOcument Retrieval using Multi-level Aspect-based quEries (DORIS-MAE), which is designed to handle the complex nature of user queries in scientific research. We developed a benchmark dataset within the field of computer science, consisting of 100 human-authored complex query cases. For each complex query, we assembled a collection of 100 relevant documents and produced annotated relevance scores for ranking them. Recognizing the significant labor of expert annotation, we also introduce Anno-GPT, a scalable framework for validating the performance of Large Language Models (LLMs) on expert-level dataset annotation tasks. LLM annotation of the DORIS-MAE dataset resulted in a 500x reduction in cost, without compromising quality. Furthermore, due to the multi-tiered structure of these complex queries, the DORIS-MAE dataset can be extended to over 4,000 sub-query test cases without requiring additional annotation. We evaluated 17 recent retrieval methods on DORIS-MAE, observing notable performance drops compared to traditional datasets. This highlights the need for better approaches to handle complex, multifaceted queries in scientific research. Our dataset and codebase are available at https://github.com/Real-Doris-Mae/Doris-Mae-Dataset.

{{</citation>}}


### (52/67) Unbiased and Robust: External Attention-enhanced Graph Contrastive Learning for Cross-domain Sequential Recommendation (Xinhua Wang et al., 2023)

{{<citation>}}

Xinhua Wang, Houping Yue, Zizheng Wang, Liancheng Xu, Jinyu Zhang. (2023)  
**Unbiased and Robust: External Attention-enhanced Graph Contrastive Learning for Cross-domain Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Attention, Contrastive Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.04633v1)  

---


**ABSTRACT**  
Cross-domain sequential recommenders (CSRs) are gaining considerable research attention as they can capture user sequential preference by leveraging side information from multiple domains. However, these works typically follow an ideal setup, i.e., different domains obey similar data distribution, which ignores the bias brought by asymmetric interaction densities (a.k.a. the inter-domain density bias). Besides, the frequently adopted mechanism (e.g., the self-attention network) in sequence encoder only focuses on the interactions within a local view, which overlooks the global correlations between different training batches. To this end, we propose an External Attention-enhanced Graph Contrastive Learning framework, namely EA-GCL. Specifically, to remove the impact of the inter-domain density bias, an auxiliary Self-Supervised Learning (SSL) task is attached to the traditional graph encoder under a multi-task learning manner. To robustly capture users' behavioral patterns, we develop an external attention-based sequence encoder that contains an MLP-based memory-sharing structure. Unlike the self-attention mechanism, such a structure can effectively alleviate the bias interference from the batch-based training scheme. Extensive experiments on two real-world datasets demonstrate that EA-GCL outperforms several state-of-the-art baselines on CSR tasks. The source codes and relevant datasets are available at https://github.com/HoupingY/EA-GCL.

{{</citation>}}


## cs.FL (1)



### (53/67) Lemur: Integrating Large Language Models in Automated Program Verification (Haoze Wu et al., 2023)

{{<citation>}}

Haoze Wu, Clark Barrett, Nina Narodytska. (2023)  
**Lemur: Integrating Large Language Models in Automated Program Verification**  

---
Primary Category: cs.FL  
Categories: cs-AI, cs-FL, cs-LG, cs-LO, cs.FL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.04870v1)  

---


**ABSTRACT**  
The demonstrated code-understanding capability of LLMs raises the question of whether they can be used for automated program verification, a task that typically demands high-level abstract reasoning about program properties that is challenging for verification tools. We propose a general methodology to combine the power of LLMs and automated reasoners for automated program verification. We formally describe this methodology as a set of derivation rules and prove its soundness. We instantiate the calculus as a sound automated verification procedure, which led to practical improvements on a set of synthetic and competition benchmarks.

{{</citation>}}


## cs.HC (3)



### (54/67) ILuvUI: Instruction-tuned LangUage-Vision modeling of UIs from Machine Conversations (Yue Jiang et al., 2023)

{{<citation>}}

Yue Jiang, Eldon Schoop, Amanda Swearngin, Jeffrey Nichols. (2023)  
**ILuvUI: Instruction-tuned LangUage-Vision modeling of UIs from Machine Conversations**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-CV, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.04869v1)  

---


**ABSTRACT**  
Multimodal Vision-Language Models (VLMs) enable powerful applications from their fused understanding of images and language, but many perform poorly on UI tasks due to the lack of UI training data. In this paper, we adapt a recipe for generating paired text-image training data for VLMs to the UI domain by combining existing pixel-based methods with a Large Language Model (LLM). Unlike prior art, our method requires no human-provided annotations, and it can be applied to any dataset of UI screenshots. We generate a dataset of 335K conversational examples paired with UIs that cover Q&A, UI descriptions, and planning, and use it to fine-tune a conversational VLM for UI tasks. To assess the performance of our model, we benchmark it on UI element detection tasks, evaluate response quality, and showcase its applicability to multi-step UI navigation and planning.

{{</citation>}}


### (55/67) Validating Drone Trust Testing in Navigation Deviation Cases in Simulation (Zahra Rezaei Khavas et al., 2023)

{{<citation>}}

Zahra Rezaei Khavas, Edwin Meriaux, Amin Majdi, Paul Robinette. (2023)  
**Validating Drone Trust Testing in Navigation Deviation Cases in Simulation**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.04794v1)  

---


**ABSTRACT**  
Developing videos for trust testing is very time-consuming, expensive and potentially dangerous. For trust tests, it requires a person to be flying the drone while another might be filming. The drones can be very expensive and if something goes wrong the costs might be very high. In previous work, we have looked at how collisions and basic communication loss can be accurately modeled in simulation and to be able to generate the same trust results from users. That work looked at two specific cases using two drones, but to expand upon this in other cases more testing is required. This paper looks to propose how to test and evaluate the change in user's trust of a drone when it is experiencing path deviation in simulation. If the environment is very realistic can simulations be a good alternative to real life videos for trust testing when there is path deviation? This deviation can occur due to the physical conditions of the space, faulty piloting, or communication loss.

{{</citation>}}


### (56/67) Trust in Generative AI among students: An Exploratory Study (Matin Amoozadeh et al., 2023)

{{<citation>}}

Matin Amoozadeh, David Daniels, Daye Nam, Stella Chen, Michael Hilton, Sruti Srinivasa Ragavan, Mohammad Amin Alipour. (2023)  
**Trust in Generative AI among students: An Exploratory Study**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.04631v1)  

---


**ABSTRACT**  
Generative artificial systems (GenAI) have experienced exponential growth in the past couple of years. These systems offer exciting capabilities, such as generating programs, that students can well utilize for their learning. Among many dimensions that might affect the effective adoption of GenAI, in this paper, we investigate students' \textit{trust}. Trust in GenAI influences the extent to which students adopt GenAI, in turn affecting their learning. In this study, we surveyed 253 students at two large universities to understand how much they trust \genai tools and their feedback on how GenAI impacts their performance in CS courses. Our results show that students have different levels of trust in GenAI. We also observe different levels of confidence and motivation, highlighting the need for further understanding of factors impacting trust.

{{</citation>}}


## eess.AS (3)



### (57/67) Conditional Diffusion Model for Target Speaker Extraction (Theodor Nguyen et al., 2023)

{{<citation>}}

Theodor Nguyen, Guangzhi Sun, Xianrui Zheng, Chao Zhang, Philip C Woodland. (2023)  
**Conditional Diffusion Model for Target Speaker Extraction**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.04791v1)  

---


**ABSTRACT**  
We propose DiffSpEx, a generative target speaker extraction method based on score-based generative modelling through stochastic differential equations. DiffSpEx deploys a continuous-time stochastic diffusion process in the complex short-time Fourier transform domain, starting from the target speaker source and converging to a Gaussian distribution centred on the mixture of sources. For the reverse-time process, a parametrised score function is conditioned on a target speaker embedding to extract the target speaker from the mixture of sources. We utilise ECAPA-TDNN target speaker embeddings and condition the score function alternately on the SDE time embedding and the target speaker embedding. The potential of DiffSpEx is demonstrated with the WSJ0-2mix dataset, achieving an SI-SDR of 12.9 dB and a NISQA score of 3.56. Moreover, we show that fine-tuning a pre-trained DiffSpEx model to a specific speaker further improves performance, enabling personalisation in target speaker extraction.

{{</citation>}}


### (58/67) Multi-objective Progressive Clustering for Semi-supervised Domain Adaptation in Speaker Verification (Ze Li et al., 2023)

{{<citation>}}

Ze Li, Yuke Lin, Ning Jiang, Xiaoyi Qin, Guoqing Zhao, Haiying Wu, Ming Li. (2023)  
**Multi-objective Progressive Clustering for Semi-supervised Domain Adaptation in Speaker Verification**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2310.04760v1)  

---


**ABSTRACT**  
Utilizing the pseudo-labeling algorithm with large-scale unlabeled data becomes crucial for semi-supervised domain adaptation in speaker verification tasks. In this paper, we propose a novel pseudo-labeling method named Multi-objective Progressive Clustering (MoPC), specifically designed for semi-supervised domain adaptation. Firstly, we utilize limited labeled data from the target domain to derive domain-specific descriptors based on multiple distinct objectives, namely within-graph denoising, intra-class denoising and inter-class denoising. Then, the Infomap algorithm is adopted for embedding clustering, and the descriptors are leveraged to further refine the target domain's pseudo-labels. Moreover, to further improve the quality of pseudo labels, we introduce the subcenter-purification and progressive-merging strategy for label denoising. Our proposed MoPC method achieves 4.95% EER and ranked the 1$^{st}$ place on the evaluation set of VoxSRC 2023 track 3. We also conduct additional experiments on the FFSVC dataset and yield promising results.

{{</citation>}}


### (59/67) Spike-Triggered Contextual Biasing for End-to-End Mandarin Speech Recognition (Kaixun Huang et al., 2023)

{{<citation>}}

Kaixun Huang, Ao Zhang, Binbin Zhang, Tianyi Xu, Xingchen Song, Lei Xie. (2023)  
**Spike-Triggered Contextual Biasing for End-to-End Mandarin Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Bias, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2310.04657v1)  

---


**ABSTRACT**  
The attention-based deep contextual biasing method has been demonstrated to effectively improve the recognition performance of end-to-end automatic speech recognition (ASR) systems on given contextual phrases. However, unlike shallow fusion methods that directly bias the posterior of the ASR model, deep biasing methods implicitly integrate contextual information, making it challenging to control the degree of bias. In this study, we introduce a spike-triggered deep biasing method that simultaneously supports both explicit and implicit bias. Moreover, both bias approaches exhibit significant improvements and can be cascaded with shallow fusion methods for better results. Furthermore, we propose a context sampling enhancement strategy and improve the contextual phrase filtering algorithm. Experiments on the public WenetSpeech Mandarin biased-word dataset show a 32.0% relative CER reduction compared to the baseline model, with an impressively 68.6% relative CER reduction on contextual phrases.

{{</citation>}}


## eess.IV (2)



### (60/67) TransCC: Transformer Network for Coronary Artery CCTA Segmentation (Chenchu Xu et al., 2023)

{{<citation>}}

Chenchu Xu, Meng Li, Xue Wu. (2023)  
**TransCC: Transformer Network for Coronary Artery CCTA Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.04779v1)  

---


**ABSTRACT**  
The accurate segmentation of Coronary Computed Tomography Angiography (CCTA) images holds substantial clinical value for the early detection and treatment of Coronary Heart Disease (CHD). The Transformer, utilizing a self-attention mechanism, has demonstrated commendable performance in the realm of medical image processing. However, challenges persist in coronary segmentation tasks due to (1) the damage to target local structures caused by fixed-size image patch embedding, and (2) the critical role of both global and local features in medical image segmentation tasks.To address these challenges, we propose a deep learning framework, TransCC, that effectively amalgamates the Transformer and convolutional neural networks for CCTA segmentation. Firstly, we introduce a Feature Interaction Extraction (FIE) module designed to capture the characteristics of image patches, thereby circumventing the loss of semantic information inherent in the original method. Secondly, we devise a Multilayer Enhanced Perceptron (MEP) to augment attention to local information within spatial dimensions, serving as a complement to the self-attention mechanism. Experimental results indicate that TransCC outperforms existing methods in segmentation performance, boasting an average Dice coefficient of 0.730 and an average Intersection over Union (IoU) of 0.582. These results underscore the effectiveness of TransCC in CCTA image segmentation.

{{</citation>}}


### (61/67) Metadata-Conditioned Generative Models to Synthesize Anatomically-Plausible 3D Brain MRIs (Wei Peng et al., 2023)

{{<citation>}}

Wei Peng, Tomas Bosschieter, Jiahong Ouyang, Robert Paul, Ehsan Adeli, Qingyu Zhao, Kilian M. Pohl. (2023)  
**Metadata-Conditioned Generative Models to Synthesize Anatomically-Plausible 3D Brain MRIs**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.04630v1)  

---


**ABSTRACT**  
Generative AI models hold great potential in creating synthetic brain MRIs that advance neuroimaging studies by, for example, enriching data diversity. However, the mainstay of AI research only focuses on optimizing the visual quality (such as signal-to-noise ratio) of the synthetic MRIs while lacking insights into their relevance to neuroscience. To gain these insights with respect to T1-weighted MRIs, we first propose a new generative model, BrainSynth, to synthesize metadata-conditioned (e.g., age- and sex-specific) MRIs that achieve state-of-the-art visual quality. We then extend our evaluation with a novel procedure to quantify anatomical plausibility, i.e., how well the synthetic MRIs capture macrostructural properties of brain regions, and how accurately they encode the effects of age and sex. Results indicate that more than half of the brain regions in our synthetic MRIs are anatomically accurate, i.e., with a small effect size between real and synthetic MRIs. Moreover, the anatomical plausibility varies across cortical regions according to their geometric complexity. As is, our synthetic MRIs can significantly improve the training of a Convolutional Neural Network to identify accelerated aging effects in an independent study. These results highlight the opportunities of using generative AI to aid neuroimaging research and point to areas for further improvement.

{{</citation>}}


## cs.SD (2)



### (62/67) VoiceExtender: Short-utterance Text-independent Speaker Verification with Guided Diffusion Model (Yayun He et al., 2023)

{{<citation>}}

Yayun He, Zuheng Kang, Jianzong Wang, Junqing Peng, Jing Xiao. (2023)  
**VoiceExtender: Short-utterance Text-independent Speaker Verification with Guided Diffusion Model**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Speaker Verification  
[Paper Link](http://arxiv.org/abs/2310.04681v1)  

---


**ABSTRACT**  
Speaker verification (SV) performance deteriorates as utterances become shorter. To this end, we propose a new architecture called VoiceExtender which provides a promising solution for improving SV performance when handling short-duration speech signals. We use two guided diffusion models, the built-in and the external speaker embedding (SE) guided diffusion model, both of which utilize a diffusion model-based sample generator that leverages SE guidance to augment the speech features based on a short utterance. Extensive experimental results on the VoxCeleb1 dataset show that our method outperforms the baseline, with relative improvements in equal error rate (EER) of 46.1%, 35.7%, 10.4%, and 5.7% for the short utterance conditions of 0.5, 1.0, 1.5, and 2.0 seconds, respectively.

{{</citation>}}


### (63/67) LauraGPT: Listen, Attend, Understand, and Regenerate Audio with GPT (Jiaming Wang^ et al., 2023)

{{<citation>}}

Jiaming Wang^, Zhihao Du^, Qian Chen, Yunfei Chu, Zhifu Gao, Zerui Li, Kai Hu, Xiaohuan Zhou, Jin Xu, Ziyang Ma, Wen Wang, Siqi Zheng, Chang Zhou, Zhijie Yan, Shiliang Zhang. (2023)  
**LauraGPT: Listen, Attend, Understand, and Regenerate Audio with GPT**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-LG, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2310.04673v1)  

---


**ABSTRACT**  
Generative Pre-trained Transformer (GPT) models have achieved remarkable performance on various natural language processing tasks. However, there has been limited research on applying similar frameworks to audio tasks. Previously proposed large language models for audio tasks either lack sufficient quantitative evaluations, or are limited to tasks for recognizing and understanding audio content, or significantly underperform existing state-of-the-art (SOTA) models. In this paper, we propose LauraGPT, a unified GPT model for audio recognition, understanding, and generation. LauraGPT is a versatile language model that can process both audio and text inputs and generate outputs in either modalities. It can perform a wide range of tasks related to content, semantics, paralinguistics, and audio-signal analysis. Some of its noteworthy tasks include automatic speech recognition, speech-to-text translation, text-to-speech synthesis, machine translation, speech enhancement, automated audio captioning, speech emotion recognition, and spoken language understanding. To achieve this goal, we use a combination of continuous and discrete features for audio. We encode input audio into continuous representations using an audio encoder and decode output audio from discrete codec codes. We then fine-tune a large decoder-only Transformer-based language model on multiple audio-to-text, text-to-audio, audio-to-audio, and text-to-text tasks using a supervised multitask learning approach. Extensive experiments show that LauraGPT achieves competitive or superior performance compared to existing SOTA models on various audio processing benchmarks.

{{</citation>}}


## cs.RO (1)



### (64/67) Terrain-Aware Quadrupedal Locomotion via Reinforcement Learning (Haojie Shi et al., 2023)

{{<citation>}}

Haojie Shi, Qingxu Zhu, Lei Han, Wanchao Chi, Tingguang Li, Max Q. -H. Meng. (2023)  
**Terrain-Aware Quadrupedal Locomotion via Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.04675v1)  

---


**ABSTRACT**  
In nature, legged animals have developed the ability to adapt to challenging terrains through perception, allowing them to plan safe body and foot trajectories in advance, which leads to safe and energy-efficient locomotion. Inspired by this observation, we present a novel approach to train a Deep Neural Network (DNN) policy that integrates proprioceptive and exteroceptive states with a parameterized trajectory generator for quadruped robots to traverse rough terrains. Our key idea is to use a DNN policy that can modify the parameters of the trajectory generator, such as foot height and frequency, to adapt to different terrains. To encourage the robot to step on safe regions and save energy consumption, we propose foot terrain reward and lifting foot height reward, respectively. By incorporating these rewards, our method can learn a safer and more efficient terrain-aware locomotion policy that can move a quadruped robot flexibly in any direction. To evaluate the effectiveness of our approach, we conduct simulation experiments on challenging terrains, including stairs, stepping stones, and poles. The simulation results demonstrate that our approach can successfully direct the robot to traverse such tough terrains in any direction. Furthermore, we validate our method on a real legged robot, which learns to traverse stepping stones with gaps over 25.5cm.

{{</citation>}}


## cs.CR (1)



### (65/67) VLAttack: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models (Ziyi Yin et al., 2023)

{{<citation>}}

Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma. (2023)  
**VLAttack: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs.CR  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2310.04655v1)  

---


**ABSTRACT**  
Vision-Language (VL) pre-trained models have shown their superiority on many multimodal tasks. However, the adversarial robustness of such models has not been fully explored. Existing approaches mainly focus on exploring the adversarial robustness under the white-box setting, which is unrealistic. In this paper, we aim to investigate a new yet practical task to craft image and text perturbations using pre-trained VL models to attack black-box fine-tuned models on different downstream tasks. Towards this end, we propose VLAttack to generate adversarial samples by fusing perturbations of images and texts from both single-modal and multimodal levels. At the single-modal level, we propose a new block-wise similarity attack (BSA) strategy to learn image perturbations for disrupting universal representations. Besides, we adopt an existing text attack strategy to generate text perturbations independent of the image-modal attack. At the multimodal level, we design a novel iterative cross-search attack (ICSA) method to update adversarial image-text pairs periodically, starting with the outputs from the single-modal level. We conduct extensive experiments to attack three widely-used VL pretrained models for six tasks on eight datasets. Experimental results show that the proposed VLAttack framework achieves the highest attack success rates on all tasks compared with state-of-the-art baselines, which reveals a significant blind spot in the deployment of pre-trained VL models. Codes will be released soon.

{{</citation>}}


## cs.DC (1)



### (66/67) DxPU: Large Scale Disaggregated GPU Pools in the Datacenter (Bowen He et al., 2023)

{{<citation>}}

Bowen He, Xiao Zheng, Yuan Chen, Weinan Li, Yajin Zhou, Xin Long, Pengcheng Zhang, Xiaowei Lu, Linquan Jiang, Qiang Liu, Dennis Cai, Xiantao Zhang. (2023)  
**DxPU: Large Scale Disaggregated GPU Pools in the Datacenter**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.04648v1)  

---


**ABSTRACT**  
The rapid adoption of AI and convenience offered by cloud services have resulted in the growing demands for GPUs in the cloud. Generally, GPUs are physically attached to host servers as PCIe devices. However, the fixed assembly combination of host servers and GPUs is extremely inefficient in resource utilization, upgrade, and maintenance. Due to these issues, the GPU disaggregation technique has been proposed to decouple GPUs from host servers. It aggregates GPUs into a pool, and allocates GPU node(s) according to user demands. However, existing GPU disaggregation systems have flaws in software-hardware compatibility, disaggregation scope, and capacity. In this paper, we present a new implementation of datacenter-scale GPU disaggregation, named DxPU. DxPU efficiently solves the above problems and can flexibly allocate as many GPU node(s) as users demand. In order to understand the performance overhead incurred by DxPU, we build up a performance model for AI specific workloads. With the guidance of modeling results, we develop a prototype system, which has been deployed into the datacenter of a leading cloud provider for a test run. We also conduct detailed experiments to evaluate the performance overhead caused by our system. The results show that the overhead of DxPU is less than 10%, compared with native GPU servers, in most of user scenarios.

{{</citation>}}


## q-bio.NC (1)



### (67/67) Do self-supervised speech and language models extract similar representations as human brain? (Peili Chen et al., 2023)

{{<citation>}}

Peili Chen, Linyang He, Li Fu, Lu Fan, Edward F. Chang, Yuanning Li. (2023)  
**Do self-supervised speech and language models extract similar representations as human brain?**  

---
Primary Category: q-bio.NC  
Categories: cs-AI, cs-CL, eess-AS, q-bio-NC, q-bio.NC  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.04645v1)  

---


**ABSTRACT**  
Speech and language models trained through self-supervised learning (SSL) demonstrate strong alignment with brain activity during speech and language perception. However, given their distinct training modalities, it remains unclear whether they correlate with the same neural aspects. We directly address this question by evaluating the brain prediction performance of two representative SSL models, Wav2Vec2.0 and GPT-2, designed for speech and language tasks. Our findings reveal that both models accurately predict speech responses in the auditory cortex, with a significant correlation between their brain predictions. Notably, shared speech contextual information between Wav2Vec2.0 and GPT-2 accounts for the majority of explained variance in brain activity, surpassing static semantic and lower-level acoustic-phonetic information. These results underscore the convergence of speech contextual representations in SSL models and their alignment with the neural network underlying speech perception, offering valuable insights into both SSL models and the neural basis of speech and language processing.

{{</citation>}}
