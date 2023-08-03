---
draft: false
title: "arXiv @ 2023.07.31"
date: 2023-07-31
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.31"
    identifier: arxiv_20230731
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (8)](#cslg-8)
- [cs.CL (8)](#cscl-8)
- [cs.CR (3)](#cscr-3)
- [cs.RO (2)](#csro-2)
- [cs.CV (4)](#cscv-4)
- [cs.IR (1)](#csir-1)
- [eess.IV (1)](#eessiv-1)
- [cs.AI (2)](#csai-2)
- [cs.SD (1)](#cssd-1)
- [cs.MM (1)](#csmm-1)
- [cs.NI (1)](#csni-1)

## cs.LG (8)



### (1/32) ADR-GNN: Advection-Diffusion-Reaction Graph Neural Networks (Moshe Eliasof et al., 2023)

{{<citation>}}

Moshe Eliasof, Eldad Haber, Eran Treister. (2023)  
**ADR-GNN: Advection-Diffusion-Reaction Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.16092v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have shown remarkable success in learning representations for graph-structured data. However, GNNs still face challenges in modeling complex phenomena that involve advection. In this paper, we propose a novel GNN architecture based on Advection-Diffusion-Reaction systems, called ADR-GNN. Advection models the directed transportation of information, diffusion captures the local smoothing of information, and reaction represents the non-linear transformation of information in channels. We provide an analysis of the qualitative behavior of ADR-GNN, that shows the benefit of combining advection, diffusion, and reaction. To demonstrate its efficacy, we evaluate ADR-GNN on real-world node classification and spatio-temporal datasets, and show that it improves or offers competitive performance compared to state-of-the-art networks.

{{</citation>}}


### (2/32) MUSE: Multi-View Contrastive Learning for Heterophilic Graphs (Mengyi Yuan et al., 2023)

{{<citation>}}

Mengyi Yuan, Minjie Chen, Xiang Li. (2023)  
**MUSE: Multi-View Contrastive Learning for Heterophilic Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Contrastive Learning, GNN  
[Paper Link](http://arxiv.org/abs/2307.16026v1)  

---


**ABSTRACT**  
In recent years, self-supervised learning has emerged as a promising approach in addressing the issues of label dependency and poor generalization performance in traditional GNNs. However, existing self-supervised methods have limited effectiveness on heterophilic graphs, due to the homophily assumption that results in similar node representations for connected nodes. In this work, we propose a multi-view contrastive learning model for heterophilic graphs, namely, MUSE. Specifically, we construct two views to capture the information of the ego node and its neighborhood by GNNs enhanced with contrastive learning, respectively. Then we integrate the information from these two views to fuse the node representations. Fusion contrast is utilized to enhance the effectiveness of fused node representations. Further, considering that the influence of neighboring contextual information on information fusion may vary across different ego nodes, we employ an information fusion controller to model the diversity of node-neighborhood similarity at both the local and global levels. Finally, an alternating training scheme is adopted to ensure that unsupervised node representation learning and information fusion controller can mutually reinforce each other. We conduct extensive experiments to evaluate the performance of MUSE on 9 benchmark datasets. Our results show the effectiveness of MUSE on both node classification and clustering tasks.

{{</citation>}}


### (3/32) Graph Condensation for Inductive Node Representation Learning (Xinyi Gao et al., 2023)

{{<citation>}}

Xinyi Gao, Tong Chen, Yilong Zang, Wentao Zhang, Quoc Viet Hung Nguyen, Kai Zheng, Hongzhi Yin. (2023)  
**Graph Condensation for Inductive Node Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.15967v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) encounter significant computational challenges when handling large-scale graphs, which severely restricts their efficacy across diverse applications. To address this limitation, graph condensation has emerged as a promising technique, which constructs a small synthetic graph for efficiently training GNNs while retaining performance. However, due to the topology structure among nodes, graph condensation is limited to condensing only the observed training nodes and their corresponding structure, thus lacking the ability to effectively handle the unseen data. Consequently, the original large graph is still required in the inference stage to perform message passing to inductive nodes, resulting in substantial computational demands. To overcome this issue, we propose mapping-aware graph condensation (MCond), explicitly learning the one-to-many node mapping from original nodes to synthetic nodes to seamlessly integrate new nodes into the synthetic graph for inductive representation learning. This enables direct information propagation on the synthetic graph, which is much more efficient than on the original large graph. Specifically, MCond employs an alternating optimization scheme with innovative loss terms from transductive and inductive perspectives, facilitating the mutual promotion between graph condensation and node mapping learning. Extensive experiments demonstrate the efficacy of our approach in inductive inference. On the Reddit dataset, MCond achieves up to 121.5x inference speedup and 55.9x reduction in storage requirements compared with counterparts based on the original graph.

{{</citation>}}


### (4/32) Towards the Visualization of Aggregated Class Activation Maps to Analyse the Global Contribution of Class Features (Igor Cherepanov et al., 2023)

{{<citation>}}

Igor Cherepanov, David Sessler, Alex Ulmer, Hendrik Lücke-Tieke, Jörn Kohlhammer. (2023)  
**Towards the Visualization of Aggregated Class Activation Maps to Analyse the Global Contribution of Class Features**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-HC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.00710v1)  

---


**ABSTRACT**  
Deep learning (DL) models achieve remarkable performance in classification tasks. However, models with high complexity can not be used in many risk-sensitive applications unless a comprehensible explanation is presented. Explainable artificial intelligence (xAI) focuses on the research to explain the decision-making of AI systems like DL. We extend a recent method of Class Activation Maps (CAMs) which visualizes the importance of each feature of a data sample contributing to the classification. In this paper, we aggregate CAMs from multiple samples to show a global explanation of the classification for semantically structured data. The aggregation allows the analyst to make sophisticated assumptions and analyze them with further drill-down visualizations. Our visual representation for the global CAM illustrates the impact of each feature with a square glyph containing two indicators. The color of the square indicates the classification impact of this feature. The size of the filled square describes the variability of the impact between single samples. For interesting features that require further analysis, a detailed view is necessary that provides the distribution of these values. We propose an interactive histogram to filter samples and refine the CAM to show relevant samples only. Our approach allows an analyst to detect important features of high-dimensional data and derive adjustments to the AI model based on our global explanation visualization.

{{</citation>}}


### (5/32) A Theory for Emergence of Complex Skills in Language Models (Sanjeev Arora et al., 2023)

{{<citation>}}

Sanjeev Arora, Anirudh Goyal. (2023)  
**A Theory for Emergence of Complex Skills in Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2307.15936v1)  

---


**ABSTRACT**  
A major driver of AI products today is the fact that new skills emerge in language models when their parameter set and training corpora are scaled up. This phenomenon is poorly understood, and a mechanistic explanation via mathematical analysis of gradient-based training seems difficult. The current paper takes a different approach, analysing emergence using the famous (and empirical) Scaling Laws of LLMs and a simple statistical framework. Contributions include: (a) A statistical framework that relates cross-entropy loss of LLMs to competence on the basic skills that underlie language tasks. (b) Mathematical analysis showing that the Scaling Laws imply a strong form of inductive bias that allows the pre-trained model to learn very efficiently. We informally call this {\em slingshot generalization} since naively viewed it appears to give competence levels at skills that violate usual generalization theory. (c) A key example of slingshot generalization, that competence at executing tasks involving $k$-tuples of skills emerges essentially at the same scaling and same rate as competence on the elementary skills themselves.

{{</citation>}}


### (6/32) Dynamic deep-reinforcement-learning algorithm in Partially Observed Markov Decision Processes (Saki Omi et al., 2023)

{{<citation>}}

Saki Omi, Hyo-Sang Shin, Namhoon Cho, Antonios Tsourdos. (2023)  
**Dynamic deep-reinforcement-learning algorithm in Partially Observed Markov Decision Processes**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.15931v1)  

---


**ABSTRACT**  
Reinforcement learning has been greatly improved in recent studies and an increased interest in real-world implementation has emerged in recent years. In many cases, due to the non-static disturbances, it becomes challenging for the agent to keep the performance. The disturbance results in the environment called Partially Observable Markov Decision Process. In common practice, Partially Observable Markov Decision Process is handled by introducing an additional estimator, or Recurrent Neural Network is utilized in the context of reinforcement learning. Both of the cases require to process sequential information on the trajectory. However, there are only a few studies investigating the effect of information to consider and the network structure to handle them. This study shows the benefit of action sequence inclusion in order to solve Partially Observable Markov Decision Process. Several structures and approaches are proposed to extend one of the latest deep reinforcement learning algorithms with LSTM networks. The developed algorithms showed enhanced robustness of controller performance against different types of external disturbances that are added to observation.

{{</citation>}}


### (7/32) Opportunistic Air Quality Monitoring and Forecasting with Expandable Graph Neural Networks (Jingwei Zuo et al., 2023)

{{<citation>}}

Jingwei Zuo, Wenbin Li, Michele Baldo, Hakim Hacid. (2023)  
**Opportunistic Air Quality Monitoring and Forecasting with Expandable Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DB, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.15916v1)  

---


**ABSTRACT**  
Air Quality Monitoring and Forecasting has been a popular research topic in recent years. Recently, data-driven approaches for air quality forecasting have garnered significant attention, owing to the availability of well-established data collection facilities in urban areas. Fixed infrastructures, typically deployed by national institutes or tech giants, often fall short in meeting the requirements of diverse personalized scenarios, e.g., forecasting in areas without any existing infrastructure. Consequently, smaller institutes or companies with limited budgets are compelled to seek tailored solutions by introducing more flexible infrastructures for data collection. In this paper, we propose an expandable graph attention network (EGAT) model, which digests data collected from existing and newly-added infrastructures, with different spatial structures. Additionally, our proposal can be embedded into any air quality forecasting models, to apply to the scenarios with evolving spatial structures. The proposal is validated over real air quality data from PurpleAir.

{{</citation>}}


### (8/32) Efficient Semi-Supervised Federated Learning for Heterogeneous Participants (Zhipeng Sun et al., 2023)

{{<citation>}}

Zhipeng Sun, Yang Xu, Hongli Xu, Zhiyuan Wang. (2023)  
**Efficient Semi-Supervised Federated Learning for Heterogeneous Participants**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.15870v1)  

---


**ABSTRACT**  
Federated Learning (FL) has emerged to allow multiple clients to collaboratively train machine learning models on their private data. However, training and deploying large models for broader applications is challenging in resource-constrained environments. Fortunately, Split Federated Learning (SFL) offers an excellent solution by alleviating the computation and communication burden on the clients SFL often assumes labeled data for local training on clients, however, it is not the case in practice.Prior works have adopted semi-supervised techniques for leveraging unlabeled data in FL, but data non-IIDness poses another challenge to ensure training efficiency. Herein, we propose Pseudo-Clustering Semi-SFL, a novel system for training models in scenarios where labeled data reside on the server. By introducing Clustering Regularization, model performance under data non-IIDness can be improved. Besides, our theoretical and experimental investigations into model convergence reveal that the inconsistent training processes on labeled and unlabeled data impact the effectiveness of clustering regularization. Upon this, we develop a control algorithm for global updating frequency adaptation, which dynamically adjusts the number of supervised training iterations to mitigate the training inconsistency. Extensive experiments on benchmark models and datasets show that our system provides a 3.3x speed-up in training time and reduces the communication cost by about 80.1% while reaching the target accuracy, and achieves up to 6.9% improvement in accuracy under non-IID scenarios compared to the state-of-the-art.

{{</citation>}}


## cs.CL (8)



### (9/32) EnrichEvent: Enriching Social Data with Contextual Information for Emerging Event Extraction (Mohammadali Sefidi Esfahani et al., 2023)

{{<citation>}}

Mohammadali Sefidi Esfahani, Mohammad Akbari. (2023)  
**EnrichEvent: Enriching Social Data with Contextual Information for Emerging Event Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Event Extraction  
[Paper Link](http://arxiv.org/abs/2307.16082v1)  

---


**ABSTRACT**  
Social platforms have emerged as a crucial platform for disseminating and discussing information about real-life events, which offers an excellent opportunity for early detection of newsworthy events. However, most existing approaches for event detection solely exploit keyword burstiness or network structures to detect hot events. Thus, they often fail to identify emerging social events before reaching a trending state regarding the challenging nature of events and social data. Social data, e.g., tweets, is characterized by misspellings, incompleteness, ambiguity, and irregular language, as well as variation in aspects of opinions. Moreover, learning the evolving characteristics of the events utilizing limited contextual knowledge is almost infeasible for machine learning models. To address these problems, in this paper, we propose a framework that exploits the lexical, semantic, and contextual representations of streaming social data. In particular, we leverage contextual knowledge to detect semantically related tweets in their earliest emergence and enhance the quality of produced clusters. We next produce a cluster chains for each event to show the evolving variation of the event through time. We conducted extensive experiments to evaluate our framework, validating the effectiveness of the proposed framework in detecting and distinguishing social events.

{{</citation>}}


### (10/32) Roll Up Your Sleeves: Working with a Collaborative and Engaging Task-Oriented Dialogue System (Lingbo Mo et al., 2023)

{{<citation>}}

Lingbo Mo, Shijie Chen, Ziru Chen, Xiang Deng, Ashley Lewis, Sunit Singh, Samuel Stevens, Chang-You Tai, Zhen Wang, Xiang Yue, Tianshu Zhang, Yu Su, Huan Sun. (2023)  
**Roll Up Your Sleeves: Working with a Collaborative and Engaging Task-Oriented Dialogue System**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.16081v1)  

---


**ABSTRACT**  
We introduce TacoBot, a user-centered task-oriented digital assistant designed to guide users through complex real-world tasks with multiple steps. Covering a wide range of cooking and how-to tasks, we aim to deliver a collaborative and engaging dialogue experience. Equipped with language understanding, dialogue management, and response generation components supported by a robust search engine, TacoBot ensures efficient task assistance. To enhance the dialogue experience, we explore a series of data augmentation strategies using LLMs to train advanced neural models continuously. TacoBot builds upon our successful participation in the inaugural Alexa Prize TaskBot Challenge, where our team secured third place among ten competing teams. We offer TacoBot as an open-source framework that serves as a practical example for deploying task-oriented dialogue systems.

{{</citation>}}


### (11/32) Automatic Extraction of the Romanian Academic Word List: Data and Methods (Ana-Maria Bucur et al., 2023)

{{<citation>}}

Ana-Maria Bucur, Andreea Dincă, Mădălina Chitez, Roxana Rogobete. (2023)  
**Automatic Extraction of the Romanian Academic Word List: Data and Methods**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2307.16045v1)  

---


**ABSTRACT**  
This paper presents the methodology and data used for the automatic extraction of the Romanian Academic Word List (Ro-AWL). Academic Word Lists are useful in both L2 and L1 teaching contexts. For the Romanian language, no such resource exists so far. Ro-AWL has been generated by combining methods from corpus and computational linguistics with L2 academic writing approaches. We use two types of data: (a) existing data, such as the Romanian Frequency List based on the ROMBAC corpus, and (b) self-compiled data, such as the expert academic writing corpus EXPRES. For constructing the academic word list, we follow the methodology for building the Academic Vocabulary List for the English language. The distribution of Ro-AWL features (general distribution, POS distribution) into four disciplinary datasets is in line with previous research. Ro-AWL is freely available and can be used for teaching, research and NLP applications.

{{</citation>}}


### (12/32) Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback (Viet Dac Lai et al., 2023)

{{<citation>}}

Viet Dac Lai, Chien Van Nguyen, Nghia Trung Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan A. Rossi, Thien Huu Nguyen. (2023)  
**Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.16039v2)  

---


**ABSTRACT**  
A key technology for the development of large language models (LLMs) involves instruction tuning that helps align the models' responses with human expectations to realize impressive learning abilities. Two major approaches for instruction tuning characterize supervised fine-tuning (SFT) and reinforcement learning from human feedback (RLHF), which are currently applied to produce the best commercial LLMs (e.g., ChatGPT). To improve the accessibility of LLMs for research and development efforts, various instruction-tuned open-source LLMs have also been introduced recently, e.g., Alpaca, Vicuna, to name a few. However, existing open-source LLMs have only been instruction-tuned for English and a few popular languages, thus hindering their impacts and accessibility to many other languages in the world. Among a few very recent work to explore instruction tuning for LLMs in multiple languages, SFT has been used as the only approach to instruction-tune LLMs for multiple languages. This has left a significant gap for fine-tuned LLMs based on RLHF in diverse languages and raised important questions on how RLHF can boost the performance of multilingual instruction tuning. To overcome this issue, we present Okapi, the first system with instruction-tuned LLMs based on RLHF for multiple languages. Okapi introduces instruction and response-ranked data in 26 diverse languages to facilitate the experiments and development of future multilingual LLM research. We also present benchmark datasets to enable the evaluation of generative LLMs in multiple languages. Our experiments demonstrate the advantages of RLHF for multilingual instruction over SFT for different base models and datasets. Our framework and resources are released at https://github.com/nlp-uoregon/Okapi.

{{</citation>}}


### (13/32) RoCar: A Relationship Network-based Evaluation Method to Large Language Models (Ming Wang et al., 2023)

{{<citation>}}

Ming Wang, Wenfang Wu, Chongyun Gao, Daling Wang, Shi Feng, Yifei Zhang. (2023)  
**RoCar: A Relationship Network-based Evaluation Method to Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.15997v1)  

---


**ABSTRACT**  
Large language models (LLMs) have received increasing attention. However, due to the complexity of its capabilities, how to rationally evaluate the capabilities of LLMs is still a task to be solved. We propose the RoCar method, which utilizes the defined basic schemas to randomly construct a task graph and generates natural language evaluation tasks based on the task graph to evaluate the reasoning and memory abilities of LLMs respectively. Due to the very large randomness of the task construction process, it is possible to ensure that none of the LLMs to be tested has directly learned the evaluation tasks, guaranteeing the fairness of the evaluation method.

{{</citation>}}


### (14/32) Towards Codable Text Watermarking for Large Language Models (Lean Wang et al., 2023)

{{<citation>}}

Lean Wang, Wenkai Yang, Deli Chen, Hao Zhou, Yankai Lin, Fandong Meng, Jie Zhou, Xu Sun. (2023)  
**Towards Codable Text Watermarking for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.15992v1)  

---


**ABSTRACT**  
As large language models (LLMs) generate texts with increasing fluency and realism, there is a growing need to identify the source of texts to prevent the abuse of LLMs. Text watermarking techniques have proven reliable in distinguishing whether a text is generated by LLMs by injecting hidden patterns into the generated texts. However, we argue that existing watermarking methods for LLMs are encoding-inefficient (only contain one bit of information - whether it is generated from an LLM or not) and cannot flexibly meet the diverse information encoding needs (such as encoding model version, generation time, user id, etc.) in different LLMs application scenarios. In this work, we conduct the first systematic study on the topic of Codable Text Watermarking for LLMs (CTWL) that allows text watermarks to carry more customizable information. First of all, we study the taxonomy of LLM watermarking technology and give a mathematical formulation for CTWL. Additionally, we provide a comprehensive evaluation system for CTWL: (1) watermarking success rate, (2) robustness against various corruptions, (3) coding rate of payload information, (4) encoding and decoding efficiency, (5) impacts on the quality of the generated text. To meet the requirements of these non-Pareto-improving metrics, we devise a CTWL method named Balance-Marking, based on the motivation of ensuring that available and unavailable vocabularies for encoding information have approximately equivalent probabilities. Compared to the random vocabulary partitioning extended from the existing work, a probability-balanced vocabulary partition can significantly improve the quality of the generated text. Extensive experimental results have shown that our method outperforms a direct baseline under comprehensive evaluation.

{{</citation>}}


### (15/32) GeneMask: Fast Pretraining of Gene Sequences to Enable Few-Shot Learning (Soumyadeep Roy et al., 2023)

{{<citation>}}

Soumyadeep Roy, Jonas Wallat, Sowmya S Sundaram, Wolfgang Nejdl, Niloy Ganguly. (2023)  
**GeneMask: Fast Pretraining of Gene Sequences to Enable Few-Shot Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2307.15933v1)  

---


**ABSTRACT**  
Large-scale language models such as DNABert and LOGO aim to learn optimal gene representations and are trained on the entire Human Reference Genome. However, standard tokenization schemes involve a simple sliding window of tokens like k-mers that do not leverage any gene-based semantics and thus may lead to (trivial) masking of easily predictable sequences and subsequently inefficient Masked Language Modeling (MLM) training. Therefore, we propose a novel masking algorithm, GeneMask, for MLM training of gene sequences, where we randomly identify positions in a gene sequence as mask centers and locally select the span around the mask center with the highest Normalized Pointwise Mutual Information (NPMI) to mask. We observe that in the absence of human-understandable semantics in the genomics domain (in contrast, semantic units like words and phrases are inherently available in NLP), GeneMask-based models substantially outperform the SOTA models (DNABert and LOGO) over four benchmark gene sequence classification datasets in five few-shot settings (10 to 1000-shot). More significantly, the GeneMask-based DNABert model is trained for less than one-tenth of the number of epochs of the original SOTA model. We also observe a strong correlation between top-ranked PMI tokens and conserved DNA sequence motifs, which may indicate the incorporation of latent genomic information. The codes (including trained models) and datasets are made publicly available at https://github.com/roysoumya/GeneMask.

{{</citation>}}


### (16/32) ATESA-BÆRT: A Heterogeneous Ensemble Learning Model for Aspect-Based Sentiment Analysis (Elena-Simona Apostol et al., 2023)

{{<citation>}}

Elena-Simona Apostol, Alin-Georgian Pisică, Ciprian-Octavian Truică. (2023)  
**ATESA-BÆRT: A Heterogeneous Ensemble Learning Model for Aspect-Based Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2307.15920v1)  

---


**ABSTRACT**  
The increasing volume of online reviews has made possible the development of sentiment analysis models for determining the opinion of customers regarding different products and services. Until now, sentiment analysis has proven to be an effective tool for determining the overall polarity of reviews. To improve the granularity at the aspect level for a better understanding of the service or product, the task of aspect-based sentiment analysis aims to first identify aspects and then determine the user's opinion about them. The complexity of this task lies in the fact that the same review can present multiple aspects, each with its own polarity. Current solutions have poor performance on such data. We address this problem by proposing ATESA-B{\AE}RT, a heterogeneous ensemble learning model for Aspect-Based Sentiment Analysis. Firstly, we divide our problem into two sub-tasks, i.e., Aspect Term Extraction and Aspect Term Sentiment Analysis. Secondly, we use the \textit{argmax} multi-class classification on six transformers-based learners for each sub-task. Initial experiments on two datasets prove that ATESA-B{\AE}RT outperforms current state-of-the-art solutions while solving the many aspects problem.

{{</citation>}}


## cs.CR (3)



### (17/32) Vulnerability Detection Approaches on Application Behaviors in Mobile Environment (Abdellah Ouaguid et al., 2023)

{{<citation>}}

Abdellah Ouaguid, Mohamed Ouzzif, Noreddine Abghour. (2023)  
**Vulnerability Detection Approaches on Application Behaviors in Mobile Environment**  

---
Primary Category: cs.CR  
Categories: A-0, cs-CR, cs-ET, cs.CR  
Keywords: Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2307.16064v1)  

---


**ABSTRACT**  
Several solutions ensuring the dynamic detection of malicious activities on Android ecosystem have been proposed. These are represented by generic rules and models that identify any purported malicious behavior. However, the approaches adopted are far from being effective in detecting malware (listed or not) and whose form and behavior are likely to be different depending on the execution environment or the design of the malware itself (polymorphic for example). An additional difficulty is added when these approaches are unable to capture, analyze, and classify all the execution paths incorporated in the analyzed application earlier. This suggests that the functionality of the analyzed application can constitute a potential risk but never explored or revealed. We have studied some malware detection techniques based on behavioral analysis of applications. The description, characteristics, and results obtained from each technique are presented in this article wherein we have also highlighted some open problems, challenges as well as the different possible future directions of research concerning behavioral analysis of malware.

{{</citation>}}


### (18/32) Analyzing Cryptocurrency trends using Tweet Sentiment Data and User Meta-Data (Samyak Jain et al., 2023)

{{<citation>}}

Samyak Jain, Sarthak Johari, Radhakrishnan Delhibabu. (2023)  
**Analyzing Cryptocurrency trends using Tweet Sentiment Data and User Meta-Data**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: LSTM, Twitter  
[Paper Link](http://arxiv.org/abs/2307.15956v1)  

---


**ABSTRACT**  
Cryptocurrency is a form of digital currency using cryptographic techniques in a decentralized system for secure peer-to-peer transactions. It is gaining much popularity over traditional methods of payments because it facilitates a very fast, easy and secure way of transactions. However, it is very volatile and is influenced by a range of factors, with social media being a major one. Thus, with over four billion active users of social media, we need to understand its influence on the crypto market and how it can lead to fluctuations in the values of these cryptocurrencies. In our work, we analyze the influence of activities on Twitter, in particular the sentiments of the tweets posted regarding cryptocurrencies and how it influences their prices. In addition, we also collect metadata related to tweets and users. We use all these features to also predict the price of cryptocurrency for which we use some regression-based models and an LSTM-based model.

{{</citation>}}


### (19/32) JFinder: A Novel Architecture for Java Vulnerability Identification Based Quad Self-Attention and Pre-training Mechanism (Jin Wang et al., 2023)

{{<citation>}}

Jin Wang, Zishan Huang, Hui Xiao, Yinhao Xiao. (2023)  
**JFinder: A Novel Architecture for Java Vulnerability Identification Based Quad Self-Attention and Pre-training Mechanism**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2307.15915v1)  

---


**ABSTRACT**  
Software vulnerabilities pose significant risks to computer systems, impacting our daily lives, productivity, and even our health. Identifying and addressing security vulnerabilities in a timely manner is crucial to prevent hacking and data breaches. Unfortunately, current vulnerability identification methods, including classical and deep learning-based approaches, exhibit critical drawbacks that prevent them from meeting the demands of the contemporary software industry. To tackle these issues, we present JFinder, a novel architecture for Java vulnerability identification that leverages quad self-attention and pre-training mechanisms to combine structural information and semantic representations. Experimental results demonstrate that JFinder outperforms all baseline methods, achieving an accuracy of 0.97 on the CWE dataset and an F1 score of 0.84 on the PROMISE dataset. Furthermore, a case study reveals that JFinder can accurately identify four cases of vulnerabilities after patching.

{{</citation>}}


## cs.RO (2)



### (20/32) Using Implicit Behavior Cloning and Dynamic Movement Primitive to Facilitate Reinforcement Learning for Robot Motion Planning (Zengjie Zhang et al., 2023)

{{<citation>}}

Zengjie Zhang, Jayden Hong, Amir Soufi Enayati, Homayoun Najjaran. (2023)  
**Using Implicit Behavior Cloning and Dynamic Movement Primitive to Facilitate Reinforcement Learning for Robot Motion Planning**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.16062v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) for motion planning of multi-degree-of-freedom robots still suffers from low efficiency in terms of slow training speed and poor generalizability. In this paper, we propose a novel RL-based robot motion planning framework that uses implicit behavior cloning (IBC) and dynamic movement primitive (DMP) to improve the training speed and generalizability of an off-policy RL agent. IBC utilizes human demonstration data to leverage the training speed of RL, and DMP serves as a heuristic model that transfers motion planning into a simpler planning space. To support this, we also create a human demonstration dataset using a pick-and-place experiment that can be used for similar studies. Comparison studies in simulation reveal the advantage of the proposed method over the conventional RL agents with faster training speed and higher scores. A real-robot experiment indicates the applicability of the proposed method to a simple assembly task. Our work provides a novel perspective on using motion primitives and human demonstration to leverage the performance of RL for robot applications.

{{</citation>}}


### (21/32) PIMbot: Policy and Incentive Manipulation for Multi-Robot Reinforcement Learning in Social Dilemmas (Shahab Nikkhoo et al., 2023)

{{<citation>}}

Shahab Nikkhoo, Zexin Li, Aritra Samanta, Yufei Li, Cong Liu. (2023)  
**PIMbot: Policy and Incentive Manipulation for Multi-Robot Reinforcement Learning in Social Dilemmas**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.15944v1)  

---


**ABSTRACT**  
Recent research has demonstrated the potential of reinforcement learning (RL) in enabling effective multi-robot collaboration, particularly in social dilemmas where robots face a trade-off between self-interests and collective benefits. However, environmental factors such as miscommunication and adversarial robots can impact cooperation, making it crucial to explore how multi-robot communication can be manipulated to achieve different outcomes. This paper presents a novel approach, namely PIMbot, to manipulating the reward function in multi-robot collaboration through two distinct forms of manipulation: policy and incentive manipulation. Our work introduces a new angle for manipulation in recent multi-agent RL social dilemmas that utilize a unique reward function for incentivization. By utilizing our proposed PIMbot mechanisms, a robot is able to manipulate the social dilemma environment effectively. PIMbot has the potential for both positive and negative impacts on the task outcome, where positive impacts lead to faster convergence to the global optimum and maximized rewards for any chosen robot. Conversely, negative impacts can have a detrimental effect on the overall task performance. We present comprehensive experimental results that demonstrate the effectiveness of our proposed methods in the Gazebo-simulated multi-robot environment. Our work provides insights into how inter-robot communication can be manipulated and has implications for various robotic applications. %, including robotics, transportation, and manufacturing.

{{</citation>}}


## cs.CV (4)



### (22/32) HandMIM: Pose-Aware Self-Supervised Learning for 3D Hand Mesh Estimation (Zuyan Liu et al., 2023)

{{<citation>}}

Zuyan Liu, Gaojie Lin, Congyi Wang, Min Zheng, Feida Zhu. (2023)  
**HandMIM: Pose-Aware Self-Supervised Learning for 3D Hand Mesh Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2307.16061v1)  

---


**ABSTRACT**  
With an enormous number of hand images generated over time, unleashing pose knowledge from unlabeled images for supervised hand mesh estimation is an emerging yet challenging topic. To alleviate this issue, semi-supervised and self-supervised approaches have been proposed, but they are limited by the reliance on detection models or conventional ResNet backbones. In this paper, inspired by the rapid progress of Masked Image Modeling (MIM) in visual classification tasks, we propose a novel self-supervised pre-training strategy for regressing 3D hand mesh parameters. Our approach involves a unified and multi-granularity strategy that includes a pseudo keypoint alignment module in the teacher-student framework for learning pose-aware semantic class tokens. For patch tokens with detailed locality, we adopt a self-distillation manner between teacher and student network based on MIM pre-training. To better fit low-level regression tasks, we incorporate pixel reconstruction tasks for multi-level representation learning. Additionally, we design a strong pose estimation baseline using a simple vanilla vision Transformer (ViT) as the backbone and attach a PyMAF head after tokens for regression. Extensive experiments demonstrate that our proposed approach, named HandMIM, achieves strong performance on various hand mesh estimation tasks. Notably, HandMIM outperforms specially optimized architectures, achieving 6.29mm and 8.00mm PAVPE (Vertex-Point-Error) on challenging FreiHAND and HO3Dv2 test sets, respectively, establishing new state-of-the-art records on 3D hand mesh estimation.

{{</citation>}}


### (23/32) Enhancing Object Detection in Ancient Documents with Synthetic Data Generation and Transformer-Based Models (Zahra Ziran et al., 2023)

{{<citation>}}

Zahra Ziran, Francesco Leotta, Massimo Mecella. (2023)  
**Enhancing Object Detection in Ancient Documents with Synthetic Data Generation and Transformer-Based Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2307.16005v1)  

---


**ABSTRACT**  
The study of ancient documents provides a glimpse into our past. However, the low image quality and intricate details commonly found in these documents present significant challenges for accurate object detection. The objective of this research is to enhance object detection in ancient documents by reducing false positives and improving precision. To achieve this, we propose a method that involves the creation of synthetic datasets through computational mediation, along with the integration of visual feature extraction into the object detection process. Our approach includes associating objects with their component parts and introducing a visual feature map to enable the model to discern between different symbols and document elements. Through our experiments, we demonstrate that improved object detection has a profound impact on the field of Paleography, enabling in-depth analysis and fostering a greater understanding of these valuable historical artifacts.

{{</citation>}}


### (24/32) Class-Specific Distribution Alignment for Semi-Supervised Medical Image Classification (Zhongzheng Huang et al., 2023)

{{<citation>}}

Zhongzheng Huang, Jiawei Wu, Tao Wang, Zuoyong Li, Anastasia Ioannou. (2023)  
**Class-Specific Distribution Alignment for Semi-Supervised Medical Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.15987v1)  

---


**ABSTRACT**  
Despite the success of deep neural networks in medical image classification, the problem remains challenging as data annotation is time-consuming, and the class distribution is imbalanced due to the relative scarcity of diseases. To address this problem, we propose Class-Specific Distribution Alignment (CSDA), a semi-supervised learning framework based on self-training that is suitable to learn from highly imbalanced datasets. Specifically, we first provide a new perspective to distribution alignment by considering the process as a change of basis in the vector space spanned by marginal predictions, and then derive CSDA to capture class-dependent marginal predictions on both labeled and unlabeled data, in order to avoid the bias towards majority classes. Furthermore, we propose a Variable Condition Queue (VCQ) module to maintain a proportionately balanced number of unlabeled samples for each class. Experiments on three public datasets HAM10000, CheXpert and Kvasir show that our method provides competitive performance on semi-supervised skin disease, thoracic disease, and endoscopic image classification tasks.

{{</citation>}}


### (25/32) CMDA: Cross-Modality Domain Adaptation for Nighttime Semantic Segmentation (Ruihao Xia et al., 2023)

{{<citation>}}

Ruihao Xia, Chaoqiang Zhao, Meng Zheng, Ziyan Wu, Qiyu Sun, Yang Tang. (2023)  
**CMDA: Cross-Modality Domain Adaptation for Nighttime Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.15942v1)  

---


**ABSTRACT**  
Most nighttime semantic segmentation studies are based on domain adaptation approaches and image input. However, limited by the low dynamic range of conventional cameras, images fail to capture structural details and boundary information in low-light conditions. Event cameras, as a new form of vision sensors, are complementary to conventional cameras with their high dynamic range. To this end, we propose a novel unsupervised Cross-Modality Domain Adaptation (CMDA) framework to leverage multi-modality (Images and Events) information for nighttime semantic segmentation, with only labels on daytime images. In CMDA, we design the Image Motion-Extractor to extract motion information and the Image Content-Extractor to extract content information from images, in order to bridge the gap between different modalities (Images to Events) and domains (Day to Night). Besides, we introduce the first image-event nighttime semantic segmentation dataset. Extensive experiments on both the public image dataset and the proposed image-event dataset demonstrate the effectiveness of our proposed approach. We open-source our code, models, and dataset at https://github.com/XiaRho/CMDA.

{{</citation>}}


## cs.IR (1)



### (26/32) Click-Conversion Multi-Task Model with Position Bias Mitigation for Sponsored Search in eCommerce (Yibo Wang et al., 2023)

{{<citation>}}

Yibo Wang, Yanbing Xue, Bo Liu, Musen Wen, Wenting Zhao, Stephen Guo, Philip S. Yu. (2023)  
**Click-Conversion Multi-Task Model with Position Bias Mitigation for Sponsored Search in eCommerce**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Bias, Embedding, Position Embedding  
[Paper Link](http://arxiv.org/abs/2307.16060v1)  

---


**ABSTRACT**  
Position bias, the phenomenon whereby users tend to focus on higher-ranked items of the search result list regardless of the actual relevance to queries, is prevailing in many ranking systems. Position bias in training data biases the ranking model, leading to increasingly unfair item rankings, click-through-rate (CTR), and conversion rate (CVR) predictions. To jointly mitigate position bias in both item CTR and CVR prediction, we propose two position-bias-free CTR and CVR prediction models: Position-Aware Click-Conversion (PACC) and PACC via Position Embedding (PACC-PE). PACC is built upon probability decomposition and models position information as a probability. PACC-PE utilizes neural networks to model product-specific position information as embedding. Experiments on the E-commerce sponsored product search dataset show that our proposed models have better ranking effectiveness and can greatly alleviate position bias in both CTR and CVR prediction.

{{</citation>}}


## eess.IV (1)



### (27/32) CoVid-19 Detection leveraging Vision Transformers and Explainable AI (Pangoth Santhosh Kumar et al., 2023)

{{<citation>}}

Pangoth Santhosh Kumar, Kundrapu Supriya, Mallikharjuna Rao K. (2023)  
**CoVid-19 Detection leveraging Vision Transformers and Explainable AI**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.16033v1)  

---


**ABSTRACT**  
Lung disease is a common health problem in many parts of the world. It is a significant risk to people health and quality of life all across the globe since it is responsible for five of the top thirty leading causes of death. Among them are COVID 19, pneumonia, and tuberculosis, to name just a few. It is critical to diagnose lung diseases in their early stages. Several different models including machine learning and image processing have been developed for this purpose. The earlier a condition is diagnosed, the better the patient chances of making a full recovery and surviving into the long term. Thanks to deep learning algorithms, there is significant promise for the autonomous, rapid, and accurate identification of lung diseases based on medical imaging. Several different deep learning strategies, including convolutional neural networks (CNN), vanilla neural networks, visual geometry group based networks (VGG), and capsule networks , are used for the goal of making lung disease forecasts. The standard CNN has a poor performance when dealing with rotated, tilted, or other aberrant picture orientations. As a result of this, within the scope of this study, we have suggested a vision transformer based approach end to end framework for the diagnosis of lung disorders. In the architecture, data augmentation, training of the suggested models, and evaluation of the models are all included. For the purpose of detecting lung diseases such as pneumonia, Covid 19, lung opacity, and others, a specialised Compact Convolution Transformers (CCT) model have been tested and evaluated on datasets such as the Covid 19 Radiography Database. The model has achieved a better accuracy for both its training and validation purposes on the Covid 19 Radiography Database.

{{</citation>}}


## cs.AI (2)



### (28/32) Marrying Dialogue Systems with Data Visualization: Interactive Data Visualization Generation from Natural Language Conversations (Yuanfeng Song et al., 2023)

{{<citation>}}

Yuanfeng Song, Xuefang Zhao, Raymond Chi-Wing Wong. (2023)  
**Marrying Dialogue Systems with Data Visualization: Interactive Data Visualization Generation from Natural Language Conversations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-DB, cs.AI  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.16013v1)  

---


**ABSTRACT**  
Data visualization (DV) has become the prevailing tool in the market due to its effectiveness into illustrating insights in vast amounts of data. To lower the barrier of using DVs, automatic DV tasks, such as natural language question (NLQ) to visualization translation (formally called text-to-vis), have been investigated in the research community. However, text-to-vis assumes the NLQ to be well-organized and expressed in a single sentence. However, in real-world settings, complex DV is needed through consecutive exchanges between the DV system and the users. In this paper, we propose a new task named CoVis, short for Conversational text-to-Visualization, aiming at constructing DVs through a series of interactions between users and the system. Since it is the task which has not been studied in the literature, we first build a benchmark dataset named Dial-NVBench, including dialogue sessions with a sequence of queries from a user and responses from the system. Then, we propose a multi-modal neural network named MMCoVisNet to answer these DV-related queries. In particular, MMCoVisNet first fully understands the dialogue context and determines the corresponding responses. Then, it uses adaptive decoders to provide the appropriate replies: (i) a straightforward text decoder is used to produce general responses, (ii) an SQL-form decoder is applied to synthesize data querying responses, and (iii) a DV-form decoder tries to construct the appropriate DVs. We comparatively evaluate MMCoVisNet with other baselines over our proposed benchmark dataset. Experimental results validate that MMCoVisNet performs better than existing baselines and achieves a state-of-the-art performance.

{{</citation>}}


### (29/32) Reinforcement Learning Under Probabilistic Spatio-Temporal Constraints with Time Windows (Xiaoshan Lin et al., 2023)

{{<citation>}}

Xiaoshan Lin, Abbasali Koochakzadeh, Yasin Yazicioglu, Derya Aksaray. (2023)  
**Reinforcement Learning Under Probabilistic Spatio-Temporal Constraints with Time Windows**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-FL, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.15910v1)  

---


**ABSTRACT**  
We propose an automata-theoretic approach for reinforcement learning (RL) under complex spatio-temporal constraints with time windows. The problem is formulated using a Markov decision process under a bounded temporal logic constraint. Different from existing RL methods that can eventually learn optimal policies satisfying such constraints, our proposed approach enforces a desired probability of constraint satisfaction throughout learning. This is achieved by translating the bounded temporal logic constraint into a total automaton and avoiding "unsafe" actions based on the available prior information regarding the transition probabilities, i.e., a pair of upper and lower bounds for each transition probability. We provide theoretical guarantees on the resulting probability of constraint satisfaction. We also provide numerical results in a scenario where a robot explores the environment to discover high-reward regions while fulfilling some periodic pick-up and delivery tasks that are encoded as temporal logic constraints.

{{</citation>}}


## cs.SD (1)



### (30/32) Monaural Multi-Speaker Speech Separation Using Efficient Transformer Model (S. Rijal et al., 2023)

{{<citation>}}

S. Rijal, R. Neupane, S. P. Mainali, S. K. Regmi, S. Maharjan. (2023)  
**Monaural Multi-Speaker Speech Separation Using Efficient Transformer Model**  

---
Primary Category: cs.SD  
Categories: 68T10, I-2-m, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.00010v1)  

---


**ABSTRACT**  
Cocktail party problem is the scenario where it is difficult to separate or distinguish individual speaker from a mixed speech from several speakers. There have been several researches going on in this field but the size and complexity of the model is being traded off with the accuracy and robustness of speech separation. "Monaural multi-speaker speech separation" presents a speech-separation model based on the Transformer architecture and its efficient forms. The model has been trained with the LibriMix dataset containing diverse speakers' utterances. The model separates 2 distinct speaker sources from a mixed audio input. The developed model approaches the reduction in computational complexity of the speech separation model, with minimum tradeoff with the performance of prevalent speech separation model and it has shown significant movement towards that goal. This project foresees, a rise in contribution towards the ongoing research in the field of speech separation with computational efficiency at its core.

{{</citation>}}


## cs.MM (1)



### (31/32) Instance-Wise Adaptive Tuning and Caching for Vision-Language Models (Chunjin Yang et al., 2023)

{{<citation>}}

Chunjin Yang, Fanman Meng, Shuai Chen, Mingyu Liu, Runtong Zhang. (2023)  
**Instance-Wise Adaptive Tuning and Caching for Vision-Language Models**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.15983v1)  

---


**ABSTRACT**  
Large-scale vision-language models (LVLMs) pretrained on massive image-text pairs have achieved remarkable success in visual representations. However, existing paradigms to transfer LVLMs to downstream tasks encounter two primary challenges. Firstly, the text features remain fixed after being calculated and cannot be adjusted according to image features, which decreases the model's adaptability. Secondly, the model's output solely depends on the similarity between the text and image features, leading to excessive reliance on LVLMs. To address these two challenges, we introduce a novel two-branch model named the Instance-Wise Adaptive Tuning and Caching (ATC). Specifically, one branch implements our proposed ConditionNet, which guides image features to form an adaptive textual cache that adjusts based on image features, achieving instance-wise inference and improving the model's adaptability. The other branch introduces the similarities between images and incorporates a learnable visual cache, designed to decouple new and previous knowledge, allowing the model to acquire new knowledge while preserving prior knowledge. The model's output is jointly determined by the two branches, thus overcoming the limitations of existing methods that rely solely on LVLMs. Additionally, our method requires limited computing resources to tune parameters, yet outperforms existing methods on 11 benchmark datasets.

{{</citation>}}


## cs.NI (1)



### (32/32) Distributed Traffic Engineering in Hybrid Software Defined Networks: A Multi-agent Reinforcement Learning Framework (Yingya Guo et al., 2023)

{{<citation>}}

Yingya Guo, Qi Tang, Yulong Ma, Han Tian, Kai Chen. (2023)  
**Distributed Traffic Engineering in Hybrid Software Defined Networks: A Multi-agent Reinforcement Learning Framework**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.15922v1)  

---


**ABSTRACT**  
Traffic Engineering (TE) is an efficient technique to balance network flows and thus improves the performance of a hybrid Software Defined Network (SDN). Previous TE solutions mainly leverage heuristic algorithms to centrally optimize link weight setting or traffic splitting ratios under the static traffic demand. Note that as the network scale becomes larger and network management gains more complexity, it is notably that the centralized TE methods suffer from a high computation overhead and a long reaction time to optimize routing of flows when the network traffic demand dynamically fluctuates or network failures happen. To enable adaptive and efficient routing in TE, we propose a Multi-agent Reinforcement Learning method CMRL that divides the routing optimization of a large network into multiple small-scale routing decisionmaking problems. To coordinate the multiple agents for achieving a global optimization goal, we construct an interactive environment for training the routing agents that own partial link utilization observations. To optimize credit assignment of multi-agent, we introduce the difference reward assignment mechanism for encouraging agents to take better action. Extensive simulations conducted on the real traffic traces demonstrate the superiority of CMRL in improving TE performance, especially when traffic demands change or network failures happen.

{{</citation>}}
