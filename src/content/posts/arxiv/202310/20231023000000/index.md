---
draft: false
title: "arXiv @ 2023.10.23"
date: 2023-10-23
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.23"
    identifier: arxiv_20231023
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (8)](#cslg-8)
- [cs.CL (21)](#cscl-21)
- [cs.IR (4)](#csir-4)
- [cs.CR (1)](#cscr-1)
- [cs.RO (2)](#csro-2)
- [math.NA (1)](#mathna-1)
- [eess.IV (2)](#eessiv-2)
- [cs.DS (2)](#csds-2)
- [cs.CY (1)](#cscy-1)
- [cs.CV (5)](#cscv-5)
- [cs.SE (1)](#csse-1)
- [cs.FL (1)](#csfl-1)
- [cs.HC (1)](#cshc-1)
- [cs.SI (1)](#cssi-1)
- [stat.ML (1)](#statml-1)

## cs.LG (8)



### (1/52) Optimal Batched Best Arm Identification (Tianyuan Jin et al., 2023)

{{<citation>}}

Tianyuan Jin, Yu Yang, Jing Tang, Xiaokui Xiao, Pan Xu. (2023)  
**Optimal Batched Best Arm Identification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.14129v1)  

---


**ABSTRACT**  
We study the batched best arm identification (BBAI) problem, where the learner's goal is to identify the best arm while switching the policy as less as possible. In particular, we aim to find the best arm with probability $1-\delta$ for some small constant $\delta>0$ while minimizing both the sample complexity (total number of arm pulls) and the batch complexity (total number of batches). We propose the three-batch best arm identification (Tri-BBAI) algorithm, which is the first batched algorithm that achieves the optimal sample complexity in the asymptotic setting (i.e., $\delta\rightarrow 0$) and runs only in at most $3$ batches. Based on Tri-BBAI, we further propose the almost optimal batched best arm identification (Opt-BBAI) algorithm, which is the first algorithm that achieves the near-optimal sample and batch complexity in the non-asymptotic setting (i.e., $\delta>0$ is arbitrarily fixed), while enjoying the same batch and sample complexity as Tri-BBAI when $\delta$ tends to zero. Moreover, in the non-asymptotic setting, the complexity of previous batch algorithms is usually conditioned on the event that the best arm is returned (with a probability of at least $1-\delta$), which is potentially unbounded in cases where a sub-optimal arm is returned. In contrast, the complexity of Opt-BBAI does not rely on such an event. This is achieved through a novel procedure that we design for checking whether the best arm is eliminated, which is of independent interest.

{{</citation>}}


### (2/52) Revisiting Instruction Fine-tuned Model Evaluation to Guide Industrial Applications (Manuel Faysse et al., 2023)

{{<citation>}}

Manuel Faysse, Gautier Viaud, Céline Hudelot, Pierre Colombo. (2023)  
**Revisiting Instruction Fine-tuned Model Evaluation to Guide Industrial Applications**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.14103v1)  

---


**ABSTRACT**  
Instruction Fine-Tuning (IFT) is a powerful paradigm that strengthens the zero-shot capabilities of Large Language Models (LLMs), but in doing so induces new evaluation metric requirements. We show LLM-based metrics to be well adapted to these requirements, and leverage them to conduct an investigation of task-specialization strategies, quantifying the trade-offs that emerge in practical industrial settings. Our findings offer practitioners actionable insights for real-world IFT model deployment.

{{</citation>}}


### (3/52) Beyond Accuracy: Evaluating Self-Consistency of Code Large Language Models with IdentityChain (Marcus J. Min et al., 2023)

{{<citation>}}

Marcus J. Min, Yangruibo Ding, Luca Buratti, Saurabh Pujar, Gail Kaiser, Suman Jana, Baishakhi Ray. (2023)  
**Beyond Accuracy: Evaluating Self-Consistency of Code Large Language Models with IdentityChain**  

---
Primary Category: cs.LG  
Categories: 68, I-2; D-2, cs-CL, cs-LG, cs-SE, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.14053v1)  

---


**ABSTRACT**  
Code Large Language Models (Code LLMs) are being increasingly employed in real-life applications, so evaluating them is critical. While the general accuracy of Code LLMs on individual tasks has been extensively evaluated, their self-consistency across different tasks is overlooked. Intuitively, a trustworthy model should be self-consistent when generating natural language specifications for its own code and generating code for its own specifications. Failure to preserve self-consistency reveals a lack of understanding of the shared semantics underlying natural language and programming language, and therefore undermines the trustworthiness of a model. In this paper, we first formally define the self-consistency of Code LLMs and then design a framework, IdentityChain, which effectively and efficiently evaluates the self-consistency and general accuracy of a model at the same time. We study eleven Code LLMs and show that they fail to preserve self-consistency, which is indeed a distinct aspect from general accuracy. Furthermore, we show that IdentityChain can be used as a model debugging tool to expose weaknesses of Code LLMs by demonstrating three major weaknesses that we identify in current models using IdentityChain. Our code is available at https://github.com/marcusm117/IdentityChain.

{{</citation>}}


### (4/52) Filling the Missing: Exploring Generative AI for Enhanced Federated Learning over Heterogeneous Mobile Edge Devices (Peichun Li et al., 2023)

{{<citation>}}

Peichun Li, Hanwen Zhang, Yuan Wu, Liping Qian, Rong Yu, Dusit Niyato, Xuemin, Shen. (2023)  
**Filling the Missing: Exploring Generative AI for Enhanced Federated Learning over Heterogeneous Mobile Edge Devices**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.13981v1)  

---


**ABSTRACT**  
Distributed Artificial Intelligence (AI) model training over mobile edge networks encounters significant challenges due to the data and resource heterogeneity of edge devices. The former hampers the convergence rate of the global model, while the latter diminishes the devices' resource utilization efficiency. In this paper, we propose a generative AI-empowered federated learning to address these challenges by leveraging the idea of FIlling the MIssing (FIMI) portion of local data. Specifically, FIMI can be considered as a resource-aware data augmentation method that effectively mitigates the data heterogeneity while ensuring efficient FL training. We first quantify the relationship between the training data amount and the learning performance. We then study the FIMI optimization problem with the objective of minimizing the device-side overall energy consumption subject to required learning performance constraints. The decomposition-based analysis and the cross-entropy searching method are leveraged to derive the solution, where each device is assigned suitable AI-synthesized data and resource utilization policy. Experiment results demonstrate that FIMI can save up to 50% of the device-side energy to achieve the target global test accuracy in comparison with the existing methods. Meanwhile, FIMI can significantly enhance the converged global accuracy under the non-independently-and-identically distribution (non-IID) data.

{{</citation>}}


### (5/52) Toward Generative Data Augmentation for Traffic Classification (Chao Wang et al., 2023)

{{<citation>}}

Chao Wang, Alessandro Finamore, Pietro Michiardi, Massimo Gallo, Dario Rossi. (2023)  
**Toward Generative Data Augmentation for Traffic Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation, Computer Vision  
[Paper Link](http://arxiv.org/abs/2310.13935v1)  

---


**ABSTRACT**  
Data Augmentation (DA)-augmenting training data with synthetic samples-is wildly adopted in Computer Vision (CV) to improve models performance. Conversely, DA has not been yet popularized in networking use cases, including Traffic Classification (TC). In this work, we present a preliminary study of 14 hand-crafted DAs applied on the MIRAGE19 dataset. Our results (i) show that DA can reap benefits previously unexplored in TC and (ii) foster a research agenda on the use of generative models to automate DA design.

{{</citation>}}


### (6/52) Pre-Training on Large-Scale Generated Docking Conformations with HelixDock to Unlock the Potential of Protein-ligand Structure Prediction Models (Lihang Liu et al., 2023)

{{<citation>}}

Lihang Liu, Donglong He, Xianbin Ye, Shanzhuo Zhang, Xiaonan Zhang, Jingbo Zhou, Jun Li, Hua Chai, Fan Wang, Jingzhou He, Liang Zheng, Yonghui Li, Xiaomin Fang. (2023)  
**Pre-Training on Large-Scale Generated Docking Conformations with HelixDock to Unlock the Potential of Protein-ligand Structure Prediction Models**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG, q-bio-BM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13913v1)  

---


**ABSTRACT**  
Molecular docking, a pivotal computational tool for drug discovery, predicts the binding interactions between small molecules (ligands) and target proteins (receptors). Conventional physics-based docking tools, though widely used, face limitations in precision due to restricted conformational sampling and imprecise scoring functions. Recent endeavors have employed deep learning techniques to enhance docking accuracy, but their generalization remains a concern due to limited training data. Leveraging the success of extensive and diverse data in other domains, we introduce HelixDock, a novel approach for site-specific molecular docking. Hundreds of millions of binding poses are generated by traditional docking tools, encompassing diverse protein targets and small molecules. Our deep learning-based docking model, a SE(3)-equivariant network, is pre-trained with this large-scale dataset and then fine-tuned with a small number of precise receptor-ligand complex structures. Comparative analyses against physics-based and deep learning-based baseline methods highlight HelixDock's superiority, especially on challenging test sets. Our study elucidates the scaling laws of the pre-trained molecular docking models, showcasing consistent improvements with increased model parameters and pre-train data quantities. Harnessing the power of extensive and diverse generated data holds promise for advancing AI-driven drug discovery.

{{</citation>}}


### (7/52) Towards Hyperparameter-Agnostic DNN Training via Dynamical System Insights (Carmel Fiscko et al., 2023)

{{<citation>}}

Carmel Fiscko, Aayushya Agarwal, Yihan Ruan, Soummya Kar, Larry Pileggi, Bruno Sinopoli. (2023)  
**Towards Hyperparameter-Agnostic DNN Training via Dynamical System Insights**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.13901v1)  

---


**ABSTRACT**  
We present a stochastic first-order optimization method specialized for deep neural networks (DNNs), ECCO-DNN. This method models the optimization variable trajectory as a dynamical system and develops a discretization algorithm that adaptively selects step sizes based on the trajectory's shape. This provides two key insights: designing the dynamical system for fast continuous-time convergence and developing a time-stepping algorithm to adaptively select step sizes based on principles of numerical integration and neural network structure. The result is an optimizer with performance that is insensitive to hyperparameter variations and that achieves comparable performance to state-of-the-art optimizers including ADAM, SGD, RMSProp, and AdaGrad. We demonstrate this in training DNN models and datasets, including CIFAR-10 and CIFAR-100 using ECCO-DNN and find that ECCO-DNN's single hyperparameter can be changed by three orders of magnitude without affecting the trained models' accuracies. ECCO-DNN's insensitivity reduces the data and computation needed for hyperparameter tuning, making it advantageous for rapid prototyping and for applications with new datasets. To validate the efficacy of our proposed optimizer, we train an LSTM architecture on a household power consumption dataset with ECCO-DNN and achieve an optimal mean-square-error without tuning hyperparameters.

{{</citation>}}


### (8/52) The Hidden Adversarial Vulnerabilities of Medical Federated Learning (Erfan Darzi et al., 2023)

{{<citation>}}

Erfan Darzi, Florian Dubost, Nanna. M. Sijtsema, P. M. A van Ooijen. (2023)  
**The Hidden Adversarial Vulnerabilities of Medical Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13893v1)  

---


**ABSTRACT**  
In this paper, we delve into the susceptibility of federated medical image analysis systems to adversarial attacks. Our analysis uncovers a novel exploitation avenue: using gradient information from prior global model updates, adversaries can enhance the efficiency and transferability of their attacks. Specifically, we demonstrate that single-step attacks (e.g. FGSM), when aptly initialized, can outperform the efficiency of their iterative counterparts but with reduced computational demand. Our findings underscore the need to revisit our understanding of AI security in federated healthcare settings.

{{</citation>}}


## cs.CL (21)



### (9/52) Ask To The Point: Open-Domain Entity-Centric Question Generation (Yuxiang Liu et al., 2023)

{{<citation>}}

Yuxiang Liu, Jie Huang, Kevin Chen-Chuan Chang. (2023)  
**Ask To The Point: Open-Domain Entity-Centric Question Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Question Generation  
[Paper Link](http://arxiv.org/abs/2310.14126v1)  

---


**ABSTRACT**  
We introduce a new task called *entity-centric question generation* (ECQG), motivated by real-world applications such as topic-specific learning, assisted reading, and fact-checking. The task aims to generate questions from an entity perspective. To solve ECQG, we propose a coherent PLM-based framework GenCONE with two novel modules: content focusing and question verification. The content focusing module first identifies a focus as "what to ask" to form draft questions, and the question verification module refines the questions afterwards by verifying the answerability. We also construct a large-scale open-domain dataset from SQuAD to support this task. Our extensive experiments demonstrate that GenCONE significantly and consistently outperforms various baselines, and two modules are effective and complementary in generating high-quality questions.

{{</citation>}}


### (10/52) Structural generalization in COGS: Supertagging is (almost) all you need (Alban Petit et al., 2023)

{{<citation>}}

Alban Petit, Caio Corro, François Yvon. (2023)  
**Structural generalization in COGS: Supertagging is (almost) all you need**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.14124v1)  

---


**ABSTRACT**  
In many Natural Language Processing applications, neural networks have been found to fail to generalize on out-of-distribution examples. In particular, several recent semantic parsing datasets have put forward important limitations of neural networks in cases where compositional generalization is required. In this work, we extend a neural graph-based semantic parsing framework in several ways to alleviate this issue. Notably, we propose: (1) the introduction of a supertagging step with valency constraints, expressed as an integer linear program; (2) a reduction of the graph prediction problem to the maximum matching problem; (3) the design of an incremental early-stopping training strategy to prevent overfitting. Experimentally, our approach significantly improves results on examples that require structural generalization in the COGS dataset, a known challenging benchmark for compositional generalization. Overall, our results confirm that structural constraints are important for generalization in semantic parsing.

{{</citation>}}


### (11/52) Sentiment Analysis Across Multiple African Languages: A Current Benchmark (Saurav K. Aryal et al., 2023)

{{<citation>}}

Saurav K. Aryal, Howard Prioleau, Surakshya Aryal. (2023)  
**Sentiment Analysis Across Multiple African Languages: A Current Benchmark**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2310.14120v1)  

---


**ABSTRACT**  
Sentiment analysis is a fundamental and valuable task in NLP. However, due to limitations in data and technological availability, research into sentiment analysis of African languages has been fragmented and lacking. With the recent release of the AfriSenti-SemEval Shared Task 12, hosted as a part of The 17th International Workshop on Semantic Evaluation, an annotated sentiment analysis of 14 African languages was made available. We benchmarked and compared current state-of-art transformer models across 12 languages and compared the performance of training one-model-per-language versus single-model-all-languages. We also evaluated the performance of standard multilingual models and their ability to learn and transfer cross-lingual representation from non-African to African languages. Our results show that despite work in low resource modeling, more data still produces better models on a per-language basis. Models explicitly developed for African languages outperform other models on all tasks. Additionally, no one-model-fits-all solution exists for a per-language evaluation of the models evaluated. Moreover, for some languages with a smaller sample size, a larger multilingual model may perform better than a dedicated per-language model for sentiment classification.

{{</citation>}}


### (12/52) Finite-context Indexing of Restricted Output Space for NLP Models Facing Noisy Input (Minh Nguyen et al., 2023)

{{<citation>}}

Minh Nguyen, Nancy F. Chen. (2023)  
**Finite-context Indexing of Restricted Output Space for NLP Models Facing Noisy Input**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.14110v1)  

---


**ABSTRACT**  
NLP models excel on tasks with clean inputs, but are less accurate with noisy inputs. In particular, character-level noise such as human-written typos and adversarially-engineered realistic-looking misspellings often appears in text and can easily trip up NLP models. Prior solutions to address character-level noise often alter the content of the inputs (low fidelity), thus inadvertently lowering model accuracy on clean inputs. We proposed FiRo, an approach to boost NLP model performance on noisy inputs without sacrificing performance on clean inputs. FiRo sanitizes the input text while preserving its fidelity by inferring the noise-free form for each token in the input. FiRo uses finite-context aggregation to obtain contextual embeddings which is then used to find the noise-free form within a restricted output space. The output space is restricted to a small cluster of probable candidates in order to predict the noise-free tokens more accurately. Although the clusters are small, FiRo's effective vocabulary (union of all clusters) can be scaled up to better preserve the input content. Experimental results show NLP models that use FiRo outperforming baselines on six classification tasks and one sequence labeling task at various degrees of noise.

{{</citation>}}


### (13/52) Leveraging Knowledge Graphs for Orphan Entity Allocation in Resume Processing (Aagam Bakliwal et al., 2023)

{{<citation>}}

Aagam Bakliwal, Shubham Manish Gandhi, Yashodhara Haribhakta. (2023)  
**Leveraging Knowledge Graphs for Orphan Entity Allocation in Resume Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.14093v1)  

---


**ABSTRACT**  
Significant challenges are posed in talent acquisition and recruitment by processing and analyzing unstructured data, particularly resumes. This research presents a novel approach for orphan entity allocation in resume processing using knowledge graphs. Techniques of association mining, concept extraction, external knowledge linking, named entity recognition, and knowledge graph construction are integrated into our pipeline. By leveraging these techniques, the aim is to automate and enhance the efficiency of the job screening process by successfully bucketing orphan entities within resumes. This allows for more effective matching between candidates and job positions, streamlining the resume screening process, and enhancing the accuracy of candidate-job matching. The approach's exceptional effectiveness and resilience are highlighted through extensive experimentation and evaluation, ensuring that alternative measures can be relied upon for seamless processing and orphan entity allocation in case of any component failure. The capabilities of knowledge graphs in generating valuable insights through intelligent information extraction and representation, specifically in the domain of categorizing orphan entities, are highlighted by the results of our research.

{{</citation>}}


### (14/52) MedEval: A Multi-Level, Multi-Task, and Multi-Domain Medical Benchmark for Language Model Evaluation (Zexue He et al., 2023)

{{<citation>}}

Zexue He, Yu Wang, An Yan, Yao Liu, Eric Y. Chang, Amilcare Gentili, Julian McAuley, Chun-Nan Hsu. (2023)  
**MedEval: A Multi-Level, Multi-Task, and Multi-Domain Medical Benchmark for Language Model Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.14088v1)  

---


**ABSTRACT**  
Curated datasets for healthcare are often limited due to the need of human annotations from experts. In this paper, we present MedEval, a multi-level, multi-task, and multi-domain medical benchmark to facilitate the development of language models for healthcare. MedEval is comprehensive and consists of data from several healthcare systems and spans 35 human body regions from 8 examination modalities. With 22,779 collected sentences and 21,228 reports, we provide expert annotations at multiple levels, offering a granular potential usage of the data and supporting a wide range of tasks. Moreover, we systematically evaluated 10 generic and domain-specific language models under zero-shot and finetuning settings, from domain-adapted baselines in healthcare to general-purposed state-of-the-art large language models (e.g., ChatGPT). Our evaluations reveal varying effectiveness of the two categories of language models across different tasks, from which we notice the importance of instruction tuning for few-shot usage of large language models. Our investigation paves the way toward benchmarking language models for healthcare and provides valuable insights into the strengths and limitations of adopting large language models in medical domains, informing their practical applications and future advancements.

{{</citation>}}


### (15/52) Code-Switching with Word Senses for Pretraining in Neural Machine Translation (Vivek Iyer et al., 2023)

{{<citation>}}

Vivek Iyer, Edoardo Barba, Alexandra Birch, Jeff Z. Pan, Roberto Navigli. (2023)  
**Code-Switching with Word Senses for Pretraining in Neural Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2310.14050v1)  

---


**ABSTRACT**  
Lexical ambiguity is a significant and pervasive challenge in Neural Machine Translation (NMT), with many state-of-the-art (SOTA) NMT systems struggling to handle polysemous words (Campolungo et al., 2022). The same holds for the NMT pretraining paradigm of denoising synthetic "code-switched" text (Pan et al., 2021; Iyer et al., 2023), where word senses are ignored in the noising stage -- leading to harmful sense biases in the pretraining data that are subsequently inherited by the resulting models. In this work, we introduce Word Sense Pretraining for Neural Machine Translation (WSP-NMT) - an end-to-end approach for pretraining multilingual NMT models leveraging word sense-specific information from Knowledge Bases. Our experiments show significant improvements in overall translation quality. Then, we show the robustness of our approach to scale to various challenging data and resource-scarce scenarios and, finally, report fine-grained accuracy improvements on the DiBiMT disambiguation benchmark. Our studies yield interesting and novel insights into the merits and challenges of integrating word sense information and structured knowledge in multilingual pretraining for NMT.

{{</citation>}}


### (16/52) MeaeQ: Mount Model Extraction Attacks with Efficient Queries (Chengwei Dai et al., 2023)

{{<citation>}}

Chengwei Dai, Minxuan Lv, Kun Li, Wei Zhou. (2023)  
**MeaeQ: Mount Model Extraction Attacks with Efficient Queries**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.14047v1)  

---


**ABSTRACT**  
We study model extraction attacks in natural language processing (NLP) where attackers aim to steal victim models by repeatedly querying the open Application Programming Interfaces (APIs). Recent works focus on limited-query budget settings and adopt random sampling or active learning-based sampling strategies on publicly available, unannotated data sources. However, these methods often result in selected queries that lack task relevance and data diversity, leading to limited success in achieving satisfactory results with low query costs. In this paper, we propose MeaeQ (Model extraction attack with efficient Queries), a straightforward yet effective method to address these issues. Specifically, we initially utilize a zero-shot sequence inference classifier, combined with API service information, to filter task-relevant data from a public text corpus instead of a problem domain-specific dataset. Furthermore, we employ a clustering-based data reduction technique to obtain representative data as queries for the attack. Extensive experiments conducted on four benchmark datasets demonstrate that MeaeQ achieves higher functional similarity to the victim model than baselines while requiring fewer queries. Our code is available at https://github.com/C-W-D/MeaeQ.

{{</citation>}}


### (17/52) Analysing State-Backed Propaganda Websites: a New Dataset and Linguistic Study (Freddy Heppell et al., 2023)

{{<citation>}}

Freddy Heppell, Kalina Bontcheva, Carolina Scarton. (2023)  
**Analysing State-Backed Propaganda Websites: a New Dataset and Linguistic Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.14032v1)  

---


**ABSTRACT**  
This paper analyses two hitherto unstudied sites sharing state-backed disinformation, Reliable Recent News (rrn.world) and WarOnFakes (waronfakes.com), which publish content in Arabic, Chinese, English, French, German, and Spanish. We describe our content acquisition methodology and perform cross-site unsupervised topic clustering on the resulting multilingual dataset. We also perform linguistic and temporal analysis of the web page translations and topics over time, and investigate articles with false publication dates. We make publicly available this new dataset of 14,053 articles, annotated with each language version, and additional metadata such as links and images. The main contribution of this paper for the NLP community is in the novel dataset which enables studies of disinformation networks, and the training of NLP tools for disinformation detection.

{{</citation>}}


### (18/52) LLM-Prop: Predicting Physical And Electronic Properties Of Crystalline Solids From Their Text Descriptions (Andre Niyongabo Rubungo et al., 2023)

{{<citation>}}

Andre Niyongabo Rubungo, Craig Arnold, Barry P. Rand, Adji Bousso Dieng. (2023)  
**LLM-Prop: Predicting Physical And Electronic Properties Of Crystalline Solids From Their Text Descriptions**  

---
Primary Category: cs.CL  
Categories: cond-mat-mtrl-sci, cs-CL, cs.CL  
Keywords: BERT, GNN  
[Paper Link](http://arxiv.org/abs/2310.14029v1)  

---


**ABSTRACT**  
The prediction of crystal properties plays a crucial role in the crystal design process. Current methods for predicting crystal properties focus on modeling crystal structures using graph neural networks (GNNs). Although GNNs are powerful, accurately modeling the complex interactions between atoms and molecules within a crystal remains a challenge. Surprisingly, predicting crystal properties from crystal text descriptions is understudied, despite the rich information and expressiveness that text data offer. One of the main reasons is the lack of publicly available data for this task. In this paper, we develop and make public a benchmark dataset (called TextEdge) that contains text descriptions of crystal structures with their properties. We then propose LLM-Prop, a method that leverages the general-purpose learning capabilities of large language models (LLMs) to predict the physical and electronic properties of crystals from their text descriptions. LLM-Prop outperforms the current state-of-the-art GNN-based crystal property predictor by about 4% in predicting band gap, 3% in classifying whether the band gap is direct or indirect, and 66% in predicting unit cell volume. LLM-Prop also outperforms a finetuned MatBERT, a domain-specific pre-trained BERT model, despite having 3 times fewer parameters. Our empirical results may highlight the current inability of GNNs to capture information pertaining to space group symmetry and Wyckoff sites for accurate crystal property prediction.

{{</citation>}}


### (19/52) GASCOM: Graph-based Attentive Semantic Context Modeling for Online Conversation Understanding (Vibhor Agarwal et al., 2023)

{{<citation>}}

Vibhor Agarwal, Yu Chen, Nishanth Sastry. (2023)  
**GASCOM: Graph-based Attentive Semantic Context Modeling for Online Conversation Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.14028v1)  

---


**ABSTRACT**  
Online conversation understanding is an important yet challenging NLP problem which has many useful applications (e.g., hate speech detection). However, online conversations typically unfold over a series of posts and replies to those posts, forming a tree structure within which individual posts may refer to semantic context from higher up the tree. Such semantic cross-referencing makes it difficult to understand a single post by itself; yet considering the entire conversation tree is not only difficult to scale but can also be misleading as a single conversation may have several distinct threads or points, not all of which are relevant to the post being considered. In this paper, we propose a Graph-based Attentive Semantic COntext Modeling (GASCOM) framework for online conversation understanding. Specifically, we design two novel algorithms that utilise both the graph structure of the online conversation as well as the semantic information from individual posts for retrieving relevant context nodes from the whole conversation. We further design a token-level multi-head graph attention mechanism to pay different attentions to different tokens from different selected context utterances for fine-grained conversation context modeling. Using this semantic conversational context, we re-examine two well-studied problems: polarity prediction and hate speech detection. Our proposed framework significantly outperforms state-of-the-art methods on both tasks, improving macro-F1 scores by 4.5% for polarity prediction and by 5% for hate speech detection. The GASCOM context weights also enhance interpretability.

{{</citation>}}


### (20/52) Large Language Models and Multimodal Retrieval for Visual Word Sense Disambiguation (Anastasia Kritharoula et al., 2023)

{{<citation>}}

Anastasia Kritharoula, Maria Lymperaiou, Giorgos Stamou. (2023)  
**Large Language Models and Multimodal Retrieval for Visual Word Sense Disambiguation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA, Word Sense Disambiguation  
[Paper Link](http://arxiv.org/abs/2310.14025v1)  

---


**ABSTRACT**  
Visual Word Sense Disambiguation (VWSD) is a novel challenging task with the goal of retrieving an image among a set of candidates, which better represents the meaning of an ambiguous word within a given context. In this paper, we make a substantial step towards unveiling this interesting task by applying a varying set of approaches. Since VWSD is primarily a text-image retrieval task, we explore the latest transformer-based methods for multimodal retrieval. Additionally, we utilize Large Language Models (LLMs) as knowledge bases to enhance the given phrases and resolve ambiguity related to the target word. We also study VWSD as a unimodal problem by converting to text-to-text and image-to-image retrieval, as well as question-answering (QA), to fully explore the capabilities of relevant models. To tap into the implicit knowledge of LLMs, we experiment with Chain-of-Thought (CoT) prompting to guide explainable answer generation. On top of all, we train a learn to rank (LTR) model in order to combine our different modules, achieving competitive ranking results. Extensive experiments on VWSD demonstrate valuable insights to effectively drive future directions.

{{</citation>}}


### (21/52) Toward Stronger Textual Attack Detectors (Pierre Colombo et al., 2023)

{{<citation>}}

Pierre Colombo, Marine Picot, Nathan Noiry, Guillaume Staerman, Pablo Piantanida. (2023)  
**Toward Stronger Textual Attack Detectors**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.14001v1)  

---


**ABSTRACT**  
The landscape of available textual adversarial attacks keeps growing, posing severe threats and raising concerns regarding the deep NLP system's integrity. However, the crucial problem of defending against malicious attacks has only drawn the attention of the NLP community. The latter is nonetheless instrumental in developing robust and trustworthy systems. This paper makes two important contributions in this line of search: (i) we introduce LAROUSSE, a new framework to detect textual adversarial attacks and (ii) we introduce STAKEOUT, a new benchmark composed of nine popular attack methods, three datasets, and two pre-trained models. LAROUSSE is ready-to-use in production as it is unsupervised, hyperparameter-free, and non-differentiable, protecting it against gradient-based methods. Our new benchmark STAKEOUT allows for a robust evaluation framework: we conduct extensive numerical experiments which demonstrate that LAROUSSE outperforms previous methods, and which allows to identify interesting factors of detection rate variations.

{{</citation>}}


### (22/52) Transductive Learning for Textual Few-Shot Classification in API-based Embedding Models (Pierre Colombo et al., 2023)

{{<citation>}}

Pierre Colombo, Victor Pellegrain, Malik Boudiaf, Victor Storchan, Myriam Tami, Ismail Ben Ayed, Celine Hudelot, Pablo Piantanida. (2023)  
**Transductive Learning for Textual Few-Shot Classification in API-based Embedding Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Few-Shot, NLP  
[Paper Link](http://arxiv.org/abs/2310.13998v1)  

---


**ABSTRACT**  
Proprietary and closed APIs are becoming increasingly common to process natural language, and are impacting the practical applications of natural language processing, including few-shot classification. Few-shot classification involves training a model to perform a new classification task with a handful of labeled data. This paper presents three contributions. First, we introduce a scenario where the embedding of a pre-trained model is served through a gated API with compute-cost and data-privacy constraints. Second, we propose a transductive inference, a learning paradigm that has been overlooked by the NLP community. Transductive inference, unlike traditional inductive learning, leverages the statistics of unlabeled data. We also introduce a new parameter-free transductive regularizer based on the Fisher-Rao loss, which can be used on top of the gated API embeddings. This method fully utilizes unlabeled data, does not share any label with the third-party API provider and could serve as a baseline for future research. Third, we propose an improved experimental setting and compile a benchmark of eight datasets involving multiclass classification in four different languages, with up to 151 classes. We evaluate our methods using eight backbone models, along with an episodic evaluation over 1,000 episodes, which demonstrate the superiority of transductive inference over the standard inductive setting.

{{</citation>}}


### (23/52) Emulating the Human Mind: A Neural-symbolic Link Prediction Model with Fast and Slow Reasoning and Filtered Rules (Mohammad Hossein Khojasteh et al., 2023)

{{<citation>}}

Mohammad Hossein Khojasteh, Najmeh Torabian, Ali Farjami, Saeid Hosseini, Behrouz Minaei-Bidgoli. (2023)  
**Emulating the Human Mind: A Neural-symbolic Link Prediction Model with Fast and Slow Reasoning and Filtered Rules**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLI, Natural Language Inference, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.13996v1)  

---


**ABSTRACT**  
Link prediction is an important task in addressing the incompleteness problem of knowledge graphs (KG). Previous link prediction models suffer from issues related to either performance or explanatory capability. Furthermore, models that are capable of generating explanations, often struggle with erroneous paths or reasoning leading to the correct answer. To address these challenges, we introduce a novel Neural-Symbolic model named FaSt-FLiP (stands for Fast and Slow Thinking with Filtered rules for Link Prediction task), inspired by two distinct aspects of human cognition: "commonsense reasoning" and "thinking, fast and slow." Our objective is to combine a logical and neural model for enhanced link prediction. To tackle the challenge of dealing with incorrect paths or rules generated by the logical model, we propose a semi-supervised method to convert rules into sentences. These sentences are then subjected to assessment and removal of incorrect rules using an NLI (Natural Language Inference) model. Our approach to combining logical and neural models involves first obtaining answers from both the logical and neural models. These answers are subsequently unified using an Inference Engine module, which has been realized through both algorithmic implementation and a novel neural model architecture. To validate the efficacy of our model, we conducted a series of experiments. The results demonstrate the superior performance of our model in both link prediction metrics and the generation of more reliable explanations.

{{</citation>}}


### (24/52) On Bilingual Lexicon Induction with Large Language Models (Yaoyiran Li et al., 2023)

{{<citation>}}

Yaoyiran Li, Anna Korhonen, Ivan Vulić. (2023)  
**On Bilingual Lexicon Induction with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.13995v1)  

---


**ABSTRACT**  
Bilingual Lexicon Induction (BLI) is a core task in multilingual NLP that still, to a large extent, relies on calculating cross-lingual word representations. Inspired by the global paradigm shift in NLP towards Large Language Models (LLMs), we examine the potential of the latest generation of LLMs for the development of bilingual lexicons. We ask the following research question: Is it possible to prompt and fine-tune multilingual LLMs (mLLMs) for BLI, and how does this approach compare against and complement current BLI approaches? To this end, we systematically study 1) zero-shot prompting for unsupervised BLI and 2) few-shot in-context prompting with a set of seed translation pairs, both without any LLM fine-tuning, as well as 3) standard BLI-oriented fine-tuning of smaller LLMs. We experiment with 18 open-source text-to-text mLLMs of different sizes (from 0.3B to 13B parameters) on two standard BLI benchmarks covering a range of typologically diverse languages. Our work is the first to demonstrate strong BLI capabilities of text-to-text mLLMs. The results reveal that few-shot prompting with in-context examples from nearest neighbours achieves the best performance, establishing new state-of-the-art BLI scores for many language pairs. We also conduct a series of in-depth analyses and ablation studies, providing more insights on BLI with (m)LLMs, also along with their limitations.

{{</citation>}}


### (25/52) GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4 (Tom Kocmi et al., 2023)

{{<citation>}}

Tom Kocmi, Christian Federmann. (2023)  
**GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.13988v1)  

---


**ABSTRACT**  
This paper introduces GEMBA-MQM, a GPT-based evaluation metric designed to detect translation quality errors, specifically for the quality estimation setting without the need for human reference translations. Based on the power of large language models (LLM), GEMBA-MQM employs a fixed three-shot prompting technique, querying the GPT-4 model to mark error quality spans. Compared to previous works, our method has language-agnostic prompts, thus avoiding the need for manual prompt preparation for new languages.   While preliminary results indicate that GEMBA-MQM achieves state-of-the-art accuracy for system ranking, we advise caution when using it in academic works to demonstrate improvements over other methods due to its dependence on the proprietary, black-box GPT model.

{{</citation>}}


### (26/52) HateRephrase: Zero- and Few-Shot Reduction of Hate Intensity in Online Posts using Large Language Models (Vibhor Agarwal et al., 2023)

{{<citation>}}

Vibhor Agarwal, Yu Chen, Nishanth Sastry. (2023)  
**HateRephrase: Zero- and Few-Shot Reduction of Hate Intensity in Online Posts using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Few-Shot, GPT, GPT-3.5, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13985v1)  

---


**ABSTRACT**  
Hate speech has become pervasive in today's digital age. Although there has been considerable research to detect hate speech or generate counter speech to combat hateful views, these approaches still cannot completely eliminate the potential harmful societal consequences of hate speech -- hate speech, even when detected, can often not be taken down or is often not taken down enough; and hate speech unfortunately spreads quickly, often much faster than any generated counter speech.   This paper investigates a relatively new yet simple and effective approach of suggesting a rephrasing of potential hate speech content even before the post is made. We show that Large Language Models (LLMs) perform well on this task, outperforming state-of-the-art baselines such as BART-Detox. We develop 4 different prompts based on task description, hate definition, few-shot demonstrations and chain-of-thoughts for comprehensive experiments and conduct experiments on open-source LLMs such as LLaMA-1, LLaMA-2 chat, Vicuna as well as OpenAI's GPT-3.5. We propose various evaluation metrics to measure the efficacy of the generated text and ensure the generated text has reduced hate intensity without drastically changing the semantic meaning of the original text.   We find that LLMs with a few-shot demonstrations prompt work the best in generating acceptable hate-rephrased text with semantic meaning similar to the original text. Overall, we find that GPT-3.5 outperforms the baseline and open-source models for all the different kinds of prompts. We also perform human evaluations and interestingly, find that the rephrasings generated by GPT-3.5 outperform even the human-generated ground-truth rephrasings in the dataset. We also conduct detailed ablation studies to investigate why LLMs work satisfactorily on this task and conduct a failure analysis to understand the gaps.

{{</citation>}}


### (27/52) Values, Ethics, Morals? On the Use of Moral Concepts in NLP Research (Karina Vida et al., 2023)

{{<citation>}}

Karina Vida, Judith Simon, Anne Lauscher. (2023)  
**Values, Ethics, Morals? On the Use of Moral Concepts in NLP Research**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.13915v1)  

---


**ABSTRACT**  
With language technology increasingly affecting individuals' lives, many recent works have investigated the ethical aspects of NLP. Among other topics, researchers focused on the notion of morality, investigating, for example, which moral judgements language models make. However, there has been little to no discussion of the terminology and the theories underpinning those efforts and their implications. This lack is highly problematic, as it hides the works' underlying assumptions and hinders a thorough and targeted scientific debate of morality in NLP. In this work, we address this research gap by (a) providing an overview of some important ethical concepts stemming from philosophy and (b) systematically surveying the existing literature on moral NLP w.r.t. their philosophical foundation, terminology, and data basis. For instance, we analyse what ethical theory an approach is based on, how this decision is justified, and what implications it entails. Our findings surveying 92 papers show that, for instance, most papers neither provide a clear definition of the terms they use nor adhere to definitions from philosophy. Finally, (c) we give three recommendations for future research in the field. We hope our work will lead to a more informed, careful, and sound discussion of morality in language technology.

{{</citation>}}


### (28/52) RTSUM: Relation Triple-based Interpretable Summarization with Multi-level Salience Visualization (Seonglae Cho et al., 2023)

{{<citation>}}

Seonglae Cho, Yonggi Cho, HoonJae Lee, Myungha Jang, Jinyoung Yeo, Dongha Lee. (2023)  
**RTSUM: Relation Triple-based Interpretable Summarization with Multi-level Salience Visualization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2310.13895v1)  

---


**ABSTRACT**  
In this paper, we present RTSUM, an unsupervised summarization framework that utilizes relation triples as the basic unit for summarization. Given an input document, RTSUM first selects salient relation triples via multi-level salience scoring and then generates a concise summary from the selected relation triples by using a text-to-text language model. On the basis of RTSUM, we also develop a web demo for an interpretable summarizing tool, providing fine-grained interpretations with the output summary. With support for customization options, our tool visualizes the salience for textual units at three distinct levels: sentences, relation triples, and phrases. The codes,are publicly available.

{{</citation>}}


### (29/52) RECAP: Towards Precise Radiology Report Generation via Dynamic Disease Progression Reasoning (Wenjun Hou et al., 2023)

{{<citation>}}

Wenjun Hou, Yi Cheng, Kaishuai Xu, Wenjie Li, Jiang Liu. (2023)  
**RECAP: Towards Precise Radiology Report Generation via Dynamic Disease Progression Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.13864v1)  

---


**ABSTRACT**  
Automating radiology report generation can significantly alleviate radiologists' workloads. Previous research has primarily focused on realizing highly concise observations while neglecting the precise attributes that determine the severity of diseases (e.g., small pleural effusion). Since incorrect attributes will lead to imprecise radiology reports, strengthening the generation process with precise attribute modeling becomes necessary. Additionally, the temporal information contained in the historical records, which is crucial in evaluating a patient's current condition (e.g., heart size is unchanged), has also been largely disregarded. To address these issues, we propose RECAP, which generates precise and accurate radiology reports via dynamic disease progression reasoning. Specifically, RECAP first predicts the observations and progressions (i.e., spatiotemporal information) given two consecutive radiographs. It then combines the historical records, spatiotemporal information, and radiographs for report generation, where a disease progression graph and dynamic progression reasoning mechanism are devised to accurately select the attributes of each observation and progression. Extensive experiments on two publicly available datasets demonstrate the effectiveness of our model.

{{</citation>}}


## cs.IR (4)



### (30/52) Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels (Honglei Zhuang et al., 2023)

{{<citation>}}

Honglei Zhuang, Zhen Qin, Kai Hui, Junru Wu, Le Yan, Xuanhui Wang, Michael Berdersky. (2023)  
**Beyond Yes and No: Improving Zero-Shot LLM Rankers via Scoring Fine-Grained Relevance Labels**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.14122v1)  

---


**ABSTRACT**  
Zero-shot text rankers powered by recent LLMs achieve remarkable ranking performance by simply prompting. Existing prompts for pointwise LLM rankers mostly ask the model to choose from binary relevance labels like "Yes" and "No". However, the lack of intermediate relevance label options may cause the LLM to provide noisy or biased answers for documents that are partially relevant to the query. We propose to incorporate fine-grained relevance labels into the prompt for LLM rankers, enabling them to better differentiate among documents with different levels of relevance to the query and thus derive a more accurate ranking. We study two variants of the prompt template, coupled with different numbers of relevance levels. Our experiments on 8 BEIR data sets show that adding fine-grained relevance labels significantly improves the performance of LLM rankers.

{{</citation>}}


### (31/52) Unlock Multi-Modal Capability of Dense Retrieval via Visual Module Plugin (Tianshuo Zhou et al., 2023)

{{<citation>}}

Tianshuo Zhou, Sen Mei, Xinze Li, Zhenghao Liu, Chenyan Xiong, Zhiyuan Liu, Yu Gu, Ge Yu. (2023)  
**Unlock Multi-Modal Capability of Dense Retrieval via Visual Module Plugin**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: QA, T5  
[Paper Link](http://arxiv.org/abs/2310.14037v1)  

---


**ABSTRACT**  
This paper proposes Multi-modAl Retrieval model via Visual modulE pLugin (MARVEL) to learn an embedding space for queries and multi-modal documents to conduct retrieval. MARVEL encodes queries and multi-modal documents with a unified encoder model, which helps to alleviate the modality gap between images and texts. Specifically, we enable the image understanding ability of a well-trained dense retriever, T5-ANCE, by incorporating the image features encoded by the visual module as its inputs. To facilitate the multi-modal retrieval tasks, we build the ClueWeb22-MM dataset based on the ClueWeb22 dataset, which regards anchor texts as queries, and exact the related texts and image documents from anchor linked web pages. Our experiments show that MARVEL significantly outperforms the state-of-the-art methods on the multi-modal retrieval dataset WebQA and ClueWeb22-MM. Our further analyses show that the visual module plugin method is tailored to enable the image understanding ability for an existing dense retrieval model. Besides, we also show that the language model has the ability to extract image semantics from image encoders and adapt the image features in the input space of language models. All codes are available at https://github.com/OpenMatch/MARVEL.

{{</citation>}}


### (32/52) Towards dialogue based, computer aided software requirements elicitation (Vasiliy Seibert, 2023)

{{<citation>}}

Vasiliy Seibert. (2023)  
**Towards dialogue based, computer aided software requirements elicitation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13953v1)  

---


**ABSTRACT**  
Several approaches have been presented, which aim to extract models from natural language specifications. These approaches have inherent weaknesses for they assume an initial problem understanding that is perfect, and they leave no room for feedback. Motivated by real-world collaboration settings between requirements engineers and customers, this paper proposes an interaction blueprint that aims for dialogue based, computer aided software requirements analysis. Compared to mere model extraction approaches, this interaction blueprint encourages individuality, creativity and genuine compromise. A simplistic Experiment was conducted to showcase the general idea. This paper discusses the experiment as well as the proposed interaction blueprint and argues, that advancements in natural language processing and generative AI might lead to significant progress in a foreseeable future. However, for that, there is a need to move away from a magical black box expectation and instead moving towards a dialogue based approach that recognizes the individuality that is an undeniable part of requirements engineering.

{{</citation>}}


### (33/52) Meta-optimized Joint Generative and Contrastive Learning for Sequential Recommendation (Yongjing Hao et al., 2023)

{{<citation>}}

Yongjing Hao, Pengpeng Zhao, Junhua Fang, Jianfeng Qu, Guanfeng Liu, Fuzhen Zhuang, Victor S. Sheng, Xiaofang Zhou. (2023)  
**Meta-optimized Joint Generative and Contrastive Learning for Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Contrastive Learning, Seq2Seq, Sequence-to-Sequence  
[Paper Link](http://arxiv.org/abs/2310.13925v1)  

---


**ABSTRACT**  
Sequential Recommendation (SR) has received increasing attention due to its ability to capture user dynamic preferences. Recently, Contrastive Learning (CL) provides an effective approach for sequential recommendation by learning invariance from different views of an input. However, most existing data or model augmentation methods may destroy semantic sequential interaction characteristics and often rely on the hand-crafted property of their contrastive view-generation strategies. In this paper, we propose a Meta-optimized Seq2Seq Generator and Contrastive Learning (Meta-SGCL) for sequential recommendation, which applies the meta-optimized two-step training strategy to adaptive generate contrastive views. Specifically, Meta-SGCL first introduces a simple yet effective augmentation method called Sequence-to-Sequence (Seq2Seq) generator, which treats the Variational AutoEncoders (VAE) as the view generator and can constitute contrastive views while preserving the original sequence's semantics. Next, the model employs a meta-optimized two-step training strategy, which aims to adaptively generate contrastive views without relying on manually designed view-generation techniques. Finally, we evaluate our proposed method Meta-SGCL using three public real-world datasets. Compared with the state-of-the-art methods, our experimental results demonstrate the effectiveness of our model and the code is available.

{{</citation>}}


## cs.CR (1)



### (34/52) Preventing Supply Chain Vulnerabilities in Java with a Fine-Grained Permission Manager (Paschal C. Amusuo et al., 2023)

{{<citation>}}

Paschal C. Amusuo, Kyle A. Robinson, Santiago Torres-Arias, Laurent Simon, James C. Davis. (2023)  
**Preventing Supply Chain Vulnerabilities in Java with a Fine-Grained Permission Manager**  

---
Primary Category: cs.CR  
Categories: K-6-5; D-4-6, cs-CR, cs-SE, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.14117v1)  

---


**ABSTRACT**  
Integrating third-party packages accelerates modern software engineering, but introduces the risk of software supply chain vulnerabilities. Vulnerabilities in applications' dependencies are being exploited worldwide. Often, these exploits leverage features that are present in a package, yet unneeded by an application. Unfortunately, the current generation of permission managers, such as SELinux, Docker containers, and the Java Security Manager, are too coarse-grained to usefully support engineers and operators in mitigating these vulnerabilities. Current approaches offer permissions only at the application's granularity, lumping legitimate operations made by safe packages with illegitimate operations made by exploited packages. This strategy does not reflect modern engineering practice. we need a permission manager capable of distinguishing between actions taken by different packages in an application's supply chain.   In this paper, we describe Next-JSM, the first fine-grained ("supply chain aware") permission manager for Java applications. Next-JSM supports permission management at package-level granularity. Next-JSM faces three key challenges: operating on existing JVMs and without access to application or package source code, minimizing performance overhead in applications with many packages, and helping operators manage finer-grained permissions. We show that these challenges can be addressed through bytecode rewriting; appropriate data structures and algorithms; and an expressive permission notation plus automated tooling to establish default permission. In our evaluation, we report that Next-JSM mitigates 11 of the 12 package vulnerabilities we evaluated and incurs an average 2.72% overhead on the Dacapobench benchmark. Qualitatively, we argue that Next-JSM addresses the shortcomings of the (recently deprecated) Java Security Manager (JSM).

{{</citation>}}


## cs.RO (2)



### (35/52) Learning Reward for Physical Skills using Large Language Model (Yuwei Zeng et al., 2023)

{{<citation>}}

Yuwei Zeng, Yiqing Xu. (2023)  
**Learning Reward for Physical Skills using Large Language Model**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.14092v1)  

---


**ABSTRACT**  
Learning reward functions for physical skills are challenging due to the vast spectrum of skills, the high-dimensionality of state and action space, and nuanced sensory feedback. The complexity of these tasks makes acquiring expert demonstration data both costly and time-consuming. Large Language Models (LLMs) contain valuable task-related knowledge that can aid in learning these reward functions. However, the direct application of LLMs for proposing reward functions has its limitations such as numerical instability and inability to incorporate the environment feedback. We aim to extract task knowledge from LLMs using environment feedback to create efficient reward functions for physical skills. Our approach consists of two components. We first use the LLM to propose features and parameterization of the reward function. Next, we update the parameters of this proposed reward function through an iterative self-alignment process. In particular, this process minimizes the ranking inconsistency between the LLM and our learned reward functions based on the new observations. We validated our method by testing it on three simulated physical skill learning tasks, demonstrating effective support for our design choices.

{{</citation>}}


### (36/52) Concept-based Anomaly Detection in Retail Stores for Automatic Correction using Mobile Robots (Aditya Kapoor et al., 2023)

{{<citation>}}

Aditya Kapoor, Vartika Sengar, Nijil George, Vighnesh Vatsal, Jayavardhana Gubbi, Balamuralidhar P, Arpan Pal. (2023)  
**Concept-based Anomaly Detection in Retail Stores for Automatic Correction using Mobile Robots**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Anomaly Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2310.14063v1)  

---


**ABSTRACT**  
Tracking of inventory and rearrangement of misplaced items are some of the most labor-intensive tasks in a retail environment. While there have been attempts at using vision-based techniques for these tasks, they mostly use planogram compliance for detection of any anomalies, a technique that has been found lacking in robustness and scalability. Moreover, existing systems rely on human intervention to perform corrective actions after detection. In this paper, we present Co-AD, a Concept-based Anomaly Detection approach using a Vision Transformer (ViT) that is able to flag misplaced objects without using a prior knowledge base such as a planogram. It uses an auto-encoder architecture followed by outlier detection in the latent space. Co-AD has a peak success rate of 89.90% on anomaly detection image sets of retail objects drawn from the RP2K dataset, compared to 80.81% on the best-performing baseline of a standard ViT auto-encoder. To demonstrate its utility, we describe a robotic mobile manipulation pipeline to autonomously correct the anomalies flagged by Co-AD. This work is ultimately aimed towards developing autonomous mobile robot solutions that reduce the need for human intervention in retail store management.

{{</citation>}}


## math.NA (1)



### (37/52) Graph Neural Networks and Applied Linear Algebra (Nicholas S. Moore et al., 2023)

{{<citation>}}

Nicholas S. Moore, Eric C. Cyr, Peter Ohm, Christopher M. Siefert, Raymond S. Tuminaro. (2023)  
**Graph Neural Networks and Applied Linear Algebra**  

---
Primary Category: math.NA  
Categories: cs-CE, cs-LG, cs-NA, math-NA, math.NA  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.14084v1)  

---


**ABSTRACT**  
Sparse matrix computations are ubiquitous in scientific computing. With the recent interest in scientific machine learning, it is natural to ask how sparse matrix computations can leverage neural networks (NN). Unfortunately, multi-layer perceptron (MLP) neural networks are typically not natural for either graph or sparse matrix computations. The issue lies with the fact that MLPs require fixed-sized inputs while scientific applications generally generate sparse matrices with arbitrary dimensions and a wide range of nonzero patterns (or matrix graph vertex interconnections). While convolutional NNs could possibly address matrix graphs where all vertices have the same number of nearest neighbors, a more general approach is needed for arbitrary sparse matrices, e.g. arising from discretized partial differential equations on unstructured meshes. Graph neural networks (GNNs) are one approach suitable to sparse matrices. GNNs define aggregation functions (e.g., summations) that operate on variable size input data to produce data of a fixed output size so that MLPs can be applied. The goal of this paper is to provide an introduction to GNNs for a numerical linear algebra audience. Concrete examples are provided to illustrate how many common linear algebra tasks can be accomplished using GNNs. We focus on iterative methods that employ computational kernels such as matrix-vector products, interpolation, relaxation methods, and strength-of-connection measures. Our GNN examples include cases where parameters are determined a-priori as well as cases where parameters must be learned. The intent with this article is to help computational scientists understand how GNNs can be used to adapt machine learning concepts to computational tasks associated with sparse matrices. It is hoped that this understanding will stimulate data-driven extensions of classical sparse linear algebra tasks.

{{</citation>}}


## eess.IV (2)



### (38/52) Unleashing Modified Deep Learning Models in Efficient COVID19 Detection (Md Aminul Islam et al., 2023)

{{<citation>}}

Md Aminul Islam, Shabbir Ahmed Shuvo, Mohammad Abu Tareq Rony, M Raihan, Md Abu Sufian. (2023)  
**Unleashing Modified Deep Learning Models in Efficient COVID19 Detection**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-NE, eess-IV, eess.IV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.14081v1)  

---


**ABSTRACT**  
The COVID19 pandemic, a unique and devastating respiratory disease outbreak, has affected global populations as the disease spreads rapidly. Recent Deep Learning breakthroughs may improve COVID19 prediction and forecasting as a tool of precise and fast detection, however, current methods are still being examined to achieve higher accuracy and precision. This study analyzed the collection contained 8055 CT image samples, 5427 of which were COVID cases and 2628 non COVID. The 9544 Xray samples included 4044 COVID patients and 5500 non COVID cases. The most accurate models are MobileNet V3 (97.872 percent), DenseNet201 (97.567 percent), and GoogleNet Inception V1 (97.643 percent). High accuracy indicates that these models can make many accurate predictions, as well as others, are also high for MobileNetV3 and DenseNet201. An extensive evaluation using accuracy, precision, and recall allows a comprehensive comparison to improve predictive models by combining loss optimization with scalable batch normalization in this study. Our analysis shows that these tactics improve model performance and resilience for advancing COVID19 prediction and detection and shows how Deep Learning can improve disease handling. The methods we suggest would strengthen healthcare systems, policymakers, and researchers to make educated decisions to reduce COVID19 and other contagious diseases.   CCS CONCEPTS Covid,Deep Learning, Image Processing   KEYWORDS Covid, Deep Learning, DenseNet201, MobileNet, ResNet, DenseNet, GoogleNet, Image Processing, Disease Detection.

{{</citation>}}


### (39/52) Ophthalmic Biomarker Detection Using Ensembled Vision Transformers -- Winning Solution to IEEE SPS VIP Cup 2023 (H. A. Z. Sameen Shahgir et al., 2023)

{{<citation>}}

H. A. Z. Sameen Shahgir, Khondker Salman Sayeed, Tanjeem Azwad Zaman, Md. Asif Haider, Sheikh Saifur Rahman Jony, M. Sohel Rahman. (2023)  
**Ophthalmic Biomarker Detection Using Ensembled Vision Transformers -- Winning Solution to IEEE SPS VIP Cup 2023**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.14005v1)  

---


**ABSTRACT**  
This report outlines our approach in the IEEE SPS VIP Cup 2023: Ophthalmic Biomarker Detection competition. Our primary objective in this competition was to identify biomarkers from Optical Coherence Tomography (OCT) images obtained from a diverse range of patients. Using robust augmentations and 5-fold cross-validation, we trained two vision transformer-based models: MaxViT and EVA-02, and ensembled them at inference time. We find MaxViT's use of convolution layers followed by strided attention to be better suited for the detection of local features while EVA-02's use of normal attention mechanism and knowledge distillation is better for detecting global features. Ours was the best-performing solution in the competition, achieving a patient-wise F1 score of 0.814 in the first phase and 0.8527 in the second and final phase of VIP Cup 2023, scoring 3.8% higher than the next-best solution.

{{</citation>}}


## cs.DS (2)



### (40/52) Online Duet between Metric Embeddings and Minimum-Weight Perfect Matchings (Sujoy Bhore et al., 2023)

{{<citation>}}

Sujoy Bhore, Arnold Filtser, Csaba D. Tóth. (2023)  
**Online Duet between Metric Embeddings and Minimum-Weight Perfect Matchings**  

---
Primary Category: cs.DS  
Categories: cs-CG, cs-DS, cs.DS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.14078v1)  

---


**ABSTRACT**  
Low-distortional metric embeddings are a crucial component in the modern algorithmic toolkit. In an online metric embedding, points arrive sequentially and the goal is to embed them into a simple space irrevocably, while minimizing the distortion. Our first result is a deterministic online embedding of a general metric into Euclidean space with distortion $O(\log n)\cdot\min\{\sqrt{\log\Phi},\sqrt{n}\}$ (or, $O(d)\cdot\min\{\sqrt{\log\Phi},\sqrt{n}\}$ if the metric has doubling dimension $d$), solving a conjecture by Newman and Rabinovich (2020), and quadratically improving the dependence on the aspect ratio $\Phi$ from Indyk et al.\ (2010). Our second result is a stochastic embedding of a metric space into trees with expected distortion $O(d\cdot \log\Phi)$, generalizing previous results (Indyk et al.\ (2010), Bartal et al.\ (2020)).   Next, we study the \emph{online minimum-weight perfect matching} problem, where a sequence of $2n$ metric points arrive in pairs, and one has to maintain a perfect matching at all times. We allow recourse (as otherwise the order of arrival determines the matching). The goal is to return a perfect matching that approximates the \emph{minimum-weight} perfect matching at all times, while minimizing the recourse. Our third result is a randomized algorithm with competitive ratio $O(d\cdot \log \Phi)$ and recourse $O(\log \Phi)$ against an oblivious adversary, this result is obtained via our new stochastic online embedding. Our fourth result is a deterministic algorithm against an adaptive adversary, using $O(\log^2 n)$ recourse, that maintains a matching of weight at most $O(\log n)$ times the weight of the MST, i.e., a matching of lightness $O(\log n)$. We complement our upper bounds with a strategy for an oblivious adversary that, with recourse $r$, establishes a lower bound of $\Omega(\frac{\log n}{r \log r})$ for both competitive ratio and lightness.

{{</citation>}}


### (41/52) Fast Approximation of Similarity Graphs with Kernel Density Estimation (Peter Macgregor et al., 2023)

{{<citation>}}

Peter Macgregor, He Sun. (2023)  
**Fast Approximation of Similarity Graphs with Kernel Density Estimation**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs-LG, cs.DS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13870v1)  

---


**ABSTRACT**  
Constructing a similarity graph from a set $X$ of data points in $\mathbb{R}^d$ is the first step of many modern clustering algorithms. However, typical constructions of a similarity graph have high time complexity, and a quadratic space dependency with respect to $|X|$. We address this limitation and present a new algorithmic framework that constructs a sparse approximation of the fully connected similarity graph while preserving its cluster structure. Our presented algorithm is based on the kernel density estimation problem, and is applicable for arbitrary kernel functions. We compare our designed algorithm with the well-known implementations from the scikit-learn library and the FAISS library, and find that our method significantly outperforms the implementation from both libraries on a variety of datasets.

{{</citation>}}


## cs.CY (1)



### (42/52) An Offer you Cannot Refuse? Trends in the Coerciveness of Amazon Book Recommendations (Jonathan H. Rystrøm, 2023)

{{<citation>}}

Jonathan H. Rystrøm. (2023)  
**An Offer you Cannot Refuse? Trends in the Coerciveness of Amazon Book Recommendations**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-SI, cs.CY  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2310.14060v1)  

---


**ABSTRACT**  
Recommender systems can be a helpful tool for recommending content but they can also influence users' preferences. One sociological theory for this influence is that companies are incentivised to influence preferences to make users easier to predict and thus more profitable by making it harder to change preferences. This paper seeks to test that theory empirically. We use \textit{Barrier-to-Exit}, a metric for how difficult it is for users to change preferences, to analyse a large dataset of Amazon Book Ratings from 1998 to 2018. We focus the analysis on users who have changed preferences according to Barrier-to-Exit. To assess the growth of Barrier-to-Exit over time, we developed a linear mixed-effects model with crossed random effects for users and categories. Our findings indicate a highly significant growth of Barrier-to-Exit over time, suggesting that it has become more difficult for the analysed subset of users to change their preferences. However, it should be noted that these findings come with several statistical and methodological caveats including sample bias and construct validity issues related to Barrier-to-Exit. We discuss the strengths and limitations of our approach and its implications. Additionally, we highlight the challenges of creating context-sensitive and generalisable measures for complex socio-technical concepts such as "difficulty to change preferences." We conclude with a call for further research: to curb the potential threats of preference manipulation, we need more measures that allow us to compare commercial as well as non-commercial systems.

{{</citation>}}


## cs.CV (5)



### (43/52) You Only Condense Once: Two Rules for Pruning Condensed Datasets (Yang He et al., 2023)

{{<citation>}}

Yang He, Lingao Xiao, Joey Tianyi Zhou. (2023)  
**You Only Condense Once: Two Rules for Pruning Condensed Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Pruning  
[Paper Link](http://arxiv.org/abs/2310.14019v1)  

---


**ABSTRACT**  
Dataset condensation is a crucial tool for enhancing training efficiency by reducing the size of the training dataset, particularly in on-device scenarios. However, these scenarios have two significant challenges: 1) the varying computational resources available on the devices require a dataset size different from the pre-defined condensed dataset, and 2) the limited computational resources often preclude the possibility of conducting additional condensation processes. We introduce You Only Condense Once (YOCO) to overcome these limitations. On top of one condensed dataset, YOCO produces smaller condensed datasets with two embarrassingly simple dataset pruning rules: Low LBPE Score and Balanced Construction. YOCO offers two key advantages: 1) it can flexibly resize the dataset to fit varying computational constraints, and 2) it eliminates the need for extra condensation processes, which can be computationally prohibitive. Experiments validate our findings on networks including ConvNet, ResNet and DenseNet, and datasets including CIFAR-10, CIFAR-100 and ImageNet. For example, our YOCO surpassed various dataset condensation and dataset pruning methods on CIFAR-10 with ten Images Per Class (IPC), achieving 6.98-8.89% and 6.31-23.92% accuracy gains, respectively. The code is available at: https://github.com/he-y/you-only-condense-once.

{{</citation>}}


### (44/52) Competitive Ensembling Teacher-Student Framework for Semi-Supervised Left Atrium MRI Segmentation (Yuyan Shi et al., 2023)

{{<citation>}}

Yuyan Shi, Yichi Zhang, Shasha Wang. (2023)  
**Competitive Ensembling Teacher-Student Framework for Semi-Supervised Left Atrium MRI Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.13955v1)  

---


**ABSTRACT**  
Semi-supervised learning has greatly advanced medical image segmentation since it effectively alleviates the need of acquiring abundant annotations from experts and utilizes unlabeled data which is much easier to acquire. Among existing perturbed consistency learning methods, mean-teacher model serves as a standard baseline for semi-supervised medical image segmentation. In this paper, we present a simple yet efficient competitive ensembling teacher student framework for semi-supervised for left atrium segmentation from 3D MR images, in which two student models with different task-level disturbances are introduced to learn mutually, while a competitive ensembling strategy is performed to ensemble more reliable information to teacher model. Different from the one-way transfer between teacher and student models, our framework facilitates the collaborative learning procedure of different student models with the guidance of teacher model and motivates different training networks for a competitive learning and ensembling procedure to achieve better performance. We evaluate our proposed method on the public Left Atrium (LA) dataset and it obtains impressive performance gains by exploiting the unlabeled data effectively and outperforms several existing semi-supervised methods.

{{</citation>}}


### (45/52) Fuzzy-NMS: Improving 3D Object Detection with Fuzzy Classification in NMS (Li Wang et al., 2023)

{{<citation>}}

Li Wang, Xinyu Zhang, Fachuan Zhao, Chuze Wu, Yichen Wang, Ziying Song, Lei Yang, Jun Li, Huaping Liu. (2023)  
**Fuzzy-NMS: Improving 3D Object Detection with Fuzzy Classification in NMS**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.13951v1)  

---


**ABSTRACT**  
Non-maximum suppression (NMS) is an essential post-processing module used in many 3D object detection frameworks to remove overlapping candidate bounding boxes. However, an overreliance on classification scores and difficulties in determining appropriate thresholds can affect the resulting accuracy directly. To address these issues, we introduce fuzzy learning into NMS and propose a novel generalized Fuzzy-NMS module to achieve finer candidate bounding box filtering. The proposed Fuzzy-NMS module combines the volume and clustering density of candidate bounding boxes, refining them with a fuzzy classification method and optimizing the appropriate suppression thresholds to reduce uncertainty in the NMS process. Adequate validation experiments are conducted using the mainstream KITTI and large-scale Waymo 3D object detection benchmarks. The results of these tests demonstrate the proposed Fuzzy-NMS module can improve the accuracy of numerous recently NMS-based detectors significantly, including PointPillars, PV-RCNN, and IA-SSD, etc. This effect is particularly evident for small objects such as pedestrians and bicycles. As a plug-and-play module, Fuzzy-NMS does not need to be retrained and produces no obvious increases in inference time.

{{</citation>}}


### (46/52) Exploring Driving Behavior for Autonomous Vehicles Based on Gramian Angular Field Vision Transformer (Junwei You et al., 2023)

{{<citation>}}

Junwei You, Ying Chen, Zhuoyu Jiang, Zhangchi Liu, Zilin Huang, Yifeng Ding, Bin Ran. (2023)  
**Exploring Driving Behavior for Autonomous Vehicles Based on Gramian Angular Field Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.13906v1)  

---


**ABSTRACT**  
Effective classification of autonomous vehicle (AV) driving behavior emerges as a critical area for diagnosing AV operation faults, enhancing autonomous driving algorithms, and reducing accident rates. This paper presents the Gramian Angular Field Vision Transformer (GAF-ViT) model, designed to analyze AV driving behavior. The proposed GAF-ViT model consists of three key components: GAF Transformer Module, Channel Attention Module, and Multi-Channel ViT Module. These modules collectively convert representative sequences of multivariate behavior into multi-channel images and employ image recognition techniques for behavior classification. A channel attention mechanism is applied to multi-channel images to discern the impact of various driving behavior features. Experimental evaluation on the Waymo Open Dataset of trajectories demonstrates that the proposed model achieves state-of-the-art performance. Furthermore, an ablation study effectively substantiates the efficacy of individual modules within the model.

{{</citation>}}


### (47/52) Multimodal Transformer Using Cross-Channel attention for Object Detection in Remote Sensing Images (Bissmella Bahaduri et al., 2023)

{{<citation>}}

Bissmella Bahaduri, Zuheng Ming, Fangchen Feng, Anissa Mokraou. (2023)  
**Multimodal Transformer Using Cross-Channel attention for Object Detection in Remote Sensing Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2310.13876v1)  

---


**ABSTRACT**  
Object detection in Remote Sensing Images (RSI) is a critical task for numerous applications in Earth Observation (EO). Unlike general object detection, object detection in RSI has specific challenges: 1) the scarcity of labeled data in RSI compared to general object detection datasets, and 2) the small objects presented in a high-resolution image with a vast background. To address these challenges, we propose a multimodal transformer exploring multi-source remote sensing data for object detection. Instead of directly combining the multimodal input through a channel-wise concatenation, which ignores the heterogeneity of different modalities, we propose a cross-channel attention module. This module learns the relationship between different channels, enabling the construction of a coherent multimodal input by aligning the different modalities at the early stage. We also introduce a new architecture based on the Swin transformer that incorporates convolution layers in non-shifting blocks while maintaining fixed dimensions, allowing for the generation of fine-to-coarse representations with a favorable accuracy-computation trade-off. The extensive experiments prove the effectiveness of the proposed multimodal fusion module and architecture, demonstrating their applicability to multimodal aerial imagery.

{{</citation>}}


## cs.SE (1)



### (48/52) Advancing Requirements Engineering through Generative AI: Assessing the Role of LLMs (Chetan Arora et al., 2023)

{{<citation>}}

Chetan Arora, John Grundy, Mohamed Abdelrazek. (2023)  
**Advancing Requirements Engineering through Generative AI: Assessing the Role of LLMs**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.13976v1)  

---


**ABSTRACT**  
Requirements Engineering (RE) is a critical phase in software development including the elicitation, analysis, specification, and validation of software requirements. Despite the importance of RE, it remains a challenging process due to the complexities of communication, uncertainty in the early stages and inadequate automation support. In recent years, large-language models (LLMs) have shown significant promise in diverse domains, including natural language processing, code generation, and program understanding. This chapter explores the potential of LLMs in driving RE processes, aiming to improve the efficiency and accuracy of requirements-related tasks. We propose key directions and SWOT analysis for research and development in using LLMs for RE, focusing on the potential for requirements elicitation, analysis, specification, and validation. We further present the results from a preliminary evaluation, in this context.

{{</citation>}}


## cs.FL (1)



### (49/52) Masked Hard-Attention Transformers and Boolean RASP Recognize Exactly the Star-Free Languages (Dana Angluin et al., 2023)

{{<citation>}}

Dana Angluin, David Chiang, Andy Yang. (2023)  
**Masked Hard-Attention Transformers and Boolean RASP Recognize Exactly the Star-Free Languages**  

---
Primary Category: cs.FL  
Categories: cs-FL, cs-LG, cs-LO, cs.FL  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.13897v1)  

---


**ABSTRACT**  
We consider transformer encoders with hard attention (in which all attention is focused on exactly one position) and strict future masking (in which each position only attends to positions strictly to its left), and prove that the class of languages recognized by these networks is exactly the star-free languages. Adding position embeddings increases the class of recognized languages to other well-studied classes. A key technique in these proofs is Boolean RASP, a variant of RASP that is restricted to Boolean values. Via the star-free languages, we relate transformers to first-order logic, temporal logic, and algebraic automata theory.

{{</citation>}}


## cs.HC (1)



### (50/52) GPTutor: an open-source AI pair programming tool alternative to Copilot (Eason Chen et al., 2023)

{{<citation>}}

Eason Chen, Ray Huang, Justa Liang, Damien Chen, Pierce Hung. (2023)  
**GPTutor: an open-source AI pair programming tool alternative to Copilot**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13896v3)  

---


**ABSTRACT**  
This paper presents the latest progress of GPTutor: a ChatGPT-powered programming tool extension in Visual Studio Code. The emergence of Large Language Models (LLMs) has improved software development efficiency, but their performance can be hindered by training data limitations and prompt design issues. Existing LLM development tools often operate as black boxes, with users unable to view the prompts used and unable to improve performance by correcting prompts when errors occur. To address the aforementioned issues, GPTutor was introduced as an open-source AI pair programming tool, offering an alternative to Copilot. GPTutor empowers users to customize prompts for various programming languages and scenarios, with support for 120+ human languages and 50+ programming languages. Users can fine-tune prompts to correct the errors from LLM for precision and efficient code generation. At the end of the paper, we underscore GPTutor's potential through examples, including demonstrating its proficiency in interpreting and generating Sui-Move, a newly introduced smart contract language, using prompt engineering.

{{</citation>}}


## cs.SI (1)



### (51/52) COVIDFakeExplainer: An Explainable Machine Learning based Web Application for Detecting COVID-19 Fake News (Dylan Warman et al., 2023)

{{<citation>}}

Dylan Warman, Muhammad Ashad Kabir. (2023)  
**COVIDFakeExplainer: An Explainable Machine Learning based Web Application for Detecting COVID-19 Fake News**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-CY, cs-SI, cs.SI  
Keywords: AWS, Amazon, BERT, Fake News  
[Paper Link](http://arxiv.org/abs/2310.13890v1)  

---


**ABSTRACT**  
Fake news has emerged as a critical global issue, magnified by the COVID-19 pandemic, underscoring the need for effective preventive tools. Leveraging machine learning, including deep learning techniques, offers promise in combatting fake news. This paper goes beyond by establishing BERT as the superior model for fake news detection and demonstrates its utility as a tool to empower the general populace. We have implemented a browser extension, enhanced with explainability features, enabling real-time identification of fake news and delivering easily interpretable explanations. To achieve this, we have employed two publicly available datasets and created seven distinct data configurations to evaluate three prominent machine learning architectures. Our comprehensive experiments affirm BERT's exceptional accuracy in detecting COVID-19-related fake news. Furthermore, we have integrated an explainability component into the BERT model and deployed it as a service through Amazon's cloud API hosting (AWS). We have developed a browser extension that interfaces with the API, allowing users to select and transmit data from web pages, receiving an intelligible classification in return. This paper presents a practical end-to-end solution, highlighting the feasibility of constructing a holistic system for fake news detection, which can significantly benefit society.

{{</citation>}}


## stat.ML (1)



### (52/52) Distributionally Robust Optimization with Bias and Variance Reduction (Ronak Mehta et al., 2023)

{{<citation>}}

Ronak Mehta, Vincent Roulet, Krishna Pillutla, Zaid Harchaoui. (2023)  
**Distributionally Robust Optimization with Bias and Variance Reduction**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-OC, stat-ML, stat.ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.13863v1)  

---


**ABSTRACT**  
We consider the distributionally robust optimization (DRO) problem with spectral risk-based uncertainty set and $f$-divergence penalty. This formulation includes common risk-sensitive learning objectives such as regularized condition value-at-risk (CVaR) and average top-$k$ loss. We present Prospect, a stochastic gradient-based algorithm that only requires tuning a single learning rate hyperparameter, and prove that it enjoys linear convergence for smooth regularized losses. This contrasts with previous algorithms that either require tuning multiple hyperparameters or potentially fail to converge due to biased gradient estimates or inadequate regularization. Empirically, we show that Prospect can converge 2-3$\times$ faster than baselines such as stochastic gradient and stochastic saddle-point methods on distribution shift and fairness benchmarks spanning tabular, vision, and language domains.

{{</citation>}}
