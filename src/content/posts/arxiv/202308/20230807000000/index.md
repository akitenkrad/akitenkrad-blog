---
draft: false
title: "arXiv @ 2023.08.07"
date: 2023-08-07
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.07"
    identifier: arxiv_20230807
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.DC (1)](#csdc-1)
- [cs.SE (2)](#csse-2)
- [cs.AI (6)](#csai-6)
- [cs.IR (4)](#csir-4)
- [cs.LG (4)](#cslg-4)
- [stat.ML (1)](#statml-1)
- [cs.GR (1)](#csgr-1)
- [cs.CV (12)](#cscv-12)
- [cs.CL (3)](#cscl-3)
- [eess.IV (2)](#eessiv-2)
- [q-fin.PM (1)](#q-finpm-1)
- [cs.MM (1)](#csmm-1)
- [cs.CR (2)](#cscr-2)
- [cs.SI (1)](#cssi-1)
- [cs.HC (1)](#cshc-1)
- [eess.SY (1)](#eesssy-1)

## cs.DC (1)



### (1/43) Resource Management for GPT-based Model Deployed on Clouds: Challenges, Solutions, and Future Directions (Yongkang Dang et al., 2023)

{{<citation>}}

Yongkang Dang, Minxian Xu, Kejiang Ye. (2023)  
**Resource Management for GPT-based Model Deployed on Clouds: Challenges, Solutions, and Future Directions**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Azure, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2308.02970v1)  

---


**ABSTRACT**  
The widespread adoption of the large language model (LLM), e.g. Generative Pre-trained Transformer (GPT), deployed on cloud computing environment (e.g. Azure) has led to a huge increased demand for resources. This surge in demand poses significant challenges to resource management in clouds. This paper aims to highlight these challenges by first identifying the unique characteristics of resource management for the GPT-based model. Building upon this understanding, we analyze the specific challenges faced by resource management in the context of GPT-based model deployed on clouds, and propose corresponding potential solutions. To facilitate effective resource management, we introduce a comprehensive resource management framework and present resource scheduling algorithms specifically designed for the GPT-based model. Furthermore, we delve into the future directions for resource management in the GPT-based model, highlighting potential areas for further exploration and improvement. Through this study, we aim to provide valuable insights into resource management for GPT-based models deployed in clouds and promote their sustainable development for GPT-based models and applications.

{{</citation>}}


## cs.SE (2)



### (2/43) Who is Smarter? An Empirical Study of AI-based Smart Contract Creation (Rabimba Karanjai et al., 2023)

{{<citation>}}

Rabimba Karanjai, Edward Li, Lei Xu, Weidong Shi. (2023)  
**Who is Smarter? An Empirical Study of AI-based Smart Contract Creation**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Google  
[Paper Link](http://arxiv.org/abs/2308.02955v1)  

---


**ABSTRACT**  
The introduction of large language models (LLMs) like ChatGPT and Google Palm2 for smart contract generation seems to be the first well-established instance of an AI pair programmer. LLMs have access to a large number of open-source smart contracts, enabling them to utilize more extensive code in Solidity than other code generation tools. Although the initial and informal assessments of LLMs for smart contract generation are promising, a systematic evaluation is needed to explore the limits and benefits of these models. The main objective of this study is to assess the quality of generated code provided by LLMs for smart contracts. We also aim to evaluate the impact of the quality and variety of input parameters fed to LLMs. To achieve this aim, we created an experimental setup for evaluating the generated code in terms of validity, correctness, and efficiency. Our study finds crucial evidence of security bugs getting introduced in the generated smart contracts as well as the overall quality and correctness of the code getting impacted. However, we also identified the areas where it can be improved. The paper also proposes several potential research directions to improve the process, quality and safety of generated smart contract codes.

{{</citation>}}


### (3/43) LLM is Like a Box of Chocolates: the Non-determinism of ChatGPT in Code Generation (Shuyin Ouyang et al., 2023)

{{<citation>}}

Shuyin Ouyang, Jie M. Zhang, Mark Harman, Meng Wang. (2023)  
**LLM is Like a Box of Chocolates: the Non-determinism of ChatGPT in Code Generation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.02828v1)  

---


**ABSTRACT**  
There has been a recent explosion of research on Large Language Models (LLMs) for software engineering tasks, in particular code generation. However, results from LLMs can be highly unstable; nondeterministically returning very different codes for the same prompt. Non-determinism is a potential menace to scientific conclusion validity. When non-determinism is high, scientific conclusions simply cannot be relied upon unless researchers change their behaviour to control for it in their empirical analyses. This paper conducts an empirical study to demonstrate that non-determinism is, indeed, high, thereby underlining the need for this behavioural change. We choose to study ChatGPT because it is already highly prevalent in the code generation research literature. We report results from a study of 829 code generation problems from three code generation benchmarks (i.e., CodeContests, APPS, and HumanEval). Our results reveal high degrees of non-determinism: the ratio of coding tasks with zero equal test output across different requests is 72.73%, 60.40%, and 65.85% for CodeContests, APPS, and HumanEval, respectively. In addition, we find that setting the temperature to 0 does not guarantee determinism in code generation, although it indeed brings less non-determinism than the default configuration (temperature=1). These results confirm that there is, currently, a significant threat to scientific conclusion validity. In order to put LLM-based research on firmer scientific foundations, researchers need to take into account non-determinism in drawing their conclusions.

{{</citation>}}


## cs.AI (6)



### (4/43) A criterion for Artificial General Intelligence: hypothetic-deductive reasoning, tested on ChatGPT (Louis Vervoort et al., 2023)

{{<citation>}}

Louis Vervoort, Vitaliy Mizyakov, Anastasia Ugleva. (2023)  
**A criterion for Artificial General Intelligence: hypothetic-deductive reasoning, tested on ChatGPT**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2308.02950v1)  

---


**ABSTRACT**  
We argue that a key reasoning skill that any advanced AI, say GPT-4, should master in order to qualify as 'thinking machine', or AGI, is hypothetic-deductive reasoning. Problem-solving or question-answering can quite generally be construed as involving two steps: hypothesizing that a certain set of hypotheses T applies to the problem or question at hand, and deducing the solution or answer from T - hence the term hypothetic-deductive reasoning. An elementary proxy of hypothetic-deductive reasoning is causal reasoning. We propose simple tests for both types of reasoning, and apply them to ChatGPT. Our study shows that, at present, the chatbot has a limited capacity for either type of reasoning, as soon as the problems considered are somewhat complex. However, we submit that if an AI would be capable of this type of reasoning in a sufficiently wide range of contexts, it would be an AGI.

{{</citation>}}


### (5/43) dPASP: A Comprehensive Differentiable Probabilistic Answer Set Programming Environment For Neurosymbolic Learning and Reasoning (Renato Lui Geh et al., 2023)

{{<citation>}}

Renato Lui Geh, Jonas Gonçalves, Igor Cataneo Silveira, Denis Deratani Mauá, Fabio Gagliardi Cozman. (2023)  
**dPASP: A Comprehensive Differentiable Probabilistic Answer Set Programming Environment For Neurosymbolic Learning and Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-LO, cs-NE, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.02944v1)  

---


**ABSTRACT**  
We present dPASP, a novel declarative probabilistic logic programming framework for differentiable neuro-symbolic reasoning. The framework allows for the specification of discrete probabilistic models with neural predicates, logic constraints and interval-valued probabilistic choices, thus supporting models that combine low-level perception (images, texts, etc), common-sense reasoning, and (vague) statistical knowledge. To support all such features, we discuss the several semantics for probabilistic logic programs that can express nondeterministic, contradictory, incomplete and/or statistical knowledge. We also discuss how gradient-based learning can be performed with neural predicates and probabilistic choices under selected semantics. We then describe an implemented package that supports inference and learning in the language, along with several example programs. The package requires minimal user knowledge of deep learning system's inner workings, while allowing end-to-end training of rather sophisticated models and loss functions.

{{</citation>}}


### (6/43) ConvFormer: Revisiting Transformer for Sequential User Modeling (Hao Wang et al., 2023)

{{<citation>}}

Hao Wang, Jianxun Lian, Mingqi Wu, Haoxuan Li, Jiajun Fan, Wanyue Xu, Chaozhuo Li, Xing Xie. (2023)  
**ConvFormer: Revisiting Transformer for Sequential User Modeling**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IR, cs-SI, cs.AI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.02925v1)  

---


**ABSTRACT**  
Sequential user modeling, a critical task in personalized recommender systems, focuses on predicting the next item a user would prefer, requiring a deep understanding of user behavior sequences. Despite the remarkable success of Transformer-based models across various domains, their full potential in comprehending user behavior remains untapped. In this paper, we re-examine Transformer-like architectures aiming to advance state-of-the-art performance. We start by revisiting the core building blocks of Transformer-based methods, analyzing the effectiveness of the item-to-item mechanism within the context of sequential user modeling. After conducting a thorough experimental analysis, we identify three essential criteria for devising efficient sequential user models, which we hope will serve as practical guidelines to inspire and shape future designs. Following this, we introduce ConvFormer, a simple but powerful modification to the Transformer architecture that meets these criteria, yielding state-of-the-art results. Additionally, we present an acceleration technique to minimize the complexity associated with processing extremely long sequences. Experiments on four public datasets showcase ConvFormer's superiority and confirm the validity of our proposed criteria.

{{</citation>}}


### (7/43) Anomaly Detection in Global Financial Markets with Graph Neural Networks and Nonextensive Entropy (Kleyton da Costa, 2023)

{{<citation>}}

Kleyton da Costa. (2023)  
**Anomaly Detection in Global Financial Markets with Graph Neural Networks and Nonextensive Entropy**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, q-fin-GN  
Keywords: Anomaly Detection, Financial, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.02914v1)  

---


**ABSTRACT**  
Anomaly detection is a challenging task, particularly in systems with many variables. Anomalies are outliers that statistically differ from the analyzed data and can arise from rare events, malfunctions, or system misuse. This study investigated the ability to detect anomalies in global financial markets through Graph Neural Networks (GNN) considering an uncertainty scenario measured by a nonextensive entropy. The main findings show that the complex structure of highly correlated assets decreases in a crisis, and the number of anomalies is statistically different for nonextensive entropy parameters considering before, during, and after crisis.

{{</citation>}}


### (8/43) feather -- a Python SDK to share and deploy models (Nihir Vedd et al., 2023)

{{<citation>}}

Nihir Vedd, Paul Riga. (2023)  
**feather -- a Python SDK to share and deploy models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02838v1)  

---


**ABSTRACT**  
At its core, feather was a tool that allowed model developers to build shareable user interfaces for their models in under 20 lines of code. Using the Python SDK, developers specified visual components that users would interact with. (e.g. a FileUpload component to allow users to upload a file). Our service then provided 1) a URL that allowed others to access and use the model visually via a user interface; 2) an API endpoint to allow programmatic requests to a model. In this paper, we discuss feather's motivations and the value we intended to offer AI researchers and developers. For example, the SDK can support multi-step models and can be extended to run automatic evaluation against held out datasets. We additionally provide comprehensive technical and implementation details.   N.B. feather is presently a dormant project. We have open sourced our code for research purposes: https://github.com/feather-ai/

{{</citation>}}


### (9/43) Physics-Based Task Generation Through Causal Sequence of Physical Interactions (Chathura Gamage et al., 2023)

{{<citation>}}

Chathura Gamage, Vimukthini Pinto, Matthew Stephenson, Jochen Renz. (2023)  
**Physics-Based Task Generation Through Causal Sequence of Physical Interactions**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02835v1)  

---


**ABSTRACT**  
Performing tasks in a physical environment is a crucial yet challenging problem for AI systems operating in the real world. Physics simulation-based tasks are often employed to facilitate research that addresses this challenge. In this paper, first, we present a systematic approach for defining a physical scenario using a causal sequence of physical interactions between objects. Then, we propose a methodology for generating tasks in a physics-simulating environment using these defined scenarios as inputs. Our approach enables a better understanding of the granular mechanics required for solving physics-based tasks, thereby facilitating accurate evaluation of AI systems' physical reasoning capabilities. We demonstrate our proposed task generation methodology using the physics-based puzzle game Angry Birds and evaluate the generated tasks using a range of metrics, including physical stability, solvability using intended physical interactions, and accidental solvability using unintended solutions. We believe that the tasks generated using our proposed methodology can facilitate a nuanced evaluation of physical reasoning agents, thus paving the way for the development of agents for more sophisticated real-world applications.

{{</citation>}}


## cs.IR (4)



### (10/43) Towards Consistency Filtering-Free Unsupervised Learning for Dense Retrieval (Haoxiang Shi et al., 2023)

{{<citation>}}

Haoxiang Shi, Sumio Fujita, Tetsuya Sakai. (2023)  
**Towards Consistency Filtering-Free Unsupervised Learning for Dense Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs-LG, cs-NI, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2308.02926v1)  

---


**ABSTRACT**  
Domain transfer is a prevalent challenge in modern neural Information Retrieval (IR). To overcome this problem, previous research has utilized domain-specific manual annotations and synthetic data produced by consistency filtering to finetune a general ranker and produce a domain-specific ranker. However, training such consistency filters are computationally expensive, which significantly reduces the model efficiency. In addition, consistency filtering often struggles to identify retrieval intentions and recognize query and corpus distributions in a target domain. In this study, we evaluate a more efficient solution: replacing the consistency filter with either direct pseudo-labeling, pseudo-relevance feedback, or unsupervised keyword generation methods for achieving consistent filtering-free unsupervised dense retrieval. Our extensive experimental evaluations demonstrate that, on average, TextRank-based pseudo relevance feedback outperforms other methods. Furthermore, we analyzed the training and inference efficiency of the proposed paradigm. The results indicate that filtering-free unsupervised learning can continuously improve training and inference efficiency while maintaining retrieval performance. In some cases, it can even improve performance based on particular datasets.

{{</citation>}}


### (11/43) Disentangled Counterfactual Reasoning for Unbiased Sequential Recommendation (Yi Ren et al., 2023)

{{<citation>}}

Yi Ren, Xu Zhao, Hongyan Tang, Shuai Li. (2023)  
**Disentangled Counterfactual Reasoning for Unbiased Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.02900v1)  

---


**ABSTRACT**  
Sequential recommender systems have achieved state-of-the-art recommendation performance by modeling the sequential dynamics of user activities. However, in most recommendation scenarios, the popular items comprise the major part of the previous user actions. Therefore, the learned models are biased towards the popular items irrespective of the user's real interests. In this paper, we propose a structural causal model-based method to address the popularity bias issue for sequential recommendation model learning. For more generalizable modeling, we disentangle the popularity and interest representations at both the item side and user context side. Based on the disentangled representation, we identify a more effective structural causal graph for general recommendation applications. Then, we design delicate sequential models to apply the aforementioned causal graph to the sequential recommendation scenario for unbiased prediction with counterfactual reasoning. Furthermore, we conduct extensive offline experiments and online A/B tests to verify the proposed \textbf{DCR} (Disentangled Counterfactual Reasoning) method's superior overall performance and understand the effectiveness of the various introduced components. Based on our knowledge, this is the first structural causal model specifically designed for the popularity bias correction of sequential recommendation models, which achieves significant performance gains over the existing methods.

{{</citation>}}


### (12/43) Group Membership Bias (Ali Vardasbi et al., 2023)

{{<citation>}}

Ali Vardasbi, Maarten de Rijke, Fernando Diaz, Mostafa Dehghani. (2023)  
**Group Membership Bias**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.02887v1)  

---


**ABSTRACT**  
When learning to rank from user interactions, search and recommendation systems must address biases in user behavior to provide a high-quality ranking. One type of bias that has recently been studied in the ranking literature is when sensitive attributes, such as gender, have an impact on a user's judgment about an item's utility. For example, in a search for an expertise area, some users may be biased towards clicking on male candidates over female candidates. We call this type of bias group membership bias or group bias for short. Increasingly, we seek rankings that not only have high utility but are also fair to individuals and sensitive groups. Merit-based fairness measures rely on the estimated merit or utility of the items. With group bias, the utility of the sensitive groups is under-estimated, hence, without correcting for this bias, a supposedly fair ranking is not truly fair. In this paper, first, we analyze the impact of group bias on ranking quality as well as two well-known merit-based fairness metrics and show that group bias can hurt both ranking and fairness. Then, we provide a correction method for group bias that is based on the assumption that the utility score of items in different groups comes from the same distribution. This assumption has two potential issues of sparsity and equality-instead-of-equity, which we use an amortized approach to solve. We show that our correction method can consistently compensate for the negative impact of group bias on ranking quality and fairness metrics.

{{</citation>}}


### (13/43) Bootstrapping Contrastive Learning Enhanced Music Cold-Start Matching (Xinping Zhao et al., 2023)

{{<citation>}}

Xinping Zhao, Ying Zhang, Qiang Xiao, Yuming Ren, Yingchun Yang. (2023)  
**Bootstrapping Contrastive Learning Enhanced Music Cold-Start Matching**  

---
Primary Category: cs.IR  
Categories: F-2-2; I-2-8, cs-IR, cs-SD, cs.IR, eess-AS  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.02844v1)  

---


**ABSTRACT**  
We study a particular matching task we call Music Cold-Start Matching. In short, given a cold-start song request, we expect to retrieve songs with similar audiences and then fastly push the cold-start song to the audiences of the retrieved songs to warm up it. However, there are hardly any studies done on this task. Therefore, in this paper, we will formalize the problem of Music Cold-Start Matching detailedly and give a scheme. During the offline training, we attempt to learn high-quality song representations based on song content features. But, we find supervision signals typically follow power-law distribution causing skewed representation learning. To address this issue, we propose a novel contrastive learning paradigm named Bootstrapping Contrastive Learning (BCL) to enhance the quality of learned representations by exerting contrastive regularization. During the online serving, to locate the target audiences more accurately, we propose Clustering-based Audience Targeting (CAT) that clusters audience representations to acquire a few cluster centroids and then locate the target audiences by measuring the relevance between the audience representations and the cluster centroids. Extensive experiments on the offline dataset and online system demonstrate the effectiveness and efficiency of our method. Currently, we have deployed it on NetEase Cloud Music, affecting millions of users. Code will be released in the future.

{{</citation>}}


## cs.LG (4)



### (14/43) An AI-Enabled Framework to Defend Ingenious MDT-based Attacks on the Emerging Zero Touch Cellular Networks (Aneeqa Ijaz et al., 2023)

{{<citation>}}

Aneeqa Ijaz, Waseem Raza, Hasan Farooq, Marvin Manalastas, Ali Imran. (2023)  
**An AI-Enabled Framework to Defend Ingenious MDT-based Attacks on the Emerging Zero Touch Cellular Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02923v1)  

---


**ABSTRACT**  
Deep automation provided by self-organizing network (SON) features and their emerging variants such as zero touch automation solutions is a key enabler for increasingly dense wireless networks and pervasive Internet of Things (IoT). To realize their objectives, most automation functionalities rely on the Minimization of Drive Test (MDT) reports. The MDT reports are used to generate inferences about network state and performance, thus dynamically change network parameters accordingly. However, the collection of MDT reports from commodity user devices, particularly low cost IoT devices, make them a vulnerable entry point to launch an adversarial attack on emerging deeply automated wireless networks. This adds a new dimension to the security threats in the IoT and cellular networks. Existing literature on IoT, SON, or zero touch automation does not address this important problem. In this paper, we investigate an impactful, first of its kind adversarial attack that can be launched by exploiting the malicious MDT reports from the compromised user equipment (UE). We highlight the detrimental repercussions of this attack on the performance of common network automation functions. We also propose a novel Malicious MDT Reports Identification framework (MRIF) as a countermeasure to detect and eliminate the malicious MDT reports using Machine Learning and verify it through a use-case. Thus, the defense mechanism can provide the resilience and robustness for zero touch automation SON engines against the adversarial MDT attacks

{{</citation>}}


### (15/43) Adversarial Erasing with Pruned Elements: Towards Better Graph Lottery Ticket (Yuwen Wang et al., 2023)

{{<citation>}}

Yuwen Wang, Shunyu Liu, Kaixuan Chen, Tongtian Zhu, Ji Qiao, Mengjie Shi, Yuanyu Wan, Mingli Song. (2023)  
**Adversarial Erasing with Pruned Elements: Towards Better Graph Lottery Ticket**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.02916v1)  

---


**ABSTRACT**  
Graph Lottery Ticket (GLT), a combination of core subgraph and sparse subnetwork, has been proposed to mitigate the computational cost of deep Graph Neural Networks (GNNs) on large input graphs while preserving original performance. However, the winning GLTs in exisiting studies are obtained by applying iterative magnitude-based pruning (IMP) without re-evaluating and re-considering the pruned information, which disregards the dynamic changes in the significance of edges/weights during graph/model structure pruning, and thus limits the appeal of the winning tickets. In this paper, we formulate a conjecture, i.e., existing overlooked valuable information in the pruned graph connections and model parameters which can be re-grouped into GLT to enhance the final performance. Specifically, we propose an adversarial complementary erasing (ACE) framework to explore the valuable information from the pruned components, thereby developing a more powerful GLT, referred to as the ACE-GLT. The main idea is to mine valuable information from pruned edges/weights after each round of IMP, and employ the ACE technique to refine the GLT processing. Finally, experimental results demonstrate that our ACE-GLT outperforms existing methods for searching GLT in diverse tasks. Our code will be made publicly available.

{{</citation>}}


### (16/43) OBESEYE: Interpretable Diet Recommender for Obesity Management using Machine Learning and Explainable AI (Mrinmoy Roy et al., 2023)

{{<citation>}}

Mrinmoy Roy, Srabonti Das, Anica Tasnim Protity. (2023)  
**OBESEYE: Interpretable Diet Recommender for Obesity Management using Machine Learning and Explainable AI**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02796v1)  

---


**ABSTRACT**  
Obesity, the leading cause of many non-communicable diseases, occurs mainly for eating more than our body requirements and lack of proper activity. So, being healthy requires heathy diet plans, especially for patients with comorbidities. But it is difficult to figure out the exact quantity of each nutrient because nutrients requirement varies based on physical and disease conditions. In our study we proposed a novel machine learning based system to predict the amount of nutrients one individual requires for being healthy. We applied different machine learning algorithms: linear regression, support vector machine (SVM), decision tree, random forest, XGBoost, LightGBM on fluid and 3 other major micronutrients: carbohydrate, protein, fat consumption prediction. We achieved high accuracy with low root mean square error (RMSE) by using linear regression in fluid prediction, random forest in carbohydrate prediction and LightGBM in protein and fat prediction. We believe our diet recommender system, OBESEYE, is the only of its kind which recommends diet with the consideration of comorbidities and physical conditions and promote encouragement to get rid of obesity.

{{</citation>}}


### (17/43) DaMSTF: Domain Adversarial Learning Enhanced Meta Self-Training for Domain Adaptation (Menglong Lu et al., 2023)

{{<citation>}}

Menglong Lu, Zhen Huang, Yunxiang Zhao, Zhiliang Tian, Yang Liu, Dongsheng Li. (2023)  
**DaMSTF: Domain Adversarial Learning Enhanced Meta Self-Training for Domain Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.02753v1)  

---


**ABSTRACT**  
Self-training emerges as an important research line on domain adaptation. By taking the model's prediction as the pseudo labels of the unlabeled data, self-training bootstraps the model with pseudo instances in the target domain. However, the prediction errors of pseudo labels (label noise) challenge the performance of self-training. To address this problem, previous approaches only use reliable pseudo instances, i.e., pseudo instances with high prediction confidence, to retrain the model. Although these strategies effectively reduce the label noise, they are prone to miss the hard examples. In this paper, we propose a new self-training framework for domain adaptation, namely Domain adversarial learning enhanced Self-Training Framework (DaMSTF). Firstly, DaMSTF involves meta-learning to estimate the importance of each pseudo instance, so as to simultaneously reduce the label noise and preserve hard examples. Secondly, we design a meta constructor for constructing the meta-validation set, which guarantees the effectiveness of the meta-learning module by improving the quality of the meta-validation set. Thirdly, we find that the meta-learning module suffers from the training guidance vanishment and tends to converge to an inferior optimal. To this end, we employ domain adversarial learning as a heuristic neural network initialization method, which can help the meta-learning module converge to a better optimal. Theoretically and experimentally, we demonstrate the effectiveness of the proposed DaMSTF. On the cross-domain sentiment classification task, DaMSTF improves the performance of BERT with an average of nearly 4%.

{{</citation>}}


## stat.ML (1)



### (18/43) Structured Low-Rank Tensors for Generalized Linear Models (Batoul Taki et al., 2023)

{{<citation>}}

Batoul Taki, Anand D. Sarwate, Waheed U. Bajwa. (2023)  
**Structured Low-Rank Tensors for Generalized Linear Models**  

---
Primary Category: stat.ML  
Categories: cs-LG, eess-SP, math-ST, stat-ML, stat-TH, stat.ML  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2308.02922v1)  

---


**ABSTRACT**  
Recent works have shown that imposing tensor structures on the coefficient tensor in regression problems can lead to more reliable parameter estimation and lower sample complexity compared to vector-based methods. This work investigates a new low-rank tensor model, called Low Separation Rank (LSR), in Generalized Linear Model (GLM) problems. The LSR model -- which generalizes the well-known Tucker and CANDECOMP/PARAFAC (CP) models, and is a special case of the Block Tensor Decomposition (BTD) model -- is imposed onto the coefficient tensor in the GLM model. This work proposes a block coordinate descent algorithm for parameter estimation in LSR-structured tensor GLMs. Most importantly, it derives a minimax lower bound on the error threshold on estimating the coefficient tensor in LSR tensor GLM problems. The minimax bound is proportional to the intrinsic degrees of freedom in the LSR tensor GLM problem, suggesting that its sample complexity may be significantly lower than that of vectorized GLMs. This result can also be specialised to lower bound the estimation error in CP and Tucker-structured GLMs. The derived bounds are comparable to tight bounds in the literature for Tucker linear regression, and the tightness of the minimax lower bound is further assessed numerically. Finally, numerical experiments on synthetic datasets demonstrate the efficacy of the proposed LSR tensor model for three regression types (linear, logistic and Poisson). Experiments on a collection of medical imaging datasets demonstrate the usefulness of the LSR model over other tensor models (Tucker and CP) on real, imbalanced data with limited available samples.

{{</citation>}}


## cs.GR (1)



### (19/43) DiffDance: Cascaded Human Motion Diffusion Model for Dance Generation (Qiaosong Qi et al., 2023)

{{<citation>}}

Qiaosong Qi, Le Zhuo, Aixi Zhang, Yue Liao, Fei Fang, Si Liu, Shuicheng Yan. (2023)  
**DiffDance: Cascaded Human Motion Diffusion Model for Dance Generation**  

---
Primary Category: cs.GR  
Categories: cs-CV, cs-GR, cs-SD, cs.GR, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02915v1)  

---


**ABSTRACT**  
When hearing music, it is natural for people to dance to its rhythm. Automatic dance generation, however, is a challenging task due to the physical constraints of human motion and rhythmic alignment with target music. Conventional autoregressive methods introduce compounding errors during sampling and struggle to capture the long-term structure of dance sequences. To address these limitations, we present a novel cascaded motion diffusion model, DiffDance, designed for high-resolution, long-form dance generation. This model comprises a music-to-dance diffusion model and a sequence super-resolution diffusion model. To bridge the gap between music and motion for conditional generation, DiffDance employs a pretrained audio representation learning model to extract music embeddings and further align its embedding space to motion via contrastive loss. During training our cascaded diffusion model, we also incorporate multiple geometric losses to constrain the model outputs to be physically plausible and add a dynamic loss weight that adaptively changes over diffusion timesteps to facilitate sample diversity. Through comprehensive experiments performed on the benchmark dataset AIST++, we demonstrate that DiffDance is capable of generating realistic dance sequences that align effectively with the input music. These results are comparable to those achieved by state-of-the-art autoregressive methods.

{{</citation>}}


## cs.CV (12)



### (20/43) Where and How: Mitigating Confusion in Neural Radiance Fields from Sparse Inputs (Yanqi Bao et al., 2023)

{{<citation>}}

Yanqi Bao, Yuxin Li, Jing Huo, Tianyu Ding, Xinyue Liang, Wenbin Li, Yang Gao. (2023)  
**Where and How: Mitigating Confusion in Neural Radiance Fields from Sparse Inputs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.02908v1)  

---


**ABSTRACT**  
Neural Radiance Fields from Sparse input} (NeRF-S) have shown great potential in synthesizing novel views with a limited number of observed viewpoints. However, due to the inherent limitations of sparse inputs and the gap between non-adjacent views, rendering results often suffer from over-fitting and foggy surfaces, a phenomenon we refer to as "CONFUSION" during volume rendering. In this paper, we analyze the root cause of this confusion and attribute it to two fundamental questions: "WHERE" and "HOW". To this end, we present a novel learning framework, WaH-NeRF, which effectively mitigates confusion by tackling the following challenges: (i)"WHERE" to Sample? in NeRF-S -- we introduce a Deformable Sampling strategy and a Weight-based Mutual Information Loss to address sample-position confusion arising from the limited number of viewpoints; and (ii) "HOW" to Predict? in NeRF-S -- we propose a Semi-Supervised NeRF learning Paradigm based on pose perturbation and a Pixel-Patch Correspondence Loss to alleviate prediction confusion caused by the disparity between training and testing viewpoints. By integrating our proposed modules and loss functions, WaH-NeRF outperforms previous methods under the NeRF-S setting. Code is available https://github.com/bbbbby-99/WaH-NeRF.

{{</citation>}}


### (21/43) An Adaptive Model Ensemble Adversarial Attack for Boosting Adversarial Transferability (Bin Chen et al., 2023)

{{<citation>}}

Bin Chen, Jia-Li Yin, Shukai Chen, Bo-Hao Chen, Ximeng Liu. (2023)  
**An Adaptive Model Ensemble Adversarial Attack for Boosting Adversarial Transferability**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2308.02897v1)  

---


**ABSTRACT**  
While the transferability property of adversarial examples allows the adversary to perform black-box attacks (i.e., the attacker has no knowledge about the target model), the transfer-based adversarial attacks have gained great attention. Previous works mostly study gradient variation or image transformations to amplify the distortion on critical parts of inputs. These methods can work on transferring across models with limited differences, i.e., from CNNs to CNNs, but always fail in transferring across models with wide differences, such as from CNNs to ViTs. Alternatively, model ensemble adversarial attacks are proposed to fuse outputs from surrogate models with diverse architectures to get an ensemble loss, making the generated adversarial example more likely to transfer to other models as it can fool multiple models concurrently. However, existing ensemble attacks simply fuse the outputs of the surrogate models evenly, thus are not efficacious to capture and amplify the intrinsic transfer information of adversarial examples. In this paper, we propose an adaptive ensemble attack, dubbed AdaEA, to adaptively control the fusion of the outputs from each model, via monitoring the discrepancy ratio of their contributions towards the adversarial objective. Furthermore, an extra disparity-reduced filter is introduced to further synchronize the update direction. As a result, we achieve considerable improvement over the existing ensemble attacks on various datasets, and the proposed AdaEA can also boost existing transfer-based attacks, which further demonstrates its efficacy and versatility.

{{</citation>}}


### (22/43) Cross-modal & Cross-domain Learning for Unsupervised LiDAR Semantic Segmentation (Yiyang Chen et al., 2023)

{{<citation>}}

Yiyang Chen, Shanshan Zhao, Changxing Ding, Liyao Tang, Chaoyue Wang, Dacheng Tao. (2023)  
**Cross-modal & Cross-domain Learning for Unsupervised LiDAR Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.02883v1)  

---


**ABSTRACT**  
In recent years, cross-modal domain adaptation has been studied on the paired 2D image and 3D LiDAR data to ease the labeling costs for 3D LiDAR semantic segmentation (3DLSS) in the target domain. However, in such a setting the paired 2D and 3D data in the source domain are still collected with additional effort. Since the 2D-3D projections can enable the 3D model to learn semantic information from the 2D counterpart, we ask whether we could further remove the need of source 3D data and only rely on the source 2D images. To answer it, this paper studies a new 3DLSS setting where a 2D dataset (source) with semantic annotations and a paired but unannotated 2D image and 3D LiDAR data (target) are available. To achieve 3DLSS in this scenario, we propose Cross-Modal and Cross-Domain Learning (CoMoDaL). Specifically, our CoMoDaL aims at modeling 1) inter-modal cross-domain distillation between the unpaired source 2D image and target 3D LiDAR data, and 2) the intra-domain cross-modal guidance between the target 2D image and 3D LiDAR data pair. In CoMoDaL, we propose to apply several constraints, such as point-to-pixel and prototype-to-pixel alignments, to associate the semantics in different modalities and domains by constructing mixed samples in two modalities. The experimental results on several datasets show that in the proposed setting, the developed CoMoDaL can achieve segmentation without the supervision of labeled LiDAR data. Ablations are also conducted to provide more analysis. Code will be available publicly.

{{</citation>}}


### (23/43) Sketch and Text Guided Diffusion Model for Colored Point Cloud Generation (Zijie Wu et al., 2023)

{{<citation>}}

Zijie Wu, Yaonan Wang, Mingtao Feng, He Xie, Ajmal Mian. (2023)  
**Sketch and Text Guided Diffusion Model for Colored Point Cloud Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2308.02874v1)  

---


**ABSTRACT**  
Diffusion probabilistic models have achieved remarkable success in text guided image generation. However, generating 3D shapes is still challenging due to the lack of sufficient data containing 3D models along with their descriptions. Moreover, text based descriptions of 3D shapes are inherently ambiguous and lack details. In this paper, we propose a sketch and text guided probabilistic diffusion model for colored point cloud generation that conditions the denoising process jointly with a hand drawn sketch of the object and its textual description. We incrementally diffuse the point coordinates and color values in a joint diffusion process to reach a Gaussian distribution. Colored point cloud generation thus amounts to learning the reverse diffusion process, conditioned by the sketch and text, to iteratively recover the desired shape and color. Specifically, to learn effective sketch-text embedding, our model adaptively aggregates the joint embedding of text prompt and the sketch based on a capsule attention network. Our model uses staged diffusion to generate the shape and then assign colors to different parts conditioned on the appearance prompt while preserving precise shapes from the first stage. This gives our model the flexibility to extend to multiple tasks, such as appearance re-editing and part segmentation. Experimental results demonstrate that our model outperforms recent state-of-the-art in point cloud generation.

{{</citation>}}


### (24/43) NP-SemiSeg: When Neural Processes meet Semi-Supervised Semantic Segmentation (Jianfeng Wang et al., 2023)

{{<citation>}}

Jianfeng Wang, Daniela Massiceti, Xiaolin Hu, Vladimir Pavlovic, Thomas Lukasiewicz. (2023)  
**NP-SemiSeg: When Neural Processes meet Semi-Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.02866v1)  

---


**ABSTRACT**  
Semi-supervised semantic segmentation involves assigning pixel-wise labels to unlabeled images at training time. This is useful in a wide range of real-world applications where collecting pixel-wise labels is not feasible in time or cost. Current approaches to semi-supervised semantic segmentation work by predicting pseudo-labels for each pixel from a class-wise probability distribution output by a model. If the predicted probability distribution is incorrect, however, this leads to poor segmentation results, which can have knock-on consequences in safety critical systems, like medical images or self-driving cars. It is, therefore, important to understand what a model does not know, which is mainly achieved by uncertainty quantification. Recently, neural processes (NPs) have been explored in semi-supervised image classification, and they have been a computationally efficient and effective method for uncertainty quantification. In this work, we move one step forward by adapting NPs to semi-supervised semantic segmentation, resulting in a new model called NP-SemiSeg. We experimentally evaluated NP-SemiSeg on the public benchmarks PASCAL VOC 2012 and Cityscapes, with different training settings, and the results verify its effectiveness.

{{</citation>}}


### (25/43) Improving Generalization of Image Captioning with Unsupervised Prompt Learning (Hongchen Wei et al., 2023)

{{<citation>}}

Hongchen Wei, Zhenzhong Chen. (2023)  
**Improving Generalization of Image Captioning with Unsupervised Prompt Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2308.02862v1)  

---


**ABSTRACT**  
Pretrained visual-language models have demonstrated impressive zero-shot abilities in image captioning, when accompanied by hand-crafted prompts. Meanwhile, hand-crafted prompts utilize human prior knowledge to guide the model. However, due to the diversity between different domains, such hand-crafted prompt that provide invariant prior knowledge may result in mode collapse for some domains. Some researches attempted to incorporate expert knowledge and instruction datasets, but the results were costly and led to hallucinations. In this paper, we propose an unsupervised prompt learning method to improve Generalization of Image Captioning (GeneIC), which learns a domain-specific prompt vector for the target domain without requiring annotated data. GeneIC aligns visual and language modalities with a pre-trained Contrastive Language-Image Pre-Training (CLIP) model, thus optimizing the domain-specific prompt vector from two aspects: attribute and semantic consistency. Specifically, GeneIC first generates attribute-transferred images with differing attributes, while retaining semantic similarity with original images. Then, GeneIC uses CLIP to measure the similarity between the images and the generated sentences. By exploring the variable and invariant features in the original images and attribute-transferred images, attribute consistency constrains the attribute change direction of both images and sentences to learn domain-specific knowledge. The semantic consistency directly measures the similarity between the generated sentences and images to ensure the accuracy and comprehensiveness of the generated sentences. Consequently, GeneIC only optimizes the prompt vectors, which effectively retains the knowledge in the large model and introduces domain-specific knowledge.

{{</citation>}}


### (26/43) A Comprehensive Analysis of Real-World Image Captioning and Scene Identification (Sai Suprabhanu Nallapaneni et al., 2023)

{{<citation>}}

Sai Suprabhanu Nallapaneni, Subrahmanyam Konakanchi. (2023)  
**A Comprehensive Analysis of Real-World Image Captioning and Scene Identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2308.02833v1)  

---


**ABSTRACT**  
Image captioning is a computer vision task that involves generating natural language descriptions for images. This method has numerous applications in various domains, including image retrieval systems, medicine, and various industries. However, while there has been significant research in image captioning, most studies have focused on high quality images or controlled environments, without exploring the challenges of real-world image captioning. Real-world image captioning involves complex and dynamic environments with numerous points of attention, with images which are often very poor in quality, making it a challenging task, even for humans. This paper evaluates the performance of various models that are built on top of different encoding mechanisms, language decoders and training procedures using a newly created real-world dataset that consists of over 800+ images of over 65 different scene classes, built using MIT Indoor scenes dataset. This dataset is captioned using the IC3 approach that generates more descriptive captions by summarizing the details that are covered by standard image captioning models from unique view-points of the image.

{{</citation>}}


### (27/43) A Symbolic Character-Aware Model for Solving Geometry Problems (Maizhen Ning et al., 2023)

{{<citation>}}

Maizhen Ning, Qiu-Feng Wang, Kaizhu Huang, Xiaowei Huang. (2023)  
**A Symbolic Character-Aware Model for Solving Geometry Problems**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2308.02823v1)  

---


**ABSTRACT**  
AI has made significant progress in solving math problems, but geometry problems remain challenging due to their reliance on both text and diagrams. In the text description, symbolic characters such as "$\triangle$ABC" often serve as a bridge to connect the corresponding diagram. However, by simply tokenizing symbolic characters into individual letters (e.g., 'A', 'B' and 'C'), existing works fail to study them explicitly and thus lose the semantic relationship with the diagram. In this paper, we develop a symbolic character-aware model to fully explore the role of these characters in both text and diagram understanding and optimize the model under a multi-modal reasoning framework. In the text encoder, we propose merging individual symbolic characters to form one semantic unit along with geometric information from the corresponding diagram. For the diagram encoder, we pre-train it under a multi-label classification framework with the symbolic characters as labels. In addition, we enhance the geometry diagram understanding ability via a self-supervised learning method under the masked image modeling auxiliary task. By integrating the proposed model into a general encoder-decoder pipeline for solving geometry problems, we demonstrate its superiority on two benchmark datasets, including GeoQA and Geometry3K, with extensive experiments. Specifically, on GeoQA, the question-solving accuracy is increased from 60.0\% to 64.1\%, achieving a new state-of-the-art accuracy; on Geometry3K, we reduce the question average solving steps from 6.9 down to 6.0 with marginally higher solving accuracy.

{{</citation>}}


### (28/43) MiAMix: Enhancing Image Classification through a Multi-stage Augmented Mixied Sample Data Augmentation Method (Wen Liang et al., 2023)

{{<citation>}}

Wen Liang, Youzhi Liang, Jianguo Jia. (2023)  
**MiAMix: Enhancing Image Classification through a Multi-stage Augmented Mixied Sample Data Augmentation Method**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Augmentation, Image Classification  
[Paper Link](http://arxiv.org/abs/2308.02804v1)  

---


**ABSTRACT**  
Despite substantial progress in the field of deep learning, overfitting persists as a critical challenge, and data augmentation has emerged as a particularly promising approach due to its capacity to enhance model generalization in various computer vision tasks. While various strategies have been proposed, Mixed Sample Data Augmentation (MSDA) has shown great potential for enhancing model performance and generalization. We introduce a novel mixup method called MiAMix, which stands for Multi-stage Augmented Mixup. MiAMix integrates image augmentation into the mixup framework, utilizes multiple diversified mixing methods concurrently, and improves the mixing method by randomly selecting mixing mask augmentation methods. Recent methods utilize saliency information and the MiAMix is designed for computational efficiency as well, reducing additional overhead and offering easy integration into existing training pipelines. We comprehensively evaluate MiaMix using four image benchmarks and pitting it against current state-of-the-art mixed sample data augmentation techniques to demonstrate that MIAMix improves performance without heavy computational overhead.

{{</citation>}}


### (29/43) Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution (Yong Liu et al., 2023)

{{<citation>}}

Yong Liu, Hang Dong, Boyang Liang, Songwei Liu, Qingji Dong, Kai Chen, Fangmin Chen, Lean Fu, Fei Wang. (2023)  
**Unfolding Once is Enough: A Deployment-Friendly Transformer Unit for Super-Resolution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.02794v1)  

---


**ABSTRACT**  
Recent years have witnessed a few attempts of vision transformers for single image super-resolution (SISR). Since the high resolution of intermediate features in SISR models increases memory and computational requirements, efficient SISR transformers are more favored. Based on some popular transformer backbone, many methods have explored reasonable schemes to reduce the computational complexity of the self-attention module while achieving impressive performance. However, these methods only focus on the performance on the training platform (e.g., Pytorch/Tensorflow) without further optimization for the deployment platform (e.g., TensorRT). Therefore, they inevitably contain some redundant operators, posing challenges for subsequent deployment in real-world applications. In this paper, we propose a deployment-friendly transformer unit, namely UFONE (i.e., UnFolding ONce is Enough), to alleviate these problems. In each UFONE, we introduce an Inner-patch Transformer Layer (ITL) to efficiently reconstruct the local structural information from patches and a Spatial-Aware Layer (SAL) to exploit the long-range dependencies between patches. Based on UFONE, we propose a Deployment-friendly Inner-patch Transformer Network (DITN) for the SISR task, which can achieve favorable performance with low latency and memory usage on both training and deployment platforms. Furthermore, to further boost the deployment efficiency of the proposed DITN on TensorRT, we also provide an efficient substitution for layer normalization and propose a fusion optimization strategy for specific operators. Extensive experiments show that our models can achieve competitive results in terms of qualitative and quantitative performance with high deployment efficiency. Code is available at \url{https://github.com/yongliuy/DITN}.

{{</citation>}}


### (30/43) Few-shot Class-Incremental Semantic Segmentation via Pseudo-Labeling and Knowledge Distillation (Chengjia Jiang et al., 2023)

{{<citation>}}

Chengjia Jiang, Tao Wang, Sien Li, Jinyang Wang, Shirui Wang, Antonios Antoniou. (2023)  
**Few-shot Class-Incremental Semantic Segmentation via Pseudo-Labeling and Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.02790v1)  

---


**ABSTRACT**  
We address the problem of learning new classes for semantic segmentation models from few examples, which is challenging because of the following two reasons. Firstly, it is difficult to learn from limited novel data to capture the underlying class distribution. Secondly, it is challenging to retain knowledge for existing classes and to avoid catastrophic forgetting. For learning from limited data, we propose a pseudo-labeling strategy to augment the few-shot training annotations in order to learn novel classes more effectively. Given only one or a few images labeled with the novel classes and a much larger set of unlabeled images, we transfer the knowledge from labeled images to unlabeled images with a coarse-to-fine pseudo-labeling approach in two steps. Specifically, we first match each labeled image to its nearest neighbors in the unlabeled image set at the scene level, in order to obtain images with a similar scene layout. This is followed by obtaining pseudo-labels within this neighborhood by applying classifiers learned on the few-shot annotations. In addition, we use knowledge distillation on both labeled and unlabeled data to retain knowledge on existing classes. We integrate the above steps into a single convolutional neural network with a unified learning objective. Extensive experiments on the Cityscapes and KITTI datasets validate the efficacy of the proposed approach in the self-driving domain. Code is available from https://github.com/ChasonJiang/FSCILSS.

{{</citation>}}


### (31/43) Dual Degradation-Inspired Deep Unfolding Network for Low-Light Image Enhancement (Huake Wang et al., 2023)

{{<citation>}}

Huake Wang, Xingsong Hou, Xiaoyang Yan. (2023)  
**Dual Degradation-Inspired Deep Unfolding Network for Low-Light Image Enhancement**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.02776v1)  

---


**ABSTRACT**  
Although low-light image enhancement has achieved great stride based on deep enhancement models, most of them mainly stress on enhancement performance via an elaborated black-box network and rarely explore the physical significance of enhancement models. Towards this issue, we propose a Dual degrAdation-inSpired deep Unfolding network, termed DASUNet, for low-light image enhancement. Specifically, we construct a dual degradation model (DDM) to explicitly simulate the deterioration mechanism of low-light images. It learns two distinct image priors via considering degradation specificity between luminance and chrominance spaces. To make the proposed scheme tractable, we design an alternating optimization solution to solve the proposed DDM. Further, the designed solution is unfolded into a specified deep network, imitating the iteration updating rules, to form DASUNet. Local and long-range information are obtained by prior modeling module (PMM), inheriting the advantages of convolution and Transformer, to enhance the representation capability of dual degradation priors. Additionally, a space aggregation module (SAM) is presented to boost the interaction of two degradation models. Extensive experiments on multiple popular low-light image datasets validate the effectiveness of DASUNet compared to canonical state-of-the-art low-light image enhancement methods. Our source code and pretrained model will be publicly available.

{{</citation>}}


## cs.CL (3)



### (32/43) LaDA: Latent Dialogue Action For Zero-shot Cross-lingual Neural Network Language Modeling (Zhanyu Ma et al., 2023)

{{<citation>}}

Zhanyu Ma, Jian Ye, Shuang Cheng. (2023)  
**LaDA: Latent Dialogue Action For Zero-shot Cross-lingual Neural Network Language Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2308.02903v1)  

---


**ABSTRACT**  
Cross-lingual adaptation has proven effective in spoken language understanding (SLU) systems with limited resources. Existing methods are frequently unsatisfactory for intent detection and slot filling, particularly for distant languages that differ significantly from the source language in scripts, morphology, and syntax. Latent Dialogue Action (LaDA) layer is proposed to optimize decoding strategy in order to address the aforementioned issues. The model consists of an additional layer of latent dialogue action. It enables our model to improve a system's capability of handling conversations with complex multilingual intent and slot values of distant languages. To the best of our knowledge, this is the first exhaustive investigation of the use of latent variables for optimizing cross-lingual SLU policy during the decode stage. LaDA obtains state-of-the-art results on public datasets for both zero-shot and few-shot adaptation.

{{</citation>}}


### (33/43) ApproBiVT: Lead ASR Models to Generalize Better Using Approximated Bias-Variance Tradeoff Guided Early Stopping and Checkpoint Averaging (Fangyuan Wang et al., 2023)

{{<citation>}}

Fangyuan Wang, Ming Hao, Yuhai Shi, Bo Xu. (2023)  
**ApproBiVT: Lead ASR Models to Generalize Better Using Approximated Bias-Variance Tradeoff Guided Early Stopping and Checkpoint Averaging**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: AI, Bias, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2308.02870v1)  

---


**ABSTRACT**  
The conventional recipe for Automatic Speech Recognition (ASR) models is to 1) train multiple checkpoints on a training set while relying on a validation set to prevent overfitting using early stopping and 2) average several last checkpoints or that of the lowest validation losses to obtain the final model. In this paper, we rethink and update the early stopping and checkpoint averaging from the perspective of the bias-variance tradeoff. Theoretically, the bias and variance represent the fitness and variability of a model and the tradeoff of them determines the overall generalization error. But, it's impractical to evaluate them precisely. As an alternative, we take the training loss and validation loss as proxies of bias and variance and guide the early stopping and checkpoint averaging using their tradeoff, namely an Approximated Bias-Variance Tradeoff (ApproBiVT). When evaluating with advanced ASR models, our recipe provides 2.5%-3.7% and 3.1%-4.6% CER reduction on the AISHELL-1 and AISHELL-2, respectively.

{{</citation>}}


### (34/43) EduChat: A Large-Scale Language Model-based Chatbot System for Intelligent Education (Yuhao Dan et al., 2023)

{{<citation>}}

Yuhao Dan, Zhikai Lei, Yiyang Gu, Yong Li, Jianghao Yin, Jiaju Lin, Linhao Ye, Zhiyan Tie, Yougen Zhou, Yilei Wang, Aimin Zhou, Ze Zhou, Qin Chen, Jie Zhou, Liang He, Xipeng Qiu. (2023)  
**EduChat: A Large-Scale Language Model-based Chatbot System for Intelligent Education**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.02773v1)  

---


**ABSTRACT**  
EduChat (https://www.educhat.top/) is a large-scale language model (LLM)-based chatbot system in the education domain. Its goal is to support personalized, fair, and compassionate intelligent education, serving teachers, students, and parents. Guided by theories from psychology and education, it further strengthens educational functions such as open question answering, essay assessment, Socratic teaching, and emotional support based on the existing basic LLMs. Particularly, we learn domain-specific knowledge by pre-training on the educational corpus and stimulate various skills with tool use by fine-tuning on designed system prompts and instructions. Currently, EduChat is available online as an open-source project, with its code, data, and model parameters available on platforms (e.g., GitHub https://github.com/icalk-nlp/EduChat, Hugging Face https://huggingface.co/ecnu-icalk ). We also prepare a demonstration of its capabilities online (https://vimeo.com/851004454). This initiative aims to promote research and applications of LLMs for intelligent education.

{{</citation>}}


## eess.IV (2)



### (35/43) Generative Adversarial Networks for Stain Normalisation in Histopathology (Jack Breen et al., 2023)

{{<citation>}}

Jack Breen, Kieran Zucker, Katie Allen, Nishant Ravikumar, Nicolas M. Orsi. (2023)  
**Generative Adversarial Networks for Stain Normalisation in Histopathology**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.02851v1)  

---


**ABSTRACT**  
The rapid growth of digital pathology in recent years has provided an ideal opportunity for the development of artificial intelligence-based tools to improve the accuracy and efficiency of clinical diagnoses. One of the significant roadblocks to current research is the high level of visual variability across digital pathology images, causing models to generalise poorly to unseen data. Stain normalisation aims to standardise the visual profile of digital pathology images without changing the structural content of the images. In this chapter, we explore different techniques which have been used for stain normalisation in digital pathology, with a focus on approaches which utilise generative adversarial networks (GANs). Typically, GAN-based methods outperform non-generative approaches but at the cost of much greater computational requirements. However, it is not clear which method is best for stain normalisation in general, with different GAN and non-GAN approaches outperforming each other in different scenarios and according to different performance metrics. This is an ongoing field of study as researchers aim to identify a method which efficiently and effectively normalises pathology images to make AI models more robust and generalisable.

{{</citation>}}


### (36/43) Landmark Detection using Transformer Toward Robot-assisted Nasal Airway Intubation (Tianhang Liu et al., 2023)

{{<citation>}}

Tianhang Liu, Hechen Li, Long Bai, Yanan Wu, An Wang, Mobarakol Islam, Hongliang Ren. (2023)  
**Landmark Detection using Transformer Toward Robot-assisted Nasal Airway Intubation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-RO, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.02845v1)  

---


**ABSTRACT**  
Robot-assisted airway intubation application needs high accuracy in locating targets and organs. Two vital landmarks, nostrils and glottis, can be detected during the intubation to accommodate the stages of nasal intubation. Automated landmark detection can provide accurate localization and quantitative evaluation. The Detection Transformer (DeTR) leads object detectors to a new paradigm with long-range dependence. However, current DeTR requires long iterations to converge, and does not perform well in detecting small objects. This paper proposes a transformer-based landmark detection solution with deformable DeTR and the semantic-aligned-matching module for detecting landmarks in robot-assisted intubation. The semantics aligner can effectively align the semantics of object queries and image features in the same embedding space using the most discriminative features. To evaluate the performance of our solution, we utilize a publicly accessible glottis dataset and automatically annotate a nostril detection dataset. The experimental results demonstrate our competitive performance in detection accuracy. Our code is publicly accessible.

{{</citation>}}


## q-fin.PM (1)



### (37/43) Reinforcement Learning for Financial Index Tracking (Xianhua Peng et al., 2023)

{{<citation>}}

Xianhua Peng, Chenyin Gong, Xue Dong He. (2023)  
**Reinforcement Learning for Financial Index Tracking**  

---
Primary Category: q-fin.PM  
Categories: cs-LG, q-fin-PM, q-fin.PM  
Keywords: Financial, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02820v1)  

---


**ABSTRACT**  
We propose the first discrete-time infinite-horizon dynamic formulation of the financial index tracking problem under both return-based tracking error and value-based tracking error. The formulation overcomes the limitations of existing models by incorporating the intertemporal dynamics of market information variables not limited to prices, allowing exact calculation of transaction costs, accounting for the tradeoff between overall tracking error and transaction costs, allowing effective use of data in a long time period, etc. The formulation also allows novel decision variables of cash injection or withdraw. We propose to solve the portfolio rebalancing equation using a Banach fixed point iteration, which allows to accurately calculate the transaction costs specified as nonlinear functions of trading volumes in practice. We propose an extension of deep reinforcement learning (RL) method to solve the dynamic formulation. Our RL method resolves the issue of data limitation resulting from the availability of a single sample path of financial data by a novel training scheme. A comprehensive empirical study based on a 17-year-long testing set demonstrates that the proposed method outperforms a benchmark method in terms of tracking accuracy and has the potential for earning extra profit through cash withdraw strategy.

{{</citation>}}


## cs.MM (1)



### (38/43) PromptCARE: Prompt Copyright Protection by Watermark Injection and Verification (Hongwei Yao et al., 2023)

{{<citation>}}

Hongwei Yao, Jian Lou, Kui Ren, Zhan Qin. (2023)  
**PromptCARE: Prompt Copyright Protection by Watermark Injection and Verification**  

---
Primary Category: cs.MM  
Categories: cs-CR, cs-MM, cs.MM  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2308.02816v1)  

---


**ABSTRACT**  
Large language models (LLMs) have witnessed a meteoric rise in popularity among the general public users over the past few months, facilitating diverse downstream tasks with human-level accuracy and proficiency. Prompts play an essential role in this success, which efficiently adapt pre-trained LLMs to task-specific applications by simply prepending a sequence of tokens to the query texts. However, designing and selecting an optimal prompt can be both expensive and demanding, leading to the emergence of Prompt-as-a-Service providers who profit by providing well-designed prompts for authorized use. With the growing popularity of prompts and their indispensable role in LLM-based services, there is an urgent need to protect the copyright of prompts against unauthorized use.   In this paper, we propose PromptCARE, the first framework for prompt copyright protection through watermark injection and verification. Prompt watermarking presents unique challenges that render existing watermarking techniques developed for model and dataset copyright verification ineffective. PromptCARE overcomes these hurdles by proposing watermark injection and verification schemes tailor-made for prompts and NLP characteristics. Extensive experiments on six well-known benchmark datasets, using three prevalent pre-trained LLMs (BERT, RoBERTa, and Facebook OPT-1.3b), demonstrate the effectiveness, harmlessness, robustness, and stealthiness of PromptCARE.

{{</citation>}}


## cs.CR (2)



### (39/43) Meta-Analysis and Systematic Review for Anomaly Network Intrusion Detection Systems: Detection Methods, Dataset, Validation Methodology, and Challenges (Ziadoon K. Maseer et al., 2023)

{{<citation>}}

Ziadoon K. Maseer, Robiah Yusof, Baidaa Al-Bander, Abdu Saif, Qusay Kanaan Kadhim. (2023)  
**Meta-Analysis and Systematic Review for Anomaly Network Intrusion Detection Systems: Detection Methods, Dataset, Validation Methodology, and Challenges**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SY, cs.CR, eess-SY  
Keywords: AI, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2308.02805v1)  

---


**ABSTRACT**  
Intrusion detection systems (IDSs) built on artificial intelligence (AI) are presented as latent mechanisms for actively detecting fresh attacks over a complex network. Although review papers are used the systematic review or simple methods to analyse and criticize the anomaly NIDS works, the current review uses a traditional way as a quantitative description to find current gaps by synthesizing and summarizing the data comparison without considering algorithms performance. This paper presents a systematic and meta-analysis study of AI for network intrusion detection systems (NIDS) focusing on deep learning (DL) and machine learning (ML) approaches in network security. Deep learning algorithms are explained in their structure, and data intrusion network is justified based on an infrastructure of networks and attack types. By conducting a meta-analysis and debating the validation of the DL and ML approach by effectiveness, used dataset, detected attacks, classification task, and time complexity, we offer a thorough benchmarking assessment of the current NIDS-based publications-based systematic approach. The proposed method is considered reviewing works for the anomaly-based network intrusion detection system (anomaly-NIDS) models. Furthermore, the effectiveness of proposed algorithms and selected datasets are discussed for the recent direction and improvements of ML and DL to the NIDS. The future trends for improving an anomaly-IDS for continuing detection in the evolution of cyberattacks are highlighted in several research studies.

{{</citation>}}


### (40/43) DiSPEL: Distributed Security Policy Enforcement for Bus-based SoC (Sudipta Paria et al., 2023)

{{<citation>}}

Sudipta Paria, Swarup Bhunia. (2023)  
**DiSPEL: Distributed Security Policy Enforcement for Bus-based SoC**  

---
Primary Category: cs.CR  
Categories: cs-AR, cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2308.02792v1)  

---


**ABSTRACT**  
The current zero trust model adopted in System-on-Chip (SoC) design is vulnerable to various malicious entities, and modern SoC designs must incorporate various security policies to protect sensitive assets from unauthorized access. These policies involve complex interactions between multiple IP blocks, which poses challenges for SoC designers and security experts when implementing these policies and for system validators when ensuring compliance. Difficulties arise when upgrading policies, reusing IPs for systems targeting different security requirements, and the subsequent increase in design time and time-to-market. This paper proposes a generic and flexible framework, called DiSPEL, for enforcing security policies defined by the user represented in a formal way for any bus-based SoC design. It employs a distributed deployment strategy while ensuring trusted bus operations despite the presence of untrusted IPs. It relies on incorporating a dedicated, centralized module capable of implementing diverse security policies involving bus-level interactions while generating the necessary logic and appending in the bus-level wrapper for IP-level policies. The proposed architecture is generic and independent of specific security policy types supporting both synthesizable and non-synthesizable solutions. The experimental results demonstrate its effectiveness and correctness in enforcing the security requirements and viability due to low overhead in terms of area, delay, and power consumption tested on open-source standard SoC benchmarks.

{{</citation>}}


## cs.SI (1)



### (41/43) Crowdsourcing Fraud Detection over Heterogeneous Temporal MMMA Graph (Zequan Xu et al., 2023)

{{<citation>}}

Zequan Xu, Qihang Sun, Shaofeng Hu, Jieming Shi, Hui Li. (2023)  
**Crowdsourcing Fraud Detection over Heterogeneous Temporal MMMA Graph**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: Fraud Detection  
[Paper Link](http://arxiv.org/abs/2308.02793v1)  

---


**ABSTRACT**  
The rise of the click farm business using Multi-purpose Messaging Mobile Apps (MMMAs) tempts cybercriminals to perpetrate crowdsourcing frauds that cause financial losses to click farm workers. In this paper, we propose a novel contrastive multi-view learning method named CMT for crowdsourcing fraud detection over the heterogeneous temporal graph (HTG) of MMMA. CMT captures both heterogeneity and dynamics of HTG and generates high-quality representations for crowdsourcing fraud detection in a self-supervised manner. We deploy CMT to detect crowdsourcing frauds on an industry-size HTG of a representative MMMA WeChat and it significantly outperforms other methods. CMT also shows promising results for fraud detection on a large-scale public financial HTG, indicating that it can be applied in other graph anomaly detection tasks.

{{</citation>}}


## cs.HC (1)



### (42/43) SUMMIT: Scaffolding OSS Issue Discussion Through Summarization (Saskia Gilmer et al., 2023)

{{<citation>}}

Saskia Gilmer, Avinash Bhat, Shuvam Shah, Kevin Cherry, Jinghui Cheng, Jin L. C. Guo. (2023)  
**SUMMIT: Scaffolding OSS Issue Discussion Through Summarization**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2308.02780v1)  

---


**ABSTRACT**  
For Open Source Software (OSS) projects, discussions in Issue Tracking Systems (ITS) serve as a crucial collaboration mechanism for diverse stakeholders. However, these discussions can become lengthy and entangled, making it hard to find relevant information and make further contributions. In this work, we study the use of summarization to aid users in collaboratively making sense of OSS issue discussion threads. We reveal a complex picture of how summarization is used by issue users in practice as a strategy to help develop and manage their discussions. Grounded on the different objectives served by the summaries and the outcome of our formative study with OSS stakeholders, we identified a set of guidelines to inform the design of collaborative summarization tools for OSS issue discussions. We then developed SUMMIT, a tool that allows issue users to collectively construct summaries of different types of information discussed, as well as a set of comments representing continuous conversations within the thread. To alleviate the manual effort involved, SUMMIT uses techniques that automatically detect information types and summarize texts to facilitate the generation of these summaries. A lab user study indicates that, as the users of SUMMIT, OSS stakeholders adopted different strategies to acquire information on issue threads. Furthermore, different features of SUMMIT effectively lowered the perceived difficulty of locating information from issue threads and enabled the users to prioritize their effort. Overall, our findings demonstrated the potential of SUMMIT, and the corresponding design guidelines, in supporting users to acquire information from lengthy discussions in ITSs. Our work sheds light on key design considerations and features when exploring crowd-based and machine-learning-enabled instruments for asynchronous collaboration on complex tasks such as OSS development.

{{</citation>}}


## eess.SY (1)



### (43/43) Surrogate Empowered Sim2Real Transfer of Deep Reinforcement Learning for ORC Superheat Control (Runze Lin et al., 2023)

{{<citation>}}

Runze Lin, Yangyang Luo, Xialai Wu, Junghui Chen, Biao Huang, Lei Xie, Hongye Su. (2023)  
**Surrogate Empowered Sim2Real Transfer of Deep Reinforcement Learning for ORC Superheat Control**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.02765v1)  

---


**ABSTRACT**  
The Organic Rankine Cycle (ORC) is widely used in industrial waste heat recovery due to its simple structure and easy maintenance. However, in the context of smart manufacturing in the process industry, traditional model-based optimization control methods are unable to adapt to the varying operating conditions of the ORC system or sudden changes in operating modes. Deep reinforcement learning (DRL) has significant advantages in situations with uncertainty as it directly achieves control objectives by interacting with the environment without requiring an explicit model of the controlled plant. Nevertheless, direct application of DRL to physical ORC systems presents unacceptable safety risks, and its generalization performance under model-plant mismatch is insufficient to support ORC control requirements. Therefore, this paper proposes a Sim2Real transfer learning-based DRL control method for ORC superheat control, which aims to provide a new simple, feasible, and user-friendly solution for energy system optimization control. Experimental results show that the proposed method greatly improves the training speed of DRL in ORC control problems and solves the generalization performance issue of the agent under multiple operating conditions through Sim2Real transfer.

{{</citation>}}
