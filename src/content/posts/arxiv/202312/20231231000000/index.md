---
draft: false
title: "arXiv @ 2023.12.31"
date: 2023-12-31
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.31"
    identifier: arxiv_20231231
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (10)](#cslg-10)
- [cs.CR (4)](#cscr-4)
- [cs.CY (1)](#cscy-1)
- [cs.IR (4)](#csir-4)
- [eess.SY (1)](#eesssy-1)
- [cs.CL (13)](#cscl-13)
- [cs.CV (10)](#cscv-10)
- [cs.RO (5)](#csro-5)
- [cs.AI (4)](#csai-4)
- [eess.IV (1)](#eessiv-1)
- [cs.LO (1)](#cslo-1)
- [cs.AR (1)](#csar-1)
- [eess.AS (1)](#eessas-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.SE (1)](#csse-1)
- [cs.DB (1)](#csdb-1)
- [cs.ET (1)](#cset-1)

## cs.LG (10)



### (1/60) Fairness-Enhancing Vehicle Rebalancing in the Ride-hailing System (Xiaotong Guo et al., 2023)

{{<citation>}}

Xiaotong Guo, Hanyong Xu, Dingyi Zhuang, Yunhan Zheng, Jinhua Zhao. (2023)  
**Fairness-Enhancing Vehicle Rebalancing in the Ride-hailing System**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2401.00093v1)  

---


**ABSTRACT**  
The rapid growth of the ride-hailing industry has revolutionized urban transportation worldwide. Despite its benefits, equity concerns arise as underserved communities face limited accessibility to affordable ride-hailing services. A key issue in this context is the vehicle rebalancing problem, where idle vehicles are moved to areas with anticipated demand. Without equitable approaches in demand forecasting and rebalancing strategies, these practices can further deepen existing inequities. In the realm of ride-hailing, three main facets of fairness are recognized: algorithmic fairness, fairness to drivers, and fairness to riders. This paper focuses on enhancing both algorithmic and rider fairness through a novel vehicle rebalancing method. We introduce an approach that combines a Socio-Aware Spatial-Temporal Graph Convolutional Network (SA-STGCN) for refined demand prediction and a fairness-integrated Matching-Integrated Vehicle Rebalancing (MIVR) model for subsequent vehicle rebalancing. Our methodology is designed to reduce prediction discrepancies and ensure equitable service provision across diverse regions. The effectiveness of our system is evaluated using simulations based on real-world ride-hailing data. The results suggest that our proposed method enhances both accuracy and fairness in forecasting ride-hailing demand, ultimately resulting in more equitable vehicle rebalancing in subsequent operations. Specifically, the algorithm developed in this study effectively reduces the standard deviation and average customer wait times by 6.48% and 0.49%, respectively. This achievement signifies a beneficial outcome for ride-hailing platforms, striking a balance between operational efficiency and fairness.

{{</citation>}}


### (2/60) Data Augmentation for Supervised Graph Outlier Detection with Latent Diffusion Models (Kay Liu et al., 2023)

{{<citation>}}

Kay Liu, Hengrui Zhang, Ziqing Hu, Fangxin Wang, Philip S. Yu. (2023)  
**Data Augmentation for Supervised Graph Outlier Detection with Latent Diffusion Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.17679v1)  

---


**ABSTRACT**  
Graph outlier detection is a prominent task of research and application in the realm of graph neural networks. It identifies the outlier nodes that exhibit deviation from the majority in the graph. One of the fundamental challenges confronting supervised graph outlier detection algorithms is the prevalent issue of class imbalance, where the scarcity of outlier instances compared to normal instances often results in suboptimal performance. Conventional methods mitigate the imbalance by reweighting instances in the estimation of the loss function, assigning higher weights to outliers and lower weights to inliers. Nonetheless, these strategies are prone to overfitting and underfitting, respectively. Recently, generative models, especially diffusion models, have demonstrated their efficacy in synthesizing high-fidelity images. Despite their extraordinary generation quality, their potential in data augmentation for supervised graph outlier detection remains largely underexplored.   To bridge this gap, we introduce GODM, a novel data augmentation for mitigating class imbalance in supervised Graph Outlier detection with latent Diffusion Models. Specifically, our proposed method consists of three key components: (1) Variantioanl Encoder maps the heterogeneous information inherent within the graph data into a unified latent space. (2) Graph Generator synthesizes graph data that are statistically similar to real outliers from latent space, and (3) Latent Diffusion Model learns the latent space distribution of real organic data by iterative denoising. Extensive experiments conducted on multiple datasets substantiate the effectiveness and efficiency of GODM. The case study further demonstrated the generation quality of our synthetic data. To foster accessibility and reproducibility, we encapsulate GODM into a plug-and-play package and release it at the Python Package Index (PyPI).

{{</citation>}}


### (3/60) AIJack: Security and Privacy Risk Simulator for Machine Learning (Hideaki Takahashi, 2023)

{{<citation>}}

Hideaki Takahashi. (2023)  
**AIJack: Security and Privacy Risk Simulator for Machine Learning**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2312.17667v1)  

---


**ABSTRACT**  
This paper introduces AIJack, an open-source library designed to assess security and privacy risks associated with the training and deployment of machine learning models. Amid the growing interest in big data and AI, advancements in machine learning research and business are accelerating. However, recent studies reveal potential threats, such as the theft of training data and the manipulation of models by malicious attackers. Therefore, a comprehensive understanding of machine learning's security and privacy vulnerabilities is crucial for the safe integration of machine learning into real-world products. AIJack aims to address this need by providing a library with various attack and defense methods through a unified API. The library is publicly available on GitHub (https://github.com/Koukyosyumei/AIJack).

{{</citation>}}


### (4/60) XAI for In-hospital Mortality Prediction via Multimodal ICU Data (Xingqiao Li et al., 2023)

{{<citation>}}

Xingqiao Li, Jindong Gu, Zhiyong Wang, Yancheng Yuan, Bo Du, Fengxiang He. (2023)  
**XAI for In-hospital Mortality Prediction via Multimodal ICU Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.17624v1)  

---


**ABSTRACT**  
Predicting in-hospital mortality for intensive care unit (ICU) patients is key to final clinical outcomes. AI has shown advantaged accuracy but suffers from the lack of explainability. To address this issue, this paper proposes an eXplainable Multimodal Mortality Predictor (X-MMP) approaching an efficient, explainable AI solution for predicting in-hospital mortality via multimodal ICU data. We employ multimodal learning in our framework, which can receive heterogeneous inputs from clinical data and make decisions. Furthermore, we introduce an explainable method, namely Layer-Wise Propagation to Transformer, as a proper extension of the LRP method to Transformers, producing explanations over multimodal inputs and revealing the salient features attributed to prediction. Moreover, the contribution of each modality to clinical outcomes can be visualized, assisting clinicians in understanding the reasoning behind decision-making. We construct a multimodal dataset based on MIMIC-III and MIMIC-III Waveform Database Matched Subset. Comprehensive experiments on benchmark datasets demonstrate that our proposed framework can achieve reasonable interpretation with competitive prediction accuracy. In particular, our framework can be easily transferred to other clinical tasks, which facilitates the discovery of crucial factors in healthcare research.

{{</citation>}}


### (5/60) Interpretable and Explainable Machine Learning Methods for Predictive Process Monitoring: A Systematic Literature Review (Nijat Mehdiyev et al., 2023)

{{<citation>}}

Nijat Mehdiyev, Maxim Majlatow, Peter Fettke. (2023)  
**Interpretable and Explainable Machine Learning Methods for Predictive Process Monitoring: A Systematic Literature Review**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.17584v1)  

---


**ABSTRACT**  
This paper presents a systematic literature review (SLR) on the explainability and interpretability of machine learning (ML) models within the context of predictive process mining, using the PRISMA framework. Given the rapid advancement of artificial intelligence (AI) and ML systems, understanding the "black-box" nature of these technologies has become increasingly critical. Focusing specifically on the domain of process mining, this paper delves into the challenges of interpreting ML models trained with complex business process data. We differentiate between intrinsically interpretable models and those that require post-hoc explanation techniques, providing a comprehensive overview of the current methodologies and their applications across various application domains. Through a rigorous bibliographic analysis, this research offers a detailed synthesis of the state of explainability and interpretability in predictive process mining, identifying key trends, challenges, and future directions. Our findings aim to equip researchers and practitioners with a deeper understanding of how to develop and implement more trustworthy, transparent, and effective intelligent systems for predictive process analytics.

{{</citation>}}


### (6/60) Embedded feature selection in LSTM networks with multi-objective evolutionary ensemble learning for time series forecasting (Raquel Espinosa et al., 2023)

{{<citation>}}

Raquel Espinosa, Fernando Jiménez, José Palma. (2023)  
**Embedded feature selection in LSTM networks with multi-objective evolutionary ensemble learning for time series forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.17517v1)  

---


**ABSTRACT**  
Time series forecasting plays a crucial role in diverse fields, necessitating the development of robust models that can effectively handle complex temporal patterns. In this article, we present a novel feature selection method embedded in Long Short-Term Memory networks, leveraging a multi-objective evolutionary algorithm. Our approach optimizes the weights and biases of the LSTM in a partitioned manner, with each objective function of the evolutionary algorithm targeting the root mean square error in a specific data partition. The set of non-dominated forecast models identified by the algorithm is then utilized to construct a meta-model through stacking-based ensemble learning. Furthermore, our proposed method provides an avenue for attribute importance determination, as the frequency of selection for each attribute in the set of non-dominated forecasting models reflects their significance. This attribute importance insight adds an interpretable dimension to the forecasting process. Experimental evaluations on air quality time series data from Italy and southeast Spain demonstrate that our method substantially improves the generalization ability of conventional LSTMs, effectively reducing overfitting. Comparative analyses against state-of-the-art CancelOut and EAR-FS methods highlight the superior performance of our approach.

{{</citation>}}


### (7/60) HiBid: A Cross-Channel Constrained Bidding System with Budget Allocation by Hierarchical Offline Deep Reinforcement Learning (Hao Wang et al., 2023)

{{<citation>}}

Hao Wang, Bo Tang, Chi Harold Liu, Shangqin Mao, Jiahong Zhou, Zipeng Dai, Yaqi Sun, Qianlong Xie, Xingxing Wang, Dong Wang. (2023)  
**HiBid: A Cross-Channel Constrained Bidding System with Budget Allocation by Hierarchical Offline Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-GT, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.17503v1)  

---


**ABSTRACT**  
Online display advertising platforms service numerous advertisers by providing real-time bidding (RTB) for the scale of billions of ad requests every day. The bidding strategy handles ad requests cross multiple channels to maximize the number of clicks under the set financial constraints, i.e., total budget and cost-per-click (CPC), etc. Different from existing works mainly focusing on single channel bidding, we explicitly consider cross-channel constrained bidding with budget allocation. Specifically, we propose a hierarchical offline deep reinforcement learning (DRL) framework called ``HiBid'', consisted of a high-level planner equipped with auxiliary loss for non-competitive budget allocation, and a data augmentation enhanced low-level executor for adaptive bidding strategy in response to allocated budgets. Additionally, a CPC-guided action selection mechanism is introduced to satisfy the cross-channel CPC constraint. Through extensive experiments on both the large-scale log data and online A/B testing, we confirm that HiBid outperforms six baselines in terms of the number of clicks, CPC satisfactory ratio, and return-on-investment (ROI). We also deploy HiBid on Meituan advertising platform to already service tens of thousands of advertisers every day.

{{</citation>}}


### (8/60) Integrating Chemical Language and Molecular Graph in Multimodal Fused Deep Learning for Drug Property Prediction (Xiaohua Lu et al., 2023)

{{<citation>}}

Xiaohua Lu, Liangxu Xie, Lei Xu, Rongzhi Mao, Shan Chang, Xiaojun Xu. (2023)  
**Integrating Chemical Language and Molecular Graph in Multimodal Fused Deep Learning for Drug Property Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-bio-ph, q-bio-BM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.17495v1)  

---


**ABSTRACT**  
Accurately predicting molecular properties is a challenging but essential task in drug discovery. Recently, many mono-modal deep learning methods have been successfully applied to molecular property prediction. However, the inherent limitation of mono-modal learning arises from relying solely on one modality of molecular representation, which restricts a comprehensive understanding of drug molecules and hampers their resilience against data noise. To overcome the limitations, we construct multimodal deep learning models to cover different molecular representations. We convert drug molecules into three molecular representations, SMILES-encoded vectors, ECFP fingerprints, and molecular graphs. To process the modal information, Transformer-Encoder, bi-directional gated recurrent units (BiGRU), and graph convolutional network (GCN) are utilized for feature learning respectively, which can enhance the model capability to acquire complementary and naturally occurring bioinformatics information. We evaluated our triple-modal model on six molecule datasets. Different from bi-modal learning models, we adopt five fusion methods to capture the specific features and leverage the contribution of each modal information better. Compared with mono-modal models, our multimodal fused deep learning (MMFDL) models outperform single models in accuracy, reliability, and resistance capability against noise. Moreover, we demonstrate its generalization ability in the prediction of binding constants for protein-ligand complex molecules in the refined set of PDBbind. The advantage of the multimodal model lies in its ability to process diverse sources of data using proper models and suitable fusion methods, which would enhance the noise resistance of the model while obtaining data diversity.

{{</citation>}}


### (9/60) Differentially Private Low-Rank Adaptation of Large Language Model Using Federated Learning (Xiao-Yang Liu et al., 2023)

{{<citation>}}

Xiao-Yang Liu, Rongyi Zhu, Daochen Zha, Jiechao Gao, Shan Zhong, Meikang Qiu. (2023)  
**Differentially Private Low-Rank Adaptation of Large Language Model Using Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.17493v1)  

---


**ABSTRACT**  
The surge in interest and application of large language models (LLMs) has sparked a drive to fine-tune these models to suit specific applications, such as finance and medical science. However, concerns regarding data privacy have emerged, especially when multiple stakeholders aim to collaboratively enhance LLMs using sensitive data. In this scenario, federated learning becomes a natural choice, allowing decentralized fine-tuning without exposing raw data to central servers. Motivated by this, we investigate how data privacy can be ensured in LLM fine-tuning through practical federated learning approaches, enabling secure contributions from multiple parties to enhance LLMs. Yet, challenges arise: 1) despite avoiding raw data exposure, there is a risk of inferring sensitive information from model outputs, and 2) federated learning for LLMs incurs notable communication overhead. To address these challenges, this article introduces DP-LoRA, a novel federated learning algorithm tailored for LLMs. DP-LoRA preserves data privacy by employing a Gaussian mechanism that adds noise in weight updates, maintaining individual data privacy while facilitating collaborative model training. Moreover, DP-LoRA optimizes communication efficiency via low-rank adaptation, minimizing the transmission of updated weights during distributed training. The experimental results across medical, financial, and general datasets using various LLMs demonstrate that DP-LoRA effectively ensures strict privacy constraints while minimizing communication overhead.

{{</citation>}}


### (10/60) ClST: A Convolutional Transformer Framework for Automatic Modulation Recognition by Knowledge Distillation (Dongbin Hou et al., 2023)

{{<citation>}}

Dongbin Hou, Lixin Li, Wensheng Lin, Junli Liang, Zhu Han. (2023)  
**ClST: A Convolutional Transformer Framework for Automatic Modulation Recognition by Knowledge Distillation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP  
Keywords: Knowledge Distillation, Transformer  
[Paper Link](http://arxiv.org/abs/2312.17446v1)  

---


**ABSTRACT**  
With the rapid development of deep learning (DL) in recent years, automatic modulation recognition (AMR) with DL has achieved high accuracy. However, insufficient training signal data in complicated channel environments and large-scale DL models are critical factors that make DL methods difficult to deploy in practice. Aiming to these problems, we propose a novel neural network named convolution-linked signal transformer (ClST) and a novel knowledge distillation method named signal knowledge distillation (SKD). The ClST is accomplished through three primary modifications: a hierarchy of transformer containing convolution, a novel attention mechanism named parallel spatial-channel attention (PSCA) mechanism and a novel convolutional transformer block named convolution-transformer projection (CTP) to leverage a convolutional projection. The SKD is a knowledge distillation method to effectively reduce the parameters and complexity of neural networks. We train two lightweight neural networks using the SKD algorithm, KD-CNN and KD-MobileNet, to meet the demand that neural networks can be used on miniaturized devices. The simulation results demonstrate that the ClST outperforms advanced neural networks on all datasets. Moreover, both KD-CNN and KD-MobileNet obtain higher recognition accuracy with less network complexity, which is very beneficial for the deployment of AMR on miniaturized communication devices.

{{</citation>}}


## cs.CR (4)



### (11/60) Quantifying Policy Administration Cost in an Active Learning Framework (Si Zhang et al., 2023)

{{<citation>}}

Si Zhang, Philip W. L. Fong. (2023)  
**Quantifying Policy Administration Cost in an Active Learning Framework**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2401.00086v1)  

---


**ABSTRACT**  
This paper proposes a computational model for policy administration. As an organization evolves, new users and resources are gradually placed under the mediation of the access control model. Each time such new entities are added, the policy administrator must deliberate on how the access control policy shall be revised to reflect the new reality. A well-designed access control model must anticipate such changes so that the administration cost does not become prohibitive when the organization scales up. Unfortunately, past Access Control research does not offer a formal way to quantify the cost of policy administration. In this work, we propose to model ongoing policy administration in an active learning framework. Administration cost can be quantified in terms of query complexity. We demonstrate the utility of this approach by applying it to the evolution of protection domains. We also modelled different policy administration strategies in our framework. This allowed us to formally demonstrate that domain-based policies have a cost advantage over access control matrices because of the use of heuristic reasoning when the policy evolves. To the best of our knowledge, this is the first work to employ an active learning framework to study the cost of policy deliberation and demonstrate the cost advantage of heuristic policy administration.

{{</citation>}}


### (12/60) Comparing Effectiveness and Efficiency of Interactive Application Security Testing (IAST) and Runtime Application Self-Protection (RASP) Tools in a Large Java-based System (Aishwarya Seth et al., 2023)

{{<citation>}}

Aishwarya Seth, Saikath Bhattacharya, Sarah Elder, Nusrat Zahan, Laurie Williams. (2023)  
**Comparing Effectiveness and Efficiency of Interactive Application Security Testing (IAST) and Runtime Application Self-Protection (RASP) Tools in a Large Java-based System**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.17726v1)  

---


**ABSTRACT**  
Security resources are scarce, and practitioners need guidance in the effective and efficient usage of techniques and tools available in the cybersecurity industry. Two emerging tool types, Interactive Application Security Testing (IAST) and Runtime Application Self-Protection (RASP), have not been thoroughly evaluated against well-established counterparts such as Dynamic Application Security Testing (DAST) and Static Application Security Testing (SAST). The goal of this research is to aid practitioners in making informed choices about the use of Interactive Application Security Testing (IAST) and Runtime Application Self-Protection (RASP) tools through an analysis of their effectiveness and efficiency in comparison with different vulnerability detection and prevention techniques and tools. We apply IAST and RASP on OpenMRS, an open-source Java-based online application. We compare the efficiency and effectiveness of IAST and RASP with techniques applied on OpenMRS in prior work. We measure efficiency and effectiveness in terms of the number and type of vulnerabilities detected and prevented per hour. Our study shows IAST performed relatively well compared to other techniques, performing second-best in both efficiency and effectiveness. IAST detected eight Top-10 OWASP security risks compared to nine by SMPT and seven for EMPT, DAST, and SAST. IAST found more vulnerabilities than SMPT. The efficiency of IAST (2.14 VpH) is second to only EMPT (2.22 VpH). These findings imply that our study benefited from using IAST when conducting black-box security testing. In the context of a large, enterprise-scale web application such as OpenMRS, RASP does not replace vulnerability detection, while IAST is a powerful tool that complements other techniques.

{{</citation>}}


### (13/60) Malware Detection in IOT Systems Using Machine Learning Techniques (Ali Mehrban et al., 2023)

{{<citation>}}

Ali Mehrban, Pegah Ahadian. (2023)  
**Malware Detection in IOT Systems Using Machine Learning Techniques**  

---
Primary Category: cs.CR  
Categories: airccse-org/, cs-CR, cs-LG, cs-NI, cs.CR  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.17683v1)  

---


**ABSTRACT**  
Malware detection in IoT environments necessitates robust methodologies. This study introduces a CNN-LSTM hybrid model for IoT malware identification and evaluates its performance against established methods. Leveraging K-fold cross-validation, the proposed approach achieved 95.5% accuracy, surpassing existing methods. The CNN algorithm enabled superior learning model construction, and the LSTM classifier exhibited heightened accuracy in classification. Comparative analysis against prevalent techniques demonstrated the efficacy of the proposed model, highlighting its potential for enhancing IoT security. The study advocates for future exploration of SVMs as alternatives, emphasizes the need for distributed detection strategies, and underscores the importance of predictive analyses for a more powerful IOT security. This research serves as a platform for developing more resilient security measures in IoT ecosystems.

{{</citation>}}


### (14/60) Jatmo: Prompt Injection Defense by Task-Specific Finetuning (Julien Piet et al., 2023)

{{<citation>}}

Julien Piet, Maha Alrashed, Chawin Sitawarin, Sizhe Chen, Zeming Wei, Elizabeth Sun, Basel Alomair, David Wagner. (2023)  
**Jatmo: Prompt Injection Defense by Task-Specific Finetuning**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2312.17673v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are attracting significant research attention due to their instruction-following abilities, allowing users and developers to leverage LLMs for a variety of tasks. However, LLMs are vulnerable to prompt-injection attacks: a class of attacks that hijack the model's instruction-following abilities, changing responses to prompts to undesired, possibly malicious ones. In this work, we introduce Jatmo, a method for generating task-specific models resilient to prompt-injection attacks. Jatmo leverages the fact that LLMs can only follow instructions once they have undergone instruction tuning. It harnesses a teacher instruction-tuned model to generate a task-specific dataset, which is then used to fine-tune a base model (i.e., a non-instruction-tuned model). Jatmo only needs a task prompt and a dataset of inputs for the task: it uses the teacher model to generate outputs. For situations with no pre-existing datasets, Jatmo can use a single example, or in some cases none at all, to produce a fully synthetic dataset. Our experiments on six tasks show that Jatmo models provide the same quality of outputs on their specific task as standard LLMs, while being resilient to prompt injections. The best attacks succeeded in less than 0.5% of cases against our models, versus over 90% success rate against GPT-3.5-Turbo. We release Jatmo at https://github.com/wagner-group/prompt-injection-defense.

{{</citation>}}


## cs.CY (1)



### (15/60) ChatEd: A Chatbot Leveraging ChatGPT for an Enhanced Learning Experience in Higher Education (Kevin Wang et al., 2023)

{{<citation>}}

Kevin Wang, Jason Ramos, Ramon Lawrence. (2023)  
**ChatEd: A Chatbot Leveraging ChatGPT for an Enhanced Learning Experience in Higher Education**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs.CY  
Keywords: ChatGPT, GPT, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.00052v1)  

---


**ABSTRACT**  
With the rapid evolution of Natural Language Processing (NLP), Large Language Models (LLMs) like ChatGPT have emerged as powerful tools capable of transforming various sectors. Their vast knowledge base and dynamic interaction capabilities represent significant potential in improving education by operating as a personalized assistant. However, the possibility of generating incorrect, biased, or unhelpful answers are a key challenge to resolve when deploying LLMs in an education context. This work introduces an innovative architecture that combines the strengths of ChatGPT with a traditional information retrieval based chatbot framework to offer enhanced student support in higher education. Our empirical evaluations underscore the high promise of this approach.

{{</citation>}}


## cs.IR (4)



### (16/60) K-PERM: Personalized Response Generation Using Dynamic Knowledge Retrieval and Persona-Adaptive Queries (Kanak Raj et al., 2023)

{{<citation>}}

Kanak Raj, Kaushik Roy, Manas Gaur. (2023)  
**K-PERM: Personalized Response Generation Using Dynamic Knowledge Retrieval and Persona-Adaptive Queries**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.17748v1)  

---


**ABSTRACT**  
Personalizing conversational agents can enhance the quality of conversations and increase user engagement. However, they often lack external knowledge to appropriately tend to a user's persona. This is particularly crucial for practical applications like mental health support, nutrition planning, culturally sensitive conversations, or reducing toxic behavior in conversational agents. To enhance the relevance and comprehensiveness of personalized responses, we propose using a two-step approach that involves (1) selectively integrating user personas and (2) contextualizing the response with supplementing information from a background knowledge source. We develop K-PERM (Knowledge-guided PErsonalization with Reward Modulation), a dynamic conversational agent that combines these elements. K-PERM achieves state-of-the-art performance on the popular FoCus dataset, containing real-world personalized conversations concerning global landmarks. We show that using responses from K-PERM can improve performance in state-of-the-art LLMs (GPT 3.5) by 10.5%, highlighting the impact of K-PERM for personalizing chatbots.

{{</citation>}}


### (17/60) Investigating the Effects of Sparse Attention on Cross-Encoders (Ferdinand Schlatt et al., 2023)

{{<citation>}}

Ferdinand Schlatt, Maik Fröbe, Matthias Hagen. (2023)  
**Investigating the Effects of Sparse Attention on Cross-Encoders**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.17649v1)  

---


**ABSTRACT**  
Cross-encoders are effective passage and document re-rankers but less efficient than other neural or classic retrieval models. A few previous studies have applied windowed self-attention to make cross-encoders more efficient. However, these studies did not investigate the potential and limits of different attention patterns or window sizes. We close this gap and systematically analyze how token interactions can be reduced without harming the re-ranking effectiveness. Experimenting with asymmetric attention and different window sizes, we find that the query tokens do not need to attend to the passage or document tokens for effective re-ranking and that very small window sizes suffice. In our experiments, even windows of 4 tokens still yield effectiveness on par with previous cross-encoders while reducing the memory requirements to at most 78% / 41% and being 1% / 43% faster at inference time for passages / documents.

{{</citation>}}


### (18/60) Towards Mitigating Dimensional Collapse of Representations in Collaborative Filtering (Huiyuan Chen et al., 2023)

{{<citation>}}

Huiyuan Chen, Vivian Lai, Hongye Jin, Zhimeng Jiang, Mahashweta Das, Xia Hu. (2023)  
**Towards Mitigating Dimensional Collapse of Representations in Collaborative Filtering**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.17468v1)  

---


**ABSTRACT**  
Contrastive Learning (CL) has shown promising performance in collaborative filtering. The key idea is to generate augmentation-invariant embeddings by maximizing the Mutual Information between different augmented views of the same instance. However, we empirically observe that existing CL models suffer from the \textsl{dimensional collapse} issue, where user/item embeddings only span a low-dimension subspace of the entire feature space. This suppresses other dimensional information and weakens the distinguishability of embeddings. Here we propose a non-contrastive learning objective, named nCL, which explicitly mitigates dimensional collapse of representations in collaborative filtering. Our nCL aims to achieve geometric properties of \textsl{Alignment} and \textsl{Compactness} on the embedding space. In particular, the alignment tries to push together representations of positive-related user-item pairs, while compactness tends to find the optimal coding length of user/item embeddings, subject to a given distortion. More importantly, our nCL does not require data augmentation nor negative sampling during training, making it scalable to large datasets. Experimental results demonstrate the superiority of our nCL.

{{</citation>}}


### (19/60) Break Out of a Pigeonhole: A Unified Framework for Examining Miscalibration, Bias, and Stereotype in Recommender Systems (Yongsu Ahn et al., 2023)

{{<citation>}}

Yongsu Ahn, Yu-Ru Lin. (2023)  
**Break Out of a Pigeonhole: A Unified Framework for Examining Miscalibration, Bias, and Stereotype in Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.17443v1)  

---


**ABSTRACT**  
Despite the benefits of personalizing items and information tailored to users' needs, it has been found that recommender systems tend to introduce biases that favor popular items or certain categories of items, and dominant user groups. In this study, we aim to characterize the systematic errors of a recommendation system and how they manifest in various accountability issues, such as stereotypes, biases, and miscalibration. We propose a unified framework that distinguishes the sources of prediction errors into a set of key measures that quantify the various types of system-induced effects, both at the individual and collective levels. Based on our measuring framework, we examine the most widely adopted algorithms in the context of movie recommendation. Our research reveals three important findings: (1) Differences between algorithms: recommendations generated by simpler algorithms tend to be more stereotypical but less biased than those generated by more complex algorithms. (2) Disparate impact on groups and individuals: system-induced biases and stereotypes have a disproportionate effect on atypical users and minority groups (e.g., women and older users). (3) Mitigation opportunity: using structural equation modeling, we identify the interactions between user characteristics (typicality and diversity), system-induced effects, and miscalibration. We further investigate the possibility of mitigating system-induced effects by oversampling underrepresented groups and individuals, which was found to be effective in reducing stereotypes and improving recommendation quality. Our research is the first systematic examination of not only system-induced effects and miscalibration but also the stereotyping issue in recommender systems.

{{</citation>}}


## eess.SY (1)



### (20/60) Physics-informed Graphical Neural Network for Power System State Estimation (Quang-Ha Ngo et al., 2023)

{{<citation>}}

Quang-Ha Ngo, Bang L. H. Nguyen, Tuyen V. Vu, Jianhua Zhang, Tuan Ngo. (2023)  
**Physics-informed Graphical Neural Network for Power System State Estimation**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.17738v1)  

---


**ABSTRACT**  
State estimation is highly critical for accurately observing the dynamic behavior of the power grids and minimizing risks from cyber threats. However, existing state estimation methods encounter challenges in accurately capturing power system dynamics, primarily because of limitations in encoding the grid topology and sparse measurements. This paper proposes a physics-informed graphical learning state estimation method to address these limitations by leveraging both domain physical knowledge and a graph neural network (GNN). We employ a GNN architecture that can handle the graph-structured data of power systems more effectively than traditional data-driven methods. The physics-based knowledge is constructed from the branch current formulation, making the approach adaptable to both transmission and distribution systems. The validation results of three IEEE test systems show that the proposed method can achieve lower mean square error more than 20% than the conventional methods.

{{</citation>}}


## cs.CL (13)



### (21/60) Principled Gradient-based Markov Chain Monte Carlo for Text Generation (Li Du et al., 2023)

{{<citation>}}

Li Du, Afra Amini, Lucas Torroba Hennigen, Xinyan Velocity Yu, Jason Eisner, Holden Lee, Ryan Cotterell. (2023)  
**Principled Gradient-based Markov Chain Monte Carlo for Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2312.17710v1)  

---


**ABSTRACT**  
Recent papers have demonstrated the possibility of energy-based text generation by adapting gradient-based sampling algorithms, a paradigm of MCMC algorithms that promises fast convergence. However, as we show in this paper, previous attempts on this approach to text generation all fail to sample correctly from the target language model distributions. To address this limitation, we consider the problem of designing text samplers that are faithful, meaning that they have the target text distribution as its limiting distribution. We propose several faithful gradient-based sampling algorithms to sample from the target energy-based text distribution correctly, and study their theoretical properties. Through experiments on various forms of text generation, we demonstrate that faithful samplers are able to generate more fluent text while adhering to the control objectives better.

{{</citation>}}


### (22/60) TuPy-E: detecting hate speech in Brazilian Portuguese social media with a novel dataset and comprehensive analysis of models (Felipe Oliveira et al., 2023)

{{<citation>}}

Felipe Oliveira, Victoria Reis, Nelson Ebecken. (2023)  
**TuPy-E: detecting hate speech in Brazilian Portuguese social media with a novel dataset and comprehensive analysis of models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2312.17704v1)  

---


**ABSTRACT**  
Social media has become integral to human interaction, providing a platform for communication and expression. However, the rise of hate speech on these platforms poses significant risks to individuals and communities. Detecting and addressing hate speech is particularly challenging in languages like Portuguese due to its rich vocabulary, complex grammar, and regional variations. To address this, we introduce TuPy-E, the largest annotated Portuguese corpus for hate speech detection. TuPy-E leverages an open-source approach, fostering collaboration within the research community. We conduct a detailed analysis using advanced techniques like BERT models, contributing to both academic understanding and practical applications

{{</citation>}}


### (23/60) Gemini in Reasoning: Unveiling Commonsense in Multimodal Large Language Models (Yuqing Wang et al., 2023)

{{<citation>}}

Yuqing Wang, Yun Zhao. (2023)  
**Gemini in Reasoning: Unveiling Commonsense in Multimodal Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: AI, GPT, GPT-4, Google, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.17661v1)  

---


**ABSTRACT**  
The burgeoning interest in Multimodal Large Language Models (MLLMs), such as OpenAI's GPT-4V(ision), has significantly impacted both academic and industrial realms. These models enhance Large Language Models (LLMs) with advanced visual understanding capabilities, facilitating their application in a variety of multimodal tasks. Recently, Google introduced Gemini, a cutting-edge MLLM designed specifically for multimodal integration. Despite its advancements, preliminary benchmarks indicate that Gemini lags behind GPT models in commonsense reasoning tasks. However, this assessment, based on a limited dataset (i.e., HellaSWAG), does not fully capture Gemini's authentic commonsense reasoning potential. To address this gap, our study undertakes a thorough evaluation of Gemini's performance in complex reasoning tasks that necessitate the integration of commonsense knowledge across modalities. We carry out a comprehensive analysis of 12 commonsense reasoning datasets, ranging from general to domain-specific tasks. This includes 11 datasets focused solely on language, as well as one that incorporates multimodal elements. Our experiments across four LLMs and two MLLMs demonstrate Gemini's competitive commonsense reasoning capabilities. Additionally, we identify common challenges faced by current LLMs and MLLMs in addressing commonsense problems, underscoring the need for further advancements in enhancing the commonsense reasoning abilities of these models.

{{</citation>}}


### (24/60) Large Language Models for Generative Information Extraction: A Survey (Derong Xu et al., 2023)

{{<citation>}}

Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, Enhong Chen. (2023)  
**Large Language Models for Generative Information Extraction: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction, Language Model  
[Paper Link](http://arxiv.org/abs/2312.17617v1)  

---


**ABSTRACT**  
Information extraction (IE) aims to extract structural knowledge (such as entities, relations, and events) from plain natural language texts. Recently, generative Large Language Models (LLMs) have demonstrated remarkable capabilities in text understanding and generation, allowing for generalization across various domains and tasks. As a result, numerous works have been proposed to harness abilities of LLMs and offer viable solutions for IE tasks based on a generative paradigm. To conduct a comprehensive systematic review and exploration of LLM efforts for IE tasks, in this study, we survey the most recent advancements in this field. We first present an extensive overview by categorizing these works in terms of various IE subtasks and learning paradigms, then we empirically analyze the most advanced methods and discover the emerging trend of IE tasks with LLMs. Based on thorough review conducted, we identify several insights in technique and promising research directions that deserve further exploration in future studies. We maintain a public repository and consistently update related resources at: \url{https://github.com/quqxui/Awesome-LLM4IE-Papers}.

{{</citation>}}


### (25/60) Towards Faithful Explanations for Text Classification with Robustness Improvement and Explanation Guided Training (Dongfang Li et al., 2023)

{{<citation>}}

Dongfang Li, Baotian Hu, Qingcai Chen, Shan He. (2023)  
**Towards Faithful Explanations for Text Classification with Robustness Improvement and Explanation Guided Training**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Text Classification  
[Paper Link](http://arxiv.org/abs/2312.17591v1)  

---


**ABSTRACT**  
Feature attribution methods highlight the important input tokens as explanations to model predictions, which have been widely applied to deep neural networks towards trustworthy AI. However, recent works show that explanations provided by these methods face challenges of being faithful and robust. In this paper, we propose a method with Robustness improvement and Explanation Guided training towards more faithful EXplanations (REGEX) for text classification. First, we improve model robustness by input gradient regularization technique and virtual adversarial training. Secondly, we use salient ranking to mask noisy tokens and maximize the similarity between model attention and feature attribution, which can be seen as a self-training procedure without importing other external information. We conduct extensive experiments on six datasets with five attribution methods, and also evaluate the faithfulness in the out-of-domain setting. The results show that REGEX improves fidelity metrics of explanations in all settings and further achieves consistent gains based on two randomization tests. Moreover, we show that using highlight explanations produced by REGEX to train select-then-predict models results in comparable task performance to the end-to-end method.

{{</citation>}}


### (26/60) Action-Item-Driven Summarization of Long Meeting Transcripts (Logan Golia et al., 2023)

{{<citation>}}

Logan Golia, Jugal Kalita. (2023)  
**Action-Item-Driven Summarization of Long Meeting Transcripts**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: BERT, Summarization, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.17581v1)  

---


**ABSTRACT**  
The increased prevalence of online meetings has significantly enhanced the practicality of a model that can automatically generate the summary of a given meeting. This paper introduces a novel and effective approach to automate the generation of meeting summaries. Current approaches to this problem generate general and basic summaries, considering the meeting simply as a long dialogue. However, our novel algorithms can generate abstractive meeting summaries that are driven by the action items contained in the meeting transcript. This is done by recursively generating summaries and employing our action-item extraction algorithm for each section of the meeting in parallel. All of these sectional summaries are then combined and summarized together to create a coherent and action-item-driven summary. In addition, this paper introduces three novel methods for dividing up long transcripts into topic-based sections to improve the time efficiency of our algorithm, as well as to resolve the issue of large language models (LLMs) forgetting long-term dependencies. Our pipeline achieved a BERTScore of 64.98 across the AMI corpus, which is an approximately 4.98% increase from the current state-of-the-art result produced by a fine-tuned BART (Bidirectional and Auto-Regressive Transformers) model.

{{</citation>}}


### (27/60) Building Efficient Universal Classifiers with Natural Language Inference (Moritz Laurer et al., 2023)

{{<citation>}}

Moritz Laurer, Wouter van Atteveldt, Andreu Casas, Kasper Welbers. (2023)  
**Building Efficient Universal Classifiers with Natural Language Inference**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model, NLI, Natural Language Inference  
[Paper Link](http://arxiv.org/abs/2312.17543v1)  

---


**ABSTRACT**  
Generative Large Language Models (LLMs) have become the mainstream choice for fewshot and zeroshot learning thanks to the universality of text generation. Many users, however, do not need the broad capabilities of generative LLMs when they only want to automate a classification task. Smaller BERT-like models can also learn universal tasks, which allow them to do any text classification task without requiring fine-tuning (zeroshot classification) or to learn new tasks with only a few examples (fewshot), while being significantly more efficient than generative LLMs. This paper (1) explains how Natural Language Inference (NLI) can be used as a universal classification task that follows similar principles as instruction fine-tuning of generative LLMs, (2) provides a step-by-step guide with reusable Jupyter notebooks for building a universal classifier, and (3) shares the resulting universal classifier that is trained on 33 datasets with 389 diverse classes. Parts of the code we share has been used to train our older zeroshot classifiers that have been downloaded more than 55 million times via the Hugging Face Hub as of December 2023. Our new classifier improves zeroshot performance by 9.4%.

{{</citation>}}


### (28/60) Enhancing Quantitative Reasoning Skills of Large Language Models through Dimension Perception (Yuncheng Huang et al., 2023)

{{<citation>}}

Yuncheng Huang, Qianyu He, Jiaqing Liang, Sihang Jiang, Yanghua Xiao, Yunwen Chen. (2023)  
**Enhancing Quantitative Reasoning Skills of Large Language Models through Dimension Perception**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.17532v1)  

---


**ABSTRACT**  
Quantities are distinct and critical components of texts that characterize the magnitude properties of entities, providing a precise perspective for the understanding of natural language, especially for reasoning tasks. In recent years, there has been a flurry of research on reasoning tasks based on large language models (LLMs), most of which solely focus on numerical values, neglecting the dimensional concept of quantities with units despite its importance. We argue that the concept of dimension is essential for precisely understanding quantities and of great significance for LLMs to perform quantitative reasoning. However, the lack of dimension knowledge and quantity-related benchmarks has resulted in low performance of LLMs. Hence, we present a framework to enhance the quantitative reasoning ability of language models based on dimension perception. We first construct a dimensional unit knowledge base (DimUnitKB) to address the knowledge gap in this area. We propose a benchmark DimEval consisting of seven tasks of three categories to probe and enhance the dimension perception skills of LLMs. To evaluate the effectiveness of our methods, we propose a quantitative reasoning task and conduct experiments. The experimental results show that our dimension perception method dramatically improves accuracy (43.55%->50.67%) on quantitative reasoning tasks compared to GPT-4.

{{</citation>}}


### (29/60) Cooperation on the Fly: Exploring Language Agents for Ad Hoc Teamwork in the Avalon Game (Zijing Shi et al., 2023)

{{<citation>}}

Zijing Shi, Meng Fang, Shunfeng Zheng, Shilong Deng, Ling Chen, Yali Du. (2023)  
**Cooperation on the Fly: Exploring Language Agents for Ad Hoc Teamwork in the Avalon Game**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.17515v1)  

---


**ABSTRACT**  
Multi-agent collaboration with Large Language Models (LLMs) demonstrates proficiency in basic tasks, yet its efficiency in more complex scenarios remains unexplored. In gaming environments, these agents often face situations without established coordination protocols, requiring them to make intelligent inferences about teammates from limited data. This problem motivates the area of ad hoc teamwork, in which an agent may potentially cooperate with a variety of teammates to achieve a shared goal. Our study focuses on the ad hoc teamwork problem where the agent operates in an environment driven by natural language. Our findings reveal the potential of LLM agents in team collaboration, highlighting issues related to hallucinations in communication. To address this issue, we develop CodeAct, a general agent that equips LLM with enhanced memory and code-driven reasoning, enabling the repurposing of partial information for rapid adaptation to new teammates.

{{</citation>}}


### (30/60) Truth Forest: Toward Multi-Scale Truthfulness in Large Language Models through Intervention without Tuning (Zhongzhi Chen et al., 2023)

{{<citation>}}

Zhongzhi Chen, Xingwu Sun, Xianfeng Jiao, Fengzong Lian, Zhanhui Kang, Di Wang, Cheng-Zhong Xu. (2023)  
**Truth Forest: Toward Multi-Scale Truthfulness in Large Language Models through Intervention without Tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.17484v1)  

---


**ABSTRACT**  
Despite the great success of large language models (LLMs) in various tasks, they suffer from generating hallucinations. We introduce Truth Forest, a method that enhances truthfulness in LLMs by uncovering hidden truth representations using multi-dimensional orthogonal probes. Specifically, it creates multiple orthogonal bases for modeling truth by incorporating orthogonal constraints into the probes. Moreover, we introduce Random Peek, a systematic technique considering an extended range of positions within the sequence, reducing the gap between discerning and generating truth features in LLMs. By employing this approach, we improved the truthfulness of Llama-2-7B from 40.8\% to 74.5\% on TruthfulQA. Likewise, significant improvements are observed in fine-tuned models. We conducted a thorough analysis of truth features using probes. Our visualization results show that orthogonal probes capture complementary truth-related features, forming well-defined clusters that reveal the inherent structure of the dataset. Code: \url{https://github.com/jongjyh/trfr}

{{</citation>}}


### (31/60) MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining (Jacob Portes et al., 2023)

{{<citation>}}

Jacob Portes, Alex Trott, Sam Havens, Daniel King, Abhinav Venigalla, Moin Nadeem, Nikhil Sardana, Daya Khudia, Jonathan Frankle. (2023)  
**MosaicBERT: A Bidirectional Encoder Optimized for Fast Pretraining**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Attention, BERT, Bias, GLUE, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2312.17482v1)  

---


**ABSTRACT**  
Although BERT-style encoder models are heavily used in NLP research, many researchers do not pretrain their own BERTs from scratch due to the high cost of training. In the past half-decade since BERT first rose to prominence, many advances have been made with other transformer architectures and training configurations that have yet to be systematically incorporated into BERT. Here, we introduce MosaicBERT, a BERT-style encoder architecture and training recipe that is empirically optimized for fast pretraining. This efficient architecture incorporates FlashAttention, Attention with Linear Biases (ALiBi), Gated Linear Units (GLU), a module to dynamically remove padded tokens, and low precision LayerNorm into the classic transformer encoder block. The training recipe includes a 30% masking ratio for the Masked Language Modeling (MLM) objective, bfloat16 precision, and vocabulary size optimized for GPU throughput, in addition to best-practices from RoBERTa and other encoder models. When pretrained from scratch on the C4 dataset, this base model achieves a downstream average GLUE (dev) score of 79.6 in 1.13 hours on 8 A100 80 GB GPUs at a cost of roughly $20. We plot extensive accuracy vs. pretraining speed Pareto curves and show that MosaicBERT base and large are consistently Pareto optimal when compared to a competitive BERT base and large. This empirical speed up in pretraining enables researchers and engineers to pretrain custom BERT-style models at low cost instead of finetune on existing generic models. We open source our model weights and code.

{{</citation>}}


### (32/60) Exploring the Sensitivity of LLMs' Decision-Making Capabilities: Insights from Prompt Variation and Hyperparameters (Manikanta Loya et al., 2023)

{{<citation>}}

Manikanta Loya, Divya Anand Sinha, Richard Futrell. (2023)  
**Exploring the Sensitivity of LLMs' Decision-Making Capabilities: Insights from Prompt Variation and Hyperparameters**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.17476v1)  

---


**ABSTRACT**  
The advancement of Large Language Models (LLMs) has led to their widespread use across a broad spectrum of tasks including decision making. Prior studies have compared the decision making abilities of LLMs with those of humans from a psychological perspective. However, these studies have not always properly accounted for the sensitivity of LLMs' behavior to hyperparameters and variations in the prompt. In this study, we examine LLMs' performance on the Horizon decision making task studied by Binz and Schulz (2023) analyzing how LLMs respond to variations in prompts and hyperparameters. By experimenting on three OpenAI language models possessing different capabilities, we observe that the decision making abilities fluctuate based on the input prompts and temperature settings. Contrary to previous findings language models display a human-like exploration exploitation tradeoff after simple adjustments to the prompt.

{{</citation>}}


### (33/60) EHR Interaction Between Patients and AI: NoteAid EHR Interaction (Xiaocheng Zhang et al., 2023)

{{<citation>}}

Xiaocheng Zhang, Zonghai Yao, Hong Yu. (2023)  
**EHR Interaction Between Patients and AI: NoteAid EHR Interaction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.17475v1)  

---


**ABSTRACT**  
With the rapid advancement of Large Language Models (LLMs) and their outstanding performance in semantic and contextual comprehension, the potential of LLMs in specialized domains warrants exploration. This paper introduces the NoteAid EHR Interaction Pipeline, an innovative approach developed using generative LLMs to assist in patient education, a task stemming from the need to aid patients in understanding Electronic Health Records (EHRs). Building upon the NoteAid work, we designed two novel tasks from the patient's perspective: providing explanations for EHR content that patients may not understand and answering questions posed by patients after reading their EHRs. We extracted datasets containing 10,000 instances from MIMIC Discharge Summaries and 876 instances from the MADE medical notes collection, respectively, executing the two tasks through the NoteAid EHR Interaction Pipeline with these data. Performance data of LLMs on these tasks were collected and constructed as the corresponding NoteAid EHR Interaction Dataset. Through a comprehensive evaluation of the entire dataset using LLM assessment and a rigorous manual evaluation of 64 instances, we showcase the potential of LLMs in patient education. Besides, the results provide valuable data support for future exploration and applications in this domain while also supplying high-quality synthetic datasets for in-house system training.

{{</citation>}}


## cs.CV (10)



### (34/60) Multiscale Vision Transformers meet Bipartite Matching for efficient single-stage Action Localization (Ioanna Ntinou et al., 2023)

{{<citation>}}

Ioanna Ntinou, Enrique Sanchez, Georgios Tzimiropoulos. (2023)  
**Multiscale Vision Transformers meet Bipartite Matching for efficient single-stage Action Localization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.17686v1)  

---


**ABSTRACT**  
Action Localization is a challenging problem that combines detection and recognition tasks, which are often addressed separately. State-of-the-art methods rely on off-the-shelf bounding box detections pre-computed at high resolution and propose transformer models that focus on the classification task alone. Such two-stage solutions are prohibitive for real-time deployment. On the other hand, single-stage methods target both tasks by devoting part of the network (generally the backbone) to sharing the majority of the workload, compromising performance for speed. These methods build on adding a DETR head with learnable queries that, after cross- and self-attention can be sent to corresponding MLPs for detecting a person's bounding box and action. However, DETR-like architectures are challenging to train and can incur in big complexity.   In this paper, we observe that a straight bipartite matching loss can be applied to the output tokens of a vision transformer. This results in a backbone + MLP architecture that can do both tasks without the need of an extra encoder-decoder head and learnable queries. We show that a single MViT-S architecture trained with bipartite matching to perform both tasks surpasses the same MViT-S when trained with RoI align on pre-computed bounding boxes. With a careful design of token pooling and the proposed training pipeline, our MViTv2-S model achieves +3 mAP on AVA2.2. w.r.t. the two-stage counterpart. Code and models will be released after paper revision.

{{</citation>}}


### (35/60) One-Shot Multi-Rate Pruning of Graph Convolutional Networks (Hichem Sahbi, 2023)

{{<citation>}}

Hichem Sahbi. (2023)  
**One-Shot Multi-Rate Pruning of Graph Convolutional Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Graph Convolutional Network, Pruning  
[Paper Link](http://arxiv.org/abs/2312.17615v1)  

---


**ABSTRACT**  
In this paper, we devise a novel lightweight Graph Convolutional Network (GCN) design dubbed as Multi-Rate Magnitude Pruning (MRMP) that jointly trains network topology and weights. Our method is variational and proceeds by aligning the weight distribution of the learned networks with an a priori distribution. In the one hand, this allows implementing any fixed pruning rate, and also enhancing the generalization performances of the designed lightweight GCNs. In the other hand, MRMP achieves a joint training of multiple GCNs, on top of shared weights, in order to extrapolate accurate networks at any targeted pruning rate without retraining their weights. Extensive experiments conducted on the challenging task of skeleton-based recognition show a substantial gain of our lightweight GCNs particularly at very high pruning regimes.

{{</citation>}}


### (36/60) P2M2-Net: Part-Aware Prompt-Guided Multimodal Point Cloud Completion (Linlian Jiang et al., 2023)

{{<citation>}}

Linlian Jiang, Pan Chen, Ye Wang, Tieru Wu, Rui Ma. (2023)  
**P2M2-Net: Part-Aware Prompt-Guided Multimodal Point Cloud Completion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.17611v1)  

---


**ABSTRACT**  
Inferring missing regions from severely occluded point clouds is highly challenging. Especially for 3D shapes with rich geometry and structure details, inherent ambiguities of the unknown parts are existing. Existing approaches either learn a one-to-one mapping in a supervised manner or train a generative model to synthesize the missing points for the completion of 3D point cloud shapes. These methods, however, lack the controllability for the completion process and the results are either deterministic or exhibiting uncontrolled diversity. Inspired by the prompt-driven data generation and editing, we propose a novel prompt-guided point cloud completion framework, coined P2M2-Net, to enable more controllable and more diverse shape completion. Given an input partial point cloud and a text prompt describing the part-aware information such as semantics and structure of the missing region, our Transformer-based completion network can efficiently fuse the multimodal features and generate diverse results following the prompt guidance. We train the P2M2-Net on a new large-scale PartNet-Prompt dataset and conduct extensive experiments on two challenging shape completion benchmarks. Quantitative and qualitative results show the efficacy of incorporating prompts for more controllable part-aware point cloud completion and generation. Code and data are available at https://github.com/JLU-ICL/P2M2-Net.

{{</citation>}}


### (37/60) Informative Rays Selection for Few-Shot Neural Radiance Fields (Marco Orsingher et al., 2023)

{{<citation>}}

Marco Orsingher, Anthony Dell'Eva, Paolo Zani, Paolo Medici, Massimo Bertozzi. (2023)  
**Informative Rays Selection for Few-Shot Neural Radiance Fields**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.17561v1)  

---


**ABSTRACT**  
Neural Radiance Fields (NeRF) have recently emerged as a powerful method for image-based 3D reconstruction, but the lengthy per-scene optimization limits their practical usage, especially in resource-constrained settings. Existing approaches solve this issue by reducing the number of input views and regularizing the learned volumetric representation with either complex losses or additional inputs from other modalities. In this paper, we present KeyNeRF, a simple yet effective method for training NeRF in few-shot scenarios by focusing on key informative rays. Such rays are first selected at camera level by a view selection algorithm that promotes baseline diversity while guaranteeing scene coverage, then at pixel level by sampling from a probability distribution based on local image entropy. Our approach performs favorably against state-of-the-art methods, while requiring minimal changes to existing NeRF codebases.

{{</citation>}}


### (38/60) A Fully Automated Pipeline Using Swin Transformers for Deep Learning-Based Blood Segmentation on Head CT Scans After Aneurysmal Subarachnoid Hemorrhage (Sergio Garcia Garcia et al., 2023)

{{<citation>}}

Sergio Garcia Garcia, Santiago Cepeda, Ignacio Arrese, Rosario Sarabia. (2023)  
**A Fully Automated Pipeline Using Swin Transformers for Deep Learning-Based Blood Segmentation on Head CT Scans After Aneurysmal Subarachnoid Hemorrhage**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, stat-ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.17553v1)  

---


**ABSTRACT**  
Background: Accurate volumetric assessment of spontaneous subarachnoid hemorrhage (SAH) is a labor-intensive task performed with current manual and semiautomatic methods that might be relevant for its clinical and prognostic implications. In the present research, we sought to develop and validate an artificial intelligence-driven, fully automated blood segmentation tool for SAH patients via noncontrast computed tomography (NCCT) scans employing a transformer-based Swin UNETR architecture. Methods: We retrospectively analyzed NCCT scans from patients with confirmed aneurysmal subarachnoid hemorrhage (aSAH) utilizing the Swin UNETR for segmentation. The performance of the proposed method was evaluated against manually segmented ground truth data using metrics such as Dice score, intersection over union (IoU), the volumetric similarity index (VSI), the symmetric average surface distance (SASD), and sensitivity and specificity. A validation cohort from an external institution was included to test the generalizability of the model. Results: The model demonstrated high accuracy with robust performance metrics across the internal and external validation cohorts. Notably, it achieved high Dice coefficient (0.873), IoU (0.810), VSI (0.840), sensitivity (0.821) and specificity (0.996) values and a low SASD (1.866), suggesting proficiency in segmenting blood in SAH patients. The model's efficiency was reflected in its processing speed, indicating potential for real-time applications. Conclusions: Our Swin UNETR-based model offers significant advances in the automated segmentation of blood after aSAH on NCCT images. Despite the computational intensity, the model operates effectively on standard hardware with a user-friendly interface, facilitating broader clinical adoption. Further validation across diverse datasets is warranted to confirm its clinical reliability.

{{</citation>}}


### (39/60) FerKD: Surgical Label Adaptation for Efficient Distillation (Zhiqiang Shen, 2023)

{{<citation>}}

Zhiqiang Shen. (2023)  
**FerKD: Surgical Label Adaptation for Efficient Distillation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.17473v1)  

---


**ABSTRACT**  
We present FerKD, a novel efficient knowledge distillation framework that incorporates partial soft-hard label adaptation coupled with a region-calibration mechanism. Our approach stems from the observation and intuition that standard data augmentations, such as RandomResizedCrop, tend to transform inputs into diverse conditions: easy positives, hard positives, or hard negatives. In traditional distillation frameworks, these transformed samples are utilized equally through their predictive probabilities derived from pretrained teacher models. However, merely relying on prediction values from a pretrained teacher, a common practice in prior studies, neglects the reliability of these soft label predictions. To address this, we propose a new scheme that calibrates the less-confident regions to be the context using softened hard groundtruth labels. Our approach involves the processes of hard regions mining + calibration. We demonstrate empirically that this method can dramatically improve the convergence speed and final accuracy. Additionally, we find that a consistent mixing strategy can stabilize the distributions of soft supervision, taking advantage of the soft labels. As a result, we introduce a stabilized SelfMix augmentation that weakens the variation of the mixed images and corresponding soft labels through mixing similar regions within the same image. FerKD is an intuitive and well-designed learning system that eliminates several heuristics and hyperparameters in former FKD solution. More importantly, it achieves remarkable improvement on ImageNet-1K and downstream tasks. For instance, FerKD achieves 81.2% on ImageNet-1K with ResNet-50, outperforming FKD and FunMatch by remarkable margins. Leveraging better pre-trained weights and larger architectures, our finetuned ViT-G14 even achieves 89.9%. Our code is available at https://github.com/szq0214/FKD/tree/main/FerKD.

{{</citation>}}


### (40/60) Tracking with Human-Intent Reasoning (Jiawen Zhu et al., 2023)

{{<citation>}}

Jiawen Zhu, Zhi-Qi Cheng, Jun-Yan He, Chenyang Li, Bin Luo, Huchuan Lu, Yifeng Geng, Xuansong Xie. (2023)  
**Tracking with Human-Intent Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.17448v1)  

---


**ABSTRACT**  
Advances in perception modeling have significantly improved the performance of object tracking. However, the current methods for specifying the target object in the initial frame are either by 1) using a box or mask template, or by 2) providing an explicit language description. These manners are cumbersome and do not allow the tracker to have self-reasoning ability. Therefore, this work proposes a new tracking task -- Instruction Tracking, which involves providing implicit tracking instructions that require the trackers to perform tracking automatically in video frames. To achieve this, we investigate the integration of knowledge and reasoning capabilities from a Large Vision-Language Model (LVLM) for object tracking. Specifically, we propose a tracker called TrackGPT, which is capable of performing complex reasoning-based tracking. TrackGPT first uses LVLM to understand tracking instructions and condense the cues of what target to track into referring embeddings. The perception component then generates the tracking results based on the embeddings. To evaluate the performance of TrackGPT, we construct an instruction tracking benchmark called InsTrack, which contains over one thousand instruction-video pairs for instruction tuning and evaluation. Experiments show that TrackGPT achieves competitive performance on referring video object segmentation benchmarks, such as getting a new state-of the-art performance of 66.5 $\mathcal{J}\&\mathcal{F}$ on Refer-DAVIS. It also demonstrates a superior performance of instruction tracking under new evaluation protocols. The code and models are available at \href{https://github.com/jiawen-zhu/TrackGPT}{https://github.com/jiawen-zhu/TrackGPT}.

{{</citation>}}


### (41/60) An Empirical Study of Scaling Law for OCR (Miao Rang et al., 2023)

{{<citation>}}

Miao Rang, Zhenni Bi, Chuanjian Liu, Yunhe Wang, Kai Han. (2023)  
**An Empirical Study of Scaling Law for OCR**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NLP, Natural Language Processing, OCR  
[Paper Link](http://arxiv.org/abs/2401.00028v1)  

---


**ABSTRACT**  
The laws of model size, data volume, computation and model performance have been extensively studied in the field of Natural Language Processing (NLP). However, the scaling laws in Optical Character Recognition (OCR) have not yet been investigated. To address this, we conducted comprehensive studies that involved examining the correlation between performance and the scale of models, data volume and computation in the field of text recognition.Conclusively, the study demonstrates smooth power laws between performance and model size, as well as training data volume, when other influencing factors are held constant. Additionally, we have constructed a large-scale dataset called REBU-Syn, which comprises 6 million real samples and 18 million synthetic samples. Based on our scaling law and new dataset, we have successfully trained a scene text recognition model, achieving a new state-ofthe-art on 6 common test benchmarks with a top-1 average accuracy of 97.42%.

{{</citation>}}


### (42/60) Video Understanding with Large Language Models: A Survey (Yunlong Tang et al., 2023)

{{<citation>}}

Yunlong Tang, Jing Bi, Siting Xu, Luchuan Song, Susan Liang, Teng Wang, Daoan Zhang, Jie An, Jingyang Lin, Rongyi Zhu, Ali Vosoughi, Chao Huang, Zeliang Zhang, Feng Zheng, Jianguo Zhang, Ping Luo, Jiebo Luo, Chenliang Xu. (2023)  
**Video Understanding with Large Language Models: A Survey**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.17432v1)  

---


**ABSTRACT**  
With the burgeoning growth of online video platforms and the escalating volume of video content, the demand for proficient video understanding tools has intensified markedly. With Large Language Models (LLMs) showcasing remarkable capabilities in key language tasks, this survey provides a detailed overview of the recent advancements in video understanding harnessing the power of LLMs (Vid-LLMs). The emergent capabilities of Vid-LLMs are surprisingly advanced, particularly their ability for open-ended spatial-temporal reasoning combined with commonsense knowledge, suggesting a promising path for future video understanding. We examine the unique characteristics and capabilities of Vid-LLMs, categorizing the approaches into four main types: LLM-based Video Agents, Vid-LLMs Pretraining, Vid-LLMs Instruction Tuning, and Hybrid Methods. Furthermore, this survey also presents a comprehensive study of the tasks and datasets for Vid-LLMs, along with the methodologies employed for evaluation. Additionally, the survey explores the expansive applications of Vid-LLMs across various domains, thereby showcasing their remarkable scalability and versatility in addressing challenges in real-world video understanding. Finally, the survey summarizes the limitations of existing Vid-LLMs and the directions for future research. For more information, we recommend readers visit the repository at https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding.

{{</citation>}}


### (43/60) Commonsense for Zero-Shot Natural Language Video Localization (Meghana Holla et al., 2023)

{{<citation>}}

Meghana Holla, Ismini Lourentzou. (2023)  
**Commonsense for Zero-Shot Natural Language Video Localization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.17429v1)  

---


**ABSTRACT**  
Zero-shot Natural Language-Video Localization (NLVL) methods have exhibited promising results in training NLVL models exclusively with raw video data by dynamically generating video segments and pseudo-query annotations. However, existing pseudo-queries often lack grounding in the source video, resulting in unstructured and disjointed content. In this paper, we investigate the effectiveness of commonsense reasoning in zero-shot NLVL. Specifically, we present CORONET, a zero-shot NLVL framework that leverages commonsense to bridge the gap between videos and generated pseudo-queries via a commonsense enhancement module. CORONET employs Graph Convolution Networks (GCN) to encode commonsense information extracted from a knowledge graph, conditioned on the video, and cross-attention mechanisms to enhance the encoded video and pseudo-query representations prior to localization. Through empirical evaluations on two benchmark datasets, we demonstrate that CORONET surpasses both zero-shot and weakly supervised baselines, achieving improvements up to 32.13% across various recall thresholds and up to 6.33% in mIoU. These results underscore the significance of leveraging commonsense reasoning for zero-shot NLVL.

{{</citation>}}


## cs.RO (5)



### (44/60) Vocalics in Human-Drone Interaction (Marc Lieser et al., 2023)

{{<citation>}}

Marc Lieser, Ulrich Schwanecke. (2023)  
**Vocalics in Human-Drone Interaction**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2312.17668v1)  

---


**ABSTRACT**  
As the presence of flying robots continues to grow in both commercial and private sectors, it necessitates an understanding of appropriate methods for nonverbal interaction with humans. While visual cues, such as gestures incorporated into trajectories, are more apparent and thoroughly researched, acoustic cues have remained unexplored, despite their potential to enhance human-drone interaction. Given that additional audiovisual and sensory equipment is not always desired or practicable, and flight noise often masks potential acoustic communication in rotary-wing drones, such as through a loudspeaker, the rotors themselves offer potential for nonverbal communication. In this paper, quadrotor trajectories are augmented by acoustic information that does not visually affect the flight, but adds audible information that significantly facilitates distinctiveness. A user study (N=192) demonstrates that sonically augmenting the trajectories of two aerial gestures makes them more easily distinguishable. This enhancement contributes to human-drone interaction through onboard means, particularly in situations where the human cannot see or look at the drone.

{{</citation>}}


### (45/60) Adaptive Control Strategy for Quadruped Robots in Actuator Degradation Scenarios (Xinyuan Wu et al., 2023)

{{<citation>}}

Xinyuan Wu, Wentao Dong, Hang Lai, Yong Yu, Ying Wen. (2023)  
**Adaptive Control Strategy for Quadruped Robots in Actuator Degradation Scenarios**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.17606v1)  

---


**ABSTRACT**  
Quadruped robots have strong adaptability to extreme environments but may also experience faults. Once these faults occur, robots must be repaired before returning to the task, reducing their practical feasibility. One prevalent concern among these faults is actuator degradation, stemming from factors like device aging or unexpected operational events. Traditionally, addressing this problem has relied heavily on intricate fault-tolerant design, which demands deep domain expertise from developers and lacks generalizability. Learning-based approaches offer effective ways to mitigate these limitations, but a research gap exists in effectively deploying such methods on real-world quadruped robots. This paper introduces a pioneering teacher-student framework rooted in reinforcement learning, named Actuator Degradation Adaptation Transformer (ADAPT), aimed at addressing this research gap. This framework produces a unified control strategy, enabling the robot to sustain its locomotion and perform tasks despite sudden joint actuator faults, relying exclusively on its internal sensors. Empirical evaluations on the Unitree A1 platform validate the deployability and effectiveness of Adapt on real-world quadruped robots, and affirm the robustness and practicality of our approach.

{{</citation>}}


### (46/60) Unified Task and Motion Planning using Object-centric Abstractions of Motion Constraints (Alejandro Agostini et al., 2023)

{{<citation>}}

Alejandro Agostini, Justus Piater. (2023)  
**Unified Task and Motion Planning using Object-centric Abstractions of Motion Constraints**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.17605v1)  

---


**ABSTRACT**  
In task and motion planning (TAMP), the ambiguity and underdetermination of abstract descriptions used by task planning methods make it difficult to characterize physical constraints needed to successfully execute a task. The usual approach is to overlook such constraints at task planning level and to implement expensive sub-symbolic geometric reasoning techniques that perform multiple calls on unfeasible actions, plan corrections, and re-planning until a feasible solution is found. We propose an alternative TAMP approach that unifies task and motion planning into a single heuristic search. Our approach is based on an object-centric abstraction of motion constraints that permits leveraging the computational efficiency of off-the-shelf AI heuristic search to yield physically feasible plans. These plans can be directly transformed into object and motion parameters for task execution without the need of intensive sub-symbolic geometric reasoning.

{{</citation>}}


### (47/60) Exploring Deep Reinforcement Learning for Robust Target Tracking using Micro Aerial Vehicles (Alberto Dionigi et al., 2023)

{{<citation>}}

Alberto Dionigi, Mirko Leomanni, Alessandro Saviolo, Giuseppe Loianno, Gabriele Costante. (2023)  
**Exploring Deep Reinforcement Learning for Robust Target Tracking using Micro Aerial Vehicles**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.17552v1)  

---


**ABSTRACT**  
The capability to autonomously track a non-cooperative target is a key technological requirement for micro aerial vehicles. In this paper, we propose an output feedback control scheme based on deep reinforcement learning for controlling a micro aerial vehicle to persistently track a flying target while maintaining visual contact. The proposed method leverages relative position data for control, relaxing the assumption of having access to full state information which is typical of related approaches in literature. Moreover, we exploit classical robustness indicators in the learning process through domain randomization to increase the robustness of the learned policy. Experimental results validate the proposed approach for target tracking, demonstrating high performance and robustness with respect to mass mismatches and control delays. The resulting nonlinear controller significantly outperforms a standard model-based design in numerous off-nominal scenarios.

{{</citation>}}


### (48/60) Actuator-Constrained Reinforcement Learning for High-Speed Quadrupedal Locomotion (Young-Ha Shin et al., 2023)

{{<citation>}}

Young-Ha Shin, Tae-Gyu Song, Gwanghyeon Ji, Hae-Won Park. (2023)  
**Actuator-Constrained Reinforcement Learning for High-Speed Quadrupedal Locomotion**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.17507v1)  

---


**ABSTRACT**  
This paper presents a method for achieving high-speed running of a quadruped robot by considering the actuator torque-speed operating region in reinforcement learning. The physical properties and constraints of the actuator are included in the training process to reduce state transitions that are infeasible in the real world due to motor torque-speed limitations. The gait reward is designed to distribute motor torque evenly across all legs, contributing to more balanced power usage and mitigating performance bottlenecks due to single-motor saturation. Additionally, we designed a lightweight foot to enhance the robot's agility. We observed that applying the motor operating region as a constraint helps the policy network avoid infeasible areas during sampling. With the trained policy, KAIST Hound, a 45 kg quadruped robot, can run up to 6.5 m/s, which is the fastest speed among electric motor-based quadruped robots.

{{</citation>}}


## cs.AI (4)



### (49/60) Research on the Laws of Multimodal Perception and Cognition from a Cross-cultural Perspective -- Taking Overseas Chinese Gardens as an Example (Ran Chen et al., 2023)

{{<citation>}}

Ran Chen, Xueqi Yao, Jing Zhao, Shuhan Xu, Sirui Zhang, Yijun Mao. (2023)  
**Research on the Laws of Multimodal Perception and Cognition from a Cross-cultural Perspective -- Taking Overseas Chinese Gardens as an Example**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs-SI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.17642v1)  

---


**ABSTRACT**  
This study aims to explore the complex relationship between perceptual and cognitive interactions in multimodal data analysis,with a specific emphasis on spatial experience design in overseas Chinese gardens. It is found that evaluation content and images on social media can reflect individuals' concerns and sentiment responses, providing a rich data base for cognitive research that contains both sentimental and image-based cognitive information. Leveraging deep learning techniques, we analyze textual and visual data from social media, thereby unveiling the relationship between people's perceptions and sentiment cognition within the context of overseas Chinese gardens. In addition, our study introduces a multi-agent system (MAS)alongside AI agents. Each agent explores the laws of aesthetic cognition through chat scene simulation combined with web search. This study goes beyond the traditional approach of translating perceptions into sentiment scores, allowing for an extension of the research methodology in terms of directly analyzing texts and digging deeper into opinion data. This study provides new perspectives for understanding aesthetic experience and its impact on architecture and landscape design across diverse cultural contexts, which is an essential contribution to the field of cultural communication and aesthetic understanding.

{{</citation>}}


### (50/60) Olapa-MCoT: Enhancing the Chinese Mathematical Reasoning Capability of LLMs (Shaojie Zhu et al., 2023)

{{<citation>}}

Shaojie Zhu, Zhaobin Wang, Chengxiang Zhuo, Hui Lu, Bo Hu, Zang Li. (2023)  
**Olapa-MCoT: Enhancing the Chinese Mathematical Reasoning Capability of LLMs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-HC, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.17535v1)  

---


**ABSTRACT**  
CoT (Chain-of-Thought) is a way to solve reasoning problems for LLMs . Recently, many researches appear for improving the CoT capability of LLMs. In this work, we also proposed Olapa-MCoT, which is a LLMs based on llama2-13B PLM for finetuning and alignment learning. During the alignment training, we proposed the SimRRHF algorithm and Incorrect Data Relearning and mainly focused on optimizing the Chinese mathematical reasoning ability of Olapa-MCoT. The experiment achieved significant results, with the accuracy of Chinese mathematical reasoning up to 50%, 36% rise compared to llama2-13B. In addition, the accuracy of English reasoning ability also increased by nearly 4%.

{{</citation>}}


### (51/60) Culturally-Attuned Moral Machines: Implicit Learning of Human Value Systems by AI through Inverse Reinforcement Learning (Nigini Oliveira et al., 2023)

{{<citation>}}

Nigini Oliveira, Jasmine Li, Koosha Khalvati, Rodolfo Cortes Barragan, Katharina Reinecke, Andrew N. Meltzoff, Rajesh P. N. Rao. (2023)  
**Culturally-Attuned Moral Machines: Implicit Learning of Human Value Systems by AI through Inverse Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-HC, cs-LG, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.17479v1)  

---


**ABSTRACT**  
Constructing a universal moral code for artificial intelligence (AI) is difficult or even impossible, given that different human cultures have different definitions of morality and different societal norms. We therefore argue that the value system of an AI should be culturally attuned: just as a child raised in a particular culture learns the specific values and norms of that culture, we propose that an AI agent operating in a particular human community should acquire that community's moral, ethical, and cultural codes. How AI systems might acquire such codes from human observation and interaction has remained an open question. Here, we propose using inverse reinforcement learning (IRL) as a method for AI agents to acquire a culturally-attuned value system implicitly. We test our approach using an experimental paradigm in which AI agents use IRL to learn different reward functions, which govern the agents' moral values, by observing the behavior of different cultural groups in an online virtual world requiring real-time decision making. We show that an AI agent learning from the average behavior of a particular cultural group can acquire altruistic characteristics reflective of that group's behavior, and this learned value system can generalize to new scenarios requiring altruistic judgments. Our results provide, to our knowledge, the first demonstration that AI agents could potentially be endowed with the ability to continually learn their values and norms from observing and interacting with humans, thereby becoming attuned to the culture they are operating in.

{{</citation>}}


### (52/60) SMoT: Think in State Machine (Jia Liu et al., 2023)

{{<citation>}}

Jia Liu, Jie Shuai. (2023)  
**SMoT: Think in State Machine**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.17445v1)  

---


**ABSTRACT**  
Current prompting approach for language model inference mainly rely on Language Model's (LLM) autonomous exploration of reasoning paths, confronts an inevitable retracing operation when erroneous routes are encountered. This is followed by the pursuit of alternative reasoning paths. However, humans are adept at abstracting optimal solutions from problems, thereby facilitating swift and precise reasoning for similar problems resolution. In light of this, we delves into the potential of harnessing expert knowledge to enhance problem-solving within LLMs. We introduce a novel paradigm, the State Machine of Thought (SMoT), which employs predefined state machines to furnish LLMs with efficient reasoning paths, thereby eliminating fruitless exploration. Furthermore, we propose a multi-agent mechanism that assigns different objectives to agents, aiming to enhance the accuracy of SMoT reasoning. The experimental results, derived from an array reasoning task, reveal that SMoT realizes an extraordinary accuracy of 95\%, surpassing the performance of the state-of-the-art baselines.

{{</citation>}}


## eess.IV (1)



### (53/60) Distribution-based Low-rank Embedding (Bardia Yousefi, 2023)

{{<citation>}}

Bardia Yousefi. (2023)  
**Distribution-based Low-rank Embedding**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.17579v1)  

---


**ABSTRACT**  
The early detection of breast abnormalities is a matter of critical significance. Notably, infrared thermography has emerged as a valuable tool in breast cancer screening and clinical breast examination (CBE). Measuring heterogeneous thermal patterns is the key to incorporating computational dynamic thermography, which can be achieved by matrix factorization techniques. These approaches focus on extracting the predominant thermal patterns from the entire thermal sequence. Yet, the task of singling out the dominant image that effectively represents the prevailing temporal changes remains a challenging pursuit within the field of computational thermography. In this context, we propose applying James-Stein for eigenvector (JSE) and Weibull embedding approaches, as two novel strategies in response to this challenge. The primary objective is to create a low-dimensional (LD) representation of the thermal data stream. This LD approximation serves as the foundation for extracting thermomics and training a classification model with optimized hyperparameters, for early breast cancer detection. Furthermore, we conduct a comparative analysis of various embedding adjuncts to matrix factorization methods. The results of the proposed method indicate an enhancement in the projection of the predominant basis vector, yielding classification accuracy of 81.7% (+/-5.2%) using Weibull embedding, which outperformed other embedding approaches we proposed previously. In comparison analysis, Sparse PCT and Deep SemiNMF showed the highest accuracies having 80.9% and 78.6%, respectively. These findings suggest that JSE and Weibull embedding techniques substantially help preserve crucial thermal patterns as a biomarker leading to improved CBE and enabling the very early detection of breast cancer.

{{</citation>}}


## cs.LO (1)



### (54/60) Higher Order Model Checking in Isabelle for Human Centric Infrastructure Security (Florian Kammüller, 2023)

{{<citation>}}

Florian Kammüller. (2023)  
**Higher Order Model Checking in Isabelle for Human Centric Infrastructure Security**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs.LO  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.17555v1)  

---


**ABSTRACT**  
In this paper we present an efficient approach to implementing model checking in the Higher Order Logic (HOL) of Isabelle. This is a non-trivial task since model checking is restricted to finite state sets. By restricting our scope to considering security attacks, we achieve an efficient executable specification of a model checking algorithm for attack trees. We provide the existing background, the necessary theory and illustrate its application. Theory and application are fully formalized in Isabelle thus providing an executable model checking algorithm.

{{</citation>}}


## cs.AR (1)



### (55/60) Design Space Exploration of Approximate Computing Techniques with a Reinforcement Learning Approach (Sepide Saeedi et al., 2023)

{{<citation>}}

Sepide Saeedi, Alessandro Savino, Stefano Di Carlo. (2023)  
**Design Space Exploration of Approximate Computing Techniques with a Reinforcement Learning Approach**  

---
Primary Category: cs.AR  
Categories: C-1, cs-AR, cs-LG, cs-PF, cs.AR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.17525v1)  

---


**ABSTRACT**  
Approximate Computing (AxC) techniques have become increasingly popular in trading off accuracy for performance gains in various applications. Selecting the best AxC techniques for a given application is challenging. Among proposed approaches for exploring the design space, Machine Learning approaches such as Reinforcement Learning (RL) show promising results. In this paper, we proposed an RL-based multi-objective Design Space Exploration strategy to find the approximate versions of the application that balance accuracy degradation and power and computation time reduction. Our experimental results show a good trade-off between accuracy degradation and decreased power and computation time for some benchmarks.

{{</citation>}}


## eess.AS (1)



### (56/60) Attention-based Interactive Disentangling Network for Instance-level Emotional Voice Conversion (Yun Chen et al., 2023)

{{<citation>}}

Yun Chen, Lingxiao Yang, Qi Chen, Jian-Huang Lai, Xiaohua Xie. (2023)  
**Attention-based Interactive Disentangling Network for Instance-level Emotional Voice Conversion**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-SD, eess-AS, eess.AS  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2312.17508v1)  

---


**ABSTRACT**  
Emotional Voice Conversion aims to manipulate a speech according to a given emotion while preserving non-emotion components. Existing approaches cannot well express fine-grained emotional attributes. In this paper, we propose an Attention-based Interactive diseNtangling Network (AINN) that leverages instance-wise emotional knowledge for voice conversion. We introduce a two-stage pipeline to effectively train our network: Stage I utilizes inter-speech contrastive learning to model fine-grained emotion and intra-speech disentanglement learning to better separate emotion and content. In Stage II, we propose to regularize the conversion with a multi-view consistency mechanism. This technique helps us transfer fine-grained emotion and maintain speech content. Extensive experiments show that our AINN outperforms state-of-the-arts in both objective and subjective metrics.

{{</citation>}}


## q-bio.QM (1)



### (57/60) A graph neural network-based model with Out-of-Distribution Robustness for enhancing Antiretroviral Therapy Outcome Prediction for HIV-1 (Giulia Di Teodoro et al., 2023)

{{<citation>}}

Giulia Di Teodoro, Federico Siciliano, Valerio Guarrasi, Anne-Mieke Vandamme, Valeria Ghisetti, Anders Sönnerborg, Maurizio Zazzi, Fabrizio Silvestri, Laura Palagi. (2023)  
**A graph neural network-based model with Out-of-Distribution Robustness for enhancing Antiretroviral Therapy Outcome Prediction for HIV-1**  

---
Primary Category: q-bio.QM  
Categories: 68, I-2-6, cs-LG, q-bio-QM, q-bio.QM  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2312.17506v1)  

---


**ABSTRACT**  
Predicting the outcome of antiretroviral therapies for HIV-1 is a pressing clinical challenge, especially when the treatment regimen includes drugs for which limited effectiveness data is available. This scarcity of data can arise either due to the introduction of a new drug to the market or due to limited use in clinical settings. To tackle this issue, we introduce a novel joint fusion model, which combines features from a Fully Connected (FC) Neural Network and a Graph Neural Network (GNN). The FC network employs tabular data with a feature vector made up of viral mutations identified in the most recent genotypic resistance test, along with the drugs used in therapy. Conversely, the GNN leverages knowledge derived from Stanford drug-resistance mutation tables, which serve as benchmark references for deducing in-vivo treatment efficacy based on the viral genetic sequence, to build informative graphs. We evaluated these models' robustness against Out-of-Distribution drugs in the test set, with a specific focus on the GNN's role in handling such scenarios. Our comprehensive analysis demonstrates that the proposed model consistently outperforms the FC model, especially when considering Out-of-Distribution drugs. These results underscore the advantage of integrating Stanford scores in the model, thereby enhancing its generalizability and robustness, but also extending its utility in real-world applications with limited data availability. This research highlights the potential of our approach to inform antiretroviral therapy outcome prediction and contribute to more informed clinical decisions.

{{</citation>}}


## cs.SE (1)



### (58/60) The Right Prompts for the Job: Repair Code-Review Defects with Large Language Model (Zelin Zhao et al., 2023)

{{<citation>}}

Zelin Zhao, Zhaogui Xu, Jialong Zhu, Peng Di, Yuan Yao, Xiaoxing Ma. (2023)  
**The Right Prompts for the Job: Repair Code-Review Defects with Large Language Model**  

---
Primary Category: cs.SE  
Categories: 68T01, I-2-5; D-2-0, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.17485v1)  

---


**ABSTRACT**  
Automatic program repair (APR) techniques have the potential to reduce manual efforts in uncovering and repairing program defects during the code review (CR) process. However, the limited accuracy and considerable time costs associated with existing APR approaches hinder their adoption in industrial practice. One key factor is the under-utilization of review comments, which provide valuable insights into defects and potential fixes. Recent advancements in Large Language Models (LLMs) have enhanced their ability to comprehend natural and programming languages, enabling them to generate patches based on review comments. This paper conducts a comprehensive investigation into the effective utilization of LLMs for repairing CR defects. In this study, various prompts are designed and compared across mainstream LLMs using two distinct datasets from human reviewers and automated checkers. Experimental results demonstrate a remarkable repair rate of 72.97% with the best prompt, highlighting a substantial improvement in the effectiveness and practicality of automatic repair techniques.

{{</citation>}}


## cs.DB (1)



### (59/60) DB-GPT: Empowering Database Interactions with Private Large Language Models (Siqiao Xue et al., 2023)

{{<citation>}}

Siqiao Xue, Caigao Jiang, Wenhui Shi, Fangyin Chen, Keting Chen, Hongjun Yang, Zhiping Zhang, Jianshan He, Hongyang Zhang, Ganglin Wei, Wang Zhao, Fan Zhou, Danrui Qi, Hong Yi, Shaodong Liu, Faqiang Chen. (2023)  
**DB-GPT: Empowering Database Interactions with Private Large Language Models**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.17449v1)  

---


**ABSTRACT**  
The recent breakthroughs in large language models (LLMs) are positioned to transition many areas of software. Database technologies particularly have an important entanglement with LLMs as efficient and intuitive database interactions are paramount. In this paper, we present DB-GPT, a revolutionary and production-ready project that integrates LLMs with traditional database systems to enhance user experience and accessibility. DB-GPT is designed to understand natural language queries, provide context-aware responses, and generate complex SQL queries with high accuracy, making it an indispensable tool for users ranging from novice to expert. The core innovation in DB-GPT lies in its private LLM technology, which is fine-tuned on domain-specific corpora to maintain user privacy and ensure data security while offering the benefits of state-of-the-art LLMs. We detail the architecture of DB-GPT, which includes a novel retrieval augmented generation (RAG) knowledge system, an adaptive learning mechanism to continuously improve performance based on user feedback and a service-oriented multi-model framework (SMMF) with powerful data-driven agents. Our extensive experiments and user studies confirm that DB-GPT represents a paradigm shift in database interactions, offering a more natural, efficient, and secure way to engage with data repositories. The paper concludes with a discussion of the implications of DB-GPT framework on the future of human-database interaction and outlines potential avenues for further enhancements and applications in the field. The project code is available at https://github.com/eosphoros-ai/DB-GPT. Experience DB-GPT for yourself by installing it with the instructions https://github.com/eosphoros-ai/DB-GPT#install and view a concise 10-minute video at https://www.youtube.com/watch?v=KYs4nTDzEhk.

{{</citation>}}


## cs.ET (1)



### (60/60) Low Power and Temperature-Resilient Compute-In-Memory Based on Subthreshold-FeFET (Yifei Zhou et al., 2023)

{{<citation>}}

Yifei Zhou, Xuchu Huang, Jianyi Yang, Kai Ni, Hussam Amrouch, Cheng Zhuo, Xunxhao Yin. (2023)  
**Low Power and Temperature-Resilient Compute-In-Memory Based on Subthreshold-FeFET**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs.ET  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.17442v1)  

---


**ABSTRACT**  
Compute-in-memory (CiM) is a promising solution for addressing the challenges of artificial intelligence (AI) and the Internet of Things (IoT) hardware such as 'memory wall' issue. Specifically, CiM employing nonvolatile memory (NVM) devices in a crossbar structure can efficiently accelerate multiply-accumulation (MAC) computation, a crucial operator in neural networks among various AI models. Low power CiM designs are thus highly desired for further energy efficiency optimization on AI models. Ferroelectric FET (FeFET), an emerging device, is attractive for building ultra-low power CiM array due to CMOS compatibility, high ION/IOFF ratio, etc. Recent studies have explored FeFET based CiM designs that achieve low power consumption. Nevertheless, subthreshold-operated FeFETs, where the operating voltages are scaled down to the subthreshold region to reduce array power consumption, are particularly vulnerable to temperature drift, leading to accuracy degradation. To address this challenge, we propose a temperature-resilient 2T-1FeFET CiM design that performs MAC operations reliably at subthreahold region from 0 to 85 Celsius, while consuming ultra-low power. Benchmarked against the VGG neural network architecture running the CIFAR-10 dataset, the proposed 2T-1FeFET CiM design achieves 89.45% CIFAR-10 test accuracy. Compared to previous FeFET based CiM designs, it exhibits immunity to temperature drift at an 8-bit wordlength scale, and achieves better energy efficiency with 2866 TOPS/W.

{{</citation>}}
