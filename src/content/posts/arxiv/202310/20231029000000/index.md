---
draft: false
title: "arXiv @ 2023.10.29"
date: 2023-10-29
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.29"
    identifier: arxiv_20231029
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (12)](#cslg-12)
- [cs.RO (1)](#csro-1)
- [cs.CV (16)](#cscv-16)
- [quant-ph (1)](#quant-ph-1)
- [cs.CL (28)](#cscl-28)
- [cs.CY (2)](#cscy-2)
- [cs.AI (4)](#csai-4)
- [eess.IV (2)](#eessiv-2)
- [math.AP (1)](#mathap-1)
- [cs.HC (2)](#cshc-2)
- [cs.SD (1)](#cssd-1)
- [cs.CR (1)](#cscr-1)
- [cs.NI (1)](#csni-1)
- [cs.DS (1)](#csds-1)
- [cs.IR (1)](#csir-1)
- [cs.SE (1)](#csse-1)
- [cs.IT (1)](#csit-1)
- [stat.ML (1)](#statml-1)
- [cs.GR (1)](#csgr-1)

## cs.LG (12)



### (1/78) FP8-LM: Training FP8 Large Language Models (Houwen Peng et al., 2023)

{{<citation>}}

Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, Ruihang Li, Miaosen Zhang, Chen Li, Jia Ning, Ruizhe Wang, Zheng Zhang, Shuguang Liu, Joe Chau, Han Hu, Peng Cheng. (2023)  
**FP8-LM: Training FP8 Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Azure, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.18313v1)  

---


**ABSTRACT**  
In this paper, we explore FP8 low-bit data formats for efficient training of large language models (LLMs). Our key insight is that most variables, such as gradients and optimizer states, in LLM training can employ low-precision data formats without compromising model accuracy and requiring no changes to hyper-parameters. Specifically, we propose a new FP8 automatic mixed-precision framework for training LLMs. This framework offers three levels of FP8 utilization to streamline mixed-precision and distributed parallel training for LLMs. It gradually incorporates 8-bit gradients, optimizer states, and distributed learning in an incremental manner. Experiment results show that, during the training of GPT-175B model on H100 GPU platform, our FP8 mixed-precision training framework not only achieved a remarkable 42% reduction in real memory usage but also ran 64% faster than the widely adopted BF16 framework (i.e., Megatron-LM), surpassing the speed of Nvidia Transformer Engine by 17%. This largely reduces the training costs for large foundation models. Furthermore, our FP8 mixed-precision training methodology is generic. It can be seamlessly applied to other tasks such as LLM instruction tuning and reinforcement learning with human feedback, offering savings in fine-tuning expenses. Our FP8 low-precision training framework is open-sourced at {https://github.com/Azure/MS-AMP}{aka.ms/MS.AMP}.

{{</citation>}}


### (2/78) Heterogeneous Federated Learning with Group-Aware Prompt Tuning (Wenlong Deng et al., 2023)

{{<citation>}}

Wenlong Deng, Christos Thrampoulidis, Xiaoxiao Li. (2023)  
**Heterogeneous Federated Learning with Group-Aware Prompt Tuning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.18285v1)  

---


**ABSTRACT**  
Transformers have achieved remarkable success in various machine-learning tasks, prompting their widespread adoption. In this paper, we explore their application in the context of federated learning (FL), with a particular focus on heterogeneous scenarios where individual clients possess diverse local datasets. To meet the computational and communication demands of FL, we leverage pre-trained Transformers and use an efficient prompt-tuning strategy. Our strategy introduces the concept of learning both shared and group prompts, enabling the acquisition of universal knowledge and group-specific knowledge simultaneously. Additionally, a prompt selection module assigns personalized group prompts to each input, aligning the global model with the data distribution of each client. This approach allows us to train a single global model that can automatically adapt to various local client data distributions without requiring local fine-tuning. In this way, our proposed method effectively bridges the gap between global and personalized local models in Federated Learning and surpasses alternative approaches that lack the capability to adapt to previously unseen clients. The effectiveness of our approach is rigorously validated through extensive experimentation and ablation studies.

{{</citation>}}


### (3/78) Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-Opt (Yining Ma et al., 2023)

{{<citation>}}

Yining Ma, Zhiguang Cao, Yeow Meng Chee. (2023)  
**Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-Opt**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.18264v1)  

---


**ABSTRACT**  
In this paper, we present Neural k-Opt (NeuOpt), a novel learning-to-search (L2S) solver for routing problems. It learns to perform flexible k-opt exchanges based on a tailored action factorization method and a customized recurrent dual-stream decoder. As a pioneering work to circumvent the pure feasibility masking scheme and enable the autonomous exploration of both feasible and infeasible regions, we then propose the Guided Infeasible Region Exploration (GIRE) scheme, which supplements the NeuOpt policy network with feasibility-related features and leverages reward shaping to steer reinforcement learning more effectively. Additionally, we equip NeuOpt with Dynamic Data Augmentation (D2A) for more diverse searches during inference. Extensive experiments on the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) demonstrate that our NeuOpt not only significantly outstrips existing (masking-based) L2S solvers, but also showcases superiority over the learning-to-construct (L2C) and learning-to-predict (L2P) solvers. Notably, we offer fresh perspectives on how neural solvers can handle VRP constraints. Our code is available: https://github.com/yining043/NeuOpt.

{{</citation>}}


### (4/78) Guided Data Augmentation for Offline Reinforcement Learning and Imitation Learning (Nicholas E. Corrado et al., 2023)

{{<citation>}}

Nicholas E. Corrado, Yuxiao Qu, John U. Balis, Adam Labiosa, Josiah P. Hanna. (2023)  
**Guided Data Augmentation for Offline Reinforcement Learning and Imitation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Augmentation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.18247v1)  

---


**ABSTRACT**  
Learning from demonstration (LfD) is a popular technique that uses expert demonstrations to learn robot control policies. However, the difficulty in acquiring expert-quality demonstrations limits the applicability of LfD methods: real-world data collection is often costly, and the quality of the demonstrations depends greatly on the demonstrator's abilities and safety concerns. A number of works have leveraged data augmentation (DA) to inexpensively generate additional demonstration data, but most DA works generate augmented data in a random fashion and ultimately produce highly suboptimal data. In this work, we propose Guided Data Augmentation (GuDA), a human-guided DA framework that generates expert-quality augmented data. The key insight of GuDA is that while it may be difficult to demonstrate the sequence of actions required to produce expert data, a user can often easily identify when an augmented trajectory segment represents task progress. Thus, the user can impose a series of simple rules on the DA process to automatically generate augmented samples that approximate expert behavior. To extract a policy from GuDA, we use off-the-shelf offline reinforcement learning and behavior cloning algorithms. We evaluate GuDA on a physical robot soccer task as well as simulated D4RL navigation tasks, a simulated autonomous driving task, and a simulated soccer task. Empirically, we find that GuDA enables learning from a small set of potentially suboptimal demonstrations and substantially outperforms a DA strategy that samples augmented data randomly.

{{</citation>}}


### (5/78) Alignment and Outer Shell Isotropy for Hyperbolic Graph Contrastive Learning (Yifei Zhang et al., 2023)

{{<citation>}}

Yifei Zhang, Hao Zhu, Jiahong Liu, Piotr Koniusz, Irwin King. (2023)  
**Alignment and Outer Shell Isotropy for Hyperbolic Graph Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.18209v1)  

---


**ABSTRACT**  
Learning good self-supervised graph representations that are beneficial to downstream tasks is challenging. Among a variety of methods, contrastive learning enjoys competitive performance. The embeddings of contrastive learning are arranged on a hypersphere that enables the Cosine distance measurement in the Euclidean space. However, the underlying structure of many domains such as graphs exhibits highly non-Euclidean latent geometry. To this end, we propose a novel contrastive learning framework to learn high-quality graph embedding. Specifically, we design the alignment metric that effectively captures the hierarchical data-invariant information, as well as we propose a substitute of uniformity metric to prevent the so-called dimensional collapse. We show that in the hyperbolic space one has to address the leaf- and height-level uniformity which are related to properties of trees, whereas in the ambient space of the hyperbolic manifold, these notions translate into imposing an isotropic ring density towards boundaries of Poincar\'e ball. This ring density can be easily imposed by promoting the isotropic feature distribution on the tangent space of manifold. In the experiments, we demonstrate the efficacy of our proposed method across different hyperbolic graph embedding techniques in both supervised and self-supervised learning settings.

{{</citation>}}


### (6/78) Ask more, know better: Reinforce-Learned Prompt Questions for Decision Making with Large Language Models (Xue Yan et al., 2023)

{{<citation>}}

Xue Yan, Yan Song, Xinyu Cui, Filippos Christianos, Haifeng Zhang, David Henry Mguni, Jun Wang. (2023)  
**Ask more, know better: Reinforce-Learned Prompt Questions for Decision Making with Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.18127v1)  

---


**ABSTRACT**  
Large language models (LLMs) demonstrate their promise in tackling complicated practical challenges by combining action-based policies with chain of thought (CoT) reasoning. Having high-quality prompts on hand, however, is vital to the framework's effectiveness. Currently, these prompts are handcrafted utilizing extensive human labor, resulting in CoT policies that frequently fail to generalize. Human intervention is also required in order to develop grounding functions that ensure low-level controllers appropriately process CoT reasoning. In this paper, we take the first step towards a fully integrated end-to-end framework for task-solving in real settings employing complicated reasoning. To that purpose, we offer a new leader-follower bilevel framework capable of learning to ask relevant questions (prompts) and subsequently undertaking reasoning to guide the learning of actions to be performed in an environment. A good prompt should make introspective revisions based on historical findings, leading the CoT to consider the anticipated goals. A prompt-generator policy has its own aim in our system, allowing it to adapt to the action policy and automatically root the CoT process towards outputs that lead to decisive, high-performing actions. Meanwhile, the action policy is learning how to use the CoT outputs to take specific actions. Our empirical data reveal that our system outperforms leading methods in agent learning benchmarks such as Overcooked and FourRoom.

{{</citation>}}


### (7/78) Adversarial Anomaly Detection using Gaussian Priors and Nonlinear Anomaly Scores (Fiete Lüer et al., 2023)

{{<citation>}}

Fiete Lüer, Tobias Weber, Maxim Dolgich, Christian Böhm. (2023)  
**Adversarial Anomaly Detection using Gaussian Priors and Nonlinear Anomaly Scores**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.18091v1)  

---


**ABSTRACT**  
Anomaly detection in imbalanced datasets is a frequent and crucial problem, especially in the medical domain where retrieving and labeling irregularities is often expensive. By combining the generative stability of a $\beta$-variational autoencoder (VAE) with the discriminative strengths of generative adversarial networks (GANs), we propose a novel model, $\beta$-VAEGAN. We investigate methods for composing anomaly scores based on the discriminative and reconstructive capabilities of our model. Existing work focuses on linear combinations of these components to determine if data is anomalous. We advance existing work by training a kernelized support vector machine (SVM) on the respective error components to also consider nonlinear relationships. This improves anomaly detection performance, while allowing faster optimization. Lastly, we use the deviations from the Gaussian prior of $\beta$-VAEGAN to form a novel anomaly score component. In comparison to state-of-the-art work, we improve the $F_1$ score during anomaly detection from 0.85 to 0.92 on the widely used MITBIH Arrhythmia Database.

{{</citation>}}


### (8/78) Unveiling the Potential of Probabilistic Embeddings in Self-Supervised Learning (Denis Janiak et al., 2023)

{{<citation>}}

Denis Janiak, Jakub Binkowski, Piotr Bielak, Tomasz Kajdanowicz. (2023)  
**Unveiling the Potential of Probabilistic Embeddings in Self-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.18080v1)  

---


**ABSTRACT**  
In recent years, self-supervised learning has played a pivotal role in advancing machine learning by allowing models to acquire meaningful representations from unlabeled data. An intriguing research avenue involves developing self-supervised models within an information-theoretic framework, but many studies often deviate from the stochasticity assumptions made when deriving their objectives. To gain deeper insights into this issue, we propose to explicitly model the representation with stochastic embeddings and assess their effects on performance, information compression and potential for out-of-distribution detection. From an information-theoretic perspective, we seek to investigate the impact of probabilistic modeling on the information bottleneck, shedding light on a trade-off between compression and preservation of information in both representation and loss space. Emphasizing the importance of distinguishing between these two spaces, we demonstrate how constraining one can affect the other, potentially leading to performance degradation. Moreover, our findings suggest that introducing an additional bottleneck in the loss space can significantly enhance the ability to detect out-of-distribution examples, only leveraging either representation features or the variance of their underlying distribution.

{{</citation>}}


### (9/78) Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning (Shenzhi Wang et al., 2023)

{{<citation>}}

Shenzhi Wang, Qisen Yang, Jiawei Gao, Matthieu Gaetan Lin, Hao Chen, Liwei Wu, Ning Jia, Shiji Song, Gao Huang. (2023)  
**Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.17966v1)  

---


**ABSTRACT**  
Offline-to-online reinforcement learning (RL) is a training paradigm that combines pre-training on a pre-collected dataset with fine-tuning in an online environment. However, the incorporation of online fine-tuning can intensify the well-known distributional shift problem. Existing solutions tackle this problem by imposing a policy constraint on the policy improvement objective in both offline and online learning. They typically advocate a single balance between policy improvement and constraints across diverse data collections. This one-size-fits-all manner may not optimally leverage each collected sample due to the significant variation in data quality across different states. To this end, we introduce Family Offline-to-Online RL (FamO2O), a simple yet effective framework that empowers existing algorithms to determine state-adaptive improvement-constraint balances. FamO2O utilizes a universal model to train a family of policies with different improvement/constraint intensities, and a balance model to select a suitable policy for each state. Theoretically, we prove that state-adaptive balances are necessary for achieving a higher policy performance upper bound. Empirically, extensive experiments show that FamO2O offers a statistically significant improvement over various existing methods, achieving state-of-the-art performance on the D4RL benchmark. Codes are available at https://github.com/LeapLabTHU/FamO2O.

{{</citation>}}


### (10/78) Improving the Knowledge Gradient Algorithm (Yang Le et al., 2023)

{{<citation>}}

Yang Le, Gao Siyang, Ho Chin Pang. (2023)  
**Improving the Knowledge Gradient Algorithm**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17901v1)  

---


**ABSTRACT**  
The knowledge gradient (KG) algorithm is a popular policy for the best arm identification (BAI) problem. It is built on the simple idea of always choosing the measurement that yields the greatest expected one-step improvement in the estimate of the best mean of the arms. In this research, we show that this policy has limitations, causing the algorithm not asymptotically optimal. We next provide a remedy for it, by following the manner of one-step look ahead of KG, but instead choosing the measurement that yields the greatest one-step improvement in the probability of selecting the best arm. The new policy is called improved knowledge gradient (iKG). iKG can be shown to be asymptotically optimal. In addition, we show that compared to KG, it is easier to extend iKG to variant problems of BAI, with the $\epsilon$-good arm identification and feasible arm identification as two examples. The superior performances of iKG on these problems are further demonstrated using numerical examples.

{{</citation>}}


### (11/78) A Data-Centric Online Market for Machine Learning: From Discovery to Pricing (Minbiao Han et al., 2023)

{{<citation>}}

Minbiao Han, Jonathan Light, Steven Xia, Sainyam Galhotra, Raul Castro Fernandez, Haifeng Xu. (2023)  
**A Data-Centric Online Market for Machine Learning: From Discovery to Pricing**  

---
Primary Category: cs.LG  
Categories: cs-GT, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17843v1)  

---


**ABSTRACT**  
Data fuels machine learning (ML) - rich and high-quality training data is essential to the success of ML. However, to transform ML from the race among a few large corporations to an accessible technology that serves numerous normal users' data analysis requests, there still exist important challenges. One gap we observed is that many ML users can benefit from new data that other data owners possess, whereas these data owners sit on piles of data without knowing who can benefit from it. This gap creates the opportunity for building an online market that can automatically connect supply with demand. While online matching markets are prevalent (e.g., ride-hailing systems), designing a data-centric market for ML exhibits many unprecedented challenges.   This paper develops new techniques to tackle two core challenges in designing such a market: (a) to efficiently match demand with supply, we design an algorithm to automatically discover useful data for any ML task from a pool of thousands of datasets, achieving high-quality matching between ML models and data; (b) to encourage market participation of ML users without much ML expertise, we design a new pricing mechanism for selling data-augmented ML models. Furthermore, our market is designed to be API-compatible with existing online ML markets like Vertex AI and Sagemaker, making it easy to use while providing better results due to joint data and model search. We envision that the synergy of our data and model discovery algorithm and pricing mechanism will be an important step towards building a new data-centric online market that serves ML users effectively.

{{</citation>}}


### (12/78) Positional Encoding-based Resident Identification in Multi-resident Smart Homes (Zhiyi Song et al., 2023)

{{<citation>}}

Zhiyi Song, Dipankar Chaki, Abdallah Lakhdari, Athman Bouguettaya. (2023)  
**Positional Encoding-based Resident Identification in Multi-resident Smart Homes**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.17836v1)  

---


**ABSTRACT**  
We propose a novel resident identification framework to identify residents in a multi-occupant smart environment. The proposed framework employs a feature extraction model based on the concepts of positional encoding. The feature extraction model considers the locations of homes as a graph. We design a novel algorithm to build such graphs from layout maps of smart environments. The Node2Vec algorithm is used to transform the graph into high-dimensional node embeddings. A Long Short-Term Memory (LSTM) model is introduced to predict the identities of residents using temporal sequences of sensor events with the node embeddings. Extensive experiments show that our proposed scheme effectively identifies residents in a multi-occupant environment. Evaluation results on two real-world datasets demonstrate that our proposed approach achieves 94.5% and 87.9% accuracy, respectively.

{{</citation>}}


## cs.RO (1)



### (13/78) Socially Cognizant Robotics for a Technology Enhanced Society (Kristin J. Dana et al., 2023)

{{<citation>}}

Kristin J. Dana, Clinton Andrews, Kostas Bekris, Jacob Feldman, Matthew Stone, Pernille Hemmer, Aaron Mazzeo, Hal Salzman, Jingang Yi. (2023)  
**Socially Cognizant Robotics for a Technology Enhanced Society**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.18303v1)  

---


**ABSTRACT**  
Emerging applications of robotics, and concerns about their impact, require the research community to put human-centric objectives front-and-center. To meet this challenge, we advocate an interdisciplinary approach, socially cognizant robotics, which synthesizes technical and social science methods. We argue that this approach follows from the need to empower stakeholder participation (from synchronous human feedback to asynchronous societal assessment) in shaping AI-driven robot behavior at all levels, and leads to a range of novel research perspectives and problems both for improving robots' interactions with individuals and impacts on society. Drawing on these arguments, we develop best practices for socially cognizant robot design that balance traditional technology-based metrics (e.g. efficiency, precision and accuracy) with critically important, albeit challenging to measure, human and society-based metrics.

{{</citation>}}


## cs.CV (16)



### (14/78) Always Clear Days: Degradation Type and Severity Aware All-In-One Adverse Weather Removal (Yu-Wei Chen et al., 2023)

{{<citation>}}

Yu-Wei Chen, Soo-Chang Pei. (2023)  
**Always Clear Days: Degradation Type and Severity Aware All-In-One Adverse Weather Removal**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.18293v1)  

---


**ABSTRACT**  
All-in-one adverse weather removal is an emerging topic on image restoration, which aims to restore multiple weather degradation in an unified model, and the challenging are twofold. First, discovering and handling the property of multi-domain in target distribution formed by multiple weather conditions. Second, design efficient and effective operations for different degradation types. To address this problem, most prior works focus on the multi-domain caused by weather type. Inspired by inter\&intra-domain adaptation literature, we observed that not only weather type but also weather severity introduce multi-domain within each weather type domain, which is ignored by previous methods, and further limit their performance. To this end, we proposed a degradation type and severity aware model, called \textbf{UtilityIR}, for blind all-in-one bad weather image restoration. To extract weather information from single image, we proposed a novel Marginal Quality Ranking Loss (MQRL) and utilized Contrastive Loss (CL) to guide weather severity and type extraction, and leverage a bag of novel techniques such as Multi-Head Cross Attention (MHCA) and Local-Global Adaptive Instance Normalization (LG-AdaIN) to efficiently restore spatial varying weather degradation. The proposed method can significantly outperform the SOTA methods subjectively and objectively on different weather restoration tasks with a large margin, and enjoy less model parameters. Proposed method even can restore \textbf{unseen} domain combined multiple degradation images, and modulating restoration level. Implementation code will be available at {https://github.com/fordevoted/UtilityIR}{\textit{this repository}}

{{</citation>}}


### (15/78) A Self-Supervised Approach to Land Cover Segmentation (Charles Moore et al., 2023)

{{<citation>}}

Charles Moore, Dakota Hester. (2023)  
**A Self-Supervised Approach to Land Cover Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.18251v1)  

---


**ABSTRACT**  
Land use/land cover change (LULC) maps are integral resources in earth science and agricultural research. Due to the nature of such maps, the creation of LULC maps is often constrained by the time and human resources necessary to accurately annotate satellite imagery and remote sensing data. While computer vision models that perform semantic segmentation to create detailed labels from such data are not uncommon, litle research has been done on self-supervised and unsupervised approaches to labelling LULC maps without the use of ground-truth masks. Here, we demonstrate a self-supervised method of land cover segmentation that has no need for high-quality ground truth labels. The proposed deep learning employs a frozen pre-trained ViT backbone transferred from DINO in a STEGO architecture and is fine-tuned using a custom dataset consisting of very high resolution (VHR) sattelite imagery. After only 10 epochs of fine-tuning, an accuracy of roughly 52% was observed across 5 samples, signifying the feasibility of self-supervised models for the automated labelling of VHR LULC maps.

{{</citation>}}


### (16/78) Generative AI Model for Artistic Style Transfer Using Convolutional Neural Networks (Jonayet Miah et al., 2023)

{{<citation>}}

Jonayet Miah, Duc M Cao, Md Abu Sayed, Md. Sabbirul Haque. (2023)  
**Generative AI Model for Artistic Style Transfer Using Convolutional Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs.CV  
Keywords: AI, Generative AI, Style Transfer  
[Paper Link](http://arxiv.org/abs/2310.18237v1)  

---


**ABSTRACT**  
Artistic style transfer, a captivating application of generative artificial intelligence, involves fusing the content of one image with the artistic style of another to create unique visual compositions. This paper presents a comprehensive overview of a novel technique for style transfer using Convolutional Neural Networks (CNNs). By leveraging deep image representations learned by CNNs, we demonstrate how to separate and manipulate image content and style, enabling the synthesis of high-quality images that combine content and style in a harmonious manner. We describe the methodology, including content and style representations, loss computation, and optimization, and showcase experimental results highlighting the effectiveness and versatility of the approach across different styles and content

{{</citation>}}


### (17/78) Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-Image Generation (Jaemin Cho et al., 2023)

{{<citation>}}

Jaemin Cho, Yushi Hu, Roopal Garg, Peter Anderson, Ranjay Krishna, Jason Baldridge, Mohit Bansal, Jordi Pont-Tuset, Su Wang. (2023)  
**Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.18235v1)  

---


**ABSTRACT**  
Evaluating text-to-image models is notoriously difficult. A strong recent approach for assessing text-image faithfulness is based on QG/A (question generation and answering), which uses pre-trained foundational models to automatically generate a set of questions and answers from the prompt, and output images are scored based on whether these answers extracted with a visual question answering model are consistent with the prompt-based answers. This kind of evaluation is naturally dependent on the quality of the underlying QG and QA models. We identify and address several reliability challenges in existing QG/A work: (a) QG questions should respect the prompt (avoiding hallucinations, duplications, and omissions) and (b) VQA answers should be consistent (not asserting that there is no motorcycle in an image while also claiming the motorcycle is blue). We address these issues with Davidsonian Scene Graph (DSG), an empirically grounded evaluation framework inspired by formal semantics. DSG is an automatic, graph-based QG/A that is modularly implemented to be adaptable to any QG/A module. DSG produces atomic and unique questions organized in dependency graphs, which (i) ensure appropriate semantic coverage and (ii) sidestep inconsistent answers. With extensive experimentation and human evaluation on a range of model configurations (LLM, VQA, and T2I), we empirically demonstrate that DSG addresses the challenges noted above. Finally, we present DSG-1k, an open-sourced evaluation benchmark that includes 1,060 prompts, covering a wide range of fine-grained semantic categories with a balanced distribution. We will release the DSG-1k prompts and the corresponding DSG questions.

{{</citation>}}


### (18/78) Semi-Supervised Panoptic Narrative Grounding (Danni Yang et al., 2023)

{{<citation>}}

Danni Yang, Jiayi Ji, Xiaoshuai Sun, Haowei Wang, Yinan Li, Yiwei Ma, Rongrong Ji. (2023)  
**Semi-Supervised Panoptic Narrative Grounding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.18142v1)  

---


**ABSTRACT**  
Despite considerable progress, the advancement of Panoptic Narrative Grounding (PNG) remains hindered by costly annotations. In this paper, we introduce a novel Semi-Supervised Panoptic Narrative Grounding (SS-PNG) learning scheme, capitalizing on a smaller set of labeled image-text pairs and a larger set of unlabeled pairs to achieve competitive performance. Unlike visual segmentation tasks, PNG involves one pixel belonging to multiple open-ended nouns. As a result, existing multi-class based semi-supervised segmentation frameworks cannot be directly applied to this task. To address this challenge, we first develop a novel SS-PNG Network (SS-PNG-NW) tailored to the SS-PNG setting. We thoroughly investigate strategies such as Burn-In and data augmentation to determine the optimal generic configuration for the SS-PNG-NW. Additionally, to tackle the issue of imbalanced pseudo-label quality, we propose a Quality-Based Loss Adjustment (QLA) approach to adjust the semi-supervised objective, resulting in an enhanced SS-PNG-NW+. Employing our proposed QLA, we improve BCE Loss and Dice loss at pixel and mask levels, respectively. We conduct extensive experiments on PNG datasets, with our SS-PNG-NW+ demonstrating promising results comparable to fully-supervised models across all data ratios. Remarkably, our SS-PNG-NW+ outperforms fully-supervised models with only 30% and 50% supervision data, exceeding their performance by 0.8% and 1.1% respectively. This highlights the effectiveness of our proposed SS-PNG-NW+ in overcoming the challenges posed by limited annotations and enhancing the applicability of PNG tasks. The source code is available at https://github.com/nini0919/SSPNG.

{{</citation>}}


### (19/78) Unsupervised Representation Learning for Diverse Deformable Shape Collections (Sara Hahner et al., 2023)

{{<citation>}}

Sara Hahner, Souhaib Attaiki, Jochen Garcke, Maks Ovsjanikov. (2023)  
**Unsupervised Representation Learning for Diverse Deformable Shape Collections**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.18141v1)  

---


**ABSTRACT**  
We introduce a novel learning-based method for encoding and manipulating 3D surface meshes. Our method is specifically designed to create an interpretable embedding space for deformable shape collections. Unlike previous 3D mesh autoencoders that require meshes to be in a 1-to-1 correspondence, our approach is trained on diverse meshes in an unsupervised manner. Central to our method is a spectral pooling technique that establishes a universal latent space, breaking free from traditional constraints of mesh connectivity and shape categories. The entire process consists of two stages. In the first stage, we employ the functional map paradigm to extract point-to-point (p2p) maps between a collection of shapes in an unsupervised manner. These p2p maps are then utilized to construct a common latent space, which ensures straightforward interpretation and independence from mesh connectivity and shape category. Through extensive experiments, we demonstrate that our method achieves excellent reconstructions and produces more realistic and smoother interpolations than baseline approaches.

{{</citation>}}


### (20/78) ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Real Image (Kyle Sargent et al., 2023)

{{<citation>}}

Kyle Sargent, Zizhang Li, Tanmay Shah, Charles Herrmann, Hong-Xing Yu, Yunzhi Zhang, Eric Ryan Chan, Dmitry Lagun, Li Fei-Fei, Deqing Sun, Jiajun Wu. (2023)  
**ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Real Image**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.17994v1)  

---


**ABSTRACT**  
We introduce a 3D-aware diffusion model, ZeroNVS, for single-image novel view synthesis for in-the-wild scenes. While existing methods are designed for single objects with masked backgrounds, we propose new techniques to address challenges introduced by in-the-wild multi-object scenes with complex backgrounds. Specifically, we train a generative prior on a mixture of data sources that capture object-centric, indoor, and outdoor scenes. To address issues from data mixture such as depth-scale ambiguity, we propose a novel camera conditioning parameterization and normalization scheme. Further, we observe that Score Distillation Sampling (SDS) tends to truncate the distribution of complex backgrounds during distillation of 360-degree scenes, and propose "SDS anchoring" to improve the diversity of synthesized novel views. Our model sets a new state-of-the-art result in LPIPS on the DTU dataset in the zero-shot setting, even outperforming methods specifically trained on DTU. We further adapt the challenging Mip-NeRF 360 dataset as a new benchmark for single-image novel view synthesis, and demonstrate strong performance in this setting. Our code and data are at http://kylesargent.github.io/zeronvs/

{{</citation>}}


### (21/78) FaultSeg Swin-UNETR: Transformer-Based Self-Supervised Pretraining Model for Fault Recognition (Zeren Zhang et al., 2023)

{{<citation>}}

Zeren Zhang, Ran Chen, Jinwen Ma. (2023)  
**FaultSeg Swin-UNETR: Transformer-Based Self-Supervised Pretraining Model for Fault Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2310.17974v1)  

---


**ABSTRACT**  
This paper introduces an approach to enhance seismic fault recognition through self-supervised pretraining. Seismic fault interpretation holds great significance in the fields of geophysics and geology. However, conventional methods for seismic fault recognition encounter various issues, including dependence on data quality and quantity, as well as susceptibility to interpreter subjectivity. Currently, automated fault recognition methods proposed based on small synthetic datasets experience performance degradation when applied to actual seismic data. To address these challenges, we have introduced the concept of self-supervised learning, utilizing a substantial amount of relatively easily obtainable unlabeled seismic data for pretraining. Specifically, we have employed the Swin Transformer model as the core network and employed the SimMIM pretraining task to capture unique features related to discontinuities in seismic data. During the fine-tuning phase, inspired by edge detection techniques, we have also refined the structure of the Swin-UNETR model, enabling multiscale decoding and fusion for more effective fault detection. Experimental results demonstrate that our proposed method attains state-of-the-art performance on the Thebe dataset, as measured by the OIS and ODS metrics.

{{</citation>}}


### (22/78) Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare (Junling Liu et al., 2023)

{{<citation>}}

Junling Liu, Ziming Wang, Qichen Ye, Dading Chong, Peilin Zhou, Yining Hua. (2023)  
**Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.17956v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have introduced a new era of proficiency in comprehending complex healthcare and biomedical topics. However, there is a noticeable lack of models in languages other than English and models that can interpret multi-modal input, which is crucial for global healthcare accessibility. In response, this study introduces Qilin-Med-VL, the first Chinese large vision-language model designed to integrate the analysis of textual and visual data. Qilin-Med-VL combines a pre-trained Vision Transformer (ViT) with a foundational LLM. It undergoes a thorough two-stage curriculum training process that includes feature alignment and instruction tuning. This method enhances the model's ability to generate medical captions and answer complex medical queries. We also release ChiMed-VL, a dataset consisting of more than 1M image-text pairs. This dataset has been carefully curated to enable detailed and comprehensive interpretation of medical data using various types of images.

{{</citation>}}


### (23/78) Shape-centered Representation Learning for Visible-Infrared Person Re-identification (Shuang Li et al., 2023)

{{<citation>}}

Shuang Li, Jiaxu Leng, Ji Gan, Mengjingcheng Mo, Xinbo Gao. (2023)  
**Shape-centered Representation Learning for Visible-Infrared Person Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.17952v1)  

---


**ABSTRACT**  
Current Visible-Infrared Person Re-Identification (VI-ReID) methods prioritize extracting distinguishing appearance features, ignoring the natural resistance of body shape against modality changes. Initially, we gauged the discriminative potential of shapes by a straightforward concatenation of shape and appearance features. However, two unresolved issues persist in the utilization of shape features. One pertains to the dependence on auxiliary models for shape feature extraction in the inference phase, along with the errors in generated infrared shapes due to the intrinsic modality disparity. The other issue involves the inadequately explored correlation between shape and appearance features. To tackle the aforementioned challenges, we propose the Shape-centered Representation Learning framework (ScRL), which focuses on learning shape features and appearance features associated with shapes. Specifically, we devise the Shape Feature Propagation (SFP), facilitating direct extraction of shape features from original images with minimal complexity costs during inference. To restitute inaccuracies in infrared body shapes at the feature level, we present the Infrared Shape Restitution (ISR). Furthermore, to acquire appearance features related to shape, we design the Appearance Feature Enhancement (AFE), which accentuates identity-related features while suppressing identity-unrelated features guided by shape features. Extensive experiments are conducted to validate the effectiveness of the proposed ScRL. Achieving remarkable results, the Rank-1 (mAP) accuracy attains 76.1%, 71.2%, 92.4% (72.6%, 52.9%, 86.7%) on the SYSU-MM01, HITSZ-VCM, RegDB datasets respectively, outperforming existing state-of-the-art methods.

{{</citation>}}


### (24/78) Understanding Parameter Saliency via Extreme Value Theory (Shuo Wang et al., 2023)

{{<citation>}}

Shuo Wang, Issei Sato. (2023)  
**Understanding Parameter Saliency via Extreme Value Theory**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.17951v1)  

---


**ABSTRACT**  
Deep neural networks are being increasingly implemented throughout society in recent years. It is useful to identify which parameters trigger misclassification in diagnosing undesirable model behaviors. The concept of parameter saliency is proposed and used to diagnose convolutional neural networks (CNNs) by ranking convolution filters that may have caused misclassification on the basis of parameter saliency. It is also shown that fine-tuning the top ranking salient filters has efficiently corrected misidentification on ImageNet. However, there is still a knowledge gap in terms of understanding why parameter saliency ranking can find the filters inducing misidentification. In this work, we attempt to bridge the gap by analyzing parameter saliency ranking from a statistical viewpoint, namely, extreme value theory. We first show that the existing work implicitly assumes that the gradient norm computed for each filter follows a normal distribution. Then, we clarify the relationship between parameter saliency and the score based on the peaks-over-threshold (POT) method, which is often used to model extreme values. Finally, we reformulate parameter saliency in terms of the POT method, where this reformulation is regarded as statistical anomaly detection and does not require the implicit assumptions of the existing parameter-saliency formulation. Our experimental results demonstrate that our reformulation can detect malicious filters as well. Furthermore, we show that the existing parameter saliency method exhibits a bias against the depth of layers in deep neural networks. In particular, this bias has the potential to inhibit the discovery of filters that cause misidentification in situations where domain shift occurs. In contrast, parameter saliency based on POT shows less of this bias.

{{</citation>}}


### (25/78) Instance Segmentation under Occlusions via Location-aware Copy-Paste Data Augmentation (Son Nguyen et al., 2023)

{{<citation>}}

Son Nguyen, Mikel Lainsa, Hung Dao, Daeyoung Kim, Giang Nguyen. (2023)  
**Instance Segmentation under Occlusions via Location-aware Copy-Paste Data Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.17949v1)  

---


**ABSTRACT**  
Occlusion is a long-standing problem in computer vision, particularly in instance segmentation. ACM MMSports 2023 DeepSportRadar has introduced a dataset that focuses on segmenting human subjects within a basketball context and a specialized evaluation metric for occlusion scenarios. Given the modest size of the dataset and the highly deformable nature of the objects to be segmented, this challenge demands the application of robust data augmentation techniques and wisely-chosen deep learning architectures. Our work (ranked 1st in the competition) first proposes a novel data augmentation technique, capable of generating more training samples with wider distribution. Then, we adopt a new architecture - Hybrid Task Cascade (HTC) framework with CBNetV2 as backbone and MaskIoU head to improve segmentation performance. Furthermore, we employ a Stochastic Weight Averaging (SWA) training strategy to improve the model's generalization. As a result, we achieve a remarkable occlusion score (OM) of 0.533 on the challenge dataset, securing the top-1 position on the leaderboard. Source code is available at this https://github.com/nguyendinhson-kaist/MMSports23-Seg-AutoID.

{{</citation>}}


### (26/78) 3D-Aware Visual Question Answering about Parts, Poses and Occlusions (Xingrui Wang et al., 2023)

{{<citation>}}

Xingrui Wang, Wufei Ma, Zhuowan Li, Adam Kortylewski, Alan Yuille. (2023)  
**3D-Aware Visual Question Answering about Parts, Poses and Occlusions**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.17914v1)  

---


**ABSTRACT**  
Despite rapid progress in Visual question answering (VQA), existing datasets and models mainly focus on testing reasoning in 2D. However, it is important that VQA models also understand the 3D structure of visual scenes, for example to support tasks like navigation or manipulation. This includes an understanding of the 3D object pose, their parts and occlusions. In this work, we introduce the task of 3D-aware VQA, which focuses on challenging questions that require a compositional reasoning over the 3D structure of visual scenes. We address 3D-aware VQA from both the dataset and the model perspective. First, we introduce Super-CLEVR-3D, a compositional reasoning dataset that contains questions about object parts, their 3D poses, and occlusions. Second, we propose PO3D-VQA, a 3D-aware VQA model that marries two powerful ideas: probabilistic neural symbolic program execution for reasoning and deep neural networks with 3D generative representations of objects for robust visual recognition. Our experimental results show our model PO3D-VQA outperforms existing methods significantly, but we still observe a significant performance gap compared to 2D VQA benchmarks, indicating that 3D-aware VQA remains an important open research area.

{{</citation>}}


### (27/78) SmooSeg: Smoothness Prior for Unsupervised Semantic Segmentation (Mengcheng Lan et al., 2023)

{{<citation>}}

Mengcheng Lan, Xinjiang Wang, Yiping Ke, Jiaxing Xu, Litong Feng, Wayne Zhang. (2023)  
**SmooSeg: Smoothness Prior for Unsupervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.17874v1)  

---


**ABSTRACT**  
Unsupervised semantic segmentation is a challenging task that segments images into semantic groups without manual annotation. Prior works have primarily focused on leveraging prior knowledge of semantic consistency or priori concepts from self-supervised learning methods, which often overlook the coherence property of image segments. In this paper, we demonstrate that the smoothness prior, asserting that close features in a metric space share the same semantics, can significantly simplify segmentation by casting unsupervised semantic segmentation as an energy minimization problem. Under this paradigm, we propose a novel approach called SmooSeg that harnesses self-supervised learning methods to model the closeness relationships among observations as smoothness signals. To effectively discover coherent semantic segments, we introduce a novel smoothness loss that promotes piecewise smoothness within segments while preserving discontinuities across different segments. Additionally, to further enhance segmentation quality, we design an asymmetric teacher-student style predictor that generates smoothly updated pseudo labels, facilitating an optimal fit between observations and labeling outputs. Thanks to the rich supervision cues of the smoothness prior, our SmooSeg significantly outperforms STEGO in terms of pixel accuracy on three datasets: COCOStuff (+14.9%), Cityscapes (+13.0%), and Potsdam-3 (+5.7%).

{{</citation>}}


### (28/78) Grid Jigsaw Representation with CLIP: A New Perspective on Image Clustering (Zijie Song et al., 2023)

{{<citation>}}

Zijie Song, Zhenzhen Hu, Richang Hong. (2023)  
**Grid Jigsaw Representation with CLIP: A New Perspective on Image Clustering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.17869v1)  

---


**ABSTRACT**  
Unsupervised representation learning for image clustering is essential in computer vision. Although the advancement of visual models has improved image clustering with efficient visual representations, challenges still remain. Firstly, these features often lack the ability to represent the internal structure of images, hindering the accurate clustering of visually similar images. Secondly, the existing features tend to lack finer-grained semantic labels, limiting the ability to capture nuanced differences and similarities between images.   In this paper, we first introduce Jigsaw based strategy method for image clustering called Grid Jigsaw Representation (GJR) with systematic exposition from pixel to feature in discrepancy against human and computer. We emphasize that this algorithm, which mimics human jigsaw puzzle, can effectively improve the model to distinguish the spatial feature between different samples and enhance the clustering ability. GJR modules are appended to a variety of deep convolutional networks and tested with significant improvements on a wide range of benchmark datasets including CIFAR-10, CIFAR-100/20, STL-10, ImageNet-10 and ImageNetDog-15.   On the other hand, convergence efficiency is always an important challenge for unsupervised image clustering. Recently, pretrained representation learning has made great progress and released models can extract mature visual representations. It is obvious that use the pretrained model as feature extractor can speed up the convergence of clustering where our aim is to provide new perspective in image clustering with reasonable resource application and provide new baseline. Further, we innovate pretrain-based Grid Jigsaw Representation (pGJR) with improvement by GJR. The experiment results show the effectiveness on the clustering task with respect to the ACC, NMI and ARI three metrics and super fast convergence speed.

{{</citation>}}


### (29/78) What You See Is What You Detect: Towards better Object Densification in 3D detection (Tianran Liu et al., 2023)

{{<citation>}}

Tianran Liu, Zeping Zhang Morteza Mousa Pasandi, Robert Laganiere. (2023)  
**What You See Is What You Detect: Towards better Object Densification in 3D detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.17842v1)  

---


**ABSTRACT**  
Recent works have demonstrated the importance of object completion in 3D Perception from Lidar signal. Several methods have been proposed in which modules were used to densify the point clouds produced by laser scanners, leading to better recall and more accurate results. Pursuing in that direction, we present, in this work, a counter-intuitive perspective: the widely-used full-shape completion approach actually leads to a higher error-upper bound especially for far away objects and small objects like pedestrians. Based on this observation, we introduce a visible part completion method that requires only 11.3\% of the prediction points that previous methods generate. To recover the dense representation, we propose a mesh-deformation-based method to augment the point set associated with visible foreground objects. Considering that our approach focuses only on the visible part of the foreground objects to achieve accurate 3D detection, we named our method What You See Is What You Detect (WYSIWYD). Our proposed method is thus a detector-independent model that consists of 2 parts: an Intra-Frustum Segmentation Transformer (IFST) and a Mesh Depth Completion Network(MDCNet) that predicts the foreground depth from mesh deformation. This way, our model does not require the time-consuming full-depth completion task used by most pseudo-lidar-based methods. Our experimental evaluation shows that our approach can provide up to 12.2\% performance improvements over most of the public baseline models on the KITTI and NuScenes dataset bringing the state-of-the-art to a new level. The codes will be available at \textcolor[RGB]{0,0,255}{\url{{https://github.com/Orbis36/WYSIWYD}}

{{</citation>}}


## quant-ph (1)



### (30/78) Exploring Non-Linear Programming Formulations in QuantumCircuitOpt for Optimal Circuit Design (Elena R. Henderson et al., 2023)

{{<citation>}}

Elena R. Henderson, Harsha Nagarajan, Carleton Coffrin. (2023)  
**Exploring Non-Linear Programming Formulations in QuantumCircuitOpt for Optimal Circuit Design**  

---
Primary Category: quant-ph  
Categories: cs-SY, eess-SY, quant-ph, quant-ph  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.18281v1)  

---


**ABSTRACT**  
Given the limitations of current hardware, the theoretical gains promised by quantum computing remain unrealized across practical applications. But the gap between theory and hardware is closing, assisted by developments in quantum algorithmic modeling. One such recent development is QuantumCircuitOpt (QCOpt), an open-source software framework that leverages state-of-the-art optimization-based solvers to find provably optimal compact circuit decompositions, which are exact up to global phase and machine precision. The quantum circuit design problem can be modeled using non-linear, non-convex constraints. However, QCOpt reformulates these non-linear constraints using well-known linearization techniques such that the resulting design problem is solved as a Mixed-Integer Linear Programming (MILP) model. In this work, we instead explore whether the QCOpt could also be effective with a continuous Non-Linear Programming (NLP) model obtained via relaxation of the integer variables in the non-linear constraints. We are able to present not only multiple significant enhancements to QCOpt, with up to 11.3x speed-up in run times on average, but also opportunities for more generally exploring the behavior of gradient-based NLP solvers.

{{</citation>}}


## cs.CL (28)



### (31/78) MalFake: A Multimodal Fake News Identification for Malayalam using Recurrent Neural Networks and VGG-16 (Adhish S. Sujan et al., 2023)

{{<citation>}}

Adhish S. Sujan, Ajitha. V, Aleena Benny, Amiya M. P., V. S. Anoop. (2023)  
**MalFake: A Multimodal Fake News Identification for Malayalam using Recurrent Neural Networks and VGG-16**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2310.18263v1)  

---


**ABSTRACT**  
The amount of news being consumed online has substantially expanded in recent years. Fake news has become increasingly common, especially in regional languages like Malayalam, due to the rapid publication and lack of editorial standards on some online sites. Fake news may have a terrible effect on society, causing people to make bad judgments, lose faith in authorities, and even engage in violent behavior. When we take into the context of India, there are many regional languages, and fake news is spreading in every language. Therefore, providing efficient techniques for identifying false information in regional tongues is crucial. Until now, little to no work has been done in Malayalam, extracting features from multiple modalities to classify fake news. Multimodal approaches are more accurate in detecting fake news, as features from multiple modalities are extracted to build the deep learning classification model. As far as we know, this is the first piece of work in Malayalam that uses multimodal deep learning to tackle false information. Models trained with more than one modality typically outperform models taught with only one modality. Our study in the Malayalam language utilizing multimodal deep learning is a significant step toward more effective misinformation detection and mitigation.

{{</citation>}}


### (32/78) Revising with a Backward Glance: Regressions and Skips during Reading as Cognitive Signals for Revision Policies in Incremental Processing (Brielen Madureira et al., 2023)

{{<citation>}}

Brielen Madureira, Pelin Çelikkol, David Schlangen. (2023)  
**Revising with a Backward Glance: Regressions and Skips during Reading as Cognitive Signals for Revision Policies in Incremental Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LSTM, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2310.18229v1)  

---


**ABSTRACT**  
In NLP, incremental processors produce output in instalments, based on incoming prefixes of the linguistic input. Some tokens trigger revisions, causing edits to the output hypothesis, but little is known about why models revise when they revise. A policy that detects the time steps where revisions should happen can improve efficiency. Still, retrieving a suitable signal to train a revision policy is an open problem, since it is not naturally available in datasets. In this work, we investigate the appropriateness of regressions and skips in human reading eye-tracking data as signals to inform revision policies in incremental sequence labelling. Using generalised mixed-effects models, we find that the probability of regressions and skips by humans can potentially serve as useful predictors for revisions in BiLSTMs and Transformer models, with consistent results for various languages.

{{</citation>}}


### (33/78) ArcheType: A Novel Framework for Open-Source Column Type Annotation using Large Language Models (Benjamin Feuer et al., 2023)

{{<citation>}}

Benjamin Feuer, Yurong Liu, Chinmay Hegde, Juliana Freire. (2023)  
**ArcheType: A Novel Framework for Open-Source Column Type Annotation using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.18208v1)  

---


**ABSTRACT**  
Existing deep-learning approaches to semantic column type annotation (CTA) have important shortcomings: they rely on semantic types which are fixed at training time; require a large number of training samples per type and incur large run-time inference costs; and their performance can degrade when evaluated on novel datasets, even when types remain constant. Large language models have exhibited strong zero-shot classification performance on a wide range of tasks and in this paper we explore their use for CTA. We introduce ArcheType, a simple, practical method for context sampling, prompt serialization, model querying, and label remapping, which enables large language models to solve column type annotation problems in a fully zero-shot manner. We ablate each component of our method separately, and establish that improvements to context sampling and label remapping provide the most consistent gains. ArcheType establishes new state-of-the-art performance on both zero-shot and fine-tuned CTA, including three new domain-specific benchmarks, which we release, along with the code to reproduce our results at https://github.com/penfever/ArcheType.

{{</citation>}}


### (34/78) INA: An Integrative Approach for Enhancing Negotiation Strategies with Reward-Based Dialogue System (Zishan Ahmad et al., 2023)

{{<citation>}}

Zishan Ahmad, Suman Saurabh, Vaishakh Sreekanth Menon, Asif Ekbal, Roshni Ramnani, Anutosh Maitra. (2023)  
**INA: An Integrative Approach for Enhancing Negotiation Strategies with Reward-Based Dialogue System**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, GPT  
[Paper Link](http://arxiv.org/abs/2310.18207v1)  

---


**ABSTRACT**  
In this paper, we propose a novel negotiation dialogue agent designed for the online marketplace. Our agent is integrative in nature i.e, it possesses the capability to negotiate on price as well as other factors, such as the addition or removal of items from a deal bundle, thereby offering a more flexible and comprehensive negotiation experience. We create a new dataset called Integrative Negotiation Dataset (IND) to enable this functionality. For this dataset creation, we introduce a new semi-automated data creation method, which combines defining negotiation intents, actions, and intent-action simulation between users and the agent to generate potential dialogue flows. Finally, the prompting of GPT-J, a state-of-the-art language model, is done to generate dialogues for a given intent, with a human-in-the-loop process for post-editing and refining minor errors to ensure high data quality. We employ a set of novel rewards, specifically tailored for the negotiation task to train our Negotiation Agent, termed as the Integrative Negotiation Agent (INA). These rewards incentivize the chatbot to learn effective negotiation strategies that can adapt to various contextual requirements and price proposals. By leveraging the IND, we train our model and conduct experiments to evaluate the effectiveness of our reward-based dialogue system for negotiation. Our results demonstrate that the proposed approach and reward system significantly enhance the agent's negotiation capabilities. The INA successfully engages in integrative negotiations, displaying the ability to dynamically adjust prices and negotiate the inclusion or exclusion of items in a bundle deal

{{</citation>}}


### (35/78) Lost in Translation, Found in Spans: Identifying Claims in Multilingual Social Media (Shubham Mittal et al., 2023)

{{<citation>}}

Shubham Mittal, Megha Sundriyal, Preslav Nakov. (2023)  
**Lost in Translation, Found in Spans: Identifying Claims in Multilingual Social Media**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, Multilingual, Social Media  
[Paper Link](http://arxiv.org/abs/2310.18205v1)  

---


**ABSTRACT**  
Claim span identification (CSI) is an important step in fact-checking pipelines, aiming to identify text segments that contain a checkworthy claim or assertion in a social media post. Despite its importance to journalists and human fact-checkers, it remains a severely understudied problem, and the scarce research on this topic so far has only focused on English. Here we aim to bridge this gap by creating a novel dataset, X-CLAIM, consisting of 7K real-world claims collected from numerous social media platforms in five Indian languages and English. We report strong baselines with state-of-the-art encoder-only language models (e.g., XLM-R) and we demonstrate the benefits of training on multiple languages over alternative cross-lingual transfer methods such as zero-shot transfer, or training on translated data, from a high-resource language such as English. We evaluate generative large language models from the GPT series using prompting methods on the X-CLAIM dataset and we find that they underperform the smaller encoder-only language models for low-resource languages.

{{</citation>}}


### (36/78) Personas as a Way to Model Truthfulness in Language Models (Nitish Joishi et al., 2023)

{{<citation>}}

Nitish Joishi, Javier Rando, Abulhair Saparov, Najoung Kim, He He. (2023)  
**Personas as a Way to Model Truthfulness in Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.18168v1)  

---


**ABSTRACT**  
Large Language Models are trained on vast amounts of text from the internet, which contains both factual and misleading information about the world. Can language models discern truth from falsehood in this contradicting data? Expanding on the view that LLMs can model different agents producing the corpora, we hypothesize that they can cluster truthful text by modeling a truthful persona: a group of agents that are likely to produce truthful text and share similar features. For example, trustworthy sources like Wikipedia and Science usually use formal writing styles and make consistent claims. By modeling this persona, LLMs can generalize truthfulness beyond the specific contexts in which each agent generated the training text. For example, the model can infer that the agent "Wikipedia" will behave truthfully on topics that were only generated by "Science" because they share a persona. We first show evidence for the persona hypothesis via two observations: (1) we can probe whether a model's answer will be truthful before it is generated; (2) finetuning a model on a set of facts improves its truthfulness on unseen topics. Next, using arithmetics as a synthetic environment, we show that language models can separate true and false statements, and generalize truthfulness across agents; but only if agents in the training data share a truthful generative process that enables the creation of a truthful persona. Overall, our findings suggest that models can exploit hierarchical structures in the data to learn abstract concepts like truthfulness.

{{</citation>}}


### (37/78) MPrompt: Exploring Multi-level Prompt Tuning for Machine Reading Comprehension (Guoxin Chen et al., 2023)

{{<citation>}}

Guoxin Chen, Yiming Qian, Bowen Wang, Liangzhi Li. (2023)  
**MPrompt: Exploring Multi-level Prompt Tuning for Machine Reading Comprehension**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Reading Comprehension, QA  
[Paper Link](http://arxiv.org/abs/2310.18167v1)  

---


**ABSTRACT**  
The large language models have achieved superior performance on various natural language tasks. One major drawback of such approaches is they are resource-intensive in fine-tuning new datasets. Soft-prompt tuning presents a resource-efficient solution to fine-tune the pre-trained language models (PLMs) while keeping their weight frozen. Existing soft prompt methods mainly focus on designing the input-independent prompts that steer the model to fit the domain of the new dataset. Those methods often ignore the fine-grained information about the task and context of the text. In this paper, we propose a multi-level prompt tuning (MPrompt) method for machine reading comprehension. It utilizes prompts at task-specific, domain-specific, and context-specific levels to enhance the comprehension of input semantics at different granularities. We also propose an independence constraint to steer each domain-specific prompt to focus on information within its domain to avoid redundancy. Moreover, we present a prompt generator that incorporates context-related knowledge in the prompt generation to enhance contextual relevancy. We conducted extensive experiments on 12 benchmarks of various QA formats and achieved an average improvement of 1.94\% over the state-of-the-art methods.

{{</citation>}}


### (38/78) Disentangled Representation Learning with Large Language Models for Text-Attributed Graphs (Yijian Qin et al., 2023)

{{<citation>}}

Yijian Qin, Xin Wang, Ziwei Zhang, Wenwu Zhu. (2023)  
**Disentangled Representation Learning with Large Language Models for Text-Attributed Graphs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GNN, Language Model, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.18152v1)  

---


**ABSTRACT**  
Text-attributed graphs (TAGs) are prevalent on the web and research over TAGs such as citation networks, e-commerce networks and social networks has attracted considerable attention in the web community. Recently, large language models (LLMs) have demonstrated exceptional capabilities across a wide range of tasks. However, the existing works focus on harnessing the potential of LLMs solely relying on prompts to convey graph structure information to LLMs, thus suffering from insufficient understanding of the complex structural relationships within TAGs. To address this problem, in this paper we present the Disentangled Graph-Text Learner (DGTL) model, which is able to enhance the reasoning and predicting capabilities of LLMs for TAGs. Our proposed DGTL model incorporates graph structure information through tailored disentangled graph neural network (GNN) layers, enabling LLMs to capture the intricate relationships hidden in text-attributed graphs from multiple structural factors. Furthermore, DGTL operates with frozen pre-trained LLMs, reducing computational costs and allowing much more flexibility in combining with different LLM models. Experimental evaluations demonstrate the effectiveness of the proposed DGTL model on achieving superior or comparable performance over state-of-the-art baselines. Additionally, we also demonstrate that our DGTL model can offer natural language explanations for predictions, thereby significantly enhancing model interpretability.

{{</citation>}}


### (39/78) OpinSummEval: Revisiting Automated Evaluation for Opinion Summarization (Yuchen Shen et al., 2023)

{{<citation>}}

Yuchen Shen, Xiaojun Wan. (2023)  
**OpinSummEval: Revisiting Automated Evaluation for Opinion Summarization**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Summarization  
[Paper Link](http://arxiv.org/abs/2310.18122v1)  

---


**ABSTRACT**  
Opinion summarization sets itself apart from other types of summarization tasks due to its distinctive focus on aspects and sentiments. Although certain automated evaluation methods like ROUGE have gained popularity, we have found them to be unreliable measures for assessing the quality of opinion summaries. In this paper, we present OpinSummEval, a dataset comprising human judgments and outputs from 14 opinion summarization models. We further explore the correlation between 24 automatic metrics and human ratings across four dimensions. Our findings indicate that metrics based on neural networks generally outperform non-neural ones. However, even metrics built on powerful backbones, such as BART and GPT-3/3.5, do not consistently correlate well across all dimensions, highlighting the need for advancements in automated evaluation methods for opinion summarization. The code and data are publicly available at https://github.com/A-Chicharito-S/OpinSummEval/tree/main.

{{</citation>}}


### (40/78) Towards a Unified Conversational Recommendation System: Multi-task Learning via Contextualized Knowledge Distillation (Yeongseo Jung et al., 2023)

{{<citation>}}

Yeongseo Jung, Eunseo Jung, Lei Chen. (2023)  
**Towards a Unified Conversational Recommendation System: Multi-task Learning via Contextualized Knowledge Distillation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Conversational Recommendation, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.18119v1)  

---


**ABSTRACT**  
In Conversational Recommendation System (CRS), an agent is asked to recommend a set of items to users within natural language conversations. To address the need for both conversational capability and personalized recommendations, prior works have utilized separate recommendation and dialogue modules. However, such approach inevitably results in a discrepancy between recommendation results and generated responses. To bridge the gap, we propose a multi-task learning for a unified CRS, where a single model jointly learns both tasks via Contextualized Knowledge Distillation (ConKD). We introduce two versions of ConKD: hard gate and soft gate. The former selectively gates between two task-specific teachers, while the latter integrates knowledge from both teachers. Our gates are computed on-the-fly in a context-specific manner, facilitating flexible integration of relevant knowledge. Extensive experiments demonstrate that our single model significantly improves recommendation performance while enhancing fluency, and achieves comparable results in terms of diversity.

{{</citation>}}


### (41/78) Lost in Translation -- Multilingual Misinformation and its Evolution (Dorian Quelle et al., 2023)

{{<citation>}}

Dorian Quelle, Calvin Cheng, Alexandre Bovet, Scott A. Hale. (2023)  
**Lost in Translation -- Multilingual Misinformation and its Evolution**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-SI, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2310.18089v1)  

---


**ABSTRACT**  
Misinformation and disinformation are growing threats in the digital age, spreading rapidly across languages and borders. This paper investigates the prevalence and dynamics of multilingual misinformation through an analysis of over 250,000 unique fact-checks spanning 95 languages. First, we find that while the majority of misinformation claims are only fact-checked once, 11.7%, corresponding to more than 21,000 claims, are checked multiple times. Using fact-checks as a proxy for the spread of misinformation, we find 33% of repeated claims cross linguistic boundaries, suggesting that some misinformation permeates language barriers. However, spreading patterns exhibit strong homophily, with misinformation more likely to spread within the same language. To study the evolution of claims over time and mutations across languages, we represent fact-checks with multilingual sentence embeddings and cluster semantically similar claims. We analyze the connected components and shortest paths connecting different versions of a claim finding that claims gradually drift over time and undergo greater alteration when traversing languages. Overall, this novel investigation of multilingual misinformation provides key insights. It quantifies redundant fact-checking efforts, establishes that some claims diffuse across languages, measures linguistic homophily, and models the temporal and cross-lingual evolution of claims. The findings advocate for expanded information sharing between fact-checkers globally while underscoring the importance of localized verification.

{{</citation>}}


### (42/78) Detrimental Contexts in Open-Domain Question Answering (Philhoon Oh et al., 2023)

{{<citation>}}

Philhoon Oh, James Thorne. (2023)  
**Detrimental Contexts in Open-Domain Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.18077v1)  

---


**ABSTRACT**  
For knowledge intensive NLP tasks, it has been widely accepted that accessing more information is a contributing factor to improvements in the model's end-to-end performance. However, counter-intuitively, too much context can have a negative impact on the model when evaluated on common question answering (QA) datasets. In this paper, we analyze how passages can have a detrimental effect on retrieve-then-read architectures used in question answering. Our empirical evidence indicates that the current read architecture does not fully leverage the retrieved passages and significantly degrades its performance when using the whole passages compared to utilizing subsets of them. Our findings demonstrate that model accuracy can be improved by 10% on two popular QA datasets by filtering out detrimental passages. Additionally, these outcomes are attained by utilizing existing retrieval methods without further training or data. We further highlight the challenges associated with identifying the detrimental passages. First, even with the correct context, the model can make an incorrect prediction, posing a challenge in determining which passages are most influential. Second, evaluation typically considers lexical matching, which is not robust to variations of correct answers. Despite these limitations, our experimental results underscore the pivotal role of identifying and removing these detrimental passages for the context-efficient retrieve-then-read pipeline. Code and data are available at https://github.com/xfactlab/emnlp2023-damaging-retrieval

{{</citation>}}


### (43/78) Knowledge Corpus Error in Question Answering (Yejoon Lee et al., 2023)

{{<citation>}}

Yejoon Lee, Philhoon Oh, James Thorne. (2023)  
**Knowledge Corpus Error in Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.18076v1)  

---


**ABSTRACT**  
Recent works in open-domain question answering (QA) have explored generating context passages from large language models (LLMs), replacing the traditional retrieval step in the QA pipeline. However, it is not well understood why generated passages can be more effective than retrieved ones. This study revisits the conventional formulation of QA and introduces the concept of knowledge corpus error. This error arises when the knowledge corpus used for retrieval is only a subset of the entire string space, potentially excluding more helpful passages that exist outside the corpus. LLMs may mitigate this shortcoming by generating passages in a larger space. We come up with an experiment of paraphrasing human-annotated gold context using LLMs to observe knowledge corpus error empirically. Our results across three QA benchmarks reveal an increased performance (10% - 13%) when using paraphrased passage, indicating a signal for the existence of knowledge corpus error. Our code is available at https://github.com/xfactlab/emnlp2023-knowledge-corpus-error

{{</citation>}}


### (44/78) DUMA: a Dual-Mind Conversational Agent with Fast and Slow Thinking (Xiaoyu Tian et al., 2023)

{{<citation>}}

Xiaoyu Tian, Liangyu Chen, Na Liu, Yaxuan Liu, Wei Zou, Kaijiang Chen, Ming Cui. (2023)  
**DUMA: a Dual-Mind Conversational Agent with Fast and Slow Thinking**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.18075v1)  

---


**ABSTRACT**  
Inspired by the dual-process theory of human cognition, we introduce DUMA, a novel conversational agent framework that embodies a dual-mind mechanism through the utilization of two generative Large Language Models (LLMs) dedicated to fast and slow thinking respectively. The fast thinking model serves as the primary interface for external interactions and initial response generation, evaluating the necessity for engaging the slow thinking model based on the complexity of the complete response. When invoked, the slow thinking model takes over the conversation, engaging in meticulous planning, reasoning, and tool utilization to provide a well-analyzed response. This dual-mind configuration allows for a seamless transition between intuitive responses and deliberate problem-solving processes based on the situation. We have constructed a conversational agent to handle online inquiries in the real estate industry. The experiment proves that our method balances effectiveness and efficiency, and has a significant improvement compared to the baseline.

{{</citation>}}


### (45/78) Multi-grained Evidence Inference for Multi-choice Reading Comprehension (Yilin Zhao et al., 2023)

{{<citation>}}

Yilin Zhao, Hai Zhao, Sufeng Duan. (2023)  
**Multi-grained Evidence Inference for Multi-choice Reading Comprehension**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Reading Comprehension  
[Paper Link](http://arxiv.org/abs/2310.18070v1)  

---


**ABSTRACT**  
Multi-choice Machine Reading Comprehension (MRC) is a major and challenging task for machines to answer questions according to provided options. Answers in multi-choice MRC cannot be directly extracted in the given passages, and essentially require machines capable of reasoning from accurate extracted evidence. However, the critical evidence may be as simple as just one word or phrase, while it is hidden in the given redundant, noisy passage with multiple linguistic hierarchies from phrase, fragment, sentence until the entire passage. We thus propose a novel general-purpose model enhancement which integrates multi-grained evidence comprehensively, named Multi-grained evidence inferencer (Mugen), to make up for the inability. Mugen extracts three different granularities of evidence: coarse-, middle- and fine-grained evidence, and integrates evidence with the original passages, achieving significant and consistent performance improvement on four multi-choice MRC benchmarks.

{{</citation>}}


### (46/78) ViCLEVR: A Visual Reasoning Dataset and Hybrid Multimodal Fusion Model for Visual Question Answering in Vietnamese (Khiem Vinh Tran et al., 2023)

{{<citation>}}

Khiem Vinh Tran, Hao Phu Phan, Kiet Van Nguyen, Ngan Luu Thuy Nguyen. (2023)  
**ViCLEVR: A Visual Reasoning Dataset and Hybrid Multimodal Fusion Model for Visual Question Answering in Vietnamese**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.18046v1)  

---


**ABSTRACT**  
In recent years, Visual Question Answering (VQA) has gained significant attention for its diverse applications, including intelligent car assistance, aiding visually impaired individuals, and document image information retrieval using natural language queries. VQA requires effective integration of information from questions and images to generate accurate answers. Neural models for VQA have made remarkable progress on large-scale datasets, with a primary focus on resource-rich languages like English. To address this, we introduce the ViCLEVR dataset, a pioneering collection for evaluating various visual reasoning capabilities in Vietnamese while mitigating biases. The dataset comprises over 26,000 images and 30,000 question-answer pairs (QAs), each question annotated to specify the type of reasoning involved. Leveraging this dataset, we conduct a comprehensive analysis of contemporary visual reasoning systems, offering valuable insights into their strengths and limitations. Furthermore, we present PhoVIT, a comprehensive multimodal fusion that identifies objects in images based on questions. The architecture effectively employs transformers to enable simultaneous reasoning over textual and visual data, merging both modalities at an early model stage. The experimental findings demonstrate that our proposed model achieves state-of-the-art performance across four evaluation metrics. The accompanying code and dataset have been made publicly accessible at \url{https://github.com/kvt0012/ViCLEVR}. This provision seeks to stimulate advancements within the research community, fostering the development of more multimodal fusion algorithms, specifically tailored to address the nuances of low-resource languages, exemplified by Vietnamese.

{{</citation>}}


### (47/78) On General Language Understanding (David Schlangen, 2023)

{{<citation>}}

David Schlangen. (2023)  
**On General Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.18038v1)  

---


**ABSTRACT**  
Natural Language Processing prides itself to be an empirically-minded, if not outright empiricist field, and yet lately it seems to get itself into essentialist debates on issues of meaning and measurement ("Do Large Language Models Understand Language, And If So, How Much?"). This is not by accident: Here, as everywhere, the evidence underspecifies the understanding. As a remedy, this paper sketches the outlines of a model of understanding, which can ground questions of the adequacy of current methods of measurement of model quality. The paper makes three claims: A) That different language use situation types have different characteristics, B) That language understanding is a multifaceted phenomenon, bringing together individualistic and social processes, and C) That the choice of Understanding Indicator marks the limits of benchmarking, and the beginnings of considerations of the ethics of NLP use.

{{</citation>}}


### (48/78) Large language models for aspect-based sentiment analysis (Paul F. Simmering et al., 2023)

{{<citation>}}

Paul F. Simmering, Paavo Huoviala. (2023)  
**Large language models for aspect-based sentiment analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.18025v1)  

---


**ABSTRACT**  
Large language models (LLMs) offer unprecedented text completion capabilities. As general models, they can fulfill a wide range of roles, including those of more specialized models. We assess the performance of GPT-4 and GPT-3.5 in zero shot, few shot and fine-tuned settings on the aspect-based sentiment analysis (ABSA) task. Fine-tuned GPT-3.5 achieves a state-of-the-art F1 score of 83.8 on the joint aspect term extraction and polarity classification task of the SemEval-2014 Task 4, improving upon InstructABSA [@scaria_instructabsa_2023] by 5.7%. However, this comes at the price of 1000 times more model parameters and thus increased inference cost. We discuss the the cost-performance trade-offs of different models, and analyze the typical errors that they make. Our results also indicate that detailed prompts improve performance in zero-shot and few-shot settings but are not necessary for fine-tuned models. This evidence is relevant for practioners that are faced with the choice of prompt engineering versus fine-tuning when using LLMs for ABSA.

{{</citation>}}


### (49/78) SentMix-3L: A Bangla-English-Hindi Code-Mixed Dataset for Sentiment Analysis (Md Nishat Raihan et al., 2023)

{{<citation>}}

Md Nishat Raihan, Dhiman Goswami, Antara Mahmud, Antonios Anstasopoulos, Marcos Zampieri. (2023)  
**SentMix-3L: A Bangla-English-Hindi Code-Mixed Dataset for Sentiment Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2310.18023v1)  

---


**ABSTRACT**  
Code-mixing is a well-studied linguistic phenomenon when two or more languages are mixed in text or speech. Several datasets have been build with the goal of training computational models for code-mixing. Although it is very common to observe code-mixing with multiple languages, most datasets available contain code-mixed between only two languages. In this paper, we introduce SentMix-3L, a novel dataset for sentiment analysis containing code-mixed data between three languages Bangla, English, and Hindi. We carry out a comprehensive evaluation using SentMix-3L. We show that zero-shot prompting with GPT-3.5 outperforms all transformer-based models on SentMix-3L.

{{</citation>}}


### (50/78) NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark (Oscar Sainz et al., 2023)

{{<citation>}}

Oscar Sainz, Jon Ander Campos, Iker García-Ferrero, Julen Etxaniz, Oier Lopez de Lacalle, Eneko Agirre. (2023)  
**NLP Evaluation in trouble: On the Need to Measure LLM Data Contamination for each Benchmark**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.18018v1)  

---


**ABSTRACT**  
In this position paper, we argue that the classical evaluation on Natural Language Processing (NLP) tasks using annotated benchmarks is in trouble. The worst kind of data contamination happens when a Large Language Model (LLM) is trained on the test split of a benchmark, and then evaluated in the same benchmark. The extent of the problem is unknown, as it is not straightforward to measure. Contamination causes an overestimation of the performance of a contaminated model in a target benchmark and associated task with respect to their non-contaminated counterparts. The consequences can be very harmful, with wrong scientific conclusions being published while other correct ones are discarded. This position paper defines different levels of data contamination and argues for a community effort, including the development of automatic and semi-automatic measures to detect when data from a benchmark was exposed to a model, and suggestions for flagging papers with conclusions that are compromised by data contamination.

{{</citation>}}


### (51/78) Does Role-Playing Chatbots Capture the Character Personalities? Assessing Personality Traits for Role-Playing Chatbots (Xintao Wang et al., 2023)

{{<citation>}}

Xintao Wang, Xintao Wang, Yaying Fei, Ziang Leng, Cheng Li. (2023)  
**Does Role-Playing Chatbots Capture the Character Personalities? Assessing Personality Traits for Role-Playing Chatbots**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17976v1)  

---


**ABSTRACT**  
The emergence of large-scale pretrained language models has revolutionized the capabilities of new AI application, especially in the realm of crafting chatbots with distinct personas. Given the "stimulus-response" nature of chatbots, this paper unveils an innovative open-ended interview-style approach for personality assessment on role-playing chatbots, which offers a richer comprehension of their intrinsic personalities. We conduct personality assessments on 32 role-playing chatbots created by the ChatHaruhi library, across both the Big Five and MBTI dimensions, and measure their alignment with human perception. Evaluation results underscore that modern role-playing chatbots based on LLMs can effectively portray personality traits of corresponding characters, with an alignment rate of 82.8% compared with human-perceived personalities. Besides, we also suggest potential strategies for shaping chatbots' personalities. Hence, this paper serves as a cornerstone study for role-playing chatbots that intersects computational linguistics and psychology. Our resources are available at https://github.com/LC1332/Chat-Haruhi-Suzumiya

{{</citation>}}


### (52/78) Transformers as Graph-to-Graph Models (James Henderson et al., 2023)

{{<citation>}}

James Henderson, Alireza Mohammadshahi, Andrei C. Coman, Lesly Miculicich. (2023)  
**Transformers as Graph-to-Graph Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.17936v1)  

---


**ABSTRACT**  
We argue that Transformers are essentially graph-to-graph models, with sequences just being a special case. Attention weights are functionally equivalent to graph edges. Our Graph-to-Graph Transformer architecture makes this ability explicit, by inputting graph edges into the attention weight computations and predicting graph edges with attention-like functions, thereby integrating explicit graphs into the latent graphs learned by pretrained Transformers. Adding iterative graph refinement provides a joint embedding of input, output, and latent graphs, allowing non-autoregressive graph prediction to optimise the complete graph without any bespoke pipeline or decoding strategy. Empirical results show that this architecture achieves state-of-the-art accuracies for modelling a variety of linguistic structures, integrating very effectively with the latent linguistic representations learned by pretraining.

{{</citation>}}


### (53/78) SOUL: Towards Sentiment and Opinion Understanding of Language (Yue Deng et al., 2023)

{{<citation>}}

Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, Lidong Bing. (2023)  
**SOUL: Towards Sentiment and Opinion Understanding of Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, NLP  
[Paper Link](http://arxiv.org/abs/2310.17924v1)  

---


**ABSTRACT**  
Sentiment analysis is a well-established natural language processing task, with sentiment polarity classification being one of its most popular and representative tasks. However, despite the success of pre-trained language models in this area, they often fall short of capturing the broader complexities of sentiment analysis. To address this issue, we propose a new task called Sentiment and Opinion Understanding of Language (SOUL). SOUL aims to evaluate sentiment understanding through two subtasks: Review Comprehension (RC) and Justification Generation (JG). RC seeks to validate statements that focus on subjective information based on a review text, while JG requires models to provide explanations for their sentiment predictions. To enable comprehensive evaluation, we annotate a new dataset comprising 15,028 statements from 3,638 reviews. Experimental results indicate that SOUL is a challenging task for both small and large language models, with a performance gap of up to 27% when compared to human performance. Furthermore, evaluations conducted with both human experts and GPT-4 highlight the limitations of the small language model in generating reasoning-based justifications. These findings underscore the challenging nature of the SOUL task for existing models, emphasizing the need for further advancements in sentiment analysis to address its complexities. The new dataset and code are available at https://github.com/DAMO-NLP-SG/SOUL.

{{</citation>}}


### (54/78) Knowing What LLMs DO NOT Know: A Simple Yet Effective Self-Detection Method (Yukun Zhao et al., 2023)

{{<citation>}}

Yukun Zhao, Lingyong Yan, Weiwei Sun, Guoliang Xing, Chong Meng, Shuaiqiang Wang, Zhicong Cheng, Zhaochun Ren, Dawei Yin. (2023)  
**Knowing What LLMs DO NOT Know: A Simple Yet Effective Self-Detection Method**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.17918v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown great potential in Natural Language Processing (NLP) tasks. However, recent literature reveals that LLMs generate nonfactual responses intermittently, which impedes the LLMs' reliability for further utilization. In this paper, we propose a novel self-detection method to detect which questions that a LLM does not know that are prone to generate nonfactual results. Specifically, we first diversify the textual expressions for a given question and collect the corresponding answers. Then we examine the divergencies between the generated answers to identify the questions that the model may generate falsehoods. All of the above steps can be accomplished by prompting the LLMs themselves without referring to any other external resources. We conduct comprehensive experiments and demonstrate the effectiveness of our method on recently released LLMs, e.g., Vicuna, ChatGPT, and GPT-4.

{{</citation>}}


### (55/78) Natural Language Interfaces for Tabular Data Querying and Visualization: A Survey (Weixu Zhang et al., 2023)

{{<citation>}}

Weixu Zhang, Yifei Wang, Yuanfeng Song, Victor Junqiu Wei, Yuxing Tian, Yiyan Qi, Jonathan H. Chan, Raymond Chi-Wing Wong, Haiqin Yang. (2023)  
**Natural Language Interfaces for Tabular Data Querying and Visualization: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.17894v1)  

---


**ABSTRACT**  
The emergence of natural language processing has revolutionized the way users interact with tabular data, enabling a shift from traditional query languages and manual plotting to more intuitive, language-based interfaces. The rise of large language models (LLMs) such as ChatGPT and its successors has further advanced this field, opening new avenues for natural language processing techniques. This survey presents a comprehensive overview of natural language interfaces for tabular data querying and visualization, which allow users to interact with data using natural language queries. We introduce the fundamental concepts and techniques underlying these interfaces with a particular emphasis on semantic parsing, the key technology facilitating the translation from natural language to SQL queries or data visualization commands. We then delve into the recent advancements in Text-to-SQL and Text-to-Vis problems from the perspectives of datasets, methodologies, metrics, and system designs. This includes a deep dive into the influence of LLMs, highlighting their strengths, limitations, and potential for future improvements. Through this survey, we aim to provide a roadmap for researchers and practitioners interested in developing and applying natural language interfaces for data interaction in the era of large language models.

{{</citation>}}


### (56/78) ASPIRO: Any-shot Structured Parsing-error-Induced ReprOmpting for Consistent Data-to-Text Generation (Martin Vejvar et al., 2023)

{{<citation>}}

Martin Vejvar, Yasutaka Fujimoto. (2023)  
**ASPIRO: Any-shot Structured Parsing-error-Induced ReprOmpting for Consistent Data-to-Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BLEU, Text Generation  
[Paper Link](http://arxiv.org/abs/2310.17877v1)  

---


**ABSTRACT**  
We present ASPIRO, an approach for structured data verbalisation into short template sentences in zero to few-shot settings. Unlike previous methods, our approach prompts large language models (LLMs) to directly produce entity-agnostic templates, rather than relying on LLMs to faithfully copy the given example entities, or validating/crafting the templates manually. We incorporate LLM re-prompting, triggered by algorithmic parsing checks, as well as the PARENT metric induced consistency validation to identify and rectify template generation problems in real-time. ASPIRO, compared to direct LLM output, averages 66\% parsing error rate reduction in generated verbalisations of RDF triples on the DART dataset. Our best 5-shot text-davinci-003 setup, scoring BLEU of 50.62, METEOR of 45.16, BLEURT of 0.82, NUBIA of 0.87, and PARENT of 0.8962 on the Rel2Text dataset, competes effectively with recent fine-tuned pre-trained language models.

{{</citation>}}


### (57/78) TarGEN: Targeted Data Generation with Large Language Models (Himanshu Gupta et al., 2023)

{{<citation>}}

Himanshu Gupta, Kevin Scaria, Ujjwala Anantheswaran, Shreyas Verma, Mihir Parmar, Saurabh Arjun Sawant, Swaroop Mishra, Chitta Baral. (2023)  
**TarGEN: Targeted Data Generation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE, Language Model, SuperGLUE, T5  
[Paper Link](http://arxiv.org/abs/2310.17876v1)  

---


**ABSTRACT**  
The rapid advancement of large language models (LLMs) has sparked interest in data synthesis techniques, aiming to generate diverse and high-quality synthetic datasets. However, these synthetic datasets often suffer from a lack of diversity and added noise. In this paper, we present TarGEN, a multi-step prompting strategy for generating high-quality synthetic datasets utilizing a LLM. An advantage of TarGEN is its seedless nature; it does not require specific task instances, broadening its applicability beyond task replication. We augment TarGEN with a method known as self-correction empowering LLMs to rectify inaccurately labeled instances during dataset creation, ensuring reliable labels. To assess our technique's effectiveness, we emulate 8 tasks from the SuperGLUE benchmark and finetune various language models, including encoder-only, encoder-decoder, and decoder-only models on both synthetic and original training sets. Evaluation on the original test set reveals that models trained on datasets generated by TarGEN perform approximately 1-2% points better than those trained on original datasets (82.84% via syn. vs. 81.12% on og. using Flan-T5). When incorporating instruction tuning, the performance increases to 84.54% on synthetic data vs. 81.49% on original data by Flan-T5. A comprehensive analysis of the synthetic dataset compared to the original dataset reveals that the synthetic dataset demonstrates similar or higher levels of dataset complexity and diversity. Furthermore, the synthetic dataset displays a bias level that aligns closely with the original dataset. Finally, when pre-finetuned on our synthetic SuperGLUE dataset, T5-3B yields impressive results on the OpenLLM leaderboard, surpassing the model trained on the Self-Instruct dataset by 4.14% points. We hope that TarGEN can be helpful for quality data generation and reducing the human efforts to create complex benchmarks.

{{</citation>}}


### (58/78) From Values to Opinions: Predicting Human Behaviors and Stances Using Value-Injected Large Language Models (Dongjun Kang et al., 2023)

{{<citation>}}

Dongjun Kang, Joonsuk Park, Yohan Jo, JinYeong Bak. (2023)  
**From Values to Opinions: Predicting Human Behaviors and Stances Using Value-Injected Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17857v1)  

---


**ABSTRACT**  
Being able to predict people's opinions on issues and behaviors in realistic scenarios can be helpful in various domains, such as politics and marketing. However, conducting large-scale surveys like the European Social Survey to solicit people's opinions on individual issues can incur prohibitive costs. Leveraging prior research showing influence of core human values on individual decisions and actions, we propose to use value-injected large language models (LLM) to predict opinions and behaviors. To this end, we present Value Injection Method (VIM), a collection of two methods -- argument generation and question answering -- designed to inject targeted value distributions into LLMs via fine-tuning. We then conduct a series of experiments on four tasks to test the effectiveness of VIM and the possibility of using value-injected LLMs to predict opinions and behaviors of people. We find that LLMs value-injected with variations of VIM substantially outperform the baselines. Also, the results suggest that opinions and behaviors can be better predicted using value-injected LLMs than the baseline approaches.

{{</citation>}}


## cs.CY (2)



### (59/78) A Review of the Evidence for Existential Risk from AI via Misaligned Power-Seeking (Rose Hadshar, 2023)

{{<citation>}}

Rose Hadshar. (2023)  
**A Review of the Evidence for Existential Risk from AI via Misaligned Power-Seeking**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.18244v1)  

---


**ABSTRACT**  
Rapid advancements in artificial intelligence (AI) have sparked growing concerns among experts, policymakers, and world leaders regarding the potential for increasingly advanced AI systems to pose existential risks. This paper reviews the evidence for existential risks from AI via misalignment, where AI systems develop goals misaligned with human values, and power-seeking, where misaligned AIs actively seek power. The review examines empirical findings, conceptual arguments and expert opinion relating to specification gaming, goal misgeneralization, and power-seeking. The current state of the evidence is found to be concerning but inconclusive regarding the existence of extreme forms of misaligned power-seeking. Strong empirical evidence of specification gaming combined with strong conceptual evidence for power-seeking make it difficult to dismiss the possibility of existential risk from misaligned power-seeking. On the other hand, to date there are no public empirical examples of misaligned power-seeking in AI systems, and so arguments that future systems will pose an existential risk remain somewhat speculative. Given the current state of the evidence, it is hard to be extremely confident either that misaligned power-seeking poses a large existential risk, or that it poses no existential risk. The fact that we cannot confidently rule out existential risk from AI via misaligned power-seeking is cause for serious concern.

{{</citation>}}


### (60/78) Large Language Models as Subpopulation Representative Models: A Review (Gabriel Simmons et al., 2023)

{{<citation>}}

Gabriel Simmons, Christopher Hare. (2023)  
**Large Language Models as Subpopulation Representative Models: A Review**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.17888v1)  

---


**ABSTRACT**  
Of the many commercial and scientific opportunities provided by large language models (LLMs; including Open AI's ChatGPT, Meta's LLaMA, and Anthropic's Claude), one of the more intriguing applications has been the simulation of human behavior and opinion. LLMs have been used to generate human simulcra to serve as experimental participants, survey respondents, or other independent agents, with outcomes that often closely parallel the observed behavior of their genuine human counterparts. Here, we specifically consider the feasibility of using LLMs to estimate subpopulation representative models (SRMs). SRMs could provide an alternate or complementary way to measure public opinion among demographic, geographic, or political segments of the population. However, the introduction of new technology to the socio-technical infrastructure does not come without risk. We provide an overview of behavior elicitation techniques for LLMs, and a survey of existing SRM implementations. We offer frameworks for the analysis, development, and practical implementation of LLMs as SRMs, consider potential risks, and suggest directions for future work.

{{</citation>}}


## cs.AI (4)



### (61/78) Fine-Tuning Language Models Using Formal Methods Feedback (Yunhao Yang et al., 2023)

{{<citation>}}

Yunhao Yang, Neel P. Bhatt, Tyler Ingebrand, William Ward, Steven Carr, Zhangyang Wang, Ufuk Topcu. (2023)  
**Fine-Tuning Language Models Using Formal Methods Feedback**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-FL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.18239v1)  

---


**ABSTRACT**  
Although pre-trained language models encode generic knowledge beneficial for planning and control, they may fail to generate appropriate control policies for domain-specific tasks. Existing fine-tuning methods use human feedback to address this limitation, however, sourcing human feedback is labor intensive and costly. We present a fully automated approach to fine-tune pre-trained language models for applications in autonomous systems, bridging the gap between generic knowledge and domain-specific requirements while reducing cost. The method synthesizes automaton-based controllers from pre-trained models guided by natural language task descriptions. These controllers are verifiable against independently provided specifications within a world model, which can be abstract or obtained from a high-fidelity simulator. Controllers with high compliance with the desired specifications receive higher ranks, guiding the iterative fine-tuning process. We provide quantitative evidences, primarily in autonomous driving, to demonstrate the method's effectiveness across multiple tasks. The results indicate an improvement in percentage of specifications satisfied by the controller from 60% to 90%.

{{</citation>}}


### (62/78) Moral Responsibility for AI Systems (Sander Beckers, 2023)

{{<citation>}}

Sander Beckers. (2023)  
**Moral Responsibility for AI Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-LO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.18040v1)  

---


**ABSTRACT**  
As more and more decisions that have a significant ethical dimension are being outsourced to AI systems, it is important to have a definition of moral responsibility that can be applied to AI systems. Moral responsibility for an outcome of an agent who performs some action is commonly taken to involve both a causal condition and an epistemic condition: the action should cause the outcome, and the agent should have been aware -- in some form or other -- of the possible moral consequences of their action. This paper presents a formal definition of both conditions within the framework of causal models. I compare my approach to the existing approaches of Braham and van Hees (BvH) and of Halpern and Kleiman-Weiner (HK). I then generalize my definition into a degree of responsibility.

{{</citation>}}


### (63/78) FormalGeo: The First Step Toward Human-like IMO-level Geometric Automated Reasoning (Xiaokai Zhang et al., 2023)

{{<citation>}}

Xiaokai Zhang, Na Zhu, Yiming He, Jia Zou, Qike Huang, Xiaoxiao Jin, Yanjun Guo, Chenyang Mao, Zhe Zhu, Dengfeng Yue, Fangzhen Zhu, Yang Li, Yifan Wang, Yiwen Huang, Runan Wang, Cheng Qin, Zhenbing Zeng, Shaorong Xie, Xiangfeng Luo, Tuo Leng. (2023)  
**FormalGeo: The First Step Toward Human-like IMO-level Geometric Automated Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.18021v1)  

---


**ABSTRACT**  
This is the first article of our work over the past decade. In this series of papers, we have constructed a complete and compatible formal plane geometry system. This will serve as a crucial bridge between IMO-level plane geometry challenges and readable AI automated reasoning. With this formal system in place, we have been able to seamlessly integrate modern AI models with our formal system. Within this formal framework, AI is now capable of providing deductive reasoning solutions to IMO-level plane geometry problems, just like handling other natural languages, and these proofs are readable, traceable, and verifiable. We propose the geometry formalization theory (GFT) to guide the development of the geometry formal system. Based on the GFT, we have established the FormalGeo, which consists of 88 geometric predicates and 196 theorems. It can represent, validate, and solve IMO-level geometry problems. we also have crafted the FGPS (formal geometry problem solver) in Python. It serves as both an interactive assistant for verifying problem-solving processes and an automated problem solver, utilizing various methods such as forward search, backward search and AI-assisted search. We've annotated the FormalGeo7k dataset, containing 6,981 (expand to 186,832 through data augmentation) geometry problems with complete formal language annotations. Implementation of the formal system and experiments on the FormalGeo7k validate the correctness and utility of the GFT. The backward depth-first search method only yields a 2.42% problem-solving failure rate, and we can incorporate deep learning techniques to achieve lower one. The source code of FGPS and FormalGeo7k dataset are available at https://github.com/BitSecret/FormalGeo.

{{</citation>}}


### (64/78) Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory (Niloofar Mireshghallah et al., 2023)

{{<citation>}}

Niloofar Mireshghallah, Hyunwoo Kim, Xuhui Zhou, Yulia Tsvetkov, Maarten Sap, Reza Shokri, Yejin Choi. (2023)  
**Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CR, cs.AI  
Keywords: AI, ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.17884v1)  

---


**ABSTRACT**  
The interactive use of large language models (LLMs) in AI assistants (at work, home, etc.) introduces a new set of inference-time privacy risks: LLMs are fed different types of information from multiple sources in their inputs and are expected to reason about what to share in their outputs, for what purpose and with whom, within a given context. In this work, we draw attention to the highly critical yet overlooked notion of contextual privacy by proposing ConfAIde, a benchmark designed to identify critical weaknesses in the privacy reasoning capabilities of instruction-tuned LLMs. Our experiments show that even the most capable models such as GPT-4 and ChatGPT reveal private information in contexts that humans would not, 39% and 57% of the time, respectively. This leakage persists even when we employ privacy-inducing prompts or chain-of-thought reasoning. Our work underscores the immediate need to explore novel inference-time privacy-preserving approaches, based on reasoning and theory of mind.

{{</citation>}}


## eess.IV (2)



### (65/78) Edge AI-Based Vein Detector for Efficient Venipuncture in the Antecubital Fossa (Edwin Salcedo et al., 2023)

{{<citation>}}

Edwin Salcedo, Patricia Peñaloza. (2023)  
**Edge AI-Based Vein Detector for Efficient Venipuncture in the Antecubital Fossa**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, Quantization  
[Paper Link](http://arxiv.org/abs/2310.18234v1)  

---


**ABSTRACT**  
Assessing the condition and visibility of veins is a crucial step before obtaining intravenous access in the antecubital fossa, which is a common procedure to draw blood or administer intravenous therapies (IV therapies). Even though medical practitioners are highly skilled at intravenous cannulation, they usually struggle to perform the procedure in patients with low visible veins due to fluid retention, age, overweight, dark skin tone, or diabetes. Recently, several investigations proposed combining Near Infrared (NIR) imaging and deep learning (DL) techniques for forearm vein segmentation. Although they have demonstrated compelling results, their use has been rather limited owing to the portability and precision requirements to perform venipuncture. In this paper, we aim to contribute to bridging this gap using three strategies. First, we introduce a new NIR-based forearm vein segmentation dataset of 2,016 labelled images collected from 1,008 subjects with low visible veins. Second, we propose a modified U-Net architecture that locates veins specifically in the antecubital fossa region of the examined patient. Finally, a compressed version of the proposed architecture was deployed inside a bespoke, portable vein finder device after testing four common embedded microcomputers and four common quantization modalities. Experimental results showed that the model compressed with Dynamic Range Quantization and deployed on a Raspberry Pi 4B card produced the best execution time and precision balance, with 5.14 FPS and 0.957 of latency and Intersection over Union (IoU), respectively. These results show promising performance inside a resource-restricted low-cost device.

{{</citation>}}


### (66/78) Multivessel Coronary Artery Segmentation and Stenosis Localisation using Ensemble Learning (Muhammad Bilal et al., 2023)

{{<citation>}}

Muhammad Bilal, Dinis Martinho, Reiner Sim, Adnan Qayyum, Hunaid Vohra, Massimo Caputo, Taofeek Akinosho, Sofiat Abioye, Zaheer Khan, Waleed Niaz, Junaid Qadir. (2023)  
**Multivessel Coronary Artery Segmentation and Stenosis Localisation using Ensemble Learning**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.17954v1)  

---


**ABSTRACT**  
Coronary angiography analysis is a common clinical task performed by cardiologists to diagnose coronary artery disease (CAD) through an assessment of atherosclerotic plaque's accumulation. This study introduces an end-to-end machine learning solution developed as part of our solution for the MICCAI 2023 Automatic Region-based Coronary Artery Disease diagnostics using x-ray angiography imagEs (ARCADE) challenge, which aims to benchmark solutions for multivessel coronary artery segmentation and potential stenotic lesion localisation from X-ray coronary angiograms. We adopted a robust baseline model training strategy to progressively improve performance, comprising five successive stages of binary class pretraining, multivessel segmentation, fine-tuning using class frequency weighted dataloaders, fine-tuning using F1-based curriculum learning strategy (F1-CLS), and finally multi-target angiogram view classifier-based collective adaptation. Unlike many other medical imaging procedures, this task exhibits a notable degree of interobserver variability. %, making it particularly amenable to automated analysis. Our ensemble model combines the outputs from six baseline models using the weighted ensembling approach, which our analysis shows is found to double the predictive accuracy of the proposed solution. The final prediction was further refined, targeting the correction of misclassified blobs. Our solution achieved a mean F1 score of $37.69\%$ for coronary artery segmentation, and $39.41\%$ for stenosis localisation, positioning our team in the 5th position on both leaderboards. This work demonstrates the potential of automated tools to aid CAD diagnosis, guide interventions, and improve the accuracy of stent injections in clinical settings.

{{</citation>}}


## math.AP (1)



### (67/78) On Residual Minimization for PDEs: Failure of PINN, Modified Equation, and Implicit Bias (Tao Luo et al., 2023)

{{<citation>}}

Tao Luo, Qixuan Zhou. (2023)  
**On Residual Minimization for PDEs: Failure of PINN, Modified Equation, and Implicit Bias**  

---
Primary Category: math.AP  
Categories: 35D30, 35D35, 35R05, 35R06, 65N15, cs-NA, math-AP, math-NA, math.AP  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.18201v1)  

---


**ABSTRACT**  
We study the failure phenomenon on the residual minimization (RM) methods, in particular, the physics-informed neural network (PINN), for the elliptic equations with discontinuous coefficients. To explain the failure phenomenon, we conduct numerical experiments with insightful observations and propose a modified equation to model the numerical solution via RM method. We then investigate the solution to the modified equation and, in particular, show that the modified solution deviates from the original exact solution. The proof uses a necessary and sufficient condition on characterizing the singularity in the coefficients. This equivalent condition is a useful by-product of this paper, which can be extended to other types of equations in the future. Finally, we prove that RM method implicitly biases the numerical solution against the exact solution and towards a modified solution. Extensions of the results to the quasilinear elliptic equations are also discussed.

{{</citation>}}


## cs.HC (2)



### (68/78) Deep3DSketch+\+: High-Fidelity 3D Modeling from Single Free-hand Sketches (Ying Zang et al., 2023)

{{<citation>}}

Ying Zang, Chaotao Ding, Tianrun Chen, Papa Mao, Wenjun Hu. (2023)  
**Deep3DSketch+\+: High-Fidelity 3D Modeling from Single Free-hand Sketches**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2310.18178v1)  

---


**ABSTRACT**  
The rise of AR/VR has led to an increased demand for 3D content. However, the traditional method of creating 3D content using Computer-Aided Design (CAD) is a labor-intensive and skill-demanding process, making it difficult to use for novice users. Sketch-based 3D modeling provides a promising solution by leveraging the intuitive nature of human-computer interaction. However, generating high-quality content that accurately reflects the creator's ideas can be challenging due to the sparsity and ambiguity of sketches. Furthermore, novice users often find it challenging to create accurate drawings from multiple perspectives or follow step-by-step instructions in existing methods. To address this, we introduce a groundbreaking end-to-end approach in our work, enabling 3D modeling from a single free-hand sketch, Deep3DSketch+$\backslash$+. The issue of sparsity and ambiguity using single sketch is resolved in our approach by leveraging the symmetry prior and structural-aware shape discriminator. We conducted comprehensive experiments on diverse datasets, including both synthetic and real data, to validate the efficacy of our approach and demonstrate its state-of-the-art (SOTA) performance. Users are also more satisfied with results generated by our approach according to our user study. We believe our approach has the potential to revolutionize the process of 3D modeling by offering an intuitive and easy-to-use solution for novice users.

{{</citation>}}


### (69/78) Reality3DSketch: Rapid 3D Modeling of Objects from Single Freehand Sketches (Tianrun Chen et al., 2023)

{{<citation>}}

Tianrun Chen, Chaotao Ding, Lanyun Zhu, Ying Zang, Yiyi Liao, Zejian Li, Lingyun Sun. (2023)  
**Reality3DSketch: Rapid 3D Modeling of Objects from Single Freehand Sketches**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2310.18148v1)  

---


**ABSTRACT**  
The emerging trend of AR/VR places great demands on 3D content. However, most existing software requires expertise and is difficult for novice users to use. In this paper, we aim to create sketch-based modeling tools for user-friendly 3D modeling. We introduce Reality3DSketch with a novel application of an immersive 3D modeling experience, in which a user can capture the surrounding scene using a monocular RGB camera and can draw a single sketch of an object in the real-time reconstructed 3D scene. A 3D object is generated and placed in the desired location, enabled by our novel neural network with the input of a single sketch. Our neural network can predict the pose of a drawing and can turn a single sketch into a 3D model with view and structural awareness, which addresses the challenge of sparse sketch input and view ambiguity. We conducted extensive experiments synthetic and real-world datasets and achieved state-of-the-art (SOTA) results in both sketch view estimation and 3D modeling performance. According to our user study, our method of performing 3D modeling in a scene is $>$5x faster than conventional methods. Users are also more satisfied with the generated 3D model than the results of existing methods.

{{</citation>}}


## cs.SD (1)



### (70/78) Style Description based Text-to-Speech with Conditional Prosodic Layer Normalization based Diffusion GAN (Neeraj Kumar et al., 2023)

{{<citation>}}

Neeraj Kumar, Ankur Narang, Brejesh Lall. (2023)  
**Style Description based Text-to-Speech with Conditional Prosodic Layer Normalization based Diffusion GAN**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2310.18169v1)  

---


**ABSTRACT**  
In this paper, we present a Diffusion GAN based approach (Prosodic Diff-TTS) to generate the corresponding high-fidelity speech based on the style description and content text as an input to generate speech samples within only 4 denoising steps. It leverages the novel conditional prosodic layer normalization to incorporate the style embeddings into the multi head attention based phoneme encoder and mel spectrogram decoder based generator architecture to generate the speech. The style embedding is generated by fine tuning the pretrained BERT model on auxiliary tasks such as pitch, speaking speed, emotion,gender classifications. We demonstrate the efficacy of our proposed architecture on multi-speaker LibriTTS and PromptSpeech datasets, using multiple quantitative metrics that measure generated accuracy and MOS.

{{</citation>}}


## cs.CR (1)



### (71/78) Enhancing Enterprise Network Security: Comparing Machine-Level and Process-Level Analysis for Dynamic Malware Detection (Baskoro Adi Pratomo et al., 2023)

{{<citation>}}

Baskoro Adi Pratomo, Toby Jackson, Pete Burnap, Andrew Hood, Eirini Anthi. (2023)  
**Enhancing Enterprise Network Security: Comparing Machine-Level and Process-Level Analysis for Dynamic Malware Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Network Security, Security  
[Paper Link](http://arxiv.org/abs/2310.18165v1)  

---


**ABSTRACT**  
Analysing malware is important to understand how malicious software works and to develop appropriate detection and prevention methods. Dynamic analysis can overcome evasion techniques commonly used to bypass static analysis and provide insights into malware runtime activities. Much research on dynamic analysis focused on investigating machine-level information (e.g., CPU, memory, network usage) to identify whether a machine is running malicious activities. A malicious machine does not necessarily mean all running processes on the machine are also malicious. If we can isolate the malicious process instead of isolating the whole machine, we could kill the malicious process, and the machine can keep doing its job. Another challenge dynamic malware detection research faces is that the samples are executed in one machine without any background applications running. It is unrealistic as a computer typically runs many benign (background) applications when a malware incident happens. Our experiment with machine-level data shows that the existence of background applications decreases previous state-of-the-art accuracy by about 20.12% on average. We also proposed a process-level Recurrent Neural Network (RNN)-based detection model. Our proposed model performs better than the machine-level detection model; 0.049 increase in detection rate and a false-positive rate below 0.1.

{{</citation>}}


## cs.NI (1)



### (72/78) DESiRED -- Dynamic, Enhanced, and Smart iRED: A P4-AQM with Deep Reinforcement Learning and In-band Network Telemetry (Leandro C. de Almeida et al., 2023)

{{<citation>}}

Leandro C. de Almeida, Washington Rodrigo Dias da Silva, Thiago C. Tavares, Rafael Pasquini, Chrysa Papagianni, Fábio L. Verdi. (2023)  
**DESiRED -- Dynamic, Enhanced, and Smart iRED: A P4-AQM with Deep Reinforcement Learning and In-band Network Telemetry**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.18159v1)  

---


**ABSTRACT**  
Active Queue Management (AQM) is a mechanism employed to alleviate transient congestion in network device buffers, such as routers and switches. Traditional AQM algorithms use fixed thresholds, like target delay or queue occupancy, to compute random packet drop probabilities. A very small target delay can increase packet losses and reduce link utilization, while a large target delay may increase queueing delays while lowering drop probability. Due to dynamic network traffic characteristics, where traffic fluctuations can lead to significant queue variations, maintaining a fixed threshold AQM may not suit all applications. Consequently, we explore the question: \textit{What is the ideal threshold (target delay) for AQMs?} In this work, we introduce DESiRED (Dynamic, Enhanced, and Smart iRED), a P4-based AQM that leverages precise network feedback from In-band Network Telemetry (INT) to feed a Deep Reinforcement Learning (DRL) model. This model dynamically adjusts the target delay based on rewards that maximize application Quality of Service (QoS). We evaluate DESiRED in a realistic P4-based test environment running an MPEG-DASH service. Our findings demonstrate up to a 90x reduction in video stall and a 42x increase in high-resolution video playback quality when the target delay is adjusted dynamically by DESiRED.

{{</citation>}}


## cs.DS (1)



### (73/78) Sketching and Streaming for Dictionary Compression (Ruben Becker et al., 2023)

{{<citation>}}

Ruben Becker, Matteo Canton, Davide Cenzato, Sung-Hwan Kim, Bojana Kodric, Nicola Prezza. (2023)  
**Sketching and Streaming for Dictionary Compression**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs.DS  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2310.17980v1)  

---


**ABSTRACT**  
We initiate the study of sub-linear sketching and streaming techniques for estimating the output size of common dictionary compressors such as Lempel-Ziv '77, the run-length Burrows-Wheeler transform, and grammar compression. To this end, we focus on a measure that has recently gained much attention in the information-theoretic community and which approximates up to a polylogarithmic multiplicative factor the output sizes of those compressors: the normalized substring complexity function $\delta$. As a matter of fact, $\delta$ itself is a very accurate measure of compressibility: it is monotone under concatenation, invariant under reversals and alphabet permutations, sub-additive, and asymptotically tight (in terms of worst-case entropy) for representing strings, up to polylogarithmic factors.   We present a data sketch of $O(\epsilon^{-3}\log n + \epsilon^{-1}\log^2 n)$ words that allows computing a multiplicative $(1\pm \epsilon)$-approximation of $\delta$ with high probability, where $n$ is the string length. The sketches of two strings $S_1,S_2$ can be merged in $O(\epsilon^{-1}\log^2 n)$ time to yield the sketch of $\{S_1,S_2\}$, speeding up by orders of magnitude tasks such as the computation of all-pairs \emph{Normalized Compression Distances} (NCD). If random access is available on the input, our sketch can be updated in $O(\epsilon^{-1}\log^2 n)$ time for each character right-extension of the string. This yields a polylogarithmic-space algorithm for approximating $\delta$, improving exponentially over the working space of the state-of-the-art algorithms running in nearly-linear time. Motivated by the fact that random access is not always available on the input data, we then present a streaming algorithm computing our sketch in $O(\sqrt n \cdot \log n)$ working space and $O(\epsilon^{-1}\log^2 n)$ worst-case delay per character.

{{</citation>}}


## cs.IR (1)



### (74/78) Chain-of-Choice Hierarchical Policy Learning for Conversational Recommendation (Wei Fan et al., 2023)

{{<citation>}}

Wei Fan, Weijia Zhang, Weiqi Wang, Yangqiu Song, Hao Liu. (2023)  
**Chain-of-Choice Hierarchical Policy Learning for Conversational Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Conversational Recommendation  
[Paper Link](http://arxiv.org/abs/2310.17922v1)  

---


**ABSTRACT**  
Conversational Recommender Systems (CRS) illuminate user preferences via multi-round interactive dialogues, ultimately navigating towards precise and satisfactory recommendations. However, contemporary CRS are limited to inquiring binary or multi-choice questions based on a single attribute type (e.g., color) per round, which causes excessive rounds of interaction and diminishes the user's experience. To address this, we propose a more realistic and efficient conversational recommendation problem setting, called Multi-Type-Attribute Multi-round Conversational Recommendation (MTAMCR), which enables CRS to inquire about multi-choice questions covering multiple types of attributes in each round, thereby improving interactive efficiency. Moreover, by formulating MTAMCR as a hierarchical reinforcement learning task, we propose a Chain-of-Choice Hierarchical Policy Learning (CoCHPL) framework to enhance both the questioning efficiency and recommendation effectiveness in MTAMCR. Specifically, a long-term policy over options (i.e., ask or recommend) determines the action type, while two short-term intra-option policies sequentially generate the chain of attributes or items through multi-step reasoning and selection, optimizing the diversity and interdependence of questioning attributes. Finally, extensive experiments on four benchmarks demonstrate the superior performance of CoCHPL over prevailing state-of-the-art methods.

{{</citation>}}


## cs.SE (1)



### (75/78) Pitfalls in Language Models for Code Intelligence: A Taxonomy and Survey (Xinyu She et al., 2023)

{{<citation>}}

Xinyu She, Yue Liu, Yanjie Zhao, Yiling He, Li Li, Chakkrit Tantithamthavorn, Zhan Qin, Haoyu Wang. (2023)  
**Pitfalls in Language Models for Code Intelligence: A Taxonomy and Survey**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17903v1)  

---


**ABSTRACT**  
Modern language models (LMs) have been successfully employed in source code generation and understanding, leading to a significant increase in research focused on learning-based code intelligence, such as automated bug repair, and test case generation. Despite their great potential, language models for code intelligence (LM4Code) are susceptible to potential pitfalls, which hinder realistic performance and further impact their reliability and applicability in real-world deployment. Such challenges drive the need for a comprehensive understanding - not just identifying these issues but delving into their possible implications and existing solutions to build more reliable language models tailored to code intelligence. Based on a well-defined systematic research approach, we conducted an extensive literature review to uncover the pitfalls inherent in LM4Code. Finally, 67 primary studies from top-tier venues have been identified. After carefully examining these studies, we designed a taxonomy of pitfalls in LM4Code research and conducted a systematic study to summarize the issues, implications, current solutions, and challenges of different pitfalls for LM4Code systems. We developed a comprehensive classification scheme that dissects pitfalls across four crucial aspects: data collection and labeling, system design and learning, performance evaluation, and deployment and maintenance. Through this study, we aim to provide a roadmap for researchers and practitioners, facilitating their understanding and utilization of LM4Code in reliable and trustworthy ways.

{{</citation>}}


## cs.IT (1)



### (76/78) User Association and Resource Allocation in Large Language Model Based Mobile Edge Computing System over Wireless Communications (Liangxin Qian et al., 2023)

{{<citation>}}

Liangxin Qian, Jun Zhao. (2023)  
**User Association and Resource Allocation in Large Language Model Based Mobile Edge Computing System over Wireless Communications**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17872v1)  

---


**ABSTRACT**  
In the rapidly evolving landscape of large language models (LLMs) and mobile edge computing, the need for efficient service delivery to mobile users with constrained computational resources has become paramount. Addressing this, our paper delves into a collaborative framework for model training where user data and model adapters are shared with servers to optimize performance. Within this framework, users initially update the first several layers of the adapters while freezing the other layers of them, leveraging their local datasets. Once this step is complete, these partially trained parameters are transmitted to servers. The servers, equipped with more robust computational capabilities, then update the subsequent layers. After this training, they send the enhanced parameters back to the users. This collaborative training approach ensures that mobile users with limited computational capacities can still benefit from advanced LLM services without being burdened by exhaustive computations. Central to our methodology is the DASHF algorithm, which encapsulates the Dinkelbach algorithm, alternating optimization, semidefinite relaxation (SDR), the Hungarian method, and a pioneering fractional programming technique from our recent IEEE JSAC paper "Human-Centric Resource Allocation in the Metaverse over Wireless Communications". The crux of DASHF is its capability to reformulate an optimization problem as Quadratically Constrained Quadratic Programming (QCQP) via meticulously crafted transformations, making it solvable by SDR and the Hungarian algorithm. Through extensive simulations, we demonstrate the effectiveness of the DASHF algorithm, offering significant insights for the advancement of collaborative LLM service deployments.

{{</citation>}}


## stat.ML (1)



### (77/78) Boosting Data Analytics With Synthetic Volume Expansion (Xiaotong Shen et al., 2023)

{{<citation>}}

Xiaotong Shen, Yifei Liu, Rex Shen. (2023)  
**Boosting Data Analytics With Synthetic Volume Expansion**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.17848v1)  

---


**ABSTRACT**  
Synthetic data generation, a cornerstone of Generative Artificial Intelligence, signifies a paradigm shift in data science by addressing data scarcity and privacy while enabling unprecedented performance. As synthetic data gains prominence, questions arise concerning the accuracy of statistical methods when applied to synthetic data compared to raw data. In this article, we introduce the Synthetic Data Generation for Analytics framework. This framework employs statistical methods on high-fidelity synthetic data generated by advanced models such as tabular diffusion and Generative Pre-trained Transformer models. These models, trained on raw data, are further enhanced with insights from pertinent studies. A significant discovery within this framework is the generational effect: the error of a statistical method on synthetic data initially diminishes with added synthetic data but may eventually increase or plateau. This phenomenon, rooted in the complexities of replicating raw data distributions, highlights a "reflection point"--an optimal threshold in the size of synthetic data determined by specific error metrics. Through three illustrative case studies-sentiment analysis of texts, predictive modeling of structured data, and inference in tabular data--we demonstrate the effectiveness of this framework over traditional ones. We underline its potential to amplify various statistical methods, including gradient boosting for prediction and hypothesis testing, thereby underscoring the transformative potential of synthetic data generation in data science.

{{</citation>}}


## cs.GR (1)



### (78/78) Real-time Animation Generation and Control on Rigged Models via Large Language Models (Han Huang et al., 2023)

{{<citation>}}

Han Huang, Fernanda De La Torre, Cathy Mengying Fang, Andrzej Banburski-Fahey, Judith Amores, Jaron Lanier. (2023)  
**Real-time Animation Generation and Control on Rigged Models via Large Language Models**  

---
Primary Category: cs.GR  
Categories: cs-AI, cs-GR, cs.GR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.17838v1)  

---


**ABSTRACT**  
We introduce a novel method for real-time animation control and generation on rigged models using natural language input. First, we embed a large language model (LLM) in Unity to output structured texts that can be parsed into diverse and realistic animations. Second, we illustrate LLM's potential to enable flexible state transition between existing animations. We showcase the robustness of our approach through qualitative results on various rigged models and motions.

{{</citation>}}
