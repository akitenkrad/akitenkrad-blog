---
draft: false
title: "arXiv @ 2023.12.14"
date: 2023-12-14
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.14"
    identifier: arxiv_20231214
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (3)](#csro-3)
- [cs.LG (24)](#cslg-24)
- [cs.CL (22)](#cscl-22)
- [cs.CR (5)](#cscr-5)
- [cs.AI (21)](#csai-21)
- [cs.HC (7)](#cshc-7)
- [eess.IV (4)](#eessiv-4)
- [q-bio.NC (1)](#q-bionc-1)
- [cs.CV (48)](#cscv-48)
- [cs.DS (1)](#csds-1)
- [eess.AS (2)](#eessas-2)
- [cs.MM (2)](#csmm-2)
- [cs.NI (1)](#csni-1)
- [cs.SE (2)](#csse-2)
- [cs.IR (1)](#csir-1)
- [physics.med-ph (1)](#physicsmed-ph-1)
- [cs.DC (1)](#csdc-1)
- [cs.SD (1)](#cssd-1)
- [eess.SY (1)](#eesssy-1)
- [cs.NE (1)](#csne-1)

## cs.RO (3)



### (1/149) Feasible Space Monitoring for Multiple Control Barrier Functions with application to Large Scale Indoor Navigation (Hardik Parwana et al., 2023)

{{<citation>}}

Hardik Parwana, Mitchell Black, Bardh Hoxha, Hideki Okamoto, Georgios Fainekos, Danil Prokhorov, Dimitra Panagou. (2023)  
**Feasible Space Monitoring for Multiple Control Barrier Functions with application to Large Scale Indoor Navigation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO, math-OC  
Keywords: AWS  
[Paper Link](http://arxiv.org/abs/2312.07803v1)  

---


**ABSTRACT**  
Quadratic programs (QP) subject to multiple time-dependent control barrier function (CBF) based constraints have been used to design safety-critical controllers. However, ensuring the existence of a solution at all times to the QP subject to multiple CBF constraints is non-trivial. We quantify the feasible solution space of the QP in terms of its volume. We introduce a novel feasible space volume monitoring control barrier function that promotes compatibility of barrier functions and, hence, existence of a solution at all times. We show empirically that our approach not only enhances feasibility but also exhibits reduced sensitivity to changes in the hyperparameters such as gains of nominal controller. Finally, paired with a global planner, we evaluate our controller for navigation among humans in the AWS Hospital gazebo environment. The proposed controller is demonstrated to outperform the standard CBF-QP controller in maintaining feasibility.

{{</citation>}}


### (2/149) Reacting like Humans: Incorporating Intrinsic Human Behaviors into NAO through Sound-Based Reactions for Enhanced Sociability (Ali Ghadami et al., 2023)

{{<citation>}}

Ali Ghadami, Mohammadreza Taghimohammadi, Mohammad Mohammadzadeh, Mohammad Hosseinipour, Alireza Taheri. (2023)  
**Reacting like Humans: Incorporating Intrinsic Human Behaviors into NAO through Sound-Based Reactions for Enhanced Sociability**  

---
Primary Category: cs.RO  
Categories: 68T40, cs-AI, cs-LG, cs-RO, cs-SD, cs.RO, eess-AS, eess-IV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.07671v1)  

---


**ABSTRACT**  
Robots' acceptability among humans and their sociability can be significantly enhanced by incorporating human-like reactions. Humans can react to environmental events very quickly and without thinking. An instance where humans display natural reactions is when they encounter a sudden and loud sound that startles or frightens them. During such moments, individuals may instinctively move their hands, turn toward the origin of the sound, and try to determine the event's cause. This inherent behavior motivated us to explore this less-studied part of social robotics. In this work, a multi-modal system composed of an action generator, sound classifier, and YOLO object detector was designed to sense the environment and, in the presence of sudden loud sounds, show natural human fear reactions, and finally, locate the fear-causing sound source in the environment. These unique and valid generated motions and inferences could imitate intrinsic human reactions and enhance the sociability of robots. For motion generation, a model based on LSTM and MDN networks was proposed to synthesize various motions. Also, in the case of sound detection, a transfer learning model was preferred that used the spectrogram of sound signals as its input. After developing individual models for sound detection, motion generation, and image recognition, they were integrated into a comprehensive fear module that was implemented on the NAO robot. Finally, the fear module was tested in practical application and two groups of experts and non-experts filled out a questionnaire to evaluate the performance of the robot. Given our promising results, this preliminary exploratory research provides a fresh perspective on social robotics and could be a starting point for modeling intrinsic human behaviors and emotions in robots.

{{</citation>}}


### (3/149) Daily Assistive View Control Learning of Low-Cost Low-Rigidity Robot via Large-Scale Vision-Language Model (Kento Kawaharazuka et al., 2023)

{{<citation>}}

Kento Kawaharazuka, Naoaki Kanazawa, Yoshiki Obinata, Kei Okada, Masayuki Inaba. (2023)  
**Daily Assistive View Control Learning of Low-Cost Low-Rigidity Robot via Large-Scale Vision-Language Model**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07451v1)  

---


**ABSTRACT**  
In this study, we develop a simple daily assistive robot that controls its own vision according to linguistic instructions. The robot performs several daily tasks such as recording a user's face, hands, or screen, and remotely capturing images of desired locations. To construct such a robot, we combine a pre-trained large-scale vision-language model with a low-cost low-rigidity robot arm. The correlation between the robot's physical and visual information is learned probabilistically using a neural network, and changes in the probability distribution based on changes in time and environment are considered by parametric bias, which is a learnable network input variable. We demonstrate the effectiveness of this learning method by open-vocabulary view control experiments with an actual robot arm, MyCobot.

{{</citation>}}


## cs.LG (24)



### (4/149) Estimation of embedding vectors in high dimensions (Golara Ahmadi Azar et al., 2023)

{{<citation>}}

Golara Ahmadi Azar, Melika Emami, Alyson Fletcher, Sundeep Rangan. (2023)  
**Estimation of embedding vectors in high dimensions**  

---
Primary Category: cs.LG  
Categories: cs-IT, cs-LG, cs.LG, math-IT, stat-ML  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.07802v1)  

---


**ABSTRACT**  
Embeddings are a basic initial feature extraction step in many machine learning models, particularly in natural language processing. An embedding attempts to map data tokens to a low-dimensional space where similar tokens are mapped to vectors that are close to one another by some metric in the embedding space. A basic question is how well can such embedding be learned? To study this problem, we consider a simple probability model for discrete data where there is some "true" but unknown embedding where the correlation of random variables is related to the similarity of the embeddings. Under this model, it is shown that the embeddings can be learned by a variant of low-rank approximate message passing (AMP) method. The AMP approach enables precise predictions of the accuracy of the estimation in certain high-dimensional limits. In particular, the methodology provides insight on the relations of key parameters such as the number of samples per value, the frequency of the terms, and the strength of the embedding correlation on the probability distribution. Our theoretical findings are validated by simulations on both synthetic data and real text data.

{{</citation>}}


### (5/149) Traffic Signal Control Using Lightweight Transformers: An Offline-to-Online RL Approach (Xingshuai Huang et al., 2023)

{{<citation>}}

Xingshuai Huang, Di Wu, Benoit Boulet. (2023)  
**Traffic Signal Control Using Lightweight Transformers: An Offline-to-Online RL Approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.07795v1)  

---


**ABSTRACT**  
Efficient traffic signal control is critical for reducing traffic congestion and improving overall transportation efficiency. The dynamic nature of traffic flow has prompted researchers to explore Reinforcement Learning (RL) for traffic signal control (TSC). Compared with traditional methods, RL-based solutions have shown preferable performance. However, the application of RL-based traffic signal controllers in the real world is limited by the low sample efficiency and high computational requirements of these solutions. In this work, we propose DTLight, a simple yet powerful lightweight Decision Transformer-based TSC method that can learn policy from easily accessible offline datasets. DTLight novelly leverages knowledge distillation to learn a lightweight controller from a well-trained larger teacher model to reduce implementation computation. Additionally, it integrates adapter modules to mitigate the expenses associated with fine-tuning, which makes DTLight practical for online adaptation with minimal computation and only a few fine-tuning steps during real deployment. Moreover, DTLight is further enhanced to be more applicable to real-world TSC problems. Extensive experiments on synthetic and real-world scenarios show that DTLight pre-trained purely on offline datasets can outperform state-of-the-art online RL-based methods in most scenarios. Experiment results also show that online fine-tuning further improves the performance of DTLight by up to 42.6% over the best online RL baseline methods. In this work, we also introduce Datasets specifically designed for TSC with offline RL (referred to as DTRL). Our datasets and code are publicly available.

{{</citation>}}


### (6/149) IDKM: Memory Efficient Neural Network Quantization via Implicit, Differentiable $k$-Means (Sean Jaffe et al., 2023)

{{<citation>}}

Sean Jaffe, Ambuj K. Singh, Francesco Bullo. (2023)  
**IDKM: Memory Efficient Neural Network Quantization via Implicit, Differentiable $k$-Means**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2312.07759v1)  

---


**ABSTRACT**  
Compressing large neural networks with minimal performance loss is crucial to enabling their deployment on edge devices. (Cho et al., 2022) proposed a weight quantization method that uses an attention-based clustering algorithm called differentiable $k$-means (DKM). Despite achieving state-of-the-art results, DKM's performance is constrained by its heavy memory dependency. We propose an implicit, differentiable $k$-means algorithm (IDKM), which eliminates the major memory restriction of DKM. Let $t$ be the number of $k$-means iterations, $m$ be the number of weight-vectors, and $b$ be the number of bits per cluster address. IDKM reduces the overall memory complexity of a single $k$-means layer from $\mathcal{O}(t \cdot m \cdot 2^b)$ to $\mathcal{O}( m \cdot 2^b)$. We also introduce a variant, IDKM with Jacobian-Free-Backpropagation (IDKM-JFB), for which the time complexity of the gradient calculation is independent of $t$ as well. We provide a proof of concept of our methods by showing that, under the same settings, IDKM achieves comparable performance to DKM with less compute time and less memory. We also use IDKM and IDKM-JFB to quantize a large neural network, Resnet18, on hardware where DKM cannot train at all.

{{</citation>}}


### (7/149) FULL-W2V: Fully Exploiting Data Reuse for W2V on GPU-Accelerated Systems (Thomas Randall et al., 2023)

{{<citation>}}

Thomas Randall, Tyler Allen, Rong Ge. (2023)  
**FULL-W2V: Fully Exploiting Data Reuse for W2V on GPU-Accelerated Systems**  

---
Primary Category: cs.LG  
Categories: I-2-7; D-1-3; G-4, cs-CL, cs-DC, cs-LG, cs.LG  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.07743v1)  

---


**ABSTRACT**  
Word2Vec remains one of the highly-impactful innovations in the field of Natural Language Processing (NLP) that represents latent grammatical and syntactical information in human text with dense vectors in a low dimension. Word2Vec has high computational cost due to the algorithm's inherent sequentiality, intensive memory accesses, and the large vocabularies it represents. While prior studies have investigated technologies to explore parallelism and improve memory system performance, they struggle to effectively gain throughput on powerful GPUs.   We identify memory data access and latency as the primary bottleneck in prior works on GPUs, which prevents highly optimized kernels from attaining the architecture's peak performance. We present a novel algorithm, FULL-W2V, which maximally exploits the opportunities for data reuse in the W2V algorithm and leverages GPU architecture and resources to reduce access to low memory levels and improve temporal locality. FULL-W2V is capable of reducing accesses to GPU global memory significantly, e.g., by more than 89\%, compared to prior state-of-the-art GPU implementations, resulting in significant performance improvement that scales across successive hardware generations. Our prototype implementation achieves 2.97X speedup when ported from Nvidia Pascal P100 to Volta V100 cards, and outperforms the state-of-the-art by 5.72X on V100 cards with the same embedding quality. In-depth analysis indicates that the reduction of memory accesses through register and shared memory caching and high-throughput shared memory reduction leads to a significantly improved arithmetic intensity. FULL-W2V can potentially benefit many applications in NLP and other domains.

{{</citation>}}


### (8/149) Hierarchical Classification of Financial Transactions Through Context-Fusion of Transformer-based Embeddings and Taxonomy-aware Attention Layer (Antonio J. G. Busson et al., 2023)

{{<citation>}}

Antonio J. G. Busson, Rafael Rocha, Rennan Gaio, Rafael Miceli, Ivan Pereira, Daniel de S. Moraes, Sérgio Colcher, Alvaro Veiga, Bruno Rizzi, Francisco Evangelista, Leandro Santos, Fellipe Marques, Marcos Rabaioli, Diego Feldberg, Debora Mattos, João Pasqua, Diogo Dias. (2023)  
**Hierarchical Classification of Financial Transactions Through Context-Fusion of Transformer-based Embeddings and Taxonomy-aware Attention Layer**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Embedding, Financial, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.07730v1)  

---


**ABSTRACT**  
This work proposes the Two-headed DragoNet, a Transformer-based model for hierarchical multi-label classification of financial transactions. Our model is based on a stack of Transformers encoder layers that generate contextual embeddings from two short textual descriptors (merchant name and business activity), followed by a Context Fusion layer and two output heads that classify transactions according to a hierarchical two-level taxonomy (macro and micro categories). Finally, our proposed Taxonomy-aware Attention Layer corrects predictions that break categorical hierarchy rules defined in the given taxonomy. Our proposal outperforms classical machine learning methods in experiments of macro-category classification by achieving an F1-score of 93\% on a card dataset and 95% on a current account dataset.

{{</citation>}}


### (9/149) A Perspective of Q-value Estimation on Offline-to-Online Reinforcement Learning (Yinmin Zhang et al., 2023)

{{<citation>}}

Yinmin Zhang, Jie Liu, Chuming Li, Yazhe Niu, Yaodong Yang, Yu Liu, Wanli Ouyang. (2023)  
**A Perspective of Q-value Estimation on Offline-to-Online Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.07685v1)  

---


**ABSTRACT**  
Offline-to-online Reinforcement Learning (O2O RL) aims to improve the performance of offline pretrained policy using only a few online samples. Built on offline RL algorithms, most O2O methods focus on the balance between RL objective and pessimism, or the utilization of offline and online samples. In this paper, from a novel perspective, we systematically study the challenges that remain in O2O RL and identify that the reason behind the slow improvement of the performance and the instability of online finetuning lies in the inaccurate Q-value estimation inherited from offline pretraining. Specifically, we demonstrate that the estimation bias and the inaccurate rank of Q-value cause a misleading signal for the policy update, making the standard offline RL algorithms, such as CQL and TD3-BC, ineffective in the online finetuning. Based on this observation, we address the problem of Q-value estimation by two techniques: (1) perturbed value update and (2) increased frequency of Q-value updates. The first technique smooths out biased Q-value estimation with sharp peaks, preventing early-stage policy exploitation of sub-optimal actions. The second one alleviates the estimation bias inherited from offline pretraining by accelerating learning. Extensive experiments on the MuJoco and Adroit environments demonstrate that the proposed method, named SO2, significantly alleviates Q-value estimation issues, and consistently improves the performance against the state-of-the-art methods by up to 83.1%.

{{</citation>}}


### (10/149) I Open at the Close: A Deep Reinforcement Learning Evaluation of Open Streets Initiatives (R. Teal Witter et al., 2023)

{{<citation>}}

R. Teal Witter, Lucas Rosenblatt. (2023)  
**I Open at the Close: A Deep Reinforcement Learning Evaluation of Open Streets Initiatives**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.07680v1)  

---


**ABSTRACT**  
The open streets initiative "opens" streets to pedestrians and bicyclists by closing them to cars and trucks. The initiative, adopted by many cities across North America, increases community space in urban environments. But could open streets also make cities safer and less congested? We study this question by framing the choice of which streets to open as a reinforcement learning problem. In order to simulate the impact of opening streets, we first compare models for predicting vehicle collisions given network and temporal data. We find that a recurrent graph neural network, leveraging the graph structure and the short-term temporal dependence of the data, gives the best predictive performance. Then, with the ability to simulate collisions and traffic, we frame a reinforcement learning problem to find which streets to open. We compare the streets in the NYC Open Streets program to those proposed by a Q-learning algorithm. We find that the streets proposed by the Q-learning algorithm have reliably better outcomes, while streets in the program have similar outcomes to randomly selected streets. We present our work as a step toward principally choosing which streets to open for safer and less congested cities. All our code and data are available on Github.

{{</citation>}}


### (11/149) Bayesian Online Learning for Consensus Prediction (Sam Showalter et al., 2023)

{{<citation>}}

Sam Showalter, Alex Boyd, Padhraic Smyth, Mark Steyvers. (2023)  
**Bayesian Online Learning for Consensus Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.07679v1)  

---


**ABSTRACT**  
Given a pre-trained classifier and multiple human experts, we investigate the task of online classification where model predictions are provided for free but querying humans incurs a cost. In this practical but under-explored setting, oracle ground truth is not available. Instead, the prediction target is defined as the consensus vote of all experts. Given that querying full consensus can be costly, we propose a general framework for online Bayesian consensus estimation, leveraging properties of the multivariate hypergeometric distribution. Based on this framework, we propose a family of methods that dynamically estimate expert consensus from partial feedback by producing a posterior over expert and model beliefs. Analyzing this posterior induces an interpretable trade-off between querying cost and classification performance. We demonstrate the efficacy of our framework against a variety of baselines on CIFAR-10H and ImageNet-16H, two large-scale crowdsourced datasets.

{{</citation>}}


### (12/149) A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems (Alexandre Duval et al., 2023)

{{<citation>}}

Alexandre Duval, Simon V. Mathis, Chaitanya K. Joshi, Victor Schmidt, Santiago Miret, Fragkiskos D. Malliaros, Taco Cohen, Pietro Lio, Yoshua Bengio, Michael Bronstein. (2023)  
**A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-QM, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.07511v1)  

---


**ABSTRACT**  
Recent advances in computational modelling of atomic systems, spanning molecules, proteins, and materials, represent them as geometric graphs with atoms embedded as nodes in 3D Euclidean space. In these graphs, the geometric attributes transform according to the inherent physical symmetries of 3D atomic systems, including rotations and translations in Euclidean space, as well as node permutations. In recent years, Geometric Graph Neural Networks have emerged as the preferred machine learning architecture powering applications ranging from protein structure prediction to molecular simulations and material generation. Their specificity lies in the inductive biases they leverage -- such as physical symmetries and chemical properties -- to learn informative representations of these geometric graphs. In this opinionated paper, we provide a comprehensive and self-contained overview of the field of Geometric GNNs for 3D atomic systems. We cover fundamental background material and introduce a pedagogical taxonomy of Geometric GNN architectures:(1) invariant networks, (2) equivariant networks in Cartesian basis, (3) equivariant networks in spherical basis, and (4) unconstrained networks. Additionally, we outline key datasets and application areas and suggest future research directions. The objective of this work is to present a structured perspective on the field, making it accessible to newcomers and aiding practitioners in gaining an intuition for its mathematical abstractions.

{{</citation>}}


### (13/149) BIRB: A Generalization Benchmark for Information Retrieval in Bioacoustics (Jenny Hamer et al., 2023)

{{<citation>}}

Jenny Hamer, Eleni Triantafillou, Bart van Merriënboer, Stefan Kahl, Holger Klinck, Tom Denton, Vincent Dumoulin. (2023)  
**BIRB: A Generalization Benchmark for Information Retrieval in Bioacoustics**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2312.07439v2)  

---


**ABSTRACT**  
The ability for a machine learning model to cope with differences in training and deployment conditions--e.g. in the presence of distribution shift or the generalization to new classes altogether--is crucial for real-world use cases. However, most empirical work in this area has focused on the image domain with artificial benchmarks constructed to measure individual aspects of generalization. We present BIRB, a complex benchmark centered on the retrieval of bird vocalizations from passively-recorded datasets given focal recordings from a large citizen science corpus available for training. We propose a baseline system for this collection of tasks using representation learning and a nearest-centroid search. Our thorough empirical evaluation and analysis surfaces open research directions, suggesting that BIRB fills the need for a more realistic and complex benchmark to drive progress on robustness to distribution shifts and generalization of ML models.

{{</citation>}}


### (14/149) How Well Does GPT-4V(ision) Adapt to Distribution Shifts? A Preliminary Investigation (Zhongyi Han et al., 2023)

{{<citation>}}

Zhongyi Han, Guanglin Zhou, Rundong He, Jindong Wang, Tailin Wu, Yilong Yin, Salman Khan, Lina Yao, Tongliang Liu, Kun Zhang. (2023)  
**How Well Does GPT-4V(ision) Adapt to Distribution Shifts? A Preliminary Investigation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.07424v2)  

---


**ABSTRACT**  
In machine learning, generalization against distribution shifts -- where deployment conditions diverge from the training scenarios -- is crucial, particularly in fields like climate modeling, biomedicine, and autonomous driving. The emergence of foundation models, distinguished by their extensive pretraining and task versatility, has led to an increased interest in their adaptability to distribution shifts. GPT-4V(ision) acts as the most advanced publicly accessible multimodal foundation model, with extensive applications across various domains, including anomaly detection, video understanding, image generation, and medical diagnosis. However, its robustness against data distributions remains largely underexplored. Addressing this gap, this study rigorously evaluates GPT-4V's adaptability and generalization capabilities in dynamic environments, benchmarking against prominent models like CLIP and LLaVA. We delve into GPT-4V's zero-shot generalization across 13 diverse datasets spanning natural, medical, and molecular domains. We further investigate its adaptability to controlled data perturbations and examine the efficacy of in-context learning as a tool to enhance its adaptation. Our findings delineate GPT-4V's capability boundaries in distribution shifts, shedding light on its strengths and limitations across various scenarios. Importantly, this investigation contributes to our understanding of how AI foundation models generalize to distribution shifts, offering pivotal insights into their adaptability and robustness. Code is publicly available at https://github.com/jameszhou-gl/gpt-4v-distribution-shift.

{{</citation>}}


### (15/149) ReRoGCRL: Representation-based Robustness in Goal-Conditioned Reinforcement Learning (Xiangyu Yin et al., 2023)

{{<citation>}}

Xiangyu Yin, Sihao Wu, Jiaxu Liu, Meng Fang, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan. (2023)  
**ReRoGCRL: Representation-based Robustness in Goal-Conditioned Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Augmentation, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.07392v1)  

---


**ABSTRACT**  
While Goal-Conditioned Reinforcement Learning (GCRL) has gained attention, its algorithmic robustness, particularly against adversarial perturbations, remains unexplored. Unfortunately, the attacks and robust representation training methods specifically designed for traditional RL are not so effective when applied to GCRL. To address this challenge, we propose the \textit{Semi-Contrastive Representation} attack, a novel approach inspired by the adversarial contrastive attack. Unlike existing attacks in RL, it only necessitates information from the policy function and can be seamlessly implemented during deployment. Furthermore, to mitigate the vulnerability of existing GCRL algorithms, we introduce \textit{Adversarial Representation Tactics}. This strategy combines \textit{Semi-Contrastive Adversarial Augmentation} with \textit{Sensitivity-Aware Regularizer}. It improves the adversarial robustness of the underlying agent against various types of perturbations. Extensive experiments validate the superior performance of our attack and defence mechanism across multiple state-of-the-art GCRL algorithms. Our tool {\bf ReRoGCRL} is available at \url{https://github.com/TrustAI/ReRoGCRL}.

{{</citation>}}


### (16/149) Privacy-Aware Energy Consumption Modeling of Connected Battery Electric Vehicles using Federated Learning (Sen Yan et al., 2023)

{{<citation>}}

Sen Yan, Hongyuan Fang, Ji Li, Tomas Ward, Noel O'Connor, Mingming Liu. (2023)  
**Privacy-Aware Energy Consumption Modeling of Connected Battery Electric Vehicles using Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG, physics-soc-ph  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.07371v1)  

---


**ABSTRACT**  
Battery Electric Vehicles (BEVs) are increasingly significant in modern cities due to their potential to reduce air pollution. Precise and real-time estimation of energy consumption for them is imperative for effective itinerary planning and optimizing vehicle systems, which can reduce driving range anxiety and decrease energy costs. As public awareness of data privacy increases, adopting approaches that safeguard data privacy in the context of BEV energy consumption modeling is crucial. Federated Learning (FL) is a promising solution mitigating the risk of exposing sensitive information to third parties by allowing local data to remain on devices and only sharing model updates with a central server. Our work investigates the potential of using FL methods, such as FedAvg, and FedPer, to improve BEV energy consumption prediction while maintaining user privacy. We conducted experiments using data from 10 BEVs under simulated real-world driving conditions. Our results demonstrate that the FedAvg-LSTM model achieved a reduction of up to 67.84\% in the MAE value of the prediction results. Furthermore, we explored various real-world scenarios and discussed how FL methods can be employed in those cases. Our findings show that FL methods can effectively improve the performance of BEV energy consumption prediction while maintaining user privacy.

{{</citation>}}


### (17/149) Complex Recurrent Spectral Network (Lorenzo Chicchi et al., 2023)

{{<citation>}}

Lorenzo Chicchi, Lorenzo Giambagli, Lorenzo Buffoni, Raffaele Marino, Duccio Fanelli. (2023)  
**Complex Recurrent Spectral Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07296v1)  

---


**ABSTRACT**  
This paper presents a novel approach to advancing artificial intelligence (AI) through the development of the Complex Recurrent Spectral Network ($\mathbb{C}$-RSN), an innovative variant of the Recurrent Spectral Network (RSN) model. The $\mathbb{C}$-RSN is designed to address a critical limitation in existing neural network models: their inability to emulate the complex processes of biological neural networks dynamically and accurately. By integrating key concepts from dynamical systems theory and leveraging principles from statistical mechanics, the $\mathbb{C}$-RSN model introduces localized non-linearity, complex fixed eigenvalues, and a distinct separation of memory and input processing functionalities. These features collectively enable the $\mathbb{C}$-RSN evolving towards a dynamic, oscillating final state that more closely mirrors biological cognition. Central to this work is the exploration of how the $\mathbb{C}$-RSN manages to capture the rhythmic, oscillatory dynamics intrinsic to biological systems, thanks to its complex eigenvalue structure and the innovative segregation of its linear and non-linear components. The model's ability to classify data through a time-dependent function, and the localization of information processing, is demonstrated with an empirical evaluation using the MNIST dataset. Remarkably, distinct items supplied as a sequential input yield patterns in time which bear the indirect imprint of the insertion order (and of the time of separation between contiguous insertions).

{{</citation>}}


### (18/149) Multi-Granularity Framework for Unsupervised Representation Learning of Time Series (Chengyang Ye et al., 2023)

{{<citation>}}

Chengyang Ye, Qiang Ma. (2023)  
**Multi-Granularity Framework for Unsupervised Representation Learning of Time Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2312.07248v1)  

---


**ABSTRACT**  
Representation learning plays a critical role in the analysis of time series data and has high practical value across a wide range of applications. including trend analysis, time series data retrieval and forecasting. In practice, data confusion is a significant issue as it can considerably impact the effectiveness and accuracy of data analysis, machine learning models and decision-making processes. In general, previous studies did not consider the variability at various levels of granularity, thus resulting in inadequate information utilization, which further exacerbated the issue of data confusion. This paper proposes an unsupervised framework to realize multi-granularity representation learning for time series. Specifically, we employed a cross-granularity transformer to develop an association between fine- and coarse-grained representations. In addition, we introduced a retrieval task as an unsupervised training task to learn the multi-granularity representation of time series. Moreover, a novel loss function was designed to obtain the comprehensive multi-granularity representation of the time series via unsupervised learning. The experimental results revealed that the proposed framework demonstrates significant advantages over alternative representation learning models.

{{</citation>}}


### (19/149) Beyond Expected Return: Accounting for Policy Reproducibility when Evaluating Reinforcement Learning Algorithms (Manon Flageat et al., 2023)

{{<citation>}}

Manon Flageat, Bryan Lim, Antoine Cully. (2023)  
**Beyond Expected Return: Accounting for Policy Reproducibility when Evaluating Reinforcement Learning Algorithms**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.07178v1)  

---


**ABSTRACT**  
Many applications in Reinforcement Learning (RL) usually have noise or stochasticity present in the environment. Beyond their impact on learning, these uncertainties lead the exact same policy to perform differently, i.e. yield different return, from one roll-out to another. Common evaluation procedures in RL summarise the consequent return distributions using solely the expected return, which does not account for the spread of the distribution. Our work defines this spread as the policy reproducibility: the ability of a policy to obtain similar performance when rolled out many times, a crucial property in some real-world applications. We highlight that existing procedures that only use the expected return are limited on two fronts: first an infinite number of return distributions with a wide range of performance-reproducibility trade-offs can have the same expected return, limiting its effectiveness when used for comparing policies; second, the expected return metric does not leave any room for practitioners to choose the best trade-off value for considered applications. In this work, we address these limitations by recommending the use of Lower Confidence Bound, a metric taken from Bayesian optimisation that provides the user with a preference parameter to choose a desired performance-reproducibility trade-off. We also formalise and quantify policy reproducibility, and demonstrate the benefit of our metrics using extensive experiments of popular RL algorithms on common uncertain RL tasks.

{{</citation>}}


### (20/149) SE(3)-Invariant Multiparameter Persistent Homology for Chiral-Sensitive Molecular Property Prediction (Andac Demir et al., 2023)

{{<citation>}}

Andac Demir, Francis Prael III, Bulent Kiziltan. (2023)  
**SE(3)-Invariant Multiparameter Persistent Homology for Chiral-Sensitive Molecular Property Prediction**  

---
Primary Category: cs.LG  
Categories: cs-CG, cs-LG, cs.LG, math-AT  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.07633v1)  

---


**ABSTRACT**  
In this study, we present a novel computational method for generating molecular fingerprints using multiparameter persistent homology (MPPH). This technique holds considerable significance for drug discovery and materials science, where precise molecular property prediction is vital. By integrating SE(3)-invariance with Vietoris-Rips persistent homology, we effectively capture the three-dimensional representations of molecular chirality. This non-superimposable mirror image property directly influences the molecular interactions, serving as an essential factor in molecular property prediction. We explore the underlying topologies and patterns in molecular structures by applying Vietoris-Rips persistent homology across varying scales and parameters such as atomic weight, partial charge, bond type, and chirality. Our method's efficacy can be improved by incorporating additional parameters such as aromaticity, orbital hybridization, bond polarity, conjugated systems, as well as bond and torsion angles. Additionally, we leverage Stochastic Gradient Langevin Boosting in a Bayesian ensemble of GBDTs to obtain aleatoric and epistemic uncertainty estimates for gradient boosting models. With these uncertainty estimates, we prioritize high-uncertainty samples for active learning and model fine-tuning, benefiting scenarios where data labeling is costly or time consuming. Compared to conventional GNNs which usually suffer from oversmoothing and oversquashing, MPPH provides a more comprehensive and interpretable characterization of molecular data topology. We substantiate our approach with theoretical stability guarantees and demonstrate its superior performance over existing state-of-the-art methods in predicting molecular properties through extensive evaluations on the MoleculeNet benchmark datasets.

{{</citation>}}


### (21/149) Toward Robustness in Multi-label Classification: A Data Augmentation Strategy against Imbalance and Noise (Hwanjun Song et al., 2023)

{{<citation>}}

Hwanjun Song, Minseok Kim, Jae-Gil Lee. (2023)  
**Toward Robustness in Multi-label Classification: A Data Augmentation Strategy against Imbalance and Noise**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.07087v1)  

---


**ABSTRACT**  
Multi-label classification poses challenges due to imbalanced and noisy labels in training data. We propose a unified data augmentation method, named BalanceMix, to address these challenges. Our approach includes two samplers for imbalanced labels, generating minority-augmented instances with high diversity. It also refines multi-labels at the label-wise granularity, categorizing noisy labels as clean, re-labeled, or ambiguous for robust optimization. Extensive experiments on three benchmark datasets demonstrate that BalanceMix outperforms existing state-of-the-art methods. We release the code at https://github.com/DISL-Lab/BalanceMix.

{{</citation>}}


### (22/149) Focus on Hiders: Exploring Hidden Threats for Enhancing Adversarial Training (Qian Li et al., 2023)

{{<citation>}}

Qian Li, Yuxiao Hu, Yinpeng Dong, Dongxiao Zhang, Yuntian Chen. (2023)  
**Focus on Hiders: Exploring Hidden Threats for Enhancing Adversarial Training**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG, stat-AP  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2312.07067v1)  

---


**ABSTRACT**  
Adversarial training is often formulated as a min-max problem, however, concentrating only on the worst adversarial examples causes alternating repetitive confusion of the model, i.e., previously defended or correctly classified samples are not defensible or accurately classifiable in subsequent adversarial training. We characterize such non-ignorable samples as "hiders", which reveal the hidden high-risk regions within the secure area obtained through adversarial training and prevent the model from finding the real worst cases. We demand the model to prevent hiders when defending against adversarial examples for improving accuracy and robustness simultaneously. By rethinking and redefining the min-max optimization problem for adversarial training, we propose a generalized adversarial training algorithm called Hider-Focused Adversarial Training (HFAT). HFAT introduces the iterative evolution optimization strategy to simplify the optimization problem and employs an auxiliary model to reveal hiders, effectively combining the optimization directions of standard adversarial training and prevention hiders. Furthermore, we introduce an adaptive weighting mechanism that facilitates the model in adaptively adjusting its focus between adversarial examples and hiders during different training periods. We demonstrate the effectiveness of our method based on extensive experiments, and ensure that HFAT can provide higher robustness and accuracy.

{{</citation>}}


### (23/149) Rethinking Compression: Reduced Order Modelling of Latent Features in Large Language Models (Arnav Chavan et al., 2023)

{{<citation>}}

Arnav Chavan, Nahush Lele, Deepak Gupta. (2023)  
**Rethinking Compression: Reduced Order Modelling of Latent Features in Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07046v1)  

---


**ABSTRACT**  
Due to the substantial scale of Large Language Models (LLMs), the direct application of conventional compression methodologies proves impractical. The computational demands associated with even minimal gradient updates present challenges, particularly on consumer-grade hardware. This paper introduces an innovative approach for the parametric and practical compression of LLMs based on reduced order modelling, which entails low-rank decomposition within the feature space and re-parameterization in the weight space. Notably, this compression technique operates in a layer-wise manner, obviating the need for a GPU device and enabling the compression of billion-scale models within stringent constraints of both memory and time. Our method represents a significant advancement in model compression by leveraging matrix decomposition, demonstrating superior efficacy compared to the prevailing state-of-the-art structured pruning method.

{{</citation>}}


### (24/149) AI Control: Improving Safety Despite Intentional Subversion (Ryan Greenblatt et al., 2023)

{{<citation>}}

Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, Fabien Roger. (2023)  
**AI Control: Improving Safety Despite Intentional Subversion**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.06942v1)  

---


**ABSTRACT**  
As large language models (LLMs) become more powerful and are deployed more autonomously, it will be increasingly important to prevent them from causing harmful outcomes. Researchers have investigated a variety of safety techniques for this purpose, e.g. using models to review the outputs of other models, or red-teaming techniques to surface subtle failure modes. However, researchers have not evaluated whether such techniques still ensure safety if the model is itself intentionally trying to subvert them. In this paper, we develop and evaluate pipelines of safety techniques ("protocols") that are robust to intentional subversion.   We investigate a scenario in which we want to solve a sequence of programming problems, using access to a powerful but untrusted model (in our case, GPT-4), access to a less powerful trusted model (in our case, GPT-3.5), and limited access to human contractors who provide high-quality trusted labor. We investigate protocols that aim to never submit solutions containing backdoors, which we operationalize here as logical errors that are not caught by test cases.   We investigate a range of protocols and test each against strategies that the untrusted model could use to subvert them. One protocol is what we call trusted editing. This protocol first asks GPT-4 to write code, and then asks GPT-3.5 to rate the suspiciousness of that code. If the code is below some suspiciousness threshold, it is submitted. Otherwise, GPT-3.5 edits the solution to remove parts that seem suspicious and then submits the edited code. Another protocol is untrusted monitoring. This protocol asks GPT-4 to write code, and then asks another instance of GPT-4 whether the code is backdoored, using various techniques to prevent the GPT-4 instances from colluding. These protocols improve substantially on simple baselines.

{{</citation>}}


### (25/149) Humans vs Large Language Models: Judgmental Forecasting in an Era of Advanced AI (MAhdi Abolghasemi et al., 2023)

{{<citation>}}

MAhdi Abolghasemi, Odkhishig Ganbold, Kristian Rotaru. (2023)  
**Humans vs Large Language Models: Judgmental Forecasting in an Era of Advanced AI**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.06941v1)  

---


**ABSTRACT**  
This study investigates the forecasting accuracy of human experts versus Large Language Models (LLMs) in the retail sector, particularly during standard and promotional sales periods. Utilizing a controlled experimental setup with 123 human forecasters and five LLMs, including ChatGPT4, ChatGPT3.5, Bard, Bing, and Llama2, we evaluated forecasting precision through Mean Absolute Percentage Error. Our analysis centered on the effect of the following factors on forecasters performance: the supporting statistical model (baseline and advanced), whether the product was on promotion, and the nature of external impact. The findings indicate that LLMs do not consistently outperform humans in forecasting accuracy and that advanced statistical forecasting models do not uniformly enhance the performance of either human forecasters or LLMs. Both human and LLM forecasters exhibited increased forecasting errors, particularly during promotional periods and under the influence of positive external impacts. Our findings call for careful consideration when integrating LLMs into practical forecasting processes.

{{</citation>}}


### (26/149) Can a Transformer Represent a Kalman Filter? (Gautam Goel et al., 2023)

{{<citation>}}

Gautam Goel, Peter Bartlett. (2023)  
**Can a Transformer Represent a Kalman Filter?**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.06937v2)  

---


**ABSTRACT**  
Transformers are a class of autoregressive deep learning architectures which have recently achieved state-of-the-art performance in various vision, language, and robotics tasks. We revisit the problem of Kalman Filtering in linear dynamical systems and show that Transformers can approximate the Kalman Filter in a strong sense. Specifically, for any observable LTI system we construct an explicit causally-masked Transformer which implements the Kalman Filter, up to a small additive error which is bounded uniformly in time; we call our construction the Transformer Filter. Our construction is based on a two-step reduction. We first show that a softmax self-attention block can exactly represent a certain Gaussian kernel smoothing estimator. We then show that this estimator closely approximates the Kalman Filter. We also investigate how the Transformer Filter can be used for measurement-feedback control and prove that the resulting nonlinear controllers closely approximate the performance of standard optimal control policies such as the LQG controller.

{{</citation>}}


### (27/149) Perseus: Removing Energy Bloat from Large Model Training (Jae-Won Chung et al., 2023)

{{<citation>}}

Jae-Won Chung, Yile Gu, Insu Jang, Luoxi Meng, Nikhil Bansal, Mosharaf Chowdhury. (2023)  
**Perseus: Removing Energy Bloat from Large Model Training**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2312.06902v1)  

---


**ABSTRACT**  
Training large AI models on numerous GPUs consumes a massive amount of energy. We observe that not all energy consumed during training directly contributes to end-to-end training throughput, and a significant portion can be removed without slowing down training, which we call energy bloat.   In this work, we identify two independent sources of energy bloat in large model training, intrinsic and extrinsic, and propose Perseus, a unified optimization framework that mitigates both. Perseus obtains the "iteration time-energy" Pareto frontier of any large model training job using an efficient iterative graph cut-based algorithm and schedules energy consumption of its forward and backward computations across time to remove intrinsic and extrinsic energy bloat. Evaluation on large models like GPT-3 and Bloom shows that Perseus reduces energy consumption of large model training by up to 30%, enabling savings otherwise unobtainable before.

{{</citation>}}


## cs.CL (22)



### (28/149) Sentiment analysis in Tourism: Fine-tuning BERT or sentence embeddings concatenation? (Ibrahim Bouabdallaoui et al., 2023)

{{<citation>}}

Ibrahim Bouabdallaoui, Fatima Guerouate, Samya Bouhaddour, Chaimae Saadi, Mohammed Sbihi. (2023)  
**Sentiment analysis in Tourism: Fine-tuning BERT or sentence embeddings concatenation?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Named Entity Recognition, Natural Language Processing, Sentiment Analysis, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.07797v1)  

---


**ABSTRACT**  
Undoubtedly that the Bidirectional Encoder representations from Transformers is the most powerful technique in making Natural Language Processing tasks such as Named Entity Recognition, Question & Answers or Sentiment Analysis, however, the use of traditional techniques remains a major potential for the improvement of recent models, in particular word tokenization techniques and embeddings, but also the improvement of neural network architectures which are now the core of each architecture. recent. In this paper, we conduct a comparative study between Fine-Tuning the Bidirectional Encoder Representations from Transformers and a method of concatenating two embeddings to boost the performance of a stacked Bidirectional Long Short-Term Memory-Bidirectional Gated Recurrent Units model; these two approaches are applied in the context of sentiment analysis of shopping places in Morocco. A search for the best learning rate was made at the level of the two approaches, and a comparison of the best optimizers was made for each sentence embedding combination with regard to the second approach.

{{</citation>}}


### (29/149) BaRDa: A Belief and Reasoning Dataset that Separates Factual Accuracy and Reasoning Ability (Peter Clark et al., 2023)

{{<citation>}}

Peter Clark, Bhavana Dalvi Mishra, Oyvind Tafjord. (2023)  
**BaRDa: A Belief and Reasoning Dataset that Separates Factual Accuracy and Reasoning Ability**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.07527v1)  

---


**ABSTRACT**  
While there are numerous benchmarks comparing the performance of modern language models (LMs), end-task evaluations often conflate notions of *factual accuracy* ("truth") and *reasoning ability* ("rationality", or "honesty" in the sense of correctly reporting implications of beliefs). Our goal is a dataset that clearly distinguishes these two notions. Our approach is to leverage and extend a collection of human-annotated *entailment trees*, engineered to express both good and bad chains of reasoning, and using a mixture of true and false facts, in particular including counterfactual examples, to avoid belief bias (also known as the "content effect"). The resulting dataset, called BaRDa, contains 3000 entailments (1787 valid, 1213 invalid), using 6681 true and 2319 false statements. Testing on four GPT-series models, GPT3(curie)/GPT3(davinici)/3.5/4, we find factual accuracy (truth) scores of 74.1/80.6/82.6/87.1 and reasoning accuracy scores of 63.1/78.0/71.8/79.2. This shows the clear progression of models towards improved factual accuracy and entailment reasoning, and the dataset provides a new benchmark that more cleanly separates and quantifies these two notions.

{{</citation>}}


### (30/149) SocialStigmaQA: A Benchmark to Uncover Stigma Amplification in Generative Language Models (Manish Nagireddy et al., 2023)

{{<citation>}}

Manish Nagireddy, Lamogha Chiazor, Moninder Singh, Ioana Baldini. (2023)  
**SocialStigmaQA: A Benchmark to Uncover Stigma Amplification in Generative Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.07492v2)  

---


**ABSTRACT**  
Current datasets for unwanted social bias auditing are limited to studying protected demographic features such as race and gender. In this work, we introduce a comprehensive benchmark that is meant to capture the amplification of social bias, via stigmas, in generative language models. We start with a comprehensive list of 93 stigmas documented in social science literature and curate a question-answering (QA) dataset which involves simple social situations. Our benchmark, SocialStigmaQA, contains roughly 10K prompts, with a variety of prompt styles, carefully constructed to systematically test for both social bias and model robustness. We present results for SocialStigmaQA with two widely used open source generative language models and we demonstrate that the output generated by these models considerably amplifies existing social bias against stigmatized groups. Specifically, we find that the proportion of socially biased output ranges from 45% to 59% across a variety of decoding strategies and prompting styles. We discover that the deliberate design of the templates in our benchmark (e.g., by adding biasing text to the prompt or varying the answer that indicates bias) impact the model tendencies to generate socially biased output. Additionally, we report on patterns in the generated chain-of-thought output, finding a variety of problems from subtle bias to evidence of a lack of reasoning.   Warning: This paper contains examples of text which is toxic, biased, and harmful.

{{</citation>}}


### (31/149) Comparable Demonstrations are Important in In-Context Learning: A Novel Perspective on Demonstration Selection (Caoyun Fan et al., 2023)

{{<citation>}}

Caoyun Fan, Jidong Tian, Yitian Li, Hao He, Yaohui Jin. (2023)  
**Comparable Demonstrations are Important in In-Context Learning: A Novel Perspective on Demonstration Selection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07476v1)  

---


**ABSTRACT**  
In-Context Learning (ICL) is an important paradigm for adapting Large Language Models (LLMs) to downstream tasks through a few demonstrations. Despite the great success of ICL, the limitation of the demonstration number may lead to demonstration bias, i.e. the input-label mapping induced by LLMs misunderstands the task's essence. Inspired by human experience, we attempt to mitigate such bias through the perspective of the inter-demonstration relationship. Specifically, we construct Comparable Demonstrations (CDs) by minimally editing the texts to flip the corresponding labels, in order to highlight the task's essence and eliminate potential spurious correlations through the inter-demonstration comparison. Through a series of experiments on CDs, we find that (1) demonstration bias does exist in LLMs, and CDs can significantly reduce such bias; (2) CDs exhibit good performance in ICL, especially in out-of-distribution scenarios. In summary, this study explores the ICL mechanisms from a novel perspective, providing a deeper insight into the demonstration selection strategy for ICL.

{{</citation>}}


### (32/149) Towards Faster k-Nearest-Neighbor Machine Translation (Xiangyu Shi et al., 2023)

{{<citation>}}

Xiangyu Shi, Yunlong Liang, Jinan Xu, Yufeng Chen. (2023)  
**Towards Faster k-Nearest-Neighbor Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-NE, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.07419v1)  

---


**ABSTRACT**  
Recent works have proven the effectiveness of k-nearest-neighbor machine translation(a.k.a kNN-MT) approaches to produce remarkable improvement in cross-domain translations. However, these models suffer from heavy retrieve overhead on the entire datastore when decoding each token. We observe that during the decoding phase, about 67% to 84% of tokens are unvaried after searching over the corpus datastore, which means most of the tokens cause futile retrievals and introduce unnecessary computational costs by initiating k-nearest-neighbor searches. We consider this phenomenon is explainable in linguistics and propose a simple yet effective multi-layer perceptron (MLP) network to predict whether a token should be translated jointly by the neural machine translation model and probabilities produced by the kNN or just by the neural model. The results show that our method succeeds in reducing redundant retrieval operations and significantly reduces the overhead of kNN retrievals by up to 53% at the expense of a slight decline in translation quality. Moreover, our method could work together with all existing kNN-MT systems.

{{</citation>}}


### (33/149) Large Language Models are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales (Taeyoon Kwon et al., 2023)

{{<citation>}}

Taeyoon Kwon, Kai Tzu-iunn Ong, Dongjin Kang, Seungjun Moon, Jeong Ryong Lee, Dosik Hwang, Yongsik Sim, Beomseok Sohn, Dongha Lee, Jinyoung Yeo. (2023)  
**Large Language Models are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Clinical, Language Model, NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.07399v1)  

---


**ABSTRACT**  
Machine reasoning has made great progress in recent years owing to large language models (LLMs). In the clinical domain, however, most NLP-driven projects mainly focus on clinical classification or reading comprehension, and under-explore clinical reasoning for disease diagnosis due to the expensive rationale annotation with clinicians. In this work, we present a ``reasoning-aware'' diagnosis framework that rationalizes the diagnostic process via prompt-based learning in a time- and labor-efficient manner, and learns to reason over the prompt-generated rationales. Specifically, we address the clinical reasoning for disease diagnosis, where the LLM generates diagnostic rationales providing its insight on presented patient data and the reasoning path towards the diagnosis, namely Clinical Chain-of-Thought (Clinical CoT). We empirically demonstrate LLMs/LMs' ability of clinical reasoning via extensive experiments and analyses on both rationale generation and disease diagnosis in various settings. We further propose a novel set of criteria for evaluating machine-generated rationales' potential for real-world clinical settings, facilitating and benefiting future research in this area.

{{</citation>}}


### (34/149) Self-supervised Adaptive Pre-training of Multilingual Speech Models for Language and Dialect Identification (Mohammed Maqsood Shaik et al., 2023)

{{<citation>}}

Mohammed Maqsood Shaik, Dietrich Klakow, Badr M. Abdullah. (2023)  
**Self-supervised Adaptive Pre-training of Multilingual Speech Models for Language and Dialect Identification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Multilingual, Transformer  
[Paper Link](http://arxiv.org/abs/2312.07338v1)  

---


**ABSTRACT**  
Pre-trained Transformer-based speech models have shown striking performance when fine-tuned on various downstream tasks such as automatic speech recognition and spoken language identification (SLID). However, the problem of domain mismatch remains a challenge in this area, where the domain of the pre-training data might differ from that of the downstream labeled data used for fine-tuning. In multilingual tasks such as SLID, the pre-trained speech model may not support all the languages in the downstream task. To address this challenge, we propose self-supervised adaptive pre-training (SAPT) to adapt the pre-trained model to the target domain and languages of the downstream task. We apply SAPT to the XLSR-128 model and investigate the effectiveness of this approach for the SLID task. First, we demonstrate that SAPT improves XLSR performance on the FLEURS benchmark with substantial gains up to 40.1% for under-represented languages. Second, we apply SAPT on four different datasets in a few-shot learning setting, showing that our approach improves the sample efficiency of XLSR during fine-tuning. Our experiments provide strong empirical evidence that continual adaptation via self-supervision improves downstream performance for multilingual speech models.

{{</citation>}}


### (35/149) SCCA: Shifted Cross Chunk Attention for long contextual semantic expansion (Yuxiang Guo, 2023)

{{<citation>}}

Yuxiang Guo. (2023)  
**SCCA: Shifted Cross Chunk Attention for long contextual semantic expansion**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, LLaMA  
[Paper Link](http://arxiv.org/abs/2312.07305v1)  

---


**ABSTRACT**  
Sparse attention as a efficient method can significantly decrease the computation cost, but current sparse attention tend to rely on window self attention which block the global information flow. For this problem, we present Shifted Cross Chunk Attention (SCCA), using different KV shifting strategy to extend respective field in each attention layer. Except, we combine Dilated Attention(DA) and Dilated Neighborhood Attention(DNA) to present Shifted Dilated Attention(SDA). Both SCCA and SDA can accumulate attention results in multi head attention to obtain approximate respective field in full attention. In this paper, we conduct language modeling experiments using different pattern of SCCA and combination of SCCA and SDA. The proposed shifted cross chunk attention (SCCA) can effectively extend large language models (LLMs) to longer context combined with Positional interpolation(PI) and LoRA than current sparse attention. Notably, SCCA adopts LLaMA2 7B from 4k context to 8k in single V100. This attention pattern can provide a Plug-and-play fine-tuning method to extend model context while retaining their original architectures, and is compatible with most existing techniques.

{{</citation>}}


### (36/149) Towards Equipping Transformer with the Ability of Systematic Compositionality (Chen Huang et al., 2023)

{{<citation>}}

Chen Huang, Peixin Qin, Wenqiang Lei, Jiancheng Lv. (2023)  
**Towards Equipping Transformer with the Ability of Systematic Compositionality**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.07280v1)  

---


**ABSTRACT**  
One of the key factors in language productivity and human cognition is the ability of systematic compositionality, which refers to understanding composed unseen examples of seen primitives. However, recent evidence reveals that the Transformers have difficulty generalizing the composed context based on the seen primitives. To this end, we take the first step to propose a compositionality-aware Transformer called CAT and two novel pre-training tasks to facilitate systematic compositionality. We tentatively provide a successful implementation of a multi-layer CAT on the basis of the especially popular BERT. The experimental results demonstrate that CAT outperforms baselines on compositionality-aware tasks with minimal impact on the effectiveness on standardized language understanding tasks.

{{</citation>}}


### (37/149) The GUA-Speech System Description for CNVSRC Challenge 2023 (Shengqiang Li et al., 2023)

{{<citation>}}

Shengqiang Li, Chao Lei, Baozhong Ma, Binbin Zhang, Fuping Pan. (2023)  
**The GUA-Speech System Description for CNVSRC Challenge 2023**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.07254v1)  

---


**ABSTRACT**  
This study describes our system for Task 1 Single-speaker Visual Speech Recognition (VSR) fixed track in the Chinese Continuous Visual Speech Recognition Challenge (CNVSRC) 2023. Specifically, we use intermediate connectionist temporal classification (Inter CTC) residual modules to relax the conditional independence assumption of CTC in our model. Then we use a bi-transformer decoder to enable the model to capture both past and future contextual information. In addition, we use Chinese characters as the modeling units to improve the recognition accuracy of our model. Finally, we use a recurrent neural network language model (RNNLM) for shallow fusion in the inference stage. Experiments show that our system achieves a character error rate (CER) of 38.09% on the Eval set which reaches a relative CER reduction of 21.63% over the official baseline, and obtains a second place in the challenge.

{{</citation>}}


### (38/149) Neural Machine Translation of Clinical Text: An Empirical Investigation into Multilingual Pre-Trained Language Models and Transfer-Learning (Lifeng Han et al., 2023)

{{<citation>}}

Lifeng Han, Serge Gladkoff, Gleb Erofeev, Irina Sorokina, Betty Galiano, Goran Nenadic. (2023)  
**Neural Machine Translation of Clinical Text: An Empirical Investigation into Multilingual Pre-Trained Language Models and Transfer-Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Clinical, Language Model, Machine Translation, Multilingual, Transformer  
[Paper Link](http://arxiv.org/abs/2312.07250v1)  

---


**ABSTRACT**  
We conduct investigations on clinical text machine translation by examining multilingual neural network models using deep learning such as Transformer based structures. Furthermore, to address the language resource imbalance issue, we also carry out experiments using a transfer learning methodology based on massive multilingual pre-trained language models (MMPLMs). The experimental results on three subtasks including 1) clinical case (CC), 2) clinical terminology (CT), and 3) ontological concept (OC) show that our models achieved top-level performances in the ClinSpEn-2022 shared task on English-Spanish clinical domain data. Furthermore, our expert-based human evaluations demonstrate that the small-sized pre-trained language model (PLM) won over the other two extra-large language models by a large margin, in the clinical domain fine-tuning, which finding was never reported in the field. Finally, the transfer learning method works well in our experimental setting using the WMT21fb model to accommodate a new language space Spanish that was not seen at the pre-training stage within WMT21fb itself, which deserves more exploitation for clinical knowledge transformation, e.g. to investigate into more languages. These research findings can shed some light on domain-specific machine translation development, especially in clinical and healthcare fields. Further research projects can be carried out based on our work to improve healthcare text analytics and knowledge transformation.

{{</citation>}}


### (39/149) Multilingual large language models leak human stereotypes across language boundaries (Yang Trista Cao et al., 2023)

{{<citation>}}

Yang Trista Cao, Anna Sotnikova, Jieyu Zhao, Linda X. Zou, Rachel Rudinger, Hal Daume III. (2023)  
**Multilingual large language models leak human stereotypes across language boundaries**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT, Multilingual, T5  
[Paper Link](http://arxiv.org/abs/2312.07141v1)  

---


**ABSTRACT**  
Multilingual large language models have been increasingly popular for their proficiency in comprehending and generating text across various languages. Previous research has shown that the presence of stereotypes and biases in monolingual large language models can be attributed to the nature of their training data, which is collected from humans and reflects societal biases. Multilingual language models undergo the same training procedure as monolingual ones, albeit with training data sourced from various languages. This raises the question: do stereotypes present in one social context leak across languages within the model? In our work, we first define the term ``stereotype leakage'' and propose a framework for its measurement. With this framework, we investigate how stereotypical associations leak across four languages: English, Russian, Chinese, and Hindi. To quantify the stereotype leakage, we employ an approach from social psychology, measuring stereotypes via group-trait associations. We evaluate human stereotypes and stereotypical associations manifested in multilingual large language models such as mBERT, mT5, and ChatGPT. Our findings show a noticeable leakage of positive, negative, and non-polar associations across all languages. Notably, Hindi within multilingual models appears to be the most susceptible to influence from other languages, while Chinese is the least. Additionally, ChatGPT exhibits a better alignment with human scores than other models.

{{</citation>}}


### (40/149) BED: Bi-Encoder-Decoder Model for Canonical Relation Extraction (Nantao Zheng et al., 2023)

{{<citation>}}

Nantao Zheng, Siyu Long, Xinyu Dai. (2023)  
**BED: Bi-Encoder-Decoder Model for Canonical Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2312.07088v1)  

---


**ABSTRACT**  
Canonical relation extraction aims to extract relational triples from sentences, where the triple elements (entity pairs and their relationship) are mapped to the knowledge base. Recently, methods based on the encoder-decoder architecture are proposed and achieve promising results. However, these methods cannot well utilize the entity information, which is merely used as augmented training data. Moreover, they are incapable of representing novel entities, since no embeddings have been learned for them. In this paper, we propose a novel framework, Bi-Encoder-Decoder (BED), to solve the above issues. Specifically, to fully utilize entity information, we employ an encoder to encode semantics of this information, leading to high-quality entity representations. For novel entities, given a trained entity encoder, their representations can be easily generated. Experimental results on two datasets show that, our method achieves a significant performance improvement over the previous state-of-the-art and handle novel entities well without retraining.

{{</citation>}}


### (41/149) Context Matter: Data-Efficient Augmentation of Large Language Models for Scientific Applications (Xiang Li et al., 2023)

{{<citation>}}

Xiang Li, Haoran Tang, Siyu Chen, Ziwei Wang, Anurag Maravi, Marcin Abram. (2023)  
**Context Matter: Data-Efficient Augmentation of Large Language Models for Scientific Applications**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Augmentation, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.07069v1)  

---


**ABSTRACT**  
In this paper, we explore the challenges inherent to Large Language Models (LLMs) like GPT-4, particularly their propensity for hallucinations, logic mistakes, and incorrect conclusions when tasked with answering complex questions. The capacity of LLMs to present erroneous answers in a coherent and semantically rigorous manner further complicates the detection of factual inaccuracies. This issue is especially pronounced in fields that require specialized expertise. Our work delves into these challenges, aiming to enhance the understanding and mitigation of such errors, thereby contributing to the improvement of LLM accuracy and reliability in scientific and other specialized domains. Our findings reveal a non-linear relationship between the context's relevancy and the answers' measured quality. In addition, we demonstrate that with the correct calibration, it is possible to automate the grading procedure -- a finding suggesting that, at least to some degree, the LLMs can be used to self-examine the quality of their own performance. Finally, we describe an experimental platform that can be seen as a proof-of-concept of the techniques described in this work.

{{</citation>}}


### (42/149) DiffuVST: Narrating Fictional Scenes with Global-History-Guided Denoising Models (Shengguang Wu et al., 2023)

{{<citation>}}

Shengguang Wu, Mei Yuan, Qi Su. (2023)  
**DiffuVST: Narrating Fictional Scenes with Global-History-Guided Denoising Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07066v1)  

---


**ABSTRACT**  
Recent advances in image and video creation, especially AI-based image synthesis, have led to the production of numerous visual scenes that exhibit a high level of abstractness and diversity. Consequently, Visual Storytelling (VST), a task that involves generating meaningful and coherent narratives from a collection of images, has become even more challenging and is increasingly desired beyond real-world imagery. While existing VST techniques, which typically use autoregressive decoders, have made significant progress, they suffer from low inference speed and are not well-suited for synthetic scenes. To this end, we propose a novel diffusion-based system DiffuVST, which models the generation of a series of visual descriptions as a single conditional denoising process. The stochastic and non-autoregressive nature of DiffuVST at inference time allows it to generate highly diverse narratives more efficiently. In addition, DiffuVST features a unique design with bi-directional text history guidance and multimodal adapter modules, which effectively improve inter-sentence coherence and image-to-text fidelity. Extensive experiments on the story generation task covering four fictional visual-story datasets demonstrate the superiority of DiffuVST over traditional autoregressive models in terms of both text quality and inference speed.

{{</citation>}}


### (43/149) Improving Factual Error Correction by Learning to Inject Factual Errors (Xingwei He et al., 2023)

{{<citation>}}

Xingwei He, Qianru Zhang, A-Long Jin, Jun Ma, Yuan Yuan, Siu Ming Yiu. (2023)  
**Improving Factual Error Correction by Learning to Inject Factual Errors**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.07049v1)  

---


**ABSTRACT**  
Factual error correction (FEC) aims to revise factual errors in false claims with minimal editing, making them faithful to the provided evidence. This task is crucial for alleviating the hallucination problem encountered by large language models. Given the lack of paired data (i.e., false claims and their corresponding correct claims), existing methods typically adopt the mask-then-correct paradigm. This paradigm relies solely on unpaired false claims and correct claims, thus being referred to as distantly supervised methods. These methods require a masker to explicitly identify factual errors within false claims before revising with a corrector. However, the absence of paired data to train the masker makes accurately pinpointing factual errors within claims challenging. To mitigate this, we propose to improve FEC by Learning to Inject Factual Errors (LIFE), a three-step distantly supervised method: mask-corrupt-correct. Specifically, we first train a corruptor using the mask-then-corrupt procedure, allowing it to deliberately introduce factual errors into correct text. The corruptor is then applied to correct claims, generating a substantial amount of paired data. After that, we filter out low-quality data, and use the remaining data to train a corrector. Notably, our corrector does not require a masker, thus circumventing the bottleneck associated with explicit factual error identification. Our experiments on a public dataset verify the effectiveness of LIFE in two key aspects: Firstly, it outperforms the previous best-performing distantly supervised method by a notable margin of 10.59 points in SARI Final (19.3% improvement). Secondly, even compared to ChatGPT prompted with in-context examples, LIFE achieves a superiority of 7.16 points in SARI Final.

{{</citation>}}


### (44/149) Dynamic Corrective Self-Distillation for Better Fine-Tuning of Pretrained Models (Ibtihel Amara et al., 2023)

{{<citation>}}

Ibtihel Amara, Vinija Jain, Aman Chadha. (2023)  
**Dynamic Corrective Self-Distillation for Better Fine-Tuning of Pretrained Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GLUE  
[Paper Link](http://arxiv.org/abs/2312.07028v1)  

---


**ABSTRACT**  
We tackle the challenging issue of aggressive fine-tuning encountered during the process of transfer learning of pre-trained language models (PLMs) with limited labeled downstream data. This problem primarily results in a decline in performance on the subsequent task. Inspired by the adaptive boosting method in traditional machine learning, we present an effective dynamic corrective self-distillation (DCS) approach to improve the fine-tuning of the PLMs. Our technique involves performing a self-distillation mechanism where, at each iteration, the student model actively adapts and corrects itself by dynamically adjusting the weights assigned to individual data points. This iterative self-correcting process significantly enhances the overall fine-tuning capability of PLMs, leading to improved performance and robustness. We conducted comprehensive evaluations using the GLUE benchmark demonstrating the efficacy of our method in enhancing the fine-tuning process for various PLMs across diverse downstream tasks.

{{</citation>}}


### (45/149) Alignment for Honesty (Yuqing Yang et al., 2023)

{{<citation>}}

Yuqing Yang, Ethan Chern, Xipeng Qiu, Graham Neubig, Pengfei Liu. (2023)  
**Alignment for Honesty**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, NLP  
[Paper Link](http://arxiv.org/abs/2312.07000v1)  

---


**ABSTRACT**  
Recent research has made significant strides in applying alignment techniques to enhance the helpfulness and harmlessness of large language models (LLMs) in accordance with human intentions. In this paper, we argue for the importance of alignment for honesty, ensuring that LLMs proactively refuse to answer questions when they lack knowledge, while still not being overly conservative. However, a pivotal aspect of alignment for honesty involves discerning the limits of an LLM's knowledge, which is far from straightforward. This challenge demands comprehensive solutions in terms of metric development, benchmark creation, and training methodologies. In this paper, we address these challenges by first establishing a precise problem definition and defining ``honesty'' inspired by the Analects of Confucius. This serves as a cornerstone for developing metrics that effectively measure an LLM's honesty by quantifying its progress post-alignment. Furthermore, we introduce a flexible training framework which is further instantiated by several efficient fine-tuning techniques that emphasize honesty without sacrificing performance on other tasks. Our extensive experiments reveal that these aligned models show a marked increase in honesty, as indicated by our proposed metrics. We open-source a wealth of resources to facilitate future research at https://github.com/GAIR-NLP/alignment-for-honesty, including honesty-aligned models, training and evaluation datasets for honesty alignment, concept glossary, as well as all relevant source code.

{{</citation>}}


### (46/149) SM70: A Large Language Model for Medical Devices (Anubhav Bhatti et al., 2023)

{{<citation>}}

Anubhav Bhatti, Surajsinh Parmar, San Lee. (2023)  
**SM70: A Large Language Model for Medical Devices**  

---
Primary Category: cs.CL  
Categories: 68T50, cs-AI, cs-CL, cs.CL  
Keywords: Clinical, GPT, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.06974v1)  

---


**ABSTRACT**  
We are introducing SM70, a 70 billion-parameter Large Language Model that is specifically designed for SpassMed's medical devices under the brand name 'JEE1' (pronounced as G1 and means 'Life'). This large language model provides more accurate and safe responses to medical-domain questions. To fine-tune SM70, we used around 800K data entries from the publicly available dataset MedAlpaca. The Llama2 70B open-sourced model served as the foundation for SM70, and we employed the QLoRA technique for fine-tuning. The evaluation is conducted across three benchmark datasets - MEDQA - USMLE, PUBMEDQA, and USMLE - each representing a unique aspect of medical knowledge and reasoning. The performance of SM70 is contrasted with other notable LLMs, including Llama2 70B, Clinical Camel 70 (CC70), GPT 3.5, GPT 4, and Med-Palm, to provide a comparative understanding of its capabilities within the medical domain. Our results indicate that SM70 outperforms several established models in these datasets, showcasing its proficiency in handling a range of medical queries, from fact-based questions derived from PubMed abstracts to complex clinical decision-making scenarios. The robust performance of SM70, particularly in the USMLE and PUBMEDQA datasets, suggests its potential as an effective tool in clinical decision support and medical information retrieval. Despite its promising results, the paper also acknowledges the areas where SM70 lags behind the most advanced model, GPT 4, thereby highlighting the need for further development, especially in tasks demanding extensive medical knowledge and intricate reasoning.

{{</citation>}}


### (47/149) Content-Localization based Neural Machine Translation for Informal Dialectal Arabic: Spanish/French to Levantine/Gulf Arabic (Fatimah Alzamzami et al., 2023)

{{<citation>}}

Fatimah Alzamzami, Abdulmotaleb El Saddik. (2023)  
**Content-Localization based Neural Machine Translation for Informal Dialectal Arabic: Spanish/French to Levantine/Gulf Arabic**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.06926v1)  

---


**ABSTRACT**  
Resources in high-resource languages have not been efficiently exploited in low-resource languages to solve language-dependent research problems. Spanish and French are considered high resource languages in which an adequate level of data resources for informal online social behavior modeling, is observed. However, a machine translation system to access those data resources and transfer their context and tone to a low-resource language like dialectal Arabic, does not exist. In response, we propose a framework that localizes contents of high-resource languages to a low-resource language/dialects by utilizing AI power. To the best of our knowledge, we are the first work to provide a parallel translation dataset from/to informal Spanish and French to/from informal Arabic dialects. Using this, we aim to enrich the under-resource-status dialectal Arabic and fast-track the research of diverse online social behaviors within and across smart cities in different geo-regions. The experimental results have illustrated the capability of our proposed solution in exploiting the resources between high and low resource languages and dialects. Not only this, but it has also been proven that ignoring dialects within the same language could lead to misleading analysis of online social behavior.

{{</citation>}}


### (48/149) Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack (Yu Fu et al., 2023)

{{<citation>}}

Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong. (2023)  
**Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, NLP, QA, Summarization  
[Paper Link](http://arxiv.org/abs/2312.06924v1)  

---


**ABSTRACT**  
Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integraty of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.

{{</citation>}}


### (49/149) Mathematical Language Models: A Survey (Wentao Liu et al., 2023)

{{<citation>}}

Wentao Liu, Hanglei Hu, Jie Zhou, Yuyang Ding, Junsong Li, Jiayi Zeng, Mengliang He, Qin Chen, Bo Jiang, Aimin Zhou, Liang He. (2023)  
**Mathematical Language Models: A Survey**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07622v2)  

---


**ABSTRACT**  
In recent years, there has been remarkable progress in leveraging Language Models (LMs), encompassing Pre-trained Language Models (PLMs) and Large-scale Language Models (LLMs), within the domain of mathematics. This paper conducts a comprehensive survey of mathematical LMs, systematically categorizing pivotal research endeavors from two distinct perspectives: tasks and methodologies. The landscape reveals a large number of proposed mathematical LLMs, which are further delineated into instruction learning, tool-based methods, fundamental CoT techniques, and advanced CoT methodologies. In addition, our survey entails the compilation of over 60 mathematical datasets, including training datasets, benchmark datasets, and augmented datasets. Addressing the primary challenges and delineating future trajectories within the field of mathematical LMs, this survey is positioned as a valuable resource, poised to facilitate and inspire future innovation among researchers invested in advancing this domain.

{{</citation>}}


## cs.CR (5)



### (50/149) BarraCUDA: Bringing Electromagnetic Side Channel Into Play to Steal the Weights of Neural Networks from NVIDIA GPUs (Peter Horvath et al., 2023)

{{<citation>}}

Peter Horvath, Lukasz Chmielewski, Leo Weissbart, Lejla Batina, Yuval Yarom. (2023)  
**BarraCUDA: Bringing Electromagnetic Side Channel Into Play to Steal the Weights of Neural Networks from NVIDIA GPUs**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: ChatGPT, GPT, Google  
[Paper Link](http://arxiv.org/abs/2312.07783v1)  

---


**ABSTRACT**  
Over the last decade, applications of neural networks have spread to cover all aspects of life. A large number of companies base their businesses on building products that use neural networks for tasks such as face recognition, machine translation, and autonomous cars. They are being used in safety and security-critical applications like high definition maps and medical wristbands, or in globally used products like Google Translate and ChatGPT. Much of the intellectual property underpinning these products is encoded in the exact configuration of the neural networks. Consequently, protecting these is of utmost priority to businesses. At the same time, many of these products need to operate under a strong threat model, in which the adversary has unfettered physical control of the product.   Past work has demonstrated that with physical access, attackers can reverse engineer neural networks that run on scalar microcontrollers, like ARM Cortex M3. However, for performance reasons, neural networks are often implemented on highly-parallel general purpose graphics processing units (GPGPUs), and so far, attacks on these have only recovered course-grained information on the structure of the neural network, but failed to retrieve the weights and biases.   In this work, we present BarraCUDA, a novel attack on GPGPUs that can completely extract the parameters of neural networks. BarraCUDA uses correlation electromagnetic analysis to recover the weights and biases in the convolutional layers of neural networks. We use BarraCUDA to attack the popular NVIDIA Jetson Nano device, demonstrating successful parameter extraction of neural networks in a highly parallel and noisy environment.

{{</citation>}}


### (51/149) Real-time Network Intrusion Detection via Decision Transformers (Jingdi Chen et al., 2023)

{{<citation>}}

Jingdi Chen, Hanhan Zhou, Yongsheng Mei, Gina Adam, Nathaniel D. Bastian, Tian Lan. (2023)  
**Real-time Network Intrusion Detection via Decision Transformers**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Intrusion Detection, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.07696v1)  

---


**ABSTRACT**  
Many cybersecurity problems that require real-time decision-making based on temporal observations can be abstracted as a sequence modeling problem, e.g., network intrusion detection from a sequence of arriving packets. Existing approaches like reinforcement learning may not be suitable for such cybersecurity decision problems, since the Markovian property may not necessarily hold and the underlying network states are often not observable. In this paper, we cast the problem of real-time network intrusion detection as casual sequence modeling and draw upon the power of the transformer architecture for real-time decision-making. By conditioning a causal decision transformer on past trajectories, consisting of the rewards, network packets, and detection decisions, our proposed framework will generate future detection decisions to achieve the desired return. It enables decision transformers to be applied to real-time network intrusion detection, as well as a novel tradeoff between the accuracy and timeliness of detection. The proposed solution is evaluated on public network intrusion detection datasets and outperforms several baseline algorithms using reinforcement learning and sequence modeling, in terms of detection accuracy and timeliness.

{{</citation>}}


### (52/149) EdgePruner: Poisoned Edge Pruning in Graph Contrastive Learning (Hiroya Kato et al., 2023)

{{<citation>}}

Hiroya Kato, Kento Hasegawa, Seira Hidano, Kazuhide Fukushima. (2023)  
**EdgePruner: Poisoned Edge Pruning in Graph Contrastive Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Contrastive Learning, Pruning  
[Paper Link](http://arxiv.org/abs/2312.07022v1)  

---


**ABSTRACT**  
Graph Contrastive Learning (GCL) is unsupervised graph representation learning that can obtain useful representation of unknown nodes. The node representation can be utilized as features of downstream tasks. However, GCL is vulnerable to poisoning attacks as with existing learning models. A state-of-the-art defense cannot sufficiently negate adverse effects by poisoned graphs although such a defense introduces adversarial training in the GCL. To achieve further improvement, pruning adversarial edges is important. To the best of our knowledge, the feasibility remains unexplored in the GCL domain. In this paper, we propose a simple defense for GCL, EdgePruner. We focus on the fact that the state-of-the-art poisoning attack on GCL tends to mainly add adversarial edges to create poisoned graphs, which means that pruning edges is important to sanitize the graphs. Thus, EdgePruner prunes edges that contribute to minimizing the contrastive loss based on the node representation obtained after training on poisoned graphs by GCL. Furthermore, we focus on the fact that nodes with distinct features are connected by adversarial edges in poisoned graphs. Thus, we introduce feature similarity between neighboring nodes to help more appropriately determine adversarial edges. This similarity is helpful in further eliminating adverse effects from poisoned graphs on various datasets. Finally, EdgePruner outputs a graph that yields the minimum contrastive loss as the sanitized graph. Our results demonstrate that pruning adversarial edges is feasible on six datasets. EdgePruner can improve the accuracy of node classification under the attack by up to 5.55% compared with that of the state-of-the-art defense. Moreover, we show that EdgePruner is immune to an adaptive attack.

{{</citation>}}


### (53/149) Task-Agnostic Privacy-Preserving Representation Learning for Federated Learning Against Attribute Inference Attacks (Caridad Arroyo Arevalo et al., 2023)

{{<citation>}}

Caridad Arroyo Arevalo, Sayedeh Leila Noorbakhsh, Yun Dong, Yuan Hong, Binghui Wang. (2023)  
**Task-Agnostic Privacy-Preserving Representation Learning for Federated Learning Against Attribute Inference Attacks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.06989v1)  

---


**ABSTRACT**  
Federated learning (FL) has been widely studied recently due to its property to collaboratively train data from different devices without sharing the raw data. Nevertheless, recent studies show that an adversary can still be possible to infer private information about devices' data, e.g., sensitive attributes such as income, race, and sexual orientation. To mitigate the attribute inference attacks, various existing privacy-preserving FL methods can be adopted/adapted. However, all these existing methods have key limitations: they need to know the FL task in advance, or have intolerable computational overheads or utility losses, or do not have provable privacy guarantees.   We address these issues and design a task-agnostic privacy-preserving presentation learning method for FL ({\bf TAPPFL}) against attribute inference attacks. TAPPFL is formulated via information theory. Specifically, TAPPFL has two mutual information goals, where one goal learns task-agnostic data representations that contain the least information about the private attribute in each device's data, and the other goal ensures the learnt data representations include as much information as possible about the device data to maintain FL utility. We also derive privacy guarantees of TAPPFL against worst-case attribute inference attacks, as well as the inherent tradeoff between utility preservation and privacy protection. Extensive results on multiple datasets and applications validate the effectiveness of TAPPFL to protect data privacy, maintain the FL utility, and be efficient as well. Experimental results also show that TAPPFL outperforms the existing defenses\footnote{Source code and full version: \url{https://github.com/TAPPFL}}.

{{</citation>}}


### (54/149) Blockchain-Based Security Architecture for Unmanned Aerial Vehicles in B5G/6G Services and Beyond: A Comprehensive Approach (Senthil Kumar Jagatheesaperumal et al., 2023)

{{<citation>}}

Senthil Kumar Jagatheesaperumal, Mohamed Rahouti, Kaiqi Xiong, Abdellah Chehri, Nasir Ghani, Jan Bieniek. (2023)  
**Blockchain-Based Security Architecture for Unmanned Aerial Vehicles in B5G/6G Services and Beyond: A Comprehensive Approach**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-RO, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.06928v1)  

---


**ABSTRACT**  
Unmanned Aerial Vehicles (UAVs), previously favored by enthusiasts, have evolved into indispensable tools for effectively managing disasters and responding to emergencies. For example, one of their most critical applications is to provide seamless wireless communication services in remote rural areas. Thus, it is substantial to identify and consider the different security challenges in the research and development associated with advanced UAV-based B5G/6G architectures. Following this requirement, the present study thoroughly examines the security considerations about UAVs in relation to the architectural framework of the 5G/6G system, the technologies that facilitate its operation, and the concerns surrounding privacy. It exhibits security integration at all the protocol stack layers and analyzes the existing mechanisms to secure UAV-based B5G/6G communications and its energy and power optimization factors. Last, this article also summarizes modern technological trends for establishing security and protecting UAV-based systems, along with the open challenges and strategies for future research work.

{{</citation>}}


## cs.AI (21)



### (55/149) Tell, don't show: Declarative facts influence how LLMs generalize (Alexander Meinke et al., 2023)

{{<citation>}}

Alexander Meinke, Owain Evans. (2023)  
**Tell, don't show: Declarative facts influence how LLMs generalize**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07779v1)  

---


**ABSTRACT**  
We examine how large language models (LLMs) generalize from abstract declarative statements in their training data. As an illustration, consider an LLM that is prompted to generate weather reports for London in 2050. One possibility is that the temperatures in the reports match the mean and variance of reports from 2023 (i.e. matching the statistics of pretraining). Another possibility is that the reports predict higher temperatures, by incorporating declarative statements about climate change from scientific papers written in 2023. An example of such a declarative statement is "global temperatures will increase by $1^{\circ} \mathrm{C}$ by 2050".   To test the influence of abstract declarative statements, we construct tasks in which LLMs are finetuned on both declarative and procedural information. We find that declarative statements influence model predictions, even when they conflict with procedural information. In particular, finetuning on a declarative statement $S$ increases the model likelihood for logical consequences of $S$. The effect of declarative statements is consistent across three domains: aligning an AI assistant, predicting weather, and predicting demographic features. Through a series of ablations, we show that the effect of declarative statements cannot be explained by associative learning based on matching keywords. Nevertheless, the effect of declarative statements on model likelihoods is small in absolute terms and increases surprisingly little with model size (i.e. from 330 million to 175 billion parameters). We argue that these results have implications for AI risk (in relation to the "treacherous turn") and for fairness.

{{</citation>}}


### (56/149) Polynomial-based Self-Attention for Table Representation learning (Jayoung Kim et al., 2023)

{{<citation>}}

Jayoung Kim, Yehjin Shin, Noseong Park. (2023)  
**Polynomial-based Self-Attention for Table Representation learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Attention, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.07753v1)  

---


**ABSTRACT**  
Structured data, which constitutes a significant portion of existing data types, has been a long-standing research topic in the field of machine learning. Various representation learning methods for tabular data have been proposed, ranging from encoder-decoder structures to Transformers. Among these, Transformer-based methods have achieved state-of-the-art performance not only in tabular data but also in various other fields, including computer vision and natural language processing. However, recent studies have revealed that self-attention, a key component of Transformers, can lead to an oversmoothing issue. We show that Transformers for tabular data also face this problem, and to address the problem, we propose a novel matrix polynomial-based self-attention layer as a substitute for the original self-attention layer, which enhances model scalability. In our experiments with three representative table learning models equipped with our proposed layer, we illustrate that the layer effectively mitigates the oversmoothing problem and enhances the representation performance of the existing methods, outperforming the state-of-the-art table representation methods.

{{</citation>}}


### (57/149) Saturn Platform: Foundation Model Operations and Generative AI for Financial Services (Antonio J. G. Busson et al., 2023)

{{<citation>}}

Antonio J. G. Busson, Rennan Gaio, Rafael H. Rocha, Francisco Evangelista, Bruno Rizzi, Luan Carvalho, Rafael Miceli, Marcos Rabaioli, David Favaro. (2023)  
**Saturn Platform: Foundation Model Operations and Generative AI for Financial Services**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Financial, Generative AI  
[Paper Link](http://arxiv.org/abs/2312.07721v1)  

---


**ABSTRACT**  
Saturn is an innovative platform that assists Foundation Model (FM) building and its integration with IT operations (Ops). It is custom-made to meet the requirements of data scientists, enabling them to effectively create and implement FMs while enhancing collaboration within their technical domain. By offering a wide range of tools and features, Saturn streamlines and automates different stages of FM development, making it an invaluable asset for data science teams. This white paper introduces prospective applications of generative AI models derived from FMs in the financial sector.

{{</citation>}}


### (58/149) Leveraging Large Language Models to Build and Execute Computational Workflows (Alejandro Duque et al., 2023)

{{<citation>}}

Alejandro Duque, Abdullah Syed, Kastan V. Day, Matthew J. Berry, Daniel S. Katz, Volodymyr V. Kindratenko. (2023)  
**Leveraging Large Language Models to Build and Execute Computational Workflows**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.07711v1)  

---


**ABSTRACT**  
The recent development of large language models (LLMs) with multi-billion parameters, coupled with the creation of user-friendly application programming interfaces (APIs), has paved the way for automatically generating and executing code in response to straightforward human queries. This paper explores how these emerging capabilities can be harnessed to facilitate complex scientific workflows, eliminating the need for traditional coding methods. We present initial findings from our attempt to integrate Phyloflow with OpenAI's function-calling API, and outline a strategy for developing a comprehensive workflow management system based on these concepts.

{{</citation>}}


### (59/149) diff History for Long-Context Language Agents (Ulyana Piterbarg et al., 2023)

{{<citation>}}

Ulyana Piterbarg, Lerrel Pinto, Rob Fergus. (2023)  
**diff History for Long-Context Language Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07540v1)  

---


**ABSTRACT**  
Language Models (LMs) offer an exciting solution for general-purpose embodied control. However, a key technical issue arises when using an LM-based controller: environment observations must be converted to text, which coupled with history, leads to prohibitively large textual prompts. As a result, prior work in LM agents is limited to restricted domains with either small observation size or minimal needs for interaction history. In this paper, we introduce a simple and highly effective solution to these issues. We exploit the fact that consecutive text observations have high similarity and propose to compress them via the Unix diff command. We demonstrate our approach in NetHack, a complex rogue-like video game, that requires long-horizon reasoning for decision-making and is far from solved, particularly for neural agents. Diff history offers an average of 4x increase in the length of the text-based interaction history available to the LM. This observational compression along with the benefits of abstraction yields a 7x improvement in game score on held-out environment instances over state-of-the-art baselines. It also outperforms prior agents that use visual observations by over 40%.

{{</citation>}}


### (60/149) AI capabilities can be significantly improved without expensive retraining (Tom Davidson et al., 2023)

{{<citation>}}

Tom Davidson, Jean-Stanislas Denain, Pablo Villalobos, Guillem Bas. (2023)  
**AI capabilities can be significantly improved without expensive retraining**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07413v1)  

---


**ABSTRACT**  
State-of-the-art AI systems can be significantly improved without expensive retraining via "post-training enhancements"-techniques applied after initial training like fine-tuning the system to use a web browser. We review recent post-training enhancements, categorizing them into five types: tool-use, prompting methods, scaffolding, solution selection, and data generation. Different enhancements improve performance on different tasks, making it hard to compare their significance. So we translate improvements from different enhancements into a common currency, the compute-equivalent gain: how much additional training compute would be needed to improve performance by the same amount as the enhancement. Our non-experimental work shows that post-training enhancements have significant benefits: most surveyed enhancements improve benchmark performance by more than a 5x increase in training compute, some by more than 20x. Post-training enhancements are relatively cheap to develop: fine-tuning costs are typically <1% of the original training cost. Governing the development of capable post-training enhancements may be challenging because frontier models could be enhanced by a wide range of actors.

{{</citation>}}


### (61/149) On Diverse Preferences for Large Language Model Alignment (Dun Zeng et al., 2023)

{{<citation>}}

Dun Zeng, Yong Dai, Pengyu Cheng, Tianhao Hu, Wanshun Chen, Nan Du, Zenglin Xu. (2023)  
**On Diverse Preferences for Large Language Model Alignment**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07401v1)  

---


**ABSTRACT**  
The alignment of large language models (LLMs) with human values is crucial for the development of artificial general intelligence (AGI). One promising approach to achieve this alignment is reinforcement learning from human feedback, which employs a reward model (RM) learned from human preference datasets to guide LLMs in generating text that aligns with human preferences. Through intensive experiments and analysis of reward distribution, this paper finds that preference datasets are diverse from each other, even though they are all proposed to align human preference. Hence, mixing diverse human preference datasets to increase data size for enhancing reward modeling could fail. To address the issue and capture the shared human values from diverse preferences, a new training policy called MORE is introduced, which minimizes preference bias by adaptively adjusting the preference objective across diverse preferences. Experiments with the Pythia-1.4B model and five mixed preference datasets show that MORE achieves superior reward accuracy and lower calibration error, highlighting its ability to leverage diverse human preference data.

{{</citation>}}


### (62/149) LLMEval: A Preliminary Study on How to Evaluate Large Language Models (Yue Zhang et al., 2023)

{{<citation>}}

Yue Zhang, Ming Zhang, Haipeng Yuan, Shichun Liu, Yongyao Shi, Tao Gui, Qi Zhang, Xuanjing Huang. (2023)  
**LLMEval: A Preliminary Study on How to Evaluate Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.07398v1)  

---


**ABSTRACT**  
Recently, the evaluation of Large Language Models has emerged as a popular area of research. The three crucial questions for LLM evaluation are ``what, where, and how to evaluate''. However, the existing research mainly focuses on the first two questions, which are basically what tasks to give the LLM during testing and what kind of knowledge it should deal with. As for the third question, which is about what standards to use, the types of evaluators, how to score, and how to rank, there hasn't been much discussion. In this paper, we analyze evaluation methods by comparing various criteria with both manual and automatic evaluation, utilizing onsite, crowd-sourcing, public annotators and GPT-4, with different scoring methods and ranking systems. We propose a new dataset, LLMEval and conduct evaluations on 20 LLMs. A total of 2,186 individuals participated, leading to the generation of 243,337 manual annotations and 57,511 automatic evaluation results. We perform comparisons and analyses of different settings and conduct 10 conclusions that can provide some insights for evaluating LLM in the future. The dataset and the results are publicly available at https://github.com/llmeval .

{{</citation>}}


### (63/149) A Unified Sampling Framework for Solver Searching of Diffusion Probabilistic Models (Enshu Liu et al., 2023)

{{<citation>}}

Enshu Liu, Xuefei Ning, Huazhong Yang, Yu Wang. (2023)  
**A Unified Sampling Framework for Solver Searching of Diffusion Probabilistic Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.07243v1)  

---


**ABSTRACT**  
Recent years have witnessed the rapid progress and broad application of diffusion probabilistic models (DPMs). Sampling from DPMs can be viewed as solving an ordinary differential equation (ODE). Despite the promising performance, the generation of DPMs usually consumes much time due to the large number of function evaluations (NFE). Though recent works have accelerated the sampling to around 20 steps with high-order solvers, the sample quality with less than 10 NFE can still be improved. In this paper, we propose a unified sampling framework (USF) to study the optional strategies for solver. Under this framework, we further reveal that taking different solving strategies at different timesteps may help further decrease the truncation error, and a carefully designed \emph{solver schedule} has the potential to improve the sample quality by a large margin. Therefore, we propose a new sampling framework based on the exponential integral formulation that allows free choices of solver strategy at each step and design specific decisions for the framework. Moreover, we propose $S^3$, a predictor-based search method that automatically optimizes the solver schedule to get a better time-quality trade-off of sampling. We demonstrate that $S^3$ can find outstanding solver schedules which outperform the state-of-the-art sampling methods on CIFAR-10, CelebA, ImageNet, and LSUN-Bedroom datasets. Specifically, we achieve 2.69 FID with 10 NFE and 6.86 FID with 5 NFE on CIFAR-10 dataset, outperforming the SOTA method significantly. We further apply $S^3$ to Stable-Diffusion model and get an acceleration ratio of 2$\times$, showing the feasibility of sampling in very few steps without retraining the neural network.

{{</citation>}}


### (64/149) Cost Aware Untargeted Poisoning Attack against Graph Neural Networks, (Yuwei Han et al., 2023)

{{<citation>}}

Yuwei Han, Yuni Lai, Yulin Zhu, Kai Zhou. (2023)  
**Cost Aware Untargeted Poisoning Attack against Graph Neural Networks,**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.07158v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have become widely used in the field of graph mining. However, these networks are vulnerable to structural perturbations. While many research efforts have focused on analyzing vulnerability through poisoning attacks, we have identified an inefficiency in current attack losses. These losses steer the attack strategy towards modifying edges targeting misclassified nodes or resilient nodes, resulting in a waste of structural adversarial perturbation. To address this issue, we propose a novel attack loss framework called the Cost Aware Poisoning Attack (CA-attack) to improve the allocation of the attack budget by dynamically considering the classification margins of nodes. Specifically, it prioritizes nodes with smaller positive margins while postponing nodes with negative margins. Our experiments demonstrate that the proposed CA-attack significantly enhances existing attack strategies

{{</citation>}}


### (65/149) Responsibility in Extensive Form Games (Qi Shi, 2023)

{{<citation>}}

Qi Shi. (2023)  
**Responsibility in Extensive Form Games**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07637v1)  

---


**ABSTRACT**  
Two different forms of responsibility, counterfactual and seeing-to-it, have been extensively discussed in the philosophy and AI in the context of a single agent or multiple agents acting simultaneously. Although the generalisation of counterfactual responsibility to a setting where multiple agents act in some order is relatively straightforward, the same cannot be said about seeing-to-it responsibility. Two versions of seeing-to-it modality applicable to such settings have been proposed in the literature. Neither of them perfectly captures the intuition of responsibility. This paper proposes a definition of seeing-to-it responsibility for such settings that amalgamate the two modalities.   This paper shows that the newly proposed notion of responsibility and counterfactual responsibility are not definable through each other and studies the responsibility gap for these two forms of responsibility. It shows that although these two forms of responsibility are not enough to ascribe responsibility in each possible situation, this gap does not exist if higher-order responsibility is taken into account.

{{</citation>}}


### (66/149) Divide-and-Conquer Attack: Harnessing the Power of LLM to Bypass the Censorship of Text-to-Image Generation Model (Yimo Deng et al., 2023)

{{<citation>}}

Yimo Deng, Huangxun Chen. (2023)  
**Divide-and-Conquer Attack: Harnessing the Power of LLM to Bypass the Censorship of Text-to-Image Generation Model**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: ChatGPT, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2312.07130v1)  

---


**ABSTRACT**  
Text-to-image generative models offer many innovative services but also raise ethical concerns due to their potential to generate unethical images. Most publicly available text-to-image models employ safety filters to prevent unintended generation intents. In this work, we introduce the Divide-and-Conquer Attack to circumvent the safety filters of state-of-the-art text-to-image models. Our attack leverages LLMs as agents for text transformation, creating adversarial prompts from sensitive ones. We have developed effective helper prompts that enable LLMs to break down sensitive drawing prompts into multiple harmless descriptions, allowing them to bypass safety filters while still generating sensitive images. This means that the latent harmful meaning only becomes apparent when all individual elements are drawn together. Our evaluation demonstrates that our attack successfully circumvents the closed-box safety filter of SOTA DALLE-3 integrated natively into ChatGPT to generate unethical images. This approach, which essentially uses LLM-generated adversarial prompts against GPT-4-assisted DALLE-3, is akin to using one's own spear to breach their shield. It could have more severe security implications than previous manual crafting or iterative model querying methods, and we hope it stimulates more attention towards similar efforts. Our code and data are available at: https://github.com/researchcode001/Divide-and-Conquer-Attack

{{</citation>}}


### (67/149) Neural Reasoning About Agents' Goals, Preferences, and Actions (Matteo Bortoletto et al., 2023)

{{<citation>}}

Matteo Bortoletto, Lei Shi, Andreas Bulling. (2023)  
**Neural Reasoning About Agents' Goals, Preferences, and Actions**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.07122v1)  

---


**ABSTRACT**  
We propose the Intuitive Reasoning Network (IRENE) - a novel neural model for intuitive psychological reasoning about agents' goals, preferences, and actions that can generalise previous experiences to new situations. IRENE combines a graph neural network for learning agent and world state representations with a transformer to encode the task context. When evaluated on the challenging Baby Intuitions Benchmark, IRENE achieves new state-of-the-art performance on three out of its five tasks - with up to 48.9% improvement. In contrast to existing methods, IRENE is able to bind preferences to specific agents, to better distinguish between rational and irrational agents, and to better understand the role of blocking obstacles. We also investigate, for the first time, the influence of the training tasks on test performance. Our analyses demonstrate the effectiveness of IRENE in combining prior knowledge gained during training for unseen evaluation tasks.

{{</citation>}}


### (68/149) Clash of the Explainers: Argumentation for Context-Appropriate Explanations (Leila Methnani et al., 2023)

{{<citation>}}

Leila Methnani, Virginia Dignum, Andreas Theodorou. (2023)  
**Clash of the Explainers: Argumentation for Context-Appropriate Explanations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07635v1)  

---


**ABSTRACT**  
Understanding when and why to apply any given eXplainable Artificial Intelligence (XAI) technique is not a straightforward task. There is no single approach that is best suited for a given context. This paper aims to address the challenge of selecting the most appropriate explainer given the context in which an explanation is required. For AI explainability to be effective, explanations and how they are presented needs to be oriented towards the stakeholder receiving the explanation. If -- in general -- no single explanation technique surpasses the rest, then reasoning over the available methods is required in order to select one that is context-appropriate. Due to the transparency they afford, we propose employing argumentation techniques to reach an agreement over the most suitable explainers from a given set of possible explainers.   In this paper, we propose a modular reasoning system consisting of a given mental model of the relevant stakeholder, a reasoner component that solves the argumentation problem generated by a multi-explainer component, and an AI model that is to be explained suitably to the stakeholder of interest. By formalising supporting premises -- and inferences -- we can map stakeholder characteristics to those of explanation techniques. This allows us to reason over the techniques and prioritise the best one for the given context, while also offering transparency into the selection decision.

{{</citation>}}


### (69/149) Efficiently Programming Large Language Models using SGLang (Lianmin Zheng et al., 2023)

{{<citation>}}

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng. (2023)  
**Efficiently Programming Large Language Models using SGLang**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-PL, cs.AI  
Keywords: Attention, Language Model  
[Paper Link](http://arxiv.org/abs/2312.07104v1)  

---


**ABSTRACT**  
Large language models (LLMs) are increasingly used for complex tasks requiring multiple chained generation calls, advanced prompting techniques, control flow, and interaction with external environments. However, efficient systems for programming and executing these applications are lacking. To bridge this gap, we introduce SGLang, a Structured Generation Language for LLMs. SGLang is designed for the efficient programming of LLMs and incorporates primitives for common LLM programming patterns. We have implemented SGLang as a domain-specific language embedded in Python, and we developed an interpreter, a compiler, and a high-performance runtime for SGLang. These components work together to enable optimizations such as parallelism, batching, caching, sharing, and other compilation techniques. Additionally, we propose RadixAttention, a novel technique that maintains a Least Recently Used (LRU) cache of the Key-Value (KV) cache for all requests in a radix tree, enabling automatic KV cache reuse across multiple generation calls at runtime. SGLang simplifies the writing of LLM programs and boosts execution efficiency. Our experiments demonstrate that SGLang can speed up common LLM tasks by up to 5x, while reducing code complexity and enhancing control.

{{</citation>}}


### (70/149) Navigating the generative AI era: Introducing the AI assessment scale for ethical GenAI assessment (Mike Perkins et al., 2023)

{{<citation>}}

Mike Perkins, Leon Furze, Jasper Roe, Jason MacVaugh. (2023)  
**Navigating the generative AI era: Introducing the AI assessment scale for ethical GenAI assessment**  

---
Primary Category: cs.AI  
Categories: K-4, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07086v1)  

---


**ABSTRACT**  
Recent developments in Generative Artificial Intelligence (GenAI) have created a paradigm shift in multiple areas of society, and the use of these technologies is likely to become a defining feature of education in coming decades. GenAI offers transformative pedagogical opportunities, while simultaneously posing ethical and academic challenges. Against this backdrop, we outline a practical, simple, and sufficiently comprehensive tool to allow for the integration of GenAI tools into educational assessment: the AI Assessment Scale (AIAS). The AIAS empowers educators to select the appropriate level of GenAI usage in assessments based on the learning outcomes they seek to address. The AIAS offers greater clarity and transparency for students and educators, provides a fair and equitable policy tool for institutions to work with, and offers a nuanced approach which embraces the opportunities of GenAI while recognising that there are instances where such tools may not be pedagogically appropriate or necessary. By adopting a practical, flexible approach that can be implemented quickly, the AIAS can form a much-needed starting point to address the current uncertainty and anxiety regarding GenAI in education. As a secondary objective, we engage with the current literature and advocate for a refocused discourse on GenAI tools in education, one which foregrounds how technologies can help support and enhance teaching and learning, which contrasts with the current focus on GenAI as a facilitator of academic misconduct.

{{</citation>}}


### (71/149) Noise Distribution Decomposition based Multi-Agent Distributional Reinforcement Learning (Wei Geng et al., 2023)

{{<citation>}}

Wei Geng, Baidi Xiao, Rongpeng Li, Ning Wei, Dong Wang, Zhifeng Zhao. (2023)  
**Noise Distribution Decomposition based Multi-Agent Distributional Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.07025v1)  

---


**ABSTRACT**  
Generally, Reinforcement Learning (RL) agent updates its policy by repetitively interacting with the environment, contingent on the received rewards to observed states and undertaken actions. However, the environmental disturbance, commonly leading to noisy observations (e.g., rewards and states), could significantly shape the performance of agent. Furthermore, the learning performance of Multi-Agent Reinforcement Learning (MARL) is more susceptible to noise due to the interference among intelligent agents. Therefore, it becomes imperative to revolutionize the design of MARL, so as to capably ameliorate the annoying impact of noisy rewards. In this paper, we propose a novel decomposition-based multi-agent distributional RL method by approximating the globally shared noisy reward by a Gaussian mixture model (GMM) and decomposing it into the combination of individual distributional local rewards, with which each agent can be updated locally through distributional RL. Moreover, a diffusion model (DM) is leveraged for reward generation in order to mitigate the issue of costly interaction expenditure for learning distributions. Furthermore, the optimality of the distribution decomposition is theoretically validated, while the design of loss function is carefully calibrated to avoid the decomposition ambiguity. We also verify the effectiveness of the proposed method through extensive simulation experiments with noisy rewards. Besides, different risk-sensitive policies are evaluated in order to demonstrate the superiority of distributional RL in different MARL tasks.

{{</citation>}}


### (72/149) RACER: Rational Artificial Intelligence Car-following-model Enhanced by Reality (Tianyi Li et al., 2023)

{{<citation>}}

Tianyi Li, Alexander Halatsis, Raphael Stern. (2023)  
**RACER: Rational Artificial Intelligence Car-following-model Enhanced by Reality**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-RO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07003v1)  

---


**ABSTRACT**  
This paper introduces RACER, the Rational Artificial Intelligence Car-following model Enhanced by Reality, a cutting-edge deep learning car-following model, that satisfies partial derivative constraints, designed to predict Adaptive Cruise Control (ACC) driving behavior while staying theoretically feasible. Unlike conventional models, RACER effectively integrates Rational Driving Constraints (RDCs), crucial tenets of actual driving, resulting in strikingly accurate and realistic predictions. Against established models like the Optimal Velocity Relative Velocity (OVRV), a car-following Neural Network (NN), and a car-following Physics-Informed Neural Network (PINN), RACER excels across key metrics, such as acceleration, velocity, and spacing. Notably, it displays a perfect adherence to the RDCs, registering zero violations, in stark contrast to other models. This study highlights the immense value of incorporating physical constraints within AI models, especially for augmenting safety measures in transportation. It also paves the way for future research to test these models against human driving data, with the potential to guide safer and more rational driving behavior. The versatility of the proposed model, including its potential to incorporate additional derivative constraints and broader architectural applications, enhances its appeal and broadens its impact within the scientific community.

{{</citation>}}


### (73/149) AI-based Wildfire Prevention, Detection and Suppression System (Prisha Shroff, 2023)

{{<citation>}}

Prisha Shroff. (2023)  
**AI-based Wildfire Prevention, Detection and Suppression System**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Drone  
[Paper Link](http://arxiv.org/abs/2312.06990v1)  

---


**ABSTRACT**  
Wildfires pose a serious threat to the environment of the world. The global wildfire season length has increased by 19% and severe wildfires have besieged nations around the world. Every year, forests are burned by wildfires, causing vast amounts of carbon dioxide to be released into the atmosphere, contributing to climate change. There is a need for a system which prevents, detects, and suppresses wildfires. The AI based Wildfire Prevention, Detection and Suppression System (WPDSS) is a novel, fully automated, end to end, AI based solution to effectively predict hotspots and detect wildfires, deploy drones to spray fire retardant, preventing and suppressing wildfires. WPDSS consists of four steps. 1. Preprocessing: WPDSS loads real time satellite data from NASA and meteorological data from NOAA of vegetation, temperature, precipitation, wind, soil moisture, and land cover for prevention. For detection, it loads the real time data of Land Cover, Humidity, Temperature, Vegetation, Burned Area Index, Ozone, and CO2. It uses the process of masking to eliminate not hotspots and not wildfires such as water bodies, and rainfall. 2. Learning: The AI model consists of a random forest classifier, which is trained using a labeled dataset of hotspots and wildfires and not hotspots and not wildfires. 3. Identification of hotspots and wildfires: WPDSS runs the real time data through the model to automatically identify hotspots and wildfires. 4. Drone deployment: The drone flies to the identified hotspot or wildfire location. WPDSS attained a 98.6% accuracy in identifying hotspots and a 98.7% accuracy in detecting wildfires. WPDSS will reduce the impacts of climate change, protect ecosystems and biodiversity, avert huge economic losses, and save human lives. The power of WPDSS developed can be applied to any location globally to prevent and suppress wildfires, reducing climate change.

{{</citation>}}


### (74/149) Anytime Approximate Formal Feature Attribution (Jinqiang Yu et al., 2023)

{{<citation>}}

Jinqiang Yu, Graham Farr, Alexey Ignatiev, Peter J. Stuckey. (2023)  
**Anytime Approximate Formal Feature Attribution**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-LO, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.06973v1)  

---


**ABSTRACT**  
Widespread use of artificial intelligence (AI) algorithms and machine learning (ML) models on the one hand and a number of crucial issues pertaining to them warrant the need for explainable artificial intelligence (XAI). A key explainability question is: given this decision was made, what are the input features which contributed to the decision? Although a range of XAI approaches exist to tackle this problem, most of them have significant limitations. Heuristic XAI approaches suffer from the lack of quality guarantees, and often try to approximate Shapley values, which is not the same as explaining which features contribute to a decision. A recent alternative is so-called formal feature attribution (FFA), which defines feature importance as the fraction of formal abductive explanations (AXp's) containing the given feature. This measures feature importance from the view of formally reasoning about the model's behavior. It is challenging to compute FFA using its definition because that involves counting AXp's, although one can approximate it. Based on these results, this paper makes several contributions. First, it gives compelling evidence that computing FFA is intractable, even if the set of contrastive formal explanations (CXp's) is provided, by proving that the problem is #P-hard. Second, by using the duality between AXp's and CXp's, it proposes an efficient heuristic to switch from CXp enumeration to AXp enumeration on-the-fly resulting in an adaptive explanation enumeration algorithm effectively approximating FFA in an anytime fashion. Finally, experimental results obtained on a range of widely used datasets demonstrate the effectiveness of the proposed FFA approximation approach in terms of the error of FFA approximation as well as the number of explanations computed and their diversity given a fixed time limit.

{{</citation>}}


### (75/149) Unsupervised Extractive Summarization with Learnable Length Control Strategies (Renlong Jie et al., 2023)

{{<citation>}}

Renlong Jie, Xiaojun Meng, Xin Jiang, Qun Liu. (2023)  
**Unsupervised Extractive Summarization with Learnable Length Control Strategies**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2312.06901v1)  

---


**ABSTRACT**  
Unsupervised extractive summarization is an important technique in information extraction and retrieval. Compared with supervised method, it does not require high-quality human-labelled summaries for training and thus can be easily applied for documents with different types, domains or languages. Most of existing unsupervised methods including TextRank and PACSUM rely on graph-based ranking on sentence centrality. However, this scorer can not be directly applied in end-to-end training, and the positional-related prior assumption is often needed for achieving good summaries. In addition, less attention is paid to length-controllable extractor, where users can decide to summarize texts under particular length constraint. This paper introduces an unsupervised extractive summarization model based on a siamese network, for which we develop a trainable bidirectional prediction objective between the selected summary and the original document. Different from the centrality-based ranking methods, our extractive scorer can be trained in an end-to-end manner, with no other requirement of positional assumption. In addition, we introduce a differentiable length control module by approximating 0-1 knapsack solver for end-to-end length-controllable extracting. Experiments show that our unsupervised method largely outperforms the centrality-based baseline using a same sentence encoder. In terms of length control ability, via our trainable knapsack module, the performance consistently outperforms the strong baseline without utilizing end-to-end training. Human evaluation further evidences that our method performs the best among baselines in terms of relevance and consistency.

{{</citation>}}


## cs.HC (7)



### (76/149) Designing with Language: Wireframing UI Design Intent with Generative Large Language Models (Sidong Feng et al., 2023)

{{<citation>}}

Sidong Feng, Mingyue Yuan, Jieshan Chen, Zhenchang Xing, Chunyang Chen. (2023)  
**Designing with Language: Wireframing UI Design Intent with Generative Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07755v1)  

---


**ABSTRACT**  
Wireframing is a critical step in the UI design process. Mid-fidelity wireframes offer more impactful and engaging visuals compared to low-fidelity versions. However, their creation can be time-consuming and labor-intensive, requiring the addition of actual content and semantic icons. In this paper, we introduce a novel solution WireGen, to automatically generate mid-fidelity wireframes with just a brief design intent description using the generative Large Language Models (LLMs). Our experiments demonstrate the effectiveness of WireGen in producing 77.5% significantly better wireframes, outperforming two widely-used in-context learning baselines. A user study with 5 designers further validates its real-world usefulness, highlighting its potential value to enhance UI design process.

{{</citation>}}


### (77/149) Scaling Culture in Blockchain Gaming: Generative AI and Pseudonymous Engagement (Henrik Axelsen et al., 2023)

{{<citation>}}

Henrik Axelsen, Sebastian Axelsen, Valdemar Licht, Jason Potts. (2023)  
**Scaling Culture in Blockchain Gaming: Generative AI and Pseudonymous Engagement**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, GPT, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.07693v1)  

---


**ABSTRACT**  
Managing rapidly growing decentralized gaming communities brings unique challenges at the nexus of cultural economics and technology. This paper introduces a streamlined analytical framework that utilizes Large Language Models (LLMs), in this instance open-access generative pre-trained transformer (GPT) models, offering an efficient solution with deeper insights into community dynamics. The framework aids moderators in identifying pseudonymous actor intent, moderating toxic behavior, rewarding desired actions to avoid unintended consequences of blockchain-based gaming, and gauging community sentiment as communities venture into metaverse platforms and plan for hypergrowth. This framework strengthens community controls, eases onboarding, and promotes a common moral mission across communities while reducing agency costs by 95 pct. Highlighting the transformative role of generative AI, the paper emphasizes its potential to redefine the cost of cultural production. It showcases the utility of GPTs in digital community management, expanding their implications in cultural economics and transmedia storytelling.

{{</citation>}}


### (78/149) Can ChatGPT Play the Role of a Teaching Assistant in an Introductory Programming Course? (Anishka et al., 2023)

{{<citation>}}

Anishka, Atharva Mehta, Nipun Gupta, Dhruv Kumar, Pankaj Jalote. (2023)  
**Can ChatGPT Play the Role of a Teaching Assistant in an Introductory Programming Course?**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.07343v1)  

---


**ABSTRACT**  
The emergence of Large language models (LLMs) is expected to have a major impact on education. This paper explores the potential of using ChatGPT, an LLM, as a virtual Teaching Assistant (TA) in an Introductory Programming Course. We evaluate ChatGPT's capabilities by comparing its performance with that of human TAs in some TA functions. The TA functions which we focus on include (1) solving programming assignments, (2) grading student code submissions, and (3) providing feedback to undergraduate students in an introductory programming course. Firstly, we investigate how closely ChatGPT's solutions align with those submitted by students. This analysis goes beyond code correctness and also considers code quality. Secondly, we assess ChatGPT's proficiency in grading student code submissions using a given grading rubric and compare its performance with the grades assigned by human TAs. Thirdly, we analyze the quality and relevance of the feedback provided by ChatGPT. This evaluation considers how well ChatGPT addresses mistakes and offers suggestions for improvement in student solutions from both code correctness and code quality perspectives. We conclude with a discussion on the implications of integrating ChatGPT into computing education for automated grading, personalized learning experiences, and instructional support.

{{</citation>}}


### (79/149) Exploring Large Language Models to Facilitate Variable Autonomy for Human-Robot Teaming (Younes Lakhnati et al., 2023)

{{<citation>}}

Younes Lakhnati, Max Pascher, Jens Gerken. (2023)  
**Exploring Large Language Models to Facilitate Variable Autonomy for Human-Robot Teaming**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-RO, cs.HC  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.07214v1)  

---


**ABSTRACT**  
In a rapidly evolving digital landscape autonomous tools and robots are becoming commonplace. Recognizing the significance of this development, this paper explores the integration of Large Language Models (LLMs) like Generative pre-trained transformer (GPT) into human-robot teaming environments to facilitate variable autonomy through the means of verbal human-robot communication. In this paper, we introduce a novel framework for such a GPT-powered multi-robot testbed environment, based on a Unity Virtual Reality (VR) setting. This system allows users to interact with robot agents through natural language, each powered by individual GPT cores. By means of OpenAI's function calling, we bridge the gap between unstructured natural language input and structure robot actions. A user study with 12 participants explores the effectiveness of GPT-4 and, more importantly, user strategies when being given the opportunity to converse in natural language within a multi-robot environment. Our findings suggest that users may have preconceived expectations on how to converse with robots and seldom try to explore the actual language and cognitive capabilities of their robot collaborators. Still, those users who did explore where able to benefit from a much more natural flow of communication and human-like back-and-forth. We provide a set of lessons learned for future research and technical implementations of similar systems.

{{</citation>}}


### (80/149) Towards Enhanced Human Activity Recognition through Natural Language Generation and Pose Estimation (Nikhil Kashyap et al., 2023)

{{<citation>}}

Nikhil Kashyap, Manas Satish Bedmutha, Prerit Chaudhary, Brian Wood, Wanda Pratt, Janice Sabin, Andrea Hartzler, Nadir Weibel. (2023)  
**Towards Enhanced Human Activity Recognition through Natural Language Generation and Pose Estimation**  

---
Primary Category: cs.HC  
Categories: I-2-7, cs-HC, cs.HC  
Keywords: Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2312.06965v1)  

---


**ABSTRACT**  
Vision-based human activity recognition (HAR) has made substantial progress in recognizing predefined gestures but lacks adaptability for emerging activities. This paper introduces a paradigm shift by harnessing generative modeling and large language models (LLMs) to enhance vision-based HAR. We propose utilizing LLMs to generate descriptive textual representations of activities using pose keypoints as an intermediate representation. Incorporating pose keypoints adds contextual depth to the recognition process, allowing for sequences of vectors resembling text chunks, compatible with LLMs. This innovative fusion of computer vision and natural language processing holds significant potential for revolutionizing activity recognition. A proof of concept study on a Kinetics700 dataset subset validates the approach's efficacy, highlighting improved accuracy and interpretability. Future implications encompass enhanced accuracy, novel research avenues, model generalization, and ethical considerations for transparency. This framework has real-world applications, including personalized gym workout feedback and nuanced sports training insights. By connecting visual cues to interpretable textual descriptions, the proposed framework advances HAR accuracy and applicability, shaping the landscape of pervasive computing and activity recognition research. As this approach evolves, it promises a more insightful understanding of human activities across diverse contexts, marking a significant step towards a better world.

{{</citation>}}


### (81/149) Facial Emotion Recognition in VR Games (Fatemeh Dehghani et al., 2023)

{{<citation>}}

Fatemeh Dehghani, Loutfouz Zaman. (2023)  
**Facial Emotion Recognition in VR Games**  

---
Primary Category: cs.HC  
Categories: cs-CV, cs-HC, cs-LG, cs.HC  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2312.06925v1)  

---


**ABSTRACT**  
Emotion detection is a crucial component of Games User Research (GUR), as it allows game developers to gain insights into players' emotional experiences and tailor their games accordingly. However, detecting emotions in Virtual Reality (VR) games is challenging due to the Head-Mounted Display (HMD) that covers the top part of the player's face, namely, their eyes and eyebrows, which provide crucial information for recognizing the impression. To tackle this we used a Convolutional Neural Network (CNN) to train a model to predict emotions in full-face images where the eyes and eyebrows are covered. We used the FER2013 dataset, which we modified to cover eyes and eyebrows in images. The model in these images can accurately recognize seven different emotions which are anger, happiness, disgust, fear, impartiality, sadness and surprise.   We assessed the model's performance by testing it on two VR games and using it to detect players' emotions. We collected self-reported emotion data from the players after the gameplay sessions. We analyzed the data collected from our experiment to understand which emotions players experience during the gameplay. We found that our approach has the potential to enhance gameplay analysis by enabling the detection of players' emotions in VR games, which can help game developers create more engaging and immersive game experiences.

{{</citation>}}


### (82/149) 'I Want It That Way': Enabling Interactive Decision Support Using Large Language Models and Constraint Programming (Connor Lawless et al., 2023)

{{<citation>}}

Connor Lawless, Jakob Schoeffer, Lindy Le, Kael Rowan, Shilad Sen, Cristina St. Hill, Jina Suh, Bahar Sarrafzadeh. (2023)  
**'I Want It That Way': Enabling Interactive Decision Support Using Large Language Models and Constraint Programming**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.06908v1)  

---


**ABSTRACT**  
A critical factor in the success of decision support systems is the accurate modeling of user preferences. Psychology research has demonstrated that users often develop their preferences during the elicitation process, highlighting the pivotal role of system-user interaction in developing personalized systems. This paper introduces a novel approach, combining Large Language Models (LLMs) with Constraint Programming to facilitate interactive decision support. We study this hybrid framework through the lens of meeting scheduling, a time-consuming daily activity faced by a multitude of information workers. We conduct three studies to evaluate the novel framework, including a diary study (n=64) to characterize contextual scheduling preferences, a quantitative evaluation of the system's performance, and a user study (n=10) with a prototype system. Our work highlights the potential for a hybrid LLM and optimization approach for iterative preference elicitation and design considerations for building systems that support human-system collaborative decision-making processes.

{{</citation>}}


## eess.IV (4)



### (83/149) MedYOLO: A Medical Image Object Detection Framework (Joseph Sobek et al., 2023)

{{<citation>}}

Joseph Sobek, Jose R. Medina Inojosa, Betsy J. Medina Inojosa, S. M. Rassoulinejad-Mousavi, Gian Marco Conte, Francisco Lopez-Jimenez, Bradley J. Erickson. (2023)  
**MedYOLO: A Medical Image Object Detection Framework**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.07729v1)  

---


**ABSTRACT**  
Artificial intelligence-enhanced identification of organs, lesions, and other structures in medical imaging is typically done using convolutional neural networks (CNNs) designed to make voxel-accurate segmentations of the region of interest. However, the labels required to train these CNNs are time-consuming to generate and require attention from subject matter experts to ensure quality. For tasks where voxel-level precision is not required, object detection models offer a viable alternative that can reduce annotation effort. Despite this potential application, there are few options for general purpose object detection frameworks available for 3-D medical imaging. We report on MedYOLO, a 3-D object detection framework using the one-shot detection method of the YOLO family of models and designed for use with medical imaging. We tested this model on four different datasets: BRaTS, LIDC, an abdominal organ Computed Tomography (CT) dataset, and an ECG-gated heart CT dataset. We found our models achieve high performance on commonly present medium and large-sized structures such as the heart, liver, and pancreas even without hyperparameter tuning. However, the models struggle with very small or rarely present structures.

{{</citation>}}


### (84/149) Super-Resolution on Rotationally Scanned Photoacoustic Microscopy Images Incorporating Scanning Prior (Kai Pan et al., 2023)

{{<citation>}}

Kai Pan, Linyang Li, Li Lin, Pujin Cheng, Junyan Lyu, Lei Xi, Xiaoyin Tang. (2023)  
**Super-Resolution on Rotationally Scanned Photoacoustic Microscopy Images Incorporating Scanning Prior**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.07226v1)  

---


**ABSTRACT**  
Photoacoustic Microscopy (PAM) images integrating the advantages of optical contrast and acoustic resolution have been widely used in brain studies. However, there exists a trade-off between scanning speed and image resolution. Compared with traditional raster scanning, rotational scanning provides good opportunities for fast PAM imaging by optimizing the scanning mechanism. Recently, there is a trend to incorporate deep learning into the scanning process to further increase the scanning speed.Yet, most such attempts are performed for raster scanning while those for rotational scanning are relatively rare. In this study, we propose a novel and well-performing super-resolution framework for rotational scanning-based PAM imaging. To eliminate adjacent rows' displacements due to subject motion or high-frequency scanning distortion,we introduce a registration module across odd and even rows in the preprocessing and incorporate displacement degradation in the training. Besides, gradient-based patch selection is proposed to increase the probability of blood vessel patches being selected for training. A Transformer-based network with a global receptive field is applied for better performance. Experimental results on both synthetic and real datasets demonstrate the effectiveness and generalizability of our proposed framework for rotationally scanned PAM images'super-resolution, both quantitatively and qualitatively. Code is available at https://github.com/11710615/PAMSR.git.

{{</citation>}}


### (85/149) MS-Twins: Multi-Scale Deep Self-Attention Networks for Medical Image Segmentation (Jing Xu, 2023)

{{<citation>}}

Jing Xu. (2023)  
**MS-Twins: Multi-Scale Deep Self-Attention Networks for Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2312.07128v1)  

---


**ABSTRACT**  
Although transformer is preferred in natural language processing, few studies have applied it in the field of medical imaging. For its long-term dependency, the transformer is expected to contribute to unconventional convolution neural net conquer their inherent spatial induction bias. The lately suggested transformer-based partition method only uses the transformer as an auxiliary module to help encode the global context into a convolutional representation. There is hardly any study about how to optimum bond self-attention (the kernel of transformers) with convolution. To solve the problem, the article proposes MS-Twins (Multi-Scale Twins), which is a powerful segmentation model on account of the bond of self-attention and convolution. MS-Twins can better capture semantic and fine-grained information by combining different scales and cascading features. Compared with the existing network structure, MS-Twins has made significant progress on the previous method based on the transformer of two in common use data sets, Synapse and ACDC. In particular, the performance of MS-Twins on Synapse is 8% higher than SwinUNet. Even compared with nnUNet, the best entirely convoluted medical image segmentation network, the performance of MS-Twins on Synapse and ACDC still has a bit advantage.

{{</citation>}}


### (86/149) On the notion of Hallucinations from the lens of Bias and Validity in Synthetic CXR Images (Gauri Bhardwaj et al., 2023)

{{<citation>}}

Gauri Bhardwaj, Yuvaraj Govindarajulu, Sundaraparipurnan Narayanan, Pavan Kulkarni, Manojkumar Parmar. (2023)  
**On the notion of Hallucinations from the lens of Bias and Validity in Synthetic CXR Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.06979v1)  

---


**ABSTRACT**  
Medical imaging has revolutionized disease diagnosis, yet the potential is hampered by limited access to diverse and privacy-conscious datasets. Open-source medical datasets, while valuable, suffer from data quality and clinical information disparities. Generative models, such as diffusion models, aim to mitigate these challenges. At Stanford, researchers explored the utility of a fine-tuned Stable Diffusion model (RoentGen) for medical imaging data augmentation. Our work examines specific considerations to expand the Stanford research question, Could Stable Diffusion Solve a Gap in Medical Imaging Data? from the lens of bias and validity of the generated outcomes. We leveraged RoentGen to produce synthetic Chest-XRay (CXR) images and conducted assessments on bias, validity, and hallucinations. Diagnostic accuracy was evaluated by a disease classifier, while a COVID classifier uncovered latent hallucinations. The bias analysis unveiled disparities in classification performance among various subgroups, with a pronounced impact on the Female Hispanic subgroup. Furthermore, incorporating race and gender into input prompts exacerbated fairness issues in the generated images. The quality of synthetic images exhibited variability, particularly in certain disease classes, where there was more significant uncertainty compared to the original images. Additionally, we observed latent hallucinations, with approximately 42% of the images incorrectly indicating COVID, hinting at the presence of hallucinatory elements. These identifications provide new research directions towards interpretability of synthetic CXR images, for further understanding of associated risks and patient safety in medical applications.

{{</citation>}}


## q-bio.NC (1)



### (87/149) Brain-optimized inference improves reconstructions of fMRI brain activity (Reese Kneeland et al., 2023)

{{<citation>}}

Reese Kneeland, Jordyn Ojeda, Ghislain St-Yves, Thomas Naselaris. (2023)  
**Brain-optimized inference improves reconstructions of fMRI brain activity**  

---
Primary Category: q-bio.NC  
Categories: cs-AI, cs-CV, cs-LG, q-bio-NC, q-bio.NC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07705v1)  

---


**ABSTRACT**  
The release of large datasets and developments in AI have led to dramatic improvements in decoding methods that reconstruct seen images from human brain activity. We evaluate the prospect of further improving recent decoding methods by optimizing for consistency between reconstructions and brain activity during inference. We sample seed reconstructions from a base decoding method, then iteratively refine these reconstructions using a brain-optimized encoding model that maps images to brain activity. At each iteration, we sample a small library of images from an image distribution (a diffusion model) conditioned on a seed reconstruction from the previous iteration. We select those that best approximate the measured brain activity when passed through our encoding model, and use these images for structural guidance during the generation of the small library in the next iteration. We reduce the stochasticity of the image distribution at each iteration, and stop when a criterion on the "width" of the image distribution is met. We show that when this process is applied to recent decoding methods, it outperforms the base decoding method as measured by human raters, a variety of image feature metrics, and alignment to brain activity. These results demonstrate that reconstruction quality can be significantly improved by explicitly aligning decoding distributions to brain activity distributions, even when the seed reconstruction is output from a state-of-the-art decoding algorithm. Interestingly, the rate of refinement varies systematically across visual cortex, with earlier visual areas generally converging more slowly and preferring narrower image distributions, relative to higher-level brain areas. Brain-optimized inference thus offers a succinct and novel method for improving reconstructions and exploring the diversity of representations across visual brain areas.

{{</citation>}}


## cs.CV (48)



### (88/149) FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition (Sicheng Mo et al., 2023)

{{<citation>}}

Sicheng Mo, Fangzhou Mu, Kuan Heng Lin, Yanli Liu, Bochen Guan, Yin Li, Bolei Zhou. (2023)  
**FreeControl: Training-Free Spatial Control of Any Text-to-Image Diffusion Model with Any Condition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07536v1)  

---


**ABSTRACT**  
Recent approaches such as ControlNet offer users fine-grained spatial control over text-to-image (T2I) diffusion models. However, auxiliary modules have to be trained for each type of spatial condition, model architecture, and checkpoint, putting them at odds with the diverse intents and preferences a human designer would like to convey to the AI models during the content creation process. In this work, we present FreeControl, a training-free approach for controllable T2I generation that supports multiple conditions, architectures, and checkpoints simultaneously. FreeControl designs structure guidance to facilitate the structure alignment with a guidance image, and appearance guidance to enable the appearance sharing between images generated using the same seed. Extensive qualitative and quantitative experiments demonstrate the superior performance of FreeControl across a variety of pre-trained T2I models. In particular, FreeControl facilitates convenient training-free control over many different architectures and checkpoints, allows the challenging input conditions on which most of the existing training-free methods fail, and achieves competitive synthesis quality with training-based approaches.

{{</citation>}}


### (89/149) VILA: On Pre-training for Visual Language Models (Ji Lin et al., 2023)

{{<citation>}}

Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz, Mohammad Shoeybi, Song Han. (2023)  
**VILA: On Pre-training for Visual Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07533v2)  

---


**ABSTRACT**  
Visual language models (VLMs) rapidly progressed with the recent success of large language models. There have been growing efforts on visual instruction tuning to extend the LLM with visual inputs, but lacks an in-depth study of the visual language pre-training process, where the model learns to perform joint modeling on both modalities. In this work, we examine the design options for VLM pre-training by augmenting LLM towards VLM through step-by-step controllable comparisons. We introduce three main findings: (1) freezing LLMs during pre-training can achieve decent zero-shot performance, but lack in-context learning capability, which requires unfreezing the LLM; (2) interleaved pre-training data is beneficial whereas image-text pairs alone are not optimal; (3) re-blending text-only instruction data to image-text data during instruction fine-tuning not only remedies the degradation of text-only tasks, but also boosts VLM task accuracy. With an enhanced pre-training recipe we build VILA, a Visual Language model family that consistently outperforms the state-of-the-art models, e.g., LLaVA-1.5, across main benchmarks without bells and whistles. Multi-modal pre-training also helps unveil appealing properties of VILA, including multi-image reasoning, enhanced in-context learning, and better world knowledge.

{{</citation>}}


### (90/149) Interfacing Foundation Models' Embeddings (Xueyan Zou et al., 2023)

{{<citation>}}

Xueyan Zou, Linjie Li, Jianfeng Wang, Jianwei Yang, Mingyu Ding, Zhengyuan Yang, Feng Li, Hao Zhang, Shilong Liu, Arul Aravinthan, Yong Jae Lee, Lijuan Wang. (2023)  
**Interfacing Foundation Models' Embeddings**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.07532v1)  

---


**ABSTRACT**  
We present FIND, a generalized interface for aligning foundation models' embeddings. As shown in teaser figure, a lightweight transformer interface without tuning any foundation model weights is enough for a unified image (segmentation) and dataset-level (retrieval) understanding. The proposed interface has the following favorable attributes: (1) Generalizable. It applies to various tasks spanning retrieval, segmentation, \textit{etc.}, under the same architecture and weights. (2) Prototypable. Different tasks are able to be implemented through prototyping attention masks and embedding types. (3) Extendable. The proposed interface is adaptive to new tasks, and new models. (4) Interleavable. With the benefit of multi-task multi-modal training, the proposed interface creates an interleaved shared embedding space. In light of the interleaved embedding space, we introduce the FIND-Bench, which introduces new training and evaluation annotations to the COCO dataset for interleave segmentation and retrieval. Our approach achieves state-of-the-art performance on FIND-Bench and competitive performance on standard retrieval and segmentation settings. The training, evaluation, and demo code as well as the dataset have been released at https://github.com/UX-Decoder/FIND.

{{</citation>}}


### (91/149) Weakly Supervised 3D Object Detection via Multi-Level Visual Guidance (Kuan-Chih Huang et al., 2023)

{{<citation>}}

Kuan-Chih Huang, Yi-Hsuan Tsai, Ming-Hsuan Yang. (2023)  
**Weakly Supervised 3D Object Detection via Multi-Level Visual Guidance**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.07530v1)  

---


**ABSTRACT**  
Weakly supervised 3D object detection aims to learn a 3D detector with lower annotation cost, e.g., 2D labels. Unlike prior work which still relies on few accurate 3D annotations, we propose a framework to study how to leverage constraints between 2D and 3D domains without requiring any 3D labels. Specifically, we employ visual data from three perspectives to establish connections between 2D and 3D domains. First, we design a feature-level constraint to align LiDAR and image features based on object-aware regions. Second, the output-level constraint is developed to enforce the overlap between 2D and projected 3D box estimations. Finally, the training-level constraint is utilized by producing accurate and consistent 3D pseudo-labels that align with the visual data. We conduct extensive experiments on the KITTI dataset to validate the effectiveness of the proposed three constraints. Without using any 3D labels, our method achieves favorable performance against state-of-the-art approaches and is competitive with the method that uses 500-frame 3D annotations. Code and models will be made publicly available at https://github.com/kuanchihhuang/VG-W3D.

{{</citation>}}


### (92/149) NAC-TCN: Temporal Convolutional Networks with Causal Dilated Neighborhood Attention for Emotion Understanding (Alexander Mehta et al., 2023)

{{<citation>}}

Alexander Mehta, William Yang. (2023)  
**NAC-TCN: Temporal Convolutional Networks with Causal Dilated Neighborhood Attention for Emotion Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, LSTM, Self-Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.07507v1)  

---


**ABSTRACT**  
In the task of emotion recognition from videos, a key improvement has been to focus on emotions over time rather than a single frame. There are many architectures to address this task such as GRUs, LSTMs, Self-Attention, Transformers, and Temporal Convolutional Networks (TCNs). However, these methods suffer from high memory usage, large amounts of operations, or poor gradients. We propose a method known as Neighborhood Attention with Convolutions TCN (NAC-TCN) which incorporates the benefits of attention and Temporal Convolutional Networks while ensuring that causal relationships are understood which results in a reduction in computation and memory cost. We accomplish this by introducing a causal version of Dilated Neighborhood Attention while incorporating it with convolutions. Our model achieves comparable, better, or state-of-the-art performance over TCNs, TCAN, LSTMs, and GRUs while requiring fewer parameters on standard emotion recognition datasets. We publish our code online for easy reproducibility and use in other projects.

{{</citation>}}


### (93/149) Exploring Plain ViT Reconstruction for Multi-class Unsupervised Anomaly Detection (Jiangning Zhang et al., 2023)

{{<citation>}}

Jiangning Zhang, Xuhai Chen, Yabiao Wang, Chengjie Wang, Yong Liu, Xiangtai Li, Ming-Hsuan Yang, Dacheng Tao. (2023)  
**Exploring Plain ViT Reconstruction for Multi-class Unsupervised Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2312.07495v1)  

---


**ABSTRACT**  
This work studies the recently proposed challenging and practical Multi-class Unsupervised Anomaly Detection (MUAD) task, which only requires normal images for training while simultaneously testing both normal/anomaly images for multiple classes. Existing reconstruction-based methods typically adopt pyramid networks as encoders/decoders to obtain multi-resolution features, accompanied by elaborate sub-modules with heavier handcraft engineering designs for more precise localization. In contrast, a plain Vision Transformer (ViT) with simple architecture has been shown effective in multiple domains, which is simpler, more effective, and elegant. Following this spirit, this paper explores plain ViT architecture for MUAD. Specifically, we abstract a Meta-AD concept by inducing current reconstruction-based methods. Then, we instantiate a novel and elegant plain ViT-based symmetric ViTAD structure, effectively designed step by step from three macro and four micro perspectives. In addition, this paper reveals several interesting findings for further exploration. Finally, we propose a comprehensive and fair evaluation benchmark on eight metrics for the MUAD task. Based on a naive training recipe, ViTAD achieves state-of-the-art (SoTA) results and efficiency on the MVTec AD and VisA datasets without bells and whistles, obtaining 85.4 mAD that surpasses SoTA UniAD by +3.0, and only requiring 1.1 hours and 2.3G GPU memory to complete model training by a single V100 GPU. Source code, models, and more results are available at https://zhangzjn.github.io/projects/ViTAD.

{{</citation>}}


### (94/149) NearbyPatchCL: Leveraging Nearby Patches for Self-Supervised Patch-Level Multi-Class Classification in Whole-Slide Images (Gia-Bao Le et al., 2023)

{{<citation>}}

Gia-Bao Le, Van-Tien Nguyen, Trung-Nghia Le, Minh-Triet Tran. (2023)  
**NearbyPatchCL: Leveraging Nearby Patches for Self-Supervised Patch-Level Multi-Class Classification in Whole-Slide Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.07489v1)  

---


**ABSTRACT**  
Whole-slide image (WSI) analysis plays a crucial role in cancer diagnosis and treatment. In addressing the demands of this critical task, self-supervised learning (SSL) methods have emerged as a valuable resource, leveraging their efficiency in circumventing the need for a large number of annotations, which can be both costly and time-consuming to deploy supervised methods. Nevertheless, patch-wise representation may exhibit instability in performance, primarily due to class imbalances stemming from patch selection within WSIs. In this paper, we introduce Nearby Patch Contrastive Learning (NearbyPatchCL), a novel self-supervised learning method that leverages nearby patches as positive samples and a decoupled contrastive loss for robust representation learning. Our method demonstrates a tangible enhancement in performance for downstream tasks involving patch-level multi-class classification. Additionally, we curate a new dataset derived from WSIs sourced from the Canine Cutaneous Cancer Histology, thus establishing a benchmark for the rigorous evaluation of patch-level multi-class classification methodologies. Intensive experiments show that our method significantly outperforms the supervised baseline and state-of-the-art SSL methods with top-1 classification accuracy of 87.56%. Our method also achieves comparable results while utilizing a mere 1% of labeled data, a stark contrast to the 100% labeled data requirement of other approaches. Source code: https://github.com/nvtien457/NearbyPatchCL

{{</citation>}}


### (95/149) LMDrive: Closed-Loop End-to-End Driving with Large Language Models (Hao Shao et al., 2023)

{{<citation>}}

Hao Shao, Yuxuan Hu, Letian Wang, Steven L. Waslander, Yu Liu, Hongsheng Li. (2023)  
**LMDrive: Closed-Loop End-to-End Driving with Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-RO, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07488v1)  

---


**ABSTRACT**  
Despite significant recent progress in the field of autonomous driving, modern methods still struggle and can incur serious accidents when encountering long-tail unforeseen events and challenging urban scenarios. On the one hand, large language models (LLM) have shown impressive reasoning capabilities that approach "Artificial General Intelligence". On the other hand, previous autonomous driving methods tend to rely on limited-format inputs (e.g. sensor data and navigation waypoints), restricting the vehicle's ability to understand language information and interact with humans. To this end, this paper introduces LMDrive, a novel language-guided, end-to-end, closed-loop autonomous driving framework. LMDrive uniquely processes and integrates multi-modal sensor data with natural language instructions, enabling interaction with humans and navigation software in realistic instructional settings. To facilitate further research in language-based closed-loop autonomous driving, we also publicly release the corresponding dataset which includes approximately 64K instruction-following data clips, and the LangAuto benchmark that tests the system's ability to handle complex instructions and challenging driving scenarios. Extensive closed-loop experiments are conducted to demonstrate LMDrive's effectiveness. To the best of our knowledge, we're the very first work to leverage LLMs for closed-loop end-to-end autonomous driving. Codes can be found at https://github.com/opendilab/LMDrive

{{</citation>}}


### (96/149) MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception (Yiran Qin et al., 2023)

{{<citation>}}

Yiran Qin, Enshen Zhou, Qichang Liu, Zhenfei Yin, Lu Sheng, Ruimao Zhang, Yu Qiao, Jing Shao. (2023)  
**MP5: A Multi-modal Open-ended Embodied System in Minecraft via Active Perception**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07472v2)  

---


**ABSTRACT**  
It is a long-lasting goal to design an embodied system that can solve long-horizon open-world tasks in human-like ways. However, existing approaches usually struggle with compound difficulties caused by the logic-aware decomposition and context-aware execution of these tasks. To this end, we introduce MP5, an open-ended multimodal embodied system built upon the challenging Minecraft simulator, which can decompose feasible sub-objectives, design sophisticated situation-aware plans, and perform embodied action control, with frequent communication with a goal-conditioned active perception scheme. Specifically, MP5 is developed on top of recent advances in Multimodal Large Language Models (MLLMs), and the system is modulated into functional modules that can be scheduled and collaborated to ultimately solve pre-defined context- and process-dependent tasks. Extensive experiments prove that MP5 can achieve a 22% success rate on difficult process-dependent tasks and a 91% success rate on tasks that heavily depend on the context. Moreover, MP5 exhibits a remarkable ability to address many open-ended tasks that are entirely novel.

{{</citation>}}


### (97/149) Efficient Object Detection in Autonomous Driving using Spiking Neural Networks: Performance, Energy Consumption Analysis, and Insights into Open-set Object Discovery (Aitor Martinez Seras et al., 2023)

{{<citation>}}

Aitor Martinez Seras, Javier Del Ser, Pablo Garcia-Bringas. (2023)  
**Efficient Object Detection in Autonomous Driving using Spiking Neural Networks: Performance, Energy Consumption Analysis, and Insights into Open-set Object Discovery**  

---
Primary Category: cs.CV  
Categories: 68, I-2; I-4; I-5, cs-AI, cs-CV, cs-LG, cs-NE, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.07466v1)  

---


**ABSTRACT**  
Besides performance, efficiency is a key design driver of technologies supporting vehicular perception. Indeed, a well-balanced trade-off between performance and energy consumption is crucial for the sustainability of autonomous vehicles. In this context, the diversity of real-world contexts in which autonomous vehicles can operate motivates the need for empowering perception models with the capability to detect, characterize and identify newly appearing objects by themselves. In this manuscript we elaborate on this threefold conundrum (performance, efficiency and open-world learning) for object detection modeling tasks over image data collected from vehicular scenarios. Specifically, we show that well-performing and efficient models can be realized by virtue of Spiking Neural Networks (SNNs), reaching competitive levels of detection performance when compared to their non-spiking counterparts at dramatic energy consumption savings (up to 85%) and a slightly improved robustness against image noise. Our experiments herein offered also expose qualitatively the complexity of detecting new objects based on the preliminary results of a simple approach to discriminate potential object proposals in the captured image.

{{</citation>}}


### (98/149) Medical Image Classification Using Transfer Learning and Chaos Game Optimization on the Internet of Medical Things (Alhassan Mabrouk et al., 2023)

{{<citation>}}

Alhassan Mabrouk, Abdelghani Dahou, Mohamed Abd Elaziz, Rebeca P. Díaz Redondo, Mohammed Kayed. (2023)  
**Medical Image Classification Using Transfer Learning and Chaos Game Optimization on the Internet of Medical Things**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2312.07437v1)  

---


**ABSTRACT**  
The Internet of Medical Things (IoMT) has dramatically benefited medical professionals that patients and physicians can access from all regions. Although the automatic detection and prediction of diseases such as melanoma and leukemia is still being researched and studied in IoMT, existing approaches are not able to achieve a high degree of efficiency. Thus, with a new approach that provides better results, patients would access the adequate treatments earlier and the death rate would be reduced. Therefore, this paper introduces an IoMT proposal for medical images classification that may be used anywhere, i.e. it is an ubiquitous approach. It was design in two stages: first, we employ a Transfer Learning (TL)-based method for feature extraction, which is carried out using MobileNetV3; second, we use the Chaos Game Optimization (CGO) for feature selection, with the aim of excluding unnecessary features and improving the performance, which is key in IoMT. Our methodology was evaluated using ISIC-2016, PH2, and Blood-Cell datasets. The experimental results indicated that the proposed approach obtained an accuracy of 88.39% on ISIC-2016, 97.52% on PH2, and 88.79% on Blood-cell. Moreover, our approach had successful performances for the metrics employed compared to other existing methods.

{{</citation>}}


### (99/149) Cross-modal Contrastive Learning with Asymmetric Co-attention Network for Video Moment Retrieval (Love Panta et al., 2023)

{{<citation>}}

Love Panta, Prashant Shrestha, Brabeem Sapkota, Amrita Bhattarai, Suresh Manandhar, Anand Kumar Sah. (2023)  
**Cross-modal Contrastive Learning with Asymmetric Co-attention Network for Video Moment Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.07435v1)  

---


**ABSTRACT**  
Video moment retrieval is a challenging task requiring fine-grained interactions between video and text modalities. Recent work in image-text pretraining has demonstrated that most existing pretrained models suffer from information asymmetry due to the difference in length between visual and textual sequences. We question whether the same problem also exists in the video-text domain with an auxiliary need to preserve both spatial and temporal information. Thus, we evaluate a recently proposed solution involving the addition of an asymmetric co-attention network for video grounding tasks. Additionally, we incorporate momentum contrastive loss for robust, discriminative representation learning in both modalities. We note that the integration of these supplementary modules yields better performance compared to state-of-the-art models on the TACoS dataset and comparable results on ActivityNet Captions, all while utilizing significantly fewer parameters with respect to baseline.

{{</citation>}}


### (100/149) Attention Based Encoder Decoder Model for Video Captioning in Nepali (2023) (Kabita Parajuli et al., 2023)

{{<citation>}}

Kabita Parajuli, Shashidhar Ram Joshi. (2023)  
**Attention Based Encoder Decoder Model for Video Captioning in Nepali (2023)**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, BLEU, Google, LSTM, Microsoft  
[Paper Link](http://arxiv.org/abs/2312.07418v1)  

---


**ABSTRACT**  
Video captioning in Nepali, a language written in the Devanagari script, presents a unique challenge due to the lack of existing academic work in this domain. This work develops a novel encoder-decoder paradigm for Nepali video captioning to tackle this difficulty. LSTM and GRU sequence-to-sequence models are used in the model to produce related textual descriptions based on features retrieved from video frames using CNNs. Using Google Translate and manual post-editing, a Nepali video captioning dataset is generated from the Microsoft Research Video Description Corpus (MSVD) dataset created using Google Translate, and manual post-editing work. The efficacy of the model for Devanagari-scripted video captioning is demonstrated by BLEU, METOR, and ROUGE measures, which are used to assess its performance.

{{</citation>}}


### (101/149) Turbo: Informativity-Driven Acceleration Plug-In for Vision-Language Models (Chen Ju et al., 2023)

{{<citation>}}

Chen Ju, Haicheng Wang, Zeqian Li, Xu Chen, Zhonghua Zhai, Weilin Huang, Shuai Xiao. (2023)  
**Turbo: Informativity-Driven Acceleration Plug-In for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.07408v1)  

---


**ABSTRACT**  
Vision-Language Large Models (VLMs) have become primary backbone of AI, due to the impressive performance. However, their expensive computation costs, i.e., throughput and delay, impede potentials in real-world scenarios. To achieve acceleration for VLMs, most existing methods focus on the model perspective: pruning, distillation, quantification, but completely overlook the data-perspective redundancy. To fill the overlook, this paper pioneers the severity of data redundancy, and designs one plug-and-play Turbo module guided by information degree to prune inefficient tokens from visual or textual data. In pursuit of efficiency-performance trade-offs, information degree takes two key factors into consideration: mutual redundancy and semantic value. Concretely, the former evaluates the data duplication between sequential tokens; while the latter evaluates each token by its contribution to the overall semantics. As a result, tokens with high information degree carry less redundancy and stronger semantics. For VLMs' calculation, Turbo works as a user-friendly plug-in that sorts data referring to information degree, utilizing only top-level ones to save costs. Its advantages are multifaceted, e.g., being generally compatible to various VLMs across understanding and generation, simple use without retraining and trivial engineering efforts. On multiple public VLMs benchmarks, we conduct extensive experiments to reveal the gratifying acceleration of Turbo, under negligible performance drop.

{{</citation>}}


### (102/149) Eroding Trust In Aerial Imagery: Comprehensive Analysis and Evaluation Of Adversarial Attacks In Geospatial Systems (Michael Lanier et al., 2023)

{{<citation>}}

Michael Lanier, Aayush Dhakal, Zhexiao Xiong, Arthur Li, Nathan Jacobs, Yevgeniy Vorobeychik. (2023)  
**Eroding Trust In Aerial Imagery: Comprehensive Analysis and Evaluation Of Adversarial Attacks In Geospatial Systems**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.07389v1)  

---


**ABSTRACT**  
In critical operations where aerial imagery plays an essential role, the integrity and trustworthiness of data are paramount. The emergence of adversarial attacks, particularly those that exploit control over labels or employ physically feasible trojans, threatens to erode that trust, making the analysis and mitigation of these attacks a matter of urgency. We demonstrate how adversarial attacks can degrade confidence in geospatial systems, specifically focusing on scenarios where the attacker's control over labels is restricted and the use of realistic threat vectors. Proposing and evaluating several innovative attack methodologies, including those tailored to overhead images, we empirically show their threat to remote sensing systems using high-quality SpaceNet datasets. Our experimentation reflects the unique challenges posed by aerial imagery, and these preliminary results not only reveal the potential risks but also highlight the non-trivial nature of the problem compared to recent works.

{{</citation>}}


### (103/149) X4D-SceneFormer: Enhanced Scene Understanding on 4D Point Cloud Videos through Cross-modal Knowledge Transfer (Linglin Jing et al., 2023)

{{<citation>}}

Linglin Jing, Ying Xue, Xu Yan, Chaoda Zheng, Dong Wang, Ruimao Zhang, Zhigang Wang, Hui Fang, Bin Zhao, Zhen Li. (2023)  
**X4D-SceneFormer: Enhanced Scene Understanding on 4D Point Cloud Videos through Cross-modal Knowledge Transfer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.07378v1)  

---


**ABSTRACT**  
The field of 4D point cloud understanding is rapidly developing with the goal of analyzing dynamic 3D point cloud sequences. However, it remains a challenging task due to the sparsity and lack of texture in point clouds. Moreover, the irregularity of point cloud poses a difficulty in aligning temporal information within video sequences. To address these issues, we propose a novel cross-modal knowledge transfer framework, called X4D-SceneFormer. This framework enhances 4D-Scene understanding by transferring texture priors from RGB sequences using a Transformer architecture with temporal relationship mining. Specifically, the framework is designed with a dual-branch architecture, consisting of an 4D point cloud transformer and a Gradient-aware Image Transformer (GIT). During training, we employ multiple knowledge transfer techniques, including temporal consistency losses and masked self-attention, to strengthen the knowledge transfer between modalities. This leads to enhanced performance during inference using single-modal 4D point cloud inputs. Extensive experiments demonstrate the superior performance of our framework on various 4D point cloud video understanding tasks, including action recognition, action segmentation and semantic segmentation. The results achieve 1st places, i.e., 85.3% (+7.9%) accuracy and 47.3% (+5.0%) mIoU for 4D action segmentation and semantic segmentation, on the HOI4D challenge\footnote{\url{http://www.hoi4d.top/}.}, outperforming previous state-of-the-art by a large margin. We release the code at https://github.com/jinglinglingling/X4D

{{</citation>}}


### (104/149) Adversarial Semi-Supervised Domain Adaptation for Semantic Segmentation: A New Role for Labeled Target Samples (Marwa Kechaou et al., 2023)

{{<citation>}}

Marwa Kechaou, Mokhtar Z. Alaya, Romain Hérault, Gilles Gasso. (2023)  
**Adversarial Semi-Supervised Domain Adaptation for Semantic Segmentation: A New Role for Labeled Target Samples**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.07370v1)  

---


**ABSTRACT**  
Adversarial learning baselines for domain adaptation (DA) approaches in the context of semantic segmentation are under explored in semi-supervised framework. These baselines involve solely the available labeled target samples in the supervision loss. In this work, we propose to enhance their usefulness on both semantic segmentation and the single domain classifier neural networks. We design new training objective losses for cases when labeled target data behave as source samples or as real target samples. The underlying rationale is that considering the set of labeled target samples as part of source domain helps reducing the domain discrepancy and, hence, improves the contribution of the adversarial loss. To support our approach, we consider a complementary method that mixes source and labeled target data, then applies the same adaptation process. We further propose an unsupervised selection procedure using entropy to optimize the choice of labeled target samples for adaptation. We illustrate our findings through extensive experiments on the benchmarks GTA5, SYNTHIA, and Cityscapes. The empirical evaluation highlights competitive performance of our proposed approach.

{{</citation>}}


### (105/149) Collapse-Oriented Adversarial Training with Triplet Decoupling for Robust Image Retrieval (Qiwei Tian et al., 2023)

{{<citation>}}

Qiwei Tian, Chenhao Lin, Qian Li, Zhengyu Zhao, Chao Shen. (2023)  
**Collapse-Oriented Adversarial Training with Triplet Decoupling for Robust Image Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2312.07364v1)  

---


**ABSTRACT**  
Adversarial training has achieved substantial performance in defending image retrieval systems against adversarial examples. However, existing studies still suffer from two major limitations: model collapse and weak adversary. This paper addresses these two limitations by proposing collapse-oriented (COLO) adversarial training with triplet decoupling (TRIDE). Specifically, COLO prevents model collapse by temporally orienting the perturbation update direction with a new collapse metric, while TRIDE yields a strong adversary by spatially decoupling the update targets of perturbation into the anchor and the two candidates of a triplet. Experimental results demonstrate that our COLO-TRIDE outperforms the current state of the art by 7% on average over 10 robustness metrics and across 3 popular datasets. In addition, we identify the fairness limitations of commonly used robustness metrics in image retrieval and propose a new metric for more meaningful robustness evaluation. Codes will be made publicly available on GitHub.

{{</citation>}}


### (106/149) Expand-and-Quantize: Unsupervised Semantic Segmentation Using High-Dimensional Space and Product Quantization (Jiyoung Kim et al., 2023)

{{<citation>}}

Jiyoung Kim, Kyuhong Shim, Insu Lee, Byonghyo Shim. (2023)  
**Expand-and-Quantize: Unsupervised Semantic Segmentation Using High-Dimensional Space and Product Quantization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.07342v1)  

---


**ABSTRACT**  
Unsupervised semantic segmentation (USS) aims to discover and recognize meaningful categories without any labels. For a successful USS, two key abilities are required: 1) information compression and 2) clustering capability. Previous methods have relied on feature dimension reduction for information compression, however, this approach may hinder the process of clustering. In this paper, we propose a novel USS framework called Expand-and-Quantize Unsupervised Semantic Segmentation (EQUSS), which combines the benefits of high-dimensional spaces for better clustering and product quantization for effective information compression. Our extensive experiments demonstrate that EQUSS achieves state-of-the-art results on three standard benchmarks. In addition, we analyze the entropy of USS features, which is the first step towards understanding USS from the perspective of information theory.

{{</citation>}}


### (107/149) Scalable Motion Style Transfer with Constrained Diffusion Generation (Wenjie Yin et al., 2023)

{{<citation>}}

Wenjie Yin, Yi Yu, Hang Yin, Danica Kragic, Mårten Björkman. (2023)  
**Scalable Motion Style Transfer with Constrained Diffusion Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2312.07311v1)  

---


**ABSTRACT**  
Current training of motion style transfer systems relies on consistency losses across style domains to preserve contents, hindering its scalable application to a large number of domains and private data. Recent image transfer works show the potential of independent training on each domain by leveraging implicit bridging between diffusion models, with the content preservation, however, limited to simple data patterns. We address this by imposing biased sampling in backward diffusion while maintaining the domain independence in the training stage. We construct the bias from the source domain keyframes and apply them as the gradient of content constraints, yielding a framework with keyframe manifold constraint gradients (KMCGs). Our validation demonstrates the success of training separate models to transfer between as many as ten dance motion styles. Comprehensive experiments find a significant improvement in preserving motion contents in comparison to baseline and ablative diffusion-based style transfer models. In addition, we perform a human study for a subjective assessment of the quality of generated dance motions. The results validate the competitiveness of KMCGs.

{{</citation>}}


### (108/149) Benchmarking Pretrained Vision Embeddings for Near- and Duplicate Detection in Medical Images (Tuan Truong et al., 2023)

{{<citation>}}

Tuan Truong, Farnaz Khun Jush, Matthias Lenga. (2023)  
**Benchmarking Pretrained Vision Embeddings for Near- and Duplicate Detection in Medical Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.07273v1)  

---


**ABSTRACT**  
Near- and duplicate image detection is a critical concern in the field of medical imaging. Medical datasets often contain similar or duplicate images from various sources, which can lead to significant performance issues and evaluation biases, especially in machine learning tasks due to data leakage between training and testing subsets. In this paper, we present an approach for identifying near- and duplicate 3D medical images leveraging publicly available 2D computer vision embeddings. We assessed our approach by comparing embeddings extracted from two state-of-the-art self-supervised pretrained models and two different vector index structures for similarity retrieval. We generate an experimental benchmark based on the publicly available Medical Segmentation Decathlon dataset. The proposed method yields promising results for near- and duplicate image detection achieving a mean sensitivity and specificity of 0.9645 and 0.8559, respectively.

{{</citation>}}


### (109/149) ProxyDet: Synthesizing Proxy Novel Classes via Classwise Mixup for Open Vocabulary Object Detection (Joonhyun Jeong et al., 2023)

{{<citation>}}

Joonhyun Jeong, Geondo Park, Jayeon Yoo, Hyungsik Jung, Heesu Kim. (2023)  
**ProxyDet: Synthesizing Proxy Novel Classes via Classwise Mixup for Open Vocabulary Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.07266v1)  

---


**ABSTRACT**  
Open-vocabulary object detection (OVOD) aims to recognize novel objects whose categories are not included in training set. In order to classify these unseen classes during training, many OVOD frameworks leverage the zero-shot capability of largely pretrained vision and language models, such as CLIP. To further improve generalization on the unseen novel classes, several approaches proposed to additionally train with pseudo region labeling on the external data sources that contain a substantial number of novel category labels beyond the existing training data. Albeit its simplicity, these pseudo-labeling methods still exhibit limited improvement with regard to the genuine novel classes that were not pseudo-labeled. In this paper, we present a novel, yet simple technique that helps generalization on the overall distribution of novel classes. Inspired by our observation that numerous novel classes reside within the convex hull constructed by the base (seen) classes in the CLIP embedding space, we propose to synthesize proxy-novel classes approximating novel classes via linear mixup between a pair of base classes. By training our detector with these synthetic proxy-novel classes, we effectively explore the embedding space of novel classes. The experimental results on various OVOD benchmarks such as LVIS and COCO demonstrate superior performance on novel classes compared to the other state-of-the-art methods.

{{</citation>}}


### (110/149) SSTA: Salient Spatially Transformed Attack (Renyang Liu et al., 2023)

{{<citation>}}

Renyang Liu, Wei Zhou, Sixin Wu, Jun Zhao, Kwok-Yan Lam. (2023)  
**SSTA: Salient Spatially Transformed Attack**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07258v1)  

---


**ABSTRACT**  
Extensive studies have demonstrated that deep neural networks (DNNs) are vulnerable to adversarial attacks, which brings a huge security risk to the further application of DNNs, especially for the AI models developed in the real world. Despite the significant progress that has been made recently, existing attack methods still suffer from the unsatisfactory performance of escaping from being detected by naked human eyes due to the formulation of adversarial example (AE) heavily relying on a noise-adding manner. Such mentioned challenges will significantly increase the risk of exposure and result in an attack to be failed. Therefore, in this paper, we propose the Salient Spatially Transformed Attack (SSTA), a novel framework to craft imperceptible AEs, which enhance the stealthiness of AEs by estimating a smooth spatial transform metric on a most critical area to generate AEs instead of adding external noise to the whole image. Compared to state-of-the-art baselines, extensive experiments indicated that SSTA could effectively improve the imperceptibility of the AEs while maintaining a 100\% attack success rate.

{{</citation>}}


### (111/149) Fast Training of Diffusion Transformer with Extreme Masking for 3D Point Clouds Generation (Shentong Mo et al., 2023)

{{<citation>}}

Shentong Mo, Enze Xie, Yue Wu, Junsong Chen, Matthias Nießner, Zhenguo Li. (2023)  
**Fast Training of Diffusion Transformer with Extreme Masking for 3D Point Clouds Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.07231v1)  

---


**ABSTRACT**  
Diffusion Transformers have recently shown remarkable effectiveness in generating high-quality 3D point clouds. However, training voxel-based diffusion models for high-resolution 3D voxels remains prohibitively expensive due to the cubic complexity of attention operators, which arises from the additional dimension of voxels. Motivated by the inherent redundancy of 3D compared to 2D, we propose FastDiT-3D, a novel masked diffusion transformer tailored for efficient 3D point cloud generation, which greatly reduces training costs. Specifically, we draw inspiration from masked autoencoders to dynamically operate the denoising process on masked voxelized point clouds. We also propose a novel voxel-aware masking strategy to adaptively aggregate background/foreground information from voxelized point clouds. Our method achieves state-of-the-art performance with an extreme masking ratio of nearly 99%. Moreover, to improve multi-category 3D generation, we introduce Mixture-of-Expert (MoE) in 3D diffusion model. Each category can learn a distinct diffusion path with different experts, relieving gradient conflict. Experimental results on the ShapeNet dataset demonstrate that our method achieves state-of-the-art high-fidelity and diverse 3D point cloud generation performance. Our FastDiT-3D improves 1-Nearest Neighbor Accuracy and Coverage metrics when generating 128-resolution voxel point clouds, using only 6.5% of the original training cost.

{{</citation>}}


### (112/149) Transferring CLIP's Knowledge into Zero-Shot Point Cloud Semantic Segmentation (Yuanbin Wang et al., 2023)

{{<citation>}}

Yuanbin Wang, Shaofei Huang, Yulu Gao, Zhen Wang, Rui Wang, Kehua Sheng, Bo Zhang, Si Liu. (2023)  
**Transferring CLIP's Knowledge into Zero-Shot Point Cloud Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.07221v1)  

---


**ABSTRACT**  
Traditional 3D segmentation methods can only recognize a fixed range of classes that appear in the training set, which limits their application in real-world scenarios due to the lack of generalization ability. Large-scale visual-language pre-trained models, such as CLIP, have shown their generalization ability in the zero-shot 2D vision tasks, but are still unable to be applied to 3D semantic segmentation directly. In this work, we focus on zero-shot point cloud semantic segmentation and propose a simple yet effective baseline to transfer the visual-linguistic knowledge implied in CLIP to point cloud encoder at both feature and output levels. Both feature-level and output-level alignments are conducted between 2D and 3D encoders for effective knowledge transfer. Concretely, a Multi-granularity Cross-modal Feature Alignment (MCFA) module is proposed to align 2D and 3D features from global semantic and local position perspectives for feature-level alignment. For the output level, per-pixel pseudo labels of unseen classes are extracted using the pre-trained CLIP model as supervision for the 3D segmentation model to mimic the behavior of the CLIP image encoder. Extensive experiments are conducted on two popular benchmarks of point cloud segmentation. Our method outperforms significantly previous state-of-the-art methods under zero-shot setting (+29.2% mIoU on SemanticKITTI and 31.8% mIoU on nuScenes), and further achieves promising results in the annotation-free point cloud semantic segmentation setting, showing its great potential for label-efficient learning.

{{</citation>}}


### (113/149) MCFNet: Multi-scale Covariance Feature Fusion Network for Real-time Semantic Segmentation (Xiaojie Fang et al., 2023)

{{<citation>}}

Xiaojie Fang, Xingguo Song, Xiangyin Meng, Xu Fang, Sheng Jin. (2023)  
**MCFNet: Multi-scale Covariance Feature Fusion Network for Real-time Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, stat-ML  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.07207v1)  

---


**ABSTRACT**  
The low-level spatial detail information and high-level semantic abstract information are both essential to the semantic segmentation task. The features extracted by the deep network can obtain rich semantic information, while a lot of spatial information is lost. However, how to recover spatial detail information effectively and fuse it with high-level semantics has not been well addressed so far. In this paper, we propose a new architecture based on Bilateral Segmentation Network (BiseNet) called Multi-scale Covariance Feature Fusion Network (MCFNet). Specifically, this network introduces a new feature refinement module and a new feature fusion module. Furthermore, a gating unit named L-Gate is proposed to filter out invalid information and fuse multi-scale features. We evaluate our proposed model on Cityscapes, CamVid datasets and compare it with the state-of-the-art methods. Extensive experiments show that our method achieves competitive success. On Cityscapes, we achieve 75.5% mIOU with a speed of 151.3 FPS.

{{</citation>}}


### (114/149) Semi-supervised Active Learning for Video Action Detection (Aayush Singh et al., 2023)

{{<citation>}}

Aayush Singh, Aayush J Rana, Akash Kumar, Shruti Vyas, Yogesh Singh Rawat. (2023)  
**Semi-supervised Active Learning for Video Action Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2312.07169v1)  

---


**ABSTRACT**  
In this work, we focus on label efficient learning for video action detection. We develop a novel semi-supervised active learning approach which utilizes both labeled as well as unlabeled data along with informative sample selection for action detection. Video action detection requires spatio-temporal localization along with classification, which poses several challenges for both active learning informative sample selection as well as semi-supervised learning pseudo label generation. First, we propose NoiseAug, a simple augmentation strategy which effectively selects informative samples for video action detection. Next, we propose fft-attention, a novel technique based on high-pass filtering which enables effective utilization of pseudo label for SSL in video action detection by emphasizing on relevant activity region within a video. We evaluate the proposed approach on three different benchmark datasets, UCF-101-24, JHMDB-21, and Youtube-VOS. First, we demonstrate its effectiveness on video action detection where the proposed approach outperforms prior works in semi-supervised and weakly-supervised learning along with several baseline approaches in both UCF101-24 and JHMDB-21. Next, we also show its effectiveness on Youtube-VOS for video object segmentation demonstrating its generalization capability for other dense prediction tasks in videos.

{{</citation>}}


### (115/149) Language-Guided Transformer for Federated Multi-Label Classification (I-Jieh Liu et al., 2023)

{{<citation>}}

I-Jieh Liu, Ci-Siang Lin, Fu-En Yang, Yu-Chiang Frank Wang. (2023)  
**Language-Guided Transformer for Federated Multi-Label Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.07165v1)  

---


**ABSTRACT**  
Federated Learning (FL) is an emerging paradigm that enables multiple users to collaboratively train a robust model in a privacy-preserving manner without sharing their private data. Most existing approaches of FL only consider traditional single-label image classification, ignoring the impact when transferring the task to multi-label image classification. Nevertheless, it is still challenging for FL to deal with user heterogeneity in their local data distribution in the real-world FL scenario, and this issue becomes even more severe in multi-label image classification. Inspired by the recent success of Transformers in centralized settings, we propose a novel FL framework for multi-label classification. Since partial label correlation may be observed by local clients during training, direct aggregation of locally updated models would not produce satisfactory performances. Thus, we propose a novel FL framework of Language-Guided Transformer (FedLGT) to tackle this challenging task, which aims to exploit and transfer knowledge across different clients for learning a robust global model. Through extensive experiments on various multi-label datasets (e.g., FLAIR, MS-COCO, etc.), we show that our FedLGT is able to achieve satisfactory performance and outperforms standard FL techniques under multi-label FL scenarios. Code is available at https://github.com/Jack24658735/FedLGT.

{{</citation>}}


### (116/149) Image Content Generation with Causal Reasoning (Xiaochuan Li et al., 2023)

{{<citation>}}

Xiaochuan Li, Baoyu Fan, Runze Zhang, Liang Jin, Di Wang, Zhenhua Guo, Yaqian Zhao, Rengang Li. (2023)  
**Image Content Generation with Causal Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: AI, ChatGPT, GPT, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.07132v1)  

---


**ABSTRACT**  
The emergence of ChatGPT has once again sparked research in generative artificial intelligence (GAI). While people have been amazed by the generated results, they have also noticed the reasoning potential reflected in the generated textual content. However, this current ability for causal reasoning is primarily limited to the domain of language generation, such as in models like GPT-3. In visual modality, there is currently no equivalent research. Considering causal reasoning in visual content generation is significant. This is because visual information contains infinite granularity. Particularly, images can provide more intuitive and specific demonstrations for certain reasoning tasks, especially when compared to coarse-grained text. Hence, we propose a new image generation task called visual question answering with image (VQAI) and establish a dataset of the same name based on the classic \textit{Tom and Jerry} animated series. Additionally, we develop a new paradigm for image generation to tackle the challenges of this task. Finally, we perform extensive experiments and analyses, including visualizations of the generated content and discussions on the potentials and limitations. The code and data are publicly available under the license of CC BY-NC-SA 4.0 for academic and non-commercial usage. The code and dataset are publicly available at: https://github.com/IEIT-AGI/MIX-Shannon/blob/main/projects/VQAI/lgd_vqai.md.

{{</citation>}}


### (117/149) Efficient Few-Shot Clinical Task Adaptation with Large Language Models (Kaipeng Zheng et al., 2023)

{{<citation>}}

Kaipeng Zheng, Weiran Huang, Lichao Sun. (2023)  
**Efficient Few-Shot Clinical Task Adaptation with Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Clinical, Few-Shot, Language Model  
[Paper Link](http://arxiv.org/abs/2312.07125v1)  

---


**ABSTRACT**  
Few-shot learning has been studied to adapt models to tasks with very few samples. It holds profound significance, particularly in clinical tasks, due to the high annotation cost of medical images. Several works have explored few-shot learning on medical images, yet they still require a large number of medical images for pre-training models to gain domain-specific priors. Vision foundation models recently have achieved remarkable success in natural images. Hence, adapting rapidly advancing vision foundation models from natural images to few-shot clinical tasks holds great promise. MedFMC has recently organized a challenge to shed more light on this topic at NeurIPS 2023. In this work, we present our challenge solution. We observe that a simple variant of fine-tuning with partial freezing shows remarkable performance. Empirical evidence demonstrates that this approach could outperform various common fine-tuning methods under limited sample sizes. Additionally, we explore enhanced utilization of semantic supervision to boost performance. We propose a novel approach that contextualizes labels via large language models (LLMs). Our findings reveal that the context generated by LLMs significantly enhances the discrimination of semantic embeddings for similar categories, resulting in a notable performance improvement of 3%-5% in 1-shot settings compared to commonly employed one-hot labels and other semantic supervision methods. Our solution secures the 1st place in the MedFMC challenge.

{{</citation>}}


### (118/149) Pre-trained Universal Medical Image Transformer (Lingxiao Luo et al., 2023)

{{<citation>}}

Lingxiao Luo, Xuanzhong Chen, Bingda Tang, Xinsheng Chen, Chengpeng Hu, Yujiang Li, Rong Han, Ting Chen. (2023)  
**Pre-trained Universal Medical Image Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.07630v1)  

---


**ABSTRACT**  
Self-supervised learning has emerged as a viable method to leverage the abundance of unlabeled medical imaging data, addressing the challenge of labeled data scarcity in medical image analysis. In particular, masked image modeling (MIM) with visual token reconstruction has shown promising results in the general computer vision (CV) domain and serves as a candidate for medical image analysis. However, the presence of heterogeneous 2D and 3D medical images often limits the volume and diversity of training data that can be effectively used for a single model structure. In this work, we propose a spatially adaptive convolution (SAC) module, which adaptively adjusts convolution parameters based on the voxel spacing of the input images. Employing this SAC module, we build a universal visual tokenizer and a universal Vision Transformer (ViT) capable of effectively processing a wide range of medical images with various imaging modalities and spatial properties. Moreover, in order to enhance the robustness of the visual tokenizer's reconstruction objective for MIM, we suggest to generalize the discrete token output of the visual tokenizer to a probabilistic soft token. We show that the generalized soft token representation can be effectively integrated with the prior distribution regularization through a constructive interpretation. As a result, we pre-train a universal visual tokenizer followed by a universal ViT via visual token reconstruction on 55 public medical image datasets, comprising over 9 million 2D slices (including over 48,000 3D images). This represents the largest, most comprehensive, and diverse dataset for pre-training 3D medical image models to our knowledge. Experimental results on downstream medical image classification and segmentation tasks demonstrate the superior performance of our model and improved label efficiency.

{{</citation>}}


### (119/149) ThinkBot: Embodied Instruction Following with Thought Chain Reasoning (Guanxing Lu et al., 2023)

{{<citation>}}

Guanxing Lu, Ziwei Wang, Changliu Liu, Jiwen Lu, Yansong Tang. (2023)  
**ThinkBot: Embodied Instruction Following with Thought Chain Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2312.07062v2)  

---


**ABSTRACT**  
Embodied Instruction Following (EIF) requires agents to complete human instruction by interacting objects in complicated surrounding environments. Conventional methods directly consider the sparse human instruction to generate action plans for agents, which usually fail to achieve human goals because of the instruction incoherence in action descriptions. On the contrary, we propose ThinkBot that reasons the thought chain in human instruction to recover the missing action descriptions, so that the agent can successfully complete human goals by following the coherent instruction. Specifically, we first design an instruction completer based on large language models to recover the missing actions with interacted objects between consecutive human instruction, where the perceived surrounding environments and the completed sub-goals are considered for instruction completion. Based on the partially observed scene semantic maps, we present an object localizer to infer the position of interacted objects for agents to achieve complex human goals. Extensive experiments in the simulated environment show that our ThinkBot outperforms the state-of-the-art EIF methods by a sizable margin in both success rate and execution efficiency.

{{</citation>}}


### (120/149) MaxQ: Multi-Axis Query for N:M Sparsity Network (Jingyang Xiang et al., 2023)

{{<citation>}}

Jingyang Xiang, Siqi Li, Junhao Chen, Zhuangzhi Chen, Tianxin Huang, Linpeng Peng, Yong Liu. (2023)  
**MaxQ: Multi-Axis Query for N:M Sparsity Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.07061v1)  

---


**ABSTRACT**  
N:M sparsity has received increasing attention due to its remarkable performance and latency trade-off compared with structured and unstructured sparsity. However, existing N:M sparsity methods do not differentiate the relative importance of weights among blocks and leave important weights underappreciated. Besides, they directly apply N:M sparsity to the whole network, which will cause severe information loss. Thus, they are still sub-optimal. In this paper, we propose an efficient and effective Multi-Axis Query methodology, dubbed as MaxQ, to rectify these problems. During the training, MaxQ employs a dynamic approach to generate soft N:M masks, considering the weight importance across multiple axes. This method enhances the weights with more importance and ensures more effective updates. Meanwhile, a sparsity strategy that gradually increases the percentage of N:M weight blocks is applied, which allows the network to heal from the pruning-induced damage progressively. During the runtime, the N:M soft masks can be precomputed as constants and folded into weights without causing any distortion to the sparse pattern and incurring additional computational overhead. Comprehensive experiments demonstrate that MaxQ achieves consistent improvements across diverse CNN architectures in various computer vision tasks, including image classification, object detection and instance segmentation. For ResNet50 with 1:16 sparse pattern, MaxQ can achieve 74.6\% top-1 accuracy on ImageNet and improve by over 2.8\% over the state-of-the-art.

{{</citation>}}


### (121/149) Adjustable Robust Transformer for High Myopia Screening in Optical Coherence Tomography (Xiao Ma et al., 2023)

{{<citation>}}

Xiao Ma, Zetian Zhang, Zexuan Ji, Kun Huang, Na Su, Songtao Yuan, Qiang Chen. (2023)  
**Adjustable Robust Transformer for High Myopia Screening in Optical Coherence Tomography**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.07052v1)  

---


**ABSTRACT**  
Myopia is a manifestation of visual impairment caused by an excessively elongated eyeball. Image data is critical material for studying high myopia and pathological myopia. Measurements of spherical equivalent and axial length are the gold standards for identifying high myopia, but the available image data for matching them is scarce. In addition, the criteria for defining high myopia vary from study to study, and therefore the inclusion of samples in automated screening efforts requires an appropriate assessment of interpretability. In this work, we propose a model called adjustable robust transformer (ARTran) for high myopia screening of optical coherence tomography (OCT) data. Based on vision transformer, we propose anisotropic patch embedding (APE) to capture more discriminative features of high myopia. To make the model effective under variable screening conditions, we propose an adjustable class embedding (ACE) to replace the fixed class token, which changes the output to adapt to different conditions. Considering the confusion of the data at high myopia and low myopia threshold, we introduce the label noise learning strategy and propose a shifted subspace transition matrix (SST) to enhance the robustness of the model. Besides, combining the two structures proposed above, the model can provide evidence for uncertainty evaluation. The experimental results demonstrate the effectiveness and reliability of the proposed method. Code is available at: https://github.com/maxiao0234/ARTran.

{{</citation>}}


### (122/149) Edge Wasserstein Distance Loss for Oriented Object Detection (Yuke Zhu et al., 2023)

{{<citation>}}

Yuke Zhu, Yumeng Ruan, Zihua Xiong, Sheng Guo. (2023)  
**Edge Wasserstein Distance Loss for Oriented Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.07048v1)  

---


**ABSTRACT**  
Regression loss design is an essential topic for oriented object detection. Due to the periodicity of the angle and the ambiguity of width and height definition, traditional L1-distance loss and its variants have been suffered from the metric discontinuity and the square-like problem. As a solution, the distribution based methods show significant advantages by representing oriented boxes as distributions. Differing from exploited the Gaussian distribution to get analytical form of distance measure, we propose a novel oriented regression loss, Wasserstein Distance(EWD) loss, to alleviate the square-like problem. Specifically, for the oriented box(OBox) representation, we choose a specially-designed distribution whose probability density function is only nonzero over the edges. On this basis, we develop Wasserstein distance as the measure. Besides, based on the edge representation of OBox, the EWD loss can be generalized to quadrilateral and polynomial regression scenarios. Experiments on multiple popular datasets and different detectors show the effectiveness of the proposed method.

{{</citation>}}


### (123/149) Diff-OP3D: Bridging 2D Diffusion for Open Pose 3D Zero-Shot Classification (Weiguang Zhao et al., 2023)

{{<citation>}}

Weiguang Zhao, Guanyu Yang, Chaolong Yang, Chenru Jiang, Yuyao Yan, Rui Zhang, Kaizhu Huang. (2023)  
**Diff-OP3D: Bridging 2D Diffusion for Open Pose 3D Zero-Shot Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.07039v1)  

---


**ABSTRACT**  
With the explosive 3D data growth, the urgency of utilizing zero-shot learning to facilitate data labeling becomes evident. Recently, the methods via transferring Contrastive Language-Image Pre-training (CLIP) to 3D vision have made great progress in the 3D zero-shot classification task. However, these methods primarily focus on aligned pose 3D objects (ap-3os), overlooking the recognition of 3D objects with open poses (op-3os) typically encountered in real-world scenarios, such as an overturned chair or a lying teddy bear. To this end, we propose a more challenging benchmark for 3D open-pose zero-shot classification. Echoing our benchmark, we design a concise angle-refinement mechanism that automatically optimizes one ideal pose as well as classifies these op-3os. Furthermore, we make a first attempt to bridge 2D pre-trained diffusion model as a classifer to 3D zero-shot classification without any additional training. Such 2D diffusion to 3D objects proves vital in improving zero-shot classification for both ap-3os and op-3os. Our model notably improves by 3.5% and 15.8% on ModelNet10$^{\ddag}$ and McGill$^{\ddag}$ open pose benchmarks, respectively, and surpasses the current state-of-the-art by 6.8% on the aligned pose ModelNet10, affirming diffusion's efficacy in 3D zero-shot tasks.

{{</citation>}}


### (124/149) Multimodal Sentiment Analysis: Perceived vs Induced Sentiments (Aditi Aggarwal et al., 2023)

{{<citation>}}

Aditi Aggarwal, Deepika Varshney, Saurabh Patel. (2023)  
**Multimodal Sentiment Analysis: Perceived vs Induced Sentiments**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-SI, cs.CV  
Keywords: OCR, Sentiment Analysis, Twitter  
[Paper Link](http://arxiv.org/abs/2312.07627v1)  

---


**ABSTRACT**  
Social media has created a global network where people can easily access and exchange vast information. This information gives rise to a variety of opinions, reflecting both positive and negative viewpoints. GIFs stand out as a multimedia format offering a visually engaging way for users to communicate. In this research, we propose a multimodal framework that integrates visual and textual features to predict the GIF sentiment. It also incorporates attributes including face emotion detection and OCR generated captions to capture the semantic aspects of the GIF. The developed classifier achieves an accuracy of 82.7% on Twitter GIFs, which is an improvement over state-of-the-art models. Moreover, we have based our research on the ReactionGIF dataset, analysing the variance in sentiment perceived by the author and sentiment induced in the reader

{{</citation>}}


### (125/149) Mixed Pseudo Labels for Semi-Supervised Object Detection (Zeming Chen et al., 2023)

{{<citation>}}

Zeming Chen, Wenwei Zhang, Xinjiang Wang, Kai Chen, Zhi Wang. (2023)  
**Mixed Pseudo Labels for Semi-Supervised Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.07006v1)  

---


**ABSTRACT**  
While the pseudo-label method has demonstrated considerable success in semi-supervised object detection tasks, this paper uncovers notable limitations within this approach. Specifically, the pseudo-label method tends to amplify the inherent strengths of the detector while accentuating its weaknesses, which is manifested in the missed detection of pseudo-labels, particularly for small and tail category objects. To overcome these challenges, this paper proposes Mixed Pseudo Labels (MixPL), consisting of Mixup and Mosaic for pseudo-labeled data, to mitigate the negative impact of missed detections and balance the model's learning across different object scales. Additionally, the model's detection performance on tail categories is improved by resampling labeled data with relevant instances. Notably, MixPL consistently improves the performance of various detectors and obtains new state-of-the-art results with Faster R-CNN, FCOS, and DINO on COCO-Standard and COCO-Full benchmarks. Furthermore, MixPL also exhibits good scalability on large models, improving DINO Swin-L by 2.5% mAP and achieving nontrivial new records (60.2% mAP) on the COCO val2017 benchmark without extra annotations.

{{</citation>}}


### (126/149) Supervised Contrastive Learning for Fine-grained Chromosome Recognition (Ruijia Chang et al., 2023)

{{<citation>}}

Ruijia Chang, Suncheng Xiang, Chengyu Zhou, Kui Su, Dahong Qian, Jun Wang. (2023)  
**Supervised Contrastive Learning for Fine-grained Chromosome Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.07623v1)  

---


**ABSTRACT**  
Chromosome recognition is an essential task in karyotyping, which plays a vital role in birth defect diagnosis and biomedical research. However, existing classification methods face significant challenges due to the inter-class similarity and intra-class variation of chromosomes. To address this issue, we propose a supervised contrastive learning strategy that is tailored to train model-agnostic deep networks for reliable chromosome classification. This method enables extracting fine-grained chromosomal embeddings in latent space. These embeddings effectively expand inter-class boundaries and reduce intra-class variations, enhancing their distinctiveness in predicting chromosome types. On top of two large-scale chromosome datasets, we comprehensively validate the power of our contrastive learning strategy in boosting cutting-edge deep networks such as Transformers and ResNets. Extensive results demonstrate that it can significantly improve models' generalization performance, with an accuracy improvement up to +4.5%. Codes and pretrained models will be released upon acceptance of this work.

{{</citation>}}


### (127/149) Transformer-based No-Reference Image Quality Assessment via Supervised Contrastive Learning (Jinsong Shi et al., 2023)

{{<citation>}}

Jinsong Shi, Pan Gao, Jie Qin. (2023)  
**Transformer-based No-Reference Image Quality Assessment via Supervised Contrastive Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Attention, Contrastive Learning, QA, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.06995v1)  

---


**ABSTRACT**  
Image Quality Assessment (IQA) has long been a research hotspot in the field of image processing, especially No-Reference Image Quality Assessment (NR-IQA). Due to the powerful feature extraction ability, existing Convolution Neural Network (CNN) and Transformers based NR-IQA methods have achieved considerable progress. However, they still exhibit limited capability when facing unknown authentic distortion datasets. To further improve NR-IQA performance, in this paper, a novel supervised contrastive learning (SCL) and Transformer-based NR-IQA model SaTQA is proposed. We first train a model on a large-scale synthetic dataset by SCL (no image subjective score is required) to extract degradation features of images with various distortion types and levels. To further extract distortion information from images, we propose a backbone network incorporating the Multi-Stream Block (MSB) by combining the CNN inductive bias and Transformer long-term dependence modeling capability. Finally, we propose the Patch Attention Block (PAB) to obtain the final distorted image quality score by fusing the degradation features learned from contrastive learning with the perceptual distortion information extracted by the backbone network. Experimental results on seven standard IQA datasets show that SaTQA outperforms the state-of-the-art methods for both synthetic and authentic datasets. Code is available at https://github.com/I2-Multimedia-Lab/SaTQA

{{</citation>}}


### (128/149) Attacking the Loop: Adversarial Attacks on Graph-based Loop Closure Detection (Jonathan J. Y. Kim et al., 2023)

{{<citation>}}

Jonathan J. Y. Kim, Martin Urschler, Patricia J. Riddle, Jorg S. Wicker. (2023)  
**Attacking the Loop: Adversarial Attacks on Graph-based Loop Closure Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.06991v1)  

---


**ABSTRACT**  
With the advancement in robotics, it is becoming increasingly common for large factories and warehouses to incorporate visual SLAM (vSLAM) enabled automated robots that operate closely next to humans. This makes any adversarial attacks on vSLAM components potentially detrimental to humans working alongside them. Loop Closure Detection (LCD) is a crucial component in vSLAM that minimizes the accumulation of drift in mapping, since even a small drift can accumulate into a significant drift over time. A prior work by Kim et al., SymbioLCD2, unified visual features and semantic objects into a single graph structure for finding loop closure candidates. While this provided a performance improvement over visual feature-based LCD, it also created a single point of vulnerability for potential graph-based adversarial attacks. Unlike previously reported visual-patch based attacks, small graph perturbations are far more challenging to detect, making them a more significant threat. In this paper, we present Adversarial-LCD, a novel black-box evasion attack framework that employs an eigencentrality-based perturbation method and an SVM-RBF surrogate model with a Weisfeiler-Lehman feature extractor for attacking graph-based LCD. Our evaluation shows that the attack performance of Adversarial-LCD with the SVM-RBF surrogate model was superior to that of other machine learning surrogate algorithms, including SVM-linear, SVM-polynomial, and Bayesian classifier, demonstrating the effectiveness of our attack framework. Furthermore, we show that our eigencentrality-based perturbation method outperforms other algorithms, such as Random-walk and Shortest-path, highlighting the efficiency of Adversarial-LCD's perturbation selection method.

{{</citation>}}


### (129/149) CLASS-M: Adaptive stain separation-based contrastive learning with pseudo-labeling for histopathological image classification (Bodong Zhang et al., 2023)

{{<citation>}}

Bodong Zhang, Hamid Manoochehri, Man Minh Ho, Fahimeh Fooladgar, Yosep Chong, Beatrice S. Knudsen, Deepika Sirohi, Tolga Tasdizen. (2023)  
**CLASS-M: Adaptive stain separation-based contrastive learning with pseudo-labeling for histopathological image classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.06978v2)  

---


**ABSTRACT**  
Histopathological image classification is one of the critical aspects in medical image analysis. Due to the high expense associated with the labeled data in model training, semi-supervised learning methods have been proposed to alleviate the need of extensively labeled datasets. In this work, we propose a model for semi-supervised classification tasks on digital histopathological Hematoxylin and Eosin (H&E) images. We call the new model Contrastive Learning with Adaptive Stain Separation and MixUp (CLASS-M). Our model is formed by two main parts: contrastive learning between adaptively stain separated Hematoxylin images and Eosin images, and pseudo-labeling using MixUp. We compare our model with other state-of-the-art models on clear cell renal cell carcinoma (ccRCC) datasets from our institution and The Cancer Genome Atlas Program (TCGA). We demonstrate that our CLASS-M model has the best performance on both datasets. The contributions of different parts in our model are also analyzed.

{{</citation>}}


### (130/149) Hallucination Augmented Contrastive Learning for Multimodal Large Language Model (Chaoya Jiang et al., 2023)

{{<citation>}}

Chaoya Jiang, Haiyang Xu, Mengfan Dong, Jiaxing Chen, Wei Ye, Ming Yan, Qinghao Ye, Ji Zhang, Fei Huang, Shikun Zhang. (2023)  
**Hallucination Augmented Contrastive Learning for Multimodal Large Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.06968v2)  

---


**ABSTRACT**  
Multi-modal large language models (MLLMs) have been shown to efficiently integrate natural language with visual information to handle multi-modal tasks. However, MLLMs still face a fundamental limitation of hallucinations, where they tend to generate erroneous or fabricated information. In this paper, we address hallucinations in MLLMs from a novel perspective of representation learning. We first analyzed the representation distribution of textual and visual tokens in MLLM, revealing two important findings: 1) there is a significant gap between textual and visual representations, indicating unsatisfactory cross-modal representation alignment; 2) representations of texts that contain and do not contain hallucinations are entangled, making it challenging to distinguish them. These two observations inspire us with a simple yet effective method to mitigate hallucinations. Specifically, we introduce contrastive learning into MLLMs and use text with hallucination as hard negative examples, naturally bringing representations of non-hallucinative text and visual samples closer while pushing way representations of non-hallucinating and hallucinative text. We evaluate our method quantitatively and qualitatively, showing its effectiveness in reducing hallucination occurrences and improving performance across multiple benchmarks. On the MMhal-Bench benchmark, our method obtains a 34.66% /29.5% improvement over the baseline MiniGPT-4/LLaVA.

{{</citation>}}


### (131/149) IA2U: A Transfer Plugin with Multi-Prior for In-Air Model to Underwater (Jingchun Zhou et al., 2023)

{{<citation>}}

Jingchun Zhou, Qilin Gai, Weishi Zhang, Kin-man Lam, Xianping Fu, Ting Li, Chongyi Li. (2023)  
**IA2U: A Transfer Plugin with Multi-Prior for In-Air Model to Underwater**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.06955v1)  

---


**ABSTRACT**  
In underwater environments, variations in suspended particle concentration and turbidity cause severe image degradation, posing significant challenges to image enhancement (IE) and object detection (OD) tasks. Currently, in-air image enhancement and detection methods have made notable progress, but their application in underwater conditions is limited due to the complexity and variability of these environments. Fine-tuning in-air models saves high overhead and has more optional reference work than building an underwater model from scratch. To address these issues, we design a transfer plugin with multiple priors for converting in-air models to underwater applications, named IA2U. IA2U enables efficient application in underwater scenarios, thereby improving performance in Underwater IE and OD. IA2U integrates three types of underwater priors: the water type prior that characterizes the degree of image degradation, such as color and visibility; the degradation prior, focusing on differences in details and textures; and the sample prior, considering the environmental conditions at the time of capture and the characteristics of the photographed object. Utilizing a Transformer-like structure, IA2U employs these priors as query conditions and a joint task loss function to achieve hierarchical enhancement of task-level underwater image features, therefore considering the requirements of two different tasks, IE and OD. Experimental results show that IA2U combined with an in-air model can achieve superior performance in underwater image enhancement and object detection tasks. The code will be made publicly available.

{{</citation>}}


### (132/149) READ-PVLA: Recurrent Adapter with Partial Video-Language Alignment for Parameter-Efficient Transfer Learning in Low-Resource Video-Language Modeling (Thong Nguyen et al., 2023)

{{<citation>}}

Thong Nguyen, Xiaobao Wu, Xinshuai Dong, Khoi Le, Zhiyuan Hu, Cong-Duy Nguyen, See-Kiong Ng, Luu Anh Tuan. (2023)  
**READ-PVLA: Recurrent Adapter with Partial Video-Language Alignment for Parameter-Efficient Transfer Learning in Low-Resource Video-Language Modeling**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model, Low-Resource  
[Paper Link](http://arxiv.org/abs/2312.06950v1)  

---


**ABSTRACT**  
Fully fine-tuning pretrained large-scale transformer models has become a popular paradigm for video-language modeling tasks, such as temporal language grounding and video-language summarization. With a growing number of tasks and limited training data, such full fine-tuning approach leads to costly model storage and unstable training. To overcome these shortcomings, we introduce lightweight adapters to the pre-trained model and only update them at fine-tuning time. However, existing adapters fail to capture intrinsic temporal relations among video frames or textual words. Moreover, they neglect the preservation of critical task-related information that flows from the raw video-language input into the adapter's low-dimensional space. To address these issues, we first propose a novel REcurrent ADapter (READ) that employs recurrent computation to enable temporal modeling capability. Second, we propose Partial Video-Language Alignment (PVLA) objective via the use of partial optimal transport to maintain task-related information flowing into our READ modules. We validate our READ-PVLA framework through extensive experiments where READ-PVLA significantly outperforms all existing fine-tuning strategies on multiple low-resource temporal language grounding and video-language summarization benchmarks.

{{</citation>}}


### (133/149) Benchmarking Deep Learning Classifiers for SAR Automatic Target Recognition (Jacob Fein-Ashley et al., 2023)

{{<citation>}}

Jacob Fein-Ashley, Tian Ye, Rajgopal Kannan, Viktor Prasanna, Carl Busart. (2023)  
**Benchmarking Deep Learning Classifiers for SAR Automatic Target Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: GNN, Graph Neural Network, Transformer  
[Paper Link](http://arxiv.org/abs/2312.06940v1)  

---


**ABSTRACT**  
Synthetic Aperture Radar SAR Automatic Target Recognition ATR is a key technique of remote-sensing image recognition which can be supported by deep neural networks The existing works of SAR ATR mostly focus on improving the accuracy of the target recognition while ignoring the systems performance in terms of speed and storage which is critical to real-world applications of SAR ATR For decision-makers aiming to identify a proper deep learning model to deploy in a SAR ATR system it is important to understand the performance of different candidate deep learning models and determine the best model accordingly This paper comprehensively benchmarks several advanced deep learning models for SAR ATR with multiple distinct SAR imagery datasets Specifically we train and test five SAR image classifiers based on Residual Neural Networks ResNet18 ResNet34 ResNet50 Graph Neural Network GNN and Vision Transformer for Small-Sized Datasets (SS-ViT) We select three datasets MSTAR GBSAR and SynthWakeSAR that offer heterogeneity We evaluate and compare the five classifiers concerning their classification accuracy runtime performance in terms of inference throughput and analytical performance in terms of number of parameters number of layers model size and number of operations Experimental results show that the GNN classifier outperforms with respect to throughput and latency However it is also shown that no clear model winner emerges from all of our chosen metrics and a one model rules all case is doubtful in the domain of SAR ATR

{{</citation>}}


### (134/149) Toward Real Text Manipulation Detection: New Dataset and New Solution (Dongliang Luo et al., 2023)

{{<citation>}}

Dongliang Luo, Yuliang Liu, Rui Yang, Xianjin Liu, Jishen Zeng, Yu Zhou, Xiang Bai. (2023)  
**Toward Real Text Manipulation Detection: New Dataset and New Solution**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.06934v1)  

---


**ABSTRACT**  
With the surge in realistic text tampering, detecting fraudulent text in images has gained prominence for maintaining information security. However, the high costs associated with professional text manipulation and annotation limit the availability of real-world datasets, with most relying on synthetic tampering, which inadequately replicates real-world tampering attributes. To address this issue, we present the Real Text Manipulation (RTM) dataset, encompassing 14,250 text images, which include 5,986 manually and 5,258 automatically tampered images, created using a variety of techniques, alongside 3,006 unaltered text images for evaluating solution stability. Our evaluations indicate that existing methods falter in text forgery detection on the RTM dataset. We propose a robust baseline solution featuring a Consistency-aware Aggregation Hub and a Gated Cross Neighborhood-attention Fusion module for efficient multi-modal information fusion, supplemented by a Tampered-Authentic Contrastive Learning module during training, enriching feature representation distinction. This framework, extendable to other dual-stream architectures, demonstrated notable localization performance improvements of 7.33% and 6.38% on manual and overall manipulations, respectively. Our contributions aim to propel advancements in real-world text tampering detection. Code and dataset will be made available at https://github.com/DrLuo/RTM

{{</citation>}}


### (135/149) When Bio-Inspired Computing meets Deep Learning: Low-Latency, Accurate, & Energy-Efficient Spiking Neural Networks from Artificial Neural Networks (Gourav Datta et al., 2023)

{{<citation>}}

Gourav Datta, Zeyu Liu, James Diffenderfer, Bhavya Kailkhura, Peter A. Beerel. (2023)  
**When Bio-Inspired Computing meets Deep Learning: Low-Latency, Accurate, & Energy-Efficient Spiking Neural Networks from Artificial Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.06900v1)  

---


**ABSTRACT**  
Bio-inspired Spiking Neural Networks (SNN) are now demonstrating comparable accuracy to intricate convolutional neural networks (CNN), all while delivering remarkable energy and latency efficiency when deployed on neuromorphic hardware. In particular, ANN-to-SNN conversion has recently gained significant traction in developing deep SNNs with close to state-of-the-art (SOTA) test accuracy on complex image recognition tasks. However, advanced ANN-to-SNN conversion approaches demonstrate that for lossless conversion, the number of SNN time steps must equal the number of quantization steps in the ANN activation function. Reducing the number of time steps significantly increases the conversion error. Moreover, the spiking activity of the SNN, which dominates the compute energy in neuromorphic chips, does not reduce proportionally with the number of time steps. To mitigate the accuracy concern, we propose a novel ANN-to-SNN conversion framework, that incurs an exponentially lower number of time steps compared to that required in the SOTA conversion approaches. Our framework modifies the SNN integrate-and-fire (IF) neuron model with identical complexity and shifts the bias term of each batch normalization (BN) layer in the trained ANN. To mitigate the spiking activity concern, we propose training the source ANN with a fine-grained L1 regularizer with surrogate gradients that encourages high spike sparsity in the converted SNN. Our proposed framework thus yields lossless SNNs with ultra-low latency, ultra-low compute energy, thanks to the ultra-low timesteps and high spike sparsity, and ultra-high test accuracy, for example, 73.30% with only 4 time steps on the ImageNet dataset.

{{</citation>}}


## cs.DS (1)



### (136/149) Improved Frequency Estimation Algorithms with and without Predictions (Anders Aamand et al., 2023)

{{<citation>}}

Anders Aamand, Justin Y. Chen, Huy Lê Nguyen, Sandeep Silwal, Ali Vakilian. (2023)  
**Improved Frequency Estimation Algorithms with and without Predictions**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs-LG, cs.DS  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2312.07535v1)  

---


**ABSTRACT**  
Estimating frequencies of elements appearing in a data stream is a key task in large-scale data analysis. Popular sketching approaches to this problem (e.g., CountMin and CountSketch) come with worst-case guarantees that probabilistically bound the error of the estimated frequencies for any possible input. The work of Hsu et al. (2019) introduced the idea of using machine learning to tailor sketching algorithms to the specific data distribution they are being run on. In particular, their learning-augmented frequency estimation algorithm uses a learned heavy-hitter oracle which predicts which elements will appear many times in the stream. We give a novel algorithm, which in some parameter regimes, already theoretically outperforms the learning based algorithm of Hsu et al. without the use of any predictions. Augmenting our algorithm with heavy-hitter predictions further reduces the error and improves upon the state of the art. Empirically, our algorithms achieve superior performance in all experiments compared to prior approaches.

{{</citation>}}


## eess.AS (2)



### (137/149) NeuroHeed+: Improving Neuro-steered Speaker Extraction with Joint Auditory Attention Detection (Zexu Pan et al., 2023)

{{<citation>}}

Zexu Pan, Gordon Wichern, Francois G. Germain, Sameer Khurana, Jonathan Le Roux. (2023)  
**NeuroHeed+: Improving Neuro-steered Speaker Extraction with Joint Auditory Attention Detection**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.07513v1)  

---


**ABSTRACT**  
Neuro-steered speaker extraction aims to extract the listener's brain-attended speech signal from a multi-talker speech signal, in which the attention is derived from the cortical activity. This activity is usually recorded using electroencephalography (EEG) devices. Though promising, current methods often have a high speaker confusion error, where the interfering speaker is extracted instead of the attended speaker, degrading the listening experience. In this work, we aim to reduce the speaker confusion error in the neuro-steered speaker extraction model through a jointly fine-tuned auxiliary auditory attention detection model. The latter reinforces the consistency between the extracted target speech signal and the EEG representation, and also improves the EEG representation. Experimental results show that the proposed network significantly outperforms the baseline in terms of speaker confusion and overall signal quality in two-talker scenarios.

{{</citation>}}


### (138/149) w2v-SELD: A Sound Event Localization and Detection Framework for Self-Supervised Spatial Audio Pre-Training (Orlem Lima dos Santos et al., 2023)

{{<citation>}}

Orlem Lima dos Santos, Karen Rosero, Roberto de Alencar Lotufo. (2023)  
**w2v-SELD: A Sound Event Localization and Detection Framework for Self-Supervised Spatial Audio Pre-Training**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Event Detection, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.06907v1)  

---


**ABSTRACT**  
Sound Event Detection and Localization (SELD) constitutes a complex task that depends on extensive multichannel audio recordings with annotated sound events and their respective locations. In this paper, we introduce a self-supervised approach for SELD adapted from the pre-training methodology of wav2vec 2.0, which learns representations directly from raw audio data, eliminating the need for supervision. By applying this approach to SELD, we can leverage a substantial amount of unlabeled 3D audio data to learn robust representations of sound events and their locations. Our method comprises two primary stages: pre-training and fine-tuning. In the pre-training phase, unlabeled 3D audio datasets are utilized to train our w2v-SELD model, capturing intricate high-level features and contextual information inherent in audio signals. Subsequently, in the fine-tuning stage, a smaller dataset with labeled SELD data fine-tunes the pre-trained model. Experimental results on benchmark datasets demonstrate the effectiveness of the proposed self-supervised approach for SELD. The model surpasses baseline systems provided with the datasets and achieves competitive performance comparable to state-of-the-art supervised methods. The code and pre-trained parameters of our w2v-SELD model are available in this repository.

{{</citation>}}


## cs.MM (2)



### (139/149) Probing Commonsense Reasoning Capability of Text-to-Image Generative Models via Non-visual Description (Mianzhi Pan et al., 2023)

{{<citation>}}

Mianzhi Pan, Jianfei Li, Mingyue Yu, Zheng Ma, Kanzhi Cheng, Jianbing Zhang, Jiajun Chen. (2023)  
**Probing Commonsense Reasoning Capability of Text-to-Image Generative Models via Non-visual Description**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.07294v1)  

---


**ABSTRACT**  
Commonsense reasoning, the ability to make logical assumptions about daily scenes, is one core intelligence of human beings. In this work, we present a novel task and dataset for evaluating the ability of text-to-image generative models to conduct commonsense reasoning, which we call PAINTaboo. Given a description with few visual clues of one object, the goal is to generate images illustrating the object correctly. The dataset was carefully hand-curated and covered diverse object categories to analyze model performance comprehensively. Our investigation of several prevalent text-to-image generative models reveals that these models are not proficient in commonsense reasoning, as anticipated. We trust that PAINTaboo can improve our understanding of the reasoning abilities of text-to-image generative models.

{{</citation>}}


### (140/149) More than Vanilla Fusion: a Simple, Decoupling-free, Attention Module for Multimodal Fusion Based on Signal Theory (Peiwen Sun et al., 2023)

{{<citation>}}

Peiwen Sun, Yifan Zhang, Zishan Liu, Donghao Chen, Honggang Zhang. (2023)  
**More than Vanilla Fusion: a Simple, Decoupling-free, Attention Module for Multimodal Fusion Based on Signal Theory**  

---
Primary Category: cs.MM  
Categories: cs-AI, cs-MM, cs-SD, cs.MM, eess-AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.07212v1)  

---


**ABSTRACT**  
The vanilla fusion methods still dominate a large percentage of mainstream audio-visual tasks. However, the effectiveness of vanilla fusion from a theoretical perspective is still worth discussing. Thus, this paper reconsiders the signal fused in the multimodal case from a bionics perspective and proposes a simple, plug-and-play, attention module for vanilla fusion based on fundamental signal theory and uncertainty theory. In addition, previous work on multimodal dynamic gradient modulation still relies on decoupling the modalities. So, a decoupling-free gradient modulation scheme has been designed in conjunction with the aforementioned attention module, which has various advantages over the decoupled one. Experiment results show that just a few lines of code can achieve up to 2.0% performance improvements to several multimodal classification methods. Finally, quantitative evaluation of other fusion tasks reveals the potential for additional application scenarios.

{{</citation>}}


## cs.NI (1)



### (141/149) Your Vulnerability Disclosure Is Important To Us: An Analysis of Coordinated Vulnerability Disclosure Responses Using a Real Security Issue (Koen van Hove et al., 2023)

{{<citation>}}

Koen van Hove, Jeroen van der Ham-de Vos, Roland van Rijswijk-Deij. (2023)  
**Your Vulnerability Disclosure Is Important To Us: An Analysis of Coordinated Vulnerability Disclosure Responses Using a Real Security Issue**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.07284v1)  

---


**ABSTRACT**  
It is a public secret that doing email securely is fraught with challenges. We found a vulnerability present at many email providers, allowing us to spoof email on behalf of many organisations. As email vulnerabilities are ten a penny, instead of focusing on yet another email vulnerability we ask a different question: how do organisations react to the disclosure of such a security issue in the wild? We specifically focus on organisations from the public and critical infrastructure sector who are required to respond to such notifications by law. We find that many organisations are difficult to reach when it concerns security issues, even if they have a security contact point. Additionally, our findings show that having policy in place improves the response and resolution rate, but that even with a policy in place, half of our reports remain unanswered and unsolved after 90~days. Based on these findings we provide recommendations to organisations and bodies such as ENISA to improve future coordinated vulnerability disclosure processes.

{{</citation>}}


## cs.SE (2)



### (142/149) Learning from Interaction: User Interface Adaptation using Reinforcement Learning (Daniel Gaspar-Figueiredo, 2023)

{{<citation>}}

Daniel Gaspar-Figueiredo. (2023)  
**Learning from Interaction: User Interface Adaptation using Reinforcement Learning**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.07216v1)  

---


**ABSTRACT**  
The continuous adaptation of software systems to meet the evolving needs of users is very important for enhancing user experience (UX). User interface (UI) adaptation, which involves adjusting the layout, navigation, and content presentation based on user preferences and contextual conditions, plays an important role in achieving this goal. However, suggesting the right adaptation at the right time and in the right place remains a challenge in order to make it valuable for the end-user. To tackle this challenge, machine learning approaches could be used. In particular, we are using Reinforcement Learning (RL) due to its ability to learn from interaction with the users. In this approach, the feedback is very important and the use of physiological data could be benefitial to obtain objective insights into how users are reacting to the different adaptations. Thus, in this PhD thesis, we propose an RL-based UI adaptation framework that uses physiological data. The framework aims to learn from user interactions and make informed adaptations to improve UX. To this end, our research aims to answer the following questions: Does the use of an RL-based approach improve UX? How effective is RL in guiding UI adaptation? and Can physiological data support UI adaptation for enhancing UX? The evaluation plan involves conducting user studies to evaluate answer these questions. The empirical evaluation will provide a strong empirical foundation for building, evaluating, and improving the proposed adaptation framework. The expected contributions of this research include the development of a novel framework for intelligent Adaptive UIs, insights into the effectiveness of RL algorithms in guiding UI adaptation, the integration of physiological data as objective measures of UX, and empirical validation of the proposed framework's impact on UX.

{{</citation>}}


### (143/149) Code Membership Inference for Detecting Unauthorized Data Use in Code Pre-trained Language Models (Sheng Zhang et al., 2023)

{{<citation>}}

Sheng Zhang, Hui Li. (2023)  
**Code Membership Inference for Detecting Unauthorized Data Use in Code Pre-trained Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07200v1)  

---


**ABSTRACT**  
Code pre-trained language models (CPLMs) have received great attention since they can benefit various tasks that facilitate software development and maintenance. However, CPLMs are trained on massive open-source code, raising concerns about potential data infringement. This paper launches the first study of detecting unauthorized code use in CPLMs, i.e., Code Membership Inference (CMI) task. We design a framework Buzzer for different settings of CMI. Buzzer deploys several inference techniques, including distilling the target CPLM, ensemble inference, and unimodal and bimodal calibration. Extensive experiments show that CMI can be achieved with high accuracy using Buzzer. Hence, Buzzer can serve as a CMI tool and help protect intellectual property rights.

{{</citation>}}


## cs.IR (1)



### (144/149) Audience Prospecting for Dynamic-Product-Ads in Native Advertising (Eliran Abutbul et al., 2023)

{{<citation>}}

Eliran Abutbul, Yohay Kaplan, Naama Krasne, Oren Somekh, Or David, Omer Duvdevany, Evgeny Segal. (2023)  
**Audience Prospecting for Dynamic-Product-Ads in Native Advertising**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2312.07160v2)  

---


**ABSTRACT**  
With yearly revenue exceeding one billion USD, Yahoo Gemini native advertising marketplace serves more than two billion impressions daily to hundreds of millions of unique users. One of the fastest growing segments of Gemini native is dynamic-product-ads (DPA), where major advertisers, such as Amazon and Walmart, provide catalogs with millions of products for the system to choose from and present to users. The subject of this work is finding and expanding the right audience for each DPA ad, which is one of the many challenges DPA presents. Approaches such as targeting various user groups, e.g., users who already visited the advertisers' websites (Retargeting), users that searched for certain products (Search-Prospecting), or users that reside in preferred locations (Location-Prospecting), have limited audience expansion capabilities. In this work we present two new approaches for audience expansion that also maintain predefined performance goals. The Conversion-Prospecting approach predicts DPA conversion rates based on Gemini native logged data, and calculates the expected cost-per-action (CPA) for determining users' eligibility to products and optimizing DPA bids in Gemini native auctions. To support new advertisers and products, the Trending-Prospecting approach matches trending products to users by learning their tendency towards products from advertisers' sites logged events. The tendency scores indicate the popularity of the product and the similarity of the user to those who have previously engaged with this product. The two new prospecting approaches were tested online, serving real Gemini native traffic, demonstrating impressive DPA delivery and DPA revenue lifts while maintaining most traffic within the acceptable CPA range (i.e., performance goal). After a successful testing phase, the proposed approaches are currently in production and serve all Gemini native traffic.

{{</citation>}}


## physics.med-ph (1)



### (145/149) AI-driven projection tomography with multicore fibre-optic cell rotation (Jiawei Sun et al., 2023)

{{<citation>}}

Jiawei Sun, Bin Yang, Nektarios Koukourakis, Jochen Guck, Juergen W. Czarske. (2023)  
**AI-driven projection tomography with multicore fibre-optic cell rotation**  

---
Primary Category: physics.med-ph  
Categories: cs-AI, eess-IV, physics-bio-ph, physics-med-ph, physics-optics, physics.med-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07631v1)  

---


**ABSTRACT**  
Optical tomography has emerged as a non-invasive imaging method, providing three-dimensional insights into subcellular structures and thereby enabling a deeper understanding of cellular functions, interactions, and processes. Conventional optical tomography methods are constrained by a limited illumination scanning range, leading to anisotropic resolution and incomplete imaging of cellular structures. To overcome this problem, we employ a compact multi-core fibre-optic cell rotator system that facilitates precise optical manipulation of cells within a microfluidic chip, achieving full-angle projection tomography with isotropic resolution. Moreover, we demonstrate an AI-driven tomographic reconstruction workflow, which can be a paradigm shift from conventional computational methods, often demanding manual processing, to a fully autonomous process. The performance of the proposed cell rotation tomography approach is validated through the three-dimensional reconstruction of cell phantoms and HL60 human cancer cells. The versatility of this learning-based tomographic reconstruction workflow paves the way for its broad application across diverse tomographic imaging modalities, including but not limited to flow cytometry tomography and acoustic rotation tomography. Therefore, this AI-driven approach can propel advancements in cell biology, aiding in the inception of pioneering therapeutics, and augmenting early-stage cancer diagnostics.

{{</citation>}}


## cs.DC (1)



### (146/149) Layered Randomized Quantization for Communication-Efficient and Privacy-Preserving Distributed Learning (Guangfeng Yan et al., 2023)

{{<citation>}}

Guangfeng Yan, Tan Li, Tian Lan, Kui Wu, Linqi Song. (2023)  
**Layered Randomized Quantization for Communication-Efficient and Privacy-Preserving Distributed Learning**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2312.07060v1)  

---


**ABSTRACT**  
Next-generation wireless networks, such as edge intelligence and wireless distributed learning, face two critical challenges: communication efficiency and privacy protection. In this work, our focus is on addressing these issues in a distributed learning framework. We consider a new approach that simultaneously achieves communication efficiency and privacy protection by exploiting the privacy advantage offered by quantization. Specifically, we use a quantization scheme called \textbf{Gau}ssian \textbf{L}ayered \textbf{R}andomized \textbf{Q}uantization (Gau-LRQ) that compresses the raw model gradients using a layer multishift coupler. By adjusting the parameters of Gau-LRQ, we shape the quantization error to follow the expected Gaussian distribution, thus ensuring client-level differential privacy (CLDP). We demonstrate the effectiveness of our proposed Gau-LRQ in the distributed stochastic gradient descent (SGD) framework and theoretically quantify the trade-offs between communication, privacy, and convergence performance. We further improve the convergence performance by enabling dynamic private budget and quantization bit allocation. We achieve this by using an optimization formula that minimizes convergence error subject to the privacy budget constraint. We evaluate our approach on multiple datasets, including MNIST, CIFAR-10, and CIFAR-100, and show that our proposed method outperforms the baselines in terms of learning performance under various privacy constraints. Moreover, we observe that dynamic privacy allocation yields additional accuracy improvements for the models compared to the fixed scheme.

{{</citation>}}


## cs.SD (1)



### (147/149) LSTM-CNN Network for Audio Signature Analysis in Noisy Environments (Praveen Damacharla et al., 2023)

{{<citation>}}

Praveen Damacharla, Hamid Rajabalipanah, Mohammad Hosein Fakheri. (2023)  
**LSTM-CNN Network for Audio Signature Analysis in Noisy Environments**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-HC, cs-SD, cs.SD, eess-AS  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.07059v1)  

---


**ABSTRACT**  
There are multiple applications to automatically count people and specify their gender at work, exhibitions, malls, sales, and industrial usage. Although current speech detection methods are supposed to operate well, in most situations, in addition to genders, the number of current speakers is unknown and the classification methods are not suitable due to many possible classes. In this study, we focus on a long-short-term memory convolutional neural network (LSTM-CNN) to extract time and / or frequency-dependent features of the sound data to estimate the number / gender of simultaneous active speakers at each frame in noisy environments. Considering the maximum number of speakers as 10, we have utilized 19000 audio samples with diverse combinations of males, females, and background noise in public cities, industrial situations, malls, exhibitions, workplaces, and nature for learning purposes. This proof of concept shows promising performance with training/validation MSE values of about 0.019/0.017 in detecting count and gender.

{{</citation>}}


## eess.SY (1)



### (148/149) Large Foundation Models for Power Systems (Chenghao Huang et al., 2023)

{{<citation>}}

Chenghao Huang, Siyang Li, Ruohong Liu, Hao Wang, Yize Chen. (2023)  
**Large Foundation Models for Power Systems**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2312.07044v1)  

---


**ABSTRACT**  
Foundation models, such as Large Language Models (LLMs), can respond to a wide range of format-free queries without any task-specific data collection or model training, creating various research and application opportunities for the modeling and operation of large-scale power systems. In this paper, we outline how such large foundation model such as GPT-4 are developed, and discuss how they can be leveraged in challenging power and energy system tasks. We first investigate the potential of existing foundation models by validating their performance on four representative tasks across power system domains, including the optimal power flow (OPF), electric vehicle (EV) scheduling, knowledge retrieval for power engineering technical reports, and situation awareness. Our results indicate strong capabilities of such foundation models on boosting the efficiency and reliability of power system operational pipelines. We also provide suggestions and projections on future deployment of foundation models in power system applications.

{{</citation>}}


## cs.NE (1)



### (149/149) Astrocyte-Enabled Advancements in Spiking Neural Networks for Large Language Modeling (Guobin Shen et al., 2023)

{{<citation>}}

Guobin Shen, Dongcheng Zhao, Yiting Dong, Yang Li, Jindong Li, Yi Zeng. (2023)  
**Astrocyte-Enabled Advancements in Spiking Neural Networks for Large Language Modeling**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-NE, cs.NE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.07625v1)  

---


**ABSTRACT**  
Within the complex neuroarchitecture of the brain, astrocytes play crucial roles in development, structure, and metabolism. These cells regulate neural activity through tripartite synapses, directly impacting cognitive processes such as learning and memory. Despite the growing recognition of astrocytes' significance, traditional Spiking Neural Network (SNN) models remain predominantly neuron-centric, overlooking the profound influence of astrocytes on neural dynamics. Inspired by these biological insights, we have developed an Astrocyte-Modulated Spiking Unit (AM-SU), an innovative framework that integrates neuron-astrocyte interactions into the computational paradigm, demonstrating wide applicability across various hardware platforms. Our Astrocyte-Modulated Spiking Neural Network (AM-SNet) exhibits exceptional performance in tasks involving memory retention and natural language generation, particularly in handling long-term dependencies and complex linguistic structures. The design of AM-SNet not only enhances its biological authenticity but also introduces novel computational dynamics, enabling more effective processing of complex temporal dependencies. Furthermore, AM-SNet shows low latency, high throughput, and reduced memory usage in practical applications, making it highly suitable for resource-constrained environments. By successfully integrating astrocytic dynamics into intelligent neural networks, our work narrows the gap between biological plausibility and neural modeling, laying the groundwork for future biologically-inspired neural computing research that includes both neurons and astrocytes.

{{</citation>}}
